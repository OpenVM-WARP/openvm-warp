use core::borrow::Borrow;

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{CHUNK, D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, PrimeCharacteristicRing};
use p3_matrix::Matrix;

use super::{MULTI_CONSTRAINT_BATCHING_DOMAIN_TAG, MULTI_CONSTRAINT_PROOF_DOMAIN_TAG};
use crate::{
    bus::{
        CertifiedTranscriptCheckpointBus, CertifiedTranscriptCheckpointMessage, CommitmentsBus,
        CommitmentsBusMessage, StackingIndexMessage, StackingIndicesBus, TranscriptBus,
        WhirModuleBus, WhirModuleMessage, WhirMuBus, WhirMuMessage,
    },
    define_typed_per_proof_lookup_bus, define_typed_per_proof_permutation_bus,
    primitives::bus::{ExpBitsLenBus, ExpBitsLenMessage},
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{
        ext_field_add, ext_field_multiply, interpolate_quadratic, mobius_eq_1, pow_tidx_count,
    },
    whir::bus::{
        FinalPolyMleEvalBus, FinalPolyMleEvalMessage, WhirAlphaBus, WhirAlphaMessage,
        WhirCompletionBus, WhirCompletionMessage, WhirFinalPolyBus, WhirFinalPolyBusMessage,
        WhirSumcheckBus, WhirSumcheckBusMessage, WhirTerminalBus, WhirTerminalMessage,
    },
};

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct MultiConstraintPointMessage<T> {
    pub constraint_idx: T,
    pub coordinate_idx: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_lookup_bus!(MultiConstraintPointBus, MultiConstraintPointMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct MultiConstraintOpeningMessage<T> {
    pub constraint_idx: T,
    pub opening_idx: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_lookup_bus!(MultiConstraintOpeningBus, MultiConstraintOpeningMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct MultiConstraintCallerPrefixMessage<T> {
    pub tidx: T,
}

define_typed_per_proof_permutation_bus!(
    MultiConstraintCallerPrefixBus,
    MultiConstraintCallerPrefixMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct MultiConstraintRhoMessage<T> {
    pub constraint_idx: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_lookup_bus!(MultiConstraintRhoBus, MultiConstraintRhoMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct MultiConstraintMuMessage<T> {
    pub value: [T; D_EF],
}

define_typed_per_proof_lookup_bus!(MultiConstraintMuBus, MultiConstraintMuMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct MultiConstraintInitialTargetMessage<T> {
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(
    MultiConstraintInitialTargetBus,
    MultiConstraintInitialTargetMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct MultiConstraintFinalWeightMessage<T> {
    pub constraint_idx: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(
    MultiConstraintFinalWeightBus,
    MultiConstraintFinalWeightMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct MultiConstraintFinalContributionMessage<T> {
    pub constraint_idx: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(
    MultiConstraintFinalContributionBus,
    MultiConstraintFinalContributionMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct MultiConstraintFinalFoldingMessage<T> {
    pub constraint_idx: T,
    pub depth: T,
    pub node_idx: T,
    pub num_nodes_in_layer: T,
    pub value: [T; D_EF],
}

define_typed_per_proof_permutation_bus!(
    MultiConstraintFinalFoldingBus,
    MultiConstraintFinalFoldingMessage
);

/// One retained setup commitment consumed by the direct multi-WHIR owner.
///
/// The authority statement AIR is the sender.  The receiver below republishes
/// the root on the ordinary commitments bus and every `(commit, column)` pair
/// on the ordinary stacking-indices bus, so the reused Merkle/query/opened-row
/// AIRs cannot verify a different setup commitment or width.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct MultiConstraintInitialCommitmentMessage<T> {
    pub commit_idx: T,
    pub width: T,
    pub commitment: [T; CHUNK],
}

define_typed_per_proof_permutation_bus!(
    MultiConstraintInitialCommitmentBus,
    MultiConstraintInitialCommitmentMessage
);

/// Buses exposed to the caller-owned authority statement AIR.
///
/// The caller must bind points/openings and the prefix transcript index before
/// sending these messages.  This prevents a prover from choosing a different
/// multi-opening statement only inside the WHIR verifier.
#[derive(Clone, Copy, Debug)]
pub struct MultiConstraintStatementBuses {
    pub caller_prefix: MultiConstraintCallerPrefixBus,
    pub point: MultiConstraintPointBus,
    pub opening: MultiConstraintOpeningBus,
}

impl MultiConstraintStatementBuses {
    pub fn new(caller_prefix_idx: BusIndex, point_idx: BusIndex, opening_idx: BusIndex) -> Self {
        Self {
            caller_prefix: MultiConstraintCallerPrefixBus::new(caller_prefix_idx),
            point: MultiConstraintPointBus::new(point_idx),
            opening: MultiConstraintOpeningBus::new(opening_idx),
        }
    }
}

/// Internal buses for the new weight path.  They are deliberately distinct
/// from `WhirEqAlphaUBus`; the latter remains the legacy single-point product.
#[derive(Clone, Copy, Debug)]
pub struct MultiConstraintWeightBuses {
    pub rho: MultiConstraintRhoBus,
    pub mu: MultiConstraintMuBus,
    pub initial_target: MultiConstraintInitialTargetBus,
    pub final_weight: MultiConstraintFinalWeightBus,
    pub final_contribution: MultiConstraintFinalContributionBus,
    pub final_folding: MultiConstraintFinalFoldingBus,
}

impl MultiConstraintWeightBuses {
    pub fn new(indices: [BusIndex; 6]) -> Self {
        Self {
            rho: MultiConstraintRhoBus::new(indices[0]),
            mu: MultiConstraintMuBus::new(indices[1]),
            initial_target: MultiConstraintInitialTargetBus::new(indices[2]),
            final_weight: MultiConstraintFinalWeightBus::new(indices[3]),
            final_contribution: MultiConstraintFinalContributionBus::new(indices[4]),
            final_folding: MultiConstraintFinalFoldingBus::new(indices[5]),
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct MultiConstraintInitialCommitmentCols<T> {
    pub is_enabled: T,
    pub proof_idx: T,
    pub commit_idx: T,
    pub col_idx: T,
    pub is_first_in_proof: T,
    pub is_first_in_commit: T,
    pub width: T,
    pub commitment: [T; CHUNK],
}

/// Bridge from the caller-owned setup statement to the ordinary WHIR owners.
///
/// There is one valid row per committed column.  Consequently the width in
/// the authority message is not metadata trusted by the host: it is forced to
/// equal the number of exported stacking indices for that commitment.
#[derive(ColumnsAir)]
#[columns_via(MultiConstraintInitialCommitmentCols<u8>)]
pub struct MultiConstraintInitialCommitmentAir {
    pub initial_commitment_bus: MultiConstraintInitialCommitmentBus,
    pub commitments_bus: CommitmentsBus,
    pub stacking_indices_bus: StackingIndicesBus,
    pub commitment_count: usize,
    pub commitment_lookup_mult: usize,
    pub stacking_index_lookup_mult: usize,
}

impl BaseAirWithPublicValues<F> for MultiConstraintInitialCommitmentAir {}
impl PartitionedBaseAir<F> for MultiConstraintInitialCommitmentAir {}

impl BaseAir<F> for MultiConstraintInitialCommitmentAir {
    fn width(&self) -> usize {
        MultiConstraintInitialCommitmentCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for MultiConstraintInitialCommitmentAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("trace row");
        let next_row = main.row_slice(1).expect("next trace row");
        let local: &MultiConstraintInitialCommitmentCols<AB::Var> = (*local_row).borrow();
        let next: &MultiConstraintInitialCommitmentCols<AB::Var> = (*next_row).borrow();

        builder.assert_bool(local.is_enabled);
        builder.assert_bool(local.is_first_in_proof);
        builder.assert_bool(local.is_first_in_commit);

        NestedForLoopSubAir::<3>.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.is_enabled,
                    counter: [local.proof_idx, local.commit_idx, local.col_idx],
                    is_first: [
                        local.is_first_in_proof,
                        local.is_first_in_commit,
                        local.is_enabled,
                    ],
                }
                .map_into(),
                NestedForLoopIoCols {
                    is_enabled: next.is_enabled,
                    counter: [next.proof_idx, next.commit_idx, next.col_idx],
                    is_first: [
                        next.is_first_in_proof,
                        next.is_first_in_commit,
                        next.is_enabled,
                    ],
                }
                .map_into(),
            ),
        );

        let is_same_proof = next.is_enabled - next.is_first_in_proof;
        let is_same_commit = next.is_enabled - next.is_first_in_commit;
        let is_last_in_commit = local.is_enabled - is_same_commit.clone();
        let is_last_in_proof = local.is_enabled - is_same_proof;

        builder
            .when(local.is_first_in_proof)
            .assert_zero(local.commit_idx);
        builder
            .when(local.is_first_in_commit)
            .assert_zero(local.col_idx);
        builder
            .when(is_last_in_commit)
            .assert_eq(local.col_idx + AB::Expr::ONE, local.width);
        builder.when(is_last_in_proof).assert_eq(
            local.commit_idx,
            AB::Expr::from_usize(self.commitment_count - 1),
        );

        builder
            .when(is_same_commit.clone())
            .assert_eq(next.width, local.width);
        assert_array_eq(
            &mut builder.when(is_same_commit),
            next.commitment,
            local.commitment.map(Into::into),
        );

        self.initial_commitment_bus.receive(
            builder,
            local.proof_idx,
            MultiConstraintInitialCommitmentMessage {
                commit_idx: local.commit_idx.into(),
                width: local.width.into(),
                commitment: local.commitment.map(Into::into),
            },
            local.is_first_in_commit,
        );
        self.commitments_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            CommitmentsBusMessage {
                major_idx: AB::Expr::ZERO,
                minor_idx: local.commit_idx.into(),
                commitment: local.commitment.map(Into::into),
            },
            local.is_first_in_commit * AB::Expr::from_usize(self.commitment_lookup_mult),
        );
        self.stacking_indices_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            StackingIndexMessage {
                commit_idx: local.commit_idx.into(),
                col_idx: local.col_idx.into(),
            },
            local.is_enabled * AB::Expr::from_usize(self.stacking_index_lookup_mult),
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct MultiConstraintPrefixCols<T> {
    pub is_enabled: T,
    pub proof_idx: T,
    pub constraint_idx: T,
    pub is_first_in_proof: T,
    pub prefix_tidx: T,
    pub batching_gamma: [T; D_EF],
    pub rho: [T; D_EF],
    pub mu_pow_witness: T,
    pub mu_pow_sample: T,
    pub mu: [T; D_EF],
    pub initial_target: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(MultiConstraintPrefixCols<u8>)]
pub struct MultiConstraintPrefixAir {
    pub statement_buses: MultiConstraintStatementBuses,
    pub weight_buses: MultiConstraintWeightBuses,
    pub transcript_bus: TranscriptBus,
    pub whir_module_bus: WhirModuleBus,
    pub whir_mu_bus: WhirMuBus,
    pub exp_bits_len_bus: ExpBitsLenBus,
    pub constraint_count: usize,
    pub mu_pow_bits: usize,
    pub generator: F,
}

impl BaseAirWithPublicValues<F> for MultiConstraintPrefixAir {}
impl PartitionedBaseAir<F> for MultiConstraintPrefixAir {}

impl<F> BaseAir<F> for MultiConstraintPrefixAir {
    fn width(&self) -> usize {
        MultiConstraintPrefixCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for MultiConstraintPrefixAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("trace row");
        let next_row = main.row_slice(1).expect("next trace row");
        let local: &MultiConstraintPrefixCols<AB::Var> = (*local_row).borrow();
        let next: &MultiConstraintPrefixCols<AB::Var> = (*next_row).borrow();

        builder.assert_bool(local.is_enabled);
        builder.assert_bool(local.is_first_in_proof);
        let is_same_proof = next.is_enabled - next.is_first_in_proof;

        NestedForLoopSubAir::<2>.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.is_enabled,
                    counter: [local.proof_idx, local.constraint_idx],
                    is_first: [local.is_first_in_proof, local.is_enabled],
                }
                .map_into(),
                NestedForLoopIoCols {
                    is_enabled: next.is_enabled,
                    counter: [next.proof_idx, next.constraint_idx],
                    is_first: [next.is_first_in_proof, next.is_enabled],
                }
                .map_into(),
            ),
        );

        let is_last = local.is_enabled - is_same_proof.clone();
        builder.when(is_last.clone()).assert_eq(
            local.constraint_idx,
            AB::Expr::from_usize(self.constraint_count - 1),
        );
        builder
            .when(local.is_first_in_proof)
            .assert_zero(local.constraint_idx);

        assert_array_eq(
            &mut builder.when(local.is_first_in_proof),
            local.rho,
            [AB::F::ONE, AB::F::ZERO, AB::F::ZERO, AB::F::ZERO],
        );
        assert_array_eq(
            &mut builder.when(is_same_proof.clone()),
            next.rho,
            ext_field_multiply(local.rho, local.batching_gamma),
        );
        assert_array_eq(
            &mut builder.when(is_same_proof.clone()),
            next.batching_gamma,
            local.batching_gamma,
        );
        assert_array_eq(&mut builder.when(is_same_proof.clone()), next.mu, local.mu);
        assert_array_eq(
            &mut builder.when(is_same_proof.clone()),
            next.initial_target,
            local.initial_target,
        );
        builder
            .when(is_same_proof.clone())
            .assert_eq(next.prefix_tidx, local.prefix_tidx);
        builder
            .when(is_same_proof.clone())
            .assert_eq(next.mu_pow_witness, local.mu_pow_witness);
        builder
            .when(is_same_proof)
            .assert_eq(next.mu_pow_sample, local.mu_pow_sample);

        self.statement_buses.caller_prefix.receive(
            builder,
            local.proof_idx,
            MultiConstraintCallerPrefixMessage {
                tidx: local.prefix_tidx.into(),
            },
            local.is_first_in_proof,
        );

        let count = AB::Expr::from_usize(self.constraint_count);
        let version =
            AB::Expr::from_u32(openvm_stark_backend::whir::MULTI_CONSTRAINT_WHIR_PROTOCOL_VERSION);
        let batch_tag = AB::Expr::from_u64(MULTI_CONSTRAINT_BATCHING_DOMAIN_TAG);
        let proof_tag = AB::Expr::from_u64(MULTI_CONSTRAINT_PROOF_DOMAIN_TAG);
        let gamma_tidx = local.prefix_tidx + AB::Expr::from_usize(3);
        let proof_prefix_tidx = gamma_tidx.clone() + AB::Expr::from_usize(D_EF);
        let mu_pow_tidx = proof_prefix_tidx.clone() + AB::Expr::from_usize(3);
        let mu_tidx = mu_pow_tidx.clone() + AB::Expr::from_usize(pow_tidx_count(self.mu_pow_bits));
        let post_mu_tidx = mu_tidx.clone() + AB::Expr::from_usize(D_EF);

        self.transcript_bus.observe(
            builder,
            local.proof_idx,
            local.prefix_tidx,
            batch_tag,
            local.is_first_in_proof,
        );
        self.transcript_bus.observe(
            builder,
            local.proof_idx,
            local.prefix_tidx + AB::Expr::ONE,
            version.clone(),
            local.is_first_in_proof,
        );
        self.transcript_bus.observe(
            builder,
            local.proof_idx,
            local.prefix_tidx + AB::Expr::TWO,
            count.clone(),
            local.is_first_in_proof,
        );
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            gamma_tidx,
            local.batching_gamma,
            local.is_first_in_proof,
        );
        self.transcript_bus.observe(
            builder,
            local.proof_idx,
            proof_prefix_tidx.clone(),
            proof_tag,
            local.is_first_in_proof,
        );
        self.transcript_bus.observe(
            builder,
            local.proof_idx,
            proof_prefix_tidx.clone() + AB::Expr::ONE,
            version,
            local.is_first_in_proof,
        );
        self.transcript_bus.observe(
            builder,
            local.proof_idx,
            proof_prefix_tidx + AB::Expr::TWO,
            count,
            local.is_first_in_proof,
        );
        if self.mu_pow_bits > 0 {
            self.transcript_bus.observe(
                builder,
                local.proof_idx,
                mu_pow_tidx.clone(),
                local.mu_pow_witness,
                local.is_first_in_proof,
            );
            self.transcript_bus.sample(
                builder,
                local.proof_idx,
                mu_pow_tidx + AB::Expr::ONE,
                local.mu_pow_sample,
                local.is_first_in_proof,
            );
            self.exp_bits_len_bus.lookup_key(
                builder,
                ExpBitsLenMessage {
                    base: self.generator.into(),
                    bit_src: local.mu_pow_sample.into(),
                    num_bits: AB::Expr::from_usize(self.mu_pow_bits),
                    result: AB::Expr::ONE,
                },
                local.is_first_in_proof,
            );
        }
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            mu_tidx,
            local.mu,
            local.is_first_in_proof,
        );

        self.weight_buses.rho.add_key_with_lookups(
            builder,
            local.proof_idx,
            MultiConstraintRhoMessage {
                constraint_idx: local.constraint_idx.into(),
                value: local.rho.map(Into::into),
            },
            local.is_enabled * AB::Expr::TWO,
        );
        self.weight_buses.mu.add_key_with_lookups(
            builder,
            local.proof_idx,
            MultiConstraintMuMessage {
                value: local.mu.map(Into::into),
            },
            local.is_first_in_proof,
        );
        self.weight_buses.initial_target.receive(
            builder,
            local.proof_idx,
            MultiConstraintInitialTargetMessage {
                value: local.initial_target.map(Into::into),
            },
            local.is_first_in_proof,
        );
        self.whir_module_bus.send(
            builder,
            local.proof_idx,
            WhirModuleMessage {
                tidx: post_mu_tidx,
                claim: local.initial_target.map(Into::into),
            },
            local.is_first_in_proof,
        );
        self.whir_mu_bus.send(
            builder,
            local.proof_idx,
            WhirMuMessage {
                mu: local.mu.map(Into::into),
            },
            local.is_first_in_proof,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct MultiConstraintTargetCols<T> {
    pub is_enabled: T,
    pub proof_idx: T,
    pub constraint_idx: T,
    pub opening_idx: T,
    pub is_first_in_proof: T,
    pub is_first_in_constraint: T,
    pub rho: [T; D_EF],
    pub mu: [T; D_EF],
    pub mu_power: [T; D_EF],
    pub opening: [T; D_EF],
    pub accumulator: [T; D_EF],
    pub next_accumulator: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(MultiConstraintTargetCols<u8>)]
pub struct MultiConstraintInitialTargetAir {
    pub statement_buses: MultiConstraintStatementBuses,
    pub weight_buses: MultiConstraintWeightBuses,
    pub constraint_count: usize,
    pub total_width: usize,
}

impl BaseAirWithPublicValues<F> for MultiConstraintInitialTargetAir {}
impl PartitionedBaseAir<F> for MultiConstraintInitialTargetAir {}

impl<F> BaseAir<F> for MultiConstraintInitialTargetAir {
    fn width(&self) -> usize {
        MultiConstraintTargetCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for MultiConstraintInitialTargetAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("trace row");
        let next_row = main.row_slice(1).expect("next trace row");
        let local: &MultiConstraintTargetCols<AB::Var> = (*local_row).borrow();
        let next: &MultiConstraintTargetCols<AB::Var> = (*next_row).borrow();
        builder.assert_bool(local.is_enabled);
        builder.assert_bool(local.is_first_in_proof);
        builder.assert_bool(local.is_first_in_constraint);
        let is_same_proof = next.is_enabled - next.is_first_in_proof;
        let is_same_constraint = next.is_enabled - next.is_first_in_constraint;

        NestedForLoopSubAir::<3>.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.is_enabled,
                    counter: [local.proof_idx, local.constraint_idx, local.opening_idx],
                    is_first: [
                        local.is_first_in_proof,
                        local.is_first_in_constraint,
                        local.is_enabled,
                    ],
                }
                .map_into(),
                NestedForLoopIoCols {
                    is_enabled: next.is_enabled,
                    counter: [next.proof_idx, next.constraint_idx, next.opening_idx],
                    is_first: [
                        next.is_first_in_proof,
                        next.is_first_in_constraint,
                        next.is_enabled,
                    ],
                }
                .map_into(),
            ),
        );

        builder
            .when(local.is_first_in_constraint)
            .assert_zero(local.opening_idx);
        builder
            .when(local.is_enabled - is_same_constraint.clone())
            .assert_eq(
                local.opening_idx,
                AB::Expr::from_usize(self.total_width - 1),
            );
        builder
            .when(local.is_enabled - is_same_proof.clone())
            .assert_eq(
                local.constraint_idx,
                AB::Expr::from_usize(self.constraint_count - 1),
            );

        assert_array_eq(
            &mut builder.when(local.is_first_in_proof),
            local.accumulator,
            [AB::F::ZERO; D_EF],
        );
        assert_array_eq(
            &mut builder.when(local.is_first_in_constraint),
            local.mu_power,
            [AB::F::ONE, AB::F::ZERO, AB::F::ZERO, AB::F::ZERO],
        );
        assert_array_eq(
            &mut builder.when(is_same_constraint.clone()),
            next.mu_power,
            ext_field_multiply(local.mu_power, local.mu),
        );
        assert_array_eq(&mut builder.when(is_same_proof.clone()), next.mu, local.mu);
        assert_array_eq(
            &mut builder.when(is_same_constraint.clone()),
            next.rho,
            local.rho,
        );

        let contribution =
            ext_field_multiply(local.rho, ext_field_multiply(local.mu_power, local.opening));
        assert_array_eq(
            builder,
            local.next_accumulator,
            ext_field_add(local.accumulator, contribution),
        );
        assert_array_eq(
            &mut builder.when(is_same_proof.clone()),
            next.accumulator,
            local.next_accumulator,
        );

        self.weight_buses.mu.lookup_key(
            builder,
            local.proof_idx,
            MultiConstraintMuMessage {
                value: local.mu.map(Into::into),
            },
            local.is_first_in_proof,
        );
        self.weight_buses.rho.lookup_key(
            builder,
            local.proof_idx,
            MultiConstraintRhoMessage {
                constraint_idx: local.constraint_idx.into(),
                value: local.rho.map(Into::into),
            },
            local.is_first_in_constraint,
        );
        self.statement_buses.opening.lookup_key(
            builder,
            local.proof_idx,
            MultiConstraintOpeningMessage {
                constraint_idx: local.constraint_idx.into(),
                opening_idx: local.opening_idx.into(),
                value: local.opening.map(Into::into),
            },
            local.is_enabled,
        );
        self.weight_buses.initial_target.send(
            builder,
            local.proof_idx,
            MultiConstraintInitialTargetMessage {
                value: local.next_accumulator.map(Into::into),
            },
            local.is_enabled - is_same_proof,
        );
    }
}

/// Sumcheck verifier with the legacy transcript schedule but without the
/// single-point `eq_partial` recurrence.  Weight rows consume every alpha via
/// `alpha_bus`, so the challenges are still tied to the same proof transcript.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct MultiConstraintSumcheckCols<T> {
    pub is_enabled: T,
    pub proof_idx: T,
    pub whir_round: T,
    pub subidx: T,
    pub is_first_in_proof: T,
    pub is_first_in_round: T,
    pub tidx: T,
    pub ev1: [T; D_EF],
    pub ev2: [T; D_EF],
    pub folding_pow_witness: T,
    pub folding_pow_sample: T,
    pub alpha: [T; D_EF],
    pub pre_claim: [T; D_EF],
    pub post_group_claim: [T; D_EF],
    pub alpha_lookup_count: T,
}

#[derive(ColumnsAir)]
#[columns_via(MultiConstraintSumcheckCols<u8>)]
pub struct MultiConstraintSumcheckAir {
    pub sumcheck_bus: WhirSumcheckBus,
    pub transcript_bus: TranscriptBus,
    pub exp_bits_len_bus: ExpBitsLenBus,
    pub alpha_bus: WhirAlphaBus,
    pub k: usize,
    pub folding_pow_bits: usize,
    pub generator: F,
}

impl BaseAirWithPublicValues<F> for MultiConstraintSumcheckAir {}
impl PartitionedBaseAir<F> for MultiConstraintSumcheckAir {}
impl<F> BaseAir<F> for MultiConstraintSumcheckAir {
    fn width(&self) -> usize {
        MultiConstraintSumcheckCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for MultiConstraintSumcheckAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("trace row");
        let next_row = main.row_slice(1).expect("next trace row");
        let local: &MultiConstraintSumcheckCols<AB::Var> = (*local_row).borrow();
        let next: &MultiConstraintSumcheckCols<AB::Var> = (*next_row).borrow();
        let is_same_round = next.is_enabled - next.is_first_in_round;

        NestedForLoopSubAir::<3>.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.is_enabled,
                    counter: [local.proof_idx, local.whir_round, local.subidx],
                    is_first: [
                        local.is_first_in_proof,
                        local.is_first_in_round,
                        local.is_enabled,
                    ],
                }
                .map_into(),
                NestedForLoopIoCols {
                    is_enabled: next.is_enabled,
                    counter: [next.proof_idx, next.whir_round, next.subidx],
                    is_first: [
                        next.is_first_in_proof,
                        next.is_first_in_round,
                        next.is_enabled,
                    ],
                }
                .map_into(),
            ),
        );
        builder
            .when(local.is_enabled - is_same_round.clone())
            .assert_eq(local.subidx, AB::Expr::from_usize(self.k - 1));

        let sumcheck_idx = local.whir_round * AB::Expr::from_usize(self.k) + local.subidx;
        self.sumcheck_bus.receive(
            builder,
            local.proof_idx,
            WhirSumcheckBusMessage {
                tidx: local.tidx.into(),
                sumcheck_idx: sumcheck_idx.clone(),
                pre_claim: local.pre_claim.map(Into::into),
                post_claim: local.post_group_claim.map(Into::into),
            },
            local.is_first_in_round,
        );
        let post_claim = interpolate_quadratic(local.pre_claim, local.ev1, local.ev2, local.alpha);
        assert_array_eq(
            &mut builder.when(local.is_enabled - is_same_round.clone()),
            post_claim.clone(),
            local.post_group_claim,
        );
        assert_array_eq(
            &mut builder.when(is_same_round.clone()),
            post_claim,
            next.pre_claim,
        );
        assert_array_eq(
            &mut builder.when(is_same_round.clone()),
            next.post_group_claim,
            local.post_group_claim,
        );
        let folding_pow_offset = pow_tidx_count(self.folding_pow_bits);
        builder.when(is_same_round).assert_eq(
            next.tidx,
            local.tidx + AB::Expr::from_usize(3 * D_EF + folding_pow_offset),
        );

        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.tidx,
            local.ev1,
            local.is_enabled,
        );
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.tidx + AB::Expr::from_usize(D_EF),
            local.ev2,
            local.is_enabled,
        );
        if self.folding_pow_bits > 0 {
            self.transcript_bus.observe(
                builder,
                local.proof_idx,
                local.tidx + AB::Expr::from_usize(2 * D_EF),
                local.folding_pow_witness,
                local.is_enabled,
            );
            self.transcript_bus.sample(
                builder,
                local.proof_idx,
                local.tidx + AB::Expr::from_usize(2 * D_EF + 1),
                local.folding_pow_sample,
                local.is_enabled,
            );
            self.exp_bits_len_bus.lookup_key(
                builder,
                ExpBitsLenMessage {
                    base: self.generator.into(),
                    bit_src: local.folding_pow_sample.into(),
                    num_bits: AB::Expr::from_usize(self.folding_pow_bits),
                    result: AB::Expr::ONE,
                },
                local.is_enabled,
            );
        }
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.tidx + AB::Expr::from_usize(2 * D_EF + folding_pow_offset),
            local.alpha,
            local.is_enabled,
        );
        self.alpha_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            WhirAlphaMessage {
                idx: sumcheck_idx,
                challenge: local.alpha.map(Into::into),
            },
            local.is_enabled * local.alpha_lookup_count,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct MultiConstraintWeightCols<T> {
    pub is_enabled: T,
    pub proof_idx: T,
    pub constraint_idx: T,
    pub round_idx: T,
    pub is_first_in_proof: T,
    pub is_first_in_constraint: T,
    pub rho: [T; D_EF],
    pub point: [T; D_EF],
    pub alpha: [T; D_EF],
    pub partial_before: [T; D_EF],
    pub partial_after: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(MultiConstraintWeightCols<u8>)]
pub struct MultiConstraintWeightAir {
    pub statement_buses: MultiConstraintStatementBuses,
    pub weight_buses: MultiConstraintWeightBuses,
    pub alpha_bus: WhirAlphaBus,
    pub constraint_count: usize,
    pub num_sumcheck_rounds: usize,
}

impl BaseAirWithPublicValues<F> for MultiConstraintWeightAir {}
impl PartitionedBaseAir<F> for MultiConstraintWeightAir {}
impl<F> BaseAir<F> for MultiConstraintWeightAir {
    fn width(&self) -> usize {
        MultiConstraintWeightCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for MultiConstraintWeightAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("trace row");
        let next_row = main.row_slice(1).expect("next trace row");
        let local: &MultiConstraintWeightCols<AB::Var> = (*local_row).borrow();
        let next: &MultiConstraintWeightCols<AB::Var> = (*next_row).borrow();
        builder.assert_bool(local.is_enabled);
        builder.assert_bool(local.is_first_in_proof);
        builder.assert_bool(local.is_first_in_constraint);
        let is_same_proof = next.is_enabled - next.is_first_in_proof;
        let is_same_constraint = next.is_enabled - next.is_first_in_constraint;

        NestedForLoopSubAir::<3>.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.is_enabled,
                    counter: [local.proof_idx, local.constraint_idx, local.round_idx],
                    is_first: [
                        local.is_first_in_proof,
                        local.is_first_in_constraint,
                        local.is_enabled,
                    ],
                }
                .map_into(),
                NestedForLoopIoCols {
                    is_enabled: next.is_enabled,
                    counter: [next.proof_idx, next.constraint_idx, next.round_idx],
                    is_first: [
                        next.is_first_in_proof,
                        next.is_first_in_constraint,
                        next.is_enabled,
                    ],
                }
                .map_into(),
            ),
        );
        builder
            .when(local.is_first_in_constraint)
            .assert_zero(local.round_idx);
        builder
            .when(local.is_enabled - is_same_constraint.clone())
            .assert_eq(
                local.round_idx,
                AB::Expr::from_usize(self.num_sumcheck_rounds - 1),
            );
        builder.when(local.is_enabled - is_same_proof).assert_eq(
            local.constraint_idx,
            AB::Expr::from_usize(self.constraint_count - 1),
        );

        assert_array_eq(
            &mut builder.when(local.is_first_in_constraint),
            local.partial_before,
            [AB::F::ONE, AB::F::ZERO, AB::F::ZERO, AB::F::ZERO],
        );
        assert_array_eq(
            builder,
            local.partial_after,
            ext_field_multiply(local.partial_before, mobius_eq_1(local.point, local.alpha)),
        );
        assert_array_eq(
            &mut builder.when(is_same_constraint.clone()),
            next.partial_before,
            local.partial_after,
        );
        assert_array_eq(
            &mut builder.when(is_same_constraint.clone()),
            next.rho,
            local.rho,
        );

        self.weight_buses.rho.lookup_key(
            builder,
            local.proof_idx,
            MultiConstraintRhoMessage {
                constraint_idx: local.constraint_idx.into(),
                value: local.rho.map(Into::into),
            },
            local.is_first_in_constraint,
        );
        self.statement_buses.point.lookup_key(
            builder,
            local.proof_idx,
            MultiConstraintPointMessage {
                constraint_idx: local.constraint_idx.into(),
                coordinate_idx: local.round_idx.into(),
                value: local.point.map(Into::into),
            },
            local.is_enabled,
        );
        self.alpha_bus.lookup_key(
            builder,
            local.proof_idx,
            WhirAlphaMessage {
                idx: local.round_idx.into(),
                challenge: local.alpha.map(Into::into),
            },
            local.is_enabled,
        );
        self.weight_buses.final_weight.send(
            builder,
            local.proof_idx,
            MultiConstraintFinalWeightMessage {
                constraint_idx: local.constraint_idx.into(),
                value: ext_field_multiply(local.rho, local.partial_after),
            },
            local.is_enabled - is_same_constraint,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct MultiConstraintFinalPolyCols<T> {
    pub is_enabled: T,
    pub proof_idx: T,
    pub constraint_idx: T,
    pub is_first_in_proof: T,
    pub is_first_in_constraint: T,
    pub is_root: T,
    pub layer: T,
    pub layer_inv: T,
    pub node_idx: T,
    pub node_idx_inv: T,
    pub num_nodes_in_layer: T,
    pub tidx_final_poly_start: T,
    pub point: [T; D_EF],
    pub left_value: [T; D_EF],
    pub right_value: [T; D_EF],
    pub value: [T; D_EF],
    pub final_weight: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(MultiConstraintFinalPolyCols<u8>)]
pub struct MultiConstraintFinalPolyMleAir {
    pub statement_buses: MultiConstraintStatementBuses,
    pub weight_buses: MultiConstraintWeightBuses,
    pub transcript_bus: TranscriptBus,
    pub final_poly_bus: WhirFinalPolyBus,
    pub num_vars: usize,
    pub num_sumcheck_rounds: usize,
    pub constraint_count: usize,
    pub total_whir_queries: usize,
}

impl BaseAirWithPublicValues<F> for MultiConstraintFinalPolyMleAir {}
impl PartitionedBaseAir<F> for MultiConstraintFinalPolyMleAir {}
impl<F> BaseAir<F> for MultiConstraintFinalPolyMleAir {
    fn width(&self) -> usize {
        MultiConstraintFinalPolyCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for MultiConstraintFinalPolyMleAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("trace row");
        let local: &MultiConstraintFinalPolyCols<AB::Var> = (*local_row).borrow();
        builder.assert_bool(local.is_enabled);
        builder.assert_bool(local.is_first_in_proof);
        builder.assert_bool(local.is_first_in_constraint);
        builder.assert_bool(local.is_root);

        let is_nonleaf = local.layer * local.layer_inv;
        builder.when(local.layer).assert_one(is_nonleaf.clone());
        let is_leaf = local.is_enabled - is_nonleaf.clone();
        let node_nonzero = local.node_idx * local.node_idx_inv;
        builder
            .when(local.node_idx)
            .assert_one(node_nonzero.clone());
        builder
            .when(local.is_root)
            .assert_eq(local.layer, AB::Expr::from_usize(self.num_vars));
        builder.when(local.is_root).assert_zero(local.node_idx);
        builder
            .when(local.is_root)
            .assert_eq(local.num_nodes_in_layer, AB::Expr::ONE);

        let coordinate_idx =
            AB::Expr::from_usize(self.num_sumcheck_rounds + self.num_vars) - local.layer;
        self.statement_buses.point.lookup_key(
            builder,
            local.proof_idx,
            MultiConstraintPointMessage {
                constraint_idx: local.constraint_idx.into(),
                coordinate_idx,
                value: local.point.map(Into::into),
            },
            is_nonleaf.clone(),
        );

        let left_idx = local.node_idx;
        let right_idx = local.node_idx + local.num_nodes_in_layer;
        let child_depth = local.layer - AB::Expr::ONE;
        self.weight_buses.final_folding.receive(
            builder,
            local.proof_idx,
            MultiConstraintFinalFoldingMessage {
                constraint_idx: local.constraint_idx.into(),
                depth: child_depth.clone(),
                node_idx: left_idx.into(),
                num_nodes_in_layer: local.num_nodes_in_layer * AB::Expr::TWO,
                value: local.left_value.map(Into::into),
            },
            is_nonleaf.clone(),
        );
        self.weight_buses.final_folding.receive(
            builder,
            local.proof_idx,
            MultiConstraintFinalFoldingMessage {
                constraint_idx: local.constraint_idx.into(),
                depth: child_depth,
                node_idx: right_idx,
                num_nodes_in_layer: local.num_nodes_in_layer * AB::Expr::TWO,
                value: local.right_value.map(Into::into),
            },
            is_nonleaf.clone(),
        );
        self.weight_buses.final_folding.send(
            builder,
            local.proof_idx,
            MultiConstraintFinalFoldingMessage {
                constraint_idx: local.constraint_idx.into(),
                depth: local.layer.into(),
                node_idx: local.node_idx.into(),
                num_nodes_in_layer: local.num_nodes_in_layer.into(),
                value: local.value.map(Into::into),
            },
            local.is_enabled - local.is_root,
        );
        assert_array_eq(
            &mut builder.when(is_nonleaf),
            local.value,
            ext_field_add(
                local.left_value,
                ext_field_multiply(
                    crate::utils::ext_field_subtract(local.right_value, local.left_value),
                    local.point,
                ),
            ),
        );

        // Constraint zero owns the transcript observation and final-poly table.
        // Later constraints only look up the same coefficients; query chips keep
        // their existing lookups on this table.
        let is_first_constraint_leaf = is_leaf.clone() * local.is_first_in_constraint;
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.tidx_final_poly_start + local.node_idx * AB::Expr::from_usize(D_EF),
            local.value,
            is_first_constraint_leaf.clone(),
        );
        self.final_poly_bus.add_key_with_lookups(
            builder,
            local.proof_idx,
            WhirFinalPolyBusMessage {
                idx: local.node_idx.into(),
                coeff: local.value.map(Into::into),
            },
            is_first_constraint_leaf
                * AB::Expr::from_usize(self.total_whir_queries + self.constraint_count - 1),
        );
        self.final_poly_bus.lookup_key(
            builder,
            local.proof_idx,
            WhirFinalPolyBusMessage {
                idx: local.node_idx.into(),
                coeff: local.value.map(Into::into),
            },
            is_leaf * (local.is_enabled - local.is_first_in_constraint),
        );

        self.weight_buses.final_weight.receive(
            builder,
            local.proof_idx,
            MultiConstraintFinalWeightMessage {
                constraint_idx: local.constraint_idx.into(),
                value: local.final_weight.map(Into::into),
            },
            local.is_root,
        );
        self.weight_buses.final_contribution.send(
            builder,
            local.proof_idx,
            MultiConstraintFinalContributionMessage {
                constraint_idx: local.constraint_idx.into(),
                value: ext_field_multiply(local.final_weight, local.value),
            },
            local.is_root,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct MultiConstraintFinalAggregateCols<T> {
    pub is_enabled: T,
    pub proof_idx: T,
    pub constraint_idx: T,
    pub is_first_in_proof: T,
    pub contribution: [T; D_EF],
    pub accumulator: [T; D_EF],
    pub next_accumulator: [T; D_EF],
    pub tidx_final_poly_start: T,
}

#[derive(ColumnsAir)]
#[columns_via(MultiConstraintFinalAggregateCols<u8>)]
pub struct MultiConstraintFinalAggregateAir {
    pub weight_buses: MultiConstraintWeightBuses,
    pub final_poly_mle_eval_bus: FinalPolyMleEvalBus,
    pub constraint_count: usize,
    pub num_whir_rounds: usize,
}

impl BaseAirWithPublicValues<F> for MultiConstraintFinalAggregateAir {}
impl PartitionedBaseAir<F> for MultiConstraintFinalAggregateAir {}
impl<F> BaseAir<F> for MultiConstraintFinalAggregateAir {
    fn width(&self) -> usize {
        MultiConstraintFinalAggregateCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for MultiConstraintFinalAggregateAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("trace row");
        let next_row = main.row_slice(1).expect("next trace row");
        let local: &MultiConstraintFinalAggregateCols<AB::Var> = (*local_row).borrow();
        let next: &MultiConstraintFinalAggregateCols<AB::Var> = (*next_row).borrow();
        builder.assert_bool(local.is_enabled);
        builder.assert_bool(local.is_first_in_proof);
        let is_same_proof = next.is_enabled - next.is_first_in_proof;
        NestedForLoopSubAir::<2>.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.is_enabled,
                    counter: [local.proof_idx, local.constraint_idx],
                    is_first: [local.is_first_in_proof, local.is_enabled],
                }
                .map_into(),
                NestedForLoopIoCols {
                    is_enabled: next.is_enabled,
                    counter: [next.proof_idx, next.constraint_idx],
                    is_first: [next.is_first_in_proof, next.is_enabled],
                }
                .map_into(),
            ),
        );
        builder
            .when(local.is_first_in_proof)
            .assert_zero(local.constraint_idx);
        builder
            .when(local.is_enabled - is_same_proof.clone())
            .assert_eq(
                local.constraint_idx,
                AB::Expr::from_usize(self.constraint_count - 1),
            );
        assert_array_eq(
            &mut builder.when(local.is_first_in_proof),
            local.accumulator,
            [AB::F::ZERO; D_EF],
        );
        assert_array_eq(
            builder,
            local.next_accumulator,
            ext_field_add(local.accumulator, local.contribution),
        );
        assert_array_eq(
            &mut builder.when(is_same_proof.clone()),
            next.accumulator,
            local.next_accumulator,
        );
        builder
            .when(is_same_proof.clone())
            .assert_eq(next.tidx_final_poly_start, local.tidx_final_poly_start);
        self.weight_buses.final_contribution.receive(
            builder,
            local.proof_idx,
            MultiConstraintFinalContributionMessage {
                constraint_idx: local.constraint_idx.into(),
                value: local.contribution.map(Into::into),
            },
            local.is_enabled,
        );
        self.final_poly_mle_eval_bus.receive(
            builder,
            local.proof_idx,
            FinalPolyMleEvalMessage {
                tidx: local.tidx_final_poly_start.into(),
                num_whir_rounds: AB::Expr::from_usize(self.num_whir_rounds),
                value: local.next_accumulator.map(Into::into),
            },
            local.is_enabled - is_same_proof,
        );
    }
}

/// Joins the last WHIR scalar check to a row-aligned transcript checkpoint.
/// This is the only sender on the externally exposed completion bus.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct MultiConstraintCompletionCols<T> {
    pub is_enabled: T,
    pub proof_idx: T,
    pub is_first_in_proof: T,
    pub class_index: T,
    pub end_tidx: T,
    pub sample_count: T,
    pub state: [T; POSEIDON2_WIDTH],
    pub final_aggregate: [T; D_EF],
    pub final_claim: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(MultiConstraintCompletionCols<u8>)]
pub struct MultiConstraintCompletionAir {
    pub terminal_bus: WhirTerminalBus,
    pub checkpoint_bus: CertifiedTranscriptCheckpointBus,
    pub completion_bus: WhirCompletionBus,
    pub class_index: usize,
    pub checkpoint_kind: usize,
}

impl BaseAirWithPublicValues<F> for MultiConstraintCompletionAir {}
impl PartitionedBaseAir<F> for MultiConstraintCompletionAir {}
impl<F> BaseAir<F> for MultiConstraintCompletionAir {
    fn width(&self) -> usize {
        MultiConstraintCompletionCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for MultiConstraintCompletionAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("trace row");
        let next_row = main.row_slice(1).expect("next trace row");
        let local: &MultiConstraintCompletionCols<AB::Var> = (*local_row).borrow();
        let next: &MultiConstraintCompletionCols<AB::Var> = (*next_row).borrow();
        builder.assert_bool(local.is_enabled);
        builder.assert_bool(local.is_first_in_proof);
        builder
            .when(local.is_enabled)
            .assert_eq(local.class_index, AB::Expr::from_usize(self.class_index));

        NestedForLoopSubAir::<1>.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.is_enabled,
                    counter: [local.proof_idx],
                    is_first: [local.is_first_in_proof],
                }
                .map_into(),
                NestedForLoopIoCols {
                    is_enabled: next.is_enabled,
                    counter: [next.proof_idx],
                    is_first: [next.is_first_in_proof],
                }
                .map_into(),
            ),
        );

        self.terminal_bus.receive(
            builder,
            local.proof_idx,
            WhirTerminalMessage {
                end_tidx: local.end_tidx.into(),
                final_claim: local.final_claim.map(Into::into),
                final_aggregate: local.final_aggregate.map(Into::into),
            },
            local.is_enabled,
        );
        self.checkpoint_bus.receive(
            builder,
            local.proof_idx,
            CertifiedTranscriptCheckpointMessage {
                kind: AB::Expr::from_usize(self.checkpoint_kind),
                tidx: local.end_tidx.into(),
                sample_count: local.sample_count.into(),
                state: local.state.map(Into::into),
            },
            local.is_enabled,
        );
        self.completion_bus.send(
            builder,
            local.proof_idx,
            WhirCompletionMessage {
                proof_idx: local.proof_idx.into(),
                class_index: local.class_index.into(),
                end_tidx: local.end_tidx.into(),
                sample_count: local.sample_count.into(),
                state: local.state.map(Into::into),
                final_aggregate: local.final_aggregate.map(Into::into),
                final_claim: local.final_claim.map(Into::into),
            },
            local.is_enabled,
        );
    }
}

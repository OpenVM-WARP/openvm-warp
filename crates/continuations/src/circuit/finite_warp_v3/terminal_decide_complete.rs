//! Direct terminal `Decide` owner for the genuine complete verifier PESAT.
//!
//! This module deliberately composes the semantic complete-relation owner
//! with the relation-independent WHIR/RS-adjoint tail.  It never constructs a
//! `FixedMultiAirPesatIndex`, converts a complete proof to the legacy terminal
//! proof, or discovers AIR slots by string name.

use core::borrow::Borrow;
use std::sync::Arc;

use openvm_recursion_circuit::{
    bus::{Poseidon2CompressBus, Poseidon2CompressMessage},
    native_warp::{
        generate_native_leaf_hash_trace, generate_native_merkle_multiproof_trace, terminal::*,
        NativeAccumulatorAlgebraicDigestBus, NativeAccumulatorAlgebraicDigestMessage,
        NativeAccumulatorDigestElementBus, NativeAccumulatorDigestElementMessage,
        NativeLeafHashAir, NativeLeafHashInput, NativeMerkleMultiproofAir,
    },
    primitives::exp_bits_len::ExpBitsLenAir,
    system::BusIndexManager,
    transcript::{transcript::TranscriptAir, Poseidon2BusOwner},
};
use openvm_stark_backend::{
    air_builders::inlined_cached::InlinedCachedAir,
    interaction::{BusIndex, InteractionBuilder},
    native_warp::FixedMultiAirCompletePesatIndex,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    warp_accum::{BinaryMerkleMultiproofRecord, WhirInitialRsLayout},
    AirRef, BaseAirWithPublicValues, PartitionedBaseAir, StarkProtocolConfig, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config, Digest, DIGEST_SIZE, D_EF, F,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::{
    finite_warp_v3_terminal_index_digest, finite_warp_v3_terminal_transcript_seam_trace,
    generate_finite_warp_v3_complete_two_coset_whir_prefix_trace,
    generate_finite_warp_v3_resumed_terminal_transcript_trace,
    generate_finite_warp_v3_rs_statement_trace,
    generate_finite_warp_v3_terminal_statement_prefix_trace,
    generate_finite_warp_v3_two_coset_terminal_whir_folding_trace,
    generate_finite_warp_v3_two_coset_terminal_whir_opened_trace,
    generate_finite_warp_v3_two_coset_terminal_whir_query_trace,
    generate_two_coset_adjoint_q_trace, generate_two_coset_adjoint_selector_trace,
    generate_two_coset_adjoint_y_trace, generate_two_coset_terminal_exp_bits_trace,
    FiniteWarpV3CompleteTerminalTraceWitness, FiniteWarpV3CompleteTerminalTranscriptSeamAir,
    FiniteWarpV3CompleteTwoCosetWhirPrefixAir, FiniteWarpV3RsStatementAir,
    FiniteWarpV3RsStatementEndBus, FiniteWarpV3RsStatementProfile,
    FiniteWarpV3TerminalAccumulatorLinkBus, FiniteWarpV3TerminalAccumulatorLinkMessage,
    FiniteWarpV3TerminalDecideBinding, FiniteWarpV3TerminalDecideBuses,
    FiniteWarpV3TerminalDecideError, FiniteWarpV3TerminalReceiptBus,
    FiniteWarpV3TerminalReceiptMessage, FiniteWarpV3TerminalReceiptRecord,
    FiniteWarpV3TerminalStatementPrefixAir, FiniteWarpV3TerminalStatementPrefixProfile,
    FiniteWarpV3TerminalStatementStartBus, FiniteWarpV3TerminalTranscriptCheckpoint,
    FiniteWarpV3TwoCosetRsAdjointQAir, FiniteWarpV3TwoCosetRsAdjointSelectorAir,
    FiniteWarpV3TwoCosetRsAdjointYAir, FiniteWarpV3TwoCosetTerminalProfile,
    FiniteWarpV3TwoCosetTerminalWhirFoldingAir, FiniteWarpV3TwoCosetTerminalWhirOpenedAir,
    FiniteWarpV3TwoCosetTerminalWhirQueryAir, FINITE_WARP_V3_MAX_CALLS, FINITE_WARP_V3_TWO_COSET_K,
};
use crate::circuit::native_warp_accumulator::{
    generate_native_accumulator_digest_traces, NativeAccumulatorHashAir,
    NativeAccumulatorRootDigestCols, NativeAccumulatorValueCols, NativePrivateAccumulatorLayout,
};

#[derive(Clone, Debug)]
pub struct FiniteWarpV3CompleteTerminalDecideProfile {
    pub complete: Arc<FixedMultiAirCompleteTerminalCircuitProfile>,
    pub statement_prefix: Arc<FiniteWarpV3TerminalStatementPrefixProfile>,
    pub rs_statement: Arc<FiniteWarpV3RsStatementProfile>,
    pub linearizer_components: Vec<FixedMultiAirLinearizerRawComponentPlan>,
    pub point_lookup_counts: Vec<usize>,
    pub weight_term_count: usize,
}

impl FiniteWarpV3CompleteTerminalDecideProfile {
    fn new(
        relation: Arc<FixedMultiAirCompletePesatIndex<F, Digest>>,
        two_coset: &FiniteWarpV3TwoCosetTerminalProfile,
        terminal_index_digest: Digest,
    ) -> Result<Self, FiniteWarpV3TerminalDecideError> {
        let description = relation.description();
        let relation_digest = description.relation_digest;
        let code = description.code_class;
        let alpha_len = usize::from(code.log_codeword_len);
        let log_message_len = usize::from(code.log_message_len);
        let beta_len = relation.pesat_shape().beta_len();
        if relation_digest == [F::ZERO; DIGEST_SIZE]
            || alpha_len != two_coset.alpha_len()
            || log_message_len != two_coset.log_message_len()
            || two_coset.whir_k() != FINITE_WARP_V3_TWO_COSET_K
            || alpha_len != log_message_len + 1
            || code.log_blowup != 1
            || code.initial_folding_factor != 0
            || usize::from(code.rows_per_query) != (1usize << FINITE_WARP_V3_TWO_COSET_K)
            || beta_len == 0
        {
            return Err(FiniteWarpV3TerminalDecideError::UnsupportedTwoCosetProfile);
        }

        // External consumers are exact and setup-owned:
        // - binding: terminal statement prefix, coefficient prefix and receipt;
        // - every instance coordinate: native bridge and digest binder;
        // - every instance coordinate is observed twice by the exact native statement/constrained
        //   prefix;
        // - mu additionally feeds the coefficient prefix.
        let downstream = FixedMultiAirCompleteExternalLookupCounts {
            binding: 3,
            alpha: vec![4; alpha_len],
            mu: 5,
            beta: vec![4; beta_len],
            eta: 4,
            local_claims: vec![0; relation.region_count()],
            interaction_claims: vec![0; relation.region_count()],
        };
        let complete = Arc::new(
            FixedMultiAirCompleteTerminalCircuitProfile::new(relation, downstream)
                .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalAirInventory)?,
        );
        let statement_prefix = Arc::new(
            FiniteWarpV3TerminalStatementPrefixProfile::new(
                two_coset,
                relation_digest,
                terminal_index_digest,
                beta_len,
            )
            .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalAirInventory)?,
        );
        let rs_statement = Arc::new(
            FiniteWarpV3RsStatementProfile::new(complete.plan.clone())
                .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalAirInventory)?,
        );
        let linearizer_components =
            FixedMultiAirLinearizerRawComponentPlan::from_complete_plan(&complete.plan)
                .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalAirInventory)?;
        if linearizer_components.is_empty() {
            return Err(FiniteWarpV3TerminalDecideError::TerminalAirInventory);
        }
        let point_lookup_counts = terminal_point_lookup_counts(
            two_coset.whir_k(),
            two_coset.num_queries_per_round(),
            two_coset.final_poly_len(),
            log_message_len,
        )
        .ok_or(FiniteWarpV3TerminalDecideError::UnsupportedTwoCosetProfile)?;
        let weight_term_count = two_coset
            .num_queries_per_round()
            .iter()
            .try_fold(two_coset.whir_round_count() - 1, |total, &queries| {
                total.checked_add(queries)
            })
            .ok_or(FiniteWarpV3TerminalDecideError::UnsupportedTwoCosetProfile)?;
        Ok(Self {
            complete,
            statement_prefix,
            rs_statement,
            linearizer_components,
            point_lookup_counts,
            weight_term_count,
        })
    }

    #[must_use]
    pub fn relation_digest(&self) -> Digest {
        self.complete.plan.metadata.relation_digest
    }

    #[must_use]
    pub fn alpha_len(&self) -> usize {
        usize::from(self.complete.plan.metadata.code_class.log_codeword_len)
    }

    #[must_use]
    pub fn log_message_len(&self) -> usize {
        usize::from(self.complete.plan.metadata.code_class.log_message_len)
    }

    #[must_use]
    pub fn beta_len(&self) -> usize {
        usize::from(self.complete.plan.metadata.log_constraints)
            + usize::try_from(self.complete.plan.metadata.explicit_len)
                .expect("validated complete terminal beta length")
    }
}

/// Compact generic-tail namespace.  The legacy relation-specific buses in
/// `FixedMultiAirTerminalBusInventory` remain unused; only the generic
/// structured-linearization, WHIR, Merkle and adjoint buses are connected.
/// Keeping this setup carrier temporarily avoids duplicating their canonical
/// allocation order while the legacy terminal producer stays fail-closed.
pub type FiniteWarpV3CompleteTerminalTailBuses = FixedMultiAirTerminalBusInventory;

pub struct FiniteWarpV3CompleteTerminalInstanceAir {
    pub layout: NativePrivateAccumulatorLayout,
    pub terminal_instance_bus: FixedMultiAirCompleteInstanceValueBus,
    pub digest_element_bus: NativeAccumulatorDigestElementBus,
}

impl BaseAir<F> for FiniteWarpV3CompleteTerminalInstanceAir {
    fn width(&self) -> usize {
        NativeAccumulatorValueCols::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for FiniteWarpV3CompleteTerminalInstanceAir {}
impl PartitionedBaseAir<F> for FiniteWarpV3CompleteTerminalInstanceAir {}

impl<AB> Air<AB> for FiniteWarpV3CompleteTerminalInstanceAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("complete terminal instance row");
        let next_row = main
            .row_slice(1)
            .expect("complete terminal instance next row");
        let local: &NativeAccumulatorValueCols<AB::Var> = (*local_row).borrow();
        let next: &NativeAccumulatorValueCols<AB::Var> = (*next_row).borrow();

        for flag in [
            local.active,
            local.is_first,
            local.is_last,
            local.section_last,
        ] {
            builder.assert_bool(flag);
        }
        for flag in local.section {
            builder.assert_bool(flag);
        }
        let [is_alpha, is_mu, is_beta, is_eta] = local.section.map(AB::Expr::from);
        let section_sum = is_alpha.clone() + is_mu.clone() + is_beta.clone() + is_eta.clone();
        builder.when(local.active).assert_one(section_sum.clone());
        builder
            .when(AB::Expr::ONE - local.active)
            .assert_zero(section_sum);
        builder.when(local.active).assert_zero(local.proof_idx);

        let section_len = is_alpha.clone() * AB::Expr::from_usize(self.layout.alpha_len)
            + is_mu.clone()
            + is_beta.clone() * AB::Expr::from_usize(self.layout.beta_len)
            + is_eta.clone();
        builder
            .when(local.active)
            .assert_eq(local.coordinate + local.remaining, section_len);
        let remaining_minus_one = local.remaining - AB::Expr::ONE;
        builder
            .when(local.active * local.section_last)
            .assert_zero(remaining_minus_one.clone());
        builder
            .when(local.active * (AB::Expr::ONE - local.section_last))
            .assert_one(remaining_minus_one * local.section_last_inverse);
        builder
            .when(local.active)
            .assert_eq(local.is_last, is_eta.clone() * local.section_last);

        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when_first_row().assert_one(local.section[0]);
        builder.when_first_row().assert_zero(local.coordinate);
        builder.when_first_row().assert_zero(local.ordinal);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder.when_transition().assert_eq(
            local.active - next.active,
            local.is_last * (AB::Expr::ONE - next.active),
        );
        let mut transition = builder.when_transition();
        let mut next_active = transition.when(next.active);
        next_active.assert_zero(next.is_first);
        next_active.assert_zero(next.proof_idx);
        next_active.assert_eq(next.ordinal, local.ordinal + AB::F::ONE);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);

        let continue_section = next.active * (AB::Expr::ONE - local.section_last);
        let mut transition = builder.when_transition();
        let mut same = transition.when(continue_section);
        for index in 0..4 {
            same.assert_eq(next.section[index], local.section[index]);
        }
        same.assert_eq(next.coordinate, local.coordinate + AB::F::ONE);
        same.assert_eq(next.remaining, local.remaining - AB::F::ONE);

        let advance_section = next.active * local.section_last;
        let mut transition = builder.when_transition();
        let mut advance = transition.when(advance_section);
        advance.assert_zero(next.coordinate);
        advance.assert_eq(next.section[0], AB::Expr::ZERO);
        advance.assert_eq(next.section[1], is_alpha.clone());
        advance.assert_eq(next.section[2], is_mu.clone());
        advance.assert_eq(next.section[3], is_beta.clone());

        let section = is_mu + is_beta * AB::Expr::from_u32(2) + is_eta * AB::Expr::from_u32(3);
        self.terminal_instance_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInstanceValueMessage {
                section,
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active,
        );
        for limb in 0..D_EF {
            self.digest_element_bus.send(
                builder,
                NativeAccumulatorDigestElementMessage {
                    proof_idx: AB::Expr::ZERO,
                    state: AB::Expr::ZERO,
                    index: AB::Expr::from_usize(3)
                        + local.ordinal * AB::Expr::from_usize(D_EF)
                        + AB::Expr::from_usize(limb),
                    value: local.value[limb].into(),
                },
                local.active,
            );
        }
    }
}

pub struct FiniteWarpV3CompleteTerminalReceiptAir {
    pub binding: FiniteWarpV3TerminalDecideBinding,
    pub relation_digest: Digest,
    pub alpha_len: usize,
    pub beta_len: usize,
    pub receipt_bus: FiniteWarpV3TerminalReceiptBus,
    pub terminal_binding_bus: FixedMultiAirCompleteBindingBus,
    pub terminal_whir_root_bus: NativeTerminalAccumulatorRootBus,
    pub algebraic_digest_bus: NativeAccumulatorAlgebraicDigestBus,
    pub compress_bus: Poseidon2CompressBus,
    pub terminal_accumulator_link_bus: FiniteWarpV3TerminalAccumulatorLinkBus,
}

impl BaseAir<F> for FiniteWarpV3CompleteTerminalReceiptAir {
    fn width(&self) -> usize {
        NativeAccumulatorRootDigestCols::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for FiniteWarpV3CompleteTerminalReceiptAir {}
impl PartitionedBaseAir<F> for FiniteWarpV3CompleteTerminalReceiptAir {}

impl<AB> Air<AB> for FiniteWarpV3CompleteTerminalReceiptAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("complete terminal receipt row");
        let local: &NativeAccumulatorRootDigestCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_transition().assert_zero(local.active);
        builder.when(local.active).assert_zero(local.proof_idx);
        let enabled = Into::<AB::Expr>::into(local.active);
        self.terminal_binding_bus.lookup_key(
            builder,
            FixedMultiAirCompleteBindingMessage {
                relation_digest: self.relation_digest.map(Into::into),
                root: local.root.map(Into::into),
                alpha_len: AB::Expr::from_usize(self.alpha_len),
                beta_len: AB::Expr::from_usize(self.beta_len),
            },
            enabled.clone(),
        );
        self.terminal_whir_root_bus.receive(
            builder,
            NativeTerminalAccumulatorRootMessage {
                root: local.root.map(Into::into),
            },
            enabled.clone(),
        );
        self.algebraic_digest_bus.receive(
            builder,
            NativeAccumulatorAlgebraicDigestMessage {
                proof_idx: AB::Expr::ZERO,
                state: AB::Expr::ZERO,
                digest: local.algebraic_digest.map(Into::into),
            },
            enabled.clone(),
        );
        self.compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: core::array::from_fn(|index| {
                    if index < DIGEST_SIZE {
                        local.root[index].into()
                    } else {
                        local.algebraic_digest[index - DIGEST_SIZE].into()
                    }
                }),
                output: local.instance_digest.map(Into::into),
            },
            enabled.clone(),
        );
        self.terminal_accumulator_link_bus.lookup_key(
            builder,
            FiniteWarpV3TerminalAccumulatorLinkMessage {
                root: local.root.map(Into::into),
                digest: local.instance_digest.map(Into::into),
            },
            enabled.clone(),
        );
        self.receipt_bus.add_key_with_lookups(
            builder,
            FiniteWarpV3TerminalReceiptMessage {
                protocol_digest: self.binding.protocol_digest.map(Into::into),
                relation_digest: self.relation_digest.map(Into::into),
                terminal_index_digest: self.binding.terminal_index_digest.map(Into::into),
                verifier_component_digest: self.binding.verifier_component_digest.map(Into::into),
                final_accumulator_digest: local.instance_digest.map(Into::into),
                final_accumulator_root: local.root.map(Into::into),
            },
            enabled,
        );
    }
}

pub struct FiniteWarpV3CompleteTerminalDecideProducer {
    pub profile: FiniteWarpV3CompleteTerminalDecideProfile,
    pub complete: Arc<FixedMultiAirCompleteTerminalCircuit>,
    pub tail: FiniteWarpV3CompleteTerminalTailBuses,
    two_coset: FiniteWarpV3TwoCosetTerminalProfile,
    binding: FiniteWarpV3TerminalDecideBinding,
    buses: FiniteWarpV3TerminalDecideBuses,
    poseidon: Poseidon2BusOwner,
    system_params: SystemParams,
    layout: NativePrivateAccumulatorLayout,
    statement_start_bus: FiniteWarpV3TerminalStatementStartBus,
    rs_statement_end_bus: FiniteWarpV3RsStatementEndBus,
}

impl FiniteWarpV3CompleteTerminalDecideProducer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        relation: Arc<FixedMultiAirCompletePesatIndex<F, Digest>>,
        two_coset: FiniteWarpV3TwoCosetTerminalProfile,
        binding: FiniteWarpV3TerminalDecideBinding,
        buses: FiniteWarpV3TerminalDecideBuses,
        poseidon: Poseidon2BusOwner,
        system_params: SystemParams,
        bus_indices: &mut BusIndexManager,
    ) -> Result<Self, FiniteWarpV3TerminalDecideError> {
        let profile = FiniteWarpV3CompleteTerminalDecideProfile::new(
            relation,
            &two_coset,
            binding.terminal_index_digest,
        )?;
        let relation_digest = profile.relation_digest();
        if [
            binding.protocol_digest,
            binding.terminal_index_digest,
            binding.verifier_component_digest,
            binding.warp_index_digest,
            binding.vacc_setup_digest,
            binding.schedule_digest,
        ]
        .contains(&[F::ZERO; DIGEST_SIZE])
        {
            return Err(FiniteWarpV3TerminalDecideError::ZeroProtocolDigest);
        }
        if binding.warp_call_count == 0 || binding.warp_call_count > FINITE_WARP_V3_MAX_CALLS {
            return Err(FiniteWarpV3TerminalDecideError::InvalidWarpCallCount);
        }
        if finite_warp_v3_terminal_index_digest(
            relation_digest,
            two_coset.descriptor_metadata_words(),
        ) != binding.terminal_index_digest
        {
            return Err(FiniteWarpV3TerminalDecideError::TerminalIndexDigest);
        }

        let statement_start_bus =
            FiniteWarpV3TerminalStatementStartBus::new(bus_indices.new_bus_idx());
        let rs_statement_end_bus = FiniteWarpV3RsStatementEndBus::new(bus_indices.new_bus_idx());
        let tail = FixedMultiAirTerminalBusInventory::new(bus_indices.next_bus_idx());
        reserve_through(bus_indices, tail.next_bus_idx())?;
        let complete = Arc::new(FixedMultiAirCompleteTerminalCircuit::new(
            profile.complete.clone(),
            tail.transcript,
            0,
            bus_indices.next_bus_idx(),
        ));
        reserve_through(bus_indices, complete.next_bus_idx())?;
        let layout =
            NativePrivateAccumulatorLayout::new(0, profile.alpha_len(), profile.beta_len());
        let producer = Self {
            profile,
            complete,
            tail,
            two_coset,
            binding,
            buses,
            poseidon,
            system_params,
            layout,
            statement_start_bus,
            rs_statement_end_bus,
        };
        // Construction validates every fallible AIR geometry before the
        // verifier key can commit to the inventory.
        producer.validate_air_geometry()?;
        Ok(producer)
    }

    fn validate_air_geometry(&self) -> Result<(), FiniteWarpV3TerminalDecideError> {
        self.complete
            .airs::<BabyBearPoseidon2Config>()
            .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalAirInventory)?;
        self.round_air()?;
        FiniteWarpV3TwoCosetTerminalWhirQueryAir::new(
            &self.two_coset,
            self.tail.transcript,
            self.tail.whir_verify_queries,
            self.tail.whir_query,
            self.tail.weight_term,
            self.tail.exp_bits_len,
            self.tail.right_shift,
            self.two_coset.whir_round_count(),
            self.two_coset.final_poly_len(),
            self.two_coset.whir_round_count(),
            0,
        )
        .map_err(|_| FiniteWarpV3TerminalDecideError::UnsupportedTwoCosetProfile)?;
        FiniteWarpV3TwoCosetTerminalWhirOpenedAir::new(
            &self.two_coset,
            self.tail.whir_query,
            self.tail.whir_folding,
            self.tail.leaf_value,
            self.tail.opening_leaf,
            self.tail.merkle_root,
        )
        .map_err(|_| FiniteWarpV3TerminalDecideError::UnsupportedTwoCosetProfile)?;
        FiniteWarpV3TwoCosetTerminalWhirFoldingAir::new(
            &self.two_coset,
            self.tail.whir_alpha,
            self.tail.whir_folding,
        )
        .map_err(|_| FiniteWarpV3TerminalDecideError::UnsupportedTwoCosetProfile)?;
        FiniteWarpV3TwoCosetRsAdjointYAir::new(
            self.tail.adjoint_round,
            self.tail.adjoint_y,
            self.profile.log_message_len(),
        )?;
        Ok(())
    }

    fn round_air(&self) -> Result<NativeTerminalWhirRoundAir, FiniteWarpV3TerminalDecideError> {
        NativeTerminalWhirRoundAir::new(
            self.tail.transcript,
            self.tail.whir_statement,
            self.tail.whir_round,
            self.tail.whir_verify_queries,
            self.tail.whir_final_claim,
            self.tail.weight_term,
            self.tail.merkle_root,
            self.tail.exp_bits_len,
            self.two_coset.whir_k(),
            self.two_coset.alpha_len(),
            self.two_coset.final_poly_len(),
            self.two_coset.query_phase_pow_bits(),
            self.two_coset.folding_pow_bits(),
            F::GENERATOR,
            0,
            self.two_coset.num_queries_per_round().to_vec(),
        )
        .map_err(|_| FiniteWarpV3TerminalDecideError::UnsupportedTwoCosetProfile)
    }

    #[must_use]
    pub const fn poseidon2_bus_owner(&self) -> Poseidon2BusOwner {
        self.poseidon
    }

    #[must_use]
    pub fn system_params(&self) -> &SystemParams {
        &self.system_params
    }

    /// Semantic AIR order.  There is no slot replacement or string lookup.
    pub fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        let b = &self.tail;
        let p = &self.profile;
        let mut airs = vec![Arc::new(TranscriptAir {
            transcript_bus: b.transcript,
            poseidon2_permute_bus: self.poseidon.permute_bus,
            final_state_bus: None,
            resume_state_bus: Some(self.buses.terminal_resume),
            end_index_bus: None,
            checkpoint_state_bus: None,
        }) as AirRef<SC>];
        add(
            &mut airs,
            FiniteWarpV3CompleteTerminalTranscriptSeamAir {
                final_vacc_checkpoint_bus: self.buses.final_vacc_checkpoint,
                terminal_resume_bus: self.buses.terminal_resume,
                terminal_accumulator_link_bus: self.buses.terminal_accumulator_link,
                statement_start_bus: self.statement_start_bus,
                binding: self.binding.clone(),
                relation_digest: p.relation_digest(),
            },
        );
        add_inlined(
            &mut airs,
            FiniteWarpV3TerminalStatementPrefixAir {
                profile: p.statement_prefix.clone(),
                transcript_bus: b.transcript,
                start_bus: self.statement_start_bus,
                complete_start_bus: self.complete.buses.statement_start_cursor,
                binding_bus: self.complete.buses.binding,
                instance_bus: self.complete.buses.instance,
            },
        );
        airs.extend(
            self.complete
                .airs::<SC>()
                .expect("validated complete terminal AIR inventory"),
        );
        add_inlined(
            &mut airs,
            FiniteWarpV3RsStatementAir {
                profile: p.rs_statement.clone(),
                transcript_bus: b.transcript,
                source_cursor_bus: self.complete.buses.whir_prefix_cursor,
                end_cursor_bus: self.rs_statement_end_bus,
                source_header_bus: self.complete.buses.structured_header,
                source_term_bus: self.complete.buses.mapped_term,
                source_point_bus: self.complete.buses.structured_point,
                output_header_bus: b.structured_header,
                output_term_bus: b.mapped_term,
                output_point_bus: b.structured_point,
            },
        );
        add(
            &mut airs,
            FiniteWarpV3CompleteTwoCosetWhirPrefixAir {
                transcript_bus: b.transcript,
                cursor_bus: self.rs_statement_end_bus,
                binding_bus: self.complete.buses.binding,
                instance_bus: self.complete.buses.instance,
                start_bus: b.whir_start,
                root_bus: b.accumulator_root,
                relation_digest: p.relation_digest(),
                profile: self.two_coset.clone(),
                beta_len: p.beta_len(),
            },
        );
        add_inlined(
            &mut airs,
            FixedMultiAirStructuredTargetAir {
                start_bus: b.whir_start,
                header_bus: b.structured_header,
                batched_claim_bus: b.batched_claim,
                statement_bus: b.whir_statement,
                claim_count: 1,
            },
        );
        add_inlined(
            &mut airs,
            FixedMultiAirCompleteAccumulatorBridgeAir {
                fixed_bus: self.complete.buses.instance,
                native_bus: b.accumulator_value,
            },
        );
        add_inlined(
            &mut airs,
            FixedMultiAirLinearizerSumcheckAir {
                transcript_bus: b.transcript,
                point_bus: b.linearizer_aux_point,
                final_bus: b.linearizer_sumcheck_final,
                log_message_len: p.log_message_len(),
            },
        );
        add_inlined(
            &mut airs,
            FixedMultiAirLinearizerYAir {
                point_bus: b.linearizer_aux_point,
                whir_point_bus: b.whir_point,
                y_bus: b.linearizer_y,
                log_message_len: p.log_message_len(),
                initial_folding_factor: p.log_message_len(),
            },
        );
        add(
            &mut airs,
            FixedMultiAirLinearizerSelectorAir {
                y_bus: b.linearizer_y,
                selector_bus: b.linearizer_selector,
                log_message_len: p.log_message_len(),
            },
        );
        add_inlined(
            &mut airs,
            FixedMultiAirLinearizerRawAir {
                batched_claim_bus: b.batched_claim,
                mapped_term_bus: b.mapped_term,
                structured_point_bus: b.structured_point,
                aux_point_bus: b.linearizer_aux_point,
                raw_term_bus: b.linearizer_raw_term,
                log_message_len: p.log_message_len(),
                component_count: p.linearizer_components.len(),
            },
        );
        add(
            &mut airs,
            FixedMultiAirLinearizerRawSumAir {
                term_bus: b.linearizer_raw_term,
                output_bus: b.linearizer_raw_weight,
                component_count: p.linearizer_components.len(),
            },
        );
        add(
            &mut airs,
            FixedMultiAirLinearizerFinalAir {
                statement_bus: b.whir_statement,
                sumcheck_bus: b.linearizer_sumcheck_final,
                raw_weight_bus: b.linearizer_raw_weight,
                selector_bus: b.linearizer_selector,
                output_bus: b.linearizer_weight,
            },
        );
        add(
            &mut airs,
            NativeTerminalWhirSumcheckAir {
                transcript_bus: b.transcript,
                round_bus: b.whir_round,
                alpha_bus: b.whir_alpha,
                exp_bits_len_bus: b.exp_bits_len,
                k: self.two_coset.whir_k(),
                round_count: self.two_coset.whir_round_count(),
                folding_pow_bits: self.two_coset.folding_pow_bits(),
                generator: F::GENERATOR,
            },
        );
        add(
            &mut airs,
            self.round_air().expect("validated WHIR round AIR"),
        );
        add(
            &mut airs,
            FiniteWarpV3TwoCosetTerminalWhirQueryAir::new(
                &self.two_coset,
                b.transcript,
                b.whir_verify_queries,
                b.whir_query,
                b.weight_term,
                b.exp_bits_len,
                b.right_shift,
                self.two_coset.whir_round_count(),
                self.two_coset.final_poly_len(),
                self.two_coset.whir_round_count(),
                0,
            )
            .expect("validated two-coset query AIR"),
        );
        add(
            &mut airs,
            FiniteWarpV3TwoCosetTerminalWhirOpenedAir::new(
                &self.two_coset,
                b.whir_query,
                b.whir_folding,
                b.leaf_value,
                b.opening_leaf,
                b.merkle_root,
            )
            .expect("validated two-coset opened AIR"),
        );
        add(
            &mut airs,
            FiniteWarpV3TwoCosetTerminalWhirFoldingAir::new(
                &self.two_coset,
                b.whir_alpha,
                b.whir_folding,
            )
            .expect("validated two-coset folding AIR"),
        );
        add(
            &mut airs,
            NativeLeafHashAir {
                permute_bus: self.poseidon.permute_bus,
                value_bus: b.leaf_value,
                leaf_bus: b.opening_leaf,
            },
        );
        add(
            &mut airs,
            NativeMerkleMultiproofAir {
                compress_bus: self.poseidon.compress_bus,
                leaf_bus: b.opening_leaf,
                node_bus: b.merkle_node,
                root_bus: b.merkle_root,
            },
        );
        add(
            &mut airs,
            NativeTerminalWhirFinalTableAir {
                transcript_bus: b.transcript,
                final_claim_bus: b.whir_final_claim,
                final_context_bus: b.whir_final_context,
                final_poly_bus: b.whir_final_poly,
                final_weight_bus: b.whir_final_weight,
                final_len: self.two_coset.final_poly_len(),
            },
        );
        add(
            &mut airs,
            NativeTerminalWhirMobiusAir::new(
                b.whir_final_poly,
                self.two_coset.whir_remaining_dimension(),
            ),
        );
        add(
            &mut airs,
            NativeTerminalWhirPointAir::new(
                b.transcript,
                b.whir_alpha,
                b.whir_final_context,
                b.whir_point,
                self.two_coset.whir_k(),
                self.two_coset.whir_round_count(),
                self.two_coset.final_poly_len(),
                p.point_lookup_counts.clone(),
            ),
        );
        add(
            &mut airs,
            NativeTerminalWhirFinalCheckAir {
                final_context_bus: b.whir_final_context,
                final_poly_bus: b.whir_final_poly,
                final_weight_bus: b.whir_final_weight,
                point_bus: b.whir_point,
                actual_weight_bus: b.whir_actual_weight,
                point_prefix_len: self.two_coset.whir_sumcheck_round_count(),
                final_log_len: self.two_coset.whir_remaining_dimension(),
            },
        );
        add(
            &mut airs,
            NativeTerminalWhirWeightTermAir {
                term_bus: b.weight_term,
                result_bus: b.weight_term_result,
                point_bus: b.whir_point,
            },
        );
        add(
            &mut airs,
            NativeTerminalRsAdjointSumcheckAir::new(
                b.transcript,
                b.adjoint_round,
                b.adjoint_claim,
                p.alpha_len(),
                p.log_message_len() + 1,
            ),
        );
        add(
            &mut airs,
            FiniteWarpV3TwoCosetRsAdjointQAir {
                round_bus: b.adjoint_round,
                accumulator_value_bus: b.accumulator_value,
                q_bus: b.adjoint_q,
                log_message_len: p.log_message_len(),
            },
        );
        add(
            &mut airs,
            FiniteWarpV3TwoCosetRsAdjointYAir::new(
                b.adjoint_round,
                b.adjoint_y,
                p.log_message_len(),
            )
            .expect("validated two-coset adjoint Y AIR"),
        );
        add(
            &mut airs,
            FiniteWarpV3TwoCosetRsAdjointSelectorAir {
                point_bus: b.whir_point,
                q_bus: b.adjoint_q,
                y_bus: b.adjoint_y,
                claim_bus: b.adjoint_claim,
                value_bus: b.adjoint_value,
                log_message_len: p.log_message_len(),
            },
        );
        add(
            &mut airs,
            NativeTerminalWhirExpectedWeightAir {
                actual_bus: b.whir_actual_weight,
                adjoint_bus: b.adjoint_value,
                linearizer_bus: b.linearizer_weight,
                term_bus: b.weight_term_result,
                term_count: p.weight_term_count,
            },
        );
        add(&mut airs, ExpBitsLenAir::new(b.exp_bits_len, b.right_shift));
        add(
            &mut airs,
            FiniteWarpV3CompleteTerminalInstanceAir {
                layout: self.layout.clone(),
                terminal_instance_bus: self.complete.buses.instance,
                digest_element_bus: self.buses.digest_element,
            },
        );
        add(
            &mut airs,
            NativeAccumulatorHashAir {
                state: 0,
                allow_empty: false,
                alpha_len: p.alpha_len(),
                beta_len: p.beta_len(),
                digest_element_bus: self.buses.digest_element,
                algebraic_digest_bus: self.buses.algebraic_digest,
                permute_bus: self.poseidon.permute_bus,
                compress_bus: self.poseidon.compress_bus,
            },
        );
        add(
            &mut airs,
            FiniteWarpV3CompleteTerminalReceiptAir {
                binding: self.binding.clone(),
                relation_digest: p.relation_digest(),
                alpha_len: p.alpha_len(),
                beta_len: p.beta_len(),
                receipt_bus: self.buses.receipt,
                terminal_binding_bus: self.complete.buses.binding,
                terminal_whir_root_bus: b.accumulator_root,
                algebraic_digest_bus: self.buses.algebraic_digest,
                compress_bus: self.poseidon.compress_bus,
                terminal_accumulator_link_bus: self.buses.terminal_accumulator_link,
            },
        );
        airs
    }

    /// Generate the complete semantic AIR packet directly from the native
    /// terminal verification witness. No legacy terminal trace is generated
    /// and no matrix is installed by positional replacement.
    pub fn generate_traces(
        &self,
        witness: FiniteWarpV3CompleteTerminalTraceWitness<'_>,
        final_warp_checkpoint: FiniteWarpV3TerminalTranscriptCheckpoint,
    ) -> Result<FiniteWarpV3CompleteTerminalDecideTraceData, FiniteWarpV3TerminalDecideError> {
        let p = &self.profile;
        let instance = witness.instance;
        let verification = witness.whir_verification;
        if witness.relation.description().relation_digest != p.relation_digest()
            || instance.alpha.len() != p.alpha_len()
            || instance.beta.len() != p.beta_len()
            || witness.descriptor.root != instance.rt
            || verification.root != instance.rt
            || witness.statement.linearizer_claims.len() != 1
            || witness.statement.linearizer_claims[0].len() != (1usize << p.log_message_len())
        {
            return Err(FiniteWarpV3TerminalDecideError::TerminalTrace);
        }

        let (resumed_transcript, mut poseidon2_permute_inputs) =
            generate_finite_warp_v3_resumed_terminal_transcript_trace(
                witness.transcript,
                final_warp_checkpoint,
                None,
            )
            .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let digest = generate_native_accumulator_digest_traces(0, instance, &self.layout)
            .ok_or(FiniteWarpV3TerminalDecideError::InvalidAccumulatorShape)?;
        let seam = finite_warp_v3_terminal_transcript_seam_trace(
            final_warp_checkpoint,
            instance.rt,
            digest.instance_digest,
        );
        let statement_prefix = generate_finite_warp_v3_terminal_statement_prefix_trace(
            &p.statement_prefix,
            instance,
            digest.instance_digest,
            witness.transcript,
            final_warp_checkpoint,
            None,
        )
        .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;

        let nonlinear = self
            .complete
            .generate_traces(FixedMultiAirCompleteTerminalTraceWitness {
                authenticated_root: instance.rt,
                instance,
                proof: &witness.reduction_proof.linearizer,
                transcript: witness.transcript,
                statement_start_tidx: statement_prefix.end_tidx,
            })
            .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        if nonlinear.structured_target != witness.statement.linearizer_claims[0].target {
            return Err(FiniteWarpV3TerminalDecideError::TerminalTrace);
        }

        let rs_statement = generate_finite_warp_v3_rs_statement_trace(
            &p.rs_statement,
            witness.statement,
            witness.transcript,
            nonlinear.whir_prefix_start_tidx,
            None,
        )
        .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        if rs_statement.end_tidx != verification.transcript_start.operations {
            return Err(FiniteWarpV3TerminalDecideError::TerminalTrace);
        }

        let prefix = generate_finite_warp_v3_complete_two_coset_whir_prefix_trace(
            &self.two_coset,
            witness.descriptor,
            instance,
            witness.transcript,
            rs_statement.end_tidx,
        )
        .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let statement_tidx = rs_statement
            .end_tidx
            .checked_add(self.two_coset.observation_len())
            .and_then(|value| value.checked_add(D_EF))
            .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let target = generate_fixed_multi_air_structured_target_traces(
            witness.descriptor,
            statement_tidx,
            instance.mu,
            verification.batching_challenge,
            &[FixedMultiAirStructuredClaimRecord {
                kind: 0,
                log_message_len: p.log_message_len(),
                term_count: p.complete.mapping.opening_count,
                point_len: 0,
                target: nonlinear.structured_target,
                endpoint_lookup_count: p.linearizer_components.len(),
            }],
            None,
        )
        .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        if target.initial_claim != verification.initial_claim {
            return Err(FiniteWarpV3TerminalDecideError::TerminalTrace);
        }

        let native_counts = [
            vec![1; p.alpha_len()],
            vec![0],
            vec![0; p.beta_len()],
            vec![0],
        ];
        let (bridge_cached, bridge_common) =
            generate_fixed_multi_air_complete_accumulator_bridge_traces(
                instance,
                &native_counts,
                None,
            )
            .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;

        let whir_point = complete_whir_point(verification, self.two_coset.whir_k())
            .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let custom_start = verification.transcript_end.operations;
        let provisional_point = linearizer_point_from_transcript(
            witness.linearizer_adjoint_proof,
            witness.transcript,
            custom_start,
            p.log_message_len(),
        )
        .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let raw = generate_fixed_multi_air_linearizer_raw_traces(
            p.log_message_len(),
            &p.linearizer_components,
            &witness.statement.linearizer_claims,
            &[verification.batching_challenge],
            &provisional_point,
            None,
        )
        .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        // CoefficientTwoCosetGrs uses the entire Boolean point in the
        // structured selector; every coordinate has exactly one Y consumer.
        let point_counts = raw
            .aux_point_lookup_counts
            .iter()
            .map(|&count| count.checked_add(1))
            .collect::<Option<Vec<_>>>()
            .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let linearizer_sumcheck_air = FixedMultiAirLinearizerSumcheckAir {
            transcript_bus: self.tail.transcript,
            point_bus: self.tail.linearizer_aux_point,
            final_bus: self.tail.linearizer_sumcheck_final,
            log_message_len: p.log_message_len(),
        };
        let linearizer_sumcheck = generate_fixed_multi_air_linearizer_sumcheck_traces(
            &linearizer_sumcheck_air,
            witness.linearizer_adjoint_proof,
            witness.transcript,
            custom_start,
            &point_counts,
            None,
        )
        .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        if linearizer_sumcheck.point != provisional_point {
            return Err(FiniteWarpV3TerminalDecideError::TerminalTrace);
        }
        let (linearizer_y_cached, linearizer_y_common, selector_factors) =
            generate_fixed_multi_air_linearizer_y_traces_for_layout(
                &linearizer_sumcheck.point,
                &whir_point,
                0,
                WhirInitialRsLayout::CoefficientTwoCosetGrs,
                None,
            )
            .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let (linearizer_selector, selector) =
            generate_fixed_multi_air_linearizer_selector_trace(&selector_factors, None)
                .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let linearizer_raw_sum =
            generate_fixed_multi_air_linearizer_raw_sum_trace(&raw.term_values, None)
                .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let linearizer_expected =
            terminal_linearizer_expected_weight(verification, self.two_coset.whir_k())
                .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        if linearizer_sumcheck.initial_claim != linearizer_expected {
            return Err(FiniteWarpV3TerminalDecideError::TerminalTrace);
        }
        let linearizer_final = generate_fixed_multi_air_linearizer_final_trace(
            statement_tidx,
            instance.rt,
            verification.batching_challenge,
            verification.initial_claim,
            linearizer_sumcheck.initial_claim,
            linearizer_sumcheck.final_claim,
            raw.value,
            selector,
        )
        .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;

        let whir_sumcheck = generate_native_terminal_whir_sumcheck_trace(
            verification,
            witness.transcript,
            self.two_coset.folding_pow_bits(),
            1,
            None,
        )
        .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let round_air = self.round_air()?;
        let whir_round = generate_native_terminal_whir_round_trace(
            &round_air,
            verification,
            witness.transcript,
            None,
        )
        .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let whir_query = generate_finite_warp_v3_two_coset_terminal_whir_query_trace(
            &self.two_coset,
            verification,
            witness.transcript,
            self.two_coset.whir_round_count(),
            0,
            self.two_coset.whir_round_count(),
            self.two_coset.final_poly_len(),
            None,
        )
        .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let config = BabyBearPoseidon2Config::default_from_params(self.system_params.clone());
        let whir_opened = generate_finite_warp_v3_two_coset_terminal_whir_opened_trace(
            &self.two_coset,
            config.hasher(),
            verification,
            witness.transcript,
            self.two_coset.whir_round_count(),
            0,
            None,
        )
        .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let whir_folding = generate_finite_warp_v3_two_coset_terminal_whir_folding_trace(
            &self.two_coset,
            verification,
            None,
        )
        .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;

        let leaf_inputs = whir_opened
            .leaves
            .iter()
            .map(|leaf| NativeLeafHashInput {
                proof_idx: 0,
                tree_id: leaf.tree_id,
                leaf_index: leaf.leaf_index,
                values: &leaf.values,
                lookup_counts: &leaf.lookup_counts,
            })
            .collect::<Vec<_>>();
        let leaf_hash = generate_native_leaf_hash_trace(&leaf_inputs, None)
            .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        poseidon2_permute_inputs.extend(leaf_hash.permutation_inputs.iter().copied());
        let mut merkle_records = whir_opened
            .inner_merkle
            .iter()
            .map(|(tree, record)| (*tree, record))
            .collect::<Vec<(u32, &BinaryMerkleMultiproofRecord<Digest>)>>();
        merkle_records.extend(
            verification
                .rounds
                .iter()
                .enumerate()
                .map(|(round, verification)| (round as u32, &verification.multiproof)),
        );
        let merkle = generate_native_merkle_multiproof_trace(0, &merkle_records, None)
            .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let mut poseidon2_compress_inputs = merkle_records
            .iter()
            .flat_map(|(_, record)| {
                record.compressions.iter().map(|compression| {
                    core::array::from_fn(|index| {
                        if index < DIGEST_SIZE {
                            compression.left[index]
                        } else {
                            compression.right[index - DIGEST_SIZE]
                        }
                    })
                })
            })
            .collect::<Vec<_>>();

        let final_table = generate_native_terminal_whir_final_table_trace(verification, None)
            .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let mobius_air = NativeTerminalWhirMobiusAir::new(
            self.tail.whir_final_poly,
            self.two_coset.whir_remaining_dimension(),
        );
        let mobius = generate_native_terminal_whir_mobius_trace(&mobius_air, verification, None)
            .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let point_air = NativeTerminalWhirPointAir::new(
            self.tail.transcript,
            self.tail.whir_alpha,
            self.tail.whir_final_context,
            self.tail.whir_point,
            self.two_coset.whir_k(),
            self.two_coset.whir_round_count(),
            self.two_coset.final_poly_len(),
            p.point_lookup_counts.clone(),
        );
        let point = generate_native_terminal_whir_point_trace(&point_air, verification, None)
            .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let final_check_air = NativeTerminalWhirFinalCheckAir {
            final_context_bus: self.tail.whir_final_context,
            final_poly_bus: self.tail.whir_final_poly,
            final_weight_bus: self.tail.whir_final_weight,
            point_bus: self.tail.whir_point,
            actual_weight_bus: self.tail.whir_actual_weight,
            point_prefix_len: self.two_coset.whir_sumcheck_round_count(),
            final_log_len: self.two_coset.whir_remaining_dimension(),
        };
        let final_check =
            generate_native_terminal_whir_final_check_trace(&final_check_air, verification, None)
                .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let weight_term = generate_native_terminal_whir_weight_term_trace(
            verification,
            self.two_coset.whir_k(),
            None,
        )
        .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let adjoint_sumcheck_air = NativeTerminalRsAdjointSumcheckAir::new(
            self.tail.transcript,
            self.tail.adjoint_round,
            self.tail.adjoint_claim,
            p.alpha_len(),
            p.log_message_len() + 1,
        );
        let adjoint_sumcheck = generate_native_terminal_rs_adjoint_sumcheck_trace(
            &adjoint_sumcheck_air,
            verification,
            witness.transcript,
            None,
        )
        .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let adjoint_q =
            generate_two_coset_adjoint_q_trace(verification, &instance.alpha, p.log_message_len())?;
        let adjoint_y_air = FiniteWarpV3TwoCosetRsAdjointYAir::new(
            self.tail.adjoint_round,
            self.tail.adjoint_y,
            p.log_message_len(),
        )?;
        let adjoint_y = generate_two_coset_adjoint_y_trace(&adjoint_y_air, verification)?;
        let adjoint_selector = generate_two_coset_adjoint_selector_trace(
            verification,
            &instance.alpha,
            p.log_message_len(),
        )?;
        let expected_weight = generate_native_terminal_whir_expected_weight_trace(
            verification,
            self.two_coset.whir_k(),
            None,
        )
        .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let exp_bits = generate_two_coset_terminal_exp_bits_trace(
            verification,
            witness.transcript,
            self.two_coset.alpha_len(),
            self.two_coset.folding_pow_bits(),
            self.two_coset.query_phase_pow_bits(),
        )?;

        let mut traces = vec![
            FixedMultiAirCompleteTerminalPartitionedTrace::simple(resumed_transcript),
            FixedMultiAirCompleteTerminalPartitionedTrace::simple(seam),
            cached(statement_prefix.cached, statement_prefix.common),
        ];
        traces.extend(nonlinear.traces);
        traces.extend([
            cached(rs_statement.cached, rs_statement.common),
            simple(prefix),
            cached(target.cached, target.common),
            cached(bridge_cached, bridge_common),
            cached(linearizer_sumcheck.cached, linearizer_sumcheck.common),
            cached(linearizer_y_cached, linearizer_y_common),
            simple(linearizer_selector),
            cached(raw.cached, raw.common),
            simple(linearizer_raw_sum),
            simple(linearizer_final),
            simple(whir_sumcheck),
            simple(whir_round),
            simple(whir_query),
            simple(whir_opened.matrix),
            simple(whir_folding),
            simple(leaf_hash.matrix),
            simple(merkle),
            simple(final_table),
            simple(mobius),
            simple(point),
            simple(final_check),
            simple(weight_term),
            simple(adjoint_sumcheck),
            simple(adjoint_q),
            simple(adjoint_y),
            simple(adjoint_selector),
            simple(expected_weight),
            simple(exp_bits),
            simple(digest.values),
            simple(digest.hash),
            simple(digest.root),
        ]);
        poseidon2_permute_inputs.extend(digest.poseidon2_permute_inputs);
        poseidon2_compress_inputs.extend(digest.poseidon2_compress_inputs);
        let expected = self.airs::<BabyBearPoseidon2Config>().len();
        if traces.len() != expected {
            return Err(FiniteWarpV3TerminalDecideError::AirTraceCount {
                expected,
                actual: traces.len(),
            });
        }
        for trace in &mut traces {
            trace
                .inline_cached_mains()
                .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        }
        Ok(FiniteWarpV3CompleteTerminalDecideTraceData {
            traces,
            poseidon2_permute_inputs,
            poseidon2_compress_inputs,
            receipt: FiniteWarpV3TerminalReceiptRecord {
                final_accumulator_digest: digest.instance_digest,
                final_accumulator_root: instance.rt,
            },
        })
    }
}

#[derive(Clone, Debug)]
pub struct FiniteWarpV3CompleteTerminalDecideTraceData {
    pub traces: Vec<FixedMultiAirCompleteTerminalPartitionedTrace>,
    pub poseidon2_permute_inputs: Vec<[F; 16]>,
    pub poseidon2_compress_inputs: Vec<[F; 16]>,
    pub receipt: FiniteWarpV3TerminalReceiptRecord,
}

fn simple(matrix: RowMajorMatrix<F>) -> FixedMultiAirCompleteTerminalPartitionedTrace {
    FixedMultiAirCompleteTerminalPartitionedTrace::simple(matrix)
}

fn cached(
    cached: RowMajorMatrix<F>,
    common: RowMajorMatrix<F>,
) -> FixedMultiAirCompleteTerminalPartitionedTrace {
    FixedMultiAirCompleteTerminalPartitionedTrace::cached(cached, common)
}

fn reserve_through(
    manager: &mut BusIndexManager,
    target: BusIndex,
) -> Result<(), FiniteWarpV3TerminalDecideError> {
    if manager.next_bus_idx() > target {
        return Err(FiniteWarpV3TerminalDecideError::TerminalAirInventory);
    }
    while manager.next_bus_idx() < target {
        manager.new_bus_idx();
    }
    Ok(())
}

fn terminal_point_lookup_counts(
    k: usize,
    queries: &[usize],
    final_poly_len: usize,
    log_message_len: usize,
) -> Option<Vec<usize>> {
    let round_count = queries.len();
    let point_len = k
        .checked_mul(round_count)?
        .checked_add(final_poly_len.ilog2() as usize)?;
    if point_len != log_message_len {
        return None;
    }
    // Structured-linearizer selector and coefficient-two-coset adjoint.
    let mut counts = vec![2usize; point_len];
    for (round, &query_count) in queries.iter().enumerate() {
        let after_folds = (round + 1).checked_mul(k)?;
        let terms = query_count.checked_add(usize::from(round + 1 != round_count))?;
        for count in counts.get_mut(after_folds..)? {
            *count = count.checked_add(terms)?;
        }
    }
    let suffix_start = k.checked_mul(round_count)?;
    for count in counts.get_mut(suffix_start..)? {
        *count = count.checked_add(final_poly_len)?;
    }
    Some(counts)
}

fn add<SC, A>(airs: &mut Vec<AirRef<SC>>, air: A)
where
    SC: StarkProtocolConfig,
    A: openvm_stark_backend::AnyAir<SC> + 'static,
{
    airs.push(Arc::new(air));
}

fn add_inlined<SC, A>(airs: &mut Vec<AirRef<SC>>, air: A)
where
    SC: StarkProtocolConfig,
    A: PartitionedBaseAir<SC::F>,
    InlinedCachedAir<A>: openvm_stark_backend::AnyAir<SC> + 'static,
{
    airs.push(Arc::new(InlinedCachedAir::new(air)));
}

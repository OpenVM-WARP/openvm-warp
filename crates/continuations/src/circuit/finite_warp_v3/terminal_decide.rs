//! Terminal-`Decide` components for the bounded finite-WARP v3 wrapper.
//!
//! The reusable coefficient-two-coset transcript, WHIR, exact RS-adjoint, and
//! receipt AIRs live here.  The prior producer attempted to wrap
//! [`FixedMultiAirTerminalCircuit`], but that circuit owns the local-only
//! `FixedMultiAirPesatIndex`/`FixedMultiAirTerminalProof` relation.  Native
//! finite-v3 Decide owns the distinct complete relation and
//! `FixedMultiAirCompleteTerminalProof`, including inverse and global LogUp
//! constraints.  Consequently [`FiniteWarpV3TerminalDecideProducer::new`]
//! fails closed until a genuine complete-relation recursive owner exists.
//!
//! The security-critical interaction graph is:
//!
//! ```text
//! fixed terminal prefix -- full (alpha,mu,beta,eta) --> instance bridge
//!          |                                             |
//!          | relation/root                              v
//!          +------------------------------> accumulator hash
//!          |                                             |
//!          | same root                                   v
//!          +--> nonlinear reduction --> adjoint --> WHIR +--> v3 receipt
//! ```
//!
//! `generate_traces` performs fail-fast witness validation, but acceptance is
//! determined by the AIR interactions above.  In particular, callers cannot
//! construct a receipt from a host-side result without supplying the complete
//! terminal verifier traces.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_circuit_primitives::{encoder::Encoder, utils::assert_array_eq, ColumnsAir, SubAir};
use openvm_recursion_circuit::{
    bus::{
        Poseidon2CompressBus, Poseidon2CompressMessage, Poseidon2PermuteBus,
        ResumeTranscriptStateBus,
    },
    native_warp::{
        generate_native_leaf_hash_trace, generate_native_merkle_multiproof_trace,
        populate_coefficient_two_coset_query_aux_requests,
        terminal::{
            FixedMultiAirLinearizerAdjointProof, FixedMultiAirTerminalBindingBus,
            FixedMultiAirTerminalBindingMessage, FixedMultiAirTerminalCircuit,
            FixedMultiAirTerminalInstanceValueBus, FixedMultiAirTerminalInstanceValueMessage,
            FixedMultiAirTerminalPartitionedTrace, FixedMultiAirTerminalTraceData,
            FixedMultiAirTerminalTraceWitness, NativeTerminalAccumulatorRootBus,
            NativeTerminalAccumulatorRootMessage, NativeTerminalAccumulatorValueBus,
            NativeTerminalAccumulatorValueMessage, NativeTerminalRsAdjointClaimBus,
            NativeTerminalRsAdjointClaimMessage, NativeTerminalRsAdjointQBus,
            NativeTerminalRsAdjointQCols, NativeTerminalRsAdjointQMessage,
            NativeTerminalRsAdjointRoundBus, NativeTerminalRsAdjointRoundMessage,
            NativeTerminalRsAdjointSelectorCols, NativeTerminalRsAdjointValueBus,
            NativeTerminalRsAdjointValueMessage, NativeTerminalRsAdjointYBus,
            NativeTerminalRsAdjointYCols, NativeTerminalRsAdjointYMessage,
            NativeTerminalWhirPointBus, NativeTerminalWhirPointMessage,
            NATIVE_TERMINAL_SELECTOR_MAX_FLAG_DEGREE,
        },
        NativeAccumulatorAlgebraicDigestBus, NativeAccumulatorAlgebraicDigestMessage,
        NativeAccumulatorDigestElementBus, NativeAccumulatorDigestElementMessage,
        NativeLeafHashCols, NativeLeafHashInput, NativeMerkleCompressionCols,
    },
    primitives::exp_bits_len::ExpBitsLenCpuTraceGenerator,
    transcript::{
        transcript::{TranscriptAir, TranscriptCols},
        Poseidon2BusOwner,
    },
    utils::{ext_field_add, ext_field_multiply, ext_field_subtract},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    native_warp::{
        FixedMultiAirCompletePesatIndex, FixedMultiAirCompleteTerminalProof,
        FINITE_COMPLETE_TERMINAL_INDEX_TAG, FINITE_COMPLETE_TERMINAL_PACKAGE_VERSION,
    },
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{
        extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing,
        PrimeField32, TwoAdicField,
    },
    transcript::TranscriptLog,
    warp_accum::{
        BinaryMerkleMultiproofRecord, TerminalConstrainedLinearizerProof,
        TerminalConstrainedRsStatement, TerminalDescriptor, TerminalWhirVerification,
        WhirInitialRsLayout,
    },
    warp_pesat::AccumulatorInstance,
    AirRef, BaseAirWithPublicValues, FiatShamirTranscript, PartitionedBaseAir, StarkProtocolConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, BabyBearPoseidon2Config, Digest, DuplexSpongeRecorder,
    DIGEST_SIZE, D_EF, EF, F,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::{
    finite_warp_v3_terminal_transcript_seam_trace,
    generate_finite_warp_v3_resumed_terminal_transcript_trace,
    generate_finite_warp_v3_two_coset_terminal_whir_folding_trace,
    generate_finite_warp_v3_two_coset_terminal_whir_opened_trace,
    generate_finite_warp_v3_two_coset_terminal_whir_query_trace,
    generate_finite_warp_v3_two_coset_whir_prefix_trace, FiniteWarpV3FinalVaccCheckpointBus,
    FiniteWarpV3TerminalAccumulatorLinkBus, FiniteWarpV3TerminalAccumulatorLinkMessage,
    FiniteWarpV3TerminalReceiptBus, FiniteWarpV3TerminalReceiptMessage,
    FiniteWarpV3TerminalTranscriptCheckpoint, FiniteWarpV3TerminalTranscriptSeamAir,
    FiniteWarpV3TwoCosetTerminalProfile, FiniteWarpV3TwoCosetTerminalWhirFoldingAir,
    FiniteWarpV3TwoCosetTerminalWhirOpenedAir, FiniteWarpV3TwoCosetTerminalWhirQueryAir,
    FiniteWarpV3TwoCosetWhirPrefixAir, FINITE_WARP_V3_MAX_CALLS, FINITE_WARP_V3_TWO_COSET_K,
};
use crate::circuit::native_warp_accumulator::{
    generate_native_accumulator_digest_traces, NativeAccumulatorHashAir,
    NativeAccumulatorRootDigestCols, NativeAccumulatorValueCols, NativePrivateAccumulatorLayout,
};

/// Setup-owned buses added by this receipt adapter.  Their indices must be
/// allocated by the enclosing circuit so they cannot alias another producer.
#[derive(Clone, Copy, Debug)]
pub struct FiniteWarpV3TerminalDecideBuses {
    pub receipt: FiniteWarpV3TerminalReceiptBus,
    pub digest_element: NativeAccumulatorDigestElementBus,
    pub algebraic_digest: NativeAccumulatorAlgebraicDigestBus,
    /// Authenticated end checkpoint emitted by the final WARP verifier call.
    pub final_vacc_checkpoint: FiniteWarpV3FinalVaccCheckpointBus,
    /// Private handoff bus consumed by the replacement terminal transcript AIR.
    pub terminal_resume: ResumeTranscriptStateBus,
    /// Private root/digest bridge from the transcript seam to the receipt AIR.
    pub terminal_accumulator_link: FiniteWarpV3TerminalAccumulatorLinkBus,
}

/// Fixed statement constants carried by the terminal receipt lookup.
///
/// `terminal_index_digest` is validated against the exact relation digest and
/// fixed terminal descriptor metadata in [`FiniteWarpV3TerminalDecideProducer::new`].
/// The enclosing v3 setup is responsible for deriving
/// `verifier_component_digest` from the complete manifest/WARP/terminal AIR
/// inventory before constructing this producer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteWarpV3TerminalDecideBinding {
    pub protocol_digest: Digest,
    pub terminal_index_digest: Digest,
    pub verifier_component_digest: Digest,
    pub warp_index_digest: Digest,
    pub vacc_setup_digest: Digest,
    pub schedule_digest: Digest,
    pub warp_call_count: usize,
}

/// Exact nonlinear proof type carried by
/// `openvm_sdk::prover::native_warp::FiniteCompleteTerminalProof`.
///
/// This alias is intentionally incompatible with the legacy
/// `FixedMultiAirTerminalProof<EF>` accepted by
/// `FixedMultiAirTerminalTraceWitness`.
pub type FiniteWarpV3CompleteNonlinearProof =
    TerminalConstrainedLinearizerProof<FixedMultiAirCompleteTerminalProof<EF>>;

/// Required input contract for the missing complete-relation recursive owner.
///
/// The SDK adapter must construct this directly from a successfully replayed
/// native `FiniteCompleteTerminalProof` and its
/// `FiniteCompleteTerminalVerification`.  No field may be converted through
/// the legacy local-only proof or relation.
pub struct FiniteWarpV3CompleteTerminalTraceWitness<'a> {
    pub relation: &'a FixedMultiAirCompletePesatIndex<F, Digest>,
    pub descriptor: &'a TerminalDescriptor<Digest>,
    pub instance: &'a AccumulatorInstance<EF, Digest>,
    pub reduction_proof: &'a FiniteWarpV3CompleteNonlinearProof,
    pub statement: &'a TerminalConstrainedRsStatement<EF>,
    pub whir_verification: &'a TerminalWhirVerification<F, EF, Digest>,
    pub linearizer_adjoint_proof: &'a FixedMultiAirLinearizerAdjointProof,
    pub transcript: &'a TranscriptLog<F, [F; 16]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiniteWarpV3TerminalDecideError {
    ZeroProtocolDigest,
    ZeroRelationDigest,
    ZeroTerminalIndexDigest,
    ZeroVerifierComponentDigest,
    ZeroWarpIndexDigest,
    ZeroVaccSetupDigest,
    ZeroScheduleDigest,
    InvalidWarpCallCount,
    /// The in-tree recursive owner accepts only the legacy local-AIR
    /// `FixedMultiAirTerminalProof`; production requires a distinct owner for
    /// `FixedMultiAirCompleteTerminalProof`.
    CompleteRelationCircuitUnavailable,
    TerminalIndexDigest,
    UnsupportedTwoCosetProfile,
    TerminalAirInventory,
    InvalidAccumulatorShape,
    TwoCosetAdjoint,
    PoseidonSuffix,
    TerminalTrace,
    AirTraceCount {
        expected: usize,
        actual: usize,
    },
}

/// Recompute the native finite-terminal index digest from the data fixed in
/// the terminal verifier key.
///
/// `metadata_words` is exactly `TerminalDescriptor::metadata_words()` and
/// intentionally excludes the runtime accumulator root.  The root is instead
/// linked by both the terminal binding bus and the WHIR root bus.
#[must_use]
pub fn finite_warp_v3_terminal_index_digest(
    relation_digest: Digest,
    metadata_words: &[u64],
) -> Digest {
    let mut transcript = default_duplex_sponge_recorder();
    observe_u64(&mut transcript, FINITE_COMPLETE_TERMINAL_INDEX_TAG);
    observe_u64(
        &mut transcript,
        u64::from(FINITE_COMPLETE_TERMINAL_PACKAGE_VERSION),
    );
    for coordinate in relation_digest {
        <DuplexSpongeRecorder as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
            &mut transcript,
            coordinate,
        );
    }
    observe_u64(&mut transcript, metadata_words.len() as u64);
    for &word in metadata_words {
        observe_u64(&mut transcript, word);
    }
    core::array::from_fn(|_| {
        <DuplexSpongeRecorder as FiatShamirTranscript<BabyBearPoseidon2Config>>::sample(
            &mut transcript,
        )
    })
}

fn observe_u64(transcript: &mut DuplexSpongeRecorder, value: u64) {
    // Four canonical 16-bit limbs are used by the native terminal protocol.
    // A single reduced field element would identify distinct u64 words.
    for shift in [0, 16, 32, 48] {
        <DuplexSpongeRecorder as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
            transcript,
            F::from_u32(((value >> shift) & 0xffff) as u32),
        );
    }
}

fn is_zero_digest(digest: Digest) -> bool {
    digest == [F::ZERO; DIGEST_SIZE]
}

/// Row-oriented binder for every extension-field coordinate of the final
/// accumulator.  Its state machine fixes the exact sequence
/// `alpha || mu || beta || eta`, receives each value from the genuine terminal
/// prefix, and sends the canonical base-field digest preimage to
/// [`NativeAccumulatorHashAir`].
pub struct FiniteWarpV3TerminalInstanceAir {
    pub layout: NativePrivateAccumulatorLayout,
    pub terminal_instance_bus: FixedMultiAirTerminalInstanceValueBus,
    pub digest_element_bus: NativeAccumulatorDigestElementBus,
}

impl BaseAir<F> for FiniteWarpV3TerminalInstanceAir {
    fn width(&self) -> usize {
        NativeAccumulatorValueCols::<F>::width()
    }
}

impl BaseAirWithPublicValues<F> for FiniteWarpV3TerminalInstanceAir {}
impl PartitionedBaseAir<F> for FiniteWarpV3TerminalInstanceAir {}

impl<AB> Air<AB> for FiniteWarpV3TerminalInstanceAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main
            .row_slice(0)
            .expect("finite WARP v3 terminal instance row");
        let next_row = main
            .row_slice(1)
            .expect("finite WARP v3 terminal instance next row");
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

        let section = is_mu.clone()
            + is_beta.clone() * AB::Expr::from_u32(2)
            + is_eta.clone() * AB::Expr::from_u32(3);
        self.terminal_instance_bus.lookup_key(
            builder,
            FixedMultiAirTerminalInstanceValueMessage {
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

/// Final one-row bridge.  It consumes all authorities emitted by the complete
/// terminal verifier and produces exactly one v3 terminal receipt.
pub struct FiniteWarpV3TerminalDecideReceiptAir {
    pub binding: FiniteWarpV3TerminalDecideBinding,
    pub relation_digest: Digest,
    pub alpha_len: usize,
    pub beta_len: usize,
    pub receipt_bus: FiniteWarpV3TerminalReceiptBus,
    pub terminal_binding_bus: FixedMultiAirTerminalBindingBus,
    pub terminal_whir_root_bus: NativeTerminalAccumulatorRootBus,
    pub algebraic_digest_bus: NativeAccumulatorAlgebraicDigestBus,
    pub compress_bus: Poseidon2CompressBus,
    pub terminal_accumulator_link_bus: FiniteWarpV3TerminalAccumulatorLinkBus,
}

impl BaseAir<F> for FiniteWarpV3TerminalDecideReceiptAir {
    fn width(&self) -> usize {
        NativeAccumulatorRootDigestCols::<F>::width()
    }
}

impl BaseAirWithPublicValues<F> for FiniteWarpV3TerminalDecideReceiptAir {}
impl PartitionedBaseAir<F> for FiniteWarpV3TerminalDecideReceiptAir {}

impl<AB> Air<AB> for FiniteWarpV3TerminalDecideReceiptAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("finite WARP v3 terminal receipt row");
        let local: &NativeAccumulatorRootDigestCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_transition().assert_zero(local.active);
        builder.when(local.active).assert_zero(local.proof_idx);

        let enabled = Into::<AB::Expr>::into(local.active);
        self.terminal_binding_bus.receive(
            builder,
            FixedMultiAirTerminalBindingMessage {
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

/// Coefficient-native equality kernel for the RS-adjoint sumcheck endpoint.
///
/// The backend orders the interleaved codeword variables as
/// `coset_bit || subgroup_bits_msb_first`, while the accumulator stores
/// `alpha` in Boolean-MLE order.  The legacy vector-alphabet Q AIR drops the
/// first `k` coordinates and is therefore not a valid adapter for this code.
#[derive(Clone, Debug)]
pub struct FiniteWarpV3TwoCosetRsAdjointQAir {
    pub round_bus: NativeTerminalRsAdjointRoundBus,
    pub accumulator_value_bus: NativeTerminalAccumulatorValueBus,
    pub q_bus: NativeTerminalRsAdjointQBus,
    pub log_message_len: usize,
}

impl BaseAirWithPublicValues<F> for FiniteWarpV3TwoCosetRsAdjointQAir {}
impl PartitionedBaseAir<F> for FiniteWarpV3TwoCosetRsAdjointQAir {}
impl ColumnsAir for FiniteWarpV3TwoCosetRsAdjointQAir {}
impl BaseAir<F> for FiniteWarpV3TwoCosetRsAdjointQAir {
    fn width(&self) -> usize {
        NativeTerminalRsAdjointQCols::<F>::width()
    }
}

impl<AB> Air<AB> for FiniteWarpV3TwoCosetRsAdjointQAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main
            .row_slice(0)
            .expect("finite two-coset terminal adjoint q row");
        let next_row = main
            .row_slice(1)
            .expect("finite two-coset terminal next adjoint q row");
        let local: &NativeTerminalRsAdjointQCols<AB::Var> = (*local_row).borrow();
        let next: &NativeTerminalRsAdjointQCols<AB::Var> = (*next_row).borrow();

        for flag in [local.active, local.is_first, local.is_last, local.is_column] {
            builder.assert_bool(flag);
        }
        builder
            .when(local.active)
            .assert_eq(local.is_column, local.is_first);
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when_first_row().assert_zero(local.round);
        builder
            .when(local.active * local.is_last)
            .assert_eq(local.round, AB::Expr::from_usize(self.log_message_len));
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.is_last);
        let mut transition = builder.when_transition();
        let mut same = transition.when(next.active);
        same.assert_zero(next.is_first);
        same.assert_eq(next.round, local.round + AB::F::ONE);
        assert_array_eq(&mut same, next.prefix_before, local.prefix_after);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);

        let one = one_ext::<AB>();
        assert_array_eq(
            &mut builder.when(local.is_first),
            local.prefix_before,
            one.clone(),
        );
        let factor = ext_field_add::<AB::Expr>(
            ext_field_multiply::<AB::Expr>(
                ext_field_subtract::<AB::Expr>(one.clone(), local.alpha),
                ext_field_subtract::<AB::Expr>(one, local.challenge),
            ),
            ext_field_multiply::<AB::Expr>(local.alpha, local.challenge),
        );
        assert_array_eq(&mut builder.when(local.active), local.factor, factor);
        assert_array_eq(
            &mut builder.when(local.active),
            local.prefix_after,
            ext_field_multiply::<AB::Expr>(local.prefix_before, local.factor),
        );
        // These columns exist only because the production adapter reuses the
        // established bounded Q row layout.  They have no two-coset meaning.
        for limb in local.whir_point {
            builder.when(local.active).assert_zero(limb);
        }

        self.round_bus.lookup_key(
            builder,
            NativeTerminalRsAdjointRoundMessage {
                round: local.round.into(),
                challenge: local.challenge.map(Into::into),
            },
            local.active,
        );
        // round 0 consumes the physical coset selector alpha[m]; subsequent
        // rounds consume subgroup coordinates alpha[0], ..., alpha[m-1].
        self.accumulator_value_bus.lookup_key(
            builder,
            NativeTerminalAccumulatorValueMessage {
                section: AB::Expr::ZERO,
                coordinate: local.round
                    + local.is_first * AB::Expr::from_usize(self.log_message_len + 1)
                    - AB::Expr::ONE,
                value: local.alpha.map(Into::into),
            },
            local.active,
        );
        self.q_bus.send(
            builder,
            NativeTerminalRsAdjointQMessage {
                value: local.prefix_after.map(Into::into),
            },
            local.is_last,
        );
    }
}

/// Exact geometric factors of the coefficient-two-coset encoder transpose.
/// Roots are setup-fixed and mirror `TwoCosetAdjointKernel` in the backend.
pub struct FiniteWarpV3TwoCosetRsAdjointYAir {
    pub round_bus: NativeTerminalRsAdjointRoundBus,
    pub y_bus: NativeTerminalRsAdjointYBus,
    pub log_message_len: usize,
    encoder: Encoder,
    schedule: Vec<(usize, usize, F)>,
}

impl core::fmt::Debug for FiniteWarpV3TwoCosetRsAdjointYAir {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("FiniteWarpV3TwoCosetRsAdjointYAir")
            .field("log_message_len", &self.log_message_len)
            .field("schedule_len", &self.schedule.len())
            .finish_non_exhaustive()
    }
}

impl FiniteWarpV3TwoCosetRsAdjointYAir {
    pub fn new(
        round_bus: NativeTerminalRsAdjointRoundBus,
        y_bus: NativeTerminalRsAdjointYBus,
        log_message_len: usize,
    ) -> Result<Self, FiniteWarpV3TerminalDecideError> {
        if log_message_len == 0 || log_message_len > F::TWO_ADICITY {
            return Err(FiniteWarpV3TerminalDecideError::TwoCosetAdjoint);
        }
        let schedule = two_coset_adjoint_root_schedule(log_message_len)?;
        let encoder = Encoder::new(
            schedule.len().max(2),
            NATIVE_TERMINAL_SELECTOR_MAX_FLAG_DEGREE,
            false,
        );
        Ok(Self {
            round_bus,
            y_bus,
            log_message_len,
            encoder,
            schedule,
        })
    }

    fn eval_impl<AB, const ENC_WIDTH: usize>(&self, builder: &mut AB)
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
        AB::Expr: From<AB::Var>,
        <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
    {
        let main = builder.main();
        let local_row = main
            .row_slice(0)
            .expect("finite two-coset terminal adjoint y row");
        let next_row = main
            .row_slice(1)
            .expect("finite two-coset terminal next adjoint y row");
        let local: &NativeTerminalRsAdjointYCols<AB::Var, ENC_WIDTH> = (*local_row).borrow();
        let next: &NativeTerminalRsAdjointYCols<AB::Var, ENC_WIDTH> = (*next_row).borrow();

        for flag in [
            local.active,
            local.is_first_in_bit,
            local.is_last_in_bit,
            local.is_first,
            local.is_last,
        ] {
            builder.assert_bool(flag);
        }
        self.encoder.eval(builder, &local.encoding);
        let decoded = |values: Vec<(usize, usize)>| {
            self.encoder.flag_with_val::<AB>(&local.encoding, &values)
        };
        builder.when(local.active).assert_eq(
            local.ordinal,
            decoded(
                (0..self.schedule.len())
                    .map(|ordinal| (ordinal, ordinal))
                    .collect(),
            ),
        );
        builder.when(local.active).assert_eq(
            local.message_bit,
            decoded(
                self.schedule
                    .iter()
                    .enumerate()
                    .map(|(ordinal, &(bit, _, _))| (ordinal, bit))
                    .collect(),
            ),
        );
        builder.when(local.active).assert_eq(
            local.round,
            decoded(
                self.schedule
                    .iter()
                    .enumerate()
                    .map(|(ordinal, &(_, round, _))| (ordinal, round))
                    .collect(),
            ),
        );
        builder.when(local.active).assert_eq(
            local.root,
            decoded(
                self.schedule
                    .iter()
                    .enumerate()
                    .map(|(ordinal, &(_, _, root))| (ordinal, root.as_canonical_u32() as usize))
                    .collect(),
            ),
        );
        builder.when(local.active).assert_eq(
            local.is_first_in_bit,
            decoded(
                self.schedule
                    .iter()
                    .enumerate()
                    .map(|(ordinal, &(_, round, _))| (ordinal, usize::from(round == 0)))
                    .collect(),
            ),
        );
        builder.when(local.active).assert_eq(
            local.is_last_in_bit,
            decoded(
                self.schedule
                    .iter()
                    .enumerate()
                    .map(|(ordinal, &(_, round, _))| {
                        (ordinal, usize::from(round == self.log_message_len))
                    })
                    .collect(),
            ),
        );
        builder.when(local.active).assert_eq(
            local.is_first,
            decoded(
                (0..self.schedule.len())
                    .map(|ordinal| (ordinal, usize::from(ordinal == 0)))
                    .collect(),
            ),
        );
        builder.when(local.active).assert_eq(
            local.is_last,
            decoded(
                (0..self.schedule.len())
                    .map(|ordinal| (ordinal, usize::from(ordinal + 1 == self.schedule.len())))
                    .collect(),
            ),
        );

        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.is_last);
        let mut transition = builder.when_transition();
        transition
            .when(next.active)
            .assert_eq(next.ordinal, local.ordinal + AB::F::ONE);
        let same_bit = next.active * (AB::Expr::ONE - next.is_first_in_bit);
        assert_array_eq(
            &mut builder.when_transition().when(same_bit),
            next.prefix_before,
            local.prefix_after,
        );
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);

        assert_array_eq(
            &mut builder.when(local.is_first_in_bit),
            local.prefix_before,
            one_ext::<AB>(),
        );
        let factor = ext_field_add::<AB::Expr>(
            ext_field_subtract::<AB::Expr>(one_ext::<AB>(), local.challenge),
            core::array::from_fn(|limb| AB::Expr::from(local.challenge[limb]) * local.root),
        );
        assert_array_eq(&mut builder.when(local.active), local.factor, factor);
        assert_array_eq(
            &mut builder.when(local.active),
            local.prefix_after,
            ext_field_multiply::<AB::Expr>(local.prefix_before, local.factor),
        );
        self.round_bus.lookup_key(
            builder,
            NativeTerminalRsAdjointRoundMessage {
                round: local.round.into(),
                challenge: local.challenge.map(Into::into),
            },
            local.active,
        );
        self.y_bus.send(
            builder,
            NativeTerminalRsAdjointYMessage {
                message_bit: local.message_bit.into(),
                value: local.prefix_after.map(Into::into),
            },
            local.active * local.is_last_in_bit,
        );
    }
}

impl BaseAirWithPublicValues<F> for FiniteWarpV3TwoCosetRsAdjointYAir {}
impl PartitionedBaseAir<F> for FiniteWarpV3TwoCosetRsAdjointYAir {}
impl ColumnsAir for FiniteWarpV3TwoCosetRsAdjointYAir {}
impl BaseAir<F> for FiniteWarpV3TwoCosetRsAdjointYAir {
    fn width(&self) -> usize {
        match self.encoder.width() {
            1 => NativeTerminalRsAdjointYCols::<F, 1>::width(),
            2 => NativeTerminalRsAdjointYCols::<F, 2>::width(),
            3 => NativeTerminalRsAdjointYCols::<F, 3>::width(),
            4 => NativeTerminalRsAdjointYCols::<F, 4>::width(),
            5 => NativeTerminalRsAdjointYCols::<F, 5>::width(),
            6 => NativeTerminalRsAdjointYCols::<F, 6>::width(),
            7 => NativeTerminalRsAdjointYCols::<F, 7>::width(),
            8 => NativeTerminalRsAdjointYCols::<F, 8>::width(),
            9 => NativeTerminalRsAdjointYCols::<F, 9>::width(),
            10 => NativeTerminalRsAdjointYCols::<F, 10>::width(),
            11 => NativeTerminalRsAdjointYCols::<F, 11>::width(),
            _ => unreachable!("validated two-coset adjoint schedule"),
        }
    }
}

impl<AB> Air<AB> for FiniteWarpV3TwoCosetRsAdjointYAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        match self.encoder.width() {
            1 => self.eval_impl::<AB, 1>(builder),
            2 => self.eval_impl::<AB, 2>(builder),
            3 => self.eval_impl::<AB, 3>(builder),
            4 => self.eval_impl::<AB, 4>(builder),
            5 => self.eval_impl::<AB, 5>(builder),
            6 => self.eval_impl::<AB, 6>(builder),
            7 => self.eval_impl::<AB, 7>(builder),
            8 => self.eval_impl::<AB, 8>(builder),
            9 => self.eval_impl::<AB, 9>(builder),
            10 => self.eval_impl::<AB, 10>(builder),
            11 => self.eval_impl::<AB, 11>(builder),
            _ => unreachable!("validated two-coset adjoint schedule"),
        }
    }
}

/// Final coefficient-native endpoint.  This is intentionally a distinct AIR
/// type from the vector selector so the verifier-key inventory cannot silently
/// retain the old `(m-k, offset=k)` semantics.
#[derive(Clone, Debug)]
pub struct FiniteWarpV3TwoCosetRsAdjointSelectorAir {
    pub point_bus: NativeTerminalWhirPointBus,
    pub q_bus: NativeTerminalRsAdjointQBus,
    pub y_bus: NativeTerminalRsAdjointYBus,
    pub claim_bus: NativeTerminalRsAdjointClaimBus,
    pub value_bus: NativeTerminalRsAdjointValueBus,
    pub log_message_len: usize,
}

impl BaseAirWithPublicValues<F> for FiniteWarpV3TwoCosetRsAdjointSelectorAir {}
impl PartitionedBaseAir<F> for FiniteWarpV3TwoCosetRsAdjointSelectorAir {}
impl ColumnsAir for FiniteWarpV3TwoCosetRsAdjointSelectorAir {}
impl BaseAir<F> for FiniteWarpV3TwoCosetRsAdjointSelectorAir {
    fn width(&self) -> usize {
        NativeTerminalRsAdjointSelectorCols::<F>::width()
    }
}

impl<AB> Air<AB> for FiniteWarpV3TwoCosetRsAdjointSelectorAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main
            .row_slice(0)
            .expect("finite two-coset terminal adjoint selector row");
        let next_row = main
            .row_slice(1)
            .expect("finite two-coset terminal next adjoint selector row");
        let local: &NativeTerminalRsAdjointSelectorCols<AB::Var> = (*local_row).borrow();
        let next: &NativeTerminalRsAdjointSelectorCols<AB::Var> = (*next_row).borrow();

        for flag in [local.active, local.is_first, local.is_last] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when_first_row().assert_zero(local.message_bit);
        builder.when(local.active * local.is_last).assert_eq(
            local.message_bit,
            AB::Expr::from_usize(self.log_message_len - 1),
        );
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.is_last);
        let mut transition = builder.when_transition();
        let mut same = transition.when(next.active);
        same.assert_zero(next.is_first);
        same.assert_eq(next.message_bit, local.message_bit + AB::F::ONE);
        assert_array_eq(&mut same, next.selector_before, local.selector_after);
        assert_array_eq(&mut same, next.q, local.q);
        assert_array_eq(&mut same, next.claimed_value, local.claimed_value);
        assert_array_eq(&mut same, next.final_claim, local.final_claim);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);

        assert_array_eq(
            &mut builder.when(local.is_first),
            local.selector_before,
            one_ext::<AB>(),
        );
        let twice_minus_one = ext_field_subtract::<AB::Expr>(
            ext_field_add::<AB::Expr>(local.whir_point, local.whir_point),
            one_ext::<AB>(),
        );
        let factor = ext_field_add::<AB::Expr>(
            ext_field_subtract::<AB::Expr>(one_ext::<AB>(), local.whir_point),
            ext_field_multiply::<AB::Expr>(twice_minus_one, local.y_power),
        );
        assert_array_eq(&mut builder.when(local.active), local.factor, factor);
        assert_array_eq(
            &mut builder.when(local.active),
            local.selector_after,
            ext_field_multiply::<AB::Expr>(local.selector_before, local.factor),
        );
        assert_array_eq(
            &mut builder.when(local.is_last),
            local.final_claim,
            ext_field_multiply::<AB::Expr>(local.q, local.selector_after),
        );

        self.point_bus.lookup_key(
            builder,
            NativeTerminalWhirPointMessage {
                coordinate: local.message_bit.into(),
                value: local.whir_point.map(Into::into),
            },
            local.active,
        );
        self.y_bus.receive(
            builder,
            NativeTerminalRsAdjointYMessage {
                message_bit: local.message_bit.into(),
                value: local.y_power.map(Into::into),
            },
            local.active,
        );
        self.q_bus.receive(
            builder,
            NativeTerminalRsAdjointQMessage {
                value: local.q.map(Into::into),
            },
            local.is_first,
        );
        self.claim_bus.receive(
            builder,
            NativeTerminalRsAdjointClaimMessage {
                claimed_value: local.claimed_value.map(Into::into),
                final_claim: local.final_claim.map(Into::into),
            },
            local.is_first,
        );
        self.value_bus.send(
            builder,
            NativeTerminalRsAdjointValueMessage {
                value: local.claimed_value.map(Into::into),
            },
            local.is_last,
        );
    }
}

fn one_ext<AB: AirBuilder<F = F>>() -> [AB::Expr; D_EF] {
    core::array::from_fn(|limb| {
        if limb == 0 {
            AB::Expr::ONE
        } else {
            AB::Expr::ZERO
        }
    })
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

fn two_coset_adjoint_root_schedule(
    log_message_len: usize,
) -> Result<Vec<(usize, usize, F)>, FiniteWarpV3TerminalDecideError> {
    if log_message_len == 0 || log_message_len > F::TWO_ADICITY {
        return Err(FiniteWarpV3TerminalDecideError::TwoCosetAdjoint);
    }
    let omega = F::two_adic_generator(log_message_len);
    let mut schedule = Vec::with_capacity(log_message_len * (log_message_len + 1));
    for message_bit in 0..log_message_len {
        schedule.push((message_bit, 0, F::GENERATOR.exp_power_of_2(message_bit)));
        for domain_var in 1..=log_message_len {
            let exponent_power = log_message_len - domain_var + message_bit;
            let root = if exponent_power >= log_message_len {
                F::ONE
            } else {
                omega.exp_power_of_2(exponent_power)
            };
            schedule.push((message_bit, domain_var, root));
        }
    }
    Ok(schedule)
}

fn terminal_whir_point(
    verification: &TerminalWhirVerification<F, EF, Digest>,
    log_message_len: usize,
) -> Result<Vec<EF>, FiniteWarpV3TerminalDecideError> {
    let point = verification
        .rounds
        .iter()
        .flat_map(|round| round.alphas.iter().copied())
        .chain(verification.suffix_point.iter().copied())
        .collect::<Vec<_>>();
    if point.len() != log_message_len {
        return Err(FiniteWarpV3TerminalDecideError::TwoCosetAdjoint);
    }
    Ok(point)
}

pub(super) fn generate_two_coset_adjoint_q_trace(
    verification: &TerminalWhirVerification<F, EF, Digest>,
    alpha: &[EF],
    log_message_len: usize,
) -> Result<RowMajorMatrix<F>, FiniteWarpV3TerminalDecideError> {
    let adjoint = &verification.accumulator_adjoint;
    if alpha.len() != log_message_len + 1
        || adjoint.point.len() != log_message_len + 1
        || adjoint.rounds.len() != log_message_len + 1
        || adjoint.expected_final != adjoint.final_claim
    {
        return Err(FiniteWarpV3TerminalDecideError::TwoCosetAdjoint);
    }
    let valid_rows = log_message_len + 1;
    let height = valid_rows.next_power_of_two();
    let width = NativeTerminalRsAdjointQCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    let mut prefix = EF::ONE;
    for round in 0..valid_rows {
        let alpha_coordinate = if round == 0 {
            alpha[log_message_len]
        } else {
            alpha[round - 1]
        };
        let challenge = adjoint.point[round];
        let factor =
            (EF::ONE - alpha_coordinate) * (EF::ONE - challenge) + alpha_coordinate * challenge;
        let before = prefix;
        prefix *= factor;
        let row = &mut values[round * width..(round + 1) * width];
        let cols: &mut NativeTerminalRsAdjointQCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.round = F::from_usize(round);
        cols.is_first = F::from_bool(round == 0);
        cols.is_last = F::from_bool(round + 1 == valid_rows);
        cols.is_column = cols.is_first;
        copy_ext(&mut cols.challenge, challenge);
        copy_ext(&mut cols.alpha, alpha_coordinate);
        copy_ext(&mut cols.factor, factor);
        copy_ext(&mut cols.prefix_before, before);
        copy_ext(&mut cols.prefix_after, prefix);
    }
    Ok(RowMajorMatrix::new(values, width))
}

pub(super) fn generate_two_coset_adjoint_y_trace(
    air: &FiniteWarpV3TwoCosetRsAdjointYAir,
    verification: &TerminalWhirVerification<F, EF, Digest>,
) -> Result<RowMajorMatrix<F>, FiniteWarpV3TerminalDecideError> {
    match air.encoder.width() {
        1 => generate_two_coset_adjoint_y_trace_impl::<1>(air, verification),
        2 => generate_two_coset_adjoint_y_trace_impl::<2>(air, verification),
        3 => generate_two_coset_adjoint_y_trace_impl::<3>(air, verification),
        4 => generate_two_coset_adjoint_y_trace_impl::<4>(air, verification),
        5 => generate_two_coset_adjoint_y_trace_impl::<5>(air, verification),
        6 => generate_two_coset_adjoint_y_trace_impl::<6>(air, verification),
        7 => generate_two_coset_adjoint_y_trace_impl::<7>(air, verification),
        8 => generate_two_coset_adjoint_y_trace_impl::<8>(air, verification),
        9 => generate_two_coset_adjoint_y_trace_impl::<9>(air, verification),
        10 => generate_two_coset_adjoint_y_trace_impl::<10>(air, verification),
        11 => generate_two_coset_adjoint_y_trace_impl::<11>(air, verification),
        _ => Err(FiniteWarpV3TerminalDecideError::TwoCosetAdjoint),
    }
}

fn generate_two_coset_adjoint_y_trace_impl<const ENC_WIDTH: usize>(
    air: &FiniteWarpV3TwoCosetRsAdjointYAir,
    verification: &TerminalWhirVerification<F, EF, Digest>,
) -> Result<RowMajorMatrix<F>, FiniteWarpV3TerminalDecideError> {
    let point = &verification.accumulator_adjoint.point;
    if point.len() != air.log_message_len + 1 || air.schedule.is_empty() {
        return Err(FiniteWarpV3TerminalDecideError::TwoCosetAdjoint);
    }
    let valid_rows = air.schedule.len();
    let height = valid_rows.next_power_of_two();
    let width = NativeTerminalRsAdjointYCols::<F, ENC_WIDTH>::width();
    let mut values = F::zero_vec(height * width);
    let mut prefix = EF::ONE;
    for (ordinal, &(message_bit, round, root)) in air.schedule.iter().enumerate() {
        if round == 0 {
            prefix = EF::ONE;
        }
        let challenge = point[round];
        let before = prefix;
        let factor = (EF::ONE - challenge) + challenge * EF::from(root);
        prefix *= factor;
        let row = &mut values[ordinal * width..(ordinal + 1) * width];
        let cols: &mut NativeTerminalRsAdjointYCols<F, ENC_WIDTH> = row.borrow_mut();
        cols.active = F::ONE;
        cols.ordinal = F::from_usize(ordinal);
        cols.message_bit = F::from_usize(message_bit);
        cols.round = F::from_usize(round);
        cols.is_first_in_bit = F::from_bool(round == 0);
        cols.is_last_in_bit = F::from_bool(round == air.log_message_len);
        cols.is_first = F::from_bool(ordinal == 0);
        cols.is_last = F::from_bool(ordinal + 1 == valid_rows);
        cols.root = root;
        copy_ext(&mut cols.challenge, challenge);
        copy_ext(&mut cols.factor, factor);
        copy_ext(&mut cols.prefix_before, before);
        copy_ext(&mut cols.prefix_after, prefix);
        for (target, value) in cols
            .encoding
            .iter_mut()
            .zip(air.encoder.get_flag_pt(ordinal))
        {
            *target = F::from_u32(value);
        }
    }
    Ok(RowMajorMatrix::new(values, width))
}

pub(super) fn generate_two_coset_adjoint_selector_trace(
    verification: &TerminalWhirVerification<F, EF, Digest>,
    alpha: &[EF],
    log_message_len: usize,
) -> Result<RowMajorMatrix<F>, FiniteWarpV3TerminalDecideError> {
    let whir_point = terminal_whir_point(verification, log_message_len)?;
    let adjoint = &verification.accumulator_adjoint;
    if log_message_len == 0
        || alpha.len() != log_message_len + 1
        || adjoint.point.len() != log_message_len + 1
        || adjoint.expected_final != adjoint.final_claim
    {
        return Err(FiniteWarpV3TerminalDecideError::TwoCosetAdjoint);
    }
    let ordered_alpha = core::iter::once(alpha[log_message_len])
        .chain(alpha[..log_message_len].iter().copied())
        .collect::<Vec<_>>();
    let q = ordered_alpha
        .iter()
        .zip(&adjoint.point)
        .map(|(&alpha, &challenge)| (EF::ONE - alpha) * (EF::ONE - challenge) + alpha * challenge)
        .product::<EF>();
    let schedule = two_coset_adjoint_root_schedule(log_message_len)?;
    let mut y_powers = vec![EF::ONE; log_message_len];
    for &(message_bit, round, root) in &schedule {
        let challenge = adjoint.point[round];
        y_powers[message_bit] *= (EF::ONE - challenge) + challenge * EF::from(root);
    }

    let height = log_message_len.next_power_of_two();
    let width = NativeTerminalRsAdjointSelectorCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    let mut selector = EF::ONE;
    for message_bit in 0..log_message_len {
        let point = whir_point[message_bit];
        let factor = (EF::ONE - point) + (point.double() - EF::ONE) * y_powers[message_bit];
        let before = selector;
        selector *= factor;
        let row = &mut values[message_bit * width..(message_bit + 1) * width];
        let cols: &mut NativeTerminalRsAdjointSelectorCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.message_bit = F::from_usize(message_bit);
        cols.is_first = F::from_bool(message_bit == 0);
        cols.is_last = F::from_bool(message_bit + 1 == log_message_len);
        copy_ext(&mut cols.whir_point, point);
        copy_ext(&mut cols.y_power, y_powers[message_bit]);
        copy_ext(&mut cols.factor, factor);
        copy_ext(&mut cols.selector_before, before);
        copy_ext(&mut cols.selector_after, selector);
        copy_ext(&mut cols.q, q);
        copy_ext(&mut cols.claimed_value, adjoint.claimed_value);
        copy_ext(&mut cols.final_claim, adjoint.final_claim);
    }
    if q * selector != adjoint.final_claim {
        return Err(FiniteWarpV3TerminalDecideError::TwoCosetAdjoint);
    }
    Ok(RowMajorMatrix::new(values, width))
}

pub(super) fn generate_two_coset_terminal_exp_bits_trace(
    verification: &TerminalWhirVerification<F, EF, Digest>,
    transcript: &openvm_stark_backend::transcript::TranscriptLog<F, [F; 16]>,
    initial_log_domain_size: usize,
    folding_pow_bits: usize,
    query_pow_bits: usize,
) -> Result<RowMajorMatrix<F>, FiniteWarpV3TerminalDecideError> {
    let generator = ExpBitsLenCpuTraceGenerator::default();
    for round in &verification.rounds {
        for sumcheck in &round.sumcheck_rounds {
            if folding_pow_bits != 0 {
                let sample_tidx = sumcheck
                    .transcript_span
                    .start
                    .operations
                    .checked_add(2 * D_EF + 1)
                    .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)?;
                let sample = *transcript
                    .values()
                    .get(sample_tidx)
                    .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)?;
                if transcript.samples().get(sample_tidx) != Some(&true) {
                    return Err(FiniteWarpV3TerminalDecideError::TerminalTrace);
                }
                generator.add_request(F::GENERATOR, sample, folding_pow_bits);
            }
        }
        if query_pow_bits != 0 {
            let query_tidx = round
                .transcript_span
                .end
                .operations
                .checked_sub(D_EF + round.query_indices.len())
                .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)?;
            let pow_tidx = query_tidx
                .checked_sub(1)
                .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)?;
            let sample = *transcript
                .values()
                .get(pow_tidx)
                .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)?;
            if transcript.samples().get(pow_tidx) != Some(&true) {
                return Err(FiniteWarpV3TerminalDecideError::TerminalTrace);
            }
            generator.add_request(F::GENERATOR, sample, query_pow_bits);
        }
    }
    populate_coefficient_two_coset_query_aux_requests(
        verification,
        transcript,
        initial_log_domain_size,
        &generator,
    )
    .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
    generator
        .generate_trace_row_major(None)
        .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)
}

fn transcript_permutation_inputs(
    trace: &RowMajorMatrix<F>,
) -> Result<Vec<[F; 16]>, FiniteWarpV3TerminalDecideError> {
    let base_width = TranscriptCols::<F>::width();
    if trace.width() < base_width {
        return Err(FiniteWarpV3TerminalDecideError::PoseidonSuffix);
    }
    let mut inputs = Vec::new();
    for row in 0..trace.height().saturating_sub(1) {
        let local_row = trace
            .row_slice(row)
            .ok_or(FiniteWarpV3TerminalDecideError::PoseidonSuffix)?;
        let next_row = trace
            .row_slice(row + 1)
            .ok_or(FiniteWarpV3TerminalDecideError::PoseidonSuffix)?;
        let local: &TranscriptCols<F> = local_row[..base_width].borrow();
        let next: &TranscriptCols<F> = next_row[..base_width].borrow();
        let active = local.mask[0] == F::ONE;
        let next_active = next.mask[0] == F::ONE;
        if active && next_active && !(local.is_sample == F::ONE && next.is_sample == F::ZERO) {
            inputs.push(local.prev_state);
        }
    }
    Ok(inputs)
}

fn leaf_permutation_inputs(
    trace: &RowMajorMatrix<F>,
) -> Result<Vec<[F; 16]>, FiniteWarpV3TerminalDecideError> {
    if trace.width() != NativeLeafHashCols::<F>::width() {
        return Err(FiniteWarpV3TerminalDecideError::PoseidonSuffix);
    }
    (0..trace.height())
        .map(|row| {
            let row = trace
                .row_slice(row)
                .ok_or(FiniteWarpV3TerminalDecideError::PoseidonSuffix)?;
            let cols: &NativeLeafHashCols<F> = (*row).borrow();
            if cols.active == F::ONE {
                Ok(Some(cols.input))
            } else if cols.active == F::ZERO {
                Ok(None)
            } else {
                Err(FiniteWarpV3TerminalDecideError::PoseidonSuffix)
            }
        })
        .filter_map(Result::transpose)
        .collect()
}

fn merkle_compression_inputs(
    trace: &RowMajorMatrix<F>,
) -> Result<Vec<[F; 16]>, FiniteWarpV3TerminalDecideError> {
    if trace.width() != NativeMerkleCompressionCols::<F>::width() {
        return Err(FiniteWarpV3TerminalDecideError::PoseidonSuffix);
    }
    (0..trace.height())
        .map(|row| {
            let row = trace
                .row_slice(row)
                .ok_or(FiniteWarpV3TerminalDecideError::PoseidonSuffix)?;
            let cols: &NativeMerkleCompressionCols<F> = (*row).borrow();
            if cols.active == F::ONE {
                Ok(Some(core::array::from_fn(|index| {
                    if index < DIGEST_SIZE {
                        cols.left[index]
                    } else {
                        cols.right[index - DIGEST_SIZE]
                    }
                })))
            } else if cols.active == F::ZERO {
                Ok(None)
            } else {
                Err(FiniteWarpV3TerminalDecideError::PoseidonSuffix)
            }
        })
        .filter_map(Result::transpose)
        .collect()
}

/// Fail-closed placeholder retaining the reusable coefficient-two-coset
/// components while the complete-relation recursive owner is implemented.
///
/// This type cannot be constructed through its public API.  In particular,
/// the legacy `FixedMultiAirTerminalCircuit` must not be treated as a verifier
/// for native `FixedMultiAirCompleteTerminalProof`.
pub struct FiniteWarpV3TerminalDecideProducer {
    terminal: Arc<FixedMultiAirTerminalCircuit>,
    two_coset_profile: FiniteWarpV3TwoCosetTerminalProfile,
    resumed_transcript_air: Arc<TranscriptAir>,
    transcript_seam_air: Arc<FiniteWarpV3TerminalTranscriptSeamAir>,
    two_coset_prefix_air: Arc<FiniteWarpV3TwoCosetWhirPrefixAir>,
    two_coset_query_air: Arc<FiniteWarpV3TwoCosetTerminalWhirQueryAir>,
    two_coset_opened_air: Arc<FiniteWarpV3TwoCosetTerminalWhirOpenedAir>,
    two_coset_folding_air: Arc<FiniteWarpV3TwoCosetTerminalWhirFoldingAir>,
    two_coset_adjoint_q_air: Arc<FiniteWarpV3TwoCosetRsAdjointQAir>,
    two_coset_adjoint_y_air: Arc<FiniteWarpV3TwoCosetRsAdjointYAir>,
    two_coset_adjoint_selector_air: Arc<FiniteWarpV3TwoCosetRsAdjointSelectorAir>,
    transcript_index: usize,
    two_coset_prefix_index: usize,
    two_coset_query_index: usize,
    two_coset_opened_index: usize,
    two_coset_folding_index: usize,
    leaf_hash_index: usize,
    merkle_index: usize,
    two_coset_adjoint_q_index: usize,
    two_coset_adjoint_y_index: usize,
    two_coset_adjoint_selector_index: usize,
    exp_bits_len_index: usize,
    instance_air: Arc<FiniteWarpV3TerminalInstanceAir>,
    hash_air: Arc<NativeAccumulatorHashAir>,
    receipt_air: Arc<FiniteWarpV3TerminalDecideReceiptAir>,
    poseidon_owner: Poseidon2BusOwner,
    layout: NativePrivateAccumulatorLayout,
}

impl FiniteWarpV3TerminalDecideProducer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        terminal: Arc<FixedMultiAirTerminalCircuit>,
        two_coset_profile: FiniteWarpV3TwoCosetTerminalProfile,
        binding: FiniteWarpV3TerminalDecideBinding,
        buses: FiniteWarpV3TerminalDecideBuses,
        poseidon_permute_bus: Poseidon2PermuteBus,
        poseidon_compress_bus: Poseidon2CompressBus,
    ) -> Result<Self, FiniteWarpV3TerminalDecideError> {
        let relation_digest = terminal.profile.relation_digest;
        if is_zero_digest(binding.protocol_digest) {
            return Err(FiniteWarpV3TerminalDecideError::ZeroProtocolDigest);
        }
        if is_zero_digest(relation_digest) {
            return Err(FiniteWarpV3TerminalDecideError::ZeroRelationDigest);
        }
        if is_zero_digest(binding.terminal_index_digest) {
            return Err(FiniteWarpV3TerminalDecideError::ZeroTerminalIndexDigest);
        }
        if is_zero_digest(binding.verifier_component_digest) {
            return Err(FiniteWarpV3TerminalDecideError::ZeroVerifierComponentDigest);
        }
        if is_zero_digest(binding.warp_index_digest) {
            return Err(FiniteWarpV3TerminalDecideError::ZeroWarpIndexDigest);
        }
        if is_zero_digest(binding.vacc_setup_digest) {
            return Err(FiniteWarpV3TerminalDecideError::ZeroVaccSetupDigest);
        }
        if is_zero_digest(binding.schedule_digest) {
            return Err(FiniteWarpV3TerminalDecideError::ZeroScheduleDigest);
        }
        if binding.warp_call_count == 0 || binding.warp_call_count > FINITE_WARP_V3_MAX_CALLS {
            return Err(FiniteWarpV3TerminalDecideError::InvalidWarpCallCount);
        }

        // P0 fail-closed guard. `terminal.profile.relation` is statically a
        // `FixedMultiAirPesatIndex`, and its trace witness statically contains
        // `FixedMultiAirTerminalProof<EF>`.  The native production package is
        // `FixedMultiAirCompleteTerminalProof<EF>` over
        // `FixedMultiAirCompletePesatIndex`.  There is no sound conversion:
        // it would discard the inverse and global interaction constraints.
        let _ = (
            &two_coset_profile,
            &buses,
            &poseidon_permute_bus,
            &poseidon_compress_bus,
        );
        return Err(FiniteWarpV3TerminalDecideError::CompleteRelationCircuitUnavailable);

        #[allow(unreachable_code)]
        {
            if terminal.profile.alpha_len != two_coset_profile.alpha_len()
                || terminal.profile.log_message_len != two_coset_profile.log_message_len()
                || terminal.profile.k != two_coset_profile.whir_k()
                || terminal.profile.initial_rs_layout != WhirInitialRsLayout::CoefficientTwoCosetGrs
                || terminal.profile.initial_folding_factor != 0
                || terminal.profile.selector_folding_factor != terminal.profile.log_message_len
                || terminal.profile.alpha_len != terminal.profile.log_message_len + 1
                || terminal.profile.rs_adjoint_round_count != terminal.profile.alpha_len
                || terminal.profile.rs_adjoint_degree != terminal.profile.log_message_len + 1
                || terminal.profile.k != FINITE_WARP_V3_TWO_COSET_K
            {
                return Err(FiniteWarpV3TerminalDecideError::UnsupportedTwoCosetProfile);
            }
            let expected_index = finite_warp_v3_terminal_index_digest(
                relation_digest,
                two_coset_profile.descriptor_metadata_words(),
            );
            if binding.terminal_index_digest != expected_index {
                return Err(FiniteWarpV3TerminalDecideError::TerminalIndexDigest);
            }
            let terminal_owner = terminal.poseidon2_bus_owner();
            if terminal_owner.permute_bus.index() != poseidon_permute_bus.index()
                || terminal_owner.compress_bus.index() != poseidon_compress_bus.index()
            {
                return Err(FiniteWarpV3TerminalDecideError::TerminalTrace);
            }

            let alpha_len = terminal.profile.alpha_len;
            let beta_len = terminal.profile.beta_len;
            if alpha_len == 0 || beta_len == 0 {
                return Err(FiniteWarpV3TerminalDecideError::InvalidAccumulatorShape);
            }
            let layout = NativePrivateAccumulatorLayout::new(0, alpha_len, beta_len);
            let terminal_airs = terminal.airs::<BabyBearPoseidon2Config>();
            let find_unique = |name: &str| -> Result<usize, FiniteWarpV3TerminalDecideError> {
                let mut matches = terminal_airs
                    .iter()
                    .enumerate()
                    .filter_map(|(index, air)| (air.name() == name).then_some(index));
                let index = matches
                    .next()
                    .ok_or(FiniteWarpV3TerminalDecideError::TerminalAirInventory)?;
                if matches.next().is_some() {
                    return Err(FiniteWarpV3TerminalDecideError::TerminalAirInventory);
                }
                Ok(index)
            };
            let transcript_index = find_unique("TranscriptAir")?;
            let two_coset_prefix_index = find_unique("FixedMultiAirWhirPrefixAir")?;
            let two_coset_query_index = find_unique("NativeTerminalWhirQueryAir")?;
            let two_coset_opened_index = find_unique("NativeTerminalWhirOpenedAir")?;
            let two_coset_folding_index = find_unique("NativeTerminalWhirFoldingAir")?;
            let leaf_hash_index = find_unique("NativeLeafHashAir")?;
            let merkle_index = find_unique("NativeMerkleMultiproofAir")?;
            let two_coset_adjoint_q_index = find_unique("NativeTerminalRsAdjointQAir")?;
            let two_coset_adjoint_y_index = find_unique("NativeTerminalRsAdjointYAir")?;
            let two_coset_adjoint_selector_index =
                find_unique("NativeTerminalRsAdjointSelectorAir")?;
            let exp_bits_len_index = find_unique("ExpBitsLenAir")?;
            // This is the exact unaligned, two-carry structured-message adjoint.
            // It remains in place and is required to occur exactly once.
            let _linearizer_raw_index = find_unique("FixedMultiAirLinearizerRawAir")?;
            let b = &terminal.buses;
            let resumed_transcript_air = Arc::new(TranscriptAir {
                transcript_bus: b.transcript,
                poseidon2_permute_bus: poseidon_permute_bus,
                final_state_bus: None,
                resume_state_bus: Some(buses.terminal_resume),
                end_index_bus: None,
                checkpoint_state_bus: None,
            });
            let transcript_seam_air = Arc::new(FiniteWarpV3TerminalTranscriptSeamAir {
                final_vacc_checkpoint_bus: buses.final_vacc_checkpoint,
                terminal_resume_bus: buses.terminal_resume,
                terminal_accumulator_link_bus: buses.terminal_accumulator_link,
                protocol_digest: binding.protocol_digest,
                relation_digest,
                warp_index_digest: binding.warp_index_digest,
                setup_digest: binding.vacc_setup_digest,
                schedule_digest: binding.schedule_digest,
                call_count: binding.warp_call_count,
            });
            let two_coset_prefix_air = Arc::new(FiniteWarpV3TwoCosetWhirPrefixAir {
                transcript_bus: b.transcript,
                binding_bus: b.binding,
                instance_bus: b.instance,
                start_bus: b.whir_start,
                root_bus: b.accumulator_root,
                relation_digest,
                profile: two_coset_profile.clone(),
                beta_len,
            });
            let two_coset_query_air = Arc::new(
                FiniteWarpV3TwoCosetTerminalWhirQueryAir::new(
                    &two_coset_profile,
                    b.transcript,
                    b.whir_verify_queries,
                    b.whir_query,
                    b.weight_term,
                    b.exp_bits_len,
                    b.right_shift,
                    terminal.profile.round_count(),
                    terminal.profile.final_poly_len,
                    terminal.profile.round_count(),
                    0,
                )
                .map_err(|_| FiniteWarpV3TerminalDecideError::UnsupportedTwoCosetProfile)?,
            );
            let two_coset_opened_air = Arc::new(
                FiniteWarpV3TwoCosetTerminalWhirOpenedAir::new(
                    &two_coset_profile,
                    b.whir_query,
                    b.whir_folding,
                    b.leaf_value,
                    b.opening_leaf,
                    b.merkle_root,
                )
                .map_err(|_| FiniteWarpV3TerminalDecideError::UnsupportedTwoCosetProfile)?,
            );
            let two_coset_folding_air = Arc::new(
                FiniteWarpV3TwoCosetTerminalWhirFoldingAir::new(
                    &two_coset_profile,
                    b.whir_alpha,
                    b.whir_folding,
                )
                .map_err(|_| FiniteWarpV3TerminalDecideError::UnsupportedTwoCosetProfile)?,
            );
            let two_coset_adjoint_q_air = Arc::new(FiniteWarpV3TwoCosetRsAdjointQAir {
                round_bus: b.adjoint_round,
                accumulator_value_bus: b.accumulator_value,
                q_bus: b.adjoint_q,
                log_message_len: terminal.profile.log_message_len,
            });
            let two_coset_adjoint_y_air = Arc::new(FiniteWarpV3TwoCosetRsAdjointYAir::new(
                b.adjoint_round,
                b.adjoint_y,
                terminal.profile.log_message_len,
            )?);
            let two_coset_adjoint_selector_air =
                Arc::new(FiniteWarpV3TwoCosetRsAdjointSelectorAir {
                    point_bus: b.whir_point,
                    q_bus: b.adjoint_q,
                    y_bus: b.adjoint_y,
                    claim_bus: b.adjoint_claim,
                    value_bus: b.adjoint_value,
                    log_message_len: terminal.profile.log_message_len,
                });
            let instance_air = Arc::new(FiniteWarpV3TerminalInstanceAir {
                layout: layout.clone(),
                terminal_instance_bus: terminal.instance_bus(),
                digest_element_bus: buses.digest_element,
            });
            let hash_air = Arc::new(NativeAccumulatorHashAir {
                state: 0,
                allow_empty: false,
                alpha_len,
                beta_len,
                digest_element_bus: buses.digest_element,
                algebraic_digest_bus: buses.algebraic_digest,
                permute_bus: poseidon_permute_bus,
                compress_bus: poseidon_compress_bus,
            });
            let receipt_air = Arc::new(FiniteWarpV3TerminalDecideReceiptAir {
                binding,
                relation_digest,
                alpha_len,
                beta_len,
                receipt_bus: buses.receipt,
                terminal_binding_bus: terminal.binding_bus(),
                terminal_whir_root_bus: terminal.whir_root_bus(),
                algebraic_digest_bus: buses.algebraic_digest,
                compress_bus: poseidon_compress_bus,
                terminal_accumulator_link_bus: buses.terminal_accumulator_link,
            });
            Ok(Self {
                terminal,
                two_coset_profile,
                resumed_transcript_air,
                transcript_seam_air,
                two_coset_prefix_air,
                two_coset_query_air,
                two_coset_opened_air,
                two_coset_folding_air,
                two_coset_adjoint_q_air,
                two_coset_adjoint_y_air,
                two_coset_adjoint_selector_air,
                transcript_index,
                two_coset_prefix_index,
                two_coset_query_index,
                two_coset_opened_index,
                two_coset_folding_index,
                leaf_hash_index,
                merkle_index,
                two_coset_adjoint_q_index,
                two_coset_adjoint_y_index,
                two_coset_adjoint_selector_index,
                exp_bits_len_index,
                instance_air,
                hash_air,
                receipt_air,
                poseidon_owner: Poseidon2BusOwner {
                    permute_bus: poseidon_permute_bus,
                    compress_bus: poseidon_compress_bus,
                },
                layout,
            })
        }
    }

    #[must_use]
    pub fn terminal(&self) -> &FixedMultiAirTerminalCircuit {
        self.terminal.as_ref()
    }

    #[must_use]
    pub fn two_coset_profile(&self) -> &FiniteWarpV3TwoCosetTerminalProfile {
        &self.two_coset_profile
    }

    #[must_use]
    pub const fn poseidon2_bus_owner(&self) -> Poseidon2BusOwner {
        self.poseidon_owner
    }

    /// AIR order matched exactly by [`Self::generate_traces`].  The enclosing
    /// wrapper must append one physical shared-Poseidon AIR for this owner's
    /// requests; this method intentionally does not create a second table.
    #[must_use]
    pub fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        let mut airs = self.terminal.airs::<SC>();
        // These slots are found and validated once against the setup-fixed
        // legacy inventory.  Production finite-v3 keys commit to the complete
        // coefficient-native replacement set returned here.  Keeping this
        // atomic prevents a prefix/query-only adapter from becoming callable.
        airs[self.transcript_index] = self.resumed_transcript_air.clone() as AirRef<SC>;
        airs[self.two_coset_prefix_index] = self.two_coset_prefix_air.clone() as AirRef<SC>;
        airs[self.two_coset_query_index] = self.two_coset_query_air.clone() as AirRef<SC>;
        airs[self.two_coset_opened_index] = self.two_coset_opened_air.clone() as AirRef<SC>;
        airs[self.two_coset_folding_index] = self.two_coset_folding_air.clone() as AirRef<SC>;
        airs[self.two_coset_adjoint_q_index] = self.two_coset_adjoint_q_air.clone() as AirRef<SC>;
        airs[self.two_coset_adjoint_y_index] = self.two_coset_adjoint_y_air.clone() as AirRef<SC>;
        airs[self.two_coset_adjoint_selector_index] =
            self.two_coset_adjoint_selector_air.clone() as AirRef<SC>;
        airs.extend([
            self.transcript_seam_air.clone() as AirRef<SC>,
            self.instance_air.clone() as AirRef<SC>,
            self.hash_air.clone() as AirRef<SC>,
            self.receipt_air.clone() as AirRef<SC>,
        ]);
        airs
    }

    /// Complete a production terminal packet with the exact finite-v3
    /// coefficient-two-coset trace set and the final receipt traces.
    ///
    /// `terminal` is produced by the surrounding production scalar-terminal
    /// owner.  This method intentionally does not call the legacy
    /// vector-layout aggregate generator: doing so and overwriting its prefix
    /// later would still make witness generation depend on the wrong parser.
    /// Every trace in the packet remains constrained by [`Self::airs`].
    pub fn generate_traces(
        &self,
        witness: FixedMultiAirTerminalTraceWitness<'_>,
        mut terminal: FixedMultiAirTerminalTraceData,
        final_warp_checkpoint: FiniteWarpV3TerminalTranscriptCheckpoint,
    ) -> Result<FiniteWarpV3TerminalDecideTraceData, FiniteWarpV3TerminalDecideError> {
        let instance = witness.instance;
        if instance.alpha.len() != self.layout.alpha_len
            || instance.beta.len() != self.layout.beta_len
        {
            return Err(FiniteWarpV3TerminalDecideError::InvalidAccumulatorShape);
        }
        let digest = generate_native_accumulator_digest_traces(0, instance, &self.layout)
            .ok_or(FiniteWarpV3TerminalDecideError::InvalidAccumulatorShape)?;
        let receipt = FiniteWarpV3TerminalReceiptRecord {
            final_accumulator_digest: digest.instance_digest,
            final_accumulator_root: instance.rt,
        };
        let (resumed_transcript, resumed_permutation_inputs) =
            generate_finite_warp_v3_resumed_terminal_transcript_trace(
                witness.transcript,
                final_warp_checkpoint,
                None,
            )
            .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let transcript_seam = finite_warp_v3_terminal_transcript_seam_trace(
            final_warp_checkpoint,
            instance.rt,
            digest.instance_digest,
        );
        let two_coset_prefix = generate_finite_warp_v3_two_coset_whir_prefix_trace(
            &self.two_coset_profile,
            witness.descriptor,
            witness.instance,
            witness.transcript,
            witness.whir_verification.transcript_start.operations,
        )
        .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let two_coset_query = generate_finite_warp_v3_two_coset_terminal_whir_query_trace(
            &self.two_coset_profile,
            witness.whir_verification,
            witness.transcript,
            self.terminal.profile.round_count(),
            0,
            self.terminal.profile.round_count(),
            self.terminal.profile.final_poly_len,
            None,
        )
        .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let config =
            BabyBearPoseidon2Config::default_from_params(self.terminal.system_params().clone());
        let two_coset_opened = generate_finite_warp_v3_two_coset_terminal_whir_opened_trace(
            &self.two_coset_profile,
            config.hasher(),
            witness.whir_verification,
            witness.transcript,
            self.terminal.profile.round_count(),
            0,
            None,
        )
        .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let two_coset_folding = generate_finite_warp_v3_two_coset_terminal_whir_folding_trace(
            &self.two_coset_profile,
            witness.whir_verification,
            None,
        )
        .map_err(|_| FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let leaf_inputs = two_coset_opened
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
        let mut merkle_records = two_coset_opened
            .inner_merkle
            .iter()
            .map(|(tree, record)| (*tree, record))
            .collect::<Vec<(u32, &BinaryMerkleMultiproofRecord<Digest>)>>();
        merkle_records.extend(
            witness
                .whir_verification
                .rounds
                .iter()
                .enumerate()
                .map(|(round, verification)| (round as u32, &verification.multiproof)),
        );
        let merkle = generate_native_merkle_multiproof_trace(0, &merkle_records, None)
            .ok_or(FiniteWarpV3TerminalDecideError::TerminalTrace)?;
        let two_coset_adjoint_q = generate_two_coset_adjoint_q_trace(
            witness.whir_verification,
            &witness.instance.alpha,
            self.terminal.profile.log_message_len,
        )?;
        let two_coset_adjoint_y = generate_two_coset_adjoint_y_trace(
            &self.two_coset_adjoint_y_air,
            witness.whir_verification,
        )?;
        let two_coset_adjoint_selector = generate_two_coset_adjoint_selector_trace(
            witness.whir_verification,
            &witness.instance.alpha,
            self.terminal.profile.log_message_len,
        )?;
        let exp_bits_len = generate_two_coset_terminal_exp_bits_trace(
            witness.whir_verification,
            witness.transcript,
            self.terminal.profile.initial_log_domain_size,
            self.terminal.profile.folding_pow_bits,
            self.terminal.profile.query_pow_bits,
        )?;

        let old_transcript_inputs =
            transcript_permutation_inputs(&terminal.traces[self.transcript_index].common_main)?;
        let old_leaf_inputs =
            leaf_permutation_inputs(&terminal.traces[self.leaf_hash_index].common_main)?;
        let expected_old_permutations = old_transcript_inputs
            .into_iter()
            .chain(old_leaf_inputs)
            .collect::<Vec<_>>();
        if terminal.poseidon2_permute_inputs != expected_old_permutations {
            return Err(FiniteWarpV3TerminalDecideError::PoseidonSuffix);
        }
        let old_compressions =
            merkle_compression_inputs(&terminal.traces[self.merkle_index].common_main)?;
        if terminal.poseidon2_compress_inputs != old_compressions {
            return Err(FiniteWarpV3TerminalDecideError::PoseidonSuffix);
        }
        terminal.poseidon2_permute_inputs = resumed_permutation_inputs;
        terminal
            .poseidon2_permute_inputs
            .extend(leaf_hash.permutation_inputs.iter().copied());
        terminal.poseidon2_compress_inputs = merkle_compression_inputs(&merkle)?;

        self.install_two_coset_terminal_traces(
            &mut terminal,
            resumed_transcript,
            two_coset_prefix,
            two_coset_query,
            two_coset_opened.matrix,
            two_coset_folding,
            leaf_hash.matrix,
            merkle,
            two_coset_adjoint_q,
            two_coset_adjoint_y,
            two_coset_adjoint_selector,
            exp_bits_len,
        )?;
        terminal.traces.extend([
            FixedMultiAirTerminalPartitionedTrace::simple(transcript_seam),
            FixedMultiAirTerminalPartitionedTrace::simple(digest.values),
            FixedMultiAirTerminalPartitionedTrace::simple(digest.hash),
            FixedMultiAirTerminalPartitionedTrace::simple(digest.root),
        ]);
        terminal
            .poseidon2_permute_inputs
            .extend(digest.poseidon2_permute_inputs);
        terminal
            .poseidon2_compress_inputs
            .extend(digest.poseidon2_compress_inputs);
        let expected = self.airs::<BabyBearPoseidon2Config>().len();
        if terminal.traces.len() != expected {
            return Err(FiniteWarpV3TerminalDecideError::AirTraceCount {
                expected,
                actual: terminal.traces.len(),
            });
        }
        Ok(FiniteWarpV3TerminalDecideTraceData {
            traces: terminal.traces,
            poseidon2_permute_inputs: terminal.poseidon2_permute_inputs,
            poseidon2_compress_inputs: terminal.poseidon2_compress_inputs,
            receipt,
        })
    }

    /// Atomically replace every layout-dependent terminal matrix.  This is
    /// private so no caller can construct the retired prefix/query-only packet.
    #[allow(clippy::too_many_arguments)]
    fn install_two_coset_terminal_traces(
        &self,
        terminal: &mut FixedMultiAirTerminalTraceData,
        transcript: RowMajorMatrix<F>,
        prefix: RowMajorMatrix<F>,
        query: RowMajorMatrix<F>,
        opened: RowMajorMatrix<F>,
        folding: RowMajorMatrix<F>,
        leaf_hash: RowMajorMatrix<F>,
        merkle: RowMajorMatrix<F>,
        adjoint_q: RowMajorMatrix<F>,
        adjoint_y: RowMajorMatrix<F>,
        adjoint_selector: RowMajorMatrix<F>,
        exp_bits_len: RowMajorMatrix<F>,
    ) -> Result<(), FiniteWarpV3TerminalDecideError> {
        let terminal_airs = self.terminal.airs::<BabyBearPoseidon2Config>();
        if terminal.traces.len() != terminal_airs.len()
            || transcript.width()
                != <TranscriptAir as BaseAir<F>>::width(&self.resumed_transcript_air)
            || prefix.width() != self.two_coset_prefix_air.width()
            || query.width() != self.two_coset_query_air.width()
            || opened.width() != self.two_coset_opened_air.width()
            || folding.width() != self.two_coset_folding_air.width()
            || leaf_hash.width() != terminal.traces[self.leaf_hash_index].common_main.width()
            || merkle.width() != terminal.traces[self.merkle_index].common_main.width()
            || adjoint_q.width() != self.two_coset_adjoint_q_air.width()
            || adjoint_y.width() != self.two_coset_adjoint_y_air.width()
            || adjoint_selector.width() != self.two_coset_adjoint_selector_air.width()
            || exp_bits_len.width() != terminal.traces[self.exp_bits_len_index].common_main.width()
        {
            return Err(FiniteWarpV3TerminalDecideError::AirTraceCount {
                expected: terminal_airs.len(),
                actual: terminal.traces.len(),
            });
        }
        terminal.traces[self.transcript_index] =
            FixedMultiAirTerminalPartitionedTrace::simple(transcript);
        terminal.traces[self.two_coset_prefix_index] =
            FixedMultiAirTerminalPartitionedTrace::simple(prefix);
        terminal.traces[self.two_coset_query_index] =
            FixedMultiAirTerminalPartitionedTrace::simple(query);
        terminal.traces[self.two_coset_opened_index] =
            FixedMultiAirTerminalPartitionedTrace::simple(opened);
        terminal.traces[self.two_coset_folding_index] =
            FixedMultiAirTerminalPartitionedTrace::simple(folding);
        terminal.traces[self.leaf_hash_index] =
            FixedMultiAirTerminalPartitionedTrace::simple(leaf_hash);
        terminal.traces[self.merkle_index] = FixedMultiAirTerminalPartitionedTrace::simple(merkle);
        terminal.traces[self.two_coset_adjoint_q_index] =
            FixedMultiAirTerminalPartitionedTrace::simple(adjoint_q);
        terminal.traces[self.two_coset_adjoint_y_index] =
            FixedMultiAirTerminalPartitionedTrace::simple(adjoint_y);
        terminal.traces[self.two_coset_adjoint_selector_index] =
            FixedMultiAirTerminalPartitionedTrace::simple(adjoint_selector);
        terminal.traces[self.exp_bits_len_index] =
            FixedMultiAirTerminalPartitionedTrace::simple(exp_bits_len);
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteWarpV3TerminalReceiptRecord {
    pub final_accumulator_digest: Digest,
    pub final_accumulator_root: Digest,
}

/// Backend-neutral trace packet.  CPU and CUDA adapters transport these
/// matrices exactly as they already do for `FixedMultiAirTerminalTraceData`.
#[derive(Clone, Debug)]
pub struct FiniteWarpV3TerminalDecideTraceData {
    pub traces: Vec<FixedMultiAirTerminalPartitionedTrace>,
    pub poseidon2_permute_inputs: Vec<[F; 16]>,
    pub poseidon2_compress_inputs: Vec<[F; 16]>,
    pub receipt: FiniteWarpV3TerminalReceiptRecord,
}

/// Helper used by focused tests and SDK adapters to build the receipt matrix
/// without reimplementing the canonical accumulator digest.
pub fn finite_warp_v3_terminal_receipt_trace(
    instance: &AccumulatorInstance<EF, Digest>,
) -> Result<(RowMajorMatrix<F>, FiniteWarpV3TerminalReceiptRecord), FiniteWarpV3TerminalDecideError>
{
    if instance.alpha.is_empty() || instance.beta.is_empty() {
        return Err(FiniteWarpV3TerminalDecideError::InvalidAccumulatorShape);
    }
    let layout = NativePrivateAccumulatorLayout::new(0, instance.alpha.len(), instance.beta.len());
    let digest = generate_native_accumulator_digest_traces(0, instance, &layout)
        .ok_or(FiniteWarpV3TerminalDecideError::InvalidAccumulatorShape)?;
    Ok((
        digest.root,
        FiniteWarpV3TerminalReceiptRecord {
            final_accumulator_digest: digest.instance_digest,
            final_accumulator_root: instance.rt,
        },
    ))
}

#[cfg(test)]
#[path = "terminal_decide_tests.rs"]
mod terminal_decide_tests;

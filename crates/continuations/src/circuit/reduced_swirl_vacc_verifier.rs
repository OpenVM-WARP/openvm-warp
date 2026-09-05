//! Verifier AIR composition for one reduced-SWIRL WARP VACC step.
//!
//! This module consumes the verifier-derived [`WarpVaccStepVerification`] and
//! exact [`TranscriptLog`].  It authenticates fresh and prior codeword
//! openings, certifies both sumchecks and their Fiat--Shamir challenges, and
//! publishes the exact typed statements consumed by the transition aggregate.
//! PESAT satisfaction is deliberately absent from this transition verifier:
//! terminal Decide evaluates the fixed reduced-SWIRL relation.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit::{
    bus::{TranscriptBus, TranscriptBusMessage},
    native_warp::*,
    primitives::{
        bus::{ExpBitsLenBus, ExpBitsLenMessage, RightShiftBus, RightShiftMessage},
        exp_bits_len::{ExpBitsLenAir, ExpBitsLenCols, ExpBitsLenCpuTraceGenerator},
    },
    system::BusInventory,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    native_warp::NativeWarpFamilyParams,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    warp_accum::{
        twin_degree, BinaryMerkleMultiproofRecord, MerkleBatchOpeningProof,
        MerkleBatchOpeningVerification, NativeSumcheckKind, NativeTranscriptPhase,
        NativeTranscriptPhaseSpan, ReducedWarpVaccStepProof, StackedRsBatchOpeningProof,
        StackedRsBatchOpeningVerification, StackedRsFreshCommitment, WarpStepInputKind,
        WarpVaccStepVerification, WARP_CALL_END_TAG,
    },
    warp_pesat::{AccumulatorInstance, FreshPesatClaim},
    AirRef, AnyAir, StarkProtocolConfig, SystemParams, TranscriptCheckpoint, TranscriptEvent,
    TranscriptEventKind, TranscriptLog,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as NativeSC, Digest, CHUNK, DIGEST_SIZE, D_EF, EF, F,
};

use crate::circuit::native_warp_accumulator::{
    generate_native_accumulator_digest_traces, NativeAccumulatorBindingMode,
    NativeAccumulatorDigestTraces, NativeAccumulatorHashAir, NativeAccumulatorHashCols,
    NativeAccumulatorRootDigestCols, NativeAccumulatorValueCols, NativePrivateAccumulatorLayout,
};

const MAX_RAW_MESSAGE_POINT_LEN_: usize = 32;
const MAX_FRESH_BETA_LEN_: usize = 160;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct TranscriptCheckpointRecord {
    operation_index: u32,
    sample_count: u8,
    state: [F; POSEIDON2_WIDTH],
}

pub type ReducedSwirlVaccProof = ReducedWarpVaccStepProof<
    EF,
    Digest,
    StackedRsBatchOpeningProof<F, Digest>,
    MerkleBatchOpeningProof<EF, Digest>,
    StackedRsFreshCommitment<EF, Digest>,
>;

pub type ReducedSwirlVaccVerification = WarpVaccStepVerification<
    EF,
    Digest,
    StackedRsBatchOpeningVerification<F, EF, Digest>,
    MerkleBatchOpeningVerification<EF, Digest>,
>;

/// One genuine reduced native transition qualified by its authoritative
/// transcript interval. `proof_idx` is the canonical chain step; callers do
/// not provide any verifier-success bit.
pub struct ReducedSwirlVaccVerifierRecord<'a> {
    /// Canonical block-wide WARP call index.  This value is serialized in the
    /// native transcript and therefore must never be renumbered by a
    /// recursive wrapper.
    pub proof_idx: usize,
    /// Dense transcript/AIR namespace inside the current bounded wrapper
    /// leaf.  Whole-block batches set this equal to `proof_idx`; recursive
    /// transition leaves restart it at zero without changing any transcript
    /// value.
    pub local_proof_idx: usize,
    /// Whether this is the final call of the complete block schedule.  Only
    /// that call owns the manifest-footer and terminal-Decide transcript
    /// suffix; the final call of an intermediate physical leaf does not.
    pub is_final_call: bool,
    pub proof: &'a ReducedSwirlVaccProof,
    pub verification: &'a ReducedSwirlVaccVerification,
    pub transcript: &'a TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    pub prior: Option<&'a AccumulatorInstance<EF, Digest>>,
    pub batch_start_tidx: usize,
    pub vacc_start_tidx: usize,
    pub vacc_end_tidx: usize,
    /// First transcript operation of every canonical source commitment
    /// descriptor, in source order.
    pub commitment_tidxs: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReducedSwirlVaccVerifierError {
    InvalidProfile,
    RecordShape(&'static str),
    Transcript(&'static str),
    Merkle(&'static str),
    Algebra(&'static str),
    PriorChain,
    AirTraceCount,
}

fn vacc_cursor_role_lengths_exact(
    profile: &NativeStandardVaccShapeProfile,
    input_arity: usize,
    fresh_count: usize,
    include_prior: bool,
) -> [usize; VACC_ROLE_COUNT] {
    let mut lengths = [0usize; VACC_ROLE_COUNT];
    // Each stacked-source descriptor observes six domain/layout values,
    // source ordinal, two code dimensions, and the root limbs.
    lengths[VACC_ROLE_FRESH_ROOT] = fresh_count * (9 + DIGEST_SIZE);
    lengths[VACC_ROLE_FRESH_ALPHA] = fresh_count * profile.log_codeword_len;
    lengths[VACC_ROLE_FRESH_MU] = fresh_count;
    lengths[VACC_ROLE_FRESH_BETA_TAIL] = fresh_count * (profile.beta_len - profile.log_constraints);
    if include_prior {
        lengths[VACC_ROLE_PRIOR_ROOT] = DIGEST_SIZE;
        lengths[VACC_ROLE_PRIOR_ALPHA] = profile.log_codeword_len;
        lengths[VACC_ROLE_PRIOR_MU] = 1;
        lengths[VACC_ROLE_PRIOR_BETA] = profile.beta_len;
        lengths[VACC_ROLE_PRIOR_ETA] = 1;
    }
    lengths[VACC_ROLE_FRESH_TAU] = fresh_count * profile.log_constraints;
    lengths[VACC_ROLE_OMEGA] = 1;
    lengths[VACC_ROLE_SELECTOR] = input_arity.ilog2() as usize;
    lengths[VACC_ROLE_TWIN_SUMCHECK] = (input_arity.ilog2() as usize)
        * (twin_degree(
            profile.log_codeword_len,
            profile.log_constraints,
            profile.max_degree,
        ) + 2);
    lengths[VACC_ROLE_OUTPUT_ROOT] = DIGEST_SIZE;
    lengths[VACC_ROLE_NU] = 1;
    lengths[VACC_ROLE_ETA] = 1;
    lengths[VACC_ROLE_OOD_POINT] = profile.num_ood * profile.log_codeword_len;
    lengths[VACC_ROLE_OOD_ANSWER] = profile.num_ood;
    lengths[VACC_ROLE_SHIFT] = profile.num_shift_queries;
    lengths[VACC_ROLE_XI] = profile.batching_arity.ilog2() as usize;
    lengths[VACC_ROLE_BATCHING_SUMCHECK] = profile.log_codeword_len * 4;
    lengths[VACC_ROLE_MU] = 1;
    lengths
}

fn next_nonempty_vacc_roles(lengths: &[usize; VACC_ROLE_COUNT]) -> [usize; VACC_ROLE_COUNT] {
    core::array::from_fn(|role| {
        ((role + 1)..VACC_ROLE_COUNT)
            .find(|&candidate| lengths[candidate] != 0)
            .unwrap_or(VACC_ROLE_COUNT)
    })
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ReducedSwirlVaccTranscriptCursorCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub tidx: T,
    pub role: T,
    pub ordinal: T,
    pub role_flags: [T; VACC_ROLE_COUNT],
    pub is_ext: T,
    pub is_sample: T,
    pub is_first_ordinal: T,
    pub is_last_ordinal: T,
    pub ordinal_inverse: T,
    pub last_ordinal_inverse: T,
    pub sumcheck_round: T,
    pub sumcheck_step: T,
    pub is_last_sumcheck_step: T,
    pub sumcheck_last_step_inverse: T,
    /// Prepared products keep the exact cursor state machine within the
    /// recursive degree-four envelope.  Every helper is constrained below;
    /// none is a host acceptance bit.
    pub is_sumcheck: T,
    pub sumcheck_step_continues: T,
    pub continues_proof: T,
    pub continues_sumcheck_phase: T,
    pub starts_proof: T,
    pub value: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(ReducedSwirlVaccTranscriptCursorCols<u8>)]
pub struct ReducedSwirlVaccTranscriptCursorAir {
    pub transcript_bus: openvm_recursion_circuit::bus::TranscriptBus,
    pub semantic_bus: openvm_recursion_circuit::bus::TranscriptBus,
    pub phase_cursor_bus: NativeVaccPhaseCursorBus,
    pub role_bus: NativeVaccTranscriptRoleBus,
    pub lengths: [usize; VACC_ROLE_COUNT],
    pub next_roles: [usize; VACC_ROLE_COUNT],
    pub twin_event_count: usize,
    /// Reduced-SWIRL has setup-fixed arity but a runtime-constrained active
    /// fresh prefix and variable original-root tuples. In this mode role ends
    /// are proved by typed consumers and the cursor transition itself rather
    /// than by verifier-key constant role lengths.
    pub dynamic_role_lengths: bool,
    /// Bootstrap omits the entire prior-accumulator role interval; a
    /// continuation must include it because its typed prior consumers are
    /// active. This permits exactly that one protocol-prescribed skip.
    pub dynamic_optional_prior: bool,
    /// Standalone fresh commitments have dedicated consumers on `role_bus`.
    /// The reduced stacked-RS path instead authenticates every descriptor
    /// field and original root directly at its exact transcript index on
    /// `semantic_bus`; emitting a second role lookup there would be an
    /// unconsumed duplicate statement.
    pub emit_fresh_root_role: bool,
}

impl openvm_stark_backend::BaseAirWithPublicValues<F> for ReducedSwirlVaccTranscriptCursorAir {}
impl openvm_stark_backend::PartitionedBaseAir<F> for ReducedSwirlVaccTranscriptCursorAir {}
impl BaseAir<F> for ReducedSwirlVaccTranscriptCursorAir {
    fn width(&self) -> usize {
        ReducedSwirlVaccTranscriptCursorCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for ReducedSwirlVaccTranscriptCursorAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("direct VACC cursor row");
        let next_row = main.row_slice(1).expect("direct VACC cursor next row");
        let local: &ReducedSwirlVaccTranscriptCursorCols<AB::Var> = (*local_row).borrow();
        let next: &ReducedSwirlVaccTranscriptCursorCols<AB::Var> = (*next_row).borrow();

        for flag in [
            local.active,
            local.is_ext,
            local.is_sample,
            local.is_first_ordinal,
            local.is_last_ordinal,
            local.is_last_sumcheck_step,
            local.is_sumcheck,
            local.sumcheck_step_continues,
            local.continues_proof,
            local.continues_sumcheck_phase,
            local.starts_proof,
        ] {
            builder.assert_bool(flag);
        }
        for flag in local.role_flags {
            builder.assert_bool(flag);
        }
        builder.assert_eq(
            local.active,
            local
                .role_flags
                .into_iter()
                .map(AB::Expr::from)
                .sum::<AB::Expr>(),
        );
        builder.when_first_row().assert_one(local.active);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);

        let role = local
            .role_flags
            .into_iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (index, flag)| {
                acc + flag * AB::Expr::from_usize(index)
            });
        builder.when(local.active).assert_eq(local.role, role);
        for (role, &length) in self.lengths.iter().enumerate() {
            if length == 0 {
                builder.assert_zero(local.role_flags[role]);
            }
        }
        let nominal_len = local
            .role_flags
            .into_iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (role, flag)| {
                acc + flag * AB::Expr::from_usize(self.lengths[role])
            });
        let distance = nominal_len - AB::Expr::ONE - local.ordinal;
        builder
            .when(local.active * local.is_first_ordinal)
            .assert_zero(local.ordinal);
        builder
            .when(local.active * (AB::Expr::ONE - local.is_first_ordinal))
            .assert_one(local.ordinal * local.ordinal_inverse);
        if !self.dynamic_role_lengths {
            builder
                .when(local.active * local.is_last_ordinal)
                .assert_zero(distance.clone());
            builder
                .when(local.active * (AB::Expr::ONE - local.is_last_ordinal))
                .assert_one(distance * local.last_ordinal_inverse);
        }

        let is_shift = local.role_flags[VACC_ROLE_SHIFT];
        builder
            .when(local.active)
            .assert_eq(local.is_ext, AB::Expr::ONE - is_shift);
        for limb in &local.value[1..] {
            builder.when(local.active * is_shift).assert_zero(*limb);
        }

        let twin_sumcheck = local.role_flags[VACC_ROLE_TWIN_SUMCHECK];
        let batching_sumcheck = local.role_flags[VACC_ROLE_BATCHING_SUMCHECK];
        let is_sumcheck = twin_sumcheck + batching_sumcheck;
        builder.assert_eq(local.is_sumcheck, is_sumcheck.clone());
        let event_count = twin_sumcheck * AB::Expr::from_usize(self.twin_event_count)
            + batching_sumcheck * AB::Expr::from_usize(4);
        builder.when(local.active * is_sumcheck.clone()).assert_eq(
            local.ordinal,
            local.sumcheck_round * event_count.clone() + local.sumcheck_step,
        );
        builder
            .when(local.active * is_sumcheck.clone() * local.is_first_ordinal)
            .assert_zero(local.sumcheck_round);
        builder
            .when(local.active * is_sumcheck.clone() * local.is_first_ordinal)
            .assert_zero(local.sumcheck_step);
        let step_distance = event_count - AB::Expr::ONE - local.sumcheck_step;
        builder
            .when(local.active * is_sumcheck.clone() * local.is_last_sumcheck_step)
            .assert_zero(step_distance.clone());
        builder.assert_eq(
            local.sumcheck_step_continues,
            local.active * local.is_sumcheck * (AB::Expr::ONE - local.is_last_sumcheck_step),
        );
        builder
            .when(local.sumcheck_step_continues)
            .assert_one(step_distance * local.sumcheck_last_step_inverse);
        for witness in [
            local.sumcheck_round,
            local.sumcheck_step,
            local.sumcheck_last_step_inverse,
        ] {
            builder
                .when(local.active * (AB::Expr::ONE - is_sumcheck.clone()))
                .assert_zero(witness);
        }
        builder
            .when(local.active * (AB::Expr::ONE - is_sumcheck.clone()))
            .assert_zero(local.is_last_sumcheck_step);
        let always_sample = local.role_flags[VACC_ROLE_FRESH_TAU]
            + local.role_flags[VACC_ROLE_OMEGA]
            + local.role_flags[VACC_ROLE_SELECTOR]
            + local.role_flags[VACC_ROLE_OOD_POINT]
            + local.role_flags[VACC_ROLE_SHIFT]
            + local.role_flags[VACC_ROLE_XI];
        builder.when(local.active).assert_eq(
            local.is_sample,
            always_sample + is_sumcheck.clone() * local.is_last_sumcheck_step,
        );

        let width = AB::Expr::ONE + local.is_ext * AB::Expr::from_usize(D_EF - 1);
        let phase_end = AB::Expr::from(local.is_last_ordinal);
        let proof_end = local.role_flags[VACC_ROLE_MU] * phase_end.clone();
        let starts_proof = local.role_flags[VACC_ROLE_FRESH_ROOT] * local.is_first_ordinal;
        builder.assert_eq(local.starts_proof, local.active * starts_proof.clone());
        builder.assert_eq(local.continues_proof, local.active - local.starts_proof);
        builder.assert_eq(
            local.continues_sumcheck_phase,
            local.active * local.is_sumcheck * (AB::Expr::ONE - local.is_first_ordinal),
        );
        self.phase_cursor_bus.receive(
            builder,
            NativeVaccPhaseCursorMessage {
                proof_idx: local.proof_idx.into(),
                boundary: AB::Expr::ONE,
                tidx: local.tidx.into(),
            },
            local.active * starts_proof,
        );
        self.phase_cursor_bus.send(
            builder,
            NativeVaccPhaseCursorMessage {
                proof_idx: local.proof_idx.into(),
                boundary: AB::Expr::from_usize(2),
                tidx: local.tidx + width.clone(),
            },
            local.active * proof_end.clone(),
        );

        let expected_next_role = local
            .role_flags
            .into_iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (role, flag)| {
                acc + flag * AB::Expr::from_usize(self.next_roles[role])
            });
        let same_phase = AB::Expr::ONE - phase_end.clone();
        {
            let mut transition = builder.when_transition();
            let mut continuing = transition.when(next.continues_proof);
            continuing.assert_eq(next.proof_idx, local.proof_idx);
            continuing.assert_eq(next.tidx, local.tidx + width.clone());
            if self.dynamic_optional_prior {
                continuing
                    .when(same_phase.clone())
                    .assert_eq(next.role, local.role);
            } else {
                continuing.assert_eq(
                    next.role,
                    same_phase.clone() * local.role + phase_end.clone() * expected_next_role,
                );
            }
            continuing.assert_eq(
                next.ordinal,
                same_phase.clone() * (local.ordinal + AB::Expr::ONE),
            );
        }
        if self.dynamic_optional_prior {
            let after_prior = self.next_roles[VACC_ROLE_PRIOR_ETA];
            let optional_skip =
                local.role_flags[VACC_ROLE_FRESH_BETA_TAIL] * next.role_flags[after_prior];
            let ordinary_advance = local.role_flags.into_iter().enumerate().fold(
                AB::Expr::ZERO,
                |sum, (role, flag)| {
                    let next_role = self.next_roles[role];
                    if next_role < VACC_ROLE_COUNT {
                        sum + flag * next.role_flags[next_role]
                    } else {
                        // `VACC_ROLE_COUNT` is the sentinel for the terminal
                        // role. The surrounding constraint is disabled when
                        // no row continues this proof, but Rust indexing is
                        // eager, so represent the sentinel contribution as
                        // zero rather than indexing one past the flag array.
                        sum
                    }
                },
            );
            builder
                .when_transition()
                .when(next.continues_proof * phase_end.clone())
                .assert_one(ordinary_advance + optional_skip);
        }
        {
            let mut transition = builder.when_transition();
            let mut sumcheck_transition = transition.when(next.continues_sumcheck_phase);
            sumcheck_transition.assert_eq(
                next.sumcheck_round,
                local.sumcheck_round + local.is_last_sumcheck_step,
            );
            sumcheck_transition.assert_eq(
                next.sumcheck_step,
                (AB::Expr::ONE - local.is_last_sumcheck_step) * (local.sumcheck_step + AB::F::ONE),
            );
        }
        {
            let mut transition = builder.when_transition();
            let mut next_proof = transition.when(next.starts_proof);
            next_proof.assert_one(proof_end.clone());
            next_proof.assert_eq(next.proof_idx, local.proof_idx + AB::F::ONE);
            next_proof.assert_eq(next.role, AB::Expr::from_usize(VACC_ROLE_FRESH_ROOT));
            next_proof.assert_zero(next.ordinal);
        }
        builder
            .when_transition()
            .when(local.active - next.active)
            .assert_one(proof_end.clone());
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(proof_end);

        let message = NativeVaccTranscriptRoleMessage {
            proof_idx: local.proof_idx.into(),
            role: local.role.into(),
            ordinal: local.ordinal.into(),
            tidx: local.tidx.into(),
            value: local.value.map(Into::into),
            is_ext: local.is_ext.into(),
            is_sample: local.is_sample.into(),
        };
        self.role_bus.send(
            builder,
            message,
            local.active
                * (AB::Expr::ONE
                    - local.role_flags[VACC_ROLE_FRESH_ROOT]
                        * AB::Expr::from_bool(!self.emit_fresh_root_role)),
        );
        for (limb, value) in local.value.into_iter().enumerate() {
            let enabled = local.active
                * (local.is_ext + (AB::Expr::ONE - local.is_ext) * AB::Expr::from_bool(limb == 0));
            let transcript_message = TranscriptBusMessage {
                tidx: local.tidx + AB::Expr::from_usize(limb),
                value: value.into(),
                is_sample: local.is_sample.into(),
            };
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                transcript_message.clone(),
                enabled.clone(),
            );
            self.semantic_bus
                .send(builder, local.proof_idx, transcript_message, enabled);
        }
    }
}

pub struct ReducedSwirlVaccVectorCoordinateAir {
    pub inner: NativeVectorCoordinateAir,
    pub role_bus: NativeVaccTranscriptRoleBus,
    pub selector_vector: usize,
    pub xi_vector: usize,
}

impl openvm_stark_backend::BaseAirWithPublicValues<F> for ReducedSwirlVaccVectorCoordinateAir {}
impl openvm_stark_backend::PartitionedBaseAir<F> for ReducedSwirlVaccVectorCoordinateAir {}
impl BaseAir<F> for ReducedSwirlVaccVectorCoordinateAir {
    fn width(&self) -> usize {
        NativeVectorCoordinateCols::<F>::width()
    }
}
impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for ReducedSwirlVaccVectorCoordinateAir {
    fn eval(&self, builder: &mut AB) {
        self.inner.eval(builder);
        let main = builder.main();
        let row = main.row_slice(0).expect("direct VACC vector role row");
        let local: &NativeVectorCoordinateCols<AB::Var> = (*row).borrow();
        let is_transcript = local.source_kind[VECTOR_SOURCE_TRANSCRIPT];
        let selector = AB::Expr::from_usize(self.selector_vector);
        let xi = AB::Expr::from_usize(self.xi_vector);
        builder
            .when(local.active * is_transcript)
            .assert_zero((local.vector - selector.clone()) * (local.vector - xi.clone()));
        let is_xi = (local.vector - selector.clone())
            * (AB::F::from_usize(self.xi_vector) - AB::F::from_usize(self.selector_vector))
                .inverse();
        builder
            .when(local.active * is_transcript)
            .assert_bool(is_xi.clone());
        self.role_bus.receive(
            builder,
            NativeVaccTranscriptRoleMessage {
                proof_idx: local.proof_idx.into(),
                role: AB::Expr::from_usize(VACC_ROLE_SELECTOR)
                    + is_xi * AB::Expr::from_usize(VACC_ROLE_XI - VACC_ROLE_SELECTOR),
                ordinal: local.coordinate.into(),
                tidx: local.tidx.into(),
                value: local.value.map(Into::into),
                is_ext: AB::Expr::ONE,
                is_sample: AB::Expr::ONE,
            },
            local.active * is_transcript,
        );
    }
}

pub struct ReducedSwirlVaccTwinOmegaAir {
    pub inner: NativeTwinOmegaAir,
    pub role_bus: NativeVaccTranscriptRoleBus,
}

impl openvm_stark_backend::BaseAirWithPublicValues<F> for ReducedSwirlVaccTwinOmegaAir {}
impl openvm_stark_backend::PartitionedBaseAir<F> for ReducedSwirlVaccTwinOmegaAir {}
impl BaseAir<F> for ReducedSwirlVaccTwinOmegaAir {
    fn width(&self) -> usize {
        NativeTwinOmegaCols::<F>::width()
    }
}

pub struct ReducedSwirlVaccCoefficientSumcheckAir {
    pub inner: NativeCoefficientSumcheckAir,
    pub role_bus: NativeVaccTranscriptRoleBus,
}

impl openvm_stark_backend::BaseAirWithPublicValues<F> for ReducedSwirlVaccCoefficientSumcheckAir {}
impl openvm_stark_backend::PartitionedBaseAir<F> for ReducedSwirlVaccCoefficientSumcheckAir {}
impl BaseAir<F> for ReducedSwirlVaccCoefficientSumcheckAir {
    fn width(&self) -> usize {
        NativeCoefficientSumcheckCols::<F>::width()
    }
}
impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for ReducedSwirlVaccCoefficientSumcheckAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield:
        openvm_stark_backend::p3_field::extension::BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        self.inner.eval(builder);
        let main = builder.main();
        let row = main.row_slice(0).expect("direct VACC sumcheck role row");
        let local: &NativeCoefficientSumcheckCols<AB::Var> = (*row).borrow();
        let degree = AB::Expr::from_usize(self.inner.twin_degree)
            + local.kind
                * (AB::Expr::from_usize(self.inner.batching_degree)
                    - AB::Expr::from_usize(self.inner.twin_degree));
        let role = AB::Expr::from_usize(VACC_ROLE_TWIN_SUMCHECK)
            + local.kind
                * AB::Expr::from_usize(VACC_ROLE_BATCHING_SUMCHECK - VACC_ROLE_TWIN_SUMCHECK);
        let event_count = degree.clone() + AB::Expr::from_usize(2);
        self.role_bus.receive(
            builder,
            NativeVaccTranscriptRoleMessage {
                proof_idx: local.proof_idx.into(),
                role: role.clone(),
                ordinal: local.round * event_count.clone() + local.coefficient_index,
                tidx: local.tidx + local.coefficient_index * AB::Expr::from_usize(D_EF),
                value: local.coefficient.map(Into::into),
                is_ext: AB::Expr::ONE,
                is_sample: AB::Expr::ZERO,
            },
            local.active,
        );
        self.role_bus.receive(
            builder,
            NativeVaccTranscriptRoleMessage {
                proof_idx: local.proof_idx.into(),
                role,
                ordinal: local.round * event_count + degree.clone() + AB::Expr::ONE,
                tidx: local.tidx + (degree + AB::Expr::ONE) * AB::Expr::from_usize(D_EF),
                value: local.challenge.map(Into::into),
                is_ext: AB::Expr::ONE,
                is_sample: AB::Expr::ONE,
            },
            local.active * local.is_last_coefficient,
        );
    }
}
impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for ReducedSwirlVaccTwinOmegaAir {
    fn eval(&self, builder: &mut AB) {
        self.inner.eval(builder);
        let main = builder.main();
        let row = main.row_slice(0).expect("direct VACC omega role row");
        let local: &NativeTwinOmegaCols<AB::Var> = (*row).borrow();
        self.role_bus.receive(
            builder,
            NativeVaccTranscriptRoleMessage {
                proof_idx: local.proof_idx.into(),
                role: AB::Expr::from_usize(VACC_ROLE_OMEGA),
                ordinal: AB::Expr::ZERO,
                tidx: local.tidx.into(),
                value: local.value.map(Into::into),
                is_ext: AB::Expr::ONE,
                is_sample: AB::Expr::ONE,
            },
            local.active,
        );
    }
}

pub struct ReducedSwirlVaccTwinFinalAir {
    pub inner: NativeTwinFinalAir,
    pub role_bus: NativeVaccTranscriptRoleBus,
}

impl openvm_stark_backend::BaseAirWithPublicValues<F> for ReducedSwirlVaccTwinFinalAir {}
impl openvm_stark_backend::PartitionedBaseAir<F> for ReducedSwirlVaccTwinFinalAir {}
impl BaseAir<F> for ReducedSwirlVaccTwinFinalAir {
    fn width(&self) -> usize {
        NativeTwinFinalCols::<F>::width()
    }
}
impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for ReducedSwirlVaccTwinFinalAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield:
        openvm_stark_backend::p3_field::extension::BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        self.inner.eval(builder);
        let main = builder.main();
        let row = main.row_slice(0).expect("direct VACC twin-final role row");
        let local: &NativeTwinFinalCols<AB::Var> = (*row).borrow();
        for (role, tidx, value) in [
            (VACC_ROLE_NU, local.nu_tidx, local.nu_0),
            (VACC_ROLE_ETA, local.eta_tidx, local.eta),
        ] {
            self.role_bus.receive(
                builder,
                NativeVaccTranscriptRoleMessage {
                    proof_idx: local.proof_idx.into(),
                    role: AB::Expr::from_usize(role),
                    ordinal: AB::Expr::ZERO,
                    tidx: tidx.into(),
                    value: value.map(Into::into),
                    is_ext: AB::Expr::ONE,
                    is_sample: AB::Expr::ZERO,
                },
                local.active,
            );
        }
    }
}

pub struct ReducedSwirlVaccOodClaimAir {
    pub inner: NativeOodClaimAir,
    pub role_bus: NativeVaccTranscriptRoleBus,
    pub dimension: usize,
}

impl openvm_stark_backend::BaseAirWithPublicValues<F> for ReducedSwirlVaccOodClaimAir {}
impl openvm_stark_backend::PartitionedBaseAir<F> for ReducedSwirlVaccOodClaimAir {}
impl BaseAir<F> for ReducedSwirlVaccOodClaimAir {
    fn width(&self) -> usize {
        NativeOodClaimCols::<F>::width()
    }
}
impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for ReducedSwirlVaccOodClaimAir {
    fn eval(&self, builder: &mut AB) {
        self.inner.eval(builder);
        let main = builder.main();
        let row = main.row_slice(0).expect("direct VACC OOD role row");
        let local: &NativeOodClaimCols<AB::Var> = (*row).borrow();
        self.role_bus.receive(
            builder,
            NativeVaccTranscriptRoleMessage {
                proof_idx: local.proof_idx.into(),
                role: AB::Expr::from_usize(VACC_ROLE_OOD_POINT)
                    + local.is_target
                        * AB::Expr::from_usize(VACC_ROLE_OOD_ANSWER - VACC_ROLE_OOD_POINT),
                ordinal: (AB::Expr::ONE - local.is_target)
                    * (local.ood * AB::Expr::from_usize(self.dimension) + local.coordinate)
                    + local.is_target * local.ood,
                tidx: local.tidx.into(),
                value: local.value.map(Into::into),
                is_ext: AB::Expr::ONE,
                is_sample: AB::Expr::ONE - local.is_target,
            },
            local.active,
        );
    }
}

pub struct ReducedSwirlVaccShiftScheduleAir {
    pub transcript_bus: TranscriptBus,
    pub exp_bits_len_bus: ExpBitsLenBus,
    pub right_shift_bus: RightShiftBus,
    pub shift_index_bus: NativeShiftIndexBus,
    pub opening_bus: NativeOpeningClaimBus,
    pub log_codeword_len: usize,
    pub opening_claim_offset: usize,
    pub role_bus: NativeVaccTranscriptRoleBus,
}

fn generate_reduced_swirl_vacc_shift_schedule_trace(
    proof_idx: usize,
    samples: &[F],
    sample_tidx: &[usize],
    indices: &[u32],
    index_lookup_count: u32,
    log_codeword_len: usize,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if samples.is_empty()
        || samples.len() != sample_tidx.len()
        || samples.len() != indices.len()
        || log_codeword_len == 0
        // The backend samples one canonical BabyBear element and masks its
        // low bits. This verifier mirrors that `sample_bits` rule and supports
        // every bit length accepted by the backend:
        // `2^bits < p`, i.e. at most 30 bits for BabyBear. The shared
        // ExpBitsLen/RightShift tables authenticate the full canonical
        // 31-bit decomposition, including the quotient above bit 27.
        || log_codeword_len > 30
    {
        return None;
    }
    let valid_rows = samples.len() * log_codeword_len;
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeShiftScheduleCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    let mask = (1u32 << log_codeword_len) - 1;
    for shift in 0..samples.len() {
        let canonical = samples[shift].as_canonical_u32();
        if canonical & mask != indices[shift] {
            return None;
        }
        let quotient = canonical >> log_codeword_len;
        let mut reconstructed = 0u32;
        for coordinate in 0..log_codeword_len {
            let bit_index = log_codeword_len - 1 - coordinate;
            let bit = (indices[shift] >> bit_index) & 1;
            let before = reconstructed;
            reconstructed += bit << bit_index;
            let row_index = shift * log_codeword_len + coordinate;
            let cols: &mut NativeShiftScheduleCols<F> =
                trace[row_index * width..(row_index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.shift = F::from_usize(shift);
            cols.coordinate = F::from_usize(coordinate);
            cols.is_first = F::from_bool(coordinate == 0);
            cols.is_last = F::from_bool(coordinate + 1 == log_codeword_len);
            cols.is_first_shift = F::from_bool(shift == 0 && coordinate == 0);
            cols.proof_idx = F::from_usize(proof_idx);
            cols.tidx = F::from_usize(sample_tidx[shift]);
            cols.sample = samples[shift];
            cols.quotient = F::from_u32(quotient);
            cols.index = F::from_u32(indices[shift]);
            cols.bit = F::from_u32(bit);
            cols.power = F::from_u32(1u32 << bit_index);
            cols.reconstructed_before = F::from_u32(before);
            cols.reconstructed_after = F::from_u32(reconstructed);
            cols.value[0] = F::from_u32(bit);
            cols.index_lookup_count = if coordinate == 0 {
                F::from_u32(index_lookup_count)
            } else {
                F::ZERO
            };
        }
    }
    Some(RowMajorMatrix::new(trace, width))
}

impl openvm_stark_backend::BaseAirWithPublicValues<F> for ReducedSwirlVaccShiftScheduleAir {}
impl openvm_stark_backend::PartitionedBaseAir<F> for ReducedSwirlVaccShiftScheduleAir {}
impl BaseAir<F> for ReducedSwirlVaccShiftScheduleAir {
    fn width(&self) -> usize {
        NativeShiftScheduleCols::<F>::width()
    }
}
impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for ReducedSwirlVaccShiftScheduleAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("direct VACC shift row");
        let next_row = main.row_slice(1).expect("direct VACC next shift row");
        let local: &NativeShiftScheduleCols<AB::Var> = (*local_row).borrow();
        let next: &NativeShiftScheduleCols<AB::Var> = (*next_row).borrow();
        for flag in [
            local.active,
            local.is_first,
            local.is_last,
            local.is_first_shift,
            local.bit,
        ] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first_shift);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .when(local.active - next.active)
            .assert_one(local.is_last);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);
        builder
            .when(local.active * local.is_first_shift)
            .assert_one(local.is_first);
        builder
            .when(local.active * local.is_first_shift)
            .assert_zero(local.shift);
        builder
            .when(local.active * local.is_first)
            .assert_zero(local.coordinate);
        builder.when(local.active * local.is_last).assert_eq(
            local.coordinate,
            AB::Expr::from_usize(self.log_codeword_len - 1),
        );
        builder
            .when(local.active * local.is_first)
            .assert_zero(local.reconstructed_before);
        builder.when(local.active * local.is_first).assert_eq(
            local.power,
            AB::Expr::from_u32(1u32 << (self.log_codeword_len - 1)),
        );
        builder.when(local.active).assert_eq(
            local.reconstructed_after,
            local.reconstructed_before + local.bit * local.power,
        );
        let same_shift = next.active * (AB::Expr::ONE - next.is_first);
        let mut transition = builder.when_transition();
        let mut same = transition.when(same_shift);
        same.assert_eq(next.proof_idx, local.proof_idx);
        same.assert_zero(next.is_first_shift);
        same.assert_eq(next.shift, local.shift);
        same.assert_eq(next.coordinate, local.coordinate + AB::F::ONE);
        same.assert_eq(next.tidx, local.tidx);
        same.assert_eq(next.sample, local.sample);
        same.assert_eq(next.quotient, local.quotient);
        same.assert_eq(next.index, local.index);
        same.assert_eq(local.power, next.power * AB::F::TWO);
        same.assert_eq(next.reconstructed_before, local.reconstructed_after);
        let mut transition = builder.when_transition();
        let mut next_shift = transition.when(next.active * next.is_first);
        next_shift.assert_one(local.is_last);
        next_shift
            .when(next.is_first_shift)
            .assert_eq(next.proof_idx, local.proof_idx + AB::F::ONE);
        next_shift
            .when(AB::Expr::ONE - next.is_first_shift)
            .assert_eq(next.proof_idx, local.proof_idx);
        next_shift
            .when(AB::Expr::ONE - next.is_first_shift)
            .assert_eq(next.shift, local.shift + AB::F::ONE);
        builder
            .when(local.active * local.is_last)
            .assert_eq(local.reconstructed_after, local.index);

        self.transcript_bus.sample(
            builder,
            local.proof_idx,
            local.tidx,
            local.sample,
            local.active * local.is_first,
        );
        self.right_shift_bus.lookup_key(
            builder,
            RightShiftMessage {
                input: local.sample.into(),
                shift_bits: AB::Expr::from_usize(self.log_codeword_len),
                result: local.quotient.into(),
            },
            local.active * local.is_first,
        );
        self.exp_bits_len_bus.lookup_key(
            builder,
            ExpBitsLenMessage {
                base: AB::Expr::ONE,
                bit_src: local.sample.into(),
                num_bits: AB::Expr::ZERO,
                result: AB::Expr::ONE,
            },
            local.active * local.is_first,
        );
        builder.when(local.active * local.is_first).assert_eq(
            local.sample,
            local.index + local.quotient * AB::Expr::from_u32(1u32 << self.log_codeword_len),
        );
        self.shift_index_bus.add_key_with_lookups(
            builder,
            NativeShiftIndexMessage {
                proof_idx: local.proof_idx.into(),
                shift: local.shift.into(),
                index: local.index.into(),
            },
            local.index_lookup_count,
        );
        builder
            .when(local.active)
            .assert_eq(local.value[0], local.bit);
        for limb in &local.value[1..] {
            builder.when(local.active).assert_zero(*limb);
        }
        self.opening_bus.send(
            builder,
            NativeOpeningClaimMessage {
                proof_idx: local.proof_idx.into(),
                claim: AB::Expr::from_usize(self.opening_claim_offset) + local.shift,
                section: AB::Expr::from_usize(OPENING_SECTION_POINT),
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active,
        );
        self.role_bus.receive(
            builder,
            NativeVaccTranscriptRoleMessage {
                proof_idx: local.proof_idx.into(),
                role: AB::Expr::from_usize(VACC_ROLE_SHIFT),
                ordinal: local.shift.into(),
                tidx: local.tidx.into(),
                value: [
                    local.sample.into(),
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                ],
                is_ext: AB::Expr::ZERO,
                is_sample: AB::Expr::ONE,
            },
            local.active * local.is_first,
        );
    }
}

pub struct ReducedSwirlVaccBatchingFinalAir {
    pub inner: NativeBatchingFinalAir,
    pub role_bus: NativeVaccTranscriptRoleBus,
}

impl openvm_stark_backend::BaseAirWithPublicValues<F> for ReducedSwirlVaccBatchingFinalAir {}
impl openvm_stark_backend::PartitionedBaseAir<F> for ReducedSwirlVaccBatchingFinalAir {}
impl BaseAir<F> for ReducedSwirlVaccBatchingFinalAir {
    fn width(&self) -> usize {
        NativeBatchingFinalCols::<F>::width()
    }
}
impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for ReducedSwirlVaccBatchingFinalAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield:
        openvm_stark_backend::p3_field::extension::BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        self.inner.eval(builder);
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("direct VACC batching-final role row");
        let local: &NativeBatchingFinalCols<AB::Var> = (*row).borrow();
        self.role_bus.receive(
            builder,
            NativeVaccTranscriptRoleMessage {
                proof_idx: local.proof_idx.into(),
                role: AB::Expr::from_usize(VACC_ROLE_MU),
                ordinal: AB::Expr::ZERO,
                tidx: local.mu_tidx.into(),
                value: local.mu.map(Into::into),
                is_ext: AB::Expr::ONE,
                is_sample: AB::Expr::ZERO,
            },
            local.active * local.is_last,
        );
    }
}

/// Tree identifiers used by reduced-SWIRL source and accumulator openings.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlVaccTreeLayout {
    pub fresh_outer: usize,
    pub fresh_rows: usize,
    pub prior_outer: usize,
    pub prior_rows: usize,
}

/// Verifier-key shape for the reduced-SWIRL original-root source lane. Every
/// native call uses this one fixed arity. Runtime `fresh_count` is constrained
/// by the outer reduced schedule and may vary only through the active prefix;
/// the relation never becomes keyed by an individual call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlVaccVerifierConfig {
    pub input_arity: usize,
    pub max_roots_per_source: usize,
    pub projection_sources_per_shard: usize,
    pub max_projection_height: usize,
}

impl ReducedSwirlVaccVerifierConfig {
    pub fn validate(&self) -> Result<(), ReducedSwirlVaccVerifierError> {
        if self.input_arity < 2
            || self.input_arity > 64
            || !self.input_arity.is_power_of_two()
            || self.max_roots_per_source == 0
            || self.projection_sources_per_shard == 0
            || self.projection_sources_per_shard > self.input_arity
            || self.max_projection_height == 0
            || !self.max_projection_height.is_power_of_two()
        {
            return Err(ReducedSwirlVaccVerifierError::InvalidProfile);
        }
        Ok(())
    }
}

/// Verifier-key composition for reduced-SWIRL WARP accumulation.
///
/// One fixed AIR inventory verifies every transition. Runtime arity occupancy
/// and bootstrap/continuation state are constrained rows rather than distinct
/// verifier keys.
pub struct ReducedSwirlVaccVerifierModule {
    pub profile: NativeStandardVaccShapeProfile,
    pub shared: BusInventory,
    pub buses: NativeWarpVerifierBusInventory,
    pub end_bus: NativeStandardVaccEndBus,
    pub statement_root_bus: NativeStandardVaccRootBus,
    pub statement_digest_bus: NativeStandardVaccDigestBus,
    pub algebra: NativeWarpAlgebraLayout,
    pub private_accumulator: NativePrivateAccumulatorLayout,
    pub trees: ReducedSwirlVaccTreeLayout,
    pub transcript: NativeWarpTranscriptModule,
    /// Number of rows in the vector-alphabet RS oracle. The flattened WARP
    /// codeword length remains `2^profile.log_codeword_len`.
    pub oracle_height: usize,
    pub query_count: usize,
    pub reduced_swirl: ReducedSwirlVaccVerifierConfig,
}

impl ReducedSwirlVaccVerifierModule {
    /// Construct the setup-fixed verifier shared by every reduced-SWIRL WARP
    /// transition in a block.
    #[allow(clippy::too_many_arguments)]
    pub fn new_reduced_swirl_batched(
        profile: NativeStandardVaccProfile,
        reduced_swirl: ReducedSwirlVaccVerifierConfig,
        shared: BusInventory,
        buses: NativeWarpVerifierBusInventory,
        end_bus: NativeStandardVaccEndBus,
        statement_root_bus: NativeStandardVaccRootBus,
        statement_digest_bus: NativeStandardVaccDigestBus,
        system_params: SystemParams,
    ) -> Result<Self, ReducedSwirlVaccVerifierError> {
        profile
            .validate()
            .map_err(|_| ReducedSwirlVaccVerifierError::InvalidProfile)?;
        reduced_swirl.validate()?;

        let profile = profile.shape_profile();
        if profile.log_codeword_len > MAX_RAW_MESSAGE_POINT_LEN_
            || profile.beta_len > MAX_FRESH_BETA_LEN_
        {
            return Err(ReducedSwirlVaccVerifierError::InvalidProfile);
        }
        let codeword_len = 1usize
            .checked_shl(profile.log_codeword_len as u32)
            .ok_or(ReducedSwirlVaccVerifierError::InvalidProfile)?;
        let alphabet_width = 1usize
            .checked_shl(profile.initial_folding_factor as u32)
            .ok_or(ReducedSwirlVaccVerifierError::InvalidProfile)?;
        let oracle_height = codeword_len
            .checked_div(alphabet_width)
            .ok_or(ReducedSwirlVaccVerifierError::InvalidProfile)?;
        let query_count = oracle_height
            .checked_div(profile.rows_per_query)
            .filter(|count| count.is_power_of_two())
            .ok_or(ReducedSwirlVaccVerifierError::InvalidProfile)?;
        let family = family_from_shape_profile(
            &profile,
            reduced_swirl.input_arity,
            reduced_swirl.input_arity,
        );
        let fresh_tree_count = reduced_swirl
            .input_arity
            .checked_mul(reduced_swirl.max_roots_per_source)
            .and_then(|count| count.checked_mul(1 + profile.num_shift_queries))
            .ok_or(ReducedSwirlVaccVerifierError::InvalidProfile)?;
        let transcript = NativeWarpTranscriptModule::new_with_certified_checkpoints(
            &shared,
            &buses,
            system_params,
            false,
            true,
        );

        Ok(Self {
            algebra: NativeWarpAlgebraLayout::new(&family),
            private_accumulator: NativePrivateAccumulatorLayout::new(
                0,
                profile.log_codeword_len,
                profile.beta_len,
            ),
            trees: ReducedSwirlVaccTreeLayout {
                fresh_outer: 0,
                fresh_rows: 1,
                prior_outer: fresh_tree_count,
                prior_rows: fresh_tree_count + 1,
            },
            profile,
            shared,
            buses,
            end_bus,
            statement_root_bus,
            statement_digest_bus,
            transcript,
            oracle_height,
            query_count,
            reduced_swirl,
        })
    }

    fn input_arity(&self) -> usize {
        self.reduced_swirl.input_arity
    }

    /// AIRs in the exact order returned by the trace generator below.
    #[must_use]
    pub fn airs<PCS: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<PCS>> {
        self.reduced_swirl_airs::<PCS>()
    }
    /// One fixed AIR inventory for every reduced-SWIRL call. Runtime calls
    /// are rows, never verifier-key-specialized AIR clones.
    fn reduced_swirl_airs<PCS: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<PCS>> {
        let reduced = &self.reduced_swirl;
        let input_arity = reduced.input_arity;
        let twin_rounds = input_arity.ilog2() as usize;
        let constraint_degree = twin_degree(
            self.profile.log_codeword_len,
            self.profile.log_constraints,
            self.profile.max_degree,
        );
        let algebra_profile = NativeVaccCallAlgebraProfile {
            input_arity,
            twin_rounds,
            twin_last_round: twin_rounds - 1,
            twin_degree: constraint_degree,
            batching_rounds: self.profile.log_codeword_len,
            batching_degree: 2,
        };
        let family = family_from_shape_profile(&self.profile, input_arity, input_arity);
        let claim_count = NativeWarpAlgebraLayout::claim_count(&family);
        let authenticated_claim_count = 1 + self.profile.num_ood + self.profile.num_shift_queries;
        let opening_padding_count = claim_count - authenticated_claim_count;
        let log_claim_count = claim_count.ilog2() as usize;
        let query_stride = (1usize << self.profile.log_codeword_len) / self.profile.rows_per_query;
        let outer_depth = query_stride.ilog2() as usize;
        let root_tree_stride = 1 + self.profile.num_shift_queries;
        let tree_source_stride = reduced.max_roots_per_source * root_tree_stride;

        let mut airs = self.transcript.airs::<PCS>();
        let mut role_lengths =
            vacc_cursor_role_lengths_exact(&self.profile, input_arity, input_arity, true);
        role_lengths[VACC_ROLE_FRESH_ROOT] =
            input_arity * (9 + reduced.max_roots_per_source * (1 + DIGEST_SIZE));
        add_air(
            &mut airs,
            ReducedSwirlVaccTranscriptCursorAir {
                transcript_bus: self.buses.transcript,
                semantic_bus: self.buses.vacc_semantic_transcript,
                phase_cursor_bus: self.buses.vacc_phase_cursor,
                role_bus: self.buses.vacc_transcript_role,
                next_roles: next_nonempty_vacc_roles(&role_lengths),
                lengths: role_lengths,
                twin_event_count: constraint_degree + 2,
                dynamic_role_lengths: true,
                dynamic_optional_prior: true,
                emit_fresh_root_role: false,
            },
        );
        add_air(
            &mut airs,
            ReducedSwirlVaccEndAir {
                transcript_bus: self.buses.transcript,
                phase_cursor_bus: self.buses.vacc_phase_cursor,
                end_bus: self.end_bus,
                end_tag: WARP_CALL_END_TAG,
            },
        );
        add_air(
            &mut airs,
            NativeStandardClaimValueAir {
                claim_bus: self.buses.claim_value,
                transcript_bus: self.buses.vacc_semantic_transcript,
                transcript_role_bus: self.buses.vacc_transcript_role,
                layout_bus: self.buses.claim_layout,
                slot_bus: self.buses.input_slot_layout,
                // Algebra plus the authoritative reduced-source linker.
                fresh_claim_consumer_count: 2,
                // Algebra plus the dynamic-prior adapter.
                prior_claim_consumer_count: 2,
                fresh_alpha_len: self.profile.log_codeword_len,
                fresh_tau_len: self.profile.log_constraints,
                fresh_beta_tail_len: self.profile.beta_len - self.profile.log_constraints,
            },
        );
        add_air(
            &mut airs,
            NativeClaimLayoutAir {
                layout_bus: self.buses.claim_layout,
            },
        );
        add_air(
            &mut airs,
            NativeDirectFreshCommitmentAir {
                transcript_bus: self.buses.vacc_semantic_transcript,
                source_bus: self.buses.direct_fresh_source,
                activity_bus: self.buses.fresh_source_activity,
                fresh_count_bus: self.buses.fresh_count,
                digest_element_bus: None,
                max_fresh: input_arity,
                max_roots: reduced.max_roots_per_source,
                shift_count: self.profile.num_shift_queries,
                expected_log_message_len: self.profile.log_message_len,
                expected_log_codeword_len: self.profile.log_codeword_len,
                expected_rows_per_query: self.profile.rows_per_query,
                tree_source_stride,
                first_tree_id: 0,
                digest_metadata_len: 0,
                digest_source_width: 0,
                digest_alpha_len: 0,
                digest_beta_len: 0,
                claim_rows_per_source: 0,
            },
        );
        add_air(
            &mut airs,
            NativeDirectFreshRootAir {
                transcript_bus: self.buses.vacc_semantic_transcript,
                source_bus: self.buses.direct_fresh_source,
                root_bus: self.buses.direct_fresh_root,
                merkle_root_bus: self.buses.merkle_root,
                digest_element_bus: None,
                max_fresh: input_arity,
                max_roots: reduced.max_roots_per_source,
                shift_count: self.profile.num_shift_queries,
                root_tree_stride,
                outer_depth,
                digest_metadata_len: 0,
                digest_source_width: 0,
            },
        );
        for first_source in (0..input_arity).step_by(reduced.projection_sources_per_shard) {
            add_air(
                &mut airs,
                NativeDirectFreshProjectionAir {
                    source_bus: self.buses.direct_fresh_source,
                    root_bus: self.buses.direct_fresh_root,
                    shift_index_bus: self.buses.shift_index,
                    leaf_value_bus: self.buses.leaf_value,
                    authenticated_bus: self.buses.authenticated_shift,
                    shift_count: self.profile.num_shift_queries,
                    query_stride,
                    root_tree_stride,
                    first_source,
                    allow_empty: first_source != 0,
                },
            );
        }
        add_air(
            &mut airs,
            NativeAccumulatorProjectionAir {
                shift_index_bus: self.buses.shift_index,
                leaf_value_bus: self.buses.leaf_value,
                authenticated_bus: self.buses.authenticated_shift,
                slot_bus: self.buses.input_slot_layout,
                oracle_height: self.oracle_height,
                query_count: self.query_count,
                row_tree_id_offset: self.trees.prior_rows,
            },
        );
        self.add_root_air(
            &mut airs,
            1,
            self.trees.prior_outer,
            outer_depth,
            true,
            true,
        );
        add_air(
            &mut airs,
            NativeLeafHashAir {
                permute_bus: self.shared.poseidon2_permute_bus,
                value_bus: self.buses.leaf_value,
                leaf_bus: self.buses.opening_leaf,
            },
        );
        add_air(
            &mut airs,
            NativeMerkleMultiproofAir {
                compress_bus: self.shared.poseidon2_compress_bus,
                leaf_bus: self.buses.opening_leaf,
                node_bus: self.buses.merkle_node,
                root_bus: self.buses.merkle_root,
            },
        );
        add_air(
            &mut airs,
            NativeMerkleLeafAdapterAir {
                inner_depth: self.profile.rows_per_query.ilog2() as usize,
                leaf_bus: self.buses.opening_leaf,
                root_bus: self.buses.merkle_root,
            },
        );
        add_air(
            &mut airs,
            ReducedSwirlVaccVectorCoordinateAir {
                inner: NativeVectorCoordinateAir {
                    vector_bus: self.buses.vector_coordinate,
                    sumcheck_bus: self.buses.sumcheck_challenge,
                    transcript_bus: self.buses.vacc_semantic_transcript,
                    folded_bus: self.buses.folded_claim,
                    opening_bus: self.buses.opening_claim,
                },
                role_bus: self.buses.vacc_transcript_role,
                selector_vector: self.algebra.selector_tau_vector as usize,
                xi_vector: self.algebra.xi_vector as usize,
            },
        );
        for (dimensions, group_offset) in [
            (twin_rounds, 0usize),
            (log_claim_count, 2 * input_arity + 1),
            (
                self.profile.log_codeword_len,
                2 * input_arity + 1 + claim_count,
            ),
        ] {
            add_air(
                &mut airs,
                NativeEqEvaluationAir {
                    result_bus: self.buses.eq_result,
                    vector_bus: self.buses.vector_coordinate,
                    dimensions,
                    group_offset,
                },
            );
        }
        add_air(
            &mut airs,
            ReducedSwirlVaccTwinOmegaAir {
                inner: NativeTwinOmegaAir {
                    transcript_bus: self.buses.vacc_semantic_transcript,
                    omega_bus: self.buses.twin_omega,
                },
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            algebra_profile.twin_sigma_air(
                self.buses.claim_value,
                self.buses.eq_result,
                self.buses.sumcheck_initial,
                self.buses.twin_omega,
            ),
        );
        add_air(
            &mut airs,
            ReducedSwirlVaccCoefficientSumcheckAir {
                inner: algebra_profile.coefficient_sumcheck_air(
                    self.buses.vacc_semantic_transcript,
                    self.buses.sumcheck_round,
                    self.buses.sumcheck_initial,
                    self.buses.sumcheck_challenge,
                ),
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            algebra_profile.twin_fold_air(
                self.buses.claim_value,
                self.buses.eq_result,
                self.buses.folded_claim,
            ),
        );
        add_air(
            &mut airs,
            ReducedSwirlVaccTwinFinalAir {
                inner: algebra_profile.twin_final_air(
                    self.buses.sumcheck_round,
                    self.buses.eq_result,
                    self.buses.twin_scalar,
                    self.algebra.selector_at_gamma_group as usize,
                    self.buses.twin_omega,
                    self.buses.vacc_semantic_transcript,
                ),
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            NativeInitialOpeningPointAir {
                folded_bus: self.buses.folded_claim,
                opening_bus: self.buses.opening_claim,
                copies: opening_padding_count + 1,
            },
        );
        add_air(
            &mut airs,
            NativeInitialOpeningTargetAir {
                twin_bus: self.buses.twin_scalar,
                opening_bus: self.buses.opening_claim,
                copies: opening_padding_count + 1,
            },
        );
        add_air(
            &mut airs,
            NativeOpeningPaddingAir {
                opening_bus: self.buses.opening_claim,
            },
        );
        add_air(
            &mut airs,
            ReducedSwirlVaccOodClaimAir {
                inner: NativeOodClaimAir {
                    transcript_bus: self.buses.vacc_semantic_transcript,
                    opening_bus: self.buses.opening_claim,
                },
                role_bus: self.buses.vacc_transcript_role,
                dimension: self.profile.log_codeword_len,
            },
        );
        add_air(
            &mut airs,
            ReducedSwirlVaccShiftScheduleAir {
                transcript_bus: self.buses.vacc_semantic_transcript,
                exp_bits_len_bus: self.shared.exp_bits_len_bus,
                right_shift_bus: self.shared.right_shift_bus,
                shift_index_bus: self.buses.shift_index,
                opening_bus: self.buses.opening_claim,
                log_codeword_len: self.profile.log_codeword_len,
                opening_claim_offset: 1 + self.profile.num_ood,
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            NativeShiftMergeAir {
                authenticated_bus: self.buses.authenticated_shift,
                eq_bus: self.buses.eq_result,
                opening_bus: self.buses.opening_claim,
                slot_bus: self.buses.input_slot_layout,
                input_arity,
                gamma_eq_group_offset: input_arity,
                opening_claim_offset: 1 + self.profile.num_ood,
            },
        );
        add_air(
            &mut airs,
            NativeBatchingSigmaAir {
                eq_bus: self.buses.eq_result,
                opening_bus: self.buses.opening_claim,
                sumcheck_initial_bus: Some(self.buses.sumcheck_initial),
                certified_claim_bus: None,
                claim_count,
                xi_eq_group_offset: self.algebra.xi_weight_groups[0] as usize,
            },
        );
        add_air(
            &mut airs,
            ReducedSwirlVaccBatchingFinalAir {
                inner: NativeBatchingFinalAir {
                    eq_bus: self.buses.eq_result,
                    sumcheck_round_bus: self.buses.sumcheck_round,
                    output_bus: self.buses.batching_output,
                    claim_count,
                    xi_eq_group_offset: self.algebra.xi_weight_groups[0] as usize,
                    point_eq_group_offset: self.algebra.opening_at_alpha_groups[0] as usize,
                    last_round: self.profile.log_codeword_len - 1,
                    transcript_bus: self.buses.vacc_semantic_transcript,
                },
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            NativeBatchingAlphaAir {
                challenge_bus: self.buses.sumcheck_challenge,
                output_bus: self.buses.batching_output,
            },
        );
        self.add_root_air(&mut airs, 2, 0, 0, false, true);
        add_air(
            &mut airs,
            ReducedSwirlPriorClaimAdapterAir {
                claim_bus: self.buses.claim_value,
                slot_bus: self.buses.input_slot_layout,
                virtual_source: input_arity - 1,
            },
        );
        self.add_reduced_accumulator_airs(&mut airs, NativeAccumulatorBindingMode::Prior, true);
        self.add_reduced_accumulator_airs(&mut airs, NativeAccumulatorBindingMode::Output, false);
        add_air(
            &mut airs,
            ExpBitsLenAir::new(self.shared.exp_bits_len_bus, self.shared.right_shift_bus),
        );
        airs
    }

    fn add_reduced_accumulator_airs<PCS: StarkProtocolConfig<F = F>>(
        &self,
        airs: &mut Vec<AirRef<PCS>>,
        mode: NativeAccumulatorBindingMode,
        allow_empty: bool,
    ) {
        let state = usize::from(mode == NativeAccumulatorBindingMode::Output);
        add_air(
            airs,
            ReducedSwirlAccumulatorValueAir {
                mode,
                allow_empty,
                layout: self.private_accumulator.clone(),
                claim_source: self.input_arity() - 1,
                digest_element_bus: self.buses.accumulator_digest_element,
                claim_bus: self.buses.claim_value,
                batching_bus: self.buses.batching_output,
                folded_bus: self.buses.folded_claim,
                twin_bus: self.buses.twin_scalar,
            },
        );
        add_air(
            airs,
            NativeAccumulatorHashAir {
                state,
                allow_empty,
                alpha_len: self.private_accumulator.alpha_len,
                beta_len: self.private_accumulator.beta_len,
                digest_element_bus: self.buses.accumulator_digest_element,
                algebraic_digest_bus: self.buses.accumulator_algebraic_digest,
                permute_bus: self.shared.poseidon2_permute_bus,
                compress_bus: self.shared.poseidon2_compress_bus,
            },
        );
        add_air(
            airs,
            ReducedSwirlAccumulatorRootDigestAir {
                mode,
                allow_empty,
                root_bus: self.buses.accumulator_root,
                algebraic_digest_bus: self.buses.accumulator_algebraic_digest,
                compress_bus: self.shared.poseidon2_compress_bus,
                digest_bus: self.statement_digest_bus,
            },
        );
    }

    /// Return the verifier AIRs while delegating the physical Poseidon table
    /// to the reduced-SWIRL parent composition. Transcript and authentication
    /// AIR order remains unchanged, and the parent supplies the grouped
    /// inputs returned by [`Self::generate_traces_for_shared_poseidon`].
    #[must_use]
    pub fn airs_without_poseidon<PCS: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<PCS>> {
        let mut airs = self.airs::<PCS>();
        assert!(airs.len() >= 2, "direct VACC verifier AIR prefix");
        airs.remove(1);
        airs
    }

    fn add_root_air<PCS: StarkProtocolConfig<F = F>>(
        &self,
        airs: &mut Vec<AirRef<PCS>>,
        kind: usize,
        tree_id: usize,
        depth: usize,
        authenticate_merkle: bool,
        bind_accumulator: bool,
    ) {
        add_air(
            airs,
            NativeStandardVaccCommitmentRootAir {
                transcript_bus: self.buses.vacc_semantic_transcript,
                transcript_role_bus: self.buses.vacc_transcript_role,
                merkle_root_bus: authenticate_merkle.then_some(self.buses.merkle_root),
                accumulator_root_bus: bind_accumulator.then_some(self.buses.accumulator_root),
                statement_root_bus: self.statement_root_bus,
                proof_kind: kind,
                transcript_role: match kind {
                    0 => VACC_ROLE_FRESH_ROOT,
                    1 => VACC_ROLE_PRIOR_ROOT,
                    2 => VACC_ROLE_OUTPUT_ROOT,
                    _ => unreachable!("standard VACC root kind is verifier-key data"),
                },
                expected_tree_id: tree_id,
                expected_depth: depth,
            },
        );
    }
}

fn family_from_shape_profile(
    profile: &NativeStandardVaccShapeProfile,
    input_arity: usize,
    fresh_count: usize,
) -> NativeWarpFamilyParams {
    NativeWarpFamilyParams {
        max_shape_slots: 1,
        input_arity,
        max_fresh_per_step: fresh_count,
        num_ood: profile.num_ood,
        num_shift_queries: profile.num_shift_queries,
        max_stacked_roots: 1,
        max_stacked_width: 1 << profile.initial_folding_factor,
        max_public_values: 0,
        log_message_height: profile.log_message_len - profile.initial_folding_factor,
        accumulator_rows_per_query: profile.rows_per_query,
        source_rows_per_query: profile.rows_per_query,
        log_codeword_len: profile.log_codeword_len,
        log_constraints: profile.log_constraints,
        explicit_len: profile.beta_len - profile.log_constraints,
        beta_len: profile.beta_len,
    }
}

fn add_air<PCS, A>(airs: &mut Vec<AirRef<PCS>>, air: A)
where
    PCS: StarkProtocolConfig,
    A: AnyAir<PCS> + 'static,
{
    airs.push(Arc::new(air));
}

/// Transcript coordinates of the prior accumulator, when one is present.
#[derive(Clone, Debug, PartialEq, Eq)]
struct ReducedSwirlPriorTranscriptSchedule {
    root_tidx: usize,
    alpha_tidx: Vec<usize>,
    mu_tidx: usize,
    beta_tidx: Vec<usize>,
    eta_tidx: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReducedSwirlSampleBitsSchedule {
    operation_range: core::ops::Range<usize>,
    result: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ReducedSwirlVaccTranscriptSchedule {
    proof_idx: usize,
    start_tidx: usize,
    fresh_root_tidx: usize,
    fresh_commitment_extra_tidx: Vec<usize>,
    fresh_alpha_tidx: Vec<usize>,
    fresh_mu_tidx: Vec<usize>,
    fresh_beta_tail_tidx: Vec<usize>,
    fresh_tau_tidx: Vec<usize>,
    prior: Option<ReducedSwirlPriorTranscriptSchedule>,
    omega_tidx: usize,
    selector_tidx: Vec<usize>,
    gamma_tidx: Vec<usize>,
    output_root_tidx: usize,
    nu_tidx: usize,
    eta_tidx: usize,
    ood_point_tidx: Vec<usize>,
    ood_answer_tidx: Vec<usize>,
    shifts: Vec<ReducedSwirlSampleBitsSchedule>,
    xi_tidx: Vec<usize>,
    mu_tidx: usize,
    end_tag_tidx: usize,
    end_tidx: usize,
    discarded_end_sample: EF,
    claimed: Vec<bool>,
}

impl ReducedSwirlVaccTranscriptSchedule {
    fn from_reduced_swirl_record(
        module: &ReducedSwirlVaccVerifierModule,
        record: &ReducedSwirlVaccVerifierRecord<'_>,
    ) -> Result<Self, ReducedSwirlVaccVerifierError> {
        let reduced = &module.reduced_swirl;
        let verification = record.verification;
        let log = record.transcript;
        let fresh_count = record.proof.inner().fresh_claims.len();
        let has_prior = record.prior.is_some();
        let selector_rounds = reduced.input_arity.ilog2() as usize;
        if fresh_count == 0
            || fresh_count + usize::from(has_prior) > reduced.input_arity
            || verification.fresh_authentication.len() != fresh_count
            || verification.prior_authentication.is_some() != has_prior
            || verification.twin.fresh_taus.len() != fresh_count
            || verification
                .twin
                .fresh_taus
                .iter()
                .any(|tau| tau.len() != module.profile.log_constraints)
            || verification.twin.selector_tau.len() != selector_rounds
            || verification.twin.gamma.len() != selector_rounds
            || verification.twin.zeta_0.len() != module.profile.log_codeword_len
            || verification.twin.output_beta.len() != module.profile.beta_len
            || verification.ood.len() != module.profile.num_ood
            || verification.shifts.len() != module.profile.num_shift_queries
            || verification.batching.alpha.len() != module.profile.log_codeword_len
            || verification.batching.xi.len() != module.profile.batching_arity.ilog2() as usize
            || record.commitment_tidxs.len() != fresh_count
            || has_prior != (record.proof_idx != 0)
        {
            return Err(ReducedSwirlVaccVerifierError::RecordShape(
                "reduced-SWIRL VACC dimensions",
            ));
        }
        validate_phase_ranges(
            log,
            &verification.transcript_phases,
            record.vacc_start_tidx,
            record.vacc_end_tidx,
        )?;
        let gamma_tidx = validate_sumcheck_spans(
            log,
            &verification.transcript_phases,
            &verification.twin.sumcheck,
            NativeSumcheckKind::TwinConstraint,
        )?;
        let batching_alpha_tidx = validate_sumcheck_spans(
            log,
            &verification.transcript_phases,
            &verification.batching.sumcheck,
            NativeSumcheckKind::MultilinearBatching,
        )?;
        if gamma_tidx.len() != selector_rounds
            || batching_alpha_tidx.len() != module.profile.log_codeword_len
        {
            return Err(ReducedSwirlVaccVerifierError::RecordShape(
                "reduced-SWIRL sumcheck rounds",
            ));
        }
        let protocol = unique_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::VaccProtocolPrefix,
        )?;
        if protocol.operation_range.start != record.vacc_start_tidx {
            return Err(ReducedSwirlVaccVerifierError::Transcript(
                "reduced-SWIRL VACC prefix start",
            ));
        }
        let commitments = unique_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::FreshCommitments,
        )?;
        let commitment_events = phase_events(log, commitments)?;
        let expected_commitment_events = record
            .proof
            .inner()
            .fresh_claims
            .iter()
            .try_fold(0usize, |total, claim| {
                total.checked_add(9 + claim.commitment.roots.len() * (1 + DIGEST_SIZE))
            })
            .ok_or(ReducedSwirlVaccVerifierError::RecordShape(
                "reduced-SWIRL commitment event count",
            ))?;
        if commitment_events.len() != expected_commitment_events {
            return Err(ReducedSwirlVaccVerifierError::Transcript(
                "reduced-SWIRL commitment events",
            ));
        }
        let commitment_tidx = commitment_events
            .iter()
            .map(|event| match event.kind {
                TranscriptEventKind::ObserveExt { degree: D_EF }
                    if event.operation_range.len() == D_EF =>
                {
                    Ok(event.operation_range.start)
                }
                _ => Err(ReducedSwirlVaccVerifierError::Transcript(
                    "reduced-SWIRL commitment event kind",
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut offset = 0usize;
        for (source, claim) in record.proof.inner().fresh_claims.iter().enumerate() {
            if commitment_tidx[offset] != record.commitment_tidxs[source] {
                return Err(ReducedSwirlVaccVerifierError::Transcript(
                    "reduced-SWIRL commitment descriptor boundary",
                ));
            }
            offset += 9 + claim.commitment.roots.len() * (1 + DIGEST_SIZE);
        }
        let fresh_root_tidx = commitment_tidx[0];
        let fresh_commitment_extra_tidx = commitment_tidx[1..].to_vec();

        let fresh_claims = unique_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::FreshClaims,
        )?;
        let mut events = phase_events(log, fresh_claims)?.iter();
        let fresh_alpha_tidx = take_ext_events(
            &mut events,
            fresh_count * module.profile.log_codeword_len,
            false,
        )?;
        let fresh_mu_tidx = take_ext_events(&mut events, fresh_count, false)?;
        let fresh_beta_tail_tidx = take_ext_events(
            &mut events,
            fresh_count * (module.profile.beta_len - module.profile.log_constraints),
            false,
        )?;
        ensure_no_events(events)?;
        let prior_phase = unique_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::PriorAccumulator,
        )?;
        let mut events = phase_events(log, prior_phase)?.iter();
        let prior = has_prior
            .then(|| {
                Ok(ReducedSwirlPriorTranscriptSchedule {
                    root_tidx: take_lifted_digest(&mut events)?,
                    alpha_tidx: take_ext_events(
                        &mut events,
                        module.profile.log_codeword_len,
                        false,
                    )?,
                    mu_tidx: take_ext_event(&mut events, false)?,
                    beta_tidx: take_ext_events(&mut events, module.profile.beta_len, false)?,
                    eta_tidx: take_ext_event(&mut events, false)?,
                })
            })
            .transpose()?;
        ensure_no_events(events)?;
        let tau_phase = unique_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::FreshZeroCheckPoints,
        )?;
        let mut events = phase_events(log, tau_phase)?.iter();
        let fresh_tau_tidx = take_ext_events(
            &mut events,
            fresh_count * module.profile.log_constraints,
            true,
        )?;
        ensure_no_events(events)?;
        let twin = unique_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::TwinChallenges,
        )?;
        let mut events = phase_events(log, twin)?.iter();
        let omega_tidx = take_ext_event(&mut events, true)?;
        let selector_tidx = take_ext_events(&mut events, selector_rounds, true)?;
        ensure_no_events(events)?;
        let output = unique_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::OutputAccumulatorCommitment,
        )?;
        let mut events = phase_events(log, output)?.iter();
        let output_root_tidx = take_lifted_digest(&mut events)?;
        let nu_tidx = take_ext_event(&mut events, false)?;
        let eta_tidx = take_ext_event(&mut events, false)?;
        ensure_no_events(events)?;
        let ood_points = unique_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::OodPoints,
        )?;
        let mut events = phase_events(log, ood_points)?.iter();
        let ood_point_tidx = (0..module.profile.num_ood)
            .map(|_| {
                take_ext_events(&mut events, module.profile.log_codeword_len, true)?
                    .first()
                    .copied()
                    .ok_or(ReducedSwirlVaccVerifierError::Transcript(
                        "reduced-SWIRL empty OOD point",
                    ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        ensure_no_events(events)?;
        let ood_answers = unique_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::OodAnswers,
        )?;
        let mut events = phase_events(log, ood_answers)?.iter();
        let ood_answer_tidx = take_ext_events(&mut events, module.profile.num_ood, false)?;
        ensure_no_events(events)?;
        let shift_phase = unique_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::ShiftIndices,
        )?;
        let shift_events = phase_events(log, shift_phase)?;
        if shift_events.len() != module.profile.num_shift_queries {
            return Err(ReducedSwirlVaccVerifierError::Transcript(
                "reduced-SWIRL shift event count",
            ));
        }
        let shifts = shift_events
            .iter()
            .map(|event| match event.kind {
                TranscriptEventKind::SampleBits { bits, result }
                    if bits == module.profile.log_codeword_len
                        && event.operation_range.len() == 1 =>
                {
                    Ok(ReducedSwirlSampleBitsSchedule {
                        operation_range: event.operation_range.clone(),
                        result,
                    })
                }
                _ => Err(ReducedSwirlVaccVerifierError::Transcript(
                    "reduced-SWIRL shift sampling event",
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let batching = unique_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::BatchingChallenge,
        )?;
        let mut events = phase_events(log, batching)?.iter();
        let xi_tidx = take_ext_events(
            &mut events,
            module.profile.batching_arity.ilog2() as usize,
            true,
        )?;
        ensure_no_events(events)?;
        let target = unique_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::FinalTarget,
        )?;
        let mut events = phase_events(log, target)?.iter();
        let mu_tidx = take_ext_event(&mut events, false)?;
        ensure_no_events(events)?;
        let boundary = unique_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::CallBoundary {
                call: record.proof_idx.try_into().map_err(|_| {
                    ReducedSwirlVaccVerifierError::Transcript(
                        "reduced-SWIRL boundary call-index width",
                    )
                })?,
            },
        )?;
        let end_tag_tidx = boundary.operation_range.start;
        let sample_tidx =
            end_tag_tidx
                .checked_add(D_EF)
                .ok_or(ReducedSwirlVaccVerifierError::Transcript(
                    "reduced-SWIRL end-tag overflow",
                ))?;
        let expected_tag = EF::from_u64(WARP_CALL_END_TAG);
        if end_tag_tidx != target.operation_range.end
            || boundary.operation_range.end != record.vacc_end_tidx
            || sample_tidx.checked_add(D_EF) != Some(record.vacc_end_tidx)
            || log.values().get(end_tag_tidx..sample_tidx)
                != Some(expected_tag.as_basis_coefficients_slice())
            || log
                .samples()
                .get(end_tag_tidx..sample_tidx)
                .is_none_or(|samples| samples.iter().any(|&sample| sample))
            || log
                .samples()
                .get(sample_tidx..record.vacc_end_tidx)
                .is_none_or(|samples| samples.iter().any(|&sample| !sample))
        {
            return Err(ReducedSwirlVaccVerifierError::Transcript(
                "reduced-SWIRL VACC transcript terminator",
            ));
        }
        let discarded_end_sample = EF::from_basis_coefficients_slice(
            log.values().get(sample_tidx..record.vacc_end_tidx).ok_or(
                ReducedSwirlVaccVerifierError::Transcript("reduced-SWIRL VACC end sample"),
            )?,
        )
        .ok_or(ReducedSwirlVaccVerifierError::Transcript(
            "reduced-SWIRL VACC end sample degree",
        ))?;
        let mut claimed = vec![false; log.len()];
        for phase in &verification.transcript_phases {
            mark_claimed(&mut claimed, phase.operation_range.clone())?;
        }
        let schedule = Self {
            proof_idx: record.local_proof_idx,
            start_tidx: record.vacc_start_tidx,
            fresh_root_tidx,
            fresh_commitment_extra_tidx,
            fresh_alpha_tidx,
            fresh_mu_tidx,
            fresh_beta_tail_tidx,
            fresh_tau_tidx,
            prior,
            omega_tidx,
            selector_tidx,
            gamma_tidx,
            output_root_tidx,
            nu_tidx,
            eta_tidx,
            ood_point_tidx,
            ood_answer_tidx,
            shifts,
            xi_tidx,
            mu_tidx,
            end_tag_tidx,
            end_tidx: record.vacc_end_tidx,
            discarded_end_sample,
            claimed,
        };
        schedule.validate_reduced_values(module, record)?;
        Ok(schedule)
    }

    fn validate_reduced_values(
        &self,
        module: &ReducedSwirlVaccVerifierModule,
        record: &ReducedSwirlVaccVerifierRecord<'_>,
    ) -> Result<(), ReducedSwirlVaccVerifierError> {
        let proof = record.proof.inner();
        let verification = record.verification;
        let alpha_len = module.profile.log_codeword_len;
        let tau_len = module.profile.log_constraints;
        let beta_tail_len = module.profile.beta_len - tau_len;
        for (source, claim) in proof.fresh_claims.iter().enumerate() {
            if claim.alpha.len() != alpha_len
                || claim.beta.len() != module.profile.beta_len
                || claim.eta != EF::ZERO
            {
                return Err(ReducedSwirlVaccVerifierError::RecordShape(
                    "reduced-SWIRL fresh claim dimensions",
                ));
            }
            for (coordinate, &value) in claim.alpha.iter().enumerate() {
                expect_ext(
                    record.transcript,
                    self.fresh_alpha_tidx[source * alpha_len + coordinate],
                    value,
                )?;
            }
            expect_ext(record.transcript, self.fresh_mu_tidx[source], claim.mu)?;
            for (coordinate, &value) in claim.beta[tau_len..].iter().enumerate() {
                expect_ext(
                    record.transcript,
                    self.fresh_beta_tail_tidx[source * beta_tail_len + coordinate],
                    value,
                )?;
            }
            for (coordinate, (&sampled, &retained)) in verification.twin.fresh_taus[source]
                .iter()
                .zip(&claim.beta[..tau_len])
                .enumerate()
            {
                expect_ext(
                    record.transcript,
                    self.fresh_tau_tidx[source * tau_len + coordinate],
                    sampled,
                )?;
                if sampled != retained {
                    return Err(ReducedSwirlVaccVerifierError::RecordShape(
                        "reduced-SWIRL fresh beta/tau prefix",
                    ));
                }
            }
        }
        expect_ext(record.transcript, self.omega_tidx, verification.twin.omega)?;
        for (tidx, &value) in self
            .selector_tidx
            .iter()
            .zip(&verification.twin.selector_tau)
        {
            expect_ext(record.transcript, *tidx, value)?;
        }
        for (tidx, &value) in self.gamma_tidx.iter().zip(&verification.twin.gamma) {
            expect_ext(record.transcript, *tidx, value)?;
        }
        expect_digest(
            record.transcript,
            self.output_root_tidx,
            &verification.output_instance.rt,
        )?;
        expect_ext(record.transcript, self.nu_tidx, verification.twin.nu_0)?;
        expect_ext(record.transcript, self.eta_tidx, verification.twin.eta)?;
        for (ordinal, ood) in verification.ood.iter().enumerate() {
            for (coordinate, &value) in ood.point.iter().enumerate() {
                expect_ext(
                    record.transcript,
                    self.ood_point_tidx[ordinal] + coordinate * D_EF,
                    value,
                )?;
            }
            expect_ext(record.transcript, self.ood_answer_tidx[ordinal], ood.answer)?;
        }
        for (sample, shift) in self.shifts.iter().zip(&verification.shifts) {
            if sample.result != u64::from(shift.index) {
                return Err(ReducedSwirlVaccVerifierError::Transcript(
                    "reduced-SWIRL shift result",
                ));
            }
        }
        for (tidx, &value) in self.xi_tidx.iter().zip(&verification.batching.xi) {
            expect_ext(record.transcript, *tidx, value)?;
        }
        expect_ext(
            record.transcript,
            self.mu_tidx,
            verification.batching.mu_final,
        )?;
        if let (Some(prior_schedule), Some(prior)) = (self.prior.as_ref(), record.prior) {
            expect_digest(record.transcript, prior_schedule.root_tidx, &prior.rt)?;
            for (tidx, &value) in prior_schedule.alpha_tidx.iter().zip(&prior.alpha) {
                expect_ext(record.transcript, *tidx, value)?;
            }
            expect_ext(record.transcript, prior_schedule.mu_tidx, prior.mu)?;
            for (tidx, &value) in prior_schedule.beta_tidx.iter().zip(&prior.beta) {
                expect_ext(record.transcript, *tidx, value)?;
            }
            expect_ext(record.transcript, prior_schedule.eta_tidx, prior.eta)?;
        }
        Ok(())
    }
}
fn validate_phase_ranges(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    phases: &[NativeTranscriptPhaseSpan],
    start: usize,
    end: usize,
) -> Result<(), ReducedSwirlVaccVerifierError> {
    let mut previous_event = phases.first().map_or(0, |phase| phase.event_range.start);
    let mut previous_operation = start;
    let mut previous_permutation = phases
        .first()
        .map_or(0, |phase| phase.permutation_range.start);
    for phase in phases {
        if phase.event_range.start != previous_event
            || phase.operation_range.start != previous_operation
            || phase.permutation_range.start != previous_permutation
            || phase.event_range.end > log.events().len()
            || phase.operation_range.end > log.len()
            || phase.operation_range.end > end
            || phase.permutation_range.end > log.permutation_transitions().len()
            || phase.event_range.start > phase.event_range.end
            || phase.operation_range.start > phase.operation_range.end
            || phase.permutation_range.start > phase.permutation_range.end
        {
            return Err(ReducedSwirlVaccVerifierError::Transcript(
                "non-canonical phase ranges",
            ));
        }
        let events = phase_events(log, phase)?;
        if events.first().map(|event| event.operation_range.start)
            != (!events.is_empty()).then_some(phase.operation_range.start)
            || events.last().map(|event| event.operation_range.end)
                != (!events.is_empty()).then_some(phase.operation_range.end)
        {
            return Err(ReducedSwirlVaccVerifierError::Transcript(
                "phase event boundaries",
            ));
        }
        previous_event = phase.event_range.end;
        previous_operation = phase.operation_range.end;
        previous_permutation = phase.permutation_range.end;
    }
    Ok(())
}

fn validate_sumcheck_spans(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    phases: &[NativeTranscriptPhaseSpan],
    verification: &openvm_stark_backend::warp_accum::CoefficientSumcheckVerification<EF>,
    kind: NativeSumcheckKind,
) -> Result<Vec<usize>, ReducedSwirlVaccVerifierError> {
    if verification.kind != kind {
        return Err(ReducedSwirlVaccVerifierError::Algebra("sumcheck kind"));
    }
    verification
        .rounds
        .iter()
        .map(|round| {
            let phase_kind = match kind {
                NativeSumcheckKind::TwinConstraint => {
                    NativeTranscriptPhase::TwinSumcheckRound { round: round.round }
                }
                NativeSumcheckKind::MultilinearBatching => {
                    NativeTranscriptPhase::BatchingSumcheckRound { round: round.round }
                }
            };
            let phase = unique_phase(phases, &phase_kind)?;
            if phase != &round.transcript_span {
                return Err(ReducedSwirlVaccVerifierError::Transcript(
                    "sumcheck phase span",
                ));
            }
            let events = phase_events(log, phase)?;
            if events.len() != round.coefficients.len() + 1 {
                return Err(ReducedSwirlVaccVerifierError::Transcript(
                    "sumcheck event count",
                ));
            }
            for event in &events[..round.coefficients.len()] {
                ext_event_tidx(event, false)?;
            }
            ext_event_tidx(&events[round.coefficients.len()], true)
        })
        .collect()
}

fn unique_phase<'a>(
    phases: &'a [NativeTranscriptPhaseSpan],
    wanted: &NativeTranscriptPhase,
) -> Result<&'a NativeTranscriptPhaseSpan, ReducedSwirlVaccVerifierError> {
    let mut matches = phases.iter().filter(|phase| &phase.phase == wanted);
    let phase = matches
        .next()
        .ok_or(ReducedSwirlVaccVerifierError::Transcript("missing phase"))?;
    if matches.next().is_some() {
        return Err(ReducedSwirlVaccVerifierError::Transcript("duplicate phase"));
    }
    Ok(phase)
}

fn phase_events<'a>(
    log: &'a TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    phase: &NativeTranscriptPhaseSpan,
) -> Result<&'a [TranscriptEvent], ReducedSwirlVaccVerifierError> {
    log.events()
        .get(phase.event_range.clone())
        .ok_or(ReducedSwirlVaccVerifierError::Transcript(
            "phase event range",
        ))
}

fn take_ext_events<'a>(
    events: &mut impl Iterator<Item = &'a TranscriptEvent>,
    count: usize,
    sampled: bool,
) -> Result<Vec<usize>, ReducedSwirlVaccVerifierError> {
    (0..count)
        .map(|_| take_ext_event(events, sampled))
        .collect()
}

fn take_ext_event<'a>(
    events: &mut impl Iterator<Item = &'a TranscriptEvent>,
    sampled: bool,
) -> Result<usize, ReducedSwirlVaccVerifierError> {
    ext_event_tidx(
        events
            .next()
            .ok_or(ReducedSwirlVaccVerifierError::Transcript("missing event"))?,
        sampled,
    )
}

fn take_lifted_digest<'a>(
    events: &mut impl Iterator<Item = &'a TranscriptEvent>,
) -> Result<usize, ReducedSwirlVaccVerifierError> {
    let first = take_ext_event(events, false)?;
    for limb in 1..DIGEST_SIZE {
        if take_ext_event(events, false)? != first + limb * D_EF {
            return Err(ReducedSwirlVaccVerifierError::Transcript(
                "lifted digest layout",
            ));
        }
    }
    Ok(first)
}

fn ensure_no_events<'a>(
    mut events: impl Iterator<Item = &'a TranscriptEvent>,
) -> Result<(), ReducedSwirlVaccVerifierError> {
    if events.next().is_some() {
        Err(ReducedSwirlVaccVerifierError::Transcript("phase tail"))
    } else {
        Ok(())
    }
}

fn ext_event_tidx(
    event: &TranscriptEvent,
    sampled: bool,
) -> Result<usize, ReducedSwirlVaccVerifierError> {
    let expected = if sampled {
        TranscriptEventKind::SampleExt { degree: D_EF }
    } else {
        TranscriptEventKind::ObserveExt { degree: D_EF }
    };
    if event.kind != expected || event.operation_range.len() != D_EF {
        return Err(ReducedSwirlVaccVerifierError::Transcript("extension event"));
    }
    Ok(event.operation_range.start)
}

fn mark_claimed(
    claimed: &mut [bool],
    range: core::ops::Range<usize>,
) -> Result<(), ReducedSwirlVaccVerifierError> {
    let values = claimed
        .get_mut(range)
        .ok_or(ReducedSwirlVaccVerifierError::Transcript(
            "claimed operation range",
        ))?;
    if values.iter().any(|&value| value) {
        return Err(ReducedSwirlVaccVerifierError::Transcript(
            "duplicate claimed operation",
        ));
    }
    values.fill(true);
    Ok(())
}

fn expect_ext(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    tidx: usize,
    expected: EF,
) -> Result<(), ReducedSwirlVaccVerifierError> {
    if log.values().get(tidx..tidx + D_EF) != Some(expected.as_basis_coefficients_slice()) {
        return Err(ReducedSwirlVaccVerifierError::Transcript(
            "extension value mismatch",
        ));
    }
    Ok(())
}

fn expect_digest(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    tidx: usize,
    digest: &Digest,
) -> Result<(), ReducedSwirlVaccVerifierError> {
    for (limb, &value) in digest.iter().enumerate() {
        expect_ext(
            log,
            tidx + limb * D_EF,
            EF::from_basis_coefficients_slice(&[value, F::ZERO, F::ZERO, F::ZERO])
                .expect("EF4 basis"),
        )?;
    }
    Ok(())
}

struct ReducedSwirlVaccAlgebraTraces {
    claim_values: RowMajorMatrix<F>,
    claim_layout: RowMajorMatrix<F>,
    vector: RowMajorMatrix<F>,
    selector_eq: RowMajorMatrix<F>,
    xi_eq: RowMajorMatrix<F>,
    opening_eq: RowMajorMatrix<F>,
    omega: RowMajorMatrix<F>,
    twin_sigma: RowMajorMatrix<F>,
    sumcheck: RowMajorMatrix<F>,
    twin_fold: RowMajorMatrix<F>,
    twin_final: RowMajorMatrix<F>,
    initial_point: RowMajorMatrix<F>,
    initial_target: RowMajorMatrix<F>,
    opening_padding: RowMajorMatrix<F>,
    ood_claims: RowMajorMatrix<F>,
    shift_schedule: RowMajorMatrix<F>,
    shift_merge: RowMajorMatrix<F>,
    batching_sigma: RowMajorMatrix<F>,
    batching_final: RowMajorMatrix<F>,
    batching_alpha: RowMajorMatrix<F>,
}

#[derive(Clone, Debug)]
struct ReducedSwirlVaccFreshClaimData {
    alpha: Vec<EF>,
    beta: Vec<EF>,
    mu: EF,
    eta: EF,
}

fn shared_twin_lookup_count(input_arity: usize) -> Result<u32, ReducedSwirlVaccVerifierError> {
    u32::try_from(
        input_arity
            .checked_add(1)
            .ok_or(ReducedSwirlVaccVerifierError::InvalidProfile)?,
    )
    .map_err(|_| ReducedSwirlVaccVerifierError::InvalidProfile)
}

fn vacc_eq_group_offsets(
    module: &ReducedSwirlVaccVerifierModule,
) -> Result<(usize, usize), ReducedSwirlVaccVerifierError> {
    let xi = *module
        .algebra
        .xi_weight_groups
        .first()
        .ok_or(ReducedSwirlVaccVerifierError::InvalidProfile)? as usize;
    let opening = *module
        .algebra
        .opening_at_alpha_groups
        .first()
        .ok_or(ReducedSwirlVaccVerifierError::InvalidProfile)? as usize;
    Ok((xi, opening))
}

impl<Comm> From<&FreshPesatClaim<EF, Comm>> for ReducedSwirlVaccFreshClaimData {
    fn from(claim: &FreshPesatClaim<EF, Comm>) -> Self {
        Self {
            alpha: claim.alpha.clone(),
            beta: claim.beta.clone(),
            mu: claim.mu,
            eta: claim.eta,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_reduced_swirl_vacc_algebra_traces_from_parts(
    module: &ReducedSwirlVaccVerifierModule,
    proof_idx: usize,
    fresh_claims: &[ReducedSwirlVaccFreshClaimData],
    v: &ReducedSwirlVaccVerification,
    transcript: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    prior: Option<&AccumulatorInstance<EF, Digest>>,
    schedule: &ReducedSwirlVaccTranscriptSchedule,
) -> Result<ReducedSwirlVaccAlgebraTraces, ReducedSwirlVaccVerifierError> {
    macro_rules! algebra {
        ($value:expr, $stage:literal) => {
            $value.ok_or(ReducedSwirlVaccVerifierError::Algebra($stage))?
        };
    }
    let profile = &module.profile;
    // `include_prior` is a verifier-key capability for the setup-fixed
    // reduced-SWIRL module.  Bootstrap deliberately exercises that same AIR
    // inventory without a prior accumulator, so trace generation must use
    // the authenticated record's actual mode.  Treating the capability bit
    // as runtime presence creates input_arity + 1 slots on a full bootstrap
    // batch and incorrectly includes a prior in the shift reduction.
    let has_prior = prior.is_some();
    let zero_alpha = vec![EF::ZERO; profile.log_codeword_len];
    let zero_beta = vec![EF::ZERO; profile.beta_len];
    let mut alphas = fresh_claims
        .iter()
        .map(|claim| claim.alpha.clone())
        .collect::<Vec<_>>();
    let mut betas = fresh_claims
        .iter()
        .map(|claim| claim.beta.clone())
        .collect::<Vec<_>>();
    let mut mus = fresh_claims
        .iter()
        .map(|claim| claim.mu)
        .collect::<Vec<_>>();
    let mut etas = fresh_claims
        .iter()
        .map(|claim| claim.eta)
        .collect::<Vec<_>>();
    let mut kinds = vec![(true, false, false); fresh_claims.len()];
    let alpha_len = profile.log_codeword_len;
    let tau_len = profile.log_constraints;
    let beta_tail_len = profile.beta_len - tau_len;
    let mut transcript_bindings = (0..fresh_claims.len())
        .map(|source| {
            let mut bindings = Vec::with_capacity(alpha_len + profile.beta_len + 2);
            bindings.extend(
                schedule.fresh_alpha_tidx[source * alpha_len..(source + 1) * alpha_len]
                    .iter()
                    .map(|&tidx| Some((proof_idx, tidx, false))),
            );
            bindings.extend(
                schedule.fresh_tau_tidx[source * tau_len..(source + 1) * tau_len]
                    .iter()
                    .map(|&tidx| Some((proof_idx, tidx, true))),
            );
            bindings.extend(
                schedule.fresh_beta_tail_tidx[source * beta_tail_len..(source + 1) * beta_tail_len]
                    .iter()
                    .map(|&tidx| Some((proof_idx, tidx, false))),
            );
            bindings.push(Some((proof_idx, schedule.fresh_mu_tidx[source], false)));
            bindings.push(None);
            bindings
        })
        .collect::<Vec<_>>();
    if let (Some(prior), Some(prior_schedule)) = (prior, schedule.prior.as_ref()) {
        if prior.alpha.len() != profile.log_codeword_len || prior.beta.len() != profile.beta_len {
            return Err(ReducedSwirlVaccVerifierError::RecordShape(
                "prior accumulator dimensions",
            ));
        }
        alphas.push(prior.alpha.clone());
        betas.push(prior.beta.clone());
        mus.push(prior.mu);
        etas.push(prior.eta);
        kinds.push((false, true, false));
        let mut bindings = Vec::with_capacity(profile.log_codeword_len + profile.beta_len + 2);
        bindings.extend(
            prior_schedule
                .alpha_tidx
                .iter()
                .map(|&tidx| Some((proof_idx, tidx, false))),
        );
        bindings.extend(
            prior_schedule
                .beta_tidx
                .iter()
                .map(|&tidx| Some((proof_idx, tidx, false))),
        );
        bindings.push(Some((proof_idx, prior_schedule.mu_tidx, false)));
        bindings.push(Some((proof_idx, prior_schedule.eta_tidx, false)));
        transcript_bindings.push(bindings);
    }
    // Padding is independent of prior presence.  A continuation's final
    // fresh batch may be partial, so it contains fresh claims, one prior, and
    // canonical zero slots up to the setup-fixed input arity.
    while alphas.len() < module.input_arity() {
        alphas.push(zero_alpha.clone());
        betas.push(zero_beta.clone());
        mus.push(EF::ZERO);
        etas.push(EF::ZERO);
        kinds.push((false, false, true));
        transcript_bindings.push(vec![None; profile.log_codeword_len + profile.beta_len + 2]);
    }
    let input_arity = module.input_arity();
    if alphas.len() != input_arity {
        return Err(ReducedSwirlVaccVerifierError::RecordShape(
            "exact input claim count",
        ));
    }
    let claim_inputs = (0..input_arity)
        .map(|source| NativeClaimTraceInput {
            proof_idx,
            alpha: &alphas[source],
            beta: &betas[source],
            mu: mus[source],
            eta: etas[source],
            is_fresh: kinds[source].0,
            is_prior: kinds[source].1,
            is_dummy: kinds[source].2,
            transcript_bindings: &transcript_bindings[source],
        })
        .collect::<Vec<_>>();
    let claim_values = algebra!(
        generate_native_claim_value_trace(&claim_inputs, profile.log_constraints, None),
        "claim values"
    );
    let claim_layout = algebra!(
        generate_native_claim_layout_trace(
            profile.log_codeword_len,
            profile.beta_len,
            profile.log_constraints,
            input_arity,
        ),
        "claim layout"
    );
    let shift_count = v.shifts.len();

    let claim_count = module.profile.batching_arity;
    let selector_rounds = input_arity.ilog2() as usize;
    let selector_indices = (0..input_arity)
        .map(|index| boolean_point(index, selector_rounds))
        .collect::<Vec<_>>();
    let claim_indices = (0..claim_count)
        .map(|index| boolean_point(index, claim_count.ilog2() as usize))
        .collect::<Vec<_>>();
    let mut opening_points = opening_points(v);
    let mut opening_targets = opening_targets(v);
    opening_points.resize(claim_count, opening_points[0].clone());
    opening_targets.resize(claim_count, opening_targets[0]);

    let mut vector_values = Vec::<Vec<EF>>::new();
    let mut vector_sources = Vec::<NativeVectorSource>::new();
    let mut vector_ids = Vec::<u32>::new();
    let mut vector_counts = Vec::<Vec<u32>>::new();
    let shared_twin_lookup_count = shared_twin_lookup_count(input_arity)?;
    push_vector(
        &mut vector_values,
        &mut vector_sources,
        &mut vector_ids,
        &mut vector_counts,
        module.algebra.selector_tau_vector,
        v.twin.selector_tau.clone(),
        NativeVectorSource::Transcript {
            proof_idx: proof_idx as u32,
            tidx: schedule.selector_tidx.iter().map(|&x| x as u32).collect(),
        },
        shared_twin_lookup_count,
    );
    push_vector(
        &mut vector_values,
        &mut vector_sources,
        &mut vector_ids,
        &mut vector_counts,
        module.algebra.gamma_vector,
        v.twin.gamma.clone(),
        NativeVectorSource::Sumcheck { kind: 0 },
        shared_twin_lookup_count,
    );
    push_vector(
        &mut vector_values,
        &mut vector_sources,
        &mut vector_ids,
        &mut vector_counts,
        module.algebra.xi_vector,
        v.batching.xi.clone(),
        NativeVectorSource::Transcript {
            proof_idx: proof_idx as u32,
            tidx: schedule.xi_tidx.iter().map(|&x| x as u32).collect(),
        },
        claim_count as u32,
    );
    push_vector(
        &mut vector_values,
        &mut vector_sources,
        &mut vector_ids,
        &mut vector_counts,
        module.algebra.alpha_vector,
        v.batching.alpha.clone(),
        NativeVectorSource::Sumcheck { kind: 1 },
        claim_count as u32,
    );
    for (id, values) in module
        .algebra
        .input_index_vectors
        .iter()
        .zip(&selector_indices)
    {
        push_vector(
            &mut vector_values,
            &mut vector_sources,
            &mut vector_ids,
            &mut vector_counts,
            *id,
            values.clone(),
            NativeVectorSource::Boolean,
            2,
        );
    }
    for (id, values) in module
        .algebra
        .claim_index_vectors
        .iter()
        .zip(&claim_indices)
    {
        push_vector(
            &mut vector_values,
            &mut vector_sources,
            &mut vector_ids,
            &mut vector_counts,
            *id,
            values.clone(),
            NativeVectorSource::Boolean,
            1,
        );
    }
    for (claim, (id, values)) in module
        .algebra
        .opening_point_vectors
        .iter()
        .zip(&opening_points)
        .enumerate()
    {
        push_vector(
            &mut vector_values,
            &mut vector_sources,
            &mut vector_ids,
            &mut vector_counts,
            *id,
            values.clone(),
            NativeVectorSource::OpeningPoint {
                claim: claim as u32,
            },
            1,
        );
    }
    let vector_inputs = (0..vector_values.len())
        .map(|index| NativeVectorTraceInput {
            proof_idx: proof_idx as u32,
            vector: vector_ids[index],
            values: &vector_values[index],
            source: vector_sources[index].clone(),
            lookup_counts: &vector_counts[index],
        })
        .collect::<Vec<_>>();
    let vector = algebra!(
        generate_native_vector_coordinate_trace(&vector_inputs, None),
        "vector coordinates"
    );

    let selector_result_lookups =
        (profile.log_codeword_len + profile.beta_len + shift_count) as u32;
    let mut selector_pairs = Vec::with_capacity(5);
    for (index, point) in selector_indices.iter().enumerate() {
        selector_pairs.push(NativeEqTraceInput {
            left_vector: module.algebra.selector_tau_vector,
            right_vector: module.algebra.input_index_vectors[index],
            left: &v.twin.selector_tau,
            right: point,
            lookup_count: 1,
        });
    }
    for (index, point) in selector_indices.iter().enumerate() {
        selector_pairs.push(NativeEqTraceInput {
            left_vector: module.algebra.gamma_vector,
            right_vector: module.algebra.input_index_vectors[index],
            left: &v.twin.gamma,
            right: point,
            lookup_count: selector_result_lookups,
        });
    }
    selector_pairs.push(NativeEqTraceInput {
        left_vector: module.algebra.selector_tau_vector,
        right_vector: module.algebra.gamma_vector,
        left: &v.twin.selector_tau,
        right: &v.twin.gamma,
        lookup_count: 1,
    });
    let selector_eq = algebra!(
        generate_native_eq_trace(proof_idx, &selector_pairs, 0, None),
        "selector equality"
    );
    let xi_pairs = claim_indices
        .iter()
        .enumerate()
        .map(|(claim, point)| NativeEqTraceInput {
            left_vector: module.algebra.xi_vector,
            right_vector: module.algebra.claim_index_vectors[claim],
            left: &v.batching.xi,
            right: point,
            lookup_count: 2,
        })
        .collect::<Vec<_>>();
    let (xi_eq_group_offset, opening_eq_group_offset) = vacc_eq_group_offsets(module)?;
    let xi_eq = algebra!(
        generate_native_eq_trace(proof_idx, &xi_pairs, xi_eq_group_offset, None,),
        "batch selector equality"
    );
    let opening_pairs = opening_points
        .iter()
        .enumerate()
        .map(|(claim, point)| NativeEqTraceInput {
            left_vector: module.algebra.opening_point_vectors[claim],
            right_vector: module.algebra.alpha_vector,
            left: point,
            right: &v.batching.alpha,
            lookup_count: 1,
        })
        .collect::<Vec<_>>();
    let opening_eq = algebra!(
        generate_native_eq_trace(proof_idx, &opening_pairs, opening_eq_group_offset, None,),
        "opening equality"
    );
    let selector_weights = selector_indices
        .iter()
        .map(|point| multilinear_eq(&v.twin.selector_tau, point))
        .collect::<Vec<_>>();
    let gamma_weights = selector_indices
        .iter()
        .map(|point| multilinear_eq(&v.twin.gamma, point))
        .collect::<Vec<_>>();
    let xi_weights = claim_indices
        .iter()
        .map(|point| multilinear_eq(&v.batching.xi, point))
        .collect::<Vec<_>>();
    let point_eq_alpha = opening_points
        .iter()
        .map(|point| multilinear_eq(point, &v.batching.alpha))
        .collect::<Vec<_>>();

    let omega = generate_native_twin_omega_trace(
        proof_idx,
        schedule.omega_tidx,
        v.twin.omega,
        shared_twin_lookup_count as usize,
    );
    let twin_sigma = algebra!(
        generate_native_twin_sigma_trace(
            proof_idx,
            &selector_weights,
            &mus.iter()
                .copied()
                .zip(etas.iter().copied())
                .collect::<Vec<_>>(),
            v.twin.omega,
            None,
        ),
        "twin sigma"
    );
    let sumcheck = algebra!(
        generate_native_coefficient_sumcheck_trace(
            proof_idx,
            proof_idx,
            &[&v.twin.sumcheck, &v.batching.sumcheck],
            None,
        ),
        "sumchecks"
    );
    let twin_fold = algebra!(
        generate_native_twin_fold_trace(
            proof_idx,
            &[
                (CLAIM_SECTION_ALPHA, alphas.as_slice()),
                (CLAIM_SECTION_BETA, betas.as_slice()),
            ],
            &gamma_weights,
            None,
        ),
        "twin fold"
    );
    let twin_last = v.twin.sumcheck.rounds.len().checked_sub(1).ok_or(
        ReducedSwirlVaccVerifierError::Algebra("empty twin sumcheck"),
    )?;
    let twin_pre_claim = v
        .twin
        .sumcheck
        .pre_claim_at_round(twin_last)
        .ok_or(ReducedSwirlVaccVerifierError::Algebra("twin pre-claim"))?;
    let twin_challenge = *v
        .twin
        .sumcheck
        .point
        .get(twin_last)
        .ok_or(ReducedSwirlVaccVerifierError::Algebra("twin challenge"))?;
    let twin_final = generate_native_twin_final_trace(
        twin_pre_claim,
        v.twin.sumcheck.final_claim,
        twin_challenge,
        multilinear_eq(&v.twin.selector_tau, &v.twin.gamma),
        v.twin.omega,
        v.twin.nu_0,
        v.twin.eta,
        proof_idx,
        schedule.nu_tidx,
        schedule.eta_tidx,
    );
    let initial_point = algebra!(
        generate_native_initial_opening_point_trace(proof_idx, &v.twin.zeta_0),
        "initial point"
    );
    let initial_target = generate_native_initial_opening_target_trace(proof_idx, v.twin.nu_0);
    let authenticated_claim_count = 1 + v.ood.len() + v.shifts.len();
    let opening_padding = algebra!(
        generate_native_opening_padding_trace(
            proof_idx,
            authenticated_claim_count,
            claim_count,
            &opening_points[0],
            opening_targets[0],
            None,
        ),
        "opening padding"
    );
    let ood_claims = algebra!(
        generate_native_ood_claim_trace(
            proof_idx,
            &v.ood
                .iter()
                .map(|ood| ood.point.clone())
                .collect::<Vec<_>>(),
            &schedule.ood_point_tidx,
            &v.ood.iter().map(|ood| ood.answer).collect::<Vec<_>>(),
            &schedule.ood_answer_tidx,
        ),
        "OOD claims"
    );
    let shift_samples = schedule
        .shifts
        .iter()
        .map(|sample| {
            transcript
                .values()
                .get(sample.operation_range.end.checked_sub(1)?)
                .copied()
        })
        .collect::<Option<Vec<_>>>()
        .ok_or(ReducedSwirlVaccVerifierError::Transcript(
            "accepted shift samples",
        ))?;
    let shift_tidx = schedule
        .shifts
        .iter()
        .map(|sample| sample.operation_range.end - 1)
        .collect::<Vec<_>>();
    let shift_schedule = algebra!(
        generate_reduced_swirl_vacc_shift_schedule_trace(
            proof_idx,
            &shift_samples,
            &shift_tidx,
            &v.shifts.iter().map(|shift| shift.index).collect::<Vec<_>>(),
            (fresh_claims.len() + usize::from(has_prior)) as u32,
            profile.log_codeword_len,
            None,
        ),
        "shift schedule"
    );
    let shift_answers = v
        .shifts
        .iter()
        .map(|shift| {
            let mut answers = shift.fresh_answers.clone();
            if let Some(prior) = shift.prior_answer {
                answers.push(prior);
            }
            answers.resize(input_arity, EF::ZERO);
            answers
        })
        .collect::<Vec<_>>();
    let shift_merge = algebra!(
        generate_native_shift_merge_trace(
            proof_idx,
            &gamma_weights,
            &shift_answers,
            fresh_claims.len(),
            has_prior,
            None,
        ),
        "shift merge"
    );
    let batching_sigma = algebra!(
        generate_native_batching_sigma_trace(proof_idx, &xi_weights, &opening_targets, None,),
        "batching sigma"
    );
    let batching_last = v.batching.sumcheck.rounds.len().checked_sub(1).ok_or(
        ReducedSwirlVaccVerifierError::Algebra("empty batching sumcheck"),
    )?;
    let batching_pre_claim = v
        .batching
        .sumcheck
        .pre_claim_at_round(batching_last)
        .ok_or(ReducedSwirlVaccVerifierError::Algebra("batching pre-claim"))?;
    let batching_challenge = *v
        .batching
        .sumcheck
        .point
        .get(batching_last)
        .ok_or(ReducedSwirlVaccVerifierError::Algebra("batching challenge"))?;
    let batching_final = algebra!(
        generate_native_batching_final_trace(
            &xi_weights,
            &point_eq_alpha,
            batching_pre_claim,
            v.batching.sumcheck.final_claim,
            batching_challenge,
            v.batching.mu_final,
            proof_idx,
            schedule.mu_tidx,
            None,
        ),
        "batching final"
    );
    let batching_alpha = algebra!(
        generate_native_batching_alpha_trace(proof_idx, &v.batching.alpha),
        "batching alpha"
    );
    Ok(ReducedSwirlVaccAlgebraTraces {
        claim_values,
        claim_layout,
        vector,
        selector_eq,
        xi_eq,
        opening_eq,
        omega,
        twin_sigma,
        sumcheck,
        twin_fold,
        twin_final,
        initial_point,
        initial_target,
        opening_padding,
        ood_claims,
        shift_schedule,
        shift_merge,
        batching_sigma,
        batching_final,
        batching_alpha,
    })
}

#[allow(clippy::too_many_arguments)]
fn push_vector(
    values: &mut Vec<Vec<EF>>,
    sources: &mut Vec<NativeVectorSource>,
    ids: &mut Vec<u32>,
    counts: &mut Vec<Vec<u32>>,
    id: u32,
    vector: Vec<EF>,
    source: NativeVectorSource,
    lookup_count: u32,
) {
    counts.push(vec![lookup_count; vector.len()]);
    values.push(vector);
    sources.push(source);
    ids.push(id);
}

fn boolean_point(index: usize, dimensions: usize) -> Vec<EF> {
    (0..dimensions)
        .map(|coordinate| EF::from_bool(((index >> (dimensions - 1 - coordinate)) & 1) == 1))
        .collect()
}

fn multilinear_eq(left: &[EF], right: &[EF]) -> EF {
    openvm_stark_backend::warp_pesat::eval_eq_points(left, right)
}

fn opening_points<FreshVerification>(
    v: &WarpVaccStepVerification<
        EF,
        Digest,
        FreshVerification,
        MerkleBatchOpeningVerification<EF, Digest>,
    >,
) -> Vec<Vec<EF>> {
    core::iter::once(v.twin.zeta_0.clone())
        .chain(v.ood.iter().map(|ood| ood.point.clone()))
        .chain(v.shifts.iter().map(|shift| shift.boolean_point.clone()))
        .collect()
}

fn opening_targets<FreshVerification>(
    v: &WarpVaccStepVerification<
        EF,
        Digest,
        FreshVerification,
        MerkleBatchOpeningVerification<EF, Digest>,
    >,
) -> Vec<EF> {
    core::iter::once(v.twin.nu_0)
        .chain(v.ood.iter().map(|ood| ood.answer))
        .chain(v.shifts.iter().map(|shift| shift.merged_answer))
        .collect()
}

#[derive(Clone, Debug)]
struct ReducedSwirlVaccCursorEvent {
    role: usize,
    ordinal: usize,
    tidx: usize,
    value: [F; D_EF],
    is_ext: bool,
    is_sample: bool,
}

fn push_cursor_ext(
    events: &mut Vec<ReducedSwirlVaccCursorEvent>,
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    role: usize,
    ordinal: usize,
    tidx: usize,
    is_sample: bool,
) -> Result<(), ReducedSwirlVaccVerifierError> {
    let values =
        log.values()
            .get(tidx..tidx + D_EF)
            .ok_or(ReducedSwirlVaccVerifierError::Transcript(
                "cursor extension",
            ))?;
    let samples =
        log.samples()
            .get(tidx..tidx + D_EF)
            .ok_or(ReducedSwirlVaccVerifierError::Transcript(
                "cursor extension",
            ))?;
    if samples.iter().any(|&sample| sample != is_sample) {
        return Err(ReducedSwirlVaccVerifierError::Transcript(
            "cursor extension kind",
        ));
    }
    events.push(ReducedSwirlVaccCursorEvent {
        role,
        ordinal,
        tidx,
        value: values
            .try_into()
            .map_err(|_| ReducedSwirlVaccVerifierError::Transcript("cursor extension"))?,
        is_ext: true,
        is_sample,
    });
    Ok(())
}

fn generate_reduced_swirl_vacc_cursor_trace_from_parts(
    module: &ReducedSwirlVaccVerifierModule,
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    verification: &ReducedSwirlVaccVerification,
    schedule: &ReducedSwirlVaccTranscriptSchedule,
) -> Result<RowMajorMatrix<F>, ReducedSwirlVaccVerifierError> {
    let mut events = Vec::<ReducedSwirlVaccCursorEvent>::new();
    for (ordinal, tidx) in core::iter::once(schedule.fresh_root_tidx)
        .chain(schedule.fresh_commitment_extra_tidx.iter().copied())
        .enumerate()
    {
        push_cursor_ext(&mut events, log, VACC_ROLE_FRESH_ROOT, ordinal, tidx, false)?;
    }
    for (ordinal, &tidx) in schedule.fresh_alpha_tidx.iter().enumerate() {
        push_cursor_ext(
            &mut events,
            log,
            VACC_ROLE_FRESH_ALPHA,
            ordinal,
            tidx,
            false,
        )?;
    }
    for (ordinal, &tidx) in schedule.fresh_mu_tidx.iter().enumerate() {
        push_cursor_ext(&mut events, log, VACC_ROLE_FRESH_MU, ordinal, tidx, false)?;
    }
    for (ordinal, &tidx) in schedule.fresh_beta_tail_tidx.iter().enumerate() {
        push_cursor_ext(
            &mut events,
            log,
            VACC_ROLE_FRESH_BETA_TAIL,
            ordinal,
            tidx,
            false,
        )?;
    }
    if let Some(prior) = schedule.prior.as_ref() {
        for limb in 0..DIGEST_SIZE {
            push_cursor_ext(
                &mut events,
                log,
                VACC_ROLE_PRIOR_ROOT,
                limb,
                prior.root_tidx + limb * D_EF,
                false,
            )?;
        }
        for (ordinal, &tidx) in prior.alpha_tidx.iter().enumerate() {
            push_cursor_ext(
                &mut events,
                log,
                VACC_ROLE_PRIOR_ALPHA,
                ordinal,
                tidx,
                false,
            )?;
        }
        push_cursor_ext(
            &mut events,
            log,
            VACC_ROLE_PRIOR_MU,
            0,
            prior.mu_tidx,
            false,
        )?;
        for (ordinal, &tidx) in prior.beta_tidx.iter().enumerate() {
            push_cursor_ext(&mut events, log, VACC_ROLE_PRIOR_BETA, ordinal, tidx, false)?;
        }
        push_cursor_ext(
            &mut events,
            log,
            VACC_ROLE_PRIOR_ETA,
            0,
            prior.eta_tidx,
            false,
        )?;
    }
    for (ordinal, &tidx) in schedule.fresh_tau_tidx.iter().enumerate() {
        push_cursor_ext(&mut events, log, VACC_ROLE_FRESH_TAU, ordinal, tidx, true)?;
    }
    push_cursor_ext(
        &mut events,
        log,
        VACC_ROLE_OMEGA,
        0,
        schedule.omega_tidx,
        true,
    )?;
    for (ordinal, &tidx) in schedule.selector_tidx.iter().enumerate() {
        push_cursor_ext(&mut events, log, VACC_ROLE_SELECTOR, ordinal, tidx, true)?;
    }
    for round in &verification.twin.sumcheck.rounds {
        let event_count = round.coefficients.len() + 1;
        for coefficient in 0..round.coefficients.len() {
            push_cursor_ext(
                &mut events,
                log,
                VACC_ROLE_TWIN_SUMCHECK,
                round.round as usize * event_count + coefficient,
                round.transcript_span.operation_range.start + coefficient * D_EF,
                false,
            )?;
        }
        push_cursor_ext(
            &mut events,
            log,
            VACC_ROLE_TWIN_SUMCHECK,
            round.round as usize * event_count + round.coefficients.len(),
            round.transcript_span.operation_range.end - D_EF,
            true,
        )?;
    }
    for limb in 0..DIGEST_SIZE {
        push_cursor_ext(
            &mut events,
            log,
            VACC_ROLE_OUTPUT_ROOT,
            limb,
            schedule.output_root_tidx + limb * D_EF,
            false,
        )?;
    }
    for (role, tidx) in [
        (VACC_ROLE_NU, schedule.nu_tidx),
        (VACC_ROLE_ETA, schedule.eta_tidx),
    ] {
        push_cursor_ext(&mut events, log, role, 0, tidx, false)?;
    }
    for (ood, &start) in schedule.ood_point_tidx.iter().enumerate() {
        for coordinate in 0..module.profile.log_codeword_len {
            push_cursor_ext(
                &mut events,
                log,
                VACC_ROLE_OOD_POINT,
                ood * module.profile.log_codeword_len + coordinate,
                start + coordinate * D_EF,
                true,
            )?;
        }
    }
    for (ordinal, &tidx) in schedule.ood_answer_tidx.iter().enumerate() {
        push_cursor_ext(&mut events, log, VACC_ROLE_OOD_ANSWER, ordinal, tidx, false)?;
    }
    for (shift, sample) in schedule.shifts.iter().enumerate() {
        let tidx = sample.operation_range.start;
        let value = *log
            .values()
            .get(tidx)
            .ok_or(ReducedSwirlVaccVerifierError::Transcript("shift cursor"))?;
        if sample.operation_range.end != tidx + 1 || log.samples().get(tidx).copied() != Some(true)
        {
            return Err(ReducedSwirlVaccVerifierError::Transcript(
                "shift cursor kind",
            ));
        }
        events.push(ReducedSwirlVaccCursorEvent {
            role: VACC_ROLE_SHIFT,
            ordinal: shift,
            tidx,
            value: [value, F::ZERO, F::ZERO, F::ZERO],
            is_ext: false,
            is_sample: true,
        });
    }
    for (ordinal, &tidx) in schedule.xi_tidx.iter().enumerate() {
        push_cursor_ext(&mut events, log, VACC_ROLE_XI, ordinal, tidx, true)?;
    }
    for round in &verification.batching.sumcheck.rounds {
        let event_count = round.coefficients.len() + 1;
        for coefficient in 0..round.coefficients.len() {
            push_cursor_ext(
                &mut events,
                log,
                VACC_ROLE_BATCHING_SUMCHECK,
                round.round as usize * event_count + coefficient,
                round.transcript_span.operation_range.start + coefficient * D_EF,
                false,
            )?;
        }
        push_cursor_ext(
            &mut events,
            log,
            VACC_ROLE_BATCHING_SUMCHECK,
            round.round as usize * event_count + round.coefficients.len(),
            round.transcript_span.operation_range.end - D_EF,
            true,
        )?;
    }
    push_cursor_ext(&mut events, log, VACC_ROLE_MU, 0, schedule.mu_tidx, false)?;

    let first = events
        .first()
        .ok_or(ReducedSwirlVaccVerifierError::Transcript(
            "empty VACC cursor",
        ))?;
    let last = events
        .last()
        .ok_or(ReducedSwirlVaccVerifierError::Transcript(
            "empty VACC cursor",
        ))?;
    if first.tidx != schedule.fresh_root_tidx
        || last.tidx + D_EF != schedule.end_tag_tidx
        || events
            .windows(2)
            .any(|pair| pair[1].tidx != pair[0].tidx + if pair[0].is_ext { D_EF } else { 1 })
    {
        return Err(ReducedSwirlVaccVerifierError::Transcript(
            "non-contiguous VACC cursor",
        ));
    }
    let mut lengths = [0usize; VACC_ROLE_COUNT];
    for event in &events {
        lengths[event.role] = lengths[event.role].max(event.ordinal + 1);
    }
    let height = events.len().next_power_of_two();
    let width = ReducedSwirlVaccTranscriptCursorCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    let twin_event_count = twin_degree(
        module.profile.log_codeword_len,
        module.profile.log_constraints,
        module.profile.max_degree,
    ) + 2;
    for (row, event) in events.iter().enumerate() {
        let cols: &mut ReducedSwirlVaccTranscriptCursorCols<F> =
            trace[row * width..(row + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.proof_idx = F::from_usize(schedule.proof_idx);
        cols.tidx = F::from_usize(event.tidx);
        cols.role = F::from_usize(event.role);
        cols.ordinal = F::from_usize(event.ordinal);
        cols.role_flags[event.role] = F::ONE;
        cols.is_ext = F::from_bool(event.is_ext);
        cols.is_sample = F::from_bool(event.is_sample);
        cols.is_first_ordinal = F::from_bool(event.ordinal == 0);
        cols.is_last_ordinal = F::from_bool(
            events
                .get(row + 1)
                .is_none_or(|next| next.role != event.role),
        );
        cols.ordinal_inverse = if event.ordinal == 0 {
            F::ZERO
        } else {
            F::from_usize(event.ordinal).inverse()
        };
        let distance = lengths[event.role] - 1 - event.ordinal;
        cols.last_ordinal_inverse = if distance == 0 {
            F::ZERO
        } else {
            F::from_usize(distance).inverse()
        };
        let event_count = if event.role == VACC_ROLE_TWIN_SUMCHECK {
            twin_event_count
        } else if event.role == VACC_ROLE_BATCHING_SUMCHECK {
            4
        } else {
            0
        };
        cols.is_sumcheck = F::from_bool(event_count != 0);
        if event_count != 0 {
            cols.sumcheck_round = F::from_usize(event.ordinal / event_count);
            cols.sumcheck_step = F::from_usize(event.ordinal % event_count);
            let step = event.ordinal % event_count;
            let step_distance = event_count - 1 - step;
            cols.is_last_sumcheck_step = F::from_bool(step_distance == 0);
            cols.sumcheck_last_step_inverse = if step_distance == 0 {
                F::ZERO
            } else {
                F::from_usize(step_distance).inverse()
            };
            cols.sumcheck_step_continues = F::from_bool(step_distance != 0);
        }
        let starts_proof = event.role == VACC_ROLE_FRESH_ROOT && event.ordinal == 0;
        cols.starts_proof = F::from_bool(starts_proof);
        cols.continues_proof = F::from_bool(!starts_proof);
        cols.continues_sumcheck_phase = F::from_bool(event_count != 0 && event.ordinal != 0);
        cols.value = event.value;
    }
    Ok(RowMajorMatrix::new(trace, width))
}

/// Verifier witness packet for a parent-owned Poseidon table.
///
/// `traces` matches
/// [`ReducedSwirlVaccVerifierModule::airs_without_poseidon`] exactly. The two
/// input vectors are complete for this verifier bus owner, including transcript,
/// Merkle, accumulator-digest, and optional CUDA shared-forest requests. They
/// must be inserted as one owner entry in a multi-bus Poseidon table; combining
/// their multiplicities with another owner's bus would erase namespace
/// separation.
pub struct ReducedSwirlVaccSharedPoseidonBatchTrace {
    pub traces: Vec<RowMajorMatrix<F>>,
    pub poseidon_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub poseidon_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub output_instances: Vec<AccumulatorInstance<EF, Digest>>,
    pub output_instance_digests: Vec<Digest>,
}

struct ReducedSwirlRecordTrace {
    /// Matrices after TranscriptAir, in reduced verifier AIR order. The
    /// parent-owned Poseidon table is omitted.
    traces: Vec<RowMajorMatrix<F>>,
    schedule: ReducedSwirlVaccTranscriptSchedule,
    external_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    external_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    output_instance: AccumulatorInstance<EF, Digest>,
    output_instance_digest: Digest,
    end_state: [F; POSEIDON2_WIDTH],
}

fn resumable_suffix(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    start_tidx: usize,
    expected_state: [F; POSEIDON2_WIDTH],
) -> Result<TranscriptLog<F, [F; POSEIDON2_WIDTH]>, ReducedSwirlVaccVerifierError> {
    let (event_index, event) = log
        .events()
        .iter()
        .enumerate()
        .find(|(_, event)| event.operation_range.start == start_tidx)
        .ok_or(ReducedSwirlVaccVerifierError::Transcript(
            "VACC resume event boundary",
        ))?;
    let checkpoint = TranscriptCheckpoint {
        operations: start_tidx,
        events: event_index,
        permutations: event.permutation_range.start,
    };
    if log.resumable_checkpoint_before::<CHUNK>(checkpoint) != Some(checkpoint) {
        return Err(ReducedSwirlVaccVerifierError::Transcript(
            "non-resumable VACC checkpoint",
        ));
    }
    let suffix = log
        .suffix(checkpoint)
        .ok_or(ReducedSwirlVaccVerifierError::Transcript(
            "VACC transcript suffix",
        ))?;
    if suffix.is_empty()
        || suffix.perm_results().first().copied() != Some(expected_state)
        || suffix
            .events()
            .first()
            .is_none_or(|event| event.operation_range.start != 0)
    {
        return Err(ReducedSwirlVaccVerifierError::Transcript(
            "VACC resume state",
        ));
    }
    Ok(suffix)
}

fn prefix_through(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    end_tidx: usize,
) -> Result<TranscriptLog<F, [F; POSEIDON2_WIDTH]>, ReducedSwirlVaccVerifierError> {
    let (event_index, event) = log
        .events()
        .iter()
        .enumerate()
        .find(|(_, event)| event.operation_range.end == end_tidx)
        .ok_or(ReducedSwirlVaccVerifierError::Transcript(
            "VACC call end boundary",
        ))?;
    log.prefix(TranscriptCheckpoint {
        operations: end_tidx,
        events: event_index + 1,
        permutations: event.permutation_range.end,
    })
    .ok_or(ReducedSwirlVaccVerifierError::Transcript(
        "VACC call prefix",
    ))
}

fn checkpoint_at(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    operation_index: usize,
) -> Result<TranscriptCheckpointRecord, ReducedSwirlVaccVerifierError> {
    if operation_index == 0 || operation_index > log.len() {
        return Err(ReducedSwirlVaccVerifierError::Transcript(
            "VACC checkpoint operation index",
        ));
    }
    let prefix = prefix_through(log, operation_index)?;
    let sample_count = prefix
        .samples()
        .iter()
        .rev()
        .take_while(|&&is_sample| is_sample)
        .count();
    if sample_count == 0 || sample_count > CHUNK {
        return Err(ReducedSwirlVaccVerifierError::Transcript(
            "VACC checkpoint sample count",
        ));
    }
    let state =
        prefix
            .perm_results()
            .last()
            .copied()
            .ok_or(ReducedSwirlVaccVerifierError::Transcript(
                "VACC checkpoint sponge state",
            ))?;
    Ok(TranscriptCheckpointRecord {
        operation_index: operation_index.try_into().map_err(|_| {
            ReducedSwirlVaccVerifierError::Transcript("VACC checkpoint index width")
        })?,
        sample_count: sample_count.try_into().map_err(|_| {
            ReducedSwirlVaccVerifierError::Transcript("VACC checkpoint sample width")
        })?,
        state,
    })
}

fn resumable_interval(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    start_tidx: usize,
    end_tidx: usize,
    expected_state: [F; POSEIDON2_WIDTH],
) -> Result<TranscriptLog<F, [F; POSEIDON2_WIDTH]>, ReducedSwirlVaccVerifierError> {
    let prefix = prefix_through(log, end_tidx)?;
    resumable_suffix(&prefix, start_tidx, expected_state)
}

impl ReducedSwirlVaccVerifierModule {
    /// Derive a proof-qualified exact-finite record without accepting host
    /// checkpoint metadata. The native verifier's phase spans determine the
    /// call interval; the authenticated transcript determines terminal sample
    /// counts and sponge states. Call zero uses the canonical no-resume state
    /// while its operation index remains the derived end of the schedule
    /// prefix proved in the same transcript AIR.
    /// Generate one setup-fixed reduced-SWIRL verifier batch. Every native
    /// transition remains an ordinary WARP Verify record; only the physical
    /// AIR tables are merged. Original application roots are authenticated by
    /// the direct stacked-RS opening tables and are never replaced by a scalar
    /// commitment.
    pub fn generate_reduced_swirl_traces_for_shared_poseidon(
        &self,
        config: &NativeSC,
        records: &[ReducedSwirlVaccVerifierRecord<'_>],
    ) -> Result<ReducedSwirlVaccSharedPoseidonBatchTrace, ReducedSwirlVaccVerifierError> {
        if records.is_empty() {
            return Err(ReducedSwirlVaccVerifierError::RecordShape(
                "empty reduced-SWIRL VACC batch",
            ));
        }
        // Merge each verifier record as soon as it has been generated.  A
        // reduced-SWIRL record contains several independently padded tables;
        // retaining all records and then merging them made peak host memory
        // proportional to the sum of those padded tables (more than 40 GiB
        // for a 38-call real block).  The merged AIRs need only the active
        // rows, while transcript construction needs compact checkpoints and
        // Poseidon requests.  This is the same consume-after-parent pattern
        // used by OpenVM's recursive proof tree.
        let mut merged_values: Vec<Vec<F>> = Vec::new();
        let mut merged_widths = Vec::new();
        let mut claim_layout = None;
        let mut schedule_end_tidxs = Vec::with_capacity(records.len());
        let mut end_states = Vec::with_capacity(records.len());
        let mut external_permutation_inputs = Vec::new();
        let mut external_compression_inputs = Vec::new();
        let mut output_instances = Vec::with_capacity(records.len());
        let mut output_instance_digests = Vec::with_capacity(records.len());
        let mut start_states = Vec::with_capacity(records.len());
        let has_any_prior = records.iter().any(|record| record.prior.is_some());
        // Per-record tail order begins cursor, end, claim values, then the
        // setup-fixed claim-layout table. Dynamic slot rows are owned by the
        // reduced aggregate and are intentionally absent here.
        let claim_layout_index = 3;
        for (proof_idx, record) in records.iter().enumerate() {
            if record.local_proof_idx != proof_idx
                || record.prior.is_some() != (record.proof_idx != 0)
                || (proof_idx != 0
                    && (record.batch_start_tidx != records[proof_idx - 1].vacc_end_tidx
                        || record.prior
                            != Some(&records[proof_idx - 1].verification.output_instance)))
            {
                return Err(ReducedSwirlVaccVerifierError::PriorChain);
            }
            let start_state = if record.proof_idx == 0 {
                [F::ZERO; POSEIDON2_WIDTH]
            } else {
                checkpoint_at(record.transcript, record.batch_start_tidx)?.state
            };
            if proof_idx != 0 && start_state != end_states[proof_idx - 1] {
                return Err(ReducedSwirlVaccVerifierError::PriorChain);
            }
            let generated = self.generate_reduced_swirl_record_trace(config, record)?;
            if proof_idx == 0 {
                merged_values.reserve_exact(generated.traces.len());
                merged_widths.reserve_exact(generated.traces.len());
                for matrix in &generated.traces {
                    let width = matrix.width();
                    let capacity = matrix
                        .values
                        .len()
                        .checked_mul(records.len())
                        .ok_or(ReducedSwirlVaccVerifierError::AirTraceCount)?;
                    merged_values.push(Vec::with_capacity(capacity));
                    merged_widths.push(width);
                }
            } else if generated.traces.len() != merged_values.len() {
                return Err(ReducedSwirlVaccVerifierError::AirTraceCount);
            }
            // The final nine record-local tables are, in order: output root,
            // prior adapter, prior value/hash/root, output value/hash/root,
            // and ExpBitsLen.  Validate this fixed inventory before using the
            // generic active-row compactor below.  In particular, an output
            // accumulator is never optional.
            let trailing = generated
                .traces
                .len()
                .checked_sub(9)
                .ok_or(ReducedSwirlVaccVerifierError::AirTraceCount)?;
            let prior_value_index = trailing + 2;
            let prior_hash_index = trailing + 3;
            let output_value_index = trailing + 5;
            let output_hash_index = trailing + 6;
            let output_root_index = trailing + 7;
            if generated.traces[prior_value_index].width()
                != NativeAccumulatorValueCols::<F>::width()
                || generated.traces[prior_hash_index].width()
                    != NativeAccumulatorHashCols::<F>::width()
                || generated.traces[output_value_index].width()
                    != NativeAccumulatorValueCols::<F>::width()
                || generated.traces[output_hash_index].width()
                    != NativeAccumulatorHashCols::<F>::width()
                || generated.traces[output_root_index].width()
                    != NativeAccumulatorRootDigestCols::<F>::width()
                || generated.traces[output_value_index].values.first() != Some(&F::ONE)
                || generated.traces[output_hash_index].values.first() != Some(&F::ONE)
                || generated.traces[output_root_index].values.first() != Some(&F::ONE)
            {
                return Err(ReducedSwirlVaccVerifierError::RecordShape(
                    "reduced-SWIRL accumulator trace inventory",
                ));
            }
            for (matrix_index, matrix) in generated.traces.iter().enumerate() {
                let width = *merged_widths
                    .get(matrix_index)
                    .ok_or(ReducedSwirlVaccVerifierError::AirTraceCount)?;
                if width == 0 || matrix.width() != width {
                    return Err(ReducedSwirlVaccVerifierError::AirTraceCount);
                }
                if matrix_index == claim_layout_index {
                    if proof_idx == 0 {
                        claim_layout = Some(matrix.clone());
                    }
                    continue;
                }
                let destination = merged_values
                    .get_mut(matrix_index)
                    .ok_or(ReducedSwirlVaccVerifierError::AirTraceCount)?;
                for (row_index, row) in matrix.values.chunks_exact(width).enumerate() {
                    // `NativeAccumulator{Value,Hash}Air` require a canonical
                    // first-row marker even when an optional prior table is
                    // wholly empty.  Keep that marker only when the complete
                    // physical batch has no prior.  If later records do have
                    // priors, retaining an inactive bootstrap row before
                    // their active rows would violate the table's monotone
                    // activity constraint, so the marker must be compacted
                    // away in that case.
                    let keep_empty_prior_marker = !has_any_prior
                        && row_index == 0
                        && (matrix_index == prior_value_index || matrix_index == prior_hash_index);
                    if row[0] != F::ZERO || keep_empty_prior_marker {
                        destination.extend_from_slice(row);
                    }
                }
            }
            schedule_end_tidxs.push(generated.schedule.end_tidx);
            end_states.push(generated.end_state);
            external_permutation_inputs.extend(generated.external_permutation_inputs);
            external_compression_inputs.extend(generated.external_compression_inputs);
            output_instances.push(generated.output_instance);
            output_instance_digests.push(generated.output_instance_digest);
            start_states.push(start_state);
            // `generated` and all of its padded per-record matrices are
            // released here instead of remaining live for the whole batch.
        }

        let mut owned_logs = Vec::with_capacity(records.len());
        let mut resumes = Vec::with_capacity(records.len());
        for (proof_idx, record) in records.iter().enumerate() {
            // The final physical transcript row set owns the one canonical
            // suffix after VACC as well: the authenticated manifest footer
            // followed immediately by terminal Decide/WHIR.  Footer and
            // terminal AIRs consume that suffix from the final call's proof
            // namespace.  Stopping every log at `vacc_end_tidx` would leave
            // those consumers unauthenticated (and unbalanced) even though
            // the native verifier recorded the complete transcript.
            let transcript_end = if record.is_final_call {
                record.transcript.len()
            } else {
                record.vacc_end_tidx
            };
            if record.proof_idx == 0 {
                owned_logs.push(prefix_through(record.transcript, transcript_end)?);
                resumes.push(None);
            } else {
                owned_logs.push(resumable_interval(
                    record.transcript,
                    record.batch_start_tidx,
                    transcript_end,
                    start_states[proof_idx],
                )?);
                resumes.push(Some((record.batch_start_tidx, start_states[proof_idx])));
            }
        }
        let logs = owned_logs.iter().collect::<Vec<_>>();
        let checkpoints = schedule_end_tidxs
            .iter()
            .copied()
            .map(|end_tidx| Some([None, Some(end_tidx)]))
            .collect::<Vec<_>>();
        let transcript = self
            .transcript
            .generate_trace_inputs_with_external_resumed_and_optional_checkpoints(
                &logs,
                &resumes,
                &checkpoints,
                external_permutation_inputs,
                external_compression_inputs,
                None,
            )
            .ok_or(ReducedSwirlVaccVerifierError::Transcript(
                "reduced-SWIRL transcript trace generation",
            ))?;

        let matrix_count = merged_values.len();
        if matrix_count == 0 || merged_widths.len() != matrix_count {
            return Err(ReducedSwirlVaccVerifierError::AirTraceCount);
        }
        let mut traces = Vec::with_capacity(matrix_count + 1);
        traces.push(transcript.trace);
        for matrix_index in 0..matrix_count {
            let merged = if matrix_index == claim_layout_index {
                scale_claim_layout_lookups(
                    claim_layout
                        .take()
                        .ok_or(ReducedSwirlVaccVerifierError::AirTraceCount)?,
                    records.len(),
                )
            } else {
                let width = merged_widths[matrix_index];
                let mut values = core::mem::take(&mut merged_values[matrix_index]);
                let active_rows = values.len() / width;
                let height = active_rows.max(1).next_power_of_two();
                values.resize(
                    height
                        .checked_mul(width)
                        .ok_or(ReducedSwirlVaccVerifierError::AirTraceCount)?,
                    F::ZERO,
                );
                if matrix_index + 1 == matrix_count {
                    // `ExpBitsLenAir` has a multiplicative neutral inactive
                    // row instead of the generic all-zero inactive row.
                    let exp_width = ExpBitsLenCols::<F>::width();
                    if width != exp_width {
                        return Err(ReducedSwirlVaccVerifierError::AirTraceCount);
                    }
                    for row in values[active_rows * width..].chunks_exact_mut(width) {
                        let cols: &mut ExpBitsLenCols<F> = row.borrow_mut();
                        cols.result = F::ONE;
                        cols.result_multiplier = F::ONE;
                    }
                }
                RowMajorMatrix::new(values, width)
            };
            traces.push(merged);
        }
        let airs = self.airs_without_poseidon::<NativeSC>();
        if traces.len() != airs.len()
            || traces
                .iter()
                .zip(&airs)
                .any(|(trace, air)| trace.width() != BaseAir::<F>::width(air.as_ref()))
        {
            return Err(ReducedSwirlVaccVerifierError::AirTraceCount);
        }
        let output_value_index = traces
            .len()
            .checked_sub(4)
            .ok_or(ReducedSwirlVaccVerifierError::AirTraceCount)?;
        let output_hash_index = traces.len() - 3;
        let output_root_index = traces.len() - 2;
        if traces[output_value_index].values.first() != Some(&F::ONE)
            || traces[output_hash_index].values.first() != Some(&F::ONE)
            || traces[output_root_index].values.first() != Some(&F::ONE)
        {
            return Err(ReducedSwirlVaccVerifierError::RecordShape(
                "empty reduced-SWIRL output accumulator batch",
            ));
        }
        Ok(ReducedSwirlVaccSharedPoseidonBatchTrace {
            traces,
            poseidon_permutation_inputs: transcript.permutation_inputs,
            poseidon_compression_inputs: transcript.compression_inputs,
            output_instances,
            output_instance_digests,
        })
    }

    fn generate_reduced_swirl_record_trace(
        &self,
        config: &NativeSC,
        record: &ReducedSwirlVaccVerifierRecord<'_>,
    ) -> Result<ReducedSwirlRecordTrace, ReducedSwirlVaccVerifierError> {
        let reduced = &self.reduced_swirl;
        let proof = record.proof.inner();
        let local_proof_idx = record.local_proof_idx;
        let fresh_count = proof.fresh_claims.len();
        let expected_kinds = (0..reduced.input_arity)
            .map(|slot| {
                if slot < fresh_count {
                    WarpStepInputKind::Fresh
                } else if record.prior.is_some() && slot == fresh_count {
                    WarpStepInputKind::PriorAccumulator
                } else {
                    WarpStepInputKind::Padding
                }
            })
            .collect::<Vec<_>>();
        if proof.input_kinds != expected_kinds
            || proof.output_instance != record.verification.output_instance
            || proof.openings.fresh_opening_proofs.len() != fresh_count
            || proof.openings.acc_opening_proofs.len() != usize::from(record.prior.is_some())
            || record.verification.prior_authentication.is_some() != record.prior.is_some()
            || record.vacc_start_tidx <= record.batch_start_tidx
            || record.vacc_end_tidx <= record.vacc_start_tidx
        {
            return Err(ReducedSwirlVaccVerifierError::RecordShape(
                "reduced-SWIRL proof envelope",
            ));
        }
        let schedule = ReducedSwirlVaccTranscriptSchedule::from_reduced_swirl_record(self, record)?;
        let cursor = generate_reduced_swirl_vacc_cursor_trace_from_parts(
            self,
            record.transcript,
            record.verification,
            &schedule,
        )?;
        let fresh_claims = proof
            .fresh_claims
            .iter()
            .map(ReducedSwirlVaccFreshClaimData::from)
            .collect::<Vec<_>>();
        let algebra = generate_reduced_swirl_vacc_algebra_traces_from_parts(
            self,
            local_proof_idx,
            &fresh_claims,
            record.verification,
            record.transcript,
            record.prior,
            &schedule,
        )?;
        let root_tree_stride = 1 + self.profile.num_shift_queries;
        let fresh_opening = generate_native_direct_fresh_opening_traces(
            config.hasher(),
            &proof
                .fresh_claims
                .iter()
                .map(|claim| claim.commitment.clone())
                .collect::<Vec<_>>(),
            &record.verification.fresh_authentication,
            &record.commitment_tidxs,
            local_proof_idx,
            reduced.input_arity,
            reduced.max_roots_per_source,
            0,
            u32::try_from(root_tree_stride)
                .map_err(|_| ReducedSwirlVaccVerifierError::InvalidProfile)?,
            reduced.projection_sources_per_shard,
            reduced.max_projection_height,
        )
        .ok_or(ReducedSwirlVaccVerifierError::Merkle(
            "reduced-SWIRL original-root openings",
        ))?;
        let prior_projection = record
            .verification
            .prior_authentication
            .as_ref()
            .map(|authentication| {
                generate_native_accumulator_projection_trace_checked(
                    config.hasher(),
                    local_proof_idx,
                    authentication,
                    self.profile.log_codeword_len,
                    self.profile.rows_per_query,
                    self.trees.prior_rows,
                    u32::try_from(self.trees.prior_outer)
                        .map_err(|_| ReducedSwirlVaccVerifierError::InvalidProfile)?,
                    fresh_count,
                    reduced.input_arity,
                    None,
                )
                .map_err(ReducedSwirlVaccVerifierError::Merkle)
            })
            .transpose()?;

        let mut leaf_hash_matrices = vec![fresh_opening.leaf_hash.matrix];
        let mut merkle_matrices = vec![fresh_opening.merkle];
        let mut adapter_matrices = vec![fresh_opening.leaf_adapter];
        let mut permutation_inputs = fresh_opening.leaf_hash.permutation_inputs;
        let mut compression_inputs = fresh_opening.compression_inputs;
        if let (Some(projection), Some(authentication)) = (
            prior_projection.as_ref(),
            record.verification.prior_authentication.as_ref(),
        ) {
            self.collect_projection_authentication(
                config,
                local_proof_idx,
                projection,
                authentication,
                self.trees.prior_outer,
                &mut leaf_hash_matrices,
                &mut merkle_matrices,
                &mut adapter_matrices,
                &mut permutation_inputs,
                &mut compression_inputs,
            )?;
        }
        let leaf_hash = merge_active_rows(&leaf_hash_matrices, NativeLeafHashCols::<F>::width())
            .ok_or(ReducedSwirlVaccVerifierError::Merkle(
                "reduced-SWIRL leaf hash merge",
            ))?;
        let merkle = merge_active_rows(&merkle_matrices, NativeMerkleCompressionCols::<F>::width())
            .ok_or(ReducedSwirlVaccVerifierError::Merkle(
                "reduced-SWIRL Merkle merge",
            ))?;
        let leaf_adapter =
            merge_active_rows(&adapter_matrices, NativeMerkleLeafAdapterCols::<F>::width()).ok_or(
                ReducedSwirlVaccVerifierError::Merkle("reduced-SWIRL leaf-adapter merge"),
            )?;

        let prior_digest = record
            .prior
            .map(|prior| {
                generate_native_accumulator_digest_traces(
                    local_proof_idx,
                    prior,
                    &self.private_accumulator,
                )
                .ok_or(ReducedSwirlVaccVerifierError::RecordShape(
                    "reduced-SWIRL prior accumulator digest",
                ))
            })
            .transpose()?;
        let output_digest = generate_native_accumulator_digest_traces(
            local_proof_idx,
            &record.verification.output_instance,
            &self.private_accumulator,
        )
        .ok_or(ReducedSwirlVaccVerifierError::RecordShape(
            "reduced-SWIRL output accumulator digest",
        ))?;
        append_accumulator_poseidon(
            prior_digest.as_ref(),
            &mut permutation_inputs,
            &mut compression_inputs,
        );
        append_accumulator_poseidon(
            Some(&output_digest),
            &mut permutation_inputs,
            &mut compression_inputs,
        );

        let end = generate_reduced_swirl_vacc_end_trace(
            local_proof_idx,
            schedule.end_tag_tidx,
            schedule.end_tidx,
            schedule.discarded_end_sample,
        );
        let prior_root = schedule.prior.as_ref().map(|prior| {
            generate_native_standard_vacc_commitment_root_trace(
                local_proof_idx,
                prior.root_tidx,
                record.prior.expect("validated reduced prior").rt,
            )
        });
        let output_root = generate_native_standard_vacc_commitment_root_trace(
            local_proof_idx,
            schedule.output_root_tidx,
            record.verification.output_instance.rt,
        );
        let prior_adapter = generate_reduced_swirl_prior_claim_adapter_trace(
            &record
                .prior
                .map(|prior| vec![(local_proof_idx, fresh_count, prior)])
                .unwrap_or_default(),
            reduced.input_arity,
            &self.private_accumulator,
        )?;
        let exp_bits = generate_exp_bits_trace_from_log(
            record.transcript,
            &schedule,
            self.profile.log_codeword_len,
        )?;

        let empty_prior = empty_accumulator_digest_traces(&self.private_accumulator);
        let prior_digest = prior_digest.unwrap_or(empty_prior);
        let empty_projection = RowMajorMatrix::new(
            F::zero_vec(NativeAccumulatorProjectionCols::<F>::width() * 2),
            NativeAccumulatorProjectionCols::<F>::width(),
        );
        let empty_root = RowMajorMatrix::new(
            F::zero_vec(NativeStandardVaccCommitmentRootCols::<F>::width() * 2),
            NativeStandardVaccCommitmentRootCols::<F>::width(),
        );
        let mut traces = vec![
            cursor,
            end,
            algebra.claim_values,
            algebra.claim_layout,
            fresh_opening.commitment,
            fresh_opening.roots,
        ];
        traces.extend(fresh_opening.projections);
        traces.push(prior_projection.map_or(empty_projection, |projection| projection.matrix));
        traces.push(prior_root.unwrap_or(empty_root));
        traces.extend([leaf_hash, merkle, leaf_adapter]);
        traces.extend([
            algebra.vector,
            algebra.selector_eq,
            algebra.xi_eq,
            algebra.opening_eq,
            algebra.omega,
            algebra.twin_sigma,
            algebra.sumcheck,
            algebra.twin_fold,
            algebra.twin_final,
            algebra.initial_point,
            algebra.initial_target,
            algebra.opening_padding,
            algebra.ood_claims,
            algebra.shift_schedule,
            algebra.shift_merge,
            algebra.batching_sigma,
            algebra.batching_final,
            algebra.batching_alpha,
            output_root,
            prior_adapter,
            prior_digest.values,
            prior_digest.hash,
            prior_digest.root,
            output_digest.values,
            output_digest.hash,
            output_digest.root,
            exp_bits,
        ]);
        if traces.len() + 1 != self.airs_without_poseidon::<NativeSC>().len() {
            return Err(ReducedSwirlVaccVerifierError::AirTraceCount);
        }
        let end_state = checkpoint_at(record.transcript, record.vacc_end_tidx)?.state;
        Ok(ReducedSwirlRecordTrace {
            traces,
            schedule,
            external_permutation_inputs: permutation_inputs,
            external_compression_inputs: compression_inputs,
            output_instance: record.verification.output_instance.clone(),
            output_instance_digest: output_digest.instance_digest,
            end_state,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_projection_authentication(
        &self,
        config: &NativeSC,
        proof_idx: usize,
        projection: &NativeAccumulatorProjectionTrace,
        authentication: &MerkleBatchOpeningVerification<EF, Digest>,
        outer_tree_id: usize,
        leaf_hash_matrices: &mut Vec<RowMajorMatrix<F>>,
        merkle_matrices: &mut Vec<RowMajorMatrix<F>>,
        adapter_matrices: &mut Vec<RowMajorMatrix<F>>,
        permutation_inputs: &mut Vec<[F; POSEIDON2_WIDTH]>,
        compression_inputs: &mut Vec<[F; POSEIDON2_WIDTH]>,
    ) -> Result<(), ReducedSwirlVaccVerifierError> {
        self.collect_projection_authentication_record(
            config,
            proof_idx,
            projection,
            &authentication.multiproof,
            outer_tree_id,
            leaf_hash_matrices,
            merkle_matrices,
            adapter_matrices,
            permutation_inputs,
            compression_inputs,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_projection_authentication_record(
        &self,
        _config: &NativeSC,
        proof_idx: usize,
        projection: &NativeAccumulatorProjectionTrace,
        outer_multiproof: &BinaryMerkleMultiproofRecord<Digest>,
        outer_tree_id: usize,
        leaf_hash_matrices: &mut Vec<RowMajorMatrix<F>>,
        merkle_matrices: &mut Vec<RowMajorMatrix<F>>,
        adapter_matrices: &mut Vec<RowMajorMatrix<F>>,
        permutation_inputs: &mut Vec<[F; POSEIDON2_WIDTH]>,
        compression_inputs: &mut Vec<[F; POSEIDON2_WIDTH]>,
    ) -> Result<(), ReducedSwirlVaccVerifierError> {
        let leaf = generate_native_leaf_hash_trace(&projection.leaf_hash_inputs(), None)
            .ok_or(ReducedSwirlVaccVerifierError::Merkle("leaf hash"))?;
        permutation_inputs.extend(leaf.permutation_inputs.iter().copied());
        leaf_hash_matrices.push(leaf.matrix);
        let mut records = projection
            .inner_merkle
            .iter()
            .map(|(tree, proof)| (*tree, proof))
            .collect::<Vec<_>>();
        records.push((
            u32::try_from(outer_tree_id)
                .map_err(|_| ReducedSwirlVaccVerifierError::Merkle("outer tree id"))?,
            outer_multiproof,
        ));
        compression_inputs.extend(merkle_compression_inputs(&records));
        if records
            .iter()
            .any(|(_, proof)| !proof.compressions.is_empty())
        {
            merkle_matrices.push(
                generate_native_merkle_multiproof_trace(proof_idx, &records, None)
                    .ok_or(ReducedSwirlVaccVerifierError::Merkle("Merkle multiproof"))?,
            );
        }
        if !projection.leaf_adapters.is_empty() {
            adapter_matrices.push(
                generate_native_merkle_leaf_adapter_trace(
                    &projection.leaf_adapters,
                    self.profile.rows_per_query.ilog2() as usize,
                    None,
                )
                .ok_or(ReducedSwirlVaccVerifierError::Merkle("leaf adapter"))?,
            );
        }
        Ok(())
    }
}

fn generate_exp_bits_trace_from_log(
    transcript: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    schedule: &ReducedSwirlVaccTranscriptSchedule,
    log_codeword_len: usize,
) -> Result<RowMajorMatrix<F>, ReducedSwirlVaccVerifierError> {
    let generator = ExpBitsLenCpuTraceGenerator::default();
    for sample in &schedule.shifts {
        let accepted = *transcript
            .values()
            .get(sample.operation_range.end - 1)
            .ok_or(ReducedSwirlVaccVerifierError::Transcript(
                "shift exponent request",
            ))?;
        generator.add_requests_with_shift([(F::ONE, accepted, 0, log_codeword_len, 1)]);
    }
    generator
        .generate_trace_row_major(None)
        .ok_or(ReducedSwirlVaccVerifierError::Algebra(
            "shift exponent table",
        ))
}

fn append_accumulator_poseidon(
    traces: Option<&NativeAccumulatorDigestTraces>,
    permutations: &mut Vec<[F; POSEIDON2_WIDTH]>,
    compressions: &mut Vec<[F; POSEIDON2_WIDTH]>,
) {
    if let Some(traces) = traces {
        permutations.extend(traces.poseidon2_permute_inputs.iter().copied());
        compressions.extend(traces.poseidon2_compress_inputs.iter().copied());
    }
}

fn merkle_compression_inputs(
    records: &[(u32, &BinaryMerkleMultiproofRecord<Digest>)],
) -> Vec<[F; POSEIDON2_WIDTH]> {
    records
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
        .collect()
}

fn merge_active_rows(matrices: &[RowMajorMatrix<F>], width: usize) -> Option<RowMajorMatrix<F>> {
    merge_active_row_refs(matrices.iter(), width)
}

fn merge_active_row_refs<'a>(
    matrices: impl IntoIterator<Item = &'a RowMajorMatrix<F>>,
    width: usize,
) -> Option<RowMajorMatrix<F>> {
    let matrices = matrices.into_iter().collect::<Vec<_>>();
    if width == 0 || matrices.is_empty() || matrices.iter().any(|matrix| matrix.width() != width) {
        return None;
    }
    let active = matrices
        .iter()
        .flat_map(|matrix| matrix.values.chunks_exact(width))
        .filter(|row| row[0] != F::ZERO)
        .count();
    let height = active.max(1).next_power_of_two();
    let mut values = Vec::with_capacity(height * width);
    for matrix in matrices {
        for row in matrix.values.chunks_exact(width) {
            if row[0] != F::ZERO {
                values.extend_from_slice(row);
            }
        }
    }
    values.resize(height * width, F::ZERO);
    Some(RowMajorMatrix::new(values, width))
}

fn scale_claim_layout_lookups(mut matrix: RowMajorMatrix<F>, factor: usize) -> RowMajorMatrix<F> {
    let width = NativeClaimLayoutCols::<F>::width();
    let factor = F::from_usize(factor);
    for row in matrix.values.chunks_exact_mut(width) {
        let cols: &mut NativeClaimLayoutCols<F> = row.borrow_mut();
        cols.lookup_count *= factor;
    }
    matrix
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ReducedSwirlPriorClaimAdapterCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub source: T,
    pub variant: T,
    pub section: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(ReducedSwirlPriorClaimAdapterCols<u8>)]
pub struct ReducedSwirlPriorClaimAdapterAir {
    pub claim_bus: NativeClaimValueBus,
    pub slot_bus: NativeInputSlotLayoutBus,
    pub virtual_source: usize,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ReducedSwirlVaccEndCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub cursor_end_tidx: T,
    pub end_tidx: T,
    pub discarded_sample: [T; D_EF],
}

/// The semantic VACC cursor ends at the final target; the authenticated call
/// endpoint additionally includes the resumable observe-and-sample boundary.
#[derive(ColumnsAir)]
#[columns_via(ReducedSwirlVaccEndCols<u8>)]
pub struct ReducedSwirlVaccEndAir {
    pub transcript_bus: openvm_recursion_circuit::bus::TranscriptBus,
    pub phase_cursor_bus: NativeVaccPhaseCursorBus,
    pub end_bus: NativeStandardVaccEndBus,
    pub end_tag: u64,
}

impl BaseAir<F> for ReducedSwirlVaccEndAir {
    fn width(&self) -> usize {
        ReducedSwirlVaccEndCols::<F>::width()
    }
}
impl openvm_stark_backend::BaseAirWithPublicValues<F> for ReducedSwirlVaccEndAir {}
impl openvm_stark_backend::PartitionedBaseAir<F> for ReducedSwirlVaccEndAir {}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for ReducedSwirlVaccEndAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("reduced VACC end row");
        let local: &ReducedSwirlVaccEndCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        self.phase_cursor_bus.receive(
            builder,
            NativeVaccPhaseCursorMessage {
                proof_idx: local.proof_idx.into(),
                boundary: AB::Expr::from_usize(2),
                tidx: local.cursor_end_tidx.into(),
            },
            local.active,
        );
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.cursor_end_tidx,
            core::array::from_fn(|limb| {
                if limb == 0 {
                    AB::Expr::from_u64(self.end_tag)
                } else {
                    AB::Expr::ZERO
                }
            }),
            local.active,
        );
        let sample_tidx = AB::Expr::from(local.cursor_end_tidx) + AB::Expr::from_usize(D_EF);
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            sample_tidx.clone(),
            local.discarded_sample,
            local.active,
        );
        builder
            .when(local.active)
            .assert_eq(local.end_tidx, sample_tidx + AB::Expr::from_usize(D_EF));
        self.end_bus.add_key_with_lookups(
            builder,
            NativeStandardVaccEndMessage {
                proof_idx: local.proof_idx.into(),
                end_tidx: local.end_tidx.into(),
            },
            local.active,
        );
    }
}

impl BaseAir<F> for ReducedSwirlPriorClaimAdapterAir {
    fn width(&self) -> usize {
        ReducedSwirlPriorClaimAdapterCols::<F>::width()
    }
}
impl openvm_stark_backend::BaseAirWithPublicValues<F> for ReducedSwirlPriorClaimAdapterAir {}
impl openvm_stark_backend::PartitionedBaseAir<F> for ReducedSwirlPriorClaimAdapterAir {}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for ReducedSwirlPriorClaimAdapterAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("reduced prior claim adapter row");
        let local: &ReducedSwirlPriorClaimAdapterCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        self.slot_bus.lookup_key(
            builder,
            NativeInputSlotLayoutMessage {
                variant: local.variant.into(),
                source: local.source.into(),
                kind: [AB::Expr::ZERO, AB::Expr::ONE, AB::Expr::ZERO],
            },
            local.active,
        );
        let dynamic = NativeClaimValueMessage {
            proof_idx: local.proof_idx.into(),
            source: local.source.into(),
            section: local.section.into(),
            coordinate: local.coordinate.into(),
            value: local.value.map(Into::into),
        };
        self.claim_bus.receive(builder, dynamic, local.active);
        self.claim_bus.send(
            builder,
            NativeClaimValueMessage {
                proof_idx: local.proof_idx.into(),
                source: AB::Expr::from_usize(self.virtual_source),
                section: local.section.into(),
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active,
        );
    }
}

fn generate_reduced_swirl_prior_claim_adapter_trace(
    records: &[(usize, usize, &AccumulatorInstance<EF, Digest>)],
    input_arity: usize,
    layout: &NativePrivateAccumulatorLayout,
) -> Result<RowMajorMatrix<F>, ReducedSwirlVaccVerifierError> {
    let width = ReducedSwirlPriorClaimAdapterCols::<F>::width();
    let rows_per_prior = layout.alpha_len + layout.beta_len + 2;
    let active_rows = records
        .len()
        .checked_mul(rows_per_prior)
        .ok_or(ReducedSwirlVaccVerifierError::AirTraceCount)?;
    let height = active_rows.max(1).next_power_of_two();
    let mut values = F::zero_vec(height * width);
    let mut output_row = 0usize;
    for &(proof_idx, fresh_count, prior) in records {
        if fresh_count + 1 > input_arity {
            return Err(ReducedSwirlVaccVerifierError::RecordShape(
                "reduced prior slot",
            ));
        }
        let digest = generate_native_accumulator_digest_traces(proof_idx, prior, layout).ok_or(
            ReducedSwirlVaccVerifierError::RecordShape("reduced prior accumulator dimensions"),
        )?;
        for row in 0..digest.values.height() {
            let source = digest
                .values
                .row_slice(row)
                .ok_or(ReducedSwirlVaccVerifierError::AirTraceCount)?;
            let source: &NativeAccumulatorValueCols<F> = (*source).borrow();
            if source.active == F::ZERO {
                continue;
            }
            let target: &mut ReducedSwirlPriorClaimAdapterCols<F> =
                values[output_row * width..(output_row + 1) * width].borrow_mut();
            target.active = F::ONE;
            target.proof_idx = F::from_usize(proof_idx);
            target.source = F::from_usize(fresh_count);
            target.variant = F::from_usize(fresh_count + input_arity + 1);
            target.section = source.section[2]
                + source.section[1] * F::from_usize(CLAIM_SECTION_MU)
                + source.section[3] * F::from_usize(CLAIM_SECTION_ETA);
            target.coordinate = source.coordinate;
            target.value = source.value;
            output_row += 1;
        }
    }
    if output_row != active_rows {
        return Err(ReducedSwirlVaccVerifierError::AirTraceCount);
    }
    Ok(RowMajorMatrix::new(values, width))
}

fn generate_reduced_swirl_vacc_end_trace(
    proof_idx: usize,
    cursor_end_tidx: usize,
    end_tidx: usize,
    discarded_sample: EF,
) -> RowMajorMatrix<F> {
    let width = ReducedSwirlVaccEndCols::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut ReducedSwirlVaccEndCols<F> = values.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.proof_idx = F::from_usize(proof_idx);
    cols.cursor_end_tidx = F::from_usize(cursor_end_tidx);
    cols.end_tidx = F::from_usize(end_tidx);
    cols.discarded_sample
        .copy_from_slice(discarded_sample.as_basis_coefficients_slice());
    RowMajorMatrix::new(values, width)
}

/// Canonical inactive marker for the optional prior tables. The marker is not
/// an unconstrained all-zero witness: the value/hash AIRs still require their
/// first-row framing columns, while every lookup-producing row stays inactive.
/// Output tables never use this helper and remain non-empty by construction.
fn empty_accumulator_digest_traces(
    layout: &NativePrivateAccumulatorLayout,
) -> NativeAccumulatorDigestTraces {
    let value_width = NativeAccumulatorValueCols::<F>::width();
    let mut value_rows = F::zero_vec(value_width * 2);
    let value: &mut NativeAccumulatorValueCols<F> = value_rows[..value_width].borrow_mut();
    value.is_first = F::ONE;
    value.section[0] = F::ONE;

    let hash_width = NativeAccumulatorHashCols::<F>::width();
    let mut hash_rows = F::zero_vec(hash_width * 2);
    let hash: &mut NativeAccumulatorHashCols<F> = hash_rows[..hash_width].borrow_mut();
    hash.is_first = F::ONE;

    let root_width = NativeAccumulatorRootDigestCols::<F>::width();
    let _ = layout;
    NativeAccumulatorDigestTraces {
        values: RowMajorMatrix::new(value_rows, value_width),
        hash: RowMajorMatrix::new(hash_rows, hash_width),
        root: RowMajorMatrix::new(F::zero_vec(root_width * 2), root_width),
        poseidon2_permute_inputs: Vec::new(),
        poseidon2_compress_inputs: Vec::new(),
        instance_digest: [F::ZERO; DIGEST_SIZE],
    }
}

/// Value binder for one prior/output accumulator instance. It uses the same
/// canonical digest preimage as terminal Decide and reads the standard-VACC
/// claim and algebra buses directly.
pub struct ReducedSwirlAccumulatorValueAir {
    pub mode: NativeAccumulatorBindingMode,
    pub allow_empty: bool,
    pub layout: NativePrivateAccumulatorLayout,
    /// Prior accumulators occupy the final input slot; its setup-fixed index
    /// depends on the configured input arity.
    pub claim_source: usize,
    pub digest_element_bus: NativeAccumulatorDigestElementBus,
    pub claim_bus: NativeClaimValueBus,
    pub batching_bus: NativeBatchingOutputBus,
    pub folded_bus: NativeFoldedClaimBus,
    pub twin_bus: NativeTwinScalarBus,
}

impl BaseAir<F> for ReducedSwirlAccumulatorValueAir {
    fn width(&self) -> usize {
        NativeAccumulatorValueCols::<F>::width()
    }
}
impl openvm_stark_backend::BaseAirWithPublicValues<F> for ReducedSwirlAccumulatorValueAir {}
impl openvm_stark_backend::PartitionedBaseAir<F> for ReducedSwirlAccumulatorValueAir {}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for ReducedSwirlAccumulatorValueAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("direct accumulator value row");
        let next_row = main
            .row_slice(1)
            .expect("direct accumulator value next row");
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
        builder
            .when(local.active)
            .assert_one(is_alpha.clone() + is_mu.clone() + is_beta.clone() + is_eta.clone());
        let section_sum = is_alpha.clone() + is_mu.clone() + is_beta.clone() + is_eta.clone();
        if self.allow_empty {
            // The canonical empty bootstrap marker retains only the mandatory
            // first-row framing bit. Every later inactive padding row is all
            // zero. This is verifier-key-selected, not a host acceptance bit.
            builder
                .when(AB::Expr::ONE - local.active)
                .assert_eq(section_sum, local.is_first);
        } else {
            builder
                .when(AB::Expr::ONE - local.active)
                .assert_zero(section_sum);
        }
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
        if !self.allow_empty {
            builder.when_first_row().assert_one(local.active);
        }
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
        let mut same_proof = transition.when(next.active * (AB::Expr::ONE - next.is_first));
        same_proof.assert_eq(next.proof_idx, local.proof_idx);
        same_proof.assert_eq(next.ordinal, local.ordinal + AB::F::ONE);
        let mut transition = builder.when_transition();
        let mut next_proof = transition.when(next.active * next.is_first);
        next_proof.assert_one(local.is_last);
        next_proof.assert_eq(next.proof_idx, local.proof_idx + AB::F::ONE);
        next_proof.assert_zero(next.ordinal);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);
        let continue_section =
            next.active * (AB::Expr::ONE - next.is_first) * (AB::Expr::ONE - local.section_last);
        let mut transition = builder.when_transition();
        let mut same = transition.when(continue_section);
        for index in 0..4 {
            same.assert_eq(next.section[index], local.section[index]);
        }
        same.assert_eq(next.coordinate, local.coordinate + AB::F::ONE);
        same.assert_eq(next.remaining, local.remaining - AB::F::ONE);
        let advance_section = next.active * (AB::Expr::ONE - next.is_first) * local.section_last;
        let mut transition = builder.when_transition();
        let mut advance = transition.when(advance_section);
        advance.assert_zero(next.coordinate);
        advance.assert_eq(next.section[0], AB::Expr::ZERO);
        advance.assert_eq(next.section[1], is_alpha.clone());
        advance.assert_eq(next.section[2], is_mu.clone());
        advance.assert_eq(next.section[3], is_beta.clone());

        let state = match self.mode {
            NativeAccumulatorBindingMode::Prior => 0,
            NativeAccumulatorBindingMode::Output => 1,
        };
        for limb in 0..D_EF {
            self.digest_element_bus.send(
                builder,
                NativeAccumulatorDigestElementMessage {
                    proof_idx: local.proof_idx.into(),
                    state: AB::Expr::from_usize(state),
                    index: AB::Expr::from_usize(3)
                        + local.ordinal * AB::Expr::from_usize(D_EF)
                        + AB::Expr::from_usize(limb),
                    value: local.value[limb].into(),
                },
                local.active,
            );
        }
        let section = is_beta.clone()
            + is_mu.clone() * AB::Expr::from_usize(CLAIM_SECTION_MU)
            + is_eta.clone() * AB::Expr::from_usize(CLAIM_SECTION_ETA);
        match self.mode {
            NativeAccumulatorBindingMode::Prior => self.claim_bus.receive(
                builder,
                NativeClaimValueMessage {
                    proof_idx: local.proof_idx.into(),
                    source: AB::Expr::from_usize(self.claim_source),
                    section,
                    coordinate: local.coordinate.into(),
                    value: local.value.map(Into::into),
                },
                local.active,
            ),
            NativeAccumulatorBindingMode::Output => {
                self.batching_bus.receive(
                    builder,
                    NativeBatchingOutputMessage {
                        proof_idx: local.proof_idx.into(),
                        section: AB::Expr::from_usize(BATCHING_OUTPUT_ALPHA),
                        coordinate: local.coordinate.into(),
                        value: local.value.map(Into::into),
                    },
                    local.active * is_alpha.clone(),
                );
                self.batching_bus.receive(
                    builder,
                    NativeBatchingOutputMessage {
                        proof_idx: local.proof_idx.into(),
                        section: AB::Expr::from_usize(BATCHING_OUTPUT_MU),
                        coordinate: AB::Expr::ZERO,
                        value: local.value.map(Into::into),
                    },
                    local.active * is_mu.clone(),
                );
                self.folded_bus.receive(
                    builder,
                    NativeFoldedClaimMessage {
                        proof_idx: local.proof_idx.into(),
                        section: AB::Expr::from_usize(CLAIM_SECTION_BETA),
                        coordinate: local.coordinate.into(),
                        value: local.value.map(Into::into),
                    },
                    local.active * is_beta.clone(),
                );
                self.twin_bus.receive(
                    builder,
                    NativeTwinScalarMessage {
                        proof_idx: local.proof_idx.into(),
                        kind: AB::Expr::from_usize(TWIN_SCALAR_ETA),
                        value: local.value.map(Into::into),
                    },
                    local.active * is_eta.clone(),
                );
            }
        }
    }
}

pub struct ReducedSwirlAccumulatorRootDigestAir {
    pub mode: NativeAccumulatorBindingMode,
    pub allow_empty: bool,
    pub root_bus: NativeAccumulatorRootBus,
    pub algebraic_digest_bus: NativeAccumulatorAlgebraicDigestBus,
    pub compress_bus: openvm_recursion_circuit::bus::Poseidon2CompressBus,
    pub digest_bus: NativeStandardVaccDigestBus,
}

impl BaseAir<F> for ReducedSwirlAccumulatorRootDigestAir {
    fn width(&self) -> usize {
        NativeAccumulatorRootDigestCols::<F>::width()
    }
}
impl openvm_stark_backend::BaseAirWithPublicValues<F> for ReducedSwirlAccumulatorRootDigestAir {}
impl openvm_stark_backend::PartitionedBaseAir<F> for ReducedSwirlAccumulatorRootDigestAir {}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for ReducedSwirlAccumulatorRootDigestAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("direct accumulator digest row");
        let local: &NativeAccumulatorRootDigestCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        if !self.allow_empty {
            builder.when_first_row().assert_one(local.active);
        }
        let state = match self.mode {
            NativeAccumulatorBindingMode::Prior => 0,
            NativeAccumulatorBindingMode::Output => 1,
        };
        self.algebraic_digest_bus.receive(
            builder,
            NativeAccumulatorAlgebraicDigestMessage {
                proof_idx: local.proof_idx.into(),
                state: AB::Expr::from_usize(state),
                digest: local.algebraic_digest.map(Into::into),
            },
            local.active,
        );
        self.root_bus.receive(
            builder,
            NativeAccumulatorRootMessage {
                proof_idx: local.proof_idx.into(),
                state: AB::Expr::from_usize(state),
                digest: local.root.map(Into::into),
            },
            local.active,
        );
        self.compress_bus.lookup_key(
            builder,
            openvm_recursion_circuit::bus::Poseidon2CompressMessage {
                input: core::array::from_fn(|index| {
                    if index < DIGEST_SIZE {
                        local.root[index].into()
                    } else {
                        local.algebraic_digest[index - DIGEST_SIZE].into()
                    }
                }),
                output: local.instance_digest.map(Into::into),
            },
            local.active,
        );
        self.digest_bus.add_key_with_lookups(
            builder,
            NativeStandardVaccDigestMessage {
                proof_idx: local.proof_idx.into(),
                state: AB::Expr::from_usize(state),
                digest: local.instance_digest.map(Into::into),
            },
            local.active,
        );
    }
}

const _: () = assert!(POSEIDON2_WIDTH == 2 * DIGEST_SIZE);

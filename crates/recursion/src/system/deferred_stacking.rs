//! Typed statement boundary for a recursive verifier that stops before SWIRL stacking.
//!
//! AIR and LogUp are fully reduced by the ordinary GKR and batch-constraint modules. The narrow
//! claims AIR below binds every resulting column claim to both the verifier computation and the
//! transcript. The checkpoint AIR then publishes the exact transcript endpoint. A terminal
//! block-wide stacking/WHIR proof replays the same retained reduction, checks this endpoint, and
//! opens the original commitments. Neither AIR is a PCS-opening predicate.

use openvm_circuit_primitives::utils::not;
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{
    interaction::InteractionBuilder, keygen::types::MultiStarkVerifyingKey, proof::Proof,
    BaseAirWithPublicValues, PartitionedBaseAir, PermutationCause,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, D_EF, F};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::{
        ColumnClaimsBus, ColumnClaimsMessage, FinalTranscriptStateBus, FinalTranscriptStateMessage,
        StackingModuleBus, StackingModuleMessage, TranscriptBus, TranscriptBusMessage,
        TranscriptEndIndexBus, TranscriptEndIndexMessage,
    },
    stacking::sorted_column_claims,
    system::Preflight,
    transcript::poseidon2::CHUNK,
};

const PROOF_IDX: usize = 0;
const IS_VALID: usize = 1;
const IS_FIRST: usize = 2;
const IS_LAST: usize = 3;
const CLAIM_IDX: usize = 4;
const SORT_IDX: usize = 5;
const PART_IDX: usize = 6;
const COL_IDX: usize = 7;
const COL_CLAIM: usize = 8;
const ROT_CLAIM: usize = COL_CLAIM + D_EF;
const NEED_ROT: usize = ROT_CLAIM + D_EF;
const TIDX: usize = NEED_ROT + 1;
const CLAIMS_WIDTH: usize = TIDX + 1;

/// Narrow, row-oriented bridge from the constraint-reduction claim bus to the transcript.
///
/// The previous prototype placed all claims in one row. That had the same algebraic content but
/// produced hundreds of thousands of columns and exceeded CUDA's `u16` symbolic-column format.
/// This layout is the natural SWIRL layout: one original column-claim pair per row.
#[derive(Clone, Copy, Debug)]
pub struct ConstraintReductionClaimsAir {
    stacking_module_bus: StackingModuleBus,
    column_claims_bus: ColumnClaimsBus,
    transcript_bus: TranscriptBus,
}

impl ConstraintReductionClaimsAir {
    #[must_use]
    pub const fn new(
        stacking_module_bus: StackingModuleBus,
        column_claims_bus: ColumnClaimsBus,
        transcript_bus: TranscriptBus,
    ) -> Self {
        Self {
            stacking_module_bus,
            column_claims_bus,
            transcript_bus,
        }
    }

    pub fn generate_trace(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        preflights: &[Preflight],
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        if proofs.len() != preflights.len() || proofs.is_empty() {
            return None;
        }
        let claims = proofs
            .iter()
            .zip(preflights)
            .map(|(proof, preflight)| {
                sorted_column_claims(child_vk, proof, &preflight.proof_shape.sorted_trace_vdata)
            })
            .collect::<Vec<_>>();
        if claims.iter().any(Vec::is_empty) {
            return None;
        }
        let minimum_height = claims.iter().map(Vec::len).sum::<usize>();
        let height = required_height.unwrap_or_else(|| minimum_height.next_power_of_two());
        if height < minimum_height {
            return None;
        }

        let mut trace = F::zero_vec(height * CLAIMS_WIDTH);
        let mut rows = trace.chunks_exact_mut(CLAIMS_WIDTH);
        for (proof_idx, (proof_claims, preflight)) in claims.iter().zip(preflights).enumerate() {
            let start_tidx = preflight.batch_constraint.tidx_before_column_openings;
            if preflight.transcript.len() != start_tidx + proof_claims.len() * 2 * D_EF {
                return None;
            }
            for (claim_idx, claim) in proof_claims.iter().enumerate() {
                let row = rows.next()?;
                let air_id = preflight.proof_shape.sorted_trace_vdata[claim.sort_idx].0;
                row[PROOF_IDX] = F::from_usize(proof_idx);
                row[IS_VALID] = F::ONE;
                row[IS_FIRST] = F::from_bool(claim_idx == 0);
                row[IS_LAST] = F::from_bool(claim_idx + 1 == proof_claims.len());
                row[CLAIM_IDX] = F::from_usize(claim_idx);
                row[SORT_IDX] = F::from_usize(claim.sort_idx);
                row[PART_IDX] = F::from_usize(claim.part_idx);
                row[COL_IDX] = F::from_usize(claim.col_idx);
                row[COL_CLAIM..COL_CLAIM + D_EF]
                    .copy_from_slice(claim.col_claim.as_basis_coefficients_slice());
                row[ROT_CLAIM..ROT_CLAIM + D_EF]
                    .copy_from_slice(claim.rot_claim.as_basis_coefficients_slice());
                row[NEED_ROT] = F::from_bool(child_vk.inner.per_air[air_id].params.need_rot);
                row[TIDX] = F::from_usize(start_tidx + claim_idx * 2 * D_EF);
            }
        }
        Some(RowMajorMatrix::new(trace, CLAIMS_WIDTH))
    }
}

impl BaseAir<F> for ConstraintReductionClaimsAir {
    fn width(&self) -> usize {
        CLAIMS_WIDTH
    }
}

impl BaseAirWithPublicValues<F> for ConstraintReductionClaimsAir {}
impl PartitionedBaseAir<F> for ConstraintReductionClaimsAir {}

impl<AB> Air<AB> for ConstraintReductionClaimsAir
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("constraint-reduction claim row");
        let next = main
            .row_slice(1)
            .expect("constraint-reduction next claim row");

        builder.assert_bool(local[IS_VALID]);
        builder.assert_bool(local[IS_FIRST]);
        builder.assert_bool(local[IS_LAST]);
        builder.assert_bool(local[NEED_ROT]);
        builder.when_first_row().assert_one(local[IS_VALID]);
        builder.when_first_row().assert_one(local[IS_FIRST]);
        builder.when_first_row().assert_zero(local[PROOF_IDX]);
        builder.when_first_row().assert_zero(local[CLAIM_IDX]);
        builder
            .when_last_row()
            .when(local[IS_VALID])
            .assert_one(local[IS_LAST]);

        let inactive = not::<AB::Expr>(local[IS_VALID]);
        for value in local.iter().skip(2) {
            builder.when(inactive.clone()).assert_zero(*value);
        }
        let transition = builder.is_transition_window(2);
        // The active-prefix constraint below implies `local.is_valid` whenever
        // `next.is_valid`. Omitting that redundant factor keeps the fixed
        // relation within OpenVM's degree-4 recursion bound.
        let both_valid: AB::Expr = transition.clone() * Into::<AB::Expr>::into(next[IS_VALID]);
        builder
            .when(transition.clone())
            .assert_zero(next[IS_VALID] * not::<AB::Expr>(local[IS_VALID]));
        builder
            .when(both_valid.clone())
            .assert_eq(next[IS_FIRST], Into::<AB::Expr>::into(local[IS_LAST]));
        builder.when(both_valid.clone()).assert_eq(
            next[PROOF_IDX],
            Into::<AB::Expr>::into(local[PROOF_IDX]) + local[IS_LAST],
        );
        builder
            .when(both_valid.clone() * not::<AB::Expr>(local[IS_LAST]))
            .assert_eq(
                next[CLAIM_IDX],
                Into::<AB::Expr>::into(local[CLAIM_IDX]) + AB::Expr::ONE,
            );
        builder
            .when(both_valid.clone() * not::<AB::Expr>(local[IS_LAST]))
            .assert_eq(
                next[TIDX],
                Into::<AB::Expr>::into(local[TIDX]) + AB::Expr::from_usize(2 * D_EF),
            );
        builder
            .when(both_valid * local[IS_LAST])
            .assert_zero(next[CLAIM_IDX]);
        builder
            .when(transition * local[IS_VALID] * not::<AB::Expr>(next[IS_VALID]))
            .assert_one(local[IS_LAST]);

        self.stacking_module_bus.receive(
            builder,
            local[PROOF_IDX].into(),
            StackingModuleMessage {
                tidx: local[TIDX].into(),
            },
            local[IS_VALID] * local[IS_FIRST],
        );

        let current = core::array::from_fn(|limb| local[COL_CLAIM + limb]);
        let rotated = core::array::from_fn(|limb| local[ROT_CLAIM + limb]);
        self.column_claims_bus.send(
            builder,
            local[PROOF_IDX].into(),
            ColumnClaimsMessage {
                sort_idx: local[SORT_IDX].into(),
                part_idx: local[PART_IDX].into(),
                col_idx: local[COL_IDX].into(),
                claim: current.map(Into::into),
                is_rot: AB::Expr::ZERO,
            },
            local[IS_VALID].into(),
        );
        for limb in 0..D_EF {
            builder
                .when(local[IS_VALID] * not::<AB::Expr>(local[NEED_ROT]))
                .assert_zero(local[ROT_CLAIM + limb]);
        }
        self.column_claims_bus.send(
            builder,
            local[PROOF_IDX].into(),
            ColumnClaimsMessage {
                sort_idx: local[SORT_IDX].into(),
                part_idx: local[PART_IDX].into(),
                col_idx: local[COL_IDX].into(),
                claim: rotated.map(Into::into),
                is_rot: AB::Expr::ONE,
            },
            Into::<AB::Expr>::into(local[IS_VALID]) * local[NEED_ROT],
        );

        for limb in 0..D_EF {
            self.transcript_bus.receive(
                builder,
                local[PROOF_IDX].into(),
                TranscriptBusMessage {
                    tidx: Into::<AB::Expr>::into(local[TIDX]) + AB::Expr::from_usize(limb),
                    value: current[limb].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local[IS_VALID].into(),
            );
            self.transcript_bus.receive(
                builder,
                local[PROOF_IDX].into(),
                TranscriptBusMessage {
                    tidx: Into::<AB::Expr>::into(local[TIDX]) + AB::Expr::from_usize(D_EF + limb),
                    value: rotated[limb].into(),
                    is_sample: AB::Expr::ZERO,
                },
                local[IS_VALID].into(),
            );
        }
    }
}

const CHECKPOINT_SLOT_WIDTH: usize = 2 + POSEIDON2_WIDTH;

/// Reconstruct the state emitted by the recursive [`crate::transcript::transcript::TranscriptAir`].
///
/// The AIR deliberately leaves a final absorb block unpermuted until a later squeeze. That lazy
/// endpoint differs from the eager continuation state returned by
/// `TranscriptLog::duplex_final_state` when the prefix ends exactly on a full rate block, which
/// every claim pair does here.
fn transcript_air_final_state(preflight: &Preflight) -> Option<[F; POSEIDON2_WIDTH]> {
    let values = preflight.transcript.values();
    let samples = preflight.transcript.samples();
    if values.is_empty() || values.len() != samples.len() {
        return None;
    }
    let mut state = [F::ZERO; POSEIDON2_WIDTH];
    let mut tidx = 0usize;
    let mut permutation_idx = 0usize;
    let mut ended_with_full_absorb = false;
    while tidx < values.len() {
        let is_sample = samples[tidx];
        let mut count = 0usize;
        while tidx < values.len() && samples[tidx] == is_sample && count < CHUNK {
            if is_sample {
                if state[CHUNK - 1 - count] != values[tidx] {
                    return None;
                }
            } else {
                state[count] = values[tidx];
            }
            tidx += 1;
            count += 1;
        }
        if !is_sample {
            state[CHUNK] += F::from_usize(count);
        }
        ended_with_full_absorb = !is_sample && count == CHUNK && tidx == values.len();
        let permuted = if tidx == values.len() {
            false
        } else if samples[tidx] != is_sample {
            samples[tidx]
        } else {
            debug_assert_eq!(count, CHUNK);
            !is_sample || samples[tidx]
        };
        if permuted {
            // `TranscriptLog::perm_results` starts with the recorder's initial
            // state, then stores one output for every permutation transition.
            state = *preflight
                .transcript
                .perm_results()
                .get(permutation_idx + 1)?;
            permutation_idx += 1;
        }
    }
    let transitions = preflight.transcript.permutation_transitions();
    if permutation_idx == transitions.len() {
        return Some(state);
    }
    // The native sponge eagerly permutes when its final absorb fills the rate,
    // while TranscriptAir intentionally leaves that state unpermuted until a
    // later squeeze. Validate and ignore exactly that one trailing transition.
    let trailing = transitions.get(permutation_idx)?;
    (ended_with_full_absorb
        && permutation_idx + 1 == transitions.len()
        && trailing.cause == PermutationCause::Absorb
        && trailing.input == state)
        .then_some(state)
}

/// Compact public statement for a pre-stacking reduction.
///
/// The final transcript state commits to the exact roots, proof messages, sampled opening point,
/// and column claims consumed above. The operation count prevents length ambiguity. The terminal
/// proof must replay the retained prefix and match both before using its original PCS data.
#[derive(Clone, Copy, Debug)]
pub struct ConstraintReductionCheckpointAir<const MAX_NUM_PROOFS: usize> {
    final_state_bus: FinalTranscriptStateBus,
    end_index_bus: TranscriptEndIndexBus,
}

impl<const MAX_NUM_PROOFS: usize> ConstraintReductionCheckpointAir<MAX_NUM_PROOFS> {
    #[must_use]
    pub const fn new(
        final_state_bus: FinalTranscriptStateBus,
        end_index_bus: TranscriptEndIndexBus,
    ) -> Self {
        Self {
            final_state_bus,
            end_index_bus,
        }
    }

    #[must_use]
    pub const fn public_width() -> usize {
        MAX_NUM_PROOFS * CHECKPOINT_SLOT_WIDTH
    }

    pub fn generate_trace(
        &self,
        preflights: &[Preflight],
        required_height: Option<usize>,
    ) -> Option<(RowMajorMatrix<F>, Vec<F>)> {
        if preflights.is_empty()
            || preflights.len() > MAX_NUM_PROOFS
            || required_height.is_some_and(|height| height != 1)
        {
            return None;
        }
        let mut row = F::zero_vec(Self::public_width());
        for (proof_idx, preflight) in preflights.iter().enumerate() {
            let slot = &mut row
                [proof_idx * CHECKPOINT_SLOT_WIDTH..(proof_idx + 1) * CHECKPOINT_SLOT_WIDTH];
            slot[0] = F::ONE;
            slot[1] = F::from_usize(preflight.transcript.len());
            slot[2..].copy_from_slice(&transcript_air_final_state(preflight)?);
        }
        Some((RowMajorMatrix::new(row.clone(), Self::public_width()), row))
    }
}

impl<const MAX_NUM_PROOFS: usize> BaseAir<F> for ConstraintReductionCheckpointAir<MAX_NUM_PROOFS> {
    fn width(&self) -> usize {
        Self::public_width()
    }
}

impl<const MAX_NUM_PROOFS: usize> BaseAirWithPublicValues<F>
    for ConstraintReductionCheckpointAir<MAX_NUM_PROOFS>
{
    fn num_public_values(&self) -> usize {
        Self::public_width()
    }
}

impl<const MAX_NUM_PROOFS: usize> PartitionedBaseAir<F>
    for ConstraintReductionCheckpointAir<MAX_NUM_PROOFS>
{
}

impl<AB, const MAX_NUM_PROOFS: usize> Air<AB> for ConstraintReductionCheckpointAir<MAX_NUM_PROOFS>
where
    AB: AirBuilder<F = F> + AirBuilderWithPublicValues + InteractionBuilder,
    AB::Var: Copy,
    AB::PublicVar: Copy,
    AB::Expr: From<AB::PublicVar>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("constraint-reduction checkpoint row");
        let public = builder.public_values().to_vec();
        let mut previous_active = AB::Expr::ONE;
        for proof_idx in 0..MAX_NUM_PROOFS {
            let slot =
                &row[proof_idx * CHECKPOINT_SLOT_WIDTH..(proof_idx + 1) * CHECKPOINT_SLOT_WIDTH];
            let public_slot =
                &public[proof_idx * CHECKPOINT_SLOT_WIDTH..(proof_idx + 1) * CHECKPOINT_SLOT_WIDTH];
            let active: AB::Expr = slot[0].into();
            builder.assert_bool(slot[0]);
            builder.assert_zero(active.clone() * (AB::Expr::ONE - previous_active));
            previous_active = active.clone();
            for (actual, expected) in slot.iter().zip(public_slot) {
                builder.assert_eq(*actual, *expected);
            }
            for value in slot.iter().skip(1) {
                builder
                    .when(AB::Expr::ONE - active.clone())
                    .assert_zero(*value);
            }
            self.end_index_bus.receive(
                builder,
                AB::Expr::from_usize(proof_idx),
                TranscriptEndIndexMessage {
                    tidx: slot[1].into(),
                },
                active.clone(),
            );
            self.final_state_bus.receive(
                builder,
                AB::Expr::from_usize(proof_idx),
                FinalTranscriptStateMessage {
                    state: core::array::from_fn(|limb| slot[2 + limb]),
                },
                active,
            );
        }
    }
}

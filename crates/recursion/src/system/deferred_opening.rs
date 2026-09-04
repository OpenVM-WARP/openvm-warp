//! Public checkpoint for a recursive verifier that stops after SWIRL stacking.
//!
//! The checkpoint is deliberately not a PCS-opening predicate.  The ordinary
//! verifier modules constrain AIR, LogUp, transcript, and stacking.  This AIR
//! only publishes the transcript state after one domain-separated final
//! squeeze, so a separately authenticated terminal PCS obligation can be
//! joined to the same roots, opening point, and claimed values.

use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::{
        FinalTranscriptStateBus, FinalTranscriptStateMessage, TranscriptBus, TranscriptBusMessage,
    },
    system::Preflight,
};

const ACTIVE_OFFSET: usize = 0;
const TIDX_OFFSET: usize = 1;
const SAMPLE_OFFSET: usize = 2;
const STATE_OFFSET: usize = SAMPLE_OFFSET + D_EF;
const SLOT_WIDTH: usize = STATE_OFFSET + POSEIDON2_WIDTH;
const SLOT_PUBLIC_WIDTH: usize = 2 + POSEIDON2_WIDTH;

/// Verifier-derived endpoint of one child SWIRL transcript immediately after
/// the stacking reduction.
///
/// This is an incomplete-proof checkpoint.  It authenticates neither a PCS
/// opening nor an accepting child proof on its own; a caller must connect it
/// to the same committed codeword through WARP and terminal Decide/WHIR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeferredOpeningCheckpoint {
    pub transcript_index: usize,
    pub transcript_state: [F; POSEIDON2_WIDTH],
}

/// Private witness carried by the ordinary deferred-opening checkpoint trace.
///
/// SDK adapters decode this trace instead of replaying verifier preflight.
/// Acceptance still comes from the transcript, end-index and final-state
/// buses in the enclosing proof.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeferredOpeningCheckpointWitness {
    pub checkpoint: DeferredOpeningCheckpoint,
    pub samples: EF,
}

/// One-row statement AIR for at most `MAX_NUM_PROOFS` independently replayed
/// child transcripts.
#[derive(Clone, Copy, Debug)]
pub struct DeferredOpeningCheckpointAir<const MAX_NUM_PROOFS: usize> {
    pub transcript_bus: TranscriptBus,
    pub final_state_bus: FinalTranscriptStateBus,
}

impl<const MAX_NUM_PROOFS: usize> DeferredOpeningCheckpointAir<MAX_NUM_PROOFS> {
    #[must_use]
    pub const fn new(
        transcript_bus: TranscriptBus,
        final_state_bus: FinalTranscriptStateBus,
    ) -> Self {
        Self {
            transcript_bus,
            final_state_bus,
        }
    }

    #[must_use]
    pub const fn public_width() -> usize {
        MAX_NUM_PROOFS * SLOT_PUBLIC_WIDTH
    }

    /// Decode the public checkpoint emitted by [`Self::generate_trace`].
    ///
    /// Active slots must form a canonical prefix and every inactive slot must
    /// be zero.  Keeping this parser next to the AIR prevents the SDK wrapper
    /// from maintaining a second, potentially divergent slot layout.
    pub fn decode_public_values(
        public_values: &[F],
    ) -> Result<Vec<DeferredOpeningCheckpoint>, &'static str> {
        if public_values.len() != Self::public_width() {
            return Err("deferred-opening checkpoint public-value width");
        }

        let mut checkpoints = Vec::with_capacity(MAX_NUM_PROOFS);
        let mut inactive_seen = false;
        for slot in public_values.chunks_exact(SLOT_PUBLIC_WIDTH) {
            let active = slot[0];
            if active == F::ZERO {
                inactive_seen = true;
                if slot[1..].iter().any(|value| *value != F::ZERO) {
                    return Err("non-canonical inactive deferred-opening checkpoint");
                }
                continue;
            }
            if active != F::ONE {
                return Err("non-boolean deferred-opening checkpoint activity");
            }
            if inactive_seen {
                return Err("non-prefix deferred-opening checkpoints");
            }
            checkpoints.push(DeferredOpeningCheckpoint {
                transcript_index: slot[1].as_canonical_u32() as usize,
                transcript_state: slot[2..]
                    .try_into()
                    .map_err(|_| "deferred-opening checkpoint transcript state")?,
            });
        }
        Ok(checkpoints)
    }

    /// Decode the private rows emitted by [`Self::generate_trace`].
    pub fn decode_trace(
        trace: &RowMajorMatrix<F>,
    ) -> Result<Vec<DeferredOpeningCheckpointWitness>, &'static str> {
        if trace.height() != 1 || trace.width() != MAX_NUM_PROOFS * SLOT_WIDTH {
            return Err("deferred-opening checkpoint trace shape");
        }
        let row = trace
            .row_slice(0)
            .ok_or("deferred-opening checkpoint trace row")?;
        let mut checkpoints = Vec::with_capacity(MAX_NUM_PROOFS);
        let mut inactive_seen = false;
        for slot in row.chunks_exact(SLOT_WIDTH) {
            let active = slot[ACTIVE_OFFSET];
            if active == F::ZERO {
                inactive_seen = true;
                if slot[1..].iter().any(|value| *value != F::ZERO) {
                    return Err("non-canonical inactive deferred-opening checkpoint trace");
                }
                continue;
            }
            if active != F::ONE {
                return Err("non-boolean deferred-opening checkpoint trace activity");
            }
            if inactive_seen {
                return Err("non-prefix deferred-opening checkpoint trace");
            }
            checkpoints.push(DeferredOpeningCheckpointWitness {
                checkpoint: DeferredOpeningCheckpoint {
                    transcript_index: slot[TIDX_OFFSET].as_canonical_u32() as usize,
                    transcript_state: slot[STATE_OFFSET..]
                        .try_into()
                        .map_err(|_| "deferred-opening checkpoint transcript state")?,
                },
                samples: EF::from_basis_coefficients_slice(&slot[SAMPLE_OFFSET..STATE_OFFSET])
                    .ok_or("deferred-opening checkpoint samples")?,
            });
        }
        Ok(checkpoints)
    }

    /// Build the exact one-row witness and public checkpoint from transcripts
    /// whose final operation is the dedicated `D_EF`-limb squeeze.
    pub fn generate_trace(
        &self,
        preflights: &[Preflight],
        required_height: Option<usize>,
    ) -> Option<(RowMajorMatrix<F>, Vec<F>)> {
        if preflights.len() > MAX_NUM_PROOFS || required_height.is_some_and(|height| height != 1) {
            return None;
        }

        let mut row = F::zero_vec(self.width());
        let mut public_values = F::zero_vec(Self::public_width());
        for (proof_idx, preflight) in preflights.iter().enumerate() {
            let values = preflight.transcript.values();
            let samples = preflight.transcript.samples();
            if values.len() < D_EF
                || samples.len() != values.len()
                || !samples[values.len() - D_EF..].iter().all(|sample| *sample)
            {
                return None;
            }
            let state = *preflight.transcript.perm_results().last()?;
            let tidx = values.len() - D_EF;
            let slot = &mut row[proof_idx * SLOT_WIDTH..(proof_idx + 1) * SLOT_WIDTH];
            slot[ACTIVE_OFFSET] = F::ONE;
            slot[TIDX_OFFSET] = F::from_usize(tidx);
            slot[SAMPLE_OFFSET..STATE_OFFSET].copy_from_slice(&values[tidx..]);
            slot[STATE_OFFSET..].copy_from_slice(&state);

            let public = &mut public_values
                [proof_idx * SLOT_PUBLIC_WIDTH..(proof_idx + 1) * SLOT_PUBLIC_WIDTH];
            public[ACTIVE_OFFSET] = F::ONE;
            public[TIDX_OFFSET] = F::from_usize(tidx);
            public[2..].copy_from_slice(&state);
        }
        Some((RowMajorMatrix::new(row, self.width()), public_values))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn checkpoint_values() -> Vec<F> {
        let mut values = F::zero_vec(DeferredOpeningCheckpointAir::<2>::public_width());
        values[0] = F::ONE;
        values[1] = F::from_u32(19);
        for (index, value) in values[2..2 + POSEIDON2_WIDTH].iter_mut().enumerate() {
            *value = F::from_usize(index + 1);
        }
        values
    }

    #[test]
    fn decodes_canonical_active_prefix() {
        let values = checkpoint_values();
        let decoded = DeferredOpeningCheckpointAir::<2>::decode_public_values(&values).unwrap();
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].transcript_index, 19);
        assert_eq!(decoded[0].transcript_state[0], F::ONE);
    }

    #[test]
    fn rejects_non_prefix_and_nonzero_inactive_slots() {
        let mut non_prefix = checkpoint_values();
        non_prefix[0] = F::ZERO;
        non_prefix[1] = F::ZERO;
        non_prefix[2..2 + POSEIDON2_WIDTH].fill(F::ZERO);
        non_prefix[SLOT_PUBLIC_WIDTH] = F::ONE;
        assert!(DeferredOpeningCheckpointAir::<2>::decode_public_values(&non_prefix).is_err());

        let mut nonzero_inactive = checkpoint_values();
        nonzero_inactive[SLOT_PUBLIC_WIDTH + 1] = F::ONE;
        assert!(
            DeferredOpeningCheckpointAir::<2>::decode_public_values(&nonzero_inactive).is_err()
        );
    }
}

impl<const MAX_NUM_PROOFS: usize> BaseAir<F> for DeferredOpeningCheckpointAir<MAX_NUM_PROOFS> {
    fn width(&self) -> usize {
        MAX_NUM_PROOFS * SLOT_WIDTH
    }
}

impl<const MAX_NUM_PROOFS: usize> BaseAirWithPublicValues<F>
    for DeferredOpeningCheckpointAir<MAX_NUM_PROOFS>
{
    fn num_public_values(&self) -> usize {
        Self::public_width()
    }
}

impl<const MAX_NUM_PROOFS: usize> PartitionedBaseAir<F>
    for DeferredOpeningCheckpointAir<MAX_NUM_PROOFS>
{
}

impl<AB, const MAX_NUM_PROOFS: usize> Air<AB> for DeferredOpeningCheckpointAir<MAX_NUM_PROOFS>
where
    AB: AirBuilder<F = F> + AirBuilderWithPublicValues + InteractionBuilder,
    AB::Var: Copy,
    AB::PublicVar: Copy,
    AB::Expr: From<AB::PublicVar>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("deferred-opening checkpoint row");
        let public = builder.public_values().to_vec();
        let mut previous_active = AB::Expr::ONE;

        for proof_idx in 0..MAX_NUM_PROOFS {
            let slot = &row[proof_idx * SLOT_WIDTH..(proof_idx + 1) * SLOT_WIDTH];
            let active: AB::Expr = slot[ACTIVE_OFFSET].into();
            builder.assert_bool(slot[ACTIVE_OFFSET]);
            // Active proofs are a canonical prefix.  This prevents equivalent
            // statements from acquiring multiple inactive-slot encodings.
            builder.assert_zero(active.clone() * (AB::Expr::ONE - previous_active));
            previous_active = active.clone();

            let public_slot =
                &public[proof_idx * SLOT_PUBLIC_WIDTH..(proof_idx + 1) * SLOT_PUBLIC_WIDTH];
            builder.assert_eq(slot[ACTIVE_OFFSET], public_slot[ACTIVE_OFFSET]);
            builder.assert_eq(slot[TIDX_OFFSET], public_slot[TIDX_OFFSET]);
            for (state, expected) in slot[STATE_OFFSET..].iter().zip(public_slot[2..].iter()) {
                builder.assert_eq(*state, *expected);
            }

            let inactive = AB::Expr::ONE - active.clone();
            for value in &slot[TIDX_OFFSET..] {
                builder.when(inactive.clone()).assert_zero(*value);
            }

            for limb in 0..D_EF {
                self.transcript_bus.receive(
                    builder,
                    AB::Expr::from_usize(proof_idx),
                    TranscriptBusMessage {
                        tidx: Into::<AB::Expr>::into(slot[TIDX_OFFSET])
                            + AB::Expr::from_usize(limb),
                        value: slot[SAMPLE_OFFSET + limb].into(),
                        is_sample: AB::Expr::ONE,
                    },
                    active.clone(),
                );
            }
            self.final_state_bus.receive(
                builder,
                AB::Expr::from_usize(proof_idx),
                FinalTranscriptStateMessage {
                    state: core::array::from_fn(|idx| slot[STATE_OFFSET + idx]),
                },
                active,
            );
        }
    }
}

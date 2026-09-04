use core::borrow::Borrow;

use openvm_circuit_primitives::{
    utils::{and, assert_array_eq, not, or},
    ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::Matrix;

use crate::{
    bus::{
        CertifiedTranscriptCheckpointBus, CertifiedTranscriptCheckpointMessage,
        FinalTranscriptStateBus, FinalTranscriptStateMessage, Poseidon2PermuteBus,
        Poseidon2PermuteMessage, ResumeTranscriptStateBus, ResumeTranscriptStateMessage,
        TranscriptBus, TranscriptBusMessage, TranscriptEndIndexBus, TranscriptEndIndexMessage,
    },
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    transcript::poseidon2::{CHUNK, POSEIDON2_WIDTH},
};

#[repr(C)]
#[derive(AlignedBorrow, Debug, StructReflection)]
pub struct TranscriptCols<T> {
    pub proof_idx: T,
    pub is_proof_start: T,

    pub tidx: T,
    /// Indicator for sample/observe.
    pub is_sample: T,
    /// 0/1 indicators for the positions that we are absorbing/squeezing (i.e., are in
    /// the transcript). Constrained to be "decreasing". Because transcript_bus messages
    /// are sent with multiplicity 0 or 1, this also functions as the lookup count.
    pub mask: [T; CHUNK],

    /// The poseidon2 state.
    pub prev_state: [T; POSEIDON2_WIDTH],
    pub post_state: [T; POSEIDON2_WIDTH],
}

/// Trailing columns present only when [`TranscriptAir::resume_state_bus`] is set.
///
/// Held apart from [`TranscriptCols`] so a circuit that starts its transcripts at
/// the canonical zero sponge -- every circuit except the native WARP recursive
/// history stage -- keeps the width it had. Widening the shared struct instead
/// would grow the recursive aggregation lane's transcript trace for no benefit,
/// which would also distort the very lane comparison this work is measured by.
#[repr(C)]
#[derive(AlignedBorrow, Debug, StructReflection)]
pub struct TranscriptResumeCols<T> {
    /// The sponge state handed over by the proof this one continues. Only the
    /// first row of each proof reads it.
    pub state: [T; POSEIDON2_WIDTH],
}

/// Trailing selectors present only when an intermediate checkpoint bus is
/// enabled.  Kind zero is the pre-VACC boundary and kind one the post-VACC
/// boundary.  Keeping them optional leaves the recursive lane's shared
/// transcript width unchanged.
#[repr(C)]
#[derive(AlignedBorrow, Debug, StructReflection)]
pub struct TranscriptCheckpointCols<T> {
    pub selected: [T; 2],
}

#[derive(ColumnsAir)]
#[columns_via(TranscriptCols<u8>)]
pub struct TranscriptAir {
    pub transcript_bus: TranscriptBus,
    pub poseidon2_permute_bus: Poseidon2PermuteBus,
    pub final_state_bus: Option<FinalTranscriptStateBus>,
    /// Set when the proofs in this circuit continue an earlier transcript
    /// instead of starting at the canonical zero sponge.
    ///
    /// Keyed rather than witnessed, so a circuit cannot choose per proof whether
    /// the zero-state rule binds it: the native WARP genesis stage is built
    /// without this and every later stage with it.
    pub resume_state_bus: Option<ResumeTranscriptStateBus>,
    /// Emits this proof's end index; see [`TranscriptEndIndexBus`].
    pub end_index_bus: Option<TranscriptEndIndexBus>,
    /// Emits two row-aligned intermediate states selected by witness columns.
    /// A companion AIR fixes their indices and values, while boolean selectors
    /// ensure they can only name actual sample rows.
    pub checkpoint_state_bus: Option<CertifiedTranscriptCheckpointBus>,
}

impl TranscriptAir {
    /// Width of the trailing resume columns, zero when not resuming.
    pub fn resume_width<F: Field>(&self) -> usize {
        if self.resume_state_bus.is_some() {
            TranscriptResumeCols::<F>::width()
        } else {
            0
        }
    }

    /// Total row width, which trace generation must match.
    pub fn row_width<F: Field>(&self) -> usize {
        TranscriptCols::<F>::width() + self.resume_width::<F>() + self.checkpoint_width::<F>()
    }

    pub fn checkpoint_width<F: Field>(&self) -> usize {
        if self.checkpoint_state_bus.is_some() {
            TranscriptCheckpointCols::<F>::width()
        } else {
            0
        }
    }
}

impl<F: Field> BaseAir<F> for TranscriptAir {
    fn width(&self) -> usize {
        self.row_width::<F>()
    }
}

impl<F: Field> BaseAirWithPublicValues<F> for TranscriptAir {}
impl<F: Field> PartitionedBaseAir<F> for TranscriptAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for TranscriptAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("window should have two elements"),
            main.row_slice(1).expect("window should have two elements"),
        );
        let base_width = TranscriptCols::<AB::Var>::width();
        let resume_width = self.resume_width::<AB::F>();
        let resume_end = base_width + resume_width;
        let local_resume = local[base_width..resume_end].to_vec();
        let local_checkpoint = local[resume_end..].to_vec();
        let local: &TranscriptCols<AB::Var> = local[..base_width].borrow();
        let next: &TranscriptCols<AB::Var> = next[..base_width].borrow();

        ///////////////////////////////////////////////////////////////////////
        // Constraints
        ///////////////////////////////////////////////////////////////////////
        let is_valid = local.mask[0];
        let next_valid = next.mask[0];

        NestedForLoopSubAir::<1> {}.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: is_valid,
                    counter: [local.proof_idx],
                    is_first: [local.is_proof_start],
                }
                .map_into(),
                NestedForLoopIoCols {
                    is_enabled: next_valid,
                    counter: [next.proof_idx],
                    is_first: [next.is_proof_start],
                }
                .map_into(),
            ),
        );

        builder.when(local.is_proof_start).assert_one(is_valid);
        builder.assert_bool(local.is_sample);

        // Initial state constraints.
        //
        // A proof's first row does not preserve the state it starts from: an
        // observe overwrites rate lanes `0..count` with its own operands. Only
        // the capacity half and the rate lanes this row leaves alone are visible
        // here, so those are the only ones either branch can constrain.
        //
        // Without resumption they must all be zero. With it they must equal the
        // handed-over state, which is received in full because the sender cannot
        // know which lanes get overwritten. Everything after the first row is
        // unchanged -- `next.tidx = local.tidx + count` and the post/prev state
        // chaining still force the sequence -- so resumption relocates where a
        // proof begins and cannot perturb what follows.
        // Length binding: an observe row adds its operation count to the
        // first capacity lane of its own permutation input, mirroring
        // `DuplexSponge::absorb`'s per-absorb counter. Sample rows add zero.
        let mut local_count = AB::Expr::ZERO;
        let mut next_count = AB::Expr::ZERO;
        for i in 0..CHUNK {
            local_count += local.mask[i].into();
            next_count += next.mask[i].into();
        }
        let local_capacity_add = local_count * (AB::Expr::ONE - local.is_sample);
        let next_capacity_add = next_count * (AB::Expr::ONE - next.is_sample);

        if let Some(resume_state_bus) = self.resume_state_bus {
            let resume: &TranscriptResumeCols<AB::Var> = local_resume.as_slice().borrow();
            resume_state_bus.receive(
                builder,
                local.proof_idx,
                ResumeTranscriptStateMessage {
                    tidx: local.tidx.into(),
                    state: resume.state.map(Into::into),
                },
                local.is_proof_start,
            );
            for i in 0..CHUNK {
                if i == 0 {
                    builder.when(local.is_proof_start).assert_eq(
                        local.prev_state[CHUNK],
                        resume.state[CHUNK] + local_capacity_add.clone(),
                    );
                } else {
                    builder
                        .when(local.is_proof_start)
                        .assert_eq(local.prev_state[i + CHUNK], resume.state[i + CHUNK]);
                }

                builder
                    .when(local.is_proof_start * (AB::Expr::ONE - local.mask[i]))
                    .assert_eq(local.prev_state[i], resume.state[i]);
            }
        } else {
            builder.when(local.is_proof_start).assert_zero(local.tidx);
            for i in 0..CHUNK {
                if i == 0 {
                    builder
                        .when(local.is_proof_start)
                        .assert_eq(local.prev_state[CHUNK], local_capacity_add.clone());
                } else {
                    builder
                        .when(local.is_proof_start)
                        .assert_eq(local.prev_state[i + CHUNK], AB::Expr::ZERO);
                }

                builder
                    .when(local.is_proof_start * (AB::Expr::ONE - local.mask[i]))
                    .assert_eq(local.prev_state[i], AB::Expr::ZERO);
            }
        }

        let mut count = AB::Expr::ZERO;
        let local_next_same_proof = next_valid - next.is_proof_start;
        for i in 0..CHUNK {
            builder.assert_bool(local.mask[i]);
            count += local.mask[i].into();

            let skip = local.mask[i] - AB::Expr::ONE;
            if i < CHUNK - 1 {
                // if mask[i] = 0, then mask[i+1] = 0
                builder.when(skip.clone()).assert_zero(local.mask[i + 1]);
            }

            // The state after permutation of this round, should check against next round's input
            // if next.mask[i] = 0 --> i-th not touched --> it should stay the same (if next is
            // valid)
            builder
                .when((AB::Expr::ONE - next.mask[i]) * local_next_same_proof.clone())
                .assert_eq(local.post_state[i], next.prev_state[i]);
            // When it's squeeze(sample), the state always remains the same
            builder
                .when(next.is_sample * local_next_same_proof.clone())
                .assert_eq(local.post_state[i], next.prev_state[i]);

            // The capacity part carries over, with the next row's absorb
            // count added to the first capacity lane (length binding).
            if i == 0 {
                builder.when(local_next_same_proof.clone()).assert_eq(
                    next.prev_state[CHUNK],
                    local.post_state[CHUNK] + next_capacity_add.clone(),
                );
            } else {
                builder
                    .when(local_next_same_proof.clone()) // if next is valid
                    .assert_eq(local.post_state[i + CHUNK], next.prev_state[i + CHUNK]);
            }
        }

        // One past this row's last operation; on a proof's final row this is
        // the proof's end index.
        let end_tidx = local.tidx + count.clone();

        if let Some(checkpoint_state_bus) = self.checkpoint_state_bus {
            let checkpoint: &TranscriptCheckpointCols<AB::Var> =
                local_checkpoint.as_slice().borrow();
            for (kind, selected) in checkpoint.selected.into_iter().enumerate() {
                builder.assert_bool(selected);
                builder.when(selected).assert_one(is_valid);
                builder.when(selected).assert_one(local.is_sample);
                checkpoint_state_bus.send(
                    builder,
                    local.proof_idx,
                    CertifiedTranscriptCheckpointMessage {
                        kind: AB::Expr::from_usize(kind),
                        tidx: end_tidx.clone(),
                        sample_count: count.clone(),
                        state: local.post_state.map(Into::into),
                    },
                    selected,
                );
            }
        }

        let mut when_same_proof = builder.when(local_next_same_proof.clone());
        when_same_proof.assert_eq(next.tidx, end_tidx.clone());

        // If local.is_sample = next.is_sample, there have to be CHUNK operations
        when_same_proof
            .when_ne(local.is_sample, not(next.is_sample))
            .assert_eq(count, AB::Expr::from_usize(CHUNK));

        ///////////////////////////////////////////////////////////////////////
        // Interactions
        ///////////////////////////////////////////////////////////////////////
        for i in 0..CHUNK {
            // When absorb, it's normal order (0 -> RATE)
            let observe_message = TranscriptBusMessage {
                tidx: local.tidx + AB::Expr::from_usize(i),
                value: local.prev_state[i].into(),
                is_sample: AB::Expr::ZERO,
            };
            // When squeeze, it's reverse RATE -> 0, so i means RATE - 1 - i
            let sample_message = TranscriptBusMessage {
                tidx: local.tidx + AB::Expr::from_usize(i),
                value: local.prev_state[CHUNK - 1 - i].into(),
                is_sample: AB::Expr::ONE,
            };
            self.transcript_bus.send(
                builder,
                local.proof_idx,
                observe_message,
                local.mask[i] * (AB::Expr::ONE - local.is_sample),
            );
            self.transcript_bus.send(
                builder,
                local.proof_idx,
                sample_message,
                local.mask[i] * local.is_sample,
            );
        }

        // Permute on all non-final rows except when going from sample to observe,
        // and never on the final row.
        let permuted =
            local_next_same_proof * not::<AB::Expr>(and(local.is_sample, not(next.is_sample)));
        self.poseidon2_permute_bus.lookup_key(
            builder,
            Poseidon2PermuteMessage {
                input: local.prev_state,
                output: local.post_state,
            },
            permuted.clone(),
        );

        assert_array_eq(
            &mut builder.when(not::<AB::Expr>(permuted)),
            local.prev_state,
            local.post_state,
        );

        if let Some(end_index_bus) = self.end_index_bus {
            end_index_bus.send(
                builder,
                local.proof_idx,
                TranscriptEndIndexMessage { tidx: end_tidx },
                and(is_valid, or(not(next_valid), next.is_proof_start)),
            );
        }

        if let Some(final_state_bus) = self.final_state_bus {
            final_state_bus.send(
                builder,
                local.proof_idx,
                FinalTranscriptStateMessage {
                    state: local.post_state,
                },
                and(is_valid, or(not(next_valid), next.is_proof_start)),
            );
        }
    }
}

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper, SubAir};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::F;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;

use crate::{
    bus::{ResumeTranscriptStateBus, ResumeTranscriptStateMessage, TranscriptBus},
    proof_shape::bus::{
        RebasedTranscriptStartBus, RebasedTranscriptStartMessage, StartingTidxBus,
        StartingTidxMessage,
    },
    subairs::proof_idx::{ProofIdxIoCols, ProofIdxSubAir},
    system::Preflight,
    tracegen::RowMajorChip,
};

/// Backend transcript separator used by `verify_logup_only_prefix`.
pub const BATCH_CONSTRAINT_MODE_TAG: u64 = 0x4243_4d4f; // "BCMO"
pub const BATCH_CONSTRAINT_MODE_VERSION: u32 = 1;
pub const BATCH_CONSTRAINT_LOGUP_ONLY_MODE: u32 = 1;
pub const BATCH_CONSTRAINT_MODE_SEPARATOR_LEN: usize = 3;

#[repr(C)]
#[derive(AlignedBorrow, Copy, Clone, Debug, StructReflection)]
pub struct RebasedProofShapeStartCols<T> {
    pub is_enabled: T,
    pub proof_idx: T,
    pub start_tidx: T,
    pub state: [T; POSEIDON2_WIDTH],
}

/// Authenticates a partial transcript's source-manifest checkpoint and the
/// exact interaction-only equation separator.
///
/// The checkpoint is received once from the caller and forwarded unchanged to
/// `TranscriptAir`. The three separator operations are looked up at the same
/// absolute cursor, and only their one-past index is handed to ProofShape/GKR.
#[derive(ColumnsAir)]
#[columns_via(RebasedProofShapeStartCols<u8>)]
pub struct RebasedProofShapeStartAir {
    pub start_bus: RebasedTranscriptStartBus,
    pub resume_state_bus: ResumeTranscriptStateBus,
    pub starting_tidx_bus: StartingTidxBus,
    pub transcript_bus: TranscriptBus,
}

impl<Fld: Field> BaseAir<Fld> for RebasedProofShapeStartAir {
    fn width(&self) -> usize {
        RebasedProofShapeStartCols::<Fld>::width()
    }
}

impl<Fld: Field> BaseAirWithPublicValues<Fld> for RebasedProofShapeStartAir {}
impl<Fld: Field> PartitionedBaseAir<Fld> for RebasedProofShapeStartAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for RebasedProofShapeStartAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main
            .row_slice(0)
            .expect("window should have at least one row");
        let next_row = main
            .row_slice(1)
            .expect("window should have at least two rows");
        let local: &RebasedProofShapeStartCols<AB::Var> = (*local_row).borrow();
        let next: &RebasedProofShapeStartCols<AB::Var> = (*next_row).borrow();

        ProofIdxSubAir.eval(
            builder,
            (
                ProofIdxIoCols {
                    is_enabled: local.is_enabled,
                    proof_idx: local.proof_idx,
                }
                .map_into(),
                ProofIdxIoCols {
                    is_enabled: next.is_enabled,
                    proof_idx: next.proof_idx,
                }
                .map_into(),
            ),
        );

        self.start_bus.receive(
            builder,
            local.proof_idx,
            RebasedTranscriptStartMessage {
                tidx: local.start_tidx.into(),
                state: local.state.map(Into::into),
            },
            local.is_enabled,
        );
        self.resume_state_bus.send(
            builder,
            local.proof_idx,
            ResumeTranscriptStateMessage {
                tidx: local.start_tidx.into(),
                state: local.state.map(Into::into),
            },
            local.is_enabled,
        );

        let separator = [
            AB::F::from_u64(BATCH_CONSTRAINT_MODE_TAG),
            AB::F::from_u32(BATCH_CONSTRAINT_MODE_VERSION),
            AB::F::from_u32(BATCH_CONSTRAINT_LOGUP_ONLY_MODE),
        ];
        for (offset, value) in separator.into_iter().enumerate() {
            self.transcript_bus.observe(
                builder,
                local.proof_idx,
                local.start_tidx + AB::Expr::from_usize(offset),
                value,
                local.is_enabled,
            );
        }

        self.starting_tidx_bus.send(
            builder,
            local.proof_idx,
            StartingTidxMessage {
                air_idx: AB::Expr::ZERO,
                tidx: local.start_tidx + AB::Expr::from_usize(BATCH_CONSTRAINT_MODE_SEPARATOR_LEN),
            },
            local.is_enabled,
        );
    }
}

pub struct RebasedProofShapeStartTraceGenerator;

impl RowMajorChip<F> for RebasedProofShapeStartTraceGenerator {
    type Ctx<'a> = &'a [Preflight];

    fn generate_trace(
        &self,
        preflights: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        let num_valid_rows = preflights.len();
        let height = required_height.unwrap_or_else(|| num_valid_rows.max(1).next_power_of_two());
        if height < num_valid_rows {
            return None;
        }
        let width = RebasedProofShapeStartCols::<F>::width();
        let mut values = vec![F::ZERO; height * width];
        values[..num_valid_rows * width]
            .par_chunks_exact_mut(width)
            .zip(preflights.par_iter())
            .enumerate()
            .for_each(|(proof_idx, (row, preflight))| {
                let start = preflight
                    .rebased_transcript
                    .expect("rebased proof-shape trace requires a certified start");
                let cols: &mut RebasedProofShapeStartCols<F> = row.borrow_mut();
                cols.is_enabled = F::ONE;
                cols.proof_idx = F::from_usize(proof_idx);
                cols.start_tidx = F::from_usize(start.start_tidx);
                cols.state = start.state;
            });
        Some(RowMajorMatrix::new(values, width))
    }
}

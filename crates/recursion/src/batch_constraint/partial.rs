use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper, SubAir};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;

use crate::{
    batch_constraint::bus::{
        BatchConstraintEndpointBus, BatchConstraintEndpointClaimBus,
        BatchConstraintEndpointClaimMessage, BatchConstraintEndpointMessage,
    },
    bus::{
        StackingModuleBus, StackingModuleMessage, TranscriptBus, TranscriptEndIndexBus,
        TranscriptEndIndexMessage,
    },
    subairs::proof_idx::{ProofIdxIoCols, ProofIdxSubAir},
    system::Preflight,
    tracegen::RowMajorChip,
};

#[repr(C)]
#[derive(AlignedBorrow, Copy, Clone, Debug, StructReflection)]
pub struct PartialBatchConstraintEndpointCols<T> {
    pub is_enabled: T,
    pub proof_idx: T,
    pub lambda_tidx: T,
    pub lambda: [T; D_EF],
    pub tidx: T,
    pub final_claim: [T; D_EF],
}

/// Consumes the standard sumcheck-to-stacking cursor and pairs it with the
/// claim already checked by `ExpressionClaimAir`. This closes every standard
/// `StackingModuleBus` interaction without instantiating stacking or WHIR.
#[derive(ColumnsAir)]
#[columns_via(PartialBatchConstraintEndpointCols<u8>)]
pub struct PartialBatchConstraintEndpointAir {
    pub transcript_bus: TranscriptBus,
    pub stacking_module_bus: StackingModuleBus,
    pub transcript_end_index_bus: TranscriptEndIndexBus,
    pub claim_bus: BatchConstraintEndpointClaimBus,
    pub endpoint_bus: BatchConstraintEndpointBus,
}

impl<Fld: Field> BaseAir<Fld> for PartialBatchConstraintEndpointAir {
    fn width(&self) -> usize {
        PartialBatchConstraintEndpointCols::<Fld>::width()
    }
}
impl<Fld: Field> BaseAirWithPublicValues<Fld> for PartialBatchConstraintEndpointAir {}
impl<Fld: Field> PartitionedBaseAir<Fld> for PartialBatchConstraintEndpointAir {}

impl<AB: AirBuilder + InteractionBuilder> Air<AB> for PartialBatchConstraintEndpointAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main
            .row_slice(0)
            .expect("window should have at least one row");
        let next_row = main
            .row_slice(1)
            .expect("window should have at least two rows");
        let local: &PartialBatchConstraintEndpointCols<AB::Var> = (*local_row).borrow();
        let next: &PartialBatchConstraintEndpointCols<AB::Var> = (*next_row).borrow();

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

        self.stacking_module_bus.receive(
            builder,
            local.proof_idx,
            StackingModuleMessage {
                tidx: local.tidx.into(),
            },
            local.is_enabled,
        );
        // LogUpOnly retains the ordinary verifier's lambda challenge slot for
        // transcript parity, even though no AIR-constraint term uses it.
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.lambda_tidx,
            local.lambda,
            local.is_enabled,
        );
        self.transcript_end_index_bus.receive(
            builder,
            local.proof_idx,
            TranscriptEndIndexMessage {
                tidx: local.tidx.into(),
            },
            local.is_enabled,
        );
        self.claim_bus.receive(
            builder,
            local.proof_idx,
            BatchConstraintEndpointClaimMessage {
                value: local.final_claim.map(Into::into),
            },
            local.is_enabled,
        );
        self.endpoint_bus.send(
            builder,
            local.proof_idx,
            BatchConstraintEndpointMessage {
                tidx: local.tidx.into(),
                final_claim: local.final_claim.map(Into::into),
            },
            local.is_enabled,
        );
    }
}

pub struct PartialBatchConstraintEndpointTraceGenerator;

impl RowMajorChip<F> for PartialBatchConstraintEndpointTraceGenerator {
    type Ctx<'a> = &'a [&'a Preflight];

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
        let width = PartialBatchConstraintEndpointCols::<F>::width();
        let mut values = vec![F::ZERO; height * width];
        values[..num_valid_rows * width]
            .par_chunks_exact_mut(width)
            .zip(preflights.par_iter())
            .enumerate()
            .for_each(|(proof_idx, (row, preflight))| {
                let cols: &mut PartialBatchConstraintEndpointCols<F> = row.borrow_mut();
                cols.is_enabled = F::ONE;
                cols.proof_idx = F::from_usize(proof_idx);
                cols.lambda_tidx = F::from_usize(preflight.batch_constraint.lambda_tidx);
                cols.lambda.copy_from_slice(
                    preflight.transcript_values_at(preflight.batch_constraint.lambda_tidx, D_EF),
                );
                cols.tidx = F::from_usize(preflight.batch_constraint.tidx_before_column_openings);
                cols.final_claim.copy_from_slice(
                    preflight
                        .batch_constraint
                        .final_claim
                        .as_basis_coefficients_slice(),
                );
            });
        Some(RowMajorMatrix::new(values, width))
    }
}

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    native_warp::{
        batching::OPENING_SECTION_POINT,
        bus::{
            NativeFoldedClaimBus, NativeFoldedClaimMessage, NativeOpeningClaimBus,
            NativeOpeningClaimMessage, NativeSumcheckChallengeBus, NativeSumcheckChallengeMessage,
            NativeVectorCoordinateBus, NativeVectorCoordinateMessage,
        },
    },
};

pub const VECTOR_SOURCE_SUMCHECK: usize = 0;
pub const VECTOR_SOURCE_TRANSCRIPT: usize = 1;
pub const VECTOR_SOURCE_FOLDED_CLAIM: usize = 2;
pub const VECTOR_SOURCE_OPENING_POINT: usize = 3;
pub const VECTOR_SOURCE_BOOLEAN: usize = 4;

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeVectorCoordinateCols<T> {
    pub active: T,
    pub vector: T,
    pub coordinate: T,
    pub source_kind: [T; 5],
    pub source: T,
    pub section: T,
    pub proof_idx: T,
    pub tidx: T,
    pub bit: T,
    pub value: [T; D_EF],
    pub lookup_count: T,
}

#[derive(ColumnsAir)]
#[columns_via(NativeVectorCoordinateCols<u8>)]
pub struct NativeVectorCoordinateAir {
    pub vector_bus: NativeVectorCoordinateBus,
    pub sumcheck_bus: NativeSumcheckChallengeBus,
    pub transcript_bus: TranscriptBus,
    pub folded_bus: NativeFoldedClaimBus,
    pub opening_bus: NativeOpeningClaimBus,
}

impl BaseAirWithPublicValues<F> for NativeVectorCoordinateAir {}
impl PartitionedBaseAir<F> for NativeVectorCoordinateAir {}
impl<F> BaseAir<F> for NativeVectorCoordinateAir {
    fn width(&self) -> usize {
        NativeVectorCoordinateCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeVectorCoordinateAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("native vector coordinate row");
        let local: &NativeVectorCoordinateCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        for flag in local.source_kind {
            builder.assert_bool(flag);
        }
        builder.when(local.active).assert_one(
            local
                .source_kind
                .into_iter()
                .map(AB::Expr::from)
                .sum::<AB::Expr>(),
        );
        // Padding rows are not vector-table authorities.  In particular,
        // `lookup_count` must not let an inactive row satisfy an EQ consumer
        // without any transcript/sumcheck/folded/opening source event.
        builder
            .when(AB::Expr::ONE - local.active)
            .assert_zero(local.lookup_count);
        self.vector_bus.add_key_with_lookups(
            builder,
            NativeVectorCoordinateMessage {
                proof_idx: local.proof_idx.into(),
                vector: local.vector.into(),
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active * local.lookup_count,
        );
        self.sumcheck_bus.lookup_key(
            builder,
            NativeSumcheckChallengeMessage {
                proof_idx: local.proof_idx.into(),
                kind: local.section.into(),
                round: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active * local.source_kind[VECTOR_SOURCE_SUMCHECK],
        );
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.tidx,
            local.value,
            local.active * local.source_kind[VECTOR_SOURCE_TRANSCRIPT],
        );
        self.folded_bus.receive(
            builder,
            NativeFoldedClaimMessage {
                proof_idx: local.proof_idx.into(),
                section: local.section.into(),
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active * local.source_kind[VECTOR_SOURCE_FOLDED_CLAIM],
        );
        self.opening_bus.receive(
            builder,
            NativeOpeningClaimMessage {
                proof_idx: local.proof_idx.into(),
                claim: local.source.into(),
                section: AB::Expr::from_usize(OPENING_SECTION_POINT),
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active * local.source_kind[VECTOR_SOURCE_OPENING_POINT],
        );
        builder
            .when(local.active * local.source_kind[VECTOR_SOURCE_BOOLEAN])
            .assert_bool(local.bit);
        builder
            .when(local.active * local.source_kind[VECTOR_SOURCE_BOOLEAN])
            .assert_eq(local.value[0], local.bit);
        for limb in &local.value[1..] {
            builder
                .when(local.active * local.source_kind[VECTOR_SOURCE_BOOLEAN])
                .assert_zero(*limb);
        }
    }
}

#[derive(Clone, Debug)]
pub enum NativeVectorSource {
    Sumcheck { kind: u32 },
    Transcript { proof_idx: u32, tidx: Vec<u32> },
    FoldedClaim { section: u32 },
    OpeningPoint { claim: u32 },
    Boolean,
}

#[derive(Clone, Debug)]
pub struct NativeVectorTraceInput<'a> {
    pub proof_idx: u32,
    pub vector: u32,
    pub values: &'a [EF],
    pub source: NativeVectorSource,
    pub lookup_counts: &'a [u32],
}

pub fn generate_native_vector_coordinate_trace(
    vectors: &[NativeVectorTraceInput<'_>],
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    let valid_rows = vectors
        .iter()
        .map(|vector| vector.values.len())
        .sum::<usize>();
    if valid_rows == 0
        || vectors.iter().any(|vector| {
            vector.values.len() != vector.lookup_counts.len()
                || matches!(
                    &vector.source,
                    NativeVectorSource::Transcript { proof_idx, .. }
                        if *proof_idx != vector.proof_idx
                )
        })
    {
        return None;
    }
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeVectorCoordinateCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    let mut row_index = 0usize;
    for vector in vectors {
        if matches!(&vector.source, NativeVectorSource::Transcript { tidx, .. } if tidx.len() != vector.values.len())
        {
            return None;
        }
        for (coordinate, &value) in vector.values.iter().enumerate() {
            let cols: &mut NativeVectorCoordinateCols<F> =
                trace[row_index * width..(row_index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.proof_idx = F::from_u32(vector.proof_idx);
            cols.vector = F::from_u32(vector.vector);
            cols.coordinate = F::from_usize(coordinate);
            cols.value
                .copy_from_slice(value.as_basis_coefficients_slice());
            cols.lookup_count = F::from_u32(vector.lookup_counts[coordinate]);
            match &vector.source {
                NativeVectorSource::Sumcheck { kind } => {
                    cols.source_kind[VECTOR_SOURCE_SUMCHECK] = F::ONE;
                    cols.section = F::from_u32(*kind);
                }
                NativeVectorSource::Transcript { proof_idx, tidx } => {
                    cols.source_kind[VECTOR_SOURCE_TRANSCRIPT] = F::ONE;
                    debug_assert_eq!(*proof_idx, vector.proof_idx);
                    cols.tidx = F::from_u32(tidx[coordinate]);
                }
                NativeVectorSource::FoldedClaim { section } => {
                    cols.source_kind[VECTOR_SOURCE_FOLDED_CLAIM] = F::ONE;
                    cols.section = F::from_u32(*section);
                }
                NativeVectorSource::OpeningPoint { claim } => {
                    cols.source_kind[VECTOR_SOURCE_OPENING_POINT] = F::ONE;
                    cols.source = F::from_u32(*claim);
                }
                NativeVectorSource::Boolean => {
                    let basis: &[F] = value.as_basis_coefficients_slice();
                    if basis[1..].iter().any(|&limb| limb != F::ZERO)
                        || (basis[0] != F::ZERO && basis[0] != F::ONE)
                    {
                        return None;
                    }
                    cols.source_kind[VECTOR_SOURCE_BOOLEAN] = F::ONE;
                    cols.bit = basis[0];
                }
            }
            row_index += 1;
        }
    }
    Some(RowMajorMatrix::new(trace, width))
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use openvm_stark_backend::air_builders::debug::check_constraints;
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config as NativeSC;

    use super::*;

    fn air() -> NativeVectorCoordinateAir {
        NativeVectorCoordinateAir {
            vector_bus: NativeVectorCoordinateBus::new(0),
            sumcheck_bus: NativeSumcheckChallengeBus::new(1),
            transcript_bus: TranscriptBus::new(2),
            folded_bus: NativeFoldedClaimBus::new(3),
            opening_bus: NativeOpeningClaimBus::new(4),
        }
    }

    fn check(trace: &RowMajorMatrix<F>) {
        check_constraints::<_, NativeSC>(
            &air(),
            "NativeVectorCoordinateAir",
            &None,
            &[trace.as_view()],
            &[],
        );
    }

    #[test]
    fn inactive_vector_row_cannot_own_lookup() {
        let width = NativeVectorCoordinateCols::<F>::width();
        let honest_padding = RowMajorMatrix::new(F::zero_vec(width), width);
        check(&honest_padding);

        let mut forged = honest_padding;
        let cols: &mut NativeVectorCoordinateCols<F> = forged.values.as_mut_slice().borrow_mut();
        cols.proof_idx = F::from_u32(3);
        cols.vector = F::from_u32(7);
        cols.coordinate = F::from_u32(11);
        cols.value[0] = F::from_u32(19);
        cols.lookup_count = F::ONE;
        assert!(catch_unwind(AssertUnwindSafe(|| check(&forged))).is_err());
    }
}

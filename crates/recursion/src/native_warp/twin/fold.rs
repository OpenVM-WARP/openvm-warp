use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::native_warp::{
    bus::{
        NativeClaimValueBus, NativeClaimValueMessage, NativeEqResultBus, NativeEqResultMessage,
        NativeFoldedClaimBus, NativeFoldedClaimMessage,
    },
    ext::{ext_field_add, ext_field_multiply},
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeTwinFoldCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub section: T,
    pub coordinate: T,
    pub source: T,
    pub is_first_source: T,
    pub is_last_source: T,
    pub weight: [T; D_EF],
    pub value: [T; D_EF],
    pub accumulator_before: [T; D_EF],
    pub accumulator_after: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(NativeTwinFoldCols<u8>)]
pub struct NativeTwinFoldAir {
    pub claim_bus: NativeClaimValueBus,
    pub eq_bus: NativeEqResultBus,
    pub folded_bus: NativeFoldedClaimBus,
    pub input_arity: usize,
    pub weight_group_offset: usize,
}

impl BaseAirWithPublicValues<F> for NativeTwinFoldAir {}
impl PartitionedBaseAir<F> for NativeTwinFoldAir {}

impl<F> BaseAir<F> for NativeTwinFoldAir {
    fn width(&self) -> usize {
        NativeTwinFoldCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTwinFoldAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("native twin fold row"),
            main.row_slice(1).expect("native twin fold next row"),
        );
        let local: &NativeTwinFoldCols<AB::Var> = (*local).borrow();
        let next: &NativeTwinFoldCols<AB::Var> = (*next).borrow();
        builder.assert_bool(local.active);
        builder.assert_bool(local.is_first_source);
        builder.assert_bool(local.is_last_source);
        builder.when_first_row().assert_one(local.active);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when(local.active * local.is_first_source)
            .assert_zero(local.source);
        builder
            .when(local.active * local.is_last_source)
            .assert_eq(local.source, AB::Expr::from_usize(self.input_arity - 1));

        self.eq_bus.lookup_key(
            builder,
            NativeEqResultMessage {
                proof_idx: local.proof_idx.into(),
                group: AB::Expr::from_usize(self.weight_group_offset) + local.source,
                value: local.weight.map(Into::into),
            },
            local.active,
        );
        self.claim_bus.receive(
            builder,
            NativeClaimValueMessage {
                proof_idx: local.proof_idx.into(),
                source: local.source.into(),
                section: local.section.into(),
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active,
        );
        let zero = [AB::Expr::ZERO; D_EF];
        assert_array_eq(
            &mut builder.when(local.active * local.is_first_source),
            local.accumulator_before,
            zero,
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.accumulator_after,
            ext_field_add::<AB::Expr>(
                local.accumulator_before,
                ext_field_multiply::<AB::Expr>(local.weight, local.value),
            ),
        );
        let same_coordinate = next.active * (AB::Expr::ONE - next.is_first_source);
        let mut transition = builder.when_transition();
        let mut when_same = transition.when(same_coordinate);
        when_same.assert_zero(local.is_last_source);
        when_same.assert_eq(next.proof_idx, local.proof_idx);
        when_same.assert_eq(next.section, local.section);
        when_same.assert_eq(next.coordinate, local.coordinate);
        when_same.assert_eq(next.source, local.source + AB::F::ONE);
        assert_array_eq(
            &mut when_same,
            next.accumulator_before,
            local.accumulator_after,
        );
        let starts_coordinate = next.active * next.is_first_source;
        let mut transition = builder.when_transition();
        let mut when_next = transition.when(starts_coordinate);
        when_next.assert_one(local.is_last_source);

        self.folded_bus.send(
            builder,
            NativeFoldedClaimMessage {
                proof_idx: local.proof_idx.into(),
                section: local.section.into(),
                coordinate: local.coordinate.into(),
                value: local.accumulator_after.map(Into::into),
            },
            local.active * local.is_last_source,
        );
    }
}

pub fn generate_native_twin_fold_trace(
    proof_idx: usize,
    sections: &[(usize, &[Vec<EF>])],
    weights: &[EF],
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if weights.is_empty()
        || sections.is_empty()
        || sections.iter().any(|(_, sources)| {
            sources.len() != weights.len()
                || sources
                    .first()
                    .is_some_and(|first| sources.iter().any(|source| source.len() != first.len()))
        })
    {
        return None;
    }
    let valid_rows = sections
        .iter()
        .map(|(_, sources)| sources[0].len() * weights.len())
        .sum::<usize>();
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeTwinFoldCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    let mut row_index = 0usize;
    for &(section, sources) in sections {
        for coordinate in 0..sources[0].len() {
            let mut accumulator = EF::ZERO;
            for source in 0..sources.len() {
                let before = accumulator;
                accumulator += weights[source] * sources[source][coordinate];
                let row = &mut trace[row_index * width..(row_index + 1) * width];
                let cols: &mut NativeTwinFoldCols<F> = row.borrow_mut();
                cols.active = F::ONE;
                cols.proof_idx = F::from_usize(proof_idx);
                cols.section = F::from_usize(section);
                cols.coordinate = F::from_usize(coordinate);
                cols.source = F::from_usize(source);
                cols.is_first_source = F::from_bool(source == 0);
                cols.is_last_source = F::from_bool(source + 1 == sources.len());
                cols.weight
                    .copy_from_slice(weights[source].as_basis_coefficients_slice());
                cols.value
                    .copy_from_slice(sources[source][coordinate].as_basis_coefficients_slice());
                cols.accumulator_before
                    .copy_from_slice(before.as_basis_coefficients_slice());
                cols.accumulator_after
                    .copy_from_slice(accumulator.as_basis_coefficients_slice());
                row_index += 1;
            }
        }
    }
    Some(RowMajorMatrix::new(trace, width))
}

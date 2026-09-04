use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::F;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::native_warp::bus::{
    NativeFreshCountBus, NativeFreshCountMessage, NativeInputSlotLayoutBus,
    NativeInputSlotLayoutMessage, NativePcdStateReadBus, NativePcdStateReadMessage,
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeInputSlotLayoutCols<T> {
    pub active: T,
    pub variant: T,
    pub source: T,
    pub kind: [T; 3],
    pub is_first: T,
    pub is_last: T,
    pub fresh_count: T,
    pub prior_present: T,
    pub fresh_remaining: T,
    pub prior_remaining: T,
    pub fresh_is_zero: T,
    pub fresh_inverse: T,
    pub lookup_count: T,
}

#[derive(ColumnsAir)]
#[columns_via(NativeInputSlotLayoutCols<u8>)]
pub struct NativeInputSlotLayoutAir {
    pub bus: NativeInputSlotLayoutBus,
    pub fresh_count_bus: NativeFreshCountBus,
    pub state_read_bus: NativePcdStateReadBus,
    pub accumulator_is_set_offset: usize,
    pub input_arity: usize,
}

impl BaseAirWithPublicValues<F> for NativeInputSlotLayoutAir {}
impl PartitionedBaseAir<F> for NativeInputSlotLayoutAir {}
impl<F> BaseAir<F> for NativeInputSlotLayoutAir {
    fn width(&self) -> usize {
        NativeInputSlotLayoutCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeInputSlotLayoutAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("native input slot layout row");
        let local: &NativeInputSlotLayoutCols<AB::Var> = (*row).borrow();
        for flag in [
            local.active,
            local.is_first,
            local.is_last,
            local.prior_present,
            local.prior_remaining,
            local.fresh_is_zero,
        ] {
            builder.assert_bool(flag);
        }
        for flag in local.kind {
            builder.assert_bool(flag);
        }
        builder
            .when(local.active)
            .assert_one(local.kind.into_iter().map(AB::Expr::from).sum::<AB::Expr>());
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder
            .when(local.active * local.is_first)
            .assert_zero(local.source);
        builder
            .when(local.active * local.is_first)
            .assert_eq(local.fresh_remaining, local.fresh_count);
        builder
            .when(local.active * local.is_first)
            .assert_eq(local.prior_remaining, local.prior_present);
        builder.when(local.active).assert_eq(
            local.variant,
            local.fresh_count + local.prior_present * AB::Expr::from_usize(self.input_arity + 1),
        );
        builder
            .when(local.active * local.fresh_is_zero)
            .assert_zero(local.fresh_remaining);
        builder
            .when(local.active * (AB::Expr::ONE - local.fresh_is_zero))
            .assert_one(local.fresh_remaining * local.fresh_inverse);
        builder
            .when(local.active)
            .assert_eq(local.kind[0], AB::Expr::ONE - local.fresh_is_zero);
        builder
            .when(local.active)
            .assert_eq(local.kind[1], local.fresh_is_zero * local.prior_remaining);
        builder.when(local.active).assert_eq(
            local.kind[2],
            local.fresh_is_zero * (AB::Expr::ONE - local.prior_remaining),
        );
        let next_row = main.row_slice(1).expect("native input slot next row");
        let next: &NativeInputSlotLayoutCols<AB::Var> = (*next_row).borrow();
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        let mut transition = builder.when_transition();
        let mut same = transition.when(next.active);
        same.assert_eq(next.source, local.source + AB::F::ONE);
        same.assert_eq(next.variant, local.variant);
        same.assert_eq(next.fresh_count, local.fresh_count);
        same.assert_eq(next.prior_present, local.prior_present);
        same.assert_eq(next.fresh_remaining, local.fresh_remaining - local.kind[0]);
        same.assert_eq(next.prior_remaining, local.prior_remaining - local.kind[1]);
        builder
            .when(local.active * local.is_last)
            .assert_eq(local.source, AB::Expr::from_usize(self.input_arity - 1));
        builder
            .when_transition()
            .when(local.active - next.active)
            .assert_one(local.is_last);
        self.fresh_count_bus.lookup_key(
            builder,
            NativeFreshCountMessage {
                count: local.fresh_count.into(),
            },
            local.active * local.is_first,
        );
        self.state_read_bus.lookup_key(
            builder,
            NativePcdStateReadMessage {
                state: AB::Expr::ZERO,
                coordinate: AB::Expr::from_usize(self.accumulator_is_set_offset),
                value: local.prior_present.into(),
            },
            local.active * local.is_first,
        );
        self.bus.add_key_with_lookups(
            builder,
            NativeInputSlotLayoutMessage {
                variant: local.variant.into(),
                source: local.source.into(),
                kind: local.kind.map(Into::into),
            },
            local.lookup_count,
        );
    }
}

pub fn generate_native_input_slot_layout_trace(
    input_arity: usize,
    selected_fresh_count: usize,
    selected_prior_present: bool,
    selected_lookup_counts: &[u32],
) -> Option<RowMajorMatrix<F>> {
    if input_arity < 2
        || !input_arity.is_power_of_two()
        || selected_fresh_count == 0
        || selected_fresh_count + usize::from(selected_prior_present) > input_arity
        || selected_lookup_counts.len() != input_arity
    {
        return None;
    }
    let valid_rows = input_arity;
    let height = valid_rows.next_power_of_two();
    let width = NativeInputSlotLayoutCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    let selected_variant =
        selected_fresh_count + usize::from(selected_prior_present) * (input_arity + 1);
    for source in 0..input_arity {
        let cols: &mut NativeInputSlotLayoutCols<F> =
            trace[source * width..(source + 1) * width].borrow_mut();
        let fresh_remaining = selected_fresh_count.saturating_sub(source);
        let prior_remaining = selected_prior_present && source <= selected_fresh_count;
        cols.active = F::ONE;
        cols.variant = F::from_usize(selected_variant);
        cols.source = F::from_usize(source);
        cols.is_first = F::from_bool(source == 0);
        cols.is_last = F::from_bool(source + 1 == input_arity);
        cols.fresh_count = F::from_usize(selected_fresh_count);
        cols.prior_present = F::from_bool(selected_prior_present);
        cols.fresh_remaining = F::from_usize(fresh_remaining);
        cols.prior_remaining = F::from_bool(prior_remaining);
        cols.fresh_is_zero = F::from_bool(fresh_remaining == 0);
        cols.fresh_inverse = if fresh_remaining == 0 {
            F::ZERO
        } else {
            F::from_usize(fresh_remaining).inverse()
        };
        cols.kind[if source < selected_fresh_count {
            0
        } else if selected_prior_present && source == selected_fresh_count {
            1
        } else {
            2
        }] = F::ONE;
        cols.lookup_count = F::from_u32(selected_lookup_counts[source]);
    }
    Some(RowMajorMatrix::new(trace, width))
}

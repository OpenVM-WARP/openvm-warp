use core::borrow::Borrow;

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, PrimeCharacteristicRing};
use p3_matrix::Matrix;

use crate::native_warp::{
    bus::{
        NativeEqResultBus, NativeEqResultMessage, NativeVectorCoordinateBus,
        NativeVectorCoordinateMessage,
    },
    ext::{eq_1, ext_field_multiply},
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeEqEvaluationCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub group: T,
    pub coordinate: T,
    pub is_first: T,
    pub is_last: T,
    pub is_first_group: T,
    /// Locally constrained transition selectors.  Reading these from the
    /// next row avoids multiplying three dynamic flags under the transition
    /// selector and keeps the recursive AIR at degree four.
    pub continues_group: T,
    pub starts_first_group: T,
    pub starts_nonfirst_group: T,
    pub first_inverse: T,
    pub last_inverse: T,
    pub first_group_inverse: T,
    pub left_vector: T,
    pub right_vector: T,
    pub left: [T; D_EF],
    pub right: [T; D_EF],
    pub accumulator_before: [T; D_EF],
    pub accumulator_after: [T; D_EF],
    pub lookup_count: T,
}

#[derive(ColumnsAir)]
#[columns_via(NativeEqEvaluationCols<u8>)]
pub struct NativeEqEvaluationAir {
    pub result_bus: NativeEqResultBus,
    pub vector_bus: NativeVectorCoordinateBus,
    pub dimensions: usize,
    pub group_offset: usize,
}

impl BaseAirWithPublicValues<F> for NativeEqEvaluationAir {}
impl PartitionedBaseAir<F> for NativeEqEvaluationAir {}

impl<F> BaseAir<F> for NativeEqEvaluationAir {
    fn width(&self) -> usize {
        NativeEqEvaluationCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeEqEvaluationAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("native eq row"),
            main.row_slice(1).expect("native eq next row"),
        );
        let local: &NativeEqEvaluationCols<AB::Var> = (*local).borrow();
        let next: &NativeEqEvaluationCols<AB::Var> = (*next).borrow();

        builder.assert_bool(local.active);
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);
        builder.assert_bool(local.is_first_group);
        builder.assert_bool(local.continues_group);
        builder.assert_bool(local.starts_first_group);
        builder.assert_bool(local.starts_nonfirst_group);
        builder.assert_eq(
            local.continues_group,
            local.active * (AB::Expr::ONE - local.is_first),
        );
        builder.assert_eq(
            local.starts_first_group,
            local.active * local.is_first * local.is_first_group,
        );
        builder.assert_eq(
            local.starts_nonfirst_group,
            local.active * local.is_first * (AB::Expr::ONE - local.is_first_group),
        );
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first_group);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);

        builder
            .when(local.active * local.is_first)
            .assert_zero(local.coordinate);
        builder
            .when(local.active * local.is_first_group)
            .assert_one(local.is_first);
        builder
            .when(local.active * (AB::Expr::ONE - local.is_first))
            .assert_one(local.coordinate * local.first_inverse);
        let distance = AB::Expr::from_usize(self.dimensions - 1) - local.coordinate;
        builder
            .when(local.active * local.is_last)
            .assert_zero(distance.clone());
        builder
            .when(local.active * (AB::Expr::ONE - local.is_last))
            .assert_one(distance * local.last_inverse);
        // `is_first_group` marks the first *row* of this proof's EQ table, not
        // every coordinate belonging to its first group.  Flatten the
        // (group, coordinate) pair so only (group_offset, 0) has distance
        // zero.  This also gives each proof in a merged trace a clean reset.
        let first_group_distance = (local.group - AB::Expr::from_usize(self.group_offset))
            * AB::Expr::from_usize(self.dimensions)
            + local.coordinate;
        builder
            .when(local.active * local.is_first_group)
            .assert_zero(first_group_distance.clone());
        builder
            .when(local.active * (AB::Expr::ONE - local.is_first_group))
            .assert_one(first_group_distance * local.first_group_inverse);
        for (vector, value) in [
            (local.left_vector, local.left),
            (local.right_vector, local.right),
        ] {
            self.vector_bus.lookup_key(
                builder,
                NativeVectorCoordinateMessage {
                    proof_idx: local.proof_idx.into(),
                    vector: vector.into(),
                    coordinate: local.coordinate.into(),
                    value: value.map(Into::into),
                },
                local.active,
            );
        }

        let one = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            }
        });
        assert_array_eq(
            &mut builder.when(local.active * local.is_first),
            local.accumulator_before,
            one,
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.accumulator_after,
            ext_field_multiply::<AB::Expr>(
                local.accumulator_before,
                eq_1::<AB::Expr>(local.left, local.right),
            ),
        );

        let mut transition = builder.when_transition();
        let mut when_same = transition.when(next.continues_group);
        when_same.assert_zero(local.is_last);
        when_same.assert_eq(next.proof_idx, local.proof_idx);
        when_same.assert_zero(next.is_first_group);
        when_same.assert_eq(next.group, local.group);
        when_same.assert_eq(next.coordinate, local.coordinate + AB::F::ONE);
        when_same.assert_eq(next.left_vector, local.left_vector);
        when_same.assert_eq(next.right_vector, local.right_vector);
        assert_array_eq(
            &mut when_same,
            next.accumulator_before,
            local.accumulator_after,
        );

        let starts_group = next.starts_first_group + next.starts_nonfirst_group;
        let mut transition = builder.when_transition();
        let mut when_next = transition.when(starts_group);
        when_next.assert_one(local.is_last);
        when_next
            .when(next.starts_first_group)
            .assert_eq(next.proof_idx, local.proof_idx + AB::F::ONE);
        when_next
            .when(next.starts_nonfirst_group)
            .assert_eq(next.proof_idx, local.proof_idx);
        when_next
            .when(next.starts_nonfirst_group)
            .assert_eq(next.group, local.group + AB::F::ONE);
        builder
            .when_transition()
            .when(local.active - next.active)
            .assert_one(local.is_last);

        self.result_bus.add_key_with_lookups(
            builder,
            NativeEqResultMessage {
                proof_idx: local.proof_idx.into(),
                group: local.group.into(),
                value: local.accumulator_after.map(Into::into),
            },
            local.lookup_count,
        );
    }
}

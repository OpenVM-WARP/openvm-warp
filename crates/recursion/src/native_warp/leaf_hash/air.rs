use core::borrow::Borrow;

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{CHUNK, DIGEST_SIZE, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;

use crate::{
    bus::{Poseidon2PermuteBus, Poseidon2PermuteMessage},
    native_warp::bus::{
        NativeLeafValueBus, NativeLeafValueMessage, NativeOpeningLeafBus, NativeOpeningLeafMessage,
    },
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeLeafHashCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub tree_id: T,
    pub leaf_index: T,
    pub block: T,
    pub is_first: T,
    pub is_last: T,
    /// Prefix mask selecting fields overwritten in this sponge block.
    pub mask: [T; CHUNK],
    /// Number of circuit consumers for each authenticated field.
    pub lookup_count: [T; CHUNK],
    pub before: [T; POSEIDON2_WIDTH],
    pub input: [T; POSEIDON2_WIDTH],
    pub output: [T; POSEIDON2_WIDTH],
}

#[derive(ColumnsAir)]
#[columns_via(NativeLeafHashCols<u8>)]
pub struct NativeLeafHashAir {
    pub permute_bus: Poseidon2PermuteBus,
    pub value_bus: NativeLeafValueBus,
    pub leaf_bus: NativeOpeningLeafBus,
}

impl BaseAirWithPublicValues<F> for NativeLeafHashAir {}
impl PartitionedBaseAir<F> for NativeLeafHashAir {}

impl<F> BaseAir<F> for NativeLeafHashAir {
    fn width(&self) -> usize {
        NativeLeafHashCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeLeafHashAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("native leaf hash row"),
            main.row_slice(1).expect("native leaf hash next row"),
        );
        let local: &NativeLeafHashCols<AB::Var> = (*local).borrow();
        let next: &NativeLeafHashCols<AB::Var> = (*next).borrow();

        builder.assert_bool(local.active);
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);
        builder.when_first_row().assert_one(local.active);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        for index in 0..CHUNK {
            builder.assert_bool(local.mask[index]);
            if index + 1 < CHUNK {
                builder
                    .when(local.active * (AB::Expr::ONE - local.mask[index]))
                    .assert_zero(local.mask[index + 1]);
            }
            builder
                .when(local.active * local.mask[index])
                .assert_eq(local.input[index], local.input[index]);
            builder
                .when(local.active * (AB::Expr::ONE - local.mask[index]))
                .assert_eq(local.input[index], local.before[index]);
            self.value_bus.add_key_with_lookups(
                builder,
                NativeLeafValueMessage {
                    proof_idx: local.proof_idx.into(),
                    tree_id: local.tree_id.into(),
                    index: local.leaf_index.into(),
                    position: local.block * AB::Expr::from_usize(CHUNK)
                        + AB::Expr::from_usize(index),
                    value: local.input[index].into(),
                },
                local.lookup_count[index],
            );
        }
        for index in CHUNK..POSEIDON2_WIDTH {
            builder
                .when(local.active)
                .assert_eq(local.input[index], local.before[index]);
        }
        for value in local.before {
            builder
                .when(local.active * local.is_first)
                .assert_zero(value);
        }
        builder
            .when(local.active * local.is_first)
            .assert_zero(local.block);

        let same_leaf = next.active * (AB::Expr::ONE - next.is_first);
        let mut transition = builder.when_transition();
        let mut when_same = transition.when(same_leaf);
        when_same.assert_zero(local.is_last);
        when_same.assert_eq(next.proof_idx, local.proof_idx);
        when_same.assert_eq(next.tree_id, local.tree_id);
        when_same.assert_eq(next.leaf_index, local.leaf_index);
        when_same.assert_eq(next.block, local.block + AB::F::ONE);
        assert_array_eq(&mut when_same, next.before, local.output);

        let starts_leaf = next.active * next.is_first;
        let mut transition = builder.when_transition();
        let mut when_next = transition.when(starts_leaf);
        when_next.assert_one(local.is_last);
        builder
            .when_transition()
            .when(local.active - next.active)
            .assert_one(local.is_last);

        self.permute_bus.lookup_key(
            builder,
            Poseidon2PermuteMessage {
                input: local.input.map(Into::into),
                output: local.output.map(Into::into),
            },
            local.active,
        );
        self.leaf_bus.send(
            builder,
            NativeOpeningLeafMessage {
                proof_idx: local.proof_idx.into(),
                tree_id: local.tree_id.into(),
                index: local.leaf_index.into(),
                digest: core::array::from_fn(|index| local.output[index].into()),
            },
            local.active * local.is_last,
        );
    }
}

const _: () = assert!(CHUNK == DIGEST_SIZE);

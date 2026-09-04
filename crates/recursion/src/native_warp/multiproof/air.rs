use core::borrow::Borrow;

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::Matrix;

use crate::{
    bus::{Poseidon2CompressBus, Poseidon2CompressMessage},
    native_warp::bus::{
        NativeMerkleNodeBus, NativeMerkleNodeMessage, NativeMerkleRootBus, NativeMerkleRootMessage,
        NativeOpeningLeafBus, NativeOpeningLeafMessage,
    },
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeMerkleCompressionCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub tree_id: T,
    /// Child level. Leaves are level zero; output is at `level + 1`.
    pub level: T,
    /// Height of the tree this row belongs to.
    ///
    /// Deliberately a witness column, unlike the depths on the AIRs that *consume*
    /// the root bus. One instance of this AIR carries rows for trees of several
    /// different heights at once -- application outer and inner, scalar outer and
    /// inner -- so no single constant would serve. Its honesty comes from the
    /// consumer: whoever receives the root pins the depth to a keygen constant for
    /// that tree's layout class, so a row claiming the wrong height finds no
    /// receiver and the bus goes unbalanced.
    pub depth: T,
    pub parent_index: T,
    pub is_root: T,
    pub root_inverse: T,
    /// Origin one-hot order: opened leaf, proof sibling, computed node.
    pub left_origin: [T; 3],
    pub right_origin: [T; 3],
    /// Number of verifier queries represented by an opened frontier leaf.
    pub left_leaf_multiplicity: T,
    pub left_leaf_multiplicity_inverse: T,
    pub right_leaf_multiplicity: T,
    pub right_leaf_multiplicity_inverse: T,
    pub left: [T; DIGEST_SIZE],
    pub right: [T; DIGEST_SIZE],
    pub output: [T; DIGEST_SIZE],
}

#[derive(ColumnsAir)]
#[columns_via(NativeMerkleCompressionCols<u8>)]
pub struct NativeMerkleMultiproofAir {
    pub compress_bus: Poseidon2CompressBus,
    pub leaf_bus: NativeOpeningLeafBus,
    pub node_bus: NativeMerkleNodeBus,
    pub root_bus: NativeMerkleRootBus,
}

/// Converts a fully compressed rows-per-query subtree root into one outer
/// Merkle leaf. For `rows_per_query == 1`, trace generation supplies the row
/// digest directly through `leaf_bus` and uses the bypass flag.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeMerkleLeafAdapterCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub outer_multiplicity: T,
    pub outer_multiplicity_inverse: T,
    pub inner_tree_id: T,
    pub outer_tree_id: T,
    pub query_index: T,
    pub digest: [T; DIGEST_SIZE],
}

#[derive(ColumnsAir)]
#[columns_via(NativeMerkleLeafAdapterCols<u8>)]
pub struct NativeMerkleLeafAdapterAir {
    pub leaf_bus: NativeOpeningLeafBus,
    pub root_bus: NativeMerkleRootBus,
    /// Depth of the per-query subtree over one leaf group, `log2(rows_per_query)`.
    ///
    /// A keygen constant, and the branch below is chosen at build time from it. Both
    /// were witness columns -- `inner_depth` fed straight into the root bus, and
    /// nothing tied `bypass` to `inner_depth == 0` -- so a prover could claim either
    /// shape for a tree. Every producer feeding this instance (the fresh scalar
    /// oracle and the prior accumulator) is accumulator-class, so one constant is
    /// enough here; the projection's adapter serves two classes and is instantiated
    /// once per class instead.
    pub inner_depth: usize,
}

impl BaseAirWithPublicValues<F> for NativeMerkleLeafAdapterAir {}
impl PartitionedBaseAir<F> for NativeMerkleLeafAdapterAir {}
impl<F> BaseAir<F> for NativeMerkleLeafAdapterAir {
    fn width(&self) -> usize {
        NativeMerkleLeafAdapterCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeMerkleLeafAdapterAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("native Merkle leaf adapter row");
        let local: &NativeMerkleLeafAdapterCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder
            .when(local.active)
            .assert_one(local.outer_multiplicity * local.outer_multiplicity_inverse);
        if self.inner_depth == 0 {
            // One row per leaf: there is no subtree, so the row digest *is* the outer
            // leaf and arrives on the leaf bus.
            self.leaf_bus.receive(
                builder,
                NativeOpeningLeafMessage {
                    proof_idx: local.proof_idx.into(),
                    tree_id: local.inner_tree_id.into(),
                    index: AB::Expr::ZERO,
                    digest: local.digest.map(Into::into),
                },
                local.active,
            );
        } else {
            self.root_bus.receive(
                builder,
                NativeMerkleRootMessage {
                    proof_idx: local.proof_idx.into(),
                    tree_id: local.inner_tree_id.into(),
                    depth: AB::Expr::from_usize(self.inner_depth),
                    digest: local.digest.map(Into::into),
                },
                local.active,
            );
        }
        self.leaf_bus.send(
            builder,
            NativeOpeningLeafMessage {
                proof_idx: local.proof_idx.into(),
                tree_id: local.outer_tree_id.into(),
                index: local.query_index.into(),
                digest: local.digest.map(Into::into),
            },
            local.active * local.outer_multiplicity,
        );
    }
}

impl BaseAirWithPublicValues<F> for NativeMerkleMultiproofAir {}
impl PartitionedBaseAir<F> for NativeMerkleMultiproofAir {}

impl<F> BaseAir<F> for NativeMerkleMultiproofAir {
    fn width(&self) -> usize {
        NativeMerkleCompressionCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeMerkleMultiproofAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("native Merkle compression row");
        let local: &NativeMerkleCompressionCols<AB::Var> = (*row).borrow();

        builder.assert_bool(local.active);
        builder.assert_bool(local.is_root);
        for flag in local.left_origin.into_iter().chain(local.right_origin) {
            builder.assert_bool(flag);
        }
        builder.when(local.active).assert_one(
            local
                .left_origin
                .iter()
                .copied()
                .map(AB::Expr::from)
                .sum::<AB::Expr>(),
        );
        builder.when(local.active).assert_one(
            local
                .right_origin
                .iter()
                .copied()
                .map(AB::Expr::from)
                .sum::<AB::Expr>(),
        );

        let output_level = local.level + AB::F::ONE;
        let distance_to_root = local.depth - output_level.clone();
        builder
            .when(local.active * local.is_root)
            .assert_zero(distance_to_root.clone());
        builder
            .when(local.active * (AB::Expr::ONE - local.is_root))
            .assert_one(distance_to_root * local.root_inverse);
        builder
            .when(local.active * local.is_root)
            .assert_zero(local.parent_index);

        let left_index = local.parent_index * AB::F::TWO;
        let right_index = left_index.clone() + AB::Expr::ONE;
        let left_is_leaf = local.left_origin[0];
        let left_is_computed = local.left_origin[2];
        let right_is_leaf = local.right_origin[0];
        let right_is_computed = local.right_origin[2];
        builder
            .when(local.active * left_is_leaf)
            .assert_one(local.left_leaf_multiplicity * local.left_leaf_multiplicity_inverse);
        builder
            .when(local.active * (AB::Expr::ONE - left_is_leaf))
            .assert_zero(local.left_leaf_multiplicity);
        builder
            .when(local.active * (AB::Expr::ONE - left_is_leaf))
            .assert_zero(local.left_leaf_multiplicity_inverse);
        builder
            .when(local.active * right_is_leaf)
            .assert_one(local.right_leaf_multiplicity * local.right_leaf_multiplicity_inverse);
        builder
            .when(local.active * (AB::Expr::ONE - right_is_leaf))
            .assert_zero(local.right_leaf_multiplicity);
        builder
            .when(local.active * (AB::Expr::ONE - right_is_leaf))
            .assert_zero(local.right_leaf_multiplicity_inverse);
        builder
            .when(local.active * (left_is_leaf + right_is_leaf))
            .assert_zero(local.level);

        self.leaf_bus.receive(
            builder,
            NativeOpeningLeafMessage {
                proof_idx: local.proof_idx.into(),
                tree_id: local.tree_id.into(),
                index: left_index.clone(),
                digest: local.left.map(Into::into),
            },
            local.active * left_is_leaf * local.left_leaf_multiplicity,
        );
        self.leaf_bus.receive(
            builder,
            NativeOpeningLeafMessage {
                proof_idx: local.proof_idx.into(),
                tree_id: local.tree_id.into(),
                index: right_index.clone(),
                digest: local.right.map(Into::into),
            },
            local.active * right_is_leaf * local.right_leaf_multiplicity,
        );
        self.node_bus.receive(
            builder,
            NativeMerkleNodeMessage {
                proof_idx: local.proof_idx.into(),
                tree_id: local.tree_id.into(),
                level: local.level.into(),
                index: left_index,
                digest: local.left.map(Into::into),
            },
            local.active * left_is_computed,
        );
        self.node_bus.receive(
            builder,
            NativeMerkleNodeMessage {
                proof_idx: local.proof_idx.into(),
                tree_id: local.tree_id.into(),
                level: local.level.into(),
                index: right_index,
                digest: local.right.map(Into::into),
            },
            local.active * right_is_computed,
        );

        self.compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: core::array::from_fn::<_, POSEIDON2_WIDTH, _>(|index| {
                    if index < DIGEST_SIZE {
                        local.left[index].into()
                    } else {
                        local.right[index - DIGEST_SIZE].into()
                    }
                }),
                output: local.output.map(Into::into),
            },
            local.active,
        );
        self.node_bus.send(
            builder,
            NativeMerkleNodeMessage {
                proof_idx: local.proof_idx.into(),
                tree_id: local.tree_id.into(),
                level: output_level,
                index: local.parent_index.into(),
                digest: local.output.map(Into::into),
            },
            local.active * (AB::Expr::ONE - local.is_root),
        );
        self.root_bus.send(
            builder,
            NativeMerkleRootMessage {
                proof_idx: local.proof_idx.into(),
                tree_id: local.tree_id.into(),
                depth: local.depth.into(),
                digest: local.output.map(Into::into),
            },
            local.active * local.is_root,
        );
    }
}

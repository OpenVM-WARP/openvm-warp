//! Authentication of block user public values against the final VM memory root.
//!
//! This is deliberately a separate wrapper-stage obligation. The fixed
//! verifier source and standard VACC authenticate execution and accumulation;
//! neither authenticates the user-public-values commitment. The composition
//! therefore uses the ordinary [`UserPvsCommitAir`] to commit the public
//! values and this AIR to verify that commitment's canonical memory path. Only
//! this AIR may publish [`VerifierWarpBlockPublicValuesMessageV2`].

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::POSEIDON2_WIDTH,
    system::memory::{dimensions::MemoryDimensions, merkle::public_values::PUBLIC_VALUES_AS},
};
use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper, SubAir};
use openvm_recursion_circuit::bus::{Poseidon2CompressBus, Poseidon2CompressMessage};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    p3_util::log2_strict_usize,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, Digest, DIGEST_SIZE, F,
};

use super::{
    VerifierWarpBlockPublicValuesBusV2, VerifierWarpBlockPublicValuesMessageV2,
    VERIFIER_WARP_HISTORY_PROTOCOL_V2,
};
use crate::circuit::subair::{
    MerklePathRowView, MerklePathSubAir, MerklePathSubAirContext, MerkleRootBus, MerkleRootMessage,
};

/// Domain separator for the digest exposed by History v2.
pub const VERIFIER_WARP_BLOCK_PUBLIC_VALUES_TAG_V2: u32 = 0x5657_5002;

/// Canonical digest later authenticated by the terminal block-PV AIR. Chunk
/// leaves may carry this value publicly; only the recursive root is allowed
/// to treat it as an authenticated block statement.
#[must_use]
pub fn verifier_warp_block_public_values_digest_v2(
    num_user_public_values: usize,
    public_values_commitment: Digest,
) -> Digest {
    let metadata = core::array::from_fn(|limb| match limb {
        0 => F::from_u32(VERIFIER_WARP_BLOCK_PUBLIC_VALUES_TAG_V2),
        1 => F::from_u32(VERIFIER_WARP_HISTORY_PROTOCOL_V2),
        2 => F::from_usize(num_user_public_values),
        _ => F::ZERO,
    });
    poseidon2_compress_with_capacity(metadata, public_values_commitment).0
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct VerifierWarpBlockPublicValuesColsV2<T> {
    // 0 for padding, 1 for an ordinary valid row, 2 for the first valid row.
    pub is_valid: T,
    pub is_right_child: T,
    pub node_commit: [T; DIGEST_SIZE],
    pub sibling: [T; DIGEST_SIZE],
    pub row_idx_exp_2: T,
    pub merkle_path_branch_bits: T,
    pub public_values_digest: [T; DIGEST_SIZE],
}

#[derive(Clone, ColumnsAir)]
#[columns_via(VerifierWarpBlockPublicValuesColsV2<u8>)]
pub struct VerifierWarpBlockPublicValuesAirV2 {
    pub merkle_path_subair: MerklePathSubAir,
    pub merkle_root_bus: MerkleRootBus,
    pub compress_bus: Poseidon2CompressBus,
    pub block_bus: VerifierWarpBlockPublicValuesBusV2,
    pub num_user_public_values: usize,
}

impl VerifierWarpBlockPublicValuesAirV2 {
    pub fn new(
        compress_bus: Poseidon2CompressBus,
        merkle_root_bus: MerkleRootBus,
        block_bus: VerifierWarpBlockPublicValuesBusV2,
        memory_dimensions: MemoryDimensions,
        num_user_public_values: usize,
    ) -> Result<Self, &'static str> {
        if num_user_public_values < DIGEST_SIZE
            || !num_user_public_values.is_multiple_of(DIGEST_SIZE)
            || !(num_user_public_values / DIGEST_SIZE).is_power_of_two()
            || memory_dimensions.addr_space_height <= 1
        {
            return Err("invalid verifier-WARP user-public-values shape");
        }
        let pv_start_idx = memory_dimensions.label_to_index((PUBLIC_VALUES_AS, 0));
        let pv_height = log2_strict_usize(num_user_public_values / DIGEST_SIZE);
        let branch_bits = u32::try_from(pv_start_idx >> pv_height)
            .map_err(|_| "verifier-WARP public-values branch exceeds u32")?;
        let expected_proof_len = memory_dimensions
            .overall_height()
            .checked_sub(pv_height)
            .ok_or("verifier-WARP public-values proof height")?;
        Ok(Self {
            merkle_path_subair: MerklePathSubAir::new(
                compress_bus,
                expected_proof_len,
                branch_bits,
            ),
            merkle_root_bus,
            compress_bus,
            block_bus,
            num_user_public_values,
        })
    }

    fn digest_metadata<Expr: PrimeCharacteristicRing>(&self) -> [Expr; DIGEST_SIZE] {
        core::array::from_fn(|limb| match limb {
            0 => Expr::from_u32(VERIFIER_WARP_BLOCK_PUBLIC_VALUES_TAG_V2),
            1 => Expr::from_u32(VERIFIER_WARP_HISTORY_PROTOCOL_V2),
            2 => Expr::from_usize(self.num_user_public_values),
            _ => Expr::ZERO,
        })
    }
}

impl BaseAir<F> for VerifierWarpBlockPublicValuesAirV2 {
    fn width(&self) -> usize {
        VerifierWarpBlockPublicValuesColsV2::<u8>::width()
    }
}

impl BaseAirWithPublicValues<F> for VerifierWarpBlockPublicValuesAirV2 {}
impl PartitionedBaseAir<F> for VerifierWarpBlockPublicValuesAirV2 {}

impl<AB> Air<AB> for VerifierWarpBlockPublicValuesAirV2
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("block public-values row");
        let next_row = main.row_slice(1).expect("block public-values next row");
        let local: &VerifierWarpBlockPublicValuesColsV2<AB::Var> = (*local_row).borrow();
        let next: &VerifierWarpBlockPublicValuesColsV2<AB::Var> = (*next_row).borrow();

        builder.assert_zero(
            local.is_valid * (local.is_valid - AB::Expr::ONE) * (local.is_valid - AB::Expr::TWO),
        );
        let is_first = local.is_valid * (local.is_valid - AB::Expr::ONE) * AB::F::TWO.inverse();
        let is_last = local.is_valid * (AB::Expr::ONE - next.is_valid).square();

        self.merkle_root_bus.receive(
            builder,
            MerkleRootMessage {
                merkle_root: local.node_commit.map(Into::into),
                idx: AB::Expr::ZERO,
                num_rows_or_zero: AB::Expr::ZERO,
            },
            is_first.clone(),
        );

        self.merkle_path_subair.eval(
            builder,
            (
                MerklePathSubAirContext {
                    local: MerklePathRowView {
                        is_valid: &local.is_valid,
                        is_right_child: &local.is_right_child,
                        node_commit: &local.node_commit,
                        sibling: &local.sibling,
                        row_idx_exp_2: &local.row_idx_exp_2,
                        merkle_path_branch_bits: &local.merkle_path_branch_bits,
                    },
                    next: MerklePathRowView {
                        is_valid: &next.is_valid,
                        is_right_child: &next.is_right_child,
                        node_commit: &next.node_commit,
                        sibling: &next.sibling,
                        row_idx_exp_2: &next.row_idx_exp_2,
                        merkle_path_branch_bits: &next.merkle_path_branch_bits,
                    },
                },
                AB::Expr::ZERO,
                AB::Expr::ZERO,
                AB::Expr::ZERO,
            ),
        );

        self.compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: core::array::from_fn(|index| {
                    if index < DIGEST_SIZE {
                        self.digest_metadata::<AB::Expr>()[index].clone()
                    } else {
                        local.node_commit[index - DIGEST_SIZE].into()
                    }
                }),
                output: local.public_values_digest.map(Into::into),
            },
            is_first,
        );
        for limb in 0..DIGEST_SIZE {
            builder.when_transition().when(next.is_valid).assert_eq(
                next.public_values_digest[limb],
                local.public_values_digest[limb],
            );
        }
        self.block_bus.add_key_with_lookups(
            builder,
            VerifierWarpBlockPublicValuesMessageV2 {
                final_memory_root: local.node_commit.map(Into::into),
                public_values_digest: local.public_values_digest.map(Into::into),
            },
            is_last,
        );
    }
}

pub struct VerifierWarpBlockPublicValuesTraceV2 {
    pub matrix: RowMajorMatrix<F>,
    pub compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub public_values_digest: Digest,
    pub final_memory_root: Digest,
}

pub fn generate_verifier_warp_block_public_values_trace_v2(
    air: &VerifierWarpBlockPublicValuesAirV2,
    public_values_commitment: Digest,
    merkle_proof: &[Digest],
) -> Result<VerifierWarpBlockPublicValuesTraceV2, &'static str> {
    if merkle_proof.len() != air.merkle_path_subair.expected_proof_len {
        return Err("verifier-WARP public-values Merkle path length");
    }
    let width = VerifierWarpBlockPublicValuesColsV2::<u8>::width();
    let height = (merkle_proof.len() + 1).next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    let metadata = air.digest_metadata::<F>();
    let public_values_digest = verifier_warp_block_public_values_digest_v2(
        air.num_user_public_values,
        public_values_commitment,
    );
    let mut compression_inputs = vec![core::array::from_fn(|index| {
        if index < DIGEST_SIZE {
            metadata[index]
        } else {
            public_values_commitment[index - DIGEST_SIZE]
        }
    })];

    let pv_height = log2_strict_usize(air.num_user_public_values / DIGEST_SIZE);
    let pv_start_idx = air.merkle_path_subair.expected_branch_bits << pv_height;
    let merkle_path_branch_bits = pv_start_idx >> pv_height;
    let mut accumulated_branch_bits = 0usize;
    let mut current = public_values_commitment;
    for (row_index, &sibling) in merkle_proof.iter().enumerate() {
        let row: &mut VerifierWarpBlockPublicValuesColsV2<F> =
            values[row_index * width..(row_index + 1) * width].borrow_mut();
        let right = merkle_path_branch_bits & (1 << row_index) != 0;
        accumulated_branch_bits += usize::from(right) << row_index;
        row.is_valid = if row_index == 0 { F::TWO } else { F::ONE };
        row.is_right_child = F::from_bool(right);
        row.node_commit = current;
        row.sibling = sibling;
        row.row_idx_exp_2 = F::from_usize(1 << row_index);
        row.merkle_path_branch_bits = F::from_usize(accumulated_branch_bits);
        row.public_values_digest = public_values_digest;
        let (left, right_node) = if right {
            (sibling, current)
        } else {
            (current, sibling)
        };
        compression_inputs.push(core::array::from_fn(|index| {
            if index < DIGEST_SIZE {
                left[index]
            } else {
                right_node[index - DIGEST_SIZE]
            }
        }));
        current = poseidon2_compress_with_capacity(left, right_node).0;
    }
    let last_index = merkle_proof.len();
    let last: &mut VerifierWarpBlockPublicValuesColsV2<F> =
        values[last_index * width..(last_index + 1) * width].borrow_mut();
    last.is_valid = F::ONE;
    last.node_commit = current;
    last.row_idx_exp_2 = F::from_usize(1 << last_index);
    last.merkle_path_branch_bits = F::from_usize(accumulated_branch_bits);
    last.public_values_digest = public_values_digest;

    Ok(VerifierWarpBlockPublicValuesTraceV2 {
        matrix: RowMajorMatrix::new(values, width),
        compression_inputs,
        public_values_digest,
        final_memory_root: current,
    })
}

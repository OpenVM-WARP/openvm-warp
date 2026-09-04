use core::borrow::BorrowMut;

use openvm_stark_backend::warp_accum::{BinaryMerkleMultiproofRecord, MerkleNodeOrigin};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, F};
use p3_field::{Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

use super::{NativeMerkleCompressionCols, NativeMerkleLeafAdapterCols};

#[derive(Clone, Debug)]
pub struct NativeMerkleLeafAdapterInput {
    pub proof_idx: u32,
    pub bypass: bool,
    pub outer_multiplicity: u32,
    pub inner_tree_id: u32,
    pub outer_tree_id: u32,
    pub query_index: u32,
    pub inner_depth: u32,
    pub digest: Digest,
}

pub fn generate_native_merkle_multiproof_trace(
    proof_idx: usize,
    records: &[(u32, &BinaryMerkleMultiproofRecord<Digest>)],
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    let valid_rows = records
        .iter()
        .map(|(_, record)| record.compressions.len())
        .sum::<usize>();
    if valid_rows == 0 {
        return None;
    }
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeMerkleCompressionCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    let mut row_index = 0usize;
    for &(tree_id, record) in records {
        for compression in &record.compressions {
            let row = &mut trace[row_index * width..(row_index + 1) * width];
            let cols: &mut NativeMerkleCompressionCols<F> = row.borrow_mut();
            cols.active = F::ONE;
            cols.proof_idx = F::from_usize(proof_idx);
            cols.tree_id = F::from_u32(tree_id);
            cols.level = F::from_u32(compression.level);
            cols.depth = F::from_u32(record.depth);
            cols.parent_index = F::from_u32(compression.parent_index);
            let is_root = compression.level + 1 == record.depth;
            cols.is_root = F::from_bool(is_root);
            let distance = record.depth - (compression.level + 1);
            cols.root_inverse = if distance == 0 {
                F::ZERO
            } else {
                F::from_u32(distance).inverse()
            };
            cols.left_origin = origin_flags(&compression.left_origin);
            cols.right_origin = origin_flags(&compression.right_origin);
            let left_multiplicity = opened_leaf_multiplicity(&compression.left_origin);
            let right_multiplicity = opened_leaf_multiplicity(&compression.right_origin);
            cols.left_leaf_multiplicity = F::from_u32(left_multiplicity);
            cols.left_leaf_multiplicity_inverse = multiplicity_inverse(left_multiplicity);
            cols.right_leaf_multiplicity = F::from_u32(right_multiplicity);
            cols.right_leaf_multiplicity_inverse = multiplicity_inverse(right_multiplicity);
            cols.left = compression.left;
            cols.right = compression.right;
            cols.output = compression.output;
            row_index += 1;
        }
    }
    Some(RowMajorMatrix::new(trace, width))
}

fn origin_flags(origin: &MerkleNodeOrigin) -> [F; 3] {
    match origin {
        MerkleNodeOrigin::OpenedLeaf { .. } => [F::ONE, F::ZERO, F::ZERO],
        MerkleNodeOrigin::ProofSibling { .. } => [F::ZERO, F::ONE, F::ZERO],
        MerkleNodeOrigin::Computed { .. } => [F::ZERO, F::ZERO, F::ONE],
    }
}

fn opened_leaf_multiplicity(origin: &MerkleNodeOrigin) -> u32 {
    match origin {
        MerkleNodeOrigin::OpenedLeaf { multiplicity, .. } => *multiplicity,
        MerkleNodeOrigin::ProofSibling { .. } | MerkleNodeOrigin::Computed { .. } => 0,
    }
}

fn multiplicity_inverse(multiplicity: u32) -> F {
    if multiplicity == 0 {
        F::ZERO
    } else {
        F::from_u32(multiplicity).inverse()
    }
}

/// `inner_depth` is the AIR's keygen constant. Every input must agree with it: the
/// AIR emits only the matching branch, so an input claiming a different subtree shape
/// would be silently reinterpreted rather than rejected.
pub fn generate_native_merkle_leaf_adapter_trace(
    inputs: &[NativeMerkleLeafAdapterInput],
    inner_depth: usize,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if inputs.is_empty()
        || inputs.iter().any(|input| {
            input.inner_depth as usize != inner_depth || input.bypass != (inner_depth == 0)
        })
    {
        return None;
    }
    let height = required_height.unwrap_or_else(|| inputs.len().next_power_of_two());
    if height < inputs.len() {
        return None;
    }
    let width = NativeMerkleLeafAdapterCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    for (index, input) in inputs.iter().enumerate() {
        let cols: &mut NativeMerkleLeafAdapterCols<F> =
            trace[index * width..(index + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.proof_idx = F::from_u32(input.proof_idx);
        cols.outer_multiplicity = F::from_u32(input.outer_multiplicity);
        cols.outer_multiplicity_inverse = if input.outer_multiplicity == 0 {
            return None;
        } else {
            cols.outer_multiplicity.inverse()
        };
        cols.inner_tree_id = F::from_u32(input.inner_tree_id);
        cols.outer_tree_id = F::from_u32(input.outer_tree_id);
        cols.query_index = F::from_u32(input.query_index);
        cols.digest = input.digest;
    }
    Some(RowMajorMatrix::new(trace, width))
}

#[cfg(test)]
mod tests {
    use core::borrow::Borrow;

    use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

    use super::*;

    #[test]
    fn leaf_adapter_preserves_duplicate_outer_query_multiplicity() {
        let multiplicity = 3;
        let trace = generate_native_merkle_leaf_adapter_trace(
            &[NativeMerkleLeafAdapterInput {
                proof_idx: 0,
                bypass: false,
                outer_multiplicity: multiplicity,
                inner_tree_id: 7,
                outer_tree_id: 11,
                query_index: 13,
                inner_depth: 4,
                digest: [F::from_u32(17); DIGEST_SIZE],
            }],
            4,
            None,
        )
        .unwrap();
        let cols: &NativeMerkleLeafAdapterCols<F> = trace.values.as_slice().borrow();
        assert_eq!(cols.outer_multiplicity, F::from_u32(multiplicity));
        assert_eq!(
            cols.outer_multiplicity * cols.outer_multiplicity_inverse,
            F::ONE
        );
    }

    #[test]
    fn leaf_adapter_rejects_zero_outer_multiplicity() {
        assert!(generate_native_merkle_leaf_adapter_trace(
            &[NativeMerkleLeafAdapterInput {
                proof_idx: 0,
                bypass: true,
                outer_multiplicity: 0,
                inner_tree_id: 1,
                outer_tree_id: 2,
                query_index: 3,
                inner_depth: 0,
                digest: [F::ZERO; DIGEST_SIZE],
            }],
            0,
            None,
        )
        .is_none());
    }
}

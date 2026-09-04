use core::borrow::BorrowMut;

use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_sdk::config::baby_bear_poseidon2::{poseidon2_perm, CHUNK, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::*;
use p3_symmetric::Permutation;

use super::NativeLeafHashCols;

#[derive(Clone, Debug)]
pub struct NativeLeafHashInput<'a> {
    pub proof_idx: u32,
    pub tree_id: u32,
    pub leaf_index: u32,
    pub values: &'a [F],
    pub lookup_counts: &'a [u32],
}

pub struct NativeLeafHashTrace {
    pub matrix: RowMajorMatrix<F>,
    pub permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
}

pub fn generate_native_leaf_hash_trace(
    leaves: &[NativeLeafHashInput<'_>],
    required_height: Option<usize>,
) -> Option<NativeLeafHashTrace> {
    if leaves.is_empty()
        || leaves
            .iter()
            .any(|leaf| leaf.values.is_empty() || leaf.lookup_counts.len() != leaf.values.len())
    {
        return None;
    }
    let valid_rows = leaves
        .iter()
        .map(|leaf| leaf.values.len().div_ceil(CHUNK))
        .sum::<usize>();
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeLeafHashCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];

    // Each leaf is an independent sponge: only the blocks *within* a leaf chain
    // through `state`. Hand every leaf its own row range up front so the leaves
    // run in parallel, which matters because this is where native WARP's
    // history circuit spends most of its witness generation -- 354k
    // permutations per root step on a six-segment program, and they were
    // serial.
    //
    // Row order, and therefore `permutation_inputs` order, is the same as a
    // sequential pass: the ranges are assigned in leaf order and each leaf
    // fills its own range front to back.
    let mut leaf_slices: Vec<(&NativeLeafHashInput<'_>, usize, &mut [F])> =
        Vec::with_capacity(leaves.len());
    let mut remaining = trace.as_mut_slice();
    for leaf in leaves {
        let blocks = leaf.values.len().div_ceil(CHUNK);
        let (owned, rest) = remaining.split_at_mut(blocks * width);
        leaf_slices.push((leaf, blocks, owned));
        remaining = rest;
    }

    let per_leaf_inputs: Vec<Vec<[F; POSEIDON2_WIDTH]>> = leaf_slices
        .into_par_iter()
        .map(|(leaf, blocks, rows)| {
            let mut inputs = Vec::with_capacity(blocks);
            let mut state = [F::ZERO; POSEIDON2_WIDTH];
            for (block, row) in rows.chunks_exact_mut(width).enumerate() {
                let before = state;
                let values =
                    &leaf.values[block * CHUNK..(leaf.values.len()).min((block + 1) * CHUNK)];
                for (target, &value) in state.iter_mut().zip(values) {
                    *target = value;
                }
                let input = state;
                poseidon2_perm().permute_mut(&mut state);
                let cols: &mut NativeLeafHashCols<F> = row.borrow_mut();
                cols.active = F::ONE;
                cols.proof_idx = F::from_u32(leaf.proof_idx);
                cols.tree_id = F::from_u32(leaf.tree_id);
                cols.leaf_index = F::from_u32(leaf.leaf_index);
                cols.block = F::from_usize(block);
                cols.is_first = F::from_bool(block == 0);
                cols.is_last = F::from_bool(block + 1 == blocks);
                for mask in cols.mask.iter_mut().take(values.len()) {
                    *mask = F::ONE;
                }
                for (target, &count) in cols
                    .lookup_count
                    .iter_mut()
                    .zip(&leaf.lookup_counts[block * CHUNK..block * CHUNK + values.len()])
                {
                    *target = F::from_u32(count);
                }
                cols.before = before;
                cols.input = input;
                cols.output = state;
                inputs.push(input);
            }
            inputs
        })
        .collect();

    let mut permutation_inputs = Vec::with_capacity(valid_rows);
    for inputs in per_leaf_inputs {
        permutation_inputs.extend(inputs);
    }
    Some(NativeLeafHashTrace {
        matrix: RowMajorMatrix::new(trace, width),
        permutation_inputs,
    })
}

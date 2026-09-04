use core::borrow::BorrowMut;

use openvm_stark_backend::warp_pesat::eval_eq_points;
use openvm_stark_sdk::config::baby_bear_poseidon2::{EF, F};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

use super::NativeEqEvaluationCols;

pub struct NativeEqTraceInput<'a> {
    pub left_vector: u32,
    pub right_vector: u32,
    pub left: &'a [EF],
    pub right: &'a [EF],
    pub lookup_count: u32,
}

pub fn generate_native_eq_trace(
    proof_idx: usize,
    pairs: &[NativeEqTraceInput<'_>],
    group_offset: usize,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    let dimensions = pairs.first()?.left.len();
    if dimensions == 0
        || pairs
            .iter()
            .any(|pair| pair.left.len() != dimensions || pair.right.len() != dimensions)
    {
        return None;
    }
    let valid_rows = pairs.len() * dimensions;
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeEqEvaluationCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    for (group, pair) in pairs.iter().enumerate() {
        let (left, right) = (pair.left, pair.right);
        let mut accumulator = EF::ONE;
        for coordinate in 0..dimensions {
            let before = accumulator;
            accumulator *= openvm_stark_backend::warp_pesat::eval_eq_points(
                &left[coordinate..=coordinate],
                &right[coordinate..=coordinate],
            );
            let row_index = group * dimensions + coordinate;
            let row = &mut trace[row_index * width..(row_index + 1) * width];
            let cols: &mut NativeEqEvaluationCols<F> = row.borrow_mut();
            cols.active = F::ONE;
            cols.proof_idx = F::from_usize(proof_idx);
            cols.group = F::from_usize(group_offset + group);
            cols.coordinate = F::from_usize(coordinate);
            cols.is_first = F::from_bool(coordinate == 0);
            cols.is_last = F::from_bool(coordinate + 1 == dimensions);
            cols.is_first_group = F::from_bool(group == 0 && coordinate == 0);
            cols.continues_group = F::from_bool(coordinate != 0);
            cols.starts_first_group = F::from_bool(group == 0 && coordinate == 0);
            cols.starts_nonfirst_group = F::from_bool(group != 0 && coordinate == 0);
            cols.first_inverse = if coordinate == 0 {
                F::ZERO
            } else {
                F::from_usize(coordinate).inverse()
            };
            let distance = dimensions - 1 - coordinate;
            cols.last_inverse = if distance == 0 {
                F::ZERO
            } else {
                F::from_usize(distance).inverse()
            };
            let first_group_distance = group * dimensions + coordinate;
            cols.first_group_inverse = if first_group_distance == 0 {
                F::ZERO
            } else {
                F::from_usize(first_group_distance).inverse()
            };
            cols.left_vector = F::from_u32(pair.left_vector);
            cols.right_vector = F::from_u32(pair.right_vector);
            cols.left
                .copy_from_slice(left[coordinate].as_basis_coefficients_slice());
            cols.right
                .copy_from_slice(right[coordinate].as_basis_coefficients_slice());
            cols.accumulator_before
                .copy_from_slice(before.as_basis_coefficients_slice());
            cols.accumulator_after
                .copy_from_slice(accumulator.as_basis_coefficients_slice());
            if coordinate + 1 == dimensions {
                cols.lookup_count = F::from_u32(pair.lookup_count);
            }
        }
        debug_assert_eq!(accumulator, eval_eq_points(left, right));
    }
    Some(RowMajorMatrix::new(trace, width))
}

use std::sync::atomic::{AtomicU32, Ordering};

use itertools::Itertools;
use openvm_stark_sdk::config::baby_bear_poseidon2::F;
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::primitives::range::air::RangeCheckerCols;

#[derive(Debug)]
pub struct RangeCheckerCpuTraceGenerator<const NUM_BITS: usize> {
    count: Vec<AtomicU32>,
}

impl<const NUM_BITS: usize> Default for RangeCheckerCpuTraceGenerator<NUM_BITS> {
    fn default() -> Self {
        let mut count = Vec::with_capacity(1 << NUM_BITS);
        for _ in 0..(1 << NUM_BITS) {
            count.push(AtomicU32::new(0));
        }
        Self { count }
    }
}

impl<const NUM_BITS: usize> RangeCheckerCpuTraceGenerator<NUM_BITS> {
    /// Merge a separately generated request set into this table.
    ///
    /// This is useful when independent circuit shards generate witnesses in
    /// parallel but share one range-table AIR. Counts are the table's complete
    /// semantic state, so component-wise addition is equivalent to registering
    /// the same requests serially.
    pub fn merge_from(&self, other: Self) {
        for (count, other_count) in self.count.iter().zip(other.count) {
            count.fetch_add(other_count.load(Ordering::Relaxed), Ordering::Relaxed);
        }
    }

    pub fn add_count(&self, value: usize) {
        self.add_count_mult(value, 1);
    }

    pub fn add_count_mult(&self, value: usize, mult: u32) {
        self.count[value].fetch_add(mult, Ordering::Relaxed);
    }

    #[tracing::instrument(name = "generate_trace", level = "trace", skip_all)]
    pub fn generate_trace_row_major(&self) -> RowMajorMatrix<F> {
        let trace = self
            .count
            .iter()
            .enumerate()
            .flat_map(|(value, mult)| {
                [
                    F::from_usize(value),
                    F::from_u32(mult.load(Ordering::Relaxed)),
                ]
            })
            .collect_vec();
        RowMajorMatrix::new(trace, RangeCheckerCols::<u8>::width())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merged_counts_match_sequential_registration() {
        let sequential = RangeCheckerCpuTraceGenerator::<8>::default();
        sequential.add_count_mult(3, 2);
        sequential.add_count(7);
        sequential.add_count_mult(3, 4);

        let merged = RangeCheckerCpuTraceGenerator::<8>::default();
        merged.add_count_mult(3, 2);
        merged.add_count(7);
        let suffix = RangeCheckerCpuTraceGenerator::<8>::default();
        suffix.add_count_mult(3, 4);
        merged.merge_from(suffix);

        assert_eq!(
            merged.generate_trace_row_major().values,
            sequential.generate_trace_row_major().values
        );
    }
}

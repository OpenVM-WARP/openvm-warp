use std::borrow::BorrowMut;

use openvm_cpu_backend::CpuBackend;
use openvm_stark_backend::prover::AirProvingContext;
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, F};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::circuit::inner::unset::UnsetPvsCols;

pub fn generate_proving_ctx(
    unset_proof_idxs: &[usize],
    child_is_app: bool,
    required_height: Option<usize>,
) -> AirProvingContext<CpuBackend<BabyBearPoseidon2Config>> {
    let num_valid = if child_is_app {
        0
    } else {
        unset_proof_idxs.len()
    };

    let natural_height = num_valid.next_power_of_two();
    let height = required_height.unwrap_or(natural_height);
    assert!(
        height.is_power_of_two() && height >= natural_height,
        "fixed unset-PVS height must be a power of two at least the natural height"
    );
    let width = UnsetPvsCols::<u8>::width();
    let mut trace = vec![F::ZERO; height * width];
    let mut chunks = trace.chunks_exact_mut(width);

    // `UnsetPvsAir` constrains `proof_idx` to increase on every physical row,
    // including inactive padding rows.  Populate that fixed counter first;
    // `is_valid` still gates all bus traffic from the padded suffix.
    let first_proof_idx = unset_proof_idxs.first().copied().unwrap_or(0);
    for (row_idx, chunk) in chunks.by_ref().enumerate() {
        let cols: &mut UnsetPvsCols<F> = chunk.borrow_mut();
        cols.proof_idx = F::from_usize(first_proof_idx + row_idx);
    }

    for (row_idx, proof_idx) in unset_proof_idxs.iter().take(num_valid).enumerate() {
        let chunk = &mut trace[row_idx * width..(row_idx + 1) * width];
        let cols: &mut UnsetPvsCols<F> = chunk.borrow_mut();
        cols.is_valid = F::ONE;
        cols.proof_idx = F::from_usize(*proof_idx);
    }

    AirProvingContext::simple_no_pis(RowMajorMatrix::new(trace, width))
}

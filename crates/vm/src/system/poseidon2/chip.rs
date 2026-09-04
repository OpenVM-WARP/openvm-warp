use std::{
    array,
    sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering},
};

use dashmap::DashMap;
use openvm_poseidon2_air::{Poseidon2Config, Poseidon2SubChip};
use rustc_hash::FxBuildHasher;

use super::{PERIPHERY_POSEIDON2_CHUNK_SIZE, PERIPHERY_POSEIDON2_WIDTH};
use crate::arch::{
    hasher::{Hasher, HasherChip},
    VmField,
};

#[derive(Debug)]
pub struct Poseidon2PeripheryBaseChip<F: VmField, const SBOX_REGISTERS: usize> {
    pub subchip: Poseidon2SubChip<F, SBOX_REGISTERS>,
    pub records: DashMap<[F; PERIPHERY_POSEIDON2_WIDTH], AtomicU32, FxBuildHasher>,
    pub nonempty: AtomicBool,
    /// Optional exact trace height for the next proving context. Native-WARP
    /// plans from metered execution are conservative because duplicate hash
    /// inputs are coalesced here; pinning preserves the setup-owned shard
    /// shape while retaining zero-multiplicity padding rows.
    pub forced_height: AtomicUsize,
}

impl<F: VmField, const SBOX_REGISTERS: usize> Poseidon2PeripheryBaseChip<F, SBOX_REGISTERS> {
    pub fn new(poseidon2_config: Poseidon2Config<F>) -> Self {
        let subchip = Poseidon2SubChip::new(poseidon2_config.constants);
        Self {
            subchip,
            records: DashMap::default(),
            nonempty: AtomicBool::new(false),
            forced_height: AtomicUsize::new(0),
        }
    }

    pub fn set_forced_height(&self, height: usize) {
        assert!(
            height == 0 || height.is_power_of_two(),
            "forced Poseidon2 trace height {height} must be zero or a power of two"
        );
        self.forced_height.store(height, Ordering::Release);
    }

    pub(crate) fn take_forced_height(&self) -> usize {
        self.forced_height.swap(0, Ordering::AcqRel)
    }
}

impl<F: VmField, const SBOX_REGISTERS: usize> Hasher<PERIPHERY_POSEIDON2_CHUNK_SIZE, F>
    for Poseidon2PeripheryBaseChip<F, SBOX_REGISTERS>
{
    fn compress(
        &self,
        lhs: &[F; PERIPHERY_POSEIDON2_CHUNK_SIZE],
        rhs: &[F; PERIPHERY_POSEIDON2_CHUNK_SIZE],
    ) -> [F; PERIPHERY_POSEIDON2_CHUNK_SIZE] {
        let mut input_state = [F::ZERO; PERIPHERY_POSEIDON2_WIDTH];
        input_state[..PERIPHERY_POSEIDON2_CHUNK_SIZE].copy_from_slice(lhs);
        input_state[PERIPHERY_POSEIDON2_CHUNK_SIZE..].copy_from_slice(rhs);

        let output = self.subchip.permute(input_state);
        array::from_fn(|i| output[i])
    }
}

impl<F: VmField, const SBOX_REGISTERS: usize> HasherChip<PERIPHERY_POSEIDON2_CHUNK_SIZE, F>
    for Poseidon2PeripheryBaseChip<F, SBOX_REGISTERS>
{
    /// Key method for Hasher trait.
    ///
    /// Takes two chunks, hashes them, and returns the result. Total width 3 * DIGEST_WIDTH, exposed
    /// in `direct_interaction_width()`.
    ///
    /// No interactions with other chips.
    fn compress_and_record(
        &self,
        lhs: &[F; PERIPHERY_POSEIDON2_CHUNK_SIZE],
        rhs: &[F; PERIPHERY_POSEIDON2_CHUNK_SIZE],
    ) -> [F; PERIPHERY_POSEIDON2_CHUNK_SIZE] {
        let mut input = [F::ZERO; PERIPHERY_POSEIDON2_WIDTH];
        input[..PERIPHERY_POSEIDON2_CHUNK_SIZE].copy_from_slice(lhs);
        input[PERIPHERY_POSEIDON2_CHUNK_SIZE..].copy_from_slice(rhs);

        let count = self.records.entry(input).or_insert(AtomicU32::new(0));
        count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.nonempty
            .store(true, std::sync::atomic::Ordering::Relaxed);

        let output = self.subchip.permute(input);
        array::from_fn(|i| output[i])
    }
}

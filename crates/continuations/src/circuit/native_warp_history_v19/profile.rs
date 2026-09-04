use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, F};

use super::{compute_history_profile_digest_v19, DigestV19, HistoryV19Error};

pub const NATIVE_WARP_HISTORY_PROTOCOL_V19: u32 = 19;

/// Verifier-key-fixed limits and authenticated shard catalog.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryVerifierProfileV19 {
    pub app_vk_digest: DigestV19,
    pub shard_catalog_root: DigestV19,
    pub shard_catalog_len: u16,
    pub product_tree_height: u8,
    pub max_segments_per_chunk: u16,
    pub max_active_shards_per_segment: u16,
    pub profile_digest: DigestV19,
    /// Fixed padded trace height.  It is derived from the verifier limits, not
    /// from the prover's record.
    pub trace_height: usize,
}

impl HistoryVerifierProfileV19 {
    pub fn new(
        app_vk_digest: DigestV19,
        shard_catalog_root: DigestV19,
        shard_catalog_len: u16,
        product_tree_height: u8,
        max_segments_per_chunk: u16,
        max_active_shards_per_segment: u16,
    ) -> Result<Self, HistoryV19Error> {
        if shard_catalog_len == 0 {
            return Err(HistoryV19Error::InvalidProfile("empty shard catalog"));
        }
        if product_tree_height == 0 || product_tree_height > 16 {
            return Err(HistoryV19Error::InvalidProfile(
                "product tree height must be in 1..=16",
            ));
        }
        if usize::from(shard_catalog_len) > (1usize << product_tree_height) {
            return Err(HistoryV19Error::InvalidProfile(
                "catalog does not fit product tree",
            ));
        }
        if max_segments_per_chunk == 0 || max_active_shards_per_segment == 0 {
            return Err(HistoryV19Error::InvalidProfile(
                "history bounds must be nonzero",
            ));
        }
        if max_active_shards_per_segment > shard_catalog_len {
            return Err(HistoryV19Error::InvalidProfile(
                "active shard bound exceeds catalog",
            ));
        }
        let rows_per_shard = 1usize + usize::from(product_tree_height);
        let maximum_rows = usize::from(max_segments_per_chunk)
            .checked_mul(
                3usize
                    .checked_add(
                        usize::from(max_active_shards_per_segment)
                            .checked_mul(rows_per_shard)
                            .ok_or(HistoryV19Error::IntegerOverflow)?,
                    )
                    .ok_or(HistoryV19Error::IntegerOverflow)?,
            )
            .and_then(|rows| rows.checked_add(1))
            .ok_or(HistoryV19Error::IntegerOverflow)?;
        // Reserve one verifier-fixed inactive row after the terminal event.
        // The AIR constrains the physical last row inactive, so an adversary
        // cannot omit `is_final` by keeping every row active.
        let trace_height = maximum_rows
            .checked_add(1)
            .ok_or(HistoryV19Error::IntegerOverflow)?
            .checked_next_power_of_two()
            .ok_or(HistoryV19Error::IntegerOverflow)?;
        let profile_digest = compute_history_profile_digest_v19(
            app_vk_digest,
            shard_catalog_root,
            shard_catalog_len,
            product_tree_height,
            max_segments_per_chunk,
            max_active_shards_per_segment,
            None,
        );
        Ok(Self {
            app_vk_digest,
            shard_catalog_root,
            shard_catalog_len,
            product_tree_height,
            max_segments_per_chunk,
            max_active_shards_per_segment,
            profile_digest,
            trace_height,
        })
    }

    #[must_use]
    pub fn maximum_active_rows(&self) -> usize {
        usize::from(self.max_segments_per_chunk)
            * (3 + usize::from(self.max_active_shards_per_segment)
                * (1 + usize::from(self.product_tree_height)))
            + 1
    }
}

const _: () = assert!(DIGEST_SIZE == 8);

#[allow(dead_code)]
fn _field_anchor() -> F {
    F::ZERO
}

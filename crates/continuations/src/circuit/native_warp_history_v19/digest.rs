use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, DIGEST_SIZE, D_EF, F,
};

use super::{
    DigestV19, ExtensionV19, HistoryBoundaryV19, SegmentManifestRecordV19, ShardKeyRecordV19,
    WarpCertifiedReplayRecordV19, NATIVE_WARP_HISTORY_PROTOCOL_V19,
};

pub(crate) const TAG_PROFILE_V19: u32 = 0x19_01;
pub(crate) const TAG_KEY_META_V19: u32 = 0x19_02;
pub(crate) const TAG_REPLAY_META_V19: u32 = 0x19_03;
pub(crate) const TAG_SEGMENT_HEADER_V19: u32 = 0x19_04;
pub(crate) const TAG_LOGUP_ENDPOINT_V19: u32 = 0x19_05;
pub(crate) const TAG_LOGUP_META_V19: u32 = 0x19_08;
pub(crate) const TAG_SEGMENT_END_V19: u32 = 0x19_09;
pub(crate) const TAG_TERMINAL_V19: u32 = 0x19_0a;

/// Poseidon2 inputs emitted by the slow CPU oracle.  Integration feeds these
/// directly to the standard recursion Poseidon2 compression chip.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DigestCollectorV19 {
    pub compression_inputs: Vec<[F; 2 * DIGEST_SIZE]>,
}

impl DigestCollectorV19 {
    #[inline]
    pub fn compress(&mut self, left: DigestV19, right: DigestV19) -> DigestV19 {
        self.compression_inputs.push(join_digests(left, right));
        poseidon2_compress_with_capacity(left, right).0
    }
}

fn collector_or_local<'a>(
    collector: Option<&'a mut DigestCollectorV19>,
    local: &'a mut DigestCollectorV19,
) -> &'a mut DigestCollectorV19 {
    collector.unwrap_or(local)
}

pub fn compute_history_profile_digest_v19(
    app_vk_digest: DigestV19,
    shard_catalog_root: DigestV19,
    shard_catalog_len: u16,
    product_tree_height: u8,
    max_segments_per_chunk: u16,
    max_active_shards_per_segment: u16,
    collector: Option<&mut DigestCollectorV19>,
) -> DigestV19 {
    let mut local = DigestCollectorV19::default();
    let collector = collector_or_local(collector, &mut local);
    let metadata = scalar_digest(&[
        TAG_PROFILE_V19,
        NATIVE_WARP_HISTORY_PROTOCOL_V19,
        u32::from(shard_catalog_len),
        u32::from(product_tree_height),
        u32::from(max_segments_per_chunk),
        u32::from(max_active_shards_per_segment),
    ]);
    let profile_and_vk = collector.compress(metadata, app_vk_digest);
    collector.compress(profile_and_vk, shard_catalog_root)
}

pub fn compute_shard_key_digest_v19(
    key: &ShardKeyRecordV19,
    collector: Option<&mut DigestCollectorV19>,
) -> DigestV19 {
    let mut local = DigestCollectorV19::default();
    let collector = collector_or_local(collector, &mut local);
    let (air_lo, air_hi) = split_u32(key.air_id);
    let metadata = scalar_digest(&[
        TAG_KEY_META_V19,
        NATIVE_WARP_HISTORY_PROTOCOL_V19,
        u32::from(key.ordinal),
        u32::from(air_lo),
        u32::from(air_hi),
        u32::from(key.log_height),
    ]);
    let key_meta = collector.compress(metadata, key.app_vk_digest);
    let layout = collector.compress(key.trace_layout_digest, key.public_schema_digest);
    let schemas = collector.compress(key.interaction_schema_digest, key.code_class_digest);
    let relation = collector.compress(key_meta, key.relation_digest);
    let structure = collector.compress(layout, schemas);
    collector.compress(relation, structure)
}

pub fn compute_replay_binding_digest_v19(
    segment_index: u32,
    key_digest: DigestV19,
    relation_digest: DigestV19,
    replay: &WarpCertifiedReplayRecordV19,
    collector: Option<&mut DigestCollectorV19>,
) -> DigestV19 {
    let mut local = DigestCollectorV19::default();
    let collector = collector_or_local(collector, &mut local);
    let (segment_lo, segment_hi) = split_u32(segment_index);
    let replay_meta = collector.compress(
        scalar_digest(&[
            TAG_REPLAY_META_V19,
            NATIVE_WARP_HISTORY_PROTOCOL_V19,
            u32::from(segment_lo),
            u32::from(segment_hi),
        ]),
        relation_digest,
    );
    let accumulators = collector.compress(
        replay.previous_accumulator_digest,
        replay.next_accumulator_digest,
    );
    let checkpoints = collector.compress(
        replay.previous_checkpoint_digest,
        replay.next_checkpoint_digest,
    );
    let source_and_endpoint =
        collector.compress(replay.opening_claim_digest, replay.replay_endpoint_digest);
    let meta_and_accumulators = collector.compress(replay_meta, accumulators);
    let checkpoints_and_endpoint = collector.compress(checkpoints, source_and_endpoint);
    let replay_digest = collector.compress(meta_and_accumulators, checkpoints_and_endpoint);
    collector.compress(key_digest, replay_digest)
}

#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn compute_product_leaf_digest_v19(
    key_digest: DigestV19,
    accumulator_digest: DigestV19,
    checkpoint_digest: DigestV19,
    collector: &mut DigestCollectorV19,
) -> DigestV19 {
    let payload = collector.compress(accumulator_digest, checkpoint_digest);
    collector.compress(key_digest, payload)
}

pub(crate) fn compute_endpoint_digest_v19(
    tag: u32,
    endpoint: ExtensionV19,
    collector: &mut DigestCollectorV19,
) -> DigestV19 {
    let mut left = [F::ZERO; DIGEST_SIZE];
    left[0] = F::from_u32(tag);
    left[1..1 + D_EF].copy_from_slice(&endpoint);
    collector.compress(left, [F::ZERO; DIGEST_SIZE])
}

/// Slow, allocation-light oracle for the canonical manifest hash.  It does
/// not validate paths or replay certificates; witness generation performs
/// those checks before accepting this digest.
pub fn compute_manifest_digest_v19(
    segment: &SegmentManifestRecordV19,
    collector: Option<&mut DigestCollectorV19>,
) -> DigestV19 {
    let mut local = DigestCollectorV19::default();
    let collector = collector_or_local(collector, &mut local);
    let (segment_lo, segment_hi) = split_u32(segment.segment_index);
    let header_meta = collector.compress(
        scalar_digest(&[
            TAG_SEGMENT_HEADER_V19,
            NATIVE_WARP_HISTORY_PROTOCOL_V19,
            u32::from(segment_lo),
            u32::from(segment_hi),
            segment.active_shards.len() as u32,
            u32::from(segment.terminates),
        ]),
        [F::ZERO; DIGEST_SIZE],
    );
    let boundary = collector.compress(segment.from_vm_state, segment.to_vm_state);
    let roots = collector.compress(
        segment.previous_product_state_root,
        segment.next_product_state_root,
    );
    let openings = collector.compress(
        segment.logup.segment_openings_digest,
        [F::ZERO; DIGEST_SIZE],
    );
    let source_and_public =
        collector.compress(segment.source_forest_root, segment.public_values_digest);
    let public_and_openings = collector.compress(source_and_public, openings);
    let header_left = collector.compress(header_meta, boundary);
    let header_right = collector.compress(roots, public_and_openings);
    let mut running = collector.compress(header_left, header_right);

    for shard in &segment.active_shards {
        let key = compute_shard_key_digest_with_collector(&shard.key, collector);
        let replay = compute_replay_binding_digest_with_collector(
            segment.segment_index,
            key,
            shard.key.relation_digest,
            &shard.replay,
            collector,
        );
        let event = collector.compress(key, replay);
        running = collector.compress(running, event);
    }

    let endpoint = compute_endpoint_digest_v19(
        TAG_LOGUP_ENDPOINT_V19,
        segment.logup.verifier_endpoint,
        collector,
    );
    let openings_and_endpoint = collector.compress(segment.logup.segment_openings_digest, endpoint);
    let logup_meta = collector.compress(
        scalar_digest(&[
            TAG_LOGUP_META_V19,
            NATIVE_WARP_HISTORY_PROTOCOL_V19,
            segment.logup.mode_tag,
            u32::from(segment_lo),
            u32::from(segment_hi),
        ]),
        segment.logup.checkpoint_digest,
    );
    let logup_event = collector.compress(logup_meta, openings_and_endpoint);
    running = collector.compress(running, logup_event);

    let end = collector.compress(
        scalar_digest(&[
            TAG_SEGMENT_END_V19,
            NATIVE_WARP_HISTORY_PROTOCOL_V19,
            u32::from(segment_lo),
            u32::from(segment_hi),
            segment.active_shards.len() as u32,
            u32::from(segment.terminates),
        ]),
        [F::ZERO; DIGEST_SIZE],
    );
    collector.compress(running, end)
}

pub fn compute_terminal_certificate_digest_v19(
    profile_digest: DigestV19,
    boundary: &HistoryBoundaryV19,
    terminal_chunk: bool,
    collector: Option<&mut DigestCollectorV19>,
) -> DigestV19 {
    let mut local = DigestCollectorV19::default();
    let collector = collector_or_local(collector, &mut local);
    let (count_lo, count_hi) = split_u32(boundary.segment_count);
    let metadata = collector.compress(
        scalar_digest(&[
            TAG_TERMINAL_V19,
            NATIVE_WARP_HISTORY_PROTOCOL_V19,
            u32::from(count_lo),
            u32::from(count_hi),
            u32::from(boundary.terminated),
            u32::from(terminal_chunk),
        ]),
        profile_digest,
    );
    let state = collector.compress(boundary.vm_state, boundary.product_state_root);
    let history_and_public =
        collector.compress(boundary.history_root, boundary.public_values_digest);
    let boundary_digest = collector.compress(state, history_and_public);
    collector.compress(metadata, boundary_digest)
}

fn compute_shard_key_digest_with_collector(
    key: &ShardKeyRecordV19,
    collector: &mut DigestCollectorV19,
) -> DigestV19 {
    let (air_lo, air_hi) = split_u32(key.air_id);
    let metadata = scalar_digest(&[
        TAG_KEY_META_V19,
        NATIVE_WARP_HISTORY_PROTOCOL_V19,
        u32::from(key.ordinal),
        u32::from(air_lo),
        u32::from(air_hi),
        u32::from(key.log_height),
    ]);
    let key_meta = collector.compress(metadata, key.app_vk_digest);
    let layout = collector.compress(key.trace_layout_digest, key.public_schema_digest);
    let schemas = collector.compress(key.interaction_schema_digest, key.code_class_digest);
    let relation = collector.compress(key_meta, key.relation_digest);
    let structure = collector.compress(layout, schemas);
    collector.compress(relation, structure)
}

fn compute_replay_binding_digest_with_collector(
    segment_index: u32,
    key_digest: DigestV19,
    relation_digest: DigestV19,
    replay: &WarpCertifiedReplayRecordV19,
    collector: &mut DigestCollectorV19,
) -> DigestV19 {
    let (segment_lo, segment_hi) = split_u32(segment_index);
    let replay_meta = collector.compress(
        scalar_digest(&[
            TAG_REPLAY_META_V19,
            NATIVE_WARP_HISTORY_PROTOCOL_V19,
            u32::from(segment_lo),
            u32::from(segment_hi),
        ]),
        relation_digest,
    );
    let accumulators = collector.compress(
        replay.previous_accumulator_digest,
        replay.next_accumulator_digest,
    );
    let checkpoints = collector.compress(
        replay.previous_checkpoint_digest,
        replay.next_checkpoint_digest,
    );
    let source_and_endpoint =
        collector.compress(replay.opening_claim_digest, replay.replay_endpoint_digest);
    let meta_and_accumulators = collector.compress(replay_meta, accumulators);
    let checkpoints_and_endpoint = collector.compress(checkpoints, source_and_endpoint);
    let replay_digest = collector.compress(meta_and_accumulators, checkpoints_and_endpoint);
    collector.compress(key_digest, replay_digest)
}

#[inline]
pub(crate) fn scalar_digest(values: &[u32]) -> DigestV19 {
    let mut digest = [F::ZERO; DIGEST_SIZE];
    for (slot, value) in digest.iter_mut().zip(values.iter().copied()) {
        *slot = F::from_u32(value);
    }
    digest
}

#[inline]
pub(crate) fn join_digests(left: DigestV19, right: DigestV19) -> [F; 2 * DIGEST_SIZE] {
    core::array::from_fn(|index| {
        if index < DIGEST_SIZE {
            left[index]
        } else {
            right[index - DIGEST_SIZE]
        }
    })
}

#[inline]
const fn split_u32(value: u32) -> (u16, u16) {
    (value as u16, (value >> 16) as u16)
}

const _: () = assert!(2 * DIGEST_SIZE == 16);

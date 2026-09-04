use core::borrow::{Borrow, BorrowMut};

use openvm_stark_backend::{
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::dense::RowMajorMatrix,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, D_EF, F};

use super::{
    air::HASH_SLOTS_V19,
    digest::{
        scalar_digest, TAG_KEY_META_V19, TAG_LOGUP_ENDPOINT_V19, TAG_LOGUP_META_V19,
        TAG_REPLAY_META_V19, TAG_SEGMENT_END_V19, TAG_SEGMENT_HEADER_V19, TAG_TERMINAL_V19,
    },
    record::split_u32,
    CertifiedWarpReplayMessageV19, DigestCollectorV19, DigestV19, ExtensionV19, HistoryBoundaryV19,
    HistoryPublicValuesV19, HistoryRecordV19, HistoryRowColsV19, HistoryV19Error,
    HistoryVerifierProfileV19, LogUpOnlyHistoryMessageV19, SegmentManifestRecordV19,
    ShardTransitionRecordV19, LOGUP_ONLY_MODE_TAG_V19, NATIVE_WARP_HISTORY_PROTOCOL_V19,
};

#[derive(Clone, Debug)]
pub struct HistoryTraceV19 {
    pub matrix: RowMajorMatrix<F>,
    pub public_values: Vec<F>,
    pub compression_inputs: Vec<[F; 2 * DIGEST_SIZE]>,
    pub certified_replays: Vec<CertifiedWarpReplayMessageV19<F>>,
    pub logup_updates: Vec<LogUpOnlyHistoryMessageV19<F>>,
    pub active_rows: usize,
}

#[derive(Clone)]
struct State {
    count: u32,
    vm: DigestV19,
    product: DigestV19,
    history: DigestV19,
    public_values: DigestV19,
    terminated: bool,
}

impl From<&HistoryBoundaryV19> for State {
    fn from(value: &HistoryBoundaryV19) -> Self {
        Self {
            count: value.segment_count,
            vm: value.vm_state,
            product: value.product_state_root,
            history: value.history_root,
            public_values: value.public_values_digest,
            terminated: value.terminated,
        }
    }
}

impl State {
    fn boundary(&self) -> HistoryBoundaryV19 {
        HistoryBoundaryV19 {
            segment_count: self.count,
            vm_state: self.vm,
            product_state_root: self.product,
            history_root: self.history,
            public_values_digest: self.public_values,
            terminated: self.terminated,
        }
    }
}

#[derive(Clone, Copy, Default)]
struct OrderState {
    seen: bool,
    last_ordinal: u16,
    count: u16,
}

/// Slow, deterministic CPU witness generator and protocol oracle.
pub fn generate_history_trace_v19(
    profile: &HistoryVerifierProfileV19,
    record: &HistoryRecordV19,
) -> Result<HistoryTraceV19, HistoryV19Error> {
    if record.segments.is_empty() {
        return Err(HistoryV19Error::EmptyHistory);
    }
    if record.segments.len() > usize::from(profile.max_segments_per_chunk) {
        return Err(HistoryV19Error::TooManySegments {
            maximum: usize::from(profile.max_segments_per_chunk),
            actual: record.segments.len(),
        });
    }
    let mut collector = DigestCollectorV19::default();
    let mut replays = Vec::new();
    let mut logups = Vec::new();
    let width = HistoryRowColsV19::<F>::width();
    let mut rows = Vec::<Vec<F>>::new();
    let mut state = State::from(&record.initial);

    for (segment_position, segment) in record.segments.iter().enumerate() {
        validate_segment_header(profile, segment, segment_position, &state)?;
        if segment.active_shards.len() > usize::from(profile.max_active_shards_per_segment) {
            return Err(HistoryV19Error::TooManyShards {
                segment: segment_position,
                maximum: usize::from(profile.max_active_shards_per_segment),
                actual: segment.active_shards.len(),
            });
        }
        if segment.terminates && segment_position + 1 != record.segments.len() {
            return Err(HistoryV19Error::SegmentAfterTermination {
                segment: segment_position + 1,
            });
        }

        // Validate the complete canonical manifest order before performing any
        // path work.  This makes order rejection independent of stale Merkle
        // paths created by an adversarial permutation.
        let mut validation_order = OrderState::default();
        for (shard_position, shard) in segment.active_shards.iter().enumerate() {
            validate_shard_header(
                profile,
                shard,
                segment_position,
                shard_position,
                validation_order,
            )?;
            validation_order.seen = true;
            validation_order.last_ordinal = shard.key.ordinal;
            validation_order.count = validation_order
                .count
                .checked_add(1)
                .ok_or(HistoryV19Error::IntegerOverflow)?;
        }

        let mut order = OrderState::default();
        let mut manifest = [F::ZERO; DIGEST_SIZE];
        push_segment_start(
            &mut rows,
            width,
            &mut collector,
            &state,
            segment,
            order,
            &mut manifest,
        );

        for (shard_position, shard) in segment.active_shards.iter().enumerate() {
            let (key_digest, previous_leaf, next_leaf, replay_binding) = push_shard(
                &mut rows,
                width,
                &mut collector,
                profile,
                &state,
                segment,
                shard,
                order,
                manifest,
            );
            if replay_binding != shard.replay.replay_binding_digest {
                return Err(HistoryV19Error::ReplayBindingMismatch {
                    segment: segment_position,
                    shard: shard_position,
                });
            }
            replays.push(CertifiedWarpReplayMessageV19 {
                protocol_version: F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                segment_index_lo: F::from_u16(split_u32(segment.segment_index).0),
                segment_index_hi: F::from_u16(split_u32(segment.segment_index).1),
                update_index_lo: F::from_u16(split_u32(shard.replay.update_index).0),
                update_index_hi: F::from_u16(split_u32(shard.replay.update_index).1),
                shard_ordinal: F::from_u16(shard.key.ordinal),
                key_digest,
                relation_digest: shard.key.relation_digest,
                source_forest_root: segment.source_forest_root,
                opening_claim_digest: shard.replay.opening_claim_digest,
                fresh_instance_digest: shard.replay.fresh_instance_digest,
                segment_openings_digest: shard.replay.segment_openings_digest,
                prior_root: shard.replay.prior_root,
                fresh_root: shard.replay.fresh_root,
                next_root: shard.replay.next_root,
                previous_accumulator_digest: shard.replay.previous_accumulator_digest,
                next_accumulator_digest: shard.replay.next_accumulator_digest,
                authenticated_batching_claim: shard.replay.authenticated_batching_claim,
                previous_checkpoint_digest: shard.replay.previous_checkpoint_digest,
                next_checkpoint_digest: shard.replay.next_checkpoint_digest,
                replay_endpoint_digest: shard.replay.replay_endpoint_digest,
                replay_binding_digest: replay_binding,
            });
            order.seen = true;
            order.last_ordinal = shard.key.ordinal;
            order.count = order
                .count
                .checked_add(1)
                .ok_or(HistoryV19Error::IntegerOverflow)?;
            manifest = row_cols(rows.last().expect("shard row"), width).manifest_after;

            push_merkle_path(
                &mut rows,
                width,
                &mut collector,
                profile,
                &mut state,
                segment,
                shard,
                key_digest,
                previous_leaf,
                next_leaf,
                manifest,
                order,
                segment_position,
                shard_position,
            )?;
        }

        validate_logup(segment, segment_position, &state)?;
        push_logup(
            &mut rows,
            width,
            &mut collector,
            &mut state,
            segment,
            manifest,
            order,
        );
        manifest = row_cols(rows.last().expect("logup row"), width).manifest_after;
        logups.push(LogUpOnlyHistoryMessageV19 {
            protocol_version: F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            mode_tag: F::from_u32(segment.logup.mode_tag),
            segment_index_lo: F::from_u16(split_u32(segment.segment_index).0),
            segment_index_hi: F::from_u16(split_u32(segment.segment_index).1),
            app_vk_digest: profile.app_vk_digest,
            source_forest_root: segment.source_forest_root,
            segment_openings_digest: segment.logup.segment_openings_digest,
            verifier_endpoint: segment.logup.verifier_endpoint,
            checkpoint_digest: segment.logup.checkpoint_digest,
        });

        push_segment_end(
            &mut rows,
            width,
            &mut collector,
            profile,
            &mut state,
            segment,
            manifest,
            order,
            segment_position,
        )?;
    }

    let final_boundary = state.boundary();
    if final_boundary != record.expected_final {
        return Err(HistoryV19Error::FinalBoundaryMismatch);
    }
    if record.terminal_chunk && !state.terminated {
        return Err(HistoryV19Error::TerminalStateRequired);
    }
    let terminal_digest = push_final(
        &mut rows,
        width,
        &mut collector,
        profile,
        &state,
        record.terminal_chunk,
    );
    let active_rows = rows.len();
    if active_rows > profile.maximum_active_rows() || active_rows > profile.trace_height {
        return Err(HistoryV19Error::TraceCapacityExceeded {
            maximum: profile.maximum_active_rows(),
            actual: active_rows,
        });
    }
    rows.resize_with(profile.trace_height, || vec![F::ZERO; width]);
    let matrix = RowMajorMatrix::new(rows.into_iter().flatten().collect(), width);
    let public_values = HistoryPublicValuesV19::from_boundaries(
        profile.profile_digest,
        profile.app_vk_digest,
        &record.initial,
        &record.expected_final,
        record.segments.last(),
        record.terminal_chunk,
        terminal_digest,
    );
    Ok(HistoryTraceV19 {
        matrix,
        public_values,
        compression_inputs: collector.compression_inputs,
        certified_replays: replays,
        logup_updates: logups,
        active_rows,
    })
}

fn validate_segment_header(
    _profile: &HistoryVerifierProfileV19,
    segment: &SegmentManifestRecordV19,
    segment_position: usize,
    state: &State,
) -> Result<(), HistoryV19Error> {
    if state.terminated {
        return Err(HistoryV19Error::SegmentAfterTermination {
            segment: segment_position,
        });
    }
    if segment.segment_index != state.count {
        return Err(HistoryV19Error::SegmentIndexMismatch {
            segment: segment_position,
        });
    }
    if segment.from_vm_state != state.vm {
        return Err(HistoryV19Error::VmBoundaryDiscontinuity {
            segment: segment_position,
        });
    }
    if segment.previous_product_state_root != state.product {
        return Err(HistoryV19Error::ProductRootMismatch {
            segment: segment_position,
            shard: None,
        });
    }
    Ok(())
}

fn validate_shard_header(
    profile: &HistoryVerifierProfileV19,
    shard: &ShardTransitionRecordV19,
    segment_position: usize,
    shard_position: usize,
    order: OrderState,
) -> Result<(), HistoryV19Error> {
    if shard.key.app_vk_digest != profile.app_vk_digest {
        return Err(HistoryV19Error::ShardAppVkMismatch {
            segment: segment_position,
            shard: shard_position,
        });
    }
    if shard.key.ordinal >= profile.shard_catalog_len {
        return Err(HistoryV19Error::ShardOrdinalOutOfRange {
            segment: segment_position,
            shard: shard_position,
        });
    }
    if order.seen && shard.key.ordinal == order.last_ordinal {
        return Err(HistoryV19Error::DuplicateShardKey {
            segment: segment_position,
            shard: shard_position,
        });
    }
    if order.seen && shard.key.ordinal < order.last_ordinal {
        return Err(HistoryV19Error::NonCanonicalShardOrder {
            segment: segment_position,
            shard: shard_position,
        });
    }
    let expected_len = usize::from(profile.product_tree_height);
    if shard.product_siblings.len() != expected_len || shard.catalog_siblings.len() != expected_len
    {
        return Err(HistoryV19Error::MerklePathLengthMismatch {
            segment: segment_position,
            shard: shard_position,
        });
    }
    Ok(())
}

fn validate_logup(
    segment: &SegmentManifestRecordV19,
    segment_position: usize,
    _state: &State,
) -> Result<(), HistoryV19Error> {
    if segment.logup.mode_tag != LOGUP_ONLY_MODE_TAG_V19 {
        return Err(HistoryV19Error::LogUpModeMismatch {
            segment: segment_position,
        });
    }
    Ok(())
}

fn push_segment_start(
    rows: &mut Vec<Vec<F>>,
    width: usize,
    collector: &mut DigestCollectorV19,
    state: &State,
    segment: &SegmentManifestRecordV19,
    order: OrderState,
    manifest: &mut DigestV19,
) {
    let mut row = vec![F::ZERO; width];
    let cols: &mut HistoryRowColsV19<F> = row.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.is_segment_start = F::ONE;
    fill_state(cols, state, state, false);
    fill_context(cols, segment);
    fill_order(cols, order, order);
    let (segment_lo, segment_hi) = split_u32(segment.segment_index);
    cols.hash_outputs[0] = collector.compress(
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
    cols.hash_outputs[1] = collector.compress(segment.from_vm_state, segment.to_vm_state);
    cols.hash_outputs[2] = collector.compress(
        segment.previous_product_state_root,
        segment.next_product_state_root,
    );
    cols.hash_outputs[3] = collector.compress(
        segment.logup.segment_openings_digest,
        [F::ZERO; DIGEST_SIZE],
    );
    cols.hash_outputs[4] =
        collector.compress(segment.source_forest_root, segment.public_values_digest);
    cols.hash_outputs[5] = collector.compress(cols.hash_outputs[4], cols.hash_outputs[3]);
    cols.hash_outputs[6] = collector.compress(cols.hash_outputs[0], cols.hash_outputs[1]);
    cols.hash_outputs[7] = collector.compress(cols.hash_outputs[2], cols.hash_outputs[5]);
    cols.hash_outputs[8] = collector.compress(cols.hash_outputs[6], cols.hash_outputs[7]);
    cols.manifest_after = cols.hash_outputs[8];
    *manifest = cols.manifest_after;
    rows.push(row);
}

#[allow(clippy::too_many_arguments)]
fn push_shard(
    rows: &mut Vec<Vec<F>>,
    width: usize,
    collector: &mut DigestCollectorV19,
    profile: &HistoryVerifierProfileV19,
    state: &State,
    segment: &SegmentManifestRecordV19,
    shard: &ShardTransitionRecordV19,
    order_before: OrderState,
    manifest_before: DigestV19,
) -> (DigestV19, DigestV19, DigestV19, DigestV19) {
    let mut order_after = order_before;
    order_after.seen = true;
    order_after.last_ordinal = shard.key.ordinal;
    order_after.count += 1;
    let mut row = vec![F::ZERO; width];
    let cols: &mut HistoryRowColsV19<F> = row.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.is_shard = F::ONE;
    fill_state(cols, state, state, false);
    fill_context(cols, segment);
    fill_order(cols, order_before, order_after);
    fill_shard(cols, profile, shard, order_before);
    cols.manifest_before = manifest_before;

    let (air_lo, air_hi) = split_u32(shard.key.air_id);
    cols.hash_outputs[0] = collector.compress(
        scalar_digest(&[
            TAG_KEY_META_V19,
            NATIVE_WARP_HISTORY_PROTOCOL_V19,
            u32::from(shard.key.ordinal),
            u32::from(air_lo),
            u32::from(air_hi),
            u32::from(shard.key.log_height),
        ]),
        shard.key.app_vk_digest,
    );
    cols.hash_outputs[1] = collector.compress(
        shard.key.trace_layout_digest,
        shard.key.public_schema_digest,
    );
    cols.hash_outputs[2] = collector.compress(
        shard.key.interaction_schema_digest,
        shard.key.code_class_digest,
    );
    cols.hash_outputs[3] = collector.compress(cols.hash_outputs[0], shard.key.relation_digest);
    cols.hash_outputs[4] = collector.compress(cols.hash_outputs[1], cols.hash_outputs[2]);
    cols.hash_outputs[5] = collector.compress(cols.hash_outputs[3], cols.hash_outputs[4]);
    let key_digest = cols.hash_outputs[5];

    let (segment_lo, segment_hi) = split_u32(segment.segment_index);
    cols.hash_outputs[6] = collector.compress(
        scalar_digest(&[
            TAG_REPLAY_META_V19,
            NATIVE_WARP_HISTORY_PROTOCOL_V19,
            u32::from(segment_lo),
            u32::from(segment_hi),
        ]),
        shard.key.relation_digest,
    );
    cols.hash_outputs[7] = collector.compress(
        shard.replay.previous_accumulator_digest,
        shard.replay.next_accumulator_digest,
    );
    cols.hash_outputs[8] = collector.compress(
        shard.replay.previous_checkpoint_digest,
        shard.replay.next_checkpoint_digest,
    );
    cols.hash_outputs[9] = collector.compress(
        shard.replay.opening_claim_digest,
        shard.replay.replay_endpoint_digest,
    );
    cols.hash_outputs[10] = collector.compress(cols.hash_outputs[6], cols.hash_outputs[7]);
    cols.hash_outputs[11] = collector.compress(cols.hash_outputs[8], cols.hash_outputs[9]);
    cols.hash_outputs[12] = collector.compress(cols.hash_outputs[10], cols.hash_outputs[11]);
    cols.hash_outputs[13] = collector.compress(key_digest, cols.hash_outputs[12]);
    let replay_binding = cols.hash_outputs[13];
    cols.replay_binding_digest = shard.replay.replay_binding_digest;

    cols.hash_outputs[14] = collector.compress(
        shard.replay.previous_accumulator_digest,
        shard.replay.previous_checkpoint_digest,
    );
    cols.hash_outputs[15] = collector.compress(
        shard.replay.next_accumulator_digest,
        shard.replay.next_checkpoint_digest,
    );
    cols.hash_outputs[16] = collector.compress(key_digest, cols.hash_outputs[14]);
    cols.hash_outputs[17] = collector.compress(key_digest, cols.hash_outputs[15]);
    cols.previous_leaf_digest = cols.hash_outputs[16];
    cols.next_leaf_digest = cols.hash_outputs[17];
    cols.hash_outputs[18] = collector.compress(key_digest, replay_binding);
    cols.hash_outputs[19] = collector.compress(manifest_before, cols.hash_outputs[18]);
    cols.manifest_after = cols.hash_outputs[19];
    let previous_leaf = cols.previous_leaf_digest;
    let next_leaf = cols.next_leaf_digest;
    rows.push(row);
    (key_digest, previous_leaf, next_leaf, replay_binding)
}

#[allow(clippy::too_many_arguments)]
fn push_merkle_path(
    rows: &mut Vec<Vec<F>>,
    width: usize,
    collector: &mut DigestCollectorV19,
    profile: &HistoryVerifierProfileV19,
    state: &mut State,
    segment: &SegmentManifestRecordV19,
    shard: &ShardTransitionRecordV19,
    key_digest: DigestV19,
    previous_leaf: DigestV19,
    next_leaf: DigestV19,
    manifest: DigestV19,
    order: OrderState,
    segment_position: usize,
    shard_position: usize,
) -> Result<(), HistoryV19Error> {
    let mut old_node = previous_leaf;
    let mut new_node = next_leaf;
    let mut catalog_node = key_digest;
    let mut index_accumulator = 0u32;
    let mut weight = 1u32;
    for depth in 0..usize::from(profile.product_tree_height) {
        let state_before = state.clone();
        let bit = ((u32::from(shard.key.ordinal) >> depth) & 1) == 1;
        let product_sibling = shard.product_siblings[depth];
        let catalog_sibling = shard.catalog_siblings[depth];
        let old_parent = merkle_parent(old_node, product_sibling, bit, collector);
        let new_parent = merkle_parent(new_node, product_sibling, bit, collector);
        let catalog_parent = merkle_parent(catalog_node, catalog_sibling, bit, collector);
        let is_last = depth + 1 == usize::from(profile.product_tree_height);
        index_accumulator += u32::from(bit) * weight;
        if is_last {
            if old_parent != state.product {
                return Err(HistoryV19Error::ProductRootMismatch {
                    segment: segment_position,
                    shard: Some(shard_position),
                });
            }
            if catalog_parent != profile.shard_catalog_root {
                return Err(HistoryV19Error::ShardCatalogAuthenticationFailed {
                    segment: segment_position,
                    shard: shard_position,
                });
            }
            state.product = new_parent;
        }
        let mut row = vec![F::ZERO; width];
        let cols: &mut HistoryRowColsV19<F> = row.as_mut_slice().borrow_mut();
        cols.active = F::ONE;
        cols.is_path = F::ONE;
        fill_state(cols, &state_before, state, false);
        fill_context(cols, segment);
        fill_order(cols, order, order);
        cols.manifest_before = manifest;
        cols.manifest_after = manifest;
        fill_u16(cols.shard_ordinal_bits.as_mut_slice(), shard.key.ordinal);
        cols.shard_ordinal = F::from_u16(shard.key.ordinal);
        cols.previous_leaf_digest = previous_leaf;
        cols.next_leaf_digest = next_leaf;
        cols.hash_outputs[5] = key_digest;
        cols.path_is_first = F::from_bool(depth == 0);
        cols.path_is_last = F::from_bool(is_last);
        cols.path_depth = F::from_usize(depth);
        let final_depth = F::from_u8(profile.product_tree_height - 1);
        cols.path_not_last_inv = if is_last {
            F::ZERO
        } else {
            (cols.path_depth - final_depth).inverse()
        };
        cols.path_index_bit = F::from_bool(bit);
        cols.path_index_before = F::from_u32(index_accumulator - u32::from(bit) * weight);
        cols.path_index_after = F::from_u32(index_accumulator);
        cols.path_bit_weight = F::from_u32(weight);
        cols.path_old_node = old_node;
        cols.path_new_node = new_node;
        cols.path_catalog_node = catalog_node;
        cols.path_product_sibling = product_sibling;
        cols.path_catalog_sibling = catalog_sibling;
        cols.path_old_parent = old_parent;
        cols.path_new_parent = new_parent;
        cols.path_catalog_parent = catalog_parent;
        cols.hash_outputs[0] = old_parent;
        cols.hash_outputs[1] = new_parent;
        cols.hash_outputs[2] = catalog_parent;
        rows.push(row);
        old_node = old_parent;
        new_node = new_parent;
        catalog_node = catalog_parent;
        weight = weight
            .checked_mul(2)
            .ok_or(HistoryV19Error::IntegerOverflow)?;
    }
    Ok(())
}

fn push_logup(
    rows: &mut Vec<Vec<F>>,
    width: usize,
    collector: &mut DigestCollectorV19,
    state: &mut State,
    segment: &SegmentManifestRecordV19,
    manifest_before: DigestV19,
    order: OrderState,
) {
    let state_before = state.clone();
    let mut row = vec![F::ZERO; width];
    let cols: &mut HistoryRowColsV19<F> = row.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.is_logup = F::ONE;
    fill_state(cols, &state_before, state, false);
    fill_context(cols, segment);
    fill_order(cols, order, order);
    cols.manifest_before = manifest_before;
    cols.logup_mode_tag = F::from_u32(segment.logup.mode_tag);
    cols.logup_checkpoint_digest = segment.logup.checkpoint_digest;
    cols.logup_segment_openings_digest = segment.logup.segment_openings_digest;
    cols.logup_verifier_endpoint = segment.logup.verifier_endpoint;
    cols.hash_outputs[0] = endpoint_digest(
        TAG_LOGUP_ENDPOINT_V19,
        segment.logup.verifier_endpoint,
        collector,
    );
    cols.hash_outputs[1] =
        collector.compress(segment.logup.segment_openings_digest, cols.hash_outputs[0]);
    let (segment_lo, segment_hi) = split_u32(segment.segment_index);
    cols.hash_outputs[2] = collector.compress(
        scalar_digest(&[
            TAG_LOGUP_META_V19,
            NATIVE_WARP_HISTORY_PROTOCOL_V19,
            segment.logup.mode_tag,
            u32::from(segment_lo),
            u32::from(segment_hi),
        ]),
        segment.logup.checkpoint_digest,
    );
    cols.hash_outputs[3] = collector.compress(cols.hash_outputs[2], cols.hash_outputs[1]);
    cols.hash_outputs[4] = collector.compress(manifest_before, cols.hash_outputs[3]);
    cols.manifest_after = cols.hash_outputs[4];
    rows.push(row);
}

#[allow(clippy::too_many_arguments)]
fn push_segment_end(
    rows: &mut Vec<Vec<F>>,
    width: usize,
    collector: &mut DigestCollectorV19,
    profile: &HistoryVerifierProfileV19,
    state: &mut State,
    segment: &SegmentManifestRecordV19,
    manifest_before: DigestV19,
    order: OrderState,
    segment_position: usize,
) -> Result<(), HistoryV19Error> {
    if state.product != segment.next_product_state_root {
        return Err(HistoryV19Error::ProductRootMismatch {
            segment: segment_position,
            shard: None,
        });
    }
    let state_before = state.clone();
    let (segment_lo, segment_hi) = split_u32(segment.segment_index);
    let end_meta = collector.compress(
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
    let manifest = collector.compress(manifest_before, end_meta);
    if manifest != segment.manifest_digest {
        return Err(HistoryV19Error::ManifestDigestMismatch {
            segment: segment_position,
        });
    }
    let next_history = collector.compress(state.history, manifest);
    state.count = state
        .count
        .checked_add(1)
        .ok_or(HistoryV19Error::IntegerOverflow)?;
    state.vm = segment.to_vm_state;
    state.history = next_history;
    state.public_values = segment.public_values_digest;
    state.terminated = segment.terminates;

    let mut row = vec![F::ZERO; width];
    let cols: &mut HistoryRowColsV19<F> = row.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.is_segment_end = F::ONE;
    let carry = split_u32(state_before.count).0 == u16::MAX;
    fill_state(cols, &state_before, state, carry);
    fill_context(cols, segment);
    fill_order(cols, order, order);
    cols.manifest_before = manifest_before;
    cols.manifest_after = manifest;
    cols.hash_outputs[0] = end_meta;
    cols.hash_outputs[1] = manifest;
    cols.hash_outputs[2] = next_history;
    fill_u16(
        cols.shard_capacity_remaining_bits.as_mut_slice(),
        profile
            .max_active_shards_per_segment
            .saturating_sub(order.count),
    );
    rows.push(row);
    Ok(())
}

fn push_final(
    rows: &mut Vec<Vec<F>>,
    width: usize,
    collector: &mut DigestCollectorV19,
    profile: &HistoryVerifierProfileV19,
    state: &State,
    terminal_chunk: bool,
) -> DigestV19 {
    let mut row = vec![F::ZERO; width];
    let cols: &mut HistoryRowColsV19<F> = row.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.is_final = F::ONE;
    fill_state(cols, state, state, false);
    let (count_lo, count_hi) = split_u32(state.count);
    cols.hash_outputs[0] = collector.compress(
        scalar_digest(&[
            TAG_TERMINAL_V19,
            NATIVE_WARP_HISTORY_PROTOCOL_V19,
            u32::from(count_lo),
            u32::from(count_hi),
            u32::from(state.terminated),
            u32::from(terminal_chunk),
        ]),
        profile.profile_digest,
    );
    cols.hash_outputs[1] = collector.compress(state.vm, state.product);
    cols.hash_outputs[2] = collector.compress(state.history, state.public_values);
    cols.hash_outputs[3] = collector.compress(cols.hash_outputs[1], cols.hash_outputs[2]);
    cols.hash_outputs[4] = collector.compress(cols.hash_outputs[0], cols.hash_outputs[3]);
    let result = cols.hash_outputs[4];
    rows.push(row);
    result
}

fn fill_state(cols: &mut HistoryRowColsV19<F>, before: &State, after: &State, count_carry: bool) {
    let (before_lo, before_hi) = split_u32(before.count);
    let (after_lo, after_hi) = split_u32(after.count);
    cols.state_before_count_lo = F::from_u16(before_lo);
    cols.state_before_count_hi = F::from_u16(before_hi);
    fill_u16(&mut cols.state_before_count_bits[0], before_lo);
    fill_u16(&mut cols.state_before_count_bits[1], before_hi);
    cols.state_before_vm = before.vm;
    cols.state_before_product = before.product;
    cols.state_before_history = before.history;
    cols.state_before_public_values = before.public_values;
    cols.state_before_terminated = F::from_bool(before.terminated);
    cols.state_after_count_lo = F::from_u16(after_lo);
    cols.state_after_count_hi = F::from_u16(after_hi);
    fill_u16(&mut cols.state_after_count_bits[0], after_lo);
    fill_u16(&mut cols.state_after_count_bits[1], after_hi);
    cols.state_after_vm = after.vm;
    cols.state_after_product = after.product;
    cols.state_after_history = after.history;
    cols.state_after_public_values = after.public_values;
    cols.state_after_terminated = F::from_bool(after.terminated);
    cols.count_increment_carry = F::from_bool(count_carry);
}

fn fill_context(cols: &mut HistoryRowColsV19<F>, segment: &SegmentManifestRecordV19) {
    let (lo, hi) = split_u32(segment.segment_index);
    cols.ctx_segment_index_lo = F::from_u16(lo);
    cols.ctx_segment_index_hi = F::from_u16(hi);
    fill_u16(&mut cols.ctx_segment_index_bits[0], lo);
    fill_u16(&mut cols.ctx_segment_index_bits[1], hi);
    cols.ctx_initial_pc = segment.initial_pc;
    cols.ctx_final_pc = segment.final_pc;
    cols.ctx_exit_code = segment.exit_code;
    cols.ctx_initial_memory_root = segment.initial_memory_root;
    cols.ctx_final_memory_root = segment.final_memory_root;
    cols.ctx_program_fingerprint = segment.program_fingerprint;
    cols.ctx_program_fingerprint_digest = segment.program_fingerprint_digest;
    cols.ctx_program_registry_digest = segment.program_registry_digest;
    cols.ctx_program_relation_digest = segment.program_relation_digest;
    cols.ctx_program_log_height = F::from_u8(segment.program_log_height);
    cols.ctx_program_cached_width = F::from_u32(segment.program_cached_width);
    cols.ctx_from_vm = segment.from_vm_state;
    cols.ctx_to_vm = segment.to_vm_state;
    cols.ctx_previous_product = segment.previous_product_state_root;
    cols.ctx_next_product = segment.next_product_state_root;
    cols.ctx_source_forest_root = segment.source_forest_root;
    cols.ctx_public_values = segment.public_values_digest;
    cols.ctx_terminates = F::from_bool(segment.terminates);
    cols.ctx_expected_shards = F::from_usize(segment.active_shards.len());
    fill_u16(
        &mut cols.ctx_expected_shards_bits,
        segment.active_shards.len() as u16,
    );
    cols.ctx_manifest_digest = segment.manifest_digest;
    cols.ctx_segment_openings_digest = segment.logup.segment_openings_digest;
}

fn fill_order(cols: &mut HistoryRowColsV19<F>, before: OrderState, after: OrderState) {
    cols.order_seen_before = F::from_bool(before.seen);
    cols.order_seen_after = F::from_bool(after.seen);
    cols.last_ordinal_before = F::from_u16(before.last_ordinal);
    cols.last_ordinal_after = F::from_u16(after.last_ordinal);
    cols.shard_count_before = F::from_u16(before.count);
    cols.shard_count_after = F::from_u16(after.count);
    fill_u16(&mut cols.shard_count_before_bits, before.count);
    fill_u16(&mut cols.shard_count_after_bits, after.count);
}

fn fill_shard(
    cols: &mut HistoryRowColsV19<F>,
    profile: &HistoryVerifierProfileV19,
    shard: &ShardTransitionRecordV19,
    order_before: OrderState,
) {
    cols.shard_ordinal = F::from_u16(shard.key.ordinal);
    fill_u16(&mut cols.shard_ordinal_bits, shard.key.ordinal);
    let gap = if order_before.seen {
        shard.key.ordinal - order_before.last_ordinal - 1
    } else {
        0
    };
    fill_u16(&mut cols.ordinal_gap_bits, gap);
    fill_u16(
        &mut cols.catalog_remaining_bits,
        profile
            .shard_catalog_len
            .saturating_sub(shard.key.ordinal)
            .saturating_sub(1),
    );
    cols.key_app_vk_digest = shard.key.app_vk_digest;
    let (air_lo, air_hi) = split_u32(shard.key.air_id);
    cols.key_air_id_lo = F::from_u16(air_lo);
    cols.key_air_id_hi = F::from_u16(air_hi);
    cols.key_log_height = F::from_u8(shard.key.log_height);
    cols.key_trace_layout_digest = shard.key.trace_layout_digest;
    cols.key_public_schema_digest = shard.key.public_schema_digest;
    cols.key_interaction_schema_digest = shard.key.interaction_schema_digest;
    cols.key_relation_digest = shard.key.relation_digest;
    cols.key_code_class_digest = shard.key.code_class_digest;
    let (update_lo, update_hi) = split_u32(shard.replay.update_index);
    cols.update_index_lo = F::from_u16(update_lo);
    cols.update_index_hi = F::from_u16(update_hi);
    fill_u16(&mut cols.update_index_bits[0], update_lo);
    fill_u16(&mut cols.update_index_bits[1], update_hi);
    cols.opening_claim_digest = shard.replay.opening_claim_digest;
    cols.fresh_instance_digest = shard.replay.fresh_instance_digest;
    cols.segment_openings_digest = shard.replay.segment_openings_digest;
    cols.prior_root = shard.replay.prior_root;
    cols.fresh_root = shard.replay.fresh_root;
    cols.next_root = shard.replay.next_root;
    cols.previous_accumulator_digest = shard.replay.previous_accumulator_digest;
    cols.next_accumulator_digest = shard.replay.next_accumulator_digest;
    cols.authenticated_batching_claim = shard.replay.authenticated_batching_claim;
    cols.previous_checkpoint_digest = shard.replay.previous_checkpoint_digest;
    cols.next_checkpoint_digest = shard.replay.next_checkpoint_digest;
    cols.replay_endpoint_digest = shard.replay.replay_endpoint_digest;
    cols.replay_binding_digest = shard.replay.replay_binding_digest;
}

fn endpoint_digest(
    tag: u32,
    endpoint: ExtensionV19,
    collector: &mut DigestCollectorV19,
) -> DigestV19 {
    let mut left = [F::ZERO; DIGEST_SIZE];
    left[0] = F::from_u32(tag);
    left[1..1 + D_EF].copy_from_slice(&endpoint);
    collector.compress(left, [F::ZERO; DIGEST_SIZE])
}

fn merkle_parent(
    node: DigestV19,
    sibling: DigestV19,
    index_bit: bool,
    collector: &mut DigestCollectorV19,
) -> DigestV19 {
    if index_bit {
        collector.compress(sibling, node)
    } else {
        collector.compress(node, sibling)
    }
}

fn fill_u16(target: &mut [F], value: u16) {
    for (bit, target) in target.iter_mut().enumerate() {
        *target = F::from_bool(((value >> bit) & 1) == 1);
    }
}

fn row_cols(row: &[F], width: usize) -> &HistoryRowColsV19<F> {
    debug_assert_eq!(row.len(), width);
    row.borrow()
}

const _: () = assert!(HASH_SLOTS_V19 >= 20);

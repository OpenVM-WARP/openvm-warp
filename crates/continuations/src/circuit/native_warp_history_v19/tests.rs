use core::borrow::{Borrow, BorrowMut};

use openvm_circuit::system::connector::DEFAULT_SUSPEND_EXIT_CODE;
use openvm_recursion_circuit::{
    bus::{CertifiedTranscriptCheckpointBus, CertifiedTranscriptCheckpointMessage},
    native_warp::{
        NativeCertifiedAccumulatorDigestBus, NativeCertifiedAccumulatorDigestMessage,
        NativeCertifiedBatchingClaimBus, NativeCertifiedBatchingClaimMessage,
    },
};
use openvm_stark_backend::{
    air_builders::{
        debug::{check_constraints, check_logup},
        symbolic::{get_symbolic_builder, SymbolicRapBuilder},
    },
    interaction::{InteractionBuilder, SymbolicInteraction},
    keygen::types::TraceWidth,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, BabyBearPoseidon2Config, DIGEST_SIZE, D_EF, F,
};

use super::{
    compute_manifest_digest_v19, compute_replay_binding_digest_v19, compute_shard_key_digest_v19,
    digest::compute_product_leaf_digest_v19, generate_history_trace_v19,
    generate_logup_only_producer_trace_v19, generate_warp_replay_producer_trace_v19,
    CertifiedDirectAirVaccInputBusV19, CertifiedDirectAirVaccInputMessageV19,
    CertifiedFreshExplicitDigestBusV19, CertifiedLogUpOnlyEndpointBusV19,
    CertifiedLogUpOnlyEndpointMessageV19, CertifiedProgramFingerprintBusV19,
    CertifiedProgramFingerprintMessageV19, CertifiedSwirlRawOpeningBusV19,
    CertifiedSwirlRawOpeningMessageV19, CertifiedVmSegmentMetadataBusV19,
    CertifiedVmSegmentMetadataMessageV19, CertifiedWarpReplayBusV19, CertifiedWarpReplayMessageV19,
    DigestCollectorV19, DigestV19, DirectAirVaccContextBusV19, DirectAirVaccContextMessageV19,
    DirectAirVaccProducerRecordV19, HistoryAirV19, HistoryBoundaryV19, HistoryGenesisConfigV19,
    HistoryPoseidon2CompressBusV19, HistoryPoseidon2CompressMessageV19, HistoryPublicValuesV19,
    HistoryRecordV19, HistoryRowColsV19, HistoryV19Error, HistoryVerifierProfileV19,
    LogUpOnlyHistoryBusV19, LogUpOnlyHistoryMessageV19, LogUpOnlyProducerAirV19,
    LogUpOnlyProducerColsV19, LogUpOnlyProducerRecordV19, LogUpOnlyRecordV19,
    PositiveProducerErrorV19, PositiveProducerTraceV19, SegmentManifestRecordV19,
    ShardKeyRecordV19, ShardTransitionRecordV19, TranscriptCheckpointRecordV19,
    WarpCertifiedReplayRecordV19, WarpReplayProducerAirV19, WarpReplayProducerColsV19,
    LOGUP_ONLY_MODE_TAG_V19, MAX_FRESH_BETA_LEN_V19, NATIVE_WARP_HISTORY_PROTOCOL_V19,
};

const COMPRESS_BUS: u16 = 0;
const HISTORY_WARP_BUS: u16 = 1;
const HISTORY_LOGUP_BUS: u16 = 2;
const CONTEXT_BUS: u16 = 3;
const SWIRL_OPENING_BUS: u16 = 4;
const VACC_INPUT_BUS: u16 = 5;
const CHECKPOINT_BUS: u16 = 6;
const BATCHING_BUS: u16 = 7;
const NEXT_ACCUMULATOR_BUS: u16 = 8;
const LOGUP_ENDPOINT_BUS: u16 = 9;
const VM_METADATA_BUS: u16 = 10;
const PROGRAM_FINGERPRINT_BUS: u16 = 11;
const FRESH_EXPLICIT_BUS: u16 = 12;
const LOG_MESSAGE_LEN: usize = 3;
const LOG_CODEWORD_LEN: usize = 5;
const BETA_LEN: usize = 6;

fn digest(seed: u32) -> DigestV19 {
    core::array::from_fn(|index| F::from_u32(seed + index as u32))
}

fn extension(seed: u32) -> [F; D_EF] {
    core::array::from_fn(|index| F::from_u32(seed + index as u32))
}

fn checkpoint(seed: u32, operation_index: u32) -> TranscriptCheckpointRecordV19 {
    TranscriptCheckpointRecordV19 {
        operation_index,
        sample_count: (seed % 5) as u8,
        state: core::array::from_fn(|index| F::from_u32(seed + index as u32)),
    }
}

fn key(app_vk: DigestV19, ordinal: u16) -> ShardKeyRecordV19 {
    ShardKeyRecordV19 {
        ordinal,
        app_vk_digest: app_vk,
        air_id: 1000 + u32::from(ordinal),
        log_height: 10,
        trace_layout_digest: digest(100 + 10 * u32::from(ordinal)),
        public_schema_digest: digest(200 + 10 * u32::from(ordinal)),
        interaction_schema_digest: digest(300 + 10 * u32::from(ordinal)),
        relation_digest: digest(400 + 10 * u32::from(ordinal)),
        code_class_digest: digest(500),
    }
}

fn root_and_path(leaves: &[DigestV19], index: usize) -> (DigestV19, Vec<DigestV19>) {
    assert!(leaves.len().is_power_of_two());
    let mut layer = leaves.to_vec();
    let mut position = index;
    let mut path = Vec::new();
    while layer.len() > 1 {
        path.push(layer[position ^ 1]);
        layer = layer
            .chunks_exact(2)
            .map(|pair| poseidon2_compress_with_capacity(pair[0], pair[1]).0)
            .collect();
        position >>= 1;
    }
    (layer[0], path)
}

fn product_leaf(key: DigestV19, accumulator: DigestV19, checkpoint: DigestV19) -> DigestV19 {
    compute_product_leaf_digest_v19(
        key,
        accumulator,
        checkpoint,
        &mut DigestCollectorV19::default(),
    )
}

fn warp_air() -> WarpReplayProducerAirV19 {
    WarpReplayProducerAirV19 {
        log_message_len: LOG_MESSAGE_LEN,
        log_codeword_len: LOG_CODEWORD_LEN,
        beta_len: BETA_LEN,
        compress_bus: HistoryPoseidon2CompressBusV19::new(COMPRESS_BUS),
        history_bus: CertifiedWarpReplayBusV19::new(HISTORY_WARP_BUS),
        context_bus: DirectAirVaccContextBusV19::new(CONTEXT_BUS),
        swirl_opening_bus: CertifiedSwirlRawOpeningBusV19::new(SWIRL_OPENING_BUS),
        vacc_input_bus: CertifiedDirectAirVaccInputBusV19::new(VACC_INPUT_BUS),
        canonical_vacc_input_bus: None,
        fresh_explicit_bus: CertifiedFreshExplicitDigestBusV19::new(FRESH_EXPLICIT_BUS),
        fresh_explicit_lookup_count: 0,
        checkpoint_bus: CertifiedTranscriptCheckpointBus::new(CHECKPOINT_BUS),
        batching_claim_bus: NativeCertifiedBatchingClaimBus::new(BATCHING_BUS),
        next_accumulator_digest_bus: NativeCertifiedAccumulatorDigestBus::new(NEXT_ACCUMULATOR_BUS),
        setup_schedule: None,
    }
}

fn logup_air() -> LogUpOnlyProducerAirV19 {
    LogUpOnlyProducerAirV19 {
        compress_bus: HistoryPoseidon2CompressBusV19::new(COMPRESS_BUS),
        history_bus: LogUpOnlyHistoryBusV19::new(HISTORY_LOGUP_BUS),
        endpoint_bus: CertifiedLogUpOnlyEndpointBusV19::new(LOGUP_ENDPOINT_BUS),
        checkpoint_bus: CertifiedTranscriptCheckpointBus::new(CHECKPOINT_BUS),
    }
}

fn warp_record(
    key: &ShardKeyRecordV19,
    key_digest: DigestV19,
    source_root: DigestV19,
    openings_digest: DigestV19,
) -> DirectAirVaccProducerRecordV19 {
    let seed = 10_000 + 100 * u32::from(key.ordinal);
    let opening_point = (0..LOG_MESSAGE_LEN)
        .map(|index| extension(seed + 10 * index as u32))
        .collect::<Vec<_>>();
    let mut fresh_alpha = opening_point.clone();
    fresh_alpha.resize(LOG_CODEWORD_LEN, [F::ZERO; D_EF]);
    let opening_value = extension(seed + 80);
    DirectAirVaccProducerRecordV19 {
        proof_index: 200 + u32::from(key.ordinal),
        segment_index: 7,
        update_index: 20 + u32::from(key.ordinal),
        shard_ordinal: key.ordinal,
        has_prior: true,
        key_digest,
        relation_digest: key.relation_digest,
        source_forest_root: source_root,
        segment_openings_digest: openings_digest,
        prior_root: digest(seed + 100),
        fresh_root: digest(seed + 120),
        next_root: digest(seed + 140),
        opening_point,
        opening_value,
        fresh_alpha,
        fresh_mu: opening_value,
        fresh_beta: (0..BETA_LEN)
            .map(|index| extension(seed + 160 + 10 * index as u32))
            .collect(),
        fresh_eta: [F::ZERO; D_EF],
        previous_accumulator_digest: digest(seed + 240),
        next_accumulator_digest: digest(seed + 260),
        previous_checkpoint_digest: digest(seed + 270),
        next_checkpoint_digest: digest(seed + 275),
        authenticated_batching_claim: extension(seed + 280),
        start_checkpoint: checkpoint(seed + 300, 50 + u32::from(key.ordinal)),
        end_checkpoint: checkpoint(seed + 400, 70 + u32::from(key.ordinal)),
    }
}

struct Fixture {
    profile: HistoryVerifierProfileV19,
    history: HistoryRecordV19,
    warp_records: Vec<DirectAirVaccProducerRecordV19>,
    warp_trace: PositiveProducerTraceV19<CertifiedWarpReplayMessageV19<F>>,
    logup_record: LogUpOnlyProducerRecordV19,
    logup_trace: PositiveProducerTraceV19<LogUpOnlyHistoryMessageV19<F>>,
}

fn fixture() -> Fixture {
    let app_vk = digest(1);
    let source_root = digest(8000);
    let openings_digest = digest(8100);
    let keys = (0..4).map(|i| key(app_vk, i)).collect::<Vec<_>>();
    let key_digests = keys
        .iter()
        .map(|key| compute_shard_key_digest_v19(key, None))
        .collect::<Vec<_>>();
    let (catalog_root, _) = root_and_path(&key_digests, 0);
    let profile = HistoryVerifierProfileV19::new(app_vk, catalog_root, 4, 2, 4, 3).unwrap();
    let warp_records = [0usize, 2]
        .into_iter()
        .map(|i| warp_record(&keys[i], key_digests[i], source_root, openings_digest))
        .collect::<Vec<_>>();
    let warp_trace =
        generate_warp_replay_producer_trace_v19(&warp_air(), &warp_records, 4).unwrap();

    let mut previous_accumulators = [digest(9000), digest(9100), digest(9200), digest(9300)];
    let mut previous_checkpoints = [digest(9400), digest(9500), digest(9600), digest(9700)];
    for (record, message) in warp_records.iter().zip(&warp_trace.messages) {
        let i = usize::from(record.shard_ordinal);
        previous_accumulators[i] = record.previous_accumulator_digest;
        previous_checkpoints[i] = message.previous_checkpoint_digest;
    }
    let mut leaves = (0..4)
        .map(|i| {
            product_leaf(
                key_digests[i],
                previous_accumulators[i],
                previous_checkpoints[i],
            )
        })
        .collect::<Vec<_>>();
    let (initial_product_root, _) = root_and_path(&leaves, 0);
    let mut active_shards = Vec::new();
    for (record, message) in warp_records.iter().zip(&warp_trace.messages) {
        let i = usize::from(record.shard_ordinal);
        let (_, product_siblings) = root_and_path(&leaves, i);
        let (_, catalog_siblings) = root_and_path(&key_digests, i);
        let replay = WarpCertifiedReplayRecordV19 {
            update_index: record.update_index,
            opening_claim_digest: message.opening_claim_digest,
            fresh_instance_digest: message.fresh_instance_digest,
            segment_openings_digest: message.segment_openings_digest,
            prior_root: message.prior_root,
            fresh_root: message.fresh_root,
            next_root: message.next_root,
            previous_accumulator_digest: message.previous_accumulator_digest,
            next_accumulator_digest: message.next_accumulator_digest,
            authenticated_batching_claim: message.authenticated_batching_claim,
            previous_checkpoint_digest: message.previous_checkpoint_digest,
            next_checkpoint_digest: message.next_checkpoint_digest,
            replay_endpoint_digest: message.replay_endpoint_digest,
            replay_binding_digest: message.replay_binding_digest,
        };
        assert_eq!(
            replay.replay_binding_digest,
            compute_replay_binding_digest_v19(
                7,
                key_digests[i],
                keys[i].relation_digest,
                &replay,
                None,
            )
        );
        leaves[i] = product_leaf(
            key_digests[i],
            replay.next_accumulator_digest,
            replay.next_checkpoint_digest,
        );
        active_shards.push(ShardTransitionRecordV19 {
            key: keys[i].clone(),
            replay,
            product_siblings,
            catalog_siblings,
        });
    }
    let (next_product_root, _) = root_and_path(&leaves, 0);

    let logup_record = LogUpOnlyProducerRecordV19 {
        proof_index: 300,
        segment_index: 7,
        app_vk_digest: app_vk,
        source_forest_root: source_root,
        segment_openings_digest: openings_digest,
        verifier_endpoint: extension(15_000),
        segment_sum_before: [F::ZERO; D_EF],
        segment_sum_after: [F::ZERO; D_EF],
        start_checkpoint: checkpoint(15_100, 90),
        end_checkpoint: checkpoint(15_200, 110),
    };
    let logup_trace =
        generate_logup_only_producer_trace_v19(core::slice::from_ref(&logup_record), 2).unwrap();
    let logup = &logup_trace.messages[0];
    let initial = HistoryBoundaryV19 {
        segment_count: 7,
        vm_state: digest(3000),
        product_state_root: initial_product_root,
        history_root: digest(3100),
        public_values_digest: [F::ZERO; DIGEST_SIZE],
        terminated: false,
    };
    let mut segment = SegmentManifestRecordV19 {
        segment_index: 7,
        initial_pc: F::from_u32(100),
        final_pc: F::from_u32(104),
        exit_code: F::ZERO,
        initial_memory_root: digest(15_500),
        final_memory_root: digest(15_600),
        program_fingerprint: extension(15_700),
        program_fingerprint_digest: digest(15_800),
        program_registry_digest: digest(15_900),
        program_relation_digest: digest(16_000),
        program_log_height: 10,
        program_cached_width: 32,
        from_vm_state: initial.vm_state,
        to_vm_state: digest(16_100),
        source_forest_root: source_root,
        previous_product_state_root: initial_product_root,
        next_product_state_root: next_product_root,
        active_shards,
        logup: LogUpOnlyRecordV19 {
            mode_tag: LOGUP_ONLY_MODE_TAG_V19,
            verifier_endpoint: logup.verifier_endpoint,
            segment_openings_digest: logup.segment_openings_digest,
            checkpoint_digest: logup.checkpoint_digest,
        },
        public_values_digest: digest(17_000),
        terminates: true,
        manifest_digest: [F::ZERO; DIGEST_SIZE],
    };
    segment.manifest_digest = compute_manifest_digest_v19(&segment, None);
    let expected_final = HistoryBoundaryV19 {
        segment_count: 8,
        vm_state: segment.to_vm_state,
        product_state_root: segment.next_product_state_root,
        history_root: poseidon2_compress_with_capacity(
            initial.history_root,
            segment.manifest_digest,
        )
        .0,
        public_values_digest: segment.public_values_digest,
        terminated: true,
    };
    Fixture {
        profile,
        history: HistoryRecordV19 {
            initial,
            segments: vec![segment],
            expected_final,
            terminal_chunk: true,
        },
        warp_records,
        warp_trace,
        logup_record,
        logup_trace,
    }
}

fn refresh_manifest_and_final(record: &mut HistoryRecordV19) {
    let segment = &mut record.segments[0];
    segment.manifest_digest = compute_manifest_digest_v19(segment, None);
    record.expected_final.segment_count = record.initial.segment_count + 1;
    record.expected_final.vm_state = segment.to_vm_state;
    record.expected_final.product_state_root = segment.next_product_state_root;
    record.expected_final.history_root =
        poseidon2_compress_with_capacity(record.initial.history_root, segment.manifest_digest).0;
    record.expected_final.public_values_digest = segment.public_values_digest;
    record.expected_final.terminated = segment.terminates;
}

fn history_air(profile: HistoryVerifierProfileV19) -> HistoryAirV19 {
    HistoryAirV19 {
        profile,
        genesis: None,
        compress_bus: HistoryPoseidon2CompressBusV19::new(COMPRESS_BUS),
        certified_replay_bus: CertifiedWarpReplayBusV19::new(HISTORY_WARP_BUS),
        logup_only_bus: LogUpOnlyHistoryBusV19::new(HISTORY_LOGUP_BUS),
        certified_vm_bus: CertifiedVmSegmentMetadataBusV19::new(VM_METADATA_BUS),
        certified_program_bus: CertifiedProgramFingerprintBusV19::new(PROGRAM_FINGERPRINT_BUS),
        public_values_bus: None,
    }
}

fn history_air_with_genesis(
    profile: HistoryVerifierProfileV19,
    genesis: HistoryGenesisConfigV19,
) -> HistoryAirV19 {
    HistoryAirV19 {
        genesis: Some(genesis),
        ..history_air(profile)
    }
}

/// Recompute every single-segment boundary field affected by the initial
/// state. Keeping the shard list empty makes arbitrary product roots valid
/// continuation inputs without fabricating Merkle authentication paths.
fn recompute_empty_shard_chunk(record: &mut HistoryRecordV19) {
    let segment = &mut record.segments[0];
    segment.segment_index = record.initial.segment_count;
    segment.from_vm_state = record.initial.vm_state;
    segment.previous_product_state_root = record.initial.product_state_root;
    segment.next_product_state_root = record.initial.product_state_root;
    segment.active_shards.clear();
    refresh_manifest_and_final(record);
}

fn stage_zero_fixture() -> Fixture {
    let mut fixture = fixture();
    fixture.history.initial.segment_count = 0;
    fixture.history.segments[0].terminates = false;
    fixture.history.segments[0].exit_code = F::from_u32(DEFAULT_SUSPEND_EXIT_CODE);
    fixture.history.terminal_chunk = false;
    recompute_empty_shard_chunk(&mut fixture.history);
    fixture
}

fn genesis_config(fixture: &Fixture) -> HistoryGenesisConfigV19 {
    HistoryGenesisConfigV19 {
        canonical_empty_product_state_root: fixture.history.initial.product_state_root,
        canonical_initial_history_root: fixture.history.initial.history_root,
    }
}

fn check_air<R>(air: &R, name: &str, matrix: &RowMajorMatrix<F>, public_values: &[F])
where
    R: for<'a> Air<
            openvm_stark_backend::air_builders::debug::DebugConstraintBuilder<
                'a,
                BabyBearPoseidon2Config,
            >,
        > + BaseAir<F>
        + PartitionedBaseAir<F>,
{
    check_constraints::<_, BabyBearPoseidon2Config>(
        air,
        name,
        &None,
        &[matrix.as_view()],
        public_values,
    );
}

#[derive(Clone, Debug)]
struct CertifiedWarpVerifierSourceAirV19 {
    context_bus: DirectAirVaccContextBusV19,
    opening_bus: CertifiedSwirlRawOpeningBusV19,
    vacc_bus: CertifiedDirectAirVaccInputBusV19,
    checkpoint_bus: CertifiedTranscriptCheckpointBus,
    batching_bus: NativeCertifiedBatchingClaimBus,
    accumulator_bus: NativeCertifiedAccumulatorDigestBus,
}

impl BaseAir<F> for CertifiedWarpVerifierSourceAirV19 {
    fn width(&self) -> usize {
        WarpReplayProducerColsV19::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for CertifiedWarpVerifierSourceAirV19 {}
impl PartitionedBaseAir<F> for CertifiedWarpVerifierSourceAirV19 {}

impl<AB> Air<AB> for CertifiedWarpVerifierSourceAirV19
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("WARP verifier source row");
        let local: &WarpReplayProducerColsV19<AB::Var> = (*row).borrow();
        let enabled = local.active;
        let proof = join_limbs::<AB>(local.proof_index_lo, local.proof_index_hi);
        self.context_bus.add_key_with_lookups(
            builder,
            DirectAirVaccContextMessageV19 {
                proof_index: proof.clone(),
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                update_index_lo: local.update_index_lo.into(),
                update_index_hi: local.update_index_hi.into(),
                shard_ordinal: local.shard_ordinal.into(),
                has_prior: local.has_prior.into(),
                key_digest: local.key_digest.map(Into::into),
                relation_digest: local.relation_digest.map(Into::into),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                prior_root: local.prior_root.map(Into::into),
                next_root: local.next_root.map(Into::into),
                previous_accumulator_digest: local.previous_accumulator_digest.map(Into::into),
                previous_checkpoint_digest: local.previous_checkpoint_digest.map(Into::into),
                next_checkpoint_digest: local.next_checkpoint_digest.map(Into::into),
            },
            enabled,
        );
        self.opening_bus.add_key_with_lookups(
            builder,
            CertifiedSwirlRawOpeningMessageV19 {
                proof_index: proof.clone(),
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                mode_tag: AB::Expr::from_u32(LOGUP_ONLY_MODE_TAG_V19),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                shard_ordinal: local.shard_ordinal.into(),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                root: local.fresh_root.map(Into::into),
                point_len: AB::Expr::from_usize(LOG_MESSAGE_LEN),
                point: local.opening_point.map(|point| point.map(Into::into)),
                value: local.opening_value.map(Into::into),
            },
            enabled,
        );
        self.vacc_bus.add_key_with_lookups(
            builder,
            CertifiedDirectAirVaccInputMessageV19 {
                proof_index: proof.clone(),
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                update_index_lo: local.update_index_lo.into(),
                update_index_hi: local.update_index_hi.into(),
                shard_ordinal: local.shard_ordinal.into(),
                relation_digest: local.relation_digest.map(Into::into),
                root: local.fresh_root.map(Into::into),
                alpha_len: AB::Expr::from_usize(LOG_CODEWORD_LEN),
                alpha: local.fresh_alpha.map(|point| point.map(Into::into)),
                mu: local.fresh_mu.map(Into::into),
                beta_len: AB::Expr::from_usize(BETA_LEN),
                beta: local.fresh_beta.map(|point| point.map(Into::into)),
                eta: local.fresh_eta.map(Into::into),
            },
            enabled,
        );
        self.batching_bus.send(
            builder,
            NativeCertifiedBatchingClaimMessage {
                proof_idx: proof.clone(),
                claim: local.authenticated_batching_claim.map(Into::into),
            },
            enabled,
        );
        self.accumulator_bus.send(
            builder,
            NativeCertifiedAccumulatorDigestMessage {
                proof_idx: proof.clone(),
                digest: local.next_accumulator_digest.map(Into::into),
            },
            enabled,
        );
        for (kind, lo, hi, count, state) in [
            (
                0usize,
                local.start_tidx_lo,
                local.start_tidx_hi,
                local.start_sample_count,
                local.start_state,
            ),
            (
                1usize,
                local.end_tidx_lo,
                local.end_tidx_hi,
                local.end_sample_count,
                local.end_state,
            ),
        ] {
            self.checkpoint_bus.send(
                builder,
                proof.clone(),
                CertifiedTranscriptCheckpointMessage {
                    kind: AB::Expr::from_usize(kind),
                    tidx: join_limbs::<AB>(lo, hi),
                    sample_count: count.into(),
                    state: state.map(Into::into),
                },
                enabled,
            );
        }
    }
}

#[derive(Clone, Debug)]
struct CertifiedLogUpVerifierSourceAirV19 {
    endpoint_bus: CertifiedLogUpOnlyEndpointBusV19,
    checkpoint_bus: CertifiedTranscriptCheckpointBus,
}

impl BaseAir<F> for CertifiedLogUpVerifierSourceAirV19 {
    fn width(&self) -> usize {
        LogUpOnlyProducerColsV19::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for CertifiedLogUpVerifierSourceAirV19 {}
impl PartitionedBaseAir<F> for CertifiedLogUpVerifierSourceAirV19 {}

impl<AB> Air<AB> for CertifiedLogUpVerifierSourceAirV19
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("LogUp verifier source row");
        let local: &LogUpOnlyProducerColsV19<AB::Var> = (*row).borrow();
        let enabled = local.active;
        let proof = join_limbs::<AB>(local.proof_index_lo, local.proof_index_hi);
        self.endpoint_bus.add_key_with_lookups(
            builder,
            CertifiedLogUpOnlyEndpointMessageV19 {
                proof_index: proof.clone(),
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                mode_tag: local.mode_tag.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                app_vk_digest: local.app_vk_digest.map(Into::into),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                verifier_endpoint: local.verifier_endpoint.map(Into::into),
                segment_sum_before: local.segment_sum_before.map(Into::into),
                segment_sum_after: local.segment_sum_after.map(Into::into),
            },
            enabled,
        );
        for (kind, lo, hi, count, state) in [
            (
                0usize,
                local.start_tidx_lo,
                local.start_tidx_hi,
                local.start_sample_count,
                local.start_state,
            ),
            (
                1usize,
                local.end_tidx_lo,
                local.end_tidx_hi,
                local.end_sample_count,
                local.end_state,
            ),
        ] {
            self.checkpoint_bus.send(
                builder,
                proof.clone(),
                CertifiedTranscriptCheckpointMessage {
                    kind: AB::Expr::from_usize(kind),
                    tidx: join_limbs::<AB>(lo, hi),
                    sample_count: count.into(),
                    state: state.map(Into::into),
                },
                enabled,
            );
        }
    }
}

fn join_limbs<AB: AirBuilder<F = F>>(lo: AB::Var, hi: AB::Var) -> AB::Expr
where
    AB::Var: Copy,
{
    AB::Expr::from(lo) + AB::Expr::from(hi) * AB::Expr::from_u32(1 << 16)
}

#[derive(Clone, Debug)]
struct CompressionSourceAirV19(HistoryPoseidon2CompressBusV19);

#[derive(Clone, Debug)]
struct CertifiedVmProgramSourceAirV19 {
    app_vk_digest: DigestV19,
    vm_bus: CertifiedVmSegmentMetadataBusV19,
    program_bus: CertifiedProgramFingerprintBusV19,
}

impl BaseAir<F> for CertifiedVmProgramSourceAirV19 {
    fn width(&self) -> usize {
        HistoryRowColsV19::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for CertifiedVmProgramSourceAirV19 {}
impl PartitionedBaseAir<F> for CertifiedVmProgramSourceAirV19 {}

impl<AB> Air<AB> for CertifiedVmProgramSourceAirV19
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("History VM source row");
        let local: &HistoryRowColsV19<AB::Var> = (*row).borrow();
        let enabled = local.is_segment_start;
        self.vm_bus.send(
            builder,
            CertifiedVmSegmentMetadataMessageV19 {
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                segment_index_lo: local.ctx_segment_index_lo.into(),
                segment_index_hi: local.ctx_segment_index_hi.into(),
                initial_pc: local.ctx_initial_pc.into(),
                final_pc: local.ctx_final_pc.into(),
                exit_code: local.ctx_exit_code.into(),
                is_terminate: local.ctx_terminates.into(),
                initial_memory_root: local.ctx_initial_memory_root.map(Into::into),
                final_memory_root: local.ctx_final_memory_root.map(Into::into),
                program_fingerprint: local.ctx_program_fingerprint.map(Into::into),
                program_fingerprint_digest: local.ctx_program_fingerprint_digest.map(Into::into),
                from_vm_state: local.ctx_from_vm.map(Into::into),
                to_vm_state: local.ctx_to_vm.map(Into::into),
            },
            enabled,
        );
        self.program_bus.send(
            builder,
            CertifiedProgramFingerprintMessageV19 {
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                segment_index_lo: local.ctx_segment_index_lo.into(),
                segment_index_hi: local.ctx_segment_index_hi.into(),
                app_vk_digest: self.app_vk_digest.map(Into::into),
                registry_digest: local.ctx_program_registry_digest.map(Into::into),
                relation_digest: local.ctx_program_relation_digest.map(Into::into),
                log_height: local.ctx_program_log_height.into(),
                cached_width: local.ctx_program_cached_width.into(),
                value: local.ctx_program_fingerprint.map(Into::into),
                digest: local.ctx_program_fingerprint_digest.map(Into::into),
            },
            enabled,
        );
    }
}

impl BaseAir<F> for CompressionSourceAirV19 {
    fn width(&self) -> usize {
        1 + 3 * DIGEST_SIZE
    }
}
impl BaseAirWithPublicValues<F> for CompressionSourceAirV19 {}
impl PartitionedBaseAir<F> for CompressionSourceAirV19 {}

impl<AB> Air<AB> for CompressionSourceAirV19
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("compression source row");
        let next = main.row_slice(1).expect("compression source next row");
        let enabled = local[0];
        builder.assert_bool(enabled);
        builder.when_transition().when(next[0]).assert_one(enabled);
        self.0.add_key_with_lookups(
            builder,
            HistoryPoseidon2CompressMessageV19 {
                input: core::array::from_fn(|i| local[1 + i].into()),
                output: core::array::from_fn(|i| local[1 + 2 * DIGEST_SIZE + i].into()),
            },
            enabled,
        );
    }
}

fn compression_trace(inputs: &[[F; 2 * DIGEST_SIZE]]) -> RowMajorMatrix<F> {
    let width = 1 + 3 * DIGEST_SIZE;
    let height = (inputs.len() + 1).next_power_of_two();
    let mut values = F::zero_vec(width * height);
    for (i, input) in inputs.iter().enumerate() {
        let row = &mut values[i * width..(i + 1) * width];
        row[0] = F::ONE;
        row[1..1 + 2 * DIGEST_SIZE].copy_from_slice(input);
        let left = input[..DIGEST_SIZE].try_into().unwrap();
        let right = input[DIGEST_SIZE..].try_into().unwrap();
        row[1 + 2 * DIGEST_SIZE..]
            .copy_from_slice(&poseidon2_compress_with_capacity(left, right).0);
    }
    RowMajorMatrix::new(values, width)
}

fn symbolic_interactions<R>(air: &R) -> Vec<SymbolicInteraction<F>>
where
    R: Air<SymbolicRapBuilder<F>> + BaseAir<F> + BaseAirWithPublicValues<F> + PartitionedBaseAir<F>,
{
    get_symbolic_builder(
        air,
        &TraceWidth {
            preprocessed: None,
            cached_mains: air.cached_main_widths(),
            common_main: air.common_main_width(),
        },
    )
    .constraints()
    .interactions
}

fn check_composed(
    fixture: &Fixture,
    warp_matrix: &RowMajorMatrix<F>,
    logup_matrix: &RowMajorMatrix<F>,
) {
    let history_trace = generate_history_trace_v19(&fixture.profile, &fixture.history).unwrap();
    let history = history_air(fixture.profile.clone());
    let warp = warp_air();
    let logup = logup_air();
    check_air(
        &history,
        "HistoryAirV19",
        &history_trace.matrix,
        &history_trace.public_values,
    );
    check_air(&warp, "WarpReplayProducerAirV19", warp_matrix, &[]);
    check_air(&logup, "LogUpOnlyProducerAirV19", logup_matrix, &[]);
    let warp_source = CertifiedWarpVerifierSourceAirV19 {
        context_bus: DirectAirVaccContextBusV19::new(CONTEXT_BUS),
        opening_bus: CertifiedSwirlRawOpeningBusV19::new(SWIRL_OPENING_BUS),
        vacc_bus: CertifiedDirectAirVaccInputBusV19::new(VACC_INPUT_BUS),
        checkpoint_bus: CertifiedTranscriptCheckpointBus::new(CHECKPOINT_BUS),
        batching_bus: NativeCertifiedBatchingClaimBus::new(BATCHING_BUS),
        accumulator_bus: NativeCertifiedAccumulatorDigestBus::new(NEXT_ACCUMULATOR_BUS),
    };
    let logup_source = CertifiedLogUpVerifierSourceAirV19 {
        endpoint_bus: CertifiedLogUpOnlyEndpointBusV19::new(LOGUP_ENDPOINT_BUS),
        checkpoint_bus: CertifiedTranscriptCheckpointBus::new(CHECKPOINT_BUS),
    };
    let compression_source =
        CompressionSourceAirV19(HistoryPoseidon2CompressBusV19::new(COMPRESS_BUS));
    let vm_program_source = CertifiedVmProgramSourceAirV19 {
        app_vk_digest: fixture.profile.app_vk_digest,
        vm_bus: CertifiedVmSegmentMetadataBusV19::new(VM_METADATA_BUS),
        program_bus: CertifiedProgramFingerprintBusV19::new(PROGRAM_FINGERPRINT_BUS),
    };
    let inputs = history_trace
        .compression_inputs
        .iter()
        .chain(&fixture.warp_trace.compression_inputs)
        .chain(&fixture.logup_trace.compression_inputs)
        .copied()
        .collect::<Vec<_>>();
    let compression = compression_trace(&inputs);
    let interactions = vec![
        symbolic_interactions(&history),
        symbolic_interactions(&warp),
        symbolic_interactions(&logup),
        symbolic_interactions(&warp_source),
        symbolic_interactions(&logup_source),
        symbolic_interactions(&vm_program_source),
        symbolic_interactions(&compression_source),
    ];
    let matrices = [
        vec![history_trace.matrix.as_view()],
        vec![warp_matrix.as_view()],
        vec![logup_matrix.as_view()],
        vec![fixture.warp_trace.matrix.as_view()],
        vec![fixture.logup_trace.matrix.as_view()],
        vec![history_trace.matrix.as_view()],
        vec![compression.as_view()],
    ];
    check_logup(
        &[
            "history".into(),
            "WARP producer".into(),
            "LogUp producer".into(),
            "certified WARP source".into(),
            "certified LogUp source".into(),
            "certified VM/Program source".into(),
            "compression source".into(),
        ],
        &interactions,
        &[None, None, None, None, None, None, None],
        &matrices,
        &[
            history_trace.public_values,
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
            vec![],
        ],
    );
}

#[test]
fn honest_history_and_positive_producers_are_constrained_and_balanced() {
    let fixture = fixture();
    let history = generate_history_trace_v19(&fixture.profile, &fixture.history).unwrap();
    assert_eq!(history.matrix.height(), fixture.profile.trace_height);
    assert_eq!(history.certified_replays, fixture.warp_trace.messages);
    assert_eq!(history.logup_updates, fixture.logup_trace.messages);
    check_composed(
        &fixture,
        &fixture.warp_trace.matrix,
        &fixture.logup_trace.matrix,
    );
}

#[test]
fn systematic_lift_suffix_mu_and_eta_tampering_is_rejected() {
    let fixture = fixture();
    for mutation in 0..4 {
        let mut matrix = fixture.warp_trace.matrix.clone();
        let row: &mut WarpReplayProducerColsV19<F> = matrix.row_mut(0).borrow_mut();
        match mutation {
            0 => row.fresh_alpha[0][0] += F::ONE,
            1 => row.fresh_alpha[LOG_MESSAGE_LEN][0] += F::ONE,
            2 => row.fresh_mu[0] += F::ONE,
            3 => row.fresh_eta[0] += F::ONE,
            _ => unreachable!(),
        }
        assert!(std::panic::catch_unwind(|| {
            check_air(&warp_air(), "WARP producer", &matrix, &[])
        })
        .is_err());
    }
}

#[test]
fn every_warp_certificate_component_is_lookup_bound() {
    let fixture = fixture();
    for mutation in 0..15 {
        let mut matrix = fixture.warp_trace.matrix.clone();
        let row: &mut WarpReplayProducerColsV19<F> = matrix.row_mut(0).borrow_mut();
        match mutation {
            0 => row.relation_digest[0] += F::ONE,
            1 => row.source_forest_root[0] += F::ONE,
            2 => row.segment_index_lo += F::ONE,
            3 => row.update_index_lo += F::ONE,
            4 => row.prior_root[0] += F::ONE,
            5 => row.fresh_root[0] += F::ONE,
            6 => row.next_root[0] += F::ONE,
            7 => row.opening_point[0][0] += F::ONE,
            8 => row.opening_value[0] += F::ONE,
            9 => row.fresh_beta[0][0] += F::ONE,
            10 => row.authenticated_batching_claim[0] += F::ONE,
            11 => row.start_state[0] += F::ONE,
            12 => row.segment_openings_digest[0] += F::ONE,
            13 => row.previous_checkpoint_digest[0] += F::ONE,
            14 => row.next_checkpoint_digest[0] += F::ONE,
            _ => unreachable!(),
        }
        assert!(std::panic::catch_unwind(|| {
            check_composed(&fixture, &matrix, &fixture.logup_trace.matrix)
        })
        .is_err());
    }
}

#[test]
fn product_checkpoints_are_not_transcript_state_digests() {
    let fixture = fixture();
    let row_slice = fixture.warp_trace.matrix.row_slice(0).unwrap();
    let row: &WarpReplayProducerColsV19<F> = row_slice.as_ref().borrow();
    assert_ne!(row.previous_checkpoint_digest, row.transcript_start_digest);
    assert_ne!(row.next_checkpoint_digest, row.transcript_end_digest);
    assert_eq!(
        row.previous_checkpoint_digest,
        fixture.warp_records[0].previous_checkpoint_digest
    );
    assert_eq!(
        row.next_checkpoint_digest,
        fixture.warp_records[0].next_checkpoint_digest
    );
}

#[test]
fn production_v19_sources_exclude_opening_pesat_and_old_vacc_tags() {
    let sources = [
        include_str!("air.rs"),
        include_str!("cuda_shared_forest.rs"),
        include_str!("digest.rs"),
        include_str!("error.rs"),
        include_str!("logup_swirl_composite.rs"),
        include_str!("logup_swirl_manifest.rs"),
        include_str!("logup_swirl_source.rs"),
        include_str!("logup_swirl_verifier.rs"),
        include_str!("mod.rs"),
        include_str!("producer.rs"),
        include_str!("profile.rs"),
        include_str!("record.rs"),
        include_str!("trace.rs"),
        include_str!("vacc_groups.rs"),
        include_str!("vacc_verifier.rs"),
        include_str!("vacc_verifier_cuda.rs"),
    ];
    let forbidden = [
        concat!("opening_", "explicit"),
        concat!("NativeOpening", "ExplicitAir"),
        concat!("mapped_column", "_descriptors"),
        concat!("OpenVmNative", "AirPesat"),
        concat!("NativeWarpOnline", "HistoryBuilder"),
        concat!("openvm-native-warp-vacc-all-ef-", "v3"),
    ];
    for needle in forbidden {
        assert!(
            sources.iter().all(|source| !source.contains(needle)),
            "forbidden v19 production source token: {needle}"
        );
    }

    let crate_root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    for retired in [
        "src/circuit/native_warp_history/mod.rs",
        "src/prover/native_warp_history.rs",
        "../recursion/src/native_warp/fresh/opening_explicit.rs",
        "../recursion/src/native_warp/reduction/mod.rs",
    ] {
        assert!(
            !crate_root.join(retired).is_file(),
            "retired opening-PESAT source is still present: {retired}"
        );
    }
}

#[test]
fn logup_mode_endpoint_checkpoint_and_openings_are_bound() {
    let fixture = fixture();
    for mutation in 0..6 {
        let mut matrix = fixture.logup_trace.matrix.clone();
        let row: &mut LogUpOnlyProducerColsV19<F> = matrix.row_mut(0).borrow_mut();
        match mutation {
            0 => row.mode_tag += F::ONE,
            1 => row.verifier_endpoint[0] += F::ONE,
            2 => row.source_forest_root[0] += F::ONE,
            3 => row.segment_openings_digest[0] += F::ONE,
            4 => row.start_state[0] += F::ONE,
            5 => row.segment_index_lo += F::ONE,
            _ => unreachable!(),
        }
        assert!(std::panic::catch_unwind(|| {
            check_composed(&fixture, &fixture.warp_trace.matrix, &matrix)
        })
        .is_err());
    }
}

#[test]
fn logup_segment_starts_and_ends_at_zero() {
    let fixture = fixture();
    for before in [true, false] {
        let mut record = fixture.logup_record.clone();
        if before {
            record.segment_sum_before[0] = F::ONE;
        } else {
            record.segment_sum_after[0] = F::ONE;
        }
        assert_eq!(
            generate_logup_only_producer_trace_v19(&[record], 2).unwrap_err(),
            PositiveProducerErrorV19::NonZeroLogUpBoundary
        );
    }
}

#[test]
fn malformed_lift_is_rejected_before_trace_generation() {
    let fixture = fixture();
    let mut record = fixture.warp_records[0].clone();
    record.fresh_alpha[LOG_MESSAGE_LEN][0] = F::ONE;
    assert_eq!(
        generate_warp_replay_producer_trace_v19(&warp_air(), &[record], 2).unwrap_err(),
        PositiveProducerErrorV19::RecordShape
    );
}

#[test]
fn warp_replay_producer_supports_max_beta_profile_in_trace_and_symbolic_eval() {
    let mut air = warp_air();
    air.beta_len = MAX_FRESH_BETA_LEN_V19;
    let app_vk = digest(1);
    let shard_key = key(app_vk, 0);
    let key_digest = compute_shard_key_digest_v19(&shard_key, None);
    let mut record = warp_record(&shard_key, key_digest, digest(8000), digest(8100));
    record.fresh_beta = (0..MAX_FRESH_BETA_LEN_V19)
        .map(|index| extension(20_000 + 10 * index as u32))
        .collect();

    let trace = generate_warp_replay_producer_trace_v19(&air, &[record], 2).unwrap();
    check_air(&air, "WarpReplayProducerAirV19", &trace.matrix, &[]);
    assert!(!symbolic_interactions(&air).is_empty());
}

#[test]
fn history_rejects_order_path_replay_mode_and_vm_tampering() {
    let mut reordered = fixture();
    reordered.history.segments[0].active_shards.swap(0, 1);
    assert_eq!(
        generate_history_trace_v19(&reordered.profile, &reordered.history).unwrap_err(),
        HistoryV19Error::NonCanonicalShardOrder {
            segment: 0,
            shard: 1,
        }
    );

    let mut stale_path = fixture();
    stale_path.history.segments[0].active_shards[0].product_siblings[0][0] += F::ONE;
    assert_eq!(
        generate_history_trace_v19(&stale_path.profile, &stale_path.history).unwrap_err(),
        HistoryV19Error::ProductRootMismatch {
            segment: 0,
            shard: Some(0),
        }
    );

    let mut cross_segment = fixture();
    cross_segment.history.initial.segment_count += 1;
    cross_segment.history.segments[0].segment_index += 1;
    refresh_manifest_and_final(&mut cross_segment.history);
    assert_eq!(
        generate_history_trace_v19(&cross_segment.profile, &cross_segment.history).unwrap_err(),
        HistoryV19Error::ReplayBindingMismatch {
            segment: 0,
            shard: 0,
        }
    );

    let mut wrong_mode = fixture();
    wrong_mode.history.segments[0].logup.mode_tag ^= 1;
    assert_eq!(
        generate_history_trace_v19(&wrong_mode.profile, &wrong_mode.history).unwrap_err(),
        HistoryV19Error::LogUpModeMismatch { segment: 0 }
    );

    let mut discontinuous = fixture();
    discontinuous.history.segments[0].from_vm_state[0] += F::ONE;
    assert_eq!(
        generate_history_trace_v19(&discontinuous.profile, &discontinuous.history).unwrap_err(),
        HistoryV19Error::VmBoundaryDiscontinuity { segment: 0 }
    );
}

#[test]
fn history_air_rejects_noncanonical_ordinal_bits() {
    let fixture = fixture();
    let mut trace = generate_history_trace_v19(&fixture.profile, &fixture.history).unwrap();
    let row: &mut HistoryRowColsV19<F> = trace.matrix.row_mut(1).borrow_mut();
    row.shard_ordinal += F::ONE;
    assert!(std::panic::catch_unwind(|| {
        check_air(
            &history_air(fixture.profile.clone()),
            "history",
            &trace.matrix,
            &trace.public_values,
        )
    })
    .is_err());
}

#[test]
fn history_public_values_bind_terminal_vm_and_program_boundary() {
    let fixture = fixture();
    let trace = generate_history_trace_v19(&fixture.profile, &fixture.history).unwrap();
    for mutation in 0..4 {
        let mut public_values = trace.public_values.clone();
        let pvs: &mut super::HistoryPublicValuesV19<F> = public_values.as_mut_slice().borrow_mut();
        match mutation {
            0 => pvs.final_pc += F::ONE,
            1 => pvs.final_exit_code += F::ONE,
            2 => pvs.final_memory_root[0] += F::ONE,
            3 => pvs.program_fingerprint_digest[0] += F::ONE,
            _ => unreachable!(),
        }
        assert!(std::panic::catch_unwind(|| {
            check_air(
                &history_air(fixture.profile.clone()),
                "history terminal boundary",
                &trace.matrix,
                &public_values,
            )
        })
        .is_err());
    }
}

#[test]
fn history_genesis_accepts_canonical_boundary_and_dynamic_certified_vm_state() {
    let mut fixture = stage_zero_fixture();
    let genesis = genesis_config(&fixture);

    // VM state is intentionally absent from the setup-fixed genesis
    // configuration. It remains dynamic, but the segment-start constraints
    // tie it to the exact state in the certified VM metadata statement.
    fixture.history.initial.vm_state = digest(18_000);
    recompute_empty_shard_chunk(&mut fixture.history);
    let mut trace = generate_history_trace_v19(&fixture.profile, &fixture.history).unwrap();
    check_air(
        &history_air_with_genesis(fixture.profile.clone(), genesis.clone()),
        "canonical History genesis",
        &trace.matrix,
        &trace.public_values,
    );

    {
        let first_row = trace.matrix.row_slice(0).unwrap();
        let first: &HistoryRowColsV19<F> = first_row.as_ref().borrow();
        assert_eq!(first.state_before_vm, fixture.history.initial.vm_state);
        assert_eq!(first.ctx_from_vm, fixture.history.initial.vm_state);
    }

    // A different first-segment metadata state cannot be substituted while
    // retaining the canonical public initial VM state.
    let first: &mut HistoryRowColsV19<F> = trace.matrix.row_mut(0).borrow_mut();
    first.ctx_from_vm[0] += F::ONE;
    assert!(std::panic::catch_unwind(|| {
        check_air(
            &history_air_with_genesis(fixture.profile.clone(), genesis),
            "tampered genesis VM metadata",
            &trace.matrix,
            &trace.public_values,
        )
    })
    .is_err());
}

#[test]
fn history_genesis_rejects_recomputed_noncanonical_initial_boundaries() {
    let canonical = stage_zero_fixture();
    let genesis = genesis_config(&canonical);

    let mut cases = Vec::new();

    let mut count = stage_zero_fixture();
    count.history.initial.segment_count = 9;
    recompute_empty_shard_chunk(&mut count.history);
    cases.push(("initial segment count", count));

    let mut product = stage_zero_fixture();
    product.history.initial.product_state_root[0] += F::ONE;
    recompute_empty_shard_chunk(&mut product.history);
    cases.push(("initial product root", product));

    let mut history = stage_zero_fixture();
    history.history.initial.history_root[0] += F::ONE;
    recompute_empty_shard_chunk(&mut history.history);
    cases.push(("initial history root", history));

    let mut public_values = stage_zero_fixture();
    public_values.history.initial.public_values_digest[0] += F::ONE;
    recompute_empty_shard_chunk(&mut public_values.history);
    cases.push(("initial public-values digest", public_values));

    for (name, fixture) in cases {
        // These are fully recomputed, valid continuation statements. Their
        // rejection below therefore exercises only the setup-fixed genesis
        // boundary, not a stale manifest or transition witness.
        let trace = generate_history_trace_v19(&fixture.profile, &fixture.history).unwrap();
        check_air(
            &history_air(fixture.profile.clone()),
            name,
            &trace.matrix,
            &trace.public_values,
        );
        assert!(std::panic::catch_unwind(|| {
            check_air(
                &history_air_with_genesis(fixture.profile.clone(), genesis.clone()),
                name,
                &trace.matrix,
                &trace.public_values,
            )
        })
        .is_err());
    }
}

#[test]
fn history_genesis_rejects_initial_termination() {
    let fixture = stage_zero_fixture();
    let genesis = genesis_config(&fixture);

    // No valid continuation can start after termination: the record oracle
    // rejects this before witness generation, independently of genesis mode.
    let mut terminated_record = fixture.history.clone();
    terminated_record.initial.terminated = true;
    recompute_empty_shard_chunk(&mut terminated_record);
    assert_eq!(
        generate_history_trace_v19(&fixture.profile, &terminated_record).unwrap_err(),
        HistoryV19Error::SegmentAfterTermination { segment: 0 }
    );

    // Also exercise the setup-fixed first-row constraint directly. Propagate
    // the tampered bit through the pre-end rows so this is not merely a stale
    // public-value mutation.
    let mut trace = generate_history_trace_v19(&fixture.profile, &fixture.history).unwrap();
    let public: &mut HistoryPublicValuesV19<F> = trace.public_values.as_mut_slice().borrow_mut();
    public.initial_terminated = F::ONE;
    for row_index in 0..trace.active_rows {
        let row: &mut HistoryRowColsV19<F> = trace.matrix.row_mut(row_index).borrow_mut();
        row.state_before_terminated = F::ONE;
        if row.is_segment_end == F::ONE {
            break;
        }
        row.state_after_terminated = F::ONE;
    }
    assert!(std::panic::catch_unwind(|| {
        check_air(
            &history_air_with_genesis(fixture.profile.clone(), genesis),
            "terminated History genesis",
            &trace.matrix,
            &trace.public_values,
        )
    })
    .is_err());
}

#[test]
fn non_genesis_history_chunks_remain_chainable() {
    let first = stage_zero_fixture();
    let genesis = genesis_config(&first);
    let first_trace = generate_history_trace_v19(&first.profile, &first.history).unwrap();
    check_air(
        &history_air_with_genesis(first.profile.clone(), genesis.clone()),
        "first History chunk",
        &first_trace.matrix,
        &first_trace.public_values,
    );

    let mut second = stage_zero_fixture();
    second.history.initial = first.history.expected_final.clone();
    second.history.segments[0].to_vm_state = digest(18_100);
    second.history.segments[0].public_values_digest = digest(18_200);
    recompute_empty_shard_chunk(&mut second.history);
    assert_eq!(first.history.expected_final, second.history.initial);

    let second_trace = generate_history_trace_v19(&second.profile, &second.history).unwrap();
    check_air(
        &history_air(second.profile.clone()),
        "non-genesis History chunk",
        &second_trace.matrix,
        &second_trace.public_values,
    );

    let first_public: &HistoryPublicValuesV19<F> = first_trace.public_values.as_slice().borrow();
    let second_public: &HistoryPublicValuesV19<F> = second_trace.public_values.as_slice().borrow();
    assert_eq!(
        first_public.final_segment_count_lo,
        second_public.initial_segment_count_lo
    );
    assert_eq!(
        first_public.final_segment_count_hi,
        second_public.initial_segment_count_hi
    );
    assert_eq!(first_public.final_vm_state, second_public.initial_vm_state);
    assert_eq!(
        first_public.final_product_state_root,
        second_public.initial_product_state_root
    );
    assert_eq!(
        first_public.final_history_root,
        second_public.initial_history_root
    );
    assert_eq!(
        first_public.final_public_values_digest,
        second_public.initial_public_values_digest
    );
    assert_eq!(
        first_public.final_terminated,
        second_public.initial_terminated
    );

    // A continuation is accepted by the non-genesis key but cannot be
    // relabelled as stage zero.
    assert!(std::panic::catch_unwind(|| {
        check_air(
            &history_air_with_genesis(second.profile.clone(), genesis),
            "continuation relabelled as genesis",
            &second_trace.matrix,
            &second_trace.public_values,
        )
    })
    .is_err());
}

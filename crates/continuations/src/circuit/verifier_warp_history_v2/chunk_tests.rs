use core::borrow::{Borrow, BorrowMut};
use std::{panic::AssertUnwindSafe, sync::Arc};

use openvm_circuit::system::connector::DEFAULT_SUSPEND_EXIT_CODE;
use openvm_cpu_backend::CpuBackend;
use openvm_recursion_circuit::{
    system::{BusIndexManager, BusInventory},
    transcript::{Poseidon2BusOwner, TranscriptModule},
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::{
        debug::{check_constraints, check_logup},
        symbolic::get_symbolic_builder,
    },
    interaction::{InteractionBuilder, SymbolicInteraction},
    keygen::types::TraceWidth,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::AirProvingContext,
    AirRef, AnyAir, BaseAirWithPublicValues, PartitionedBaseAir, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as NativeSC, Digest, DIGEST_SIZE, F,
};

use super::*;

const TRANSITION_COUNT: usize = 4;
const SPLIT_AT: usize = 2;

fn digest(seed: u32) -> Digest {
    core::array::from_fn(|limb| F::from_u32(seed + limb as u32))
}

fn state(index: usize) -> VerifierWarpVmStateV2 {
    VerifierWarpVmStateV2 {
        pc: F::from_usize(100 + index),
        memory_root: digest(1_000 + 100 * index as u32),
    }
}

fn transition(index: usize) -> VerifierWarpHistoryTransitionRecordV2 {
    let terminal = index + 1 == TRANSITION_COUNT;
    let child = VerifierWarpChildRecordV2 {
        occupied: true,
        input: state(index),
        output: state(index + 1),
        exit_code: if terminal {
            F::ZERO
        } else {
            F::from_u32(DEFAULT_SUSPEND_EXIT_CODE)
        },
        terminates: terminal,
    };
    VerifierWarpHistoryTransitionRecordV2 {
        protocol_version: VERIFIER_WARP_HISTORY_PROTOCOL_V2,
        vacc_input_arity: VERIFIER_WARP_VACC_INPUT_ARITY_V2 as u8,
        protocol_digest: digest(10_000),
        relation_digest: digest(11_000),
        batch_index: index as u32,
        active_child_count: if terminal {
            2
        } else {
            VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2 as u8
        },
        children: {
            let mut children =
                [VerifierWarpChildRecordV2::padding(); VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2];
            children[0] = child;
            children
        },
        program_commitment: digest(12_000),
        prior_accumulator_digest: digest(20_000 + 100 * index as u32),
        source_accumulator_digest: digest(30_000 + 100 * index as u32),
        output_accumulator_digest: digest(20_000 + 100 * (index + 1) as u32),
        source_commitment_root: digest(40_000 + 100 * index as u32),
        external_logup_gkr_digest: digest(50_000 + 100 * index as u32),
        source_functional_digest: digest(60_000 + 100 * index as u32),
        setup_openings_digest: digest(70_000 + 100 * index as u32),
        source_statement_digest: digest(80_000 + 100 * index as u32),
        transition_transcript_digest: digest(90_000 + 100 * index as u32),
        public_values_digest: if terminal {
            digest(100_000)
        } else {
            [F::ZERO; DIGEST_SIZE]
        },
    }
}

fn honest_records() -> Vec<VerifierWarpHistoryTransitionRecordV2> {
    (0..TRANSITION_COUNT).map(transition).collect()
}

fn genesis_context(end_is_terminal: bool) -> VerifierWarpHistoryChunkContextV3 {
    VerifierWarpHistoryChunkContextV3 {
        total_batch_count: TRANSITION_COUNT as u32,
        chunk_index: 0,
        start_batch_index: 0,
        active_child_count_before: 0,
        observation_count_before: 0,
        history_hash_before: verifier_warp_history_initial_hash_v2(),
        is_genesis: true,
        is_terminal: end_is_terminal,
    }
}

fn next_context(
    previous: &VerifierWarpHistoryChunkPublicValuesV3,
    is_terminal: bool,
) -> VerifierWarpHistoryChunkContextV3 {
    VerifierWarpHistoryChunkContextV3 {
        total_batch_count: previous.total_batch_count,
        chunk_index: previous.chunk_index + 1,
        start_batch_index: previous.end_batch_index,
        active_child_count_before: previous.active_child_count_after,
        observation_count_before: previous.observation_count_after,
        history_hash_before: previous.history_hash_after,
        is_genesis: false,
        is_terminal,
    }
}

fn honest_chunks() -> (
    Vec<VerifierWarpHistoryTransitionRecordV2>,
    VerifierWarpHistoryChunkTraceV3,
    VerifierWarpHistoryChunkTraceV3,
) {
    let records = honest_records();
    let first =
        generate_verifier_warp_history_chunk_trace_v3(&records[..SPLIT_AT], genesis_context(false))
            .unwrap();
    let second = generate_verifier_warp_history_chunk_trace_v3(
        &records[SPLIT_AT..],
        next_context(&first.public_values, true),
    )
    .unwrap();
    (records, first, second)
}

/// Reference predicate for the exact equalities the recursive composition AIR
/// must impose. A standalone chunk cannot enforce these equalities because its
/// predecessor is intentionally outside that proof.
fn chunks_chain(
    left: &VerifierWarpHistoryChunkPublicValuesV3,
    right: &VerifierWarpHistoryChunkPublicValuesV3,
) -> bool {
    left.chunk_protocol_version == right.chunk_protocol_version
        && left.source_history_protocol_version == right.source_history_protocol_version
        && left.source_child_capacity == right.source_child_capacity
        && left.vacc_input_arity == right.vacc_input_arity
        && left.protocol_digest == right.protocol_digest
        && left.relation_digest == right.relation_digest
        && left.program_commitment == right.program_commitment
        && left.total_batch_count == right.total_batch_count
        && left.chunk_index.checked_add(1) == Some(right.chunk_index)
        && left.end_batch_index == right.start_batch_index
        && left.active_child_count_after == right.active_child_count_before
        && left.final_state == right.initial_state
        && left.final_accumulator_digest == right.initial_accumulator_digest
        && left.history_hash_after == right.history_hash_before
        && left.observation_count_after == right.observation_count_before
        && left.is_genesis
        && !left.is_terminal
        && !right.is_genesis
        && right.is_terminal
        && left.public_values_digest == [F::ZERO; DIGEST_SIZE]
}

fn child_message(child: VerifierWarpChildRecordV2) -> VerifierWarpCertifiedChildMessageV2<F> {
    VerifierWarpCertifiedChildMessageV2 {
        occupied: F::from_bool(child.occupied),
        input_pc: child.input.pc,
        input_memory_root: child.input.memory_root,
        output_pc: child.output.pc,
        output_memory_root: child.output.memory_root,
        exit_code: child.exit_code,
        terminates: F::from_bool(child.terminates),
    }
}

fn source_message(
    record: &VerifierWarpHistoryTransitionRecordV2,
) -> VerifierWarpSourceCertificateMessageV2<F> {
    VerifierWarpSourceCertificateMessageV2 {
        protocol_version: F::from_u32(record.protocol_version),
        source_child_capacity: F::from_usize(VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2),
        batch_index_lo: F::from_u16(record.batch_index as u16),
        batch_index_hi: F::from_u16((record.batch_index >> 16) as u16),
        active_child_count: F::from_u8(record.active_child_count),
        protocol_digest: record.protocol_digest,
        relation_digest: record.relation_digest,
        program_commitment: record.program_commitment,
        children: record.children.map(child_message),
        source_accumulator_digest: record.source_accumulator_digest,
        source_commitment_root: record.source_commitment_root,
        external_logup_gkr_digest: record.external_logup_gkr_digest,
        source_functional_digest: record.source_functional_digest,
        setup_openings_digest: record.setup_openings_digest,
        source_statement_digest: record.source_statement_digest,
    }
}

fn vacc_message(
    record: &VerifierWarpHistoryTransitionRecordV2,
) -> VerifierWarpVaccCertificateMessageV2<F> {
    VerifierWarpVaccCertificateMessageV2 {
        protocol_version: F::from_u32(record.protocol_version),
        input_arity: F::from_u8(record.vacc_input_arity),
        batch_index_lo: F::from_u16(record.batch_index as u16),
        batch_index_hi: F::from_u16((record.batch_index >> 16) as u16),
        relation_digest: record.relation_digest,
        prior_accumulator_digest: record.prior_accumulator_digest,
        source_accumulator_digest: record.source_accumulator_digest,
        output_accumulator_digest: record.output_accumulator_digest,
        source_commitment_root: record.source_commitment_root,
        transition_transcript_digest: record.transition_transcript_digest,
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct ChunkAuthorityCols<T> {
    active: T,
    source: VerifierWarpSourceCertificateMessageV2<T>,
    vacc: VerifierWarpVaccCertificateMessageV2<T>,
}

#[derive(Clone, Copy)]
struct ChunkAuthorityAir {
    source_bus: VerifierWarpSourceCertificateBusV2,
    vacc_bus: VerifierWarpVaccCertificateBusV2,
}

impl BaseAir<F> for ChunkAuthorityAir {
    fn width(&self) -> usize {
        core::mem::size_of::<ChunkAuthorityCols<u8>>()
    }
}

impl BaseAirWithPublicValues<F> for ChunkAuthorityAir {}
impl PartitionedBaseAir<F> for ChunkAuthorityAir {}

impl<AB> Air<AB> for ChunkAuthorityAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("chunk authority row");
        let local: &ChunkAuthorityCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        self.source_bus
            .add_key_with_lookups(builder, local.source.clone(), local.active);
        self.vacc_bus
            .add_key_with_lookups(builder, local.vacc.clone(), local.active);
    }
}

fn authority_trace(records: &[VerifierWarpHistoryTransitionRecordV2]) -> RowMajorMatrix<F> {
    let width = core::mem::size_of::<ChunkAuthorityCols<u8>>();
    let height = records.len().next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    for (index, record) in records.iter().enumerate() {
        let row = &mut values[index * width..(index + 1) * width];
        let cols: &mut ChunkAuthorityCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.source = source_message(record);
        cols.vacc = vacc_message(record);
    }
    RowMajorMatrix::new(values, width)
}

fn symbolic_interactions(air: &dyn AnyAir<NativeSC>) -> Vec<SymbolicInteraction<F>> {
    let preprocessed = BaseAir::<F>::preprocessed_trace(air).map(|trace| trace.width());
    get_symbolic_builder(
        air,
        &TraceWidth {
            preprocessed,
            cached_mains: air.cached_main_widths(),
            common_main: air.common_main_width(),
        },
    )
    .constraints()
    .interactions
}

fn check_chunk_air_and_buses(
    records: &[VerifierWarpHistoryTransitionRecordV2],
    context: VerifierWarpHistoryChunkContextV3,
) {
    let mut manager = BusIndexManager::new();
    let inventory = BusInventory::new(&mut manager);
    let transcript = TranscriptModule::<1>::new(
        inventory.clone(),
        SystemParams::new_for_testing(10),
        false,
        false,
    );
    let poseidon_owner = transcript.poseidon2_bus_owner();
    let source_bus = VerifierWarpSourceCertificateBusV2::new(manager.new_bus_idx());
    let vacc_bus = VerifierWarpVaccCertificateBusV2::new(manager.new_bus_idx());
    let block_bus = VerifierWarpBlockPublicValuesBusV2::new(manager.new_bus_idx());
    let history_air = VerifierWarpHistoryAirV2 {
        source_bus,
        vacc_bus,
        block_public_values_bus: block_bus,
        compress_bus: inventory.poseidon2_compress_bus,
        output_mode: VerifierWarpHistoryOutputModeV2::ChunkPublicValuesV3,
    };
    let authority_air = ChunkAuthorityAir {
        source_bus,
        vacc_bus,
    };
    let trace = generate_verifier_warp_history_chunk_trace_v3(records, context).unwrap();
    let authority = authority_trace(records);
    let poseidon = transcript
        .build_poseidon2_multibus_traces(vec![(Vec::new(), trace.compression_inputs.clone())])
        .unwrap()
        .pop()
        .unwrap();
    let poseidon_air =
        transcript.multi_bus_poseidon2_air_for_owners::<NativeSC>(&[Poseidon2BusOwner {
            permute_bus: poseidon_owner.permute_bus,
            compress_bus: poseidon_owner.compress_bus,
        }]);

    let airs: Vec<AirRef<NativeSC>> =
        vec![Arc::new(history_air), Arc::new(authority_air), poseidon_air];
    let contexts: Vec<AirProvingContext<CpuBackend<NativeSC>>> = vec![
        AirProvingContext::new(Vec::new(), trace.matrix, trace.public_values.to_vec()),
        AirProvingContext::simple_no_pis(authority),
        AirProvingContext::simple_no_pis(poseidon),
    ];
    for (air, proving_context) in airs.iter().zip(&contexts) {
        let preprocessed_owned = BaseAir::<F>::preprocessed_trace(air.as_ref());
        let preprocessed = preprocessed_owned.as_ref().map(RowMajorMatrix::as_view);
        check_constraints::<_, NativeSC>(
            air.as_ref(),
            &air.name(),
            &preprocessed,
            &[proving_context.common_main.as_view()],
            &proving_context.public_values,
        );
    }

    let preprocessed_owned = airs
        .iter()
        .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
        .collect::<Vec<_>>();
    let preprocessed = preprocessed_owned
        .iter()
        .map(|trace| trace.as_ref().map(RowMajorMatrix::as_view))
        .collect::<Vec<_>>();
    let interactions = airs
        .iter()
        .map(|air| symbolic_interactions(air.as_ref()))
        .collect::<Vec<_>>();
    let mains = contexts
        .iter()
        .map(|context| vec![context.common_main.as_view()])
        .collect::<Vec<_>>();
    let names = airs.iter().map(|air| air.name()).collect::<Vec<_>>();
    let public_values = contexts
        .iter()
        .map(|context| context.public_values.clone())
        .collect::<Vec<_>>();
    check_logup(&names, &interactions, &preprocessed, &mains, &public_values);
}

fn public_values_mutation_is_rejected(
    trace: &VerifierWarpHistoryChunkTraceV3,
    public_index: usize,
) -> bool {
    let mut manager = BusIndexManager::new();
    let inventory = BusInventory::new(&mut manager);
    let air = VerifierWarpHistoryAirV2 {
        source_bus: VerifierWarpSourceCertificateBusV2::new(manager.new_bus_idx()),
        vacc_bus: VerifierWarpVaccCertificateBusV2::new(manager.new_bus_idx()),
        block_public_values_bus: VerifierWarpBlockPublicValuesBusV2::new(manager.new_bus_idx()),
        compress_bus: inventory.poseidon2_compress_bus,
        output_mode: VerifierWarpHistoryOutputModeV2::ChunkPublicValuesV3,
    };
    let mut public_values = trace.public_values.to_vec();
    public_values[public_index] += F::ONE;
    std::panic::catch_unwind(AssertUnwindSafe(|| {
        check_constraints::<_, NativeSC>(
            &air,
            "mutated chunk-v3 public values",
            &None,
            &[trace.matrix.as_view()],
            &public_values,
        );
    }))
    .is_err()
}

fn assert_chunk_error(
    result: Result<VerifierWarpHistoryChunkTraceV3, VerifierWarpHistoryTraceErrorV2>,
    expected: VerifierWarpHistoryTraceErrorV2,
) {
    match result {
        Err(error) => assert_eq!(error, expected),
        Ok(_) => panic!("malformed History chunk was accepted"),
    }
}

#[test]
fn honest_chunks_chain_and_match_monolithic_v2() {
    let (records, first, second) = honest_chunks();
    assert!(chunks_chain(&first.public_values, &second.public_values));
    assert_eq!(
        next_context(&first.public_values, true),
        VerifierWarpHistoryChunkContextV3 {
            total_batch_count: second.public_values.total_batch_count,
            chunk_index: second.public_values.chunk_index,
            start_batch_index: second.public_values.start_batch_index,
            active_child_count_before: second.public_values.active_child_count_before,
            observation_count_before: second.public_values.observation_count_before,
            history_hash_before: second.public_values.history_hash_before,
            is_genesis: second.public_values.is_genesis,
            is_terminal: second.public_values.is_terminal,
        }
    );

    let monolithic = generate_verifier_warp_history_trace_v2(&records).unwrap();
    assert_eq!(
        second.public_values.history_hash_after,
        monolithic.public_values.history_statement_digest
    );
    assert_eq!(
        second.public_values.active_child_count_after,
        monolithic.public_values.active_child_count
    );
    assert_eq!(
        second.public_values.total_batch_count,
        monolithic.public_values.batch_count
    );
    assert_eq!(
        first.public_values.initial_state,
        monolithic.public_values.initial_state
    );
    assert_eq!(
        second.public_values.final_state,
        monolithic.public_values.final_state
    );
    assert_eq!(
        first.public_values.initial_accumulator_digest,
        monolithic.public_values.genesis_accumulator_digest
    );
    assert_eq!(
        second.public_values.final_accumulator_digest,
        monolithic.public_values.final_accumulator_digest
    );
    assert_eq!(
        second.public_values.public_values_digest,
        monolithic.public_values.public_values_digest
    );

    let monolithic_last = monolithic.matrix.row_slice(TRANSITION_COUNT - 1).unwrap();
    let monolithic_cols: &VerifierWarpHistoryRowColsV2<F> = (*monolithic_last).borrow();
    assert_eq!(
        monolithic_cols.observation_count_after,
        [
            F::from_u16(second.public_values.observation_count_after as u16),
            F::from_u16((second.public_values.observation_count_after >> 16) as u16),
        ]
    );
    assert_eq!(
        second.public_values.observation_count_after as usize,
        HISTORY_HEADER_OBSERVATION_COUNT_V2 + TRANSITION_COUNT * MANIFEST_OBSERVATION_COUNT_V2 + 1
    );
}

#[test]
fn genesis_chunk_public_values_v3_pass_plain_air_and_all_bus_checks() {
    let records = honest_records();
    check_chunk_air_and_buses(&records[..SPLIT_AT], genesis_context(false));
}

#[test]
fn terminal_chunk_public_values_v3_pass_plain_air_and_all_bus_checks() {
    let (records, first, _) = honest_chunks();
    check_chunk_air_and_buses(
        &records[SPLIT_AT..],
        next_context(&first.public_values, true),
    );
}

#[test]
fn chunk_air_binds_chaining_public_values() {
    let (_, first, second) = honest_chunks();
    // Canonical v3 PV offsets: start index, incoming active counter,
    // incoming History hash, incoming observation counter, and both flags.
    for index in [32, 36, 78, 94, 98, 99] {
        assert!(
            public_values_mutation_is_rejected(&second, index),
            "chunk AIR accepted mutated public value {index}"
        );
    }
    // The genesis chunk binds the same coordinates, including its zero/initial
    // incoming context.
    for index in [32, 36, 78, 94, 98, 99] {
        assert!(
            public_values_mutation_is_rejected(&first, index),
            "genesis chunk AIR accepted mutated public value {index}"
        );
    }
}

#[test]
fn malformed_chunk_contexts_and_local_receipt_order_are_rejected() {
    let records = honest_records();

    let mut bad_start = next_context(
        &generate_verifier_warp_history_chunk_trace_v3(
            &records[..SPLIT_AT],
            genesis_context(false),
        )
        .unwrap()
        .public_values,
        true,
    );
    bad_start.start_batch_index -= 1;
    assert_chunk_error(
        generate_verifier_warp_history_chunk_trace_v3(&records[SPLIT_AT..], bad_start),
        VerifierWarpHistoryTraceErrorV2::ChunkContext,
    );

    assert_chunk_error(
        generate_verifier_warp_history_chunk_trace_v3(&records[..SPLIT_AT], genesis_context(true)),
        VerifierWarpHistoryTraceErrorV2::ChunkContext,
    );
    let (_, first, _) = honest_chunks();
    assert_chunk_error(
        generate_verifier_warp_history_chunk_trace_v3(
            &records[SPLIT_AT..],
            next_context(&first.public_values, false),
        ),
        VerifierWarpHistoryTraceErrorV2::ChunkContext,
    );

    let mut premature = records[..SPLIT_AT].to_vec();
    premature[1].children[0].terminates = true;
    premature[1].children[0].exit_code = F::ZERO;
    assert_chunk_error(
        generate_verifier_warp_history_chunk_trace_v3(&premature, genesis_context(false)),
        VerifierWarpHistoryTraceErrorV2::PrematureFinal(1),
    );
    let mut missing_terminal = records[SPLIT_AT..].to_vec();
    missing_terminal[1].children[0].terminates = false;
    missing_terminal[1].children[0].exit_code = F::from_u32(DEFAULT_SUSPEND_EXIT_CODE);
    assert_chunk_error(
        generate_verifier_warp_history_chunk_trace_v3(
            &missing_terminal,
            next_context(&first.public_values, true),
        ),
        VerifierWarpHistoryTraceErrorV2::MissingTermination,
    );

    let mut reordered = records[..SPLIT_AT].to_vec();
    reordered.swap(0, 1);
    assert!(matches!(
        generate_verifier_warp_history_chunk_trace_v3(&reordered, genesis_context(false)),
        Err(VerifierWarpHistoryTraceErrorV2::BatchIndex(_))
    ));
    let duplicated = vec![records[0].clone(), records[0].clone()];
    assert_chunk_error(
        generate_verifier_warp_history_chunk_trace_v3(&duplicated, genesis_context(false)),
        VerifierWarpHistoryTraceErrorV2::BatchIndex(1),
    );
    let dropped = vec![records[0].clone(), records[2].clone()];
    assert_chunk_error(
        generate_verifier_warp_history_chunk_trace_v3(&dropped, genesis_context(false)),
        VerifierWarpHistoryTraceErrorV2::BatchIndex(1),
    );
}

#[test]
fn cross_chunk_mutations_gaps_overlaps_and_reordering_are_rejected() {
    let (records, first, second) = honest_chunks();
    let left = first.public_values;
    let honest_right_context = next_context(&left, true);

    for mutate in [
        |context: &mut VerifierWarpHistoryChunkContextV3| context.chunk_index += 1,
        |context: &mut VerifierWarpHistoryChunkContextV3| context.active_child_count_before += 1,
        |context: &mut VerifierWarpHistoryChunkContextV3| context.observation_count_before += 1,
        |context: &mut VerifierWarpHistoryChunkContextV3| context.history_hash_before[0] += F::ONE,
    ] {
        let mut context = honest_right_context;
        mutate(&mut context);
        let mutated =
            generate_verifier_warp_history_chunk_trace_v3(&records[SPLIT_AT..], context).unwrap();
        assert!(!chunks_chain(&left, &mutated.public_values));
    }

    let mut boundary_state = records[SPLIT_AT..].to_vec();
    boundary_state[0].children[0].input.pc += F::ONE;
    let boundary_state =
        generate_verifier_warp_history_chunk_trace_v3(&boundary_state, honest_right_context)
            .unwrap();
    assert!(!chunks_chain(&left, &boundary_state.public_values));

    let mut boundary_accumulator = records[SPLIT_AT..].to_vec();
    boundary_accumulator[0].prior_accumulator_digest[0] += F::ONE;
    let boundary_accumulator =
        generate_verifier_warp_history_chunk_trace_v3(&boundary_accumulator, honest_right_context)
            .unwrap();
    assert!(!chunks_chain(&left, &boundary_accumulator.public_values));

    // Gap: both chunks are locally valid, but transition 1 is absent between
    // their public intervals.
    let short_left =
        generate_verifier_warp_history_chunk_trace_v3(&records[..1], genesis_context(false))
            .unwrap();
    let honest_prefix =
        generate_verifier_warp_history_chunk_trace_v3(&records[..SPLIT_AT], genesis_context(false))
            .unwrap();
    let gap_right = generate_verifier_warp_history_chunk_trace_v3(
        &records[SPLIT_AT..],
        next_context(&honest_prefix.public_values, true),
    )
    .unwrap();
    assert!(!chunks_chain(
        &short_left.public_values,
        &gap_right.public_values
    ));

    // Overlap: transition 1 appears in both locally valid intervals.
    let one_record_prefix =
        generate_verifier_warp_history_chunk_trace_v3(&records[..1], genesis_context(false))
            .unwrap();
    let overlap_right = generate_verifier_warp_history_chunk_trace_v3(
        &records[1..],
        next_context(&one_record_prefix.public_values, true),
    )
    .unwrap();
    assert!(!chunks_chain(&left, &overlap_right.public_values));

    assert!(!chunks_chain(&second.public_values, &first.public_values));
}

use core::borrow::{Borrow, BorrowMut};
use std::{panic::AssertUnwindSafe, sync::Arc};

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
    AirRef, AnyAir, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as NativeSC, Digest, DIGEST_SIZE, F,
};

use super::*;
use crate::circuit::verifier_warp_history_v2::{
    VerifierWarpHistoryChunkPublicValuesV3, VerifierWarpVmStateV2,
};

fn digest(seed: u32) -> Digest {
    core::array::from_fn(|index| F::from_u32(seed + index as u32))
}

fn adjacent_record() -> VerifierWarpHistoryChunkCompositionRecordV3 {
    let protocol_digest = digest(100);
    let relation_digest = digest(200);
    let program_commitment = digest(300);
    let middle_state = VerifierWarpVmStateV2 {
        pc: F::from_u32(17),
        memory_root: digest(500),
    };
    let middle_accumulator = digest(700);
    let middle_history = digest(900);
    let left = VerifierWarpHistoryChunkPublicValuesV3 {
        chunk_protocol_version: 3,
        source_history_protocol_version: 2,
        source_child_capacity: 4,
        vacc_input_arity: 2,
        protocol_digest,
        relation_digest,
        program_commitment,
        total_batch_count: 8,
        chunk_index: 0,
        start_batch_index: 0,
        end_batch_index: 4,
        active_child_count_before: 0,
        active_child_count_after: 13,
        initial_state: VerifierWarpVmStateV2 {
            pc: F::from_u32(11),
            memory_root: digest(400),
        },
        final_state: middle_state,
        initial_accumulator_digest: digest(600),
        final_accumulator_digest: middle_accumulator,
        history_hash_before: digest(800),
        history_hash_after: middle_history,
        observation_count_before: 0,
        observation_count_after: 331,
        is_genesis: true,
        is_terminal: false,
        public_values_digest: [F::ZERO; DIGEST_SIZE],
    };
    let right = VerifierWarpHistoryChunkPublicValuesV3 {
        chunk_index: 1,
        start_batch_index: left.end_batch_index,
        end_batch_index: left.total_batch_count,
        active_child_count_before: left.active_child_count_after,
        active_child_count_after: 29,
        initial_state: middle_state,
        final_state: VerifierWarpVmStateV2 {
            pc: F::from_u32(23),
            memory_root: digest(501),
        },
        initial_accumulator_digest: middle_accumulator,
        final_accumulator_digest: digest(701),
        history_hash_before: middle_history,
        history_hash_after: digest(901),
        observation_count_before: left.observation_count_after,
        observation_count_after: 669,
        is_genesis: false,
        is_terminal: true,
        public_values_digest: digest(1_000),
        ..left
    };
    VerifierWarpHistoryChunkCompositionRecordV3 { left, right }
}

fn standalone_air() -> VerifierWarpHistoryChunkCompositionAirV3 {
    VerifierWarpHistoryChunkCompositionAirV3::new_standalone(
        VerifierWarpHistoryChunkIntervalBusV3::new(10),
        VerifierWarpHistoryChunkIntervalBusV3::new(11),
    )
    .unwrap()
}

fn check_plain(
    air: &VerifierWarpHistoryChunkCompositionAirV3,
    matrix: &RowMajorMatrix<F>,
    public_values: &[F],
) {
    check_constraints::<_, NativeSC>(
        air,
        "History-v3 two-child interval composition",
        &None,
        &[matrix.as_view()],
        public_values,
    );
}

fn assert_plain_rejects(
    air: &VerifierWarpHistoryChunkCompositionAirV3,
    matrix: &RowMajorMatrix<F>,
    public_values: &[F],
    label: &str,
) {
    assert!(
        std::panic::catch_unwind(AssertUnwindSafe(|| {
            check_plain(air, matrix, public_values)
        }))
        .is_err(),
        "plain constraints accepted {label}"
    );
}

fn mutate_child(matrix: &mut RowMajorMatrix<F>, right: bool, index: usize, delta: F) {
    let width = matrix.width();
    let local: &mut VerifierWarpHistoryChunkCompositionColsV3<F> =
        matrix.values[..width].borrow_mut();
    let message = if right {
        &mut local.right
    } else {
        &mut local.left
    };
    message.fields[index] += delta;
}

fn set_chunk_index(matrix: &mut RowMajorMatrix<F>, right: bool, value: u32) {
    let width = matrix.width();
    let local: &mut VerifierWarpHistoryChunkCompositionColsV3<F> =
        matrix.values[..width].borrow_mut();
    let message = if right {
        &mut local.right
    } else {
        &mut local.left
    };
    message.fields[CHUNK_INDEX_V3.start] = F::from_u16(value as u16);
    message.fields[CHUNK_INDEX_V3.start + 1] = F::from_u16((value >> 16) as u16);
    let bits = if right {
        &mut local.right_chunk_index_bits
    } else {
        &mut local.left_chunk_index_bits
    };
    for (limb_index, limb_bits) in bits.iter_mut().enumerate() {
        let limb = ((value >> (16 * limb_index)) & 0xffff) as u16;
        for (bit_index, bit) in limb_bits.iter_mut().enumerate() {
            *bit = F::from_bool(((limb >> bit_index) & 1) == 1);
        }
    }
}

#[test]
fn honest_interval_merge_satisfies_plain_constraints_and_maps_all_fields() {
    let air = standalone_air();
    let record = adjacent_record();
    let trace = air.generate_trace(&record).unwrap();
    check_plain(&air, &trace.matrix, &trace.public_values);

    let left = record.left.to_vec();
    let right = record.right.to_vec();
    let merged = trace.merged.to_vec();
    assert_eq!(merged.len(), VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3);
    for index in 0..VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3 {
        let expected = if FIXED_BINDINGS_V3.contains(&index)
            || TOTAL_BATCH_COUNT_V3.contains(&index)
            || START_BATCH_INDEX_V3.contains(&index)
            || ACTIVE_CHILD_COUNT_BEFORE_V3.contains(&index)
            || index == INITIAL_PC_V3
            || INITIAL_MEMORY_ROOT_V3.contains(&index)
            || INITIAL_ACCUMULATOR_DIGEST_V3.contains(&index)
            || HISTORY_HASH_BEFORE_V3.contains(&index)
            || OBSERVATION_COUNT_BEFORE_V3.contains(&index)
            || index == IS_GENESIS_V3
        {
            left[index]
        } else {
            right[index]
        };
        assert_eq!(merged[index], expected, "wrong merged field {index}");
    }
}

#[test]
fn plain_constraints_cover_every_fixed_and_continuity_coordinate() {
    let air = standalone_air();
    let trace = air.generate_trace(&adjacent_record()).unwrap();

    for index in FIXED_BINDINGS_V3
        .chain(TOTAL_BATCH_COUNT_V3)
        .chain(START_BATCH_INDEX_V3)
        .chain(ACTIVE_CHILD_COUNT_BEFORE_V3)
        .chain(core::iter::once(INITIAL_PC_V3))
        .chain(INITIAL_MEMORY_ROOT_V3)
        .chain(INITIAL_ACCUMULATOR_DIGEST_V3)
        .chain(HISTORY_HASH_BEFORE_V3)
        .chain(OBSERVATION_COUNT_BEFORE_V3)
    {
        let mut bad = trace.matrix.clone();
        mutate_child(&mut bad, true, index, F::ONE);
        assert_plain_rejects(
            &air,
            &bad,
            &trace.public_values,
            &format!("right continuity field {index}"),
        );
    }

    for index in PUBLIC_VALUES_DIGEST_V3 {
        let mut bad = trace.matrix.clone();
        mutate_child(&mut bad, false, index, F::ONE);
        assert_plain_rejects(
            &air,
            &bad,
            &trace.public_values,
            &format!("left nonterminal PVS field {index}"),
        );
    }
}

#[test]
fn plain_constraints_reject_bad_indices_flags_and_nonterminal_right_pvs() {
    let air = standalone_air();
    let trace = air.generate_trace(&adjacent_record()).unwrap();

    let mut bad_index = trace.matrix.clone();
    set_chunk_index(&mut bad_index, true, 2);
    assert_plain_rejects(
        &air,
        &bad_index,
        &trace.public_values,
        "skipped chunk index",
    );

    for (right, index, label) in [
        (true, IS_GENESIS_V3, "genesis on right"),
        (false, IS_TERMINAL_V3, "terminal on left"),
    ] {
        let mut bad = trace.matrix.clone();
        mutate_child(&mut bad, right, index, F::ONE);
        assert_plain_rejects(&air, &bad, &trace.public_values, label);
    }

    let mut non_boolean = trace.matrix.clone();
    mutate_child(&mut non_boolean, false, IS_GENESIS_V3, F::ONE);
    assert_plain_rejects(
        &air,
        &non_boolean,
        &trace.public_values,
        "non-boolean genesis",
    );

    let mut nonterminal_pvs = trace.matrix.clone();
    mutate_child(&mut nonterminal_pvs, true, IS_TERMINAL_V3, -F::ONE);
    assert_plain_rejects(
        &air,
        &nonterminal_pvs,
        &trace.public_values,
        "nonterminal right carrying PVS",
    );
}

#[test]
fn host_oracle_rejects_every_composition_rule() {
    let honest = adjacent_record();
    assert!(honest.merged().is_ok());

    let mut cases = Vec::new();
    let mut bad = honest;
    bad.right.protocol_digest[0] += F::ONE;
    cases.push(bad);
    let mut bad = honest;
    bad.right.total_batch_count += 1;
    cases.push(bad);
    let mut bad = honest;
    bad.right.start_batch_index += 1;
    cases.push(bad);
    let mut bad = honest;
    bad.right.initial_state.pc += F::ONE;
    cases.push(bad);
    let mut bad = honest;
    bad.right.initial_accumulator_digest[0] += F::ONE;
    cases.push(bad);
    let mut bad = honest;
    bad.right.history_hash_before[0] += F::ONE;
    cases.push(bad);
    let mut bad = honest;
    bad.right.active_child_count_before += 1;
    cases.push(bad);
    let mut bad = honest;
    bad.right.observation_count_before += 1;
    cases.push(bad);
    let mut bad = honest;
    bad.right.chunk_index += 1;
    cases.push(bad);
    let mut bad = honest;
    bad.right.is_genesis = true;
    cases.push(bad);
    let mut bad = honest;
    bad.left.is_terminal = true;
    cases.push(bad);
    let mut bad = honest;
    bad.left.public_values_digest[0] = F::ONE;
    cases.push(bad);
    let mut bad = honest;
    bad.right.is_terminal = false;
    cases.push(bad);

    for (index, case) in cases.into_iter().enumerate() {
        assert!(case.merged().is_err(), "host oracle accepted case {index}");
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
struct TestIntervalBusColsV3<T> {
    active: T,
    message: VerifierWarpHistoryChunkIntervalMessageV3<T>,
}

#[derive(Clone, Copy, Debug)]
enum TestIntervalBusDirectionV3 {
    Provide,
    Consume,
}

#[derive(Clone, Debug)]
struct TestIntervalBusAirV3 {
    bus: VerifierWarpHistoryChunkIntervalBusV3,
    direction: TestIntervalBusDirectionV3,
}

impl BaseAir<F> for TestIntervalBusAirV3 {
    fn width(&self) -> usize {
        core::mem::size_of::<TestIntervalBusColsV3<u8>>()
    }
}

impl BaseAirWithPublicValues<F> for TestIntervalBusAirV3 {}
impl PartitionedBaseAir<F> for TestIntervalBusAirV3 {}

impl<AB> Air<AB> for TestIntervalBusAirV3
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("test interval bus row");
        let next_row = main.row_slice(1).expect("test interval bus padding row");
        let local: &TestIntervalBusColsV3<AB::Var> = (*local_row).borrow();
        let next: &TestIntervalBusColsV3<AB::Var> = (*next_row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_last_row().assert_zero(local.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.active);
        match self.direction {
            TestIntervalBusDirectionV3::Provide => {
                self.bus
                    .add_key_with_lookups(builder, local.message.clone(), local.active);
            }
            TestIntervalBusDirectionV3::Consume => {
                self.bus
                    .lookup_key(builder, local.message.clone(), local.active);
            }
        }
    }
}

fn test_bus_trace(message: VerifierWarpHistoryChunkIntervalMessageV3<F>) -> RowMajorMatrix<F> {
    let width = core::mem::size_of::<TestIntervalBusColsV3<u8>>();
    let mut values = F::zero_vec(2 * width);
    let local: &mut TestIntervalBusColsV3<F> = values[..width].borrow_mut();
    local.active = F::ONE;
    local.message = message;
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

struct BusHarnessV3 {
    airs: Vec<AirRef<NativeSC>>,
    interactions: Vec<Vec<SymbolicInteraction<F>>>,
    honest_matrices: Vec<RowMajorMatrix<F>>,
}

impl BusHarnessV3 {
    fn new() -> Self {
        let left_bus = VerifierWarpHistoryChunkIntervalBusV3::new(20);
        let right_bus = VerifierWarpHistoryChunkIntervalBusV3::new(21);
        let output_bus = VerifierWarpHistoryChunkIntervalBusV3::new(22);
        let composition = VerifierWarpHistoryChunkCompositionAirV3::new_with_output_bus(
            left_bus, right_bus, output_bus, 1,
        )
        .unwrap();
        let record = adjacent_record();
        let trace = composition.generate_trace(&record).unwrap();
        let airs: Vec<AirRef<NativeSC>> = vec![
            Arc::new(TestIntervalBusAirV3 {
                bus: left_bus,
                direction: TestIntervalBusDirectionV3::Provide,
            }),
            Arc::new(TestIntervalBusAirV3 {
                bus: right_bus,
                direction: TestIntervalBusDirectionV3::Provide,
            }),
            Arc::new(composition),
            Arc::new(TestIntervalBusAirV3 {
                bus: output_bus,
                direction: TestIntervalBusDirectionV3::Consume,
            }),
        ];
        let honest_matrices = vec![
            test_bus_trace(
                VerifierWarpHistoryChunkIntervalMessageV3::from_public_values(record.left),
            ),
            test_bus_trace(
                VerifierWarpHistoryChunkIntervalMessageV3::from_public_values(record.right),
            ),
            trace.matrix,
            test_bus_trace(
                VerifierWarpHistoryChunkIntervalMessageV3::from_public_values(trace.merged),
            ),
        ];
        let interactions = airs
            .iter()
            .map(|air| symbolic_interactions(air.as_ref()))
            .collect();
        Self {
            airs,
            interactions,
            honest_matrices,
        }
    }

    fn check_plain(&self, matrices: &[RowMajorMatrix<F>]) {
        for (air, matrix) in self.airs.iter().zip(matrices) {
            check_constraints::<_, NativeSC>(
                air.as_ref(),
                &air.name(),
                &None,
                &[matrix.as_view()],
                &[],
            );
        }
    }

    fn check_buses(&self, matrices: &[RowMajorMatrix<F>]) {
        let names = self.airs.iter().map(|air| air.name()).collect::<Vec<_>>();
        let preprocessed = vec![None; self.airs.len()];
        let views = matrices
            .iter()
            .map(|matrix| vec![matrix.as_view()])
            .collect::<Vec<_>>();
        check_logup(
            &names,
            &self.interactions,
            &preprocessed,
            &views,
            &vec![Vec::new(); self.airs.len()],
        );
    }
}

#[test]
fn typed_child_and_output_buses_balance_for_honest_merge() {
    let harness = BusHarnessV3::new();
    harness.check_plain(&harness.honest_matrices);
    harness.check_buses(&harness.honest_matrices);
    let interaction_widths = harness
        .interactions
        .iter()
        .flatten()
        .map(|interaction| interaction.message.len())
        .collect::<Vec<_>>();
    assert_eq!(interaction_widths, vec![108, 108, 108, 108, 108, 108]);
}

#[test]
fn every_left_and_right_child_field_is_bound_by_its_typed_bus() {
    let harness = BusHarnessV3::new();
    for right in [false, true] {
        for index in 0..VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3 {
            let mut matrices = harness.honest_matrices.clone();
            mutate_child(&mut matrices[2], right, index, F::ONE);
            assert!(
                std::panic::catch_unwind(AssertUnwindSafe(|| harness.check_buses(&matrices)))
                    .is_err(),
                "typed bus ignored {} child field {index}",
                if right { "right" } else { "left" }
            );
        }
    }
}

#[test]
fn every_merged_output_field_is_bound_by_the_typed_bus() {
    let harness = BusHarnessV3::new();
    for index in 0..VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3 {
        let mut matrices = harness.honest_matrices.clone();
        let width = matrices[3].width();
        let local: &mut TestIntervalBusColsV3<F> = matrices[3].values[..width].borrow_mut();
        local.message.fields[index] += F::ONE;
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| harness.check_buses(&matrices))).is_err(),
            "typed output bus ignored merged field {index}"
        );
    }
}

#[test]
fn bus_configuration_rejects_aliases_and_zero_output_count() {
    let bus_1 = VerifierWarpHistoryChunkIntervalBusV3::new(30);
    let bus_2 = VerifierWarpHistoryChunkIntervalBusV3::new(31);
    let bus_3 = VerifierWarpHistoryChunkIntervalBusV3::new(32);
    assert_eq!(
        VerifierWarpHistoryChunkCompositionAirV3::new_standalone(bus_1, bus_1).unwrap_err(),
        VerifierWarpHistoryChunkCompositionConfigErrorV3::ReusedBusIndex
    );
    assert_eq!(
        VerifierWarpHistoryChunkCompositionAirV3::new_with_output_bus(bus_1, bus_2, bus_1, 1)
            .unwrap_err(),
        VerifierWarpHistoryChunkCompositionConfigErrorV3::ReusedBusIndex
    );
    assert_eq!(
        VerifierWarpHistoryChunkCompositionAirV3::new_with_output_bus(bus_1, bus_2, bus_3, 0)
            .unwrap_err(),
        VerifierWarpHistoryChunkCompositionConfigErrorV3::ZeroOutputLookupCount
    );
}

fn terminal_root_air() -> VerifierWarpHistoryTerminalRootAirV3 {
    VerifierWarpHistoryTerminalRootAirV3::new(
        VerifierWarpHistoryChunkIntervalBusV3::new(40),
        crate::circuit::verifier_warp_history_v2::VerifierWarpBlockPublicValuesBusV2::new(41),
        openvm_recursion_circuit::bus::PublicValuesBus::new(42),
        0,
        4_000,
        DIGEST_SIZE,
        8,
        1,
    )
    .unwrap()
}

fn set_u32_field(values: &mut [F], range: core::ops::Range<usize>, value: u32) {
    values[1 + range.start] = F::from_u16(value as u16);
    values[1 + range.start + 1] = F::from_u16((value >> 16) as u16);
}

fn set_u32_public_field(values: &mut [F], range: core::ops::Range<usize>, value: u32) {
    values[range.start] = F::from_u16(value as u16);
    values[range.start + 1] = F::from_u16((value >> 16) as u16);
}

#[test]
fn terminal_root_fixes_complete_history_size_and_last_chunk_in_the_air() {
    let air = terminal_root_air();
    let endpoint = adjacent_record().merged().unwrap();
    let user_public_values = digest(1_200);
    let (trace, public_values) = air.generate_trace(endpoint, &user_public_values).unwrap();
    check_constraints::<_, NativeSC>(
        &air,
        "History-v3 terminal root",
        &None,
        &[trace.as_view()],
        &public_values,
    );

    let mut wrong_total_trace = trace.clone();
    let mut wrong_total_public = public_values.clone();
    set_u32_field(&mut wrong_total_trace.values, TOTAL_BATCH_COUNT_V3, 9);
    set_u32_field(&mut wrong_total_trace.values, END_BATCH_INDEX_V3, 9);
    set_u32_public_field(&mut wrong_total_public, TOTAL_BATCH_COUNT_V3, 9);
    set_u32_public_field(&mut wrong_total_public, END_BATCH_INDEX_V3, 9);
    assert!(
        std::panic::catch_unwind(AssertUnwindSafe(|| check_constraints::<_, NativeSC>(
            &air,
            "History-v3 terminal root wrong total",
            &None,
            &[wrong_total_trace.as_view()],
            &wrong_total_public,
        )))
        .is_err(),
        "terminal root AIR accepted a shorter or longer history"
    );

    let mut wrong_chunk_trace = trace.clone();
    let mut wrong_chunk_public = public_values.clone();
    set_u32_field(&mut wrong_chunk_trace.values, CHUNK_INDEX_V3, 2);
    set_u32_public_field(&mut wrong_chunk_public, CHUNK_INDEX_V3, 2);
    assert!(
        std::panic::catch_unwind(AssertUnwindSafe(|| check_constraints::<_, NativeSC>(
            &air,
            "History-v3 terminal root wrong chunk",
            &None,
            &[wrong_chunk_trace.as_view()],
            &wrong_chunk_public,
        )))
        .is_err(),
        "terminal root AIR accepted a noncanonical final chunk"
    );
}

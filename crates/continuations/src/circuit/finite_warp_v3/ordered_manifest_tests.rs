use std::{panic::AssertUnwindSafe, sync::Arc};

use openvm_recursion_circuit::system::{BusIndexManager, BusInventory};
use openvm_stark_backend::{
    air_builders::{
        debug::{check_constraints, check_logup},
        symbolic::get_symbolic_builder,
    },
    interaction::{InteractionBuilder, SymbolicInteraction},
    keygen::types::TraceWidth,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    test_utils::test_system_params_small,
    AnyAir, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as NativeSC, Digest, DIGEST_SIZE, EF, F,
};

use super::*;
use crate::circuit::finite_warp_v3::{
    FiniteWarpV3ManifestCallMessage, FiniteWarpV3ManifestReceiptMessage, FiniteWarpV3ReceiptBuses,
};

fn digest(seed: u32) -> Digest {
    core::array::from_fn(|limb| F::from_u32(seed + limb as u32))
}

fn profile(normalized_leaf_count: u32) -> OrderedManifestProfile {
    OrderedManifestProfile {
        protocol_digest: digest(10),
        relation_digest: digest(20),
        warp_index_digest: digest(30),
        verifier_component_digest: digest(40),
        prefix: OrderedManifestPrefixProfile {
            index_digest: digest(50),
            l_skip: 0,
            log_message_len: 8,
            log_blowup: 1,
            log_codeword_len: 9,
            rows_per_leaf: 16,
            trace_prefix_len: 128,
            active_count_block_start: 64,
            active_count_log_height: 3,
        },
        expected_normalized_leaf_count: normalized_leaf_count,
        max_active_children_per_source: 2,
        suspend_exit_code: 2,
    }
}

fn record() -> OrderedManifestRecord {
    let source_count = 7usize;
    let program = digest(100);
    let sources = (0..source_count)
        .map(|index| OrderedManifestSourceRecord {
            source_index: index as u32,
            call_index: u32::from(index >= 4),
            fresh_index_in_call: if index < 4 {
                index as u32
            } else {
                (index - 4) as u32
            },
            normalized_leaf_start: (2 * index) as u32,
            active_child_count: 2,
            program_commitment: program,
            initial_state: OrderedManifestVmState {
                pc: F::from_usize(index),
                memory_root: digest(200 + 10 * index as u32),
            },
            final_state: OrderedManifestVmState {
                pc: F::from_usize(index + 1),
                memory_root: digest(200 + 10 * (index + 1) as u32),
            },
            exit_code: if index + 1 == source_count {
                F::ZERO
            } else {
                F::from_u32(2)
            },
            is_terminate: F::from_bool(index + 1 == source_count),
            source_instance_digest: digest(400 + 10 * index as u32),
        })
        .collect();
    OrderedManifestRecord {
        calls: vec![
            OrderedManifestCallRecord {
                call_index: 0,
                input_arity: 4,
                source_start: 0,
                source_count: 4,
                base_root: digest(500),
                full_root: digest(510),
                logup_alpha: EF::from_u32(7),
                logup_beta: EF::from_u32(11),
            },
            OrderedManifestCallRecord {
                call_index: 1,
                input_arity: 4,
                source_start: 4,
                source_count: 3,
                base_root: digest(520),
                full_root: digest(530),
                logup_alpha: EF::from_u32(13),
                logup_beta: EF::from_u32(17),
            },
        ],
        sources,
        final_accumulator_digest: digest(600),
    }
}

fn allocate_buses() -> (BusInventory, FiniteWarpV3ReceiptBuses, OrderedManifestBuses) {
    let mut indices = BusIndexManager::new();
    let shared = BusInventory::new(&mut indices);
    let wrapper = FiniteWarpV3ReceiptBuses::new(indices.next_bus_idx());
    let mut indices = BusIndexManager::from_next_bus_idx(wrapper.next_bus_idx());
    let manifest = OrderedManifestBuses::new(&mut indices);
    (shared, wrapper, manifest)
}

#[test]
fn slow_oracle_rejects_dropped_duplicated_reordered_and_discontinuous_sources() {
    let profile = profile(14);
    let valid = record();
    derive_ordered_manifest(&profile, &valid).expect("valid manifest");

    let mut dropped = valid.clone();
    dropped.sources.remove(3);
    assert!(derive_ordered_manifest(&profile, &dropped).is_err());

    let mut duplicated = valid.clone();
    duplicated.sources[3] = duplicated.sources[2].clone();
    assert!(derive_ordered_manifest(&profile, &duplicated).is_err());

    let mut reordered = valid.clone();
    reordered.sources.swap(2, 3);
    assert!(derive_ordered_manifest(&profile, &reordered).is_err());

    let mut discontinuous = valid;
    discontinuous.sources[4].initial_state.memory_root = digest(999);
    assert!(matches!(
        derive_ordered_manifest(&profile, &discontinuous),
        Err(OrderedManifestError::StateContinuity(4))
    ));
}

#[test]
fn transcript_digests_bind_every_ordering_and_occupancy_field() {
    let manifest_profile = profile(14);
    let original = derive_ordered_manifest(&manifest_profile, &record()).unwrap();
    let bundle = ordered_manifest_transcript_bundle(&manifest_profile, &original).unwrap();
    assert_eq!(bundle.logs.len(), 4);
    assert_eq!(bundle.schedule_digest, original.schedule_digest);
    assert_eq!(bundle.source_order_digests, original.source_order_digests);
    assert_eq!(bundle.manifest_digest, original.manifest_digest);

    let mut changed_record = record();
    changed_record.sources[2].source_instance_digest = digest(777);
    let changed = derive_ordered_manifest(&manifest_profile, &changed_record).unwrap();
    assert_eq!(changed.schedule_digest, original.schedule_digest);
    assert_eq!(changed.source_order_digests, original.source_order_digests);
    assert_ne!(changed.manifest_digest, original.manifest_digest);

    let mut changed_record = record();
    changed_record.sources[2].active_child_count = 1;
    for source in &mut changed_record.sources[3..] {
        source.normalized_leaf_start -= 1;
    }
    let changed_profile = profile(13);
    let changed = derive_ordered_manifest(&changed_profile, &changed_record).unwrap();
    assert_ne!(changed.source_order_digests, original.source_order_digests);
    assert_ne!(changed.manifest_digest, original.manifest_digest);
}

#[derive(Clone, Debug)]
struct TestBoundaryAir {
    profile: OrderedManifestProfile,
    derived: OrderedManifestDerivedRecord,
    wrapper: FiniteWarpV3ReceiptBuses,
    buses: OrderedManifestBuses,
}

impl BaseAir<F> for TestBoundaryAir {
    fn width(&self) -> usize {
        1
    }
}
impl BaseAirWithPublicValues<F> for TestBoundaryAir {}
impl PartitionedBaseAir<F> for TestBoundaryAir {}

impl<AB> Air<AB> for TestBoundaryAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("test authority row")[0];
        let next = main.row_slice(1).expect("test authority padding")[0];
        builder.assert_bool(local);
        builder.when_first_row().assert_one(local);
        builder.when_last_row().assert_zero(local);
        builder.when_transition().assert_eq(local - next, local);

        for source in &self.derived.record.sources {
            self.buses.source_authority.add_key_with_lookups(
                builder,
                source_message::<AB>(&self.profile, source),
                local,
            );
        }
        for (index, call) in self.derived.record.calls.iter().enumerate() {
            let prefix = prefix_message::<AB>(&self.profile, &self.derived, call, index);
            self.buses
                .prefix_authority
                .add_key_with_lookups(builder, prefix.clone(), local);
            self.buses.prefix_binding.lookup_key(builder, prefix, local);
        }

        let first = self.derived.record.sources.first().unwrap();
        let last = self.derived.record.sources.last().unwrap();
        let manifest = FiniteWarpV3ManifestReceiptMessage {
            protocol_digest: constant_digest::<AB>(self.profile.protocol_digest),
            relation_digest: constant_digest::<AB>(self.profile.relation_digest),
            warp_index_digest: constant_digest::<AB>(self.profile.warp_index_digest),
            verifier_component_digest: constant_digest::<AB>(
                self.profile.verifier_component_digest,
            ),
            schedule_digest: constant_digest::<AB>(self.derived.schedule_digest),
            manifest_digest: constant_digest::<AB>(self.derived.manifest_digest),
            source_count: AB::Expr::from_usize(self.derived.record.sources.len()),
            call_count: AB::Expr::from_usize(self.derived.record.calls.len()),
            program_commitment: constant_digest::<AB>(first.program_commitment),
            initial_pc: constant::<AB>(first.initial_state.pc),
            initial_root: constant_digest::<AB>(first.initial_state.memory_root),
            final_pc: constant::<AB>(last.final_state.pc),
            final_root: constant_digest::<AB>(last.final_state.memory_root),
            final_accumulator_digest: constant_digest::<AB>(
                self.derived.record.final_accumulator_digest,
            ),
            calls: core::array::from_fn(|index| {
                self.derived.record.calls.get(index).map_or_else(
                    || FiniteWarpV3ManifestCallMessage {
                        active: AB::Expr::ZERO,
                        source_start: AB::Expr::ZERO,
                        source_count: AB::Expr::ZERO,
                        input_arity: AB::Expr::ZERO,
                        fresh_stacked_root: [AB::Expr::ZERO; DIGEST_SIZE],
                    },
                    |call| FiniteWarpV3ManifestCallMessage {
                        active: AB::Expr::ONE,
                        source_start: AB::Expr::from_u32(call.source_start),
                        source_count: AB::Expr::from_u32(call.source_count),
                        input_arity: AB::Expr::from_u32(call.input_arity),
                        fresh_stacked_root: constant_digest::<AB>(call.full_root),
                    },
                )
            }),
        };
        self.wrapper.manifest.lookup_key(builder, manifest, local);
    }
}

fn source_message<AB: AirBuilder<F = F>>(
    profile: &OrderedManifestProfile,
    source: &OrderedManifestSourceRecord,
) -> OrderedManifestSourceReceiptMessage<AB::Expr> {
    OrderedManifestSourceReceiptMessage {
        protocol_digest: constant_digest::<AB>(profile.protocol_digest),
        relation_digest: constant_digest::<AB>(profile.relation_digest),
        warp_index_digest: constant_digest::<AB>(profile.warp_index_digest),
        source_index: AB::Expr::from_u32(source.source_index),
        call_index: AB::Expr::from_u32(source.call_index),
        fresh_index_in_call: AB::Expr::from_u32(source.fresh_index_in_call),
        normalized_leaf_start: AB::Expr::from_u32(source.normalized_leaf_start),
        active_child_count: AB::Expr::from_u32(source.active_child_count.into()),
        program_commitment: constant_digest::<AB>(source.program_commitment),
        initial_pc: constant::<AB>(source.initial_state.pc),
        initial_root: constant_digest::<AB>(source.initial_state.memory_root),
        final_pc: constant::<AB>(source.final_state.pc),
        final_root: constant_digest::<AB>(source.final_state.memory_root),
        exit_code: constant::<AB>(source.exit_code),
        is_terminate: constant::<AB>(source.is_terminate),
        source_instance_digest: constant_digest::<AB>(source.source_instance_digest),
    }
}

fn prefix_message<AB: AirBuilder<F = F>>(
    profile: &OrderedManifestProfile,
    derived: &OrderedManifestDerivedRecord,
    call: &OrderedManifestCallRecord,
    call_position: usize,
) -> OrderedManifestPrefixReceiptMessage<AB::Expr> {
    let mut counts = [AB::Expr::ZERO; ORDERED_MANIFEST_PREFIX_SLOTS];
    let start = call.source_start as usize;
    for (slot, source) in derived.record.sources[start..start + call.source_count as usize]
        .iter()
        .enumerate()
    {
        counts[slot] = AB::Expr::from_u32(source.active_child_count.into());
    }
    OrderedManifestPrefixReceiptMessage {
        protocol_version: AB::Expr::from_u32(ORDERED_MANIFEST_PREFIX_PROTOCOL_VERSION),
        relation_digest: constant_digest::<AB>(profile.relation_digest),
        index_digest: constant_digest::<AB>(profile.prefix.index_digest),
        schedule_digest: constant_digest::<AB>(derived.schedule_digest),
        source_order_digest: constant_digest::<AB>(derived.source_order_digests[call_position]),
        call_index: AB::Expr::from_u32(call.call_index),
        source_start: AB::Expr::from_u32(call.source_start),
        source_count: AB::Expr::from_u32(call.source_count),
        expected_active_child_counts: counts,
        l_skip: AB::Expr::from_u32(profile.prefix.l_skip),
        log_message_len: AB::Expr::from_u32(profile.prefix.log_message_len),
        log_blowup: AB::Expr::from_u32(profile.prefix.log_blowup),
        log_codeword_len: AB::Expr::from_u32(profile.prefix.log_codeword_len),
        rows_per_leaf: AB::Expr::from_u32(profile.prefix.rows_per_leaf),
        trace_prefix_len_lo: AB::Expr::from_u32(profile.prefix.trace_prefix_len as u32),
        trace_prefix_len_hi: AB::Expr::from_u32((profile.prefix.trace_prefix_len >> 32) as u32),
        active_count_block_start_lo: AB::Expr::from_u32(
            profile.prefix.active_count_block_start as u32,
        ),
        active_count_block_start_hi: AB::Expr::from_u32(
            (profile.prefix.active_count_block_start >> 32) as u32,
        ),
        active_count_log_height: AB::Expr::from_u32(profile.prefix.active_count_log_height),
        base_root: constant_digest::<AB>(call.base_root),
        full_root: constant_digest::<AB>(call.full_root),
        logup_alpha: core::array::from_fn(|index| {
            constant::<AB>(call.logup_alpha.as_basis_coefficients_slice()[index])
        }),
        logup_beta: core::array::from_fn(|index| {
            constant::<AB>(call.logup_beta.as_basis_coefficients_slice()[index])
        }),
    }
}

fn constant<AB: AirBuilder<F = F>>(value: F) -> AB::Expr {
    AB::Expr::from_u32(value.as_canonical_u32())
}

fn constant_digest<AB: AirBuilder<F = F>>(digest: Digest) -> [AB::Expr; DIGEST_SIZE] {
    digest.map(constant::<AB>)
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

#[test]
fn receipt_air_and_exact_transcript_balance_end_to_end() {
    let profile = profile(14);
    let derived = derive_ordered_manifest(&profile, &record()).unwrap();
    let (shared, wrapper, buses) = allocate_buses();
    let producer = OrderedManifestReceiptProducer::new(
        profile.clone(),
        wrapper,
        buses,
        &shared,
        test_system_params_small(0, 8, 4),
    )
    .unwrap();
    assert!(
        symbolic_interactions(&producer.header_air)
            .iter()
            .all(|interaction| interaction.bus_index != wrapper.execution.index()),
        "the manifest receipt already binds execution fields; the sole execution owner must remain the VmPvs AIR"
    );
    let traces = producer.generate_traces(&derived.record).unwrap();
    let authority = TestBoundaryAir {
        profile,
        derived,
        wrapper,
        buses,
    };
    let authority_trace = RowMajorMatrix::new(vec![F::ONE, F::ZERO], 1);

    let mut airs = producer.airs::<NativeSC>();
    airs.push(Arc::new(authority));
    let matrices = vec![
        traces.header,
        traces.calls,
        traces.sources,
        traces.transcript.trace,
        traces.transcript.poseidon2_trace,
        authority_trace,
    ];
    assert_eq!(airs.len(), matrices.len());
    let public_values = vec![Vec::<F>::new(); airs.len()];
    for ((air, matrix), pvs) in airs.iter().zip(&matrices).zip(&public_values) {
        let preprocessed = BaseAir::<F>::preprocessed_trace(air.as_ref());
        check_constraints::<_, NativeSC>(
            air.as_ref(),
            &air.name(),
            &preprocessed.as_ref().map(RowMajorMatrix::as_view),
            &[matrix.as_view()],
            pvs,
        );
    }
    let preprocessed_owned = airs
        .iter()
        .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
        .collect::<Vec<_>>();
    let preprocessed = preprocessed_owned
        .iter()
        .map(|matrix| matrix.as_ref().map(RowMajorMatrix::as_view))
        .collect::<Vec<_>>();
    let interactions = airs
        .iter()
        .map(|air| symbolic_interactions(air.as_ref()))
        .collect::<Vec<_>>();
    let views = matrices
        .iter()
        .map(|matrix| vec![matrix.as_view()])
        .collect::<Vec<_>>();
    check_logup(
        &airs.iter().map(|air| air.name()).collect::<Vec<_>>(),
        &interactions,
        &preprocessed,
        &views,
        &public_values,
    );
}

#[test]
fn source_index_mutation_is_rejected_by_constraints() {
    let profile = profile(14);
    let derived = derive_ordered_manifest(&profile, &record()).unwrap();
    let (shared, wrapper, buses) = allocate_buses();
    let producer = OrderedManifestReceiptProducer::new(
        profile,
        wrapper,
        buses,
        &shared,
        test_system_params_small(0, 8, 4),
    )
    .unwrap();
    let mut traces = producer.generate_traces(&derived.record).unwrap();
    let width = traces.sources.width();
    // active, is_last, source_index
    traces.sources.values[width + 2] = F::from_u32(99);
    let rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        check_constraints::<_, NativeSC>(
            &producer.source_air,
            "OrderedManifestSourceAir",
            &None,
            &[traces.sources.as_view()],
            &[],
        );
    }));
    assert!(rejected.is_err(), "mutated source index was accepted");
}

use core::borrow::{Borrow, BorrowMut};
use std::{panic::AssertUnwindSafe, sync::Arc};

use openvm_circuit::system::memory::dimensions::MemoryDimensions;
use openvm_cpu_backend::CpuBackend;
use openvm_recursion_circuit::{
    native_warp::{
        NativeStandardVaccDigestBus, NativeStandardVaccEndBus, NativeStandardVaccProfile,
        NativeStandardVaccProtocolBus, NativeStandardVaccRootBus, NativeWarpPcdBusInventory,
    },
    system::{BusIndexManager, BusInventory},
    transcript::{Poseidon2BusOwner, TranscriptModule},
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::{
        debug::{check_constraints, check_logup},
        symbolic::get_symbolic_builder,
    },
    interaction::{BusIndex, InteractionBuilder, SymbolicInteraction},
    keygen::types::TraceWidth,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{BasedVectorSpace, PrimeCharacteristicRing},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::AirProvingContext,
    AirRef, AnyAir, BaseAirWithPublicValues, PartitionedBaseAir, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, Digest, DIGEST_SIZE, D_EF, EF, F,
};
use openvm_verify_stark_host::pvs::VmPvs;

use super::*;
use crate::circuit::{
    native_warp_history_v19::{
        appendix_d_fixture_for_fixed_source, certified_direct_air_vacc_input_message_v19,
        generate_logup_only_producer_trace_v19, generate_warp_replay_producer_trace_v19,
        retained_fixed_multi_air_fixture, CertifiedDirectAirVaccInputBusV19,
        CertifiedSwirlRawOpeningBusV19, CertifiedWarpReplayBusV19, DirectAirVaccContextBusV19,
        DirectAirVaccHistoryBusesV19, DirectAirVaccVerifierModuleV19,
        DirectAirVaccVerifierRecordV19, HistoryPoseidon2CompressBusV19, LogUpOnlyHistoryBusV19,
        LogUpOnlyProducerAirV19, WarpReplayProducerAirV19,
    },
    root::commit::{generate_proving_ctx as generate_user_pvs_commit_ctx, UserPvsCommitAir},
    subair::{MerkleRootBus, MerkleTreeInternalBus},
};

type NativeSC = openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

fn digest(seed: u32) -> Digest {
    core::array::from_fn(|limb| F::from_u32(seed + limb as u32))
}

fn ext_array(value: EF) -> [F; D_EF] {
    value.as_basis_coefficients_slice().try_into().unwrap()
}

fn aggregate_child(final_memory_root: Digest, terminates: bool) -> VerifierWarpChildRecordV2 {
    VerifierWarpChildRecordV2 {
        occupied: true,
        input: VerifierWarpVmStateV2 {
            pc: F::from_u32(11),
            memory_root: digest(4_100),
        },
        output: VerifierWarpVmStateV2 {
            pc: F::from_u32(19),
            memory_root: final_memory_root,
        },
        exit_code: if terminates { F::ZERO } else { F::from_u32(1) },
        terminates,
    }
}

fn history_record(
    final_memory_root: Digest,
    public_values_digest: Digest,
) -> VerifierWarpHistoryTransitionRecordV2 {
    VerifierWarpHistoryTransitionRecordV2 {
        protocol_version: VERIFIER_WARP_HISTORY_PROTOCOL_V2,
        vacc_input_arity: VERIFIER_WARP_VACC_INPUT_ARITY_V2 as u8,
        protocol_digest: digest(200),
        relation_digest: digest(300),
        batch_index: 0,
        active_child_count: 1,
        children: {
            let mut children =
                [VerifierWarpChildRecordV2::padding(); VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2];
            children[0] = aggregate_child(final_memory_root, true);
            children
        },
        program_commitment: digest(400),
        prior_accumulator_digest: digest(500),
        source_accumulator_digest: digest(600),
        output_accumulator_digest: digest(700),
        source_commitment_root: digest(800),
        external_logup_gkr_digest: digest(900),
        source_functional_digest: digest(1_000),
        setup_openings_digest: digest(1_100),
        source_statement_digest: digest(1_200),
        transition_transcript_digest: digest(1_300),
        public_values_digest,
    }
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

fn symbolic_max_constraint_degree(air: &dyn AnyAir<NativeSC>) -> usize {
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
    .max_constraint_degree()
}

#[derive(Clone, Copy)]
struct TestHistoryEndpointConsumerAirV2 {
    endpoint_bus: VerifierWarpHistoryEndpointBusV2,
}

impl BaseAir<F> for TestHistoryEndpointConsumerAirV2 {
    fn width(&self) -> usize {
        VerifierWarpHistoryPublicValuesV2::WIDTH
    }
}

impl BaseAirWithPublicValues<F> for TestHistoryEndpointConsumerAirV2 {}
impl PartitionedBaseAir<F> for TestHistoryEndpointConsumerAirV2 {}

impl<AB> Air<AB> for TestHistoryEndpointConsumerAirV2
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("History endpoint consumer row");
        assert_eq!(row.len(), VerifierWarpHistoryPublicValuesV2::WIDTH);
        let mut cursor = 0usize;
        let mut take = || {
            let value = row[cursor];
            cursor += 1;
            value.into()
        };
        let message = VerifierWarpHistoryEndpointMessageV2 {
            protocol_version: take(),
            source_child_capacity: take(),
            vacc_input_arity: take(),
            protocol_digest: core::array::from_fn(|_| take()),
            relation_digest: core::array::from_fn(|_| take()),
            program_commitment: core::array::from_fn(|_| take()),
            batch_count: core::array::from_fn(|_| take()),
            active_child_count: core::array::from_fn(|_| take()),
            initial_pc: take(),
            initial_memory_root: core::array::from_fn(|_| take()),
            final_pc: take(),
            final_memory_root: core::array::from_fn(|_| take()),
            genesis_accumulator_digest: core::array::from_fn(|_| take()),
            final_accumulator_digest: core::array::from_fn(|_| take()),
            public_values_digest: core::array::from_fn(|_| take()),
            history_statement_digest: core::array::from_fn(|_| take()),
        };
        assert_eq!(cursor, VerifierWarpHistoryPublicValuesV2::WIDTH);
        self.endpoint_bus
            .lookup_key(builder, message, AB::Expr::ONE);
    }
}

/// Test-only positive authority for C1 in the C3 composition fixture. C2 is
/// exported by the genuine v19 mapped-functional/reduction path below.
#[repr(C)]
#[derive(AlignedBorrow)]
struct BridgeC1AuthorityColsV2<T> {
    active: T,
    fixed_setup: FixedSetupOpeningCertificateMessageV2<T>,
}

#[derive(Clone)]
struct BridgeC1AuthorityAirV2 {
    fixed_setup_bus: FixedSetupOpeningCertificateBusV2,
}

impl BaseAir<F> for BridgeC1AuthorityAirV2 {
    fn width(&self) -> usize {
        core::mem::size_of::<BridgeC1AuthorityColsV2<u8>>()
    }
}

impl BaseAirWithPublicValues<F> for BridgeC1AuthorityAirV2 {}
impl PartitionedBaseAir<F> for BridgeC1AuthorityAirV2 {}

impl<AB> Air<AB> for BridgeC1AuthorityAirV2
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("C1 authority row");
        let local: &BridgeC1AuthorityColsV2<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        self.fixed_setup_bus
            .send(builder, local.fixed_setup.clone(), local.active);
    }
}

fn bridge_c1_authority_trace_v2(
    fixed_setup: &FixedSetupOpeningCertificateMessageV2<F>,
) -> RowMajorMatrix<F> {
    let width = core::mem::size_of::<BridgeC1AuthorityColsV2<u8>>();
    let mut values = F::zero_vec(2 * width);
    let row: &mut BridgeC1AuthorityColsV2<F> = values[..width].borrow_mut();
    row.active = F::ONE;
    row.fixed_setup = fixed_setup.clone();
    RowMajorMatrix::new(values, width)
}

struct BlockComposition {
    airs: Vec<AirRef<NativeSC>>,
    contexts: Vec<AirProvingContext<CpuBackend<NativeSC>>>,
    selected_bus_indices: Vec<BusIndex>,
}

impl BlockComposition {
    fn check(&self) {
        assert_eq!(self.airs.len(), self.contexts.len());
        for (air, context) in self.airs.iter().zip(&self.contexts) {
            let preprocessed_owned = BaseAir::<F>::preprocessed_trace(air.as_ref());
            let preprocessed = preprocessed_owned.as_ref().map(RowMajorMatrix::as_view);
            let mut mains = context
                .cached_mains
                .iter()
                .map(|cached| cached.trace.as_view())
                .collect::<Vec<_>>();
            mains.push(context.common_main.as_view());
            check_constraints::<_, NativeSC>(
                air.as_ref(),
                &air.name(),
                &preprocessed,
                &mains,
                &context.public_values,
            );
        }
        let preprocessed_owned = self
            .airs
            .iter()
            .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
            .collect::<Vec<_>>();
        let preprocessed = preprocessed_owned
            .iter()
            .map(|matrix| matrix.as_ref().map(RowMajorMatrix::as_view))
            .collect::<Vec<_>>();
        let interactions = self
            .airs
            .iter()
            .map(|air| {
                symbolic_interactions(air.as_ref())
                    .into_iter()
                    .filter(|interaction| {
                        self.selected_bus_indices.is_empty()
                            || self.selected_bus_indices.contains(&interaction.bus_index)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let views = self
            .contexts
            .iter()
            .map(|context| {
                let mut mains = context
                    .cached_mains
                    .iter()
                    .map(|cached| cached.trace.as_view())
                    .collect::<Vec<_>>();
                mains.push(context.common_main.as_view());
                mains
            })
            .collect::<Vec<_>>();
        let names = self.airs.iter().map(|air| air.name()).collect::<Vec<_>>();
        let public_values = self
            .contexts
            .iter()
            .map(|context| context.public_values.clone())
            .collect::<Vec<_>>();
        check_logup(&names, &interactions, &preprocessed, &views, &public_values);
    }
}

fn block_composition(
    mutated_history_digest: Option<Digest>,
    direct_endpoint_mutation: Option<usize>,
) -> BlockComposition {
    let params = SystemParams::new_for_testing(10);
    let mut manager = BusIndexManager::new();
    let inventory = BusInventory::new(&mut manager);
    let transcript = TranscriptModule::<1>::new(inventory.clone(), params, false, false);
    let poseidon_owner = transcript.poseidon2_bus_owner();
    let merkle_root_bus = MerkleRootBus::new(manager.new_bus_idx());
    let merkle_internal_bus = MerkleTreeInternalBus::new(manager.new_bus_idx());
    let source_bus = VerifierWarpSourceCertificateBusV2::new(manager.new_bus_idx());
    let vacc_bus = VerifierWarpVaccCertificateBusV2::new(manager.new_bus_idx());
    let block_bus_index = manager.new_bus_idx();
    let block_bus = VerifierWarpBlockPublicValuesBusV2::new(block_bus_index);
    let endpoint_bus_index = manager.new_bus_idx();
    let endpoint_bus = VerifierWarpHistoryEndpointBusV2::new(endpoint_bus_index);

    let memory_dimensions = MemoryDimensions::new(4, 4);
    let user_public_values = (0..DIGEST_SIZE)
        .map(|index| F::from_usize(31 + index))
        .collect::<Vec<_>>();
    let user_leaf: Digest = user_public_values.clone().try_into().unwrap();
    let public_values_commitment =
        poseidon2_compress_with_capacity(user_leaf, [F::ZERO; DIGEST_SIZE]).0;
    let merkle_path = (0..memory_dimensions.overall_height())
        .map(|level| digest(2_000 + 20 * level as u32))
        .collect::<Vec<_>>();

    let commit_air = UserPvsCommitAir::new(
        inventory.poseidon2_compress_bus,
        merkle_root_bus,
        merkle_internal_bus,
        user_public_values.len(),
    );
    let block_air = VerifierWarpBlockPublicValuesAirV2::new(
        inventory.poseidon2_compress_bus,
        merkle_root_bus,
        block_bus,
        memory_dimensions,
        user_public_values.len(),
    )
    .unwrap();
    let history_air = VerifierWarpHistoryAirV2 {
        source_bus,
        vacc_bus,
        block_public_values_bus: block_bus,
        compress_bus: inventory.poseidon2_compress_bus,
        output_mode: if direct_endpoint_mutation.is_some() {
            VerifierWarpHistoryOutputModeV2::DirectFinalProvider {
                endpoint_bus,
                lookup_count: 1,
            }
        } else {
            VerifierWarpHistoryOutputModeV2::StandalonePublicValues
        },
    };

    let (commit_context, commit_compressions): (AirProvingContext<CpuBackend<NativeSC>>, _) =
        generate_user_pvs_commit_ctx(user_public_values.clone());
    let block_trace = generate_verifier_warp_block_public_values_trace_v2(
        &block_air,
        public_values_commitment,
        &merkle_path,
    )
    .unwrap();
    let history_digest = mutated_history_digest.unwrap_or(block_trace.public_values_digest);
    let history_trace = generate_verifier_warp_history_trace_v2(&[history_record(
        block_trace.final_memory_root,
        history_digest,
    )])
    .unwrap();

    let mut compression_inputs = commit_compressions;
    compression_inputs.extend(block_trace.compression_inputs.iter().copied());
    compression_inputs.extend(history_trace.compression_inputs.iter().copied());
    let poseidon_matrix = transcript
        .build_poseidon2_multibus_traces(vec![(Vec::new(), compression_inputs)])
        .unwrap()
        .pop()
        .unwrap();
    let poseidon_air =
        transcript.multi_bus_poseidon2_air_for_owners::<NativeSC>(&[Poseidon2BusOwner {
            permute_bus: poseidon_owner.permute_bus,
            compress_bus: poseidon_owner.compress_bus,
        }]);

    let mut airs: Vec<AirRef<NativeSC>> = vec![
        Arc::new(commit_air),
        Arc::new(block_air),
        Arc::new(history_air),
    ];
    let mut contexts = vec![
        commit_context,
        AirProvingContext::simple_no_pis(block_trace.matrix),
        AirProvingContext::new(
            Vec::new(),
            history_trace.matrix,
            if direct_endpoint_mutation.is_some() {
                vec![]
            } else {
                history_trace.public_values.to_vec()
            },
        ),
    ];
    if let Some(index) = direct_endpoint_mutation {
        let mut endpoint = history_trace.public_values.to_vec();
        if index < endpoint.len() {
            endpoint[index] += F::ONE;
        }
        airs.push(Arc::new(TestHistoryEndpointConsumerAirV2 { endpoint_bus }));
        contexts.push(AirProvingContext::simple_no_pis(RowMajorMatrix::new(
            endpoint,
            VerifierWarpHistoryPublicValuesV2::WIDTH,
        )));
    }
    airs.push(poseidon_air);
    contexts.push(AirProvingContext::simple_no_pis(poseidon_matrix));
    BlockComposition {
        airs,
        contexts,
        selected_bus_indices: vec![
            poseidon_owner.permute_bus.index(),
            poseidon_owner.compress_bus.index(),
            merkle_root_bus.index(),
            merkle_internal_bus.index(),
            block_bus_index,
            endpoint_bus_index,
        ],
    }
}

#[test]
fn block_public_values_are_authenticated_through_commit_path_and_history() {
    block_composition(None, None).check();
}

#[test]
fn history_hash_copy_forward_keeps_symbolic_degree_bounded() {
    let composition = block_composition(None, None);
    let history_air = composition.airs[2].as_ref();
    assert!(
        symbolic_max_constraint_degree(history_air) <= 8,
        "History-v2 exceeds the final proof's degree-eight profile"
    );
}

#[test]
fn changed_history_public_values_digest_breaks_the_authenticated_block_bus() {
    let bad = block_composition(Some(digest(9_999)), None);
    assert!(std::panic::catch_unwind(AssertUnwindSafe(|| bad.check())).is_err());
}

#[test]
fn direct_history_endpoint_has_zero_pvs_and_balances_complete_typed_message() {
    // `usize::MAX` selects direct mode without mutating any of the 83 fields.
    block_composition(None, Some(usize::MAX)).check();
}

#[test]
fn every_direct_history_endpoint_field_is_bound_on_the_provider_bus() {
    for index in 0..VerifierWarpHistoryPublicValuesV2::WIDTH {
        let bad = block_composition(None, Some(index));
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| bad.check())).is_err(),
            "endpoint field {index} was not constrained by the typed bus"
        );
    }
}

fn aggregate_vm_pvs(final_memory_root: Digest) -> VmPvs<F> {
    VmPvs {
        program_commit: digest(4_000),
        initial_pc: F::from_u32(11),
        final_pc: F::from_u32(19),
        exit_code: F::ZERO,
        is_terminate: F::ONE,
        initial_root: digest(4_100),
        final_root: final_memory_root,
    }
}

#[allow(clippy::type_complexity)]
fn rebound_vacc_module(
    profile: &NativeStandardVaccProfile,
    appendix_d: bool,
    start: BusIndex,
    source_opening_bus: CertifiedSwirlRawOpeningBusV19,
    replay_bus: CertifiedWarpReplayBusV19,
    history_compress_bus: HistoryPoseidon2CompressBusV19,
) -> (
    DirectAirVaccVerifierModuleV19,
    WarpReplayProducerAirV19,
    BusIndex,
) {
    let params = SystemParams::new_for_testing(10);
    let mut manager = BusIndexManager::from_next_bus_idx(start);
    let shared = BusInventory::new(&mut manager);
    let buses = NativeWarpPcdBusInventory::new(manager.next_bus_idx());
    let mut extra = BusIndexManager::from_next_bus_idx(buses.next_bus_idx());
    assert_eq!(extra.new_bus_idx(), buses.next_bus_idx());
    let history_buses = DirectAirVaccHistoryBusesV19 {
        context: DirectAirVaccContextBusV19::new(extra.new_bus_idx()),
        input: CertifiedDirectAirVaccInputBusV19::new(extra.new_bus_idx()),
    };
    let protocol_bus = NativeStandardVaccProtocolBus::new(extra.new_bus_idx());
    let end_bus = NativeStandardVaccEndBus::new(extra.new_bus_idx());
    let root_bus = NativeStandardVaccRootBus::new(extra.new_bus_idx());
    let digest_bus = NativeStandardVaccDigestBus::new(extra.new_bus_idx());
    let mut module = if appendix_d {
        DirectAirVaccVerifierModuleV19::new_shape_batched_appendix_d(
            vec![profile.clone()],
            false,
            shared,
            buses,
            history_buses,
            protocol_bus,
            end_bus,
            root_bus,
            digest_bus,
            params,
        )
        .unwrap()
    } else {
        DirectAirVaccVerifierModuleV19::new(
            profile.clone(),
            false,
            shared,
            buses,
            history_buses,
            protocol_bus,
            end_bus,
            root_bus,
            digest_bus,
            params,
        )
        .unwrap()
    };
    module.set_vacc_input_lookup_count(2);
    let producer = WarpReplayProducerAirV19 {
        log_message_len: profile.log_message_len,
        log_codeword_len: profile.log_codeword_len,
        beta_len: profile.beta_len,
        compress_bus: history_compress_bus,
        history_bus: replay_bus,
        context_bus: history_buses.context,
        swirl_opening_bus: source_opening_bus,
        vacc_input_bus: history_buses.input,
        canonical_vacc_input_bus: None,
        fresh_explicit_bus:
            crate::circuit::native_warp_history_v19::CertifiedFreshExplicitDigestBusV19::new(
                extra.new_bus_idx(),
            ),
        fresh_explicit_lookup_count: 1,
        checkpoint_bus: module.buses.transcript_checkpoint,
        batching_claim_bus: module.buses.certified_batching_claim,
        next_accumulator_digest_bus: module.buses.certified_accumulator_digest,
        setup_schedule: None,
    };
    (module, producer, extra.next_bus_idx())
}

fn authenticated_producer_composition(splice_source_opening_point: bool) -> BlockComposition {
    // Build the block-PV endpoint first so the aggregate VmPvs carried in the
    // certified fresh beta has the exact terminal memory root.
    let memory_dimensions = MemoryDimensions::new(4, 4);
    let user_public_values = (0..DIGEST_SIZE)
        .map(|index| F::from_usize(51 + index))
        .collect::<Vec<_>>();
    let user_leaf: Digest = user_public_values.clone().try_into().unwrap();
    let public_values_commitment =
        poseidon2_compress_with_capacity(user_leaf, [F::ZERO; DIGEST_SIZE]).0;
    let merkle_path = (0..memory_dimensions.overall_height())
        .map(|level| digest(5_000 + 20 * level as u32))
        .collect::<Vec<_>>();
    let dummy_compress = openvm_recursion_circuit::bus::Poseidon2CompressBus::new(0);
    let dummy_root = MerkleRootBus::new(1);
    let dummy_block_bus = VerifierWarpBlockPublicValuesBusV2::new(2);
    let dummy_block_air = VerifierWarpBlockPublicValuesAirV2::new(
        dummy_compress,
        dummy_root,
        dummy_block_bus,
        memory_dimensions,
        user_public_values.len(),
    )
    .unwrap();
    let preliminary_block = generate_verifier_warp_block_public_values_trace_v2(
        &dummy_block_air,
        public_values_commitment,
        &merkle_path,
    )
    .unwrap();
    let vm_pvs = aggregate_vm_pvs(preliminary_block.final_memory_root);

    // The source reduction is built first from one real fixed-message trace.
    // It returns the exact BabyBear message and retained Merkle authority that
    // Appendix-D VACC must consume; no root, point, or value is synthesized or
    // rewritten after either proof is generated.
    let (_engine, mut logup_module, logup_records, appendix_d_source) =
        retained_fixed_multi_air_fixture(vm_pvs.as_slice().to_vec());
    logup_module
        .attach_history_v2_active_count_projection()
        .unwrap();
    let fixed_boundary = &logup_records.boundary_shards[0];
    assert_eq!(fixed_boundary.source_root, appendix_d_source.root);
    let vacc_fixture = appendix_d_fixture_for_fixed_source(
        Arc::clone(&appendix_d_source.relation),
        appendix_d_source.message.clone(),
        appendix_d_source.explicit.clone(),
        appendix_d_source.root,
        appendix_d_source.base_prover_data.clone(),
        appendix_d_source.base_codeword.clone(),
        fixed_boundary.opening_point.clone(),
        fixed_boundary.opening_value,
        fixed_boundary.source_forest_root,
        fixed_boundary.segment_openings_digest,
    );
    assert_eq!(
        vacc_fixture.producer.relation_digest,
        appendix_d_source.relation.description().relation_digest
    );
    assert_eq!(vacc_fixture.producer.fresh_root, appendix_d_source.root);
    let logup_packet = logup_module
        .generate_proving_contexts_without_poseidon::<NativeSC>(&logup_records)
        .unwrap();
    assert_eq!(logup_packet.logup_producers.len(), 1);
    let logup_history = logup_module.history_buses();

    let mut manager = BusIndexManager::from_next_bus_idx(logup_module.next_bus_idx());
    let replay_bus = CertifiedWarpReplayBusV19::new(manager.new_bus_idx());
    let logup_bus = LogUpOnlyHistoryBusV19::new(manager.new_bus_idx());
    let (vacc_module, replay_producer_air, next_bus_idx) = rebound_vacc_module(
        &vacc_fixture.profile,
        true,
        manager.next_bus_idx(),
        logup_history.opening,
        replay_bus,
        logup_history.compress,
    );
    manager = BusIndexManager::from_next_bus_idx(next_bus_idx);
    let fixed_source_bus = CertifiedFixedMultiAirSourceBusV2::new(manager.new_bus_idx());
    let fixed_setup_opening_bus = FixedSetupOpeningCertificateBusV2::new(manager.new_bus_idx());
    let active_count_bus = VerifierWarpCertifiedActiveCountBusV2::new(manager.new_bus_idx());
    let source_bus = VerifierWarpSourceCertificateBusV2::new(manager.new_bus_idx());
    let vacc_bus = VerifierWarpVaccCertificateBusV2::new(manager.new_bus_idx());
    let block_bus = VerifierWarpBlockPublicValuesBusV2::new(manager.new_bus_idx());
    let merkle_root_bus = MerkleRootBus::new(manager.new_bus_idx());
    let merkle_internal_bus = MerkleTreeInternalBus::new(manager.new_bus_idx());

    let logup_producer_air = LogUpOnlyProducerAirV19 {
        compress_bus: logup_history.compress,
        history_bus: logup_bus,
        endpoint_bus: logup_history.endpoint,
        checkpoint_bus: logup_module.prefix_bridge_air.checkpoint_bus,
    };
    let logup_producer_trace =
        generate_logup_only_producer_trace_v19(&logup_packet.logup_producers, 2).unwrap();
    let logup_message = logup_producer_trace.messages[0].clone();
    let fixed_boundary_profile = FixedMultiAirSourceBoundaryProfileV2 {
        segment_start: 0,
        app_vk_digest: logup_module.child_vk.pre_hash,
        relation_digest: vacc_fixture.producer.relation_digest,
        log_message_len: vacc_fixture.profile.log_message_len as u8,
        log_codeword_len: vacc_fixture.profile.log_codeword_len as u8,
        active_child_counts: Arc::from([1]),
    };
    let fixed_boundary_air = FixedMultiAirSourceBoundaryAirV2 {
        profile: fixed_boundary_profile,
        arithmetic_bus: logup_module.fixed_boundary_inputs.arithmetic,
        source_leaf_bus: logup_module.fixed_boundary_inputs.source_leaf,
        raw_opening_bus: logup_module.fixed_boundary_inputs.raw_opening,
        certified_endpoint_bus: logup_history.endpoint,
        certified_opening_bus: logup_history.opening,
        certified_source_bus: fixed_source_bus,
        certified_source_lookup_count: 2,
    };
    let fixed_boundary = &logup_records.boundary_shards[0];
    let fixed_boundary_record = FixedMultiAirSourceBoundaryRecordV2 {
        source_forest_root: fixed_boundary.source_forest_root,
        segment_openings_digest: fixed_boundary.segment_openings_digest,
        source_root: fixed_boundary.source_root,
        point: fixed_boundary
            .opening_point
            .iter()
            .copied()
            .map(ext_array)
            .collect(),
        value: ext_array(fixed_boundary.opening_value),
        verifier_endpoint: ext_array(fixed_boundary.verifier_endpoint),
    };
    let fixed_boundary_trace = generate_fixed_multi_air_source_boundary_trace_v2(
        &fixed_boundary_air,
        core::slice::from_ref(&fixed_boundary_record),
    )
    .unwrap();

    let vacc_record = vacc_fixture.producer.clone();
    assert_eq!(
        vacc_record.source_forest_root,
        logup_message.source_forest_root
    );
    assert_eq!(
        vacc_record.segment_openings_digest,
        logup_message.segment_openings_digest
    );
    assert_eq!(vacc_record.fresh_root, fixed_boundary_record.source_root);
    assert_eq!(vacc_record.opening_point, fixed_boundary_record.point);
    assert_eq!(vacc_record.opening_value, fixed_boundary_record.value);
    let vacc_verifier_record = DirectAirVaccVerifierRecordV19 {
        producer: &vacc_record,
        verification: &vacc_fixture.verification,
        transcript: &vacc_fixture.transcript,
        prior: None,
    };
    let vacc_packet = vacc_module
        .generate_appendix_d_traces_for_shared_poseidon(
            &vacc_fixture.config,
            &[vacc_verifier_record],
        )
        .unwrap();
    let replay_trace = generate_warp_replay_producer_trace_v19(
        &replay_producer_air,
        core::slice::from_ref(&vacc_record),
        2,
    )
    .unwrap();
    let replay_message = replay_trace.messages[0].clone();
    let fresh_input =
        certified_direct_air_vacc_input_message_v19(&replay_producer_air, &vacc_record).unwrap();
    let mut fixed_source_message = CertifiedFixedMultiAirSourceMessageV2 {
        proof_index: F::ZERO,
        segment_index_lo: F::ZERO,
        segment_index_hi: F::ZERO,
        active_child_count: F::ONE,
        app_vk_digest: logup_module.child_vk.pre_hash,
        relation_digest: vacc_record.relation_digest,
        source_forest_root: fixed_boundary_record.source_forest_root,
        segment_openings_digest: fixed_boundary_record.segment_openings_digest,
        source_root: fixed_boundary_record.source_root,
        point_len: F::from_usize(fixed_boundary_record.point.len()),
        point: core::array::from_fn(|index| {
            fixed_boundary_record
                .point
                .get(index)
                .copied()
                .unwrap_or([F::ZERO; D_EF])
        }),
        value: fixed_boundary_record.value,
        verifier_endpoint: fixed_boundary_record.verifier_endpoint,
    };
    if splice_source_opening_point {
        fixed_source_message.point[0][0] += F::ONE;
    }

    let setup_openings_digest = digest(5_700);
    let fixed_setup_opening = FixedSetupOpeningCertificateMessageV2 {
        protocol_version: F::from_u32(FIXED_SETUP_OPENING_PROTOCOL_V2),
        proof_index: F::ZERO,
        canonical_claim_count: F::ONE,
        source_relation_vk_digest: logup_message.app_vk_digest,
        setup_openings_digest,
    };
    let active_count_profile_digest = digest(5_800);
    let native_active_count_profile = logup_module
        .active_count_transcript_air
        .as_ref()
        .unwrap()
        .profile
        .clone();
    let active_count_profile = VerifierWarpActiveCountProfileV2 {
        relation_digest: native_active_count_profile.relation_digest,
        profile_digest: active_count_profile_digest,
        profile_segment_count: native_active_count_profile.segment_count(),
        profile_batch_count: native_active_count_profile.active_child_counts.len() as u64,
        batch_arity: VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 as u64,
        final_batch_active_count: u64::from(
            *native_active_count_profile
                .active_child_counts
                .last()
                .unwrap(),
        ),
        trace_heights: native_active_count_profile
            .trace_heights
            .iter()
            .copied()
            .map(u64::from)
            .collect::<Vec<_>>()
            .into(),
        vm_pvs_air_id: native_active_count_profile.vm_pvs_air_id,
        is_valid_common_main_column: native_active_count_profile.is_valid_common_main_column,
        is_valid_message_block_start: native_active_count_profile.is_valid_message_block_start,
        log_height: native_active_count_profile.vm_pvs_log_height,
        log_message_len: vacc_fixture.profile.log_message_len as u8,
    };
    let active_count_projection_air = VerifierWarpActiveCountProjectionAirV2 {
        profile: active_count_profile,
        transition_count: u32::try_from(native_active_count_profile.active_child_counts.len())
            .unwrap(),
        segment_start: native_active_count_profile.segment_start,
        source_log_codeword_len: vacc_fixture.profile.log_codeword_len as u8,
        auxiliary_challenge_bus: logup_module
            .active_count_transcript_air
            .as_ref()
            .unwrap()
            .challenge_bus,
        stream_cursor_bus: logup_module.finalization_air.stream_cursor_bus,
        raw_opening_bus: logup_module.fixed_boundary_inputs.raw_opening,
        fixed_source_bus,
        certified_count_bus: active_count_bus,
    };
    let mapped_source = &logup_records.mapped_sources[0];
    let reduction = &logup_records.reductions[0];
    let reduction_end_tidx =
        reduction.start_tidx + u32::try_from((4 * reduction.point.len() + 1) * D_EF).unwrap();
    let active_count_projection =
        generate_verifier_warp_active_count_projection_trace_and_messages_v2(
            &active_count_projection_air,
            &[VerifierWarpActiveCountProjectionRecordV2 {
                fixed_source: fixed_source_message.clone(),
                batching_coefficient: mapped_source.active_count_challenge,
                reduction_end_tidx,
            }],
        )
        .unwrap();
    let active_count = active_count_projection.messages[0].clone();
    let active_count_projection_trace = active_count_projection.matrix;
    let c1_authority_air = BridgeC1AuthorityAirV2 {
        fixed_setup_bus: fixed_setup_opening_bus,
    };
    let c1_authority_trace = bridge_c1_authority_trace_v2(&fixed_setup_opening);

    let protocol_digest = digest(6_000);
    let bridge_air = VerifierWarpProducerBridgeAirV2 {
        segment_start: 0,
        fixed_public_values: VerifierWarpFixedPublicValuesProfileV2 {
            log_constraints: vacc_fixture.profile.log_constraints,
            sources: core::iter::once(VerifierWarpFixedPublicValueSourceV2::TrustedConstant(
                F::ONE,
            ))
            .chain(
                (0..VmPvs::<u8>::width())
                    .map(VerifierWarpFixedPublicValueSourceV2::VmPvsCoordinate),
            )
            .collect::<Vec<_>>()
            .into(),
            batch_count: 1,
        },
        protocol_digest,
        admitted_relation_digest: vacc_record.relation_digest,
        expected_key_digest: replay_message.key_digest,
        source_relation_vk_digest: logup_message.app_vk_digest,
        source_log_message_len: vacc_fixture.profile.log_message_len,
        source_log_codeword_len: vacc_fixture.profile.log_codeword_len,
        active_count_profile_digest,
        logup_bus,
        fixed_source_bus,
        vacc_input_bus: vacc_module.history_buses.input,
        fresh_explicit_bus: replay_producer_air.fresh_explicit_bus,
        replay_bus,
        fixed_public_values_bus: logup_module.fixed_public_values_bus,
        fixed_setup_opening_bus: Some(fixed_setup_opening_bus),
        active_count_bus,
        source_bus,
        vacc_bus,
    };
    let bridge_trace = generate_verifier_warp_producer_bridge_trace_v2(
        &bridge_air,
        &[VerifierWarpProducerBridgeRecordV2 {
            vm_pvs,
            logup: logup_message.clone(),
            fixed_source: fixed_source_message,
            fresh_input,
            fresh_explicit: replay_trace.fresh_explicit_messages[0].clone(),
            replay: replay_message.clone(),
            fixed_setup_opening: fixed_setup_opening.clone(),
            active_count: active_count.clone(),
        }],
    )
    .unwrap();

    let block_air = VerifierWarpBlockPublicValuesAirV2::new(
        logup_history.compress,
        merkle_root_bus,
        block_bus,
        memory_dimensions,
        user_public_values.len(),
    )
    .unwrap();
    let block_trace = generate_verifier_warp_block_public_values_trace_v2(
        &block_air,
        public_values_commitment,
        &merkle_path,
    )
    .unwrap();
    assert_eq!(block_trace.final_memory_root, vm_pvs.final_root);
    let history_air = VerifierWarpHistoryAirV2 {
        source_bus,
        vacc_bus,
        block_public_values_bus: block_bus,
        compress_bus: logup_history.compress,
        output_mode: VerifierWarpHistoryOutputModeV2::StandalonePublicValues,
    };
    let history_trace =
        generate_verifier_warp_history_trace_v2(&[VerifierWarpHistoryTransitionRecordV2 {
            protocol_version: VERIFIER_WARP_HISTORY_PROTOCOL_V2,
            vacc_input_arity: VERIFIER_WARP_VACC_INPUT_ARITY_V2 as u8,
            protocol_digest,
            relation_digest: vacc_record.relation_digest,
            batch_index: 0,
            active_child_count: 1,
            children: {
                let mut children =
                    [VerifierWarpChildRecordV2::padding(); VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2];
                children[0] = aggregate_child(vm_pvs.final_root, true);
                children
            },
            program_commitment: vm_pvs.program_commit,
            prior_accumulator_digest: replay_message.previous_accumulator_digest,
            source_accumulator_digest: replay_message.fresh_instance_digest,
            output_accumulator_digest: replay_message.next_accumulator_digest,
            source_commitment_root: active_count.source_root,
            external_logup_gkr_digest: logup_message.checkpoint_digest,
            source_functional_digest: replay_message.opening_claim_digest,
            setup_openings_digest,
            source_statement_digest: replay_message.replay_endpoint_digest,
            transition_transcript_digest: replay_message.replay_binding_digest,
            public_values_digest: block_trace.public_values_digest,
        }])
        .unwrap();

    let user_commit_air = UserPvsCommitAir::new(
        logup_history.compress,
        merkle_root_bus,
        merkle_internal_bus,
        user_public_values.len(),
    );
    let (user_commit_context, user_compressions) =
        generate_user_pvs_commit_ctx::<NativeSC>(user_public_values);

    let logup_permutations = logup_packet.poseidon.permutation_inputs;
    let mut logup_compressions = logup_packet.poseidon.compression_inputs;
    logup_compressions.extend(logup_producer_trace.compression_inputs.iter().copied());
    logup_compressions.extend(replay_trace.compression_inputs.iter().copied());
    logup_compressions.extend(user_compressions);
    logup_compressions.extend(block_trace.compression_inputs.iter().copied());
    logup_compressions.extend(history_trace.compression_inputs.iter().copied());
    let poseidon_context = logup_module
        .build_shared_poseidon_context::<NativeSC>(vec![
            (logup_permutations, logup_compressions),
            (
                vacc_packet.poseidon_permutation_inputs,
                vacc_packet.poseidon_compression_inputs,
            ),
        ])
        .unwrap();
    let owners = [
        logup_module.shared_poseidon_owner(),
        vacc_module.transcript.poseidon2_bus_owner(),
    ];
    let poseidon_air = logup_module.shared_poseidon_air::<NativeSC>(&owners);

    let mut airs = logup_module.airs_without_poseidon::<NativeSC>();
    let mut contexts = logup_packet.contexts;
    airs.extend(vacc_module.airs_without_poseidon::<NativeSC>());
    contexts.extend(
        vacc_packet
            .traces
            .into_iter()
            .map(AirProvingContext::simple_no_pis),
    );
    airs.extend([
        Arc::new(fixed_boundary_air) as AirRef<NativeSC>,
        Arc::new(logup_producer_air) as AirRef<NativeSC>,
        Arc::new(replay_producer_air),
        Arc::new(active_count_projection_air),
        Arc::new(c1_authority_air),
        Arc::new(bridge_air),
        Arc::new(user_commit_air),
        Arc::new(block_air),
        Arc::new(history_air),
        poseidon_air,
    ]);
    contexts.extend([
        AirProvingContext::simple_no_pis(fixed_boundary_trace),
        AirProvingContext::simple_no_pis(logup_producer_trace.matrix),
        AirProvingContext::simple_no_pis(replay_trace.matrix),
        AirProvingContext::simple_no_pis(active_count_projection_trace),
        AirProvingContext::simple_no_pis(c1_authority_trace),
        AirProvingContext::simple_no_pis(bridge_trace),
        user_commit_context,
        AirProvingContext::simple_no_pis(block_trace.matrix),
        AirProvingContext::new(
            Vec::new(),
            history_trace.matrix,
            history_trace.public_values.to_vec(),
        ),
        poseidon_context,
    ]);
    BlockComposition {
        airs,
        contexts,
        selected_bus_indices: Vec::new(),
    }
}

#[test]
fn real_v19_producers_balance_against_v2_bridge_and_history() {
    authenticated_producer_composition(false).check();
}

#[test]
fn real_appendix_d_composition_rejects_source_vacc_opening_point_splice() {
    let rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        authenticated_producer_composition(true).check();
    }));
    assert!(rejected.is_err());
}

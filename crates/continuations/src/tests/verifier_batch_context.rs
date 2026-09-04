//! Real app-proof fixture for the complete-verifier WARP relation.
//!
//! The recursive leaf prover already exposes the desired protocol boundary:
//! `generate_proving_ctx` arithmetizes complete app-STARK verification, while
//! `agg_prove` additionally calls `engine.prove`. This test deliberately stops
//! at that boundary and records the fixed capacity-four verifier-batch trace
//! shape. Partial batches must inhabit this same relation class.

use std::{borrow::BorrowMut, cmp::Reverse, sync::Arc};

use eyre::{eyre, Result};
use openvm_circuit::arch::{
    instructions::exe::VmExe, ContinuationVmProver, VirtualMachine, VmInstance,
};
use openvm_circuit_primitives::StructReflectionHelper;
use openvm_recursion_circuit::system::VerifierSubCircuit;
use openvm_riscv_circuit::Rv64ImCpuBuilder;
use openvm_riscv_transpiler::{
    Rv64ITranspilerExtension, Rv64IoTranspilerExtension, Rv64MTranspilerExtension,
};
use openvm_stark_backend::{
    air_builders::symbolic::{symbolic_variable::Entry, SymbolicExpressionNode},
    hasher::MerkleHasher,
    keygen::types::MultiStarkProvingKey,
    native_warp::{
        DirectAirCodeClass, DirectAirFixedTrace, DirectAirPesatIndex, DirectAirPesatInstance,
        DirectAirPublicSchema, FixedMultiAirPesatIndex,
    },
    p3_matrix::Matrix,
    StarkEngine, StarkProtocolConfig, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config, BabyBearPoseidon2CpuEngine, Digest, DuplexSponge, F,
};
use openvm_transpiler::{
    elf::Elf, openvm_platform::memory::MEM_SIZE, transpiler::Transpiler, FromElf,
};
use openvm_verify_stark_host::pvs::{VmPvs, VM_PVS_AIR_ID};
use p3_field::PrimeCharacteristicRing;

use super::{app_system_params, leaf_system_params, test_rv64im_config};
use crate::{
    circuit::inner::{vm_pvs::VmPvsCols, InnerTraceGenImpl, ProofsType},
    prover::{engine_device_ctx, ChildVkKind, InnerAggregationProver},
    SC,
};

const MAX_PROOFS: usize = 4;
const FIB_INPUT_FOR_FOUR_SEGMENTS: u64 = 400_000;
const FIXTURE_SEGMENTATION_MAX_MEMORY: usize = 1 << 30;
const EXPECTED_ACTIVE_AIRS: usize = 43;
const EXPECTED_RELATION_DEGREE: usize = 5;
const EXPECTED_EXACT_RELATION_DEGREE: usize = 4;
const EXPECTED_EXPLICIT_LEN: usize = 95;
const EXPECTED_RAW_MESSAGE_LEN: usize = 102_577_326;
const EXPECTED_PADDED_MESSAGE_LEN: usize = 134_217_728;
const EXPECTED_FIXED_CACHED_TRACE_CELLS: usize = 196_608;
const EXPECTED_AIR_CONSTRAINT_LEN: usize = 182_854_190;
const EXPECTED_PADDING_CONSTRAINT_LEN: usize = 31_640_402;
const EXPECTED_REAL_CONSTRAINT_LEN: usize = 214_494_592;
const EXPECTED_PADDED_CONSTRAINT_LEN: usize = 268_435_456;
const VERIFIER_WARP_PUBLIC_SCHEMA_TAG: u64 = 0x5657_5055_424c_0002;

type Engine = BabyBearPoseidon2CpuEngine<DuplexSponge>;
type LeafProver = InnerAggregationProver<
    openvm_cpu_backend::CpuBackend<SC>,
    VerifierSubCircuit<MAX_PROOFS>,
    InnerTraceGenImpl,
>;

/// Verifier-visible trace shape. Cached columns are commitment-bound by the
/// child VK and also remain raw witness columns in the current backend index.
#[derive(Clone, Debug, Eq, PartialEq)]
struct VerifierBatchTraceShape {
    air_id: usize,
    height: usize,
    width: usize,
    cached_mains: usize,
    public_values: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CanonicalDirectRegion {
    air_id: usize,
    height: usize,
    cached_main_widths: Vec<usize>,
    common_main_width: usize,
    message_start: usize,
    message_len: usize,
    constraints_per_row: usize,
    real_constraint_count: usize,
    constraint_block_len: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CanonicalDirectLayout {
    regions: Vec<CanonicalDirectRegion>,
    explicit_len: usize,
    raw_message_len: usize,
    padded_message_len: usize,
    fixed_cached_trace_cells: usize,
    air_constraint_len: usize,
    padding_constraint_len: usize,
    real_constraint_len: usize,
    occupied_constraint_len: usize,
    padded_constraint_len: usize,
    relation_degree: usize,
    exact_relation_degree: usize,
}

fn direct_code_class(params: &SystemParams, raw_message_len: usize) -> Result<DirectAirCodeClass> {
    let log_message_len = raw_message_len
        .checked_next_power_of_two()
        .ok_or_else(|| eyre!("padded direct message length"))?
        .ilog2() as usize;
    let log_codeword_len = log_message_len
        .checked_add(params.log_blowup)
        .ok_or_else(|| eyre!("direct codeword length"))?;
    let codeword_len = 1usize
        .checked_shl(log_codeword_len as u32)
        .ok_or_else(|| eyre!("direct codeword length"))?;
    Ok(DirectAirCodeClass {
        log_message_len: log_message_len.try_into()?,
        log_blowup: params.log_blowup.try_into()?,
        log_codeword_len: log_codeword_len.try_into()?,
        initial_folding_factor: 0,
        rows_per_query: params.fold_rows_per_query().min(codeword_len).try_into()?,
    })
}

fn compile_backend_fixed_multi_air_index(
    context: &openvm_stark_backend::prover::ProvingContext<openvm_cpu_backend::CpuBackend<SC>>,
    pk: &MultiStarkProvingKey<SC>,
) -> Result<FixedMultiAirPesatIndex<F, Digest>> {
    let vk = pk.get_vk();
    let config = BabyBearPoseidon2Config::default_from_params(pk.params.clone());
    let mut ordered = context.per_trace.iter().collect::<Vec<_>>();
    ordered.sort_by_key(|(air_id, air)| (Reverse(air.common_main.height()), *air_id));
    let mut direct = Vec::with_capacity(ordered.len());
    for &(air_id, air) in &ordered {
        let air_vk = &vk.inner.per_air[*air_id];
        let height = air.common_main.height();
        let fixed = if let Some(data) = pk.per_air[*air_id].preprocessed_data.as_ref() {
            let matrix = data.mat_view(0).to_row_major_matrix();
            Some(DirectAirFixedTrace {
                width: u32::try_from(Matrix::width(&matrix))?,
                values: matrix.values,
            })
        } else {
            None
        };
        let raw_message_len = height
            .checked_mul(air_vk.params.width.main_widths().iter().sum())
            .ok_or_else(|| eyre!("local raw message length"))?;
        let public_schema = DirectAirPublicSchema {
            public_values_len: air_vk.params.num_public_values.try_into()?,
            boundary_values_len: 0,
            schema_digest: config.hasher().hash_slice(&[
                F::from_u64(VERIFIER_WARP_PUBLIC_SCHEMA_TAG),
                F::from_usize(*air_id),
                F::from_usize(air_vk.params.num_public_values),
                F::ZERO,
            ]),
        };
        direct.push(DirectAirPesatIndex::from_verifying_key(
            config.hasher(),
            pk.vk_pre_hash,
            *air_id,
            height.ilog2() as usize,
            air_vk,
            fixed,
            public_schema,
            direct_code_class(&pk.params, raw_message_len)?,
        )?);
    }
    let raw_message_len = direct
        .iter()
        .try_fold(0usize, |total, relation| {
            total.checked_add(relation.raw_witness_len())
        })
        .ok_or_else(|| eyre!("global raw message length"))?;
    Ok(FixedMultiAirPesatIndex::from_direct_air_regions(
        config.hasher(),
        pk.vk_pre_hash,
        direct,
        direct_code_class(&pk.params, raw_message_len)?,
    )?)
}

fn canonical_direct_layout(
    context: &openvm_stark_backend::prover::ProvingContext<openvm_cpu_backend::CpuBackend<SC>>,
    index: &FixedMultiAirPesatIndex<F, Digest>,
) -> CanonicalDirectLayout {
    let description = index.description();
    let regions = description
        .regions
        .iter()
        .map(|region| CanonicalDirectRegion {
            air_id: region.air_id as usize,
            height: 1usize << region.log_height,
            cached_main_widths: region
                .trace_layout
                .cached_main_widths
                .iter()
                .map(|&width| width as usize)
                .collect(),
            common_main_width: region.trace_layout.common_main_width as usize,
            message_start: region.message_offset as usize,
            message_len: region.raw_message_len as usize,
            constraints_per_row: region.constraints_per_row as usize,
            real_constraint_count: region.constraint_count as usize,
            constraint_block_len: region.constraint_block_len as usize,
        })
        .collect::<Vec<_>>();
    let fixed_cached_trace_cells = context
        .per_trace
        .iter()
        .flat_map(|(_, air)| &air.cached_mains)
        .map(|cached| cached.trace.height() * cached.trace.width())
        .sum();
    let air_constraint_len = regions
        .iter()
        .map(|region| region.real_constraint_count)
        .sum();
    let padding_constraint_len = description.padding_constraint_count as usize;
    let occupied_constraint_len = description
        .regions
        .iter()
        .map(|region| region.constraint_offset + region.constraint_block_len)
        .chain(core::iter::once(
            description.padding_constraint_offset + description.padding_constraint_block_len,
        ))
        .max()
        .unwrap_or(0) as usize;
    CanonicalDirectLayout {
        regions,
        explicit_len: description.explicit_len as usize,
        raw_message_len: description.raw_message_len as usize,
        padded_message_len: description.padded_message_len as usize,
        fixed_cached_trace_cells,
        air_constraint_len,
        padding_constraint_len,
        real_constraint_len: air_constraint_len + padding_constraint_len,
        occupied_constraint_len,
        padded_constraint_len: description.padded_constraint_count as usize,
        relation_degree: description.warp_degree_envelope as usize,
        exact_relation_degree: description.exact_max_degree as usize,
    }
}

fn backend_dense_relation_is_satisfied(
    context: &openvm_stark_backend::prover::ProvingContext<openvm_cpu_backend::CpuBackend<SC>>,
    index: &FixedMultiAirPesatIndex<F, Digest>,
) -> Result<bool> {
    let mut instances = Vec::with_capacity(index.region_count());
    let mut witnesses = Vec::with_capacity(index.region_count());
    for (ordinal, region) in index.description().regions.iter().enumerate() {
        let air = context
            .per_trace
            .iter()
            .find_map(|(air_id, air)| (*air_id == region.air_id as usize).then_some(air))
            .ok_or_else(|| eyre!("missing backend region AIR {}", region.air_id))?;
        let cached = air
            .cached_mains
            .iter()
            .map(|cached| cached.trace.clone())
            .collect::<Vec<_>>();
        witnesses.push(
            index
                .region_relation(ordinal)
                .ok_or_else(|| eyre!("missing backend region {ordinal}"))?
                .witness_from_row_major_parts(&cached, Some(&air.common_main))?,
        );
        instances.push(DirectAirPesatInstance {
            public_values: air.public_values.clone(),
            boundary_values: Vec::new(),
        });
    }
    let witness = index.stack_witnesses(&witnesses)?;
    Ok(index.is_satisfied_reference(
        &openvm_stark_backend::native_warp::FixedMultiAirPesatInstance { regions: instances },
        &witness,
    )?)
}

fn backend_region_is_satisfied(
    context: &openvm_stark_backend::prover::ProvingContext<openvm_cpu_backend::CpuBackend<SC>>,
    index: &FixedMultiAirPesatIndex<F, Digest>,
    air_id: usize,
) -> Result<bool> {
    let ordinal = index
        .description()
        .regions
        .iter()
        .position(|region| region.air_id as usize == air_id)
        .ok_or_else(|| eyre!("missing backend region AIR {air_id}"))?;
    let relation = index
        .region_relation(ordinal)
        .ok_or_else(|| eyre!("missing backend region {ordinal}"))?;
    let air = context
        .per_trace
        .iter()
        .find_map(|(candidate, air)| (*candidate == air_id).then_some(air))
        .ok_or_else(|| eyre!("missing context AIR {air_id}"))?;
    let cached = air
        .cached_mains
        .iter()
        .map(|cached| cached.trace.clone())
        .collect::<Vec<_>>();
    let witness = relation.witness_from_row_major_parts(&cached, Some(&air.common_main))?;
    Ok(relation.is_satisfied_reference(
        &DirectAirPesatInstance {
            public_values: air.public_values.clone(),
            boundary_values: Vec::new(),
        },
        &witness,
    )?)
}

/// Read occupancy from the genuine VM-PVS common-main trace using the same
/// reflected column identity consumed by the SDK's ephemeral WARP source
/// claim. This deliberately rejects non-Boolean entries instead of treating
/// arbitrary field values as host counts.
fn genuine_vm_pvs_active_occupancy(
    context: &openvm_stark_backend::prover::ProvingContext<openvm_cpu_backend::CpuBackend<SC>>,
) -> Result<(usize, usize)> {
    let columns = VmPvsCols::<u8>::struct_reflection()
        .ok_or_else(|| eyre!("missing VmPvsCols reflection"))?;
    let mut matching = columns
        .iter()
        .enumerate()
        .filter_map(|(index, name)| (name == "is_valid").then_some(index));
    let is_valid_column = matching
        .next()
        .ok_or_else(|| eyre!("missing reflected VmPvsCols::is_valid"))?;
    if matching.next().is_some() || columns.len() != VmPvsCols::<u8>::width() {
        return Err(eyre!("ambiguous VmPvsCols reflection"));
    }
    let vm_pvs = context
        .per_trace
        .iter()
        .find_map(|(air_id, air)| (*air_id == VM_PVS_AIR_ID).then_some(air))
        .ok_or_else(|| eyre!("missing VM-PVS AIR"))?;
    let width = Matrix::width(&vm_pvs.common_main);
    let height = vm_pvs.common_main.height();
    if width < VmPvsCols::<u8>::width() || vm_pvs.common_main.values.len() != width * height {
        return Err(eyre!("invalid VM-PVS common-main shape"));
    }
    let mut occupancy = 0usize;
    for row in 0..height {
        match vm_pvs.common_main.values[row * width + is_valid_column] {
            value if value == F::ZERO => {}
            value if value == F::ONE => occupancy += 1,
            _ => return Err(eyre!("non-Boolean VM-PVS is_valid")),
        }
    }
    Ok((occupancy, is_valid_column))
}

#[test]
fn four_real_app_proofs_generate_the_fixed_leaf_verifier_batch_shape() -> Result<()> {
    let mut config = test_rv64im_config();
    config.rv64i.system.segmentation_max_memory = FIXTURE_SEGMENTATION_MAX_MEMORY;
    let elf = Elf::decode(
        include_bytes!("../../programs/examples/fibonacci.elf"),
        MEM_SIZE as u32,
    )?;
    let exe = VmExe::from_elf(
        elf,
        Transpiler::<F>::default()
            .with_extension(Rv64ITranspilerExtension)
            .with_extension(Rv64MTranspilerExtension)
            .with_extension(Rv64IoTranspilerExtension),
    )?;
    let input = FIB_INPUT_FOR_FOUR_SEGMENTS
        .to_le_bytes()
        .map(F::from_u8)
        .to_vec();

    let app_engine = Engine::new(app_system_params());
    let (vm, app_pk) = VirtualMachine::new_with_keygen(app_engine, Rv64ImCpuBuilder, config)?;
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    let mut instance = VmInstance::new(vm, exe.into(), cached_program_trace)?;
    let plan = instance.plan_continuations_native_warp(vec![input.clone()])?;
    assert_eq!(
        plan.segment_count(),
        MAX_PROOFS,
        "fixture's metered schedule must contain exactly four app proofs"
    );
    let app_proof = instance.prove(vec![input])?;
    assert_eq!(
        app_proof.per_segment.len(),
        MAX_PROOFS,
        "fixture must exercise the full capacity-four leaf-verifier batch class"
    );

    // Prove that these are real, independently accepted app STARK proofs
    // before using their verifier traces as the fixture anchor.
    let app_vk = Arc::new(app_pk.get_vk());
    let verify_engine = Engine::new(app_vk.inner.params.clone());
    for proof in &app_proof.per_segment {
        verify_engine.verify(&app_vk, proof)?;
    }

    let leaf_prover = LeafProver::new::<Engine>(app_vk.clone(), leaf_system_params(), false, None);
    let leaf_engine = Engine::new(leaf_prover.get_pk().params.clone());
    let context = leaf_prover.generate_proving_ctx(
        &app_proof.per_segment,
        ChildVkKind::App,
        ProofsType::Vm,
        None,
        engine_device_ctx(&leaf_engine),
    );

    let shape = context
        .per_trace
        .iter()
        .map(|(air_id, air)| VerifierBatchTraceShape {
            air_id: *air_id,
            height: air.common_main.height(),
            width: air.common_main.width(),
            cached_mains: air.cached_mains.len(),
            public_values: air.public_values.len(),
        })
        .collect::<Vec<_>>();

    eprintln!("capacity-four leaf verifier-batch trace shape: {shape:#?}");
    assert!(!shape.is_empty());
    assert!(shape.iter().all(|entry| entry.height.is_power_of_two()));
    assert!(shape.windows(2).all(|pair| pair[0].air_id < pair[1].air_id));

    // Re-running trace generation must not change the relation class. This
    // deliberately does not call `engine.prove`: the WARP source witness is
    // the complete verifier execution trace, not an aggregate STARK proof.
    let mut second_context = leaf_prover.generate_proving_ctx(
        &app_proof.per_segment,
        ChildVkKind::App,
        ProofsType::Vm,
        None,
        engine_device_ctx(&leaf_engine),
    );
    let second_shape = second_context
        .per_trace
        .iter()
        .map(|(air_id, air)| VerifierBatchTraceShape {
            air_id: *air_id,
            height: air.common_main.height(),
            width: air.common_main.width(),
            cached_mains: air.cached_mains.len(),
            public_values: air.public_values.len(),
        })
        .collect::<Vec<_>>();
    assert_eq!(shape, second_shape);

    let aggregation_pk = leaf_prover.get_pk();
    let aggregation_vk = aggregation_pk.get_vk();
    assert!(
        aggregation_vk.inner.per_air.iter().all(|air| air
            .symbolic_constraints
            .constraints
            .nodes
            .iter()
            .all(|node| !matches!(
                node,
                SymbolicExpressionNode::Variable(variable)
                    if variable.entry == Entry::Challenge
            ))),
        "the fixed verifier relation must not leave transcript challenges as unconstrained inputs"
    );
    let backend_index = compile_backend_fixed_multi_air_index(&context, &aggregation_pk)?;
    let layout = canonical_direct_layout(&context, &backend_index);
    assert!(
        backend_dense_relation_is_satisfied(&context, &backend_index)?,
        "the full four-proof verifier execution must satisfy the compiled backend relation"
    );
    let vm_pvs_region = backend_index
        .description()
        .regions
        .iter()
        .find(|region| region.air_id as usize == VM_PVS_AIR_ID)
        .ok_or_else(|| eyre!("missing aggregate VmPvs AIR region"))?;
    assert_eq!(vm_pvs_region.explicit_len as usize, VmPvs::<u8>::width());
    assert!(vm_pvs_region.explicit_offset > 0);
    let (full_occupancy, reflected_is_valid_column) = genuine_vm_pvs_active_occupancy(&context)?;
    assert_eq!(reflected_is_valid_column, 1);
    assert_eq!(full_occupancy, MAX_PROOFS);
    let vm_pvs_height = 1usize << vm_pvs_region.log_height;
    let cached_width = vm_pvs_region
        .trace_layout
        .cached_main_widths
        .iter()
        .map(|&width| width as usize)
        .sum::<usize>();
    let common_main_start = usize::try_from(vm_pvs_region.message_offset)?
        .checked_add(vm_pvs_height * cached_width)
        .ok_or_else(|| eyre!("VM-PVS common-main offset"))?;
    let is_valid_block_start = common_main_start
        .checked_add(vm_pvs_height * reflected_is_valid_column)
        .ok_or_else(|| eyre!("VM-PVS is_valid message block"))?;
    assert!(is_valid_block_start.is_multiple_of(vm_pvs_height));
    assert!(is_valid_block_start + vm_pvs_height <= layout.raw_message_len);

    // The aggregate batch VmPvs are relation inputs emitted by the existing
    // VmPvsAir, not trusted host metadata. Mutating one public coordinate while
    // keeping its complete verifier trace fixed must fail the local PESAT.
    let (_, mutated_vm_pvs) = second_context
        .per_trace
        .iter_mut()
        .find(|(air_id, _)| *air_id == VM_PVS_AIR_ID)
        .ok_or_else(|| eyre!("missing aggregate VmPvs AIR"))?;
    mutated_vm_pvs.public_values[0] += F::ONE;
    assert!(
        !backend_region_is_satisfied(&second_context, &backend_index, VM_PVS_AIR_ID)?,
        "mutated aggregate VmPvs escaped the fixed verifier relation"
    );

    // The final partial batch must not introduce a second WARP index. Derive
    // setup heights from the actual full capacity-four execution, indexed by
    // AIR id, then differentially check every possible non-empty partial
    // batch. The existing verifier AIRs mark the unused suffix inactive. This
    // is ordinary witness data inside one predicate, not a selector choosing
    // among relations.
    let mut required_heights = vec![0usize; aggregation_vk.inner.per_air.len()];
    for (air_id, air) in &context.per_trace {
        assert_eq!(
            required_heights[*air_id], 0,
            "full fixture contains duplicate AIR id {air_id}"
        );
        required_heights[*air_id] = air.common_main.height();
    }
    assert!(
        required_heights.iter().all(|height| *height != 0),
        "the full capacity-four execution must instantiate every AIR in the fixed index"
    );
    for partial_count in 1..MAX_PROOFS {
        let mut partial_context = leaf_prover.generate_fixed_shape_proving_ctx(
            &app_proof.per_segment[..partial_count],
            ChildVkKind::App,
            ProofsType::Vm,
            None,
            &required_heights,
            engine_device_ctx(&leaf_engine),
        );
        let partial_shape = partial_context
            .per_trace
            .iter()
            .map(|(air_id, air)| VerifierBatchTraceShape {
                air_id: *air_id,
                height: air.common_main.height(),
                width: air.common_main.width(),
                cached_mains: air.cached_mains.len(),
                public_values: air.public_values.len(),
            })
            .collect::<Vec<_>>();
        assert_eq!(
            shape, partial_shape,
            "{partial_count}-proof context escaped the capacity-four relation class"
        );
        assert_eq!(
            layout,
            canonical_direct_layout(&partial_context, &backend_index),
            "{partial_count}-proof context changed the direct multi-AIR index"
        );
        assert!(
            backend_dense_relation_is_satisfied(&partial_context, &backend_index)?,
            "{partial_count}-proof context failed the compiled backend relation"
        );
        let (active_occupancy, partial_is_valid_column) =
            genuine_vm_pvs_active_occupancy(&partial_context)?;
        assert_eq!(partial_is_valid_column, reflected_is_valid_column);
        assert_eq!(active_occupancy, partial_count);
        #[cfg(debug_assertions)]
        crate::prover::debug_constraints::<SC, _, _>(
            leaf_prover.get_circuit().as_ref(),
            &partial_context,
            &leaf_engine,
        );

        // `generate_fixed_shape_proving_ctx` emits an AIR-constrained inactive
        // suffix; it is not accepted merely because the host initialized the
        // extra rows to zero. Activating the first absent VM-proof slot while
        // leaving its constrained proof index unchanged must violate the same
        // backend relation used by WARP.
        let (_, vm_pvs) = partial_context
            .per_trace
            .iter_mut()
            .find(|(air_id, _)| *air_id == VM_PVS_AIR_ID)
            .ok_or_else(|| eyre!("missing VM-PVS AIR"))?;
        let width = Matrix::width(&vm_pvs.common_main);
        let inactive_row =
            &mut vm_pvs.common_main.values[partial_count * width..(partial_count + 1) * width];
        let inactive_cols: &mut VmPvsCols<F> = inactive_row.borrow_mut();
        assert_eq!(inactive_cols.is_valid, F::ZERO);
        inactive_cols.is_valid = F::ONE;
        assert!(
            !backend_region_is_satisfied(&partial_context, &backend_index, VM_PVS_AIR_ID)?,
            "{partial_count}-proof context accepted a forged active capacity slot"
        );
    }

    assert_eq!(context.per_trace.len(), EXPECTED_ACTIVE_AIRS);
    assert_eq!(aggregation_vk.inner.per_air.len(), EXPECTED_ACTIVE_AIRS);
    assert_eq!(layout.relation_degree, EXPECTED_RELATION_DEGREE);
    assert_eq!(layout.exact_relation_degree, EXPECTED_EXACT_RELATION_DEGREE);
    assert_eq!(layout.explicit_len, EXPECTED_EXPLICIT_LEN);
    assert_eq!(layout.raw_message_len, EXPECTED_RAW_MESSAGE_LEN);
    assert_eq!(layout.padded_message_len, EXPECTED_PADDED_MESSAGE_LEN);
    assert_eq!(
        layout.fixed_cached_trace_cells,
        EXPECTED_FIXED_CACHED_TRACE_CELLS
    );
    assert_eq!(layout.air_constraint_len, EXPECTED_AIR_CONSTRAINT_LEN);
    assert_eq!(
        layout.padding_constraint_len,
        EXPECTED_PADDING_CONSTRAINT_LEN
    );
    assert_eq!(layout.real_constraint_len, EXPECTED_REAL_CONSTRAINT_LEN);
    assert_eq!(
        layout.occupied_constraint_len,
        EXPECTED_PADDED_CONSTRAINT_LEN
    );
    assert_eq!(layout.padded_constraint_len, EXPECTED_PADDED_CONSTRAINT_LEN);
    assert!(layout.regions.windows(2).all(|pair| {
        pair[0].height > pair[1].height
            || (pair[0].height == pair[1].height && pair[0].air_id < pair[1].air_id)
    }));
    assert!(layout.regions.iter().all(|region| {
        region.message_start.is_multiple_of(region.height)
            && region.message_len
                == region.height
                    * (region.common_main_width + region.cached_main_widths.iter().sum::<usize>())
    }));
    let cached_commitments = context
        .per_trace
        .iter()
        .flat_map(|(_, air)| &air.cached_mains)
        .map(|cached| cached.commitment)
        .collect::<Vec<_>>();
    assert_eq!(
        cached_commitments,
        vec![leaf_prover.get_vk_commit(false).cached_commit],
        "the child-VK commitment is fixed metadata even though its rows are WARP message cells"
    );

    eprintln!(
        "VERIFIER_WARP_FIXED_MULTI_AIR_LAYOUT child_capacity={} occupied_children={} \
         air_count={} active_air_count={} direct_trace_cells={} fixed_cached_trace_cells={} \
         explicit_len={} raw_message_len={} padded_message_len={} air_constraint_len={} \
         padding_constraint_len={} real_constraint_len={} occupied_constraint_len={} \
         padded_constraint_len={} exact_relation_degree={} warp_degree_envelope={}",
        MAX_PROOFS,
        app_proof.per_segment.len(),
        aggregation_vk.inner.per_air.len(),
        context.per_trace.len(),
        layout.raw_message_len,
        layout.fixed_cached_trace_cells,
        layout.explicit_len,
        layout.raw_message_len,
        layout.padded_message_len,
        layout.air_constraint_len,
        layout.padding_constraint_len,
        layout.real_constraint_len,
        layout.occupied_constraint_len,
        layout.padded_constraint_len,
        layout.exact_relation_degree,
        layout.relation_degree,
    );

    Ok(())
}

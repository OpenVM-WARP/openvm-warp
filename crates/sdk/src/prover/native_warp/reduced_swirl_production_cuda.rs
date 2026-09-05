//! Production CUDA orchestration for reduced-source SWIRL-to-WARP.
//!
//! The application prover stops after AIR/LogUp, stacked-opening reduction,
//! and the original stacked RS commitment. Native WARP consumes those exact
//! committed codewords in setup-fixed arity-eight calls. Each completed call
//! is immediately certified by one bounded transition leaf; ordinary OpenVM
//! recursion reduces those leaves to one root. A single finalizer verifies
//! that root, reconciles the block manifest, resumes the native transcript and
//! runs terminal Decide/RS-adjoint/WHIR. The existing recursive adapter then
//! emits the standalone proof expected by the SDK.

use std::{sync::Arc, time::Instant};

use openvm_circuit::{
    arch::{Executor, MeteredExecutor, PreflightExecutor, VmBuilder, VmExecutionConfig},
    system::{memory::dimensions::MemoryDimensions, SystemWithFixedTraceHeights},
};
use openvm_continuations::circuit::reduced_swirl_transition_leaf::REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY;
use openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
use openvm_recursion_circuit::native_warp::ReducedSwirlTerminalProductionSetup;
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey,
    p3_field::{ExtensionField, TwoAdicField},
    proof::Proof,
    SystemParams,
};
use openvm_stark_sdk::config::{baby_bear_poseidon2::EF, params_with_100_bits_security};
use openvm_verify_stark_host::{vk::VmStarkVerifyingKey, VmStarkProof};

use super::{
    reduced_swirl_execution_cuda::{
        prove_reduced_swirl_cuda_transition_execution, ReducedSwirlCudaExecutionError,
        ReducedSwirlCudaTransitionExecution,
    },
    reduced_swirl_native::{
        verify_reduced_swirl_native_recorded, ReducedSwirlNativeError,
        ReducedSwirlNativeProverOutput, ReducedSwirlNativeVerification, REDUCED_SWIRL_MAX_SOURCES,
    },
    reduced_swirl_native_cuda::ReducedSwirlNativeCudaTelemetry,
    reduced_swirl_recursive_adapter::{
        package_reduced_swirl_recursive_proof, reduced_swirl_canonical_adapter_params,
        ReducedSwirlRecursiveAdapter,
    },
    reduced_swirl_transition_finalizer::{
        ReducedSwirlTransitionFinalizerComponents, ReducedSwirlTransitionFinalizerCudaProver,
        ReducedSwirlTransitionFinalizerSystemError,
    },
    reduced_swirl_transition_tree::{
        reduced_swirl_transition_tree_params, ReducedSwirlTransitionTreeCudaProver,
    },
};
use crate::{prover::AppProver, StdIn, F, SC};

/// Setup and steady-state phase timings. Native telemetry contains the
/// finer-grained CUDA residency and transfer counters.
#[derive(Clone, Copy, Debug, Default)]
pub struct ReducedSwirlProductionCudaTelemetry {
    /// Segment reduction, WARP calls and their bounded transition leaves.
    pub source_and_warp_ms: f64,
    /// Foreground construction of bounded source/VACC verifier packets.
    pub transition_packet_ms: f64,
    /// Background transition-leaf proving work. This overlaps source reduction
    /// and therefore is not an additive end-to-end phase.
    pub transition_leaf_active_ms: f64,
    /// Foreground cost of copying one bounded completed-call snapshot into the
    /// worker queue. No codeword or accumulator prover data is copied.
    pub transition_snapshot_ms: f64,
    /// Critical-path wait before VACC for the prior background leaf.
    pub transition_leaf_barrier_wait_ms: f64,
    pub native_verify_ms: f64,
    pub component_setup_ms: f64,
    pub transition_tree_prove_ms: f64,
    /// Fixed finalizer MultiSTARK key generation.
    pub wrapper_keygen_ms: f64,
    /// Fixed finalizer MultiSTARK proving, including terminal Decide/WHIR.
    pub wrapper_prove_ms: f64,
    pub recursive_adapter_setup_ms: f64,
    pub recursive_adapter_prove_ms: f64,
    pub total_ms: f64,
    pub source_count: usize,
    pub vacc_call_count: usize,
}

/// Complete standalone proof artifact plus native audit material.
pub struct ReducedSwirlProductionCudaOutput {
    pub native: ReducedSwirlNativeProverOutput,
    pub native_verification: ReducedSwirlNativeVerification,
    pub cuda_telemetry: ReducedSwirlNativeCudaTelemetry,
    /// The single transition-tree finalizer proof before normalisation.
    pub wrapper_proof: Proof<SC>,
    pub wrapper_vk: Arc<MultiStarkVerifyingKey<SC>>,
    pub succinct_proof: VmStarkProof,
    pub succinct_vk: VmStarkVerifyingKey,
    pub telemetry: ReducedSwirlProductionCudaTelemetry,
}

#[derive(Debug, thiserror::Error)]
pub enum ReducedSwirlProductionCudaError {
    #[error(transparent)]
    Execution(#[from] ReducedSwirlCudaExecutionError),
    #[error(transparent)]
    Native(#[from] ReducedSwirlNativeError),
    #[error(transparent)]
    Finalizer(#[from] ReducedSwirlTransitionFinalizerSystemError),
    #[error("invalid reduced-SWIRL production setup: {0}")]
    Setup(String),
    #[error("reduced-SWIRL {phase} proof failed: {message}")]
    ProofPhase {
        phase: &'static str,
        message: String,
    },
    #[error("recursive reduced-SWIRL compression failed: {0}")]
    Recursive(String),
    #[error("CUDA phase cleanup failed after {phase}: {message}")]
    CudaCleanup {
        phase: &'static str,
        message: String,
    },
}

/// Prove and compress one execution with the setup-fixed arity-eight running
/// accumulator. `recursive_leaf_params` and `recursive_internal_params` come
/// from the ordinary OpenVM aggregation configuration used as the baseline.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn prove_reduced_swirl_production_cuda<VB>(
    mut app_prover: AppProver<BabyBearPoseidon2GpuEngine, VB>,
    input: StdIn<F>,
    input_arity: usize,
    family_target_bits: usize,
    recursive_leaf_params: SystemParams,
    recursive_internal_params: SystemParams,
) -> Result<ReducedSwirlProductionCudaOutput, ReducedSwirlProductionCudaError>
where
    VB: VmBuilder<BabyBearPoseidon2GpuEngine>,
    VB::SystemChipInventory: SystemWithFixedTraceHeights,
    <VB::VmConfig as VmExecutionConfig<F>>::Executor:
        Executor<F> + MeteredExecutor<F> + PreflightExecutor<F, VB::RecordArena>,
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField + Ord,
{
    if input_arity != REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY {
        return Err(ReducedSwirlProductionCudaError::Setup(format!(
            "production WARP arity must be {}",
            REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY
        )));
    }

    let total_started = Instant::now();
    let app_vk = app_prover.app_vm_vk().clone();
    let app_params = app_vk.inner.params.clone();
    let app_exe_commit = app_prover.app_exe_commit();
    let memory_dimensions: MemoryDimensions = app_prover.memory_dimensions();
    let num_user_pvs = app_prover.num_user_pvs();
    let wrapper_params = reduced_swirl_transition_wrapper_params(&app_params);

    let source_started = Instant::now();
    let execution = prove_reduced_swirl_cuda_transition_execution(
        &mut app_prover,
        input,
        input_arity,
        family_target_bits,
        wrapper_params.clone(),
    )?;
    let source_and_warp_ms = elapsed_ms(source_started);
    report_phase(
        "native execution/WARP/transition leaves",
        source_and_warp_ms,
    );

    let ReducedSwirlCudaTransitionExecution {
        setup,
        native: native_cuda,
        transition_leaf_proofs,
        transition_leaf_vk,
        recursive_app_vk_commit,
        transition_chain_endpoint,
        initial_transition_state,
        final_transition_state,
        transition_packet_ms,
        transition_leaf_active_ms,
        transition_snapshot_ms,
        transition_leaf_barrier_wait_ms,
        segment_metadata,
        user_public_values,
    } = execution;
    report_phase("transition packet construction", transition_packet_ms);
    report_phase("transition leaf background work", transition_leaf_active_ms);
    report_phase("transition snapshot", transition_snapshot_ms);
    report_phase(
        "transition leaf barrier wait",
        transition_leaf_barrier_wait_ms,
    );
    if transition_chain_endpoint != final_transition_state.manifest_chain {
        return Err(ReducedSwirlProductionCudaError::Setup(
            "transition-chain endpoint does not match the final authenticated state".to_owned(),
        ));
    }
    let native_setup = setup.cpu_setup().clone();
    let cuda_telemetry = native_cuda.telemetry;
    let native = native_cuda.native;
    let source_count = native.authoritative_claims.len();
    let vacc_call_count = native.proof.vacc.steps.len();

    // Match the ordinary recursive lane's ownership discipline. At this point
    // all large source witnesses and original resident codewords have either
    // been consumed into a transition leaf or the running accumulator.
    drop(app_prover);
    drop(setup);
    drop(segment_metadata);
    cleanup_cuda_phase("native execution/WARP/transition leaves")?;

    let native_verify_started = Instant::now();
    let native_verification = verify_reduced_swirl_native_recorded(
        &native_setup,
        &native.proof.statement,
        &native.authoritative_claims,
        &native.proof,
    )?;
    let native_verify_ms = elapsed_ms(native_verify_started);
    report_phase("native verification", native_verify_ms);

    let terminal_setup = ReducedSwirlTerminalProductionSetup::from_native_fixed_params(
        app_params.clone(),
        input_arity,
        family_target_bits,
        REDUCED_SWIRL_MAX_SOURCES,
    )
    .map_err(|error| ReducedSwirlProductionCudaError::Setup(format!("{error:?}")))?;

    let component_setup_started = Instant::now();
    let transition_tree_params = reduced_swirl_transition_tree_params(&recursive_leaf_params)
        .map_err(|error| ReducedSwirlProductionCudaError::Setup(error.to_string()))?;
    let transition_tree_prover =
        ReducedSwirlTransitionTreeCudaProver::new(transition_leaf_vk, transition_tree_params);
    let mut component_setup_ms = elapsed_ms(component_setup_started);

    let transition_tree_started = Instant::now();
    let transition_tree = transition_tree_prover
        .prove(transition_leaf_proofs, recursive_app_vk_commit)
        .map_err(|error| ReducedSwirlProductionCudaError::ProofPhase {
            phase: "transition proof tree",
            message: error.to_string(),
        })?;
    let transition_tree_prove_ms = elapsed_ms(transition_tree_started);
    report_phase("transition proof tree", transition_tree_prove_ms);
    cleanup_cuda_phase("transition proof tree")?;

    let finalizer_components_started = Instant::now();
    let finalizer_params = reduced_swirl_transition_finalizer_params(
        &wrapper_params,
        &transition_tree.root_vk.inner.params,
    );
    let finalizer_components = Arc::new(ReducedSwirlTransitionFinalizerComponents::new(
        Arc::clone(&transition_tree.root_vk),
        transition_tree.trusted_vk_commits,
        &native_setup,
        &terminal_setup,
        app_params,
        finalizer_params,
    )?);
    component_setup_ms += elapsed_ms(finalizer_components_started);

    let wrapper_keygen_started = Instant::now();
    let mut finalizer_prover =
        ReducedSwirlTransitionFinalizerCudaProver::new(Arc::clone(&finalizer_components))?;
    let wrapper_keygen_ms = elapsed_ms(wrapper_keygen_started);
    report_phase("transition finalizer keygen", wrapper_keygen_ms);

    let binding = finalizer_components.binding().clone();
    let wrapper_prove_started = Instant::now();
    let wrapper_proof = finalizer_prover.prove(
        &transition_tree.proof,
        initial_transition_state,
        final_transition_state,
        &native.proof.statement.source_bindings,
        &native_setup,
        &native,
        &native_verification,
    )?;
    let wrapper_prove_ms = elapsed_ms(wrapper_prove_started);
    report_phase("transition finalizer proof", wrapper_prove_ms);
    let wrapper_vk = finalizer_prover.keys().verifying_key();
    drop(finalizer_prover);
    drop(finalizer_components);
    drop(transition_tree);
    cleanup_cuda_phase("transition finalizer")?;

    let adapter_setup_started = Instant::now();
    let adapter_params =
        reduced_swirl_canonical_adapter_params(&wrapper_vk.inner.params, recursive_internal_params);
    let adapter = ReducedSwirlRecursiveAdapter::new(Arc::clone(&wrapper_vk), adapter_params);
    let recursive_adapter_setup_ms = elapsed_ms(adapter_setup_started);

    let recursive_prove_started = Instant::now();
    let (succinct_proof, succinct_vk) = package_reduced_swirl_recursive_proof(
        &adapter,
        wrapper_proof.clone(),
        binding,
        user_public_values,
        app_exe_commit,
        memory_dimensions,
        num_user_pvs,
    )
    .map_err(|error| ReducedSwirlProductionCudaError::Recursive(error.to_string()))?;
    let recursive_adapter_prove_ms = elapsed_ms(recursive_prove_started);
    report_phase("recursive adapter proof", recursive_adapter_prove_ms);

    Ok(ReducedSwirlProductionCudaOutput {
        native,
        native_verification,
        cuda_telemetry,
        wrapper_proof,
        wrapper_vk,
        succinct_proof,
        succinct_vk,
        telemetry: ReducedSwirlProductionCudaTelemetry {
            source_and_warp_ms,
            transition_packet_ms,
            transition_leaf_active_ms,
            transition_snapshot_ms,
            transition_leaf_barrier_wait_ms,
            native_verify_ms,
            component_setup_ms,
            transition_tree_prove_ms,
            wrapper_keygen_ms,
            wrapper_prove_ms,
            recursive_adapter_setup_ms,
            recursive_adapter_prove_ms,
            total_ms: elapsed_ms(total_started),
            source_count,
            vacc_call_count,
        },
    })
}

/// Rebuild the direct-source verifier profile at the application's exact RS
/// geometry while raising only the constraint-degree envelope needed by the
/// source, VACC, and finalizer AIRs.
fn reduced_swirl_transition_wrapper_params(native: &SystemParams) -> SystemParams {
    params_with_100_bits_security(
        native.log_blowup,
        native.l_skip,
        native.n_stack,
        native.w_stack,
        native.whir.folding_pow_bits,
        native.whir.mu_pow_bits,
        native.whir.proximity,
        8,
        native.whir.query_phase_pow_bits,
        native.whir.k,
        native.log_commit_rows_per_query,
    )
}

/// Derive the finalizer's proof-system envelope from the ordinary recursive
/// child verifier, while retaining the transition leaf's larger local degree
/// or width requirements.  This mirrors OpenVM's recursive adapter policy:
/// parameters are fixed by the two verifying keys, never by a block witness.
/// The native constrained-RS/WARP geometry remains separately keyed by the
/// application parameters passed to the finalizer components.
fn reduced_swirl_transition_finalizer_params(
    transition_leaf: &SystemParams,
    recursive_child: &SystemParams,
) -> SystemParams {
    let recursive_width = recursive_child.w_stack.next_power_of_two();
    let required_width = transition_leaf
        .w_stack
        .max(recursive_width)
        .next_power_of_two();
    let extra_width_bits = required_width
        .ilog2()
        .saturating_sub(recursive_width.ilog2()) as usize;
    let log_stacked_height = recursive_child
        .log_stacked_height()
        .max(transition_leaf.log_stacked_height());
    let mut params = params_with_100_bits_security(
        recursive_child.log_blowup,
        recursive_child.l_skip,
        log_stacked_height - recursive_child.l_skip,
        required_width,
        recursive_child.whir.folding_pow_bits,
        recursive_child
            .whir
            .mu_pow_bits
            .saturating_add(extra_width_bits),
        recursive_child.whir.proximity,
        recursive_child
            .max_constraint_degree
            .max(transition_leaf.max_constraint_degree),
        recursive_child.whir.query_phase_pow_bits,
        recursive_child.whir.k,
        recursive_child.log_commit_rows_per_query,
    );
    let required_logup_message_length = recursive_child
        .logup
        .log_max_message_length
        .max(transition_leaf.logup.log_max_message_length);
    let extra_message_bits =
        required_logup_message_length.saturating_sub(params.logup.log_max_message_length);
    params.logup.log_max_message_length = required_logup_message_length;
    params.logup.pow_bits += extra_message_bits as usize;
    params
}

fn elapsed_ms(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

fn report_phase(phase: &str, elapsed_ms: f64) {
    eprintln!("REDUCED_SWIRL_PHASE phase={phase:?} elapsed_ms={elapsed_ms:.3}");
}

/// Match the ordinary recursive lane's phase boundary: all owners of live
/// device allocations are dropped by the caller, outstanding kernels finish,
/// then both OpenVM CUDA allocators return idle pages.
fn cleanup_cuda_phase(phase: &'static str) -> Result<usize, ReducedSwirlProductionCudaError> {
    openvm_cuda_common::stream::device_synchronize().map_err(|error| {
        ReducedSwirlProductionCudaError::CudaCleanup {
            phase,
            message: error.to_string(),
        }
    })?;
    let vmm = openvm_cuda_common::memory_manager::trim_device_memory_pool();
    let asynchronous =
        openvm_cuda_common::memory_manager::trim_cuda_async_memory_pool(0).map_err(|error| {
            ReducedSwirlProductionCudaError::CudaCleanup {
                phase,
                message: error.to_string(),
            }
        })?;
    Ok(vmm.saturating_add(asynchronous))
}

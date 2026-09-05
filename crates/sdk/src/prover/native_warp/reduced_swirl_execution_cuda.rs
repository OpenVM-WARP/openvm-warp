//! End-to-end CUDA execution at SWIRL's deferred-opening boundary.
//!
//! This is the device counterpart of `reduced_swirl_execution_cpu`.  The VM
//! proves AIR/LogUp and the stacked reduction once, then moves the exact
//! retained codeword matrices and Merkle trees into constrained-code WARP.
//! It never creates a complete per-segment WHIR proof and never re-encodes a
//! projected scalar source.

use std::{
    cell::{Cell, RefCell},
    rc::Rc,
    sync::{
        mpsc::{sync_channel, Receiver, SyncSender},
        Arc, Condvar, Mutex,
    },
    thread::JoinHandle,
    time::Instant,
};

use openvm_circuit::{
    arch::{
        instructions::exe::VmExe, verify_segment_metadata_sequence, Executor, MeteredExecutor,
        PreflightExecutor, VirtualMachineError, VmBuilder, VmExecutionConfig, VmSegmentMetadata,
    },
    system::{
        connector::DEFAULT_SUSPEND_EXIT_CODE, memory::merkle::public_values::UserPublicValuesProof,
        SystemWithFixedTraceHeights,
    },
};
use openvm_continuations::circuit::{
    reduced_swirl_transition_leaf::{
        reduced_swirl_transition_chain_genesis, ReducedSwirlTransitionLeafBinding,
        ReducedSwirlTransitionState, REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY,
        REDUCED_SWIRL_TRANSITION_LEAF_PROTOCOL_VERSION,
    },
    reduced_swirl_warp::ReducedSwirlSourceReceiptBus,
};
use openvm_cuda_backend::{
    reduced_swirl_source::CudaReducedSwirlConstrainedCodeSource, BabyBearPoseidon2GpuEngine,
    GpuBackend,
};
use openvm_cuda_common::stream::mark_thread_streams_background;
use openvm_recursion_circuit::{
    native_warp::{
        ReducedSwirlSourceAuthorityBus, ReducedSwirlSourceProfile,
        ReducedSwirlTerminalProductionSetup,
    },
    system::{BusIndexManager, RetainedStackingProof},
};
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey,
    native_warp::native_accumulator_instance_digest,
    p3_field::{ExtensionField, PrimeCharacteristicRing, TwoAdicField},
    proof::Proof,
    prover::CommittedTraceData,
    warp_accum::ExternalCommittedConstrainedCodeSource,
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, EF};
use openvm_verify_stark_host::pvs::VkCommit;

use super::{
    reduced_swirl_boundary::{
        AuthoritativeSwirlConstrainedRsClaim, PendingConstrainedCodePublicClaim, ReducedSwirlPrefix,
    },
    reduced_swirl_native::{reduced_swirl_source_challenger, REDUCED_SWIRL_MAX_SOURCES},
    reduced_swirl_native_cuda::{
        ReducedSwirlNativeCudaOutput, ReducedSwirlNativeCudaSetup, ReducedSwirlNativeCudaStream,
    },
    reduced_swirl_source_receipt::{
        reduced_swirl_recursive_app_vk_commit, reduced_swirl_source_receipt_profile,
        ProductionReducedSwirlSourceReceiptComponent,
    },
    reduced_swirl_transition_leaf::{
        ProductionReducedSwirlTransitionLeafComponents, ReducedSwirlTransitionLeafCudaProver,
        ReducedSwirlTransitionLeafCudaSystem,
    },
    reduced_swirl_vacc_component::ProductionReducedSwirlVaccComponent,
    reduced_swirl_wrapper_components::ReducedSwirlVerifierComponent,
};
use crate::{
    prover::{vm::types::VmProvingKey, AppProver},
    StdIn, F, SC,
};

/// Construct the segment prover with full initial RS ownership enabled before
/// the proving key and cached program are committed.
///
/// Calling ordinary `AppProver::new` and toggling the cache afterward is not
/// equivalent: preprocessed and program roots would already own root-only
/// trees and recovering their matrices would require duplicate commitments.
pub fn new_reduced_swirl_cuda_app_prover<VB>(
    vm_builder: VB,
    app_vm_pk: &VmProvingKey<VB::VmConfig>,
    app_exe: std::sync::Arc<VmExe<F>>,
) -> Result<AppProver<BabyBearPoseidon2GpuEngine, VB>, VirtualMachineError>
where
    VB: VmBuilder<BabyBearPoseidon2GpuEngine>,
{
    let mut engine = BabyBearPoseidon2GpuEngine::new(app_vm_pk.get_params());
    engine.device_mut().set_cache_rs_code_matrix(true);
    AppProver::new_with_engine(engine, vm_builder, app_vm_pk, app_exe)
}

/// Complete native CUDA result before the single recursive wrapper proof.
/// Large PCS owners are consumed on their originating device stream; only the
/// logarithmic retained prefixes and verifier-owned claims survive.
pub struct ReducedSwirlCudaExecution {
    pub setup: ReducedSwirlNativeCudaSetup,
    pub native: ReducedSwirlNativeCudaOutput,
    pub retained_prefixes: Vec<RetainedStackingProof>,
    pub authoritative_wrapper_claims: Vec<AuthoritativeSwirlConstrainedRsClaim>,
    pub segment_metadata: Vec<VmSegmentMetadata<SC>>,
    pub user_public_values: UserPublicValuesProof<DIGEST_SIZE, F>,
}

/// Streaming counterpart used by the production recursive wrapper.  At most
/// one deterministic WARP fresh batch and one combined recursive-leaf witness
/// are retained. Completed leaves are ordinary compact MultiSTARK proofs.
pub struct ReducedSwirlCudaTransitionExecution {
    pub setup: ReducedSwirlNativeCudaSetup,
    pub native: ReducedSwirlNativeCudaOutput,
    pub transition_leaf_proofs: Vec<Proof<SC>>,
    pub transition_leaf_vk: Arc<MultiStarkVerifyingKey<SC>>,
    /// Fixed recursive application identity committed by every transition
    /// leaf and reused as the trusted root-tree binding.
    pub recursive_app_vk_commit: VkCommit<F>,
    pub transition_chain_endpoint: Digest,
    pub initial_transition_state: ReducedSwirlTransitionState,
    pub final_transition_state: ReducedSwirlTransitionState,
    /// Sum of transition-receipt packet construction time on the background
    /// worker. This may overlap native source reduction.
    pub transition_packet_ms: f64,
    /// Foreground time needed to copy the bounded verifier state into an owned
    /// worker job. This is the packet pipeline's remaining critical-path cost.
    pub transition_snapshot_ms: f64,
    /// Sum of background transition-leaf prover time. This may overlap native
    /// reduction and must not be added to end-to-end wall time.
    pub transition_leaf_active_ms: f64,
    /// Time for which native execution was forced to wait before a memory-heavy
    /// VACC because the preceding background leaf had not completed.
    pub transition_leaf_barrier_wait_ms: f64,
    pub segment_metadata: Vec<VmSegmentMetadata<SC>>,
    pub user_public_values: UserPublicValuesProof<DIGEST_SIZE, F>,
}

#[derive(Debug, thiserror::Error)]
pub enum ReducedSwirlCudaExecutionError {
    #[error("native reduced-SWIRL CUDA setup failed: {0}")]
    Setup(String),
    #[error("native reduced-SWIRL CUDA VM stream failed: {0}")]
    Vm(String),
    #[error("native reduced-SWIRL CUDA segment {segment} failed: {message}")]
    Segment { segment: usize, message: String },
    #[error("native reduced-SWIRL CUDA segment continuity failed: {0}")]
    Continuity(String),
    #[error("native reduced-SWIRL CUDA stream returned no segment")]
    EmptyExecution,
    #[error("native reduced-SWIRL CUDA transition worker failed: {0}")]
    TransitionWorker(String),
}

/// One bounded transition certificate handed from the sequential native
/// WARP stream to a second CUDA stream. The channel holding these jobs has
/// capacity one, so memory remains independent of block length.
struct ReducedSwirlTransitionLeafJob {
    segment_index: usize,
    prefixes: Vec<RetainedStackingProof>,
    claims: Vec<AuthoritativeSwirlConstrainedRsClaim>,
    completed: super::reduced_swirl_native_cuda::ReducedSwirlNativeCudaCompletedStepOwned,
    completed_after: usize,
    source_end: usize,
    total_source_count: usize,
    call_index: usize,
    source_start: usize,
    batch_start_tidx: usize,
    start_sample_count: usize,
    start_state: [F; 16],
    end_tidx: usize,
    end_sample_count: usize,
    end_state: [F; 16],
    expected_prior_root: Digest,
    expected_prior_digest: Digest,
    expected_output_root: Digest,
    expected_output_digest: Digest,
    first_metadata: VmSegmentMetadata<SC>,
    last_metadata: VmSegmentMetadata<SC>,
}

#[derive(Clone, Copy)]
struct ReducedSwirlTransitionLeafFixedState {
    source_protocol_digest: Digest,
    warp_protocol_digest: Digest,
    relation_digest: Digest,
    warp_index_digest: Digest,
    schedule_digest: Digest,
}

struct ReducedSwirlTransitionLeafWorkerOutput {
    proofs: Vec<Proof<SC>>,
    chain_endpoint: Digest,
    initial_state: ReducedSwirlTransitionState,
    final_state: ReducedSwirlTransitionState,
    active_ms: f64,
    packet_ms: f64,
}

type ReducedSwirlTransitionLeafWorker = (
    SyncSender<ReducedSwirlTransitionLeafJob>,
    JoinHandle<Result<ReducedSwirlTransitionLeafWorkerOutput, ReducedSwirlCudaExecutionError>>,
    Arc<MultiStarkVerifyingKey<SC>>,
    ReducedSwirlTransitionLeafProgress,
);

#[derive(Default)]
struct ReducedSwirlTransitionLeafProgressState {
    completed: usize,
    failure: Option<String>,
}

type ReducedSwirlTransitionLeafProgress =
    Arc<(Mutex<ReducedSwirlTransitionLeafProgressState>, Condvar)>;

fn wait_for_transition_leaf_progress(
    progress: &ReducedSwirlTransitionLeafProgress,
    expected_completed: usize,
) -> Result<(), ReducedSwirlCudaExecutionError> {
    let (lock, ready) = progress.as_ref();
    let mut state = lock.lock().map_err(|_| {
        ReducedSwirlCudaExecutionError::TransitionWorker(
            "transition worker progress lock poisoned".to_owned(),
        )
    })?;
    while state.completed < expected_completed && state.failure.is_none() {
        state = ready.wait(state).map_err(|_| {
            ReducedSwirlCudaExecutionError::TransitionWorker(
                "transition worker progress wait poisoned".to_owned(),
            )
        })?;
    }
    if let Some(message) = &state.failure {
        return Err(ReducedSwirlCudaExecutionError::TransitionWorker(
            message.clone(),
        ));
    }
    Ok(())
}

/// Start the independent transition-certificate stream. Native WARP remains
/// strictly sequential on the foreground stream; this worker only proves the
/// already-fixed transition statement and therefore cannot affect any native
/// Fiat--Shamir challenge or accumulator state.
fn spawn_transition_leaf_worker(
    system: Arc<ReducedSwirlTransitionLeafCudaSystem>,
    source_component: Arc<
        ProductionReducedSwirlSourceReceiptComponent<REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY>,
    >,
    vacc_component: Arc<ProductionReducedSwirlVaccComponent>,
    app_vk: Arc<MultiStarkVerifyingKey<SC>>,
    fixed: ReducedSwirlTransitionLeafFixedState,
    proof_capacity: usize,
) -> Result<ReducedSwirlTransitionLeafWorker, ReducedSwirlCudaExecutionError> {
    let (jobs_tx, jobs_rx) = sync_channel::<ReducedSwirlTransitionLeafJob>(1);
    let (ready_tx, ready_rx) = sync_channel::<Result<Arc<MultiStarkVerifyingKey<SC>>, String>>(0);
    let progress = Arc::new((
        Mutex::new(ReducedSwirlTransitionLeafProgressState::default()),
        Condvar::new(),
    ));
    let worker_progress = Arc::clone(&progress);
    let handle = std::thread::Builder::new()
        .name("reduced-swirl-transition-leaf".to_owned())
        .spawn(move || {
            // The application reduction and native WARP streams stay at the
            // foreground CUDA priority. This stream fills their bubbles and
            // yields SMs whenever foreground kernels are runnable.
            mark_thread_streams_background();
            let prover = ReducedSwirlTransitionLeafCudaProver::new(system).map_err(|error| {
                ReducedSwirlCudaExecutionError::TransitionWorker(error.to_string())
            });
            let mut prover = match prover {
                Ok(prover) => {
                    let vk = prover.keys().verifying_key();
                    if ready_tx.send(Ok(vk)).is_err() {
                        return Err(ReducedSwirlCudaExecutionError::TransitionWorker(
                            "transition worker setup receiver closed".to_owned(),
                        ));
                    }
                    prover
                }
                Err(error) => {
                    let message = error.to_string();
                    let _ = ready_tx.send(Err(message));
                    return Err(error);
                }
            };
            // The child VK is immutable setup data. Keep its committed trace
            // on the worker stream and share it across every bounded packet.
            let source_cached_vk =
                source_component.commit_cuda_child_vk(app_vk.as_ref(), prover.engine());
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                prove_transition_leaf_jobs(
                    &mut prover,
                    jobs_rx,
                    source_component.as_ref(),
                    vacc_component.as_ref(),
                    app_vk.as_ref(),
                    &source_cached_vk,
                    fixed,
                    proof_capacity,
                    &worker_progress,
                )
            }));
            let result = match result {
                Ok(result) => result,
                Err(panic) => {
                    let message = panic
                        .downcast_ref::<&str>()
                        .map(|message| (*message).to_owned())
                        .or_else(|| panic.downcast_ref::<String>().cloned())
                        .unwrap_or_else(|| "transition worker panicked".to_owned());
                    Err(ReducedSwirlCudaExecutionError::TransitionWorker(message))
                }
            };
            if let Err(error) = &result {
                let (lock, ready) = worker_progress.as_ref();
                if let Ok(mut state) = lock.lock() {
                    state.failure = Some(error.to_string());
                    ready.notify_all();
                }
            }
            result
        })
        .map_err(|error| ReducedSwirlCudaExecutionError::TransitionWorker(error.to_string()))?;
    let vk = match ready_rx.recv() {
        Ok(Ok(vk)) => vk,
        Ok(Err(message)) => {
            let _ = handle.join();
            return Err(ReducedSwirlCudaExecutionError::TransitionWorker(message));
        }
        Err(error) => {
            let _ = handle.join();
            return Err(ReducedSwirlCudaExecutionError::TransitionWorker(
                error.to_string(),
            ));
        }
    };
    Ok((jobs_tx, handle, vk, progress))
}

fn prove_transition_leaf_jobs(
    prover: &mut ReducedSwirlTransitionLeafCudaProver,
    jobs: Receiver<ReducedSwirlTransitionLeafJob>,
    source_component: &ProductionReducedSwirlSourceReceiptComponent<
        REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY,
    >,
    vacc_component: &ProductionReducedSwirlVaccComponent,
    app_vk: &MultiStarkVerifyingKey<SC>,
    source_cached_vk: &CommittedTraceData<GpuBackend>,
    fixed: ReducedSwirlTransitionLeafFixedState,
    proof_capacity: usize,
    progress: &ReducedSwirlTransitionLeafProgress,
) -> Result<ReducedSwirlTransitionLeafWorkerOutput, ReducedSwirlCudaExecutionError> {
    let mut proofs = Vec::with_capacity(proof_capacity);
    let mut chain = reduced_swirl_transition_chain_genesis(fixed.source_protocol_digest);
    let mut initial_state = None;
    let mut final_state = None;
    let mut active_ms = 0.0;
    let mut packet_ms = 0.0;

    while let Ok(job) = jobs.recv() {
        let ReducedSwirlTransitionLeafJob {
            segment_index,
            prefixes,
            claims,
            completed,
            completed_after,
            source_end,
            total_source_count,
            call_index,
            source_start,
            batch_start_tidx,
            start_sample_count,
            start_state,
            end_tidx,
            end_sample_count,
            end_state,
            expected_prior_root,
            expected_prior_digest,
            expected_output_root,
            expected_output_digest,
            first_metadata,
            last_metadata,
        } = job;
        let packet_started = Instant::now();
        let completed_view = completed.as_borrowed();
        let source_packet = source_component
            .generate_cuda_in_flight_packet_with_cached_vk(
                app_vk,
                &prefixes,
                &claims,
                completed_view.total_source_count,
                u32::try_from(completed_view.source_start).map_err(|_| {
                    ReducedSwirlCudaExecutionError::Segment {
                        segment: segment_index,
                        message: "source offset exceeds u32".to_owned(),
                    }
                })?,
                completed_view.source_bindings,
                &[],
                source_cached_vk,
                prover.engine(),
            )
            .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                segment: segment_index,
                message: error.to_string(),
            })?;
        let vacc_packet = vacc_component
            .generate_transition_cpu_packet_from_completed_step(
                &completed_view,
                &source_packet.block,
            )
            .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                segment: segment_index,
                message: error.to_string(),
            })?;
        packet_ms += packet_started.elapsed().as_secs_f64() * 1000.0;
        let chain_before = chain;
        let prove_started = Instant::now();
        let leaf = prover
            .prove_packet(source_packet, vacc_packet, chain_before)
            .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                segment: segment_index,
                message: error.to_string(),
            })?;
        active_ms += prove_started.elapsed().as_secs_f64() * 1000.0;
        let fixed_state_matches = |state: &ReducedSwirlTransitionState| {
            state.source_protocol_digest == fixed.source_protocol_digest
                && state.warp_protocol_digest == fixed.warp_protocol_digest
                && state.relation_digest == fixed.relation_digest
                && state.warp_index_digest == fixed.warp_index_digest
                && state.schedule_digest == fixed.schedule_digest
                && state.total_source_count == F::from_usize(total_source_count)
                && state.program_commitment == first_metadata.program_commit
        };
        if usize::try_from(leaf.call_end).ok() != Some(completed_after)
            || usize::try_from(leaf.source_end).ok() != Some(source_end)
            || !fixed_state_matches(&leaf.initial_state)
            || !fixed_state_matches(&leaf.final_state)
            || leaf.initial_state.call_cursor != F::from_usize(call_index)
            || leaf.initial_state.source_cursor != F::from_usize(source_start)
            || leaf.initial_state.transcript_tidx != F::from_usize(batch_start_tidx)
            || leaf.initial_state.transcript_sample_count != F::from_usize(start_sample_count)
            || leaf.initial_state.transcript_state != start_state
            || leaf.initial_state.accumulator_root != expected_prior_root
            || leaf.initial_state.accumulator_digest != expected_prior_digest
            || leaf.initial_state.vm_pc != first_metadata.initial_pc
            || leaf.initial_state.vm_root != first_metadata.initial_memory_root
            || leaf.initial_state.manifest_chain != chain_before
            || leaf.final_state.call_cursor != F::from_usize(completed_after)
            || leaf.final_state.source_cursor != F::from_usize(source_end)
            || leaf.final_state.transcript_tidx != F::from_usize(end_tidx)
            || leaf.final_state.transcript_sample_count != F::from_usize(end_sample_count)
            || leaf.final_state.transcript_state != end_state
            || leaf.final_state.accumulator_root != expected_output_root
            || leaf.final_state.accumulator_digest != expected_output_digest
            || leaf.final_state.vm_pc != last_metadata.final_pc
            || leaf.final_state.vm_root != last_metadata.final_memory_root
            || leaf.final_state.manifest_chain != leaf.chain_after
        {
            return Err(ReducedSwirlCudaExecutionError::Segment {
                segment: segment_index,
                message: "transition leaf authenticated state".to_owned(),
            });
        }
        chain = leaf.chain_after;
        if initial_state.is_none() {
            initial_state = Some(leaf.initial_state);
        }
        final_state = Some(leaf.final_state);
        proofs.push(leaf.proof);
        let (lock, ready) = progress.as_ref();
        let mut state = lock.lock().map_err(|_| {
            ReducedSwirlCudaExecutionError::TransitionWorker(
                "transition worker progress lock poisoned".to_owned(),
            )
        })?;
        state.completed = proofs.len();
        ready.notify_all();
    }

    Ok(ReducedSwirlTransitionLeafWorkerOutput {
        proofs,
        chain_endpoint: chain,
        initial_state: initial_state.ok_or(ReducedSwirlCudaExecutionError::EmptyExecution)?,
        final_state: final_state.ok_or(ReducedSwirlCudaExecutionError::EmptyExecution)?,
        active_ms,
        packet_ms,
    })
}

/// Stream native reduced-SWIRL sources into WARP and immediately prove one
/// bounded recursive leaf whenever the deterministic WARP schedule completes
/// a call. This is the production ownership path: retained source prefixes
/// never grow with the block and no complete per-segment WHIR proof is built.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn prove_reduced_swirl_cuda_transition_execution<VB>(
    app_prover: &mut AppProver<BabyBearPoseidon2GpuEngine, VB>,
    input: StdIn<F>,
    input_arity: usize,
    family_target_bits: usize,
    wrapper_params: SystemParams,
) -> Result<ReducedSwirlCudaTransitionExecution, ReducedSwirlCudaExecutionError>
where
    VB: VmBuilder<BabyBearPoseidon2GpuEngine>,
    VB::SystemChipInventory: SystemWithFixedTraceHeights,
    <VB::VmConfig as VmExecutionConfig<F>>::Executor:
        Executor<F> + MeteredExecutor<F> + PreflightExecutor<F, VB::RecordArena>,
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField + Ord,
{
    if input_arity != REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY {
        return Err(ReducedSwirlCudaExecutionError::Setup(format!(
            "production transition arity must be {}",
            REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY
        )));
    }
    let plan = app_prover
        .plan_warp_stream(input.clone())
        .map_err(|error| ReducedSwirlCudaExecutionError::Vm(error.to_string()))?;
    let source_count = plan.segment_count();
    if source_count == 0 {
        return Err(ReducedSwirlCudaExecutionError::EmptyExecution);
    }
    let prover_config = app_prover.vm().engine.device().prover_config();
    if !prover_config.cache_rs_code_matrix || prover_config.rs_code_matrix_tile_columns != 0 {
        return Err(ReducedSwirlCudaExecutionError::Setup(
            "use new_reduced_swirl_cuda_app_prover so every original RS commitment remains resident"
                .to_owned(),
        ));
    }

    let app_vk = app_prover.app_vm_vk().clone();
    let app_params = app_vk.inner.params.clone();
    let config = app_prover.vm().engine.config().clone();
    let device_ctx = app_prover.vm().engine.device().device_ctx.clone();
    let setup = ReducedSwirlNativeCudaSetup::new(
        &config,
        &app_params,
        input_arity,
        family_target_bits,
        REDUCED_SWIRL_MAX_SOURCES,
        device_ctx.clone(),
    )
    .map_err(|error| ReducedSwirlCudaExecutionError::Setup(error.to_string()))?;
    // This is the sole source-protocol authority.  The source receipt, VACC
    // verifier, transition-chain genesis, and terminal Decide all derive their
    // identity from the same native application parameters.  In particular,
    // callers cannot inject a digest and the source component's own wrapper
    // digest is never confused with the receipt protocol digest.
    let terminal_setup = ReducedSwirlTerminalProductionSetup::from_native_fixed_params(
        app_params.clone(),
        input_arity,
        family_target_bits,
        REDUCED_SWIRL_MAX_SOURCES,
    )
    .map_err(|error| ReducedSwirlCudaExecutionError::Setup(format!("{error:?}")))?;
    let source_receipt_protocol_digest = terminal_setup.protocol_digest();

    let maximum_roots_per_source = 1usize
        .checked_add(
            app_vk
                .inner
                .per_air
                .iter()
                .map(|air| air.params.width.cached_mains.len())
                .sum::<usize>(),
        )
        .ok_or_else(|| {
            ReducedSwirlCudaExecutionError::Setup(
                "application cached-root capacity overflow".to_owned(),
            )
        })?;
    let source_profile = ReducedSwirlSourceProfile {
        maximum_sources: REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY,
        maximum_roots_per_source,
        maximum_openings_per_source: app_params.w_stack,
        l_skip: app_params.l_skip,
        n_stack: app_params.n_stack,
        log_blowup: app_params.log_blowup,
        log_commit_rows_per_query: app_params.log_commit_rows_per_query,
    };
    let receipt_profile = reduced_swirl_source_receipt_profile(
        &app_vk,
        source_profile,
        source_receipt_protocol_digest,
        DEFAULT_SUSPEND_EXIT_CODE,
    )
    .map_err(|error| ReducedSwirlCudaExecutionError::Setup(error.to_string()))?;
    let mut buses = BusIndexManager::from_next_bus_idx(0);
    let receipt_bus = ReducedSwirlSourceReceiptBus::new(buses.new_bus_idx());
    let authority_bus = ReducedSwirlSourceAuthorityBus::new(buses.new_bus_idx());
    let source_component = Arc::new(
        ProductionReducedSwirlSourceReceiptComponent::<
            REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY,
        >::new(
            Arc::new(app_vk.clone()),
            receipt_profile,
            wrapper_params.clone(),
            receipt_bus,
            authority_bus,
            buses,
        )
        .map_err(|error| ReducedSwirlCudaExecutionError::Setup(error.to_string()))?,
    );
    let recursive_app_vk_commit =
        reduced_swirl_recursive_app_vk_commit(source_component.as_ref(), &app_vk, &wrapper_params)
            .map_err(|error| ReducedSwirlCudaExecutionError::Setup(error.to_string()))?;
    let vacc_component = Arc::new(
        ProductionReducedSwirlVaccComponent::from_native_setup_for_transition_leaf(
            setup.cpu_setup(),
            wrapper_params.clone(),
            source_component.as_ref(),
            BusIndexManager::from_next_bus_idx(source_component.inner().next_bus_idx()),
        )
        .map_err(|error| ReducedSwirlCudaExecutionError::Setup(error.to_string()))?,
    );
    if vacc_component.profile().relation_digest != terminal_setup.relation_digest() {
        return Err(ReducedSwirlCudaExecutionError::Setup(
            "transition and terminal setups disagree on the native reduced-SWIRL relation"
                .to_owned(),
        ));
    }
    let transition_components = Arc::new(
        ProductionReducedSwirlTransitionLeafComponents::new(
            Arc::clone(&source_component),
            Arc::clone(&vacc_component),
            wrapper_params.clone(),
        )
        .map_err(|error| ReducedSwirlCudaExecutionError::Setup(error.to_string()))?,
    );
    let transition_binding = ReducedSwirlTransitionLeafBinding {
        protocol_version: REDUCED_SWIRL_TRANSITION_LEAF_PROTOCOL_VERSION,
        source_capacity: REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY as u32,
        source_component_digest: source_component.protocol_digest(),
        vacc_component_digest: vacc_component.protocol_digest(),
        recursive_app_vk_commit,
    };
    let transition_system = Arc::new(
        ReducedSwirlTransitionLeafCudaSystem::new(
            transition_binding,
            transition_components,
            wrapper_params,
        )
        .map_err(|error| ReducedSwirlCudaExecutionError::Setup(error.to_string()))?,
    );
    let profile = vacc_component.profile();
    let fixed_transition_state = ReducedSwirlTransitionLeafFixedState {
        source_protocol_digest: source_receipt_protocol_digest,
        warp_protocol_digest: profile.protocol_digest,
        relation_digest: profile.relation_digest,
        warp_index_digest: profile.warp_index_digest,
        schedule_digest: profile.schedule_digest,
    };
    let proof_capacity = source_count.div_ceil(input_arity.saturating_sub(1));
    let (transition_jobs, transition_worker, transition_leaf_vk, transition_progress) =
        spawn_transition_leaf_worker(
            transition_system,
            Arc::clone(&source_component),
            Arc::clone(&vacc_component),
            Arc::new(app_vk.clone()),
            fixed_transition_state,
            proof_capacity,
        )?;

    let stream = Rc::new(RefCell::new(Some(
        ReducedSwirlNativeCudaStream::new(&setup, source_count)
            .map_err(|error| ReducedSwirlCudaExecutionError::Setup(error.to_string()))?,
    )));
    let pending_prefixes = Rc::new(RefCell::new(Vec::with_capacity(input_arity)));
    let pending_claims = Rc::new(RefCell::new(Vec::with_capacity(input_arity)));
    let segment_metadata = Rc::new(RefCell::new(Vec::with_capacity(source_count)));
    let transition_snapshot_ms = Rc::new(Cell::new(0.0));
    let transition_leaf_barrier_wait_ms = Rc::new(Cell::new(0.0));

    let callback_stream = Rc::clone(&stream);
    let callback_prefixes = Rc::clone(&pending_prefixes);
    let callback_claims = Rc::clone(&pending_claims);
    let callback_metadata = Rc::clone(&segment_metadata);
    let app_vk_for_callback = app_vk.clone();
    let config_for_callback = config.clone();
    let callback_transition_progress = Arc::clone(&transition_progress);
    let callback_transition_snapshot_ms = Rc::clone(&transition_snapshot_ms);
    let callback_transition_leaf_barrier_wait_ms = Rc::clone(&transition_leaf_barrier_wait_ms);
    let mut launched_transition_jobs = 0usize;

    let user_public_values_result = app_prover
        .prove_warp_stream_prepared(input, plan, move |segment_index, engine, pk, context| {
            let reduction = engine
                .prover()
                .prove_native_stacking_reduction(pk, context)
                .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                    segment: segment_index,
                    message: error.to_string(),
                })?;
            let prefix = ReducedSwirlPrefix::from_native_reduction(
                &app_vk_for_callback,
                segment_index,
                reduction,
            )
            .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                segment: segment_index,
                message: error.to_string(),
            })?;
            let metadata = openvm_circuit::arch::vm_segment_metadata_from_parts(
                &app_vk_for_callback,
                &prefix.retained().trace_vdata,
                &prefix.retained().public_values,
            )
            .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                segment: segment_index,
                message: error.to_string(),
            })?;
            let pending_public = PendingConstrainedCodePublicClaim {
                metadata: prefix.pending_witness().metadata().clone(),
                swirl_tilde_u: prefix.pending_witness().terminal_point().clone(),
                stacking_openings: prefix.retained().stacking_proof.stacking_openings.clone(),
            };
            let manifest_prefix = prefix.manifest_prefix(segment_index).map_err(|error| {
                ReducedSwirlCudaExecutionError::Segment {
                    segment: segment_index,
                    message: error.to_string(),
                }
            })?;
            let (retained, pending_witness) = prefix.into_parts();
            let mut source_challenger = reduced_swirl_source_challenger();
            let source = CudaReducedSwirlConstrainedCodeSource::try_from_pending(
                pending_witness,
                &retained.stacking_proof.stacking_openings,
                &mut source_challenger,
                device_ctx.clone(),
            )
            .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                segment: segment_index,
                message: error.to_string(),
            })?;
            let backend_claim = source.claim();
            let wrapper_claim = pending_public
                .authoritative_claim_from_backend(&backend_claim)
                .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                    segment: segment_index,
                    message: error.to_string(),
                })?;
            let source_binding =
                manifest_prefix
                    .digest_with_claim(&wrapper_claim)
                    .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                        segment: segment_index,
                        message: error.to_string(),
                    })?;
            callback_prefixes.borrow_mut().push(retained);
            callback_claims.borrow_mut().push(wrapper_claim);
            callback_metadata.borrow_mut().push(metadata);

            let completes_next_step = callback_stream
                .borrow()
                .as_ref()
                .ok_or(ReducedSwirlCudaExecutionError::EmptyExecution)?
                .next_source_completes_step()
                .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                    segment: segment_index,
                    message: error.to_string(),
                })?;
            // Keep one certificate at most in flight. In particular, finish
            // leaf i before VACC i+1 begins so their peak scratch allocations
            // never overlap. The leaf normally completes while the next fresh
            // source batch is being reduced, making this wait a no-op.
            if completes_next_step && launched_transition_jobs != 0 {
                let wait_started = Instant::now();
                wait_for_transition_leaf_progress(
                    &callback_transition_progress,
                    launched_transition_jobs,
                )?;
                callback_transition_leaf_barrier_wait_ms.set(
                    callback_transition_leaf_barrier_wait_ms.get()
                        + wait_started.elapsed().as_secs_f64() * 1000.0,
                );
            }
            let completed_before = callback_stream
                .borrow()
                .as_ref()
                .ok_or(ReducedSwirlCudaExecutionError::EmptyExecution)?
                .completed_step_count();
            callback_stream
                .borrow_mut()
                .as_mut()
                .ok_or(ReducedSwirlCudaExecutionError::EmptyExecution)?
                .push_source(source_binding, source)
                .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                    segment: segment_index,
                    message: error.to_string(),
                })?;
            let completed_after = callback_stream
                .borrow()
                .as_ref()
                .ok_or(ReducedSwirlCudaExecutionError::EmptyExecution)?
                .completed_step_count();
            if completed_after == completed_before {
                return Ok(());
            }
            if completed_after != completed_before + 1 {
                return Err(ReducedSwirlCudaExecutionError::Segment {
                    segment: segment_index,
                    message: "non-canonical transition completion count".to_owned(),
                });
            }

            let prefixes = core::mem::take(&mut *callback_prefixes.borrow_mut());
            let claims = core::mem::take(&mut *callback_claims.borrow_mut());
            let stream_ref = callback_stream.borrow();
            let completed = stream_ref
                .as_ref()
                .ok_or(ReducedSwirlCudaExecutionError::EmptyExecution)?
                .latest_completed_step()
                .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                    segment: segment_index,
                    message: error.to_string(),
                })?
                .ok_or(ReducedSwirlCudaExecutionError::EmptyExecution)?;
            if prefixes.len() != completed.fresh_count || claims.len() != completed.fresh_count {
                return Err(ReducedSwirlCudaExecutionError::Segment {
                    segment: segment_index,
                    message: "bounded transition source inventory".to_owned(),
                });
            }
            let source_end = completed
                .source_start
                .checked_add(completed.fresh_count)
                .ok_or_else(|| ReducedSwirlCudaExecutionError::Segment {
                    segment: segment_index,
                    message: "transition source range overflow".to_owned(),
                })?;
            let expected_prior_root = completed
                .prior_instance
                .map_or([F::ZERO; DIGEST_SIZE], |instance| instance.rt);
            let expected_prior_digest = completed
                .prior_instance
                .map_or([F::ZERO; DIGEST_SIZE], |instance| {
                    native_accumulator_instance_digest::<SC>(&config_for_callback, instance)
                });
            let expected_output_digest = native_accumulator_instance_digest::<SC>(
                &config_for_callback,
                completed.output_instance,
            );
            let metadata = callback_metadata.borrow();
            let first_metadata =
                metadata
                    .get(completed.source_start)
                    .cloned()
                    .ok_or_else(|| ReducedSwirlCudaExecutionError::Segment {
                        segment: segment_index,
                        message: "transition leaf initial VM boundary".to_owned(),
                    })?;
            let last_metadata = metadata.get(source_end - 1).cloned().ok_or_else(|| {
                ReducedSwirlCudaExecutionError::Segment {
                    segment: segment_index,
                    message: "transition leaf final VM boundary".to_owned(),
                }
            })?;
            drop(metadata);
            // Clone only the bounded verifier state required by this one
            // transition. Source receipt trace generation and VACC packet
            // construction then run entirely on the background worker while
            // the native stream starts reducing the next source batch.
            let snapshot_started = Instant::now();
            let completed_snapshot = completed.to_owned_snapshot();
            callback_transition_snapshot_ms.set(
                callback_transition_snapshot_ms.get()
                    + snapshot_started.elapsed().as_secs_f64() * 1000.0,
            );
            let job = ReducedSwirlTransitionLeafJob {
                segment_index,
                prefixes,
                claims,
                completed: completed_snapshot,
                completed_after,
                source_end,
                total_source_count: completed.total_source_count,
                call_index: completed.call_index,
                source_start: completed.source_start,
                batch_start_tidx: completed.batch_start_checkpoint.operations,
                start_sample_count: completed.start_sample_count,
                start_state: completed.start_state,
                end_tidx: completed.end_checkpoint.operations,
                end_sample_count: completed.end_sample_count,
                end_state: completed.end_state,
                expected_prior_root,
                expected_prior_digest,
                expected_output_root: completed.output_instance.rt,
                expected_output_digest,
                first_metadata,
                last_metadata,
            };
            drop(stream_ref);
            transition_jobs
                .send(job)
                .map_err(|_| ReducedSwirlCudaExecutionError::Segment {
                    segment: segment_index,
                    message: "transition leaf worker disconnected".to_owned(),
                })?;
            launched_transition_jobs += 1;
            Ok(())
        })
        .map_err(|error| match error {
            openvm_circuit::arch::NativeWarpStreamError::Vm(error) => {
                ReducedSwirlCudaExecutionError::Vm(error.to_string())
            }
            openvm_circuit::arch::NativeWarpStreamError::Segment(error) => error,
        });

    // The callback (and therefore the sole job sender) is gone here. Joining
    // drains the final bounded leaf before terminal WARP Decide starts, so the
    // two peak-memory phases cannot overlap even on a short final batch.
    let transition_worker_result = transition_worker.join().map_err(|panic| {
        let message = panic
            .downcast_ref::<&str>()
            .map(|message| (*message).to_owned())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "transition worker panicked".to_owned());
        ReducedSwirlCudaExecutionError::TransitionWorker(message)
    })?;
    let user_public_values = user_public_values_result?;
    let transition_output = transition_worker_result?;

    if !pending_prefixes.borrow().is_empty() || !pending_claims.borrow().is_empty() {
        return Err(ReducedSwirlCudaExecutionError::EmptyExecution);
    }
    let native = stream
        .borrow_mut()
        .take()
        .ok_or(ReducedSwirlCudaExecutionError::EmptyExecution)?
        .finish()
        .map_err(|error| ReducedSwirlCudaExecutionError::Setup(error.to_string()))?;
    drop(stream);
    native
        .telemetry
        .assert_resident_terminal_contract()
        .map_err(|error| ReducedSwirlCudaExecutionError::Setup(error.to_owned()))?;
    let segment_metadata = Rc::try_unwrap(segment_metadata)
        .map_err(|_| ReducedSwirlCudaExecutionError::EmptyExecution)?
        .into_inner();
    verify_segment_metadata_sequence(&segment_metadata)
        .map_err(|error| ReducedSwirlCudaExecutionError::Continuity(error.to_string()))?;
    let transition_leaf_proofs = transition_output.proofs;
    if transition_leaf_proofs.len() != native.native.proof.vacc.steps.len()
        || segment_metadata.len() != source_count
    {
        return Err(ReducedSwirlCudaExecutionError::EmptyExecution);
    }
    let transition_chain_endpoint = transition_output.chain_endpoint;
    let initial_transition_state = transition_output.initial_state;
    let final_transition_state = transition_output.final_state;
    let transition_leaf_active_ms = transition_output.active_ms;
    let transition_packet_ms = transition_output.packet_ms;
    let first_metadata = segment_metadata
        .first()
        .ok_or(ReducedSwirlCudaExecutionError::EmptyExecution)?;
    let last_metadata = segment_metadata
        .last()
        .ok_or(ReducedSwirlCudaExecutionError::EmptyExecution)?;
    let call_count = native.native.proof.vacc.steps.len();
    let final_instance = &native.native.proof.vacc.final_instance;
    let final_instance_digest = native_accumulator_instance_digest::<SC>(&config, final_instance);
    let profile = vacc_component.profile();
    let genesis = reduced_swirl_transition_chain_genesis(source_receipt_protocol_digest);
    let fixed_initial_state = initial_transition_state.source_protocol_digest
        == source_receipt_protocol_digest
        && initial_transition_state.warp_protocol_digest == profile.protocol_digest
        && initial_transition_state.relation_digest == terminal_setup.relation_digest()
        && initial_transition_state.warp_index_digest == profile.warp_index_digest
        && initial_transition_state.schedule_digest == profile.schedule_digest
        && initial_transition_state.total_source_count == F::from_usize(source_count)
        && initial_transition_state.call_cursor == F::ZERO
        && initial_transition_state.source_cursor == F::ZERO
        && initial_transition_state.transcript_sample_count == F::ZERO
        && initial_transition_state.transcript_state == [F::ZERO; 16]
        && initial_transition_state.accumulator_root == [F::ZERO; DIGEST_SIZE]
        && initial_transition_state.accumulator_digest == [F::ZERO; DIGEST_SIZE]
        && initial_transition_state.vm_pc == first_metadata.initial_pc
        && initial_transition_state.vm_root == first_metadata.initial_memory_root
        && initial_transition_state.manifest_chain == genesis
        && initial_transition_state.program_commitment == first_metadata.program_commit;
    let fixed_final_state = final_transition_state.source_protocol_digest
        == source_receipt_protocol_digest
        && final_transition_state.warp_protocol_digest == profile.protocol_digest
        && final_transition_state.relation_digest == terminal_setup.relation_digest()
        && final_transition_state.warp_index_digest == profile.warp_index_digest
        && final_transition_state.schedule_digest == profile.schedule_digest
        && final_transition_state.total_source_count == F::from_usize(source_count)
        && final_transition_state.call_cursor == F::from_usize(call_count)
        && final_transition_state.source_cursor == F::from_usize(source_count)
        && final_transition_state.transcript_tidx != F::ZERO
        && final_transition_state.transcript_sample_count != F::ZERO
        && final_transition_state.accumulator_root == final_instance.rt
        && final_transition_state.accumulator_digest == final_instance_digest
        && final_transition_state.vm_pc == last_metadata.final_pc
        && final_transition_state.vm_root == last_metadata.final_memory_root
        && final_transition_state.manifest_chain == transition_chain_endpoint
        && final_transition_state.program_commitment == last_metadata.program_commit;
    if !fixed_initial_state
        || !fixed_final_state
        || first_metadata.program_commit != last_metadata.program_commit
        || native.native.proof.terminal.descriptor.root != final_instance.rt
        || native.native.transition_records.len() != call_count
        || native
            .native
            .transition_records
            .last()
            .map(|record| &record.output_instance)
            != Some(final_instance)
    {
        return Err(ReducedSwirlCudaExecutionError::Setup(
            "terminal transition state does not match the authenticated native stream".to_owned(),
        ));
    }
    Ok(ReducedSwirlCudaTransitionExecution {
        setup,
        native,
        transition_leaf_proofs,
        transition_leaf_vk,
        recursive_app_vk_commit,
        transition_chain_endpoint,
        initial_transition_state,
        final_transition_state,
        transition_packet_ms,
        transition_snapshot_ms: transition_snapshot_ms.get(),
        transition_leaf_active_ms,
        transition_leaf_barrier_wait_ms: transition_leaf_barrier_wait_ms.get(),
        segment_metadata,
        user_public_values,
    })
}

/// Prove every VM segment through one bounded resident constrained-code WARP
/// stream.  Retaining the original RS code matrices is a protocol requirement
/// of this implementation: if they do not fit, source construction fails
/// closed instead of silently re-encoding or falling back to CPU.
pub fn prove_reduced_swirl_cuda_execution<VB>(
    app_prover: &mut AppProver<BabyBearPoseidon2GpuEngine, VB>,
    input: StdIn<F>,
    input_arity: usize,
    family_target_bits: usize,
) -> Result<ReducedSwirlCudaExecution, ReducedSwirlCudaExecutionError>
where
    VB: VmBuilder<BabyBearPoseidon2GpuEngine>,
    VB::SystemChipInventory: SystemWithFixedTraceHeights,
    <VB::VmConfig as VmExecutionConfig<F>>::Executor:
        Executor<F> + MeteredExecutor<F> + PreflightExecutor<F, VB::RecordArena>,
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField + Ord,
{
    // Planning is shape-only and precedes every transcript and commitment.
    let plan = app_prover
        .plan_warp_stream(input.clone())
        .map_err(|error| ReducedSwirlCudaExecutionError::Vm(error.to_string()))?;
    let source_count = plan.segment_count();
    if source_count == 0 {
        return Err(ReducedSwirlCudaExecutionError::EmptyExecution);
    }

    // The exact original codeword matrices are WARP's fresh commitment.  This
    // policy must have been installed before AppProver construction so the PK
    // and cached program own those matrices too.  Fail closed instead of
    // recommitting either object or selecting a host reconstruction fallback.
    let prover_config = app_prover.vm().engine.device().prover_config();
    if !prover_config.cache_rs_code_matrix || prover_config.rs_code_matrix_tile_columns != 0 {
        return Err(ReducedSwirlCudaExecutionError::Setup(
            "use new_reduced_swirl_cuda_app_prover so every original RS commitment remains resident"
                .to_owned(),
        ));
    }

    let app_vk = app_prover.app_vm_vk().clone();
    let config = app_prover.vm().engine.config().clone();
    let device_ctx = app_prover.vm().engine.device().device_ctx.clone();
    let setup = ReducedSwirlNativeCudaSetup::new(
        &config,
        &app_vk.inner.params,
        input_arity,
        family_target_bits,
        REDUCED_SWIRL_MAX_SOURCES,
        device_ctx.clone(),
    )
    .map_err(|error| ReducedSwirlCudaExecutionError::Setup(error.to_string()))?;

    let stream = Rc::new(RefCell::new(Some(
        ReducedSwirlNativeCudaStream::new(&setup, source_count)
            .map_err(|error| ReducedSwirlCudaExecutionError::Setup(error.to_string()))?,
    )));
    let retained_prefixes = Rc::new(RefCell::new(Vec::with_capacity(source_count)));
    let wrapper_claims = Rc::new(RefCell::new(Vec::with_capacity(source_count)));
    let segment_metadata = Rc::new(RefCell::new(Vec::with_capacity(source_count)));
    let app_vk_for_callback = app_vk.clone();
    let callback_stream = Rc::clone(&stream);
    let callback_prefixes = Rc::clone(&retained_prefixes);
    let callback_claims = Rc::clone(&wrapper_claims);
    let callback_metadata = Rc::clone(&segment_metadata);

    let user_public_values = app_prover
        .prove_warp_stream_prepared(input, plan, move |segment_index, engine, pk, context| {
            let reduction = engine
                .prover()
                .prove_native_stacking_reduction(pk, context)
                .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                    segment: segment_index,
                    message: error.to_string(),
                })?;
            let prefix = ReducedSwirlPrefix::from_native_reduction(
                &app_vk_for_callback,
                segment_index,
                reduction,
            )
            .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                segment: segment_index,
                message: error.to_string(),
            })?;
            let metadata = openvm_circuit::arch::vm_segment_metadata_from_parts(
                &app_vk_for_callback,
                &prefix.retained().trace_vdata,
                &prefix.retained().public_values,
            )
            .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                segment: segment_index,
                message: error.to_string(),
            })?;
            let pending_public = PendingConstrainedCodePublicClaim {
                metadata: prefix.pending_witness().metadata().clone(),
                swirl_tilde_u: prefix.pending_witness().terminal_point().clone(),
                stacking_openings: prefix.retained().stacking_proof.stacking_openings.clone(),
            };
            let manifest_prefix = prefix.manifest_prefix(segment_index).map_err(|error| {
                ReducedSwirlCudaExecutionError::Segment {
                    segment: segment_index,
                    message: error.to_string(),
                }
            })?;
            let (retained, pending_witness) = prefix.into_parts();
            let mut source_challenger = reduced_swirl_source_challenger();
            let source = CudaReducedSwirlConstrainedCodeSource::try_from_pending(
                pending_witness,
                &retained.stacking_proof.stacking_openings,
                &mut source_challenger,
                device_ctx.clone(),
            )
            .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                segment: segment_index,
                message: error.to_string(),
            })?;
            #[cfg(debug_assertions)]
            source
                .validate_message_opening_resident_reference()
                .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                    segment: segment_index,
                    message: format!("resident constrained-code differential check: {error}"),
                })?;
            let backend_claim = source.claim();
            let wrapper_claim = pending_public
                .authoritative_claim_from_backend(&backend_claim)
                .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                    segment: segment_index,
                    message: error.to_string(),
                })?;
            let source_binding =
                manifest_prefix
                    .digest_with_claim(&wrapper_claim)
                    .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                        segment: segment_index,
                        message: error.to_string(),
                    })?;
            callback_stream
                .borrow_mut()
                .as_mut()
                .ok_or(ReducedSwirlCudaExecutionError::EmptyExecution)?
                .push_source(source_binding, source)
                .map_err(|error| ReducedSwirlCudaExecutionError::Segment {
                    segment: segment_index,
                    message: error.to_string(),
                })?;
            callback_prefixes.borrow_mut().push(retained);
            callback_claims.borrow_mut().push(wrapper_claim);
            callback_metadata.borrow_mut().push(metadata);
            Ok(())
        })
        .map_err(|error| match error {
            openvm_circuit::arch::NativeWarpStreamError::Vm(error) => {
                ReducedSwirlCudaExecutionError::Vm(error.to_string())
            }
            openvm_circuit::arch::NativeWarpStreamError::Segment(error) => error,
        })?;

    let native = stream
        .borrow_mut()
        .take()
        .ok_or(ReducedSwirlCudaExecutionError::EmptyExecution)?
        .finish()
        .map_err(|error| ReducedSwirlCudaExecutionError::Setup(error.to_string()))?;
    // Release the lifetime-bearing empty stream before moving its setup into
    // the returned execution artifact.
    drop(stream);
    native
        .telemetry
        .assert_resident_terminal_contract()
        .map_err(|error| ReducedSwirlCudaExecutionError::Setup(error.to_owned()))?;
    let retained_prefixes = Rc::try_unwrap(retained_prefixes)
        .map_err(|_| ReducedSwirlCudaExecutionError::EmptyExecution)?
        .into_inner();
    let authoritative_wrapper_claims = Rc::try_unwrap(wrapper_claims)
        .map_err(|_| ReducedSwirlCudaExecutionError::EmptyExecution)?
        .into_inner();
    let segment_metadata = Rc::try_unwrap(segment_metadata)
        .map_err(|_| ReducedSwirlCudaExecutionError::EmptyExecution)?
        .into_inner();
    if retained_prefixes.len() != source_count
        || authoritative_wrapper_claims.len() != source_count
        || segment_metadata.len() != source_count
        || native.native.authoritative_claims.len() != source_count
    {
        return Err(ReducedSwirlCudaExecutionError::EmptyExecution);
    }
    verify_segment_metadata_sequence(&segment_metadata)
        .map_err(|error| ReducedSwirlCudaExecutionError::Continuity(error.to_string()))?;

    Ok(ReducedSwirlCudaExecution {
        setup,
        native,
        retained_prefixes,
        authoritative_wrapper_claims,
        segment_metadata,
        user_public_values,
    })
}

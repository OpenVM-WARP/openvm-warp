//! CUDA-resident orchestration for native reduced-SWIRL constrained-code WARP.
//!
//! The source boundary is the exact `NativeStackingReduction` produced after
//! SWIRL has proved AIR/LogUp and stacked its columns, but before per-segment
//! WHIR.  Each pending source is projected on its originating CUDA stream and
//! enters Construction 9.4 as SWIRL's constrained-RS claim.  It is never
//! re-encoded or committed as a second scalar oracle.
//!
//! Memory is bounded by one deterministic WARP invocation plus one resident
//! accumulator.  A transition consumes the complete fresh batch, retains only
//! the output accumulator, and poisons the stream on error.  There is no
//! spill/restore policy and no CPU source fallback.  The terminal prover reuses
//! that accumulator's exact coefficient-subgroup codeword and Merkle root.

use core::{fmt::Display, mem::size_of};
use std::{sync::Arc, time::Instant};

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_cuda_backend::{
    hash_scheme::{BabyBearPoseidon2HashScheme, Poseidon2MerkleHash},
    reduced_swirl_source::{
        CudaReducedSwirlConstrainedCodeSource, CudaReducedSwirlOpeningAdapter,
        CudaReducedSwirlSourceError,
    },
    resident_terminal_whir::{CudaResidentTerminalWhirProver, ResidentTerminalWhirInstrumentation},
    resident_warp::{
        CudaResidentWarpCode, CudaResidentWarpOpeningBackend, CudaResidentWarpProverData,
    },
    stacked_reduction::StackedPcsData2,
    warp_batching_sumcheck::GpuBatchingSumcheck,
    GpuBackend,
};
use openvm_cuda_common::{
    copy::pcie_counters,
    memory_manager::{device_memory_snapshot, DeviceMemorySnapshot},
    stream::GpuDeviceCtx,
};
use openvm_recursion_circuit::system::RetainedStackingProof;
use openvm_stark_backend::{
    native_warp::NativeWarpChallenger,
    p3_field::{BasedVectorSpace, PrimeCharacteristicRing},
    proof::{BatchConstraintProof, GkrProof, StackingProof},
    prover::{NativeStackingReduction, PendingConstrainedCodeWitness},
    warp_accum::{
        derive_swirl_constrained_rs_terminal_statement, finish_exact_finite_warp_call, Accumulator,
        ExternalCommittedConstrainedCodeSource, FieldElementDigestObserver,
        ReducedConstrainedCodeClaim, ReducedConstrainedCodeRelation, ReducedWarpVaccRootProof,
        StackedRsFreshCommitment, StackedRsOpeningBackend, TerminalDescriptor, WarpAccumError,
        WarpProverValidation, WarpRootProver, WarpVaccStepProverRecord,
    },
    warp_pesat::{AccumulatorInstance, AlgebraicChallenger, LinearChainSchedule},
    StarkProtocolConfig, SystemParams, TranscriptCheckpoint, TranscriptHistory, TranscriptLog,
    WhirConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, Digest, DuplexSpongeRecorder, CHUNK, EF, F,
};
use serde::{Deserialize, Serialize};

use super::reduced_swirl_native::{
    reduced_swirl_manifest_digest, ReducedSwirlNativeError, ReducedSwirlNativeProof,
    ReducedSwirlNativeProverOutput, ReducedSwirlNativeSetup, ReducedSwirlNativeStatement,
    ReducedSwirlNativeTerminalProof, ReducedSwirlPowerBatchSecurityBudget,
    ReducedSwirlStepVerification, ReducedSwirlVaccStepProof, REDUCED_SWIRL_NATIVE_PROTOCOL_VERSION,
};
use crate::SC;

const REDUCED_SWIRL_NATIVE_TRANSCRIPT_TAG: &[u8] = b"openvm.native-warp.swirl-reduced-source.v3";
const REDUCED_SWIRL_NATIVE_SOURCE_BATCH_TAG: &[u8] =
    b"openvm.native-warp.swirl-reduced-source.batch.v3";
const REDUCED_SWIRL_NATIVE_MANIFEST_FOOTER_TAG: &[u8] =
    b"openvm.native-warp.swirl-reduced-source.manifest-footer.v3";

pub type ReducedSwirlCudaPendingWitness =
    PendingConstrainedCodeWitness<Digest, Vec<EF>, Vec<StackedPcsData2<Digest>>>;
pub type ReducedSwirlCudaNativeReduction = NativeStackingReduction<
    SC,
    GpuBackend,
    (GkrProof<SC>, BatchConstraintProof<SC>),
    StackingProof<SC>,
    Vec<EF>,
    Vec<StackedPcsData2<Digest>>,
>;
pub type ReducedSwirlCudaCode = CudaResidentWarpCode<
    <SC as StarkProtocolConfig>::Hasher,
    Poseidon2MerkleHash,
    FieldElementDigestObserver,
>;
pub type ReducedSwirlCudaAccumulator =
    Accumulator<EF, Digest, Arc<CudaResidentWarpProverData<Digest>>>;

#[derive(Debug, thiserror::Error)]
pub enum ReducedSwirlNativeCudaError {
    #[error(transparent)]
    Native(#[from] ReducedSwirlNativeError),
    #[error("reduced-SWIRL CUDA source failed: {0}")]
    Source(#[source] CudaReducedSwirlSourceError),
    #[error("reduced-SWIRL CUDA WARP failed: {0:?}")]
    Warp(WarpAccumError),
    #[error("reduced-SWIRL CUDA terminal failed: {0}")]
    Terminal(String),
    #[error("reduced-SWIRL CUDA telemetry failed: {0}")]
    Telemetry(String),
    #[error("invalid reduced-SWIRL CUDA stream state: {0}")]
    State(&'static str),
    #[error("source provider failed: {0}")]
    SourceProvider(String),
}

impl From<WarpAccumError> for ReducedSwirlNativeCudaError {
    fn from(error: WarpAccumError) -> Self {
        Self::Warp(error)
    }
}

#[derive(Clone)]
pub struct ReducedSwirlNativeCudaSetup {
    cpu: ReducedSwirlNativeSetup,
    warp: WarpRootProver<ReducedSwirlCudaCode>,
    whir: WhirConfig,
    device_ctx: GpuDeviceCtx,
}

impl ReducedSwirlNativeCudaSetup {
    pub fn new(
        config: &SC,
        params: &SystemParams,
        input_arity: usize,
        family_target_bits: usize,
        maximum_source_count: usize,
        device_ctx: GpuDeviceCtx,
    ) -> Result<Self, ReducedSwirlNativeCudaError> {
        let cpu = ReducedSwirlNativeSetup::new(
            config,
            params,
            input_arity,
            family_target_bits,
            maximum_source_count,
        )?;
        let schedule = LinearChainSchedule::new(input_arity).map_err(WarpAccumError::from)?;
        let code = CudaResidentWarpCode::new(cpu.code().clone(), device_ctx.clone());
        let warp = WarpRootProver::new(code, schedule, cpu.security().params)?
            .with_validation(WarpProverValidation::TrustedConstructedState);
        Ok(Self {
            cpu,
            warp,
            whir: params.whir.clone(),
            device_ctx,
        })
    }

    #[must_use]
    pub const fn cpu_setup(&self) -> &ReducedSwirlNativeSetup {
        &self.cpu
    }

    #[must_use]
    pub const fn warp(&self) -> &WarpRootProver<ReducedSwirlCudaCode> {
        &self.warp
    }

    #[must_use]
    pub const fn device_ctx(&self) -> &GpuDeviceCtx {
        &self.device_ctx
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReducedSwirlCudaTransferTelemetry {
    pub h2d_calls: u64,
    pub h2d_bytes: u64,
    pub d2h_calls: u64,
    pub d2h_bytes: u64,
    pub d2d_calls: u64,
    pub d2d_bytes: u64,
}

impl ReducedSwirlCudaTransferTelemetry {
    fn saturating_add(self, other: Self) -> Self {
        Self {
            h2d_calls: self.h2d_calls.saturating_add(other.h2d_calls),
            h2d_bytes: self.h2d_bytes.saturating_add(other.h2d_bytes),
            d2h_calls: self.d2h_calls.saturating_add(other.d2h_calls),
            d2h_bytes: self.d2h_bytes.saturating_add(other.d2h_bytes),
            d2d_calls: self.d2d_calls.saturating_add(other.d2d_calls),
            d2d_bytes: self.d2d_bytes.saturating_add(other.d2d_bytes),
        }
    }

    fn saturating_add_assign(&mut self, other: Self) {
        *self = self.saturating_add(other);
    }
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReducedSwirlCudaStepTelemetry {
    pub step_index: usize,
    pub source_start: usize,
    pub fresh_count: usize,
    pub source_projection_ms: f64,
    pub vacc_ms: f64,
    pub projected_message_bytes: usize,
    pub projected_codeword_bytes: usize,
    pub resident_accumulator_bytes: usize,
    pub boundary_live_gpu_bytes: usize,
    pub boundary_driver_used_gpu_bytes: usize,
    pub source_projection_transfers: ReducedSwirlCudaTransferTelemetry,
    pub vacc_transfers: ReducedSwirlCudaTransferTelemetry,
    pub transfers: ReducedSwirlCudaTransferTelemetry,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ReducedSwirlNativeCudaTelemetry {
    pub source_count: usize,
    pub vacc_step_count: usize,
    pub maximum_pending_sources: usize,
    pub maximum_scheduled_fresh_count: usize,
    pub projected_message_bytes: usize,
    pub projected_codeword_bytes: usize,
    pub peak_boundary_live_gpu_bytes: usize,
    pub peak_lifetime_live_gpu_bytes: usize,
    pub peak_driver_used_gpu_bytes: usize,
    pub driver_total_gpu_bytes: usize,
    pub accumulator_spill_count: usize,
    pub accumulator_restore_count: usize,
    pub accumulator_lifecycle_h2d_bytes: usize,
    pub accumulator_lifecycle_d2h_bytes: usize,
    pub fresh_reencodes: usize,
    pub fresh_recommits: usize,
    pub terminal_reused_initial_roots: usize,
    pub terminal_accumulator_reencodes: usize,
    pub terminal_accumulator_recommits: usize,
    pub terminal_full_message_d2h_bytes: usize,
    pub terminal_bounded_proof_d2h_bytes: usize,
    pub source_projection_ms: f64,
    pub vacc_ms: f64,
    pub terminal_ms: f64,
    pub terminal_boundary_live_gpu_bytes: usize,
    pub terminal_boundary_driver_used_gpu_bytes: usize,
    pub terminal_transfers: ReducedSwirlCudaTransferTelemetry,
    pub total_wall_ms: f64,
    pub transfers: ReducedSwirlCudaTransferTelemetry,
    pub steps: Vec<ReducedSwirlCudaStepTelemetry>,
}

impl ReducedSwirlNativeCudaTelemetry {
    /// Structural invariant of this orchestration path.  Bounded challenge,
    /// opened-row, Merkle-path, and proof transfers remain visible in
    /// `transfers`; only full-payload lifecycle traffic is forbidden here.
    pub fn assert_no_duplicate_payload_pipeline(&self) -> Result<(), &'static str> {
        if self.accumulator_spill_count != 0
            || self.accumulator_restore_count != 0
            || self.accumulator_lifecycle_h2d_bytes != 0
            || self.accumulator_lifecycle_d2h_bytes != 0
            || self.fresh_reencodes != 0
            || self.fresh_recommits != 0
            || self.terminal_reused_initial_roots != 1
            || self.terminal_accumulator_reencodes != 0
            || self.terminal_accumulator_recommits != 0
            || self.terminal_full_message_d2h_bytes != 0
        {
            return Err("duplicate or host-staged reduced-SWIRL CUDA payload pipeline");
        }
        Ok(())
    }
}

pub struct ReducedSwirlNativeCudaOutput {
    pub native: ReducedSwirlNativeProverOutput,
    pub telemetry: ReducedSwirlNativeCudaTelemetry,
}

/// Borrowed, verifier-authenticated view of the most recently completed WARP
/// call. The view is invalidated by the next mutable stream operation, forcing
/// an orchestrator to consume the bounded transition immediately instead of
/// retaining or cloning the growing native output.
///
/// `transcript_prefix` is the independent native verifier's exact transcript
/// through `end_checkpoint`. It is borrowed in place; constructing this view
/// never clones the transcript, proof history, claims, or source bindings.
pub struct ReducedSwirlNativeCudaCompletedStep<'a> {
    pub total_source_count: usize,
    pub call_index: usize,
    pub source_start: usize,
    pub fresh_count: usize,
    pub proof: &'a ReducedSwirlVaccStepProof,
    pub prover_record: &'a WarpVaccStepProverRecord<EF, Digest>,
    pub verification: &'a ReducedSwirlStepVerification,
    pub prior_instance: Option<&'a AccumulatorInstance<EF, Digest>>,
    pub output_instance: &'a AccumulatorInstance<EF, Digest>,
    pub transcript_prefix: &'a TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    pub batch_start_checkpoint: TranscriptCheckpoint,
    pub vacc_start_checkpoint: TranscriptCheckpoint,
    pub end_checkpoint: TranscriptCheckpoint,
    pub start_sample_count: usize,
    pub start_state: [F; POSEIDON2_WIDTH],
    pub end_sample_count: usize,
    pub end_state: [F; POSEIDON2_WIDTH],
    pub authoritative_claims:
        &'a [ReducedConstrainedCodeClaim<EF, StackedRsFreshCommitment<EF, Digest>>],
    pub source_bindings: &'a [Digest],
    _sealed: (),
}

/// Bounded owned snapshot of one completed native WARP call.
///
/// This exists solely to let the transition-certificate worker construct its
/// source and VACC traces after the native stream has resumed. The snapshot
/// contains no source codeword or accumulator prover data; it clones only the
/// call proof, verifier record, transcript log, claims, and roots that the
/// recursive transition AIR already consumes. At most one snapshot is queued.
#[derive(Clone)]
pub struct ReducedSwirlNativeCudaCompletedStepOwned {
    pub total_source_count: usize,
    pub call_index: usize,
    pub source_start: usize,
    pub fresh_count: usize,
    pub proof: ReducedSwirlVaccStepProof,
    pub prover_record: WarpVaccStepProverRecord<EF, Digest>,
    pub verification: ReducedSwirlStepVerification,
    pub prior_instance: Option<AccumulatorInstance<EF, Digest>>,
    pub output_instance: AccumulatorInstance<EF, Digest>,
    pub transcript_prefix: TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    pub batch_start_checkpoint: TranscriptCheckpoint,
    pub vacc_start_checkpoint: TranscriptCheckpoint,
    pub end_checkpoint: TranscriptCheckpoint,
    pub start_sample_count: usize,
    pub start_state: [F; POSEIDON2_WIDTH],
    pub end_sample_count: usize,
    pub end_state: [F; POSEIDON2_WIDTH],
    pub authoritative_claims:
        Vec<ReducedConstrainedCodeClaim<EF, StackedRsFreshCommitment<EF, Digest>>>,
    pub source_bindings: Vec<Digest>,
}

impl ReducedSwirlNativeCudaCompletedStep<'_> {
    #[must_use]
    pub fn to_owned_snapshot(&self) -> ReducedSwirlNativeCudaCompletedStepOwned {
        ReducedSwirlNativeCudaCompletedStepOwned {
            total_source_count: self.total_source_count,
            call_index: self.call_index,
            source_start: self.source_start,
            fresh_count: self.fresh_count,
            proof: self.proof.clone(),
            prover_record: self.prover_record.clone(),
            verification: self.verification.clone(),
            prior_instance: self.prior_instance.cloned(),
            output_instance: self.output_instance.clone(),
            transcript_prefix: self.transcript_prefix.clone(),
            batch_start_checkpoint: self.batch_start_checkpoint,
            vacc_start_checkpoint: self.vacc_start_checkpoint,
            end_checkpoint: self.end_checkpoint,
            start_sample_count: self.start_sample_count,
            start_state: self.start_state,
            end_sample_count: self.end_sample_count,
            end_state: self.end_state,
            authoritative_claims: self.authoritative_claims.to_vec(),
            source_bindings: self.source_bindings.to_vec(),
        }
    }
}

impl ReducedSwirlNativeCudaCompletedStepOwned {
    #[must_use]
    pub fn as_borrowed(&self) -> ReducedSwirlNativeCudaCompletedStep<'_> {
        ReducedSwirlNativeCudaCompletedStep {
            total_source_count: self.total_source_count,
            call_index: self.call_index,
            source_start: self.source_start,
            fresh_count: self.fresh_count,
            proof: &self.proof,
            prover_record: &self.prover_record,
            verification: &self.verification,
            prior_instance: self.prior_instance.as_ref(),
            output_instance: &self.output_instance,
            transcript_prefix: &self.transcript_prefix,
            batch_start_checkpoint: self.batch_start_checkpoint,
            vacc_start_checkpoint: self.vacc_start_checkpoint,
            end_checkpoint: self.end_checkpoint,
            start_sample_count: self.start_sample_count,
            start_state: self.start_state,
            end_sample_count: self.end_sample_count,
            end_state: self.end_state,
            authoritative_claims: &self.authoritative_claims,
            source_bindings: &self.source_bindings,
            _sealed: (),
        }
    }
}

struct ReducedSwirlNativeCudaCompletedStepState {
    call_index: usize,
    source_start: usize,
    fresh_count: usize,
    prior_instance: Option<AccumulatorInstance<EF, Digest>>,
    verification: ReducedSwirlStepVerification,
    batch_start_checkpoint: TranscriptCheckpoint,
    vacc_start_checkpoint: TranscriptCheckpoint,
    end_checkpoint: TranscriptCheckpoint,
    start_sample_count: usize,
    start_state: [F; POSEIDON2_WIDTH],
    end_sample_count: usize,
    end_state: [F; POSEIDON2_WIDTH],
}

/// One-stream, bounded-memory CUDA prover.
///
/// The maximum live logical payload is `schedule.max_fresh_count()` projected
/// source messages/codewords plus one resident accumulator.  The source batch
/// is consumed before the output accumulator becomes the next prior state.
pub struct ReducedSwirlNativeCudaStream<'a> {
    setup: &'a ReducedSwirlNativeCudaSetup,
    expected_source_count: usize,
    step_fresh_counts: Vec<usize>,
    challenger: NativeWarpChallenger<SC, DuplexSpongeRecorder>,
    pending_sources: Vec<CudaReducedSwirlConstrainedCodeSource>,
    pending_bindings: Vec<Digest>,
    pending_projection_ms: f64,
    pending_projection_transfers: ReducedSwirlCudaTransferTelemetry,
    pending_message_bytes: usize,
    pending_codeword_bytes: usize,
    source_bindings: Vec<Digest>,
    authoritative_claims:
        Vec<ReducedConstrainedCodeClaim<EF, StackedRsFreshCommitment<EF, Digest>>>,
    accumulator: Option<ReducedSwirlCudaAccumulator>,
    steps: Vec<ReducedSwirlVaccStepProof>,
    transition_records: Vec<WarpVaccStepProverRecord<EF, Digest>>,
    verifier_transcript: Option<DuplexSpongeRecorder>,
    verifier_prior: Option<AccumulatorInstance<EF, Digest>>,
    latest_completed_step: Option<ReducedSwirlNativeCudaCompletedStepState>,
    accelerator: GpuBatchingSumcheck,
    transfer_start: (u64, u64, u64, u64, u64, u64),
    started: Instant,
    telemetry: ReducedSwirlNativeCudaTelemetry,
    poisoned: bool,
}

impl<'a> ReducedSwirlNativeCudaStream<'a> {
    pub fn new(
        setup: &'a ReducedSwirlNativeCudaSetup,
        expected_source_count: usize,
    ) -> Result<Self, ReducedSwirlNativeCudaError> {
        if expected_source_count == 0 {
            return Err(ReducedSwirlNativeCudaError::State("empty source list"));
        }
        if expected_source_count > setup.cpu.maximum_source_count() {
            return Err(ReducedSwirlNativeCudaError::State(
                "source count exceeds fixed setup capacity",
            ));
        }
        let step_fresh_counts = setup
            .warp
            .schedule()
            .step_fresh_counts(expected_source_count);
        if step_fresh_counts.is_empty() || step_fresh_counts.contains(&0) {
            return Err(ReducedSwirlNativeCudaError::State("empty WARP schedule"));
        }
        let max_pending = step_fresh_counts.iter().copied().max().unwrap_or(0);
        let mut challenger = NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
        observe_statement_header(
            setup,
            REDUCED_SWIRL_NATIVE_PROTOCOL_VERSION,
            expected_source_count,
            &mut challenger,
        );
        let mut verifier_challenger =
            NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
        observe_statement_header(
            setup,
            REDUCED_SWIRL_NATIVE_PROTOCOL_VERSION,
            expected_source_count,
            &mut verifier_challenger,
        );
        let verifier_transcript = verifier_challenger.into_inner();
        let initial_memory = memory_snapshot()?;
        let mut telemetry = ReducedSwirlNativeCudaTelemetry {
            maximum_scheduled_fresh_count: max_pending,
            ..Default::default()
        };
        observe_memory(&mut telemetry, initial_memory);
        Ok(Self {
            setup,
            expected_source_count,
            step_fresh_counts: step_fresh_counts.clone(),
            challenger,
            pending_sources: Vec::with_capacity(max_pending),
            pending_bindings: Vec::with_capacity(max_pending),
            pending_projection_ms: 0.0,
            pending_projection_transfers: ReducedSwirlCudaTransferTelemetry::default(),
            pending_message_bytes: 0,
            pending_codeword_bytes: 0,
            source_bindings: Vec::with_capacity(expected_source_count),
            authoritative_claims: Vec::with_capacity(expected_source_count),
            accumulator: None,
            steps: Vec::with_capacity(step_fresh_counts.len()),
            transition_records: Vec::with_capacity(step_fresh_counts.len()),
            verifier_transcript: Some(verifier_transcript),
            verifier_prior: None,
            latest_completed_step: None,
            accelerator: GpuBatchingSumcheck::new(setup.device_ctx.clone())
                .with_claim65_scratch_retention_limit(0)
                .with_lazy_linear_fold_len(setup.cpu.code().message_len())
                .with_lazy_linear_fold_len(setup.cpu.code().codeword_len()),
            transfer_start: pcie_counters::snapshot(),
            started: Instant::now(),
            telemetry,
            poisoned: false,
        })
    }

    #[must_use]
    pub fn expected_next_source_index(&self) -> usize {
        self.source_bindings.len()
    }

    #[must_use]
    pub fn pending_source_count(&self) -> usize {
        self.pending_sources.len()
    }

    /// Number of genuine WARP calls completed by this stream.
    #[must_use]
    pub fn completed_step_count(&self) -> usize {
        self.steps.len()
    }

    /// Whether accepting exactly one more source will execute the next
    /// deterministic WARP transition.
    ///
    /// Orchestrators use this as a memory barrier for independent background
    /// certificate work. It is derived from the setup-bound finite schedule;
    /// callers must not infer it from the nominal input arity because the
    /// final transition may contain fewer fresh sources.
    pub fn next_source_completes_step(&self) -> Result<bool, ReducedSwirlNativeCudaError> {
        self.ensure_live()?;
        let expected = *self.step_fresh_counts.get(self.steps.len()).ok_or(
            ReducedSwirlNativeCudaError::State("too many WARP transitions"),
        )?;
        let next_pending =
            self.pending_sources
                .len()
                .checked_add(1)
                .ok_or(ReducedSwirlNativeCudaError::State(
                    "pending source count overflow",
                ))?;
        if next_pending > expected {
            return Err(ReducedSwirlNativeCudaError::State(
                "fresh batch exceeds deterministic schedule",
            ));
        }
        Ok(next_pending == expected)
    }

    /// Borrow the latest completed call and its independently verified native
    /// transcript. The returned local claim/binding slices contain at most one
    /// deterministic WARP call (arity eight in production).
    pub fn latest_completed_step(
        &self,
    ) -> Result<Option<ReducedSwirlNativeCudaCompletedStep<'_>>, ReducedSwirlNativeCudaError> {
        let Some(completed) = self.latest_completed_step.as_ref() else {
            return Ok(None);
        };
        let source_end = completed
            .source_start
            .checked_add(completed.fresh_count)
            .ok_or(ReducedSwirlNativeCudaError::State("completed source range"))?;
        let proof =
            self.steps
                .get(completed.call_index)
                .ok_or(ReducedSwirlNativeCudaError::State(
                    "completed proof inventory",
                ))?;
        let prover_record = self.transition_records.get(completed.call_index).ok_or(
            ReducedSwirlNativeCudaError::State("completed prover-record inventory"),
        )?;
        let authoritative_claims = self
            .authoritative_claims
            .get(completed.source_start..source_end)
            .ok_or(ReducedSwirlNativeCudaError::State(
                "completed authoritative-claim inventory",
            ))?;
        let source_bindings = self
            .source_bindings
            .get(completed.source_start..source_end)
            .ok_or(ReducedSwirlNativeCudaError::State(
                "completed source-binding inventory",
            ))?;
        let transcript_prefix = &self
            .verifier_transcript
            .as_ref()
            .ok_or(ReducedSwirlNativeCudaError::State(
                "missing incremental verifier transcript",
            ))?
            .log;
        if transcript_prefix.len() != completed.end_checkpoint.operations {
            return Err(ReducedSwirlNativeCudaError::State(
                "completed verifier checkpoint",
            ));
        }
        Ok(Some(ReducedSwirlNativeCudaCompletedStep {
            total_source_count: self.expected_source_count,
            call_index: completed.call_index,
            source_start: completed.source_start,
            fresh_count: completed.fresh_count,
            proof,
            prover_record,
            verification: &completed.verification,
            prior_instance: completed.prior_instance.as_ref(),
            output_instance: &completed.verification.output_instance,
            transcript_prefix,
            batch_start_checkpoint: completed.batch_start_checkpoint,
            vacc_start_checkpoint: completed.vacc_start_checkpoint,
            end_checkpoint: completed.end_checkpoint,
            start_sample_count: completed.start_sample_count,
            start_state: completed.start_state,
            end_sample_count: completed.end_sample_count,
            end_state: completed.end_state,
            authoritative_claims,
            source_bindings,
            _sealed: (),
        }))
    }

    /// Consume a genuine pending SWIRL source.  `source_challenger` is the
    /// segment-prefix transcript used to derive SWIRL's theta challenge; the
    /// block-wide WARP challenger remains independently owned by this stream.
    pub fn push_pending<Ch>(
        &mut self,
        source_binding: Digest,
        pending: ReducedSwirlCudaPendingWitness,
        stacking_openings: &[Vec<EF>],
        source_challenger: &mut Ch,
    ) -> Result<(), ReducedSwirlNativeCudaError>
    where
        Ch: AlgebraicChallenger<EF>,
    {
        self.ensure_live()?;
        let started = Instant::now();
        let transfer_start = pcie_counters::snapshot();
        let source = match CudaReducedSwirlConstrainedCodeSource::try_from_pending(
            pending,
            stacking_openings,
            source_challenger,
            self.setup.device_ctx.clone(),
        ) {
            Ok(source) => source,
            Err(error) => {
                self.poisoned = true;
                return Err(ReducedSwirlNativeCudaError::Source(error));
            }
        };
        let projection_ms = elapsed_ms(started);
        let projection_transfers = transfer_delta(transfer_start, pcie_counters::snapshot());
        self.push_source_with_projection_telemetry(
            source_binding,
            source,
            projection_ms,
            projection_transfers,
        )
    }

    /// Split a native reduction exactly once.  The retained recursive prefix
    /// is returned to the caller for the succinct wrapper; the opaque pending
    /// PCS owner moves directly into this CUDA stream.
    pub fn push_native_reduction<Ch>(
        &mut self,
        source_binding: Digest,
        reduction: ReducedSwirlCudaNativeReduction,
        source_challenger: &mut Ch,
    ) -> Result<RetainedStackingProof, ReducedSwirlNativeCudaError>
    where
        Ch: AlgebraicChallenger<EF>,
    {
        let (retained, pending) = RetainedStackingProof::split_native_reduction(reduction);
        self.push_pending(
            source_binding,
            pending,
            &retained.stacking_proof.stacking_openings,
            source_challenger,
        )?;
        Ok(retained)
    }

    /// Accept an already constructed device-authoritative source.  This is
    /// useful when the caller owns the retained-prefix transcript machinery.
    pub fn push_source(
        &mut self,
        source_binding: Digest,
        source: CudaReducedSwirlConstrainedCodeSource,
    ) -> Result<(), ReducedSwirlNativeCudaError> {
        self.push_source_with_projection_telemetry(
            source_binding,
            source,
            0.0,
            ReducedSwirlCudaTransferTelemetry::default(),
        )
    }

    fn push_source_with_projection_telemetry(
        &mut self,
        source_binding: Digest,
        source: CudaReducedSwirlConstrainedCodeSource,
        projection_ms: f64,
        projection_transfers: ReducedSwirlCudaTransferTelemetry,
    ) -> Result<(), ReducedSwirlNativeCudaError> {
        self.ensure_live()?;
        let source_index = self.source_bindings.len();
        if source_index >= self.expected_source_count {
            return self.fail("too many reduced-SWIRL sources");
        }
        if is_zero_digest(&source_binding) {
            return self.fail("unset source manifest digest");
        }
        if let Err(error) = validate_source(self.setup, &source, source_index) {
            self.poisoned = true;
            return Err(error);
        }
        let message_bytes = source.message_len().checked_mul(size_of::<EF>()).ok_or(
            ReducedSwirlNativeCudaError::State("projected message byte count"),
        )?;
        let codeword_bytes = source.codeword_len().checked_mul(size_of::<EF>()).ok_or(
            ReducedSwirlNativeCudaError::State("projected codeword byte count"),
        )?;
        let claim = source.authoritative_claim_root().0.clone();
        self.authoritative_claims.push(claim);
        self.source_bindings.push(source_binding);
        self.pending_bindings.push(source_binding);
        self.pending_sources.push(source);
        self.pending_projection_ms += projection_ms;
        self.pending_projection_transfers
            .saturating_add_assign(projection_transfers);
        self.pending_message_bytes = self
            .pending_message_bytes
            .checked_add(message_bytes)
            .ok_or(ReducedSwirlNativeCudaError::State(
                "projected message byte count",
            ))?;
        self.pending_codeword_bytes = self
            .pending_codeword_bytes
            .checked_add(codeword_bytes)
            .ok_or(ReducedSwirlNativeCudaError::State(
                "projected codeword byte count",
            ))?;
        self.telemetry.maximum_pending_sources = self
            .telemetry
            .maximum_pending_sources
            .max(self.pending_sources.len());
        let source_memory = match memory_snapshot() {
            Ok(memory) => memory,
            Err(error) => {
                self.poisoned = true;
                return Err(error);
            }
        };
        observe_memory(&mut self.telemetry, source_memory);

        let expected = *self.step_fresh_counts.get(self.steps.len()).ok_or(
            ReducedSwirlNativeCudaError::State("too many WARP transitions"),
        )?;
        if self.pending_sources.len() > expected {
            return self.fail("fresh batch exceeds deterministic schedule");
        }
        if self.pending_sources.len() == expected {
            if let Err(error) = self.flush_pending_step() {
                self.poisoned = true;
                return Err(error);
            }
        }
        Ok(())
    }

    fn flush_pending_step(&mut self) -> Result<(), ReducedSwirlNativeCudaError> {
        self.ensure_live()?;
        let step_index = self.steps.len();
        let fresh_count =
            *self
                .step_fresh_counts
                .get(step_index)
                .ok_or(ReducedSwirlNativeCudaError::State(
                    "too many WARP transitions",
                ))?;
        if self.pending_sources.len() != fresh_count || self.pending_bindings.len() != fresh_count {
            return Err(ReducedSwirlNativeCudaError::State(
                "incomplete scheduled fresh batch",
            ));
        }
        let source_start = self
            .source_bindings
            .len()
            .checked_sub(fresh_count)
            .ok_or(ReducedSwirlNativeCudaError::State("source range"))?;
        observe_source_binding_batch(
            step_index,
            source_start,
            &self.pending_bindings,
            &mut self.challenger,
        )?;
        let fresh_openings = CudaReducedSwirlOpeningAdapter::new(StackedRsOpeningBackend::new(
            self.setup.cpu.code().hasher().clone(),
        ));
        let accumulator_openings = CudaResidentWarpOpeningBackend::new(
            self.setup.cpu.code().hasher().clone(),
            self.setup.device_ctx.clone(),
        );
        let step_transfer_start = pcie_counters::snapshot();
        let step_started = Instant::now();
        let sources = core::mem::take(&mut self.pending_sources);
        self.pending_bindings.clear();
        let prior = self.accumulator.take();
        let prior_instance = prior
            .as_ref()
            .map(|accumulator| accumulator.instance.clone());
        let (next, proof, mut record) = self
            .setup
            .warp
            .prove_reduced_constrained_code_step_recorded::<F, EF, _, _, _, _, _, _>(
                self.setup.cpu.relation(),
                self.setup.cpu.binding(),
                step_index,
                sources,
                prior,
                &mut self.challenger,
                &fresh_openings,
                &accumulator_openings,
                &self.accelerator,
            )?;
        if let Some(boundary) = finish_exact_finite_warp_call(&mut self.challenger, step_index) {
            record.transcript_phases.push(boundary);
        }

        // Re-run the native verifier at the call boundary. The recursive
        // wrapper consumes this verifier-derived authentication record; a
        // successful prover invocation or its diagnostic record is never
        // treated as authority.
        let verifier_transcript =
            self.verifier_transcript
                .take()
                .ok_or(ReducedSwirlNativeCudaError::State(
                    "missing incremental verifier transcript",
                ))?;
        let batch_start_checkpoint = verifier_transcript.checkpoint();
        let mut verifier_challenger = NativeWarpChallenger::<SC, _>::new(verifier_transcript);
        observe_source_binding_batch(
            step_index,
            source_start,
            &self.source_bindings[source_start..source_start + fresh_count],
            &mut verifier_challenger,
        )?;
        let verifier_transcript = verifier_challenger.into_inner();
        let vacc_start_checkpoint = verifier_transcript.checkpoint();
        let mut verifier_challenger = NativeWarpChallenger::<SC, _>::new(verifier_transcript);
        let fresh_verifier =
            StackedRsOpeningBackend::new_recording(self.setup.cpu.code().hasher().clone());
        // Use the resident backend's recording verifier implementation. Its
        // verification path is ordinary host-side Merkle checking, while its
        // type remains compatible with the resident CUDA prover-data type
        // selected by `CudaResidentWarpCode`.
        let accumulator_verifier = CudaResidentWarpOpeningBackend::new_recording(
            self.setup.cpu.code().hasher().clone(),
            self.setup.device_ctx.clone(),
        );
        let mut verification = self
            .setup
            .warp
            .verify_reduced_constrained_code_step_recorded::<F, EF, _, _, _, _>(
                self.setup.cpu.relation(),
                self.setup.cpu.binding(),
                step_index,
                self.verifier_prior.as_ref(),
                &self.authoritative_claims[source_start..source_start + fresh_count],
                &proof,
                &mut verifier_challenger,
                &fresh_verifier,
                &accumulator_verifier,
            )?;
        if let Some(boundary) = finish_exact_finite_warp_call(&mut verifier_challenger, step_index)
        {
            verification.transcript_phases.push(boundary);
        }
        let verifier_transcript = verifier_challenger.into_inner();
        let end_checkpoint = verifier_transcript.checkpoint();
        let (end_sample_count, end_state) =
            exact_checkpoint_state(&verifier_transcript.log, end_checkpoint)?;
        let (start_sample_count, start_state) = if step_index == 0 {
            (0, [F::ZERO; POSEIDON2_WIDTH])
        } else {
            exact_checkpoint_state(&verifier_transcript.log, batch_start_checkpoint)?
        };
        if verification.output_instance != next.instance
            || prior_instance.as_ref() != self.verifier_prior.as_ref()
            || !same_recorded_transition(&record, &verification)
        {
            return Err(ReducedSwirlNativeCudaError::State(
                "incremental native recorded verification mismatch",
            ));
        }
        let vacc_ms = elapsed_ms(step_started);
        let resident_accumulator_bytes = self
            .setup
            .warp
            .code()
            .accumulator_resident_bytes(&next)?
            .total_bytes;
        let expected_resident_bytes = self
            .setup
            .warp
            .code()
            .expected_resident_bytes()?
            .total_bytes;
        if resident_accumulator_bytes != expected_resident_bytes {
            return Err(ReducedSwirlNativeCudaError::State(
                "resident accumulator allocation accounting",
            ));
        }
        let memory = memory_snapshot()?;
        observe_memory(&mut self.telemetry, memory);
        let vacc_transfers = transfer_delta(step_transfer_start, pcie_counters::snapshot());
        let transfers = self
            .pending_projection_transfers
            .saturating_add(vacc_transfers);
        self.telemetry.projected_message_bytes = self
            .telemetry
            .projected_message_bytes
            .saturating_add(self.pending_message_bytes);
        self.telemetry.projected_codeword_bytes = self
            .telemetry
            .projected_codeword_bytes
            .saturating_add(self.pending_codeword_bytes);
        self.telemetry.steps.push(ReducedSwirlCudaStepTelemetry {
            step_index,
            source_start,
            fresh_count,
            source_projection_ms: self.pending_projection_ms,
            vacc_ms,
            projected_message_bytes: self.pending_message_bytes,
            projected_codeword_bytes: self.pending_codeword_bytes,
            resident_accumulator_bytes,
            boundary_live_gpu_bytes: memory.live_bytes,
            boundary_driver_used_gpu_bytes: memory.driver_used_bytes,
            source_projection_transfers: self.pending_projection_transfers,
            vacc_transfers,
            transfers,
        });
        self.telemetry.source_projection_ms += self.pending_projection_ms;
        self.telemetry.vacc_ms += vacc_ms;
        self.pending_projection_ms = 0.0;
        self.pending_projection_transfers = ReducedSwirlCudaTransferTelemetry::default();
        self.pending_message_bytes = 0;
        self.pending_codeword_bytes = 0;
        self.accumulator = Some(next);
        self.steps.push(proof);
        self.transition_records.push(record);
        self.verifier_prior = Some(verification.output_instance.clone());
        self.verifier_transcript = Some(verifier_transcript);
        self.latest_completed_step = Some(ReducedSwirlNativeCudaCompletedStepState {
            call_index: step_index,
            source_start,
            fresh_count,
            prior_instance,
            verification,
            batch_start_checkpoint,
            vacc_start_checkpoint,
            end_checkpoint,
            start_sample_count,
            start_state,
            end_sample_count,
            end_state,
        });
        Ok(())
    }

    pub fn finish(mut self) -> Result<ReducedSwirlNativeCudaOutput, ReducedSwirlNativeCudaError> {
        self.ensure_live()?;
        if !self.pending_sources.is_empty()
            || self.source_bindings.len() != self.expected_source_count
            || self.steps.len() != self.step_fresh_counts.len()
        {
            return Err(ReducedSwirlNativeCudaError::State(
                "incomplete reduced-SWIRL CUDA stream",
            ));
        }
        let block_manifest_digest = reduced_swirl_manifest_digest(&self.source_bindings)?;
        let statement = ReducedSwirlNativeStatement {
            protocol_version: REDUCED_SWIRL_NATIVE_PROTOCOL_VERSION,
            block_manifest_digest,
            source_bindings: self.source_bindings,
        };
        statement.validate()?;
        observe_manifest_footer(&block_manifest_digest, &mut self.challenger);
        let accumulator = self
            .accumulator
            .take()
            .ok_or(ReducedSwirlNativeCudaError::State(
                "missing final accumulator",
            ))?;
        let final_instance = accumulator.instance.clone();
        if self.verifier_prior.as_ref() != Some(&final_instance)
            || self
                .latest_completed_step
                .as_ref()
                .map(|step| step.call_index + 1)
                != Some(self.steps.len())
        {
            return Err(ReducedSwirlNativeCudaError::State(
                "incomplete incremental native verification",
            ));
        }
        let vacc = ReducedWarpVaccRootProof {
            params: self.setup.warp.params(),
            schedule: self.setup.warp.schedule(),
            steps: self.steps,
            final_instance: final_instance.clone(),
        };
        let descriptor = TerminalDescriptor::from_whir_initial_rs(
            final_instance.rt,
            self.setup.cpu.code(),
            &self.setup.whir,
            <EF as BasedVectorSpace<F>>::DIMENSION,
        );
        let terminal_statement = derive_swirl_constrained_rs_terminal_statement(
            self.setup.cpu.relation(),
            self.setup.cpu.code(),
            &final_instance,
        )
        .map_err(|error| ReducedSwirlNativeCudaError::Terminal(format!("{error:?}")))?;
        #[cfg(debug_assertions)]
        {
            let (&normalized_target, point) =
                final_instance
                    .beta
                    .split_last()
                    .ok_or(ReducedSwirlNativeCudaError::State(
                        "missing normalized constrained-code target",
                    ))?;
            let actual = accumulator
                .witness
                .prepared_message
                .as_ref()
                .ok_or(ReducedSwirlNativeCudaError::State(
                    "missing resident accumulator message",
                ))?
                .evaluate_mle(point)
                .map_err(ReducedSwirlNativeCudaError::State)?;
            if actual != normalized_target + final_instance.eta {
                return Err(ReducedSwirlNativeCudaError::State(
                    "final WARP accumulator violates its constrained-code message claim",
                ));
            }
        }
        let mut transcript = self.challenger.into_inner();
        let terminal_prover = CudaResidentTerminalWhirProver::<BabyBearPoseidon2HashScheme>::new(
            self.setup.device_ctx.clone(),
        );
        let terminal_started = Instant::now();
        let terminal_transfer_start = pcie_counters::snapshot();
        let (same_root_whir, terminal_instrumentation) = terminal_prover
            .prove_terminal_constrained_whir_resident_instrumented::<
                SC,
                FieldElementDigestObserver,
                _,
            >(
                &self.setup.whir,
                self.setup.cpu.code(),
                &descriptor,
                &accumulator,
                &terminal_statement,
                &mut transcript,
            )
            .map_err(|error| ReducedSwirlNativeCudaError::Terminal(format!("{error:?}")))?;
        self.telemetry.terminal_ms = elapsed_ms(terminal_started);
        self.telemetry.terminal_transfers =
            transfer_delta(terminal_transfer_start, pcie_counters::snapshot());
        let terminal_memory = memory_snapshot()?;
        self.telemetry.terminal_boundary_live_gpu_bytes = terminal_memory.live_bytes;
        self.telemetry.terminal_boundary_driver_used_gpu_bytes = terminal_memory.driver_used_bytes;
        observe_memory(&mut self.telemetry, terminal_memory);
        terminal_instrumentation
            .assert_canonical_single_root()
            .map_err(|error| ReducedSwirlNativeCudaError::Terminal(format!("{error:?}")))?;
        copy_terminal_telemetry(&mut self.telemetry, terminal_instrumentation);
        self.telemetry.source_count = self.expected_source_count;
        self.telemetry.vacc_step_count = self.transition_records.len();
        self.telemetry.total_wall_ms = elapsed_ms(self.started);
        self.telemetry.transfers = transfer_delta(self.transfer_start, pcie_counters::snapshot());
        observe_memory(&mut self.telemetry, memory_snapshot()?);
        self.telemetry
            .assert_no_duplicate_payload_pipeline()
            .map_err(ReducedSwirlNativeCudaError::State)?;
        let swirl_power_batch_security =
            ReducedSwirlPowerBatchSecurityBudget::derive(&self.authoritative_claims)?;

        Ok(ReducedSwirlNativeCudaOutput {
            native: ReducedSwirlNativeProverOutput {
                proof: ReducedSwirlNativeProof {
                    statement,
                    vacc,
                    terminal: ReducedSwirlNativeTerminalProof {
                        descriptor,
                        same_root_whir,
                    },
                },
                authoritative_claims: self.authoritative_claims,
                transition_records: self.transition_records,
                security: *self.setup.cpu.security(),
                swirl_power_batch_security,
            },
            telemetry: self.telemetry,
        })
    }

    fn ensure_live(&self) -> Result<(), ReducedSwirlNativeCudaError> {
        if self.poisoned {
            return Err(ReducedSwirlNativeCudaError::State("poisoned prover"));
        }
        Ok(())
    }

    fn fail<T>(&mut self, message: &'static str) -> Result<T, ReducedSwirlNativeCudaError> {
        self.poisoned = true;
        Err(ReducedSwirlNativeCudaError::State(message))
    }
}

/// Supplier-oriented adapter for callers that already stream pending owners.
pub fn prove_reduced_swirl_native_cuda_streaming<Supply, SupplyError>(
    setup: &ReducedSwirlNativeCudaSetup,
    statement: ReducedSwirlNativeStatement,
    mut supply: Supply,
) -> Result<ReducedSwirlNativeCudaOutput, ReducedSwirlNativeCudaError>
where
    Supply: FnMut(
        usize,
    ) -> Result<
        (
            ReducedSwirlCudaPendingWitness,
            Vec<Vec<EF>>,
            NativeWarpChallenger<SC, DuplexSpongeRecorder>,
        ),
        SupplyError,
    >,
    SupplyError: Display,
{
    statement.validate()?;
    let source_count = statement.source_bindings.len();
    let mut stream = ReducedSwirlNativeCudaStream::new(setup, source_count)?;
    for source_index in 0..source_count {
        let (pending, openings, mut source_challenger) = supply(source_index)
            .map_err(|error| ReducedSwirlNativeCudaError::SourceProvider(error.to_string()))?;
        stream.push_pending(
            statement.source_bindings[source_index],
            pending,
            &openings,
            &mut source_challenger,
        )?;
    }
    let output = stream.finish()?;
    if output.native.proof.statement != statement {
        return Err(ReducedSwirlNativeCudaError::State(
            "streamed public statement mismatch",
        ));
    }
    Ok(output)
}

fn validate_source(
    setup: &ReducedSwirlNativeCudaSetup,
    source: &CudaReducedSwirlConstrainedCodeSource,
    _index: usize,
) -> Result<(), ReducedSwirlNativeCudaError> {
    let claim = source.authoritative_claim_root().0;
    if source.relation() != setup.cpu.relation()
        || &claim.binding != setup.cpu.binding()
        || &claim.commitment != source.authoritative_claim_root().1
        || claim.commitment.log_message_len != setup.cpu.code().log_message_len()
        || claim.commitment.log_codeword_len != setup.cpu.code().log_codeword_len()
        || claim.commitment.rows_per_query != setup.cpu.code().rows_per_query()
        || claim.alpha.len() != setup.cpu.code().log_codeword_len()
        || claim.alpha.iter().any(|value| *value != EF::ZERO)
        || source.message_len() != setup.cpu.code().message_len()
        || source.codeword_len() != setup.cpu.code().codeword_len()
    {
        return Err(ReducedSwirlNativeCudaError::State(
            "source does not match trusted constrained-code setup",
        ));
    }
    <openvm_stark_backend::warp_accum::SwirlConstrainedRsRelation<EF> as
        ReducedConstrainedCodeRelation<F, EF>>::validate_public_claim(
        setup.cpu.relation(),
        &claim.beta,
        claim.eta,
    )
    .map_err(ReducedSwirlNativeCudaError::State)?;
    Ok(())
}

fn same_recorded_transition(
    prover: &WarpVaccStepProverRecord<EF, Digest>,
    verifier: &ReducedSwirlStepVerification,
) -> bool {
    prover.output_instance == verifier.output_instance
        && prover.transcript_phases == verifier.transcript_phases
        && prover.twin == verifier.twin
        && prover.ood == verifier.ood
        && prover.shifts == verifier.shifts
        && prover.batching == verifier.batching
}

fn exact_checkpoint_state(
    transcript: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    checkpoint: TranscriptCheckpoint,
) -> Result<(usize, [F; POSEIDON2_WIDTH]), ReducedSwirlNativeCudaError> {
    let Some(event) = checkpoint
        .events
        .checked_sub(1)
        .and_then(|index| transcript.events().get(index))
    else {
        return Err(ReducedSwirlNativeCudaError::State(
            "non-canonical incremental transcript checkpoint",
        ));
    };
    if checkpoint.operations == 0
        || checkpoint.operations > transcript.len()
        || checkpoint.permutations >= transcript.perm_results().len()
        || event.operation_range.end != checkpoint.operations
        || event.permutation_range.end != checkpoint.permutations
    {
        return Err(ReducedSwirlNativeCudaError::State(
            "non-canonical incremental transcript checkpoint",
        ));
    }
    let sample_count = transcript.samples()[..checkpoint.operations]
        .iter()
        .rev()
        .take_while(|&&is_sample| is_sample)
        .count();
    if sample_count == 0 || sample_count > CHUNK {
        return Err(ReducedSwirlNativeCudaError::State(
            "incremental checkpoint has no trailing sample",
        ));
    }
    Ok((
        sample_count,
        transcript.perm_results()[checkpoint.permutations],
    ))
}

fn observe_statement_header<Ch>(
    setup: &ReducedSwirlNativeCudaSetup,
    protocol_version: u32,
    source_count: usize,
    challenger: &mut Ch,
) where
    Ch: AlgebraicChallenger<EF>,
{
    observe_bytes(REDUCED_SWIRL_NATIVE_TRANSCRIPT_TAG, challenger);
    observe_u64(u64::from(protocol_version), challenger);
    observe_u64(source_count as u64, challenger);
    for value in [
        setup.cpu.code().log_message_len(),
        setup.cpu.code().log_codeword_len(),
        setup.cpu.code().rows_per_query(),
        setup.warp.schedule().arity,
        setup.warp.params().num_ood,
        setup.warp.params().num_shift_queries,
        setup.warp.params().batching_arity(),
        setup.cpu.family_target_bits(),
    ] {
        observe_u64(value as u64, challenger);
    }
    observe_bytes(&setup.cpu.binding().source_domain, challenger);
    challenger.observe_slice(&setup.cpu.binding().relation_binding);
    challenger.observe_slice(&setup.cpu.binding().code_binding);
}

fn observe_manifest_footer<Ch>(manifest_digest: &Digest, challenger: &mut Ch)
where
    Ch: AlgebraicChallenger<EF>,
{
    observe_bytes(REDUCED_SWIRL_NATIVE_MANIFEST_FOOTER_TAG, challenger);
    observe_digest(manifest_digest, challenger);
}

fn observe_source_binding_batch<Ch>(
    step_index: usize,
    source_start: usize,
    source_bindings: &[Digest],
    challenger: &mut Ch,
) -> Result<(), ReducedSwirlNativeCudaError>
where
    Ch: AlgebraicChallenger<EF>,
{
    if source_bindings.is_empty() || source_bindings.iter().any(is_zero_digest) {
        return Err(ReducedSwirlNativeCudaError::State(
            "invalid source binding batch",
        ));
    }
    observe_bytes(REDUCED_SWIRL_NATIVE_SOURCE_BATCH_TAG, challenger);
    observe_u64(step_index as u64, challenger);
    observe_u64(source_start as u64, challenger);
    observe_u64(source_bindings.len() as u64, challenger);
    for digest in source_bindings {
        observe_digest(digest, challenger);
    }
    Ok(())
}

fn is_zero_digest(digest: &Digest) -> bool {
    digest.iter().all(|value| *value == F::ZERO)
}

fn observe_digest<Ch>(digest: &Digest, challenger: &mut Ch)
where
    Ch: AlgebraicChallenger<EF>,
{
    for &value in digest {
        challenger.observe(EF::from(value));
    }
}

fn observe_bytes<Ch>(bytes: &[u8], challenger: &mut Ch)
where
    Ch: AlgebraicChallenger<EF>,
{
    observe_u64(bytes.len() as u64, challenger);
    for &byte in bytes {
        challenger.observe(EF::from_u8(byte));
    }
}

fn observe_u64<Ch>(value: u64, challenger: &mut Ch)
where
    Ch: AlgebraicChallenger<EF>,
{
    for byte in value.to_le_bytes() {
        challenger.observe(EF::from_u8(byte));
    }
}

fn elapsed_ms(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn memory_snapshot() -> Result<DeviceMemorySnapshot, ReducedSwirlNativeCudaError> {
    device_memory_snapshot()
        .map_err(|error| ReducedSwirlNativeCudaError::Telemetry(format!("{error:?}")))
}

fn observe_memory(telemetry: &mut ReducedSwirlNativeCudaTelemetry, snapshot: DeviceMemorySnapshot) {
    telemetry.peak_boundary_live_gpu_bytes = telemetry
        .peak_boundary_live_gpu_bytes
        .max(snapshot.live_bytes);
    telemetry.peak_lifetime_live_gpu_bytes = telemetry
        .peak_lifetime_live_gpu_bytes
        .max(snapshot.lifetime_peak_live_bytes);
    telemetry.peak_driver_used_gpu_bytes = telemetry
        .peak_driver_used_gpu_bytes
        .max(snapshot.driver_used_bytes);
    telemetry.driver_total_gpu_bytes = snapshot.driver_total_bytes;
}

fn transfer_delta(
    start: (u64, u64, u64, u64, u64, u64),
    end: (u64, u64, u64, u64, u64, u64),
) -> ReducedSwirlCudaTransferTelemetry {
    ReducedSwirlCudaTransferTelemetry {
        h2d_calls: end.0.saturating_sub(start.0),
        h2d_bytes: end.1.saturating_sub(start.1),
        d2h_calls: end.2.saturating_sub(start.2),
        d2h_bytes: end.3.saturating_sub(start.3),
        d2d_calls: end.4.saturating_sub(start.4),
        d2d_bytes: end.5.saturating_sub(start.5),
    }
}

fn copy_terminal_telemetry(
    telemetry: &mut ReducedSwirlNativeCudaTelemetry,
    terminal: ResidentTerminalWhirInstrumentation,
) {
    telemetry.terminal_reused_initial_roots = terminal.reused_initial_roots;
    telemetry.terminal_accumulator_reencodes = terminal.accumulator_reencodes;
    telemetry.terminal_accumulator_recommits = terminal.accumulator_recommits;
    telemetry.terminal_full_message_d2h_bytes = terminal.full_accumulator_message_d2h_bytes;
    telemetry.terminal_bounded_proof_d2h_bytes = terminal.bounded_proof_d2h_bytes;
}

#[cfg(all(test, feature = "test-utils"))]
mod tests {
    use openvm_circuit::system::connector::DEFAULT_SUSPEND_EXIT_CODE;
    use openvm_continuations::circuit::{
        reduced_swirl_source_receipt::ReducedSwirlSourceReceiptBlock,
        reduced_swirl_warp::ReducedSwirlSourceReceiptBus,
    };
    use openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
    use openvm_recursion_circuit::{
        native_warp::{ReducedSwirlSourceAuthorityBus, ReducedSwirlSourceProfile},
        system::BusIndexManager,
    };
    use openvm_stark_backend::{
        prover::DeviceDataTransporter,
        test_utils::{default_test_params_small, PreprocessedFibFixture, TestFixture},
        StarkEngine,
    };

    use super::*;
    use crate::prover::native_warp::{
        reduced_swirl_boundary::{PendingConstrainedCodePublicClaim, ReducedSwirlPrefix},
        reduced_swirl_native::{
            reduced_swirl_source_challenger, verify_reduced_swirl_native_recorded,
        },
        reduced_swirl_source_receipt::{
            reduced_swirl_source_receipt_profile, ProductionReducedSwirlSourceReceiptComponent,
        },
        reduced_swirl_vacc_component::{
            build_reduced_swirl_vacc_aggregate_record, ProductionReducedSwirlVaccComponent,
        },
    };

    #[test]
    fn genuine_multi_step_cuda_stream_reuses_roots_and_cpu_verifies() -> eyre::Result<()> {
        let params = default_test_params_small();
        let mut engine = BabyBearPoseidon2GpuEngine::new(params.clone());
        engine.device_mut().set_cache_rs_code_matrix(true);
        let selectors = vec![true; 1 << 5];
        let key_fixture = PreprocessedFibFixture::new(0, 1, selectors.clone());
        let (host_pk, vk) = key_fixture.keygen(&engine);
        let device_pk = engine.device().transport_pk_to_device(&host_pk);
        let source_count = 3;
        let setup = ReducedSwirlNativeCudaSetup::new(
            engine.config(),
            &params,
            2,
            16,
            source_count,
            engine.device().device_ctx.clone(),
        )?;
        let terminal_setup =
            openvm_recursion_circuit::native_warp::ReducedSwirlTerminalProductionSetup::from_native_fixed_params(
                params.clone(),
                2,
                16,
                source_count,
            )?;
        let maximum_roots_per_source = 1usize
            + vk.inner
                .per_air
                .iter()
                .map(|air| air.params.width.cached_mains.len())
                .sum::<usize>();
        let source_profile = ReducedSwirlSourceProfile {
            maximum_sources: 2,
            maximum_roots_per_source,
            maximum_openings_per_source: params.w_stack,
            l_skip: params.l_skip,
            n_stack: params.n_stack,
            log_blowup: params.log_blowup,
            log_commit_rows_per_query: params.log_commit_rows_per_query,
        };
        let receipt_profile = reduced_swirl_source_receipt_profile(
            &vk,
            source_profile,
            terminal_setup.protocol_digest(),
            DEFAULT_SUSPEND_EXIT_CODE,
        )?;
        let mut buses = BusIndexManager::from_next_bus_idx(0);
        let receipt_bus = ReducedSwirlSourceReceiptBus::new(buses.new_bus_idx());
        let authority_bus = ReducedSwirlSourceAuthorityBus::new(buses.new_bus_idx());
        let source_component = ProductionReducedSwirlSourceReceiptComponent::<2>::new(
            Arc::new(vk.clone()),
            receipt_profile,
            params.clone(),
            receipt_bus,
            authority_bus,
            buses,
        )?;
        let vacc_component =
            ProductionReducedSwirlVaccComponent::from_native_setup_for_transition_leaf(
                setup.cpu_setup(),
                params.clone(),
                &source_component,
                BusIndexManager::from_next_bus_idx(source_component.inner().next_bus_idx()),
            )?;
        let mut stream = ReducedSwirlNativeCudaStream::new(&setup, source_count)?;
        let mut retained = Vec::new();
        let mut wrapper_claims = Vec::new();
        let mut source_bindings = Vec::new();
        let mut incremental_records = Vec::new();
        let mut incremental_checkpoints = Vec::new();
        let mut incremental_receipts = Vec::new();
        let mut receipt_sources = Vec::new();
        for (source_index, (a, b)) in [(0, 1), (2, 3), (5, 8)].into_iter().enumerate() {
            let fixture = PreprocessedFibFixture::new(a, b, selectors.clone());
            let proving_context = engine
                .device()
                .transport_proving_ctx_to_device(&fixture.generate_proving_ctx());
            let reduction = engine
                .prover()
                .prove_native_stacking_reduction(&device_pk, proving_context)?;
            let prefix = ReducedSwirlPrefix::from_native_reduction(&vk, source_index, reduction)?;
            let pending_public = PendingConstrainedCodePublicClaim {
                metadata: prefix.pending_witness().metadata().clone(),
                swirl_tilde_u: prefix.pending_witness().terminal_point().clone(),
                stacking_openings: prefix.retained().stacking_proof.stacking_openings.clone(),
            };
            let manifest_prefix = prefix.manifest_prefix(source_index)?;
            let (retained_prefix, pending_witness) = prefix.into_parts();
            let mut source_challenger = reduced_swirl_source_challenger();
            let source = CudaReducedSwirlConstrainedCodeSource::try_from_pending(
                pending_witness,
                &retained_prefix.stacking_proof.stacking_openings,
                &mut source_challenger,
                engine.device().device_ctx.clone(),
            )?;
            let wrapper_claim = pending_public.authoritative_claim_from_backend(&source.claim())?;
            let source_binding = manifest_prefix.digest_with_claim(&wrapper_claim)?;
            stream.push_source(source_binding, source)?;
            retained.push(retained_prefix);
            wrapper_claims.push(wrapper_claim);
            source_bindings.push(source_binding);
            if stream.completed_step_count() > incremental_records.len() {
                let completed = stream.latest_completed_step()?.ok_or_else(|| {
                    eyre::eyre!("completed-step count has no completed-step view")
                })?;
                assert_eq!(completed.call_index, incremental_records.len());
                assert_eq!(completed.proof.fresh_count(), completed.fresh_count);
                assert_eq!(
                    completed.authoritative_claims.len(),
                    completed.source_bindings.len()
                );
                assert_eq!(
                    completed.transcript_prefix.len(),
                    completed.end_checkpoint.operations
                );
                incremental_records.push(completed.verification.clone());
                incremental_checkpoints.push(completed.end_checkpoint);
                let source_end = completed.source_start + completed.fresh_count;
                let source_packet = source_component.generate_cpu_in_flight_inline_packet(
                    &vk,
                    &retained[completed.source_start..source_end],
                    &wrapper_claims[completed.source_start..source_end],
                    source_count,
                    completed.source_start.try_into()?,
                    completed.source_bindings,
                    &[],
                )?;
                let packet = vacc_component.generate_transition_cpu_packet_from_completed_step(
                    &completed,
                    &source_packet.block,
                )?;
                let mut tampered_source_block = source_packet.block.clone();
                tampered_source_block.sources[0].entry_digest[0] += F::ONE;
                assert!(matches!(
                    vacc_component.generate_transition_cpu_packet_from_completed_step(
                        &completed,
                        &tampered_source_block,
                    ),
                    Err(crate::prover::native_warp::reduced_swirl_vacc_component::ReducedSwirlVaccComponentError::SourceBinding { .. })
                ));
                incremental_receipts.push((
                    packet.receipt,
                    packet.entry_digests,
                    packet.manifest_digest,
                ));
                receipt_sources.extend(source_packet.block.sources);
            }
        }
        let output = stream.finish()?;
        let statement = ReducedSwirlNativeStatement {
            protocol_version: REDUCED_SWIRL_NATIVE_PROTOCOL_VERSION,
            block_manifest_digest: reduced_swirl_manifest_digest(&source_bindings)?,
            source_bindings,
        };
        assert_eq!(retained.len(), source_count);
        assert_eq!(output.native.proof.statement, statement);
        assert_eq!(output.native.proof.vacc.steps.len(), 2);
        assert_eq!(incremental_records.len(), 2);
        assert_eq!(incremental_receipts.len(), 2);
        assert_eq!(output.telemetry.maximum_pending_sources, 2);
        assert_eq!(output.telemetry.maximum_scheduled_fresh_count, 2);
        assert!(output
            .telemetry
            .assert_no_duplicate_payload_pipeline()
            .is_ok());

        let verified = verify_reduced_swirl_native_recorded(
            setup.cpu_setup(),
            &statement,
            &output.native.authoritative_claims,
            &output.native.proof,
        )?;
        assert_eq!(
            verified.final_instance,
            output.native.proof.vacc.final_instance
        );
        assert_eq!(verified.terminal.root, verified.final_instance.rt);
        assert_eq!(incremental_records, verified.transition_records);
        let source_block = ReducedSwirlSourceReceiptBlock {
            source_offset: 0,
            sources: receipt_sources,
            manifest_digest: statement.block_manifest_digest,
        };
        let aggregate_record = build_reduced_swirl_vacc_aggregate_record(
            vacc_component.profile(),
            &output.native,
            &verified,
            &source_block,
        )?;
        for (call_index, expected) in incremental_receipts.iter().enumerate() {
            let legacy = vacc_component.generate_transition_cpu_packet(
                &output.native,
                &verified,
                &aggregate_record,
                call_index,
            )?;
            assert_eq!(&legacy.receipt, &expected.0);
            assert_eq!(&legacy.entry_digests, &expected.1);
            assert_eq!(&legacy.manifest_digest, &expected.2);
        }
        for (checkpoint, record) in incremental_checkpoints
            .iter()
            .zip(&verified.transition_records)
        {
            let boundary = record
                .transcript_phases
                .iter()
                .find(|phase| {
                    matches!(
                        phase.phase,
                        openvm_stark_backend::warp_accum::NativeTranscriptPhase::ExactFiniteCallBoundary { .. }
                    )
                })
                .ok_or_else(|| eyre::eyre!("missing incremental call boundary"))?;
            assert_eq!(checkpoint.operations, boundary.operation_range.end);
            assert_eq!(checkpoint.events, boundary.event_range.end);
            assert_eq!(checkpoint.permutations, boundary.permutation_range.end);
        }
        Ok(())
    }
}

#[cfg(test)]
mod invariant_tests {
    use super::*;

    #[test]
    fn canonical_resident_pipeline_telemetry_is_accepted() {
        let telemetry = ReducedSwirlNativeCudaTelemetry {
            terminal_reused_initial_roots: 1,
            ..Default::default()
        };
        assert!(telemetry.assert_no_duplicate_payload_pipeline().is_ok());
    }

    #[test]
    fn duplicate_or_host_staged_payload_is_rejected() {
        let mut telemetry = ReducedSwirlNativeCudaTelemetry {
            terminal_reused_initial_roots: 1,
            ..Default::default()
        };
        telemetry.fresh_reencodes = 1;
        assert!(telemetry.assert_no_duplicate_payload_pipeline().is_err());

        telemetry.fresh_reencodes = 0;
        telemetry.terminal_full_message_d2h_bytes = size_of::<EF>();
        assert!(telemetry.assert_no_duplicate_payload_pipeline().is_err());
    }

    #[test]
    fn transfer_delta_is_monotone_and_saturating() {
        let delta = transfer_delta((4, 40, 7, 70, 9, 90), (6, 64, 5, 50, 12, 123));
        assert_eq!(delta.h2d_calls, 2);
        assert_eq!(delta.h2d_bytes, 24);
        assert_eq!(delta.d2h_calls, 0);
        assert_eq!(delta.d2h_bytes, 0);
        assert_eq!(delta.d2d_calls, 3);
        assert_eq!(delta.d2d_bytes, 33);
    }
}

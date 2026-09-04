//! First-class partial recursive verifier for the backend `LogUpOnly` prefix.
//!
//! This assembly intentionally ends after GKR and the batch-constraint
//! sumcheck. The caller supplies authenticated column claims and consumes the
//! exported endpoint; stacking and WHIR are not instantiated.

use std::{iter, sync::Arc};

use openvm_cpu_backend::CpuBackend;
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{
    interaction::BusIndex, keygen::types::MultiStarkVerifyingKey, proof::Proof,
    prover::AirProvingContext, transcript::TranscriptLog, AirRef, FiatShamirTranscript,
    StarkProtocolConfig, SystemParams, TranscriptHistory,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, F};
use p3_field::PrimeCharacteristicRing;

use crate::{
    batch_constraint::{
        expr_eval::CachedTraceRecord, BatchConstraintModule, PartialBatchConstraintExports,
        LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX,
    },
    bus::CertifiedTranscriptCheckpointBus,
    gkr::GkrModule,
    primitives::{
        exp_bits_len::{ExpBitsLenAir, ExpBitsLenCpuTraceGenerator},
        pow::{PowerCheckerAir, PowerCheckerCpuTraceGenerator},
    },
    proof_shape::ProofShapeModule,
    system::{
        frame::MultiStarkVkeyFrame, AggregationSubCircuit, AirModule, BusIndexManager,
        BusInventory, CachedTraceCtx, Preflight, RebasedTranscriptPreflight,
        RetainedLogUpOnlyProof, TraceGenModule, VerifierEquationMode, POW_CHECKER_HEIGHT,
    },
    transcript::{Poseidon2BusOwner, Poseidon2MultibusInputs, TranscriptModule},
};

/// Buses that the enclosing protocol-v19 circuit must connect.
///
/// `rebased_start_bus` receives the certified manifest checkpoint,
/// `column_claims_bus` receives current/rotated openings from the source
/// manifest, and `endpoint_bus` must be consumed by the segment binding AIR.
#[derive(Clone, Copy, Debug)]
pub struct LogUpOnlyPartialVerifierExports {
    pub rebased_start_bus: crate::proof_shape::bus::RebasedTranscriptStartBus,
    pub column_claims_bus: crate::bus::ColumnClaimsBus,
    pub endpoint_bus: crate::batch_constraint::bus::BatchConstraintEndpointBus,
    pub opening_point_bus: crate::batch_constraint::bus::LogUpOnlyOpeningPointBus,
}

/// Partial-verifier contexts with the physical Poseidon owner omitted.
/// `contexts` matches [`LogUpOnlyPartialVerifier::airs_without_poseidon`]
/// exactly; the two input vectors form one logical owner entry in a parent
/// [`Poseidon2MultibusInputs`] packet.
pub struct LogUpOnlySharedPoseidonContexts<SC: StarkProtocolConfig<F = F>> {
    pub contexts: Vec<AirProvingContext<CpuBackend<SC>>>,
    pub poseidon2_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub poseidon2_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
}

impl<SC: StarkProtocolConfig<F = F>> LogUpOnlySharedPoseidonContexts<SC> {
    #[must_use]
    pub fn grouped_input(self) -> (Vec<[F; POSEIDON2_WIDTH]>, Vec<[F; POSEIDON2_WIDTH]>) {
        (
            self.poseidon2_permutation_inputs,
            self.poseidon2_compression_inputs,
        )
    }
}

/// The recursion-side verifier for exactly
/// `verify_logup_only_prefix`, without stacking or WHIR.
pub struct LogUpOnlyPartialVerifier<const MAX_NUM_PROOFS: usize> {
    bus_inventory: BusInventory,
    bus_idx_manager: BusIndexManager,
    transcript: TranscriptModule,
    proof_shape: ProofShapeModule,
    gkr: GkrModule,
    batch_constraint: BatchConstraintModule,
}

impl<const MAX_NUM_PROOFS: usize> LogUpOnlyPartialVerifier<MAX_NUM_PROOFS> {
    #[must_use]
    pub fn new(
        child_mvk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        has_cached: bool,
    ) -> Self {
        let mut bus_idx_manager = BusIndexManager::new();
        let bus_inventory = BusInventory::new(&mut bus_idx_manager);
        let transcript = TranscriptModule::new(
            bus_inventory.clone(),
            child_mvk.inner.params.clone(),
            false,
            true,
        );
        let frame = MultiStarkVkeyFrame::from(child_mvk);
        let mut proof_shape = ProofShapeModule::new_rebased_logup_only(
            &frame,
            &mut bus_idx_manager,
            bus_inventory.clone(),
            false,
            MAX_NUM_PROOFS,
        );
        proof_shape.set_partial_assembly_exports(0);
        let gkr = GkrModule::new_with_equation_mode(
            child_mvk,
            &mut bus_idx_manager,
            bus_inventory.clone(),
            VerifierEquationMode::LogUpOnly,
        );
        let batch_constraint = BatchConstraintModule::new_with_equation_mode(
            child_mvk,
            &mut bus_idx_manager,
            bus_inventory.clone(),
            MAX_NUM_PROOFS,
            has_cached,
            VerifierEquationMode::LogUpOnly,
        );
        Self {
            bus_inventory,
            bus_idx_manager,
            transcript,
            proof_shape,
            gkr,
            batch_constraint,
        }
    }

    /// Bind the no-cached symbolic-expression trace to the exact child-VK DAG
    /// digest while keeping that setup constant out of the outer proof's
    /// public-value schedule.
    pub fn bind_fixed_dag_commit(
        &mut self,
        expected: [F; openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE],
    ) -> Result<(), &'static str> {
        self.batch_constraint.bind_fixed_dag_commit(expected)
    }

    #[must_use]
    pub fn exports(&self) -> LogUpOnlyPartialVerifierExports {
        let PartialBatchConstraintExports {
            endpoint_bus,
            column_claims_bus,
            opening_point_bus,
        } = self
            .batch_constraint
            .partial_exports()
            .expect("LogUpOnly batch module must expose a partial endpoint");
        LogUpOnlyPartialVerifierExports {
            rebased_start_bus: self
                .proof_shape
                .rebased_start_bus()
                .expect("rebased proof shape must expose its start bus"),
            column_claims_bus,
            endpoint_bus,
            opening_point_bus,
        }
    }

    /// Ask the resumed transcript AIR to certify two row-aligned checkpoints
    /// per proof. Protocol v19 uses kind zero for the LogUp endpoint and kind
    /// one for the end of the SWIRL one-shot stream.
    pub fn set_checkpoint_state_bus(
        &mut self,
        checkpoint_state_bus: CertifiedTranscriptCheckpointBus,
    ) {
        self.transcript
            .set_checkpoint_state_bus(checkpoint_state_bus);
    }

    /// The logically independent Poseidon lookup-bus pair owned by this
    /// verifier. Parent assemblies must preserve its position relative to the
    /// matching grouped input packet.
    #[must_use]
    pub fn poseidon2_bus_owner(&self) -> Poseidon2BusOwner {
        self.transcript.poseidon2_bus_owner()
    }

    /// AIR order with only this verifier's physical Poseidon table removed.
    /// Transcript and Merkle AIRs remain in their ordinary positions.
    #[must_use]
    pub fn airs_without_poseidon<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        let mut airs = <Self as AggregationSubCircuit>::airs::<SC>(self);
        airs.remove(self.poseidon_air_index::<SC>());
        airs
    }

    /// Index occupied by the local Poseidon owner in the compatibility AIR
    /// order returned by [`AggregationSubCircuit::airs`].
    #[must_use]
    pub fn poseidon_air_index<SC: StarkProtocolConfig<F = F>>(&self) -> usize {
        self.batch_constraint.airs::<SC>().len() + 1
    }

    #[must_use]
    pub fn multi_bus_poseidon_air<SC: StarkProtocolConfig<F = F>>(
        &self,
        owners: &[Poseidon2BusOwner],
    ) -> AirRef<SC> {
        self.transcript.multi_bus_poseidon2_air_for_owners(owners)
    }

    pub fn build_poseidon2_multibus_traces(
        &self,
        grouped_inputs: Poseidon2MultibusInputs,
    ) -> Option<Vec<p3_matrix::dense::RowMajorMatrix<F>>> {
        self.transcript
            .build_poseidon2_multibus_traces(grouped_inputs)
    }

    pub fn build_poseidon2_multibus_sharded_traces(
        &self,
        grouped_inputs: Poseidon2MultibusInputs,
        shard_count: usize,
        max_rows: usize,
    ) -> Option<Vec<p3_matrix::dense::RowMajorMatrix<F>>> {
        self.transcript.build_poseidon2_multibus_sharded_traces(
            grouped_inputs,
            shard_count,
            max_rows,
        )
    }

    pub fn build_local_poseidon2_trace(
        &self,
        permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    ) -> Option<p3_matrix::dense::RowMajorMatrix<F>> {
        self.transcript
            .build_poseidon2_trace(permutation_inputs, compression_inputs, None)
    }

    /// Replay the retained proof from an already-absorbed source-manifest
    /// transcript and keep only the suffix beginning at `start`.
    ///
    /// The prefix must end at a transcript event boundary and at an absorb
    /// boundary. Protocol v19 certifies the latter when it constructs the
    /// source-manifest checkpoint. The state equality below prevents a host
    /// preflight from generating a suffix against a different checkpoint.
    pub fn run_preflight<TS>(
        &self,
        mut transcript: TS,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proof: &Proof<BabyBearPoseidon2Config>,
        start: RebasedTranscriptPreflight,
    ) -> Preflight
    where
        TS: FiatShamirTranscript<BabyBearPoseidon2Config>
            + TranscriptHistory<F = F, State = [F; POSEIDON2_WIDTH]>,
    {
        assert_eq!(
            transcript.len(),
            start.start_tidx,
            "manifest transcript cursor does not match certified rebase"
        );
        let prefix_checkpoint = transcript.checkpoint();
        let mut preflight = Preflight::default();
        self.proof_shape.run_preflight_rebased_logup_only(
            child_vk,
            proof,
            &mut preflight,
            &mut transcript,
            start,
        );
        self.gkr
            .run_preflight(proof, &mut preflight, &mut transcript);
        self.batch_constraint
            .run_preflight(child_vk, proof, &mut preflight, &mut transcript);

        let full_log = transcript.into_log();
        let suffix = full_log
            .suffix(prefix_checkpoint)
            .expect("manifest checkpoint must be a transcript event boundary");
        assert_eq!(
            suffix.perm_results().first(),
            Some(&start.state),
            "manifest transcript state does not match certified rebase"
        );
        preflight.transcript = suffix;
        preflight
    }

    /// Same preflight entry point using the backend-neutral retained proof
    /// object, which has no stacking or WHIR fields.
    pub fn run_preflight_retained<TS>(
        &self,
        transcript: TS,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proof: RetainedLogUpOnlyProof,
        start: RebasedTranscriptPreflight,
    ) -> (Proof<BabyBearPoseidon2Config>, Preflight)
    where
        TS: FiatShamirTranscript<BabyBearPoseidon2Config>
            + TranscriptHistory<F = F, State = [F; POSEIDON2_WIDTH]>,
    {
        let proof = proof.into_partial_proof();
        let preflight = self.run_preflight(transcript, child_vk, &proof, start);
        (proof, preflight)
    }

    #[must_use]
    pub fn cached_trace_record(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> CachedTraceRecord {
        self.batch_constraint.cached_trace_record(child_vk)
    }

    /// Generate CPU traces for the retained partial verifier. The returned
    /// order exactly matches [`AggregationSubCircuit::airs`]. Caller-owned
    /// start/column/endpoint adapter AIRs are intentionally not included.
    pub fn generate_proving_ctxs<SC: StarkProtocolConfig<F = F>>(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        cached_trace_ctx: CachedTraceCtx<CpuBackend<SC>>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        preflights: &[Preflight],
    ) -> Option<Vec<AirProvingContext<CpuBackend<SC>>>> {
        self.generate_proving_ctxs_extended(
            child_vk,
            cached_trace_ctx,
            proofs,
            preflights,
            None,
            None,
            Vec::new(),
            Vec::new(),
        )
    }

    /// Generate the partial verifier while letting an enclosing protocol own
    /// the remainder of the same resumed transcript.
    ///
    /// Every supplied log must begin with the exact verifier suffix recorded
    /// in `preflights[i]`.  Additional operations are therefore append-only;
    /// callers cannot replace any LogUp challenge or proof observation.  The
    /// extra Poseidon inputs are folded into this verifier's sole table so
    /// source-manifest and boundary hash lookups remain balanced.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_proving_ctxs_extended<SC: StarkProtocolConfig<F = F>>(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        cached_trace_ctx: CachedTraceCtx<CpuBackend<SC>>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        preflights: &[Preflight],
        extended_logs: Option<&[TranscriptLog<F, [F; POSEIDON2_WIDTH]>]>,
        checkpoint_targets: Option<&[[usize; 2]]>,
        additional_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        additional_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    ) -> Option<Vec<AirProvingContext<CpuBackend<SC>>>> {
        let packet = self.generate_proving_ctxs_extended_for_shared_poseidon(
            child_vk,
            cached_trace_ctx,
            proofs,
            preflights,
            extended_logs,
            checkpoint_targets,
            additional_permutation_inputs,
            additional_compression_inputs,
        )?;
        let poseidon = self.transcript.build_poseidon2_trace(
            packet.poseidon2_permutation_inputs,
            packet.poseidon2_compression_inputs,
            None,
        )?;
        let mut contexts = packet.contexts;
        contexts.insert(
            self.poseidon_air_index::<SC>(),
            AirProvingContext::simple_no_pis(poseidon),
        );
        (contexts.len() == <Self as AggregationSubCircuit>::airs::<SC>(self).len())
            .then_some(contexts)
    }

    /// Packet-only counterpart of [`Self::generate_proving_ctxs_extended`].
    /// It performs the same strict transcript-prefix checks and emits all
    /// contexts except the physical Poseidon table.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_proving_ctxs_extended_for_shared_poseidon<SC: StarkProtocolConfig<F = F>>(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        cached_trace_ctx: CachedTraceCtx<CpuBackend<SC>>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        preflights: &[Preflight],
        extended_logs: Option<&[TranscriptLog<F, [F; POSEIDON2_WIDTH]>]>,
        checkpoint_targets: Option<&[[usize; 2]]>,
        additional_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        additional_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    ) -> Option<LogUpOnlySharedPoseidonContexts<SC>> {
        if proofs.len() > MAX_NUM_PROOFS || proofs.len() != preflights.len() {
            return None;
        }
        if let Some(logs) = extended_logs {
            if logs.len() != preflights.len()
                || logs.iter().zip(preflights).any(|(extended, preflight)| {
                    let prefix = &preflight.transcript;
                    extended.values().get(..prefix.len()) != Some(prefix.values())
                        || extended.samples().get(..prefix.len()) != Some(prefix.samples())
                })
            {
                return None;
            }
        }
        if checkpoint_targets.is_some_and(|targets| targets.len() != preflights.len()) {
            return None;
        }
        if checkpoint_targets.is_some() && extended_logs.is_none() {
            return None;
        }
        let power_checker =
            Arc::new(PowerCheckerCpuTraceGenerator::<2, POW_CHECKER_HEIGHT>::default());
        let exp_bits_len = ExpBitsLenCpuTraceGenerator::default();
        let cached_record = match &cached_trace_ctx {
            CachedTraceCtx::PcsData(_) | CachedTraceCtx::SetupBound => None,
            CachedTraceCtx::Records(record) => Some(record),
        };

        let mut batch = self.batch_constraint.generate_proving_ctxs(
            child_vk,
            proofs,
            preflights,
            &(cached_record, power_checker.clone()),
            None,
        )?;
        match cached_trace_ctx {
            CachedTraceCtx::PcsData(data) => {
                assert!(self.batch_constraint.has_cached);
                batch[LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX].cached_mains = vec![data];
            }
            CachedTraceCtx::SetupBound => {
                assert!(self.batch_constraint.has_cached);
            }
            CachedTraceCtx::Records(record) => {
                assert!(!self.batch_constraint.has_cached);
                let commit = record.dag_commit_info.expect("DAG commitment").commit;
                if let Some(expected) = self.batch_constraint.fixed_dag_commit() {
                    if commit != expected {
                        return None;
                    }
                } else {
                    batch[LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX].public_values = commit.to_vec();
                }
            }
        }

        let (mut transcript, poseidon2_permutation_inputs, poseidon2_compression_inputs) =
            generate_extended_rebased_transcript_ctxs_without_poseidon::<SC>(
                &self.transcript,
                preflights,
                extended_logs,
                checkpoint_targets,
                additional_permutation_inputs,
                additional_compression_inputs,
            )?;
        let external_range_checks: &[usize] = &[];
        let mut proof_shape = self.proof_shape.generate_proving_ctxs(
            child_vk,
            proofs,
            preflights,
            &(power_checker.clone(), external_range_checks),
            None,
        )?;
        let mut gkr =
            self.gkr
                .generate_proving_ctxs(child_vk, proofs, preflights, &exp_bits_len, None)?;

        let mut result =
            Vec::with_capacity(batch.len() + transcript.len() + proof_shape.len() + gkr.len() + 2);
        result.append(&mut batch);
        result.append(&mut transcript);
        result.append(&mut proof_shape);
        result.append(&mut gkr);
        result.push(AirProvingContext::simple_no_pis(
            power_checker.generate_trace_row_major(),
        ));
        result.push(AirProvingContext::simple_no_pis(
            exp_bits_len.generate_trace_row_major(None)?,
        ));
        if result.len() != self.airs_without_poseidon::<SC>().len() {
            return None;
        }
        Some(LogUpOnlySharedPoseidonContexts {
            contexts: result,
            poseidon2_permutation_inputs,
            poseidon2_compression_inputs,
        })
    }

    /// CUDA counterpart of
    /// [`Self::generate_proving_ctxs_extended_for_shared_poseidon`].
    ///
    /// The output is bounded by `MAX_NUM_PROOFS`, follows
    /// [`Self::airs_without_poseidon`] exactly, and explicitly tags the two
    /// CPU-generated resumed-transcript contexts and the CPU rebase adapter.
    /// Batch-constraint, the remaining proof-shape contexts, GKR, and
    /// primitive contexts remain device-resident.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn generate_proving_ctxs_extended_for_shared_poseidon_cuda(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        cached_trace_ctx: CachedTraceCtx<openvm_cuda_backend::GpuBackend>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        preflights: &[Preflight],
        extended_logs: Option<&[TranscriptLog<F, [F; POSEIDON2_WIDTH]>]>,
        checkpoint_targets: Option<&[[usize; 2]]>,
        additional_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        additional_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        device_ctx: &openvm_cuda_common::stream::GpuDeviceCtx,
    ) -> Option<crate::batch_constraint::cuda_tracegen::LogUpOnlySharedPoseidonCudaContexts> {
        use openvm_cuda_backend::GpuBackend;

        use crate::{
            batch_constraint::cuda_tracegen::{
                LogUpOnlyCudaProvingContext, LogUpOnlySharedPoseidonCudaContexts,
            },
            cuda::{preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu},
            primitives::{
                exp_bits_len::ExpBitsLenTraceGenerator as GpuExpBitsLenTraceGenerator,
                pow::cuda::PowerCheckerGpuTraceGenerator,
            },
            proof_shape::rebased::RebasedProofShapeStartTraceGenerator,
            tracegen::RowMajorChip,
        };

        if proofs.len() > MAX_NUM_PROOFS || proofs.len() != preflights.len() {
            return None;
        }
        if preflights.iter().any(|preflight| {
            preflight.batch_constraint.equation_mode != VerifierEquationMode::LogUpOnly
        }) {
            return None;
        }
        if let Some(logs) = extended_logs {
            if logs.len() != preflights.len()
                || logs.iter().zip(preflights).any(|(extended, preflight)| {
                    let prefix = &preflight.transcript;
                    extended.values().get(..prefix.len()) != Some(prefix.values())
                        || extended.samples().get(..prefix.len()) != Some(prefix.samples())
                })
            {
                return None;
            }
        }
        if checkpoint_targets.is_some_and(|targets| targets.len() != preflights.len())
            || (checkpoint_targets.is_some() && extended_logs.is_none())
        {
            return None;
        }

        let (transcript, poseidon2_permutation_inputs, poseidon2_compression_inputs) =
            generate_extended_rebased_transcript_ctxs_without_poseidon::<BabyBearPoseidon2Config>(
                &self.transcript,
                preflights,
                extended_logs,
                checkpoint_targets,
                additional_permutation_inputs,
                additional_compression_inputs,
            )?;

        let child_vk_gpu = VerifyingKeyGpu::new(child_vk, device_ctx);
        let proofs_gpu = proofs
            .iter()
            .map(|proof| ProofGpu::new(child_vk, proof, device_ctx))
            .collect::<Vec<_>>();
        let preflights_gpu = proofs
            .iter()
            .zip(preflights)
            .map(|(proof, preflight)| PreflightGpu::new(child_vk, proof, preflight, device_ctx))
            .collect::<Vec<_>>();
        let power_checker = Arc::new(
            PowerCheckerGpuTraceGenerator::<2, POW_CHECKER_HEIGHT>::hybrid(device_ctx.clone()),
        );
        let exp_bits_len = GpuExpBitsLenTraceGenerator::new(device_ctx.clone());
        let cached_record = match &cached_trace_ctx {
            CachedTraceCtx::PcsData(_) | CachedTraceCtx::SetupBound => None,
            CachedTraceCtx::Records(record) => Some(record),
        };

        let batch = self.batch_constraint.generate_proving_ctxs(
            &child_vk_gpu,
            proofs_gpu.as_slice(),
            preflights_gpu.as_slice(),
            &(cached_record, power_checker.cpu_checker()?, device_ctx),
            None,
        );
        let mut batch = batch?;
        match cached_trace_ctx {
            CachedTraceCtx::PcsData(data) => {
                if !self.batch_constraint.has_cached {
                    return None;
                }
                batch[LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX].cached_mains = vec![data];
            }
            CachedTraceCtx::SetupBound => {
                if !self.batch_constraint.has_cached {
                    return None;
                }
            }
            CachedTraceCtx::Records(record) => {
                if self.batch_constraint.has_cached {
                    return None;
                }
                let commit = record.dag_commit_info?.commit;
                if let Some(expected) = self.batch_constraint.fixed_dag_commit() {
                    if commit != expected {
                        return None;
                    }
                } else {
                    batch[LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX].public_values = commit.to_vec();
                }
            }
        }

        let external_range_checks: &[usize] = &[];
        let proof_shape = self.proof_shape.generate_proving_ctxs(
            &child_vk_gpu,
            proofs_gpu.as_slice(),
            preflights_gpu.as_slice(),
            &(power_checker.clone(), external_range_checks, device_ctx),
            None,
        );
        let mut proof_shape = proof_shape?;
        let rebased_proof_shape = AirProvingContext::simple_no_pis(
            RebasedProofShapeStartTraceGenerator.generate_trace(&preflights, None)?,
        );
        let gkr = self.gkr.generate_proving_ctxs(
            &child_vk_gpu,
            proofs_gpu.as_slice(),
            preflights_gpu.as_slice(),
            &(&exp_bits_len, device_ctx),
            None,
        );
        let mut gkr = gkr?;

        // Match the regular recursive CUDA prover: all module kernels that
        // feed primitive multiplicity tables complete before those tables are
        // materialized, and device-owned temporary blobs remain alive until
        // their kernels have consumed them.
        device_ctx.stream.synchronize().ok()?;

        let mut contexts =
            Vec::with_capacity(batch.len() + transcript.len() + proof_shape.len() + gkr.len() + 3);
        contexts.extend(batch.drain(..).map(LogUpOnlyCudaProvingContext::Device));
        contexts.extend(
            transcript
                .into_iter()
                .map(LogUpOnlyCudaProvingContext::CpuTranscript),
        );
        contexts.extend(
            proof_shape
                .drain(..)
                .map(LogUpOnlyCudaProvingContext::Device),
        );
        contexts.push(LogUpOnlyCudaProvingContext::CpuRebasedProofShape(
            rebased_proof_shape,
        ));
        contexts.extend(gkr.drain(..).map(LogUpOnlyCudaProvingContext::Device));
        contexts.push(LogUpOnlyCudaProvingContext::Device(AirProvingContext::<
            GpuBackend,
        >::simple_no_pis(
            power_checker.generate_trace(),
        )));
        let exp_bits_len = exp_bits_len.generate_trace_device(None);
        contexts.push(LogUpOnlyCudaProvingContext::Device(AirProvingContext::<
            GpuBackend,
        >::simple_no_pis(
            exp_bits_len?
        )));
        let expected_contexts = self
            .airs_without_poseidon::<BabyBearPoseidon2Config>()
            .len();
        if contexts.len() != expected_contexts {
            return None;
        }
        Some(LogUpOnlySharedPoseidonCudaContexts {
            contexts,
            poseidon2_permutation_inputs,
            poseidon2_compression_inputs,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn generate_extended_rebased_transcript_ctxs_without_poseidon<SC: StarkProtocolConfig<F = F>>(
    module: &TranscriptModule,
    preflights: &[Preflight],
    extended_logs: Option<&[TranscriptLog<F, [F; POSEIDON2_WIDTH]>]>,
    checkpoint_targets: Option<&[[usize; 2]]>,
    mut additional_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    additional_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
) -> Option<(
    Vec<AirProvingContext<CpuBackend<SC>>>,
    Vec<[F; POSEIDON2_WIDTH]>,
    Vec<[F; POSEIDON2_WIDTH]>,
)> {
    if let Some(logs) = extended_logs {
        let logs = logs.iter().collect::<Vec<_>>();
        let resumes = preflights
            .iter()
            .map(|preflight| {
                preflight
                    .rebased_transcript
                    .map(|start| (start.start_tidx, start.state))
            })
            .collect::<Vec<_>>();
        if resumes.iter().any(Option::is_none) {
            return None;
        }
        let checkpoints = checkpoint_targets
            .map(|targets| targets.iter().copied().map(Some).collect::<Vec<_>>())
            .unwrap_or_default();
        let artifacts = module.build_transcript_trace_artifacts(
            &logs,
            &resumes,
            &checkpoints,
            Vec::new(),
            Vec::new(),
            None,
        )?;
        additional_permutation_inputs.extend(artifacts.poseidon2_perm_inputs);
        let compression_inputs = artifacts
            .poseidon2_compress_inputs
            .into_iter()
            .chain(additional_compression_inputs)
            .collect();
        let transcript_airs = module.airs::<SC>();
        let merkle_width = p3_air::BaseAir::<F>::width(transcript_airs[2].as_ref());
        let empty_merkle =
            p3_matrix::dense::RowMajorMatrix::new(vec![F::ZERO; merkle_width], merkle_width);
        Some((
            vec![
                AirProvingContext::simple_no_pis(artifacts.transcript_trace),
                AirProvingContext::simple_no_pis(empty_merkle),
            ],
            additional_permutation_inputs,
            compression_inputs,
        ))
    } else {
        if !additional_permutation_inputs.is_empty() || !additional_compression_inputs.is_empty() {
            return None;
        }
        generate_rebased_transcript_ctxs_without_poseidon::<SC>(module, preflights)
    }
}

/// Prefix transcript owner used by protocol-v19 before the recursive
/// LogUp-only suffix begins.  It deliberately exposes only its Transcript AIR;
/// the enclosing partial verifier owns the single shared Poseidon table.
pub struct LogUpOnlyPrefixTranscript {
    module: TranscriptModule,
}

impl LogUpOnlyPrefixTranscript {
    #[must_use]
    pub fn new(
        buses: BusInventory,
        params: SystemParams,
        checkpoint_bus: CertifiedTranscriptCheckpointBus,
    ) -> Self {
        Self {
            module: TranscriptModule::new_with_checkpoint_bus(
                buses,
                params,
                false,
                false,
                Some(checkpoint_bus),
            ),
        }
    }

    #[must_use]
    pub fn air<SC: StarkProtocolConfig<F = F>>(&self) -> AirRef<SC> {
        self.module.airs::<SC>().remove(0)
    }

    pub fn generate_trace<SC: StarkProtocolConfig<F = F>>(
        &self,
        logs: &[TranscriptLog<F, [F; POSEIDON2_WIDTH]>],
        end_indices: &[usize],
    ) -> Option<(
        AirProvingContext<CpuBackend<SC>>,
        Vec<[F; POSEIDON2_WIDTH]>,
        Vec<[F; POSEIDON2_WIDTH]>,
    )> {
        if logs.len() != end_indices.len() {
            return None;
        }
        let logs = logs.iter().collect::<Vec<_>>();
        let checkpoints = end_indices
            .iter()
            .map(|&end| Some([end, end]))
            .collect::<Vec<_>>();
        let artifacts = self.module.build_transcript_trace_artifacts(
            &logs,
            &[],
            &checkpoints,
            Vec::new(),
            Vec::new(),
            None,
        )?;
        Some((
            AirProvingContext::simple_no_pis(artifacts.transcript_trace),
            artifacts.poseidon2_perm_inputs,
            artifacts.poseidon2_compress_inputs,
        ))
    }
}

fn generate_rebased_transcript_ctxs_without_poseidon<SC: StarkProtocolConfig<F = F>>(
    module: &TranscriptModule,
    preflights: &[Preflight],
) -> Option<(
    Vec<AirProvingContext<CpuBackend<SC>>>,
    Vec<[F; POSEIDON2_WIDTH]>,
    Vec<[F; POSEIDON2_WIDTH]>,
)> {
    let logs = preflights
        .iter()
        .map(|preflight| &preflight.transcript)
        .collect::<Vec<_>>();
    let resumes = preflights
        .iter()
        .map(|preflight| {
            preflight
                .rebased_transcript
                .map(|start| (start.start_tidx, start.state))
        })
        .collect::<Vec<_>>();
    if resumes.iter().any(Option::is_none) {
        return None;
    }
    let artifacts = module.build_transcript_trace_artifacts(
        &logs,
        &resumes,
        &[],
        Vec::new(),
        Vec::new(),
        None,
    )?;
    let transcript_airs = module.airs::<SC>();
    let merkle_width = p3_air::BaseAir::<F>::width(transcript_airs[2].as_ref());
    let empty_merkle =
        p3_matrix::dense::RowMajorMatrix::new(vec![F::ZERO; merkle_width], merkle_width);
    Some((
        vec![
            AirProvingContext::simple_no_pis(artifacts.transcript_trace),
            AirProvingContext::simple_no_pis(empty_merkle),
        ],
        artifacts.poseidon2_perm_inputs,
        artifacts.poseidon2_compress_inputs,
    ))
}

impl<const MAX_NUM_PROOFS: usize> AggregationSubCircuit
    for LogUpOnlyPartialVerifier<MAX_NUM_PROOFS>
{
    fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        let power_checker = PowerCheckerAir::<2, POW_CHECKER_HEIGHT> {
            pow_bus: self.bus_inventory.power_checker_bus,
            range_bus: self.bus_inventory.range_checker_bus,
        };
        let exp_bits_len = ExpBitsLenAir::new(
            self.bus_inventory.exp_bits_len_bus,
            self.bus_inventory.right_shift_bus,
        );
        // SymbolicExpressionAir must remain first.
        iter::empty()
            .chain(self.batch_constraint.airs())
            .chain(self.transcript.airs())
            .chain(self.proof_shape.airs())
            .chain(self.gkr.airs())
            .chain([
                Arc::new(power_checker) as AirRef<_>,
                Arc::new(exp_bits_len) as AirRef<_>,
            ])
            .collect()
    }

    fn bus_inventory(&self) -> &BusInventory {
        &self.bus_inventory
    }

    fn next_bus_idx(&self) -> BusIndex {
        self.bus_idx_manager.next_bus_idx()
    }

    fn max_num_proofs(&self) -> usize {
        MAX_NUM_PROOFS
    }
}

#[cfg(test)]
mod tests {
    use core::borrow::{Borrow, BorrowMut};

    use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
    use openvm_cpu_backend::{logup_zerocheck::prove_logup_only_recorded, CpuBackend};
    #[cfg(feature = "cuda")]
    use openvm_cuda_backend::data_transporter::assert_eq_host_and_device_matrix;
    #[cfg(feature = "cuda")]
    use openvm_cuda_common::stream::GpuDeviceCtx;
    use openvm_recursion_circuit_derive::AlignedBorrow;
    use openvm_stark_backend::{
        interaction::InteractionBuilder,
        p3_field::TwoAdicField,
        proof::{column_openings_by_rot, TraceVData},
        prover::{DeviceDataTransporter, MatrixDimensions, ProvingContext},
        test_utils::{
            default_test_params_small, FibFixture, InteractionsFixture11, MixtureFixture,
            MixtureFixtureEnum, TestFixture,
        },
        verifier::batch_constraints::{verify_logup_only_prefix, BatchConstraintMode},
        BaseAirWithPublicValues, FiatShamirTranscript, PartitionedBaseAir, StarkEngine,
        TranscriptHistory,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::{
        default_duplex_sponge_recorder, BabyBearPoseidon2Config as SC, BabyBearPoseidon2CpuEngine,
        DuplexSponge, DuplexSpongeRecorder, DIGEST_SIZE, D_EF, EF,
    };
    use p3_air::{Air, AirBuilder, BaseAir};
    use p3_field::{BasedVectorSpace, Field};
    use p3_matrix::{dense::RowMajorMatrix, Matrix};

    use super::*;
    use crate::{
        batch_constraint::{
            bus::{
                BatchConstraintEndpointBus, BatchConstraintEndpointMessage,
                LogUpOnlyOpeningPointBus, LogUpOnlyOpeningPointMessage,
            },
            expression_claim::ExpressionClaimCols,
            partial::PartialBatchConstraintEndpointCols,
            LOCAL_EXPRESSION_CLAIM_AIR_IDX,
        },
        bus::{ColumnClaimsBus, ColumnClaimsMessage},
        proof_shape::bus::{RebasedTranscriptStartBus, RebasedTranscriptStartMessage},
    };

    #[repr(C)]
    #[derive(AlignedBorrow, Copy, Clone, Debug, StructReflection)]
    struct RebaseSourceCols<T> {
        is_valid: T,
        proof_idx: T,
        tidx: T,
        state: [T; POSEIDON2_WIDTH],
    }

    #[derive(ColumnsAir)]
    #[columns_via(RebaseSourceCols<u8>)]
    struct RebaseSourceAir(RebasedTranscriptStartBus);

    impl<Fld: Field> BaseAir<Fld> for RebaseSourceAir {
        fn width(&self) -> usize {
            RebaseSourceCols::<Fld>::width()
        }
    }
    impl<Fld: Field> BaseAirWithPublicValues<Fld> for RebaseSourceAir {}
    impl<Fld: Field> PartitionedBaseAir<Fld> for RebaseSourceAir {}
    impl<AB: AirBuilder + InteractionBuilder> Air<AB> for RebaseSourceAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).unwrap();
            let local: &RebaseSourceCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.is_valid);
            self.0.send(
                builder,
                local.proof_idx,
                RebasedTranscriptStartMessage {
                    tidx: local.tidx.into(),
                    state: local.state.map(Into::into),
                },
                local.is_valid,
            );
        }
    }

    #[repr(C)]
    #[derive(AlignedBorrow, Copy, Clone, Debug, StructReflection)]
    struct ColumnSourceCols<T> {
        is_valid: T,
        proof_idx: T,
        sort_idx: T,
        part_idx: T,
        col_idx: T,
        claim: [T; D_EF],
        is_rot: T,
    }

    #[derive(ColumnsAir)]
    #[columns_via(ColumnSourceCols<u8>)]
    struct ColumnSourceAir(ColumnClaimsBus);

    impl<Fld: Field> BaseAir<Fld> for ColumnSourceAir {
        fn width(&self) -> usize {
            ColumnSourceCols::<Fld>::width()
        }
    }
    impl<Fld: Field> BaseAirWithPublicValues<Fld> for ColumnSourceAir {}
    impl<Fld: Field> PartitionedBaseAir<Fld> for ColumnSourceAir {}
    impl<AB: AirBuilder + InteractionBuilder> Air<AB> for ColumnSourceAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).unwrap();
            let local: &ColumnSourceCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.is_valid);
            builder.when(local.is_valid).assert_bool(local.is_rot);
            self.0.send(
                builder,
                local.proof_idx,
                ColumnClaimsMessage {
                    sort_idx: local.sort_idx.into(),
                    part_idx: local.part_idx.into(),
                    col_idx: local.col_idx.into(),
                    claim: local.claim.map(Into::into),
                    is_rot: local.is_rot.into(),
                },
                local.is_valid,
            );
        }
    }

    #[repr(C)]
    #[derive(AlignedBorrow, Copy, Clone, Debug, StructReflection)]
    struct EndpointSinkCols<T> {
        is_valid: T,
        proof_idx: T,
        tidx: T,
        final_claim: [T; D_EF],
    }

    #[repr(C)]
    #[derive(AlignedBorrow, Copy, Clone, Debug, StructReflection)]
    struct OpeningPointSinkCols<T> {
        is_valid: T,
        proof_idx: T,
        index: T,
        value: [T; D_EF],
    }

    #[derive(ColumnsAir)]
    #[columns_via(OpeningPointSinkCols<u8>)]
    struct OpeningPointSinkAir(LogUpOnlyOpeningPointBus);

    impl<Fld: Field> BaseAir<Fld> for OpeningPointSinkAir {
        fn width(&self) -> usize {
            OpeningPointSinkCols::<Fld>::width()
        }
    }
    impl<Fld: Field> BaseAirWithPublicValues<Fld> for OpeningPointSinkAir {}
    impl<Fld: Field> PartitionedBaseAir<Fld> for OpeningPointSinkAir {}
    impl<AB: AirBuilder + InteractionBuilder> Air<AB> for OpeningPointSinkAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).unwrap();
            let local: &OpeningPointSinkCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.is_valid);
            self.0.receive(
                builder,
                local.proof_idx,
                LogUpOnlyOpeningPointMessage {
                    index: local.index.into(),
                    value: local.value.map(Into::into),
                },
                local.is_valid,
            );
        }
    }

    #[derive(ColumnsAir)]
    #[columns_via(EndpointSinkCols<u8>)]
    struct EndpointSinkAir(BatchConstraintEndpointBus);

    impl<Fld: Field> BaseAir<Fld> for EndpointSinkAir {
        fn width(&self) -> usize {
            EndpointSinkCols::<Fld>::width()
        }
    }
    impl<Fld: Field> BaseAirWithPublicValues<Fld> for EndpointSinkAir {}
    impl<Fld: Field> PartitionedBaseAir<Fld> for EndpointSinkAir {}
    impl<AB: AirBuilder + InteractionBuilder> Air<AB> for EndpointSinkAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).unwrap();
            let local: &EndpointSinkCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.is_valid);
            self.0.receive(
                builder,
                local.proof_idx,
                BatchConstraintEndpointMessage {
                    tidx: local.tidx.into(),
                    final_claim: local.final_claim.map(Into::into),
                },
                local.is_valid,
            );
        }
    }

    struct FixtureData {
        vk: MultiStarkVerifyingKey<SC>,
        proof: Proof<SC>,
        air_ids: Vec<usize>,
        n_per_trace: Vec<isize>,
        prefix: DuplexSpongeRecorder,
        start: RebasedTranscriptPreflight,
        backend_endpoint:
            openvm_stark_backend::verifier::batch_constraints::BatchConstraintEndpoint<SC>,
        backend_suffix: openvm_stark_backend::TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    }

    fn fixture_data() -> FixtureData {
        let mut params = default_test_params_small();
        params.logup.pow_bits = 0;
        let fixture = MixtureFixture::<SC>::new(vec![
            MixtureFixtureEnum::FibFixture(FibFixture::new(1, 1, 8)),
            MixtureFixtureEnum::InteractionsFixture11(InteractionsFixture11),
        ]);
        let engine = BabyBearPoseidon2CpuEngine::<DuplexSpongeRecorder>::new(params);
        let (pk, vk) = fixture.keygen(&engine);
        let host_ctx = fixture.generate_proving_ctx();
        let mut trace_vdata = vec![None; vk.inner.per_air.len()];
        let mut public_values = vec![Vec::new(); vk.inner.per_air.len()];
        for (air_id, trace) in &host_ctx.per_trace {
            trace_vdata[*air_id] = Some(TraceVData {
                log_height: MatrixDimensions::height(&trace.common_main).ilog2() as usize,
                cached_commitments: trace
                    .cached_mains
                    .iter()
                    .map(|main| main.commitment)
                    .collect(),
            });
            public_values[*air_id] = trace.public_values.clone();
        }
        let device = engine.device();
        let pk =
            <_ as DeviceDataTransporter<SC, CpuBackend<SC>>>::transport_pk_to_device(device, &pk);
        let ctx =
            <_ as DeviceDataTransporter<SC, CpuBackend<SC>>>::transport_proving_ctx_to_device(
                device, &host_ctx,
            )
            .into_sorted();
        let air_ids = ctx
            .per_trace
            .iter()
            .map(|(air_id, _)| *air_id)
            .collect::<Vec<_>>();
        let n_per_trace = ctx
            .common_main_traces()
            .map(|(_, trace)| {
                MatrixDimensions::height(trace).ilog2() as isize - vk.inner.params.l_skip as isize
            })
            .collect::<Vec<_>>();

        // One complete rate block gives protocol v19 its required absorb
        // boundary while making the rebase nontrivial.
        let mut prefix = default_duplex_sponge_recorder();
        FiatShamirTranscript::<SC>::observe_commit(
            &mut prefix,
            core::array::from_fn(|i| F::from_usize(i + 17)),
        );
        let sponge_checkpoint = prefix.inner.checkpoint();
        assert_eq!(sponge_checkpoint.absorb_idx, 0);
        let start = RebasedTranscriptPreflight {
            start_tidx: prefix.len(),
            state: sponge_checkpoint.state,
        };

        let mut prover_transcript = prefix.clone();
        let (gkr, batch, _) = prove_logup_only_recorded(&mut prover_transcript, &pk, &ctx).unwrap();
        let proof = RetainedLogUpOnlyProof {
            common_main_commit: [F::ZERO; DIGEST_SIZE],
            trace_vdata,
            public_values,
            gkr_proof: gkr,
            batch_constraint_proof: batch,
        }
        .into_partial_proof();

        let omega = F::two_adic_generator(vk.inner.params.l_skip);
        let omega_skip_pows = omega.powers().take(1 << vk.inner.params.l_skip).collect();
        let mut verifier_transcript = prefix.clone();
        let prefix_checkpoint = verifier_transcript.checkpoint();
        let backend_endpoint = verify_logup_only_prefix(
            &mut verifier_transcript,
            &vk.inner,
            &proof.gkr_proof,
            (&proof.batch_constraint_proof).into(),
            &air_ids,
            &n_per_trace,
            &omega_skip_pows,
            None,
        )
        .unwrap();
        let backend_suffix = verifier_transcript
            .into_log()
            .suffix(prefix_checkpoint)
            .unwrap();
        FixtureData {
            vk,
            proof,
            air_ids,
            n_per_trace,
            prefix,
            start,
            backend_endpoint,
            backend_suffix,
        }
    }

    fn rebase_source_trace(start: RebasedTranscriptPreflight) -> RowMajorMatrix<F> {
        let width = RebaseSourceCols::<F>::width();
        let mut values = vec![F::ZERO; width];
        let cols: &mut RebaseSourceCols<F> = values.as_mut_slice().borrow_mut();
        cols.is_valid = F::ONE;
        cols.tidx = F::from_usize(start.start_tidx);
        cols.state = start.state;
        RowMajorMatrix::new(values, width)
    }

    fn column_source_trace(data: &FixtureData, preflight: &Preflight) -> RowMajorMatrix<F> {
        let mut records = Vec::<(usize, usize, usize, EF, bool)>::new();
        for (sort_idx, parts) in data
            .proof
            .batch_constraint_proof
            .column_openings
            .iter()
            .enumerate()
        {
            let air_id = preflight.proof_shape.sorted_trace_vdata[sort_idx].0;
            let need_rot = data.vk.inner.per_air[air_id].params.need_rot;
            for (part_idx, openings) in parts.iter().enumerate() {
                for (col_idx, (claim, rotated)) in
                    column_openings_by_rot(openings, need_rot).enumerate()
                {
                    records.push((sort_idx, part_idx, col_idx, claim, false));
                    if need_rot {
                        records.push((sort_idx, part_idx, col_idx, rotated, true));
                    }
                }
            }
        }
        let width = ColumnSourceCols::<F>::width();
        let height = records.len().max(1).next_power_of_two();
        let mut values = vec![F::ZERO; height * width];
        for (row, (sort_idx, part_idx, col_idx, claim, is_rot)) in
            values.chunks_exact_mut(width).zip(records)
        {
            let cols: &mut ColumnSourceCols<F> = row.borrow_mut();
            cols.is_valid = F::ONE;
            cols.sort_idx = F::from_usize(sort_idx);
            cols.part_idx = F::from_usize(part_idx);
            cols.col_idx = F::from_usize(col_idx);
            cols.claim
                .copy_from_slice(claim.as_basis_coefficients_slice());
            cols.is_rot = F::from_bool(is_rot);
        }
        RowMajorMatrix::new(values, width)
    }

    fn endpoint_sink_trace(preflight: &Preflight, delta: EF) -> RowMajorMatrix<F> {
        let width = EndpointSinkCols::<F>::width();
        let mut values = vec![F::ZERO; width];
        let cols: &mut EndpointSinkCols<F> = values.as_mut_slice().borrow_mut();
        cols.is_valid = F::ONE;
        cols.tidx = F::from_usize(preflight.batch_constraint.post_tidx);
        cols.final_claim.copy_from_slice(
            (preflight.batch_constraint.final_claim + delta).as_basis_coefficients_slice(),
        );
        RowMajorMatrix::new(values, width)
    }

    fn opening_point_sink_trace(preflight: &Preflight) -> RowMajorMatrix<F> {
        let point = &preflight.batch_constraint.sumcheck_rnd;
        let width = OpeningPointSinkCols::<F>::width();
        let height = point.len().max(1).next_power_of_two();
        let mut values = vec![F::ZERO; height * width];
        for (index, (row, value)) in values.chunks_exact_mut(width).zip(point.iter()).enumerate() {
            let cols: &mut OpeningPointSinkCols<F> = row.borrow_mut();
            cols.is_valid = F::ONE;
            cols.index = F::from_usize(index);
            cols.value
                .copy_from_slice(value.as_basis_coefficients_slice());
        }
        RowMajorMatrix::new(values, width)
    }

    fn partial_airs_and_ctxs(
        data: &FixtureData,
        preflight: &Preflight,
        endpoint_delta: EF,
    ) -> (Vec<AirRef<SC>>, Vec<AirProvingContext<CpuBackend<SC>>>) {
        let verifier = LogUpOnlyPartialVerifier::<1>::new(&data.vk, false);
        let exports = verifier.exports();
        let cached = verifier.cached_trace_record(&data.vk);
        let mut airs = verifier.airs::<SC>();
        let mut ctxs = verifier
            .generate_proving_ctxs::<SC>(
                &data.vk,
                CachedTraceCtx::Records(cached),
                core::slice::from_ref(&data.proof),
                core::slice::from_ref(preflight),
            )
            .expect("partial verifier traces");
        airs.extend([
            Arc::new(RebaseSourceAir(exports.rebased_start_bus)) as AirRef<_>,
            Arc::new(ColumnSourceAir(exports.column_claims_bus)) as AirRef<_>,
            Arc::new(OpeningPointSinkAir(exports.opening_point_bus)) as AirRef<_>,
            Arc::new(EndpointSinkAir(exports.endpoint_bus)) as AirRef<_>,
        ]);
        ctxs.extend([
            AirProvingContext::simple_no_pis(rebase_source_trace(data.start)),
            AirProvingContext::simple_no_pis(column_source_trace(data, preflight)),
            AirProvingContext::simple_no_pis(opening_point_sink_trace(preflight)),
            AirProvingContext::simple_no_pis(endpoint_sink_trace(preflight, endpoint_delta)),
        ]);
        (airs, ctxs)
    }

    fn debug_partial(airs: &[AirRef<SC>], ctxs: Vec<AirProvingContext<CpuBackend<SC>>>) {
        let mut params = default_test_params_small();
        params.max_constraint_degree = 4;
        let engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
        engine.debug(
            airs,
            &ProvingContext::new(ctxs.into_iter().enumerate().collect()),
        );
    }

    #[test]
    fn logup_only_rebased_preflight_matches_backend_prefix() {
        let data = fixture_data();
        let verifier = LogUpOnlyPartialVerifier::<1>::new(&data.vk, false);
        let preflight =
            verifier.run_preflight(data.prefix.clone(), &data.vk, &data.proof, data.start);

        assert_eq!(data.backend_endpoint.mode, BatchConstraintMode::LogUpOnly);
        assert_eq!(
            preflight.batch_constraint.equation_mode,
            VerifierEquationMode::LogUpOnly
        );
        assert_eq!(
            preflight.batch_constraint.final_claim,
            data.backend_endpoint.final_claim
        );
        assert_eq!(preflight.batch_constraint.xi, data.backend_endpoint.xi);
        assert_eq!(
            preflight.batch_constraint.sumcheck_rnd,
            data.backend_endpoint.rs
        );
        assert_eq!(
            preflight.batch_constraint.eq_ns_frontloaded,
            data.backend_endpoint.eq_ns
        );
        assert_eq!(
            preflight.batch_constraint.eq_sharp_ns_frontloaded,
            data.backend_endpoint.eq_sharp_ns
        );
        assert_eq!(preflight.transcript.values(), data.backend_suffix.values());
        assert_eq!(
            preflight.transcript.samples(),
            data.backend_suffix.samples()
        );
        assert_eq!(
            preflight.batch_constraint.post_tidx,
            data.start.start_tidx + data.backend_suffix.len()
        );
        assert_eq!(data.air_ids.len(), data.n_per_trace.len());
    }

    #[test]
    fn logup_only_rebase_rejects_wrong_cursor_and_state() {
        let data = fixture_data();
        let verifier = LogUpOnlyPartialVerifier::<1>::new(&data.vk, false);
        let wrong_cursor = RebasedTranscriptPreflight {
            start_tidx: data.start.start_tidx + 1,
            ..data.start
        };
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            verifier.run_preflight(data.prefix.clone(), &data.vk, &data.proof, wrong_cursor)
        }))
        .is_err());

        let mut wrong_state = data.start;
        wrong_state.state[0] += F::ONE;
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            verifier.run_preflight(data.prefix, &data.vk, &data.proof, wrong_state)
        }))
        .is_err());
    }

    #[test]
    fn logup_only_partial_air_assembly_accepts_backend_proof() {
        let data = fixture_data();
        let verifier = LogUpOnlyPartialVerifier::<1>::new(&data.vk, false);
        let preflight =
            verifier.run_preflight(data.prefix.clone(), &data.vk, &data.proof, data.start);
        let (airs, ctxs) = partial_airs_and_ctxs(&data, &preflight, EF::ZERO);
        debug_partial(&airs, ctxs);
    }

    #[test]
    fn logup_only_partial_air_rejects_nonzero_endpoint_delta() {
        let data = fixture_data();
        let verifier = LogUpOnlyPartialVerifier::<1>::new(&data.vk, false);
        let preflight =
            verifier.run_preflight(data.prefix.clone(), &data.vk, &data.proof, data.start);
        let (airs, ctxs) = partial_airs_and_ctxs(&data, &preflight, EF::ONE);
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            debug_partial(&airs, ctxs)
        }))
        .is_err());
    }

    #[test]
    fn logup_only_partial_air_rejects_standard_mode_separator() {
        let data = fixture_data();
        let verifier = LogUpOnlyPartialVerifier::<1>::new(&data.vk, false);
        let mut preflight =
            verifier.run_preflight(data.prefix.clone(), &data.vk, &data.proof, data.start);
        // Replace LogUpOnly(1) with AirAndLogUp(0) at the authenticated BCMO
        // mode slot while leaving the proof and all sampled challenges intact.
        preflight.transcript.values_mut()[2] = F::ZERO;
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let (airs, ctxs) = partial_airs_and_ctxs(&data, &preflight, EF::ZERO);
            debug_partial(&airs, ctxs)
        }))
        .is_err());
    }

    #[test]
    fn logup_only_partial_air_rejects_injected_air_constraint_group() {
        let data = fixture_data();
        let verifier = LogUpOnlyPartialVerifier::<1>::new(&data.vk, false);
        let preflight =
            verifier.run_preflight(data.prefix.clone(), &data.vk, &data.proof, data.start);
        let (airs, mut ctxs) = partial_airs_and_ctxs(&data, &preflight, EF::ZERO);
        let expression_claim = &mut ctxs[LOCAL_EXPRESSION_CLAIM_AIR_IDX].common_main;
        let width = ExpressionClaimCols::<F>::width();
        let first: &mut ExpressionClaimCols<F> = expression_claim.values[..width].borrow_mut();
        first.group_idx = F::ONE;
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            debug_partial(&airs, ctxs)
        }))
        .is_err());
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn logup_only_cuda_extended_packet_matches_fresh_cpu_oracle() {
        use crate::batch_constraint::cuda_tracegen::LogUpOnlyCudaProvingContext;

        let data = fixture_data();
        let verifier = LogUpOnlyPartialVerifier::<1>::new(&data.vk, false);
        let preflight =
            verifier.run_preflight(data.prefix.clone(), &data.vk, &data.proof, data.start);
        let extended_logs = vec![preflight.transcript.clone()];
        let additional_permutations =
            vec![core::array::from_fn(|index| F::from_usize(0x100 + index))];
        let additional_compressions =
            vec![core::array::from_fn(|index| F::from_usize(0x200 + index))];

        let cpu = verifier
            .generate_proving_ctxs_extended_for_shared_poseidon::<SC>(
                &data.vk,
                CachedTraceCtx::Records(verifier.cached_trace_record(&data.vk)),
                core::slice::from_ref(&data.proof),
                core::slice::from_ref(&preflight),
                Some(&extended_logs),
                None,
                additional_permutations.clone(),
                additional_compressions.clone(),
            )
            .expect("CPU LogUpOnly packet");
        let device_ctx = GpuDeviceCtx::for_current_device().expect("CUDA device");
        let cuda = verifier
            .generate_proving_ctxs_extended_for_shared_poseidon_cuda(
                &data.vk,
                CachedTraceCtx::Records(verifier.cached_trace_record(&data.vk)),
                core::slice::from_ref(&data.proof),
                core::slice::from_ref(&preflight),
                Some(&extended_logs),
                None,
                additional_permutations,
                additional_compressions,
                &device_ctx,
            )
            .expect("CUDA LogUpOnly packet");

        assert_eq!(
            cpu.poseidon2_permutation_inputs,
            cuda.poseidon2_permutation_inputs
        );
        assert_eq!(
            cpu.poseidon2_compression_inputs,
            cuda.poseidon2_compression_inputs
        );
        assert_eq!(cpu.contexts.len(), cuda.contexts.len());
        assert_eq!(
            cuda.contexts.len(),
            verifier.airs_without_poseidon::<SC>().len()
        );

        let transcript_start = verifier.batch_constraint.airs::<SC>().len();
        let mut cpu_transcript_indices = Vec::new();
        let mut cpu_rebased_proof_shape_indices = Vec::new();
        for (index, (cpu_ctx, cuda_ctx)) in cpu.contexts.iter().zip(&cuda.contexts).enumerate() {
            match cuda_ctx {
                LogUpOnlyCudaProvingContext::Device(cuda_ctx) => {
                    assert_eq!(cpu_ctx.public_values, cuda_ctx.public_values, "PI {index}");
                    assert_eq!(
                        cpu_ctx.cached_mains.len(),
                        cuda_ctx.cached_mains.len(),
                        "cached-main count {index}"
                    );
                    assert_eq_host_and_device_matrix(
                        Arc::new(cpu_ctx.common_main.clone()),
                        &cuda_ctx.common_main,
                        &device_ctx,
                    );
                }
                LogUpOnlyCudaProvingContext::CpuTranscript(cuda_ctx) => {
                    cpu_transcript_indices.push(index);
                    assert_eq!(cpu_ctx.public_values, cuda_ctx.public_values, "PI {index}");
                    assert_eq!(
                        cpu_ctx.cached_mains.len(),
                        cuda_ctx.cached_mains.len(),
                        "cached-main count {index}"
                    );
                    assert_eq!(
                        cpu_ctx.common_main.width, cuda_ctx.common_main.width,
                        "width {index}"
                    );
                    assert_eq!(
                        cpu_ctx.common_main.values, cuda_ctx.common_main.values,
                        "cells {index}"
                    );
                }
                LogUpOnlyCudaProvingContext::CpuRebasedProofShape(cuda_ctx) => {
                    cpu_rebased_proof_shape_indices.push(index);
                    assert_eq!(cpu_ctx.public_values, cuda_ctx.public_values, "PI {index}");
                    assert_eq!(
                        cpu_ctx.cached_mains.len(),
                        cuda_ctx.cached_mains.len(),
                        "cached-main count {index}"
                    );
                    assert_eq!(
                        cpu_ctx.common_main.width, cuda_ctx.common_main.width,
                        "width {index}"
                    );
                    assert_eq!(
                        cpu_ctx.common_main.values, cuda_ctx.common_main.values,
                        "cells {index}"
                    );
                }
            }
        }
        assert_eq!(
            cpu_transcript_indices,
            [transcript_start, transcript_start + 1]
        );
        assert_eq!(cpu_rebased_proof_shape_indices.len(), 1);

        // AIR 11 is mode-dependent. In LogUpOnly it must be the endpoint,
        // and its CPU oracle is already equal cell-for-cell to the CUDA trace.
        let endpoint = &cpu.contexts[11].common_main;
        let endpoint_width = PartialBatchConstraintEndpointCols::<F>::width();
        let endpoint_cols: &PartialBatchConstraintEndpointCols<F> =
            endpoint.values[..endpoint_width].borrow();
        assert_eq!(
            endpoint_cols.tidx,
            F::from_usize(preflight.batch_constraint.tidx_before_column_openings)
        );
        assert_eq!(
            endpoint_cols.final_claim,
            preflight
                .batch_constraint
                .final_claim
                .as_basis_coefficients_slice()
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn logup_only_cuda_rejects_equation_mode_corruption_and_over_capacity() {
        let data = fixture_data();
        let verifier = LogUpOnlyPartialVerifier::<1>::new(&data.vk, false);
        let preflight =
            verifier.run_preflight(data.prefix.clone(), &data.vk, &data.proof, data.start);
        let device_ctx = GpuDeviceCtx::for_current_device().expect("CUDA device");

        let mut corrupted = preflight.clone();
        corrupted.batch_constraint.equation_mode = VerifierEquationMode::AirAndLogUp;
        assert!(verifier
            .generate_proving_ctxs_extended_for_shared_poseidon_cuda(
                &data.vk,
                CachedTraceCtx::Records(verifier.cached_trace_record(&data.vk)),
                core::slice::from_ref(&data.proof),
                core::slice::from_ref(&corrupted),
                None,
                None,
                Vec::new(),
                Vec::new(),
                &device_ctx,
            )
            .is_none());

        assert!(verifier
            .generate_proving_ctxs_extended_for_shared_poseidon_cuda(
                &data.vk,
                CachedTraceCtx::Records(verifier.cached_trace_record(&data.vk)),
                &[data.proof.clone(), data.proof],
                &[preflight.clone(), preflight],
                None,
                None,
                Vec::new(),
                Vec::new(),
                &device_ctx,
            )
            .is_none());
    }
}

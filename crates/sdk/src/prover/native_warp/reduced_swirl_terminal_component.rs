//! SDK ownership adapter for the reduced-SWIRL terminal `Decide` component.
//!
//! The native verifier remains the authority for the final accumulator,
//! terminal WHIR record, and complete transcript.  This adapter only turns
//! those already checked objects into the borrowed record expected by the
//! recursion AIR trace generator.  It does not accept a host success bit and
//! it never reinterprets a PCS opening as PESAT.

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_continuations::circuit::reduced_swirl_transition_leaf::ReducedSwirlTransitionState;
use openvm_recursion_circuit::native_warp::{
    reduced_swirl_vacc_footer_elements, ReducedSwirlManifestReconciliationReceiptMessage,
    ReducedSwirlTerminalComponent, ReducedSwirlTerminalCpuPacket, ReducedSwirlTerminalError,
    ReducedSwirlTerminalFooterRecord, ReducedSwirlTerminalRecord,
};
use openvm_stark_backend::{
    hasher::MerkleHasher,
    native_warp::native_accumulator_instance_digest_preimage,
    p3_field::{BasedVectorSpace, PrimeCharacteristicRing},
    transcript::{TranscriptCheckpoint, TranscriptHistory, TranscriptLog},
    warp_accum::{derive_swirl_constrained_rs_terminal_statement, TerminalConstrainedRsStatement},
    AirRef, StarkProtocolConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, D_EF, EF, F};

use super::{
    reduced_swirl_native::{
        ReducedSwirlNativeProverOutput, ReducedSwirlNativeSetup, ReducedSwirlNativeVerification,
    },
    reduced_swirl_wrapper_components::ReducedSwirlVerifierComponent,
};
use crate::SC;

#[derive(Debug, thiserror::Error)]
pub enum ReducedSwirlTerminalAdapterError {
    #[error("native reduced-SWIRL terminal inventory is malformed: {0}")]
    Inventory(&'static str),
    #[error("native reduced-SWIRL terminal statement derivation failed: {0}")]
    Statement(String),
    #[error("native reduced-SWIRL finalizer binding is malformed: {0}")]
    Finalizer(&'static str),
    #[error(transparent)]
    Terminal(#[from] ReducedSwirlTerminalError),
}

/// Owned portion of the recursion terminal record.
///
/// The final accumulator, descriptor, and recorded WHIR verification remain
/// borrowed from the independently verified native result when `as_record`
/// is called.  Keeping them out of this carrier prevents accidental copies or
/// caller-selected replacements.
pub struct PreparedReducedSwirlTerminalRecord {
    pub footer: ReducedSwirlTerminalFooterRecord,
    pub statement: TerminalConstrainedRsStatement<EF>,
    pub transcript: TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
}

/// Transcript material retained by the streaming transition-tree finalizer.
///
/// `suffix` starts at `start.operations`, immediately before the canonical
/// block-manifest footer.  Its operation and event ranges are rebased by
/// [`TranscriptLog::suffix`], while `start.operations` remains the absolute
/// cursor passed to a resumed `TranscriptAir`.  Consequently no preceding
/// WARP call is copied into the finalizer witness.
pub struct ReducedSwirlFinalizerTranscriptSuffix {
    suffix: TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    start: TranscriptCheckpoint,
    start_sample_count: usize,
    start_state: [F; POSEIDON2_WIDTH],
    terminal_start: TranscriptCheckpoint,
    terminal_end: TranscriptCheckpoint,
}

impl ReducedSwirlFinalizerTranscriptSuffix {
    #[must_use]
    pub const fn local_proof_idx(&self) -> usize {
        0
    }

    #[must_use]
    pub const fn start_checkpoint(&self) -> TranscriptCheckpoint {
        self.start
    }

    #[must_use]
    pub const fn start_sample_count(&self) -> usize {
        self.start_sample_count
    }

    #[must_use]
    pub const fn terminal_start_checkpoint(&self) -> TranscriptCheckpoint {
        self.terminal_start
    }

    #[must_use]
    pub const fn terminal_end_checkpoint(&self) -> TranscriptCheckpoint {
        self.terminal_end
    }

    #[must_use]
    pub const fn log(&self) -> &TranscriptLog<F, [F; POSEIDON2_WIDTH]> {
        &self.suffix
    }

    /// Exact entry consumed by
    /// `NativeWarpTranscriptModule::generate_trace_inputs_with_external_resumed`.
    #[must_use]
    pub const fn resume_input(&self) -> (usize, [F; POSEIDON2_WIDTH]) {
        (self.start.operations, self.start_state)
    }
}

/// Finalizer-owned terminal preparation for a streaming transition tree.
///
/// The global call index remains in `footer.proof_idx`; only the one physical
/// transcript log proved by the finalizer is re-keyed to local namespace zero.
/// The native terminal verification and full log remain borrowed when
/// `as_record` is called, while `transcript` owns only the resumed suffix.
pub struct PreparedReducedSwirlFinalizerTerminalRecord {
    pub footer: ReducedSwirlTerminalFooterRecord,
    pub statement: TerminalConstrainedRsStatement<EF>,
    transcript: ReducedSwirlFinalizerTranscriptSuffix,
}

impl PreparedReducedSwirlFinalizerTerminalRecord {
    pub fn as_record<'a>(
        &'a self,
        output: &'a ReducedSwirlNativeProverOutput,
        verification: &'a ReducedSwirlNativeVerification,
    ) -> Result<ReducedSwirlTerminalRecord<'a>, ReducedSwirlTerminalAdapterError> {
        validate_native_terminal_inventory(output, verification)?;
        Ok(ReducedSwirlTerminalRecord {
            footer: &self.footer,
            instance: &verification.final_instance,
            descriptor: &output.proof.terminal.descriptor,
            statement: &self.statement,
            verification: &verification.terminal,
            transcript: &verification.complete_transcript.log,
        })
    }

    #[must_use]
    pub const fn transcript(&self) -> &ReducedSwirlFinalizerTranscriptSuffix {
        &self.transcript
    }
}

impl PreparedReducedSwirlTerminalRecord {
    pub fn as_record<'a>(
        &'a self,
        output: &'a ReducedSwirlNativeProverOutput,
        verification: &'a ReducedSwirlNativeVerification,
    ) -> Result<ReducedSwirlTerminalRecord<'a>, ReducedSwirlTerminalAdapterError> {
        validate_native_terminal_inventory(output, verification)?;
        Ok(ReducedSwirlTerminalRecord {
            footer: &self.footer,
            instance: &verification.final_instance,
            descriptor: &output.proof.terminal.descriptor,
            statement: &self.statement,
            verification: &verification.terminal,
            transcript: &self.transcript,
        })
    }
}

/// Derive the exact terminal AIR witness boundary from an independently
/// recorded native verification.
pub fn prepare_reduced_swirl_terminal_record(
    setup: &ReducedSwirlNativeSetup,
    output: &ReducedSwirlNativeProverOutput,
    verification: &ReducedSwirlNativeVerification,
) -> Result<PreparedReducedSwirlTerminalRecord, ReducedSwirlTerminalAdapterError> {
    validate_native_terminal_inventory(output, verification)?;
    let source_count = output.authoritative_claims.len();
    let call_count = output.proof.vacc.steps.len();
    let statement = derive_swirl_constrained_rs_terminal_statement(
        setup.relation(),
        setup.code(),
        &verification.final_instance,
    )
    .map_err(|error| ReducedSwirlTerminalAdapterError::Statement(format!("{error:?}")))?;
    let transcript = TranscriptHistory::into_log(verification.complete_transcript.clone());
    let footer_element_count = reduced_swirl_vacc_footer_elements(
        source_count,
        output.proof.statement.block_manifest_digest,
    )
    .map_err(ReducedSwirlTerminalAdapterError::Inventory)?
    .len();
    let footer_width = footer_element_count.checked_mul(D_EF).ok_or(
        ReducedSwirlTerminalAdapterError::Inventory("manifest-footer width overflow"),
    )?;
    let end_tidx = verification.terminal.transcript_start.operations;
    let start_tidx =
        end_tidx
            .checked_sub(footer_width)
            .ok_or(ReducedSwirlTerminalAdapterError::Inventory(
                "manifest-footer transcript interval",
            ))?;
    if end_tidx > transcript.len()
        || call_count == 0
        || verification.terminal.transcript_end.operations != transcript.len()
    {
        return Err(ReducedSwirlTerminalAdapterError::Inventory(
            "terminal transcript boundary",
        ));
    }
    Ok(PreparedReducedSwirlTerminalRecord {
        footer: ReducedSwirlTerminalFooterRecord {
            source_count,
            call_count,
            proof_idx: call_count - 1,
            local_proof_idx: call_count - 1,
            start_tidx,
            end_tidx,
            manifest_digest: output.proof.statement.block_manifest_digest,
        },
        statement,
        transcript,
    })
}

/// Prepare the terminal `Decide` witness for the streaming transition-tree
/// finalizer without cloning or replaying the preceding WARP calls.
///
/// The reconciliation receipt is circuit-produced: its flat digest binds the
/// native terminal footer and its rolling endpoint binds the final transition
/// state.  There is deliberately no caller-provided success flag.
pub fn prepare_reduced_swirl_finalizer_terminal_record(
    setup: &ReducedSwirlNativeSetup,
    output: &ReducedSwirlNativeProverOutput,
    verification: &ReducedSwirlNativeVerification,
    final_state: &ReducedSwirlTransitionState,
    reconciliation: &ReducedSwirlManifestReconciliationReceiptMessage<F>,
) -> Result<PreparedReducedSwirlFinalizerTerminalRecord, ReducedSwirlTerminalAdapterError> {
    validate_native_terminal_inventory(output, verification)?;
    let source_count = output.authoritative_claims.len();
    let call_count = output.proof.vacc.steps.len();
    let source_count_field = F::from_usize(source_count);
    let call_count_field = F::from_usize(call_count);
    if final_state.total_source_count != source_count_field
        || final_state.source_cursor != source_count_field
        || final_state.call_cursor != call_count_field
        || reconciliation.source_count != source_count_field
        || reconciliation.call_count != call_count_field
    {
        return Err(ReducedSwirlTerminalAdapterError::Finalizer(
            "source/call count",
        ));
    }
    if reconciliation.flat_manifest_digest != output.proof.statement.block_manifest_digest
        || reconciliation.rolling_chain_endpoint != final_state.manifest_chain
    {
        return Err(ReducedSwirlTerminalAdapterError::Finalizer(
            "manifest reconciliation",
        ));
    }

    let final_accumulator_digest = canonical_final_accumulator_digest(setup, verification);
    if final_state.accumulator_root != verification.final_instance.rt
        || final_state.accumulator_digest != final_accumulator_digest
    {
        return Err(ReducedSwirlTerminalAdapterError::Finalizer(
            "final accumulator",
        ));
    }

    let footer_element_count =
        reduced_swirl_vacc_footer_elements(source_count, reconciliation.flat_manifest_digest)
            .map_err(ReducedSwirlTerminalAdapterError::Inventory)?
            .len();
    let footer_width = footer_element_count.checked_mul(D_EF).ok_or(
        ReducedSwirlTerminalAdapterError::Inventory("manifest-footer width overflow"),
    )?;
    let footer_end_tidx = verification.terminal.transcript_start.operations;
    let footer_start_tidx = footer_end_tidx.checked_sub(footer_width).ok_or(
        ReducedSwirlTerminalAdapterError::Inventory("manifest-footer transcript interval"),
    )?;
    let full_log = &verification.complete_transcript.log;
    if verification.terminal.transcript_end.operations != full_log.len() {
        return Err(ReducedSwirlTerminalAdapterError::Finalizer(
            "terminal transcript end",
        ));
    }
    let start = resumable_checkpoint_at(full_log, footer_start_tidx)?;
    let start_sample_count = trailing_sample_count(full_log, start.operations)?;
    let suffix = full_log
        .suffix(start)
        .ok_or(ReducedSwirlTerminalAdapterError::Finalizer(
            "terminal transcript suffix",
        ))?;
    let start_state = suffix.perm_results().first().copied().ok_or(
        ReducedSwirlTerminalAdapterError::Finalizer("terminal resume state"),
    )?;
    if final_state.transcript_tidx != F::from_usize(footer_start_tidx)
        || final_state.transcript_sample_count != F::from_usize(start_sample_count)
        || final_state.transcript_state != start_state
    {
        return Err(ReducedSwirlTerminalAdapterError::Finalizer(
            "terminal footer start checkpoint",
        ));
    }

    let statement = derive_swirl_constrained_rs_terminal_statement(
        setup.relation(),
        setup.code(),
        &verification.final_instance,
    )
    .map_err(|error| ReducedSwirlTerminalAdapterError::Statement(format!("{error:?}")))?;
    Ok(PreparedReducedSwirlFinalizerTerminalRecord {
        footer: ReducedSwirlTerminalFooterRecord {
            source_count,
            call_count,
            proof_idx: call_count - 1,
            local_proof_idx: 0,
            start_tidx: footer_start_tidx,
            end_tidx: footer_end_tidx,
            manifest_digest: reconciliation.flat_manifest_digest,
        },
        statement,
        transcript: ReducedSwirlFinalizerTranscriptSuffix {
            suffix,
            start,
            start_sample_count,
            start_state,
            terminal_start: verification.terminal.transcript_start,
            terminal_end: verification.terminal.transcript_end,
        },
    })
}

/// Generate the setup-fixed terminal contexts and typed wrapper receipt.
pub fn generate_reduced_swirl_terminal_cpu_packet(
    component: &ReducedSwirlTerminalComponent,
    prepared: &PreparedReducedSwirlTerminalRecord,
    output: &ReducedSwirlNativeProverOutput,
    verification: &ReducedSwirlNativeVerification,
) -> Result<ReducedSwirlTerminalCpuPacket<SC>, ReducedSwirlTerminalAdapterError> {
    Ok(component.generate_cpu_contexts::<SC>(prepared.as_record(output, verification)?)?)
}

/// Generate the unchanged terminal `Decide`/WHIR contexts from a finalizer
/// preparation. The existing native record is borrowed; no prior-call
/// transcript is cloned into the preparation.
pub fn generate_reduced_swirl_finalizer_terminal_cpu_packet(
    component: &ReducedSwirlTerminalComponent,
    prepared: &PreparedReducedSwirlFinalizerTerminalRecord,
    output: &ReducedSwirlNativeProverOutput,
    verification: &ReducedSwirlNativeVerification,
) -> Result<ReducedSwirlTerminalCpuPacket<SC>, ReducedSwirlTerminalAdapterError> {
    Ok(component.generate_cpu_contexts::<SC>(prepared.as_record(output, verification)?)?)
}

impl ReducedSwirlVerifierComponent for ReducedSwirlTerminalComponent {
    fn protocol_digest(&self) -> Digest {
        ReducedSwirlTerminalComponent::protocol_digest(self)
    }

    fn airs<C: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<C>> {
        ReducedSwirlTerminalComponent::airs::<C>(self)
    }
}

fn validate_native_terminal_inventory(
    output: &ReducedSwirlNativeProverOutput,
    verification: &ReducedSwirlNativeVerification,
) -> Result<(), ReducedSwirlTerminalAdapterError> {
    if output.authoritative_claims.is_empty()
        || output.proof.vacc.steps.is_empty()
        || output.proof.statement.source_bindings.len() != output.authoritative_claims.len()
        || output.transition_records.len() != output.proof.vacc.steps.len()
        || verification.transition_records.len() != output.proof.vacc.steps.len()
        || verification.final_instance != output.proof.vacc.final_instance
        || verification.terminal.root != verification.final_instance.rt
        || output.proof.terminal.descriptor.root != verification.final_instance.rt
        || verification.final_instance.alpha.len()
            != output.proof.terminal.descriptor.log_codeword_len as usize
        || verification.final_instance.beta.len()
            != output.proof.terminal.descriptor.log_message_len as usize + 1
    {
        return Err(ReducedSwirlTerminalAdapterError::Inventory(
            "native terminal proof/verification mismatch",
        ));
    }
    // Keep the field-basis assertion visible at this ownership boundary: the
    // terminal AIR and backend both operate on the same EF4 coordinates.
    if <EF as BasedVectorSpace<F>>::DIMENSION != D_EF {
        return Err(ReducedSwirlTerminalAdapterError::Inventory(
            "extension-field basis dimension",
        ));
    }
    Ok(())
}

fn canonical_final_accumulator_digest(
    setup: &ReducedSwirlNativeSetup,
    verification: &ReducedSwirlNativeVerification,
) -> Digest {
    let preimage = native_accumulator_instance_digest_preimage::<SC>(&verification.final_instance);
    let payload = setup.code().hasher().hash_slice(&preimage);
    setup
        .code()
        .hasher()
        .compress(verification.final_instance.rt, payload)
}

fn resumable_checkpoint_at(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    operation: usize,
) -> Result<TranscriptCheckpoint, ReducedSwirlTerminalAdapterError> {
    let (events, event) = log
        .events()
        .iter()
        .enumerate()
        .find(|(_, event)| event.operation_range.start == operation)
        .ok_or(ReducedSwirlTerminalAdapterError::Finalizer(
            "terminal footer event boundary",
        ))?;
    let checkpoint = TranscriptCheckpoint {
        operations: operation,
        events,
        permutations: event.permutation_range.start,
    };
    if log.resumable_checkpoint_before::<DIGEST_SIZE>(checkpoint) != Some(checkpoint) {
        return Err(ReducedSwirlTerminalAdapterError::Finalizer(
            "terminal footer is not resumable",
        ));
    }
    Ok(checkpoint)
}

fn trailing_sample_count(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    operation: usize,
) -> Result<usize, ReducedSwirlTerminalAdapterError> {
    let samples =
        log.samples()
            .get(..operation)
            .ok_or(ReducedSwirlTerminalAdapterError::Finalizer(
                "terminal checkpoint cursor",
            ))?;
    let count = samples
        .iter()
        .rev()
        .take_while(|&&is_sample| is_sample)
        .count();
    if count == 0 || count > DIGEST_SIZE {
        return Err(ReducedSwirlTerminalAdapterError::Finalizer(
            "terminal checkpoint sample cursor",
        ));
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use openvm_cpu_backend::CpuReducedSwirlSource;
    use openvm_recursion_circuit::native_warp::{
        generate_reduced_swirl_manifest_reconciliation_trace,
        ReducedSwirlManifestReconciliationReceiptMessage,
    };
    use openvm_stark_backend::{
        prover::{DeviceDataTransporter, NativeStackingReduction},
        test_utils::{default_test_params_small, PreprocessedFibFixture, TestFixture},
        StarkEngine, WhirProximityStrategy,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2CpuEngine, DuplexSponge};

    use super::*;
    use crate::prover::native_warp::reduced_swirl_native::{
        reduced_swirl_source_challenger, verify_reduced_swirl_native_recorded,
        ReducedSwirlNativeCpuStream,
    };

    type Engine = BabyBearPoseidon2CpuEngine<DuplexSponge>;

    fn digest(seed: u32) -> Digest {
        core::array::from_fn(|index| F::from_u32(seed + index as u32 + 1))
    }

    fn genuine_finalizer_fixture() -> eyre::Result<(
        ReducedSwirlNativeSetup,
        ReducedSwirlNativeProverOutput,
        ReducedSwirlNativeVerification,
        ReducedSwirlTransitionState,
        ReducedSwirlManifestReconciliationReceiptMessage<F>,
    )> {
        let mut params = default_test_params_small();
        params.whir.proximity = WhirProximityStrategy::UniqueDecoding;
        let engine = Engine::new(params.clone());
        let selectors = vec![true; 1 << 5];
        let key_fixture = PreprocessedFibFixture::new(0, 1, selectors.clone());
        let (host_pk, _vk) = key_fixture.keygen(&engine);
        let device_pk = engine.device().transport_pk_to_device(&host_pk);
        let source_count = 3;
        let setup = ReducedSwirlNativeSetup::new(engine.config(), &params, 2, 16, source_count)?;
        let mut stream = ReducedSwirlNativeCpuStream::new(&setup, source_count)?;
        for (source_index, (a, b)) in [(0, 1), (2, 3), (4, 5)].into_iter().enumerate() {
            let fixture = PreprocessedFibFixture::new(a, b, selectors.clone());
            let proving_context = engine
                .device()
                .transport_proving_ctx_to_device(&fixture.generate_proving_ctx());
            let mut prefix_prover = engine.prover();
            let NativeStackingReduction {
                stacking_proof,
                pending_witness,
                ..
            } = prefix_prover.prove_native_stacking_reduction(&device_pk, proving_context)?;
            let mut source_challenger = reduced_swirl_source_challenger();
            let source = CpuReducedSwirlSource::try_from_pending(
                pending_witness,
                &stacking_proof.stacking_openings,
                &mut source_challenger,
            )?;
            stream.push_source(digest(1_000 + source_index as u32 * 20), source)?;
        }
        let output = stream.finish()?;
        let verification = verify_reduced_swirl_native_recorded(
            &setup,
            &output.proof.statement,
            &output.authoritative_claims,
            &output.proof,
        )?;
        let source_protocol_digest = digest(2_000);
        let reconciliation = generate_reduced_swirl_manifest_reconciliation_trace(
            2,
            source_protocol_digest,
            &output.proof.statement.source_bindings,
        )
        .map_err(eyre::Report::msg)?
        .receipt;
        let footer_width =
            reduced_swirl_vacc_footer_elements(source_count, reconciliation.flat_manifest_digest)
                .map_err(eyre::Report::msg)?
                .len()
                * D_EF;
        let footer_start_tidx = verification
            .terminal
            .transcript_start
            .operations
            .checked_sub(footer_width)
            .ok_or_else(|| eyre::eyre!("footer cursor"))?;
        let checkpoint =
            resumable_checkpoint_at(&verification.complete_transcript.log, footer_start_tidx)?;
        let transcript_state =
            verification.complete_transcript.log.perm_results()[checkpoint.permutations];
        let transcript_sample_count =
            trailing_sample_count(&verification.complete_transcript.log, checkpoint.operations)?;
        let final_state = ReducedSwirlTransitionState {
            source_protocol_digest,
            warp_protocol_digest: digest(2_100),
            relation_digest: digest(2_200),
            warp_index_digest: digest(2_300),
            schedule_digest: digest(2_400),
            total_source_count: F::from_usize(source_count),
            call_cursor: F::from_usize(output.proof.vacc.steps.len()),
            source_cursor: F::from_usize(source_count),
            transcript_tidx: F::from_usize(footer_start_tidx),
            transcript_sample_count: F::from_usize(transcript_sample_count),
            transcript_state,
            accumulator_root: verification.final_instance.rt,
            accumulator_digest: canonical_final_accumulator_digest(&setup, &verification),
            vm_pc: F::from_u32(99),
            vm_root: digest(2_500),
            manifest_chain: reconciliation.rolling_chain_endpoint,
            program_commitment: digest(2_600),
        };
        Ok((setup, output, verification, final_state, reconciliation))
    }

    #[test]
    fn finalizer_preparation_is_suffix_only_and_rejects_bound_state_mutations() -> eyre::Result<()>
    {
        let (setup, output, verification, state, reconciliation) = genuine_finalizer_fixture()?;
        assert_eq!(output.proof.vacc.steps.len(), 2);
        let prepared = prepare_reduced_swirl_finalizer_terminal_record(
            &setup,
            &output,
            &verification,
            &state,
            &reconciliation,
        )?;
        assert_eq!(prepared.footer.proof_idx, 1);
        assert_eq!(prepared.footer.local_proof_idx, 0);
        assert_eq!(prepared.transcript().local_proof_idx(), 0);
        assert_eq!(
            prepared.transcript().resume_input(),
            (prepared.footer.start_tidx, state.transcript_state)
        );
        assert_eq!(
            F::from_usize(prepared.transcript().start_sample_count()),
            state.transcript_sample_count
        );
        assert_eq!(
            prepared.transcript().log().values(),
            &verification.complete_transcript.log.values()[prepared.footer.start_tidx..]
        );
        assert!(prepared.transcript().log().len() < verification.complete_transcript.log.len());
        assert!(core::ptr::eq(
            prepared.as_record(&output, &verification)?.transcript,
            &verification.complete_transcript.log,
        ));

        let rejects = |candidate_state: &ReducedSwirlTransitionState,
                       candidate_reconciliation: &ReducedSwirlManifestReconciliationReceiptMessage<F>| {
            prepare_reduced_swirl_finalizer_terminal_record(
                &setup,
                &output,
                &verification,
                candidate_state,
                candidate_reconciliation,
            )
            .is_err()
        };

        let mut wrong = state.clone();
        wrong.source_cursor += F::ONE;
        assert!(rejects(&wrong, &reconciliation));
        let mut wrong = state.clone();
        wrong.call_cursor += F::ONE;
        assert!(rejects(&wrong, &reconciliation));
        let mut wrong = state.clone();
        wrong.accumulator_root[0] += F::ONE;
        assert!(rejects(&wrong, &reconciliation));
        let mut wrong = state.clone();
        wrong.accumulator_digest[0] += F::ONE;
        assert!(rejects(&wrong, &reconciliation));
        let mut wrong = state.clone();
        wrong.transcript_tidx += F::ONE;
        assert!(rejects(&wrong, &reconciliation));
        let mut wrong = state.clone();
        wrong.transcript_sample_count += F::ONE;
        assert!(rejects(&wrong, &reconciliation));
        let mut wrong = state.clone();
        wrong.transcript_state[0] += F::ONE;
        assert!(rejects(&wrong, &reconciliation));
        let mut wrong_reconciliation = reconciliation.clone();
        wrong_reconciliation.flat_manifest_digest[0] += F::ONE;
        assert!(rejects(&state, &wrong_reconciliation));
        let mut wrong_reconciliation = reconciliation.clone();
        wrong_reconciliation.rolling_chain_endpoint[0] += F::ONE;
        assert!(rejects(&state, &wrong_reconciliation));
        Ok(())
    }
}

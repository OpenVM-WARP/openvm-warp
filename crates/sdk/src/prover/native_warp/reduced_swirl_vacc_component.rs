//! Production composition for recursive verification of reduced-SWIRL WARP
//! transitions.
//!
//! The native proof has one continuous transcript split into setup-fixed
//! bootstrap and continuation records.  The recursion layer authenticates
//! the original SWIRL root tuples and the ordinary WARP algebra; it never
//! replaces those roots by a scalar commitment and never treats native
//! verifier success as a witness bit.

use std::sync::Arc;

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_continuations::circuit::{
    native_warp_accumulator::{
        generate_native_accumulator_digest_traces, NativePrivateAccumulatorLayout,
    },
    native_warp_history_v19::{
        CertifiedDirectAirVaccInputBusV19, DirectAirReducedSwirlVaccConfigV19,
        DirectAirReducedSwirlVaccVerifierRecordV19, DirectAirVaccContextBusV19,
        DirectAirVaccHistoryBusesV19, DirectAirVaccVerifierModuleV19,
    },
    reduced_swirl_source_receipt::ReducedSwirlSourceReceiptBlock,
    reduced_swirl_warp::{ReducedSwirlVaccReceiptBus, ReducedSwirlVaccReceiptMessage},
};
use openvm_cpu_backend::CpuBackend;
use openvm_recursion_circuit::{
    bus::TranscriptBus,
    native_warp::{
        reduced_swirl_source_entry_digest, reduced_swirl_vacc_protocol_prefix_elements,
        reduced_swirl_vacc_schedule, validate_reduced_swirl_fresh_batch,
        NativeStandardVaccDigestBus, NativeStandardVaccEndBus, NativeStandardVaccProfile,
        NativeStandardVaccProtocolBus, NativeStandardVaccRootBus, NativeWarpPcdBusInventory,
        NativeWarpTranscriptModule, RecursiveReducedSwirlClaim, ReducedSwirlManifestDigestBus,
        ReducedSwirlSourceAuthorityRecord, ReducedSwirlSourceDigestRecord,
        ReducedSwirlSourceEntryDigestBus, ReducedSwirlTerminalProductionSetup,
        ReducedSwirlVaccAggregateBuses, ReducedSwirlVaccAggregateComponent,
        ReducedSwirlVaccAggregateRecord, ReducedSwirlVaccCallBus, ReducedSwirlVaccCallRecord,
        ReducedSwirlVaccChainEndBus, ReducedSwirlVaccChainReceiptBus,
        ReducedSwirlVaccChainReceiptMessage, ReducedSwirlVaccDetachedSourceBus,
        ReducedSwirlVaccFooterBus, ReducedSwirlVaccHeaderEndBus, ReducedSwirlVaccProfile,
        ReducedSwirlVaccSourceMode, ReducedSwirlVaccSourceSlotBus,
        ReducedSwirlVaccSourceSummaryBus, ReducedSwirlVaccSourceSummaryMessage,
        ReducedSwirlVaccSummaryManifestBus, ReducedSwirlVaccTransitionAggregateComponent,
        ReducedSwirlVaccTransitionAggregateRecord, ReducedSwirlVaccTransitionEndBus,
        ReducedSwirlVaccTransitionReceiptBus, ReducedSwirlVaccTransitionReceiptMessage,
        REDUCED_SWIRL_VACC_MAX_ROOTS, REDUCED_SWIRL_VACC_SOURCE_LEAF_CAPACITY,
        REDUCED_SWIRL_VACC_SOURCE_LEAF_PROTOCOL_VERSION,
    },
    system::{BusIndexManager, BusInventory},
};
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{BasedVectorSpace, PrimeCharacteristicRing},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::AirProvingContext,
    warp_accum::{
        canonical_reduced_constrained_code_protocol_binding, canonical_swirl_reduced_code_binding,
        NativeTranscriptPhase, NativeTranscriptPhaseSpan, ReducedConstrainedCodeRelation,
    },
    AirRef, BaseAirWithPublicValues, FiatShamirTranscript, PartitionedBaseAir, StarkProtocolConfig,
    SystemParams, TranscriptCheckpoint, TranscriptHistory, TranscriptLog,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, BabyBearPoseidon2Config, Digest, CHUNK, DIGEST_SIZE, D_EF, EF,
    F,
};

#[cfg(feature = "cuda")]
use super::reduced_swirl_native_cuda::ReducedSwirlNativeCudaCompletedStep;
use super::{
    reduced_swirl_native::{
        ReducedSwirlNativeProverOutput, ReducedSwirlNativeSetup, ReducedSwirlNativeVerification,
    },
    reduced_swirl_source_receipt::ProductionReducedSwirlSourceReceiptComponent,
    reduced_swirl_wrapper_components::ReducedSwirlVerifierComponent,
};
use crate::SC;

const RECEIPT_FIELD_COUNT: usize = 7 * DIGEST_SIZE + 2;
const RECEIPT_BRIDGE_WIDTH: usize = 1 + RECEIPT_FIELD_COUNT;

#[derive(Debug, thiserror::Error)]
pub enum ReducedSwirlVaccComponentError {
    #[error("invalid reduced-SWIRL VACC profile: {0}")]
    Profile(&'static str),
    #[error("native prover and verifier transition inventories differ")]
    TransitionInventory,
    #[error("native transition {step} differs from its recorded verification")]
    TransitionMismatch { step: usize },
    #[error("native transition {step} has malformed transcript phases: {message}")]
    Transcript { step: usize, message: &'static str },
    #[error("native transition {step} does not authenticate its original SWIRL roots")]
    FreshAuthentication { step: usize },
    #[error("source receipt inventory differs from the native statement")]
    SourceInventory,
    #[error(
        "source {source_index} canonical schedule/authority digest differs from the native binding"
    )]
    SourceBinding { source_index: usize },
    #[error("native accumulator digest generation failed at transition {step}")]
    AccumulatorDigest { step: usize },
    #[error("reduced-SWIRL VACC aggregate record failed: {0}")]
    Aggregate(&'static str),
    #[error("reduced-SWIRL VACC component AIR/context inventory differs")]
    Inventory,
    #[error("reduced-SWIRL VACC production setup failed: {0}")]
    Setup(String),
    #[error("ordinary v19 reduced-SWIRL VACC verification failed: {0}")]
    StandardVacc(String),
    #[error("reduced-SWIRL VACC transcript witness failed: {0}")]
    TranscriptWitness(&'static str),
}

/// Build the recursion-local record exclusively from native proof material,
/// verifier-recorded Fiat--Shamir state, and source AIR authority records.
///
/// The source block is a necessary private witness.  It cannot be recovered
/// from `ReducedSwirlNativeProverOutput` or `ReducedSwirlNativeVerification`:
/// those structures intentionally retain only hashes of VM boundaries and
/// deferred-SWIRL checkpoints.  Accepting only the two native carriers would
/// force the recursive prover to invent hash preimages, so the production
/// wrapper passes the block emitted by the source component alongside them.
pub fn build_reduced_swirl_vacc_aggregate_record(
    profile: &ReducedSwirlVaccProfile,
    output: &ReducedSwirlNativeProverOutput,
    verification: &ReducedSwirlNativeVerification,
    source_block: &ReducedSwirlSourceReceiptBlock,
) -> Result<ReducedSwirlVaccAggregateRecord, ReducedSwirlVaccComponentError> {
    profile
        .validate()
        .map_err(ReducedSwirlVaccComponentError::Profile)?;
    let source_count = output.authoritative_claims.len();
    if source_count == 0
        || source_count > profile.maximum_sources
        || output.proof.statement.source_bindings.len() != source_count
        || source_block.sources.len() != source_count
        || output.proof.vacc.steps.len() != output.transition_records.len()
        || output.transition_records.len() != verification.transition_records.len()
    {
        return Err(ReducedSwirlVaccComponentError::SourceInventory);
    }
    let expected_calls = reduced_swirl_vacc_schedule(source_count, profile.input_arity)
        .map_err(ReducedSwirlVaccComponentError::Profile)?;
    if expected_calls.len() != output.transition_records.len() {
        return Err(ReducedSwirlVaccComponentError::TransitionInventory);
    }

    let transcript = openvm_stark_backend::transcript::TranscriptHistory::into_log(
        verification.complete_transcript.clone(),
    );
    let accumulator_layout = NativePrivateAccumulatorLayout::new(
        0,
        profile.log_codeword_len(),
        profile.normalized_beta_len(),
    );
    let mut calls: Vec<ReducedSwirlVaccCallRecord> = Vec::with_capacity(expected_calls.len());
    let mut sources = Vec::with_capacity(source_count);
    let mut source_offset = 0usize;
    let mut previous_output_digest = None;
    let mut previous_end_sample_count = 0usize;
    let mut previous_end_state = [F::ZERO; POSEIDON2_WIDTH];
    let tree_source_stride = profile
        .source
        .maximum_roots_per_source
        .checked_mul(1 + profile.num_shift_queries)
        .ok_or(ReducedSwirlVaccComponentError::Aggregate(
            "direct fresh tree-source stride",
        ))?;
    let header_end = openvm_recursion_circuit::native_warp::reduced_swirl_vacc_header_elements(
        profile,
        source_count,
    )
    .map_err(ReducedSwirlVaccComponentError::Aggregate)?
    .len()
        * D_EF;

    for (step, (((call, step_proof), prover_record), verifier_record)) in expected_calls
        .iter()
        .copied()
        .zip(&output.proof.vacc.steps)
        .zip(&output.transition_records)
        .zip(&verification.transition_records)
        .enumerate()
    {
        if step_proof.fresh_count() != call.fresh_count
            || !same_recorded_transition(prover_record, verifier_record)
        {
            return Err(ReducedSwirlVaccComponentError::TransitionMismatch { step });
        }
        let source_end = source_offset + call.fresh_count;
        let local_claims = output.authoritative_claims[source_offset..source_end]
            .iter()
            .map(recursive_claim_from_native)
            .collect::<Vec<_>>();
        validate_reduced_swirl_fresh_batch(profile, &local_claims, step_proof, verifier_record)
            .map_err(|_| ReducedSwirlVaccComponentError::FreshAuthentication { step })?;

        let protocol = unique_phase(
            step,
            &verifier_record.transcript_phases,
            &NativeTranscriptPhase::VaccProtocolPrefix,
        )?;
        let commitments = unique_phase(
            step,
            &verifier_record.transcript_phases,
            &NativeTranscriptPhase::FreshCommitments,
        )?;
        let target = unique_phase(
            step,
            &verifier_record.transcript_phases,
            &NativeTranscriptPhase::FinalTarget,
        )?;
        let boundary = unique_phase(
            step,
            &verifier_record.transcript_phases,
            &NativeTranscriptPhase::ExactFiniteCallBoundary {
                call: step
                    .try_into()
                    .map_err(|_| ReducedSwirlVaccComponentError::Transcript {
                        step,
                        message: "call-index width",
                    })?,
            },
        )?;
        let vacc_start_tidx = protocol.operation_range.start;
        let vacc_end_tidx = boundary.operation_range.end;
        let expected_protocol_end = vacc_start_tidx
            .checked_add(
                reduced_swirl_vacc_protocol_prefix_elements(profile, call)
                    .map_err(ReducedSwirlVaccComponentError::Aggregate)?
                    .len()
                    .checked_mul(D_EF)
                    .ok_or(ReducedSwirlVaccComponentError::Aggregate(
                        "VACC protocol prefix width",
                    ))?,
            )
            .ok_or(ReducedSwirlVaccComponentError::Aggregate(
                "VACC protocol prefix boundary",
            ))?;
        let batch_start_tidx = if step == 0 {
            header_end
        } else {
            calls[step - 1].vacc_end_tidx
        };
        if vacc_start_tidx <= batch_start_tidx
            || commitments.operation_range.start < protocol.operation_range.end
            || protocol.operation_range.end != expected_protocol_end
            || commitments.operation_range.start != expected_protocol_end
            || target.operation_range.end != boundary.operation_range.start
            || target.operation_range.end > transcript.len()
            || vacc_end_tidx > transcript.len()
        {
            return Err(ReducedSwirlVaccComponentError::Transcript {
                step,
                message: "non-canonical call interval",
            });
        }
        let (end_sample_count, end_state) = transcript_checkpoint(&transcript, vacc_end_tidx)
            .ok_or(ReducedSwirlVaccComponentError::Transcript {
                step,
                message: "non-canonical call checkpoint",
            })?;

        let output_digest = generate_native_accumulator_digest_traces(
            step,
            &verifier_record.output_instance,
            &accumulator_layout,
        )
        .ok_or(ReducedSwirlVaccComponentError::AccumulatorDigest { step })?
        .instance_digest;
        let prior = (step != 0).then(|| &verification.transition_records[step - 1].output_instance);
        let prior_digest = if let Some(prior) = prior {
            Some(
                generate_native_accumulator_digest_traces(step, prior, &accumulator_layout)
                    .ok_or(ReducedSwirlVaccComponentError::AccumulatorDigest { step })?
                    .instance_digest,
            )
        } else {
            None
        };
        if prior_digest != previous_output_digest {
            return Err(ReducedSwirlVaccComponentError::TransitionMismatch { step });
        }

        let commitment_tidxs =
            commitment_descriptor_tidxs(step, &transcript, commitments, &local_claims)?;
        for (slot, ((claim, receipt), commitment_tidx)) in local_claims
            .iter()
            .zip(&source_block.sources[source_offset..source_end])
            .zip(commitment_tidxs)
            .enumerate()
        {
            let authority = ReducedSwirlSourceAuthorityRecord {
                segment_index: receipt.segment_index,
                common_main_root: *receipt.roots.first().ok_or(
                    ReducedSwirlVaccComponentError::SourceBinding {
                        source_index: source_offset + slot,
                    },
                )?,
                trace_layout_digest: receipt.layout_digest,
                pending_claim_digest: receipt.pending_digest,
                checkpoint_tidx: receipt.checkpoint_tidx,
                checkpoint_state: receipt.checkpoint_state,
                program_commitment: receipt.vm.program_commitment,
                initial_pc: receipt.vm.initial_pc,
                initial_root: receipt.vm.initial_root,
                final_pc: receipt.vm.final_pc,
                final_root: receipt.vm.final_root,
                exit_code: receipt.vm.exit_code,
                is_terminate: receipt.vm.is_terminate,
            };
            let source = source_offset + slot;
            let entry_digest =
                reduced_swirl_source_entry_digest(profile, source, call, slot, claim, &authority)
                    .map_err(|_| ReducedSwirlVaccComponentError::SourceBinding {
                    source_index: source,
                })?;
            if output.proof.statement.source_bindings[source] != entry_digest {
                return Err(ReducedSwirlVaccComponentError::SourceBinding {
                    source_index: source,
                });
            }
            sources.push(ReducedSwirlSourceDigestRecord {
                claim: claim.clone(),
                authority,
                commitment_tidx,
                first_tree_id: slot
                    .checked_mul(tree_source_stride)
                    .and_then(|value| value.try_into().ok())
                    .ok_or(ReducedSwirlVaccComponentError::SourceBinding {
                        source_index: source,
                    })?,
            });
        }

        calls.push(ReducedSwirlVaccCallRecord {
            call,
            batch_start_tidx,
            vacc_start_tidx,
            vacc_end_tidx,
            start_sample_count: previous_end_sample_count,
            start_state: if step == 0 {
                [F::ZERO; POSEIDON2_WIDTH]
            } else {
                previous_end_state
            },
            end_sample_count,
            end_state,
            prior_root: prior.map(|instance| instance.rt),
            output_root: verifier_record.output_instance.rt,
            prior_digest,
            output_digest,
        });
        previous_output_digest = Some(output_digest);
        previous_end_sample_count = end_sample_count;
        previous_end_state = end_state;
        source_offset = source_end;
    }
    if source_offset != source_count
        || verification.final_instance
            != verification
                .transition_records
                .last()
                .ok_or(ReducedSwirlVaccComponentError::TransitionInventory)?
                .output_instance
    {
        return Err(ReducedSwirlVaccComponentError::TransitionInventory);
    }
    Ok(ReducedSwirlVaccAggregateRecord { calls, sources })
}

fn recursive_claim_from_native(
    claim: &openvm_stark_backend::warp_accum::ReducedConstrainedCodeClaim<
        EF,
        super::reduced_swirl_native::ReducedSwirlFreshCommitment,
    >,
) -> RecursiveReducedSwirlClaim {
    RecursiveReducedSwirlClaim {
        roots: claim.commitment.roots.clone(),
        widths: claim.commitment.widths.clone(),
        theta: claim.commitment.theta,
        alpha: claim.alpha.clone(),
        mu: claim.mu,
        beta: claim.beta.clone(),
        eta: claim.eta,
    }
}

fn same_recorded_transition(
    prover: &openvm_stark_backend::warp_accum::WarpVaccStepProverRecord<EF, Digest>,
    verifier: &super::reduced_swirl_native::ReducedSwirlStepVerification,
) -> bool {
    prover.output_instance == verifier.output_instance
        && prover.transcript_phases == verifier.transcript_phases
        && prover.twin == verifier.twin
        && prover.ood == verifier.ood
        && prover.shifts == verifier.shifts
        && prover.batching == verifier.batching
}

fn unique_phase<'a>(
    step: usize,
    phases: &'a [NativeTranscriptPhaseSpan],
    wanted: &NativeTranscriptPhase,
) -> Result<&'a NativeTranscriptPhaseSpan, ReducedSwirlVaccComponentError> {
    let mut matches = phases.iter().filter(|phase| &phase.phase == wanted);
    let phase = matches
        .next()
        .ok_or(ReducedSwirlVaccComponentError::Transcript {
            step,
            message: "missing phase",
        })?;
    if matches.next().is_some() {
        return Err(ReducedSwirlVaccComponentError::Transcript {
            step,
            message: "duplicate phase",
        });
    }
    Ok(phase)
}

fn transcript_checkpoint(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    operation_index: usize,
) -> Option<(usize, [F; POSEIDON2_WIDTH])> {
    let (event_index, event) = log
        .events()
        .iter()
        .enumerate()
        .find(|(_, event)| event.operation_range.end == operation_index)?;
    let prefix = log.prefix(TranscriptCheckpoint {
        operations: operation_index,
        events: event_index + 1,
        permutations: event.permutation_range.end,
    })?;
    let sample_count = prefix
        .samples()
        .iter()
        .rev()
        .take_while(|&&sample| sample)
        .count();
    if sample_count > CHUNK {
        return None;
    }
    Some((sample_count, *prefix.perm_results().last()?))
}

fn commitment_descriptor_tidxs(
    step: usize,
    transcript: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    phase: &NativeTranscriptPhaseSpan,
    claims: &[RecursiveReducedSwirlClaim],
) -> Result<Vec<usize>, ReducedSwirlVaccComponentError> {
    let events = transcript.events().get(phase.event_range.clone()).ok_or(
        ReducedSwirlVaccComponentError::Transcript {
            step,
            message: "commitment event range",
        },
    )?;
    // The backend absorbs, for every source: domain/version and six layout
    // scalars, then `(width, root)` for every original SWIRL root, followed by
    // theta. A digest contributes `DIGEST_SIZE` transcript events. Root count
    // is intentionally variable; assuming one root would authenticate the
    // wrong descriptor boundary for heterogeneous SWIRL layouts.
    let expected_events = claims.iter().try_fold(0usize, |total, claim| {
        total.checked_add(9 + claim.roots.len() * (1 + DIGEST_SIZE))
    });
    if expected_events != Some(events.len()) {
        return Err(ReducedSwirlVaccComponentError::Transcript {
            step,
            message: "commitment descriptor event count",
        });
    }
    let mut result = Vec::with_capacity(claims.len());
    let mut event_offset = 0usize;
    for claim in claims {
        let event = &events[event_offset];
        if event.operation_range.len() != D_EF {
            return Err(ReducedSwirlVaccComponentError::Transcript {
                step,
                message: "commitment descriptor event width",
            });
        }
        result.push(event.operation_range.start);
        event_offset += 9 + claim.roots.len() * (1 + DIGEST_SIZE);
    }
    debug_assert_eq!(event_offset, events.len());
    Ok(result)
}

/// A typed, constrained projection between the recursion-local chain receipt
/// and the continuations wrapper receipt.  Equal field layouts on different
/// bus indices are not interchangeable; this AIR is the unique producer on
/// the wrapper bus.
#[derive(Clone, Copy, Debug)]
pub struct ReducedSwirlVaccReceiptBridgeAir {
    pub chain_bus: ReducedSwirlVaccChainReceiptBus,
    pub wrapper_bus: ReducedSwirlVaccReceiptBus,
}

impl BaseAirWithPublicValues<F> for ReducedSwirlVaccReceiptBridgeAir {}
impl PartitionedBaseAir<F> for ReducedSwirlVaccReceiptBridgeAir {}

impl BaseAir<F> for ReducedSwirlVaccReceiptBridgeAir {
    fn width(&self) -> usize {
        RECEIPT_BRIDGE_WIDTH
    }
}

impl<AB> Air<AB> for ReducedSwirlVaccReceiptBridgeAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("reduced-SWIRL VACC receipt bridge row");
        let values = &row[1..];
        let active = row[0];
        builder.assert_one(active);

        let digest = |start: usize| core::array::from_fn(|limb| values[start + limb].into());
        let protocol_digest = digest(0);
        let relation_digest = digest(DIGEST_SIZE);
        let warp_index_digest = digest(2 * DIGEST_SIZE);
        let schedule_digest = digest(3 * DIGEST_SIZE);
        let manifest_digest = digest(4 * DIGEST_SIZE);
        let source_count = values[5 * DIGEST_SIZE].into();
        let call_count = values[5 * DIGEST_SIZE + 1].into();
        let final_accumulator_digest = digest(5 * DIGEST_SIZE + 2);
        let final_accumulator_root = digest(6 * DIGEST_SIZE + 2);

        self.chain_bus.lookup_key(
            builder,
            ReducedSwirlVaccChainReceiptMessage {
                protocol_digest: protocol_digest.clone(),
                relation_digest: relation_digest.clone(),
                warp_index_digest: warp_index_digest.clone(),
                schedule_digest: schedule_digest.clone(),
                manifest_digest: manifest_digest.clone(),
                source_count: source_count.clone(),
                call_count: call_count.clone(),
                final_accumulator_digest: final_accumulator_digest.clone(),
                final_accumulator_root: final_accumulator_root.clone(),
            },
            active,
        );
        self.wrapper_bus.add_key_with_lookups(
            builder,
            ReducedSwirlVaccReceiptMessage {
                protocol_digest,
                relation_digest,
                warp_index_digest,
                schedule_digest,
                manifest_digest,
                source_count,
                call_count,
                final_accumulator_digest,
                final_accumulator_root,
            },
            active,
        );
    }
}

/// Witness for [`ReducedSwirlVaccReceiptBridgeAir`].
pub fn generate_reduced_swirl_vacc_receipt_bridge_trace(
    receipt: &ReducedSwirlVaccChainReceiptMessage<F>,
) -> RowMajorMatrix<F> {
    let mut values = F::zero_vec(RECEIPT_BRIDGE_WIDTH);
    values[0] = F::ONE;
    let mut offset = 1usize;
    for digest in [
        receipt.protocol_digest,
        receipt.relation_digest,
        receipt.warp_index_digest,
        receipt.schedule_digest,
        receipt.manifest_digest,
    ] {
        values[offset..offset + DIGEST_SIZE].copy_from_slice(&digest);
        offset += DIGEST_SIZE;
    }
    values[offset] = receipt.source_count;
    offset += 1;
    values[offset] = receipt.call_count;
    offset += 1;
    for digest in [
        receipt.final_accumulator_digest,
        receipt.final_accumulator_root,
    ] {
        values[offset..offset + DIGEST_SIZE].copy_from_slice(&digest);
        offset += DIGEST_SIZE;
    }
    debug_assert_eq!(offset, RECEIPT_BRIDGE_WIDTH);
    RowMajorMatrix::new(values, RECEIPT_BRIDGE_WIDTH)
}

/// The already-complete recursion-local aggregate plus its mandatory receipt
/// bus bridge.  The ordinary v19 algebra/transcript modules are installed by
/// the production constructor below; keeping this small carrier private
/// prevents callers from mistaking the aggregate adapter for a complete WARP
/// verifier component.
struct ReducedSwirlVaccAggregateWithBridge {
    aggregate: ReducedSwirlVaccAggregateComponent,
    bridge: ReducedSwirlVaccReceiptBridgeAir,
}

impl ReducedSwirlVaccAggregateWithBridge {
    fn airs<C: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<C>> {
        self.aggregate
            .airs::<C>()
            .into_iter()
            .chain([Arc::new(self.bridge) as AirRef<C>])
            .collect()
    }

    fn contexts(
        &self,
        mut aggregate: Vec<RowMajorMatrix<F>>,
        receipt: &ReducedSwirlVaccChainReceiptMessage<F>,
    ) -> Result<Vec<AirProvingContext<CpuBackend<SC>>>, &'static str> {
        if aggregate.len() != self.aggregate.airs::<SC>().len() {
            return Err("reduced-SWIRL VACC aggregate trace inventory");
        }
        aggregate.push(generate_reduced_swirl_vacc_receipt_bridge_trace(receipt));
        if aggregate.len() != self.airs::<SC>().len() {
            return Err("reduced-SWIRL VACC bridge trace inventory");
        }
        Ok(aggregate
            .into_iter()
            .map(AirProvingContext::simple_no_pis)
            .collect())
    }
}

const REDUCED_SWIRL_VACC_SCHEDULE_DIGEST_TAG: &[u8] =
    b"openvm.native-warp.reduced-swirl.vacc-schedule.v2";
const REDUCED_SWIRL_WARP_INDEX_DIGEST_TAG: &[u8] =
    b"openvm.native-warp.reduced-swirl.warp-index.v1";
const REDUCED_SWIRL_VACC_COMPONENT_DIGEST_TAG: &[u8] =
    b"openvm.native-warp.reduced-swirl.vacc-component.v4";

fn detached_source_profile(
    native_setup: &ReducedSwirlNativeSetup,
    system_params: &SystemParams,
) -> Result<
    openvm_recursion_circuit::native_warp::ReducedSwirlSourceProfile,
    ReducedSwirlVaccComponentError,
> {
    let expected_binding = canonical_swirl_reduced_code_binding(
        native_setup.relation(),
        system_params.l_skip,
        system_params.n_stack,
        system_params.log_blowup,
        system_params.log_commit_rows_per_query,
    );
    if expected_binding != *native_setup.binding() {
        return Err(ReducedSwirlVaccComponentError::Setup(
            "detached VACC system parameters differ from native setup".to_owned(),
        ));
    }
    let maximum_openings_per_source = system_params.w_stack;
    let maximum_roots_per_source = maximum_openings_per_source.min(REDUCED_SWIRL_VACC_MAX_ROOTS);
    let source = openvm_recursion_circuit::native_warp::ReducedSwirlSourceProfile {
        maximum_sources: native_setup.maximum_source_count(),
        maximum_roots_per_source,
        maximum_openings_per_source,
        l_skip: system_params.l_skip,
        n_stack: system_params.n_stack,
        log_blowup: system_params.log_blowup,
        log_commit_rows_per_query: system_params.log_commit_rows_per_query,
    };
    source
        .validate()
        .map_err(ReducedSwirlVaccComponentError::Profile)?;
    Ok(source)
}

/// Complete production recursion component for reduced-SWIRL VACC.
///
/// There is one setup-fixed ordinary v19 verifier module for all calls. The
/// runtime bootstrap/continuation distinction, final partial fresh batch, and
/// inactive padding slots are rows constrained through typed schedule buses.
/// The two source-digest transcripts and the ordinary WARP transcript share
/// one physical Poseidon table owned here.
pub struct ProductionReducedSwirlVaccComponent {
    profile: ReducedSwirlVaccProfile,
    standard: DirectAirVaccVerifierModuleV19,
    digest_transcript: NativeWarpTranscriptModule,
    manifest_transcript: NativeWarpTranscriptModule,
    aggregate: ReducedSwirlVaccAggregateWithBridge,
    transition_aggregate: ReducedSwirlVaccTransitionAggregateComponent,
    shared: BusInventory,
    main_transcript_bus: TranscriptBus,
    footer_bus: ReducedSwirlVaccFooterBus,
    source_mode: ReducedSwirlVaccSourceMode,
    source_authority_bus:
        Option<openvm_recursion_circuit::native_warp::ReducedSwirlSourceAuthorityBus>,
    source_summary_bus: Option<ReducedSwirlVaccSourceSummaryBus>,
    source_summary_bus_index: Option<BusIndex>,
    next_bus_idx: BusIndex,
    component_digest: Digest,
    config: SC,
}

#[derive(Clone, Copy)]
struct InlineReducedSwirlSourceBuses {
    claim: openvm_recursion_circuit::native_warp::ReducedSwirlSourceClaimBus,
    root: openvm_recursion_circuit::native_warp::ReducedSwirlSourceRootBus,
    beta: openvm_recursion_circuit::native_warp::ReducedSwirlSourceBetaBus,
    authority: openvm_recursion_circuit::native_warp::ReducedSwirlSourceAuthorityBus,
}

/// CPU witness packet in exactly the AIR order returned by
/// [`ProductionReducedSwirlVaccComponent::airs`]. The Poseidon requests are
/// retained so a terminal component using the same shared buses can merge its
/// requests before the unique Poseidon table is finalized.
pub struct ReducedSwirlVaccCpuPacket {
    pub contexts: Vec<AirProvingContext<CpuBackend<SC>>>,
    pub receipt: ReducedSwirlVaccChainReceiptMessage<F>,
    pub entry_digests: Vec<Digest>,
    pub manifest_digest: Digest,
    /// Detached-only authenticated statement for the source-tree tail bridge.
    /// The component intentionally supplies no sink for its typed lookup bus.
    pub source_summary: Option<ReducedSwirlVaccSourceSummaryMessage<F>>,
    pub poseidon_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub poseidon_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
}

/// Bounded witness packet for exactly one canonical global WARP transition.
/// It excludes the whole-block footer and terminal suffix, and therefore can
/// be concatenated with one bounded inline source-verifier packet in a fixed
/// recursive leaf.
pub struct ReducedSwirlVaccTransitionCpuPacket {
    pub contexts: Vec<AirProvingContext<CpuBackend<SC>>>,
    pub receipt: ReducedSwirlVaccTransitionReceiptMessage<F>,
    pub entry_digests: Vec<Digest>,
    pub manifest_digest: Digest,
    pub poseidon_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub poseidon_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
}

impl ProductionReducedSwirlVaccComponent {
    /// Deterministically instantiate the complete VACC component from the
    /// exact native fixed setup and the already-keyed source component.
    /// Caller-selected relation/protocol/index digests are not accepted.
    #[allow(clippy::too_many_arguments)]
    pub fn from_native_setup(
        native_setup: &ReducedSwirlNativeSetup,
        system_params: SystemParams,
        source_component: &ProductionReducedSwirlSourceReceiptComponent,
        wrapper_receipt_bus: ReducedSwirlVaccReceiptBus,
        bus_idx_manager: BusIndexManager,
    ) -> Result<Self, ReducedSwirlVaccComponentError> {
        let source_inner = source_component.inner();
        if bus_idx_manager.next_bus_idx() < source_inner.next_bus_idx()
            || wrapper_receipt_bus.index() >= bus_idx_manager.next_bus_idx()
        {
            return Err(ReducedSwirlVaccComponentError::Setup(
                "VACC buses do not follow the source component namespace".to_owned(),
            ));
        }
        let source_air = source_inner.source_air();
        Self::from_native_setup_with_source_mode(
            native_setup,
            system_params,
            source_inner.receipt_air().profile.source,
            ReducedSwirlVaccSourceMode::Inline,
            Some(InlineReducedSwirlSourceBuses {
                claim: source_air.claim_export_bus,
                root: source_air.root_export_bus,
                beta: source_air.beta_export_bus,
                authority: source_inner.receipt_air().authority_bus,
            }),
            wrapper_receipt_bus,
            bus_idx_manager,
        )
    }

    /// Construct the VACC side of a fixed-capacity combined transition leaf.
    /// The source component may be much smaller than the block-wide native
    /// source bound (production uses capacity eight); only relation/code
    /// dimensions and its inline export buses are reused here.
    pub fn from_native_setup_for_transition_leaf<const CAPACITY: usize>(
        native_setup: &ReducedSwirlNativeSetup,
        system_params: SystemParams,
        source_component: &ProductionReducedSwirlSourceReceiptComponent<CAPACITY>,
        mut bus_idx_manager: BusIndexManager,
    ) -> Result<Self, ReducedSwirlVaccComponentError> {
        let source_inner = source_component.inner();
        let warp_arity = native_setup.security().params.input_arity;
        if CAPACITY == 0
            || CAPACITY > warp_arity
            || source_inner.receipt_air().profile.source.maximum_sources != CAPACITY
            || bus_idx_manager.next_bus_idx() < source_inner.next_bus_idx()
        {
            return Err(ReducedSwirlVaccComponentError::Setup(
                "transition-leaf source capacity/namespace".to_owned(),
            ));
        }
        let source_air = source_inner.source_air();
        // The whole-block bridge is not part of `transition_airs`, but the
        // common constructor requires a private, setup-fixed bus for that
        // dormant inventory. It has no interactions in a transition proof.
        let dormant_wrapper_receipt =
            ReducedSwirlVaccReceiptBus::new(bus_idx_manager.new_bus_idx());
        let mut source_profile = source_inner.receipt_air().profile.source;
        source_profile.maximum_sources = native_setup.maximum_source_count();
        Self::from_native_setup_with_source_mode(
            native_setup,
            system_params,
            source_profile,
            ReducedSwirlVaccSourceMode::Inline,
            Some(InlineReducedSwirlSourceBuses {
                claim: source_air.claim_export_bus,
                root: source_air.root_export_bus,
                beta: source_air.beta_export_bus,
                authority: source_inner.receipt_air().authority_bus,
            }),
            dormant_wrapper_receipt,
            bus_idx_manager,
        )
    }

    /// Instantiate a setup-fixed VACC component whose source receipt is proven
    /// in a separate certificate MultiSTARK.
    ///
    /// There are no source-receipt bus interactions in this mode. The ordinary
    /// WARP verifier still authenticates every root and claim coordinate, and
    /// this component still recomputes the canonical ordered manifest emitted
    /// in its VACC receipt. A tail wrapper must constrain that receipt manifest
    /// equal to the independently verified source-certificate receipt.
    pub fn from_native_setup_detached(
        native_setup: &ReducedSwirlNativeSetup,
        system_params: SystemParams,
        wrapper_receipt_bus: ReducedSwirlVaccReceiptBus,
        bus_idx_manager: BusIndexManager,
    ) -> Result<Self, ReducedSwirlVaccComponentError> {
        if wrapper_receipt_bus.index() >= bus_idx_manager.next_bus_idx() {
            return Err(ReducedSwirlVaccComponentError::Setup(
                "detached VACC receipt bus is outside the existing namespace".to_owned(),
            ));
        }
        let source = detached_source_profile(native_setup, &system_params)?;
        Self::from_native_setup_with_source_mode(
            native_setup,
            system_params,
            source,
            ReducedSwirlVaccSourceMode::Detached,
            None,
            wrapper_receipt_bus,
            bus_idx_manager,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_native_setup_with_source_mode(
        native_setup: &ReducedSwirlNativeSetup,
        system_params: SystemParams,
        source: openvm_recursion_circuit::native_warp::ReducedSwirlSourceProfile,
        source_mode: ReducedSwirlVaccSourceMode,
        inline_source_buses: Option<InlineReducedSwirlSourceBuses>,
        wrapper_receipt_bus: ReducedSwirlVaccReceiptBus,
        mut bus_idx_manager: BusIndexManager,
    ) -> Result<Self, ReducedSwirlVaccComponentError> {
        if source_mode == ReducedSwirlVaccSourceMode::Inline && inline_source_buses.is_none()
            || source_mode == ReducedSwirlVaccSourceMode::Detached && inline_source_buses.is_some()
        {
            return Err(ReducedSwirlVaccComponentError::Setup(
                "reduced-SWIRL VACC source-linkage mode".to_owned(),
            ));
        }
        let warp = native_setup.security().params;
        let terminal_setup = ReducedSwirlTerminalProductionSetup::from_native_fixed_params(
            system_params.clone(),
            warp.input_arity,
            native_setup.family_target_bits(),
            native_setup.maximum_source_count(),
        )
        .map_err(|error| ReducedSwirlVaccComponentError::Setup(format!("{error:?}")))?;
        let relation_shape =
            <_ as ReducedConstrainedCodeRelation<F, EF>>::shape(native_setup.relation());
        let external_protocol_binding =
            canonical_reduced_constrained_code_protocol_binding::<F, EF, _, _>(
                native_setup.relation(),
                native_setup.binding(),
                native_setup.code(),
            )
            .map_err(|error| ReducedSwirlVaccComponentError::Setup(format!("{error:?}")))?;
        if source.maximum_sources < native_setup.maximum_source_count()
            || source.l_skip != system_params.l_skip
            || source.n_stack != system_params.n_stack
            || source.log_blowup != system_params.log_blowup
            || source.log_commit_rows_per_query != system_params.log_commit_rows_per_query
            || source.log_message_len() != relation_shape.log_message_len
            // Fresh constrained-RS claims use the SWIRL opening point.  WARP
            // appends the normalized target only to accumulator beta.
            || relation_shape.beta_len != source.log_message_len()
        {
            return Err(ReducedSwirlVaccComponentError::Setup(
                "native/source relation dimensions differ".to_owned(),
            ));
        }
        let protocol_digest = terminal_setup.protocol_digest();
        let relation_digest = terminal_setup.relation_digest();
        let schedule_digest =
            reduced_swirl_schedule_digest(warp.input_arity, native_setup.maximum_source_count())?;
        let warp_index_digest = derive_reduced_swirl_warp_index_digest(
            native_setup,
            &system_params,
            protocol_digest,
            relation_digest,
            &external_protocol_binding,
        )?;
        let profile = ReducedSwirlVaccProfile {
            maximum_sources: native_setup.maximum_source_count(),
            input_arity: warp.input_arity,
            num_ood: warp.num_ood,
            num_shift_queries: warp.num_shift_queries,
            batching_arity: warp.batching_arity,
            family_target_bits: native_setup.family_target_bits(),
            source,
            source_domain: native_setup.binding().source_domain.clone(),
            relation_binding: native_setup.binding().relation_binding.clone(),
            code_binding: native_setup.binding().code_binding.clone(),
            external_protocol_binding,
            protocol_digest,
            relation_digest,
            warp_index_digest,
            schedule_digest,
        };
        profile
            .validate()
            .map_err(ReducedSwirlVaccComponentError::Profile)?;
        // Keep this comparison at the completed profile boundary.  The
        // terminal setup is the single fixed-material authority for these two
        // identities; profile construction must never acquire an independent
        // digest formula that can silently drift from terminal Decide.
        if profile.protocol_digest != terminal_setup.protocol_digest()
            || profile.relation_digest != terminal_setup.relation_digest()
        {
            return Err(ReducedSwirlVaccComponentError::Setup(
                "native/terminal fixed digest mismatch".to_owned(),
            ));
        }

        let shared = BusInventory::new(&mut bus_idx_manager);
        let native_buses = NativeWarpPcdBusInventory::new(bus_idx_manager.next_bus_idx());
        let mut internal = BusIndexManager::from_next_bus_idx(native_buses.next_bus_idx());
        // DirectAirVaccVerifierModuleV19 reserves this exact first free index
        // for its setup-fixed prefix-event catalog.
        let _prefix_event_bus = internal.new_bus_idx();
        let history_buses = DirectAirVaccHistoryBusesV19 {
            context: DirectAirVaccContextBusV19::new(internal.new_bus_idx()),
            input: CertifiedDirectAirVaccInputBusV19::new(internal.new_bus_idx()),
        };
        let protocol_bus = NativeStandardVaccProtocolBus::new(internal.new_bus_idx());
        let end_bus = NativeStandardVaccEndBus::new(internal.new_bus_idx());
        let root_bus = NativeStandardVaccRootBus::new(internal.new_bus_idx());
        let digest_bus = NativeStandardVaccDigestBus::new(internal.new_bus_idx());

        let standard_profile = NativeStandardVaccProfile::from_reduced_constrained_code(
            native_setup.binding().source_domain.clone(),
            relation_digest,
            warp,
            profile.source.log_message_len(),
            profile.source.log_codeword_len(),
            0,
            0,
            profile.normalized_beta_len(),
            relation_shape.max_constraint_degree,
            profile.source.rows_per_query(),
        )
        .map_err(|error| ReducedSwirlVaccComponentError::Setup(error.to_owned()))?;
        if standard_profile.beta_len != relation_shape.beta_len + 1 {
            return Err(ReducedSwirlVaccComponentError::Setup(
                "fresh/normalized reduced beta dimensions differ".to_owned(),
            ));
        }
        let projection_height = profile
            .source
            .maximum_openings_per_source
            .checked_mul(profile.num_shift_queries)
            .filter(|value| *value != 0)
            .ok_or_else(|| {
                ReducedSwirlVaccComponentError::Setup(
                    "reduced direct-opening projection capacity".to_owned(),
                )
            })?
            .next_power_of_two();
        let reduced_config = DirectAirReducedSwirlVaccConfigV19 {
            input_arity: profile.input_arity,
            max_roots_per_source: profile.source.maximum_roots_per_source,
            projection_sources_per_shard: 1,
            max_projection_height: projection_height,
        };
        let standard = DirectAirVaccVerifierModuleV19::new_reduced_swirl_batched(
            standard_profile,
            reduced_config,
            shared.clone(),
            native_buses.clone(),
            history_buses,
            protocol_bus,
            end_bus,
            root_bus,
            digest_bus,
            system_params.clone(),
        )
        .map_err(|error| ReducedSwirlVaccComponentError::Setup(format!("{error:?}")))?;

        let digest_transcript_bus = TranscriptBus::new(internal.new_bus_idx());
        let manifest_transcript_bus = TranscriptBus::new(internal.new_bus_idx());
        let source_authority_bus = inline_source_buses.map(|buses| buses.authority);
        let (
            detached_source_bus,
            summary_manifest_bus,
            source_summary_bus,
            source_summary_bus_index,
        ) = if source_mode == ReducedSwirlVaccSourceMode::Detached {
            let detached_source_bus_index = internal.new_bus_idx();
            let summary_manifest_bus_index = internal.new_bus_idx();
            let source_summary_bus_index = internal.new_bus_idx();
            (
                Some(ReducedSwirlVaccDetachedSourceBus::new(
                    detached_source_bus_index,
                )),
                Some(ReducedSwirlVaccSummaryManifestBus::new(
                    summary_manifest_bus_index,
                )),
                Some(ReducedSwirlVaccSourceSummaryBus::new(
                    source_summary_bus_index,
                )),
                Some(source_summary_bus_index),
            )
        } else {
            (None, None, None, None)
        };
        let aggregate_buses = ReducedSwirlVaccAggregateBuses {
            main_transcript: native_buses.transcript,
            digest_transcript: digest_transcript_bus,
            manifest_transcript: manifest_transcript_bus,
            phase_cursor: native_buses.vacc_phase_cursor,
            certified_checkpoint: native_buses.transcript_checkpoint,
            resume_state: shared.resume_state_bus,
            transcript_end_index: shared.transcript_end_index_bus,
            standard_vacc_end: end_bus,
            standard_vacc_root: root_bus,
            standard_vacc_digest: digest_bus,
            direct_fresh_source: native_buses.direct_fresh_source,
            direct_fresh_root: native_buses.direct_fresh_root,
            vacc_claim: native_buses.claim_value,
            input_slot_layout: native_buses.input_slot_layout,
            fresh_count: native_buses.fresh_count,
            source_claim: inline_source_buses.map(|buses| buses.claim),
            source_root: inline_source_buses.map(|buses| buses.root),
            source_beta: inline_source_buses.map(|buses| buses.beta),
            source_authority: source_authority_bus,
            detached_source: detached_source_bus,
            summary_manifest: summary_manifest_bus,
            source_summary: source_summary_bus,
            poseidon_compress: (source_mode == ReducedSwirlVaccSourceMode::Detached)
                .then_some(shared.poseidon2_compress_bus),
            call: ReducedSwirlVaccCallBus::new(internal.new_bus_idx()),
            source_slot: ReducedSwirlVaccSourceSlotBus::new(internal.new_bus_idx()),
            source_entry_digest: ReducedSwirlSourceEntryDigestBus::new(internal.new_bus_idx()),
            header_end: ReducedSwirlVaccHeaderEndBus::new(internal.new_bus_idx()),
            manifest_digest: ReducedSwirlManifestDigestBus::new(internal.new_bus_idx()),
            chain_end: ReducedSwirlVaccChainEndBus::new(internal.new_bus_idx()),
            footer: ReducedSwirlVaccFooterBus::new(internal.new_bus_idx()),
            receipt: ReducedSwirlVaccChainReceiptBus::new(internal.new_bus_idx()),
        };
        let transition_end_bus = ReducedSwirlVaccTransitionEndBus::new(internal.new_bus_idx());
        let transition_receipt_bus =
            ReducedSwirlVaccTransitionReceiptBus::new(internal.new_bus_idx());
        let aggregate = ReducedSwirlVaccAggregateWithBridge {
            aggregate: ReducedSwirlVaccAggregateComponent {
                profile: profile.clone(),
                source_mode,
                buses: aggregate_buses,
                root_tree_stride: 1 + profile.num_shift_queries,
            },
            bridge: ReducedSwirlVaccReceiptBridgeAir {
                chain_bus: aggregate_buses.receipt,
                wrapper_bus: wrapper_receipt_bus,
            },
        };
        let transition_aggregate = ReducedSwirlVaccTransitionAggregateComponent {
            profile: profile.clone(),
            buses: aggregate_buses,
            root_tree_stride: 1 + profile.num_shift_queries,
            transition_end_bus,
            transition_receipt_bus,
        };
        let digest_transcript = NativeWarpTranscriptModule::new_for_bus(
            &shared,
            digest_transcript_bus,
            system_params.clone(),
        );
        let manifest_transcript = NativeWarpTranscriptModule::new_for_bus(
            &shared,
            manifest_transcript_bus,
            system_params.clone(),
        );
        let next_bus_idx = internal.next_bus_idx();
        let component_digest = reduced_swirl_vacc_component_digest(
            &profile,
            &shared,
            source_mode,
            native_buses.transcript,
            aggregate_buses.footer,
            source_authority_bus,
            source_summary_bus_index,
            wrapper_receipt_bus,
            next_bus_idx,
            standard.airs_without_poseidon::<SC>().len(),
            aggregate.airs::<SC>().len(),
        );
        if component_digest.iter().all(|value| *value == F::ZERO) {
            return Err(ReducedSwirlVaccComponentError::Setup(
                "zero reduced-SWIRL VACC component digest".to_owned(),
            ));
        }
        Ok(Self {
            profile,
            standard,
            digest_transcript,
            manifest_transcript,
            aggregate,
            transition_aggregate,
            shared,
            main_transcript_bus: native_buses.transcript,
            footer_bus: aggregate_buses.footer,
            source_mode,
            source_authority_bus,
            source_summary_bus,
            source_summary_bus_index,
            next_bus_idx,
            component_digest,
            config: SC::default_from_params(system_params),
        })
    }

    #[must_use]
    pub const fn profile(&self) -> &ReducedSwirlVaccProfile {
        &self.profile
    }

    #[must_use]
    pub const fn shared_bus_inventory(&self) -> &BusInventory {
        &self.shared
    }

    #[must_use]
    pub const fn main_transcript_bus(&self) -> TranscriptBus {
        self.main_transcript_bus
    }

    #[must_use]
    pub const fn footer_bus(&self) -> ReducedSwirlVaccFooterBus {
        self.footer_bus
    }

    #[must_use]
    pub const fn source_mode(&self) -> ReducedSwirlVaccSourceMode {
        self.source_mode
    }

    #[must_use]
    pub const fn source_authority_bus(
        &self,
    ) -> openvm_recursion_circuit::native_warp::ReducedSwirlSourceAuthorityBus {
        match self.source_authority_bus {
            Some(bus) => bus,
            None => panic!("detached reduced-SWIRL VACC has no source-authority bus"),
        }
    }

    #[must_use]
    pub const fn optional_source_authority_bus(
        &self,
    ) -> Option<openvm_recursion_circuit::native_warp::ReducedSwirlSourceAuthorityBus> {
        self.source_authority_bus
    }

    /// Detached-mode tail-link seam. A source-tree bridge must receive the
    /// exact [`ReducedSwirlVaccSourceSummaryMessage`] on this bus. Inline mode
    /// returns `None` and retains its existing source-authority linkage.
    #[must_use]
    pub const fn source_summary_bus(&self) -> Option<ReducedSwirlVaccSourceSummaryBus> {
        self.source_summary_bus
    }

    #[must_use]
    pub const fn source_summary_bus_index(&self) -> Option<BusIndex> {
        self.source_summary_bus_index
    }

    #[must_use]
    pub const fn next_bus_idx(&self) -> BusIndex {
        self.next_bus_idx
    }

    #[must_use]
    pub const fn protocol_digest(&self) -> Digest {
        self.component_digest
    }

    /// Domain-separated identity of the ordinary reduced-SWIRL WARP index.
    /// This is intentionally distinct from terminal WHIR's index digest.
    #[must_use]
    pub const fn warp_index_digest(&self) -> Digest {
        self.profile.warp_index_digest
    }

    #[must_use]
    pub fn air_counts(&self) -> ReducedSwirlVaccAirCounts {
        ReducedSwirlVaccAirCounts {
            standard_without_poseidon: self.standard.airs_without_poseidon::<SC>().len(),
            digest_transcripts: 2,
            aggregate_and_bridge: self.aggregate.airs::<SC>().len(),
            poseidon: 1,
        }
    }

    /// AIR inventory for one bounded transition leaf.  Unlike [`Self::airs`],
    /// this contains neither the whole-block footer nor its wrapper-receipt
    /// bridge.  All dynamic call/source coordinates remain constrained by the
    /// one-call aggregate and its typed transition receipt.
    #[must_use]
    pub fn transition_airs<C: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<C>> {
        let mut airs = self.standard.airs_without_poseidon::<C>();
        airs.push(self.digest_transcript.airs::<C>()[0].clone());
        airs.push(self.manifest_transcript.airs::<C>()[0].clone());
        airs.extend(self.transition_aggregate.airs::<C>());
        airs.push(self.standard.airs::<C>()[1].clone());
        airs
    }

    #[must_use]
    pub const fn transition_receipt_bus(&self) -> ReducedSwirlVaccTransitionReceiptBus {
        self.transition_aggregate.transition_receipt_bus
    }

    /// Generate the bounded recursion witness for one already-verified native
    /// WARP call.  The caller builds `aggregate_record` once and can then drop
    /// each returned packet after proving its recursive leaf.
    pub fn generate_transition_cpu_packet(
        &self,
        output: &ReducedSwirlNativeProverOutput,
        verification: &ReducedSwirlNativeVerification,
        aggregate_record: &ReducedSwirlVaccAggregateRecord,
        call_index: usize,
    ) -> Result<ReducedSwirlVaccTransitionCpuPacket, ReducedSwirlVaccComponentError> {
        let call = aggregate_record
            .calls
            .get(call_index)
            .ok_or(ReducedSwirlVaccComponentError::TransitionInventory)?;
        if call.call.step != call_index
            || output.proof.vacc.steps.len() != aggregate_record.calls.len()
            || verification.transition_records.len() != aggregate_record.calls.len()
        {
            return Err(ReducedSwirlVaccComponentError::TransitionInventory);
        }
        let source_start = call.call.source_start;
        let source_end = source_start
            .checked_add(call.call.fresh_count)
            .ok_or(ReducedSwirlVaccComponentError::TransitionInventory)?;
        let sources = aggregate_record
            .sources
            .get(source_start..source_end)
            .ok_or(ReducedSwirlVaccComponentError::TransitionInventory)?;
        let transcript = TranscriptHistory::into_log(verification.complete_transcript.clone());
        self.generate_transition_cpu_packet_from_parts(
            aggregate_record.sources.len(),
            call,
            sources,
            &output.proof.vacc.steps[call_index],
            &verification.transition_records[call_index],
            (call_index != 0)
                .then(|| &verification.transition_records[call_index - 1].output_instance),
            &transcript,
        )
    }

    /// Generate exactly one recursion transition packet directly from a live
    /// CUDA stream's independently verified completed call.
    ///
    /// Unlike [`Self::generate_transition_cpu_packet`], this path neither
    /// clones the complete native transcript nor constructs a whole-block
    /// aggregate record. `source_block` must be the bounded Inline receipt for
    /// this call, so all temporary vectors are limited by the WARP input arity.
    #[cfg(feature = "cuda")]
    pub fn generate_transition_cpu_packet_from_completed_step(
        &self,
        completed: &ReducedSwirlNativeCudaCompletedStep<'_>,
        source_block: &ReducedSwirlSourceReceiptBlock,
    ) -> Result<ReducedSwirlVaccTransitionCpuPacket, ReducedSwirlVaccComponentError> {
        self.profile
            .validate()
            .map_err(ReducedSwirlVaccComponentError::Profile)?;
        if completed.total_source_count == 0
            || completed.total_source_count > self.profile.maximum_sources
        {
            return Err(ReducedSwirlVaccComponentError::SourceInventory);
        }
        let expected_calls =
            reduced_swirl_vacc_schedule(completed.total_source_count, self.profile.input_arity)
                .map_err(ReducedSwirlVaccComponentError::Profile)?;
        let call = *expected_calls
            .get(completed.call_index)
            .ok_or(ReducedSwirlVaccComponentError::TransitionInventory)?;
        if call.step != completed.call_index
            || call.source_start != completed.source_start
            || call.fresh_count != completed.fresh_count
            || completed.proof.fresh_count() != completed.fresh_count
            || usize::from(completed.prior_instance.is_some()) != call.prior_count
            || completed.authoritative_claims.len() != completed.fresh_count
            || completed.source_bindings.len() != completed.fresh_count
            || source_block.source_offset as usize != completed.source_start
            || source_block.sources.len() != completed.fresh_count
            || completed.transcript_prefix.len() != completed.end_checkpoint.operations
            || completed.batch_start_checkpoint.events > completed.vacc_start_checkpoint.events
            || completed.vacc_start_checkpoint.events > completed.end_checkpoint.events
            || completed.batch_start_checkpoint.permutations
                > completed.vacc_start_checkpoint.permutations
            || completed.vacc_start_checkpoint.permutations > completed.end_checkpoint.permutations
            || &completed.verification.output_instance != completed.output_instance
            || !same_recorded_transition(completed.prover_record, completed.verification)
        {
            return Err(ReducedSwirlVaccComponentError::TransitionInventory);
        }

        let claims = completed
            .authoritative_claims
            .iter()
            .map(recursive_claim_from_native)
            .collect::<Vec<_>>();
        validate_reduced_swirl_fresh_batch(
            &self.profile,
            &claims,
            completed.proof,
            completed.verification,
        )
        .map_err(|_| ReducedSwirlVaccComponentError::FreshAuthentication {
            step: completed.call_index,
        })?;

        let protocol = unique_phase(
            completed.call_index,
            &completed.verification.transcript_phases,
            &NativeTranscriptPhase::VaccProtocolPrefix,
        )?;
        let commitments = unique_phase(
            completed.call_index,
            &completed.verification.transcript_phases,
            &NativeTranscriptPhase::FreshCommitments,
        )?;
        let target = unique_phase(
            completed.call_index,
            &completed.verification.transcript_phases,
            &NativeTranscriptPhase::FinalTarget,
        )?;
        let boundary = unique_phase(
            completed.call_index,
            &completed.verification.transcript_phases,
            &NativeTranscriptPhase::ExactFiniteCallBoundary {
                call: completed.call_index.try_into().map_err(|_| {
                    ReducedSwirlVaccComponentError::Transcript {
                        step: completed.call_index,
                        message: "call-index width",
                    }
                })?,
            },
        )?;
        let expected_protocol_end = protocol
            .operation_range
            .start
            .checked_add(
                reduced_swirl_vacc_protocol_prefix_elements(&self.profile, call)
                    .map_err(ReducedSwirlVaccComponentError::Aggregate)?
                    .len()
                    .checked_mul(D_EF)
                    .ok_or(ReducedSwirlVaccComponentError::Aggregate(
                        "VACC protocol prefix width",
                    ))?,
            )
            .ok_or(ReducedSwirlVaccComponentError::Aggregate(
                "VACC protocol prefix boundary",
            ))?;
        if protocol.operation_range.start != completed.vacc_start_checkpoint.operations
            || protocol.event_range.start != completed.vacc_start_checkpoint.events
            || protocol.permutation_range.start != completed.vacc_start_checkpoint.permutations
            || protocol.operation_range.end != expected_protocol_end
            || commitments.operation_range.start != expected_protocol_end
            || target.operation_range.end != boundary.operation_range.start
            || boundary.operation_range.end != completed.end_checkpoint.operations
            || boundary.event_range.end != completed.end_checkpoint.events
            || boundary.permutation_range.end != completed.end_checkpoint.permutations
            || completed.vacc_start_checkpoint.operations
                <= completed.batch_start_checkpoint.operations
        {
            return Err(ReducedSwirlVaccComponentError::Transcript {
                step: completed.call_index,
                message: "non-canonical incremental call interval",
            });
        }

        let accumulator_layout = NativePrivateAccumulatorLayout::new(
            0,
            self.profile.log_codeword_len(),
            self.profile.normalized_beta_len(),
        );
        let output_digest = generate_native_accumulator_digest_traces(
            0,
            completed.output_instance,
            &accumulator_layout,
        )
        .ok_or(ReducedSwirlVaccComponentError::AccumulatorDigest {
            step: completed.call_index,
        })?
        .instance_digest;
        let prior_digest = completed
            .prior_instance
            .map(|prior| {
                generate_native_accumulator_digest_traces(0, prior, &accumulator_layout)
                    .ok_or(ReducedSwirlVaccComponentError::AccumulatorDigest {
                        step: completed.call_index,
                    })
                    .map(|traces| traces.instance_digest)
            })
            .transpose()?;
        let commitment_tidxs = commitment_descriptor_tidxs(
            completed.call_index,
            completed.transcript_prefix,
            commitments,
            &claims,
        )?;
        let tree_source_stride = self
            .profile
            .source
            .maximum_roots_per_source
            .checked_mul(1 + self.profile.num_shift_queries)
            .ok_or(ReducedSwirlVaccComponentError::Aggregate(
                "direct fresh tree-source stride",
            ))?;
        let mut sources = Vec::with_capacity(completed.fresh_count);
        for (slot, (((claim, receipt), &binding), commitment_tidx)) in claims
            .iter()
            .zip(&source_block.sources)
            .zip(completed.source_bindings)
            .zip(commitment_tidxs)
            .enumerate()
        {
            let authority = ReducedSwirlSourceAuthorityRecord {
                segment_index: receipt.segment_index,
                common_main_root: *receipt.roots.first().ok_or(
                    ReducedSwirlVaccComponentError::SourceBinding {
                        source_index: completed.source_start + slot,
                    },
                )?,
                trace_layout_digest: receipt.layout_digest,
                pending_claim_digest: receipt.pending_digest,
                checkpoint_tidx: receipt.checkpoint_tidx,
                checkpoint_state: receipt.checkpoint_state,
                program_commitment: receipt.vm.program_commitment,
                initial_pc: receipt.vm.initial_pc,
                initial_root: receipt.vm.initial_root,
                final_pc: receipt.vm.final_pc,
                final_root: receipt.vm.final_root,
                exit_code: receipt.vm.exit_code,
                is_terminate: receipt.vm.is_terminate,
            };
            let source_index = completed.source_start + slot;
            let entry_digest = reduced_swirl_source_entry_digest(
                &self.profile,
                source_index,
                call,
                slot,
                claim,
                &authority,
            )
            .map_err(|_| ReducedSwirlVaccComponentError::SourceBinding { source_index })?;
            if entry_digest != binding || receipt.entry_digest != binding {
                return Err(ReducedSwirlVaccComponentError::SourceBinding { source_index });
            }
            sources.push(ReducedSwirlSourceDigestRecord {
                claim: claim.clone(),
                authority,
                commitment_tidx,
                first_tree_id: slot
                    .checked_mul(tree_source_stride)
                    .and_then(|value| value.try_into().ok())
                    .ok_or(ReducedSwirlVaccComponentError::SourceBinding { source_index })?,
            });
        }
        let call = ReducedSwirlVaccCallRecord {
            call,
            batch_start_tidx: completed.batch_start_checkpoint.operations,
            vacc_start_tidx: completed.vacc_start_checkpoint.operations,
            vacc_end_tidx: completed.end_checkpoint.operations,
            start_sample_count: completed.start_sample_count,
            start_state: completed.start_state,
            end_sample_count: completed.end_sample_count,
            end_state: completed.end_state,
            prior_root: completed.prior_instance.map(|instance| instance.rt),
            output_root: completed.output_instance.rt,
            prior_digest,
            output_digest,
        };
        let packet = self.generate_transition_cpu_packet_from_parts(
            completed.total_source_count,
            &call,
            &sources,
            completed.proof,
            completed.verification,
            completed.prior_instance,
            completed.transcript_prefix,
        )?;
        if packet.manifest_digest != source_block.manifest_digest {
            return Err(ReducedSwirlVaccComponentError::SourceInventory);
        }
        Ok(packet)
    }

    #[allow(clippy::too_many_arguments)]
    fn generate_transition_cpu_packet_from_parts(
        &self,
        total_source_count: usize,
        call: &ReducedSwirlVaccCallRecord,
        sources: &[ReducedSwirlSourceDigestRecord],
        proof: &super::reduced_swirl_native::ReducedSwirlVaccStepProof,
        verification: &super::reduced_swirl_native::ReducedSwirlStepVerification,
        prior: Option<&openvm_stark_backend::warp_pesat::AccumulatorInstance<EF, Digest>>,
        transcript: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    ) -> Result<ReducedSwirlVaccTransitionCpuPacket, ReducedSwirlVaccComponentError> {
        let direct_record = DirectAirReducedSwirlVaccVerifierRecordV19 {
            proof_idx: call.call.step,
            local_proof_idx: 0,
            // Every transition leaf terminates at the VACC checkpoint.  The
            // finalizer owns the globally-final footer and Decide/WHIR suffix.
            is_final_call: false,
            proof,
            verification,
            transcript,
            prior,
            batch_start_tidx: call.batch_start_tidx,
            vacc_start_tidx: call.vacc_start_tidx,
            vacc_end_tidx: call.vacc_end_tidx,
            commitment_tidxs: sources
                .iter()
                .map(|source| source.commitment_tidx)
                .collect(),
        };
        let standard = self
            .standard
            .generate_reduced_swirl_traces_for_shared_poseidon(
                &self.config,
                core::slice::from_ref(&direct_record),
            )
            .map_err(|error| ReducedSwirlVaccComponentError::StandardVacc(format!("{error:?}")))?;
        let transition_record = ReducedSwirlVaccTransitionAggregateRecord {
            total_source_count,
            call: call.clone(),
            sources: sources.to_vec(),
        };
        let aggregate = self
            .transition_aggregate
            .generate_traces(&transition_record)
            .map_err(ReducedSwirlVaccComponentError::Aggregate)?;
        if standard.output_instances.as_slice()
            != core::slice::from_ref(&verification.output_instance)
            || standard.output_instance_digests.as_slice()
                != core::slice::from_ref(&call.output_digest)
        {
            return Err(ReducedSwirlVaccComponentError::TransitionInventory);
        }

        let digest_logs = aggregate.digest_transcript_logs.iter().collect::<Vec<_>>();
        let digest_transcript = self
            .digest_transcript
            .generate_trace_inputs_with_external(&digest_logs, Vec::new(), Vec::new(), None)
            .ok_or(ReducedSwirlVaccComponentError::TranscriptWitness(
                "bounded source digest transcript",
            ))?;
        let manifest_transcript = self
            .manifest_transcript
            .generate_trace_inputs_with_external(
                &[&aggregate.manifest_transcript_log],
                Vec::new(),
                Vec::new(),
                None,
            )
            .ok_or(ReducedSwirlVaccComponentError::TranscriptWitness(
                "bounded manifest transcript",
            ))?;
        let mut permutation_inputs = standard.poseidon_permutation_inputs;
        permutation_inputs.extend(digest_transcript.permutation_inputs);
        permutation_inputs.extend(manifest_transcript.permutation_inputs);
        let mut compression_inputs = standard.poseidon_compression_inputs;
        compression_inputs.extend(digest_transcript.compression_inputs);
        compression_inputs.extend(manifest_transcript.compression_inputs);
        let poseidon_trace = self
            .standard
            .transcript
            .build_poseidon2_trace(permutation_inputs.clone(), compression_inputs.clone(), None)
            .ok_or(ReducedSwirlVaccComponentError::TranscriptWitness(
                "bounded shared Poseidon table",
            ))?;
        let mut contexts = standard
            .traces
            .into_iter()
            .map(AirProvingContext::simple_no_pis)
            .collect::<Vec<_>>();
        contexts.push(AirProvingContext::simple_no_pis(digest_transcript.trace));
        contexts.push(AirProvingContext::simple_no_pis(manifest_transcript.trace));
        contexts.extend(
            aggregate
                .traces
                .into_iter()
                .map(AirProvingContext::simple_no_pis),
        );
        contexts.push(AirProvingContext::simple_no_pis(poseidon_trace));
        if contexts.len() != self.transition_airs::<SC>().len() {
            return Err(ReducedSwirlVaccComponentError::Inventory);
        }
        Ok(ReducedSwirlVaccTransitionCpuPacket {
            contexts,
            receipt: aggregate.receipt,
            entry_digests: aggregate.entry_digests,
            manifest_digest: aggregate.manifest_digest,
            poseidon_permutation_inputs: permutation_inputs,
            poseidon_compression_inputs: compression_inputs,
        })
    }

    /// Generate the complete CPU packet and finalize the shared Poseidon table
    /// with no external owner requests.
    pub fn generate_cpu_packet(
        &self,
        output: &ReducedSwirlNativeProverOutput,
        verification: &ReducedSwirlNativeVerification,
        source_block: &ReducedSwirlSourceReceiptBlock,
    ) -> Result<ReducedSwirlVaccCpuPacket, ReducedSwirlVaccComponentError> {
        self.generate_cpu_packet_with_shared_poseidon(
            output,
            verification,
            source_block,
            Vec::new(),
            Vec::new(),
        )
    }

    /// Generate the complete packet while merging requests from a terminal
    /// component configured with [`Self::shared_bus_inventory`]. Namespace
    /// equality is preserved because all requests land on this component's
    /// single setup-owned Poseidon buses.
    pub fn generate_cpu_packet_with_shared_poseidon(
        &self,
        output: &ReducedSwirlNativeProverOutput,
        verification: &ReducedSwirlNativeVerification,
        source_block: &ReducedSwirlSourceReceiptBlock,
        mut extra_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        mut extra_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    ) -> Result<ReducedSwirlVaccCpuPacket, ReducedSwirlVaccComponentError> {
        let aggregate_record = build_reduced_swirl_vacc_aggregate_record(
            &self.profile,
            output,
            verification,
            source_block,
        )?;
        let transcript = openvm_stark_backend::transcript::TranscriptHistory::into_log(
            verification.complete_transcript.clone(),
        );
        let mut source_offset = 0usize;
        let records = aggregate_record
            .calls
            .iter()
            .enumerate()
            .map(|(proof_idx, call)| {
                let source_end = source_offset + call.call.fresh_count;
                let commitment_tidxs = aggregate_record.sources[source_offset..source_end]
                    .iter()
                    .map(|source| source.commitment_tidx)
                    .collect::<Vec<_>>();
                source_offset = source_end;
                DirectAirReducedSwirlVaccVerifierRecordV19 {
                    proof_idx,
                    local_proof_idx: proof_idx,
                    is_final_call: proof_idx + 1 == aggregate_record.calls.len(),
                    proof: &output.proof.vacc.steps[proof_idx],
                    verification: &verification.transition_records[proof_idx],
                    transcript: &transcript,
                    prior: (proof_idx != 0)
                        .then(|| &verification.transition_records[proof_idx - 1].output_instance),
                    batch_start_tidx: call.batch_start_tidx,
                    vacc_start_tidx: call.vacc_start_tidx,
                    vacc_end_tidx: call.vacc_end_tidx,
                    commitment_tidxs,
                }
            })
            .collect::<Vec<_>>();
        if source_offset != aggregate_record.sources.len() {
            return Err(ReducedSwirlVaccComponentError::Inventory);
        }
        let standard = self
            .standard
            .generate_reduced_swirl_traces_for_shared_poseidon(&self.config, &records)
            .map_err(|error| ReducedSwirlVaccComponentError::StandardVacc(format!("{error:?}")))?;
        let aggregate = self
            .aggregate
            .aggregate
            .generate_traces(&aggregate_record)
            .map_err(ReducedSwirlVaccComponentError::Aggregate)?;
        if standard.output_instances.len() != aggregate_record.calls.len()
            || standard.output_instance_digests.len() != aggregate_record.calls.len()
            || standard
                .output_instances
                .iter()
                .zip(&aggregate_record.calls)
                .any(|(instance, call)| instance.rt != call.output_root)
            || standard
                .output_instance_digests
                .iter()
                .zip(&aggregate_record.calls)
                .any(|(digest, call)| *digest != call.output_digest)
        {
            return Err(ReducedSwirlVaccComponentError::TransitionInventory);
        }
        let digest_logs = aggregate.digest_transcript_logs.iter().collect::<Vec<_>>();
        let digest_transcript = self
            .digest_transcript
            .generate_trace_inputs_with_external(&digest_logs, Vec::new(), Vec::new(), None)
            .ok_or(ReducedSwirlVaccComponentError::TranscriptWitness(
                "source digest transcript",
            ))?;
        let mut manifest_logs =
            Vec::with_capacity(1 + aggregate.source_summary_manifest_logs.len());
        manifest_logs.push(&aggregate.manifest_transcript_log);
        manifest_logs.extend(aggregate.source_summary_manifest_logs.iter());
        let manifest_transcript = self
            .manifest_transcript
            .generate_trace_inputs_with_external(&manifest_logs, Vec::new(), Vec::new(), None)
            .ok_or(ReducedSwirlVaccComponentError::TranscriptWitness(
                "manifest digest transcript",
            ))?;

        let mut permutation_inputs = standard.poseidon_permutation_inputs;
        permutation_inputs.extend(digest_transcript.permutation_inputs);
        permutation_inputs.extend(manifest_transcript.permutation_inputs);
        permutation_inputs.append(&mut extra_permutation_inputs);
        let mut compression_inputs = standard.poseidon_compression_inputs;
        compression_inputs.extend(digest_transcript.compression_inputs);
        compression_inputs.extend(manifest_transcript.compression_inputs);
        compression_inputs.extend(aggregate.source_summary_compression_inputs.iter().copied());
        compression_inputs.append(&mut extra_compression_inputs);
        let poseidon_trace = self
            .standard
            .transcript
            .build_poseidon2_trace(permutation_inputs.clone(), compression_inputs.clone(), None)
            .ok_or(ReducedSwirlVaccComponentError::TranscriptWitness(
                "shared Poseidon table",
            ))?;

        let mut contexts = standard
            .traces
            .into_iter()
            .map(AirProvingContext::simple_no_pis)
            .collect::<Vec<_>>();
        contexts.push(AirProvingContext::simple_no_pis(digest_transcript.trace));
        contexts.push(AirProvingContext::simple_no_pis(manifest_transcript.trace));
        contexts.extend(
            self.aggregate
                .contexts(aggregate.traces, &aggregate.receipt)
                .map_err(ReducedSwirlVaccComponentError::Aggregate)?,
        );
        contexts.push(AirProvingContext::simple_no_pis(poseidon_trace));
        if contexts.len() != self.airs::<SC>().len() {
            return Err(ReducedSwirlVaccComponentError::Inventory);
        }
        Ok(ReducedSwirlVaccCpuPacket {
            contexts,
            receipt: aggregate.receipt,
            entry_digests: aggregate.entry_digests,
            manifest_digest: aggregate.manifest_digest,
            source_summary: aggregate.source_summary,
            poseidon_permutation_inputs: permutation_inputs,
            poseidon_compression_inputs: compression_inputs,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReducedSwirlVaccAirCounts {
    pub standard_without_poseidon: usize,
    pub digest_transcripts: usize,
    pub aggregate_and_bridge: usize,
    pub poseidon: usize,
}

impl ReducedSwirlVerifierComponent for ProductionReducedSwirlVaccComponent {
    fn protocol_digest(&self) -> Digest {
        self.component_digest
    }

    fn airs<C: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<C>> {
        let mut airs = self.standard.airs_without_poseidon::<C>();
        airs.push(self.digest_transcript.airs::<C>()[0].clone());
        airs.push(self.manifest_transcript.airs::<C>()[0].clone());
        airs.extend(self.aggregate.airs::<C>());
        airs.push(self.standard.airs::<C>()[1].clone());
        airs
    }
}

fn reduced_swirl_schedule_digest(
    input_arity: usize,
    maximum_sources: usize,
) -> Result<Digest, ReducedSwirlVaccComponentError> {
    let counts = reduced_swirl_vacc_schedule(maximum_sources, input_arity)
        .map_err(ReducedSwirlVaccComponentError::Profile)?;
    let mut transcript = default_duplex_sponge_recorder();
    observe_component_bytes(&mut transcript, REDUCED_SWIRL_VACC_SCHEDULE_DIGEST_TAG);
    for value in [input_arity, maximum_sources, counts.len()] {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
            &mut transcript,
            F::from_usize(value),
        );
    }
    for call in counts {
        for value in [
            call.step,
            call.source_start,
            call.fresh_count,
            call.prior_count,
            call.input_arity,
        ] {
            <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
                &mut transcript,
                F::from_usize(value),
            );
        }
    }
    Ok(core::array::from_fn(|_| {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::sample(&mut transcript)
    }))
}

/// Derive the reduced-SWIRL WARP index identity from exactly the fixed native
/// relation, code, schedule, and proven-security material. Terminal WHIR's
/// index is deliberately not an input or a substitute for this identity.
pub fn derive_reduced_swirl_warp_index_digest(
    native_setup: &ReducedSwirlNativeSetup,
    system_params: &SystemParams,
    protocol_digest: Digest,
    relation_digest: Digest,
    external_protocol_binding: &[EF],
) -> Result<Digest, ReducedSwirlVaccComponentError> {
    let security = native_setup.security();
    let params = security.params;
    let relation_shape =
        <_ as ReducedConstrainedCodeRelation<F, EF>>::shape(native_setup.relation());
    let calls =
        reduced_swirl_vacc_schedule(native_setup.maximum_source_count(), params.input_arity)
            .map_err(ReducedSwirlVaccComponentError::Profile)?;
    let mut transcript = default_duplex_sponge_recorder();
    observe_component_bytes(&mut transcript, REDUCED_SWIRL_WARP_INDEX_DIGEST_TAG);
    for digest in [protocol_digest, relation_digest] {
        for limb in digest {
            <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(&mut transcript, limb);
        }
    }
    observe_component_u64(
        &mut transcript,
        u64::from(native_setup.binding().protocol_version),
    );
    observe_component_bytes(&mut transcript, &native_setup.binding().source_domain);
    for values in [
        native_setup.binding().relation_binding.as_slice(),
        native_setup.binding().code_binding.as_slice(),
        external_protocol_binding,
    ] {
        observe_component_u64(&mut transcript, values.len() as u64);
        for value in values {
            for &limb in value.as_basis_coefficients_slice() {
                <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
                    &mut transcript,
                    limb,
                );
            }
        }
    }
    for value in [
        params.input_arity,
        params.num_ood,
        params.num_shift_queries,
        params.batching_arity,
        native_setup.maximum_source_count(),
        calls.len(),
        native_setup.family_target_bits(),
        security.target_bits,
        security.interleaving_width,
        relation_shape.log_message_len,
        relation_shape.beta_len,
        relation_shape.max_constraint_degree,
        system_params.l_skip,
        system_params.n_stack,
        system_params.log_blowup,
        system_params.log_commit_rows_per_query,
        1usize << system_params.log_commit_rows_per_query,
    ] {
        observe_component_u64(&mut transcript, value as u64);
    }
    <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
        &mut transcript,
        F::from_bool(security.rs_specific_proximity_bound),
    );
    for value in [
        security.code_rate,
        security.relative_distance_lower_bound,
        security.proximity_radius,
        security.proximity_bits,
        security.field_security_bits,
        security.round_by_round_bits,
        security.required_field_bits,
        security.available_field_bits,
    ] {
        observe_component_u64(&mut transcript, value.to_bits());
    }
    for call in calls {
        for value in [
            call.step,
            call.source_start,
            call.fresh_count,
            call.prior_count,
            call.input_arity,
        ] {
            observe_component_u64(&mut transcript, value as u64);
        }
    }
    Ok(core::array::from_fn(|_| {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::sample(&mut transcript)
    }))
}

#[allow(clippy::too_many_arguments)]
fn reduced_swirl_vacc_component_digest(
    profile: &ReducedSwirlVaccProfile,
    shared: &BusInventory,
    source_mode: ReducedSwirlVaccSourceMode,
    main_transcript: TranscriptBus,
    footer: ReducedSwirlVaccFooterBus,
    source_authority: Option<openvm_recursion_circuit::native_warp::ReducedSwirlSourceAuthorityBus>,
    source_summary_bus_index: Option<BusIndex>,
    wrapper_receipt: ReducedSwirlVaccReceiptBus,
    next_bus_idx: BusIndex,
    standard_air_count: usize,
    aggregate_air_count: usize,
) -> Digest {
    let mut transcript = default_duplex_sponge_recorder();
    observe_component_bytes(&mut transcript, REDUCED_SWIRL_VACC_COMPONENT_DIGEST_TAG);
    for digest in [
        profile.protocol_digest,
        profile.relation_digest,
        profile.warp_index_digest,
        profile.schedule_digest,
    ] {
        for limb in digest {
            <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(&mut transcript, limb);
        }
    }
    for value in [
        profile.maximum_sources,
        profile.input_arity,
        profile.num_ood,
        profile.num_shift_queries,
        profile.batching_arity,
        profile.family_target_bits,
        profile.source.maximum_roots_per_source,
        profile.source.maximum_openings_per_source,
        profile.source.log_message_len(),
        profile.source.log_codeword_len(),
        profile.source.rows_per_query(),
        source_mode.setup_tag(),
        REDUCED_SWIRL_VACC_SOURCE_LEAF_CAPACITY,
        REDUCED_SWIRL_VACC_SOURCE_LEAF_PROTOCOL_VERSION as usize,
        shared.poseidon2_permute_bus.index() as usize,
        shared.poseidon2_compress_bus.index() as usize,
        main_transcript.index() as usize,
        footer.index() as usize,
        source_authority.map_or(0, |bus| bus.index() as usize),
        source_summary_bus_index.map_or(0, |index| index as usize),
        wrapper_receipt.index() as usize,
        next_bus_idx as usize,
        standard_air_count,
        aggregate_air_count,
    ] {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
            &mut transcript,
            F::from_usize(value),
        );
    }
    core::array::from_fn(|_| {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::sample(&mut transcript)
    })
}

fn observe_component_bytes(
    transcript: &mut openvm_stark_sdk::config::baby_bear_poseidon2::DuplexSpongeRecorder,
    bytes: &[u8],
) {
    <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
        transcript,
        F::from_usize(bytes.len()),
    );
    for &byte in bytes {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(transcript, F::from_u8(byte));
    }
}

fn observe_component_u64(
    transcript: &mut openvm_stark_sdk::config::baby_bear_poseidon2::DuplexSpongeRecorder,
    value: u64,
) {
    for byte in value.to_le_bytes() {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(transcript, F::from_u8(byte));
    }
}

// This assertion catches accidental receipt-layout drift at compile time.
const _: [(); 58] = [(); RECEIPT_FIELD_COUNT];

#[cfg(test)]
mod tests {
    use core::borrow::BorrowMut;
    use std::panic::AssertUnwindSafe;

    use openvm_continuations::circuit::{
        reduced_swirl_source_leaf::{
            reduced_swirl_source_chain_genesis, ReducedSwirlSourceLeafRecord,
            REDUCED_SWIRL_SOURCE_LEAF_CAPACITY, REDUCED_SWIRL_SOURCE_LEAF_PROTOCOL_VERSION,
        },
        reduced_swirl_source_receipt::{
            canonicalize_reduced_swirl_source_receipt_block, ReducedSwirlReceiptLayoutEntry,
            ReducedSwirlReceiptVmBoundary, ReducedSwirlSourceReceiptProfile,
            ReducedSwirlSourceReceiptRecord,
        },
    };
    use openvm_recursion_circuit::{
        bus::Poseidon2CompressBus,
        native_warp::{
            generate_reduced_swirl_source_digest_trace,
            generate_reduced_swirl_vacc_source_summary_trace, ReducedSwirlSourceDigestRecord,
            ReducedSwirlSourceProfile, ReducedSwirlVaccDetachedSourceBus,
            ReducedSwirlVaccSourceSummaryAir, ReducedSwirlVaccSourceSummaryBus,
            ReducedSwirlVaccSourceSummaryCols, ReducedSwirlVaccSummaryManifestBus,
        },
    };
    use openvm_stark_backend::{
        air_builders::{debug::check_constraints, symbolic::get_symbolic_builder},
        interaction::SymbolicInteraction,
        keygen::types::TraceWidth,
        AnyAir, WhirProximityStrategy,
    };

    use super::*;
    use crate::prover::native_warp::reduced_swirl_boundary::{
        AuthoritativeSwirlConstrainedRsClaim, ReducedSwirlSourceManifestPrefix,
        ReducedSwirlVmBoundary, ReducedSwirlVmState,
    };

    fn digest(seed: u32) -> Digest {
        core::array::from_fn(|limb| F::from_u32(seed + limb as u32 + 1))
    }

    fn unique_decoding_test_params(log_max_height: usize) -> SystemParams {
        let mut params = SystemParams::new_for_testing(log_max_height);
        params.whir.proximity = WhirProximityStrategy::UniqueDecoding;
        params
    }

    fn symbolic_interactions(air: &dyn AnyAir<SC>) -> Vec<SymbolicInteraction<F>> {
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

    fn receipt() -> ReducedSwirlVaccChainReceiptMessage<F> {
        ReducedSwirlVaccChainReceiptMessage {
            protocol_digest: digest(10),
            relation_digest: digest(30),
            warp_index_digest: digest(50),
            schedule_digest: digest(70),
            manifest_digest: digest(90),
            source_count: F::from_u32(429),
            call_count: F::from_u32(62),
            final_accumulator_digest: digest(110),
            final_accumulator_root: digest(130),
        }
    }

    fn native_warp_and_terminal_indices(
        mut system_params: SystemParams,
        input_arity: usize,
        family_target_bits: usize,
    ) -> (Digest, Digest) {
        system_params.whir.proximity = WhirProximityStrategy::UniqueDecoding;
        let config = SC::default_from_params(system_params.clone());
        let native = ReducedSwirlNativeSetup::new(
            &config,
            &system_params,
            input_arity,
            family_target_bits,
            16,
        )
        .unwrap();
        let terminal = ReducedSwirlTerminalProductionSetup::from_native_fixed_params(
            system_params.clone(),
            input_arity,
            family_target_bits,
            16,
        )
        .unwrap();
        let external = canonical_reduced_constrained_code_protocol_binding::<F, EF, _, _>(
            native.relation(),
            native.binding(),
            native.code(),
        )
        .unwrap();
        let index = derive_reduced_swirl_warp_index_digest(
            &native,
            &system_params,
            terminal.protocol_digest(),
            terminal.relation_digest(),
            &external,
        )
        .unwrap();
        (index, terminal.terminal_index_digest())
    }

    #[test]
    fn warp_index_is_domain_separated_and_binds_arity_security_and_layout() {
        let baseline = native_warp_and_terminal_indices(SystemParams::new_for_testing(10), 2, 8);
        assert!(baseline.0.iter().any(|limb| *limb != F::ZERO));
        assert_ne!(baseline.0, baseline.1);
        let changed_arity =
            native_warp_and_terminal_indices(SystemParams::new_for_testing(10), 4, 8);
        let changed_security =
            native_warp_and_terminal_indices(SystemParams::new_for_testing(10), 2, 9);
        let changed_layout =
            native_warp_and_terminal_indices(SystemParams::new_for_testing(11), 2, 8);
        assert_ne!(baseline.0, changed_arity.0);
        assert_ne!(baseline.0, changed_security.0);
        assert_ne!(baseline.0, changed_layout.0);
    }

    #[test]
    fn fresh_swirl_beta_excludes_target_while_vacc_accumulator_beta_includes_it() {
        assert_eq!(
            openvm_recursion_circuit::native_warp::REDUCED_SWIRL_VACC_PROTOCOL_VERSION,
            super::super::reduced_swirl_native::REDUCED_SWIRL_NATIVE_PROTOCOL_VERSION
        );
        let system_params = unique_decoding_test_params(10);
        let config = SC::default_from_params(system_params.clone());
        let native = ReducedSwirlNativeSetup::new(&config, &system_params, 2, 8, 16).unwrap();
        let shape = <_ as ReducedConstrainedCodeRelation<F, EF>>::shape(native.relation());
        assert_eq!(shape.beta_len, shape.log_message_len);

        let normalized_beta_len = shape.beta_len + 1;
        let standard = NativeStandardVaccProfile::from_reduced_constrained_code(
            native.binding().source_domain.clone(),
            digest(1_300),
            native.security().params,
            shape.log_message_len,
            native.code().log_codeword_len(),
            0,
            0,
            normalized_beta_len,
            shape.max_constraint_degree,
            1usize << system_params.log_commit_rows_per_query,
        )
        .unwrap();
        let accumulator =
            NativePrivateAccumulatorLayout::new(0, standard.log_codeword_len, standard.beta_len);
        assert_eq!(standard.beta_len, shape.beta_len + 1);
        assert_eq!(accumulator.beta_len, shape.beta_len + 1);
    }

    #[test]
    fn detached_component_is_setup_derived_and_mode_bound() {
        let system_params = unique_decoding_test_params(10);
        let config = SC::default_from_params(system_params.clone());
        let native = ReducedSwirlNativeSetup::new(&config, &system_params, 2, 8, 16).unwrap();
        let wrapper_receipt_bus = ReducedSwirlVaccReceiptBus::new(0);
        let detached = ProductionReducedSwirlVaccComponent::from_native_setup_detached(
            &native,
            system_params.clone(),
            wrapper_receipt_bus,
            BusIndexManager::from_next_bus_idx(1),
        )
        .unwrap();
        assert_eq!(detached.source_mode(), ReducedSwirlVaccSourceMode::Detached);
        assert!(detached.optional_source_authority_bus().is_none());
        assert!(detached.source_summary_bus().is_some());
        assert!(detached.source_summary_bus_index().is_some());
        assert!(detached.aggregate.aggregate.buses.source_claim.is_none());
        assert!(detached.aggregate.aggregate.buses.source_root.is_none());
        assert!(detached.aggregate.aggregate.buses.source_beta.is_none());
        assert!(detached
            .aggregate
            .aggregate
            .buses
            .source_authority
            .is_none());
        assert_eq!(
            detached.profile().source.maximum_sources,
            native.maximum_source_count()
        );
        assert_eq!(
            detached.profile().source.maximum_openings_per_source,
            system_params.w_stack
        );
        assert_eq!(
            detached.profile().source.maximum_roots_per_source,
            system_params.w_stack.min(REDUCED_SWIRL_VACC_MAX_ROOTS)
        );
        let aggregate_airs = detached.aggregate.aggregate.airs::<SC>();
        let interactions = aggregate_airs
            .iter()
            .flat_map(|air| symbolic_interactions(air.as_ref()))
            .collect::<Vec<_>>();
        let summary_bus_index = detached.source_summary_bus_index().unwrap();
        let summary_interactions = interactions
            .iter()
            .filter(|interaction| interaction.bus_index == summary_bus_index)
            .collect::<Vec<_>>();
        assert_eq!(summary_interactions.len(), 1);
        assert_eq!(summary_interactions[0].count_weight, 0);
        for bus_index in [detached.aggregate.aggregate.buses.vacc_claim.index()] {
            assert!(
                interactions
                    .iter()
                    .any(|interaction| interaction.bus_index == bus_index),
                "detached source must remain constrained by standard WARP bus {bus_index}"
            );
        }
        let mut invalid_mode = detached.aggregate.aggregate.clone();
        invalid_mode.source_mode = ReducedSwirlVaccSourceMode::Inline;
        assert_eq!(
            invalid_mode.validate(),
            Err("reduced-SWIRL source-linkage mode")
        );

        let mut manager = BusIndexManager::from_next_bus_idx(1);
        let shared = BusInventory::new(&mut manager);
        let profile = detached.profile().clone();
        let common = (
            &profile,
            &shared,
            TranscriptBus::new(30),
            ReducedSwirlVaccFooterBus::new(31),
            None,
            wrapper_receipt_bus,
            BusIndex::from(32u16),
            11,
            7,
        );
        let inline_digest = reduced_swirl_vacc_component_digest(
            common.0,
            common.1,
            ReducedSwirlVaccSourceMode::Inline,
            common.2,
            common.3,
            common.4,
            None,
            common.5,
            common.6,
            common.7,
            common.8,
        );
        let detached_digest = reduced_swirl_vacc_component_digest(
            common.0,
            common.1,
            ReducedSwirlVaccSourceMode::Detached,
            common.2,
            common.3,
            common.4,
            Some(BusIndex::from(29u16)),
            common.5,
            common.6,
            common.7,
            common.8,
        );
        assert_ne!(inline_digest, detached_digest);
    }

    #[test]
    fn detached_component_rejects_system_parameters_not_bound_by_native_setup() {
        let system_params = unique_decoding_test_params(10);
        let config = SC::default_from_params(system_params.clone());
        let native = ReducedSwirlNativeSetup::new(&config, &system_params, 2, 8, 16).unwrap();
        let error = ProductionReducedSwirlVaccComponent::from_native_setup_detached(
            &native,
            unique_decoding_test_params(11),
            ReducedSwirlVaccReceiptBus::new(0),
            BusIndexManager::from_next_bus_idx(1),
        )
        .err()
        .expect("mismatched setup must be rejected");
        assert!(matches!(error, ReducedSwirlVaccComponentError::Setup(_)));
    }

    fn canonical_digest_profiles() -> (ReducedSwirlSourceReceiptProfile, ReducedSwirlVaccProfile) {
        let source = ReducedSwirlSourceProfile {
            maximum_sources: 1024,
            maximum_roots_per_source: 1,
            maximum_openings_per_source: 1,
            l_skip: 1,
            n_stack: 1,
            log_blowup: 1,
            log_commit_rows_per_query: 0,
        };
        let receipt = ReducedSwirlSourceReceiptProfile {
            source: source.clone(),
            protocol_digest: digest(200),
            child_vk_pre_hash: digest(220),
            child_air_count: 1,
            child_l_skip: 1,
            cached_global_indices: vec![vec![0]],
            suspend_exit_code: 2,
        };
        let vacc = ReducedSwirlVaccProfile {
            maximum_sources: 1024,
            input_arity: 8,
            num_ood: 1,
            num_shift_queries: 2,
            batching_arity: 4,
            family_target_bits: 80,
            source,
            source_domain: b"reduced-swirl-canonical-digest-test".to_vec(),
            relation_binding: vec![EF::from_u32(3)],
            code_binding: vec![EF::from_u32(5)],
            external_protocol_binding: vec![EF::from_u32(7)],
            protocol_digest: digest(240),
            relation_digest: digest(260),
            warp_index_digest: digest(280),
            schedule_digest: digest(300),
        };
        (receipt, vacc)
    }

    fn canonical_source_block(
        profile: &ReducedSwirlSourceReceiptProfile,
        source_count: usize,
    ) -> ReducedSwirlSourceReceiptBlock {
        let program = digest(400);
        let mut sources = Vec::with_capacity(source_count);
        let mut initial_root = digest(420);
        let mut initial_pc = F::from_u32(3);
        for source in 0..source_count {
            let final_root = digest(500 + 20 * source as u32);
            let final_pc = initial_pc + F::from_u32(4);
            let point = vec![EF::from_usize(10 + source), EF::from_usize(30 + source)];
            sources.push(ReducedSwirlSourceReceiptRecord {
                segment_index: source as u32,
                checkpoint_tidx: 64 + 4 * source,
                checkpoint_samples: EF::from_usize(50 + source),
                checkpoint_state: core::array::from_fn(|limb| {
                    F::from_usize(600 + source * POSEIDON2_WIDTH + limb)
                }),
                layout: vec![ReducedSwirlReceiptLayoutEntry {
                    log_height: Some(4),
                    cached_commitments: vec![program],
                }],
                roots: vec![digest(1_000 + 20 * source as u32)],
                widths: vec![1],
                stacking_point: point.clone(),
                stacking_openings: vec![vec![EF::from_usize(70 + source)]],
                theta: EF::from_usize(90 + source),
                mu: EF::from_usize(110 + source),
                beta: point.into_iter().rev().collect(),
                eta: EF::from_usize(130 + source),
                vm: ReducedSwirlReceiptVmBoundary {
                    program_commitment: program,
                    initial_pc,
                    initial_root,
                    final_pc,
                    final_root,
                    exit_code: F::from_u32(if source + 1 == source_count { 0 } else { 2 }),
                    is_terminate: F::from_bool(source + 1 == source_count),
                },
                layout_digest: [F::ZERO; DIGEST_SIZE],
                pending_digest: [F::ZERO; DIGEST_SIZE],
                claim_digest: [F::ZERO; DIGEST_SIZE],
                entry_digest: [F::ZERO; DIGEST_SIZE],
            });
            initial_pc = final_pc;
            initial_root = final_root;
        }
        let mut block = ReducedSwirlSourceReceiptBlock {
            source_offset: 0,
            sources,
            manifest_digest: [F::ZERO; DIGEST_SIZE],
        };
        canonicalize_reduced_swirl_source_receipt_block(profile, &mut block).unwrap();
        block
    }

    fn canonical_vacc_claim(
        source: &ReducedSwirlSourceReceiptRecord,
    ) -> RecursiveReducedSwirlClaim {
        RecursiveReducedSwirlClaim {
            roots: source.roots.clone(),
            widths: source.widths.clone(),
            theta: source.theta,
            alpha: vec![EF::ZERO; 3],
            mu: source.mu,
            beta: source.beta.clone(),
            eta: source.eta,
        }
    }

    fn canonical_vacc_source_records(
        block: &ReducedSwirlSourceReceiptBlock,
    ) -> Vec<ReducedSwirlSourceDigestRecord> {
        block
            .sources
            .iter()
            .enumerate()
            .map(|(source_index, source)| ReducedSwirlSourceDigestRecord {
                claim: canonical_vacc_claim(source),
                authority: ReducedSwirlSourceAuthorityRecord {
                    segment_index: source.segment_index,
                    common_main_root: source.roots[0],
                    trace_layout_digest: source.layout_digest,
                    pending_claim_digest: source.pending_digest,
                    checkpoint_tidx: source.checkpoint_tidx,
                    checkpoint_state: source.checkpoint_state,
                    program_commitment: source.vm.program_commitment,
                    initial_pc: source.vm.initial_pc,
                    initial_root: source.vm.initial_root,
                    final_pc: source.vm.final_pc,
                    final_root: source.vm.final_root,
                    exit_code: source.vm.exit_code,
                    is_terminate: source.vm.is_terminate,
                },
                commitment_tidx: 4_000 + source_index,
                first_tree_id: source_index as u32,
            })
            .collect()
    }

    fn canonical_execution_claim(
        source: &ReducedSwirlSourceReceiptRecord,
    ) -> AuthoritativeSwirlConstrainedRsClaim {
        AuthoritativeSwirlConstrainedRsClaim {
            root_tuple: source.roots.clone(),
            commitment_widths: source.widths.clone(),
            l_skip: 1,
            n_stack: 1,
            log_blowup: 1,
            log_commit_rows_per_query: 0,
            theta: source.theta,
            alpha: vec![EF::ZERO; 3],
            mu: source.mu,
            beta: source.beta.clone(),
            eta: source.eta,
        }
    }

    fn digest_test_calls(
        profile: &ReducedSwirlVaccProfile,
        source_count: usize,
    ) -> Vec<ReducedSwirlVaccCallRecord> {
        reduced_swirl_vacc_schedule(source_count, profile.input_arity)
            .unwrap()
            .into_iter()
            .map(|call| ReducedSwirlVaccCallRecord {
                call,
                batch_start_tidx: 1_000 + call.step * 1_000,
                vacc_start_tidx: 1_100 + call.step * 1_000,
                vacc_end_tidx: 1_900 + call.step * 1_000,
                start_sample_count: usize::from(call.step != 0),
                start_state: core::array::from_fn(|limb| {
                    F::from_usize(2_000 + call.step * POSEIDON2_WIDTH + limb)
                }),
                end_sample_count: 1,
                end_state: core::array::from_fn(|limb| {
                    F::from_usize(3_000 + call.step * POSEIDON2_WIDTH + limb)
                }),
                prior_root: (call.step != 0).then(|| digest(3_200 + 20 * call.step as u32)),
                output_root: digest(3_220 + 20 * call.step as u32),
                prior_digest: (call.step != 0).then(|| digest(3_400 + 20 * call.step as u32)),
                output_digest: digest(3_420 + 20 * call.step as u32),
            })
            .collect()
    }

    #[test]
    fn canonical_source_digest_is_equal_across_execution_receipt_vacc_host_and_air() {
        let (receipt_profile, vacc_profile) = canonical_digest_profiles();
        // Bootstrap, exact first-arity boundary, partial continuation, and a
        // later continuation after more than one VACC call.
        for (source_count, selected) in [(1, 0), (8, 7), (9, 8), (17, 16)] {
            let block = canonical_source_block(&receipt_profile, source_count);
            let calls = digest_test_calls(&vacc_profile, source_count);
            let records = canonical_vacc_source_records(&block);
            let air_artifacts =
                generate_reduced_swirl_source_digest_trace(&vacc_profile, 1, &calls, &records)
                    .unwrap();
            let source = &block.sources[selected];
            let execution_claim = canonical_execution_claim(source);
            let execution_digest = ReducedSwirlSourceManifestPrefix {
                source_index: selected as u32,
                segment_index: source.segment_index,
                common_main_root: source.roots[0],
                trace_layout_digest: source.layout_digest,
                pending_claim_digest: source.pending_digest,
                vm_pvs: ReducedSwirlVmBoundary {
                    program_commitment: source.vm.program_commitment,
                    initial_state: ReducedSwirlVmState {
                        pc: source.vm.initial_pc,
                        memory_root: source.vm.initial_root,
                    },
                    final_state: ReducedSwirlVmState {
                        pc: source.vm.final_pc,
                        memory_root: source.vm.final_root,
                    },
                    exit_code: source.vm.exit_code,
                    is_terminate: source.vm.is_terminate,
                },
            }
            .digest_with_claim(&execution_claim)
            .unwrap();
            let (call_index, call) =
                reduced_swirl_vacc_schedule(source_count, vacc_profile.input_arity)
                    .unwrap()
                    .into_iter()
                    .enumerate()
                    .find(|(_, call)| {
                        selected >= call.source_start
                            && selected < call.source_start + call.fresh_count
                    })
                    .unwrap();
            let slot = selected - call.source_start;
            let host_digest = reduced_swirl_source_entry_digest(
                &vacc_profile,
                selected,
                call,
                slot,
                &records[selected].claim,
                &records[selected].authority,
            )
            .unwrap();
            assert_eq!(call_index, call.step);
            assert_eq!(execution_digest, source.entry_digest);
            assert_eq!(execution_digest, host_digest);
            assert_eq!(execution_digest, air_artifacts.entry_digests[selected]);
            let entry_log = &air_artifacts.transcript_logs[2 * selected + 1];
            assert_eq!(
                &entry_log.values()[entry_log.len() - DIGEST_SIZE..],
                &execution_digest
            );
        }
    }

    #[test]
    fn detached_summary_matches_fixed_source_leaf_chain_and_boundaries() {
        assert_eq!(
            REDUCED_SWIRL_VACC_SOURCE_LEAF_CAPACITY,
            REDUCED_SWIRL_SOURCE_LEAF_CAPACITY
        );
        assert_eq!(
            REDUCED_SWIRL_VACC_SOURCE_LEAF_PROTOCOL_VERSION,
            REDUCED_SWIRL_SOURCE_LEAF_PROTOCOL_VERSION
        );
        let (receipt_profile, vacc_profile) = canonical_digest_profiles();
        let block = canonical_source_block(&receipt_profile, 65);
        let records = canonical_vacc_source_records(&block);
        let entry_digests = block
            .sources
            .iter()
            .map(|source| source.entry_digest)
            .collect::<Vec<_>>();
        let vacc = generate_reduced_swirl_vacc_source_summary_trace(
            &vacc_profile,
            &records,
            &entry_digests,
            block.manifest_digest,
        )
        .unwrap();
        let summary_air = ReducedSwirlVaccSourceSummaryAir {
            profile: vacc_profile.clone(),
            source_bus: ReducedSwirlVaccDetachedSourceBus::new(1),
            manifest_bus: ReducedSwirlVaccSummaryManifestBus::new(2),
            transcript_bus: TranscriptBus::new(3),
            compress_bus: Poseidon2CompressBus::new(4),
            summary_bus: ReducedSwirlVaccSourceSummaryBus::new(5),
        };
        let summary_symbolic = get_symbolic_builder(
            &summary_air,
            &TraceWidth {
                preprocessed: None,
                cached_mains: summary_air.cached_main_widths(),
                common_main: summary_air.common_main_width(),
            },
        )
        .constraints();
        assert!(summary_symbolic.max_constraint_degree() <= 4);
        check_constraints::<_, SC>(
            &summary_air,
            "ReducedSwirlVaccSourceSummaryAir",
            &None,
            &[vacc.trace.as_view()],
            &[],
        );
        let mut malformed = vacc.trace.clone();
        let width = summary_air.width();
        let second: &mut ReducedSwirlVaccSourceSummaryCols<F> =
            malformed.values[width..2 * width].borrow_mut();
        second.source += F::ONE;
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
            check_constraints::<_, SC>(
                &summary_air,
                "ReducedSwirlVaccSourceSummaryAir",
                &None,
                &[malformed.as_view()],
                &[],
            );
        }))
        .is_err());

        let mut source_chain = reduced_swirl_source_chain_genesis(vacc_profile.protocol_digest);
        for source_offset in (0..block.sources.len()).step_by(REDUCED_SWIRL_SOURCE_LEAF_CAPACITY) {
            let source_end =
                (source_offset + REDUCED_SWIRL_SOURCE_LEAF_CAPACITY).min(block.sources.len());
            let mut chunk = ReducedSwirlSourceReceiptBlock {
                source_offset: source_offset as u32,
                sources: block.sources[source_offset..source_end].to_vec(),
                manifest_digest: [F::ZERO; DIGEST_SIZE],
            };
            canonicalize_reduced_swirl_source_receipt_block(&receipt_profile, &mut chunk).unwrap();
            let leaf = ReducedSwirlSourceLeafRecord::from_block(
                &chunk,
                vacc_profile.protocol_digest,
                source_chain,
            )
            .unwrap();
            source_chain = leaf.derive().unwrap().chain_after;
        }

        let first = &block.sources[0].vm;
        let last = &block.sources.last().unwrap().vm;
        assert_eq!(vacc.summary.chunk_chain_endpoint, source_chain);
        assert_eq!(vacc.summary.program_commitment, first.program_commitment);
        assert_eq!(vacc.summary.initial_pc, first.initial_pc);
        assert_eq!(vacc.summary.initial_root, first.initial_root);
        assert_eq!(vacc.summary.final_pc, last.final_pc);
        assert_eq!(vacc.summary.final_root, last.final_root);
        assert_eq!(vacc.summary.exit_code, last.exit_code);
        assert_eq!(vacc.summary.is_terminate, last.is_terminate);
        assert_eq!(vacc.summary.manifest_digest, block.manifest_digest);
    }

    #[test]
    fn receipt_bridge_is_constrained_and_layout_exact() {
        let air = ReducedSwirlVaccReceiptBridgeAir {
            chain_bus: ReducedSwirlVaccChainReceiptBus::new(3),
            wrapper_bus: ReducedSwirlVaccReceiptBus::new(7),
        };
        let trace = generate_reduced_swirl_vacc_receipt_bridge_trace(&receipt());
        check_constraints::<_, SC>(
            &air,
            "ReducedSwirlVaccReceiptBridgeAir",
            &None,
            &[trace.as_view()],
            &[],
        );
    }

    #[test]
    fn receipt_bridge_rejects_inactive_or_mutated_local_row() {
        let air = ReducedSwirlVaccReceiptBridgeAir {
            chain_bus: ReducedSwirlVaccChainReceiptBus::new(3),
            wrapper_bus: ReducedSwirlVaccReceiptBus::new(7),
        };
        let honest = generate_reduced_swirl_vacc_receipt_bridge_trace(&receipt());
        let mut inactive = honest.clone();
        inactive.values[0] = F::ZERO;
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
            check_constraints::<_, SC>(
                &air,
                "ReducedSwirlVaccReceiptBridgeAir",
                &None,
                &[inactive.as_view()],
                &[],
            );
        }))
        .is_err());
    }
}

//! One fixed-key CUDA proof joining the transition tree to terminal Decide.
//!
//! The transition leaves already certify the SWIRL source reduction and the
//! genuine WARP/VACC update for every call. This module verifies their ordinary
//! OpenVM recursion-tree root once, reconciles the call-partitioned manifest
//! with the canonical flat block manifest, resumes the native transcript at
//! the final call boundary, and reuses the existing terminal RS-adjoint/WHIR
//! component. No transition is replayed and no PCS opening is presented as an
//! execution PESAT relation.

use core::mem::size_of;
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
};

use openvm_continuations::circuit::{
    reduced_swirl_source_tree_bridge::ReducedSwirlSourceTreeTrustedVkCommits,
    reduced_swirl_transition_finalizer::{
        ReducedSwirlTransitionFinalizerBinding, ReducedSwirlTransitionFinalizerCircuit,
        ReducedSwirlTransitionFinalizerRecord, ReducedSwirlTransitionTerminalJoinAir,
        ReducedSwirlTransitionTerminalJoinBinding, ReducedSwirlTransitionTreeTrustedVkCommits,
    },
    reduced_swirl_transition_leaf::ReducedSwirlTransitionState,
    reduced_swirl_warp::{
        ReducedSwirlExecutionBus, ReducedSwirlWrapperBinding, ReducedSwirlWrapperReceiptBuses,
        ReducedSwirlWrapperVerifierPvsAir, ReducedSwirlWrapperVmPvsAir,
        REDUCED_SWIRL_WRAPPER_PROTOCOL_VERSION,
    },
    Circuit,
};
use openvm_cuda_backend::{
    BabyBearPoseidon2GpuEngine, GpuBackend, GpuDevice, GpuPreprocessedCommitter,
};
use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_recursion_circuit::{
    bus::{ResumeTranscriptStateBus, TranscriptBus},
    native_warp::{
        generate_reduced_swirl_vacc_footer_record_trace, NativeWarpTranscriptModule,
        ReducedSwirlLocalTerminalReceiptBus, ReducedSwirlManifestDigestBus,
        ReducedSwirlManifestReconciliationComponent, ReducedSwirlManifestReconciliationReceiptBus,
        ReducedSwirlTerminalComponent, ReducedSwirlTerminalProductionSetup,
        ReducedSwirlVaccChainEndBus, ReducedSwirlVaccChainReceiptBus, ReducedSwirlVaccFooterAir,
        ReducedSwirlVaccFooterBus, ReducedSwirlVaccProfile,
    },
    system::{
        AggregationSubCircuit, BusIndexManager, CachedTraceCtx, VerifierExternalData,
        VerifierSubCircuit, VerifierTraceGen,
    },
};
use openvm_stark_backend::{
    interaction::BusIndex,
    keygen::{
        types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
        MultiStarkKeygenBuilder,
    },
    p3_field::PrimeCharacteristicRing,
    p3_matrix::dense::RowMajorMatrix,
    proof::Proof,
    prover::{
        AirProvingContext, DeviceDataTransporter, DeviceMultiStarkProvingKey, MatrixDimensions,
        ProvingContext,
    },
    AirRef, FiatShamirTranscript, StarkEngine, StarkProtocolConfig, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, BabyBearPoseidon2Config, BabyBearPoseidon2CpuEngine, Digest,
    DuplexSponge, F,
};
use openvm_verify_stark_host::pvs::{VerifierBasePvs, VmPvs};

use super::{
    reduced_swirl_native::{
        ReducedSwirlNativeProverOutput, ReducedSwirlNativeSetup, ReducedSwirlNativeVerification,
    },
    reduced_swirl_terminal_component::{
        generate_reduced_swirl_finalizer_terminal_cpu_packet,
        prepare_reduced_swirl_finalizer_terminal_record, ReducedSwirlTerminalAdapterError,
    },
    reduced_swirl_vacc_component::{
        ProductionReducedSwirlVaccComponent, ReducedSwirlVaccComponentError,
    },
    reduced_swirl_wrapper_system::ReducedSwirlWrapperSystemError,
    reduced_swirl_wrapper_system_cuda::transport_reduced_swirl_wrapper_contexts_to_cuda,
};
use crate::SC;

type CpuEngine = BabyBearPoseidon2CpuEngine<DuplexSponge>;

const FINALIZER_COMPONENT_TAG: &[u8] = b"openvm.native-warp.reduced-swirl.transition-finalizer.v1";

#[derive(Debug, thiserror::Error)]
pub enum ReducedSwirlTransitionFinalizerSystemError {
    #[error("invalid reduced-SWIRL transition finalizer setup: {0}")]
    Setup(String),
    #[error("transition-tree proof is malformed")]
    MalformedChild,
    #[error("transition-tree verification failed: {0}")]
    ChildVerification(String),
    #[error("transition-tree verifier trace generation failed")]
    ChildTrace,
    #[error(transparent)]
    Terminal(#[from] ReducedSwirlTerminalAdapterError),
    #[error(transparent)]
    Vacc(#[from] ReducedSwirlVaccComponentError),
    #[error(transparent)]
    Wrapper(#[from] ReducedSwirlWrapperSystemError),
    #[error("transition-finalizer key generation failed: {0}")]
    Keygen(String),
    #[error("transition-finalizer key integrity failed")]
    KeyIntegrity,
    #[error("transition-finalizer witness inventory is malformed")]
    Witness,
    #[error(
        "transition-finalizer AIR {air_id} ({air_name}) has log height {log_height}, exceeding the setup-fixed stacked height {log_stacked_height}"
    )]
    TraceHeight {
        air_id: usize,
        air_name: String,
        log_height: usize,
        log_stacked_height: usize,
    },
    #[error("transition-finalizer proof failed: {0}")]
    Prover(String),
    #[error("transition-finalizer verification failed: {0}")]
    Verifier(String),
}

#[derive(Clone, Copy)]
struct FinalizerBusIndices {
    reconciliation_receipt: BusIndex,
    resume: BusIndex,
    chain_end: BusIndex,
    manifest_digest: BusIndex,
    execution: BusIndex,
    main_transcript: BusIndex,
    manifest_transcript: BusIndex,
    footer: BusIndex,
    chain_receipt: BusIndex,
    terminal_receipt: BusIndex,
}

impl FinalizerBusIndices {
    fn allocate(manager: &mut BusIndexManager) -> Self {
        Self {
            reconciliation_receipt: manager.new_bus_idx(),
            resume: manager.new_bus_idx(),
            chain_end: manager.new_bus_idx(),
            manifest_digest: manager.new_bus_idx(),
            execution: manager.new_bus_idx(),
            main_transcript: manager.new_bus_idx(),
            manifest_transcript: manager.new_bus_idx(),
            footer: manager.new_bus_idx(),
            chain_receipt: manager.new_bus_idx(),
            terminal_receipt: manager.new_bus_idx(),
        }
    }

    fn as_array(self) -> [BusIndex; 10] {
        [
            self.reconciliation_receipt,
            self.resume,
            self.chain_end,
            self.manifest_digest,
            self.execution,
            self.main_transcript,
            self.manifest_transcript,
            self.footer,
            self.chain_receipt,
            self.terminal_receipt,
        ]
    }
}

/// Complete fixed AIR inventory for the one-proof transition-tree finalizer.
pub struct ReducedSwirlTransitionFinalizerComponents {
    binding: ReducedSwirlWrapperBinding,
    verifier_pvs: Arc<ReducedSwirlWrapperVerifierPvsAir>,
    vm_pvs: Arc<ReducedSwirlWrapperVmPvsAir>,
    transition: Arc<ReducedSwirlTransitionFinalizerCircuit>,
    reconciliation: ReducedSwirlManifestReconciliationComponent,
    manifest_transcript: NativeWarpTranscriptModule,
    terminal_transcript: NativeWarpTranscriptModule,
    footer: Arc<ReducedSwirlVaccFooterAir>,
    terminal: ReducedSwirlTerminalComponent,
    join: Arc<ReducedSwirlTransitionTerminalJoinAir>,
    vacc_profile: ReducedSwirlVaccProfile,
    params: SystemParams,
}

impl ReducedSwirlTransitionFinalizerComponents {
    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn new(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        trusted: ReducedSwirlSourceTreeTrustedVkCommits,
        native_setup: &ReducedSwirlNativeSetup,
        terminal_setup: &ReducedSwirlTerminalProductionSetup,
        native_params: SystemParams,
        proof_params: SystemParams,
    ) -> Result<Self, ReducedSwirlTransitionFinalizerSystemError> {
        // `native_params` define the fixed constrained-RS/WARP statement.
        // `proof_params` define only the PCS envelope used to prove this
        // finalizer, which must also accommodate the recursive child verifier.
        // Keeping them separate prevents a larger recursive wrapper from
        // silently changing the relation or terminal transcript identity.
        let vacc_profile = derive_vacc_profile(native_setup, native_params)?;
        if terminal_setup.protocol_digest() != vacc_profile.protocol_digest
            || terminal_setup.relation_digest() != vacc_profile.relation_digest
            || terminal_setup.input_arity() != vacc_profile.input_arity
            || terminal_setup.maximum_source_count() != vacc_profile.maximum_sources
        {
            return Err(ReducedSwirlTransitionFinalizerSystemError::Setup(
                "native/VACC/terminal fixed identity mismatch".to_owned(),
            ));
        }

        let mut manager = BusIndexManager::from_next_bus_idx(0);
        let indices = FinalizerBusIndices::allocate(&mut manager);
        let transition_binding = ReducedSwirlTransitionFinalizerBinding {
            source_protocol_digest: terminal_setup.protocol_digest(),
            warp_protocol_digest: vacc_profile.protocol_digest,
            relation_digest: vacc_profile.relation_digest,
            warp_index_digest: vacc_profile.warp_index_digest,
            schedule_digest: vacc_profile.schedule_digest,
            trusted_vk_commits: ReducedSwirlTransitionTreeTrustedVkCommits {
                app_vk_commit: trusted.app_vk_commit,
                transition_leaf_vk_commit: trusted.leaf_vk_commit,
                internal_for_leaf_vk_commit: trusted.internal_for_leaf_vk_commit,
                recursive_vk_commit: trusted.recursive_vk_commit,
            },
        };
        let transition = Arc::new(
            ReducedSwirlTransitionFinalizerCircuit::new(
                child_vk,
                transition_binding,
                indices.reconciliation_receipt,
                ResumeTranscriptStateBus::new(indices.resume),
                ReducedSwirlVaccChainEndBus::new(indices.chain_end),
                ReducedSwirlManifestDigestBus::new(indices.manifest_digest),
                ReducedSwirlExecutionBus::new(indices.execution),
                manager,
            )
            .map_err(|error| {
                ReducedSwirlTransitionFinalizerSystemError::Setup(format!("{error:?}"))
            })?,
        );
        let shared = transition.verifier().bus_inventory().clone();
        let manifest_transcript = NativeWarpTranscriptModule::new_for_bus(
            &shared,
            TranscriptBus::new(indices.manifest_transcript),
            proof_params.clone(),
        );
        let mut resumed_shared = shared.clone();
        resumed_shared.resume_state_bus = ResumeTranscriptStateBus::new(indices.resume);
        let terminal_transcript = NativeWarpTranscriptModule::new_for_bus_resumed_with_end_index(
            &resumed_shared,
            TranscriptBus::new(indices.main_transcript),
            proof_params.clone(),
        );
        let reconciliation = ReducedSwirlManifestReconciliationComponent {
            input_arity: vacc_profile.input_arity,
            source_protocol_digest: terminal_setup.protocol_digest(),
            transcript_bus: TranscriptBus::new(indices.manifest_transcript),
            compress_bus: shared.poseidon2_compress_bus,
            receipt_bus: ReducedSwirlManifestReconciliationReceiptBus::new(
                indices.reconciliation_receipt,
            ),
        };
        reconciliation.validate().map_err(|message| {
            ReducedSwirlTransitionFinalizerSystemError::Setup(message.to_owned())
        })?;
        let footer = Arc::new(ReducedSwirlVaccFooterAir {
            profile: vacc_profile.clone(),
            transcript_bus: TranscriptBus::new(indices.main_transcript),
            chain_end_bus: ReducedSwirlVaccChainEndBus::new(indices.chain_end),
            manifest_digest_bus: ReducedSwirlManifestDigestBus::new(indices.manifest_digest),
            footer_bus: ReducedSwirlVaccFooterBus::new(indices.footer),
            receipt_bus: ReducedSwirlVaccChainReceiptBus::new(indices.chain_receipt),
        });
        let terminal_component_digest = terminal_setup
            .component_protocol_digest(
                &shared,
                TranscriptBus::new(indices.main_transcript),
                ReducedSwirlVaccFooterBus::new(indices.footer),
                indices.terminal_receipt,
                transition.next_bus_idx(),
            )
            .map_err(|error| {
                ReducedSwirlTransitionFinalizerSystemError::Setup(format!("{error:?}"))
            })?;
        let verifier_component_digest = finalizer_component_digest(
            &proof_params,
            transition_binding,
            terminal_setup,
            terminal_component_digest,
            indices,
            <ReducedSwirlTransitionFinalizerCircuit as Circuit<SC>>::airs(transition.as_ref())
                .len(),
        )?;
        let terminal = terminal_setup
            .instantiate(
                verifier_component_digest,
                &shared,
                TranscriptBus::new(indices.main_transcript),
                ReducedSwirlVaccFooterBus::new(indices.footer),
                indices.terminal_receipt,
                transition.next_bus_idx(),
            )
            .map_err(|error| {
                ReducedSwirlTransitionFinalizerSystemError::Setup(format!("{error:?}"))
            })?;
        let join = Arc::new(
            ReducedSwirlTransitionTerminalJoinAir::new(
                ReducedSwirlTransitionTerminalJoinBinding {
                    protocol_digest: vacc_profile.protocol_digest,
                    relation_digest: vacc_profile.relation_digest,
                    warp_index_digest: vacc_profile.warp_index_digest,
                    schedule_digest: vacc_profile.schedule_digest,
                    terminal_index_digest: terminal_setup.terminal_index_digest(),
                    verifier_component_digest,
                },
                ReducedSwirlVaccChainReceiptBus::new(indices.chain_receipt),
                ReducedSwirlLocalTerminalReceiptBus::new(indices.terminal_receipt),
            )
            .map_err(|error| {
                ReducedSwirlTransitionFinalizerSystemError::Setup(format!("{error:?}"))
            })?,
        );
        let binding = ReducedSwirlWrapperBinding {
            protocol_version: REDUCED_SWIRL_WRAPPER_PROTOCOL_VERSION,
            protocol_digest: vacc_profile.protocol_digest,
            relation_digest: vacc_profile.relation_digest,
            warp_index_digest: vacc_profile.warp_index_digest,
            terminal_index_digest: terminal_setup.terminal_index_digest(),
            verifier_component_digest,
            input_arity: u32::try_from(vacc_profile.input_arity).map_err(|_| {
                ReducedSwirlTransitionFinalizerSystemError::Setup(
                    "WARP arity exceeds u32".to_owned(),
                )
            })?,
            recursive_app_vk_commit: trusted.app_vk_commit,
        };
        binding.validate().map_err(|message| {
            ReducedSwirlTransitionFinalizerSystemError::Setup(message.to_owned())
        })?;
        let components = Self {
            verifier_pvs: Arc::new(ReducedSwirlWrapperVerifierPvsAir::new(&binding)),
            vm_pvs: Arc::new(ReducedSwirlWrapperVmPvsAir::new(
                ReducedSwirlExecutionBus::new(indices.execution),
            )),
            binding,
            transition,
            reconciliation,
            manifest_transcript,
            terminal_transcript,
            footer,
            terminal,
            join,
            vacc_profile,
            params: proof_params,
        };
        components.validate_air_inventory()?;
        Ok(components)
    }

    #[must_use]
    pub const fn binding(&self) -> &ReducedSwirlWrapperBinding {
        &self.binding
    }

    #[must_use]
    pub fn child_vk(&self) -> &Arc<MultiStarkVerifyingKey<SC>> {
        self.transition.child_vk()
    }

    #[must_use]
    pub const fn params(&self) -> &SystemParams {
        &self.params
    }

    #[must_use]
    pub fn airs(&self) -> Vec<AirRef<SC>> {
        <Self as Circuit<SC>>::airs(self)
    }

    fn validate_air_inventory(&self) -> Result<(), ReducedSwirlTransitionFinalizerSystemError> {
        self.binding.validate().map_err(|message| {
            ReducedSwirlTransitionFinalizerSystemError::Setup(message.to_owned())
        })?;
        let airs = self.airs();
        if airs.len()
            != 2 + <ReducedSwirlTransitionFinalizerCircuit as Circuit<SC>>::airs(
                self.transition.as_ref(),
            )
            .len()
                + 1
                + 1
                + 1
                + 1
                + ReducedSwirlTerminalComponent::COMPONENT_COUNT
                + 1
            || airs.first().map(|air| air.num_public_values())
                != Some(size_of::<VerifierBasePvs<u8>>())
            || airs.get(1).map(|air| air.num_public_values()) != Some(size_of::<VmPvs<u8>>())
            || airs
                .get(2..)
                .is_none_or(|airs| airs.iter().any(|air| air.num_public_values() != 0))
        {
            return Err(ReducedSwirlTransitionFinalizerSystemError::KeyIntegrity);
        }
        Ok(())
    }
}

impl<C: StarkProtocolConfig<F = F>> Circuit<C> for ReducedSwirlTransitionFinalizerComponents {
    fn airs(&self) -> Vec<AirRef<C>> {
        let mut airs = vec![
            self.verifier_pvs.clone() as AirRef<C>,
            self.vm_pvs.clone() as AirRef<C>,
        ];
        airs.extend(
            <ReducedSwirlTransitionFinalizerCircuit as Circuit<C>>::airs(self.transition.as_ref()),
        );
        airs.push(self.reconciliation.air::<C>());
        airs.push(self.manifest_transcript.airs::<C>()[0].clone());
        airs.push(self.terminal_transcript.airs::<C>()[0].clone());
        airs.push(self.footer.clone() as AirRef<C>);
        airs.extend(self.terminal.airs::<C>());
        airs.push(self.join.clone() as AirRef<C>);
        airs
    }
}

#[derive(Clone)]
pub struct ReducedSwirlTransitionFinalizerCudaKeys {
    binding: ReducedSwirlWrapperBinding,
    proving_key: Arc<MultiStarkProvingKey<SC>>,
    verifying_key: Arc<MultiStarkVerifyingKey<SC>>,
}

impl ReducedSwirlTransitionFinalizerCudaKeys {
    #[must_use]
    pub fn verifying_key(&self) -> Arc<MultiStarkVerifyingKey<SC>> {
        Arc::clone(&self.verifying_key)
    }
}

pub struct ReducedSwirlTransitionFinalizerCudaProver {
    components: Arc<ReducedSwirlTransitionFinalizerComponents>,
    keys: ReducedSwirlTransitionFinalizerCudaKeys,
    device_key: DeviceMultiStarkProvingKey<GpuBackend>,
    engine: BabyBearPoseidon2GpuEngine,
}

impl ReducedSwirlTransitionFinalizerCudaProver {
    pub fn new(
        components: Arc<ReducedSwirlTransitionFinalizerComponents>,
    ) -> Result<Self, ReducedSwirlTransitionFinalizerSystemError> {
        components.validate_air_inventory()?;
        let mut engine = BabyBearPoseidon2GpuEngine::new(components.params.clone());
        engine.device_mut().prover_config_mut().compile_monomials = false;
        let config = SC::default_from_params(components.params.clone());
        let mut builder = MultiStarkKeygenBuilder::with_preprocessed_committer(
            config,
            Arc::new(GpuPreprocessedCommitter::new(engine.device())),
        );
        for air in components.airs() {
            builder.add_required_air(air);
        }
        let proving_key = Arc::new(builder.generate_pk().map_err(|error| {
            ReducedSwirlTransitionFinalizerSystemError::Keygen(error.to_string())
        })?);
        let verifying_key = Arc::new(proving_key.get_vk());
        let keys = ReducedSwirlTransitionFinalizerCudaKeys {
            binding: components.binding.clone(),
            proving_key,
            verifying_key,
        };
        validate_keys(components.as_ref(), &keys)?;
        let prepared = <GpuDevice as DeviceDataTransporter<SC, GpuBackend>>::prepare_pk_for_device(
            engine.device(),
            keys.proving_key.as_ref(),
        );
        let device_key =
            <GpuDevice as DeviceDataTransporter<SC, GpuBackend>>::transport_prepared_pk_to_device(
                engine.device(),
                keys.proving_key.as_ref(),
                prepared,
            );
        Ok(Self {
            components,
            keys,
            device_key,
            engine,
        })
    }

    #[must_use]
    pub fn keys(&self) -> &ReducedSwirlTransitionFinalizerCudaKeys {
        &self.keys
    }

    #[allow(clippy::too_many_arguments, clippy::too_many_lines)]
    pub fn prove(
        &mut self,
        tree_proof: &Proof<SC>,
        initial_state: ReducedSwirlTransitionState,
        final_state: ReducedSwirlTransitionState,
        source_bindings: &[Digest],
        native_setup: &ReducedSwirlNativeSetup,
        output: &ReducedSwirlNativeProverOutput,
        verification: &ReducedSwirlNativeVerification,
    ) -> Result<Proof<SC>, ReducedSwirlTransitionFinalizerSystemError> {
        validate_keys(self.components.as_ref(), &self.keys)?;
        validate_child_public_values(self.components.child_vk(), tree_proof)?;
        catch_unwind(AssertUnwindSafe(|| {
            CpuEngine::new(self.components.child_vk().inner.params.clone())
                .verify(self.components.child_vk().as_ref(), tree_proof)
        }))
        .map_err(|_| ReducedSwirlTransitionFinalizerSystemError::MalformedChild)?
        .map_err(|error| {
            ReducedSwirlTransitionFinalizerSystemError::ChildVerification(error.to_string())
        })?;

        let reconciliation = self
            .components
            .reconciliation
            .generate_trace(source_bindings)
            .map_err(|message| {
                ReducedSwirlTransitionFinalizerSystemError::Setup(message.to_owned())
            })?;
        let record = ReducedSwirlTransitionFinalizerRecord {
            initial_state,
            final_state,
            reconciliation: reconciliation.receipt.clone(),
        };
        let finalizer_trace = self
            .components
            .transition
            .finalizer()
            .generate_trace(&tree_proof.public_values, &record)
            .map_err(|error| {
                ReducedSwirlTransitionFinalizerSystemError::Setup(format!("{error:?}"))
            })?;
        let prepared = prepare_reduced_swirl_finalizer_terminal_record(
            native_setup,
            output,
            verification,
            &record.final_state,
            &record.reconciliation,
        )?;
        let terminal_packet = generate_reduced_swirl_finalizer_terminal_cpu_packet(
            &self.components.terminal,
            &prepared,
            output,
            verification,
        )?;
        let footer = &prepared.footer;
        let footer_trace = generate_reduced_swirl_vacc_footer_record_trace(
            &self.components.vacc_profile,
            footer.source_count,
            footer.call_count,
            footer.proof_idx,
            footer.local_proof_idx,
            footer.start_tidx,
            footer.end_tidx,
            footer.manifest_digest,
            record.final_state.accumulator_digest,
            record.final_state.accumulator_root,
        )
        .map_err(|message| ReducedSwirlTransitionFinalizerSystemError::Setup(message.to_owned()))?;
        let join_trace = self
            .components
            .join
            .generate_trace(&record)
            .map_err(|error| {
                ReducedSwirlTransitionFinalizerSystemError::Setup(format!("{error:?}"))
            })?;

        let manifest_logs = core::iter::once(&reconciliation.flat_manifest_log)
            .chain(reconciliation.local_manifest_logs.iter())
            .collect::<Vec<_>>();
        let manifest_transcript = self
            .components
            .manifest_transcript
            .generate_trace_inputs_with_external(&manifest_logs, Vec::new(), Vec::new(), None)
            .ok_or(ReducedSwirlTransitionFinalizerSystemError::Witness)?;
        let terminal_transcript = self
            .components
            .terminal_transcript
            .generate_trace_inputs_with_external_resumed(
                &[prepared.transcript().log()],
                &[Some(prepared.transcript().resume_input())],
                Vec::new(),
                Vec::new(),
                None,
            )
            .ok_or(ReducedSwirlTransitionFinalizerSystemError::Witness)?;

        let mut poseidon_permutations = terminal_packet.poseidon2_permutation_inputs;
        poseidon_permutations.extend(manifest_transcript.permutation_inputs);
        poseidon_permutations.extend(terminal_transcript.permutation_inputs);
        let mut poseidon_compressions = terminal_packet.poseidon2_compression_inputs;
        poseidon_compressions.extend(reconciliation.compression_inputs);
        poseidon_compressions.extend(finalizer_trace.compression_inputs);
        poseidon_compressions.extend(manifest_transcript.compression_inputs);
        poseidon_compressions.extend(terminal_transcript.compression_inputs);
        let empty_usize = Vec::<usize>::new();
        let mut external = VerifierExternalData {
            poseidon2_compress_inputs: &poseidon_compressions,
            poseidon2_permute_inputs: &poseidon_permutations,
            range_check_inputs: &empty_usize,
            power_check_inputs: &empty_usize,
            required_heights: None,
            final_transcript_state: None,
        };
        let cached = self
            .components
            .transition
            .verifier()
            .cached_trace_record_for_child(self.components.child_vk());
        let child_contexts = catch_unwind(AssertUnwindSafe(|| {
            <VerifierSubCircuit<1> as VerifierTraceGen<GpuBackend, SC, GpuDeviceCtx>>::generate_proving_ctxs(
                self.components.transition.verifier(),
                self.components.child_vk(),
                CachedTraceCtx::Records(cached),
                core::slice::from_ref(tree_proof),
                &mut external,
                &self.engine.device().device_ctx,
                default_duplex_sponge_recorder(),
            )
        }))
        .map_err(|_| ReducedSwirlTransitionFinalizerSystemError::MalformedChild)?
        .ok_or(ReducedSwirlTransitionFinalizerSystemError::ChildTrace)?;

        let pvs_trace = selector_trace();
        let vm_trace = selector_trace();
        let verifier_public_values = self.components.binding.verifier_pvs().as_slice().to_vec();
        let vm_public_values = finalizer_trace.real_vm_pvs.as_slice().to_vec();
        let transition_verifier_count = child_contexts.len();
        let transition_air_count = <ReducedSwirlTransitionFinalizerCircuit as Circuit<SC>>::airs(
            self.components.transition.as_ref(),
        )
        .len();
        if transition_verifier_count + 1 != transition_air_count
            || terminal_packet.contexts.len() != ReducedSwirlTerminalComponent::COMPONENT_COUNT
        {
            return Err(ReducedSwirlTransitionFinalizerSystemError::Witness);
        }
        let reconciliation_air = 2 + transition_air_count;
        let manifest_transcript_air = reconciliation_air + 1;
        let terminal_transcript_air = manifest_transcript_air + 1;
        let footer_air = terminal_transcript_air + 1;
        let terminal_air_start = footer_air + 1;
        let join_air = terminal_air_start + ReducedSwirlTerminalComponent::COMPONENT_COUNT;
        let mut cpu_contexts = vec![
            (
                0,
                AirProvingContext::simple(pvs_trace, verifier_public_values.clone()),
            ),
            (
                1,
                AirProvingContext::simple(vm_trace, vm_public_values.clone()),
            ),
            (2, AirProvingContext::simple_no_pis(finalizer_trace.matrix)),
            (
                reconciliation_air,
                AirProvingContext::simple_no_pis(reconciliation.trace),
            ),
            (
                manifest_transcript_air,
                AirProvingContext::simple_no_pis(manifest_transcript.trace),
            ),
            (
                terminal_transcript_air,
                AirProvingContext::simple_no_pis(terminal_transcript.trace),
            ),
            (footer_air, AirProvingContext::simple_no_pis(footer_trace)),
            (join_air, AirProvingContext::simple_no_pis(join_trace)),
        ];
        cpu_contexts.extend(
            terminal_packet
                .contexts
                .into_iter()
                .enumerate()
                .map(|(index, context)| (terminal_air_start + index, context)),
        );
        let mut contexts =
            transport_reduced_swirl_wrapper_contexts_to_cuda(self.engine.device(), cpu_contexts)?;
        contexts.extend(
            child_contexts
                .into_iter()
                .enumerate()
                .map(|(index, context)| (3 + index, context)),
        );
        contexts.sort_unstable_by_key(|(air, _)| *air);
        validate_contexts(&self.components.airs(), self.components.params(), &contexts)?;
        let proof = self
            .engine
            .prove(&self.device_key, ProvingContext::new(contexts))
            .map_err(|error| {
                ReducedSwirlTransitionFinalizerSystemError::Prover(error.to_string())
            })?;
        let mut expected = vec![verifier_public_values, vm_public_values];
        expected.resize_with(self.components.as_ref().airs().len(), Vec::new);
        if proof.public_values != expected {
            return Err(ReducedSwirlTransitionFinalizerSystemError::Witness);
        }
        CpuEngine::new(self.keys.verifying_key.inner.params.clone())
            .verify(self.keys.verifying_key.as_ref(), &proof)
            .map_err(|error| {
                ReducedSwirlTransitionFinalizerSystemError::Verifier(error.to_string())
            })?;
        Ok(proof)
    }
}

fn derive_vacc_profile(
    native_setup: &ReducedSwirlNativeSetup,
    params: SystemParams,
) -> Result<ReducedSwirlVaccProfile, ReducedSwirlTransitionFinalizerSystemError> {
    // Profile derivation is setup-only. The temporary component is discarded
    // before key generation; no witness or proof-dependent bus is retained.
    let receipt_buses = ReducedSwirlWrapperReceiptBuses::new(0);
    let component = ProductionReducedSwirlVaccComponent::from_native_setup_detached(
        native_setup,
        params,
        receipt_buses.vacc,
        BusIndexManager::from_next_bus_idx(receipt_buses.next_bus_idx()),
    )?;
    Ok(component.profile().clone())
}

#[allow(clippy::too_many_arguments)]
fn finalizer_component_digest(
    params: &SystemParams,
    transition: ReducedSwirlTransitionFinalizerBinding,
    terminal_setup: &ReducedSwirlTerminalProductionSetup,
    terminal_component_digest: Digest,
    indices: FinalizerBusIndices,
    transition_air_count: usize,
) -> Result<Digest, ReducedSwirlTransitionFinalizerSystemError> {
    let mut transcript = default_duplex_sponge_recorder();
    for &byte in FINALIZER_COMPONENT_TAG {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
            &mut transcript,
            F::from_u8(byte),
        );
    }
    for digest in [
        transition.source_protocol_digest,
        transition.warp_protocol_digest,
        transition.relation_digest,
        transition.warp_index_digest,
        transition.schedule_digest,
        terminal_setup.terminal_index_digest(),
        terminal_component_digest,
        transition.trusted_vk_commits.app_vk_commit.cached_commit,
        transition.trusted_vk_commits.app_vk_commit.vk_pre_hash,
        transition
            .trusted_vk_commits
            .transition_leaf_vk_commit
            .cached_commit,
        transition
            .trusted_vk_commits
            .transition_leaf_vk_commit
            .vk_pre_hash,
        transition
            .trusted_vk_commits
            .internal_for_leaf_vk_commit
            .cached_commit,
        transition
            .trusted_vk_commits
            .internal_for_leaf_vk_commit
            .vk_pre_hash,
        transition
            .trusted_vk_commits
            .recursive_vk_commit
            .cached_commit,
        transition
            .trusted_vk_commits
            .recursive_vk_commit
            .vk_pre_hash,
    ] {
        for value in digest {
            <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(&mut transcript, value);
        }
    }
    for value in [
        terminal_setup.input_arity(),
        terminal_setup.family_target_bits(),
        terminal_setup.maximum_source_count(),
        transition_air_count,
        ReducedSwirlTerminalComponent::COMPONENT_COUNT,
        params.l_skip,
        params.n_stack,
        params.w_stack,
        params.log_blowup,
        params.log_commit_rows_per_query,
    ] {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
            &mut transcript,
            F::from_usize(value),
        );
    }
    for index in indices.as_array() {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
            &mut transcript,
            F::from_u16(index),
        );
    }
    Ok(core::array::from_fn(|_| {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::sample(&mut transcript)
    }))
}

fn selector_trace() -> RowMajorMatrix<F> {
    RowMajorMatrix::new(vec![F::ONE, F::ZERO], 1)
}

fn validate_child_public_values(
    child_vk: &MultiStarkVerifyingKey<SC>,
    proof: &Proof<SC>,
) -> Result<(), ReducedSwirlTransitionFinalizerSystemError> {
    if proof.public_values.len() != child_vk.inner.per_air.len()
        || proof
            .public_values
            .iter()
            .zip(&child_vk.inner.per_air)
            .any(|(values, air)| values.len() != air.params.num_public_values)
    {
        return Err(ReducedSwirlTransitionFinalizerSystemError::MalformedChild);
    }
    Ok(())
}

fn validate_keys(
    components: &ReducedSwirlTransitionFinalizerComponents,
    keys: &ReducedSwirlTransitionFinalizerCudaKeys,
) -> Result<(), ReducedSwirlTransitionFinalizerSystemError> {
    components.validate_air_inventory()?;
    let airs = components.airs();
    let config = SC::default_from_params(components.params.clone());
    if keys.binding != components.binding
        || keys.proving_key.params != components.params
        || keys.verifying_key.inner.params != components.params
        || keys.proving_key.per_air.len() != airs.len()
        || keys.verifying_key.inner.per_air.len() != airs.len()
        || keys.proving_key.vk_pre_hash != keys.verifying_key.pre_hash
        || !keys.verifying_key.has_consistent_pre_hash(&config)
    {
        return Err(ReducedSwirlTransitionFinalizerSystemError::KeyIntegrity);
    }
    for (vk, air) in keys.verifying_key.inner.per_air.iter().zip(airs) {
        if !vk.is_required
            || vk.params.num_public_values != air.num_public_values()
            || vk.params.width.common_main != air.common_main_width()
            || vk.params.width.cached_mains != air.cached_main_widths()
        {
            return Err(ReducedSwirlTransitionFinalizerSystemError::KeyIntegrity);
        }
    }
    Ok(())
}

fn validate_contexts(
    airs: &[AirRef<SC>],
    params: &SystemParams,
    contexts: &[(usize, AirProvingContext<GpuBackend>)],
) -> Result<(), ReducedSwirlTransitionFinalizerSystemError> {
    if contexts.len() != airs.len()
        || contexts
            .iter()
            .enumerate()
            .any(|(expected, (actual, _))| expected != *actual)
    {
        return Err(ReducedSwirlTransitionFinalizerSystemError::Witness);
    }
    let mut tallest_overflow = None;
    for ((air_id, context), air) in contexts.iter().zip(airs) {
        let height = context.common_main.height();
        if height == 0
            || !height.is_power_of_two()
            || context.common_main.width() != air.common_main_width()
            || context.public_values.len() != air.num_public_values()
            || context.cached_mains.len() != air.cached_main_widths().len()
            || context
                .cached_mains
                .iter()
                .zip(air.cached_main_widths())
                .any(|(cached, width)| {
                    cached.trace.width() != width || cached.trace.height() != height
                })
        {
            return Err(ReducedSwirlTransitionFinalizerSystemError::Witness);
        }
        let log_height = height.ilog2() as usize;
        if log_height > params.log_stacked_height()
            && tallest_overflow
                .as_ref()
                .is_none_or(|(_, _, tallest)| log_height > *tallest)
        {
            tallest_overflow = Some((*air_id, air.name(), log_height));
        }
    }
    if let Some((air_id, air_name, log_height)) = tallest_overflow {
        return Err(ReducedSwirlTransitionFinalizerSystemError::TraceHeight {
            air_id,
            air_name,
            log_height,
            log_stacked_height: params.log_stacked_height(),
        });
    }
    Ok(())
}

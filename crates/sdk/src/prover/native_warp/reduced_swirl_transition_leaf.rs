//! Production composition and CUDA prover for one bounded source-plus-WARP
//! transition leaf.

#[cfg(feature = "cuda")]
use core::mem::size_of;
use std::sync::Arc;

#[cfg(feature = "cuda")]
use openvm_continuations::circuit::{
    reduced_swirl_transition_leaf::{
        generate_reduced_swirl_transition_leaf_core_traces, ReducedSwirlTransitionLeafBinding,
        ReducedSwirlTransitionLeafCircuit, ReducedSwirlTransitionLeafRecord,
        ReducedSwirlTransitionState,
    },
    reduced_swirl_warp::ReducedSwirlSourceReceiptMessage,
    Circuit,
};
use openvm_continuations::circuit::{
    reduced_swirl_transition_leaf::{
        ReducedSwirlTransitionLeafComponents, REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY,
    },
    reduced_swirl_warp::{ReducedSwirlExecutionBus, ReducedSwirlSourceReceiptBus},
};
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{
    BabyBearPoseidon2GpuEngine, GpuBackend, GpuDevice, GpuPreprocessedCommitter,
};
use openvm_recursion_circuit::{
    bus::{Poseidon2CompressBus, TranscriptBus},
    native_warp::NativeWarpTranscriptModule,
    system::AggregationSubCircuit,
};
#[cfg(feature = "cuda")]
use openvm_stark_backend::{
    keygen::{
        types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
        MultiStarkKeygenBuilder,
    },
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    proof::Proof,
    prover::{
        AirProvingContext, DeviceDataTransporter, DeviceMultiStarkProvingKey, MatrixDimensions,
        ProverBackend, ProvingContext,
    },
    StarkEngine,
};
use openvm_stark_backend::{AirRef, StarkProtocolConfig, SystemParams};
#[cfg(feature = "cuda")]
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2CpuEngine, DuplexSponge};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, F};
#[cfg(feature = "cuda")]
use openvm_verify_stark_host::pvs::{VerifierBasePvs, VmPvs};

#[cfg(feature = "cuda")]
use super::reduced_swirl_cuda_transport::transport_reduced_swirl_contexts_to_cuda;
use super::{
    reduced_swirl_error::ReducedSwirlWrapperSystemError,
    reduced_swirl_source_receipt::ProductionReducedSwirlSourceReceiptComponent,
    reduced_swirl_vacc_component::ProductionReducedSwirlVaccComponent,
    reduced_swirl_wrapper_components::ReducedSwirlVerifierComponent,
};
#[cfg(feature = "cuda")]
use super::{
    reduced_swirl_source_receipt::ReducedSwirlSourceReceiptCudaPacket,
    reduced_swirl_vacc_component::ReducedSwirlVaccTransitionCpuPacket,
};
#[cfg(feature = "cuda")]
use crate::SC;

#[cfg(feature = "cuda")]
type CpuEngine = BabyBearPoseidon2CpuEngine<DuplexSponge>;

/// AIR provider shared by CPU key generation and the persistent CUDA leaf
/// prover. Source and VACC components retain their existing setup and bus
/// inventories; only their bounded inventories are concatenated.
pub struct ProductionReducedSwirlTransitionLeafComponents<
    const CAPACITY: usize = REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY,
> {
    source: Arc<ProductionReducedSwirlSourceReceiptComponent<CAPACITY>>,
    vacc: Arc<ProductionReducedSwirlVaccComponent>,
    execution_bus: ReducedSwirlExecutionBus,
    boundary_poseidon: NativeWarpTranscriptModule,
}

impl<const CAPACITY: usize> ProductionReducedSwirlTransitionLeafComponents<CAPACITY> {
    pub fn new(
        source: Arc<ProductionReducedSwirlSourceReceiptComponent<CAPACITY>>,
        vacc: Arc<ProductionReducedSwirlVaccComponent>,
        params: SystemParams,
    ) -> Result<Self, ReducedSwirlWrapperSystemError> {
        if CAPACITY == 0
            || CAPACITY > REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY
            || !CAPACITY.is_power_of_two()
            || source.inner().receipt_air().profile.source.maximum_sources != CAPACITY
            || source.inner().params() != &params
            || vacc.profile().input_arity != CAPACITY
            || vacc.source_authority_bus().index()
                != source.inner().receipt_air().authority_bus.index()
        {
            return Err(ReducedSwirlWrapperSystemError::KeyIntegrity(
                "transition-leaf component capacity",
            ));
        }
        let execution_bus = ReducedSwirlExecutionBus::new(vacc.next_bus_idx());
        let boundary_poseidon = NativeWarpTranscriptModule::new_for_bus(
            source.inner().verifier().bus_inventory(),
            TranscriptBus::new(vacc.next_bus_idx() + 1),
            params,
        );
        Ok(Self {
            source,
            vacc,
            execution_bus,
            boundary_poseidon,
        })
    }

    #[must_use]
    pub fn source(&self) -> Arc<ProductionReducedSwirlSourceReceiptComponent<CAPACITY>> {
        Arc::clone(&self.source)
    }

    #[must_use]
    pub fn vacc(&self) -> Arc<ProductionReducedSwirlVaccComponent> {
        Arc::clone(&self.vacc)
    }

    #[must_use]
    pub fn source_air_count(&self) -> usize {
        self.source.inner().airs::<crate::SC>().len()
    }

    #[must_use]
    pub fn vacc_air_count(&self) -> usize {
        self.vacc.transition_airs::<crate::SC>().len()
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn boundary_poseidon(&self) -> &NativeWarpTranscriptModule {
        &self.boundary_poseidon
    }

    fn boundary_poseidon_air<SC: StarkProtocolConfig<F = F>>(&self) -> AirRef<SC> {
        self.boundary_poseidon
            .airs::<SC>()
            .into_iter()
            .nth(1)
            .expect("transition-leaf boundary Poseidon AIR")
    }
}

impl<const CAPACITY: usize> ReducedSwirlTransitionLeafComponents
    for ProductionReducedSwirlTransitionLeafComponents<CAPACITY>
{
    fn source_receipt_bus(&self) -> ReducedSwirlSourceReceiptBus {
        self.source.inner().receipt_air().receipt_bus
    }

    fn transition_receipt_bus(
        &self,
    ) -> openvm_recursion_circuit::native_warp::ReducedSwirlVaccTransitionReceiptBus {
        self.vacc.transition_receipt_bus()
    }

    fn execution_bus(&self) -> ReducedSwirlExecutionBus {
        self.execution_bus
    }

    fn compress_bus(&self) -> Poseidon2CompressBus {
        self.source
            .inner()
            .verifier()
            .bus_inventory()
            .poseidon2_compress_bus
    }

    fn source_component_digest(&self) -> Digest {
        self.source.protocol_digest()
    }

    fn vacc_component_digest(&self) -> Digest {
        self.vacc.protocol_digest()
    }

    fn component_air_count(&self) -> usize {
        self.source_air_count() + self.vacc_air_count() + 1
    }

    fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        self.source
            .airs::<SC>()
            .into_iter()
            .chain(self.vacc.transition_airs::<SC>())
            .chain(core::iter::once(self.boundary_poseidon_air::<SC>()))
            .collect()
    }
}

/// Setup-fixed CUDA system for a bounded source-plus-VACC transition leaf.
///
/// The AIR inventory is keyed once and reused by
/// [`ReducedSwirlTransitionLeafCudaProver`] for every canonical WARP call.
/// Dynamic block data therefore cannot select a different relation or table
/// layout.
#[cfg(feature = "cuda")]
pub struct ReducedSwirlTransitionLeafCudaSystem<
    const CAPACITY: usize = REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY,
> {
    circuit: Arc<
        ReducedSwirlTransitionLeafCircuit<ProductionReducedSwirlTransitionLeafComponents<CAPACITY>>,
    >,
    params: SystemParams,
}

#[cfg(feature = "cuda")]
impl<const CAPACITY: usize> ReducedSwirlTransitionLeafCudaSystem<CAPACITY> {
    pub fn new(
        binding: ReducedSwirlTransitionLeafBinding,
        components: Arc<ProductionReducedSwirlTransitionLeafComponents<CAPACITY>>,
        params: SystemParams,
    ) -> Result<Self, ReducedSwirlWrapperSystemError> {
        if components.source.inner().params() != &params {
            return Err(ReducedSwirlWrapperSystemError::KeyIntegrity(
                "transition-leaf system parameters",
            ));
        }
        let circuit = Arc::new(
            ReducedSwirlTransitionLeafCircuit::new(binding, components)
                .map_err(ReducedSwirlWrapperSystemError::Binding)?,
        );
        let system = Self { circuit, params };
        system.validate_air_inventory()?;
        Ok(system)
    }

    #[must_use]
    pub fn binding(&self) -> &ReducedSwirlTransitionLeafBinding {
        &self.circuit.binding
    }

    #[must_use]
    pub fn circuit(
        &self,
    ) -> Arc<
        ReducedSwirlTransitionLeafCircuit<ProductionReducedSwirlTransitionLeafComponents<CAPACITY>>,
    > {
        Arc::clone(&self.circuit)
    }

    #[must_use]
    pub fn airs(&self) -> Vec<AirRef<SC>> {
        self.circuit.airs()
    }

    fn validate_air_inventory(&self) -> Result<(), ReducedSwirlWrapperSystemError> {
        self.binding()
            .validate()
            .map_err(ReducedSwirlWrapperSystemError::Binding)?;
        let airs = self.airs();
        if self.binding().source_capacity as usize != CAPACITY
            || self.circuit.components.source_air_count() == 0
            || self.circuit.components.vacc_air_count() == 0
            || airs.len() != 3 + self.circuit.components.component_air_count()
            || airs.first().map(|air| air.num_public_values())
                != Some(size_of::<VerifierBasePvs<u8>>())
            || airs.get(1).map(|air| air.num_public_values()) != Some(size_of::<VmPvs<u8>>())
            || airs
                .get(2..)
                .is_none_or(|airs| airs.iter().any(|air| air.num_public_values() != 0))
        {
            return Err(ReducedSwirlWrapperSystemError::KeyIntegrity(
                "transition-leaf AIR inventory",
            ));
        }
        Ok(())
    }

    fn keygen_cuda(
        &self,
        device: &GpuDevice,
    ) -> Result<ReducedSwirlTransitionLeafCudaKeys, ReducedSwirlWrapperSystemError> {
        self.validate_air_inventory()?;
        let config = SC::default_from_params(self.params.clone());
        let mut builder = MultiStarkKeygenBuilder::with_preprocessed_committer(
            config,
            Arc::new(GpuPreprocessedCommitter::new(device)),
        );
        for air in self.airs() {
            builder.add_required_air(air);
        }
        let proving_key = builder
            .generate_pk()
            .map_err(|error| ReducedSwirlWrapperSystemError::Keygen(error.to_string()))?;
        let verifying_key = proving_key.get_vk();
        let keys = ReducedSwirlTransitionLeafCudaKeys {
            binding: self.binding().clone(),
            proving_key: Arc::new(proving_key),
            verifying_key: Arc::new(verifying_key),
        };
        validate_keys(self, &keys)?;
        Ok(keys)
    }
}

#[cfg(feature = "cuda")]
#[derive(Clone)]
pub struct ReducedSwirlTransitionLeafCudaKeys {
    binding: ReducedSwirlTransitionLeafBinding,
    proving_key: Arc<MultiStarkProvingKey<SC>>,
    verifying_key: Arc<MultiStarkVerifyingKey<SC>>,
}

#[cfg(feature = "cuda")]
impl ReducedSwirlTransitionLeafCudaKeys {
    #[must_use]
    pub fn proving_key(&self) -> Arc<MultiStarkProvingKey<SC>> {
        Arc::clone(&self.proving_key)
    }

    #[must_use]
    pub fn verifying_key(&self) -> Arc<MultiStarkVerifyingKey<SC>> {
        Arc::clone(&self.verifying_key)
    }
}

/// Compact result retained after all source and VACC witness matrices for one
/// transition have been released.
#[cfg(feature = "cuda")]
pub struct ReducedSwirlTransitionLeafProof {
    pub proof: Proof<SC>,
    pub chain_after: Digest,
    pub source_end: u32,
    pub call_end: u32,
    /// Private opening of the synthetic input root. Only the first leaf's
    /// value is retained by the streaming orchestrator for the finalizer.
    pub initial_state: ReducedSwirlTransitionState,
    /// Private opening of the synthetic output root. Each new leaf replaces
    /// the preceding value, so storage remains constant in execution depth.
    pub final_state: ReducedSwirlTransitionState,
}

/// Persistent CUDA prover for a sequence of bounded transition leaves.
///
/// The device proving key and engine survive across calls. Source verifier
/// matrices arrive resident on the same CUDA device; a caller using another
/// stream must establish an event dependency before calling [`Self::prove_packet`].
/// Only the bounded VACC contexts and three tiny core tables cross from host
/// to device.
#[cfg(feature = "cuda")]
pub struct ReducedSwirlTransitionLeafCudaProver<
    const CAPACITY: usize = REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY,
> {
    system: Arc<ReducedSwirlTransitionLeafCudaSystem<CAPACITY>>,
    keys: ReducedSwirlTransitionLeafCudaKeys,
    device_key: DeviceMultiStarkProvingKey<GpuBackend>,
    engine: BabyBearPoseidon2GpuEngine,
}

#[cfg(feature = "cuda")]
impl<const CAPACITY: usize> ReducedSwirlTransitionLeafCudaProver<CAPACITY> {
    pub fn new(
        system: Arc<ReducedSwirlTransitionLeafCudaSystem<CAPACITY>>,
    ) -> Result<Self, ReducedSwirlWrapperSystemError> {
        let mut engine = BabyBearPoseidon2GpuEngine::new(system.params.clone());
        engine.device_mut().prover_config_mut().compile_monomials = false;
        let keys = system.keygen_cuda(engine.device())?;
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
            system,
            keys,
            device_key,
            engine,
        })
    }

    #[must_use]
    pub fn keys(&self) -> &ReducedSwirlTransitionLeafCudaKeys {
        &self.keys
    }

    #[must_use]
    pub fn engine(&self) -> &BabyBearPoseidon2GpuEngine {
        &self.engine
    }

    /// Prove one canonical transition and consume both witness packets.
    ///
    /// `source_packet` must have been generated on the same device and be
    /// ordered before this prover's stream. The source and
    /// VACC typed receipts are joined by the transition boundary AIR;
    /// `chain_before` is authenticated into both input and output state
    /// digests.
    pub fn prove_packet(
        &mut self,
        source_packet: ReducedSwirlSourceReceiptCudaPacket,
        vacc_packet: ReducedSwirlVaccTransitionCpuPacket,
        chain_before: Digest,
    ) -> Result<ReducedSwirlTransitionLeafProof, ReducedSwirlWrapperSystemError> {
        validate_keys(self.system.as_ref(), &self.keys)?;
        let source_message = source_receipt_message(
            self.system.circuit.components.source.as_ref(),
            &source_packet,
        )?;
        let record = ReducedSwirlTransitionLeafRecord {
            source: source_message,
            transition: vacc_packet.receipt,
            chain_before,
        };
        let core = generate_reduced_swirl_transition_leaf_core_traces(
            self.system.binding(),
            self.system.circuit.boundary_air.as_ref(),
            &record,
        )
        .map_err(ReducedSwirlWrapperSystemError::Binding)?;

        let source_air_count = self.system.circuit.components.source_air_count();
        let vacc_air_count = self.system.circuit.components.vacc_air_count();
        if source_packet.contexts.len() != source_air_count
            || source_packet
                .contexts
                .iter()
                .enumerate()
                .any(|(expected, (actual, _))| expected != *actual)
            || vacc_packet.contexts.len() != vacc_air_count
        {
            return Err(ReducedSwirlWrapperSystemError::Context(
                "transition-leaf component context inventory",
            ));
        }

        let airs = self.system.airs();
        let boundary_poseidon = self
            .system
            .circuit
            .components
            .boundary_poseidon()
            .build_poseidon2_trace_gpu(
                Vec::new(),
                core.derived.compression_inputs.clone(),
                None,
                &self.engine.device().device_ctx,
            )
            .ok_or(ReducedSwirlWrapperSystemError::Context(
                "transition-leaf boundary Poseidon trace",
            ))?;

        let indexed_vacc = vacc_packet
            .contexts
            .into_iter()
            .enumerate()
            .collect::<Vec<_>>();
        let device_vacc =
            transport_reduced_swirl_contexts_to_cuda(self.engine.device(), indexed_vacc)?;
        let expected = expected_public_values(
            core.verifier_public_values.clone(),
            core.vm_public_values.clone(),
            self.system.circuit.components.component_air_count(),
        );
        let device = self.engine.device();
        let mut per_trace = vec![
            (
                0,
                AirProvingContext::new(
                    Vec::new(),
                    <GpuDevice as DeviceDataTransporter<SC, GpuBackend>>::
                        transport_row_major_matrix_to_device(device, &core.verifier_pvs),
                    core.verifier_public_values,
                ),
            ),
            (
                1,
                AirProvingContext::new(
                    Vec::new(),
                    <GpuDevice as DeviceDataTransporter<SC, GpuBackend>>::
                        transport_row_major_matrix_to_device(device, &core.vm_pvs),
                    core.vm_public_values,
                ),
            ),
            (
                2,
                AirProvingContext::new(
                    Vec::new(),
                    <GpuDevice as DeviceDataTransporter<SC, GpuBackend>>::
                        transport_row_major_matrix_to_device(device, &core.boundary),
                    Vec::new(),
                ),
            ),
        ];
        for (component_air, context) in source_packet.contexts {
            let absolute_air = component_air + 3;
            validate_component_context(&airs[absolute_air], &context)?;
            per_trace.push((absolute_air, context));
        }
        for (component_air, context) in device_vacc {
            let absolute_air = 3 + source_air_count + component_air;
            validate_component_context(&airs[absolute_air], &context)?;
            per_trace.push((absolute_air, context));
        }
        let boundary_poseidon_air = 3 + source_air_count + vacc_air_count;
        let boundary_poseidon = AirProvingContext::simple_no_pis(boundary_poseidon);
        validate_component_context(&airs[boundary_poseidon_air], &boundary_poseidon)?;
        per_trace.push((boundary_poseidon_air, boundary_poseidon));

        let context = ProvingContext::new(per_trace);
        #[cfg(debug_assertions)]
        if std::env::var("OPENVM_SKIP_DEBUG") != Ok(String::from("1")) {
            self.engine.debug(&airs, &context);
        }
        let proof = self
            .engine
            .prove(&self.device_key, context)
            .map_err(|error| ReducedSwirlWrapperSystemError::Prover(error.to_string()))?;
        if proof.public_values != expected {
            return Err(ReducedSwirlWrapperSystemError::PublicValues);
        }
        #[cfg(debug_assertions)]
        if std::env::var("OPENVM_SKIP_DEBUG") != Ok(String::from("1"))
            || std::env::var_os("OPENVM_REDUCED_SWIRL_VERIFY_TRANSITION_LEAF").is_some()
        {
            CpuEngine::new(self.keys.verifying_key.inner.params.clone())
                .verify(self.keys.verifying_key.as_ref(), &proof)
                .map_err(|error| ReducedSwirlWrapperSystemError::Verifier(error.to_string()))?;
        }
        Ok(ReducedSwirlTransitionLeafProof {
            proof,
            chain_after: core.derived.chain_after,
            source_end: core.derived.source_end.as_canonical_u32(),
            call_end: core.derived.call_end.as_canonical_u32(),
            initial_state: record.initial_state(),
            final_state: record.final_state(core.derived.chain_after),
        })
    }
}

#[cfg(feature = "cuda")]
fn source_receipt_message<const CAPACITY: usize>(
    component: &ProductionReducedSwirlSourceReceiptComponent<CAPACITY>,
    packet: &ReducedSwirlSourceReceiptCudaPacket,
) -> Result<ReducedSwirlSourceReceiptMessage<F>, ReducedSwirlWrapperSystemError> {
    let block = &packet.block;
    if block.sources.is_empty()
        || block.sources.len() > CAPACITY
        || block.manifest_digest.iter().all(|value| *value == F::ZERO)
    {
        return Err(ReducedSwirlWrapperSystemError::Context(
            "transition-leaf source packet capacity",
        ));
    }
    let count = u32::try_from(block.sources.len()).map_err(|_| {
        ReducedSwirlWrapperSystemError::Context("transition-leaf source count overflow")
    })?;
    for (local_index, source) in block.sources.iter().enumerate() {
        let global_index = block
            .source_offset
            .checked_add(u32::try_from(local_index).map_err(|_| {
                ReducedSwirlWrapperSystemError::Context("transition-leaf source index overflow")
            })?)
            .ok_or(ReducedSwirlWrapperSystemError::Context(
                "transition-leaf source index overflow",
            ))?;
        if source.segment_index != global_index {
            return Err(ReducedSwirlWrapperSystemError::Context(
                "transition-leaf source global index",
            ));
        }
        if let Some(previous) = local_index
            .checked_sub(1)
            .and_then(|index| block.sources.get(index))
        {
            if previous.vm.program_commitment != source.vm.program_commitment
                || previous.vm.final_pc != source.vm.initial_pc
                || previous.vm.final_root != source.vm.initial_root
                || previous.vm.is_terminate != F::ZERO
            {
                return Err(ReducedSwirlWrapperSystemError::Context(
                    "transition-leaf source VM continuity",
                ));
            }
        }
    }
    let first = block
        .sources
        .first()
        .ok_or(ReducedSwirlWrapperSystemError::Context(
            "transition-leaf empty source packet",
        ))?;
    let last = block
        .sources
        .last()
        .ok_or(ReducedSwirlWrapperSystemError::Context(
            "transition-leaf empty source packet",
        ))?;
    Ok(ReducedSwirlSourceReceiptMessage {
        protocol_digest: component.inner().receipt_air().profile.protocol_digest,
        manifest_digest: block.manifest_digest,
        source_offset: F::from_u32(block.source_offset),
        source_count: F::from_u32(count),
        program_commitment: first.vm.program_commitment,
        initial_pc: first.vm.initial_pc,
        initial_root: first.vm.initial_root,
        final_pc: last.vm.final_pc,
        final_root: last.vm.final_root,
        exit_code: last.vm.exit_code,
        is_terminate: last.vm.is_terminate,
    })
}

#[cfg(feature = "cuda")]
fn validate_keys<const CAPACITY: usize>(
    system: &ReducedSwirlTransitionLeafCudaSystem<CAPACITY>,
    keys: &ReducedSwirlTransitionLeafCudaKeys,
) -> Result<(), ReducedSwirlWrapperSystemError> {
    system.validate_air_inventory()?;
    let airs = system.airs();
    let config = SC::default_from_params(keys.verifying_key.inner.params.clone());
    if keys.binding != *system.binding()
        || keys.proving_key.params != system.params
        || keys.verifying_key.inner.params != system.params
        || keys.proving_key.per_air.len() != airs.len()
        || keys.verifying_key.inner.per_air.len() != airs.len()
        || keys.proving_key.vk_pre_hash != keys.verifying_key.pre_hash
        || !keys.verifying_key.has_consistent_pre_hash(&config)
    {
        return Err(ReducedSwirlWrapperSystemError::KeyIntegrity(
            "transition-leaf key",
        ));
    }
    for (vk, air) in keys.verifying_key.inner.per_air.iter().zip(airs) {
        if !vk.is_required
            || vk.params.num_public_values != air.num_public_values()
            || vk.params.width.common_main != air.common_main_width()
            || vk.params.width.cached_mains != air.cached_main_widths()
        {
            return Err(ReducedSwirlWrapperSystemError::KeyIntegrity(
                "transition-leaf AIR key shape",
            ));
        }
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn validate_component_context<PB: ProverBackend<Val = F>>(
    air: &AirRef<SC>,
    context: &AirProvingContext<PB>,
) -> Result<(), ReducedSwirlWrapperSystemError> {
    let height = context.common_main.height();
    if !context.public_values.is_empty()
        || height == 0
        || !height.is_power_of_two()
        || context.common_main.width() != air.common_main_width()
        || context.cached_mains.len() != air.cached_main_widths().len()
        || context
            .cached_mains
            .iter()
            .zip(air.cached_main_widths())
            .any(|(cached, width)| cached.trace.width() != width || cached.trace.height() != height)
    {
        return Err(ReducedSwirlWrapperSystemError::Context(
            "transition-leaf component trace shape",
        ));
    }
    Ok(())
}

#[cfg(feature = "cuda")]
fn expected_public_values(verifier: Vec<F>, vm: Vec<F>, component_air_count: usize) -> Vec<Vec<F>> {
    let mut expected = vec![verifier, vm, Vec::new()];
    expected.resize_with(3 + component_air_count, Vec::new);
    expected
}

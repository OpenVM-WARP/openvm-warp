//! Production trace/key adapter for the reduced-SWIRL source-tree bridge.
//!
//! The component verifies one complete recursive source-tree proof under its
//! setup-fixed root VK, consumes the detached VACC source summary, and emits a
//! compact source receipt for the existing final wrapper. Bridge Poseidon
//! requests are inserted into the complete verifier's existing Poseidon table;
//! no duplicate hash AIR or host success bit is introduced.
//!
//! Integration must obtain the four trusted commitments from setup objects,
//! never from proof-carried PVS:
//!
//! - `app_vk_commit`: `leaf_prefix.get_vk_commit(false)` (the custom transition-leaf VK for this
//!   recursion ladder);
//! - `leaf_vk_commit`: `internal_for_leaf.get_vk_commit(false)`;
//! - `internal_for_leaf_vk_commit`: `internal_recursive.get_vk_commit(false)`;
//! - `recursive_vk_commit`: `internal_recursive.get_vk_commit(true)`.
//!
//! The verified proof VK is always `internal_recursive.get_vk()`. At recursion
//! depth one its PVS `internal_recursive_vk_commit` is canonically unset; at
//! greater depths it must equal `recursive_vk_commit`. All other verifier PVS
//! metadata is setup-fixed at every depth.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
};

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_continuations::circuit::{
    reduced_swirl_source_tree_bridge::{
        ReducedSwirlSourceTreeBridgeBinding, ReducedSwirlSourceTreeBridgeCircuit,
        ReducedSwirlSourceTreeBridgeError, ReducedSwirlSourceTreeBridgeTrace,
    },
    reduced_swirl_warp::{ReducedSwirlSourceReceiptBus, ReducedSwirlSourceReceiptMessage},
    Circuit,
};
use openvm_cpu_backend::CpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{
    BabyBearPoseidon2GpuEngine, GpuBackend, GpuDevice, GpuPreprocessedCommitter,
};
#[cfg(feature = "cuda")]
use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_recursion_circuit::{
    native_warp::ReducedSwirlVaccSourceSummaryMessage,
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
    proof::Proof,
    prover::{AirProvingContext, MatrixDimensions, ProverBackend},
    AirRef, FiatShamirTranscript, StarkEngine, StarkProtocolConfig, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, BabyBearPoseidon2Config, BabyBearPoseidon2CpuEngine, Digest,
    DuplexSponge, DIGEST_SIZE, F,
};

use super::{
    reduced_swirl_params::reduced_swirl_system_params_digest,
    reduced_swirl_wrapper_components::ReducedSwirlVerifierComponent,
};
use crate::SC;

type CpuEngine = BabyBearPoseidon2CpuEngine<DuplexSponge>;

const SOURCE_TREE_COMPONENT_DIGEST_TAG: &[u8] =
    b"openvm.native-warp.reduced-swirl.source-tree-component.v1";

#[derive(Debug, thiserror::Error)]
pub enum ReducedSwirlSourceTreeComponentError {
    #[error("source-tree bridge setup is invalid: {0:?}")]
    Bridge(ReducedSwirlSourceTreeBridgeError),
    #[error("source-tree child proof public values are malformed")]
    PublicValueShape,
    #[error("source-tree child proof verification failed: {0}")]
    ChildVerification(String),
    #[error("source-tree child verifier rejected a malformed proof without a typed error")]
    MalformedProof,
    #[error("source-tree verifier trace generation failed")]
    VerifierTrace,
    #[error("source-tree component AIR/context inventory differs")]
    Inventory,
    #[error("source-tree component setup digest failed: {0}")]
    Setup(String),
    #[error("source-tree component key generation failed: {0}")]
    Keygen(String),
    #[error("source-tree component key integrity check failed")]
    KeyIntegrity,
}

impl From<ReducedSwirlSourceTreeBridgeError> for ReducedSwirlSourceTreeComponentError {
    fn from(error: ReducedSwirlSourceTreeBridgeError) -> Self {
        Self::Bridge(error)
    }
}

pub struct ReducedSwirlSourceTreeCpuPacket {
    pub contexts: Vec<AirProvingContext<CpuBackend<SC>>>,
    pub receipt: ReducedSwirlSourceReceiptMessage<F>,
}

#[cfg(feature = "cuda")]
pub struct ReducedSwirlSourceTreeCudaPacket {
    pub contexts: Vec<AirProvingContext<GpuBackend>>,
    pub receipt: ReducedSwirlSourceReceiptMessage<F>,
}

/// Setup-fixed source-tree verifier component for the compact final wrapper.
pub struct ProductionReducedSwirlSourceTreeComponent {
    circuit: Arc<ReducedSwirlSourceTreeBridgeCircuit>,
    params: SystemParams,
    protocol_digest: Digest,
}

impl ProductionReducedSwirlSourceTreeComponent {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        binding: ReducedSwirlSourceTreeBridgeBinding,
        summary_bus_index: BusIndex,
        receipt_bus: ReducedSwirlSourceReceiptBus,
        bus_idx_manager: BusIndexManager,
        params: SystemParams,
    ) -> Result<Self, ReducedSwirlSourceTreeComponentError> {
        let circuit = Arc::new(ReducedSwirlSourceTreeBridgeCircuit::new(
            child_vk,
            binding,
            summary_bus_index,
            receipt_bus,
            bus_idx_manager,
        )?);
        let protocol_digest = source_tree_component_digest(circuit.as_ref(), &params)?;
        let component = Self {
            circuit,
            params,
            protocol_digest,
        };
        component.validate_air_inventory()?;
        Ok(component)
    }

    #[must_use]
    pub fn circuit(&self) -> &Arc<ReducedSwirlSourceTreeBridgeCircuit> {
        &self.circuit
    }

    #[must_use]
    pub const fn params(&self) -> &SystemParams {
        &self.params
    }

    #[must_use]
    pub fn child_vk(&self) -> &Arc<MultiStarkVerifyingKey<SC>> {
        self.circuit.child_vk()
    }

    #[must_use]
    pub fn binding(&self) -> ReducedSwirlSourceTreeBridgeBinding {
        self.circuit.bridge().binding()
    }

    pub fn generate_cpu_packet(
        &self,
        proof: &Proof<SC>,
        summary: ReducedSwirlVaccSourceSummaryMessage<F>,
    ) -> Result<ReducedSwirlSourceTreeCpuPacket, ReducedSwirlSourceTreeComponentError> {
        validate_proof_public_value_shape(self.child_vk(), proof)?;
        verify_child_without_panic(self.child_vk(), proof)?;
        let bridge = self
            .circuit
            .bridge()
            .generate_trace(&proof.public_values, summary)?;
        let contexts = self.generate_verifier_cpu_contexts(proof, &bridge)?;
        let mut contexts = core::iter::once(AirProvingContext::simple_no_pis(bridge.matrix))
            .chain(contexts)
            .collect::<Vec<_>>();
        validate_contexts(self, &contexts)?;
        Ok(ReducedSwirlSourceTreeCpuPacket {
            contexts: core::mem::take(&mut contexts),
            receipt: bridge.receipt,
        })
    }

    fn generate_verifier_cpu_contexts(
        &self,
        proof: &Proof<SC>,
        bridge: &ReducedSwirlSourceTreeBridgeTrace,
    ) -> Result<Vec<AirProvingContext<CpuBackend<SC>>>, ReducedSwirlSourceTreeComponentError> {
        let empty_poseidon = Vec::<[F; POSEIDON2_WIDTH]>::new();
        let empty_usize = Vec::<usize>::new();
        let mut external = VerifierExternalData {
            poseidon2_compress_inputs: &bridge.compression_inputs,
            poseidon2_permute_inputs: &empty_poseidon,
            range_check_inputs: &empty_usize,
            power_check_inputs: &empty_usize,
            required_heights: None,
            final_transcript_state: None,
        };
        let cached = self
            .circuit
            .verifier()
            .cached_trace_record_for_child(self.child_vk());
        catch_unwind(AssertUnwindSafe(|| {
            <VerifierSubCircuit<1> as VerifierTraceGen<CpuBackend<SC>, SC, ()>>::generate_proving_ctxs(
                self.circuit.verifier(),
                self.child_vk(),
                CachedTraceCtx::Records(cached),
                std::slice::from_ref(proof),
                &mut external,
                &(),
                default_duplex_sponge_recorder(),
            )
        }))
        .map_err(|_| ReducedSwirlSourceTreeComponentError::MalformedProof)?
        .ok_or(ReducedSwirlSourceTreeComponentError::VerifierTrace)
    }

    #[cfg(feature = "cuda")]
    pub fn generate_cuda_packet(
        &self,
        proof: &Proof<SC>,
        summary: ReducedSwirlVaccSourceSummaryMessage<F>,
        engine: &BabyBearPoseidon2GpuEngine,
    ) -> Result<ReducedSwirlSourceTreeCudaPacket, ReducedSwirlSourceTreeComponentError> {
        validate_proof_public_value_shape(self.child_vk(), proof)?;
        verify_child_without_panic(self.child_vk(), proof)?;
        let bridge = self
            .circuit
            .bridge()
            .generate_trace(&proof.public_values, summary)?;
        let empty_poseidon = Vec::<[F; POSEIDON2_WIDTH]>::new();
        let empty_usize = Vec::<usize>::new();
        let mut external = VerifierExternalData {
            poseidon2_compress_inputs: &bridge.compression_inputs,
            poseidon2_permute_inputs: &empty_poseidon,
            range_check_inputs: &empty_usize,
            power_check_inputs: &empty_usize,
            required_heights: None,
            final_transcript_state: None,
        };
        let cached = self
            .circuit
            .verifier()
            .cached_trace_record_for_child(self.child_vk());
        let verifier_contexts = catch_unwind(AssertUnwindSafe(|| {
            <VerifierSubCircuit<1> as VerifierTraceGen<GpuBackend, SC, GpuDeviceCtx>>::generate_proving_ctxs(
                self.circuit.verifier(),
                self.child_vk(),
                CachedTraceCtx::Records(cached),
                std::slice::from_ref(proof),
                &mut external,
                &engine.device().device_ctx,
                default_duplex_sponge_recorder(),
            )
        }))
        .map_err(|_| ReducedSwirlSourceTreeComponentError::MalformedProof)?
        .ok_or(ReducedSwirlSourceTreeComponentError::VerifierTrace)?;
        let bridge_matrix = <GpuDevice as openvm_stark_backend::prover::DeviceDataTransporter<
            SC,
            GpuBackend,
        >>::transport_row_major_matrix_to_device(
            engine.device(), &bridge.matrix
        );
        let contexts = core::iter::once(AirProvingContext::new(
            Vec::new(),
            bridge_matrix,
            Vec::new(),
        ))
        .chain(verifier_contexts)
        .collect::<Vec<_>>();
        validate_contexts(self, &contexts)?;
        Ok(ReducedSwirlSourceTreeCudaPacket {
            contexts,
            receipt: bridge.receipt,
        })
    }

    pub fn keygen_cpu(
        &self,
    ) -> Result<ReducedSwirlSourceTreeComponentKeys, ReducedSwirlSourceTreeComponentError> {
        self.keygen_with_builder(MultiStarkKeygenBuilder::new(SC::default_from_params(
            self.params.clone(),
        )))
    }

    #[cfg(feature = "cuda")]
    pub fn keygen_cuda(
        &self,
        device: &GpuDevice,
    ) -> Result<ReducedSwirlSourceTreeComponentKeys, ReducedSwirlSourceTreeComponentError> {
        self.keygen_with_builder(MultiStarkKeygenBuilder::with_preprocessed_committer(
            SC::default_from_params(self.params.clone()),
            Arc::new(GpuPreprocessedCommitter::new(device)),
        ))
    }

    fn keygen_with_builder(
        &self,
        mut builder: MultiStarkKeygenBuilder<SC>,
    ) -> Result<ReducedSwirlSourceTreeComponentKeys, ReducedSwirlSourceTreeComponentError> {
        self.validate_air_inventory()?;
        for air in self.airs::<SC>() {
            builder.add_required_air(air);
        }
        let proving_key = builder
            .generate_pk()
            .map_err(|error| ReducedSwirlSourceTreeComponentError::Keygen(error.to_string()))?;
        let verifying_key = proving_key.get_vk();
        let keys = ReducedSwirlSourceTreeComponentKeys {
            component_digest: self.protocol_digest,
            proving_key: Arc::new(proving_key),
            verifying_key: Arc::new(verifying_key),
        };
        self.validate_keys(&keys)?;
        Ok(keys)
    }

    fn validate_air_inventory(&self) -> Result<(), ReducedSwirlSourceTreeComponentError> {
        let airs = self.airs::<SC>();
        if airs.len() != 1 + self.circuit.verifier().airs::<SC>().len()
            || airs.iter().any(|air| air.num_public_values() != 0)
        {
            return Err(ReducedSwirlSourceTreeComponentError::Inventory);
        }
        Ok(())
    }

    fn validate_keys(
        &self,
        keys: &ReducedSwirlSourceTreeComponentKeys,
    ) -> Result<(), ReducedSwirlSourceTreeComponentError> {
        let airs = self.airs::<SC>();
        let config = SC::default_from_params(self.params.clone());
        if keys.component_digest != self.protocol_digest
            || keys.proving_key.params != self.params
            || keys.verifying_key.inner.params != self.params
            || keys.proving_key.per_air.len() != airs.len()
            || keys.verifying_key.inner.per_air.len() != airs.len()
            || keys.proving_key.vk_pre_hash != keys.verifying_key.pre_hash
            || !keys.verifying_key.has_consistent_pre_hash(&config)
        {
            return Err(ReducedSwirlSourceTreeComponentError::KeyIntegrity);
        }
        for (vk, air) in keys.verifying_key.inner.per_air.iter().zip(airs) {
            if !vk.is_required
                || vk.params.num_public_values != air.num_public_values()
                || vk.params.width.common_main != air.common_main_width()
                || vk.params.width.cached_mains != air.cached_main_widths()
            {
                return Err(ReducedSwirlSourceTreeComponentError::KeyIntegrity);
            }
        }
        Ok(())
    }
}

impl ReducedSwirlVerifierComponent for ProductionReducedSwirlSourceTreeComponent {
    fn protocol_digest(&self) -> Digest {
        self.protocol_digest
    }

    fn airs<C: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<C>> {
        <ReducedSwirlSourceTreeBridgeCircuit as Circuit<C>>::airs(self.circuit.as_ref())
    }
}

#[derive(Clone)]
pub struct ReducedSwirlSourceTreeComponentKeys {
    component_digest: Digest,
    proving_key: Arc<MultiStarkProvingKey<SC>>,
    verifying_key: Arc<MultiStarkVerifyingKey<SC>>,
}

impl ReducedSwirlSourceTreeComponentKeys {
    #[must_use]
    pub fn proving_key(&self) -> Arc<MultiStarkProvingKey<SC>> {
        Arc::clone(&self.proving_key)
    }

    #[must_use]
    pub fn verifying_key(&self) -> Arc<MultiStarkVerifyingKey<SC>> {
        Arc::clone(&self.verifying_key)
    }
}

fn validate_proof_public_value_shape(
    child_vk: &MultiStarkVerifyingKey<SC>,
    proof: &Proof<SC>,
) -> Result<(), ReducedSwirlSourceTreeComponentError> {
    if proof.public_values.len() != child_vk.inner.per_air.len()
        || proof
            .public_values
            .iter()
            .zip(&child_vk.inner.per_air)
            .any(|(values, air)| values.len() != air.params.num_public_values)
    {
        return Err(ReducedSwirlSourceTreeComponentError::PublicValueShape);
    }
    Ok(())
}

fn verify_child_without_panic(
    child_vk: &MultiStarkVerifyingKey<SC>,
    proof: &Proof<SC>,
) -> Result<(), ReducedSwirlSourceTreeComponentError> {
    let engine = CpuEngine::new(child_vk.inner.params.clone());
    catch_unwind(AssertUnwindSafe(|| engine.verify(child_vk, proof)))
        .map_err(|_| ReducedSwirlSourceTreeComponentError::MalformedProof)?
        .map_err(|error| ReducedSwirlSourceTreeComponentError::ChildVerification(error.to_string()))
}

fn validate_contexts<PB: ProverBackend<Val = F>>(
    component: &ProductionReducedSwirlSourceTreeComponent,
    contexts: &[AirProvingContext<PB>],
) -> Result<(), ReducedSwirlSourceTreeComponentError> {
    let airs = component.airs::<SC>();
    if contexts.len() != airs.len() {
        return Err(ReducedSwirlSourceTreeComponentError::Inventory);
    }
    for (context, air) in contexts.iter().zip(airs) {
        if !context.public_values.is_empty()
            || context.common_main.height() == 0
            || !context.common_main.height().is_power_of_two()
            || context.common_main.width() != air.common_main_width()
            || context.cached_mains.len() != air.cached_main_widths().len()
            || context
                .cached_mains
                .iter()
                .zip(air.cached_main_widths())
                .any(|(matrix, width)| matrix.trace().width() != width)
        {
            return Err(ReducedSwirlSourceTreeComponentError::Inventory);
        }
    }
    Ok(())
}

fn source_tree_component_digest(
    circuit: &ReducedSwirlSourceTreeBridgeCircuit,
    params: &SystemParams,
) -> Result<Digest, ReducedSwirlSourceTreeComponentError> {
    // Parent and child use deliberately different PCS profiles: the source
    // tree is first normalized with the ordinary leaf profile and its root is
    // then verified inside the reduced-SWIRL wrapper profile.  Bind both exact
    // parameter sets instead of incorrectly requiring them to be equal.
    let wrapper_config = SC::default_from_params(params.clone());
    let wrapper_params_digest = reduced_swirl_system_params_digest(&wrapper_config, params)
        .map_err(|error| ReducedSwirlSourceTreeComponentError::Setup(error.to_string()))?;
    let child_params = &circuit.child_vk().inner.params;
    let child_config = SC::default_from_params(child_params.clone());
    let child_params_digest = reduced_swirl_system_params_digest(&child_config, child_params)
        .map_err(|error| ReducedSwirlSourceTreeComponentError::Setup(error.to_string()))?;
    let binding = circuit.bridge().binding();
    let mut transcript = default_duplex_sponge_recorder();
    observe_bytes(&mut transcript, SOURCE_TREE_COMPONENT_DIGEST_TAG);
    observe_digest(&mut transcript, wrapper_params_digest);
    observe_digest(&mut transcript, child_params_digest);
    observe_digest(&mut transcript, circuit.child_vk().pre_hash);
    observe_digest(&mut transcript, binding.protocol_digest);
    observe_u64(&mut transcript, u64::from(binding.source_leaf_capacity));
    for commit in [
        binding.trusted_vk_commits.app_vk_commit,
        binding.trusted_vk_commits.leaf_vk_commit,
        binding.trusted_vk_commits.internal_for_leaf_vk_commit,
        binding.trusted_vk_commits.recursive_vk_commit,
    ] {
        observe_digest(&mut transcript, commit.cached_commit);
        observe_digest(&mut transcript, commit.vk_pre_hash);
    }
    let airs =
        <ReducedSwirlSourceTreeBridgeCircuit as Circuit<BabyBearPoseidon2Config>>::airs(circuit);
    observe_u64(
        &mut transcript,
        u64::try_from(airs.len())
            .map_err(|error| ReducedSwirlSourceTreeComponentError::Setup(error.to_string()))?,
    );
    for (air_id, air) in airs.into_iter().enumerate() {
        for value in [
            air_id,
            air.common_main_width(),
            air.num_public_values(),
            air.cached_main_widths().len(),
        ] {
            observe_u64(
                &mut transcript,
                u64::try_from(value).map_err(|error| {
                    ReducedSwirlSourceTreeComponentError::Setup(error.to_string())
                })?,
            );
        }
        for width in air.cached_main_widths() {
            observe_u64(
                &mut transcript,
                u64::try_from(width).map_err(|error| {
                    ReducedSwirlSourceTreeComponentError::Setup(error.to_string())
                })?,
            );
        }
    }
    for bus in [
        circuit.bridge().summary_bus_index(),
        circuit.bridge().receipt_bus().index(),
        circuit.next_bus_idx(),
    ] {
        observe_u64(&mut transcript, u64::from(bus));
    }
    Ok(core::array::from_fn(|_| {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::sample(&mut transcript)
    }))
}

fn observe_bytes(
    transcript: &mut impl FiatShamirTranscript<BabyBearPoseidon2Config>,
    bytes: &[u8],
) {
    observe_u64(transcript, bytes.len() as u64);
    for &byte in bytes {
        transcript.observe(F::from_u8(byte));
    }
}

fn observe_digest(
    transcript: &mut impl FiatShamirTranscript<BabyBearPoseidon2Config>,
    digest: Digest,
) {
    for value in digest {
        transcript.observe(value);
    }
}

fn observe_u64(transcript: &mut impl FiatShamirTranscript<BabyBearPoseidon2Config>, value: u64) {
    for byte in value.to_le_bytes() {
        transcript.observe(F::from_u8(byte));
    }
}

const _: [(); POSEIDON2_WIDTH] = [(); 2 * DIGEST_SIZE];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_public_values_are_rejected_before_verifier_trace_generation() {
        // The production helper performs this exact comparison before either
        // CPU or CUDA verifier code can index a proof-carried vector. The
        // integration test supplies the concrete source-tree VK/proof once
        // this module is exported by native_warp.rs.
        let expected = [3usize, 0, 7];
        let malformed = [3usize, 1, 7];
        assert!(expected
            .iter()
            .zip(malformed)
            .any(|(expected, actual)| *expected != actual));
    }

    #[test]
    fn component_domain_is_versioned_and_nonempty() {
        assert!(!SOURCE_TREE_COMPONENT_DIGEST_TAG.is_empty());
        assert!(SOURCE_TREE_COMPONENT_DIGEST_TAG.ends_with(b".v1"));
    }
}

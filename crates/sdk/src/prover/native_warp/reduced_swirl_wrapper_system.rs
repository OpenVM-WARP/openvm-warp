//! Generic keyed runtime for the active-prefix reduced-SWIRL WARP wrapper.
//!
//! Protocol-specific components provide genuine deferred-prefix, VACC-chain,
//! and terminal verifier traces. This layer only performs MultiSTARK keygen,
//! shape validation, proving, and verification. Repeated component AIR IDs are
//! accepted so one setup-fixed verifier inventory can process any active
//! prefix up to the key capacity without cloning AIR definitions per segment.

use core::mem::size_of;
use std::{collections::BTreeSet, sync::Arc};

use openvm_continuations::circuit::{
    reduced_swirl_warp::{
        generate_reduced_swirl_wrapper_core_traces, ReducedSwirlWrapperBinding,
        ReducedSwirlWrapperCircuit, ReducedSwirlWrapperRecord,
        ReducedSwirlWrapperVerifierComponents,
    },
    Circuit,
};
use openvm_cpu_backend::CpuBackend;
use openvm_stark_backend::{
    keygen::{
        types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
        MultiStarkKeygenBuilder,
    },
    proof::Proof,
    prover::{
        AirProvingContext, DeviceDataTransporter, DeviceMultiStarkProvingKey, MatrixDimensions,
        ProverBackend, ProvingContext,
    },
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2CpuEngine, DuplexSponge, F};
use openvm_verify_stark_host::pvs::{VerifierBasePvs, VmPvs};

use crate::SC;

type CpuEngine = BabyBearPoseidon2CpuEngine<DuplexSponge>;

#[derive(Debug, thiserror::Error)]
pub enum ReducedSwirlWrapperSystemError {
    #[error("invalid reduced-SWIRL wrapper binding: {0}")]
    Binding(&'static str),
    #[error("reduced-SWIRL wrapper keygen failed: {0}")]
    Keygen(String),
    #[error("reduced-SWIRL wrapper key integrity failed: {0}")]
    KeyIntegrity(&'static str),
    #[error("reduced-SWIRL wrapper component context error: {0}")]
    Context(&'static str),
    #[error("reduced-SWIRL wrapper prover failed: {0}")]
    Prover(String),
    #[error("reduced-SWIRL wrapper verifier failed: {0}")]
    Verifier(String),
    #[error("reduced-SWIRL wrapper public values differ")]
    PublicValues,
}

/// Authenticated component traces in circuit-local AIR coordinates. The same
/// `air_index` may occur more than once; OpenVM/SWIRL stacks those traces under
/// one verifying-key entry.
pub struct ReducedSwirlWrapperWitness<PB: ProverBackend<Val = F>> {
    pub component_contexts: Vec<(usize, AirProvingContext<PB>)>,
}

pub struct ReducedSwirlWrapperSystem<C: ReducedSwirlWrapperVerifierComponents> {
    circuit: Arc<ReducedSwirlWrapperCircuit<C>>,
    params: SystemParams,
}

impl<C: ReducedSwirlWrapperVerifierComponents> ReducedSwirlWrapperSystem<C> {
    pub fn new(
        binding: ReducedSwirlWrapperBinding,
        components: Arc<C>,
        params: SystemParams,
    ) -> Result<Self, ReducedSwirlWrapperSystemError> {
        let circuit = ReducedSwirlWrapperCircuit::new(binding, components)
            .map_err(ReducedSwirlWrapperSystemError::Binding)?;
        let system = Self {
            circuit: Arc::new(circuit),
            params,
        };
        system.validate_air_inventory()?;
        Ok(system)
    }

    #[must_use]
    pub fn binding(&self) -> &ReducedSwirlWrapperBinding {
        &self.circuit.binding
    }

    #[must_use]
    pub fn circuit(&self) -> Arc<ReducedSwirlWrapperCircuit<C>> {
        Arc::clone(&self.circuit)
    }

    #[must_use]
    pub fn component_air_count(&self) -> usize {
        self.circuit.components.component_air_count()
    }

    #[must_use]
    pub fn airs(&self) -> Vec<openvm_stark_backend::AirRef<SC>> {
        <ReducedSwirlWrapperCircuit<C> as Circuit<SC>>::airs(self.circuit.as_ref())
    }

    fn validate_air_inventory(&self) -> Result<(), ReducedSwirlWrapperSystemError> {
        let airs = self.airs();
        if airs.len() != 3 + self.component_air_count()
            || airs[0].num_public_values() != size_of::<VerifierBasePvs<u8>>()
            || airs[1].num_public_values() != size_of::<VmPvs<u8>>()
            || airs[2..].iter().any(|air| air.num_public_values() != 0)
        {
            return Err(ReducedSwirlWrapperSystemError::KeyIntegrity(
                "AIR inventory",
            ));
        }
        Ok(())
    }

    pub fn keygen(&self) -> Result<ReducedSwirlWrapperKeys, ReducedSwirlWrapperSystemError> {
        self.validate_air_inventory()?;
        let config = SC::default_from_params(self.params.clone());
        let mut builder = MultiStarkKeygenBuilder::new(config);
        for air in self.airs() {
            builder.add_required_air(air);
        }
        let proving_key = builder
            .generate_pk()
            .map_err(|error| ReducedSwirlWrapperSystemError::Keygen(error.to_string()))?;
        let verifying_key = proving_key.get_vk();
        let keys = ReducedSwirlWrapperKeys {
            binding: self.binding().clone(),
            component_air_count: self.component_air_count(),
            proving_key: Arc::new(proving_key),
            verifying_key: Arc::new(verifying_key),
        };
        validate_keys(self, &keys)?;
        Ok(keys)
    }
}

#[derive(Clone)]
pub struct ReducedSwirlWrapperKeys {
    binding: ReducedSwirlWrapperBinding,
    component_air_count: usize,
    proving_key: Arc<MultiStarkProvingKey<SC>>,
    verifying_key: Arc<MultiStarkVerifyingKey<SC>>,
}

impl ReducedSwirlWrapperKeys {
    #[must_use]
    pub fn proving_key(&self) -> Arc<MultiStarkProvingKey<SC>> {
        Arc::clone(&self.proving_key)
    }
    #[must_use]
    pub fn verifying_key(&self) -> Arc<MultiStarkVerifyingKey<SC>> {
        Arc::clone(&self.verifying_key)
    }
}

pub struct ReducedSwirlWrapperCpuProver<C: ReducedSwirlWrapperVerifierComponents> {
    system: Arc<ReducedSwirlWrapperSystem<C>>,
    keys: ReducedSwirlWrapperKeys,
    device_key: DeviceMultiStarkProvingKey<CpuBackend<SC>>,
}

impl<C: ReducedSwirlWrapperVerifierComponents> ReducedSwirlWrapperCpuProver<C> {
    pub fn new(
        system: Arc<ReducedSwirlWrapperSystem<C>>,
    ) -> Result<Self, ReducedSwirlWrapperSystemError> {
        let keys = system.keygen()?;
        Self::from_keys(system, keys)
    }

    pub fn from_keys(
        system: Arc<ReducedSwirlWrapperSystem<C>>,
        keys: ReducedSwirlWrapperKeys,
    ) -> Result<Self, ReducedSwirlWrapperSystemError> {
        validate_keys(system.as_ref(), &keys)?;
        let engine = CpuEngine::new(keys.proving_key.params.clone());
        let device_key = engine
            .device()
            .transport_pk_to_device(keys.proving_key.as_ref());
        Ok(Self {
            system,
            keys,
            device_key,
        })
    }

    #[must_use]
    pub fn keys(&self) -> &ReducedSwirlWrapperKeys {
        &self.keys
    }

    pub fn prove(
        &self,
        record: &ReducedSwirlWrapperRecord,
        witness: ReducedSwirlWrapperWitness<CpuBackend<SC>>,
    ) -> Result<Proof<SC>, ReducedSwirlWrapperSystemError> {
        let (context, expected) = assemble_cpu_context(self.system.as_ref(), record, witness)?;
        let engine = CpuEngine::new(self.keys.proving_key.params.clone());
        let proof = engine
            .prove(&self.device_key, context)
            .map_err(|error| ReducedSwirlWrapperSystemError::Prover(error.to_string()))?;
        if proof.public_values != expected {
            return Err(ReducedSwirlWrapperSystemError::PublicValues);
        }
        engine
            .verify(self.keys.verifying_key.as_ref(), &proof)
            .map_err(|error| ReducedSwirlWrapperSystemError::Verifier(error.to_string()))?;
        Ok(proof)
    }

    pub fn verify(
        &self,
        record: &ReducedSwirlWrapperRecord,
        proof: &Proof<SC>,
    ) -> Result<(), ReducedSwirlWrapperSystemError> {
        let expected = expected_public_values(self.system.as_ref(), record)?;
        if proof.public_values != expected {
            return Err(ReducedSwirlWrapperSystemError::PublicValues);
        }
        CpuEngine::new(self.keys.verifying_key.inner.params.clone())
            .verify(self.keys.verifying_key.as_ref(), proof)
            .map_err(|error| ReducedSwirlWrapperSystemError::Verifier(error.to_string()))
    }
}

fn validate_keys<C: ReducedSwirlWrapperVerifierComponents>(
    system: &ReducedSwirlWrapperSystem<C>,
    keys: &ReducedSwirlWrapperKeys,
) -> Result<(), ReducedSwirlWrapperSystemError> {
    let expected = 3 + system.component_air_count();
    let config = SC::default_from_params(keys.verifying_key.inner.params.clone());
    if keys.binding != *system.binding()
        || keys.component_air_count != system.component_air_count()
        || keys.proving_key.params != system.params
        || keys.verifying_key.inner.params != system.params
        || keys.proving_key.per_air.len() != expected
        || keys.verifying_key.inner.per_air.len() != expected
        || keys.proving_key.vk_pre_hash != keys.verifying_key.pre_hash
        || !keys.verifying_key.has_consistent_pre_hash(&config)
        || keys
            .verifying_key
            .inner
            .per_air
            .iter()
            .any(|air| !air.is_required)
    {
        return Err(ReducedSwirlWrapperSystemError::KeyIntegrity("wrapper key"));
    }
    Ok(())
}

fn expected_public_values<C: ReducedSwirlWrapperVerifierComponents>(
    system: &ReducedSwirlWrapperSystem<C>,
    record: &ReducedSwirlWrapperRecord,
) -> Result<Vec<Vec<F>>, ReducedSwirlWrapperSystemError> {
    let core = generate_reduced_swirl_wrapper_core_traces(
        system.binding(),
        system.circuit.components.receipt_buses(),
        record,
    )
    .map_err(ReducedSwirlWrapperSystemError::Binding)?;
    let mut expected = vec![
        core.verifier_public_values,
        core.vm_public_values,
        Vec::new(),
    ];
    expected.resize_with(3 + system.component_air_count(), Vec::new);
    Ok(expected)
}

fn assemble_cpu_context<C: ReducedSwirlWrapperVerifierComponents>(
    system: &ReducedSwirlWrapperSystem<C>,
    record: &ReducedSwirlWrapperRecord,
    witness: ReducedSwirlWrapperWitness<CpuBackend<SC>>,
) -> Result<(ProvingContext<CpuBackend<SC>>, Vec<Vec<F>>), ReducedSwirlWrapperSystemError> {
    let core = generate_reduced_swirl_wrapper_core_traces(
        system.binding(),
        system.circuit.components.receipt_buses(),
        record,
    )
    .map_err(ReducedSwirlWrapperSystemError::Binding)?;
    validate_component_contexts(system, &witness.component_contexts)?;
    let expected = expected_public_values(system, record)?;
    let mut per_trace = vec![
        (
            0,
            AirProvingContext::simple(core.verifier_pvs, core.verifier_public_values),
        ),
        (
            1,
            AirProvingContext::simple(core.vm_pvs, core.vm_public_values),
        ),
        (2, AirProvingContext::simple_no_pis(core.statement)),
    ];
    per_trace.extend(
        witness
            .component_contexts
            .into_iter()
            .map(|(air, context)| (air + 3, context)),
    );
    Ok((ProvingContext::new(per_trace), expected))
}

fn validate_component_contexts<C, PB>(
    system: &ReducedSwirlWrapperSystem<C>,
    contexts: &[(usize, AirProvingContext<PB>)],
) -> Result<(), ReducedSwirlWrapperSystemError>
where
    C: ReducedSwirlWrapperVerifierComponents,
    PB: ProverBackend<Val = F>,
{
    let airs = system.airs();
    let mut present = BTreeSet::new();
    for (component_air, context) in contexts {
        let air = airs
            .get(component_air + 3)
            .ok_or(ReducedSwirlWrapperSystemError::Context("AIR index"))?;
        if !context.public_values.is_empty()
            || context.common_main.width() != air.common_main_width()
            || context.cached_mains.len() != air.cached_main_widths().len()
            || context.common_main.height() == 0
            || !context.common_main.height().is_power_of_two()
        {
            return Err(ReducedSwirlWrapperSystemError::Context("trace shape"));
        }
        for (cached, width) in context.cached_mains.iter().zip(air.cached_main_widths()) {
            if cached.trace.width() != width
                || cached.trace.height() != context.common_main.height()
            {
                return Err(ReducedSwirlWrapperSystemError::Context(
                    "cached trace shape",
                ));
            }
        }
        present.insert(*component_air);
    }
    if present.len() != system.component_air_count()
        || (0..system.component_air_count()).any(|index| !present.contains(&index))
    {
        return Err(ReducedSwirlWrapperSystemError::Context(
            "required AIR coverage",
        ));
    }
    Ok(())
}

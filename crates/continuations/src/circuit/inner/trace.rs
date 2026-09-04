use itertools::Itertools;
#[cfg(feature = "cuda")]
use openvm_circuit_primitives::hybrid_chip::cpu_proving_ctx_to_gpu;
use openvm_cpu_backend::CpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::GpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_stark_backend::{
    proof::Proof,
    prover::{AirProvingContext, ProverBackend},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, F};
use openvm_verify_stark_host::pvs::{DeferralPvs, VkCommit, VM_PVS_AIR_ID};

use crate::circuit::{SingleAirTraceData, SubCircuitTraceData};

#[derive(Copy, Clone)]
pub enum ProofsType {
    Vm,
    Deferral,
    Mix,
    Combined,
}

// Trait that inner provers use to remain generic in PB
pub trait InnerTraceGen<PB: ProverBackend, DC: Clone + Send + Sync> {
    fn new(deferral_enabled: bool) -> Self;
    fn generate_pre_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
        child_is_app: bool,
        child_vk_commit: VkCommit<F>,
        required_heights: Option<&[usize]>,
        device_ctx: &DC,
    ) -> SubCircuitTraceData<PB>;
    fn generate_post_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        child_is_app: bool,
        required_heights: Option<&[usize]>,
        device_ctx: &DC,
    ) -> Vec<AirProvingContext<PB>>;
}

pub struct InnerTraceGenImpl {
    pub deferral_enabled: bool,
}

/// App-style recursive prefix for the reduced-SWIRL terminal wrapper.
///
/// The child is a custom MultiSTARK rather than an OpenVM application proof,
/// so its execution identity comes from its authenticated `VmPvs` AIR. The
/// matching aggregation sub-circuit translates those values onto the normal
/// app connector/Merkle buses and binds the program commitment through
/// `VerifierExecutionIdentityBus`.
pub struct ReducedSwirlWrapperPrefixTraceGen {
    pub deferral_enabled: bool,
}

impl InnerTraceGen<CpuBackend<BabyBearPoseidon2Config>, ()> for ReducedSwirlWrapperPrefixTraceGen {
    fn new(deferral_enabled: bool) -> Self {
        Self { deferral_enabled }
    }

    fn generate_pre_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
        child_is_app: bool,
        child_vk_commit: VkCommit<F>,
        required_heights: Option<&[usize]>,
        _device_ctx: &(),
    ) -> SubCircuitTraceData<CpuBackend<BabyBearPoseidon2Config>> {
        assert!(child_is_app, "reduced-SWIRL prefix requires the app route");
        if let Some(required_heights) = required_heights {
            assert_eq!(required_heights.len(), 3);
        }
        let SingleAirTraceData {
            air_proving_ctx: verifier_pvs_ctx,
            mut poseidon2_compress_inputs,
            poseidon2_permute_inputs,
            mut range_check_inputs,
        } = super::verifier::generate_proving_ctx(
            proofs,
            proofs_type,
            true,
            child_vk_commit,
            self.deferral_enabled,
            required_heights.map(|heights| heights[0]),
        );
        let vm_pvs_ctx = super::vm_pvs::generate_proving_ctx_from_authenticated_vm_pvs(
            proofs,
            proofs_type,
            self.deferral_enabled,
            required_heights.map(|heights| heights[1]),
            VM_PVS_AIR_ID,
        );
        let idx2_ctx = if self.deferral_enabled {
            let (def_pvs_ctx, def_poseidon2_inputs, def_range_check_inputs) =
                super::def_pvs::generate_proving_ctx(
                    proofs,
                    proofs_type,
                    true,
                    absent_trace_pvs,
                    required_heights.map(|heights| heights[2]),
                );
            poseidon2_compress_inputs.extend_from_slice(&def_poseidon2_inputs);
            range_check_inputs.extend(def_range_check_inputs);
            def_pvs_ctx
        } else {
            super::unset::generate_proving_ctx(
                &[],
                true,
                required_heights.map(|heights| heights[2]),
            )
        };
        SubCircuitTraceData {
            air_proving_ctxs: vec![verifier_pvs_ctx, vm_pvs_ctx, idx2_ctx],
            poseidon2_compress_inputs,
            poseidon2_permute_inputs,
            range_check_inputs,
        }
    }

    fn generate_post_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        _child_is_app: bool,
        required_heights: Option<&[usize]>,
        device_ctx: &(),
    ) -> Vec<AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>> {
        <InnerTraceGenImpl as InnerTraceGen<CpuBackend<BabyBearPoseidon2Config>, ()>>::generate_post_verifier_subcircuit_ctxs(
            &InnerTraceGenImpl { deferral_enabled: self.deferral_enabled },
            proofs,
            proofs_type,
            true,
            required_heights,
            device_ctx,
        )
    }
}

impl InnerTraceGen<CpuBackend<BabyBearPoseidon2Config>, ()> for InnerTraceGenImpl {
    fn new(deferral_enabled: bool) -> Self {
        Self { deferral_enabled }
    }

    fn generate_pre_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
        child_is_app: bool,
        child_vk_commit: VkCommit<F>,
        required_heights: Option<&[usize]>,
        _device_ctx: &(),
    ) -> SubCircuitTraceData<CpuBackend<BabyBearPoseidon2Config>> {
        if let Some(required_heights) = required_heights {
            assert_eq!(required_heights.len(), 3);
        }
        let SingleAirTraceData {
            air_proving_ctx: verifier_pvs_ctx,
            mut poseidon2_compress_inputs,
            poseidon2_permute_inputs,
            mut range_check_inputs,
        } = super::verifier::generate_proving_ctx(
            proofs,
            proofs_type,
            child_is_app,
            child_vk_commit,
            self.deferral_enabled,
            required_heights.map(|heights| heights[0]),
        );
        let vm_pvs_ctx = super::vm_pvs::generate_proving_ctx(
            proofs,
            proofs_type,
            child_is_app,
            self.deferral_enabled,
            required_heights.map(|heights| heights[1]),
        );

        let idx2_ctx = if self.deferral_enabled {
            let (def_pvs_ctx, def_poseidon2_inputs, def_range_check_inputs) =
                super::def_pvs::generate_proving_ctx(
                    proofs,
                    proofs_type,
                    child_is_app,
                    absent_trace_pvs,
                    required_heights.map(|heights| heights[2]),
                );
            poseidon2_compress_inputs.extend_from_slice(&def_poseidon2_inputs);
            range_check_inputs.extend(def_range_check_inputs);
            def_pvs_ctx
        } else {
            super::unset::generate_proving_ctx(
                &[],
                child_is_app,
                required_heights.map(|heights| heights[2]),
            )
        };

        SubCircuitTraceData {
            air_proving_ctxs: vec![verifier_pvs_ctx, vm_pvs_ctx, idx2_ctx],
            poseidon2_compress_inputs,
            poseidon2_permute_inputs,
            range_check_inputs,
        }
    }

    fn generate_post_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        child_is_app: bool,
        required_heights: Option<&[usize]>,
        _device_ctx: &(),
    ) -> Vec<AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>> {
        if !self.deferral_enabled {
            assert!(required_heights.is_none_or(<[usize]>::is_empty));
            return vec![];
        }
        if let Some(required_heights) = required_heights {
            assert_eq!(required_heights.len(), 2);
        }

        let (vm_unset, def_unset) = match proofs_type {
            ProofsType::Vm => (
                vec![],
                proofs.iter().enumerate().map(|(i, _)| i).collect_vec(),
            ),
            ProofsType::Deferral => (
                proofs.iter().enumerate().map(|(i, _)| i).collect_vec(),
                vec![],
            ),
            ProofsType::Mix => (vec![1], vec![0]),
            ProofsType::Combined => (vec![], vec![]),
        };
        vec![
            super::unset::generate_proving_ctx(
                &vm_unset,
                child_is_app,
                required_heights.map(|heights| heights[0]),
            ),
            super::unset::generate_proving_ctx(
                &def_unset,
                child_is_app,
                required_heights.map(|heights| heights[1]),
            ),
        ]
    }
}

#[cfg(feature = "cuda")]
impl InnerTraceGen<GpuBackend, GpuDeviceCtx> for InnerTraceGenImpl {
    fn new(deferral_enabled: bool) -> Self {
        Self { deferral_enabled }
    }

    fn generate_pre_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
        child_is_app: bool,
        child_vk_commit: VkCommit<F>,
        required_heights: Option<&[usize]>,
        device_ctx: &GpuDeviceCtx,
    ) -> SubCircuitTraceData<GpuBackend> {
        let data =
            <Self as InnerTraceGen<CpuBackend<BabyBearPoseidon2Config>, ()>>::generate_pre_verifier_subcircuit_ctxs(
                self,
                proofs,
                proofs_type,
                absent_trace_pvs,
                child_is_app,
                child_vk_commit,
                required_heights,
                &(),
            );
        SubCircuitTraceData {
            air_proving_ctxs: data
                .air_proving_ctxs
                .into_iter()
                .map(|air_ctx| cpu_proving_ctx_to_gpu(air_ctx, device_ctx))
                .collect_vec(),
            poseidon2_compress_inputs: data.poseidon2_compress_inputs,
            poseidon2_permute_inputs: data.poseidon2_permute_inputs,
            range_check_inputs: data.range_check_inputs,
        }
    }

    fn generate_post_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        child_is_app: bool,
        required_heights: Option<&[usize]>,
        device_ctx: &GpuDeviceCtx,
    ) -> Vec<AirProvingContext<GpuBackend>> {
        let cpu_ctxs =
            <Self as InnerTraceGen<CpuBackend<BabyBearPoseidon2Config>, ()>>::generate_post_verifier_subcircuit_ctxs(
                self,
                proofs,
                proofs_type,
                child_is_app,
                required_heights,
                &(),
            );
        cpu_ctxs
            .into_iter()
            .map(|air_ctx| cpu_proving_ctx_to_gpu(air_ctx, device_ctx))
            .collect_vec()
    }
}

#[cfg(feature = "cuda")]
impl InnerTraceGen<GpuBackend, GpuDeviceCtx> for ReducedSwirlWrapperPrefixTraceGen {
    fn new(deferral_enabled: bool) -> Self {
        Self { deferral_enabled }
    }

    fn generate_pre_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
        child_is_app: bool,
        child_vk_commit: VkCommit<F>,
        required_heights: Option<&[usize]>,
        device_ctx: &GpuDeviceCtx,
    ) -> SubCircuitTraceData<GpuBackend> {
        let data = <Self as InnerTraceGen<CpuBackend<BabyBearPoseidon2Config>, ()>>::generate_pre_verifier_subcircuit_ctxs(
            self,
            proofs,
            proofs_type,
            absent_trace_pvs,
            child_is_app,
            child_vk_commit,
            required_heights,
            &(),
        );
        SubCircuitTraceData {
            air_proving_ctxs: data
                .air_proving_ctxs
                .into_iter()
                .map(|air_ctx| cpu_proving_ctx_to_gpu(air_ctx, device_ctx))
                .collect_vec(),
            poseidon2_compress_inputs: data.poseidon2_compress_inputs,
            poseidon2_permute_inputs: data.poseidon2_permute_inputs,
            range_check_inputs: data.range_check_inputs,
        }
    }

    fn generate_post_verifier_subcircuit_ctxs(
        &self,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        proofs_type: ProofsType,
        child_is_app: bool,
        required_heights: Option<&[usize]>,
        device_ctx: &GpuDeviceCtx,
    ) -> Vec<AirProvingContext<GpuBackend>> {
        let cpu_ctxs = <Self as InnerTraceGen<CpuBackend<BabyBearPoseidon2Config>, ()>>::generate_post_verifier_subcircuit_ctxs(
            self,
            proofs,
            proofs_type,
            child_is_app,
            required_heights,
            &(),
        );
        cpu_ctxs
            .into_iter()
            .map(|air_ctx| cpu_proving_ctx_to_gpu(air_ctx, device_ctx))
            .collect_vec()
    }
}

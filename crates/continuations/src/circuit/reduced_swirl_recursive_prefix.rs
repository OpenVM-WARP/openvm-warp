//! Minimal recursion entry point for one reduced-SWIRL WARP wrapper proof.
//!
//! The wrapper is a custom MultiSTARK, not an OpenVM application proof and
//! not an already-normalized recursive leaf. This sub-circuit verifies it as
//! an app-style child, then projects its authenticated `VmPvs` into the exact
//! connector, Merkle, and execution-identity messages consumed by OpenVM's
//! ordinary leaf normalizer. No WARP transition or History replay is added.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_cpu_backend::CpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{base::DeviceMatrix, hash_scheme::GpuHashScheme, GenericGpuBackend};
#[cfg(feature = "cuda")]
use openvm_cuda_common::{copy::MemCopyH2D, stream::GpuDeviceCtx};
use openvm_recursion_circuit::{
    bus::{CachedCommitBus, CachedCommitBusMessage, PublicValuesBus, PublicValuesBusMessage},
    system::{
        AggregationSubCircuit, BusIndexManager, BusInventory, CachedTraceCtx, VerifierConfig,
        VerifierExternalData, VerifierSubCircuit, VerifierTraceGen,
    },
};
use openvm_recursion_circuit_derive::AlignedBorrow;
#[cfg(feature = "cuda")]
use openvm_stark_backend::prover::ColMajorMatrix;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder},
    keygen::types::MultiStarkVerifyingKey,
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    proof::Proof,
    prover::{AirProvingContext, CommittedTraceData},
    AirRef, BaseAirWithPublicValues, EngineDeviceCtx, FiatShamirTranscript, PartitionedBaseAir,
    StarkEngine, StarkProtocolConfig, TranscriptHistory,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE, F};
use openvm_verify_stark_host::pvs::{VerifierBasePvs, VmPvs, VERIFIER_PVS_AIR_ID, VM_PVS_AIR_ID};

use crate::circuit::inner::{
    app::{CONNECTOR_AIR_ID, MERKLE_AIR_ID},
    bus::{
        VerifierExecutionIdentityBus, VerifierExecutionIdentityMessage, VerifierLayerIdentityBus,
        VerifierLayerIdentityMessage,
    },
};

/// Setup-fixed fan-in for the reduced-SWIRL wrapper prefix.
///
/// Four is the largest first-stage fan-in that keeps the real-block verifier
/// trace inside the setup-fixed log-height-23 envelope on the 32 GiB CUDA
/// target.  It is deliberately part of the verifying key rather than selected
/// from a block: executions therefore use the same relation and only vary the
/// active suffix of the last group.  Later ordinary recursive stages retain
/// fan-in eight, so the setup-supported 1024-source bound still reaches one
/// root without invoking `RecursiveSelf`.
pub const REDUCED_SWIRL_RECURSIVE_PREFIX_ARITY: usize = 4;

#[repr(C)]
#[derive(AlignedBorrow)]
struct ReducedSwirlWrapperPvsProjectionCols<T> {
    active: T,
    verifier_pvs: VerifierBasePvs<T>,
    vm_pvs: VmPvs<T>,
}

#[derive(Clone, Debug)]
struct CachedCommitLocation {
    air_idx: usize,
    cached_idx: usize,
    global_cached_idx: usize,
}

#[derive(Clone, Debug)]
struct ReducedSwirlWrapperPvsProjectionAir {
    proof_index: usize,
    public_values_bus: PublicValuesBus,
    cached_commit_bus: CachedCommitBus,
    cached_commit_locations: Vec<CachedCommitLocation>,
    verifier_layer_identity_bus: VerifierLayerIdentityBus,
    verifier_execution_identity_bus: VerifierExecutionIdentityBus,
}

impl ReducedSwirlWrapperPvsProjectionAir {
    fn generate_trace(
        &self,
        proof: Option<&Proof<BabyBearPoseidon2Config>>,
    ) -> Option<RowMajorMatrix<F>> {
        let width = self.width();
        let base_width = core::mem::size_of::<ReducedSwirlWrapperPvsProjectionCols<u8>>();
        let mut values = F::zero_vec(2 * width);
        if let Some(proof) = proof {
            let verifier = proof.public_values.get(VERIFIER_PVS_AIR_ID)?;
            let vm = proof.public_values.get(VM_PVS_AIR_ID)?;
            if verifier.len() != VerifierBasePvs::<u8>::width() || vm.len() != VmPvs::<u8>::width()
            {
                return None;
            }
            let local: &mut ReducedSwirlWrapperPvsProjectionCols<F> =
                values[..base_width].borrow_mut();
            local.active = F::ONE;
            local.verifier_pvs = *verifier.as_slice().borrow();
            local.vm_pvs = *vm.as_slice().borrow();
            for (location_idx, location) in self.cached_commit_locations.iter().enumerate() {
                let commitment = proof
                    .trace_vdata
                    .get(location.air_idx)?
                    .as_ref()?
                    .cached_commitments
                    .get(location.cached_idx)?;
                let start = base_width + location_idx * DIGEST_SIZE;
                values[start..start + DIGEST_SIZE].copy_from_slice(commitment);
            }
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAir<F> for ReducedSwirlWrapperPvsProjectionAir {
    fn width(&self) -> usize {
        core::mem::size_of::<ReducedSwirlWrapperPvsProjectionCols<u8>>()
            + self.cached_commit_locations.len() * DIGEST_SIZE
    }
}

impl BaseAirWithPublicValues<F> for ReducedSwirlWrapperPvsProjectionAir {}
impl PartitionedBaseAir<F> for ReducedSwirlWrapperPvsProjectionAir {}

impl<AB> Air<AB> for ReducedSwirlWrapperPvsProjectionAir
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main
            .row_slice(0)
            .expect("reduced-SWIRL wrapper projection row");
        let next_row = main
            .row_slice(1)
            .expect("reduced-SWIRL wrapper projection padding");
        let base_width = core::mem::size_of::<ReducedSwirlWrapperPvsProjectionCols<u8>>();
        let local: &ReducedSwirlWrapperPvsProjectionCols<AB::Var> =
            local_row[..base_width].borrow();
        let next: &ReducedSwirlWrapperPvsProjectionCols<AB::Var> = next_row[..base_width].borrow();
        let active = Into::<AB::Expr>::into(local.active);

        builder.assert_bool(local.active);
        builder.when_last_row().assert_zero(local.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.active);

        // Consume the second copy of every child public value emitted by the
        // continuations-enabled ProofShape module. The first copy is consumed
        // by the child's symbolic constraint evaluator.
        for (pv_index, value) in local.verifier_pvs.as_slice().iter().enumerate() {
            self.public_values_bus.receive(
                builder,
                AB::Expr::from_usize(self.proof_index),
                PublicValuesBusMessage {
                    air_idx: AB::Expr::from_usize(VERIFIER_PVS_AIR_ID),
                    pv_idx: AB::Expr::from_usize(pv_index),
                    value: (*value).into(),
                },
                active.clone(),
            );
        }
        for (pv_index, value) in local.vm_pvs.as_slice().iter().enumerate() {
            self.public_values_bus.receive(
                builder,
                AB::Expr::from_usize(self.proof_index),
                PublicValuesBusMessage {
                    air_idx: AB::Expr::from_usize(VM_PVS_AIR_ID),
                    pv_idx: AB::Expr::from_usize(pv_index),
                    value: (*value).into(),
                },
                active.clone(),
            );
        }

        // `ProofShapeAir` publishes one continuation copy of every cached
        // child commitment. A custom app child has no ProgramAir consumer for
        // those copies, so consume the exact proof-authenticated values here.
        for (location_idx, location) in self.cached_commit_locations.iter().enumerate() {
            let start = base_width + location_idx * DIGEST_SIZE;
            let cached_commit = core::array::from_fn(|limb| local_row[start + limb].into());
            self.cached_commit_bus.receive(
                builder,
                AB::Expr::from_usize(self.proof_index),
                CachedCommitBusMessage {
                    air_idx: AB::Expr::from_usize(location.air_idx),
                    cached_idx: AB::Expr::from_usize(location.cached_idx),
                    global_cached_idx: AB::Expr::from_usize(location.global_cached_idx),
                    cached_commit,
                },
                active.clone(),
            );
        }

        // Re-emit the execution boundary in the app-proof coordinates used
        // by the unmodified OpenVM VmPvsAir. Program commitment is carried by
        // the dedicated execution-identity bus below, so no fake cached
        // ProgramAir commitment is introduced.
        for (pv_index, value) in [
            local.vm_pvs.initial_pc,
            local.vm_pvs.final_pc,
            local.vm_pvs.exit_code,
            local.vm_pvs.is_terminate,
        ]
        .into_iter()
        .enumerate()
        {
            self.public_values_bus.send(
                builder,
                AB::Expr::from_usize(self.proof_index),
                PublicValuesBusMessage {
                    air_idx: AB::Expr::from_usize(CONNECTOR_AIR_ID),
                    pv_idx: AB::Expr::from_usize(pv_index),
                    value: value.into(),
                },
                active.clone(),
            );
        }
        for (pv_index, value) in local
            .vm_pvs
            .initial_root
            .iter()
            .chain(local.vm_pvs.final_root.iter())
            .enumerate()
        {
            self.public_values_bus.send(
                builder,
                AB::Expr::from_usize(self.proof_index),
                PublicValuesBusMessage {
                    air_idx: AB::Expr::from_usize(MERKLE_AIR_ID),
                    pv_idx: AB::Expr::from_usize(pv_index),
                    value: (*value).into(),
                },
                active.clone(),
            );
        }

        self.verifier_layer_identity_bus.lookup_key(
            builder,
            AB::Expr::from_usize(self.proof_index),
            VerifierLayerIdentityMessage {
                internal_flag: AB::Expr::ZERO,
                recursion_depth: AB::Expr::ZERO,
            },
            active.clone(),
        );
        self.verifier_execution_identity_bus.lookup_key(
            builder,
            AB::Expr::from_usize(self.proof_index),
            VerifierExecutionIdentityMessage {
                program_commit: local.vm_pvs.program_commit.map(Into::into),
                initial_pc: local.vm_pvs.initial_pc.into(),
                final_pc: local.vm_pvs.final_pc.into(),
                exit_code: local.vm_pvs.exit_code.into(),
                is_terminate: local.vm_pvs.is_terminate.into(),
                initial_root: local.vm_pvs.initial_root.map(Into::into),
                final_root: local.vm_pvs.final_root.map(Into::into),
            },
            active,
        );
    }
}

/// One fixed wrapper-verifier relation plus three tiny inactive-capable PVS
/// projections.  The final-wrapper adapter uses one active slot, while the
/// source-tree entry point may fill all three in canonical segment order.
pub struct ReducedSwirlWrapperPrefixSubCircuit {
    verifier: VerifierSubCircuit<REDUCED_SWIRL_RECURSIVE_PREFIX_ARITY>,
    projections: [ReducedSwirlWrapperPvsProjectionAir; REDUCED_SWIRL_RECURSIVE_PREFIX_ARITY],
    verifier_layer_identity_bus: VerifierLayerIdentityBus,
    verifier_execution_identity_bus: VerifierExecutionIdentityBus,
    next_bus_idx: BusIndex,
}

impl ReducedSwirlWrapperPrefixSubCircuit {
    fn new_inner(
        child_vk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
        config: VerifierConfig,
    ) -> Self {
        assert!(
            config.has_cached,
            "reduced-SWIRL prefix requires cached VK mode"
        );
        assert_eq!(
            child_vk.inner.per_air[VERIFIER_PVS_AIR_ID]
                .params
                .num_public_values,
            VerifierBasePvs::<u8>::width()
        );
        assert_eq!(
            child_vk.inner.per_air[VM_PVS_AIR_ID]
                .params
                .num_public_values,
            VmPvs::<u8>::width()
        );
        assert!(child_vk
            .inner
            .per_air
            .iter()
            .enumerate()
            .all(|(air_id, air)| air_id == VERIFIER_PVS_AIR_ID
                || air_id == VM_PVS_AIR_ID
                || air.params.num_public_values == 0));

        let mut global_cached_idx = 0usize;
        let mut cached_commit_locations = Vec::new();
        for (air_idx, air) in child_vk.inner.per_air.iter().enumerate() {
            assert!(
                air.is_required || air.params.width.cached_mains.is_empty(),
                "optional cached child AIRs are unsupported by the reduced-SWIRL prefix"
            );
            for cached_idx in 0..air.params.width.cached_mains.len() {
                cached_commit_locations.push(CachedCommitLocation {
                    air_idx,
                    cached_idx,
                    global_cached_idx,
                });
                global_cached_idx += 1;
            }
        }

        let verifier = VerifierSubCircuit::new_with_options(child_vk, config);
        let mut buses = BusIndexManager::from_next_bus_idx(verifier.next_bus_idx());
        let verifier_layer_identity_bus = VerifierLayerIdentityBus::new(buses.new_bus_idx());
        let verifier_execution_identity_bus =
            VerifierExecutionIdentityBus::new(buses.new_bus_idx());
        let projections = core::array::from_fn(|proof_index| ReducedSwirlWrapperPvsProjectionAir {
            proof_index,
            public_values_bus: verifier.bus_inventory().public_values_bus,
            cached_commit_bus: verifier.bus_inventory().cached_commit_bus,
            cached_commit_locations: cached_commit_locations.clone(),
            verifier_layer_identity_bus,
            verifier_execution_identity_bus,
        });
        Self {
            verifier,
            projections,
            verifier_layer_identity_bus,
            verifier_execution_identity_bus,
            next_bus_idx: buses.next_bus_idx(),
        }
    }
}

impl AggregationSubCircuit for ReducedSwirlWrapperPrefixSubCircuit {
    fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        self.verifier
            .airs::<SC>()
            .into_iter()
            .chain(
                self.projections
                    .iter()
                    .cloned()
                    .map(|air| Arc::new(air) as AirRef<SC>),
            )
            .collect()
    }

    fn bus_inventory(&self) -> &BusInventory {
        self.verifier.bus_inventory()
    }

    fn next_bus_idx(&self) -> BusIndex {
        self.next_bus_idx
    }

    fn max_num_proofs(&self) -> usize {
        REDUCED_SWIRL_RECURSIVE_PREFIX_ARITY
    }

    fn verifier_layer_identity_bus_idx(&self) -> Option<BusIndex> {
        Some(self.verifier_layer_identity_bus.index())
    }

    fn verifier_execution_identity_bus_idx(&self) -> Option<BusIndex> {
        Some(self.verifier_execution_identity_bus.index())
    }
}

impl<SC: StarkProtocolConfig<F = F>> VerifierTraceGen<CpuBackend<SC>, SC, ()>
    for ReducedSwirlWrapperPrefixSubCircuit
{
    fn new(
        child_mvk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
        config: VerifierConfig,
    ) -> Self {
        Self::new_inner(child_mvk, config)
    }

    fn commit_child_vk<E: StarkEngine<SC = SC, PB = CpuBackend<SC>>>(
        &self,
        engine: &E,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> CommittedTraceData<CpuBackend<SC>>
    where
        (): From<EngineDeviceCtx<E>>,
    {
        <VerifierSubCircuit<REDUCED_SWIRL_RECURSIVE_PREFIX_ARITY> as VerifierTraceGen<
            CpuBackend<SC>,
            SC,
            (),
        >>::commit_child_vk(&self.verifier, engine, child_vk)
    }

    fn cached_trace_record(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> openvm_recursion_circuit::batch_constraint::expr_eval::CachedTraceRecord {
        <VerifierSubCircuit<REDUCED_SWIRL_RECURSIVE_PREFIX_ARITY> as VerifierTraceGen<
            CpuBackend<SC>,
            SC,
            (),
        >>::cached_trace_record(&self.verifier, child_vk)
    }

    fn generate_proving_ctxs<TS>(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        cached_trace_ctx: CachedTraceCtx<CpuBackend<SC>>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        external_data: &mut VerifierExternalData,
        device_ctx: &(),
        initial_transcript: TS,
    ) -> Option<Vec<AirProvingContext<CpuBackend<SC>>>>
    where
        TS: FiatShamirTranscript<BabyBearPoseidon2Config>
            + TranscriptHistory<F = F, State = [F; POSEIDON2_WIDTH]>,
    {
        if proofs.is_empty() || proofs.len() > REDUCED_SWIRL_RECURSIVE_PREFIX_ARITY {
            return None;
        }
        let verifier_air_count = self.verifier.airs::<SC>().len();
        let verifier_heights = match external_data.required_heights {
            None => None,
            Some(heights) => {
                if heights.len() != verifier_air_count + self.projections.len()
                    || heights[verifier_air_count..]
                        .iter()
                        .any(|height| *height != 2)
                {
                    return None;
                }
                Some(&heights[..verifier_air_count])
            }
        };
        let mut delegated = VerifierExternalData {
            poseidon2_compress_inputs: external_data.poseidon2_compress_inputs,
            poseidon2_permute_inputs: external_data.poseidon2_permute_inputs,
            range_check_inputs: external_data.range_check_inputs,
            power_check_inputs: external_data.power_check_inputs,
            required_heights: verifier_heights,
            final_transcript_state: external_data.final_transcript_state.as_deref_mut(),
        };
        let mut contexts =
            <VerifierSubCircuit<REDUCED_SWIRL_RECURSIVE_PREFIX_ARITY> as VerifierTraceGen<
                CpuBackend<SC>,
                SC,
                (),
            >>::generate_proving_ctxs(
                &self.verifier,
                child_vk,
                cached_trace_ctx,
                proofs,
                &mut delegated,
                device_ctx,
                initial_transcript,
            )?;
        for (slot, projection) in self.projections.iter().enumerate() {
            let trace = projection.generate_trace(proofs.get(slot))?;
            contexts.push(AirProvingContext::simple_no_pis(trace));
        }
        Some(contexts)
    }
}

#[cfg(feature = "cuda")]
impl<HS: GpuHashScheme> VerifierTraceGen<GenericGpuBackend<HS>, HS::SC, GpuDeviceCtx>
    for ReducedSwirlWrapperPrefixSubCircuit
{
    fn new(
        child_mvk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
        config: VerifierConfig,
    ) -> Self {
        Self::new_inner(child_mvk, config)
    }

    fn commit_child_vk<E: StarkEngine<SC = HS::SC, PB = GenericGpuBackend<HS>>>(
        &self,
        engine: &E,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> CommittedTraceData<GenericGpuBackend<HS>>
    where
        GpuDeviceCtx: From<EngineDeviceCtx<E>>,
    {
        <VerifierSubCircuit<REDUCED_SWIRL_RECURSIVE_PREFIX_ARITY> as VerifierTraceGen<
            GenericGpuBackend<HS>,
            HS::SC,
            GpuDeviceCtx,
        >>::commit_child_vk(&self.verifier, engine, child_vk)
    }

    fn cached_trace_record(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> openvm_recursion_circuit::batch_constraint::expr_eval::CachedTraceRecord {
        <VerifierSubCircuit<REDUCED_SWIRL_RECURSIVE_PREFIX_ARITY> as VerifierTraceGen<
            GenericGpuBackend<HS>,
            HS::SC,
            GpuDeviceCtx,
        >>::cached_trace_record(&self.verifier, child_vk)
    }

    fn generate_proving_ctxs<TS>(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        cached_trace_ctx: CachedTraceCtx<GenericGpuBackend<HS>>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        external_data: &mut VerifierExternalData,
        device_ctx: &GpuDeviceCtx,
        initial_transcript: TS,
    ) -> Option<Vec<AirProvingContext<GenericGpuBackend<HS>>>>
    where
        TS: FiatShamirTranscript<BabyBearPoseidon2Config>
            + TranscriptHistory<F = F, State = [F; POSEIDON2_WIDTH]>,
    {
        if proofs.is_empty() || proofs.len() > REDUCED_SWIRL_RECURSIVE_PREFIX_ARITY {
            return None;
        }
        let verifier_air_count = self.verifier.airs::<HS::SC>().len();
        let verifier_heights = match external_data.required_heights {
            None => None,
            Some(heights) => {
                if heights.len() != verifier_air_count + self.projections.len()
                    || heights[verifier_air_count..]
                        .iter()
                        .any(|height| *height != 2)
                {
                    return None;
                }
                Some(&heights[..verifier_air_count])
            }
        };
        let mut delegated = VerifierExternalData {
            poseidon2_compress_inputs: external_data.poseidon2_compress_inputs,
            poseidon2_permute_inputs: external_data.poseidon2_permute_inputs,
            range_check_inputs: external_data.range_check_inputs,
            power_check_inputs: external_data.power_check_inputs,
            required_heights: verifier_heights,
            final_transcript_state: external_data.final_transcript_state.as_deref_mut(),
        };
        let mut contexts =
            <VerifierSubCircuit<REDUCED_SWIRL_RECURSIVE_PREFIX_ARITY> as VerifierTraceGen<
                GenericGpuBackend<HS>,
                HS::SC,
                GpuDeviceCtx,
            >>::generate_proving_ctxs(
                &self.verifier,
                child_vk,
                cached_trace_ctx,
                proofs,
                &mut delegated,
                device_ctx,
                initial_transcript,
            )?;
        for (slot, projection) in self.projections.iter().enumerate() {
            let trace = projection.generate_trace(proofs.get(slot))?;
            let height = Matrix::height(&trace);
            let width = Matrix::width(&trace);
            let host = ColMajorMatrix::from_row_major(&trace);
            let buffer = host.values.to_device_on(device_ctx).ok()?;
            let matrix = DeviceMatrix::new(Arc::new(buffer), height, width);
            contexts.push(AirProvingContext::new(Vec::new(), matrix, Vec::new()));
        }
        Some(contexts)
    }
}

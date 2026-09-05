use std::sync::Arc;

use eyre::Result;
use openvm_recursion_circuit::system::{
    AggregationSubCircuit, VerifierConfig, VerifierTailMode, VerifierTraceGen,
};
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    prover::{
        CommittedTraceData, DeviceDataTransporter, DeviceMultiStarkProvingKey, ProverBackend,
        ProverDevice,
    },
    EngineDeviceCtx, StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, EF, F};
use openvm_verify_stark_host::pvs::{DeferralPvs, VkCommit};
use p3_matrix::dense::RowMajorMatrix;
use tracing::instrument;

use crate::{
    circuit::{
        inner::{InnerCircuit, InnerTraceGen, ProofsType},
        Circuit,
    },
    prover::trace_heights_tracing_info,
    SC,
};

mod trace;

/// Generates an aggregation proof for inner layers (leaf and internal).
pub struct InnerAggregationProver<
    PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
    T,
> {
    pk: Arc<MultiStarkProvingKey<SC>>,
    d_pk: Option<DeviceMultiStarkProvingKey<PB>>,
    vk: Arc<MultiStarkVerifyingKey<SC>>,

    agg_node_tracegen: T,

    // TODO: tracegen currently requires storing these, we should revisit this
    child_vk: Arc<MultiStarkVerifyingKey<SC>>,
    child_vk_pcs_data: Option<CommittedTraceData<PB>>,
    child_vk_cached_commitment: Digest,
    /// Backend-independent reconstruction of the exact child-VK cached table
    /// committed by `child_vk_pcs_data`. This remains setup data and is used
    /// to authenticate cached-table openings outside the ordinary STARK PCS.
    child_vk_cached_trace: RowMajorMatrix<F>,
    circuit: Arc<InnerCircuit<S>>,
    verifier_has_cached: bool,
    verifier_dag_commit: Option<Digest>,

    self_vk_pcs_data: Option<CommittedTraceData<PB>>,
    /// Commitment retained after releasing the self-VK PCS codeword/tree.
    /// Recursive trace generation binds this digest but does not need the
    /// committed device buffers once the fixed WARP index has copied setup
    /// authority.
    self_vk_cached_commitment: Option<Digest>,
    /// Backend-independent fixed cached table for the self-recursive VK.
    /// This differs from `child_vk_cached_trace` and must seed a WARP relation
    /// whose fixed verifier route is `RecursiveSelf`.
    self_vk_cached_trace: Option<RowMajorMatrix<F>>,
}

/// Struct to determine if InnerAggregationProver is proving a special case,
/// i.e. if the child_vk is the app_vk or if it should use its own vk as child.
#[derive(Clone, Copy)]
pub enum ChildVkKind {
    Standard,
    App,
    RecursiveSelf,
}

impl<PB, S, T> InnerAggregationProver<PB, S, T>
where
    PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
    PB::Matrix: Clone,
{
    #[instrument(name = "total_proof", skip_all)]
    pub fn agg_prove<E: StarkEngine<SC = SC, PB = PB>>(
        &self,
        proofs: &[Proof<SC>],
        child_vk_kind: ChildVkKind,
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
    ) -> Result<Proof<SC>>
    where
        S: VerifierTraceGen<PB, SC, EngineDeviceCtx<E>>,
        T: InnerTraceGen<PB, EngineDeviceCtx<E>>,
    {
        let engine = E::new(self.pk.params.clone());
        let ctx = self.generate_proving_ctx(
            proofs,
            child_vk_kind,
            proofs_type,
            absent_trace_pvs,
            engine.device().device_ctx(),
        );
        if tracing::enabled!(tracing::Level::DEBUG) {
            trace_heights_tracing_info::<_, SC>(&ctx.per_trace, &self.circuit.airs());
        }
        #[cfg(debug_assertions)]
        if crate::prover::debug_checks_enabled() {
            crate::prover::debug_constraints(&self.circuit, &ctx, &engine);
        }
        let proof = engine.prove(
            self.d_pk
                .as_ref()
                .expect("recursive proving requires the device proving key"),
            ctx,
        )?;
        #[cfg(debug_assertions)]
        if crate::prover::debug_checks_enabled() {
            engine.verify(&self.vk, &proof)?;
        }
        Ok(proof)
    }

    pub fn agg_prove_no_def<E: StarkEngine<SC = SC, PB = PB>>(
        &self,
        proofs: &[Proof<SC>],
        child_vk_kind: ChildVkKind,
    ) -> Result<Proof<SC>>
    where
        S: VerifierTraceGen<PB, SC, EngineDeviceCtx<E>>,
        T: InnerTraceGen<PB, EngineDeviceCtx<E>>,
    {
        self.agg_prove::<E>(proofs, child_vk_kind, ProofsType::Vm, None)
    }
    pub fn new<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        system_params: SystemParams,
        is_self_recursive: bool,
        def_hook_cached_commit: Option<Digest>,
    ) -> Self
    where
        S: VerifierTraceGen<PB, SC, EngineDeviceCtx<E>>,
        T: InnerTraceGen<PB, EngineDeviceCtx<E>>,
    {
        Self::new_with_cached_mode::<E>(
            child_vk,
            system_params,
            is_self_recursive,
            def_hook_cached_commit,
            true,
            VerifierTailMode::Complete,
        )
    }

    /// Construct an inner verifier whose symbolic-expression DAG is rebuilt
    /// in common-main and committed by `DagCommitSubAir`, matching the root
    /// prover's no-cached mode. This is the only suitable source shape for a
    /// verifier-WARP relation that must not carry an unauthenticated cached
    /// main table.
    pub fn new_without_cached<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        system_params: SystemParams,
        is_self_recursive: bool,
        def_hook_cached_commit: Option<Digest>,
    ) -> Self
    where
        S: VerifierTraceGen<PB, SC, EngineDeviceCtx<E>>,
        T: InnerTraceGen<PB, EngineDeviceCtx<E>>,
    {
        Self::new_with_cached_mode::<E>(
            child_vk,
            system_params,
            is_self_recursive,
            def_hook_cached_commit,
            false,
            VerifierTailMode::Complete,
        )
    }

    /// Build the same recursive verifier data plane but stop after the SWIRL
    /// stacking reduction.  The resulting context is not a standalone STARK
    /// proof: its public checkpoint must be joined to the retained PCS data by
    /// the caller's terminal opening obligation.
    pub fn new_deferred_opening<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        system_params: SystemParams,
        is_self_recursive: bool,
        def_hook_cached_commit: Option<Digest>,
    ) -> Self
    where
        S: VerifierTraceGen<PB, SC, EngineDeviceCtx<E>>,
        T: InnerTraceGen<PB, EngineDeviceCtx<E>>,
    {
        Self::new_with_cached_mode::<E>(
            child_vk,
            system_params,
            is_self_recursive,
            def_hook_cached_commit,
            true,
            VerifierTailMode::DeferredWhir,
        )
    }

    /// Deferred-opening verifier with its symbolic-expression DAG embedded
    /// into common main instead of retained as cached setup data.
    pub fn new_deferred_opening_without_cached<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        system_params: SystemParams,
        is_self_recursive: bool,
        def_hook_cached_commit: Option<Digest>,
    ) -> Self
    where
        S: VerifierTraceGen<PB, SC, EngineDeviceCtx<E>>,
        T: InnerTraceGen<PB, EngineDeviceCtx<E>>,
    {
        Self::new_with_cached_mode::<E>(
            child_vk,
            system_params,
            is_self_recursive,
            def_hook_cached_commit,
            false,
            VerifierTailMode::DeferredWhir,
        )
    }

    /// Build a verifier PESAT that checks AIR/LogUp and exports the exact
    /// original commitment/opening manifest, but performs no per-child
    /// stacking or WHIR work.
    pub fn new_deferred_stacking<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        system_params: SystemParams,
        is_self_recursive: bool,
        def_hook_cached_commit: Option<Digest>,
    ) -> Self
    where
        S: VerifierTraceGen<PB, SC, EngineDeviceCtx<E>>,
        T: InnerTraceGen<PB, EngineDeviceCtx<E>>,
    {
        Self::new_with_cached_mode::<E>(
            child_vk,
            system_params,
            is_self_recursive,
            def_hook_cached_commit,
            true,
            VerifierTailMode::DeferredStacking,
        )
    }

    fn new_with_cached_mode<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        system_params: SystemParams,
        is_self_recursive: bool,
        def_hook_cached_commit: Option<Digest>,
        verifier_has_cached: bool,
        tail_mode: VerifierTailMode,
    ) -> Self
    where
        S: VerifierTraceGen<PB, SC, EngineDeviceCtx<E>>,
        T: InnerTraceGen<PB, EngineDeviceCtx<E>>,
    {
        let verifier_circuit = S::new(
            child_vk.clone(),
            VerifierConfig {
                continuations_enabled: true,
                has_cached: verifier_has_cached,
                tail_mode,
                ..Default::default()
            },
        );
        let engine = E::new(system_params);
        let verifier_dag_commit = (!verifier_has_cached).then(|| {
            verifier_circuit
                .cached_trace_record(&child_vk)
                .dag_commit_info
                .expect("no-cached verifier must derive a DAG commitment")
                .commit
        });
        let child_vk_cached_trace = verifier_circuit
            .cached_trace_record(&child_vk)
            .to_cached_trace_matrix();
        let child_vk_pcs_data = verifier_circuit.commit_child_vk(&engine, &child_vk);
        let circuit = Arc::new(InnerCircuit::new(
            Arc::new(verifier_circuit),
            def_hook_cached_commit.map(|d| d.into()),
        ));
        let (pk, vk) = engine.keygen(&circuit.airs());
        let d_pk = engine.device().transport_pk_to_device(&pk);
        let self_vk_pcs_data = if is_self_recursive {
            Some(circuit.verifier_circuit.commit_child_vk(&engine, &vk))
        } else {
            None
        };
        let self_vk_cached_trace = is_self_recursive.then(|| {
            circuit
                .verifier_circuit
                .cached_trace_record(&vk)
                .to_cached_trace_matrix()
        });
        let agg_node_tracegen = InnerTraceGen::new(def_hook_cached_commit.is_some());
        let self_vk_cached_commitment = self_vk_pcs_data.as_ref().map(|data| data.commitment);
        Self {
            pk: Arc::new(pk),
            d_pk: Some(d_pk),
            vk: Arc::new(vk),
            agg_node_tracegen,
            child_vk,
            child_vk_cached_commitment: child_vk_pcs_data.commitment,
            child_vk_pcs_data: Some(child_vk_pcs_data),
            child_vk_cached_trace,
            circuit,
            verifier_has_cached,
            verifier_dag_commit,
            self_vk_pcs_data,
            self_vk_cached_commitment,
            self_vk_cached_trace,
        }
    }

    pub fn from_pk<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        pk: Arc<MultiStarkProvingKey<SC>>,
        is_self_recursive: bool,
        def_hook_cached_commit: Option<Digest>,
    ) -> Self
    where
        S: VerifierTraceGen<PB, SC, EngineDeviceCtx<E>>,
        T: InnerTraceGen<PB, EngineDeviceCtx<E>>,
    {
        Self::from_pk_with_cached_mode::<E>(
            child_vk,
            pk,
            is_self_recursive,
            def_hook_cached_commit,
            true,
            VerifierTailMode::Complete,
        )
    }

    pub fn from_pk_deferred_opening<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        pk: Arc<MultiStarkProvingKey<SC>>,
        is_self_recursive: bool,
        def_hook_cached_commit: Option<Digest>,
    ) -> Self
    where
        S: VerifierTraceGen<PB, SC, EngineDeviceCtx<E>>,
        T: InnerTraceGen<PB, EngineDeviceCtx<E>>,
    {
        Self::from_pk_with_cached_mode::<E>(
            child_vk,
            pk,
            is_self_recursive,
            def_hook_cached_commit,
            true,
            VerifierTailMode::DeferredWhir,
        )
    }

    fn from_pk_with_cached_mode<E: StarkEngine<SC = SC, PB = PB>>(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        pk: Arc<MultiStarkProvingKey<SC>>,
        is_self_recursive: bool,
        def_hook_cached_commit: Option<Digest>,
        verifier_has_cached: bool,
        tail_mode: VerifierTailMode,
    ) -> Self
    where
        S: VerifierTraceGen<PB, SC, EngineDeviceCtx<E>>,
        T: InnerTraceGen<PB, EngineDeviceCtx<E>>,
    {
        let verifier_circuit = S::new(
            child_vk.clone(),
            VerifierConfig {
                continuations_enabled: true,
                has_cached: verifier_has_cached,
                tail_mode,
                ..Default::default()
            },
        );
        let engine = E::new(pk.params.clone());
        let verifier_dag_commit = (!verifier_has_cached).then(|| {
            verifier_circuit
                .cached_trace_record(&child_vk)
                .dag_commit_info
                .expect("no-cached verifier must derive a DAG commitment")
                .commit
        });
        let child_vk_cached_trace = verifier_circuit
            .cached_trace_record(&child_vk)
            .to_cached_trace_matrix();
        let child_vk_pcs_data = verifier_circuit.commit_child_vk(&engine, &child_vk);
        let circuit = Arc::new(InnerCircuit::new(
            Arc::new(verifier_circuit),
            def_hook_cached_commit.map(|d| d.into()),
        ));
        let vk = Arc::new(pk.get_vk());
        let d_pk = engine.device().transport_pk_to_device(&pk);
        let self_vk_pcs_data = if is_self_recursive {
            Some(circuit.verifier_circuit.commit_child_vk(&engine, &vk))
        } else {
            None
        };
        let self_vk_cached_trace = is_self_recursive.then(|| {
            circuit
                .verifier_circuit
                .cached_trace_record(&vk)
                .to_cached_trace_matrix()
        });
        let agg_node_tracegen = InnerTraceGen::new(def_hook_cached_commit.is_some());
        let self_vk_cached_commitment = self_vk_pcs_data.as_ref().map(|data| data.commitment);
        Self {
            pk,
            d_pk: Some(d_pk),
            vk,
            agg_node_tracegen,
            child_vk,
            child_vk_cached_commitment: child_vk_pcs_data.commitment,
            child_vk_pcs_data: Some(child_vk_pcs_data),
            child_vk_cached_trace,
            circuit,
            verifier_has_cached,
            verifier_dag_commit,
            self_vk_pcs_data,
            self_vk_cached_commitment,
            self_vk_cached_trace,
        }
    }

    pub fn get_circuit(&self) -> Arc<InnerCircuit<S>> {
        self.circuit.clone()
    }

    pub fn get_pk(&self) -> Arc<MultiStarkProvingKey<SC>> {
        self.pk.clone()
    }

    /// Device proving key for protocol components that reuse this circuit's
    /// committed traces without producing a complete recursive STARK proof.
    /// In particular, proof-verifier WARP uses it for the external LogUp/GKR
    /// obligation bound to the same fixed verifier-PESAT source.
    pub fn get_device_pk(&self) -> &DeviceMultiStarkProvingKey<PB> {
        self.d_pk
            .as_ref()
            .expect("device proving key was released for trace-only WARP")
    }

    /// Retained preprocessed setup commitments used by this recursive
    /// verifier, in canonical AIR-ID order.
    ///
    /// These are borrowed directly from the device proving key. A setup PCS
    /// authority may clone each entry's `Arc<PcsData>` to extend its lifetime,
    /// but must not re-encode or recommit the associated trace.
    pub fn get_preprocessed_setup_committed_traces(
        &self,
    ) -> impl Iterator<Item = (usize, &CommittedTraceData<PB>)> {
        self.d_pk
            .as_ref()
            .expect("device proving key was released for trace-only WARP")
            .preprocessed_committed_traces()
    }

    /// Exact child-VK cached commitment used by the recursive verifier.
    ///
    /// The returned object carries the original commitment, unstacked trace,
    /// and genuine backend PCS data. It is suitable for a setup PCS authority;
    /// unlike a root-only object, it retains the proving data needed to answer
    /// openings.
    #[must_use]
    pub const fn get_child_vk_setup_committed_trace(&self) -> &CommittedTraceData<PB> {
        self.child_vk_pcs_data
            .as_ref()
            .expect("child-VK PCS data was released for trace-only WARP")
    }

    /// Arc-clone the genuine child-VK setup PCS data without copying or
    /// reconstructing its codewords and Merkle tree.
    #[must_use]
    pub fn clone_child_vk_setup_pcs_data(&self) -> Arc<PB::PcsData> {
        self.get_child_vk_setup_committed_trace().clone_pcs_data()
    }

    pub fn get_vk(&self) -> Arc<MultiStarkVerifyingKey<SC>> {
        self.vk.clone()
    }

    /// Exact setup-owned child-VK cached trace used by the recursive lane.
    #[must_use]
    pub const fn get_child_vk_cached_trace(&self) -> &RowMajorMatrix<F> {
        &self.child_vk_cached_trace
    }

    /// Exact setup-owned self-VK cached trace used by recursive-self proofs.
    #[must_use]
    pub fn get_self_vk_cached_trace(&self) -> Option<&RowMajorMatrix<F>> {
        self.self_vk_cached_trace.as_ref()
    }

    pub fn deferral_enabled(&self) -> bool {
        self.circuit.def_hook_cached_commit.is_some()
    }

    #[must_use]
    pub const fn verifier_has_cached(&self) -> bool {
        self.verifier_has_cached
    }

    /// Setup-derived symbolic-expression DAG commitment in no-cached mode.
    /// It is part of the verifier relation binding and must match the explicit
    /// `DagCommitPvs` coordinates in every source assignment.
    #[must_use]
    pub const fn verifier_dag_commit(&self) -> Option<Digest> {
        self.verifier_dag_commit
    }

    pub fn get_vk_commit(&self, is_self_recursive: bool) -> VkCommit<PB::Val> {
        if is_self_recursive {
            VkCommit {
                cached_commit: self
                    .self_vk_cached_commitment
                    .expect("self-recursive prover is missing its VK commitment"),
                vk_pre_hash: self.vk.pre_hash,
            }
        } else {
            VkCommit {
                cached_commit: self.child_vk_cached_commitment,
                vk_pre_hash: self.child_vk.pre_hash,
            }
        }
    }

    /// Drop device-only PCS/proving-key state after a caller has authenticated
    /// and copied every fixed column into another setup-bound relation.
    ///
    /// The remaining object can generate the same dynamic verifier traces,
    /// but cannot produce a recursive STARK proof or supply cached PCS data.
    pub fn release_device_proving_data_for_trace_only(&mut self) {
        self.d_pk.take();
        self.child_vk_pcs_data.take();
        self.self_vk_pcs_data.take();
    }

    pub fn get_self_vk_pcs_data(&self) -> Option<CommittedTraceData<PB>>
    where
        CommittedTraceData<PB>: Clone,
    {
        self.self_vk_pcs_data.clone()
    }
}

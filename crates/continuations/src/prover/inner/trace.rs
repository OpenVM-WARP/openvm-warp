use itertools::Itertools;
use openvm_recursion_circuit::system::{
    AggregationSubCircuit, CachedTraceCtx, VerifierExternalData, VerifierTraceGen,
};
use openvm_stark_backend::{
    proof::Proof,
    prover::{ProverBackend, ProvingContext},
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, Digest, EF, F,
};
use openvm_verify_stark_host::pvs::{DeferralPvs, VkCommit};
use tracing::instrument;

use super::{ChildVkKind, InnerAggregationProver};
use crate::{
    circuit::inner::{InnerTraceGen, ProofsType},
    SC,
};

impl<PB, S, T> InnerAggregationProver<PB, S, T>
where
    PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    S: AggregationSubCircuit,
    PB::Matrix: Clone,
{
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx<DC>(
        &self,
        proofs: &[Proof<SC>],
        child_vk_kind: ChildVkKind,
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
        device_ctx: &DC,
    ) -> ProvingContext<PB>
    where
        S: VerifierTraceGen<PB, SC, DC>,
        T: InnerTraceGen<PB, DC>,
        DC: Clone + Send + Sync,
    {
        self.generate_proving_ctx_with_optional_heights(
            proofs,
            child_vk_kind,
            proofs_type,
            absent_trace_pvs,
            None,
            device_ctx,
        )
    }

    /// Generate the complete recursive-verifier execution at one setup-fixed
    /// multi-AIR shape.
    ///
    /// This is the trace boundary used by verifier-PESAT WARP: a partial
    /// 1--4-child batch may occupy the same four-child relation index through
    /// ordinary inactive rows inside the existing verifier AIRs.  The heights
    /// are trusted relation parameters, not witness-selected shape tags.
    pub fn generate_fixed_shape_proving_ctx<DC>(
        &self,
        proofs: &[Proof<SC>],
        child_vk_kind: ChildVkKind,
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
        required_heights: &[usize],
        device_ctx: &DC,
    ) -> ProvingContext<PB>
    where
        S: VerifierTraceGen<PB, SC, DC>,
        T: InnerTraceGen<PB, DC>,
        DC: Clone + Send + Sync,
    {
        self.generate_proving_ctx_with_optional_heights(
            proofs,
            child_vk_kind,
            proofs_type,
            absent_trace_pvs,
            Some(required_heights),
            device_ctx,
        )
    }

    fn generate_proving_ctx_with_optional_heights<DC>(
        &self,
        proofs: &[Proof<SC>],
        child_vk_kind: ChildVkKind,
        proofs_type: ProofsType,
        absent_trace_pvs: Option<(DeferralPvs<F>, bool)>,
        required_heights: Option<&[usize]>,
        device_ctx: &DC,
    ) -> ProvingContext<PB>
    where
        S: VerifierTraceGen<PB, SC, DC>,
        T: InnerTraceGen<PB, DC>,
        DC: Clone + Send + Sync,
    {
        assert!(proofs.len() <= self.circuit.verifier_circuit.max_num_proofs());

        let verifier_air_count = self.circuit.verifier_circuit.airs::<SC>().len();
        let post_air_count = if self.circuit.def_hook_cached_commit.is_some() {
            2
        } else {
            0
        };
        let (pre_required, verifier_required, post_required) =
            if let Some(required_heights) = required_heights {
                assert_eq!(
                    required_heights.len(),
                    3 + verifier_air_count + post_air_count,
                    "fixed inner-verifier height vector must cover every AIR"
                );
                let (pre, remainder) = required_heights.split_at(3);
                let (verifier, post) = remainder.split_at(verifier_air_count);
                (Some(pre), Some(verifier), Some(post))
            } else {
                (None, None, None)
            };

        let (child_vk, child_vk_pcs_data, cached_commit) = match child_vk_kind {
            ChildVkKind::RecursiveSelf => {
                let pcs = self.self_vk_pcs_data.clone();
                let commit = self
                    .self_vk_cached_commitment
                    .expect("self-recursive trace generation requires a self VK commitment");
                (&self.vk, pcs, commit)
            }
            _ => (
                &self.child_vk,
                self.child_vk_pcs_data.clone(),
                self.child_vk_cached_commitment,
            ),
        };
        let child_is_app = matches!(child_vk_kind, ChildVkKind::App);
        let child_vk_commit = VkCommit {
            cached_commit,
            vk_pre_hash: child_vk.pre_hash,
        };

        let pre_data = self
            .agg_node_tracegen
            .generate_pre_verifier_subcircuit_ctxs(
                proofs,
                proofs_type,
                absent_trace_pvs,
                child_is_app,
                child_vk_commit,
                pre_required,
                device_ctx,
            );

        let power_check_inputs = vec![];
        let mut external_data = VerifierExternalData {
            poseidon2_compress_inputs: &pre_data.poseidon2_compress_inputs,
            poseidon2_permute_inputs: &pre_data.poseidon2_permute_inputs,
            range_check_inputs: &pre_data.range_check_inputs,
            power_check_inputs: &power_check_inputs,
            required_heights: verifier_required,
            final_transcript_state: None,
        };

        let cached_trace_ctx = if self.verifier_has_cached() {
            child_vk_pcs_data.map_or(CachedTraceCtx::SetupBound, CachedTraceCtx::PcsData)
        } else {
            CachedTraceCtx::Records(self.circuit.verifier_circuit.cached_trace_record(child_vk))
        };
        let subcircuit_ctxs = self
            .circuit
            .verifier_circuit
            .generate_proving_ctxs(
                child_vk,
                cached_trace_ctx,
                proofs,
                &mut external_data,
                device_ctx,
                default_duplex_sponge_recorder(),
            )
            .unwrap();
        let post_ctxs = self
            .agg_node_tracegen
            .generate_post_verifier_subcircuit_ctxs(
                proofs,
                proofs_type,
                child_is_app,
                post_required,
                device_ctx,
            );

        ProvingContext {
            per_trace: pre_data
                .air_proving_ctxs
                .into_iter()
                .chain(subcircuit_ctxs)
                .chain(post_ctxs)
                .enumerate()
                .collect_vec(),
        }
    }
}

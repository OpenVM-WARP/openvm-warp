use std::sync::Arc;

use eyre::Result;
use openvm_recursion_circuit::system::{
    AggregationSubCircuit, VerifierConfig, VerifierTailMode, VerifierTraceGen,
};
use openvm_stark_backend::{
    air_builders::symbolic::SymbolicExpressionNode,
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    prover::{
        CommittedTraceData, DeviceDataTransporter, DeviceMultiStarkProvingKey, MatrixDimensions,
        ProverBackend, ProverDevice, ProvingContext,
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

fn symbolic_node_degree(node: &SymbolicExpressionNode<F>) -> usize {
    match node {
        SymbolicExpressionNode::Variable(variable) => variable.degree_multiple(),
        SymbolicExpressionNode::IsFirstRow
        | SymbolicExpressionNode::IsLastRow
        | SymbolicExpressionNode::IsTransition => 1,
        SymbolicExpressionNode::Constant(_) => 0,
        SymbolicExpressionNode::Add {
            degree_multiple, ..
        }
        | SymbolicExpressionNode::Sub {
            degree_multiple, ..
        }
        | SymbolicExpressionNode::Neg {
            degree_multiple, ..
        }
        | SymbolicExpressionNode::Mul {
            degree_multiple, ..
        } => *degree_multiple,
    }
}

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
        self.maybe_report_fixed_warp_relation_profile(&ctx);
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

    /// Emit a compile-only size model for the selector-free quadratic
    /// AIR+LogUp PESAT. This is deliberately opt-in: it is used to choose the
    /// recursive normalization boundary before allocating a WARP witness.
    fn maybe_report_fixed_warp_relation_profile(&self, ctx: &ProvingContext<PB>) {
        if std::env::var_os("OPENVM_WARP_NORMALIZED_PROFILE").is_none() {
            return;
        }
        self.report_fixed_warp_relation_profile("natural", ctx);
    }

    /// Emit the exact size model for a caller-selected fixed verifier shape.
    ///
    /// This remains a profiling API: it neither changes the relation nor
    /// allocates a WARP message. Callers use it after taking componentwise
    /// maximum AIR heights over a complete block.
    pub fn report_fixed_warp_relation_profile(&self, profile: &str, ctx: &ProvingContext<PB>) {
        let mut common_main_cells = 0u128;
        let mut fixed_cached_trace_cells = 0u128;
        let mut public_value_cells = 0u128;
        let mut air_product_gates = 0u128;
        let mut air_constraint_gates = 0u128;
        let mut interaction_records = 0u128;
        let mut lifted_interaction_records = 0u64;
        let mut interaction_gates = 0u128;
        let mut max_message_len = 0u128;
        let mut max_interaction_message_degree = 0usize;
        let mut max_interaction_count_degree = 0usize;
        let l_skip = self.vk.inner.params.l_skip;

        for (air_id, trace) in &ctx.per_trace {
            let height = trace.common_main.height() as u128;
            common_main_cells += height * trace.common_main.width() as u128;
            fixed_cached_trace_cells += trace
                .cached_mains
                .iter()
                .map(|cached| cached.trace.height() as u128 * cached.trace.width() as u128)
                .sum::<u128>();
            public_value_cells += trace.public_values.len() as u128;

            let symbolic = &self.vk.inner.per_air[*air_id].symbolic_constraints;
            let product_nodes = symbolic
                .constraints
                .nodes
                .iter()
                .filter(|node| matches!(node, SymbolicExpressionNode::Mul { .. }))
                .count() as u128;
            air_product_gates += height * product_nodes;
            air_constraint_gates += height * symbolic.constraints.constraint_idx.len() as u128;
            let message_degree = symbolic
                .interactions
                .iter()
                .flat_map(|interaction| &interaction.message)
                .map(|&node_idx| symbolic_node_degree(&symbolic.constraints.nodes[node_idx]))
                .max()
                .unwrap_or(0);
            let count_degree = symbolic
                .interactions
                .iter()
                .map(|interaction| {
                    symbolic_node_degree(&symbolic.constraints.nodes[interaction.count])
                })
                .max()
                .unwrap_or(0);
            max_interaction_message_degree = max_interaction_message_degree.max(message_degree);
            max_interaction_count_degree = max_interaction_count_degree.max(count_degree);
            let log_height = (height as u64).ilog2() as usize;
            lifted_interaction_records = lifted_interaction_records
                .checked_add(
                    (symbolic.interactions.len() as u64)
                        .checked_shl(log_height.max(l_skip) as u32)
                        .expect("fixed verifier lifted interaction count overflow"),
                )
                .expect("fixed verifier lifted interaction count overflow");
            for interaction in &symbolic.interactions {
                let message_len = interaction.message.len() as u128;
                interaction_records += height;
                interaction_gates += height * (5 + message_len);
                max_message_len = max_message_len.max(message_len);
            }
        }

        let padded_records = if interaction_records == 0 {
            0
        } else {
            interaction_records.next_power_of_two()
        };
        let padding_gates = (padded_records - interaction_records) * 3;
        let tree_gates = padded_records.saturating_sub(1) * 4;
        let main_cells = common_main_cells + fixed_cached_trace_cells;
        let used_witness = main_cells
            + air_product_gates
            + max_message_len
            + interaction_gates
            + padding_gates
            + tree_gates;
        let used_constraints = air_product_gates
            + air_constraint_gates
            + max_message_len
            + interaction_gates
            + padding_gates
            + tree_gates
            + u128::from(interaction_records != 0);
        let padded_witness = used_witness.max(1).next_power_of_two();
        let padded_constraints = used_constraints.max(1).next_power_of_two();

        // WARP natively supports constant-degree PESAT (including CCS), so a
        // direct AIR lowering need not materialize one witness variable and
        // one equation for every multiplication node. Setup-owned cached
        // tables are part of the fixed relation index, not the per-instance
        // witness. Keep this model separate from the quadratic lowering so a
        // feasibility decision cannot accidentally charge avoidable cells or
        // hide the unavoidable common-main and interaction footprint.
        let direct_used_witness = used_witness
            .saturating_sub(air_product_gates)
            .saturating_sub(fixed_cached_trace_cells);
        let direct_used_constraints = used_constraints.saturating_sub(air_product_gates);
        let direct_padded_witness = direct_used_witness.max(1).next_power_of_two();
        let direct_padded_constraints = direct_used_constraints.max(1).next_power_of_two();
        let direct_degree = self.vk.max_constraint_degree().max(3);

        // Exact size model for the smallest complete direct verifier PESAT.
        // Unlike the legacy quadratic CCS estimate above, this retains the
        // verifier trace directly and closes every interaction with the same
        // projective LogUp fraction tree used by the SWIRL prover.  Each EF4
        // numerator/denominator pair occupies eight BabyBear cells. Cached
        // main tables are setup-fixed index data and therefore excluded.
        let n_logup = openvm_stark_backend::calculate_n_logup(l_skip, lifted_interaction_records);
        let logup_leaf_capacity = 1u128
            .checked_shl((l_skip + n_logup) as u32)
            .expect("fixed verifier LogUp leaf capacity overflow");
        let logup_internal_nodes = logup_leaf_capacity.saturating_sub(1);
        let complete_used_witness = common_main_cells
            .checked_add(8u128.saturating_mul(interaction_records))
            .and_then(|value| value.checked_add(8u128.saturating_mul(logup_internal_nodes)))
            .expect("fixed verifier complete witness size overflow");
        let complete_padded_witness = complete_used_witness.max(1).next_power_of_two();
        let beta_power_constraints = 4u128.saturating_mul(max_message_len.saturating_sub(1));
        let complete_semantic_constraints = air_constraint_gates
            .checked_add(8u128.saturating_mul(interaction_records))
            .and_then(|value| value.checked_add(8u128.saturating_mul(logup_internal_nodes)))
            .and_then(|value| value.checked_add(4))
            .and_then(|value| value.checked_add(beta_power_constraints))
            .and_then(|value| value.checked_add(1))
            .expect("fixed verifier complete constraint size overflow");
        let complete_padding_constraints =
            complete_padded_witness.saturating_sub(complete_used_witness);
        let complete_used_constraints = complete_semantic_constraints
            .checked_add(complete_padding_constraints)
            .expect("fixed verifier complete constraint size overflow");
        let complete_padded_constraints = complete_used_constraints.max(1).next_power_of_two();
        let complete_degree = usize::from(self.vk.max_constraint_degree())
            .max(max_interaction_count_degree)
            .max(max_interaction_message_degree.saturating_add(1))
            .max(2);
        let complete_explicit_growth = 4u128
            .saturating_mul(max_message_len.saturating_add(1))
            .saturating_add(1);

        // Smaller complete LogUp lowering used by the high-arity feasibility
        // gate.  For every interaction record the witness stores only the EF4
        // inverse of its transcript-derived denominator. The fresh Appendix-D
        // codeword stores the four BabyBear coordinates of that EF4 inverse,
        // but PESAT itself is over EF4: reconstructing the inverse is linear,
        // so one EF4 equation constrains `denominator * inverse = 1` and one
        // EF4 equation constrains the complete `multiplicity * inverse` sum to
        // zero. PESAT polynomials may have large fixed arithmetic circuits,
        // so the global sum does not require a materialized running-sum or
        // fraction tree. Unused power-of-two message coordinates are
        // existentially free and need no zero constraints.
        let inverse_sum_used_witness = common_main_cells
            .checked_add(4u128.saturating_mul(interaction_records))
            .expect("fixed verifier inverse-sum witness size overflow");
        let inverse_sum_padded_witness = inverse_sum_used_witness.max(1).next_power_of_two();
        let inverse_sum_used_constraints = air_constraint_gates
            .checked_add(interaction_records)
            // Explicit powers are `(p_0, ..., p_L)`: constrain `p_0 = 1`
            // and all L recurrences `p_{j+1} = p_j * beta`.
            .and_then(|value| value.checked_add(max_message_len.saturating_add(1)))
            // One global EF4 inverse-sum equation.
            .and_then(|value| value.checked_add(1))
            .expect("fixed verifier inverse-sum constraint size overflow");
        let inverse_sum_padded_constraints =
            inverse_sum_used_constraints.max(1).next_power_of_two();
        let inverse_sum_degree = usize::from(self.vk.max_constraint_degree())
            .max(max_interaction_count_degree.saturating_add(1))
            .max(max_interaction_message_degree.saturating_add(1))
            .max(2);
        // Shared alpha, shared beta, and `(p_0, ..., p_L)` are all explicit.
        let inverse_sum_explicit_growth = max_message_len.saturating_add(3);

        eprintln!(
            "OPENVM_WARP_NORMALIZED_PROFILE profile={} airs={} main_cells={} common_main_cells={} \
             fixed_cached_trace_cells={} public_value_cells={} interaction_records={} \
             used_witness={} log_witness={} used_constraints={} log_constraints={} \
             air_product_gates={} interaction_gates={} direct_degree={} \
             direct_used_witness={} direct_log_witness={} direct_used_constraints={} \
             direct_log_constraints={} l_skip={} lifted_interaction_records={} n_logup={} \
             logup_leaf_capacity={} complete_used_witness={} complete_log_witness={} \
             complete_semantic_constraints={} complete_padding_constraints={} \
             complete_used_constraints={} complete_log_constraints={} complete_degree={} \
             complete_explicit_growth={} inverse_sum_used_witness={} \
             inverse_sum_log_witness={} inverse_sum_used_constraints={} \
             inverse_sum_log_constraints={} inverse_sum_degree={} \
             inverse_sum_explicit_growth={}",
            profile,
            ctx.per_trace.len(),
            main_cells,
            common_main_cells,
            fixed_cached_trace_cells,
            public_value_cells,
            interaction_records,
            used_witness,
            padded_witness.ilog2(),
            used_constraints,
            padded_constraints.ilog2(),
            air_product_gates,
            interaction_gates,
            direct_degree,
            direct_used_witness,
            direct_padded_witness.ilog2(),
            direct_used_constraints,
            direct_padded_constraints.ilog2(),
            l_skip,
            lifted_interaction_records,
            n_logup,
            logup_leaf_capacity,
            complete_used_witness,
            complete_padded_witness.ilog2(),
            complete_semantic_constraints,
            complete_padding_constraints,
            complete_used_constraints,
            complete_padded_constraints.ilog2(),
            complete_degree,
            complete_explicit_growth,
            inverse_sum_used_witness,
            inverse_sum_padded_witness.ilog2(),
            inverse_sum_used_constraints,
            inverse_sum_padded_constraints.ilog2(),
            inverse_sum_degree,
            inverse_sum_explicit_growth,
        );
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

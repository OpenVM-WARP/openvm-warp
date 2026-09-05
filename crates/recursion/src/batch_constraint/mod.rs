use core::iter::zip;
use std::sync::Arc;

use itertools::Itertools;
use openvm_cpu_backend::CpuBackend;
use openvm_stark_backend::{
    air_builders::symbolic::{symbolic_variable::Entry, SymbolicExpressionNode},
    keygen::types::MultiStarkVerifyingKey,
    poly_common::{eval_eq_sharp_uni, eval_eq_uni, eval_eq_uni_at_one},
    proof::{column_openings_by_rot, BatchConstraintProof, Proof},
    prover::{AirProvingContext, CommittedTraceData, TraceCommitter},
    AirRef, FiatShamirTranscript, StarkEngine, StarkProtocolConfig, TranscriptHistory,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, EF, F};
use p3_baby_bear::BabyBear;
use p3_field::{Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;
use p3_maybe_rayon::prelude::{IntoParallelRefIterator, ParallelIterator};

use crate::{
    batch_constraint::{
        bus::{
            BatchConstraintConductorBus, BatchConstraintEndpointBus,
            BatchConstraintEndpointClaimBus, ConstraintsFoldingBus, Eq3bBus, EqNOuterBus,
            EqNegInternalBus, EqSharpUniBus, EqZeroNBus, ExpressionClaimBus,
            InteractionsFoldingBus, SumcheckClaimBus, SymbolicExpressionBus,
            UnivariateSumcheckInputBus,
        },
        eq_airs::{
            generate_eq_sharp_uni_blob, Eq3bAir, Eq3bBlob, EqNegAir, EqNegTraceGenerator, EqNsAir,
            EqSharpUniAir, EqSharpUniBlob, EqSharpUniReceiverAir, EqUniAir,
        },
        expr_eval::{
            CachedTraceRecord, ConstraintsFoldingAir, ConstraintsFoldingBlob, DagCommitSubAir,
            InteractionsFoldingAir, InteractionsFoldingBlob, SymbolicExpressionAir,
        },
        expression_claim::{
            generate_expression_claim_blob_for_mode, ExpressionClaimAir, ExpressionClaimBlob,
            ExpressionClaimTraceGenerator,
        },
        fractions_folder::{FractionsFolderAir, FractionsFolderTraceGenerator},
        partial::{
            PartialBatchConstraintEndpointAir, PartialBatchConstraintEndpointTraceGenerator,
        },
        sumcheck::{
            multilinear::MultilinearSumcheckTraceGenerator,
            univariate::UnivariateSumcheckTraceGenerator, MultilinearSumcheckAir,
            UnivariateSumcheckAir,
        },
    },
    bus::{
        AirPresenceBus, AirShapeBus, BatchConstraintModuleBus, ColumnClaimsBus,
        ConstraintSumcheckRandomnessBus, ConstraintsFoldingInputBus, Eq3bShapeBus,
        EqNegBaseRandBus, EqNegResultBus, EqNsNLogupMaxBus, ExpressionClaimNMaxBus,
        FractionFolderInputBus, HyperdimBus, InteractionsFoldingInputBus, NLiftBus,
        PublicValuesBus, SelHypercubeBus, SelUniBus, StackingModuleBus, TranscriptBus,
        TranscriptEndIndexBus, XiRandomnessBus,
    },
    primitives::{bus::PowerCheckerBus, pow::PowerCheckerCpuTraceGenerator},
    system::{
        AirModule, BatchConstraintPreflight, BusIndexManager, BusInventory, GlobalCtxCpu,
        Preflight, TraceGenModule, VerifierEquationMode, POW_CHECKER_HEIGHT,
    },
    tracegen::{ModuleChip, RowMajorChip, StandardTracegenCtx},
    utils::{MultiProofVecVec, MultiVecWithBounds},
};

pub mod bus;
pub mod eq_airs;
pub mod expr_eval;
pub mod expression_claim;
pub mod fractions_folder;
pub mod partial;
pub mod sumcheck;

#[cfg(feature = "cuda")]
mod cuda_abi;
#[cfg(feature = "cuda")]
mod cuda_utils;

/// AIR index within the BatchConstraintModule
pub const LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX: usize = 0;
/// Stable local AIR index of [`ExpressionClaimAir`]. Partial recursion
/// assemblies use this to replace only the terminal claim-fold trace while
/// retaining the standard GKR, sumcheck, and expression-evaluation AIRs.
pub const LOCAL_EXPRESSION_CLAIM_AIR_IDX: usize = 9;

#[derive(Clone, Copy, Debug)]
pub struct PartialBatchConstraintExports {
    pub endpoint_bus: BatchConstraintEndpointBus,
    pub column_claims_bus: ColumnClaimsBus,
    pub opening_point_bus: crate::batch_constraint::bus::LogUpOnlyOpeningPointBus,
}

pub struct BatchConstraintModule {
    transcript_bus: TranscriptBus,
    transcript_end_index_bus: TranscriptEndIndexBus,
    constraint_sumcheck_randomness_bus: ConstraintSumcheckRandomnessBus,
    xi_randomness_bus: XiRandomnessBus,
    gkr_claim_bus: BatchConstraintModuleBus,
    constraints_folding_input_bus: ConstraintsFoldingInputBus,
    interactions_folding_input_bus: InteractionsFoldingInputBus,
    fraction_folder_input_bus: FractionFolderInputBus,
    univariate_sumcheck_input_bus: UnivariateSumcheckInputBus,
    stacking_module_bus: StackingModuleBus,
    column_opening_bus: ColumnClaimsBus,
    air_shape_bus: AirShapeBus,
    air_presence_bus: AirPresenceBus,
    hyperdim_bus: HyperdimBus,
    public_values_bus: PublicValuesBus,
    sel_uni_bus: SelUniBus,
    eq_n_outer_bus: EqNOuterBus,

    batch_constraint_conductor_bus: BatchConstraintConductorBus,
    sumcheck_bus: SumcheckClaimBus,
    expression_claim_n_max_bus: ExpressionClaimNMaxBus,
    n_lift_bus: NLiftBus,
    eq_n_logup_n_max_bus: EqNsNLogupMaxBus,
    eq_3b_shape_bus: Eq3bShapeBus,

    zero_n_bus: EqZeroNBus,
    eq_sharp_uni_bus: EqSharpUniBus,
    eq_3b_bus: Eq3bBus,
    sel_hypercube_bus: SelHypercubeBus,
    eq_neg_result_bus: EqNegResultBus,
    eq_neg_base_rand_bus: EqNegBaseRandBus,
    eq_neg_internal_bus: EqNegInternalBus,

    symbolic_expression_bus: SymbolicExpressionBus,
    expression_claim_bus: ExpressionClaimBus,
    interactions_folding_bus: InteractionsFoldingBus,
    constraints_folding_bus: ConstraintsFoldingBus,
    power_checker_bus: PowerCheckerBus,

    l_skip: usize,
    max_constraint_degree: usize,

    max_num_proofs: usize,
    pub(crate) has_cached: bool,
    /// In no-cached mode, optionally bind the reconstructed symbolic DAG to
    /// this verifier-key-owned digest instead of exposing the digest as PIs.
    fixed_dag_commit: Option<[F; openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE]>,
    equation_mode: VerifierEquationMode,
    endpoint_claim_bus: Option<BatchConstraintEndpointClaimBus>,
    endpoint_bus: Option<BatchConstraintEndpointBus>,
    opening_point_export_bus: Option<crate::batch_constraint::bus::LogUpOnlyOpeningPointBus>,
    deferred_stacking: bool,
}

impl BatchConstraintModule {
    pub fn new(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
        max_num_proofs: usize,
        has_cached: bool,
    ) -> Self {
        Self::new_with_equation_mode(
            child_vk,
            b,
            bus_inventory,
            max_num_proofs,
            has_cached,
            VerifierEquationMode::AirAndLogUp,
        )
    }

    /// Standard AIR-plus-LogUp reduction that internalizes the opening-point
    /// fanout because stacking is deferred to a transcript-linked terminal proof.
    pub fn new_deferred_stacking(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
        max_num_proofs: usize,
        has_cached: bool,
    ) -> Self {
        let mut module = Self::new(child_vk, b, bus_inventory, max_num_proofs, has_cached);
        module.deferred_stacking = true;
        module
    }

    pub fn new_with_equation_mode(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
        max_num_proofs: usize,
        has_cached: bool,
        equation_mode: VerifierEquationMode,
    ) -> Self {
        let l_skip = child_vk.inner.params.l_skip;
        let max_constraint_degree = child_vk.max_constraint_degree();
        let endpoint_claim_bus = (!equation_mode.includes_air())
            .then(|| BatchConstraintEndpointClaimBus::new(b.new_bus_idx()));
        let endpoint_bus = (!equation_mode.includes_air())
            .then(|| BatchConstraintEndpointBus::new(b.new_bus_idx()));
        let opening_point_export_bus = (!equation_mode.includes_air())
            .then(|| crate::batch_constraint::bus::LogUpOnlyOpeningPointBus::new(b.new_bus_idx()));
        BatchConstraintModule {
            transcript_bus: bus_inventory.transcript_bus,
            transcript_end_index_bus: bus_inventory.transcript_end_index_bus,
            constraint_sumcheck_randomness_bus: bus_inventory.constraint_randomness_bus,
            xi_randomness_bus: bus_inventory.xi_randomness_bus,
            gkr_claim_bus: bus_inventory.bc_module_bus,
            constraints_folding_input_bus: bus_inventory.constraints_folding_input_bus,
            interactions_folding_input_bus: bus_inventory.interactions_folding_input_bus,
            fraction_folder_input_bus: bus_inventory.fraction_folder_input_bus,
            stacking_module_bus: bus_inventory.stacking_module_bus,
            column_opening_bus: bus_inventory.column_claims_bus,
            air_shape_bus: bus_inventory.air_shape_bus,
            air_presence_bus: bus_inventory.air_presence_bus,
            hyperdim_bus: bus_inventory.hyperdim_bus,
            public_values_bus: bus_inventory.public_values_bus,
            sel_uni_bus: bus_inventory.sel_uni_bus,
            eq_neg_base_rand_bus: bus_inventory.eq_neg_base_rand_bus,
            eq_neg_result_bus: bus_inventory.eq_neg_result_bus,
            expression_claim_n_max_bus: bus_inventory.expression_claim_n_max_bus,
            n_lift_bus: bus_inventory.n_lift_bus,
            eq_n_logup_n_max_bus: bus_inventory.eq_n_logup_n_max_bus,
            eq_3b_shape_bus: bus_inventory.eq_3b_shape_bus,
            batch_constraint_conductor_bus: BatchConstraintConductorBus::new(b.new_bus_idx()),
            univariate_sumcheck_input_bus: UnivariateSumcheckInputBus::new(b.new_bus_idx()),
            sumcheck_bus: SumcheckClaimBus::new(b.new_bus_idx()),

            zero_n_bus: EqZeroNBus::new(b.new_bus_idx()),
            eq_sharp_uni_bus: EqSharpUniBus::new(b.new_bus_idx()),
            eq_3b_bus: Eq3bBus::new(b.new_bus_idx()),
            eq_neg_internal_bus: EqNegInternalBus::new(b.new_bus_idx()),
            sel_hypercube_bus: SelHypercubeBus::new(b.new_bus_idx()),
            eq_n_outer_bus: EqNOuterBus::new(b.new_bus_idx()),
            // sel_uni bus is shared via inventory
            symbolic_expression_bus: SymbolicExpressionBus::new(b.new_bus_idx()),
            expression_claim_bus: ExpressionClaimBus::new(b.new_bus_idx()),
            interactions_folding_bus: InteractionsFoldingBus::new(b.new_bus_idx()),
            constraints_folding_bus: ConstraintsFoldingBus::new(b.new_bus_idx()),
            power_checker_bus: bus_inventory.power_checker_bus,
            l_skip,
            max_constraint_degree,
            max_num_proofs,
            has_cached,
            fixed_dag_commit: None,
            equation_mode,
            endpoint_claim_bus,
            endpoint_bus,
            opening_point_export_bus,
            deferred_stacking: false,
        }
    }

    /// Internalize the no-cached symbolic-DAG commitment as a fixed relation
    /// constant. Cached mode cannot use this because its trace commitment
    /// requires a different authority mechanism.
    pub fn bind_fixed_dag_commit(
        &mut self,
        expected: [F; openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE],
    ) -> Result<(), &'static str> {
        if self.has_cached || self.fixed_dag_commit.replace(expected).is_some() {
            return Err("symbolic DAG commitment authority is incompatible or duplicated");
        }
        Ok(())
    }

    #[must_use]
    pub(crate) const fn fixed_dag_commit(
        &self,
    ) -> Option<[F; openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE]> {
        self.fixed_dag_commit
    }

    #[must_use]
    pub const fn equation_mode(&self) -> VerifierEquationMode {
        self.equation_mode
    }

    /// Buses a caller must consume/provide when stacking and WHIR are omitted.
    #[must_use]
    pub fn partial_exports(&self) -> Option<PartialBatchConstraintExports> {
        Some(PartialBatchConstraintExports {
            endpoint_bus: self.endpoint_bus?,
            column_claims_bus: self.column_opening_bus,
            opening_point_bus: self.opening_point_export_bus?,
        })
    }

    /// Internal expression-claim permutation bus, exposed for sound partial
    /// assemblies such as reduced-SWIRL's interaction-only LogUp verifier.
    #[must_use]
    pub const fn expression_claim_bus(&self) -> ExpressionClaimBus {
        self.expression_claim_bus
    }

    /// Internal final-sumcheck-claim bus, exposed so a partial assembly can
    /// relay the verifier-derived endpoint without trusting a host sidecar.
    #[must_use]
    pub const fn sumcheck_claim_bus(&self) -> SumcheckClaimBus {
        self.sumcheck_bus
    }

    #[tracing::instrument(level = "trace", skip_all)]
    pub fn run_preflight<TS>(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proof: &Proof<BabyBearPoseidon2Config>,
        preflight: &mut Preflight,
        ts: &mut TS,
    ) where
        TS: FiatShamirTranscript<BabyBearPoseidon2Config> + TranscriptHistory,
    {
        let BatchConstraintProof {
            numerator_term_per_air,
            denominator_term_per_air,
            univariate_round_coeffs,
            sumcheck_round_polys,
            column_openings,
        } = &proof.batch_constraint_proof;

        assert_eq!(
            preflight.rebased_transcript.is_some(),
            self.equation_mode == VerifierEquationMode::LogUpOnly,
            "equation mode and transcript framing disagree"
        );
        let mut sumcheck_rnd = vec![];

        let mut xi = preflight.gkr.xi.iter().map(|(_, x)| *x).collect_vec();
        let l_skip = preflight.proof_shape.l_skip;
        let n_global = preflight.proof_shape.n_global();
        for _ in xi.len()..(l_skip + n_global) {
            xi.push(ts.sample_ext());
        }

        // Constraint batching
        let lambda_tidx = ts.len();
        let _lambda = ts.sample_ext();

        for (sum_claim_p, sum_claim_q) in zip(numerator_term_per_air, denominator_term_per_air) {
            ts.observe_ext(*sum_claim_p);
            ts.observe_ext(*sum_claim_q);
        }
        let _mu = ts.sample_ext();

        let tidx_before_univariate = ts.len();

        // univariate round
        for coef in univariate_round_coeffs {
            ts.observe_ext(*coef);
        }
        let r0 = ts.sample_ext();
        sumcheck_rnd.push(r0);

        let tidx_before_multilinear = ts.len();

        for polys in sumcheck_round_polys {
            for eval in polys {
                ts.observe_ext(*eval);
            }
            let ri = ts.sample_ext();
            sumcheck_rnd.push(ri);
        }

        let tidx_before_column_openings = ts.len();

        if self.equation_mode.includes_air() {
            // Common main
            for (sort_idx, (air_id, _)) in
                preflight.proof_shape.sorted_trace_vdata.iter().enumerate()
            {
                let need_rot = child_vk.inner.per_air[*air_id].params.need_rot;
                for (col_opening, rot_opening) in
                    column_openings_by_rot(&column_openings[sort_idx][0], need_rot)
                {
                    ts.observe_ext(col_opening);
                    ts.observe_ext(rot_opening);
                }
            }

            for (sort_idx, (air_id, _)) in
                preflight.proof_shape.sorted_trace_vdata.iter().enumerate()
            {
                let need_rot = child_vk.inner.per_air[*air_id].params.need_rot;
                for part in column_openings[sort_idx].iter().skip(1) {
                    for (col_opening, rot_opening) in column_openings_by_rot(part, need_rot) {
                        ts.observe_ext(col_opening);
                        ts.observe_ext(rot_opening);
                    }
                }
            }
        }

        let mut final_claim = univariate_round_coeffs
            .iter()
            .rev()
            .fold(EF::ZERO, |acc, &coefficient| acc * r0 + coefficient);
        for (round, &r) in sumcheck_round_polys.iter().zip(sumcheck_rnd.iter().skip(1)) {
            final_claim = interpolate_batch_round(final_claim, round, r);
        }

        let omega_skip_pows = F::two_adic_generator(l_skip)
            .powers()
            .take(1 << l_skip)
            .collect_vec();

        let mut eq_ns = Vec::with_capacity(preflight.proof_shape.n_max + 1);
        let mut eq_sharp_ns = Vec::with_capacity(preflight.proof_shape.n_max + 1);
        let mut eq = eval_eq_uni(l_skip, xi[0], sumcheck_rnd[0]);
        let mut eq_sharp = eval_eq_sharp_uni(&omega_skip_pows, &xi[..l_skip], sumcheck_rnd[0]);
        eq_ns.push(eq);
        eq_sharp_ns.push(eq_sharp);
        for i in 0..preflight.proof_shape.n_max {
            let mult = EF::ONE - xi[l_skip + i] - sumcheck_rnd[1 + i]
                + (xi[l_skip + i] * sumcheck_rnd[1 + i]).double();
            eq *= mult;
            eq_sharp *= mult;
            eq_ns.push(eq);
            eq_sharp_ns.push(eq_sharp);
        }

        let mut r_rev_prod = sumcheck_rnd[preflight.proof_shape.n_max];
        let mut eq_ns_frontloaded = Vec::with_capacity(preflight.proof_shape.n_max + 1);
        let mut eq_sharp_ns_frontloaded = Vec::with_capacity(preflight.proof_shape.n_max + 1);
        // Product with r_i's to account for \hat{f} vs \tilde{f} for different n's in front-loaded
        // batch sumcheck.
        for i in (0..preflight.proof_shape.n_max).rev() {
            eq_ns_frontloaded.push(eq_ns[i] * r_rev_prod);
            eq_sharp_ns_frontloaded.push(eq_sharp_ns[i] * r_rev_prod);
            r_rev_prod *= sumcheck_rnd[i];
        }
        eq_ns_frontloaded.reverse();
        eq_sharp_ns_frontloaded.reverse();
        eq_ns_frontloaded.push(eq_ns[preflight.proof_shape.n_max]);
        eq_sharp_ns_frontloaded.push(eq_sharp_ns[preflight.proof_shape.n_max]);

        preflight.batch_constraint = BatchConstraintPreflight {
            equation_mode: self.equation_mode,
            lambda_tidx,
            tidx_before_univariate,
            tidx_before_multilinear,
            tidx_before_column_openings,
            post_tidx: ts.len(),
            xi,
            sumcheck_rnd,
            eq_ns,
            eq_sharp_ns,
            eq_ns_frontloaded,
            eq_sharp_ns_frontloaded,
            final_claim,
        }
    }
}

fn interpolate_batch_round(cur_sum: EF, evaluations_at_1: &[EF], point: EF) -> EF {
    let degree = evaluations_at_1.len();
    let evaluations = core::iter::once(cur_sum - evaluations_at_1[0])
        .chain(evaluations_at_1.iter().copied())
        .collect_vec();
    (0..=degree)
        .map(|i| {
            let i_f = EF::from_usize(i);
            let mut numerator = EF::ONE;
            let mut denominator = EF::ONE;
            for j in 0..=degree {
                if i != j {
                    let j_f = EF::from_usize(j);
                    numerator *= point - j_f;
                    denominator *= i_f - j_f;
                }
            }
            evaluations[i] * numerator * denominator.inverse()
        })
        .sum()
}

impl AirModule for BatchConstraintModule {
    fn num_airs(&self) -> usize {
        13
    }

    fn airs<SC: StarkProtocolConfig<F = BabyBear>>(&self) -> Vec<AirRef<SC>> {
        let l_skip = self.l_skip;

        let symbolic_expression_air = SymbolicExpressionAir::<BabyBear> {
            expr_bus: self.symbolic_expression_bus,
            air_shape_bus: self.air_shape_bus,
            air_presence_bus: self.air_presence_bus,
            column_claims_bus: self.column_opening_bus,
            interactions_folding_bus: self.interactions_folding_bus,
            constraints_folding_bus: self.constraints_folding_bus,
            hyperdim_bus: self.hyperdim_bus,
            public_values_bus: self.public_values_bus,
            sel_hypercube_bus: self.sel_hypercube_bus,
            sel_uni_bus: self.sel_uni_bus,
            cnt_proofs: self.max_num_proofs,
            dag_commit_subair: (!self.has_cached).then(|| {
                Arc::new(self.fixed_dag_commit.map_or_else(
                    DagCommitSubAir::new,
                    DagCommitSubAir::new_with_expected_commit,
                ))
            }),
            includes_air: self.equation_mode.includes_air(),
        };
        let fraction_folder_air = FractionsFolderAir {
            transcript_bus: self.transcript_bus,
            univariate_sumcheck_input_bus: self.univariate_sumcheck_input_bus,
            fraction_folder_input_bus: self.fraction_folder_input_bus,
            sumcheck_bus: self.sumcheck_bus,
            mu_bus: self.batch_constraint_conductor_bus,
            gkr_claim_bus: self.gkr_claim_bus,
        };
        let sumcheck_uni_air = UnivariateSumcheckAir {
            l_skip,
            univariate_deg: (self.max_constraint_degree + 1) * ((1 << l_skip) - 1),
            univariate_sumcheck_input_bus: self.univariate_sumcheck_input_bus,
            stacking_module_bus: self.stacking_module_bus,
            claim_bus: self.sumcheck_bus,
            transcript_bus: self.transcript_bus,
            randomness_bus: self.constraint_sumcheck_randomness_bus,
            batch_constraint_conductor_bus: self.batch_constraint_conductor_bus,
            opening_point_export_bus: self.opening_point_export_bus,
        };
        let sumcheck_lin_air = MultilinearSumcheckAir {
            max_constraint_degree: self.max_constraint_degree,
            claim_bus: self.sumcheck_bus,
            transcript_bus: self.transcript_bus,
            randomness_bus: self.constraint_sumcheck_randomness_bus,
            batch_constraint_conductor_bus: self.batch_constraint_conductor_bus,
            stacking_module_bus: self.stacking_module_bus,
            opening_point_export_bus: self.opening_point_export_bus,
        };
        let eq_ns_air = EqNsAir {
            zero_n_bus: self.zero_n_bus,
            xi_bus: self.xi_randomness_bus,
            r_xi_bus: self.batch_constraint_conductor_bus,
            sel_hypercube_bus: self.sel_hypercube_bus,
            eq_n_outer_bus: self.eq_n_outer_bus,
            eq_n_logup_n_max_bus: self.eq_n_logup_n_max_bus,
            constraint_randomness_bus: self.constraint_sumcheck_randomness_bus,
            l_skip,
            includes_air: self.equation_mode.includes_air(),
            consume_constraint_randomness: !self.equation_mode.includes_air()
                || self.deferred_stacking,
        };
        let eq_3b_air = Eq3bAir {
            eq_3b_bus: self.eq_3b_bus,
            eq_3b_shape_bus: self.eq_3b_shape_bus,
            batch_constraint_conductor_bus: self.batch_constraint_conductor_bus,
            l_skip,
        };
        let eq_sharp_uni_air = EqSharpUniAir {
            xi_bus: self.xi_randomness_bus,
            eq_bus: self.eq_sharp_uni_bus,
            batch_constraint_conductor_bus: self.batch_constraint_conductor_bus,
            l_skip,
            canonical_inverse_generator: F::two_adic_generator(l_skip).inverse(),
        };
        let eq_sharp_uni_receiver_air = EqSharpUniReceiverAir {
            r_bus: self.batch_constraint_conductor_bus,
            eq_bus: self.eq_sharp_uni_bus,
            zero_n_bus: self.zero_n_bus,
            l_skip,
        };
        let eq_uni_air = EqUniAir {
            r_xi_bus: self.batch_constraint_conductor_bus,
            zero_n_bus: self.zero_n_bus,
            l_skip,
        };
        let eq_neg_air = EqNegAir {
            result_bus: self.eq_neg_result_bus,
            base_rand_bus: self.eq_neg_base_rand_bus,
            internal_bus: self.eq_neg_internal_bus,
            sel_uni_bus: self.sel_uni_bus,
            constraint_randomness_bus: self.constraint_sumcheck_randomness_bus,
            l_skip: self.l_skip,
            emit_stacking_outputs: self.equation_mode.includes_air() && !self.deferred_stacking,
        };
        let expression_claim_air = ExpressionClaimAir {
            expression_claim_n_max_bus: self.expression_claim_n_max_bus,
            expr_claim_bus: self.expression_claim_bus,
            mu_bus: self.batch_constraint_conductor_bus,
            sumcheck_claim_bus: self.sumcheck_bus,
            eq_n_outer_bus: self.eq_n_outer_bus,
            pow_checker_bus: self.power_checker_bus,
            hyperdim_bus: self.hyperdim_bus,
            includes_air: self.equation_mode.includes_air(),
            endpoint_claim_bus: self.endpoint_claim_bus,
        };
        let interactions_folding_air = InteractionsFoldingAir {
            transcript_bus: self.transcript_bus,
            air_shape_bus: self.air_shape_bus,
            interaction_bus: self.interactions_folding_bus,
            interactions_folding_input_bus: self.interactions_folding_input_bus,
            expression_claim_bus: self.expression_claim_bus,
            eq_3b_bus: self.eq_3b_bus,
        };
        let constraints_folding_air = ConstraintsFoldingAir {
            transcript_bus: self.transcript_bus,
            constraint_bus: self.constraints_folding_bus,
            expression_claim_bus: self.expression_claim_bus,
            eq_n_outer_bus: self.eq_n_outer_bus,
            n_lift_bus: self.n_lift_bus,
            air_shape_bus: self.air_shape_bus,
            constraints_folding_input_bus: self.constraints_folding_input_bus,
        };
        let mut airs = vec![
            Arc::new(symbolic_expression_air) as AirRef<_>,
            Arc::new(fraction_folder_air) as AirRef<_>,
            Arc::new(sumcheck_uni_air) as AirRef<_>,
            Arc::new(sumcheck_lin_air) as AirRef<_>,
            Arc::new(eq_ns_air) as AirRef<_>,
            Arc::new(eq_3b_air) as AirRef<_>,
            Arc::new(eq_sharp_uni_air) as AirRef<_>,
            Arc::new(eq_sharp_uni_receiver_air) as AirRef<_>,
            Arc::new(eq_uni_air) as AirRef<_>,
            Arc::new(expression_claim_air) as AirRef<_>,
            Arc::new(interactions_folding_air) as AirRef<_>,
        ];
        if let Some(endpoint_bus) = self.endpoint_bus {
            airs.push(Arc::new(PartialBatchConstraintEndpointAir {
                transcript_bus: self.transcript_bus,
                stacking_module_bus: self.stacking_module_bus,
                transcript_end_index_bus: self.transcript_end_index_bus,
                claim_bus: self
                    .endpoint_claim_bus
                    .expect("partial endpoint claim bus must exist"),
                endpoint_bus,
            }) as AirRef<_>);
        } else {
            airs.push(Arc::new(constraints_folding_air) as AirRef<_>);
        }
        airs.push(Arc::new(eq_neg_air) as AirRef<_>);
        // WARNING: SymbolicExpressionAir MUST be the first AIR in verifier circuit.
        airs
    }
}

pub(crate) struct BatchConstraintBlob {
    // Per proof, per air (vkey order), the evaluations. For optional AIRs without traces, the
    // innermost vec is empty.
    pub expr_evals: MultiVecWithBounds<EF, 2>,
    // Per proof, per log height.
    pub selector_counts: MultiVecWithBounds<SelectorCount, 1>,

    pub eq_3b_blob: Eq3bBlob,
    pub eq_sharp_uni_blob: EqSharpUniBlob,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct SelectorCount {
    pub first: usize,
    pub last: usize,
    pub transition: usize,
}

impl BatchConstraintBlob {
    pub fn new(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proofs: &[&Proof<BabyBearPoseidon2Config>],
        preflights: &[&Preflight],
    ) -> Self {
        let child_vk = &child_vk.inner;
        let params = &child_vk.params;

        let mut expr_evals_per_proof = MultiVecWithBounds::new();
        let mut eq_r_one_counts_per_proof = MultiVecWithBounds::new();
        for (proof, preflight) in zip(proofs, preflights) {
            let rs = &preflight.batch_constraint.sumcheck_rnd;

            let (&rs_0, rs_rest) = rs.split_first().unwrap();
            let mut is_first_row_by_log_height = vec![];
            let mut is_last_row_by_log_height = vec![];
            let n_max = preflight.proof_shape.n_max;
            let mut selector_counts =
                vec![SelectorCount::default(); child_vk.params.l_skip + n_max + 1];

            let omega = F::two_adic_generator(params.l_skip);
            for log_height in 0..=params.l_skip {
                is_first_row_by_log_height.push(eval_eq_uni_at_one(
                    log_height,
                    rs_0.exp_power_of_2(params.l_skip - log_height),
                ));
                is_last_row_by_log_height.push(eval_eq_uni_at_one(
                    log_height,
                    (rs_0 * omega).exp_power_of_2(params.l_skip - log_height),
                ));
            }
            for (i, &r) in rs_rest.iter().enumerate() {
                is_first_row_by_log_height
                    .push(is_first_row_by_log_height[params.l_skip + i] * (EF::ONE - r));
                is_last_row_by_log_height.push(is_last_row_by_log_height[params.l_skip + i] * r);
            }

            let mut expr_evals_per_air = vec![];
            for (air_idx, vk) in child_vk.per_air.iter().enumerate() {
                if proof.trace_vdata[air_idx].is_none() {
                    expr_evals_per_air.push(vec![]);
                    continue;
                }

                let need_rot = child_vk.per_air[air_idx].params.need_rot;
                let openings_per_col = if need_rot { 2 } else { 1 };
                let openings = &proof.batch_constraint_proof.column_openings;
                let (sorted_idx, vdata) = preflight
                    .proof_shape
                    .sorted_trace_vdata
                    .iter()
                    .enumerate()
                    .find_map(|(sorted_idx, (idx, vdata))| {
                        if air_idx == *idx {
                            Some((sorted_idx, vdata))
                        } else {
                            None
                        }
                    })
                    .unwrap();

                let constraints = &vk.symbolic_constraints.constraints;
                let mut expr_evals =
                    vec![EF::ZERO; constraints.nodes.len() + vk.unused_variables.len()];
                let log_height = proof.trace_vdata[air_idx].as_ref().unwrap().log_height;

                for (node_idx, node) in constraints.nodes.iter().enumerate() {
                    match node {
                        SymbolicExpressionNode::Variable(var) => match var.entry {
                            Entry::Preprocessed { offset } => {
                                debug_assert!(offset < openings_per_col);
                                expr_evals[node_idx] =
                                    openings[sorted_idx][1][var.index * openings_per_col + offset];
                            }
                            Entry::Main { part_index, offset } => {
                                let part = vk.dag_main_part_index_to_commit_index(part_index);
                                debug_assert!(offset < openings_per_col);
                                expr_evals[node_idx] = openings[sorted_idx][part]
                                    [var.index * openings_per_col + offset];
                            }
                            Entry::Public => {
                                expr_evals[node_idx] =
                                    EF::from(proof.public_values[air_idx][var.index]);
                            }
                            Entry::Challenge => unreachable!(),
                        },
                        SymbolicExpressionNode::IsFirstRow => {
                            expr_evals[node_idx] = is_first_row_by_log_height[vdata.log_height];
                            selector_counts[log_height].first += 1;
                        }
                        SymbolicExpressionNode::IsLastRow => {
                            expr_evals[node_idx] = is_last_row_by_log_height[vdata.log_height];
                            selector_counts[log_height].last += 1;
                        }
                        SymbolicExpressionNode::IsTransition => {
                            expr_evals[node_idx] =
                                EF::ONE - is_last_row_by_log_height[vdata.log_height];
                            selector_counts[log_height].transition += 1;
                        }
                        SymbolicExpressionNode::Constant(val) => {
                            expr_evals[node_idx] = EF::from(*val);
                        }
                        SymbolicExpressionNode::Add {
                            left_idx,
                            right_idx,
                            degree_multiple: _,
                        } => {
                            debug_assert!(*left_idx < node_idx);
                            debug_assert!(*right_idx < node_idx);
                            expr_evals[node_idx] = expr_evals[*left_idx] + expr_evals[*right_idx];
                        }
                        SymbolicExpressionNode::Sub {
                            left_idx,
                            right_idx,
                            degree_multiple: _,
                        } => {
                            debug_assert!(*left_idx < node_idx);
                            debug_assert!(*right_idx < node_idx);
                            expr_evals[node_idx] = expr_evals[*left_idx] - expr_evals[*right_idx];
                        }
                        SymbolicExpressionNode::Neg {
                            idx,
                            degree_multiple: _,
                        } => {
                            debug_assert!(*idx < node_idx);
                            expr_evals[node_idx] = -expr_evals[*idx];
                        }
                        SymbolicExpressionNode::Mul {
                            left_idx,
                            right_idx,
                            degree_multiple: _,
                        } => {
                            debug_assert!(*left_idx < node_idx);
                            debug_assert!(*right_idx < node_idx);
                            expr_evals[node_idx] = expr_evals[*left_idx] * expr_evals[*right_idx];
                        }
                    };
                }
                let mut node_idx = constraints.nodes.len();
                for unused_var in &vk.unused_variables {
                    match unused_var.entry {
                        Entry::Preprocessed { offset } => {
                            debug_assert!(offset < openings_per_col);
                            expr_evals[node_idx] = openings[sorted_idx][1]
                                [unused_var.index * openings_per_col + offset];
                        }
                        Entry::Main { part_index, offset } => {
                            let part = vk.dag_main_part_index_to_commit_index(part_index);
                            debug_assert!(offset < openings_per_col);
                            expr_evals[node_idx] = openings[sorted_idx][part]
                                [unused_var.index * openings_per_col + offset];
                        }
                        Entry::Public | Entry::Challenge => {
                            unreachable!()
                        }
                    }
                    node_idx += 1;
                }
                expr_evals_per_air.push(expr_evals);
            }
            for v in expr_evals_per_air {
                expr_evals_per_proof.extend(v);
                expr_evals_per_proof.close_level(1);
            }
            expr_evals_per_proof.close_level(0);
            eq_r_one_counts_per_proof.extend(selector_counts);
            eq_r_one_counts_per_proof.close_level(0);
        }
        let eq_3b_blob = eq_airs::generate_eq_3b_blob(child_vk, preflights);
        let eq_sharp_uni_blob = generate_eq_sharp_uni_blob(child_vk, preflights);
        Self {
            expr_evals: expr_evals_per_proof,
            selector_counts: eq_r_one_counts_per_proof,
            eq_3b_blob,
            eq_sharp_uni_blob,
        }
    }
}

pub(crate) struct BatchConstraintBlobCpu {
    pub common_blob: BatchConstraintBlob,
    pub cf_blob: Option<ConstraintsFoldingBlob>,
    pub if_blob: Option<InteractionsFoldingBlob>,
    pub expr_claim_blob: ExpressionClaimBlob,
}

impl BatchConstraintBlobCpu {
    #[tracing::instrument(name = "generate_blob", skip_all)]
    pub fn new(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        preflights: &[Preflight],
        equation_mode: VerifierEquationMode,
    ) -> Self {
        let proofs = proofs.iter().collect_vec();
        let preflights = preflights.iter().collect_vec();
        let common_blob = BatchConstraintBlob::new(child_vk, &proofs, &preflights);
        let cf_blob = equation_mode.includes_air().then(|| {
            ConstraintsFoldingBlob::new(&child_vk.inner, &common_blob.expr_evals, &preflights)
        });
        let if_blob = InteractionsFoldingBlob::new(
            &child_vk.inner,
            &common_blob.expr_evals,
            &common_blob.eq_3b_blob,
            &preflights,
        );
        let empty_cf = MultiProofVecVec::new();
        let expr_claim_blob = generate_expression_claim_blob_for_mode(
            cf_blob
                .as_ref()
                .map_or(&empty_cf, |blob| &blob.folded_claims),
            &if_blob.folded_claims,
            equation_mode.includes_air(),
        );
        Self {
            common_blob,
            cf_blob,
            if_blob: Some(if_blob),
            expr_claim_blob,
        }
    }
}

impl<SC: StarkProtocolConfig<F = F>> TraceGenModule<GlobalCtxCpu, CpuBackend<SC>>
    for BatchConstraintModule
{
    type ModuleSpecificCtx<'a> = (
        Option<&'a CachedTraceRecord>,
        Arc<PowerCheckerCpuTraceGenerator<2, POW_CHECKER_HEIGHT>>,
    );

    /// **Note**: This generates all common main traces but leaves the cached trace for
    /// `SymbolicExpressionAir` unset. The cached trace must be loaded **after** calling this
    /// function.
    #[tracing::instrument(skip_all)]
    fn generate_proving_ctxs(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        preflights: &[Preflight],
        ctx: &Self::ModuleSpecificCtx<'_>,
        required_heights: Option<&[usize]>,
    ) -> Option<Vec<AirProvingContext<CpuBackend<SC>>>> {
        if preflights
            .iter()
            .any(|preflight| preflight.batch_constraint.equation_mode != self.equation_mode)
        {
            return None;
        }
        let blob = BatchConstraintBlobCpu::new(child_vk, proofs, preflights, self.equation_mode);
        let pow_checker = ctx.1.clone();
        let ctx = (
            StandardTracegenCtx {
                vk: child_vk,
                proofs: &proofs.iter().collect_vec(),
                preflights: &preflights.iter().collect_vec(),
            },
            blob,
            ctx.0,
        );

        let chips = self.tracegen_chips(pow_checker);

        let span = tracing::Span::current();
        chips
            .par_iter()
            .map(|chip| {
                let _guard = span.enter();
                chip.generate_proving_ctx(
                    &ctx,
                    required_heights.map(|heights| heights[chip.index()]),
                )
            })
            .collect::<Vec<_>>()
            .into_iter()
            .collect()
    }
}

impl BatchConstraintModule {
    pub fn cached_trace_record(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> CachedTraceRecord {
        expr_eval::build_cached_trace_record(child_vk, self.has_cached)
    }

    /// Generates and then commits to the cache trace for `SymbolicExpressionAir`. Returns the
    /// committed PCS data.
    /// The cached-main trace committing the child constraint DAG, for
    /// callers that commit on their own device.
    pub fn child_vk_cached_trace(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> RowMajorMatrix<F> {
        expr_eval::generate_symbolic_expr_cached_trace(&self.cached_trace_record(child_vk))
    }

    pub fn commit_child_vk<E, SC: StarkProtocolConfig<F = F>>(
        &self,
        engine: &E,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> CommittedTraceData<CpuBackend<SC>>
    where
        E: StarkEngine<SC = SC, PB = CpuBackend<SC>>,
    {
        let cached_trace =
            expr_eval::generate_symbolic_expr_cached_trace(&self.cached_trace_record(child_vk));
        let (commitment, data) = engine.device().commit(&[&cached_trace]).unwrap();
        CommittedTraceData {
            commitment,
            data: Arc::new(data),
            trace: cached_trace,
        }
    }
}

// NOTE: ordering of enum must match AIR ordering
#[derive(strum_macros::Display)]
enum BatchConstraintModuleChip {
    SymbolicExpression {
        max_num_proofs: usize,
        has_cached: bool,
    },
    FractionsFolder,
    SumcheckUni,
    SumcheckLin,
    EqNs,
    Eq3b,
    EqSharpUni,
    EqSharpUniReceiver,
    EqUni,
    ExpressionClaim {
        pow_checker: Arc<PowerCheckerCpuTraceGenerator<2, POW_CHECKER_HEIGHT>>,
        includes_air: bool,
    },
    InteractionsFolding,
    ConstraintsFolding,
    PartialEndpoint,
    EqNeg {
        uses_stacking_point: bool,
    },
}

impl BatchConstraintModule {
    /// Canonical witness-generator order for both CPU and CUDA. Keeping the
    /// mode split here prevents the CUDA path from silently reverting a
    /// `LogUpOnly` verifier to the standard AIR-plus-LogUp equation.
    fn tracegen_chips(
        &self,
        pow_checker: Arc<PowerCheckerCpuTraceGenerator<2, POW_CHECKER_HEIGHT>>,
    ) -> Vec<BatchConstraintModuleChip> {
        let mut chips = vec![
            BatchConstraintModuleChip::SymbolicExpression {
                max_num_proofs: self.max_num_proofs,
                has_cached: self.has_cached,
            },
            BatchConstraintModuleChip::FractionsFolder,
            BatchConstraintModuleChip::SumcheckUni,
            BatchConstraintModuleChip::SumcheckLin,
            BatchConstraintModuleChip::EqNs,
            BatchConstraintModuleChip::Eq3b,
            BatchConstraintModuleChip::EqSharpUni,
            BatchConstraintModuleChip::EqSharpUniReceiver,
            BatchConstraintModuleChip::EqUni,
            BatchConstraintModuleChip::ExpressionClaim {
                pow_checker,
                includes_air: self.equation_mode.includes_air(),
            },
            BatchConstraintModuleChip::InteractionsFolding,
        ];
        chips.push(if self.equation_mode.includes_air() {
            BatchConstraintModuleChip::ConstraintsFolding
        } else {
            BatchConstraintModuleChip::PartialEndpoint
        });
        chips.push(BatchConstraintModuleChip::EqNeg {
            uses_stacking_point: self.equation_mode.includes_air() && !self.deferred_stacking,
        });
        chips
    }
}

impl BatchConstraintModuleChip {
    fn index(&self) -> usize {
        match self {
            Self::SymbolicExpression { .. } => 0,
            Self::FractionsFolder => 1,
            Self::SumcheckUni => 2,
            Self::SumcheckLin => 3,
            Self::EqNs => 4,
            Self::Eq3b => 5,
            Self::EqSharpUni => 6,
            Self::EqSharpUniReceiver => 7,
            Self::EqUni => 8,
            Self::ExpressionClaim { .. } => 9,
            Self::InteractionsFolding => 10,
            Self::ConstraintsFolding | Self::PartialEndpoint => 11,
            Self::EqNeg { .. } => 12,
        }
    }

    #[cfg(feature = "cuda")]
    fn has_cuda_tracegen(&self) -> bool {
        matches!(
            self,
            Self::SymbolicExpression { .. }
                | Self::Eq3b
                | Self::InteractionsFolding
                | Self::ConstraintsFolding
        )
    }
}

impl RowMajorChip<F> for BatchConstraintModuleChip {
    type Ctx<'a> = (
        StandardTracegenCtx<'a>,
        BatchConstraintBlobCpu,
        Option<&'a CachedTraceRecord>,
    );

    #[tracing::instrument(
        name = "wrapper.generate_trace",
        level = "trace",
        skip_all,
        fields(air = %self)
    )]
    fn generate_trace(
        &self,
        ctx: &Self::Ctx<'_>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        use BatchConstraintModuleChip::*;
        let child_vk = ctx.0.vk;
        let proofs = ctx.0.proofs;
        let preflights = ctx.0.preflights;
        let blob = &ctx.1;
        let cached_trace_record = ctx.2;
        match self {
            FractionsFolder => {
                FractionsFolderTraceGenerator.generate_trace(&ctx.0, required_height)
            }
            SumcheckUni => UnivariateSumcheckTraceGenerator.generate_trace(&ctx.0, required_height),
            SumcheckLin => {
                MultilinearSumcheckTraceGenerator.generate_trace(&ctx.0, required_height)
            }
            EqNs => eq_airs::EqNsTraceGenerator.generate_trace(
                &(child_vk, preflights, &blob.common_blob.selector_counts),
                required_height,
            ),
            Eq3b => eq_airs::Eq3bTraceGenerator.generate_trace(
                &(child_vk, &blob.common_blob.eq_3b_blob, preflights),
                required_height,
            ),
            EqSharpUni => eq_airs::EqSharpUniTraceGenerator.generate_trace(
                &(child_vk, &blob.common_blob.eq_sharp_uni_blob, preflights),
                required_height,
            ),
            EqSharpUniReceiver => eq_airs::EqSharpUniReceiverTraceGenerator.generate_trace(
                &(child_vk, &blob.common_blob.eq_sharp_uni_blob, preflights),
                required_height,
            ),
            EqUni => eq_airs::EqUniTraceGenerator.generate_trace(&ctx.0, required_height),
            SymbolicExpression {
                max_num_proofs,
                has_cached,
            } => expr_eval::SymbolicExpressionTraceGenerator {
                max_num_proofs: *max_num_proofs,
                has_cached: *has_cached,
            }
            .generate_trace(
                &expr_eval::SymbolicExpressionCtx {
                    vk: child_vk,
                    preflights,
                    expr_evals: &blob.common_blob.expr_evals,
                    cached_trace_record: &cached_trace_record,
                },
                required_height,
            ),
            ExpressionClaim {
                pow_checker,
                includes_air,
            } => ExpressionClaimTraceGenerator.generate_trace(
                &expression_claim::ExpressionClaimCtx {
                    blob: &blob.expr_claim_blob,
                    proofs,
                    preflights,
                    pow_checker: pow_checker.as_ref(),
                    includes_air: *includes_air,
                },
                required_height,
            ),
            InteractionsFolding => expr_eval::InteractionsFoldingTraceGenerator
                .generate_trace(&(child_vk, blob, preflights), required_height),
            ConstraintsFolding => expr_eval::ConstraintsFoldingTraceGenerator.generate_trace(
                &(
                    blob.cf_blob
                        .as_ref()
                        .expect("constraint folding is absent in LogUpOnly mode"),
                    preflights,
                ),
                required_height,
            ),
            PartialEndpoint => PartialBatchConstraintEndpointTraceGenerator
                .generate_trace(&preflights, required_height),
            EqNeg {
                uses_stacking_point,
            } => EqNegTraceGenerator {
                uses_stacking_point: *uses_stacking_point,
            }
            .generate_trace(
                &(child_vk, preflights, &blob.common_blob.selector_counts),
                required_height,
            ),
        }
    }
}

#[cfg(feature = "cuda")]
pub mod cuda_tracegen {
    use openvm_cuda_backend::{data_transporter::transport_matrix_h2d_row, GpuBackend};
    use openvm_cuda_common::stream::GpuDeviceCtx;
    use openvm_poseidon2_air::POSEIDON2_WIDTH;

    use super::*;
    use crate::{
        batch_constraint::expr_eval::{
            build_cached_trace_record, constraints_folding::cuda::ConstraintsFoldingBlobGpu,
            interactions_folding::cuda::InteractionsFoldingBlobGpu,
        },
        cuda::{preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu, GlobalCtxGpu},
        tracegen::cuda::StandardTracegenGpuCtx,
    };

    /// One context in the canonical `LogUpOnlyPartialVerifier` AIR order.
    ///
    /// The resumed/extended transcript and its rebased proof-shape adapter are
    /// still generated by the CPU oracle; all other variants are generated
    /// directly on the CUDA device. Keeping the origin explicit prevents
    /// callers from accidentally treating a host matrix as a resident CUDA
    /// trace.
    pub enum LogUpOnlyCudaProvingContext {
        Device(AirProvingContext<GpuBackend>),
        CpuTranscript(AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>),
        CpuRebasedProofShape(AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>),
    }

    /// CUDA partial-verifier packet with the physical Poseidon owner omitted.
    /// `contexts` is in the exact order returned by
    /// `LogUpOnlyPartialVerifier::airs_without_poseidon`.
    pub struct LogUpOnlySharedPoseidonCudaContexts {
        pub contexts: Vec<LogUpOnlyCudaProvingContext>,
        pub poseidon2_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        pub poseidon2_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    }

    impl LogUpOnlySharedPoseidonCudaContexts {
        #[must_use]
        pub fn grouped_input(self) -> (Vec<[F; POSEIDON2_WIDTH]>, Vec<[F; POSEIDON2_WIDTH]>) {
            (
                self.poseidon2_permutation_inputs,
                self.poseidon2_compression_inputs,
            )
        }
    }

    impl ModuleChip<GpuBackend> for BatchConstraintModuleChip {
        type Ctx<'a> = (
            StandardTracegenGpuCtx<'a>,
            &'a BatchConstraintBlobGpu,
            Option<&'a CachedTraceRecord>,
        );

        fn generate_proving_ctx(
            &self,
            ctx: &Self::Ctx<'_>,
            required_height: Option<usize>,
        ) -> Option<AirProvingContext<GpuBackend>> {
            use BatchConstraintModuleChip::*;
            let child_vk = ctx.0.vk;
            let proofs = ctx.0.proofs;
            let preflights = ctx.0.preflights;
            let blob = ctx.1;
            let cached_trace_record = ctx.2;
            match self {
                Eq3b => eq_airs::Eq3bTraceGenerator.generate_proving_ctx(
                    &(
                        &child_vk.cpu,
                        &blob.common_blob.eq_3b_blob,
                        preflights,
                        ctx.0.device_ctx,
                    ),
                    required_height,
                ),
                SymbolicExpression {
                    max_num_proofs,
                    has_cached,
                } => expr_eval::SymbolicExpressionTraceGenerator {
                    max_num_proofs: *max_num_proofs,
                    has_cached: *has_cached,
                }
                .generate_proving_ctx(
                    &expr_eval::symbolic_expression::cuda::SymbolicExpressionGpuCtx {
                        vk: &child_vk.cpu,
                        proofs,
                        preflights,
                        expr_evals: &blob.common_blob.expr_evals,
                        cached_trace_record: &cached_trace_record,
                        device_ctx: ctx.0.device_ctx,
                    },
                    required_height,
                ),
                InteractionsFolding => expr_eval::InteractionsFoldingTraceGenerator
                    .generate_proving_ctx(
                        &(child_vk, preflights, &blob.if_blob, ctx.0.device_ctx),
                        required_height,
                    ),
                ConstraintsFolding => expr_eval::ConstraintsFoldingTraceGenerator
                    .generate_proving_ctx(
                        &(
                            child_vk,
                            preflights,
                            blob.cf_blob
                                .as_ref()
                                .expect("constraint folding is absent in LogUpOnly mode"),
                            ctx.0.device_ctx,
                        ),
                        required_height,
                    ),
                _ => unreachable!(),
            }
        }
    }

    pub(in crate::batch_constraint) struct BatchConstraintBlobGpu {
        pub common_blob: BatchConstraintBlob,
        pub cf_blob: Option<ConstraintsFoldingBlobGpu>,
        pub if_blob: InteractionsFoldingBlobGpu,
        pub expr_claim_blob: ExpressionClaimBlob,
    }

    impl BatchConstraintBlobGpu {
        #[tracing::instrument(name = "generate_blob", skip_all)]
        pub fn new(
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            device_ctx: &GpuDeviceCtx,
            equation_mode: VerifierEquationMode,
        ) -> Self {
            let cpu_proofs = proofs.iter().map(|p| &p.cpu).collect_vec();
            let cpu_preflights = preflights.iter().map(|p| &p.cpu).collect_vec();
            let common_blob = BatchConstraintBlob::new(&child_vk.cpu, &cpu_proofs, &cpu_preflights);
            let cf_blob = equation_mode.includes_air().then(|| {
                ConstraintsFoldingBlobGpu::new(
                    child_vk,
                    &common_blob.expr_evals,
                    preflights,
                    device_ctx,
                )
            });
            let if_blob = InteractionsFoldingBlobGpu::new(
                child_vk,
                &common_blob.expr_evals,
                &common_blob.eq_3b_blob,
                preflights,
                device_ctx,
            );
            let empty_cf = MultiProofVecVec::new();
            let expr_claim_blob = generate_expression_claim_blob_for_mode(
                cf_blob
                    .as_ref()
                    .map_or(&empty_cf, |blob| &blob.folded_claims),
                &if_blob.folded_claims,
                equation_mode.includes_air(),
            );
            Self {
                common_blob,
                cf_blob,
                if_blob,
                expr_claim_blob,
            }
        }
    }

    impl TraceGenModule<GlobalCtxGpu, GpuBackend> for BatchConstraintModule {
        type ModuleSpecificCtx<'a> = (
            Option<&'a CachedTraceRecord>,
            Arc<PowerCheckerCpuTraceGenerator<2, POW_CHECKER_HEIGHT>>,
            &'a openvm_cuda_common::stream::GpuDeviceCtx,
        );

        #[tracing::instrument(skip_all)]
        fn generate_proving_ctxs(
            &self,
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            module_ctx: &Self::ModuleSpecificCtx<'_>,
            required_heights: Option<&[usize]>,
        ) -> Option<Vec<AirProvingContext<GpuBackend>>> {
            let cached_trace_record = module_ctx.0;
            let pow_checker = module_ctx.1.clone();
            let device_ctx = module_ctx.2;
            if preflights
                .iter()
                .any(|preflight| preflight.cpu.batch_constraint.equation_mode != self.equation_mode)
            {
                return None;
            }
            let blob = BatchConstraintBlobGpu::new(
                child_vk,
                proofs,
                preflights,
                device_ctx,
                self.equation_mode,
            );
            let ctx = (
                StandardTracegenGpuCtx {
                    vk: child_vk,
                    proofs,
                    preflights,
                    device_ctx,
                },
                &blob,
                cached_trace_record,
            );

            let (gpu_chips, cpu_chips): (Vec<_>, Vec<_>) = self
                .tracegen_chips(pow_checker)
                .into_iter()
                .partition(BatchConstraintModuleChip::has_cuda_tracegen);
            let span = tracing::Span::current();
            // NOTE: do NOT use par_iter since that will lead to kernels on cuda streams != default
            // stream, whereas previous H2D transfer was on default stream.
            let indexed_gpu_traces = gpu_chips
                .iter()
                .map(|chip| {
                    // This span is not very useful because the kernel does not synchronize on host:
                    let _guard = span.enter();
                    (
                        chip.index(),
                        chip.generate_proving_ctx(
                            &ctx,
                            required_heights.map(|heights| heights[chip.index()]),
                        ),
                    )
                })
                .collect_vec();

            let blob = BatchConstraintBlobCpu {
                common_blob: blob.common_blob,
                cf_blob: None,
                if_blob: None,
                expr_claim_blob: blob.expr_claim_blob,
            };
            let cpu_proofs = proofs.iter().map(|p| &p.cpu).collect_vec();
            let cpu_preflights = preflights.iter().map(|p| &p.cpu).collect_vec();
            let cpu_ctx = (
                StandardTracegenCtx {
                    vk: &child_vk.cpu,
                    proofs: &cpu_proofs,
                    preflights: &cpu_preflights,
                },
                blob,
                cached_trace_record,
            );

            // Phase 1: CPU trace generation in parallel
            let indexed_cpu_rm_traces = cpu_chips
                .par_iter()
                .map(|chip| {
                    let _guard = span.enter();
                    (
                        chip.index(),
                        chip.generate_trace(
                            &cpu_ctx,
                            required_heights.map(|heights| heights[chip.index()]),
                        ),
                    )
                })
                .collect::<Vec<_>>();

            // Phase 2: H2D transfer serially on main thread
            let indexed_cpu_gpu_traces = indexed_cpu_rm_traces
                .into_iter()
                .map(|(idx, trace)| {
                    (
                        idx,
                        trace.map(|m| {
                            AirProvingContext::simple_no_pis(
                                transport_matrix_h2d_row(&m, device_ctx).unwrap(),
                            )
                        }),
                    )
                })
                .collect::<Vec<_>>();

            indexed_gpu_traces
                .into_iter()
                .chain(indexed_cpu_gpu_traces)
                .sorted_by(|a, b| a.0.cmp(&b.0))
                .map(|(_index, ctx)| ctx)
                .collect()
        }
    }

    impl BatchConstraintModule {
        /// Generates and then commits to the cache trace for `SymbolicExpressionAir`. Returns the
        /// committed PCS data. The engine may use any GPU backend (e.g. BabyBear Poseidon2 or
        /// BabyBear Bn254 Poseidon2) — only its `device().commit()` method is called.
        pub fn commit_child_vk_gpu<E>(
            &self,
            engine: &E,
            child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
            device_ctx: &GpuDeviceCtx,
        ) -> CommittedTraceData<E::PB>
        where
            E: StarkEngine,
            E::PB: openvm_stark_backend::prover::ProverBackend<
                Val = F,
                Matrix = openvm_cuda_backend::base::DeviceMatrix<F>,
            >,
            E::PD: TraceCommitter<E::PB>,
        {
            let cached_trace_record = build_cached_trace_record(child_vk, self.has_cached);
            let cached_trace = expr_eval::generate_symbolic_expr_cached_trace(&cached_trace_record);
            let d_cached_trace = transport_matrix_h2d_row(&cached_trace, device_ctx).unwrap();
            let (commitment, data) = engine.device().commit(&[&d_cached_trace]).unwrap();
            CommittedTraceData {
                commitment,
                trace: d_cached_trace,
                data: Arc::new(data),
            }
        }
    }
}

//! Setup-fixed aggregate circuit for the bounded fixed-multi-AIR terminal
//! verifier.
//!
//! This module owns the AIR order and the private terminal bus namespace.  It
//! deliberately excludes the shared Poseidon/range/power tables: terminal
//! trace generation contributes requests to the parent recursive verifier's
//! already-keyed shared tables.

use std::sync::Arc;

use openvm_stark_backend::{
    hasher::MerkleHasher,
    interaction::BusIndex,
    native_warp::FixedMultiAirPesatIndex,
    warp_accum::{
        rs_whir::{
            COEFFICIENT_SUBGROUP_RS_LAYOUT_VERSION, COEFFICIENT_TWO_COSET_GRS_INTERLEAVED_ORDERING,
            COEFFICIENT_TWO_COSET_GRS_LAYOUT_VERSION,
        },
        TerminalDescriptor, WhirInitialRsLayout,
    },
    AirRef, AnyAir, StarkProtocolConfig, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, Digest, D_EF, F};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64, TwoAdicField};

use super::*;
use crate::{
    bus::TranscriptBus,
    native_warp::{
        NativeLeafHashAir, NativeLeafValueBus, NativeMerkleMultiproofAir, NativeMerkleNodeBus,
        NativeMerkleRootBus, NativeOpeningLeafBus, NativeTerminalAccumulatorRootBus,
        NativeTerminalAccumulatorValueBus, NativeTerminalRsAdjointClaimBus,
        NativeTerminalRsAdjointQAir, NativeTerminalRsAdjointQBus, NativeTerminalRsAdjointRoundBus,
        NativeTerminalRsAdjointSelectorAir, NativeTerminalRsAdjointSumcheckAir,
        NativeTerminalRsAdjointValueBus, NativeTerminalRsAdjointYAir, NativeTerminalRsAdjointYBus,
        NativeTerminalWhirActualWeightBus, NativeTerminalWhirAlphaBus,
        NativeTerminalWhirExpectedWeightAir, NativeTerminalWhirFinalCheckAir,
        NativeTerminalWhirFinalClaimBus, NativeTerminalWhirFinalContextBus,
        NativeTerminalWhirFinalPolyBus, NativeTerminalWhirFinalTableAir,
        NativeTerminalWhirFinalWeightBus, NativeTerminalWhirFoldingAir,
        NativeTerminalWhirFoldingBus, NativeTerminalWhirLinearizerWeightBus,
        NativeTerminalWhirMobiusAir, NativeTerminalWhirOpenedAir, NativeTerminalWhirPointAir,
        NativeTerminalWhirPointBus, NativeTerminalWhirQueryAir, NativeTerminalWhirQueryBus,
        NativeTerminalWhirRoundAir, NativeTerminalWhirRoundBus, NativeTerminalWhirStatementBus,
        NativeTerminalWhirSumcheckAir, NativeTerminalWhirVerifyQueriesBus,
        NativeTerminalWhirWeightTermAir, NativeTerminalWhirWeightTermBus,
        NativeTerminalWhirWeightTermResultBus, NativeWarpTranscriptModule,
        NATIVE_TERMINAL_MAX_ADJOINT_DEGREE, NATIVE_TERMINAL_MAX_FINAL_LOG_LEN,
    },
    primitives::{
        bus::{ExpBitsLenBus, RightShiftBus},
        exp_bits_len::ExpBitsLenAir,
    },
    system::{BusIndexManager, BusInventory},
    transcript::Poseidon2BusOwner,
};

const TERMINAL_CIRCUIT_DIGEST_TAG: u64 = 0x4e57_4d41_564b_0004;
// v5 consumes compact-layout regional/padding claims as already globally
// weighted contributions. Earlier schedules applied obsolete dyadic-prefix
// factors a second time. v6 binds the initial RS layout and derives the
// structured-selector dimension independently of WHIR's round fold arity.
const TERMINAL_CIRCUIT_SCHEDULE_VERSION: u32 = 6;

// These are protocol tags from `warp_accum::terminal_descriptor`. They are
// intentionally checked here instead of inferring a code layout only from
// dimensions: ordinary scalar/vector layouts and the coefficient-native code
// can otherwise share ambiguous `(message, codeword)` lengths.
const TERMINAL_DESCRIPTOR_VERSION: u64 = 1;
const TERMINAL_LAYOUT_WHIR_INITIAL_RS: u64 = 1;
const TERMINAL_BASIS_WHIR_BOOLEAN: u64 = 1;
const TERMINAL_DOMAIN_TWO_ADIC_RS: u64 = 1;
const TERMINAL_DOMAIN_COEFFICIENT_TWO_COSET_GRS: u64 = 2;
const TERMINAL_DOMAIN_COEFFICIENT_SUBGROUP_RS: u64 = 3;
const TERMINAL_MMCS_BACKEND_MERKLE: u64 = 1;
const TERMINAL_PRODUCTION_WHIR_FOLD_LOG: usize = 4;
const TERMINAL_PRODUCTION_WHIR_FOLD_ARITY: u64 = 16;

/// Complete setup-fixed vector-alphabet profile for the fixed-multi-AIR terminal and
/// its same-root e-linear WHIR verifier.
#[derive(Clone, Debug)]
pub struct FixedMultiAirTerminalCircuitProfile {
    pub relation: Arc<FixedMultiAirPesatIndex<F, Digest>>,
    pub relation_digest: Digest,
    pub metadata_words: Vec<u64>,
    pub alpha_len: usize,
    pub beta_len: usize,
    /// Logarithm of WHIR's per-round fold arity. Production uses `k = 4`,
    /// i.e. a 16-way fold; this is not the structured selector dimension.
    pub k: usize,
    /// Authenticated initial-code layout reconstructed from the descriptor.
    pub initial_rs_layout: WhirInitialRsLayout,
    /// Code-level initial vector-alphabet folding factor.
    pub initial_folding_factor: usize,
    /// Layout-derived selector dimension used by both the structured
    /// linearizer AIR and its trace generator.
    pub selector_folding_factor: usize,
    pub initial_log_domain_size: usize,
    pub log_message_len: usize,
    pub final_poly_len: usize,
    pub query_pow_bits: usize,
    pub folding_pow_bits: usize,
    pub num_queries_per_round: Vec<usize>,
    pub point_lookup_counts: Vec<usize>,
    pub weight_term_count: usize,
    pub rs_adjoint_round_count: usize,
    pub rs_adjoint_degree: usize,
    pub decomposition: Vec<FixedMultiAirDecompositionComponentPlan>,
    pub endpoints: Vec<FixedMultiAirEndpointPlan>,
    pub linearizer_components: Vec<FixedMultiAirLinearizerRawComponentPlan>,
}

impl FixedMultiAirTerminalCircuitProfile {
    pub fn new(
        relation: Arc<FixedMultiAirPesatIndex<F, Digest>>,
        descriptor: &TerminalDescriptor<Digest>,
    ) -> Result<Self, &'static str> {
        let k = usize::try_from(descriptor.whir_k).map_err(|_| "terminal WHIR k")?;
        let initial_log_domain_size = usize::try_from(descriptor.log_codeword_len)
            .map_err(|_| "terminal codeword dimension")?;
        let log_message_len = usize::try_from(descriptor.log_message_len)
            .map_err(|_| "terminal message dimension")?;
        let initial_folding_factor = usize::try_from(descriptor.initial_folding_factor)
            .map_err(|_| "terminal initial folding factor")?;
        let initial_rs_layout = validate_terminal_descriptor_layout(
            descriptor,
            log_message_len,
            initial_log_domain_size,
            k,
        )?;
        let selector_folding_factor = structured_selector_folding_factor(
            initial_rs_layout,
            log_message_len,
            initial_folding_factor,
        )
        .map_err(|_| "terminal structured selector")?;
        let alpha_len = initial_log_domain_size;
        let beta_len = relation.pesat_shape().beta_len();
        let (rs_adjoint_round_count, rs_adjoint_degree) = match initial_rs_layout {
            WhirInitialRsLayout::OrdinarySubgroup => (
                alpha_len
                    .checked_sub(initial_folding_factor)
                    .ok_or("terminal adjoint codeword dimension")?,
                log_message_len
                    .checked_sub(initial_folding_factor)
                    .ok_or("terminal adjoint message dimension")?
                    .checked_add(1)
                    .ok_or("terminal adjoint degree")?,
            ),
            WhirInitialRsLayout::CoefficientTwoCosetGrs => (
                alpha_len,
                log_message_len
                    .checked_add(1)
                    .ok_or("terminal two-coset adjoint degree")?,
            ),
            WhirInitialRsLayout::CoefficientSubgroup => (
                alpha_len,
                log_message_len
                    .checked_add(1)
                    .ok_or("terminal coefficient-subgroup adjoint degree")?,
            ),
        };
        let num_queries_per_round = descriptor
            .whir_round_queries
            .iter()
            .map(|&value| usize::try_from(value).map_err(|_| "terminal query count"))
            .collect::<Result<Vec<_>, _>>()?;
        let round_count = num_queries_per_round.len();
        let folded_prefix = k.checked_mul(round_count).ok_or("terminal folded prefix")?;
        let final_log_len = log_message_len
            .checked_sub(folded_prefix)
            .ok_or("terminal final dimension")?;
        let final_poly_len = 1usize
            .checked_shl(u32::try_from(final_log_len).map_err(|_| "terminal final dimension")?)
            .ok_or("terminal final polynomial")?;
        let code_class = relation.description().code_class;
        if descriptor.extension_degree != D_EF as u64
            || descriptor.log_inv_rate
                != descriptor
                    .log_codeword_len
                    .checked_sub(descriptor.log_message_len)
                    .ok_or("terminal inverse rate")?
            || relation.pesat_shape().log_witness != log_message_len
            || u64::from(code_class.log_message_len) != descriptor.log_message_len
            || u64::from(code_class.log_codeword_len) != descriptor.log_codeword_len
            || u64::from(code_class.log_blowup) != descriptor.log_inv_rate
            || u64::from(code_class.initial_folding_factor) != descriptor.initial_folding_factor
            || u64::from(code_class.rows_per_query) != descriptor.rows_per_query
            || k == 0
            || round_count == 0
            || final_log_len == 0
            || final_log_len > NATIVE_TERMINAL_MAX_FINAL_LOG_LEN
            || rs_adjoint_degree > NATIVE_TERMINAL_MAX_ADJOINT_DEGREE
            || initial_log_domain_size.checked_sub(k).is_none()
            || initial_log_domain_size - k > F::TWO_ADICITY
            || num_queries_per_round.contains(&0)
            || relation.description().exact_max_degree.saturating_sub(1) as usize
                > PADDING_MAX_SCALE_POWER
        {
            return Err("terminal vector profile");
        }
        let weight_term_count = num_queries_per_round
            .iter()
            .try_fold(round_count - 1, |total, &queries| {
                total.checked_add(queries)
            })
            .ok_or("terminal weight-term count")?;
        let point_lookup_counts = terminal_point_lookup_counts(
            k,
            &num_queries_per_round,
            final_poly_len,
            log_message_len,
        )
        .ok_or("terminal point lookup profile")?;
        let decomposition = FixedMultiAirDecompositionComponentPlan::from_relation(&relation)
            .map_err(|_| "terminal decomposition profile")?;
        let endpoints = (0..relation.region_count())
            .map(|region| FixedMultiAirEndpointPlan::from_relation(&relation, region))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|_| "terminal endpoint profile")?;
        let linearizer_components =
            FixedMultiAirLinearizerRawComponentPlan::from_relation(&relation)
                .map_err(|_| "terminal linearizer profile")?;
        Ok(Self {
            relation_digest: relation.description().relation_digest,
            relation,
            metadata_words: descriptor.metadata_words().collect(),
            alpha_len,
            beta_len,
            k,
            initial_rs_layout,
            initial_folding_factor,
            selector_folding_factor,
            initial_log_domain_size,
            log_message_len,
            final_poly_len,
            query_pow_bits: usize::try_from(descriptor.whir_query_phase_pow_bits)
                .map_err(|_| "terminal query PoW")?,
            folding_pow_bits: usize::try_from(descriptor.whir_folding_pow_bits)
                .map_err(|_| "terminal folding PoW")?,
            num_queries_per_round,
            point_lookup_counts,
            weight_term_count,
            rs_adjoint_round_count,
            rs_adjoint_degree,
            decomposition,
            endpoints,
            linearizer_components,
        })
    }

    #[must_use]
    pub fn round_count(&self) -> usize {
        self.num_queries_per_round.len()
    }

    #[must_use]
    pub fn final_log_len(&self) -> usize {
        self.final_poly_len.ilog2() as usize
    }

    #[must_use]
    pub fn claim_count(&self) -> usize {
        self.relation.region_count()
            + usize::from(self.relation.description().padding_constraint_count != 0)
    }

    #[must_use]
    pub const fn whir_fold_arity(&self) -> usize {
        1usize << self.k
    }

    pub(super) fn digest_words(&self, first_bus_idx: BusIndex, next_bus_idx: BusIndex) -> Vec<F> {
        let mut words = vec![
            F::from_u64(TERMINAL_CIRCUIT_DIGEST_TAG),
            F::from_u32(TERMINAL_CIRCUIT_SCHEDULE_VERSION),
            F::from_usize(first_bus_idx as usize),
            F::from_usize(next_bus_idx as usize),
        ];
        words.extend(self.relation_digest);
        words.extend(
            [
                self.alpha_len,
                self.beta_len,
                self.k,
                match self.initial_rs_layout {
                    WhirInitialRsLayout::OrdinarySubgroup => 0,
                    WhirInitialRsLayout::CoefficientTwoCosetGrs => 1,
                    WhirInitialRsLayout::CoefficientSubgroup => 2,
                },
                self.initial_folding_factor,
                self.selector_folding_factor,
                self.initial_log_domain_size,
                self.log_message_len,
                self.final_poly_len,
                self.query_pow_bits,
                self.folding_pow_bits,
                self.weight_term_count,
                self.rs_adjoint_round_count,
                self.rs_adjoint_degree,
                self.claim_count(),
                self.linearizer_components.len(),
            ]
            .map(F::from_usize),
        );
        words.push(F::from_usize(self.metadata_words.len()));
        words.extend(self.metadata_words.iter().copied().map(F::from_u64));
        words.push(F::from_usize(self.num_queries_per_round.len()));
        words.extend(
            self.num_queries_per_round
                .iter()
                .copied()
                .map(F::from_usize),
        );
        words.push(F::from_usize(self.point_lookup_counts.len()));
        words.extend(self.point_lookup_counts.iter().copied().map(F::from_usize));
        words.push(F::from_usize(self.endpoints.len()));
        for endpoint in &self.endpoints {
            words.extend(
                [
                    endpoint.region,
                    endpoint.log_height,
                    endpoint.one_coordinate,
                    endpoint.dynamic.len(),
                    endpoint.fixed.len(),
                    endpoint.node_sources.len(),
                    endpoint.fanout.len(),
                ]
                .map(F::from_usize),
            );
        }
        // Stable AIR-order tags. Changing, inserting, or deleting an AIR must
        // bump the schedule version and this list.
        words.extend((0u32..=39).map(F::from_u32));
        words
    }
}

fn validate_terminal_descriptor_layout(
    descriptor: &TerminalDescriptor<Digest>,
    log_message_len: usize,
    initial_log_domain_size: usize,
    whir_k: usize,
) -> Result<WhirInitialRsLayout, &'static str> {
    if descriptor.version != TERMINAL_DESCRIPTOR_VERSION
        || descriptor.layout_tag != TERMINAL_LAYOUT_WHIR_INITIAL_RS
        || descriptor.basis_tag != TERMINAL_BASIS_WHIR_BOOLEAN
        || descriptor.mmcs_tag != TERMINAL_MMCS_BACKEND_MERKLE
    {
        return Err("terminal descriptor tags");
    }
    let fold_arity = 1u64
        .checked_shl(u32::try_from(whir_k).map_err(|_| "terminal WHIR folding arity")?)
        .ok_or("terminal WHIR folding arity")?;
    let non_legacy_words = [
        descriptor.code_layout_version,
        descriptor.field_modulus,
        descriptor.subgroup_generator,
        descriptor.coset_shift,
        descriptor.coordinate_ordering,
    ];
    match descriptor.domain_tag {
        TERMINAL_DOMAIN_TWO_ADIC_RS => {
            // Preserve the legacy vector-alphabet profile exactly. In
            // particular, do not silently admit scalar ordinary-RS geometry.
            if descriptor.initial_folding_factor != descriptor.whir_k
                || descriptor.initial_oracle_width != fold_arity
                || descriptor.rows_per_query != 1
                || non_legacy_words.into_iter().any(|word| word != 0)
            {
                return Err("terminal ordinary-subgroup layout");
            }
            Ok(WhirInitialRsLayout::OrdinarySubgroup)
        }
        TERMINAL_DOMAIN_COEFFICIENT_TWO_COSET_GRS => {
            if log_message_len > F::TWO_ADICITY
                || log_message_len.checked_add(1) != Some(initial_log_domain_size)
                || whir_k != TERMINAL_PRODUCTION_WHIR_FOLD_LOG
                || fold_arity != TERMINAL_PRODUCTION_WHIR_FOLD_ARITY
                || descriptor.log_inv_rate != 1
                || descriptor.initial_folding_factor != 0
                || descriptor.initial_oracle_width != 1
                || descriptor.rows_per_query != TERMINAL_PRODUCTION_WHIR_FOLD_ARITY
                || descriptor.code_layout_version
                    != u64::from(COEFFICIENT_TWO_COSET_GRS_LAYOUT_VERSION)
                || descriptor.field_modulus != F::ORDER_U64
                || descriptor.subgroup_generator
                    != F::two_adic_generator(log_message_len).as_canonical_u64()
                || descriptor.coset_shift != F::GENERATOR.as_canonical_u64()
                || descriptor.coordinate_ordering
                    != u64::from(COEFFICIENT_TWO_COSET_GRS_INTERLEAVED_ORDERING)
                || F::GENERATOR.exp_power_of_2(log_message_len) == F::ONE
            {
                return Err("terminal coefficient two-coset layout");
            }
            Ok(WhirInitialRsLayout::CoefficientTwoCosetGrs)
        }
        TERMINAL_DOMAIN_COEFFICIENT_SUBGROUP_RS => {
            if initial_log_domain_size > F::TWO_ADICITY
                || log_message_len >= initial_log_domain_size
                || whir_k != TERMINAL_PRODUCTION_WHIR_FOLD_LOG
                || fold_arity != TERMINAL_PRODUCTION_WHIR_FOLD_ARITY
                || descriptor.initial_folding_factor != 0
                || descriptor.initial_oracle_width != 1
                || descriptor.rows_per_query != TERMINAL_PRODUCTION_WHIR_FOLD_ARITY
                || descriptor.code_layout_version
                    != u64::from(COEFFICIENT_SUBGROUP_RS_LAYOUT_VERSION)
                || descriptor.field_modulus != F::ORDER_U64
                || descriptor.subgroup_generator
                    != F::two_adic_generator(initial_log_domain_size).as_canonical_u64()
                || descriptor.coset_shift != 0
                || descriptor.coordinate_ordering != 0
            {
                return Err("terminal coefficient subgroup layout");
            }
            Ok(WhirInitialRsLayout::CoefficientSubgroup)
        }
        _ => Err("terminal descriptor domain"),
    }
}

fn terminal_point_lookup_counts(
    k: usize,
    queries: &[usize],
    final_poly_len: usize,
    log_message_len: usize,
) -> Option<Vec<usize>> {
    let round_count = queries.len();
    let point_len = k
        .checked_mul(round_count)?
        .checked_add(final_poly_len.ilog2() as usize)?;
    if point_len != log_message_len {
        return None;
    }
    // Structured linearizer selector plus RS-adjoint selector.
    let mut counts = vec![2usize; point_len];
    for (round, &query_count) in queries.iter().enumerate() {
        let after_folds = (round + 1).checked_mul(k)?;
        let terms = query_count + usize::from(round + 1 != round_count);
        for count in counts.get_mut(after_folds..)? {
            *count = count.checked_add(terms)?;
        }
    }
    let suffix_start = k.checked_mul(round_count)?;
    for count in counts.get_mut(suffix_start..)? {
        *count = count.checked_add(final_poly_len)?;
    }
    Some(counts)
}

/// Private typed bus namespace. Allocation order is part of
/// [`FixedMultiAirTerminalCircuit::terminal_vk_digest`].
#[derive(Clone, Debug)]
pub struct FixedMultiAirTerminalBusInventory {
    pub transcript: TranscriptBus,
    pub exp_bits_len: ExpBitsLenBus,
    pub right_shift: RightShiftBus,
    pub beta: FixedMultiAirBetaCoordinateBus,
    pub binding: FixedMultiAirTerminalBindingBus,
    pub instance: FixedMultiAirTerminalInstanceValueBus,
    pub global_claim: FixedMultiAirGlobalClaimBus,
    pub region_claim: FixedMultiAirRegionClaimBus,
    pub padding_claim: FixedMultiAirPaddingClaimBus,
    pub padding_point: FixedMultiAirPaddingPointBus,
    pub padding_sumcheck_final: FixedMultiAirPaddingSumcheckFinalBus,
    pub padding_opening: FixedMultiAirPaddingOpeningBus,
    pub region_start: FixedMultiAirRegionStartBus,
    pub region_point: FixedMultiAirRegionPointBus,
    pub region_sumcheck_final: FixedMultiAirRegionSumcheckFinalBus,
    pub decomposition_contribution: FixedMultiAirDecompositionContributionBus,
    pub region_opening: FixedMultiAirRegionOpeningBus,
    pub region_rho: FixedMultiAirRegionRhoBus,
    pub region_local_padding_point: FixedMultiAirRegionLocalPaddingPointBus,
    pub region_final_evaluation: FixedMultiAirRegionFinalEvaluationBus,
    pub structured_header: FixedMultiAirStructuredClaimHeaderBus,
    pub mapped_term: FixedMultiAirMappedTermBus,
    pub structured_point: FixedMultiAirStructuredPointBus,
    pub whir_start: FixedMultiAirWhirStartBus,
    pub batched_claim: FixedMultiAirBatchedClaimBus,
    pub linearizer_aux_point: FixedMultiAirLinearizerAuxPointBus,
    pub linearizer_sumcheck_final: FixedMultiAirLinearizerSumcheckFinalBus,
    pub linearizer_y: FixedMultiAirLinearizerYBus,
    pub linearizer_selector: FixedMultiAirLinearizerSelectorBus,
    pub linearizer_raw_term: FixedMultiAirLinearizerRawTermBus,
    pub linearizer_raw_weight: FixedMultiAirLinearizerRawWeightBus,
    pub endpoint_node: FixedMultiAirEndpointNodeBus,
    pub endpoint_fixed_value: FixedMultiAirEndpointFixedValueBus,
    pub endpoint_fixed_state: FixedMultiAirEndpointFixedFoldStateBus,
    pub constraint_weight: FixedMultiAirConstraintWeightBus,
    pub constraint_weight_state: FixedMultiAirConstraintWeightStateBus,
    pub whir_round: NativeTerminalWhirRoundBus,
    pub whir_alpha: NativeTerminalWhirAlphaBus,
    pub whir_verify_queries: NativeTerminalWhirVerifyQueriesBus,
    pub whir_query: NativeTerminalWhirQueryBus,
    pub whir_folding: NativeTerminalWhirFoldingBus,
    pub whir_statement: NativeTerminalWhirStatementBus,
    pub whir_final_claim: NativeTerminalWhirFinalClaimBus,
    pub whir_final_context: NativeTerminalWhirFinalContextBus,
    pub whir_point: NativeTerminalWhirPointBus,
    pub whir_final_poly: NativeTerminalWhirFinalPolyBus,
    pub whir_final_weight: NativeTerminalWhirFinalWeightBus,
    pub whir_actual_weight: NativeTerminalWhirActualWeightBus,
    pub adjoint_round: NativeTerminalRsAdjointRoundBus,
    pub adjoint_claim: NativeTerminalRsAdjointClaimBus,
    pub accumulator_value: NativeTerminalAccumulatorValueBus,
    pub accumulator_root: NativeTerminalAccumulatorRootBus,
    pub adjoint_q: NativeTerminalRsAdjointQBus,
    pub adjoint_y: NativeTerminalRsAdjointYBus,
    pub adjoint_value: NativeTerminalRsAdjointValueBus,
    pub weight_term: NativeTerminalWhirWeightTermBus,
    pub weight_term_result: NativeTerminalWhirWeightTermResultBus,
    pub linearizer_weight: NativeTerminalWhirLinearizerWeightBus,
    pub leaf_value: NativeLeafValueBus,
    pub opening_leaf: NativeOpeningLeafBus,
    pub merkle_node: NativeMerkleNodeBus,
    pub merkle_root: NativeMerkleRootBus,
    next_bus_idx: BusIndex,
}

impl FixedMultiAirTerminalBusInventory {
    #[must_use]
    pub fn new(first_bus_idx: BusIndex) -> Self {
        let mut b = BusIndexManager::from_next_bus_idx(first_bus_idx);
        macro_rules! bus {
            ($ty:ty) => {{
                <$ty>::new(b.new_bus_idx())
            }};
        }
        let result = Self {
            transcript: bus!(TranscriptBus),
            exp_bits_len: bus!(ExpBitsLenBus),
            right_shift: bus!(RightShiftBus),
            beta: bus!(FixedMultiAirBetaCoordinateBus),
            binding: bus!(FixedMultiAirTerminalBindingBus),
            instance: bus!(FixedMultiAirTerminalInstanceValueBus),
            global_claim: bus!(FixedMultiAirGlobalClaimBus),
            region_claim: bus!(FixedMultiAirRegionClaimBus),
            padding_claim: bus!(FixedMultiAirPaddingClaimBus),
            padding_point: bus!(FixedMultiAirPaddingPointBus),
            padding_sumcheck_final: bus!(FixedMultiAirPaddingSumcheckFinalBus),
            padding_opening: bus!(FixedMultiAirPaddingOpeningBus),
            region_start: bus!(FixedMultiAirRegionStartBus),
            region_point: bus!(FixedMultiAirRegionPointBus),
            region_sumcheck_final: bus!(FixedMultiAirRegionSumcheckFinalBus),
            decomposition_contribution: bus!(FixedMultiAirDecompositionContributionBus),
            region_opening: bus!(FixedMultiAirRegionOpeningBus),
            region_rho: bus!(FixedMultiAirRegionRhoBus),
            region_local_padding_point: bus!(FixedMultiAirRegionLocalPaddingPointBus),
            region_final_evaluation: bus!(FixedMultiAirRegionFinalEvaluationBus),
            structured_header: bus!(FixedMultiAirStructuredClaimHeaderBus),
            mapped_term: bus!(FixedMultiAirMappedTermBus),
            structured_point: bus!(FixedMultiAirStructuredPointBus),
            whir_start: bus!(FixedMultiAirWhirStartBus),
            batched_claim: bus!(FixedMultiAirBatchedClaimBus),
            linearizer_aux_point: bus!(FixedMultiAirLinearizerAuxPointBus),
            linearizer_sumcheck_final: bus!(FixedMultiAirLinearizerSumcheckFinalBus),
            linearizer_y: bus!(FixedMultiAirLinearizerYBus),
            linearizer_selector: bus!(FixedMultiAirLinearizerSelectorBus),
            linearizer_raw_term: bus!(FixedMultiAirLinearizerRawTermBus),
            linearizer_raw_weight: bus!(FixedMultiAirLinearizerRawWeightBus),
            endpoint_node: bus!(FixedMultiAirEndpointNodeBus),
            endpoint_fixed_value: bus!(FixedMultiAirEndpointFixedValueBus),
            endpoint_fixed_state: bus!(FixedMultiAirEndpointFixedFoldStateBus),
            constraint_weight: bus!(FixedMultiAirConstraintWeightBus),
            constraint_weight_state: bus!(FixedMultiAirConstraintWeightStateBus),
            whir_round: bus!(NativeTerminalWhirRoundBus),
            whir_alpha: bus!(NativeTerminalWhirAlphaBus),
            whir_verify_queries: bus!(NativeTerminalWhirVerifyQueriesBus),
            whir_query: bus!(NativeTerminalWhirQueryBus),
            whir_folding: bus!(NativeTerminalWhirFoldingBus),
            whir_statement: bus!(NativeTerminalWhirStatementBus),
            whir_final_claim: bus!(NativeTerminalWhirFinalClaimBus),
            whir_final_context: bus!(NativeTerminalWhirFinalContextBus),
            whir_point: bus!(NativeTerminalWhirPointBus),
            whir_final_poly: bus!(NativeTerminalWhirFinalPolyBus),
            whir_final_weight: bus!(NativeTerminalWhirFinalWeightBus),
            whir_actual_weight: bus!(NativeTerminalWhirActualWeightBus),
            adjoint_round: bus!(NativeTerminalRsAdjointRoundBus),
            adjoint_claim: bus!(NativeTerminalRsAdjointClaimBus),
            accumulator_value: bus!(NativeTerminalAccumulatorValueBus),
            accumulator_root: bus!(NativeTerminalAccumulatorRootBus),
            adjoint_q: bus!(NativeTerminalRsAdjointQBus),
            adjoint_y: bus!(NativeTerminalRsAdjointYBus),
            adjoint_value: bus!(NativeTerminalRsAdjointValueBus),
            weight_term: bus!(NativeTerminalWhirWeightTermBus),
            weight_term_result: bus!(NativeTerminalWhirWeightTermResultBus),
            linearizer_weight: bus!(NativeTerminalWhirLinearizerWeightBus),
            leaf_value: bus!(NativeLeafValueBus),
            opening_leaf: bus!(NativeOpeningLeafBus),
            merkle_node: bus!(NativeMerkleNodeBus),
            merkle_root: bus!(NativeMerkleRootBus),
            next_bus_idx: 0,
        };
        Self {
            next_bus_idx: b.next_bus_idx(),
            ..result
        }
    }

    #[must_use]
    pub const fn next_bus_idx(&self) -> BusIndex {
        self.next_bus_idx
    }
}

/// Aggregate terminal circuit with bounded AIR widths and setup-fixed row
/// schedules.
pub struct FixedMultiAirTerminalCircuit {
    pub profile: FixedMultiAirTerminalCircuitProfile,
    pub buses: FixedMultiAirTerminalBusInventory,
    first_bus_idx: BusIndex,
    pub(super) transcript: NativeWarpTranscriptModule,
    shared: BusInventory,
    pub(super) config: BabyBearPoseidon2Config,
    terminal_vk_digest: Digest,
}

impl FixedMultiAirTerminalCircuit {
    pub fn new(
        profile: FixedMultiAirTerminalCircuitProfile,
        shared: &BusInventory,
        first_bus_idx: BusIndex,
        system_params: SystemParams,
        hasher: &impl MerkleHasher<F = F, Digest = Digest>,
    ) -> Result<Self, &'static str> {
        let buses = FixedMultiAirTerminalBusInventory::new(first_bus_idx);
        let transcript = NativeWarpTranscriptModule::new_for_bus(
            shared,
            buses.transcript,
            system_params.clone(),
        );
        let terminal_vk_digest =
            hasher.hash_slice(&profile.digest_words(first_bus_idx, buses.next_bus_idx()));
        let circuit = Self {
            profile,
            buses,
            first_bus_idx,
            transcript,
            shared: shared.clone(),
            config: BabyBearPoseidon2Config::default_from_params(system_params),
            terminal_vk_digest,
        };
        circuit.round_air()?;
        Ok(circuit)
    }

    #[must_use]
    pub const fn first_bus_idx(&self) -> BusIndex {
        self.first_bus_idx
    }

    #[must_use]
    pub const fn next_bus_idx(&self) -> BusIndex {
        self.buses.next_bus_idx()
    }

    #[must_use]
    pub const fn terminal_vk_digest(&self) -> Digest {
        self.terminal_vk_digest
    }

    /// Logical Poseidon bus owner contributed by the terminal transcript.
    /// Parent direct-final assemblies must include this owner in the sole
    /// shared multi-bus Poseidon table together with all History owners.
    #[must_use]
    pub fn poseidon2_bus_owner(&self) -> Poseidon2BusOwner {
        self.transcript.poseidon2_bus_owner()
    }

    /// STARK parameters used by the terminal transcript AIRs and by the SDK
    /// when committing setup-fixed cached tables. Cached commitments must use
    /// the same PCS parameters as the outer prover.
    #[must_use]
    pub fn system_params(&self) -> &SystemParams {
        self.config.params()
    }

    #[must_use]
    pub const fn binding_bus(&self) -> FixedMultiAirTerminalBindingBus {
        self.buses.binding
    }

    #[must_use]
    pub const fn instance_bus(&self) -> FixedMultiAirTerminalInstanceValueBus {
        self.buses.instance
    }

    #[must_use]
    pub const fn whir_root_bus(&self) -> NativeTerminalAccumulatorRootBus {
        self.buses.accumulator_root
    }

    pub(super) fn round_air(&self) -> Result<NativeTerminalWhirRoundAir, &'static str> {
        NativeTerminalWhirRoundAir::new(
            self.buses.transcript,
            self.buses.whir_statement,
            self.buses.whir_round,
            self.buses.whir_verify_queries,
            self.buses.whir_final_claim,
            self.buses.weight_term,
            self.buses.merkle_root,
            self.buses.exp_bits_len,
            self.profile.k,
            self.profile.initial_log_domain_size,
            self.profile.final_poly_len,
            self.profile.query_pow_bits,
            self.profile.folding_pow_bits,
            F::GENERATOR,
            0,
            self.profile.num_queries_per_round.clone(),
        )
    }

    pub(super) fn linearizer_y_air(&self) -> FixedMultiAirLinearizerYAir {
        FixedMultiAirLinearizerYAir {
            point_bus: self.buses.linearizer_aux_point,
            whir_point_bus: self.buses.whir_point,
            y_bus: self.buses.linearizer_y,
            log_message_len: self.profile.log_message_len,
            // The AIR predates coefficient-native messages and retains the
            // old field name. Its semantic value is the selector dimension
            // derived from the authenticated RS layout.
            initial_folding_factor: self.profile.selector_folding_factor,
        }
    }

    /// Exact AIR order committed by `terminal_vk_digest` and returned by the
    /// CPU/CUDA trace providers.
    #[must_use]
    pub fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        let p = &self.profile;
        let b = &self.buses;
        let mut transcript_airs = self.transcript.airs::<SC>().into_iter();
        let mut airs = vec![transcript_airs
            .next()
            .expect("terminal transcript AIR is present")];
        add(
            &mut airs,
            FixedMultiAirTerminalPrefixAir {
                transcript_bus: b.transcript,
                binding_bus: b.binding,
                instance_bus: b.instance,
                beta_bus: b.beta,
                global_bus: b.global_claim,
                region_claim_bus: b.region_claim,
                padding_claim_bus: b.padding_claim,
                relation_digest: p.relation_digest,
                alpha_len: p.alpha_len,
                beta_len: p.beta_len,
                binding_lookup_count: 2,
            },
        );
        for plan in &p.decomposition {
            add(
                &mut airs,
                FixedMultiAirDecompositionComponentAir {
                    beta_bus: b.beta,
                    global_bus: b.global_claim,
                    region_claim_bus: b.region_claim,
                    padding_claim_bus: b.padding_claim,
                    contribution_bus: b.decomposition_contribution,
                    plan: plan.clone(),
                },
            );
        }
        add(
            &mut airs,
            FixedMultiAirDecompositionSumAir {
                global_bus: b.global_claim,
                contribution_bus: b.decomposition_contribution,
                component_count: p.decomposition.len(),
            },
        );
        for endpoint in &p.endpoints {
            let region = endpoint.region;
            let local = p
                .relation
                .region_relation(region)
                .expect("validated endpoint region");
            let constraint_count = local.constraint_dag().constraint_idx.len();
            add(
                &mut airs,
                FixedMultiAirRegionPrefixAir {
                    transcript_bus: b.transcript,
                    beta_bus: b.beta,
                    region_claim_bus: b.region_claim,
                    start_bus: b.region_start,
                    region,
                },
            );
            add(
                &mut airs,
                FixedMultiAirRegionSumcheckAir::new(
                    b.transcript,
                    b.region_start,
                    b.region_point,
                    b.region_sumcheck_final,
                    region,
                    endpoint.log_height,
                    endpoint.dense_fixed_source_count(),
                    endpoint.dynamic.len()
                        + constraint_count
                        + endpoint.analytic_fixed_source_count(),
                ),
            );
            add(
                &mut airs,
                FixedMultiAirRegionTailAir {
                    transcript_bus: b.transcript,
                    sumcheck_final_bus: b.region_sumcheck_final,
                    final_evaluation_bus: b.region_final_evaluation,
                    opening_bus: b.region_opening,
                    rho_bus: b.region_rho,
                    local_padding_point_bus: b.region_local_padding_point,
                    region,
                },
            );
            add(
                &mut airs,
                FixedMultiAirEndpointFixedAir {
                    point_bus: b.region_point,
                    state_bus: b.endpoint_fixed_state,
                    fixed_value_bus: b.endpoint_fixed_value,
                    region,
                },
            );
            add(
                &mut airs,
                FixedMultiAirConstraintWeightInitAir {
                    beta_bus: b.beta,
                    state_bus: b.constraint_weight_state,
                    weight_bus: b.constraint_weight,
                    region,
                    zero_log_height: endpoint.log_height == 0,
                },
            );
            add(
                &mut airs,
                FixedMultiAirConstraintWeightDpAir {
                    beta_bus: b.beta,
                    point_bus: b.region_point,
                    state_bus: b.constraint_weight_state,
                    weight_bus: b.constraint_weight,
                    region,
                },
            );
            add(
                &mut airs,
                FixedMultiAirEndpointNodeAir {
                    opening_bus: b.region_opening,
                    fixed_value_bus: b.endpoint_fixed_value,
                    beta_bus: b.beta,
                    node_bus: b.endpoint_node,
                    region,
                    one_coordinate: endpoint.one_coordinate,
                },
            );
            add(
                &mut airs,
                FixedMultiAirEndpointFoldAir {
                    node_bus: b.endpoint_node,
                    weight_bus: b.constraint_weight,
                    beta_bus: b.beta,
                    final_evaluation_bus: b.region_final_evaluation,
                    region,
                    one_coordinate: endpoint.one_coordinate,
                },
            );
            add(
                &mut airs,
                FixedMultiAirMappedTermAir {
                    opening_bus: b.region_opening,
                    rho_bus: b.region_rho,
                    header_bus: b.structured_header,
                    term_bus: b.mapped_term,
                    region,
                    global_log_message: p.log_message_len,
                    term_count: endpoint.dynamic.len(),
                },
            );
            add(
                &mut airs,
                FixedMultiAirMappedPointAir {
                    point_bus: b.region_point,
                    structured_point_bus: b.structured_point,
                    region,
                },
            );
            debug_assert!(constraint_count > 0);
        }
        if p.relation.description().padding_constraint_count != 0 {
            add(
                &mut airs,
                FixedMultiAirPaddingSumcheckAir {
                    transcript_bus: b.transcript,
                    claim_bus: b.padding_claim,
                    point_bus: b.padding_point,
                    final_bus: b.padding_sumcheck_final,
                    round_count: p.log_message_len,
                    point_lookup_count: 1,
                },
            );
            add(
                &mut airs,
                FixedMultiAirPaddingOpeningAir {
                    transcript_bus: b.transcript,
                    final_bus: b.padding_sumcheck_final,
                    opening_bus: b.padding_opening,
                },
            );
            add(
                &mut airs,
                FixedMultiAirPaddingWeightAir {
                    beta_bus: b.beta,
                    point_bus: b.padding_point,
                    opening_bus: b.padding_opening,
                    header_bus: b.structured_header,
                    structured_point_bus: b.structured_point,
                    claim_index: p.relation.region_count(),
                    log_message_len: p.log_message_len,
                    one_coordinate: p.relation.pesat_shape().log_constraints,
                    scale_exponent: p.relation.description().exact_max_degree.saturating_sub(1)
                        as usize,
                },
            );
        }
        add(
            &mut airs,
            FixedMultiAirWhirPrefixAir {
                transcript_bus: b.transcript,
                binding_bus: b.binding,
                instance_bus: b.instance,
                start_bus: b.whir_start,
                root_bus: b.accumulator_root,
                relation_digest: p.relation_digest,
                metadata_words: p.metadata_words.clone(),
                alpha_len: p.alpha_len,
                beta_len: p.beta_len,
            },
        );
        add(
            &mut airs,
            FixedMultiAirStructuredTargetAir {
                start_bus: b.whir_start,
                header_bus: b.structured_header,
                batched_claim_bus: b.batched_claim,
                statement_bus: b.whir_statement,
                claim_count: p.claim_count(),
            },
        );
        add(
            &mut airs,
            FixedMultiAirAccumulatorBridgeAir {
                fixed_bus: b.instance,
                native_bus: b.accumulator_value,
            },
        );
        add(
            &mut airs,
            FixedMultiAirLinearizerSumcheckAir {
                transcript_bus: b.transcript,
                point_bus: b.linearizer_aux_point,
                final_bus: b.linearizer_sumcheck_final,
                log_message_len: p.log_message_len,
            },
        );
        add(&mut airs, self.linearizer_y_air());
        add(
            &mut airs,
            FixedMultiAirLinearizerSelectorAir {
                y_bus: b.linearizer_y,
                selector_bus: b.linearizer_selector,
                log_message_len: p.log_message_len,
            },
        );
        add(
            &mut airs,
            FixedMultiAirLinearizerRawAir {
                batched_claim_bus: b.batched_claim,
                mapped_term_bus: b.mapped_term,
                structured_point_bus: b.structured_point,
                aux_point_bus: b.linearizer_aux_point,
                raw_term_bus: b.linearizer_raw_term,
                log_message_len: p.log_message_len,
                component_count: p.linearizer_components.len(),
            },
        );
        add(
            &mut airs,
            FixedMultiAirLinearizerRawSumAir {
                term_bus: b.linearizer_raw_term,
                output_bus: b.linearizer_raw_weight,
                component_count: p.linearizer_components.len(),
            },
        );
        add(
            &mut airs,
            FixedMultiAirLinearizerFinalAir {
                statement_bus: b.whir_statement,
                sumcheck_bus: b.linearizer_sumcheck_final,
                raw_weight_bus: b.linearizer_raw_weight,
                selector_bus: b.linearizer_selector,
                output_bus: b.linearizer_weight,
            },
        );
        add(
            &mut airs,
            NativeTerminalWhirSumcheckAir {
                transcript_bus: b.transcript,
                round_bus: b.whir_round,
                alpha_bus: b.whir_alpha,
                exp_bits_len_bus: b.exp_bits_len,
                k: p.k,
                round_count: p.round_count(),
                folding_pow_bits: p.folding_pow_bits,
                generator: F::GENERATOR,
            },
        );
        add(
            &mut airs,
            self.round_air().expect("validated terminal round"),
        );
        add(
            &mut airs,
            NativeTerminalWhirQueryAir {
                transcript_bus: b.transcript,
                verify_queries_bus: b.whir_verify_queries,
                query_bus: b.whir_query,
                weight_term_bus: b.weight_term,
                exp_bits_len_bus: b.exp_bits_len,
                right_shift_bus: b.right_shift,
                k: p.k,
                initial_log_domain_size: p.initial_log_domain_size,
                round_count: p.round_count(),
                final_poly_len: p.final_poly_len,
                inner_tree_id_offset: p.round_count(),
                outer_tree_id_offset: 0,
                evaluation_layout: false,
                coefficient_two_coset_initial: false,
            },
        );
        add(
            &mut airs,
            NativeTerminalWhirOpenedAir {
                query_bus: b.whir_query,
                folding_bus: b.whir_folding,
                leaf_value_bus: b.leaf_value,
                opening_leaf_bus: b.opening_leaf,
                k: p.k,
                inner_depth: 0,
            },
        );
        add(
            &mut airs,
            NativeTerminalWhirFoldingAir {
                alpha_bus: b.whir_alpha,
                folding_bus: b.whir_folding,
                k: p.k,
                evaluation_layout: false,
            },
        );
        add(
            &mut airs,
            NativeLeafHashAir {
                permute_bus: self.shared.poseidon2_permute_bus,
                value_bus: b.leaf_value,
                leaf_bus: b.opening_leaf,
            },
        );
        add(
            &mut airs,
            NativeMerkleMultiproofAir {
                compress_bus: self.shared.poseidon2_compress_bus,
                leaf_bus: b.opening_leaf,
                node_bus: b.merkle_node,
                root_bus: b.merkle_root,
            },
        );
        add(
            &mut airs,
            NativeTerminalWhirFinalTableAir {
                transcript_bus: b.transcript,
                final_claim_bus: b.whir_final_claim,
                final_context_bus: b.whir_final_context,
                final_poly_bus: b.whir_final_poly,
                final_weight_bus: b.whir_final_weight,
                final_len: p.final_poly_len,
            },
        );
        add(
            &mut airs,
            NativeTerminalWhirMobiusAir::new(b.whir_final_poly, p.final_log_len()),
        );
        add(
            &mut airs,
            NativeTerminalWhirPointAir::new(
                b.transcript,
                b.whir_alpha,
                b.whir_final_context,
                b.whir_point,
                p.k,
                p.round_count(),
                p.final_poly_len,
                p.point_lookup_counts.clone(),
            ),
        );
        add(
            &mut airs,
            NativeTerminalWhirFinalCheckAir {
                final_context_bus: b.whir_final_context,
                final_poly_bus: b.whir_final_poly,
                final_weight_bus: b.whir_final_weight,
                point_bus: b.whir_point,
                actual_weight_bus: b.whir_actual_weight,
                point_prefix_len: p.k * p.round_count(),
                final_log_len: p.final_log_len(),
            },
        );
        add(
            &mut airs,
            NativeTerminalWhirWeightTermAir {
                term_bus: b.weight_term,
                result_bus: b.weight_term_result,
                point_bus: b.whir_point,
            },
        );
        add(
            &mut airs,
            NativeTerminalRsAdjointSumcheckAir::new(
                b.transcript,
                b.adjoint_round,
                b.adjoint_claim,
                p.rs_adjoint_round_count,
                p.rs_adjoint_degree,
            ),
        );
        add(
            &mut airs,
            NativeTerminalRsAdjointQAir {
                round_bus: b.adjoint_round,
                accumulator_value_bus: b.accumulator_value,
                point_bus: b.whir_point,
                q_bus: b.adjoint_q,
                round_count: p.alpha_len,
                initial_folding_factor: p.k,
            },
        );
        add(
            &mut airs,
            NativeTerminalRsAdjointYAir::new(
                b.adjoint_round,
                b.adjoint_y,
                p.log_message_len - p.k,
                p.alpha_len - p.k,
            ),
        );
        add(
            &mut airs,
            NativeTerminalRsAdjointSelectorAir {
                point_bus: b.whir_point,
                q_bus: b.adjoint_q,
                y_bus: b.adjoint_y,
                claim_bus: b.adjoint_claim,
                value_bus: b.adjoint_value,
                log_message_len: p.log_message_len - p.k,
                point_coordinate_offset: p.k,
            },
        );
        add(
            &mut airs,
            NativeTerminalWhirExpectedWeightAir {
                actual_bus: b.whir_actual_weight,
                adjoint_bus: b.adjoint_value,
                linearizer_bus: b.linearizer_weight,
                term_bus: b.weight_term_result,
                term_count: p.weight_term_count,
            },
        );
        add(&mut airs, ExpBitsLenAir::new(b.exp_bits_len, b.right_shift));
        airs
    }
}

fn add<SC, A>(airs: &mut Vec<AirRef<SC>>, air: A)
where
    SC: StarkProtocolConfig,
    A: AnyAir<SC> + 'static,
{
    airs.push(Arc::new(air));
}

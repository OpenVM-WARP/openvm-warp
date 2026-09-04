use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder,
    interaction::InteractionBuilder,
    native_warp::{
        DirectAirMappedRotation, FixedMultiAirCompleteTerminalCircuitPlan, FixedMultiAirPesatIndex,
        FIXED_MULTI_AIR_COMPLETE_TERMINAL_CIRCUIT_PLAN_VERSION,
    },
    transcript::TranscriptLog,
    warp_accum::WhirInitialRsLayout,
    warp_pesat::{
        evaluate_mle, AlgebraicChallenger, PrismalinearMappedColumnRotation,
        TerminalStructuredLinearClaim, TerminalWeightSpec,
    },
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing, TwoAdicField,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;

use crate::{
    bus::TranscriptBus,
    native_warp::terminal::{
        fixed_multi_air::{
            FixedMultiAirBatchedClaimBus, FixedMultiAirBatchedClaimMessage,
            FixedMultiAirEndpointPlan, FixedMultiAirLinearizerAuxPointBus,
            FixedMultiAirLinearizerAuxPointMessage, FixedMultiAirLinearizerRawTermBus,
            FixedMultiAirLinearizerRawTermMessage, FixedMultiAirLinearizerRawWeightBus,
            FixedMultiAirLinearizerRawWeightMessage, FixedMultiAirLinearizerSelectorBus,
            FixedMultiAirLinearizerSelectorMessage, FixedMultiAirLinearizerSumcheckFinalBus,
            FixedMultiAirLinearizerSumcheckFinalMessage, FixedMultiAirLinearizerYBus,
            FixedMultiAirLinearizerYMessage, FixedMultiAirMappedTermBus,
            FixedMultiAirMappedTermMessage,
        },
        NativeTerminalWhirLinearizerWeightBus, NativeTerminalWhirLinearizerWeightMessage,
        NativeTerminalWhirPointBus, NativeTerminalWhirPointMessage, NativeTerminalWhirStatementBus,
        NativeTerminalWhirStatementMessage,
    },
    utils::{ext_field_add, ext_field_multiply, ext_field_subtract},
};

/// BabyBear supports row-DFT roots through order 2^27.  The vector-alphabet
/// terminal layout may nevertheless contain 2^28 systematic EF4 symbols when
/// at least one low message bit is handled by the initial column alphabet.
/// Keep one extra slot so the degree-(m+1) product sumcheck has all m+2
/// evaluation rows.
pub const FIXED_MULTI_AIR_LINEARIZER_MAX_LOG_MESSAGE: usize = 28;
pub const FIXED_MULTI_AIR_LINEARIZER_MAX_EVALUATIONS: usize =
    FIXED_MULTI_AIR_LINEARIZER_MAX_LOG_MESSAGE + 2;

const LINEARIZER_ADJOINT_ROUND_TAG: u64 = 0x4e57_4d41_4c41_0001;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedMultiAirLinearizerAdjointTraceError {
    Shape,
    Transcript,
    Claim,
}

/// Private recursive-compression certificate for the exact structured
/// systematic-message dual.  It is not part of the backend WARP/WHIR proof;
/// every round is checked by the AIR below and Fiat--Shamir-bound after the
/// backend transcript.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedMultiAirLinearizerAdjointProof {
    pub round_evaluations: Vec<Vec<EF>>,
}

/// Output of the deliberately small reference prover used for differential
/// tests and as a correctness oracle for the production/CUDA implementation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedMultiAirLinearizerAdjointReferenceOutput {
    pub proof: FixedMultiAirLinearizerAdjointProof,
    pub initial_claim: EF,
    pub point: Vec<EF>,
    pub final_claim: EF,
}

/// Scheduling parameters for the production structured-dual sumcheck.
///
/// A tile is only a deterministic range of Boolean suffix indices. The
/// implementation never allocates a tile-sized field vector: each worker
/// keeps two bounded point buffers and two degree-`m + 1` coefficient arrays.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedMultiAirLinearizerAdjointTiledConfig {
    pub tile_log_size: usize,
}

impl Default for FixedMultiAirLinearizerAdjointTiledConfig {
    fn default() -> Self {
        Self { tile_log_size: 12 }
    }
}

/// Auditable memory plan for the production prover. `dense_weight_elements`
/// is deliberately always zero; changing that is a protocol-performance
/// regression and is pinned by the log-27 test.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedMultiAirLinearizerAdjointTiledPlan {
    pub log_message_len: usize,
    pub tile_log_size: usize,
    pub max_worker_field_elements: usize,
    pub dense_weight_elements: usize,
}

pub fn fixed_multi_air_linearizer_adjoint_tiled_plan(
    log_message_len: usize,
    config: FixedMultiAirLinearizerAdjointTiledConfig,
) -> Result<FixedMultiAirLinearizerAdjointTiledPlan, FixedMultiAirLinearizerAdjointTraceError> {
    if log_message_len == 0
        || log_message_len > FIXED_MULTI_AIR_LINEARIZER_MAX_LOG_MESSAGE
        || config.tile_log_size >= usize::BITS as usize
    {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
    }
    Ok(FixedMultiAirLinearizerAdjointTiledPlan {
        log_message_len,
        tile_log_size: config.tile_log_size,
        // point-at-zero, point-at-one, selector coefficients, and the
        // per-tile accumulated round polynomial.
        max_worker_field_elements: 2 * FIXED_MULTI_AIR_LINEARIZER_MAX_LOG_MESSAGE
            + 2 * FIXED_MULTI_AIR_LINEARIZER_MAX_EVALUATIONS,
        dense_weight_elements: 0,
    })
}

/// Number of low WHIR variables whose dual is the direct inverse-zeta bit
/// selector.  In the coefficient-native two-coset layout every message bit
/// has that form.  In the ordinary subgroup layout only the vector-alphabet
/// prefix does; the remaining row variables require the DFT pullback.
pub fn structured_selector_folding_factor(
    layout: WhirInitialRsLayout,
    log_message_len: usize,
    initial_folding_factor: usize,
) -> Result<usize, FixedMultiAirLinearizerAdjointTraceError> {
    if initial_folding_factor > log_message_len {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
    }
    match layout {
        WhirInitialRsLayout::OrdinarySubgroup => {
            if log_message_len - initial_folding_factor > F::TWO_ADICITY {
                return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
            }
            Ok(initial_folding_factor)
        }
        WhirInitialRsLayout::CoefficientSubgroup | WhirInitialRsLayout::CoefficientTwoCosetGrs => {
            if initial_folding_factor != 0 || log_message_len > F::TWO_ADICITY {
                return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
            }
            Ok(log_message_len)
        }
    }
}

#[derive(Clone, Copy)]
struct FixedMultiAirPreparedLinearizerComponent<'a> {
    block_start: usize,
    log_height: usize,
    rotation: DirectAirMappedRotation,
    row_point: &'a [EF],
    scale: EF,
    is_zero: bool,
}

fn prepare_fixed_multi_air_linearizer_components<'a>(
    log_message_len: usize,
    components: &[FixedMultiAirLinearizerRawComponentPlan],
    claims: &'a [TerminalStructuredLinearClaim<EF>],
    batching_scales: &[EF],
) -> Result<
    Vec<FixedMultiAirPreparedLinearizerComponent<'a>>,
    FixedMultiAirLinearizerAdjointTraceError,
> {
    if components.is_empty()
        || claims.is_empty()
        || claims.len() != batching_scales.len()
        || claims
            .iter()
            .any(|claim| claim.weight.len() != (1usize << log_message_len))
    {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
    }
    let message_len = 1usize
        .checked_shl(log_message_len as u32)
        .ok_or(FixedMultiAirLinearizerAdjointTraceError::Shape)?;
    components
        .iter()
        .enumerate()
        .map(|(ordinal, component)| {
            if component.ordinal != ordinal
                || component.claim >= claims.len()
                || component.log_height > log_message_len
                || 1usize
                    .checked_shl(component.log_height as u32)
                    .and_then(|height| component.block_start.checked_add(height))
                    .is_none_or(|end| end > message_len)
            {
                return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
            }
            let claim = &claims[component.claim];
            let (row_point, term_scale, is_zero) = match &claim.weight {
                TerminalWeightSpec::Eq { point } if component.is_eq => {
                    if point.len() != log_message_len
                        || component.log_height != log_message_len
                        || component.block_start != 0
                        || component.rotation != DirectAirMappedRotation::Current
                    {
                        return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
                    }
                    (point.as_slice(), EF::ONE, false)
                }
                TerminalWeightSpec::PrismalinearMappedColumns(weight)
                    if !component.is_eq && !component.is_zero =>
                {
                    let term = weight
                        .terms
                        .get(component.term)
                        .ok_or(FixedMultiAirLinearizerAdjointTraceError::Shape)?;
                    let rotation = match component.rotation {
                        DirectAirMappedRotation::Current => {
                            PrismalinearMappedColumnRotation::Current
                        }
                        DirectAirMappedRotation::Next => PrismalinearMappedColumnRotation::Next,
                    };
                    if weight.log_message_len != log_message_len
                        || term.block.start != component.block_start
                        || term.block.log_height != component.log_height
                        || term.l_skip != 0
                        || term.barycentric_weights.as_slice() != [EF::ONE]
                        || term.rotation != rotation
                        || term.folded_row_eq_point.len() != component.log_height
                    {
                        return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
                    }
                    (term.folded_row_eq_point.as_slice(), term.scale, false)
                }
                TerminalWeightSpec::PrismalinearMappedColumns(weight)
                    if component.is_zero && weight.terms.is_empty() =>
                {
                    (&[] as &[EF], EF::ZERO, true)
                }
                _ => return Err(FixedMultiAirLinearizerAdjointTraceError::Shape),
            };
            Ok(FixedMultiAirPreparedLinearizerComponent {
                block_start: component.block_start,
                log_height: component.log_height,
                rotation: component.rotation,
                row_point,
                scale: batching_scales[component.claim] * term_scale,
                is_zero,
            })
        })
        .collect()
}

/// Production prover for the exact structured-message dual certificate.
///
/// Challenge and evaluation order are byte-for-byte the reference protocol.
/// The prover sums each round polynomial in coefficient form over deterministic
/// suffix tiles. For a fixed suffix, the compact message weight is multilinear
/// in the active variable and the inverse-transpose zeta selector is a product
/// of at most `round + 1` nonconstant linear factors. Consequently no dense
/// `2^m` weight or selector table is materialized, including for `m = 27`.
pub fn prove_fixed_multi_air_linearizer_adjoint_tiled<Ch>(
    components: &[FixedMultiAirLinearizerRawComponentPlan],
    claims: &[TerminalStructuredLinearClaim<EF>],
    batching_scales: &[EF],
    whir_point: &[EF],
    challenger: &mut Ch,
    config: FixedMultiAirLinearizerAdjointTiledConfig,
) -> Result<FixedMultiAirLinearizerAdjointReferenceOutput, FixedMultiAirLinearizerAdjointTraceError>
where
    Ch: AlgebraicChallenger<EF>,
{
    prove_fixed_multi_air_linearizer_adjoint_tiled_factored(
        components,
        claims,
        batching_scales,
        whir_point,
        0,
        challenger,
        config,
    )
}

/// Bounded-memory production prover for the exact vector-alphabet RS dual.
pub fn prove_fixed_multi_air_linearizer_adjoint_tiled_factored<Ch>(
    components: &[FixedMultiAirLinearizerRawComponentPlan],
    claims: &[TerminalStructuredLinearClaim<EF>],
    batching_scales: &[EF],
    whir_point: &[EF],
    initial_folding_factor: usize,
    challenger: &mut Ch,
    config: FixedMultiAirLinearizerAdjointTiledConfig,
) -> Result<FixedMultiAirLinearizerAdjointReferenceOutput, FixedMultiAirLinearizerAdjointTraceError>
where
    Ch: AlgebraicChallenger<EF>,
{
    prove_fixed_multi_air_linearizer_adjoint_tiled_factored_for_layout(
        components,
        claims,
        batching_scales,
        whir_point,
        initial_folding_factor,
        WhirInitialRsLayout::OrdinarySubgroup,
        challenger,
        config,
    )
}

/// Layout-explicit structured dual prover.  Coefficient-native WARP messages
/// are already RS coefficients, so their WHIR Boolean dual is only
/// `zeta^{-T}(weight)`; the ordinary subgroup layout additionally needs the
/// DFT pullback used by the legacy entry point above.
pub fn prove_fixed_multi_air_linearizer_adjoint_tiled_factored_for_layout<Ch>(
    components: &[FixedMultiAirLinearizerRawComponentPlan],
    claims: &[TerminalStructuredLinearClaim<EF>],
    batching_scales: &[EF],
    whir_point: &[EF],
    initial_folding_factor: usize,
    layout: WhirInitialRsLayout,
    challenger: &mut Ch,
    config: FixedMultiAirLinearizerAdjointTiledConfig,
) -> Result<FixedMultiAirLinearizerAdjointReferenceOutput, FixedMultiAirLinearizerAdjointTraceError>
where
    Ch: AlgebraicChallenger<EF>,
{
    let log_message_len = whir_point.len();
    if initial_folding_factor > log_message_len
        || log_message_len - initial_folding_factor > F::TWO_ADICITY
    {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
    }
    let selector_folding_factor =
        structured_selector_folding_factor(layout, log_message_len, initial_folding_factor)?;
    let _plan = fixed_multi_air_linearizer_adjoint_tiled_plan(log_message_len, config)?;
    let prepared = prepare_fixed_multi_air_linearizer_components(
        log_message_len,
        components,
        claims,
        batching_scales,
    )?;
    let evaluation_count = log_message_len + 2;
    let row_log = log_message_len - selector_folding_factor;
    let omega = if row_log == 0 {
        F::ONE
    } else {
        F::two_adic_generator(row_log)
    };
    let mut prefix = Vec::with_capacity(log_message_len);
    let mut rounds = Vec::with_capacity(log_message_len);
    let mut running_claim = None;
    for round in 0..log_message_len {
        let remaining = log_message_len - round - 1;
        let suffix_count = 1usize << remaining;
        let tile_size = 1usize << config.tile_log_size.min(remaining);
        let tile_count = suffix_count.div_ceil(tile_size);
        let coefficients = (0..tile_count)
            .into_par_iter()
            .map(|tile| {
                let start = tile * tile_size;
                let end = (start + tile_size).min(suffix_count);
                let mut sum = [EF::ZERO; FIXED_MULTI_AIR_LINEARIZER_MAX_EVALUATIONS];
                for suffix in start..end {
                    accumulate_fixed_multi_air_linearizer_round_polynomial(
                        &mut sum,
                        &prepared,
                        whir_point,
                        selector_folding_factor,
                        &prefix,
                        round,
                        suffix,
                        omega,
                    );
                }
                sum
            })
            .par_fold_reduce(
                || [EF::ZERO; FIXED_MULTI_AIR_LINEARIZER_MAX_EVALUATIONS],
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        *left += right;
                    }
                    left
                },
                |mut left, right| {
                    for (left, right) in left.iter_mut().zip(right) {
                        *left += right;
                    }
                    left
                },
            );
        let evaluations = (0..evaluation_count)
            .map(|evaluation| {
                let point = EF::from_usize(evaluation);
                coefficients[..evaluation_count]
                    .iter()
                    .rev()
                    .fold(EF::ZERO, |value, &coefficient| value * point + coefficient)
            })
            .collect::<Vec<_>>();
        let pre_claim = evaluations[0] + evaluations[1];
        if running_claim.is_some_and(|claim| claim != pre_claim) {
            return Err(FixedMultiAirLinearizerAdjointTraceError::Claim);
        }
        challenger.observe(EF::from(F::from_u64(LINEARIZER_ADJOINT_ROUND_TAG)));
        challenger.observe(EF::from(F::from_usize(round)));
        challenger.observe(EF::from(F::from_usize(evaluation_count)));
        for &evaluation in &evaluations {
            challenger.observe(evaluation);
        }
        let challenge = challenger.sample();
        running_claim = Some(interpolate(&evaluations, challenge));
        prefix.push(challenge);
        rounds.push(evaluations);
    }
    let initial_claim = rounds[0][0] + rounds[0][1];
    Ok(FixedMultiAirLinearizerAdjointReferenceOutput {
        proof: FixedMultiAirLinearizerAdjointProof {
            round_evaluations: rounds,
        },
        initial_claim,
        point: prefix,
        final_claim: running_claim.ok_or(FixedMultiAirLinearizerAdjointTraceError::Shape)?,
    })
}

fn accumulate_fixed_multi_air_linearizer_round_polynomial(
    output: &mut [EF; FIXED_MULTI_AIR_LINEARIZER_MAX_EVALUATIONS],
    components: &[FixedMultiAirPreparedLinearizerComponent<'_>],
    whir_point: &[EF],
    initial_folding_factor: usize,
    prefix: &[EF],
    round: usize,
    suffix: usize,
    omega: F,
) {
    let log_message_len = whir_point.len();
    let remaining = log_message_len - round - 1;
    let mut point_zero = [EF::ZERO; FIXED_MULTI_AIR_LINEARIZER_MAX_LOG_MESSAGE];
    let mut point_one = [EF::ZERO; FIXED_MULTI_AIR_LINEARIZER_MAX_LOG_MESSAGE];
    point_zero[..round].copy_from_slice(prefix);
    point_one[..round].copy_from_slice(prefix);
    point_one[round] = EF::ONE;
    for coordinate in 0..remaining {
        let bit = (suffix >> (remaining - 1 - coordinate)) & 1;
        let value = EF::from_usize(bit);
        point_zero[round + 1 + coordinate] = value;
        point_one[round + 1 + coordinate] = value;
    }
    let point_zero = &point_zero[..log_message_len];
    let point_one = &point_one[..log_message_len];
    let raw_zero = fixed_multi_air_compact_raw_value(components, point_zero);
    let raw_one = fixed_multi_air_compact_raw_value(components, point_one);
    let raw_slope = raw_one - raw_zero;

    let mut selector = [EF::ZERO; FIXED_MULTI_AIR_LINEARIZER_MAX_EVALUATIONS];
    selector[0] = EF::ONE;
    let mut selector_degree = 0usize;
    for whir_bit in 0..log_message_len {
        let z = whir_point[whir_bit];
        let (factor_zero, factor_slope) = if whir_bit < initial_folding_factor {
            let raw_coordinate = initial_folding_factor - 1 - whir_bit;
            let (y_zero, y_slope) =
                raw_coordinate_affine_value(raw_coordinate, prefix, round, suffix, remaining);
            let slope_scale = z.double() + z - EF::TWO;
            ((EF::ONE - z) + slope_scale * y_zero, slope_scale * y_slope)
        } else {
            let row_log = log_message_len - initial_folding_factor;
            let row_bit = whir_bit - initial_folding_factor;
            let mut y_without_active = EF::ONE;
            let mut active_root = None;
            for row_coordinate in 0..row_log {
                let raw_coordinate = initial_folding_factor + row_coordinate;
                let exponent_power = row_log - 1 - row_coordinate + row_bit;
                let root = if exponent_power >= row_log {
                    F::ONE
                } else {
                    omega.exp_power_of_2(exponent_power)
                };
                if raw_coordinate == round {
                    active_root = Some(EF::from(root));
                } else {
                    let (value, slope) = raw_coordinate_affine_value(
                        raw_coordinate,
                        prefix,
                        round,
                        suffix,
                        remaining,
                    );
                    debug_assert_eq!(slope, EF::ZERO);
                    y_without_active *= (EF::ONE - value) + value * EF::from(root);
                }
            }
            let selector_scale = z.double() - EF::ONE;
            let factor_zero = (EF::ONE - z) + selector_scale * y_without_active;
            let factor_slope = active_root
                .map(|root| selector_scale * y_without_active * (root - EF::ONE))
                .unwrap_or(EF::ZERO);
            (factor_zero, factor_slope)
        };
        for degree in (0..=selector_degree).rev() {
            selector[degree + 1] += selector[degree] * factor_slope;
            selector[degree] *= factor_zero;
        }
        // Keep the setup-fixed degree bound even when this particular factor
        // is constant (or its sampled slope happens to vanish).
        selector_degree += 1;
    }
    for degree in 0..=selector_degree {
        output[degree] += selector[degree] * raw_zero;
        output[degree + 1] += selector[degree] * raw_slope;
    }
}

fn raw_coordinate_affine_value(
    coordinate: usize,
    prefix: &[EF],
    round: usize,
    suffix: usize,
    remaining: usize,
) -> (EF, EF) {
    if coordinate < round {
        (prefix[coordinate], EF::ZERO)
    } else if coordinate == round {
        (EF::ZERO, EF::ONE)
    } else {
        let suffix_coordinate = coordinate - round - 1;
        let bit = (suffix >> (remaining - 1 - suffix_coordinate)) & 1;
        (EF::from_usize(bit), EF::ZERO)
    }
}

fn fixed_multi_air_compact_raw_value(
    components: &[FixedMultiAirPreparedLinearizerComponent<'_>],
    point: &[EF],
) -> EF {
    let log_message_len = point.len();
    components
        .iter()
        .map(|component| {
            if component.is_zero {
                return EF::ZERO;
            }
            let block_len = 1usize << component.log_height;
            if !component.block_start.is_multiple_of(block_len) {
                let rotation = match component.rotation {
                    DirectAirMappedRotation::Current => PrismalinearMappedColumnRotation::Current,
                    DirectAirMappedRotation::Next => PrismalinearMappedColumnRotation::Next,
                };
                return component.scale
                    * fixed_multi_air_packed_column_eq_mle_prevalidated(
                        point,
                        component.block_start,
                        component.row_point,
                        rotation,
                    );
            }
            let high_count = log_message_len - component.log_height;
            let block_tag = component.block_start >> component.log_height;
            let high = (0..high_count)
                .map(|coordinate| {
                    let bit = (block_tag >> (high_count - 1 - coordinate)) & 1;
                    if bit == 0 {
                        EF::ONE - point[coordinate]
                    } else {
                        point[coordinate]
                    }
                })
                .product::<EF>();
            let mut low = if component.rotation == DirectAirMappedRotation::Next {
                [EF::ZERO, EF::ONE]
            } else {
                [EF::ONE, EF::ZERO]
            };
            for low_step in 0..component.log_height {
                let point_coordinate = component.log_height - 1 - low_step;
                let global_coordinate = log_message_len - 1 - low_step;
                let row_coordinate = component.row_point[point_coordinate];
                let aux = point[global_coordinate];
                let mut next = [EF::ZERO; 2];
                for (carry, &carry_weight) in low.iter().enumerate() {
                    for row_bit in 0..2 {
                        let sum = row_bit + carry;
                        let shifted_bit = sum & 1;
                        let next_carry = sum >> 1;
                        let row_weight = if row_bit == 0 {
                            EF::ONE - row_coordinate
                        } else {
                            row_coordinate
                        };
                        let shifted_weight = if shifted_bit == 0 { EF::ONE - aux } else { aux };
                        next[next_carry] += carry_weight * row_weight * shifted_weight;
                    }
                }
                low = next;
            }
            component.scale * high * (low[0] + low[1])
        })
        .sum()
}

/// Exact packed-column Eq MLE after component preparation has checked every
/// dimension and range. This is the same two-carry automaton as
/// `warp_pesat::evaluate_packed_column_eq_mle`: one carry tracks cyclic row
/// rotation and the other tracks addition of the arbitrary block start.
fn fixed_multi_air_packed_column_eq_mle_prevalidated(
    global_point: &[EF],
    start: usize,
    local_point: &[EF],
    rotation: PrismalinearMappedColumnRotation,
) -> EF {
    let n = global_point.len();
    let k = local_point.len();
    let mut state = [[EF::ZERO; 2]; 2];
    state[rotation.offset()][0] = EF::ONE;
    for bit in 0..n {
        let mut next = [[EF::ZERO; 2]; 2];
        for carry_rotation in 0..=1usize {
            for carry_start in 0..=1usize {
                let prefix = state[carry_rotation][carry_start];
                if prefix == EF::ZERO {
                    continue;
                }
                let choices = if bit < k { 2 } else { 1 };
                for local_bit in 0..choices {
                    let (source_bit, next_rotation) = if bit < k {
                        let sum = local_bit + carry_rotation;
                        (sum & 1, sum >> 1)
                    } else {
                        (0, 0)
                    };
                    let sum = source_bit + ((start >> bit) & 1) + carry_start;
                    let global_bit = sum & 1;
                    let next_start = sum >> 1;
                    let local_weight = if bit < k {
                        let coordinate = local_point[k - 1 - bit];
                        if local_bit == 1 {
                            coordinate
                        } else {
                            EF::ONE - coordinate
                        }
                    } else {
                        EF::ONE
                    };
                    let coordinate = global_point[n - 1 - bit];
                    let global_weight = if global_bit == 1 {
                        coordinate
                    } else {
                        EF::ONE - coordinate
                    };
                    next[next_rotation][next_start] += prefix * local_weight * global_weight;
                }
            }
        }
        state = next;
    }
    state[0][0] + state[1][0]
}

/// Exponential reference prover for the auxiliary structured RS-dual
/// sumcheck. Production must use the same polynomial through a tiled CPU/CUDA
/// prover; this routine is intentionally capped so it can never materialize a
/// production-size Boolean cube by accident.
pub fn prove_fixed_multi_air_linearizer_adjoint_reference<Ch>(
    claims: &[TerminalStructuredLinearClaim<EF>],
    batching_scales: &[EF],
    whir_point: &[EF],
    challenger: &mut Ch,
) -> Result<FixedMultiAirLinearizerAdjointReferenceOutput, FixedMultiAirLinearizerAdjointTraceError>
where
    Ch: AlgebraicChallenger<EF>,
{
    prove_fixed_multi_air_linearizer_adjoint_reference_factored(
        claims,
        batching_scales,
        whir_point,
        0,
        challenger,
    )
}

/// Exponential correctness oracle for the vector-alphabet initial RS layout.
/// The first `initial_folding_factor` WHIR variables select columns, while the
/// remaining variables select independently DFT-encoded row coefficients.
pub fn prove_fixed_multi_air_linearizer_adjoint_reference_factored<Ch>(
    claims: &[TerminalStructuredLinearClaim<EF>],
    batching_scales: &[EF],
    whir_point: &[EF],
    initial_folding_factor: usize,
    challenger: &mut Ch,
) -> Result<FixedMultiAirLinearizerAdjointReferenceOutput, FixedMultiAirLinearizerAdjointTraceError>
where
    Ch: AlgebraicChallenger<EF>,
{
    prove_fixed_multi_air_linearizer_adjoint_reference_factored_for_layout(
        claims,
        batching_scales,
        whir_point,
        initial_folding_factor,
        WhirInitialRsLayout::OrdinarySubgroup,
        challenger,
    )
}

pub fn prove_fixed_multi_air_linearizer_adjoint_reference_factored_for_layout<Ch>(
    claims: &[TerminalStructuredLinearClaim<EF>],
    batching_scales: &[EF],
    whir_point: &[EF],
    initial_folding_factor: usize,
    layout: WhirInitialRsLayout,
    challenger: &mut Ch,
) -> Result<FixedMultiAirLinearizerAdjointReferenceOutput, FixedMultiAirLinearizerAdjointTraceError>
where
    Ch: AlgebraicChallenger<EF>,
{
    const MAX_REFERENCE_LOG_MESSAGE: usize = 12;
    let log_message_len = whir_point.len();
    let selector_folding_factor =
        structured_selector_folding_factor(layout, log_message_len, initial_folding_factor)?;
    if log_message_len == 0
        || log_message_len > MAX_REFERENCE_LOG_MESSAGE
        || initial_folding_factor > log_message_len
        || claims.is_empty()
        || claims.len() != batching_scales.len()
        || claims
            .iter()
            .any(|claim| claim.weight.len() != (1usize << log_message_len))
    {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
    }
    let dense_weights = claims
        .iter()
        .map(|claim| claim.weight.materialize())
        .collect::<Vec<_>>();
    let evaluation_count = log_message_len + 2;
    let mut prefix = Vec::with_capacity(log_message_len);
    let mut rounds = Vec::with_capacity(log_message_len);
    let mut running_claim = None;
    for round in 0..log_message_len {
        let remaining = log_message_len - round - 1;
        let mut evaluations = Vec::with_capacity(evaluation_count);
        for evaluation in 0..evaluation_count {
            let mut sum = EF::ZERO;
            for suffix in 0..(1usize << remaining) {
                let mut point = prefix.clone();
                point.push(EF::from_usize(evaluation));
                point.extend((0..remaining).map(|coordinate| {
                    EF::from_usize((suffix >> (remaining - 1 - coordinate)) & 1)
                }));
                sum += fixed_multi_air_linearizer_adjoint_summand(
                    &dense_weights,
                    batching_scales,
                    whir_point,
                    &point,
                    selector_folding_factor,
                )?;
            }
            evaluations.push(sum);
        }
        let pre_claim = evaluations[0] + evaluations[1];
        if running_claim.is_some_and(|claim| claim != pre_claim) {
            return Err(FixedMultiAirLinearizerAdjointTraceError::Claim);
        }
        challenger.observe(EF::from(F::from_u64(LINEARIZER_ADJOINT_ROUND_TAG)));
        challenger.observe(EF::from(F::from_usize(round)));
        challenger.observe(EF::from(F::from_usize(evaluation_count)));
        for &evaluation in &evaluations {
            challenger.observe(evaluation);
        }
        let challenge = challenger.sample();
        let post_claim = interpolate(&evaluations, challenge);
        running_claim = Some(post_claim);
        prefix.push(challenge);
        rounds.push(evaluations);
    }
    let initial_claim = rounds[0][0] + rounds[0][1];
    Ok(FixedMultiAirLinearizerAdjointReferenceOutput {
        proof: FixedMultiAirLinearizerAdjointProof {
            round_evaluations: rounds,
        },
        initial_claim,
        point: prefix,
        final_claim: running_claim.ok_or(FixedMultiAirLinearizerAdjointTraceError::Shape)?,
    })
}

fn fixed_multi_air_linearizer_adjoint_summand(
    dense_weights: &[Vec<EF>],
    batching_scales: &[EF],
    whir_point: &[EF],
    point: &[EF],
    initial_folding_factor: usize,
) -> Result<EF, FixedMultiAirLinearizerAdjointTraceError> {
    if dense_weights.len() != batching_scales.len()
        || whir_point.len() != point.len()
        || initial_folding_factor > point.len()
        || dense_weights
            .iter()
            .any(|weight| weight.len() != (1usize << point.len()))
    {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
    }
    let raw = dense_weights
        .iter()
        .zip(batching_scales)
        .map(|(weight, &scale)| scale * evaluate_mle(weight, point))
        .sum::<EF>();
    let selector = factored_linearizer_selector(point, whir_point, initial_folding_factor)?;
    Ok(raw * selector)
}

pub(super) fn factored_linearizer_selector(
    raw_point: &[EF],
    whir_point: &[EF],
    initial_folding_factor: usize,
) -> Result<EF, FixedMultiAirLinearizerAdjointTraceError> {
    let log_message_len = raw_point.len();
    if whir_point.len() != log_message_len
        || initial_folding_factor > log_message_len
        || log_message_len - initial_folding_factor > F::TWO_ADICITY
    {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
    }
    let row_log = log_message_len - initial_folding_factor;
    let mut selector = EF::ONE;

    // Row-major WHIR flattening places the column index in the low bits.
    // WHIR folds low bits first, while structured raw points are MSB-first.
    for column_bit in 0..initial_folding_factor {
        let y = raw_point[initial_folding_factor - 1 - column_bit];
        let z = whir_point[column_bit];
        // No DFT is applied across columns.  Select the exact inverse-zeta
        // dual coefficient for the Boolean column bit: `1-z` at zero and
        // `2z-1` at one, then multilinearly extend in the raw bit `y`.
        selector *= (EF::ONE - y) * (EF::ONE - z) + y * (z.double() - EF::ONE);
    }

    if row_log != 0 {
        let omega = F::two_adic_generator(row_log);
        for row_bit in 0..row_log {
            let y = raw_point[initial_folding_factor..]
                .iter()
                .enumerate()
                .map(|(coordinate, &coordinate_value)| {
                    let exponent_power = row_log - 1 - coordinate + row_bit;
                    let root = if exponent_power >= row_log {
                        F::ONE
                    } else {
                        omega.exp_power_of_2(exponent_power)
                    };
                    (EF::ONE - coordinate_value) + coordinate_value * EF::from(root)
                })
                .product::<EF>();
            let z = whir_point[initial_folding_factor + row_bit];
            selector *= (EF::ONE - z) + (z.double() - EF::ONE) * y;
        }
    }
    Ok(selector)
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirLinearizerSumcheckScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub is_first_round: T,
    pub is_last_round: T,
    pub is_first_evaluation: T,
    pub is_last_evaluation: T,
    pub round: T,
    pub evaluation: T,
    pub evaluation_flags: [T; FIXED_MULTI_AIR_LINEARIZER_MAX_EVALUATIONS],
    pub denominator_inverse: T,
    pub point_lookup_count: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirLinearizerSumcheckCols<T> {
    pub tidx: T,
    pub initial_claim: [T; D_EF],
    pub pre_claim: [T; D_EF],
    pub at_zero: [T; D_EF],
    pub at_one: [T; D_EF],
    pub evaluation: [T; D_EF],
    pub challenge: [T; D_EF],
    pub basis_prefix: [[T; D_EF]; FIXED_MULTI_AIR_LINEARIZER_MAX_EVALUATIONS + 1],
    pub term: [T; D_EF],
    pub sum_before: [T; D_EF],
    pub sum_after: [T; D_EF],
    pub post_claim: [T; D_EF],
}

/// Bounded-width verifier for the auxiliary degree-(m+1) structured RS-dual
/// sumcheck. The width is capped by the setup-supported m<=28, never by 2^m.
pub struct FixedMultiAirLinearizerSumcheckAir {
    pub transcript_bus: TranscriptBus,
    pub point_bus: FixedMultiAirLinearizerAuxPointBus,
    pub final_bus: FixedMultiAirLinearizerSumcheckFinalBus,
    pub log_message_len: usize,
}

impl FixedMultiAirLinearizerSumcheckAir {
    #[must_use]
    pub const fn evaluation_count(&self) -> usize {
        self.log_message_len + 2
    }
}

impl BaseAirWithPublicValues<F> for FixedMultiAirLinearizerSumcheckAir {}
impl PartitionedBaseAir<F> for FixedMultiAirLinearizerSumcheckAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirLinearizerSumcheckScheduleCols::<F>::width()]
    }
    fn common_main_width(&self) -> usize {
        FixedMultiAirLinearizerSumcheckCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirLinearizerSumcheckAir {}
impl BaseAir<F> for FixedMultiAirLinearizerSumcheckAir {
    fn width(&self) -> usize {
        FixedMultiAirLinearizerSumcheckScheduleCols::<F>::width()
            + FixedMultiAirLinearizerSumcheckCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirLinearizerSumcheckAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("linearizer sumcheck schedule")
            .to_vec();
        let next_cached = builder.cached_mains()[0]
            .row_slice(1)
            .expect("next linearizer sumcheck schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("linearizer sumcheck row")
            .to_vec();
        let next_common = builder
            .common_main()
            .row_slice(1)
            .expect("next linearizer sumcheck row")
            .to_vec();
        let schedule: &FixedMultiAirLinearizerSumcheckScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let next_schedule: &FixedMultiAirLinearizerSumcheckScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FixedMultiAirLinearizerSumcheckCols<AB::Var> = common.as_slice().borrow();
        let next: &FixedMultiAirLinearizerSumcheckCols<AB::Var> = next_common.as_slice().borrow();
        let evaluation_count = self.evaluation_count();
        for flag in [
            schedule.active,
            schedule.is_first,
            schedule.is_last,
            schedule.is_first_round,
            schedule.is_last_round,
            schedule.is_first_evaluation,
            schedule.is_last_evaluation,
        ]
        .into_iter()
        .chain(schedule.evaluation_flags)
        {
            builder.assert_bool(flag);
        }
        let flag_sum = schedule
            .evaluation_flags
            .iter()
            .fold(AB::Expr::ZERO, |sum, &flag| sum + flag);
        builder.assert_eq(flag_sum, schedule.active);
        for flag in &schedule.evaluation_flags[evaluation_count..] {
            builder.assert_zero(*flag);
        }
        let selected_index = schedule
            .evaluation_flags
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |sum, (index, &flag)| {
                sum + flag * AB::Expr::from_usize(index)
            });
        builder
            .when(schedule.active)
            .assert_eq(schedule.evaluation, selected_index);
        builder.when_first_row().assert_one(schedule.active);
        builder.when_first_row().assert_one(schedule.is_first);
        builder.when_first_row().assert_one(schedule.is_first_round);
        builder.when_first_row().assert_zero(schedule.round);
        builder
            .when(schedule.active * schedule.is_last_round)
            .assert_eq(
                schedule.round,
                AB::Expr::from_usize(self.log_message_len - 1),
            );
        builder
            .when_transition()
            .assert_bool(schedule.active - next_schedule.active);
        builder
            .when_transition()
            .assert_eq(schedule.active - next_schedule.active, schedule.is_last);
        builder
            .when_last_row()
            .assert_eq(schedule.is_last, schedule.active);

        let same_round = next_schedule.active * (AB::Expr::ONE - next_schedule.is_first_evaluation);
        let next_round = next_schedule.active * next_schedule.is_first_evaluation;
        let mut transition = builder.when_transition();
        let mut same = transition.when(same_round);
        same.assert_eq(next_schedule.round, schedule.round);
        same.assert_eq(
            next_schedule.evaluation,
            schedule.evaluation + AB::Expr::ONE,
        );
        same.assert_eq(next.tidx, local.tidx);
        assert_array_eq(&mut same, next.initial_claim, local.initial_claim);
        assert_array_eq(&mut same, next.pre_claim, local.pre_claim);
        assert_array_eq(&mut same, next.at_zero, local.at_zero);
        assert_array_eq(&mut same, next.at_one, local.at_one);
        assert_array_eq(&mut same, next.challenge, local.challenge);
        assert_array_eq(&mut same, next.sum_before, local.sum_after);
        assert_array_eq(&mut same, next.post_claim, local.post_claim);
        let mut transition = builder.when_transition();
        let mut advance = transition.when(next_round);
        advance.assert_eq(next_schedule.round, schedule.round + AB::Expr::ONE);
        advance.assert_eq(
            next.tidx,
            local.tidx + AB::Expr::from_usize((3 + evaluation_count + 1) * D_EF),
        );
        advance.assert_zero(next_schedule.is_first_round);
        advance.assert_zero(schedule.is_last_round);
        assert_array_eq(&mut advance, next.initial_claim, local.initial_claim);
        assert_array_eq(&mut advance, next.pre_claim, local.post_claim);
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.pre_claim,
            ext_field_add::<AB::Expr>(local.at_zero, local.at_one),
        );
        assert_array_eq(
            &mut builder.when(schedule.is_first),
            local.pre_claim,
            local.initial_claim.map(Into::into),
        );
        let one = one_ext::<AB>();
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.basis_prefix[0],
            one,
        );
        for other in 0..evaluation_count {
            let skip = schedule.evaluation_flags[other];
            let factor = core::array::from_fn(|limb| {
                if limb == 0 {
                    AB::Expr::from(skip)
                        + (AB::Expr::ONE - AB::Expr::from(skip))
                            * (AB::Expr::from(local.challenge[limb]) - AB::Expr::from_usize(other))
                } else {
                    (AB::Expr::ONE - AB::Expr::from(skip)) * AB::Expr::from(local.challenge[limb])
                }
            });
            assert_array_eq(
                &mut builder.when(schedule.active),
                local.basis_prefix[other + 1],
                ext_field_multiply::<AB::Expr>(local.basis_prefix[other], factor),
            );
        }
        for unused in evaluation_count + 1..=FIXED_MULTI_AIR_LINEARIZER_MAX_EVALUATIONS {
            assert_array_eq(
                &mut builder.when(schedule.active),
                local.basis_prefix[unused],
                local.basis_prefix[evaluation_count].map(Into::into),
            );
        }
        let scaled_basis = local.basis_prefix[evaluation_count]
            .map(|limb| AB::Expr::from(limb) * AB::Expr::from(schedule.denominator_inverse));
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.term,
            ext_field_multiply::<AB::Expr>(local.evaluation, scaled_basis),
        );
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.sum_after,
            ext_field_add::<AB::Expr>(local.sum_before, local.term),
        );
        assert_array_eq(
            &mut builder.when(schedule.is_first_evaluation),
            local.sum_before,
            [AB::Expr::ZERO; D_EF],
        );
        assert_array_eq(
            &mut builder.when(schedule.is_last_evaluation),
            local.sum_after,
            local.post_claim.map(Into::into),
        );
        let first_evaluation = schedule.active * schedule.is_first_evaluation;
        observe_const(
            &self.transcript_bus,
            builder,
            local.tidx.into(),
            LINEARIZER_ADJOINT_ROUND_TAG,
            first_evaluation,
        );
        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.tidx) + AB::Expr::from_usize(D_EF),
            ext_from_base::<AB>(schedule.round.into()),
            schedule.active * schedule.is_first_evaluation,
        );
        observe_const(
            &self.transcript_bus,
            builder,
            AB::Expr::from(local.tidx) + AB::Expr::from_usize(2 * D_EF),
            evaluation_count as u64,
            schedule.active * schedule.is_first_evaluation,
        );
        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.tidx)
                + AB::Expr::from_usize(3 * D_EF)
                + AB::Expr::from(schedule.evaluation) * AB::Expr::from_usize(D_EF),
            local.evaluation,
            schedule.active,
        );
        self.transcript_bus.sample_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.tidx) + AB::Expr::from_usize((3 + evaluation_count) * D_EF),
            local.challenge,
            schedule.active * schedule.is_last_evaluation,
        );
        self.point_bus.add_key_with_lookups(
            builder,
            FixedMultiAirLinearizerAuxPointMessage {
                coordinate: schedule.round.into(),
                value: local.challenge.map(Into::into),
            },
            schedule.point_lookup_count,
        );
        self.final_bus.send(
            builder,
            FixedMultiAirLinearizerSumcheckFinalMessage {
                initial_claim: local.initial_claim.map(Into::into),
                final_claim: local.post_claim.map(Into::into),
            },
            schedule.is_last,
        );
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirLinearizerSumcheckTraceOutput {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub point: Vec<EF>,
    pub initial_claim: EF,
    pub final_claim: EF,
    pub end_tidx: usize,
}

pub fn generate_fixed_multi_air_linearizer_sumcheck_traces(
    air: &FixedMultiAirLinearizerSumcheckAir,
    proof: &FixedMultiAirLinearizerAdjointProof,
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
    point_lookup_counts: &[usize],
    required_height: Option<usize>,
) -> Result<FixedMultiAirLinearizerSumcheckTraceOutput, FixedMultiAirLinearizerAdjointTraceError> {
    let evaluation_count = air.evaluation_count();
    if air.log_message_len == 0
        || air.log_message_len > FIXED_MULTI_AIR_LINEARIZER_MAX_LOG_MESSAGE
        || proof.round_evaluations.len() != air.log_message_len
        || point_lookup_counts.len() != air.log_message_len
        || proof
            .round_evaluations
            .iter()
            .any(|evaluations| evaluations.len() != evaluation_count)
    {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
    }
    let valid_rows = air
        .log_message_len
        .checked_mul(evaluation_count)
        .ok_or(FixedMultiAirLinearizerAdjointTraceError::Shape)?;
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
    }
    let cached_width = FixedMultiAirLinearizerSumcheckScheduleCols::<F>::width();
    let common_width = FixedMultiAirLinearizerSumcheckCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    let mut tidx = start_tidx;
    let mut pre_claim = proof.round_evaluations[0][0] + proof.round_evaluations[0][1];
    let initial_claim = pre_claim;
    let mut point = Vec::with_capacity(air.log_message_len);
    let mut row = 0usize;
    for (round, evaluations) in proof.round_evaluations.iter().enumerate() {
        expect_ext(
            transcript,
            tidx,
            EF::from_u64(LINEARIZER_ADJOINT_ROUND_TAG),
            false,
        )?;
        expect_ext(transcript, tidx + D_EF, EF::from_usize(round), false)?;
        expect_ext(
            transcript,
            tidx + 2 * D_EF,
            EF::from_usize(evaluation_count),
            false,
        )?;
        for (index, &evaluation) in evaluations.iter().enumerate() {
            expect_ext(transcript, tidx + (3 + index) * D_EF, evaluation, false)?;
        }
        if evaluations[0] + evaluations[1] != pre_claim {
            return Err(FixedMultiAirLinearizerAdjointTraceError::Claim);
        }
        let challenge = read_ext(transcript, tidx + (3 + evaluation_count) * D_EF, true)?;
        let post_claim = interpolate(evaluations, challenge);
        let mut sum = EF::ZERO;
        for (evaluation_index, &evaluation) in evaluations.iter().enumerate() {
            let mut basis = EF::ONE;
            let mut prefixes = [EF::ONE; FIXED_MULTI_AIR_LINEARIZER_MAX_EVALUATIONS + 1];
            for other in 0..evaluation_count {
                if other != evaluation_index {
                    basis *= challenge - EF::from_usize(other);
                }
                prefixes[other + 1] = basis;
            }
            for unused in evaluation_count + 1..=FIXED_MULTI_AIR_LINEARIZER_MAX_EVALUATIONS {
                prefixes[unused] = prefixes[evaluation_count];
            }
            let denominator_inverse =
                lagrange_denominator(evaluation_count, evaluation_index).inverse();
            let term = evaluation * basis * EF::from(denominator_inverse);
            let before = sum;
            sum += term;
            let cached_row = &mut cached[row * cached_width..(row + 1) * cached_width];
            let schedule: &mut FixedMultiAirLinearizerSumcheckScheduleCols<F> =
                cached_row.borrow_mut();
            schedule.active = F::ONE;
            schedule.is_first = F::from_bool(row == 0);
            schedule.is_last = F::from_bool(row + 1 == valid_rows);
            schedule.is_first_round = F::from_bool(round == 0);
            schedule.is_last_round = F::from_bool(round + 1 == air.log_message_len);
            schedule.is_first_evaluation = F::from_bool(evaluation_index == 0);
            schedule.is_last_evaluation = F::from_bool(evaluation_index + 1 == evaluation_count);
            schedule.round = F::from_usize(round);
            schedule.evaluation = F::from_usize(evaluation_index);
            schedule.evaluation_flags[evaluation_index] = F::ONE;
            schedule.denominator_inverse = denominator_inverse;
            schedule.point_lookup_count = if evaluation_index + 1 == evaluation_count {
                F::from_usize(point_lookup_counts[round])
            } else {
                F::ZERO
            };
            let common_row = &mut common[row * common_width..(row + 1) * common_width];
            let cols: &mut FixedMultiAirLinearizerSumcheckCols<F> = common_row.borrow_mut();
            cols.tidx = F::from_usize(tidx);
            copy_ext(&mut cols.initial_claim, initial_claim);
            copy_ext(&mut cols.pre_claim, pre_claim);
            copy_ext(&mut cols.at_zero, evaluations[0]);
            copy_ext(&mut cols.at_one, evaluations[1]);
            copy_ext(&mut cols.evaluation, evaluation);
            copy_ext(&mut cols.challenge, challenge);
            for (target, value) in cols.basis_prefix.iter_mut().zip(prefixes) {
                copy_ext(target, value);
            }
            copy_ext(&mut cols.term, term);
            copy_ext(&mut cols.sum_before, before);
            copy_ext(&mut cols.sum_after, sum);
            copy_ext(&mut cols.post_claim, post_claim);
            row += 1;
        }
        point.push(challenge);
        pre_claim = post_claim;
        tidx += (3 + evaluation_count + 1) * D_EF;
    }
    Ok(FixedMultiAirLinearizerSumcheckTraceOutput {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        point,
        initial_claim,
        final_claim: pre_claim,
        end_tidx: tidx,
    })
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirLinearizerYScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub is_first_in_bit: T,
    pub is_last_in_bit: T,
    pub message_bit: T,
    pub coordinate: T,
    pub root: T,
    pub is_column: T,
    pub use_identity: T,
    pub use_dft: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirLinearizerYCols<T> {
    pub point: [T; D_EF],
    pub whir_point: [T; D_EF],
    pub factor: [T; D_EF],
    pub product_before: [T; D_EF],
    pub product_after: [T; D_EF],
    pub selector_factor: [T; D_EF],
}

/// Computes the exact DFT root products used by
/// `zeta^{-T}(DFT(raw_weight))` at the terminal WHIR point.
pub struct FixedMultiAirLinearizerYAir {
    pub point_bus: FixedMultiAirLinearizerAuxPointBus,
    pub whir_point_bus: NativeTerminalWhirPointBus,
    pub y_bus: FixedMultiAirLinearizerYBus,
    pub log_message_len: usize,
    pub initial_folding_factor: usize,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirLinearizerYAir {}
impl PartitionedBaseAir<F> for FixedMultiAirLinearizerYAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirLinearizerYScheduleCols::<F>::width()]
    }
    fn common_main_width(&self) -> usize {
        FixedMultiAirLinearizerYCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirLinearizerYAir {}
impl BaseAir<F> for FixedMultiAirLinearizerYAir {
    fn width(&self) -> usize {
        FixedMultiAirLinearizerYScheduleCols::<F>::width()
            + FixedMultiAirLinearizerYCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirLinearizerYAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("linearizer Y schedule")
            .to_vec();
        let next_cached = builder.cached_mains()[0]
            .row_slice(1)
            .expect("next linearizer Y schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("linearizer Y row")
            .to_vec();
        let next_common = builder
            .common_main()
            .row_slice(1)
            .expect("next linearizer Y row")
            .to_vec();
        let schedule: &FixedMultiAirLinearizerYScheduleCols<AB::Var> = cached.as_slice().borrow();
        let next_schedule: &FixedMultiAirLinearizerYScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FixedMultiAirLinearizerYCols<AB::Var> = common.as_slice().borrow();
        let next: &FixedMultiAirLinearizerYCols<AB::Var> = next_common.as_slice().borrow();
        for flag in [
            schedule.active,
            schedule.is_first,
            schedule.is_last,
            schedule.is_first_in_bit,
            schedule.is_last_in_bit,
            schedule.is_column,
            schedule.use_identity,
            schedule.use_dft,
        ] {
            builder.assert_bool(flag);
        }
        builder.assert_zero(schedule.use_identity * schedule.use_dft);
        builder.assert_zero(schedule.use_identity * (AB::Expr::ONE - schedule.is_column));
        builder.assert_zero(schedule.use_dft * schedule.is_column);
        builder.when_first_row().assert_one(schedule.active);
        builder.when_first_row().assert_one(schedule.is_first);
        builder
            .when_first_row()
            .assert_one(schedule.is_first_in_bit);
        builder
            .when_transition()
            .assert_bool(schedule.active - next_schedule.active);
        builder
            .when_transition()
            .assert_eq(schedule.active - next_schedule.active, schedule.is_last);
        builder
            .when_last_row()
            .assert_eq(schedule.is_last, schedule.active);
        let same_bit = next_schedule.active * (AB::Expr::ONE - next_schedule.is_first_in_bit);
        let next_bit = next_schedule.active * next_schedule.is_first_in_bit;
        let mut transition = builder.when_transition();
        let mut same = transition.when(same_bit);
        same.assert_eq(next_schedule.message_bit, schedule.message_bit);
        same.assert_eq(
            next_schedule.coordinate,
            schedule.coordinate + AB::Expr::ONE,
        );
        assert_array_eq(&mut same, next.whir_point, local.whir_point);
        assert_array_eq(&mut same, next.product_before, local.product_after);
        let mut transition = builder.when_transition();
        let mut advance = transition.when(next_bit);
        advance.assert_eq(
            next_schedule.message_bit,
            schedule.message_bit + AB::Expr::ONE,
        );
        advance.assert_zero(next_schedule.coordinate);
        assert_array_eq(&mut advance, next.product_before, one_ext::<AB>());
        assert_array_eq(
            &mut builder.when(schedule.is_first_in_bit),
            local.product_before,
            one_ext::<AB>(),
        );
        let root_point = local
            .point
            .map(|limb| AB::Expr::from(limb) * AB::Expr::from(schedule.root));
        let dft_factor: [AB::Expr; D_EF] = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE - AB::Expr::from(local.point[limb]) + root_point[limb].clone()
            } else {
                -AB::Expr::from(local.point[limb]) + root_point[limb].clone()
            }
        });
        let one = one_ext::<AB>();
        let factor = core::array::from_fn(|limb| {
            AB::Expr::from(one[limb].clone())
                + AB::Expr::from(schedule.use_identity)
                    * (AB::Expr::from(local.point[limb]) - one[limb].clone())
                + AB::Expr::from(schedule.use_dft) * (dft_factor[limb].clone() - one[limb].clone())
        });
        assert_array_eq(&mut builder.when(schedule.active), local.factor, factor);
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.product_after,
            ext_field_multiply::<AB::Expr>(local.product_before, local.factor),
        );
        self.point_bus.lookup_key(
            builder,
            FixedMultiAirLinearizerAuxPointMessage {
                coordinate: schedule.coordinate.into(),
                value: local.point.map(Into::into),
            },
            schedule.active * (schedule.use_identity + schedule.use_dft),
        );
        let y = local.product_after;
        let z = local.whir_point;
        let row_factor = ext_field_add::<AB::Expr>(
            ext_field_subtract::<AB::Expr>(one_ext::<AB>(), z),
            ext_field_multiply::<AB::Expr>(
                ext_field_subtract::<AB::Expr>(ext_field_add::<AB::Expr>(z, z), one_ext::<AB>()),
                y,
            ),
        );
        let column_factor = ext_field_add::<AB::Expr>(
            ext_field_multiply::<AB::Expr>(
                ext_field_subtract::<AB::Expr>(one_ext::<AB>(), y),
                ext_field_subtract::<AB::Expr>(one_ext::<AB>(), z),
            ),
            ext_field_multiply::<AB::Expr>(
                y,
                ext_field_subtract::<AB::Expr>(ext_field_add::<AB::Expr>(z, z), one_ext::<AB>()),
            ),
        );
        let selector_factor = core::array::from_fn(|limb| {
            schedule.is_column * column_factor[limb].clone()
                + (AB::Expr::ONE - schedule.is_column) * row_factor[limb].clone()
        });
        assert_array_eq(
            &mut builder.when(schedule.is_last_in_bit),
            local.selector_factor,
            selector_factor,
        );
        self.whir_point_bus.lookup_key(
            builder,
            NativeTerminalWhirPointMessage {
                coordinate: schedule.message_bit.into(),
                value: local.whir_point.map(Into::into),
            },
            schedule.is_last_in_bit,
        );
        self.y_bus.send(
            builder,
            FixedMultiAirLinearizerYMessage {
                message_bit: schedule.message_bit.into(),
                value: local.selector_factor.map(Into::into),
            },
            schedule.is_last_in_bit,
        );
    }
}

pub fn generate_fixed_multi_air_linearizer_y_traces(
    point: &[EF],
    whir_point: &[EF],
    selector_folding_factor: usize,
    required_height: Option<usize>,
) -> Result<(RowMajorMatrix<F>, RowMajorMatrix<F>, Vec<EF>), FixedMultiAirLinearizerAdjointTraceError>
{
    let log_message_len = point.len();
    if log_message_len == 0
        || log_message_len > FIXED_MULTI_AIR_LINEARIZER_MAX_LOG_MESSAGE
        || whir_point.len() != log_message_len
        || selector_folding_factor > log_message_len
        || log_message_len - selector_folding_factor > F::TWO_ADICITY
    {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
    }
    let valid_rows = log_message_len
        .checked_mul(log_message_len)
        .ok_or(FixedMultiAirLinearizerAdjointTraceError::Shape)?;
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
    }
    let cached_width = FixedMultiAirLinearizerYScheduleCols::<F>::width();
    let common_width = FixedMultiAirLinearizerYCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    let row_log = log_message_len - selector_folding_factor;
    let omega = if row_log == 0 {
        F::ONE
    } else {
        F::two_adic_generator(row_log)
    };
    let mut selector_factors = Vec::with_capacity(log_message_len);
    let mut row = 0usize;
    for message_bit in 0..log_message_len {
        let is_column = message_bit < selector_folding_factor;
        let mut product = EF::ONE;
        for (coordinate, &point_value) in point.iter().enumerate() {
            let use_identity = is_column && coordinate == selector_folding_factor - 1 - message_bit;
            let use_dft = !is_column && coordinate >= selector_folding_factor;
            let root = if use_dft {
                let row_coordinate = coordinate - selector_folding_factor;
                let row_bit = message_bit - selector_folding_factor;
                let exponent_power = row_log - 1 - row_coordinate + row_bit;
                if exponent_power >= row_log {
                    F::ONE
                } else {
                    omega.exp_power_of_2(exponent_power)
                }
            } else {
                F::ONE
            };
            let factor = if use_identity {
                point_value
            } else if use_dft {
                (EF::ONE - point_value) + point_value * EF::from(root)
            } else {
                EF::ONE
            };
            let before = product;
            product *= factor;
            let cached_row = &mut cached[row * cached_width..(row + 1) * cached_width];
            let schedule: &mut FixedMultiAirLinearizerYScheduleCols<F> = cached_row.borrow_mut();
            schedule.active = F::ONE;
            schedule.is_first = F::from_bool(row == 0);
            schedule.is_last = F::from_bool(row + 1 == valid_rows);
            schedule.is_first_in_bit = F::from_bool(coordinate == 0);
            schedule.is_last_in_bit = F::from_bool(coordinate + 1 == log_message_len);
            schedule.message_bit = F::from_usize(message_bit);
            schedule.coordinate = F::from_usize(coordinate);
            schedule.root = root;
            schedule.is_column = F::from_bool(is_column);
            schedule.use_identity = F::from_bool(use_identity);
            schedule.use_dft = F::from_bool(use_dft);
            let common_row = &mut common[row * common_width..(row + 1) * common_width];
            let cols: &mut FixedMultiAirLinearizerYCols<F> = common_row.borrow_mut();
            copy_ext(&mut cols.point, point_value);
            copy_ext(&mut cols.whir_point, whir_point[message_bit]);
            copy_ext(&mut cols.factor, factor);
            copy_ext(&mut cols.product_before, before);
            copy_ext(&mut cols.product_after, product);
            let selector_factor = if is_column {
                (EF::ONE - product) * (EF::ONE - whir_point[message_bit])
                    + product * (whir_point[message_bit].double() - EF::ONE)
            } else {
                (EF::ONE - whir_point[message_bit])
                    + (whir_point[message_bit].double() - EF::ONE) * product
            };
            copy_ext(&mut cols.selector_factor, selector_factor);
            row += 1;
        }
        selector_factors.push(if is_column {
            (EF::ONE - product) * (EF::ONE - whir_point[message_bit])
                + product * (whir_point[message_bit].double() - EF::ONE)
        } else {
            (EF::ONE - whir_point[message_bit])
                + (whir_point[message_bit].double() - EF::ONE) * product
        });
    }
    Ok((
        RowMajorMatrix::new(cached, cached_width),
        RowMajorMatrix::new(common, common_width),
        selector_factors,
    ))
}

/// Layout-safe selector trace entry point. In particular, coefficient-native
/// two-coset messages use all `log_message_len` selector coordinates even
/// though the WHIR round folding parameter `k` is smaller. Callers must not
/// substitute WHIR `k` for this derived selector dimension.
pub fn generate_fixed_multi_air_linearizer_y_traces_for_layout(
    point: &[EF],
    whir_point: &[EF],
    initial_folding_factor: usize,
    layout: WhirInitialRsLayout,
    required_height: Option<usize>,
) -> Result<(RowMajorMatrix<F>, RowMajorMatrix<F>, Vec<EF>), FixedMultiAirLinearizerAdjointTraceError>
{
    let selector_folding_factor =
        structured_selector_folding_factor(layout, point.len(), initial_folding_factor)?;
    generate_fixed_multi_air_linearizer_y_traces(
        point,
        whir_point,
        selector_folding_factor,
        required_height,
    )
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirLinearizerSelectorCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub message_bit: T,
    pub y: [T; D_EF],
    pub factor: [T; D_EF],
    pub product_before: [T; D_EF],
    pub product_after: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(FixedMultiAirLinearizerSelectorCols<u8>)]
pub struct FixedMultiAirLinearizerSelectorAir {
    pub y_bus: FixedMultiAirLinearizerYBus,
    pub selector_bus: FixedMultiAirLinearizerSelectorBus,
    pub log_message_len: usize,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirLinearizerSelectorAir {}
impl PartitionedBaseAir<F> for FixedMultiAirLinearizerSelectorAir {}
impl BaseAir<F> for FixedMultiAirLinearizerSelectorAir {
    fn width(&self) -> usize {
        FixedMultiAirLinearizerSelectorCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirLinearizerSelectorAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("linearizer selector row");
        let next_row = main.row_slice(1).expect("next linearizer selector row");
        let local: &FixedMultiAirLinearizerSelectorCols<AB::Var> = (*row).borrow();
        let next: &FixedMultiAirLinearizerSelectorCols<AB::Var> = (*next_row).borrow();
        for flag in [local.active, local.is_first, local.is_last] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when_first_row().assert_zero(local.message_bit);
        builder.when(local.active * local.is_last).assert_eq(
            local.message_bit,
            AB::Expr::from_usize(self.log_message_len - 1),
        );
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.is_last);
        builder
            .when_last_row()
            .assert_eq(local.is_last, local.active);
        let mut transition = builder.when_transition();
        let mut same = transition.when(next.active);
        same.assert_eq(next.message_bit, local.message_bit + AB::Expr::ONE);
        same.assert_zero(next.is_first);
        assert_array_eq(&mut same, next.product_before, local.product_after);
        assert_array_eq(
            &mut builder.when(local.is_first),
            local.product_before,
            one_ext::<AB>(),
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.factor,
            local.y.map(Into::into),
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.product_after,
            ext_field_multiply::<AB::Expr>(local.product_before, local.factor),
        );
        self.y_bus.receive(
            builder,
            FixedMultiAirLinearizerYMessage {
                message_bit: local.message_bit.into(),
                value: local.y.map(Into::into),
            },
            local.active,
        );
        self.selector_bus.send(
            builder,
            FixedMultiAirLinearizerSelectorMessage {
                value: local.product_after.map(Into::into),
            },
            local.is_last,
        );
    }
}

pub fn generate_fixed_multi_air_linearizer_selector_trace(
    factors: &[EF],
    required_height: Option<usize>,
) -> Result<(RowMajorMatrix<F>, EF), FixedMultiAirLinearizerAdjointTraceError> {
    if factors.is_empty() {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
    }
    let height = required_height.unwrap_or_else(|| factors.len().next_power_of_two());
    if height < factors.len() {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
    }
    let width = FixedMultiAirLinearizerSelectorCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    let mut product = EF::ONE;
    for (message_bit, &factor) in factors.iter().enumerate() {
        let before = product;
        product *= factor;
        let row = &mut values[message_bit * width..(message_bit + 1) * width];
        let cols: &mut FixedMultiAirLinearizerSelectorCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.is_first = F::from_bool(message_bit == 0);
        cols.is_last = F::from_bool(message_bit + 1 == factors.len());
        cols.message_bit = F::from_usize(message_bit);
        copy_ext(&mut cols.y, factor);
        copy_ext(&mut cols.factor, factor);
        copy_ext(&mut cols.product_before, before);
        copy_ext(&mut cols.product_after, product);
    }
    Ok((RowMajorMatrix::new(values, width), product))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedMultiAirLinearizerRawComponentPlan {
    pub ordinal: usize,
    pub claim: usize,
    pub term: usize,
    pub is_eq: bool,
    pub is_zero: bool,
    pub block_start: usize,
    pub log_height: usize,
    pub rotation: DirectAirMappedRotation,
}

impl FixedMultiAirLinearizerRawComponentPlan {
    pub fn from_relation(
        relation: &FixedMultiAirPesatIndex<F, Digest>,
    ) -> Result<Vec<Self>, FixedMultiAirLinearizerAdjointTraceError> {
        let mut components = Vec::new();
        for region in 0..relation.region_count() {
            let endpoint = FixedMultiAirEndpointPlan::from_relation(relation, region)
                .map_err(|_| FixedMultiAirLinearizerAdjointTraceError::Shape)?;
            if endpoint.dynamic.is_empty() {
                components.push(Self {
                    ordinal: components.len(),
                    claim: region,
                    term: 0,
                    is_eq: false,
                    is_zero: true,
                    block_start: 0,
                    log_height: 0,
                    rotation: DirectAirMappedRotation::Current,
                });
            } else {
                for (term, dynamic) in endpoint.dynamic.iter().enumerate() {
                    components.push(Self {
                        ordinal: components.len(),
                        claim: region,
                        term,
                        is_eq: false,
                        is_zero: false,
                        block_start: dynamic.global_block_start,
                        log_height: dynamic.log_height,
                        rotation: dynamic.rotation,
                    });
                }
            }
        }
        if relation.description().padding_constraint_count != 0 {
            components.push(Self {
                ordinal: components.len(),
                claim: relation.region_count(),
                term: 0,
                is_eq: true,
                is_zero: false,
                block_start: 0,
                log_height: relation.pesat_shape().log_witness,
                rotation: DirectAirMappedRotation::Current,
            });
        }
        if components.is_empty() {
            return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
        }
        Ok(components)
    }

    /// Build the generic structured-dual schedule from the genuine complete
    /// terminal plan.
    ///
    /// This is deliberately only a setup adapter. The complete nonlinear
    /// proof remains a `FixedMultiAirCompleteTerminalProof`; no legacy
    /// terminal proof or local-only relation is constructed. Each native
    /// mapped opening becomes one term of the complete relation's single
    /// structured claim, in the exact globally concatenated/rho order fixed
    /// by the backend plan.
    pub fn from_complete_plan(
        plan: &FixedMultiAirCompleteTerminalCircuitPlan<F, Digest>,
    ) -> Result<Vec<Self>, FixedMultiAirLinearizerAdjointTraceError> {
        if plan.version != FIXED_MULTI_AIR_COMPLETE_TERMINAL_CIRCUIT_PLAN_VERSION
            || plan.canonical_bytes().is_err()
        {
            return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
        }

        if plan.mapped_openings.is_empty() {
            return Ok(vec![Self {
                ordinal: 0,
                claim: 0,
                term: 0,
                is_eq: false,
                is_zero: true,
                block_start: 0,
                log_height: 0,
                rotation: DirectAirMappedRotation::Current,
            }]);
        }

        let log_message_len = usize::from(plan.metadata.code_class.log_message_len);
        let message_len = 1usize
            .checked_shl(log_message_len as u32)
            .ok_or(FixedMultiAirLinearizerAdjointTraceError::Shape)?;
        plan.mapped_openings
            .iter()
            .enumerate()
            .map(|(ordinal, opening)| {
                let global_ordinal = usize::try_from(opening.global_opening_ordinal)
                    .map_err(|_| FixedMultiAirLinearizerAdjointTraceError::Shape)?;
                let rho_ordinal = usize::try_from(opening.rho_ordinal)
                    .map_err(|_| FixedMultiAirLinearizerAdjointTraceError::Shape)?;
                let regional_ordinal = usize::try_from(opening.regional_opening_ordinal)
                    .map_err(|_| FixedMultiAirLinearizerAdjointTraceError::Shape)?;
                let region_ordinal = usize::try_from(opening.region_ordinal)
                    .map_err(|_| FixedMultiAirLinearizerAdjointTraceError::Shape)?;
                let block_start = u64::try_from(opening.block.start)
                    .map_err(|_| FixedMultiAirLinearizerAdjointTraceError::Shape)?;
                let block_height = 1usize
                    .checked_shl(opening.block.log_height as u32)
                    .ok_or(FixedMultiAirLinearizerAdjointTraceError::Shape)?;
                let expected_source_rotation = match opening.rotation {
                    PrismalinearMappedColumnRotation::Current => 0,
                    PrismalinearMappedColumnRotation::Next => 1,
                };
                if global_ordinal != ordinal
                    || rho_ordinal != ordinal
                    || opening.source.start != block_start
                    || usize::from(opening.source.log_height) != opening.block.log_height
                    || usize::from(opening.source.rotation) != expected_source_rotation
                    || opening.eq.global_log_height as usize != log_message_len
                    || opening.eq.start != block_start
                    || usize::from(opening.eq.log_height) != opening.block.log_height
                    || opening.eq.rotation != opening.source.rotation
                    || opening
                        .block
                        .start
                        .checked_add(block_height)
                        .is_none_or(|end| end > message_len)
                {
                    return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
                }
                let expected_regional = plan
                    .regions
                    .get(region_ordinal)
                    .and_then(|region| {
                        let component = match opening.component {
                            openvm_stark_backend::native_warp::FixedMultiAirCompleteTerminalCircuitComponentKind::Local => &region.local,
                            openvm_stark_backend::native_warp::FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction => &region.interaction,
                        };
                        let opening_count =
                            usize::try_from(component.identity.opening_count).ok()?;
                        if component.identity.kind != opening.component
                            || usize::try_from(component.identity.region_ordinal).ok()
                                != Some(region_ordinal)
                            || !component.identity.proof_present
                            || regional_ordinal >= opening_count
                            || component.expression.dynamic_columns.get(regional_ordinal)
                                != Some(&opening.source)
                        {
                            return None;
                        }
                        usize::try_from(component.identity.opening_ordinal_start)
                            .ok()
                            .and_then(|start| start.checked_add(regional_ordinal))
                    })
                    ;
                if expected_regional != Some(ordinal) {
                    return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
                }
                let rotation = match opening.rotation {
                    PrismalinearMappedColumnRotation::Current => {
                        DirectAirMappedRotation::Current
                    }
                    PrismalinearMappedColumnRotation::Next => DirectAirMappedRotation::Next,
                };
                Ok(Self {
                    ordinal,
                    claim: 0,
                    term: ordinal,
                    is_eq: false,
                    is_zero: false,
                    block_start: opening.block.start,
                    log_height: opening.block.log_height,
                    rotation,
                })
            })
            .collect()
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirLinearizerRawScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub is_component_first: T,
    pub is_component_last: T,
    /// Whether this LSB-first automaton step consumes a local row bit.  Once
    /// all local bits have been consumed, cyclic-rotation carry is discarded
    /// while block-start carry continues through the remaining global bits.
    pub is_source_bit: T,
    pub is_eq: T,
    pub is_zero: T,
    pub rotation: T,
    pub ordinal: T,
    pub claim: T,
    pub term: T,
    pub step: T,
    pub point_coordinate: T,
    pub global_coordinate: T,
    pub start_bit: T,
    pub block_start: T,
    pub log_height: T,
    /// For each `(rotation_carry, start_carry, local_bit)` transition, the
    /// output bit and next block-start carry. The next rotation carry is a
    /// fixed function of `is_source_bit`, `rotation_carry`, and `local_bit`.
    pub global_bit: [T; 8],
    pub next_start_carry: [T; 8],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirLinearizerRawCols<T> {
    pub kind: T,
    pub log_message_len: T,
    pub term_count: T,
    pub point_len: T,
    pub target: [T; D_EF],
    pub batching_scale: [T; D_EF],
    pub term_scale: [T; D_EF],
    pub row_point: [T; D_EF],
    pub aux_point: [T; D_EF],
    /// Exact two-carry state indexed by
    /// `2 * rotation_carry + start_carry`.
    pub state_before: [[T; D_EF]; 4],
    pub state_local_terms: [[T; D_EF]; 8],
    /// One term for every `(state, local_bit)` pair.
    pub state_terms: [[T; D_EF]; 8],
    pub state_after: [[T; D_EF]; 4],
    /// Degree-splitting auxiliaries. Keeping these products explicit bounds
    /// the gated terminal equation by degree four.
    pub scaled_value: [T; D_EF],
    pub packed_value: [T; D_EF],
    pub value: [T; D_EF],
}

/// Evaluates the complete xi-batched raw systematic-message functional at the
/// auxiliary sumcheck point.  Every mapped identity is cached key data; only
/// rho scales and row points are witness values received from constrained
/// regional producers. Every mapped component uses the native verifier's
/// exact two-carry scan: one carry for cyclic `+1` row rotation and one for
/// adding an arbitrary packed block start.
pub struct FixedMultiAirLinearizerRawAir {
    pub batched_claim_bus: FixedMultiAirBatchedClaimBus,
    pub mapped_term_bus: FixedMultiAirMappedTermBus,
    pub structured_point_bus:
        crate::native_warp::terminal::fixed_multi_air::FixedMultiAirStructuredPointBus,
    pub aux_point_bus: FixedMultiAirLinearizerAuxPointBus,
    pub raw_term_bus: FixedMultiAirLinearizerRawTermBus,
    pub log_message_len: usize,
    pub component_count: usize,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirLinearizerRawAir {}
impl PartitionedBaseAir<F> for FixedMultiAirLinearizerRawAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirLinearizerRawScheduleCols::<F>::width()]
    }
    fn common_main_width(&self) -> usize {
        FixedMultiAirLinearizerRawCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirLinearizerRawAir {}
impl BaseAir<F> for FixedMultiAirLinearizerRawAir {
    fn width(&self) -> usize {
        FixedMultiAirLinearizerRawScheduleCols::<F>::width()
            + FixedMultiAirLinearizerRawCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirLinearizerRawAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("linearizer raw schedule")
            .to_vec();
        let next_cached = builder.cached_mains()[0]
            .row_slice(1)
            .expect("next linearizer raw schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("linearizer raw row")
            .to_vec();
        let next_common = builder
            .common_main()
            .row_slice(1)
            .expect("next linearizer raw row")
            .to_vec();
        let schedule: &FixedMultiAirLinearizerRawScheduleCols<AB::Var> = cached.as_slice().borrow();
        let next_schedule: &FixedMultiAirLinearizerRawScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FixedMultiAirLinearizerRawCols<AB::Var> = common.as_slice().borrow();
        let next: &FixedMultiAirLinearizerRawCols<AB::Var> = next_common.as_slice().borrow();
        for flag in [
            schedule.active,
            schedule.is_first,
            schedule.is_last,
            schedule.is_component_first,
            schedule.is_component_last,
            schedule.is_source_bit,
            schedule.is_eq,
            schedule.is_zero,
            schedule.rotation,
            schedule.start_bit,
        ]
        .into_iter()
        .chain(schedule.global_bit)
        .chain(schedule.next_start_carry)
        {
            builder.assert_bool(flag);
        }
        builder.assert_zero(schedule.is_source_bit * schedule.is_zero);
        builder.when_first_row().assert_one(schedule.active);
        builder.when_first_row().assert_one(schedule.is_first);
        builder
            .when_first_row()
            .assert_one(schedule.is_component_first);
        builder.when_first_row().assert_zero(schedule.ordinal);
        builder.when(schedule.active * schedule.is_last).assert_eq(
            schedule.ordinal,
            AB::Expr::from_usize(self.component_count - 1),
        );
        builder
            .when_transition()
            .assert_bool(schedule.active - next_schedule.active);
        builder
            .when_transition()
            .assert_eq(schedule.active - next_schedule.active, schedule.is_last);
        builder
            .when_last_row()
            .assert_eq(schedule.is_last, schedule.active);
        let same_component =
            next_schedule.active * (AB::Expr::ONE - next_schedule.is_component_first);
        let mut transition = builder.when_transition();
        let mut same = transition.when(same_component.clone());
        same.assert_eq(next_schedule.ordinal, schedule.ordinal);
        same.assert_eq(next_schedule.step, schedule.step + AB::Expr::ONE);
        same.assert_eq(next.kind, local.kind);
        same.assert_eq(next.log_message_len, local.log_message_len);
        same.assert_eq(next.term_count, local.term_count);
        same.assert_eq(next.point_len, local.point_len);
        assert_array_eq(&mut same, next.target, local.target);
        assert_array_eq(&mut same, next.batching_scale, local.batching_scale);
        assert_array_eq(&mut same, next.term_scale, local.term_scale);
        for state in 0..4 {
            assert_array_eq(
                &mut same,
                next.state_before[state],
                local.state_after[state],
            );
        }

        let nonzero = schedule.active * (AB::Expr::ONE - schedule.is_zero);
        let initial_state = [
            AB::Expr::ONE - schedule.rotation,
            AB::Expr::ZERO,
            AB::Expr::from(schedule.rotation),
            AB::Expr::ZERO,
        ];
        for (state, initial) in initial_state.into_iter().enumerate() {
            let expected = core::array::from_fn(|limb| {
                if limb == 0 {
                    nonzero.clone() * initial.clone()
                } else {
                    AB::Expr::ZERO
                }
            });
            assert_array_eq(
                &mut builder.when(schedule.is_component_first),
                local.state_before[state],
                expected,
            );
        }

        for state in 0..4 {
            let rotation_carry = state / 2;
            let start_carry = state % 2;
            for local_bit in 0..2 {
                let term = state * 2 + local_bit;
                let source_sum = local_bit + rotation_carry;
                let source_bit = source_sum & 1;
                let local_weight = if local_bit == 0 {
                    core::array::from_fn(|limb| {
                        let point = AB::Expr::from(local.row_point[limb]);
                        if limb == 0 {
                            AB::Expr::ONE - schedule.is_source_bit * point
                        } else {
                            AB::Expr::ZERO - AB::Expr::from(schedule.is_source_bit) * point
                        }
                    })
                } else {
                    local
                        .row_point
                        .map(|point| AB::Expr::from(schedule.is_source_bit) * point)
                };
                let global_weight = eq_bit_expr::<AB>(local.aux_point, schedule.global_bit[term]);
                let state_local =
                    ext_field_multiply::<AB::Expr>(local.state_before[state], local_weight);
                assert_array_eq(
                    &mut builder.when(schedule.active),
                    local.state_local_terms[term],
                    state_local,
                );
                assert_array_eq(
                    &mut builder.when(schedule.active),
                    local.state_terms[term],
                    ext_field_multiply::<AB::Expr>(local.state_local_terms[term], global_weight),
                );

                let effective_source_bit =
                    schedule.is_source_bit * AB::Expr::from_usize(source_bit);
                builder.when(nonzero.clone()).assert_eq(
                    schedule.global_bit[term] + AB::Expr::TWO * schedule.next_start_carry[term],
                    effective_source_bit + schedule.start_bit + AB::Expr::from_usize(start_carry),
                );
            }
        }
        for target in 0..4 {
            let target_rotation = target / 2;
            let target_start = target % 2;
            let expected = core::array::from_fn(|limb| {
                (0..8).fold(AB::Expr::ZERO, |sum, term| {
                    let state = term / 2;
                    let local_bit = term % 2;
                    let rotation_carry = state / 2;
                    let source_sum = local_bit + rotation_carry;
                    let rotation_out = source_sum >> 1;
                    let next_rotation = schedule.is_source_bit * AB::Expr::from_usize(rotation_out);
                    let select_rotation = if target_rotation == 0 {
                        AB::Expr::ONE - next_rotation
                    } else {
                        next_rotation
                    };
                    let select_start = if target_start == 0 {
                        AB::Expr::ONE - schedule.next_start_carry[term]
                    } else {
                        AB::Expr::from(schedule.next_start_carry[term])
                    };
                    sum + select_rotation
                        * select_start
                        * AB::Expr::from(local.state_terms[term][limb])
                })
            });
            assert_array_eq(
                &mut builder.when(schedule.active),
                local.state_after[target],
                expected,
            );
        }
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.scaled_value,
            ext_field_multiply::<AB::Expr>(local.batching_scale, local.term_scale),
        );
        let terminal_nonzero = schedule.is_component_last * (AB::Expr::ONE - schedule.is_zero);
        let packed_value = ext_field_add::<AB::Expr>(local.state_after[0], local.state_after[2]);
        assert_array_eq(
            &mut builder.when(terminal_nonzero.clone()),
            local.packed_value,
            packed_value,
        );
        assert_array_eq(
            &mut builder.when(terminal_nonzero),
            local.value,
            ext_field_multiply::<AB::Expr>(local.scaled_value, local.packed_value),
        );
        assert_array_eq(
            &mut builder.when(schedule.is_zero),
            local.packed_value,
            [AB::Expr::ZERO; D_EF],
        );
        assert_array_eq(
            &mut builder.when(schedule.is_zero),
            local.value,
            [AB::Expr::ZERO; D_EF],
        );

        self.batched_claim_bus.lookup_key(
            builder,
            FixedMultiAirBatchedClaimMessage {
                claim: schedule.claim.into(),
                kind: local.kind.into(),
                log_message_len: local.log_message_len.into(),
                term_count: local.term_count.into(),
                point_len: local.point_len.into(),
                target: local.target.map(Into::into),
                batching_scale: local.batching_scale.map(Into::into),
            },
            schedule.is_component_first,
        );
        self.mapped_term_bus.receive(
            builder,
            FixedMultiAirMappedTermMessage {
                claim: schedule.claim.into(),
                term: schedule.term.into(),
                block_start: schedule.block_start.into(),
                log_height: schedule.log_height.into(),
                l_skip: AB::Expr::ZERO,
                rotation: schedule.rotation.into(),
                scale: local.term_scale.map(Into::into),
            },
            schedule.is_component_first
                * (AB::Expr::ONE - schedule.is_eq)
                * (AB::Expr::ONE - schedule.is_zero),
        );
        self.structured_point_bus.receive(
            builder,
            crate::native_warp::terminal::fixed_multi_air::FixedMultiAirStructuredPointMessage {
                claim: schedule.claim.into(),
                term: schedule.term.into(),
                coordinate: schedule.point_coordinate.into(),
                value: local.row_point.map(Into::into),
            },
            nonzero.clone() * schedule.is_source_bit,
        );
        self.aux_point_bus.lookup_key(
            builder,
            FixedMultiAirLinearizerAuxPointMessage {
                coordinate: schedule.global_coordinate.into(),
                value: local.aux_point.map(Into::into),
            },
            nonzero,
        );
        self.raw_term_bus.send(
            builder,
            FixedMultiAirLinearizerRawTermMessage {
                ordinal: schedule.ordinal.into(),
                value: local.value.map(Into::into),
            },
            schedule.is_component_last,
        );
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirLinearizerRawTraceOutput {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub term_values: Vec<EF>,
    pub value: EF,
    pub aux_point_lookup_counts: Vec<usize>,
}

pub fn generate_fixed_multi_air_linearizer_raw_traces(
    log_message_len: usize,
    components: &[FixedMultiAirLinearizerRawComponentPlan],
    claims: &[TerminalStructuredLinearClaim<EF>],
    batching_scales: &[EF],
    aux_point: &[EF],
    required_height: Option<usize>,
) -> Result<FixedMultiAirLinearizerRawTraceOutput, FixedMultiAirLinearizerAdjointTraceError> {
    if log_message_len == 0
        || log_message_len > FIXED_MULTI_AIR_LINEARIZER_MAX_LOG_MESSAGE
        || aux_point.len() != log_message_len
        || claims.len() != batching_scales.len()
        || components.is_empty()
    {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
    }
    let message_len = 1usize
        .checked_shl(log_message_len as u32)
        .ok_or(FixedMultiAirLinearizerAdjointTraceError::Shape)?;
    let valid_rows = components
        .len()
        .checked_mul(log_message_len)
        .ok_or(FixedMultiAirLinearizerAdjointTraceError::Shape)?;
    let natural_height = valid_rows
        .checked_next_power_of_two()
        .ok_or(FixedMultiAirLinearizerAdjointTraceError::Shape)?;
    let height = required_height.unwrap_or(natural_height);
    if height < valid_rows {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
    }
    let cached_width = FixedMultiAirLinearizerRawScheduleCols::<F>::width();
    let common_width = FixedMultiAirLinearizerRawCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    let mut term_values = Vec::with_capacity(components.len());
    let mut aux_point_lookup_counts = vec![0usize; log_message_len];
    let mut row = 0usize;
    for component in components {
        let block_len = 1usize
            .checked_shl(component.log_height as u32)
            .ok_or(FixedMultiAirLinearizerAdjointTraceError::Shape)?;
        if component.ordinal != term_values.len()
            || component.claim >= claims.len()
            || component.log_height > log_message_len
            || component
                .block_start
                .checked_add(block_len)
                .is_none_or(|end| end > message_len)
            || (component.is_zero
                && (component.is_eq
                    || component.block_start != 0
                    || component.log_height != 0
                    || component.rotation != DirectAirMappedRotation::Current))
        {
            return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
        }
        let claim = &claims[component.claim];
        let (kind, term_count, point_len, target, term_scale, row_point): (
            usize,
            usize,
            usize,
            EF,
            EF,
            &[EF],
        ) = match &claim.weight {
            TerminalWeightSpec::Eq { point } if component.is_eq => {
                if point.len() != log_message_len
                    || component.block_start != 0
                    || component.log_height != log_message_len
                    || component.rotation != DirectAirMappedRotation::Current
                {
                    return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
                }
                (
                    1usize,
                    0usize,
                    point.len(),
                    claim.target,
                    EF::ONE,
                    point.as_slice(),
                )
            }
            TerminalWeightSpec::PrismalinearMappedColumns(weight)
                if !component.is_eq && !component.is_zero =>
            {
                let term = weight
                    .terms
                    .get(component.term)
                    .ok_or(FixedMultiAirLinearizerAdjointTraceError::Shape)?;
                let expected_rotation = match component.rotation {
                    DirectAirMappedRotation::Current => PrismalinearMappedColumnRotation::Current,
                    DirectAirMappedRotation::Next => PrismalinearMappedColumnRotation::Next,
                };
                if weight.validate().is_err()
                    || weight.log_message_len != log_message_len
                    || term.block.start != component.block_start
                    || term.block.log_height != component.log_height
                    || term.l_skip != 0
                    || term.barycentric_weights.as_slice() != [EF::ONE]
                    || term.rotation != expected_rotation
                    || term.folded_row_eq_point.len() != component.log_height
                {
                    return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
                }
                (
                    0usize,
                    weight.terms.len(),
                    0usize,
                    claim.target,
                    term.scale,
                    term.folded_row_eq_point.as_slice(),
                )
            }
            TerminalWeightSpec::PrismalinearMappedColumns(weight) if component.is_zero => {
                if weight.log_message_len != log_message_len || !weight.terms.is_empty() {
                    return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
                }
                (0usize, 0usize, 0usize, claim.target, EF::ZERO, &[] as &[EF])
            }
            _ => return Err(FixedMultiAirLinearizerAdjointTraceError::Shape),
        };
        let is_zero = component.is_zero;
        let mut state = [[EF::ZERO; 2]; 2];
        if !is_zero {
            state[usize::from(component.rotation == DirectAirMappedRotation::Next)][0] = EF::ONE;
        }
        let mut component_value = EF::ZERO;
        for step in 0..log_message_len {
            let is_source_bit = !is_zero && step < component.log_height;
            let point_coordinate = if is_source_bit {
                component.log_height - 1 - step
            } else {
                0
            };
            let global_coordinate = log_message_len - 1 - step;
            if !is_zero {
                aux_point_lookup_counts[global_coordinate] += 1;
            }
            let aux = if is_zero {
                EF::ZERO
            } else {
                aux_point[global_coordinate]
            };
            let source_coordinate = if is_source_bit {
                row_point[point_coordinate]
            } else {
                EF::ZERO
            };
            let local_weights = if is_source_bit {
                [EF::ONE - source_coordinate, source_coordinate]
            } else {
                [EF::ONE, EF::ZERO]
            };
            let start_bit = (component.block_start >> step) & 1;
            let state_before = state;
            let mut global_weights = [EF::ZERO; 8];
            let mut state_local_terms = [EF::ZERO; 8];
            let mut state_terms = [EF::ZERO; 8];
            let mut state_after = [[EF::ZERO; 2]; 2];
            let mut global_bits = [0usize; 8];
            let mut next_start_carries = [0usize; 8];
            for rotation_carry in 0..=1usize {
                for start_carry in 0..=1usize {
                    let state_index = 2 * rotation_carry + start_carry;
                    for local_bit in 0..2 {
                        let term = 2 * state_index + local_bit;
                        let (source_bit, next_rotation) = if is_source_bit {
                            let sum = local_bit + rotation_carry;
                            (sum & 1, sum >> 1)
                        } else {
                            (0, 0)
                        };
                        let sum = source_bit + start_bit + start_carry;
                        let global_bit = sum & 1;
                        let next_start = sum >> 1;
                        global_bits[term] = global_bit;
                        next_start_carries[term] = next_start;
                        global_weights[term] = if global_bit == 0 { EF::ONE - aux } else { aux };
                        state_local_terms[term] =
                            state_before[rotation_carry][start_carry] * local_weights[local_bit];
                        state_terms[term] = state_local_terms[term] * global_weights[term];
                        state_after[next_rotation][next_start] += state_terms[term];
                    }
                }
            }
            state = state_after;
            if step + 1 == log_message_len {
                component_value = if is_zero {
                    EF::ZERO
                } else {
                    batching_scales[component.claim]
                        * term_scale
                        * (state_after[0][0] + state_after[1][0])
                };
            }
            let cached_row = &mut cached[row * cached_width..(row + 1) * cached_width];
            let schedule: &mut FixedMultiAirLinearizerRawScheduleCols<F> = cached_row.borrow_mut();
            schedule.active = F::ONE;
            schedule.is_first = F::from_bool(row == 0);
            schedule.is_last = F::from_bool(row + 1 == valid_rows);
            schedule.is_component_first = F::from_bool(step == 0);
            schedule.is_component_last = F::from_bool(step + 1 == log_message_len);
            schedule.is_source_bit = F::from_bool(is_source_bit);
            schedule.is_eq = F::from_bool(component.is_eq);
            schedule.is_zero = F::from_bool(component.is_zero);
            schedule.rotation = F::from_bool(component.rotation == DirectAirMappedRotation::Next);
            schedule.ordinal = F::from_usize(component.ordinal);
            schedule.claim = F::from_usize(component.claim);
            schedule.term = F::from_usize(component.term);
            schedule.step = F::from_usize(step);
            schedule.point_coordinate = F::from_usize(point_coordinate);
            schedule.global_coordinate = F::from_usize(global_coordinate);
            schedule.start_bit = F::from_usize(start_bit);
            schedule.block_start = F::from_usize(component.block_start);
            schedule.log_height = F::from_usize(component.log_height);
            for term in 0..8 {
                schedule.global_bit[term] = F::from_usize(global_bits[term]);
                schedule.next_start_carry[term] = F::from_usize(next_start_carries[term]);
            }
            let common_row = &mut common[row * common_width..(row + 1) * common_width];
            let cols: &mut FixedMultiAirLinearizerRawCols<F> = common_row.borrow_mut();
            cols.kind = F::from_usize(kind);
            cols.log_message_len = F::from_usize(log_message_len);
            cols.term_count = F::from_usize(term_count);
            cols.point_len = F::from_usize(point_len);
            copy_ext(&mut cols.target, target);
            copy_ext(&mut cols.batching_scale, batching_scales[component.claim]);
            copy_ext(&mut cols.term_scale, term_scale);
            copy_ext(&mut cols.row_point, source_coordinate);
            copy_ext(&mut cols.aux_point, aux);
            for (target, value) in cols
                .state_before
                .iter_mut()
                .zip(state_before.into_iter().flatten())
            {
                copy_ext(target, value);
            }
            for (target, value) in cols.state_local_terms.iter_mut().zip(state_local_terms) {
                copy_ext(target, value);
            }
            for (target, value) in cols.state_terms.iter_mut().zip(state_terms) {
                copy_ext(target, value);
            }
            for (target, value) in cols
                .state_after
                .iter_mut()
                .zip(state_after.into_iter().flatten())
            {
                copy_ext(target, value);
            }
            copy_ext(
                &mut cols.scaled_value,
                batching_scales[component.claim] * term_scale,
            );
            copy_ext(
                &mut cols.packed_value,
                if is_zero {
                    EF::ZERO
                } else {
                    state_after[0][0] + state_after[1][0]
                },
            );
            copy_ext(&mut cols.value, component_value);
            row += 1;
        }
        term_values.push(component_value);
    }
    let value = term_values.iter().copied().sum();
    Ok(FixedMultiAirLinearizerRawTraceOutput {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        term_values,
        value,
        aux_point_lookup_counts,
    })
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirLinearizerRawSumCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub ordinal: T,
    pub term: [T; D_EF],
    pub sum_before: [T; D_EF],
    pub sum_after: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(FixedMultiAirLinearizerRawSumCols<u8>)]
pub struct FixedMultiAirLinearizerRawSumAir {
    pub term_bus: FixedMultiAirLinearizerRawTermBus,
    pub output_bus: FixedMultiAirLinearizerRawWeightBus,
    pub component_count: usize,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirLinearizerRawSumAir {}
impl PartitionedBaseAir<F> for FixedMultiAirLinearizerRawSumAir {}
impl BaseAir<F> for FixedMultiAirLinearizerRawSumAir {
    fn width(&self) -> usize {
        FixedMultiAirLinearizerRawSumCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for FixedMultiAirLinearizerRawSumAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("linearizer raw sum row");
        let next_row = main.row_slice(1).expect("next linearizer raw sum row");
        let local: &FixedMultiAirLinearizerRawSumCols<AB::Var> = (*row).borrow();
        let next: &FixedMultiAirLinearizerRawSumCols<AB::Var> = (*next_row).borrow();
        for flag in [local.active, local.is_first, local.is_last] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when_first_row().assert_zero(local.ordinal);
        builder.when(local.active * local.is_last).assert_eq(
            local.ordinal,
            AB::Expr::from_usize(self.component_count - 1),
        );
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.is_last);
        builder
            .when_last_row()
            .assert_eq(local.is_last, local.active);
        let mut transition = builder.when_transition();
        let mut same = transition.when(next.active);
        same.assert_eq(next.ordinal, local.ordinal + AB::Expr::ONE);
        same.assert_zero(next.is_first);
        assert_array_eq(&mut same, next.sum_before, local.sum_after);
        assert_array_eq(
            &mut builder.when(local.is_first),
            local.sum_before,
            [AB::Expr::ZERO; D_EF],
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.sum_after,
            ext_field_add::<AB::Expr>(local.sum_before, local.term),
        );
        self.term_bus.receive(
            builder,
            FixedMultiAirLinearizerRawTermMessage {
                ordinal: local.ordinal.into(),
                value: local.term.map(Into::into),
            },
            local.active,
        );
        self.output_bus.send(
            builder,
            FixedMultiAirLinearizerRawWeightMessage {
                value: local.sum_after.map(Into::into),
            },
            local.is_last,
        );
    }
}

pub fn generate_fixed_multi_air_linearizer_raw_sum_trace(
    term_values: &[EF],
    required_height: Option<usize>,
) -> Result<RowMajorMatrix<F>, FixedMultiAirLinearizerAdjointTraceError> {
    if term_values.is_empty() {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
    }
    let height = required_height.unwrap_or_else(|| term_values.len().next_power_of_two());
    if height < term_values.len() {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Shape);
    }
    let width = FixedMultiAirLinearizerRawSumCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    let mut sum = EF::ZERO;
    for (ordinal, &term) in term_values.iter().enumerate() {
        let before = sum;
        sum += term;
        let row = &mut values[ordinal * width..(ordinal + 1) * width];
        let cols: &mut FixedMultiAirLinearizerRawSumCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.is_first = F::from_bool(ordinal == 0);
        cols.is_last = F::from_bool(ordinal + 1 == term_values.len());
        cols.ordinal = F::from_usize(ordinal);
        copy_ext(&mut cols.term, term);
        copy_ext(&mut cols.sum_before, before);
        copy_ext(&mut cols.sum_after, sum);
    }
    Ok(RowMajorMatrix::new(values, width))
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirLinearizerFinalCols<T> {
    pub active: T,
    pub statement_tidx: T,
    pub root: [T; openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE],
    pub batching_challenge: [T; D_EF],
    pub statement_claim: [T; D_EF],
    pub initial_claim: [T; D_EF],
    pub final_claim: [T; D_EF],
    pub raw_weight: [T; D_EF],
    pub selector: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(FixedMultiAirLinearizerFinalCols<u8>)]
pub struct FixedMultiAirLinearizerFinalAir {
    pub statement_bus: NativeTerminalWhirStatementBus,
    pub sumcheck_bus: FixedMultiAirLinearizerSumcheckFinalBus,
    pub raw_weight_bus: FixedMultiAirLinearizerRawWeightBus,
    pub selector_bus: FixedMultiAirLinearizerSelectorBus,
    pub output_bus: NativeTerminalWhirLinearizerWeightBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirLinearizerFinalAir {}
impl PartitionedBaseAir<F> for FixedMultiAirLinearizerFinalAir {}
impl BaseAir<F> for FixedMultiAirLinearizerFinalAir {
    fn width(&self) -> usize {
        FixedMultiAirLinearizerFinalCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirLinearizerFinalAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("linearizer final row");
        let local: &FixedMultiAirLinearizerFinalCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_transition().assert_zero(local.active);
        assert_array_eq(
            &mut builder.when(local.active),
            local.final_claim,
            ext_field_multiply::<AB::Expr>(local.raw_weight, local.selector),
        );
        self.statement_bus.lookup_key(
            builder,
            NativeTerminalWhirStatementMessage {
                tidx: local.statement_tidx.into(),
                root: local.root.map(Into::into),
                batching_challenge: local.batching_challenge.map(Into::into),
                initial_claim: local.statement_claim.map(Into::into),
            },
            local.active,
        );
        self.sumcheck_bus.receive(
            builder,
            FixedMultiAirLinearizerSumcheckFinalMessage {
                initial_claim: local.initial_claim.map(Into::into),
                final_claim: local.final_claim.map(Into::into),
            },
            local.active,
        );
        self.raw_weight_bus.receive(
            builder,
            FixedMultiAirLinearizerRawWeightMessage {
                value: local.raw_weight.map(Into::into),
            },
            local.active,
        );
        self.selector_bus.receive(
            builder,
            FixedMultiAirLinearizerSelectorMessage {
                value: local.selector.map(Into::into),
            },
            local.active,
        );
        self.output_bus.send(
            builder,
            NativeTerminalWhirLinearizerWeightMessage {
                value: local.initial_claim.map(Into::into),
            },
            local.active,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn generate_fixed_multi_air_linearizer_final_trace(
    statement_tidx: usize,
    root: openvm_stark_sdk::config::baby_bear_poseidon2::Digest,
    batching_challenge: EF,
    statement_claim: EF,
    initial_claim: EF,
    final_claim: EF,
    raw_weight: EF,
    selector: EF,
) -> Result<RowMajorMatrix<F>, FixedMultiAirLinearizerAdjointTraceError> {
    if final_claim != raw_weight * selector {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Claim);
    }
    let width = FixedMultiAirLinearizerFinalCols::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut FixedMultiAirLinearizerFinalCols<F> = values.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.statement_tidx = F::from_usize(statement_tidx);
    cols.root = root;
    copy_ext(&mut cols.batching_challenge, batching_challenge);
    copy_ext(&mut cols.statement_claim, statement_claim);
    copy_ext(&mut cols.initial_claim, initial_claim);
    copy_ext(&mut cols.final_claim, final_claim);
    copy_ext(&mut cols.raw_weight, raw_weight);
    copy_ext(&mut cols.selector, selector);
    Ok(RowMajorMatrix::new(values, width))
}

fn lagrange_denominator(count: usize, index: usize) -> F {
    (0..count)
        .filter(|&other| other != index)
        .map(|other| F::from_usize(index) - F::from_usize(other))
        .product()
}

fn eq_bit_expr<AB: AirBuilder<F = F>>(value: [AB::Var; D_EF], bit: AB::Var) -> [AB::Expr; D_EF] {
    core::array::from_fn(|limb| {
        let value = AB::Expr::from(value[limb].clone());
        if limb == 0 {
            AB::Expr::from(bit.clone()) * value.clone()
                + (AB::Expr::ONE - AB::Expr::from(bit.clone())) * (AB::Expr::ONE - value)
        } else {
            AB::Expr::from(bit.clone()) * value.clone()
                - (AB::Expr::ONE - AB::Expr::from(bit.clone())) * value
        }
    })
}

fn interpolate(evaluations: &[EF], point: EF) -> EF {
    evaluations
        .iter()
        .enumerate()
        .map(|(index, &evaluation)| {
            let mut numerator = EF::ONE;
            for other in 0..evaluations.len() {
                if other != index {
                    numerator *= point - EF::from_usize(other);
                }
            }
            evaluation
                * numerator
                * EF::from(lagrange_denominator(evaluations.len(), index).inverse())
        })
        .sum()
}

fn observe_const<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    tidx: AB::Expr,
    value: u64,
    count: AB::Expr,
) {
    bus.observe_ext(
        builder,
        AB::Expr::ZERO,
        tidx,
        ext_from_base::<AB>(AB::Expr::from_u64(value)),
        count,
    );
}

fn ext_from_base<AB: AirBuilder<F = F>>(value: AB::Expr) -> [AB::Expr; D_EF] {
    core::array::from_fn(|limb| {
        if limb == 0 {
            value.clone()
        } else {
            AB::Expr::ZERO
        }
    })
}

fn one_ext<AB: AirBuilder<F = F>>() -> [AB::Expr; D_EF] {
    ext_from_base::<AB>(AB::Expr::ONE)
}

fn read_ext(
    transcript: &TranscriptLog<F, [F; 16]>,
    tidx: usize,
    is_sample: bool,
) -> Result<EF, FixedMultiAirLinearizerAdjointTraceError> {
    let end = tidx
        .checked_add(D_EF)
        .ok_or(FixedMultiAirLinearizerAdjointTraceError::Transcript)?;
    let values = transcript
        .values()
        .get(tidx..end)
        .ok_or(FixedMultiAirLinearizerAdjointTraceError::Transcript)?;
    let samples = transcript
        .samples()
        .get(tidx..end)
        .ok_or(FixedMultiAirLinearizerAdjointTraceError::Transcript)?;
    if samples.iter().any(|&sample| sample != is_sample) {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Transcript);
    }
    EF::from_basis_coefficients_slice(values)
        .ok_or(FixedMultiAirLinearizerAdjointTraceError::Transcript)
}

fn expect_ext(
    transcript: &TranscriptLog<F, [F; 16]>,
    tidx: usize,
    expected: EF,
    is_sample: bool,
) -> Result<(), FixedMultiAirLinearizerAdjointTraceError> {
    if read_ext(transcript, tidx, is_sample)? != expected {
        return Err(FixedMultiAirLinearizerAdjointTraceError::Transcript);
    }
    Ok(())
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

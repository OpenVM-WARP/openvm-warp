//! Typed trace generation for setup-fixed ordered stacked-opening reductions.
//!
//! Ordinary recursion obtains stacking layouts and opening claims by parsing a complete child
//! [`Proof`]. Setup-PCS authority already has the canonical commitment-major representation used
//! by the backend stacked-reduction verifier. This module lets the same six stacking AIRs consume
//! that representation directly. It never constructs a partial or synthetic child proof.

use core::borrow::BorrowMut;
use std::collections::{HashMap, HashSet};

use itertools::Itertools;
use openvm_cpu_backend::CpuBackend;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, TwoAdicField},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    poly_common::{
        eval_eq_mle, eval_eq_prism, eval_eq_uni, eval_eq_uni_at_one, eval_in_uni,
        eval_rot_kernel_prism, interpolate_quadratic_at_012, Squarable,
    },
    proof::StackingProof,
    prover::{stacked_pcs::StackedLayout, sumcheck::sumcheck_round0_deg, AirProvingContext},
    StarkProtocolConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, D_EF, EF, F};

use super::{
    claims::StackingClaimsCols, eq_base::EqBaseCols, eq_bits::EqBitsCols,
    opening::OpeningClaimsCols, sumcheck::SumcheckRoundsCols, univariate::UnivariateRoundCols,
};
use crate::{batch_constraint::eq_airs::EqNegCols, system::StackingPreflight};

/// Value-free identity sent by `OpeningClaimsAir` on the isolated authority claim bus.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct OrderedStackingClaimIdentity {
    pub sort_idx: usize,
    pub part_idx: usize,
    pub col_idx: usize,
}

/// One canonical source-column opening in commitment/layout order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrderedStackingClaim {
    pub identity: OrderedStackingClaimIdentity,
    pub current: EF,
    /// Must be `Some` exactly when the setup-fixed layout marks this source matrix as rotated.
    pub rotated: Option<EF>,
}

/// Exact transcript indices and sampled values needed by the ordinary stacking AIRs.
#[derive(Clone, Copy, Debug)]
pub struct OrderedStackingPreflight<'a> {
    /// Transcript index of the first current-claim limb. Claims are observed current then rotated,
    /// and lambda is sampled immediately after the last claim.
    pub claim_observation_tidx: usize,
    pub stacking: &'a StackingPreflight,
    /// Source-verifier-owned opening point carried by the surrounding typed authority adapter.
    /// Keeping this separate from the proof witness makes accidental point swaps fail before
    /// allocation. This equality is defense in depth only: ordered `EqBaseAir` and
    /// `SumcheckRoundsAir` independently consume the circuit-constrained source-point bus.
    pub expected_opening_point: &'a [EF],
}

/// Borrowed witness for one genuine ordered stacked-opening reduction.
#[derive(Clone, Copy, Debug)]
pub struct OrderedStackingReduction<'a> {
    pub proof: &'a StackingProof<BabyBearPoseidon2Config>,
    pub ordered_claims: &'a [Vec<OrderedStackingClaim>],
    /// The original prismalinear point (`r` in the stacking equations).
    pub opening_point: &'a [EF],
    pub preflight: OrderedStackingPreflight<'a>,
}

#[derive(Clone, Debug)]
struct OrderedStackingSlice {
    identity: OrderedStackingClaimIdentity,
    commit_idx: usize,
    stacked_col_idx: usize,
    row_idx: usize,
    log_height: usize,
    need_rot: bool,
    is_last_for_claim: bool,
}

#[derive(Clone, Debug)]
struct OrderedStackingReductionShape {
    layouts: Vec<StackedLayout>,
    slices: Vec<OrderedStackingSlice>,
    claims_per_commit: Vec<usize>,
}

/// Keygen-fixed layouts and claim identities for every ordered reduction handled by one module.
///
/// The outer index is the reduction/proof index. Within each reduction, layouts, rotation vectors
/// and identities are commitment-major. Within a commitment, identities follow
/// `StackedLayout::sorted_cols` exactly.
#[derive(Clone, Debug)]
pub struct OrderedStackingProfile {
    l_skip: usize,
    n_stack: usize,
    w_stack: usize,
    reductions: Vec<OrderedStackingReductionShape>,
    transcript_schedule: Option<OrderedStackingTranscriptSchedule>,
}

/// Verifier-owned placement of ordered reductions in the one global authority
/// transcript.
///
/// `first_claim_observation_tidx` includes every batch-statement and
/// per-reduction domain-prefix operation before the first claim.
/// `inter_reduction_prefix_len` is the exact number of transcript field
/// operations owned by the authority statement AIR between one stacking
/// reduction's `post_tidx` and the next reduction's first claim. Recursion does
/// not assign protocol meaning to this length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrderedStackingTranscriptSchedule {
    pub first_claim_observation_tidx: usize,
    pub inter_reduction_prefix_len: usize,
}

impl OrderedStackingTranscriptSchedule {
    #[must_use]
    pub fn next_claim_observation_tidx(self, previous_post_tidx: usize) -> Option<usize> {
        previous_post_tidx.checked_add(self.inter_reduction_prefix_len)
    }
}

/// Multi-opening statement dimensions emitted by one ordered reduction.
///
/// The point is the reduced Boolean-cube point used by ordinary WHIR:
/// `u_0, u_0^2, ..., u_0^(2^(l_skip - 1)), u_1, ..., u_n_stack`.
/// Commitment widths retain the exact commitment-major stacking layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderedStackingStatementShape {
    pub point_dimension: usize,
    pub commitment_widths: Vec<usize>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrderedStackingProfileError {
    Empty,
    ReductionCount,
    ModuleParameters,
    AlreadyConfigured,
    StatementOutputsRequireOrderedProfile,
    StatementOutputsAlreadyConfigured,
    MissingTranscriptSchedule,
    TranscriptScheduleAlreadyConfigured,
    CommitmentCount(usize),
    Layout(usize),
    Rotation(usize, usize),
    Identity(usize, usize),
    DuplicateIdentity(usize),
    WStack(usize),
    Overflow,
}

impl core::fmt::Display for OrderedStackingProfileError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "invalid ordered stacking profile: {self:?}")
    }
}

impl std::error::Error for OrderedStackingProfileError {}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrderedStackingTracegenError {
    ModuleNotConfigured,
    ReductionCount,
    RequiredHeightCount,
    RequiredHeight(usize),
    ProofShape(usize, &'static str),
    ClaimShape(usize),
    ClaimOrder(usize),
    Rotation(usize),
    OpeningPoint(usize),
    Preflight(usize),
    GlobalTranscriptOrder(usize),
    Overflow,
}

impl core::fmt::Display for OrderedStackingTracegenError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "invalid ordered stacking trace input: {self:?}")
    }
}

impl std::error::Error for OrderedStackingTracegenError {}

impl OrderedStackingProfile {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        l_skip: usize,
        n_stack: usize,
        w_stack: usize,
        layouts: Vec<Vec<StackedLayout>>,
        need_rot_per_reduction: Vec<Vec<Vec<bool>>>,
        claim_identities: Vec<Vec<Vec<OrderedStackingClaimIdentity>>>,
    ) -> Result<Self, OrderedStackingProfileError> {
        if layouts.is_empty() {
            return Err(OrderedStackingProfileError::Empty);
        }
        if layouts.len() != need_rot_per_reduction.len() || layouts.len() != claim_identities.len()
        {
            return Err(OrderedStackingProfileError::ReductionCount);
        }
        let log_stacked_height = l_skip
            .checked_add(n_stack)
            .ok_or(OrderedStackingProfileError::Overflow)?;
        let stacked_height = 1usize
            .checked_shl(log_stacked_height as u32)
            .ok_or(OrderedStackingProfileError::Overflow)?;
        let mut reductions = Vec::with_capacity(layouts.len());
        for (reduction_idx, ((layouts, need_rot), identities)) in layouts
            .into_iter()
            .zip(need_rot_per_reduction)
            .zip(claim_identities)
            .enumerate()
        {
            if layouts.is_empty()
                || layouts.len() != need_rot.len()
                || layouts.len() != identities.len()
            {
                return Err(OrderedStackingProfileError::CommitmentCount(reduction_idx));
            }
            let mut slices = Vec::new();
            let mut claims_per_commit = Vec::with_capacity(layouts.len());
            let mut seen = HashSet::new();
            let mut output_width = 0usize;
            for (commit_idx, ((layout, rotations), identities)) in
                layouts.iter().zip(&need_rot).zip(&identities).enumerate()
            {
                if layout.l_skip() != l_skip
                    || layout.height() != stacked_height
                    || layout.width() == 0
                    || layout.sorted_cols.is_empty()
                    || layout.sorted_cols.len() != identities.len()
                {
                    return Err(OrderedStackingProfileError::Layout(reduction_idx));
                }
                if rotations.len() != layout.mat_starts.len()
                    || layout
                        .sorted_cols
                        .iter()
                        .any(|(matrix_idx, _, _)| *matrix_idx >= rotations.len())
                {
                    return Err(OrderedStackingProfileError::Rotation(
                        reduction_idx,
                        commit_idx,
                    ));
                }
                output_width = output_width
                    .checked_add(layout.width())
                    .ok_or(OrderedStackingProfileError::Overflow)?;
                claims_per_commit.push(layout.sorted_cols.len());
                for (source_idx, ((matrix_idx, source_col_idx, slice), identity)) in
                    layout.sorted_cols.iter().zip(identities).enumerate()
                {
                    if identity.col_idx != *source_col_idx || !seen.insert(*identity) {
                        return Err(if identity.col_idx != *source_col_idx {
                            OrderedStackingProfileError::Identity(reduction_idx, commit_idx)
                        } else {
                            OrderedStackingProfileError::DuplicateIdentity(reduction_idx)
                        });
                    }
                    let is_last_for_claim = layout
                        .sorted_cols
                        .get(source_idx + 1)
                        .is_none_or(|(_, _, next)| next.col_idx != slice.col_idx);
                    slices.push(OrderedStackingSlice {
                        identity: *identity,
                        commit_idx,
                        stacked_col_idx: slice.col_idx,
                        row_idx: slice.row_idx,
                        log_height: slice.log_height(),
                        need_rot: rotations[*matrix_idx],
                        is_last_for_claim,
                    });
                }
            }
            if output_width > w_stack {
                return Err(OrderedStackingProfileError::WStack(reduction_idx));
            }
            reductions.push(OrderedStackingReductionShape {
                layouts,
                slices,
                claims_per_commit,
            });
        }
        Ok(Self {
            l_skip,
            n_stack,
            w_stack,
            reductions,
            transcript_schedule: None,
        })
    }

    /// Bind this profile to its exact position in the setup authority's one
    /// global transcript. This must be called before ordered module
    /// configuration.
    pub fn with_transcript_schedule(
        mut self,
        schedule: OrderedStackingTranscriptSchedule,
    ) -> Result<Self, OrderedStackingProfileError> {
        if self.transcript_schedule.is_some() {
            return Err(OrderedStackingProfileError::TranscriptScheduleAlreadyConfigured);
        }
        self.transcript_schedule = Some(schedule);
        Ok(self)
    }

    pub fn validate_module_params(
        &self,
        l_skip: usize,
        n_stack: usize,
        w_stack: usize,
    ) -> Result<(), OrderedStackingProfileError> {
        if (self.l_skip, self.n_stack, self.w_stack) != (l_skip, n_stack, w_stack) {
            return Err(OrderedStackingProfileError::ModuleParameters);
        }
        if self.transcript_schedule.is_none() {
            return Err(OrderedStackingProfileError::MissingTranscriptSchedule);
        }
        Ok(())
    }

    #[must_use]
    pub fn transcript_schedule(&self) -> Option<OrderedStackingTranscriptSchedule> {
        self.transcript_schedule
    }

    #[must_use]
    pub fn reduction_count(&self) -> usize {
        self.reductions.len()
    }

    /// Return the verifier-owned multi-opening dimensions for `reduction_idx`.
    /// No witness values or bus indices are exposed by this method.
    #[must_use]
    pub fn statement_shape(&self, reduction_idx: usize) -> Option<OrderedStackingStatementShape> {
        self.reductions
            .get(reduction_idx)
            .map(|reduction| OrderedStackingStatementShape {
                point_dimension: self.l_skip + self.n_stack,
                commitment_widths: reduction.layouts.iter().map(StackedLayout::width).collect(),
            })
    }

    /// Inclusive verifier-owned bounds for the original source opening-point
    /// dimension accepted by one reduction. The lower bound is the largest
    /// source hypercube dimension actually referenced by the fixed layouts;
    /// the upper bound is the ordinary stacking sumcheck capacity.
    #[must_use]
    pub fn source_point_dimension_bounds(&self, reduction_idx: usize) -> Option<(usize, usize)> {
        self.reductions.get(reduction_idx).map(|reduction| {
            let minimum = reduction
                .slices
                .iter()
                .map(|slice| slice.log_height.saturating_sub(self.l_skip))
                .max()
                .unwrap_or(0)
                + 1;
            (minimum, self.n_stack + 1)
        })
    }

    pub(crate) fn opening_preprocessed_trace(&self) -> RowMajorMatrix<F> {
        let width = core::mem::size_of::<OrderedOpeningClaimsPrepCols<u8>>();
        let valid_rows = self
            .reductions
            .iter()
            .map(|reduction| reduction.slices.len())
            .sum::<usize>();
        let height = valid_rows.next_power_of_two().max(2);
        let mut values = F::zero_vec(width * height);
        let mut cursor = 0usize;
        for (proof_idx, reduction) in self.reductions.iter().enumerate() {
            let last_main_idx = reduction
                .slices
                .iter()
                .enumerate()
                .skip(1)
                .find_map(|(index, slice)| (slice.identity.part_idx != 0).then_some(index - 1))
                .unwrap_or(reduction.slices.len() - 1);
            for (row_idx, slice) in reduction.slices.iter().enumerate() {
                let row: &mut OrderedOpeningClaimsPrepCols<F> =
                    values[cursor * width..(cursor + 1) * width].borrow_mut();
                row.active = F::ONE;
                row.proof_idx = F::from_usize(proof_idx);
                row.is_first = F::from_bool(row_idx == 0);
                row.is_last = F::from_bool(row_idx + 1 == reduction.slices.len());
                row.sort_idx = F::from_usize(slice.identity.sort_idx);
                row.part_idx = F::from_usize(slice.identity.part_idx);
                row.col_idx = F::from_usize(slice.identity.col_idx);
                row.need_rot = F::from_bool(slice.need_rot);
                row.is_main = F::from_bool(slice.identity.part_idx == 0);
                row.is_transition_main =
                    F::from_bool(row_idx + 1 != reduction.slices.len() && row_idx != last_main_idx);
                row.commit_idx = F::from_usize(slice.commit_idx);
                row.stacked_col_idx = F::from_usize(slice.stacked_col_idx);
                row.row_idx = F::from_usize(slice.row_idx);
                row.is_last_for_claim = F::from_bool(slice.is_last_for_claim);
                let n = slice.log_height as isize - self.l_skip as isize;
                let n_lift = n.max(0) as usize;
                row.hypercube_dim = if n.is_positive() {
                    F::from_usize(n_lift)
                } else {
                    -F::from_usize(n.unsigned_abs())
                };
                row.log_lifted_height = F::from_usize(n_lift + self.l_skip);
                row.lifted_height = F::from_usize(1 << (n_lift + self.l_skip));
                cursor += 1;
            }
        }
        RowMajorMatrix::new(values, width)
    }
}

/// Setup-fixed columns paired with `OpeningClaimsAir` in ordered mode.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub(crate) struct OrderedOpeningClaimsPrepCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub is_first: T,
    pub is_last: T,
    pub sort_idx: T,
    pub part_idx: T,
    pub col_idx: T,
    pub need_rot: T,
    pub is_main: T,
    pub is_transition_main: T,
    pub commit_idx: T,
    pub stacked_col_idx: T,
    pub row_idx: T,
    pub is_last_for_claim: T,
    pub hypercube_dim: T,
    pub log_lifted_height: T,
    pub lifted_height: T,
}

struct ValidatedReduction<'a> {
    shape: &'a OrderedStackingReductionShape,
    input: OrderedStackingReduction<'a>,
    claims: Vec<&'a OrderedStackingClaim>,
}

fn validate_reductions<'a>(
    profile: &'a OrderedStackingProfile,
    reductions: &'a [OrderedStackingReduction<'a>],
) -> Result<Vec<ValidatedReduction<'a>>, OrderedStackingTracegenError> {
    if reductions.len() != profile.reductions.len() {
        return Err(OrderedStackingTracegenError::ReductionCount);
    }
    let transcript_schedule = profile
        .transcript_schedule
        .ok_or(OrderedStackingTracegenError::ModuleNotConfigured)?;
    let validated = reductions
        .iter()
        .copied()
        .zip(&profile.reductions)
        .enumerate()
        .map(|(reduction_idx, (input, shape))| {
            let proof = input.proof;
            let expected_round0 = sumcheck_round0_deg(profile.l_skip, 2) + 1;
            if proof.univariate_round_coeffs.len() != expected_round0 {
                return Err(OrderedStackingTracegenError::ProofShape(
                    reduction_idx,
                    "univariate round coefficient count",
                ));
            }
            if proof.sumcheck_round_polys.len() != profile.n_stack {
                return Err(OrderedStackingTracegenError::ProofShape(
                    reduction_idx,
                    "sumcheck round count",
                ));
            }
            if proof.stacking_openings.len() != shape.layouts.len()
                || proof
                    .stacking_openings
                    .iter()
                    .zip(&shape.layouts)
                    .any(|(openings, layout)| openings.len() != layout.width())
            {
                return Err(OrderedStackingTracegenError::ProofShape(
                    reduction_idx,
                    "stacking opening dimensions",
                ));
            }
            if input.ordered_claims.len() != shape.claims_per_commit.len()
                || input
                    .ordered_claims
                    .iter()
                    .zip(&shape.claims_per_commit)
                    .any(|(claims, &expected)| claims.len() != expected)
            {
                return Err(OrderedStackingTracegenError::ClaimShape(reduction_idx));
            }
            let claims = input.ordered_claims.iter().flatten().collect_vec();
            if claims.len() != shape.slices.len() {
                return Err(OrderedStackingTracegenError::ClaimShape(reduction_idx));
            }
            for (claim, slice) in claims.iter().zip(&shape.slices) {
                if claim.identity != slice.identity {
                    return Err(OrderedStackingTracegenError::ClaimOrder(reduction_idx));
                }
                if claim.rotated.is_some() != slice.need_rot
                    || (!slice.need_rot && claim.rotated.unwrap_or(EF::ZERO) != EF::ZERO)
                {
                    return Err(OrderedStackingTracegenError::Rotation(reduction_idx));
                }
            }
            let max_n_lift = shape
                .slices
                .iter()
                .map(|slice| slice.log_height.saturating_sub(profile.l_skip))
                .max()
                .unwrap_or(0);
            if input.opening_point.is_empty()
                || input.opening_point.len() <= max_n_lift
                || input.opening_point.len() > profile.n_stack + 1
                || input.opening_point != input.preflight.expected_opening_point
                || input.preflight.stacking.sumcheck_rnd.len() != profile.n_stack + 1
            {
                return Err(OrderedStackingTracegenError::OpeningPoint(reduction_idx));
            }
            let preflight = input.preflight.stacking;
            let claim_limb_count = claims
                .len()
                .checked_mul(2 * D_EF)
                .ok_or(OrderedStackingTracegenError::Overflow)?;
            let post_lambda = input
                .preflight
                .claim_observation_tidx
                .checked_add(claim_limb_count)
                .and_then(|value| value.checked_add(D_EF))
                .ok_or(OrderedStackingTracegenError::Overflow)?;
            let post_univariate = post_lambda
                .checked_add(proof.univariate_round_coeffs.len() * D_EF + D_EF)
                .ok_or(OrderedStackingTracegenError::Overflow)?;
            let post_sumcheck = post_univariate
                .checked_add(proof.sumcheck_round_polys.len() * 3 * D_EF)
                .ok_or(OrderedStackingTracegenError::Overflow)?;
            let opening_count = proof.stacking_openings.iter().map(Vec::len).sum::<usize>();
            let post_openings = post_sumcheck
                .checked_add(opening_count * D_EF)
                .ok_or(OrderedStackingTracegenError::Overflow)?;
            let expected_eval = proof
                .univariate_round_coeffs
                .iter()
                .zip(preflight.sumcheck_rnd[0].powers())
                .map(|(&coefficient, power)| coefficient * power)
                .sum::<EF>();
            if preflight.intermediate_tidx != [post_lambda, post_univariate, post_sumcheck]
                || preflight.post_tidx != post_openings
                || preflight.univariate_poly_rand_eval != expected_eval
                || preflight.stacking_batching_challenge != EF::ZERO
                || preflight.mu_pow_witness != F::ZERO
                || preflight.mu_pow_sample != F::ZERO
            {
                return Err(OrderedStackingTracegenError::Preflight(reduction_idx));
            }
            Ok(ValidatedReduction {
                shape,
                input,
                claims,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if validated[0].input.preflight.claim_observation_tidx
        != transcript_schedule.first_claim_observation_tidx
    {
        return Err(OrderedStackingTracegenError::GlobalTranscriptOrder(0));
    }
    for (transition, pair) in validated.windows(2).enumerate() {
        let expected_next = transcript_schedule
            .next_claim_observation_tidx(pair[0].input.preflight.stacking.post_tidx)
            .ok_or(OrderedStackingTracegenError::Overflow)?;
        if pair[1].input.preflight.claim_observation_tidx != expected_next {
            return Err(OrderedStackingTracegenError::GlobalTranscriptOrder(
                transition + 1,
            ));
        }
    }
    Ok(validated)
}

#[allow(clippy::type_complexity)]
fn compute_coefficients(
    profile: &OrderedStackingProfile,
    reduction: &ValidatedReduction<'_>,
) -> (Vec<Vec<EF>>, Vec<(EF, EF, EF)>) {
    let preflight = reduction.input.preflight.stacking;
    let u = &preflight.sumcheck_rnd;
    let r = reduction.input.opening_point;
    let lambda = preflight.lambda;
    let mut coefficients = reduction
        .shape
        .layouts
        .iter()
        .map(|layout| vec![EF::ZERO; layout.width()])
        .collect_vec();
    let lambda_powers = lambda
        .powers()
        .take(reduction.shape.slices.len() * 2)
        .collect_vec();
    let mut per_slice = Vec::with_capacity(reduction.shape.slices.len());
    for (index, slice) in reduction.shape.slices.iter().enumerate() {
        let n = slice.log_height as isize - profile.l_skip as isize;
        let n_lift = n.max(0) as usize;
        let bits = (profile.l_skip + n_lift..profile.l_skip + profile.n_stack)
            .map(|bit| F::from_bool((slice.row_idx >> bit) & 1 == 1))
            .collect_vec();
        let eq_mle = eval_eq_mle(&u[n_lift + 1..], &bits);
        let indicator = eval_in_uni(profile.l_skip, n, u[0]);
        let negative_r = n
            .is_negative()
            .then(|| r[0].exp_power_of_2(n.unsigned_abs()));
        let (skip, r_slice): (usize, &[EF]) = if let Some(ref negative_r) = negative_r {
            (
                profile.l_skip.wrapping_add_signed(n),
                core::slice::from_ref(negative_r),
            )
        } else {
            (profile.l_skip, &r[..=n_lift])
        };
        let eq_prism = eval_eq_prism(skip, &u[..=n_lift], r_slice);
        let rot_kernel = eval_rot_kernel_prism(skip, &u[..=n_lift], r_slice);
        let mut batched = lambda_powers[2 * index] * eq_prism;
        if slice.need_rot {
            batched += lambda_powers[2 * index + 1] * rot_kernel;
        }
        coefficients[slice.commit_idx][slice.stacked_col_idx] += eq_mle * batched * indicator;
        per_slice.push((eq_prism * indicator, rot_kernel * indicator, eq_mle));
    }
    (coefficients, per_slice)
}

fn requested_height(
    required_heights: Option<&[usize]>,
    air_idx: usize,
    minimum: usize,
) -> Result<usize, OrderedStackingTracegenError> {
    let height = required_heights
        .map(|heights| heights[air_idx])
        .unwrap_or_else(|| minimum.next_power_of_two().max(2));
    if height < minimum {
        return Err(OrderedStackingTracegenError::RequiredHeight(air_idx));
    }
    Ok(height)
}

fn opening_trace(
    profile: &OrderedStackingProfile,
    reductions: &[ValidatedReduction<'_>],
    required_heights: Option<&[usize]>,
) -> Result<RowMajorMatrix<F>, OrderedStackingTracegenError> {
    let width = OpeningClaimsCols::<usize>::width();
    let minimum = reductions.iter().map(|record| record.claims.len()).sum();
    let fixed_height = profile.opening_preprocessed_trace().height();
    let height = requested_height(required_heights, 0, minimum)?;
    if height != fixed_height {
        return Err(OrderedStackingTracegenError::RequiredHeight(0));
    }
    let mut trace = F::zero_vec(height * width);
    let mut chunks = trace.chunks_mut(width);
    for (proof_idx, reduction) in reductions.iter().enumerate() {
        let (_, per_slice) = compute_coefficients(profile, reduction);
        let preflight = reduction.input.preflight.stacking;
        let mut lambda_powers = preflight.lambda.square().powers();
        let mut coefficient_accumulator = EF::ZERO;
        let mut s_0 = EF::ZERO;
        let last_main_idx = reduction
            .claims
            .iter()
            .enumerate()
            .skip(1)
            .find_map(|(index, claim)| (claim.identity.part_idx != 0).then_some(index - 1))
            .unwrap_or(reduction.claims.len() - 1);
        for (row_idx, ((claim, slice), (eq_in, k_rot_in, eq_bits))) in reduction
            .claims
            .iter()
            .zip(&reduction.shape.slices)
            .zip(per_slice)
            .enumerate()
        {
            let cols: &mut OpeningClaimsCols<F> = chunks.next().unwrap().borrow_mut();
            cols.proof_idx = F::from_usize(proof_idx);
            cols.is_valid = F::ONE;
            cols.is_first = F::from_bool(row_idx == 0);
            cols.is_last = F::from_bool(row_idx + 1 == reduction.claims.len());
            cols.sort_idx = F::from_usize(claim.identity.sort_idx);
            cols.part_idx = F::from_usize(claim.identity.part_idx);
            cols.col_idx = F::from_usize(claim.identity.col_idx);
            cols.col_claim
                .copy_from_slice(claim.current.as_basis_coefficients_slice());
            let rotated = claim.rotated.unwrap_or(EF::ZERO);
            cols.rot_claim
                .copy_from_slice(rotated.as_basis_coefficients_slice());
            cols.need_rot = F::from_bool(slice.need_rot);
            cols.is_main = F::from_bool(claim.identity.part_idx == 0);
            cols.is_transition_main =
                F::from_bool(row_idx + 1 != reduction.claims.len() && row_idx != last_main_idx);
            let n = slice.log_height as isize - profile.l_skip as isize;
            let n_lift = n.max(0) as usize;
            cols.hypercube_dim = if n.is_positive() {
                F::from_usize(n_lift)
            } else {
                -F::from_usize(n.unsigned_abs())
            };
            cols.log_lifted_height = F::from_usize(n_lift + profile.l_skip);
            cols.lifted_height = F::from_usize(1 << (n_lift + profile.l_skip));
            cols.lifted_height_inv = cols.lifted_height.inverse();
            cols.tidx = F::from_usize(
                reduction.input.preflight.claim_observation_tidx + row_idx * 2 * D_EF,
            );
            cols.lambda
                .copy_from_slice(preflight.lambda.as_basis_coefficients_slice());
            let lambda_power = lambda_powers.next().unwrap();
            cols.lambda_pow
                .copy_from_slice(lambda_power.as_basis_coefficients_slice());
            cols.commit_idx = F::from_usize(slice.commit_idx);
            cols.stacked_col_idx = F::from_usize(slice.stacked_col_idx);
            cols.row_idx = F::from_usize(slice.row_idx);
            cols.is_last_for_claim = F::from_bool(slice.is_last_for_claim);
            cols.eq_in
                .copy_from_slice(eq_in.as_basis_coefficients_slice());
            cols.k_rot_in
                .copy_from_slice(k_rot_in.as_basis_coefficients_slice());
            if slice.need_rot {
                cols.k_rot_in_when_needed
                    .copy_from_slice(k_rot_in.as_basis_coefficients_slice());
            }
            cols.eq_bits
                .copy_from_slice(eq_bits.as_basis_coefficients_slice());
            let lambda_pow_eq_bits = lambda_power * eq_bits;
            cols.lambda_pow_eq_bits
                .copy_from_slice(lambda_pow_eq_bits.as_basis_coefficients_slice());
            coefficient_accumulator += lambda_pow_eq_bits
                * (eq_in + preflight.lambda * if slice.need_rot { k_rot_in } else { EF::ZERO });
            cols.stacking_claim_coefficient
                .copy_from_slice(coefficient_accumulator.as_basis_coefficients_slice());
            if slice.is_last_for_claim {
                coefficient_accumulator = EF::ZERO;
            }
            s_0 += lambda_power * (claim.current + preflight.lambda * rotated);
            cols.s_0.copy_from_slice(s_0.as_basis_coefficients_slice());
        }
    }
    let padding_idx = F::from_usize(reductions.len());
    let mut chunks = chunks.peekable();
    while let Some(chunk) = chunks.next() {
        let cols: &mut OpeningClaimsCols<F> = chunk.borrow_mut();
        cols.proof_idx = padding_idx;
        if chunks.peek().is_none() {
            cols.is_last = F::ONE;
        }
    }
    Ok(RowMajorMatrix::new(trace, width))
}

fn univariate_trace(
    profile: &OrderedStackingProfile,
    reductions: &[ValidatedReduction<'_>],
    required_heights: Option<&[usize]>,
) -> Result<RowMajorMatrix<F>, OrderedStackingTracegenError> {
    let width = UnivariateRoundCols::<usize>::width();
    let minimum = reductions
        .iter()
        .map(|record| record.input.proof.univariate_round_coeffs.len())
        .sum();
    let height = requested_height(required_heights, 1, minimum)?;
    let mut combined = Vec::with_capacity(height * width);
    for (proof_idx, reduction) in reductions.iter().enumerate() {
        let coefficients = &reduction.input.proof.univariate_round_coeffs;
        let u_0 = reduction.input.preflight.stacking.sumcheck_rnd[0];
        let d_card = 1usize << profile.l_skip;
        let mut s_0_sum_over_d = coefficients[0] * F::from_usize(d_card);
        let mut poly_rand_eval = EF::ZERO;
        for (index, (&coefficient, u_0_power)) in coefficients.iter().zip(u_0.powers()).enumerate()
        {
            let start = combined.len();
            combined.resize(start + width, F::ZERO);
            let cols: &mut UnivariateRoundCols<F> = combined[start..].borrow_mut();
            cols.proof_idx = F::from_usize(proof_idx);
            cols.is_valid = F::ONE;
            cols.is_first = F::from_bool(index == 0);
            cols.is_last = F::from_bool(index + 1 == coefficients.len());
            cols.tidx = F::from_usize(
                reduction.input.preflight.stacking.intermediate_tidx[0] + index * D_EF,
            );
            cols.u_0.copy_from_slice(u_0.as_basis_coefficients_slice());
            cols.u_0_pow
                .copy_from_slice(u_0_power.as_basis_coefficients_slice());
            cols.coeff
                .copy_from_slice(coefficient.as_basis_coefficients_slice());
            cols.coeff_idx = F::from_usize(index);
            if index == d_card {
                s_0_sum_over_d += coefficient * F::from_usize(d_card);
                cols.coeff_is_d = F::ONE;
            }
            cols.s_0_sum_over_d
                .copy_from_slice(s_0_sum_over_d.as_basis_coefficients_slice());
            poly_rand_eval += coefficient * u_0_power;
            cols.poly_rand_eval
                .copy_from_slice(poly_rand_eval.as_basis_coefficients_slice());
        }
    }
    let valid_rows = combined.len() / width;
    combined.resize(height * width, F::ZERO);
    let padding_idx = F::from_usize(reductions.len());
    let mut chunks = combined[valid_rows * width..].chunks_mut(width).peekable();
    while let Some(chunk) = chunks.next() {
        let cols: &mut UnivariateRoundCols<F> = chunk.borrow_mut();
        cols.proof_idx = padding_idx;
        if chunks.peek().is_none() {
            cols.is_last = F::ONE;
        }
    }
    Ok(RowMajorMatrix::new(combined, width))
}

fn sumcheck_trace(
    profile: &OrderedStackingProfile,
    reductions: &[ValidatedReduction<'_>],
    required_heights: Option<&[usize]>,
) -> Result<RowMajorMatrix<F>, OrderedStackingTracegenError> {
    let width = SumcheckRoundsCols::<usize>::width();
    let minimum = reductions.len() * profile.n_stack;
    let height = requested_height(required_heights, 2, minimum)?;
    let mut combined = Vec::with_capacity(height * width);
    for (proof_idx, reduction) in reductions.iter().enumerate() {
        let mut eq_mults = vec![0usize; profile.n_stack];
        let mut u_mults = vec![0usize; profile.n_stack];
        let mut seen_bits = HashSet::<(usize, usize)>::new();
        for slice in &reduction.shape.slices {
            let n_lift = slice.log_height.saturating_sub(profile.l_skip);
            if n_lift > 0 {
                eq_mults[n_lift - 1] += 1;
            }
            let b_value = slice.row_idx >> (n_lift + profile.l_skip);
            let total_bits = profile.n_stack - n_lift;
            for bits in (1..=total_bits).rev() {
                let shifted = b_value >> (total_bits - bits);
                if seen_bits.insert((shifted, bits)) {
                    u_mults[profile.n_stack - bits] += 1;
                } else {
                    break;
                }
            }
        }
        let preflight = reduction.input.preflight.stacking;
        let omega = F::two_adic_generator(profile.l_skip);
        let u_0 = preflight.sumcheck_rnd[0];
        let r_0 = reduction.input.opening_point[0];
        let eq_prism_base = eval_eq_uni(profile.l_skip, u_0, r_0);
        let eq_cube_base = eval_eq_uni(profile.l_skip, u_0, r_0 * omega);
        let rot_cube_base = eval_eq_uni_at_one(profile.l_skip, u_0)
            * eval_eq_uni_at_one(profile.l_skip, r_0 * omega);
        let mut s_eval_at_u = preflight.univariate_poly_rand_eval;
        let mut eq_cube = EF::ONE;
        let mut r_not_u_prod = EF::ONE;
        let mut rot_cube_minus_prod = EF::ZERO;
        for (round, (evaluations, &u_round)) in reduction
            .input
            .proof
            .sumcheck_round_polys
            .iter()
            .zip(&preflight.sumcheck_rnd[1..])
            .enumerate()
        {
            let start = combined.len();
            combined.resize(start + width, F::ZERO);
            let cols: &mut SumcheckRoundsCols<F> = combined[start..].borrow_mut();
            let s_eval_at_0 = s_eval_at_u - evaluations[0];
            s_eval_at_u = interpolate_quadratic_at_012(
                &[s_eval_at_0, evaluations[0], evaluations[1]],
                u_round,
            );
            cols.proof_idx = F::from_usize(proof_idx);
            cols.is_valid = F::ONE;
            cols.is_first = F::from_bool(round == 0);
            cols.is_last = F::from_bool(round + 1 == profile.n_stack);
            cols.round = F::from_usize(round + 1);
            cols.tidx = F::from_usize(preflight.intermediate_tidx[1] + round * 3 * D_EF);
            cols.s_eval_at_0
                .copy_from_slice(s_eval_at_0.as_basis_coefficients_slice());
            cols.s_eval_at_1
                .copy_from_slice(evaluations[0].as_basis_coefficients_slice());
            cols.s_eval_at_2
                .copy_from_slice(evaluations[1].as_basis_coefficients_slice());
            cols.s_eval_at_u
                .copy_from_slice(s_eval_at_u.as_basis_coefficients_slice());
            cols.u_round
                .copy_from_slice(u_round.as_basis_coefficients_slice());
            let r_round = reduction
                .input
                .opening_point
                .get(round + 1)
                .copied()
                .unwrap_or(EF::ZERO);
            if round + 1 < reduction.input.opening_point.len() {
                cols.r_round
                    .copy_from_slice(r_round.as_basis_coefficients_slice());
                cols.has_r = F::ONE;
            }
            cols.u_mult = F::from_usize(u_mults[round]);
            cols.eq_prism_base
                .copy_from_slice(eq_prism_base.as_basis_coefficients_slice());
            cols.eq_cube_base
                .copy_from_slice(eq_cube_base.as_basis_coefficients_slice());
            cols.rot_cube_base
                .copy_from_slice(rot_cube_base.as_basis_coefficients_slice());
            let u_not_r = u_round * (EF::ONE - r_round);
            let r_not_u = r_round * (EF::ONE - u_round);
            let next_eq = EF::ONE - u_not_r - r_not_u;
            eq_cube *= next_eq;
            cols.eq_cube
                .copy_from_slice(eq_cube.as_basis_coefficients_slice());
            rot_cube_minus_prod = rot_cube_minus_prod * next_eq + u_not_r * r_not_u_prod;
            r_not_u_prod *= r_not_u;
            cols.r_not_u_prod
                .copy_from_slice(r_not_u_prod.as_basis_coefficients_slice());
            cols.rot_cube_minus_prod
                .copy_from_slice(rot_cube_minus_prod.as_basis_coefficients_slice());
            cols.eq_rot_mult = F::from_usize(eq_mults[round]);
        }
    }
    let valid_rows = combined.len() / width;
    combined.resize(height * width, F::ZERO);
    let padding_idx = F::from_usize(reductions.len());
    let mut chunks = combined[valid_rows * width..].chunks_mut(width).peekable();
    while let Some(chunk) = chunks.next() {
        let cols: &mut SumcheckRoundsCols<F> = chunk.borrow_mut();
        cols.proof_idx = padding_idx;
        if chunks.peek().is_none() {
            cols.is_last = F::ONE;
        }
    }
    Ok(RowMajorMatrix::new(combined, width))
}

fn stacking_claims_trace(
    profile: &OrderedStackingProfile,
    reductions: &[ValidatedReduction<'_>],
    required_heights: Option<&[usize]>,
) -> Result<RowMajorMatrix<F>, OrderedStackingTracegenError> {
    let width = StackingClaimsCols::<usize>::width();
    let minimum = reductions.len() * profile.w_stack;
    let height = requested_height(required_heights, 3, minimum)?;
    let mut trace = F::zero_vec(height * width);
    let mut chunks = trace.chunks_mut(width);
    for (proof_idx, reduction) in reductions.iter().enumerate() {
        let claims = reduction
            .input
            .proof
            .stacking_openings
            .iter()
            .enumerate()
            .flat_map(|(commit_idx, openings)| {
                openings
                    .iter()
                    .enumerate()
                    .map(move |(column_idx, &value)| (commit_idx, column_idx, value))
            })
            .collect_vec();
        let coefficients = compute_coefficients(profile, reduction)
            .0
            .into_iter()
            .flatten()
            .collect_vec();
        let mut final_s_eval = EF::ZERO;
        let mu = reduction
            .input
            .preflight
            .stacking
            .stacking_batching_challenge;
        let mu_powers = mu.powers().take(claims.len()).collect_vec();
        let mut whir_claim = EF::ZERO;
        for (index, ((commit_idx, column_idx, claim), coefficient)) in
            claims.iter().copied().zip(coefficients).enumerate()
        {
            let cols: &mut StackingClaimsCols<F> = chunks.next().unwrap().borrow_mut();
            cols.proof_idx = F::from_usize(proof_idx);
            cols.is_valid = F::ONE;
            cols.is_first = F::from_bool(index == 0);
            cols.is_last =
                F::from_bool(claims.len() == profile.w_stack && index + 1 == claims.len());
            cols.commit_idx = F::from_usize(commit_idx);
            cols.stacked_col_idx = F::from_usize(column_idx);
            cols.global_col_idx = F::from_usize(index);
            cols.tidx = F::from_usize(
                reduction.input.preflight.stacking.intermediate_tidx[2] + index * D_EF,
            );
            cols.mu.copy_from_slice(mu.as_basis_coefficients_slice());
            cols.mu_pow
                .copy_from_slice(mu_powers[index].as_basis_coefficients_slice());
            cols.mu_pow_witness = reduction.input.preflight.stacking.mu_pow_witness;
            cols.mu_pow_sample = reduction.input.preflight.stacking.mu_pow_sample;
            cols.stacking_claim
                .copy_from_slice(claim.as_basis_coefficients_slice());
            cols.claim_coefficient
                .copy_from_slice(coefficient.as_basis_coefficients_slice());
            final_s_eval += claim * coefficient;
            cols.final_s_eval
                .copy_from_slice(final_s_eval.as_basis_coefficients_slice());
            whir_claim += mu_powers[index] * claim;
            cols.whir_claim
                .copy_from_slice(whir_claim.as_basis_coefficients_slice());
        }
        for index in claims.len()..profile.w_stack {
            let cols: &mut StackingClaimsCols<F> = chunks.next().unwrap().borrow_mut();
            cols.proof_idx = F::from_usize(proof_idx);
            cols.is_padding = F::ONE;
            cols.is_last = F::from_bool(index + 1 == profile.w_stack);
            cols.global_col_idx = F::from_usize(index);
        }
    }
    let padding_idx = F::from_usize(reductions.len());
    let mut chunks = chunks.peekable();
    while let Some(chunk) = chunks.next() {
        let cols: &mut StackingClaimsCols<F> = chunk.borrow_mut();
        cols.proof_idx = padding_idx;
        if chunks.peek().is_none() {
            cols.is_last = F::ONE;
        }
    }
    Ok(RowMajorMatrix::new(trace, width))
}

fn eq_base_trace(
    profile: &OrderedStackingProfile,
    reductions: &[ValidatedReduction<'_>],
    required_heights: Option<&[usize]>,
) -> Result<RowMajorMatrix<F>, OrderedStackingTracegenError> {
    let width = EqBaseCols::<usize>::width();
    let rows_per_reduction = profile.l_skip + 1;
    let minimum = reductions.len() * rows_per_reduction;
    let height = requested_height(required_heights, 4, minimum)?;
    let mut combined = Vec::with_capacity(height * width);
    for (proof_idx, reduction) in reductions.iter().enumerate() {
        let mut multiplicities = vec![0usize; rows_per_reduction];
        for slice in &reduction.shape.slices {
            if slice.log_height <= profile.l_skip {
                multiplicities[profile.l_skip - slice.log_height] += 1;
            }
        }
        let preflight = reduction.input.preflight.stacking;
        let omega = F::two_adic_generator(profile.l_skip);
        let mut u = preflight.sumcheck_rnd[0];
        let mut r = reduction.input.opening_point[0];
        let mut r_omega = r * omega;
        let mut prod_u_r = u * (u + r);
        let mut prod_u_r_omega = u * (u + r_omega);
        let mut prod_u_1 = u + F::ONE;
        let mut prod_r_omega_1 = r_omega + F::ONE;
        let mut in_prod = EF::ONE;
        let u_powers = u.exp_powers_of_2().take(rows_per_reduction).collect_vec();
        for row_idx in 0..rows_per_reduction {
            let start = combined.len();
            combined.resize(start + width, F::ZERO);
            let cols: &mut EqBaseCols<F> = combined[start..].borrow_mut();
            let is_last = row_idx + 1 == rows_per_reduction;
            cols.proof_idx = F::from_usize(proof_idx);
            cols.is_valid = F::ONE;
            cols.is_first = F::from_bool(row_idx == 0);
            cols.is_last = F::from_bool(is_last);
            cols.row_idx = F::from_usize(row_idx);
            cols.u_pow.copy_from_slice(u.as_basis_coefficients_slice());
            cols.r_pow.copy_from_slice(r.as_basis_coefficients_slice());
            cols.r_omega_pow
                .copy_from_slice(r_omega.as_basis_coefficients_slice());
            cols.prod_u_r
                .copy_from_slice(prod_u_r.as_basis_coefficients_slice());
            cols.prod_u_r_omega
                .copy_from_slice(prod_u_r_omega.as_basis_coefficients_slice());
            cols.prod_u_1
                .copy_from_slice(prod_u_1.as_basis_coefficients_slice());
            cols.prod_r_omega_1
                .copy_from_slice(prod_r_omega_1.as_basis_coefficients_slice());
            if is_last {
                cols.mult = F::from_usize(multiplicities[0]);
            }
            let skip = profile.l_skip - row_idx;
            let u_pow_rev = u_powers[skip];
            if row_idx != 0 {
                in_prod *= u_pow_rev + F::ONE;
                cols.eq_neg.copy_from_slice(
                    (eval_eq_uni(skip, preflight.sumcheck_rnd[0], r) * F::from_usize(1 << skip))
                        .as_basis_coefficients_slice(),
                );
                cols.k_rot_neg.copy_from_slice(
                    (eval_rot_kernel_prism(skip, &[preflight.sumcheck_rnd[0]], &[r])
                        * F::from_usize(1 << skip))
                    .as_basis_coefficients_slice(),
                );
                cols.mult_neg = F::from_usize(multiplicities[row_idx]);
            }
            cols.u_pow_rev
                .copy_from_slice(u_pow_rev.as_basis_coefficients_slice());
            cols.in_prod
                .copy_from_slice(in_prod.as_basis_coefficients_slice());
            u *= u;
            r *= r;
            r_omega *= r_omega;
            prod_u_r *= u + r;
            prod_u_r_omega *= u + r_omega;
            prod_u_1 *= u + F::ONE;
            prod_r_omega_1 *= r_omega + F::ONE;
        }
    }
    let valid_rows = combined.len() / width;
    combined.resize(height * width, F::ZERO);
    let padding_idx = F::from_usize(reductions.len());
    let mut chunks = combined[valid_rows * width..].chunks_mut(width).peekable();
    while let Some(chunk) = chunks.next() {
        let cols: &mut EqBaseCols<F> = chunk.borrow_mut();
        cols.proof_idx = padding_idx;
        if chunks.peek().is_none() {
            cols.is_last = F::ONE;
        }
    }
    Ok(RowMajorMatrix::new(combined, width))
}

/// Generate the ordinary `EqNegAir` witness paired with ordered `EqBaseAir`.
///
/// Ordered setup authority deliberately has no synthetic batch-constraint
/// [`crate::system::Preflight`].  Its two inputs are already present in the
/// validated reduction: `u` is the first stacking sumcheck challenge and `r`
/// is the first coordinate of the source opening point.  This is otherwise the
/// same recurrence as [`crate::batch_constraint::eq_airs::EqNegTraceGenerator`].
/// Selector multiplicities stay zero because this authority subcircuit has no
/// selector consumers.
fn eq_neg_trace(
    profile: &OrderedStackingProfile,
    reductions: &[ValidatedReduction<'_>],
    required_height: Option<usize>,
) -> Result<RowMajorMatrix<F>, OrderedStackingTracegenError> {
    let inputs = reductions
        .iter()
        .map(|reduction| {
            (
                reduction.input.preflight.stacking.sumcheck_rnd[0],
                reduction.input.opening_point[0],
            )
        })
        .collect_vec();
    eq_neg_trace_from_inputs(profile.l_skip, &inputs, required_height)
}

fn eq_neg_trace_from_inputs(
    l_skip: usize,
    inputs: &[(EF, EF)],
    required_height: Option<usize>,
) -> Result<RowMajorMatrix<F>, OrderedStackingTracegenError> {
    let width = EqNegCols::<usize>::width();
    if l_skip == 0 {
        let height = required_height.unwrap_or(0);
        return Ok(RowMajorMatrix::new(F::zero_vec(height * width), width));
    }

    let rows_per_reduction = l_skip
        .checked_mul(
            l_skip
                .checked_add(3)
                .ok_or(OrderedStackingTracegenError::Overflow)?,
        )
        .and_then(|value| value.checked_div(2))
        .ok_or(OrderedStackingTracegenError::Overflow)?;
    let minimum = inputs
        .len()
        .checked_mul(rows_per_reduction)
        .ok_or(OrderedStackingTracegenError::Overflow)?;
    let height = match required_height {
        Some(height) if height < minimum => {
            return Err(OrderedStackingTracegenError::RequiredHeight(6));
        }
        Some(height) => height,
        None => minimum.next_power_of_two(),
    };
    let trace_len = height
        .checked_mul(width)
        .ok_or(OrderedStackingTracegenError::Overflow)?;
    let mut trace = F::zero_vec(trace_len);
    let mut chunks = trace.chunks_exact_mut(width);

    for (proof_idx, &(initial_u, mut initial_r)) in inputs.iter().enumerate() {
        let initial_omega = F::two_adic_generator(l_skip);
        let mut initial_r_omega = initial_r * initial_omega;

        for neg_hypercube in 0..l_skip {
            let mut u = initial_u;
            let mut r = initial_r;
            let mut r_omega = initial_r_omega;

            let mut prod_u_r = u * (u + r);
            let mut prod_u_r_omega = u * (u + r_omega);
            let mut prod_1_r = r + F::ONE;
            let mut prod_1_r_omega = r_omega + F::ONE;

            let one_half = F::ONE.halve();
            let mut one_half_pow = one_half;

            for row_idx in 0..=l_skip - neg_hypercube {
                let cols: &mut EqNegCols<F> = chunks
                    .next()
                    .expect("validated ordered EqNeg row count must fit")
                    .borrow_mut();
                let is_first_hypercube = row_idx == 0;
                let is_last_hypercube = row_idx == l_skip - neg_hypercube;

                cols.proof_idx = F::from_usize(proof_idx);
                cols.is_valid = F::ONE;
                cols.is_first = F::from_bool(neg_hypercube == 0 && is_first_hypercube);
                cols.is_last = F::from_bool(neg_hypercube + 1 == l_skip && is_last_hypercube);
                cols.neg_hypercube = F::from_usize(neg_hypercube);
                cols.neg_hypercube_nz_inv = cols.neg_hypercube.try_inverse().unwrap_or_default();
                cols.row_index = F::from_usize(row_idx);
                cols.is_first_hypercube = F::from_bool(is_first_hypercube);
                cols.is_last_hypercube = F::from_bool(is_last_hypercube);

                cols.u_pow.copy_from_slice(u.as_basis_coefficients_slice());
                cols.r_pow.copy_from_slice(r.as_basis_coefficients_slice());
                cols.r_omega_pow
                    .copy_from_slice(r_omega.as_basis_coefficients_slice());
                u *= u;
                r *= r;
                r_omega *= r_omega;

                cols.prod_u_r
                    .copy_from_slice(prod_u_r.as_basis_coefficients_slice());
                cols.prod_u_r_omega
                    .copy_from_slice(prod_u_r_omega.as_basis_coefficients_slice());
                prod_u_r *= u + r;
                prod_u_r_omega *= u + r_omega;

                cols.prod_1_r
                    .copy_from_slice(prod_1_r.as_basis_coefficients_slice());
                cols.prod_1_r_omega
                    .copy_from_slice(prod_1_r_omega.as_basis_coefficients_slice());
                cols.one_half_pow = one_half_pow;
                debug_assert_eq!(
                    prod_1_r * one_half_pow,
                    eval_eq_uni_at_one(row_idx + 1, initial_r)
                );
                debug_assert_eq!(
                    prod_1_r_omega * one_half_pow,
                    eval_eq_uni_at_one(row_idx + 1, initial_r_omega)
                );
                prod_1_r *= r + F::ONE;
                prod_1_r_omega *= r_omega + F::ONE;
                one_half_pow *= one_half;

                // `sel_first_count` and `sel_last_trans_count` intentionally
                // remain zero in the zero-initialized matrix.
            }

            initial_r *= initial_r;
            initial_r_omega *= initial_r_omega;
        }
    }

    for chunk in chunks {
        let cols: &mut EqNegCols<F> = chunk.borrow_mut();
        cols.proof_idx = F::from_usize(inputs.len());
    }

    Ok(RowMajorMatrix::new(trace, width))
}

fn eq_bits_trace(
    profile: &OrderedStackingProfile,
    reductions: &[ValidatedReduction<'_>],
    required_heights: Option<&[usize]>,
) -> Result<RowMajorMatrix<F>, OrderedStackingTracegenError> {
    let width = EqBitsCols::<usize>::width();
    let mut traces = Vec::with_capacity(reductions.len());
    let mut minimum = 0usize;
    for (proof_idx, reduction) in reductions.iter().enumerate() {
        let u = &reduction.input.preflight.stacking.sumcheck_rnd[1..];
        let mut values = HashMap::<(usize, usize), (EF, EF, usize, usize)>::new();
        let mut base_internal_mult = 0usize;
        let mut base_external_mult = 0usize;
        for slice in &reduction.shape.slices {
            let n_lift = slice.log_height.saturating_sub(profile.l_skip);
            let b_value = slice.row_idx >> (n_lift + profile.l_skip);
            let total_bits = profile.n_stack - n_lift;
            if total_bits == 0 {
                base_external_mult += 1;
                continue;
            }
            let (mut latest_eval, latest_bits) = {
                let mut found = (EF::ONE, 0);
                for bits in (1..=total_bits).rev() {
                    let shifted = b_value >> (total_bits - bits);
                    if let Some((_, eval, internal_mult, external_mult)) =
                        values.get_mut(&(shifted, bits))
                    {
                        if bits < total_bits {
                            let child = b_value >> (total_bits - bits - 1);
                            *internal_mult += 1 + (child & 1);
                        } else {
                            *external_mult += 1;
                        }
                        found = (*eval, bits);
                        break;
                    }
                }
                found
            };
            if latest_bits == total_bits {
                continue;
            }
            if latest_bits == 0 {
                base_internal_mult += 1 + (b_value >> (total_bits - 1));
            }
            for bits in latest_bits + 1..=total_bits {
                let shifted = b_value >> (total_bits - bits);
                let b_lsb = EF::from_usize(shifted & 1);
                let u_value = u[profile.n_stack - bits];
                let next_eval =
                    latest_eval * (EF::ONE + EF::TWO * b_lsb * u_value - b_lsb - u_value);
                let is_last = bits == total_bits;
                values.insert(
                    (shifted, bits),
                    (
                        latest_eval,
                        next_eval,
                        usize::from(!is_last),
                        usize::from(is_last),
                    ),
                );
                latest_eval = next_eval;
            }
        }
        let row_count = values.len() + 1;
        minimum += row_count;
        let mut trace = F::zero_vec(row_count * width);
        {
            let cols: &mut EqBitsCols<F> = trace[..width].borrow_mut();
            cols.proof_idx = F::from_usize(proof_idx);
            cols.is_valid = F::ONE;
            cols.is_first = F::ONE;
            cols.sub_eval[0] = F::ONE;
            cols.internal_child_flag = F::from_usize(base_internal_mult);
            cols.external_mult = F::from_usize(base_external_mult);
        }
        for ((&(b_value, bits), &(sub_eval, _, internal_mult, external_mult)), chunk) in values
            .iter()
            .sorted_by_key(|(key, _)| **key)
            .zip(trace.chunks_mut(width).skip(1))
        {
            let cols: &mut EqBitsCols<F> = chunk.borrow_mut();
            cols.proof_idx = F::from_usize(proof_idx);
            cols.is_valid = F::ONE;
            cols.internal_child_flag = F::from_usize(internal_mult);
            cols.external_mult = F::from_usize(external_mult);
            cols.sub_b_value = F::from_usize(b_value >> 1);
            cols.num_bits = F::from_usize(bits);
            cols.b_lsb = F::from_usize(b_value & 1);
            cols.u_val
                .copy_from_slice(u[profile.n_stack - bits].as_basis_coefficients_slice());
            cols.sub_eval
                .copy_from_slice(sub_eval.as_basis_coefficients_slice());
        }
        traces.push(trace);
    }
    let height = requested_height(required_heights, 5, minimum)?;
    let mut combined = Vec::with_capacity(height * width);
    for trace in traces {
        combined.extend(trace);
    }
    combined.resize(height * width, F::ZERO);
    let padding_idx = F::from_usize(reductions.len());
    for chunk in combined[minimum * width..].chunks_mut(width) {
        let cols: &mut EqBitsCols<F> = chunk.borrow_mut();
        cols.proof_idx = padding_idx;
    }
    Ok(RowMajorMatrix::new(combined, width))
}

pub(super) fn generate_ordered_reduction_ctxs<SC: StarkProtocolConfig<F = F>>(
    profile: &OrderedStackingProfile,
    reductions: &[OrderedStackingReduction<'_>],
    required_heights: Option<&[usize]>,
) -> Result<Vec<AirProvingContext<CpuBackend<SC>>>, OrderedStackingTracegenError> {
    if required_heights.is_some_and(|heights| heights.len() != 6) {
        return Err(OrderedStackingTracegenError::RequiredHeightCount);
    }
    let validated = validate_reductions(profile, reductions)?;
    let traces = [
        opening_trace(profile, &validated, required_heights)?,
        univariate_trace(profile, &validated, required_heights)?,
        sumcheck_trace(profile, &validated, required_heights)?,
        stacking_claims_trace(profile, &validated, required_heights)?,
        eq_base_trace(profile, &validated, required_heights)?,
        eq_bits_trace(profile, &validated, required_heights)?,
    ];
    Ok(traces
        .into_iter()
        .map(AirProvingContext::simple_no_pis)
        .collect())
}

pub(super) fn generate_ordered_eq_neg_ctx<SC: StarkProtocolConfig<F = F>>(
    profile: &OrderedStackingProfile,
    reductions: &[OrderedStackingReduction<'_>],
    required_height: Option<usize>,
) -> Result<AirProvingContext<CpuBackend<SC>>, OrderedStackingTracegenError> {
    let validated = validate_reductions(profile, reductions)?;
    Ok(AirProvingContext::simple_no_pis(eq_neg_trace(
        profile,
        &validated,
        required_height,
    )?))
}

#[cfg(test)]
mod tests {
    use core::borrow::Borrow;

    use super::*;

    #[test]
    fn ordered_eq_neg_uses_explicit_u_and_r_with_zero_selectors() {
        let u = EF::from_u32(7);
        let r = EF::from_u32(11);
        let trace = eq_neg_trace_from_inputs(3, &[(u, r)], None).unwrap();

        assert_eq!(trace.height(), 16);
        for row_idx in 0..9 {
            let row = trace.row_slice(row_idx).unwrap();
            let row: &EqNegCols<F> = (&*row).borrow();
            assert_eq!(row.is_valid, F::ONE);
            assert_eq!(row.sel_first_count, F::ZERO);
            assert_eq!(row.sel_last_trans_count, F::ZERO);
        }

        let first = trace.row_slice(0).unwrap();
        let first: &EqNegCols<F> = (&*first).borrow();
        assert_eq!(&first.u_pow, u.as_basis_coefficients_slice());
        assert_eq!(&first.r_pow, r.as_basis_coefficients_slice());

        // Section starts use r, r^2, and r^4 respectively.
        let second_section = trace.row_slice(4).unwrap();
        let second_section: &EqNegCols<F> = (&*second_section).borrow();
        let third_section = trace.row_slice(7).unwrap();
        let third_section: &EqNegCols<F> = (&*third_section).borrow();
        assert_eq!(&second_section.r_pow, (r * r).as_basis_coefficients_slice());
        assert_eq!(
            &third_section.r_pow,
            (r * r * r * r).as_basis_coefficients_slice()
        );

        for row_idx in 9..trace.height() {
            let row = trace.row_slice(row_idx).unwrap();
            let row: &EqNegCols<F> = (&*row).borrow();
            assert_eq!(row.is_valid, F::ZERO);
            assert_eq!(row.proof_idx, F::ONE);
        }
    }

    #[test]
    fn ordered_eq_neg_rejects_undersized_required_height() {
        assert_eq!(
            eq_neg_trace_from_inputs(3, &[(EF::ONE, EF::TWO)], Some(8)).unwrap_err(),
            OrderedStackingTracegenError::RequiredHeight(6)
        );
    }
}

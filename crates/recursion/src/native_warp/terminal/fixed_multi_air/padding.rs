use core::borrow::{Borrow, BorrowMut};
use std::collections::BTreeMap;

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder,
    interaction::InteractionBuilder,
    native_warp::{FixedMultiAirPaddingReductionProof, FixedMultiAirPesatIndex},
    transcript::TranscriptLog,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    native_warp::terminal::fixed_multi_air::{
        FixedMultiAirBetaCoordinateBus, FixedMultiAirBetaCoordinateMessage,
        FixedMultiAirPaddingClaimBus, FixedMultiAirPaddingClaimMessage,
        FixedMultiAirPaddingOpeningBus, FixedMultiAirPaddingOpeningMessage,
        FixedMultiAirPaddingPointBus, FixedMultiAirPaddingPointMessage,
        FixedMultiAirPaddingSumcheckFinalBus, FixedMultiAirPaddingSumcheckFinalMessage,
        FixedMultiAirStructuredClaimHeaderBus, FixedMultiAirStructuredClaimHeaderMessage,
        FixedMultiAirStructuredPointBus, FixedMultiAirStructuredPointMessage,
    },
    utils::{ext_field_add, ext_field_multiply},
};

const PADDING_ROUND_TAG: u64 = 0x4e57_4d41_5450_0002;
const PADDING_OPENING_TAG: u64 = 0x4e57_4d41_544f_0002;
const EVALUATIONS: usize = 3;
const ROUND_EF_VALUES: usize = 2 + EVALUATIONS + 1;
const ROUND_BASE_VALUES: usize = ROUND_EF_VALUES * D_EF;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedMultiAirPaddingTraceError {
    Shape,
    Transcript,
    Claim,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirPaddingSumcheckScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub is_first_round: T,
    pub is_last_round: T,
    pub is_first_evaluation: T,
    pub is_last_evaluation: T,
    pub round: T,
    pub evaluation: T,
    pub evaluation_flags: [T; EVALUATIONS],
    pub denominator_inverse: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirPaddingSumcheckCols<T> {
    pub tidx: T,
    pub pre_claim: [T; D_EF],
    pub at_zero: [T; D_EF],
    pub at_one: [T; D_EF],
    pub at_two: [T; D_EF],
    pub evaluation: [T; D_EF],
    pub challenge: [T; D_EF],
    pub basis_prefix: [[T; D_EF]; EVALUATIONS + 1],
    pub term: [T; D_EF],
    pub sum_before: [T; D_EF],
    pub sum_after: [T; D_EF],
    pub post_claim: [T; D_EF],
}

/// Exact degree-two global-padding sumcheck verifier, with one row per
/// evaluation.  The proof polynomial is transcript-bound before its challenge.
pub struct FixedMultiAirPaddingSumcheckAir {
    pub transcript_bus: TranscriptBus,
    pub claim_bus: FixedMultiAirPaddingClaimBus,
    pub point_bus: FixedMultiAirPaddingPointBus,
    pub final_bus: FixedMultiAirPaddingSumcheckFinalBus,
    pub round_count: usize,
    pub point_lookup_count: usize,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirPaddingSumcheckAir {}
impl PartitionedBaseAir<F> for FixedMultiAirPaddingSumcheckAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirPaddingSumcheckScheduleCols::<F>::width()]
    }
    fn common_main_width(&self) -> usize {
        FixedMultiAirPaddingSumcheckCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirPaddingSumcheckAir {}
impl BaseAir<F> for FixedMultiAirPaddingSumcheckAir {
    fn width(&self) -> usize {
        FixedMultiAirPaddingSumcheckScheduleCols::<F>::width()
            + FixedMultiAirPaddingSumcheckCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirPaddingSumcheckAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("padding sumcheck schedule")
            .to_vec();
        let next_cached = builder.cached_mains()[0]
            .row_slice(1)
            .expect("next padding sumcheck schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("padding sumcheck row")
            .to_vec();
        let next_common = builder
            .common_main()
            .row_slice(1)
            .expect("next padding sumcheck row")
            .to_vec();
        let schedule: &FixedMultiAirPaddingSumcheckScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let next_schedule: &FixedMultiAirPaddingSumcheckScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FixedMultiAirPaddingSumcheckCols<AB::Var> = common.as_slice().borrow();
        let next: &FixedMultiAirPaddingSumcheckCols<AB::Var> = next_common.as_slice().borrow();

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
        let evaluation_sum = schedule
            .evaluation_flags
            .iter()
            .fold(AB::Expr::ZERO, |sum, &flag| sum + flag);
        builder.assert_eq(evaluation_sum, schedule.active);
        let evaluation = schedule
            .evaluation_flags
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |sum, (index, &flag)| {
                sum + flag * AB::Expr::from_usize(index)
            });
        builder
            .when(schedule.active)
            .assert_eq(schedule.evaluation, evaluation);
        builder.when_first_row().assert_one(schedule.active);
        builder.when_first_row().assert_one(schedule.is_first);
        builder.when_first_row().assert_one(schedule.is_first_round);
        builder
            .when(schedule.active * schedule.is_last_round)
            .assert_eq(schedule.round, AB::Expr::from_usize(self.round_count - 1));
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
        assert_array_eq(&mut same, next.pre_claim, local.pre_claim);
        assert_array_eq(&mut same, next.at_zero, local.at_zero);
        assert_array_eq(&mut same, next.at_one, local.at_one);
        assert_array_eq(&mut same, next.at_two, local.at_two);
        assert_array_eq(&mut same, next.challenge, local.challenge);
        assert_array_eq(&mut same, next.sum_before, local.sum_after);
        assert_array_eq(&mut same, next.post_claim, local.post_claim);
        let mut transition = builder.when_transition();
        let mut advance = transition.when(next_round);
        advance.assert_eq(next_schedule.round, schedule.round + AB::Expr::ONE);
        advance.assert_eq(
            next.tidx,
            local.tidx + AB::Expr::from_usize(ROUND_BASE_VALUES),
        );
        advance.assert_zero(next_schedule.is_first_round);
        advance.assert_zero(schedule.is_last_round);
        assert_array_eq(&mut advance, next.pre_claim, local.post_claim);

        assert_array_eq(
            &mut builder.when(schedule.active),
            local.pre_claim,
            ext_field_add::<AB::Expr>(local.at_zero, local.at_one),
        );
        let selected_evaluation = core::array::from_fn(|limb| {
            schedule
                .evaluation_flags
                .iter()
                .zip([local.at_zero, local.at_one, local.at_two])
                .fold(AB::Expr::ZERO, |sum, (&flag, value)| {
                    sum + flag * AB::Expr::from(value[limb])
                })
        });
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.evaluation,
            selected_evaluation,
        );
        let one = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            }
        });
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.basis_prefix[0],
            one,
        );
        for other in 0..EVALUATIONS {
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
        let scaled_basis = local.basis_prefix[EVALUATIONS]
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
            PADDING_ROUND_TAG,
            first_evaluation,
        );
        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.tidx) + AB::Expr::from_usize(D_EF),
            ext_from_base::<AB>(schedule.round.into()),
            schedule.active * schedule.is_first_evaluation,
        );
        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.tidx)
                + AB::Expr::from_usize(2 * D_EF)
                + AB::Expr::from(schedule.evaluation) * AB::Expr::from_usize(D_EF),
            local.evaluation,
            schedule.active,
        );
        self.transcript_bus.sample_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.tidx) + AB::Expr::from_usize((2 + EVALUATIONS) * D_EF),
            local.challenge,
            schedule.active * schedule.is_last_evaluation,
        );
        self.claim_bus.lookup_key(
            builder,
            FixedMultiAirPaddingClaimMessage {
                present: AB::Expr::ONE,
                claim: local.pre_claim.map(Into::into),
            },
            schedule.is_first,
        );
        self.point_bus.add_key_with_lookups(
            builder,
            FixedMultiAirPaddingPointMessage {
                coordinate: schedule.round.into(),
                value: local.challenge.map(Into::into),
            },
            schedule.active
                * schedule.is_last_evaluation
                * AB::Expr::from_usize(self.point_lookup_count),
        );
        self.final_bus.send(
            builder,
            FixedMultiAirPaddingSumcheckFinalMessage {
                tidx: AB::Expr::from(local.tidx) + AB::Expr::from_usize(ROUND_BASE_VALUES),
                claim: local.post_claim.map(Into::into),
            },
            schedule.is_last,
        );
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirPaddingSumcheckTraceOutput {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub point: Vec<EF>,
    pub final_claim: EF,
    pub end_tidx: usize,
}

pub fn generate_fixed_multi_air_padding_sumcheck_traces(
    initial_claim: EF,
    proof: &FixedMultiAirPaddingReductionProof<EF>,
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
    required_height: Option<usize>,
) -> Result<FixedMultiAirPaddingSumcheckTraceOutput, FixedMultiAirPaddingTraceError> {
    if proof.round_evaluations.is_empty() {
        return Err(FixedMultiAirPaddingTraceError::Shape);
    }
    let valid_rows = proof
        .round_evaluations
        .len()
        .checked_mul(EVALUATIONS)
        .ok_or(FixedMultiAirPaddingTraceError::Shape)?;
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return Err(FixedMultiAirPaddingTraceError::Shape);
    }
    let cached_width = FixedMultiAirPaddingSumcheckScheduleCols::<F>::width();
    let common_width = FixedMultiAirPaddingSumcheckCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    let mut tidx = start_tidx;
    let mut pre_claim = initial_claim;
    let mut point = Vec::with_capacity(proof.round_evaluations.len());
    let mut row = 0usize;
    for (round, evaluations) in proof.round_evaluations.iter().enumerate() {
        expect_ext(transcript, tidx, EF::from_u64(PADDING_ROUND_TAG), false)?;
        expect_ext(transcript, tidx + D_EF, EF::from_usize(round), false)?;
        for (index, &value) in evaluations.iter().enumerate() {
            expect_ext(transcript, tidx + (2 + index) * D_EF, value, false)?;
        }
        if evaluations[0] + evaluations[1] != pre_claim {
            return Err(FixedMultiAirPaddingTraceError::Claim);
        }
        let challenge = read_ext(transcript, tidx + (2 + EVALUATIONS) * D_EF, true)?;
        let post_claim = interpolate_three(*evaluations, challenge);
        let mut sum = EF::ZERO;
        for evaluation_index in 0..EVALUATIONS {
            let evaluation = evaluations[evaluation_index];
            let mut basis = EF::ONE;
            let mut prefixes = [EF::ONE; EVALUATIONS + 1];
            for other in 0..EVALUATIONS {
                if other != evaluation_index {
                    basis *= challenge - EF::from_usize(other);
                }
                prefixes[other + 1] = basis;
            }
            let denominator_inverse = lagrange_denominator(evaluation_index).inverse();
            let term = evaluation * basis * EF::from(denominator_inverse);
            let before = sum;
            sum += term;
            let cached_row = &mut cached[row * cached_width..(row + 1) * cached_width];
            let schedule: &mut FixedMultiAirPaddingSumcheckScheduleCols<F> =
                cached_row.borrow_mut();
            schedule.active = F::ONE;
            schedule.is_first = F::from_bool(row == 0);
            schedule.is_last = F::from_bool(row + 1 == valid_rows);
            schedule.is_first_round = F::from_bool(round == 0);
            schedule.is_last_round = F::from_bool(round + 1 == proof.round_evaluations.len());
            schedule.is_first_evaluation = F::from_bool(evaluation_index == 0);
            schedule.is_last_evaluation = F::from_bool(evaluation_index + 1 == EVALUATIONS);
            schedule.round = F::from_usize(round);
            schedule.evaluation = F::from_usize(evaluation_index);
            schedule.evaluation_flags[evaluation_index] = F::ONE;
            schedule.denominator_inverse = denominator_inverse;
            let common_row = &mut common[row * common_width..(row + 1) * common_width];
            let cols: &mut FixedMultiAirPaddingSumcheckCols<F> = common_row.borrow_mut();
            cols.tidx = F::from_usize(tidx);
            copy_ext(&mut cols.pre_claim, pre_claim);
            copy_ext(&mut cols.at_zero, evaluations[0]);
            copy_ext(&mut cols.at_one, evaluations[1]);
            copy_ext(&mut cols.at_two, evaluations[2]);
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
        tidx += ROUND_BASE_VALUES;
    }
    Ok(FixedMultiAirPaddingSumcheckTraceOutput {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        point,
        final_claim: pre_claim,
        end_tidx: tidx,
    })
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirPaddingOpeningCols<T> {
    pub active: T,
    pub tidx: T,
    pub claim: [T; D_EF],
    pub opening: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(FixedMultiAirPaddingOpeningCols<u8>)]
pub struct FixedMultiAirPaddingOpeningAir {
    pub transcript_bus: TranscriptBus,
    pub final_bus: FixedMultiAirPaddingSumcheckFinalBus,
    pub opening_bus: FixedMultiAirPaddingOpeningBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirPaddingOpeningAir {}
impl PartitionedBaseAir<F> for FixedMultiAirPaddingOpeningAir {}
impl BaseAir<F> for FixedMultiAirPaddingOpeningAir {
    fn width(&self) -> usize {
        FixedMultiAirPaddingOpeningCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for FixedMultiAirPaddingOpeningAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("padding opening row");
        let local: &FixedMultiAirPaddingOpeningCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_transition().assert_zero(local.active);
        observe_const(
            &self.transcript_bus,
            builder,
            local.tidx.into(),
            PADDING_OPENING_TAG,
            local.active.into(),
        );
        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.tidx) + AB::Expr::from_usize(D_EF),
            local.opening,
            local.active,
        );
        self.final_bus.receive(
            builder,
            FixedMultiAirPaddingSumcheckFinalMessage {
                tidx: local.tidx.into(),
                claim: local.claim.map(Into::into),
            },
            local.active,
        );
        self.opening_bus.send(
            builder,
            FixedMultiAirPaddingOpeningMessage {
                tidx: AB::Expr::from(local.tidx) + AB::Expr::from_usize(2 * D_EF),
                claim: local.claim.map(Into::into),
                opening: local.opening.map(Into::into),
            },
            local.active,
        );
    }
}

pub fn generate_fixed_multi_air_padding_opening_trace(
    proof: &FixedMultiAirPaddingReductionProof<EF>,
    transcript: &TranscriptLog<F, [F; 16]>,
    final_claim: EF,
    start_tidx: usize,
) -> Result<RowMajorMatrix<F>, FixedMultiAirPaddingTraceError> {
    expect_ext(
        transcript,
        start_tidx,
        EF::from_u64(PADDING_OPENING_TAG),
        false,
    )?;
    expect_ext(transcript, start_tidx + D_EF, proof.message_opening, false)?;
    let width = FixedMultiAirPaddingOpeningCols::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut FixedMultiAirPaddingOpeningCols<F> = values.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.tidx = F::from_usize(start_tidx);
    copy_ext(&mut cols.claim, final_claim);
    copy_ext(&mut cols.opening, proof.message_opening);
    Ok(RowMajorMatrix::new(values, width))
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirPaddingWeightScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub bit: T,
    pub point_coordinate: T,
    pub tau_coordinate: T,
    pub has_point: T,
    pub has_tau: T,
    pub before_active: [T; PADDING_STATE_COUNT],
    pub after_active: [T; PADDING_STATE_COUNT],
    pub enabled: [T; PADDING_TERM_COUNT],
    pub left_bit: [T; PADDING_TERM_COUNT],
    pub right_bit: [T; PADDING_TERM_COUNT],
    pub next_slot: [[T; PADDING_STATE_COUNT]; PADDING_TERM_COUNT],
    pub accept: [T; PADDING_STATE_COUNT],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirPaddingWeightCols<T> {
    pub tidx: T,
    pub point: [T; D_EF],
    pub tau: [T; D_EF],
    pub one: [T; D_EF],
    pub one_powers: [[T; D_EF]; PADDING_MAX_SCALE_POWER + 1],
    pub state_before: [[T; D_EF]; PADDING_STATE_COUNT],
    pub terms: [[T; D_EF]; PADDING_TERM_COUNT],
    pub state_after: [[T; D_EF]; PADDING_STATE_COUNT],
    pub claim: [T; D_EF],
    pub opening: [T; D_EF],
}

const PADDING_STATE_COUNT: usize = 8;
const PADDING_TERM_COUNT: usize = 2 * PADDING_STATE_COUNT;
pub(super) const PADDING_MAX_SCALE_POWER: usize = 8;

/// Exact logarithmic paired-interval digit DP for
/// `sum_j eq(tau, constraint_offset+j) eq(point, raw+j)`.
///
/// At every bit each translated carry has at most two values and subtraction
/// has one borrow bit, so eight setup-fixed slots suffice.  The cached trace
/// binds all carry keys and transitions; common columns contain only the
/// corresponding field weights.
pub struct FixedMultiAirPaddingWeightAir {
    pub beta_bus: FixedMultiAirBetaCoordinateBus,
    pub point_bus: FixedMultiAirPaddingPointBus,
    pub opening_bus: FixedMultiAirPaddingOpeningBus,
    pub header_bus: FixedMultiAirStructuredClaimHeaderBus,
    pub structured_point_bus: FixedMultiAirStructuredPointBus,
    pub claim_index: usize,
    pub log_message_len: usize,
    pub one_coordinate: usize,
    pub scale_exponent: usize,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirPaddingWeightAir {}
impl PartitionedBaseAir<F> for FixedMultiAirPaddingWeightAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirPaddingWeightScheduleCols::<F>::width()]
    }
    fn common_main_width(&self) -> usize {
        FixedMultiAirPaddingWeightCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirPaddingWeightAir {}
impl BaseAir<F> for FixedMultiAirPaddingWeightAir {
    fn width(&self) -> usize {
        FixedMultiAirPaddingWeightScheduleCols::<F>::width()
            + FixedMultiAirPaddingWeightCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirPaddingWeightAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("padding weight schedule")
            .to_vec();
        let next_cached = builder.cached_mains()[0]
            .row_slice(1)
            .expect("next padding weight schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("padding weight row")
            .to_vec();
        let next_common = builder
            .common_main()
            .row_slice(1)
            .expect("next padding weight row")
            .to_vec();
        let schedule: &FixedMultiAirPaddingWeightScheduleCols<AB::Var> = cached.as_slice().borrow();
        let next_schedule: &FixedMultiAirPaddingWeightScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FixedMultiAirPaddingWeightCols<AB::Var> = common.as_slice().borrow();
        let next: &FixedMultiAirPaddingWeightCols<AB::Var> = next_common.as_slice().borrow();
        for flag in [
            schedule.active,
            schedule.is_first,
            schedule.is_last,
            schedule.has_point,
            schedule.has_tau,
        ]
        .into_iter()
        .chain(schedule.before_active)
        .chain(schedule.after_active)
        .chain(schedule.enabled)
        .chain(schedule.left_bit)
        .chain(schedule.right_bit)
        .chain(schedule.accept)
        .chain(schedule.next_slot.into_iter().flatten())
        {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(schedule.active);
        builder.when_first_row().assert_one(schedule.is_first);
        builder
            .when_transition()
            .assert_bool(schedule.active - next_schedule.active);
        builder
            .when_transition()
            .assert_eq(schedule.active - next_schedule.active, schedule.is_last);
        builder
            .when_last_row()
            .assert_eq(schedule.is_last, schedule.active);
        let mut transition = builder.when_transition();
        let mut same = transition.when(next_schedule.active);
        same.assert_eq(next_schedule.bit, schedule.bit + AB::Expr::ONE);
        same.assert_zero(next_schedule.is_first);
        same.assert_eq(next.tidx, local.tidx);
        assert_array_eq(&mut same, next.one, local.one);
        for slot in 0..PADDING_STATE_COUNT {
            assert_array_eq(&mut same, next.state_before[slot], local.state_after[slot]);
        }
        assert_array_eq(&mut same, next.claim, local.claim);
        assert_array_eq(&mut same, next.opening, local.opening);
        let one = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            }
        });
        assert_array_eq(&mut builder.when(schedule.active), local.one_powers[0], one);
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.one_powers[1],
            local.one.map(Into::into),
        );
        for power in 1..PADDING_MAX_SCALE_POWER {
            assert_array_eq(
                &mut builder.when(schedule.active),
                local.one_powers[power + 1],
                ext_field_multiply::<AB::Expr>(local.one_powers[power], local.one_powers[1]),
            );
        }
        let initial_scale = local
            .one_powers
            .get(self.scale_exponent)
            .expect("setup-validated padding scale exponent");
        assert_array_eq(
            &mut builder.when(schedule.is_first),
            local.state_before[0],
            initial_scale.map(Into::into),
        );
        for slot in 1..PADDING_STATE_COUNT {
            assert_array_eq(
                &mut builder.when(schedule.is_first),
                local.state_before[slot],
                [AB::Expr::ZERO; D_EF],
            );
        }
        for slot in 0..PADDING_STATE_COUNT {
            for limb in 0..D_EF {
                builder.when(schedule.active).assert_zero(
                    (AB::Expr::ONE - schedule.before_active[slot]) * local.state_before[slot][limb],
                );
                builder.when(schedule.active).assert_zero(
                    (AB::Expr::ONE - schedule.after_active[slot]) * local.state_after[slot][limb],
                );
            }
            for j_bit in 0..2 {
                let term = slot * 2 + j_bit;
                let left_weight =
                    bit_weight::<AB>(local.tau, schedule.left_bit[term], schedule.has_tau);
                let right_weight =
                    bit_weight::<AB>(local.point, schedule.right_bit[term], schedule.has_point);
                let product = ext_field_multiply::<AB::Expr>(
                    local.state_before[slot],
                    ext_field_multiply::<AB::Expr>(left_weight, right_weight),
                );
                assert_array_eq(
                    // This identity is valid on padding rows too: the fixed
                    // schedule and generated common trace are zero there.
                    // Keeping it global avoids multiplying the degree-eight
                    // enabled/product expression by another activity flag.
                    builder,
                    local.terms[term],
                    product.map(|limb| schedule.enabled[term] * limb),
                );
            }
        }
        for term in 0..PADDING_TERM_COUNT {
            let next_slot_count = schedule.next_slot[term]
                .iter()
                .fold(AB::Expr::ZERO, |sum, &flag| sum + flag);
            builder
                .when(schedule.active)
                .assert_eq(next_slot_count, schedule.enabled[term]);
        }
        for next_slot in 0..PADDING_STATE_COUNT {
            let expected = core::array::from_fn(|limb| {
                (0..PADDING_TERM_COUNT).fold(AB::Expr::ZERO, |sum, term| {
                    sum + schedule.next_slot[term][next_slot]
                        * AB::Expr::from(local.terms[term][limb])
                })
            });
            assert_array_eq(
                &mut builder.when(schedule.active),
                local.state_after[next_slot],
                expected,
            );
        }
        let accepted = core::array::from_fn(|limb| {
            (0..PADDING_STATE_COUNT).fold(AB::Expr::ZERO, |sum, slot| {
                sum + schedule.accept[slot] * local.state_after[slot][limb]
            })
        });
        assert_array_eq(
            &mut builder.when(schedule.is_last),
            local.claim,
            ext_field_multiply::<AB::Expr>(accepted, local.opening),
        );
        self.point_bus.lookup_key(
            builder,
            FixedMultiAirPaddingPointMessage {
                coordinate: schedule.point_coordinate.into(),
                value: local.point.map(Into::into),
            },
            schedule.active * schedule.has_point,
        );
        self.beta_bus.lookup_key(
            builder,
            FixedMultiAirBetaCoordinateMessage {
                coordinate: schedule.tau_coordinate.into(),
                value: local.tau.map(Into::into),
            },
            schedule.active * schedule.has_tau,
        );
        self.beta_bus.lookup_key(
            builder,
            FixedMultiAirBetaCoordinateMessage {
                coordinate: AB::Expr::from_usize(self.one_coordinate),
                value: local.one.map(Into::into),
            },
            schedule.is_first,
        );
        self.opening_bus.receive(
            builder,
            FixedMultiAirPaddingOpeningMessage {
                tidx: local.tidx.into(),
                claim: local.claim.map(Into::into),
                opening: local.opening.map(Into::into),
            },
            schedule.is_first,
        );
        self.structured_point_bus.send(
            builder,
            FixedMultiAirStructuredPointMessage {
                claim: AB::Expr::from_usize(self.claim_index),
                term: AB::Expr::ZERO,
                coordinate: schedule.point_coordinate.into(),
                value: local.point.map(Into::into),
            },
            schedule.active * schedule.has_point,
        );
        self.header_bus.send(
            builder,
            FixedMultiAirStructuredClaimHeaderMessage {
                claim: AB::Expr::from_usize(self.claim_index),
                kind: AB::Expr::ONE,
                log_message_len: AB::Expr::from_usize(self.log_message_len),
                term_count: AB::Expr::ZERO,
                point_len: AB::Expr::from_usize(self.log_message_len),
                target: local.opening.map(Into::into),
            },
            schedule.is_last,
        );
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirPaddingWeightTraceOutput {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub weight: EF,
}

#[allow(clippy::too_many_arguments)]
pub fn generate_fixed_multi_air_padding_weight_traces(
    relation: &FixedMultiAirPesatIndex<F, Digest>,
    beta: &[EF],
    point: &[EF],
    claim: EF,
    opening: EF,
    end_tidx: usize,
    required_height: Option<usize>,
) -> Result<FixedMultiAirPaddingWeightTraceOutput, FixedMultiAirPaddingTraceError> {
    let description = relation.description();
    let log_message_len = relation.pesat_shape().log_witness;
    let global_log = relation.pesat_shape().log_constraints;
    if point.len() != log_message_len
        || beta.len() != relation.pesat_shape().beta_len()
        || description.raw_message_len + description.padding_constraint_count
            != description.padded_message_len
    {
        return Err(FixedMultiAirPaddingTraceError::Shape);
    }
    let tau = &beta[..global_log];
    let one = beta[global_log];
    let scale_exponent = description.exact_max_degree.saturating_sub(1) as usize;
    if scale_exponent > PADDING_MAX_SCALE_POWER {
        return Err(FixedMultiAirPaddingTraceError::Shape);
    }
    let mut one_powers = [EF::ONE; PADDING_MAX_SCALE_POWER + 1];
    for power in 0..PADDING_MAX_SCALE_POWER {
        one_powers[power + 1] = one_powers[power] * one;
    }
    let left_offset = description.padding_constraint_offset;
    let right_offset = description.raw_message_len;
    let count = description.padding_constraint_count;
    let left_len = 1u64
        .checked_shl(global_log as u32)
        .ok_or(FixedMultiAirPaddingTraceError::Shape)?;
    let right_len = 1u64
        .checked_shl(log_message_len as u32)
        .ok_or(FixedMultiAirPaddingTraceError::Shape)?;
    if count == 0
        || left_offset
            .checked_add(count)
            .is_none_or(|end| end > left_len)
        || right_offset
            .checked_add(count)
            .is_none_or(|end| end > right_len)
    {
        return Err(FixedMultiAirPaddingTraceError::Shape);
    }
    let index_bits = global_log.max(log_message_len);
    let valid_rows = index_bits + 1;
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if valid_rows == 0 || height < valid_rows {
        return Err(FixedMultiAirPaddingTraceError::Shape);
    }
    let cached_width = FixedMultiAirPaddingWeightScheduleCols::<F>::width();
    let common_width = FixedMultiAirPaddingWeightCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    let mut states =
        BTreeMap::from([((left_offset, right_offset, 0u8), one_powers[scale_exponent])]);
    for bit in 0..valid_rows {
        let before_keys = states.keys().copied().collect::<Vec<_>>();
        if before_keys.len() > PADDING_STATE_COUNT {
            return Err(FixedMultiAirPaddingTraceError::Shape);
        }
        let has_point = bit < log_message_len;
        let has_tau = bit < global_log;
        let point_coordinate = log_message_len.saturating_sub(1 + bit);
        let tau_coordinate = global_log.saturating_sub(1 + bit);
        let x = if has_point {
            point[point_coordinate]
        } else {
            EF::ZERO
        };
        let tau_value = if has_tau {
            tau[tau_coordinate]
        } else {
            EF::ZERO
        };
        let mut next = BTreeMap::new();
        let mut transitions = Vec::new();
        let count_bit = ((count >> bit) & 1) as i8;
        for (slot, &(left_carry, right_carry, borrow)) in before_keys.iter().enumerate() {
            let state = states[&(left_carry, right_carry, borrow)];
            for j_bit in 0..=u64::from(bit < index_bits) {
                let left_total = left_carry
                    .checked_add(j_bit)
                    .ok_or(FixedMultiAirPaddingTraceError::Shape)?;
                let right_total = right_carry
                    .checked_add(j_bit)
                    .ok_or(FixedMultiAirPaddingTraceError::Shape)?;
                let signed = j_bit as i8 - count_bit - borrow as i8;
                let next_borrow = u8::from(signed < 0);
                let left_bit = left_total & 1;
                let right_bit = right_total & 1;
                if (!has_tau && left_bit != 0) || (!has_point && right_bit != 0) {
                    continue;
                }
                let left_weight = if !has_tau {
                    EF::ONE
                } else if left_bit == 0 {
                    EF::ONE - tau_value
                } else {
                    tau_value
                };
                let right_weight = if !has_point {
                    EF::ONE
                } else if right_bit == 0 {
                    EF::ONE - x
                } else {
                    x
                };
                let next_key = (left_total >> 1, right_total >> 1, next_borrow);
                let term = state * left_weight * right_weight;
                *next.entry(next_key).or_insert(EF::ZERO) += term;
                transitions.push((
                    slot * 2 + j_bit as usize,
                    left_bit,
                    right_bit,
                    next_key,
                    term,
                ));
            }
        }
        let after_keys = next.keys().copied().collect::<Vec<_>>();
        if after_keys.len() > PADDING_STATE_COUNT {
            return Err(FixedMultiAirPaddingTraceError::Shape);
        }
        let cached_row = &mut cached[bit * cached_width..(bit + 1) * cached_width];
        let schedule: &mut FixedMultiAirPaddingWeightScheduleCols<F> = cached_row.borrow_mut();
        schedule.active = F::ONE;
        schedule.is_first = F::from_bool(bit == 0);
        schedule.is_last = F::from_bool(bit + 1 == valid_rows);
        schedule.bit = F::from_usize(bit);
        schedule.point_coordinate = F::from_usize(point_coordinate);
        schedule.tau_coordinate = F::from_usize(tau_coordinate);
        schedule.has_point = F::from_bool(has_point);
        schedule.has_tau = F::from_bool(has_tau);
        for slot in 0..before_keys.len() {
            schedule.before_active[slot] = F::ONE;
        }
        for slot in 0..after_keys.len() {
            schedule.after_active[slot] = F::ONE;
        }
        if bit + 1 == valid_rows {
            let accept_slot = after_keys
                .iter()
                .position(|&key| key == (0, 0, 1))
                .ok_or(FixedMultiAirPaddingTraceError::Claim)?;
            schedule.accept[accept_slot] = F::ONE;
        }
        let mut term_values = [EF::ZERO; PADDING_TERM_COUNT];
        for (term, left_bit, right_bit, next_key, value) in transitions {
            let next_slot = after_keys
                .iter()
                .position(|&key| key == next_key)
                .ok_or(FixedMultiAirPaddingTraceError::Shape)?;
            schedule.enabled[term] = F::ONE;
            schedule.left_bit[term] = F::from_u64(left_bit);
            schedule.right_bit[term] = F::from_u64(right_bit);
            schedule.next_slot[term][next_slot] = F::ONE;
            term_values[term] = value;
        }
        let common_row = &mut common[bit * common_width..(bit + 1) * common_width];
        let cols: &mut FixedMultiAirPaddingWeightCols<F> = common_row.borrow_mut();
        cols.tidx = F::from_usize(end_tidx);
        copy_ext(&mut cols.point, x);
        copy_ext(&mut cols.tau, tau_value);
        copy_ext(&mut cols.one, one);
        for (target, &value) in cols.one_powers.iter_mut().zip(&one_powers) {
            copy_ext(target, value);
        }
        for (target, key) in cols.state_before.iter_mut().zip(&before_keys) {
            copy_ext(target, states[key]);
        }
        for (target, value) in cols.terms.iter_mut().zip(term_values) {
            copy_ext(target, value);
        }
        for (target, key) in cols.state_after.iter_mut().zip(&after_keys) {
            copy_ext(target, next[key]);
        }
        copy_ext(&mut cols.claim, claim);
        copy_ext(&mut cols.opening, opening);
        states = next;
    }
    let weight = states.get(&(0, 0, 1)).copied().unwrap_or(EF::ZERO);
    if claim != weight * opening {
        return Err(FixedMultiAirPaddingTraceError::Claim);
    }
    Ok(FixedMultiAirPaddingWeightTraceOutput {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        weight,
    })
}

fn bit_weight<AB: AirBuilder<F = F>>(
    value: [AB::Var; D_EF],
    bit: AB::Var,
    present: AB::Var,
) -> [AB::Expr; D_EF] {
    let bit = AB::Expr::from(bit);
    let present = AB::Expr::from(present);
    core::array::from_fn(|limb| {
        let value = AB::Expr::from(value[limb].clone());
        if limb == 0 {
            present.clone()
                * (bit.clone() * value.clone()
                    + (AB::Expr::ONE - bit.clone()) * (AB::Expr::ONE - value))
                + (AB::Expr::ONE - present.clone()) * (AB::Expr::ONE - bit.clone())
        } else {
            present.clone() * (bit.clone() * value.clone() - (AB::Expr::ONE - bit.clone()) * value)
        }
    })
}

fn interpolate_three(values: [EF; 3], point: EF) -> EF {
    let two = EF::from_u8(2);
    let inv_two = two.inverse();
    let [zero, one, two_value] = values;
    zero * (point - EF::ONE) * (point - two) * inv_two - one * point * (point - two)
        + two_value * point * (point - EF::ONE) * inv_two
}

fn lagrange_denominator(index: usize) -> F {
    (0..EVALUATIONS)
        .filter(|&other| other != index)
        .map(|other| F::from_usize(index) - F::from_usize(other))
        .product()
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

fn read_ext(
    transcript: &TranscriptLog<F, [F; 16]>,
    tidx: usize,
    sample: bool,
) -> Result<EF, FixedMultiAirPaddingTraceError> {
    let end = tidx
        .checked_add(D_EF)
        .ok_or(FixedMultiAirPaddingTraceError::Transcript)?;
    let values = transcript
        .values()
        .get(tidx..end)
        .ok_or(FixedMultiAirPaddingTraceError::Transcript)?;
    let samples = transcript
        .samples()
        .get(tidx..end)
        .ok_or(FixedMultiAirPaddingTraceError::Transcript)?;
    if samples.iter().any(|&is_sample| is_sample != sample) {
        return Err(FixedMultiAirPaddingTraceError::Transcript);
    }
    EF::from_basis_coefficients_slice(values).ok_or(FixedMultiAirPaddingTraceError::Transcript)
}

fn expect_ext(
    transcript: &TranscriptLog<F, [F; 16]>,
    tidx: usize,
    expected: EF,
    sample: bool,
) -> Result<(), FixedMultiAirPaddingTraceError> {
    if read_ext(transcript, tidx, sample)? != expected {
        return Err(FixedMultiAirPaddingTraceError::Transcript);
    }
    Ok(())
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

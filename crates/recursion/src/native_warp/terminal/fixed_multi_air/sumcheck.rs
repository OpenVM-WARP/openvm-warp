use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{encoder::Encoder, utils::assert_array_eq, ColumnsAir, SubAir};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    native_warp::{DirectAirConstraintSumcheckProof, FIXED_MULTI_AIR_TERMINAL_ROUND_DEGREE},
    transcript::TranscriptLog,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField32,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    native_warp::terminal::fixed_multi_air::{
        FixedMultiAirRegionPointBus, FixedMultiAirRegionPointMessage, FixedMultiAirRegionStartBus,
        FixedMultiAirRegionStartMessage, FixedMultiAirRegionSumcheckFinalBus,
        FixedMultiAirRegionSumcheckFinalMessage,
    },
    utils::{ext_field_add, ext_field_multiply},
};

const DIRECT_TERMINAL_ROUND_TAG: u64 = 0x4e57_4441_4343_0002;
const ROUND_EVALUATIONS: usize = FIXED_MULTI_AIR_TERMINAL_ROUND_DEGREE + 1;
const ROUND_TRANSCRIPT_EF_VALUES: usize = 3 + ROUND_EVALUATIONS + 1;
const ROUND_TRANSCRIPT_BASE_VALUES: usize = ROUND_TRANSCRIPT_EF_VALUES * D_EF;
const SUMCHECK_SELECTOR_MAX_DEGREE: u32 = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedMultiAirRegionSumcheckTraceError {
    Shape,
    Transcript,
    Claim,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct FixedMultiAirRegionSumcheckCols<T, const ENC_WIDTH: usize> {
    pub active: T,
    pub round: T,
    pub evaluation_index: T,
    pub is_first_evaluation: T,
    pub is_last_evaluation: T,
    pub is_first_round: T,
    pub is_last_round: T,
    pub is_final: T,
    pub point_lookup_count: T,
    /// Index of the direct-AIR round domain tag.
    pub tidx: T,
    pub pre_claim: [T; D_EF],
    pub at_zero: [T; D_EF],
    pub at_one: [T; D_EF],
    pub evaluation: [T; D_EF],
    pub challenge: [T; D_EF],
    pub basis_prefix: [[T; D_EF]; ROUND_EVALUATIONS + 1],
    pub denominator_inverse: T,
    pub term: [T; D_EF],
    pub sum_before: [T; D_EF],
    pub sum_after: [T; D_EF],
    pub post_claim: [T; D_EF],
    pub evaluation_encoding: [T; ENC_WIDTH],
}

/// Evaluation-form verifier for one setup-fixed regional degree-six sumcheck.
///
/// There are seven rows per sumcheck round.  Lagrange products are split into
/// degree-two prefix recurrences, keeping the AIR degree at most five.
pub struct FixedMultiAirRegionSumcheckAir {
    pub transcript_bus: TranscriptBus,
    pub start_bus: FixedMultiAirRegionStartBus,
    pub point_bus: FixedMultiAirRegionPointBus,
    pub final_bus: FixedMultiAirRegionSumcheckFinalBus,
    pub region: usize,
    pub round_count: usize,
    /// Number of dense fixed-column MLEs folded at the endpoint. Analytic row
    /// selectors belong in `point_common_count`: each uses a point coordinate
    /// once per round rather than once per node in a dense fold tree.
    pub point_fixed_source_count: usize,
    /// Per-coordinate consumers independent of the fold layer.
    pub point_common_count: usize,
    evaluation_encoder: Encoder,
}

impl FixedMultiAirRegionSumcheckAir {
    #[must_use]
    pub fn new(
        transcript_bus: TranscriptBus,
        start_bus: FixedMultiAirRegionStartBus,
        point_bus: FixedMultiAirRegionPointBus,
        final_bus: FixedMultiAirRegionSumcheckFinalBus,
        region: usize,
        round_count: usize,
        point_fixed_source_count: usize,
        point_common_count: usize,
    ) -> Self {
        Self {
            transcript_bus,
            start_bus,
            point_bus,
            final_bus,
            region,
            round_count,
            point_fixed_source_count,
            point_common_count,
            evaluation_encoder: Encoder::new(
                ROUND_EVALUATIONS,
                SUMCHECK_SELECTOR_MAX_DEGREE,
                false,
            ),
        }
    }

    /// Height-one AIRs have no row variable and therefore no regional
    /// sumcheck round. Authenticate the transcript-prefix start claim as the
    /// final claim directly, without sampling a fictitious coordinate.
    fn eval_zero_round<AB, const ENC_WIDTH: usize>(&self, builder: &mut AB)
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
    {
        let main = builder.main();
        let local_row = main
            .row_slice(0)
            .expect("fixed multi-AIR zero-round bridge row");
        let next_row = main
            .row_slice(1)
            .expect("fixed multi-AIR next zero-round bridge row");
        let local: &FixedMultiAirRegionSumcheckCols<AB::Var, ENC_WIDTH> = (*local_row).borrow();
        let next: &FixedMultiAirRegionSumcheckCols<AB::Var, ENC_WIDTH> = (*next_row).borrow();

        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.active);

        for value in [
            local.round,
            local.evaluation_index,
            local.is_first_evaluation,
            local.is_last_evaluation,
            local.is_first_round,
            local.is_last_round,
            local.is_final,
            local.point_lookup_count,
            local.denominator_inverse,
        ] {
            builder.when(local.active).assert_zero(value);
        }
        for value in local
            .at_zero
            .iter()
            .chain(&local.at_one)
            .chain(&local.evaluation)
            .chain(&local.challenge)
            .chain(local.basis_prefix.iter().flatten())
            .chain(&local.term)
            .chain(&local.sum_before)
            .chain(&local.sum_after)
            .chain(&local.evaluation_encoding)
        {
            builder.when(local.active).assert_zero(*value);
        }
        assert_array_eq(
            &mut builder.when(local.active),
            local.post_claim,
            local.pre_claim.map(Into::into),
        );

        self.start_bus.receive(
            builder,
            FixedMultiAirRegionStartMessage {
                region: AB::Expr::from_usize(self.region),
                tidx: local.tidx.into(),
                claim: local.pre_claim.map(Into::into),
            },
            local.active,
        );
        self.final_bus.send(
            builder,
            FixedMultiAirRegionSumcheckFinalMessage {
                region: AB::Expr::from_usize(self.region),
                tidx: local.tidx.into(),
                claim: local.post_claim.map(Into::into),
            },
            local.active,
        );
    }

    #[must_use]
    pub const fn transcript_values_per_round() -> usize {
        ROUND_TRANSCRIPT_BASE_VALUES
    }

    fn eval_impl<AB, const ENC_WIDTH: usize>(&self, builder: &mut AB)
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
        <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
    {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("fixed multi-AIR sumcheck row");
        let next_row = main
            .row_slice(1)
            .expect("fixed multi-AIR next sumcheck row");
        let local: &FixedMultiAirRegionSumcheckCols<AB::Var, ENC_WIDTH> = (*local_row).borrow();
        let next: &FixedMultiAirRegionSumcheckCols<AB::Var, ENC_WIDTH> = (*next_row).borrow();

        for flag in [
            local.active,
            local.is_first_evaluation,
            local.is_last_evaluation,
            local.is_first_round,
            local.is_last_round,
            local.is_final,
        ] {
            builder.assert_bool(flag);
        }
        self.evaluation_encoder
            .eval(builder, &local.evaluation_encoding);
        let indices = (0..ROUND_EVALUATIONS)
            .map(|index| (index, index))
            .collect::<Vec<_>>();
        let first = (0..ROUND_EVALUATIONS)
            .map(|index| (index, usize::from(index == 0)))
            .collect::<Vec<_>>();
        let last = (0..ROUND_EVALUATIONS)
            .map(|index| (index, usize::from(index + 1 == ROUND_EVALUATIONS)))
            .collect::<Vec<_>>();
        builder.when(local.active).assert_eq(
            local.evaluation_index,
            self.evaluation_encoder
                .flag_with_val::<AB>(&local.evaluation_encoding, &indices),
        );
        builder.when(local.active).assert_eq(
            local.is_first_evaluation,
            self.evaluation_encoder
                .flag_with_val::<AB>(&local.evaluation_encoding, &first),
        );
        builder.when(local.active).assert_eq(
            local.is_last_evaluation,
            self.evaluation_encoder
                .flag_with_val::<AB>(&local.evaluation_encoding, &last),
        );
        builder
            .when(local.active * local.is_first_round)
            .assert_zero(local.round);
        builder
            .when(local.active * local.is_last_round)
            .assert_eq(local.round, AB::Expr::from_usize(self.round_count - 1));
        let first_point_count = self
            .point_fixed_source_count
            .checked_mul(1usize << (self.round_count - 1))
            .and_then(|count| count.checked_add(self.point_common_count))
            .expect("fixed endpoint point lookup count");
        builder.when(local.active * local.is_first_round).assert_eq(
            local.point_lookup_count,
            AB::Expr::from_usize(first_point_count),
        );
        builder.when(local.active).assert_eq(
            local.is_final,
            local.is_last_round * local.is_last_evaluation,
        );

        builder.when_first_row().assert_one(local.active);
        builder
            .when_first_row()
            .assert_one(local.is_first_evaluation);
        builder.when_first_row().assert_one(local.is_first_round);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.is_final);

        let same_round = next.active * (AB::Expr::ONE - next.is_first_evaluation);
        let next_round = next.active * next.is_first_evaluation;
        let mut transition = builder.when_transition();
        let mut same = transition.when(same_round);
        same.assert_eq(next.round, local.round);
        same.assert_eq(next.evaluation_index, local.evaluation_index + AB::F::ONE);
        same.assert_eq(next.tidx, local.tidx);
        same.assert_eq(next.is_first_round, local.is_first_round);
        same.assert_eq(next.is_last_round, local.is_last_round);
        assert_array_eq(&mut same, next.pre_claim, local.pre_claim);
        assert_array_eq(&mut same, next.at_zero, local.at_zero);
        assert_array_eq(&mut same, next.at_one, local.at_one);
        assert_array_eq(&mut same, next.challenge, local.challenge);
        assert_array_eq(&mut same, next.sum_before, local.sum_after);
        assert_array_eq(&mut same, next.post_claim, local.post_claim);
        same.assert_eq(next.point_lookup_count, local.point_lookup_count);

        let mut transition = builder.when_transition();
        let mut advance = transition.when(next_round);
        advance.assert_one(local.is_last_evaluation);
        advance.assert_eq(next.round, local.round + AB::F::ONE);
        advance.assert_eq(
            next.tidx,
            local.tidx + AB::Expr::from_usize(ROUND_TRANSCRIPT_BASE_VALUES),
        );
        advance.assert_zero(next.is_first_round);
        advance.assert_zero(local.is_last_round);
        assert_array_eq(&mut advance, next.pre_claim, local.post_claim);
        advance.assert_eq(
            next.point_lookup_count * AB::Expr::TWO,
            local.point_lookup_count + AB::Expr::from_usize(self.point_common_count),
        );

        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_final);
        assert_array_eq(
            &mut builder.when(local.active),
            local.pre_claim,
            ext_field_add::<AB::Expr>(local.at_zero, local.at_one),
        );
        assert_array_eq(
            &mut builder.when(local.active * local.is_first_evaluation),
            local.evaluation,
            local.at_zero.map(Into::into),
        );
        let is_at_one = self.evaluation_encoder.flag_with_val::<AB>(
            &local.evaluation_encoding,
            &(0..ROUND_EVALUATIONS)
                .map(|index| (index, usize::from(index == 1)))
                .collect::<Vec<_>>(),
        );
        assert_array_eq(
            &mut builder.when(local.active * is_at_one),
            local.evaluation,
            local.at_one.map(Into::into),
        );

        let one_ext = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            }
        });
        assert_array_eq(
            &mut builder.when(local.active),
            local.basis_prefix[0],
            one_ext,
        );
        for other in 0..ROUND_EVALUATIONS {
            let skip = self.evaluation_encoder.flag_with_val::<AB>(
                &local.evaluation_encoding,
                &(0..ROUND_EVALUATIONS)
                    .map(|index| (index, usize::from(index == other)))
                    .collect::<Vec<_>>(),
            );
            let factor = core::array::from_fn(|limb| {
                if limb == 0 {
                    skip.clone()
                        + (AB::Expr::ONE - skip.clone())
                            * (AB::Expr::from(local.challenge[limb]) - AB::Expr::from_usize(other))
                } else {
                    (AB::Expr::ONE - skip.clone()) * AB::Expr::from(local.challenge[limb])
                }
            });
            assert_array_eq(
                &mut builder.when(local.active),
                local.basis_prefix[other + 1],
                ext_field_multiply::<AB::Expr>(local.basis_prefix[other], factor),
            );
        }
        let denominator_inverse = self.evaluation_encoder.flag_with_val::<AB>(
            &local.evaluation_encoding,
            &(0..ROUND_EVALUATIONS)
                .map(|index| {
                    (
                        index,
                        lagrange_denominator(index).inverse().as_canonical_u32() as usize,
                    )
                })
                .collect::<Vec<_>>(),
        );
        builder
            .when(local.active)
            .assert_eq(local.denominator_inverse, denominator_inverse);
        let scaled_basis = local.basis_prefix[ROUND_EVALUATIONS]
            .map(|limb| AB::Expr::from(limb) * AB::Expr::from(local.denominator_inverse));
        assert_array_eq(
            &mut builder.when(local.active),
            local.term,
            ext_field_multiply::<AB::Expr>(local.evaluation, scaled_basis),
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.sum_after,
            ext_field_add::<AB::Expr>(local.sum_before, local.term),
        );
        assert_array_eq(
            &mut builder.when(local.is_first_evaluation),
            local.sum_before,
            [AB::Expr::ZERO; D_EF],
        );
        assert_array_eq(
            &mut builder.when(local.is_last_evaluation),
            local.sum_after,
            local.post_claim.map(Into::into),
        );

        let first_eval = local.active * local.is_first_evaluation;
        observe_ext_const(
            &self.transcript_bus,
            builder,
            local.tidx.into(),
            DIRECT_TERMINAL_ROUND_TAG,
            first_eval.clone(),
        );
        // The round and fixed evaluation-vector length are transcript constants.
        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.tidx) + AB::Expr::from_usize(D_EF),
            ext_from_base_expr::<AB>(local.round.into()),
            first_eval.clone(),
        );
        observe_ext_const(
            &self.transcript_bus,
            builder,
            AB::Expr::from(local.tidx) + AB::Expr::from_usize(2 * D_EF),
            ROUND_EVALUATIONS as u64,
            first_eval,
        );
        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.tidx)
                + AB::Expr::from_usize(3 * D_EF)
                + AB::Expr::from(local.evaluation_index) * AB::Expr::from_usize(D_EF),
            local.evaluation,
            local.active,
        );
        self.transcript_bus.sample_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.tidx) + AB::Expr::from_usize((3 + ROUND_EVALUATIONS) * D_EF),
            local.challenge,
            local.active * local.is_last_evaluation,
        );

        self.start_bus.receive(
            builder,
            FixedMultiAirRegionStartMessage {
                region: AB::Expr::from_usize(self.region),
                tidx: local.tidx.into(),
                claim: local.pre_claim.map(Into::into),
            },
            local.active * local.is_first_round * local.is_first_evaluation,
        );
        self.point_bus.add_key_with_lookups(
            builder,
            FixedMultiAirRegionPointMessage {
                region: AB::Expr::from_usize(self.region),
                coordinate: local.round.into(),
                value: local.challenge.map(Into::into),
            },
            local.active * local.is_last_evaluation * local.point_lookup_count,
        );
        self.final_bus.send(
            builder,
            FixedMultiAirRegionSumcheckFinalMessage {
                region: AB::Expr::from_usize(self.region),
                tidx: AB::Expr::from(local.tidx)
                    + AB::Expr::from_usize(ROUND_TRANSCRIPT_BASE_VALUES),
                claim: local.post_claim.map(Into::into),
            },
            local.active * local.is_final,
        );
    }
}

impl BaseAirWithPublicValues<F> for FixedMultiAirRegionSumcheckAir {}
impl PartitionedBaseAir<F> for FixedMultiAirRegionSumcheckAir {}
impl ColumnsAir for FixedMultiAirRegionSumcheckAir {}
impl BaseAir<F> for FixedMultiAirRegionSumcheckAir {
    fn width(&self) -> usize {
        match self.evaluation_encoder.width() {
            1 => FixedMultiAirRegionSumcheckCols::<F, 1>::width(),
            2 => FixedMultiAirRegionSumcheckCols::<F, 2>::width(),
            3 => FixedMultiAirRegionSumcheckCols::<F, 3>::width(),
            4 => FixedMultiAirRegionSumcheckCols::<F, 4>::width(),
            width => panic!("unsupported fixed multi-AIR sumcheck encoder width: {width}"),
        }
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for FixedMultiAirRegionSumcheckAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        if self.round_count == 0 {
            return match self.evaluation_encoder.width() {
                1 => self.eval_zero_round::<AB, 1>(builder),
                2 => self.eval_zero_round::<AB, 2>(builder),
                3 => self.eval_zero_round::<AB, 3>(builder),
                4 => self.eval_zero_round::<AB, 4>(builder),
                width => panic!("unsupported fixed multi-AIR sumcheck encoder width: {width}"),
            };
        }
        match self.evaluation_encoder.width() {
            1 => self.eval_impl::<AB, 1>(builder),
            2 => self.eval_impl::<AB, 2>(builder),
            3 => self.eval_impl::<AB, 3>(builder),
            4 => self.eval_impl::<AB, 4>(builder),
            width => panic!("unsupported fixed multi-AIR sumcheck encoder width: {width}"),
        }
    }
}

pub fn generate_fixed_multi_air_region_sumcheck_trace(
    air: &FixedMultiAirRegionSumcheckAir,
    initial_claim: EF,
    proof: &DirectAirConstraintSumcheckProof<EF>,
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
    required_height: Option<usize>,
) -> Result<RowMajorMatrix<F>, FixedMultiAirRegionSumcheckTraceError> {
    match air.evaluation_encoder.width() {
        1 => generate_trace_impl::<1>(
            air,
            initial_claim,
            proof,
            transcript,
            start_tidx,
            required_height,
        ),
        2 => generate_trace_impl::<2>(
            air,
            initial_claim,
            proof,
            transcript,
            start_tidx,
            required_height,
        ),
        3 => generate_trace_impl::<3>(
            air,
            initial_claim,
            proof,
            transcript,
            start_tidx,
            required_height,
        ),
        4 => generate_trace_impl::<4>(
            air,
            initial_claim,
            proof,
            transcript,
            start_tidx,
            required_height,
        ),
        _ => Err(FixedMultiAirRegionSumcheckTraceError::Shape),
    }
}

fn generate_trace_impl<const ENC_WIDTH: usize>(
    air: &FixedMultiAirRegionSumcheckAir,
    initial_claim: EF,
    proof: &DirectAirConstraintSumcheckProof<EF>,
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
    required_height: Option<usize>,
) -> Result<RowMajorMatrix<F>, FixedMultiAirRegionSumcheckTraceError> {
    if proof.round_evaluations.len() != air.round_count
        || proof
            .round_evaluations
            .iter()
            .any(|values| values.len() != ROUND_EVALUATIONS)
    {
        return Err(FixedMultiAirRegionSumcheckTraceError::Shape);
    }
    if air.round_count == 0 {
        let height = required_height.unwrap_or(1);
        if height == 0 {
            return Err(FixedMultiAirRegionSumcheckTraceError::Shape);
        }
        let width = FixedMultiAirRegionSumcheckCols::<F, ENC_WIDTH>::width();
        let mut trace = F::zero_vec(height * width);
        let cols: &mut FixedMultiAirRegionSumcheckCols<F, ENC_WIDTH> = trace[..width].borrow_mut();
        cols.active = F::ONE;
        cols.tidx = F::from_usize(start_tidx);
        copy_ext(&mut cols.pre_claim, initial_claim);
        copy_ext(&mut cols.post_claim, initial_claim);
        return Ok(RowMajorMatrix::new(trace, width));
    }
    let valid_rows = air
        .round_count
        .checked_mul(ROUND_EVALUATIONS)
        .ok_or(FixedMultiAirRegionSumcheckTraceError::Shape)?;
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return Err(FixedMultiAirRegionSumcheckTraceError::Shape);
    }
    let width = FixedMultiAirRegionSumcheckCols::<F, ENC_WIDTH>::width();
    let mut trace = F::zero_vec(height * width);
    let mut pre_claim = initial_claim;
    let mut tidx = start_tidx;
    let mut row_index = 0usize;
    for (round, evaluations) in proof.round_evaluations.iter().enumerate() {
        let point_lookup_count = air
            .point_fixed_source_count
            .checked_mul(1usize << (air.round_count - 1 - round))
            .and_then(|count| count.checked_add(air.point_common_count))
            .ok_or(FixedMultiAirRegionSumcheckTraceError::Shape)?;
        expect_ext(
            transcript,
            tidx,
            EF::from_u64(DIRECT_TERMINAL_ROUND_TAG),
            false,
        )?;
        expect_ext(transcript, tidx + D_EF, EF::from_usize(round), false)?;
        expect_ext(
            transcript,
            tidx + 2 * D_EF,
            EF::from_usize(ROUND_EVALUATIONS),
            false,
        )?;
        for (index, &value) in evaluations.iter().enumerate() {
            expect_ext(transcript, tidx + (3 + index) * D_EF, value, false)?;
        }
        let challenge = read_ext(transcript, tidx + (3 + ROUND_EVALUATIONS) * D_EF, true)?;
        if evaluations[0] + evaluations[1] != pre_claim {
            return Err(FixedMultiAirRegionSumcheckTraceError::Claim);
        }
        let post_claim = interpolate(evaluations, challenge);
        let mut sum = EF::ZERO;
        for (evaluation_index, &evaluation) in evaluations.iter().enumerate() {
            let mut basis = EF::ONE;
            let mut prefixes = vec![basis];
            for other in 0..ROUND_EVALUATIONS {
                if other != evaluation_index {
                    basis *= challenge - EF::from_usize(other);
                }
                prefixes.push(basis);
            }
            let denominator_inverse = lagrange_denominator(evaluation_index).inverse();
            let term = evaluation * basis * EF::from(denominator_inverse);
            let sum_before = sum;
            sum += term;
            let row = &mut trace[row_index * width..(row_index + 1) * width];
            let cols: &mut FixedMultiAirRegionSumcheckCols<F, ENC_WIDTH> = row.borrow_mut();
            cols.active = F::ONE;
            cols.round = F::from_usize(round);
            cols.evaluation_index = F::from_usize(evaluation_index);
            cols.is_first_evaluation = F::from_bool(evaluation_index == 0);
            cols.is_last_evaluation = F::from_bool(evaluation_index + 1 == ROUND_EVALUATIONS);
            cols.is_first_round = F::from_bool(round == 0);
            cols.is_last_round = F::from_bool(round + 1 == air.round_count);
            cols.is_final = F::from_bool(
                round + 1 == air.round_count && evaluation_index + 1 == ROUND_EVALUATIONS,
            );
            cols.point_lookup_count = F::from_usize(point_lookup_count);
            cols.tidx = F::from_usize(tidx);
            copy_ext(&mut cols.pre_claim, pre_claim);
            copy_ext(&mut cols.at_zero, evaluations[0]);
            copy_ext(&mut cols.at_one, evaluations[1]);
            copy_ext(&mut cols.evaluation, evaluation);
            copy_ext(&mut cols.challenge, challenge);
            for (target, value) in cols.basis_prefix.iter_mut().zip(prefixes) {
                copy_ext(target, value);
            }
            cols.denominator_inverse = denominator_inverse;
            copy_ext(&mut cols.term, term);
            copy_ext(&mut cols.sum_before, sum_before);
            copy_ext(&mut cols.sum_after, sum);
            copy_ext(&mut cols.post_claim, post_claim);
            for (target, value) in cols
                .evaluation_encoding
                .iter_mut()
                .zip(air.evaluation_encoder.get_flag_pt(evaluation_index))
            {
                *target = F::from_u32(value);
            }
            row_index += 1;
        }
        if sum != post_claim {
            return Err(FixedMultiAirRegionSumcheckTraceError::Claim);
        }
        pre_claim = post_claim;
        tidx += ROUND_TRANSCRIPT_BASE_VALUES;
    }
    Ok(RowMajorMatrix::new(trace, width))
}

fn observe_ext_const<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    tidx: AB::Expr,
    value: u64,
    enabled: AB::Expr,
) {
    bus.observe_ext(
        builder,
        AB::Expr::ZERO,
        tidx,
        core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::from_u64(value)
            } else {
                AB::Expr::ZERO
            }
        }),
        enabled,
    );
}

fn ext_from_base_expr<AB: AirBuilder<F = F>>(value: AB::Expr) -> [AB::Expr; D_EF] {
    core::array::from_fn(|limb| {
        if limb == 0 {
            value.clone()
        } else {
            AB::Expr::ZERO
        }
    })
}

fn lagrange_denominator(index: usize) -> F {
    (0..ROUND_EVALUATIONS)
        .filter(|&other| other != index)
        .map(|other| F::from_usize(index) - F::from_usize(other))
        .product()
}

fn interpolate(evaluations: &[EF], challenge: EF) -> EF {
    evaluations
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let numerator = (0..ROUND_EVALUATIONS)
                .filter(|&other| other != index)
                .map(|other| challenge - EF::from_usize(other))
                .product::<EF>();
            value * numerator * EF::from(lagrange_denominator(index).inverse())
        })
        .sum()
}

fn read_ext(
    transcript: &TranscriptLog<F, [F; 16]>,
    tidx: usize,
    is_sample: bool,
) -> Result<EF, FixedMultiAirRegionSumcheckTraceError> {
    let end = tidx
        .checked_add(D_EF)
        .ok_or(FixedMultiAirRegionSumcheckTraceError::Transcript)?;
    let values = transcript
        .values()
        .get(tidx..end)
        .ok_or(FixedMultiAirRegionSumcheckTraceError::Transcript)?;
    let flags = transcript
        .samples()
        .get(tidx..end)
        .ok_or(FixedMultiAirRegionSumcheckTraceError::Transcript)?;
    if flags.iter().any(|&flag| flag != is_sample) {
        return Err(FixedMultiAirRegionSumcheckTraceError::Transcript);
    }
    EF::from_basis_coefficients_slice(values)
        .ok_or(FixedMultiAirRegionSumcheckTraceError::Transcript)
}

fn expect_ext(
    transcript: &TranscriptLog<F, [F; 16]>,
    tidx: usize,
    expected: EF,
    is_sample: bool,
) -> Result<(), FixedMultiAirRegionSumcheckTraceError> {
    if read_ext(transcript, tidx, is_sample)? != expected {
        return Err(FixedMultiAirRegionSumcheckTraceError::Transcript);
    }
    Ok(())
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

#[cfg(test)]
mod tests {
    use openvm_stark_backend::{
        air_builders::debug::check_constraints, native_warp::DirectAirConstraintSumcheckProof,
        transcript::TranscriptLog,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;
    use p3_matrix::Matrix;

    use super::*;

    #[test]
    fn height_one_region_uses_constrained_zero_round_bridge() {
        let air = FixedMultiAirRegionSumcheckAir::new(
            TranscriptBus::new(1),
            FixedMultiAirRegionStartBus::new(2),
            FixedMultiAirRegionPointBus::new(3),
            FixedMultiAirRegionSumcheckFinalBus::new(4),
            7,
            0,
            0,
            3,
        );
        let claim = EF::from_u32(19);
        let proof = DirectAirConstraintSumcheckProof {
            round_evaluations: Vec::new(),
            opened_columns: vec![EF::from_u32(23)],
        };
        let transcript = TranscriptLog::<F, [F; 16]>::default();
        let trace = generate_fixed_multi_air_region_sumcheck_trace(
            &air,
            claim,
            &proof,
            &transcript,
            31,
            None,
        )
        .expect("zero-round bridge trace");
        assert_eq!(trace.height(), 1);
        check_constraints::<_, BabyBearPoseidon2Config>(
            &air,
            "height-one fixed multi-AIR regional bridge",
            &None,
            &[trace.as_view()],
            &[],
        );
    }
}

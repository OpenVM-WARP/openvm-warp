use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{encoder::Encoder, utils::assert_array_eq, ColumnsAir, SubAir};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    native_warp::{
        FixedMultiAirCompleteRegionSumcheckProof, COMPLETE_TERMINAL_INTERACTION_REGION_TAG,
        COMPLETE_TERMINAL_LOCAL_REGION_TAG, COMPLETE_TERMINAL_ROUND_TAG,
        FIXED_MULTI_AIR_COMPLETE_TERMINAL_ROUND_DEGREE,
    },
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
    native_warp::terminal::fixed_multi_air_complete::{
        FixedMultiAirCompleteInteractionRegionPointBus,
        FixedMultiAirCompleteInteractionRegionStartBus,
        FixedMultiAirCompleteInteractionRegionSumcheckFinalBus,
        FixedMultiAirCompleteLocalRegionPointBus, FixedMultiAirCompleteLocalRegionStartBus,
        FixedMultiAirCompleteLocalRegionSumcheckFinalBus, FixedMultiAirCompleteRegionPointMessage,
        FixedMultiAirCompleteRegionStartMessage, FixedMultiAirCompleteRegionSumcheckFinalMessage,
    },
    utils::{ext_field_add, ext_field_multiply},
};

/// The complete relation has degree six; equality weighting adds one degree.
pub const FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS: usize =
    FIXED_MULTI_AIR_COMPLETE_TERMINAL_ROUND_DEGREE + 1;

/// Native transcript footprint of the setup-fixed region tag and region id.
pub const FIXED_MULTI_AIR_COMPLETE_REGION_PREFIX_BASE_VALUES: usize = 2 * D_EF;

/// Native transcript footprint of one complete-terminal regional round.
pub const FIXED_MULTI_AIR_COMPLETE_ROUND_TRANSCRIPT_BASE_VALUES: usize =
    (3 + FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS + 1) * D_EF;

const SUMCHECK_SELECTOR_MAX_DEGREE: u32 = 2;
const EVALUATION_ENCODER_WIDTH: usize = 3;

const _: () = assert!(D_EF == 4);
const _: () = assert!(FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS == 8);

/// Fallible construction/trace error. Every proof-controlled dimension and
/// transcript index is checked before indexing or allocation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedMultiAirCompleteRegionSumcheckError {
    InvalidConfiguration(&'static str),
    ProofShape(&'static str),
    Transcript(&'static str),
    Claim { round: usize },
    HeightOverflow,
    TraceHeight,
    Allocation,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct FixedMultiAirCompleteRegionSumcheckCols<T> {
    pub active: T,
    pub round: T,
    pub evaluation_index: T,
    pub is_first_evaluation: T,
    pub is_last_evaluation: T,
    pub is_first_round: T,
    pub is_last_round: T,
    pub is_final: T,
    pub point_lookup_count: T,
    /// Index of the setup-fixed local/interaction region domain tag.
    pub region_tidx: T,
    /// Index of this round's `COMPLETE_TERMINAL_ROUND_TAG`.
    pub round_tidx: T,
    pub pre_claim: [T; D_EF],
    pub at_zero: [T; D_EF],
    pub at_one: [T; D_EF],
    pub evaluation: [T; D_EF],
    pub challenge: [T; D_EF],
    /// Prefix products for the eight-node Lagrange basis. Splitting the
    /// product into one multiplication per column keeps the AIR degree within
    /// the existing recursive profile.
    pub basis_prefix: [[T; D_EF]; FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS + 1],
    pub denominator_inverse: T,
    pub term: [T; D_EF],
    pub sum_before: [T; D_EF],
    pub sum_after: [T; D_EF],
    pub post_claim: [T; D_EF],
    pub evaluation_encoding: [T; EVALUATION_ENCODER_WIDTH],
}

#[derive(Clone, Copy)]
struct LocalRegionBuses {
    start: FixedMultiAirCompleteLocalRegionStartBus,
    point: FixedMultiAirCompleteLocalRegionPointBus,
    final_claim: FixedMultiAirCompleteLocalRegionSumcheckFinalBus,
}

#[derive(Clone, Copy)]
struct InteractionRegionBuses {
    start: FixedMultiAirCompleteInteractionRegionStartBus,
    point: FixedMultiAirCompleteInteractionRegionPointBus,
    final_claim: FixedMultiAirCompleteInteractionRegionSumcheckFinalBus,
}

#[derive(Clone, Copy)]
enum RegionBuses {
    Local(LocalRegionBuses),
    Interaction(InteractionRegionBuses),
}

impl RegionBuses {
    const fn domain_tag(self) -> u64 {
        match self {
            Self::Local(_) => COMPLETE_TERMINAL_LOCAL_REGION_TAG,
            Self::Interaction(_) => COMPLETE_TERMINAL_INTERACTION_REGION_TAG,
        }
    }

    fn receive_start<AB: InteractionBuilder>(
        self,
        builder: &mut AB,
        message: FixedMultiAirCompleteRegionStartMessage<impl Into<AB::Expr> + Clone>,
        enabled: impl Into<AB::Expr>,
    ) {
        match self {
            Self::Local(buses) => buses.start.receive(builder, message, enabled),
            Self::Interaction(buses) => buses.start.receive(builder, message, enabled),
        }
    }

    fn publish_point<AB: InteractionBuilder>(
        self,
        builder: &mut AB,
        message: FixedMultiAirCompleteRegionPointMessage<impl Into<AB::Expr> + Clone>,
        count: impl Into<AB::Expr>,
    ) {
        match self {
            Self::Local(buses) => buses.point.add_key_with_lookups(builder, message, count),
            Self::Interaction(buses) => buses.point.add_key_with_lookups(builder, message, count),
        }
    }

    fn send_final<AB: InteractionBuilder>(
        self,
        builder: &mut AB,
        message: FixedMultiAirCompleteRegionSumcheckFinalMessage<impl Into<AB::Expr> + Clone>,
        enabled: impl Into<AB::Expr>,
    ) {
        match self {
            Self::Local(buses) => buses.final_claim.send(builder, message, enabled),
            Self::Interaction(buses) => buses.final_claim.send(builder, message, enabled),
        }
    }
}

struct FixedMultiAirCompleteRegionSumcheckCore {
    transcript_bus: TranscriptBus,
    buses: RegionBuses,
    region: usize,
    round_count: usize,
    opened_column_count: usize,
    point_fixed_source_count: usize,
    point_common_count: usize,
    evaluation_encoder: Encoder,
}

impl FixedMultiAirCompleteRegionSumcheckCore {
    #[allow(clippy::too_many_arguments)]
    fn new(
        transcript_bus: TranscriptBus,
        buses: RegionBuses,
        region: usize,
        round_count: usize,
        opened_column_count: usize,
        point_fixed_source_count: usize,
        point_common_count: usize,
    ) -> Result<Self, FixedMultiAirCompleteRegionSumcheckError> {
        validate_field_word(region, "region")?;
        validate_field_word(round_count, "round count")?;
        validate_field_word(opened_column_count, "opened-column count")?;
        validate_field_word(point_fixed_source_count, "fixed point-source count")?;
        validate_field_word(point_common_count, "common point-source count")?;

        // This validates the represented Boolean-domain height and every
        // setup-fixed lookup multiplicity without an unchecked shift.
        1usize
            .checked_shl(
                u32::try_from(round_count)
                    .map_err(|_| FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?,
            )
            .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?;
        if round_count != 0 {
            for round in 0..round_count {
                point_lookup_count(
                    round_count,
                    round,
                    point_fixed_source_count,
                    point_common_count,
                )?;
            }
        }
        round_count
            .checked_mul(FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS)
            .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?;
        transcript_span(round_count)?;

        let evaluation_encoder = Encoder::new(
            FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS,
            SUMCHECK_SELECTOR_MAX_DEGREE,
            false,
        );
        if evaluation_encoder.width() != EVALUATION_ENCODER_WIDTH {
            return Err(
                FixedMultiAirCompleteRegionSumcheckError::InvalidConfiguration(
                    "evaluation encoder width",
                ),
            );
        }
        Ok(Self {
            transcript_bus,
            buses,
            region,
            round_count,
            opened_column_count,
            point_fixed_source_count,
            point_common_count,
            evaluation_encoder,
        })
    }

    fn transcript_span(&self) -> usize {
        // Construction validates this exact expression.
        FIXED_MULTI_AIR_COMPLETE_REGION_PREFIX_BASE_VALUES
            + self.round_count * FIXED_MULTI_AIR_COMPLETE_ROUND_TRANSCRIPT_BASE_VALUES
    }

    fn eval_zero_round<AB>(&self, builder: &mut AB)
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
    {
        let main = builder.main();
        let local_row = main
            .row_slice(0)
            .expect("complete zero-round regional bridge row");
        let next_row = main
            .row_slice(1)
            .expect("complete next zero-round regional bridge row");
        let local: &FixedMultiAirCompleteRegionSumcheckCols<AB::Var> = (*local_row).borrow();
        let next: &FixedMultiAirCompleteRegionSumcheckCols<AB::Var> = (*next_row).borrow();

        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .when(local.active)
            .assert_zero(next.active);

        for value in [
            local.round,
            local.evaluation_index,
            local.is_first_evaluation,
            local.is_last_evaluation,
            local.is_first_round,
            local.is_last_round,
            local.is_final,
            local.point_lookup_count,
            local.round_tidx,
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

        self.observe_region_prefix(builder, local.region_tidx.into(), local.active.into());
        self.buses.receive_start(
            builder,
            FixedMultiAirCompleteRegionStartMessage {
                region: AB::Expr::from_usize(self.region),
                tidx: local.region_tidx.into(),
                claim: local.pre_claim.map(Into::into),
            },
            local.active,
        );
        self.buses.send_final(
            builder,
            FixedMultiAirCompleteRegionSumcheckFinalMessage {
                region: AB::Expr::from_usize(self.region),
                tidx: AB::Expr::from(local.region_tidx)
                    + AB::Expr::from_usize(FIXED_MULTI_AIR_COMPLETE_REGION_PREFIX_BASE_VALUES),
                claim: local.post_claim.map(Into::into),
            },
            local.active,
        );
    }

    fn eval_rounds<AB>(&self, builder: &mut AB)
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
        <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
    {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("complete regional sumcheck row");
        let next_row = main
            .row_slice(1)
            .expect("complete next regional sumcheck row");
        let local: &FixedMultiAirCompleteRegionSumcheckCols<AB::Var> = (*local_row).borrow();
        let next: &FixedMultiAirCompleteRegionSumcheckCols<AB::Var> = (*next_row).borrow();

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
        let indices = (0..FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS)
            .map(|index| (index, index))
            .collect::<Vec<_>>();
        let first = (0..FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS)
            .map(|index| (index, usize::from(index == 0)))
            .collect::<Vec<_>>();
        let last = (0..FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS)
            .map(|index| {
                (
                    index,
                    usize::from(index + 1 == FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS),
                )
            })
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
        builder.when(local.active * local.is_first_round).assert_eq(
            local.point_lookup_count,
            AB::Expr::from_usize(
                point_lookup_count(
                    self.round_count,
                    0,
                    self.point_fixed_source_count,
                    self.point_common_count,
                )
                .expect("setup-validated complete point lookup count"),
            ),
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
        builder.when_first_row().assert_eq(
            local.round_tidx,
            AB::Expr::from(local.region_tidx)
                + AB::Expr::from_usize(FIXED_MULTI_AIR_COMPLETE_REGION_PREFIX_BASE_VALUES),
        );
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
        same.assert_eq(next.region_tidx, local.region_tidx);
        same.assert_eq(next.round_tidx, local.round_tidx);
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
        advance.assert_eq(next.region_tidx, local.region_tidx);
        advance.assert_eq(
            next.round_tidx,
            local.round_tidx
                + AB::Expr::from_usize(FIXED_MULTI_AIR_COMPLETE_ROUND_TRANSCRIPT_BASE_VALUES),
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
        let at_one = self.evaluation_encoder.flag_with_val::<AB>(
            &local.evaluation_encoding,
            &(0..FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS)
                .map(|index| (index, usize::from(index == 1)))
                .collect::<Vec<_>>(),
        );
        assert_array_eq(
            &mut builder.when(local.active * at_one),
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
        for other in 0..FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS {
            let skip = self.evaluation_encoder.flag_with_val::<AB>(
                &local.evaluation_encoding,
                &(0..FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS)
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
            &(0..FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS)
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
        let scaled_basis = local.basis_prefix[FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS]
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
            &mut builder.when(local.active * local.is_first_evaluation),
            local.sum_before,
            [AB::Expr::ZERO; D_EF],
        );
        assert_array_eq(
            &mut builder.when(local.active * local.is_last_evaluation),
            local.sum_after,
            local.post_claim.map(Into::into),
        );

        let first_row = local.active * local.is_first_round * local.is_first_evaluation;
        self.observe_region_prefix(builder, local.region_tidx.into(), first_row.clone());
        let first_eval = local.active * local.is_first_evaluation;
        observe_ext_const(
            &self.transcript_bus,
            builder,
            local.round_tidx.into(),
            COMPLETE_TERMINAL_ROUND_TAG,
            first_eval.clone(),
        );
        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.round_tidx) + AB::Expr::from_usize(D_EF),
            ext_from_base_expr::<AB>(local.round.into()),
            first_eval.clone(),
        );
        observe_ext_const(
            &self.transcript_bus,
            builder,
            AB::Expr::from(local.round_tidx) + AB::Expr::from_usize(2 * D_EF),
            FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS as u64,
            first_eval,
        );
        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.round_tidx)
                + AB::Expr::from_usize(3 * D_EF)
                + AB::Expr::from(local.evaluation_index) * AB::Expr::from_usize(D_EF),
            local.evaluation,
            local.active,
        );
        self.transcript_bus.sample_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.round_tidx)
                + AB::Expr::from_usize((3 + FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS) * D_EF),
            local.challenge,
            local.active * local.is_last_evaluation,
        );

        self.buses.receive_start(
            builder,
            FixedMultiAirCompleteRegionStartMessage {
                region: AB::Expr::from_usize(self.region),
                tidx: local.region_tidx.into(),
                claim: local.pre_claim.map(Into::into),
            },
            first_row,
        );
        self.buses.publish_point(
            builder,
            FixedMultiAirCompleteRegionPointMessage {
                region: AB::Expr::from_usize(self.region),
                coordinate: local.round.into(),
                value: local.challenge.map(Into::into),
            },
            local.active * local.is_last_evaluation * local.point_lookup_count,
        );
        self.buses.send_final(
            builder,
            FixedMultiAirCompleteRegionSumcheckFinalMessage {
                region: AB::Expr::from_usize(self.region),
                tidx: AB::Expr::from(local.round_tidx)
                    + AB::Expr::from_usize(FIXED_MULTI_AIR_COMPLETE_ROUND_TRANSCRIPT_BASE_VALUES),
                claim: local.post_claim.map(Into::into),
            },
            local.active * local.is_final,
        );
    }

    fn observe_region_prefix<AB>(&self, builder: &mut AB, tidx: AB::Expr, enabled: AB::Expr)
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
    {
        observe_ext_const(
            &self.transcript_bus,
            builder,
            tidx.clone(),
            self.buses.domain_tag(),
            enabled.clone(),
        );
        observe_ext_const(
            &self.transcript_bus,
            builder,
            tidx + AB::Expr::from_usize(D_EF),
            self.region as u64,
            enabled,
        );
    }
}

/// Exact complete-terminal local-region sumcheck AIR. Its type fixes the local
/// transcript domain and local-only typed buses at setup.
pub struct FixedMultiAirCompleteLocalRegionSumcheckAir {
    core: FixedMultiAirCompleteRegionSumcheckCore,
}

impl FixedMultiAirCompleteLocalRegionSumcheckAir {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        transcript_bus: TranscriptBus,
        start_bus: FixedMultiAirCompleteLocalRegionStartBus,
        point_bus: FixedMultiAirCompleteLocalRegionPointBus,
        final_bus: FixedMultiAirCompleteLocalRegionSumcheckFinalBus,
        region: usize,
        round_count: usize,
        opened_column_count: usize,
        point_fixed_source_count: usize,
        point_common_count: usize,
    ) -> Result<Self, FixedMultiAirCompleteRegionSumcheckError> {
        Ok(Self {
            core: FixedMultiAirCompleteRegionSumcheckCore::new(
                transcript_bus,
                RegionBuses::Local(LocalRegionBuses {
                    start: start_bus,
                    point: point_bus,
                    final_claim: final_bus,
                }),
                region,
                round_count,
                opened_column_count,
                point_fixed_source_count,
                point_common_count,
            )?,
        })
    }

    #[must_use]
    pub const fn region(&self) -> usize {
        self.core.region
    }

    #[must_use]
    pub const fn round_count(&self) -> usize {
        self.core.round_count
    }

    #[must_use]
    pub const fn opened_column_count(&self) -> usize {
        self.core.opened_column_count
    }

    #[must_use]
    pub fn transcript_span(&self) -> usize {
        self.core.transcript_span()
    }
}

/// Exact complete-terminal interaction-region sumcheck AIR. Its type fixes
/// the interaction transcript domain and interaction-only typed buses.
pub struct FixedMultiAirCompleteInteractionRegionSumcheckAir {
    core: FixedMultiAirCompleteRegionSumcheckCore,
}

impl FixedMultiAirCompleteInteractionRegionSumcheckAir {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        transcript_bus: TranscriptBus,
        start_bus: FixedMultiAirCompleteInteractionRegionStartBus,
        point_bus: FixedMultiAirCompleteInteractionRegionPointBus,
        final_bus: FixedMultiAirCompleteInteractionRegionSumcheckFinalBus,
        region: usize,
        round_count: usize,
        opened_column_count: usize,
        point_fixed_source_count: usize,
        point_common_count: usize,
    ) -> Result<Self, FixedMultiAirCompleteRegionSumcheckError> {
        Ok(Self {
            core: FixedMultiAirCompleteRegionSumcheckCore::new(
                transcript_bus,
                RegionBuses::Interaction(InteractionRegionBuses {
                    start: start_bus,
                    point: point_bus,
                    final_claim: final_bus,
                }),
                region,
                round_count,
                opened_column_count,
                point_fixed_source_count,
                point_common_count,
            )?,
        })
    }

    #[must_use]
    pub const fn region(&self) -> usize {
        self.core.region
    }

    #[must_use]
    pub const fn round_count(&self) -> usize {
        self.core.round_count
    }

    #[must_use]
    pub const fn opened_column_count(&self) -> usize {
        self.core.opened_column_count
    }

    #[must_use]
    pub fn transcript_span(&self) -> usize {
        self.core.transcript_span()
    }
}

macro_rules! impl_complete_region_air {
    ($air:ty) => {
        impl BaseAirWithPublicValues<F> for $air {}
        impl PartitionedBaseAir<F> for $air {}
        impl ColumnsAir for $air {}
        impl BaseAir<F> for $air {
            fn width(&self) -> usize {
                FixedMultiAirCompleteRegionSumcheckCols::<F>::width()
            }
        }
        impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for $air
        where
            <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
        {
            fn eval(&self, builder: &mut AB) {
                if self.core.round_count == 0 {
                    self.core.eval_zero_round(builder);
                } else {
                    self.core.eval_rounds(builder);
                }
            }
        }
    };
}

impl_complete_region_air!(FixedMultiAirCompleteLocalRegionSumcheckAir);
impl_complete_region_air!(FixedMultiAirCompleteInteractionRegionSumcheckAir);

/// Generate an exact local-region trace directly from the genuine complete
/// proof type. No legacy/direct proof conversion exists in this API.
pub fn generate_fixed_multi_air_complete_local_region_sumcheck_trace(
    air: &FixedMultiAirCompleteLocalRegionSumcheckAir,
    initial_claim: EF,
    proof: &FixedMultiAirCompleteRegionSumcheckProof<EF>,
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
    required_height: Option<usize>,
) -> Result<RowMajorMatrix<F>, FixedMultiAirCompleteRegionSumcheckError> {
    generate_trace(
        &air.core,
        initial_claim,
        proof,
        transcript,
        start_tidx,
        required_height,
    )
}

/// Generate an exact interaction-region trace directly from the genuine
/// complete proof type. The setup-fixed interaction domain is not a witness.
pub fn generate_fixed_multi_air_complete_interaction_region_sumcheck_trace(
    air: &FixedMultiAirCompleteInteractionRegionSumcheckAir,
    initial_claim: EF,
    proof: &FixedMultiAirCompleteRegionSumcheckProof<EF>,
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
    required_height: Option<usize>,
) -> Result<RowMajorMatrix<F>, FixedMultiAirCompleteRegionSumcheckError> {
    generate_trace(
        &air.core,
        initial_claim,
        proof,
        transcript,
        start_tidx,
        required_height,
    )
}

/// Recover the exact regional Boolean-cube point from the authenticated
/// transcript schedule. Aggregate trace owners use this instead of parsing
/// challenge coordinates through an independently reconstructed cursor.
pub fn fixed_multi_air_complete_region_sumcheck_point(
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
    round_count: usize,
) -> Result<Vec<EF>, FixedMultiAirCompleteRegionSumcheckError> {
    (0..round_count)
        .map(|round| {
            let tidx = FIXED_MULTI_AIR_COMPLETE_REGION_PREFIX_BASE_VALUES
                .checked_add(
                    round
                        .checked_mul(FIXED_MULTI_AIR_COMPLETE_ROUND_TRANSCRIPT_BASE_VALUES)
                        .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?,
                )
                .and_then(|offset| {
                    offset.checked_add((3 + FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS) * D_EF)
                })
                .and_then(|offset| start_tidx.checked_add(offset))
                .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?;
            read_ext(transcript, tidx, true)
        })
        .collect()
}

fn generate_trace(
    air: &FixedMultiAirCompleteRegionSumcheckCore,
    initial_claim: EF,
    proof: &FixedMultiAirCompleteRegionSumcheckProof<EF>,
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
    required_height: Option<usize>,
) -> Result<RowMajorMatrix<F>, FixedMultiAirCompleteRegionSumcheckError> {
    if proof.round_evaluations.len() != air.round_count {
        return Err(FixedMultiAirCompleteRegionSumcheckError::ProofShape(
            "round count",
        ));
    }
    if proof.opened_columns.len() != air.opened_column_count {
        return Err(FixedMultiAirCompleteRegionSumcheckError::ProofShape(
            "opened-column count",
        ));
    }
    if proof
        .round_evaluations
        .iter()
        .any(|values| values.len() != FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS)
    {
        return Err(FixedMultiAirCompleteRegionSumcheckError::ProofShape(
            "round evaluation count",
        ));
    }

    let span = transcript_span(air.round_count)?;
    let end_tidx = start_tidx
        .checked_add(span)
        .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?;
    if end_tidx >= F::ORDER_U32 as usize {
        return Err(FixedMultiAirCompleteRegionSumcheckError::Transcript(
            "transcript index exceeds base field",
        ));
    }
    expect_ext(
        transcript,
        start_tidx,
        EF::from_u64(air.buses.domain_tag()),
        false,
    )?;
    expect_ext(
        transcript,
        start_tidx
            .checked_add(D_EF)
            .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?,
        EF::from_usize(air.region),
        false,
    )?;

    let valid_rows = if air.round_count == 0 {
        1
    } else {
        air.round_count
            .checked_mul(FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS)
            .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?
    };
    let default_height = valid_rows
        .checked_next_power_of_two()
        .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?;
    let height = required_height.unwrap_or(default_height);
    if height == 0 || !height.is_power_of_two() || height < valid_rows {
        return Err(FixedMultiAirCompleteRegionSumcheckError::TraceHeight);
    }
    let width = FixedMultiAirCompleteRegionSumcheckCols::<F>::width();
    let cell_count = height
        .checked_mul(width)
        .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?;
    let mut trace = Vec::new();
    trace
        .try_reserve_exact(cell_count)
        .map_err(|_| FixedMultiAirCompleteRegionSumcheckError::Allocation)?;
    trace.resize(cell_count, F::ZERO);

    if air.round_count == 0 {
        let cols: &mut FixedMultiAirCompleteRegionSumcheckCols<F> = trace[..width].borrow_mut();
        cols.active = F::ONE;
        cols.region_tidx = F::from_usize(start_tidx);
        copy_ext(&mut cols.pre_claim, initial_claim);
        copy_ext(&mut cols.post_claim, initial_claim);
        return Ok(RowMajorMatrix::new(trace, width));
    }

    let mut pre_claim = initial_claim;
    let mut round_tidx = start_tidx
        .checked_add(FIXED_MULTI_AIR_COMPLETE_REGION_PREFIX_BASE_VALUES)
        .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?;
    let mut row_index = 0usize;
    for (round, values) in proof.round_evaluations.iter().enumerate() {
        let evaluations: &[EF; FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS] =
            values.as_slice().try_into().map_err(|_| {
                FixedMultiAirCompleteRegionSumcheckError::ProofShape("round evaluation count")
            })?;
        expect_ext(
            transcript,
            round_tidx,
            EF::from_u64(COMPLETE_TERMINAL_ROUND_TAG),
            false,
        )?;
        expect_ext(
            transcript,
            round_tidx
                .checked_add(D_EF)
                .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?,
            EF::from_usize(round),
            false,
        )?;
        expect_ext(
            transcript,
            round_tidx
                .checked_add(2 * D_EF)
                .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?,
            EF::from_usize(FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS),
            false,
        )?;
        for (index, &value) in evaluations.iter().enumerate() {
            let offset = (3 + index)
                .checked_mul(D_EF)
                .and_then(|offset| round_tidx.checked_add(offset))
                .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?;
            expect_ext(transcript, offset, value, false)?;
        }
        let sample_tidx = (3 + FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS)
            .checked_mul(D_EF)
            .and_then(|offset| round_tidx.checked_add(offset))
            .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?;
        let challenge = read_ext(transcript, sample_tidx, true)?;
        if evaluations[0] + evaluations[1] != pre_claim {
            return Err(FixedMultiAirCompleteRegionSumcheckError::Claim { round });
        }
        let post_claim = interpolate(evaluations, challenge);
        let lookup_count = point_lookup_count(
            air.round_count,
            round,
            air.point_fixed_source_count,
            air.point_common_count,
        )?;
        let mut sum = EF::ZERO;
        for (evaluation_index, &evaluation) in evaluations.iter().enumerate() {
            let mut basis = EF::ONE;
            let mut prefixes = [EF::ZERO; FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS + 1];
            prefixes[0] = basis;
            for other in 0..FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS {
                if other != evaluation_index {
                    basis *= challenge - EF::from_usize(other);
                }
                prefixes[other + 1] = basis;
            }
            let denominator_inverse = lagrange_denominator(evaluation_index).inverse();
            let term = evaluation * basis * EF::from(denominator_inverse);
            let sum_before = sum;
            sum += term;
            let row_start = row_index
                .checked_mul(width)
                .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?;
            let row_end = row_start
                .checked_add(width)
                .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?;
            let row = trace
                .get_mut(row_start..row_end)
                .ok_or(FixedMultiAirCompleteRegionSumcheckError::TraceHeight)?;
            let cols: &mut FixedMultiAirCompleteRegionSumcheckCols<F> = row.borrow_mut();
            cols.active = F::ONE;
            cols.round = F::from_usize(round);
            cols.evaluation_index = F::from_usize(evaluation_index);
            cols.is_first_evaluation = F::from_bool(evaluation_index == 0);
            cols.is_last_evaluation =
                F::from_bool(evaluation_index + 1 == FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS);
            cols.is_first_round = F::from_bool(round == 0);
            cols.is_last_round = F::from_bool(round + 1 == air.round_count);
            cols.is_final = F::from_bool(
                round + 1 == air.round_count
                    && evaluation_index + 1 == FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS,
            );
            cols.point_lookup_count = F::from_usize(lookup_count);
            cols.region_tidx = F::from_usize(start_tidx);
            cols.round_tidx = F::from_usize(round_tidx);
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
            return Err(FixedMultiAirCompleteRegionSumcheckError::Claim { round });
        }
        pre_claim = post_claim;
        round_tidx = round_tidx
            .checked_add(FIXED_MULTI_AIR_COMPLETE_ROUND_TRANSCRIPT_BASE_VALUES)
            .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?;
    }
    if round_tidx != end_tidx {
        return Err(FixedMultiAirCompleteRegionSumcheckError::Transcript(
            "round transcript endpoint",
        ));
    }
    Ok(RowMajorMatrix::new(trace, width))
}

fn validate_field_word(
    value: usize,
    name: &'static str,
) -> Result<(), FixedMultiAirCompleteRegionSumcheckError> {
    if value >= F::ORDER_U32 as usize {
        return Err(FixedMultiAirCompleteRegionSumcheckError::InvalidConfiguration(name));
    }
    Ok(())
}

fn transcript_span(round_count: usize) -> Result<usize, FixedMultiAirCompleteRegionSumcheckError> {
    round_count
        .checked_mul(FIXED_MULTI_AIR_COMPLETE_ROUND_TRANSCRIPT_BASE_VALUES)
        .and_then(|rounds| FIXED_MULTI_AIR_COMPLETE_REGION_PREFIX_BASE_VALUES.checked_add(rounds))
        .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)
}

fn point_lookup_count(
    round_count: usize,
    round: usize,
    fixed_source_count: usize,
    common_count: usize,
) -> Result<usize, FixedMultiAirCompleteRegionSumcheckError> {
    let exponent = round_count
        .checked_sub(round)
        .and_then(|remaining| remaining.checked_sub(1))
        .ok_or(
            FixedMultiAirCompleteRegionSumcheckError::InvalidConfiguration("point lookup round"),
        )?;
    let layer = 1usize
        .checked_shl(
            u32::try_from(exponent)
                .map_err(|_| FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?,
        )
        .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?;
    let count = fixed_source_count
        .checked_mul(layer)
        .and_then(|count| count.checked_add(common_count))
        .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?;
    validate_field_word(count, "point lookup multiplicity")?;
    Ok(count)
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
    (0..FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS)
        .filter(|&other| other != index)
        .map(|other| F::from_usize(index) - F::from_usize(other))
        .product()
}

fn interpolate(
    evaluations: &[EF; FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS],
    challenge: EF,
) -> EF {
    evaluations
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let numerator = (0..FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS)
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
) -> Result<EF, FixedMultiAirCompleteRegionSumcheckError> {
    let end = tidx
        .checked_add(D_EF)
        .ok_or(FixedMultiAirCompleteRegionSumcheckError::HeightOverflow)?;
    let values = transcript.values().get(tidx..end).ok_or(
        FixedMultiAirCompleteRegionSumcheckError::Transcript("operation range"),
    )?;
    let flags = transcript.samples().get(tidx..end).ok_or(
        FixedMultiAirCompleteRegionSumcheckError::Transcript("sample-flag range"),
    )?;
    if flags.iter().any(|&flag| flag != is_sample) {
        return Err(FixedMultiAirCompleteRegionSumcheckError::Transcript(
            "sample flags",
        ));
    }
    EF::from_basis_coefficients_slice(values).ok_or(
        FixedMultiAirCompleteRegionSumcheckError::Transcript("extension element"),
    )
}

fn expect_ext(
    transcript: &TranscriptLog<F, [F; 16]>,
    tidx: usize,
    expected: EF,
    is_sample: bool,
) -> Result<(), FixedMultiAirCompleteRegionSumcheckError> {
    if read_ext(transcript, tidx, is_sample)? != expected {
        return Err(FixedMultiAirCompleteRegionSumcheckError::Transcript(
            "operation value",
        ));
    }
    Ok(())
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

#[cfg(test)]
mod tests {
    use core::any::TypeId;
    use std::{
        panic::{catch_unwind, AssertUnwindSafe},
        sync::Arc,
    };

    use openvm_stark_backend::{
        air_builders::{
            debug::{check_constraints, check_logup},
            symbolic::{
                get_symbolic_builder, symbolic_expression::SymbolicExpression,
                SymbolicConstraintsDag, SymbolicRapBuilder,
            },
            PartitionedAirBuilder,
        },
        hasher::MerkleHasher,
        interaction::{InteractionBuilder, SymbolicInteraction},
        keygen::types::{StarkVerifyingKey, StarkVerifyingParams, TraceWidth},
        native_warp::{
            DirectAirCodeClass, DirectAirPesatIndex, DirectAirPesatInstance, DirectAirPublicSchema,
            FixedMultiAirCompletePesatIndex, FixedMultiAirCompletePesatInstance,
            FixedMultiAirCompleteSourceRegion, FixedMultiAirCompleteTerminalLinearizer,
            NativeWarpChallenger, COMPLETE_TERMINAL_INTERACTION_REGION_TAG,
            COMPLETE_TERMINAL_LOCAL_REGION_TAG,
        },
        transcript::{TranscriptHistory, TranscriptLog},
        warp_pesat::{evaluate_mle, AccumulatorInstance, StructuredTerminalPesatLinearizer},
        BaseAirWithPublicValues, PartitionedBaseAir, StarkProtocolConfig, SystemParams,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::{
        default_duplex_sponge_recorder, BabyBearPoseidon2Config as SC, DIGEST_SIZE,
    };
    use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
    use p3_field::PrimeCharacteristicRing;

    use super::*;
    use crate::{
        bus::{TranscriptBus, TranscriptBusMessage},
        native_warp::terminal::fixed_multi_air_complete::{
            FixedMultiAirCompleteInteractionRegionPointBus,
            FixedMultiAirCompleteInteractionRegionStartBus,
            FixedMultiAirCompleteInteractionRegionSumcheckFinalBus,
            FixedMultiAirCompleteLocalRegionPointBus, FixedMultiAirCompleteLocalRegionStartBus,
            FixedMultiAirCompleteLocalRegionSumcheckFinalBus,
            FixedMultiAirCompleteRegionPointMessage, FixedMultiAirCompleteRegionStartMessage,
            FixedMultiAirCompleteRegionSumcheckFinalMessage,
        },
    };

    type Digest = [F; DIGEST_SIZE];

    #[derive(Clone, Copy)]
    struct TestLogupAir {
        send: bool,
        bus: u16,
    }

    impl BaseAir<F> for TestLogupAir {
        fn width(&self) -> usize {
            2
        }
    }

    impl BaseAirWithPublicValues<F> for TestLogupAir {
        fn num_public_values(&self) -> usize {
            1
        }
    }

    impl PartitionedBaseAir<F> for TestLogupAir {}

    impl Air<SymbolicRapBuilder<F>> for TestLogupAir {
        fn eval(&self, builder: &mut SymbolicRapBuilder<F>) {
            let main = builder.common_main().clone();
            let local = main.row_slice(0).expect("test LogUp row");
            let value = local[0];
            let multiplicity = local[1];
            builder.assert_zero(value - builder.public_values()[0]);
            let count = SymbolicExpression::from(multiplicity);
            let count = if self.send { count } else { -count };
            builder.push_interaction(self.bus, [value, value], count, 1);
        }
    }

    fn digest(value: u32) -> Digest {
        [F::from_u32(value); DIGEST_SIZE]
    }

    fn verifying_key(air: &TestLogupAir) -> StarkVerifyingKey<F, Digest> {
        let width = TraceWidth {
            preprocessed: None,
            cached_mains: Vec::new(),
            common_main: 2,
        };
        let symbolic = get_symbolic_builder(air, &width).constraints();
        StarkVerifyingKey {
            preprocessed_data: None,
            params: StarkVerifyingParams {
                width,
                num_public_values: 1,
                need_rot: false,
            },
            max_constraint_degree: symbolic.max_constraint_degree() as u8,
            symbolic_constraints: Arc::new(SymbolicConstraintsDag::from(symbolic)),
            is_required: true,
            unused_variables: Vec::new(),
        }
    }

    fn direct_relation<H>(
        hasher: &H,
        air: &TestLogupAir,
        air_id: usize,
        log_height: usize,
    ) -> DirectAirPesatIndex<F, Digest>
    where
        H: MerkleHasher<F = F, Digest = Digest>,
    {
        DirectAirPesatIndex::from_verifying_key(
            hasher,
            digest(1),
            air_id,
            log_height,
            &verifying_key(air),
            None,
            DirectAirPublicSchema {
                public_values_len: 1,
                boundary_values_len: 0,
                schema_digest: digest(100 + air_id as u32),
            },
            DirectAirCodeClass {
                log_message_len: u8::try_from(log_height + 1).expect("small fixture height"),
                log_blowup: 1,
                log_codeword_len: u8::try_from(log_height + 2).expect("small fixture height"),
                initial_folding_factor: 0,
                rows_per_query: 2,
            },
        )
        .expect("direct complete-terminal test relation")
    }

    struct BackendFixture {
        local_claim: EF,
        interaction_claim: EF,
        local_proof: FixedMultiAirCompleteRegionSumcheckProof<EF>,
        interaction_proof: FixedMultiAirCompleteRegionSumcheckProof<EF>,
        transcript: TranscriptLog<F, [F; 16]>,
        local_start: usize,
        interaction_start: usize,
    }

    fn backend_fixture(log_height: usize) -> BackendFixture {
        let config = SC::default_from_params(SystemParams::new_for_testing(8));
        let hasher = config.hasher();
        // Two width-two regions plus one four-coordinate interaction inverse
        // per row occupy 3 * 2^(log_height + 2) cells, whose canonical padded
        // complete-relation message length is 2^(log_height + 4).
        let complete_log_message_len =
            u8::try_from(log_height + 4).expect("small fixture complete message");
        let relation = FixedMultiAirCompletePesatIndex::from_direct_air_regions(
            hasher,
            digest(1),
            vec![
                direct_relation(hasher, &TestLogupAir { send: true, bus: 7 }, 3, log_height),
                direct_relation(
                    hasher,
                    &TestLogupAir {
                        send: false,
                        bus: 7,
                    },
                    9,
                    log_height,
                ),
            ],
            DirectAirCodeClass {
                log_message_len: complete_log_message_len,
                log_blowup: 1,
                log_codeword_len: complete_log_message_len + 1,
                initial_folding_factor: 0,
                rows_per_query: 2,
            },
        )
        .expect("fixed complete relation");
        let height = 1usize << log_height;
        let source = |air_id| FixedMultiAirCompleteSourceRegion {
            air_id,
            public_values: vec![F::from_u32(7)],
            boundary_values: Vec::new(),
            common_trace_cells: (0..height)
                .map(|_| F::from_u32(7))
                .chain((0..height).map(|_| F::ONE))
                .collect(),
        };
        let public = FixedMultiAirCompletePesatInstance::from_alpha_beta(
            EF::from_u32(11),
            EF::from_u32(5),
            vec![
                DirectAirPesatInstance {
                    public_values: vec![F::from_u32(7)],
                    boundary_values: Vec::new(),
                },
                DirectAirPesatInstance {
                    public_values: vec![F::from_u32(7)],
                    boundary_values: Vec::new(),
                },
            ],
            2,
        );
        let witness = relation
            .synthesize_witness(&public, &[source(3), source(9)])
            .expect("complete witness");
        let message = relation
            .padded_witness::<EF>(&witness)
            .expect("padded complete witness");
        let explicit = relation
            .explicit_assignment(&public)
            .expect("complete explicit assignment");
        let constraints = relation
            .evaluate_reference(&public, &witness)
            .expect("complete relation evaluation");
        assert!(constraints.iter().all(|&value| value == EF::ZERO));
        let tau = (0..relation.pesat_shape().log_constraints)
            .map(|index| EF::from_usize(17 + index))
            .collect::<Vec<_>>();
        let eta = evaluate_mle(&constraints, &tau);
        let mut beta = tau;
        beta.extend(explicit.into_iter().map(EF::from));
        let instance = AccumulatorInstance {
            rt: digest(77),
            alpha: (0..usize::from(complete_log_message_len + 1))
                .map(|index| EF::from_usize(31 + index))
                .collect(),
            mu: EF::from_u32(41),
            beta,
            eta,
        };
        let linearizer =
            FixedMultiAirCompleteTerminalLinearizer::new(&relation).expect("complete linearizer");
        let mut challenger = NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
        let (proof, claims) = linearizer
            .prove_from_reader(&instance, &message, &mut challenger)
            .expect("backend complete terminal proof");
        let transcript = TranscriptHistory::into_log(challenger.into_inner());
        let mut verifier = NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
        let verified_claims = linearizer
            .verify_structured_terminal_claims(&instance, &proof, &mut verifier)
            .expect("backend complete terminal verification");
        assert_eq!(verified_claims, claims);
        let verifier_transcript = TranscriptHistory::into_log(verifier.into_inner());
        assert_eq!(transcript.values(), verifier_transcript.values());
        assert_eq!(transcript.samples(), verifier_transcript.samples());
        let local_start = find_region_start(&transcript, COMPLETE_TERMINAL_LOCAL_REGION_TAG, 0);
        let interaction_start =
            find_region_start(&transcript, COMPLETE_TERMINAL_INTERACTION_REGION_TAG, 0);
        BackendFixture {
            local_claim: proof.local_claims[0],
            interaction_claim: proof.interaction_claims[0],
            local_proof: proof.local_proofs[0].clone(),
            interaction_proof: proof.interaction_proofs[0]
                .clone()
                .expect("nonempty interaction proof"),
            transcript,
            local_start,
            interaction_start,
        }
    }

    fn find_region_start(transcript: &TranscriptLog<F, [F; 16]>, tag: u64, region: usize) -> usize {
        let tag = EF::from_u64(tag);
        let region = EF::from_usize(region);
        let tag_limbs = tag.as_basis_coefficients_slice();
        let region_limbs = region.as_basis_coefficients_slice();
        transcript
            .values()
            .windows(2 * D_EF)
            .enumerate()
            .find_map(|(start, values)| {
                let flags = transcript.samples().get(start..start + 2 * D_EF)?;
                (values[..D_EF] == *tag_limbs
                    && values[D_EF..] == *region_limbs
                    && flags.iter().all(|&flag| !flag))
                .then_some(start)
            })
            .expect("backend region domain")
    }

    fn local_air(
        round_count: usize,
        opened_column_count: usize,
    ) -> FixedMultiAirCompleteLocalRegionSumcheckAir {
        FixedMultiAirCompleteLocalRegionSumcheckAir::new(
            TranscriptBus::new(1),
            FixedMultiAirCompleteLocalRegionStartBus::new(2),
            FixedMultiAirCompleteLocalRegionPointBus::new(3),
            FixedMultiAirCompleteLocalRegionSumcheckFinalBus::new(4),
            0,
            round_count,
            opened_column_count,
            0,
            1,
        )
        .expect("local complete sumcheck AIR")
    }

    fn interaction_air(
        round_count: usize,
        opened_column_count: usize,
    ) -> FixedMultiAirCompleteInteractionRegionSumcheckAir {
        FixedMultiAirCompleteInteractionRegionSumcheckAir::new(
            TranscriptBus::new(1),
            FixedMultiAirCompleteInteractionRegionStartBus::new(5),
            FixedMultiAirCompleteInteractionRegionPointBus::new(6),
            FixedMultiAirCompleteInteractionRegionSumcheckFinalBus::new(7),
            0,
            round_count,
            opened_column_count,
            0,
            1,
        )
        .expect("interaction complete sumcheck AIR")
    }

    fn check_air<A>(air: &A, trace: &RowMajorMatrix<F>)
    where
        A: for<'a> Air<openvm_stark_backend::air_builders::debug::DebugConstraintBuilder<'a, SC>>
            + BaseAir<F>
            + BaseAirWithPublicValues<F>
            + PartitionedBaseAir<F>,
    {
        check_constraints::<_, SC>(
            air,
            core::any::type_name::<A>(),
            &None,
            &[trace.as_view()],
            &[],
        );
    }

    fn challenge_at(transcript: &TranscriptLog<F, [F; 16]>, start_tidx: usize, round: usize) -> EF {
        let tidx = start_tidx
            + FIXED_MULTI_AIR_COMPLETE_REGION_PREFIX_BASE_VALUES
            + round * FIXED_MULTI_AIR_COMPLETE_ROUND_TRANSCRIPT_BASE_VALUES
            + (3 + FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS) * D_EF;
        read_ext(transcript, tidx, true).expect("backend regional challenge")
    }

    fn reference_final_claim(
        initial: EF,
        proof: &FixedMultiAirCompleteRegionSumcheckProof<EF>,
        transcript: &TranscriptLog<F, [F; 16]>,
        start_tidx: usize,
    ) -> EF {
        proof
            .round_evaluations
            .iter()
            .enumerate()
            .fold(initial, |running, (round, evaluations)| {
                assert_eq!(evaluations[0] + evaluations[1], running);
                let evaluations: &[EF; FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS] = evaluations
                    .as_slice()
                    .try_into()
                    .expect("eight evaluations");
                interpolate(evaluations, challenge_at(transcript, start_tidx, round))
            })
    }

    fn trace_final_claim(trace: &RowMajorMatrix<F>) -> EF {
        let width = trace.width();
        trace
            .values
            .chunks_exact(width)
            .filter_map(|row| {
                let cols: &FixedMultiAirCompleteRegionSumcheckCols<F> = row.borrow();
                (cols.active == F::ONE && cols.is_final == F::ONE)
                    .then(|| EF::from_basis_coefficients_slice(&cols.post_claim).unwrap())
            })
            .next()
            .or_else(|| {
                let cols: &FixedMultiAirCompleteRegionSumcheckCols<F> =
                    trace.values[..width].borrow();
                (cols.active == F::ONE && cols.round_tidx == F::ZERO)
                    .then(|| EF::from_basis_coefficients_slice(&cols.post_claim).unwrap())
            })
            .expect("regional final claim")
    }

    #[test]
    fn backend_complete_local_and_interaction_transcripts_match_at_all_small_heights() {
        assert_ne!(
            TypeId::of::<FixedMultiAirCompleteLocalRegionSumcheckAir>(),
            TypeId::of::<FixedMultiAirCompleteInteractionRegionSumcheckAir>()
        );
        for log_height in [0, 1, 2, 3] {
            let fixture = backend_fixture(log_height);
            assert_eq!(fixture.local_proof.round_evaluations.len(), log_height);
            assert_eq!(
                fixture.interaction_proof.round_evaluations.len(),
                log_height
            );
            assert!(fixture
                .local_proof
                .round_evaluations
                .iter()
                .all(|values| { values.len() == FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS }));
            assert!(fixture
                .interaction_proof
                .round_evaluations
                .iter()
                .all(|values| { values.len() == FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS }));

            let local_air = local_air(log_height, fixture.local_proof.opened_columns.len());
            let local_trace = generate_fixed_multi_air_complete_local_region_sumcheck_trace(
                &local_air,
                fixture.local_claim,
                &fixture.local_proof,
                &fixture.transcript,
                fixture.local_start,
                None,
            )
            .expect("backend-differential local trace");
            check_air(&local_air, &local_trace);
            assert_eq!(
                trace_final_claim(&local_trace),
                reference_final_claim(
                    fixture.local_claim,
                    &fixture.local_proof,
                    &fixture.transcript,
                    fixture.local_start,
                )
            );

            let interaction_air =
                interaction_air(log_height, fixture.interaction_proof.opened_columns.len());
            let interaction_trace =
                generate_fixed_multi_air_complete_interaction_region_sumcheck_trace(
                    &interaction_air,
                    fixture.interaction_claim,
                    &fixture.interaction_proof,
                    &fixture.transcript,
                    fixture.interaction_start,
                    None,
                )
                .expect("backend-differential interaction trace");
            check_air(&interaction_air, &interaction_trace);
            assert_eq!(
                trace_final_claim(&interaction_trace),
                reference_final_claim(
                    fixture.interaction_claim,
                    &fixture.interaction_proof,
                    &fixture.transcript,
                    fixture.interaction_start,
                )
            );
            if log_height == 0 {
                assert_eq!(local_trace.height(), 1);
                assert_eq!(interaction_trace.height(), 1);
                assert_eq!(trace_final_claim(&local_trace), fixture.local_claim);
                assert_eq!(
                    trace_final_claim(&interaction_trace),
                    fixture.interaction_claim
                );
            }
        }
    }

    fn assert_error_without_panic<T>(
        action: impl FnOnce() -> Result<T, FixedMultiAirCompleteRegionSumcheckError>,
    ) {
        let result = catch_unwind(AssertUnwindSafe(action));
        assert!(result.is_ok(), "malformed input panicked");
        assert!(result.unwrap().is_err(), "malformed input was accepted");
    }

    #[test]
    fn malformed_dimensions_claims_transcript_and_heights_reject_without_panicking() {
        let fixture = backend_fixture(2);
        let air = local_air(2, fixture.local_proof.opened_columns.len());

        let mut malformed = fixture.local_proof.clone();
        malformed.round_evaluations.pop();
        assert_error_without_panic(|| {
            generate_fixed_multi_air_complete_local_region_sumcheck_trace(
                &air,
                fixture.local_claim,
                &malformed,
                &fixture.transcript,
                fixture.local_start,
                None,
            )
        });
        let mut malformed = fixture.local_proof.clone();
        malformed.round_evaluations.push(vec![
            EF::ZERO;
            FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS
        ]);
        assert_error_without_panic(|| {
            generate_fixed_multi_air_complete_local_region_sumcheck_trace(
                &air,
                fixture.local_claim,
                &malformed,
                &fixture.transcript,
                fixture.local_start,
                None,
            )
        });
        for evaluation_count in [0, 1, 7, 9] {
            let mut malformed = fixture.local_proof.clone();
            malformed.round_evaluations[0].resize(evaluation_count, EF::ZERO);
            assert_error_without_panic(|| {
                generate_fixed_multi_air_complete_local_region_sumcheck_trace(
                    &air,
                    fixture.local_claim,
                    &malformed,
                    &fixture.transcript,
                    fixture.local_start,
                    None,
                )
            });
        }
        let mut malformed = fixture.local_proof.clone();
        malformed.opened_columns.push(EF::ONE);
        assert_error_without_panic(|| {
            generate_fixed_multi_air_complete_local_region_sumcheck_trace(
                &air,
                fixture.local_claim,
                &malformed,
                &fixture.transcript,
                fixture.local_start,
                None,
            )
        });
        assert_error_without_panic(|| {
            generate_fixed_multi_air_complete_local_region_sumcheck_trace(
                &air,
                fixture.local_claim + EF::ONE,
                &fixture.local_proof,
                &fixture.transcript,
                fixture.local_start,
                None,
            )
        });
        assert_error_without_panic(|| {
            generate_fixed_multi_air_complete_local_region_sumcheck_trace(
                &air,
                fixture.local_claim,
                &fixture.local_proof,
                &fixture.transcript,
                fixture.local_start + 1,
                None,
            )
        });
        assert_error_without_panic(|| {
            generate_fixed_multi_air_complete_local_region_sumcheck_trace(
                &air,
                fixture.local_claim,
                &fixture.local_proof,
                &fixture.transcript,
                usize::MAX,
                None,
            )
        });
        for height in [0, 3, 8] {
            assert_error_without_panic(|| {
                generate_fixed_multi_air_complete_local_region_sumcheck_trace(
                    &air,
                    fixture.local_claim,
                    &fixture.local_proof,
                    &fixture.transcript,
                    fixture.local_start,
                    Some(height),
                )
            });
        }

        let prefix_flag = fixture.local_start;
        let challenge_flag = fixture.local_start
            + FIXED_MULTI_AIR_COMPLETE_REGION_PREFIX_BASE_VALUES
            + (3 + FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS) * D_EF;
        for index in [prefix_flag, challenge_flag] {
            let mut transcript = fixture.transcript.clone();
            transcript.samples_mut()[index] = !transcript.samples()[index];
            assert_error_without_panic(|| {
                generate_fixed_multi_air_complete_local_region_sumcheck_trace(
                    &air,
                    fixture.local_claim,
                    &fixture.local_proof,
                    &transcript,
                    fixture.local_start,
                    None,
                )
            });
        }

        let round_start = fixture.local_start + FIXED_MULTI_AIR_COMPLETE_REGION_PREFIX_BASE_VALUES;
        for offset in [0, D_EF, 2 * D_EF] {
            let mut transcript = fixture.transcript.clone();
            transcript.values_mut()[round_start + offset] += F::ONE;
            assert_error_without_panic(|| {
                generate_fixed_multi_air_complete_local_region_sumcheck_trace(
                    &air,
                    fixture.local_claim,
                    &fixture.local_proof,
                    &transcript,
                    fixture.local_start,
                    None,
                )
            });
        }

        assert_error_without_panic(|| {
            FixedMultiAirCompleteLocalRegionSumcheckAir::new(
                TranscriptBus::new(1),
                FixedMultiAirCompleteLocalRegionStartBus::new(2),
                FixedMultiAirCompleteLocalRegionPointBus::new(3),
                FixedMultiAirCompleteLocalRegionSumcheckFinalBus::new(4),
                0,
                usize::BITS as usize,
                0,
                1,
                1,
            )
        });
    }

    fn assert_constraints_reject<A>(air: &A, trace: RowMajorMatrix<F>)
    where
        A: for<'a> Air<openvm_stark_backend::air_builders::debug::DebugConstraintBuilder<'a, SC>>
            + BaseAir<F>
            + BaseAirWithPublicValues<F>
            + PartitionedBaseAir<F>,
    {
        assert!(catch_unwind(AssertUnwindSafe(|| check_air(air, &trace))).is_err());
    }

    #[test]
    fn every_evaluation_and_challenge_coordinate_is_constrained() {
        let fixture = backend_fixture(2);
        let air = local_air(2, fixture.local_proof.opened_columns.len());
        let trace = generate_fixed_multi_air_complete_local_region_sumcheck_trace(
            &air,
            fixture.local_claim,
            &fixture.local_proof,
            &fixture.transcript,
            fixture.local_start,
            None,
        )
        .unwrap();
        let width = trace.width();
        for evaluation in 0..FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS {
            for coordinate in 0..D_EF {
                let mut mutated = trace.clone();
                let row = &mut mutated.values[evaluation * width..(evaluation + 1) * width];
                let cols: &mut FixedMultiAirCompleteRegionSumcheckCols<F> = row.borrow_mut();
                cols.evaluation[coordinate] += F::ONE;
                assert_constraints_reject(&air, mutated);
            }
        }
        for coordinate in 0..D_EF {
            let mut mutated = trace.clone();
            let cols: &mut FixedMultiAirCompleteRegionSumcheckCols<F> =
                mutated.values[..width].borrow_mut();
            cols.challenge[coordinate] += F::ONE;
            assert_constraints_reject(&air, mutated);
        }
    }

    #[derive(Clone, Copy)]
    enum TestRegionBuses {
        Local {
            start: FixedMultiAirCompleteLocalRegionStartBus,
            point: FixedMultiAirCompleteLocalRegionPointBus,
            final_claim: FixedMultiAirCompleteLocalRegionSumcheckFinalBus,
        },
        Interaction {
            start: FixedMultiAirCompleteInteractionRegionStartBus,
            point: FixedMultiAirCompleteInteractionRegionPointBus,
            final_claim: FixedMultiAirCompleteInteractionRegionSumcheckFinalBus,
        },
    }

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct TranscriptOracleCols<T> {
        active: T,
        tidx: T,
        value: T,
        is_sample: T,
    }

    struct TranscriptOracleAir {
        bus: TranscriptBus,
    }

    impl BaseAir<F> for TranscriptOracleAir {
        fn width(&self) -> usize {
            TranscriptOracleCols::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for TranscriptOracleAir {}
    impl PartitionedBaseAir<F> for TranscriptOracleAir {}
    impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for TranscriptOracleAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("transcript oracle row");
            let local: &TranscriptOracleCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.active);
            builder.when(local.active).assert_bool(local.is_sample);
            self.bus.send(
                builder,
                AB::Expr::ZERO,
                TranscriptBusMessage {
                    tidx: local.tidx.into(),
                    value: local.value.into(),
                    is_sample: local.is_sample.into(),
                },
                local.active,
            );
        }
    }

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct RegionStartOracleCols<T> {
        active: T,
        region: T,
        tidx: T,
        claim: [T; D_EF],
    }

    struct RegionStartOracleAir {
        buses: TestRegionBuses,
    }

    impl BaseAir<F> for RegionStartOracleAir {
        fn width(&self) -> usize {
            RegionStartOracleCols::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for RegionStartOracleAir {}
    impl PartitionedBaseAir<F> for RegionStartOracleAir {}
    impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for RegionStartOracleAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("region start oracle row");
            let local: &RegionStartOracleCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.active);
            let message = FixedMultiAirCompleteRegionStartMessage {
                region: local.region.into(),
                tidx: local.tidx.into(),
                claim: local.claim.map(Into::into),
            };
            match self.buses {
                TestRegionBuses::Local { start, .. } => start.send(builder, message, local.active),
                TestRegionBuses::Interaction { start, .. } => {
                    start.send(builder, message, local.active)
                }
            }
        }
    }

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct RegionPointOracleCols<T> {
        active: T,
        region: T,
        coordinate: T,
        value: [T; D_EF],
    }

    struct RegionPointOracleAir {
        buses: TestRegionBuses,
    }

    impl BaseAir<F> for RegionPointOracleAir {
        fn width(&self) -> usize {
            RegionPointOracleCols::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for RegionPointOracleAir {}
    impl PartitionedBaseAir<F> for RegionPointOracleAir {}
    impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for RegionPointOracleAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("region point oracle row");
            let local: &RegionPointOracleCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.active);
            let message = FixedMultiAirCompleteRegionPointMessage {
                region: local.region.into(),
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            };
            match self.buses {
                TestRegionBuses::Local { point, .. } => {
                    point.lookup_key(builder, message, local.active)
                }
                TestRegionBuses::Interaction { point, .. } => {
                    point.lookup_key(builder, message, local.active)
                }
            }
        }
    }

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct RegionFinalOracleCols<T> {
        active: T,
        region: T,
        tidx: T,
        claim: [T; D_EF],
    }

    struct RegionFinalOracleAir {
        buses: TestRegionBuses,
    }

    impl BaseAir<F> for RegionFinalOracleAir {
        fn width(&self) -> usize {
            RegionFinalOracleCols::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for RegionFinalOracleAir {}
    impl PartitionedBaseAir<F> for RegionFinalOracleAir {}
    impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for RegionFinalOracleAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("region final oracle row");
            let local: &RegionFinalOracleCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.active);
            let message = FixedMultiAirCompleteRegionSumcheckFinalMessage {
                region: local.region.into(),
                tidx: local.tidx.into(),
                claim: local.claim.map(Into::into),
            };
            match self.buses {
                TestRegionBuses::Local { final_claim, .. } => {
                    final_claim.receive(builder, message, local.active)
                }
                TestRegionBuses::Interaction { final_claim, .. } => {
                    final_claim.receive(builder, message, local.active)
                }
            }
        }
    }

    fn transcript_oracle_trace(
        transcript: &TranscriptLog<F, [F; 16]>,
        start: usize,
        len: usize,
    ) -> RowMajorMatrix<F> {
        let width = TranscriptOracleCols::<F>::width();
        let height = len.max(1).next_power_of_two();
        let mut values = vec![F::ZERO; height * width];
        for ordinal in 0..len {
            let cols: &mut TranscriptOracleCols<F> =
                values[ordinal * width..(ordinal + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.tidx = F::from_usize(start + ordinal);
            cols.value = transcript.values()[start + ordinal];
            cols.is_sample = F::from_bool(transcript.samples()[start + ordinal]);
        }
        RowMajorMatrix::new(values, width)
    }

    fn start_oracle_trace(claim: EF, start: usize) -> RowMajorMatrix<F> {
        let width = RegionStartOracleCols::<F>::width();
        let mut values = vec![F::ZERO; width];
        let cols: &mut RegionStartOracleCols<F> = values.as_mut_slice().borrow_mut();
        cols.active = F::ONE;
        cols.region = F::ZERO;
        cols.tidx = F::from_usize(start);
        copy_ext(&mut cols.claim, claim);
        RowMajorMatrix::new(values, width)
    }

    fn point_oracle_trace(
        transcript: &TranscriptLog<F, [F; 16]>,
        start: usize,
        rounds: usize,
    ) -> RowMajorMatrix<F> {
        let width = RegionPointOracleCols::<F>::width();
        let height = rounds.max(1).next_power_of_two();
        let mut values = vec![F::ZERO; height * width];
        for round in 0..rounds {
            let cols: &mut RegionPointOracleCols<F> =
                values[round * width..(round + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.region = F::ZERO;
            cols.coordinate = F::from_usize(round);
            copy_ext(&mut cols.value, challenge_at(transcript, start, round));
        }
        RowMajorMatrix::new(values, width)
    }

    fn final_oracle_trace(claim: EF, end: usize) -> RowMajorMatrix<F> {
        let width = RegionFinalOracleCols::<F>::width();
        let mut values = vec![F::ZERO; width];
        let cols: &mut RegionFinalOracleCols<F> = values.as_mut_slice().borrow_mut();
        cols.active = F::ONE;
        cols.region = F::ZERO;
        cols.tidx = F::from_usize(end);
        copy_ext(&mut cols.claim, claim);
        RowMajorMatrix::new(values, width)
    }

    fn symbolic_interactions<A>(air: &A) -> Vec<SymbolicInteraction<F>>
    where
        A: Air<SymbolicRapBuilder<F>>
            + BaseAir<F>
            + BaseAirWithPublicValues<F>
            + PartitionedBaseAir<F>,
    {
        get_symbolic_builder(
            air,
            &TraceWidth {
                preprocessed: None,
                cached_mains: air.cached_main_widths(),
                common_main: air.common_main_width(),
            },
        )
        .constraints()
        .interactions
    }

    #[allow(clippy::too_many_arguments)]
    fn check_interaction_balance<A>(
        air: &A,
        trace: &RowMajorMatrix<F>,
        transcript: &TranscriptLog<F, [F; 16]>,
        start: usize,
        span: usize,
        initial_claim: EF,
        final_claim: EF,
        rounds: usize,
        buses: TestRegionBuses,
    ) where
        A: Air<SymbolicRapBuilder<F>>
            + BaseAir<F>
            + BaseAirWithPublicValues<F>
            + PartitionedBaseAir<F>,
    {
        let transcript_air = TranscriptOracleAir {
            bus: TranscriptBus::new(1),
        };
        let start_air = RegionStartOracleAir { buses };
        let point_air = RegionPointOracleAir { buses };
        let final_air = RegionFinalOracleAir { buses };
        let transcript_trace = transcript_oracle_trace(transcript, start, span);
        let start_trace = start_oracle_trace(initial_claim, start);
        let point_trace = point_oracle_trace(transcript, start, rounds);
        let final_trace = final_oracle_trace(final_claim, start + span);
        let interactions = vec![
            symbolic_interactions(air),
            symbolic_interactions(&transcript_air),
            symbolic_interactions(&start_air),
            symbolic_interactions(&point_air),
            symbolic_interactions(&final_air),
        ];
        let matrices = vec![
            vec![trace.as_view()],
            vec![transcript_trace.as_view()],
            vec![start_trace.as_view()],
            vec![point_trace.as_view()],
            vec![final_trace.as_view()],
        ];
        let names = (0..matrices.len())
            .map(|index| format!("complete-region-{index}"))
            .collect::<Vec<_>>();
        check_logup(
            &names,
            &interactions,
            &[None, None, None, None, None],
            &matrices,
            &[Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new()],
        );
    }

    fn local_test_buses() -> TestRegionBuses {
        TestRegionBuses::Local {
            start: FixedMultiAirCompleteLocalRegionStartBus::new(2),
            point: FixedMultiAirCompleteLocalRegionPointBus::new(3),
            final_claim: FixedMultiAirCompleteLocalRegionSumcheckFinalBus::new(4),
        }
    }

    fn interaction_test_buses() -> TestRegionBuses {
        TestRegionBuses::Interaction {
            start: FixedMultiAirCompleteInteractionRegionStartBus::new(5),
            point: FixedMultiAirCompleteInteractionRegionPointBus::new(6),
            final_claim: FixedMultiAirCompleteInteractionRegionSumcheckFinalBus::new(7),
        }
    }

    #[test]
    fn typed_local_and_interaction_buses_balance_and_cannot_cross_route() {
        let fixture = backend_fixture(2);
        let local_air = local_air(2, fixture.local_proof.opened_columns.len());
        let local_trace = generate_fixed_multi_air_complete_local_region_sumcheck_trace(
            &local_air,
            fixture.local_claim,
            &fixture.local_proof,
            &fixture.transcript,
            fixture.local_start,
            None,
        )
        .unwrap();
        let local_final = reference_final_claim(
            fixture.local_claim,
            &fixture.local_proof,
            &fixture.transcript,
            fixture.local_start,
        );
        check_interaction_balance(
            &local_air,
            &local_trace,
            &fixture.transcript,
            fixture.local_start,
            local_air.transcript_span(),
            fixture.local_claim,
            local_final,
            2,
            local_test_buses(),
        );
        assert!(catch_unwind(AssertUnwindSafe(|| {
            check_interaction_balance(
                &local_air,
                &local_trace,
                &fixture.transcript,
                fixture.local_start,
                local_air.transcript_span(),
                fixture.local_claim,
                local_final,
                2,
                interaction_test_buses(),
            )
        }))
        .is_err());

        let interaction_air = interaction_air(2, fixture.interaction_proof.opened_columns.len());
        let interaction_trace =
            generate_fixed_multi_air_complete_interaction_region_sumcheck_trace(
                &interaction_air,
                fixture.interaction_claim,
                &fixture.interaction_proof,
                &fixture.transcript,
                fixture.interaction_start,
                None,
            )
            .unwrap();
        let interaction_final = reference_final_claim(
            fixture.interaction_claim,
            &fixture.interaction_proof,
            &fixture.transcript,
            fixture.interaction_start,
        );
        check_interaction_balance(
            &interaction_air,
            &interaction_trace,
            &fixture.transcript,
            fixture.interaction_start,
            interaction_air.transcript_span(),
            fixture.interaction_claim,
            interaction_final,
            2,
            interaction_test_buses(),
        );
    }

    #[test]
    fn transcript_challenge_coordinate_mutations_fail_against_backend_oracle() {
        let fixture = backend_fixture(2);
        let air = local_air(2, fixture.local_proof.opened_columns.len());
        for coordinate in 0..D_EF {
            let mut mutated_transcript = fixture.transcript.clone();
            let sample_tidx = fixture.local_start
                + FIXED_MULTI_AIR_COMPLETE_REGION_PREFIX_BASE_VALUES
                + (3 + FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS) * D_EF;
            mutated_transcript.values_mut()[sample_tidx + coordinate] += F::ONE;
            let trace = generate_fixed_multi_air_complete_local_region_sumcheck_trace(
                &air,
                fixture.local_claim,
                &fixture.local_proof,
                &mutated_transcript,
                fixture.local_start,
                None,
            )
            .expect("locally consistent mutated challenge trace");
            check_air(&air, &trace);
            let final_claim = reference_final_claim(
                fixture.local_claim,
                &fixture.local_proof,
                &mutated_transcript,
                fixture.local_start,
            );
            assert!(catch_unwind(AssertUnwindSafe(|| {
                check_interaction_balance(
                    &air,
                    &trace,
                    &fixture.transcript,
                    fixture.local_start,
                    air.transcript_span(),
                    fixture.local_claim,
                    final_claim,
                    2,
                    local_test_buses(),
                )
            }))
            .is_err());
        }
    }

    #[test]
    fn transcript_round_messages_reject_every_evaluation_coordinate_mutation() {
        let fixture = backend_fixture(1);
        let air = local_air(1, fixture.local_proof.opened_columns.len());
        let round_start = fixture.local_start + FIXED_MULTI_AIR_COMPLETE_REGION_PREFIX_BASE_VALUES;
        for evaluation in 0..FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS {
            for coordinate in 0..D_EF {
                let mut transcript = fixture.transcript.clone();
                transcript.values_mut()[round_start + (3 + evaluation) * D_EF + coordinate] +=
                    F::ONE;
                assert_error_without_panic(|| {
                    generate_fixed_multi_air_complete_local_region_sumcheck_trace(
                        &air,
                        fixture.local_claim,
                        &fixture.local_proof,
                        &transcript,
                        fixture.local_start,
                        None,
                    )
                });
            }
        }
    }

    #[test]
    fn complete_sumcheck_air_stays_within_recursive_degree_profile() {
        let local = local_air(3, 2);
        let interaction = interaction_air(3, 4);
        for degree in [
            get_symbolic_builder(
                &local,
                &TraceWidth {
                    preprocessed: None,
                    cached_mains: local.cached_main_widths(),
                    common_main: local.common_main_width(),
                },
            )
            .constraints()
            .max_constraint_degree(),
            get_symbolic_builder(
                &interaction,
                &TraceWidth {
                    preprocessed: None,
                    cached_mains: interaction.cached_main_widths(),
                    common_main: interaction.common_main_width(),
                },
            )
            .constraints()
            .max_constraint_degree(),
        ] {
            assert!(degree <= 8, "complete sumcheck AIR degree {degree}");
        }
    }
}

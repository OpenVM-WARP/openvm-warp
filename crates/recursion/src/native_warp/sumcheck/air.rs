use core::borrow::Borrow;

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, PrimeCharacteristicRing};
use p3_matrix::Matrix;

use crate::{
    bus::TranscriptBus,
    native_warp::{
        bus::{
            NativeSumcheckChallengeBus, NativeSumcheckChallengeMessage, NativeSumcheckInitialBus,
            NativeSumcheckInitialMessage, NativeSumcheckRoundBus, NativeSumcheckRoundMessage,
        },
        ext::{ext_field_add, ext_field_multiply},
    },
};

/// One coefficient of one coefficient-form sumcheck round.
///
/// Coefficients are ordered from constant to highest degree. Running columns
/// simultaneously compute `h(1)` and `h(challenge)` without making AIR width
/// depend on the WARP family dimensions.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeCoefficientSumcheckCols<T> {
    pub active: T,
    pub proof_idx: T,
    /// 0 = twin-constraint sumcheck, 1 = multilinear-batching sumcheck.
    pub kind: T,
    pub round: T,
    pub coefficient_index: T,
    pub is_first_coefficient: T,
    pub is_last_coefficient: T,
    pub is_initial_round: T,
    pub is_final_round: T,
    pub first_inverse: T,
    pub last_inverse: T,
    pub round_inverse: T,
    pub final_round_inverse: T,
    pub tidx: T,
    pub coefficient: [T; D_EF],
    pub challenge: [T; D_EF],
    pub pre_claim: [T; D_EF],
    pub at_zero: [T; D_EF],
    pub at_one_acc: [T; D_EF],
    pub challenge_power: [T; D_EF],
    pub evaluation_acc: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(NativeCoefficientSumcheckCols<u8>)]
pub struct NativeCoefficientSumcheckAir {
    pub transcript_bus: TranscriptBus,
    pub round_bus: NativeSumcheckRoundBus,
    pub initial_bus: NativeSumcheckInitialBus,
    pub challenge_bus: NativeSumcheckChallengeBus,
    pub twin_degree: usize,
    pub batching_degree: usize,
    pub twin_rounds: usize,
    pub batching_rounds: usize,
}

impl BaseAirWithPublicValues<F> for NativeCoefficientSumcheckAir {}
impl PartitionedBaseAir<F> for NativeCoefficientSumcheckAir {}

impl<F> BaseAir<F> for NativeCoefficientSumcheckAir {
    fn width(&self) -> usize {
        NativeCoefficientSumcheckCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeCoefficientSumcheckAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("sumcheck row"),
            main.row_slice(1).expect("sumcheck next row"),
        );
        let local: &NativeCoefficientSumcheckCols<AB::Var> = (*local).borrow();
        let next: &NativeCoefficientSumcheckCols<AB::Var> = (*next).borrow();

        builder.assert_bool(local.active);
        builder.assert_bool(local.kind);
        builder.assert_bool(local.is_first_coefficient);
        builder.assert_bool(local.is_last_coefficient);
        builder.assert_bool(local.is_initial_round);
        builder.assert_bool(local.is_final_round);
        for flag in [
            local.kind,
            local.is_first_coefficient,
            local.is_last_coefficient,
            local.is_initial_round,
            local.is_final_round,
        ] {
            builder.when(AB::Expr::ONE - local.active).assert_zero(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);

        let degree = AB::Expr::from_usize(self.twin_degree)
            + local.kind
                * (AB::Expr::from_usize(self.batching_degree)
                    - AB::Expr::from_usize(self.twin_degree));
        let distance_to_last = degree.clone() - local.coefficient_index;

        builder
            .when(local.active * local.is_first_coefficient)
            .assert_zero(local.coefficient_index);
        builder
            .when(local.active * (AB::Expr::ONE - local.is_first_coefficient))
            .assert_one(local.coefficient_index * local.first_inverse);
        builder
            .when(local.active * local.is_last_coefficient)
            .assert_zero(distance_to_last.clone());
        builder
            .when(local.active * (AB::Expr::ONE - local.is_last_coefficient))
            .assert_one(distance_to_last * local.last_inverse);
        builder
            .when(local.active * local.is_initial_round)
            .assert_zero(local.round);
        builder
            .when(local.active * (AB::Expr::ONE - local.is_initial_round))
            .assert_one(local.round * local.round_inverse);
        let round_count = AB::Expr::from_usize(self.twin_rounds)
            + local.kind
                * (AB::Expr::from_usize(self.batching_rounds)
                    - AB::Expr::from_usize(self.twin_rounds));
        let distance_to_final_round = round_count - AB::Expr::ONE - local.round;
        builder
            .when(local.active * local.is_final_round)
            .assert_zero(distance_to_final_round.clone());
        builder
            .when(local.active * (AB::Expr::ONE - local.is_final_round))
            .assert_one(distance_to_final_round * local.final_round_inverse);

        let one_ext = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            }
        });
        let mut when_first = builder.when(local.active * local.is_first_coefficient);
        assert_array_eq(&mut when_first, local.at_zero, local.coefficient);
        assert_array_eq(&mut when_first, local.at_one_acc, local.coefficient);
        assert_array_eq(&mut when_first, local.challenge_power, one_ext);
        assert_array_eq(&mut when_first, local.evaluation_acc, local.coefficient);

        let same_round = next.active - next.is_first_coefficient;
        let mut transition = builder.when_transition();
        let mut when_same = transition.when(same_round.clone());
        when_same.assert_zero(local.is_last_coefficient);
        when_same.assert_eq(next.proof_idx, local.proof_idx);
        when_same.assert_eq(next.kind, local.kind);
        when_same.assert_eq(next.round, local.round);
        when_same.assert_eq(next.coefficient_index, local.coefficient_index + AB::F::ONE);
        when_same.assert_eq(next.tidx, local.tidx);
        assert_array_eq(&mut when_same, next.challenge, local.challenge);
        assert_array_eq(&mut when_same, next.pre_claim, local.pre_claim);
        assert_array_eq(&mut when_same, next.at_zero, local.at_zero);
        assert_array_eq(
            &mut when_same,
            next.at_one_acc,
            ext_field_add::<AB::Expr>(local.at_one_acc, next.coefficient),
        );
        let next_power = ext_field_multiply::<AB::Expr>(local.challenge_power, local.challenge);
        assert_array_eq(&mut when_same, next.challenge_power, next_power);
        assert_array_eq(
            &mut when_same,
            next.evaluation_acc,
            ext_field_add::<AB::Expr>(
                local.evaluation_acc,
                ext_field_multiply::<AB::Expr>(next.coefficient, next.challenge_power),
            ),
        );

        let starts_next = next.is_first_coefficient;
        let mut transition = builder.when_transition();
        let mut when_next = transition.when(starts_next);
        when_next.assert_one(local.is_last_coefficient);
        let starts_proof = (AB::Expr::ONE - next.kind) * next.is_initial_round;
        when_next.when(starts_proof.clone()).assert_one(local.kind);
        when_next
            .when(starts_proof.clone())
            .assert_one(local.is_final_round);
        when_next
            .when(starts_proof.clone())
            .assert_eq(next.proof_idx, local.proof_idx + AB::F::ONE);
        when_next
            .when(AB::Expr::ONE - starts_proof.clone())
            .assert_eq(next.proof_idx, local.proof_idx);
        let kind_delta = next.kind - local.kind;
        when_next
            .when(AB::Expr::ONE - starts_proof.clone())
            .assert_bool(kind_delta.clone());
        when_next
            .when((AB::Expr::ONE - starts_proof.clone()) * (AB::Expr::ONE - kind_delta.clone()))
            .assert_eq(next.round, local.round + AB::F::ONE);
        assert_array_eq(
            &mut when_next.when(
                (AB::Expr::ONE - starts_proof.clone()) * (AB::Expr::ONE - kind_delta.clone()),
            ),
            next.pre_claim,
            local.evaluation_acc,
        );
        when_next
            .when((AB::Expr::ONE - starts_proof.clone()) * kind_delta.clone())
            .assert_zero(next.round);
        when_next
            .when((AB::Expr::ONE - starts_proof) * kind_delta)
            .assert_one(local.is_final_round);

        builder
            .when_transition()
            .when(local.active - next.active)
            .assert_one(local.is_last_coefficient);

        let mut when_last = builder.when(local.active * local.is_last_coefficient);
        assert_array_eq(
            &mut when_last,
            local.pre_claim,
            ext_field_add::<AB::Expr>(local.at_zero, local.at_one_acc),
        );

        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.tidx + local.coefficient_index * AB::Expr::from_usize(D_EF),
            local.coefficient,
            local.active,
        );
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.tidx + (degree.clone() + AB::Expr::ONE) * AB::Expr::from_usize(D_EF),
            local.challenge,
            local.active * local.is_last_coefficient,
        );
        self.round_bus.send(
            builder,
            NativeSumcheckRoundMessage {
                proof_idx: local.proof_idx.into(),
                kind: local.kind.into(),
                round: local.round.into(),
                pre_claim: local.pre_claim.map(Into::into),
                post_claim: local.evaluation_acc.map(Into::into),
                challenge: local.challenge.map(Into::into),
            },
            local.active * local.is_last_coefficient * local.is_final_round,
        );
        self.challenge_bus.add_key_with_lookups(
            builder,
            NativeSumcheckChallengeMessage {
                proof_idx: local.proof_idx.into(),
                kind: local.kind.into(),
                round: local.round.into(),
                value: local.challenge.map(Into::into),
            },
            local.active * local.is_last_coefficient * (AB::Expr::ONE + local.kind),
        );
        self.initial_bus.send(
            builder,
            NativeSumcheckInitialMessage {
                proof_idx: local.proof_idx.into(),
                kind: local.kind.into(),
                claim: local.pre_claim.map(Into::into),
            },
            local.active * local.is_first_coefficient * local.is_initial_round,
        );
    }
}

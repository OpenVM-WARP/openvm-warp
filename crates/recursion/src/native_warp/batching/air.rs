use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    native_warp::{
        batching::{BATCHING_OUTPUT_ALPHA, BATCHING_OUTPUT_MU, OPENING_SECTION_TARGET},
        bus::{
            NativeBatchingOutputBus, NativeBatchingOutputMessage, NativeCertifiedBatchingClaimBus,
            NativeCertifiedBatchingClaimMessage, NativeEqResultBus, NativeEqResultMessage,
            NativeOpeningClaimBus, NativeOpeningClaimMessage, NativeSumcheckChallengeBus,
            NativeSumcheckChallengeMessage, NativeSumcheckInitialBus, NativeSumcheckInitialMessage,
            NativeSumcheckRoundBus, NativeSumcheckRoundMessage,
        },
        ext::{ext_field_add, ext_field_multiply},
    },
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeBatchingSigmaCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub claim: T,
    pub is_first: T,
    pub is_last: T,
    pub xi_weight: [T; D_EF],
    pub target: [T; D_EF],
    pub accumulator_before: [T; D_EF],
    pub accumulator_after: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(NativeBatchingSigmaCols<u8>)]
pub struct NativeBatchingSigmaAir {
    pub eq_bus: NativeEqResultBus,
    pub opening_bus: NativeOpeningClaimBus,
    pub sumcheck_initial_bus: Option<NativeSumcheckInitialBus>,
    /// Optional second consumer used by the transition certificate to bind the
    /// opening-authenticated claim into its replay seal.
    pub certified_claim_bus: Option<NativeCertifiedBatchingClaimBus>,
    pub claim_count: usize,
    pub xi_eq_group_offset: usize,
}

impl BaseAirWithPublicValues<F> for NativeBatchingSigmaAir {}
impl PartitionedBaseAir<F> for NativeBatchingSigmaAir {}
impl<F> BaseAir<F> for NativeBatchingSigmaAir {
    fn width(&self) -> usize {
        NativeBatchingSigmaCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeBatchingSigmaAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("native batching sigma row"),
            main.row_slice(1).expect("native batching sigma next row"),
        );
        let local: &NativeBatchingSigmaCols<AB::Var> = (*local).borrow();
        let next: &NativeBatchingSigmaCols<AB::Var> = (*next).borrow();
        builder.assert_bool(local.active);
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);
        builder.when_first_row().assert_one(local.active);
        builder
            .when(local.active * local.is_first)
            .assert_zero(local.claim);
        builder
            .when(local.active * local.is_last)
            .assert_eq(local.claim, AB::Expr::from_usize(self.claim_count - 1));
        self.eq_bus.lookup_key(
            builder,
            NativeEqResultMessage {
                proof_idx: local.proof_idx.into(),
                group: AB::Expr::from_usize(self.xi_eq_group_offset) + local.claim,
                value: local.xi_weight.map(Into::into),
            },
            local.active,
        );
        self.opening_bus.receive(
            builder,
            NativeOpeningClaimMessage {
                proof_idx: local.proof_idx.into(),
                claim: local.claim.into(),
                section: AB::Expr::from_usize(OPENING_SECTION_TARGET),
                coordinate: AB::Expr::ZERO,
                value: local.target.map(Into::into),
            },
            local.active,
        );
        let zero = [AB::Expr::ZERO; D_EF];
        assert_array_eq(
            &mut builder.when(local.active * local.is_first),
            local.accumulator_before,
            zero,
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.accumulator_after,
            ext_field_add::<AB::Expr>(
                local.accumulator_before,
                ext_field_multiply::<AB::Expr>(local.xi_weight, local.target),
            ),
        );
        let same = next.active * (AB::Expr::ONE - next.is_first);
        let mut transition = builder.when_transition();
        let mut when_same = transition.when(same);
        when_same.assert_zero(local.is_last);
        when_same.assert_eq(next.proof_idx, local.proof_idx);
        when_same.assert_eq(next.claim, local.claim + AB::F::ONE);
        assert_array_eq(
            &mut when_same,
            next.accumulator_before,
            local.accumulator_after,
        );
        if let Some(sumcheck_initial_bus) = self.sumcheck_initial_bus {
            sumcheck_initial_bus.receive(
                builder,
                NativeSumcheckInitialMessage {
                    proof_idx: local.proof_idx.into(),
                    kind: AB::Expr::ONE,
                    claim: local.accumulator_after.map(Into::into),
                },
                local.active * local.is_last,
            );
        }
        if let Some(certified_claim_bus) = self.certified_claim_bus {
            certified_claim_bus.send(
                builder,
                NativeCertifiedBatchingClaimMessage {
                    proof_idx: local.proof_idx.into(),
                    claim: local.accumulator_after.map(Into::into),
                },
                local.active * local.is_last,
            );
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeBatchingFinalCols<T> {
    pub active: T,
    pub claim: T,
    pub is_first: T,
    pub is_last: T,
    pub xi_weight: [T; D_EF],
    pub point_eq_alpha: [T; D_EF],
    pub accumulator_before: [T; D_EF],
    pub accumulator_after: [T; D_EF],
    pub final_pre_claim: [T; D_EF],
    pub final_claim: [T; D_EF],
    pub final_challenge: [T; D_EF],
    pub mu: [T; D_EF],
    pub proof_idx: T,
    pub mu_tidx: T,
}

#[derive(ColumnsAir)]
#[columns_via(NativeBatchingFinalCols<u8>)]
pub struct NativeBatchingFinalAir {
    pub eq_bus: NativeEqResultBus,
    pub sumcheck_round_bus: NativeSumcheckRoundBus,
    pub output_bus: NativeBatchingOutputBus,
    pub claim_count: usize,
    pub xi_eq_group_offset: usize,
    pub point_eq_group_offset: usize,
    pub last_round: usize,
    pub transcript_bus: TranscriptBus,
}

impl BaseAirWithPublicValues<F> for NativeBatchingFinalAir {}
impl PartitionedBaseAir<F> for NativeBatchingFinalAir {}
impl<F> BaseAir<F> for NativeBatchingFinalAir {
    fn width(&self) -> usize {
        NativeBatchingFinalCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeBatchingFinalAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("native batching final row"),
            main.row_slice(1).expect("native batching final next row"),
        );
        let local: &NativeBatchingFinalCols<AB::Var> = (*local).borrow();
        let next: &NativeBatchingFinalCols<AB::Var> = (*next).borrow();
        builder.assert_bool(local.active);
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);
        builder.when_first_row().assert_one(local.active);
        builder
            .when(local.active * local.is_first)
            .assert_zero(local.claim);
        builder
            .when(local.active * local.is_last)
            .assert_eq(local.claim, AB::Expr::from_usize(self.claim_count - 1));
        self.eq_bus.lookup_key(
            builder,
            NativeEqResultMessage {
                proof_idx: local.proof_idx.into(),
                group: AB::Expr::from_usize(self.xi_eq_group_offset) + local.claim,
                value: local.xi_weight.map(Into::into),
            },
            local.active,
        );
        self.eq_bus.lookup_key(
            builder,
            NativeEqResultMessage {
                proof_idx: local.proof_idx.into(),
                group: AB::Expr::from_usize(self.point_eq_group_offset) + local.claim,
                value: local.point_eq_alpha.map(Into::into),
            },
            local.active,
        );
        let zero = [AB::Expr::ZERO; D_EF];
        assert_array_eq(
            &mut builder.when(local.active * local.is_first),
            local.accumulator_before,
            zero,
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.accumulator_after,
            ext_field_add::<AB::Expr>(
                local.accumulator_before,
                ext_field_multiply::<AB::Expr>(local.xi_weight, local.point_eq_alpha),
            ),
        );
        let same = next.active * (AB::Expr::ONE - next.is_first);
        let mut transition = builder.when_transition();
        let mut when_same = transition.when(same);
        when_same.assert_zero(local.is_last);
        when_same.assert_eq(next.proof_idx, local.proof_idx);
        when_same.assert_eq(next.claim, local.claim + AB::F::ONE);
        assert_array_eq(
            &mut when_same,
            next.accumulator_before,
            local.accumulator_after,
        );
        self.sumcheck_round_bus.receive(
            builder,
            NativeSumcheckRoundMessage {
                proof_idx: local.proof_idx.into(),
                kind: AB::Expr::ONE,
                round: AB::Expr::from_usize(self.last_round),
                pre_claim: local.final_pre_claim.map(Into::into),
                post_claim: local.final_claim.map(Into::into),
                challenge: local.final_challenge.map(Into::into),
            },
            local.active * local.is_last,
        );
        assert_array_eq(
            &mut builder.when(local.active * local.is_last),
            local.final_claim,
            ext_field_multiply::<AB::Expr>(local.mu, local.accumulator_after),
        );
        self.output_bus.send(
            builder,
            NativeBatchingOutputMessage {
                proof_idx: local.proof_idx.into(),
                section: AB::Expr::from_usize(BATCHING_OUTPUT_MU),
                coordinate: AB::Expr::ZERO,
                value: local.mu.map(Into::into),
            },
            local.active * local.is_last,
        );
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.mu_tidx,
            local.mu,
            local.active * local.is_last,
        );
    }
}

pub fn generate_native_batching_sigma_trace(
    proof_idx: usize,
    weights: &[EF],
    targets: &[EF],
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if weights.is_empty() || weights.len() != targets.len() {
        return None;
    }
    let height = required_height.unwrap_or_else(|| weights.len().next_power_of_two());
    if height < weights.len() {
        return None;
    }
    let width = NativeBatchingSigmaCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    let mut accumulator = EF::ZERO;
    for claim in 0..weights.len() {
        let before = accumulator;
        accumulator += weights[claim] * targets[claim];
        let cols: &mut NativeBatchingSigmaCols<F> =
            trace[claim * width..(claim + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.proof_idx = F::from_usize(proof_idx);
        cols.claim = F::from_usize(claim);
        cols.is_first = F::from_bool(claim == 0);
        cols.is_last = F::from_bool(claim + 1 == weights.len());
        for (target, value) in [
            (&mut cols.xi_weight, weights[claim]),
            (&mut cols.target, targets[claim]),
            (&mut cols.accumulator_before, before),
            (&mut cols.accumulator_after, accumulator),
        ] {
            target.copy_from_slice(value.as_basis_coefficients_slice());
        }
    }
    Some(RowMajorMatrix::new(trace, width))
}

#[allow(clippy::too_many_arguments)]
pub fn generate_native_batching_final_trace(
    weights: &[EF],
    point_eq_alpha: &[EF],
    final_pre_claim: EF,
    final_claim: EF,
    final_challenge: EF,
    mu: EF,
    proof_idx: usize,
    mu_tidx: usize,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if weights.is_empty() || weights.len() != point_eq_alpha.len() {
        return None;
    }
    let height = required_height.unwrap_or_else(|| weights.len().next_power_of_two());
    if height < weights.len() {
        return None;
    }
    let width = NativeBatchingFinalCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    let mut accumulator = EF::ZERO;
    for claim in 0..weights.len() {
        let before = accumulator;
        accumulator += weights[claim] * point_eq_alpha[claim];
        let cols: &mut NativeBatchingFinalCols<F> =
            trace[claim * width..(claim + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.claim = F::from_usize(claim);
        cols.is_first = F::from_bool(claim == 0);
        cols.is_last = F::from_bool(claim + 1 == weights.len());
        for (target, value) in [
            (&mut cols.xi_weight, weights[claim]),
            (&mut cols.point_eq_alpha, point_eq_alpha[claim]),
            (&mut cols.accumulator_before, before),
            (&mut cols.accumulator_after, accumulator),
            (&mut cols.final_pre_claim, final_pre_claim),
            (&mut cols.final_claim, final_claim),
            (&mut cols.final_challenge, final_challenge),
            (&mut cols.mu, mu),
        ] {
            target.copy_from_slice(value.as_basis_coefficients_slice());
        }
        cols.proof_idx = F::from_usize(proof_idx);
        cols.mu_tidx = F::from_usize(mu_tidx);
    }
    Some(RowMajorMatrix::new(trace, width))
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeBatchingAlphaCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(NativeBatchingAlphaCols<u8>)]
pub struct NativeBatchingAlphaAir {
    pub challenge_bus: NativeSumcheckChallengeBus,
    pub output_bus: NativeBatchingOutputBus,
}

impl BaseAirWithPublicValues<F> for NativeBatchingAlphaAir {}
impl PartitionedBaseAir<F> for NativeBatchingAlphaAir {}
impl<F> BaseAir<F> for NativeBatchingAlphaAir {
    fn width(&self) -> usize {
        NativeBatchingAlphaCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeBatchingAlphaAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("native batching alpha row");
        let local: &NativeBatchingAlphaCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        self.challenge_bus.lookup_key(
            builder,
            NativeSumcheckChallengeMessage {
                proof_idx: local.proof_idx.into(),
                kind: AB::Expr::ONE,
                round: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active,
        );
        self.output_bus.send(
            builder,
            NativeBatchingOutputMessage {
                proof_idx: local.proof_idx.into(),
                section: AB::Expr::from_usize(BATCHING_OUTPUT_ALPHA),
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active,
        );
    }
}

pub fn generate_native_batching_alpha_trace(
    proof_idx: usize,
    alpha: &[EF],
) -> Option<RowMajorMatrix<F>> {
    if alpha.is_empty() {
        return None;
    }
    let height = alpha.len().next_power_of_two();
    let width = NativeBatchingAlphaCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    for (coordinate, value) in alpha.iter().enumerate() {
        let cols: &mut NativeBatchingAlphaCols<F> =
            trace[coordinate * width..(coordinate + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.proof_idx = F::from_usize(proof_idx);
        cols.coordinate = F::from_usize(coordinate);
        cols.value
            .copy_from_slice(value.as_basis_coefficients_slice());
    }
    Some(RowMajorMatrix::new(trace, width))
}

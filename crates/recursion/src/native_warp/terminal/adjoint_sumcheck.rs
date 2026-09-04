use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{encoder::Encoder, utils::assert_array_eq, ColumnsAir, SubAir};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    transcript::TranscriptLog,
    warp_accum::{RsAdjointEvalVerification, TerminalWhirVerification},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField32,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    native_warp::terminal::{
        NativeTerminalRsAdjointClaimBus, NativeTerminalRsAdjointClaimMessage,
        NativeTerminalRsAdjointRoundBus, NativeTerminalRsAdjointRoundMessage,
        NATIVE_TERMINAL_SUMCHECK_SELECTOR_MAX_FLAG_DEGREE,
    },
    utils::{ext_field_add, ext_field_multiply},
};

pub const NATIVE_TERMINAL_MAX_ADJOINT_DEGREE: usize = 32;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeTerminalRsAdjointSumcheckCols<T, const ENC_WIDTH: usize> {
    pub active: T,
    pub round: T,
    pub evaluation_index: T,
    pub is_first_evaluation: T,
    pub is_last_evaluation: T,
    pub is_first_round: T,
    pub is_last_round: T,
    pub is_final: T,
    pub tidx: T,
    pub claimed_value: [T; D_EF],
    pub pre_claim: [T; D_EF],
    pub at_zero: [T; D_EF],
    pub at_one: [T; D_EF],
    pub evaluation: [T; D_EF],
    pub challenge: [T; D_EF],
    pub basis_prefix: [[T; D_EF]; NATIVE_TERMINAL_MAX_ADJOINT_DEGREE + 2],
    pub denominator_inverse: T,
    pub term: [T; D_EF],
    pub sum_before: [T; D_EF],
    pub sum_after: [T; D_EF],
    pub post_claim: [T; D_EF],
    pub evaluation_encoding: [T; ENC_WIDTH],
}

/// Evaluation-form sumcheck used by the succinct RS-adjoint evaluator.
///
/// Each row handles one integer evaluation. The Lagrange numerator is built
/// through degree-two prefix recurrences, so the AIR degree remains bounded
/// independently of the terminal message dimension.
pub struct NativeTerminalRsAdjointSumcheckAir {
    pub transcript_bus: TranscriptBus,
    pub round_bus: NativeTerminalRsAdjointRoundBus,
    pub claim_bus: NativeTerminalRsAdjointClaimBus,
    pub round_count: usize,
    pub degree: usize,
    pub evaluation_encoder: Encoder,
}

impl NativeTerminalRsAdjointSumcheckAir {
    #[must_use]
    pub fn new(
        transcript_bus: TranscriptBus,
        round_bus: NativeTerminalRsAdjointRoundBus,
        claim_bus: NativeTerminalRsAdjointClaimBus,
        round_count: usize,
        degree: usize,
    ) -> Self {
        assert!(round_count > 0 && degree > 0 && degree <= NATIVE_TERMINAL_MAX_ADJOINT_DEGREE);
        Self {
            transcript_bus,
            round_bus,
            claim_bus,
            round_count,
            degree,
            evaluation_encoder: Encoder::new(
                (degree + 1).max(2),
                NATIVE_TERMINAL_SUMCHECK_SELECTOR_MAX_FLAG_DEGREE,
                false,
            ),
        }
    }

    fn eval_impl<AB, const ENC_WIDTH: usize>(&self, builder: &mut AB)
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
        <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
    {
        let main = builder.main();
        let local_row = main
            .row_slice(0)
            .expect("native terminal RS-adjoint sumcheck row");
        let next_row = main
            .row_slice(1)
            .expect("native terminal next RS-adjoint sumcheck row");
        let local: &NativeTerminalRsAdjointSumcheckCols<AB::Var, ENC_WIDTH> = (*local_row).borrow();
        let next: &NativeTerminalRsAdjointSumcheckCols<AB::Var, ENC_WIDTH> = (*next_row).borrow();

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
        let indices = (0..=self.degree).map(|index| (index, index));
        let first = (0..=self.degree).map(|index| (index, usize::from(index == 0)));
        let last = (0..=self.degree).map(|index| (index, usize::from(index == self.degree)));
        builder.when(local.active).assert_eq(
            local.evaluation_index,
            self.evaluation_encoder
                .flag_with_val::<AB>(&local.evaluation_encoding, &indices.collect::<Vec<_>>()),
        );
        builder.when(local.active).assert_eq(
            local.is_first_evaluation,
            self.evaluation_encoder
                .flag_with_val::<AB>(&local.evaluation_encoding, &first.collect::<Vec<_>>()),
        );
        builder.when(local.active).assert_eq(
            local.is_last_evaluation,
            self.evaluation_encoder
                .flag_with_val::<AB>(&local.evaluation_encoding, &last.collect::<Vec<_>>()),
        );
        builder
            .when(local.active * local.is_first_round)
            .assert_zero(local.round);
        builder
            .when(local.active * local.is_last_round)
            .assert_eq(local.round, AB::Expr::from_usize(self.round_count - 1));
        builder.when(local.active).assert_eq(
            local.is_final,
            local.is_last_evaluation * local.is_last_round,
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
        let same_round = next.active * (AB::Expr::ONE - AB::Expr::from(next.is_first_evaluation));
        let next_round = next.active * next.is_first_evaluation;
        let mut transition = builder.when_transition();
        let mut same = transition.when(same_round);
        same.assert_eq(next.evaluation_index, local.evaluation_index + AB::F::ONE);
        same.assert_eq(next.round, local.round);
        same.assert_eq(next.tidx, local.tidx);
        same.assert_eq(next.is_first_round, local.is_first_round);
        same.assert_eq(next.is_last_round, local.is_last_round);
        assert_array_eq(&mut same, next.claimed_value, local.claimed_value);
        assert_array_eq(&mut same, next.pre_claim, local.pre_claim);
        assert_array_eq(&mut same, next.at_zero, local.at_zero);
        assert_array_eq(&mut same, next.at_one, local.at_one);
        assert_array_eq(&mut same, next.challenge, local.challenge);
        assert_array_eq(&mut same, next.sum_before, local.sum_after);
        assert_array_eq(&mut same, next.post_claim, local.post_claim);

        let mut transition = builder.when_transition();
        let mut advance = transition.when(next_round);
        advance.assert_one(local.is_last_evaluation);
        advance.assert_eq(next.round, local.round + AB::F::ONE);
        advance.assert_eq(
            next.tidx,
            local.tidx + AB::Expr::from_usize((self.degree + 2) * D_EF),
        );
        advance.assert_zero(next.is_first_round);
        advance.assert_zero(local.is_last_round);
        assert_array_eq(&mut advance, next.claimed_value, local.claimed_value);
        assert_array_eq(&mut advance, next.pre_claim, local.post_claim);
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
            &(0..=self.degree)
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
        for other in 0..=self.degree {
            let skip = self.evaluation_encoder.flag_with_val::<AB>(
                &local.evaluation_encoding,
                &(0..=self.degree)
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
            &(0..=self.degree)
                .map(|index| {
                    (
                        index,
                        lagrange_denominator(index, self.degree)
                            .inverse()
                            .as_canonical_u32() as usize,
                    )
                })
                .collect::<Vec<_>>(),
        );
        builder
            .when(local.active)
            .assert_eq(local.denominator_inverse, denominator_inverse);
        let scaled_basis = local.basis_prefix[self.degree + 1]
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
        let zero_ext = core::array::from_fn(|_| AB::Expr::ZERO);
        assert_array_eq(
            &mut builder.when(local.is_first_evaluation),
            local.sum_before,
            zero_ext,
        );
        assert_array_eq(
            &mut builder.when(local.is_last_evaluation),
            local.sum_after,
            local.post_claim.map(Into::into),
        );

        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.tidx)
                + AB::Expr::from(local.evaluation_index) * AB::Expr::from_usize(D_EF),
            local.evaluation,
            local.active,
        );
        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.tidx) - AB::Expr::from_usize(D_EF),
            local.claimed_value,
            local.active * local.is_first_round * local.is_first_evaluation,
        );
        self.transcript_bus.sample_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.tidx) + AB::Expr::from_usize((self.degree + 1) * D_EF),
            local.challenge,
            local.active * local.is_last_evaluation,
        );
        self.round_bus.add_key_with_lookups(
            builder,
            NativeTerminalRsAdjointRoundMessage {
                round: local.round.into(),
                challenge: local.challenge.map(Into::into),
            },
            AB::Expr::from(local.active)
                * AB::Expr::from(local.is_last_evaluation)
                * AB::Expr::from_usize(self.degree),
        );
        self.claim_bus.send(
            builder,
            NativeTerminalRsAdjointClaimMessage {
                claimed_value: local.claimed_value.map(Into::into),
                final_claim: local.post_claim.map(Into::into),
            },
            local.active * local.is_final,
        );
    }
}

impl BaseAirWithPublicValues<F> for NativeTerminalRsAdjointSumcheckAir {}
impl PartitionedBaseAir<F> for NativeTerminalRsAdjointSumcheckAir {}
impl ColumnsAir for NativeTerminalRsAdjointSumcheckAir {}
impl BaseAir<F> for NativeTerminalRsAdjointSumcheckAir {
    fn width(&self) -> usize {
        match self.evaluation_encoder.width() {
            1 => NativeTerminalRsAdjointSumcheckCols::<F, 1>::width(),
            2 => NativeTerminalRsAdjointSumcheckCols::<F, 2>::width(),
            3 => NativeTerminalRsAdjointSumcheckCols::<F, 3>::width(),
            4 => NativeTerminalRsAdjointSumcheckCols::<F, 4>::width(),
            5 => NativeTerminalRsAdjointSumcheckCols::<F, 5>::width(),
            6 => NativeTerminalRsAdjointSumcheckCols::<F, 6>::width(),
            7 => NativeTerminalRsAdjointSumcheckCols::<F, 7>::width(),
            width => panic!("unsupported terminal adjoint encoder width: {width}"),
        }
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTerminalRsAdjointSumcheckAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        match self.evaluation_encoder.width() {
            1 => self.eval_impl::<AB, 1>(builder),
            2 => self.eval_impl::<AB, 2>(builder),
            3 => self.eval_impl::<AB, 3>(builder),
            4 => self.eval_impl::<AB, 4>(builder),
            5 => self.eval_impl::<AB, 5>(builder),
            6 => self.eval_impl::<AB, 6>(builder),
            7 => self.eval_impl::<AB, 7>(builder),
            width => panic!("unsupported terminal adjoint encoder width: {width}"),
        }
    }
}

pub fn generate_native_terminal_rs_adjoint_sumcheck_trace(
    air: &NativeTerminalRsAdjointSumcheckAir,
    verification: &TerminalWhirVerification<F, EF, Digest>,
    transcript: &TranscriptLog<F, [F; 16]>,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    match air.evaluation_encoder.width() {
        1 => generate_sumcheck_trace_impl::<1>(air, verification, transcript, required_height),
        2 => generate_sumcheck_trace_impl::<2>(air, verification, transcript, required_height),
        3 => generate_sumcheck_trace_impl::<3>(air, verification, transcript, required_height),
        4 => generate_sumcheck_trace_impl::<4>(air, verification, transcript, required_height),
        5 => generate_sumcheck_trace_impl::<5>(air, verification, transcript, required_height),
        6 => generate_sumcheck_trace_impl::<6>(air, verification, transcript, required_height),
        7 => generate_sumcheck_trace_impl::<7>(air, verification, transcript, required_height),
        _ => None,
    }
}

fn generate_sumcheck_trace_impl<const ENC_WIDTH: usize>(
    air: &NativeTerminalRsAdjointSumcheckAir,
    verification: &TerminalWhirVerification<F, EF, Digest>,
    transcript: &TranscriptLog<F, [F; 16]>,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    let adjoint: &RsAdjointEvalVerification<EF> = &verification.accumulator_adjoint;
    if adjoint.rounds.len() != air.round_count
        || adjoint.degree as usize != air.degree
        || adjoint
            .rounds
            .iter()
            .any(|round| round.evaluations.len() != air.degree + 1)
    {
        return None;
    }
    let rows_per_round = air.degree + 1;
    let valid_rows = air.round_count.checked_mul(rows_per_round)?;
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeTerminalRsAdjointSumcheckCols::<F, ENC_WIDTH>::width();
    let mut trace = F::zero_vec(height * width);
    let mut row_index = 0usize;
    for (round_index, record) in adjoint.rounds.iter().enumerate() {
        if record.round as usize != round_index
            || record.transcript_end.operations
                != record
                    .transcript_start
                    .operations
                    .checked_add((air.degree + 2).checked_mul(D_EF)?)?
        {
            return None;
        }
        let at_zero = record.evaluations[0];
        let at_one = record.evaluations[1];
        if at_zero + at_one != record.pre_claim {
            return None;
        }
        let mut sum = EF::ZERO;
        for (evaluation_index, &evaluation) in record.evaluations.iter().enumerate() {
            let mut basis = EF::ONE;
            let mut prefixes = vec![EF::ONE];
            for other in 0..=air.degree {
                if other != evaluation_index {
                    basis *= record.challenge - EF::from(F::from_usize(other));
                }
                prefixes.push(basis);
            }
            let denominator_inverse = lagrange_denominator(evaluation_index, air.degree).inverse();
            let term = evaluation * basis * EF::from(denominator_inverse);
            let sum_before = sum;
            sum += term;
            let row = &mut trace[row_index * width..(row_index + 1) * width];
            let cols: &mut NativeTerminalRsAdjointSumcheckCols<F, ENC_WIDTH> = row.borrow_mut();
            cols.active = F::ONE;
            cols.round = F::from_usize(round_index);
            cols.evaluation_index = F::from_usize(evaluation_index);
            cols.is_first_evaluation = F::from_bool(evaluation_index == 0);
            cols.is_last_evaluation = F::from_bool(evaluation_index == air.degree);
            cols.is_first_round = F::from_bool(round_index == 0);
            cols.is_last_round = F::from_bool(round_index + 1 == air.round_count);
            cols.is_final =
                F::from_bool(round_index + 1 == air.round_count && evaluation_index == air.degree);
            cols.tidx = F::from_usize(record.transcript_start.operations);
            copy_ext(&mut cols.claimed_value, adjoint.claimed_value);
            copy_ext(&mut cols.pre_claim, record.pre_claim);
            copy_ext(&mut cols.at_zero, at_zero);
            copy_ext(&mut cols.at_one, at_one);
            copy_ext(&mut cols.evaluation, evaluation);
            copy_ext(&mut cols.challenge, record.challenge);
            for (target, value) in cols.basis_prefix.iter_mut().zip(prefixes) {
                copy_ext(target, value);
            }
            cols.denominator_inverse = denominator_inverse;
            copy_ext(&mut cols.term, term);
            copy_ext(&mut cols.sum_before, sum_before);
            copy_ext(&mut cols.sum_after, sum);
            copy_ext(&mut cols.post_claim, record.post_claim);
            for (target, value) in cols
                .evaluation_encoding
                .iter_mut()
                .zip(air.evaluation_encoder.get_flag_pt(evaluation_index))
            {
                *target = F::from_u32(value);
            }
            let transcript_eval = EF::from_basis_coefficients_slice(transcript.values().get(
                record.transcript_start.operations + evaluation_index * D_EF
                    ..record.transcript_start.operations + (evaluation_index + 1) * D_EF,
            )?)?;
            if transcript_eval != evaluation {
                return None;
            }
            row_index += 1;
        }
        if sum != record.post_claim {
            return None;
        }
    }
    if adjoint.final_claim != adjoint.expected_final
        || adjoint.rounds.last()?.post_claim != adjoint.final_claim
    {
        return None;
    }
    Some(RowMajorMatrix::new(trace, width))
}

fn lagrange_denominator(index: usize, degree: usize) -> F {
    (0..=degree)
        .filter(|&other| other != index)
        .map(|other| F::from_usize(index) - F::from_usize(other))
        .product()
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjoint_sumcheck_air_supports_declared_degree_cap() {
        let air = NativeTerminalRsAdjointSumcheckAir::new(
            TranscriptBus::new(0),
            NativeTerminalRsAdjointRoundBus::new(1),
            NativeTerminalRsAdjointClaimBus::new(2),
            1,
            NATIVE_TERMINAL_MAX_ADJOINT_DEGREE,
        );
        let _ = BaseAir::<F>::width(&air);
        assert_eq!(air.evaluation_encoder.width(), 7);
    }
}

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, transcript::TranscriptLog,
    warp_accum::TerminalWhirVerification, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    native_warp::terminal::{
        NativeTerminalWhirAlphaBus, NativeTerminalWhirAlphaMessage, NativeTerminalWhirRoundBus,
        NativeTerminalWhirRoundMessage,
    },
    primitives::bus::{ExpBitsLenBus, ExpBitsLenMessage},
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{interpolate_quadratic, pow_tidx_count},
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeTerminalWhirSumcheckCols<T> {
    pub active: T,
    pub round: T,
    pub fold: T,
    pub is_first: T,
    pub is_last: T,
    pub tidx: T,
    pub pre_claim: [T; D_EF],
    pub round_pre_claim: [T; D_EF],
    pub round_post_claim: [T; D_EF],
    pub at_zero: [T; D_EF],
    pub at_one: [T; D_EF],
    pub at_two: [T; D_EF],
    pub challenge: [T; D_EF],
    pub post_claim: [T; D_EF],
    pub folding_pow_witness: T,
    pub folding_pow_sample: T,
    /// Number of folding/query rows that consume this challenge. As in the
    /// regular WHIR sumcheck AIR, LogUp balance constrains this value.
    pub alpha_lookup_count: T,
}

#[derive(ColumnsAir)]
#[columns_via(NativeTerminalWhirSumcheckCols<u8>)]
pub struct NativeTerminalWhirSumcheckAir {
    pub transcript_bus: TranscriptBus,
    pub round_bus: NativeTerminalWhirRoundBus,
    pub alpha_bus: NativeTerminalWhirAlphaBus,
    pub exp_bits_len_bus: ExpBitsLenBus,
    pub k: usize,
    pub round_count: usize,
    pub folding_pow_bits: usize,
    pub generator: F,
}

impl BaseAirWithPublicValues<F> for NativeTerminalWhirSumcheckAir {}
impl PartitionedBaseAir<F> for NativeTerminalWhirSumcheckAir {}
impl BaseAir<F> for NativeTerminalWhirSumcheckAir {
    fn width(&self) -> usize {
        NativeTerminalWhirSumcheckCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTerminalWhirSumcheckAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("terminal WHIR sumcheck row"),
            main.row_slice(1).expect("terminal WHIR next sumcheck row"),
        );
        let local: &NativeTerminalWhirSumcheckCols<AB::Var> = (*local).borrow();
        let next: &NativeTerminalWhirSumcheckCols<AB::Var> = (*next).borrow();

        debug_assert!(self.k > 0);
        debug_assert!(self.round_count > 0);

        NestedForLoopSubAir::<1>.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.active.into(),
                    counter: [local.round.into()],
                    is_first: [local.is_first.into()],
                },
                NestedForLoopIoCols {
                    is_enabled: next.active.into(),
                    counter: [next.round.into()],
                    is_first: [next.is_first.into()],
                },
            ),
        );

        builder.assert_bool(local.is_last);
        let is_same_round = next.active - next.is_first;
        let is_round_end = local.active - next.active + next.is_first;
        builder
            .when_transition()
            .assert_eq(local.is_last, is_round_end.clone());
        builder
            .when_last_row()
            .assert_eq(local.is_last, local.active);

        builder.when(local.is_first).assert_zero(local.fold);
        builder
            .when(is_same_round.clone())
            .assert_eq(next.fold, local.fold + AB::F::ONE);
        builder
            .when(is_round_end.clone())
            .assert_eq(local.fold, AB::Expr::from_usize(self.k - 1));
        builder
            .when_transition()
            .when(local.active - next.active)
            .assert_eq(local.round, AB::Expr::from_usize(self.round_count - 1));
        builder
            .when_last_row()
            .when(local.active)
            .assert_eq(local.round, AB::Expr::from_usize(self.round_count - 1));

        let mut same = builder.when(is_same_round.clone());
        same.assert_eq(
            next.tidx,
            local.tidx + AB::Expr::from_usize(3 * D_EF + pow_tidx_count(self.folding_pow_bits)),
        );
        assert_array_eq(&mut same, next.pre_claim, local.post_claim);
        assert_array_eq(&mut same, next.round_pre_claim, local.round_pre_claim);
        assert_array_eq(&mut same, next.round_post_claim, local.round_post_claim);

        let mut transition = builder.when_transition();
        let mut next_round = transition.when(next.is_first);
        next_round.assert_one(local.is_last);
        next_round.assert_eq(next.round, local.round + AB::F::ONE);

        assert_array_eq(
            &mut builder.when(local.is_first),
            local.round_pre_claim,
            local.pre_claim,
        );
        assert_array_eq(
            &mut builder.when(local.is_last),
            local.round_post_claim,
            local.post_claim,
        );

        assert_array_eq(
            &mut builder.when(local.active),
            local.pre_claim,
            crate::utils::ext_field_add::<AB::Expr>(local.at_zero, local.at_one),
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.post_claim,
            interpolate_quadratic(local.pre_claim, local.at_one, local.at_two, local.challenge),
        );

        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            local.tidx,
            local.at_one,
            local.active,
        );
        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            local.tidx + AB::Expr::from_usize(D_EF),
            local.at_two,
            local.active,
        );
        if self.folding_pow_bits > 0 {
            self.transcript_bus.observe(
                builder,
                AB::Expr::ZERO,
                local.tidx + AB::Expr::from_usize(2 * D_EF),
                local.folding_pow_witness,
                local.active,
            );
            self.transcript_bus.sample(
                builder,
                AB::Expr::ZERO,
                local.tidx + AB::Expr::from_usize(2 * D_EF + 1),
                local.folding_pow_sample,
                local.active,
            );
            self.exp_bits_len_bus.lookup_key(
                builder,
                ExpBitsLenMessage {
                    base: self.generator.into(),
                    bit_src: local.folding_pow_sample.into(),
                    num_bits: AB::Expr::from_usize(self.folding_pow_bits),
                    result: AB::Expr::ONE,
                },
                local.active,
            );
        }
        self.transcript_bus.sample_ext(
            builder,
            AB::Expr::ZERO,
            local.tidx + AB::Expr::from_usize(2 * D_EF + pow_tidx_count(self.folding_pow_bits)),
            local.challenge,
            local.active,
        );
        self.alpha_bus.add_key_with_lookups(
            builder,
            NativeTerminalWhirAlphaMessage {
                round: local.round.into(),
                fold: local.fold.into(),
                challenge: local.challenge.map(Into::into),
            },
            local.active * local.alpha_lookup_count,
        );
        self.round_bus.send(
            builder,
            NativeTerminalWhirRoundMessage {
                round: local.round.into(),
                tidx: local.tidx
                    - local.fold
                        * AB::Expr::from_usize(3 * D_EF + pow_tidx_count(self.folding_pow_bits)),
                pre_claim: local.round_pre_claim.map(Into::into),
                post_sumcheck_claim: local.round_post_claim.map(Into::into),
            },
            local.active * local.is_last,
        );
    }
}

pub fn generate_native_terminal_whir_sumcheck_trace<Digest>(
    verification: &TerminalWhirVerification<F, EF, Digest>,
    transcript: &TranscriptLog<F, [F; 16]>,
    folding_pow_bits: usize,
    extra_alpha_lookup_count: usize,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    let valid_rows = verification
        .rounds
        .iter()
        .map(|round| round.sumcheck_rounds.len())
        .sum::<usize>();
    if valid_rows == 0 {
        return None;
    }
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeTerminalWhirSumcheckCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    let pow_offset = pow_tidx_count(folding_pow_bits);
    let k = verification.rounds.first()?.sumcheck_rounds.len();
    if k == 0
        || verification
            .rounds
            .iter()
            .any(|round| round.sumcheck_rounds.len() != k)
    {
        return None;
    }
    let mut alpha_lookup_counts = vec![0usize; valid_rows];
    for (round_index, round) in verification.rounds.iter().enumerate() {
        let queries = round.query_indices.len();
        for fold in 0..k {
            alpha_lookup_counts[round_index * k + fold] =
                queries * (1usize << (k - 1 - fold)) + extra_alpha_lookup_count;
        }
    }
    let mut row_index = 0usize;
    for (round_index, round) in verification.rounds.iter().enumerate() {
        let round_pre_claim = round
            .sumcheck_rounds
            .first()
            .map(|record| record.pre_claim)?;
        let round_post_claim = round
            .sumcheck_rounds
            .last()
            .map(|record| record.post_claim)?;
        for (fold_index, record) in round.sumcheck_rounds.iter().enumerate() {
            if record.whir_round as usize != round_index || record.fold_round as usize != fold_index
            {
                return None;
            }
            let tidx = record.transcript_span.start.operations;
            if record.transcript_span.end.operations != tidx + 3 * D_EF + pow_offset {
                return None;
            }
            let row = &mut values[row_index * width..(row_index + 1) * width];
            let cols: &mut NativeTerminalWhirSumcheckCols<F> = row.borrow_mut();
            cols.active = F::ONE;
            cols.round = F::from_usize(round_index);
            cols.fold = F::from_usize(fold_index);
            cols.is_first = F::from_bool(fold_index == 0);
            cols.is_last = F::from_bool(fold_index + 1 == k);
            cols.tidx = F::from_usize(tidx);
            copy_ext(&mut cols.pre_claim, record.pre_claim);
            copy_ext(&mut cols.round_pre_claim, round_pre_claim);
            copy_ext(&mut cols.round_post_claim, round_post_claim);
            copy_ext(&mut cols.at_zero, record.at_zero);
            copy_ext(&mut cols.at_one, record.at_one);
            copy_ext(&mut cols.at_two, record.at_two);
            copy_ext(&mut cols.challenge, record.challenge);
            copy_ext(&mut cols.post_claim, record.post_claim);
            if folding_pow_bits > 0 {
                cols.folding_pow_witness = *transcript.values().get(tidx + 2 * D_EF)?;
                cols.folding_pow_sample = *transcript.values().get(tidx + 2 * D_EF + 1)?;
            }
            cols.alpha_lookup_count = F::from_usize(alpha_lookup_counts[row_index]);
            row_index += 1;
        }
    }
    debug_assert_eq!(row_index, valid_rows);
    Some(RowMajorMatrix::new(values, width))
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

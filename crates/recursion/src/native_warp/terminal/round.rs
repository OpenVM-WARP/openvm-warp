use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{encoder::Encoder, utils::assert_array_eq, ColumnsAir, SubAir};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, transcript::TranscriptLog,
    warp_accum::TerminalWhirVerification, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing, TwoAdicField,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    native_warp::{
        bus::{NativeMerkleRootBus, NativeMerkleRootMessage},
        terminal::{
            NativeTerminalWhirFinalClaimBus, NativeTerminalWhirFinalClaimMessage,
            NativeTerminalWhirRoundBus, NativeTerminalWhirRoundMessage,
            NativeTerminalWhirStatementBus, NativeTerminalWhirStatementMessage,
            NativeTerminalWhirVerifyQueriesBus, NativeTerminalWhirVerifyQueriesMessage,
            NativeTerminalWhirWeightTermBus, NativeTerminalWhirWeightTermMessage,
        },
    },
    primitives::bus::{ExpBitsLenBus, ExpBitsLenMessage},
    utils::{ext_field_multiply, pow_tidx_count},
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeTerminalWhirRoundCols<T, const ENC_WIDTH: usize> {
    pub active: T,
    pub round: T,
    pub is_first: T,
    pub is_last: T,
    pub tidx: T,
    pub query_tidx: T,
    pub num_queries: T,
    pub omega: T,
    pub current_root: [T; DIGEST_SIZE],
    pub next_root: [T; DIGEST_SIZE],
    pub ood_point: [T; D_EF],
    pub ood_value: [T; D_EF],
    pub query_pow_witness: T,
    pub query_pow_sample: T,
    pub gamma: [T; D_EF],
    pub batching_challenge: [T; D_EF],
    pub claim: [T; D_EF],
    pub post_sumcheck_claim: [T; D_EF],
    pub next_claim: [T; D_EF],
    pub round_encoding: [T; ENC_WIDTH],
}

pub struct NativeTerminalWhirRoundAir {
    pub transcript_bus: TranscriptBus,
    pub statement_bus: NativeTerminalWhirStatementBus,
    pub sumcheck_bus: NativeTerminalWhirRoundBus,
    pub verify_queries_bus: NativeTerminalWhirVerifyQueriesBus,
    pub final_claim_bus: NativeTerminalWhirFinalClaimBus,
    pub weight_term_bus: NativeTerminalWhirWeightTermBus,
    pub merkle_root_bus: NativeMerkleRootBus,
    pub exp_bits_len_bus: ExpBitsLenBus,
    pub k: usize,
    pub round_count: usize,
    pub initial_log_domain_size: usize,
    pub final_poly_len: usize,
    pub query_pow_bits: usize,
    pub folding_pow_bits: usize,
    pub generator: F,
    pub outer_tree_id_offset: usize,
    pub num_queries_per_round: Vec<usize>,
    pub round_encoder: Encoder,
}

impl NativeTerminalWhirRoundAir {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        transcript_bus: TranscriptBus,
        statement_bus: NativeTerminalWhirStatementBus,
        sumcheck_bus: NativeTerminalWhirRoundBus,
        verify_queries_bus: NativeTerminalWhirVerifyQueriesBus,
        final_claim_bus: NativeTerminalWhirFinalClaimBus,
        weight_term_bus: NativeTerminalWhirWeightTermBus,
        merkle_root_bus: NativeMerkleRootBus,
        exp_bits_len_bus: ExpBitsLenBus,
        k: usize,
        initial_log_domain_size: usize,
        final_poly_len: usize,
        query_pow_bits: usize,
        folding_pow_bits: usize,
        generator: F,
        outer_tree_id_offset: usize,
        num_queries_per_round: Vec<usize>,
    ) -> Result<Self, &'static str> {
        let round_count = num_queries_per_round.len();
        if k == 0
            || round_count == 0
            || initial_log_domain_size < k + round_count - 1
            || initial_log_domain_size - k > F::TWO_ADICITY
            || final_poly_len == 0
            || !final_poly_len.is_power_of_two()
            || num_queries_per_round.contains(&0)
        {
            return Err("native terminal WHIR round profile");
        }
        Ok(Self {
            transcript_bus,
            statement_bus,
            sumcheck_bus,
            verify_queries_bus,
            final_claim_bus,
            weight_term_bus,
            merkle_root_bus,
            exp_bits_len_bus,
            k,
            round_count,
            initial_log_domain_size,
            final_poly_len,
            query_pow_bits,
            folding_pow_bits,
            generator,
            outer_tree_id_offset,
            num_queries_per_round,
            round_encoder: Encoder::new(round_count.max(2), 2, false),
        })
    }

    fn eval_impl<AB, const ENC_WIDTH: usize>(&self, builder: &mut AB)
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
        <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield:
            BinomiallyExtendable<{ D_EF }> + TwoAdicField,
    {
        let main = builder.main();
        let local = main.row_slice(0).expect("native terminal WHIR round row");
        let next = main
            .row_slice(1)
            .expect("native terminal WHIR next round row");
        let local: &NativeTerminalWhirRoundCols<AB::Var, ENC_WIDTH> = (*local).borrow();
        let next: &NativeTerminalWhirRoundCols<AB::Var, ENC_WIDTH> = (*next).borrow();

        self.round_encoder.eval(builder, &local.round_encoding);
        let round_values = (0..self.round_count).map(|round| (round, round));
        let decoded_round = self
            .round_encoder
            .flag_with_val::<AB>(&local.round_encoding, &round_values.collect::<Vec<_>>());
        builder
            .when(local.active)
            .assert_eq(local.round, decoded_round);
        let query_values = self
            .num_queries_per_round
            .iter()
            .copied()
            .enumerate()
            .collect::<Vec<_>>();
        let expected_queries = self
            .round_encoder
            .flag_with_val::<AB>(&local.round_encoding, &query_values);
        builder
            .when(local.active)
            .assert_eq(local.num_queries, expected_queries);

        for flag in [local.active, local.is_first, local.is_last] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when_first_row().assert_zero(local.round);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.is_last);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);
        builder
            .when(local.active * local.is_last)
            .assert_eq(local.round, AB::Expr::from_usize(self.round_count - 1));

        let initial_omega = AB::Expr::from_prime_subfield(
            <<AB::Expr as PrimeCharacteristicRing>::PrimeSubfield as TwoAdicField>::two_adic_generator(
                self.initial_log_domain_size - self.k,
            ),
        );
        builder
            .when(local.is_first)
            .assert_eq(local.omega, initial_omega);
        let mut transition = builder.when_transition();
        let mut same = transition.when(next.active);
        same.assert_zero(next.is_first);
        same.assert_eq(next.round, local.round + AB::F::ONE);
        same.assert_eq(next.omega, local.omega * local.omega);
        same.assert_eq(
            next.tidx,
            local.query_tidx + local.num_queries + AB::Expr::from_usize(D_EF),
        );
        assert_array_eq(&mut same, next.claim, local.next_claim);
        assert_array_eq(&mut same, next.current_root, local.next_root);

        let sumcheck_stride = 3 * D_EF + pow_tidx_count(self.folding_pow_bits);
        let post_sumcheck_tidx = local.tidx + AB::Expr::from_usize(self.k * sumcheck_stride);
        let final_poly_tidx = post_sumcheck_tidx.clone();
        let header_tidx_count = (AB::Expr::ONE - local.is_last)
            * AB::Expr::from_usize(DIGEST_SIZE + 2 * D_EF)
            + local.is_last * AB::Expr::from_usize(self.final_poly_len * D_EF);
        let pow_tidx = post_sumcheck_tidx.clone() + header_tidx_count;
        builder.when(local.active).assert_eq(
            local.query_tidx,
            pow_tidx.clone() + AB::Expr::from_usize(pow_tidx_count(self.query_pow_bits)),
        );

        self.sumcheck_bus.receive(
            builder,
            NativeTerminalWhirRoundMessage {
                round: local.round.into(),
                tidx: local.tidx.into(),
                pre_claim: local.claim.map(Into::into),
                post_sumcheck_claim: local.post_sumcheck_claim.map(Into::into),
            },
            local.active,
        );
        self.statement_bus.lookup_key(
            builder,
            NativeTerminalWhirStatementMessage {
                tidx: local.tidx.into(),
                root: local.current_root.map(Into::into),
                batching_challenge: local.batching_challenge.map(Into::into),
                initial_claim: local.claim.map(Into::into),
            },
            local.is_first,
        );
        self.transcript_bus.observe_commit(
            builder,
            AB::Expr::ZERO,
            post_sumcheck_tidx.clone(),
            local.next_root,
            local.active * (AB::Expr::ONE - local.is_last),
        );
        self.transcript_bus.sample_ext(
            builder,
            AB::Expr::ZERO,
            post_sumcheck_tidx.clone() + AB::Expr::from_usize(DIGEST_SIZE),
            local.ood_point,
            local.active * (AB::Expr::ONE - local.is_last),
        );
        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            post_sumcheck_tidx + AB::Expr::from_usize(DIGEST_SIZE + D_EF),
            local.ood_value,
            local.active * (AB::Expr::ONE - local.is_last),
        );
        if self.query_pow_bits > 0 {
            self.transcript_bus.observe(
                builder,
                AB::Expr::ZERO,
                pow_tidx.clone(),
                local.query_pow_witness,
                local.active,
            );
            self.transcript_bus.sample(
                builder,
                AB::Expr::ZERO,
                pow_tidx + AB::Expr::ONE,
                local.query_pow_sample,
                local.active,
            );
            self.exp_bits_len_bus.lookup_key(
                builder,
                ExpBitsLenMessage {
                    base: self.generator.into(),
                    bit_src: local.query_pow_sample.into(),
                    num_bits: AB::Expr::from_usize(self.query_pow_bits),
                    result: AB::Expr::ONE,
                },
                local.active,
            );
        }
        self.transcript_bus.sample_ext(
            builder,
            AB::Expr::ZERO,
            local.query_tidx + local.num_queries,
            local.gamma,
            local.active,
        );

        let ood_term = ext_field_multiply::<AB::Expr>(local.gamma, local.ood_value);
        let pre_query_claim = core::array::from_fn(|index| {
            local.post_sumcheck_claim[index].into()
                + (AB::Expr::ONE - local.is_last) * ood_term[index].clone()
        });
        self.verify_queries_bus.send(
            builder,
            NativeTerminalWhirVerifyQueriesMessage {
                round: local.round.into(),
                tidx: local.query_tidx.into(),
                num_queries: local.num_queries.into(),
                omega: local.omega.into(),
                gamma: local.gamma.map(Into::into),
                pre_claim: pre_query_claim,
                post_claim: local.next_claim.map(Into::into),
            },
            local.active,
        );
        let term_ordinals = self
            .num_queries_per_round
            .iter()
            .scan(0usize, |query_prefix, &queries| {
                let ordinal = *query_prefix;
                *query_prefix += queries;
                Some(ordinal)
            })
            .enumerate()
            .map(|(round, query_prefix)| (round, query_prefix + round))
            .collect::<Vec<_>>();
        let term_ordinal = self
            .round_encoder
            .flag_with_val::<AB>(&local.round_encoding, &term_ordinals);
        let point_len = self.k * self.round_count + self.final_poly_len.ilog2() as usize;
        self.weight_term_bus.send(
            builder,
            NativeTerminalWhirWeightTermMessage {
                ordinal: term_ordinal,
                after_folds: (local.round + AB::F::ONE) * AB::Expr::from_usize(self.k),
                length: AB::Expr::from_usize(point_len)
                    - (local.round + AB::F::ONE) * AB::Expr::from_usize(self.k),
                generator: local.ood_point.map(Into::into),
                scale: local.gamma.map(Into::into),
            },
            local.active * (AB::Expr::ONE - local.is_last),
        );
        self.merkle_root_bus.receive(
            builder,
            NativeMerkleRootMessage {
                proof_idx: AB::Expr::ZERO,
                tree_id: local.round + AB::Expr::from_usize(self.outer_tree_id_offset),
                depth: AB::Expr::from_usize(self.initial_log_domain_size - self.k) - local.round,
                digest: local.current_root.map(Into::into),
            },
            local.active,
        );
        self.final_claim_bus.send(
            builder,
            NativeTerminalWhirFinalClaimMessage {
                final_poly_tidx,
                tidx: local.query_tidx + local.num_queries + AB::Expr::from_usize(D_EF),
                claim: local.next_claim.map(Into::into),
            },
            local.active * local.is_last,
        );
    }
}

impl BaseAirWithPublicValues<F> for NativeTerminalWhirRoundAir {}
impl PartitionedBaseAir<F> for NativeTerminalWhirRoundAir {}
impl ColumnsAir for NativeTerminalWhirRoundAir {}

impl BaseAir<F> for NativeTerminalWhirRoundAir {
    fn width(&self) -> usize {
        match self.round_encoder.width() {
            1 => NativeTerminalWhirRoundCols::<F, 1>::width(),
            2 => NativeTerminalWhirRoundCols::<F, 2>::width(),
            3 => NativeTerminalWhirRoundCols::<F, 3>::width(),
            width => panic!("unsupported native terminal round encoder width: {width}"),
        }
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTerminalWhirRoundAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield:
        BinomiallyExtendable<{ D_EF }> + TwoAdicField,
{
    fn eval(&self, builder: &mut AB) {
        match self.round_encoder.width() {
            1 => self.eval_impl::<AB, 1>(builder),
            2 => self.eval_impl::<AB, 2>(builder),
            3 => self.eval_impl::<AB, 3>(builder),
            width => panic!("unsupported native terminal round encoder width: {width}"),
        }
    }
}

pub fn generate_native_terminal_whir_round_trace(
    air: &NativeTerminalWhirRoundAir,
    verification: &TerminalWhirVerification<F, EF, Digest>,
    transcript: &TranscriptLog<F, [F; 16]>,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    match air.round_encoder.width() {
        1 => generate_round_trace_impl::<1>(air, verification, transcript, required_height),
        2 => generate_round_trace_impl::<2>(air, verification, transcript, required_height),
        3 => generate_round_trace_impl::<3>(air, verification, transcript, required_height),
        _ => None,
    }
}

fn generate_round_trace_impl<const ENC_WIDTH: usize>(
    air: &NativeTerminalWhirRoundAir,
    verification: &TerminalWhirVerification<F, EF, Digest>,
    transcript: &TranscriptLog<F, [F; 16]>,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if verification.rounds.len() != air.round_count
        || verification
            .rounds
            .iter()
            .zip(&air.num_queries_per_round)
            .any(|(round, &queries)| round.query_indices.len() != queries)
    {
        return None;
    }
    let height = required_height.unwrap_or_else(|| air.round_count.next_power_of_two());
    if height < air.round_count {
        return None;
    }
    let width = NativeTerminalWhirRoundCols::<F, ENC_WIDTH>::width();
    let mut values = F::zero_vec(height * width);
    let sumcheck_stride = 3 * D_EF + pow_tidx_count(air.folding_pow_bits);
    let query_pow_offset = pow_tidx_count(air.query_pow_bits);
    for (round_index, round) in verification.rounds.iter().enumerate() {
        if round.round as usize != round_index
            || round.sumcheck_rounds.len() != air.k
            || round.log_rs_domain_size as usize != air.initial_log_domain_size - round_index
        {
            return None;
        }
        let tidx = round
            .sumcheck_rounds
            .first()?
            .transcript_span
            .start
            .operations;
        let sumcheck_end = round.sumcheck_rounds.last()?.transcript_span.end.operations;
        if sumcheck_end != tidx + air.k * sumcheck_stride {
            return None;
        }
        let is_last = round_index + 1 == air.round_count;
        let header_count = if is_last {
            air.final_poly_len.checked_mul(D_EF)?
        } else {
            DIGEST_SIZE + 2 * D_EF
        };
        let pow_tidx = sumcheck_end.checked_add(header_count)?;
        let query_tidx = pow_tidx.checked_add(query_pow_offset)?;
        let query_end = query_tidx.checked_add(round.query_indices.len())?;
        let gamma_end = query_end.checked_add(D_EF)?;
        if round.transcript_span.start.operations != tidx
            || round.transcript_span.end.operations != gamma_end
            || transcript.values().get(query_end..gamma_end)?
                != round.gamma.as_basis_coefficients_slice()
        {
            return None;
        }
        let current_root = round.commitment;
        let next_root = if is_last {
            [F::ZERO; DIGEST_SIZE]
        } else {
            verification.rounds.get(round_index + 1)?.commitment
        };
        let row = &mut values[round_index * width..(round_index + 1) * width];
        let cols: &mut NativeTerminalWhirRoundCols<F, ENC_WIDTH> = row.borrow_mut();
        cols.active = F::ONE;
        cols.round = F::from_usize(round_index);
        cols.is_first = F::from_bool(round_index == 0);
        cols.is_last = F::from_bool(is_last);
        cols.tidx = F::from_usize(tidx);
        cols.query_tidx = F::from_usize(query_tidx);
        cols.num_queries = F::from_usize(round.query_indices.len());
        cols.omega = F::two_adic_generator(air.initial_log_domain_size - air.k - round_index);
        cols.current_root = current_root;
        cols.next_root = next_root;
        if let Some(point) = round.ood_point {
            copy_ext(&mut cols.ood_point, point);
        }
        if let Some(value) = round.ood_value {
            copy_ext(&mut cols.ood_value, value);
        }
        if air.query_pow_bits > 0 {
            cols.query_pow_witness = *transcript.values().get(pow_tidx)?;
            cols.query_pow_sample = *transcript.values().get(pow_tidx + 1)?;
        }
        copy_ext(&mut cols.gamma, round.gamma);
        copy_ext(
            &mut cols.batching_challenge,
            verification.batching_challenge,
        );
        copy_ext(&mut cols.claim, round.pre_round_claim);
        copy_ext(
            &mut cols.post_sumcheck_claim,
            round.sumcheck_rounds.last()?.post_claim,
        );
        copy_ext(&mut cols.next_claim, round.post_round_claim);
        let encoding = air.round_encoder.get_flag_pt(round_index);
        for (target, value) in cols.round_encoding.iter_mut().zip(encoding) {
            *target = F::from_u32(value);
        }
    }
    Some(RowMajorMatrix::new(values, width))
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

#[cfg(test)]
mod tests {
    use openvm_stark_backend::{
        air_builders::symbolic::get_symbolic_builder, keygen::types::TraceWidth,
    };
    use p3_field::Field;

    use super::*;

    fn round_air(
        initial_log_domain_size: usize,
        k: usize,
    ) -> Result<NativeTerminalWhirRoundAir, &'static str> {
        NativeTerminalWhirRoundAir::new(
            TranscriptBus::new(0),
            NativeTerminalWhirStatementBus::new(1),
            NativeTerminalWhirRoundBus::new(2),
            NativeTerminalWhirVerifyQueriesBus::new(3),
            NativeTerminalWhirFinalClaimBus::new(4),
            NativeTerminalWhirWeightTermBus::new(5),
            NativeMerkleRootBus::new(6),
            ExpBitsLenBus::new(7),
            k,
            initial_log_domain_size,
            2,
            0,
            0,
            F::GENERATOR,
            0,
            vec![1],
        )
    }

    #[test]
    fn vector_alphabet_reduces_the_round_evaluation_domain() {
        // Production reduced-SWIRL profiles use large RS codewords and k = 4.
        // WHIR's round polynomial is evaluated on the 2^(29-4) row domain,
        // which is supported by BabyBear even though a 2^29 root is not.
        let air = round_air(29, 4).unwrap();
        let symbolic = get_symbolic_builder(
            &air,
            &TraceWidth {
                preprocessed: None,
                cached_mains: Vec::new(),
                common_main: air.width(),
            },
        );
        let _ = symbolic.constraints();
    }

    #[test]
    fn unsupported_round_evaluation_domain_is_rejected() {
        assert!(round_air(F::TWO_ADICITY + 5, 4).is_err());
    }
}

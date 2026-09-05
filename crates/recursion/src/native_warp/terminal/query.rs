use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper, SubAir,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    transcript::TranscriptLog,
    warp_accum::{
        terminal_whir::{terminal_whir_query_root, TerminalWhirLayout},
        TerminalWhirVerification,
    },
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing,
    PrimeField32, TwoAdicField,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    native_warp::terminal::{
        NativeTerminalWhirQueryBus, NativeTerminalWhirQueryMessage,
        NativeTerminalWhirVerifyQueriesBus, NativeTerminalWhirVerifyQueriesMessage,
        NativeTerminalWhirWeightTermBus, NativeTerminalWhirWeightTermMessage,
    },
    primitives::bus::{ExpBitsLenBus, ExpBitsLenMessage, RightShiftBus, RightShiftMessage},
    subairs::nested_for_loop::{NestedForLoopIoCols, NestedForLoopSubAir},
    utils::{ext_field_add, ext_field_multiply},
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeTerminalWhirQueryCols<T> {
    pub active: T,
    pub round: T,
    pub query: T,
    pub global_query: T,
    pub is_first: T,
    pub is_first_in_round: T,
    pub tidx: T,
    pub num_queries: T,
    pub query_domain_size: T,
    pub omega: T,
    pub sample_domain_root: T,
    pub sample: T,
    pub quotient: T,
    pub merkle_index: T,
    pub inner_tree_id: T,
    pub outer_tree_id: T,
    pub zi_root: T,
    pub zi: T,
    pub raw_omega: T,
    pub subgroup_root: T,
    pub coordinate_exponent: T,
    pub yi: [T; D_EF],
    pub gamma: [T; D_EF],
    pub gamma_pow: [T; D_EF],
    pub is_final_round: T,
    pub round_delta_inverse: T,
    pub pre_claim: [T; D_EF],
    pub post_claim: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(NativeTerminalWhirQueryCols<u8>)]
pub struct NativeTerminalWhirQueryAir {
    pub transcript_bus: TranscriptBus,
    pub verify_queries_bus: NativeTerminalWhirVerifyQueriesBus,
    pub query_bus: NativeTerminalWhirQueryBus,
    pub weight_term_bus: NativeTerminalWhirWeightTermBus,
    pub exp_bits_len_bus: ExpBitsLenBus,
    pub right_shift_bus: RightShiftBus,
    pub k: usize,
    pub initial_log_domain_size: usize,
    pub round_count: usize,
    pub final_poly_len: usize,
    pub inner_tree_id_offset: usize,
    pub outer_tree_id_offset: usize,
    /// Scalar rows are evaluations on a `2^k` coset. Their authenticated
    /// coset root and post-fold query point are distinct values.
    pub evaluation_layout: bool,
}

impl BaseAirWithPublicValues<F> for NativeTerminalWhirQueryAir {}
impl PartitionedBaseAir<F> for NativeTerminalWhirQueryAir {}
impl BaseAir<F> for NativeTerminalWhirQueryAir {
    fn width(&self) -> usize {
        NativeTerminalWhirQueryCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTerminalWhirQueryAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        debug_assert!(
            self.k > 0
                && self.initial_log_domain_size > self.k
                && self.initial_log_domain_size <= 27 + self.k
        );
        let main = builder.main();
        let local = main.row_slice(0).expect("native terminal WHIR query row");
        let next = main
            .row_slice(1)
            .expect("native terminal WHIR next query row");
        let local: &NativeTerminalWhirQueryCols<AB::Var> = (*local).borrow();
        let next: &NativeTerminalWhirQueryCols<AB::Var> = (*next).borrow();

        NestedForLoopSubAir::<2>.eval(
            builder,
            (
                NestedForLoopIoCols {
                    is_enabled: local.active.into(),
                    counter: [local.round.into(), local.query.into()],
                    is_first: [local.is_first_in_round.into(), local.active.into()],
                },
                NestedForLoopIoCols {
                    is_enabled: next.active.into(),
                    counter: [next.round.into(), next.query.into()],
                    is_first: [next.is_first_in_round.into(), next.active.into()],
                },
            ),
        );
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_final_round);
        builder
            .when_first_row()
            .assert_eq(local.is_first, local.active);
        builder
            .when_transition()
            .when(next.active)
            .assert_eq(next.global_query, local.global_query + AB::F::ONE);
        builder
            .when_first_row()
            .when(local.active)
            .assert_zero(local.global_query);
        let final_round = AB::Expr::from_usize(self.round_count - 1);
        let round_delta = AB::Expr::from(local.round) - final_round;
        builder
            .when(local.active * local.is_final_round)
            .assert_zero(round_delta.clone());
        builder
            .when(local.active * (AB::Expr::ONE - local.is_final_round))
            .assert_one(round_delta * local.round_delta_inverse);

        let same_round = next.active - next.is_first_in_round;
        let round_end = local.active - same_round.clone();
        builder
            .when(round_end.clone())
            .assert_eq(local.query + AB::F::ONE, local.num_queries);
        let mut transition = builder.when_transition();
        let mut same = transition.when(same_round.clone());
        same.assert_eq(next.tidx, local.tidx + AB::F::ONE);
        same.assert_eq(next.num_queries, local.num_queries);
        same.assert_eq(next.query_domain_size, local.query_domain_size);
        same.assert_eq(next.omega, local.omega);
        same.assert_eq(next.raw_omega, local.raw_omega);
        assert_array_eq(&mut same, next.gamma, local.gamma);
        assert_array_eq(
            &mut same,
            next.gamma_pow,
            ext_field_multiply::<AB::Expr>(local.gamma, local.gamma_pow),
        );
        assert_array_eq(
            &mut same,
            next.pre_claim,
            ext_field_add::<AB::Expr>(
                local.pre_claim,
                ext_field_multiply::<AB::Expr>(local.gamma_pow, local.yi),
            ),
        );
        assert_array_eq(&mut same, next.post_claim, local.post_claim);

        let next_round = next.active * next.is_first_in_round;
        let mut transition = builder.when_transition();
        let mut advance = transition.when(next_round);
        advance.assert_eq(local.query_domain_size, next.query_domain_size * AB::F::TWO);
        advance.assert_eq(next.omega, local.omega * local.omega);
        if self.evaluation_layout {
            advance.assert_eq(next.raw_omega, local.raw_omega * local.raw_omega);
        }

        assert_array_eq(
            &mut builder.when(local.is_first_in_round),
            local.gamma_pow,
            ext_field_multiply::<AB::Expr>(local.gamma, local.gamma),
        );
        assert_array_eq(
            &mut builder.when(round_end),
            ext_field_add::<AB::Expr>(
                local.pre_claim,
                ext_field_multiply::<AB::Expr>(local.gamma_pow, local.yi),
            ),
            local.post_claim,
        );

        self.verify_queries_bus.receive(
            builder,
            NativeTerminalWhirVerifyQueriesMessage {
                round: local.round.into(),
                tidx: local.tidx.into(),
                num_queries: local.num_queries.into(),
                omega: local.omega.into(),
                gamma: local.gamma.map(Into::into),
                pre_claim: local.pre_claim.map(Into::into),
                post_claim: local.post_claim.map(Into::into),
            },
            local.is_first_in_round,
        );
        self.transcript_bus.sample(
            builder,
            AB::Expr::ZERO,
            local.tidx,
            local.sample,
            local.active,
        );
        // This request also owns the canonical right-shift witness used below
        // to prove `sample mod query_domain_size = merkle_index`.
        self.exp_bits_len_bus.lookup_key(
            builder,
            ExpBitsLenMessage {
                base: local.omega.into(),
                bit_src: local.sample.into(),
                num_bits: AB::Expr::from_usize(self.initial_log_domain_size - self.k) - local.round,
                result: local.sample_domain_root.into(),
            },
            local.active,
        );
        if self.evaluation_layout {
            let initial_raw_omega = F::two_adic_generator(self.initial_log_domain_size);
            builder
                .when_first_row()
                .when(local.active)
                .assert_eq(local.raw_omega, initial_raw_omega);
            builder
                .when(local.active)
                .assert_eq(local.coordinate_exponent, local.merkle_index);
            self.exp_bits_len_bus.lookup_key(
                builder,
                ExpBitsLenMessage {
                    base: local.raw_omega.into(),
                    bit_src: local.coordinate_exponent.into(),
                    num_bits: AB::Expr::from_usize(self.initial_log_domain_size),
                    result: local.subgroup_root.into(),
                },
                local.active,
            );
            builder
                .when(local.active)
                .assert_eq(local.zi_root, local.subgroup_root);
            self.exp_bits_len_bus.lookup_key(
                builder,
                ExpBitsLenMessage {
                    base: local.zi_root.into(),
                    bit_src: AB::Expr::from_usize(1usize << self.k),
                    num_bits: AB::Expr::from_usize(self.k + 1),
                    result: local.zi.into(),
                },
                local.active,
            );
        } else {
            builder
                .when(local.active)
                .assert_eq(local.zi_root, local.sample_domain_root);
            builder
                .when(local.active)
                .assert_eq(local.zi, local.zi_root);
        }
        self.right_shift_bus.lookup_key(
            builder,
            RightShiftMessage {
                input: local.sample.into(),
                shift_bits: AB::Expr::from_usize(self.initial_log_domain_size - self.k)
                    - local.round,
                result: local.quotient.into(),
            },
            local.active,
        );
        builder.when(local.active).assert_eq(
            local.sample,
            local.merkle_index + local.quotient * local.query_domain_size,
        );
        builder.when(local.active).assert_eq(
            local.inner_tree_id,
            local.global_query + AB::Expr::from_usize(self.inner_tree_id_offset),
        );
        builder.when(local.active).assert_eq(
            local.outer_tree_id,
            local.round + AB::Expr::from_usize(self.outer_tree_id_offset),
        );
        self.query_bus.send(
            builder,
            NativeTerminalWhirQueryMessage {
                round: local.round.into(),
                query: local.query.into(),
                global_query: local.global_query.into(),
                inner_tree_id: local.inner_tree_id.into(),
                outer_tree_id: local.outer_tree_id.into(),
                merkle_index_sample: local.sample.into(),
                merkle_index: local.merkle_index.into(),
                zi_root: local.zi_root.into(),
                zi: local.zi.into(),
                yi: local.yi.map(Into::into),
            },
            local.active,
        );
        let point_len = self.k * self.round_count + self.final_poly_len.ilog2() as usize;
        let after_folds = (local.round + AB::F::ONE) * AB::Expr::from_usize(self.k);
        self.weight_term_bus.send(
            builder,
            NativeTerminalWhirWeightTermMessage {
                ordinal: local.global_query + local.round + AB::Expr::ONE - local.is_final_round,
                after_folds: after_folds.clone(),
                length: AB::Expr::from_usize(point_len) - after_folds,
                generator: core::array::from_fn(|limb| {
                    if limb == 0 {
                        local.zi.into()
                    } else {
                        AB::Expr::ZERO
                    }
                }),
                scale: local.gamma_pow.map(Into::into),
            },
            local.active,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn generate_native_terminal_whir_query_trace(
    verification: &TerminalWhirVerification<F, EF, Digest>,
    transcript: &TranscriptLog<F, [F; 16]>,
    k: usize,
    initial_log_domain_size: usize,
    inner_tree_id_offset: usize,
    outer_tree_id_offset: usize,
    round_count: usize,
    final_poly_len: usize,
    layout: TerminalWhirLayout,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if k == 0
        || initial_log_domain_size <= k
        || round_count == 0
        || final_poly_len == 0
        || !final_poly_len.is_power_of_two()
    {
        return None;
    }
    let valid_rows = verification
        .rounds
        .iter()
        .map(|round| round.query_indices.len())
        .sum::<usize>();
    if valid_rows == 0 {
        return None;
    }
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeTerminalWhirQueryCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    let mut row_index = 0usize;
    let mut global_query = 0usize;
    for (round_index, round) in verification.rounds.iter().enumerate() {
        let bits = initial_log_domain_size.checked_sub(k + round_index)?;
        if bits == 0
            || bits >= u32::BITS as usize
            || round.round as usize != round_index
            || round.query_indices.len() != round.query_roots.len()
            || round.query_indices.len() != round.folded_values.len()
        {
            return None;
        }
        let query_tidx = round
            .transcript_span
            .end
            .operations
            .checked_sub(D_EF + round.query_indices.len())?;
        let query_domain_size = 1usize.checked_shl(bits as u32)?;
        let mask = u32::try_from(query_domain_size.checked_sub(1)?).ok()?;
        // The first `k` variables select coefficients inside a vector leaf;
        // Merkle queries range over the remaining row domain.
        let omega = F::two_adic_generator(initial_log_domain_size - k - round_index);
        let evaluation_layout = layout != TerminalWhirLayout::VectorAlphabet;
        let raw_omega = if evaluation_layout {
            F::two_adic_generator(usize::try_from(round.log_rs_domain_size).ok()?)
        } else {
            F::ZERO
        };
        let mut pre_claim = round.sumcheck_rounds.last()?.post_claim;
        if let Some(ood_value) = round.ood_value {
            pre_claim += round.gamma * ood_value;
        }
        for (query_index, ((&merkle_index, &zi), &yi)) in round
            .query_indices
            .iter()
            .zip(&round.query_roots)
            .zip(&round.folded_values)
            .enumerate()
        {
            let tidx = query_tidx.checked_add(query_index)?;
            let sample = *transcript.values().get(tidx)?;
            if sample.as_canonical_u32() & mask != merkle_index {
                return None;
            }
            let row = &mut values[row_index * width..(row_index + 1) * width];
            let cols: &mut NativeTerminalWhirQueryCols<F> = row.borrow_mut();
            cols.active = F::ONE;
            cols.round = F::from_usize(round_index);
            cols.query = F::from_usize(query_index);
            cols.global_query = F::from_usize(global_query);
            cols.is_first = F::from_bool(row_index == 0);
            cols.is_first_in_round = F::from_bool(query_index == 0);
            cols.tidx = F::from_usize(tidx);
            cols.num_queries = F::from_usize(round.query_indices.len());
            cols.query_domain_size = F::from_usize(query_domain_size);
            cols.omega = omega;
            cols.sample = sample;
            cols.sample_domain_root = omega.exp_u64(u64::from(sample.as_canonical_u32()));
            cols.quotient = F::from_u32(sample.as_canonical_u32() >> bits);
            cols.merkle_index = F::from_u32(merkle_index);
            cols.inner_tree_id = F::from_usize(inner_tree_id_offset + global_query);
            cols.outer_tree_id = F::from_usize(outer_tree_id_offset + round_index);
            if evaluation_layout {
                let zi_root = terminal_whir_query_root::<F>(
                    layout,
                    round_index == 0,
                    merkle_index as usize,
                    usize::try_from(round.log_rs_domain_size).ok()?,
                    k,
                )
                .ok()?;
                if zi_root.exp_power_of_2(k) != zi {
                    return None;
                }
                cols.zi_root = zi_root;
                cols.zi = zi;
                cols.raw_omega = raw_omega;
                cols.coordinate_exponent = F::from_u32(merkle_index);
                cols.subgroup_root = raw_omega.exp_u64(u64::from(merkle_index));
            } else {
                cols.zi_root = zi;
                cols.zi = zi;
            }
            copy_ext(&mut cols.yi, yi);
            copy_ext(&mut cols.gamma, round.gamma);
            copy_ext(
                &mut cols.gamma_pow,
                round.gamma.exp_u64(query_index as u64 + 2),
            );
            cols.is_final_round = F::from_bool(round_index + 1 == round_count);
            if round_index + 1 != round_count {
                cols.round_delta_inverse =
                    (F::from_usize(round_index) - F::from_usize(round_count - 1)).inverse();
            }
            copy_ext(&mut cols.pre_claim, pre_claim);
            copy_ext(&mut cols.post_claim, round.post_round_claim);
            pre_claim += round.gamma.exp_u64(query_index as u64 + 2) * yi;
            row_index += 1;
            global_query += 1;
        }
        if pre_claim != round.post_round_claim {
            return None;
        }
    }
    Some(RowMajorMatrix::new(values, width))
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

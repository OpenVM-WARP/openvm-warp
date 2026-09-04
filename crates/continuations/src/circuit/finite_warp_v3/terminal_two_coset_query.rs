//! Production coefficient-two-coset terminal WHIR query verification.
//!
//! The legacy terminal query AIR is for WARP's vector alphabet: its queried
//! row root is already the post-fold WHIR point.  The production terminal
//! code instead commits one scalar GRS codeword over two disjoint cosets.  In
//! its initial round, physical index `2t + b` starts the opened coset at
//! `omega^t * g^b`; in every scalar round the WHIR point is that root raised
//! to `2^k`.  Aliasing the two points accepts the wrong folding geometry.
//!
//! This module retains the terminal verifier's typed buses and transcript
//! schedule, but constrains the native two-coset root map and all four
//! squarings for the production `k = 4` profile.  Later rounds use ordinary
//! scalar-RS roots, as required by native terminal WHIR.

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{StructReflection, StructReflectionHelper, SubAir};
use openvm_recursion_circuit::{
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
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{
        extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing,
        PrimeField32, TwoAdicField,
    },
    transcript::TranscriptLog,
    warp_accum::TerminalWhirVerification,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::FiniteWarpV3TwoCosetTerminalProfile;

/// Production terminal WHIR folds sixteen scalar evaluations per query.
pub const FINITE_WARP_V3_TWO_COSET_K: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiniteWarpV3TwoCosetQueryError {
    UnsupportedProfile,
    VerificationShape,
    Transcript,
    QueryIndex,
    QueryRoot,
    FoldedClaim,
    TraceHeight,
}

impl core::fmt::Display for FiniteWarpV3TwoCosetQueryError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "finite WARP v3 terminal two-coset query: {self:?}"
        )
    }
}

impl std::error::Error for FiniteWarpV3TwoCosetQueryError {}

/// The legacy terminal query columns plus the data needed to distinguish a
/// scalar coset root from its post-fold point.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FiniteWarpV3TwoCosetTerminalWhirQueryCols<T> {
    pub active: T,
    pub round: T,
    pub query: T,
    pub global_query: T,
    pub is_first: T,
    pub is_first_in_round: T,
    pub tidx: T,
    pub num_queries: T,
    pub query_domain_size: T,
    /// Generator of the post-fold query domain, supplied by the round AIR.
    pub omega: T,
    /// Generator used to derive the pre-fold scalar coset root.
    pub root_omega: T,
    pub sample: T,
    pub quotient: T,
    pub merkle_index: T,
    pub inner_tree_id: T,
    pub outer_tree_id: T,
    /// Pre-fold scalar coset root.
    pub zi_root: T,
    /// Successive squares of `zi_root`; the final entry is the WHIR point.
    pub root_squares: [T; FINITE_WARP_V3_TWO_COSET_K],
    /// Post-fold WHIR point `zi_root^(2^k)`.
    pub zi: T,
    pub yi: [T; D_EF],
    pub gamma: [T; D_EF],
    pub gamma_pow: [T; D_EF],
    pub is_final_round: T,
    pub round_delta_inverse: T,
    pub pre_claim: [T; D_EF],
    pub post_claim: [T; D_EF],
    pub is_initial: T,
    pub parity: T,
    pub subgroup_index: T,
    pub subgroup_root: T,
}

#[derive(Clone, Debug)]
pub struct FiniteWarpV3TwoCosetTerminalWhirQueryAir {
    pub transcript_bus: TranscriptBus,
    pub verify_queries_bus: NativeTerminalWhirVerifyQueriesBus,
    pub query_bus: NativeTerminalWhirQueryBus,
    pub weight_term_bus: NativeTerminalWhirWeightTermBus,
    pub exp_bits_len_bus: ExpBitsLenBus,
    pub right_shift_bus: RightShiftBus,
    pub initial_log_domain_size: usize,
    pub round_count: usize,
    pub final_poly_len: usize,
    pub inner_tree_id_offset: usize,
    pub outer_tree_id_offset: usize,
}

impl FiniteWarpV3TwoCosetTerminalWhirQueryAir {
    pub fn new(
        profile: &FiniteWarpV3TwoCosetTerminalProfile,
        transcript_bus: TranscriptBus,
        verify_queries_bus: NativeTerminalWhirVerifyQueriesBus,
        query_bus: NativeTerminalWhirQueryBus,
        weight_term_bus: NativeTerminalWhirWeightTermBus,
        exp_bits_len_bus: ExpBitsLenBus,
        right_shift_bus: RightShiftBus,
        round_count: usize,
        final_poly_len: usize,
        inner_tree_id_offset: usize,
        outer_tree_id_offset: usize,
    ) -> Result<Self, FiniteWarpV3TwoCosetQueryError> {
        if profile.whir_k() != FINITE_WARP_V3_TWO_COSET_K
            || profile.alpha_len() != profile.log_message_len() + 1
            || profile.log_message_len() > F::TWO_ADICITY
            || profile.alpha_len() <= FINITE_WARP_V3_TWO_COSET_K
            || profile.alpha_len() - FINITE_WARP_V3_TWO_COSET_K > 27
            || round_count == 0
            || final_poly_len == 0
            || !final_poly_len.is_power_of_two()
        {
            return Err(FiniteWarpV3TwoCosetQueryError::UnsupportedProfile);
        }
        Ok(Self {
            transcript_bus,
            verify_queries_bus,
            query_bus,
            weight_term_bus,
            exp_bits_len_bus,
            right_shift_bus,
            initial_log_domain_size: profile.alpha_len(),
            round_count,
            final_poly_len,
            inner_tree_id_offset,
            outer_tree_id_offset,
        })
    }
}

impl BaseAirWithPublicValues<F> for FiniteWarpV3TwoCosetTerminalWhirQueryAir {}
impl PartitionedBaseAir<F> for FiniteWarpV3TwoCosetTerminalWhirQueryAir {}
impl BaseAir<F> for FiniteWarpV3TwoCosetTerminalWhirQueryAir {
    fn width(&self) -> usize {
        FiniteWarpV3TwoCosetTerminalWhirQueryCols::<F>::width()
    }
}

fn assert_ext_eq<AB>(builder: &mut AB, left: [AB::Var; D_EF], right: [AB::Expr; D_EF])
where
    AB: AirBuilder<F = F>,
    AB::Expr: From<AB::Var>,
{
    for (left, right) in left.into_iter().zip(right) {
        builder.assert_eq(left, right);
    }
}

impl<AB> Air<AB> for FiniteWarpV3TwoCosetTerminalWhirQueryAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield:
        BinomiallyExtendable<{ D_EF }> + TwoAdicField,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main
            .row_slice(0)
            .expect("finite two-coset terminal query row");
        let next_row = main
            .row_slice(1)
            .expect("finite two-coset terminal next query row");
        let local: &FiniteWarpV3TwoCosetTerminalWhirQueryCols<AB::Var> = (*local_row).borrow();
        let next: &FiniteWarpV3TwoCosetTerminalWhirQueryCols<AB::Var> = (*next_row).borrow();

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
        for flag in [
            local.is_first,
            local.is_final_round,
            local.is_initial,
            local.parity,
        ] {
            builder.assert_bool(flag);
        }
        builder
            .when_first_row()
            .assert_eq(local.is_first, local.active);
        builder
            .when_first_row()
            .when(local.active)
            .assert_one(local.is_initial);
        builder.when_first_row().when(local.active).assert_eq(
            local.query_domain_size,
            AB::Expr::from_usize(
                1usize << (self.initial_log_domain_size - FINITE_WARP_V3_TWO_COSET_K),
            ),
        );
        builder.when_first_row().when(local.active).assert_eq(
            local.root_omega,
            AB::Expr::from_u32(
                F::two_adic_generator(self.initial_log_domain_size - 1).as_canonical_u32(),
            ),
        );
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
        same.assert_eq(next.root_omega, local.root_omega);
        same.assert_eq(next.is_initial, local.is_initial);
        assert_ext_eq(&mut same, next.gamma, local.gamma.map(Into::into));
        assert_ext_eq(
            &mut same,
            next.gamma_pow,
            ext_field_multiply::<AB::Expr>(local.gamma, local.gamma_pow),
        );
        assert_ext_eq(
            &mut same,
            next.pre_claim,
            ext_field_add::<AB::Expr>(
                local.pre_claim,
                ext_field_multiply::<AB::Expr>(local.gamma_pow, local.yi),
            ),
        );
        assert_ext_eq(&mut same, next.post_claim, local.post_claim.map(Into::into));

        let next_round = next.active * next.is_first_in_round;
        let mut transition = builder.when_transition();
        let mut advance = transition.when(next_round);
        advance.assert_eq(local.query_domain_size, next.query_domain_size * AB::F::TWO);
        advance.assert_eq(next.omega, local.omega * local.omega);
        advance.assert_zero(next.is_initial);
        advance.assert_eq(
            next.root_omega,
            local.is_initial * local.root_omega
                + (AB::Expr::ONE - local.is_initial) * local.root_omega * local.root_omega,
        );

        assert_ext_eq(
            &mut builder.when(local.is_first_in_round),
            local.gamma_pow,
            ext_field_multiply::<AB::Expr>(local.gamma, local.gamma),
        );
        assert_ext_eq(
            &mut builder.when(round_end),
            local.post_claim,
            ext_field_add::<AB::Expr>(
                local.pre_claim,
                ext_field_multiply::<AB::Expr>(local.gamma_pow, local.yi),
            ),
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

        let query_bits =
            AB::Expr::from_usize(self.initial_log_domain_size - FINITE_WARP_V3_TWO_COSET_K)
                - local.round;
        let initial_enabled = local.active * local.is_initial;
        let ordinary_enabled = local.active * (AB::Expr::ONE - local.is_initial);
        self.right_shift_bus.lookup_key(
            builder,
            RightShiftMessage {
                input: local.sample.into(),
                shift_bits: query_bits.clone(),
                result: local.quotient.into(),
            },
            local.active,
        );
        // The initial two-coset row needs both `sample >> query_bits` and a
        // different exponentiation whose source is `merkle_index >> 1`.
        // `ExpBitsLenAir` emits one exponentiation lookup together with each
        // right-shift lookup, so bind the shift-only record to the canonical
        // identity `1^0 = 1`.  Later scalar-RS rounds combine their genuine
        // exponentiation and shift in one record.
        self.exp_bits_len_bus.lookup_key(
            builder,
            ExpBitsLenMessage {
                base: AB::Expr::ONE,
                bit_src: local.sample.into(),
                num_bits: AB::Expr::ZERO,
                result: AB::Expr::ONE,
            },
            initial_enabled.clone(),
        );
        builder.when(local.active).assert_eq(
            local.sample,
            local.merkle_index + local.quotient * local.query_domain_size,
        );

        builder.when(initial_enabled.clone()).assert_eq(
            local.merkle_index,
            AB::Expr::TWO * local.subgroup_index + local.parity,
        );
        self.exp_bits_len_bus.lookup_key(
            builder,
            ExpBitsLenMessage {
                base: local.root_omega.into(),
                bit_src: local.subgroup_index.into(),
                num_bits: AB::Expr::from_usize(
                    self.initial_log_domain_size - FINITE_WARP_V3_TWO_COSET_K - 1,
                ),
                result: local.subgroup_root.into(),
            },
            initial_enabled.clone(),
        );
        builder.when(initial_enabled).assert_eq(
            local.zi_root,
            local.subgroup_root
                * (AB::Expr::ONE
                    + local.parity
                        * (AB::Expr::from_u32(F::GENERATOR.as_canonical_u32()) - AB::Expr::ONE)),
        );

        self.exp_bits_len_bus.lookup_key(
            builder,
            ExpBitsLenMessage {
                base: local.root_omega.into(),
                bit_src: local.sample.into(),
                num_bits: query_bits,
                result: local.zi_root.into(),
            },
            ordinary_enabled.clone(),
        );
        builder
            .when(ordinary_enabled.clone())
            .assert_zero(local.parity);
        builder
            .when(ordinary_enabled.clone())
            .assert_zero(local.subgroup_index);
        builder
            .when(ordinary_enabled)
            .assert_zero(local.subgroup_root);

        builder
            .when(local.active)
            .assert_eq(local.root_squares[0], local.zi_root * local.zi_root);
        for index in 1..FINITE_WARP_V3_TWO_COSET_K {
            builder.when(local.active).assert_eq(
                local.root_squares[index],
                local.root_squares[index - 1] * local.root_squares[index - 1],
            );
        }
        builder
            .when(local.active)
            .assert_eq(local.zi, local.root_squares[FINITE_WARP_V3_TWO_COSET_K - 1]);

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
        let point_len =
            FINITE_WARP_V3_TWO_COSET_K * self.round_count + self.final_poly_len.ilog2() as usize;
        let after_folds =
            (local.round + AB::F::ONE) * AB::Expr::from_usize(FINITE_WARP_V3_TWO_COSET_K);
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

fn checked_query_sample(
    transcript: &TranscriptLog<F, [F; 16]>,
    tidx: usize,
    query_bits: usize,
    expected_index: u32,
) -> Result<F, FiniteWarpV3TwoCosetQueryError> {
    if query_bits == 0 || query_bits >= u32::BITS as usize {
        return Err(FiniteWarpV3TwoCosetQueryError::UnsupportedProfile);
    }
    let sample = *transcript
        .values()
        .get(tidx)
        .ok_or(FiniteWarpV3TwoCosetQueryError::Transcript)?;
    if transcript.samples().get(tidx) != Some(&true) {
        return Err(FiniteWarpV3TwoCosetQueryError::Transcript);
    }
    let mask = (1u32 << query_bits) - 1;
    if sample.as_canonical_u32() & mask != expected_index {
        return Err(FiniteWarpV3TwoCosetQueryError::QueryIndex);
    }
    Ok(sample)
}

fn two_coset_query_root_and_squares(
    profile: &FiniteWarpV3TwoCosetTerminalProfile,
    round_index: usize,
    merkle_index: u32,
) -> Result<(F, [F; FINITE_WARP_V3_TWO_COSET_K]), FiniteWarpV3TwoCosetQueryError> {
    let log_rs_domain_size = profile
        .alpha_len()
        .checked_sub(round_index)
        .ok_or(FiniteWarpV3TwoCosetQueryError::UnsupportedProfile)?;
    let raw_root = if round_index == 0 {
        let omega = F::two_adic_generator(profile.log_message_len());
        let mut root = omega.exp_u64((merkle_index >> 1) as u64);
        if merkle_index & 1 == 1 {
            root *= F::GENERATOR;
        }
        root
    } else {
        F::two_adic_generator(log_rs_domain_size).exp_u64(merkle_index as u64)
    };
    let mut squares = [F::ZERO; FINITE_WARP_V3_TWO_COSET_K];
    let mut square = raw_root;
    for value in &mut squares {
        square *= square;
        *value = square;
    }
    Ok((raw_root, squares))
}

/// Generate the finite-v3 replacement query trace from a native recorded
/// verification.  The native helper is the differential oracle for every
/// pre-fold root; the recorded root must equal its fourth successive square.
#[allow(clippy::too_many_arguments)]
pub fn generate_finite_warp_v3_two_coset_terminal_whir_query_trace(
    profile: &FiniteWarpV3TwoCosetTerminalProfile,
    verification: &TerminalWhirVerification<F, EF, Digest>,
    transcript: &TranscriptLog<F, [F; 16]>,
    inner_tree_id_offset: usize,
    outer_tree_id_offset: usize,
    round_count: usize,
    final_poly_len: usize,
    required_height: Option<usize>,
) -> Result<RowMajorMatrix<F>, FiniteWarpV3TwoCosetQueryError> {
    if profile.whir_k() != FINITE_WARP_V3_TWO_COSET_K
        || profile.alpha_len() != profile.log_message_len() + 1
        || profile.log_message_len() > F::TWO_ADICITY
        || round_count == 0
        || final_poly_len == 0
        || !final_poly_len.is_power_of_two()
        || verification.rounds.len() != round_count
    {
        return Err(FiniteWarpV3TwoCosetQueryError::UnsupportedProfile);
    }
    let valid_rows = verification
        .rounds
        .iter()
        .try_fold(0usize, |total, round| {
            total.checked_add(round.query_indices.len())
        })
        .ok_or(FiniteWarpV3TwoCosetQueryError::VerificationShape)?;
    if valid_rows == 0 {
        return Err(FiniteWarpV3TwoCosetQueryError::VerificationShape);
    }
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return Err(FiniteWarpV3TwoCosetQueryError::TraceHeight);
    }
    let width = FiniteWarpV3TwoCosetTerminalWhirQueryCols::<F>::width();
    let cells = height
        .checked_mul(width)
        .ok_or(FiniteWarpV3TwoCosetQueryError::TraceHeight)?;
    let mut values = F::zero_vec(cells);
    let mut row_index = 0usize;
    let mut global_query = 0usize;

    for (round_index, round) in verification.rounds.iter().enumerate() {
        let bits = profile
            .alpha_len()
            .checked_sub(FINITE_WARP_V3_TWO_COSET_K + round_index)
            .ok_or(FiniteWarpV3TwoCosetQueryError::UnsupportedProfile)?;
        if bits == 0
            || bits >= u32::BITS as usize
            || round.round as usize != round_index
            || round.log_rs_domain_size as usize != profile.alpha_len() - round_index
            || round.query_indices.is_empty()
            || round.query_indices.len() != round.query_roots.len()
            || round.query_indices.len() != round.folded_values.len()
        {
            return Err(FiniteWarpV3TwoCosetQueryError::VerificationShape);
        }
        let query_tidx = round
            .transcript_span
            .end
            .operations
            .checked_sub(D_EF + round.query_indices.len())
            .ok_or(FiniteWarpV3TwoCosetQueryError::Transcript)?;
        let query_domain_size = 1usize
            .checked_shl(bits as u32)
            .ok_or(FiniteWarpV3TwoCosetQueryError::UnsupportedProfile)?;
        let omega = F::two_adic_generator(bits);
        let root_omega_log = profile.alpha_len() - round_index.max(1);
        let root_omega = F::two_adic_generator(root_omega_log);
        let mut pre_claim = round
            .sumcheck_rounds
            .last()
            .map(|record| record.post_claim)
            .ok_or(FiniteWarpV3TwoCosetQueryError::VerificationShape)?;
        if let Some(ood_value) = round.ood_value {
            pre_claim += round.gamma * ood_value;
        }

        for (query_index, ((&merkle_index, &recorded_zi), &yi)) in round
            .query_indices
            .iter()
            .zip(&round.query_roots)
            .zip(&round.folded_values)
            .enumerate()
        {
            let tidx = query_tidx
                .checked_add(query_index)
                .ok_or(FiniteWarpV3TwoCosetQueryError::Transcript)?;
            let sample = checked_query_sample(transcript, tidx, bits, merkle_index)?;
            let (raw_root, squares) =
                two_coset_query_root_and_squares(profile, round_index, merkle_index)?;
            if squares[FINITE_WARP_V3_TWO_COSET_K - 1] != recorded_zi {
                return Err(FiniteWarpV3TwoCosetQueryError::QueryRoot);
            }

            let row = values
                .get_mut(row_index * width..(row_index + 1) * width)
                .ok_or(FiniteWarpV3TwoCosetQueryError::TraceHeight)?;
            let cols: &mut FiniteWarpV3TwoCosetTerminalWhirQueryCols<F> = row.borrow_mut();
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
            cols.root_omega = root_omega;
            cols.sample = sample;
            cols.quotient = F::from_u32(sample.as_canonical_u32() >> bits);
            cols.merkle_index = F::from_u32(merkle_index);
            cols.inner_tree_id = F::from_usize(inner_tree_id_offset + global_query);
            cols.outer_tree_id = F::from_usize(outer_tree_id_offset + round_index);
            cols.zi_root = raw_root;
            cols.root_squares = squares;
            cols.zi = recorded_zi;
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
            cols.is_initial = F::from_bool(round_index == 0);
            if round_index == 0 {
                cols.parity = F::from_u32(merkle_index & 1);
                cols.subgroup_index = F::from_u32(merkle_index >> 1);
                cols.subgroup_root = root_omega.exp_u64((merkle_index >> 1) as u64);
            }

            pre_claim += round.gamma.exp_u64(query_index as u64 + 2) * yi;
            row_index += 1;
            global_query += 1;
        }
        if pre_claim != round.post_round_claim {
            return Err(FiniteWarpV3TwoCosetQueryError::FoldedClaim);
        }
    }
    Ok(RowMajorMatrix::new(values, width))
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

#[cfg(test)]
#[path = "terminal_two_coset_query_tests.rs"]
mod tests;

//! Binary-coset folding adapter for production coefficient-two-coset terminal
//! WHIR.
//!
//! This is the recursive lane's scalar WHIR fold equation connected to the
//! terminal WARP buses.  It must not be replaced by the legacy vector-alphabet
//! rule `left + alpha * right`.

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::utils::assert_array_eq;
use openvm_recursion_circuit::{
    native_warp::terminal::{
        NativeTerminalWhirAlphaBus, NativeTerminalWhirAlphaMessage, NativeTerminalWhirFoldingBus,
        NativeTerminalWhirFoldingCols, NativeTerminalWhirFoldingMessage,
    },
    utils::{base_to_ext, ext_field_multiply, ext_field_subtract},
};
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{
        extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing,
        TwoAdicField,
    },
    warp_accum::{
        terminal_whir::{terminal_whir_query_root, TerminalWhirLayout},
        TerminalWhirVerification,
    },
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::{FiniteWarpV3TwoCosetTerminalProfile, FINITE_WARP_V3_TWO_COSET_K};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiniteWarpV3TwoCosetFoldingError {
    UnsupportedProfile,
    VerificationShape,
    QueryRoot,
    OpeningShape,
    FoldedClaim,
    TraceHeight,
}

impl core::fmt::Display for FiniteWarpV3TwoCosetFoldingError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "finite WARP v3 terminal two-coset folding: {self:?}"
        )
    }
}

impl std::error::Error for FiniteWarpV3TwoCosetFoldingError {}

#[derive(Clone, Debug)]
pub struct FiniteWarpV3TwoCosetTerminalWhirFoldingAir {
    pub alpha_bus: NativeTerminalWhirAlphaBus,
    pub folding_bus: NativeTerminalWhirFoldingBus,
    pub k: usize,
}

impl FiniteWarpV3TwoCosetTerminalWhirFoldingAir {
    pub fn new(
        profile: &FiniteWarpV3TwoCosetTerminalProfile,
        alpha_bus: NativeTerminalWhirAlphaBus,
        folding_bus: NativeTerminalWhirFoldingBus,
    ) -> Result<Self, FiniteWarpV3TwoCosetFoldingError> {
        if profile.whir_k() != FINITE_WARP_V3_TWO_COSET_K
            || profile.alpha_len() != profile.log_message_len() + 1
            || profile.log_message_len() > F::TWO_ADICITY
        {
            return Err(FiniteWarpV3TwoCosetFoldingError::UnsupportedProfile);
        }
        Ok(Self {
            alpha_bus,
            folding_bus,
            k: FINITE_WARP_V3_TWO_COSET_K,
        })
    }
}

impl BaseAirWithPublicValues<F> for FiniteWarpV3TwoCosetTerminalWhirFoldingAir {}
impl PartitionedBaseAir<F> for FiniteWarpV3TwoCosetTerminalWhirFoldingAir {}
impl BaseAir<F> for FiniteWarpV3TwoCosetTerminalWhirFoldingAir {
    fn width(&self) -> usize {
        NativeTerminalWhirFoldingCols::<F>::width()
    }
}

impl<AB> Air<AB> for FiniteWarpV3TwoCosetTerminalWhirFoldingAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main
            .row_slice(0)
            .expect("finite two-coset terminal folding row");
        let local: &NativeTerminalWhirFoldingCols<AB::Var> = (*local_row).borrow();

        builder.assert_bool(local.active);
        builder.assert_bool(local.is_root);
        builder.when(local.is_root).assert_one(local.active);
        builder.when(local.is_root).assert_one(local.twiddle);
        builder.when(local.is_root).assert_zero(local.coset_index);
        builder
            .when(local.is_root)
            .assert_eq(local.height, AB::Expr::from_usize(self.k));
        builder
            .when(local.is_root)
            .assert_eq(local.z_final, local.coset_shift * local.coset_shift);
        assert_array_eq(&mut builder.when(local.is_root), local.value, local.y_final);

        let x = local.twiddle * local.coset_shift;
        let term = ext_field_multiply::<AB::Expr>(
            ext_field_subtract::<AB::Expr>(local.alpha, base_to_ext::<AB::Expr>(x.clone())),
            ext_field_subtract::<AB::Expr>(local.left_value, local.right_value),
        );
        // value = left + (alpha - x) * (left - right) / (2x).
        assert_array_eq(
            builder,
            ext_field_multiply::<AB::Expr>(
                ext_field_subtract::<AB::Expr>(local.value, local.left_value),
                base_to_ext::<AB::Expr>(x * AB::Expr::TWO),
            ),
            term,
        );

        self.alpha_bus.lookup_key(
            builder,
            NativeTerminalWhirAlphaMessage {
                round: local.round.into(),
                fold: local.height - AB::Expr::ONE,
                challenge: local.alpha.map(Into::into),
            },
            local.active,
        );
        self.folding_bus.receive(
            builder,
            NativeTerminalWhirFoldingMessage {
                round: local.round.into(),
                query: local.query.into(),
                height: local.height - AB::Expr::ONE,
                coset_shift: local.coset_shift.into(),
                coset_size: AB::Expr::TWO * local.coset_size,
                coset_index: local.coset_index.into(),
                twiddle: local.twiddle.into(),
                value: local.left_value.map(Into::into),
                z_final: local.z_final.into(),
                y_final: local.y_final.map(Into::into),
            },
            local.active,
        );
        self.folding_bus.receive(
            builder,
            NativeTerminalWhirFoldingMessage {
                round: local.round.into(),
                query: local.query.into(),
                height: local.height - AB::Expr::ONE,
                coset_shift: local.coset_shift.into(),
                coset_size: AB::Expr::TWO * local.coset_size,
                coset_index: local.coset_index + local.coset_size,
                twiddle: -local.twiddle.into(),
                value: local.right_value.map(Into::into),
                z_final: local.z_final.into(),
                y_final: local.y_final.map(Into::into),
            },
            local.active,
        );
        self.folding_bus.send(
            builder,
            NativeTerminalWhirFoldingMessage {
                round: local.round.into(),
                query: local.query.into(),
                height: local.height.into(),
                coset_shift: local.coset_shift * local.coset_shift,
                coset_size: local.coset_size.into(),
                coset_index: local.coset_index.into(),
                twiddle: local.twiddle * local.twiddle,
                value: local.value.map(Into::into),
                z_final: local.z_final.into(),
                y_final: local.y_final.map(Into::into),
            },
            local.active - local.is_root,
        );
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct TwoCosetFoldRecord {
    round: u32,
    query: u32,
    coset_index: u32,
    height: u32,
    coset_size: u32,
    coset_shift: F,
    twiddle: F,
    z_final: F,
    value: EF,
    left_value: EF,
    right_value: EF,
    y_final: EF,
    alpha: EF,
}

/// Generate the exact scalar WHIR folding tree recorded by the native
/// verifier.  The implementation is deliberately checked and fallible even
/// though production uses fixed `k = 4`.
pub fn generate_finite_warp_v3_two_coset_terminal_whir_folding_trace(
    profile: &FiniteWarpV3TwoCosetTerminalProfile,
    verification: &TerminalWhirVerification<F, EF, Digest>,
    required_height: Option<usize>,
) -> Result<RowMajorMatrix<F>, FiniteWarpV3TwoCosetFoldingError> {
    let k = profile.whir_k();
    if k != FINITE_WARP_V3_TWO_COSET_K
        || profile.alpha_len() != profile.log_message_len() + 1
        || profile.log_message_len() > F::TWO_ADICITY
        || verification.rounds.is_empty()
    {
        return Err(FiniteWarpV3TwoCosetFoldingError::UnsupportedProfile);
    }
    let coset_size = 1usize
        .checked_shl(k as u32)
        .ok_or(FiniteWarpV3TwoCosetFoldingError::UnsupportedProfile)?;
    let per_query = coset_size
        .checked_sub(1)
        .ok_or(FiniteWarpV3TwoCosetFoldingError::UnsupportedProfile)?;
    let query_count = verification.rounds.iter().try_fold(0usize, |total, round| {
        total.checked_add(round.query_indices.len())
    });
    let query_count = query_count.ok_or(FiniteWarpV3TwoCosetFoldingError::VerificationShape)?;
    let valid_rows = query_count
        .checked_mul(per_query)
        .ok_or(FiniteWarpV3TwoCosetFoldingError::TraceHeight)?;
    if valid_rows == 0 {
        return Err(FiniteWarpV3TwoCosetFoldingError::VerificationShape);
    }
    let mut records = Vec::with_capacity(valid_rows);
    for (round_index, round) in verification.rounds.iter().enumerate() {
        let expected_log = profile
            .alpha_len()
            .checked_sub(round_index)
            .ok_or(FiniteWarpV3TwoCosetFoldingError::UnsupportedProfile)?;
        let count = round.query_indices.len();
        if round.round as usize != round_index
            || round.log_rs_domain_size as usize != expected_log
            || round.alphas.len() != k
            || count == 0
            || round.query_roots.len() != count
            || round.folded_values.len() != count
            || round.opened_rows.len() != count
        {
            return Err(FiniteWarpV3TwoCosetFoldingError::VerificationShape);
        }
        for query in 0..count {
            let opened = &round.opened_rows[query];
            if opened.len() != coset_size || opened.iter().any(|row| row.len() != 1) {
                return Err(FiniteWarpV3TwoCosetFoldingError::OpeningShape);
            }
            let raw_root = terminal_whir_query_root::<F>(
                TerminalWhirLayout::ScalarCoefficientTwoCoset,
                round_index == 0,
                round.query_indices[query] as usize,
                expected_log,
                k,
            )
            .map_err(|_| FiniteWarpV3TwoCosetFoldingError::QueryRoot)?;
            let z_final = round.query_roots[query];
            if raw_root.exp_power_of_2(k) != z_final {
                return Err(FiniteWarpV3TwoCosetFoldingError::QueryRoot);
            }
            let mut query_values = opened.iter().map(|row| row[0]).collect::<Vec<_>>();
            let record_start = records.len();
            let folded = record_binary_k_fold(
                &mut query_values,
                &round.alphas,
                raw_root,
                round_index,
                query,
                &mut records,
            )?;
            if folded != round.folded_values[query]
                || records.len().checked_sub(record_start) != Some(per_query)
            {
                return Err(FiniteWarpV3TwoCosetFoldingError::FoldedClaim);
            }
            for record in &mut records[record_start..] {
                record.z_final = z_final;
                record.y_final = folded;
            }
        }
    }
    if records.len() != valid_rows {
        return Err(FiniteWarpV3TwoCosetFoldingError::VerificationShape);
    }
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return Err(FiniteWarpV3TwoCosetFoldingError::TraceHeight);
    }
    let width = NativeTerminalWhirFoldingCols::<F>::width();
    let cells = height
        .checked_mul(width)
        .ok_or(FiniteWarpV3TwoCosetFoldingError::TraceHeight)?;
    let mut trace = F::zero_vec(cells);
    for (row, record) in trace.chunks_exact_mut(width).zip(&records) {
        let cols: &mut NativeTerminalWhirFoldingCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.round = F::from_u32(record.round);
        cols.query = F::from_u32(record.query);
        cols.is_root = F::from_bool(record.coset_size == 1);
        cols.coset_shift = record.coset_shift;
        cols.coset_index = F::from_u32(record.coset_index);
        cols.height = F::from_u32(record.height);
        cols.twiddle = record.twiddle;
        cols.coset_size = F::from_u32(record.coset_size);
        cols.z_final = record.z_final;
        copy_ext(&mut cols.value, record.value);
        copy_ext(&mut cols.left_value, record.left_value);
        copy_ext(&mut cols.right_value, record.right_value);
        copy_ext(&mut cols.y_final, record.y_final);
        copy_ext(&mut cols.alpha, record.alpha);
    }
    Ok(RowMajorMatrix::new(trace, width))
}

fn record_binary_k_fold(
    values: &mut [EF],
    alphas: &[EF],
    base_coset_shift: F,
    round: usize,
    query: usize,
    records: &mut Vec<TwoCosetFoldRecord>,
) -> Result<EF, FiniteWarpV3TwoCosetFoldingError> {
    let k = alphas.len();
    let expected = 1usize
        .checked_shl(k as u32)
        .ok_or(FiniteWarpV3TwoCosetFoldingError::OpeningShape)?;
    if k == 0 || values.len() != expected || k > F::TWO_ADICITY {
        return Err(FiniteWarpV3TwoCosetFoldingError::OpeningShape);
    }
    let round =
        u32::try_from(round).map_err(|_| FiniteWarpV3TwoCosetFoldingError::VerificationShape)?;
    let query =
        u32::try_from(query).map_err(|_| FiniteWarpV3TwoCosetFoldingError::VerificationShape)?;
    let omega_k = F::two_adic_generator(k);
    let omega_k_inv = omega_k.inverse();
    let twiddles = omega_k.powers().take(expected / 2).collect();
    let inverse_twiddles = omega_k_inv.powers().take(expected / 2).collect();
    let mut coset_shift = base_coset_shift;
    let mut coset_shift_inv = base_coset_shift.inverse();

    for (fold, &alpha) in alphas.iter().enumerate() {
        let size = expected >> (fold + 1);
        let (lo, hi) = values.split_at_mut(size);
        for index in 0..size {
            let twiddle_index = index
                .checked_shl(fold as u32)
                .ok_or(FiniteWarpV3TwoCosetFoldingError::OpeningShape)?;
            let twiddle = *twiddles
                .get(twiddle_index)
                .ok_or(FiniteWarpV3TwoCosetFoldingError::OpeningShape)?;
            let inverse_twiddle = *inverse_twiddles
                .get(twiddle_index)
                .ok_or(FiniteWarpV3TwoCosetFoldingError::OpeningShape)?;
            let x = twiddle * coset_shift;
            let x_inv = inverse_twiddle * coset_shift_inv;
            let left_value = lo[index];
            let right_value = hi[index];
            let value = left_value + (alpha - x) * (left_value - right_value) * x_inv.halve();
            records.push(TwoCosetFoldRecord {
                round,
                query,
                coset_index: u32::try_from(index)
                    .map_err(|_| FiniteWarpV3TwoCosetFoldingError::VerificationShape)?,
                height: u32::try_from(fold + 1)
                    .map_err(|_| FiniteWarpV3TwoCosetFoldingError::VerificationShape)?,
                coset_size: u32::try_from(size)
                    .map_err(|_| FiniteWarpV3TwoCosetFoldingError::VerificationShape)?,
                coset_shift,
                twiddle,
                value,
                left_value,
                right_value,
                alpha,
                ..Default::default()
            });
            lo[index] = value;
        }
        coset_shift *= coset_shift;
        coset_shift_inv *= coset_shift_inv;
    }
    values
        .first()
        .copied()
        .ok_or(FiniteWarpV3TwoCosetFoldingError::OpeningShape)
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

#[cfg(test)]
#[path = "terminal_two_coset_folding_tests.rs"]
mod tests;

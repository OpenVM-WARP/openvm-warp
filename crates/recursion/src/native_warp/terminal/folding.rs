use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    warp_accum::{
        terminal_whir::{terminal_whir_query_root, TerminalWhirLayout},
        TerminalWhirVerification,
    },
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing, TwoAdicField,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    native_warp::terminal::{
        NativeTerminalWhirAlphaBus, NativeTerminalWhirAlphaMessage, NativeTerminalWhirFoldingBus,
        NativeTerminalWhirFoldingMessage,
    },
    utils::{ext_field_add, ext_field_multiply},
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeTerminalWhirFoldingCols<T> {
    pub active: T,
    pub round: T,
    pub query: T,
    pub is_root: T,
    pub coset_shift: T,
    pub coset_index: T,
    pub height: T,
    pub twiddle: T,
    pub twiddle_inverse_half: T,
    pub coset_size: T,
    pub z_final: T,
    pub value: [T; D_EF],
    pub left_value: [T; D_EF],
    pub right_value: [T; D_EF],
    pub y_final: [T; D_EF],
    pub alpha: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(NativeTerminalWhirFoldingCols<u8>)]
pub struct NativeTerminalWhirFoldingAir {
    pub alpha_bus: NativeTerminalWhirAlphaBus,
    pub folding_bus: NativeTerminalWhirFoldingBus,
    pub k: usize,
    pub evaluation_layout: bool,
}

impl BaseAirWithPublicValues<F> for NativeTerminalWhirFoldingAir {}
impl PartitionedBaseAir<F> for NativeTerminalWhirFoldingAir {}
impl BaseAir<F> for NativeTerminalWhirFoldingAir {
    fn width(&self) -> usize {
        NativeTerminalWhirFoldingCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTerminalWhirFoldingAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("native terminal WHIR folding row");
        let local: &NativeTerminalWhirFoldingCols<AB::Var> = (*local).borrow();

        builder.assert_bool(local.active);
        builder.assert_bool(local.is_root);
        builder.when(local.is_root).assert_one(local.active);
        if !self.evaluation_layout {
            builder.when(local.is_root).assert_one(local.twiddle);
        }
        builder.when(local.is_root).assert_zero(local.coset_index);
        builder
            .when(local.is_root)
            .assert_eq(local.height, AB::Expr::from_usize(self.k));
        if self.evaluation_layout {
            // At the root, `twiddle = x^(2^(k-1))`; the post-fold WHIR
            // query point is therefore its square.
            builder
                .when(local.is_root)
                .assert_eq(local.z_final, local.twiddle * local.twiddle);
        } else {
            builder
                .when(local.is_root)
                .assert_eq(local.z_final, local.coset_shift);
        }
        assert_array_eq(&mut builder.when(local.is_root), local.value, local.y_final);

        if self.evaluation_layout {
            builder
                .when(local.active)
                .assert_one(AB::Expr::TWO * local.twiddle * local.twiddle_inverse_half);
            let alpha_minus_t = core::array::from_fn(|limb| {
                AB::Expr::from(local.alpha[limb])
                    - if limb == 0 {
                        AB::Expr::from(local.twiddle)
                    } else {
                        AB::Expr::ZERO
                    }
            });
            let lo_minus_hi = core::array::from_fn(|limb| {
                AB::Expr::from(local.left_value[limb]) - AB::Expr::from(local.right_value[limb])
            });
            let correction = ext_field_multiply::<AB::Expr>(alpha_minus_t, lo_minus_hi)
                .map(|value| value * local.twiddle_inverse_half);
            assert_array_eq(
                builder,
                local.value,
                ext_field_add::<AB::Expr>(local.left_value, correction),
            );
        } else {
            builder
                .when(local.active)
                .assert_zero(local.twiddle_inverse_half);
            assert_array_eq(
                builder,
                local.value,
                ext_field_add::<AB::Expr>(
                    local.left_value,
                    ext_field_multiply::<AB::Expr>(local.alpha, local.right_value),
                ),
            );
        }

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
                coset_index: if self.evaluation_layout {
                    local.coset_index.into()
                } else {
                    AB::Expr::TWO * local.coset_index
                },
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
                coset_index: if self.evaluation_layout {
                    local.coset_index + local.coset_size
                } else {
                    AB::Expr::TWO * local.coset_index + AB::Expr::ONE
                },
                // Binary evaluation folding pairs the points `t` and `-t`.
                // The previous layer publishes each child's squared twiddle:
                // the left child therefore keys `t`, while the right child
                // keys `-t`. Using `t` for both disconnects every right branch
                // of the folding tree even though the scalar fold value is
                // otherwise correct.
                twiddle: if self.evaluation_layout {
                    -AB::Expr::from(local.twiddle)
                } else {
                    local.twiddle.into()
                },
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
                coset_shift: local.coset_shift.into(),
                coset_size: local.coset_size.into(),
                coset_index: local.coset_index.into(),
                twiddle: if self.evaluation_layout {
                    local.twiddle * local.twiddle
                } else {
                    local.twiddle.into()
                },
                value: local.value.map(Into::into),
                z_final: local.z_final.into(),
                y_final: local.y_final.map(Into::into),
            },
            local.active - local.is_root,
        );
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct NativeTerminalFoldRecord {
    pub round: u32,
    pub query: u32,
    pub coset_index: u32,
    pub height: u32,
    pub coset_size: u32,
    pub coset_shift: F,
    pub twiddle: F,
    pub twiddle_inverse_half: F,
    pub z_final: F,
    pub value: EF,
    pub left_value: EF,
    pub right_value: EF,
    pub y_final: EF,
    pub alpha: EF,
}

pub fn generate_native_terminal_whir_folding_trace(
    verification: &TerminalWhirVerification<F, EF, Digest>,
    k: usize,
    required_height: Option<usize>,
    layout: TerminalWhirLayout,
) -> Option<RowMajorMatrix<F>> {
    if k == 0 {
        return None;
    }
    let per_query = (1usize << k).checked_sub(1)?;
    let query_count = verification
        .rounds
        .iter()
        .map(|round| round.query_indices.len())
        .sum::<usize>();
    let valid_rows = query_count.checked_mul(per_query)?;
    if valid_rows == 0 {
        return None;
    }
    let mut records = Vec::with_capacity(valid_rows);
    for (round_index, round) in verification.rounds.iter().enumerate() {
        if round.alphas.len() != k
            || round.opened_rows.len() != round.query_indices.len()
            || round.folded_values.len() != round.query_indices.len()
        {
            return None;
        }
        for query in 0..round.query_indices.len() {
            let opened = round.opened_rows.get(query)?;
            let evaluation_layout =
                opened.len() == (1usize << k) && opened.iter().all(|row| row.len() == 1);
            let mut values = if evaluation_layout {
                opened.iter().map(|row| row[0]).collect::<Vec<_>>()
            } else {
                let [opened] = opened.as_slice() else {
                    return None;
                };
                opened.clone()
            };
            let zi = round.query_roots[query];
            let yi = if evaluation_layout {
                // Use the recorded root compiled by
                // `terminal_whir_query_root`; reconstructing `x` from a
                // generic two-adic domain would ignore the descriptor-bound
                // layout.
                let x = terminal_whir_query_root::<F>(
                    layout,
                    round_index == 0,
                    round.query_indices[query] as usize,
                    usize::try_from(round.log_rs_domain_size).ok()?,
                    k,
                )
                .ok()?;
                if x.exp_power_of_2(k) != zi {
                    return None;
                }
                binary_k_fold_records(
                    &mut values,
                    &round.alphas,
                    x,
                    round_index,
                    query,
                    &mut records,
                )?
            } else {
                monomial_k_fold_records(
                    &mut values,
                    &round.alphas,
                    zi,
                    round_index,
                    query,
                    &mut records,
                )?
            };
            if yi != round.folded_values[query] {
                return None;
            }
            for record in records.iter_mut().rev().take(per_query) {
                record.z_final = zi;
                record.y_final = yi;
            }
        }
    }
    if records.len() != valid_rows {
        return None;
    }
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeTerminalWhirFoldingCols::<F>::width();
    let mut trace = F::zero_vec(height * width);
    for (row, record) in trace.chunks_exact_mut(width).take(valid_rows).zip(&records) {
        let cols: &mut NativeTerminalWhirFoldingCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.round = F::from_u32(record.round);
        cols.query = F::from_u32(record.query);
        cols.is_root = F::from_bool(record.coset_size == 1);
        cols.coset_shift = record.coset_shift;
        cols.coset_index = F::from_u32(record.coset_index);
        cols.height = F::from_u32(record.height);
        cols.twiddle = record.twiddle;
        cols.twiddle_inverse_half = record.twiddle_inverse_half;
        cols.coset_size = F::from_u32(record.coset_size);
        cols.z_final = record.z_final;
        copy_ext(&mut cols.value, record.value);
        copy_ext(&mut cols.left_value, record.left_value);
        copy_ext(&mut cols.right_value, record.right_value);
        copy_ext(&mut cols.y_final, record.y_final);
        copy_ext(&mut cols.alpha, record.alpha);
    }
    Some(RowMajorMatrix::new(trace, width))
}

fn binary_k_fold_records(
    values: &mut [EF],
    alphas: &[EF],
    x: F,
    round: usize,
    query: usize,
    records: &mut Vec<NativeTerminalFoldRecord>,
) -> Option<EF> {
    let n = values.len();
    let k = alphas.len();
    if n != 1usize.checked_shl(k as u32)? || x == F::ZERO {
        return None;
    }
    let omega = F::two_adic_generator(k);
    let omega_inverse = omega.inverse();
    let twiddles = omega.powers().take(n / 2).collect();
    let inverse_twiddles = omega_inverse.powers().take(n / 2).collect();
    let mut x_power = x;
    let mut x_inverse_power = x.inverse();
    let mut len = n;
    for (fold, &alpha) in alphas.iter().enumerate() {
        let size = len / 2;
        for index in 0..size {
            let twiddle_index = index.checked_shl(fold as u32)?;
            let t = twiddles[twiddle_index] * x_power;
            let t_inverse_half = inverse_twiddles[twiddle_index] * x_inverse_power.halve();
            let left_value = values[index];
            let right_value = values[index + size];
            let value = left_value + (alpha - t) * (left_value - right_value) * t_inverse_half;
            records.push(NativeTerminalFoldRecord {
                round: round.try_into().ok()?,
                query: query.try_into().ok()?,
                coset_index: index.try_into().ok()?,
                height: (fold + 1).try_into().ok()?,
                coset_size: size.try_into().ok()?,
                coset_shift: x,
                twiddle: t,
                twiddle_inverse_half: t_inverse_half,
                value,
                left_value,
                right_value,
                alpha,
                ..Default::default()
            });
            values[index] = value;
        }
        len = size;
        x_power *= x_power;
        x_inverse_power *= x_inverse_power;
    }
    values.first().copied()
}

/// Record the exact low-variable fold used by backend
/// `fold_monomial_coefficients`: at each round, coefficients `(2i, 2i+1)`
/// become `c[2i] + alpha * c[2i+1]`.
fn monomial_k_fold_records(
    values: &mut [EF],
    alphas: &[EF],
    base_coset_shift: F,
    round: usize,
    query: usize,
    records: &mut Vec<NativeTerminalFoldRecord>,
) -> Option<EF> {
    let n = values.len();
    let k = alphas.len();
    if n != 1usize.checked_shl(k as u32)? {
        return None;
    }
    let coset_shift = base_coset_shift;
    for (fold, &alpha) in alphas.iter().enumerate() {
        let size = n >> (fold + 1);
        for index in 0..size {
            let twiddle = F::ONE;
            let left_value = values[2 * index];
            let right_value = values[2 * index + 1];
            let value = left_value + alpha * right_value;
            records.push(NativeTerminalFoldRecord {
                round: round.try_into().ok()?,
                query: query.try_into().ok()?,
                coset_index: index.try_into().ok()?,
                height: (fold + 1).try_into().ok()?,
                coset_size: size.try_into().ok()?,
                coset_shift,
                twiddle,
                value,
                left_value,
                right_value,
                alpha,
                ..Default::default()
            });
            values[index] = value;
        }
    }
    values.first().copied()
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recorded_vector_alphabet_fold_matches_monomial_evaluation() {
        let original = (1..=16).map(EF::from_u32).collect::<Vec<_>>();
        let alphas = [2, 3, 5, 7]
            .into_iter()
            .map(EF::from_u32)
            .collect::<Vec<_>>();
        let expected = original
            .iter()
            .enumerate()
            .map(|(index, &coefficient)| {
                alphas
                    .iter()
                    .enumerate()
                    .filter(|(bit, _)| (index >> bit) & 1 == 1)
                    .fold(coefficient, |value, (_, &alpha)| value * alpha)
            })
            .sum::<EF>();
        let mut values = original.clone();
        let mut records = Vec::new();
        let actual =
            monomial_k_fold_records(&mut values, &alphas, F::from_u32(11), 2, 3, &mut records)
                .expect("monomial folding records");

        assert_eq!(actual, expected);
        assert_eq!(records.len(), 15);
        for (index, record) in records.iter().take(8).enumerate() {
            assert_eq!(record.height, 1);
            assert_eq!(record.coset_size, 8);
            assert_eq!(record.coset_index as usize, index);
            assert_eq!(record.left_value, original[2 * index]);
            assert_eq!(record.right_value, original[2 * index + 1]);
        }
    }
}

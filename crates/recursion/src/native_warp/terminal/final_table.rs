use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{encoder::Encoder, utils::assert_array_eq, ColumnsAir, SubAir};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, warp_accum::TerminalWhirVerification, BaseAirWithPublicValues,
    PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    native_warp::terminal::{
        NativeTerminalWhirFinalClaimBus, NativeTerminalWhirFinalClaimMessage,
        NativeTerminalWhirFinalContextBus, NativeTerminalWhirFinalContextMessage,
        NativeTerminalWhirFinalPolyBus, NativeTerminalWhirFinalPolyMessage,
        NativeTerminalWhirFinalWeightBus, NativeTerminalWhirFinalWeightMessage,
        NATIVE_TERMINAL_SELECTOR_MAX_FLAG_DEGREE,
    },
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeTerminalWhirFinalTableCols<T> {
    pub active: T,
    pub index: T,
    pub is_first: T,
    pub is_last: T,
    pub remaining: T,
    pub remaining_inverse: T,
    pub final_poly_tidx: T,
    pub final_weight_tidx: T,
    pub claim: [T; D_EF],
    pub coefficient: [T; D_EF],
    pub weight: [T; D_EF],
}

/// Binds the small terminal polynomial and structured-weight table to the
/// Fiat-Shamir transcript. The polynomial is still in multilinear
/// coefficient form; [`NativeTerminalWhirMobiusAir`] converts it to Boolean
/// evaluations before the final inner-product check.
pub struct NativeTerminalWhirFinalTableAir {
    pub transcript_bus: TranscriptBus,
    pub final_claim_bus: NativeTerminalWhirFinalClaimBus,
    pub final_context_bus: NativeTerminalWhirFinalContextBus,
    pub final_poly_bus: NativeTerminalWhirFinalPolyBus,
    pub final_weight_bus: NativeTerminalWhirFinalWeightBus,
    pub final_len: usize,
}

impl BaseAirWithPublicValues<F> for NativeTerminalWhirFinalTableAir {}
impl PartitionedBaseAir<F> for NativeTerminalWhirFinalTableAir {}
impl ColumnsAir for NativeTerminalWhirFinalTableAir {}
impl BaseAir<F> for NativeTerminalWhirFinalTableAir {
    fn width(&self) -> usize {
        NativeTerminalWhirFinalTableCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTerminalWhirFinalTableAir {
    fn eval(&self, builder: &mut AB) {
        debug_assert!(self.final_len.is_power_of_two());
        let main = builder.main();
        let local_row = main.row_slice(0).expect("native terminal final table row");
        let next_row = main
            .row_slice(1)
            .expect("native terminal next final table row");
        let local: &NativeTerminalWhirFinalTableCols<AB::Var> = (*local_row).borrow();
        let next: &NativeTerminalWhirFinalTableCols<AB::Var> = (*next_row).borrow();

        for flag in [local.active, local.is_first, local.is_last] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when_first_row().assert_zero(local.index);
        builder.when(local.active).assert_eq(
            local.index + local.remaining,
            AB::Expr::from_usize(self.final_len),
        );
        let remaining_minus_one = AB::Expr::from(local.remaining) - AB::Expr::ONE;
        builder
            .when(local.active * local.is_last)
            .assert_zero(remaining_minus_one.clone());
        builder
            .when(AB::Expr::from(local.active) * (AB::Expr::ONE - AB::Expr::from(local.is_last)))
            .assert_one(remaining_minus_one * local.remaining_inverse);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.is_last);
        let mut transition = builder.when_transition();
        let mut same = transition.when(next.active);
        same.assert_zero(next.is_first);
        same.assert_eq(next.index, local.index + AB::F::ONE);
        same.assert_eq(next.final_poly_tidx, local.final_poly_tidx);
        same.assert_eq(next.final_weight_tidx, local.final_weight_tidx);
        assert_array_eq(&mut same, next.claim, local.claim);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);

        self.final_claim_bus.receive(
            builder,
            NativeTerminalWhirFinalClaimMessage {
                final_poly_tidx: local.final_poly_tidx.into(),
                tidx: local.final_weight_tidx.into(),
                claim: local.claim.map(Into::into),
            },
            local.is_first,
        );
        self.final_context_bus.add_key_with_lookups(
            builder,
            NativeTerminalWhirFinalContextMessage {
                final_weight_tidx: local.final_weight_tidx.into(),
                claim: local.claim.map(Into::into),
            },
            AB::Expr::from(local.is_first) * AB::Expr::TWO,
        );
        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.final_poly_tidx)
                + AB::Expr::from(local.index) * AB::Expr::from_usize(D_EF),
            local.coefficient,
            local.active,
        );
        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.final_weight_tidx)
                + AB::Expr::from(local.index) * AB::Expr::from_usize(D_EF),
            local.weight,
            local.active,
        );
        self.final_poly_bus.send(
            builder,
            NativeTerminalWhirFinalPolyMessage {
                layer: AB::Expr::ZERO,
                index: local.index.into(),
                value: local.coefficient.map(Into::into),
            },
            local.active,
        );
        self.final_weight_bus.send(
            builder,
            NativeTerminalWhirFinalWeightMessage {
                index: local.index.into(),
                value: local.weight.map(Into::into),
            },
            local.active,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeTerminalWhirMobiusCols<T, const ENC_WIDTH: usize> {
    pub active: T,
    pub ordinal: T,
    pub is_first: T,
    pub is_last: T,
    pub layer: T,
    pub left_index: T,
    pub right_index: T,
    pub left: [T; D_EF],
    pub right: [T; D_EF],
    pub next_left: [T; D_EF],
    pub next_right: [T; D_EF],
    pub encoding: [T; ENC_WIDTH],
}

/// In-place Möbius coefficient-to-evaluation butterflies:
/// `next[left] = left`, `next[right] = right + left`.
pub struct NativeTerminalWhirMobiusAir {
    pub final_poly_bus: NativeTerminalWhirFinalPolyBus,
    pub final_log_len: usize,
    pub encoder: Encoder,
    schedule: Vec<(usize, usize, usize)>,
}

impl NativeTerminalWhirMobiusAir {
    #[must_use]
    pub fn new(final_poly_bus: NativeTerminalWhirFinalPolyBus, final_log_len: usize) -> Self {
        assert!(final_log_len > 0);
        let final_len = 1usize << final_log_len;
        let mut schedule = Vec::with_capacity(final_log_len * final_len / 2);
        for layer in 0..final_log_len {
            let step = 1usize << layer;
            let span = step << 1;
            for block in (0..final_len).step_by(span) {
                for offset in 0..step {
                    schedule.push((layer, block + offset, block + step + offset));
                }
            }
        }
        let encoder = Encoder::new(
            schedule.len().max(2),
            NATIVE_TERMINAL_SELECTOR_MAX_FLAG_DEGREE,
            false,
        );
        Self {
            final_poly_bus,
            final_log_len,
            encoder,
            schedule,
        }
    }

    fn eval_impl<AB, const ENC_WIDTH: usize>(&self, builder: &mut AB)
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
    {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("native terminal Mobius row");
        let next_row = main.row_slice(1).expect("native terminal next Mobius row");
        let local: &NativeTerminalWhirMobiusCols<AB::Var, ENC_WIDTH> = (*local_row).borrow();
        let next: &NativeTerminalWhirMobiusCols<AB::Var, ENC_WIDTH> = (*next_row).borrow();

        for flag in [local.active, local.is_first, local.is_last] {
            builder.assert_bool(flag);
        }
        self.encoder.eval(builder, &local.encoding);
        let ordinals = (0..self.schedule.len()).map(|ordinal| (ordinal, ordinal));
        let layers = self
            .schedule
            .iter()
            .enumerate()
            .map(|(ordinal, &(layer, _, _))| (ordinal, layer));
        let left_indices = self
            .schedule
            .iter()
            .enumerate()
            .map(|(ordinal, &(_, left, _))| (ordinal, left));
        let right_indices = self
            .schedule
            .iter()
            .enumerate()
            .map(|(ordinal, &(_, _, right))| (ordinal, right));
        builder.when(local.active).assert_eq(
            local.ordinal,
            self.encoder
                .flag_with_val::<AB>(&local.encoding, &ordinals.collect::<Vec<_>>()),
        );
        builder.when(local.active).assert_eq(
            local.layer,
            self.encoder
                .flag_with_val::<AB>(&local.encoding, &layers.collect::<Vec<_>>()),
        );
        builder.when(local.active).assert_eq(
            local.left_index,
            self.encoder
                .flag_with_val::<AB>(&local.encoding, &left_indices.collect::<Vec<_>>()),
        );
        builder.when(local.active).assert_eq(
            local.right_index,
            self.encoder
                .flag_with_val::<AB>(&local.encoding, &right_indices.collect::<Vec<_>>()),
        );

        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when_first_row().assert_zero(local.ordinal);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.is_last);
        let mut transition = builder.when_transition();
        let mut same = transition.when(next.active);
        same.assert_zero(next.is_first);
        same.assert_eq(next.ordinal, local.ordinal + AB::F::ONE);
        builder
            .when(local.active * local.is_last)
            .assert_eq(local.ordinal, AB::Expr::from_usize(self.schedule.len() - 1));
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);

        assert_array_eq(
            &mut builder.when(local.active),
            local.next_left,
            local.left.map(Into::into),
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.next_right,
            crate::utils::ext_field_add::<AB::Expr>(local.left, local.right),
        );
        self.final_poly_bus.receive(
            builder,
            NativeTerminalWhirFinalPolyMessage {
                layer: local.layer.into(),
                index: local.left_index.into(),
                value: local.left.map(Into::into),
            },
            local.active,
        );
        self.final_poly_bus.receive(
            builder,
            NativeTerminalWhirFinalPolyMessage {
                layer: local.layer.into(),
                index: local.right_index.into(),
                value: local.right.map(Into::into),
            },
            local.active,
        );
        self.final_poly_bus.send(
            builder,
            NativeTerminalWhirFinalPolyMessage {
                layer: AB::Expr::from(local.layer) + AB::Expr::ONE,
                index: local.left_index.into(),
                value: local.next_left.map(Into::into),
            },
            local.active,
        );
        self.final_poly_bus.send(
            builder,
            NativeTerminalWhirFinalPolyMessage {
                layer: AB::Expr::from(local.layer) + AB::Expr::ONE,
                index: local.right_index.into(),
                value: local.next_right.map(Into::into),
            },
            local.active,
        );
    }
}

impl BaseAirWithPublicValues<F> for NativeTerminalWhirMobiusAir {}
impl PartitionedBaseAir<F> for NativeTerminalWhirMobiusAir {}
impl ColumnsAir for NativeTerminalWhirMobiusAir {}
impl BaseAir<F> for NativeTerminalWhirMobiusAir {
    fn width(&self) -> usize {
        match self.encoder.width() {
            1 => NativeTerminalWhirMobiusCols::<F, 1>::width(),
            2 => NativeTerminalWhirMobiusCols::<F, 2>::width(),
            3 => NativeTerminalWhirMobiusCols::<F, 3>::width(),
            4 => NativeTerminalWhirMobiusCols::<F, 4>::width(),
            5 => NativeTerminalWhirMobiusCols::<F, 5>::width(),
            6 => NativeTerminalWhirMobiusCols::<F, 6>::width(),
            7 => NativeTerminalWhirMobiusCols::<F, 7>::width(),
            8 => NativeTerminalWhirMobiusCols::<F, 8>::width(),
            9 => NativeTerminalWhirMobiusCols::<F, 9>::width(),
            10 => NativeTerminalWhirMobiusCols::<F, 10>::width(),
            11 => NativeTerminalWhirMobiusCols::<F, 11>::width(),
            12 => NativeTerminalWhirMobiusCols::<F, 12>::width(),
            13 => NativeTerminalWhirMobiusCols::<F, 13>::width(),
            width => panic!("unsupported native terminal Mobius encoder width: {width}"),
        }
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTerminalWhirMobiusAir {
    fn eval(&self, builder: &mut AB) {
        match self.encoder.width() {
            1 => self.eval_impl::<AB, 1>(builder),
            2 => self.eval_impl::<AB, 2>(builder),
            3 => self.eval_impl::<AB, 3>(builder),
            4 => self.eval_impl::<AB, 4>(builder),
            5 => self.eval_impl::<AB, 5>(builder),
            6 => self.eval_impl::<AB, 6>(builder),
            7 => self.eval_impl::<AB, 7>(builder),
            8 => self.eval_impl::<AB, 8>(builder),
            9 => self.eval_impl::<AB, 9>(builder),
            10 => self.eval_impl::<AB, 10>(builder),
            11 => self.eval_impl::<AB, 11>(builder),
            12 => self.eval_impl::<AB, 12>(builder),
            13 => self.eval_impl::<AB, 13>(builder),
            width => panic!("unsupported native terminal Mobius encoder width: {width}"),
        }
    }
}

pub fn generate_native_terminal_whir_final_table_trace(
    verification: &TerminalWhirVerification<F, EF, Digest>,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    let final_len = verification.final_poly.len();
    if final_len == 0
        || !final_len.is_power_of_two()
        || verification.final_weight_evals.len() != final_len
    {
        return None;
    }
    let final_round = verification.rounds.last()?;
    let final_poly_tidx = final_round
        .sumcheck_rounds
        .last()?
        .transcript_span
        .end
        .operations;
    let final_weight_tidx = verification.final_weight_span.start.operations;
    if verification.final_weight_span.end.operations
        != final_weight_tidx
            .checked_add(final_len.checked_mul(D_EF)?)?
            .checked_add(verification.suffix_point.len().checked_mul(D_EF)?)?
    {
        return None;
    }
    let height = required_height.unwrap_or_else(|| final_len.next_power_of_two());
    if height < final_len {
        return None;
    }
    let width = NativeTerminalWhirFinalTableCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    for index in 0..final_len {
        let row = &mut values[index * width..(index + 1) * width];
        let cols: &mut NativeTerminalWhirFinalTableCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.index = F::from_usize(index);
        cols.is_first = F::from_bool(index == 0);
        cols.is_last = F::from_bool(index + 1 == final_len);
        cols.remaining = F::from_usize(final_len - index);
        cols.remaining_inverse = if index + 1 == final_len {
            F::ZERO
        } else {
            F::from_usize(final_len - index - 1).inverse()
        };
        cols.final_poly_tidx = F::from_usize(final_poly_tidx);
        cols.final_weight_tidx = F::from_usize(final_weight_tidx);
        copy_ext(&mut cols.claim, verification.final_claim);
        copy_ext(&mut cols.coefficient, verification.final_poly[index]);
        copy_ext(&mut cols.weight, verification.final_weight_evals[index]);
    }
    Some(RowMajorMatrix::new(values, width))
}

pub fn generate_native_terminal_whir_mobius_trace(
    air: &NativeTerminalWhirMobiusAir,
    verification: &TerminalWhirVerification<F, EF, Digest>,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    match air.encoder.width() {
        1 => generate_mobius_trace_impl::<1>(air, verification, required_height),
        2 => generate_mobius_trace_impl::<2>(air, verification, required_height),
        3 => generate_mobius_trace_impl::<3>(air, verification, required_height),
        4 => generate_mobius_trace_impl::<4>(air, verification, required_height),
        5 => generate_mobius_trace_impl::<5>(air, verification, required_height),
        6 => generate_mobius_trace_impl::<6>(air, verification, required_height),
        7 => generate_mobius_trace_impl::<7>(air, verification, required_height),
        8 => generate_mobius_trace_impl::<8>(air, verification, required_height),
        9 => generate_mobius_trace_impl::<9>(air, verification, required_height),
        10 => generate_mobius_trace_impl::<10>(air, verification, required_height),
        11 => generate_mobius_trace_impl::<11>(air, verification, required_height),
        12 => generate_mobius_trace_impl::<12>(air, verification, required_height),
        13 => generate_mobius_trace_impl::<13>(air, verification, required_height),
        _ => None,
    }
}

fn generate_mobius_trace_impl<const ENC_WIDTH: usize>(
    air: &NativeTerminalWhirMobiusAir,
    verification: &TerminalWhirVerification<F, EF, Digest>,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    let final_len = 1usize.checked_shl(air.final_log_len as u32)?;
    if verification.final_poly.len() != final_len {
        return None;
    }
    let valid_rows = air.schedule.len();
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeTerminalWhirMobiusCols::<F, ENC_WIDTH>::width();
    let mut trace = F::zero_vec(height * width);
    let mut table = verification.final_poly.clone();
    for (ordinal, &(layer, left_index, right_index)) in air.schedule.iter().enumerate() {
        let left = table[left_index];
        let right = table[right_index];
        let next_left = left;
        let next_right = right + left;
        table[left_index] = next_left;
        table[right_index] = next_right;
        let row = &mut trace[ordinal * width..(ordinal + 1) * width];
        let cols: &mut NativeTerminalWhirMobiusCols<F, ENC_WIDTH> = row.borrow_mut();
        cols.active = F::ONE;
        cols.ordinal = F::from_usize(ordinal);
        cols.is_first = F::from_bool(ordinal == 0);
        cols.is_last = F::from_bool(ordinal + 1 == valid_rows);
        cols.layer = F::from_usize(layer);
        cols.left_index = F::from_usize(left_index);
        cols.right_index = F::from_usize(right_index);
        copy_ext(&mut cols.left, left);
        copy_ext(&mut cols.right, right);
        copy_ext(&mut cols.next_left, next_left);
        copy_ext(&mut cols.next_right, next_right);
        for (target, value) in cols
            .encoding
            .iter_mut()
            .zip(air.encoder.get_flag_pt(ordinal))
        {
            *target = F::from_u32(value);
        }
    }
    Some(RowMajorMatrix::new(trace, width))
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::native_warp::terminal::NATIVE_TERMINAL_MAX_FINAL_LOG_LEN;

    #[test]
    fn mobius_air_supports_every_declared_final_table_size() {
        for final_log_len in 1..=NATIVE_TERMINAL_MAX_FINAL_LOG_LEN {
            let air = NativeTerminalWhirMobiusAir::new(
                NativeTerminalWhirFinalPolyBus::new(0),
                final_log_len,
            );
            let _ = BaseAir::<F>::width(&air);
        }

        let largest = NativeTerminalWhirMobiusAir::new(
            NativeTerminalWhirFinalPolyBus::new(0),
            NATIVE_TERMINAL_MAX_FINAL_LOG_LEN,
        );
        assert_eq!(largest.encoder.width(), 13);
    }
}

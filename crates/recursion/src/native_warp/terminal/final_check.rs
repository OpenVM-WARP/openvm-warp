use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{encoder::Encoder, utils::assert_array_eq, ColumnsAir, SubAir};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, warp_accum::TerminalWhirVerification, BaseAirWithPublicValues,
    PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    native_warp::terminal::{
        NativeTerminalWhirActualWeightBus, NativeTerminalWhirActualWeightMessage,
        NativeTerminalWhirAlphaBus, NativeTerminalWhirAlphaMessage,
        NativeTerminalWhirFinalContextBus, NativeTerminalWhirFinalContextMessage,
        NativeTerminalWhirFinalPolyBus, NativeTerminalWhirFinalPolyMessage,
        NativeTerminalWhirFinalWeightBus, NativeTerminalWhirFinalWeightMessage,
        NativeTerminalWhirPointBus, NativeTerminalWhirPointMessage,
        NATIVE_TERMINAL_SELECTOR_MAX_FLAG_DEGREE,
    },
    utils::{ext_field_add, ext_field_multiply},
};

pub const NATIVE_TERMINAL_MAX_FINAL_LOG_LEN: usize = 9;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeTerminalWhirPointCols<T, const ENC_WIDTH: usize> {
    pub active: T,
    pub coordinate: T,
    pub is_first: T,
    pub is_last: T,
    pub is_alpha: T,
    pub alpha_round: T,
    pub alpha_fold: T,
    pub final_weight_tidx: T,
    pub claim: [T; D_EF],
    pub value: [T; D_EF],
    pub lookup_count: T,
    pub encoding: [T; ENC_WIDTH],
}

/// Canonical table for the complete terminal WHIR point. Its prefix consists
/// of round-fold challenges; its suffix is sampled after the final structured
/// weight table has been observed.
pub struct NativeTerminalWhirPointAir {
    pub transcript_bus: TranscriptBus,
    pub alpha_bus: NativeTerminalWhirAlphaBus,
    pub final_context_bus: NativeTerminalWhirFinalContextBus,
    pub point_bus: NativeTerminalWhirPointBus,
    pub k: usize,
    pub round_count: usize,
    pub final_len: usize,
    pub point_lookup_counts: Vec<usize>,
    pub encoder: Encoder,
}

impl NativeTerminalWhirPointAir {
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        transcript_bus: TranscriptBus,
        alpha_bus: NativeTerminalWhirAlphaBus,
        final_context_bus: NativeTerminalWhirFinalContextBus,
        point_bus: NativeTerminalWhirPointBus,
        k: usize,
        round_count: usize,
        final_len: usize,
        point_lookup_counts: Vec<usize>,
    ) -> Self {
        assert!(k > 0 && round_count > 0 && final_len.is_power_of_two());
        assert_eq!(
            point_lookup_counts.len(),
            k * round_count + final_len.ilog2() as usize
        );
        let encoder = Encoder::new(
            point_lookup_counts.len().max(2),
            NATIVE_TERMINAL_SELECTOR_MAX_FLAG_DEGREE,
            false,
        );
        Self {
            transcript_bus,
            alpha_bus,
            final_context_bus,
            point_bus,
            k,
            round_count,
            final_len,
            point_lookup_counts,
            encoder,
        }
    }

    fn eval_impl<AB, const ENC_WIDTH: usize>(&self, builder: &mut AB)
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
    {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("native terminal point row");
        let next_row = main.row_slice(1).expect("native terminal next point row");
        let local: &NativeTerminalWhirPointCols<AB::Var, ENC_WIDTH> = (*local_row).borrow();
        let next: &NativeTerminalWhirPointCols<AB::Var, ENC_WIDTH> = (*next_row).borrow();
        let point_len = self.point_lookup_counts.len();
        let alpha_len = self.k * self.round_count;

        for flag in [local.active, local.is_first, local.is_last, local.is_alpha] {
            builder.assert_bool(flag);
        }
        self.encoder.eval(builder, &local.encoding);
        let coordinates = (0..point_len).map(|coordinate| (coordinate, coordinate));
        let first = (0..point_len).map(|coordinate| (coordinate, usize::from(coordinate == 0)));
        let last =
            (0..point_len).map(|coordinate| (coordinate, usize::from(coordinate + 1 == point_len)));
        let is_alpha =
            (0..point_len).map(|coordinate| (coordinate, usize::from(coordinate < alpha_len)));
        let alpha_round = (0..point_len).map(|coordinate| {
            (
                coordinate,
                if coordinate < alpha_len {
                    coordinate / self.k
                } else {
                    0
                },
            )
        });
        let alpha_fold = (0..point_len).map(|coordinate| {
            (
                coordinate,
                if coordinate < alpha_len {
                    coordinate % self.k
                } else {
                    0
                },
            )
        });
        let lookup_counts = self.point_lookup_counts.iter().copied().enumerate();
        let decoded = |values: Vec<(usize, usize)>| {
            self.encoder.flag_with_val::<AB>(&local.encoding, &values)
        };
        builder
            .when(local.active)
            .assert_eq(local.coordinate, decoded(coordinates.collect()));
        builder
            .when(local.active)
            .assert_eq(local.is_first, decoded(first.collect()));
        builder
            .when(local.active)
            .assert_eq(local.is_last, decoded(last.collect()));
        builder
            .when(local.active)
            .assert_eq(local.is_alpha, decoded(is_alpha.collect()));
        builder
            .when(local.active)
            .assert_eq(local.alpha_round, decoded(alpha_round.collect()));
        builder
            .when(local.active)
            .assert_eq(local.alpha_fold, decoded(alpha_fold.collect()));
        builder
            .when(local.active)
            .assert_eq(local.lookup_count, decoded(lookup_counts.collect()));

        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.is_last);
        let mut transition = builder.when_transition();
        let mut same = transition.when(next.active);
        same.assert_eq(next.coordinate, local.coordinate + AB::F::ONE);
        same.assert_eq(next.final_weight_tidx, local.final_weight_tidx);
        assert_array_eq(&mut same, next.claim, local.claim);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);

        self.final_context_bus.lookup_key(
            builder,
            NativeTerminalWhirFinalContextMessage {
                final_weight_tidx: local.final_weight_tidx.into(),
                claim: local.claim.map(Into::into),
            },
            local.is_first,
        );
        self.alpha_bus.lookup_key(
            builder,
            NativeTerminalWhirAlphaMessage {
                round: local.alpha_round.into(),
                fold: local.alpha_fold.into(),
                challenge: local.value.map(Into::into),
            },
            local.active * local.is_alpha,
        );
        let suffix_index = AB::Expr::from(local.coordinate) - AB::Expr::from_usize(alpha_len);
        self.transcript_bus.sample_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.final_weight_tidx)
                + AB::Expr::from_usize(self.final_len * D_EF)
                + suffix_index * AB::Expr::from_usize(D_EF),
            local.value,
            AB::Expr::from(local.active) * (AB::Expr::ONE - AB::Expr::from(local.is_alpha)),
        );
        self.point_bus.add_key_with_lookups(
            builder,
            NativeTerminalWhirPointMessage {
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active * local.lookup_count,
        );
    }
}

impl BaseAirWithPublicValues<F> for NativeTerminalWhirPointAir {}
impl PartitionedBaseAir<F> for NativeTerminalWhirPointAir {}
impl ColumnsAir for NativeTerminalWhirPointAir {}
impl BaseAir<F> for NativeTerminalWhirPointAir {
    fn width(&self) -> usize {
        match self.encoder.width() {
            1 => NativeTerminalWhirPointCols::<F, 1>::width(),
            2 => NativeTerminalWhirPointCols::<F, 2>::width(),
            3 => NativeTerminalWhirPointCols::<F, 3>::width(),
            4 => NativeTerminalWhirPointCols::<F, 4>::width(),
            5 => NativeTerminalWhirPointCols::<F, 5>::width(),
            6 => NativeTerminalWhirPointCols::<F, 6>::width(),
            width => panic!("unsupported native terminal point encoder width: {width}"),
        }
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTerminalWhirPointAir {
    fn eval(&self, builder: &mut AB) {
        match self.encoder.width() {
            1 => self.eval_impl::<AB, 1>(builder),
            2 => self.eval_impl::<AB, 2>(builder),
            3 => self.eval_impl::<AB, 3>(builder),
            4 => self.eval_impl::<AB, 4>(builder),
            5 => self.eval_impl::<AB, 5>(builder),
            6 => self.eval_impl::<AB, 6>(builder),
            width => panic!("unsupported native terminal point encoder width: {width}"),
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeTerminalWhirFinalCheckCols<T> {
    pub active: T,
    pub index: T,
    pub is_first: T,
    pub is_last: T,
    pub final_weight_tidx: T,
    pub claim: [T; D_EF],
    pub polynomial_eval: [T; D_EF],
    pub weight: [T; D_EF],
    pub index_bits: [T; NATIVE_TERMINAL_MAX_FINAL_LOG_LEN],
    pub suffix_point: [[T; D_EF]; NATIVE_TERMINAL_MAX_FINAL_LOG_LEN],
    pub eq_prefix: [[T; D_EF]; NATIVE_TERMINAL_MAX_FINAL_LOG_LEN + 1],
    pub inner_before: [T; D_EF],
    pub inner_after: [T; D_EF],
    pub weight_before: [T; D_EF],
    pub weight_after: [T; D_EF],
}

/// Checks both terminal conditions over the small final table:
///
/// * `<coeffs_to_evals(final_poly), final_weight> = final_claim`;
/// * `MLE(final_weight, suffix_point) = actual_weight`.
pub struct NativeTerminalWhirFinalCheckAir {
    pub final_context_bus: NativeTerminalWhirFinalContextBus,
    pub final_poly_bus: NativeTerminalWhirFinalPolyBus,
    pub final_weight_bus: NativeTerminalWhirFinalWeightBus,
    pub point_bus: NativeTerminalWhirPointBus,
    pub actual_weight_bus: NativeTerminalWhirActualWeightBus,
    pub point_prefix_len: usize,
    pub final_log_len: usize,
}

impl BaseAirWithPublicValues<F> for NativeTerminalWhirFinalCheckAir {}
impl PartitionedBaseAir<F> for NativeTerminalWhirFinalCheckAir {}
impl ColumnsAir for NativeTerminalWhirFinalCheckAir {}
impl BaseAir<F> for NativeTerminalWhirFinalCheckAir {
    fn width(&self) -> usize {
        NativeTerminalWhirFinalCheckCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTerminalWhirFinalCheckAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        debug_assert!(
            self.final_log_len > 0 && self.final_log_len <= NATIVE_TERMINAL_MAX_FINAL_LOG_LEN
        );
        let final_len = 1usize << self.final_log_len;
        let main = builder.main();
        let local_row = main.row_slice(0).expect("native terminal final check row");
        let next_row = main
            .row_slice(1)
            .expect("native terminal next final check row");
        let local: &NativeTerminalWhirFinalCheckCols<AB::Var> = (*local_row).borrow();
        let next: &NativeTerminalWhirFinalCheckCols<AB::Var> = (*next_row).borrow();

        for flag in [local.active, local.is_first, local.is_last] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when_first_row().assert_zero(local.index);
        builder
            .when(local.active * local.is_last)
            .assert_eq(local.index, AB::Expr::from_usize(final_len - 1));
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
        same.assert_eq(next.final_weight_tidx, local.final_weight_tidx);
        assert_array_eq(&mut same, next.claim, local.claim);
        assert_array_eq(&mut same, next.inner_before, local.inner_after);
        assert_array_eq(&mut same, next.weight_before, local.weight_after);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);

        self.final_context_bus.lookup_key(
            builder,
            NativeTerminalWhirFinalContextMessage {
                final_weight_tidx: local.final_weight_tidx.into(),
                claim: local.claim.map(Into::into),
            },
            local.is_first,
        );
        self.final_poly_bus.receive(
            builder,
            NativeTerminalWhirFinalPolyMessage {
                layer: AB::Expr::from_usize(self.final_log_len),
                index: local.index.into(),
                value: local.polynomial_eval.map(Into::into),
            },
            local.active,
        );
        self.final_weight_bus.receive(
            builder,
            NativeTerminalWhirFinalWeightMessage {
                index: local.index.into(),
                value: local.weight.map(Into::into),
            },
            local.active,
        );

        let one_ext = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            }
        });
        assert_array_eq(&mut builder.when(local.active), local.eq_prefix[0], one_ext);
        let mut reconstructed_index = AB::Expr::ZERO;
        for coordinate in 0..self.final_log_len {
            builder.assert_bool(local.index_bits[coordinate]);
            reconstructed_index += AB::Expr::from(local.index_bits[coordinate])
                * AB::Expr::from_usize(1usize << coordinate);
            self.point_bus.lookup_key(
                builder,
                NativeTerminalWhirPointMessage {
                    coordinate: AB::Expr::from_usize(self.point_prefix_len + coordinate),
                    value: local.suffix_point[coordinate].map(Into::into),
                },
                local.active,
            );
            let bit: AB::Expr = local.index_bits[coordinate].into();
            let factor = core::array::from_fn(|limb| {
                let point: AB::Expr = local.suffix_point[coordinate][limb].into();
                if limb == 0 {
                    bit.clone() * point.clone()
                        + (AB::Expr::ONE - bit.clone()) * (AB::Expr::ONE - point)
                } else {
                    bit.clone() * point.clone() - (AB::Expr::ONE - bit.clone()) * point
                }
            });
            assert_array_eq(
                &mut builder.when(local.active),
                local.eq_prefix[coordinate + 1],
                ext_field_multiply::<AB::Expr>(local.eq_prefix[coordinate], factor),
            );
        }
        builder
            .when(local.active)
            .assert_eq(local.index, reconstructed_index);
        for coordinate in self.final_log_len..NATIVE_TERMINAL_MAX_FINAL_LOG_LEN {
            let mut when = builder.when(local.active);
            when.assert_zero(local.index_bits[coordinate]);
            for limb in 0..D_EF {
                when.assert_zero(local.suffix_point[coordinate][limb]);
                when.assert_zero(local.eq_prefix[coordinate + 1][limb]);
            }
        }

        let inner_term = ext_field_multiply::<AB::Expr>(local.polynomial_eval, local.weight);
        let weight_term =
            ext_field_multiply::<AB::Expr>(local.weight, local.eq_prefix[self.final_log_len]);
        assert_array_eq(
            &mut builder.when(local.active),
            local.inner_after,
            ext_field_add::<AB::Expr>(local.inner_before, inner_term),
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.weight_after,
            ext_field_add::<AB::Expr>(local.weight_before, weight_term),
        );
        let zero_ext = core::array::from_fn(|_| AB::Expr::ZERO);
        assert_array_eq(
            &mut builder.when(local.is_first),
            local.inner_before,
            zero_ext.clone(),
        );
        assert_array_eq(
            &mut builder.when(local.is_first),
            local.weight_before,
            zero_ext,
        );
        assert_array_eq(
            &mut builder.when(local.is_last),
            local.inner_after,
            local.claim.map(Into::into),
        );
        self.actual_weight_bus.send(
            builder,
            NativeTerminalWhirActualWeightMessage {
                value: local.weight_after.map(Into::into),
            },
            local.is_last,
        );
    }
}

pub fn generate_native_terminal_whir_point_trace(
    air: &NativeTerminalWhirPointAir,
    verification: &TerminalWhirVerification<F, EF, Digest>,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    match air.encoder.width() {
        1 => generate_point_trace_impl::<1>(air, verification, required_height),
        2 => generate_point_trace_impl::<2>(air, verification, required_height),
        3 => generate_point_trace_impl::<3>(air, verification, required_height),
        4 => generate_point_trace_impl::<4>(air, verification, required_height),
        5 => generate_point_trace_impl::<5>(air, verification, required_height),
        6 => generate_point_trace_impl::<6>(air, verification, required_height),
        _ => None,
    }
}

fn generate_point_trace_impl<const ENC_WIDTH: usize>(
    air: &NativeTerminalWhirPointAir,
    verification: &TerminalWhirVerification<F, EF, Digest>,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    let alphas = verification
        .rounds
        .iter()
        .flat_map(|round| round.alphas.iter().copied())
        .collect::<Vec<_>>();
    let mut point = alphas.clone();
    point.extend_from_slice(&verification.suffix_point);
    if alphas.len() != air.k * air.round_count
        || point.len() != air.point_lookup_counts.len()
        || verification.final_weight_evals.len() != air.final_len
    {
        return None;
    }
    let point_len = point.len();
    let height = required_height.unwrap_or_else(|| point_len.next_power_of_two());
    if height < point_len {
        return None;
    }
    let width = NativeTerminalWhirPointCols::<F, ENC_WIDTH>::width();
    let mut values = F::zero_vec(height * width);
    for coordinate in 0..point_len {
        let row = &mut values[coordinate * width..(coordinate + 1) * width];
        let cols: &mut NativeTerminalWhirPointCols<F, ENC_WIDTH> = row.borrow_mut();
        cols.active = F::ONE;
        cols.coordinate = F::from_usize(coordinate);
        cols.is_first = F::from_bool(coordinate == 0);
        cols.is_last = F::from_bool(coordinate + 1 == point_len);
        cols.is_alpha = F::from_bool(coordinate < alphas.len());
        if coordinate < alphas.len() {
            cols.alpha_round = F::from_usize(coordinate / air.k);
            cols.alpha_fold = F::from_usize(coordinate % air.k);
        }
        cols.final_weight_tidx = F::from_usize(verification.final_weight_span.start.operations);
        copy_ext(&mut cols.claim, verification.final_claim);
        copy_ext(&mut cols.value, point[coordinate]);
        cols.lookup_count = F::from_usize(air.point_lookup_counts[coordinate]);
        for (target, value) in cols
            .encoding
            .iter_mut()
            .zip(air.encoder.get_flag_pt(coordinate))
        {
            *target = F::from_u32(value);
        }
    }
    Some(RowMajorMatrix::new(values, width))
}

pub fn generate_native_terminal_whir_final_check_trace(
    air: &NativeTerminalWhirFinalCheckAir,
    verification: &TerminalWhirVerification<F, EF, Digest>,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    let final_len = 1usize.checked_shl(air.final_log_len as u32)?;
    if air.final_log_len > NATIVE_TERMINAL_MAX_FINAL_LOG_LEN
        || verification.final_poly.len() != final_len
        || verification.final_weight_evals.len() != final_len
        || verification.suffix_point.len() != air.final_log_len
    {
        return None;
    }
    let height = required_height.unwrap_or_else(|| final_len.next_power_of_two());
    if height < final_len {
        return None;
    }
    let width = NativeTerminalWhirFinalCheckCols::<F>::width();
    let mut trace = F::zero_vec(height * width);
    let mut polynomial_evals = verification.final_poly.clone();
    coeffs_to_evals_in_place(&mut polynomial_evals);
    let mut inner = EF::ZERO;
    let mut weight_eval = EF::ZERO;
    for index in 0..final_len {
        let row = &mut trace[index * width..(index + 1) * width];
        let cols: &mut NativeTerminalWhirFinalCheckCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.index = F::from_usize(index);
        cols.is_first = F::from_bool(index == 0);
        cols.is_last = F::from_bool(index + 1 == final_len);
        cols.final_weight_tidx = F::from_usize(verification.final_weight_span.start.operations);
        copy_ext(&mut cols.claim, verification.final_claim);
        copy_ext(&mut cols.polynomial_eval, polynomial_evals[index]);
        copy_ext(&mut cols.weight, verification.final_weight_evals[index]);
        for coordinate in 0..air.final_log_len {
            cols.index_bits[coordinate] = F::from_bool((index >> coordinate) & 1 == 1);
            copy_ext(
                &mut cols.suffix_point[coordinate],
                verification.suffix_point[coordinate],
            );
        }
        let mut eq = EF::ONE;
        copy_ext(&mut cols.eq_prefix[0], eq);
        for coordinate in 0..air.final_log_len {
            let point = verification.suffix_point[coordinate];
            eq *= if (index >> coordinate) & 1 == 1 {
                point
            } else {
                EF::ONE - point
            };
            copy_ext(&mut cols.eq_prefix[coordinate + 1], eq);
        }
        copy_ext(&mut cols.inner_before, inner);
        copy_ext(&mut cols.weight_before, weight_eval);
        inner += polynomial_evals[index] * verification.final_weight_evals[index];
        weight_eval += verification.final_weight_evals[index] * eq;
        copy_ext(&mut cols.inner_after, inner);
        copy_ext(&mut cols.weight_after, weight_eval);
    }
    if inner != verification.final_inner_product || weight_eval != verification.actual_weight {
        return None;
    }
    Some(RowMajorMatrix::new(trace, width))
}

fn coeffs_to_evals_in_place(values: &mut [EF]) {
    for bit in 0..values.len().ilog2() as usize {
        let step = 1usize << bit;
        let span = step << 1;
        for start in (0..values.len()).step_by(span) {
            for offset in 0..step {
                let left = start + offset;
                let right = left + step;
                values[right] += values[left];
            }
        }
    }
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

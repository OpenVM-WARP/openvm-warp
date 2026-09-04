use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{encoder::Encoder, utils::assert_array_eq, ColumnsAir, SubAir};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, warp_accum::TerminalWhirVerification, BaseAirWithPublicValues,
    PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing, PrimeField32,
    TwoAdicField,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    native_warp::terminal::{
        NativeTerminalAccumulatorValueBus, NativeTerminalAccumulatorValueMessage,
        NativeTerminalRsAdjointClaimBus, NativeTerminalRsAdjointClaimMessage,
        NativeTerminalRsAdjointQBus, NativeTerminalRsAdjointQMessage,
        NativeTerminalRsAdjointRoundBus, NativeTerminalRsAdjointRoundMessage,
        NativeTerminalRsAdjointValueBus, NativeTerminalRsAdjointValueMessage,
        NativeTerminalRsAdjointYBus, NativeTerminalRsAdjointYMessage, NativeTerminalWhirPointBus,
        NativeTerminalWhirPointMessage, NATIVE_TERMINAL_SELECTOR_MAX_FLAG_DEGREE,
    },
    utils::{ext_field_add, ext_field_multiply, ext_field_multiply_scalar, ext_field_subtract},
};

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeTerminalRsAdjointQCols<T> {
    pub active: T,
    pub round: T,
    pub is_first: T,
    pub is_last: T,
    pub is_column: T,
    pub challenge: [T; D_EF],
    pub whir_point: [T; D_EF],
    pub alpha: [T; D_EF],
    pub factor: [T; D_EF],
    pub prefix_before: [T; D_EF],
    pub prefix_after: [T; D_EF],
}

pub struct NativeTerminalRsAdjointQAir {
    pub round_bus: NativeTerminalRsAdjointRoundBus,
    pub accumulator_value_bus: NativeTerminalAccumulatorValueBus,
    pub point_bus: NativeTerminalWhirPointBus,
    pub q_bus: NativeTerminalRsAdjointQBus,
    pub round_count: usize,
    pub initial_folding_factor: usize,
}

impl BaseAirWithPublicValues<F> for NativeTerminalRsAdjointQAir {}
impl PartitionedBaseAir<F> for NativeTerminalRsAdjointQAir {}
impl ColumnsAir for NativeTerminalRsAdjointQAir {}
impl BaseAir<F> for NativeTerminalRsAdjointQAir {
    fn width(&self) -> usize {
        NativeTerminalRsAdjointQCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTerminalRsAdjointQAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("native terminal RS-adjoint q row");
        let next_row = main
            .row_slice(1)
            .expect("native terminal next RS-adjoint q row");
        let local: &NativeTerminalRsAdjointQCols<AB::Var> = (*local_row).borrow();
        let next: &NativeTerminalRsAdjointQCols<AB::Var> = (*next_row).borrow();

        for flag in [local.active, local.is_first, local.is_last, local.is_column] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        if self.round_count == 0 || self.initial_folding_factor > self.round_count {
            builder.when(local.active).assert_zero(local.active);
            return;
        }
        builder.when_first_row().assert_one(local.is_first);
        if self.initial_folding_factor == 0 {
            builder.when(local.active).assert_zero(local.is_column);
        } else {
            builder.when_first_row().assert_one(local.is_column);
        }
        builder.when_first_row().assert_zero(local.round);
        builder.when(local.active * local.is_last).assert_eq(
            local.round,
            AB::Expr::from_usize(self.round_count.saturating_sub(1)),
        );
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.is_last);
        let mut transition = builder.when_transition();
        let mut same = transition.when(next.active);
        same.assert_zero(next.is_first);
        same.assert_eq(next.round, local.round + AB::F::ONE);
        assert_array_eq(&mut same, next.prefix_before, local.prefix_after);
        builder
            .when_transition()
            .assert_bool(local.is_column - next.is_column);
        if let Some(last_column_round) = self.initial_folding_factor.checked_sub(1) {
            builder
                .when_transition()
                .when(local.is_column - next.is_column)
                .assert_eq(local.round, AB::Expr::from_usize(last_column_round));
        }
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);

        let one = one_ext::<AB>();
        assert_array_eq(
            &mut builder.when(local.is_first),
            local.prefix_before,
            one.clone(),
        );
        let row_factor = ext_field_add::<AB::Expr>(
            ext_field_multiply::<AB::Expr>(
                ext_field_subtract::<AB::Expr>(one.clone(), local.alpha),
                ext_field_subtract::<AB::Expr>(one.clone(), local.challenge),
            ),
            ext_field_multiply::<AB::Expr>(local.alpha, local.challenge),
        );
        let column_factor = ext_field_add::<AB::Expr>(
            ext_field_multiply::<AB::Expr>(
                ext_field_subtract::<AB::Expr>(one.clone(), local.alpha),
                ext_field_subtract::<AB::Expr>(one.clone(), local.whir_point),
            ),
            ext_field_multiply::<AB::Expr>(
                local.alpha,
                ext_field_subtract::<AB::Expr>(
                    ext_field_add::<AB::Expr>(local.whir_point, local.whir_point),
                    one.clone(),
                ),
            ),
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.factor,
            core::array::from_fn(|limb| {
                local.is_column * column_factor[limb].clone()
                    + (AB::Expr::ONE - local.is_column) * row_factor[limb].clone()
            }),
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.prefix_after,
            ext_field_multiply::<AB::Expr>(local.prefix_before, local.factor),
        );

        self.round_bus.lookup_key(
            builder,
            NativeTerminalRsAdjointRoundMessage {
                round: local.round - AB::Expr::from_usize(self.initial_folding_factor),
                challenge: local.challenge.map(Into::into),
            },
            local.active * (AB::Expr::ONE - local.is_column),
        );
        let accumulator_coordinate = if self.initial_folding_factor == 0 {
            AB::Expr::from(local.round)
        } else {
            local.is_column * (AB::Expr::from_usize(self.initial_folding_factor - 1) - local.round)
                + (AB::Expr::ONE - local.is_column) * local.round
        };
        self.accumulator_value_bus.lookup_key(
            builder,
            NativeTerminalAccumulatorValueMessage {
                section: AB::Expr::ZERO,
                coordinate: accumulator_coordinate,
                value: local.alpha.map(Into::into),
            },
            local.active,
        );
        self.point_bus.lookup_key(
            builder,
            NativeTerminalWhirPointMessage {
                coordinate: local.round.into(),
                value: local.whir_point.map(Into::into),
            },
            local.active * local.is_column,
        );
        self.q_bus.send(
            builder,
            NativeTerminalRsAdjointQMessage {
                value: local.prefix_after.map(Into::into),
            },
            local.is_last,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeTerminalRsAdjointYCols<T, const ENC_WIDTH: usize> {
    pub active: T,
    pub ordinal: T,
    pub message_bit: T,
    pub round: T,
    pub is_first_in_bit: T,
    pub is_last_in_bit: T,
    pub is_first: T,
    pub is_last: T,
    pub root: T,
    pub challenge: [T; D_EF],
    pub factor: [T; D_EF],
    pub prefix_before: [T; D_EF],
    pub prefix_after: [T; D_EF],
    pub encoding: [T; ENC_WIDTH],
}

pub struct NativeTerminalRsAdjointYAir {
    pub round_bus: NativeTerminalRsAdjointRoundBus,
    pub y_bus: NativeTerminalRsAdjointYBus,
    pub log_message_len: usize,
    pub log_codeword_len: usize,
    pub encoder: Encoder,
    schedule: Vec<(usize, usize, F)>,
}

impl NativeTerminalRsAdjointYAir {
    #[must_use]
    pub fn new(
        round_bus: NativeTerminalRsAdjointRoundBus,
        y_bus: NativeTerminalRsAdjointYBus,
        log_message_len: usize,
        log_codeword_len: usize,
    ) -> Self {
        assert!(log_message_len > 0 && log_codeword_len >= log_message_len);
        let omega = F::two_adic_generator(log_codeword_len);
        let mut schedule = Vec::with_capacity(log_message_len * log_codeword_len);
        for message_bit in 0..log_message_len {
            for round in 0..log_codeword_len {
                let exponent_power = log_codeword_len - 1 - round + message_bit;
                let root = if exponent_power >= log_codeword_len {
                    F::ONE
                } else {
                    omega.exp_power_of_2(exponent_power)
                };
                schedule.push((message_bit, round, root));
            }
        }
        let encoder = Encoder::new(
            schedule.len().max(2),
            NATIVE_TERMINAL_SELECTOR_MAX_FLAG_DEGREE,
            false,
        );
        Self {
            round_bus,
            y_bus,
            log_message_len,
            log_codeword_len,
            encoder,
            schedule,
        }
    }

    fn eval_impl<AB, const ENC_WIDTH: usize>(&self, builder: &mut AB)
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
        <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
    {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("native terminal RS-adjoint y row");
        let next_row = main
            .row_slice(1)
            .expect("native terminal next RS-adjoint y row");
        let local: &NativeTerminalRsAdjointYCols<AB::Var, ENC_WIDTH> = (*local_row).borrow();
        let next: &NativeTerminalRsAdjointYCols<AB::Var, ENC_WIDTH> = (*next_row).borrow();

        for flag in [
            local.active,
            local.is_first_in_bit,
            local.is_last_in_bit,
            local.is_first,
            local.is_last,
        ] {
            builder.assert_bool(flag);
        }
        self.encoder.eval(builder, &local.encoding);
        let ordinals = (0..self.schedule.len()).map(|ordinal| (ordinal, ordinal));
        let message_bits = self
            .schedule
            .iter()
            .enumerate()
            .map(|(ordinal, &(message_bit, _, _))| (ordinal, message_bit));
        let rounds = self
            .schedule
            .iter()
            .enumerate()
            .map(|(ordinal, &(_, round, _))| (ordinal, round));
        let roots = self
            .schedule
            .iter()
            .enumerate()
            .map(|(ordinal, &(_, _, root))| (ordinal, root.as_canonical_u32() as usize));
        let first_in_bit = self
            .schedule
            .iter()
            .enumerate()
            .map(|(ordinal, &(_, round, _))| (ordinal, usize::from(round == 0)));
        let last_in_bit = self
            .schedule
            .iter()
            .enumerate()
            .map(|(ordinal, &(_, round, _))| {
                (ordinal, usize::from(round + 1 == self.log_codeword_len))
            });
        let first = (0..self.schedule.len()).map(|ordinal| (ordinal, usize::from(ordinal == 0)));
        let last = (0..self.schedule.len())
            .map(|ordinal| (ordinal, usize::from(ordinal + 1 == self.schedule.len())));
        let decoded = |values: Vec<(usize, usize)>| {
            self.encoder.flag_with_val::<AB>(&local.encoding, &values)
        };
        builder
            .when(local.active)
            .assert_eq(local.ordinal, decoded(ordinals.collect()));
        builder
            .when(local.active)
            .assert_eq(local.message_bit, decoded(message_bits.collect()));
        builder
            .when(local.active)
            .assert_eq(local.round, decoded(rounds.collect()));
        builder
            .when(local.active)
            .assert_eq(local.root, decoded(roots.collect()));
        builder
            .when(local.active)
            .assert_eq(local.is_first_in_bit, decoded(first_in_bit.collect()));
        builder
            .when(local.active)
            .assert_eq(local.is_last_in_bit, decoded(last_in_bit.collect()));
        builder
            .when(local.active)
            .assert_eq(local.is_first, decoded(first.collect()));
        builder
            .when(local.active)
            .assert_eq(local.is_last, decoded(last.collect()));

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
        same.assert_eq(next.ordinal, local.ordinal + AB::F::ONE);
        let same_bit = next.active * (AB::Expr::ONE - AB::Expr::from(next.is_first_in_bit));
        assert_array_eq(
            &mut builder.when_transition().when(same_bit),
            next.prefix_before,
            local.prefix_after,
        );
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);

        assert_array_eq(
            &mut builder.when(local.is_first_in_bit),
            local.prefix_before,
            one_ext::<AB>(),
        );
        let expected_factor = ext_field_add::<AB::Expr>(
            ext_field_subtract::<AB::Expr>(one_ext::<AB>(), local.challenge),
            ext_field_multiply_scalar::<AB::Expr>(local.challenge, local.root),
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.factor,
            expected_factor,
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.prefix_after,
            ext_field_multiply::<AB::Expr>(local.prefix_before, local.factor),
        );
        self.round_bus.lookup_key(
            builder,
            NativeTerminalRsAdjointRoundMessage {
                round: local.round.into(),
                challenge: local.challenge.map(Into::into),
            },
            local.active,
        );
        self.y_bus.send(
            builder,
            NativeTerminalRsAdjointYMessage {
                message_bit: local.message_bit.into(),
                value: local.prefix_after.map(Into::into),
            },
            local.active * local.is_last_in_bit,
        );
    }
}

impl BaseAirWithPublicValues<F> for NativeTerminalRsAdjointYAir {}
impl PartitionedBaseAir<F> for NativeTerminalRsAdjointYAir {}
impl ColumnsAir for NativeTerminalRsAdjointYAir {}
impl BaseAir<F> for NativeTerminalRsAdjointYAir {
    fn width(&self) -> usize {
        match self.encoder.width() {
            1 => NativeTerminalRsAdjointYCols::<F, 1>::width(),
            2 => NativeTerminalRsAdjointYCols::<F, 2>::width(),
            3 => NativeTerminalRsAdjointYCols::<F, 3>::width(),
            4 => NativeTerminalRsAdjointYCols::<F, 4>::width(),
            5 => NativeTerminalRsAdjointYCols::<F, 5>::width(),
            6 => NativeTerminalRsAdjointYCols::<F, 6>::width(),
            7 => NativeTerminalRsAdjointYCols::<F, 7>::width(),
            8 => NativeTerminalRsAdjointYCols::<F, 8>::width(),
            9 => NativeTerminalRsAdjointYCols::<F, 9>::width(),
            10 => NativeTerminalRsAdjointYCols::<F, 10>::width(),
            11 => NativeTerminalRsAdjointYCols::<F, 11>::width(),
            width => panic!("unsupported terminal adjoint y encoder width: {width}"),
        }
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTerminalRsAdjointYAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
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
            width => panic!("unsupported terminal adjoint y encoder width: {width}"),
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct NativeTerminalRsAdjointSelectorCols<T> {
    pub active: T,
    pub message_bit: T,
    pub is_first: T,
    pub is_last: T,
    pub whir_point: [T; D_EF],
    pub y_power: [T; D_EF],
    pub factor: [T; D_EF],
    pub selector_before: [T; D_EF],
    pub selector_after: [T; D_EF],
    pub q: [T; D_EF],
    pub claimed_value: [T; D_EF],
    pub final_claim: [T; D_EF],
}

pub struct NativeTerminalRsAdjointSelectorAir {
    pub point_bus: NativeTerminalWhirPointBus,
    pub q_bus: NativeTerminalRsAdjointQBus,
    pub y_bus: NativeTerminalRsAdjointYBus,
    pub claim_bus: NativeTerminalRsAdjointClaimBus,
    pub value_bus: NativeTerminalRsAdjointValueBus,
    pub log_message_len: usize,
    pub point_coordinate_offset: usize,
}

impl BaseAirWithPublicValues<F> for NativeTerminalRsAdjointSelectorAir {}
impl PartitionedBaseAir<F> for NativeTerminalRsAdjointSelectorAir {}
impl ColumnsAir for NativeTerminalRsAdjointSelectorAir {}
impl BaseAir<F> for NativeTerminalRsAdjointSelectorAir {
    fn width(&self) -> usize {
        NativeTerminalRsAdjointSelectorCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTerminalRsAdjointSelectorAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main
            .row_slice(0)
            .expect("native terminal RS-adjoint selector row");
        let next_row = main
            .row_slice(1)
            .expect("native terminal next RS-adjoint selector row");
        let local: &NativeTerminalRsAdjointSelectorCols<AB::Var> = (*local_row).borrow();
        let next: &NativeTerminalRsAdjointSelectorCols<AB::Var> = (*next_row).borrow();

        for flag in [local.active, local.is_first, local.is_last] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when_first_row().assert_zero(local.message_bit);
        builder.when(local.active * local.is_last).assert_eq(
            local.message_bit,
            AB::Expr::from_usize(self.log_message_len - 1),
        );
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.is_last);
        let mut transition = builder.when_transition();
        let mut same = transition.when(next.active);
        same.assert_zero(next.is_first);
        same.assert_eq(next.message_bit, local.message_bit + AB::F::ONE);
        assert_array_eq(&mut same, next.selector_before, local.selector_after);
        assert_array_eq(&mut same, next.q, local.q);
        assert_array_eq(&mut same, next.claimed_value, local.claimed_value);
        assert_array_eq(&mut same, next.final_claim, local.final_claim);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);

        assert_array_eq(
            &mut builder.when(local.is_first),
            local.selector_before,
            one_ext::<AB>(),
        );
        let point_twice_minus_one = ext_field_subtract::<AB::Expr>(
            ext_field_add::<AB::Expr>(local.whir_point, local.whir_point),
            one_ext::<AB>(),
        );
        let factor = ext_field_add::<AB::Expr>(
            ext_field_subtract::<AB::Expr>(one_ext::<AB>(), local.whir_point),
            ext_field_multiply::<AB::Expr>(point_twice_minus_one, local.y_power),
        );
        assert_array_eq(&mut builder.when(local.active), local.factor, factor);
        assert_array_eq(
            &mut builder.when(local.active),
            local.selector_after,
            ext_field_multiply::<AB::Expr>(local.selector_before, local.factor),
        );
        assert_array_eq(
            &mut builder.when(local.is_last),
            local.final_claim,
            ext_field_multiply::<AB::Expr>(local.q, local.selector_after),
        );

        self.point_bus.lookup_key(
            builder,
            NativeTerminalWhirPointMessage {
                coordinate: local.message_bit + AB::Expr::from_usize(self.point_coordinate_offset),
                value: local.whir_point.map(Into::into),
            },
            local.active,
        );
        self.y_bus.receive(
            builder,
            NativeTerminalRsAdjointYMessage {
                message_bit: local.message_bit.into(),
                value: local.y_power.map(Into::into),
            },
            local.active,
        );
        self.q_bus.receive(
            builder,
            NativeTerminalRsAdjointQMessage {
                value: local.q.map(Into::into),
            },
            local.is_first,
        );
        self.claim_bus.receive(
            builder,
            NativeTerminalRsAdjointClaimMessage {
                claimed_value: local.claimed_value.map(Into::into),
                final_claim: local.final_claim.map(Into::into),
            },
            local.is_first,
        );
        self.value_bus.send(
            builder,
            NativeTerminalRsAdjointValueMessage {
                value: local.claimed_value.map(Into::into),
            },
            local.is_last,
        );
    }
}

pub fn generate_native_terminal_rs_adjoint_q_trace(
    verification: &TerminalWhirVerification<F, EF, Digest>,
    alpha: &[EF],
    initial_folding_factor: usize,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    let adjoint = &verification.accumulator_adjoint;
    let whir_point = verification
        .rounds
        .iter()
        .flat_map(|round| round.alphas.iter().copied())
        .chain(verification.suffix_point.iter().copied())
        .collect::<Vec<_>>();
    if initial_folding_factor > whir_point.len()
        || whir_point.len() > alpha.len()
        || adjoint.point.len() + initial_folding_factor != alpha.len()
    {
        return None;
    }
    let valid_rows = alpha.len();
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeTerminalRsAdjointQCols::<F>::width();
    let mut trace = F::zero_vec(height * width);
    let mut prefix = EF::ONE;
    for round in 0..valid_rows {
        let is_column = round < initial_folding_factor;
        let (alpha_value, challenge, point, factor) = if is_column {
            let alpha_value = alpha[initial_folding_factor - 1 - round];
            let point = whir_point[round];
            let factor = (EF::ONE - alpha_value) * (EF::ONE - point)
                + alpha_value * (point.double() - EF::ONE);
            (alpha_value, EF::ZERO, point, factor)
        } else {
            let alpha_value = alpha[round];
            let challenge = adjoint.point[round - initial_folding_factor];
            let factor = (EF::ONE - alpha_value) * (EF::ONE - challenge) + alpha_value * challenge;
            (alpha_value, challenge, EF::ZERO, factor)
        };
        let before = prefix;
        prefix *= factor;
        let row = &mut trace[round * width..(round + 1) * width];
        let cols: &mut NativeTerminalRsAdjointQCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.round = F::from_usize(round);
        cols.is_first = F::from_bool(round == 0);
        cols.is_last = F::from_bool(round + 1 == valid_rows);
        cols.is_column = F::from_bool(is_column);
        copy_ext(&mut cols.challenge, challenge);
        copy_ext(&mut cols.whir_point, point);
        copy_ext(&mut cols.alpha, alpha_value);
        copy_ext(&mut cols.factor, factor);
        copy_ext(&mut cols.prefix_before, before);
        copy_ext(&mut cols.prefix_after, prefix);
    }
    Some(RowMajorMatrix::new(trace, width))
}

pub fn generate_native_terminal_rs_adjoint_y_trace(
    air: &NativeTerminalRsAdjointYAir,
    verification: &TerminalWhirVerification<F, EF, Digest>,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    match air.encoder.width() {
        1 => generate_y_trace_impl::<1>(air, verification, required_height),
        2 => generate_y_trace_impl::<2>(air, verification, required_height),
        3 => generate_y_trace_impl::<3>(air, verification, required_height),
        4 => generate_y_trace_impl::<4>(air, verification, required_height),
        5 => generate_y_trace_impl::<5>(air, verification, required_height),
        6 => generate_y_trace_impl::<6>(air, verification, required_height),
        7 => generate_y_trace_impl::<7>(air, verification, required_height),
        8 => generate_y_trace_impl::<8>(air, verification, required_height),
        9 => generate_y_trace_impl::<9>(air, verification, required_height),
        10 => generate_y_trace_impl::<10>(air, verification, required_height),
        11 => generate_y_trace_impl::<11>(air, verification, required_height),
        _ => None,
    }
}

fn generate_y_trace_impl<const ENC_WIDTH: usize>(
    air: &NativeTerminalRsAdjointYAir,
    verification: &TerminalWhirVerification<F, EF, Digest>,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    let point = &verification.accumulator_adjoint.point;
    if point.len() != air.log_codeword_len {
        return None;
    }
    let valid_rows = air.schedule.len();
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeTerminalRsAdjointYCols::<F, ENC_WIDTH>::width();
    let mut trace = F::zero_vec(height * width);
    let mut prefix = EF::ONE;
    for (ordinal, &(message_bit, round, root)) in air.schedule.iter().enumerate() {
        if round == 0 {
            prefix = EF::ONE;
        }
        let before = prefix;
        let factor = (EF::ONE - point[round]) + point[round] * EF::from(root);
        prefix *= factor;
        let row = &mut trace[ordinal * width..(ordinal + 1) * width];
        let cols: &mut NativeTerminalRsAdjointYCols<F, ENC_WIDTH> = row.borrow_mut();
        cols.active = F::ONE;
        cols.ordinal = F::from_usize(ordinal);
        cols.message_bit = F::from_usize(message_bit);
        cols.round = F::from_usize(round);
        cols.is_first_in_bit = F::from_bool(round == 0);
        cols.is_last_in_bit = F::from_bool(round + 1 == air.log_codeword_len);
        cols.is_first = F::from_bool(ordinal == 0);
        cols.is_last = F::from_bool(ordinal + 1 == valid_rows);
        cols.root = root;
        copy_ext(&mut cols.challenge, point[round]);
        copy_ext(&mut cols.factor, factor);
        copy_ext(&mut cols.prefix_before, before);
        copy_ext(&mut cols.prefix_after, prefix);
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

pub fn generate_native_terminal_rs_adjoint_selector_trace(
    verification: &TerminalWhirVerification<F, EF, Digest>,
    alpha: &[EF],
    initial_folding_factor: usize,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    let full_point = verification
        .rounds
        .iter()
        .flat_map(|round| round.alphas.iter().copied())
        .chain(verification.suffix_point.iter().copied())
        .collect::<Vec<_>>();
    let adjoint = &verification.accumulator_adjoint;
    if initial_folding_factor > full_point.len()
        || initial_folding_factor > alpha.len()
        || alpha.len() != adjoint.point.len() + initial_folding_factor
    {
        return None;
    }
    let (column_point, point) = full_point.split_at(initial_folding_factor);
    if point.is_empty() || full_point.len() > alpha.len() {
        return None;
    }
    let log_codeword_len = adjoint.point.len();
    let omega = F::two_adic_generator(log_codeword_len);
    let height = required_height.unwrap_or_else(|| point.len().next_power_of_two());
    if height < point.len() {
        return None;
    }
    let width = NativeTerminalRsAdjointSelectorCols::<F>::width();
    let mut trace = F::zero_vec(height * width);
    let (column_alpha, row_alpha) = alpha.split_at(initial_folding_factor);
    let column_factor = column_alpha
        .iter()
        .rev()
        .zip(column_point)
        .map(|(&alpha, &coordinate)| {
            (EF::ONE - alpha) * (EF::ONE - coordinate) + alpha * (coordinate.double() - EF::ONE)
        })
        .product::<EF>();
    let q = column_factor
        * row_alpha
            .iter()
            .zip(&adjoint.point)
            .map(|(&alpha, &challenge)| {
                (EF::ONE - alpha) * (EF::ONE - challenge) + alpha * challenge
            })
            .product::<EF>();
    let mut y_powers = Vec::with_capacity(point.len());
    for message_bit in 0..point.len() {
        let mut y = EF::ONE;
        for round in 0..log_codeword_len {
            let exponent_power = log_codeword_len - 1 - round + message_bit;
            let root = if exponent_power >= log_codeword_len {
                F::ONE
            } else {
                omega.exp_power_of_2(exponent_power)
            };
            y *= (EF::ONE - adjoint.point[round]) + adjoint.point[round] * EF::from(root);
        }
        y_powers.push(y);
    }
    let mut prefix = EF::ONE;
    for message_bit in 0..point.len() {
        let factor = (EF::ONE - point[message_bit])
            + (point[message_bit].double() - EF::ONE) * y_powers[message_bit];
        let before = prefix;
        prefix *= factor;
        let row = &mut trace[message_bit * width..(message_bit + 1) * width];
        let cols: &mut NativeTerminalRsAdjointSelectorCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.message_bit = F::from_usize(message_bit);
        cols.is_first = F::from_bool(message_bit == 0);
        cols.is_last = F::from_bool(message_bit + 1 == point.len());
        copy_ext(&mut cols.whir_point, point[message_bit]);
        copy_ext(&mut cols.y_power, y_powers[message_bit]);
        copy_ext(&mut cols.factor, factor);
        copy_ext(&mut cols.selector_before, before);
        copy_ext(&mut cols.selector_after, prefix);
        copy_ext(&mut cols.q, q);
        copy_ext(&mut cols.claimed_value, adjoint.claimed_value);
        copy_ext(&mut cols.final_claim, adjoint.final_claim);
    }
    if q * prefix != adjoint.final_claim {
        return None;
    }
    Some(RowMajorMatrix::new(trace, width))
}

fn one_ext<AB: AirBuilder<F = F>>() -> [AB::Expr; D_EF] {
    core::array::from_fn(|limb| {
        if limb == 0 {
            AB::Expr::ONE
        } else {
            AB::Expr::ZERO
        }
    })
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use openvm_stark_backend::{
        air_builders::debug::{check_constraints, DebugConstraintBuilder},
        transcript::TranscriptCheckpoint,
        warp_accum::{
            terminal_whir::{TerminalWhirTranscriptPhase, TerminalWhirTranscriptPhaseSpan},
            RsAdjointEvalVerification, TerminalConstrainedCodeLayout, TerminalWhirVerification,
            WhirInitialRsWarpCode,
        },
        BaseAirWithPublicValues, PartitionedBaseAir, StarkProtocolConfig, SystemParams,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config as SC;
    use p3_air::Air;
    use p3_matrix::Matrix;

    use super::*;

    fn check_air<A>(air: &A, trace: &RowMajorMatrix<F>)
    where
        A: for<'a> Air<DebugConstraintBuilder<'a, SC>>
            + BaseAir<F>
            + BaseAirWithPublicValues<F>
            + PartitionedBaseAir<F>,
    {
        check_constraints::<_, SC>(
            air,
            core::any::type_name::<A>(),
            &None,
            &[trace.as_view()],
            &[],
        );
    }

    fn checkpoint(operations: usize) -> TranscriptCheckpoint {
        TranscriptCheckpoint {
            operations,
            events: 0,
            permutations: 0,
        }
    }

    fn empty_span(phase: TerminalWhirTranscriptPhase) -> TerminalWhirTranscriptPhaseSpan {
        TerminalWhirTranscriptPhaseSpan {
            phase,
            start: checkpoint(0),
            end: checkpoint(0),
        }
    }

    fn coefficient_fixture() -> (
        WhirInitialRsWarpCode<<SC as StarkProtocolConfig>::Hasher>,
        Vec<EF>,
        Vec<EF>,
        TerminalWhirVerification<F, EF, Digest>,
    ) {
        let config = SC::default_from_params(SystemParams::new_for_testing(8));
        let code = WhirInitialRsWarpCode::try_new_coefficient_subgroup(
            config.hasher().clone(),
            3,
            1,
            0,
            2,
        )
        .expect("coefficient-subgroup adjoint fixture");
        let alpha = (0..code.log_codeword_len())
            .map(|index| EF::from_usize(11 + index * 7))
            .collect::<Vec<_>>();
        let whir_point = (0..code.log_message_len())
            .map(|index| EF::from_usize(31 + index * 13))
            .collect::<Vec<_>>();
        let adjoint_point = (0..code.log_codeword_len())
            .map(|index| EF::from_usize(71 + index * 17))
            .collect::<Vec<_>>();
        let q = alpha
            .iter()
            .zip(&adjoint_point)
            .map(|(&alpha, &challenge)| {
                (EF::ONE - alpha) * (EF::ONE - challenge) + alpha * challenge
            })
            .product::<EF>();
        let omega = F::two_adic_generator(code.log_codeword_len());
        let selector = whir_point
            .iter()
            .enumerate()
            .map(|(message_bit, &coordinate)| {
                let y = adjoint_point
                    .iter()
                    .enumerate()
                    .map(|(round, &challenge)| {
                        let exponent_power = code.log_codeword_len() - 1 - round + message_bit;
                        let root = if exponent_power >= code.log_codeword_len() {
                            F::ONE
                        } else {
                            omega.exp_power_of_2(exponent_power)
                        };
                        (EF::ONE - challenge) + challenge * EF::from(root)
                    })
                    .product::<EF>();
                (EF::ONE - coordinate) + (coordinate.double() - EF::ONE) * y
            })
            .product::<EF>();
        let claimed_value = code.accumulator_weight_eval_at(&alpha, &whir_point);
        let final_claim = q * selector;
        let verification = TerminalWhirVerification {
            transcript_start: checkpoint(0),
            descriptor_span: empty_span(TerminalWhirTranscriptPhase::Descriptor),
            batching_challenge_span: empty_span(TerminalWhirTranscriptPhase::BatchingChallenge),
            transcript_end: checkpoint(0),
            root: [F::ZERO; 8],
            batching_challenge: EF::ZERO,
            initial_claim: EF::ZERO,
            rounds: Vec::new(),
            final_poly: Vec::new(),
            final_weight_evals: Vec::new(),
            final_weight_span: empty_span(TerminalWhirTranscriptPhase::FinalWeight),
            suffix_point: whir_point.clone(),
            accumulator_adjoint: RsAdjointEvalVerification {
                transcript_start: checkpoint(0),
                transcript_end: checkpoint(0),
                claimed_value,
                degree: (code.log_message_len() + 1) as u32,
                rounds: Vec::new(),
                point: adjoint_point,
                final_claim,
                expected_final: final_claim,
            },
            expected_weight: EF::ZERO,
            actual_weight: EF::ZERO,
            final_inner_product: EF::ZERO,
            final_claim: EF::ZERO,
        };
        (code, alpha, whir_point, verification)
    }

    #[test]
    fn adjoint_y_air_supports_largest_baby_bear_domain() {
        let log_codeword_len = F::TWO_ADICITY;
        let air = NativeTerminalRsAdjointYAir::new(
            NativeTerminalRsAdjointRoundBus::new(0),
            NativeTerminalRsAdjointYBus::new(1),
            log_codeword_len - 1,
            log_codeword_len,
        );
        let _ = BaseAir::<F>::width(&air);
        assert_eq!(air.encoder.width(), 9);
    }

    #[test]
    fn coefficient_subgroup_zero_fold_adjoint_matches_backend_coordinates() {
        let (code, alpha, whir_point, verification) = coefficient_fixture();
        let q_air = NativeTerminalRsAdjointQAir {
            round_bus: NativeTerminalRsAdjointRoundBus::new(1),
            accumulator_value_bus: NativeTerminalAccumulatorValueBus::new(2),
            point_bus: NativeTerminalWhirPointBus::new(3),
            q_bus: NativeTerminalRsAdjointQBus::new(4),
            round_count: alpha.len(),
            initial_folding_factor: 0,
        };
        let q_trace = generate_native_terminal_rs_adjoint_q_trace(&verification, &alpha, 0, None)
            .expect("zero-fold q trace");
        check_air(&q_air, &q_trace);
        for row in q_trace.values.chunks_exact(q_trace.width()) {
            let cols: &NativeTerminalRsAdjointQCols<F> = row.borrow();
            if cols.active == F::ONE {
                assert_eq!(cols.is_column, F::ZERO);
            }
        }

        let y_air = NativeTerminalRsAdjointYAir::new(
            NativeTerminalRsAdjointRoundBus::new(1),
            NativeTerminalRsAdjointYBus::new(5),
            code.log_message_len(),
            code.log_codeword_len(),
        );
        let y_trace = generate_native_terminal_rs_adjoint_y_trace(&y_air, &verification, None)
            .expect("coefficient-subgroup y trace");
        check_air(&y_air, &y_trace);

        let selector_air = NativeTerminalRsAdjointSelectorAir {
            point_bus: NativeTerminalWhirPointBus::new(3),
            q_bus: NativeTerminalRsAdjointQBus::new(4),
            y_bus: NativeTerminalRsAdjointYBus::new(5),
            claim_bus: NativeTerminalRsAdjointClaimBus::new(6),
            value_bus: NativeTerminalRsAdjointValueBus::new(7),
            log_message_len: code.log_message_len(),
            point_coordinate_offset: 0,
        };
        let selector_trace =
            generate_native_terminal_rs_adjoint_selector_trace(&verification, &alpha, 0, None)
                .expect("zero-fold selector trace");
        check_air(&selector_air, &selector_trace);
        let last_row = &selector_trace.values[(whir_point.len() - 1) * selector_trace.width()
            ..whir_point.len() * selector_trace.width()];
        let last: &NativeTerminalRsAdjointSelectorCols<F> = last_row.borrow();
        assert_eq!(
            EF::from_basis_coefficients_slice(&last.final_claim).expect("EF4 final claim"),
            verification.accumulator_adjoint.final_claim,
        );
        assert_eq!(
            verification.accumulator_adjoint.claimed_value,
            code.accumulator_weight_eval_at(&alpha, &whir_point),
        );
    }

    #[test]
    fn coefficient_subgroup_adjoint_rejects_wrong_folding_coordinate_and_target() {
        let (_code, alpha, _whir_point, mut verification) = coefficient_fixture();
        assert!(
            generate_native_terminal_rs_adjoint_q_trace(&verification, &alpha, 1, None,).is_none()
        );
        assert!(
            generate_native_terminal_rs_adjoint_selector_trace(&verification, &alpha, 1, None,)
                .is_none()
        );

        let q_air = NativeTerminalRsAdjointQAir {
            round_bus: NativeTerminalRsAdjointRoundBus::new(10),
            accumulator_value_bus: NativeTerminalAccumulatorValueBus::new(11),
            point_bus: NativeTerminalWhirPointBus::new(12),
            q_bus: NativeTerminalRsAdjointQBus::new(13),
            round_count: alpha.len(),
            initial_folding_factor: 0,
        };
        let mut wrong_coordinate =
            generate_native_terminal_rs_adjoint_q_trace(&verification, &alpha, 0, None)
                .expect("honest q trace");
        let width = wrong_coordinate.width();
        let first: &mut NativeTerminalRsAdjointQCols<F> =
            wrong_coordinate.values[..width].borrow_mut();
        first.alpha[0] += F::ONE;
        assert!(catch_unwind(AssertUnwindSafe(|| {
            check_air(&q_air, &wrong_coordinate);
        }))
        .is_err());

        verification.accumulator_adjoint.final_claim += EF::ONE;
        assert!(
            generate_native_terminal_rs_adjoint_selector_trace(&verification, &alpha, 0, None,)
                .is_none()
        );
    }
}

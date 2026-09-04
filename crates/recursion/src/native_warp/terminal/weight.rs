use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
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
    native_warp::terminal::{
        NativeTerminalWhirPointBus, NativeTerminalWhirPointMessage,
        NativeTerminalWhirWeightTermBus, NativeTerminalWhirWeightTermMessage,
        NativeTerminalWhirWeightTermResultBus, NativeTerminalWhirWeightTermResultMessage,
    },
    utils::ext_field_multiply,
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeTerminalWhirWeightTermCols<T> {
    pub active: T,
    pub ordinal: T,
    pub coordinate: T,
    pub is_first: T,
    pub is_last: T,
    pub after_folds: T,
    pub length: T,
    pub generator: [T; D_EF],
    pub scale: [T; D_EF],
    pub generator_power: [T; D_EF],
    pub point: [T; D_EF],
    pub factor: [T; D_EF],
    pub eq_before: [T; D_EF],
    pub eq_after: [T; D_EF],
}

/// Evaluates one structured equality term
/// `scale * eq(generator^(2^i), point[after_folds + i])`.
#[derive(ColumnsAir)]
#[columns_via(NativeTerminalWhirWeightTermCols<u8>)]
pub struct NativeTerminalWhirWeightTermAir {
    pub term_bus: NativeTerminalWhirWeightTermBus,
    pub result_bus: NativeTerminalWhirWeightTermResultBus,
    pub point_bus: NativeTerminalWhirPointBus,
}

impl BaseAirWithPublicValues<F> for NativeTerminalWhirWeightTermAir {}
impl PartitionedBaseAir<F> for NativeTerminalWhirWeightTermAir {}
impl BaseAir<F> for NativeTerminalWhirWeightTermAir {
    fn width(&self) -> usize {
        NativeTerminalWhirWeightTermCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTerminalWhirWeightTermAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("native terminal weight term row");
        let next_row = main
            .row_slice(1)
            .expect("native terminal next weight term row");
        let local: &NativeTerminalWhirWeightTermCols<AB::Var> = (*local_row).borrow();
        let next: &NativeTerminalWhirWeightTermCols<AB::Var> = (*next_row).borrow();

        for flag in [local.active, local.is_first, local.is_last] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder
            .when(local.active * local.is_first)
            .assert_zero(local.coordinate);
        let same_term = next.active * (AB::Expr::ONE - next.is_first);
        let term_end = local.active - same_term.clone();
        builder
            .when_transition()
            .assert_eq(local.is_last, term_end.clone());
        builder
            .when_last_row()
            .assert_eq(local.is_last, local.active);
        builder
            .when(local.active * local.is_last)
            .assert_eq(local.coordinate + AB::F::ONE, local.length);

        let mut transition = builder.when_transition();
        let mut same = transition.when(same_term);
        same.assert_eq(next.ordinal, local.ordinal);
        same.assert_eq(next.coordinate, local.coordinate + AB::F::ONE);
        same.assert_eq(next.after_folds, local.after_folds);
        same.assert_eq(next.length, local.length);
        assert_array_eq(&mut same, next.generator, local.generator);
        assert_array_eq(&mut same, next.scale, local.scale);
        assert_array_eq(
            &mut same,
            next.generator_power,
            ext_field_multiply::<AB::Expr>(local.generator_power, local.generator_power),
        );
        assert_array_eq(&mut same, next.eq_before, local.eq_after);

        assert_array_eq(
            &mut builder.when(local.active * local.is_first),
            local.generator_power,
            local.generator.map(Into::into),
        );
        let one = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            }
        });
        assert_array_eq(
            &mut builder.when(local.active * local.is_first),
            local.eq_before,
            one.clone(),
        );
        let point_times_generator =
            ext_field_multiply::<AB::Expr>(local.point, local.generator_power);
        let expected_factor = core::array::from_fn(|limb| {
            one[limb].clone()
                - AB::Expr::from(local.point[limb])
                - AB::Expr::from(local.generator_power[limb])
                + AB::Expr::TWO * point_times_generator[limb].clone()
        });
        assert_array_eq(
            &mut builder.when(local.active),
            local.factor,
            expected_factor,
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.eq_after,
            ext_field_multiply::<AB::Expr>(local.eq_before, local.factor),
        );

        self.term_bus.receive(
            builder,
            NativeTerminalWhirWeightTermMessage {
                ordinal: local.ordinal.into(),
                after_folds: local.after_folds.into(),
                length: local.length.into(),
                generator: local.generator.map(Into::into),
                scale: local.scale.map(Into::into),
            },
            local.active * local.is_first,
        );
        self.point_bus.lookup_key(
            builder,
            NativeTerminalWhirPointMessage {
                coordinate: local.after_folds + local.coordinate,
                value: local.point.map(Into::into),
            },
            local.active,
        );
        self.result_bus.send(
            builder,
            NativeTerminalWhirWeightTermResultMessage {
                ordinal: local.ordinal.into(),
                value: ext_field_multiply::<AB::Expr>(local.scale, local.eq_after),
            },
            local.active * local.is_last,
        );
    }
}

pub fn generate_native_terminal_whir_weight_term_trace(
    verification: &TerminalWhirVerification<F, EF, Digest>,
    k: usize,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if k == 0 || verification.rounds.is_empty() {
        return None;
    }
    let mut point = Vec::new();
    for round in &verification.rounds {
        if round.alphas.len() != k {
            return None;
        }
        point.extend_from_slice(&round.alphas);
    }
    point.extend_from_slice(&verification.suffix_point);

    let mut terms = Vec::new();
    let mut ordinal = 0usize;
    for (round_index, round) in verification.rounds.iter().enumerate() {
        let after_folds = (round_index + 1).checked_mul(k)?;
        let length = point.len().checked_sub(after_folds)?;
        if round_index + 1 != verification.rounds.len() {
            terms.push((ordinal, after_folds, length, round.ood_point?, round.gamma));
            ordinal += 1;
        } else if round.ood_point.is_some() || round.ood_value.is_some() {
            return None;
        }
        if round.query_roots.len() != round.query_indices.len() {
            return None;
        }
        for (query_index, &zi_root) in round.query_roots.iter().enumerate() {
            terms.push((
                ordinal,
                after_folds,
                length,
                EF::from(zi_root),
                round.gamma.exp_u64(query_index as u64 + 2),
            ));
            ordinal += 1;
        }
    }
    let valid_rows = terms
        .iter()
        .try_fold(0usize, |rows, term| rows.checked_add(term.2))?;
    if valid_rows == 0 {
        return None;
    }
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeTerminalWhirWeightTermCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    let mut row_index = 0usize;
    for (ordinal, after_folds, length, generator, scale) in terms {
        let mut generator_power = generator;
        let mut eq_prefix = EF::ONE;
        for coordinate in 0..length {
            let point_value = *point.get(after_folds + coordinate)?;
            let factor = (EF::ONE - point_value) * (EF::ONE - generator_power)
                + point_value * generator_power;
            let eq_after = eq_prefix * factor;
            let row = &mut values[row_index * width..(row_index + 1) * width];
            let cols: &mut NativeTerminalWhirWeightTermCols<F> = row.borrow_mut();
            cols.active = F::ONE;
            cols.ordinal = F::from_usize(ordinal);
            cols.coordinate = F::from_usize(coordinate);
            cols.is_first = F::from_bool(coordinate == 0);
            cols.is_last = F::from_bool(coordinate + 1 == length);
            cols.after_folds = F::from_usize(after_folds);
            cols.length = F::from_usize(length);
            copy_ext(&mut cols.generator, generator);
            copy_ext(&mut cols.scale, scale);
            copy_ext(&mut cols.generator_power, generator_power);
            copy_ext(&mut cols.point, point_value);
            copy_ext(&mut cols.factor, factor);
            copy_ext(&mut cols.eq_before, eq_prefix);
            copy_ext(&mut cols.eq_after, eq_after);
            generator_power *= generator_power;
            eq_prefix = eq_after;
            row_index += 1;
        }
    }
    Some(RowMajorMatrix::new(values, width))
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

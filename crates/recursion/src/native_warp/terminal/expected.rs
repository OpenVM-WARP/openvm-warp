use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, poly_common::Squarable, warp_accum::TerminalWhirVerification,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::native_warp::terminal::{
    NativeTerminalRsAdjointValueBus, NativeTerminalRsAdjointValueMessage,
    NativeTerminalWhirActualWeightBus, NativeTerminalWhirActualWeightMessage,
    NativeTerminalWhirLinearizerWeightBus, NativeTerminalWhirLinearizerWeightMessage,
    NativeTerminalWhirWeightTermResultBus, NativeTerminalWhirWeightTermResultMessage,
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeTerminalWhirExpectedWeightCols<T> {
    pub active: T,
    pub ordinal: T,
    pub is_first: T,
    pub is_last: T,
    pub term: [T; D_EF],
    pub sum_before: [T; D_EF],
    pub sum_after: [T; D_EF],
    pub actual: [T; D_EF],
    pub adjoint: [T; D_EF],
    pub linearizer: [T; D_EF],
}

/// Checks the terminal structured-weight identity
///
/// `actual = RS-adjoint + linearizer + sum(OOD/query equality additions)`.
#[derive(ColumnsAir)]
#[columns_via(NativeTerminalWhirExpectedWeightCols<u8>)]
pub struct NativeTerminalWhirExpectedWeightAir {
    pub actual_bus: NativeTerminalWhirActualWeightBus,
    pub adjoint_bus: NativeTerminalRsAdjointValueBus,
    pub linearizer_bus: NativeTerminalWhirLinearizerWeightBus,
    pub term_bus: NativeTerminalWhirWeightTermResultBus,
    pub term_count: usize,
}

impl BaseAirWithPublicValues<F> for NativeTerminalWhirExpectedWeightAir {}
impl PartitionedBaseAir<F> for NativeTerminalWhirExpectedWeightAir {}
impl BaseAir<F> for NativeTerminalWhirExpectedWeightAir {
    fn width(&self) -> usize {
        NativeTerminalWhirExpectedWeightCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTerminalWhirExpectedWeightAir {
    fn eval(&self, builder: &mut AB) {
        debug_assert!(self.term_count > 0);
        let main = builder.main();
        let local_row = main
            .row_slice(0)
            .expect("native terminal expected weight row");
        let next_row = main
            .row_slice(1)
            .expect("native terminal next expected weight row");
        let local: &NativeTerminalWhirExpectedWeightCols<AB::Var> = (*local_row).borrow();
        let next: &NativeTerminalWhirExpectedWeightCols<AB::Var> = (*next_row).borrow();

        for flag in [local.active, local.is_first, local.is_last] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when_first_row().assert_zero(local.ordinal);
        builder
            .when(local.active * local.is_last)
            .assert_eq(local.ordinal, AB::Expr::from_usize(self.term_count - 1));
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
        assert_array_eq(&mut same, next.sum_before, local.sum_after);
        assert_array_eq(&mut same, next.actual, local.actual);
        assert_array_eq(&mut same, next.adjoint, local.adjoint);
        assert_array_eq(&mut same, next.linearizer, local.linearizer);
        builder
            .when_last_row()
            .assert_eq(local.is_last, local.active);

        assert_array_eq(
            &mut builder.when(local.active * local.is_first),
            local.sum_before,
            [AB::Expr::ZERO; D_EF],
        );
        let expected_sum = core::array::from_fn(|limb| {
            AB::Expr::from(local.sum_before[limb]) + AB::Expr::from(local.term[limb])
        });
        assert_array_eq(
            &mut builder.when(local.active),
            local.sum_after,
            expected_sum,
        );
        let expected_actual = core::array::from_fn(|limb| {
            AB::Expr::from(local.adjoint[limb])
                + AB::Expr::from(local.linearizer[limb])
                + AB::Expr::from(local.sum_after[limb])
        });
        assert_array_eq(
            &mut builder.when(local.active * local.is_last),
            local.actual,
            expected_actual,
        );

        self.actual_bus.receive(
            builder,
            NativeTerminalWhirActualWeightMessage {
                value: local.actual.map(Into::into),
            },
            local.active * local.is_first,
        );
        self.adjoint_bus.receive(
            builder,
            NativeTerminalRsAdjointValueMessage {
                value: local.adjoint.map(Into::into),
            },
            local.active * local.is_first,
        );
        self.linearizer_bus.receive(
            builder,
            NativeTerminalWhirLinearizerWeightMessage {
                value: local.linearizer.map(Into::into),
            },
            local.active * local.is_first,
        );
        self.term_bus.receive(
            builder,
            NativeTerminalWhirWeightTermResultMessage {
                ordinal: local.ordinal.into(),
                value: local.term.map(Into::into),
            },
            local.active,
        );
    }
}

pub fn generate_native_terminal_whir_expected_weight_trace(
    verification: &TerminalWhirVerification<F, EF, Digest>,
    k: usize,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if k == 0 || verification.rounds.is_empty() {
        return None;
    }
    let mut point = verification
        .rounds
        .iter()
        .flat_map(|round| round.alphas.iter().copied())
        .collect::<Vec<_>>();
    point.extend_from_slice(&verification.suffix_point);
    let mut terms = Vec::new();
    for (round_index, round) in verification.rounds.iter().enumerate() {
        let after_folds = (round_index + 1).checked_mul(k)?;
        let suffix = point.get(after_folds..)?;
        if round_index + 1 != verification.rounds.len() {
            let generator = round.ood_point?;
            let eq = generator
                .exp_powers_of_2()
                .zip(suffix)
                .map(|(left, &right)| left * right + (EF::ONE - left) * (EF::ONE - right))
                .product::<EF>();
            terms.push(round.gamma * eq);
        }
        if round.query_roots.len() != round.query_indices.len() {
            return None;
        }
        for (query_index, &zi_root) in round.query_roots.iter().enumerate() {
            let generator = EF::from(zi_root);
            let eq = generator
                .exp_powers_of_2()
                .zip(suffix)
                .map(|(left, &right)| left * right + (EF::ONE - left) * (EF::ONE - right))
                .product::<EF>();
            terms.push(round.gamma.exp_u64(query_index as u64 + 2) * eq);
        }
    }
    if terms.is_empty() {
        return None;
    }
    let height = required_height.unwrap_or_else(|| terms.len().next_power_of_two());
    if height < terms.len() {
        return None;
    }
    let width = NativeTerminalWhirExpectedWeightCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    let adjoint = verification.accumulator_adjoint.claimed_value;
    let term_sum = terms.iter().copied().sum::<EF>();
    let linearizer = verification.expected_weight - adjoint - term_sum;
    let mut sum = EF::ZERO;
    for (ordinal, &term) in terms.iter().enumerate() {
        let row = &mut values[ordinal * width..(ordinal + 1) * width];
        let cols: &mut NativeTerminalWhirExpectedWeightCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.ordinal = F::from_usize(ordinal);
        cols.is_first = F::from_bool(ordinal == 0);
        cols.is_last = F::from_bool(ordinal + 1 == terms.len());
        copy_ext(&mut cols.term, term);
        copy_ext(&mut cols.sum_before, sum);
        sum += term;
        copy_ext(&mut cols.sum_after, sum);
        copy_ext(&mut cols.actual, verification.actual_weight);
        copy_ext(&mut cols.adjoint, adjoint);
        copy_ext(&mut cols.linearizer, linearizer);
    }
    Some(RowMajorMatrix::new(values, width))
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

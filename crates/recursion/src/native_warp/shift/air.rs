use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::native_warp::{
    batching::OPENING_SECTION_TARGET,
    bus::{
        NativeAuthenticatedShiftBus, NativeAuthenticatedShiftMessage, NativeEqResultBus,
        NativeEqResultMessage, NativeInputSlotLayoutBus, NativeInputSlotLayoutMessage,
        NativeOpeningClaimBus, NativeOpeningClaimMessage,
    },
    ext::{ext_field_add, ext_field_multiply},
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeShiftMergeCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub shift: T,
    pub source: T,
    pub is_first_source: T,
    pub is_last_source: T,
    pub is_first_shift: T,
    pub variant: T,
    pub source_kind: [T; 3],
    pub gamma_weight: [T; D_EF],
    pub authenticated_value: [T; D_EF],
    pub accumulator_before: [T; D_EF],
    pub accumulator_after: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(NativeShiftMergeCols<u8>)]
pub struct NativeShiftMergeAir {
    pub authenticated_bus: NativeAuthenticatedShiftBus,
    pub eq_bus: NativeEqResultBus,
    pub opening_bus: NativeOpeningClaimBus,
    pub slot_bus: NativeInputSlotLayoutBus,
    pub input_arity: usize,
    pub gamma_eq_group_offset: usize,
    pub opening_claim_offset: usize,
}

impl BaseAirWithPublicValues<F> for NativeShiftMergeAir {}
impl PartitionedBaseAir<F> for NativeShiftMergeAir {}
impl<F> BaseAir<F> for NativeShiftMergeAir {
    fn width(&self) -> usize {
        NativeShiftMergeCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeShiftMergeAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("native shift merge row"),
            main.row_slice(1).expect("native shift merge next row"),
        );
        let local: &NativeShiftMergeCols<AB::Var> = (*local).borrow();
        let next: &NativeShiftMergeCols<AB::Var> = (*next).borrow();
        builder.assert_bool(local.active);
        builder.assert_bool(local.is_first_source);
        builder.assert_bool(local.is_last_source);
        builder.assert_bool(local.is_first_shift);
        builder
            .when(local.active * local.is_first_shift)
            .assert_one(local.is_first_source);
        builder
            .when(local.active * local.is_first_shift)
            .assert_zero(local.shift);
        for flag in local.source_kind {
            builder.assert_bool(flag);
        }
        builder.when(local.active).assert_one(
            local
                .source_kind
                .into_iter()
                .map(AB::Expr::from)
                .sum::<AB::Expr>(),
        );
        self.slot_bus.lookup_key(
            builder,
            NativeInputSlotLayoutMessage {
                variant: local.variant.into(),
                source: local.source.into(),
                kind: local.source_kind.map(Into::into),
            },
            local.active,
        );
        builder.when_first_row().assert_one(local.active);
        builder
            .when(local.active * local.is_first_source)
            .assert_zero(local.source);
        builder
            .when(local.active * local.is_last_source)
            .assert_eq(local.source, AB::Expr::from_usize(self.input_arity - 1));
        self.authenticated_bus.receive(
            builder,
            NativeAuthenticatedShiftMessage {
                proof_idx: local.proof_idx.into(),
                shift: local.shift.into(),
                source: local.source.into(),
                value: local.authenticated_value.map(Into::into),
            },
            local.active * (local.source_kind[0] + local.source_kind[1]),
        );
        for limb in local.authenticated_value {
            builder
                .when(local.active * local.source_kind[2])
                .assert_zero(limb);
        }
        self.eq_bus.lookup_key(
            builder,
            NativeEqResultMessage {
                proof_idx: local.proof_idx.into(),
                group: AB::Expr::from_usize(self.gamma_eq_group_offset) + local.source,
                value: local.gamma_weight.map(Into::into),
            },
            local.active,
        );
        let zero = [AB::Expr::ZERO; D_EF];
        assert_array_eq(
            &mut builder.when(local.active * local.is_first_source),
            local.accumulator_before,
            zero,
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.accumulator_after,
            ext_field_add::<AB::Expr>(
                local.accumulator_before,
                ext_field_multiply::<AB::Expr>(local.gamma_weight, local.authenticated_value),
            ),
        );
        let same_shift = next.active * (AB::Expr::ONE - next.is_first_source);
        let mut transition = builder.when_transition();
        let mut when_same = transition.when(same_shift);
        when_same.assert_zero(local.is_last_source);
        when_same.assert_eq(next.proof_idx, local.proof_idx);
        when_same.assert_zero(next.is_first_shift);
        when_same.assert_eq(next.shift, local.shift);
        when_same.assert_eq(next.source, local.source + AB::F::ONE);
        assert_array_eq(
            &mut when_same,
            next.accumulator_before,
            local.accumulator_after,
        );
        let next_shift = next.active * next.is_first_source;
        let mut transition = builder.when_transition();
        let mut when_next = transition.when(next_shift);
        when_next.assert_one(local.is_last_source);
        when_next
            .when(next.is_first_shift)
            .assert_eq(next.proof_idx, local.proof_idx + AB::F::ONE);
        when_next
            .when(AB::Expr::ONE - next.is_first_shift)
            .assert_eq(next.proof_idx, local.proof_idx);
        when_next
            .when(AB::Expr::ONE - next.is_first_shift)
            .assert_eq(next.shift, local.shift + AB::F::ONE);
        self.opening_bus.send(
            builder,
            NativeOpeningClaimMessage {
                proof_idx: local.proof_idx.into(),
                claim: AB::Expr::from_usize(self.opening_claim_offset) + local.shift,
                section: AB::Expr::from_usize(OPENING_SECTION_TARGET),
                coordinate: AB::Expr::ZERO,
                value: local.accumulator_after.map(Into::into),
            },
            local.active * local.is_last_source,
        );
    }
}

pub fn generate_native_shift_merge_trace(
    proof_idx: usize,
    weights: &[EF],
    shift_answers: &[Vec<EF>],
    fresh_count: usize,
    prior_present: bool,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if weights.is_empty()
        || fresh_count + usize::from(prior_present) > weights.len()
        || shift_answers.is_empty()
        || shift_answers
            .iter()
            .any(|answers| answers.len() != weights.len())
    {
        return None;
    }
    let valid_rows = weights.len() * shift_answers.len();
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeShiftMergeCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    for (shift, answers) in shift_answers.iter().enumerate() {
        let mut accumulator = EF::ZERO;
        for source in 0..weights.len() {
            let before = accumulator;
            accumulator += weights[source] * answers[source];
            let row_index = shift * weights.len() + source;
            let cols: &mut NativeShiftMergeCols<F> =
                trace[row_index * width..(row_index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.proof_idx = F::from_usize(proof_idx);
            cols.shift = F::from_usize(shift);
            cols.source = F::from_usize(source);
            cols.is_first_source = F::from_bool(source == 0);
            cols.is_last_source = F::from_bool(source + 1 == weights.len());
            cols.is_first_shift = F::from_bool(shift == 0 && source == 0);
            cols.variant =
                F::from_usize(fresh_count + usize::from(prior_present) * (weights.len() + 1));
            cols.source_kind[if source < fresh_count {
                0
            } else if prior_present && source == fresh_count {
                1
            } else {
                2
            }] = F::ONE;
            for (target, value) in [
                (&mut cols.gamma_weight, weights[source]),
                (&mut cols.authenticated_value, answers[source]),
                (&mut cols.accumulator_before, before),
                (&mut cols.accumulator_after, accumulator),
            ] {
                target.copy_from_slice(value.as_basis_coefficients_slice());
            }
        }
    }
    Some(RowMajorMatrix::new(trace, width))
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeShiftScheduleCols<T> {
    pub active: T,
    pub shift: T,
    pub coordinate: T,
    pub is_first: T,
    pub is_last: T,
    pub is_first_shift: T,
    pub proof_idx: T,
    pub tidx: T,
    pub sample: T,
    pub quotient: T,
    pub index: T,
    pub bit: T,
    pub power: T,
    pub reconstructed_before: T,
    pub reconstructed_after: T,
    pub value: [T; D_EF],
    pub index_lookup_count: T,
}

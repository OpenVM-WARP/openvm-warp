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
    bus::{
        NativeClaimValueBus, NativeClaimValueMessage, NativeEqResultBus, NativeEqResultMessage,
        NativeSumcheckInitialBus, NativeSumcheckInitialMessage, NativeTwinOmegaBus,
        NativeTwinOmegaMessage,
    },
    ext::{ext_field_add, ext_field_multiply},
    twin::{CLAIM_SECTION_ETA, CLAIM_SECTION_MU},
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeTwinSigmaCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub source: T,
    pub is_first: T,
    pub is_last: T,
    pub selector_weight: [T; D_EF],
    pub mu: [T; D_EF],
    pub eta: [T; D_EF],
    pub omega: [T; D_EF],
    pub accumulator_before: [T; D_EF],
    pub accumulator_after: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(NativeTwinSigmaCols<u8>)]
pub struct NativeTwinSigmaAir {
    pub claim_bus: NativeClaimValueBus,
    pub eq_bus: NativeEqResultBus,
    pub sumcheck_initial_bus: NativeSumcheckInitialBus,
    pub input_arity: usize,
    pub omega_bus: NativeTwinOmegaBus,
}

impl BaseAirWithPublicValues<F> for NativeTwinSigmaAir {}
impl PartitionedBaseAir<F> for NativeTwinSigmaAir {}

impl<F> BaseAir<F> for NativeTwinSigmaAir {
    fn width(&self) -> usize {
        NativeTwinSigmaCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTwinSigmaAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let (local, next) = (
            main.row_slice(0).expect("native twin sigma row"),
            main.row_slice(1).expect("native twin sigma next row"),
        );
        let local: &NativeTwinSigmaCols<AB::Var> = (*local).borrow();
        let next: &NativeTwinSigmaCols<AB::Var> = (*next).borrow();

        builder.assert_bool(local.active);
        builder.assert_bool(local.is_first);
        builder.assert_bool(local.is_last);
        builder.when_first_row().assert_one(local.active);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when(local.active * local.is_first)
            .assert_zero(local.source);
        builder
            .when(local.active * local.is_last)
            .assert_eq(local.source, AB::Expr::from_usize(self.input_arity - 1));

        self.eq_bus.lookup_key(
            builder,
            NativeEqResultMessage {
                proof_idx: local.proof_idx.into(),
                group: local.source.into(),
                value: local.selector_weight.map(Into::into),
            },
            local.active,
        );
        self.omega_bus.lookup_key(
            builder,
            NativeTwinOmegaMessage {
                proof_idx: local.proof_idx.into(),
                value: local.omega.map(Into::into),
            },
            local.active,
        );
        for (section, value) in [(CLAIM_SECTION_MU, local.mu), (CLAIM_SECTION_ETA, local.eta)] {
            self.claim_bus.receive(
                builder,
                NativeClaimValueMessage {
                    proof_idx: local.proof_idx.into(),
                    source: local.source.into(),
                    section: AB::Expr::from_usize(section),
                    coordinate: AB::Expr::ZERO,
                    value: value.map(Into::into),
                },
                local.active,
            );
        }
        let weighted = ext_field_multiply::<AB::Expr>(
            local.selector_weight,
            ext_field_add::<AB::Expr>(
                local.mu,
                ext_field_multiply::<AB::Expr>(local.omega, local.eta),
            ),
        );
        let zero = [AB::Expr::ZERO; D_EF];
        assert_array_eq(
            &mut builder.when(local.active * local.is_first),
            local.accumulator_before,
            zero,
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.accumulator_after,
            ext_field_add::<AB::Expr>(local.accumulator_before, weighted),
        );
        let same = next.active * (AB::Expr::ONE - next.is_first);
        let mut transition = builder.when_transition();
        let mut when_same = transition.when(same);
        when_same.assert_zero(local.is_last);
        when_same.assert_eq(next.proof_idx, local.proof_idx);
        when_same.assert_eq(next.source, local.source + AB::F::ONE);
        assert_array_eq(&mut when_same, next.omega, local.omega);
        assert_array_eq(
            &mut when_same,
            next.accumulator_before,
            local.accumulator_after,
        );
        builder
            .when_transition()
            .when(local.active - next.active)
            .assert_one(local.is_last);

        self.sumcheck_initial_bus.receive(
            builder,
            NativeSumcheckInitialMessage {
                proof_idx: local.proof_idx.into(),
                kind: AB::Expr::ZERO,
                claim: local.accumulator_after.map(Into::into),
            },
            local.active * local.is_last,
        );
    }
}

pub fn generate_native_twin_sigma_trace(
    proof_idx: usize,
    selector_weights: &[EF],
    claims: &[(EF, EF)],
    omega: EF,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if claims.is_empty() || claims.len() != selector_weights.len() {
        return None;
    }
    let valid_rows = claims.len();
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeTwinSigmaCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    let mut accumulator = EF::ZERO;
    for (source, (&weight, &(mu, eta))) in selector_weights.iter().zip(claims).enumerate() {
        let before = accumulator;
        accumulator += weight * (mu + omega * eta);
        let row = &mut trace[source * width..(source + 1) * width];
        let cols: &mut NativeTwinSigmaCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.proof_idx = F::from_usize(proof_idx);
        cols.source = F::from_usize(source);
        cols.is_first = F::from_bool(source == 0);
        cols.is_last = F::from_bool(source + 1 == valid_rows);
        cols.selector_weight
            .copy_from_slice(weight.as_basis_coefficients_slice());
        cols.mu.copy_from_slice(mu.as_basis_coefficients_slice());
        cols.eta.copy_from_slice(eta.as_basis_coefficients_slice());
        cols.omega
            .copy_from_slice(omega.as_basis_coefficients_slice());
        cols.accumulator_before
            .copy_from_slice(before.as_basis_coefficients_slice());
        cols.accumulator_after
            .copy_from_slice(accumulator.as_basis_coefficients_slice());
    }
    Some(RowMajorMatrix::new(trace, width))
}

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

use crate::{
    bus::TranscriptBus,
    native_warp::{
        bus::{
            NativeEqResultBus, NativeEqResultMessage, NativeSumcheckRoundBus,
            NativeSumcheckRoundMessage, NativeTwinOmegaBus, NativeTwinOmegaMessage,
            NativeTwinScalarBus, NativeTwinScalarMessage,
        },
        ext::{ext_field_add, ext_field_multiply},
        twin::{TWIN_SCALAR_ETA, TWIN_SCALAR_NU_0},
    },
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeTwinFinalCols<T> {
    pub active: T,
    pub pre_claim: [T; D_EF],
    pub final_claim: [T; D_EF],
    pub challenge: [T; D_EF],
    pub selector_eq: [T; D_EF],
    pub omega: [T; D_EF],
    pub nu_0: [T; D_EF],
    pub eta: [T; D_EF],
    pub proof_idx: T,
    pub nu_tidx: T,
    pub eta_tidx: T,
}

#[derive(ColumnsAir)]
#[columns_via(NativeTwinFinalCols<u8>)]
pub struct NativeTwinFinalAir {
    pub sumcheck_round_bus: NativeSumcheckRoundBus,
    pub eq_bus: NativeEqResultBus,
    pub scalar_bus: NativeTwinScalarBus,
    pub last_round: usize,
    pub selector_eq_group: usize,
    pub omega_bus: NativeTwinOmegaBus,
    pub transcript_bus: TranscriptBus,
}

impl BaseAirWithPublicValues<F> for NativeTwinFinalAir {}
impl PartitionedBaseAir<F> for NativeTwinFinalAir {}

impl<F> BaseAir<F> for NativeTwinFinalAir {
    fn width(&self) -> usize {
        NativeTwinFinalCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTwinFinalAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("native twin final row");
        let local: &NativeTwinFinalCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        self.sumcheck_round_bus.receive(
            builder,
            NativeSumcheckRoundMessage {
                proof_idx: local.proof_idx.into(),
                kind: AB::Expr::ZERO,
                round: AB::Expr::from_usize(self.last_round),
                pre_claim: local.pre_claim.map(Into::into),
                post_claim: local.final_claim.map(Into::into),
                challenge: local.challenge.map(Into::into),
            },
            local.active,
        );
        self.eq_bus.lookup_key(
            builder,
            NativeEqResultMessage {
                proof_idx: local.proof_idx.into(),
                group: AB::Expr::from_usize(self.selector_eq_group),
                value: local.selector_eq.map(Into::into),
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
        assert_array_eq(
            &mut builder.when(local.active),
            local.final_claim,
            ext_field_multiply::<AB::Expr>(
                local.selector_eq,
                ext_field_add::<AB::Expr>(
                    local.nu_0,
                    ext_field_multiply::<AB::Expr>(local.omega, local.eta),
                ),
            ),
        );
        for (kind, value) in [(TWIN_SCALAR_NU_0, local.nu_0), (TWIN_SCALAR_ETA, local.eta)] {
            self.scalar_bus.send(
                builder,
                NativeTwinScalarMessage {
                    proof_idx: local.proof_idx.into(),
                    kind: AB::Expr::from_usize(kind),
                    value: value.map(Into::into),
                },
                local.active,
            );
        }
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.nu_tidx,
            local.nu_0,
            local.active,
        );
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.eta_tidx,
            local.eta,
            local.active,
        );
    }
}

#[allow(clippy::too_many_arguments)]
pub fn generate_native_twin_final_trace(
    pre_claim: EF,
    final_claim: EF,
    challenge: EF,
    selector_eq: EF,
    omega: EF,
    nu_0: EF,
    eta: EF,
    proof_idx: usize,
    nu_tidx: usize,
    eta_tidx: usize,
) -> RowMajorMatrix<F> {
    let width = NativeTwinFinalCols::<F>::width();
    let mut trace = vec![F::ZERO; width];
    let cols: &mut NativeTwinFinalCols<F> = trace.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    for (target, value) in [
        (&mut cols.pre_claim, pre_claim),
        (&mut cols.final_claim, final_claim),
        (&mut cols.challenge, challenge),
        (&mut cols.selector_eq, selector_eq),
        (&mut cols.omega, omega),
        (&mut cols.nu_0, nu_0),
        (&mut cols.eta, eta),
    ] {
        target.copy_from_slice(value.as_basis_coefficients_slice());
    }
    cols.proof_idx = F::from_usize(proof_idx);
    cols.nu_tidx = F::from_usize(nu_tidx);
    cols.eta_tidx = F::from_usize(eta_tidx);
    RowMajorMatrix::new(trace, width)
}

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    native_warp::bus::{NativeTwinOmegaBus, NativeTwinOmegaMessage},
};

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeTwinOmegaCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub tidx: T,
    pub value: [T; D_EF],
    pub lookup_count: T,
}

#[derive(ColumnsAir)]
#[columns_via(NativeTwinOmegaCols<u8>)]
pub struct NativeTwinOmegaAir {
    pub transcript_bus: TranscriptBus,
    pub omega_bus: NativeTwinOmegaBus,
}

impl BaseAirWithPublicValues<F> for NativeTwinOmegaAir {}
impl PartitionedBaseAir<F> for NativeTwinOmegaAir {}
impl<F> BaseAir<F> for NativeTwinOmegaAir {
    fn width(&self) -> usize {
        NativeTwinOmegaCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTwinOmegaAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("native twin omega row");
        let local: &NativeTwinOmegaCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            local.tidx,
            local.value,
            local.active,
        );
        self.omega_bus.add_key_with_lookups(
            builder,
            NativeTwinOmegaMessage {
                proof_idx: local.proof_idx.into(),
                value: local.value.map(Into::into),
            },
            local.lookup_count,
        );
    }
}

pub fn generate_native_twin_omega_trace(
    proof_idx: usize,
    tidx: usize,
    omega: EF,
    lookup_count: usize,
) -> RowMajorMatrix<F> {
    let width = NativeTwinOmegaCols::<F>::width();
    let mut trace = vec![F::ZERO; width];
    let cols: &mut NativeTwinOmegaCols<F> = trace.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.proof_idx = F::from_usize(proof_idx);
    cols.tidx = F::from_usize(tidx);
    cols.value
        .copy_from_slice(omega.as_basis_coefficients_slice());
    cols.lookup_count = F::from_usize(lookup_count);
    RowMajorMatrix::new(trace, width)
}

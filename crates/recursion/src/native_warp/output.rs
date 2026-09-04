use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    native_warp::bus::{NativeAccumulatorRootBus, NativeAccumulatorRootMessage},
};

/// Binds the VACC output commitment both to Fiat--Shamir and to the next
/// accumulator state. The same digest drives both interactions.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeOutputRootCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub tidx: T,
    pub root: [T; DIGEST_SIZE],
}

#[derive(ColumnsAir)]
#[columns_via(NativeOutputRootCols<u8>)]
pub struct NativeOutputRootAir {
    pub transcript_bus: TranscriptBus,
    pub root_bus: NativeAccumulatorRootBus,
}

impl BaseAirWithPublicValues<F> for NativeOutputRootAir {}
impl PartitionedBaseAir<F> for NativeOutputRootAir {}
impl BaseAir<F> for NativeOutputRootAir {
    fn width(&self) -> usize {
        NativeOutputRootCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeOutputRootAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("native output root row");
        let local: &NativeOutputRootCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        for (limb, value) in local.root.into_iter().enumerate() {
            let extension: [AB::Expr; D_EF] = core::array::from_fn(|coordinate| {
                if coordinate == 0 {
                    value.into()
                } else {
                    AB::Expr::ZERO
                }
            });
            self.transcript_bus.observe_ext(
                builder,
                local.proof_idx,
                local.tidx + AB::Expr::from_usize(limb * D_EF),
                extension,
                local.active,
            );
        }
        self.root_bus.send(
            builder,
            NativeAccumulatorRootMessage {
                proof_idx: local.proof_idx.into(),
                state: AB::Expr::ONE,
                digest: local.root.map(Into::into),
            },
            local.active,
        );
    }
}

pub fn generate_native_output_root_trace(
    proof_idx: usize,
    tidx: usize,
    root: [F; DIGEST_SIZE],
) -> RowMajorMatrix<F> {
    let width = NativeOutputRootCols::<F>::width();
    let mut trace = vec![F::ZERO; width];
    let cols: &mut NativeOutputRootCols<F> = trace.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.proof_idx = F::from_usize(proof_idx);
    cols.tidx = F::from_usize(tidx);
    cols.root = root;
    RowMajorMatrix::new(trace, width)
}

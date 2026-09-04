use core::borrow::Borrow;
use std::{array::from_fn, sync::Arc};

use openvm_circuit_primitives::ColumnsAir;
use openvm_poseidon2_air::{
    Poseidon2SubAir, Poseidon2SubCols, BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::sub::SubAirBuilder, interaction::InteractionBuilder, BaseAirWithPublicValues,
    PartitionedBaseAir,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::Field;
use p3_matrix::Matrix;

use crate::bus::{
    Poseidon2CompressBus, Poseidon2CompressMessage, Poseidon2PermuteBus, Poseidon2PermuteMessage,
};

pub const CHUNK: usize = 8;
pub use openvm_poseidon2_air::POSEIDON2_WIDTH;

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct Poseidon2Cols<T, const SBOX_REGISTERS: usize> {
    pub inner: Poseidon2SubCols<T, SBOX_REGISTERS>,
    pub permute_mult: T,
    pub compress_mult: T,
}

pub struct Poseidon2Air<F: Field, const SBOX_REGISTERS: usize> {
    pub subair: Arc<Poseidon2SubAir<F, SBOX_REGISTERS>>,
    pub poseidon2_permute_bus: Poseidon2PermuteBus,
    pub poseidon2_compress_bus: Poseidon2CompressBus,
}

/// One Poseidon2 table serving several disjoint protocol bus pairs.
///
/// The expensive permutation columns are shared. Each logical owner gets its
/// own `(permute_mult, compress_mult)` pair, so lookup multiplicities cannot
/// move between buses. This lets a caller repartition one logical request set
/// into a fixed number of physical power-of-two traces without adding more
/// AIRs than the unshared layout used.
pub struct Poseidon2MultiBusAir<F: Field, const SBOX_REGISTERS: usize> {
    pub subair: Arc<Poseidon2SubAir<F, SBOX_REGISTERS>>,
    pub buses: Vec<(Poseidon2PermuteBus, Poseidon2CompressBus)>,
}

// No columns provided: `Poseidon2Cols` embeds external `Poseidon2SubCols` which doesn't derive
// `StructReflection`.
impl<F: Field, const SBOX_REGISTERS: usize> ColumnsAir for Poseidon2Air<F, SBOX_REGISTERS> {}

impl<F: Field, const SBOX_REGISTERS: usize> BaseAir<F> for Poseidon2Air<F, SBOX_REGISTERS> {
    fn width(&self) -> usize {
        Poseidon2Cols::<F, SBOX_REGISTERS>::width()
    }
}

impl<F: Field, const SBOX_REGISTERS: usize> ColumnsAir for Poseidon2MultiBusAir<F, SBOX_REGISTERS> {}

impl<F: Field, const SBOX_REGISTERS: usize> BaseAir<F> for Poseidon2MultiBusAir<F, SBOX_REGISTERS> {
    fn width(&self) -> usize {
        self.subair.width() + 2 * self.buses.len()
    }
}

impl<F: Field, const SBOX_REGISTERS: usize> BaseAirWithPublicValues<F>
    for Poseidon2MultiBusAir<F, SBOX_REGISTERS>
{
}
impl<F: Field, const SBOX_REGISTERS: usize> PartitionedBaseAir<F>
    for Poseidon2MultiBusAir<F, SBOX_REGISTERS>
{
}

impl<F: Field, const SBOX_REGISTERS: usize> BaseAirWithPublicValues<F>
    for Poseidon2Air<F, SBOX_REGISTERS>
{
}
impl<F: Field, const SBOX_REGISTERS: usize> PartitionedBaseAir<F>
    for Poseidon2Air<F, SBOX_REGISTERS>
{
}

impl<AB: AirBuilder + InteractionBuilder, const SBOX_REGISTERS: usize> Air<AB>
    for Poseidon2Air<AB::F, SBOX_REGISTERS>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main
            .row_slice(0)
            .expect("window should have at least one row");
        let local: &Poseidon2Cols<AB::Var, SBOX_REGISTERS> = (*local).borrow();

        let mut sub_builder =
            SubAirBuilder::<AB, Poseidon2SubAir<AB::F, SBOX_REGISTERS>, AB::F>::new(
                builder,
                0..self.subair.width(),
            );
        self.subair.eval(&mut sub_builder);

        self.poseidon2_permute_bus.add_key_with_lookups(
            builder,
            Poseidon2PermuteMessage {
                input: local.inner.inputs,
                output: local.inner.ending_full_rounds[BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS - 1]
                    .post,
            },
            local.permute_mult,
        );

        self.poseidon2_compress_bus.add_key_with_lookups(
            builder,
            Poseidon2CompressMessage {
                input: local.inner.inputs,
                output: from_fn(|i| {
                    local.inner.ending_full_rounds[BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS - 1].post[i]
                }),
            },
            local.compress_mult,
        );
    }
}

impl<AB: AirBuilder + InteractionBuilder, const SBOX_REGISTERS: usize> Air<AB>
    for Poseidon2MultiBusAir<AB::F, SBOX_REGISTERS>
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main
            .row_slice(0)
            .expect("window should have at least one row");
        let inner_width = self.subair.width();
        let first: &Poseidon2Cols<AB::Var, SBOX_REGISTERS> = local[..inner_width + 2].borrow();

        let mut sub_builder =
            SubAirBuilder::<AB, Poseidon2SubAir<AB::F, SBOX_REGISTERS>, AB::F>::new(
                builder,
                0..inner_width,
            );
        self.subair.eval(&mut sub_builder);

        let permute_message = Poseidon2PermuteMessage {
            input: first.inner.inputs,
            output: first.inner.ending_full_rounds[BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS - 1].post,
        };
        let compress_message = Poseidon2CompressMessage {
            input: first.inner.inputs,
            output: from_fn(|i| {
                first.inner.ending_full_rounds[BABY_BEAR_POSEIDON2_HALF_FULL_ROUNDS - 1].post[i]
            }),
        };
        for (index, (permute_bus, compress_bus)) in self.buses.iter().enumerate() {
            permute_bus.add_key_with_lookups(
                builder,
                permute_message.clone(),
                local[inner_width + 2 * index],
            );
            compress_bus.add_key_with_lookups(
                builder,
                compress_message.clone(),
                local[inner_width + 2 * index + 1],
            );
        }
    }
}

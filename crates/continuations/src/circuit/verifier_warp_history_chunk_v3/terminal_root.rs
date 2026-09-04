//! Stateless terminal root for a recursively composed verifier-WARP History.
//!
//! The root consumes the complete authenticated interval statement, requires
//! genesis-to-terminal coverage, binds its final memory root and public-value
//! digest to the genuine block public-values Merkle-path AIR, and publishes
//! the interval plus the raw user public values in one canonical AIR vector.

use core::borrow::{Borrow, BorrowMut};

use openvm_recursion_circuit::bus::{PublicValuesBus, PublicValuesBusMessage};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, F};

use super::{
    VerifierWarpHistoryChunkIntervalBusV3, VerifierWarpHistoryChunkIntervalMessageV3,
    CHUNK_INDEX_V3, END_BATCH_INDEX_V3, FINAL_MEMORY_ROOT_V3, IS_GENESIS_V3, IS_TERMINAL_V3,
    PUBLIC_VALUES_DIGEST_V3, START_BATCH_INDEX_V3, TOTAL_BATCH_COUNT_V3,
    VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3,
};
use crate::circuit::verifier_warp_history_v2::{
    VerifierWarpBlockPublicValuesBusV2, VerifierWarpBlockPublicValuesMessageV2,
    VerifierWarpHistoryChunkPublicValuesV3,
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct VerifierWarpHistoryTerminalRootColsV3<T> {
    active: T,
    interval: VerifierWarpHistoryChunkIntervalMessageV3<T>,
}

#[derive(Clone, Debug)]
pub struct VerifierWarpHistoryTerminalRootAirV3 {
    interval_bus: VerifierWarpHistoryChunkIntervalBusV3,
    block_public_values_bus: VerifierWarpBlockPublicValuesBusV2,
    raw_public_values_bus: PublicValuesBus,
    raw_public_values_proof_index: u32,
    raw_public_values_air_id: u32,
    num_user_public_values: usize,
    expected_total_batch_count: u32,
    expected_terminal_chunk_index: u32,
}

impl VerifierWarpHistoryTerminalRootAirV3 {
    pub fn new(
        interval_bus: VerifierWarpHistoryChunkIntervalBusV3,
        block_public_values_bus: VerifierWarpBlockPublicValuesBusV2,
        raw_public_values_bus: PublicValuesBus,
        raw_public_values_proof_index: u32,
        raw_public_values_air_id: u32,
        num_user_public_values: usize,
        expected_total_batch_count: u32,
        expected_terminal_chunk_index: u32,
    ) -> Result<Self, &'static str> {
        if expected_total_batch_count == 0
            || num_user_public_values < DIGEST_SIZE
            || !num_user_public_values.is_multiple_of(DIGEST_SIZE)
            || !(num_user_public_values / DIGEST_SIZE).is_power_of_two()
        {
            return Err("invalid terminal History user-public-values shape");
        }
        Ok(Self {
            interval_bus,
            block_public_values_bus,
            raw_public_values_bus,
            raw_public_values_proof_index,
            raw_public_values_air_id,
            num_user_public_values,
            expected_total_batch_count,
            expected_terminal_chunk_index,
        })
    }

    #[must_use]
    pub const fn public_values_width(&self) -> usize {
        VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3 + self.num_user_public_values
    }

    pub fn generate_trace(
        &self,
        interval: VerifierWarpHistoryChunkPublicValuesV3,
        user_public_values: &[F],
    ) -> Result<(RowMajorMatrix<F>, Vec<F>), &'static str> {
        if user_public_values.len() != self.num_user_public_values
            || interval.start_batch_index != 0
            || interval.total_batch_count != self.expected_total_batch_count
            || interval.chunk_index != self.expected_terminal_chunk_index
            || interval.end_batch_index != interval.total_batch_count
            || !interval.is_genesis
            || !interval.is_terminal
        {
            return Err("invalid terminal History endpoint");
        }
        let width = self.width();
        let mut values = F::zero_vec(2 * width);
        let local: &mut VerifierWarpHistoryTerminalRootColsV3<F> = values[..width].borrow_mut();
        local.active = F::ONE;
        local.interval = VerifierWarpHistoryChunkIntervalMessageV3::from_public_values(interval);
        let mut public_values = interval.to_vec();
        public_values.extend_from_slice(user_public_values);
        Ok((RowMajorMatrix::new(values, width), public_values))
    }
}

impl BaseAir<F> for VerifierWarpHistoryTerminalRootAirV3 {
    fn width(&self) -> usize {
        core::mem::size_of::<VerifierWarpHistoryTerminalRootColsV3<u8>>()
    }
}

impl BaseAirWithPublicValues<F> for VerifierWarpHistoryTerminalRootAirV3 {
    fn num_public_values(&self) -> usize {
        self.public_values_width()
    }
}

impl PartitionedBaseAir<F> for VerifierWarpHistoryTerminalRootAirV3 {}

impl<AB> Air<AB> for VerifierWarpHistoryTerminalRootAirV3
where
    AB: AirBuilder<F = F> + AirBuilderWithPublicValues + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
    AB::PublicVar: Copy,
    AB::Expr: From<AB::PublicVar>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("terminal History root row");
        let next_row = main
            .row_slice(1)
            .expect("terminal History root padding row");
        let local: &VerifierWarpHistoryTerminalRootColsV3<AB::Var> = (*local_row).borrow();
        let next: &VerifierWarpHistoryTerminalRootColsV3<AB::Var> = (*next_row).borrow();
        let enabled = Into::<AB::Expr>::into(local.active);
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_last_row().assert_zero(local.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.active);

        self.interval_bus
            .lookup_key(builder, local.interval.clone(), local.active);
        for index in START_BATCH_INDEX_V3 {
            builder
                .when(enabled.clone())
                .assert_zero(local.interval.fields[index]);
        }
        for (end, total) in END_BATCH_INDEX_V3.zip(TOTAL_BATCH_COUNT_V3) {
            builder
                .when(enabled.clone())
                .assert_eq(local.interval.fields[end], local.interval.fields[total]);
        }
        for (index, limb) in TOTAL_BATCH_COUNT_V3.zip(u32_limbs(self.expected_total_batch_count)) {
            builder
                .when(enabled.clone())
                .assert_eq(local.interval.fields[index], AB::Expr::from_u32(limb));
        }
        for (index, limb) in CHUNK_INDEX_V3.zip(u32_limbs(self.expected_terminal_chunk_index)) {
            builder
                .when(enabled.clone())
                .assert_eq(local.interval.fields[index], AB::Expr::from_u32(limb));
        }
        builder
            .when(enabled.clone())
            .assert_one(local.interval.fields[IS_GENESIS_V3]);
        builder
            .when(enabled.clone())
            .assert_one(local.interval.fields[IS_TERMINAL_V3]);
        self.block_public_values_bus.lookup_key(
            builder,
            VerifierWarpBlockPublicValuesMessageV2 {
                final_memory_root: core::array::from_fn(|limb| {
                    local.interval.fields[FINAL_MEMORY_ROOT_V3.start + limb].into()
                }),
                public_values_digest: core::array::from_fn(|limb| {
                    local.interval.fields[PUBLIC_VALUES_DIGEST_V3.start + limb].into()
                }),
            },
            local.active,
        );

        let public = builder.public_values().to_vec();
        assert_eq!(public.len(), self.public_values_width());
        for (index, interval_value) in local.interval.fields.iter().copied().enumerate() {
            builder
                .when(enabled.clone())
                .assert_eq(interval_value, AB::Expr::from(public[index]));
        }
        for index in 0..self.num_user_public_values {
            self.raw_public_values_bus.send(
                builder,
                AB::Expr::from_u32(self.raw_public_values_proof_index),
                PublicValuesBusMessage {
                    air_idx: AB::Expr::from_u32(self.raw_public_values_air_id),
                    pv_idx: AB::Expr::from_usize(index),
                    value: AB::Expr::from(
                        public[VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3 + index],
                    ),
                },
                enabled.clone(),
            );
        }
    }
}

const fn u32_limbs(value: u32) -> [u32; 2] {
    [value & 0xffff, value >> 16]
}

const _: () = assert!(
    core::mem::size_of::<VerifierWarpHistoryTerminalRootColsV3<u8>>()
        == 1 + VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3
);

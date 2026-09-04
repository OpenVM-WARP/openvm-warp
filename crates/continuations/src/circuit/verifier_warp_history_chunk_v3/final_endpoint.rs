//! Adapter from a complete authenticated History-v3 interval to the legacy
//! direct-final endpoint bus.
//!
//! The adapter is intentionally an AIR, not a host conversion. It consumes
//! all 108 fields from the recursively verified child, pins complete coverage
//! and the setup-fixed protocol dimensions, authenticates the block public
//! values against the final memory root, and only then publishes the 83-field
//! endpoint consumed by the existing terminal-Decide final statement AIR.

use core::borrow::{Borrow, BorrowMut};

use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::F;

use super::{
    VerifierWarpHistoryChunkIntervalBusV3, VerifierWarpHistoryChunkIntervalMessageV3,
    ACTIVE_CHILD_COUNT_AFTER_V3, CHUNK_INDEX_V3, CHUNK_PROTOCOL_VERSION_V3, END_BATCH_INDEX_V3,
    FINAL_ACCUMULATOR_DIGEST_V3, FINAL_MEMORY_ROOT_V3, FINAL_PC_V3, HISTORY_HASH_AFTER_V3,
    INITIAL_ACCUMULATOR_DIGEST_V3, INITIAL_MEMORY_ROOT_V3, INITIAL_PC_V3, IS_GENESIS_V3,
    IS_TERMINAL_V3, PROGRAM_COMMITMENT_V3, PROTOCOL_DIGEST_V3, PUBLIC_VALUES_DIGEST_V3,
    RELATION_DIGEST_V3, SOURCE_CHILD_CAPACITY_V3, SOURCE_HISTORY_PROTOCOL_VERSION_V3,
    START_BATCH_INDEX_V3, TOTAL_BATCH_COUNT_V3, VACC_INPUT_ARITY_V3,
    VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3,
};
use crate::circuit::verifier_warp_history_v2::{
    VerifierWarpBlockPublicValuesBusV2, VerifierWarpBlockPublicValuesMessageV2,
    VerifierWarpHistoryChunkPublicValuesV3, VerifierWarpHistoryEndpointBusV2,
    VerifierWarpHistoryEndpointMessageV2, VERIFIER_WARP_HISTORY_CHUNK_PROTOCOL_V3,
    VERIFIER_WARP_HISTORY_PROTOCOL_V2, VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2,
    VERIFIER_WARP_VACC_INPUT_ARITY_V2,
};

#[repr(C)]
#[derive(AlignedBorrow)]
struct VerifierWarpHistoryFinalEndpointColsV3<T> {
    active: T,
    interval: VerifierWarpHistoryChunkIntervalMessageV3<T>,
}

#[derive(Clone, Debug)]
pub struct VerifierWarpHistoryFinalEndpointAirV3 {
    interval_bus: VerifierWarpHistoryChunkIntervalBusV3,
    block_public_values_bus: VerifierWarpBlockPublicValuesBusV2,
    endpoint_bus: VerifierWarpHistoryEndpointBusV2,
    expected_total_batch_count: u32,
    expected_terminal_chunk_index: u32,
}

impl VerifierWarpHistoryFinalEndpointAirV3 {
    pub fn new(
        interval_bus: VerifierWarpHistoryChunkIntervalBusV3,
        block_public_values_bus: VerifierWarpBlockPublicValuesBusV2,
        endpoint_bus: VerifierWarpHistoryEndpointBusV2,
        expected_total_batch_count: u32,
        expected_terminal_chunk_index: u32,
    ) -> Result<Self, &'static str> {
        if expected_total_batch_count == 0 {
            return Err("invalid final History-v3 endpoint adapter profile");
        }
        Ok(Self {
            interval_bus,
            block_public_values_bus,
            endpoint_bus,
            expected_total_batch_count,
            expected_terminal_chunk_index,
        })
    }

    pub fn generate_trace(
        &self,
        interval: VerifierWarpHistoryChunkPublicValuesV3,
    ) -> Result<RowMajorMatrix<F>, &'static str> {
        if interval.chunk_protocol_version != VERIFIER_WARP_HISTORY_CHUNK_PROTOCOL_V3
            || interval.source_history_protocol_version != VERIFIER_WARP_HISTORY_PROTOCOL_V2
            || interval.source_child_capacity != VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2 as u32
            || interval.vacc_input_arity != VERIFIER_WARP_VACC_INPUT_ARITY_V2 as u32
            || interval.total_batch_count != self.expected_total_batch_count
            || interval.chunk_index != self.expected_terminal_chunk_index
            || interval.start_batch_index != 0
            || interval.end_batch_index != interval.total_batch_count
            || !interval.is_genesis
            || !interval.is_terminal
        {
            return Err("invalid final History-v3 interval endpoint");
        }
        let width = self.width();
        let mut values = F::zero_vec(2 * width);
        let local: &mut VerifierWarpHistoryFinalEndpointColsV3<F> = values[..width].borrow_mut();
        local.active = F::ONE;
        local.interval = VerifierWarpHistoryChunkIntervalMessageV3::from_public_values(interval);
        Ok(RowMajorMatrix::new(values, width))
    }
}

impl BaseAir<F> for VerifierWarpHistoryFinalEndpointAirV3 {
    fn width(&self) -> usize {
        core::mem::size_of::<VerifierWarpHistoryFinalEndpointColsV3<u8>>()
    }
}

impl BaseAirWithPublicValues<F> for VerifierWarpHistoryFinalEndpointAirV3 {}
impl PartitionedBaseAir<F> for VerifierWarpHistoryFinalEndpointAirV3 {}

impl<AB> Air<AB> for VerifierWarpHistoryFinalEndpointAirV3
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("final History endpoint row");
        let next_row = main
            .row_slice(1)
            .expect("final History endpoint padding row");
        let local: &VerifierWarpHistoryFinalEndpointColsV3<AB::Var> = (*local_row).borrow();
        let next: &VerifierWarpHistoryFinalEndpointColsV3<AB::Var> = (*next_row).borrow();
        let enabled = Into::<AB::Expr>::into(local.active);
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_last_row().assert_zero(local.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.active);

        self.interval_bus
            .lookup_key(builder, local.interval.clone(), local.active);
        assert_constant(
            builder,
            enabled.clone(),
            local.interval.fields[CHUNK_PROTOCOL_VERSION_V3],
            VERIFIER_WARP_HISTORY_CHUNK_PROTOCOL_V3,
        );
        assert_constant(
            builder,
            enabled.clone(),
            local.interval.fields[SOURCE_HISTORY_PROTOCOL_VERSION_V3],
            VERIFIER_WARP_HISTORY_PROTOCOL_V2,
        );
        assert_constant(
            builder,
            enabled.clone(),
            local.interval.fields[SOURCE_CHILD_CAPACITY_V3],
            VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2 as u32,
        );
        assert_constant(
            builder,
            enabled.clone(),
            local.interval.fields[VACC_INPUT_ARITY_V3],
            VERIFIER_WARP_VACC_INPUT_ARITY_V2 as u32,
        );
        assert_u32_limbs(
            builder,
            enabled.clone(),
            &local.interval.fields,
            TOTAL_BATCH_COUNT_V3,
            self.expected_total_batch_count,
        );
        assert_u32_limbs(
            builder,
            enabled.clone(),
            &local.interval.fields,
            CHUNK_INDEX_V3,
            self.expected_terminal_chunk_index,
        );
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
        builder
            .when(enabled.clone())
            .assert_one(local.interval.fields[IS_GENESIS_V3]);
        builder
            .when(enabled.clone())
            .assert_one(local.interval.fields[IS_TERMINAL_V3]);

        self.block_public_values_bus.lookup_key(
            builder,
            VerifierWarpBlockPublicValuesMessageV2 {
                final_memory_root: range_array(&local.interval.fields, FINAL_MEMORY_ROOT_V3),
                public_values_digest: range_array(&local.interval.fields, PUBLIC_VALUES_DIGEST_V3),
            },
            local.active,
        );
        self.endpoint_bus.add_key_with_lookups(
            builder,
            VerifierWarpHistoryEndpointMessageV2 {
                protocol_version: local.interval.fields[SOURCE_HISTORY_PROTOCOL_VERSION_V3],
                source_child_capacity: local.interval.fields[SOURCE_CHILD_CAPACITY_V3],
                vacc_input_arity: local.interval.fields[VACC_INPUT_ARITY_V3],
                protocol_digest: range_array(&local.interval.fields, PROTOCOL_DIGEST_V3),
                relation_digest: range_array(&local.interval.fields, RELATION_DIGEST_V3),
                program_commitment: range_array(&local.interval.fields, PROGRAM_COMMITMENT_V3),
                batch_count: range_array(&local.interval.fields, TOTAL_BATCH_COUNT_V3),
                active_child_count: range_array(
                    &local.interval.fields,
                    ACTIVE_CHILD_COUNT_AFTER_V3,
                ),
                initial_pc: local.interval.fields[INITIAL_PC_V3],
                initial_memory_root: range_array(&local.interval.fields, INITIAL_MEMORY_ROOT_V3),
                final_pc: local.interval.fields[FINAL_PC_V3],
                final_memory_root: range_array(&local.interval.fields, FINAL_MEMORY_ROOT_V3),
                genesis_accumulator_digest: range_array(
                    &local.interval.fields,
                    INITIAL_ACCUMULATOR_DIGEST_V3,
                ),
                final_accumulator_digest: range_array(
                    &local.interval.fields,
                    FINAL_ACCUMULATOR_DIGEST_V3,
                ),
                public_values_digest: range_array(&local.interval.fields, PUBLIC_VALUES_DIGEST_V3),
                history_statement_digest: range_array(
                    &local.interval.fields,
                    HISTORY_HASH_AFTER_V3,
                ),
            },
            local.active,
        );
    }
}

fn assert_constant<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    enabled: AB::Expr,
    value: AB::Var,
    expected: u32,
) where
    AB::Var: Copy,
{
    builder
        .when(enabled)
        .assert_eq(value, AB::Expr::from_u32(expected));
}

fn assert_u32_limbs<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    enabled: AB::Expr,
    fields: &[AB::Var; VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3],
    range: core::ops::Range<usize>,
    expected: u32,
) where
    AB::Var: Copy,
{
    for (index, limb) in range.zip([expected & 0xffff, expected >> 16]) {
        builder
            .when(enabled.clone())
            .assert_eq(fields[index], AB::Expr::from_u32(limb));
    }
}

fn range_array<T: Copy, const N: usize>(fields: &[T], range: core::ops::Range<usize>) -> [T; N] {
    fields[range]
        .try_into()
        .expect("History-v3 canonical range width")
}

const _: () = assert!(
    core::mem::size_of::<VerifierWarpHistoryFinalEndpointColsV3<u8>>()
        == 1 + VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3
);

//! Adapter from a generic recursive MultiSTARK verifier to the typed
//! History-v3 interval bus.
//!
//! The generic verifier exposes each authenticated child public value on a
//! coordinate bus. This bridge drains every coordinate, fixes all non-History
//! public values to their keygen template, authenticates the child VK hash,
//! and re-exports the complete 108-coordinate History statement as one typed
//! lookup key. No host-provided digest is accepted as a substitute.

use std::sync::Arc;

use openvm_recursion_circuit::bus::{
    PreHashBus, PreHashMessage, PublicValuesBus, PublicValuesBusMessage,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    keygen::types::MultiStarkVerifyingKey,
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, Digest, F};

use super::{
    VerifierWarpHistoryChunkIntervalBusV3, VerifierWarpHistoryChunkIntervalMessageV3,
    VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3,
};

#[derive(Clone, Copy, Debug)]
struct ChildPublicValueCoordinateV3 {
    air_id: usize,
    pv_index: usize,
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct VerifierWarpHistoryChildProofBridgeColsV3<T> {
    active: T,
    // The actual tail is setup-sized and accessed as a flat row. Keep one
    // reflected coordinate so the row has a nonzero statically known prefix.
    first_child_public_value: T,
}

#[derive(Clone, Debug)]
pub struct VerifierWarpHistoryChildProofBridgeV3 {
    child_coordinates: Arc<[ChildPublicValueCoordinateV3]>,
    child_fixed_values: Arc<[Option<F>]>,
    child_air_count: usize,
    child_history_start: usize,
    child_vk_pre_hash: Digest,
    child_public_values_bus: PublicValuesBus,
    child_pre_hash_bus: PreHashBus,
    output_bus: VerifierWarpHistoryChunkIntervalBusV3,
}

impl VerifierWarpHistoryChildProofBridgeV3 {
    pub fn new(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        child_history_air_id: usize,
        child_expected_public_values: &[Vec<F>],
        child_public_values_bus: PublicValuesBus,
        child_pre_hash_bus: PreHashBus,
        output_bus: VerifierWarpHistoryChunkIntervalBusV3,
    ) -> Result<Self, &'static str> {
        if child_expected_public_values.len() != child_vk.inner.per_air.len()
            || child_vk
                .inner
                .per_air
                .get(child_history_air_id)
                .is_none_or(|air| {
                    air.params.num_public_values != VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3
                })
        {
            return Err("invalid recursive verifier-WARP child profile");
        }
        let mut child_coordinates = Vec::new();
        let mut child_fixed_values = Vec::new();
        let mut child_history_start = None;
        for (air_id, air) in child_vk.inner.per_air.iter().enumerate() {
            let expected = child_expected_public_values
                .get(air_id)
                .ok_or("missing expected child public values")?;
            if expected.len() != air.params.num_public_values {
                return Err("expected child public-value shape differs from VK");
            }
            if air_id == child_history_air_id {
                child_history_start = Some(child_coordinates.len());
            }
            for (pv_index, &value) in expected.iter().enumerate() {
                child_coordinates.push(ChildPublicValueCoordinateV3 { air_id, pv_index });
                child_fixed_values.push((air_id != child_history_air_id).then_some(value));
            }
        }
        let child_history_start =
            child_history_start.ok_or("missing child History public values")?;
        Ok(Self {
            child_coordinates: child_coordinates.into(),
            child_fixed_values: child_fixed_values.into(),
            child_air_count: child_vk.inner.per_air.len(),
            child_history_start,
            child_vk_pre_hash: child_vk.pre_hash,
            child_public_values_bus,
            child_pre_hash_bus,
            output_bus,
        })
    }

    #[must_use]
    pub const fn output_bus(&self) -> VerifierWarpHistoryChunkIntervalBusV3 {
        self.output_bus
    }

    #[must_use]
    pub fn generate_trace(&self, child_public_values: &[Vec<F>]) -> Option<RowMajorMatrix<F>> {
        if child_public_values.len() != self.child_air_count {
            return None;
        }
        let flattened = self
            .child_coordinates
            .iter()
            .map(|coordinate| {
                child_public_values
                    .get(coordinate.air_id)?
                    .get(coordinate.pv_index)
                    .copied()
            })
            .collect::<Option<Vec<_>>>()?;
        let width = self.width();
        let mut values = F::zero_vec(2 * width);
        values[0] = F::ONE;
        values[1..1 + flattened.len()].copy_from_slice(&flattened);
        Some(RowMajorMatrix::new(values, width))
    }

    fn child_value_offset(&self, flattened_index: usize) -> usize {
        1 + flattened_index
    }
}

impl BaseAir<F> for VerifierWarpHistoryChildProofBridgeV3 {
    fn width(&self) -> usize {
        1 + self.child_coordinates.len()
    }
}

impl BaseAirWithPublicValues<F> for VerifierWarpHistoryChildProofBridgeV3 {}
impl PartitionedBaseAir<F> for VerifierWarpHistoryChildProofBridgeV3 {}

impl<AB> Air<AB> for VerifierWarpHistoryChildProofBridgeV3
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("History-v3 child bridge row");
        let next = main
            .row_slice(1)
            .expect("History-v3 child bridge padding row");
        let active = local[0];
        builder.assert_bool(active);
        builder.when_first_row().assert_one(active);
        builder.when_last_row().assert_zero(active);
        builder
            .when_transition()
            .assert_eq(active - next[0], active);

        for (flattened_index, coordinate) in self.child_coordinates.iter().enumerate() {
            let value = local[self.child_value_offset(flattened_index)];
            self.child_public_values_bus.receive(
                builder,
                AB::Expr::ZERO,
                PublicValuesBusMessage {
                    air_idx: AB::Expr::from_usize(coordinate.air_id),
                    pv_idx: AB::Expr::from_usize(coordinate.pv_index),
                    value: value.into(),
                },
                active,
            );
            if let Some(expected) = self.child_fixed_values[flattened_index] {
                builder.when(active).assert_eq(value, expected);
            }
        }
        self.child_pre_hash_bus.receive(
            builder,
            AB::Expr::ZERO,
            PreHashMessage {
                vk_pre_hash: self.child_vk_pre_hash.map(AB::Expr::from),
            },
            active,
        );

        let fields = core::array::from_fn(|index| {
            local[self.child_value_offset(self.child_history_start + index)].into()
        });
        self.output_bus.add_key_with_lookups(
            builder,
            VerifierWarpHistoryChunkIntervalMessageV3 { fields },
            active,
        );
    }
}

const _: () = assert!(core::mem::size_of::<VerifierWarpHistoryChildProofBridgeColsV3<u8>>() == 2);

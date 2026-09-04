//! Recursive chaining adapter for bounded protocol-v19 History chunks.
//!
//! The direct History AIR remains the authority for the current chunk. This
//! adapter drains every public value emitted by the generic one-child STARK
//! verifier and equates the child's final History boundary with the current
//! chunk's initial boundary. It does not merge PESAT relations.

use std::sync::Arc;

use openvm_recursion_circuit::{
    bus::{PreHashBus, PreHashMessage, PublicValuesBus, PublicValuesBusMessage},
    system::BusIndexManager,
};
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, PermutationCheckBus},
    keygen::types::MultiStarkVerifyingKey,
    p3_air::{Air, AirBuilder, BaseAir, BaseAirWithPublicValues, PairBuilder},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config, Digest, DIGEST_SIZE, F,
};

use super::history_public_values_width_v19;

/// A compact per-coordinate bus for the direct History AIR's public values.
#[derive(Clone, Copy, Debug)]
pub struct HistoryPublicValuesBusV19(PermutationCheckBus);

impl HistoryPublicValuesBusV19 {
    #[must_use]
    pub fn new(bus_index: BusIndex) -> Self {
        Self(PermutationCheckBus::new(bus_index))
    }

    pub fn send<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        index: impl Into<AB::Expr>,
        value: impl Into<AB::Expr>,
        multiplicity: impl Into<AB::Expr>,
    ) {
        self.0
            .send(builder, vec![index.into(), value.into()], multiplicity);
    }

    pub fn receive<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        index: impl Into<AB::Expr>,
        value: impl Into<AB::Expr>,
        multiplicity: impl Into<AB::Expr>,
    ) {
        self.0
            .receive(builder, vec![index.into(), value.into()], multiplicity);
    }
}

#[derive(Clone, Copy, Debug)]
struct ChildPublicValueV19 {
    air_id: usize,
    pv_idx: usize,
}

/// Layout of one bridge trace row: all child public values followed by all
/// current direct-History public values.
#[derive(Clone, Debug)]
pub struct HistoryChunkBridgeV19 {
    child_entries: Arc<[ChildPublicValueV19]>,
    /// Setup-derived fixed values in the same flattened order as
    /// `child_entries`. The child History envelope is deliberately `None`
    /// because its final boundary is chained below; every other public value
    /// (notably symbolic-expression DAG commitments) must remain fixed.
    child_fixed_values: Arc<[Option<F>]>,
    child_history_start: usize,
    current_start: usize,
    child_vk_pre_hash: Digest,
    child_public_values_bus: PublicValuesBus,
    child_pre_hash_bus: PreHashBus,
    current_public_values_bus: HistoryPublicValuesBusV19,
    current_segment_count: usize,
}

// Offsets in the repr(C), field-only `HistoryPublicValuesV19<F>` layout.
const PROFILE_START: usize = 1;
const APP_START: usize = PROFILE_START + DIGEST_SIZE;
const INITIAL_COUNT_START: usize = APP_START + DIGEST_SIZE;
const INITIAL_VM_START: usize = INITIAL_COUNT_START + 2;
const INITIAL_PRODUCT_START: usize = INITIAL_VM_START + DIGEST_SIZE;
const INITIAL_HISTORY_START: usize = INITIAL_PRODUCT_START + DIGEST_SIZE;
const INITIAL_PUBLIC_VALUES_START: usize = INITIAL_HISTORY_START + DIGEST_SIZE;
const INITIAL_TERMINATED: usize = INITIAL_PUBLIC_VALUES_START + DIGEST_SIZE;
const FINAL_COUNT_START: usize = INITIAL_TERMINATED + 1;
const FINAL_VM_START: usize = FINAL_COUNT_START + 2;
const FINAL_PRODUCT_START: usize = FINAL_VM_START + DIGEST_SIZE;
const FINAL_HISTORY_START: usize = FINAL_PRODUCT_START + DIGEST_SIZE;
const FINAL_PUBLIC_VALUES_START: usize = FINAL_HISTORY_START + DIGEST_SIZE;
const FINAL_TERMINATED: usize = FINAL_PUBLIC_VALUES_START + DIGEST_SIZE;
const TERMINAL_CHUNK: usize = FINAL_TERMINATED + 1 + 2 + 2 * DIGEST_SIZE;

impl HistoryChunkBridgeV19 {
    pub fn new(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        child_history_air_id: usize,
        child_expected_public_values: &[Vec<F>],
        child_public_values_bus: PublicValuesBus,
        child_pre_hash_bus: PreHashBus,
        current_public_values_bus: HistoryPublicValuesBusV19,
        current_segment_count: usize,
    ) -> Result<Self, &'static str> {
        let width = history_public_values_width_v19();
        if current_segment_count == 0
            || TERMINAL_CHUNK >= width
            || child_expected_public_values.len() != child_vk.inner.per_air.len()
            || child_vk
                .inner
                .per_air
                .get(child_history_air_id)
                .is_none_or(|air| air.params.num_public_values != width)
        {
            return Err("invalid recursive History chunk bridge profile");
        }
        let mut entries = Vec::new();
        let mut fixed_values = Vec::new();
        let mut child_history_start = None;
        for (air_id, air) in child_vk.inner.per_air.iter().enumerate() {
            let expected = child_expected_public_values
                .get(air_id)
                .ok_or("missing expected child public values")?;
            if expected.len() != air.params.num_public_values {
                return Err("expected child public-value shape differs from VK");
            }
            if air_id == child_history_air_id {
                child_history_start = Some(1 + entries.len());
            }
            for (pv_idx, &value) in expected.iter().enumerate() {
                entries.push(ChildPublicValueV19 { air_id, pv_idx });
                fixed_values.push((air_id != child_history_air_id).then_some(value));
            }
        }
        let child_history_start =
            child_history_start.ok_or("missing child History public values")?;
        let current_start = 1 + entries.len();
        Ok(Self {
            child_entries: entries.into(),
            child_fixed_values: fixed_values.into(),
            child_history_start,
            current_start,
            child_vk_pre_hash: child_vk.pre_hash,
            child_public_values_bus,
            child_pre_hash_bus,
            current_public_values_bus,
            current_segment_count,
        })
    }

    #[must_use]
    pub fn next_bus_idx(manager: &mut BusIndexManager) -> HistoryPublicValuesBusV19 {
        HistoryPublicValuesBusV19::new(manager.new_bus_idx())
    }

    #[must_use]
    pub fn generate_trace(
        &self,
        child_public_values: &[Vec<F>],
        current_public_values: &[F],
    ) -> Option<RowMajorMatrix<F>> {
        if child_public_values.len() == 0
            || current_public_values.len() != history_public_values_width_v19()
        {
            return None;
        }
        let child = self
            .child_entries
            .iter()
            .map(|entry| {
                child_public_values
                    .get(entry.air_id)?
                    .get(entry.pv_idx)
                    .copied()
            })
            .collect::<Option<Vec<_>>>()?;
        let width = self.width();
        let mut values = F::zero_vec(2 * width);
        values[0] = F::ONE;
        values[1..self.current_start].copy_from_slice(&child);
        values[self.current_start..self.current_start + current_public_values.len()]
            .copy_from_slice(current_public_values);
        Some(RowMajorMatrix::new(values, width))
    }

    fn equality_pairs(&self) -> Vec<(usize, usize)> {
        let child = self.child_history_start;
        let current = self.current_start;
        let mut pairs = Vec::with_capacity(2 + 5 * DIGEST_SIZE + 1);
        pairs.extend((0..DIGEST_SIZE).map(|i| (child + APP_START + i, current + APP_START + i)));
        pairs.extend((0..2).map(|i| {
            (
                child + FINAL_COUNT_START + i,
                current + INITIAL_COUNT_START + i,
            )
        }));
        for (final_start, initial_start) in [
            (FINAL_VM_START, INITIAL_VM_START),
            (FINAL_PRODUCT_START, INITIAL_PRODUCT_START),
            (FINAL_HISTORY_START, INITIAL_HISTORY_START),
            (FINAL_PUBLIC_VALUES_START, INITIAL_PUBLIC_VALUES_START),
        ] {
            pairs.extend(
                (0..DIGEST_SIZE).map(|i| (child + final_start + i, current + initial_start + i)),
            );
        }
        pairs.push((child + FINAL_TERMINATED, current + INITIAL_TERMINATED));
        pairs
    }
}

impl BaseAir<F> for HistoryChunkBridgeV19 {
    fn width(&self) -> usize {
        self.current_start + history_public_values_width_v19()
    }
}

impl BaseAirWithPublicValues<F> for HistoryChunkBridgeV19 {}
impl PartitionedBaseAir<F> for HistoryChunkBridgeV19 {}

impl<AB> Air<AB> for HistoryChunkBridgeV19
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("History chunk bridge row");
        let next = main.row_slice(1).expect("History chunk bridge padding row");
        builder.assert_bool(local[0]);
        builder.when_first_row().assert_one(local[0]);
        builder.when_last_row().assert_zero(local[0]);
        builder
            .when_transition()
            .assert_eq(local[0] - next[0], local[0]);

        for (position, entry) in self.child_entries.iter().enumerate() {
            self.child_public_values_bus.receive(
                builder,
                AB::Expr::ZERO,
                PublicValuesBusMessage {
                    air_idx: AB::Expr::from_usize(entry.air_id),
                    pv_idx: AB::Expr::from_usize(entry.pv_idx),
                    value: local[1 + position].into(),
                },
                local[0],
            );
            if let Some(expected) = self.child_fixed_values[position] {
                builder
                    .when(local[0])
                    .assert_eq(local[1 + position], expected);
            }
        }
        self.child_pre_hash_bus.receive(
            builder,
            AB::Expr::ZERO,
            PreHashMessage {
                vk_pre_hash: self.child_vk_pre_hash.map(AB::Expr::from),
            },
            local[0],
        );
        for index in 0..history_public_values_width_v19() {
            self.current_public_values_bus.receive(
                builder,
                AB::Expr::from_usize(index),
                local[self.current_start + index],
                local[0] * AB::F::from_usize(self.current_segment_count),
            );
        }
        for (left, right) in self.equality_pairs() {
            builder.when(local[0]).assert_eq(local[left], local[right]);
        }
        // Every child of another stage must be a non-terminal chunk. The last
        // chunk alone carries terminal_chunk = 1 in its direct History PVS.
        builder
            .when(local[0])
            .assert_zero(local[self.child_history_start + TERMINAL_CHUNK]);
    }
}

const _: () = assert!(TERMINAL_CHUNK + 1 + DIGEST_SIZE == 114);

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use openvm_recursion_circuit::bus::{PreHashBus, PublicValuesBus};
    use openvm_stark_backend::{
        air_builders::debug::check_constraints,
        p3_air::{AirBuilderWithPublicValues, BaseAirWithPublicValues},
        AirRef, StarkEngine,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2CpuEngine;

    use super::*;

    #[derive(Clone, Debug)]
    struct TestPublicAir(usize);

    impl BaseAir<F> for TestPublicAir {
        fn width(&self) -> usize {
            1
        }
    }

    impl BaseAirWithPublicValues<F> for TestPublicAir {
        fn num_public_values(&self) -> usize {
            self.0
        }
    }

    impl PartitionedBaseAir<F> for TestPublicAir {}

    impl<AB> Air<AB> for TestPublicAir
    where
        AB: AirBuilder<F = F> + AirBuilderWithPublicValues,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.row_slice(0).expect("test public AIR row");
            builder.assert_zero(local[0].clone());
        }
    }

    fn test_bridge() -> HistoryChunkBridgeV19 {
        let mut params =
            openvm_stark_sdk::config::native_warp_history_params_with_100_bits_security();
        params.max_constraint_degree = params.max_constraint_degree.max(2);
        let engine: BabyBearPoseidon2CpuEngine = BabyBearPoseidon2CpuEngine::new(params);
        let airs: Vec<AirRef<BabyBearPoseidon2Config>> = vec![
            Arc::new(TestPublicAir(1)),
            Arc::new(TestPublicAir(history_public_values_width_v19())),
        ];
        let (_, vk) = engine.keygen(&airs);
        let mut buses = BusIndexManager::new();
        HistoryChunkBridgeV19::new(
            &vk,
            1,
            &[
                vec![F::from_u32(7)],
                vec![F::ZERO; history_public_values_width_v19()],
            ],
            PublicValuesBus::new(buses.new_bus_idx()),
            PreHashBus::new(buses.new_bus_idx()),
            HistoryPublicValuesBusV19::new(buses.new_bus_idx()),
            1,
        )
        .unwrap()
    }

    fn check_bridge(bridge: &HistoryChunkBridgeV19, trace: &RowMajorMatrix<F>) {
        check_constraints::<_, BabyBearPoseidon2Config>(
            bridge,
            "History recursive fixed-PV bridge",
            &None,
            &[trace.as_view()],
            &[],
        );
    }

    #[test]
    fn recursive_bridge_rejects_tampered_non_history_public_value() {
        let bridge = test_bridge();
        let history = vec![F::ZERO; history_public_values_width_v19()];
        let honest_children = vec![vec![F::from_u32(7)], history.clone()];
        let honest = bridge.generate_trace(&honest_children, &history).unwrap();
        check_bridge(&bridge, &honest);

        let tampered_children = vec![vec![F::from_u32(8)], history.clone()];
        let tampered = bridge.generate_trace(&tampered_children, &history).unwrap();
        assert!(catch_unwind(AssertUnwindSafe(|| check_bridge(&bridge, &tampered))).is_err());
    }
}

//! AIR enforcing the ordered merge of two History-v3 intervals.

use core::borrow::Borrow;

use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::F;

use super::{
    VerifierWarpHistoryChunkIntervalBusV3, VerifierWarpHistoryChunkIntervalMessageV3,
    ACTIVE_CHILD_COUNT_AFTER_V3, ACTIVE_CHILD_COUNT_BEFORE_V3, CHUNK_INDEX_V3, END_BATCH_INDEX_V3,
    FINAL_ACCUMULATOR_DIGEST_V3, FINAL_MEMORY_ROOT_V3, FINAL_PC_V3, FIXED_BINDINGS_V3,
    HISTORY_HASH_AFTER_V3, HISTORY_HASH_BEFORE_V3, INITIAL_ACCUMULATOR_DIGEST_V3,
    INITIAL_MEMORY_ROOT_V3, INITIAL_PC_V3, IS_GENESIS_V3, IS_TERMINAL_V3,
    OBSERVATION_COUNT_AFTER_V3, OBSERVATION_COUNT_BEFORE_V3, PUBLIC_VALUES_DIGEST_V3,
    START_BATCH_INDEX_V3, TOTAL_BATCH_COUNT_V3, VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3,
};

const LIMB_BITS: usize = 16;
const LIMB_BASE: u32 = 1 << LIMB_BITS;

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct VerifierWarpHistoryChunkCompositionColsV3<T> {
    pub active: T,
    pub chunk_index_carry: T,
    pub left_chunk_index_bits: [[T; LIMB_BITS]; 2],
    pub right_chunk_index_bits: [[T; LIMB_BITS]; 2],
    pub left: VerifierWarpHistoryChunkIntervalMessageV3<T>,
    pub right: VerifierWarpHistoryChunkIntervalMessageV3<T>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifierWarpHistoryChunkCompositionOutputV3 {
    StandalonePublicValues,
    TypedBus {
        bus: VerifierWarpHistoryChunkIntervalBusV3,
        lookup_count: u32,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifierWarpHistoryChunkCompositionConfigErrorV3 {
    ReusedBusIndex,
    ZeroOutputLookupCount,
}

#[derive(Clone, Debug)]
pub struct VerifierWarpHistoryChunkCompositionAirV3 {
    left_child_bus: VerifierWarpHistoryChunkIntervalBusV3,
    right_child_bus: VerifierWarpHistoryChunkIntervalBusV3,
    output: VerifierWarpHistoryChunkCompositionOutputV3,
}

impl VerifierWarpHistoryChunkCompositionAirV3 {
    pub fn new_standalone(
        left_child_bus: VerifierWarpHistoryChunkIntervalBusV3,
        right_child_bus: VerifierWarpHistoryChunkIntervalBusV3,
    ) -> Result<Self, VerifierWarpHistoryChunkCompositionConfigErrorV3> {
        if left_child_bus.index() == right_child_bus.index() {
            return Err(VerifierWarpHistoryChunkCompositionConfigErrorV3::ReusedBusIndex);
        }
        Ok(Self {
            left_child_bus,
            right_child_bus,
            output: VerifierWarpHistoryChunkCompositionOutputV3::StandalonePublicValues,
        })
    }

    pub fn new_with_output_bus(
        left_child_bus: VerifierWarpHistoryChunkIntervalBusV3,
        right_child_bus: VerifierWarpHistoryChunkIntervalBusV3,
        output_bus: VerifierWarpHistoryChunkIntervalBusV3,
        lookup_count: u32,
    ) -> Result<Self, VerifierWarpHistoryChunkCompositionConfigErrorV3> {
        if left_child_bus.index() == right_child_bus.index()
            || left_child_bus.index() == output_bus.index()
            || right_child_bus.index() == output_bus.index()
        {
            return Err(VerifierWarpHistoryChunkCompositionConfigErrorV3::ReusedBusIndex);
        }
        if lookup_count == 0 {
            return Err(VerifierWarpHistoryChunkCompositionConfigErrorV3::ZeroOutputLookupCount);
        }
        Ok(Self {
            left_child_bus,
            right_child_bus,
            output: VerifierWarpHistoryChunkCompositionOutputV3::TypedBus {
                bus: output_bus,
                lookup_count,
            },
        })
    }

    #[must_use]
    pub const fn output_mode(&self) -> VerifierWarpHistoryChunkCompositionOutputV3 {
        self.output
    }

    fn merged_message<AB: AirBuilder<F = F>>(
        &self,
        left: &VerifierWarpHistoryChunkIntervalMessageV3<AB::Var>,
        right: &VerifierWarpHistoryChunkIntervalMessageV3<AB::Var>,
    ) -> VerifierWarpHistoryChunkIntervalMessageV3<AB::Expr>
    where
        AB::Var: Copy,
    {
        let fields = core::array::from_fn(|index| {
            let source = if FIXED_BINDINGS_V3.contains(&index)
                || TOTAL_BATCH_COUNT_V3.contains(&index)
                || START_BATCH_INDEX_V3.contains(&index)
                || ACTIVE_CHILD_COUNT_BEFORE_V3.contains(&index)
                || index == INITIAL_PC_V3
                || INITIAL_MEMORY_ROOT_V3.contains(&index)
                || INITIAL_ACCUMULATOR_DIGEST_V3.contains(&index)
                || HISTORY_HASH_BEFORE_V3.contains(&index)
                || OBSERVATION_COUNT_BEFORE_V3.contains(&index)
                || index == IS_GENESIS_V3
            {
                left.fields[index]
            } else {
                right.fields[index]
            };
            source.into()
        });
        VerifierWarpHistoryChunkIntervalMessageV3 { fields }
    }

    fn assert_equal_ranges<AB: AirBuilder<F = F>>(
        builder: &mut AB,
        enabled: AB::Expr,
        left: &VerifierWarpHistoryChunkIntervalMessageV3<AB::Var>,
        left_range: core::ops::Range<usize>,
        right: &VerifierWarpHistoryChunkIntervalMessageV3<AB::Var>,
        right_range: core::ops::Range<usize>,
    ) where
        AB::Var: Copy,
    {
        debug_assert_eq!(left_range.len(), right_range.len());
        for (left_index, right_index) in left_range.zip(right_range) {
            builder
                .when(enabled.clone())
                .assert_eq(left.fields[left_index], right.fields[right_index]);
        }
    }

    fn assert_u16_limbs<AB: AirBuilder<F = F>>(
        builder: &mut AB,
        enabled: AB::Expr,
        limbs: &[AB::Var; 2],
        bits: &[[AB::Var; LIMB_BITS]; 2],
    ) where
        AB::Var: Copy,
    {
        for (limb, limb_bits) in limbs.iter().copied().zip(bits) {
            let mut recomposed = AB::Expr::ZERO;
            let mut weight = AB::F::ONE;
            for bit in limb_bits.iter().copied() {
                builder.when(enabled.clone()).assert_bool(bit);
                recomposed += AB::Expr::from(bit) * weight;
                weight *= AB::F::TWO;
            }
            builder.when(enabled.clone()).assert_eq(limb, recomposed);
        }
    }
}

impl BaseAir<F> for VerifierWarpHistoryChunkCompositionAirV3 {
    fn width(&self) -> usize {
        core::mem::size_of::<VerifierWarpHistoryChunkCompositionColsV3<u8>>()
    }
}

impl BaseAirWithPublicValues<F> for VerifierWarpHistoryChunkCompositionAirV3 {
    fn num_public_values(&self) -> usize {
        match self.output {
            VerifierWarpHistoryChunkCompositionOutputV3::StandalonePublicValues => {
                VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3
            }
            VerifierWarpHistoryChunkCompositionOutputV3::TypedBus { .. } => 0,
        }
    }
}

impl PartitionedBaseAir<F> for VerifierWarpHistoryChunkCompositionAirV3 {}

impl<AB> Air<AB> for VerifierWarpHistoryChunkCompositionAirV3
where
    AB: AirBuilder<F = F> + AirBuilderWithPublicValues + InteractionBuilder,
    AB::Var: Copy,
    AB::PublicVar: Copy,
    AB::Expr: From<AB::PublicVar>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("History-v3 composition row");
        let next_row = main
            .row_slice(1)
            .expect("History-v3 composition padding row");
        let local: &VerifierWarpHistoryChunkCompositionColsV3<AB::Var> = (*local_row).borrow();
        let next: &VerifierWarpHistoryChunkCompositionColsV3<AB::Var> = (*next_row).borrow();
        let enabled = Into::<AB::Expr>::into(local.active);

        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_last_row().assert_zero(local.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.active);
        builder
            .when(enabled.clone())
            .assert_bool(local.chunk_index_carry);

        let left_chunk_index: &[AB::Var; 2] = local.left.fields[CHUNK_INDEX_V3.clone()]
            .try_into()
            .expect("fixed chunk-index width");
        let right_chunk_index: &[AB::Var; 2] = local.right.fields[CHUNK_INDEX_V3.clone()]
            .try_into()
            .expect("fixed chunk-index width");
        Self::assert_u16_limbs(
            builder,
            enabled.clone(),
            left_chunk_index,
            &local.left_chunk_index_bits,
        );
        Self::assert_u16_limbs(
            builder,
            enabled.clone(),
            right_chunk_index,
            &local.right_chunk_index_bits,
        );
        builder.when(enabled.clone()).assert_eq(
            left_chunk_index[0] + AB::Expr::ONE,
            right_chunk_index[0]
                + Into::<AB::Expr>::into(local.chunk_index_carry) * AB::Expr::from_u32(LIMB_BASE),
        );
        builder.when(enabled.clone()).assert_eq(
            right_chunk_index[1],
            left_chunk_index[1] + local.chunk_index_carry,
        );

        Self::assert_equal_ranges(
            builder,
            enabled.clone(),
            &local.left,
            FIXED_BINDINGS_V3,
            &local.right,
            FIXED_BINDINGS_V3,
        );
        Self::assert_equal_ranges(
            builder,
            enabled.clone(),
            &local.left,
            TOTAL_BATCH_COUNT_V3,
            &local.right,
            TOTAL_BATCH_COUNT_V3,
        );
        Self::assert_equal_ranges(
            builder,
            enabled.clone(),
            &local.left,
            END_BATCH_INDEX_V3,
            &local.right,
            START_BATCH_INDEX_V3,
        );
        Self::assert_equal_ranges(
            builder,
            enabled.clone(),
            &local.left,
            ACTIVE_CHILD_COUNT_AFTER_V3,
            &local.right,
            ACTIVE_CHILD_COUNT_BEFORE_V3,
        );
        builder.when(enabled.clone()).assert_eq(
            local.left.fields[FINAL_PC_V3],
            local.right.fields[INITIAL_PC_V3],
        );
        Self::assert_equal_ranges(
            builder,
            enabled.clone(),
            &local.left,
            FINAL_MEMORY_ROOT_V3,
            &local.right,
            INITIAL_MEMORY_ROOT_V3,
        );
        Self::assert_equal_ranges(
            builder,
            enabled.clone(),
            &local.left,
            FINAL_ACCUMULATOR_DIGEST_V3,
            &local.right,
            INITIAL_ACCUMULATOR_DIGEST_V3,
        );
        Self::assert_equal_ranges(
            builder,
            enabled.clone(),
            &local.left,
            HISTORY_HASH_AFTER_V3,
            &local.right,
            HISTORY_HASH_BEFORE_V3,
        );
        Self::assert_equal_ranges(
            builder,
            enabled.clone(),
            &local.left,
            OBSERVATION_COUNT_AFTER_V3,
            &local.right,
            OBSERVATION_COUNT_BEFORE_V3,
        );

        for flag in [
            local.left.fields[IS_GENESIS_V3],
            local.left.fields[IS_TERMINAL_V3],
            local.right.fields[IS_GENESIS_V3],
            local.right.fields[IS_TERMINAL_V3],
        ] {
            builder.when(enabled.clone()).assert_bool(flag);
        }
        builder
            .when(enabled.clone())
            .assert_zero(local.right.fields[IS_GENESIS_V3]);
        builder
            .when(enabled.clone())
            .assert_zero(local.left.fields[IS_TERMINAL_V3]);
        for index in PUBLIC_VALUES_DIGEST_V3 {
            builder
                .when(enabled.clone())
                .assert_zero(local.left.fields[index]);
            builder
                .when(enabled.clone() * (AB::Expr::ONE - local.right.fields[IS_TERMINAL_V3]))
                .assert_zero(local.right.fields[index]);
        }

        self.left_child_bus
            .lookup_key(builder, local.left.clone(), local.active);
        self.right_child_bus
            .lookup_key(builder, local.right.clone(), local.active);
        let merged = self.merged_message::<AB>(&local.left, &local.right);
        match self.output {
            VerifierWarpHistoryChunkCompositionOutputV3::StandalonePublicValues => {
                let public = builder.public_values().to_vec();
                assert_eq!(
                    public.len(),
                    VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3,
                    "History-v3 composition public-value width"
                );
                for (value, public_value) in merged.fields.into_iter().zip(public.iter().copied()) {
                    builder
                        .when(enabled.clone())
                        .assert_eq(value, AB::Expr::from(public_value));
                }
            }
            VerifierWarpHistoryChunkCompositionOutputV3::TypedBus { bus, lookup_count } => {
                bus.add_key_with_lookups(
                    builder,
                    merged,
                    enabled * AB::Expr::from_u32(lookup_count),
                );
            }
        }
    }
}

const _: () = assert!(
    core::mem::size_of::<VerifierWarpHistoryChunkCompositionColsV3<u8>>()
        == 2 + 4 * LIMB_BITS + 2 * VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3
);

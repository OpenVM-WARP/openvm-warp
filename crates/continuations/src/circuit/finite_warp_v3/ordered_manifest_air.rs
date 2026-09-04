use core::borrow::{Borrow, BorrowMut};

use openvm_recursion_circuit::bus::TranscriptBus;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, D_EF, F};

use super::{
    super::{
        FiniteWarpV3ManifestCallMessage, FiniteWarpV3ManifestReceiptMessage,
        FiniteWarpV3ReceiptBuses, FINITE_WARP_V3_MAX_CALLS, FINITE_WARP_V3_MAX_SOURCES,
    },
    OrderedManifestBuses, OrderedManifestCallSummaryMessage, OrderedManifestDerivedRecord,
    OrderedManifestError, OrderedManifestOccupancyMessage, OrderedManifestPrefixReceiptMessage,
    OrderedManifestProfile, OrderedManifestSourceReceiptMessage,
    OrderedManifestSourceSummaryMessage, MANIFEST_DIGEST_TAG, MANIFEST_END_TAG, MANIFEST_FINAL_TAG,
    MANIFEST_INVOCATION_TAG, MANIFEST_SOURCE_TAG, MANIFEST_START_TAG, OBSERVATION_DIGEST_TAG,
    OBSERVATION_END_TAG, OBSERVATION_FIELD_TAG, ORDERED_MANIFEST_PREFIX_PROTOCOL_VERSION,
    ORDERED_MANIFEST_PREFIX_SLOTS, ORDERED_MANIFEST_PROTOCOL_VERSION, SCHEDULE_CALL_TAG,
    SCHEDULE_DIGEST_TAG, SCHEDULE_END_TAG, SCHEDULE_START_TAG, SOURCE_ORDER_DIGEST_TAG,
};

const SOURCE_COUNT_MINUS_ONE_BITS: usize = 7;
const CALL_FRESH_COUNT_MINUS_ONE_BITS: usize = 6;
const CHILD_COUNT_MINUS_ONE_BITS: usize = 3;
const ARITY_SELECTOR_COUNT: usize = 6;
const ROOT_PAIR_COUNT: usize = 3;

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
struct DigestNonzeroCols<T> {
    selectors: [T; DIGEST_SIZE],
    inverses: [T; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
struct OrderedManifestHeaderCallCols<T> {
    active: T,
    source_start: T,
    source_count: T,
    input_arity: T,
    full_root: [T; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
struct OrderedManifestHeaderCols<T> {
    active: T,
    source_count: T,
    call_count: T,
    normalized_leaf_count: T,
    schedule_digest: [T; DIGEST_SIZE],
    manifest_digest: [T; DIGEST_SIZE],
    final_accumulator_digest: [T; DIGEST_SIZE],
    program_commitment: [T; DIGEST_SIZE],
    initial_pc: T,
    initial_root: [T; DIGEST_SIZE],
    final_pc: T,
    final_root: [T; DIGEST_SIZE],
    calls: [OrderedManifestHeaderCallCols<T>; FINITE_WARP_V3_MAX_CALLS],
    source_count_minus_one_bits: [T; SOURCE_COUNT_MINUS_ONE_BITS],
    schedule_nonzero: DigestNonzeroCols<T>,
    manifest_nonzero: DigestNonzeroCols<T>,
    accumulator_nonzero: DigestNonzeroCols<T>,
    root_pair_nonzero: [DigestNonzeroCols<T>; ROOT_PAIR_COUNT],
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
struct OrderedManifestCallCols<T> {
    active: T,
    is_last: T,
    call_index: T,
    global_call_count: T,
    global_source_count: T,
    source_start: T,
    source_count: T,
    input_arity: T,
    base_root: [T; DIGEST_SIZE],
    full_root: [T; DIGEST_SIZE],
    schedule_digest: [T; DIGEST_SIZE],
    source_order_digest: [T; DIGEST_SIZE],
    logup_alpha: [T; D_EF],
    logup_beta: [T; D_EF],
    expected_active_child_counts: [T; ORDERED_MANIFEST_PREFIX_SLOTS],
    slot_active: [T; ORDERED_MANIFEST_PREFIX_SLOTS],
    child_count_minus_one_bits: [[T; CHILD_COUNT_MINUS_ONE_BITS]; ORDERED_MANIFEST_PREFIX_SLOTS],
    source_count_minus_one_bits: [T; CALL_FRESH_COUNT_MINUS_ONE_BITS],
    arity_selectors: [T; ARITY_SELECTOR_COUNT],
    base_root_nonzero: DigestNonzeroCols<T>,
    full_root_nonzero: DigestNonzeroCols<T>,
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
struct OrderedManifestSourceCols<T> {
    active: T,
    is_last: T,
    source_index: T,
    global_source_count: T,
    global_call_count: T,
    call_index: T,
    fresh_index_in_call: T,
    normalized_leaf_start: T,
    active_child_count: T,
    child_count_minus_one_bits: [T; CHILD_COUNT_MINUS_ONE_BITS],
    program_commitment: [T; DIGEST_SIZE],
    initial_pc: T,
    initial_root: [T; DIGEST_SIZE],
    final_pc: T,
    final_root: [T; DIGEST_SIZE],
    exit_code: T,
    is_terminate: T,
    source_instance_digest: [T; DIGEST_SIZE],
    chain_initial_pc: T,
    chain_initial_root: [T; DIGEST_SIZE],
    program_nonzero: DigestNonzeroCols<T>,
    instance_nonzero: DigestNonzeroCols<T>,
}

#[derive(Clone, Debug)]
pub struct OrderedManifestHeaderAir {
    profile: OrderedManifestProfile,
    wrapper_buses: FiniteWarpV3ReceiptBuses,
    buses: OrderedManifestBuses,
    transcript_bus: TranscriptBus,
}

impl OrderedManifestHeaderAir {
    pub(crate) fn new(
        profile: OrderedManifestProfile,
        wrapper_buses: FiniteWarpV3ReceiptBuses,
        buses: OrderedManifestBuses,
        transcript_bus: TranscriptBus,
    ) -> Self {
        Self {
            profile,
            wrapper_buses,
            buses,
            transcript_bus,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OrderedManifestCallAir {
    profile: OrderedManifestProfile,
    buses: OrderedManifestBuses,
    transcript_bus: TranscriptBus,
}

impl OrderedManifestCallAir {
    pub(crate) fn new(
        profile: OrderedManifestProfile,
        buses: OrderedManifestBuses,
        transcript_bus: TranscriptBus,
    ) -> Self {
        Self {
            profile,
            buses,
            transcript_bus,
        }
    }
}

#[derive(Clone, Debug)]
pub struct OrderedManifestSourceAir {
    profile: OrderedManifestProfile,
    buses: OrderedManifestBuses,
    transcript_bus: TranscriptBus,
}

impl OrderedManifestSourceAir {
    pub(crate) fn new(
        profile: OrderedManifestProfile,
        buses: OrderedManifestBuses,
        transcript_bus: TranscriptBus,
    ) -> Self {
        Self {
            profile,
            buses,
            transcript_bus,
        }
    }
}

macro_rules! impl_air_basics {
    ($air:ty, $cols:ty) => {
        impl BaseAir<F> for $air {
            fn width(&self) -> usize {
                core::mem::size_of::<$cols>()
            }
        }

        impl BaseAirWithPublicValues<F> for $air {}
        impl PartitionedBaseAir<F> for $air {}
    };
}

impl_air_basics!(OrderedManifestHeaderAir, OrderedManifestHeaderCols<u8>);
impl_air_basics!(OrderedManifestCallAir, OrderedManifestCallCols<u8>);
impl_air_basics!(OrderedManifestSourceAir, OrderedManifestSourceCols<u8>);

impl<AB> Air<AB> for OrderedManifestHeaderAir
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("ordered manifest header row");
        let next_row = main.row_slice(1).expect("ordered manifest header padding");
        let local: &OrderedManifestHeaderCols<AB::Var> = (*local_row).borrow();
        let next: &OrderedManifestHeaderCols<AB::Var> = (*next_row).borrow();
        let active = expr::<AB>(local.active);

        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_last_row().assert_zero(local.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.active);
        for value in local.as_slice().iter().skip(1) {
            builder
                .when(AB::Expr::ONE - active.clone())
                .assert_zero(*value);
        }

        constrain_minus_one_bits::<AB>(
            builder,
            local.source_count,
            &local.source_count_minus_one_bits,
            SOURCE_COUNT_MINUS_ONE_BITS,
            active.clone(),
        );
        builder.when(active.clone()).assert_eq(
            local.normalized_leaf_count,
            AB::Expr::from_u32(self.profile.expected_normalized_leaf_count),
        );
        constrain_digest_nonzero::<AB>(
            builder,
            &local.schedule_digest,
            &local.schedule_nonzero,
            active.clone(),
        );
        constrain_digest_nonzero::<AB>(
            builder,
            &local.manifest_digest,
            &local.manifest_nonzero,
            active.clone(),
        );
        constrain_digest_nonzero::<AB>(
            builder,
            &local.final_accumulator_digest,
            &local.accumulator_nonzero,
            active.clone(),
        );

        let mut call_active_sum = AB::Expr::ZERO;
        for (index, call) in local.calls.iter().enumerate() {
            let call_active = expr::<AB>(call.active);
            builder.assert_bool(call.active);
            if index == 0 {
                builder.when(active.clone()).assert_one(call.active);
            } else {
                builder.when(active.clone()).assert_zero(
                    call_active.clone()
                        * (AB::Expr::ONE - expr::<AB>(local.calls[index - 1].active)),
                );
            }
            for value in call.as_slice().iter().skip(1) {
                builder
                    .when(AB::Expr::ONE - call_active.clone())
                    .assert_zero(*value);
            }
            call_active_sum += call_active.clone();
            self.buses.call_summary.lookup_key(
                builder,
                OrderedManifestCallSummaryMessage {
                    schedule_digest: local.schedule_digest.map(Into::into),
                    call_count: local.call_count.into(),
                    total_source_count: local.source_count.into(),
                    call_index: AB::Expr::from_usize(index),
                    source_start: call.source_start.into(),
                    source_count: call.source_count.into(),
                    input_arity: call.input_arity.into(),
                    full_root: call.full_root.map(Into::into),
                },
                call_active,
            );
        }
        builder
            .when(active.clone())
            .assert_eq(call_active_sum, local.call_count);

        for (pair_index, (left, right)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
            let pair_active =
                expr::<AB>(local.calls[left].active) * expr::<AB>(local.calls[right].active);
            constrain_digest_difference::<AB>(
                builder,
                &local.calls[left].full_root,
                &local.calls[right].full_root,
                &local.root_pair_nonzero[pair_index],
                pair_active,
            );
        }

        self.buses.source_summary.lookup_key(
            builder,
            OrderedManifestSourceSummaryMessage {
                source_count: local.source_count.into(),
                call_count: local.call_count.into(),
                normalized_leaf_count: local.normalized_leaf_count.into(),
                program_commitment: local.program_commitment.map(Into::into),
                initial_pc: local.initial_pc.into(),
                initial_root: local.initial_root.map(Into::into),
                final_pc: local.final_pc.into(),
                final_root: local.final_root.map(Into::into),
            },
            active.clone(),
        );

        let manifest_message = FiniteWarpV3ManifestReceiptMessage {
            protocol_digest: constant_digest::<AB>(self.profile.protocol_digest),
            relation_digest: constant_digest::<AB>(self.profile.relation_digest),
            warp_index_digest: constant_digest::<AB>(self.profile.warp_index_digest),
            verifier_component_digest: constant_digest::<AB>(
                self.profile.verifier_component_digest,
            ),
            schedule_digest: local.schedule_digest.map(Into::into),
            manifest_digest: local.manifest_digest.map(Into::into),
            source_count: local.source_count.into(),
            call_count: local.call_count.into(),
            program_commitment: local.program_commitment.map(Into::into),
            initial_pc: local.initial_pc.into(),
            initial_root: local.initial_root.map(Into::into),
            final_pc: local.final_pc.into(),
            final_root: local.final_root.map(Into::into),
            final_accumulator_digest: local.final_accumulator_digest.map(Into::into),
            calls: core::array::from_fn(|index| {
                let call = &local.calls[index];
                FiniteWarpV3ManifestCallMessage {
                    active: call.active.into(),
                    source_start: call.source_start.into(),
                    source_count: call.source_count.into(),
                    input_arity: call.input_arity.into(),
                    fresh_stacked_root: call.full_root.map(Into::into),
                }
            }),
        };
        self.wrapper_buses
            .manifest
            .add_key_with_lookups(builder, manifest_message, active.clone());

        emit_schedule_header_transcript::<AB>(builder, self.transcript_bus, local, active.clone());
        emit_manifest_header_transcript::<AB>(builder, self.transcript_bus, local, active);
    }
}

impl<AB> Air<AB> for OrderedManifestCallAir
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("ordered manifest call row");
        let next_row = main.row_slice(1).expect("ordered manifest call next row");
        let local: &OrderedManifestCallCols<AB::Var> = (*local_row).borrow();
        let next: &OrderedManifestCallCols<AB::Var> = (*next_row).borrow();
        let active = expr::<AB>(local.active);
        let is_last = expr::<AB>(local.is_last);

        builder.assert_bool(local.active);
        builder.assert_bool(local.is_last);
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_zero(local.call_index);
        builder.when_first_row().assert_zero(local.source_start);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .assert_eq(local.is_last, local.active - next.active);
        builder
            .when_last_row()
            .assert_eq(local.is_last, local.active);
        for value in local.as_slice().iter().skip(2) {
            builder
                .when(AB::Expr::ONE - active.clone())
                .assert_zero(*value);
        }

        let continue_next = expr::<AB>(next.active);
        builder
            .when_transition()
            .when(continue_next.clone())
            .assert_eq(
                next.call_index,
                expr::<AB>(local.call_index) + AB::Expr::ONE,
            );
        builder
            .when_transition()
            .when(continue_next.clone())
            .assert_eq(
                next.source_start,
                expr::<AB>(local.source_start) + local.source_count,
            );
        builder
            .when_transition()
            .when(continue_next.clone())
            .assert_eq(next.global_call_count, local.global_call_count);
        builder
            .when_transition()
            .when(continue_next)
            .assert_eq(next.global_source_count, local.global_source_count);

        builder.when(is_last.clone()).assert_eq(
            expr::<AB>(local.call_index) + AB::Expr::ONE,
            local.global_call_count,
        );
        builder.when(is_last.clone()).assert_eq(
            expr::<AB>(local.source_start) + local.source_count,
            local.global_source_count,
        );

        constrain_minus_one_bits::<AB>(
            builder,
            local.source_count,
            &local.source_count_minus_one_bits,
            CALL_FRESH_COUNT_MINUS_ONE_BITS,
            active.clone(),
        );
        let mut selector_sum = AB::Expr::ZERO;
        let mut selected_arity = AB::Expr::ZERO;
        for (index, selector) in local.arity_selectors.iter().enumerate() {
            builder.assert_bool(*selector);
            selector_sum += *selector;
            selected_arity += AB::Expr::from_u32(1u32 << (index + 1)) * *selector;
        }
        builder
            .when(active.clone())
            .assert_eq(selector_sum, active.clone());
        builder
            .when(active.clone())
            .assert_eq(selected_arity, local.input_arity);
        let prior = active.clone() * expr::<AB>(local.call_index);
        // Calls are contiguous and there are at most three. `call_index != 0`
        // is therefore `call_index * (3-call_index) / 2` on {0,1,2}.
        let prior = prior * (AB::Expr::from_u32(3) - local.call_index) * AB::F::TWO.inverse();
        builder
            .when(active.clone())
            .assert_eq(local.input_arity, expr::<AB>(local.source_count) + prior);
        constrain_digest_nonzero::<AB>(
            builder,
            &local.base_root,
            &local.base_root_nonzero,
            active.clone(),
        );
        constrain_digest_nonzero::<AB>(
            builder,
            &local.full_root,
            &local.full_root_nonzero,
            active.clone(),
        );

        let child_bits_used = self.profile.max_active_children_per_source.ilog2() as usize;
        let mut occupied = AB::Expr::ZERO;
        for slot in 0..ORDERED_MANIFEST_PREFIX_SLOTS {
            let slot_active = expr::<AB>(local.slot_active[slot]);
            builder.assert_bool(local.slot_active[slot]);
            if slot == 0 {
                builder
                    .when(active.clone())
                    .assert_eq(local.slot_active[slot], local.active);
            } else {
                builder.when(active.clone()).assert_zero(
                    slot_active.clone() * (AB::Expr::ONE - expr::<AB>(local.slot_active[slot - 1])),
                );
            }
            occupied += slot_active.clone();
            for (bit_index, bit) in local.child_count_minus_one_bits[slot].iter().enumerate() {
                builder.assert_bool(*bit);
                builder
                    .when(AB::Expr::ONE - slot_active.clone())
                    .assert_zero(*bit);
                if bit_index >= child_bits_used {
                    builder.when(active.clone()).assert_zero(*bit);
                }
            }
            let child_count =
                AB::Expr::ONE + recompose_bits::<AB>(&local.child_count_minus_one_bits[slot]);
            builder
                .when(slot_active.clone())
                .assert_eq(local.expected_active_child_counts[slot], child_count);
            builder
                .when(AB::Expr::ONE - slot_active.clone())
                .assert_zero(local.expected_active_child_counts[slot]);
            self.buses.occupancy.lookup_key(
                builder,
                OrderedManifestOccupancyMessage {
                    source_index: expr::<AB>(local.source_start) + AB::Expr::from_usize(slot),
                    call_index: local.call_index.into(),
                    fresh_index_in_call: AB::Expr::from_usize(slot),
                    active_child_count: local.expected_active_child_counts[slot].into(),
                },
                slot_active,
            );
        }
        builder
            .when(active.clone())
            .assert_eq(occupied, local.source_count);

        let prefix = prefix_message::<AB>(&self.profile, local);
        self.buses
            .prefix_authority
            .lookup_key(builder, prefix.clone(), active.clone());
        self.buses
            .prefix_binding
            .add_key_with_lookups(builder, prefix, active.clone());
        self.buses.call_summary.add_key_with_lookups(
            builder,
            OrderedManifestCallSummaryMessage {
                schedule_digest: local.schedule_digest.map(Into::into),
                call_count: local.global_call_count.into(),
                total_source_count: local.global_source_count.into(),
                call_index: local.call_index.into(),
                source_start: local.source_start.into(),
                source_count: local.source_count.into(),
                input_arity: local.input_arity.into(),
                full_root: local.full_root.map(Into::into),
            },
            active.clone(),
        );

        emit_schedule_call_transcript::<AB>(builder, self.transcript_bus, local, active.clone());
        emit_source_order_header_transcript::<AB>(
            builder,
            self.transcript_bus,
            local,
            active.clone(),
        );
        emit_manifest_call_transcript::<AB>(builder, self.transcript_bus, local, active);
    }
}

impl<AB> Air<AB> for OrderedManifestSourceAir
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("ordered manifest source row");
        let next_row = main.row_slice(1).expect("ordered manifest source next row");
        let local: &OrderedManifestSourceCols<AB::Var> = (*local_row).borrow();
        let next: &OrderedManifestSourceCols<AB::Var> = (*next_row).borrow();
        let active = expr::<AB>(local.active);
        let is_last = expr::<AB>(local.is_last);

        builder.assert_bool(local.active);
        builder.assert_bool(local.is_last);
        builder.assert_bool(local.is_terminate);
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_zero(local.source_index);
        builder
            .when_first_row()
            .assert_zero(local.normalized_leaf_start);
        builder
            .when_first_row()
            .assert_eq(local.chain_initial_pc, local.initial_pc);
        for limb in 0..DIGEST_SIZE {
            builder
                .when_first_row()
                .assert_eq(local.chain_initial_root[limb], local.initial_root[limb]);
        }
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .assert_eq(local.is_last, local.active - next.active);
        builder
            .when_last_row()
            .assert_eq(local.is_last, local.active);
        for value in local.as_slice().iter().skip(2) {
            builder
                .when(AB::Expr::ONE - active.clone())
                .assert_zero(*value);
        }

        let continue_next = expr::<AB>(next.active);
        builder
            .when_transition()
            .when(continue_next.clone())
            .assert_eq(
                next.source_index,
                expr::<AB>(local.source_index) + AB::Expr::ONE,
            );
        builder
            .when_transition()
            .when(continue_next.clone())
            .assert_eq(
                next.normalized_leaf_start,
                expr::<AB>(local.normalized_leaf_start) + local.active_child_count,
            );
        for limb in 0..DIGEST_SIZE {
            builder
                .when_transition()
                .when(continue_next.clone())
                .assert_eq(
                    next.program_commitment[limb],
                    local.program_commitment[limb],
                );
        }
        builder
            .when_transition()
            .when(continue_next.clone())
            .assert_eq(next.initial_pc, local.final_pc);
        for limb in 0..DIGEST_SIZE {
            builder
                .when_transition()
                .when(continue_next.clone())
                .assert_eq(next.initial_root[limb], local.final_root[limb]);
        }
        builder
            .when_transition()
            .when(continue_next.clone())
            .assert_eq(next.global_source_count, local.global_source_count);
        builder
            .when_transition()
            .when(continue_next.clone())
            .assert_eq(next.global_call_count, local.global_call_count);
        builder
            .when_transition()
            .when(continue_next.clone())
            .assert_eq(next.chain_initial_pc, local.chain_initial_pc);
        for limb in 0..DIGEST_SIZE {
            builder
                .when_transition()
                .when(continue_next.clone())
                .assert_eq(
                    next.chain_initial_root[limb],
                    local.chain_initial_root[limb],
                );
        }

        let child_bits_used = self.profile.max_active_children_per_source.ilog2() as usize;
        for (bit_index, bit) in local.child_count_minus_one_bits.iter().enumerate() {
            builder.assert_bool(*bit);
            if bit_index >= child_bits_used {
                builder.when(active.clone()).assert_zero(*bit);
            }
        }
        builder.when(active.clone()).assert_eq(
            local.active_child_count,
            AB::Expr::ONE + recompose_bits::<AB>(&local.child_count_minus_one_bits),
        );
        constrain_digest_nonzero::<AB>(
            builder,
            &local.program_commitment,
            &local.program_nonzero,
            active.clone(),
        );
        constrain_digest_nonzero::<AB>(
            builder,
            &local.source_instance_digest,
            &local.instance_nonzero,
            active.clone(),
        );

        let nonfinal = active.clone() - is_last.clone();
        builder.when(nonfinal).assert_zero(local.is_terminate);
        builder.when(active.clone() - is_last.clone()).assert_eq(
            local.exit_code,
            AB::Expr::from_u32(self.profile.suspend_exit_code),
        );
        builder.when(is_last.clone()).assert_one(local.is_terminate);
        builder.when(is_last.clone()).assert_zero(local.exit_code);
        builder.when(is_last.clone()).assert_eq(
            expr::<AB>(local.source_index) + AB::Expr::ONE,
            local.global_source_count,
        );
        builder.when(is_last.clone()).assert_eq(
            expr::<AB>(local.normalized_leaf_start) + local.active_child_count,
            AB::Expr::from_u32(self.profile.expected_normalized_leaf_count),
        );

        self.buses.source_authority.lookup_key(
            builder,
            OrderedManifestSourceReceiptMessage {
                protocol_digest: constant_digest::<AB>(self.profile.protocol_digest),
                relation_digest: constant_digest::<AB>(self.profile.relation_digest),
                warp_index_digest: constant_digest::<AB>(self.profile.warp_index_digest),
                source_index: local.source_index.into(),
                call_index: local.call_index.into(),
                fresh_index_in_call: local.fresh_index_in_call.into(),
                normalized_leaf_start: local.normalized_leaf_start.into(),
                active_child_count: local.active_child_count.into(),
                program_commitment: local.program_commitment.map(Into::into),
                initial_pc: local.initial_pc.into(),
                initial_root: local.initial_root.map(Into::into),
                final_pc: local.final_pc.into(),
                final_root: local.final_root.map(Into::into),
                exit_code: local.exit_code.into(),
                is_terminate: local.is_terminate.into(),
                source_instance_digest: local.source_instance_digest.map(Into::into),
            },
            active.clone(),
        );
        self.buses.occupancy.add_key_with_lookups(
            builder,
            OrderedManifestOccupancyMessage {
                source_index: local.source_index.into(),
                call_index: local.call_index.into(),
                fresh_index_in_call: local.fresh_index_in_call.into(),
                active_child_count: local.active_child_count.into(),
            },
            active.clone(),
        );
        self.buses.source_summary.add_key_with_lookups(
            builder,
            OrderedManifestSourceSummaryMessage {
                source_count: local.global_source_count.into(),
                call_count: local.global_call_count.into(),
                normalized_leaf_count: AB::Expr::from_u32(
                    self.profile.expected_normalized_leaf_count,
                ),
                program_commitment: local.program_commitment.map(Into::into),
                initial_pc: local.chain_initial_pc.into(),
                initial_root: local.chain_initial_root.map(Into::into),
                final_pc: local.final_pc.into(),
                final_root: local.final_root.map(Into::into),
            },
            is_last.clone(),
        );

        emit_source_order_source_transcript::<AB>(
            builder,
            self.transcript_bus,
            local,
            active.clone(),
        );
        emit_manifest_source_transcript::<AB>(builder, self.transcript_bus, local, active);
    }
}

pub(crate) fn generate_ordered_manifest_main_traces(
    profile: &OrderedManifestProfile,
    derived: &OrderedManifestDerivedRecord,
) -> Result<(RowMajorMatrix<F>, RowMajorMatrix<F>, RowMajorMatrix<F>), OrderedManifestError> {
    profile.validate()?;
    let record = &derived.record;
    let header_width = core::mem::size_of::<OrderedManifestHeaderCols<u8>>();
    let mut header_values = vec![F::ZERO; 2 * header_width];
    let header: &mut OrderedManifestHeaderCols<F> = header_values[..header_width].borrow_mut();
    header.active = F::ONE;
    header.source_count = F::from_usize(record.sources.len());
    header.call_count = F::from_usize(record.calls.len());
    header.normalized_leaf_count = F::from_u32(profile.expected_normalized_leaf_count);
    header.schedule_digest = derived.schedule_digest;
    header.manifest_digest = derived.manifest_digest;
    header.final_accumulator_digest = record.final_accumulator_digest;
    let first = record
        .sources
        .first()
        .ok_or(OrderedManifestError::EmptySources)?;
    let last = record
        .sources
        .last()
        .ok_or(OrderedManifestError::EmptySources)?;
    header.program_commitment = first.program_commitment;
    header.initial_pc = first.initial_state.pc;
    header.initial_root = first.initial_state.memory_root;
    header.final_pc = last.final_state.pc;
    header.final_root = last.final_state.memory_root;
    write_minus_one_bits(
        record.sources.len() as u32,
        &mut header.source_count_minus_one_bits,
    );
    fill_digest_nonzero(derived.schedule_digest, &mut header.schedule_nonzero)?;
    fill_digest_nonzero(derived.manifest_digest, &mut header.manifest_nonzero)?;
    fill_digest_nonzero(
        record.final_accumulator_digest,
        &mut header.accumulator_nonzero,
    )?;
    for (dst, call) in header.calls.iter_mut().zip(&record.calls) {
        dst.active = F::ONE;
        dst.source_start = F::from_u32(call.source_start);
        dst.source_count = F::from_u32(call.source_count);
        dst.input_arity = F::from_u32(call.input_arity);
        dst.full_root = call.full_root;
    }
    for (pair_index, (left, right)) in [(0, 1), (0, 2), (1, 2)].into_iter().enumerate() {
        if right < record.calls.len() {
            fill_digest_difference(
                record.calls[left].full_root,
                record.calls[right].full_root,
                &mut header.root_pair_nonzero[pair_index],
            )?;
        }
    }

    let call_width = core::mem::size_of::<OrderedManifestCallCols<u8>>();
    let mut call_values = vec![F::ZERO; 4 * call_width];
    for (row, call) in record.calls.iter().enumerate() {
        let cols: &mut OrderedManifestCallCols<F> =
            call_values[row * call_width..(row + 1) * call_width].borrow_mut();
        cols.active = F::ONE;
        cols.is_last = F::from_bool(row + 1 == record.calls.len());
        cols.call_index = F::from_u32(call.call_index);
        cols.global_call_count = F::from_usize(record.calls.len());
        cols.global_source_count = F::from_usize(record.sources.len());
        cols.source_start = F::from_u32(call.source_start);
        cols.source_count = F::from_u32(call.source_count);
        cols.input_arity = F::from_u32(call.input_arity);
        cols.base_root = call.base_root;
        cols.full_root = call.full_root;
        cols.schedule_digest = derived.schedule_digest;
        cols.source_order_digest = derived.source_order_digests[row];
        cols.logup_alpha
            .copy_from_slice(call.logup_alpha.as_basis_coefficients_slice());
        cols.logup_beta
            .copy_from_slice(call.logup_beta.as_basis_coefficients_slice());
        write_minus_one_bits(call.source_count, &mut cols.source_count_minus_one_bits);
        cols.arity_selectors[call.input_arity.trailing_zeros() as usize - 1] = F::ONE;
        fill_digest_nonzero(call.base_root, &mut cols.base_root_nonzero)?;
        fill_digest_nonzero(call.full_root, &mut cols.full_root_nonzero)?;
        let start = call.source_start as usize;
        let end = start + call.source_count as usize;
        for (slot, source) in record.sources[start..end].iter().enumerate() {
            cols.slot_active[slot] = F::ONE;
            cols.expected_active_child_counts[slot] = F::from_u8(source.active_child_count);
            write_minus_one_bits(
                u32::from(source.active_child_count),
                &mut cols.child_count_minus_one_bits[slot],
            );
        }
    }

    let source_width = core::mem::size_of::<OrderedManifestSourceCols<u8>>();
    let source_height = (FINITE_WARP_V3_MAX_SOURCES as usize).next_power_of_two();
    let mut source_values = vec![F::ZERO; source_height * source_width];
    for (row, source) in record.sources.iter().enumerate() {
        let cols: &mut OrderedManifestSourceCols<F> =
            source_values[row * source_width..(row + 1) * source_width].borrow_mut();
        cols.active = F::ONE;
        cols.is_last = F::from_bool(row + 1 == record.sources.len());
        cols.source_index = F::from_u32(source.source_index);
        cols.global_source_count = F::from_usize(record.sources.len());
        cols.global_call_count = F::from_usize(record.calls.len());
        cols.call_index = F::from_u32(source.call_index);
        cols.fresh_index_in_call = F::from_u32(source.fresh_index_in_call);
        cols.normalized_leaf_start = F::from_u32(source.normalized_leaf_start);
        cols.active_child_count = F::from_u8(source.active_child_count);
        write_minus_one_bits(
            u32::from(source.active_child_count),
            &mut cols.child_count_minus_one_bits,
        );
        cols.program_commitment = source.program_commitment;
        cols.initial_pc = source.initial_state.pc;
        cols.initial_root = source.initial_state.memory_root;
        cols.final_pc = source.final_state.pc;
        cols.final_root = source.final_state.memory_root;
        cols.exit_code = source.exit_code;
        cols.is_terminate = source.is_terminate;
        cols.source_instance_digest = source.source_instance_digest;
        cols.chain_initial_pc = first.initial_state.pc;
        cols.chain_initial_root = first.initial_state.memory_root;
        fill_digest_nonzero(source.program_commitment, &mut cols.program_nonzero)?;
        fill_digest_nonzero(source.source_instance_digest, &mut cols.instance_nonzero)?;
    }
    Ok((
        RowMajorMatrix::new(header_values, header_width),
        RowMajorMatrix::new(call_values, call_width),
        RowMajorMatrix::new(source_values, source_width),
    ))
}

fn expr<AB: AirBuilder<F = F>>(value: AB::Var) -> AB::Expr
where
    AB::Var: Copy,
{
    value.into()
}

fn constant<AB: AirBuilder<F = F>>(value: F) -> AB::Expr {
    AB::Expr::from_u32(value.as_canonical_u32())
}

fn constant_digest<AB: AirBuilder<F = F>>(digest: Digest) -> [AB::Expr; DIGEST_SIZE] {
    digest.map(constant::<AB>)
}

fn recompose_bits<AB: AirBuilder<F = F>>(bits: &[AB::Var]) -> AB::Expr
where
    AB::Var: Copy,
{
    bits.iter()
        .enumerate()
        .fold(AB::Expr::ZERO, |sum, (index, bit)| {
            sum + AB::Expr::from_u32(1u32 << index) * *bit
        })
}

fn constrain_minus_one_bits<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    value: AB::Var,
    bits: &[AB::Var],
    used_bits: usize,
    enabled: AB::Expr,
) where
    AB::Var: Copy,
{
    for (index, bit) in bits.iter().enumerate() {
        builder.assert_bool(*bit);
        if index >= used_bits {
            builder.when(enabled.clone()).assert_zero(*bit);
        }
    }
    builder
        .when(enabled)
        .assert_eq(value, AB::Expr::ONE + recompose_bits::<AB>(bits));
}

fn constrain_digest_nonzero<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    digest: &[AB::Var; DIGEST_SIZE],
    witness: &DigestNonzeroCols<AB::Var>,
    enabled: AB::Expr,
) where
    AB::Var: Copy,
{
    let mut selector_sum = AB::Expr::ZERO;
    for limb in 0..DIGEST_SIZE {
        builder.assert_bool(witness.selectors[limb]);
        selector_sum += witness.selectors[limb];
        builder.assert_zero(
            expr::<AB>(witness.selectors[limb])
                * (expr::<AB>(digest[limb]) * witness.inverses[limb] - AB::Expr::ONE),
        );
        builder.assert_zero((AB::Expr::ONE - witness.selectors[limb]) * witness.inverses[limb]);
    }
    builder.assert_eq(selector_sum, enabled);
}

fn constrain_digest_difference<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    left: &[AB::Var; DIGEST_SIZE],
    right: &[AB::Var; DIGEST_SIZE],
    witness: &DigestNonzeroCols<AB::Var>,
    enabled: AB::Expr,
) where
    AB::Var: Copy,
{
    let mut selector_sum = AB::Expr::ZERO;
    for limb in 0..DIGEST_SIZE {
        builder.assert_bool(witness.selectors[limb]);
        selector_sum += witness.selectors[limb];
        builder.assert_zero(
            expr::<AB>(witness.selectors[limb])
                * ((expr::<AB>(left[limb]) - right[limb]) * witness.inverses[limb] - AB::Expr::ONE),
        );
        builder.assert_zero((AB::Expr::ONE - witness.selectors[limb]) * witness.inverses[limb]);
    }
    builder.assert_eq(selector_sum, enabled);
}

fn write_minus_one_bits<const N: usize>(value: u32, bits: &mut [F; N]) {
    let value = value - 1;
    for (index, bit) in bits.iter_mut().enumerate() {
        *bit = F::from_bool(((value >> index) & 1) != 0);
    }
}

fn fill_digest_nonzero(
    digest: Digest,
    witness: &mut DigestNonzeroCols<F>,
) -> Result<(), OrderedManifestError> {
    let index = digest
        .iter()
        .position(|value| *value != F::ZERO)
        .ok_or(OrderedManifestError::ZeroDigest("digest"))?;
    witness.selectors[index] = F::ONE;
    witness.inverses[index] = digest[index].inverse();
    Ok(())
}

fn fill_digest_difference(
    left: Digest,
    right: Digest,
    witness: &mut DigestNonzeroCols<F>,
) -> Result<(), OrderedManifestError> {
    let index = (0..DIGEST_SIZE)
        .find(|&limb| left[limb] != right[limb])
        .ok_or(OrderedManifestError::DuplicateRoot {
            first: 0,
            second: 0,
        })?;
    witness.selectors[index] = F::ONE;
    witness.inverses[index] = (left[index] - right[index]).inverse();
    Ok(())
}

fn prefix_message<AB: AirBuilder<F = F>>(
    profile: &OrderedManifestProfile,
    local: &OrderedManifestCallCols<AB::Var>,
) -> OrderedManifestPrefixReceiptMessage<AB::Expr>
where
    AB::Var: Copy,
{
    let trace_len = profile.prefix.trace_prefix_len;
    let active_start = profile.prefix.active_count_block_start;
    OrderedManifestPrefixReceiptMessage {
        protocol_version: AB::Expr::from_u32(ORDERED_MANIFEST_PREFIX_PROTOCOL_VERSION),
        relation_digest: constant_digest::<AB>(profile.relation_digest),
        index_digest: constant_digest::<AB>(profile.prefix.index_digest),
        schedule_digest: local.schedule_digest.map(Into::into),
        source_order_digest: local.source_order_digest.map(Into::into),
        call_index: local.call_index.into(),
        source_start: local.source_start.into(),
        source_count: local.source_count.into(),
        expected_active_child_counts: local.expected_active_child_counts.map(Into::into),
        l_skip: AB::Expr::from_u32(profile.prefix.l_skip),
        log_message_len: AB::Expr::from_u32(profile.prefix.log_message_len),
        log_blowup: AB::Expr::from_u32(profile.prefix.log_blowup),
        log_codeword_len: AB::Expr::from_u32(profile.prefix.log_codeword_len),
        rows_per_leaf: AB::Expr::from_u32(profile.prefix.rows_per_leaf),
        trace_prefix_len_lo: AB::Expr::from_u32(trace_len as u32),
        trace_prefix_len_hi: AB::Expr::from_u32((trace_len >> 32) as u32),
        active_count_block_start_lo: AB::Expr::from_u32(active_start as u32),
        active_count_block_start_hi: AB::Expr::from_u32((active_start >> 32) as u32),
        active_count_log_height: AB::Expr::from_u32(profile.prefix.active_count_log_height),
        base_root: local.base_root.map(Into::into),
        full_root: local.full_root.map(Into::into),
        logup_alpha: local.logup_alpha.map(Into::into),
        logup_beta: local.logup_beta.map(Into::into),
    }
}

fn emit_raw_observe<AB: AirBuilder<F = F> + InteractionBuilder>(
    builder: &mut AB,
    bus: TranscriptBus,
    proof: AB::Expr,
    tidx: AB::Expr,
    value: AB::Expr,
    enabled: AB::Expr,
) {
    bus.observe(builder, proof, tidx, value, enabled);
}

fn emit_typed_field<AB: AirBuilder<F = F> + InteractionBuilder>(
    builder: &mut AB,
    bus: TranscriptBus,
    proof: AB::Expr,
    tidx: AB::Expr,
    value: AB::Expr,
    enabled: AB::Expr,
) {
    bus.observe(
        builder,
        proof.clone(),
        tidx.clone(),
        AB::Expr::from_u32(OBSERVATION_FIELD_TAG),
        enabled.clone(),
    );
    bus.observe(builder, proof, tidx + AB::Expr::ONE, value, enabled);
}

fn emit_typed_digest<AB: AirBuilder<F = F> + InteractionBuilder>(
    builder: &mut AB,
    bus: TranscriptBus,
    proof: AB::Expr,
    tidx: AB::Expr,
    digest: [AB::Expr; DIGEST_SIZE],
    enabled: AB::Expr,
) {
    bus.observe(
        builder,
        proof.clone(),
        tidx.clone(),
        AB::Expr::from_u32(OBSERVATION_DIGEST_TAG),
        enabled.clone(),
    );
    for (limb, value) in digest.into_iter().enumerate() {
        bus.observe(
            builder,
            proof.clone(),
            tidx.clone() + AB::Expr::from_usize(1 + limb),
            value,
            enabled.clone(),
        );
    }
}

fn emit_typed_u32_low<AB: AirBuilder<F = F> + InteractionBuilder>(
    builder: &mut AB,
    bus: TranscriptBus,
    proof: AB::Expr,
    tidx: AB::Expr,
    low: AB::Expr,
    enabled: AB::Expr,
) {
    emit_typed_field::<AB>(
        builder,
        bus,
        proof.clone(),
        tidx.clone(),
        low,
        enabled.clone(),
    );
    emit_typed_field::<AB>(
        builder,
        bus,
        proof,
        tidx + AB::Expr::from_u32(2),
        AB::Expr::ZERO,
        enabled,
    );
}

fn emit_digest_samples<AB: AirBuilder<F = F> + InteractionBuilder>(
    builder: &mut AB,
    bus: TranscriptBus,
    proof: AB::Expr,
    start: AB::Expr,
    digest: &[AB::Var; DIGEST_SIZE],
    enabled: AB::Expr,
) where
    AB::Var: Copy,
{
    for (limb, value) in digest.iter().enumerate() {
        bus.sample(
            builder,
            proof.clone(),
            start.clone() + AB::Expr::from_usize(limb),
            *value,
            enabled.clone(),
        );
    }
}

fn emit_outer_preamble<AB: AirBuilder<F = F> + InteractionBuilder>(
    builder: &mut AB,
    bus: TranscriptBus,
    proof: AB::Expr,
    domain: u32,
    observation_count: AB::Expr,
    enabled: AB::Expr,
) {
    emit_raw_observe::<AB>(
        builder,
        bus,
        proof.clone(),
        AB::Expr::ZERO,
        AB::Expr::from_u32(domain),
        enabled.clone(),
    );
    emit_raw_observe::<AB>(
        builder,
        bus,
        proof.clone(),
        AB::Expr::ONE,
        AB::Expr::from_u32(ORDERED_MANIFEST_PROTOCOL_VERSION),
        enabled.clone(),
    );
    emit_raw_observe::<AB>(
        builder,
        bus,
        proof.clone(),
        AB::Expr::from_u32(2),
        observation_count,
        enabled.clone(),
    );
    emit_raw_observe::<AB>(
        builder,
        bus,
        proof,
        AB::Expr::from_u32(3),
        AB::Expr::ZERO,
        enabled,
    );
}

fn emit_schedule_header_transcript<AB>(
    builder: &mut AB,
    bus: TranscriptBus,
    local: &OrderedManifestHeaderCols<AB::Var>,
    enabled: AB::Expr,
) where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    let proof = AB::Expr::ZERO;
    let calls = expr::<AB>(local.call_count);
    emit_outer_preamble::<AB>(
        builder,
        bus,
        proof.clone(),
        SCHEDULE_DIGEST_TAG,
        AB::Expr::from_u32(8) + AB::Expr::from_u32(9) * calls.clone(),
        enabled.clone(),
    );
    emit_typed_field::<AB>(
        builder,
        bus,
        proof.clone(),
        AB::Expr::from_u32(4),
        AB::Expr::from_u32(SCHEDULE_START_TAG),
        enabled.clone(),
    );
    emit_typed_u32_low::<AB>(
        builder,
        bus,
        proof.clone(),
        AB::Expr::from_u32(6),
        AB::Expr::from_u32(ORDERED_MANIFEST_PROTOCOL_VERSION),
        enabled.clone(),
    );
    emit_typed_u32_low::<AB>(
        builder,
        bus,
        proof.clone(),
        AB::Expr::from_u32(10),
        local.source_count.into(),
        enabled.clone(),
    );
    emit_typed_u32_low::<AB>(
        builder,
        bus,
        proof.clone(),
        AB::Expr::from_u32(14),
        calls.clone(),
        enabled.clone(),
    );
    let schedule_end = AB::Expr::from_u32(18) + AB::Expr::from_u32(18) * calls;
    emit_typed_field::<AB>(
        builder,
        bus,
        proof.clone(),
        schedule_end.clone(),
        AB::Expr::from_u32(SCHEDULE_END_TAG),
        enabled.clone(),
    );
    let outer_end = schedule_end + AB::Expr::from_u32(2);
    emit_raw_observe::<AB>(
        builder,
        bus,
        proof.clone(),
        outer_end.clone(),
        AB::Expr::from_u32(OBSERVATION_END_TAG),
        enabled.clone(),
    );
    emit_digest_samples::<AB>(
        builder,
        bus,
        proof,
        outer_end + AB::Expr::ONE,
        &local.schedule_digest,
        enabled,
    );
}

fn emit_schedule_call_transcript<AB>(
    builder: &mut AB,
    bus: TranscriptBus,
    local: &OrderedManifestCallCols<AB::Var>,
    enabled: AB::Expr,
) where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    let proof = AB::Expr::ZERO;
    let base = AB::Expr::from_u32(18) + AB::Expr::from_u32(18) * local.call_index;
    emit_typed_field::<AB>(
        builder,
        bus,
        proof.clone(),
        base.clone(),
        AB::Expr::from_u32(SCHEDULE_CALL_TAG),
        enabled.clone(),
    );
    emit_typed_u32_low::<AB>(
        builder,
        bus,
        proof.clone(),
        base.clone() + AB::Expr::from_u32(2),
        local.call_index.into(),
        enabled.clone(),
    );
    emit_typed_u32_low::<AB>(
        builder,
        bus,
        proof.clone(),
        base.clone() + AB::Expr::from_u32(6),
        local.input_arity.into(),
        enabled.clone(),
    );
    emit_typed_u32_low::<AB>(
        builder,
        bus,
        proof.clone(),
        base.clone() + AB::Expr::from_u32(10),
        local.source_count.into(),
        enabled.clone(),
    );
    let prior = expr::<AB>(local.call_index)
        * (AB::Expr::from_u32(3) - local.call_index)
        * AB::F::TWO.inverse();
    emit_typed_u32_low::<AB>(
        builder,
        bus,
        proof,
        base + AB::Expr::from_u32(14),
        prior,
        enabled,
    );
}

fn emit_source_order_header_transcript<AB>(
    builder: &mut AB,
    bus: TranscriptBus,
    local: &OrderedManifestCallCols<AB::Var>,
    enabled: AB::Expr,
) where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    let proof = AB::Expr::ONE + local.call_index;
    let count = expr::<AB>(local.source_count);
    emit_outer_preamble::<AB>(
        builder,
        bus,
        proof.clone(),
        SOURCE_ORDER_DIGEST_TAG,
        AB::Expr::from_u32(7) + AB::Expr::from_u32(16) * count.clone(),
        enabled.clone(),
    );
    emit_typed_field::<AB>(
        builder,
        bus,
        proof.clone(),
        AB::Expr::from_u32(4),
        AB::Expr::from_u32(SOURCE_ORDER_DIGEST_TAG),
        enabled.clone(),
    );
    emit_typed_u32_low::<AB>(
        builder,
        bus,
        proof.clone(),
        AB::Expr::from_u32(6),
        AB::Expr::from_u32(ORDERED_MANIFEST_PROTOCOL_VERSION),
        enabled.clone(),
    );
    emit_typed_u32_low::<AB>(
        builder,
        bus,
        proof.clone(),
        AB::Expr::from_u32(10),
        local.call_index.into(),
        enabled.clone(),
    );
    emit_typed_u32_low::<AB>(
        builder,
        bus,
        proof.clone(),
        AB::Expr::from_u32(14),
        count.clone(),
        enabled.clone(),
    );
    let outer_end = AB::Expr::from_u32(18) + AB::Expr::from_u32(53) * count;
    emit_raw_observe::<AB>(
        builder,
        bus,
        proof.clone(),
        outer_end.clone(),
        AB::Expr::from_u32(OBSERVATION_END_TAG),
        enabled.clone(),
    );
    emit_digest_samples::<AB>(
        builder,
        bus,
        proof,
        outer_end + AB::Expr::ONE,
        &local.source_order_digest,
        enabled,
    );
}

fn emit_source_order_source_transcript<AB>(
    builder: &mut AB,
    bus: TranscriptBus,
    local: &OrderedManifestSourceCols<AB::Var>,
    enabled: AB::Expr,
) where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    let proof = AB::Expr::ONE + local.call_index;
    let base = AB::Expr::from_u32(18) + AB::Expr::from_u32(53) * local.fresh_index_in_call;
    emit_source_body::<AB>(builder, bus, proof, base, local, false, enabled);
}

fn emit_manifest_header_transcript<AB>(
    builder: &mut AB,
    bus: TranscriptBus,
    local: &OrderedManifestHeaderCols<AB::Var>,
    enabled: AB::Expr,
) where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    let calls = expr::<AB>(local.call_count);
    let sources = expr::<AB>(local.source_count);
    let proof = AB::Expr::ONE + calls.clone();
    emit_outer_preamble::<AB>(
        builder,
        bus,
        proof.clone(),
        MANIFEST_DIGEST_TAG,
        AB::Expr::from_u32(11)
            + AB::Expr::from_u32(8) * calls.clone()
            + AB::Expr::from_u32(18) * sources.clone(),
        enabled.clone(),
    );
    emit_typed_field::<AB>(
        builder,
        bus,
        proof.clone(),
        AB::Expr::from_u32(4),
        AB::Expr::from_u32(MANIFEST_START_TAG),
        enabled.clone(),
    );
    emit_typed_u32_low::<AB>(
        builder,
        bus,
        proof.clone(),
        AB::Expr::from_u32(6),
        AB::Expr::from_u32(ORDERED_MANIFEST_PROTOCOL_VERSION),
        enabled.clone(),
    );
    emit_typed_digest::<AB>(
        builder,
        bus,
        proof.clone(),
        AB::Expr::from_u32(10),
        local.schedule_digest.map(Into::into),
        enabled.clone(),
    );
    emit_typed_u32_low::<AB>(
        builder,
        bus,
        proof.clone(),
        AB::Expr::from_u32(19),
        calls.clone(),
        enabled.clone(),
    );
    let source_count_tidx = AB::Expr::from_u32(23) + AB::Expr::from_u32(23) * calls.clone();
    emit_typed_u32_low::<AB>(
        builder,
        bus,
        proof.clone(),
        source_count_tidx,
        sources.clone(),
        enabled.clone(),
    );
    let final_tag = AB::Expr::from_u32(27)
        + AB::Expr::from_u32(23) * calls.clone()
        + AB::Expr::from_u32(64) * sources.clone();
    emit_typed_field::<AB>(
        builder,
        bus,
        proof.clone(),
        final_tag.clone(),
        AB::Expr::from_u32(MANIFEST_FINAL_TAG),
        enabled.clone(),
    );
    emit_typed_digest::<AB>(
        builder,
        bus,
        proof.clone(),
        final_tag.clone() + AB::Expr::from_u32(2),
        local.final_accumulator_digest.map(Into::into),
        enabled.clone(),
    );
    emit_typed_field::<AB>(
        builder,
        bus,
        proof.clone(),
        final_tag.clone() + AB::Expr::from_u32(11),
        AB::Expr::from_u32(MANIFEST_END_TAG),
        enabled.clone(),
    );
    let outer_end = final_tag + AB::Expr::from_u32(13);
    emit_raw_observe::<AB>(
        builder,
        bus,
        proof.clone(),
        outer_end.clone(),
        AB::Expr::from_u32(OBSERVATION_END_TAG),
        enabled.clone(),
    );
    emit_digest_samples::<AB>(
        builder,
        bus,
        proof,
        outer_end + AB::Expr::ONE,
        &local.manifest_digest,
        enabled,
    );
}

fn emit_manifest_call_transcript<AB>(
    builder: &mut AB,
    bus: TranscriptBus,
    local: &OrderedManifestCallCols<AB::Var>,
    enabled: AB::Expr,
) where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    let proof = AB::Expr::ONE + local.global_call_count;
    let base = AB::Expr::from_u32(23) + AB::Expr::from_u32(23) * local.call_index;
    emit_typed_field::<AB>(
        builder,
        bus,
        proof.clone(),
        base.clone(),
        AB::Expr::from_u32(MANIFEST_INVOCATION_TAG),
        enabled.clone(),
    );
    emit_typed_u32_low::<AB>(
        builder,
        bus,
        proof.clone(),
        base.clone() + AB::Expr::from_u32(2),
        local.call_index.into(),
        enabled.clone(),
    );
    emit_typed_u32_low::<AB>(
        builder,
        bus,
        proof.clone(),
        base.clone() + AB::Expr::from_u32(6),
        local.input_arity.into(),
        enabled.clone(),
    );
    emit_typed_u32_low::<AB>(
        builder,
        bus,
        proof.clone(),
        base.clone() + AB::Expr::from_u32(10),
        local.source_count.into(),
        enabled.clone(),
    );
    emit_typed_digest::<AB>(
        builder,
        bus,
        proof,
        base + AB::Expr::from_u32(14),
        local.full_root.map(Into::into),
        enabled,
    );
}

fn emit_manifest_source_transcript<AB>(
    builder: &mut AB,
    bus: TranscriptBus,
    local: &OrderedManifestSourceCols<AB::Var>,
    enabled: AB::Expr,
) where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    let proof = AB::Expr::ONE + local.global_call_count;
    let base = AB::Expr::from_u32(27)
        + AB::Expr::from_u32(23) * local.global_call_count
        + AB::Expr::from_u32(64) * local.source_index;
    emit_source_body::<AB>(builder, bus, proof, base, local, true, enabled);
}

fn emit_source_body<AB>(
    builder: &mut AB,
    bus: TranscriptBus,
    proof: AB::Expr,
    base: AB::Expr,
    local: &OrderedManifestSourceCols<AB::Var>,
    include_instance: bool,
    enabled: AB::Expr,
) where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    let mut cursor = base;
    if include_instance {
        emit_typed_field::<AB>(
            builder,
            bus,
            proof.clone(),
            cursor.clone(),
            AB::Expr::from_u32(MANIFEST_SOURCE_TAG),
            enabled.clone(),
        );
        cursor += AB::Expr::from_u32(2);
    }
    for value in [
        local.source_index,
        local.call_index,
        local.fresh_index_in_call,
        local.normalized_leaf_start,
    ] {
        emit_typed_u32_low::<AB>(
            builder,
            bus,
            proof.clone(),
            cursor.clone(),
            value.into(),
            enabled.clone(),
        );
        cursor += AB::Expr::from_u32(4);
    }
    emit_typed_field::<AB>(
        builder,
        bus,
        proof.clone(),
        cursor.clone(),
        local.active_child_count.into(),
        enabled.clone(),
    );
    cursor += AB::Expr::from_u32(2);
    emit_typed_digest::<AB>(
        builder,
        bus,
        proof.clone(),
        cursor.clone(),
        local.program_commitment.map(Into::into),
        enabled.clone(),
    );
    cursor += AB::Expr::from_u32(9);
    emit_typed_field::<AB>(
        builder,
        bus,
        proof.clone(),
        cursor.clone(),
        local.initial_pc.into(),
        enabled.clone(),
    );
    cursor += AB::Expr::from_u32(2);
    emit_typed_digest::<AB>(
        builder,
        bus,
        proof.clone(),
        cursor.clone(),
        local.initial_root.map(Into::into),
        enabled.clone(),
    );
    cursor += AB::Expr::from_u32(9);
    emit_typed_field::<AB>(
        builder,
        bus,
        proof.clone(),
        cursor.clone(),
        local.final_pc.into(),
        enabled.clone(),
    );
    cursor += AB::Expr::from_u32(2);
    emit_typed_digest::<AB>(
        builder,
        bus,
        proof.clone(),
        cursor.clone(),
        local.final_root.map(Into::into),
        enabled.clone(),
    );
    cursor += AB::Expr::from_u32(9);
    emit_typed_field::<AB>(
        builder,
        bus,
        proof.clone(),
        cursor.clone(),
        local.exit_code.into(),
        enabled.clone(),
    );
    cursor += AB::Expr::from_u32(2);
    emit_typed_field::<AB>(
        builder,
        bus,
        proof.clone(),
        cursor.clone(),
        local.is_terminate.into(),
        enabled.clone(),
    );
    cursor += AB::Expr::from_u32(2);
    if include_instance {
        emit_typed_digest::<AB>(
            builder,
            bus,
            proof,
            cursor,
            local.source_instance_digest.map(Into::into),
            enabled,
        );
    }
}

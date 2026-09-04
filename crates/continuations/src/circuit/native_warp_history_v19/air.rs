use core::borrow::Borrow;

use openvm_circuit::system::connector::DEFAULT_SUSPEND_EXIT_CODE;
use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, LookupBus},
    p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, D_EF, F};

use super::{
    digest::{
        TAG_KEY_META_V19, TAG_LOGUP_ENDPOINT_V19, TAG_LOGUP_META_V19, TAG_REPLAY_META_V19,
        TAG_SEGMENT_END_V19, TAG_SEGMENT_HEADER_V19, TAG_TERMINAL_V19,
    },
    CertifiedProgramFingerprintBusV19, CertifiedProgramFingerprintMessageV19,
    CertifiedVmSegmentMetadataBusV19, CertifiedVmSegmentMetadataMessageV19, DigestV19,
    HistoryPublicValuesBusV19, HistoryPublicValuesV19, HistoryVerifierProfileV19,
    SetupPcsSourceCheckpointBusV3, NATIVE_WARP_HISTORY_PROTOCOL_V19,
};

const LIMB_BITS: usize = 16;
const LIMB_BASE: u32 = 1 << LIMB_BITS;
pub(crate) const HASH_SLOTS_V19: usize = 20;

macro_rules! define_history_lookup_bus {
    ($Bus:ident, $Message:ident) => {
        #[derive(Copy, Clone, Debug)]
        pub struct $Bus(LookupBus);

        impl $Bus {
            #[inline]
            pub fn new(bus_index: BusIndex) -> Self {
                Self(LookupBus::new(bus_index))
            }

            pub fn lookup_key<AB: InteractionBuilder>(
                &self,
                builder: &mut AB,
                key: $Message<impl Into<AB::Expr> + Clone>,
                enabled: impl Into<AB::Expr>,
            ) {
                self.0.lookup_key(builder, key.to_vec(), enabled);
            }

            pub fn add_key_with_lookups<AB: InteractionBuilder>(
                &self,
                builder: &mut AB,
                key: $Message<impl Into<AB::Expr> + Clone>,
                count: impl Into<AB::Expr>,
            ) {
                self.0.add_key_with_lookups(builder, key.to_vec(), count);
            }
        }
    };
}

/// Use the recursion system's exact Poseidon2 bus types. This lets History,
/// SWIRL, VACC, transcript replay and Merkle verification share one physical
/// multibus Poseidon table without relying on merely layout-compatible wrapper
/// types or manually copied bus indices.
pub type HistoryPoseidon2CompressBusV19 = openvm_recursion_circuit::bus::Poseidon2CompressBus;
pub type HistoryPoseidon2CompressMessageV19<T> =
    openvm_recursion_circuit::bus::Poseidon2CompressMessage<T>;

/// Exact statement exported by an independent standard-WARP replay verifier.
/// `relation_digest` is part of the lookup key, so a replay for one PESAT
/// index cannot certify another homogeneous shard.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug, PartialEq, Eq)]
pub struct CertifiedWarpReplayMessageV19<T> {
    pub protocol_version: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub update_index_lo: T,
    pub update_index_hi: T,
    pub shard_ordinal: T,
    pub source_forest_root: [T; DIGEST_SIZE],
    pub key_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub opening_claim_digest: [T; DIGEST_SIZE],
    pub fresh_instance_digest: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub prior_root: [T; DIGEST_SIZE],
    pub fresh_root: [T; DIGEST_SIZE],
    pub next_root: [T; DIGEST_SIZE],
    pub previous_accumulator_digest: [T; DIGEST_SIZE],
    pub next_accumulator_digest: [T; DIGEST_SIZE],
    pub authenticated_batching_claim: [T; D_EF],
    pub previous_checkpoint_digest: [T; DIGEST_SIZE],
    pub next_checkpoint_digest: [T; DIGEST_SIZE],
    pub replay_endpoint_digest: [T; DIGEST_SIZE],
    pub replay_binding_digest: [T; DIGEST_SIZE],
}

define_history_lookup_bus!(CertifiedWarpReplayBusV19, CertifiedWarpReplayMessageV19);

/// Exact statement exported by the segment-wide external LogUp-only verifier.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug, PartialEq, Eq)]
pub struct LogUpOnlyHistoryMessageV19<T> {
    pub protocol_version: T,
    pub mode_tag: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub app_vk_digest: [T; DIGEST_SIZE],
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub verifier_endpoint: [T; D_EF],
    pub checkpoint_digest: [T; DIGEST_SIZE],
}

/// LogUp History lookup plus an optional exact setup-PCS provenance fanout.
/// Keeping the hook inside this bus preserves every existing
/// `LogUpOnlyProducerAirV19` struct literal: legacy callers use `new`, while
/// authority callers opt in with `with_setup_pcs_source_checkpoint_bus_v3`.
#[derive(Copy, Clone, Debug)]
pub struct LogUpOnlyHistoryBusV19 {
    lookup: LookupBus,
    setup_pcs_source_checkpoint_bus: Option<SetupPcsSourceCheckpointBusV3>,
}

impl LogUpOnlyHistoryBusV19 {
    #[inline]
    pub fn new(bus_index: BusIndex) -> Self {
        Self {
            lookup: LookupBus::new(bus_index),
            setup_pcs_source_checkpoint_bus: None,
        }
    }

    #[must_use]
    pub fn with_setup_pcs_source_checkpoint_bus_v3(
        mut self,
        bus: SetupPcsSourceCheckpointBusV3,
    ) -> Self {
        self.setup_pcs_source_checkpoint_bus = Some(bus);
        self
    }

    #[must_use]
    pub fn setup_pcs_source_checkpoint_bus_v3(&self) -> Option<SetupPcsSourceCheckpointBusV3> {
        self.setup_pcs_source_checkpoint_bus
    }

    pub fn lookup_key<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        key: LogUpOnlyHistoryMessageV19<impl Into<AB::Expr> + Clone>,
        enabled: impl Into<AB::Expr>,
    ) {
        self.lookup.lookup_key(builder, key.to_vec(), enabled);
    }

    pub fn add_key_with_lookups<AB: InteractionBuilder>(
        &self,
        builder: &mut AB,
        key: LogUpOnlyHistoryMessageV19<impl Into<AB::Expr> + Clone>,
        count: impl Into<AB::Expr>,
    ) {
        self.lookup
            .add_key_with_lookups(builder, key.to_vec(), count);
    }
}

/// One event row in the bounded history state machine.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct HistoryRowColsV19<T> {
    pub active: T,
    pub is_segment_start: T,
    pub is_shard: T,
    pub is_path: T,
    pub is_logup: T,
    pub is_segment_end: T,
    pub is_final: T,

    pub state_before_count_lo: T,
    pub state_before_count_hi: T,
    pub state_before_count_bits: [[T; LIMB_BITS]; 2],
    pub state_before_vm: [T; DIGEST_SIZE],
    pub state_before_product: [T; DIGEST_SIZE],
    pub state_before_history: [T; DIGEST_SIZE],
    pub state_before_public_values: [T; DIGEST_SIZE],
    pub state_before_terminated: T,

    pub state_after_count_lo: T,
    pub state_after_count_hi: T,
    pub state_after_count_bits: [[T; LIMB_BITS]; 2],
    pub state_after_vm: [T; DIGEST_SIZE],
    pub state_after_product: [T; DIGEST_SIZE],
    pub state_after_history: [T; DIGEST_SIZE],
    pub state_after_public_values: [T; DIGEST_SIZE],
    pub state_after_terminated: T,
    pub count_increment_carry: T,

    pub ctx_segment_index_lo: T,
    pub ctx_segment_index_hi: T,
    pub ctx_segment_index_bits: [[T; LIMB_BITS]; 2],
    pub ctx_initial_pc: T,
    pub ctx_final_pc: T,
    pub ctx_exit_code: T,
    pub ctx_initial_memory_root: [T; DIGEST_SIZE],
    pub ctx_final_memory_root: [T; DIGEST_SIZE],
    pub ctx_program_fingerprint: [T; D_EF],
    pub ctx_program_fingerprint_digest: [T; DIGEST_SIZE],
    pub ctx_program_registry_digest: [T; DIGEST_SIZE],
    pub ctx_program_relation_digest: [T; DIGEST_SIZE],
    pub ctx_program_log_height: T,
    pub ctx_program_cached_width: T,
    pub ctx_from_vm: [T; DIGEST_SIZE],
    pub ctx_to_vm: [T; DIGEST_SIZE],
    pub ctx_source_forest_root: [T; DIGEST_SIZE],
    pub ctx_previous_product: [T; DIGEST_SIZE],
    pub ctx_next_product: [T; DIGEST_SIZE],
    pub ctx_public_values: [T; DIGEST_SIZE],
    pub ctx_terminates: T,
    pub ctx_expected_shards: T,
    pub ctx_expected_shards_bits: [T; LIMB_BITS],
    pub ctx_manifest_digest: [T; DIGEST_SIZE],
    pub ctx_segment_openings_digest: [T; DIGEST_SIZE],

    pub manifest_before: [T; DIGEST_SIZE],
    pub manifest_after: [T; DIGEST_SIZE],

    pub order_seen_before: T,
    pub order_seen_after: T,
    pub last_ordinal_before: T,
    pub last_ordinal_after: T,
    pub shard_count_before: T,
    pub shard_count_after: T,
    pub shard_count_before_bits: [T; LIMB_BITS],
    pub shard_count_after_bits: [T; LIMB_BITS],
    pub shard_capacity_remaining_bits: [T; LIMB_BITS],

    pub shard_ordinal: T,
    pub shard_ordinal_bits: [T; LIMB_BITS],
    pub ordinal_gap_bits: [T; LIMB_BITS],
    pub catalog_remaining_bits: [T; LIMB_BITS],
    pub key_app_vk_digest: [T; DIGEST_SIZE],
    pub key_air_id_lo: T,
    pub key_air_id_hi: T,
    pub key_log_height: T,
    pub key_trace_layout_digest: [T; DIGEST_SIZE],
    pub key_public_schema_digest: [T; DIGEST_SIZE],
    pub key_interaction_schema_digest: [T; DIGEST_SIZE],
    pub key_relation_digest: [T; DIGEST_SIZE],
    pub key_code_class_digest: [T; DIGEST_SIZE],

    pub update_index_lo: T,
    pub update_index_hi: T,
    pub update_index_bits: [[T; LIMB_BITS]; 2],
    pub opening_claim_digest: [T; DIGEST_SIZE],
    pub fresh_instance_digest: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub prior_root: [T; DIGEST_SIZE],
    pub fresh_root: [T; DIGEST_SIZE],
    pub next_root: [T; DIGEST_SIZE],
    pub previous_accumulator_digest: [T; DIGEST_SIZE],
    pub next_accumulator_digest: [T; DIGEST_SIZE],
    pub authenticated_batching_claim: [T; D_EF],
    pub previous_checkpoint_digest: [T; DIGEST_SIZE],
    pub next_checkpoint_digest: [T; DIGEST_SIZE],
    pub replay_endpoint_digest: [T; DIGEST_SIZE],
    pub replay_binding_digest: [T; DIGEST_SIZE],
    pub previous_leaf_digest: [T; DIGEST_SIZE],
    pub next_leaf_digest: [T; DIGEST_SIZE],

    pub path_is_first: T,
    pub path_is_last: T,
    pub path_not_last_inv: T,
    pub path_depth: T,
    pub path_index_bit: T,
    pub path_index_before: T,
    pub path_index_after: T,
    pub path_bit_weight: T,
    pub path_old_node: [T; DIGEST_SIZE],
    pub path_new_node: [T; DIGEST_SIZE],
    pub path_catalog_node: [T; DIGEST_SIZE],
    pub path_product_sibling: [T; DIGEST_SIZE],
    pub path_catalog_sibling: [T; DIGEST_SIZE],
    pub path_old_parent: [T; DIGEST_SIZE],
    pub path_new_parent: [T; DIGEST_SIZE],
    pub path_catalog_parent: [T; DIGEST_SIZE],

    pub logup_mode_tag: T,
    pub logup_checkpoint_digest: [T; DIGEST_SIZE],
    pub logup_segment_openings_digest: [T; DIGEST_SIZE],
    pub logup_verifier_endpoint: [T; D_EF],

    /// Poseidon2 outputs in event-specific slots.  Inputs are derived from the
    /// constrained row values and never supplied as free witness columns.
    pub hash_outputs: [[T; DIGEST_SIZE]; HASH_SLOTS_V19],
}

/// Setup-fixed boundary for the unique stage-zero History proof.
///
/// Count, public-values digest, and termination have protocol-canonical zero
/// values. The two roots depend on the setup's product-tree and transcript
/// conventions, so they are supplied explicitly. Initial VM state is omitted
/// deliberately: it remains an execution input certified by the first
/// segment's VM metadata lookup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryGenesisConfigV19 {
    pub canonical_empty_product_state_root: DigestV19,
    pub canonical_initial_history_root: DigestV19,
}

#[derive(Clone, Debug)]
pub struct HistoryAirV19 {
    pub profile: HistoryVerifierProfileV19,
    /// `Some` only for the stage-zero proving/verifying key. Recursive child
    /// chunks use `None` and are instead linked to their parent's final public
    /// boundary by `HistoryChunkBridgeV19`.
    pub genesis: Option<HistoryGenesisConfigV19>,
    pub compress_bus: HistoryPoseidon2CompressBusV19,
    pub certified_replay_bus: CertifiedWarpReplayBusV19,
    pub logup_only_bus: LogUpOnlyHistoryBusV19,
    pub certified_vm_bus: CertifiedVmSegmentMetadataBusV19,
    pub certified_program_bus: CertifiedProgramFingerprintBusV19,
    /// Present only when this bounded chunk is embedded in a recursive
    /// History stage. Each segment-start row publishes the same fixed public
    /// values; the bridge consumes them with the setup-fixed segment
    /// multiplicity.
    pub public_values_bus: Option<HistoryPublicValuesBusV19>,
}

impl BaseAir<F> for HistoryAirV19 {
    fn width(&self) -> usize {
        HistoryRowColsV19::<F>::width()
    }
}

impl BaseAirWithPublicValues<F> for HistoryAirV19 {
    fn num_public_values(&self) -> usize {
        HistoryPublicValuesV19::<F>::width()
    }
}

impl PartitionedBaseAir<F> for HistoryAirV19 {}

impl<AB> Air<AB> for HistoryAirV19
where
    AB: AirBuilder<F = F> + AirBuilderWithPublicValues + InteractionBuilder,
    AB::Var: Copy,
    AB::PublicVar: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("history-v19 local row");
        let next_row = main.row_slice(1).expect("history-v19 next row");
        let local: &HistoryRowColsV19<AB::Var> = (*local_row).borrow();
        let next: &HistoryRowColsV19<AB::Var> = (*next_row).borrow();
        let public_values = builder.public_values().to_vec();
        let public_ref: &HistoryPublicValuesV19<AB::PublicVar> = public_values.as_slice().borrow();
        let public = public_ref.clone();
        if let Some(bus) = self.public_values_bus {
            for (index, &value) in public_values.iter().enumerate() {
                bus.send(
                    builder,
                    AB::Expr::from_usize(index),
                    value,
                    local.is_segment_start,
                );
            }
        }

        let kinds = [
            local.is_segment_start,
            local.is_shard,
            local.is_path,
            local.is_logup,
            local.is_segment_end,
            local.is_final,
        ];
        builder.assert_bool(local.active);
        for kind in kinds {
            builder.assert_bool(kind);
        }
        builder.assert_eq(
            kinds
                .into_iter()
                .fold(AB::Expr::ZERO, |sum, flag| sum + flag),
            local.active,
        );
        builder.when_first_row().assert_one(local.active);
        builder.when_last_row().assert_zero(local.active);
        builder.when_first_row().assert_one(local.is_segment_start);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .when(local.active - next.active)
            .assert_one(local.is_final);
        builder
            .when_transition()
            .when(local.is_final)
            .assert_zero(next.active);

        self.eval_sequence(builder, local, next);
        self.eval_u16s(builder, local);
        self.eval_state(builder, local, next, &public);
        self.eval_context(builder, local, next);
        self.eval_ordering(builder, local, next);
        self.eval_segment_start(builder, local);
        self.eval_shard(builder, local, next);
        self.eval_path(builder, local, next);
        self.eval_logup(builder, local);
        self.eval_segment_end(builder, local, &public);
        self.eval_final(builder, local, &public);
    }
}

impl HistoryAirV19 {
    fn eval_sequence<AB: AirBuilder<F = F>>(
        &self,
        builder: &mut AB,
        local: &HistoryRowColsV19<AB::Var>,
        next: &HistoryRowColsV19<AB::Var>,
    ) where
        AB::Var: Copy,
    {
        let transition = local.active * next.active;
        builder
            .when_transition()
            .when(transition.clone() * local.is_segment_start)
            .assert_one(next.is_shard + next.is_logup);
        builder
            .when_transition()
            .when(transition.clone() * local.is_shard)
            .assert_one(next.is_path);
        builder
            .when_transition()
            .when(transition.clone() * local.is_path * (AB::Expr::ONE - local.path_is_last))
            .assert_one(next.is_path);
        builder
            .when_transition()
            .when(transition.clone() * local.is_path * local.path_is_last)
            .assert_one(next.is_shard + next.is_logup);
        builder
            .when_transition()
            .when(transition.clone() * local.is_logup)
            .assert_one(next.is_segment_end);
        builder
            .when_transition()
            .when(transition * local.is_segment_end)
            .assert_one(next.is_segment_start + next.is_final);
    }

    fn eval_u16s<AB: AirBuilder<F = F>>(&self, builder: &mut AB, local: &HistoryRowColsV19<AB::Var>)
    where
        AB::Var: Copy,
    {
        for (value, bits) in [
            (
                local.state_before_count_lo,
                &local.state_before_count_bits[0],
            ),
            (
                local.state_before_count_hi,
                &local.state_before_count_bits[1],
            ),
            (local.state_after_count_lo, &local.state_after_count_bits[0]),
            (local.state_after_count_hi, &local.state_after_count_bits[1]),
            (local.ctx_segment_index_lo, &local.ctx_segment_index_bits[0]),
            (local.ctx_segment_index_hi, &local.ctx_segment_index_bits[1]),
            (local.ctx_expected_shards, &local.ctx_expected_shards_bits),
            (local.shard_count_before, &local.shard_count_before_bits),
            (local.shard_count_after, &local.shard_count_after_bits),
            (local.shard_ordinal, &local.shard_ordinal_bits),
            (local.update_index_lo, &local.update_index_bits[0]),
            (local.update_index_hi, &local.update_index_bits[1]),
        ] {
            self.assert_u16(builder, local.active, value, bits);
        }
        self.assert_bits(builder, local.is_shard, &local.ordinal_gap_bits);
        self.assert_bits(builder, local.is_shard, &local.catalog_remaining_bits);
        self.assert_bits(
            builder,
            local.is_segment_end,
            &local.shard_capacity_remaining_bits,
        );
    }

    fn eval_state<AB: AirBuilder<F = F> + AirBuilderWithPublicValues>(
        &self,
        builder: &mut AB,
        local: &HistoryRowColsV19<AB::Var>,
        next: &HistoryRowColsV19<AB::Var>,
        public: &HistoryPublicValuesV19<AB::PublicVar>,
    ) where
        AB::Var: Copy,
        AB::PublicVar: Copy,
    {
        builder.assert_bool(local.state_before_terminated);
        builder.assert_bool(local.state_after_terminated);
        builder.assert_bool(local.count_increment_carry);
        builder
            .when(local.active * (AB::Expr::ONE - local.is_segment_end))
            .assert_zero(local.count_increment_carry);
        builder.when(local.active).assert_eq(
            local.state_after_count_lo,
            local.state_before_count_lo + local.is_segment_end
                - local.count_increment_carry * AB::Expr::from_u32(LIMB_BASE),
        );
        builder.when(local.active).assert_eq(
            local.state_after_count_hi,
            local.state_before_count_hi + local.count_increment_carry,
        );

        let path_update = local.is_path * local.path_is_last;
        for limb in 0..DIGEST_SIZE {
            builder.when(local.active).assert_eq(
                local.state_after_vm[limb],
                local.is_segment_end * local.ctx_to_vm[limb]
                    + (AB::Expr::ONE - local.is_segment_end) * local.state_before_vm[limb],
            );
            builder.when(local.active).assert_eq(
                local.state_after_product[limb],
                path_update.clone() * local.path_new_parent[limb]
                    + (AB::Expr::ONE - path_update.clone()) * local.state_before_product[limb],
            );
            builder.when(local.active).assert_eq(
                local.state_after_history[limb],
                local.is_segment_end * local.hash_outputs[2][limb]
                    + (AB::Expr::ONE - local.is_segment_end) * local.state_before_history[limb],
            );
            builder.when(local.active).assert_eq(
                local.state_after_public_values[limb],
                local.is_segment_end * local.ctx_public_values[limb]
                    + (AB::Expr::ONE - local.is_segment_end)
                        * local.state_before_public_values[limb],
            );
        }
        builder.when(local.active).assert_eq(
            local.state_after_terminated,
            local.is_segment_end * local.ctx_terminates
                + (AB::Expr::ONE - local.is_segment_end) * local.state_before_terminated,
        );

        let link = next.active;
        let mut transition = builder.when_transition();
        transition
            .when(link)
            .assert_eq(next.state_before_count_lo, local.state_after_count_lo);
        transition
            .when(link)
            .assert_eq(next.state_before_count_hi, local.state_after_count_hi);
        transition
            .when(link)
            .assert_eq(next.state_before_terminated, local.state_after_terminated);
        for limb in 0..DIGEST_SIZE {
            transition
                .when(link)
                .assert_eq(next.state_before_vm[limb], local.state_after_vm[limb]);
            transition.when(link).assert_eq(
                next.state_before_product[limb],
                local.state_after_product[limb],
            );
            transition.when(link).assert_eq(
                next.state_before_history[limb],
                local.state_after_history[limb],
            );
            transition.when(link).assert_eq(
                next.state_before_public_values[limb],
                local.state_after_public_values[limb],
            );
        }

        builder
            .when_first_row()
            .assert_eq(local.state_before_count_lo, public.initial_segment_count_lo);
        builder
            .when_first_row()
            .assert_eq(local.state_before_count_hi, public.initial_segment_count_hi);
        builder
            .when_first_row()
            .assert_eq(local.state_before_terminated, public.initial_terminated);
        for limb in 0..DIGEST_SIZE {
            builder
                .when_first_row()
                .assert_eq(local.state_before_vm[limb], public.initial_vm_state[limb]);
            builder.when_first_row().assert_eq(
                local.state_before_product[limb],
                public.initial_product_state_root[limb],
            );
            builder.when_first_row().assert_eq(
                local.state_before_history[limb],
                public.initial_history_root[limb],
            );
            builder.when_first_row().assert_eq(
                local.state_before_public_values[limb],
                public.initial_public_values_digest[limb],
            );
        }

        // This is a setup-time branch: a genesis key has these constraints,
        // while a continuation-chunk key does not. Bind the first witness row
        // rather than host-checking public values; the equalities immediately
        // above propagate the canonical boundary to the public statement.
        if let Some(genesis) = &self.genesis {
            builder
                .when_first_row()
                .assert_zero(local.state_before_count_lo);
            builder
                .when_first_row()
                .assert_zero(local.state_before_count_hi);
            builder
                .when_first_row()
                .assert_zero(local.state_before_terminated);
            for limb in 0..DIGEST_SIZE {
                builder.when_first_row().assert_eq(
                    local.state_before_product[limb],
                    genesis.canonical_empty_product_state_root[limb],
                );
                builder.when_first_row().assert_eq(
                    local.state_before_history[limb],
                    genesis.canonical_initial_history_root[limb],
                );
                builder
                    .when_first_row()
                    .assert_zero(local.state_before_public_values[limb]);
            }
        }
    }

    fn eval_context<AB: AirBuilder<F = F>>(
        &self,
        builder: &mut AB,
        local: &HistoryRowColsV19<AB::Var>,
        next: &HistoryRowColsV19<AB::Var>,
    ) where
        AB::Var: Copy,
    {
        builder.assert_bool(local.ctx_terminates);
        let same_segment =
            next.active * (next.is_shard + next.is_path + next.is_logup + next.is_segment_end);
        for (a, b) in [
            (next.ctx_segment_index_lo, local.ctx_segment_index_lo),
            (next.ctx_segment_index_hi, local.ctx_segment_index_hi),
            (next.ctx_initial_pc, local.ctx_initial_pc),
            (next.ctx_final_pc, local.ctx_final_pc),
            (next.ctx_exit_code, local.ctx_exit_code),
            (next.ctx_program_log_height, local.ctx_program_log_height),
            (
                next.ctx_program_cached_width,
                local.ctx_program_cached_width,
            ),
            (next.ctx_terminates, local.ctx_terminates),
            (next.ctx_expected_shards, local.ctx_expected_shards),
        ] {
            builder
                .when_transition()
                .when(same_segment.clone())
                .assert_eq(a, b);
        }
        for (next_digest, local_digest) in [
            (&next.ctx_from_vm, &local.ctx_from_vm),
            (&next.ctx_to_vm, &local.ctx_to_vm),
            (
                &next.ctx_initial_memory_root,
                &local.ctx_initial_memory_root,
            ),
            (&next.ctx_final_memory_root, &local.ctx_final_memory_root),
            (
                &next.ctx_program_fingerprint_digest,
                &local.ctx_program_fingerprint_digest,
            ),
            (
                &next.ctx_program_registry_digest,
                &local.ctx_program_registry_digest,
            ),
            (
                &next.ctx_program_relation_digest,
                &local.ctx_program_relation_digest,
            ),
            (&next.ctx_source_forest_root, &local.ctx_source_forest_root),
            (&next.ctx_previous_product, &local.ctx_previous_product),
            (&next.ctx_next_product, &local.ctx_next_product),
            (&next.ctx_public_values, &local.ctx_public_values),
            (&next.ctx_manifest_digest, &local.ctx_manifest_digest),
            (
                &next.ctx_segment_openings_digest,
                &local.ctx_segment_openings_digest,
            ),
        ] {
            for limb in 0..DIGEST_SIZE {
                builder
                    .when_transition()
                    .when(same_segment.clone())
                    .assert_eq(next_digest[limb], local_digest[limb]);
            }
        }
        for limb in 0..D_EF {
            builder
                .when_transition()
                .when(same_segment.clone())
                .assert_eq(
                    next.ctx_program_fingerprint[limb],
                    local.ctx_program_fingerprint[limb],
                );
        }

        for limb in 0..DIGEST_SIZE {
            builder
                .when_transition()
                .when(same_segment.clone())
                .assert_eq(next.manifest_before[limb], local.manifest_after[limb]);
        }
    }

    fn eval_ordering<AB: AirBuilder<F = F>>(
        &self,
        builder: &mut AB,
        local: &HistoryRowColsV19<AB::Var>,
        next: &HistoryRowColsV19<AB::Var>,
    ) where
        AB::Var: Copy,
    {
        for flag in [local.order_seen_before, local.order_seen_after] {
            builder.assert_bool(flag);
        }
        builder.when(local.active).assert_eq(
            local.shard_count_after,
            local.shard_count_before + local.is_shard,
        );
        builder.when(local.active).assert_eq(
            local.order_seen_after,
            local.order_seen_before + local.is_shard * (AB::Expr::ONE - local.order_seen_before),
        );
        builder.when(local.active).assert_eq(
            local.last_ordinal_after,
            local.is_shard * local.shard_ordinal
                + (AB::Expr::ONE - local.is_shard) * local.last_ordinal_before,
        );
        let expected_ordinal = AB::Expr::from(local.last_ordinal_before)
            + AB::Expr::ONE
            + self.bits_expr::<AB>(&local.ordinal_gap_bits);
        builder
            .when(local.is_shard * local.order_seen_before)
            .assert_zero(AB::Expr::from(local.shard_ordinal) - expected_ordinal);
        builder
            .when(local.is_shard * (AB::Expr::ONE - local.order_seen_before))
            .assert_zero(self.bits_expr::<AB>(&local.ordinal_gap_bits));
        let catalog_bound = AB::Expr::from(local.shard_ordinal)
            + AB::Expr::ONE
            + self.bits_expr::<AB>(&local.catalog_remaining_bits)
            - AB::Expr::from_u16(self.profile.shard_catalog_len);
        builder.when(local.is_shard).assert_zero(catalog_bound);
        let capacity_bound = AB::Expr::from(local.shard_count_after)
            + self.bits_expr::<AB>(&local.shard_capacity_remaining_bits)
            - AB::Expr::from_u16(self.profile.max_active_shards_per_segment);
        builder
            .when(local.is_segment_end)
            .assert_zero(capacity_bound);

        let same_segment =
            next.active * (next.is_shard + next.is_path + next.is_logup + next.is_segment_end);
        let mut transition_builder = builder.when_transition();
        let mut transition = transition_builder.when(same_segment);
        transition.assert_eq(next.order_seen_before, local.order_seen_after);
        transition.assert_eq(next.last_ordinal_before, local.last_ordinal_after);
        transition.assert_eq(next.shard_count_before, local.shard_count_after);
    }

    fn eval_segment_start<AB: AirBuilder<F = F> + InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &HistoryRowColsV19<AB::Var>,
    ) where
        AB::Var: Copy,
    {
        let enabled = local.is_segment_start;
        builder
            .when(enabled)
            .assert_zero(local.state_before_terminated);
        builder
            .when(enabled)
            .assert_eq(local.ctx_segment_index_lo, local.state_before_count_lo);
        builder
            .when(enabled)
            .assert_eq(local.ctx_segment_index_hi, local.state_before_count_hi);
        builder.when(enabled).assert_zero(local.order_seen_before);
        builder.when(enabled).assert_zero(local.last_ordinal_before);
        builder.when(enabled).assert_zero(local.shard_count_before);
        for limb in 0..DIGEST_SIZE {
            builder
                .when(enabled)
                .assert_eq(local.ctx_from_vm[limb], local.state_before_vm[limb]);
            builder.when(enabled).assert_eq(
                local.ctx_previous_product[limb],
                local.state_before_product[limb],
            );
            builder
                .when(enabled)
                .assert_zero(local.manifest_before[limb]);
        }

        self.certified_vm_bus.receive(
            builder,
            CertifiedVmSegmentMetadataMessageV19 {
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                segment_index_lo: local.ctx_segment_index_lo.into(),
                segment_index_hi: local.ctx_segment_index_hi.into(),
                initial_pc: local.ctx_initial_pc.into(),
                final_pc: local.ctx_final_pc.into(),
                exit_code: local.ctx_exit_code.into(),
                is_terminate: local.ctx_terminates.into(),
                initial_memory_root: local.ctx_initial_memory_root.map(Into::into),
                final_memory_root: local.ctx_final_memory_root.map(Into::into),
                program_fingerprint: local.ctx_program_fingerprint.map(Into::into),
                program_fingerprint_digest: local.ctx_program_fingerprint_digest.map(Into::into),
                from_vm_state: local.ctx_from_vm.map(Into::into),
                to_vm_state: local.ctx_to_vm.map(Into::into),
            },
            enabled,
        );
        self.certified_program_bus.receive(
            builder,
            CertifiedProgramFingerprintMessageV19 {
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                segment_index_lo: local.ctx_segment_index_lo.into(),
                segment_index_hi: local.ctx_segment_index_hi.into(),
                app_vk_digest: self.profile.app_vk_digest.map(Into::into),
                registry_digest: local.ctx_program_registry_digest.map(Into::into),
                relation_digest: local.ctx_program_relation_digest.map(Into::into),
                log_height: local.ctx_program_log_height.into(),
                cached_width: local.ctx_program_cached_width.into(),
                value: local.ctx_program_fingerprint.map(Into::into),
                digest: local.ctx_program_fingerprint_digest.map(Into::into),
            },
            enabled,
        );

        let scalar = self.scalar_expr::<AB>([
            AB::Expr::from_u32(TAG_SEGMENT_HEADER_V19),
            AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            local.ctx_segment_index_lo.into(),
            local.ctx_segment_index_hi.into(),
            local.ctx_expected_shards.into(),
            local.ctx_terminates.into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ]);
        self.lookup_hash(
            builder,
            scalar,
            self.zero_digest::<AB>(),
            local.hash_outputs[0],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.ctx_from_vm.map(Into::into),
            local.ctx_to_vm.map(Into::into),
            local.hash_outputs[1],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.ctx_previous_product.map(Into::into),
            local.ctx_next_product.map(Into::into),
            local.hash_outputs[2],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.ctx_segment_openings_digest.map(Into::into),
            self.zero_digest::<AB>(),
            local.hash_outputs[3],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.ctx_source_forest_root.map(Into::into),
            local.ctx_public_values.map(Into::into),
            local.hash_outputs[4],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[4].map(Into::into),
            local.hash_outputs[3].map(Into::into),
            local.hash_outputs[5],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[0].map(Into::into),
            local.hash_outputs[1].map(Into::into),
            local.hash_outputs[6],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[2].map(Into::into),
            local.hash_outputs[5].map(Into::into),
            local.hash_outputs[7],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[6].map(Into::into),
            local.hash_outputs[7].map(Into::into),
            local.hash_outputs[8],
            enabled,
        );
        for limb in 0..DIGEST_SIZE {
            builder
                .when(enabled)
                .assert_eq(local.manifest_after[limb], local.hash_outputs[8][limb]);
        }
    }

    fn eval_shard<AB: AirBuilder<F = F> + InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &HistoryRowColsV19<AB::Var>,
        next: &HistoryRowColsV19<AB::Var>,
    ) where
        AB::Var: Copy,
    {
        let enabled = local.is_shard;
        for limb in 0..DIGEST_SIZE {
            builder.when(enabled).assert_eq(
                local.segment_openings_digest[limb],
                local.ctx_segment_openings_digest[limb],
            );
        }
        for limb in 0..DIGEST_SIZE {
            builder.when(enabled).assert_eq(
                local.key_app_vk_digest[limb],
                self.profile.app_vk_digest[limb],
            );
        }
        let key_meta = self.scalar_expr::<AB>([
            AB::Expr::from_u32(TAG_KEY_META_V19),
            AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            local.shard_ordinal.into(),
            local.key_air_id_lo.into(),
            local.key_air_id_hi.into(),
            local.key_log_height.into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ]);
        self.lookup_hash(
            builder,
            key_meta,
            local.key_app_vk_digest.map(Into::into),
            local.hash_outputs[0],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.key_trace_layout_digest.map(Into::into),
            local.key_public_schema_digest.map(Into::into),
            local.hash_outputs[1],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.key_interaction_schema_digest.map(Into::into),
            local.key_code_class_digest.map(Into::into),
            local.hash_outputs[2],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[0].map(Into::into),
            local.key_relation_digest.map(Into::into),
            local.hash_outputs[3],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[1].map(Into::into),
            local.hash_outputs[2].map(Into::into),
            local.hash_outputs[4],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[3].map(Into::into),
            local.hash_outputs[4].map(Into::into),
            local.hash_outputs[5],
            enabled,
        );

        let replay_meta = self.scalar_expr::<AB>([
            AB::Expr::from_u32(TAG_REPLAY_META_V19),
            AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            local.ctx_segment_index_lo.into(),
            local.ctx_segment_index_hi.into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ]);
        self.lookup_hash(
            builder,
            replay_meta,
            local.key_relation_digest.map(Into::into),
            local.hash_outputs[6],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.previous_accumulator_digest.map(Into::into),
            local.next_accumulator_digest.map(Into::into),
            local.hash_outputs[7],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.previous_checkpoint_digest.map(Into::into),
            local.next_checkpoint_digest.map(Into::into),
            local.hash_outputs[8],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.opening_claim_digest.map(Into::into),
            local.replay_endpoint_digest.map(Into::into),
            local.hash_outputs[9],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[6].map(Into::into),
            local.hash_outputs[7].map(Into::into),
            local.hash_outputs[10],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[8].map(Into::into),
            local.hash_outputs[9].map(Into::into),
            local.hash_outputs[11],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[10].map(Into::into),
            local.hash_outputs[11].map(Into::into),
            local.hash_outputs[12],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[5].map(Into::into),
            local.hash_outputs[12].map(Into::into),
            local.hash_outputs[13],
            enabled,
        );
        for limb in 0..DIGEST_SIZE {
            builder.when(enabled).assert_eq(
                local.replay_binding_digest[limb],
                local.hash_outputs[13][limb],
            );
        }
        self.certified_replay_bus.lookup_key(
            builder,
            CertifiedWarpReplayMessageV19 {
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                segment_index_lo: local.ctx_segment_index_lo.into(),
                segment_index_hi: local.ctx_segment_index_hi.into(),
                update_index_lo: local.update_index_lo.into(),
                update_index_hi: local.update_index_hi.into(),
                shard_ordinal: local.shard_ordinal.into(),
                source_forest_root: local.ctx_source_forest_root.map(Into::into),
                key_digest: local.hash_outputs[5].map(Into::into),
                relation_digest: local.key_relation_digest.map(Into::into),
                opening_claim_digest: local.opening_claim_digest.map(Into::into),
                fresh_instance_digest: local.fresh_instance_digest.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                prior_root: local.prior_root.map(Into::into),
                fresh_root: local.fresh_root.map(Into::into),
                next_root: local.next_root.map(Into::into),
                previous_accumulator_digest: local.previous_accumulator_digest.map(Into::into),
                next_accumulator_digest: local.next_accumulator_digest.map(Into::into),
                authenticated_batching_claim: local.authenticated_batching_claim.map(Into::into),
                previous_checkpoint_digest: local.previous_checkpoint_digest.map(Into::into),
                next_checkpoint_digest: local.next_checkpoint_digest.map(Into::into),
                replay_endpoint_digest: local.replay_endpoint_digest.map(Into::into),
                replay_binding_digest: local.replay_binding_digest.map(Into::into),
            },
            enabled,
        );

        self.lookup_hash(
            builder,
            local.previous_accumulator_digest.map(Into::into),
            local.previous_checkpoint_digest.map(Into::into),
            local.hash_outputs[14],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.next_accumulator_digest.map(Into::into),
            local.next_checkpoint_digest.map(Into::into),
            local.hash_outputs[15],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[5].map(Into::into),
            local.hash_outputs[14].map(Into::into),
            local.hash_outputs[16],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[5].map(Into::into),
            local.hash_outputs[15].map(Into::into),
            local.hash_outputs[17],
            enabled,
        );
        for limb in 0..DIGEST_SIZE {
            builder.when(enabled).assert_eq(
                local.previous_leaf_digest[limb],
                local.hash_outputs[16][limb],
            );
            builder
                .when(enabled)
                .assert_eq(local.next_leaf_digest[limb], local.hash_outputs[17][limb]);
        }
        self.lookup_hash(
            builder,
            local.hash_outputs[5].map(Into::into),
            local.hash_outputs[13].map(Into::into),
            local.hash_outputs[18],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.manifest_before.map(Into::into),
            local.hash_outputs[18].map(Into::into),
            local.hash_outputs[19],
            enabled,
        );
        for limb in 0..DIGEST_SIZE {
            builder
                .when(enabled)
                .assert_eq(local.manifest_after[limb], local.hash_outputs[19][limb]);
            builder.when_transition().when(enabled).assert_eq(
                next.previous_leaf_digest[limb],
                local.previous_leaf_digest[limb],
            );
            builder
                .when_transition()
                .when(enabled)
                .assert_eq(next.next_leaf_digest[limb], local.next_leaf_digest[limb]);
            builder
                .when_transition()
                .when(enabled)
                .assert_eq(next.hash_outputs[5][limb], local.hash_outputs[5][limb]);
        }
        builder
            .when_transition()
            .when(enabled)
            .assert_eq(next.shard_ordinal, local.shard_ordinal);
        builder
            .when_transition()
            .when(enabled)
            .assert_one(next.path_is_first);
        for bit in 0..LIMB_BITS {
            builder
                .when_transition()
                .when(enabled)
                .assert_eq(next.shard_ordinal_bits[bit], local.shard_ordinal_bits[bit]);
        }
    }

    fn eval_path<AB: AirBuilder<F = F> + InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &HistoryRowColsV19<AB::Var>,
        next: &HistoryRowColsV19<AB::Var>,
    ) where
        AB::Var: Copy,
    {
        let enabled = local.is_path;
        builder.assert_bool(local.path_is_first);
        builder.assert_bool(local.path_is_last);
        builder.assert_bool(local.path_index_bit);
        let final_depth = AB::Expr::from_u8(self.profile.product_tree_height - 1);
        let depth_delta = local.path_depth - final_depth;
        builder
            .when(enabled * local.path_is_last)
            .assert_zero(depth_delta.clone());
        builder.when(enabled).assert_eq(
            depth_delta * local.path_not_last_inv,
            AB::Expr::ONE - local.path_is_last,
        );
        builder
            .when(enabled * local.path_is_first)
            .assert_zero(local.path_depth);
        builder
            .when(enabled * local.path_is_first)
            .assert_zero(local.path_index_before);
        builder
            .when(enabled * local.path_is_first)
            .assert_one(local.path_bit_weight);
        builder.when(enabled).assert_eq(
            local.path_index_after,
            local.path_index_before + local.path_index_bit * local.path_bit_weight,
        );
        builder
            .when(enabled * local.path_is_last)
            .assert_eq(local.path_index_after, local.shard_ordinal);

        let bit: AB::Expr = local.path_index_bit.into();
        let left_old = core::array::from_fn(|limb| {
            (AB::Expr::ONE - bit.clone()) * local.path_old_node[limb]
                + bit.clone() * local.path_product_sibling[limb]
        });
        let right_old = core::array::from_fn(|limb| {
            bit.clone() * local.path_old_node[limb]
                + (AB::Expr::ONE - bit.clone()) * local.path_product_sibling[limb]
        });
        let left_new = core::array::from_fn(|limb| {
            (AB::Expr::ONE - bit.clone()) * local.path_new_node[limb]
                + bit.clone() * local.path_product_sibling[limb]
        });
        let right_new = core::array::from_fn(|limb| {
            bit.clone() * local.path_new_node[limb]
                + (AB::Expr::ONE - bit.clone()) * local.path_product_sibling[limb]
        });
        let left_catalog = core::array::from_fn(|limb| {
            (AB::Expr::ONE - bit.clone()) * local.path_catalog_node[limb]
                + bit.clone() * local.path_catalog_sibling[limb]
        });
        let right_catalog = core::array::from_fn(|limb| {
            bit.clone() * local.path_catalog_node[limb]
                + (AB::Expr::ONE - bit.clone()) * local.path_catalog_sibling[limb]
        });
        self.lookup_hash(builder, left_old, right_old, local.hash_outputs[0], enabled);
        self.lookup_hash(builder, left_new, right_new, local.hash_outputs[1], enabled);
        self.lookup_hash(
            builder,
            left_catalog,
            right_catalog,
            local.hash_outputs[2],
            enabled,
        );
        for limb in 0..DIGEST_SIZE {
            builder
                .when(enabled)
                .assert_eq(local.path_old_parent[limb], local.hash_outputs[0][limb]);
            builder
                .when(enabled)
                .assert_eq(local.path_new_parent[limb], local.hash_outputs[1][limb]);
            builder
                .when(enabled)
                .assert_eq(local.path_catalog_parent[limb], local.hash_outputs[2][limb]);
            builder
                .when(enabled)
                .assert_eq(local.manifest_after[limb], local.manifest_before[limb]);
            builder
                .when(enabled * local.path_is_first)
                .assert_eq(local.path_old_node[limb], local.previous_leaf_digest[limb]);
            builder
                .when(enabled * local.path_is_first)
                .assert_eq(local.path_new_node[limb], local.next_leaf_digest[limb]);
            builder
                .when(enabled * local.path_is_first)
                .assert_eq(local.path_catalog_node[limb], local.hash_outputs[5][limb]);
            builder.when(enabled * local.path_is_last).assert_eq(
                local.path_old_parent[limb],
                local.state_before_product[limb],
            );
            builder.when(enabled * local.path_is_last).assert_eq(
                local.path_catalog_parent[limb],
                self.profile.shard_catalog_root[limb],
            );
        }

        let continues = enabled * (AB::Expr::ONE - local.path_is_last);
        builder
            .when_transition()
            .when(continues.clone())
            .assert_eq(next.path_depth, local.path_depth + AB::F::ONE);
        builder
            .when_transition()
            .when(continues.clone())
            .assert_eq(next.path_index_before, local.path_index_after);
        builder
            .when_transition()
            .when(continues.clone())
            .assert_eq(next.path_bit_weight, local.path_bit_weight * AB::F::TWO);
        builder
            .when_transition()
            .when(continues.clone())
            .assert_zero(next.path_is_first);
        for limb in 0..DIGEST_SIZE {
            builder
                .when_transition()
                .when(continues.clone())
                .assert_eq(next.path_old_node[limb], local.path_old_parent[limb]);
            builder
                .when_transition()
                .when(continues.clone())
                .assert_eq(next.path_new_node[limb], local.path_new_parent[limb]);
            builder.when_transition().when(continues.clone()).assert_eq(
                next.path_catalog_node[limb],
                local.path_catalog_parent[limb],
            );
            builder.when_transition().when(continues.clone()).assert_eq(
                next.previous_leaf_digest[limb],
                local.previous_leaf_digest[limb],
            );
            builder
                .when_transition()
                .when(continues.clone())
                .assert_eq(next.next_leaf_digest[limb], local.next_leaf_digest[limb]);
            builder
                .when_transition()
                .when(continues.clone())
                .assert_eq(next.hash_outputs[5][limb], local.hash_outputs[5][limb]);
        }
        builder
            .when_transition()
            .when(continues.clone())
            .assert_eq(next.shard_ordinal, local.shard_ordinal);
        for bit_index in 0..LIMB_BITS {
            builder.when_transition().when(continues.clone()).assert_eq(
                next.shard_ordinal_bits[bit_index],
                local.shard_ordinal_bits[bit_index],
            );
        }
    }

    fn eval_logup<AB: AirBuilder<F = F> + InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &HistoryRowColsV19<AB::Var>,
    ) where
        AB::Var: Copy,
    {
        let enabled = local.is_logup;
        builder.when(enabled).assert_eq(
            local.logup_mode_tag,
            AB::Expr::from_u32(super::LOGUP_ONLY_MODE_TAG_V19),
        );
        for limb in 0..DIGEST_SIZE {
            builder.when(enabled).assert_eq(
                local.logup_segment_openings_digest[limb],
                local.ctx_segment_openings_digest[limb],
            );
        }
        self.logup_only_bus.lookup_key(
            builder,
            LogUpOnlyHistoryMessageV19 {
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                mode_tag: local.logup_mode_tag.into(),
                segment_index_lo: local.ctx_segment_index_lo.into(),
                segment_index_hi: local.ctx_segment_index_hi.into(),
                app_vk_digest: self.profile.app_vk_digest.map(Into::into),
                source_forest_root: local.ctx_source_forest_root.map(Into::into),
                segment_openings_digest: local.logup_segment_openings_digest.map(Into::into),
                verifier_endpoint: local.logup_verifier_endpoint.map(Into::into),
                checkpoint_digest: local.logup_checkpoint_digest.map(Into::into),
            },
            enabled,
        );

        let endpoint = core::array::from_fn(|index| match index {
            0 => AB::Expr::from_u32(TAG_LOGUP_ENDPOINT_V19),
            1..=D_EF => local.logup_verifier_endpoint[index - 1].into(),
            _ => AB::Expr::ZERO,
        });
        self.lookup_hash(
            builder,
            endpoint,
            self.zero_digest::<AB>(),
            local.hash_outputs[0],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.logup_segment_openings_digest.map(Into::into),
            local.hash_outputs[0].map(Into::into),
            local.hash_outputs[1],
            enabled,
        );
        let meta = self.scalar_expr::<AB>([
            AB::Expr::from_u32(TAG_LOGUP_META_V19),
            AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            local.logup_mode_tag.into(),
            local.ctx_segment_index_lo.into(),
            local.ctx_segment_index_hi.into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ]);
        self.lookup_hash(
            builder,
            meta,
            local.logup_checkpoint_digest.map(Into::into),
            local.hash_outputs[2],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[2].map(Into::into),
            local.hash_outputs[1].map(Into::into),
            local.hash_outputs[3],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.manifest_before.map(Into::into),
            local.hash_outputs[3].map(Into::into),
            local.hash_outputs[4],
            enabled,
        );
        for limb in 0..DIGEST_SIZE {
            builder
                .when(enabled)
                .assert_eq(local.manifest_after[limb], local.hash_outputs[4][limb]);
        }
    }

    fn eval_segment_end<AB: AirBuilder<F = F> + AirBuilderWithPublicValues + InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &HistoryRowColsV19<AB::Var>,
        public: &HistoryPublicValuesV19<AB::PublicVar>,
    ) where
        AB::Var: Copy,
        AB::PublicVar: Copy,
    {
        let enabled = local.is_segment_end;
        builder
            .when(enabled)
            .assert_eq(local.shard_count_after, local.ctx_expected_shards);
        builder.when(enabled).assert_eq(
            local.ctx_exit_code,
            (AB::Expr::ONE - local.ctx_terminates) * AB::Expr::from_u32(DEFAULT_SUSPEND_EXIT_CODE),
        );
        for limb in 0..DIGEST_SIZE {
            builder.when(enabled).assert_eq(
                local.state_before_product[limb],
                local.ctx_next_product[limb],
            );
        }
        let terminal = AB::Expr::from(enabled) * local.ctx_terminates;
        builder
            .when(terminal.clone())
            .assert_eq(local.ctx_final_pc, public.final_pc);
        builder
            .when(terminal.clone())
            .assert_eq(local.ctx_exit_code, public.final_exit_code);
        for limb in 0..DIGEST_SIZE {
            builder.when(terminal.clone()).assert_eq(
                local.ctx_final_memory_root[limb],
                public.final_memory_root[limb],
            );
            builder.when(terminal.clone()).assert_eq(
                local.ctx_program_fingerprint_digest[limb],
                public.program_fingerprint_digest[limb],
            );
        }
        let end = self.scalar_expr::<AB>([
            AB::Expr::from_u32(TAG_SEGMENT_END_V19),
            AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            local.ctx_segment_index_lo.into(),
            local.ctx_segment_index_hi.into(),
            local.ctx_expected_shards.into(),
            local.ctx_terminates.into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ]);
        self.lookup_hash(
            builder,
            end,
            self.zero_digest::<AB>(),
            local.hash_outputs[0],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.manifest_before.map(Into::into),
            local.hash_outputs[0].map(Into::into),
            local.hash_outputs[1],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.state_before_history.map(Into::into),
            local.hash_outputs[1].map(Into::into),
            local.hash_outputs[2],
            enabled,
        );
        for limb in 0..DIGEST_SIZE {
            builder
                .when(enabled)
                .assert_eq(local.manifest_after[limb], local.hash_outputs[1][limb]);
            builder
                .when(enabled)
                .assert_eq(local.ctx_manifest_digest[limb], local.hash_outputs[1][limb]);
        }
    }

    fn eval_final<AB: AirBuilder<F = F> + AirBuilderWithPublicValues + InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &HistoryRowColsV19<AB::Var>,
        public: &HistoryPublicValuesV19<AB::PublicVar>,
    ) where
        AB::Var: Copy,
        AB::PublicVar: Copy,
    {
        builder.when_first_row().assert_eq(
            public.protocol_version,
            AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
        );
        builder
            .when_first_row()
            .assert_bool(public.initial_terminated);
        builder
            .when_first_row()
            .assert_bool(public.final_terminated);
        builder.when_first_row().assert_bool(public.terminal_chunk);
        builder
            .when_first_row()
            .assert_zero((AB::Expr::ONE - public.terminal_chunk.into()) * public.final_pc.into());
        builder.when_first_row().assert_zero(
            (AB::Expr::ONE - public.terminal_chunk.into()) * public.final_exit_code.into(),
        );
        for limb in 0..DIGEST_SIZE {
            builder.when_first_row().assert_eq(
                public.profile_digest[limb],
                self.profile.profile_digest[limb],
            );
            builder
                .when_first_row()
                .assert_eq(public.app_vk_digest[limb], self.profile.app_vk_digest[limb]);
            builder.when_first_row().assert_zero(
                (AB::Expr::ONE - public.terminal_chunk.into())
                    * public.final_memory_root[limb].into(),
            );
            builder.when_first_row().assert_zero(
                (AB::Expr::ONE - public.terminal_chunk.into())
                    * public.program_fingerprint_digest[limb].into(),
            );
        }

        let enabled = local.is_final;
        builder
            .when(enabled)
            .assert_eq(local.state_after_count_lo, public.final_segment_count_lo);
        builder
            .when(enabled)
            .assert_eq(local.state_after_count_hi, public.final_segment_count_hi);
        builder
            .when(enabled)
            .assert_eq(local.state_after_terminated, public.final_terminated);
        builder
            .when(AB::Expr::from(enabled) * public.terminal_chunk.into())
            .assert_one(local.state_after_terminated);
        for limb in 0..DIGEST_SIZE {
            builder
                .when(enabled)
                .assert_eq(local.state_after_vm[limb], public.final_vm_state[limb]);
            builder.when(enabled).assert_eq(
                local.state_after_product[limb],
                public.final_product_state_root[limb],
            );
            builder.when(enabled).assert_eq(
                local.state_after_history[limb],
                public.final_history_root[limb],
            );
            builder.when(enabled).assert_eq(
                local.state_after_public_values[limb],
                public.final_public_values_digest[limb],
            );
        }
        let metadata = self.scalar_expr::<AB>([
            AB::Expr::from_u32(TAG_TERMINAL_V19),
            AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            local.state_after_count_lo.into(),
            local.state_after_count_hi.into(),
            local.state_after_terminated.into(),
            public.terminal_chunk.into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ]);
        self.lookup_hash(
            builder,
            metadata,
            public.profile_digest.map(Into::into),
            local.hash_outputs[0],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.state_after_vm.map(Into::into),
            local.state_after_product.map(Into::into),
            local.hash_outputs[1],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.state_after_history.map(Into::into),
            local.state_after_public_values.map(Into::into),
            local.hash_outputs[2],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[1].map(Into::into),
            local.hash_outputs[2].map(Into::into),
            local.hash_outputs[3],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[0].map(Into::into),
            local.hash_outputs[3].map(Into::into),
            local.hash_outputs[4],
            enabled,
        );
        for limb in 0..DIGEST_SIZE {
            builder.when(enabled).assert_eq(
                local.hash_outputs[4][limb],
                public.terminal_certificate_digest[limb],
            );
            builder
                .when(enabled)
                .assert_eq(local.manifest_after[limb], local.manifest_before[limb]);
        }
    }

    fn assert_u16<AB: AirBuilder<F = F>>(
        &self,
        builder: &mut AB,
        enabled: AB::Var,
        value: AB::Var,
        bits: &[AB::Var; LIMB_BITS],
    ) where
        AB::Var: Copy,
    {
        self.assert_bits(builder, enabled, bits);
        let recomposed = self.bits_expr::<AB>(bits);
        builder
            .when(enabled)
            .assert_zero(AB::Expr::from(value) - recomposed);
    }

    fn assert_bits<AB: AirBuilder<F = F>>(
        &self,
        builder: &mut AB,
        enabled: AB::Var,
        bits: &[AB::Var; LIMB_BITS],
    ) where
        AB::Var: Copy,
    {
        for &bit in bits {
            builder.when(enabled).assert_bool(bit);
        }
    }

    fn bits_expr<AB: AirBuilder<F = F>>(&self, bits: &[AB::Var; LIMB_BITS]) -> AB::Expr
    where
        AB::Var: Copy,
    {
        bits.iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |sum, (index, bit)| {
                sum + *bit * AB::Expr::from_u32(1u32 << index)
            })
    }

    fn lookup_hash<AB: AirBuilder<F = F> + InteractionBuilder>(
        &self,
        builder: &mut AB,
        left: [AB::Expr; DIGEST_SIZE],
        right: [AB::Expr; DIGEST_SIZE],
        output: [AB::Var; DIGEST_SIZE],
        enabled: AB::Var,
    ) where
        AB::Var: Copy,
    {
        self.compress_bus.lookup_key(
            builder,
            HistoryPoseidon2CompressMessageV19 {
                input: core::array::from_fn(|index| {
                    if index < DIGEST_SIZE {
                        left[index].clone()
                    } else {
                        right[index - DIGEST_SIZE].clone()
                    }
                }),
                output: output.map(Into::into),
            },
            enabled,
        );
    }

    fn scalar_expr<AB: AirBuilder<F = F>>(
        &self,
        values: [AB::Expr; DIGEST_SIZE],
    ) -> [AB::Expr; DIGEST_SIZE] {
        values
    }

    fn zero_digest<AB: AirBuilder<F = F>>(&self) -> [AB::Expr; DIGEST_SIZE] {
        core::array::from_fn(|_| AB::Expr::ZERO)
    }
}

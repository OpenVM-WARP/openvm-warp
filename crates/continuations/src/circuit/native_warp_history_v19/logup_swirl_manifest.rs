//! Canonical CPU source-manifest forest and direct-instance providers.
//!
//! The profile is fixed in the History verifying key and is constructed from
//! validated direct-AIR relation descriptions plus the exact active-shard
//! plan. Main traces contain only fresh roots, instance values, and hash
//! states. The AIR recomputes the CPU v19 leaf/forest construction and the
//! source-prefix transcript word for word.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_circuit::arch::{CONNECTOR_AIR_ID, MERKLE_AIR_ID, POSEIDON2_WIDTH};
use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit::{
    bus::{
        Poseidon2CompressBus, Poseidon2CompressMessage, Poseidon2PermuteBus,
        Poseidon2PermuteMessage, PublicValuesBus, PublicValuesBusMessage, TranscriptBus,
    },
    utils::poseidon2_hash_slice_with_states,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, PermutationCheckBus},
    native_warp::DirectAirPesatRelationDescription,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, DIGEST_SIZE, F,
};
use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::{
    SetupPcsSourceManifestBusV3, SetupPcsSourceManifestMessageV3,
    VerifiedDirectAirPublicValueBusV19, VerifiedDirectAirPublicValueMessageV19,
    VerifiedSourceForestLeafBusV19, VerifiedSourceForestLeafMessageV19,
    NATIVE_WARP_HISTORY_PROTOCOL_V19, SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3,
};

pub const SEGMENT_PREFIX_TAG_V19: u64 = 0x4e57_5345_4750_0013;
pub const SOURCE_INSTANCE_TAG_V19: u64 = 0x4e57_5349_4e53_0013;
pub const SOURCE_LEAF_TAG_V19: u64 = 0x4e57_5346_4c46_0013;
pub const SOURCE_FOREST_TAG_V19: u64 = 0x4e57_5346_5254_0013;
pub const LOGUP_START_BOUNDARY_TAG_V19: u64 = 0x4e57_4c53_5441_0013;
pub const LOGUP_END_BOUNDARY_TAG_V19: u64 = 0x4e57_4c45_4e44_0013;

macro_rules! define_permutation_bus {
    ($Bus:ident, $Message:ident) => {
        #[derive(Copy, Clone, Debug)]
        pub struct $Bus(PermutationCheckBus);

        impl $Bus {
            #[must_use]
            pub fn new(bus_index: BusIndex) -> Self {
                Self(PermutationCheckBus::new(bus_index))
            }

            pub fn send<AB: InteractionBuilder>(
                &self,
                builder: &mut AB,
                message: $Message<impl Into<AB::Expr> + Clone>,
                enabled: impl Into<AB::Expr>,
            ) {
                self.0.send(builder, message.to_vec(), enabled);
            }

            pub fn receive<AB: InteractionBuilder>(
                &self,
                builder: &mut AB,
                message: $Message<impl Into<AB::Expr> + Clone>,
                enabled: impl Into<AB::Expr>,
            ) {
                self.0.receive(builder, message.to_vec(), enabled);
            }
        }
    };
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct SourceInstanceDigestMessageV19<T> {
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub shard_ordinal: T,
    pub shard_id: T,
    pub digest: [T; DIGEST_SIZE],
}
define_permutation_bus!(SourceInstanceDigestBusV19, SourceInstanceDigestMessageV19);

/// One-way V2 adapter input for setup-fixed source public values. The V2
/// bridge sends values constrained by the authenticated fresh beta; the v19
/// source-instance sponge consumes them and derives its own canonical digest.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct FixedSourcePublicValueMessageV19<T> {
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub public_value_index: T,
    pub value: T,
}
define_permutation_bus!(
    FixedSourcePublicValueBusV19,
    FixedSourcePublicValueMessageV19
);

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct SourceForestNodeMessageV19<T> {
    pub proof_index: T,
    pub level: T,
    pub index: T,
    pub digest: [T; DIGEST_SIZE],
}
define_permutation_bus!(SourceForestNodeBusV19, SourceForestNodeMessageV19);

/// Cursor handed to the recursion LogUp-only module after the exact source
/// prefix and its discarded row-alignment sample.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct SourcePrefixEndMessageV19<T> {
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub end_tidx: T,
}
define_permutation_bus!(SourcePrefixEndBusV19, SourcePrefixEndMessageV19);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectLogUpSourceEntryProfileV19 {
    pub shard_id: u32,
    pub air_id: u32,
    pub relation_digest: [F; DIGEST_SIZE],
    pub log_height: u8,
    pub cached_width: u32,
    pub log_message_len: u8,
    pub log_codeword_len: u8,
    pub public_values_len: u32,
    pub boundary_values_len: u32,
}

impl DirectLogUpSourceEntryProfileV19 {
    /// Describe one setup-admitted fixed multi-AIR source.  Unlike
    /// [`Self::from_relation`], `air_id` is only a transcript namespace: the
    /// source instance digest is supplied by the certified fixed-relation
    /// VACC input, not by one child `PublicValuesBus` row.
    pub fn fixed_multi_air(
        relation_digest: [F; DIGEST_SIZE],
        log_message_len: u8,
        log_codeword_len: u8,
        source_public_values_len: u32,
    ) -> Result<Self, &'static str> {
        if log_message_len == 0
            || log_codeword_len < log_message_len
            || source_public_values_len == 0
        {
            return Err("invalid fixed multi-AIR source profile");
        }
        Ok(Self {
            shard_id: 0,
            air_id: u32::MAX,
            relation_digest,
            log_height: log_message_len,
            cached_width: 0,
            log_message_len,
            log_codeword_len,
            // PESAT explicit coordinate zero is the distinguished constant
            // one. It is constrained by the relation and deliberately absent
            // from the source-instance digest; only coordinates `1..` belong
            // to this public-value payload.
            public_values_len: source_public_values_len,
            boundary_values_len: 0,
        })
    }

    pub fn from_relation(
        shard_id: u32,
        relation: &DirectAirPesatRelationDescription<[F; DIGEST_SIZE]>,
    ) -> Result<Self, &'static str> {
        let key = &relation.shard_key;
        if relation.exact_max_degree > 5
            || relation.warp_degree_envelope > 5
            || key.code_class.log_message_len == 0
            || key.code_class.log_codeword_len < key.code_class.log_message_len
        {
            return Err("invalid protocol-v19 direct relation profile");
        }
        Ok(Self {
            shard_id,
            air_id: key.air_id,
            relation_digest: key.relation_digest,
            log_height: key.log_height,
            cached_width: key
                .trace_layout
                .cached_main_widths
                .first()
                .copied()
                .unwrap_or(0),
            log_message_len: key.code_class.log_message_len,
            log_codeword_len: key.code_class.log_codeword_len,
            public_values_len: key.public_schema.public_values_len,
            boundary_values_len: key.public_schema.boundary_values_len,
        })
    }
}

#[derive(Clone, Debug)]
struct PlannedSourceV19 {
    proof_index: u32,
    segment_index: u32,
    shard_ordinal: u16,
    shard_count: u16,
    entry: DirectLogUpSourceEntryProfileV19,
    range_start: u32,
    range_end: u32,
}

#[derive(Clone, Debug)]
pub struct DirectLogUpSourceManifestProfileV19 {
    pub app_vk_digest: [F; DIGEST_SIZE],
    pub registry_digest: [F; DIGEST_SIZE],
    planned: Arc<[PlannedSourceV19]>,
    setup_pcs_source_manifest_bus: Option<SetupPcsSourceManifestBusV3>,
    /// `Some(4)` only for the fixed HLeaf runtime-occupancy variant.  This is
    /// setup identity: the four planned local slots remain fixed while their
    /// `Source*ColsV19::active` multiplicities form a runtime prefix.
    runtime_capacity_v4: Option<usize>,
}

impl DirectLogUpSourceManifestProfileV19 {
    pub fn new(
        app_vk_digest: [F; DIGEST_SIZE],
        registry_digest: [F; DIGEST_SIZE],
        entries: Vec<DirectLogUpSourceEntryProfileV19>,
        segment_shard_ids: &[Vec<u32>],
    ) -> Result<Self, &'static str> {
        Self::new_with_segment_offset(
            app_vk_digest,
            registry_digest,
            entries,
            segment_shard_ids,
            0,
        )
    }

    /// Build a bounded verifier profile whose proof slots are dense within
    /// this chunk while segment identifiers remain global across the complete
    /// execution history.
    pub fn new_with_segment_offset(
        app_vk_digest: [F; DIGEST_SIZE],
        registry_digest: [F; DIGEST_SIZE],
        entries: Vec<DirectLogUpSourceEntryProfileV19>,
        segment_shard_ids: &[Vec<u32>],
        segment_start: u32,
    ) -> Result<Self, &'static str> {
        if entries.is_empty() || segment_shard_ids.is_empty() {
            return Err("empty source-manifest profile");
        }
        for (index, entry) in entries.iter().enumerate() {
            if entry.shard_id as usize != index {
                return Err("source registry is not canonical");
            }
        }
        let mut planned = Vec::new();
        for (proof_index, ids) in segment_shard_ids.iter().enumerate() {
            if ids.is_empty() || ids.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err("invalid segment source order");
            }
            let shard_count = u16::try_from(ids.len()).map_err(|_| "too many segment sources")?;
            let mut range = 0u64;
            for (ordinal, &shard_id) in ids.iter().enumerate() {
                let entry = entries
                    .get(shard_id as usize)
                    .ok_or("segment source outside registry")?
                    .clone();
                let width = 1u64
                    .checked_shl(entry.log_codeword_len.into())
                    .ok_or("source codeword range overflow")?;
                let end = range.checked_add(width).ok_or("source range overflow")?;
                if end >= 0x7800_0001u64 {
                    return Err("source range is not injective in BabyBear");
                }
                let proof_index =
                    u32::try_from(proof_index).map_err(|_| "source proof index overflow")?;
                let segment_index = segment_start
                    .checked_add(proof_index)
                    .ok_or("source segment index overflow")?;
                planned.push(PlannedSourceV19 {
                    proof_index,
                    segment_index,
                    shard_ordinal: ordinal as u16,
                    shard_count,
                    entry,
                    range_start: range as u32,
                    range_end: end as u32,
                });
                range = end;
            }
        }
        Ok(Self {
            app_vk_digest,
            registry_digest,
            planned: planned.into(),
            setup_pcs_source_manifest_bus: None,
            runtime_capacity_v4: None,
        })
    }

    /// Build one selector-free fixed multi-AIR source in each of four local
    /// HLeaf slots. Occupancy is supplied later by the main traces; it is not
    /// represented in this profile or its preprocessed matrices.
    pub fn new_fixed_capacity_v4(
        app_vk_digest: [F; DIGEST_SIZE],
        registry_digest: [F; DIGEST_SIZE],
        entry: DirectLogUpSourceEntryProfileV19,
    ) -> Result<Self, &'static str> {
        if entry.shard_id != 0 || entry.air_id != u32::MAX {
            return Err("fixed-capacity source requires one canonical multi-AIR entry");
        }
        let mut profile = Self::new(
            app_vk_digest,
            registry_digest,
            vec![entry],
            &vec![vec![0], vec![0], vec![0], vec![0]],
        )?;
        profile.runtime_capacity_v4 = Some(4);
        Ok(profile)
    }

    #[must_use]
    pub fn source_count(&self) -> usize {
        self.planned.len()
    }

    #[must_use]
    pub fn segment_start(&self) -> u32 {
        self.planned
            .first()
            .map_or(0, |source| source.segment_index)
    }

    #[must_use]
    pub const fn runtime_capacity_v4(&self) -> Option<usize> {
        self.runtime_capacity_v4
    }

    /// Enable the constrained setup-PCS provenance output without changing
    /// legacy callers.  The authority profile admits exactly one complete
    /// fixed multi-AIR source per transition; emitting a last-shard-only
    /// statement for a heterogeneous forest would be ambiguous and is
    /// rejected here.
    pub fn with_setup_pcs_source_manifest_bus_v3(
        mut self,
        bus: SetupPcsSourceManifestBusV3,
    ) -> Result<Self, &'static str> {
        if self
            .planned
            .iter()
            .any(|source| source.shard_count != 1 || source.shard_ordinal != 0)
        {
            return Err("setup PCS source provenance requires one source per transition");
        }
        self.setup_pcs_source_manifest_bus = Some(bus);
        Ok(self)
    }

    #[must_use]
    pub fn setup_pcs_source_manifest_bus_v3(&self) -> Option<SetupPcsSourceManifestBusV3> {
        self.setup_pcs_source_manifest_bus
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectLogUpFreshSourceRecordV19 {
    pub shard_id: u32,
    pub source_root: [F; DIGEST_SIZE],
    pub expected_instance_digest: [F; DIGEST_SIZE],
    pub public_values: Vec<F>,
    pub boundary_values: Vec<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectLogUpSegmentSourceRecordV19 {
    pub proof_index: u32,
    pub segment_index: u32,
    pub prefix_start_tidx: u32,
    pub alignment_sample: [F; 4],
    pub segment_openings_digest: [F; DIGEST_SIZE],
    pub sources: Vec<DirectLogUpFreshSourceRecordV19>,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
struct SourcePlanPrepColsV19<T> {
    active: T,
    is_segment_first: T,
    is_segment_last: T,
    forest_depth: T,
    proof_index: T,
    proof_index_lo: T,
    proof_index_hi: T,
    segment_index_lo: T,
    segment_index_hi: T,
    shard_ordinal: T,
    shard_count: T,
    shard_id: T,
    air_id: T,
    relation_digest: [T; DIGEST_SIZE],
    log_height: T,
    cached_width: T,
    log_message_len: T,
    log_codeword_len: T,
    public_values_len: T,
    boundary_values_len: T,
    range_start: T,
    range_end: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct SourceManifestColsV19<T> {
    pub active: T,
    pub source_root: [T; DIGEST_SIZE],
    pub instance_digest: [T; DIGEST_SIZE],
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub prefix_start_tidx: T,
    pub metadata_post_state: [T; POSEIDON2_WIDTH],
    pub instance_root_digest: [T; DIGEST_SIZE],
    pub relation_instance_root_digest: [T; DIGEST_SIZE],
    pub leaf_digest: [T; DIGEST_SIZE],
    pub count_post_state: [T; POSEIDON2_WIDTH],
    pub tree_root: [T; DIGEST_SIZE],
    pub alignment_sample: [T; 4],
}

#[derive(Clone, ColumnsAir)]
#[columns_via(SourceManifestColsV19<u8>)]
pub struct DirectLogUpSourceManifestAirV19 {
    pub profile: DirectLogUpSourceManifestProfileV19,
    pub transcript_bus: TranscriptBus,
    pub permute_bus: Poseidon2PermuteBus,
    pub compress_bus: Poseidon2CompressBus,
    pub instance_bus: SourceInstanceDigestBusV19,
    pub node_bus: SourceForestNodeBusV19,
    pub leaf_bus: VerifiedSourceForestLeafBusV19,
    pub prefix_end_bus: SourcePrefixEndBusV19,
}

impl PartitionedBaseAir<F> for DirectLogUpSourceManifestAirV19 {
    fn common_main_width(&self) -> usize {
        SourceManifestColsV19::<F>::width()
    }
}
impl BaseAir<F> for DirectLogUpSourceManifestAirV19 {
    fn width(&self) -> usize {
        SourceManifestColsV19::<F>::width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = SourcePlanPrepColsV19::<F>::width();
        let height = self.profile.planned.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(height * width);
        for (row, source) in self.profile.planned.iter().enumerate() {
            let cols: &mut SourcePlanPrepColsV19<F> =
                values[row * width..(row + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.is_segment_first = F::from_bool(source.shard_ordinal == 0);
            cols.is_segment_last = F::from_bool(source.shard_ordinal + 1 == source.shard_count);
            cols.forest_depth = F::from_u32(source.shard_count.next_power_of_two().ilog2());
            cols.proof_index = F::from_u32(source.proof_index);
            cols.proof_index_lo = F::from_u16(source.proof_index as u16);
            cols.proof_index_hi = F::from_u16((source.proof_index >> 16) as u16);
            cols.segment_index_lo = F::from_u32(source.segment_index & 0xffff);
            cols.segment_index_hi = F::from_u32(source.segment_index >> 16);
            cols.shard_ordinal = F::from_u16(source.shard_ordinal);
            cols.shard_count = F::from_u16(source.shard_count);
            cols.shard_id = F::from_u32(source.entry.shard_id);
            cols.air_id = F::from_u32(source.entry.air_id);
            cols.relation_digest = source.entry.relation_digest;
            cols.log_height = F::from_u8(source.entry.log_height);
            cols.cached_width = F::from_u32(source.entry.cached_width);
            cols.log_message_len = F::from_u8(source.entry.log_message_len);
            cols.log_codeword_len = F::from_u8(source.entry.log_codeword_len);
            cols.public_values_len = F::from_u32(source.entry.public_values_len);
            cols.boundary_values_len = F::from_u32(source.entry.boundary_values_len);
            cols.range_start = F::from_u32(source.range_start);
            cols.range_end = F::from_u32(source.range_end);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}
impl BaseAirWithPublicValues<F> for DirectLogUpSourceManifestAirV19 {}

impl<AB: AirBuilder<F = F> + InteractionBuilder + PairBuilder> Air<AB>
    for DirectLogUpSourceManifestAirV19
{
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed();
        let prep_row = preprocessed.row_slice(0).expect("source manifest plan row");
        let prep: &SourcePlanPrepColsV19<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("source manifest row");
        let local: &SourceManifestColsV19<AB::Var> = (*row).borrow();
        let next_row = main.row_slice(1).expect("source manifest next row");
        let next: &SourceManifestColsV19<AB::Var> = (*next_row).borrow();
        builder.assert_bool(prep.active);
        builder.assert_bool(local.active);
        if self.profile.runtime_capacity_v4().is_some() {
            builder.when_first_row().assert_one(local.active);
            let mut occupancy_transition = builder.when_transition();
            occupancy_transition.assert_zero(
                (AB::Expr::ONE - AB::Expr::from(local.active)) * AB::Expr::from(next.active),
            );
            let inactive = AB::Expr::from(prep.active) - AB::Expr::from(local.active);
            for value in (*row).iter().skip(1) {
                builder.when(inactive.clone()).assert_zero((*value).into());
            }
        } else {
            builder.assert_eq(local.active, prep.active);
        }
        let enabled = AB::Expr::from(local.active);
        builder.assert_bool(prep.is_segment_first);
        builder.assert_bool(prep.is_segment_last);
        let segment_first = enabled.clone() * AB::Expr::from(prep.is_segment_first);
        let segment_last = enabled.clone() * AB::Expr::from(prep.is_segment_last);
        let same_segment =
            AB::Expr::from(next.active) * (AB::Expr::ONE - AB::Expr::from(prep.is_segment_last));
        let mut transition_builder = builder.when_transition();
        let mut transition = transition_builder.when(same_segment);
        transition.assert_eq(next.prefix_start_tidx, local.prefix_start_tidx);
        for (next, local) in next
            .source_forest_root
            .iter()
            .zip(local.source_forest_root.iter())
            .chain(
                next.segment_openings_digest
                    .iter()
                    .zip(local.segment_openings_digest.iter()),
            )
            .chain(next.tree_root.iter().zip(local.tree_root.iter()))
            .chain(
                next.count_post_state
                    .iter()
                    .zip(local.count_post_state.iter()),
            )
            .chain(
                next.alignment_sample
                    .iter()
                    .zip(local.alignment_sample.iter()),
            )
        {
            transition.assert_eq(*next, *local);
        }

        self.instance_bus.receive(
            builder,
            SourceInstanceDigestMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                shard_ordinal: prep.shard_ordinal.into(),
                shard_id: prep.shard_id.into(),
                digest: local.instance_digest.map(Into::into),
            },
            enabled.clone(),
        );

        let metadata_input: [AB::Expr; POSEIDON2_WIDTH] =
            core::array::from_fn(|index| match index {
                0 => AB::Expr::from_u64(SOURCE_LEAF_TAG_V19),
                1 => prep.shard_id.into(),
                2 => prep.air_id.into(),
                3 => prep.log_height.into(),
                4 => prep.log_message_len.into(),
                5 => prep.log_codeword_len.into(),
                _ => AB::Expr::ZERO,
            });
        self.permute_bus.lookup_key(
            builder,
            Poseidon2PermuteMessage {
                input: metadata_input,
                output: local.metadata_post_state.map(Into::into),
            },
            enabled.clone(),
        );
        self.compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: join(local.instance_digest, local.source_root),
                output: local.instance_root_digest,
            },
            enabled.clone(),
        );
        self.compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: join(prep.relation_digest, local.instance_root_digest),
                output: local.relation_instance_root_digest,
            },
            enabled.clone(),
        );
        self.compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: core::array::from_fn(|index| {
                    if index < DIGEST_SIZE {
                        local.metadata_post_state[index].into()
                    } else {
                        local.relation_instance_root_digest[index - DIGEST_SIZE].into()
                    }
                }),
                output: local.leaf_digest.map(Into::into),
            },
            enabled.clone(),
        );
        self.node_bus.send(
            builder,
            SourceForestNodeMessageV19 {
                proof_index: prep.proof_index.into(),
                level: AB::Expr::ZERO,
                index: prep.shard_ordinal.into(),
                digest: local.leaf_digest.map(Into::into),
            },
            enabled.clone(),
        );

        // The root node is consumed once by the first source row.
        self.node_bus.receive(
            builder,
            SourceForestNodeMessageV19 {
                proof_index: prep.proof_index.into(),
                level: prep.forest_depth.into(),
                index: AB::Expr::ZERO,
                digest: local.tree_root.map(Into::into),
            },
            segment_first.clone(),
        );
        let count_input: [AB::Expr; POSEIDON2_WIDTH] = core::array::from_fn(|index| match index {
            0 => AB::Expr::from_u64(SOURCE_FOREST_TAG_V19),
            1 => prep.shard_count.into(),
            _ => AB::Expr::ZERO,
        });
        self.permute_bus.lookup_key(
            builder,
            Poseidon2PermuteMessage {
                input: count_input,
                output: local.count_post_state.map(Into::into),
            },
            segment_first.clone(),
        );
        self.compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: core::array::from_fn(|index| {
                    if index < DIGEST_SIZE {
                        local.count_post_state[index].into()
                    } else {
                        local.tree_root[index - DIGEST_SIZE].into()
                    }
                }),
                output: local.source_forest_root.map(Into::into),
            },
            segment_first.clone(),
        );

        self.leaf_bus.send(
            builder,
            VerifiedSourceForestLeafMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                shard_ordinal: prep.shard_ordinal.into(),
                air_id: prep.air_id.into(),
                relation_digest: prep.relation_digest.map(Into::into),
                log_height: prep.log_height.into(),
                cached_width: prep.cached_width.into(),
                log_message_len: prep.log_message_len.into(),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                source_root: local.source_root.map(Into::into),
                range_start: prep.range_start.into(),
                range_end: prep.range_end.into(),
            },
            enabled.clone(),
        );

        let header_start = AB::Expr::from(local.prefix_start_tidx);
        for (offset, value) in [
            AB::Expr::from_u64(SEGMENT_PREFIX_TAG_V19),
            AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            AB::Expr::from(prep.segment_index_lo)
                + AB::Expr::from_u32(1 << 16) * AB::Expr::from(prep.segment_index_hi),
        ]
        .into_iter()
        .enumerate()
        {
            self.transcript_bus.observe(
                builder,
                prep.proof_index,
                header_start.clone() + AB::Expr::from_usize(offset),
                value,
                segment_first.clone(),
            );
        }
        let mut header_offset = 3usize;
        for digest in [
            self.profile.app_vk_digest.map(Into::into),
            self.profile.registry_digest.map(Into::into),
            local.source_forest_root.map(Into::into),
        ] {
            self.transcript_bus.observe_commit(
                builder,
                prep.proof_index,
                header_start.clone() + AB::Expr::from_usize(header_offset),
                digest,
                segment_first.clone(),
            );
            header_offset += DIGEST_SIZE;
        }
        self.transcript_bus.observe(
            builder,
            prep.proof_index,
            header_start.clone() + AB::Expr::from_usize(header_offset),
            prep.shard_count,
            segment_first.clone(),
        );
        header_offset += 1;
        let source_tidx = header_start.clone()
            + AB::Expr::from_usize(header_offset)
            + AB::Expr::from(prep.shard_ordinal) * AB::Expr::from_usize(1 + 2 * DIGEST_SIZE);
        self.transcript_bus.observe(
            builder,
            prep.proof_index,
            source_tidx.clone(),
            prep.shard_id,
            enabled.clone(),
        );
        self.transcript_bus.observe_commit(
            builder,
            prep.proof_index,
            source_tidx.clone() + AB::Expr::ONE,
            local.instance_digest,
            enabled.clone(),
        );
        self.transcript_bus.observe_commit(
            builder,
            prep.proof_index,
            source_tidx + AB::Expr::from_usize(1 + DIGEST_SIZE),
            local.source_root,
            enabled.clone(),
        );
        let end_tidx = header_start
            + AB::Expr::from_usize(header_offset)
            + AB::Expr::from(prep.shard_count) * AB::Expr::from_usize(1 + 2 * DIGEST_SIZE);
        self.transcript_bus.observe(
            builder,
            prep.proof_index,
            end_tidx.clone(),
            AB::Expr::from_u64(LOGUP_START_BOUNDARY_TAG_V19),
            segment_last.clone(),
        );
        self.transcript_bus.sample_ext(
            builder,
            prep.proof_index,
            end_tidx.clone() + AB::Expr::ONE,
            local.alignment_sample,
            segment_last.clone(),
        );
        self.prefix_end_bus.send(
            builder,
            SourcePrefixEndMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                end_tidx: end_tidx + AB::Expr::from_usize(1 + 4),
            },
            segment_last.clone(),
        );
        if let Some(bus) = self.profile.setup_pcs_source_manifest_bus_v3() {
            bus.send(
                builder,
                SetupPcsSourceManifestMessageV3 {
                    protocol_version: AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
                    transition_index: [prep.proof_index_lo.into(), prep.proof_index_hi.into()],
                    segment_index: [prep.segment_index_lo.into(), prep.segment_index_hi.into()],
                    app_vk_digest: self.profile.app_vk_digest.map(Into::into),
                    relation_digest: prep.relation_digest.map(Into::into),
                    source_root: local.source_root.map(Into::into),
                    source_instance_digest: local.instance_digest.map(Into::into),
                    source_forest_root: local.source_forest_root.map(Into::into),
                    segment_openings_digest: local.segment_openings_digest.map(Into::into),
                },
                segment_last,
            );
        }
    }
}

const INSTANCE_KIND_CARRY: u8 = 0;
const INSTANCE_KIND_CONSTANT: u8 = 1;
const INSTANCE_KIND_PUBLIC_VALUE: u8 = 2;
const INSTANCE_KIND_BOUNDARY_VALUE: u8 = 3;

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
struct SourceInstancePrepColsV19<T> {
    active: T,
    is_first: T,
    is_last: T,
    is_fixed_multi_air: T,
    public_values_air_id: T,
    proof_index: T,
    segment_index_lo: T,
    segment_index_hi: T,
    shard_ordinal: T,
    shard_id: T,
    air_id: T,
    chunk_index: T,
    chunk_count: T,
    lane_is_carry: [T; DIGEST_SIZE],
    lane_is_constant: [T; DIGEST_SIZE],
    lane_is_public_value: [T; DIGEST_SIZE],
    lane_is_boundary_value: [T; DIGEST_SIZE],
    lane_emit_verified: [T; DIGEST_SIZE],
    lane_index: [T; DIGEST_SIZE],
    lane_constant: [T; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct SourceInstanceColsV19<T> {
    pub active: T,
    pub absorbed: [T; DIGEST_SIZE],
    pub pre_state: [T; POSEIDON2_WIDTH],
    pub post_state: [T; POSEIDON2_WIDTH],
}

#[derive(Clone, ColumnsAir)]
#[columns_via(SourceInstanceColsV19<u8>)]
pub struct DirectLogUpSourceInstanceAirV19 {
    pub profile: DirectLogUpSourceManifestProfileV19,
    pub permute_bus: Poseidon2PermuteBus,
    pub public_values_bus: PublicValuesBus,
    pub verified_public_values_bus: VerifiedDirectAirPublicValueBusV19,
    pub fixed_public_values_bus: FixedSourcePublicValueBusV19,
    pub fixed_public_values_air_id: Option<u32>,
    pub instance_bus: SourceInstanceDigestBusV19,
}

impl PartitionedBaseAir<F> for DirectLogUpSourceInstanceAirV19 {
    fn common_main_width(&self) -> usize {
        SourceInstanceColsV19::<F>::width()
    }
}
impl BaseAir<F> for DirectLogUpSourceInstanceAirV19 {
    fn width(&self) -> usize {
        SourceInstanceColsV19::<F>::width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let rows = instance_plan_rows(&self.profile);
        let width = SourceInstancePrepColsV19::<F>::width();
        let height = rows.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(height * width);
        for (row, plan) in rows.iter().enumerate() {
            let cols: &mut SourceInstancePrepColsV19<F> =
                values[row * width..(row + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.is_first = F::from_bool(plan.is_first);
            cols.is_last = F::from_bool(plan.is_last);
            cols.is_fixed_multi_air = F::from_bool(plan.source.entry.air_id == u32::MAX);
            cols.public_values_air_id = F::from_u32(if plan.source.entry.air_id == u32::MAX {
                self.fixed_public_values_air_id
                    .expect("fixed source public-values AIR")
            } else {
                plan.source.entry.air_id
            });
            cols.proof_index = F::from_u32(plan.source.proof_index);
            cols.segment_index_lo = F::from_u32(plan.source.segment_index & 0xffff);
            cols.segment_index_hi = F::from_u32(plan.source.segment_index >> 16);
            cols.shard_ordinal = F::from_u16(plan.source.shard_ordinal);
            cols.shard_id = F::from_u32(plan.source.entry.shard_id);
            cols.air_id = F::from_u32(plan.source.entry.air_id);
            cols.chunk_index = F::from_usize(plan.chunk_index);
            cols.chunk_count = F::from_usize(plan.chunk_count);
            for lane in 0..DIGEST_SIZE {
                cols.lane_is_carry[lane] =
                    F::from_bool(plan.lanes[lane].kind == INSTANCE_KIND_CARRY);
                cols.lane_is_constant[lane] =
                    F::from_bool(plan.lanes[lane].kind == INSTANCE_KIND_CONSTANT);
                cols.lane_is_public_value[lane] =
                    F::from_bool(plan.lanes[lane].kind == INSTANCE_KIND_PUBLIC_VALUE);
                cols.lane_is_boundary_value[lane] =
                    F::from_bool(plan.lanes[lane].kind == INSTANCE_KIND_BOUNDARY_VALUE);
                cols.lane_emit_verified[lane] = F::from_bool(
                    plan.lanes[lane].kind == INSTANCE_KIND_PUBLIC_VALUE
                        && ((plan.source.entry.air_id as usize == CONNECTOR_AIR_ID
                            && plan.lanes[lane].index < 4)
                            || (plan.source.entry.air_id as usize == MERKLE_AIR_ID
                                && plan.lanes[lane].index < (2 * DIGEST_SIZE) as u32)),
                );
                cols.lane_index[lane] = F::from_u32(plan.lanes[lane].index);
                cols.lane_constant[lane] = plan.lanes[lane].constant;
            }
        }
        Some(RowMajorMatrix::new(values, width))
    }
}
impl BaseAirWithPublicValues<F> for DirectLogUpSourceInstanceAirV19 {}

impl<AB: AirBuilder<F = F> + InteractionBuilder + PairBuilder> Air<AB>
    for DirectLogUpSourceInstanceAirV19
{
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed();
        let prep_row = preprocessed.row_slice(0).expect("source instance plan row");
        let prep: &SourceInstancePrepColsV19<AB::Var> = (*prep_row).borrow();
        let next_prep_row = preprocessed
            .row_slice(1)
            .expect("source instance next plan row");
        let next_prep: &SourceInstancePrepColsV19<AB::Var> = (*next_prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("source instance row");
        let local: &SourceInstanceColsV19<AB::Var> = (*row).borrow();
        let next_row = main.row_slice(1).expect("source instance next row");
        let next: &SourceInstanceColsV19<AB::Var> = (*next_row).borrow();
        builder.assert_bool(prep.active);
        builder.assert_bool(prep.is_first);
        builder.assert_bool(prep.is_last);
        builder.assert_bool(prep.is_fixed_multi_air);
        builder.assert_bool(local.active);
        if self.profile.runtime_capacity_v4().is_some() {
            builder.when_first_row().assert_one(local.active);
            let mut occupancy_transition = builder.when_transition();
            occupancy_transition
                .when(AB::Expr::ONE - AB::Expr::from(prep.is_last))
                .assert_eq(next.active, local.active);
            occupancy_transition
                .when(AB::Expr::from(prep.is_last))
                .assert_zero(
                    (AB::Expr::ONE - AB::Expr::from(local.active)) * AB::Expr::from(next.active),
                );
            let inactive = AB::Expr::from(prep.active) - AB::Expr::from(local.active);
            for value in (*row).iter().skip(1) {
                builder.when(inactive.clone()).assert_zero((*value).into());
            }
        } else {
            builder.assert_eq(local.active, prep.active);
        }
        let enabled = AB::Expr::from(local.active);
        for lane in 0..DIGEST_SIZE {
            builder
                .when(enabled.clone())
                .assert_eq(local.pre_state[lane], local.absorbed[lane]);
            for flag in [
                prep.lane_is_carry[lane],
                prep.lane_is_constant[lane],
                prep.lane_is_public_value[lane],
                prep.lane_is_boundary_value[lane],
                prep.lane_emit_verified[lane],
            ] {
                builder.assert_bool(flag);
            }
            builder.when(enabled.clone()).assert_one(
                AB::Expr::from(prep.lane_is_carry[lane])
                    + AB::Expr::from(prep.lane_is_constant[lane])
                    + AB::Expr::from(prep.lane_is_public_value[lane])
                    + AB::Expr::from(prep.lane_is_boundary_value[lane]),
            );
            builder
                .when(enabled.clone() * AB::Expr::from(prep.lane_is_constant[lane]))
                .assert_eq(local.absorbed[lane], prep.lane_constant[lane]);
            builder
                .when(
                    enabled.clone()
                        * AB::Expr::from(prep.is_first)
                        * AB::Expr::from(prep.lane_is_carry[lane]),
                )
                .assert_zero(local.absorbed[lane]);
        }

        let first = enabled.clone() * AB::Expr::from(prep.is_first);
        for limb in &local.pre_state[DIGEST_SIZE..] {
            builder.when(first.clone()).assert_zero(*limb);
        }
        let continuing =
            AB::Expr::from(next.active) * (AB::Expr::ONE - AB::Expr::from(prep.is_last));
        let mut transition_builder = builder.when_transition();
        let mut transition = transition_builder.when(continuing);
        for limb in DIGEST_SIZE..POSEIDON2_WIDTH {
            transition.assert_eq(next.pre_state[limb], local.post_state[limb]);
        }
        for lane in 0..DIGEST_SIZE {
            transition.assert_zero(
                AB::Expr::from(next_prep.lane_is_carry[lane])
                    * (AB::Expr::from(next.pre_state[lane])
                        - AB::Expr::from(local.post_state[lane])),
            );
        }

        self.permute_bus.lookup_key(
            builder,
            Poseidon2PermuteMessage {
                input: local.pre_state.map(Into::into),
                output: local.post_state.map(Into::into),
            },
            enabled.clone(),
        );
        for lane in 0..DIGEST_SIZE {
            let is_pv = AB::Expr::from(prep.lane_is_public_value[lane]);
            let is_fixed = AB::Expr::from(prep.is_fixed_multi_air);
            let ordinary = enabled.clone() * is_pv.clone() * (AB::Expr::ONE - is_fixed.clone());
            let fixed = enabled.clone() * is_pv.clone() * is_fixed;
            // Authenticate this copy against the ordinary PublicValuesAir,
            // then republish it for the symbolic-expression consumer. The
            // source manifest additionally emits the same value on its
            // dedicated verified bus below.
            self.public_values_bus.receive(
                builder,
                prep.proof_index,
                PublicValuesBusMessage {
                    air_idx: prep.public_values_air_id,
                    pv_idx: prep.lane_index[lane],
                    value: local.absorbed[lane],
                },
                ordinary.clone(),
            );
            self.public_values_bus.send(
                builder,
                prep.proof_index,
                PublicValuesBusMessage {
                    air_idx: prep.public_values_air_id,
                    pv_idx: prep.lane_index[lane],
                    value: local.absorbed[lane],
                },
                ordinary,
            );
            self.fixed_public_values_bus.receive(
                builder,
                FixedSourcePublicValueMessageV19 {
                    proof_index: prep.proof_index.into(),
                    segment_index_lo: prep.segment_index_lo.into(),
                    segment_index_hi: prep.segment_index_hi.into(),
                    public_value_index: prep.lane_index[lane].into(),
                    value: local.absorbed[lane].into(),
                },
                fixed,
            );
            self.verified_public_values_bus.send(
                builder,
                VerifiedDirectAirPublicValueMessageV19 {
                    proof_index: prep.proof_index.into(),
                    segment_index_lo: prep.segment_index_lo.into(),
                    segment_index_hi: prep.segment_index_hi.into(),
                    air_id: prep.air_id.into(),
                    public_value_index: prep.lane_index[lane].into(),
                    value: local.absorbed[lane].into(),
                },
                enabled.clone() * AB::Expr::from(prep.lane_emit_verified[lane]),
            );
        }
        self.instance_bus.send(
            builder,
            SourceInstanceDigestMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                shard_ordinal: prep.shard_ordinal.into(),
                shard_id: prep.shard_id.into(),
                digest: core::array::from_fn(|index| local.post_state[index].into()),
            },
            enabled * AB::Expr::from(prep.is_last),
        );
    }
}

#[derive(Clone, Copy)]
struct InstanceLanePlanV19 {
    kind: u8,
    index: u32,
    constant: F,
}

#[derive(Clone)]
struct InstancePlanRowV19 {
    source: PlannedSourceV19,
    chunk_index: usize,
    chunk_count: usize,
    is_first: bool,
    is_last: bool,
    lanes: [InstanceLanePlanV19; DIGEST_SIZE],
}

fn instance_plan_rows(profile: &DirectLogUpSourceManifestProfileV19) -> Vec<InstancePlanRowV19> {
    let mut rows = Vec::new();
    for source in profile.planned.iter() {
        let preimage_len =
            3 + source.entry.public_values_len as usize + source.entry.boundary_values_len as usize;
        let chunk_count = preimage_len.div_ceil(DIGEST_SIZE);
        for chunk_index in 0..chunk_count {
            let lanes = core::array::from_fn(|lane| {
                let position = chunk_index * DIGEST_SIZE + lane;
                match position {
                    0 => InstanceLanePlanV19 {
                        kind: INSTANCE_KIND_CONSTANT,
                        index: 0,
                        constant: F::from_u64(SOURCE_INSTANCE_TAG_V19),
                    },
                    1 => InstanceLanePlanV19 {
                        kind: INSTANCE_KIND_CONSTANT,
                        index: 0,
                        constant: F::from_u32(source.entry.public_values_len),
                    },
                    2 => InstanceLanePlanV19 {
                        kind: INSTANCE_KIND_CONSTANT,
                        index: 0,
                        constant: F::from_u32(source.entry.boundary_values_len),
                    },
                    position if position < 3 + source.entry.public_values_len as usize => {
                        InstanceLanePlanV19 {
                            kind: INSTANCE_KIND_PUBLIC_VALUE,
                            index: (position - 3) as u32,
                            constant: F::ZERO,
                        }
                    }
                    position if position < preimage_len => InstanceLanePlanV19 {
                        kind: INSTANCE_KIND_BOUNDARY_VALUE,
                        index: (position - 3 - source.entry.public_values_len as usize) as u32,
                        constant: F::ZERO,
                    },
                    _ => InstanceLanePlanV19 {
                        kind: INSTANCE_KIND_CARRY,
                        index: 0,
                        constant: F::ZERO,
                    },
                }
            });
            rows.push(InstancePlanRowV19 {
                source: source.clone(),
                chunk_index,
                chunk_count,
                is_first: chunk_index == 0,
                is_last: chunk_index + 1 == chunk_count,
                lanes,
            });
        }
    }
    rows
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
struct SourceForestNodePrepColsV19<T> {
    active: T,
    is_padding: T,
    proof_index: T,
    level: T,
    index: T,
    child_level: T,
    left_index: T,
    right_index: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct SourceForestNodeColsV19<T> {
    pub active: T,
    pub left: [T; DIGEST_SIZE],
    pub right: [T; DIGEST_SIZE],
    pub output: [T; DIGEST_SIZE],
    pub padding_post_state: [T; POSEIDON2_WIDTH],
}

#[derive(Clone, ColumnsAir)]
#[columns_via(SourceForestNodeColsV19<u8>)]
pub struct DirectLogUpSourceForestNodeAirV19 {
    pub profile: DirectLogUpSourceManifestProfileV19,
    pub permute_bus: Poseidon2PermuteBus,
    pub compress_bus: Poseidon2CompressBus,
    pub node_bus: SourceForestNodeBusV19,
}

impl PartitionedBaseAir<F> for DirectLogUpSourceForestNodeAirV19 {
    fn common_main_width(&self) -> usize {
        SourceForestNodeColsV19::<F>::width()
    }
}
impl BaseAir<F> for DirectLogUpSourceForestNodeAirV19 {
    fn width(&self) -> usize {
        SourceForestNodeColsV19::<F>::width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let rows = forest_node_plan_rows(&self.profile);
        let width = SourceForestNodePrepColsV19::<F>::width();
        let height = rows.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(height * width);
        for (row, plan) in rows.iter().enumerate() {
            let cols: &mut SourceForestNodePrepColsV19<F> =
                values[row * width..(row + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.is_padding = F::from_bool(plan.is_padding);
            cols.proof_index = F::from_u32(plan.proof_index);
            cols.level = F::from_u32(plan.level);
            cols.index = F::from_u32(plan.index);
            cols.child_level = F::from_u32(plan.level.saturating_sub(1));
            cols.left_index = F::from_u32(2 * plan.index);
            cols.right_index = F::from_u32(2 * plan.index + 1);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}
impl BaseAirWithPublicValues<F> for DirectLogUpSourceForestNodeAirV19 {}

impl<AB: AirBuilder<F = F> + InteractionBuilder + PairBuilder> Air<AB>
    for DirectLogUpSourceForestNodeAirV19
{
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed();
        let prep_row = preprocessed
            .row_slice(0)
            .expect("source forest node plan row");
        let prep: &SourceForestNodePrepColsV19<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("source forest node row");
        let local: &SourceForestNodeColsV19<AB::Var> = (*row).borrow();
        for flag in [prep.active, prep.is_padding, local.active] {
            builder.assert_bool(flag);
        }
        builder.assert_eq(local.active, prep.active);
        let enabled = AB::Expr::from(local.active);
        let padding = enabled.clone() * AB::Expr::from(prep.is_padding);
        let inner = enabled.clone() * (AB::Expr::ONE - AB::Expr::from(prep.is_padding));

        let padding_input: [AB::Expr; POSEIDON2_WIDTH] =
            core::array::from_fn(|index| match index {
                0 => AB::Expr::from_u64(SOURCE_FOREST_TAG_V19),
                1 => AB::Expr::ZERO,
                _ => AB::Expr::ZERO,
            });
        self.permute_bus.lookup_key(
            builder,
            Poseidon2PermuteMessage {
                input: padding_input,
                output: local.padding_post_state.map(Into::into),
            },
            padding.clone(),
        );
        for index in 0..DIGEST_SIZE {
            builder
                .when(padding.clone())
                .assert_eq(local.output[index], local.padding_post_state[index]);
        }

        self.node_bus.receive(
            builder,
            SourceForestNodeMessageV19 {
                proof_index: prep.proof_index.into(),
                level: prep.child_level.into(),
                index: prep.left_index.into(),
                digest: local.left.map(Into::into),
            },
            inner.clone(),
        );
        self.node_bus.receive(
            builder,
            SourceForestNodeMessageV19 {
                proof_index: prep.proof_index.into(),
                level: prep.child_level.into(),
                index: prep.right_index.into(),
                digest: local.right.map(Into::into),
            },
            inner.clone(),
        );
        self.compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: join(local.left, local.right),
                output: local.output,
            },
            inner,
        );
        self.node_bus.send(
            builder,
            SourceForestNodeMessageV19 {
                proof_index: prep.proof_index.into(),
                level: prep.level.into(),
                index: prep.index.into(),
                digest: local.output.map(Into::into),
            },
            enabled,
        );
    }
}

#[derive(Clone, Copy)]
struct ForestNodePlanRowV19 {
    proof_index: u32,
    is_padding: bool,
    level: u32,
    index: u32,
}

fn forest_node_plan_rows(
    profile: &DirectLogUpSourceManifestProfileV19,
) -> Vec<ForestNodePlanRowV19> {
    let mut rows = Vec::new();
    let mut cursor = 0;
    while cursor < profile.planned.len() {
        let first = &profile.planned[cursor];
        let count = first.shard_count as usize;
        let padded = count.next_power_of_two();
        for index in count..padded {
            rows.push(ForestNodePlanRowV19 {
                proof_index: first.proof_index,
                is_padding: true,
                level: 0,
                index: index as u32,
            });
        }
        let depth = padded.ilog2();
        for level in 1..=depth {
            let width = padded >> level;
            for index in 0..width {
                rows.push(ForestNodePlanRowV19 {
                    proof_index: first.proof_index,
                    is_padding: false,
                    level,
                    index: index as u32,
                });
            }
        }
        cursor += count;
    }
    rows
}

#[derive(Clone, Debug)]
pub struct DirectLogUpSourceManifestTracesV19 {
    pub instance: RowMajorMatrix<F>,
    pub manifest: RowMajorMatrix<F>,
    pub forest_nodes: RowMajorMatrix<F>,
    pub forest_roots: Vec<[F; DIGEST_SIZE]>,
    pub poseidon_permute_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub poseidon_compress_inputs: Vec<[F; POSEIDON2_WIDTH]>,
}

pub fn generate_direct_logup_source_manifest_traces_v19(
    profile: &DirectLogUpSourceManifestProfileV19,
    segments: &[DirectLogUpSegmentSourceRecordV19],
) -> Result<DirectLogUpSourceManifestTracesV19, &'static str> {
    let expected_segments = profile
        .planned
        .iter()
        .filter(|source| source.shard_ordinal == 0)
        .count();
    let runtime_capacity = profile.runtime_capacity_v4();
    if if let Some(capacity) = runtime_capacity {
        segments.is_empty() || segments.len() > capacity || expected_segments != capacity
    } else {
        segments.len() != expected_segments
    } {
        return Err("source segment count differs from fixed plan");
    }
    let mut flat = Vec::with_capacity(profile.planned.len());
    for (segment_ordinal, segment) in segments.iter().enumerate() {
        let proof_index =
            u32::try_from(segment_ordinal).map_err(|_| "source proof index overflow")?;
        if segment.proof_index != proof_index {
            return Err("invalid source segment proof index");
        }
        let expected = profile
            .planned
            .iter()
            .filter(|source| source.proof_index == proof_index)
            .collect::<Vec<_>>();
        if expected.len() != segment.sources.len()
            || expected
                .first()
                .is_none_or(|source| source.segment_index != segment.segment_index)
        {
            return Err("source count differs from fixed segment plan");
        }
        for (plan, source) in expected.into_iter().zip(&segment.sources) {
            if source.shard_id != plan.entry.shard_id
                || source.public_values.len() != plan.entry.public_values_len as usize
                || source.boundary_values.len() != plan.entry.boundary_values_len as usize
            {
                return Err("fresh source differs from fixed relation profile");
            }
            flat.push((plan, segment, source));
        }
    }
    if runtime_capacity.is_none() && flat.len() != profile.planned.len() {
        return Err("incomplete source plan");
    }

    let mut permute_inputs = Vec::new();
    let mut compress_inputs = Vec::new();
    let instance_plan = instance_plan_rows(profile);
    let instance_width = SourceInstanceColsV19::<F>::width();
    let instance_height = instance_plan.len().next_power_of_two().max(2);
    let mut instance_values = F::zero_vec(instance_height * instance_width);
    let mut instance_digests = Vec::with_capacity(flat.len());
    let mut instance_row = 0;
    for ((_, _, source), plan_source) in flat.iter().zip(profile.planned.iter()) {
        // Fixed verifier sources use the same canonical public-value digest as
        // ordinary sources. Their values arrive over the dedicated one-way
        // beta adapter bus rather than the child PublicValuesBus.
        let mut preimage = vec![
            F::from_u64(SOURCE_INSTANCE_TAG_V19),
            F::from_usize(source.public_values.len()),
            F::from_usize(source.boundary_values.len()),
        ];
        preimage.extend_from_slice(&source.public_values);
        preimage.extend_from_slice(&source.boundary_values);
        let (digest, pre_states, post_states) = poseidon2_hash_slice_with_states(&preimage);
        if digest != source.expected_instance_digest {
            return Err("retained source instance digest is not its canonical preimage hash");
        }
        permute_inputs.extend_from_slice(&pre_states);
        let plans = instance_plan
            .iter()
            .filter(|plan| {
                plan.source.proof_index == plan_source.proof_index
                    && plan.source.shard_ordinal == plan_source.shard_ordinal
            })
            .collect::<Vec<_>>();
        if plans.len() != pre_states.len() || pre_states.len() != post_states.len() {
            return Err("source instance sponge shape");
        }
        for (chunk, ((plan, pre), post)) in plans
            .into_iter()
            .zip(pre_states.into_iter())
            .zip(post_states.into_iter())
            .enumerate()
        {
            let cols: &mut SourceInstanceColsV19<F> = instance_values
                [instance_row * instance_width..(instance_row + 1) * instance_width]
                .borrow_mut();
            cols.active = F::ONE;
            cols.pre_state = pre;
            cols.post_state = post;
            for lane in 0..DIGEST_SIZE {
                cols.absorbed[lane] = match plan.lanes[lane].kind {
                    INSTANCE_KIND_CONSTANT => plan.lanes[lane].constant,
                    INSTANCE_KIND_PUBLIC_VALUE => {
                        source.public_values[plan.lanes[lane].index as usize]
                    }
                    INSTANCE_KIND_BOUNDARY_VALUE => {
                        source.boundary_values[plan.lanes[lane].index as usize]
                    }
                    INSTANCE_KIND_CARRY => pre[lane],
                    _ => return Err("invalid source instance lane kind"),
                };
            }
            if chunk + 1 == plan.chunk_count && post[..DIGEST_SIZE] != digest {
                return Err("source instance final sponge state");
            }
            instance_row += 1;
        }
        instance_digests.push(digest);
    }

    let padding_words = [F::from_u64(SOURCE_FOREST_TAG_V19), F::ZERO];
    let (padding_digest, padding_pre, padding_post) =
        poseidon2_hash_slice_with_states(&padding_words);
    let padding_input = padding_pre[0];
    let padding_output = padding_post[0];

    let mut leaves = Vec::with_capacity(flat.len());
    let mut metadata_states = Vec::with_capacity(flat.len());
    for (((plan, _, source), &instance_digest), _index) in
        flat.iter().zip(&instance_digests).zip(0..flat.len())
    {
        let words = [
            F::from_u64(SOURCE_LEAF_TAG_V19),
            F::from_u32(plan.entry.shard_id),
            F::from_u32(plan.entry.air_id),
            F::from_u8(plan.entry.log_height),
            F::from_u8(plan.entry.log_message_len),
            F::from_u8(plan.entry.log_codeword_len),
        ];
        let (metadata, pre, post) = poseidon2_hash_slice_with_states(&words);
        permute_inputs.extend_from_slice(&pre);
        let instance_root = poseidon2_compress_with_capacity(instance_digest, source.source_root).0;
        let relation_instance =
            poseidon2_compress_with_capacity(plan.entry.relation_digest, instance_root).0;
        let leaf = poseidon2_compress_with_capacity(metadata, relation_instance).0;
        compress_inputs.push(join(instance_digest, source.source_root));
        compress_inputs.push(join(plan.entry.relation_digest, instance_root));
        compress_inputs.push(join(metadata, relation_instance));
        leaves.push((instance_root, relation_instance, leaf));
        metadata_states.push(post[0]);
    }

    let node_plan = forest_node_plan_rows(profile);
    let node_width = SourceForestNodeColsV19::<F>::width();
    let node_height = node_plan.len().next_power_of_two().max(2);
    let mut node_values = F::zero_vec(node_height * node_width);
    let mut forest_roots = Vec::with_capacity(segments.len());
    let mut tree_roots = Vec::with_capacity(segments.len());
    let mut nodes = std::collections::BTreeMap::new();
    for ((plan, _, _), (_, _, leaf)) in flat.iter().zip(&leaves) {
        nodes.insert((plan.proof_index, 0u32, plan.shard_ordinal as u32), *leaf);
    }
    for plan in &node_plan {
        let cols: &mut SourceForestNodeColsV19<F> =
            node_values[forest_node_row_index(&node_plan, plan) * node_width
                ..(forest_node_row_index(&node_plan, plan) + 1) * node_width]
                .borrow_mut();
        cols.active = F::ONE;
        let output = if plan.is_padding {
            permute_inputs.push(padding_input);
            cols.padding_post_state = padding_output;
            padding_digest
        } else {
            let left = *nodes
                .get(&(plan.proof_index, plan.level - 1, 2 * plan.index))
                .ok_or("missing source forest left child")?;
            let right = *nodes
                .get(&(plan.proof_index, plan.level - 1, 2 * plan.index + 1))
                .ok_or("missing source forest right child")?;
            cols.left = left;
            cols.right = right;
            compress_inputs.push(join(left, right));
            poseidon2_compress_with_capacity(left, right).0
        };
        cols.output = output;
        nodes.insert((plan.proof_index, plan.level, plan.index), output);
    }

    for segment in segments {
        let count = segment.sources.len();
        let depth = count.next_power_of_two().ilog2();
        let tree = *nodes
            .get(&(segment.proof_index, depth, 0))
            .ok_or("missing source forest root")?;
        let count_words = [F::from_u64(SOURCE_FOREST_TAG_V19), F::from_usize(count)];
        let (count_digest, count_pre, count_post) = poseidon2_hash_slice_with_states(&count_words);
        permute_inputs.extend_from_slice(&count_pre);
        compress_inputs.push(join(count_digest, tree));
        let root = poseidon2_compress_with_capacity(count_digest, tree).0;
        forest_roots.push(root);
        tree_roots.push((tree, count_post[0]));
    }

    let manifest_width = SourceManifestColsV19::<F>::width();
    let manifest_height = runtime_capacity
        .unwrap_or(flat.len())
        .next_power_of_two()
        .max(2);
    let mut manifest_values = F::zero_vec(manifest_height * manifest_width);
    for (row, (((plan, segment, source), &instance_digest), metadata_state)) in flat
        .iter()
        .zip(&instance_digests)
        .zip(&metadata_states)
        .enumerate()
    {
        let (instance_root, relation_instance, leaf) = leaves[row];
        let local_proof_index = segment.proof_index as usize;
        let (tree_root, count_post_state) = tree_roots[local_proof_index];
        let cols: &mut SourceManifestColsV19<F> =
            manifest_values[row * manifest_width..(row + 1) * manifest_width].borrow_mut();
        cols.active = F::ONE;
        cols.source_root = source.source_root;
        cols.instance_digest = instance_digest;
        cols.source_forest_root = forest_roots[local_proof_index];
        cols.segment_openings_digest = segment.segment_openings_digest;
        cols.prefix_start_tidx = F::from_u32(segment.prefix_start_tidx);
        cols.metadata_post_state = *metadata_state;
        cols.instance_root_digest = instance_root;
        cols.relation_instance_root_digest = relation_instance;
        cols.leaf_digest = leaf;
        cols.count_post_state = count_post_state;
        cols.tree_root = tree_root;
        cols.alignment_sample = segment.alignment_sample;
        let _ = plan;
    }

    Ok(DirectLogUpSourceManifestTracesV19 {
        instance: RowMajorMatrix::new(instance_values, instance_width),
        manifest: RowMajorMatrix::new(manifest_values, manifest_width),
        forest_nodes: RowMajorMatrix::new(node_values, node_width),
        forest_roots,
        poseidon_permute_inputs: permute_inputs,
        poseidon_compress_inputs: compress_inputs,
    })
}

fn forest_node_row_index(plans: &[ForestNodePlanRowV19], target: &ForestNodePlanRowV19) -> usize {
    // Plan rows are unique by `(proof, padding, level, index)`.
    plans
        .iter()
        .position(|plan| {
            plan.proof_index == target.proof_index
                && plan.is_padding == target.is_padding
                && plan.level == target.level
                && plan.index == target.index
        })
        .expect("forest node plan row")
}

fn join<T: Copy>(left: [T; DIGEST_SIZE], right: [T; DIGEST_SIZE]) -> [T; POSEIDON2_WIDTH] {
    core::array::from_fn(|index| {
        if index < DIGEST_SIZE {
            left[index]
        } else {
            right[index - DIGEST_SIZE]
        }
    })
}

#[cfg(test)]
mod fixed_capacity_v4_tests {
    use std::panic::AssertUnwindSafe;

    use openvm_stark_backend::air_builders::debug::check_constraints;
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

    use super::*;

    fn profile() -> DirectLogUpSourceManifestProfileV19 {
        let entry = DirectLogUpSourceEntryProfileV19::fixed_multi_air(
            [F::from_u32(810); DIGEST_SIZE],
            5,
            7,
            1,
        )
        .unwrap();
        DirectLogUpSourceManifestProfileV19::new_fixed_capacity_v4(
            [F::from_u32(811); DIGEST_SIZE],
            [F::from_u32(812); DIGEST_SIZE],
            entry,
        )
        .unwrap()
    }

    fn records(occupancy: usize) -> Vec<DirectLogUpSegmentSourceRecordV19> {
        (0..occupancy)
            .map(|slot| {
                let public_values = vec![F::from_usize(slot + 1)];
                let preimage = [
                    F::from_u64(SOURCE_INSTANCE_TAG_V19),
                    F::ONE,
                    F::ZERO,
                    public_values[0],
                ];
                let (expected_instance_digest, _, _) = poseidon2_hash_slice_with_states(&preimage);
                DirectLogUpSegmentSourceRecordV19 {
                    proof_index: slot as u32,
                    segment_index: slot as u32,
                    prefix_start_tidx: 100 + slot as u32 * 20,
                    alignment_sample: [F::from_usize(slot + 2); 4],
                    segment_openings_digest: [F::from_usize(820 + slot); DIGEST_SIZE],
                    sources: vec![DirectLogUpFreshSourceRecordV19 {
                        shard_id: 0,
                        source_root: [F::from_usize(830 + slot); DIGEST_SIZE],
                        expected_instance_digest,
                        public_values,
                        boundary_values: Vec::new(),
                    }],
                }
            })
            .collect()
    }

    fn airs() -> (
        DirectLogUpSourceInstanceAirV19,
        DirectLogUpSourceManifestAirV19,
        DirectLogUpSourceForestNodeAirV19,
    ) {
        let profile = profile();
        let permute = Poseidon2PermuteBus::new(800);
        let compress = Poseidon2CompressBus::new(801);
        let instance = SourceInstanceDigestBusV19::new(802);
        let nodes = SourceForestNodeBusV19::new(803);
        (
            DirectLogUpSourceInstanceAirV19 {
                profile: profile.clone(),
                permute_bus: permute,
                public_values_bus: PublicValuesBus::new(804),
                verified_public_values_bus: VerifiedDirectAirPublicValueBusV19::new(805),
                fixed_public_values_bus: FixedSourcePublicValueBusV19::new(806),
                fixed_public_values_air_id: Some(0),
                instance_bus: instance,
            },
            DirectLogUpSourceManifestAirV19 {
                profile: profile.clone(),
                transcript_bus: TranscriptBus::new(807),
                permute_bus: permute,
                compress_bus: compress,
                instance_bus: instance,
                node_bus: nodes,
                leaf_bus: VerifiedSourceForestLeafBusV19::new(808),
                prefix_end_bus: SourcePrefixEndBusV19::new(809),
            },
            DirectLogUpSourceForestNodeAirV19 {
                profile,
                permute_bus: permute,
                compress_bus: compress,
                node_bus: nodes,
            },
        )
    }

    fn check_instance(air: &DirectLogUpSourceInstanceAirV19, trace: &RowMajorMatrix<F>) {
        let prep = air.preprocessed_trace().unwrap();
        check_constraints::<_, BabyBearPoseidon2Config>(
            air,
            "DirectLogUpSourceInstanceAirV19/v4",
            &Some(prep.as_view()),
            &[trace.as_view()],
            &[],
        );
    }

    fn check_manifest(air: &DirectLogUpSourceManifestAirV19, trace: &RowMajorMatrix<F>) {
        let prep = air.preprocessed_trace().unwrap();
        check_constraints::<_, BabyBearPoseidon2Config>(
            air,
            "DirectLogUpSourceManifestAirV19/v4",
            &Some(prep.as_view()),
            &[trace.as_view()],
            &[],
        );
    }

    fn check_nodes(air: &DirectLogUpSourceForestNodeAirV19, trace: &RowMajorMatrix<F>) {
        let prep = air.preprocessed_trace().unwrap();
        check_constraints::<_, BabyBearPoseidon2Config>(
            air,
            "DirectLogUpSourceForestNodeAirV19/v4",
            &Some(prep.as_view()),
            &[trace.as_view()],
            &[],
        );
    }

    #[test]
    fn manifest_and_instance_use_fixed_capacity_with_runtime_zero_suffix() {
        let (instance_air, manifest_air, node_air) = airs();
        let instance_prep = instance_air.preprocessed_trace().unwrap();
        let manifest_prep = manifest_air.preprocessed_trace().unwrap();
        let node_prep = node_air.preprocessed_trace().unwrap();
        for occupancy in 1..=4 {
            let traces = generate_direct_logup_source_manifest_traces_v19(
                &manifest_air.profile,
                &records(occupancy),
            )
            .unwrap();
            assert_eq!(traces.manifest.height(), 4);
            assert_eq!(traces.instance.height(), instance_prep.height());
            assert_eq!(traces.forest_nodes.height(), node_prep.height());
            assert_eq!(
                manifest_air.preprocessed_trace().unwrap().values,
                manifest_prep.values
            );
            assert_eq!(
                instance_air.preprocessed_trace().unwrap().values,
                instance_prep.values
            );
            check_instance(&instance_air, &traces.instance);
            check_manifest(&manifest_air, &traces.manifest);
            check_nodes(&node_air, &traces.forest_nodes);
        }
    }

    #[test]
    fn manifest_inactive_suffix_and_nonprefix_mutations_reject() {
        let (_, manifest_air, _) = airs();
        let traces =
            generate_direct_logup_source_manifest_traces_v19(&manifest_air.profile, &records(2))
                .unwrap();
        let width = SourceManifestColsV19::<F>::width();
        let rejects = |mutate: &dyn Fn(&mut RowMajorMatrix<F>)| {
            let mut changed = traces.manifest.clone();
            mutate(&mut changed);
            assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
                check_manifest(&manifest_air, &changed)
            }))
            .is_err());
        };
        rejects(&|trace| {
            let row: &mut SourceManifestColsV19<F> =
                trace.values[3 * width..4 * width].borrow_mut();
            row.source_root[0] = F::ONE;
        });
        rejects(&|trace| {
            let row: &mut SourceManifestColsV19<F> =
                trace.values[3 * width..4 * width].borrow_mut();
            row.active = F::ONE;
        });
    }
}

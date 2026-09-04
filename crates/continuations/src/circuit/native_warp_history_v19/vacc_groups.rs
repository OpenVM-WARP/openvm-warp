//! Shape-grouped protocol-v19 WARP replay composition.
//!
//! The grouping plan is verifier-owned. It fixes the canonical update order,
//! assigns every update to one exact `(numeric VACC shape, has_prior)` group,
//! and gives each group dense private proof identifiers. Group-local proof
//! identifiers never cross a group bus boundary. The only shared statement is
//! [`CertifiedWarpReplayMessageV19`], whose key deliberately has no proof
//! identifier: its segment, update, shard, relation, roots and digests already
//! identify the authenticated History transition.

use core::borrow::{Borrow, BorrowMut};
use std::{ops::Range, sync::Arc};

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit::{
    native_warp::{
        NativeStandardVaccDigestBus, NativeStandardVaccEndBus, NativeStandardVaccProfile,
        NativeStandardVaccProtocolBus, NativeStandardVaccRootBus, NativeStandardVaccShapeProfile,
        NativeWarpPcdBusInventory, NativeWarpTranscriptModule,
    },
    system::{BusIndexManager, BusInventory},
    transcript::Poseidon2MultibusInputs,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::{BasedVectorSpace, PrimeCharacteristicRing},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    warp_accum::NativeTranscriptPhase,
    warp_pesat::AccumulatorInstance,
    AirRef, BaseAirWithPublicValues, PartitionedBaseAir, StarkProtocolConfig, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as NativeSC, Digest, DIGEST_SIZE, D_EF, EF, F,
};

use super::{
    generate_warp_replay_producer_trace_v19, CertifiedDirectAirVaccInputBusV19,
    CertifiedFreshExplicitDigestBusV19, CertifiedSwirlRawOpeningBusV19,
    CertifiedSwirlRawOpeningMessageV19, CertifiedWarpReplayBusV19, CertifiedWarpReplayMessageV19,
    CudaDirectAirVaccVerifierErrorV19, CudaDirectAirVaccVerifierModuleV19,
    CudaDirectAirVaccVerifierRecordV19, CudaSharedForestBindingBusV19,
    CudaSharedForestBindingMessageV19, CudaSharedForestDescriptorBusV19,
    CudaSharedForestUniqueQueryV19, CudaSharedForestVerifierProfileV19,
    CudaSharedForestVerifierRecordV19, DirectAirVaccContextBusV19, DirectAirVaccHistoryBusesV19,
    DirectAirVaccProducerRecordV19, DirectAirVaccSharedPoseidonBatchTraceV19,
    DirectAirVaccVerifierErrorV19, DirectAirVaccVerifierModuleV19, DirectAirVaccVerifierRecordV19,
    HistoryPoseidon2CompressBusV19, PositiveProducerErrorV19, WarpReplayProducerAirV19,
    WarpReplayProducerScheduleV19, LOGUP_ONLY_MODE_TAG_V19, MAX_RAW_MESSAGE_POINT_LEN_V19,
    NATIVE_WARP_HISTORY_PROTOCOL_V19,
};

/// Canonical segment-local opening emitted by the SWIRL boundary.
///
/// Deliberately absent is any VACC-group-local identifier. `proof_index` is
/// canonical only inside the surrounding whole-History or chunk-local SWIRL
/// proof, while `segment_index` remains the absolute execution/transcript
/// binding. The verifier-key-owned remap AIR below derives the dense VACC-local
/// identifier from the planned update position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CanonicalSwirlRawOpeningRecordV19 {
    pub proof_index: u32,
    pub segment_index: u32,
    /// Canonical ordinal in this segment's source manifest/SWIRL stream.
    pub segment_source_ordinal: u16,
    /// Canonical registry shard selected by the setup plan for that source.
    pub registry_shard_ordinal: u16,
    pub relation_digest: Digest,
    pub source_forest_root: Digest,
    pub segment_openings_digest: Digest,
    /// Logical source descriptor committed by the SWIRL boundary.
    pub descriptor_root: Digest,
    /// Physical scalar-tree/shared-forest root consumed by standard VACC.
    pub physical_root: Digest,
    pub column_start: u32,
    pub column_width: u32,
    pub forest_width: u32,
    pub rows_per_query: u32,
    pub point: Vec<EF>,
    pub value: EF,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CertifiedSwirlOpeningIdRemapEntryV19 {
    /// Canonical position in the complete History update stream. This is
    /// retained for setup auditing; it is never supplied by the prover.
    pub global_update_position: usize,
    /// Proof identifier in the surrounding whole-History or chunk-local SWIRL
    /// namespace. It is deliberately independent of the absolute segment.
    pub canonical_proof_index: u32,
    pub segment_index: u32,
    pub segment_source_ordinal: u16,
    pub registry_shard_ordinal: u16,
    /// Dense proof identifier used only inside this physical VACC group.
    pub local_proof_index: u32,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct CertifiedSwirlOpeningIdRemapPrepColsV19<T> {
    pub active: T,
    pub canonical_proof_index: T,
    pub local_proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub segment_source_ordinal: T,
    pub registry_shard_ordinal: T,
}

/// Opening payload shared byte-for-byte (field-element-for-field-element)
/// between the canonical and group-local lookup messages. Proof IDs live only
/// in the setup-owned preprocessed columns above.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct CertifiedSwirlOpeningIdRemapColsV19<T> {
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub descriptor_root: [T; DIGEST_SIZE],
    pub physical_root: [T; DIGEST_SIZE],
    pub column_start: T,
    pub column_width: T,
    pub forest_width: T,
    pub rows_per_query: T,
    pub point_len: T,
    pub point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub value: [T; D_EF],
}

/// Sound setup-owned bridge from the segment-canonical SWIRL opening bus to a
/// dense group-local VACC opening bus.
///
/// The preprocessed table is committed in the verifying key. Every active row
/// consumes exactly one canonical `(segment_index, shard_ordinal, payload)`
/// and emits the identical payload with only `proof_index` changed to the
/// fixed dense local index. A host-side clone cannot satisfy either multiset.
#[derive(Clone, Debug, ColumnsAir)]
#[columns_via(CertifiedSwirlOpeningIdRemapColsV19<u8>)]
pub struct CertifiedSwirlOpeningIdRemapAirV19 {
    entries: Vec<CertifiedSwirlOpeningIdRemapEntryV19>,
    pub canonical_opening_bus: CertifiedSwirlRawOpeningBusV19,
    pub local_opening_bus: CertifiedSwirlRawOpeningBusV19,
    pub cuda_binding_bus: Option<CudaSharedForestBindingBusV19>,
}

impl CertifiedSwirlOpeningIdRemapAirV19 {
    fn new(
        entries: Vec<CertifiedSwirlOpeningIdRemapEntryV19>,
        canonical_opening_bus: CertifiedSwirlRawOpeningBusV19,
        local_opening_bus: CertifiedSwirlRawOpeningBusV19,
        cuda_binding_bus: Option<CudaSharedForestBindingBusV19>,
    ) -> Result<Self, VaccHistoryGroupsErrorV19> {
        if entries.is_empty()
            || entries.iter().enumerate().any(|(local_index, entry)| {
                entry.local_proof_index != u32::try_from(local_index).unwrap_or(u32::MAX)
            })
        {
            return Err(VaccHistoryGroupsErrorV19::OpeningIdRemapPlan);
        }
        Ok(Self {
            entries,
            canonical_opening_bus,
            local_opening_bus,
            cuda_binding_bus,
        })
    }

    #[must_use]
    pub fn entries(&self) -> &[CertifiedSwirlOpeningIdRemapEntryV19] {
        &self.entries
    }

    pub fn generate_trace(
        &self,
        canonical_records: &[CanonicalSwirlRawOpeningRecordV19],
    ) -> Result<RowMajorMatrix<F>, VaccHistoryGroupsErrorV19> {
        if canonical_records.len() != self.entries.len() {
            return Err(VaccHistoryGroupsErrorV19::CanonicalOpeningRecordCount);
        }
        let width = CertifiedSwirlOpeningIdRemapColsV19::<F>::width();
        let height = self.entries.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(width * height);
        for (row, (entry, record)) in self
            .entries
            .iter()
            .zip(canonical_records.iter())
            .enumerate()
        {
            if record.proof_index != entry.canonical_proof_index
                || record.segment_index != entry.segment_index
                || record.segment_source_ordinal != entry.segment_source_ordinal
                || record.registry_shard_ordinal != entry.registry_shard_ordinal
                || record.point.is_empty()
                || record.point.len() > MAX_RAW_MESSAGE_POINT_LEN_V19
            {
                return Err(VaccHistoryGroupsErrorV19::NonCanonicalOpeningRecord);
            }
            let cols: &mut CertifiedSwirlOpeningIdRemapColsV19<F> =
                values[row * width..(row + 1) * width].borrow_mut();
            cols.source_forest_root = record.source_forest_root;
            cols.segment_openings_digest = record.segment_openings_digest;
            cols.descriptor_root = record.descriptor_root;
            cols.physical_root = record.physical_root;
            cols.column_start = F::from_u32(record.column_start);
            cols.column_width = F::from_u32(record.column_width);
            cols.forest_width = F::from_u32(record.forest_width);
            cols.rows_per_query = F::from_u32(record.rows_per_query);
            cols.point_len = F::from_usize(record.point.len());
            for (target, value) in cols.point.iter_mut().zip(record.point.iter()) {
                target.copy_from_slice(value.as_basis_coefficients_slice());
            }
            cols.value
                .copy_from_slice(record.value.as_basis_coefficients_slice());
        }
        Ok(RowMajorMatrix::new(values, width))
    }
}

impl BaseAir<F> for CertifiedSwirlOpeningIdRemapAirV19 {
    fn width(&self) -> usize {
        CertifiedSwirlOpeningIdRemapColsV19::<F>::width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = CertifiedSwirlOpeningIdRemapPrepColsV19::<F>::width();
        let height = self.entries.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(width * height);
        for (row, entry) in self.entries.iter().enumerate() {
            let prep: &mut CertifiedSwirlOpeningIdRemapPrepColsV19<F> =
                values[row * width..(row + 1) * width].borrow_mut();
            prep.active = F::ONE;
            prep.canonical_proof_index = F::from_u32(entry.canonical_proof_index);
            prep.local_proof_index = F::from_u32(entry.local_proof_index);
            prep.segment_index_lo = F::from_u32(entry.segment_index & 0xffff);
            prep.segment_index_hi = F::from_u32(entry.segment_index >> 16);
            prep.segment_source_ordinal = F::from_u16(entry.segment_source_ordinal);
            prep.registry_shard_ordinal = F::from_u16(entry.registry_shard_ordinal);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for CertifiedSwirlOpeningIdRemapAirV19 {}
impl PartitionedBaseAir<F> for CertifiedSwirlOpeningIdRemapAirV19 {}

impl<AB> Air<AB> for CertifiedSwirlOpeningIdRemapAirV19
where
    AB: AirBuilder<F = F> + InteractionBuilder + PairBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed();
        let prep_row = preprocessed
            .row_slice(0)
            .expect("SWIRL opening remap preprocessed row");
        let prep: &CertifiedSwirlOpeningIdRemapPrepColsV19<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("SWIRL opening remap row");
        let local: &CertifiedSwirlOpeningIdRemapColsV19<AB::Var> = (*row).borrow();
        builder.assert_bool(prep.active);
        let enabled = AB::Expr::from(prep.active);
        let canonical_message = CertifiedSwirlRawOpeningMessageV19 {
            proof_index: prep.canonical_proof_index.into(),
            protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            mode_tag: AB::Expr::from_u32(LOGUP_ONLY_MODE_TAG_V19),
            segment_index_lo: prep.segment_index_lo.into(),
            segment_index_hi: prep.segment_index_hi.into(),
            shard_ordinal: prep.segment_source_ordinal.into(),
            source_forest_root: local.source_forest_root.map(Into::into),
            segment_openings_digest: local.segment_openings_digest.map(Into::into),
            root: local.descriptor_root.map(Into::into),
            point_len: local.point_len.into(),
            point: local.point.map(|point| point.map(Into::into)),
            value: local.value.map(Into::into),
        };
        self.canonical_opening_bus
            .lookup_key(builder, canonical_message, enabled.clone());
        self.local_opening_bus.add_key_with_lookups(
            builder,
            CertifiedSwirlRawOpeningMessageV19 {
                proof_index: prep.local_proof_index.into(),
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                mode_tag: AB::Expr::from_u32(LOGUP_ONLY_MODE_TAG_V19),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                shard_ordinal: prep.registry_shard_ordinal.into(),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                root: local.physical_root.map(Into::into),
                point_len: local.point_len.into(),
                point: local.point.map(|point| point.map(Into::into)),
                value: local.value.map(Into::into),
            },
            enabled,
        );
        if let Some(binding_bus) = self.cuda_binding_bus {
            binding_bus.lookup_key(
                builder,
                CudaSharedForestBindingMessageV19 {
                    proof_idx: prep.local_proof_index.into(),
                    descriptor_digest: local.descriptor_root.map(Into::into),
                    root: local.physical_root.map(Into::into),
                    column_start: local.column_start.into(),
                    column_width: local.column_width.into(),
                    forest_width: local.forest_width.into(),
                    rows_per_query: local.rows_per_query.into(),
                },
                prep.active,
            );
        } else {
            for limb in 0..DIGEST_SIZE {
                builder
                    .when(prep.active)
                    .assert_eq(local.descriptor_root[limb], local.physical_root[limb]);
            }
            for geometry in [
                local.column_start,
                local.column_width,
                local.forest_width,
                local.rows_per_query,
            ] {
                builder.when(prep.active).assert_zero(geometry);
            }
        }
    }
}

/// One verifier-owned update in canonical History order.
///
/// The vector position remains the canonical History update position used by
/// the replay producer. `proof_index` is the surrounding SWIRL proof namespace
/// (whole-History today, chunk-local for bounded History), while
/// `segment_index` remains absolute and is bound into the WARP transcript.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlannedVaccHistoryUpdateV19 {
    pub proof_index: u32,
    pub segment_index: u32,
    pub update_index: u32,
    pub shard_ordinal: u16,
    pub segment_source_ordinal: u16,
    pub relation_digest: Digest,
    pub include_prior: bool,
}

/// Exact heavy-verifier grouping key. Relation identity is not a shape
/// selector: all relation prefixes are authenticated by the group's fixed
/// event catalog and one ragged proof-event stream.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct VaccHistoryShapeGroupKeyV19 {
    pub shape: NativeStandardVaccShapeProfile,
    pub include_prior: bool,
}

#[derive(Clone, Debug)]
pub struct VaccHistoryShapeGroupPlanV19 {
    pub key: VaccHistoryShapeGroupKeyV19,
    pub relation_profiles: Vec<NativeStandardVaccProfile>,
    /// Global canonical update positions, in increasing order.
    pub update_positions: Vec<usize>,
}

/// Backend-neutral grouping and dense-local-ID assignment.
#[derive(Clone, Debug)]
pub struct VaccHistoryGroupPlanV19 {
    updates: Vec<PlannedVaccHistoryUpdateV19>,
    groups: Vec<VaccHistoryShapeGroupPlanV19>,
    global_to_group: Vec<(usize, usize)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VaccHistoryGroupsErrorV19 {
    EmptyRelationCatalog,
    InvalidRelationProfile,
    DuplicateRelationProfile,
    UnknownRelation,
    DuplicatePlannedUpdate,
    MissingCudaGroupProfile,
    DuplicateCudaGroupProfile,
    UnexpectedCudaGroupProfile,
    RecordCount,
    NonCanonicalRecordOrder,
    NonCanonicalGlobalProofIndex,
    RecordGroup,
    OpeningIdRemapPlan,
    CanonicalOpeningRecordCount,
    NonCanonicalOpeningRecord,
    Core(DirectAirVaccVerifierErrorV19),
    Cuda(CudaDirectAirVaccVerifierErrorV19),
    Producer(PositiveProducerErrorV19),
    AirTraceCount,
}

impl From<DirectAirVaccVerifierErrorV19> for VaccHistoryGroupsErrorV19 {
    fn from(value: DirectAirVaccVerifierErrorV19) -> Self {
        Self::Core(value)
    }
}

impl From<CudaDirectAirVaccVerifierErrorV19> for VaccHistoryGroupsErrorV19 {
    fn from(value: CudaDirectAirVaccVerifierErrorV19) -> Self {
        Self::Cuda(value)
    }
}

impl From<PositiveProducerErrorV19> for VaccHistoryGroupsErrorV19 {
    fn from(value: PositiveProducerErrorV19) -> Self {
        Self::Producer(value)
    }
}

impl VaccHistoryGroupPlanV19 {
    pub fn new(
        relation_catalog: Vec<NativeStandardVaccProfile>,
        updates: Vec<PlannedVaccHistoryUpdateV19>,
    ) -> Result<Self, VaccHistoryGroupsErrorV19> {
        if relation_catalog.is_empty() && !updates.is_empty() {
            return Err(VaccHistoryGroupsErrorV19::EmptyRelationCatalog);
        }
        for (index, profile) in relation_catalog.iter().enumerate() {
            if profile.validate().is_err() {
                return Err(VaccHistoryGroupsErrorV19::InvalidRelationProfile);
            }
            if relation_catalog[..index]
                .iter()
                .any(|prior| prior.relation_digest() == profile.relation_digest())
            {
                return Err(VaccHistoryGroupsErrorV19::DuplicateRelationProfile);
            }
        }

        let mut groups: Vec<VaccHistoryShapeGroupPlanV19> = Vec::new();
        let mut global_to_group = Vec::with_capacity(updates.len());
        for (global_index, update) in updates.iter().enumerate() {
            if updates[..global_index].iter().any(|prior| {
                (prior.segment_index == update.segment_index
                    && prior.proof_index != update.proof_index)
                    || (prior.segment_index != update.segment_index
                        && prior.proof_index == update.proof_index)
            }) {
                return Err(VaccHistoryGroupsErrorV19::OpeningIdRemapPlan);
            }
            if updates[..global_index].iter().any(|prior| {
                prior.segment_index == update.segment_index
                    && prior.shard_ordinal == update.shard_ordinal
            }) {
                return Err(VaccHistoryGroupsErrorV19::DuplicatePlannedUpdate);
            }
            let relation = relation_catalog
                .iter()
                .find(|profile| profile.relation_digest() == update.relation_digest)
                .ok_or(VaccHistoryGroupsErrorV19::UnknownRelation)?;
            let key = VaccHistoryShapeGroupKeyV19 {
                shape: relation.shape_profile(),
                include_prior: update.include_prior,
            };
            let group_index = groups
                .iter()
                .position(|group| group.key == key)
                .unwrap_or_else(|| {
                    groups.push(VaccHistoryShapeGroupPlanV19 {
                        key: key.clone(),
                        relation_profiles: Vec::new(),
                        update_positions: Vec::new(),
                    });
                    groups.len() - 1
                });
            let group = &mut groups[group_index];
            if !group
                .relation_profiles
                .iter()
                .any(|profile| profile.relation_digest() == update.relation_digest)
            {
                group.relation_profiles.push(relation.clone());
            }
            let local_index = group.update_positions.len();
            group.update_positions.push(global_index);
            global_to_group.push((group_index, local_index));
        }

        Ok(Self {
            updates,
            groups,
            global_to_group,
        })
    }

    #[must_use]
    pub fn updates(&self) -> &[PlannedVaccHistoryUpdateV19] {
        &self.updates
    }

    #[must_use]
    pub fn groups(&self) -> &[VaccHistoryShapeGroupPlanV19] {
        &self.groups
    }

    #[must_use]
    pub fn group_local_id(&self, global_index: usize) -> Option<(usize, usize)> {
        self.global_to_group.get(global_index).copied()
    }

    fn validate_record(
        &self,
        global_index: usize,
        producer: &DirectAirVaccProducerRecordV19,
        has_prior_witness: bool,
    ) -> Result<(), VaccHistoryGroupsErrorV19> {
        let expected = self
            .updates
            .get(global_index)
            .ok_or(VaccHistoryGroupsErrorV19::RecordCount)?;
        if producer.proof_index
            != u32::try_from(global_index)
                .map_err(|_| VaccHistoryGroupsErrorV19::NonCanonicalGlobalProofIndex)?
        {
            return Err(VaccHistoryGroupsErrorV19::NonCanonicalGlobalProofIndex);
        }
        if producer.segment_index != expected.segment_index
            || producer.update_index != expected.update_index
            || producer.shard_ordinal != expected.shard_ordinal
            || producer.relation_digest != expected.relation_digest
            || producer.has_prior != expected.include_prior
        {
            return Err(VaccHistoryGroupsErrorV19::NonCanonicalRecordOrder);
        }
        if has_prior_witness != expected.include_prior {
            return Err(VaccHistoryGroupsErrorV19::RecordGroup);
        }
        Ok(())
    }

    fn opening_id_remap_entries(
        &self,
        group: &VaccHistoryShapeGroupPlanV19,
    ) -> Result<Vec<CertifiedSwirlOpeningIdRemapEntryV19>, VaccHistoryGroupsErrorV19> {
        group
            .update_positions
            .iter()
            .enumerate()
            .map(|(local_index, &global_update_position)| {
                let update = self
                    .updates
                    .get(global_update_position)
                    .ok_or(VaccHistoryGroupsErrorV19::OpeningIdRemapPlan)?;
                Ok(CertifiedSwirlOpeningIdRemapEntryV19 {
                    global_update_position,
                    canonical_proof_index: update.proof_index,
                    segment_index: update.segment_index,
                    segment_source_ordinal: update.segment_source_ordinal,
                    registry_shard_ordinal: update.shard_ordinal,
                    local_proof_index: u32::try_from(local_index)
                        .map_err(|_| VaccHistoryGroupsErrorV19::OpeningIdRemapPlan)?,
                })
            })
            .collect()
    }

    fn canonical_records_for_group<'a>(
        &self,
        group: &VaccHistoryShapeGroupPlanV19,
        canonical_records: &'a [CanonicalSwirlRawOpeningRecordV19],
    ) -> Result<Vec<&'a CanonicalSwirlRawOpeningRecordV19>, VaccHistoryGroupsErrorV19> {
        if canonical_records.len() != self.updates.len() {
            return Err(VaccHistoryGroupsErrorV19::CanonicalOpeningRecordCount);
        }
        group
            .update_positions
            .iter()
            .map(|&global_index| {
                let expected = self
                    .updates
                    .get(global_index)
                    .ok_or(VaccHistoryGroupsErrorV19::OpeningIdRemapPlan)?;
                let record = canonical_records
                    .get(global_index)
                    .ok_or(VaccHistoryGroupsErrorV19::CanonicalOpeningRecordCount)?;
                if record.proof_index != expected.proof_index
                    || record.segment_index != expected.segment_index
                    || record.segment_source_ordinal != expected.segment_source_ordinal
                    || record.registry_shard_ordinal != expected.shard_ordinal
                    || record.relation_digest != expected.relation_digest
                {
                    return Err(VaccHistoryGroupsErrorV19::NonCanonicalOpeningRecord);
                }
                Ok(record)
            })
            .collect()
    }
}

struct AllocatedVaccGroupBusesV19 {
    shared: BusInventory,
    buses: NativeWarpPcdBusInventory,
    history_buses: DirectAirVaccHistoryBusesV19,
    protocol_bus: NativeStandardVaccProtocolBus,
    end_bus: NativeStandardVaccEndBus,
    root_bus: NativeStandardVaccRootBus,
    digest_bus: NativeStandardVaccDigestBus,
    swirl_opening_bus: CertifiedSwirlRawOpeningBusV19,
    fresh_explicit_bus: CertifiedFreshExplicitDigestBusV19,
    cuda_descriptor_bus: CudaSharedForestDescriptorBusV19,
    cuda_binding_bus: CudaSharedForestBindingBusV19,
}

fn allocate_group_buses_v19(manager: &mut BusIndexManager) -> AllocatedVaccGroupBusesV19 {
    let shared = BusInventory::new(manager);
    let buses = NativeWarpPcdBusInventory::new(manager.next_bus_idx());
    *manager = BusIndexManager::from_next_bus_idx(buses.next_bus_idx());
    // `DirectAirVaccVerifierModuleV19` derives its private prefix-event bus
    // from the first free inventory index. Reserve it before allocating any
    // externally visible group buses.
    let prefix_event_bus_index = manager.new_bus_idx();
    debug_assert_eq!(prefix_event_bus_index, buses.next_bus_idx());
    let history_buses = DirectAirVaccHistoryBusesV19 {
        context: DirectAirVaccContextBusV19::new(manager.new_bus_idx()),
        input: CertifiedDirectAirVaccInputBusV19::new(manager.new_bus_idx()),
    };
    AllocatedVaccGroupBusesV19 {
        shared,
        buses,
        history_buses,
        protocol_bus: NativeStandardVaccProtocolBus::new(manager.new_bus_idx()),
        end_bus: NativeStandardVaccEndBus::new(manager.new_bus_idx()),
        root_bus: NativeStandardVaccRootBus::new(manager.new_bus_idx()),
        digest_bus: NativeStandardVaccDigestBus::new(manager.new_bus_idx()),
        swirl_opening_bus: CertifiedSwirlRawOpeningBusV19::new(manager.new_bus_idx()),
        fresh_explicit_bus: CertifiedFreshExplicitDigestBusV19::new(manager.new_bus_idx()),
        cuda_descriptor_bus: CudaSharedForestDescriptorBusV19::new(manager.new_bus_idx()),
        cuda_binding_bus: CudaSharedForestBindingBusV19::new(manager.new_bus_idx()),
    }
}

fn producer_air_v19(
    module: &DirectAirVaccVerifierModuleV19,
    history_bus: CertifiedWarpReplayBusV19,
    history_compress_bus: HistoryPoseidon2CompressBusV19,
    swirl_opening_bus: CertifiedSwirlRawOpeningBusV19,
    fresh_explicit_bus: CertifiedFreshExplicitDigestBusV19,
    setup_schedule: WarpReplayProducerScheduleV19,
) -> WarpReplayProducerAirV19 {
    WarpReplayProducerAirV19 {
        log_message_len: module.profile.log_message_len,
        log_codeword_len: module.profile.log_codeword_len,
        beta_len: module.profile.beta_len,
        compress_bus: history_compress_bus,
        history_bus,
        context_bus: module.history_buses.context,
        swirl_opening_bus,
        vacc_input_bus: module.history_buses.input,
        canonical_vacc_input_bus: None,
        fresh_explicit_bus,
        fresh_explicit_lookup_count: 0,
        checkpoint_bus: module.buses.transcript_checkpoint,
        batching_claim_bus: module.buses.certified_batching_claim,
        next_accumulator_digest_bus: module.buses.certified_accumulator_digest,
        setup_schedule: Some(setup_schedule),
    }
}

pub struct CpuVaccHistoryShapeGroupV19 {
    pub plan: VaccHistoryShapeGroupPlanV19,
    pub verifier: DirectAirVaccVerifierModuleV19,
    pub opening_id_remap: CertifiedSwirlOpeningIdRemapAirV19,
    pub producer: WarpReplayProducerAirV19,
    /// Dense group-local opening bus. Only [`CertifiedSwirlOpeningIdRemapAirV19`]
    /// may publish onto it.
    pub swirl_opening_bus: CertifiedSwirlRawOpeningBusV19,
}

pub struct CpuShapeGroupedVaccHistoryV19 {
    pub plan: VaccHistoryGroupPlanV19,
    pub groups: Vec<CpuVaccHistoryShapeGroupV19>,
    pub canonical_swirl_opening_bus: CertifiedSwirlRawOpeningBusV19,
    pub history_bus: CertifiedWarpReplayBusV19,
    pub history_compress_bus: HistoryPoseidon2CompressBusV19,
}

impl CpuShapeGroupedVaccHistoryV19 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        relation_catalog: Vec<NativeStandardVaccProfile>,
        updates: Vec<PlannedVaccHistoryUpdateV19>,
        manager: &mut BusIndexManager,
        canonical_swirl_opening_bus: CertifiedSwirlRawOpeningBusV19,
        history_bus: CertifiedWarpReplayBusV19,
        history_compress_bus: HistoryPoseidon2CompressBusV19,
        system_params: SystemParams,
    ) -> Result<Self, VaccHistoryGroupsErrorV19> {
        let plan = VaccHistoryGroupPlanV19::new(relation_catalog, updates)?;
        let mut groups = Vec::with_capacity(plan.groups.len());
        for group_plan in &plan.groups {
            let allocated = allocate_group_buses_v19(manager);
            let verifier = DirectAirVaccVerifierModuleV19::new_shape_batched(
                group_plan.relation_profiles.clone(),
                group_plan.key.include_prior,
                allocated.shared,
                allocated.buses,
                allocated.history_buses,
                allocated.protocol_bus,
                allocated.end_bus,
                allocated.root_bus,
                allocated.digest_bus,
                system_params.clone(),
            )?;
            let producer = producer_air_v19(
                &verifier,
                history_bus,
                history_compress_bus,
                allocated.swirl_opening_bus,
                allocated.fresh_explicit_bus,
                WarpReplayProducerScheduleV19::new(
                    0,
                    group_plan.update_positions.len(),
                    group_plan.key.include_prior,
                )?,
            );
            let opening_id_remap = CertifiedSwirlOpeningIdRemapAirV19::new(
                plan.opening_id_remap_entries(group_plan)?,
                canonical_swirl_opening_bus,
                allocated.swirl_opening_bus,
                None,
            )?;
            groups.push(CpuVaccHistoryShapeGroupV19 {
                plan: group_plan.clone(),
                verifier,
                opening_id_remap,
                producer,
                swirl_opening_bus: allocated.swirl_opening_bus,
            });
        }
        Ok(Self {
            plan,
            groups,
            canonical_swirl_opening_bus,
            history_bus,
            history_compress_bus,
        })
    }

    /// AIR order without the physical Poseidon owner: one heavy verifier
    /// bundle (retaining its group-local TranscriptAir but omitting its local
    /// Poseidon2Air), its setup-owned opening-ID remap, then one positive replay
    /// producer for each exact shape/prior group.
    #[must_use]
    pub fn airs_without_shared_poseidon<PCS: StarkProtocolConfig<F = F>>(
        &self,
    ) -> Vec<AirRef<PCS>> {
        let mut airs = Vec::new();
        for group in &self.groups {
            airs.extend(group.verifier.airs_without_poseidon::<PCS>());
            airs.push(Arc::new(group.opening_id_remap.clone()));
            airs.push(Arc::new(group.producer.clone()));
        }
        airs
    }

    /// Distinct transcript/Poseidon owners in the same canonical order as the
    /// grouped input packet. Their transcript buses and local proof-ID
    /// namespaces remain disjoint; only the expensive permutation columns are
    /// shared by the multi-bus table.
    #[must_use]
    pub fn shared_poseidon_owners(&self) -> Vec<&NativeWarpTranscriptModule> {
        self.groups
            .iter()
            .map(|group| &group.verifier.transcript)
            .collect()
    }

    #[must_use]
    pub fn shared_poseidon_air<PCS: StarkProtocolConfig<F = F>>(&self) -> Option<AirRef<PCS>> {
        let owners = self.shared_poseidon_owners();
        (!owners.is_empty())
            .then(|| NativeWarpTranscriptModule::multi_bus_poseidon_air::<PCS>(&owners))
    }

    /// Complete AIR order: all group-local AIRs followed by exactly one
    /// multi-bus Poseidon AIR, when at least one group is active.
    #[must_use]
    pub fn airs<PCS: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<PCS>> {
        let mut airs = self.airs_without_shared_poseidon::<PCS>();
        if let Some(poseidon) = self.shared_poseidon_air::<PCS>() {
            airs.push(poseidon);
        }
        airs
    }

    /// Generate group witnesses and one complete input packet per distinct
    /// Poseidon bus owner, without constructing a physical Poseidon table.
    /// A larger History composite may consume this together with
    /// [`Self::shared_poseidon_owners`] and install the sole table itself.
    pub fn generate_traces_without_shared_poseidon(
        &self,
        config: &NativeSC,
        records: &[DirectAirVaccVerifierRecordV19<'_>],
        canonical_openings: &[CanonicalSwirlRawOpeningRecordV19],
    ) -> Result<ShapeGroupedVaccHistoryTraceV19, VaccHistoryGroupsErrorV19> {
        if records.len() != self.plan.updates.len() {
            return Err(VaccHistoryGroupsErrorV19::RecordCount);
        }
        for (global_index, record) in records.iter().enumerate() {
            self.plan
                .validate_record(global_index, record.producer, record.prior.is_some())?;
        }

        let mut output = ShapeGroupedVaccHistoryTraceV19::new(records.len());
        for (group_index, group) in self.groups.iter().enumerate() {
            let canonical_group = self
                .plan
                .canonical_records_for_group(&group.plan, canonical_openings)?;
            let canonical_group = canonical_group.into_iter().cloned().collect::<Vec<_>>();
            let opening_id_remap_trace = group.opening_id_remap.generate_trace(&canonical_group)?;
            let remapped = group
                .plan
                .update_positions
                .iter()
                .enumerate()
                .map(|(local_index, &global_index)| {
                    let mut producer = records[global_index].producer.clone();
                    producer.proof_index = u32::try_from(local_index)
                        .map_err(|_| VaccHistoryGroupsErrorV19::RecordGroup)?;
                    Ok(producer)
                })
                .collect::<Result<Vec<_>, VaccHistoryGroupsErrorV19>>()?;
            let local_records = group
                .plan
                .update_positions
                .iter()
                .enumerate()
                .map(
                    |(local_index, &global_index)| DirectAirVaccVerifierRecordV19 {
                        producer: &remapped[local_index],
                        verification: records[global_index].verification,
                        transcript: records[global_index].transcript,
                        prior: records[global_index].prior,
                    },
                )
                .collect::<Vec<_>>();
            let verifier_trace = group
                .verifier
                .generate_traces_for_shared_poseidon(config, &local_records)?;
            output.push_group(
                group_index,
                &group.plan,
                opening_id_remap_trace,
                &group.producer,
                remapped,
                verifier_trace,
            )?;
        }
        output.finish(self.airs_without_shared_poseidon::<NativeSC>().len())
    }

    pub fn generate_traces(
        &self,
        config: &NativeSC,
        records: &[DirectAirVaccVerifierRecordV19<'_>],
        canonical_openings: &[CanonicalSwirlRawOpeningRecordV19],
    ) -> Result<ShapeGroupedVaccHistoryTraceV19, VaccHistoryGroupsErrorV19> {
        let mut output =
            self.generate_traces_without_shared_poseidon(config, records, canonical_openings)?;
        attach_shared_poseidon_trace_v19(&self.shared_poseidon_owners(), &mut output)?;
        if output.traces.len() != self.airs::<NativeSC>().len() {
            return Err(VaccHistoryGroupsErrorV19::AirTraceCount);
        }
        Ok(output)
    }
}

/// CUDA forest profile keyed independently of prover records.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CudaVaccHistoryGroupProfileV19 {
    pub key: VaccHistoryShapeGroupKeyV19,
    pub forest: CudaSharedForestVerifierProfileV19,
}

pub struct CudaVaccHistoryShapeGroupV19 {
    pub plan: VaccHistoryShapeGroupPlanV19,
    pub verifier: CudaDirectAirVaccVerifierModuleV19,
    pub opening_id_remap: CertifiedSwirlOpeningIdRemapAirV19,
    pub producer: WarpReplayProducerAirV19,
    pub swirl_opening_bus: CertifiedSwirlRawOpeningBusV19,
}

pub struct CudaShapeGroupedVaccHistoryV19 {
    pub plan: VaccHistoryGroupPlanV19,
    pub groups: Vec<CudaVaccHistoryShapeGroupV19>,
    pub canonical_swirl_opening_bus: CertifiedSwirlRawOpeningBusV19,
    pub history_bus: CertifiedWarpReplayBusV19,
    pub history_compress_bus: HistoryPoseidon2CompressBusV19,
}

impl CudaShapeGroupedVaccHistoryV19 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        relation_catalog: Vec<NativeStandardVaccProfile>,
        updates: Vec<PlannedVaccHistoryUpdateV19>,
        forest_profiles: Vec<CudaVaccHistoryGroupProfileV19>,
        manager: &mut BusIndexManager,
        canonical_swirl_opening_bus: CertifiedSwirlRawOpeningBusV19,
        history_bus: CertifiedWarpReplayBusV19,
        history_compress_bus: HistoryPoseidon2CompressBusV19,
        system_params: SystemParams,
    ) -> Result<Self, VaccHistoryGroupsErrorV19> {
        let plan = VaccHistoryGroupPlanV19::new(relation_catalog, updates)?;
        for (index, profile) in forest_profiles.iter().enumerate() {
            if forest_profiles[..index]
                .iter()
                .any(|prior| prior.key == profile.key)
            {
                return Err(VaccHistoryGroupsErrorV19::DuplicateCudaGroupProfile);
            }
            if !plan.groups.iter().any(|group| group.key == profile.key) {
                return Err(VaccHistoryGroupsErrorV19::UnexpectedCudaGroupProfile);
            }
        }
        let mut groups = Vec::with_capacity(plan.groups.len());
        for group_plan in &plan.groups {
            let forest = forest_profiles
                .iter()
                .find(|profile| profile.key == group_plan.key)
                .ok_or(VaccHistoryGroupsErrorV19::MissingCudaGroupProfile)?;
            let allocated = allocate_group_buses_v19(manager);
            let verifier = CudaDirectAirVaccVerifierModuleV19::new_shape_batched(
                group_plan.relation_profiles.clone(),
                group_plan.key.include_prior,
                forest.forest.clone(),
                allocated.shared,
                allocated.buses,
                allocated.history_buses,
                allocated.protocol_bus,
                allocated.end_bus,
                allocated.root_bus,
                allocated.digest_bus,
                allocated.cuda_descriptor_bus,
                allocated.cuda_binding_bus,
                system_params.clone(),
            )?;
            let producer = producer_air_v19(
                &verifier.core,
                history_bus,
                history_compress_bus,
                allocated.swirl_opening_bus,
                allocated.fresh_explicit_bus,
                WarpReplayProducerScheduleV19::new(
                    0,
                    group_plan.update_positions.len(),
                    group_plan.key.include_prior,
                )?,
            );
            let opening_id_remap = CertifiedSwirlOpeningIdRemapAirV19::new(
                plan.opening_id_remap_entries(group_plan)?,
                canonical_swirl_opening_bus,
                allocated.swirl_opening_bus,
                Some(allocated.cuda_binding_bus),
            )?;
            groups.push(CudaVaccHistoryShapeGroupV19 {
                plan: group_plan.clone(),
                verifier,
                opening_id_remap,
                producer,
                swirl_opening_bus: allocated.swirl_opening_bus,
            });
        }
        Ok(Self {
            plan,
            groups,
            canonical_swirl_opening_bus,
            history_bus,
            history_compress_bus,
        })
    }

    #[must_use]
    pub fn airs_without_shared_poseidon<PCS: StarkProtocolConfig<F = F>>(
        &self,
    ) -> Vec<AirRef<PCS>> {
        let mut airs = Vec::new();
        for group in &self.groups {
            airs.extend(group.verifier.core.airs_without_poseidon::<PCS>());
            airs.extend(group.verifier.forest.airs::<PCS>());
            airs.push(Arc::new(group.opening_id_remap.clone()));
            airs.push(Arc::new(group.producer.clone()));
        }
        airs
    }

    #[must_use]
    pub fn shared_poseidon_owners(&self) -> Vec<&NativeWarpTranscriptModule> {
        self.groups
            .iter()
            .map(|group| &group.verifier.core.transcript)
            .collect()
    }

    #[must_use]
    pub fn shared_poseidon_air<PCS: StarkProtocolConfig<F = F>>(&self) -> Option<AirRef<PCS>> {
        let owners = self.shared_poseidon_owners();
        (!owners.is_empty())
            .then(|| NativeWarpTranscriptModule::multi_bus_poseidon_air::<PCS>(&owners))
    }

    #[must_use]
    pub fn airs<PCS: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<PCS>> {
        let mut airs = self.airs_without_shared_poseidon::<PCS>();
        if let Some(poseidon) = self.shared_poseidon_air::<PCS>() {
            airs.push(poseidon);
        }
        airs
    }

    pub fn generate_traces_without_shared_poseidon(
        &self,
        config: &NativeSC,
        records: &[CudaDirectAirVaccVerifierRecordV19<'_>],
        canonical_openings: &[CanonicalSwirlRawOpeningRecordV19],
    ) -> Result<ShapeGroupedVaccHistoryTraceV19, VaccHistoryGroupsErrorV19> {
        if records.len() != self.plan.updates.len() {
            return Err(VaccHistoryGroupsErrorV19::RecordCount);
        }
        for (global_index, record) in records.iter().enumerate() {
            self.plan
                .validate_record(global_index, record.producer, record.prior.is_some())?;
        }

        let mut output = ShapeGroupedVaccHistoryTraceV19::new(records.len());
        for (group_index, group) in self.groups.iter().enumerate() {
            let canonical_group = self
                .plan
                .canonical_records_for_group(&group.plan, canonical_openings)?;
            let canonical_group = canonical_group.into_iter().cloned().collect::<Vec<_>>();
            let opening_id_remap_trace = group.opening_id_remap.generate_trace(&canonical_group)?;
            let remapped = group
                .plan
                .update_positions
                .iter()
                .enumerate()
                .map(|(local_index, &global_index)| {
                    let mut producer = records[global_index].producer.clone();
                    producer.proof_index = u32::try_from(local_index)
                        .map_err(|_| VaccHistoryGroupsErrorV19::RecordGroup)?;
                    Ok(producer)
                })
                .collect::<Result<Vec<_>, VaccHistoryGroupsErrorV19>>()?;
            let forests = group
                .plan
                .update_positions
                .iter()
                .enumerate()
                .map(|(local_index, &global_index)| {
                    clone_cuda_forest_with_local_id_v19(
                        records[global_index].shared_forest,
                        local_index,
                    )
                })
                .collect::<Vec<_>>();
            let local_records = group
                .plan
                .update_positions
                .iter()
                .enumerate()
                .map(
                    |(local_index, &global_index)| CudaDirectAirVaccVerifierRecordV19 {
                        producer: &remapped[local_index],
                        verification: records[global_index].verification,
                        transcript: records[global_index].transcript,
                        prior: records[global_index].prior,
                        shared_forest: &forests[local_index],
                    },
                )
                .collect::<Vec<_>>();
            let verifier_trace = generate_cuda_group_traces_for_shared_poseidon_v19(
                &group.verifier,
                config,
                &local_records,
            )?;
            output.push_group(
                group_index,
                &group.plan,
                opening_id_remap_trace,
                &group.producer,
                remapped,
                verifier_trace,
            )?;
        }
        output.finish(self.airs_without_shared_poseidon::<NativeSC>().len())
    }

    pub fn generate_traces(
        &self,
        config: &NativeSC,
        records: &[CudaDirectAirVaccVerifierRecordV19<'_>],
        canonical_openings: &[CanonicalSwirlRawOpeningRecordV19],
    ) -> Result<ShapeGroupedVaccHistoryTraceV19, VaccHistoryGroupsErrorV19> {
        let mut output =
            self.generate_traces_without_shared_poseidon(config, records, canonical_openings)?;
        attach_shared_poseidon_trace_v19(&self.shared_poseidon_owners(), &mut output)?;
        if output.traces.len() != self.airs::<NativeSC>().len() {
            return Err(VaccHistoryGroupsErrorV19::AirTraceCount);
        }
        Ok(output)
    }
}

fn generate_cuda_group_traces_for_shared_poseidon_v19(
    module: &CudaDirectAirVaccVerifierModuleV19,
    config: &NativeSC,
    records: &[CudaDirectAirVaccVerifierRecordV19<'_>],
) -> Result<DirectAirVaccSharedPoseidonBatchTraceV19, CudaDirectAirVaccVerifierErrorV19> {
    if records.is_empty() {
        return Err(CudaDirectAirVaccVerifierErrorV19::RecordShape(
            "empty CUDA VACC batch",
        ));
    }
    for (proof_idx, record) in records.iter().enumerate() {
        let proof_idx_u32 = u32::try_from(proof_idx)
            .map_err(|_| CudaDirectAirVaccVerifierErrorV19::RecordShape("CUDA VACC proof index"))?;
        let fresh_spans = record
            .verification
            .transcript_phases
            .iter()
            .filter(|span| span.phase == NativeTranscriptPhase::FreshCommitments)
            .collect::<Vec<_>>();
        let same_transcript = record.transcript.values()
            == record.shared_forest.transcript.values()
            && record.transcript.samples() == record.shared_forest.transcript.samples()
            && record.transcript.perm_results() == record.shared_forest.transcript.perm_results()
            && record.transcript.events() == record.shared_forest.transcript.events()
            && record.transcript.permutation_transitions()
                == record.shared_forest.transcript.permutation_transitions();
        if record.producer.proof_index != proof_idx_u32
            || record.shared_forest.proof_idx != proof_idx
            || record.shared_forest.commitment.root != record.producer.fresh_root
            || record.verification.fresh_authentication.as_slice() != [()]
            || fresh_spans.as_slice() != [&record.shared_forest.fresh_commitments_span]
            || !same_transcript
        {
            return Err(CudaDirectAirVaccVerifierErrorV19::RecordShape(
                "spliced CUDA forest/VACC record",
            ));
        }
    }

    let forest_records = records
        .iter()
        .map(|record| record.shared_forest)
        .collect::<Vec<_>>();
    let forest = module.forest.generate_traces_from_refs(&forest_records)?;
    let core_records = records
        .iter()
        .map(|record| DirectAirVaccVerifierRecordV19 {
            producer: record.producer,
            verification: record.verification,
            transcript: record.transcript,
            prior: record.prior,
        })
        .collect::<Vec<_>>();
    let mut core = module
        .core
        .generate_cuda_shared_forest_traces_for_shared_poseidon(
            config,
            &core_records,
            forest.poseidon_permutation_inputs,
            forest.poseidon_compression_inputs,
        )?;
    core.traces.extend(forest.traces);
    let expected = module.core.airs_without_poseidon::<NativeSC>().len()
        + module.forest.airs::<NativeSC>().len();
    if core.traces.len() != expected {
        return Err(CudaDirectAirVaccVerifierErrorV19::AirTraceCount);
    }
    Ok(core)
}

fn clone_cuda_forest_with_local_id_v19<'a>(
    source: &CudaSharedForestVerifierRecordV19<'a>,
    proof_idx: usize,
) -> CudaSharedForestVerifierRecordV19<'a> {
    CudaSharedForestVerifierRecordV19 {
        proof_idx,
        log_codeword_len: source.log_codeword_len,
        codeword_len: source.codeword_len,
        oracle_height: source.oracle_height,
        query_stride: source.query_stride,
        outer_depth: source.outer_depth,
        commitment: source.commitment,
        fresh_commitments_span: source.fresh_commitments_span.clone(),
        projections: source.projections.clone(),
        unique_queries: source
            .unique_queries
            .iter()
            .map(|query| CudaSharedForestUniqueQueryV19 {
                query_index: query.query_index,
                multiplicity: query.multiplicity,
                opened_rows: query.opened_rows,
                authentication_path: query.authentication_path,
                row_digests: query.row_digests.clone(),
                query_digest: query.query_digest,
                inner_merkle: query.inner_merkle.clone(),
            })
            .collect(),
        outer_merkle: source.outer_merkle.clone(),
        transcript: source.transcript,
    }
}

#[derive(Clone, Debug)]
pub struct VaccHistoryGroupPoseidonInputsV19 {
    pub group_index: usize,
    pub verifier_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub verifier_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub history_compression_inputs: Vec<[F; 2 * DIGEST_SIZE]>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VaccHistoryGroupTraceRangeV19 {
    pub group_index: usize,
    pub verifier_airs: Range<usize>,
    pub opening_id_remap_air: usize,
    pub producer_air: usize,
    pub update_positions: Vec<usize>,
}

/// Concrete AIR witnesses in exactly the order returned by `airs()`. History
/// messages and output instances are restored to canonical global order even
/// though physical proving is group-major.
#[derive(Debug)]
pub struct ShapeGroupedVaccHistoryTraceV19 {
    pub traces: Vec<RowMajorMatrix<F>>,
    pub ranges: Vec<VaccHistoryGroupTraceRangeV19>,
    pub poseidon_inputs: Vec<VaccHistoryGroupPoseidonInputsV19>,
    /// Present only when this component physically owns the final shared
    /// multi-bus Poseidon trace. A parent History composite may request the
    /// packet-only path and install a wider sole-owner table itself.
    pub shared_poseidon_air: Option<usize>,
    pub history_compression_inputs: Vec<[F; 2 * DIGEST_SIZE]>,
    pub messages: Vec<CertifiedWarpReplayMessageV19<F>>,
    pub output_instances: Vec<AccumulatorInstance<EF, Digest>>,
    pub output_instance_digests: Vec<Digest>,
    messages_by_global: Vec<Option<CertifiedWarpReplayMessageV19<F>>>,
    instances_by_global: Vec<Option<AccumulatorInstance<EF, Digest>>>,
    digests_by_global: Vec<Option<Digest>>,
}

impl ShapeGroupedVaccHistoryTraceV19 {
    fn new(update_count: usize) -> Self {
        Self {
            traces: Vec::new(),
            ranges: Vec::new(),
            poseidon_inputs: Vec::new(),
            shared_poseidon_air: None,
            history_compression_inputs: Vec::new(),
            messages: Vec::new(),
            output_instances: Vec::new(),
            output_instance_digests: Vec::new(),
            messages_by_global: (0..update_count).map(|_| None).collect(),
            instances_by_global: (0..update_count).map(|_| None).collect(),
            digests_by_global: (0..update_count).map(|_| None).collect(),
        }
    }

    fn push_group(
        &mut self,
        group_index: usize,
        plan: &VaccHistoryShapeGroupPlanV19,
        opening_id_remap_trace: RowMajorMatrix<F>,
        producer_air: &WarpReplayProducerAirV19,
        producers: Vec<DirectAirVaccProducerRecordV19>,
        verifier_trace: DirectAirVaccSharedPoseidonBatchTraceV19,
    ) -> Result<(), VaccHistoryGroupsErrorV19> {
        let producer_trace = generate_warp_replay_producer_trace_v19(
            producer_air,
            &producers,
            producers.len().next_power_of_two(),
        )?;
        if producer_trace.messages.len() != plan.update_positions.len()
            || verifier_trace.output_instances.len() != plan.update_positions.len()
            || verifier_trace.output_instance_digests.len() != plan.update_positions.len()
        {
            return Err(VaccHistoryGroupsErrorV19::AirTraceCount);
        }

        let verifier_start = self.traces.len();
        let verifier_end = verifier_start + verifier_trace.traces.len();
        self.traces.extend(verifier_trace.traces);
        let opening_id_remap_air = self.traces.len();
        self.traces.push(opening_id_remap_trace);
        let producer_air_index = self.traces.len();
        self.traces.push(producer_trace.matrix);
        self.ranges.push(VaccHistoryGroupTraceRangeV19 {
            group_index,
            verifier_airs: verifier_start..verifier_end,
            opening_id_remap_air,
            producer_air: producer_air_index,
            update_positions: plan.update_positions.clone(),
        });

        for (local_index, &global_index) in plan.update_positions.iter().enumerate() {
            if self.messages_by_global[global_index]
                .replace(producer_trace.messages[local_index].clone())
                .is_some()
                || self.instances_by_global[global_index]
                    .replace(verifier_trace.output_instances[local_index].clone())
                    .is_some()
                || self.digests_by_global[global_index]
                    .replace(verifier_trace.output_instance_digests[local_index])
                    .is_some()
            {
                return Err(VaccHistoryGroupsErrorV19::RecordGroup);
            }
        }
        self.history_compression_inputs
            .extend(producer_trace.compression_inputs.iter().copied());
        self.poseidon_inputs
            .push(VaccHistoryGroupPoseidonInputsV19 {
                group_index,
                verifier_permutation_inputs: verifier_trace.poseidon_permutation_inputs,
                verifier_compression_inputs: verifier_trace.poseidon_compression_inputs,
                history_compression_inputs: producer_trace.compression_inputs,
            });
        Ok(())
    }

    fn finish(mut self, expected_air_count: usize) -> Result<Self, VaccHistoryGroupsErrorV19> {
        if self.traces.len() != expected_air_count {
            return Err(VaccHistoryGroupsErrorV19::AirTraceCount);
        }
        self.messages = self
            .messages_by_global
            .drain(..)
            .map(|message| message.ok_or(VaccHistoryGroupsErrorV19::RecordGroup))
            .collect::<Result<_, _>>()?;
        self.output_instances = self
            .instances_by_global
            .drain(..)
            .map(|instance| instance.ok_or(VaccHistoryGroupsErrorV19::RecordGroup))
            .collect::<Result<_, _>>()?;
        self.output_instance_digests = self
            .digests_by_global
            .drain(..)
            .map(|digest| digest.ok_or(VaccHistoryGroupsErrorV19::RecordGroup))
            .collect::<Result<_, _>>()?;
        Ok(self)
    }

    #[must_use]
    pub fn air_matrices(&self) -> &[RowMajorMatrix<F>] {
        &self.traces
    }

    /// One input pair per distinct group-local Poseidon bus owner, in the same
    /// order returned by `shared_poseidon_owners()` on the CPU/CUDA component.
    /// The pairs must not be flattened into one owner: doing so would merge
    /// independent lookup-bus namespaces.
    #[must_use]
    pub fn shared_poseidon_input_packet(&self) -> Poseidon2MultibusInputs {
        self.poseidon_inputs
            .iter()
            .map(|inputs| {
                (
                    inputs.verifier_permutation_inputs.clone(),
                    inputs.verifier_compression_inputs.clone(),
                )
            })
            .collect()
    }
}

fn attach_shared_poseidon_trace_v19(
    owners: &[&NativeWarpTranscriptModule],
    output: &mut ShapeGroupedVaccHistoryTraceV19,
) -> Result<(), VaccHistoryGroupsErrorV19> {
    if owners.len() != output.poseidon_inputs.len() || output.shared_poseidon_air.is_some() {
        return Err(VaccHistoryGroupsErrorV19::AirTraceCount);
    }
    if owners.is_empty() {
        return Ok(());
    }
    let mut tables = owners[0]
        .build_poseidon2_multibus_traces(output.shared_poseidon_input_packet())
        .ok_or(VaccHistoryGroupsErrorV19::AirTraceCount)?;
    if tables.len() != 1 {
        return Err(VaccHistoryGroupsErrorV19::AirTraceCount);
    }
    output.shared_poseidon_air = Some(output.traces.len());
    output
        .traces
        .push(tables.pop().expect("one checked Poseidon table"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use core::borrow::{Borrow, BorrowMut};
    use std::sync::Arc;

    use openvm_stark_backend::{
        air_builders::{
            debug::{check_constraints, check_logup},
            symbolic::get_symbolic_builder,
        },
        interaction::{InteractionBuilder, SymbolicInteraction},
        keygen::types::TraceWidth,
        p3_air::{Air, AirBuilder, BaseAir},
        p3_field::{PrimeCharacteristicRing, PrimeField32},
        p3_matrix::{dense::RowMajorMatrixView, Matrix},
        AnyAir, BaseAirWithPublicValues, PartitionedBaseAir, StarkEngine,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::{
        poseidon2_compress_with_capacity, BabyBearPoseidon2CpuEngine, DuplexSponge,
    };

    use super::*;
    use crate::circuit::native_warp_history_v19::{
        vacc_verifier::tests::{fixture_for_air_id, fixture_for_air_id_and_height, Fixture},
        CertifiedSwirlRawOpeningMessageV19, HistoryPoseidon2CompressMessageV19,
        WarpReplayProducerColsV19, LOGUP_ONLY_MODE_TAG_V19, NATIVE_WARP_HISTORY_PROTOCOL_V19,
    };

    struct CpuFixtureSet {
        fixtures: Vec<Fixture>,
        producers: Vec<DirectAirVaccProducerRecordV19>,
        canonical_openings: Vec<CanonicalSwirlRawOpeningRecordV19>,
        plan: Vec<PlannedVaccHistoryUpdateV19>,
        component: CpuShapeGroupedVaccHistoryV19,
    }

    fn grouping_test_params() -> SystemParams {
        let mut params = SystemParams::new_for_testing(10);
        params.max_constraint_degree = 8;
        // The direct-input lookup is intentionally a fixed wide typed message
        // (417 fields at the current v19 maxima). Production History params
        // likewise need a 512-field LogUp message envelope.
        params.logup.log_max_message_length = 9;
        params
    }

    fn fixture_set() -> CpuFixtureSet {
        // Group 0: shape A/no-prior, with two distinct fixed relations.
        // Group 1: shape B/no-prior.
        // Group 2: shape A/prior.
        let fixtures = vec![
            fixture_for_air_id(false, 9),
            fixture_for_air_id_and_height(false, 11, 3),
            fixture_for_air_id(false, 10),
            fixture_for_air_id(true, 9),
        ];
        let mut producers = fixtures
            .iter()
            .map(|fixture| fixture.producer.clone())
            .collect::<Vec<_>>();
        let shards = [0u16, 1, 2, 0];
        for (global_index, producer) in producers.iter_mut().enumerate() {
            producer.proof_index = global_index as u32;
            producer.shard_ordinal = shards[global_index];
        }
        let plan = producers
            .iter()
            .map(|producer| PlannedVaccHistoryUpdateV19 {
                proof_index: producer.segment_index,
                segment_index: producer.segment_index,
                update_index: producer.update_index,
                shard_ordinal: producer.shard_ordinal,
                segment_source_ordinal: producer.shard_ordinal,
                relation_digest: producer.relation_digest,
                include_prior: producer.has_prior,
            })
            .collect::<Vec<_>>();
        let relation_catalog = vec![
            fixtures[0].profile.clone(),
            fixtures[1].profile.clone(),
            fixtures[2].profile.clone(),
        ];
        let canonical_openings = producers
            .iter()
            .map(|producer| CanonicalSwirlRawOpeningRecordV19 {
                proof_index: producer.segment_index,
                segment_index: producer.segment_index,
                segment_source_ordinal: producer.shard_ordinal,
                registry_shard_ordinal: producer.shard_ordinal,
                relation_digest: producer.relation_digest,
                source_forest_root: producer.source_forest_root,
                segment_openings_digest: producer.segment_openings_digest,
                descriptor_root: producer.fresh_root,
                physical_root: producer.fresh_root,
                column_start: 0,
                column_width: 0,
                forest_width: 0,
                rows_per_query: 0,
                point: producer
                    .opening_point
                    .iter()
                    .map(|limbs| {
                        EF::from_basis_coefficients_slice(limbs)
                            .expect("fixture opening point has EF4 width")
                    })
                    .collect(),
                value: EF::from_basis_coefficients_slice(&producer.opening_value)
                    .expect("fixture opening value has EF4 width"),
            })
            .collect::<Vec<_>>();
        let mut manager = BusIndexManager::new();
        let canonical_swirl_opening_bus =
            CertifiedSwirlRawOpeningBusV19::new(manager.new_bus_idx());
        let history_bus = CertifiedWarpReplayBusV19::new(manager.new_bus_idx());
        let history_compress_bus = HistoryPoseidon2CompressBusV19::new(manager.new_bus_idx());
        let component = CpuShapeGroupedVaccHistoryV19::new(
            relation_catalog,
            plan.clone(),
            &mut manager,
            canonical_swirl_opening_bus,
            history_bus,
            history_compress_bus,
            grouping_test_params(),
        )
        .expect("shape-grouped CPU component");
        CpuFixtureSet {
            fixtures,
            producers,
            canonical_openings,
            plan,
            component,
        }
    }

    fn records<'a>(set: &'a CpuFixtureSet) -> Vec<DirectAirVaccVerifierRecordV19<'a>> {
        set.fixtures
            .iter()
            .enumerate()
            .map(|(index, fixture)| DirectAirVaccVerifierRecordV19 {
                producer: &set.producers[index],
                verification: &fixture.verification,
                transcript: &fixture.transcript,
                prior: fixture.prior.as_ref(),
            })
            .collect()
    }

    fn cuda_component_for_set(set: &CpuFixtureSet) -> CudaShapeGroupedVaccHistoryV19 {
        let forest_profiles = set
            .component
            .plan
            .groups()
            .iter()
            .map(|group| CudaVaccHistoryGroupProfileV19 {
                key: group.key.clone(),
                forest: CudaSharedForestVerifierProfileV19 {
                    log_codeword_len: group.key.shape.log_codeword_len,
                    column_width: 1,
                    rows_per_query: group.key.shape.rows_per_query,
                    outer_tree_id: 0,
                    row_tree_id_offset: 1,
                    input_variant: 1 + usize::from(group.key.include_prior) * 3,
                },
            })
            .collect();
        let mut manager = BusIndexManager::new();
        let canonical_swirl_opening_bus =
            CertifiedSwirlRawOpeningBusV19::new(manager.new_bus_idx());
        let history_bus = CertifiedWarpReplayBusV19::new(manager.new_bus_idx());
        let history_compress_bus = HistoryPoseidon2CompressBusV19::new(manager.new_bus_idx());
        CudaShapeGroupedVaccHistoryV19::new(
            vec![
                set.fixtures[0].profile.clone(),
                set.fixtures[1].profile.clone(),
                set.fixtures[2].profile.clone(),
            ],
            set.plan.clone(),
            forest_profiles,
            &mut manager,
            canonical_swirl_opening_bus,
            history_bus,
            history_compress_bus,
            grouping_test_params(),
        )
        .expect("shape-grouped CUDA component")
    }

    fn assert_grouped_poseidon_order(names: &[String], group_count: usize) {
        assert_eq!(
            names
                .last()
                .is_some_and(|name| name.starts_with("Poseidon2MultiBusAir")),
            true,
            "the sole physical Poseidon owner is last"
        );
        assert_eq!(
            names
                .iter()
                .filter(|name| name.starts_with("Poseidon2MultiBusAir"))
                .count(),
            1
        );
        assert_eq!(
            names
                .iter()
                .filter(|name| name.as_str() == "TranscriptAir")
                .count(),
            group_count,
            "each group retains an independent transcript namespace"
        );
        assert_eq!(
            names
                .iter()
                .filter(|name| name.as_str() == "CertifiedSwirlOpeningIdRemapAirV19")
                .count(),
            group_count
        );
        assert_eq!(
            names
                .iter()
                .filter(|name| name.as_str() == "WarpReplayProducerAirV19")
                .count(),
            group_count
        );
        for remap in names.iter().enumerate().filter_map(|(index, name)| {
            (name == "CertifiedSwirlOpeningIdRemapAirV19").then_some(index)
        }) {
            assert_eq!(names[remap + 1], "WarpReplayProducerAirV19");
        }
    }

    #[test]
    fn grouped_poseidon_air_count_drops_by_groups_minus_one_and_has_one_trace() {
        let set = fixture_set();
        let group_count = set.component.groups.len();
        let old_air_count = set
            .component
            .groups
            .iter()
            .map(|group| group.verifier.airs::<NativeSC>().len() + 2)
            .sum::<usize>();
        let new_airs = set.component.airs::<NativeSC>();
        assert_eq!(old_air_count - new_airs.len(), group_count - 1);

        let trace = set
            .component
            .generate_traces(
                &set.fixtures[0].config,
                &records(&set),
                &set.canonical_openings,
            )
            .expect("shared-Poseidon grouped trace");
        assert_eq!(trace.traces.len(), new_airs.len());
        assert_eq!(trace.shared_poseidon_air, Some(trace.traces.len() - 1));
        assert_eq!(trace.poseidon_inputs.len(), group_count);
        assert_eq!(trace.shared_poseidon_input_packet().len(), group_count);
    }

    #[test]
    fn cpu_and_cuda_grouped_air_order_share_the_same_poseidon_policy() {
        let set = fixture_set();
        let cuda = cuda_component_for_set(&set);
        let cpu_names = set
            .component
            .airs::<NativeSC>()
            .iter()
            .map(|air| air.name())
            .collect::<Vec<_>>();
        let cuda_names = cuda
            .airs::<NativeSC>()
            .iter()
            .map(|air| air.name())
            .collect::<Vec<_>>();
        assert_grouped_poseidon_order(&cpu_names, set.component.groups.len());
        assert_grouped_poseidon_order(&cuda_names, cuda.groups.len());
        assert_eq!(
            set.component.shared_poseidon_owners().len(),
            cuda.shared_poseidon_owners().len()
        );
    }

    #[test]
    fn plan_groups_exact_shapes_relations_and_prior_modes() {
        let set = fixture_set();
        assert_eq!(set.component.plan.groups().len(), 3);
        assert_eq!(set.component.plan.groups()[0].update_positions, [0, 2]);
        assert_eq!(
            set.component.plan.groups()[0].relation_profiles.len(),
            2,
            "A/B relation prefixes share one catalog/stream and one heavy numeric-shape verifier"
        );
        assert_eq!(set.component.plan.groups()[1].update_positions, [1]);
        assert_eq!(set.component.plan.groups()[2].update_positions, [3]);
        assert!(!set.component.plan.groups()[0].key.include_prior);
        assert!(set.component.plan.groups()[2].key.include_prior);
        assert_eq!(set.component.plan.group_local_id(0), Some((0, 0)));
        assert_eq!(set.component.plan.group_local_id(2), Some((0, 1)));
        assert_eq!(set.component.plan.group_local_id(3), Some((2, 0)));
        let remap = set.component.groups[0].opening_id_remap.entries();
        assert_eq!(remap.len(), 2);
        assert_eq!(remap[0].global_update_position, 0);
        assert_eq!(remap[0].local_proof_index, 0);
        assert_eq!(remap[0].canonical_proof_index, set.plan[0].proof_index);
        assert_eq!(remap[1].global_update_position, 2);
        assert_eq!(remap[1].local_proof_index, 1);
        assert_eq!(remap[1].canonical_proof_index, set.plan[2].proof_index);
    }

    #[test]
    fn remap_accepts_chunk_local_proof_ids_but_keeps_absolute_segments() {
        let set = fixture_set();
        let mut updates = set.plan.clone();
        for update in &mut updates {
            update.proof_index = update.segment_index + 17;
        }
        let plan = VaccHistoryGroupPlanV19::new(
            vec![
                set.fixtures[0].profile.clone(),
                set.fixtures[1].profile.clone(),
                set.fixtures[2].profile.clone(),
            ],
            updates.clone(),
        )
        .expect("chunk-local VACC group plan");
        let group = &plan.groups()[0];
        let entries = plan
            .opening_id_remap_entries(group)
            .expect("chunk-local remap entries");
        assert!(entries
            .iter()
            .all(|entry| entry.canonical_proof_index != entry.segment_index));

        let old_remap = &set.component.groups[0].opening_id_remap;
        let remap = CertifiedSwirlOpeningIdRemapAirV19::new(
            entries,
            old_remap.canonical_opening_bus,
            old_remap.local_opening_bus,
            None,
        )
        .expect("chunk-local remap AIR");
        let mut canonical = set.canonical_openings.clone();
        for (record, update) in canonical.iter_mut().zip(&updates) {
            record.proof_index = update.proof_index;
        }
        let group_records = plan
            .canonical_records_for_group(group, &canonical)
            .expect("chunk-local canonical records")
            .into_iter()
            .cloned()
            .collect::<Vec<_>>();
        let trace = remap
            .generate_trace(&group_records)
            .expect("chunk-local remap trace");
        let prep_matrix = remap.preprocessed_trace();
        let prep = prep_matrix.as_ref().map(|matrix| matrix.as_view());
        check_constraints::<_, NativeSC>(
            &remap,
            "CertifiedSwirlOpeningIdRemapAirV19(chunk-local)",
            &prep,
            &[trace.as_view()],
            &[],
        );

        let mut wrong_segment = canonical;
        wrong_segment[group.update_positions[0]].segment_index += 1;
        assert_eq!(
            plan.canonical_records_for_group(group, &wrong_segment)
                .unwrap_err(),
            VaccHistoryGroupsErrorV19::NonCanonicalOpeningRecord
        );

        let mut inconsistent = updates.clone();
        inconsistent[2].proof_index += 1;
        assert_eq!(
            VaccHistoryGroupPlanV19::new(
                vec![
                    set.fixtures[0].profile.clone(),
                    set.fixtures[1].profile.clone(),
                    set.fixtures[2].profile.clone(),
                ],
                inconsistent,
            )
            .unwrap_err(),
            VaccHistoryGroupsErrorV19::OpeningIdRemapPlan
        );
    }

    #[test]
    fn remap_rejects_reordered_duplicate_and_omitted_canonical_records() {
        let set = fixture_set();
        let honest_records = records(&set);

        let mut reordered = set.canonical_openings.clone();
        reordered.swap(0, 1);
        assert_eq!(
            set.component
                .generate_traces(&set.fixtures[0].config, &honest_records, &reordered)
                .unwrap_err(),
            VaccHistoryGroupsErrorV19::NonCanonicalOpeningRecord
        );

        let mut duplicated = set.canonical_openings.clone();
        duplicated[1] = duplicated[0].clone();
        assert_eq!(
            set.component
                .generate_traces(&set.fixtures[0].config, &honest_records, &duplicated)
                .unwrap_err(),
            VaccHistoryGroupsErrorV19::NonCanonicalOpeningRecord
        );

        assert_eq!(
            set.component
                .generate_traces(
                    &set.fixtures[0].config,
                    &honest_records,
                    &set.canonical_openings[..set.canonical_openings.len() - 1],
                )
                .unwrap_err(),
            VaccHistoryGroupsErrorV19::CanonicalOpeningRecordCount
        );
    }

    #[test]
    fn remap_rejects_wrong_segment_and_global_mapping() {
        let mut set = fixture_set();
        let honest_records = records(&set);
        let mut wrong_segment = set.canonical_openings.clone();
        wrong_segment[0].segment_index = wrong_segment[0].segment_index.wrapping_add(1);
        wrong_segment[0].proof_index = wrong_segment[0].segment_index;
        assert_eq!(
            set.component
                .generate_traces(&set.fixtures[0].config, &honest_records, &wrong_segment)
                .unwrap_err(),
            VaccHistoryGroupsErrorV19::NonCanonicalOpeningRecord
        );

        set.producers[2].proof_index = 0;
        let wrong_global = records(&set);
        assert_eq!(
            set.component
                .generate_traces(
                    &set.fixtures[0].config,
                    &wrong_global,
                    &set.canonical_openings,
                )
                .unwrap_err(),
            VaccHistoryGroupsErrorV19::NonCanonicalGlobalProofIndex
        );
    }

    #[test]
    fn remap_rejects_local_global_ordinal_swap() {
        let set = fixture_set();
        let honest_records = records(&set);
        let mut local = set.canonical_openings.clone();
        local[0].segment_source_ordinal = local[0].segment_source_ordinal.wrapping_add(1);
        assert_eq!(
            set.component
                .generate_traces(&set.fixtures[0].config, &honest_records, &local)
                .unwrap_err(),
            VaccHistoryGroupsErrorV19::NonCanonicalOpeningRecord
        );
        let mut global = set.canonical_openings.clone();
        global[0].registry_shard_ordinal = global[0].registry_shard_ordinal.wrapping_add(1);
        assert_eq!(
            set.component
                .generate_traces(&set.fixtures[0].config, &honest_records, &global)
                .unwrap_err(),
            VaccHistoryGroupsErrorV19::NonCanonicalOpeningRecord
        );
    }

    #[test]
    fn cpu_bridge_rejects_descriptor_physical_root_swap_and_range_splice() {
        let set = fixture_set();
        let group = &set.component.groups[0];
        let canonical = group
            .plan
            .update_positions
            .iter()
            .map(|&index| set.canonical_openings[index].clone())
            .collect::<Vec<_>>();
        for mutation in 0..3 {
            let mut malicious = canonical.clone();
            match mutation {
                0 => malicious[0].descriptor_root[0] += F::ONE,
                1 => malicious[0].physical_root[0] += F::ONE,
                2 => malicious[0].column_start = 1,
                _ => unreachable!(),
            }
            let trace = group
                .opening_id_remap
                .generate_trace(&malicious)
                .expect("shape-valid malicious bridge trace");
            let preprocessed = group
                .opening_id_remap
                .preprocessed_trace()
                .expect("bridge preprocessed plan");
            let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                check_constraints::<_, NativeSC>(
                    &group.opening_id_remap,
                    "CertifiedSwirlOpeningIdRemapAirV19",
                    &Some(preprocessed.as_view()),
                    &[trace.as_view()],
                    &[],
                );
            }));
            assert!(
                rejected.is_err(),
                "malicious bridge mutation {mutation} must fail"
            );
        }
    }

    #[test]
    fn grouped_trace_restores_exact_outgoing_history_order() {
        let set = fixture_set();
        let records = records(&set);
        let trace = set
            .component
            .generate_traces(&set.fixtures[0].config, &records, &set.canonical_openings)
            .expect("grouped traces");
        assert_eq!(trace.messages.len(), set.plan.len());
        assert_eq!(trace.ranges.len(), 3);
        for (global_index, message) in trace.messages.iter().enumerate() {
            let expected = &set.plan[global_index];
            let segment = message.segment_index_lo.as_canonical_u32()
                + (message.segment_index_hi.as_canonical_u32() << 16);
            let update = message.update_index_lo.as_canonical_u32()
                + (message.update_index_hi.as_canonical_u32() << 16);
            assert_eq!(segment, expected.segment_index);
            assert_eq!(update, expected.update_index);
            assert_eq!(
                message.shard_ordinal.as_canonical_u32(),
                u32::from(expected.shard_ordinal)
            );
            assert_eq!(message.relation_digest, expected.relation_digest);
        }
        assert_eq!(trace.ranges[0].update_positions, [0, 2]);
        assert_eq!(trace.ranges[1].update_positions, [1]);
        assert_eq!(trace.ranges[2].update_positions, [3]);
    }

    #[test]
    fn rejects_cross_group_local_id_splice_omission_and_wrong_group() {
        let mut set = fixture_set();
        set.producers[1].proof_index = 0; // Local zero from another group is not global ID 1.
        let bad_records = records(&set);
        assert_eq!(
            set.component
                .generate_traces(
                    &set.fixtures[0].config,
                    &bad_records,
                    &set.canonical_openings,
                )
                .unwrap_err(),
            VaccHistoryGroupsErrorV19::NonCanonicalGlobalProofIndex
        );

        set.producers[1].proof_index = 1;
        let omitted_records = records(&set);
        assert_eq!(
            set.component
                .generate_traces(
                    &set.fixtures[0].config,
                    &omitted_records[..3],
                    &set.canonical_openings,
                )
                .unwrap_err(),
            VaccHistoryGroupsErrorV19::RecordCount
        );

        set.producers[1].relation_digest = set.producers[0].relation_digest;
        let wrong_group_records = records(&set);
        assert_eq!(
            set.component
                .generate_traces(
                    &set.fixtures[0].config,
                    &wrong_group_records,
                    &set.canonical_openings,
                )
                .unwrap_err(),
            VaccHistoryGroupsErrorV19::NonCanonicalRecordOrder
        );
    }

    #[test]
    fn rejects_duplicate_plan_and_noncanonical_input_order() {
        let set = fixture_set();
        let mut duplicate = set.plan.clone();
        duplicate[1].segment_index = duplicate[0].segment_index;
        duplicate[1].shard_ordinal = duplicate[0].shard_ordinal;
        assert_eq!(
            VaccHistoryGroupPlanV19::new(
                vec![
                    set.fixtures[0].profile.clone(),
                    set.fixtures[1].profile.clone(),
                    set.fixtures[2].profile.clone(),
                ],
                duplicate,
            )
            .unwrap_err(),
            VaccHistoryGroupsErrorV19::DuplicatePlannedUpdate
        );

        let mut records = records(&set);
        records.swap(0, 1);
        assert!(matches!(
            set.component.generate_traces(
                &set.fixtures[0].config,
                &records,
                &set.canonical_openings,
            ),
            Err(VaccHistoryGroupsErrorV19::NonCanonicalGlobalProofIndex
                | VaccHistoryGroupsErrorV19::NonCanonicalRecordOrder)
        ));
    }

    /// Test provider standing in for the real composite boundary. It publishes
    /// the canonical opening message from an independently held copy of the
    /// bridge payload and the same verifier-key-owned mapping table.
    #[derive(Clone, Debug)]
    struct CanonicalOpeningSourceAir {
        remap: CertifiedSwirlOpeningIdRemapAirV19,
    }

    impl BaseAir<F> for CanonicalOpeningSourceAir {
        fn width(&self) -> usize {
            self.remap.width()
        }

        fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
            self.remap.preprocessed_trace()
        }
    }
    impl BaseAirWithPublicValues<F> for CanonicalOpeningSourceAir {}
    impl PartitionedBaseAir<F> for CanonicalOpeningSourceAir {}

    impl<AB> Air<AB> for CanonicalOpeningSourceAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder + PairBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let preprocessed = builder.preprocessed();
            let prep_row = preprocessed
                .row_slice(0)
                .expect("canonical opening source preprocessed row");
            let prep: &CertifiedSwirlOpeningIdRemapPrepColsV19<AB::Var> = (*prep_row).borrow();
            let main = builder.main();
            let row = main.row_slice(0).expect("canonical opening source row");
            let local: &CertifiedSwirlOpeningIdRemapColsV19<AB::Var> = (*row).borrow();
            self.remap.canonical_opening_bus.add_key_with_lookups(
                builder,
                CertifiedSwirlRawOpeningMessageV19 {
                    proof_index: prep.canonical_proof_index.into(),
                    protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                    mode_tag: AB::Expr::from_u32(LOGUP_ONLY_MODE_TAG_V19),
                    segment_index_lo: prep.segment_index_lo.into(),
                    segment_index_hi: prep.segment_index_hi.into(),
                    shard_ordinal: prep.segment_source_ordinal.into(),
                    source_forest_root: local.source_forest_root.map(Into::into),
                    segment_openings_digest: local.segment_openings_digest.map(Into::into),
                    root: local.descriptor_root.map(Into::into),
                    point_len: local.point_len.into(),
                    point: local.point.map(|point| point.map(Into::into)),
                    value: local.value.map(Into::into),
                },
                prep.active,
            );
        }
    }

    #[derive(Clone, Debug)]
    struct HistoryBoundaryAir {
        history_bus: CertifiedWarpReplayBusV19,
    }

    impl BaseAir<F> for HistoryBoundaryAir {
        fn width(&self) -> usize {
            WarpReplayProducerColsV19::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for HistoryBoundaryAir {}
    impl PartitionedBaseAir<F> for HistoryBoundaryAir {}

    impl<AB> Air<AB> for HistoryBoundaryAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("group boundary row");
            let local: &WarpReplayProducerColsV19<AB::Var> = (*row).borrow();
            let active = local.active;
            self.history_bus.lookup_key(
                builder,
                CertifiedWarpReplayMessageV19 {
                    protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                    segment_index_lo: local.segment_index_lo.into(),
                    segment_index_hi: local.segment_index_hi.into(),
                    update_index_lo: local.update_index_lo.into(),
                    update_index_hi: local.update_index_hi.into(),
                    shard_ordinal: local.shard_ordinal.into(),
                    source_forest_root: local.source_forest_root.map(Into::into),
                    key_digest: local.key_digest.map(Into::into),
                    relation_digest: local.relation_digest.map(Into::into),
                    opening_claim_digest: local.opening_claim_digest.map(Into::into),
                    fresh_instance_digest: local.fresh_instance_digest.map(Into::into),
                    segment_openings_digest: local.segment_openings_digest.map(Into::into),
                    prior_root: local.prior_root.map(Into::into),
                    fresh_root: local.fresh_root.map(Into::into),
                    next_root: local.next_root.map(Into::into),
                    previous_accumulator_digest: local.previous_accumulator_digest.map(Into::into),
                    next_accumulator_digest: local.next_accumulator_digest.map(Into::into),
                    authenticated_batching_claim: local
                        .authenticated_batching_claim
                        .map(Into::into),
                    previous_checkpoint_digest: local.previous_checkpoint_digest.map(Into::into),
                    next_checkpoint_digest: local.next_checkpoint_digest.map(Into::into),
                    replay_endpoint_digest: local.replay_endpoint_digest.map(Into::into),
                    replay_binding_digest: local.replay_binding_digest.map(Into::into),
                },
                active,
            );
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct HistoryCompressionSourceAir(HistoryPoseidon2CompressBusV19);

    impl BaseAir<F> for HistoryCompressionSourceAir {
        fn width(&self) -> usize {
            1 + 3 * DIGEST_SIZE
        }
    }
    impl BaseAirWithPublicValues<F> for HistoryCompressionSourceAir {}
    impl PartitionedBaseAir<F> for HistoryCompressionSourceAir {}

    impl<AB> Air<AB> for HistoryCompressionSourceAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("history compression row");
            let next = main.row_slice(1).expect("next history compression row");
            let active = row[0];
            builder.assert_bool(active);
            builder.when_transition().when(next[0]).assert_one(active);
            self.0.add_key_with_lookups(
                builder,
                HistoryPoseidon2CompressMessageV19 {
                    input: core::array::from_fn(|index| row[1 + index].into()),
                    output: core::array::from_fn(|index| row[1 + 2 * DIGEST_SIZE + index].into()),
                },
                active,
            );
        }
    }

    fn compression_trace(inputs: &[[F; 2 * DIGEST_SIZE]]) -> RowMajorMatrix<F> {
        let width = 1 + 3 * DIGEST_SIZE;
        let height = (inputs.len() + 1).next_power_of_two();
        let mut values = F::zero_vec(width * height);
        for (index, input) in inputs.iter().enumerate() {
            let row = &mut values[index * width..(index + 1) * width];
            row[0] = F::ONE;
            row[1..1 + 2 * DIGEST_SIZE].copy_from_slice(input);
            let left = input[..DIGEST_SIZE].try_into().unwrap();
            let right = input[DIGEST_SIZE..].try_into().unwrap();
            row[1 + 2 * DIGEST_SIZE..]
                .copy_from_slice(&poseidon2_compress_with_capacity(left, right).0);
        }
        RowMajorMatrix::new(values, width)
    }

    fn symbolic_interactions(air: &dyn AnyAir<NativeSC>) -> Vec<SymbolicInteraction<F>> {
        let preprocessed = BaseAir::<F>::preprocessed_trace(air).map(|trace| trace.width());
        get_symbolic_builder(
            air,
            &TraceWidth {
                preprocessed,
                cached_mains: air.cached_main_widths(),
                common_main: air.common_main_width(),
            },
        )
        .constraints()
        .interactions
    }

    fn complete_air_trace_set(
        set: &CpuFixtureSet,
        trace: &ShapeGroupedVaccHistoryTraceV19,
    ) -> (Vec<AirRef<NativeSC>>, Vec<RowMajorMatrix<F>>) {
        let mut airs = set.component.airs::<NativeSC>();
        let mut matrices = trace.traces.clone();
        for (group, range) in set.component.groups.iter().zip(&trace.ranges) {
            airs.push(Arc::new(CanonicalOpeningSourceAir {
                remap: group.opening_id_remap.clone(),
            }));
            // This is an independent copy of the composite provider witness;
            // mutating the bridge matrix cannot mutate the source matrix.
            matrices.push(trace.traces[range.opening_id_remap_air].clone());
            airs.push(Arc::new(HistoryBoundaryAir {
                history_bus: set.component.history_bus,
            }));
            matrices.push(trace.traces[range.producer_air].clone());
        }
        airs.push(Arc::new(HistoryCompressionSourceAir(
            set.component.history_compress_bus,
        )));
        matrices.push(compression_trace(&trace.history_compression_inputs));
        (airs, matrices)
    }

    fn check_air_trace_set(airs: &[AirRef<NativeSC>], matrices: &[RowMajorMatrix<F>]) {
        assert_eq!(airs.len(), matrices.len());
        let preprocessed_matrices = airs
            .iter()
            .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
            .collect::<Vec<_>>();
        let preprocessed = preprocessed_matrices
            .iter()
            .map(|trace| trace.as_ref().map(|matrix| matrix.as_view()))
            .collect::<Vec<_>>();
        for ((air, matrix), prep) in airs.iter().zip(matrices).zip(&preprocessed) {
            check_constraints::<_, NativeSC>(
                air.as_ref(),
                &air.name(),
                prep,
                &[RowMajorMatrixView::new(&matrix.values, matrix.width())],
                &[],
            );
        }
        let names = airs.iter().map(|air| air.name()).collect::<Vec<_>>();
        let interactions = airs
            .iter()
            .map(|air| symbolic_interactions(air.as_ref()))
            .collect::<Vec<_>>();
        let views = matrices
            .iter()
            .map(|matrix| vec![matrix.as_view()])
            .collect::<Vec<_>>();
        let public_values = (0..airs.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        check_logup(&names, &interactions, &preprocessed, &views, &public_values);
    }

    #[test]
    fn grouped_air_set_keygens_and_has_balanced_multisets() {
        let set = fixture_set();
        let records = records(&set);
        let trace = set
            .component
            .generate_traces(&set.fixtures[0].config, &records, &set.canonical_openings)
            .expect("grouped traces");
        let (airs, matrices) = complete_air_trace_set(&set, &trace);
        check_air_trace_set(&airs, &matrices);

        let engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(grouping_test_params());
        engine.keygen(&airs);
    }

    #[test]
    fn moving_a_poseidon_multiplicity_between_group_owners_is_rejected() {
        let set = fixture_set();
        let trace = set
            .component
            .generate_traces(
                &set.fixtures[0].config,
                &records(&set),
                &set.canonical_openings,
            )
            .expect("grouped traces");
        let (airs, mut matrices) = complete_air_trace_set(&set, &trace);
        let poseidon_air = trace.shared_poseidon_air.expect("shared Poseidon AIR");
        let table = &mut matrices[poseidon_air];
        let owner_count = set.component.groups.len();
        let table_width = table.width();
        let inner_width = table_width - 2 * owner_count;
        let row = table
            .values
            .chunks_exact_mut(table_width)
            .find(|row| row[inner_width] != F::ZERO)
            .expect("group zero has a Poseidon permutation request");
        row[inner_width] -= F::ONE;
        row[inner_width + 2] += F::ONE;

        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_air_trace_set(&airs, &matrices);
        }));
        assert!(
            rejected.is_err(),
            "multiplicity cannot move between distinct Poseidon bus owners"
        );
    }

    #[test]
    fn malicious_local_id_rewrite_unbalances_real_bridge_and_producer() {
        let set = fixture_set();
        let records = records(&set);
        let trace = set
            .component
            .generate_traces(&set.fixtures[0].config, &records, &set.canonical_openings)
            .expect("grouped traces");
        let (airs, mut matrices) = complete_air_trace_set(&set, &trace);

        // The first physical group has local IDs 0 and 1. Rewrite the first
        // producer row to ID 1 while leaving the setup-owned remap table at 0.
        // Its local opening receive can no longer cancel the bridge's send.
        let producer_air = trace.ranges[0].producer_air;
        let width = WarpReplayProducerColsV19::<F>::width();
        let cols: &mut WarpReplayProducerColsV19<F> =
            matrices[producer_air].values[..width].borrow_mut();
        cols.proof_index_lo = F::ONE;
        cols.proof_index_hi = F::ZERO;
        cols.proof_index_bits = core::array::from_fn(|limb| {
            core::array::from_fn(|bit| F::from_bool(limb == 0 && bit == 0))
        });

        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_air_trace_set(&airs, &matrices);
        }));
        assert!(
            rejected.is_err(),
            "prover-selected local opening ID must leave a LogUp imbalance"
        );
    }
}

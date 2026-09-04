//! Production protocol-v19 composition of the source manifest, the partial
//! recursive LogUp verifier, and the ephemeral SWIRL opening reduction.
//!
//! This module contains only verifier-authenticated handoffs.  In particular,
//! no AIR is allowed to copy a host endpoint, opening point, column claim, or
//! source descriptor onto an output bus without first consuming the matching
//! cryptographic provider bus.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_cpu_backend::CpuBackend;
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_recursion_circuit::{
    batch_constraint::bus::{
        BatchConstraintEndpointBus, BatchConstraintEndpointMessage, LogUpOnlyOpeningPointBus,
        LogUpOnlyOpeningPointMessage,
    },
    bus::{
        CertifiedTranscriptCheckpointBus, CertifiedTranscriptCheckpointMessage, ColumnClaimsBus,
        ColumnClaimsMessage, TranscriptBus, TranscriptEndIndexBus, TranscriptEndIndexMessage,
    },
    proof_shape::bus::{RebasedTranscriptStartBus, RebasedTranscriptStartMessage},
    system::{
        AggregationSubCircuit, BusIndexManager, BusInventory, CachedTraceCtx,
        LogUpOnlyPartialVerifier, LogUpOnlyPrefixTranscript, Preflight, RebasedTranscriptPreflight,
        RetainedLogUpOnlyProof,
    },
    transcript::{Poseidon2BusOwner, Poseidon2MultibusInputs},
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, LookupBus},
    keygen::types::MultiStarkVerifyingKey,
    native_warp::{direct_message_opening_reduction_claim_observations, DirectAirPesatIndex},
    proof::{Proof, TraceVData},
    prover::AirProvingContext,
    verifier::batch_constraints::{observe_batch_constraint_openings, verify_logup_only_prefix},
    warp_pesat::{
        PrismalinearMappedColumnBlock, PrismalinearMappedColumnRotation,
        PrismalinearMappedColumnTerm, PrismalinearMappedColumnWeight,
        TerminalStructuredLinearClaim, TerminalWeightSpec,
    },
    AirRef, BaseAirWithPublicValues, FiatShamirTranscript, PartitionedBaseAir, StarkProtocolConfig,
    TranscriptHistory, TranscriptLog,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, BabyBearPoseidon2Config, DIGEST_SIZE, D_EF, EF, F,
};
use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};
use p3_field::{
    extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing, TwoAdicField,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

#[cfg(test)]
use super::ACTIVE_CHILD_COUNT_TRACE_HEIGHT_V4;
use super::{
    generate_active_child_count_transcript_trace_v19,
    generate_active_child_count_transcript_trace_v4,
    generate_direct_logup_source_manifest_traces_v19,
    generate_grouped_one_shot_reduction_trace_v19, generate_logup_claim_derivation_trace_v19,
    generate_logup_swirl_boundary_trace_v19, observe_active_child_count_transcript_v19,
    observe_active_child_count_transcript_v4,
    producer::{LogUpOnlyProducerRecordV19, TranscriptCheckpointRecordV19},
    ActiveChildCountTranscriptAirV19, ActiveChildCountTranscriptAirV4,
    ActiveChildCountTranscriptProfileV19, ActiveChildCountTranscriptRecordV19,
    ActiveChildCountTranscriptRecordV4, CertifiedLogUpOnlyEndpointBusV19,
    CertifiedProgramFingerprintBusV19, CertifiedSwirlRawOpeningBusV19,
    CertifiedVmSegmentMetadataBusV19, DirectLogUpSegmentSourceRecordV19,
    DirectLogUpSourceEntryProfileV19, DirectLogUpSourceForestNodeAirV19,
    DirectLogUpSourceInstanceAirV19, DirectLogUpSourceManifestAirV19,
    DirectLogUpSourceManifestProfileV19, FixedSourcePublicValueBusV19,
    HistoryPoseidon2CompressBusV19, LogUpClaimDerivationAirV19, LogUpClaimDerivationRecordV19,
    LogUpClaimPhaseStartBusV19, LogUpClaimPhaseStartMessageV19, LogUpOpeningPhaseStartBusV19,
    LogUpOpeningPhaseStartMessageV19, LogUpSwirlBoundaryAirV19, LogUpSwirlBoundaryShardRecordV19,
    MappedAuxiliaryChallengeBusV19, MappedAuxiliaryChallengeMessageV19,
    MappedFunctionalChallengeBusV19, MappedFunctionalChallengeMessageV19,
    OneShotClaimTranscriptRecordV19, OneShotReductionAirV19, OneShotReductionRecordV19,
    OneShotRoundStartBusV19, OneShotRoundStartMessageV19, OneShotStreamCursorBusV19,
    OneShotStreamCursorMessageV19, SetupPcsSourceManifestBusV3, SourceForestNodeBusV19,
    SourceInstanceDigestBusV19, SourcePrefixEndBusV19, SourcePrefixEndMessageV19,
    VerifiedDirectAirPublicValueBusV19, VerifiedLogUpArithmeticBusV19,
    VerifiedLogUpArithmeticMessageV19, VerifiedMappedFunctionalBusV19,
    VerifiedMappedFunctionalMessageV19, VerifiedOneShotRawOpeningBusV19,
    VerifiedSourceForestLeafBusV19, VerifiedSourceForestLeafMessageV19, LOGUP_END_BOUNDARY_TAG_V19,
    LOGUP_ONLY_MODE_TAG_V19, LOGUP_START_BOUNDARY_TAG_V19, MAX_RAW_MESSAGE_POINT_LEN_V19,
    NATIVE_WARP_HISTORY_PROTOCOL_V19, PROGRAM_FINGERPRINT_DIGEST_TAG_V19,
    PROGRAM_FINGERPRINT_DOMAIN_TAG_V19, SEGMENT_PREFIX_TAG_V19, VM_STATE_HASH_TAG_V19,
};
use crate::circuit::verifier_warp_history_v2::{
    FixedSetupOpeningPointBusV2, FixedSetupOpeningPointMessageV2, VerifierWarpActiveCountProfileV4,
    VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2,
};

/// Protocol-v19 is intentionally specialized to OpenVM's metered execution
/// shape. `SegmentCtx` fixes this value when planning child traces. Supporting
/// another value requires a distinct, shape-specific mapped AIR/VK; silently
/// compiling a generic `l_skip <= 8` envelope makes this AIR thousands of
/// columns wider and is not production safe.
pub const OPENVM_DIRECT_LOGUP_L_SKIP_V19: usize = 4;
const MAX_L_SKIP_V19: usize = OPENVM_DIRECT_LOGUP_L_SKIP_V19;
const MAX_SKIP_DOMAIN_V19: usize = 1 << OPENVM_DIRECT_LOGUP_L_SKIP_V19;
/// Conservative peak multiplier for the main LDE, quotient/interaction work,
/// and prover scratch that coexist while this AIR is proved.
pub const MAPPED_FUNCTIONAL_LDE_WORKING_SET_MULTIPLIER_V19: usize = 4;
const DIRECT_OPENING_REDUCTION_TAG_V19: &[u8] = b"openvm-swirl-direct-opening-reduction-v1";

macro_rules! define_lookup_bus {
    ($Bus:ident, $Message:ident) => {
        #[derive(Copy, Clone, Debug)]
        pub struct $Bus(LookupBus);

        impl $Bus {
            #[must_use]
            pub fn new(bus_index: BusIndex) -> Self {
                Self(LookupBus::new(bus_index))
            }

            pub fn lookup_key<AB: InteractionBuilder>(
                &self,
                builder: &mut AB,
                message: $Message<impl Into<AB::Expr> + Clone>,
                enabled: impl Into<AB::Expr>,
            ) {
                self.0.lookup_key(builder, message.to_vec(), enabled);
            }

            pub fn add_key_with_lookups<AB: InteractionBuilder>(
                &self,
                builder: &mut AB,
                message: $Message<impl Into<AB::Expr> + Clone>,
                count: impl Into<AB::Expr>,
            ) {
                self.0
                    .add_key_with_lookups(builder, message.to_vec(), count);
            }
        }
    };
}

/// Composite-owned output buses exported to the enclosing History circuit.
/// The compression bus aliases the partial verifier's sole Poseidon table;
/// History must not allocate a second physical compression bus.
#[derive(Clone, Copy, Debug)]
pub struct DirectLogUpOnlyHistoryBusesV19 {
    pub endpoint: CertifiedLogUpOnlyEndpointBusV19,
    pub opening: CertifiedSwirlRawOpeningBusV19,
    pub vm_metadata: CertifiedVmSegmentMetadataBusV19,
    pub program_fingerprint: CertifiedProgramFingerprintBusV19,
    pub compress: HistoryPoseidon2CompressBusV19,
}

/// Authenticated internal outputs consumed by the fixed multi-AIR source
/// boundary instead of the legacy per-AIR boundary.
#[derive(Clone, Copy, Debug)]
pub struct FixedMultiAirBoundaryInputsV19 {
    pub arithmetic: VerifiedLogUpArithmeticBusV19,
    pub source_leaf: VerifiedSourceForestLeafBusV19,
    pub raw_opening: VerifiedOneShotRawOpeningBusV19,
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct SegmentOpeningPointMessageV19<T> {
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub index: T,
    pub value: [T; D_EF],
}
define_lookup_bus!(SegmentOpeningPointBusV19, SegmentOpeningPointMessageV19);

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct PrefixCheckpointBridgeColsV19<T> {
    pub active: T,
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub end_tidx: T,
    pub sample_count: T,
    pub state: [T; POSEIDON2_WIDTH],
}

/// Joins the manifest's exact end cursor to the state emitted by the real
/// prefix Transcript AIR.  Both checkpoint kinds are consumed at the same
/// row-aligned terminal sample so no unused transcript authority remains.
#[derive(Clone, ColumnsAir)]
#[columns_via(PrefixCheckpointBridgeColsV19<u8>)]
pub struct PrefixCheckpointBridgeAirV19 {
    /// Absolute execution-segment index represented by local proof slot zero.
    pub segment_start: u32,
    pub source_end_bus: SourcePrefixEndBusV19,
    pub checkpoint_bus: CertifiedTranscriptCheckpointBus,
    pub rebased_start_bus: RebasedTranscriptStartBus,
}

impl BaseAir<F> for PrefixCheckpointBridgeAirV19 {
    fn width(&self) -> usize {
        PrefixCheckpointBridgeColsV19::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for PrefixCheckpointBridgeAirV19 {}
impl PartitionedBaseAir<F> for PrefixCheckpointBridgeAirV19 {}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for PrefixCheckpointBridgeAirV19 {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("prefix checkpoint bridge row");
        let local: &PrefixCheckpointBridgeColsV19<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.when(local.active).assert_eq(
            AB::Expr::from(local.proof_index) + AB::Expr::from_u32(self.segment_start),
            AB::Expr::from(local.segment_index_lo)
                + AB::Expr::from_u32(1 << 16) * AB::Expr::from(local.segment_index_hi),
        );
        builder
            .when(local.active)
            .assert_eq(local.sample_count, AB::Expr::from_usize(D_EF));
        self.source_end_bus.receive(
            builder,
            SourcePrefixEndMessageV19 {
                proof_index: local.proof_index.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                end_tidx: local.end_tidx.into(),
            },
            local.active,
        );
        for kind in 0..2 {
            self.checkpoint_bus.receive(
                builder,
                local.proof_index,
                CertifiedTranscriptCheckpointMessage {
                    kind: AB::Expr::from_usize(kind),
                    tidx: local.end_tidx.into(),
                    sample_count: local.sample_count.into(),
                    state: local.state.map(Into::into),
                },
                local.active,
            );
        }
        self.rebased_start_bus.send(
            builder,
            local.proof_index,
            RebasedTranscriptStartMessage {
                tidx: local.end_tidx.into(),
                state: local.state.map(Into::into),
            },
            local.active,
        );
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PrefixCheckpointBridgeRecordV19 {
    pub proof_index: u32,
    pub segment_index: u32,
    pub end_tidx: u32,
    pub state: [F; POSEIDON2_WIDTH],
}

pub fn generate_prefix_checkpoint_bridge_trace_v19(
    records: &[PrefixCheckpointBridgeRecordV19],
) -> Result<RowMajorMatrix<F>, &'static str> {
    if records.is_empty()
        || records
            .iter()
            .enumerate()
            .any(|(index, record)| record.proof_index as usize != index || record.end_tidx == 0)
    {
        return Err("invalid prefix checkpoint bridge records");
    }
    let width = PrefixCheckpointBridgeColsV19::<F>::width();
    let height = records.len().next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    for (index, record) in records.iter().enumerate() {
        let cols: &mut PrefixCheckpointBridgeColsV19<F> =
            values[index * width..(index + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.proof_index = F::from_u32(record.proof_index);
        cols.segment_index_lo = F::from_u32(record.segment_index & 0xffff);
        cols.segment_index_hi = F::from_u32(record.segment_index >> 16);
        cols.end_tidx = F::from_u32(record.end_tidx);
        cols.sample_count = F::from_usize(D_EF);
        cols.state = record.state;
    }
    Ok(RowMajorMatrix::new(values, width))
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct OpeningPointFanoutColsV19<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub index: T,
    pub consumer_count: T,
    pub value: [T; D_EF],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
struct OpeningPointFanoutPrepColsV19<T> {
    active: T,
    proof_index: T,
    point_index: T,
    fixed_setup_consumer_count: T,
}

#[derive(Clone, ColumnsAir)]
#[columns_via(OpeningPointFanoutColsV19<u8>)]
pub struct OpeningPointFanoutAirV19 {
    pub segment_start: u32,
    pub source_bus: LogUpOnlyOpeningPointBus,
    pub lookup_bus: SegmentOpeningPointBusV19,
    pub fixed_setup_point_bus: Option<FixedSetupOpeningPointBusV2>,
    /// Canonical `(proof_index, point_index, multiplicity)` rows aligned with
    /// every row of the complete source opening-point trace. Coordinates not
    /// consumed by the setup projection carry multiplicity zero. Empty
    /// preserves legacy v19 mode.
    pub fixed_setup_point_demands: Arc<[(u32, u32, u32)]>,
}

impl BaseAir<F> for OpeningPointFanoutAirV19 {
    fn width(&self) -> usize {
        OpeningPointFanoutColsV19::<F>::width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        if self.fixed_setup_point_bus.is_none() {
            return None;
        }
        let width = OpeningPointFanoutPrepColsV19::<u8>::width();
        let height = self
            .fixed_setup_point_demands
            .len()
            .next_power_of_two()
            .max(2);
        let mut values = F::zero_vec(width * height);
        for (row, &(proof_index, point_index, count)) in
            self.fixed_setup_point_demands.iter().enumerate()
        {
            let cols: &mut OpeningPointFanoutPrepColsV19<F> =
                values[row * width..(row + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.proof_index = F::from_u32(proof_index);
            cols.point_index = F::from_u32(point_index);
            cols.fixed_setup_consumer_count = F::from_u32(count);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}
impl BaseAirWithPublicValues<F> for OpeningPointFanoutAirV19 {}
impl PartitionedBaseAir<F> for OpeningPointFanoutAirV19 {}

impl<AB: AirBuilder<F = F> + InteractionBuilder + PairBuilder> Air<AB>
    for OpeningPointFanoutAirV19
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("opening-point fanout row");
        let next_row = main.row_slice(1).expect("opening-point fanout next row");
        let local: &OpeningPointFanoutColsV19<AB::Var> = (*local_row).borrow();
        let next: &OpeningPointFanoutColsV19<AB::Var> = (*next_row).borrow();
        for bit in [local.active, local.is_first, local.is_last] {
            builder.assert_bool(bit);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when(local.active).assert_eq(
            AB::Expr::from(local.proof_index) + AB::Expr::from_u32(self.segment_start),
            AB::Expr::from(local.segment_index_lo)
                + AB::Expr::from_u32(1 << 16) * AB::Expr::from(local.segment_index_hi),
        );
        let same = AB::Expr::from(next.active) * (AB::Expr::ONE - AB::Expr::from(local.is_last));
        let mut transition_builder = builder.when_transition();
        let mut transition = transition_builder.when(same);
        transition.assert_eq(next.proof_index, local.proof_index);
        transition.assert_eq(next.segment_index_lo, local.segment_index_lo);
        transition.assert_eq(next.segment_index_hi, local.segment_index_hi);
        transition.assert_eq(next.index, AB::Expr::from(local.index) + AB::Expr::ONE);
        let enabled = AB::Expr::from(local.active);
        self.source_bus.receive(
            builder,
            local.proof_index,
            LogUpOnlyOpeningPointMessage {
                index: local.index.into(),
                value: local.value.map(Into::into),
            },
            enabled.clone(),
        );
        self.lookup_bus.add_key_with_lookups(
            builder,
            SegmentOpeningPointMessageV19 {
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                index: local.index.into(),
                value: local.value.map(Into::into),
            },
            enabled * AB::Expr::from(local.consumer_count),
        );
        if let Some(fixed_bus) = self.fixed_setup_point_bus {
            let preprocessed = builder.preprocessed();
            let prep_row = preprocessed
                .row_slice(0)
                .expect("fixed setup point demand row");
            let prep: &OpeningPointFanoutPrepColsV19<AB::Var> = (*prep_row).borrow();
            builder.assert_bool(prep.active);
            // `prep.active` marks rows belonging to the setup-fixed maximum
            // capacity. `local.active` is the runtime active prefix and may
            // therefore be zero for an inactive fixed slot. It may never be
            // active outside the fixed table.
            builder
                .when(AB::Expr::ONE - AB::Expr::from(prep.active))
                .assert_zero(local.active);
            builder
                .when_transition()
                .when(AB::Expr::ONE - AB::Expr::from(local.active))
                .assert_zero(next.active);
            builder
                .when(local.active)
                .assert_eq(local.proof_index, prep.proof_index);
            builder
                .when(local.active)
                .assert_eq(local.index, prep.point_index);
            fixed_bus.add_key_with_lookups(
                builder,
                FixedSetupOpeningPointMessageV2 {
                    proof_index: local.proof_index.into(),
                    point_index: local.index.into(),
                    value: local.value.map(Into::into),
                },
                AB::Expr::from(local.active) * AB::Expr::from(prep.fixed_setup_consumer_count),
            );
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OpeningPointFanoutRecordV19 {
    pub proof_index: u32,
    pub segment_index: u32,
    pub values: Vec<EF>,
    /// Exact fixed number of downstream lookups for each coordinate.
    pub consumer_counts: Vec<u32>,
}

pub fn generate_opening_point_fanout_trace_v19(
    records: &[OpeningPointFanoutRecordV19],
) -> Result<RowMajorMatrix<F>, &'static str> {
    let rows = records
        .iter()
        .map(|record| record.values.len())
        .sum::<usize>();
    if rows == 0 {
        return Err("empty opening-point fanout");
    }
    let width = OpeningPointFanoutColsV19::<F>::width();
    let height = rows.next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    let mut row = 0;
    for (segment_ordinal, record) in records.iter().enumerate() {
        if record.proof_index as usize != segment_ordinal
            || record.values.is_empty()
            || record.values.len() != record.consumer_counts.len()
            || record.consumer_counts.contains(&0)
        {
            return Err("invalid opening-point fanout record");
        }
        for (index, (&value, &consumer_count)) in record
            .values
            .iter()
            .zip(&record.consumer_counts)
            .enumerate()
        {
            let cols: &mut OpeningPointFanoutColsV19<F> =
                values[row * width..(row + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.is_first = F::from_bool(index == 0);
            cols.is_last = F::from_bool(index + 1 == record.values.len());
            cols.proof_index = F::from_u32(record.proof_index);
            cols.segment_index_lo = F::from_u32(record.segment_index & 0xffff);
            cols.segment_index_hi = F::from_u32(record.segment_index >> 16);
            cols.index = F::from_usize(index);
            cols.consumer_count = F::from_u32(consumer_count);
            cols.value
                .copy_from_slice(value.as_basis_coefficients_slice());
            row += 1;
        }
    }
    Ok(RowMajorMatrix::new(values, width))
}

/// Fixed-capacity counterpart of [`generate_opening_point_fanout_trace_v19`].
///
/// The preprocessed demand table belongs to the verifier key and therefore
/// has the maximum protocol height even when the current HLeaf contains fewer
/// active transitions.  CUDA and CPU AIR evaluation both require the main and
/// preprocessed matrices to use that same row domain.  Inactive suffix slots
/// are represented by canonical zero main rows and are gated by
/// `local.active`; they are not compacted away at runtime.
pub fn generate_opening_point_fanout_trace_fixed_capacity_v4(
    air: &OpeningPointFanoutAirV19,
    records: &[OpeningPointFanoutRecordV19],
) -> Result<RowMajorMatrix<F>, &'static str> {
    if air.fixed_setup_point_bus.is_none() || air.fixed_setup_point_demands.is_empty() {
        return Err("missing fixed setup opening-point schedule");
    }
    let mut trace = generate_opening_point_fanout_trace_v19(records)?;
    let target_height = air
        .fixed_setup_point_demands
        .len()
        .next_power_of_two()
        .max(2);
    if trace.height() > target_height {
        return Err("active opening-point fanout exceeds fixed capacity");
    }
    let width = trace.width();
    trace.values.resize(
        width
            .checked_mul(target_height)
            .ok_or("fixed opening-point fanout size overflow")?,
        F::ZERO,
    );
    Ok(RowMajorMatrix::new(trace.values, width))
}

/// Expand the setup authority's projected-prefix demand table to the exact
/// row schedule of the complete LogUp opening point. The main and
/// preprocessed matrices are zipped row-wise by the AIR, so compacting away
/// unconsumed suffix coordinates would pair the next proof's VK row with the
/// previous proof's witness row.
fn align_fixed_setup_point_demands_v19(
    point_lengths: &[usize],
    demands: &[(u32, u32, u32)],
) -> Result<Vec<(u32, u32, u32)>, &'static str> {
    if point_lengths.is_empty() || point_lengths.contains(&0) {
        return Err("invalid fixed setup opening-point lengths");
    }
    let mut demand_index = 0usize;
    let total = point_lengths.iter().try_fold(0usize, |sum, &len| {
        sum.checked_add(len)
            .ok_or("fixed setup opening-point demand overflow")
    })?;
    let mut aligned = Vec::with_capacity(total);
    for (proof_index, &point_len) in point_lengths.iter().enumerate() {
        let proof_index = u32::try_from(proof_index)
            .map_err(|_| "fixed setup opening-point proof index overflow")?;
        for point_index in 0..point_len {
            let point_index = u32::try_from(point_index)
                .map_err(|_| "fixed setup opening-point coordinate overflow")?;
            let count = demands
                .get(demand_index)
                .filter(|&&(proof, point, _)| proof == proof_index && point == point_index)
                .map_or(0, |&(_, _, count)| {
                    demand_index += 1;
                    count
                });
            aligned.push((proof_index, point_index, count));
        }
    }
    if demand_index != demands.len() {
        return Err("fixed setup opening-point demand outside source point");
    }
    Ok(aligned)
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ForestLeafFanoutColsV19<T> {
    pub active: T,
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub shard_ordinal: T,
    pub source_air_id: T,
    pub source_log_height: T,
    pub source_cached_width: T,
    pub air_id: T,
    pub relation_digest: [T; DIGEST_SIZE],
    pub log_height: T,
    pub cached_width: T,
    pub log_message_len: T,
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub source_root: [T; DIGEST_SIZE],
    pub range_start: T,
    pub range_end: T,
}

#[derive(Clone, ColumnsAir)]
#[columns_via(ForestLeafFanoutColsV19<u8>)]
pub struct ForestLeafFanoutAirV19 {
    pub source_bus: VerifiedSourceForestLeafBusV19,
    pub boundary_bus: VerifiedSourceForestLeafBusV19,
    pub mapped_bus: VerifiedSourceForestLeafBusV19,
}

impl BaseAir<F> for ForestLeafFanoutAirV19 {
    fn width(&self) -> usize {
        ForestLeafFanoutColsV19::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for ForestLeafFanoutAirV19 {}
impl PartitionedBaseAir<F> for ForestLeafFanoutAirV19 {}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for ForestLeafFanoutAirV19 {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("forest fanout row");
        let local: &ForestLeafFanoutColsV19<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        let message = |air_id, log_height, cached_width| VerifiedSourceForestLeafMessageV19 {
            proof_index: local.proof_index.into(),
            segment_index_lo: local.segment_index_lo.into(),
            segment_index_hi: local.segment_index_hi.into(),
            shard_ordinal: local.shard_ordinal.into(),
            air_id,
            relation_digest: local.relation_digest.map(Into::into),
            log_height,
            cached_width,
            log_message_len: local.log_message_len.into(),
            source_forest_root: local.source_forest_root.map(Into::into),
            segment_openings_digest: local.segment_openings_digest.map(Into::into),
            source_root: local.source_root.map(Into::into),
            range_start: local.range_start.into(),
            range_end: local.range_end.into(),
        };
        let source_message = || {
            message(
                local.source_air_id.into(),
                local.source_log_height.into(),
                local.source_cached_width.into(),
            )
        };
        let mapped_message = || {
            message(
                local.air_id.into(),
                local.log_height.into(),
                local.cached_width.into(),
            )
        };
        self.source_bus
            .receive(builder, source_message(), local.active);
        self.boundary_bus
            .send(builder, source_message(), local.active);
        self.mapped_bus
            .send(builder, mapped_message(), local.active);
    }
}

pub fn generate_forest_leaf_fanout_trace_v19(
    records: &[LogUpSwirlBoundaryShardRecordV19],
    fixed_multi_air: bool,
) -> Result<RowMajorMatrix<F>, &'static str> {
    if records.is_empty() {
        return Err("empty forest fanout");
    }
    let width = ForestLeafFanoutColsV19::<F>::width();
    let height = records.len().next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    for (row, record) in records.iter().enumerate() {
        let cols: &mut ForestLeafFanoutColsV19<F> =
            values[row * width..(row + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.proof_index = F::from_u32(record.proof_index);
        cols.segment_index_lo = F::from_u32(record.segment_index & 0xffff);
        cols.segment_index_hi = F::from_u32(record.segment_index >> 16);
        cols.shard_ordinal = F::from_u16(record.shard_ordinal);
        cols.source_air_id = F::from_u32(if fixed_multi_air {
            u32::MAX
        } else {
            record.air_id
        });
        cols.source_log_height = F::from_u8(if fixed_multi_air {
            record.log_message_len
        } else {
            record.log_height
        });
        cols.source_cached_width = F::from_u32(if fixed_multi_air {
            0
        } else {
            record.cached_width
        });
        cols.air_id = F::from_u32(record.air_id);
        cols.relation_digest = record.relation_digest;
        cols.log_height = F::from_u8(record.log_height);
        cols.cached_width = F::from_u32(record.cached_width);
        cols.log_message_len = F::from_u8(record.log_message_len);
        cols.source_forest_root = record.source_forest_root;
        cols.segment_openings_digest = record.segment_openings_digest;
        cols.source_root = record.source_root;
        cols.range_start = F::from_u32(record.range_start);
        cols.range_end = F::from_u32(record.range_end);
    }
    Ok(RowMajorMatrix::new(values, width))
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct LogUpEndpointBridgeColsV19<T> {
    pub active: T,
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub end_tidx: T,
    pub app_vk_digest: [T; DIGEST_SIZE],
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub verifier_endpoint: [T; D_EF],
    pub shard_count: T,
    pub checkpoint_state: [T; POSEIDON2_WIDTH],
}

/// Converts only the real partial-verifier endpoint into the v19 arithmetic
/// statement and starts claim derivation at the exact consumed cursor.
#[derive(Clone, ColumnsAir)]
#[columns_via(LogUpEndpointBridgeColsV19<u8>)]
pub struct LogUpEndpointBridgeAirV19 {
    pub segment_start: u32,
    pub endpoint_bus: BatchConstraintEndpointBus,
    pub transcript_end_index_bus: TranscriptEndIndexBus,
    pub checkpoint_bus: CertifiedTranscriptCheckpointBus,
    pub arithmetic_bus: VerifiedLogUpArithmeticBusV19,
    pub opening_phase_start_bus: LogUpOpeningPhaseStartBusV19,
}

impl BaseAir<F> for LogUpEndpointBridgeAirV19 {
    fn width(&self) -> usize {
        LogUpEndpointBridgeColsV19::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for LogUpEndpointBridgeAirV19 {}
impl PartitionedBaseAir<F> for LogUpEndpointBridgeAirV19 {}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for LogUpEndpointBridgeAirV19 {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("LogUp endpoint bridge row");
        let local: &LogUpEndpointBridgeColsV19<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.when(local.active).assert_eq(
            AB::Expr::from(local.proof_index) + AB::Expr::from_u32(self.segment_start),
            AB::Expr::from(local.segment_index_lo)
                + AB::Expr::from_u32(1 << 16) * AB::Expr::from(local.segment_index_hi),
        );
        self.endpoint_bus.receive(
            builder,
            local.proof_index,
            BatchConstraintEndpointMessage {
                tidx: local.end_tidx.into(),
                final_claim: local.verifier_endpoint.map(Into::into),
            },
            local.active,
        );
        // The resumed Transcript AIR now spans the appended SWIRL reductions,
        // so its ordinary end-index message names the later final cursor. A
        // certified kind-zero checkpoint proves that this intermediate cursor
        // is an actual squeeze-row boundary, and we re-emit precisely that
        // index for PartialBatchConstraintEndpointAir.
        let checkpoint = || CertifiedTranscriptCheckpointMessage {
            kind: AB::Expr::ZERO,
            tidx: local.end_tidx.into(),
            sample_count: AB::Expr::from_usize(D_EF),
            state: local.checkpoint_state.map(Into::into),
        };
        self.checkpoint_bus
            .receive(builder, local.proof_index, checkpoint(), local.active);
        // Forward only this verified partial-verifier endpoint to the History
        // producer. The prefix bridge has its own earlier checkpoint and does
        // not self-relay it on this bus.
        self.checkpoint_bus
            .send(builder, local.proof_index, checkpoint(), local.active);
        self.transcript_end_index_bus.send(
            builder,
            local.proof_index,
            TranscriptEndIndexMessage {
                tidx: local.end_tidx.into(),
            },
            local.active,
        );
        self.arithmetic_bus.send(
            builder,
            VerifiedLogUpArithmeticMessageV19 {
                proof_index: local.proof_index.into(),
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                mode_tag: AB::Expr::from_u32(LOGUP_ONLY_MODE_TAG_V19),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                app_vk_digest: local.app_vk_digest.map(Into::into),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                verifier_endpoint: local.verifier_endpoint.map(Into::into),
                segment_sum_before: core::array::from_fn(|_| AB::Expr::ZERO),
                segment_sum_after: core::array::from_fn(|_| AB::Expr::ZERO),
                shard_count: local.shard_count.into(),
            },
            local.active,
        );
        self.opening_phase_start_bus.send(
            builder,
            LogUpOpeningPhaseStartMessageV19 {
                proof_index: local.proof_index.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                opening_start_tidx: local.end_tidx.into(),
            },
            local.active,
        );
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogUpEndpointBridgeRecordV19 {
    pub proof_index: u32,
    pub segment_index: u32,
    pub end_tidx: u32,
    pub app_vk_digest: [F; DIGEST_SIZE],
    pub source_forest_root: [F; DIGEST_SIZE],
    pub segment_openings_digest: [F; DIGEST_SIZE],
    pub verifier_endpoint: EF,
    pub shard_count: u16,
    pub checkpoint_state: [F; POSEIDON2_WIDTH],
}

pub fn generate_logup_endpoint_bridge_trace_v19(
    records: &[LogUpEndpointBridgeRecordV19],
) -> Result<RowMajorMatrix<F>, &'static str> {
    if records.is_empty()
        || records
            .iter()
            .enumerate()
            .any(|(index, record)| record.proof_index as usize != index || record.shard_count == 0)
    {
        return Err("invalid LogUp endpoint bridge records");
    }
    let width = LogUpEndpointBridgeColsV19::<F>::width();
    let height = records.len().next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    for (row, record) in records.iter().enumerate() {
        let cols: &mut LogUpEndpointBridgeColsV19<F> =
            values[row * width..(row + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.proof_index = F::from_u32(record.proof_index);
        cols.segment_index_lo = F::from_u32(record.segment_index & 0xffff);
        cols.segment_index_hi = F::from_u32(record.segment_index >> 16);
        cols.end_tidx = F::from_u32(record.end_tidx);
        cols.app_vk_digest = record.app_vk_digest;
        cols.source_forest_root = record.source_forest_root;
        cols.segment_openings_digest = record.segment_openings_digest;
        cols.verifier_endpoint
            .copy_from_slice(record.verifier_endpoint.as_basis_coefficients_slice());
        cols.shard_count = F::from_u16(record.shard_count);
        cols.checkpoint_state = record.checkpoint_state;
    }
    Ok(RowMajorMatrix::new(values, width))
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct LogUpFinalizationColsV19<T> {
    pub active: T,
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub shard_count: T,
    pub start_tidx: T,
    pub end_tidx: T,
    pub sampled: [T; D_EF],
    pub checkpoint_state: [T; POSEIDON2_WIDTH],
}

/// Closes the ordered one-shot stream and certifies the exact transcript state
/// handed to History. This is the circuit counterpart of
/// `finish_v19_segment_logup_transcript`; no host-selected final checkpoint is
/// accepted without consuming the final stream cursor and Transcript AIR.
#[derive(Clone, ColumnsAir)]
#[columns_via(LogUpFinalizationColsV19<u8>)]
pub struct LogUpFinalizationAirV19 {
    pub segment_start: u32,
    pub transcript_bus: TranscriptBus,
    pub transcript_end_index_bus: TranscriptEndIndexBus,
    pub checkpoint_bus: CertifiedTranscriptCheckpointBus,
    pub stream_cursor_bus: OneShotStreamCursorBusV19,
}

impl BaseAir<F> for LogUpFinalizationAirV19 {
    fn width(&self) -> usize {
        LogUpFinalizationColsV19::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for LogUpFinalizationAirV19 {}
impl PartitionedBaseAir<F> for LogUpFinalizationAirV19 {}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for LogUpFinalizationAirV19 {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("LogUp finalization row");
        let local: &LogUpFinalizationColsV19<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.when(local.active).assert_eq(
            AB::Expr::from(local.proof_index) + AB::Expr::from_u32(self.segment_start),
            AB::Expr::from(local.segment_index_lo)
                + AB::Expr::from_u32(1 << 16) * AB::Expr::from(local.segment_index_hi),
        );
        builder.when(local.active).assert_eq(
            local.end_tidx,
            AB::Expr::from(local.start_tidx) + AB::Expr::from_usize(1 + D_EF),
        );
        self.stream_cursor_bus.receive(
            builder,
            OneShotStreamCursorMessageV19 {
                proof_index: local.proof_index.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                next_shard_ordinal: local.shard_count.into(),
                tidx: local.start_tidx.into(),
            },
            local.active,
        );
        self.transcript_bus.observe(
            builder,
            local.proof_index,
            local.start_tidx,
            AB::Expr::from_u64(LOGUP_END_BOUNDARY_TAG_V19),
            local.active,
        );
        self.transcript_bus.sample_ext(
            builder,
            local.proof_index,
            AB::Expr::from(local.start_tidx) + AB::Expr::ONE,
            local.sampled,
            local.active,
        );
        self.transcript_end_index_bus.receive(
            builder,
            local.proof_index,
            TranscriptEndIndexMessage {
                tidx: local.end_tidx.into(),
            },
            local.active,
        );
        let checkpoint = || CertifiedTranscriptCheckpointMessage {
            kind: AB::Expr::ONE,
            tidx: local.end_tidx.into(),
            sample_count: AB::Expr::from_usize(D_EF),
            state: local.checkpoint_state.map(Into::into),
        };
        self.checkpoint_bus
            .receive(builder, local.proof_index, checkpoint(), local.active);
        // Forward the now stream-certified end checkpoint to the History
        // LogUp producer.
        self.checkpoint_bus
            .send(builder, local.proof_index, checkpoint(), local.active);
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogUpFinalizationRecordV19 {
    pub proof_index: u32,
    pub segment_index: u32,
    pub shard_count: u16,
    pub start_tidx: u32,
    pub end_tidx: u32,
    pub sampled: EF,
    pub checkpoint_state: [F; POSEIDON2_WIDTH],
}

pub fn generate_logup_finalization_trace_v19(
    records: &[LogUpFinalizationRecordV19],
) -> Result<RowMajorMatrix<F>, &'static str> {
    if records.is_empty()
        || records.iter().enumerate().any(|(index, record)| {
            record.proof_index as usize != index
                || record.shard_count == 0
                || record.end_tidx != record.start_tidx + 1 + D_EF as u32
        })
    {
        return Err("invalid LogUp finalization records");
    }
    let width = LogUpFinalizationColsV19::<F>::width();
    let height = records.len().next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    for (row, record) in records.iter().enumerate() {
        let cols: &mut LogUpFinalizationColsV19<F> =
            values[row * width..(row + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.proof_index = F::from_u32(record.proof_index);
        cols.segment_index_lo = F::from_u32(record.segment_index & 0xffff);
        cols.segment_index_hi = F::from_u32(record.segment_index >> 16);
        cols.shard_count = F::from_u16(record.shard_count);
        cols.start_tidx = F::from_u32(record.start_tidx);
        cols.end_tidx = F::from_u32(record.end_tidx);
        cols.sampled
            .copy_from_slice(record.sampled.as_basis_coefficients_slice());
        cols.checkpoint_state = record.checkpoint_state;
    }
    Ok(RowMajorMatrix::new(values, width))
}

#[derive(Clone)]
pub struct DirectLogUpMappedRegistryEntryV19 {
    pub shard_id: u32,
    pub relation: Arc<DirectAirPesatIndex<F, [F; DIGEST_SIZE]>>,
}

/// Setup-owned placement of one ordinary verifier AIR inside the single
/// fixed multi-AIR WARP message.
#[derive(Clone)]
pub struct FixedMultiAirMappedRegionV19 {
    pub air_id: u32,
    pub message_start: u32,
    pub relation: Arc<DirectAirPesatIndex<F, [F; DIGEST_SIZE]>>,
}

/// Setup-owned global source layout used by verifier-WARP History.  One
/// logical source exists per capacity-four verifier execution; all AIR
/// openings and the occupancy term refer to this same relation/root.
#[derive(Clone)]
pub struct FixedMultiAirMappedFunctionalConfigV19 {
    pub relation_digest: [F; DIGEST_SIZE],
    pub log_message_len: u8,
    pub log_codeword_len: u8,
    pub regions: Arc<[FixedMultiAirMappedRegionV19]>,
    pub vm_pvs_air_id: u32,
    pub is_valid_common_main_column: u32,
    pub is_valid_message_block_start: u32,
    pub vm_pvs_log_height: u8,
    pub active_child_counts: Arc<[u8]>,
    pub profile_trace_heights: Arc<[u32]>,
    /// In the production setup-PCS authority path, cached-main openings are
    /// authenticated independently under the retained setup commitment.  The
    /// mapped source must publish one additional copy for the authority bridge
    /// to consume, thereby equating both claims on the canonical typed bus.
    pub authenticate_cached_setup_claims: bool,
}

/// Fixed-capacity counterpart of [`FixedMultiAirMappedFunctionalConfigV19`].
/// It contains only setup data; four local source slots are always planned,
/// while interval occupancy and per-source active counts are main-trace data.
#[derive(Clone)]
pub struct FixedMultiAirMappedFunctionalConfigV4 {
    pub relation_digest: [F; DIGEST_SIZE],
    /// Digest of the complete block-wide fixed verifier profile.  Every
    /// local HLeaf slot uses this same setup-owned identity; the global batch
    /// index and runtime occupancy remain witness values constrained by the
    /// History layer.
    pub active_count_profile_digest: [F; DIGEST_SIZE],
    pub log_message_len: u8,
    pub log_codeword_len: u8,
    /// Number of source public values, excluding PESAT's distinguished
    /// constant explicit coordinate zero. This is setup identity and is never
    /// supplied by the runtime occupancy witness.
    pub source_public_values_len: u32,
    pub regions: Arc<[FixedMultiAirMappedRegionV19]>,
    pub vm_pvs_air_id: u32,
    pub is_valid_common_main_column: u32,
    pub is_valid_message_block_start: u32,
    pub vm_pvs_log_height: u8,
    pub profile_trace_heights: Arc<[u32]>,
    pub authenticate_cached_setup_claims: bool,
}

#[derive(Clone, Debug)]
struct MappedTermPlanV19 {
    proof_index: u32,
    segment_index: u32,
    shard_ordinal: u16,
    shard_count: u16,
    shard_id: u32,
    air_id: u32,
    relation_digest: [F; DIGEST_SIZE],
    log_height: u8,
    cached_width: u32,
    log_message_len: u8,
    sort_idx: u32,
    part_idx: u32,
    col_idx: u32,
    is_rot: bool,
    is_program: bool,
    is_program_term: bool,
    /// Setup-fixed auxiliary term over the same systematic message.  This is
    /// currently the canonical active-child count and is never sourced from
    /// `ColumnClaimsBus`.
    is_active_count_term: bool,
    /// Setup-fixed cached-main term whose source claim must be equated to the
    /// independently PCS-authenticated setup claim.
    authenticate_cached_setup_claim: bool,
    active_count_expected: u8,
    block_start: u32,
    l_skip: u8,
    term_ordinal: u32,
    term_count: u32,
    observation_start: u32,
    observation_count: u32,
    program_row_point: Vec<EF>,
    program_column_challenge: EF,
}

/// Fixed mapped-functional layout derived from the child VK and the exact
/// homogeneous registry/segment plan.
#[derive(Clone)]
pub struct DirectLogUpMappedFunctionalProfileV19 {
    pub app_vk_digest: [F; DIGEST_SIZE],
    pub registry_digest: [F; DIGEST_SIZE],
    pub l_skip: usize,
    segment_start: u32,
    terms: Arc<[MappedTermPlanV19]>,
    metrics: MappedFunctionalTraceMetricsV19,
    runtime_capacity_v4: bool,
}

impl DirectLogUpMappedFunctionalProfileV19 {
    /// Build the selector-free fixed multi-AIR source profile.  Dynamic
    /// openings retain the recursive verifier's canonical trace order while
    /// their message blocks are shifted into one global systematic message.
    /// The final term is the setup-fixed `VmPvsCols::is_valid` sum.
    pub fn new_fixed_multi_air(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        source_profile: &DirectLogUpSourceManifestProfileV19,
        config: &FixedMultiAirMappedFunctionalConfigV19,
        segment_start: u32,
    ) -> Result<Self, &'static str> {
        let l_skip = child_vk.inner.params.l_skip;
        if l_skip != OPENVM_DIRECT_LOGUP_L_SKIP_V19
            || config.regions.is_empty()
            || config.active_child_counts.is_empty()
            || config.profile_trace_heights.len() != child_vk.inner.per_air.len()
            || source_profile.source_count() != config.active_child_counts.len()
            || source_profile.segment_start() != segment_start
            || config.log_message_len == 0
            || config.log_codeword_len < config.log_message_len
            || config.regions.iter().any(|region| {
                region.air_id as usize >= child_vk.inner.per_air.len()
                    || region.relation.description().shard_key.air_id != region.air_id
                    || config.profile_trace_heights[region.air_id as usize]
                        != (1u32 << region.relation.description().shard_key.log_height)
            })
            || config
                .profile_trace_heights
                .iter()
                .any(|&height| height == 0)
        {
            return Err("invalid fixed multi-AIR mapped profile");
        }
        for (index, &count) in config.active_child_counts.iter().enumerate() {
            if count == 0
                || usize::from(count) > VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2
                || (index + 1 != config.active_child_counts.len()
                    && usize::from(count) != VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2)
            {
                return Err("noncanonical fixed multi-AIR occupancy profile");
            }
        }
        let vm_height = 1u32
            .checked_shl(config.vm_pvs_log_height.into())
            .ok_or("fixed multi-AIR VM-PVS height")?;
        let padded_message_len = 1u32
            .checked_shl(config.log_message_len.into())
            .ok_or("fixed multi-AIR message length")?;
        if config.is_valid_message_block_start % vm_height != 0
            || config
                .is_valid_message_block_start
                .checked_add(vm_height)
                .is_none_or(|end| end > padded_message_len)
        {
            return Err("fixed multi-AIR active-count block");
        }

        let mut ordered_regions = config.regions.iter().collect::<Vec<_>>();
        ordered_regions.sort_by_key(|region| {
            let key = &region.relation.description().shard_key;
            (core::cmp::Reverse(key.log_height), region.air_id)
        });
        let mut terms = Vec::new();
        for (proof_index, &active_count_expected) in config.active_child_counts.iter().enumerate() {
            let proof_index = u32::try_from(proof_index).map_err(|_| "proof index overflow")?;
            let segment_index = segment_start
                .checked_add(proof_index)
                .ok_or("segment index overflow")?;
            let source_start = terms.len();
            for (sort_idx, region) in ordered_regions.iter().enumerate() {
                let key = &region.relation.description().shard_key;
                let vk = &child_vk.inner.per_air[region.air_id as usize];
                if key.app_vk_digest != source_profile.app_vk_digest
                    || key.trace_layout.common_main_width as usize != vk.params.width.common_main
                    || key
                        .trace_layout
                        .cached_main_widths
                        .iter()
                        .map(|&width| width as usize)
                        .collect::<Vec<_>>()
                        != vk.params.width.cached_mains
                    || region.relation.height() != 1usize << key.log_height
                {
                    return Err("fixed multi-AIR region differs from child VK");
                }
                let stride = 1 + usize::from(vk.params.need_rot);
                let height = region.relation.height();
                let common_start = height
                    * key
                        .trace_layout
                        .cached_main_widths
                        .iter()
                        .map(|&width| width as usize)
                        .sum::<usize>();
                let mut columns = Vec::new();
                for col in 0..key.trace_layout.common_main_width as usize {
                    for rot in 0..stride {
                        columns.push((0u32, col as u32, rot == 1, common_start + col * height));
                    }
                }
                let cached_part_start = 1 + usize::from(region.relation.fixed_trace().is_some());
                let mut local_start = 0usize;
                for (cached, &width) in key.trace_layout.cached_main_widths.iter().enumerate() {
                    for col in 0..width as usize {
                        for rot in 0..stride {
                            columns.push((
                                (cached_part_start + cached) as u32,
                                col as u32,
                                rot == 1,
                                local_start + col * height,
                            ));
                        }
                    }
                    local_start += height * width as usize;
                }
                for (part_idx, col_idx, is_rot, block_start) in columns {
                    terms.push(MappedTermPlanV19 {
                        proof_index,
                        segment_index,
                        shard_ordinal: 0,
                        shard_count: 1,
                        shard_id: 0,
                        air_id: region.air_id,
                        relation_digest: config.relation_digest,
                        log_height: key.log_height,
                        cached_width: key
                            .trace_layout
                            .cached_main_widths
                            .first()
                            .copied()
                            .unwrap_or(0),
                        log_message_len: config.log_message_len,
                        sort_idx: sort_idx as u32,
                        part_idx,
                        col_idx,
                        is_rot,
                        is_program: false,
                        is_program_term: false,
                        is_active_count_term: false,
                        authenticate_cached_setup_claim: config.authenticate_cached_setup_claims
                            && part_idx != 0,
                        active_count_expected: 0,
                        block_start: region
                            .message_start
                            .checked_add(block_start as u32)
                            .ok_or("fixed multi-AIR message offset")?,
                        l_skip: l_skip as u8,
                        term_ordinal: 0,
                        term_count: 0,
                        observation_start: 0,
                        observation_count: 0,
                        program_row_point: Vec::new(),
                        program_column_challenge: EF::ZERO,
                    });
                }
            }
            terms.push(MappedTermPlanV19 {
                proof_index,
                segment_index,
                shard_ordinal: 0,
                shard_count: 1,
                shard_id: 0,
                air_id: config.vm_pvs_air_id,
                relation_digest: config.relation_digest,
                log_height: config.vm_pvs_log_height,
                cached_width: 0,
                log_message_len: config.log_message_len,
                sort_idx: 0,
                part_idx: 0,
                col_idx: config.is_valid_common_main_column,
                is_rot: false,
                is_program: false,
                is_program_term: false,
                is_active_count_term: true,
                authenticate_cached_setup_claim: false,
                active_count_expected,
                block_start: config.is_valid_message_block_start,
                l_skip: 0,
                term_ordinal: 0,
                term_count: 0,
                observation_start: 0,
                observation_count: 0,
                program_row_point: Vec::new(),
                program_column_challenge: EF::ZERO,
            });
            let term_count = terms.len() - source_start;
            if term_count == 1 {
                return Err("fixed multi-AIR source has no ordinary terms");
            }
            let mut observation = (DIRECT_OPENING_REDUCTION_TAG_V19.len() + 3 * 8) as u32;
            for (ordinal, plan) in terms[source_start..].iter_mut().enumerate() {
                plan.term_ordinal = ordinal as u32;
                plan.term_count = term_count as u32;
                plan.observation_start = observation;
                let folded = usize::from(plan.log_height).saturating_sub(plan.l_skip as usize);
                observation = observation
                    .checked_add((5 * 8 + (1usize << plan.l_skip) + 8 + folded + 1) as u32)
                    .ok_or("fixed multi-AIR observation overflow")?;
            }
            let observation_count = observation
                .checked_add(1)
                .ok_or("fixed multi-AIR observation overflow")?;
            for plan in &mut terms[source_start..] {
                plan.observation_count = observation_count;
            }
        }
        let metrics =
            MappedFunctionalTraceMetricsV19::new(terms.len(), child_vk.inner.params.log_blowup)?;
        Ok(Self {
            app_vk_digest: source_profile.app_vk_digest,
            registry_digest: source_profile.registry_digest,
            l_skip,
            segment_start,
            terms: terms.into(),
            metrics,
            runtime_capacity_v4: false,
        })
    }

    /// Build a four-slot mapped-functional profile whose structural term plan
    /// is fixed once while source activation and the count-term claim are
    /// supplied by the runtime trace.
    pub fn new_fixed_capacity_v4(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        source_profile: &DirectLogUpSourceManifestProfileV19,
        config: &FixedMultiAirMappedFunctionalConfigV4,
    ) -> Result<Self, &'static str> {
        if source_profile.runtime_capacity_v4() != Some(4) || source_profile.segment_start() != 0 {
            return Err("fixed-capacity mapped profile requires the local four-slot source");
        }
        let legacy = FixedMultiAirMappedFunctionalConfigV19 {
            relation_digest: config.relation_digest,
            log_message_len: config.log_message_len,
            log_codeword_len: config.log_codeword_len,
            regions: config.regions.clone(),
            vm_pvs_air_id: config.vm_pvs_air_id,
            is_valid_common_main_column: config.is_valid_common_main_column,
            is_valid_message_block_start: config.is_valid_message_block_start,
            vm_pvs_log_height: config.vm_pvs_log_height,
            active_child_counts: Arc::from([VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 as u8; 4]),
            profile_trace_heights: config.profile_trace_heights.clone(),
            authenticate_cached_setup_claims: config.authenticate_cached_setup_claims,
        };
        let mut profile = Self::new_fixed_multi_air(child_vk, source_profile, &legacy, 0)?;
        let terms = Arc::make_mut(&mut profile.terms);
        for term in terms.iter_mut().filter(|term| term.is_active_count_term) {
            term.active_count_expected = 0;
        }
        profile.runtime_capacity_v4 = true;
        Ok(profile)
    }

    #[must_use]
    pub const fn runtime_capacity_v4(&self) -> bool {
        self.runtime_capacity_v4
    }

    pub fn new(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        source_profile: &DirectLogUpSourceManifestProfileV19,
        entries: Vec<DirectLogUpMappedRegistryEntryV19>,
        segment_shard_ids: &[Vec<u32>],
    ) -> Result<Self, &'static str> {
        Self::new_with_segment_offset(child_vk, source_profile, entries, segment_shard_ids, 0)
    }

    pub fn new_with_segment_offset(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        source_profile: &DirectLogUpSourceManifestProfileV19,
        entries: Vec<DirectLogUpMappedRegistryEntryV19>,
        segment_shard_ids: &[Vec<u32>],
        segment_start: u32,
    ) -> Result<Self, &'static str> {
        let l_skip = child_vk.inner.params.l_skip;
        if l_skip != OPENVM_DIRECT_LOGUP_L_SKIP_V19
            || entries.is_empty()
            || segment_shard_ids.is_empty()
            || source_profile.source_count()
                != segment_shard_ids.iter().map(Vec::len).sum::<usize>()
            || source_profile.segment_start() != segment_start
        {
            return Err("protocol-v19 requires child VK l_skip == 4 and a nonempty fixed plan");
        }
        for (index, entry) in entries.iter().enumerate() {
            if entry.shard_id as usize != index {
                return Err("noncanonical mapped-functional registry");
            }
            let key = &entry.relation.description().shard_key;
            let vk = child_vk
                .inner
                .per_air
                .get(key.air_id as usize)
                .ok_or("mapped AIR outside child VK")?;
            if key.app_vk_digest != source_profile.app_vk_digest
                || key.trace_layout.common_main_width as usize != vk.params.width.common_main
                || key
                    .trace_layout
                    .cached_main_widths
                    .iter()
                    .map(|&width| width as usize)
                    .collect::<Vec<_>>()
                    != vk.params.width.cached_mains
                || entry.relation.height() != 1usize << key.log_height
            {
                return Err("mapped relation differs from child VK");
            }
        }

        let mut terms = Vec::new();
        for (proof_index, shard_ids) in segment_shard_ids.iter().enumerate() {
            let proof_index =
                u32::try_from(proof_index).map_err(|_| "mapped proof index overflow")?;
            let segment_index = segment_start
                .checked_add(proof_index)
                .ok_or("mapped segment index overflow")?;
            if shard_ids.is_empty() || shard_ids.windows(2).any(|pair| pair[0] >= pair[1]) {
                return Err("noncanonical mapped segment plan");
            }
            let mut sorted = shard_ids
                .iter()
                .map(|&shard_id| {
                    let relation = entries
                        .get(shard_id as usize)
                        .ok_or("mapped shard outside registry")?;
                    let key = &relation.relation.description().shard_key;
                    Ok((shard_id, key.log_height, key.air_id))
                })
                .collect::<Result<Vec<_>, &'static str>>()?;
            sorted.sort_by_key(|&(_, log_height, air_id)| (core::cmp::Reverse(log_height), air_id));
            let sort_by_shard = sorted
                .into_iter()
                .enumerate()
                .map(|(sort, (shard, _, _))| (shard, sort as u32))
                .collect::<std::collections::BTreeMap<_, _>>();
            for (shard_ordinal, &shard_id) in shard_ids.iter().enumerate() {
                let entry = &entries[shard_id as usize];
                let key = &entry.relation.description().shard_key;
                let vk = &child_vk.inner.per_air[key.air_id as usize];
                let stride = 1 + usize::from(vk.params.need_rot);
                let height = entry.relation.height();
                let mut ordinary = Vec::new();
                let common_start = height
                    * key
                        .trace_layout
                        .cached_main_widths
                        .iter()
                        .map(|&width| width as usize)
                        .sum::<usize>();
                for col in 0..key.trace_layout.common_main_width as usize {
                    for rot in 0..stride {
                        ordinary.push((0u32, col as u32, rot == 1, common_start + col * height));
                    }
                }
                let cached_part_start = 1 + usize::from(entry.relation.fixed_trace().is_some());
                let mut block_start = 0usize;
                for (cached, &width) in key.trace_layout.cached_main_widths.iter().enumerate() {
                    for col in 0..width as usize {
                        for rot in 0..stride {
                            ordinary.push((
                                (cached_part_start + cached) as u32,
                                col as u32,
                                rot == 1,
                                block_start + col * height,
                            ));
                        }
                    }
                    block_start += height * width as usize;
                }

                let is_program = key.air_id as usize == openvm_circuit::arch::PROGRAM_AIR_ID;
                let (program_row_point, program_column_challenge) = if is_program {
                    derive_program_parameters_v19(
                        source_profile.app_vk_digest,
                        source_profile.registry_digest,
                        key.relation_digest,
                        key.log_height,
                        key.trace_layout.cached_main_widths[0],
                    )?
                } else {
                    (Vec::new(), EF::ZERO)
                };
                let program_terms = if is_program {
                    key.trace_layout.cached_main_widths[0] as usize
                } else {
                    0
                };
                let term_count = ordinary.len() + program_terms;
                if term_count == 0 {
                    return Err("mapped source has no dynamic terms");
                }
                let mut observation = (DIRECT_OPENING_REDUCTION_TAG_V19.len() + 3 * 8) as u32;
                for (term_ordinal, &(part_idx, col_idx, is_rot, start)) in
                    ordinary.iter().enumerate()
                {
                    // OpenVM permits tiny active traces (notably Connector) below
                    // the global PLE skip domain. The backend defines those as
                    // one folded row whose 2^l_skip barycentric positions wrap
                    // cyclically over the physical trace.
                    let folded = (key.log_height as usize).saturating_sub(l_skip);
                    let term_observations = 5 * 8 + (1 << l_skip) + 8 + folded + 1;
                    terms.push(MappedTermPlanV19 {
                        proof_index,
                        segment_index,
                        shard_ordinal: shard_ordinal as u16,
                        shard_count: shard_ids.len() as u16,
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
                        sort_idx: sort_by_shard[&shard_id],
                        part_idx,
                        col_idx,
                        is_rot,
                        is_program,
                        is_program_term: false,
                        is_active_count_term: false,
                        authenticate_cached_setup_claim: false,
                        active_count_expected: 0,
                        block_start: start as u32,
                        l_skip: l_skip as u8,
                        term_ordinal: term_ordinal as u32,
                        term_count: term_count as u32,
                        observation_start: observation,
                        observation_count: 0,
                        program_row_point: program_row_point.clone(),
                        program_column_challenge,
                    });
                    observation += term_observations as u32;
                }
                for col in 0..program_terms {
                    let term_ordinal = ordinary.len() + col;
                    let folded = key.log_height as usize;
                    let term_observations = 5 * 8 + 1 + 8 + folded + 1;
                    terms.push(MappedTermPlanV19 {
                        proof_index,
                        segment_index,
                        shard_ordinal: shard_ordinal as u16,
                        shard_count: shard_ids.len() as u16,
                        shard_id,
                        air_id: key.air_id,
                        relation_digest: key.relation_digest,
                        log_height: key.log_height,
                        cached_width: key.trace_layout.cached_main_widths[0],
                        log_message_len: key.code_class.log_message_len,
                        sort_idx: sort_by_shard[&shard_id],
                        part_idx: 0,
                        col_idx: col as u32,
                        is_rot: false,
                        is_program,
                        is_program_term: true,
                        is_active_count_term: false,
                        authenticate_cached_setup_claim: false,
                        active_count_expected: 0,
                        block_start: (col * height) as u32,
                        l_skip: 0,
                        term_ordinal: term_ordinal as u32,
                        term_count: term_count as u32,
                        observation_start: observation,
                        observation_count: 0,
                        program_row_point: program_row_point.clone(),
                        program_column_challenge,
                    });
                    observation += term_observations as u32;
                }
                let observation_count = observation + 1;
                for plan in terms.iter_mut().rev().take(term_count) {
                    plan.observation_count = observation_count;
                }
            }
        }
        let metrics =
            MappedFunctionalTraceMetricsV19::new(terms.len(), child_vk.inner.params.log_blowup)?;
        Ok(Self {
            app_vk_digest: source_profile.app_vk_digest,
            registry_digest: source_profile.registry_digest,
            l_skip,
            segment_start,
            terms: terms.into(),
            metrics,
            runtime_capacity_v4: false,
        })
    }

    #[must_use]
    pub const fn segment_start(&self) -> u32 {
        self.segment_start
    }

    /// Verifier-fixed trace/LDE sizing used by setup and deployment tooling.
    #[must_use]
    pub const fn metrics(&self) -> MappedFunctionalTraceMetricsV19 {
        self.metrics
    }

    /// Number of logical mapped terms in the fixed segment plan.
    #[must_use]
    pub fn term_count(&self) -> usize {
        self.terms.len()
    }
}

#[derive(Clone, Debug)]
struct ColumnOpeningObservationPlanV19 {
    proof_index: u32,
    segment_index: u32,
    is_segment_first: bool,
    is_segment_last: bool,
    observation_ordinal: u32,
    observation_count: u32,
    sort_idx: u32,
    part_idx: u32,
    col_idx: u32,
    is_rot: bool,
    source_index: usize,
    claim_index: Option<usize>,
}

fn column_opening_observation_plan_v19(
    profile: &DirectLogUpMappedFunctionalProfileV19,
) -> Result<Vec<ColumnOpeningObservationPlanV19>, &'static str> {
    #[derive(Clone)]
    struct Candidate {
        proof_index: u32,
        segment_index: u32,
        sort_idx: u32,
        part_idx: u32,
        col_idx: u32,
        is_rot: bool,
        source_index: usize,
        claim_index: usize,
    }

    let mut candidates = Vec::new();
    let mut source_index = 0usize;
    let mut row = 0usize;
    while row < profile.terms.len() {
        let first = &profile.terms[row];
        let term_count = first.term_count as usize;
        let terms = profile
            .terms
            .get(row..row + term_count)
            .ok_or("mapped source term range")?;
        let mut claim_index = 0usize;
        for term in terms {
            if !term.is_program_term && !term.is_active_count_term {
                candidates.push(Candidate {
                    proof_index: term.proof_index,
                    segment_index: term.segment_index,
                    sort_idx: term.sort_idx,
                    part_idx: term.part_idx,
                    col_idx: term.col_idx,
                    is_rot: term.is_rot,
                    source_index,
                    claim_index,
                });
                claim_index += 1;
            }
        }
        source_index += 1;
        row += term_count;
    }
    if candidates.is_empty() {
        return Err("empty dynamic column-opening plan");
    }

    // This is exactly `observe_batch_constraint_openings`: common-main
    // openings for every height-sorted trace first, then each trace's cached
    // parts. Within one column, current precedes rotation. AIRs without next
    // row access still absorb a canonical zero rotation.
    candidates.sort_by_key(|candidate| {
        (
            candidate.proof_index,
            u8::from(candidate.part_idx != 0),
            candidate.sort_idx,
            candidate.part_idx,
            candidate.col_idx,
            candidate.is_rot,
        )
    });
    let rotated = candidates
        .iter()
        .filter(|candidate| candidate.is_rot)
        .map(|candidate| {
            (
                candidate.proof_index,
                candidate.sort_idx,
                candidate.part_idx,
                candidate.col_idx,
            )
        })
        .collect::<std::collections::BTreeSet<_>>();
    let mut plans = Vec::with_capacity(candidates.len() * 2);
    for candidate in candidates {
        let key = (
            candidate.proof_index,
            candidate.sort_idx,
            candidate.part_idx,
            candidate.col_idx,
        );
        plans.push(ColumnOpeningObservationPlanV19 {
            proof_index: candidate.proof_index,
            segment_index: candidate.segment_index,
            is_segment_first: false,
            is_segment_last: false,
            observation_ordinal: 0,
            observation_count: 0,
            sort_idx: candidate.sort_idx,
            part_idx: candidate.part_idx,
            col_idx: candidate.col_idx,
            is_rot: candidate.is_rot,
            source_index: candidate.source_index,
            claim_index: Some(candidate.claim_index),
        });
        if !candidate.is_rot && !rotated.contains(&key) {
            plans.push(ColumnOpeningObservationPlanV19 {
                proof_index: candidate.proof_index,
                segment_index: candidate.segment_index,
                is_segment_first: false,
                is_segment_last: false,
                observation_ordinal: 0,
                observation_count: 0,
                sort_idx: candidate.sort_idx,
                part_idx: candidate.part_idx,
                col_idx: candidate.col_idx,
                is_rot: true,
                source_index: candidate.source_index,
                claim_index: None,
            });
        }
    }
    let mut start = 0usize;
    while start < plans.len() {
        let proof_index = plans[start].proof_index;
        let end = plans[start..]
            .iter()
            .position(|plan| plan.proof_index != proof_index)
            .map_or(plans.len(), |offset| start + offset);
        let count = u32::try_from(end - start).map_err(|_| "too many column openings")?;
        for (ordinal, plan) in plans[start..end].iter_mut().enumerate() {
            plan.is_segment_first = ordinal == 0;
            plan.is_segment_last = ordinal + 1 == count as usize;
            plan.observation_ordinal = ordinal as u32;
            plan.observation_count = count;
        }
        start = end;
    }
    Ok(plans)
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
struct ColumnOpeningObservationPrepColsV19<T> {
    active: T,
    is_segment_first: T,
    is_segment_last: T,
    is_column_claim: T,
    proof_index: T,
    segment_index_lo: T,
    segment_index_hi: T,
    observation_ordinal: T,
    observation_count: T,
    sort_idx: T,
    part_idx: T,
    col_idx: T,
    is_rot: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ColumnOpeningObservationColsV19<T> {
    /// Runtime source-slot activity. The fixed HLeaf key contains plans for
    /// all four slots, while a terminal leaf may contain only a nonempty
    /// prefix. Every row belonging to one source slot carries the same bit.
    pub active: T,
    pub opening_start_tidx: T,
    pub claim: [T; D_EF],
}

/// Absorbs every authenticated batch-constraint column opening before any
/// per-shard batching challenge is sampled. This is the missing SWIRL
/// Fiat--Shamir boundary: the partial verifier authenticates the values on
/// `column_claims_bus`, while this AIR binds the same values to the transcript
/// in the backend's exact common-then-cached order.
#[derive(Clone, ColumnsAir)]
#[columns_via(ColumnOpeningObservationColsV19<u8>)]
pub struct ColumnOpeningObservationAirV19 {
    segment_start: u32,
    runtime_capacity_v4: bool,
    plans: Arc<[ColumnOpeningObservationPlanV19]>,
    transcript_bus: TranscriptBus,
    column_claims_bus: ColumnClaimsBus,
    opening_phase_start_bus: LogUpOpeningPhaseStartBusV19,
    claim_phase_start_bus: LogUpClaimPhaseStartBusV19,
}

impl ColumnOpeningObservationAirV19 {
    fn new(
        profile: &DirectLogUpMappedFunctionalProfileV19,
        transcript_bus: TranscriptBus,
        column_claims_bus: ColumnClaimsBus,
        opening_phase_start_bus: LogUpOpeningPhaseStartBusV19,
        claim_phase_start_bus: LogUpClaimPhaseStartBusV19,
    ) -> Result<Self, &'static str> {
        Ok(Self {
            segment_start: profile.segment_start(),
            runtime_capacity_v4: profile.runtime_capacity_v4(),
            plans: column_opening_observation_plan_v19(profile)?.into(),
            transcript_bus,
            column_claims_bus,
            opening_phase_start_bus,
            claim_phase_start_bus,
        })
    }
}

impl BaseAir<F> for ColumnOpeningObservationAirV19 {
    fn width(&self) -> usize {
        ColumnOpeningObservationColsV19::<F>::width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = ColumnOpeningObservationPrepColsV19::<F>::width();
        let height = self.plans.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(width * height);
        for (row, plan) in self.plans.iter().enumerate() {
            let prep: &mut ColumnOpeningObservationPrepColsV19<F> =
                values[row * width..(row + 1) * width].borrow_mut();
            prep.active = F::ONE;
            prep.is_segment_first = F::from_bool(plan.is_segment_first);
            prep.is_segment_last = F::from_bool(plan.is_segment_last);
            prep.is_column_claim = F::from_bool(plan.claim_index.is_some());
            prep.proof_index = F::from_u32(plan.proof_index);
            prep.segment_index_lo = F::from_u32(plan.segment_index & 0xffff);
            prep.segment_index_hi = F::from_u32(plan.segment_index >> 16);
            prep.observation_ordinal = F::from_u32(plan.observation_ordinal);
            prep.observation_count = F::from_u32(plan.observation_count);
            prep.sort_idx = F::from_u32(plan.sort_idx);
            prep.part_idx = F::from_u32(plan.part_idx);
            prep.col_idx = F::from_u32(plan.col_idx);
            prep.is_rot = F::from_bool(plan.is_rot);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for ColumnOpeningObservationAirV19 {}
impl PartitionedBaseAir<F> for ColumnOpeningObservationAirV19 {}

impl<AB> Air<AB> for ColumnOpeningObservationAirV19
where
    AB: AirBuilder<F = F> + InteractionBuilder + PairBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let prep_matrix = builder.preprocessed();
        let prep_row = prep_matrix
            .row_slice(0)
            .expect("column-opening observation preprocessed row");
        let prep_next_row = prep_matrix
            .row_slice(1)
            .expect("column-opening observation next preprocessed row");
        let prep: &ColumnOpeningObservationPrepColsV19<AB::Var> = (*prep_row).borrow();
        let prep_next: &ColumnOpeningObservationPrepColsV19<AB::Var> = (*prep_next_row).borrow();
        let main = builder.main();
        let local_row = main.row_slice(0).expect("column-opening observation row");
        let next_row = main
            .row_slice(1)
            .expect("column-opening observation next row");
        let local: &ColumnOpeningObservationColsV19<AB::Var> = (*local_row).borrow();
        let next: &ColumnOpeningObservationColsV19<AB::Var> = (*next_row).borrow();
        for bit in [
            prep.active,
            prep.is_segment_first,
            prep.is_segment_last,
            prep.is_column_claim,
            prep.is_rot,
        ] {
            builder.assert_bool(bit);
        }
        builder.assert_bool(local.active);
        builder
            .when(AB::Expr::ONE - AB::Expr::from(prep.active))
            .assert_zero(local.active);
        builder.when_first_row().assert_one(prep.active);
        builder.when_first_row().assert_one(local.active);
        if !self.runtime_capacity_v4 {
            builder
                .when(AB::Expr::from(prep.active))
                .assert_one(local.active);
        }
        let enabled = AB::Expr::from(prep.active) * AB::Expr::from(local.active);
        let first = enabled.clone() * AB::Expr::from(prep.is_segment_first);
        let last = enabled.clone() * AB::Expr::from(prep.is_segment_last);
        builder.when(enabled.clone()).assert_eq(
            AB::Expr::from(prep.proof_index) + AB::Expr::from_u32(self.segment_start),
            AB::Expr::from(prep.segment_index_lo)
                + AB::Expr::from_u32(1 << 16) * AB::Expr::from(prep.segment_index_hi),
        );
        let planned_transition = AB::Expr::from(prep.active) * AB::Expr::from(prep_next.active);
        let continuing =
            planned_transition.clone() * (AB::Expr::ONE - AB::Expr::from(prep.is_segment_last));
        builder
            .when_transition()
            .when(continuing.clone())
            .assert_eq(next.active, local.active);
        builder
            .when_transition()
            .when(
                planned_transition
                    * AB::Expr::from(prep.is_segment_last)
                    * (AB::Expr::ONE - AB::Expr::from(local.active)),
            )
            .assert_zero(next.active);
        builder
            .when_transition()
            .when(continuing * AB::Expr::from(local.active))
            .assert_eq(next.opening_start_tidx, local.opening_start_tidx);

        self.opening_phase_start_bus.receive(
            builder,
            LogUpOpeningPhaseStartMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                opening_start_tidx: local.opening_start_tidx.into(),
            },
            first,
        );
        self.transcript_bus.observe_ext(
            builder,
            prep.proof_index,
            AB::Expr::from(local.opening_start_tidx)
                + AB::Expr::from(prep.observation_ordinal) * AB::Expr::from_usize(D_EF),
            local.claim.map(Into::into),
            enabled.clone(),
        );
        self.column_claims_bus.receive(
            builder,
            prep.proof_index,
            ColumnClaimsMessage {
                sort_idx: prep.sort_idx.into(),
                part_idx: prep.part_idx.into(),
                col_idx: prep.col_idx.into(),
                claim: local.claim.map(Into::into),
                is_rot: prep.is_rot.into(),
            },
            enabled.clone() * AB::Expr::from(prep.is_column_claim),
        );
        for limb in local.claim {
            builder
                .when(enabled.clone() * (AB::Expr::ONE - AB::Expr::from(prep.is_column_claim)))
                .assert_zero(limb);
        }
        self.claim_phase_start_bus.send(
            builder,
            LogUpClaimPhaseStartMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                first_claim_tidx: AB::Expr::from(local.opening_start_tidx)
                    + AB::Expr::from(prep.observation_count) * AB::Expr::from_usize(D_EF),
            },
            last,
        );
    }
}

fn generate_column_opening_observation_trace_v19(
    air: &ColumnOpeningObservationAirV19,
    records: &[MappedFunctionalSourceRecordV19],
    opening_start_tidxs: &[u32],
) -> Result<RowMajorMatrix<F>, &'static str> {
    if opening_start_tidxs.is_empty() {
        return Err("column-opening observation source shape");
    }
    if if air.runtime_capacity_v4 {
        opening_start_tidxs.len() != records.len()
            || opening_start_tidxs.len() > VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2
            || air.plans.iter().any(|plan| {
                let proof_index = plan.proof_index as usize;
                proof_index < opening_start_tidxs.len() && plan.source_index >= records.len()
            })
    } else {
        air.plans.iter().any(|plan| {
            plan.proof_index as usize >= opening_start_tidxs.len()
                || plan.source_index >= records.len()
        })
    } {
        return Err("column-opening observation source shape");
    }
    let width = ColumnOpeningObservationColsV19::<F>::width();
    let height = air.plans.len().next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    for (row, plan) in air.plans.iter().enumerate() {
        let cols: &mut ColumnOpeningObservationColsV19<F> =
            values[row * width..(row + 1) * width].borrow_mut();
        if air.runtime_capacity_v4 && plan.proof_index as usize >= opening_start_tidxs.len() {
            continue;
        }
        cols.active = F::ONE;
        cols.opening_start_tidx = F::from_u32(opening_start_tidxs[plan.proof_index as usize]);
        if let Some(claim_index) = plan.claim_index {
            let claim = *records[plan.source_index]
                .dynamic_column_claims
                .get(claim_index)
                .ok_or("column-opening observation claim index")?;
            copy_ext_v19(&mut cols.claim, claim);
        }
    }
    Ok(RowMajorMatrix::new(values, width))
}

fn derive_program_parameters_v19(
    app_vk_digest: [F; DIGEST_SIZE],
    registry_digest: [F; DIGEST_SIZE],
    relation_digest: [F; DIGEST_SIZE],
    log_height: u8,
    cached_width: u32,
) -> Result<(Vec<EF>, EF), &'static str> {
    let mut transcript = default_duplex_sponge_recorder();
    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(
        &mut transcript,
        F::from_u64(PROGRAM_FINGERPRINT_DOMAIN_TAG_V19),
    );
    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(
        &mut transcript,
        F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
    );
    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_commit(&mut transcript, app_vk_digest);
    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_commit(
        &mut transcript,
        registry_digest,
    );
    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_commit(
        &mut transcript,
        relation_digest,
    );
    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(
        &mut transcript,
        F::from_u8(log_height),
    );
    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(
        &mut transcript,
        F::from_u32(cached_width),
    );
    let row_point = (0..log_height)
        .map(|_| FiatShamirTranscript::<BabyBearPoseidon2Config>::sample_ext(&mut transcript))
        .collect::<Vec<_>>();
    let column = FiatShamirTranscript::<BabyBearPoseidon2Config>::sample_ext(&mut transcript);
    if column == EF::ZERO {
        return Err("zero Program column challenge");
    }
    Ok((row_point, column))
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
struct MappedTermPrepColsV19<T> {
    active: T,
    is_source_first: T,
    is_source_last: T,
    is_program: T,
    is_program_term: T,
    is_active_count_term: T,
    authenticate_cached_setup_claim: T,
    active_count_expected: T,
    active_count_height_scale: T,
    is_rot: T,
    proof_index: T,
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
    sort_idx: T,
    part_idx: T,
    col_idx: T,
    block_start: T,
    l_skip: T,
    folded_len: T,
    short_trace: T,
    term_ordinal: T,
    term_count: T,
    observation_start: T,
    observation_count: T,
    log_message_len_bytes: [T; 8],
    term_count_bytes: [T; 8],
    block_start_bytes: [T; 8],
    log_height_bytes: [T; 8],
    l_skip_bytes: [T; 8],
    rotation_bytes: [T; 8],
    barycentric_len_bytes: [T; 8],
    folded_len_bytes: [T; 8],
    is_l_skip: [T; MAX_L_SKIP_V19 + 1],
    is_effective_l_skip: [T; MAX_L_SKIP_V19 + 1],
    rs_used: [T; MAX_RAW_MESSAGE_POINT_LEN_V19],
    rs_source_index: [T; MAX_RAW_MESSAGE_POINT_LEN_V19],
    point_used: [T; MAX_RAW_MESSAGE_POINT_LEN_V19],
    prefix_used: [T; MAX_RAW_MESSAGE_POINT_LEN_V19],
    prefix_bit: [T; MAX_RAW_MESSAGE_POINT_LEN_V19],
    row_used: [T; MAX_RAW_MESSAGE_POINT_LEN_V19],
    row_point_selector: [[T; MAX_RAW_MESSAGE_POINT_LEN_V19]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    shift_row_selector: [[T; MAX_RAW_MESSAGE_POINT_LEN_V19]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    low_point_selector: [[T; MAX_RAW_MESSAGE_POINT_LEN_V19]; MAX_L_SKIP_V19],
    skip_domain: [T; MAX_SKIP_DOMAIN_V19],
    skip_len_inverse: T,
    program_row_point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    program_column_challenge: [T; D_EF],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct MappedFunctionalTermColsV19<T> {
    pub active: T,
    /// Authenticated transcript cursor at which this source's canonical
    /// one-shot claim descriptor begins. It is received on the first term and
    /// carried unchanged across all remaining terms of the source.
    pub claim_start_tidx: T,
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub source_root: [T; DIGEST_SIZE],
    pub range_start: T,
    pub range_end: T,
    pub batching_challenge: [T; D_EF],
    pub program_fingerprint: [T; D_EF],
    pub program_mix_challenge: [T; D_EF],
    pub active_count_challenge: [T; D_EF],
    pub rs: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub one_shot_point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub row_one_shot_point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub low_one_shot_point: [[T; D_EF]; MAX_L_SKIP_V19],
    pub claim: [T; D_EF],
    pub ordinary_power_before: [T; D_EF],
    pub ordinary_power_after: [T; D_EF],
    pub program_power_before: [T; D_EF],
    pub program_power_after: [T; D_EF],
    pub ordinary_target_before: [T; D_EF],
    pub ordinary_target_after: [T; D_EF],
    pub ordinary_weight_before: [T; D_EF],
    pub ordinary_weight_after: [T; D_EF],
    pub fingerprint_weight_before: [T; D_EF],
    pub fingerprint_weight_after: [T; D_EF],
    pub r0_powers: [[T; D_EF]; MAX_L_SKIP_V19 + 1],
    pub barycentric: [[T; D_EF]; MAX_SKIP_DOMAIN_V19],
    pub denominator_inverses: [[T; D_EF]; MAX_SKIP_DOMAIN_V19],
    pub prefix_products: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19 + 1],
    pub row_products: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19 + 1],
    pub shift_dp_zero: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19 + 1],
    pub shift_dp_one: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19 + 1],
    pub low_zero_products: [[T; D_EF]; MAX_L_SKIP_V19 + 1],
    pub low_fold_current: [[T; D_EF]; 2 * MAX_SKIP_DOMAIN_V19],
    pub low_fold_shifted: [[T; D_EF]; 2 * MAX_SKIP_DOMAIN_V19],
    pub term_scale: [T; D_EF],
    pub rotation_inner: [T; D_EF],
    pub term_weight_at_point: [T; D_EF],
}

/// Exact setup-time size report for the fixed mapped-functional AIR.
///
/// `estimated_working_set_bytes` is deliberately conservative: it multiplies
/// both main and preprocessed LDE storage by the documented scratch factor.
/// It is an exact-profile metric for runtime admission, not a protocol-level
/// RAM cap or a promise that the allocator reaches exactly this peak.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MappedFunctionalTraceMetricsV19 {
    pub main_width: usize,
    pub preprocessed_width: usize,
    pub logical_rows: usize,
    pub padded_rows: usize,
    pub lde_rows: usize,
    pub main_cells: usize,
    pub preprocessed_cells: usize,
    pub lde_cells: usize,
    pub lde_bytes: usize,
    pub estimated_working_set_bytes: usize,
}

impl MappedFunctionalTraceMetricsV19 {
    fn new(logical_rows: usize, log_blowup: usize) -> Result<Self, &'static str> {
        if logical_rows == 0 {
            return Err("mapped-functional profile has no rows");
        }
        let main_width = MappedFunctionalTermColsV19::<F>::width();
        let preprocessed_width = MappedTermPrepColsV19::<F>::width();
        let padded_rows = logical_rows
            .checked_next_power_of_two()
            .ok_or("mapped-functional row padding overflow")?
            .max(2);
        let lde_rows = padded_rows
            .checked_shl(u32::try_from(log_blowup).map_err(|_| "LogUp blowup exceeds u32")?)
            .ok_or("mapped-functional LDE row overflow")?;
        let main_cells = main_width
            .checked_mul(padded_rows)
            .ok_or("mapped-functional main cell overflow")?;
        let preprocessed_cells = preprocessed_width
            .checked_mul(padded_rows)
            .ok_or("mapped-functional preprocessed cell overflow")?;
        let lde_width = main_width
            .checked_add(preprocessed_width)
            .ok_or("mapped-functional LDE width overflow")?;
        let lde_cells = lde_width
            .checked_mul(lde_rows)
            .ok_or("mapped-functional LDE cell overflow")?;
        let lde_bytes = lde_cells
            .checked_mul(core::mem::size_of::<F>())
            .ok_or("mapped-functional LDE byte overflow")?;
        let estimated_working_set_bytes = lde_bytes
            .checked_mul(MAPPED_FUNCTIONAL_LDE_WORKING_SET_MULTIPLIER_V19)
            .ok_or("mapped-functional working-set overflow")?;
        Ok(Self {
            main_width,
            preprocessed_width,
            logical_rows,
            padded_rows,
            lde_rows,
            main_cells,
            preprocessed_cells,
            lde_cells,
            lde_bytes,
            estimated_working_set_bytes,
        })
    }
}

/// Verifies the exact structured mapped functional rather than accepting its
/// descriptor from the host.  One row represents one ordinary or Program
/// term; all products are split through witness recurrences so the AIR degree
/// remains at most three.
#[derive(Clone, ColumnsAir)]
#[columns_via(MappedFunctionalTermColsV19<u8>)]
pub struct MappedFunctionalTermAirV19 {
    pub profile: DirectLogUpMappedFunctionalProfileV19,
    pub forest_bus: VerifiedSourceForestLeafBusV19,
    pub challenge_bus: MappedFunctionalChallengeBusV19,
    pub auxiliary_challenge_bus: MappedAuxiliaryChallengeBusV19,
    pub opening_point_bus: SegmentOpeningPointBusV19,
    pub column_claims_bus: ColumnClaimsBus,
    pub transcript_bus: TranscriptBus,
    pub round_start_bus: OneShotRoundStartBusV19,
    pub stream_cursor_bus: OneShotStreamCursorBusV19,
    pub mapped_bus: VerifiedMappedFunctionalBusV19,
}

impl BaseAir<F> for MappedFunctionalTermAirV19 {
    fn width(&self) -> usize {
        MappedFunctionalTermColsV19::<F>::width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = MappedTermPrepColsV19::<F>::width();
        let height = self.profile.terms.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(width * height);
        for (row, plan) in self.profile.terms.iter().enumerate() {
            let prep: &mut MappedTermPrepColsV19<F> =
                values[row * width..(row + 1) * width].borrow_mut();
            prep.active = F::ONE;
            prep.is_source_first = F::from_bool(plan.term_ordinal == 0);
            prep.is_source_last = F::from_bool(plan.term_ordinal + 1 == plan.term_count);
            prep.is_program = F::from_bool(plan.is_program);
            prep.is_program_term = F::from_bool(plan.is_program_term);
            prep.is_active_count_term = F::from_bool(plan.is_active_count_term);
            prep.authenticate_cached_setup_claim =
                F::from_bool(plan.authenticate_cached_setup_claim);
            prep.active_count_expected = F::from_u8(plan.active_count_expected);
            prep.active_count_height_scale = if plan.is_active_count_term {
                F::from_usize(1usize << plan.log_height)
            } else {
                F::ZERO
            };
            prep.is_rot = F::from_bool(plan.is_rot);
            prep.proof_index = F::from_u32(plan.proof_index);
            prep.segment_index_lo = F::from_u32(plan.segment_index & 0xffff);
            prep.segment_index_hi = F::from_u32(plan.segment_index >> 16);
            prep.shard_ordinal = F::from_u16(plan.shard_ordinal);
            prep.shard_count = F::from_u16(plan.shard_count);
            prep.shard_id = F::from_u32(plan.shard_id);
            prep.air_id = F::from_u32(plan.air_id);
            prep.relation_digest = plan.relation_digest;
            prep.log_height = F::from_u8(plan.log_height);
            prep.cached_width = F::from_u32(plan.cached_width);
            prep.log_message_len = F::from_u8(plan.log_message_len);
            prep.sort_idx = F::from_u32(plan.sort_idx);
            prep.part_idx = F::from_u32(plan.part_idx);
            prep.col_idx = F::from_u32(plan.col_idx);
            prep.block_start = F::from_u32(plan.block_start);
            prep.l_skip = F::from_u8(plan.l_skip);
            let effective_l_skip = plan.l_skip.min(plan.log_height);
            let folded_len = usize::from(plan.log_height).saturating_sub(plan.l_skip as usize);
            prep.folded_len = F::from_usize(folded_len);
            prep.short_trace = F::from_bool(plan.log_height < plan.l_skip);
            prep.term_ordinal = F::from_u32(plan.term_ordinal);
            prep.term_count = F::from_u32(plan.term_count);
            prep.observation_start = F::from_u32(plan.observation_start);
            prep.observation_count = F::from_u32(plan.observation_count);
            prep.log_message_len_bytes =
                (plan.log_message_len as u64).to_le_bytes().map(F::from_u8);
            prep.term_count_bytes = (plan.term_count as u64).to_le_bytes().map(F::from_u8);
            prep.block_start_bytes = (plan.block_start as u64).to_le_bytes().map(F::from_u8);
            prep.log_height_bytes = (plan.log_height as u64).to_le_bytes().map(F::from_u8);
            prep.l_skip_bytes = (plan.l_skip as u64).to_le_bytes().map(F::from_u8);
            prep.rotation_bytes = (u64::from(plan.is_rot)).to_le_bytes().map(F::from_u8);
            let skip_len = 1usize << plan.l_skip;
            prep.barycentric_len_bytes = (skip_len as u64).to_le_bytes().map(F::from_u8);
            prep.folded_len_bytes = (folded_len as u64).to_le_bytes().map(F::from_u8);
            prep.is_l_skip[plan.l_skip as usize] = F::ONE;
            prep.is_effective_l_skip[effective_l_skip as usize] = F::ONE;
            if !plan.is_program_term {
                prep.rs_used[0] = F::ONE;
                for index in 0..folded_len {
                    prep.rs_used[index + 1] = F::ONE;
                    prep.rs_source_index[index + 1] = F::from_usize(folded_len - index);
                }
            }
            for used in prep
                .point_used
                .iter_mut()
                .take(plan.log_message_len as usize)
            {
                *used = F::ONE;
            }
            let prefix_len = (plan.log_message_len - plan.log_height) as usize;
            let block_index = (plan.block_start as usize) >> plan.log_height;
            for index in 0..prefix_len {
                prep.prefix_used[index] = F::ONE;
                prep.prefix_bit[index] =
                    F::from_bool(((block_index >> (prefix_len - 1 - index)) & 1) == 1);
            }
            for used in prep.row_used.iter_mut().take(folded_len) {
                *used = F::ONE;
            }
            for row_index in 0..folded_len {
                prep.row_point_selector[row_index][prefix_len + row_index] = F::ONE;
                prep.shift_row_selector[row_index][folded_len - 1 - row_index] = F::ONE;
            }
            for low_index in 0..effective_l_skip as usize {
                prep.low_point_selector[low_index][prefix_len + folded_len + low_index] = F::ONE;
            }
            let omega = F::two_adic_generator(plan.l_skip as usize);
            let mut omega_power = F::ONE;
            for point in prep.skip_domain.iter_mut().take(skip_len) {
                *point = omega_power;
                omega_power *= omega;
            }
            prep.skip_len_inverse = F::from_usize(skip_len).inverse();
            for (target, value) in prep
                .program_row_point
                .iter_mut()
                .zip(&plan.program_row_point)
            {
                target.copy_from_slice(value.as_basis_coefficients_slice());
            }
            prep.program_column_challenge
                .copy_from_slice(plan.program_column_challenge.as_basis_coefficients_slice());
        }
        Some(RowMajorMatrix::new(values, width))
    }
}
impl BaseAirWithPublicValues<F> for MappedFunctionalTermAirV19 {}
impl PartitionedBaseAir<F> for MappedFunctionalTermAirV19 {}

fn ext_zero_expr<FA: PrimeCharacteristicRing>() -> [FA; D_EF] {
    core::array::from_fn(|_| FA::ZERO)
}

fn ext_one_expr<FA: PrimeCharacteristicRing>() -> [FA; D_EF] {
    core::array::from_fn(|index| if index == 0 { FA::ONE } else { FA::ZERO })
}

fn ext_add_expr<FA: PrimeCharacteristicRing>(
    left: [impl Into<FA>; D_EF],
    right: [impl Into<FA>; D_EF],
) -> [FA; D_EF] {
    let left = left.map(Into::into);
    let right = right.map(Into::into);
    core::array::from_fn(|index| left[index].clone() + right[index].clone())
}

fn ext_sub_expr<FA: PrimeCharacteristicRing>(
    left: [impl Into<FA>; D_EF],
    right: [impl Into<FA>; D_EF],
) -> [FA; D_EF] {
    let left = left.map(Into::into);
    let right = right.map(Into::into);
    core::array::from_fn(|index| left[index].clone() - right[index].clone())
}

fn ext_scale_expr<FA: PrimeCharacteristicRing>(
    value: [impl Into<FA>; D_EF],
    scalar: impl Into<FA> + Clone,
) -> [FA; D_EF] {
    let value = value.map(Into::into);
    let scalar = scalar.into();
    core::array::from_fn(|index| value[index].clone() * scalar.clone())
}

fn ext_mul_expr<FA>(left: [impl Into<FA>; D_EF], right: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
    FA::PrimeSubfield: BinomiallyExtendable<D_EF>,
{
    let left = left.map(Into::into);
    let right = right.map(Into::into);
    let w = FA::from_prime_subfield(FA::PrimeSubfield::W);
    let mut output = core::array::from_fn(|_| FA::ZERO);
    for (left_degree, left_value) in left.iter().enumerate() {
        for (right_degree, right_value) in right.iter().enumerate() {
            let degree = left_degree + right_degree;
            let mut term = left_value.clone() * right_value.clone();
            if degree >= D_EF {
                term *= w.clone();
            }
            output[degree % D_EF] = output[degree % D_EF].clone() + term;
        }
    }
    output
}

impl<AB> Air<AB> for MappedFunctionalTermAirV19
where
    AB: AirBuilder<F = F> + InteractionBuilder + PairBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed();
        let prep_row = preprocessed
            .row_slice(0)
            .expect("mapped-functional plan row");
        let prep: &MappedTermPrepColsV19<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let local_row = main.row_slice(0).expect("mapped-functional row");
        let next_row = main.row_slice(1).expect("mapped-functional next row");
        let local: &MappedFunctionalTermColsV19<AB::Var> = (*local_row).borrow();
        let next: &MappedFunctionalTermColsV19<AB::Var> = (*next_row).borrow();

        for bit in [
            prep.active,
            prep.is_source_first,
            prep.is_source_last,
            prep.is_program,
            prep.is_program_term,
            prep.is_active_count_term,
            prep.authenticate_cached_setup_claim,
            prep.is_rot,
            prep.short_trace,
            local.active,
        ] {
            builder.assert_bool(bit);
        }
        if self.profile.runtime_capacity_v4() {
            builder.when_first_row().assert_one(local.active);
            let mut occupancy_transition = builder.when_transition();
            occupancy_transition
                .when(AB::Expr::ONE - AB::Expr::from(prep.is_source_last))
                .assert_eq(next.active, local.active);
            occupancy_transition
                .when(AB::Expr::from(prep.is_source_last))
                .assert_zero(
                    (AB::Expr::ONE - AB::Expr::from(local.active)) * AB::Expr::from(next.active),
                );
            let inactive = AB::Expr::from(prep.active) - AB::Expr::from(local.active);
            for value in (*local_row).iter().skip(1) {
                builder.when(inactive.clone()).assert_zero((*value).into());
            }
        } else {
            builder.assert_eq(local.active, prep.active);
        }
        let enabled = AB::Expr::from(local.active);
        let first = enabled.clone() * AB::Expr::from(prep.is_source_first);
        let last = enabled.clone() * AB::Expr::from(prep.is_source_last);
        // Preserve the exact v19 stream handoff formerly owned by
        // `OneShotClaimTranscriptAirV19`. The cursor is authenticated only on
        // the first term and then constrained to remain constant until the
        // final term emits the reduction-round start.
        self.stream_cursor_bus.receive(
            builder,
            OneShotStreamCursorMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                next_shard_ordinal: prep.shard_ordinal.into(),
                tidx: local.claim_start_tidx.into(),
            },
            first.clone(),
        );
        let active_count = enabled.clone() * AB::Expr::from(prep.is_active_count_term);
        let ordinary = enabled.clone()
            * (AB::Expr::ONE - AB::Expr::from(prep.is_program_term))
            * (AB::Expr::ONE - AB::Expr::from(prep.is_active_count_term));
        let program_term = enabled.clone() * AB::Expr::from(prep.is_program_term);
        // Ordinary and active-count rows both evaluate the same committed
        // message column; only the disjoint Program row is excluded.  Keep
        // this selector in its reduced form. Expanding it as
        //
        //   ordinary + active_count
        // = active(1 - program)(1 - count) + active*count
        //
        // introduces one redundant symbolic factor. The separately enforced
        // `program * count = 0` identity proves that expansion equal to
        // `active * (1 - program)`. The reduced gate is therefore identical
        // on every satisfying trace while keeping the barycentric constraints
        // within the protocol's degree-eight envelope.
        let mapped_column =
            enabled.clone() * (AB::Expr::ONE - AB::Expr::from(prep.is_program_term));

        builder.assert_zero(
            AB::Expr::from(prep.is_program_term) * AB::Expr::from(prep.is_active_count_term),
        );
        builder.when(active_count.clone()).assert_zero(prep.is_rot);
        builder
            .when(enabled.clone() * (AB::Expr::ONE - AB::Expr::from(prep.is_active_count_term)))
            .assert_zero(prep.active_count_expected);
        builder
            .when(enabled.clone() * (AB::Expr::ONE - AB::Expr::from(prep.is_active_count_term)))
            .assert_zero(prep.active_count_height_scale);

        self.forest_bus.receive(
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
                range_start: local.range_start.into(),
                range_end: local.range_end.into(),
            },
            first.clone(),
        );
        self.challenge_bus.receive(
            builder,
            MappedFunctionalChallengeMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                shard_ordinal: prep.shard_ordinal.into(),
                shard_id: prep.shard_id.into(),
                relation_digest: prep.relation_digest.map(Into::into),
                is_program: prep.is_program.into(),
                batching_challenge: local.batching_challenge.map(Into::into),
                program_fingerprint: local.program_fingerprint.map(Into::into),
                program_mix_challenge: local.program_mix_challenge.map(Into::into),
            },
            first.clone(),
        );
        self.auxiliary_challenge_bus.receive(
            builder,
            MappedAuxiliaryChallengeMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                shard_ordinal: prep.shard_ordinal.into(),
                relation_digest: prep.relation_digest.map(Into::into),
                expected_value: if self.profile.runtime_capacity_v4() {
                    local.claim[0].into()
                } else {
                    prep.active_count_expected.into()
                },
                challenge: local.active_count_challenge.map(Into::into),
            },
            active_count.clone(),
        );
        // The setup-derived coefficient is carried unchanged across the
        // complete source row group and is authenticated exactly once on the
        // count term. Carrying it is required by the source transition
        // constraints below; ordinary terms never multiply by it.

        // Every ordinary term independently consumes the exact verifier-exported
        // PLE point. This makes fanout multiplicity verifier-owned and avoids a
        // host-carried opening-point checkpoint.
        for index in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
            builder.assert_bool(prep.rs_used[index]);
            let used = ordinary.clone() * AB::Expr::from(prep.rs_used[index]);
            self.opening_point_bus.lookup_key(
                builder,
                SegmentOpeningPointMessageV19 {
                    segment_index_lo: prep.segment_index_lo.into(),
                    segment_index_hi: prep.segment_index_hi.into(),
                    index: prep.rs_source_index[index].into(),
                    value: local.rs[index].map(Into::into),
                },
                used.clone(),
            );
            for (limb_index, limb) in local.rs[index].iter().copied().enumerate() {
                builder
                    .when(enabled.clone() * (AB::Expr::ONE - AB::Expr::from(prep.rs_used[index])))
                    .assert_zero(limb);
                let count_value = if index == 0 {
                    AB::Expr::ZERO
                } else if limb_index == 0 {
                    AB::Expr::from(F::TWO.inverse())
                } else {
                    AB::Expr::ZERO
                };
                builder
                    .when(active_count.clone() * AB::Expr::from(prep.rs_used[index]))
                    .assert_eq(limb, count_value);
            }
            builder.assert_bool(prep.point_used[index]);
            for limb in local.one_shot_point[index] {
                builder
                    .when(
                        enabled.clone() * (AB::Expr::ONE - AB::Expr::from(prep.point_used[index])),
                    )
                    .assert_zero(limb);
            }
        }

        // Select the row and low coordinates from the eventual one-shot point.
        for row_index in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
            builder.assert_bool(prep.row_used[row_index]);
            let selected: [AB::Expr; D_EF] = core::array::from_fn(|limb| {
                (0..MAX_RAW_MESSAGE_POINT_LEN_V19).fold(AB::Expr::ZERO, |sum, point_index| {
                    sum + AB::Expr::from(prep.row_point_selector[row_index][point_index])
                        * AB::Expr::from(local.one_shot_point[point_index][limb])
                })
            });
            assert_array_eq(
                &mut builder.when(enabled.clone()),
                local.row_one_shot_point[row_index],
                selected,
            );
        }
        for low_index in 0..MAX_L_SKIP_V19 {
            let selected: [AB::Expr; D_EF] = core::array::from_fn(|limb| {
                (0..MAX_RAW_MESSAGE_POINT_LEN_V19).fold(AB::Expr::ZERO, |sum, point_index| {
                    sum + AB::Expr::from(prep.low_point_selector[low_index][point_index])
                        * AB::Expr::from(local.one_shot_point[point_index][limb])
                })
            });
            assert_array_eq(
                &mut builder.when(enabled.clone()),
                local.low_one_shot_point[low_index],
                selected,
            );
        }

        self.column_claims_bus.send(
            builder,
            prep.proof_index,
            ColumnClaimsMessage {
                sort_idx: prep.sort_idx.into(),
                part_idx: prep.part_idx.into(),
                col_idx: prep.col_idx.into(),
                claim: local.claim.map(Into::into),
                is_rot: prep.is_rot.into(),
            },
            ordinary.clone(),
        );
        // A cached-main setup column has a second, independently authenticated
        // PCS provenance. Publish one additional source copy so the authority
        // bridge can consume it and force equality. This is setup-fixed and is
        // never enabled for common-main, Program, or occupancy terms.
        self.column_claims_bus.send(
            builder,
            prep.proof_index,
            ColumnClaimsMessage {
                sort_idx: prep.sort_idx.into(),
                part_idx: prep.part_idx.into(),
                col_idx: prep.col_idx.into(),
                claim: local.claim.map(Into::into),
                is_rot: prep.is_rot.into(),
            },
            ordinary.clone() * AB::Expr::from(prep.authenticate_cached_setup_claim),
        );
        // One copy feeds the partial verifier's symbolic-expression AIR; the
        // second feeds `ColumnOpeningObservationAirV19`, which binds the same
        // authenticated claim into Fiat--Shamir before batching challenges.
        self.column_claims_bus.send(
            builder,
            prep.proof_index,
            ColumnClaimsMessage {
                sort_idx: prep.sort_idx.into(),
                part_idx: prep.part_idx.into(),
                col_idx: prep.col_idx.into(),
                claim: local.claim.map(Into::into),
                is_rot: prep.is_rot.into(),
            },
            ordinary.clone(),
        );
        for limb in local.claim {
            builder.when(program_term.clone()).assert_zero(limb);
        }
        if !self.profile.runtime_capacity_v4() {
            builder
                .when(active_count.clone())
                .assert_eq(local.claim[0], AB::Expr::from(prep.active_count_expected));
        }
        for limb in local.claim.iter().skip(1) {
            builder.when(active_count.clone()).assert_zero(*limb);
        }

        // PLE barycentric coefficients at r_0.
        assert_array_eq(
            &mut builder.when(mapped_column.clone()),
            local.r0_powers[0],
            local.rs[0].map(Into::into),
        );
        for power in 0..MAX_L_SKIP_V19 {
            let power_used = (power + 1..=MAX_L_SKIP_V19).fold(AB::Expr::ZERO, |sum, l_skip| {
                sum + AB::Expr::from(prep.is_l_skip[l_skip])
            });
            let square = ext_mul_expr::<AB::Expr>(local.r0_powers[power], local.r0_powers[power]);
            assert_array_eq(
                &mut builder.when(mapped_column.clone() * power_used),
                local.r0_powers[power + 1],
                square,
            );
        }
        let r0_to_skip: [AB::Expr; D_EF] = core::array::from_fn(|limb| {
            (0..=MAX_L_SKIP_V19).fold(AB::Expr::ZERO, |sum, l_skip| {
                sum + AB::Expr::from(prep.is_l_skip[l_skip])
                    * AB::Expr::from(local.r0_powers[l_skip][limb])
            })
        });
        let scaling = ext_scale_expr::<AB::Expr>(
            ext_sub_expr::<AB::Expr>(r0_to_skip, ext_one_expr::<AB::Expr>()),
            prep.skip_len_inverse,
        );
        for z in 0..MAX_SKIP_DOMAIN_V19 {
            let bary_used = (0..=MAX_L_SKIP_V19).fold(AB::Expr::ZERO, |sum, l_skip| {
                sum + AB::Expr::from_bool(z < (1usize << l_skip))
                    * AB::Expr::from(prep.is_l_skip[l_skip])
            });
            let ordinary_used = mapped_column.clone() * bary_used.clone();
            let denominator: [AB::Expr; D_EF] = ext_sub_expr(
                local.rs[0],
                core::array::from_fn(|limb| {
                    if limb == 0 {
                        AB::Expr::from(prep.skip_domain[z])
                    } else {
                        AB::Expr::ZERO
                    }
                }),
            );
            assert_array_eq(
                &mut builder.when(ordinary_used.clone()),
                ext_mul_expr::<AB::Expr>(denominator, local.denominator_inverses[z]),
                ext_one_expr::<AB::Expr>(),
            );
            let bary: [AB::Expr; D_EF] = ext_scale_expr(
                ext_mul_expr::<AB::Expr>(local.denominator_inverses[z], scaling.clone()),
                prep.skip_domain[z],
            );
            assert_array_eq(&mut builder.when(ordinary_used), local.barycentric[z], bary);
        }
        assert_array_eq(
            &mut builder.when(program_term.clone()),
            local.barycentric[0],
            ext_one_expr::<AB::Expr>(),
        );

        // Equality factor selecting this aligned block in the full message.
        assert_array_eq(
            &mut builder.when(enabled.clone()),
            local.prefix_products[0],
            ext_one_expr::<AB::Expr>(),
        );
        for index in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
            let p = local.one_shot_point[index].map(Into::<AB::Expr>::into);
            let factor: [AB::Expr; D_EF] = ext_add_expr::<AB::Expr>(
                ext_scale_expr::<AB::Expr>(
                    ext_sub_expr::<AB::Expr>(ext_one_expr::<AB::Expr>(), p.clone()),
                    AB::Expr::ONE - AB::Expr::from(prep.prefix_bit[index]),
                ),
                ext_scale_expr::<AB::Expr>(p, prep.prefix_bit[index]),
            );
            let multiplied: [AB::Expr; D_EF] =
                ext_mul_expr::<AB::Expr>(local.prefix_products[index], factor);
            let selected: [AB::Expr; D_EF] = ext_add_expr(
                ext_scale_expr(multiplied, prep.prefix_used[index]),
                ext_scale_expr(
                    local.prefix_products[index],
                    AB::Expr::ONE - AB::Expr::from(prep.prefix_used[index]),
                ),
            );
            assert_array_eq(
                &mut builder.when(enabled.clone()),
                local.prefix_products[index + 1],
                selected,
            );
        }

        // Equality factor for the folded row point.
        assert_array_eq(
            &mut builder.when(enabled.clone()),
            local.row_products[0],
            ext_one_expr::<AB::Expr>(),
        );
        for index in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
            // Ordinary rows use `rs[1..=folded_len]`; the final envelope row
            // is necessarily inactive because protocol-v19 fixes l_skip=4,
            // hence folded_len <= log_message_len - 4.  Keep symbolic AIR
            // evaluation in bounds even though `row_used[index]` gates that
            // padding row off.
            let ordinary_a: [AB::Expr; D_EF] = if index + 1 < MAX_RAW_MESSAGE_POINT_LEN_V19 {
                local.rs[index + 1].map(Into::into)
            } else {
                ext_zero_expr::<AB::Expr>()
            };
            let a: [AB::Expr; D_EF] = core::array::from_fn(|limb| {
                AB::Expr::from(prep.is_program_term)
                    * AB::Expr::from(prep.program_row_point[index][limb])
                    + (AB::Expr::ONE - AB::Expr::from(prep.is_program_term))
                        * ordinary_a[limb].clone()
            });
            let p = local.row_one_shot_point[index].map(Into::<AB::Expr>::into);
            let factor: [AB::Expr; D_EF] = ext_add_expr::<AB::Expr>(
                ext_mul_expr::<AB::Expr>(
                    ext_sub_expr::<AB::Expr>(ext_one_expr::<AB::Expr>(), a.clone()),
                    ext_sub_expr::<AB::Expr>(ext_one_expr::<AB::Expr>(), p.clone()),
                ),
                ext_mul_expr::<AB::Expr>(a, p),
            );
            let multiplied: [AB::Expr; D_EF] =
                ext_mul_expr::<AB::Expr>(local.row_products[index], factor);
            let selected: [AB::Expr; D_EF] = ext_add_expr(
                ext_scale_expr(multiplied, prep.row_used[index]),
                ext_scale_expr(
                    local.row_products[index],
                    AB::Expr::ONE - AB::Expr::from(prep.row_used[index]),
                ),
            );
            assert_array_eq(
                &mut builder.when(enabled.clone()),
                local.row_products[index + 1],
                selected,
            );
        }

        // Correlation between row equality tables under cyclic +1. Bits are
        // processed from least to most significant while carrying one.
        assert_array_eq(
            &mut builder.when(enabled.clone()),
            local.shift_dp_zero[0],
            ext_zero_expr::<AB::Expr>(),
        );
        assert_array_eq(
            &mut builder.when(enabled.clone()),
            local.shift_dp_one[0],
            ext_one_expr::<AB::Expr>(),
        );
        for step in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
            let a: [AB::Expr; D_EF] = core::array::from_fn(|limb| {
                (0..MAX_RAW_MESSAGE_POINT_LEN_V19).fold(AB::Expr::ZERO, |sum, coordinate| {
                    let ordinary_source = if coordinate + 1 < MAX_RAW_MESSAGE_POINT_LEN_V19 {
                        AB::Expr::from(local.rs[coordinate + 1][limb])
                    } else {
                        AB::Expr::ZERO
                    };
                    let source = AB::Expr::from(prep.is_program_term)
                        * AB::Expr::from(prep.program_row_point[coordinate][limb])
                        + (AB::Expr::ONE - AB::Expr::from(prep.is_program_term)) * ordinary_source;
                    sum + AB::Expr::from(prep.shift_row_selector[step][coordinate]) * source
                })
            });
            let p: [AB::Expr; D_EF] = core::array::from_fn(|limb| {
                (0..MAX_RAW_MESSAGE_POINT_LEN_V19).fold(AB::Expr::ZERO, |sum, coordinate| {
                    sum + AB::Expr::from(prep.shift_row_selector[step][coordinate])
                        * AB::Expr::from(local.row_one_shot_point[coordinate][limb])
                })
            });
            let same: [AB::Expr; D_EF] = ext_add_expr::<AB::Expr>(
                ext_mul_expr::<AB::Expr>(
                    ext_sub_expr::<AB::Expr>(ext_one_expr::<AB::Expr>(), a.clone()),
                    ext_sub_expr::<AB::Expr>(ext_one_expr::<AB::Expr>(), p.clone()),
                ),
                ext_mul_expr::<AB::Expr>(a.clone(), p.clone()),
            );
            let next_zero: [AB::Expr; D_EF] = ext_add_expr::<AB::Expr>(
                ext_mul_expr::<AB::Expr>(local.shift_dp_zero[step], same),
                ext_mul_expr::<AB::Expr>(
                    local.shift_dp_one[step],
                    ext_mul_expr::<AB::Expr>(
                        ext_sub_expr::<AB::Expr>(ext_one_expr::<AB::Expr>(), a.clone()),
                        p.clone(),
                    ),
                ),
            );
            let next_one: [AB::Expr; D_EF] = ext_mul_expr::<AB::Expr>(
                local.shift_dp_one[step],
                ext_mul_expr::<AB::Expr>(
                    a,
                    ext_sub_expr::<AB::Expr>(ext_one_expr::<AB::Expr>(), p),
                ),
            );
            let used = prep.row_used[step];
            assert_array_eq(
                &mut builder.when(enabled.clone()),
                local.shift_dp_zero[step + 1],
                ext_add_expr::<AB::Expr>(
                    ext_scale_expr::<AB::Expr>(next_zero, used),
                    ext_scale_expr::<AB::Expr>(
                        local.shift_dp_zero[step],
                        AB::Expr::ONE - AB::Expr::from(used),
                    ),
                ),
            );
            assert_array_eq(
                &mut builder.when(enabled.clone()),
                local.shift_dp_one[step + 1],
                ext_add_expr::<AB::Expr>(
                    ext_scale_expr::<AB::Expr>(next_one, used),
                    ext_scale_expr::<AB::Expr>(
                        local.shift_dp_one[step],
                        AB::Expr::ONE - AB::Expr::from(used),
                    ),
                ),
            );
        }

        assert_array_eq(
            &mut builder.when(enabled.clone()),
            local.low_zero_products[0],
            ext_one_expr::<AB::Expr>(),
        );
        for index in 0..MAX_L_SKIP_V19 {
            let factor: [AB::Expr; D_EF] = ext_sub_expr::<AB::Expr>(
                ext_one_expr::<AB::Expr>(),
                local.low_one_shot_point[index],
            );
            let used = (index + 1..=MAX_L_SKIP_V19).fold(AB::Expr::ZERO, |sum, l_skip| {
                sum + AB::Expr::from(prep.is_l_skip[l_skip])
            });
            assert_array_eq(
                &mut builder.when(enabled.clone()),
                local.low_zero_products[index + 1],
                ext_add_expr::<AB::Expr>(
                    ext_scale_expr::<AB::Expr>(
                        ext_mul_expr::<AB::Expr>(local.low_zero_products[index], factor),
                        used.clone(),
                    ),
                    ext_scale_expr::<AB::Expr>(
                        local.low_zero_products[index],
                        AB::Expr::ONE - used,
                    ),
                ),
            );
        }

        // Evaluate the skip-domain table by the same high-bit-first folding
        // convention as the one-shot backend. If the physical trace is shorter
        // than 2^l_skip, the backend maps every skip-domain position modulo the
        // trace height. Collapse those repeated barycentric coefficients before
        // consuming the physical low coordinates. For a next-row rotation the
        // collapsed table is cyclically shifted; the ordinary folded-row carry
        // is used only when the trace is at least the skip domain.
        let mut low_current = ext_zero_expr::<AB::Expr>();
        let mut low_shifted = ext_zero_expr::<AB::Expr>();
        for effective_l_skip in 0..=MAX_L_SKIP_V19 {
            let q = 1usize << effective_l_skip;
            let gate = enabled.clone() * AB::Expr::from(prep.is_effective_l_skip[effective_l_skip]);
            let collapsed = |physical: usize| -> [AB::Expr; D_EF] {
                core::array::from_fn(|limb| {
                    (0..MAX_SKIP_DOMAIN_V19)
                        .filter(|source| source % q == physical)
                        .fold(AB::Expr::ZERO, |sum, source| {
                            let source_used = (0..=MAX_L_SKIP_V19).fold(
                                AB::Expr::ZERO,
                                |used, protocol_l_skip| {
                                    used + AB::Expr::from_bool(source < (1usize << protocol_l_skip))
                                        * AB::Expr::from(prep.is_l_skip[protocol_l_skip])
                                },
                            );
                            sum + source_used * AB::Expr::from(local.barycentric[source][limb])
                        })
                })
            };
            for z in 0..q {
                assert_array_eq(
                    &mut builder.when(gate.clone()),
                    local.low_fold_current[z],
                    collapsed(z),
                );
                let ordinary_shifted = if z == 0 {
                    ext_zero_expr::<AB::Expr>()
                } else {
                    local.barycentric[z - 1].map(Into::into)
                };
                let cyclic_shifted = collapsed((z + q - 1) % q);
                let shifted = ext_add_expr::<AB::Expr>(
                    ext_scale_expr::<AB::Expr>(cyclic_shifted, prep.short_trace),
                    ext_scale_expr::<AB::Expr>(
                        ordinary_shifted,
                        AB::Expr::ONE - AB::Expr::from(prep.short_trace),
                    ),
                );
                assert_array_eq(
                    &mut builder.when(gate.clone()),
                    local.low_fold_shifted[z],
                    shifted,
                );
            }
            let mut offset = 0usize;
            let mut next_offset = q;
            let mut len = q;
            for level in 0..effective_l_skip {
                // This compact tree folds adjacent entries, so its first
                // challenge controls the least-significant physical bit. The
                // backend MLE convention presents points high-bit first;
                // consume the low-point suffix in reverse order.
                let challenge_index = effective_l_skip - 1 - level;
                for node in 0..len / 2 {
                    let fold = |values: &[[AB::Var; D_EF]; 2 * MAX_SKIP_DOMAIN_V19]| {
                        ext_add_expr(
                            values[offset + 2 * node],
                            ext_mul_expr::<AB::Expr>(
                                local.low_one_shot_point[challenge_index],
                                ext_sub_expr(
                                    values[offset + 2 * node + 1],
                                    values[offset + 2 * node],
                                ),
                            ),
                        )
                    };
                    assert_array_eq(
                        &mut builder.when(gate.clone()),
                        local.low_fold_current[next_offset + node],
                        fold(&local.low_fold_current),
                    );
                    assert_array_eq(
                        &mut builder.when(gate.clone()),
                        local.low_fold_shifted[next_offset + node],
                        fold(&local.low_fold_shifted),
                    );
                }
                offset = next_offset;
                next_offset += len / 2;
                len /= 2;
            }
            let root = if effective_l_skip == 0 { 0 } else { 2 * q - 2 };
            low_current = ext_add_expr::<AB::Expr>(
                low_current,
                ext_scale_expr::<AB::Expr>(
                    local.low_fold_current[root],
                    prep.is_effective_l_skip[effective_l_skip],
                ),
            );
            low_shifted = ext_add_expr::<AB::Expr>(
                low_shifted,
                ext_scale_expr::<AB::Expr>(
                    local.low_fold_shifted[root],
                    prep.is_effective_l_skip[effective_l_skip],
                ),
            );
        }

        let row_eq = local.row_products[MAX_RAW_MESSAGE_POINT_LEN_V19];
        let prefix_eq = local.prefix_products[MAX_RAW_MESSAGE_POINT_LEN_V19];
        let shift_corr: [AB::Expr; D_EF] = ext_add_expr::<AB::Expr>(
            local.shift_dp_zero[MAX_RAW_MESSAGE_POINT_LEN_V19],
            local.shift_dp_one[MAX_RAW_MESSAGE_POINT_LEN_V19],
        );
        let selected_last_bary: [AB::Expr; D_EF] = core::array::from_fn(|limb| {
            (0..=MAX_L_SKIP_V19).fold(AB::Expr::ZERO, |sum, l_skip| {
                sum + AB::Expr::from(prep.is_l_skip[l_skip])
                    * AB::Expr::from(local.barycentric[(1usize << l_skip) - 1][limb])
            })
        });
        let selected_low_zero: [AB::Expr; D_EF] = core::array::from_fn(|limb| {
            (0..=MAX_L_SKIP_V19).fold(AB::Expr::ZERO, |sum, effective_l_skip| {
                sum + AB::Expr::from(prep.is_effective_l_skip[effective_l_skip])
                    * AB::Expr::from(local.low_zero_products[effective_l_skip][limb])
            })
        });
        let next_inner: [AB::Expr; D_EF] = ext_add_expr::<AB::Expr>(
            ext_mul_expr::<AB::Expr>(row_eq, low_shifted),
            ext_scale_expr::<AB::Expr>(
                ext_mul_expr::<AB::Expr>(
                    selected_last_bary,
                    ext_mul_expr::<AB::Expr>(selected_low_zero, shift_corr),
                ),
                AB::Expr::ONE - AB::Expr::from(prep.short_trace),
            ),
        );
        let rotation_inner: [AB::Expr; D_EF] = ext_add_expr::<AB::Expr>(
            ext_scale_expr::<AB::Expr>(
                ext_mul_expr::<AB::Expr>(row_eq, low_current),
                AB::Expr::ONE - AB::Expr::from(prep.is_rot),
            ),
            ext_scale_expr::<AB::Expr>(next_inner, prep.is_rot),
        );
        assert_array_eq(
            &mut builder.when(enabled.clone()),
            local.rotation_inner,
            rotation_inner,
        );
        assert_array_eq(
            &mut builder.when(enabled.clone()),
            local.term_weight_at_point,
            ext_mul_expr::<AB::Expr>(prefix_eq, local.rotation_inner),
        );

        // Canonical powers, targets, and separate ordinary/fingerprint weight
        // evaluations. The combined descriptor scale is emitted below.
        assert_array_eq(
            &mut builder.when(first.clone()),
            local.ordinary_power_before,
            ext_one_expr::<AB::Expr>(),
        );
        assert_array_eq(
            &mut builder.when(first.clone()),
            local.program_power_before,
            ext_one_expr::<AB::Expr>(),
        );
        for accumulator in [
            local.ordinary_target_before,
            local.ordinary_weight_before,
            local.fingerprint_weight_before,
        ] {
            assert_array_eq(
                &mut builder.when(first.clone()),
                accumulator,
                ext_zero_expr::<AB::Expr>(),
            );
        }
        let ordinary_selector = (AB::Expr::ONE - AB::Expr::from(prep.is_program_term))
            * (AB::Expr::ONE - AB::Expr::from(prep.is_active_count_term));
        let ordinary_power_after: [AB::Expr; D_EF] = ext_add_expr::<AB::Expr>(
            ext_scale_expr::<AB::Expr>(
                ext_mul_expr::<AB::Expr>(local.ordinary_power_before, local.batching_challenge),
                ordinary_selector.clone(),
            ),
            ext_scale_expr::<AB::Expr>(
                local.ordinary_power_before,
                AB::Expr::ONE - ordinary_selector.clone(),
            ),
        );
        assert_array_eq(
            &mut builder.when(enabled.clone()),
            local.ordinary_power_after,
            ordinary_power_after,
        );
        let program_power_after: [AB::Expr; D_EF] = ext_add_expr::<AB::Expr>(
            ext_scale_expr::<AB::Expr>(
                local.program_power_before,
                AB::Expr::ONE - AB::Expr::from(prep.is_program_term),
            ),
            ext_scale_expr::<AB::Expr>(
                ext_mul_expr::<AB::Expr>(local.program_power_before, prep.program_column_challenge),
                prep.is_program_term,
            ),
        );
        assert_array_eq(
            &mut builder.when(enabled.clone()),
            local.program_power_after,
            program_power_after,
        );
        let ordinary_contribution: [AB::Expr; D_EF] = ext_add_expr::<AB::Expr>(
            ext_scale_expr::<AB::Expr>(
                ext_mul_expr::<AB::Expr>(local.ordinary_power_before, local.claim),
                ordinary_selector.clone(),
            ),
            ext_scale_expr::<AB::Expr>(
                ext_mul_expr::<AB::Expr>(local.active_count_challenge, local.claim),
                prep.is_active_count_term,
            ),
        );
        assert_array_eq(
            &mut builder.when(enabled.clone()),
            local.ordinary_target_after,
            ext_add_expr::<AB::Expr>(local.ordinary_target_before, ordinary_contribution),
        );
        let ordinary_weight_contribution: [AB::Expr; D_EF] = ext_add_expr::<AB::Expr>(
            ext_scale_expr::<AB::Expr>(
                ext_mul_expr::<AB::Expr>(local.ordinary_power_before, local.term_weight_at_point),
                ordinary_selector.clone(),
            ),
            ext_scale_expr::<AB::Expr>(
                ext_mul_expr::<AB::Expr>(local.active_count_challenge, local.term_weight_at_point),
                AB::Expr::from(prep.is_active_count_term)
                    * AB::Expr::from(prep.active_count_height_scale),
            ),
        );
        assert_array_eq(
            &mut builder.when(enabled.clone()),
            local.ordinary_weight_after,
            ext_add_expr::<AB::Expr>(local.ordinary_weight_before, ordinary_weight_contribution),
        );
        let fingerprint_contribution: [AB::Expr; D_EF] =
            ext_mul_expr::<AB::Expr>(local.program_power_before, local.term_weight_at_point);
        assert_array_eq(
            &mut builder.when(enabled.clone()),
            local.fingerprint_weight_after,
            ext_add_expr::<AB::Expr>(
                local.fingerprint_weight_before,
                ext_scale_expr::<AB::Expr>(fingerprint_contribution, prep.is_program_term),
            ),
        );
        let term_scale: [AB::Expr; D_EF] = ext_add_expr::<AB::Expr>(
            ext_scale_expr::<AB::Expr>(local.ordinary_power_before, ordinary_selector),
            ext_add_expr::<AB::Expr>(
                ext_scale_expr::<AB::Expr>(
                    ext_mul_expr::<AB::Expr>(
                        local.program_mix_challenge,
                        local.program_power_before,
                    ),
                    prep.is_program_term,
                ),
                ext_scale_expr::<AB::Expr>(
                    local.active_count_challenge,
                    AB::Expr::from(prep.is_active_count_term)
                        * AB::Expr::from(prep.active_count_height_scale),
                ),
            ),
        );
        assert_array_eq(
            &mut builder.when(enabled.clone()),
            local.term_scale,
            term_scale,
        );

        let continuing =
            AB::Expr::from(next.active) * (AB::Expr::ONE - AB::Expr::from(prep.is_source_last));
        let mut transition_builder = builder.when_transition();
        let mut transition = transition_builder.when(continuing);
        for (next_values, local_values) in [
            (&next.source_forest_root[..], &local.source_forest_root[..]),
            (
                &next.segment_openings_digest[..],
                &local.segment_openings_digest[..],
            ),
            (&next.source_root[..], &local.source_root[..]),
            (&next.batching_challenge[..], &local.batching_challenge[..]),
            (
                &next.program_fingerprint[..],
                &local.program_fingerprint[..],
            ),
            (
                &next.program_mix_challenge[..],
                &local.program_mix_challenge[..],
            ),
            (
                &next.active_count_challenge[..],
                &local.active_count_challenge[..],
            ),
            (
                &next.one_shot_point.concat()[..],
                &local.one_shot_point.concat()[..],
            ),
        ] {
            for (&next_value, &local_value) in next_values.iter().zip(local_values) {
                transition.assert_eq(next_value, local_value);
            }
        }
        transition.assert_eq(next.range_start, local.range_start);
        transition.assert_eq(next.range_end, local.range_end);
        transition.assert_eq(next.claim_start_tidx, local.claim_start_tidx);
        assert_array_eq(
            &mut transition,
            next.ordinary_power_before,
            local.ordinary_power_after,
        );
        assert_array_eq(
            &mut transition,
            next.program_power_before,
            local.program_power_after,
        );
        assert_array_eq(
            &mut transition,
            next.ordinary_target_before,
            local.ordinary_target_after,
        );
        assert_array_eq(
            &mut transition,
            next.ordinary_weight_before,
            local.ordinary_weight_after,
        );
        assert_array_eq(
            &mut transition,
            next.fingerprint_weight_before,
            local.fingerprint_weight_after,
        );

        // Exact backend claim-observation encoding: tag/header, each retained
        // term in canonical order, and one final combined target. This is the
        // only semantic change from the former relay: the already constrained
        // values are sent directly to TranscriptBus at the same absolute
        // indices instead of crossing a second AIR first.
        macro_rules! observe_claim {
            ($index:expr, $value:expr, $gate:expr) => {{
                let observation_index: AB::Expr = $index;
                self.transcript_bus.observe_ext(
                    builder,
                    prep.proof_index,
                    AB::Expr::from(local.claim_start_tidx)
                        + observation_index * AB::Expr::from_usize(D_EF),
                    $value,
                    $gate,
                );
            }};
        }
        let mut header_index = 0usize;
        for &byte in DIRECT_OPENING_REDUCTION_TAG_V19 {
            observe_claim!(
                AB::Expr::from_usize(header_index),
                core::array::from_fn(|limb| {
                    if limb == 0 {
                        AB::Expr::from_u8(byte)
                    } else {
                        AB::Expr::ZERO
                    }
                }),
                first.clone()
            );
            header_index += 1;
        }
        for byte_index in 0..8 {
            observe_claim!(
                AB::Expr::from_usize(header_index),
                core::array::from_fn(|limb| {
                    if limb == 0 && byte_index == 0 {
                        AB::Expr::ONE
                    } else {
                        AB::Expr::ZERO
                    }
                }),
                first.clone()
            );
            header_index += 1;
        }
        for bytes in [prep.log_message_len_bytes, prep.term_count_bytes] {
            for byte in bytes {
                observe_claim!(
                    AB::Expr::from_usize(header_index),
                    core::array::from_fn(|limb| {
                        if limb == 0 {
                            byte.into()
                        } else {
                            AB::Expr::ZERO
                        }
                    }),
                    first.clone()
                );
                header_index += 1;
            }
        }
        let mut relative = 0usize;
        for bytes in [
            prep.block_start_bytes,
            prep.log_height_bytes,
            prep.l_skip_bytes,
            prep.rotation_bytes,
            prep.barycentric_len_bytes,
        ] {
            for byte in bytes {
                observe_claim!(
                    AB::Expr::from(prep.observation_start) + AB::Expr::from_usize(relative),
                    core::array::from_fn(|limb| {
                        if limb == 0 {
                            byte.into()
                        } else {
                            AB::Expr::ZERO
                        }
                    }),
                    enabled.clone()
                );
                relative += 1;
            }
        }
        for z in 0..MAX_SKIP_DOMAIN_V19 {
            let used = (0..=MAX_L_SKIP_V19).fold(AB::Expr::ZERO, |sum, l_skip| {
                sum + AB::Expr::from_bool(z < (1usize << l_skip))
                    * AB::Expr::from(prep.is_l_skip[l_skip])
            });
            observe_claim!(
                AB::Expr::from(prep.observation_start) + AB::Expr::from_usize(relative + z),
                local.barycentric[z].map(Into::into),
                enabled.clone() * used
            );
        }
        // The absolute folded-length offset depends on q, so emit each
        // candidate under its fixed l_skip selector.
        for l_skip in 0..=MAX_L_SKIP_V19 {
            let q = 1usize << l_skip;
            let gate = enabled.clone() * AB::Expr::from(prep.is_l_skip[l_skip]);
            for (byte_index, byte) in prep.folded_len_bytes.iter().enumerate() {
                observe_claim!(
                    AB::Expr::from(prep.observation_start)
                        + AB::Expr::from_usize(40 + q + byte_index),
                    core::array::from_fn(|limb| {
                        if limb == 0 {
                            (*byte).into()
                        } else {
                            AB::Expr::ZERO
                        }
                    }),
                    gate.clone()
                );
            }
            for folded_index in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
                observe_claim!(
                    AB::Expr::from(prep.observation_start)
                        + AB::Expr::from_usize(48 + q + folded_index),
                    core::array::from_fn(|limb| {
                        AB::Expr::from(prep.is_program_term)
                            * AB::Expr::from(prep.program_row_point[folded_index][limb])
                            + (AB::Expr::ONE - AB::Expr::from(prep.is_program_term))
                                * if folded_index + 1 < MAX_RAW_MESSAGE_POINT_LEN_V19 {
                                    AB::Expr::from(local.rs[folded_index + 1][limb])
                                } else {
                                    AB::Expr::ZERO
                                }
                    }),
                    gate.clone() * AB::Expr::from(prep.row_used[folded_index])
                );
            }
            observe_claim!(
                AB::Expr::from(prep.observation_start)
                    + AB::Expr::from_usize(48 + q)
                    + AB::Expr::from(prep.folded_len),
                local.term_scale.map(Into::into),
                gate
            );
        }
        let combined_target: [AB::Expr; D_EF] = ext_add_expr(
            local.ordinary_target_after,
            ext_mul_expr::<AB::Expr>(local.program_mix_challenge, local.program_fingerprint),
        );
        observe_claim!(
            AB::Expr::from(prep.observation_count) - AB::Expr::ONE,
            combined_target,
            last.clone()
        );
        self.round_start_bus.send(
            builder,
            OneShotRoundStartMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                shard_ordinal: prep.shard_ordinal.into(),
                round_start_tidx: AB::Expr::from(local.claim_start_tidx)
                    + AB::Expr::from(prep.observation_count) * AB::Expr::from_usize(D_EF),
            },
            last.clone(),
        );

        self.mapped_bus.send(
            builder,
            VerifiedMappedFunctionalMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                shard_ordinal: prep.shard_ordinal.into(),
                is_program: prep.is_program.into(),
                source_root: local.source_root.map(Into::into),
                range_start: local.range_start.into(),
                range_end: local.range_end.into(),
                functional_digest: prep.relation_digest.map(Into::into),
                point_len: prep.log_message_len.into(),
                point: local.one_shot_point.map(|point| point.map(Into::into)),
                ordinary_target: local.ordinary_target_after.map(Into::into),
                ordinary_weight_at_point: local.ordinary_weight_after.map(Into::into),
                program_fingerprint: local.program_fingerprint.map(Into::into),
                fingerprint_weight_at_point: local.fingerprint_weight_after.map(Into::into),
            },
            last,
        );
    }
}

/// Backend-neutral private values needed to connect one authenticated LogUp
/// column-opening family to its raw direct-source message. No digest, target,
/// or challenge in this record is authoritative by itself; every field is
/// consumed against a verifier-owned bus by [`MappedFunctionalTermAirV19`].
#[derive(Clone, Debug)]
pub struct MappedFunctionalSourceRecordV19 {
    pub proof_index: u32,
    pub segment_index: u32,
    pub shard_ordinal: u16,
    pub source_forest_root: [F; DIGEST_SIZE],
    pub segment_openings_digest: [F; DIGEST_SIZE],
    pub source_root: [F; DIGEST_SIZE],
    pub range_start: u32,
    pub range_end: u32,
    pub batching_challenge: EF,
    pub program_fingerprint: EF,
    pub program_mix_challenge: EF,
    /// Non-zero verifier-derived coefficient for the setup-fixed occupancy
    /// term. It is zero exactly when this source profile has no such term.
    pub active_count_challenge: EF,
    /// Exact `r_0, ..., r_n` exported by the partial LogUp verifier.
    pub logup_opening_point: Vec<EF>,
    /// Fiat--Shamir point produced by this source's one-shot reduction.
    pub one_shot_point: Vec<EF>,
    /// Common/cached column claims in the profile's fixed term order.
    pub dynamic_column_claims: Vec<EF>,
}

fn copy_ext_v19(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

fn eq_factor_v19(left: EF, right: EF) -> EF {
    (EF::ONE - left) * (EF::ONE - right) + left * right
}

/// Generate the exact mapped-functional trace in the same fixed source/term
/// order as [`MappedFunctionalTermAirV19::preprocessed_trace`].
pub fn generate_mapped_functional_trace_v19(
    profile: &DirectLogUpMappedFunctionalProfileV19,
    records: &[MappedFunctionalSourceRecordV19],
    claim_transcripts: &[OneShotClaimTranscriptRecordV19],
) -> Result<RowMajorMatrix<F>, &'static str> {
    generate_mapped_functional_trace_inner_v19(profile, records, claim_transcripts, None)
}

/// Fixed-capacity trace generator. `active_child_counts` is witness input, not
/// authority: the AIR equates each value to the count term over the committed
/// source and to the independently range-constrained C2 certificate.
pub fn generate_mapped_functional_trace_fixed_capacity_v4(
    profile: &DirectLogUpMappedFunctionalProfileV19,
    records: &[MappedFunctionalSourceRecordV19],
    claim_transcripts: &[OneShotClaimTranscriptRecordV19],
    active_child_counts: &[u8],
) -> Result<RowMajorMatrix<F>, &'static str> {
    if !profile.runtime_capacity_v4() {
        return Err("runtime counts require a fixed-capacity mapped profile");
    }
    generate_mapped_functional_trace_inner_v19(
        profile,
        records,
        claim_transcripts,
        Some(active_child_counts),
    )
}

fn generate_mapped_functional_trace_inner_v19(
    profile: &DirectLogUpMappedFunctionalProfileV19,
    records: &[MappedFunctionalSourceRecordV19],
    claim_transcripts: &[OneShotClaimTranscriptRecordV19],
    runtime_active_child_counts: Option<&[u8]>,
) -> Result<RowMajorMatrix<F>, &'static str> {
    let source_count = profile
        .terms
        .iter()
        .filter(|term| term.term_ordinal == 0)
        .count();
    let active_source_count = if profile.runtime_capacity_v4() {
        if source_count != 4
            || records.is_empty()
            || records.len() > source_count
            || claim_transcripts.len() != records.len()
            || runtime_active_child_counts.is_none_or(|counts| counts.len() != records.len())
        {
            return Err("fixed-capacity mapped-functional source count");
        }
        records.len()
    } else {
        if runtime_active_child_counts.is_some()
            || records.len() != source_count
            || claim_transcripts.len() != source_count
        {
            return Err("mapped-functional source count");
        }
        source_count
    };
    if runtime_active_child_counts.is_some_and(|counts| {
        counts.iter().enumerate().any(|(index, &count)| {
            count == 0
                || usize::from(count) > VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2
                || (index + 1 != counts.len()
                    && usize::from(count) != VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2)
        })
    }) {
        return Err("mapped-functional source count");
    }
    let width = MappedFunctionalTermColsV19::<F>::width();
    let height = profile.terms.len().next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    let mut source_index = 0usize;
    let mut row_index = 0usize;
    while source_index < active_source_count {
        let first_plan = &profile.terms[row_index];
        let term_count = first_plan.term_count as usize;
        let plans = &profile.terms[row_index..row_index + term_count];
        let record = records
            .get(source_index)
            .ok_or("missing mapped-functional source")?;
        let claim_transcript = claim_transcripts
            .get(source_index)
            .ok_or("missing mapped-functional claim transcript")?;
        let has_active_count = plans.iter().any(|plan| plan.is_active_count_term);
        let runtime_active_count = runtime_active_child_counts.map(|counts| counts[source_index]);
        if record.proof_index != first_plan.proof_index
            || record.segment_index != first_plan.segment_index
            || record.shard_ordinal != first_plan.shard_ordinal
            || record.range_start >= record.range_end
            || record.one_shot_point.len() != first_plan.log_message_len as usize
            || record.dynamic_column_claims.len()
                != plans
                    .iter()
                    .filter(|plan| !plan.is_program_term && !plan.is_active_count_term)
                    .count()
            || (!first_plan.is_program
                && (record.program_fingerprint != EF::ZERO
                    || record.program_mix_challenge != EF::ZERO))
            || (first_plan.is_program && record.program_mix_challenge == EF::ZERO)
            || (has_active_count != (record.active_count_challenge != EF::ZERO))
            || claim_transcript.proof_index != first_plan.proof_index
            || claim_transcript.segment_index != first_plan.segment_index
            || claim_transcript.shard_ordinal != first_plan.shard_ordinal
            || claim_transcript.observations.len() != first_plan.observation_count as usize
        {
            return Err("mapped-functional source shape");
        }

        let mut ordinary_power = EF::ONE;
        let mut program_power = EF::ONE;
        let mut ordinary_target = EF::ZERO;
        let mut ordinary_weight = EF::ZERO;
        let mut fingerprint_weight = EF::ZERO;
        let mut claim_index = 0usize;
        for (term_index, plan) in plans.iter().enumerate() {
            let cols: &mut MappedFunctionalTermColsV19<F> = values
                [(row_index + term_index) * width..(row_index + term_index + 1) * width]
                .borrow_mut();
            cols.active = F::ONE;
            cols.claim_start_tidx = F::from_u32(claim_transcript.start_tidx);
            cols.source_forest_root = record.source_forest_root;
            cols.segment_openings_digest = record.segment_openings_digest;
            cols.source_root = record.source_root;
            cols.range_start = F::from_u32(record.range_start);
            cols.range_end = F::from_u32(record.range_end);
            copy_ext_v19(&mut cols.batching_challenge, record.batching_challenge);
            copy_ext_v19(&mut cols.program_fingerprint, record.program_fingerprint);
            copy_ext_v19(
                &mut cols.program_mix_challenge,
                record.program_mix_challenge,
            );
            copy_ext_v19(
                &mut cols.active_count_challenge,
                record.active_count_challenge,
            );
            for (target, &point) in cols.one_shot_point.iter_mut().zip(&record.one_shot_point) {
                copy_ext_v19(target, point);
            }

            let folded_len = usize::from(plan.log_height).saturating_sub(plan.l_skip as usize);
            let effective_l_skip = plan.l_skip.min(plan.log_height) as usize;
            let prefix_len = (plan.log_message_len - plan.log_height) as usize;
            let (barycentric, row_point) = if plan.is_program_term {
                (vec![EF::ONE], plan.program_row_point.clone())
            } else if plan.is_active_count_term {
                copy_ext_v19(&mut cols.rs[0], EF::ZERO);
                // l_skip=0 still executes the single PLE denominator check:
                // (r0 - 1) * inverse = 1 with r0=0.
                copy_ext_v19(&mut cols.denominator_inverses[0], -EF::ONE);
                for index in 0..folded_len {
                    copy_ext_v19(&mut cols.rs[index + 1], EF::TWO.inverse());
                }
                (vec![EF::ONE], vec![EF::TWO.inverse(); folded_len])
            } else {
                if record.logup_opening_point.len() < folded_len + 1 {
                    return Err("mapped-functional LogUp point length");
                }
                copy_ext_v19(&mut cols.rs[0], record.logup_opening_point[0]);
                for index in 0..folded_len {
                    copy_ext_v19(
                        &mut cols.rs[index + 1],
                        record.logup_opening_point[folded_len - index],
                    );
                }
                let r0 = record.logup_opening_point[0];
                let q = 1usize << plan.l_skip;
                let omega = F::two_adic_generator(plan.l_skip as usize);
                let scaling = (r0.exp_u64(q as u64) - EF::ONE) * EF::from_usize(q).inverse();
                let mut omega_power = F::ONE;
                let mut barycentric = Vec::with_capacity(q);
                for z in 0..q {
                    let denominator = r0 - EF::from(omega_power);
                    if denominator == EF::ZERO {
                        return Err("mapped-functional PLE denominator");
                    }
                    let inverse = denominator.inverse();
                    copy_ext_v19(&mut cols.denominator_inverses[z], inverse);
                    barycentric.push(EF::from(omega_power) * inverse * scaling);
                    omega_power *= omega;
                }
                (
                    barycentric,
                    cols.rs[1..=folded_len]
                        .iter()
                        .map(|value| {
                            EF::from_basis_coefficients_slice(value).expect("EF4 mapped row point")
                        })
                        .collect(),
                )
            };
            for (target, &weight) in cols.barycentric.iter_mut().zip(&barycentric) {
                copy_ext_v19(target, weight);
            }
            for (target, &point) in cols
                .row_one_shot_point
                .iter_mut()
                .zip(&record.one_shot_point[prefix_len..prefix_len + folded_len])
            {
                copy_ext_v19(target, point);
            }
            for (target, &point) in cols.low_one_shot_point.iter_mut().zip(
                &record.one_shot_point
                    [prefix_len + folded_len..prefix_len + folded_len + effective_l_skip],
            ) {
                copy_ext_v19(target, point);
            }

            if plan.is_active_count_term {
                copy_ext_v19(
                    &mut cols.claim,
                    EF::from(F::from_u8(
                        runtime_active_count.unwrap_or(plan.active_count_expected),
                    )),
                );
                copy_ext_v19(&mut cols.r0_powers[0], EF::ZERO);
            } else if !plan.is_program_term {
                let mut power = record.logup_opening_point[0];
                copy_ext_v19(&mut cols.r0_powers[0], power);
                for index in 0..plan.l_skip as usize {
                    power *= power;
                    copy_ext_v19(&mut cols.r0_powers[index + 1], power);
                }
                copy_ext_v19(&mut cols.claim, record.dynamic_column_claims[claim_index]);
                claim_index += 1;
            }

            let block_index = (plan.block_start as usize) >> plan.log_height;
            let mut prefix_product = EF::ONE;
            copy_ext_v19(&mut cols.prefix_products[0], prefix_product);
            for index in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
                if index < prefix_len {
                    let bit = ((block_index >> (prefix_len - 1 - index)) & 1) == 1;
                    let point = record.one_shot_point[index];
                    prefix_product *= if bit { point } else { EF::ONE - point };
                }
                copy_ext_v19(&mut cols.prefix_products[index + 1], prefix_product);
            }

            let mut row_product = EF::ONE;
            copy_ext_v19(&mut cols.row_products[0], row_product);
            for index in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
                if index < folded_len {
                    row_product *=
                        eq_factor_v19(row_point[index], record.one_shot_point[prefix_len + index]);
                }
                copy_ext_v19(&mut cols.row_products[index + 1], row_product);
            }

            let mut dp_zero = EF::ZERO;
            let mut dp_one = EF::ONE;
            copy_ext_v19(&mut cols.shift_dp_zero[0], dp_zero);
            copy_ext_v19(&mut cols.shift_dp_one[0], dp_one);
            for step in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
                if step < folded_len {
                    let coordinate = folded_len - 1 - step;
                    let a = row_point[coordinate];
                    let p = record.one_shot_point[prefix_len + coordinate];
                    let next_zero = dp_zero * eq_factor_v19(a, p) + dp_one * (EF::ONE - a) * p;
                    let next_one = dp_one * a * (EF::ONE - p);
                    dp_zero = next_zero;
                    dp_one = next_one;
                }
                copy_ext_v19(&mut cols.shift_dp_zero[step + 1], dp_zero);
                copy_ext_v19(&mut cols.shift_dp_one[step + 1], dp_one);
            }

            let mut low_zero = EF::ONE;
            copy_ext_v19(&mut cols.low_zero_products[0], low_zero);
            for index in 0..MAX_L_SKIP_V19 {
                if index < effective_l_skip {
                    low_zero *= EF::ONE - record.one_shot_point[prefix_len + folded_len + index];
                }
                copy_ext_v19(&mut cols.low_zero_products[index + 1], low_zero);
            }
            let protocol_q = 1usize << plan.l_skip;
            let q = 1usize << effective_l_skip;
            let mut current = vec![EF::ZERO; q];
            for (source, &weight) in barycentric.iter().take(protocol_q).enumerate() {
                current[source & (q - 1)] += weight;
            }
            let mut shifted = vec![EF::ZERO; q];
            if plan.log_height < plan.l_skip {
                for (source, &weight) in barycentric.iter().take(protocol_q).enumerate() {
                    shifted[(source + 1) & (q - 1)] += weight;
                }
            } else if q > 1 {
                shifted[1..].copy_from_slice(&barycentric[..q - 1]);
            }
            for z in 0..q {
                copy_ext_v19(&mut cols.low_fold_current[z], current[z]);
                copy_ext_v19(&mut cols.low_fold_shifted[z], shifted[z]);
            }
            let mut next_offset = q;
            let mut len = q;
            for level in 0..effective_l_skip {
                // Adjacent-pair folds consume the low physical bit first,
                // whereas `evaluate_mle` consumes point coordinates from the
                // high bit. Reverse this suffix to implement the same MLE.
                let challenge =
                    record.one_shot_point[prefix_len + folded_len + effective_l_skip - 1 - level];
                let mut next_current = Vec::with_capacity(len / 2);
                let mut next_shifted = Vec::with_capacity(len / 2);
                for node in 0..len / 2 {
                    let current_value =
                        current[2 * node] + challenge * (current[2 * node + 1] - current[2 * node]);
                    let shifted_value =
                        shifted[2 * node] + challenge * (shifted[2 * node + 1] - shifted[2 * node]);
                    copy_ext_v19(
                        &mut cols.low_fold_current[next_offset + node],
                        current_value,
                    );
                    copy_ext_v19(
                        &mut cols.low_fold_shifted[next_offset + node],
                        shifted_value,
                    );
                    next_current.push(current_value);
                    next_shifted.push(shifted_value);
                }
                current = next_current;
                shifted = next_shifted;
                next_offset += len / 2;
                len /= 2;
            }
            let low_current = current[0];
            let low_shifted = shifted[0];
            let shift_correlation = dp_zero + dp_one;
            let current_inner = row_product * low_current;
            let next_inner = if plan.log_height < plan.l_skip {
                row_product * low_shifted
            } else {
                row_product * low_shifted
                    + barycentric[protocol_q - 1] * low_zero * shift_correlation
            };
            let rotation_inner = if plan.is_rot {
                next_inner
            } else {
                current_inner
            };
            let term_weight = prefix_product * rotation_inner;
            copy_ext_v19(&mut cols.rotation_inner, rotation_inner);
            copy_ext_v19(&mut cols.term_weight_at_point, term_weight);

            copy_ext_v19(&mut cols.ordinary_power_before, ordinary_power);
            copy_ext_v19(&mut cols.program_power_before, program_power);
            copy_ext_v19(&mut cols.ordinary_target_before, ordinary_target);
            copy_ext_v19(&mut cols.ordinary_weight_before, ordinary_weight);
            copy_ext_v19(&mut cols.fingerprint_weight_before, fingerprint_weight);
            let term_scale = if plan.is_program_term {
                let scale = record.program_mix_challenge * program_power;
                fingerprint_weight += program_power * term_weight;
                program_power *= plan.program_column_challenge;
                scale
            } else if plan.is_active_count_term {
                let coefficient = record.active_count_challenge;
                let scale = coefficient * EF::from_usize(1usize << plan.log_height);
                ordinary_target += coefficient
                    * EF::from(F::from_u8(
                        runtime_active_count.unwrap_or(plan.active_count_expected),
                    ));
                ordinary_weight += scale * term_weight;
                scale
            } else {
                let scale = ordinary_power;
                ordinary_target += ordinary_power * record.dynamic_column_claims[claim_index - 1];
                ordinary_weight += ordinary_power * term_weight;
                ordinary_power *= record.batching_challenge;
                scale
            };
            copy_ext_v19(&mut cols.term_scale, term_scale);
            copy_ext_v19(&mut cols.ordinary_power_after, ordinary_power);
            copy_ext_v19(&mut cols.program_power_after, program_power);
            copy_ext_v19(&mut cols.ordinary_target_after, ordinary_target);
            copy_ext_v19(&mut cols.ordinary_weight_after, ordinary_weight);
            copy_ext_v19(&mut cols.fingerprint_weight_after, fingerprint_weight);
        }
        row_index += term_count;
        source_index += 1;
    }
    Ok(RowMajorMatrix::new(values, width))
}

/// Backend-neutral retained segment proof consumed by the production
/// protocol-v19 context generator.
///
/// Commitment values inside `proof_material` are deliberately ignored. The
/// generator reconstructs trace presence, heights, cached-commitment vector
/// lengths and public values from these explicit shape vectors plus the child
/// VK, then inserts zero commitment placeholders. In the rebased LogUpOnly
/// proof shape commitments are not exported and therefore are not authority.
#[derive(Clone, Debug)]
pub struct DirectLogUpOnlyRetainedProofRecordV19 {
    pub proof_index: u32,
    pub trace_air_ids: Vec<usize>,
    pub n_per_trace: Vec<isize>,
    pub need_rot_per_trace: Vec<bool>,
    pub trace_public_values: Vec<Vec<F>>,
    pub proof_material: RetainedLogUpOnlyProof,
}

/// Complete private input for one production context-generation call.
/// Every shard-indexed vector is in canonical `(segment, shard_ordinal)`
/// order. Reduction records remain logically ordered here even though the
/// generator physically groups their traces by `log_message_len`.
#[derive(Clone, Debug)]
pub struct DirectLogUpOnlyCompositeRecordsV19 {
    pub retained_proofs: Vec<DirectLogUpOnlyRetainedProofRecordV19>,
    pub sources: Vec<DirectLogUpSegmentSourceRecordV19>,
    pub claim_derivations: Vec<LogUpClaimDerivationRecordV19>,
    pub mapped_sources: Vec<MappedFunctionalSourceRecordV19>,
    pub claim_transcripts: Vec<OneShotClaimTranscriptRecordV19>,
    pub reductions: Vec<OneShotReductionRecordV19>,
    pub boundary_shards: Vec<LogUpSwirlBoundaryShardRecordV19>,
}

/// Complete Poseidon request packet for the composite's one logical owner.
/// The owner and input pair must retain this association when an outer
/// assembly concatenates it with grouped VACC or History owners.
#[derive(Clone, Debug)]
pub struct DirectLogUpOnlyCompositePoseidonPacketV19 {
    pub owner: Poseidon2BusOwner,
    pub permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
}

impl DirectLogUpOnlyCompositePoseidonPacketV19 {
    #[must_use]
    pub fn grouped_input(&self) -> (Vec<[F; POSEIDON2_WIDTH]>, Vec<[F; POSEIDON2_WIDTH]>) {
        (
            self.permutation_inputs.clone(),
            self.compression_inputs.clone(),
        )
    }

    #[must_use]
    pub fn grouped_inputs(&self) -> Poseidon2MultibusInputs {
        vec![self.grouped_input()]
    }
}

/// CPU contexts in exact
/// [`DirectLogUpOnlyVerifierModuleV19::airs_without_poseidon`] order plus the
/// omitted owner's complete Poseidon packet.
pub struct DirectLogUpOnlyCompositePacketContextsV19<SC: StarkProtocolConfig<F = F>> {
    pub contexts: Vec<AirProvingContext<CpuBackend<SC>>>,
    pub poseidon: DirectLogUpOnlyCompositePoseidonPacketV19,
    /// Canonical positive-producer records derived from the verifier replay.
    /// The enclosing History composition must instantiate
    /// `LogUpOnlyProducerAirV19` with these records; host-supplied checkpoint
    /// digests are not authority.
    pub logup_producers: Vec<LogUpOnlyProducerRecordV19>,
}

/// One context in the mixed CUDA packet, in exact
/// [`DirectLogUpOnlyVerifierModuleV19::airs_without_poseidon`] order.
///
/// The standard LogUp-only BatchConstraint, ProofShape, GKR and primitive
/// tables are always [`Self::Device`]. The only host-resident partial-verifier
/// contexts are the resumed transcript traces and their compact rebase
/// adapter. Composite-specific v19 traces are explicitly tagged separately.
#[cfg(feature = "cuda")]
pub enum DirectLogUpOnlyCompositeCudaContextV19 {
    Device(AirProvingContext<openvm_cuda_backend::GpuBackend>),
    HostExtendedTranscript(AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>),
    HostRebasedTranscriptAdapter(AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>),
    HostV19(AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>),
}

/// Mixed CUDA/CPU contexts plus the omitted shared-Poseidon owner packet.
/// Device variants remain resident until the caller proves or explicitly
/// transports them; this API performs no device-to-host transfer.
#[cfg(feature = "cuda")]
pub struct DirectLogUpOnlyCompositeCudaPacketV19 {
    pub contexts: Vec<DirectLogUpOnlyCompositeCudaContextV19>,
    pub poseidon: DirectLogUpOnlyCompositePoseidonPacketV19,
    pub logup_producers: Vec<LogUpOnlyProducerRecordV19>,
}

struct DirectLogUpOnlyCompositePreparedV19<SC: StarkProtocolConfig<F = F>> {
    shaped_proofs: Vec<Proof<BabyBearPoseidon2Config>>,
    preflights: Vec<Preflight>,
    extended_logs: Vec<TranscriptLog<F, [F; POSEIDON2_WIDTH]>>,
    checkpoint_targets: Vec<[usize; 2]>,
    additional_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    additional_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    tail_contexts: Vec<AirProvingContext<CpuBackend<SC>>>,
    logup_producers: Vec<LogUpOnlyProducerRecordV19>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DirectLogUpOnlyCompositeContextErrorV19 {
    Shape(&'static str),
    Source(&'static str),
    LogUpReplay,
    Transcript(&'static str),
    Trace(&'static str),
    PartialVerifier,
}

impl core::fmt::Display for DirectLogUpOnlyCompositeContextErrorV19 {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Shape(message) => {
                write!(formatter, "invalid production composite shape: {message}")
            }
            Self::Source(message) => write!(formatter, "invalid source provider: {message}"),
            Self::LogUpReplay => formatter.write_str("retained LogUp proof replay failed"),
            Self::Transcript(message) => {
                write!(formatter, "canonical transcript record mismatch: {message}")
            }
            Self::Trace(message) => {
                write!(formatter, "component trace generation failed: {message}")
            }
            Self::PartialVerifier => {
                formatter.write_str("partial verifier context generation failed")
            }
        }
    }
}

impl std::error::Error for DirectLogUpOnlyCompositeContextErrorV19 {}

/// Verifier-owned fixed profile.  The mapped-functional relation profile is
/// added below; keeping this skeleton public lets SDK conversion remain a
/// separate layer without exposing any test authority AIR.
pub struct DirectLogUpOnlyVerifierModuleV19<const MAX_NUM_PROOFS: usize> {
    pub child_vk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
    pub source_profile: DirectLogUpSourceManifestProfileV19,
    pub partial: LogUpOnlyPartialVerifier<MAX_NUM_PROOFS>,
    pub prefix_transcript: LogUpOnlyPrefixTranscript,
    pub source_manifest_air: DirectLogUpSourceManifestAirV19,
    pub source_instance_air: DirectLogUpSourceInstanceAirV19,
    pub source_forest_node_air: DirectLogUpSourceForestNodeAirV19,
    pub prefix_bridge_air: PrefixCheckpointBridgeAirV19,
    pub opening_point_fanout_air: OpeningPointFanoutAirV19,
    pub forest_fanout_air: ForestLeafFanoutAirV19,
    pub endpoint_bridge_air: LogUpEndpointBridgeAirV19,
    pub column_opening_observation_air: ColumnOpeningObservationAirV19,
    pub claim_derivation_air: LogUpClaimDerivationAirV19,
    pub active_count_transcript_air: Option<ActiveChildCountTranscriptAirV19>,
    /// Runtime-prefix counterpart used only by the capacity-four HLeaf. It is
    /// mutually exclusive with `active_count_transcript_air`.
    pub active_count_transcript_air_v4: Option<ActiveChildCountTranscriptAirV4>,
    pub mapped_functional_air: MappedFunctionalTermAirV19,
    pub reduction_airs: Vec<OneShotReductionAirV19>,
    pub finalization_air: LogUpFinalizationAirV19,
    pub boundary_air: LogUpSwirlBoundaryAirV19,
    pub history_buses: DirectLogUpOnlyHistoryBusesV19,
    pub fixed_boundary_inputs: FixedMultiAirBoundaryInputsV19,
    pub source_instance_bus: SourceInstanceDigestBusV19,
    pub fixed_public_values_bus: FixedSourcePublicValueBusV19,
    pub fixed_multi_air: bool,
    pub next_bus_idx: BusIndex,
}

impl<const MAX_NUM_PROOFS: usize> DirectLogUpOnlyVerifierModuleV19<MAX_NUM_PROOFS> {
    /// Construct the verifier-owned production composition. The segment plan,
    /// relation registry, child VK, and maximum proof capacity are all fixed
    /// here and therefore become part of the resulting MultiSTARK VK.
    pub fn new(
        child_vk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
        source_profile: DirectLogUpSourceManifestProfileV19,
        entries: Vec<DirectLogUpMappedRegistryEntryV19>,
        segment_shard_ids: &[Vec<u32>],
    ) -> Result<Self, &'static str> {
        Self::new_with_segment_offset(child_vk, source_profile, entries, segment_shard_ids, 0)
    }

    pub fn new_with_segment_offset(
        child_vk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
        source_profile: DirectLogUpSourceManifestProfileV19,
        entries: Vec<DirectLogUpMappedRegistryEntryV19>,
        segment_shard_ids: &[Vec<u32>],
        segment_start: u32,
    ) -> Result<Self, &'static str> {
        if child_vk.inner.params.l_skip != OPENVM_DIRECT_LOGUP_L_SKIP_V19 {
            return Err("protocol-v19 DirectLogUpOnly requires child VK l_skip == 4");
        }
        if segment_shard_ids.is_empty()
            || segment_shard_ids.len() > MAX_NUM_PROOFS
            || source_profile.app_vk_digest != child_vk.pre_hash
        {
            return Err("invalid DirectLogUpOnly verifier profile");
        }
        if source_profile.segment_start() != segment_start {
            return Err("source/mapped segment offset mismatch");
        }
        let mapped_profile = DirectLogUpMappedFunctionalProfileV19::new_with_segment_offset(
            &child_vk,
            &source_profile,
            entries,
            segment_shard_ids,
            segment_start,
        )?;
        Self::new_from_mapped_profile(child_vk, source_profile, mapped_profile, None, None, false)
    }

    /// Construct the global selector-free verifier source.  The caller must
    /// provide exactly one fixed relation source per batch in
    /// `source_profile`; its instance digest is supplied by the certified
    /// VACC/replay bridge through [`Self::fixed_public_values_bus`].
    pub fn new_fixed_multi_air(
        child_vk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
        source_profile: DirectLogUpSourceManifestProfileV19,
        config: FixedMultiAirMappedFunctionalConfigV19,
        segment_start: u32,
    ) -> Result<Self, &'static str> {
        if config.active_child_counts.len() > MAX_NUM_PROOFS
            || source_profile.app_vk_digest != child_vk.pre_hash
        {
            return Err("invalid fixed multi-AIR verifier profile");
        }
        let mapped_profile = DirectLogUpMappedFunctionalProfileV19::new_fixed_multi_air(
            &child_vk,
            &source_profile,
            &config,
            segment_start,
        )?;
        let active_count_profile = ActiveChildCountTranscriptProfileV19 {
            segment_start,
            relation_digest: config.relation_digest,
            active_child_counts: config.active_child_counts,
            vm_pvs_air_id: config.vm_pvs_air_id,
            is_valid_common_main_column: config.is_valid_common_main_column,
            is_valid_message_block_start: u64::from(config.is_valid_message_block_start),
            vm_pvs_log_height: config.vm_pvs_log_height,
            trace_heights: config.profile_trace_heights,
        };
        Self::new_from_mapped_profile(
            child_vk,
            source_profile,
            mapped_profile,
            Some(active_count_profile),
            None,
            true,
        )
    }

    /// Construct the capacity-four HLeaf verifier relation. The four local
    /// proof slots, child VK, source relation, mapped regions and code shape
    /// are setup data. Runtime occupancy and active-child counts are absent
    /// from every profile and preprocessed matrix.
    ///
    /// This constructor deliberately rebuilds both setup anchors:
    ///
    /// - every direct-AIR region's constraint relation and dimensions are reconstructed from the
    ///   supplied child VK and compared with the caller's relation index (fixed preprocessing
    ///   remains authenticated by the setup PCS authority); and
    /// - the canonical fixed-capacity source profile is reconstructed from the fixed relation/code
    ///   dimensions and compared through the exact source AIR preprocessing.
    ///
    /// Consequently a caller cannot combine a valid child VK with relations
    /// or source dimensions taken from another verifier key.
    pub fn new_fixed_capacity_v4(
        child_vk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
        source_profile: DirectLogUpSourceManifestProfileV19,
        config: FixedMultiAirMappedFunctionalConfigV4,
    ) -> Result<Self, &'static str> {
        // `MAX_NUM_PROOFS` is the History leaf's transition capacity. It was
        // previously compared with the unrelated number of SWIRL child proofs
        // verified inside each source only because both happened to be four.
        // Keep the HLeaf source inventory fixed to its own const generic; the
        // segment-child capacity is authenticated separately by the active-
        // count profile and fixed verifier VK.
        if source_profile.runtime_capacity_v4() != Some(MAX_NUM_PROOFS)
            || source_profile.source_count() != MAX_NUM_PROOFS
            || source_profile.segment_start() != 0
        {
            return Err("fixed-capacity source manifest shape");
        }
        if source_profile.app_vk_digest != child_vk.pre_hash {
            return Err("fixed-capacity source child VK");
        }
        if source_profile.setup_pcs_source_manifest_bus_v3().is_some() {
            return Err("fixed-capacity source manifest already attached to setup PCS");
        }
        if config.source_public_values_len == 0
            || config.active_count_profile_digest == [F::ZERO; DIGEST_SIZE]
        {
            return Err("fixed-capacity explicit assignment is empty");
        }
        if config.regions.len() != child_vk.inner.per_air.len() {
            return Err("fixed-capacity region count differs from child VK");
        }
        // `config.log_codeword_len` authenticates the outer WARP source
        // commitment. The child VK parameters authenticate the inner SWIRL
        // proof being verified. These are two different PCS layers and may
        // deliberately use different rates. Their respective canonical
        // profiles are reconstructed independently below.

        let child_air_count = u32::try_from(child_vk.inner.per_air.len())
            .map_err(|_| "fixed-capacity child AIR count")?;
        let padded_message_len = 1usize
            .checked_shl(u32::from(config.log_message_len))
            .ok_or("fixed-capacity message length")?;
        let mut expected_air_ids = config
            .regions
            .iter()
            .map(|region| region.air_id)
            .collect::<Vec<_>>();
        expected_air_ids.sort_unstable();
        if expected_air_ids != (0..child_air_count).collect::<Vec<_>>() {
            return Err("fixed-capacity regions do not cover the child VK");
        }
        let mut canonical_regions = config.regions.iter().collect::<Vec<_>>();
        canonical_regions.sort_by_key(|region| {
            (
                core::cmp::Reverse(region.relation.description().shard_key.log_height),
                region.air_id,
            )
        });
        let mut next_message_start = 0usize;
        for region in canonical_regions {
            if region.message_start as usize != next_message_start {
                return Err("noncanonical fixed-capacity message placement");
            }
            next_message_start = next_message_start
                .checked_add(region.relation.raw_witness_len())
                .ok_or("fixed-capacity message placement overflow")?;
        }
        if next_message_start
            .checked_next_power_of_two()
            .ok_or("fixed-capacity padded message length")?
            != padded_message_len
        {
            return Err("fixed-capacity global message dimensions mismatch");
        }

        let vm_region = config
            .regions
            .iter()
            .find(|region| region.air_id == config.vm_pvs_air_id)
            .ok_or("fixed-capacity VmPvs AIR outside relation")?;
        let vm_key = &vm_region.relation.description().shard_key;
        if config.vm_pvs_log_height != vm_key.log_height
            || config.is_valid_common_main_column >= vm_key.trace_layout.common_main_width
        {
            return Err("fixed-capacity VmPvs dimensions mismatch");
        }
        let vm_height = 1u32
            .checked_shl(u32::from(config.vm_pvs_log_height))
            .ok_or("fixed-capacity VmPvs height")?;
        let cached_start = vm_key
            .trace_layout
            .cached_main_widths
            .iter()
            .copied()
            .sum::<u32>()
            .checked_mul(vm_height)
            .ok_or("fixed-capacity VmPvs message placement")?;
        let expected_is_valid_start = vm_region
            .message_start
            .checked_add(cached_start)
            .and_then(|start| {
                config
                    .is_valid_common_main_column
                    .checked_mul(vm_height)
                    .and_then(|offset| start.checked_add(offset))
            })
            .ok_or("fixed-capacity VmPvs message placement")?;
        if config.is_valid_message_block_start != expected_is_valid_start {
            return Err("fixed-capacity VmPvs column placement mismatch");
        }

        // Rebuild every direct relation from the actual child VK. Merely
        // comparing widths is insufficient: a relation built from another
        // constraint DAG can have the same dimensions.
        let canonical_config =
            BabyBearPoseidon2Config::default_from_params(child_vk.inner.params.clone());
        for region in config.regions.iter() {
            let relation = region.relation.description();
            let key = &relation.shard_key;
            let air_vk = child_vk
                .inner
                .per_air
                .get(region.air_id as usize)
                .ok_or("fixed-capacity mapped AIR outside child VK")?;
            let rebuilt = DirectAirPesatIndex::from_verifying_key(
                canonical_config.hasher(),
                child_vk.pre_hash,
                region.air_id as usize,
                usize::from(key.log_height),
                air_vk,
                region.relation.fixed_trace().cloned(),
                key.public_schema.clone(),
                key.code_class.clone(),
            )
            .map_err(|_| "fixed-capacity relation differs from child VK")?;
            if region.air_id != key.air_id
                || rebuilt.description() != relation
                || rebuilt.canonical_description_bytes()
                    != region.relation.canonical_description_bytes()
            {
                return Err("fixed-capacity relation differs from child VK");
            }
        }

        let expected_entry = DirectLogUpSourceEntryProfileV19::fixed_multi_air(
            config.relation_digest,
            config.log_message_len,
            config.log_codeword_len,
            config.source_public_values_len,
        )?;
        let expected_source_profile = DirectLogUpSourceManifestProfileV19::new_fixed_capacity_v4(
            child_vk.pre_hash,
            source_profile.registry_digest,
            expected_entry,
        )?;
        let mapped_profile = DirectLogUpMappedFunctionalProfileV19::new_fixed_capacity_v4(
            &child_vk,
            &source_profile,
            &config,
        )?;
        let active_count_profile = VerifierWarpActiveCountProfileV4 {
            relation_digest: config.relation_digest,
            // This is the canonical block-wide profile identity also carried
            // by C2 and the History statement.  Using the source-registry
            // digest here confuses two setup namespaces and makes genuine
            // receipts fail the independent manifest adapter.
            profile_digest: config.active_count_profile_digest,
            batch_arity: VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 as u64,
            trace_heights: config
                .profile_trace_heights
                .iter()
                .copied()
                .map(u64::from)
                .collect::<Vec<_>>()
                .into(),
            vm_pvs_air_id: config.vm_pvs_air_id,
            is_valid_common_main_column: config.is_valid_common_main_column,
            is_valid_message_block_start: u64::from(config.is_valid_message_block_start),
            log_height: config.vm_pvs_log_height,
            log_message_len: config.log_message_len,
        };
        active_count_profile.validate()?;

        let module = Self::new_from_mapped_profile(
            child_vk,
            source_profile,
            mapped_profile,
            None,
            Some(active_count_profile),
            true,
        )?;

        // Compare the actual caller profile with the canonical profile using
        // both source AIRs whose preprocessing contains the complete entry
        // metadata, including relation, dimensions and explicit length.
        let mut expected_manifest_air = module.source_manifest_air.clone();
        expected_manifest_air.profile = expected_source_profile.clone();
        let mut expected_instance_air = module.source_instance_air.clone();
        expected_instance_air.profile = expected_source_profile;
        if BaseAir::<F>::preprocessed_trace(&module.source_manifest_air)
            != BaseAir::<F>::preprocessed_trace(&expected_manifest_air)
            || BaseAir::<F>::preprocessed_trace(&module.source_instance_air)
                != BaseAir::<F>::preprocessed_trace(&expected_instance_air)
        {
            return Err("fixed-capacity source relation or dimensions mismatch");
        }
        Ok(module)
    }

    fn new_from_mapped_profile(
        child_vk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
        source_profile: DirectLogUpSourceManifestProfileV19,
        mapped_profile: DirectLogUpMappedFunctionalProfileV19,
        active_count_profile: Option<ActiveChildCountTranscriptProfileV19>,
        active_count_profile_v4: Option<VerifierWarpActiveCountProfileV4>,
        fixed_multi_air: bool,
    ) -> Result<Self, &'static str> {
        if active_count_profile.is_some() && active_count_profile_v4.is_some()
            || mapped_profile.runtime_capacity_v4() != active_count_profile_v4.is_some()
        {
            return Err("inconsistent active-count transcript profile");
        }
        let mut partial = LogUpOnlyPartialVerifier::<MAX_NUM_PROOFS>::new(&child_vk, false);
        if fixed_multi_air {
            let expected_dag_commit = partial
                .cached_trace_record(&child_vk)
                .dag_commit_info
                .ok_or("fixed multi-AIR verifier is missing its child-VK DAG commitment")?
                .commit;
            partial.bind_fixed_dag_commit(expected_dag_commit)?;
        }
        let exports = partial.exports();
        let inventory = partial.bus_inventory().clone();
        let mut manager = BusIndexManager::from_next_bus_idx(partial.next_bus_idx());

        let checkpoint_bus = CertifiedTranscriptCheckpointBus::new(manager.new_bus_idx());
        partial.set_checkpoint_state_bus(checkpoint_bus);
        let source_instance_bus = SourceInstanceDigestBusV19::new(manager.new_bus_idx());
        let fixed_public_values_bus = FixedSourcePublicValueBusV19::new(manager.new_bus_idx());
        let source_node_bus = SourceForestNodeBusV19::new(manager.new_bus_idx());
        let source_prefix_end_bus = SourcePrefixEndBusV19::new(manager.new_bus_idx());
        let source_leaf_bus = VerifiedSourceForestLeafBusV19::new(manager.new_bus_idx());
        let boundary_leaf_bus = VerifiedSourceForestLeafBusV19::new(manager.new_bus_idx());
        let mapped_leaf_bus = VerifiedSourceForestLeafBusV19::new(manager.new_bus_idx());
        let verified_public_value_bus =
            VerifiedDirectAirPublicValueBusV19::new(manager.new_bus_idx());
        let opening_point_bus = SegmentOpeningPointBusV19::new(manager.new_bus_idx());
        let arithmetic_bus = VerifiedLogUpArithmeticBusV19::new(manager.new_bus_idx());
        let opening_phase_start_bus = LogUpOpeningPhaseStartBusV19::new(manager.new_bus_idx());
        let phase_start_bus = LogUpClaimPhaseStartBusV19::new(manager.new_bus_idx());
        let challenge_bus = MappedFunctionalChallengeBusV19::new(manager.new_bus_idx());
        let auxiliary_challenge_bus = MappedAuxiliaryChallengeBusV19::new(manager.new_bus_idx());
        let stream_cursor_bus = OneShotStreamCursorBusV19::new(manager.new_bus_idx());
        let claim_stream_cursor_bus =
            if active_count_profile.is_some() || active_count_profile_v4.is_some() {
                OneShotStreamCursorBusV19::new(manager.new_bus_idx())
            } else {
                stream_cursor_bus
            };
        let round_start_bus = OneShotRoundStartBusV19::new(manager.new_bus_idx());
        let mapped_bus = VerifiedMappedFunctionalBusV19::new(manager.new_bus_idx());
        let raw_opening_bus = VerifiedOneShotRawOpeningBusV19::new(manager.new_bus_idx());
        let history_buses = DirectLogUpOnlyHistoryBusesV19 {
            endpoint: CertifiedLogUpOnlyEndpointBusV19::new(manager.new_bus_idx()),
            opening: CertifiedSwirlRawOpeningBusV19::new(manager.new_bus_idx()),
            vm_metadata: CertifiedVmSegmentMetadataBusV19::new(manager.new_bus_idx()),
            program_fingerprint: CertifiedProgramFingerprintBusV19::new(manager.new_bus_idx()),
            // History and this verifier must use one owned Poseidon table. The
            // v19 alias has the exact same bus type, so no adapter is needed.
            compress: inventory.poseidon2_compress_bus,
        };

        let prefix_transcript = LogUpOnlyPrefixTranscript::new(
            inventory.clone(),
            child_vk.inner.params.clone(),
            checkpoint_bus,
        );
        let fixed_public_values_air_id = active_count_profile
            .as_ref()
            .map(|profile| profile.vm_pvs_air_id)
            .or_else(|| {
                active_count_profile_v4
                    .as_ref()
                    .map(|profile| profile.vm_pvs_air_id)
            });
        let source_instance_air = DirectLogUpSourceInstanceAirV19 {
            profile: source_profile.clone(),
            permute_bus: inventory.poseidon2_permute_bus,
            public_values_bus: inventory.public_values_bus,
            verified_public_values_bus: verified_public_value_bus,
            fixed_public_values_bus,
            fixed_public_values_air_id,
            instance_bus: source_instance_bus,
        };
        let source_manifest_air = DirectLogUpSourceManifestAirV19 {
            profile: source_profile.clone(),
            transcript_bus: inventory.transcript_bus,
            permute_bus: inventory.poseidon2_permute_bus,
            compress_bus: inventory.poseidon2_compress_bus,
            instance_bus: source_instance_bus,
            node_bus: source_node_bus,
            leaf_bus: source_leaf_bus,
            prefix_end_bus: source_prefix_end_bus,
        };
        let source_forest_node_air = DirectLogUpSourceForestNodeAirV19 {
            profile: source_profile.clone(),
            permute_bus: inventory.poseidon2_permute_bus,
            compress_bus: inventory.poseidon2_compress_bus,
            node_bus: source_node_bus,
        };
        let prefix_bridge_air = PrefixCheckpointBridgeAirV19 {
            segment_start: source_profile.segment_start(),
            source_end_bus: source_prefix_end_bus,
            checkpoint_bus,
            rebased_start_bus: exports.rebased_start_bus,
        };
        let opening_point_fanout_air = OpeningPointFanoutAirV19 {
            segment_start: source_profile.segment_start(),
            source_bus: exports.opening_point_bus,
            lookup_bus: opening_point_bus,
            fixed_setup_point_bus: None,
            fixed_setup_point_demands: Arc::from([]),
        };
        let forest_fanout_air = ForestLeafFanoutAirV19 {
            source_bus: source_leaf_bus,
            boundary_bus: boundary_leaf_bus,
            mapped_bus: mapped_leaf_bus,
        };
        let endpoint_bridge_air = LogUpEndpointBridgeAirV19 {
            segment_start: source_profile.segment_start(),
            endpoint_bus: exports.endpoint_bus,
            transcript_end_index_bus: inventory.transcript_end_index_bus,
            checkpoint_bus,
            arithmetic_bus,
            opening_phase_start_bus,
        };
        let column_opening_observation_air = ColumnOpeningObservationAirV19::new(
            &mapped_profile,
            inventory.transcript_bus,
            exports.column_claims_bus,
            opening_phase_start_bus,
            phase_start_bus,
        )?;
        let claim_derivation_air = LogUpClaimDerivationAirV19 {
            segment_start: source_profile.segment_start(),
            transcript_bus: inventory.transcript_bus,
            phase_start_bus,
            challenge_bus,
            stream_cursor_bus,
        };
        let active_count_transcript_air =
            active_count_profile.map(|profile| ActiveChildCountTranscriptAirV19 {
                profile,
                transcript_bus: inventory.transcript_bus,
                input_cursor_bus: stream_cursor_bus,
                output_cursor_bus: claim_stream_cursor_bus,
                challenge_bus: auxiliary_challenge_bus,
                challenge_lookup_count: 1,
            });
        let active_count_transcript_air_v4 =
            active_count_profile_v4.map(|profile| ActiveChildCountTranscriptAirV4 {
                profile,
                transcript_bus: inventory.transcript_bus,
                input_cursor_bus: stream_cursor_bus,
                output_cursor_bus: claim_stream_cursor_bus,
                challenge_bus: auxiliary_challenge_bus,
                challenge_lookup_count: 1,
            });
        let mapped_functional_air = MappedFunctionalTermAirV19 {
            profile: mapped_profile.clone(),
            forest_bus: mapped_leaf_bus,
            challenge_bus,
            auxiliary_challenge_bus,
            opening_point_bus,
            column_claims_bus: exports.column_claims_bus,
            transcript_bus: inventory.transcript_bus,
            round_start_bus,
            stream_cursor_bus: claim_stream_cursor_bus,
            mapped_bus,
        };
        let mut log_message_lengths = mapped_profile
            .terms
            .iter()
            .map(|term| term.log_message_len as usize)
            .collect::<Vec<_>>();
        log_message_lengths.sort_unstable();
        log_message_lengths.dedup();
        let reduction_airs = log_message_lengths
            .into_iter()
            .map(|log_message_len| {
                OneShotReductionAirV19::new(
                    log_message_len,
                    inventory.transcript_bus,
                    round_start_bus,
                    claim_stream_cursor_bus,
                    mapped_bus,
                    raw_opening_bus,
                )
            })
            .collect();
        let finalization_air = LogUpFinalizationAirV19 {
            segment_start: source_profile.segment_start(),
            transcript_bus: inventory.transcript_bus,
            transcript_end_index_bus: inventory.transcript_end_index_bus,
            checkpoint_bus,
            stream_cursor_bus: claim_stream_cursor_bus,
        };
        let boundary_air = LogUpSwirlBoundaryAirV19::new(
            source_profile.segment_start(),
            arithmetic_bus,
            boundary_leaf_bus,
            verified_public_value_bus,
            raw_opening_bus,
            history_buses.endpoint,
            history_buses.opening,
            history_buses.vm_metadata,
            history_buses.program_fingerprint,
            history_buses.compress,
        );
        Ok(Self {
            child_vk,
            source_profile,
            partial,
            prefix_transcript,
            source_manifest_air,
            source_instance_air,
            source_forest_node_air,
            prefix_bridge_air,
            opening_point_fanout_air,
            forest_fanout_air,
            endpoint_bridge_air,
            column_opening_observation_air,
            claim_derivation_air,
            active_count_transcript_air,
            active_count_transcript_air_v4,
            mapped_functional_air,
            reduction_airs,
            finalization_air,
            boundary_air,
            history_buses,
            fixed_boundary_inputs: FixedMultiAirBoundaryInputsV19 {
                arithmetic: arithmetic_bus,
                source_leaf: boundary_leaf_bus,
                raw_opening: raw_opening_bus,
            },
            source_instance_bus,
            fixed_public_values_bus,
            fixed_multi_air,
            next_bus_idx: manager.next_bus_idx(),
        })
    }

    /// Collision-free buses to pass to History and grouped VACC modules built
    /// after this composite. Their allocator must resume at [`Self::next_bus_idx`].
    #[must_use]
    pub const fn history_buses(&self) -> DirectLogUpOnlyHistoryBusesV19 {
        self.history_buses
    }

    /// First collision-free bus index available to History/VACC modules that
    /// are assembled after this verifier.
    #[must_use]
    pub const fn next_bus_idx(&self) -> BusIndex {
        self.next_bus_idx
    }

    /// Attach the setup-PCS source-manifest output after this composite has allocated its
    /// internal buses.
    ///
    /// The manifest bus must be allocated by the enclosing History circuit starting at
    /// [`Self::next_bus_idx`], so it cannot be supplied to the source profile before this module
    /// exists. Both profile copies used by AIR construction and trace generation are replaced
    /// together; accepting a bus on only one copy would silently leave the interaction
    /// unbalanced.
    pub fn attach_setup_pcs_source_manifest_v3(
        &mut self,
        bus: SetupPcsSourceManifestBusV3,
    ) -> Result<(), &'static str> {
        if self
            .source_profile
            .setup_pcs_source_manifest_bus_v3()
            .is_some()
            || self
                .source_manifest_air
                .profile
                .setup_pcs_source_manifest_bus_v3()
                .is_some()
        {
            return Err("setup PCS source manifest already attached");
        }
        let profile = self
            .source_profile
            .clone()
            .with_setup_pcs_source_manifest_bus_v3(bus)?;
        self.source_profile = profile.clone();
        self.source_manifest_air.profile = profile;
        Ok(())
    }

    /// Attach V3 setup-PCS authority to the genuine SWIRL opening-point fanout.
    /// The demand table becomes preprocessed VK data; witness-selected
    /// multiplicities are not accepted. Call before keygen/trace generation.
    pub fn attach_fixed_setup_opening_points(
        &mut self,
        bus: FixedSetupOpeningPointBusV2,
        demands: &[(u32, u32, u32)],
    ) -> Result<(), &'static str> {
        if demands.is_empty()
            || demands.iter().any(|&(_, _, count)| count == 0)
            || demands
                .windows(2)
                .any(|pair| (pair[0].0, pair[0].1) >= (pair[1].0, pair[1].1))
            || self
                .opening_point_fanout_air
                .fixed_setup_point_bus
                .is_some()
        {
            return Err("invalid fixed setup opening-point attachment");
        }
        let proof_count = self
            .mapped_functional_air
            .profile
            .terms
            .last()
            .and_then(|term| usize::try_from(term.proof_index).ok())
            .and_then(|index| index.checked_add(1))
            .ok_or("invalid fixed setup opening-point attachment")?;
        let mut point_lengths = vec![0usize; proof_count];
        for term in self
            .mapped_functional_air
            .profile
            .terms
            .iter()
            .filter(|term| !term.is_program_term && !term.is_active_count_term)
        {
            let proof_index = usize::try_from(term.proof_index)
                .map_err(|_| "invalid fixed setup opening-point attachment")?;
            let point_len = usize::from(term.log_height)
                .saturating_sub(term.l_skip as usize)
                .checked_add(1)
                .ok_or("invalid fixed setup opening-point attachment")?;
            point_lengths[proof_index] = point_lengths[proof_index].max(point_len);
        }
        let aligned = align_fixed_setup_point_demands_v19(&point_lengths, demands)?;
        self.opening_point_fanout_air.fixed_setup_point_bus = Some(bus);
        self.opening_point_fanout_air.fixed_setup_point_demands = aligned.into();
        Ok(())
    }

    /// Reserve the additional verifier-owned v19 outputs consumed by the
    /// History-v2 C2 projection. This changes only fixed bus multiplicities;
    /// it neither adds a host authority path nor changes the v19 relation.
    pub fn attach_history_v2_active_count_projection(&mut self) -> Result<(), &'static str> {
        if !self.fixed_multi_air {
            return Err("active-count projection requires fixed multi-AIR mode");
        }
        let transcript = self.active_count_transcript_air.as_mut();
        let transcript_v4 = self.active_count_transcript_air_v4.as_mut();
        let challenge_lookup_count = match (transcript, transcript_v4) {
            (Some(transcript), None) => &mut transcript.challenge_lookup_count,
            (None, Some(transcript)) => &mut transcript.challenge_lookup_count,
            _ => return Err("missing or ambiguous active-count transcript AIR"),
        };
        if *challenge_lookup_count != 1 {
            return Err("active-count projection already attached");
        }
        *challenge_lookup_count = 2;
        for reduction in &mut self.reduction_airs {
            reduction.attach_history_v2_active_count_projection()?;
        }
        Ok(())
    }

    /// Exact fixed-profile footprint for runtime resource admission.
    #[must_use]
    pub const fn mapped_functional_metrics(&self) -> MappedFunctionalTraceMetricsV19 {
        self.mapped_functional_air.profile.metrics()
    }

    /// AIRs in exactly the same order as the production context generator.
    #[must_use]
    pub fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        let mut airs = self.partial.airs::<SC>();
        airs.push(self.prefix_transcript.air::<SC>());
        airs.push(Arc::new(self.source_instance_air.clone()));
        airs.push(Arc::new(self.source_manifest_air.clone()));
        airs.push(Arc::new(self.source_forest_node_air.clone()));
        airs.push(Arc::new(self.prefix_bridge_air.clone()));
        airs.push(Arc::new(self.opening_point_fanout_air.clone()));
        airs.push(Arc::new(self.forest_fanout_air.clone()));
        airs.push(Arc::new(self.endpoint_bridge_air.clone()));
        airs.push(Arc::new(self.column_opening_observation_air.clone()));
        airs.push(Arc::new(self.claim_derivation_air.clone()));
        if let Some(air) = &self.active_count_transcript_air {
            airs.push(Arc::new(air.clone()));
        }
        if let Some(air) = &self.active_count_transcript_air_v4 {
            airs.push(Arc::new(air.clone()));
        }
        airs.push(Arc::new(self.mapped_functional_air.clone()));
        airs.extend(
            self.reduction_airs
                .iter()
                .cloned()
                .map(|air| Arc::new(air) as AirRef<SC>),
        );
        airs.push(Arc::new(self.finalization_air.clone()));
        if !self.fixed_multi_air {
            airs.push(Arc::new(self.boundary_air.clone()));
        }
        airs
    }

    /// Exact composite AIR order with only the partial verifier's physical
    /// Poseidon owner removed. Every lookup-producing AIR remains present.
    #[must_use]
    pub fn airs_without_poseidon<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        let mut airs = self.partial.airs_without_poseidon::<SC>();
        airs.push(self.prefix_transcript.air::<SC>());
        airs.push(Arc::new(self.source_instance_air.clone()));
        airs.push(Arc::new(self.source_manifest_air.clone()));
        airs.push(Arc::new(self.source_forest_node_air.clone()));
        airs.push(Arc::new(self.prefix_bridge_air.clone()));
        airs.push(Arc::new(self.opening_point_fanout_air.clone()));
        airs.push(Arc::new(self.forest_fanout_air.clone()));
        airs.push(Arc::new(self.endpoint_bridge_air.clone()));
        airs.push(Arc::new(self.column_opening_observation_air.clone()));
        airs.push(Arc::new(self.claim_derivation_air.clone()));
        if let Some(air) = &self.active_count_transcript_air {
            airs.push(Arc::new(air.clone()));
        }
        if let Some(air) = &self.active_count_transcript_air_v4 {
            airs.push(Arc::new(air.clone()));
        }
        airs.push(Arc::new(self.mapped_functional_air.clone()));
        airs.extend(
            self.reduction_airs
                .iter()
                .cloned()
                .map(|air| Arc::new(air) as AirRef<SC>),
        );
        airs.push(Arc::new(self.finalization_air.clone()));
        if !self.fixed_multi_air {
            airs.push(Arc::new(self.boundary_air.clone()));
        }
        airs
    }

    #[must_use]
    pub fn shared_poseidon_owner(&self) -> Poseidon2BusOwner {
        self.partial.poseidon2_bus_owner()
    }

    /// Primary recursion bus inventory owned by the source/LogUp verifier.
    /// Direct-final composition reuses these exact public-values and Poseidon
    /// buses; it must not allocate wrapper-local substitutes.
    #[must_use]
    pub fn bus_inventory(&self) -> &BusInventory {
        self.partial.bus_inventory()
    }

    #[must_use]
    pub fn shared_poseidon_air<SC: StarkProtocolConfig<F = F>>(
        &self,
        ordered_owners: &[Poseidon2BusOwner],
    ) -> AirRef<SC> {
        self.partial.multi_bus_poseidon_air(ordered_owners)
    }

    /// Build the sole CPU multi-bus table for an outer assembly's ordered
    /// owner/input packet. Exactly one physical matrix is expected.
    pub fn build_shared_poseidon_context<SC: StarkProtocolConfig<F = F>>(
        &self,
        grouped_inputs: Poseidon2MultibusInputs,
    ) -> Result<AirProvingContext<CpuBackend<SC>>, DirectLogUpOnlyCompositeContextErrorV19> {
        let mut traces = self
            .partial
            .build_poseidon2_multibus_traces(grouped_inputs)
            .ok_or(DirectLogUpOnlyCompositeContextErrorV19::Trace(
                "shared Poseidon packet",
            ))?;
        if traces.len() != 1 {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Trace(
                "shared Poseidon owner count",
            ));
        }
        Ok(AirProvingContext::simple_no_pis(traces.pop().ok_or(
            DirectLogUpOnlyCompositeContextErrorV19::Trace("shared Poseidon table"),
        )?))
    }

    /// Physical sharding variant of [`Self::build_shared_poseidon_context`].
    /// All returned AIR contexts service the same ordered owner buses; the
    /// outer setup installs one identical AIR per returned shard.
    pub fn build_shared_poseidon_sharded_contexts<SC: StarkProtocolConfig<F = F>>(
        &self,
        grouped_inputs: Poseidon2MultibusInputs,
        shard_count: usize,
        max_rows: usize,
    ) -> Result<Vec<AirProvingContext<CpuBackend<SC>>>, DirectLogUpOnlyCompositeContextErrorV19>
    {
        let traces = self
            .partial
            .build_poseidon2_multibus_sharded_traces(grouped_inputs, shard_count, max_rows)
            .ok_or(DirectLogUpOnlyCompositeContextErrorV19::Trace(
                "sharded shared Poseidon packet",
            ))?;
        if traces.len() != shard_count {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Trace(
                "sharded shared Poseidon owner count",
            ));
        }
        Ok(traces
            .into_iter()
            .map(AirProvingContext::simple_no_pis)
            .collect())
    }

    /// Generate CPU row-major contexts in exactly [`Self::airs`] order.
    /// These contexts may be proved by the CPU backend directly or uploaded
    /// unchanged by a CUDA orchestration layer.
    ///
    /// All transcript permutation inputs, source-manifest hashes and boundary
    /// compressions are inserted into the partial verifier's one Poseidon
    /// table. The separately listed prefix Transcript AIR contributes only a
    /// transcript trace and returns its permutation inputs to that owner.
    pub fn generate_proving_contexts<SC: StarkProtocolConfig<F = F>>(
        &self,
        records: &DirectLogUpOnlyCompositeRecordsV19,
    ) -> Result<Vec<AirProvingContext<CpuBackend<SC>>>, DirectLogUpOnlyCompositeContextErrorV19>
    {
        generate_direct_logup_only_composite_contexts_v19(self, records)
    }

    /// Fixed-capacity CPU context generation. `active_child_counts` is
    /// runtime witness data and must describe the active prefix: every
    /// nonterminal slot has count four, while the final active slot has count
    /// one through four.
    pub fn generate_proving_contexts_fixed_capacity_v4<SC: StarkProtocolConfig<F = F>>(
        &self,
        records: &DirectLogUpOnlyCompositeRecordsV19,
        active_child_counts: &[u8],
    ) -> Result<Vec<AirProvingContext<CpuBackend<SC>>>, DirectLogUpOnlyCompositeContextErrorV19>
    {
        generate_direct_logup_only_composite_contexts_fixed_capacity_v4(
            self,
            records,
            active_child_counts,
        )
    }

    /// Packet-only context generation for an outer History assembly that owns
    /// one multi-bus Poseidon table shared with VACC and History requests.
    pub fn generate_proving_contexts_without_poseidon<SC: StarkProtocolConfig<F = F>>(
        &self,
        records: &DirectLogUpOnlyCompositeRecordsV19,
    ) -> Result<
        DirectLogUpOnlyCompositePacketContextsV19<SC>,
        DirectLogUpOnlyCompositeContextErrorV19,
    > {
        generate_direct_logup_only_composite_contexts_without_poseidon_v19(self, records)
    }

    /// Packet-only counterpart of
    /// [`Self::generate_proving_contexts_fixed_capacity_v4`].
    pub fn generate_proving_contexts_without_poseidon_fixed_capacity_v4<
        SC: StarkProtocolConfig<F = F>,
    >(
        &self,
        records: &DirectLogUpOnlyCompositeRecordsV19,
        active_child_counts: &[u8],
    ) -> Result<
        DirectLogUpOnlyCompositePacketContextsV19<SC>,
        DirectLogUpOnlyCompositeContextErrorV19,
    > {
        generate_direct_logup_only_composite_contexts_without_poseidon_fixed_capacity_v4(
            self,
            records,
            active_child_counts,
        )
    }

    /// Mixed CUDA counterpart of
    /// [`Self::generate_proving_contexts_without_poseidon`].
    ///
    /// Partial-verifier CUDA contexts are retained on the device. Only the
    /// resumed transcript/rebase contexts and the compact v19 tail are host
    /// contexts. Shared Poseidon requests and positive LogUp producers are
    /// returned byte-for-byte in the same logical packet as the CPU path.
    #[cfg(feature = "cuda")]
    pub fn generate_proving_contexts_without_poseidon_cuda(
        &self,
        records: &DirectLogUpOnlyCompositeRecordsV19,
        cached_trace_ctx: CachedTraceCtx<openvm_cuda_backend::GpuBackend>,
        device_ctx: &openvm_cuda_common::stream::GpuDeviceCtx,
    ) -> Result<DirectLogUpOnlyCompositeCudaPacketV19, DirectLogUpOnlyCompositeContextErrorV19>
    {
        generate_direct_logup_only_composite_contexts_without_poseidon_cuda_v19(
            self,
            records,
            cached_trace_ctx,
            device_ctx,
        )
    }

    /// CUDA counterpart of the fixed-capacity packet generator. Runtime
    /// counts follow the same canonical-prefix rules as the CPU path.
    #[cfg(feature = "cuda")]
    pub fn generate_proving_contexts_without_poseidon_cuda_fixed_capacity_v4(
        &self,
        records: &DirectLogUpOnlyCompositeRecordsV19,
        active_child_counts: &[u8],
        cached_trace_ctx: CachedTraceCtx<openvm_cuda_backend::GpuBackend>,
        device_ctx: &openvm_cuda_common::stream::GpuDeviceCtx,
    ) -> Result<DirectLogUpOnlyCompositeCudaPacketV19, DirectLogUpOnlyCompositeContextErrorV19>
    {
        generate_direct_logup_only_composite_contexts_without_poseidon_cuda_fixed_capacity_v4(
            self,
            records,
            active_child_counts,
            cached_trace_ctx,
            device_ctx,
        )
    }
}

#[derive(Clone)]
struct DerivedMappedClaimV19 {
    observations: Vec<EF>,
    ordinary_target: EF,
    ordinary_weight_at_point: EF,
    fingerprint_weight_at_point: EF,
    #[cfg(test)]
    claim: TerminalStructuredLinearClaim<EF>,
}

fn mapped_source_plans_v19(
    profile: &DirectLogUpMappedFunctionalProfileV19,
) -> Vec<&MappedTermPlanV19> {
    profile
        .terms
        .iter()
        .filter(|plan| plan.term_ordinal == 0)
        .collect()
}

fn reconstruct_retained_logup_proof_v19(
    child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    record: &DirectLogUpOnlyRetainedProofRecordV19,
) -> Result<RetainedLogUpOnlyProof, DirectLogUpOnlyCompositeContextErrorV19> {
    let count = record.trace_air_ids.len();
    if count == 0
        || record.n_per_trace.len() != count
        || record.need_rot_per_trace.len() != count
        || record.trace_public_values.len() != count
        || record
            .proof_material
            .batch_constraint_proof
            .column_openings
            .len()
            != count
    {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
            "inconsistent retained active-trace vectors",
        ));
    }
    let l_skip = isize::try_from(child_vk.inner.params.l_skip).map_err(|_| {
        DirectLogUpOnlyCompositeContextErrorV19::Shape("child l_skip exceeds isize")
    })?;
    let mut trace_vdata = vec![None; child_vk.inner.per_air.len()];
    let mut public_values = vec![Vec::new(); child_vk.inner.per_air.len()];
    let mut previous = None;
    for (((&air_id, &n), &need_rot), values) in record
        .trace_air_ids
        .iter()
        .zip(&record.n_per_trace)
        .zip(&record.need_rot_per_trace)
        .zip(&record.trace_public_values)
    {
        let air = child_vk.inner.per_air.get(air_id).ok_or(
            DirectLogUpOnlyCompositeContextErrorV19::Shape("retained AIR outside child VK"),
        )?;
        let log_height = usize::try_from(n.checked_add(l_skip).ok_or(
            DirectLogUpOnlyCompositeContextErrorV19::Shape("retained log-height overflow"),
        )?)
        .map_err(|_| {
            DirectLogUpOnlyCompositeContextErrorV19::Shape("negative retained log height")
        })?;
        let stacking_key = (core::cmp::Reverse(log_height), air_id);
        if previous.is_some_and(|prior| prior >= stacking_key)
            || trace_vdata[air_id].is_some()
            || need_rot != air.params.need_rot
            || values.len() != air.params.num_public_values
            || air.preprocessed_data.as_ref().is_some_and(|data| {
                child_vk
                    .inner
                    .params
                    .l_skip
                    .wrapping_add_signed(data.hypercube_dim)
                    != log_height
            })
        {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "retained traces differ from canonical child-VK shape/order",
            ));
        }
        previous = Some(stacking_key);
        trace_vdata[air_id] = Some(TraceVData {
            log_height,
            cached_commitments: vec![[F::ZERO; DIGEST_SIZE]; air.params.width.cached_mains.len()],
        });
        public_values[air_id] = values.clone();
    }
    Ok(RetainedLogUpOnlyProof {
        common_main_commit: [F::ZERO; DIGEST_SIZE],
        trace_vdata,
        public_values,
        gkr_proof: record.proof_material.gkr_proof.clone(),
        batch_constraint_proof: record.proof_material.batch_constraint_proof.clone(),
    })
}

fn build_source_prefix_v19(
    app_vk_digest: [F; DIGEST_SIZE],
    registry_digest: [F; DIGEST_SIZE],
    source_forest_root: [F; DIGEST_SIZE],
    source: &DirectLogUpSegmentSourceRecordV19,
) -> Result<
    openvm_stark_sdk::config::baby_bear_poseidon2::DuplexSpongeRecorder,
    DirectLogUpOnlyCompositeContextErrorV19,
> {
    if source.prefix_start_tidx != 0 || source.sources.is_empty() {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::Source(
            "noncanonical source-prefix header",
        ));
    }
    let mut transcript = default_duplex_sponge_recorder();
    let observe = |transcript: &mut _, value| {
        FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(transcript, value)
    };
    observe(&mut transcript, F::from_u64(SEGMENT_PREFIX_TAG_V19));
    observe(
        &mut transcript,
        F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
    );
    observe(&mut transcript, F::from_u32(source.segment_index));
    for digest in [app_vk_digest, registry_digest, source_forest_root] {
        FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_commit(&mut transcript, digest);
    }
    observe(&mut transcript, F::from_usize(source.sources.len()));
    for retained in &source.sources {
        observe(&mut transcript, F::from_u32(retained.shard_id));
        FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_commit(
            &mut transcript,
            retained.expected_instance_digest,
        );
        FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_commit(
            &mut transcript,
            retained.source_root,
        );
    }
    observe(&mut transcript, F::from_u64(LOGUP_START_BOUNDARY_TAG_V19));
    let sampled = FiatShamirTranscript::<BabyBearPoseidon2Config>::sample_ext(&mut transcript);
    let expected_sample = EF::from_basis_coefficients_slice(&source.alignment_sample).ok_or(
        DirectLogUpOnlyCompositeContextErrorV19::Transcript("source alignment EF4"),
    )?;
    if sampled != expected_sample {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::Transcript(
            "source alignment sample",
        ));
    }
    let checkpoint = transcript.inner.checkpoint();
    if checkpoint.absorb_idx != 0 {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::Transcript(
            "source prefix is not an absorb boundary",
        ));
    }
    Ok(transcript)
}

fn observe_claim_derivation_v19(
    transcript: &mut openvm_stark_sdk::config::baby_bear_poseidon2::DuplexSpongeRecorder,
    record: &LogUpClaimDerivationRecordV19,
) -> Result<(), DirectLogUpOnlyCompositeContextErrorV19> {
    if record.start_tidx as usize != transcript.len() {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::Transcript(
            "claim derivation cursor",
        ));
    }
    for value in [
        F::from_u64(super::DIRECT_LOGUP_CLAIM_TAG_V19),
        F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
        F::from_u32(record.segment_index),
        F::from_u32(record.shard_id),
    ] {
        FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(transcript, value);
    }
    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_commit(
        transcript,
        record.relation_digest,
    );
    let batching = FiatShamirTranscript::<BabyBearPoseidon2Config>::sample_ext(transcript);
    if batching != record.batching_challenge {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::Transcript(
            "mapped batching challenge",
        ));
    }
    if record.is_program {
        FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(
            transcript,
            F::from_u64(super::PROGRAM_FINGERPRINT_TARGET_TAG_V19),
        );
        FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_ext(
            transcript,
            record.program_fingerprint,
        );
        let mix = FiatShamirTranscript::<BabyBearPoseidon2Config>::sample_ext(transcript);
        if mix != record.program_mix_challenge {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Transcript(
                "Program mix challenge",
            ));
        }
    }
    if record.end_tidx() as usize != transcript.len() {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::Transcript(
            "claim derivation end cursor",
        ));
    }
    Ok(())
}

fn observe_one_shot_v19(
    transcript: &mut openvm_stark_sdk::config::baby_bear_poseidon2::DuplexSpongeRecorder,
    claim: &OneShotClaimTranscriptRecordV19,
    reduction: &OneShotReductionRecordV19,
) -> Result<(), DirectLogUpOnlyCompositeContextErrorV19> {
    if claim.start_tidx as usize != transcript.len()
        || claim.round_start_tidx() != reduction.start_tidx
        || claim.proof_index != reduction.proof_index
        || claim.segment_index != reduction.segment_index
        || claim.shard_ordinal != reduction.shard_ordinal
    {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::Transcript(
            "one-shot claim/reduction cursor",
        ));
    }
    for &observation in &claim.observations {
        FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_ext(transcript, observation);
    }
    for (round, polynomial) in reduction.round_polynomials.iter().enumerate() {
        for &coefficient in polynomial {
            FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_ext(transcript, coefficient);
        }
        let challenge = FiatShamirTranscript::<BabyBearPoseidon2Config>::sample_ext(transcript);
        if reduction.point.get(round) != Some(&challenge) {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Transcript(
                "one-shot reduction challenge",
            ));
        }
    }
    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_ext(
        transcript,
        reduction.message_value,
    );
    Ok(())
}

fn evaluate_mle_v19(
    mut evaluations: Vec<EF>,
    point: &[EF],
) -> Result<EF, DirectLogUpOnlyCompositeContextErrorV19> {
    if evaluations.len() != 1usize.checked_shl(point.len() as u32).unwrap_or(0) {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
            "mapped weight/point length",
        ));
    }
    for &challenge in point {
        // Match the backend's MLE convention exactly: the next coordinate
        // selects between the lower and upper halves, not adjacent entries.
        // Using adjacent pairs reverses bit significance and makes the
        // certified terminal weight disagree with the degree-two reduction.
        let half = evaluations.len() / 2;
        let (lower, upper) = evaluations.split_at_mut(half);
        for (low, &high) in lower.iter_mut().zip(upper.iter()) {
            *low += challenge * (high - *low);
        }
        evaluations.truncate(half);
    }
    evaluations
        .first()
        .copied()
        .ok_or(DirectLogUpOnlyCompositeContextErrorV19::Shape(
            "empty mapped weight",
        ))
}

fn derive_mapped_claim_v19(
    plans: &[MappedTermPlanV19],
    mapped: &MappedFunctionalSourceRecordV19,
    runtime_active_child_count: Option<u8>,
) -> Result<DerivedMappedClaimV19, DirectLogUpOnlyCompositeContextErrorV19> {
    let first = plans
        .first()
        .ok_or(DirectLogUpOnlyCompositeContextErrorV19::Shape(
            "empty mapped source plan",
        ))?;
    let mut ordinary_terms = Vec::new();
    let mut fingerprint_terms = Vec::new();
    let mut combined_terms = Vec::new();
    let mut ordinary_power = EF::ONE;
    let mut program_power = EF::ONE;
    let mut ordinary_target = EF::ZERO;
    let mut claim_index = 0usize;
    for plan in plans {
        let (l_skip, barycentric_weights, folded_row_eq_point) = if plan.is_program_term {
            (0usize, vec![EF::ONE], plan.program_row_point.clone())
        } else if plan.is_active_count_term {
            (
                0usize,
                vec![EF::ONE],
                vec![EF::TWO.inverse(); plan.log_height as usize],
            )
        } else {
            let folded_len = usize::from(plan.log_height).saturating_sub(plan.l_skip as usize);
            if mapped.logup_opening_point.len() <= folded_len {
                return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                    "mapped LogUp point length",
                ));
            }
            let r0 = mapped.logup_opening_point[0];
            let q = 1usize << plan.l_skip;
            let omega = F::two_adic_generator(plan.l_skip as usize);
            let scaling = (r0.exp_u64(q as u64) - EF::ONE) * EF::from_usize(q).inverse();
            let mut omega_power = F::ONE;
            let mut barycentric = Vec::with_capacity(q);
            for _ in 0..q {
                let denominator = r0 - EF::from(omega_power);
                if denominator == EF::ZERO {
                    return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                        "mapped PLE denominator",
                    ));
                }
                barycentric.push(EF::from(omega_power) * denominator.inverse() * scaling);
                omega_power *= omega;
            }
            (
                plan.l_skip as usize,
                barycentric,
                (1..=folded_len)
                    .rev()
                    .map(|index| mapped.logup_opening_point[index])
                    .collect(),
            )
        };
        let base = PrismalinearMappedColumnTerm {
            block: PrismalinearMappedColumnBlock {
                start: plan.block_start as usize,
                log_height: plan.log_height as usize,
            },
            l_skip,
            barycentric_weights,
            folded_row_eq_point,
            rotation: if plan.is_rot {
                PrismalinearMappedColumnRotation::Next
            } else {
                PrismalinearMappedColumnRotation::Current
            },
            scale: EF::ZERO,
        };
        if plan.is_program_term {
            let mut fingerprint = base.clone();
            fingerprint.scale = program_power;
            fingerprint_terms.push(fingerprint.clone());
            fingerprint.scale *= mapped.program_mix_challenge;
            combined_terms.push(fingerprint);
            program_power *= plan.program_column_challenge;
        } else if plan.is_active_count_term {
            if mapped.active_count_challenge == EF::ZERO {
                return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                    "zero active-count challenge",
                ));
            }
            let coefficient = mapped.active_count_challenge;
            let mut count = base;
            count.scale = coefficient * EF::from_usize(1usize << plan.log_height);
            ordinary_target += coefficient
                * EF::from(F::from_u8(
                    runtime_active_child_count.unwrap_or(plan.active_count_expected),
                ));
            ordinary_terms.push(count.clone());
            combined_terms.push(count);
        } else {
            let claim = *mapped.dynamic_column_claims.get(claim_index).ok_or(
                DirectLogUpOnlyCompositeContextErrorV19::Shape(
                    "missing mapped dynamic column claim",
                ),
            )?;
            ordinary_target += ordinary_power * claim;
            let mut ordinary = base;
            ordinary.scale = ordinary_power;
            ordinary_terms.push(ordinary.clone());
            combined_terms.push(ordinary);
            ordinary_power *= mapped.batching_challenge;
            claim_index += 1;
        }
    }
    if claim_index != mapped.dynamic_column_claims.len() {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
            "extra mapped dynamic column claims",
        ));
    }
    let ordinary_weight = PrismalinearMappedColumnWeight {
        log_message_len: first.log_message_len as usize,
        terms: ordinary_terms,
    };
    let fingerprint_weight = PrismalinearMappedColumnWeight {
        log_message_len: first.log_message_len as usize,
        terms: fingerprint_terms,
    };
    let combined_claim = TerminalStructuredLinearClaim::new(
        TerminalWeightSpec::PrismalinearMappedColumns(PrismalinearMappedColumnWeight {
            log_message_len: first.log_message_len as usize,
            terms: combined_terms,
        }),
        ordinary_target + mapped.program_mix_challenge * mapped.program_fingerprint,
    );
    let observations = direct_message_opening_reduction_claim_observations(&combined_claim)
        .map_err(|_| {
            DirectLogUpOnlyCompositeContextErrorV19::Shape("mapped terminal claim shape")
        })?;
    Ok(DerivedMappedClaimV19 {
        observations,
        ordinary_target,
        ordinary_weight_at_point: evaluate_mle_v19(
            ordinary_weight.materialize(),
            &mapped.one_shot_point,
        )?,
        fingerprint_weight_at_point: evaluate_mle_v19(
            fingerprint_weight.materialize(),
            &mapped.one_shot_point,
        )?,
        #[cfg(test)]
        claim: combined_claim,
    })
}

fn join_digest_v19(left: [F; DIGEST_SIZE], right: [F; DIGEST_SIZE]) -> [F; POSEIDON2_WIDTH] {
    core::array::from_fn(|index| {
        if index < DIGEST_SIZE {
            left[index]
        } else {
            right[index - DIGEST_SIZE]
        }
    })
}

fn boundary_compression_inputs_v19(
    records: &[LogUpSwirlBoundaryShardRecordV19],
) -> Result<Vec<[F; POSEIDON2_WIDTH]>, DirectLogUpOnlyCompositeContextErrorV19> {
    let program = records.iter().find(|record| record.is_program).ok_or(
        DirectLogUpOnlyCompositeContextErrorV19::Shape("missing Program boundary record"),
    )?;
    let mut inputs = Vec::new();
    for first in records.iter().filter(|record| record.shard_ordinal == 0) {
        let mut fingerprint_left = [F::ZERO; DIGEST_SIZE];
        fingerprint_left[0] = F::from_u64(PROGRAM_FINGERPRINT_DIGEST_TAG_V19);
        fingerprint_left[1] = F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19);
        fingerprint_left[2] = F::from_u8(program.log_height);
        fingerprint_left[3] = F::from_u32(program.cached_width);
        fingerprint_left[4..]
            .copy_from_slice(program.program_fingerprint.as_basis_coefficients_slice());
        inputs.push(join_digest_v19(fingerprint_left, program.relation_digest));

        let mut from_meta = [F::ZERO; DIGEST_SIZE];
        from_meta[0] = F::from_u32(VM_STATE_HASH_TAG_V19);
        from_meta[1] = F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19);
        from_meta[2] = first.initial_pc;
        let mut to_meta = [F::ZERO; DIGEST_SIZE];
        to_meta[0] = F::from_u32(VM_STATE_HASH_TAG_V19);
        to_meta[1] = F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19);
        to_meta[2] = first.final_pc;
        to_meta[3] = F::from_bool(first.is_terminate);
        let from = super::compute_vm_state_digest_v19(
            program.program_fingerprint_digest,
            first.initial_pc,
            false,
            first.initial_memory_root,
        );
        let to = super::compute_vm_state_digest_v19(
            program.program_fingerprint_digest,
            first.final_pc,
            first.is_terminate,
            first.final_memory_root,
        );
        inputs.push(join_digest_v19(from_meta, first.initial_memory_root));
        inputs.push(join_digest_v19(to_meta, first.final_memory_root));
        inputs.push(join_digest_v19(
            program.program_fingerprint_digest,
            from.inner_digest,
        ));
        inputs.push(join_digest_v19(
            program.program_fingerprint_digest,
            to.inner_digest,
        ));
    }
    Ok(inputs)
}

/// Public free-function form of
/// [`DirectLogUpOnlyVerifierModuleV19::generate_proving_contexts`].
pub fn generate_direct_logup_only_composite_contexts_v19<
    SC: StarkProtocolConfig<F = F>,
    const MAX_NUM_PROOFS: usize,
>(
    module: &DirectLogUpOnlyVerifierModuleV19<MAX_NUM_PROOFS>,
    records: &DirectLogUpOnlyCompositeRecordsV19,
) -> Result<Vec<AirProvingContext<CpuBackend<SC>>>, DirectLogUpOnlyCompositeContextErrorV19> {
    let packet =
        generate_direct_logup_only_composite_contexts_without_poseidon_v19(module, records)?;
    let poseidon_trace = module
        .partial
        .build_local_poseidon2_trace(
            packet.poseidon.permutation_inputs,
            packet.poseidon.compression_inputs,
        )
        .ok_or(DirectLogUpOnlyCompositeContextErrorV19::Trace(
            "local Poseidon table",
        ))?;
    let mut contexts = packet.contexts;
    contexts.insert(
        module.partial.poseidon_air_index::<SC>(),
        AirProvingContext::simple_no_pis(poseidon_trace),
    );
    let airs = module.airs::<SC>();
    if airs.len() != contexts.len()
        || airs.iter().zip(&contexts).any(|(air, context)| {
            air.common_main_width() != context.common_main.width()
                || air.cached_main_widths().len() != context.cached_mains.len()
        })
    {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
            "AIR/context order or width",
        ));
    }
    Ok(contexts)
}

/// Full CPU context generation for the fixed-capacity HLeaf relation. The
/// active counts are supplied only to main-trace generation and never alter
/// the AIR inventory or preprocessing.
pub fn generate_direct_logup_only_composite_contexts_fixed_capacity_v4<
    SC: StarkProtocolConfig<F = F>,
    const MAX_NUM_PROOFS: usize,
>(
    module: &DirectLogUpOnlyVerifierModuleV19<MAX_NUM_PROOFS>,
    records: &DirectLogUpOnlyCompositeRecordsV19,
    active_child_counts: &[u8],
) -> Result<Vec<AirProvingContext<CpuBackend<SC>>>, DirectLogUpOnlyCompositeContextErrorV19> {
    let packet = generate_direct_logup_only_composite_contexts_without_poseidon_fixed_capacity_v4(
        module,
        records,
        active_child_counts,
    )?;
    let poseidon_trace = module
        .partial
        .build_local_poseidon2_trace(
            packet.poseidon.permutation_inputs,
            packet.poseidon.compression_inputs,
        )
        .ok_or(DirectLogUpOnlyCompositeContextErrorV19::Trace(
            "local Poseidon table",
        ))?;
    let mut contexts = packet.contexts;
    contexts.insert(
        module.partial.poseidon_air_index::<SC>(),
        AirProvingContext::simple_no_pis(poseidon_trace),
    );
    let airs = module.airs::<SC>();
    if airs.len() != contexts.len()
        || airs.iter().zip(&contexts).any(|(air, context)| {
            air.common_main_width() != context.common_main.width()
                || air.cached_main_widths().len() != context.cached_mains.len()
        })
    {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
            "AIR/context order or width",
        ));
    }
    Ok(contexts)
}

/// Shared validation, retained-proof replay, preflight and compact-v19 tail
/// builder. Neither backend may bypass this anchor, and this function never
/// generates the partial verifier's CPU traces.
#[allow(clippy::too_many_lines)]
fn prepare_direct_logup_only_composite_v19<
    SC: StarkProtocolConfig<F = F>,
    const MAX_NUM_PROOFS: usize,
>(
    module: &DirectLogUpOnlyVerifierModuleV19<MAX_NUM_PROOFS>,
    records: &DirectLogUpOnlyCompositeRecordsV19,
    runtime_active_child_counts: Option<&[u8]>,
) -> Result<DirectLogUpOnlyCompositePreparedV19<SC>, DirectLogUpOnlyCompositeContextErrorV19> {
    let all_source_plans = mapped_source_plans_v19(&module.mapped_functional_air.profile);
    let planned_source_count = all_source_plans.len();
    let segment_count = records.sources.len();
    let runtime_capacity_v4 = module.mapped_functional_air.profile.runtime_capacity_v4();
    if runtime_capacity_v4
        != (module.active_count_transcript_air_v4.is_some()
            && module.active_count_transcript_air.is_none())
        || runtime_capacity_v4 != runtime_active_child_counts.is_some()
        || runtime_active_child_counts.is_some_and(|counts| {
            counts.len() != segment_count
                || counts.iter().enumerate().any(|(index, &count)| {
                    count == 0
                        || usize::from(count) > VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2
                        || (index + 1 != counts.len()
                            && usize::from(count) != VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2)
                })
        })
    {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
            "fixed-capacity runtime active-child counts",
        ));
    }
    let source_count = if runtime_capacity_v4 {
        segment_count
    } else {
        planned_source_count
    };
    let source_plans = &all_source_plans[..source_count.min(planned_source_count)];
    if segment_count == 0
        || segment_count > MAX_NUM_PROOFS
        || records.retained_proofs.len() != segment_count
        || planned_source_count != module.source_profile.source_count()
        || source_count > planned_source_count
        || records.claim_derivations.len() != source_count
        || records.mapped_sources.len() != source_count
        || records.claim_transcripts.len() != source_count
        || records.reductions.len() != source_count
        || records.boundary_shards.len() != source_count
    {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
            "composite record cardinality",
        ));
    }

    let mut flat_sources = Vec::with_capacity(source_count);
    for (proof_index, segment) in records.sources.iter().enumerate() {
        if segment.proof_index as usize != proof_index || segment.sources.is_empty() {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Source(
                "dense source segment order",
            ));
        }
        for (ordinal, source) in segment.sources.iter().enumerate() {
            flat_sources.push((proof_index, ordinal, source));
        }
    }
    if flat_sources.len() != source_count {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::Source(
            "source count differs from fixed plan",
        ));
    }

    for index in 0..source_count {
        let plan = source_plans[index];
        let (proof_index, ordinal, source) = flat_sources[index];
        let claim = &records.claim_derivations[index];
        let mapped = &records.mapped_sources[index];
        let claim_transcript = &records.claim_transcripts[index];
        let reduction = &records.reductions[index];
        let boundary = &records.boundary_shards[index];
        if plan.proof_index as usize != proof_index
            || plan.shard_ordinal as usize != ordinal
            || source.shard_id != plan.shard_id
            || claim.proof_index != plan.proof_index
            || claim.segment_index != plan.segment_index
            || claim.shard_ordinal != plan.shard_ordinal
            || claim.shard_count != plan.shard_count
            || claim.shard_id != plan.shard_id
            || claim.relation_digest != plan.relation_digest
            || claim.is_program != plan.is_program
            || mapped.proof_index != plan.proof_index
            || mapped.segment_index != plan.segment_index
            || mapped.shard_ordinal != plan.shard_ordinal
            || mapped.batching_challenge != claim.batching_challenge
            || mapped.program_fingerprint != claim.program_fingerprint
            || mapped.program_mix_challenge != claim.program_mix_challenge
            || claim_transcript.proof_index != plan.proof_index
            || claim_transcript.segment_index != plan.segment_index
            || claim_transcript.shard_ordinal != plan.shard_ordinal
            || reduction.proof_index != plan.proof_index
            || reduction.segment_index != plan.segment_index
            || reduction.shard_ordinal != plan.shard_ordinal
            || reduction.is_program != plan.is_program
            || reduction.source_root != source.source_root
            || reduction.functional_digest != plan.relation_digest
            || boundary.proof_index != plan.proof_index
            || boundary.segment_index != plan.segment_index
            || boundary.shard_ordinal != plan.shard_ordinal
            || boundary.shard_count != plan.shard_count
            || boundary.air_id != plan.air_id
            || boundary.is_program != plan.is_program
            || boundary.relation_digest != plan.relation_digest
            || boundary.log_height != plan.log_height
            || boundary.cached_width != plan.cached_width
            || boundary.log_message_len != plan.log_message_len
            || boundary.app_vk_digest != module.source_profile.app_vk_digest
            || boundary.registry_digest != module.source_profile.registry_digest
            || boundary.source_root != source.source_root
            || mapped.source_root != source.source_root
            || mapped.range_start != reduction.range_start
            || mapped.range_end != reduction.range_end
            || mapped.range_start != boundary.range_start
            || mapped.range_end != boundary.range_end
            || mapped.one_shot_point != reduction.point
            || mapped.one_shot_point != boundary.opening_point
            || reduction.message_value != boundary.opening_value
            || reduction.program_fingerprint != mapped.program_fingerprint
            || boundary.program_fingerprint != mapped.program_fingerprint
            || boundary.segment_sum_before != EF::ZERO
            || boundary.segment_sum_after != EF::ZERO
        {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "noncanonical shard record order or binding",
            ));
        }
        let term_count = plan.term_count as usize;
        let plan_start = module
            .mapped_functional_air
            .profile
            .terms
            .iter()
            .position(|candidate| core::ptr::eq(candidate, plan))
            .ok_or(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "mapped source plan lookup",
            ))?;
        let plans = module
            .mapped_functional_air
            .profile
            .terms
            .get(plan_start..plan_start + term_count)
            .ok_or(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "mapped source plan range",
            ))?;
        let derived = derive_mapped_claim_v19(
            plans,
            mapped,
            runtime_active_child_counts.map(|counts| counts[proof_index]),
        )?;
        if claim_transcript.observations != derived.observations {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "mapped claim transcript mismatch",
            ));
        }
        if reduction.ordinary_target != derived.ordinary_target {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "mapped ordinary target mismatch",
            ));
        }
        if reduction.ordinary_weight_at_point != derived.ordinary_weight_at_point {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "mapped ordinary weight mismatch",
            ));
        }
        if reduction.fingerprint_weight_at_point != derived.fingerprint_weight_at_point {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "mapped fingerprint weight mismatch",
            ));
        }
        if reduction.program_fingerprint != mapped.program_fingerprint
            || reduction.mix_challenge != mapped.program_mix_challenge
        {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "mapped fingerprint binding mismatch",
            ));
        }
    }

    let source_traces =
        generate_direct_logup_source_manifest_traces_v19(&module.source_profile, &records.sources)
            .map_err(DirectLogUpOnlyCompositeContextErrorV19::Source)?;
    if source_traces.forest_roots.len() != segment_count {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::Source(
            "source forest root count",
        ));
    }
    for (index, boundary) in records.boundary_shards.iter().enumerate() {
        let proof_index = boundary.proof_index as usize;
        let expected_root = *source_traces.forest_roots.get(proof_index).ok_or(
            DirectLogUpOnlyCompositeContextErrorV19::Source("boundary segment forest root"),
        )?;
        if boundary.source_forest_root != expected_root
            || records.mapped_sources[index].source_forest_root != expected_root
            || records.mapped_sources[index].segment_openings_digest
                != boundary.segment_openings_digest
            || records.sources[proof_index].segment_openings_digest
                != boundary.segment_openings_digest
        {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Source(
                "source forest/openings binding",
            ));
        }
    }

    let mut shaped_proofs = Vec::with_capacity(segment_count);
    let mut prefix_logs = Vec::with_capacity(segment_count);
    let mut prefix_end_indices = Vec::with_capacity(segment_count);
    let mut prefix_bridge_records = Vec::with_capacity(segment_count);
    let mut preflights = Vec::<Preflight>::with_capacity(segment_count);
    let mut extended_logs =
        Vec::<TranscriptLog<F, [F; POSEIDON2_WIDTH]>>::with_capacity(segment_count);
    let mut endpoint_records = Vec::with_capacity(segment_count);
    let mut finalization_records = Vec::with_capacity(segment_count);
    let mut checkpoint_targets = Vec::with_capacity(segment_count);
    let mut opening_point_records = Vec::with_capacity(segment_count);
    let mut logup_producers = Vec::with_capacity(segment_count);
    let mut active_count_records = Vec::<ActiveChildCountTranscriptRecordV19>::new();
    let mut active_count_records_v4 = Vec::<ActiveChildCountTranscriptRecordV4>::new();

    for proof_index in 0..segment_count {
        let retained = &records.retained_proofs[proof_index];
        if retained.proof_index as usize != proof_index {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "dense retained proof index",
            ));
        }
        let segment_plans = source_plans
            .iter()
            .copied()
            .filter(|plan| plan.proof_index as usize == proof_index)
            .collect::<Vec<_>>();
        let segment_index = segment_plans
            .first()
            .ok_or(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "empty fixed segment plan",
            ))?
            .segment_index;
        let mut expected_shapes = if module.fixed_multi_air {
            module
                .mapped_functional_air
                .profile
                .terms
                .iter()
                .filter(|term| {
                    term.proof_index as usize == proof_index
                        && !term.is_program_term
                        && !term.is_active_count_term
                })
                .map(|term| (term.air_id as usize, term.log_height as usize))
                .collect::<Vec<_>>()
        } else {
            segment_plans
                .iter()
                .map(|plan| (plan.air_id as usize, plan.log_height as usize))
                .collect::<Vec<_>>()
        };
        expected_shapes
            .sort_by_key(|&(air_id, log_height)| (core::cmp::Reverse(log_height), air_id));
        expected_shapes.dedup();
        if expected_shapes.len() != retained.trace_air_ids.len()
            || expected_shapes
                .iter()
                .zip(&retained.trace_air_ids)
                .zip(&retained.n_per_trace)
                .any(|((expected, &air_id), &n)| {
                    expected.0 != air_id
                        || isize::try_from(expected.1).ok().and_then(|height| {
                            height.checked_sub(OPENVM_DIRECT_LOGUP_L_SKIP_V19 as isize)
                        }) != Some(n)
                })
        {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "retained proof differs from fixed segment shard plan",
            ));
        }
        let shaped = reconstruct_retained_logup_proof_v19(&module.child_vk, retained)?;

        for (source_plan, mapped) in segment_plans.iter().zip(
            records
                .mapped_sources
                .iter()
                .filter(|mapped| mapped.proof_index as usize == proof_index),
        ) {
            if mapped.logup_opening_point.is_empty() {
                return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                    "empty mapped LogUp point",
                ));
            }
            let claims = &shaped.batch_constraint_proof.column_openings;
            let dynamic = module
                .mapped_functional_air
                .profile
                .terms
                .iter()
                .filter(|term| {
                    term.segment_index == source_plan.segment_index
                        && term.shard_ordinal == source_plan.shard_ordinal
                        && !term.is_program_term
                        && !term.is_active_count_term
                })
                .map(|term| {
                    let openings_per_column = 1 + usize::from(
                        module.child_vk.inner.per_air[term.air_id as usize]
                            .params
                            .need_rot,
                    );
                    claims
                        .get(term.sort_idx as usize)
                        .and_then(|parts| parts.get(term.part_idx as usize))
                        .and_then(|part| {
                            part.get(
                                term.col_idx as usize * openings_per_column
                                    + usize::from(term.is_rot),
                            )
                        })
                        .copied()
                })
                .collect::<Option<Vec<_>>>()
                .ok_or(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                    "mapped column claim index",
                ))?;
            if dynamic != mapped.dynamic_column_claims {
                return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                    "mapped column claims differ from retained LogUp proof",
                ));
            }
        }

        let prefix = build_source_prefix_v19(
            module.source_profile.app_vk_digest,
            module.source_profile.registry_digest,
            source_traces.forest_roots[proof_index],
            &records.sources[proof_index],
        )?;
        let transcript_checkpoint = TranscriptHistory::checkpoint(&prefix);
        let sponge_checkpoint = prefix.inner.checkpoint();
        let start = RebasedTranscriptPreflight {
            start_tidx: prefix.len(),
            state: sponge_checkpoint.state,
        };
        let prefix_log = prefix.clone().into_log();
        prefix_end_indices.push(prefix.len());
        prefix_bridge_records.push(PrefixCheckpointBridgeRecordV19 {
            proof_index: proof_index as u32,
            segment_index,
            end_tidx: u32::try_from(prefix.len()).map_err(|_| {
                DirectLogUpOnlyCompositeContextErrorV19::Shape("prefix cursor exceeds u32")
            })?,
            state: sponge_checkpoint.state,
        });

        let (proof, preflight) =
            module
                .partial
                .run_preflight_retained(prefix.clone(), &module.child_vk, shaped, start);
        let mut transcript = prefix;
        let omega = F::two_adic_generator(OPENVM_DIRECT_LOGUP_L_SKIP_V19);
        let omega_skip_pows: Vec<F> = omega
            .powers()
            .take(1 << OPENVM_DIRECT_LOGUP_L_SKIP_V19)
            .collect();
        let endpoint = verify_logup_only_prefix(
            &mut transcript,
            &module.child_vk.inner,
            &proof.gkr_proof,
            (&proof.batch_constraint_proof).into(),
            &retained.trace_air_ids,
            &retained.n_per_trace,
            &omega_skip_pows,
            None,
        )
        .map_err(|_| DirectLogUpOnlyCompositeContextErrorV19::LogUpReplay)?;
        if endpoint.final_claim != preflight.batch_constraint.final_claim
            || endpoint.rs != preflight.batch_constraint.sumcheck_rnd
            || transcript.len() != preflight.batch_constraint.post_tidx
        {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::LogUpReplay);
        }
        let endpoint_checkpoint = transcript.inner.checkpoint();
        if endpoint_checkpoint.absorb_idx != 0 {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Transcript(
                "LogUp endpoint is not a squeeze boundary",
            ));
        }
        observe_batch_constraint_openings(
            &mut transcript,
            &module.child_vk.inner,
            &retained.trace_air_ids,
            &proof.batch_constraint_proof.column_openings,
        )
        .map_err(|_| DirectLogUpOnlyCompositeContextErrorV19::LogUpReplay)?;

        let segment_claims = records
            .claim_derivations
            .iter()
            .filter(|record| record.proof_index as usize == proof_index)
            .collect::<Vec<_>>();
        for claim in &segment_claims {
            observe_claim_derivation_v19(&mut transcript, claim)?;
        }
        if let Some(active_count_air) = &module.active_count_transcript_air {
            let record = observe_active_child_count_transcript_v19(
                &active_count_air.profile,
                proof_index,
                &mut transcript,
            )
            .map_err(DirectLogUpOnlyCompositeContextErrorV19::Transcript)?;
            let sampled_challenge = if record.sampled == EF::ZERO {
                EF::ONE
            } else {
                record.sampled
            };
            let mapped = records
                .mapped_sources
                .iter()
                .find(|record| record.proof_index as usize == proof_index)
                .ok_or(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                    "missing fixed multi-AIR mapped source",
                ))?;
            if mapped.active_count_challenge != sampled_challenge {
                return Err(DirectLogUpOnlyCompositeContextErrorV19::Transcript(
                    "active-child-count challenge",
                ));
            }
            active_count_records.push(record);
        }
        if let Some(active_count_air) = &module.active_count_transcript_air_v4 {
            let active_child_count = runtime_active_child_counts
                .and_then(|counts| counts.get(proof_index))
                .copied()
                .ok_or(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                    "missing fixed-capacity active-child count",
                ))?;
            let record = observe_active_child_count_transcript_v4(
                &active_count_air.profile,
                proof_index,
                active_child_count,
                proof_index + 1 == segment_count,
                &mut transcript,
            )
            .map_err(DirectLogUpOnlyCompositeContextErrorV19::Transcript)?;
            let sampled_challenge = if record.sampled == EF::ZERO {
                EF::ONE
            } else {
                record.sampled
            };
            let mapped = records
                .mapped_sources
                .iter()
                .find(|record| record.proof_index as usize == proof_index)
                .ok_or(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                    "missing fixed-capacity mapped source",
                ))?;
            if mapped.active_count_challenge != sampled_challenge {
                return Err(DirectLogUpOnlyCompositeContextErrorV19::Transcript(
                    "fixed-capacity active-child-count challenge",
                ));
            }
            active_count_records_v4.push(record);
        }
        let segment_claim_transcripts = records
            .claim_transcripts
            .iter()
            .filter(|record| record.proof_index as usize == proof_index)
            .collect::<Vec<_>>();
        let segment_reductions = records
            .reductions
            .iter()
            .filter(|record| record.proof_index as usize == proof_index)
            .collect::<Vec<_>>();
        if segment_claim_transcripts.len() != segment_reductions.len() {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "segment one-shot count",
            ));
        }
        for (claim, reduction) in segment_claim_transcripts
            .into_iter()
            .zip(segment_reductions)
        {
            observe_one_shot_v19(&mut transcript, claim, reduction)?;
        }
        let finalization_start = transcript.len();
        FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(
            &mut transcript,
            F::from_u64(LOGUP_END_BOUNDARY_TAG_V19),
        );
        let final_sample =
            FiatShamirTranscript::<BabyBearPoseidon2Config>::sample_ext(&mut transcript);
        let final_checkpoint = transcript.inner.checkpoint();
        if final_checkpoint.absorb_idx != 0 {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Transcript(
                "LogUp finalization is not a squeeze boundary",
            ));
        }
        let final_tidx = transcript.len();
        let shard_count = records
            .boundary_shards
            .iter()
            .find(|record| record.proof_index as usize == proof_index && record.shard_ordinal == 0)
            .ok_or(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "missing segment boundary first row",
            ))?
            .shard_count;
        finalization_records.push(LogUpFinalizationRecordV19 {
            proof_index: proof_index as u32,
            segment_index,
            shard_count,
            start_tidx: u32::try_from(finalization_start).map_err(|_| {
                DirectLogUpOnlyCompositeContextErrorV19::Shape("finalization cursor exceeds u32")
            })?,
            end_tidx: u32::try_from(final_tidx).map_err(|_| {
                DirectLogUpOnlyCompositeContextErrorV19::Shape("final cursor exceeds u32")
            })?,
            sampled: final_sample,
            checkpoint_state: final_checkpoint.state,
        });
        checkpoint_targets.push([preflight.batch_constraint.post_tidx, final_tidx]);
        let full_log = transcript.into_log();
        let suffix = full_log.suffix(transcript_checkpoint).ok_or(
            DirectLogUpOnlyCompositeContextErrorV19::Transcript("LogUp suffix checkpoint"),
        )?;
        if suffix.values().get(..preflight.transcript.len()) != Some(preflight.transcript.values())
            || suffix.samples().get(..preflight.transcript.len())
                != Some(preflight.transcript.samples())
        {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Transcript(
                "partial verifier suffix",
            ));
        }

        let first_boundary = records
            .boundary_shards
            .iter()
            .find(|record| record.proof_index as usize == proof_index && record.shard_ordinal == 0)
            .ok_or(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "missing segment boundary first row",
            ))?;
        if first_boundary.verifier_endpoint != preflight.batch_constraint.final_claim {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "boundary verifier endpoint",
            ));
        }
        endpoint_records.push(LogUpEndpointBridgeRecordV19 {
            proof_index: proof_index as u32,
            segment_index,
            end_tidx: u32::try_from(preflight.batch_constraint.post_tidx).map_err(|_| {
                DirectLogUpOnlyCompositeContextErrorV19::Shape("endpoint cursor exceeds u32")
            })?,
            app_vk_digest: module.source_profile.app_vk_digest,
            source_forest_root: source_traces.forest_roots[proof_index],
            segment_openings_digest: records.sources[proof_index].segment_openings_digest,
            verifier_endpoint: preflight.batch_constraint.final_claim,
            shard_count: first_boundary.shard_count,
            checkpoint_state: endpoint_checkpoint.state,
        });
        logup_producers.push(LogUpOnlyProducerRecordV19 {
            proof_index: proof_index as u32,
            segment_index,
            app_vk_digest: module.source_profile.app_vk_digest,
            source_forest_root: source_traces.forest_roots[proof_index],
            segment_openings_digest: records.sources[proof_index].segment_openings_digest,
            verifier_endpoint: first_boundary
                .verifier_endpoint
                .as_basis_coefficients_slice()
                .try_into()
                .map_err(|_| {
                    DirectLogUpOnlyCompositeContextErrorV19::Shape(
                        "verifier endpoint extension degree",
                    )
                })?,
            segment_sum_before: [F::ZERO; D_EF],
            segment_sum_after: [F::ZERO; D_EF],
            start_checkpoint: TranscriptCheckpointRecordV19 {
                operation_index: u32::try_from(preflight.batch_constraint.post_tidx).map_err(
                    |_| {
                        DirectLogUpOnlyCompositeContextErrorV19::Shape(
                            "endpoint checkpoint cursor exceeds u32",
                        )
                    },
                )?,
                sample_count: u8::try_from(DIGEST_SIZE - endpoint_checkpoint.sample_idx).map_err(
                    |_| {
                        DirectLogUpOnlyCompositeContextErrorV19::Shape(
                            "endpoint checkpoint sample count exceeds u8",
                        )
                    },
                )?,
                state: endpoint_checkpoint.state,
            },
            end_checkpoint: TranscriptCheckpointRecordV19 {
                operation_index: u32::try_from(final_tidx).map_err(|_| {
                    DirectLogUpOnlyCompositeContextErrorV19::Shape(
                        "final checkpoint cursor exceeds u32",
                    )
                })?,
                sample_count: u8::try_from(DIGEST_SIZE - final_checkpoint.sample_idx).map_err(
                    |_| {
                        DirectLogUpOnlyCompositeContextErrorV19::Shape(
                            "final checkpoint sample count exceeds u8",
                        )
                    },
                )?,
                state: final_checkpoint.state,
            },
        });
        let mut counts = vec![0u32; preflight.batch_constraint.sumcheck_rnd.len()];
        for term in module
            .mapped_functional_air
            .profile
            .terms
            .iter()
            .filter(|term| {
                term.proof_index as usize == proof_index
                    && !term.is_program_term
                    && !term.is_active_count_term
            })
        {
            let folded_len = usize::from(term.log_height).saturating_sub(term.l_skip as usize);
            for count in counts.iter_mut().take(folded_len + 1) {
                *count =
                    count
                        .checked_add(1)
                        .ok_or(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                            "opening-point fanout count overflow",
                        ))?;
            }
        }
        if counts.contains(&0) {
            return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "unconsumed LogUp opening-point coordinate",
            ));
        }
        opening_point_records.push(OpeningPointFanoutRecordV19 {
            proof_index: proof_index as u32,
            segment_index,
            values: preflight.batch_constraint.sumcheck_rnd.clone(),
            consumer_counts: counts,
        });
        shaped_proofs.push(proof);
        preflights.push(preflight);
        prefix_logs.push(prefix_log);
        extended_logs.push(suffix);
    }

    let (prefix_context, mut prefix_permutation_inputs, prefix_compression_inputs) = module
        .prefix_transcript
        .generate_trace::<SC>(&prefix_logs, &prefix_end_indices)
        .ok_or(DirectLogUpOnlyCompositeContextErrorV19::Trace(
            "prefix transcript",
        ))?;
    let mut additional_permutation_inputs = source_traces.poseidon_permute_inputs.clone();
    additional_permutation_inputs.append(&mut prefix_permutation_inputs);
    let mut additional_compression_inputs = source_traces.poseidon_compress_inputs.clone();
    additional_compression_inputs.extend(prefix_compression_inputs);
    if !module.fixed_multi_air {
        additional_compression_inputs
            .extend(boundary_compression_inputs_v19(&records.boundary_shards)?);
    }
    let mut tail_contexts = Vec::new();
    tail_contexts.push(prefix_context);
    tail_contexts.push(AirProvingContext::simple_no_pis(source_traces.instance));
    tail_contexts.push(AirProvingContext::simple_no_pis(source_traces.manifest));
    tail_contexts.push(AirProvingContext::simple_no_pis(source_traces.forest_nodes));
    tail_contexts.push(AirProvingContext::simple_no_pis(
        generate_prefix_checkpoint_bridge_trace_v19(&prefix_bridge_records)
            .map_err(DirectLogUpOnlyCompositeContextErrorV19::Trace)?,
    ));
    let opening_point_fanout_trace = if runtime_active_child_counts.is_some() {
        generate_opening_point_fanout_trace_fixed_capacity_v4(
            &module.opening_point_fanout_air,
            &opening_point_records,
        )
    } else {
        generate_opening_point_fanout_trace_v19(&opening_point_records)
    }
    .map_err(DirectLogUpOnlyCompositeContextErrorV19::Trace)?;
    tail_contexts.push(AirProvingContext::simple_no_pis(opening_point_fanout_trace));
    tail_contexts.push(AirProvingContext::simple_no_pis(
        generate_forest_leaf_fanout_trace_v19(&records.boundary_shards, module.fixed_multi_air)
            .map_err(DirectLogUpOnlyCompositeContextErrorV19::Trace)?,
    ));
    tail_contexts.push(AirProvingContext::simple_no_pis(
        generate_logup_endpoint_bridge_trace_v19(&endpoint_records)
            .map_err(DirectLogUpOnlyCompositeContextErrorV19::Trace)?,
    ));
    let opening_start_tidxs = endpoint_records
        .iter()
        .map(|record| record.end_tidx)
        .collect::<Vec<_>>();
    tail_contexts.push(AirProvingContext::simple_no_pis(
        generate_column_opening_observation_trace_v19(
            &module.column_opening_observation_air,
            &records.mapped_sources,
            &opening_start_tidxs,
        )
        .map_err(DirectLogUpOnlyCompositeContextErrorV19::Trace)?,
    ));
    tail_contexts.push(AirProvingContext::simple_no_pis(
        generate_logup_claim_derivation_trace_v19(&records.claim_derivations)
            .map_err(DirectLogUpOnlyCompositeContextErrorV19::Trace)?,
    ));
    if let Some(active_count_air) = &module.active_count_transcript_air {
        tail_contexts.push(AirProvingContext::simple_no_pis(
            generate_active_child_count_transcript_trace_v19(
                active_count_air,
                &active_count_records,
            )
            .map_err(DirectLogUpOnlyCompositeContextErrorV19::Trace)?,
        ));
    }
    if let Some(active_count_air) = &module.active_count_transcript_air_v4 {
        tail_contexts.push(AirProvingContext::simple_no_pis(
            generate_active_child_count_transcript_trace_v4(
                active_count_air,
                &active_count_records_v4,
            )
            .map_err(DirectLogUpOnlyCompositeContextErrorV19::Trace)?,
        ));
    }
    let mapped_trace = if let Some(active_child_counts) = runtime_active_child_counts {
        generate_mapped_functional_trace_fixed_capacity_v4(
            &module.mapped_functional_air.profile,
            &records.mapped_sources,
            &records.claim_transcripts,
            active_child_counts,
        )
    } else {
        generate_mapped_functional_trace_v19(
            &module.mapped_functional_air.profile,
            &records.mapped_sources,
            &records.claim_transcripts,
        )
    }
    .map_err(DirectLogUpOnlyCompositeContextErrorV19::Trace)?;
    tail_contexts.push(AirProvingContext::simple_no_pis(mapped_trace));
    for reduction_air in &module.reduction_airs {
        let grouped = records
            .reductions
            .iter()
            .filter(|record| record.point.len() == reduction_air.log_message_len())
            .cloned()
            .collect::<Vec<_>>();
        tail_contexts.push(AirProvingContext::simple_no_pis(
            generate_grouped_one_shot_reduction_trace_v19(reduction_air, &grouped)
                .map_err(DirectLogUpOnlyCompositeContextErrorV19::Trace)?,
        ));
    }
    tail_contexts.push(AirProvingContext::simple_no_pis(
        generate_logup_finalization_trace_v19(&finalization_records)
            .map_err(DirectLogUpOnlyCompositeContextErrorV19::Trace)?,
    ));
    if !module.fixed_multi_air {
        tail_contexts.push(AirProvingContext::simple_no_pis(
            generate_logup_swirl_boundary_trace_v19(&module.boundary_air, &records.boundary_shards)
                .map_err(DirectLogUpOnlyCompositeContextErrorV19::Trace)?,
        ));
    }

    let airs = module.airs_without_poseidon::<SC>();
    let partial_air_count = module.partial.airs_without_poseidon::<SC>().len();
    let tail_airs =
        airs.get(partial_air_count..)
            .ok_or(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "partial AIR prefix",
            ))?;
    if tail_airs.len() != tail_contexts.len()
        || tail_airs.iter().zip(&tail_contexts).any(|(air, context)| {
            air.common_main_width() != context.common_main.width()
                || air.cached_main_widths().len() != context.cached_mains.len()
        })
    {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
            "AIR/context order or width",
        ));
    }
    Ok(DirectLogUpOnlyCompositePreparedV19 {
        shaped_proofs,
        preflights,
        extended_logs,
        checkpoint_targets,
        additional_permutation_inputs,
        additional_compression_inputs,
        tail_contexts,
        logup_producers,
    })
}

fn validate_logup_only_prepared_v19<SC: StarkProtocolConfig<F = F>>(
    prepared: &DirectLogUpOnlyCompositePreparedV19<SC>,
) -> Result<(), DirectLogUpOnlyCompositeContextErrorV19> {
    use openvm_recursion_circuit::system::VerifierEquationMode;

    if prepared.shaped_proofs.len() != prepared.preflights.len()
        || prepared.extended_logs.len() != prepared.preflights.len()
        || prepared.checkpoint_targets.len() != prepared.preflights.len()
        || prepared.preflights.iter().any(|preflight| {
            preflight.batch_constraint.equation_mode != VerifierEquationMode::LogUpOnly
        })
    {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::PartialVerifier);
    }
    Ok(())
}

/// Public packet-only form of
/// [`DirectLogUpOnlyVerifierModuleV19::generate_proving_contexts_without_poseidon`].
pub fn generate_direct_logup_only_composite_contexts_without_poseidon_v19<
    SC: StarkProtocolConfig<F = F>,
    const MAX_NUM_PROOFS: usize,
>(
    module: &DirectLogUpOnlyVerifierModuleV19<MAX_NUM_PROOFS>,
    records: &DirectLogUpOnlyCompositeRecordsV19,
) -> Result<DirectLogUpOnlyCompositePacketContextsV19<SC>, DirectLogUpOnlyCompositeContextErrorV19>
{
    let prepared =
        prepare_direct_logup_only_composite_v19::<SC, MAX_NUM_PROOFS>(module, records, None)?;
    generate_direct_logup_only_composite_contexts_from_prepared_cpu_v19(module, prepared)
}

/// Packet-only fixed-capacity path. Runtime counts are threaded through the
/// transcript and mapped-functional main traces but never through keygen.
pub fn generate_direct_logup_only_composite_contexts_without_poseidon_fixed_capacity_v4<
    SC: StarkProtocolConfig<F = F>,
    const MAX_NUM_PROOFS: usize,
>(
    module: &DirectLogUpOnlyVerifierModuleV19<MAX_NUM_PROOFS>,
    records: &DirectLogUpOnlyCompositeRecordsV19,
    active_child_counts: &[u8],
) -> Result<DirectLogUpOnlyCompositePacketContextsV19<SC>, DirectLogUpOnlyCompositeContextErrorV19>
{
    let prepared = prepare_direct_logup_only_composite_v19::<SC, MAX_NUM_PROOFS>(
        module,
        records,
        Some(active_child_counts),
    )?;
    generate_direct_logup_only_composite_contexts_from_prepared_cpu_v19(module, prepared)
}

fn generate_direct_logup_only_composite_contexts_from_prepared_cpu_v19<
    SC: StarkProtocolConfig<F = F>,
    const MAX_NUM_PROOFS: usize,
>(
    module: &DirectLogUpOnlyVerifierModuleV19<MAX_NUM_PROOFS>,
    prepared: DirectLogUpOnlyCompositePreparedV19<SC>,
) -> Result<DirectLogUpOnlyCompositePacketContextsV19<SC>, DirectLogUpOnlyCompositeContextErrorV19>
{
    validate_logup_only_prepared_v19(&prepared)?;
    let DirectLogUpOnlyCompositePreparedV19 {
        shaped_proofs,
        preflights,
        extended_logs,
        checkpoint_targets,
        additional_permutation_inputs,
        additional_compression_inputs,
        tail_contexts,
        logup_producers,
    } = prepared;
    let cached_trace_ctx =
        CachedTraceCtx::Records(module.partial.cached_trace_record(&module.child_vk));
    let partial_packet = module
        .partial
        .generate_proving_ctxs_extended_for_shared_poseidon::<SC>(
            &module.child_vk,
            cached_trace_ctx,
            &shaped_proofs,
            &preflights,
            Some(&extended_logs),
            Some(&checkpoint_targets),
            additional_permutation_inputs,
            additional_compression_inputs,
        )
        .ok_or(DirectLogUpOnlyCompositeContextErrorV19::PartialVerifier)?;
    let mut contexts = partial_packet.contexts;
    contexts.extend(tail_contexts);
    let airs = module.airs_without_poseidon::<SC>();
    if airs.len() != contexts.len()
        || airs.iter().zip(&contexts).any(|(air, context)| {
            air.common_main_width() != context.common_main.width()
                || air.cached_main_widths().len() != context.cached_mains.len()
        })
    {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
            "AIR/context order or width",
        ));
    }
    Ok(DirectLogUpOnlyCompositePacketContextsV19 {
        contexts,
        poseidon: DirectLogUpOnlyCompositePoseidonPacketV19 {
            owner: module.shared_poseidon_owner(),
            permutation_inputs: partial_packet.poseidon2_permutation_inputs,
            compression_inputs: partial_packet.poseidon2_compression_inputs,
        },
        logup_producers,
    })
}

#[cfg(feature = "cuda")]
fn generate_direct_logup_only_composite_contexts_from_prepared_cuda_v19<
    const MAX_NUM_PROOFS: usize,
>(
    module: &DirectLogUpOnlyVerifierModuleV19<MAX_NUM_PROOFS>,
    prepared: DirectLogUpOnlyCompositePreparedV19<BabyBearPoseidon2Config>,
    cached_trace_ctx: CachedTraceCtx<openvm_cuda_backend::GpuBackend>,
    device_ctx: &openvm_cuda_common::stream::GpuDeviceCtx,
) -> Result<DirectLogUpOnlyCompositeCudaPacketV19, DirectLogUpOnlyCompositeContextErrorV19> {
    use openvm_recursion_circuit::batch_constraint::cuda_tracegen::LogUpOnlyCudaProvingContext;

    validate_logup_only_prepared_v19(&prepared)?;
    let DirectLogUpOnlyCompositePreparedV19 {
        shaped_proofs,
        preflights,
        extended_logs,
        checkpoint_targets,
        additional_permutation_inputs,
        additional_compression_inputs,
        tail_contexts,
        logup_producers,
    } = prepared;
    let partial_packet = module
        .partial
        .generate_proving_ctxs_extended_for_shared_poseidon_cuda(
            &module.child_vk,
            cached_trace_ctx,
            &shaped_proofs,
            &preflights,
            Some(&extended_logs),
            Some(&checkpoint_targets),
            additional_permutation_inputs,
            additional_compression_inputs,
            device_ctx,
        )
        .ok_or(DirectLogUpOnlyCompositeContextErrorV19::PartialVerifier)?;

    let mut contexts = Vec::with_capacity(partial_packet.contexts.len() + tail_contexts.len());
    contexts.extend(
        partial_packet
            .contexts
            .into_iter()
            .map(|context| match context {
                LogUpOnlyCudaProvingContext::Device(context) => {
                    DirectLogUpOnlyCompositeCudaContextV19::Device(context)
                }
                LogUpOnlyCudaProvingContext::CpuTranscript(context) => {
                    DirectLogUpOnlyCompositeCudaContextV19::HostExtendedTranscript(context)
                }
                LogUpOnlyCudaProvingContext::CpuRebasedProofShape(context) => {
                    DirectLogUpOnlyCompositeCudaContextV19::HostRebasedTranscriptAdapter(context)
                }
            }),
    );
    contexts.extend(
        tail_contexts
            .into_iter()
            .map(DirectLogUpOnlyCompositeCudaContextV19::HostV19),
    );

    let airs = module.airs_without_poseidon::<BabyBearPoseidon2Config>();
    let shape_matches = |air: &AirRef<BabyBearPoseidon2Config>,
                         context: &DirectLogUpOnlyCompositeCudaContextV19| {
        let (common_width, cached_count) = match context {
            DirectLogUpOnlyCompositeCudaContextV19::Device(context) => (
                openvm_stark_backend::prover::MatrixDimensions::width(&context.common_main),
                context.cached_mains.len(),
            ),
            DirectLogUpOnlyCompositeCudaContextV19::HostExtendedTranscript(context)
            | DirectLogUpOnlyCompositeCudaContextV19::HostRebasedTranscriptAdapter(context)
            | DirectLogUpOnlyCompositeCudaContextV19::HostV19(context) => {
                (context.common_main.width(), context.cached_mains.len())
            }
        };
        air.common_main_width() == common_width && air.cached_main_widths().len() == cached_count
    };
    if airs.len() != contexts.len()
        || airs
            .iter()
            .zip(&contexts)
            .any(|(air, context)| !shape_matches(air, context))
    {
        return Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
            "AIR/context order or width",
        ));
    }
    Ok(DirectLogUpOnlyCompositeCudaPacketV19 {
        contexts,
        poseidon: DirectLogUpOnlyCompositePoseidonPacketV19 {
            owner: module.shared_poseidon_owner(),
            permutation_inputs: partial_packet.poseidon2_permutation_inputs,
            compression_inputs: partial_packet.poseidon2_compression_inputs,
        },
        logup_producers,
    })
}

/// Public mixed CUDA form of
/// [`DirectLogUpOnlyVerifierModuleV19::generate_proving_contexts_without_poseidon_cuda`].
#[cfg(feature = "cuda")]
pub fn generate_direct_logup_only_composite_contexts_without_poseidon_cuda_v19<
    const MAX_NUM_PROOFS: usize,
>(
    module: &DirectLogUpOnlyVerifierModuleV19<MAX_NUM_PROOFS>,
    records: &DirectLogUpOnlyCompositeRecordsV19,
    cached_trace_ctx: CachedTraceCtx<openvm_cuda_backend::GpuBackend>,
    device_ctx: &openvm_cuda_common::stream::GpuDeviceCtx,
) -> Result<DirectLogUpOnlyCompositeCudaPacketV19, DirectLogUpOnlyCompositeContextErrorV19> {
    let prepared = prepare_direct_logup_only_composite_v19::<
        BabyBearPoseidon2Config,
        MAX_NUM_PROOFS,
    >(module, records, None)?;
    generate_direct_logup_only_composite_contexts_from_prepared_cuda_v19(
        module,
        prepared,
        cached_trace_ctx,
        device_ctx,
    )
}

/// Fixed-capacity CUDA packet path with the same runtime-count validation and
/// host tail traces as the CPU implementation.
#[cfg(feature = "cuda")]
pub fn generate_direct_logup_only_composite_contexts_without_poseidon_cuda_fixed_capacity_v4<
    const MAX_NUM_PROOFS: usize,
>(
    module: &DirectLogUpOnlyVerifierModuleV19<MAX_NUM_PROOFS>,
    records: &DirectLogUpOnlyCompositeRecordsV19,
    active_child_counts: &[u8],
    cached_trace_ctx: CachedTraceCtx<openvm_cuda_backend::GpuBackend>,
    device_ctx: &openvm_cuda_common::stream::GpuDeviceCtx,
) -> Result<DirectLogUpOnlyCompositeCudaPacketV19, DirectLogUpOnlyCompositeContextErrorV19> {
    let prepared = prepare_direct_logup_only_composite_v19::<
        BabyBearPoseidon2Config,
        MAX_NUM_PROOFS,
    >(module, records, Some(active_child_counts))?;
    generate_direct_logup_only_composite_contexts_from_prepared_cuda_v19(
        module,
        prepared,
        cached_trace_ctx,
        device_ctx,
    )
}

#[cfg(test)]
pub(crate) mod tests {
    use openvm_cpu_backend::logup_zerocheck::prove_logup_only_recorded;
    #[cfg(feature = "cuda")]
    use openvm_cuda_backend::data_transporter::assert_eq_host_and_device_matrix;
    #[cfg(feature = "cuda")]
    use openvm_cuda_common::stream::GpuDeviceCtx;
    use openvm_recursion_circuit::utils::poseidon2_hash_slice_with_states;
    use openvm_stark_backend::{
        air_builders::{
            debug::{check_constraints, check_logup},
            symbolic::get_symbolic_builder,
        },
        interaction::{BusIndex, SymbolicInteraction},
        keygen::types::TraceWidth,
        native_warp::{
            prove_direct_message_opening_reduction, DirectAirCodeClass, DirectAirPesatInstance,
            DirectAirPesatWitness, DirectAirPublicSchema, FixedMultiAirPesatIndex,
            FixedMultiAirPesatInstance, NativeWarpChallenger,
        },
        prover::{
            stacked_pcs::stacked_commit, AirProvingContext, ColMajorMatrix, CommittedTraceData,
            CpuColMajorBackend, DeviceDataTransporter, MatrixDimensions, ProvingContext,
        },
        test_utils::{
            default_test_params_small, CachedFixture11, FibFixture, InteractionsFixture11,
            MixtureFixture, MixtureFixtureEnum, TestFixture,
        },
        warp_accum::{WarpLinearCode, WhirInitialRsWarpCode, WhirRsCodeProverData},
        AnyAir, StarkEngine,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::{
        BabyBearPoseidon2Config, BabyBearPoseidon2CpuEngine, Digest, DuplexSponge,
    };
    use p3_field::ExtensionField;

    use super::*;
    use crate::circuit::{
        native_warp_history_v19::{
            compute_program_fingerprint_digest_v19, generate_one_shot_claim_transcript_trace_v19,
            DirectLogUpFreshSourceRecordV19, DirectLogUpSourceEntryProfileV19,
            OneShotClaimTranscriptColsV19, DIRECT_LOGUP_CLAIM_TAG_V19,
            PROGRAM_FINGERPRINT_TARGET_TAG_V19, SOURCE_INSTANCE_TAG_V19,
        },
        verifier_warp_history_v2::{
            generate_fixed_setup_opening_trace_v2, FixedSetupMatrixV2, FixedSetupOpeningAirV2,
            FixedSetupOpeningCertificateBusV2, FixedSetupOpeningClaimRecordV2,
            FixedSetupOpeningInstanceV2, FixedSetupOpeningNodeBusV2, FixedSetupOpeningPointBusV2,
            FixedSetupOpeningProfileV2, FixedSetupOpeningProofRecordV2,
            VerifiedFixedSetupOpeningPairBusV2,
        },
    };

    #[derive(Clone, Debug)]
    struct AppendixDSourceAir {
        num_public_values: usize,
    }

    impl BaseAir<F> for AppendixDSourceAir {
        fn width(&self) -> usize {
            2
        }
    }

    impl BaseAirWithPublicValues<F> for AppendixDSourceAir {
        fn num_public_values(&self) -> usize {
            self.num_public_values
        }
    }

    impl PartitionedBaseAir<F> for AppendixDSourceAir {}

    impl<AB> Air<AB> for AppendixDSourceAir
    where
        AB: p3_air::AirBuilderWithPublicValues<F = F> + InteractionBuilder,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.row_slice(0).expect("Appendix-D source row");
            let multiplicity = AB::Expr::from(local[0]);
            let public_values = builder.public_values().to_vec();
            for public_value in public_values {
                let fields = [AB::Expr::from(local[1]), public_value.into()];
                builder.push_interaction(0, fields.clone(), multiplicity.clone(), 1);
                builder.push_interaction(0, fields, -multiplicity.clone(), 1);
            }
        }
    }

    /// Test-only source material shared byte-for-byte by the recursive source
    /// reduction and the Appendix-D VACC replay.
    pub(crate) struct RetainedAppendixDSourceV19 {
        pub(crate) relation: Arc<FixedMultiAirPesatIndex<F, Digest>>,
        pub(crate) message: Vec<EF>,
        pub(crate) explicit: Vec<EF>,
        pub(crate) root: Digest,
        pub(crate) base_prover_data: WhirRsCodeProverData<F, Digest>,
        pub(crate) base_codeword: Vec<F>,
        pub(crate) expected_instance_digest: Digest,
    }

    fn composite_fixture() -> (
        BabyBearPoseidon2CpuEngine<DuplexSponge>,
        DirectLogUpOnlyVerifierModuleV19<4>,
        DirectLogUpSourceManifestProfileV19,
        Vec<DirectLogUpSegmentSourceRecordV19>,
    ) {
        let mut params = default_test_params_small();
        params.l_skip = OPENVM_DIRECT_LOGUP_L_SKIP_V19;
        params.logup.pow_bits = 0;
        params.logup.log_max_message_length = 10;
        params.max_constraint_degree = 8;
        let engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
        let fixture = MixtureFixture::<BabyBearPoseidon2Config>::new(vec![
            MixtureFixtureEnum::FibFixture(FibFixture::new(1, 1, 32)),
            MixtureFixtureEnum::InteractionsFixture11(InteractionsFixture11),
        ]);
        let (_child_pk, mut child_vk) = fixture.keygen(&engine);
        assert_eq!(child_vk.inner.params.l_skip, OPENVM_DIRECT_LOGUP_L_SKIP_V19);
        assert!(child_vk.inner.per_air.len() >= 2);

        // MixtureFixture uses dense test AIR ids that overlap OpenVM's fixed
        // Program/Connector ids. Install two copies at non-system ids so this
        // test exercises the ordinary multi-AIR mapped path.
        let selected_air_vks = child_vk.inner.per_air[..2].to_vec();
        while child_vk.inner.per_air.len() < 10 {
            child_vk.inner.per_air.push(selected_air_vks[0].clone());
        }
        child_vk.inner.per_air.extend(selected_air_vks.clone());

        let log_height = 5usize;
        let mut relations = Vec::new();
        for (offset, air_vk) in selected_air_vks.iter().enumerate() {
            let air_id = 10 + offset;
            assert!(air_vk.preprocessed_data.is_none());
            let raw_width = air_vk.params.width.common_main
                + air_vk.params.width.cached_mains.iter().sum::<usize>();
            let raw_len = (1usize << log_height) * raw_width;
            let log_message_len = raw_len.next_power_of_two().ilog2() as u8;
            let relation = DirectAirPesatIndex::from_verifying_key(
                engine.config().hasher(),
                child_vk.pre_hash,
                air_id,
                log_height,
                air_vk,
                None,
                DirectAirPublicSchema {
                    public_values_len: air_vk.params.num_public_values as u32,
                    boundary_values_len: 0,
                    schema_digest: [F::from_usize(100 + air_id); DIGEST_SIZE],
                },
                DirectAirCodeClass {
                    log_message_len,
                    log_blowup: child_vk.inner.params.log_blowup as u8,
                    log_codeword_len: log_message_len + child_vk.inner.params.log_blowup as u8,
                    initial_folding_factor: 0,
                    rows_per_query: 2,
                },
            )
            .expect("direct test relation");
            relations.push(Arc::new(relation));
        }

        let source_entries = relations
            .iter()
            .enumerate()
            .map(|(shard_id, relation)| {
                DirectLogUpSourceEntryProfileV19::from_relation(
                    shard_id as u32,
                    relation.description(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let plan = vec![vec![0, 1]];
        let source_profile = DirectLogUpSourceManifestProfileV19::new(
            child_vk.pre_hash,
            [F::from_u32(211); DIGEST_SIZE],
            source_entries.clone(),
            &plan,
        )
        .unwrap();
        let mapped_entries = relations
            .into_iter()
            .enumerate()
            .map(|(shard_id, relation)| DirectLogUpMappedRegistryEntryV19 {
                shard_id: shard_id as u32,
                relation,
            })
            .collect();
        let module = DirectLogUpOnlyVerifierModuleV19::<4>::new(
            Arc::new(child_vk),
            source_profile.clone(),
            mapped_entries,
            &plan,
        )
        .unwrap();

        let sources = source_entries
            .iter()
            .map(|entry| {
                let public_values = vec![F::ZERO; entry.public_values_len as usize];
                let boundary_values = vec![F::ZERO; entry.boundary_values_len as usize];
                let mut preimage = vec![
                    F::from_u64(SOURCE_INSTANCE_TAG_V19),
                    F::from_usize(public_values.len()),
                    F::from_usize(boundary_values.len()),
                ];
                preimage.extend_from_slice(&public_values);
                preimage.extend_from_slice(&boundary_values);
                let (expected_instance_digest, _, _) = poseidon2_hash_slice_with_states(&preimage);
                DirectLogUpFreshSourceRecordV19 {
                    shard_id: entry.shard_id,
                    source_root: [F::from_u32(300 + entry.shard_id); DIGEST_SIZE],
                    expected_instance_digest,
                    public_values,
                    boundary_values,
                }
            })
            .collect();
        let records = vec![DirectLogUpSegmentSourceRecordV19 {
            proof_index: 0,
            segment_index: 0,
            prefix_start_tidx: 0,
            alignment_sample: [F::from_u32(401); D_EF],
            segment_openings_digest: [F::from_u32(409); DIGEST_SIZE],
            sources,
        }];
        (engine, module, source_profile, records)
    }

    fn repeat_col_major(matrix: &ColMajorMatrix<F>, factor: usize) -> ColMajorMatrix<F> {
        let mut values = Vec::with_capacity(matrix.values.len() * factor);
        for column in matrix.columns() {
            for _ in 0..factor {
                values.extend_from_slice(column);
            }
        }
        ColMajorMatrix::new(values, matrix.width())
    }

    fn fixed_capacity_v4_constructor_fixture() -> (
        BabyBearPoseidon2CpuEngine<DuplexSponge>,
        Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
        DirectLogUpSourceManifestProfileV19,
        FixedMultiAirMappedFunctionalConfigV4,
    ) {
        let mut params = default_test_params_small();
        params.l_skip = OPENVM_DIRECT_LOGUP_L_SKIP_V19;
        params.logup.pow_bits = 0;
        params.logup.log_max_message_length = 10;
        params.max_constraint_degree = 8;
        let engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
        let fixture = FibFixture::new(1, 1, 32);
        let (_child_pk, child_vk) = fixture.keygen(&engine);
        let air_vk = &child_vk.inner.per_air[0];
        let log_height = 5usize;
        let raw_len = (1usize << log_height)
            * (air_vk.params.width.common_main
                + air_vk.params.width.cached_mains.iter().sum::<usize>());
        let log_message_len = raw_len.next_power_of_two().ilog2() as u8;
        // Exercise the protocol boundary explicitly: the outer WARP source
        // code rate need not equal the child SWIRL proof's PCS rate.
        let warp_log_blowup = u8::try_from(child_vk.inner.params.log_blowup + 1).unwrap();
        let log_codeword_len = log_message_len + warp_log_blowup;
        let relation = Arc::new(
            DirectAirPesatIndex::from_verifying_key(
                engine.config().hasher(),
                child_vk.pre_hash,
                0,
                log_height,
                air_vk,
                None,
                DirectAirPublicSchema {
                    public_values_len: air_vk.params.num_public_values as u32,
                    boundary_values_len: 0,
                    schema_digest: [F::from_u32(0x44_01); DIGEST_SIZE],
                },
                DirectAirCodeClass {
                    log_message_len,
                    log_blowup: warp_log_blowup,
                    log_codeword_len,
                    initial_folding_factor: 0,
                    rows_per_query: 2,
                },
            )
            .expect("fixed-capacity direct relation"),
        );
        let relation_digest = [F::from_u32(0x44_02); DIGEST_SIZE];
        let registry_digest = [F::from_u32(0x44_03); DIGEST_SIZE];
        let source_public_values_len = air_vk.params.num_public_values as u32;
        let source_entry = DirectLogUpSourceEntryProfileV19::fixed_multi_air(
            relation_digest,
            log_message_len,
            log_codeword_len,
            source_public_values_len,
        )
        .unwrap();
        let source_profile = DirectLogUpSourceManifestProfileV19::new_fixed_capacity_v4(
            child_vk.pre_hash,
            registry_digest,
            source_entry,
        )
        .unwrap();
        let config = FixedMultiAirMappedFunctionalConfigV4 {
            relation_digest,
            active_count_profile_digest: registry_digest,
            log_message_len,
            log_codeword_len,
            source_public_values_len,
            regions: Arc::from([FixedMultiAirMappedRegionV19 {
                air_id: 0,
                message_start: 0,
                relation,
            }]),
            vm_pvs_air_id: 0,
            is_valid_common_main_column: 0,
            is_valid_message_block_start: 0,
            vm_pvs_log_height: log_height as u8,
            profile_trace_heights: Arc::from([1u32 << log_height]),
            authenticate_cached_setup_claims: false,
        };
        (engine, Arc::new(child_vk), source_profile, config)
    }

    fn fixed_capacity_v4_active_count_records(
        occupancy: usize,
    ) -> Vec<ActiveChildCountTranscriptRecordV4> {
        (0..occupancy)
            .map(|slot| ActiveChildCountTranscriptRecordV4 {
                proof_index: slot as u32,
                active_child_count: if slot + 1 == occupancy {
                    occupancy as u8
                } else {
                    VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 as u8
                },
                start_tidx: 2_000 + slot as u32 * 200,
                sampled: EF::from_u32(2_100 + slot as u32),
            })
            .collect()
    }

    fn fixed_capacity_v4_air_identity(
        airs: &[AirRef<BabyBearPoseidon2Config>],
    ) -> Vec<(
        String,
        usize,
        usize,
        Vec<usize>,
        Option<(usize, usize, Vec<F>)>,
    )> {
        airs.iter()
            .map(|air| {
                let preprocessed = BaseAir::<F>::preprocessed_trace(air.as_ref()).map(|matrix| {
                    (
                        Matrix::width(&matrix),
                        Matrix::height(&matrix),
                        matrix.values,
                    )
                });
                (
                    air.name(),
                    air.common_main_width(),
                    air.num_public_values(),
                    air.cached_main_widths(),
                    preprocessed,
                )
            })
            .collect()
    }

    #[test]
    fn fixed_capacity_v4_constructor_has_one_key_for_all_runtime_occupancies() {
        let (engine, child_vk, source_profile, config) = fixed_capacity_v4_constructor_fixture();
        let module = DirectLogUpOnlyVerifierModuleV19::<4>::new_fixed_capacity_v4(
            child_vk,
            source_profile,
            config,
        )
        .expect("fixed-capacity production module");
        assert_eq!(
            module
                .airs_without_poseidon::<BabyBearPoseidon2Config>()
                .len(),
            40,
            "the runtime active-count verifier is part of the fixed source AIR inventory",
        );
        assert!(module.active_count_transcript_air.is_none());
        let active_count_air = module
            .active_count_transcript_air_v4
            .as_ref()
            .expect("runtime active-count AIR");
        assert!(module.mapped_functional_air.profile.runtime_capacity_v4());
        assert_eq!(module.source_profile.runtime_capacity_v4(), Some(4));

        let airs = module.airs_without_poseidon::<BabyBearPoseidon2Config>();
        let expected_identity = fixed_capacity_v4_air_identity(&airs);
        let expected_key = engine.keygen(&airs).1.pre_hash;
        let expected_preprocessed = active_count_air
            .preprocessed_trace()
            .expect("active-count preprocessing");
        let mut runtime_traces = Vec::new();
        for occupancy in 1..=VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 {
            let trace = generate_active_child_count_transcript_trace_v4(
                active_count_air,
                &fixed_capacity_v4_active_count_records(occupancy),
            )
            .expect("runtime occupancy trace");
            assert_eq!(Matrix::height(&trace), ACTIVE_CHILD_COUNT_TRACE_HEIGHT_V4);
            assert_eq!(
                active_count_air
                    .preprocessed_trace()
                    .expect("stable active-count preprocessing")
                    .values,
                expected_preprocessed.values
            );
            assert_eq!(fixed_capacity_v4_air_identity(&airs), expected_identity);
            assert_eq!(engine.keygen(&airs).1.pre_hash, expected_key);
            runtime_traces.push(trace.values);
        }
        assert!(runtime_traces.windows(2).all(|pair| pair[0] != pair[1]));

        let mut noncanonical = fixed_capacity_v4_active_count_records(2);
        noncanonical[0].active_child_count = 1;
        assert!(
            generate_active_child_count_transcript_trace_v4(active_count_air, &noncanonical)
                .is_err()
        );
    }

    #[test]
    fn fixed_capacity_v4_source_payload_excludes_distinguished_pesat_coordinate() {
        let (_engine, _child_vk, source_profile, config) = fixed_capacity_v4_constructor_fixture();
        let public_values = (0..config.source_public_values_len)
            .map(|index| F::from_u32(0x66_00 + index))
            .collect::<Vec<_>>();
        let mut preimage = vec![
            F::from_u64(SOURCE_INSTANCE_TAG_V19),
            F::from_usize(public_values.len()),
            F::ZERO,
        ];
        preimage.extend_from_slice(&public_values);
        let (expected_instance_digest, _, _) = poseidon2_hash_slice_with_states(&preimage);
        let source = DirectLogUpSegmentSourceRecordV19 {
            proof_index: 0,
            segment_index: 0,
            prefix_start_tidx: 0,
            alignment_sample: [F::ZERO; D_EF],
            segment_openings_digest: [F::from_u32(0x66_80); DIGEST_SIZE],
            sources: vec![DirectLogUpFreshSourceRecordV19 {
                shard_id: 0,
                source_root: [F::from_u32(0x66_81); DIGEST_SIZE],
                expected_instance_digest,
                public_values,
                boundary_values: Vec::new(),
            }],
        };
        generate_direct_logup_source_manifest_traces_v19(&source_profile, &[source.clone()])
            .expect("source payload coordinates 1.. are admitted");

        let mut with_distinguished_one = source;
        with_distinguished_one.sources[0]
            .public_values
            .insert(0, F::ONE);
        assert_eq!(
            generate_direct_logup_source_manifest_traces_v19(
                &source_profile,
                &[with_distinguished_one],
            )
            .unwrap_err(),
            "fresh source differs from fixed relation profile",
            "PESAT coordinate zero is relation-owned, not source-instance data",
        );
    }

    #[test]
    fn fixed_capacity_v4_column_observations_use_a_runtime_prefix() {
        let (_engine, child_vk, source_profile, config) = fixed_capacity_v4_constructor_fixture();
        let module = DirectLogUpOnlyVerifierModuleV19::<4>::new_fixed_capacity_v4(
            child_vk,
            source_profile,
            config,
        )
        .expect("fixed-capacity production module");
        let air = &module.column_opening_observation_air;
        assert!(air.runtime_capacity_v4);

        for occupancy in 1..=VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 {
            let records = (0..occupancy)
                .map(|proof_index| {
                    let claim_count = air
                        .plans
                        .iter()
                        .filter(|plan| plan.proof_index as usize == proof_index)
                        .filter_map(|plan| plan.claim_index)
                        .max()
                        .map_or(0, |index| index + 1);
                    MappedFunctionalSourceRecordV19 {
                        proof_index: proof_index as u32,
                        segment_index: proof_index as u32,
                        shard_ordinal: 0,
                        source_forest_root: [F::ZERO; DIGEST_SIZE],
                        segment_openings_digest: [F::ZERO; DIGEST_SIZE],
                        source_root: [F::ZERO; DIGEST_SIZE],
                        range_start: 0,
                        range_end: 1,
                        batching_challenge: EF::ONE,
                        program_fingerprint: EF::ZERO,
                        program_mix_challenge: EF::ZERO,
                        active_count_challenge: EF::ONE,
                        logup_opening_point: Vec::new(),
                        one_shot_point: Vec::new(),
                        dynamic_column_claims: (0..claim_count)
                            .map(|index| EF::from_u32(0x67_00 + index as u32))
                            .collect(),
                    }
                })
                .collect::<Vec<_>>();
            let opening_start_tidxs = (0..occupancy)
                .map(|index| 0x68_00 + index as u32)
                .collect::<Vec<_>>();
            let trace =
                generate_column_opening_observation_trace_v19(air, &records, &opening_start_tidxs)
                    .expect("runtime-prefix observation trace");
            let preprocessed =
                BaseAir::<F>::preprocessed_trace(air).expect("fixed column-observation plan");
            check_constraints::<_, BabyBearPoseidon2Config>(
                air,
                "ColumnOpeningObservationAirV19",
                &Some(preprocessed.as_view()),
                &[trace.as_view()],
                &[],
            );
            let width = ColumnOpeningObservationColsV19::<F>::width();
            for (row, plan) in air.plans.iter().enumerate() {
                let cols: &ColumnOpeningObservationColsV19<F> =
                    trace.values[row * width..(row + 1) * width].borrow();
                assert_eq!(
                    cols.active,
                    F::from_bool((plan.proof_index as usize) < occupancy)
                );
            }
        }
    }

    #[test]
    fn fixed_capacity_v4_constructor_rejects_source_relation_dimension_and_child_vk_mutations() {
        let (_engine, child_vk, source_profile, config) = fixed_capacity_v4_constructor_fixture();

        let wrong_relation_entry = DirectLogUpSourceEntryProfileV19::fixed_multi_air(
            [F::from_u32(0x55_01); DIGEST_SIZE],
            config.log_message_len,
            config.log_codeword_len,
            config.source_public_values_len,
        )
        .unwrap();
        let wrong_relation_source = DirectLogUpSourceManifestProfileV19::new_fixed_capacity_v4(
            child_vk.pre_hash,
            source_profile.registry_digest,
            wrong_relation_entry,
        )
        .unwrap();
        assert!(
            DirectLogUpOnlyVerifierModuleV19::<4>::new_fixed_capacity_v4(
                Arc::clone(&child_vk),
                wrong_relation_source,
                config.clone(),
            )
            .is_err()
        );

        let wrong_dimension_entry = DirectLogUpSourceEntryProfileV19::fixed_multi_air(
            config.relation_digest,
            config.log_message_len,
            config.log_codeword_len + 1,
            config.source_public_values_len,
        )
        .unwrap();
        let wrong_dimension_source = DirectLogUpSourceManifestProfileV19::new_fixed_capacity_v4(
            child_vk.pre_hash,
            source_profile.registry_digest,
            wrong_dimension_entry,
        )
        .unwrap();
        assert!(
            DirectLogUpOnlyVerifierModuleV19::<4>::new_fixed_capacity_v4(
                Arc::clone(&child_vk),
                wrong_dimension_source,
                config.clone(),
            )
            .is_err()
        );

        let mut foreign_air_vk = child_vk.inner.per_air[0].clone();
        let dag = Arc::make_mut(&mut foreign_air_vk.symbolic_constraints);
        let first_constraint = *dag
            .constraints
            .constraint_idx
            .first()
            .expect("Fib constraint");
        dag.constraints.constraint_idx.push(first_constraint);
        let original_relation = config.regions[0].relation.description();
        let key = &original_relation.shard_key;
        let foreign_relation = Arc::new(
            DirectAirPesatIndex::from_verifying_key(
                BabyBearPoseidon2Config::default_from_params(child_vk.inner.params.clone())
                    .hasher(),
                child_vk.pre_hash,
                0,
                usize::from(key.log_height),
                &foreign_air_vk,
                None,
                key.public_schema.clone(),
                key.code_class.clone(),
            )
            .expect("same-shape foreign relation"),
        );
        assert_ne!(
            foreign_relation.description().shard_key.relation_digest,
            original_relation.shard_key.relation_digest
        );
        let mut wrong_constraint_relation = config.clone();
        wrong_constraint_relation.regions = Arc::from([FixedMultiAirMappedRegionV19 {
            air_id: 0,
            message_start: 0,
            relation: foreign_relation,
        }]);
        assert!(
            DirectLogUpOnlyVerifierModuleV19::<4>::new_fixed_capacity_v4(
                Arc::clone(&child_vk),
                source_profile.clone(),
                wrong_constraint_relation,
            )
            .is_err()
        );

        let mut wrong_child_vk = (*child_vk).clone();
        wrong_child_vk.pre_hash[0] += F::ONE;
        assert!(
            DirectLogUpOnlyVerifierModuleV19::<4>::new_fixed_capacity_v4(
                Arc::new(wrong_child_vk),
                source_profile,
                config,
            )
            .is_err()
        );
    }

    /// A real retained LogUp-only proof over four active AIRs in the exact
    /// OpenVM system slots: cached Program/Connector followed by
    /// Boundary/Merkle. The traces are repeated uniformly so l_skip=4 is
    /// supported without changing the balanced interaction multiset.
    pub(crate) fn retained_composite_fixture() -> (
        BabyBearPoseidon2CpuEngine<DuplexSponge>,
        DirectLogUpOnlyVerifierModuleV19<4>,
        DirectLogUpOnlyCompositeRecordsV19,
    ) {
        retained_composite_fixture_with_first_source_root(None)
    }

    pub(crate) fn retained_composite_fixture_with_first_source_root(
        first_source_root: Option<[F; DIGEST_SIZE]>,
    ) -> (
        BabyBearPoseidon2CpuEngine<DuplexSponge>,
        DirectLogUpOnlyVerifierModuleV19<4>,
        DirectLogUpOnlyCompositeRecordsV19,
    ) {
        let (engine, module, records, _) = retained_composite_fixture_mode(first_source_root, None);
        (engine, module, records)
    }

    /// Fixed single-source variant used by verifier-WARP History v2. The
    /// returned BabyBear message, Merkle root/data and opening reduction all
    /// originate from the same source trace.
    pub(crate) fn retained_fixed_multi_air_fixture(
        public_values: Vec<F>,
    ) -> (
        BabyBearPoseidon2CpuEngine<DuplexSponge>,
        DirectLogUpOnlyVerifierModuleV19<4>,
        DirectLogUpOnlyCompositeRecordsV19,
        RetainedAppendixDSourceV19,
    ) {
        let (engine, module, records, source) =
            retained_composite_fixture_mode(None, Some(public_values));
        (
            engine,
            module,
            records,
            source.expect("fixed Appendix-D source"),
        )
    }

    fn retained_composite_fixture_mode(
        first_source_root: Option<[F; DIGEST_SIZE]>,
        fixed_public_values: Option<Vec<F>>,
    ) -> (
        BabyBearPoseidon2CpuEngine<DuplexSponge>,
        DirectLogUpOnlyVerifierModuleV19<4>,
        DirectLogUpOnlyCompositeRecordsV19,
        Option<RetainedAppendixDSourceV19>,
    ) {
        let mut params = default_test_params_small();
        params.l_skip = OPENVM_DIRECT_LOGUP_L_SKIP_V19;
        params.logup.pow_bits = 0;
        params.logup.log_max_message_length = 10;
        params.max_constraint_degree = 8;
        let engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
        let (child_pk, child_vk, host) = if let Some(public_values) = &fixed_public_values {
            assert!(!public_values.is_empty());
            let air: AirRef<BabyBearPoseidon2Config> = Arc::new(AppendixDSourceAir {
                num_public_values: public_values.len(),
            });
            let (child_pk, child_vk) = engine.keygen(core::slice::from_ref(&air));
            let height = 1usize << OPENVM_DIRECT_LOGUP_L_SKIP_V19;
            let mut values = vec![F::ZERO; height * 2];
            values[0] = F::ONE;
            for row in 0..height {
                values[row * 2 + 1] = F::from_usize(17 + row);
            }
            let trace = RowMajorMatrix::new(values, 2);
            let host = ProvingContext::new(vec![(
                0,
                AirProvingContext::simple(
                    ColMajorMatrix::from_row_major(&trace),
                    public_values.clone(),
                ),
            )]);
            (child_pk, child_vk, host)
        } else {
            let fixture = MixtureFixture::<BabyBearPoseidon2Config>::new(vec![
                MixtureFixtureEnum::CachedFixture11(CachedFixture11::new(engine.config().clone())),
                MixtureFixtureEnum::InteractionsFixture11(InteractionsFixture11),
            ]);
            let (child_pk, child_vk) = fixture.keygen(&engine);
            let host = fixture.generate_proving_ctx();
            (child_pk, child_vk, host)
        };
        if fixed_public_values.is_some() {
            assert_eq!(child_vk.inner.per_air.len(), 1);
        } else {
            assert_eq!(child_vk.inner.per_air.len(), 4);
        }

        let repeat_factor = if fixed_public_values.is_some() { 1 } else { 4 };
        let expanded = ProvingContext::<CpuColMajorBackend<BabyBearPoseidon2Config>>::new(
            host.per_trace
                .into_iter()
                .map(|(air_id, trace)| {
                    let common_main = repeat_col_major(&trace.common_main, repeat_factor);
                    let cached_mains = trace
                        .cached_mains
                        .into_iter()
                        .map(|cached| {
                            let cached_trace = repeat_col_major(&cached.trace, repeat_factor);
                            let params = engine.config().params();
                            let (commitment, data) = stacked_commit(
                                engine.config().hasher(),
                                params.l_skip,
                                params.n_stack,
                                params.log_blowup,
                                params.log_commit_rows_per_query,
                                &[&cached_trace],
                            )
                            .unwrap();
                            CommittedTraceData {
                                commitment,
                                trace: cached_trace,
                                data: Arc::new(data),
                            }
                        })
                        .collect();
                    (
                        air_id,
                        AirProvingContext {
                            cached_mains,
                            common_main,
                            public_values: trace.public_values,
                        },
                    )
                })
                .collect(),
        );

        let mut relations = Vec::new();
        let mut entries = Vec::new();
        for (shard_id, (air_id, trace)) in expanded.per_trace.iter().enumerate() {
            assert_eq!(shard_id, *air_id);
            let air_vk = &child_vk.inner.per_air[*air_id];
            let log_height = trace.common_main.height().ilog2() as usize;
            let raw_width = air_vk.params.width.common_main
                + air_vk.params.width.cached_mains.iter().sum::<usize>();
            let raw_len = (1usize << log_height) * raw_width;
            let log_message_len = raw_len.next_power_of_two().ilog2() as u8;
            let relation = Arc::new(
                DirectAirPesatIndex::from_verifying_key(
                    engine.config().hasher(),
                    child_vk.pre_hash,
                    *air_id,
                    log_height,
                    air_vk,
                    None,
                    DirectAirPublicSchema {
                        public_values_len: air_vk.params.num_public_values as u32,
                        boundary_values_len: 0,
                        schema_digest: [F::from_usize(700 + shard_id); DIGEST_SIZE],
                    },
                    DirectAirCodeClass {
                        log_message_len,
                        log_blowup: child_vk.inner.params.log_blowup as u8,
                        log_codeword_len: log_message_len + child_vk.inner.params.log_blowup as u8,
                        initial_folding_factor: 0,
                        rows_per_query: 2,
                    },
                )
                .unwrap(),
            );
            entries.push(
                DirectLogUpSourceEntryProfileV19::from_relation(
                    shard_id as u32,
                    relation.description(),
                )
                .unwrap(),
            );
            relations.push(relation);
        }
        let registry_digest = [F::from_u32(733); DIGEST_SIZE];
        let mut retained_appendix_d_source = None;
        let (source_profile, module, entries) = if fixed_public_values.is_some() {
            let mut message_start = 0u32;
            let mut regions = Vec::with_capacity(relations.len());
            let mut trace_heights = vec![0u32; child_vk.inner.per_air.len()];
            for (relation, (air_id, trace)) in relations.iter().zip(&expanded.per_trace) {
                let height = u32::try_from(trace.common_main.height()).unwrap();
                trace_heights[*air_id] = height;
                regions.push(FixedMultiAirMappedRegionV19 {
                    air_id: *air_id as u32,
                    message_start,
                    relation: Arc::clone(relation),
                });
                let width = relation.description().shard_key.trace_layout.total_width();
                message_start = message_start.checked_add(height * width as u32).unwrap();
            }
            let log_message_len = message_start.next_power_of_two().ilog2() as u8;
            let log_codeword_len = log_message_len + child_vk.inner.params.log_blowup as u8;
            assert!(message_start <= (1u32 << log_message_len));
            let code_class = DirectAirCodeClass {
                log_message_len,
                log_blowup: child_vk.inner.params.log_blowup as u8,
                log_codeword_len,
                initial_folding_factor: 1,
                rows_per_query: 1,
            };
            let fixed_relation = Arc::new(
                FixedMultiAirPesatIndex::from_direct_air_regions(
                    engine.config().hasher(),
                    child_vk.pre_hash,
                    relations
                        .iter()
                        .map(|relation| relation.as_ref().clone())
                        .collect(),
                    code_class,
                )
                .unwrap(),
            );
            let relation_digest = fixed_relation.description().relation_digest;
            let local_witnesses = relations
                .iter()
                .zip(&expanded.per_trace)
                .map(|(relation, (_, trace))| {
                    let mut cells = Vec::with_capacity(relation.raw_witness_len());
                    for cached in &trace.cached_mains {
                        cells.extend_from_slice(&cached.trace.values);
                    }
                    cells.extend_from_slice(&trace.common_main.values);
                    DirectAirPesatWitness { cells }
                })
                .collect::<Vec<_>>();
            let instance = FixedMultiAirPesatInstance {
                regions: expanded
                    .per_trace
                    .iter()
                    .map(|(_, trace)| DirectAirPesatInstance {
                        public_values: trace.public_values.clone(),
                        boundary_values: Vec::new(),
                    })
                    .collect(),
            };
            let witness = fixed_relation.stack_witnesses(&local_witnesses).unwrap();
            assert!(fixed_relation
                .is_satisfied_reference(&instance, &witness)
                .unwrap());
            let message = fixed_relation.padded_witness::<EF>(&witness).unwrap();
            let base_message = message
                .iter()
                .copied()
                .map(|value| value.as_base().expect("base fixed source witness"))
                .collect::<Vec<_>>();
            let explicit = fixed_relation
                .explicit_assignment(&instance)
                .unwrap()
                .into_iter()
                .map(EF::from)
                .collect::<Vec<_>>();
            let code = WhirInitialRsWarpCode::new(
                engine.config().hasher().clone(),
                log_message_len as usize,
                child_vk.inner.params.log_blowup,
                code_class.initial_folding_factor as usize,
                code_class.rows_per_query as usize,
            );
            let (root, base_prover_data, base_codeword): (
                Digest,
                WhirRsCodeProverData<F, Digest>,
                Vec<F>,
            ) = code.commit_message(&base_message).unwrap();
            let public_values = expanded.per_trace[0].1.public_values.clone();
            let mut instance_preimage = vec![
                F::from_u64(SOURCE_INSTANCE_TAG_V19),
                F::from_usize(public_values.len()),
                F::ZERO,
            ];
            instance_preimage.extend_from_slice(&public_values);
            let (expected_instance_digest, _, _) =
                poseidon2_hash_slice_with_states(&instance_preimage);
            retained_appendix_d_source = Some(RetainedAppendixDSourceV19 {
                relation: Arc::clone(&fixed_relation),
                message,
                explicit,
                root,
                base_prover_data,
                base_codeword,
                expected_instance_digest,
            });
            let fixed_entry = DirectLogUpSourceEntryProfileV19::fixed_multi_air(
                relation_digest,
                log_message_len,
                log_codeword_len,
                u32::try_from(expanded.per_trace[0].1.public_values.len()).unwrap(),
            )
            .unwrap();
            let plan = vec![vec![0]];
            let source_profile = DirectLogUpSourceManifestProfileV19::new(
                child_vk.pre_hash,
                registry_digest,
                vec![fixed_entry.clone()],
                &plan,
            )
            .unwrap();
            let first_relation = &relations[0];
            let first_key = &first_relation.description().shard_key;
            let first_height = 1u32 << first_key.log_height;
            let is_valid_message_block_start = first_height
                * first_key
                    .trace_layout
                    .cached_main_widths
                    .iter()
                    .copied()
                    .sum::<u32>();
            let config = FixedMultiAirMappedFunctionalConfigV19 {
                relation_digest,
                log_message_len,
                log_codeword_len,
                regions: regions.into(),
                vm_pvs_air_id: 0,
                is_valid_common_main_column: 0,
                is_valid_message_block_start,
                vm_pvs_log_height: first_key.log_height,
                active_child_counts: Arc::from([1]),
                profile_trace_heights: trace_heights.into(),
                authenticate_cached_setup_claims: false,
            };
            let module = DirectLogUpOnlyVerifierModuleV19::<4>::new_fixed_multi_air(
                Arc::new(child_vk),
                source_profile.clone(),
                config,
                0,
            )
            .unwrap();
            (source_profile, module, vec![fixed_entry])
        } else {
            let plan = vec![vec![0, 1, 2, 3]];
            let source_profile = DirectLogUpSourceManifestProfileV19::new(
                child_vk.pre_hash,
                registry_digest,
                entries.clone(),
                &plan,
            )
            .unwrap();
            let mapped_entries = relations
                .iter()
                .enumerate()
                .map(|(shard_id, relation)| DirectLogUpMappedRegistryEntryV19 {
                    shard_id: shard_id as u32,
                    relation: relation.clone(),
                })
                .collect();
            let module = DirectLogUpOnlyVerifierModuleV19::<4>::new(
                Arc::new(child_vk),
                source_profile.clone(),
                mapped_entries,
                &plan,
            )
            .unwrap();
            (source_profile, module, entries)
        };

        let mut source_records = Vec::new();
        for (entry, (_, trace)) in entries.iter().zip(&expanded.per_trace) {
            let public_values = trace.public_values.clone();
            let boundary_values = Vec::new();
            let mut preimage = vec![
                F::from_u64(SOURCE_INSTANCE_TAG_V19),
                F::from_usize(public_values.len()),
                F::from_usize(boundary_values.len()),
            ];
            preimage.extend_from_slice(&public_values);
            preimage.extend_from_slice(&boundary_values);
            let (derived_instance_digest, _, _) = poseidon2_hash_slice_with_states(&preimage);
            let expected_instance_digest = derived_instance_digest;
            if let Some(source) = &retained_appendix_d_source {
                assert_eq!(source.expected_instance_digest, expected_instance_digest);
            }
            source_records.push(DirectLogUpFreshSourceRecordV19 {
                shard_id: entry.shard_id,
                source_root: if entry.shard_id == 0 {
                    retained_appendix_d_source.as_ref().map_or_else(
                        || {
                            first_source_root
                                .unwrap_or([F::from_u32(800 + entry.shard_id); DIGEST_SIZE])
                        },
                        |source| source.root,
                    )
                } else {
                    [F::from_u32(800 + entry.shard_id); DIGEST_SIZE]
                },
                expected_instance_digest,
                public_values,
                boundary_values,
            });
        }
        let segment_openings_digest = [F::from_u32(850); DIGEST_SIZE];
        let mut sources = vec![DirectLogUpSegmentSourceRecordV19 {
            proof_index: 0,
            segment_index: 0,
            prefix_start_tidx: 0,
            alignment_sample: [F::ZERO; D_EF],
            segment_openings_digest,
            sources: source_records,
        }];
        let source_traces =
            generate_direct_logup_source_manifest_traces_v19(&source_profile, &sources).unwrap();
        let source_forest_root = source_traces.forest_roots[0];
        let mut prefix = default_duplex_sponge_recorder();
        for value in [
            F::from_u64(SEGMENT_PREFIX_TAG_V19),
            F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            F::ZERO,
        ] {
            FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(&mut prefix, value);
        }
        for digest in [
            module.child_vk.pre_hash,
            registry_digest,
            source_forest_root,
        ] {
            FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_commit(&mut prefix, digest);
        }
        FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(
            &mut prefix,
            F::from_usize(sources[0].sources.len()),
        );
        for source in &sources[0].sources {
            FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(
                &mut prefix,
                F::from_u32(source.shard_id),
            );
            FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_commit(
                &mut prefix,
                source.expected_instance_digest,
            );
            FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_commit(
                &mut prefix,
                source.source_root,
            );
        }
        FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(
            &mut prefix,
            F::from_u64(LOGUP_START_BOUNDARY_TAG_V19),
        );
        let alignment = FiatShamirTranscript::<BabyBearPoseidon2Config>::sample_ext(&mut prefix);
        sources[0]
            .alignment_sample
            .copy_from_slice(alignment.as_basis_coefficients_slice());
        let mut transcript = build_source_prefix_v19(
            module.child_vk.pre_hash,
            registry_digest,
            source_forest_root,
            &sources[0],
        )
        .unwrap();
        let logup_prefix = transcript.clone();

        let device = engine.device();
        let device_pk = <_ as DeviceDataTransporter<
            BabyBearPoseidon2Config,
            CpuBackend<BabyBearPoseidon2Config>,
        >>::transport_pk_to_device(device, &child_pk);
        let device_ctx = <_ as DeviceDataTransporter<
            BabyBearPoseidon2Config,
            CpuBackend<BabyBearPoseidon2Config>,
        >>::transport_proving_ctx_to_device(device, &expanded)
        .into_sorted();
        let trace_air_ids = device_ctx
            .per_trace
            .iter()
            .map(|(air_id, _)| *air_id)
            .collect::<Vec<_>>();
        let n_per_trace = device_ctx
            .common_main_traces()
            .map(|(_, trace)| {
                MatrixDimensions::height(trace).ilog2() as isize
                    - OPENVM_DIRECT_LOGUP_L_SKIP_V19 as isize
            })
            .collect::<Vec<_>>();
        let need_rot_per_trace = trace_air_ids
            .iter()
            .map(|&air_id| module.child_vk.inner.per_air[air_id].params.need_rot)
            .collect::<Vec<_>>();
        let trace_public_values = device_ctx
            .per_trace
            .iter()
            .map(|(_, trace)| trace.public_values.clone())
            .collect::<Vec<_>>();
        let (gkr_proof, batch_constraint_proof, prover_endpoint) =
            prove_logup_only_recorded(&mut transcript, &device_pk, &device_ctx).unwrap();
        // `prove_logup_only_recorded` continues through the prover's final
        // column-opening observations. Rebuild from the verifier-owned prefix
        // and absorb those openings explicitly: the production composition
        // authenticates them through ColumnClaimsBus and binds the identical
        // vector into Fiat--Shamir before deriving per-shard challenges.
        transcript = logup_prefix;
        let omega = F::two_adic_generator(OPENVM_DIRECT_LOGUP_L_SKIP_V19);
        let omega_skip_pows = omega
            .powers()
            .take(1 << OPENVM_DIRECT_LOGUP_L_SKIP_V19)
            .collect();
        let endpoint = verify_logup_only_prefix(
            &mut transcript,
            &module.child_vk.inner,
            &gkr_proof,
            (&batch_constraint_proof).into(),
            &trace_air_ids,
            &n_per_trace,
            &omega_skip_pows,
            None,
        )
        .unwrap();
        assert_eq!(endpoint.final_claim, prover_endpoint.final_claim);
        assert_eq!(endpoint.rs, prover_endpoint.rs);
        observe_batch_constraint_openings(
            &mut transcript,
            &module.child_vk.inner,
            &trace_air_ids,
            &batch_constraint_proof.column_openings,
        )
        .unwrap();
        let proof_material = RetainedLogUpOnlyProof {
            common_main_commit: [F::ZERO; DIGEST_SIZE],
            trace_vdata: Vec::new(),
            public_values: Vec::new(),
            gkr_proof,
            batch_constraint_proof,
        };
        let retained_proofs = vec![DirectLogUpOnlyRetainedProofRecordV19 {
            proof_index: 0,
            trace_air_ids,
            n_per_trace,
            need_rot_per_trace,
            trace_public_values,
            proof_material,
        }];

        let source_plans = mapped_source_plans_v19(&module.mapped_functional_air.profile);
        let program_fingerprint = EF::from_u32(877);
        let mut claim_derivations = Vec::new();
        for plan in &source_plans {
            let start_tidx = transcript.len() as u32;
            for value in [
                F::from_u64(DIRECT_LOGUP_CLAIM_TAG_V19),
                F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                F::ZERO,
                F::from_u32(plan.shard_id),
            ] {
                FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(&mut transcript, value);
            }
            FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_commit(
                &mut transcript,
                plan.relation_digest,
            );
            let batching_challenge =
                FiatShamirTranscript::<BabyBearPoseidon2Config>::sample_ext(&mut transcript);
            let (fingerprint, mix) = if plan.is_program {
                FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(
                    &mut transcript,
                    F::from_u64(PROGRAM_FINGERPRINT_TARGET_TAG_V19),
                );
                FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_ext(
                    &mut transcript,
                    program_fingerprint,
                );
                (
                    program_fingerprint,
                    FiatShamirTranscript::<BabyBearPoseidon2Config>::sample_ext(&mut transcript),
                )
            } else {
                (EF::ZERO, EF::ZERO)
            };
            claim_derivations.push(LogUpClaimDerivationRecordV19 {
                proof_index: 0,
                segment_index: 0,
                shard_ordinal: plan.shard_ordinal,
                shard_count: plan.shard_count,
                shard_id: plan.shard_id,
                relation_digest: plan.relation_digest,
                is_program: plan.is_program,
                batching_challenge,
                program_fingerprint: fingerprint,
                program_mix_challenge: mix,
                start_tidx,
            });
        }

        let active_count_challenge =
            if let Some(active_count_air) = &module.active_count_transcript_air {
                let record = observe_active_child_count_transcript_v19(
                    &active_count_air.profile,
                    0,
                    &mut transcript,
                )
                .unwrap();
                if record.sampled == EF::ZERO {
                    EF::ONE
                } else {
                    record.sampled
                }
            } else {
                EF::ZERO
            };

        let proof = &retained_proofs[0].proof_material;
        let mut mapped_sources = Vec::new();
        let mut range_start = 0u32;
        for (source_index, plan) in source_plans.iter().enumerate() {
            let plans = &module.mapped_functional_air.profile.terms[module
                .mapped_functional_air
                .profile
                .terms
                .iter()
                .position(|candidate| core::ptr::eq(candidate, *plan))
                .unwrap()..][..plan.term_count as usize];
            let openings_per_column = 1 + usize::from(
                module.child_vk.inner.per_air[plan.air_id as usize]
                    .params
                    .need_rot,
            );
            let dynamic_column_claims = plans
                .iter()
                .filter(|term| !term.is_program_term && !term.is_active_count_term)
                .map(|term| {
                    proof.batch_constraint_proof.column_openings[term.sort_idx as usize]
                        [term.part_idx as usize]
                        [term.col_idx as usize * openings_per_column + usize::from(term.is_rot)]
                })
                .collect();
            let range_end = range_start + (1u32 << entries[source_index].log_codeword_len);
            let derivation = &claim_derivations[source_index];
            mapped_sources.push(MappedFunctionalSourceRecordV19 {
                proof_index: 0,
                segment_index: 0,
                shard_ordinal: plan.shard_ordinal,
                source_forest_root,
                segment_openings_digest,
                source_root: sources[0].sources[source_index].source_root,
                range_start,
                range_end,
                batching_challenge: derivation.batching_challenge,
                program_fingerprint: derivation.program_fingerprint,
                program_mix_challenge: derivation.program_mix_challenge,
                active_count_challenge,
                logup_opening_point: endpoint.rs.clone(),
                one_shot_point: vec![EF::ZERO; plan.log_message_len as usize],
                dynamic_column_claims,
            });
            range_start = range_end;
        }

        let mut claim_transcripts = Vec::new();
        let mut reductions = Vec::new();
        for source_index in 0..source_plans.len() {
            let plan = source_plans[source_index];
            let start = module
                .mapped_functional_air
                .profile
                .terms
                .iter()
                .position(|candidate| core::ptr::eq(candidate, plan))
                .unwrap();
            let plans = &module.mapped_functional_air.profile.terms
                [start..start + plan.term_count as usize];
            let provisional =
                derive_mapped_claim_v19(plans, &mapped_sources[source_index], None).unwrap();
            let message = if let Some(source) = &retained_appendix_d_source {
                assert_eq!(source.message.len(), provisional.claim.len());
                let evaluated = provisional
                    .claim
                    .weight
                    .materialize()
                    .iter()
                    .zip(&source.message)
                    .fold(EF::ZERO, |sum, (&weight, &value)| sum + weight * value);
                assert_eq!(evaluated, provisional.claim.target);
                source.message.clone()
            } else {
                let mut message = vec![EF::ZERO; provisional.claim.len()];
                if provisional.claim.target != EF::ZERO {
                    let weights = provisional.claim.weight.materialize();
                    let (index, &weight) = weights
                        .iter()
                        .enumerate()
                        .find(|(_, weight)| **weight != EF::ZERO)
                        .unwrap();
                    message[index] = provisional.claim.target * weight.inverse();
                }
                message
            };
            let start_tidx = transcript.len() as u32;
            let owned = core::mem::replace(&mut transcript, default_duplex_sponge_recorder());
            let mut challenger = NativeWarpChallenger::<BabyBearPoseidon2Config, _>::new(owned);
            let (reduction, opening) = prove_direct_message_opening_reduction(
                &provisional.claim,
                &message,
                &mut challenger,
            )
            .unwrap();
            transcript = challenger.into_inner();
            mapped_sources[source_index].one_shot_point = opening.point.clone();
            let derived =
                derive_mapped_claim_v19(plans, &mapped_sources[source_index], None).unwrap();
            let claim_record = OneShotClaimTranscriptRecordV19::from_claim(
                0,
                0,
                plan.shard_ordinal,
                start_tidx,
                &derived.claim,
            )
            .unwrap();
            let round_start = claim_record.round_start_tidx();
            claim_transcripts.push(claim_record);
            reductions.push(OneShotReductionRecordV19 {
                proof_index: 0,
                segment_index: 0,
                shard_ordinal: plan.shard_ordinal,
                is_program: plan.is_program,
                source_root: mapped_sources[source_index].source_root,
                range_start: mapped_sources[source_index].range_start,
                range_end: mapped_sources[source_index].range_end,
                functional_digest: plan.relation_digest,
                ordinary_target: derived.ordinary_target,
                ordinary_weight_at_point: derived.ordinary_weight_at_point,
                program_fingerprint: mapped_sources[source_index].program_fingerprint,
                fingerprint_weight_at_point: derived.fingerprint_weight_at_point,
                mix_challenge: mapped_sources[source_index].program_mix_challenge,
                start_tidx: round_start,
                round_polynomials: reduction
                    .sumcheck
                    .round_polys
                    .into_iter()
                    .map(|polynomial| polynomial.try_into().unwrap())
                    .collect(),
                point: opening.point,
                message_value: opening.value,
            });
        }

        let program_digest = source_plans
            .iter()
            .find(|plan| plan.is_program)
            .map(|program_plan| {
                compute_program_fingerprint_digest_v19(
                    program_plan.relation_digest,
                    program_plan.log_height,
                    program_plan.cached_width,
                    program_fingerprint,
                )
            })
            .unwrap_or([F::ZERO; DIGEST_SIZE]);
        let mut boundary_shards = Vec::new();
        for (index, plan) in source_plans.iter().enumerate() {
            boundary_shards.push(LogUpSwirlBoundaryShardRecordV19 {
                proof_index: 0,
                segment_index: 0,
                shard_ordinal: plan.shard_ordinal,
                shard_count: plan.shard_count,
                air_id: plan.air_id,
                is_program: plan.is_program,
                relation_digest: plan.relation_digest,
                log_height: plan.log_height,
                cached_width: plan.cached_width,
                log_message_len: plan.log_message_len,
                app_vk_digest: module.child_vk.pre_hash,
                registry_digest,
                source_forest_root,
                segment_openings_digest,
                source_root: mapped_sources[index].source_root,
                range_start: mapped_sources[index].range_start,
                range_end: mapped_sources[index].range_end,
                opening_point: reductions[index].point.clone(),
                opening_value: reductions[index].message_value,
                verifier_endpoint: endpoint.final_claim,
                segment_sum_before: EF::ZERO,
                segment_sum_after: EF::ZERO,
                initial_pc: F::from_u32(3),
                final_pc: F::from_u32(7),
                exit_code: F::ZERO,
                is_terminate: true,
                initial_memory_root: [F::from_u32(901); DIGEST_SIZE],
                final_memory_root: [F::from_u32(902); DIGEST_SIZE],
                program_fingerprint: if plan.is_program {
                    program_fingerprint
                } else {
                    EF::ZERO
                },
                program_fingerprint_digest: program_digest,
            });
        }
        (
            engine,
            module,
            DirectLogUpOnlyCompositeRecordsV19 {
                retained_proofs,
                sources,
                claim_derivations,
                mapped_sources,
                claim_transcripts,
                reductions,
                boundary_shards,
            },
            retained_appendix_d_source,
        )
    }

    fn assert_context_equivalent(
        left: &AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>,
        right: &AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>,
    ) {
        assert_eq!(
            Matrix::width(&left.common_main),
            Matrix::width(&right.common_main)
        );
        assert_eq!(left.common_main.values, right.common_main.values);
        assert_eq!(left.public_values, right.public_values);
        assert_eq!(left.cached_mains.len(), right.cached_mains.len());
        for (left, right) in left.cached_mains.iter().zip(&right.cached_mains) {
            assert_eq!(Matrix::width(&left.trace), Matrix::width(&right.trace));
            assert_eq!(left.trace.values, right.trace.values);
        }
    }

    fn symbolic_interactions(
        air: &dyn AnyAir<BabyBearPoseidon2Config>,
    ) -> Vec<SymbolicInteraction<F>> {
        let preprocessed = BaseAir::<F>::preprocessed_trace(air).map(|trace| Matrix::width(&trace));
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

    #[test]
    fn mapped_functional_symbolic_degree_stays_within_protocol_limit() {
        let (_engine, module, _source_profile, _records) = composite_fixture();
        let air = &module.mapped_functional_air;
        let symbolic = get_symbolic_builder(
            air,
            &TraceWidth {
                preprocessed: BaseAir::<F>::preprocessed_trace(air)
                    .map(|trace| Matrix::width(&trace)),
                cached_mains: air.cached_main_widths(),
                common_main: air.common_main_width(),
            },
        )
        .constraints();
        assert_eq!(symbolic.max_constraint_degree(), 8);
    }

    fn check_mapped_functional_constraints(
        air: &MappedFunctionalTermAirV19,
        trace: &RowMajorMatrix<F>,
    ) {
        let preprocessed = BaseAir::<F>::preprocessed_trace(air).expect("fixed mapped profile");
        check_constraints::<_, BabyBearPoseidon2Config>(
            air,
            "MappedFunctionalTermAirV19",
            &Some(preprocessed.as_view()),
            &[trace.as_view()],
            &[],
        );
    }

    fn assert_mapped_functional_mutation_rejected(
        air: &MappedFunctionalTermAirV19,
        honest: &RowMajorMatrix<F>,
        row: usize,
        mutate: impl FnOnce(&mut MappedFunctionalTermColsV19<F>),
    ) {
        let mut tampered = honest.clone();
        let width = Matrix::width(&tampered);
        let cols: &mut MappedFunctionalTermColsV19<F> =
            tampered.values[row * width..(row + 1) * width].borrow_mut();
        mutate(cols);
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_mapped_functional_constraints(air, &tampered);
        }));
        assert!(rejected.is_err(), "mapped-functional mutation must reject");
    }

    fn base_claim_observation(value: F) -> [F; D_EF] {
        core::array::from_fn(|limb| if limb == 0 { value } else { F::ZERO })
    }

    /// Independently materialize the transcript events encoded by the fused
    /// mapped-functional trace. This deliberately follows the backend's
    /// descriptor layout rather than calling the production AIR expression
    /// builder, and is compared against the retained unfused relay oracle.
    fn fused_claim_events_from_trace(
        profile: &DirectLogUpMappedFunctionalProfileV19,
        trace: &RowMajorMatrix<F>,
    ) -> Vec<(F, F, [F; D_EF])> {
        let mut events = Vec::new();
        let mut row = 0usize;
        while row < profile.terms.len() {
            let first = &profile.terms[row];
            let term_count = first.term_count as usize;
            let rows = &profile.terms[row..row + term_count];
            let first_values = trace.row_slice(row).expect("fused first mapped row");
            let first_cols: &MappedFunctionalTermColsV19<F> = (&*first_values).borrow();
            let mut observations = vec![None; first.observation_count as usize];
            let mut set = |index: usize, value: [F; D_EF]| {
                assert!(
                    observations[index].replace(value).is_none(),
                    "duplicate observation"
                );
            };

            let mut header = 0usize;
            for &byte in DIRECT_OPENING_REDUCTION_TAG_V19 {
                set(header, base_claim_observation(F::from_u8(byte)));
                header += 1;
            }
            for byte in 1u64.to_le_bytes() {
                set(header, base_claim_observation(F::from_u8(byte)));
                header += 1;
            }
            for byte in (first.log_message_len as u64)
                .to_le_bytes()
                .into_iter()
                .chain((first.term_count as u64).to_le_bytes())
            {
                set(header, base_claim_observation(F::from_u8(byte)));
                header += 1;
            }

            for (offset, plan) in rows.iter().enumerate() {
                let values = trace
                    .row_slice(row + offset)
                    .expect("fused mapped term row");
                let cols: &MappedFunctionalTermColsV19<F> = (&*values).borrow();
                let q = 1usize << plan.l_skip;
                let folded_len = usize::from(plan.log_height).saturating_sub(plan.l_skip as usize);
                let mut index = plan.observation_start as usize;
                for value in [
                    plan.block_start as u64,
                    plan.log_height as u64,
                    plan.l_skip as u64,
                    u64::from(plan.is_rot),
                    q as u64,
                ] {
                    for byte in value.to_le_bytes() {
                        set(index, base_claim_observation(F::from_u8(byte)));
                        index += 1;
                    }
                }
                for value in cols.barycentric.iter().take(q) {
                    set(index, *value);
                    index += 1;
                }
                for byte in (folded_len as u64).to_le_bytes() {
                    set(index, base_claim_observation(F::from_u8(byte)));
                    index += 1;
                }
                for folded_index in 0..folded_len {
                    let value: [F; D_EF] = if plan.is_program_term {
                        plan.program_row_point[folded_index]
                            .as_basis_coefficients_slice()
                            .try_into()
                            .expect("EF4 Program point")
                    } else {
                        cols.rs[folded_index + 1]
                    };
                    set(index, value);
                    index += 1;
                }
                set(index, cols.term_scale);
            }

            let mix = EF::from_basis_coefficients_slice(&first_cols.program_mix_challenge)
                .expect("EF4 mix challenge");
            let fingerprint = EF::from_basis_coefficients_slice(&first_cols.program_fingerprint)
                .expect("EF4 fingerprint");
            let last_values = trace
                .row_slice(row + term_count - 1)
                .expect("fused final mapped row");
            let last_cols: &MappedFunctionalTermColsV19<F> = (&*last_values).borrow();
            let ordinary_target =
                EF::from_basis_coefficients_slice(&last_cols.ordinary_target_after)
                    .expect("EF4 final ordinary target");
            let combined_target = ordinary_target + mix * fingerprint;
            set(
                first.observation_count as usize - 1,
                combined_target
                    .as_basis_coefficients_slice()
                    .try_into()
                    .expect("EF4 combined target"),
            );
            drop(set);

            events.extend(observations.into_iter().enumerate().map(|(index, value)| {
                (
                    F::from_u32(first.proof_index),
                    first_cols.claim_start_tidx + F::from_usize(index * D_EF),
                    value.expect("complete fused observation schedule"),
                )
            }));
            row += term_count;
        }
        events
    }

    fn reference_claim_events(
        records: &[OneShotClaimTranscriptRecordV19],
    ) -> Vec<(F, F, [F; D_EF])> {
        let trace = generate_one_shot_claim_transcript_trace_v19(records)
            .expect("unfused claim transcript oracle");
        let rows = records
            .iter()
            .map(|record| record.observations.len())
            .sum::<usize>();
        (0..rows)
            .map(|row| {
                let values = trace.row_slice(row).expect("unfused oracle row");
                let cols: &OneShotClaimTranscriptColsV19<F> = (&*values).borrow();
                (cols.proof_index, cols.tidx, cols.value)
            })
            .collect()
    }

    #[test]
    fn fused_claim_transcript_is_bit_identical_to_unfused_v19_oracle() {
        let (_engine, module, records, _source) = retained_fixed_multi_air_fixture(vec![F::ONE]);
        let trace = generate_mapped_functional_trace_v19(
            &module.mapped_functional_air.profile,
            &records.mapped_sources,
            &records.claim_transcripts,
        )
        .expect("fused mapped-functional trace");
        assert_eq!(
            fused_claim_events_from_trace(&module.mapped_functional_air.profile, &trace),
            reference_claim_events(&records.claim_transcripts),
        );
        for (claim, reduction) in records.claim_transcripts.iter().zip(&records.reductions) {
            assert_eq!(claim.round_start_tidx(), reduction.start_tidx);
        }
    }

    #[test]
    fn fused_claim_transcript_record_and_cursor_mutations_reject() {
        let (_engine, module, records, _source) = retained_fixed_multi_air_fixture(vec![F::ONE]);
        let first_plan = module
            .mapped_functional_air
            .profile
            .terms
            .first()
            .expect("first mapped term");
        let q = 1usize << first_plan.l_skip;
        let folded_len =
            usize::from(first_plan.log_height).saturating_sub(first_plan.l_skip as usize);
        let term = first_plan.observation_start as usize;
        let mut mutation_indices = vec![
            0,                                                   // domain tag
            DIRECT_OPENING_REDUCTION_TAG_V19.len(),              // variant
            DIRECT_OPENING_REDUCTION_TAG_V19.len() + 8,          // message length
            DIRECT_OPENING_REDUCTION_TAG_V19.len() + 16,         // term count
            term,                                                // block start
            term + 8,                                            // log height
            term + 16,                                           // l_skip
            term + 24,                                           // rotation
            term + 32,                                           // barycentric length
            term + 40,                                           // first barycentric value
            term + 40 + q,                                       // folded-point length
            term + 48 + q + folded_len,                          // scale
            records.claim_transcripts[0].observations.len() - 1, // target
        ];
        if folded_len != 0 {
            mutation_indices.push(term + 48 + q); // first folded-point value
        }
        mutation_indices.sort_unstable();
        mutation_indices.dedup();
        for index in mutation_indices {
            let mut value_mutation = records.clone();
            value_mutation.claim_transcripts[0].observations[index] += EF::ONE;
            assert!(matches!(
                module.generate_proving_contexts::<BabyBearPoseidon2Config>(&value_mutation),
                Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                    "mapped claim transcript mismatch"
                ))
            ));
        }

        let mut cursor_mutation = records;
        cursor_mutation.claim_transcripts[0].start_tidx += 1;
        assert!(matches!(
            module.generate_proving_contexts::<BabyBearPoseidon2Config>(&cursor_mutation),
            Err(DirectLogUpOnlyCompositeContextErrorV19::Transcript(
                "one-shot claim/reduction cursor"
            ))
        ));
    }

    #[test]
    fn mapped_functional_active_count_challenge_count_weight_and_point_mutations_reject() {
        let (_engine, module, records, _source) = retained_fixed_multi_air_fixture(vec![F::ONE]);
        let air = &module.mapped_functional_air;
        let honest = generate_mapped_functional_trace_v19(
            &air.profile,
            &records.mapped_sources,
            &records.claim_transcripts,
        )
        .expect("honest mapped-functional trace");
        check_mapped_functional_constraints(air, &honest);
        let active_count_row = air
            .profile
            .terms
            .iter()
            .position(|term| term.is_active_count_term)
            .expect("active-count term");

        assert_mapped_functional_mutation_rejected(air, &honest, active_count_row, |cols| {
            cols.active_count_challenge[0] += F::ONE;
        });
        assert_mapped_functional_mutation_rejected(air, &honest, active_count_row, |cols| {
            cols.claim[0] += F::ONE;
        });
        assert_mapped_functional_mutation_rejected(air, &honest, active_count_row, |cols| {
            cols.term_weight_at_point[0] += F::ONE;
        });
        assert_mapped_functional_mutation_rejected(air, &honest, active_count_row, |cols| {
            cols.one_shot_point[0][0] += F::ONE;
        });
    }

    #[test]
    fn active_count_composite_honest_trace_keygens_at_degree_eight() {
        let (engine, module, records, _source) = retained_fixed_multi_air_fixture(vec![F::ONE]);
        let airs = module.airs::<BabyBearPoseidon2Config>();
        engine.keygen(&airs);
        let contexts = module
            .generate_proving_contexts::<BabyBearPoseidon2Config>(&records)
            .expect("honest active-count composition");
        assert_eq!(airs.len(), contexts.len());
        for (air_id, (air, context)) in airs.iter().zip(&contexts).enumerate() {
            assert_eq!(
                air.num_public_values(),
                0,
                "fixed multi-AIR direct verifier leaked PIs at AIR {air_id}"
            );
            assert!(
                context.public_values.is_empty(),
                "fixed multi-AIR direct verifier supplied PIs at AIR {air_id}"
            );
            let preprocessed_owned = BaseAir::<F>::preprocessed_trace(air.as_ref());
            let preprocessed = preprocessed_owned.as_ref().map(RowMajorMatrix::as_view);
            let mut mains = context
                .cached_mains
                .iter()
                .map(|cached| cached.trace.as_view())
                .collect::<Vec<_>>();
            mains.push(context.common_main.as_view());
            check_constraints::<_, BabyBearPoseidon2Config>(
                air.as_ref(),
                &air.name(),
                &preprocessed,
                &mains,
                &context.public_values,
            );
        }
    }

    #[test]
    fn fixed_multi_air_forest_fanout_separates_source_and_mapped_namespaces() {
        let (_engine, module, records, _source) = retained_fixed_multi_air_fixture(vec![F::ONE]);
        let first_plan = module
            .mapped_functional_air
            .profile
            .terms
            .first()
            .expect("fixed mapped source plan");
        assert_eq!(first_plan.term_ordinal, 0);

        let trace = generate_forest_leaf_fanout_trace_v19(&records.boundary_shards, true)
            .expect("fixed forest fanout trace");
        let row = trace.row_slice(0).expect("fixed forest fanout row");
        let cols: &ForestLeafFanoutColsV19<F> = (&*row).borrow();
        assert_eq!(cols.source_air_id, F::from_u32(u32::MAX));
        assert_eq!(
            cols.source_log_height,
            F::from_u8(first_plan.log_message_len)
        );
        assert_eq!(cols.source_cached_width, F::ZERO);
        assert_eq!(cols.air_id, F::from_u32(first_plan.air_id));
        assert_eq!(cols.log_height, F::from_u8(first_plan.log_height));
        assert_eq!(cols.cached_width, F::from_u32(first_plan.cached_width));
        assert_ne!(cols.source_air_id, cols.air_id);

        // The host adapter must not put the canonical fixed-source namespace
        // on the mapped side. That would balance the source leaf twice while
        // leaving the first mapped-functional lookup unauthenticated.
        let mut conflated = records;
        conflated.boundary_shards[0].air_id = u32::MAX;
        conflated.boundary_shards[0].log_height = first_plan.log_message_len;
        conflated.boundary_shards[0].cached_width = 0;
        assert!(matches!(
            module.generate_proving_contexts::<BabyBearPoseidon2Config>(&conflated),
            Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "noncanonical shard record order or binding"
            ))
        ));
    }

    /// Check only the two Poseidon lookup buses. Other exported History buses
    /// intentionally remain open until the outer History AIRs are attached.
    fn check_packet_poseidon_balance(
        owner: Poseidon2BusOwner,
        airs: &[AirRef<BabyBearPoseidon2Config>],
        contexts: &[AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>],
    ) {
        assert_eq!(airs.len(), contexts.len());
        let preprocessed_matrices = airs
            .iter()
            .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
            .collect::<Vec<_>>();
        let preprocessed = preprocessed_matrices
            .iter()
            .map(|trace| trace.as_ref().map(|matrix| matrix.as_view()))
            .collect::<Vec<_>>();
        let owner_indices = [owner.permute_bus.index(), owner.compress_bus.index()];
        let interactions = airs
            .iter()
            .map(|air| {
                symbolic_interactions(air.as_ref())
                    .into_iter()
                    .filter(|interaction| owner_indices.contains(&interaction.bus_index))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert!(interactions.iter().flatten().next().is_some());
        let views = contexts
            .iter()
            .map(|context| {
                assert!(context.cached_mains.is_empty());
                vec![context.common_main.as_view()]
            })
            .collect::<Vec<_>>();
        let public_values = contexts
            .iter()
            .map(|context| context.public_values.clone())
            .collect::<Vec<_>>();
        let names = airs.iter().map(|air| air.name()).collect::<Vec<_>>();
        check_logup(&names, &interactions, &preprocessed, &views, &public_values);
    }

    fn check_transcript_bus_balance(
        transcript_bus_index: BusIndex,
        airs: &[AirRef<BabyBearPoseidon2Config>],
        contexts: &[AirProvingContext<CpuBackend<BabyBearPoseidon2Config>>],
    ) {
        assert_eq!(airs.len(), contexts.len());
        let preprocessed_matrices = airs
            .iter()
            .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
            .collect::<Vec<_>>();
        let preprocessed = preprocessed_matrices
            .iter()
            .map(|trace| trace.as_ref().map(|matrix| matrix.as_view()))
            .collect::<Vec<_>>();
        let interactions = airs
            .iter()
            .map(|air| {
                symbolic_interactions(air.as_ref())
                    .into_iter()
                    .filter(|interaction| interaction.bus_index == transcript_bus_index)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert!(interactions.iter().flatten().next().is_some());
        let views = contexts
            .iter()
            .map(|context| {
                let mut traces = context
                    .cached_mains
                    .iter()
                    .map(|cached| cached.trace.as_view())
                    .collect::<Vec<_>>();
                traces.push(context.common_main.as_view());
                traces
            })
            .collect::<Vec<_>>();
        let public_values = contexts
            .iter()
            .map(|context| context.public_values.clone())
            .collect::<Vec<_>>();
        let names = airs.iter().map(|air| air.name()).collect::<Vec<_>>();
        check_logup(&names, &interactions, &preprocessed, &views, &public_values);
    }

    #[test]
    fn fused_claim_start_cursor_is_constrained_by_the_existing_transcript_bus() {
        let (_engine, module, records, _source) = retained_fixed_multi_air_fixture(vec![F::ONE]);
        let airs = module.airs::<BabyBearPoseidon2Config>();
        let mut contexts = module
            .generate_proving_contexts::<BabyBearPoseidon2Config>(&records)
            .expect("fused production contexts");
        let transcript_bus_index = module.bus_inventory().transcript_bus.index();
        check_transcript_bus_balance(transcript_bus_index, &airs, &contexts);

        let mapped_index = airs
            .iter()
            .position(|air| air.name().contains("MappedFunctionalTermAirV19"))
            .expect("mapped-functional AIR index");
        let mapped = &mut contexts[mapped_index].common_main;
        let width = Matrix::width(mapped);
        let first: &mut MappedFunctionalTermColsV19<F> = mapped.values[..width].borrow_mut();
        first.claim_start_tidx += F::ONE;
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_transcript_bus_balance(transcript_bus_index, &airs, &contexts);
        }));
        assert!(
            rejected.is_err(),
            "a shifted fused claim cursor must not balance TranscriptBus"
        );
    }

    #[test]
    fn mapped_functional_low_fold_matches_dense_backend_mle() {
        let (_engine, module, records) = retained_composite_fixture();
        let trace = generate_mapped_functional_trace_v19(
            &module.mapped_functional_air.profile,
            &records.mapped_sources,
            &records.claim_transcripts,
        )
        .unwrap();
        for (row, plan) in module
            .mapped_functional_air
            .profile
            .terms
            .iter()
            .enumerate()
            .filter(|(_, plan)| plan.term_ordinal + 1 == plan.term_count)
        {
            let row_values = trace.row_slice(row).unwrap();
            let cols: &MappedFunctionalTermColsV19<F> = (&*row_values).borrow();
            let reduction = records
                .reductions
                .iter()
                .find(|record| record.shard_ordinal == plan.shard_ordinal)
                .unwrap();
            assert_eq!(
                EF::from_basis_coefficients_slice(&cols.ordinary_weight_after).unwrap(),
                reduction.ordinary_weight_at_point,
                "ordinary mapped weight for shard {}",
                plan.shard_ordinal,
            );
            assert_eq!(
                EF::from_basis_coefficients_slice(&cols.fingerprint_weight_after).unwrap(),
                reduction.fingerprint_weight_at_point,
                "fingerprint mapped weight for shard {}",
                plan.shard_ordinal,
            );
        }
    }

    #[test]
    fn production_composite_constructs_keygens_and_rejects_missing_or_tampered_source_provider() {
        let (engine, module, source_profile, records) = composite_fixture();
        let metrics = module.mapped_functional_metrics();
        assert_eq!(module.mapped_functional_air.profile.l_skip, 4);
        assert_eq!(
            metrics.logical_rows,
            module.mapped_functional_air.profile.term_count()
        );
        assert_eq!(metrics.main_cells, metrics.main_width * metrics.padded_rows);
        assert_eq!(
            metrics.lde_bytes,
            metrics.lde_cells * core::mem::size_of::<F>()
        );
        // The l_skip=4 specialization must not regress to the old >7k-column
        // generic envelope.
        assert!(metrics.main_width < 2_000, "metrics={metrics:?}");

        let airs = module.airs::<BabyBearPoseidon2Config>();
        engine.keygen(&airs);
        generate_direct_logup_source_manifest_traces_v19(&source_profile, &records).unwrap();

        let mut missing = records.clone();
        missing[0].sources.pop();
        assert!(
            generate_direct_logup_source_manifest_traces_v19(&source_profile, &missing).is_err()
        );

        let mut tampered = records.clone();
        tampered[0].sources[0].expected_instance_digest[0] += F::ONE;
        assert!(
            generate_direct_logup_source_manifest_traces_v19(&source_profile, &tampered).is_err()
        );

        let mut reordered = records;
        reordered[0].sources.swap(0, 1);
        assert!(
            generate_direct_logup_source_manifest_traces_v19(&source_profile, &reordered).is_err()
        );
    }

    #[test]
    fn production_composite_contexts_match_air_order_and_reject_malformed_records() {
        let (_engine, module, records) = retained_composite_fixture();
        let airs = module.airs::<BabyBearPoseidon2Config>();
        let contexts = module
            .generate_proving_contexts::<BabyBearPoseidon2Config>(&records)
            .expect("production composite contexts");
        assert_eq!(contexts.len(), airs.len());
        for (index, (air, context)) in airs.iter().zip(&contexts).enumerate() {
            assert_eq!(
                Matrix::width(&context.common_main),
                air.common_main_width(),
                "common-main width at AIR/context index {index}"
            );
            assert_eq!(
                context.cached_mains.len(),
                air.cached_main_widths().len(),
                "cached-main arity at AIR/context index {index}"
            );
            for (cached, expected_width) in
                context.cached_mains.iter().zip(air.cached_main_widths())
            {
                assert_eq!(
                    Matrix::width(&cached.trace),
                    expected_width,
                    "cached width at {index}"
                );
            }
        }

        let mut missing = records.clone();
        missing.reductions.pop();
        assert!(matches!(
            module.generate_proving_contexts::<BabyBearPoseidon2Config>(&missing),
            Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "composite record cardinality"
            ))
        ));

        let mut reordered = records.clone();
        reordered.mapped_sources.swap(0, 1);
        assert!(matches!(
            module.generate_proving_contexts::<BabyBearPoseidon2Config>(&reordered),
            Err(DirectLogUpOnlyCompositeContextErrorV19::Shape(
                "noncanonical shard record order or binding"
            ))
        ));

        let mut tampered = records;
        tampered.sources[0].sources[0].expected_instance_digest[0] += F::ONE;
        assert!(matches!(
            module.generate_proving_contexts::<BabyBearPoseidon2Config>(&tampered),
            Err(DirectLogUpOnlyCompositeContextErrorV19::Source(_))
        ));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn production_composite_mixed_cuda_packet_matches_fresh_cpu_oracle() {
        let (_engine, module, records) = retained_composite_fixture();
        let cpu = module
            .generate_proving_contexts_without_poseidon::<BabyBearPoseidon2Config>(&records)
            .expect("CPU composite packet");
        let device_ctx = GpuDeviceCtx::for_current_device().expect("CUDA device");
        let mixed = module
            .generate_proving_contexts_without_poseidon_cuda(
                &records,
                CachedTraceCtx::Records(module.partial.cached_trace_record(&module.child_vk)),
                &device_ctx,
            )
            .expect("mixed CUDA composite packet");

        assert_eq!(
            cpu.poseidon.owner.permute_bus.index(),
            mixed.poseidon.owner.permute_bus.index()
        );
        assert_eq!(
            cpu.poseidon.owner.compress_bus.index(),
            mixed.poseidon.owner.compress_bus.index()
        );
        assert_eq!(
            cpu.poseidon.permutation_inputs,
            mixed.poseidon.permutation_inputs
        );
        assert_eq!(
            cpu.poseidon.compression_inputs,
            mixed.poseidon.compression_inputs
        );
        assert_eq!(cpu.logup_producers, mixed.logup_producers);
        assert_eq!(cpu.contexts.len(), mixed.contexts.len());
        assert_eq!(
            mixed.contexts.len(),
            module
                .airs_without_poseidon::<BabyBearPoseidon2Config>()
                .len()
        );

        let partial_count = module
            .partial
            .airs_without_poseidon::<BabyBearPoseidon2Config>()
            .len();
        let mut extended_transcript_indices = Vec::new();
        let mut rebased_transcript_indices = Vec::new();
        let mut device_indices = Vec::new();
        for (index, (cpu_context, mixed_context)) in
            cpu.contexts.iter().zip(&mixed.contexts).enumerate()
        {
            match mixed_context {
                DirectLogUpOnlyCompositeCudaContextV19::Device(device_context) => {
                    assert!(
                        index < partial_count,
                        "device context escaped partial prefix"
                    );
                    device_indices.push(index);
                    assert_eq!(
                        cpu_context.public_values, device_context.public_values,
                        "public values at {index}"
                    );
                    assert_eq!(
                        cpu_context.cached_mains.len(),
                        device_context.cached_mains.len(),
                        "cached-main count at {index}"
                    );
                    assert_eq_host_and_device_matrix(
                        Arc::new(cpu_context.common_main.clone()),
                        &device_context.common_main,
                        &device_ctx,
                    );
                    for (cached_index, (cpu_cached, device_cached)) in cpu_context
                        .cached_mains
                        .iter()
                        .zip(&device_context.cached_mains)
                        .enumerate()
                    {
                        assert_eq!(
                            cpu_cached.commitment, device_cached.commitment,
                            "cached commitment at {index}:{cached_index}"
                        );
                        assert_eq_host_and_device_matrix(
                            Arc::new(cpu_cached.trace.clone()),
                            &device_cached.trace,
                            &device_ctx,
                        );
                    }
                }
                DirectLogUpOnlyCompositeCudaContextV19::HostExtendedTranscript(host_context) => {
                    assert!(index < partial_count);
                    extended_transcript_indices.push(index);
                    assert_context_equivalent(cpu_context, host_context);
                }
                DirectLogUpOnlyCompositeCudaContextV19::HostRebasedTranscriptAdapter(
                    host_context,
                ) => {
                    assert!(index < partial_count);
                    rebased_transcript_indices.push(index);
                    assert_context_equivalent(cpu_context, host_context);
                }
                DirectLogUpOnlyCompositeCudaContextV19::HostV19(host_context) => {
                    assert!(index >= partial_count, "v19 context entered partial prefix");
                    assert_context_equivalent(cpu_context, host_context);
                }
            }
        }
        assert_eq!(extended_transcript_indices.len(), 2);
        assert_eq!(
            extended_transcript_indices[1],
            extended_transcript_indices[0] + 1
        );
        assert_eq!(rebased_transcript_indices.len(), 1);
        assert!(!device_indices.is_empty());
        assert_eq!(
            mixed.contexts[partial_count..]
                .iter()
                .filter(|context| matches!(
                    context,
                    DirectLogUpOnlyCompositeCudaContextV19::HostV19(_)
                ))
                .count(),
            mixed.contexts.len() - partial_count
        );
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn production_composite_mixed_cuda_rejects_wrong_equation_mode_before_tracegen() {
        use openvm_recursion_circuit::system::VerifierEquationMode;

        let (_engine, module, records) = retained_composite_fixture();
        let mut prepared = prepare_direct_logup_only_composite_v19::<BabyBearPoseidon2Config, 4>(
            &module, &records, None,
        )
        .expect("valid shared preflight");
        prepared.preflights[0].batch_constraint.equation_mode = VerifierEquationMode::AirAndLogUp;
        let device_ctx = GpuDeviceCtx::for_current_device().expect("CUDA device");
        let result = generate_direct_logup_only_composite_contexts_from_prepared_cuda_v19(
            &module,
            prepared,
            CachedTraceCtx::Records(module.partial.cached_trace_record(&module.child_vk)),
            &device_ctx,
        );
        assert!(matches!(
            result,
            Err(DirectLogUpOnlyCompositeContextErrorV19::PartialVerifier)
        ));
    }

    #[test]
    fn production_composite_packet_poseidon_matches_owner_path_and_rejects_omission() {
        let (_engine, module, records) = retained_composite_fixture();
        let owner_contexts = module
            .generate_proving_contexts::<BabyBearPoseidon2Config>(&records)
            .expect("owner contexts");
        let packet = module
            .generate_proving_contexts_without_poseidon::<BabyBearPoseidon2Config>(&records)
            .expect("packet contexts");
        let owner_index = module
            .partial
            .poseidon_air_index::<BabyBearPoseidon2Config>();
        let owner_airs = module.airs::<BabyBearPoseidon2Config>();
        let packet_only_airs = module.airs_without_poseidon::<BabyBearPoseidon2Config>();
        assert_eq!(owner_airs.len(), packet_only_airs.len() + 1);
        for (index, packet_air) in packet_only_airs.iter().enumerate() {
            let owner_air = &owner_airs[index + usize::from(index >= owner_index)];
            assert_eq!(owner_air.name(), packet_air.name());
            assert_eq!(
                owner_air.common_main_width(),
                packet_air.common_main_width()
            );
            assert_eq!(
                owner_air.cached_main_widths(),
                packet_air.cached_main_widths()
            );
        }
        assert_eq!(owner_contexts.len(), packet.contexts.len() + 1);
        for (index, packet_context) in packet.contexts.iter().enumerate() {
            let owner_context = &owner_contexts[index + usize::from(index >= owner_index)];
            assert_context_equivalent(owner_context, packet_context);
        }

        let shared_context = module
            .build_shared_poseidon_context::<BabyBearPoseidon2Config>(
                packet.poseidon.grouped_inputs(),
            )
            .expect("one-owner multi-bus table");
        assert_context_equivalent(&owner_contexts[owner_index], &shared_context);
        check_packet_poseidon_balance(packet.poseidon.owner, &owner_airs, &owner_contexts);

        let mut packet_airs = packet_only_airs;
        packet_airs
            .push(module.shared_poseidon_air::<BabyBearPoseidon2Config>(&[packet.poseidon.owner]));
        let mut packet_contexts = packet.contexts;
        packet_contexts.push(shared_context);
        check_packet_poseidon_balance(packet.poseidon.owner, &packet_airs, &packet_contexts);

        let mut omitted_inputs = packet.poseidon.grouped_inputs();
        assert!(omitted_inputs[0].0.pop().is_some());
        let omitted_context = module
            .build_shared_poseidon_context::<BabyBearPoseidon2Config>(omitted_inputs)
            .expect("well-formed but incomplete table");
        *packet_contexts.last_mut().expect("shared table context") = omitted_context;
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_packet_poseidon_balance(packet.poseidon.owner, &packet_airs, &packet_contexts);
        }));
        assert!(
            rejected.is_err(),
            "omitting one packet permutation must unbalance the owner bus"
        );
    }

    #[test]
    fn genuine_source_fanout_authenticates_c1_point_and_rejects_mutation() {
        let (_engine, mut module, records, _) =
            retained_fixed_multi_air_fixture(vec![F::from_u32(7); DIGEST_SIZE]);
        let source_point = records.mapped_sources[0].logup_opening_point[0];
        let mut manager = BusIndexManager::from_next_bus_idx(module.next_bus_idx());
        let point_bus = FixedSetupOpeningPointBusV2::new(manager.new_bus_idx());
        let profile = FixedSetupOpeningProfileV2::new(
            module.child_vk.pre_hash,
            vec![FixedSetupMatrixV2 {
                setup_index: 0,
                air_id: 0,
                relation_digest: module.mapped_functional_air.profile.terms[0].relation_digest,
                width: 1,
                height: 1 << OPENVM_DIRECT_LOGUP_L_SKIP_V19,
                values: vec![F::from_u32(31); 1 << OPENVM_DIRECT_LOGUP_L_SKIP_V19].into(),
            }],
            vec![FixedSetupOpeningInstanceV2 {
                proof_index: 0,
                matrix_index: 0,
                sort_idx: 0,
                part_idx: 1,
                l_skip: OPENVM_DIRECT_LOGUP_L_SKIP_V19,
                log_height: OPENVM_DIRECT_LOGUP_L_SKIP_V19,
                need_rot: true,
            }],
        )
        .expect("C1 profile");
        module
            .attach_fixed_setup_opening_points(point_bus, profile.point_lookup_demands())
            .expect("attach genuine source fanout to C1");
        let c1_air = FixedSetupOpeningAirV2 {
            profile: profile.clone(),
            point_bus,
            node_bus: FixedSetupOpeningNodeBusV2::new(manager.new_bus_idx()),
            verified_pair_bus: VerifiedFixedSetupOpeningPairBusV2::new(manager.new_bus_idx()),
            column_claims_bus: module.bus_inventory().column_claims_bus,
            certificate_bus: FixedSetupOpeningCertificateBusV2::new(manager.new_bus_idx()),
            compress_bus: module.history_buses().compress,
        };
        let c1_records = |point: EF| {
            vec![FixedSetupOpeningProofRecordV2 {
                proof_index: 0,
                opening_point: vec![point],
                claims: vec![FixedSetupOpeningClaimRecordV2 {
                    setup_index: 0,
                    air_id: 0,
                    sort_idx: 0,
                    part_idx: 1,
                    col_idx: 0,
                    current: EF::from_u32(31),
                    rotated: Some(EF::from_u32(31)),
                }],
            }]
        };
        let honest_c1 = generate_fixed_setup_opening_trace_v2(&c1_air, &c1_records(source_point))
            .expect("honest C1 trace");
        let source_packet = module
            .generate_proving_contexts_without_poseidon::<BabyBearPoseidon2Config>(&records)
            .expect("genuine source contexts");
        let source_airs = module.airs_without_poseidon::<BabyBearPoseidon2Config>();
        let fanout_index = source_airs
            .iter()
            .position(|air| air.name().contains("OpeningPointFanoutAirV19"))
            .expect("opening-point fanout AIR slot");
        let fanout_air = source_airs[fanout_index].as_ref();
        let fanout_context = &source_packet.contexts[fanout_index];

        let check = |c1_matrix: &RowMajorMatrix<F>| {
            let airs: Vec<&dyn AnyAir<BabyBearPoseidon2Config>> = vec![fanout_air, &c1_air];
            let preprocessed_owned = airs
                .iter()
                .map(|air| BaseAir::<F>::preprocessed_trace(*air))
                .collect::<Vec<_>>();
            let preprocessed = preprocessed_owned
                .iter()
                .map(|trace| trace.as_ref().map(RowMajorMatrix::as_view))
                .collect::<Vec<_>>();
            let interactions = airs
                .iter()
                .map(|air| {
                    symbolic_interactions(*air)
                        .into_iter()
                        .filter(|interaction| interaction.bus_index == point_bus.index())
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let views = vec![
                vec![fanout_context.common_main.as_view()],
                vec![c1_matrix.as_view()],
            ];
            check_logup(
                &["genuine source fanout".to_owned(), "C1".to_owned()],
                &interactions,
                &preprocessed,
                &views,
                &[Vec::new(), Vec::new()],
            );
        };
        check(&honest_c1.matrix);

        let wrong_c1 =
            generate_fixed_setup_opening_trace_v2(&c1_air, &c1_records(source_point + EF::ONE))
                .expect("internally consistent wrong-point C1 trace");
        let rejected = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check(&wrong_c1.matrix);
        }));
        assert!(
            rejected.is_err(),
            "mutating C1's point must unbalance the genuine source fanout bus"
        );
    }

    #[test]
    fn fixed_setup_point_demands_preserve_full_source_row_alignment() {
        let aligned =
            align_fixed_setup_point_demands_v19(&[4, 3], &[(0, 0, 2), (0, 1, 2), (1, 0, 2)])
                .expect("prefix demands fit full source points");
        assert_eq!(
            aligned,
            vec![
                (0, 0, 2),
                (0, 1, 2),
                (0, 2, 0),
                (0, 3, 0),
                (1, 0, 2),
                (1, 1, 0),
                (1, 2, 0),
            ]
        );
        assert!(align_fixed_setup_point_demands_v19(&[2], &[(0, 2, 1)]).is_err());
        assert!(align_fixed_setup_point_demands_v19(&[2, 0], &[(0, 0, 1)]).is_err());
    }

    #[test]
    fn fixed_capacity_opening_fanout_keeps_inactive_slots_on_key_domain() {
        let air = OpeningPointFanoutAirV19 {
            segment_start: 0,
            source_bus: LogUpOnlyOpeningPointBus::new(BusIndex::from(1u16)),
            lookup_bus: SegmentOpeningPointBusV19::new(BusIndex::from(2u16)),
            fixed_setup_point_bus: Some(FixedSetupOpeningPointBusV2::new(BusIndex::from(3u16))),
            fixed_setup_point_demands: vec![(0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1)].into(),
        };
        let trace = generate_opening_point_fanout_trace_fixed_capacity_v4(
            &air,
            &[OpeningPointFanoutRecordV19 {
                proof_index: 0,
                segment_index: 0,
                values: vec![EF::from_u32(9)],
                consumer_counts: vec![1],
            }],
        )
        .expect("one active slot fits fixed fanout");
        let prep = air.preprocessed_trace().expect("fixed fanout prep");
        assert_eq!(Matrix::height(&trace), Matrix::height(&prep));
        assert_eq!(Matrix::height(&trace), 4);
        assert!(trace.values[air.width()..]
            .iter()
            .all(|value| *value == F::ZERO));
        check_constraints::<_, BabyBearPoseidon2Config>(
            &air,
            "OpeningPointFanoutAirV19",
            &Some(prep.as_view()),
            &[trace.as_view()],
            &[],
        );
    }
}

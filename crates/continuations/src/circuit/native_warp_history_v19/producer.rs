//! Sound positive lookup producers for protocol-v19 History.
//!
//! The segment-local SWIRL verifier exports exactly one raw-message MLE
//! opening `(r, v)` per active homogeneous shard.  The direct WARP verifier
//! exports the fresh input it actually consumed.  This module equates those
//! records and enforces the systematic RS lift
//! `alpha = (r, 0^log_blowup)`, `mu = v`, and `eta = 0` before it publishes a
//! positive History lookup.  No mapped-column claim survives the segment.

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_recursion_circuit::{
    bus::{CertifiedTranscriptCheckpointBus, CertifiedTranscriptCheckpointMessage},
    native_warp::{
        NativeCertifiedAccumulatorDigestBus, NativeCertifiedAccumulatorDigestMessage,
        NativeCertifiedBatchingClaimBus, NativeCertifiedBatchingClaimMessage,
        NativeFixedHLeafVaccRouteBusV4, NativeFixedHLeafVaccRouteMessageV4,
        NativeFixedHLeafVaccRouteV4, NativeFixedHLeafVaccSeedV4,
        NativeFixedHLeafVaccTransitionBusV4, NativeFixedHLeafVaccTransitionMessageV4,
        FIXED_HLEAF_CONTINUATION_WARP_STEP_V4, FIXED_HLEAF_VACC_CAPACITY_V4,
    },
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, LookupBus},
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, D_EF, F};

use super::{
    digest::{scalar_digest, TAG_REPLAY_META_V19},
    record::split_u32,
    CertifiedWarpReplayBusV19, CertifiedWarpReplayMessageV19, DigestCollectorV19, DigestV19,
    ExtensionV19, HistoryPoseidon2CompressBusV19, HistoryPoseidon2CompressMessageV19,
    LogUpOnlyHistoryBusV19, LogUpOnlyHistoryMessageV19, SetupPcsSourceCheckpointMessageV3,
    LOGUP_ONLY_MODE_TAG_V19, NATIVE_WARP_HISTORY_PROTOCOL_V19,
    SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3,
};

pub const TRANSCRIPT_WIDTH_V19: usize = 16;
pub const MAX_RAW_MESSAGE_POINT_LEN_V19: usize = 32;
/// Maximum length of the authenticated direct-PESAT `beta` vector.
///
/// The capacity-four complete SWIRL verifier relation has 29 constraint
/// coordinates and 103 explicit coordinates (the distinguished one, the
/// setup-fixed verifier/DAG public values, and the aggregate VM public
/// values), for a total of 132.  This is a fixed History inventory bound, not
/// a proof-selected relation dimension.  Leave headroom for setup-compatible
/// verifier profiles while keeping every unused slot constrained to zero.
pub const MAX_FRESH_BETA_LEN_V19: usize = 160;
const LIMB_BITS: usize = 16;
// The first 72 slots authenticate transcript/opening/alpha data. Slot 72 is
// the beta-domain separator and the following fixed-capacity chain hashes all
// 160 beta coordinates, including constrained zero padding. Keeping the
// padded chain in the digest makes its layout independent of the selected
// fixed relation profile while `beta_len` in the domain separator binds the
// actual relation dimension.
const WARP_ALPHA_DIGEST_HASH_V19: usize = 71;
const WARP_BETA_META_HASH_V19: usize = 72;
const WARP_BETA_HASH_START_V19: usize = WARP_BETA_META_HASH_V19 + 1;
const WARP_BETA_DIGEST_HASH_V19: usize = WARP_BETA_HASH_START_V19 + MAX_FRESH_BETA_LEN_V19 - 1;
// Fresh-instance, replay-endpoint, and replay-binding trees use 28 hashes
// after the beta digest. This was formerly hard-coded for a 64-coordinate
// beta and silently became invalid when the fixed verifier relation grew.
const WARP_POST_BETA_HASH_START_V19: usize = WARP_BETA_DIGEST_HASH_V19 + 1;
const WARP_POST_BETA_HASH_SLOTS_V19: usize = 28;
const WARP_HASH_SLOTS_V19: usize = WARP_POST_BETA_HASH_START_V19 + WARP_POST_BETA_HASH_SLOTS_V19;
const LOGUP_HASH_SLOTS_V19: usize = 12;

const TAG_WARP_CHECKPOINT_V19: u32 = 0x19_20;
const TAG_WARP_BATCHING_CLAIM_V19: u32 = 0x19_21;
const TAG_RAW_OPENING_V19: u32 = 0x19_22;
const TAG_ALPHA_V19: u32 = 0x19_23;
const TAG_BETA_V19: u32 = 0x19_24;
const TAG_FRESH_INSTANCE_V19: u32 = 0x19_25;
const TAG_WARP_ENDPOINT_V19: u32 = 0x19_26;
const TAG_LOGUP_CHECKPOINT_V19: u32 = 0x19_27;
const TAG_LOGUP_ENDPOINT_V19: u32 = 0x19_28;
const TAG_LOGUP_PRODUCER_V19: u32 = 0x19_29;

macro_rules! define_producer_lookup_bus {
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

/// Metadata authenticated by the fixed direct-AIR index, source forest, and
/// prior/output accumulator verifier gadgets.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct DirectAirVaccContextMessageV19<T> {
    pub proof_index: T,
    pub protocol_version: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub update_index_lo: T,
    pub update_index_hi: T,
    pub shard_ordinal: T,
    pub has_prior: T,
    pub key_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub prior_root: [T; DIGEST_SIZE],
    pub next_root: [T; DIGEST_SIZE],
    pub previous_accumulator_digest: [T; DIGEST_SIZE],
    /// Product-state checkpoint supplied to the shard transcript factory.
    /// This is not a digest of the transcript's full sponge state.
    pub previous_checkpoint_digest: [T; DIGEST_SIZE],
    /// Product-state checkpoint returned by the shard transcript factory.
    /// This is not a digest of the transcript's full sponge state.
    pub next_checkpoint_digest: [T; DIGEST_SIZE],
}

define_producer_lookup_bus!(DirectAirVaccContextBusV19, DirectAirVaccContextMessageV19);

/// One raw-message opening certified by the segment-local SWIRL opening
/// reduction.  Unused coordinates are canonical zero padding.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct CertifiedSwirlRawOpeningMessageV19<T> {
    pub proof_index: T,
    pub protocol_version: T,
    pub mode_tag: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub shard_ordinal: T,
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub root: [T; DIGEST_SIZE],
    pub point_len: T,
    pub point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub value: [T; D_EF],
}

define_producer_lookup_bus!(
    CertifiedSwirlRawOpeningBusV19,
    CertifiedSwirlRawOpeningMessageV19
);

/// Fresh public input consumed by the ordinary direct-AIR WARP VACC verifier.
/// This is an output of that verifier, not a History witness assertion.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct CertifiedDirectAirVaccInputMessageV19<T> {
    pub proof_index: T,
    pub protocol_version: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub update_index_lo: T,
    pub update_index_hi: T,
    pub shard_ordinal: T,
    pub relation_digest: [T; DIGEST_SIZE],
    pub root: [T; DIGEST_SIZE],
    pub alpha_len: T,
    pub alpha: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub mu: [T; D_EF],
    pub beta_len: T,
    pub beta: [[T; D_EF]; MAX_FRESH_BETA_LEN_V19],
    pub eta: [T; D_EF],
}

define_producer_lookup_bus!(
    CertifiedDirectAirVaccInputBusV19,
    CertifiedDirectAirVaccInputMessageV19
);

/// Relation-bound digest of the complete fresh explicit vector (`beta`).
/// This is independent of alpha/mu, so the source transcript may absorb it
/// before deriving the one-shot same-root opening point.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct CertifiedFreshExplicitDigestMessageV19<T> {
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub relation_digest: [T; DIGEST_SIZE],
    pub digest: [T; DIGEST_SIZE],
}

define_producer_lookup_bus!(
    CertifiedFreshExplicitDigestBusV19,
    CertifiedFreshExplicitDigestMessageV19
);

/// Endpoint exported only after the mode-aware recursive GKR and batch-
/// constraint verifier has checked LogUpOnly and derived the canonical raw
/// openings aggregate.  Both sums are explicit so the producer can enforce
/// segment-local cancellation rather than a cross-segment recurrence.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct CertifiedLogUpOnlyEndpointMessageV19<T> {
    pub proof_index: T,
    pub protocol_version: T,
    pub mode_tag: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub app_vk_digest: [T; DIGEST_SIZE],
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub verifier_endpoint: [T; D_EF],
    pub segment_sum_before: [T; D_EF],
    pub segment_sum_after: [T; D_EF],
}

define_producer_lookup_bus!(
    CertifiedLogUpOnlyEndpointBusV19,
    CertifiedLogUpOnlyEndpointMessageV19
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TranscriptCheckpointRecordV19 {
    pub operation_index: u32,
    pub sample_count: u8,
    pub state: [F; TRANSCRIPT_WIDTH_V19],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectAirVaccProducerRecordV19 {
    pub proof_index: u32,
    pub segment_index: u32,
    pub update_index: u32,
    pub shard_ordinal: u16,
    pub has_prior: bool,
    pub key_digest: DigestV19,
    pub relation_digest: DigestV19,
    pub source_forest_root: DigestV19,
    pub segment_openings_digest: DigestV19,
    pub prior_root: DigestV19,
    pub fresh_root: DigestV19,
    pub next_root: DigestV19,
    pub opening_point: Vec<ExtensionV19>,
    pub opening_value: ExtensionV19,
    pub fresh_alpha: Vec<ExtensionV19>,
    pub fresh_mu: ExtensionV19,
    pub fresh_beta: Vec<ExtensionV19>,
    pub fresh_eta: ExtensionV19,
    pub previous_accumulator_digest: DigestV19,
    pub next_accumulator_digest: DigestV19,
    pub previous_checkpoint_digest: DigestV19,
    pub next_checkpoint_digest: DigestV19,
    pub authenticated_batching_claim: ExtensionV19,
    pub start_checkpoint: TranscriptCheckpointRecordV19,
    pub end_checkpoint: TranscriptCheckpointRecordV19,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogUpOnlyProducerRecordV19 {
    pub proof_index: u32,
    pub segment_index: u32,
    pub app_vk_digest: DigestV19,
    pub source_forest_root: DigestV19,
    pub segment_openings_digest: DigestV19,
    pub verifier_endpoint: ExtensionV19,
    pub segment_sum_before: ExtensionV19,
    pub segment_sum_after: ExtensionV19,
    pub start_checkpoint: TranscriptCheckpointRecordV19,
    pub end_checkpoint: TranscriptCheckpointRecordV19,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PositiveProducerErrorV19 {
    InvalidFixedHeight,
    CapacityExceeded,
    InvalidProfile,
    RecordShape,
    InvalidPriorRoot,
    NonZeroLogUpBoundary,
}

#[derive(Clone, Debug)]
pub struct PositiveProducerTraceV19<M> {
    pub matrix: RowMajorMatrix<F>,
    pub compression_inputs: Vec<[F; 2 * DIGEST_SIZE]>,
    pub messages: Vec<M>,
    pub fresh_explicit_messages: Vec<CertifiedFreshExplicitDigestMessageV19<F>>,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct WarpReplayProducerColsV19<T> {
    pub active: T,
    pub proof_index_lo: T,
    pub proof_index_hi: T,
    pub proof_index_bits: [[T; LIMB_BITS]; 2],
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub segment_index_bits: [[T; LIMB_BITS]; 2],
    pub update_index_lo: T,
    pub update_index_hi: T,
    pub update_index_bits: [[T; LIMB_BITS]; 2],
    pub shard_ordinal: T,
    pub shard_ordinal_bits: [T; LIMB_BITS],
    pub has_prior: T,
    pub key_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub prior_root: [T; DIGEST_SIZE],
    pub fresh_root: [T; DIGEST_SIZE],
    pub next_root: [T; DIGEST_SIZE],
    pub opening_point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub opening_value: [T; D_EF],
    pub fresh_alpha: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub fresh_mu: [T; D_EF],
    pub fresh_beta: [[T; D_EF]; MAX_FRESH_BETA_LEN_V19],
    pub fresh_eta: [T; D_EF],
    pub previous_accumulator_digest: [T; DIGEST_SIZE],
    pub next_accumulator_digest: [T; DIGEST_SIZE],
    pub authenticated_batching_claim: [T; D_EF],
    pub start_tidx_lo: T,
    pub start_tidx_hi: T,
    pub start_sample_count: T,
    pub start_state: [T; TRANSCRIPT_WIDTH_V19],
    pub end_tidx_lo: T,
    pub end_tidx_hi: T,
    pub end_sample_count: T,
    pub end_state: [T; TRANSCRIPT_WIDTH_V19],
    pub opening_claim_digest: [T; DIGEST_SIZE],
    pub fresh_instance_digest: [T; DIGEST_SIZE],
    /// History-local digest of the certified replay start sponge state.
    pub transcript_start_digest: [T; DIGEST_SIZE],
    /// History-local digest of the certified replay end sponge state.
    pub transcript_end_digest: [T; DIGEST_SIZE],
    /// Exact product-state checkpoint digests, kept separate from the two
    /// transcript-state digests above.
    pub previous_checkpoint_digest: [T; DIGEST_SIZE],
    pub next_checkpoint_digest: [T; DIGEST_SIZE],
    pub replay_endpoint_digest: [T; DIGEST_SIZE],
    pub replay_binding_digest: [T; DIGEST_SIZE],
    pub hash_outputs: [[T; DIGEST_SIZE]; WARP_HASH_SLOTS_V19],
}

#[derive(Clone, Debug)]
pub struct WarpReplayProducerAirV19 {
    pub log_message_len: usize,
    pub log_codeword_len: usize,
    pub beta_len: usize,
    pub compress_bus: HistoryPoseidon2CompressBusV19,
    pub history_bus: CertifiedWarpReplayBusV19,
    pub context_bus: DirectAirVaccContextBusV19,
    pub swirl_opening_bus: CertifiedSwirlRawOpeningBusV19,
    pub vacc_input_bus: CertifiedDirectAirVaccInputBusV19,
    /// Optional canonical History-facing VACC-input bus.  The verifier
    /// module uses dense group-local proof identifiers, while the History
    /// composition uses setup-fixed global batch identifiers.  When this bus
    /// is present the producer authenticates the local input on
    /// `vacc_input_bus` and republishes the same statement under the global
    /// proof identifier.  This is a constrained namespace remap, not a
    /// second source of authority.
    pub canonical_vacc_input_bus: Option<CertifiedDirectAirVaccInputBusV19>,
    pub fresh_explicit_bus: CertifiedFreshExplicitDigestBusV19,
    pub fresh_explicit_lookup_count: usize,
    pub checkpoint_bus: CertifiedTranscriptCheckpointBus,
    pub batching_claim_bus: NativeCertifiedBatchingClaimBus,
    pub next_accumulator_digest_bus: NativeCertifiedAccumulatorDigestBus,
    /// Optional setup-owned proof-index/mode schedule. Production composites
    /// must set this; `None` is retained only for legacy standalone fixtures
    /// whose sparse proof identifiers are supplied by a surrounding AIR.
    pub setup_schedule: Option<WarpReplayProducerScheduleV19>,
}

/// Setup-fixed partition of replay records authenticated by one VACC module.
///
/// Bootstrap and prior-bearing VACC modules own disjoint PCD bus inventories,
/// so their replay producers must also own disjoint, contiguous proof-index
/// ranges. This schedule prevents routing a continuation record through the
/// bootstrap producer (or silently omitting the continuation producer).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WarpReplayProducerScheduleV19 {
    pub proof_index_start: u32,
    pub proof_count: u32,
    pub has_prior: bool,
}

impl WarpReplayProducerScheduleV19 {
    pub fn new(
        proof_index_start: u32,
        proof_count: usize,
        has_prior: bool,
    ) -> Result<Self, PositiveProducerErrorV19> {
        let proof_count =
            u32::try_from(proof_count).map_err(|_| PositiveProducerErrorV19::InvalidProfile)?;
        if proof_count == 0 || proof_index_start.checked_add(proof_count - 1).is_none() {
            return Err(PositiveProducerErrorV19::InvalidProfile);
        }
        Ok(Self {
            proof_index_start,
            proof_count,
            has_prior,
        })
    }

    #[must_use]
    pub fn proof_index_end(self) -> u32 {
        self.proof_index_start + self.proof_count - 1
    }
}

impl WarpReplayProducerAirV19 {
    pub fn validate_profile(&self) -> Result<(), PositiveProducerErrorV19> {
        if self.log_message_len == 0
            || self.log_message_len > MAX_RAW_MESSAGE_POINT_LEN_V19
            || self.log_codeword_len < self.log_message_len
            || self.log_codeword_len > MAX_RAW_MESSAGE_POINT_LEN_V19
            || self.beta_len == 0
            || self.beta_len > MAX_FRESH_BETA_LEN_V19
        {
            Err(PositiveProducerErrorV19::InvalidProfile)
        } else {
            Ok(())
        }
    }

    /// Production validation: a replay producer without a setup-fixed range
    /// is not authority for a multi-transition History component.
    pub fn validate_setup_schedule(&self) -> Result<(), PositiveProducerErrorV19> {
        self.validate_profile()?;
        let schedule = self
            .setup_schedule
            .ok_or(PositiveProducerErrorV19::InvalidProfile)?;
        WarpReplayProducerScheduleV19::new(
            schedule.proof_index_start,
            schedule.proof_count as usize,
            schedule.has_prior,
        )?;
        Ok(())
    }
}

impl BaseAir<F> for WarpReplayProducerAirV19 {
    fn width(&self) -> usize {
        WarpReplayProducerColsV19::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for WarpReplayProducerAirV19 {}
impl PartitionedBaseAir<F> for WarpReplayProducerAirV19 {}

impl<AB> Air<AB> for WarpReplayProducerAirV19
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(
            self.validate_profile().is_ok(),
            "invalid WARP producer profile"
        );
        let main = builder.main();
        let local_row = main.row_slice(0).expect("WARP producer row");
        let next_row = main.row_slice(1).expect("WARP producer next row");
        let local: &WarpReplayProducerColsV19<AB::Var> = (*local_row).borrow();
        let next: &WarpReplayProducerColsV19<AB::Var> = (*next_row).borrow();
        let enabled = local.active;
        builder.assert_bool(enabled);
        builder
            .when_transition()
            .when(next.active)
            .assert_one(local.active);
        builder.when(enabled).assert_bool(local.has_prior);
        assert_u32(
            builder,
            local.proof_index_lo,
            local.proof_index_hi,
            &local.proof_index_bits,
            enabled,
        );
        assert_u32(
            builder,
            local.segment_index_lo,
            local.segment_index_hi,
            &local.segment_index_bits,
            enabled,
        );
        assert_u32(
            builder,
            local.update_index_lo,
            local.update_index_hi,
            &local.update_index_bits,
            enabled,
        );
        assert_bits(builder, &local.shard_ordinal_bits, enabled);
        builder.when(enabled).assert_eq(
            local.shard_ordinal,
            bits_expr::<AB>(&local.shard_ordinal_bits),
        );

        for index in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
            for limb in 0..D_EF {
                if index >= self.log_message_len {
                    builder
                        .when(enabled)
                        .assert_zero(local.opening_point[index][limb]);
                }
                if index < self.log_message_len {
                    builder.when(enabled).assert_eq(
                        local.fresh_alpha[index][limb],
                        local.opening_point[index][limb],
                    );
                } else {
                    builder
                        .when(enabled)
                        .assert_zero(local.fresh_alpha[index][limb]);
                }
            }
        }
        for index in self.beta_len..MAX_FRESH_BETA_LEN_V19 {
            for limb in 0..D_EF {
                builder
                    .when(enabled)
                    .assert_zero(local.fresh_beta[index][limb]);
            }
        }
        for limb in 0..D_EF {
            builder
                .when(enabled)
                .assert_eq(local.fresh_mu[limb], local.opening_value[limb]);
            builder.when(enabled).assert_zero(local.fresh_eta[limb]);
        }
        for limb in 0..DIGEST_SIZE {
            builder
                .when(AB::Expr::from(enabled) * (AB::Expr::ONE - local.has_prior))
                .assert_zero(local.prior_root[limb]);
        }

        let proof_index = join_u32_expr::<AB>(local.proof_index_lo, local.proof_index_hi);
        if let Some(schedule) = self.setup_schedule {
            let next_proof_index = join_u32_expr::<AB>(next.proof_index_lo, next.proof_index_hi);
            let final_proof_index = AB::Expr::from_u32(schedule.proof_index_end());
            builder.when_first_row().assert_one(enabled);
            builder.when_first_row().assert_eq(
                proof_index.clone(),
                AB::Expr::from_u32(schedule.proof_index_start),
            );
            builder
                .when(enabled)
                .assert_eq(local.has_prior, AB::Expr::from_bool(schedule.has_prior));
            builder
                .when_transition()
                .when(next.active)
                .assert_eq(next_proof_index, proof_index.clone() + AB::Expr::ONE);
            builder
                .when_transition()
                .when(AB::Expr::from(enabled) * (AB::Expr::ONE - next.active))
                .assert_eq(proof_index.clone(), final_proof_index.clone());
            builder
                .when_last_row()
                .when(enabled)
                .assert_eq(proof_index.clone(), final_proof_index);
        }
        // Verifier modules deliberately use dense group-local proof IDs.
        // Production replay schedules use global History IDs.  The
        // setup-fixed range proves that subtraction cannot underflow and
        // provides the unique local<->global mapping for this group.
        let input_proof_index = if let Some(schedule) = self.setup_schedule {
            proof_index.clone() - AB::Expr::from_u32(schedule.proof_index_start)
        } else {
            proof_index.clone()
        };
        self.context_bus.lookup_key(
            builder,
            DirectAirVaccContextMessageV19 {
                proof_index: input_proof_index.clone(),
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                update_index_lo: local.update_index_lo.into(),
                update_index_hi: local.update_index_hi.into(),
                shard_ordinal: local.shard_ordinal.into(),
                has_prior: local.has_prior.into(),
                key_digest: local.key_digest.map(Into::into),
                relation_digest: local.relation_digest.map(Into::into),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                prior_root: local.prior_root.map(Into::into),
                next_root: local.next_root.map(Into::into),
                previous_accumulator_digest: local.previous_accumulator_digest.map(Into::into),
                previous_checkpoint_digest: local.previous_checkpoint_digest.map(Into::into),
                next_checkpoint_digest: local.next_checkpoint_digest.map(Into::into),
            },
            enabled,
        );
        self.swirl_opening_bus.lookup_key(
            builder,
            CertifiedSwirlRawOpeningMessageV19 {
                proof_index: proof_index.clone(),
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                mode_tag: AB::Expr::from_u32(LOGUP_ONLY_MODE_TAG_V19),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                shard_ordinal: local.shard_ordinal.into(),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                root: local.fresh_root.map(Into::into),
                point_len: AB::Expr::from_usize(self.log_message_len),
                point: local.opening_point.map(|point| point.map(Into::into)),
                value: local.opening_value.map(Into::into),
            },
            enabled,
        );
        let local_vacc_input = CertifiedDirectAirVaccInputMessageV19 {
            proof_index: input_proof_index.clone(),
            protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            segment_index_lo: local.segment_index_lo.into(),
            segment_index_hi: local.segment_index_hi.into(),
            update_index_lo: local.update_index_lo.into(),
            update_index_hi: local.update_index_hi.into(),
            shard_ordinal: local.shard_ordinal.into(),
            relation_digest: local.relation_digest.map(Into::into),
            root: local.fresh_root.map(Into::into),
            alpha_len: AB::Expr::from_usize(self.log_codeword_len),
            alpha: local.fresh_alpha.map(|point| point.map(Into::into)),
            mu: local.fresh_mu.map(Into::into),
            beta_len: AB::Expr::from_usize(self.beta_len),
            beta: local.fresh_beta.map(|point| point.map(Into::into)),
            eta: local.fresh_eta.map(Into::into),
        };
        self.vacc_input_bus
            .lookup_key(builder, local_vacc_input, enabled);
        if let Some(canonical_bus) = self.canonical_vacc_input_bus {
            canonical_bus.add_key_with_lookups(
                builder,
                CertifiedDirectAirVaccInputMessageV19 {
                    proof_index: proof_index.clone(),
                    protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                    segment_index_lo: local.segment_index_lo.into(),
                    segment_index_hi: local.segment_index_hi.into(),
                    update_index_lo: local.update_index_lo.into(),
                    update_index_hi: local.update_index_hi.into(),
                    shard_ordinal: local.shard_ordinal.into(),
                    relation_digest: local.relation_digest.map(Into::into),
                    root: local.fresh_root.map(Into::into),
                    alpha_len: AB::Expr::from_usize(self.log_codeword_len),
                    alpha: local.fresh_alpha.map(|point| point.map(Into::into)),
                    mu: local.fresh_mu.map(Into::into),
                    beta_len: AB::Expr::from_usize(self.beta_len),
                    beta: local.fresh_beta.map(|point| point.map(Into::into)),
                    eta: local.fresh_eta.map(Into::into),
                },
                enabled,
            );
        }
        self.batching_claim_bus.receive(
            builder,
            NativeCertifiedBatchingClaimMessage {
                proof_idx: input_proof_index.clone(),
                claim: local.authenticated_batching_claim.map(Into::into),
            },
            enabled,
        );
        self.next_accumulator_digest_bus.receive(
            builder,
            NativeCertifiedAccumulatorDigestMessage {
                proof_idx: input_proof_index.clone(),
                digest: local.next_accumulator_digest.map(Into::into),
            },
            enabled,
        );
        for (kind, tidx, count, state) in [
            (
                0usize,
                join_u32_expr::<AB>(local.start_tidx_lo, local.start_tidx_hi),
                local.start_sample_count,
                &local.start_state,
            ),
            (
                1usize,
                join_u32_expr::<AB>(local.end_tidx_lo, local.end_tidx_hi),
                local.end_sample_count,
                &local.end_state,
            ),
        ] {
            self.checkpoint_bus.receive(
                builder,
                input_proof_index.clone(),
                CertifiedTranscriptCheckpointMessage {
                    kind: AB::Expr::from_usize(kind),
                    tidx,
                    sample_count: count.into(),
                    state: state.map(Into::into),
                },
                enabled,
            );
        }

        self.eval_hashes(builder, local, enabled);
        self.history_bus.add_key_with_lookups(
            builder,
            history_warp_message_expr::<AB>(local),
            enabled,
        );
    }
}

impl WarpReplayProducerAirV19 {
    fn eval_hashes<AB>(
        &self,
        builder: &mut AB,
        local: &WarpReplayProducerColsV19<AB::Var>,
        enabled: AB::Var,
    ) where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        let proof_index = join_u32_expr::<AB>(local.proof_index_lo, local.proof_index_hi);
        self.lookup_hash(
            builder,
            first_half::<AB>(&local.start_state),
            second_half::<AB>(&local.start_state),
            local.hash_outputs[0],
            enabled,
        );
        self.lookup_hash(
            builder,
            checkpoint_meta_expr::<AB>(
                TAG_WARP_CHECKPOINT_V19,
                0,
                local.proof_index_lo,
                local.proof_index_hi,
                local.start_tidx_lo,
                local.start_tidx_hi,
                local.start_sample_count,
            ),
            local.hash_outputs[0].map(Into::into),
            local.hash_outputs[1],
            enabled,
        );
        self.lookup_hash(
            builder,
            first_half::<AB>(&local.end_state),
            second_half::<AB>(&local.end_state),
            local.hash_outputs[2],
            enabled,
        );
        self.lookup_hash(
            builder,
            checkpoint_meta_expr::<AB>(
                TAG_WARP_CHECKPOINT_V19,
                1,
                local.proof_index_lo,
                local.proof_index_hi,
                local.end_tidx_lo,
                local.end_tidx_hi,
                local.end_sample_count,
            ),
            local.hash_outputs[2].map(Into::into),
            local.hash_outputs[3],
            enabled,
        );
        for limb in 0..DIGEST_SIZE {
            builder.when(enabled).assert_eq(
                local.transcript_start_digest[limb],
                local.hash_outputs[1][limb],
            );
            builder.when(enabled).assert_eq(
                local.transcript_end_digest[limb],
                local.hash_outputs[3][limb],
            );
        }
        let batching = tagged_extension_expr::<AB>(
            TAG_WARP_BATCHING_CLAIM_V19,
            &local.authenticated_batching_claim,
        );
        self.lookup_hash(
            builder,
            batching,
            zero_digest::<AB>(),
            local.hash_outputs[4],
            enabled,
        );

        let opening_meta = [
            AB::Expr::from_u32(TAG_RAW_OPENING_V19),
            AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            local.segment_index_lo.into(),
            local.segment_index_hi.into(),
            local.shard_ordinal.into(),
            AB::Expr::from_usize(self.log_message_len),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        self.lookup_hash(
            builder,
            opening_meta,
            local.fresh_root.map(Into::into),
            local.hash_outputs[5],
            enabled,
        );
        for index in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
            self.lookup_hash(
                builder,
                local.hash_outputs[5 + index].map(Into::into),
                extension_expr::<AB>(&local.opening_point[index]),
                local.hash_outputs[6 + index],
                enabled,
            );
        }
        self.lookup_hash(
            builder,
            local.hash_outputs[37].map(Into::into),
            extension_expr::<AB>(&local.opening_value),
            local.hash_outputs[38],
            enabled,
        );
        for limb in 0..DIGEST_SIZE {
            builder.when(enabled).assert_eq(
                local.opening_claim_digest[limb],
                local.hash_outputs[38][limb],
            );
        }

        let alpha_meta = [
            AB::Expr::from_u32(TAG_ALPHA_V19),
            AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            AB::Expr::from_usize(self.log_message_len),
            AB::Expr::from_usize(self.log_codeword_len),
            AB::Expr::from_usize(self.log_codeword_len - self.log_message_len),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        self.lookup_hash(
            builder,
            alpha_meta,
            local.fresh_root.map(Into::into),
            local.hash_outputs[39],
            enabled,
        );
        for index in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
            self.lookup_hash(
                builder,
                local.hash_outputs[39 + index].map(Into::into),
                extension_expr::<AB>(&local.fresh_alpha[index]),
                local.hash_outputs[40 + index],
                enabled,
            );
        }
        let beta_meta = [
            AB::Expr::from_u32(TAG_BETA_V19),
            AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            AB::Expr::from_usize(self.beta_len),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        self.lookup_hash(
            builder,
            beta_meta,
            local.relation_digest.map(Into::into),
            local.hash_outputs[WARP_BETA_META_HASH_V19],
            enabled,
        );
        for index in 0..MAX_FRESH_BETA_LEN_V19 {
            self.lookup_hash(
                builder,
                local.hash_outputs[WARP_BETA_META_HASH_V19 + index].map(Into::into),
                extension_expr::<AB>(&local.fresh_beta[index]),
                local.hash_outputs[WARP_BETA_HASH_START_V19 + index],
                enabled,
            );
        }
        self.lookup_hash(
            builder,
            local.fresh_root.map(Into::into),
            local.hash_outputs[WARP_ALPHA_DIGEST_HASH_V19].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19],
            enabled,
        );
        self.lookup_hash(
            builder,
            extension_expr::<AB>(&local.fresh_mu),
            local.hash_outputs[WARP_BETA_DIGEST_HASH_V19].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 1],
            enabled,
        );
        self.lookup_hash(
            builder,
            tagged_extension_expr::<AB>(TAG_FRESH_INSTANCE_V19, &local.fresh_eta),
            local.relation_digest.map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 2],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 1].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 3],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 3].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 2].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 4],
            enabled,
        );
        for limb in 0..DIGEST_SIZE {
            builder.when(enabled).assert_eq(
                local.fresh_instance_digest[limb],
                local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 4][limb],
            );
        }
        self.fresh_explicit_bus.add_key_with_lookups(
            builder,
            CertifiedFreshExplicitDigestMessageV19 {
                proof_index: proof_index.clone(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                relation_digest: local.relation_digest.map(Into::into),
                digest: local.hash_outputs[WARP_BETA_DIGEST_HASH_V19].map(Into::into),
            },
            enabled * AB::Expr::from_usize(self.fresh_explicit_lookup_count),
        );

        let context = [
            AB::Expr::from_u32(TAG_WARP_ENDPOINT_V19),
            AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            local.segment_index_lo.into(),
            local.segment_index_hi.into(),
            local.update_index_lo.into(),
            local.update_index_hi.into(),
            local.shard_ordinal.into(),
            local.has_prior.into(),
        ];
        self.lookup_hash(
            builder,
            context,
            local.relation_digest.map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 5],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.prior_root.map(Into::into),
            local.fresh_root.map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 6],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 6].map(Into::into),
            local.next_root.map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 7],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.opening_claim_digest.map(Into::into),
            local.fresh_instance_digest.map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 8],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.source_forest_root.map(Into::into),
            local.segment_openings_digest.map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 9],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.previous_accumulator_digest.map(Into::into),
            local.next_accumulator_digest.map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 10],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.previous_checkpoint_digest.map(Into::into),
            local.next_checkpoint_digest.map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 11],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.transcript_start_digest.map(Into::into),
            local.transcript_end_digest.map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 12],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 7].map(Into::into),
            local.hash_outputs[4].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 13],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 5].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 8].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 14],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 9].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 10].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 15],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 11].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 12].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 16],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 13].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 14].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 17],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 15].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 16].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 18],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 17].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 18].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 19],
            enabled,
        );
        for limb in 0..DIGEST_SIZE {
            builder.when(enabled).assert_eq(
                local.replay_endpoint_digest[limb],
                local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 19][limb],
            );
        }

        let replay_meta = [
            AB::Expr::from_u32(TAG_REPLAY_META_V19),
            AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            local.segment_index_lo.into(),
            local.segment_index_hi.into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        self.lookup_hash(
            builder,
            replay_meta,
            local.relation_digest.map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 20],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.previous_accumulator_digest.map(Into::into),
            local.next_accumulator_digest.map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 21],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.previous_checkpoint_digest.map(Into::into),
            local.next_checkpoint_digest.map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 22],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.opening_claim_digest.map(Into::into),
            local.replay_endpoint_digest.map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 23],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 20].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 21].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 24],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 22].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 23].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 25],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 24].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 25].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 26],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.key_digest.map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 26].map(Into::into),
            local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 27],
            enabled,
        );
        for limb in 0..DIGEST_SIZE {
            builder.when(enabled).assert_eq(
                local.replay_binding_digest[limb],
                local.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 27][limb],
            );
        }
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
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct LogUpOnlyProducerColsV19<T> {
    pub active: T,
    pub proof_index_lo: T,
    pub proof_index_hi: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub mode_tag: T,
    pub app_vk_digest: [T; DIGEST_SIZE],
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub verifier_endpoint: [T; D_EF],
    pub segment_sum_before: [T; D_EF],
    pub segment_sum_after: [T; D_EF],
    pub start_tidx_lo: T,
    pub start_tidx_hi: T,
    pub start_sample_count: T,
    pub start_state: [T; TRANSCRIPT_WIDTH_V19],
    pub end_tidx_lo: T,
    pub end_tidx_hi: T,
    pub end_sample_count: T,
    pub end_state: [T; TRANSCRIPT_WIDTH_V19],
    pub checkpoint_digest: [T; DIGEST_SIZE],
    pub hash_outputs: [[T; DIGEST_SIZE]; LOGUP_HASH_SLOTS_V19],
}

#[derive(Clone, Debug)]
pub struct LogUpOnlyProducerAirV19 {
    pub compress_bus: HistoryPoseidon2CompressBusV19,
    pub history_bus: LogUpOnlyHistoryBusV19,
    pub endpoint_bus: CertifiedLogUpOnlyEndpointBusV19,
    pub checkpoint_bus: CertifiedTranscriptCheckpointBus,
}

impl BaseAir<F> for LogUpOnlyProducerAirV19 {
    fn width(&self) -> usize {
        LogUpOnlyProducerColsV19::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for LogUpOnlyProducerAirV19 {}
impl PartitionedBaseAir<F> for LogUpOnlyProducerAirV19 {}

impl<AB> Air<AB> for LogUpOnlyProducerAirV19
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("LogUp producer row");
        let next_row = main.row_slice(1).expect("LogUp producer next row");
        let local: &LogUpOnlyProducerColsV19<AB::Var> = (*local_row).borrow();
        let next: &LogUpOnlyProducerColsV19<AB::Var> = (*next_row).borrow();
        let enabled = local.active;
        builder.assert_bool(enabled);
        builder
            .when_transition()
            .when(next.active)
            .assert_one(local.active);
        builder
            .when(enabled)
            .assert_eq(local.mode_tag, AB::Expr::from_u32(LOGUP_ONLY_MODE_TAG_V19));
        for limb in 0..D_EF {
            builder
                .when(enabled)
                .assert_zero(local.segment_sum_before[limb]);
            builder
                .when(enabled)
                .assert_zero(local.segment_sum_after[limb]);
        }
        let proof_index = join_u32_expr::<AB>(local.proof_index_lo, local.proof_index_hi);
        self.endpoint_bus.lookup_key(
            builder,
            CertifiedLogUpOnlyEndpointMessageV19 {
                proof_index: proof_index.clone(),
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                mode_tag: local.mode_tag.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                app_vk_digest: local.app_vk_digest.map(Into::into),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                verifier_endpoint: local.verifier_endpoint.map(Into::into),
                segment_sum_before: local.segment_sum_before.map(Into::into),
                segment_sum_after: local.segment_sum_after.map(Into::into),
            },
            enabled,
        );
        for (kind, tidx, count, state) in [
            (
                0usize,
                join_u32_expr::<AB>(local.start_tidx_lo, local.start_tidx_hi),
                local.start_sample_count,
                &local.start_state,
            ),
            (
                1usize,
                join_u32_expr::<AB>(local.end_tidx_lo, local.end_tidx_hi),
                local.end_sample_count,
                &local.end_state,
            ),
        ] {
            self.checkpoint_bus.receive(
                builder,
                proof_index.clone(),
                CertifiedTranscriptCheckpointMessage {
                    kind: AB::Expr::from_usize(kind),
                    tidx,
                    sample_count: count.into(),
                    state: state.map(Into::into),
                },
                enabled,
            );
        }
        self.eval_hashes(builder, local, enabled);
        self.history_bus.add_key_with_lookups(
            builder,
            LogUpOnlyHistoryMessageV19 {
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                mode_tag: local.mode_tag.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                app_vk_digest: local.app_vk_digest.map(Into::into),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                verifier_endpoint: local.verifier_endpoint.map(Into::into),
                checkpoint_digest: local.checkpoint_digest.map(Into::into),
            },
            enabled,
        );
        if let Some(bus) = self.history_bus.setup_pcs_source_checkpoint_bus_v3() {
            bus.send(
                builder,
                SetupPcsSourceCheckpointMessageV3 {
                    protocol_version: AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
                    transition_index: [local.proof_index_lo.into(), local.proof_index_hi.into()],
                    segment_index: [local.segment_index_lo.into(), local.segment_index_hi.into()],
                    app_vk_digest: local.app_vk_digest.map(Into::into),
                    source_forest_root: local.source_forest_root.map(Into::into),
                    segment_openings_digest: local.segment_openings_digest.map(Into::into),
                    verifier_endpoint: local.verifier_endpoint.map(Into::into),
                    end_tidx: [local.end_tidx_lo.into(), local.end_tidx_hi.into()],
                    end_sample_count: local.end_sample_count.into(),
                    end_state: local.end_state.map(Into::into),
                    logup_history_digest: local.checkpoint_digest.map(Into::into),
                },
                enabled,
            );
        }
    }
}

impl LogUpOnlyProducerAirV19 {
    fn eval_hashes<AB>(
        &self,
        builder: &mut AB,
        local: &LogUpOnlyProducerColsV19<AB::Var>,
        enabled: AB::Var,
    ) where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        self.lookup_hash(
            builder,
            first_half::<AB>(&local.start_state),
            second_half::<AB>(&local.start_state),
            local.hash_outputs[0],
            enabled,
        );
        self.lookup_hash(
            builder,
            first_half::<AB>(&local.end_state),
            second_half::<AB>(&local.end_state),
            local.hash_outputs[1],
            enabled,
        );
        self.lookup_hash(
            builder,
            checkpoint_meta_expr::<AB>(
                TAG_LOGUP_CHECKPOINT_V19,
                0,
                local.proof_index_lo,
                local.proof_index_hi,
                local.start_tidx_lo,
                local.start_tidx_hi,
                local.start_sample_count,
            ),
            local.hash_outputs[0].map(Into::into),
            local.hash_outputs[2],
            enabled,
        );
        self.lookup_hash(
            builder,
            checkpoint_meta_expr::<AB>(
                TAG_LOGUP_CHECKPOINT_V19,
                1,
                local.proof_index_lo,
                local.proof_index_hi,
                local.end_tidx_lo,
                local.end_tidx_hi,
                local.end_sample_count,
            ),
            local.hash_outputs[1].map(Into::into),
            local.hash_outputs[3],
            enabled,
        );
        self.lookup_hash(
            builder,
            tagged_extension_expr::<AB>(TAG_LOGUP_ENDPOINT_V19, &local.verifier_endpoint),
            zero_digest::<AB>(),
            local.hash_outputs[4],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.segment_openings_digest.map(Into::into),
            local.hash_outputs[4].map(Into::into),
            local.hash_outputs[5],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.app_vk_digest.map(Into::into),
            local.source_forest_root.map(Into::into),
            local.hash_outputs[6],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[2].map(Into::into),
            local.hash_outputs[3].map(Into::into),
            local.hash_outputs[7],
            enabled,
        );
        let meta = [
            AB::Expr::from_u32(TAG_LOGUP_PRODUCER_V19),
            AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            local.mode_tag.into(),
            local.segment_index_lo.into(),
            local.segment_index_hi.into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        self.lookup_hash(
            builder,
            meta,
            local.hash_outputs[6].map(Into::into),
            local.hash_outputs[8],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[5].map(Into::into),
            local.hash_outputs[7].map(Into::into),
            local.hash_outputs[9],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[8].map(Into::into),
            local.hash_outputs[9].map(Into::into),
            local.hash_outputs[10],
            enabled,
        );
        self.lookup_hash(
            builder,
            local.hash_outputs[10].map(Into::into),
            zero_digest::<AB>(),
            local.hash_outputs[11],
            enabled,
        );
        for limb in 0..DIGEST_SIZE {
            builder
                .when(enabled)
                .assert_eq(local.checkpoint_digest[limb], local.hash_outputs[11][limb]);
        }
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
}

pub fn generate_warp_replay_producer_trace_v19(
    air: &WarpReplayProducerAirV19,
    records: &[DirectAirVaccProducerRecordV19],
    fixed_height: usize,
) -> Result<PositiveProducerTraceV19<CertifiedWarpReplayMessageV19<F>>, PositiveProducerErrorV19> {
    air.validate_profile()?;
    validate_height(records.len(), fixed_height)?;
    if let Some(schedule) = air.setup_schedule {
        if records.len() != schedule.proof_count as usize {
            return Err(PositiveProducerErrorV19::RecordShape);
        }
        for (offset, record) in records.iter().enumerate() {
            let expected = schedule
                .proof_index_start
                .checked_add(offset as u32)
                .ok_or(PositiveProducerErrorV19::RecordShape)?;
            if record.proof_index != expected || record.has_prior != schedule.has_prior {
                return Err(PositiveProducerErrorV19::RecordShape);
            }
        }
    }
    let width = WarpReplayProducerColsV19::<F>::width();
    let mut values = F::zero_vec(width * fixed_height);
    let mut collector = DigestCollectorV19::default();
    let mut messages = Vec::with_capacity(records.len());
    let mut fresh_explicit_messages = Vec::with_capacity(records.len());
    for (row_index, record) in records.iter().enumerate() {
        validate_warp_record(air, record)?;
        let cols: &mut WarpReplayProducerColsV19<F> =
            values[row_index * width..(row_index + 1) * width].borrow_mut();
        fill_warp_row(air, cols, record, &mut collector);
        messages.push(history_warp_message(cols));
        fresh_explicit_messages.push(CertifiedFreshExplicitDigestMessageV19 {
            proof_index: F::from_u32(record.proof_index),
            segment_index_lo: cols.segment_index_lo,
            segment_index_hi: cols.segment_index_hi,
            relation_digest: cols.relation_digest,
            digest: cols.hash_outputs[WARP_BETA_DIGEST_HASH_V19],
        });
    }
    Ok(PositiveProducerTraceV19 {
        matrix: RowMajorMatrix::new(values, width),
        compression_inputs: collector.compression_inputs,
        messages,
        fresh_explicit_messages,
    })
}

/// Fixed local-slot table for the one batched, prior-bearing HLeaf replay
/// package. Genesis consumes the setup-authenticated valid seed; there is no
/// bootstrap producer and no neutral/fallback accumulator.
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct FixedHLeafWarpReplayPrepColsV4<T> {
    pub local_slot: T,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct FixedHLeafWarpReplayColsV4<T> {
    pub active: T,
    pub is_global_zero: T,
    pub global_transition_bits: [[T; 16]; 2],
    pub global_bit_sum_inverse: T,
    pub route: NativeFixedHLeafVaccRouteMessageV4<T>,
    pub replay: CertifiedWarpReplayMessageV19<T>,
}

#[derive(Clone, Debug)]
pub struct FixedHLeafWarpReplayAirV4 {
    pub route_bus: NativeFixedHLeafVaccRouteBusV4,
    pub replay_bus: CertifiedWarpReplayBusV19,
    pub canonical_replay_bus: CertifiedWarpReplayBusV19,
    pub transition_bus: NativeFixedHLeafVaccTransitionBusV4,
    pub seed: NativeFixedHLeafVaccSeedV4,
}

impl BaseAir<F> for FixedHLeafWarpReplayAirV4 {
    fn width(&self) -> usize {
        core::mem::size_of::<FixedHLeafWarpReplayColsV4<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = core::mem::size_of::<FixedHLeafWarpReplayPrepColsV4<u8>>();
        let mut values = F::zero_vec(width * FIXED_HLEAF_VACC_CAPACITY_V4);
        for slot in 0..FIXED_HLEAF_VACC_CAPACITY_V4 {
            let cols: &mut FixedHLeafWarpReplayPrepColsV4<F> =
                values[slot * width..(slot + 1) * width].borrow_mut();
            cols.local_slot = F::from_usize(slot);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}
impl BaseAirWithPublicValues<F> for FixedHLeafWarpReplayAirV4 {}
impl PartitionedBaseAir<F> for FixedHLeafWarpReplayAirV4 {}

impl<AB> Air<AB> for FixedHLeafWarpReplayAirV4
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let prep = builder.preprocessed();
        let prep_row = prep.row_slice(0).expect("fixed HLeaf replay prep row");
        let prep: &FixedHLeafWarpReplayPrepColsV4<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("fixed HLeaf replay row");
        let next_row = main.row_slice(1).expect("fixed HLeaf replay next row");
        let local: &FixedHLeafWarpReplayColsV4<AB::Var> = (*row).borrow();
        let next: &FixedHLeafWarpReplayColsV4<AB::Var> = (*next_row).borrow();
        let enabled = AB::Expr::from(local.active);
        builder.assert_bool(local.active);
        builder.assert_bool(local.is_global_zero);
        builder.assert_bool(local.route.has_prior);
        builder.when_first_row().assert_one(local.active);
        builder
            .when_transition()
            .assert_zero(next.active * (AB::Expr::ONE - local.active));
        for value in row.iter().skip(1) {
            builder
                .when(AB::Expr::ONE - enabled.clone())
                .assert_zero(*value);
        }

        builder
            .when(enabled.clone())
            .assert_eq(local.route.local_slot, prep.local_slot);
        builder
            .when(enabled.clone())
            .assert_eq(local.route.module_proof_idx, prep.local_slot);
        builder
            .when(enabled.clone())
            .assert_one(local.route.has_prior);
        builder.when(enabled.clone()).assert_eq(
            local.route.warp_step,
            AB::Expr::from_u32(FIXED_HLEAF_CONTINUATION_WARP_STEP_V4),
        );

        // Safely derive History genesis from the 32 constrained bits. Never
        // pack a u32 into BabyBear for a zero test: p would alias zero.
        let mut global_bit_sum = AB::Expr::ZERO;
        for (limb, bits) in [
            local.route.global_transition_index_lo,
            local.route.global_transition_index_hi,
        ]
        .into_iter()
        .zip(local.global_transition_bits)
        {
            let mut reconstructed = AB::Expr::ZERO;
            for (bit_index, bit) in bits.into_iter().enumerate() {
                builder.when(enabled.clone()).assert_bool(bit);
                reconstructed += bit * AB::Expr::from_u32(1 << bit_index);
                global_bit_sum += bit;
            }
            builder.when(enabled.clone()).assert_eq(limb, reconstructed);
        }
        builder
            .when(enabled.clone() * local.is_global_zero)
            .assert_zero(global_bit_sum.clone());
        builder
            .when(enabled.clone() * local.is_global_zero)
            .assert_zero(local.global_bit_sum_inverse);
        builder
            .when(enabled.clone() * (AB::Expr::ONE - local.is_global_zero))
            .assert_one(global_bit_sum * local.global_bit_sum_inverse);

        builder.when(enabled.clone()).assert_eq(
            local.replay.protocol_version,
            AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
        );
        for (lo, hi) in [(local.replay.segment_index_lo, local.replay.segment_index_hi)] {
            builder.when(enabled.clone()).assert_eq(lo, prep.local_slot);
            builder.when(enabled.clone()).assert_zero(hi);
        }
        builder
            .when(enabled.clone())
            .assert_eq(local.replay.update_index_lo, local.route.warp_step);
        builder
            .when(enabled.clone())
            .assert_zero(local.replay.update_index_hi);
        builder
            .when(enabled.clone())
            .assert_zero(local.replay.shard_ordinal);
        for limb in 0..DIGEST_SIZE {
            builder.when(enabled.clone()).assert_eq(
                local.replay.key_digest[limb],
                AB::Expr::from(self.seed.key_digest[limb]),
            );
            builder.when(enabled.clone()).assert_eq(
                local.replay.relation_digest[limb],
                AB::Expr::from(self.seed.relation_digest[limb]),
            );
            let genesis = enabled.clone() * local.is_global_zero;
            builder.when(genesis.clone()).assert_eq(
                local.replay.prior_root[limb],
                AB::Expr::from(self.seed.root[limb]),
            );
            builder.when(genesis.clone()).assert_eq(
                local.replay.previous_accumulator_digest[limb],
                AB::Expr::from(self.seed.accumulator_digest[limb]),
            );
        }

        self.replay_bus
            .lookup_key(builder, local.replay.clone(), enabled.clone());
        self.route_bus
            .lookup_key(builder, local.route.clone(), enabled.clone());
        self.canonical_replay_bus.add_key_with_lookups(
            builder,
            local.replay.clone(),
            enabled.clone(),
        );
        self.transition_bus.add_key_with_lookups(
            builder,
            NativeFixedHLeafVaccTransitionMessageV4 {
                route: NativeFixedHLeafVaccRouteMessageV4 {
                    local_slot: local.route.local_slot.into(),
                    node_index_lo: local.route.node_index_lo.into(),
                    node_index_hi: local.route.node_index_hi.into(),
                    global_transition_index_lo: local.route.global_transition_index_lo.into(),
                    global_transition_index_hi: local.route.global_transition_index_hi.into(),
                    module_proof_idx: local.route.module_proof_idx.into(),
                    has_prior: local.route.has_prior.into(),
                    warp_step: local.route.warp_step.into(),
                },
                key_digest: local.replay.key_digest.map(Into::into),
                relation_digest: local.replay.relation_digest.map(Into::into),
                source_forest_root: local.replay.source_forest_root.map(Into::into),
                segment_openings_digest: local.replay.segment_openings_digest.map(Into::into),
                prior_root: local.replay.prior_root.map(Into::into),
                fresh_root: local.replay.fresh_root.map(Into::into),
                output_root: local.replay.next_root.map(Into::into),
                previous_accumulator_digest: local
                    .replay
                    .previous_accumulator_digest
                    .map(Into::into),
                output_accumulator_digest: local.replay.next_accumulator_digest.map(Into::into),
                previous_checkpoint_digest: local.replay.previous_checkpoint_digest.map(Into::into),
                output_checkpoint_digest: local.replay.next_checkpoint_digest.map(Into::into),
                replay_endpoint_digest: local.replay.replay_endpoint_digest.map(Into::into),
                replay_binding_digest: local.replay.replay_binding_digest.map(Into::into),
            },
            enabled.clone(),
        );

        let chained = next.active;
        for limb in 0..DIGEST_SIZE {
            builder
                .when_transition()
                .when(chained)
                .assert_eq(next.replay.prior_root[limb], local.replay.next_root[limb]);
            builder.when_transition().when(chained).assert_eq(
                next.replay.previous_accumulator_digest[limb],
                local.replay.next_accumulator_digest[limb],
            );
        }
        // Transcript checkpoints are local to one independently domain-bound
        // source-plus-VACC transcript. The next transition starts from the
        // canonical transcript initialization and binds its own manifest; it
        // does not resume the preceding transition's sponge. The replay and
        // transition buses still authenticate both checkpoint endpoints for
        // each slot. Only the accumulator instance and application boundary
        // are cross-transition state.
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedHLeafWarpReplayErrorV4 {
    Empty,
    Capacity,
    Count,
    Route(usize),
    Replay(usize),
    Seed(usize),
    Chain(usize),
}

pub fn generate_fixed_hleaf_warp_replay_trace_v4(
    air: &FixedHLeafWarpReplayAirV4,
    routes: &[NativeFixedHLeafVaccRouteV4],
    replays: &[CertifiedWarpReplayMessageV19<F>],
) -> Result<RowMajorMatrix<F>, FixedHLeafWarpReplayErrorV4> {
    if routes.is_empty() {
        return Err(FixedHLeafWarpReplayErrorV4::Empty);
    }
    if routes.len() > FIXED_HLEAF_VACC_CAPACITY_V4 {
        return Err(FixedHLeafWarpReplayErrorV4::Capacity);
    }
    if routes.len() != replays.len() {
        return Err(FixedHLeafWarpReplayErrorV4::Count);
    }
    let width = core::mem::size_of::<FixedHLeafWarpReplayColsV4<u8>>();
    let mut values = F::zero_vec(width * FIXED_HLEAF_VACC_CAPACITY_V4);
    for (slot, (route, replay)) in routes.iter().zip(replays).enumerate() {
        route
            .validate()
            .map_err(|_| FixedHLeafWarpReplayErrorV4::Route(slot))?;
        if route.local_slot as usize != slot
            || route.module_proof_idx != slot as u32
            || !route.has_prior
            || route.warp_step != FIXED_HLEAF_CONTINUATION_WARP_STEP_V4
            || replay.protocol_version != F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19)
            || replay.segment_index_lo != F::from_usize(slot)
            || replay.segment_index_hi != F::ZERO
            || replay.update_index_lo != F::from_u32(route.warp_step)
            || replay.update_index_hi != F::ZERO
            || replay.shard_ordinal != F::ZERO
        {
            return Err(FixedHLeafWarpReplayErrorV4::Replay(slot));
        }
        if replay.key_digest != air.seed.key_digest
            || replay.relation_digest != air.seed.relation_digest
        {
            return Err(FixedHLeafWarpReplayErrorV4::Seed(slot));
        }
        if route.global_transition_index == 0
            && (replay.prior_root != air.seed.root
                || replay.previous_accumulator_digest != air.seed.accumulator_digest)
        {
            return Err(FixedHLeafWarpReplayErrorV4::Seed(slot));
        }
        if slot != 0 {
            let prior = &replays[slot - 1];
            if replay.prior_root != prior.next_root
                || replay.previous_accumulator_digest != prior.next_accumulator_digest
            {
                return Err(FixedHLeafWarpReplayErrorV4::Chain(slot));
            }
        }
        let cols: &mut FixedHLeafWarpReplayColsV4<F> =
            values[slot * width..(slot + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.is_global_zero = F::from_bool(route.global_transition_index == 0);
        for (limb_index, limb) in [
            route.global_transition_index as u16,
            (route.global_transition_index >> 16) as u16,
        ]
        .into_iter()
        .enumerate()
        {
            for bit in 0..16 {
                cols.global_transition_bits[limb_index][bit] =
                    F::from_bool(((limb >> bit) & 1) == 1);
            }
        }
        let bit_sum = route.global_transition_index.count_ones();
        cols.global_bit_sum_inverse = if bit_sum == 0 {
            F::ZERO
        } else {
            F::from_u32(bit_sum).inverse()
        };
        cols.route = NativeFixedHLeafVaccRouteMessageV4 {
            local_slot: F::from_u32(route.local_slot),
            node_index_lo: F::from_u32(route.node_index & 0xffff),
            node_index_hi: F::from_u32(route.node_index >> 16),
            global_transition_index_lo: F::from_u32(route.global_transition_index & 0xffff),
            global_transition_index_hi: F::from_u32(route.global_transition_index >> 16),
            module_proof_idx: F::from_u32(route.module_proof_idx),
            has_prior: F::from_bool(route.has_prior),
            warp_step: F::from_u32(route.warp_step),
        };
        cols.replay = replay.clone();
    }
    Ok(RowMajorMatrix::new(values, width))
}

/// One fixed-capacity producer for the one batched prior-bearing verifier.
/// Runtime occupancy is a nonempty prefix of local proof IDs 0..3. Inactive
/// suffix rows are canonical zero; no omitted verifier package is simulated.
pub struct FixedHLeafWarpReplayProducerAirV4 {
    pub inner: WarpReplayProducerAirV19,
}

impl BaseAir<F> for FixedHLeafWarpReplayProducerAirV4 {
    fn width(&self) -> usize {
        self.inner.width()
    }
}
impl BaseAirWithPublicValues<F> for FixedHLeafWarpReplayProducerAirV4 {}
impl PartitionedBaseAir<F> for FixedHLeafWarpReplayProducerAirV4 {}

impl<AB> Air<AB> for FixedHLeafWarpReplayProducerAirV4
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(
            self.inner.setup_schedule.is_none(),
            "fixed HLeaf replay route uses its own activation schedule"
        );
        self.inner.eval(builder);
        let main = builder.main();
        let row = main.row_slice(0).expect("fixed HLeaf replay producer row");
        let next_row = main
            .row_slice(1)
            .expect("fixed HLeaf replay producer next row");
        let local: &WarpReplayProducerColsV19<AB::Var> = (*row).borrow();
        let next: &WarpReplayProducerColsV19<AB::Var> = (*next_row).borrow();
        builder.when_first_row().assert_one(local.active);
        builder.when(local.active).assert_one(local.has_prior);
        builder.when(local.active).assert_eq(
            local.update_index_lo,
            AB::Expr::from_u32(FIXED_HLEAF_CONTINUATION_WARP_STEP_V4),
        );
        builder
            .when(local.active)
            .assert_zero(local.update_index_hi);
        builder.when_first_row().assert_zero(local.proof_index_lo);
        builder.when_first_row().assert_zero(local.proof_index_hi);
        builder
            .when(local.active)
            .assert_eq(local.segment_index_lo, local.proof_index_lo);
        builder
            .when(local.active)
            .assert_eq(local.segment_index_hi, local.proof_index_hi);
        builder
            .when_transition()
            .when(next.active)
            .assert_eq(next.proof_index_lo, local.proof_index_lo + AB::Expr::ONE);
        builder
            .when_transition()
            .when(next.active)
            .assert_zero(next.proof_index_hi);
    }
}

pub fn generate_fixed_hleaf_warp_replay_producer_trace_v4(
    air: &FixedHLeafWarpReplayProducerAirV4,
    records: &[DirectAirVaccProducerRecordV19],
) -> Result<PositiveProducerTraceV19<CertifiedWarpReplayMessageV19<F>>, PositiveProducerErrorV19> {
    if air.inner.setup_schedule.is_some()
        || records.is_empty()
        || records.len() > FIXED_HLEAF_VACC_CAPACITY_V4
    {
        return Err(PositiveProducerErrorV19::InvalidProfile);
    }
    for (slot, record) in records.iter().enumerate() {
        if record.proof_index != slot as u32
            || record.segment_index != slot as u32
            || record.update_index != FIXED_HLEAF_CONTINUATION_WARP_STEP_V4
            || !record.has_prior
        {
            return Err(PositiveProducerErrorV19::RecordShape);
        }
    }
    generate_warp_replay_producer_trace_v19(&air.inner, records, FIXED_HLEAF_VACC_CAPACITY_V4)
}

/// Reconstruct the exact certified fresh-input message consumed by
/// [`WarpReplayProducerAirV19`] from one canonical producer record.
///
/// This helper is deliberately profile-bound by `air`: it validates all
/// variable-length coordinates and zero-pads the fixed bus message exactly as
/// the producer AIR does.  Outer compositions should use this value for their
/// bridge witness instead of independently re-encoding beta or its root.
pub fn certified_direct_air_vacc_input_message_v19(
    air: &WarpReplayProducerAirV19,
    record: &DirectAirVaccProducerRecordV19,
) -> Result<CertifiedDirectAirVaccInputMessageV19<F>, PositiveProducerErrorV19> {
    air.validate_profile()?;
    validate_warp_record(air, record)?;
    let mut alpha = [[F::ZERO; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19];
    alpha[..record.fresh_alpha.len()].copy_from_slice(&record.fresh_alpha);
    let mut beta = [[F::ZERO; D_EF]; MAX_FRESH_BETA_LEN_V19];
    beta[..record.fresh_beta.len()].copy_from_slice(&record.fresh_beta);
    let (segment_index_lo, segment_index_hi) = split_u32(record.segment_index);
    let (update_index_lo, update_index_hi) = split_u32(record.update_index);
    Ok(CertifiedDirectAirVaccInputMessageV19 {
        proof_index: F::from_u32(record.proof_index),
        protocol_version: F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
        segment_index_lo: F::from_u16(segment_index_lo),
        segment_index_hi: F::from_u16(segment_index_hi),
        update_index_lo: F::from_u16(update_index_lo),
        update_index_hi: F::from_u16(update_index_hi),
        shard_ordinal: F::from_u16(record.shard_ordinal),
        relation_digest: record.relation_digest,
        root: record.fresh_root,
        alpha_len: F::from_usize(air.log_codeword_len),
        alpha,
        mu: record.fresh_mu,
        beta_len: F::from_usize(air.beta_len),
        beta,
        eta: record.fresh_eta,
    })
}

pub fn generate_logup_only_producer_trace_v19(
    records: &[LogUpOnlyProducerRecordV19],
    fixed_height: usize,
) -> Result<PositiveProducerTraceV19<LogUpOnlyHistoryMessageV19<F>>, PositiveProducerErrorV19> {
    validate_height(records.len(), fixed_height)?;
    let width = LogUpOnlyProducerColsV19::<F>::width();
    let mut values = F::zero_vec(width * fixed_height);
    let mut collector = DigestCollectorV19::default();
    let mut messages = Vec::with_capacity(records.len());
    for (row_index, record) in records.iter().enumerate() {
        if record.segment_sum_before != [F::ZERO; D_EF]
            || record.segment_sum_after != [F::ZERO; D_EF]
        {
            return Err(PositiveProducerErrorV19::NonZeroLogUpBoundary);
        }
        let cols: &mut LogUpOnlyProducerColsV19<F> =
            values[row_index * width..(row_index + 1) * width].borrow_mut();
        fill_logup_row(cols, record, &mut collector);
        messages.push(history_logup_message(cols));
    }
    Ok(PositiveProducerTraceV19 {
        matrix: RowMajorMatrix::new(values, width),
        compression_inputs: collector.compression_inputs,
        messages,
        fresh_explicit_messages: Vec::new(),
    })
}

fn validate_height(count: usize, height: usize) -> Result<(), PositiveProducerErrorV19> {
    if height == 0 || !height.is_power_of_two() {
        return Err(PositiveProducerErrorV19::InvalidFixedHeight);
    }
    if count > height {
        return Err(PositiveProducerErrorV19::CapacityExceeded);
    }
    Ok(())
}

fn validate_warp_record(
    air: &WarpReplayProducerAirV19,
    record: &DirectAirVaccProducerRecordV19,
) -> Result<(), PositiveProducerErrorV19> {
    if record.opening_point.len() != air.log_message_len
        || record.fresh_alpha.len() != air.log_codeword_len
        || record.fresh_beta.len() != air.beta_len
        || record.fresh_mu != record.opening_value
        || record.fresh_eta != [F::ZERO; D_EF]
    {
        return Err(PositiveProducerErrorV19::RecordShape);
    }
    if record.fresh_alpha[..air.log_message_len] != record.opening_point
        || record.fresh_alpha[air.log_message_len..]
            .iter()
            .any(|value| *value != [F::ZERO; D_EF])
    {
        return Err(PositiveProducerErrorV19::RecordShape);
    }
    if !record.has_prior && record.prior_root != [F::ZERO; DIGEST_SIZE] {
        return Err(PositiveProducerErrorV19::InvalidPriorRoot);
    }
    Ok(())
}

fn fill_warp_row(
    air: &WarpReplayProducerAirV19,
    cols: &mut WarpReplayProducerColsV19<F>,
    record: &DirectAirVaccProducerRecordV19,
    collector: &mut DigestCollectorV19,
) {
    cols.active = F::ONE;
    fill_u32(
        record.proof_index,
        &mut cols.proof_index_lo,
        &mut cols.proof_index_hi,
        &mut cols.proof_index_bits,
    );
    fill_u32(
        record.segment_index,
        &mut cols.segment_index_lo,
        &mut cols.segment_index_hi,
        &mut cols.segment_index_bits,
    );
    fill_u32(
        record.update_index,
        &mut cols.update_index_lo,
        &mut cols.update_index_hi,
        &mut cols.update_index_bits,
    );
    cols.shard_ordinal = F::from_u16(record.shard_ordinal);
    fill_bits(&mut cols.shard_ordinal_bits, record.shard_ordinal);
    cols.has_prior = F::from_bool(record.has_prior);
    cols.key_digest = record.key_digest;
    cols.relation_digest = record.relation_digest;
    cols.source_forest_root = record.source_forest_root;
    cols.segment_openings_digest = record.segment_openings_digest;
    cols.prior_root = record.prior_root;
    cols.fresh_root = record.fresh_root;
    cols.next_root = record.next_root;
    for (dst, src) in cols.opening_point.iter_mut().zip(&record.opening_point) {
        *dst = *src;
    }
    cols.opening_value = record.opening_value;
    for (dst, src) in cols.fresh_alpha.iter_mut().zip(&record.fresh_alpha) {
        *dst = *src;
    }
    cols.fresh_mu = record.fresh_mu;
    for (dst, src) in cols.fresh_beta.iter_mut().zip(&record.fresh_beta) {
        *dst = *src;
    }
    cols.fresh_eta = record.fresh_eta;
    cols.previous_accumulator_digest = record.previous_accumulator_digest;
    cols.next_accumulator_digest = record.next_accumulator_digest;
    cols.previous_checkpoint_digest = record.previous_checkpoint_digest;
    cols.next_checkpoint_digest = record.next_checkpoint_digest;
    cols.authenticated_batching_claim = record.authenticated_batching_claim;
    fill_checkpoint(
        &record.start_checkpoint,
        &mut cols.start_tidx_lo,
        &mut cols.start_tidx_hi,
        &mut cols.start_sample_count,
        &mut cols.start_state,
    );
    fill_checkpoint(
        &record.end_checkpoint,
        &mut cols.end_tidx_lo,
        &mut cols.end_tidx_hi,
        &mut cols.end_sample_count,
        &mut cols.end_state,
    );
    fill_warp_hashes(air, cols, collector);
}

fn fill_warp_hashes(
    air: &WarpReplayProducerAirV19,
    cols: &mut WarpReplayProducerColsV19<F>,
    collector: &mut DigestCollectorV19,
) {
    cols.hash_outputs[0] = collector.compress(
        cols.start_state[..DIGEST_SIZE].try_into().unwrap(),
        cols.start_state[DIGEST_SIZE..].try_into().unwrap(),
    );
    cols.hash_outputs[1] = collector.compress(
        checkpoint_meta(
            TAG_WARP_CHECKPOINT_V19,
            0,
            join_u32(cols.proof_index_lo, cols.proof_index_hi),
            join_u32(cols.start_tidx_lo, cols.start_tidx_hi),
            cols.start_sample_count,
        ),
        cols.hash_outputs[0],
    );
    cols.transcript_start_digest = cols.hash_outputs[1];
    cols.hash_outputs[2] = collector.compress(
        cols.end_state[..DIGEST_SIZE].try_into().unwrap(),
        cols.end_state[DIGEST_SIZE..].try_into().unwrap(),
    );
    cols.hash_outputs[3] = collector.compress(
        checkpoint_meta(
            TAG_WARP_CHECKPOINT_V19,
            1,
            join_u32(cols.proof_index_lo, cols.proof_index_hi),
            join_u32(cols.end_tidx_lo, cols.end_tidx_hi),
            cols.end_sample_count,
        ),
        cols.hash_outputs[2],
    );
    cols.transcript_end_digest = cols.hash_outputs[3];
    cols.hash_outputs[4] = collector.compress(
        tagged_extension(
            TAG_WARP_BATCHING_CLAIM_V19,
            cols.authenticated_batching_claim,
        ),
        [F::ZERO; DIGEST_SIZE],
    );
    cols.hash_outputs[5] = collector.compress(
        scalar_digest(&[
            TAG_RAW_OPENING_V19,
            NATIVE_WARP_HISTORY_PROTOCOL_V19,
            cols.segment_index_lo.as_canonical_u32(),
            cols.segment_index_hi.as_canonical_u32(),
            cols.shard_ordinal.as_canonical_u32(),
            air.log_message_len as u32,
        ]),
        cols.fresh_root,
    );
    for index in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
        cols.hash_outputs[6 + index] = collector.compress(
            cols.hash_outputs[5 + index],
            extension_digest(cols.opening_point[index]),
        );
    }
    cols.hash_outputs[38] =
        collector.compress(cols.hash_outputs[37], extension_digest(cols.opening_value));
    cols.opening_claim_digest = cols.hash_outputs[38];
    cols.hash_outputs[39] = collector.compress(
        scalar_digest(&[
            TAG_ALPHA_V19,
            NATIVE_WARP_HISTORY_PROTOCOL_V19,
            air.log_message_len as u32,
            air.log_codeword_len as u32,
            (air.log_codeword_len - air.log_message_len) as u32,
        ]),
        cols.fresh_root,
    );
    for index in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
        cols.hash_outputs[40 + index] = collector.compress(
            cols.hash_outputs[39 + index],
            extension_digest(cols.fresh_alpha[index]),
        );
    }
    cols.hash_outputs[WARP_BETA_META_HASH_V19] = collector.compress(
        scalar_digest(&[
            TAG_BETA_V19,
            NATIVE_WARP_HISTORY_PROTOCOL_V19,
            air.beta_len as u32,
        ]),
        cols.relation_digest,
    );
    for index in 0..MAX_FRESH_BETA_LEN_V19 {
        cols.hash_outputs[WARP_BETA_HASH_START_V19 + index] = collector.compress(
            cols.hash_outputs[WARP_BETA_META_HASH_V19 + index],
            extension_digest(cols.fresh_beta[index]),
        );
    }
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19] = collector.compress(
        cols.fresh_root,
        cols.hash_outputs[WARP_ALPHA_DIGEST_HASH_V19],
    );
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 1] = collector.compress(
        extension_digest(cols.fresh_mu),
        cols.hash_outputs[WARP_BETA_DIGEST_HASH_V19],
    );
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 2] = collector.compress(
        tagged_extension(TAG_FRESH_INSTANCE_V19, cols.fresh_eta),
        cols.relation_digest,
    );
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 3] = collector.compress(
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19],
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 1],
    );
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 4] = collector.compress(
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 3],
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 2],
    );
    cols.fresh_instance_digest = cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 4];
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 5] = collector.compress(
        scalar_digest(&[
            TAG_WARP_ENDPOINT_V19,
            NATIVE_WARP_HISTORY_PROTOCOL_V19,
            cols.segment_index_lo.as_canonical_u32(),
            cols.segment_index_hi.as_canonical_u32(),
            cols.update_index_lo.as_canonical_u32(),
            cols.update_index_hi.as_canonical_u32(),
            cols.shard_ordinal.as_canonical_u32(),
            cols.has_prior.as_canonical_u32(),
        ]),
        cols.relation_digest,
    );
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 6] =
        collector.compress(cols.prior_root, cols.fresh_root);
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 7] = collector.compress(
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 6],
        cols.next_root,
    );
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 8] =
        collector.compress(cols.opening_claim_digest, cols.fresh_instance_digest);
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 9] =
        collector.compress(cols.source_forest_root, cols.segment_openings_digest);
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 10] = collector.compress(
        cols.previous_accumulator_digest,
        cols.next_accumulator_digest,
    );
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 11] =
        collector.compress(cols.previous_checkpoint_digest, cols.next_checkpoint_digest);
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 12] =
        collector.compress(cols.transcript_start_digest, cols.transcript_end_digest);
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 13] = collector.compress(
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 7],
        cols.hash_outputs[4],
    );
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 14] = collector.compress(
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 5],
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 8],
    );
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 15] = collector.compress(
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 9],
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 10],
    );
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 16] = collector.compress(
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 11],
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 12],
    );
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 17] = collector.compress(
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 13],
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 14],
    );
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 18] = collector.compress(
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 15],
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 16],
    );
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 19] = collector.compress(
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 17],
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 18],
    );
    cols.replay_endpoint_digest = cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 19];
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 20] = collector.compress(
        scalar_digest(&[
            TAG_REPLAY_META_V19,
            NATIVE_WARP_HISTORY_PROTOCOL_V19,
            cols.segment_index_lo.as_canonical_u32(),
            cols.segment_index_hi.as_canonical_u32(),
        ]),
        cols.relation_digest,
    );
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 21] = collector.compress(
        cols.previous_accumulator_digest,
        cols.next_accumulator_digest,
    );
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 22] =
        collector.compress(cols.previous_checkpoint_digest, cols.next_checkpoint_digest);
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 23] =
        collector.compress(cols.opening_claim_digest, cols.replay_endpoint_digest);
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 24] = collector.compress(
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 20],
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 21],
    );
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 25] = collector.compress(
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 22],
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 23],
    );
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 26] = collector.compress(
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 24],
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 25],
    );
    cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 27] = collector.compress(
        cols.key_digest,
        cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 26],
    );
    cols.replay_binding_digest = cols.hash_outputs[WARP_POST_BETA_HASH_START_V19 + 27];
}

fn fill_logup_row(
    cols: &mut LogUpOnlyProducerColsV19<F>,
    record: &LogUpOnlyProducerRecordV19,
    collector: &mut DigestCollectorV19,
) {
    cols.active = F::ONE;
    let (proof_lo, proof_hi) = split_u32(record.proof_index);
    cols.proof_index_lo = F::from_u16(proof_lo);
    cols.proof_index_hi = F::from_u16(proof_hi);
    let (segment_lo, segment_hi) = split_u32(record.segment_index);
    cols.segment_index_lo = F::from_u16(segment_lo);
    cols.segment_index_hi = F::from_u16(segment_hi);
    cols.mode_tag = F::from_u32(LOGUP_ONLY_MODE_TAG_V19);
    cols.app_vk_digest = record.app_vk_digest;
    cols.source_forest_root = record.source_forest_root;
    cols.segment_openings_digest = record.segment_openings_digest;
    cols.verifier_endpoint = record.verifier_endpoint;
    cols.segment_sum_before = record.segment_sum_before;
    cols.segment_sum_after = record.segment_sum_after;
    fill_checkpoint(
        &record.start_checkpoint,
        &mut cols.start_tidx_lo,
        &mut cols.start_tidx_hi,
        &mut cols.start_sample_count,
        &mut cols.start_state,
    );
    fill_checkpoint(
        &record.end_checkpoint,
        &mut cols.end_tidx_lo,
        &mut cols.end_tidx_hi,
        &mut cols.end_sample_count,
        &mut cols.end_state,
    );
    cols.hash_outputs[0] = collector.compress(
        cols.start_state[..DIGEST_SIZE].try_into().unwrap(),
        cols.start_state[DIGEST_SIZE..].try_into().unwrap(),
    );
    cols.hash_outputs[1] = collector.compress(
        cols.end_state[..DIGEST_SIZE].try_into().unwrap(),
        cols.end_state[DIGEST_SIZE..].try_into().unwrap(),
    );
    cols.hash_outputs[2] = collector.compress(
        checkpoint_meta(
            TAG_LOGUP_CHECKPOINT_V19,
            0,
            record.proof_index,
            record.start_checkpoint.operation_index,
            F::from_u8(record.start_checkpoint.sample_count),
        ),
        cols.hash_outputs[0],
    );
    cols.hash_outputs[3] = collector.compress(
        checkpoint_meta(
            TAG_LOGUP_CHECKPOINT_V19,
            1,
            record.proof_index,
            record.end_checkpoint.operation_index,
            F::from_u8(record.end_checkpoint.sample_count),
        ),
        cols.hash_outputs[1],
    );
    cols.hash_outputs[4] = collector.compress(
        tagged_extension(TAG_LOGUP_ENDPOINT_V19, record.verifier_endpoint),
        [F::ZERO; DIGEST_SIZE],
    );
    cols.hash_outputs[5] = collector.compress(record.segment_openings_digest, cols.hash_outputs[4]);
    cols.hash_outputs[6] = collector.compress(record.app_vk_digest, record.source_forest_root);
    cols.hash_outputs[7] = collector.compress(cols.hash_outputs[2], cols.hash_outputs[3]);
    cols.hash_outputs[8] = collector.compress(
        scalar_digest(&[
            TAG_LOGUP_PRODUCER_V19,
            NATIVE_WARP_HISTORY_PROTOCOL_V19,
            LOGUP_ONLY_MODE_TAG_V19,
            u32::from(segment_lo),
            u32::from(segment_hi),
        ]),
        cols.hash_outputs[6],
    );
    cols.hash_outputs[9] = collector.compress(cols.hash_outputs[5], cols.hash_outputs[7]);
    cols.hash_outputs[10] = collector.compress(cols.hash_outputs[8], cols.hash_outputs[9]);
    cols.hash_outputs[11] = collector.compress(cols.hash_outputs[10], [F::ZERO; DIGEST_SIZE]);
    cols.checkpoint_digest = cols.hash_outputs[11];
}

fn history_warp_message(cols: &WarpReplayProducerColsV19<F>) -> CertifiedWarpReplayMessageV19<F> {
    CertifiedWarpReplayMessageV19 {
        protocol_version: F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
        segment_index_lo: cols.segment_index_lo,
        segment_index_hi: cols.segment_index_hi,
        update_index_lo: cols.update_index_lo,
        update_index_hi: cols.update_index_hi,
        shard_ordinal: cols.shard_ordinal,
        source_forest_root: cols.source_forest_root,
        key_digest: cols.key_digest,
        relation_digest: cols.relation_digest,
        opening_claim_digest: cols.opening_claim_digest,
        fresh_instance_digest: cols.fresh_instance_digest,
        segment_openings_digest: cols.segment_openings_digest,
        prior_root: cols.prior_root,
        fresh_root: cols.fresh_root,
        next_root: cols.next_root,
        previous_accumulator_digest: cols.previous_accumulator_digest,
        next_accumulator_digest: cols.next_accumulator_digest,
        authenticated_batching_claim: cols.authenticated_batching_claim,
        previous_checkpoint_digest: cols.previous_checkpoint_digest,
        next_checkpoint_digest: cols.next_checkpoint_digest,
        replay_endpoint_digest: cols.replay_endpoint_digest,
        replay_binding_digest: cols.replay_binding_digest,
    }
}

fn history_warp_message_expr<AB: AirBuilder<F = F>>(
    cols: &WarpReplayProducerColsV19<AB::Var>,
) -> CertifiedWarpReplayMessageV19<AB::Expr>
where
    AB::Var: Copy,
{
    CertifiedWarpReplayMessageV19 {
        protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
        segment_index_lo: cols.segment_index_lo.into(),
        segment_index_hi: cols.segment_index_hi.into(),
        update_index_lo: cols.update_index_lo.into(),
        update_index_hi: cols.update_index_hi.into(),
        shard_ordinal: cols.shard_ordinal.into(),
        source_forest_root: cols.source_forest_root.map(Into::into),
        key_digest: cols.key_digest.map(Into::into),
        relation_digest: cols.relation_digest.map(Into::into),
        opening_claim_digest: cols.opening_claim_digest.map(Into::into),
        fresh_instance_digest: cols.fresh_instance_digest.map(Into::into),
        segment_openings_digest: cols.segment_openings_digest.map(Into::into),
        prior_root: cols.prior_root.map(Into::into),
        fresh_root: cols.fresh_root.map(Into::into),
        next_root: cols.next_root.map(Into::into),
        previous_accumulator_digest: cols.previous_accumulator_digest.map(Into::into),
        next_accumulator_digest: cols.next_accumulator_digest.map(Into::into),
        authenticated_batching_claim: cols.authenticated_batching_claim.map(Into::into),
        previous_checkpoint_digest: cols.previous_checkpoint_digest.map(Into::into),
        next_checkpoint_digest: cols.next_checkpoint_digest.map(Into::into),
        replay_endpoint_digest: cols.replay_endpoint_digest.map(Into::into),
        replay_binding_digest: cols.replay_binding_digest.map(Into::into),
    }
}

fn history_logup_message(cols: &LogUpOnlyProducerColsV19<F>) -> LogUpOnlyHistoryMessageV19<F> {
    LogUpOnlyHistoryMessageV19 {
        protocol_version: F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
        mode_tag: cols.mode_tag,
        segment_index_lo: cols.segment_index_lo,
        segment_index_hi: cols.segment_index_hi,
        app_vk_digest: cols.app_vk_digest,
        source_forest_root: cols.source_forest_root,
        segment_openings_digest: cols.segment_openings_digest,
        verifier_endpoint: cols.verifier_endpoint,
        checkpoint_digest: cols.checkpoint_digest,
    }
}

fn first_half<AB: AirBuilder<F = F>>(
    state: &[AB::Var; TRANSCRIPT_WIDTH_V19],
) -> [AB::Expr; DIGEST_SIZE]
where
    AB::Var: Copy,
{
    core::array::from_fn(|index| state[index].into())
}
fn second_half<AB: AirBuilder<F = F>>(
    state: &[AB::Var; TRANSCRIPT_WIDTH_V19],
) -> [AB::Expr; DIGEST_SIZE]
where
    AB::Var: Copy,
{
    core::array::from_fn(|index| state[DIGEST_SIZE + index].into())
}
fn extension_expr<AB: AirBuilder<F = F>>(value: &[AB::Var; D_EF]) -> [AB::Expr; DIGEST_SIZE]
where
    AB::Var: Copy,
{
    core::array::from_fn(|index| {
        if index < D_EF {
            value[index].into()
        } else {
            AB::Expr::ZERO
        }
    })
}
fn tagged_extension_expr<AB: AirBuilder<F = F>>(
    tag: u32,
    value: &[AB::Var; D_EF],
) -> [AB::Expr; DIGEST_SIZE]
where
    AB::Var: Copy,
{
    core::array::from_fn(|index| match index {
        0 => AB::Expr::from_u32(tag),
        1..=D_EF => value[index - 1].into(),
        _ => AB::Expr::ZERO,
    })
}
fn zero_digest<AB: AirBuilder<F = F>>() -> [AB::Expr; DIGEST_SIZE] {
    core::array::from_fn(|_| AB::Expr::ZERO)
}
fn checkpoint_meta_expr<AB: AirBuilder<F = F>>(
    tag: u32,
    kind: u32,
    proof_lo: AB::Var,
    proof_hi: AB::Var,
    tidx_lo: AB::Var,
    tidx_hi: AB::Var,
    sample_count: AB::Var,
) -> [AB::Expr; DIGEST_SIZE]
where
    AB::Var: Copy,
{
    [
        AB::Expr::from_u32(tag),
        AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
        AB::Expr::from_u32(kind),
        proof_lo.into(),
        proof_hi.into(),
        tidx_lo.into(),
        tidx_hi.into(),
        sample_count.into(),
    ]
}
fn join_u32_expr<AB: AirBuilder<F = F>>(lo: AB::Var, hi: AB::Var) -> AB::Expr
where
    AB::Var: Copy,
{
    AB::Expr::from(lo) + AB::Expr::from(hi) * AB::Expr::from_u32(1 << LIMB_BITS)
}
fn bits_expr<AB: AirBuilder<F = F>>(bits: &[AB::Var; LIMB_BITS]) -> AB::Expr
where
    AB::Var: Copy,
{
    bits.iter()
        .enumerate()
        .fold(AB::Expr::ZERO, |sum, (index, bit)| {
            sum + *bit * AB::Expr::from_u32(1u32 << index)
        })
}
fn assert_bits<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    bits: &[AB::Var; LIMB_BITS],
    enabled: AB::Var,
) where
    AB::Var: Copy,
{
    for bit in bits {
        builder.when(enabled).assert_bool(*bit);
    }
}
fn assert_u32<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    lo: AB::Var,
    hi: AB::Var,
    bits: &[[AB::Var; LIMB_BITS]; 2],
    enabled: AB::Var,
) where
    AB::Var: Copy,
{
    for (value, limb) in [lo, hi].into_iter().zip(bits) {
        assert_bits(builder, limb, enabled);
        builder
            .when(enabled)
            .assert_eq(value, bits_expr::<AB>(limb));
    }
}
fn fill_bits(target: &mut [F; LIMB_BITS], value: u16) {
    for (index, bit) in target.iter_mut().enumerate() {
        *bit = F::from_bool(((value >> index) & 1) == 1);
    }
}
fn fill_u32(value: u32, lo: &mut F, hi: &mut F, bits: &mut [[F; LIMB_BITS]; 2]) {
    let (low, high) = split_u32(value);
    *lo = F::from_u16(low);
    *hi = F::from_u16(high);
    fill_bits(&mut bits[0], low);
    fill_bits(&mut bits[1], high);
}
fn fill_checkpoint(
    checkpoint: &TranscriptCheckpointRecordV19,
    lo: &mut F,
    hi: &mut F,
    sample_count: &mut F,
    state: &mut [F; TRANSCRIPT_WIDTH_V19],
) {
    let (low, high) = split_u32(checkpoint.operation_index);
    *lo = F::from_u16(low);
    *hi = F::from_u16(high);
    *sample_count = F::from_u8(checkpoint.sample_count);
    *state = checkpoint.state;
}
fn join_u32(lo: F, hi: F) -> u32 {
    lo.as_canonical_u32() | (hi.as_canonical_u32() << LIMB_BITS)
}
fn extension_digest(value: ExtensionV19) -> DigestV19 {
    let mut digest = [F::ZERO; DIGEST_SIZE];
    digest[..D_EF].copy_from_slice(&value);
    digest
}
fn tagged_extension(tag: u32, value: ExtensionV19) -> DigestV19 {
    let mut digest = [F::ZERO; DIGEST_SIZE];
    digest[0] = F::from_u32(tag);
    digest[1..1 + D_EF].copy_from_slice(&value);
    digest
}
fn checkpoint_meta(tag: u32, kind: u32, proof_index: u32, tidx: u32, sample_count: F) -> DigestV19 {
    let (proof_lo, proof_hi) = split_u32(proof_index);
    let (tidx_lo, tidx_hi) = split_u32(tidx);
    scalar_digest(&[
        tag,
        NATIVE_WARP_HISTORY_PROTOCOL_V19,
        kind,
        u32::from(proof_lo),
        u32::from(proof_hi),
        u32::from(tidx_lo),
        u32::from(tidx_hi),
        sample_count.as_canonical_u32(),
    ])
}

const _: () = assert!(TRANSCRIPT_WIDTH_V19 == 2 * DIGEST_SIZE);

#[cfg(test)]
mod fixed_hleaf_v4_tests {
    use openvm_recursion_circuit::native_warp::{
        NativeFixedHLeafVaccRouteBusV4, NativeFixedHLeafVaccRouteV4, NativeFixedHLeafVaccSeedV4,
        NativeFixedHLeafVaccTransitionBusV4, FIXED_HLEAF_VACC_CAPACITY_V4,
    };
    use openvm_stark_backend::interaction::BusIndex;

    use super::*;

    fn digest(seed: u32) -> DigestV19 {
        core::array::from_fn(|index| F::from_u32(seed + index as u32))
    }

    fn replay(slot: usize, step: u32) -> CertifiedWarpReplayMessageV19<F> {
        CertifiedWarpReplayMessageV19 {
            protocol_version: F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            segment_index_lo: F::from_usize(slot),
            segment_index_hi: F::ZERO,
            update_index_lo: F::from_u32(step),
            update_index_hi: F::ZERO,
            shard_ordinal: F::ZERO,
            source_forest_root: digest(10 + slot as u32),
            key_digest: digest(20),
            relation_digest: digest(30),
            opening_claim_digest: digest(40 + slot as u32),
            fresh_instance_digest: digest(50 + slot as u32),
            segment_openings_digest: digest(60 + slot as u32),
            prior_root: [F::ZERO; DIGEST_SIZE],
            fresh_root: digest(70 + slot as u32),
            next_root: digest(80 + slot as u32),
            previous_accumulator_digest: [F::ZERO; DIGEST_SIZE],
            next_accumulator_digest: digest(90 + slot as u32),
            authenticated_batching_claim: [F::ZERO; D_EF],
            previous_checkpoint_digest: [F::ZERO; DIGEST_SIZE],
            next_checkpoint_digest: digest(100 + slot as u32),
            replay_endpoint_digest: digest(110 + slot as u32),
            replay_binding_digest: digest(120 + slot as u32),
        }
    }

    fn seed() -> NativeFixedHLeafVaccSeedV4 {
        NativeFixedHLeafVaccSeedV4 {
            key_digest: digest(20),
            relation_digest: digest(30),
            root: digest(200),
            accumulator_digest: digest(210),
            genesis_anchor_digest: digest(220),
        }
    }

    fn air() -> FixedHLeafWarpReplayAirV4 {
        FixedHLeafWarpReplayAirV4 {
            route_bus: NativeFixedHLeafVaccRouteBusV4::new(BusIndex::from(1u16)),
            replay_bus: CertifiedWarpReplayBusV19::new(BusIndex::from(2u16)),
            canonical_replay_bus: CertifiedWarpReplayBusV19::new(BusIndex::from(3u16)),
            transition_bus: NativeFixedHLeafVaccTransitionBusV4::new(BusIndex::from(4u16)),
            seed: seed(),
        }
    }

    #[test]
    fn honest_seeded_continuation_chain() {
        let routes = (0..FIXED_HLEAF_VACC_CAPACITY_V4)
            .map(|slot| {
                NativeFixedHLeafVaccRouteV4::derive(slot, slot as u32, slot as u32).unwrap()
            })
            .collect::<Vec<_>>();
        let mut replays = (0..FIXED_HLEAF_VACC_CAPACITY_V4)
            .map(|slot| replay(slot, 1))
            .collect::<Vec<_>>();
        replays[0].prior_root = seed().root;
        replays[0].previous_accumulator_digest = seed().accumulator_digest;
        replays[0].previous_checkpoint_digest = digest(219);
        for slot in 1..replays.len() {
            replays[slot].prior_root = replays[slot - 1].next_root;
            replays[slot].previous_accumulator_digest = replays[slot - 1].next_accumulator_digest;
            // Every transition owns an independent source/VACC transcript.
            replays[slot].previous_checkpoint_digest = digest(219 + slot as u32);
        }
        let trace = generate_fixed_hleaf_warp_replay_trace_v4(&air(), &routes, &replays).unwrap();
        assert_eq!(trace.height(), FIXED_HLEAF_VACC_CAPACITY_V4);
    }

    #[test]
    fn later_slot_zero_is_continuation_and_wrong_step_fails() {
        let route = NativeFixedHLeafVaccRouteV4::derive(0, 4, 0).unwrap();
        let mut statement = replay(0, 1);
        statement.prior_root = digest(200);
        statement.previous_accumulator_digest = digest(210);
        statement.previous_checkpoint_digest = digest(220);
        generate_fixed_hleaf_warp_replay_trace_v4(&air(), &[route], &[statement.clone()]).unwrap();

        statement.update_index_lo = F::ZERO;
        assert!(matches!(
            generate_fixed_hleaf_warp_replay_trace_v4(&air(), &[route], &[statement]),
            Err(FixedHLeafWarpReplayErrorV4::Replay(0))
        ));
    }

    #[test]
    fn wrong_seed_chain_and_inactive_suffix_are_rejected_or_zero() {
        let routes = [
            NativeFixedHLeafVaccRouteV4::derive(0, 0, 0).unwrap(),
            NativeFixedHLeafVaccRouteV4::derive(1, 1, 1).unwrap(),
        ];
        let mut first = replay(0, 1);
        first.prior_root = seed().root;
        first.previous_accumulator_digest = seed().accumulator_digest;
        first.previous_checkpoint_digest = digest(219);
        let mut second = replay(1, 1);
        second.prior_root = digest(999);
        second.previous_accumulator_digest = first.next_accumulator_digest;
        second.previous_checkpoint_digest = digest(500);
        assert!(matches!(
            generate_fixed_hleaf_warp_replay_trace_v4(&air(), &routes, &[first, second]),
            Err(FixedHLeafWarpReplayErrorV4::Chain(1))
        ));

        let mut wrong_seed = replay(0, 1);
        wrong_seed.prior_root = digest(999);
        wrong_seed.previous_accumulator_digest = seed().accumulator_digest;
        wrong_seed.previous_checkpoint_digest = digest(219);
        assert!(matches!(
            generate_fixed_hleaf_warp_replay_trace_v4(
                &air(),
                &[NativeFixedHLeafVaccRouteV4::derive(0, 0, 0).unwrap()],
                &[wrong_seed]
            ),
            Err(FixedHLeafWarpReplayErrorV4::Seed(0))
        ));

        let route = NativeFixedHLeafVaccRouteV4::derive(0, 0, 0).unwrap();
        let mut genesis = replay(0, 1);
        genesis.prior_root = seed().root;
        genesis.previous_accumulator_digest = seed().accumulator_digest;
        genesis.previous_checkpoint_digest = digest(219);
        let trace =
            generate_fixed_hleaf_warp_replay_trace_v4(&air(), &[route], &[genesis]).unwrap();
        let width = trace.width();
        assert!(trace.values[width..].iter().all(|value| *value == F::ZERO));
    }

    #[test]
    fn adjacent_transitions_do_not_resume_each_others_transcript() {
        let routes = [
            NativeFixedHLeafVaccRouteV4::derive(0, 0, 0).unwrap(),
            NativeFixedHLeafVaccRouteV4::derive(1, 1, 1).unwrap(),
        ];
        let mut first = replay(0, 1);
        first.prior_root = seed().root;
        first.previous_accumulator_digest = seed().accumulator_digest;
        first.previous_checkpoint_digest = digest(301);
        let mut second = replay(1, 1);
        second.prior_root = first.next_root;
        second.previous_accumulator_digest = first.next_accumulator_digest;
        second.previous_checkpoint_digest = digest(777);
        assert_ne!(
            second.previous_checkpoint_digest,
            first.next_checkpoint_digest
        );

        generate_fixed_hleaf_warp_replay_trace_v4(&air(), &routes, &[first, second])
            .expect("independently initialized transition transcripts are canonical");
    }
}

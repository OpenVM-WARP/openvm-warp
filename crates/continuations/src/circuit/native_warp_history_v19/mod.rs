//! Compact protocol-v19 history AIR for the homogeneous product-WARP design.
//!
//! This module proves the ordered *history of statements*.  It does not put a
//! PCS opening into PESAT, does not verify a universal selector relation, and
//! does not expose the transition history in the final wire proof.  The long
//! [`HistoryRecordV19`] is private witness material.  The eventual compact
//! STARK certificate exposes only [`HistoryPublicValuesV19`].
//!
//! The first implementation deliberately separates two responsibilities:
//!
//! - this AIR enforces manifest ordering, VM continuity, authenticated product leaf replacement,
//!   segment-local LogUp cancellation, and the final boundary;
//! - dedicated verifier AIRs certify each standard WARP replay and each `LogUpOnly` proof, then
//!   publish their exact statements on the typed buses consumed here.
//!
//! Physical CPU/CUDA batching may share launches, but the replay message keeps
//! the relation digest, segment index, and per-shard transcript checkpoints so
//! logical WARP transcripts remain independent.

mod air;
mod cuda_shared_forest;
mod digest;
mod error;
mod logup_swirl_composite;
mod logup_swirl_manifest;
mod logup_swirl_source;
mod logup_swirl_verifier;
mod producer;
mod profile;
mod record;
mod recursive_chunk;
mod setup_pcs_source_provenance_bus_v3;
mod trace;
mod vacc_groups;
mod vacc_verifier;
mod vacc_verifier_cuda;

pub use air::{
    CertifiedWarpReplayBusV19, CertifiedWarpReplayMessageV19, HistoryAirV19,
    HistoryGenesisConfigV19, HistoryPoseidon2CompressBusV19, HistoryPoseidon2CompressMessageV19,
    HistoryRowColsV19, LogUpOnlyHistoryBusV19, LogUpOnlyHistoryMessageV19,
};
pub use cuda_shared_forest::*;
pub use digest::{
    compute_history_profile_digest_v19, compute_manifest_digest_v19,
    compute_replay_binding_digest_v19, compute_shard_key_digest_v19,
    compute_terminal_certificate_digest_v19, DigestCollectorV19,
};
pub use error::HistoryV19Error;
#[cfg(test)]
pub(crate) use logup_swirl_composite::tests::retained_fixed_multi_air_fixture;
pub use logup_swirl_composite::*;
pub use logup_swirl_manifest::*;
pub use logup_swirl_source::*;
pub use logup_swirl_verifier::*;
pub use producer::{
    certified_direct_air_vacc_input_message_v19, generate_logup_only_producer_trace_v19,
    generate_warp_replay_producer_trace_v19, CertifiedDirectAirVaccInputBusV19,
    CertifiedDirectAirVaccInputMessageV19, CertifiedFreshExplicitDigestBusV19,
    CertifiedFreshExplicitDigestMessageV19, CertifiedLogUpOnlyEndpointBusV19,
    CertifiedLogUpOnlyEndpointMessageV19, CertifiedSwirlRawOpeningBusV19,
    CertifiedSwirlRawOpeningMessageV19, DirectAirVaccContextBusV19, DirectAirVaccContextMessageV19,
    DirectAirVaccProducerRecordV19, LogUpOnlyProducerAirV19, LogUpOnlyProducerColsV19,
    LogUpOnlyProducerRecordV19, PositiveProducerErrorV19, PositiveProducerTraceV19,
    TranscriptCheckpointRecordV19, WarpReplayProducerAirV19, WarpReplayProducerColsV19,
    WarpReplayProducerScheduleV19, MAX_FRESH_BETA_LEN_V19, MAX_RAW_MESSAGE_POINT_LEN_V19,
    TRANSCRIPT_WIDTH_V19,
};
pub use profile::{HistoryVerifierProfileV19, NATIVE_WARP_HISTORY_PROTOCOL_V19};
pub use record::{
    history_public_values_width_v19, DigestV19, ExtensionV19, HistoryBoundaryV19,
    HistoryPublicValuesV19, HistoryRecordV19, LogUpOnlyRecordV19, SegmentManifestRecordV19,
    ShardKeyRecordV19, ShardTransitionRecordV19, WarpCertifiedReplayRecordV19,
    LOGUP_ONLY_MODE_TAG_V19,
};
pub use recursive_chunk::{HistoryChunkBridgeV19, HistoryPublicValuesBusV19};
pub use setup_pcs_source_provenance_bus_v3::*;
pub use trace::{generate_history_trace_v19, HistoryTraceV19};
pub use vacc_groups::*;
#[cfg(test)]
pub(crate) use vacc_verifier::tests::appendix_d_fixture_for_fixed_source;
pub use vacc_verifier::*;
pub use vacc_verifier_cuda::*;

#[cfg(test)]
mod tests;

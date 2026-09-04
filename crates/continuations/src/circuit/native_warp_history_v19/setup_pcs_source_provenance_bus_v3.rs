//! Typed boundaries for setup-PCS source provenance.
//!
//! These messages authenticate the complete-SWIRL source statement used by
//! terminal setup-PCS authority.  The `raw_message_*` fields belong to the
//! one-shot WARP systematic-message opening.  They are intentionally present
//! only in the receipt hash and are never a setup PLE/stacking opening point.

use openvm_recursion_circuit::define_typed_permutation_bus;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, D_EF};

use super::{MAX_RAW_MESSAGE_POINT_LEN_V19, TRANSCRIPT_WIDTH_V19};

// Protocol revision 4 adds the absolute segment index alongside the local
// transition slot. The Rust type names remain V3 to avoid a mechanical API
// fork, but the transcript/bus version is bumped because this changes the
// constrained message schema and every derived relation key.
pub const SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3: u32 = 4;
/// Production fanout of one certified fixed source: producer bridge,
/// active-count functional check, and setup-PCS source provenance.
pub const SETUP_PCS_SOURCE_FIXED_LOOKUP_COUNT_V3: u32 = 3;
/// The constrained receipt is consumed by the setup-authority bridge, the
/// authority transition-statement owner, and the source-to-VACC transcript
/// resume bridge.
pub const SETUP_PCS_SOURCE_PROVENANCE_MULTIPLICITY_V3: u32 = 3;

/// Source-manifest statement emitted by the genuine direct SWIRL manifest
/// AIR on the unique last source row of a transition.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug, PartialEq, Eq)]
pub struct SetupPcsSourceManifestMessageV3<T> {
    pub protocol_version: T,
    /// Setup-authority-local transition slot in `0..transition_count`.
    pub transition_index: [T; 2],
    /// Absolute WARP transition index in the complete block history.
    pub segment_index: [T; 2],
    pub app_vk_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub source_root: [T; DIGEST_SIZE],
    pub source_instance_digest: [T; DIGEST_SIZE],
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
}
define_typed_permutation_bus!(SetupPcsSourceManifestBusV3, SetupPcsSourceManifestMessageV3);

/// End-checkpoint statement republished by `LogUpOnlyProducerAirV19` only
/// after it has consumed the genuine start/end transcript checkpoints and the
/// certified LogUp endpoint.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug, PartialEq, Eq)]
pub struct SetupPcsSourceCheckpointMessageV3<T> {
    pub protocol_version: T,
    /// Setup-authority-local transition slot in `0..transition_count`.
    pub transition_index: [T; 2],
    /// Absolute WARP transition index in the complete block history.
    pub segment_index: [T; 2],
    pub app_vk_digest: [T; DIGEST_SIZE],
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub verifier_endpoint: [T; D_EF],
    pub end_tidx: [T; 2],
    pub end_sample_count: T,
    pub end_state: [T; TRANSCRIPT_WIDTH_V19],
    pub logup_history_digest: [T; DIGEST_SIZE],
}
define_typed_permutation_bus!(
    SetupPcsSourceCheckpointBusV3,
    SetupPcsSourceCheckpointMessageV3
);

/// Canonical constrained source receipt.  It deliberately exposes only
/// digests for the raw-message opening; the setup PLE point is transported on
/// its independent `FixedSetupOpeningPointBusV2` path.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug, PartialEq, Eq)]
pub struct SetupPcsSourceProvenanceMessageV3<T> {
    pub protocol_version: T,
    /// Setup-authority-local transition slot in `0..transition_count`.
    pub transition_index: [T; 2],
    /// Absolute WARP transition index in the complete block history.
    pub segment_index: [T; 2],
    pub app_vk_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub source_root: [T; DIGEST_SIZE],
    pub source_instance_digest: [T; DIGEST_SIZE],
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub source_checkpoint_digest: [T; DIGEST_SIZE],
    pub source_manifest_digest: [T; DIGEST_SIZE],
    pub source_receipt_digest: [T; DIGEST_SIZE],
    pub end_tidx: [T; 2],
    pub end_sample_count: T,
    pub end_state: [T; TRANSCRIPT_WIDTH_V19],
}
define_typed_permutation_bus!(
    SetupPcsSourceProvenanceBusV3,
    SetupPcsSourceProvenanceMessageV3
);

/// Exact raw-message opening material consumed privately by the provenance
/// AIR.  This is a host record helper, not an interaction message.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetupPcsRawMessageOpeningV3<T> {
    pub point_len: T,
    pub point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub value: [T; D_EF],
}

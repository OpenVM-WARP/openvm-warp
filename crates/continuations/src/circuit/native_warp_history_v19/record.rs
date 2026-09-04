use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, D_EF, F};

pub type DigestV19 = [F; DIGEST_SIZE];
pub type ExtensionV19 = [F; D_EF];

/// Transcript-domain value fixed by protocol v19 for the external LogUp-only
/// proof.  It is deliberately distinct from the AIR-and-LogUp mode.
pub const LOGUP_ONLY_MODE_TAG_V19: u32 = 0x19_4c_55;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryBoundaryV19 {
    pub segment_count: u32,
    pub vm_state: DigestV19,
    pub product_state_root: DigestV19,
    pub history_root: DigestV19,
    pub public_values_digest: DigestV19,
    pub terminated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardKeyRecordV19 {
    /// Canonical verifier-profile slot.  Strict slot order is the manifest
    /// order; the profile Merkle proof binds this slot to every key field.
    pub ordinal: u16,
    pub app_vk_digest: DigestV19,
    pub air_id: u32,
    pub log_height: u8,
    pub trace_layout_digest: DigestV19,
    pub public_schema_digest: DigestV19,
    pub interaction_schema_digest: DigestV19,
    pub relation_digest: DigestV19,
    pub code_class_digest: DigestV19,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WarpCertifiedReplayRecordV19 {
    /// Shard-local VACC step index.  It is distinct from the global segment
    /// index and is certified by the fixed direct-AIR verifier context.
    pub update_index: u32,
    /// Digest of the one raw-message MLE opening `(r, v)` certified for this
    /// shard by the segment-local SWIRL reduction.
    pub opening_claim_digest: DigestV19,
    /// Digest of the fresh WARP input `(root, alpha, mu, beta, eta)` after the
    /// producer has enforced `alpha = (r, 0^b)`, `mu = v`, and `eta = 0`.
    pub fresh_instance_digest: DigestV19,
    /// Canonical aggregate of all raw openings in this segment.  It is
    /// segment-local and is never folded into a persistent mapped-column claim.
    pub segment_openings_digest: DigestV19,
    pub prior_root: DigestV19,
    pub fresh_root: DigestV19,
    pub next_root: DigestV19,
    pub previous_accumulator_digest: DigestV19,
    pub next_accumulator_digest: DigestV19,
    pub authenticated_batching_claim: ExtensionV19,
    pub previous_checkpoint_digest: DigestV19,
    pub next_checkpoint_digest: DigestV19,
    pub replay_endpoint_digest: DigestV19,
    /// Output of the certified replay verifier.  The history AIR recomputes
    /// the same binding and consumes it on `CertifiedWarpReplayBusV19`.
    pub replay_binding_digest: DigestV19,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardTransitionRecordV19 {
    pub key: ShardKeyRecordV19,
    pub replay: WarpCertifiedReplayRecordV19,
    /// Siblings for replacing this shard leaf in the product-state tree.
    pub product_siblings: Vec<DigestV19>,
    /// Siblings authenticating `key` at the same ordinal in the VK catalog.
    pub catalog_siblings: Vec<DigestV19>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogUpOnlyRecordV19 {
    pub mode_tag: u32,
    /// Verifier-derived terminal value of the LogUpOnly GKR/batch-constraint
    /// endpoint.  This is not the History accumulator contribution.
    pub verifier_endpoint: ExtensionV19,
    /// Canonical segment-local aggregate of the raw-message MLE openings
    /// derived from the authenticated LogUp column openings.
    pub segment_openings_digest: DigestV19,
    /// Binds the external LogUp-only transcript endpoint/proof statement.
    pub checkpoint_digest: DigestV19,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SegmentManifestRecordV19 {
    pub segment_index: u32,
    /// Raw VM boundary values certified by the segment-wide direct-AIR
    /// verifier.  History consumes the exact certificate rather than trusting
    /// host-computed `from_vm_state` / `to_vm_state` digests.
    pub initial_pc: F,
    pub final_pc: F,
    pub exit_code: F,
    pub initial_memory_root: DigestV19,
    pub final_memory_root: DigestV19,
    /// Stable Program identity certified by the Program direct-AIR opening.
    pub program_fingerprint: ExtensionV19,
    pub program_fingerprint_digest: DigestV19,
    pub program_registry_digest: DigestV19,
    pub program_relation_digest: DigestV19,
    pub program_log_height: u8,
    pub program_cached_width: u32,
    pub from_vm_state: DigestV19,
    pub to_vm_state: DigestV19,
    /// Root of the segment-wide stacked source commitment / Merkle forest.
    /// Both external certificate buses consume this exact root.
    pub source_forest_root: DigestV19,
    pub previous_product_state_root: DigestV19,
    pub next_product_state_root: DigestV19,
    pub active_shards: Vec<ShardTransitionRecordV19>,
    pub logup: LogUpOnlyRecordV19,
    pub public_values_digest: DigestV19,
    pub terminates: bool,
    /// Canonical digest of header, ordered shard events, LogUp event, and end.
    pub manifest_digest: DigestV19,
}

/// Private witness consumed by the history prover.  This type must never be a
/// field of the final protocol-v19 proof envelope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HistoryRecordV19 {
    pub initial: HistoryBoundaryV19,
    pub segments: Vec<SegmentManifestRecordV19>,
    pub expected_final: HistoryBoundaryV19,
    /// A non-terminal chunk may be certified and chained later.  LogUp is
    /// nevertheless segment-local: every segment starts and ends at zero.
    pub terminal_chunk: bool,
}

/// Public statement of one compact History-AIR certificate.  Its width is
/// independent of the number of segments and transitions in the private
/// witness.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Debug)]
pub struct HistoryPublicValuesV19<T> {
    pub protocol_version: T,
    pub profile_digest: [T; DIGEST_SIZE],
    pub app_vk_digest: [T; DIGEST_SIZE],

    pub initial_segment_count_lo: T,
    pub initial_segment_count_hi: T,
    pub initial_vm_state: [T; DIGEST_SIZE],
    pub initial_product_state_root: [T; DIGEST_SIZE],
    pub initial_history_root: [T; DIGEST_SIZE],
    pub initial_public_values_digest: [T; DIGEST_SIZE],
    pub initial_terminated: T,

    pub final_segment_count_lo: T,
    pub final_segment_count_hi: T,
    pub final_vm_state: [T; DIGEST_SIZE],
    pub final_product_state_root: [T; DIGEST_SIZE],
    pub final_history_root: [T; DIGEST_SIZE],
    pub final_public_values_digest: [T; DIGEST_SIZE],
    pub final_terminated: T,

    /// Fixed-size terminal execution boundary.  These values are certified by
    /// the terminating segment's direct-AIR VM semantics and let a standalone
    /// verifier check the user-public-values Merkle proof without receiving a
    /// private segment record.
    pub final_pc: T,
    pub final_exit_code: T,
    pub final_memory_root: [T; DIGEST_SIZE],
    pub program_fingerprint_digest: [T; DIGEST_SIZE],

    pub terminal_chunk: T,
    pub terminal_certificate_digest: [T; DIGEST_SIZE],
}

impl HistoryPublicValuesV19<F> {
    pub fn from_boundaries(
        profile_digest: DigestV19,
        app_vk_digest: DigestV19,
        initial: &HistoryBoundaryV19,
        final_boundary: &HistoryBoundaryV19,
        terminal_segment: Option<&SegmentManifestRecordV19>,
        terminal_chunk: bool,
        terminal_certificate_digest: DigestV19,
    ) -> Vec<F> {
        let mut values = vec![F::ZERO; Self::width()];
        let pvs: &mut Self = values.as_mut_slice().borrow_mut();
        pvs.protocol_version = F::from_u32(super::NATIVE_WARP_HISTORY_PROTOCOL_V19);
        pvs.profile_digest = profile_digest;
        pvs.app_vk_digest = app_vk_digest;
        let (initial_lo, initial_hi) = split_u32(initial.segment_count);
        pvs.initial_segment_count_lo = F::from_u16(initial_lo);
        pvs.initial_segment_count_hi = F::from_u16(initial_hi);
        pvs.initial_vm_state = initial.vm_state;
        pvs.initial_product_state_root = initial.product_state_root;
        pvs.initial_history_root = initial.history_root;
        pvs.initial_public_values_digest = initial.public_values_digest;
        pvs.initial_terminated = F::from_bool(initial.terminated);
        let (final_lo, final_hi) = split_u32(final_boundary.segment_count);
        pvs.final_segment_count_lo = F::from_u16(final_lo);
        pvs.final_segment_count_hi = F::from_u16(final_hi);
        pvs.final_vm_state = final_boundary.vm_state;
        pvs.final_product_state_root = final_boundary.product_state_root;
        pvs.final_history_root = final_boundary.history_root;
        pvs.final_public_values_digest = final_boundary.public_values_digest;
        pvs.final_terminated = F::from_bool(final_boundary.terminated);
        if let Some(segment) = terminal_segment.filter(|_| terminal_chunk) {
            pvs.final_pc = segment.final_pc;
            pvs.final_exit_code = segment.exit_code;
            pvs.final_memory_root = segment.final_memory_root;
            pvs.program_fingerprint_digest = segment.program_fingerprint_digest;
        }
        pvs.terminal_chunk = F::from_bool(terminal_chunk);
        pvs.terminal_certificate_digest = terminal_certificate_digest;
        values
    }
}

/// Fixed protocol-v19 public-value width without requiring downstream crates
/// to depend on the struct-reflection implementation detail.
#[must_use]
pub fn history_public_values_width_v19() -> usize {
    HistoryPublicValuesV19::<F>::width()
}

#[inline]
pub(crate) const fn split_u32(value: u32) -> (u16, u16) {
    (value as u16, (value >> 16) as u16)
}

#[allow(dead_code)]
fn _assert_borrow_layout(values: &[F]) -> Option<&HistoryPublicValuesV19<F>> {
    (values.len() == HistoryPublicValuesV19::<F>::width()).then(|| values.borrow())
}

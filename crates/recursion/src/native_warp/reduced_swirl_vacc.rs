//! Recursion-side authority for reduced-SWIRL WARP transitions.
//!
//! This module is the adapter between the retained SWIRL commitment tuple and
//! the ordinary WARP `Verify` AIRs.  It deliberately does not define a PESAT
//! relation.  Instead it:
//!
//! 1. links the authoritative constrained-RS claim exported by [`super::ReducedSwirlSourceAir`] to
//!    WARP's normalized fresh claim;
//! 2. authenticates WARP shift openings against the exact ordered tuple of original SWIRL roots
//!    with [`super::generate_native_direct_fresh_opening_traces`];
//! 3. constrains the fixed-arity linear schedule, including a final partial fresh batch and
//!    distinct bootstrap/continuation modes;
//! 4. constrains the call header, ordered source-entry batch, ordinary-WARP prefix, and transition
//!    receipt used by the recursive transition tree.
//!
//! The remaining twin/constraint sumcheck, prior authentication, output
//! commitment, and batching sumcheck use the standard VACC verifier
//! machinery. Keeping that algebra unchanged is the transcript and
//! accumulator anchor for this adapter.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    transcript::{TranscriptHistory, TranscriptLog},
    warp_accum::{
        MerkleBatchOpeningVerification, ReducedWarpVaccStepProof, StackedRsBatchOpeningProof,
        StackedRsBatchOpeningVerification, StackedRsFreshCommitment, WarpVaccStepVerification,
        NATIVE_WARP_VACC_PROTOCOL_TAG,
    },
    warp_pesat::LinearChainSchedule,
    AirRef, BaseAirWithPublicValues, FiatShamirTranscript, PartitionedBaseAir, StarkProtocolConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, poseidon2_compress_with_capacity, BabyBearPoseidon2Config,
    Digest, CHUNK, DIGEST_SIZE, D_EF, EF, F,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::{
        CertifiedTranscriptCheckpointBus, CertifiedTranscriptCheckpointMessage,
        Poseidon2CompressBus, Poseidon2CompressMessage, ResumeTranscriptStateBus,
        ResumeTranscriptStateMessage, TranscriptBus,
    },
    native_warp::{
        NativeClaimValueBus, NativeClaimValueMessage, NativeDirectFreshRootBus,
        NativeDirectFreshRootMessage, NativeDirectFreshSourceBus, NativeDirectFreshSourceMessage,
        NativeFreshCountBus, NativeFreshCountMessage, NativeInputSlotLayoutBus,
        NativeInputSlotLayoutMessage, NativeStandardVaccDigestBus, NativeStandardVaccDigestMessage,
        NativeStandardVaccEndBus, NativeStandardVaccEndMessage, NativeStandardVaccRootBus,
        NativeStandardVaccRootMessage, NativeVaccPhaseCursorBus, NativeVaccPhaseCursorMessage,
        RecursiveReducedSwirlClaim, ReducedSwirlSourceBetaBus, ReducedSwirlSourceBetaMessage,
        ReducedSwirlSourceClaimBus, ReducedSwirlSourceClaimMessage, ReducedSwirlSourceProfile,
        ReducedSwirlSourceRootBus, ReducedSwirlSourceRootMessage, CLAIM_SECTION_ALPHA,
        CLAIM_SECTION_BETA, CLAIM_SECTION_ETA, CLAIM_SECTION_MU,
    },
};

/// Corrected outer transcript header.  A manifest is not accepted here: it is
/// derived from all ordered source entries and absorbed by the footer.
pub const REDUCED_SWIRL_VACC_HEADER_TAG: &[u8] = b"openvm.native-warp.swirl-reduced-source.v3";
pub const REDUCED_SWIRL_VACC_SOURCE_BATCH_TAG: &[u8] =
    b"openvm.native-warp.swirl-reduced-source.batch.v3";
pub const REDUCED_SWIRL_VACC_MANIFEST_TAG: &[u8] =
    b"openvm.native-warp.swirl-reduced-source.manifest.v3";
pub const REDUCED_SWIRL_VACC_FOOTER_TAG: &[u8] =
    b"openvm.native-warp.swirl-reduced-source.manifest-footer.v3";
/// Must equal the native reduced-SWIRL protocol version. Version 3 binds the
/// corrected fresh relation shape where `beta` excludes the normalized
/// accumulator target.
pub const REDUCED_SWIRL_VACC_PROTOCOL_VERSION: u32 = 3;
pub const REDUCED_SWIRL_VACC_MAX_INPUT_ARITY: usize = 64;
pub const REDUCED_SWIRL_VACC_MAX_ROOTS: usize = 64;
pub const REDUCED_SWIRL_VACC_MAX_LOG_CODEWORD_LEN: usize = 32;
/// Setup-fixed maximum for the terminal manifest reconciliation.  This is a
/// finite statement over one block, not a selector-tagged universal relation.
pub const REDUCED_SWIRL_MANIFEST_RECONCILIATION_MAX_SOURCES: usize = 1024;
/// These constants are field-identical to the transition-leaf protocol in
/// `openvm-continuations`.  The dependency direction does not permit this
/// recursion crate to import that circuit.  Callers must retain a
/// differential test against `reduced_swirl_transition_chain_{genesis,append}`
/// whenever either protocol version changes.
pub const REDUCED_SWIRL_TRANSITION_CHAIN_METADATA_TAG: u32 = 0x5254_4c01;
pub const REDUCED_SWIRL_TRANSITION_LEAF_PROTOCOL_VERSION: u32 = 2;
pub const REDUCED_SWIRL_TRANSITION_CHAIN_GENESIS_TAG: u32 = 0x5254_4c03;
const REDUCED_SWIRL_SCHEDULE_BITS: usize = 16;
const REDUCED_SWIRL_AUTHORITY_BITS: usize = 30;
const REDUCED_SWIRL_SOURCE_DIGEST_TAG: u32 = 0x5253_0101;
const REDUCED_SWIRL_FIELD_TAG: u32 = 0x5253_0201;
const REDUCED_SWIRL_DIGEST_TAG: u32 = 0x5253_0202;
const REDUCED_SWIRL_EXTENSION_TAG: u32 = 0x5253_0203;
const REDUCED_SWIRL_END_TAG: u32 = 0x5253_02ff;

crate::define_typed_lookup_bus!(ReducedSwirlVaccCallBus, ReducedSwirlVaccCallMessage);
crate::define_typed_lookup_bus!(
    ReducedSwirlVaccSourceSlotBus,
    ReducedSwirlVaccSourceSlotMessage
);
crate::define_typed_permutation_bus!(
    ReducedSwirlSourceAuthorityBus,
    ReducedSwirlSourceAuthorityMessage
);
crate::define_typed_lookup_bus!(
    ReducedSwirlSourceEntryDigestBus,
    ReducedSwirlSourceEntryDigestMessage
);
crate::define_typed_permutation_bus!(
    ReducedSwirlVaccHeaderEndBus,
    ReducedSwirlVaccHeaderEndMessage
);
crate::define_typed_permutation_bus!(
    ReducedSwirlManifestDigestBus,
    ReducedSwirlManifestDigestMessage
);
crate::define_typed_permutation_bus!(ReducedSwirlVaccFooterBus, ReducedSwirlVaccFooterMessage);
crate::define_typed_permutation_bus!(ReducedSwirlVaccChainEndBus, ReducedSwirlVaccChainEndMessage);
crate::define_typed_lookup_bus!(
    ReducedSwirlVaccChainReceiptBus,
    ReducedSwirlVaccChainReceiptMessage
);
crate::define_typed_permutation_bus!(
    ReducedSwirlVaccTransitionEndBus,
    ReducedSwirlVaccTransitionEndMessage
);
crate::define_typed_lookup_bus!(
    ReducedSwirlVaccTransitionReceiptBus,
    ReducedSwirlVaccTransitionReceiptMessage
);
crate::define_typed_lookup_bus!(
    ReducedSwirlManifestReconciliationReceiptBus,
    ReducedSwirlManifestReconciliationReceiptMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ReducedSwirlVaccCallMessage<T> {
    pub step: T,
    pub source_start: T,
    pub fresh_count: T,
    pub prior_count: T,
    pub input_arity: T,
    pub batch_start_tidx: T,
    pub vacc_start_tidx: T,
    pub vacc_end_tidx: T,
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ReducedSwirlVaccSourceSlotMessage<T> {
    pub source: T,
    pub step: T,
    pub slot: T,
    pub fresh_count: T,
    pub prior_count: T,
    pub input_arity: T,
    pub batch_start_tidx: T,
    pub vacc_start_tidx: T,
}

/// Authority supplied by the retained recursive SWIRL prefix and VM-boundary
/// verifier.  These are values already constrained by that verifier, not host
/// booleans or proof-carried claims.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ReducedSwirlSourceAuthorityMessage<T> {
    pub source: T,
    pub segment_index: T,
    pub common_main_root: [T; DIGEST_SIZE],
    pub trace_layout_digest: [T; DIGEST_SIZE],
    pub pending_claim_digest: [T; DIGEST_SIZE],
    pub checkpoint_tidx: T,
    pub checkpoint_state: [T; POSEIDON2_WIDTH],
    pub program_commitment: [T; DIGEST_SIZE],
    pub initial_pc: T,
    pub initial_root: [T; DIGEST_SIZE],
    pub final_pc: T,
    pub final_root: [T; DIGEST_SIZE],
    pub exit_code: T,
    pub is_terminate: T,
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ReducedSwirlSourceEntryDigestMessage<T> {
    pub source: T,
    pub digest: [T; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ReducedSwirlVaccHeaderEndMessage<T> {
    pub source_count: T,
    pub end_tidx: T,
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ReducedSwirlManifestDigestMessage<T> {
    pub source_count: T,
    pub digest: [T; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ReducedSwirlVaccFooterMessage<T> {
    pub source_count: T,
    pub call_count: T,
    pub start_tidx: T,
    pub end_tidx: T,
    pub manifest_digest: [T; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ReducedSwirlVaccChainEndMessage<T> {
    pub source_count: T,
    pub call_count: T,
    pub proof_idx: T,
    pub footer_start_tidx: T,
    pub final_accumulator_digest: [T; DIGEST_SIZE],
    pub final_accumulator_root: [T; DIGEST_SIZE],
}

/// Recursion-local form of `ReducedSwirlVaccReceiptMessage`.  The field order
/// intentionally matches the continuations message so an adapter is a typed
/// bus projection rather than a second statement definition.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone, PartialEq, Eq)]
pub struct ReducedSwirlVaccChainReceiptMessage<T> {
    pub protocol_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub warp_index_digest: [T; DIGEST_SIZE],
    pub schedule_digest: [T; DIGEST_SIZE],
    pub manifest_digest: [T; DIGEST_SIZE],
    pub source_count: T,
    pub call_count: T,
    pub final_accumulator_digest: [T; DIGEST_SIZE],
    pub final_accumulator_root: [T; DIGEST_SIZE],
}

/// Complete authenticated boundary of one native reduced-SWIRL WARP call.
/// The global call/source coordinates remain distinct from leaf-local AIR and
/// transcript namespaces.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone, PartialEq, Eq)]
pub struct ReducedSwirlVaccTransitionEndMessage<T> {
    pub total_source_count: T,
    pub call_index: T,
    pub source_start: T,
    pub fresh_count: T,
    pub prior_count: T,
    pub batch_start_tidx: T,
    pub vacc_start_tidx: T,
    pub vacc_end_tidx: T,
    pub start_sample_count: T,
    pub start_state: [T; POSEIDON2_WIDTH],
    pub end_sample_count: T,
    pub end_state: [T; POSEIDON2_WIDTH],
    pub prior_root: [T; DIGEST_SIZE],
    pub output_root: [T; DIGEST_SIZE],
    pub prior_digest: [T; DIGEST_SIZE],
    pub output_digest: [T; DIGEST_SIZE],
    pub is_final: T,
}

/// One fixed-key recursive-leaf receipt.  It joins the genuine VACC verifier
/// endpoint to the canonical manifest of exactly the fresh source interval.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone, PartialEq, Eq)]
pub struct ReducedSwirlVaccTransitionReceiptMessage<T> {
    pub protocol_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub warp_index_digest: [T; DIGEST_SIZE],
    pub schedule_digest: [T; DIGEST_SIZE],
    pub total_source_count: T,
    pub call_index: T,
    pub source_start: T,
    pub fresh_count: T,
    pub prior_count: T,
    pub batch_start_tidx: T,
    pub vacc_start_tidx: T,
    pub vacc_end_tidx: T,
    pub start_sample_count: T,
    pub start_state: [T; POSEIDON2_WIDTH],
    pub end_sample_count: T,
    pub end_state: [T; POSEIDON2_WIDTH],
    pub prior_root: [T; DIGEST_SIZE],
    pub output_root: [T; DIGEST_SIZE],
    pub prior_digest: [T; DIGEST_SIZE],
    pub output_digest: [T; DIGEST_SIZE],
    pub manifest_digest: [T; DIGEST_SIZE],
    pub is_final: T,
}

/// Constant-size equality witness between the canonical flat source manifest
/// used by terminal VACC and the call-partitioned rolling manifest carried by
/// transition-tree public state.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone, PartialEq, Eq)]
pub struct ReducedSwirlManifestReconciliationReceiptMessage<T> {
    pub source_count: T,
    pub call_count: T,
    pub flat_manifest_digest: [T; DIGEST_SIZE],
    pub rolling_chain_endpoint: [T; DIGEST_SIZE],
}

/// Setup-fixed reduced-SWIRL VACC family.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlVaccProfile {
    pub maximum_sources: usize,
    pub input_arity: usize,
    pub num_ood: usize,
    pub num_shift_queries: usize,
    pub batching_arity: usize,
    pub family_target_bits: usize,
    pub source: ReducedSwirlSourceProfile,
    /// Exact setup-owned pieces observed by the v2 outer header.
    pub source_domain: Vec<u8>,
    pub relation_binding: Vec<EF>,
    pub code_binding: Vec<EF>,
    /// Exact vector supplied to backend `bind_vacc_relation_and_code`.
    pub external_protocol_binding: Vec<EF>,
    pub protocol_digest: Digest,
    pub relation_digest: Digest,
    pub warp_index_digest: Digest,
    pub schedule_digest: Digest,
}

impl ReducedSwirlVaccProfile {
    pub fn validate(&self) -> Result<(), &'static str> {
        self.source.validate()?;
        if self.maximum_sources == 0
            || self.maximum_sources > u16::MAX as usize
            || self.maximum_sources > self.source.maximum_sources
            || self.input_arity < 2
            || self.input_arity > REDUCED_SWIRL_VACC_MAX_INPUT_ARITY
            || !self.input_arity.is_power_of_two()
            || self.source.maximum_roots_per_source > REDUCED_SWIRL_VACC_MAX_ROOTS
            || self.source.log_codeword_len() > REDUCED_SWIRL_VACC_MAX_LOG_CODEWORD_LEN
            || self.num_shift_queries == 0
            || self.batching_arity
                != (1 + self.num_ood + self.num_shift_queries).next_power_of_two()
            || self.family_target_bits == 0
            || self.source_domain.is_empty()
            || self.relation_binding.is_empty()
            || self.code_binding.is_empty()
            || self.external_protocol_binding.is_empty()
        {
            return Err("reduced-SWIRL VACC profile");
        }
        for digest in [
            self.protocol_digest,
            self.relation_digest,
            self.warp_index_digest,
            self.schedule_digest,
        ] {
            if digest.iter().all(|value| *value == F::ZERO) {
                return Err("reduced-SWIRL VACC unset digest");
            }
        }
        if self.maximum_call_count() == 0 {
            return Err("reduced-SWIRL VACC call capacity");
        }
        Ok(())
    }

    #[must_use]
    pub fn maximum_call_count(&self) -> usize {
        reduced_swirl_vacc_call_count(self.maximum_sources, self.input_arity).unwrap_or(0)
    }

    #[must_use]
    pub const fn normalized_beta_len(&self) -> usize {
        self.source.log_message_len() + 1
    }

    #[must_use]
    pub const fn log_codeword_len(&self) -> usize {
        self.source.log_codeword_len()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReducedSwirlVaccCall {
    pub step: usize,
    pub source_start: usize,
    pub fresh_count: usize,
    pub prior_count: usize,
    pub input_arity: usize,
}

/// Exact backend linear-chain schedule.  The final invocation may contain
/// fewer fresh sources than its setup-fixed input arity; remaining slots are
/// ordinary WARP padding.
pub fn reduced_swirl_vacc_schedule(
    source_count: usize,
    input_arity: usize,
) -> Result<Vec<ReducedSwirlVaccCall>, &'static str> {
    if source_count == 0 || input_arity < 2 || !input_arity.is_power_of_two() {
        return Err("reduced-SWIRL VACC schedule");
    }
    let schedule =
        LinearChainSchedule::new(input_arity).map_err(|_| "reduced-SWIRL VACC schedule")?;
    let counts = schedule.step_fresh_counts(source_count);
    let mut source_start = 0usize;
    let calls = counts
        .into_iter()
        .enumerate()
        .map(|(step, fresh_count)| {
            let call = ReducedSwirlVaccCall {
                step,
                source_start,
                fresh_count,
                prior_count: usize::from(step != 0),
                input_arity,
            };
            source_start += fresh_count;
            call
        })
        .collect::<Vec<_>>();
    if source_start != source_count {
        return Err("reduced-SWIRL VACC source coverage");
    }
    Ok(calls)
}

pub fn reduced_swirl_vacc_call_count(
    source_count: usize,
    input_arity: usize,
) -> Result<usize, &'static str> {
    Ok(reduced_swirl_vacc_schedule(source_count, input_arity)?.len())
}

/// Canonical WARP protocol-prefix observations for one reduced-source call.
/// This is byte-for-byte the backend's `bind_vacc_protocol` with
/// `externally_bound_relation = Some(...)` and the all-extension alphabet.
pub fn reduced_swirl_vacc_protocol_prefix_elements(
    profile: &ReducedSwirlVaccProfile,
    call: ReducedSwirlVaccCall,
) -> Result<Vec<EF>, &'static str> {
    profile.validate()?;
    validate_call(profile, call)?;
    let mut values = Vec::new();
    values.extend(
        NATIVE_WARP_VACC_PROTOCOL_TAG
            .iter()
            .copied()
            .map(EF::from_u8),
    );
    push_u64_ext(&mut values, 1);
    push_u64_ext(&mut values, profile.external_protocol_binding.len() as u64);
    values.extend_from_slice(&profile.external_protocol_binding);
    for value in [
        profile.input_arity,
        profile.num_ood,
        profile.num_shift_queries,
        profile.batching_arity,
        profile.source.log_message_len(),
        profile.source.log_codeword_len(),
        call.step,
        call.fresh_count,
        call.prior_count,
    ] {
        push_u64_ext(&mut values, value as u64);
    }
    Ok(values)
}

/// Corrected header observations.  No caller-provided manifest appears here.
pub fn reduced_swirl_vacc_header_elements(
    profile: &ReducedSwirlVaccProfile,
    source_count: usize,
) -> Result<Vec<EF>, &'static str> {
    profile.validate()?;
    if source_count == 0 || source_count > profile.maximum_sources {
        return Err("reduced-SWIRL source count");
    }
    let mut values = Vec::new();
    push_bytes_ext(&mut values, REDUCED_SWIRL_VACC_HEADER_TAG);
    push_u64_ext(&mut values, u64::from(REDUCED_SWIRL_VACC_PROTOCOL_VERSION));
    push_u64_ext(&mut values, source_count as u64);
    for value in [
        profile.source.log_message_len(),
        profile.source.log_codeword_len(),
        profile.source.rows_per_query(),
        profile.input_arity,
        profile.num_ood,
        profile.num_shift_queries,
        profile.batching_arity,
        profile.family_target_bits,
    ] {
        push_u64_ext(&mut values, value as u64);
    }
    push_bytes_ext(&mut values, &profile.source_domain);
    values.extend_from_slice(&profile.relation_binding);
    values.extend_from_slice(&profile.code_binding);
    Ok(values)
}

pub fn reduced_swirl_vacc_footer_elements(
    source_count: usize,
    manifest_digest: Digest,
) -> Result<Vec<EF>, &'static str> {
    if source_count == 0 || manifest_digest.iter().all(|value| *value == F::ZERO) {
        return Err("reduced-SWIRL manifest footer");
    }
    let mut values = Vec::new();
    push_bytes_ext(&mut values, REDUCED_SWIRL_VACC_FOOTER_TAG);
    values.extend(manifest_digest.into_iter().map(EF::from));
    Ok(values)
}

fn push_bytes_ext(values: &mut Vec<EF>, bytes: &[u8]) {
    push_u64_ext(values, bytes.len() as u64);
    values.extend(bytes.iter().copied().map(EF::from_u8));
}

fn push_u64_ext(values: &mut Vec<EF>, value: u64) {
    values.extend(value.to_le_bytes().into_iter().map(EF::from_u8));
}

fn validate_call(
    profile: &ReducedSwirlVaccProfile,
    call: ReducedSwirlVaccCall,
) -> Result<(), &'static str> {
    let capacity = profile.input_arity - call.prior_count;
    if call.input_arity != profile.input_arity
        || call.prior_count != usize::from(call.step != 0)
        || call.fresh_count == 0
        || call.fresh_count > capacity
    {
        return Err("reduced-SWIRL VACC call");
    }
    Ok(())
}

/// Host-side security boundary for one fresh batch.  It rejects any attempt
/// to replace the original SWIRL root tuple by a synthetic scalar root.
pub fn validate_reduced_swirl_fresh_batch<AccProof>(
    profile: &ReducedSwirlVaccProfile,
    sources: &[RecursiveReducedSwirlClaim],
    step_proof: &ReducedWarpVaccStepProof<
        EF,
        Digest,
        StackedRsBatchOpeningProof<F, Digest>,
        AccProof,
        StackedRsFreshCommitment<EF, Digest>,
    >,
    verification: &WarpVaccStepVerification<
        EF,
        Digest,
        StackedRsBatchOpeningVerification<F, EF, Digest>,
        MerkleBatchOpeningVerification<EF, Digest>,
    >,
) -> Result<(), &'static str> {
    profile.validate()?;
    let inner = step_proof.inner();
    if sources.is_empty()
        || sources.len() != inner.fresh_claims.len()
        || sources.len() != verification.fresh_authentication.len()
    {
        return Err("reduced-SWIRL fresh inventory");
    }
    for (source_index, ((source, fresh), opening)) in sources
        .iter()
        .zip(&inner.fresh_claims)
        .zip(&verification.fresh_authentication)
        .enumerate()
    {
        let expected = StackedRsFreshCommitment {
            roots: source.roots.clone(),
            widths: source.widths.clone(),
            l_skip: profile.source.l_skip,
            native_log_message_len: profile.source.log_message_len(),
            log_message_len: profile.source.log_message_len(),
            log_codeword_len: profile.source.log_codeword_len(),
            rows_per_query: profile.source.rows_per_query(),
            theta: source.theta,
        };
        let mut expected_beta = source.beta.clone();
        expected_beta.push(source.eta);
        if fresh.commitment != expected
            || fresh.alpha.len() != profile.source.log_codeword_len()
            || fresh.alpha.iter().any(|value| *value != EF::ZERO)
            || fresh.beta != expected_beta
            || fresh.mu != source.mu
            || fresh.eta != EF::ZERO
            || !opening.recorded
            || opening.theta != source.theta
            || opening.values.len() != profile.num_shift_queries
            || opening.roots.len() != source.roots.len()
            || opening
                .roots
                .iter()
                .enumerate()
                .any(|(root_ordinal, root)| {
                    root.root != source.roots[root_ordinal]
                        || root.width as usize != source.widths[root_ordinal]
                        || root.multiproof.expected_root != source.roots[root_ordinal]
                })
        {
            let _ = source_index;
            return Err("reduced-SWIRL authoritative fresh claim");
        }
    }
    if verification.shifts.len() != profile.num_shift_queries {
        return Err("reduced-SWIRL shift count");
    }
    for (shift, shift_record) in verification.shifts.iter().enumerate() {
        if shift_record.fresh_answers.len() != sources.len()
            || verification
                .fresh_authentication
                .iter()
                .enumerate()
                .any(|(source, opening)| {
                    opening.flat_indices.get(shift).copied() != Some(shift_record.index)
                        || opening.values.get(shift).copied()
                            != Some(shift_record.fresh_answers[source])
                })
        {
            return Err("reduced-SWIRL authenticated WARP shifts");
        }
    }
    Ok(())
}

/// One source's retained checkpoint and VM boundary, used by the canonical
/// source-entry hash.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlSourceAuthorityRecord {
    pub segment_index: u32,
    pub common_main_root: Digest,
    pub trace_layout_digest: Digest,
    pub pending_claim_digest: Digest,
    pub checkpoint_tidx: usize,
    pub checkpoint_state: [F; POSEIDON2_WIDTH],
    pub program_commitment: Digest,
    pub initial_pc: F,
    pub initial_root: Digest,
    pub final_pc: F,
    pub final_root: Digest,
    pub exit_code: F,
    pub is_terminate: F,
}

#[derive(Clone, Copy)]
enum ReducedSwirlDigestObservation {
    Field(F),
    Digest(Digest),
    Extension(EF),
}

/// Exact SDK `AuthoritativeSwirlConstrainedRsClaim::digest` oracle.  This is
/// also the host input used by [`ReducedSwirlSourceDigestAir`].
pub fn reduced_swirl_authoritative_claim_digest(
    profile: &ReducedSwirlVaccProfile,
    claim: &RecursiveReducedSwirlClaim,
) -> Result<Digest, &'static str> {
    profile.validate()?;
    if claim.roots.is_empty()
        || claim.roots.len() > profile.source.maximum_roots_per_source
        || claim.roots.len() != claim.widths.len()
        || claim.widths.iter().any(|&width| width == 0)
        || claim.alpha.len() != profile.source.log_codeword_len()
        || claim.alpha.iter().any(|&value| value != EF::ZERO)
        || claim.beta.len() != profile.source.log_message_len()
    {
        return Err("reduced-SWIRL authoritative claim shape");
    }
    let mut observations = Vec::new();
    push_digest_usize(&mut observations, claim.roots.len())?;
    observations.extend(
        claim
            .roots
            .iter()
            .copied()
            .map(ReducedSwirlDigestObservation::Digest),
    );
    push_digest_usize(&mut observations, claim.widths.len())?;
    for &width in &claim.widths {
        push_digest_usize(&mut observations, width)?;
    }
    for value in [
        profile.source.l_skip,
        profile.source.n_stack,
        profile.source.log_blowup,
        profile.source.log_commit_rows_per_query,
    ] {
        push_digest_usize(&mut observations, value)?;
    }
    observations.push(ReducedSwirlDigestObservation::Extension(claim.theta));
    push_digest_extension_slice(&mut observations, &claim.alpha)?;
    observations.push(ReducedSwirlDigestObservation::Extension(claim.mu));
    push_digest_extension_slice(&mut observations, &claim.beta)?;
    observations.push(ReducedSwirlDigestObservation::Extension(claim.eta));
    Ok(reduced_swirl_digest_observations(REDUCED_SWIRL_SOURCE_DIGEST_TAG ^ 0x40, &observations).0)
}

/// Exact SDK `ReducedSwirlSourceManifestPrefix::digest_with_claim` oracle.
///
/// Schedule coordinates and retained checkpoints remain constrained by the
/// source-slot and authority buses, respectively, but are deliberately not
/// duplicated in the canonical manifest preimage.
pub fn reduced_swirl_source_entry_digest(
    profile: &ReducedSwirlVaccProfile,
    source: usize,
    call: ReducedSwirlVaccCall,
    slot: usize,
    claim: &RecursiveReducedSwirlClaim,
    authority: &ReducedSwirlSourceAuthorityRecord,
) -> Result<Digest, &'static str> {
    validate_call(profile, call)?;
    if source != call.source_start + slot || slot >= call.fresh_count || source > u32::MAX as usize
    {
        return Err("reduced-SWIRL source-entry shape");
    }
    let claim_digest = reduced_swirl_authoritative_claim_digest(profile, claim)?;
    let mut observations = Vec::new();
    for value in [source as u32, authority.segment_index] {
        push_digest_u32(&mut observations, value);
    }
    for digest in [
        authority.common_main_root,
        authority.trace_layout_digest,
        authority.pending_claim_digest,
        claim_digest,
    ] {
        observations.push(ReducedSwirlDigestObservation::Digest(digest));
    }
    observations.push(ReducedSwirlDigestObservation::Digest(
        authority.program_commitment,
    ));
    observations.push(ReducedSwirlDigestObservation::Field(authority.initial_pc));
    observations.push(ReducedSwirlDigestObservation::Digest(
        authority.initial_root,
    ));
    observations.push(ReducedSwirlDigestObservation::Field(authority.final_pc));
    observations.push(ReducedSwirlDigestObservation::Digest(authority.final_root));
    observations.push(ReducedSwirlDigestObservation::Field(authority.exit_code));
    observations.push(ReducedSwirlDigestObservation::Field(authority.is_terminate));
    Ok(reduced_swirl_digest_observations(REDUCED_SWIRL_SOURCE_DIGEST_TAG ^ 0x70, &observations).0)
}

fn push_digest_u32(observations: &mut Vec<ReducedSwirlDigestObservation>, value: u32) {
    observations.push(ReducedSwirlDigestObservation::Field(F::from_u32(
        value & 0xffff,
    )));
    observations.push(ReducedSwirlDigestObservation::Field(F::from_u32(
        value >> 16,
    )));
}

fn push_digest_usize(
    observations: &mut Vec<ReducedSwirlDigestObservation>,
    value: usize,
) -> Result<(), &'static str> {
    push_digest_u32(
        observations,
        u32::try_from(value).map_err(|_| "reduced-SWIRL digest integer width")?,
    );
    Ok(())
}

fn push_digest_extension_slice(
    observations: &mut Vec<ReducedSwirlDigestObservation>,
    values: &[EF],
) -> Result<(), &'static str> {
    push_digest_usize(observations, values.len())?;
    observations.extend(
        values
            .iter()
            .copied()
            .map(ReducedSwirlDigestObservation::Extension),
    );
    Ok(())
}

fn authoritative_claim_observations(
    profile: &ReducedSwirlVaccProfile,
    claim: &RecursiveReducedSwirlClaim,
) -> Result<Vec<ReducedSwirlDigestObservation>, &'static str> {
    if claim.roots.is_empty()
        || claim.roots.len() > profile.source.maximum_roots_per_source
        || claim.roots.len() != claim.widths.len()
        || claim.widths.iter().any(|&width| width == 0)
        || claim.alpha.len() != profile.source.log_codeword_len()
        || claim.alpha.iter().any(|&value| value != EF::ZERO)
        || claim.beta.len() != profile.source.log_message_len()
    {
        return Err("reduced-SWIRL authoritative claim shape");
    }
    let mut observations = Vec::new();
    push_digest_usize(&mut observations, claim.roots.len())?;
    observations.extend(
        claim
            .roots
            .iter()
            .copied()
            .map(ReducedSwirlDigestObservation::Digest),
    );
    push_digest_usize(&mut observations, claim.widths.len())?;
    for &width in &claim.widths {
        push_digest_usize(&mut observations, width)?;
    }
    for value in [
        profile.source.l_skip,
        profile.source.n_stack,
        profile.source.log_blowup,
        profile.source.log_commit_rows_per_query,
    ] {
        push_digest_usize(&mut observations, value)?;
    }
    observations.push(ReducedSwirlDigestObservation::Extension(claim.theta));
    push_digest_extension_slice(&mut observations, &claim.alpha)?;
    observations.push(ReducedSwirlDigestObservation::Extension(claim.mu));
    push_digest_extension_slice(&mut observations, &claim.beta)?;
    observations.push(ReducedSwirlDigestObservation::Extension(claim.eta));
    Ok(observations)
}

fn source_entry_observations(
    source: usize,
    call: ReducedSwirlVaccCall,
    slot: usize,
    authority: &ReducedSwirlSourceAuthorityRecord,
    claim_digest: Digest,
) -> Result<Vec<ReducedSwirlDigestObservation>, &'static str> {
    if source != call.source_start + slot || slot >= call.fresh_count || source > u32::MAX as usize
    {
        return Err("reduced-SWIRL source-entry integer width");
    }
    let mut observations = Vec::new();
    for value in [source as u32, authority.segment_index] {
        push_digest_u32(&mut observations, value);
    }
    for digest in [
        authority.common_main_root,
        authority.trace_layout_digest,
        authority.pending_claim_digest,
        claim_digest,
    ] {
        observations.push(ReducedSwirlDigestObservation::Digest(digest));
    }
    observations.push(ReducedSwirlDigestObservation::Digest(
        authority.program_commitment,
    ));
    observations.push(ReducedSwirlDigestObservation::Field(authority.initial_pc));
    observations.push(ReducedSwirlDigestObservation::Digest(
        authority.initial_root,
    ));
    observations.push(ReducedSwirlDigestObservation::Field(authority.final_pc));
    observations.push(ReducedSwirlDigestObservation::Digest(authority.final_root));
    observations.push(ReducedSwirlDigestObservation::Field(authority.exit_code));
    observations.push(ReducedSwirlDigestObservation::Field(authority.is_terminate));
    Ok(observations)
}

fn reduced_swirl_digest_observations(
    domain_tag: u32,
    observations: &[ReducedSwirlDigestObservation],
) -> (Digest, TranscriptLog<F, [F; POSEIDON2_WIDTH]>) {
    let mut transcript = default_duplex_sponge_recorder();
    for value in [
        F::from_u32(domain_tag),
        F::from_u32(REDUCED_SWIRL_VACC_PROTOCOL_VERSION),
        F::from_u32(observations.len() as u32 & 0xffff),
        F::from_u32(observations.len() as u32 >> 16),
    ] {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(&mut transcript, value);
    }
    for observation in observations {
        match observation {
            ReducedSwirlDigestObservation::Field(value) => {
                <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
                    &mut transcript,
                    F::from_u32(REDUCED_SWIRL_FIELD_TAG),
                );
                <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
                    &mut transcript,
                    *value,
                );
            }
            ReducedSwirlDigestObservation::Digest(digest) => {
                <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
                    &mut transcript,
                    F::from_u32(REDUCED_SWIRL_DIGEST_TAG),
                );
                for &value in digest {
                    <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
                        &mut transcript,
                        value,
                    );
                }
            }
            ReducedSwirlDigestObservation::Extension(value) => {
                <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
                    &mut transcript,
                    F::from_u32(REDUCED_SWIRL_EXTENSION_TAG),
                );
                for &limb in value.as_basis_coefficients_slice() {
                    <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
                        &mut transcript,
                        limb,
                    );
                }
            }
        }
    }
    <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
        &mut transcript,
        F::from_u32(REDUCED_SWIRL_END_TAG),
    );
    let digest = core::array::from_fn(|_| {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::sample(&mut transcript)
    });
    let log = TranscriptHistory::into_log(transcript);
    (digest, log)
}

fn reduced_swirl_source_digest_logs(
    profile: &ReducedSwirlVaccProfile,
    source: usize,
    call: ReducedSwirlVaccCall,
    slot: usize,
    claim: &RecursiveReducedSwirlClaim,
    authority: &ReducedSwirlSourceAuthorityRecord,
) -> Result<
    (
        Digest,
        TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
        Digest,
        TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    ),
    &'static str,
> {
    let claim_digest = reduced_swirl_authoritative_claim_digest(profile, claim)?;
    let entry_digest =
        reduced_swirl_source_entry_digest(profile, source, call, slot, claim, authority)?;

    // Rebuild the two observation vectors only for trace logs.  Keeping this
    // oracle beside the public digest functions makes transcript differential
    // tests independent of any AIR witness generator.
    let claim_observations = authoritative_claim_observations(profile, claim)?;
    let entry_observations =
        source_entry_observations(source, call, slot, authority, claim_digest)?;
    let (logged_claim, claim_log) = reduced_swirl_digest_observations(
        REDUCED_SWIRL_SOURCE_DIGEST_TAG ^ 0x40,
        &claim_observations,
    );
    let (logged_entry, entry_log) = reduced_swirl_digest_observations(
        REDUCED_SWIRL_SOURCE_DIGEST_TAG ^ 0x70,
        &entry_observations,
    );
    if logged_claim != claim_digest || logged_entry != entry_digest {
        return Err("reduced-SWIRL digest log mismatch");
    }
    Ok((claim_digest, claim_log, entry_digest, entry_log))
}

/// Exact independent-manifest transcript from SDK protocol v2.  The domain
/// bytes are raw observations (there is deliberately no byte-count prefix),
/// followed by scalar protocol/count/index values and eight scalar samples.
pub fn reduced_swirl_manifest_digest(entry_digests: &[Digest]) -> Result<Digest, &'static str> {
    if entry_digests.is_empty()
        || entry_digests
            .iter()
            .any(|digest| digest.iter().all(|value| *value == F::ZERO))
    {
        return Err("reduced-SWIRL manifest entries");
    }
    let mut transcript = default_duplex_sponge_recorder();
    for &byte in REDUCED_SWIRL_VACC_MANIFEST_TAG {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
            &mut transcript,
            F::from_u8(byte),
        );
    }
    <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
        &mut transcript,
        F::from_u32(REDUCED_SWIRL_VACC_PROTOCOL_VERSION),
    );
    <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
        &mut transcript,
        F::from_u32(u32::try_from(entry_digests.len()).map_err(|_| "source manifest count")?),
    );
    for (source, digest) in entry_digests.iter().enumerate() {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
            &mut transcript,
            F::from_u32(u32::try_from(source).map_err(|_| "source manifest index")?),
        );
        for &limb in digest {
            <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(&mut transcript, limb);
        }
    }
    Ok(core::array::from_fn(|_| {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::sample(&mut transcript)
    }))
}

// Adapter AIRs and trace generators consume the standard VACC verifier's
// authenticated buses; no success bit
// or host-side verification result enters the statement.

fn observe_ext_expr<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Expr,
    tidx: &mut AB::Expr,
    value: [AB::Expr; D_EF],
    enabled: AB::Expr,
) {
    bus.observe_ext(builder, proof_idx, tidx.clone(), value, enabled.clone());
    *tidx = tidx.clone() + enabled * AB::Expr::from_usize(D_EF);
}

fn observe_base_expr<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Expr,
    tidx: &mut AB::Expr,
    value: AB::Expr,
    is_sample: bool,
    enabled: AB::Expr,
) {
    if is_sample {
        bus.sample(builder, proof_idx, tidx.clone(), value, enabled.clone());
    } else {
        bus.observe(builder, proof_idx, tidx.clone(), value, enabled.clone());
    }
    *tidx = tidx.clone() + enabled;
}

fn constrain_bits<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    value: AB::Var,
    bits: &[AB::Var],
    enabled: AB::Expr,
) where
    AB::Var: Copy,
{
    let mut reconstructed = AB::Expr::ZERO;
    let mut power = F::ONE;
    for &bit in bits {
        builder.assert_bool(bit);
        builder
            .when(AB::Expr::ONE - enabled.clone())
            .assert_zero(bit);
        reconstructed += AB::Expr::from(bit) * AB::Expr::from(power);
        power += power;
    }
    builder.when(enabled).assert_eq(value, reconstructed);
}

fn byte_from_bits<AB: AirBuilder<F = F>>(bits: &[AB::Var], byte: usize) -> AB::Expr
where
    AB::Var: Copy,
{
    let mut value = AB::Expr::ZERO;
    for bit in 0..8 {
        let index = byte * 8 + bit;
        if index < bits.len() {
            value += AB::Expr::from(bits[index]) * AB::Expr::from_u32(1 << bit);
        }
    }
    value
}

fn observe_u64_bits_air<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Expr,
    tidx: &mut AB::Expr,
    bits: &[AB::Var],
    enabled: AB::Expr,
) where
    AB::Var: Copy,
{
    for byte in 0..8 {
        observe_ext_expr(
            bus,
            builder,
            proof_idx.clone(),
            tidx,
            [
                byte_from_bits::<AB>(bits, byte),
                AB::Expr::ZERO,
                AB::Expr::ZERO,
                AB::Expr::ZERO,
            ],
            enabled.clone(),
        );
    }
}

fn observe_const_ext<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Expr,
    tidx: &mut AB::Expr,
    value: EF,
    enabled: AB::Expr,
) {
    let coefficients: &[F] = value.as_basis_coefficients_slice();
    observe_ext_expr(
        bus,
        builder,
        proof_idx,
        tidx,
        core::array::from_fn(|limb| AB::Expr::from(coefficients[limb])),
        enabled,
    );
}

fn observe_const_u64_ext<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Expr,
    tidx: &mut AB::Expr,
    value: u64,
    enabled: AB::Expr,
) {
    for byte in value.to_le_bytes() {
        observe_const_ext(
            bus,
            builder,
            proof_idx.clone(),
            tidx,
            EF::from_u8(byte),
            enabled.clone(),
        );
    }
}

fn observe_const_bytes_ext<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Expr,
    tidx: &mut AB::Expr,
    bytes: &[u8],
    with_length: bool,
    enabled: AB::Expr,
) {
    if with_length {
        observe_const_u64_ext(
            bus,
            builder,
            proof_idx.clone(),
            tidx,
            bytes.len() as u64,
            enabled.clone(),
        );
    }
    for &byte in bytes {
        observe_const_ext(
            bus,
            builder,
            proof_idx.clone(),
            tidx,
            EF::from_u8(byte),
            enabled.clone(),
        );
    }
}

fn fill_bits(target: &mut [F], value: usize) {
    for (bit, output) in target.iter_mut().enumerate() {
        *output = F::from_bool(((value >> bit) & 1) == 1);
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ReducedSwirlVaccHeaderCols<T> {
    pub active: T,
    pub source_count: T,
    pub source_count_bits: [T; REDUCED_SWIRL_SCHEDULE_BITS],
    pub end_tidx: T,
}

#[derive(ColumnsAir)]
#[columns_via(ReducedSwirlVaccHeaderCols<u8>)]
pub struct ReducedSwirlVaccHeaderAir {
    pub profile: ReducedSwirlVaccProfile,
    pub transcript_bus: TranscriptBus,
    pub end_bus: ReducedSwirlVaccHeaderEndBus,
}

impl BaseAirWithPublicValues<F> for ReducedSwirlVaccHeaderAir {}
impl PartitionedBaseAir<F> for ReducedSwirlVaccHeaderAir {}
impl BaseAir<F> for ReducedSwirlVaccHeaderAir {
    fn width(&self) -> usize {
        ReducedSwirlVaccHeaderCols::<F>::width()
    }
}

impl<AB> Air<AB> for ReducedSwirlVaccHeaderAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.profile.validate().is_ok());
        let main = builder.main();
        let row = main.row_slice(0).expect("reduced-SWIRL header row");
        let local: &ReducedSwirlVaccHeaderCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        let active = AB::Expr::from(local.active);
        constrain_bits(
            builder,
            local.source_count,
            &local.source_count_bits,
            active.clone(),
        );
        let proof_idx = AB::Expr::ZERO;
        let mut tidx = AB::Expr::ZERO;
        observe_const_bytes_ext(
            &self.transcript_bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            REDUCED_SWIRL_VACC_HEADER_TAG,
            true,
            active.clone(),
        );
        observe_const_u64_ext(
            &self.transcript_bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            u64::from(REDUCED_SWIRL_VACC_PROTOCOL_VERSION),
            active.clone(),
        );
        observe_u64_bits_air(
            &self.transcript_bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            &local.source_count_bits,
            active.clone(),
        );
        for value in [
            self.profile.source.log_message_len(),
            self.profile.source.log_codeword_len(),
            self.profile.source.rows_per_query(),
            self.profile.input_arity,
            self.profile.num_ood,
            self.profile.num_shift_queries,
            self.profile.batching_arity,
            self.profile.family_target_bits,
        ] {
            observe_const_u64_ext(
                &self.transcript_bus,
                builder,
                proof_idx.clone(),
                &mut tidx,
                value as u64,
                active.clone(),
            );
        }
        observe_const_bytes_ext(
            &self.transcript_bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            &self.profile.source_domain,
            true,
            active.clone(),
        );
        for &value in self
            .profile
            .relation_binding
            .iter()
            .chain(&self.profile.code_binding)
        {
            observe_const_ext(
                &self.transcript_bus,
                builder,
                proof_idx.clone(),
                &mut tidx,
                value,
                active.clone(),
            );
        }
        builder.when(active.clone()).assert_eq(local.end_tidx, tidx);
        self.end_bus.send(
            builder,
            ReducedSwirlVaccHeaderEndMessage {
                source_count: local.source_count.into(),
                end_tidx: local.end_tidx.into(),
            },
            local.active,
        );
    }
}

pub fn generate_reduced_swirl_vacc_header_trace(
    profile: &ReducedSwirlVaccProfile,
    source_count: usize,
) -> Result<RowMajorMatrix<F>, &'static str> {
    let elements = reduced_swirl_vacc_header_elements(profile, source_count)?;
    let width = ReducedSwirlVaccHeaderCols::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut ReducedSwirlVaccHeaderCols<F> = values.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.source_count = F::from_usize(source_count);
    fill_bits(&mut cols.source_count_bits, source_count);
    cols.end_tidx = F::from_usize(elements.len() * D_EF);
    Ok(RowMajorMatrix::new(values, width))
}

pub fn generate_reduced_swirl_vacc_optional_header_trace(
    profile: &ReducedSwirlVaccProfile,
    total_source_count: usize,
    is_genesis: bool,
) -> Result<RowMajorMatrix<F>, &'static str> {
    if is_genesis {
        generate_reduced_swirl_vacc_header_trace(profile, total_source_count)
    } else {
        Ok(RowMajorMatrix::new(
            F::zero_vec(ReducedSwirlVaccHeaderCols::<F>::width()),
            ReducedSwirlVaccHeaderCols::<F>::width(),
        ))
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ReducedSwirlVaccBatchCols<T> {
    pub active: T,
    pub local_proof_idx: T,
    pub step: T,
    pub source_start: T,
    pub fresh_count: T,
    pub prior_count: T,
    pub input_arity: T,
    pub batch_start_tidx: T,
    pub vacc_start_tidx: T,
    pub vacc_end_tidx: T,
    pub step_bits: [T; REDUCED_SWIRL_SCHEDULE_BITS],
    pub source_start_bits: [T; REDUCED_SWIRL_SCHEDULE_BITS],
    pub fresh_count_bits: [T; REDUCED_SWIRL_SCHEDULE_BITS],
}

#[derive(ColumnsAir)]
#[columns_via(ReducedSwirlVaccBatchCols<u8>)]
pub struct ReducedSwirlVaccBatchAir {
    pub transcript_bus: TranscriptBus,
    pub call_bus: ReducedSwirlVaccCallBus,
}

impl BaseAirWithPublicValues<F> for ReducedSwirlVaccBatchAir {}
impl PartitionedBaseAir<F> for ReducedSwirlVaccBatchAir {}
impl BaseAir<F> for ReducedSwirlVaccBatchAir {
    fn width(&self) -> usize {
        ReducedSwirlVaccBatchCols::<F>::width()
    }
}

impl<AB> Air<AB> for ReducedSwirlVaccBatchAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("reduced-SWIRL batch row");
        let next_row = main.row_slice(1).expect("reduced-SWIRL batch next row");
        let local: &ReducedSwirlVaccBatchCols<AB::Var> = (*row).borrow();
        let next: &ReducedSwirlVaccBatchCols<AB::Var> = (*next_row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_zero(local.local_proof_idx);
        builder
            .when_transition()
            .assert_zero(next.active * (AB::Expr::ONE - local.active));
        builder.when_transition().when(next.active).assert_eq(
            next.local_proof_idx,
            AB::Expr::from(local.local_proof_idx) + AB::Expr::ONE,
        );
        constrain_bits(builder, local.step, &local.step_bits, local.active.into());
        constrain_bits(
            builder,
            local.source_start,
            &local.source_start_bits,
            local.active.into(),
        );
        constrain_bits(
            builder,
            local.fresh_count,
            &local.fresh_count_bits,
            local.active.into(),
        );
        self.call_bus.lookup_key(
            builder,
            ReducedSwirlVaccCallMessage {
                step: local.step.into(),
                source_start: local.source_start.into(),
                fresh_count: local.fresh_count.into(),
                prior_count: local.prior_count.into(),
                input_arity: local.input_arity.into(),
                batch_start_tidx: local.batch_start_tidx.into(),
                vacc_start_tidx: local.vacc_start_tidx.into(),
                vacc_end_tidx: local.vacc_end_tidx.into(),
            },
            local.active,
        );
        let proof_idx = AB::Expr::from(local.local_proof_idx);
        let mut tidx = AB::Expr::from(local.batch_start_tidx);
        observe_const_bytes_ext(
            &self.transcript_bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            REDUCED_SWIRL_VACC_SOURCE_BATCH_TAG,
            true,
            local.active.into(),
        );
        for bits in [
            &local.step_bits[..],
            &local.source_start_bits[..],
            &local.fresh_count_bits[..],
        ] {
            observe_u64_bits_air(
                &self.transcript_bus,
                builder,
                proof_idx.clone(),
                &mut tidx,
                bits,
                local.active.into(),
            );
        }
        builder.when(local.active).assert_eq(
            local.vacc_start_tidx,
            tidx + AB::Expr::from(local.fresh_count) * AB::Expr::from_usize(DIGEST_SIZE * D_EF),
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ReducedSwirlVaccPrefixCols<T> {
    pub active: T,
    pub local_proof_idx: T,
    pub step: T,
    pub source_start: T,
    pub fresh_count: T,
    pub prior_count: T,
    pub input_arity: T,
    pub batch_start_tidx: T,
    pub vacc_start_tidx: T,
    pub vacc_end_tidx: T,
    pub step_bits: [T; REDUCED_SWIRL_SCHEDULE_BITS],
    pub fresh_count_bits: [T; REDUCED_SWIRL_SCHEDULE_BITS],
    pub prior_count_bits: [T; REDUCED_SWIRL_SCHEDULE_BITS],
}

#[derive(ColumnsAir)]
#[columns_via(ReducedSwirlVaccPrefixCols<u8>)]
pub struct ReducedSwirlVaccPrefixAir {
    pub transcript_bus: TranscriptBus,
    pub phase_cursor_bus: NativeVaccPhaseCursorBus,
    pub call_bus: ReducedSwirlVaccCallBus,
    pub profile: ReducedSwirlVaccProfile,
    /// Whole-block traces begin at call zero. Bounded transition leaves use
    /// an authenticated nonzero global call while restarting only the local
    /// transcript namespace.
    pub require_genesis_first: bool,
}

impl BaseAirWithPublicValues<F> for ReducedSwirlVaccPrefixAir {}
impl PartitionedBaseAir<F> for ReducedSwirlVaccPrefixAir {}
impl BaseAir<F> for ReducedSwirlVaccPrefixAir {
    fn width(&self) -> usize {
        ReducedSwirlVaccPrefixCols::<F>::width()
    }
}

impl<AB> Air<AB> for ReducedSwirlVaccPrefixAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.profile.validate().is_ok());
        let main = builder.main();
        let row = main.row_slice(0).expect("reduced-SWIRL prefix row");
        let next_row = main.row_slice(1).expect("reduced-SWIRL prefix next row");
        let local: &ReducedSwirlVaccPrefixCols<AB::Var> = (*row).borrow();
        let next: &ReducedSwirlVaccPrefixCols<AB::Var> = (*next_row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_zero(local.local_proof_idx);
        if self.require_genesis_first {
            builder.when_first_row().assert_zero(local.step);
            builder.when_first_row().assert_zero(local.prior_count);
        }
        builder
            .when_transition()
            .assert_zero(next.active * (AB::Expr::ONE - local.active));
        builder
            .when_transition()
            .when(next.active)
            .assert_eq(next.step, AB::Expr::from(local.step) + AB::Expr::ONE);
        builder.when_transition().when(next.active).assert_eq(
            next.local_proof_idx,
            AB::Expr::from(local.local_proof_idx) + AB::Expr::ONE,
        );
        builder
            .when_transition()
            .when(next.active)
            .assert_one(next.prior_count);
        constrain_bits(builder, local.step, &local.step_bits, local.active.into());
        constrain_bits(
            builder,
            local.fresh_count,
            &local.fresh_count_bits,
            local.active.into(),
        );
        constrain_bits(
            builder,
            local.prior_count,
            &local.prior_count_bits,
            local.active.into(),
        );
        builder.when(local.active).assert_eq(
            local.input_arity,
            AB::Expr::from_usize(self.profile.input_arity),
        );
        self.call_bus.lookup_key(
            builder,
            ReducedSwirlVaccCallMessage {
                step: local.step.into(),
                source_start: local.source_start.into(),
                fresh_count: local.fresh_count.into(),
                prior_count: local.prior_count.into(),
                input_arity: local.input_arity.into(),
                batch_start_tidx: local.batch_start_tidx.into(),
                vacc_start_tidx: local.vacc_start_tidx.into(),
                vacc_end_tidx: local.vacc_end_tidx.into(),
            },
            local.active,
        );
        let placeholder = ReducedSwirlVaccCall {
            step: 0,
            source_start: 0,
            fresh_count: 1,
            prior_count: 0,
            input_arity: self.profile.input_arity,
        };
        let constants = reduced_swirl_vacc_protocol_prefix_elements(&self.profile, placeholder)
            .expect("valid reduced-SWIRL prefix key");
        let constant_len = constants.len() - 3 * 8;
        let proof_idx = AB::Expr::from(local.local_proof_idx);
        let mut tidx = AB::Expr::from(local.vacc_start_tidx);
        for &value in &constants[..constant_len] {
            observe_const_ext(
                &self.transcript_bus,
                builder,
                proof_idx.clone(),
                &mut tidx,
                value,
                local.active.into(),
            );
        }
        for bits in [
            &local.step_bits[..],
            &local.fresh_count_bits[..],
            &local.prior_count_bits[..],
        ] {
            observe_u64_bits_air(
                &self.transcript_bus,
                builder,
                proof_idx.clone(),
                &mut tidx,
                bits,
                local.active.into(),
            );
        }
        // The aggregate itself emits and constrains the complete protocol
        // prefix.  Only its end is handed to the ordinary VACC semantic
        // cursor; there is no separate authority AIR consuming a start token.
        self.phase_cursor_bus.send(
            builder,
            NativeVaccPhaseCursorMessage {
                proof_idx,
                boundary: AB::Expr::ONE,
                tidx,
            },
            local.active,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ReducedSwirlVaccScheduleCols<T> {
    pub active: T,
    /// End of the current physical recursive-leaf chunk. This differs from
    /// `is_last`, which is the terminal call of the complete block.
    pub chunk_last: T,
    pub is_last: T,
    pub local_proof_idx: T,
    pub step: T,
    pub source_count: T,
    pub source_start: T,
    pub fresh_count: T,
    pub prior_count: T,
    pub batch_start_tidx: T,
    pub vacc_start_tidx: T,
    pub vacc_end_tidx: T,
    pub fresh_selector: [T; REDUCED_SWIRL_VACC_MAX_INPUT_ARITY + 1],
    pub start_sample_count: T,
    pub start_state: [T; POSEIDON2_WIDTH],
    pub end_sample_count: T,
    pub end_state: [T; POSEIDON2_WIDTH],
    pub prior_root: [T; DIGEST_SIZE],
    pub output_root: [T; DIGEST_SIZE],
    pub prior_digest: [T; DIGEST_SIZE],
    pub output_digest: [T; DIGEST_SIZE],
}

#[derive(ColumnsAir)]
#[columns_via(ReducedSwirlVaccScheduleCols<u8>)]
pub struct ReducedSwirlVaccScheduleAir {
    pub profile: ReducedSwirlVaccProfile,
    pub call_bus: ReducedSwirlVaccCallBus,
    pub slot_bus: ReducedSwirlVaccSourceSlotBus,
    /// Setup-fixed standard-WARP input layout. The schedule, rather than a
    /// host-provided mode bit, determines the active fresh prefix, the prior
    /// slot immediately following it, and all trailing zero-padding slots.
    pub input_slot_bus: NativeInputSlotLayoutBus,
    pub fresh_count_bus: NativeFreshCountBus,
    pub header_end_bus: ReducedSwirlVaccHeaderEndBus,
    pub vacc_end_bus: NativeStandardVaccEndBus,
    pub vacc_root_bus: NativeStandardVaccRootBus,
    pub vacc_digest_bus: NativeStandardVaccDigestBus,
    pub checkpoint_bus: CertifiedTranscriptCheckpointBus,
    pub resume_bus: ResumeTranscriptStateBus,
    pub transcript_end_index_bus: crate::bus::TranscriptEndIndexBus,
    pub chain_end_bus: ReducedSwirlVaccChainEndBus,
    /// Whole-block mode fixes the first row to the bootstrap call. Bounded
    /// leaves instead bind their global start through `transition_end_bus` and
    /// recursive state continuity.
    pub require_complete_chain: bool,
    /// A transition leaf ends its physical transcript at `vacc_end_tidx` even
    /// when the call is globally final; footer and terminal suffixes are
    /// verified once by the finalizer.
    pub leaf_physical_end: bool,
    pub transition_end_bus: Option<ReducedSwirlVaccTransitionEndBus>,
}

impl BaseAirWithPublicValues<F> for ReducedSwirlVaccScheduleAir {}
impl PartitionedBaseAir<F> for ReducedSwirlVaccScheduleAir {}
impl BaseAir<F> for ReducedSwirlVaccScheduleAir {
    fn width(&self) -> usize {
        ReducedSwirlVaccScheduleCols::<F>::width()
    }
}

impl<AB> Air<AB> for ReducedSwirlVaccScheduleAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.profile.validate().is_ok());
        let main = builder.main();
        let row = main.row_slice(0).expect("reduced-SWIRL schedule row");
        let next_row = main.row_slice(1).expect("reduced-SWIRL schedule next row");
        let local: &ReducedSwirlVaccScheduleCols<AB::Var> = (*row).borrow();
        let next: &ReducedSwirlVaccScheduleCols<AB::Var> = (*next_row).borrow();
        builder.assert_bool(local.active);
        builder.assert_bool(local.chunk_last);
        builder.assert_bool(local.is_last);
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_zero(local.local_proof_idx);
        if self.require_complete_chain {
            builder.when_first_row().assert_zero(local.step);
            builder.when_first_row().assert_zero(local.source_start);
            builder.when_first_row().assert_zero(local.prior_count);
        }
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.chunk_last);
        builder
            .when_transition()
            .assert_eq(next.active, AB::Expr::from(local.active) - local.chunk_last);
        let mut transition = builder.when_transition();
        let mut continued = transition.when(next.active);
        continued.assert_eq(next.step, AB::Expr::from(local.step) + AB::Expr::ONE);
        continued.assert_eq(
            next.local_proof_idx,
            AB::Expr::from(local.local_proof_idx) + AB::Expr::ONE,
        );
        continued.assert_eq(
            next.source_start,
            AB::Expr::from(local.source_start) + local.fresh_count,
        );
        continued.assert_eq(next.source_count, local.source_count);
        continued.assert_one(next.prior_count);
        continued.assert_eq(next.batch_start_tidx, local.vacc_end_tidx);
        for limb in 0..POSEIDON2_WIDTH {
            continued.assert_eq(next.start_state[limb], local.end_state[limb]);
        }
        continued.assert_eq(next.start_sample_count, local.end_sample_count);
        for limb in 0..DIGEST_SIZE {
            continued.assert_eq(next.prior_root[limb], local.output_root[limb]);
            continued.assert_eq(next.prior_digest[limb], local.output_digest[limb]);
        }

        let active = AB::Expr::from(local.active);
        let mut selector_sum = AB::Expr::ZERO;
        let mut selected_fresh = AB::Expr::ZERO;
        for (count, &selector) in local.fresh_selector.iter().enumerate() {
            builder.assert_bool(selector);
            selector_sum += selector;
            selected_fresh += AB::Expr::from(selector) * AB::Expr::from_usize(count);
            let forbidden_bootstrap = count == 0 || count > self.profile.input_arity;
            let forbidden_continuation = count == 0 || count + 1 > self.profile.input_arity;
            if forbidden_bootstrap {
                builder
                    .when(active.clone() * (AB::Expr::ONE - local.prior_count))
                    .assert_zero(selector);
            }
            if forbidden_continuation {
                builder
                    .when(active.clone() * local.prior_count)
                    .assert_zero(selector);
            }
        }
        builder.assert_eq(selector_sum, active.clone());
        builder
            .when(active.clone())
            .assert_eq(local.fresh_count, selected_fresh);
        builder.when(active.clone()).assert_eq(
            AB::Expr::from_usize(self.profile.input_arity),
            AB::Expr::from(local.fresh_count)
                + local.prior_count
                + local.is_last
                    * (AB::Expr::from_usize(self.profile.input_arity)
                        - local.prior_count
                        - local.fresh_count),
        );
        builder.when(active.clone() * local.is_last).assert_eq(
            AB::Expr::from(local.source_start) + local.fresh_count,
            local.source_count,
        );

        self.call_bus.add_key_with_lookups(
            builder,
            ReducedSwirlVaccCallMessage {
                step: local.step.into(),
                source_start: local.source_start.into(),
                fresh_count: local.fresh_count.into(),
                prior_count: local.prior_count.into(),
                input_arity: AB::Expr::from_usize(self.profile.input_arity),
                batch_start_tidx: local.batch_start_tidx.into(),
                vacc_start_tidx: local.vacc_start_tidx.into(),
                vacc_end_tidx: local.vacc_end_tidx.into(),
            },
            active.clone() * AB::Expr::from_usize(2),
        );
        self.fresh_count_bus.add_key_with_lookups(
            builder,
            NativeFreshCountMessage {
                count: local.fresh_count.into(),
            },
            active.clone(),
        );
        for slot in 0..REDUCED_SWIRL_VACC_MAX_INPUT_ARITY {
            let slot_active = local
                .fresh_selector
                .iter()
                .enumerate()
                .skip(slot + 1)
                .fold(AB::Expr::ZERO, |sum, (_, &selector)| sum + selector);
            self.slot_bus.add_key_with_lookups(
                builder,
                ReducedSwirlVaccSourceSlotMessage {
                    source: AB::Expr::from(local.source_start) + AB::Expr::from_usize(slot),
                    step: local.step.into(),
                    slot: AB::Expr::from_usize(slot),
                    fresh_count: local.fresh_count.into(),
                    prior_count: local.prior_count.into(),
                    input_arity: AB::Expr::from_usize(self.profile.input_arity),
                    batch_start_tidx: local.batch_start_tidx.into(),
                    vacc_start_tidx: local.vacc_start_tidx.into(),
                },
                slot_active,
            );
        }
        let claim_rows = self.profile.log_codeword_len() + self.profile.normalized_beta_len() + 2;
        let shift_rows = self.profile.num_shift_queries;
        let variant = AB::Expr::from(local.fresh_count)
            + local.prior_count * AB::Expr::from_usize(self.profile.input_arity + 1);
        for slot in 0..self.profile.input_arity {
            let fresh = local
                .fresh_selector
                .iter()
                .enumerate()
                .skip(slot + 1)
                .fold(AB::Expr::ZERO, |sum, (_, &selector)| sum + selector);
            let prior = local.prior_count * local.fresh_selector[slot];
            let dummy = active.clone() - fresh.clone() - prior.clone();
            // This table authenticates the slot kind, not the claim-value bus
            // multiplicity. A fresh slot is read once by the canonical claim
            // table and once per shift by `NativeShiftMergeAir`; its second
            // claim-value consumer (the reduced-source linker) does not read
            // the slot bus. A prior slot additionally passes through the
            // dynamic-prior adapter and accumulator projection, hence one
            // extra claim-sized and one extra shift-sized family.
            let lookup_count = fresh.clone() * AB::Expr::from_usize(claim_rows + shift_rows)
                + prior.clone() * AB::Expr::from_usize(2 * claim_rows + 2 * shift_rows)
                + dummy.clone() * AB::Expr::from_usize(claim_rows + shift_rows);
            self.input_slot_bus.add_key_with_lookups(
                builder,
                NativeInputSlotLayoutMessage {
                    variant: variant.clone(),
                    source: AB::Expr::from_usize(slot),
                    kind: [fresh, prior, dummy],
                },
                lookup_count,
            );
        }
        self.header_end_bus.receive(
            builder,
            ReducedSwirlVaccHeaderEndMessage {
                source_count: local.source_count.into(),
                end_tidx: local.batch_start_tidx.into(),
            },
            active.clone() * (AB::Expr::ONE - local.prior_count),
        );
        self.vacc_end_bus.lookup_key(
            builder,
            NativeStandardVaccEndMessage {
                proof_idx: local.local_proof_idx.into(),
                end_tidx: local.vacc_end_tidx.into(),
            },
            active.clone(),
        );
        for (kind, root, enabled) in [
            (1usize, local.prior_root, active.clone() * local.prior_count),
            (2usize, local.output_root, active.clone()),
        ] {
            self.vacc_root_bus.lookup_key(
                builder,
                NativeStandardVaccRootMessage {
                    proof_idx: local.local_proof_idx.into(),
                    kind: AB::Expr::from_usize(kind),
                    root: root.map(Into::into),
                },
                enabled,
            );
        }
        for (state, digest, enabled) in [
            (
                0usize,
                local.prior_digest,
                active.clone() * local.prior_count,
            ),
            (1usize, local.output_digest, active.clone()),
        ] {
            self.vacc_digest_bus.lookup_key(
                builder,
                NativeStandardVaccDigestMessage {
                    proof_idx: local.local_proof_idx.into(),
                    state: AB::Expr::from_usize(state),
                    digest: digest.map(Into::into),
                },
                enabled,
            );
        }
        self.checkpoint_bus.receive(
            builder,
            local.local_proof_idx,
            CertifiedTranscriptCheckpointMessage {
                kind: AB::Expr::ONE,
                tidx: local.vacc_end_tidx.into(),
                sample_count: local.end_sample_count.into(),
                state: local.end_state.map(Into::into),
            },
            active.clone(),
        );
        self.resume_bus.send(
            builder,
            local.local_proof_idx,
            ResumeTranscriptStateMessage {
                // Call zero owns the complete transcript prefix and starts at
                // the canonical `(tidx=0, state=0)`. Continuations are suffix
                // traces resumed at the preceding authenticated call end.
                tidx: (local.prior_count * local.batch_start_tidx).into(),
                state: local.start_state.map(Into::into),
            },
            active.clone(),
        );
        for limb in 0..POSEIDON2_WIDTH {
            builder
                .when(active.clone() * (AB::Expr::ONE - local.prior_count))
                .assert_zero(local.start_state[limb]);
        }
        builder
            .when(active.clone() * (AB::Expr::ONE - local.prior_count))
            .assert_zero(local.start_sample_count);
        for limb in 0..DIGEST_SIZE {
            builder
                .when(active.clone() * (AB::Expr::ONE - local.prior_count))
                .assert_zero(local.prior_root[limb]);
            builder
                .when(active.clone() * (AB::Expr::ONE - local.prior_count))
                .assert_zero(local.prior_digest[limb]);
        }
        self.transcript_end_index_bus.receive(
            builder,
            local.local_proof_idx,
            crate::bus::TranscriptEndIndexMessage {
                tidx: local.vacc_end_tidx.into(),
            },
            // Every non-final physical log ends at its VACC checkpoint. The
            // final log additionally owns the manifest-footer and terminal
            // Decide suffix, so its actual end index is consumed by the
            // terminal bridge instead.
            active.clone()
                * if self.leaf_physical_end {
                    AB::Expr::from(local.chunk_last)
                } else {
                    AB::Expr::ONE - local.is_last
                },
        );
        self.chain_end_bus.send(
            builder,
            ReducedSwirlVaccChainEndMessage {
                source_count: local.source_count.into(),
                call_count: AB::Expr::from(local.step) + AB::Expr::ONE,
                proof_idx: local.step.into(),
                footer_start_tidx: local.vacc_end_tidx.into(),
                final_accumulator_digest: local.output_digest.map(Into::into),
                final_accumulator_root: local.output_root.map(Into::into),
            },
            active * local.is_last,
        );
        if let Some(transition_end_bus) = self.transition_end_bus {
            transition_end_bus.send(
                builder,
                ReducedSwirlVaccTransitionEndMessage {
                    total_source_count: local.source_count.into(),
                    call_index: local.step.into(),
                    source_start: local.source_start.into(),
                    fresh_count: local.fresh_count.into(),
                    prior_count: local.prior_count.into(),
                    batch_start_tidx: local.batch_start_tidx.into(),
                    vacc_start_tidx: local.vacc_start_tidx.into(),
                    vacc_end_tidx: local.vacc_end_tidx.into(),
                    start_sample_count: local.start_sample_count.into(),
                    start_state: local.start_state.map(Into::into),
                    end_sample_count: local.end_sample_count.into(),
                    end_state: local.end_state.map(Into::into),
                    prior_root: local.prior_root.map(Into::into),
                    output_root: local.output_root.map(Into::into),
                    prior_digest: local.prior_digest.map(Into::into),
                    output_digest: local.output_digest.map(Into::into),
                    is_final: local.is_last.into(),
                },
                AB::Expr::from(local.active) * local.chunk_last,
            );
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlVaccCallRecord {
    pub call: ReducedSwirlVaccCall,
    pub batch_start_tidx: usize,
    pub vacc_start_tidx: usize,
    pub vacc_end_tidx: usize,
    pub start_sample_count: usize,
    pub start_state: [F; POSEIDON2_WIDTH],
    pub end_sample_count: usize,
    pub end_state: [F; POSEIDON2_WIDTH],
    pub prior_root: Option<Digest>,
    pub output_root: Digest,
    pub prior_digest: Option<Digest>,
    pub output_digest: Digest,
}

fn adapter_trace_height(rows: usize, minimum: usize) -> usize {
    rows.max(minimum).next_power_of_two()
}

pub fn generate_reduced_swirl_vacc_schedule_trace(
    profile: &ReducedSwirlVaccProfile,
    source_count: usize,
    calls: &[ReducedSwirlVaccCallRecord],
) -> Result<RowMajorMatrix<F>, &'static str> {
    profile.validate()?;
    let expected = reduced_swirl_vacc_schedule(source_count, profile.input_arity)?;
    if calls.len() != expected.len() || calls.is_empty() {
        return Err("reduced-SWIRL VACC schedule record count");
    }
    let header_end = reduced_swirl_vacc_header_elements(profile, source_count)?.len() * D_EF;
    let width = ReducedSwirlVaccScheduleCols::<F>::width();
    let height = adapter_trace_height(calls.len(), profile.maximum_call_count());
    let mut values = F::zero_vec(width * height);
    for (index, (record, &expected_call)) in calls.iter().zip(&expected).enumerate() {
        if record.call != expected_call
            || record.batch_start_tidx
                != if index == 0 {
                    header_end
                } else {
                    calls[index - 1].vacc_end_tidx
                }
            || record.vacc_start_tidx
                != record.batch_start_tidx
                    + (8 + REDUCED_SWIRL_VACC_SOURCE_BATCH_TAG.len()
                        + 3 * 8
                        + DIGEST_SIZE * record.call.fresh_count)
                        * D_EF
            || record.end_sample_count > CHUNK
            || record.output_root.iter().all(|&value| value == F::ZERO)
            || record.output_digest.iter().all(|&value| value == F::ZERO)
            || (index == 0
                && (record.prior_root.is_some()
                    || record.prior_digest.is_some()
                    || record.start_sample_count != 0
                    || record.start_state != [F::ZERO; POSEIDON2_WIDTH]))
            || (index != 0
                && (record.prior_root != Some(calls[index - 1].output_root)
                    || record.prior_digest != Some(calls[index - 1].output_digest)
                    || record.start_sample_count != calls[index - 1].end_sample_count
                    || record.start_state != calls[index - 1].end_state))
        {
            return Err("reduced-SWIRL VACC schedule record");
        }
        let row = &mut values[index * width..(index + 1) * width];
        let cols: &mut ReducedSwirlVaccScheduleCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.chunk_last = F::from_bool(index + 1 == calls.len());
        cols.is_last = F::from_bool(index + 1 == calls.len());
        cols.local_proof_idx = F::from_usize(index);
        cols.step = F::from_usize(record.call.step);
        cols.source_count = F::from_usize(source_count);
        cols.source_start = F::from_usize(record.call.source_start);
        cols.fresh_count = F::from_usize(record.call.fresh_count);
        cols.prior_count = F::from_usize(record.call.prior_count);
        cols.batch_start_tidx = F::from_usize(record.batch_start_tidx);
        cols.vacc_start_tidx = F::from_usize(record.vacc_start_tidx);
        cols.vacc_end_tidx = F::from_usize(record.vacc_end_tidx);
        cols.fresh_selector[record.call.fresh_count] = F::ONE;
        cols.start_sample_count = F::from_usize(record.start_sample_count);
        cols.start_state = record.start_state;
        cols.end_sample_count = F::from_usize(record.end_sample_count);
        cols.end_state = record.end_state;
        cols.prior_root = record.prior_root.unwrap_or([F::ZERO; DIGEST_SIZE]);
        cols.output_root = record.output_root;
        cols.prior_digest = record.prior_digest.unwrap_or([F::ZERO; DIGEST_SIZE]);
        cols.output_digest = record.output_digest;
    }
    Ok(RowMajorMatrix::new(values, width))
}

/// One fixed-capacity recursive-leaf slice of the canonical global schedule.
/// Global call/source indices and absolute transcript positions are retained;
/// only `local_proof_idx` is rebased to the physical leaf.
pub fn generate_reduced_swirl_vacc_schedule_range_trace(
    profile: &ReducedSwirlVaccProfile,
    total_source_count: usize,
    calls: &[ReducedSwirlVaccCallRecord],
    physical_call_capacity: usize,
) -> Result<RowMajorMatrix<F>, &'static str> {
    profile.validate()?;
    let expected = reduced_swirl_vacc_schedule(total_source_count, profile.input_arity)?;
    if calls.is_empty()
        || physical_call_capacity == 0
        || calls.len() > physical_call_capacity
        || calls.len() > expected.len()
    {
        return Err("reduced-SWIRL VACC schedule range count");
    }
    let header_end = reduced_swirl_vacc_header_elements(profile, total_source_count)?.len() * D_EF;
    let width = ReducedSwirlVaccScheduleCols::<F>::width();
    let height = adapter_trace_height(calls.len(), physical_call_capacity);
    let mut values = F::zero_vec(width * height);
    for (local_index, record) in calls.iter().enumerate() {
        let expected_call = *expected
            .get(record.call.step)
            .ok_or("reduced-SWIRL VACC schedule range step")?;
        let globally_final = record.call.step + 1 == expected.len();
        let expected_batch_start = if record.call.step == 0 {
            Some(header_end)
        } else if local_index != 0 {
            Some(calls[local_index - 1].vacc_end_tidx)
        } else {
            None
        };
        if record.call != expected_call
            || expected_batch_start.is_some_and(|start| record.batch_start_tidx != start)
            || record.vacc_start_tidx
                != record.batch_start_tidx
                    + (8 + REDUCED_SWIRL_VACC_SOURCE_BATCH_TAG.len()
                        + 3 * 8
                        + DIGEST_SIZE * record.call.fresh_count)
                        * D_EF
            || record.end_sample_count > CHUNK
            || record.output_root.iter().all(|&value| value == F::ZERO)
            || record.output_digest.iter().all(|&value| value == F::ZERO)
            || (record.call.step == 0
                && (record.prior_root.is_some()
                    || record.prior_digest.is_some()
                    || record.start_sample_count != 0
                    || record.start_state != [F::ZERO; POSEIDON2_WIDTH]))
            || (record.call.step != 0
                && (record.prior_root.is_none() || record.prior_digest.is_none()))
            || (local_index != 0
                && (record.prior_root != Some(calls[local_index - 1].output_root)
                    || record.prior_digest != Some(calls[local_index - 1].output_digest)
                    || record.start_sample_count != calls[local_index - 1].end_sample_count
                    || record.start_state != calls[local_index - 1].end_state))
        {
            return Err("reduced-SWIRL VACC schedule range record");
        }
        let row = &mut values[local_index * width..(local_index + 1) * width];
        let cols: &mut ReducedSwirlVaccScheduleCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.chunk_last = F::from_bool(local_index + 1 == calls.len());
        cols.is_last = F::from_bool(globally_final);
        cols.local_proof_idx = F::from_usize(local_index);
        cols.step = F::from_usize(record.call.step);
        cols.source_count = F::from_usize(total_source_count);
        cols.source_start = F::from_usize(record.call.source_start);
        cols.fresh_count = F::from_usize(record.call.fresh_count);
        cols.prior_count = F::from_usize(record.call.prior_count);
        cols.batch_start_tidx = F::from_usize(record.batch_start_tidx);
        cols.vacc_start_tidx = F::from_usize(record.vacc_start_tidx);
        cols.vacc_end_tidx = F::from_usize(record.vacc_end_tidx);
        cols.fresh_selector[record.call.fresh_count] = F::ONE;
        cols.start_sample_count = F::from_usize(record.start_sample_count);
        cols.start_state = record.start_state;
        cols.end_sample_count = F::from_usize(record.end_sample_count);
        cols.end_state = record.end_state;
        cols.prior_root = record.prior_root.unwrap_or([F::ZERO; DIGEST_SIZE]);
        cols.output_root = record.output_root;
        cols.prior_digest = record.prior_digest.unwrap_or([F::ZERO; DIGEST_SIZE]);
        cols.output_digest = record.output_digest;
    }
    Ok(RowMajorMatrix::new(values, width))
}

pub fn generate_reduced_swirl_vacc_batch_range_trace(
    calls: &[ReducedSwirlVaccCallRecord],
    physical_call_capacity: usize,
) -> Result<RowMajorMatrix<F>, &'static str> {
    if calls.is_empty() || physical_call_capacity == 0 || calls.len() > physical_call_capacity {
        return Err("reduced-SWIRL VACC batch range count");
    }
    let width = ReducedSwirlVaccBatchCols::<F>::width();
    let height = adapter_trace_height(calls.len(), physical_call_capacity);
    let mut values = F::zero_vec(width * height);
    for (index, record) in calls.iter().enumerate() {
        let cols: &mut ReducedSwirlVaccBatchCols<F> =
            values[index * width..(index + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.local_proof_idx = F::from_usize(index);
        cols.step = F::from_usize(record.call.step);
        cols.source_start = F::from_usize(record.call.source_start);
        cols.fresh_count = F::from_usize(record.call.fresh_count);
        cols.prior_count = F::from_usize(record.call.prior_count);
        cols.input_arity = F::from_usize(record.call.input_arity);
        cols.batch_start_tidx = F::from_usize(record.batch_start_tidx);
        cols.vacc_start_tidx = F::from_usize(record.vacc_start_tidx);
        cols.vacc_end_tidx = F::from_usize(record.vacc_end_tidx);
        fill_bits(&mut cols.step_bits, record.call.step);
        fill_bits(&mut cols.source_start_bits, record.call.source_start);
        fill_bits(&mut cols.fresh_count_bits, record.call.fresh_count);
    }
    Ok(RowMajorMatrix::new(values, width))
}

pub fn generate_reduced_swirl_vacc_prefix_range_trace(
    calls: &[ReducedSwirlVaccCallRecord],
    physical_call_capacity: usize,
) -> Result<RowMajorMatrix<F>, &'static str> {
    if calls.is_empty() || physical_call_capacity == 0 || calls.len() > physical_call_capacity {
        return Err("reduced-SWIRL VACC prefix range count");
    }
    let width = ReducedSwirlVaccPrefixCols::<F>::width();
    let height = adapter_trace_height(calls.len(), physical_call_capacity);
    let mut values = F::zero_vec(width * height);
    for (index, record) in calls.iter().enumerate() {
        let cols: &mut ReducedSwirlVaccPrefixCols<F> =
            values[index * width..(index + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.local_proof_idx = F::from_usize(index);
        cols.step = F::from_usize(record.call.step);
        cols.source_start = F::from_usize(record.call.source_start);
        cols.fresh_count = F::from_usize(record.call.fresh_count);
        cols.prior_count = F::from_usize(record.call.prior_count);
        cols.input_arity = F::from_usize(record.call.input_arity);
        cols.batch_start_tidx = F::from_usize(record.batch_start_tidx);
        cols.vacc_start_tidx = F::from_usize(record.vacc_start_tidx);
        cols.vacc_end_tidx = F::from_usize(record.vacc_end_tidx);
        fill_bits(&mut cols.step_bits, record.call.step);
        fill_bits(&mut cols.fresh_count_bits, record.call.fresh_count);
        fill_bits(&mut cols.prior_count_bits, record.call.prior_count);
    }
    Ok(RowMajorMatrix::new(values, width))
}

fn digest_start_air<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Expr,
    domain: u32,
    count: AB::Expr,
    enabled: AB::Expr,
) -> AB::Expr {
    let mut tidx = AB::Expr::ZERO;
    for value in [
        AB::Expr::from_u32(domain),
        AB::Expr::from_u32(REDUCED_SWIRL_VACC_PROTOCOL_VERSION),
        count,
        AB::Expr::ZERO,
    ] {
        observe_base_expr(
            bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            value,
            false,
            enabled.clone(),
        );
    }
    tidx
}

fn digest_field_air<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Expr,
    tidx: &mut AB::Expr,
    value: AB::Expr,
    enabled: AB::Expr,
) {
    for value in [AB::Expr::from_u32(REDUCED_SWIRL_FIELD_TAG), value] {
        observe_base_expr(
            bus,
            builder,
            proof_idx.clone(),
            tidx,
            value,
            false,
            enabled.clone(),
        );
    }
}

fn digest_u16_air<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Expr,
    tidx: &mut AB::Expr,
    low: AB::Expr,
    high: AB::Expr,
    enabled: AB::Expr,
) {
    digest_field_air(bus, builder, proof_idx.clone(), tidx, low, enabled.clone());
    digest_field_air(bus, builder, proof_idx, tidx, high, enabled);
}

fn digest_value_air<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Expr,
    tidx: &mut AB::Expr,
    digest: [AB::Expr; DIGEST_SIZE],
    enabled: AB::Expr,
) {
    observe_base_expr(
        bus,
        builder,
        proof_idx.clone(),
        tidx,
        AB::Expr::from_u32(REDUCED_SWIRL_DIGEST_TAG),
        false,
        enabled.clone(),
    );
    for value in digest {
        observe_base_expr(
            bus,
            builder,
            proof_idx.clone(),
            tidx,
            value,
            false,
            enabled.clone(),
        );
    }
}

fn digest_ext_air<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Expr,
    tidx: &mut AB::Expr,
    value: [AB::Expr; D_EF],
    enabled: AB::Expr,
) {
    observe_base_expr(
        bus,
        builder,
        proof_idx.clone(),
        tidx,
        AB::Expr::from_u32(REDUCED_SWIRL_EXTENSION_TAG),
        false,
        enabled.clone(),
    );
    for limb in value {
        observe_base_expr(
            bus,
            builder,
            proof_idx.clone(),
            tidx,
            limb,
            false,
            enabled.clone(),
        );
    }
}

fn digest_finish_air<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Expr,
    tidx: &mut AB::Expr,
    digest: [AB::Var; DIGEST_SIZE],
    enabled: AB::Expr,
) {
    observe_base_expr(
        bus,
        builder,
        proof_idx.clone(),
        tidx,
        AB::Expr::from_u32(REDUCED_SWIRL_END_TAG),
        false,
        enabled.clone(),
    );
    for value in digest {
        observe_base_expr(
            bus,
            builder,
            proof_idx.clone(),
            tidx,
            value.into(),
            true,
            enabled.clone(),
        );
    }
}

fn split_bits_expr<AB: AirBuilder<F = F>>(bits: &[AB::Var], start: usize, end: usize) -> AB::Expr
where
    AB::Var: Copy,
{
    let mut value = AB::Expr::ZERO;
    let mut power = F::ONE;
    for &bit in &bits[start..end.min(bits.len())] {
        value += AB::Expr::from(bit) * AB::Expr::from(power);
        power += power;
    }
    value
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ReducedSwirlSourceDigestCols<T> {
    pub active: T,
    /// Global source index of the first row in this physical trace.  The
    /// `source` column below remains dense and leaf-local so transcript and
    /// lookup namespaces can restart at zero in bounded recursive leaves.
    pub source_offset: T,
    pub source: T,
    pub local_proof_idx: T,
    pub step: T,
    pub slot: T,
    pub fresh_count: T,
    pub prior_count: T,
    pub input_arity: T,
    pub batch_start_tidx: T,
    pub vacc_start_tidx: T,
    pub root_count: T,
    pub opening_count: T,
    pub commitment_tidx: T,
    pub first_tree_id: T,
    pub root_active: [T; REDUCED_SWIRL_VACC_MAX_ROOTS],
    pub root_tree_id: [T; REDUCED_SWIRL_VACC_MAX_ROOTS],
    pub root_width: [T; REDUCED_SWIRL_VACC_MAX_ROOTS],
    pub root_width_bits: [[T; REDUCED_SWIRL_SCHEDULE_BITS]; REDUCED_SWIRL_VACC_MAX_ROOTS],
    pub roots: [[T; DIGEST_SIZE]; REDUCED_SWIRL_VACC_MAX_ROOTS],
    pub theta: [T; D_EF],
    pub mu: [T; D_EF],
    pub eta: [T; D_EF],
    pub beta: [[T; D_EF]; REDUCED_SWIRL_VACC_MAX_LOG_CODEWORD_LEN],
    pub segment_index: T,
    pub segment_index_bits: [T; REDUCED_SWIRL_AUTHORITY_BITS],
    pub common_main_root: [T; DIGEST_SIZE],
    pub trace_layout_digest: [T; DIGEST_SIZE],
    pub pending_claim_digest: [T; DIGEST_SIZE],
    pub checkpoint_tidx: T,
    pub checkpoint_tidx_bits: [T; REDUCED_SWIRL_AUTHORITY_BITS],
    pub checkpoint_state: [T; POSEIDON2_WIDTH],
    pub program_commitment: [T; DIGEST_SIZE],
    pub initial_pc: T,
    pub initial_root: [T; DIGEST_SIZE],
    pub final_pc: T,
    pub final_root: [T; DIGEST_SIZE],
    pub exit_code: T,
    pub is_terminate: T,
    pub claim_digest: [T; DIGEST_SIZE],
    pub entry_digest: [T; DIGEST_SIZE],
}

#[derive(ColumnsAir)]
#[columns_via(ReducedSwirlSourceDigestCols<u8>)]
pub struct ReducedSwirlSourceDigestAir {
    pub profile: ReducedSwirlVaccProfile,
    pub root_tree_stride: usize,
    pub digest_transcript_bus: TranscriptBus,
    pub main_transcript_bus: TranscriptBus,
    pub slot_bus: ReducedSwirlVaccSourceSlotBus,
    pub authority_bus: ReducedSwirlSourceAuthorityBus,
    pub source_claim_bus: ReducedSwirlSourceClaimBus,
    pub source_root_bus: ReducedSwirlSourceRootBus,
    pub source_beta_bus: ReducedSwirlSourceBetaBus,
    pub direct_source_bus: NativeDirectFreshSourceBus,
    pub direct_root_bus: NativeDirectFreshRootBus,
    pub vacc_claim_bus: NativeClaimValueBus,
    pub entry_digest_bus: ReducedSwirlSourceEntryDigestBus,
}

impl BaseAirWithPublicValues<F> for ReducedSwirlSourceDigestAir {}
impl PartitionedBaseAir<F> for ReducedSwirlSourceDigestAir {}
impl BaseAir<F> for ReducedSwirlSourceDigestAir {
    fn width(&self) -> usize {
        ReducedSwirlSourceDigestCols::<F>::width()
    }
}

impl<AB> Air<AB> for ReducedSwirlSourceDigestAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.profile.validate().is_ok());
        assert!(self.root_tree_stride > 0);
        let main = builder.main();
        let row = main.row_slice(0).expect("reduced-SWIRL source digest row");
        let next_row = main
            .row_slice(1)
            .expect("reduced-SWIRL source digest next row");
        let local: &ReducedSwirlSourceDigestCols<AB::Var> = (*row).borrow();
        let next: &ReducedSwirlSourceDigestCols<AB::Var> = (*next_row).borrow();
        builder.assert_bool(local.active);
        builder.assert_bool(local.is_terminate);
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_zero(local.source);
        builder
            .when_transition()
            .assert_zero(next.active * (AB::Expr::ONE - local.active));
        builder
            .when_transition()
            .when(next.active)
            .assert_eq(next.source, AB::Expr::from(local.source) + AB::Expr::ONE);
        builder
            .when_transition()
            .when(next.active)
            .assert_eq(next.source_offset, local.source_offset);
        let global_source = AB::Expr::from(local.source_offset) + AB::Expr::from(local.source);
        let active = AB::Expr::from(local.active);
        constrain_bits(
            builder,
            local.segment_index,
            &local.segment_index_bits,
            active.clone(),
        );
        constrain_bits(
            builder,
            local.checkpoint_tidx,
            &local.checkpoint_tidx_bits,
            active.clone(),
        );
        self.slot_bus.lookup_key(
            builder,
            ReducedSwirlVaccSourceSlotMessage {
                source: global_source.clone(),
                step: local.step.into(),
                slot: local.slot.into(),
                fresh_count: local.fresh_count.into(),
                prior_count: local.prior_count.into(),
                input_arity: local.input_arity.into(),
                batch_start_tidx: local.batch_start_tidx.into(),
                vacc_start_tidx: local.vacc_start_tidx.into(),
            },
            active.clone(),
        );
        self.authority_bus.receive(
            builder,
            ReducedSwirlSourceAuthorityMessage {
                source: local.source.into(),
                segment_index: local.segment_index.into(),
                common_main_root: local.common_main_root.map(Into::into),
                trace_layout_digest: local.trace_layout_digest.map(Into::into),
                pending_claim_digest: local.pending_claim_digest.map(Into::into),
                checkpoint_tidx: local.checkpoint_tidx.into(),
                checkpoint_state: local.checkpoint_state.map(Into::into),
                program_commitment: local.program_commitment.map(Into::into),
                initial_pc: local.initial_pc.into(),
                initial_root: local.initial_root.map(Into::into),
                final_pc: local.final_pc.into(),
                final_root: local.final_root.map(Into::into),
                exit_code: local.exit_code.into(),
                is_terminate: local.is_terminate.into(),
            },
            active.clone(),
        );
        self.source_claim_bus.lookup_key(
            builder,
            ReducedSwirlSourceClaimMessage {
                source: local.source.into(),
                root_count: local.root_count.into(),
                opening_count: local.opening_count.into(),
                l_skip: AB::Expr::from_usize(self.profile.source.l_skip),
                n_stack: AB::Expr::from_usize(self.profile.source.n_stack),
                log_blowup: AB::Expr::from_usize(self.profile.source.log_blowup),
                log_commit_rows_per_query: AB::Expr::from_usize(
                    self.profile.source.log_commit_rows_per_query,
                ),
                log_message_len: AB::Expr::from_usize(self.profile.source.log_message_len()),
                log_codeword_len: AB::Expr::from_usize(self.profile.source.log_codeword_len()),
                rows_per_query: AB::Expr::from_usize(self.profile.source.rows_per_query()),
                coefficient_layout_tag: AB::Expr::from_u64(
                    super::REDUCED_SWIRL_COEFFICIENT_SUBGROUP_TAG,
                ),
                coefficient_layout_version: AB::Expr::from_u32(
                    super::REDUCED_SWIRL_COEFFICIENT_SUBGROUP_VERSION,
                ),
                alpha_is_zero: AB::Expr::ONE,
                theta: local.theta.map(Into::into),
                mu: local.mu.map(Into::into),
                eta: local.eta.map(Into::into),
            },
            active.clone(),
        );
        self.direct_source_bus.lookup_key(
            builder,
            NativeDirectFreshSourceMessage {
                source: local.slot.into(),
                active: AB::Expr::ONE,
                root_count: local.root_count.into(),
                first_tree_id: local.first_tree_id.into(),
                commitment_tidx: local.commitment_tidx.into(),
                theta: local.theta.map(Into::into),
            },
            active.clone(),
        );

        let mut counted_roots = AB::Expr::ZERO;
        for ordinal in 0..REDUCED_SWIRL_VACC_MAX_ROOTS {
            let root_active = local.root_active[ordinal];
            builder.assert_bool(root_active);
            if ordinal == 0 {
                builder.when(active.clone()).assert_one(root_active);
            } else {
                builder.assert_zero(
                    AB::Expr::from(root_active) * (AB::Expr::ONE - local.root_active[ordinal - 1]),
                );
            }
            if ordinal >= self.profile.source.maximum_roots_per_source {
                builder.assert_zero(root_active);
            }
            counted_roots += root_active;
            constrain_bits(
                builder,
                local.root_width[ordinal],
                &local.root_width_bits[ordinal],
                root_active.into(),
            );
            builder.when(root_active).assert_eq(
                local.root_tree_id[ordinal],
                AB::Expr::from(local.first_tree_id)
                    + AB::Expr::from_usize(ordinal * self.root_tree_stride),
            );
            self.source_root_bus.lookup_key(
                builder,
                ReducedSwirlSourceRootMessage {
                    source: local.source.into(),
                    root_ordinal: AB::Expr::from_usize(ordinal),
                    width: local.root_width[ordinal].into(),
                    root: local.roots[ordinal].map(Into::into),
                },
                root_active,
            );
            self.direct_root_bus.lookup_key(
                builder,
                NativeDirectFreshRootMessage {
                    source: local.slot.into(),
                    root_ordinal: AB::Expr::from_usize(ordinal),
                    tree_id: local.root_tree_id[ordinal].into(),
                    width: local.root_width[ordinal].into(),
                    root: local.roots[ordinal].map(Into::into),
                },
                root_active,
            );
        }
        builder
            .when(active.clone())
            .assert_eq(local.root_count, counted_roots);

        for coordinate in 0..self.profile.source.log_codeword_len() {
            self.vacc_claim_bus.receive(
                builder,
                NativeClaimValueMessage {
                    proof_idx: local.local_proof_idx.into(),
                    source: local.slot.into(),
                    section: AB::Expr::from_usize(CLAIM_SECTION_ALPHA),
                    coordinate: AB::Expr::from_usize(coordinate),
                    value: core::array::from_fn(|_| AB::Expr::ZERO),
                },
                active.clone(),
            );
        }
        for coordinate in 0..self.profile.source.log_message_len() {
            let beta = local.beta[coordinate].map(Into::into);
            self.source_beta_bus.lookup_key(
                builder,
                ReducedSwirlSourceBetaMessage {
                    source: local.source.into(),
                    coordinate: AB::Expr::from_usize(coordinate),
                    value: beta.clone(),
                },
                active.clone(),
            );
            self.vacc_claim_bus.receive(
                builder,
                NativeClaimValueMessage {
                    proof_idx: local.local_proof_idx.into(),
                    source: local.slot.into(),
                    section: AB::Expr::from_usize(CLAIM_SECTION_BETA),
                    coordinate: AB::Expr::from_usize(coordinate),
                    value: beta,
                },
                active.clone(),
            );
        }
        self.vacc_claim_bus.receive(
            builder,
            NativeClaimValueMessage {
                proof_idx: local.local_proof_idx.into(),
                source: local.slot.into(),
                section: AB::Expr::from_usize(CLAIM_SECTION_BETA),
                coordinate: AB::Expr::from_usize(self.profile.source.log_message_len()),
                value: local.eta.map(Into::into),
            },
            active.clone(),
        );
        self.vacc_claim_bus.receive(
            builder,
            NativeClaimValueMessage {
                proof_idx: local.local_proof_idx.into(),
                source: local.slot.into(),
                section: AB::Expr::from_usize(CLAIM_SECTION_MU),
                coordinate: AB::Expr::ZERO,
                value: local.mu.map(Into::into),
            },
            active.clone(),
        );
        self.vacc_claim_bus.receive(
            builder,
            NativeClaimValueMessage {
                proof_idx: local.local_proof_idx.into(),
                source: local.slot.into(),
                section: AB::Expr::from_usize(CLAIM_SECTION_ETA),
                coordinate: AB::Expr::ZERO,
                value: core::array::from_fn(|_| AB::Expr::ZERO),
            },
            active.clone(),
        );

        self.eval_claim_digest(builder, local, active.clone());
        self.eval_entry_digest(builder, local, active.clone());
        let batch_prefix_elements = 8 + REDUCED_SWIRL_VACC_SOURCE_BATCH_TAG.len() + 3 * 8;
        let mut entry_tidx = AB::Expr::from(local.batch_start_tidx)
            + AB::Expr::from_usize(batch_prefix_elements * D_EF)
            + AB::Expr::from(local.slot) * AB::Expr::from_usize(DIGEST_SIZE * D_EF);
        for &limb in &local.entry_digest {
            observe_ext_expr(
                &self.main_transcript_bus,
                builder,
                local.local_proof_idx.into(),
                &mut entry_tidx,
                [limb.into(), AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO],
                active.clone(),
            );
        }
        self.entry_digest_bus.add_key_with_lookups(
            builder,
            ReducedSwirlSourceEntryDigestMessage {
                source: global_source,
                digest: local.entry_digest.map(Into::into),
            },
            active,
        );
    }
}

impl ReducedSwirlSourceDigestAir {
    fn eval_claim_digest<AB>(
        &self,
        builder: &mut AB,
        local: &ReducedSwirlSourceDigestCols<AB::Var>,
        enabled: AB::Expr,
    ) where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        let proof_idx = AB::Expr::from(local.source) * AB::Expr::from_usize(2);
        let observation_count = AB::Expr::from_usize(
            19 + self.profile.source.log_codeword_len() + self.profile.source.log_message_len(),
        ) + AB::Expr::from(local.root_count) * AB::Expr::from_usize(3);
        let mut tidx = digest_start_air(
            &self.digest_transcript_bus,
            builder,
            proof_idx.clone(),
            REDUCED_SWIRL_SOURCE_DIGEST_TAG ^ 0x40,
            observation_count,
            enabled.clone(),
        );
        digest_u16_air(
            &self.digest_transcript_bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            local.root_count.into(),
            AB::Expr::ZERO,
            enabled.clone(),
        );
        for ordinal in 0..REDUCED_SWIRL_VACC_MAX_ROOTS {
            digest_value_air(
                &self.digest_transcript_bus,
                builder,
                proof_idx.clone(),
                &mut tidx,
                local.roots[ordinal].map(Into::into),
                AB::Expr::from(local.root_active[ordinal]),
            );
        }
        digest_u16_air(
            &self.digest_transcript_bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            local.root_count.into(),
            AB::Expr::ZERO,
            enabled.clone(),
        );
        for ordinal in 0..REDUCED_SWIRL_VACC_MAX_ROOTS {
            digest_u16_air(
                &self.digest_transcript_bus,
                builder,
                proof_idx.clone(),
                &mut tidx,
                split_bits_expr::<AB>(
                    &local.root_width_bits[ordinal],
                    0,
                    REDUCED_SWIRL_SCHEDULE_BITS,
                ),
                AB::Expr::ZERO,
                AB::Expr::from(local.root_active[ordinal]),
            );
        }
        for value in [
            self.profile.source.l_skip,
            self.profile.source.n_stack,
            self.profile.source.log_blowup,
            self.profile.source.log_commit_rows_per_query,
        ] {
            digest_u16_air(
                &self.digest_transcript_bus,
                builder,
                proof_idx.clone(),
                &mut tidx,
                AB::Expr::from_usize(value),
                AB::Expr::ZERO,
                enabled.clone(),
            );
        }
        digest_ext_air(
            &self.digest_transcript_bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            local.theta.map(Into::into),
            enabled.clone(),
        );
        digest_u16_air(
            &self.digest_transcript_bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            AB::Expr::from_usize(self.profile.source.log_codeword_len()),
            AB::Expr::ZERO,
            enabled.clone(),
        );
        for _ in 0..self.profile.source.log_codeword_len() {
            digest_ext_air(
                &self.digest_transcript_bus,
                builder,
                proof_idx.clone(),
                &mut tidx,
                core::array::from_fn(|_| AB::Expr::ZERO),
                enabled.clone(),
            );
        }
        digest_ext_air(
            &self.digest_transcript_bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            local.mu.map(Into::into),
            enabled.clone(),
        );
        digest_u16_air(
            &self.digest_transcript_bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            AB::Expr::from_usize(self.profile.source.log_message_len()),
            AB::Expr::ZERO,
            enabled.clone(),
        );
        for coordinate in 0..self.profile.source.log_message_len() {
            digest_ext_air(
                &self.digest_transcript_bus,
                builder,
                proof_idx.clone(),
                &mut tidx,
                local.beta[coordinate].map(Into::into),
                enabled.clone(),
            );
        }
        digest_ext_air(
            &self.digest_transcript_bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            local.eta.map(Into::into),
            enabled.clone(),
        );
        digest_finish_air(
            &self.digest_transcript_bus,
            builder,
            proof_idx,
            &mut tidx,
            local.claim_digest,
            enabled,
        );
    }

    fn eval_entry_digest<AB>(
        &self,
        builder: &mut AB,
        local: &ReducedSwirlSourceDigestCols<AB::Var>,
        enabled: AB::Expr,
    ) where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        let proof_idx = AB::Expr::from(local.source) * AB::Expr::from_usize(2) + AB::Expr::ONE;
        let global_source = AB::Expr::from(local.source_offset) + AB::Expr::from(local.source);
        let mut tidx = digest_start_air(
            &self.digest_transcript_bus,
            builder,
            proof_idx.clone(),
            REDUCED_SWIRL_SOURCE_DIGEST_TAG ^ 0x70,
            AB::Expr::from_usize(15),
            enabled.clone(),
        );
        digest_u16_air(
            &self.digest_transcript_bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            global_source,
            AB::Expr::ZERO,
            enabled.clone(),
        );
        digest_u16_air(
            &self.digest_transcript_bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            split_bits_expr::<AB>(&local.segment_index_bits, 0, 16),
            split_bits_expr::<AB>(&local.segment_index_bits, 16, REDUCED_SWIRL_AUTHORITY_BITS),
            enabled.clone(),
        );
        for digest in [
            local.common_main_root,
            local.trace_layout_digest,
            local.pending_claim_digest,
            local.claim_digest,
        ] {
            digest_value_air(
                &self.digest_transcript_bus,
                builder,
                proof_idx.clone(),
                &mut tidx,
                digest.map(Into::into),
                enabled.clone(),
            );
        }
        digest_value_air(
            &self.digest_transcript_bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            local.program_commitment.map(Into::into),
            enabled.clone(),
        );
        digest_field_air(
            &self.digest_transcript_bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            local.initial_pc.into(),
            enabled.clone(),
        );
        digest_value_air(
            &self.digest_transcript_bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            local.initial_root.map(Into::into),
            enabled.clone(),
        );
        digest_field_air(
            &self.digest_transcript_bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            local.final_pc.into(),
            enabled.clone(),
        );
        digest_value_air(
            &self.digest_transcript_bus,
            builder,
            proof_idx.clone(),
            &mut tidx,
            local.final_root.map(Into::into),
            enabled.clone(),
        );
        for value in [local.exit_code, local.is_terminate] {
            digest_field_air(
                &self.digest_transcript_bus,
                builder,
                proof_idx.clone(),
                &mut tidx,
                value.into(),
                enabled.clone(),
            );
        }
        digest_finish_air(
            &self.digest_transcript_bus,
            builder,
            proof_idx,
            &mut tidx,
            local.entry_digest,
            enabled,
        );
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlSourceDigestRecord {
    pub claim: RecursiveReducedSwirlClaim,
    pub authority: ReducedSwirlSourceAuthorityRecord,
    pub commitment_tidx: usize,
    pub first_tree_id: u32,
}

pub struct ReducedSwirlSourceDigestTraceArtifacts {
    pub trace: RowMajorMatrix<F>,
    pub transcript_logs: Vec<TranscriptLog<F, [F; POSEIDON2_WIDTH]>>,
    pub entry_digests: Vec<Digest>,
}

pub fn generate_reduced_swirl_source_digest_trace(
    profile: &ReducedSwirlVaccProfile,
    root_tree_stride: usize,
    calls: &[ReducedSwirlVaccCallRecord],
    sources: &[ReducedSwirlSourceDigestRecord],
) -> Result<ReducedSwirlSourceDigestTraceArtifacts, &'static str> {
    profile.validate()?;
    if root_tree_stride == 0 || sources.is_empty() || sources.len() > profile.maximum_sources {
        return Err("reduced-SWIRL source digest inventory");
    }
    let schedule = reduced_swirl_vacc_schedule(sources.len(), profile.input_arity)?;
    if schedule.len() != calls.len()
        || calls
            .iter()
            .zip(schedule)
            .any(|(record, expected)| record.call != expected)
    {
        return Err("reduced-SWIRL source digest schedule");
    }
    generate_reduced_swirl_source_digest_range_trace(
        profile,
        root_tree_stride,
        0,
        calls,
        sources,
        profile.maximum_sources,
    )
}

/// Generate a bounded physical source-digest trace for a contiguous global
/// interval.  The canonical entry digest retains the global source index,
/// while AIR rows and digest-transcript proof IDs restart at zero.  This is
/// the recursive-leaf analogue of OpenVM's group-local proof numbering.
pub fn generate_reduced_swirl_source_digest_range_trace(
    profile: &ReducedSwirlVaccProfile,
    root_tree_stride: usize,
    source_offset: usize,
    calls: &[ReducedSwirlVaccCallRecord],
    sources: &[ReducedSwirlSourceDigestRecord],
    physical_capacity: usize,
) -> Result<ReducedSwirlSourceDigestTraceArtifacts, &'static str> {
    profile.validate()?;
    let source_end = source_offset
        .checked_add(sources.len())
        .ok_or("reduced-SWIRL source digest range")?;
    if root_tree_stride == 0
        || sources.is_empty()
        || sources.len() > physical_capacity
        || physical_capacity == 0
        || source_end > profile.maximum_sources
        || calls.is_empty()
    {
        return Err("reduced-SWIRL source digest inventory");
    }
    let width = ReducedSwirlSourceDigestCols::<F>::width();
    let height = adapter_trace_height(sources.len(), physical_capacity);
    let mut values = F::zero_vec(width * height);
    let mut transcript_logs = Vec::with_capacity(2 * sources.len());
    let mut entry_digests = Vec::with_capacity(sources.len());
    for (source, record) in sources.iter().enumerate() {
        let global_source = source_offset + source;
        let (call_index, call) = calls
            .iter()
            .enumerate()
            .find_map(|(index, record)| {
                let call = record.call;
                (global_source >= call.source_start
                    && global_source < call.source_start + call.fresh_count)
                    .then_some((index, call))
            })
            .ok_or("reduced-SWIRL source call mapping")?;
        let slot = global_source - call.source_start;
        let (claim_digest, claim_log, entry_digest, entry_log) = reduced_swirl_source_digest_logs(
            profile,
            global_source,
            call,
            slot,
            &record.claim,
            &record.authority,
        )?;
        if record
            .claim
            .widths
            .iter()
            .any(|&width| width > u16::MAX as usize)
            || record.authority.segment_index >= (1 << REDUCED_SWIRL_AUTHORITY_BITS)
            || record.authority.checkpoint_tidx >= (1 << REDUCED_SWIRL_AUTHORITY_BITS)
        {
            return Err("reduced-SWIRL source digest range");
        }
        let call_record = &calls[call_index];
        let cols: &mut ReducedSwirlSourceDigestCols<F> =
            values[source * width..(source + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.source_offset = F::from_usize(source_offset);
        cols.source = F::from_usize(source);
        cols.local_proof_idx = F::from_usize(call_index);
        cols.step = F::from_usize(call.step);
        cols.slot = F::from_usize(slot);
        cols.fresh_count = F::from_usize(call.fresh_count);
        cols.prior_count = F::from_usize(call.prior_count);
        cols.input_arity = F::from_usize(call.input_arity);
        cols.batch_start_tidx = F::from_usize(call_record.batch_start_tidx);
        cols.vacc_start_tidx = F::from_usize(call_record.vacc_start_tidx);
        cols.root_count = F::from_usize(record.claim.roots.len());
        cols.opening_count = F::from_usize(record.claim.widths.iter().sum());
        cols.commitment_tidx = F::from_usize(record.commitment_tidx);
        cols.first_tree_id = F::from_u32(record.first_tree_id);
        for ordinal in 0..record.claim.roots.len() {
            cols.root_active[ordinal] = F::ONE;
            cols.root_tree_id[ordinal] =
                F::from_usize(record.first_tree_id as usize + ordinal * root_tree_stride);
            cols.root_width[ordinal] = F::from_usize(record.claim.widths[ordinal]);
            fill_bits(
                &mut cols.root_width_bits[ordinal],
                record.claim.widths[ordinal],
            );
            cols.roots[ordinal] = record.claim.roots[ordinal];
        }
        cols.theta
            .copy_from_slice(record.claim.theta.as_basis_coefficients_slice());
        cols.mu
            .copy_from_slice(record.claim.mu.as_basis_coefficients_slice());
        cols.eta
            .copy_from_slice(record.claim.eta.as_basis_coefficients_slice());
        for (coordinate, value) in record.claim.beta.iter().enumerate() {
            cols.beta[coordinate].copy_from_slice(value.as_basis_coefficients_slice());
        }
        cols.segment_index = F::from_u32(record.authority.segment_index);
        fill_bits(
            &mut cols.segment_index_bits,
            record.authority.segment_index as usize,
        );
        cols.common_main_root = record.authority.common_main_root;
        cols.trace_layout_digest = record.authority.trace_layout_digest;
        cols.pending_claim_digest = record.authority.pending_claim_digest;
        cols.checkpoint_tidx = F::from_usize(record.authority.checkpoint_tidx);
        fill_bits(
            &mut cols.checkpoint_tidx_bits,
            record.authority.checkpoint_tidx,
        );
        cols.checkpoint_state = record.authority.checkpoint_state;
        cols.program_commitment = record.authority.program_commitment;
        cols.initial_pc = record.authority.initial_pc;
        cols.initial_root = record.authority.initial_root;
        cols.final_pc = record.authority.final_pc;
        cols.final_root = record.authority.final_root;
        cols.exit_code = record.authority.exit_code;
        cols.is_terminate = record.authority.is_terminate;
        cols.claim_digest = claim_digest;
        cols.entry_digest = entry_digest;
        transcript_logs.extend([claim_log, entry_log]);
        entry_digests.push(entry_digest);
    }
    Ok(ReducedSwirlSourceDigestTraceArtifacts {
        trace: RowMajorMatrix::new(values, width),
        transcript_logs,
        entry_digests,
    })
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ReducedSwirlManifestDigestCols<T> {
    pub active: T,
    pub is_last: T,
    /// Global index of the first source in this bounded manifest.  The
    /// manifest transcript itself remains the canonical local chunk digest;
    /// only the entry-authority lookup uses the global coordinate.
    pub source_offset: T,
    pub source: T,
    pub source_count: T,
    pub entry_digest: [T; DIGEST_SIZE],
    pub manifest_digest: [T; DIGEST_SIZE],
}

#[derive(ColumnsAir)]
#[columns_via(ReducedSwirlManifestDigestCols<u8>)]
pub struct ReducedSwirlManifestDigestAir {
    pub maximum_sources: usize,
    pub transcript_bus: TranscriptBus,
    pub entry_digest_bus: ReducedSwirlSourceEntryDigestBus,
    pub manifest_digest_bus: ReducedSwirlManifestDigestBus,
}

impl BaseAirWithPublicValues<F> for ReducedSwirlManifestDigestAir {}
impl PartitionedBaseAir<F> for ReducedSwirlManifestDigestAir {}
impl BaseAir<F> for ReducedSwirlManifestDigestAir {
    fn width(&self) -> usize {
        ReducedSwirlManifestDigestCols::<F>::width()
    }
}

impl<AB> Air<AB> for ReducedSwirlManifestDigestAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.maximum_sources > 0);
        let main = builder.main();
        let row = main.row_slice(0).expect("reduced-SWIRL manifest row");
        let next_row = main.row_slice(1).expect("reduced-SWIRL manifest next row");
        let local: &ReducedSwirlManifestDigestCols<AB::Var> = (*row).borrow();
        let next: &ReducedSwirlManifestDigestCols<AB::Var> = (*next_row).borrow();
        builder.assert_bool(local.active);
        builder.assert_bool(local.is_last);
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_zero(local.source);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);
        builder
            .when_transition()
            .assert_eq(next.active, AB::Expr::from(local.active) - local.is_last);
        builder
            .when_transition()
            .when(next.active)
            .assert_eq(next.source, AB::Expr::from(local.source) + AB::Expr::ONE);
        builder
            .when_transition()
            .when(next.active)
            .assert_eq(next.source_count, local.source_count);
        builder
            .when_transition()
            .when(next.active)
            .assert_eq(next.source_offset, local.source_offset);
        for limb in 0..DIGEST_SIZE {
            builder
                .when_transition()
                .when(next.active)
                .assert_eq(next.manifest_digest[limb], local.manifest_digest[limb]);
        }
        builder.when(local.active * local.is_last).assert_eq(
            AB::Expr::from(local.source) + AB::Expr::ONE,
            local.source_count,
        );
        self.entry_digest_bus.lookup_key(
            builder,
            ReducedSwirlSourceEntryDigestMessage {
                source: (AB::Expr::from(local.source_offset) + local.source).into(),
                digest: local.entry_digest.map(Into::into),
            },
            local.active,
        );

        let proof_idx = AB::Expr::ZERO;
        let first = builder.is_first_row();
        let mut tidx = AB::Expr::ZERO;
        for &byte in REDUCED_SWIRL_VACC_MANIFEST_TAG {
            observe_base_expr(
                &self.transcript_bus,
                builder,
                proof_idx.clone(),
                &mut tidx,
                AB::Expr::from_u8(byte),
                false,
                first.clone(),
            );
        }
        for value in [
            AB::Expr::from_u32(REDUCED_SWIRL_VACC_PROTOCOL_VERSION),
            AB::Expr::from(local.source_count),
        ] {
            observe_base_expr(
                &self.transcript_bus,
                builder,
                proof_idx.clone(),
                &mut tidx,
                value,
                false,
                first.clone(),
            );
        }
        let mut entry_tidx = AB::Expr::from_usize(REDUCED_SWIRL_VACC_MANIFEST_TAG.len() + 2)
            + AB::Expr::from(local.source) * AB::Expr::from_usize(1 + DIGEST_SIZE);
        observe_base_expr(
            &self.transcript_bus,
            builder,
            proof_idx.clone(),
            &mut entry_tidx,
            local.source.into(),
            false,
            local.active.into(),
        );
        for &limb in &local.entry_digest {
            observe_base_expr(
                &self.transcript_bus,
                builder,
                proof_idx.clone(),
                &mut entry_tidx,
                limb.into(),
                false,
                local.active.into(),
            );
        }
        let mut sample_tidx = AB::Expr::from_usize(REDUCED_SWIRL_VACC_MANIFEST_TAG.len() + 2)
            + AB::Expr::from(local.source_count) * AB::Expr::from_usize(1 + DIGEST_SIZE);
        for &limb in &local.manifest_digest {
            observe_base_expr(
                &self.transcript_bus,
                builder,
                proof_idx.clone(),
                &mut sample_tidx,
                limb.into(),
                true,
                AB::Expr::from(local.active) * local.is_last,
            );
        }
        self.manifest_digest_bus.send(
            builder,
            ReducedSwirlManifestDigestMessage {
                source_count: local.source_count.into(),
                digest: local.manifest_digest.map(Into::into),
            },
            local.active * local.is_last,
        );
    }
}

fn reduced_swirl_manifest_digest_with_log(
    entry_digests: &[Digest],
) -> Result<(Digest, TranscriptLog<F, [F; POSEIDON2_WIDTH]>), &'static str> {
    let digest = reduced_swirl_manifest_digest(entry_digests)?;
    let mut transcript = default_duplex_sponge_recorder();
    for &byte in REDUCED_SWIRL_VACC_MANIFEST_TAG {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
            &mut transcript,
            F::from_u8(byte),
        );
    }
    <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
        &mut transcript,
        F::from_u32(REDUCED_SWIRL_VACC_PROTOCOL_VERSION),
    );
    <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
        &mut transcript,
        F::from_usize(entry_digests.len()),
    );
    for (source, entry) in entry_digests.iter().enumerate() {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
            &mut transcript,
            F::from_usize(source),
        );
        for &limb in entry {
            <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(&mut transcript, limb);
        }
    }
    let logged = core::array::from_fn(|_| {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::sample(&mut transcript)
    });
    if logged != digest {
        return Err("reduced-SWIRL manifest transcript mismatch");
    }
    Ok((digest, TranscriptHistory::into_log(transcript)))
}

pub fn generate_reduced_swirl_manifest_digest_trace(
    maximum_sources: usize,
    entry_digests: &[Digest],
) -> Result<
    (
        RowMajorMatrix<F>,
        Digest,
        TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    ),
    &'static str,
> {
    generate_reduced_swirl_manifest_digest_range_trace(0, maximum_sources, entry_digests)
}

/// Generate the canonical local manifest for a bounded source interval while
/// authenticating each entry under its global source coordinate.
pub fn generate_reduced_swirl_manifest_digest_range_trace(
    source_offset: usize,
    physical_capacity: usize,
    entry_digests: &[Digest],
) -> Result<
    (
        RowMajorMatrix<F>,
        Digest,
        TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    ),
    &'static str,
> {
    if entry_digests.is_empty() || entry_digests.len() > physical_capacity {
        return Err("reduced-SWIRL manifest trace inventory");
    }
    let (manifest_digest, transcript_log) = reduced_swirl_manifest_digest_with_log(entry_digests)?;
    let width = ReducedSwirlManifestDigestCols::<F>::width();
    let height = adapter_trace_height(entry_digests.len(), physical_capacity);
    let mut values = F::zero_vec(width * height);
    for (source, &entry_digest) in entry_digests.iter().enumerate() {
        let cols: &mut ReducedSwirlManifestDigestCols<F> =
            values[source * width..(source + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.is_last = F::from_bool(source + 1 == entry_digests.len());
        cols.source_offset = F::from_usize(source_offset);
        cols.source = F::from_usize(source);
        cols.source_count = F::from_usize(entry_digests.len());
        cols.entry_digest = entry_digest;
        cols.manifest_digest = manifest_digest;
    }
    Ok((
        RowMajorMatrix::new(values, width),
        manifest_digest,
        transcript_log,
    ))
}

/// Exact transition-leaf metadata for one setup-fixed linear-chain call.
/// Keep this field order synchronized with
/// `openvm_continuations::circuit::reduced_swirl_transition_leaf`.
pub fn reduced_swirl_reconciliation_chain_metadata(
    total_source_count: usize,
    call_index: usize,
    source_start: usize,
    fresh_count: usize,
    prior_count: usize,
    is_final: bool,
) -> Digest {
    [
        F::from_u32(REDUCED_SWIRL_TRANSITION_CHAIN_METADATA_TAG),
        F::from_u32(REDUCED_SWIRL_TRANSITION_LEAF_PROTOCOL_VERSION),
        F::from_usize(total_source_count),
        F::from_usize(call_index),
        F::from_usize(source_start),
        F::from_usize(fresh_count),
        F::from_usize(prior_count),
        F::from_bool(is_final),
    ]
}

/// Exact transition-leaf rolling-chain genesis.
pub fn reduced_swirl_reconciliation_chain_genesis(source_protocol_digest: Digest) -> Digest {
    let metadata = core::array::from_fn(|index| match index {
        0 => F::from_u32(REDUCED_SWIRL_TRANSITION_CHAIN_GENESIS_TAG),
        1 => F::from_u32(REDUCED_SWIRL_TRANSITION_LEAF_PROTOCOL_VERSION),
        _ => F::ZERO,
    });
    poseidon2_compress_with_capacity(source_protocol_digest, metadata).0
}

/// Exact transition-leaf rolling-chain update for one local manifest.
pub fn reduced_swirl_reconciliation_chain_append(
    chain_before: Digest,
    metadata: Digest,
    local_manifest_digest: Digest,
) -> Digest {
    let chunk = poseidon2_compress_with_capacity(metadata, local_manifest_digest).0;
    poseidon2_compress_with_capacity(chain_before, chunk).0
}

fn record_reconciliation_compression(
    left: Digest,
    right: Digest,
    requests: &mut Vec<[F; 2 * DIGEST_SIZE]>,
) -> Digest {
    requests.push(core::array::from_fn(|index| {
        if index < DIGEST_SIZE {
            left[index]
        } else {
            right[index - DIGEST_SIZE]
        }
    }));
    poseidon2_compress_with_capacity(left, right).0
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ReducedSwirlManifestReconciliationCols<T> {
    pub active: T,
    pub is_last: T,
    pub chunk_first: T,
    pub chunk_last: T,
    pub capacity_end: T,
    pub capacity_delta_inverse: T,
    pub source: T,
    pub source_count: T,
    pub call_index: T,
    pub source_start: T,
    pub slot: T,
    pub fresh_count: T,
    pub prior_count: T,
    pub entry_digest: [T; DIGEST_SIZE],
    pub entry_nonzero_selector: [T; DIGEST_SIZE],
    pub entry_nonzero_inverse: T,
    pub flat_manifest_digest: [T; DIGEST_SIZE],
    pub local_manifest_digest: [T; DIGEST_SIZE],
    pub chain_before: [T; DIGEST_SIZE],
    pub chunk_commitment: [T; DIGEST_SIZE],
    pub chain_after: [T; DIGEST_SIZE],
}

/// Setup-fixed reconciliation of the old flat terminal manifest with the
/// WARP-call rolling commitment authenticated by transition leaves.  Every
/// source is consumed exactly once under its global index.  Heterogeneity is
/// in private digest values only; the schedule and relation shape are fixed by
/// the verifying key.
#[derive(ColumnsAir)]
#[columns_via(ReducedSwirlManifestReconciliationCols<u8>)]
pub struct ReducedSwirlManifestReconciliationAir {
    pub input_arity: usize,
    pub source_protocol_digest: Digest,
    pub transcript_bus: TranscriptBus,
    pub compress_bus: Poseidon2CompressBus,
    pub receipt_bus: ReducedSwirlManifestReconciliationReceiptBus,
}

impl BaseAirWithPublicValues<F> for ReducedSwirlManifestReconciliationAir {}
impl PartitionedBaseAir<F> for ReducedSwirlManifestReconciliationAir {}
impl BaseAir<F> for ReducedSwirlManifestReconciliationAir {
    fn width(&self) -> usize {
        ReducedSwirlManifestReconciliationCols::<F>::width()
    }
}

impl<AB> Air<AB> for ReducedSwirlManifestReconciliationAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(
            self.input_arity >= 2
                && self.input_arity <= REDUCED_SWIRL_VACC_MAX_INPUT_ARITY
                && self.input_arity.is_power_of_two()
                && self
                    .source_protocol_digest
                    .iter()
                    .any(|value| *value != F::ZERO)
        );
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("reduced-SWIRL manifest reconciliation row");
        let next_row = main
            .row_slice(1)
            .expect("reduced-SWIRL manifest reconciliation next row");
        let local: &ReducedSwirlManifestReconciliationCols<AB::Var> = (*row).borrow();
        let next: &ReducedSwirlManifestReconciliationCols<AB::Var> = (*next_row).borrow();
        let active = AB::Expr::from(local.active);
        for flag in [
            local.active,
            local.is_last,
            local.chunk_first,
            local.chunk_last,
            local.capacity_end,
            local.prior_count,
        ] {
            builder.assert_bool(flag);
        }
        builder
            .when(AB::Expr::ONE - active.clone())
            .assert_zero(local.is_last);
        builder
            .when(AB::Expr::ONE - active.clone())
            .assert_zero(local.chunk_first);
        builder
            .when(AB::Expr::ONE - active.clone())
            .assert_zero(local.chunk_last);
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.chunk_first);
        builder.when_first_row().assert_zero(local.source);
        builder.when_first_row().assert_zero(local.call_index);
        builder.when_first_row().assert_zero(local.source_start);
        builder.when_first_row().assert_zero(local.slot);
        builder.when_first_row().assert_zero(local.prior_count);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);
        builder
            .when_transition()
            .assert_eq(next.active, active.clone() - local.is_last);

        // A digest cannot be the all-zero sentinel.  A one-hot selector picks
        // one genuinely nonzero limb; unlike a random linear combination,
        // this cannot cancel in the field.
        let mut selected_limb = AB::Expr::ZERO;
        let mut selector_sum = AB::Expr::ZERO;
        for (&selector, &limb) in local.entry_nonzero_selector.iter().zip(&local.entry_digest) {
            builder.assert_bool(selector);
            selector_sum += selector;
            selected_limb += AB::Expr::from(selector) * limb;
        }
        builder.when(active.clone()).assert_one(selector_sum);
        builder
            .when(active.clone())
            .assert_one(selected_limb * local.entry_nonzero_inverse);

        builder.when(active.clone()).assert_eq(
            local.source,
            AB::Expr::from(local.source_start) + local.slot,
        );
        builder.when(active.clone() * local.is_last).assert_eq(
            AB::Expr::from(local.source) + AB::Expr::ONE,
            local.source_count,
        );

        // `capacity_end` is the constrained zero test for
        // slot + 1 == arity - prior_count.  Consequently a chunk closes
        // exactly at capacity or at the globally final source, never at a
        // prover-chosen earlier boundary.
        let capacity = AB::Expr::from_usize(self.input_arity) - local.prior_count;
        let capacity_delta = AB::Expr::from(local.slot) + AB::Expr::ONE - capacity;
        builder
            .when(active.clone())
            .assert_zero(capacity_delta.clone() * local.capacity_end);
        builder.when(active.clone()).assert_eq(
            capacity_delta * local.capacity_delta_inverse,
            AB::Expr::ONE - local.capacity_end,
        );
        let expected_chunk_last = AB::Expr::from(local.is_last) + local.capacity_end
            - AB::Expr::from(local.is_last) * local.capacity_end;
        builder
            .when(active.clone())
            .assert_eq(local.chunk_last, expected_chunk_last);
        builder.when(active.clone() * local.chunk_last).assert_eq(
            local.fresh_count,
            AB::Expr::from(local.slot) + AB::Expr::ONE,
        );

        let transition = builder.is_transition() * AB::Expr::from(next.active);
        let within_chunk = transition.clone() * (AB::Expr::ONE - local.chunk_last);
        let next_chunk = transition.clone() * local.chunk_last;
        builder
            .when(transition.clone())
            .assert_eq(next.source, AB::Expr::from(local.source) + AB::Expr::ONE);
        builder
            .when(transition.clone())
            .assert_eq(next.source_count, local.source_count);
        builder
            .when(transition.clone())
            .assert_eq(next.chunk_first, local.chunk_last);
        builder
            .when(within_chunk.clone())
            .assert_eq(next.call_index, local.call_index);
        builder
            .when(within_chunk.clone())
            .assert_eq(next.source_start, local.source_start);
        builder
            .when(within_chunk.clone())
            .assert_eq(next.slot, AB::Expr::from(local.slot) + AB::Expr::ONE);
        builder
            .when(within_chunk.clone())
            .assert_eq(next.fresh_count, local.fresh_count);
        builder
            .when(within_chunk.clone())
            .assert_eq(next.prior_count, local.prior_count);
        builder.when(next_chunk.clone()).assert_eq(
            next.call_index,
            AB::Expr::from(local.call_index) + AB::Expr::ONE,
        );
        builder.when(next_chunk.clone()).assert_eq(
            next.source_start,
            AB::Expr::from(local.source) + AB::Expr::ONE,
        );
        builder.when(next_chunk.clone()).assert_zero(next.slot);
        builder
            .when(next_chunk.clone())
            .assert_one(next.prior_count);
        for limb in 0..DIGEST_SIZE {
            builder.when(transition.clone()).assert_eq(
                next.flat_manifest_digest[limb],
                local.flat_manifest_digest[limb],
            );
            builder.when(within_chunk.clone()).assert_eq(
                next.local_manifest_digest[limb],
                local.local_manifest_digest[limb],
            );
            builder
                .when(within_chunk.clone())
                .assert_eq(next.chain_before[limb], local.chain_before[limb]);
            builder
                .when(next_chunk.clone())
                .assert_eq(next.chain_before[limb], local.chain_after[limb]);
        }

        // Namespace zero is the canonical all-source transcript.  Call-local
        // transcripts use call_index + 1 and locally numbered entries; the
        // flat transcript simultaneously fixes each entry's global index.
        let mut flat_header_tidx = AB::Expr::ZERO;
        for &byte in REDUCED_SWIRL_VACC_MANIFEST_TAG {
            observe_base_expr(
                &self.transcript_bus,
                builder,
                AB::Expr::ZERO,
                &mut flat_header_tidx,
                AB::Expr::from_u8(byte),
                false,
                builder.is_first_row(),
            );
        }
        for value in [
            AB::Expr::from_u32(REDUCED_SWIRL_VACC_PROTOCOL_VERSION),
            local.source_count.into(),
        ] {
            observe_base_expr(
                &self.transcript_bus,
                builder,
                AB::Expr::ZERO,
                &mut flat_header_tidx,
                value,
                false,
                builder.is_first_row(),
            );
        }
        let mut flat_entry_tidx = AB::Expr::from_usize(REDUCED_SWIRL_VACC_MANIFEST_TAG.len() + 2)
            + AB::Expr::from(local.source) * AB::Expr::from_usize(1 + DIGEST_SIZE);
        observe_base_expr(
            &self.transcript_bus,
            builder,
            AB::Expr::ZERO,
            &mut flat_entry_tidx,
            local.source.into(),
            false,
            active.clone(),
        );
        for &limb in &local.entry_digest {
            observe_base_expr(
                &self.transcript_bus,
                builder,
                AB::Expr::ZERO,
                &mut flat_entry_tidx,
                limb.into(),
                false,
                active.clone(),
            );
        }
        let mut flat_sample_tidx = AB::Expr::from_usize(REDUCED_SWIRL_VACC_MANIFEST_TAG.len() + 2)
            + AB::Expr::from(local.source_count) * AB::Expr::from_usize(1 + DIGEST_SIZE);
        for &limb in &local.flat_manifest_digest {
            observe_base_expr(
                &self.transcript_bus,
                builder,
                AB::Expr::ZERO,
                &mut flat_sample_tidx,
                limb.into(),
                true,
                active.clone() * local.is_last,
            );
        }

        let local_proof_idx = AB::Expr::from(local.call_index) + AB::Expr::ONE;
        let mut local_header_tidx = AB::Expr::ZERO;
        for &byte in REDUCED_SWIRL_VACC_MANIFEST_TAG {
            observe_base_expr(
                &self.transcript_bus,
                builder,
                local_proof_idx.clone(),
                &mut local_header_tidx,
                AB::Expr::from_u8(byte),
                false,
                active.clone() * local.chunk_first,
            );
        }
        for value in [
            AB::Expr::from_u32(REDUCED_SWIRL_VACC_PROTOCOL_VERSION),
            local.fresh_count.into(),
        ] {
            observe_base_expr(
                &self.transcript_bus,
                builder,
                local_proof_idx.clone(),
                &mut local_header_tidx,
                value,
                false,
                active.clone() * local.chunk_first,
            );
        }
        let mut local_entry_tidx = AB::Expr::from_usize(REDUCED_SWIRL_VACC_MANIFEST_TAG.len() + 2)
            + AB::Expr::from(local.slot) * AB::Expr::from_usize(1 + DIGEST_SIZE);
        observe_base_expr(
            &self.transcript_bus,
            builder,
            local_proof_idx.clone(),
            &mut local_entry_tidx,
            local.slot.into(),
            false,
            active.clone(),
        );
        for &limb in &local.entry_digest {
            observe_base_expr(
                &self.transcript_bus,
                builder,
                local_proof_idx.clone(),
                &mut local_entry_tidx,
                limb.into(),
                false,
                active.clone(),
            );
        }
        let mut local_sample_tidx = AB::Expr::from_usize(REDUCED_SWIRL_VACC_MANIFEST_TAG.len() + 2)
            + AB::Expr::from(local.fresh_count) * AB::Expr::from_usize(1 + DIGEST_SIZE);
        for &limb in &local.local_manifest_digest {
            observe_base_expr(
                &self.transcript_bus,
                builder,
                local_proof_idx.clone(),
                &mut local_sample_tidx,
                limb.into(),
                true,
                active.clone() * local.chunk_last,
            );
        }

        let genesis_metadata = core::array::from_fn(|index| match index {
            0 => AB::Expr::from_u32(REDUCED_SWIRL_TRANSITION_CHAIN_GENESIS_TAG),
            1 => AB::Expr::from_u32(REDUCED_SWIRL_TRANSITION_LEAF_PROTOCOL_VERSION),
            _ => AB::Expr::ZERO,
        });
        self.lookup_compression(
            builder,
            self.source_protocol_digest.map(AB::Expr::from),
            genesis_metadata,
            local.chain_before.map(Into::into),
            builder.is_first_row(),
        );
        let chain_metadata = [
            AB::Expr::from_u32(REDUCED_SWIRL_TRANSITION_CHAIN_METADATA_TAG),
            AB::Expr::from_u32(REDUCED_SWIRL_TRANSITION_LEAF_PROTOCOL_VERSION),
            local.source_count.into(),
            local.call_index.into(),
            local.source_start.into(),
            local.fresh_count.into(),
            local.prior_count.into(),
            local.is_last.into(),
        ];
        self.lookup_compression(
            builder,
            chain_metadata,
            local.local_manifest_digest.map(Into::into),
            local.chunk_commitment.map(Into::into),
            active.clone() * local.chunk_last,
        );
        self.lookup_compression(
            builder,
            local.chain_before.map(Into::into),
            local.chunk_commitment.map(Into::into),
            local.chain_after.map(Into::into),
            active.clone() * local.chunk_last,
        );
        self.receipt_bus.add_key_with_lookups(
            builder,
            ReducedSwirlManifestReconciliationReceiptMessage {
                source_count: local.source_count.into(),
                call_count: AB::Expr::from(local.call_index) + AB::Expr::ONE,
                flat_manifest_digest: local.flat_manifest_digest.map(Into::into),
                rolling_chain_endpoint: local.chain_after.map(Into::into),
            },
            local.active * local.is_last,
        );
    }
}

impl ReducedSwirlManifestReconciliationAir {
    fn lookup_compression<AB: AirBuilder<F = F> + InteractionBuilder>(
        &self,
        builder: &mut AB,
        left: [AB::Expr; DIGEST_SIZE],
        right: [AB::Expr; DIGEST_SIZE],
        output: [AB::Expr; DIGEST_SIZE],
        enabled: AB::Expr,
    ) {
        self.compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: core::array::from_fn(|index| {
                    if index < DIGEST_SIZE {
                        left[index].clone()
                    } else {
                        right[index - DIGEST_SIZE].clone()
                    }
                }),
                output,
            },
            enabled,
        );
    }
}

#[derive(Clone, Debug)]
pub struct ReducedSwirlManifestReconciliationComponent {
    pub input_arity: usize,
    pub source_protocol_digest: Digest,
    pub transcript_bus: TranscriptBus,
    pub compress_bus: Poseidon2CompressBus,
    pub receipt_bus: ReducedSwirlManifestReconciliationReceiptBus,
}

impl ReducedSwirlManifestReconciliationComponent {
    pub fn validate(&self) -> Result<(), &'static str> {
        if !(2..=REDUCED_SWIRL_VACC_MAX_INPUT_ARITY).contains(&self.input_arity)
            || !self.input_arity.is_power_of_two()
            || self
                .source_protocol_digest
                .iter()
                .all(|value| *value == F::ZERO)
        {
            return Err("reduced-SWIRL manifest reconciliation setup");
        }
        Ok(())
    }

    #[must_use]
    pub fn air<PCS>(&self) -> AirRef<PCS>
    where
        PCS: StarkProtocolConfig<F = F>,
    {
        self.validate()
            .expect("valid reduced-SWIRL manifest reconciliation setup");
        Arc::new(ReducedSwirlManifestReconciliationAir {
            input_arity: self.input_arity,
            source_protocol_digest: self.source_protocol_digest,
            transcript_bus: self.transcript_bus,
            compress_bus: self.compress_bus,
            receipt_bus: self.receipt_bus,
        })
    }

    pub fn generate_trace(
        &self,
        entry_digests: &[Digest],
    ) -> Result<ReducedSwirlManifestReconciliationTraceArtifacts, &'static str> {
        self.validate()?;
        generate_reduced_swirl_manifest_reconciliation_trace(
            self.input_arity,
            self.source_protocol_digest,
            entry_digests,
        )
    }
}

pub struct ReducedSwirlManifestReconciliationTraceArtifacts {
    pub trace: RowMajorMatrix<F>,
    pub flat_manifest_log: TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    pub local_manifest_logs: Vec<TranscriptLog<F, [F; POSEIDON2_WIDTH]>>,
    pub compression_inputs: Vec<[F; 2 * DIGEST_SIZE]>,
    pub receipt: ReducedSwirlManifestReconciliationReceiptMessage<F>,
}

/// Generate the fixed 1024-row reconciliation witness.  The schedule comes
/// from the backend `LinearChainSchedule`; no caller-supplied call boundaries
/// or success flags are accepted.
pub fn generate_reduced_swirl_manifest_reconciliation_trace(
    input_arity: usize,
    source_protocol_digest: Digest,
    entry_digests: &[Digest],
) -> Result<ReducedSwirlManifestReconciliationTraceArtifacts, &'static str> {
    if !(2..=REDUCED_SWIRL_VACC_MAX_INPUT_ARITY).contains(&input_arity)
        || !input_arity.is_power_of_two()
        || source_protocol_digest.iter().all(|value| *value == F::ZERO)
        || entry_digests.is_empty()
        || entry_digests.len() > REDUCED_SWIRL_MANIFEST_RECONCILIATION_MAX_SOURCES
        || entry_digests
            .iter()
            .any(|digest| digest.iter().all(|value| *value == F::ZERO))
    {
        return Err("reduced-SWIRL manifest reconciliation inventory");
    }
    let calls = reduced_swirl_vacc_schedule(entry_digests.len(), input_arity)?;
    let (flat_manifest_digest, flat_manifest_log) =
        reduced_swirl_manifest_digest_with_log(entry_digests)?;
    let width = ReducedSwirlManifestReconciliationCols::<F>::width();
    let height = adapter_trace_height(
        entry_digests.len(),
        REDUCED_SWIRL_MANIFEST_RECONCILIATION_MAX_SOURCES,
    );
    let mut values = F::zero_vec(width * height);
    let mut local_manifest_logs = Vec::with_capacity(calls.len());
    let mut compression_inputs = Vec::with_capacity(1 + calls.len() * 2);
    let genesis_metadata = core::array::from_fn(|index| match index {
        0 => F::from_u32(REDUCED_SWIRL_TRANSITION_CHAIN_GENESIS_TAG),
        1 => F::from_u32(REDUCED_SWIRL_TRANSITION_LEAF_PROTOCOL_VERSION),
        _ => F::ZERO,
    });
    let mut chain_before = record_reconciliation_compression(
        source_protocol_digest,
        genesis_metadata,
        &mut compression_inputs,
    );
    for call in &calls {
        let call_end = call.source_start + call.fresh_count;
        let chunk = &entry_digests[call.source_start..call_end];
        let (local_manifest_digest, local_manifest_log) =
            reduced_swirl_manifest_digest_with_log(chunk)?;
        local_manifest_logs.push(local_manifest_log);
        let metadata = reduced_swirl_reconciliation_chain_metadata(
            entry_digests.len(),
            call.step,
            call.source_start,
            call.fresh_count,
            call.prior_count,
            call_end == entry_digests.len(),
        );
        let chunk_commitment = record_reconciliation_compression(
            metadata,
            local_manifest_digest,
            &mut compression_inputs,
        );
        let chain_after = record_reconciliation_compression(
            chain_before,
            chunk_commitment,
            &mut compression_inputs,
        );
        for (slot, &entry_digest) in chunk.iter().enumerate() {
            let source = call.source_start + slot;
            let cols: &mut ReducedSwirlManifestReconciliationCols<F> =
                values[source * width..(source + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.is_last = F::from_bool(source + 1 == entry_digests.len());
            cols.chunk_first = F::from_bool(slot == 0);
            cols.chunk_last = F::from_bool(slot + 1 == call.fresh_count);
            let capacity = input_arity - call.prior_count;
            let capacity_delta = F::from_usize(slot + 1) - F::from_usize(capacity);
            cols.capacity_end = F::from_bool(slot + 1 == capacity);
            cols.capacity_delta_inverse = if slot + 1 == capacity {
                F::ZERO
            } else {
                capacity_delta.inverse()
            };
            cols.source = F::from_usize(source);
            cols.source_count = F::from_usize(entry_digests.len());
            cols.call_index = F::from_usize(call.step);
            cols.source_start = F::from_usize(call.source_start);
            cols.slot = F::from_usize(slot);
            cols.fresh_count = F::from_usize(call.fresh_count);
            cols.prior_count = F::from_usize(call.prior_count);
            cols.entry_digest = entry_digest;
            let nonzero_limb = entry_digest
                .iter()
                .position(|value| *value != F::ZERO)
                .ok_or("reduced-SWIRL zero manifest entry")?;
            cols.entry_nonzero_selector[nonzero_limb] = F::ONE;
            cols.entry_nonzero_inverse = entry_digest[nonzero_limb].inverse();
            cols.flat_manifest_digest = flat_manifest_digest;
            cols.local_manifest_digest = local_manifest_digest;
            cols.chain_before = chain_before;
            if slot + 1 == call.fresh_count {
                cols.chunk_commitment = chunk_commitment;
                cols.chain_after = chain_after;
            }
        }
        chain_before = chain_after;
    }
    let receipt = ReducedSwirlManifestReconciliationReceiptMessage {
        source_count: F::from_usize(entry_digests.len()),
        call_count: F::from_usize(calls.len()),
        flat_manifest_digest,
        rolling_chain_endpoint: chain_before,
    };
    Ok(ReducedSwirlManifestReconciliationTraceArtifacts {
        trace: RowMajorMatrix::new(values, width),
        flat_manifest_log,
        local_manifest_logs,
        compression_inputs,
        receipt,
    })
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ReducedSwirlVaccTransitionReceiptCols<T> {
    pub active: T,
    pub total_source_count: T,
    pub call_index: T,
    pub source_start: T,
    pub fresh_count: T,
    pub prior_count: T,
    pub batch_start_tidx: T,
    pub vacc_start_tidx: T,
    pub vacc_end_tidx: T,
    pub start_sample_count: T,
    pub start_state: [T; POSEIDON2_WIDTH],
    pub end_sample_count: T,
    pub end_state: [T; POSEIDON2_WIDTH],
    pub prior_root: [T; DIGEST_SIZE],
    pub output_root: [T; DIGEST_SIZE],
    pub prior_digest: [T; DIGEST_SIZE],
    pub output_digest: [T; DIGEST_SIZE],
    pub manifest_digest: [T; DIGEST_SIZE],
    pub is_final: T,
}

/// Joins a one-call WARP endpoint with the manifest of exactly that call's
/// fresh source interval.  The continuations-side leaf boundary consumes this
/// typed receipt together with the inline source-verifier receipt.
#[derive(ColumnsAir)]
#[columns_via(ReducedSwirlVaccTransitionReceiptCols<u8>)]
pub struct ReducedSwirlVaccTransitionReceiptAir {
    pub profile: ReducedSwirlVaccProfile,
    pub transition_end_bus: ReducedSwirlVaccTransitionEndBus,
    pub manifest_digest_bus: ReducedSwirlManifestDigestBus,
    pub chain_end_bus: ReducedSwirlVaccChainEndBus,
    pub receipt_bus: ReducedSwirlVaccTransitionReceiptBus,
}

impl BaseAirWithPublicValues<F> for ReducedSwirlVaccTransitionReceiptAir {}
impl PartitionedBaseAir<F> for ReducedSwirlVaccTransitionReceiptAir {}
impl BaseAir<F> for ReducedSwirlVaccTransitionReceiptAir {
    fn width(&self) -> usize {
        ReducedSwirlVaccTransitionReceiptCols::<F>::width()
    }
}

impl<AB> Air<AB> for ReducedSwirlVaccTransitionReceiptAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.profile.validate().is_ok());
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("reduced-SWIRL transition receipt row");
        let local: &ReducedSwirlVaccTransitionReceiptCols<AB::Var> = (*row).borrow();
        builder.assert_one(local.active);
        builder.assert_bool(local.is_final);

        self.transition_end_bus.receive(
            builder,
            ReducedSwirlVaccTransitionEndMessage {
                total_source_count: local.total_source_count.into(),
                call_index: local.call_index.into(),
                source_start: local.source_start.into(),
                fresh_count: local.fresh_count.into(),
                prior_count: local.prior_count.into(),
                batch_start_tidx: local.batch_start_tidx.into(),
                vacc_start_tidx: local.vacc_start_tidx.into(),
                vacc_end_tidx: local.vacc_end_tidx.into(),
                start_sample_count: local.start_sample_count.into(),
                start_state: local.start_state.map(Into::into),
                end_sample_count: local.end_sample_count.into(),
                end_state: local.end_state.map(Into::into),
                prior_root: local.prior_root.map(Into::into),
                output_root: local.output_root.map(Into::into),
                prior_digest: local.prior_digest.map(Into::into),
                output_digest: local.output_digest.map(Into::into),
                is_final: local.is_final.into(),
            },
            local.active,
        );
        self.manifest_digest_bus.receive(
            builder,
            ReducedSwirlManifestDigestMessage {
                source_count: local.fresh_count.into(),
                digest: local.manifest_digest.map(Into::into),
            },
            local.active,
        );
        self.chain_end_bus.receive(
            builder,
            ReducedSwirlVaccChainEndMessage {
                source_count: local.total_source_count.into(),
                call_count: (AB::Expr::from(local.call_index) + AB::Expr::ONE).into(),
                proof_idx: local.call_index.into(),
                footer_start_tidx: local.vacc_end_tidx.into(),
                final_accumulator_digest: local.output_digest.map(Into::into),
                final_accumulator_root: local.output_root.map(Into::into),
            },
            AB::Expr::from(local.active) * local.is_final,
        );
        self.receipt_bus.add_key_with_lookups(
            builder,
            ReducedSwirlVaccTransitionReceiptMessage {
                protocol_digest: self.profile.protocol_digest.map(AB::Expr::from),
                relation_digest: self.profile.relation_digest.map(AB::Expr::from),
                warp_index_digest: self.profile.warp_index_digest.map(AB::Expr::from),
                schedule_digest: self.profile.schedule_digest.map(AB::Expr::from),
                total_source_count: local.total_source_count.into(),
                call_index: local.call_index.into(),
                source_start: local.source_start.into(),
                fresh_count: local.fresh_count.into(),
                prior_count: local.prior_count.into(),
                batch_start_tidx: local.batch_start_tidx.into(),
                vacc_start_tidx: local.vacc_start_tidx.into(),
                vacc_end_tidx: local.vacc_end_tidx.into(),
                start_sample_count: local.start_sample_count.into(),
                start_state: local.start_state.map(Into::into),
                end_sample_count: local.end_sample_count.into(),
                end_state: local.end_state.map(Into::into),
                prior_root: local.prior_root.map(Into::into),
                output_root: local.output_root.map(Into::into),
                prior_digest: local.prior_digest.map(Into::into),
                output_digest: local.output_digest.map(Into::into),
                manifest_digest: local.manifest_digest.map(Into::into),
                is_final: local.is_final.into(),
            },
            local.active,
        );
    }
}

pub fn generate_reduced_swirl_vacc_transition_receipt_trace(
    profile: &ReducedSwirlVaccProfile,
    total_source_count: usize,
    call: &ReducedSwirlVaccCallRecord,
    manifest_digest: Digest,
) -> Result<
    (
        RowMajorMatrix<F>,
        ReducedSwirlVaccTransitionReceiptMessage<F>,
    ),
    &'static str,
> {
    profile.validate()?;
    let expected = reduced_swirl_vacc_schedule(total_source_count, profile.input_arity)?;
    let expected_call = expected
        .get(call.call.step)
        .ok_or("reduced-SWIRL transition receipt call")?;
    if &call.call != expected_call
        || manifest_digest.iter().all(|value| *value == F::ZERO)
        || call.output_root.iter().all(|value| *value == F::ZERO)
        || call.output_digest.iter().all(|value| *value == F::ZERO)
    {
        return Err("reduced-SWIRL transition receipt");
    }
    let is_final = call.call.step + 1 == expected.len();
    let prior_root = call.prior_root.unwrap_or([F::ZERO; DIGEST_SIZE]);
    let prior_digest = call.prior_digest.unwrap_or([F::ZERO; DIGEST_SIZE]);
    let receipt = ReducedSwirlVaccTransitionReceiptMessage {
        protocol_digest: profile.protocol_digest,
        relation_digest: profile.relation_digest,
        warp_index_digest: profile.warp_index_digest,
        schedule_digest: profile.schedule_digest,
        total_source_count: F::from_usize(total_source_count),
        call_index: F::from_usize(call.call.step),
        source_start: F::from_usize(call.call.source_start),
        fresh_count: F::from_usize(call.call.fresh_count),
        prior_count: F::from_usize(call.call.prior_count),
        batch_start_tidx: F::from_usize(call.batch_start_tidx),
        vacc_start_tidx: F::from_usize(call.vacc_start_tidx),
        vacc_end_tidx: F::from_usize(call.vacc_end_tidx),
        start_sample_count: F::from_usize(call.start_sample_count),
        start_state: call.start_state,
        end_sample_count: F::from_usize(call.end_sample_count),
        end_state: call.end_state,
        prior_root,
        output_root: call.output_root,
        prior_digest,
        output_digest: call.output_digest,
        manifest_digest,
        is_final: F::from_bool(is_final),
    };
    let width = ReducedSwirlVaccTransitionReceiptCols::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut ReducedSwirlVaccTransitionReceiptCols<F> = values.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.total_source_count = receipt.total_source_count;
    cols.call_index = receipt.call_index;
    cols.source_start = receipt.source_start;
    cols.fresh_count = receipt.fresh_count;
    cols.prior_count = receipt.prior_count;
    cols.batch_start_tidx = receipt.batch_start_tidx;
    cols.vacc_start_tidx = receipt.vacc_start_tidx;
    cols.vacc_end_tidx = receipt.vacc_end_tidx;
    cols.start_sample_count = receipt.start_sample_count;
    cols.start_state = receipt.start_state;
    cols.end_sample_count = receipt.end_sample_count;
    cols.end_state = receipt.end_state;
    cols.prior_root = receipt.prior_root;
    cols.output_root = receipt.output_root;
    cols.prior_digest = receipt.prior_digest;
    cols.output_digest = receipt.output_digest;
    cols.manifest_digest = receipt.manifest_digest;
    cols.is_final = receipt.is_final;
    Ok((RowMajorMatrix::new(values, width), receipt))
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ReducedSwirlVaccFooterCols<T> {
    pub active: T,
    pub source_count: T,
    pub call_count: T,
    pub proof_idx: T,
    pub local_proof_idx: T,
    pub start_tidx: T,
    pub end_tidx: T,
    pub manifest_digest: [T; DIGEST_SIZE],
    pub final_accumulator_digest: [T; DIGEST_SIZE],
    pub final_accumulator_root: [T; DIGEST_SIZE],
}

#[derive(ColumnsAir)]
#[columns_via(ReducedSwirlVaccFooterCols<u8>)]
pub struct ReducedSwirlVaccFooterAir {
    pub profile: ReducedSwirlVaccProfile,
    pub transcript_bus: TranscriptBus,
    pub chain_end_bus: ReducedSwirlVaccChainEndBus,
    pub manifest_digest_bus: ReducedSwirlManifestDigestBus,
    pub footer_bus: ReducedSwirlVaccFooterBus,
    pub receipt_bus: ReducedSwirlVaccChainReceiptBus,
}

impl BaseAirWithPublicValues<F> for ReducedSwirlVaccFooterAir {}
impl PartitionedBaseAir<F> for ReducedSwirlVaccFooterAir {}
impl BaseAir<F> for ReducedSwirlVaccFooterAir {
    fn width(&self) -> usize {
        ReducedSwirlVaccFooterCols::<F>::width()
    }
}

impl<AB> Air<AB> for ReducedSwirlVaccFooterAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.profile.validate().is_ok());
        let main = builder.main();
        let row = main.row_slice(0).expect("reduced-SWIRL footer row");
        let local: &ReducedSwirlVaccFooterCols<AB::Var> = (*row).borrow();
        builder.assert_one(local.active);
        self.chain_end_bus.receive(
            builder,
            ReducedSwirlVaccChainEndMessage {
                source_count: local.source_count.into(),
                call_count: local.call_count.into(),
                proof_idx: local.proof_idx.into(),
                footer_start_tidx: local.start_tidx.into(),
                final_accumulator_digest: local.final_accumulator_digest.map(Into::into),
                final_accumulator_root: local.final_accumulator_root.map(Into::into),
            },
            local.active,
        );
        self.manifest_digest_bus.receive(
            builder,
            ReducedSwirlManifestDigestMessage {
                source_count: local.source_count.into(),
                digest: local.manifest_digest.map(Into::into),
            },
            local.active,
        );
        let mut tidx = AB::Expr::from(local.start_tidx);
        observe_const_bytes_ext(
            &self.transcript_bus,
            builder,
            local.local_proof_idx.into(),
            &mut tidx,
            REDUCED_SWIRL_VACC_FOOTER_TAG,
            true,
            AB::Expr::ONE,
        );
        for &limb in &local.manifest_digest {
            observe_ext_expr(
                &self.transcript_bus,
                builder,
                local.local_proof_idx.into(),
                &mut tidx,
                [limb.into(), AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO],
                AB::Expr::ONE,
            );
        }
        builder.assert_eq(local.end_tidx, tidx);
        self.footer_bus.send(
            builder,
            ReducedSwirlVaccFooterMessage {
                source_count: local.source_count.into(),
                call_count: local.call_count.into(),
                start_tidx: local.start_tidx.into(),
                end_tidx: local.end_tidx.into(),
                manifest_digest: local.manifest_digest.map(Into::into),
            },
            local.active,
        );
        self.receipt_bus.add_key_with_lookups(
            builder,
            ReducedSwirlVaccChainReceiptMessage {
                protocol_digest: self.profile.protocol_digest.map(AB::Expr::from),
                relation_digest: self.profile.relation_digest.map(AB::Expr::from),
                warp_index_digest: self.profile.warp_index_digest.map(AB::Expr::from),
                schedule_digest: self.profile.schedule_digest.map(AB::Expr::from),
                manifest_digest: local.manifest_digest.map(Into::into),
                source_count: local.source_count.into(),
                call_count: local.call_count.into(),
                final_accumulator_digest: local.final_accumulator_digest.map(Into::into),
                final_accumulator_root: local.final_accumulator_root.map(Into::into),
            },
            local.active,
        );
    }
}

pub fn generate_reduced_swirl_vacc_footer_range_trace(
    profile: &ReducedSwirlVaccProfile,
    source_count: usize,
    call_count: usize,
    last: &ReducedSwirlVaccCallRecord,
    manifest_digest: Digest,
    local_proof_idx: usize,
) -> Result<RowMajorMatrix<F>, &'static str> {
    if call_count == 0 || last.call.step + 1 != call_count {
        return Err("reduced-SWIRL VACC footer call count");
    }
    generate_reduced_swirl_vacc_footer_record_trace(
        profile,
        source_count,
        call_count,
        last.call.step,
        local_proof_idx,
        last.vacc_end_tidx,
        last.vacc_end_tidx
            .checked_add(
                reduced_swirl_vacc_footer_elements(source_count, manifest_digest)?
                    .len()
                    .checked_mul(D_EF)
                    .ok_or("reduced-SWIRL footer width overflow")?,
            )
            .ok_or("reduced-SWIRL footer interval overflow")?,
        manifest_digest,
        last.output_digest,
        last.output_root,
    )
}

/// Generate the one-row VACC footer from an already authenticated terminal
/// endpoint.
///
/// A streaming transition-tree finalizer no longer retains the complete
/// `ReducedSwirlVaccCallRecord` vector. Its finalizer AIR supplies the exact
/// call/source counts, transcript cursor, flat manifest, and accumulator
/// endpoint on typed buses. This constructor materializes only the matching
/// footer row; it does not infer or accept a host success flag.
#[allow(clippy::too_many_arguments)]
pub fn generate_reduced_swirl_vacc_footer_record_trace(
    profile: &ReducedSwirlVaccProfile,
    source_count: usize,
    call_count: usize,
    proof_idx: usize,
    local_proof_idx: usize,
    start_tidx: usize,
    end_tidx: usize,
    manifest_digest: Digest,
    final_accumulator_digest: Digest,
    final_accumulator_root: Digest,
) -> Result<RowMajorMatrix<F>, &'static str> {
    profile.validate()?;
    let footer = reduced_swirl_vacc_footer_elements(source_count, manifest_digest)?;
    let footer_width = footer
        .len()
        .checked_mul(D_EF)
        .ok_or("reduced-SWIRL footer width overflow")?;
    if source_count == 0
        || call_count == 0
        || proof_idx + 1 != call_count
        || end_tidx
            != start_tidx
                .checked_add(footer_width)
                .ok_or("reduced-SWIRL footer interval overflow")?
        || manifest_digest.iter().all(|value| *value == F::ZERO)
        || final_accumulator_digest
            .iter()
            .all(|value| *value == F::ZERO)
        || final_accumulator_root.iter().all(|value| *value == F::ZERO)
    {
        return Err("reduced-SWIRL VACC footer record");
    }
    let width = ReducedSwirlVaccFooterCols::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut ReducedSwirlVaccFooterCols<F> = values.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.source_count = F::from_usize(source_count);
    cols.call_count = F::from_usize(call_count);
    cols.proof_idx = F::from_usize(proof_idx);
    cols.local_proof_idx = F::from_usize(local_proof_idx);
    cols.start_tidx = F::from_usize(start_tidx);
    cols.end_tidx = F::from_usize(end_tidx);
    cols.manifest_digest = manifest_digest;
    cols.final_accumulator_digest = final_accumulator_digest;
    cols.final_accumulator_root = final_accumulator_root;
    Ok(RowMajorMatrix::new(values, width))
}

/// All buses needed by the recursion-local adapter. The standard VACC buses
/// are outputs of the transition verifier; the reduced-SWIRL buses
/// are supplied by the retained-prefix source verifier and consumed here.
#[derive(Clone, Copy, Debug)]
pub struct ReducedSwirlVaccTransitionBuses {
    pub main_transcript: TranscriptBus,
    pub digest_transcript: TranscriptBus,
    pub manifest_transcript: TranscriptBus,
    pub phase_cursor: NativeVaccPhaseCursorBus,
    pub certified_checkpoint: CertifiedTranscriptCheckpointBus,
    pub resume_state: ResumeTranscriptStateBus,
    pub transcript_end_index: crate::bus::TranscriptEndIndexBus,
    pub standard_vacc_end: NativeStandardVaccEndBus,
    pub standard_vacc_root: NativeStandardVaccRootBus,
    pub standard_vacc_digest: NativeStandardVaccDigestBus,
    pub direct_fresh_source: NativeDirectFreshSourceBus,
    pub direct_fresh_root: NativeDirectFreshRootBus,
    pub vacc_claim: NativeClaimValueBus,
    pub input_slot_layout: NativeInputSlotLayoutBus,
    pub fresh_count: NativeFreshCountBus,
    pub source_claim: ReducedSwirlSourceClaimBus,
    pub source_root: ReducedSwirlSourceRootBus,
    pub source_beta: ReducedSwirlSourceBetaBus,
    pub source_authority: ReducedSwirlSourceAuthorityBus,
    pub call: ReducedSwirlVaccCallBus,
    pub source_slot: ReducedSwirlVaccSourceSlotBus,
    pub source_entry_digest: ReducedSwirlSourceEntryDigestBus,
    pub header_end: ReducedSwirlVaccHeaderEndBus,
    pub manifest_digest: ReducedSwirlManifestDigestBus,
    pub chain_end: ReducedSwirlVaccChainEndBus,
}

/// Setup-fixed aggregate for one physical recursive leaf.  It proves one
/// canonical global WARP transition while rebasing only AIR/transcript proof
/// namespaces to zero.  Source verification is inline and bounded by the
/// WARP input arity; whole-block footer/terminal work is deliberately absent.
#[derive(Clone, Debug)]
pub struct ReducedSwirlVaccTransitionAggregateComponent {
    pub profile: ReducedSwirlVaccProfile,
    pub buses: ReducedSwirlVaccTransitionBuses,
    pub root_tree_stride: usize,
    pub transition_end_bus: ReducedSwirlVaccTransitionEndBus,
    pub transition_receipt_bus: ReducedSwirlVaccTransitionReceiptBus,
}

impl ReducedSwirlVaccTransitionAggregateComponent {
    pub fn validate(&self) -> Result<(), &'static str> {
        self.profile.validate()?;
        if self.root_tree_stride == 0 {
            return Err("reduced-SWIRL transition aggregate wiring");
        }
        Ok(())
    }

    #[must_use]
    pub fn airs<PCS>(&self) -> Vec<AirRef<PCS>>
    where
        PCS: StarkProtocolConfig<F = F>,
    {
        self.validate()
            .expect("valid reduced-SWIRL transition aggregate");
        vec![
            Arc::new(ReducedSwirlVaccHeaderAir {
                profile: self.profile.clone(),
                transcript_bus: self.buses.main_transcript,
                end_bus: self.buses.header_end,
            }),
            Arc::new(ReducedSwirlVaccBatchAir {
                transcript_bus: self.buses.main_transcript,
                call_bus: self.buses.call,
            }),
            Arc::new(ReducedSwirlVaccPrefixAir {
                transcript_bus: self.buses.main_transcript,
                phase_cursor_bus: self.buses.phase_cursor,
                call_bus: self.buses.call,
                profile: self.profile.clone(),
                require_genesis_first: false,
            }),
            Arc::new(ReducedSwirlVaccScheduleAir {
                profile: self.profile.clone(),
                call_bus: self.buses.call,
                slot_bus: self.buses.source_slot,
                input_slot_bus: self.buses.input_slot_layout,
                fresh_count_bus: self.buses.fresh_count,
                header_end_bus: self.buses.header_end,
                vacc_end_bus: self.buses.standard_vacc_end,
                vacc_root_bus: self.buses.standard_vacc_root,
                vacc_digest_bus: self.buses.standard_vacc_digest,
                checkpoint_bus: self.buses.certified_checkpoint,
                resume_bus: self.buses.resume_state,
                transcript_end_index_bus: self.buses.transcript_end_index,
                chain_end_bus: self.buses.chain_end,
                require_complete_chain: false,
                leaf_physical_end: true,
                transition_end_bus: Some(self.transition_end_bus),
            }),
            Arc::new(ReducedSwirlSourceDigestAir {
                profile: self.profile.clone(),
                root_tree_stride: self.root_tree_stride,
                digest_transcript_bus: self.buses.digest_transcript,
                main_transcript_bus: self.buses.main_transcript,
                slot_bus: self.buses.source_slot,
                authority_bus: self.buses.source_authority,
                source_claim_bus: self.buses.source_claim,
                source_root_bus: self.buses.source_root,
                source_beta_bus: self.buses.source_beta,
                direct_source_bus: self.buses.direct_fresh_source,
                direct_root_bus: self.buses.direct_fresh_root,
                vacc_claim_bus: self.buses.vacc_claim,
                entry_digest_bus: self.buses.source_entry_digest,
            }),
            Arc::new(ReducedSwirlManifestDigestAir {
                maximum_sources: self.profile.input_arity,
                transcript_bus: self.buses.manifest_transcript,
                entry_digest_bus: self.buses.source_entry_digest,
                manifest_digest_bus: self.buses.manifest_digest,
            }),
            Arc::new(ReducedSwirlVaccTransitionReceiptAir {
                profile: self.profile.clone(),
                transition_end_bus: self.transition_end_bus,
                manifest_digest_bus: self.buses.manifest_digest,
                chain_end_bus: self.buses.chain_end,
                receipt_bus: self.transition_receipt_bus,
            }),
        ]
    }

    pub fn generate_traces(
        &self,
        record: &ReducedSwirlVaccTransitionAggregateRecord,
    ) -> Result<ReducedSwirlVaccTransitionAggregateTraceArtifacts, &'static str> {
        self.validate()?;
        if record.sources.len() != record.call.call.fresh_count
            || record.sources.len() > self.profile.input_arity
        {
            return Err("reduced-SWIRL transition source count");
        }
        let call = core::slice::from_ref(&record.call);
        let header = generate_reduced_swirl_vacc_optional_header_trace(
            &self.profile,
            record.total_source_count,
            record.call.call.step == 0,
        )?;
        let batch = generate_reduced_swirl_vacc_batch_range_trace(call, 1)?;
        let prefix = generate_reduced_swirl_vacc_prefix_range_trace(call, 1)?;
        let schedule = generate_reduced_swirl_vacc_schedule_range_trace(
            &self.profile,
            record.total_source_count,
            call,
            1,
        )?;
        let source = generate_reduced_swirl_source_digest_range_trace(
            &self.profile,
            self.root_tree_stride,
            record.call.call.source_start,
            call,
            &record.sources,
            self.profile.input_arity,
        )?;
        let (manifest, manifest_digest, manifest_transcript_log) =
            generate_reduced_swirl_manifest_digest_range_trace(
                record.call.call.source_start,
                self.profile.input_arity,
                &source.entry_digests,
            )?;
        let (receipt, receipt_message) = generate_reduced_swirl_vacc_transition_receipt_trace(
            &self.profile,
            record.total_source_count,
            &record.call,
            manifest_digest,
        )?;
        Ok(ReducedSwirlVaccTransitionAggregateTraceArtifacts {
            traces: vec![
                header,
                batch,
                prefix,
                schedule,
                source.trace,
                manifest,
                receipt,
            ],
            digest_transcript_logs: source.transcript_logs,
            manifest_transcript_log,
            entry_digests: source.entry_digests,
            manifest_digest,
            receipt: receipt_message,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlVaccTransitionAggregateRecord {
    pub total_source_count: usize,
    pub call: ReducedSwirlVaccCallRecord,
    pub sources: Vec<ReducedSwirlSourceDigestRecord>,
}

pub struct ReducedSwirlVaccTransitionAggregateTraceArtifacts {
    pub traces: Vec<RowMajorMatrix<F>>,
    pub digest_transcript_logs: Vec<TranscriptLog<F, [F; POSEIDON2_WIDTH]>>,
    pub manifest_transcript_log: TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    pub entry_digests: Vec<Digest>,
    pub manifest_digest: Digest,
    pub receipt: ReducedSwirlVaccTransitionReceiptMessage<F>,
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;

    use openvm_stark_backend::{
        air_builders::{debug::check_constraints, symbolic::get_symbolic_builder},
        keygen::types::TraceWidth,
    };

    use super::*;

    fn digest(seed: u32) -> Digest {
        core::array::from_fn(|index| F::from_u32(seed + index as u32 + 1))
    }

    fn profile(arity: usize) -> ReducedSwirlVaccProfile {
        ReducedSwirlVaccProfile {
            maximum_sources: 1024,
            input_arity: arity,
            num_ood: 1,
            num_shift_queries: 2,
            batching_arity: 4,
            family_target_bits: 80,
            source: ReducedSwirlSourceProfile {
                maximum_sources: 1024,
                maximum_roots_per_source: 3,
                maximum_openings_per_source: 8,
                l_skip: 1,
                n_stack: 2,
                log_blowup: 1,
                log_commit_rows_per_query: 1,
            },
            source_domain: b"swirl".to_vec(),
            relation_binding: vec![EF::from_u32(3)],
            code_binding: vec![EF::from_u32(5)],
            external_protocol_binding: vec![EF::from_u32(7), EF::from_u32(11)],
            protocol_digest: digest(10),
            relation_digest: digest(30),
            warp_index_digest: digest(50),
            schedule_digest: digest(70),
        }
    }

    #[test]
    fn schedule_matches_backend_and_keeps_final_partial_batch() {
        for arity in [2, 8, 32, 64] {
            for sources in [1, arity - 1, arity, arity + 1, 100, 429, 1024] {
                let calls = reduced_swirl_vacc_schedule(sources, arity).unwrap();
                let backend = LinearChainSchedule::new(arity)
                    .unwrap()
                    .step_fresh_counts(sources);
                assert_eq!(
                    calls
                        .iter()
                        .map(|call| call.fresh_count)
                        .collect::<Vec<_>>(),
                    backend
                );
                assert_eq!(
                    calls.last().unwrap().source_start + calls.last().unwrap().fresh_count,
                    sources
                );
                assert!(calls.iter().skip(1).all(|call| call.prior_count == 1));
            }
        }
        let calls = reduced_swirl_vacc_schedule(100, 32).unwrap();
        assert_eq!(
            calls
                .iter()
                .map(|call| call.fresh_count)
                .collect::<Vec<_>>(),
            vec![32, 31, 31, 6]
        );
    }

    #[test]
    fn corrected_header_has_no_manifest_slot() {
        let profile = profile(32);
        let header = reduced_swirl_vacc_header_elements(&profile, 100).unwrap();
        let changed = reduced_swirl_vacc_header_elements(&profile, 101).unwrap();
        assert_ne!(header, changed);
        assert!(header.len() < 256);
        let prefix = reduced_swirl_vacc_protocol_prefix_elements(
            &profile,
            reduced_swirl_vacc_schedule(100, 32).unwrap()[3],
        )
        .unwrap();
        assert_eq!(
            &prefix[..NATIVE_WARP_VACC_PROTOCOL_TAG.len()],
            &NATIVE_WARP_VACC_PROTOCOL_TAG
                .iter()
                .copied()
                .map(EF::from_u8)
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn manifest_is_order_count_and_entry_sensitive() {
        let entries = vec![digest(1), digest(20), digest(40)];
        let honest = reduced_swirl_manifest_digest(&entries).unwrap();
        let mut swapped = entries.clone();
        swapped.swap(0, 1);
        assert_ne!(honest, reduced_swirl_manifest_digest(&swapped).unwrap());
        assert_ne!(
            honest,
            reduced_swirl_manifest_digest(&entries[..2]).unwrap()
        );
        let mut changed = entries.clone();
        changed[1][3] += F::ONE;
        assert_ne!(honest, reduced_swirl_manifest_digest(&changed).unwrap());
        assert!(reduced_swirl_manifest_digest(&[]).is_err());
    }

    #[test]
    fn source_entry_is_canonical_manifest_while_schedule_and_checkpoint_are_external() {
        let profile = profile(8);
        let call = reduced_swirl_vacc_schedule(9, 8).unwrap()[0];
        let claim = RecursiveReducedSwirlClaim {
            roots: vec![digest(100), digest(120)],
            widths: vec![2, 3],
            theta: EF::from_u32(5),
            alpha: vec![EF::ZERO; profile.source.log_codeword_len()],
            mu: EF::from_u32(6),
            beta: vec![EF::from_u32(7); profile.source.log_message_len()],
            eta: EF::from_u32(8),
        };
        let authority = ReducedSwirlSourceAuthorityRecord {
            segment_index: 17,
            common_main_root: digest(130),
            trace_layout_digest: digest(135),
            pending_claim_digest: digest(138),
            checkpoint_tidx: 77,
            checkpoint_state: core::array::from_fn(|i| F::from_usize(i + 1)),
            program_commitment: digest(140),
            initial_pc: F::from_u32(4),
            initial_root: digest(160),
            final_pc: F::from_u32(8),
            final_root: digest(180),
            exit_code: F::ZERO,
            is_terminate: F::ONE,
        };
        let honest =
            reduced_swirl_source_entry_digest(&profile, 0, call, 0, &claim, &authority).unwrap();
        let mut changed = authority.clone();
        changed.final_pc += F::from_u32(4);
        let changed =
            reduced_swirl_source_entry_digest(&profile, 0, call, 0, &claim, &changed).unwrap();
        assert_ne!(honest, changed);
        let mut checkpoint_only = authority.clone();
        checkpoint_only.checkpoint_tidx += 4;
        checkpoint_only.checkpoint_state[0] += F::ONE;
        assert_eq!(
            honest,
            reduced_swirl_source_entry_digest(&profile, 0, call, 0, &claim, &checkpoint_only,)
                .unwrap()
        );
        assert!(
            reduced_swirl_source_entry_digest(&profile, 1, call, 0, &claim, &authority).is_err()
        );

        let mut changed_claim = claim.clone();
        changed_claim.widths[0] += 1;
        assert_ne!(
            honest,
            reduced_swirl_source_entry_digest(&profile, 0, call, 0, &changed_claim, &authority)
                .unwrap()
        );
    }

    fn call_and_source_records(
        profile: &ReducedSwirlVaccProfile,
        source_count: usize,
    ) -> (
        Vec<ReducedSwirlVaccCallRecord>,
        Vec<ReducedSwirlSourceDigestRecord>,
    ) {
        let schedule = reduced_swirl_vacc_schedule(source_count, profile.input_arity).unwrap();
        let mut calls = Vec::new();
        let mut batch_start = reduced_swirl_vacc_header_elements(profile, source_count)
            .unwrap()
            .len()
            * D_EF;
        let mut previous_root = None;
        let mut previous_digest = None;
        let mut start_sample_count = 0usize;
        let mut start_state = [F::ZERO; POSEIDON2_WIDTH];
        for call in schedule {
            let vacc_start = batch_start
                + (8 + REDUCED_SWIRL_VACC_SOURCE_BATCH_TAG.len()
                    + 3 * 8
                    + DIGEST_SIZE * call.fresh_count)
                    * D_EF;
            let vacc_end = vacc_start + 512;
            let output_root = digest(300 + call.step as u32 * 20);
            let output_digest = digest(500 + call.step as u32 * 20);
            let end_state = core::array::from_fn(|limb| {
                F::from_usize(700 + call.step * POSEIDON2_WIDTH + limb)
            });
            calls.push(ReducedSwirlVaccCallRecord {
                call,
                batch_start_tidx: batch_start,
                vacc_start_tidx: vacc_start,
                vacc_end_tidx: vacc_end,
                start_sample_count,
                start_state,
                end_sample_count: 1,
                end_state,
                prior_root: previous_root,
                output_root,
                prior_digest: previous_digest,
                output_digest,
            });
            batch_start = vacc_end;
            previous_root = Some(output_root);
            previous_digest = Some(output_digest);
            start_sample_count = 1;
            start_state = end_state;
        }
        let sources = (0..source_count)
            .map(|source| ReducedSwirlSourceDigestRecord {
                claim: RecursiveReducedSwirlClaim {
                    roots: vec![digest(1_000 + source as u32 * 20)],
                    widths: vec![2],
                    theta: EF::from_usize(10 + source),
                    alpha: vec![EF::ZERO; profile.source.log_codeword_len()],
                    mu: EF::from_usize(20 + source),
                    beta: vec![EF::from_usize(30 + source); profile.source.log_message_len()],
                    eta: EF::from_usize(40 + source),
                },
                authority: ReducedSwirlSourceAuthorityRecord {
                    segment_index: source as u32,
                    common_main_root: digest(2_000 + source as u32 * 20),
                    trace_layout_digest: digest(3_000 + source as u32 * 20),
                    pending_claim_digest: digest(4_000 + source as u32 * 20),
                    checkpoint_tidx: 100 + source,
                    checkpoint_state: core::array::from_fn(|limb| {
                        F::from_usize(5_000 + source * POSEIDON2_WIDTH + limb)
                    }),
                    program_commitment: digest(6_000),
                    initial_pc: F::from_usize(source * 4),
                    initial_root: digest(7_000 + source as u32 * 20),
                    final_pc: F::from_usize((source + 1) * 4),
                    final_root: digest(8_000 + source as u32 * 20),
                    exit_code: F::ZERO,
                    is_terminate: F::from_bool(source + 1 == source_count),
                },
                commitment_tidx: 64,
                first_tree_id: (source * 8) as u32,
            })
            .collect();
        (calls, sources)
    }

    #[test]
    fn bounded_source_digest_range_keeps_global_manifest_index_and_local_namespace() {
        let mut profile = profile(8);
        profile.maximum_sources = 16;
        profile.source.maximum_sources = 16;
        let (calls, sources) = call_and_source_records(&profile, 9);
        let complete =
            generate_reduced_swirl_source_digest_trace(&profile, 1, &calls, &sources).unwrap();
        let bounded = generate_reduced_swirl_source_digest_range_trace(
            &profile,
            1,
            8,
            &calls[1..],
            &sources[8..],
            8,
        )
        .unwrap();
        assert_eq!(bounded.entry_digests, complete.entry_digests[8..]);
        assert_eq!(bounded.transcript_logs.len(), 2);
        let row = bounded.trace.row_slice(0).unwrap();
        let cols: &ReducedSwirlSourceDigestCols<F> = (*row).borrow();
        assert_eq!(cols.source, F::ZERO);
        assert_eq!(cols.source_offset, F::from_usize(8));
        assert_eq!(cols.step, F::ONE);
        assert_eq!(cols.slot, F::ZERO);
        // Moving the same physical row into a different global interval must
        // not silently change its canonical source identity.
        assert!(generate_reduced_swirl_source_digest_range_trace(
            &profile,
            1,
            7,
            &calls[1..],
            &sources[8..],
            8,
        )
        .is_err());
    }

    fn reconciliation_air(
        input_arity: usize,
        source_protocol_digest: Digest,
    ) -> ReducedSwirlManifestReconciliationAir {
        ReducedSwirlManifestReconciliationAir {
            input_arity,
            source_protocol_digest,
            transcript_bus: TranscriptBus::new(41),
            compress_bus: Poseidon2CompressBus::new(43),
            receipt_bus: ReducedSwirlManifestReconciliationReceiptBus::new(44),
        }
    }

    #[test]
    fn manifest_reconciliation_matches_flat_and_linear_chain_oracles() {
        let entries = (0..18)
            .map(|source| digest(10_000 + source * 20))
            .collect::<Vec<_>>();
        let protocol = digest(20_000);
        let artifacts =
            generate_reduced_swirl_manifest_reconciliation_trace(8, protocol, &entries).unwrap();
        let schedule = reduced_swirl_vacc_schedule(entries.len(), 8).unwrap();
        assert_eq!(
            schedule
                .iter()
                .map(|call| call.fresh_count)
                .collect::<Vec<_>>(),
            vec![8, 7, 3]
        );
        assert_eq!(
            artifacts.receipt.flat_manifest_digest,
            reduced_swirl_manifest_digest(&entries).unwrap()
        );
        assert_eq!(artifacts.receipt.source_count, F::from_usize(18));
        assert_eq!(artifacts.receipt.call_count, F::from_usize(3));
        assert_eq!(artifacts.local_manifest_logs.len(), 3);
        assert_eq!(artifacts.compression_inputs.len(), 7);
        assert_eq!(
            &artifacts.flat_manifest_log.values()
                [artifacts.flat_manifest_log.len() - DIGEST_SIZE..],
            &artifacts.receipt.flat_manifest_digest
        );

        let mut expected = reduced_swirl_reconciliation_chain_genesis(protocol);
        for call in schedule {
            let end = call.source_start + call.fresh_count;
            let local = reduced_swirl_manifest_digest(&entries[call.source_start..end]).unwrap();
            let metadata = reduced_swirl_reconciliation_chain_metadata(
                entries.len(),
                call.step,
                call.source_start,
                call.fresh_count,
                call.prior_count,
                end == entries.len(),
            );
            expected = reduced_swirl_reconciliation_chain_append(expected, metadata, local);
        }
        assert_eq!(artifacts.receipt.rolling_chain_endpoint, expected);
        check_constraints::<_, BabyBearPoseidon2Config>(
            &reconciliation_air(8, protocol),
            "ReducedSwirlManifestReconciliationAir",
            &None,
            &[artifacts.trace.as_view()],
            &[],
        );
    }

    #[test]
    fn manifest_reconciliation_exposes_every_hash_and_transcript_operation() {
        let air = reconciliation_air(8, digest(25_000));
        let interactions = get_symbolic_builder(
            &air,
            &TraceWidth {
                preprocessed: None,
                cached_mains: air.cached_main_widths(),
                common_main: air.common_main_width(),
            },
        )
        .constraints()
        .interactions;
        let count = |bus_index| {
            interactions
                .iter()
                .filter(|interaction| interaction.bus_index == bus_index)
                .count()
        };
        assert_eq!(count(43), 3, "genesis and two call-chain compressions");
        assert_eq!(count(44), 1, "one terminal typed receipt");
        assert!(
            count(41) > 2 * (DIGEST_SIZE + 1),
            "flat and local transcripts"
        );
    }

    #[test]
    fn manifest_reconciliation_rejects_reorder_drop_duplicate_and_zero_inventory() {
        let entries = (0..10)
            .map(|source| digest(30_000 + source * 20))
            .collect::<Vec<_>>();
        let protocol = digest(40_000);
        let honest =
            generate_reduced_swirl_manifest_reconciliation_trace(8, protocol, &entries).unwrap();

        let mut reordered = entries.clone();
        reordered.swap(2, 3);
        let reordered =
            generate_reduced_swirl_manifest_reconciliation_trace(8, protocol, &reordered).unwrap();
        assert_ne!(honest.receipt, reordered.receipt);

        let dropped = generate_reduced_swirl_manifest_reconciliation_trace(
            8,
            protocol,
            &entries[..entries.len() - 1],
        )
        .unwrap();
        assert_ne!(honest.receipt, dropped.receipt);

        let mut duplicated = entries.clone();
        // Preserve the total count while replacing one source by a duplicate;
        // this cannot be detected by count binding alone.
        duplicated[5] = entries[4];
        let duplicated =
            generate_reduced_swirl_manifest_reconciliation_trace(8, protocol, &duplicated).unwrap();
        assert_ne!(honest.receipt, duplicated.receipt);

        assert!(generate_reduced_swirl_manifest_reconciliation_trace(8, protocol, &[]).is_err());
        let mut zero_entry = entries;
        zero_entry[3] = [F::ZERO; DIGEST_SIZE];
        assert!(
            generate_reduced_swirl_manifest_reconciliation_trace(8, protocol, &zero_entry).is_err()
        );

        let maximum = (0..REDUCED_SWIRL_MANIFEST_RECONCILIATION_MAX_SOURCES)
            .map(|source| digest(70_000 + source as u32 * 20))
            .collect::<Vec<_>>();
        let maximum_artifacts =
            generate_reduced_swirl_manifest_reconciliation_trace(8, protocol, &maximum).unwrap();
        assert_eq!(
            maximum_artifacts.receipt.source_count,
            F::from_usize(REDUCED_SWIRL_MANIFEST_RECONCILIATION_MAX_SOURCES)
        );
        let mut too_many = maximum;
        too_many.push(digest(100_000));
        assert!(
            generate_reduced_swirl_manifest_reconciliation_trace(8, protocol, &too_many).is_err()
        );
        for invalid_arity in [0, 1, 3, 65] {
            assert!(generate_reduced_swirl_manifest_reconciliation_trace(
                invalid_arity,
                protocol,
                &[digest(101_000)]
            )
            .is_err());
        }
    }

    #[test]
    fn transition_chain_is_sensitive_to_every_metadata_coordinate() {
        let chain = digest(50_000);
        let local_manifest = digest(51_000);
        let honest_metadata = reduced_swirl_reconciliation_chain_metadata(100, 4, 29, 7, 1, false);
        let honest =
            reduced_swirl_reconciliation_chain_append(chain, honest_metadata, local_manifest);
        let mutations = [
            reduced_swirl_reconciliation_chain_metadata(101, 4, 29, 7, 1, false),
            reduced_swirl_reconciliation_chain_metadata(100, 5, 29, 7, 1, false),
            reduced_swirl_reconciliation_chain_metadata(100, 4, 30, 7, 1, false),
            reduced_swirl_reconciliation_chain_metadata(100, 4, 29, 6, 1, false),
            reduced_swirl_reconciliation_chain_metadata(100, 4, 29, 7, 0, false),
            reduced_swirl_reconciliation_chain_metadata(100, 4, 29, 7, 1, true),
        ];
        for metadata in mutations {
            assert_ne!(
                honest,
                reduced_swirl_reconciliation_chain_append(chain, metadata, local_manifest)
            );
        }
    }

    #[test]
    fn manifest_reconciliation_constraints_reject_schedule_and_chain_end_mutations() {
        let entries = (0..18)
            .map(|source| digest(60_000 + source * 20))
            .collect::<Vec<_>>();
        let protocol = digest(61_000);
        let artifacts =
            generate_reduced_swirl_manifest_reconciliation_trace(8, protocol, &entries).unwrap();
        let air = reconciliation_air(8, protocol);
        let width = ReducedSwirlManifestReconciliationCols::<F>::width();

        let mut wrong_metadata = artifacts.trace.clone();
        let second_call: &mut ReducedSwirlManifestReconciliationCols<F> =
            wrong_metadata.values[8 * width..9 * width].borrow_mut();
        second_call.call_index += F::ONE;
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
            check_constraints::<_, BabyBearPoseidon2Config>(
                &air,
                "ReducedSwirlManifestReconciliationAir",
                &None,
                &[wrong_metadata.as_view()],
                &[],
            );
        }))
        .is_err());

        let mut wrong_chain = artifacts.trace.clone();
        let second_call: &mut ReducedSwirlManifestReconciliationCols<F> =
            wrong_chain.values[8 * width..9 * width].borrow_mut();
        second_call.chain_before[0] += F::ONE;
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
            check_constraints::<_, BabyBearPoseidon2Config>(
                &air,
                "ReducedSwirlManifestReconciliationAir",
                &None,
                &[wrong_chain.as_view()],
                &[],
            );
        }))
        .is_err());

        let mut forged_receipt = artifacts.receipt.clone();
        forged_receipt.rolling_chain_endpoint[0] += F::ONE;
        let last_row = artifacts.trace.row_slice(entries.len() - 1).unwrap();
        let last: &ReducedSwirlManifestReconciliationCols<F> = (*last_row).borrow();
        let terminal_compression = artifacts.compression_inputs.last().unwrap();
        let mut left = [F::ZERO; DIGEST_SIZE];
        let mut right = [F::ZERO; DIGEST_SIZE];
        left.copy_from_slice(&terminal_compression[..DIGEST_SIZE]);
        right.copy_from_slice(&terminal_compression[DIGEST_SIZE..]);
        let independently_recomputed = poseidon2_compress_with_capacity(left, right).0;
        assert_eq!(artifacts.receipt.rolling_chain_endpoint, last.chain_after);
        assert_eq!(
            artifacts.receipt.rolling_chain_endpoint,
            independently_recomputed
        );
        assert_ne!(
            forged_receipt.rolling_chain_endpoint,
            independently_recomputed
        );
    }

    #[test]
    fn pcs_opening_is_not_a_relation_description() {
        assert_ne!(
            openvm_stark_backend::warp_accum::NATIVE_WARP_REDUCED_CONSTRAINED_CODE_PROTOCOL_TAG,
            REDUCED_SWIRL_VACC_SOURCE_BATCH_TAG
        );
        assert_eq!(
            profile(8).source.log_message_len() + 1,
            profile(8).normalized_beta_len()
        );
    }
}

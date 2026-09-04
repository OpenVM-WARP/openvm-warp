//! Constrained owner of the setup-PCS authority statement and transcript gaps.
//!
//! This module deliberately owns only data that ordinary ordered stacking does not own:
//!
//! 1. it receives the certified complete-SWIRL source provenance, the genuine setup PLE point, and
//!    every canonical setup-column claim;
//! 2. it computes the exact SDK v3 transition-statement digest with the BabyBear Poseidon2
//!    padding-free sponge;
//! 3. it observes the exact batch-statement prefix from transcript index zero; and
//! 4. immediately before each ordered reduction, it observes the five-field v3 stacking domain
//!    prefix and publishes the setup-fixed cursor at which that reduction must begin.
//!
//! Ordered stacking remains the sole owner of claim observations, challenges, proof messages and
//! reduced outputs.  Multi-constraint WHIR begins directly at the final stacking `post_tidx`.
//! In particular, this module never consumes the WARP raw one-shot opening point and never hashes
//! a host-provided transition digest as authority.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_cpu_backend::CpuBackend;
use openvm_recursion_circuit::{
    bus::{
        ColumnClaimsBus, ColumnClaimsMessage, Poseidon2CompressBus, Poseidon2CompressMessage,
        Poseidon2PermuteBus, Poseidon2PermuteMessage, TranscriptBus,
    },
    define_typed_permutation_bus,
    utils::poseidon2_hash_slice_with_states,
    whir::multi_constraint::{
        air::{
            MultiConstraintCallerPrefixBus, MultiConstraintCallerPrefixMessage,
            MultiConstraintInitialCommitmentBus, MultiConstraintInitialCommitmentMessage,
        },
        MultiConstraintWhirInitialCommitment,
    },
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    prover::AirProvingContext,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, Digest, DIGEST_SIZE, D_EF, EF, F,
};

use super::{
    FixedSetupOpeningPointBusV2, FixedSetupOpeningPointMessageV2, SetupPcsAuthorityBoundBatchBusV3,
    SetupPcsAuthorityBoundBatchMessageV3, SetupPcsAuthorityBoundTransitionBusV3,
    SetupPcsAuthorityBoundTransitionMessageV3, SetupPcsAuthorityTransitionStatementBusV3,
    SetupPcsAuthorityTransitionStatementMessageV3, VERIFIER_WARP_HISTORY_MAX_TRANSITIONS_V2,
};
use crate::circuit::native_warp_history_v19::{
    SetupPcsSourceProvenanceBusV3, SetupPcsSourceProvenanceMessageV3,
    SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3,
};

pub const SETUP_PCS_AUTHORITY_STATEMENT_PROTOCOL_V3: u32 = 3;

// Mirrored from the SDK v3 wire format.  A differential test below pins the complete preimage,
// including these tags, to the native hash_slice convention.
pub const SETUP_PCS_AUTHORITY_TRANSITION_DIGEST_TAG_V3: u64 = 0x5657_5350_5452_0301;
pub const SETUP_PCS_AUTHORITY_PROFILE_DIGEST_TAG_V3: u64 = 0x5657_5350_5052_0301;
pub const SETUP_PCS_AUTHORITY_BATCH_DIGEST_TAG_V3: u64 = 0x5657_5350_4241_0301;
pub const SETUP_PCS_AUTHORITY_TRANSCRIPT_STATEMENT_TAG_V3: u64 = 0x5657_5350_4653_0301;
pub const SETUP_PCS_AUTHORITY_STACKING_DOMAIN_TAG_V3: u64 = 0x5657_5350_5352_0301;

const POSEIDON_WIDTH: usize = 16;
const POSEIDON_RATE: usize = 8;
const HEADER_VARIABLE_BASE_COUNT: usize = 4 * DIGEST_SIZE + POSEIDON_WIDTH + 16;
const DERIVED_DIGEST_PROFILE_V3: u32 = 1;
const DERIVED_DIGEST_BATCH_V3: u32 = 2;
const DERIVED_DIGEST_TRANSITION_V3: u32 = 3;

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct SetupPcsAuthorityCanonicalFieldMessageV3<T> {
    pub transition_index: [T; 2],
    pub variable_index: [T; 2],
    pub value: T,
}
define_typed_permutation_bus!(
    SetupPcsAuthorityCanonicalFieldBusV3,
    SetupPcsAuthorityCanonicalFieldMessageV3
);

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct SetupPcsAuthoritySourceReceiptMessageV3<T> {
    pub transition_index: [T; 2],
    pub source_receipt_digest: [T; DIGEST_SIZE],
}
define_typed_permutation_bus!(
    SetupPcsAuthoritySourceReceiptBusV3,
    SetupPcsAuthoritySourceReceiptMessageV3
);

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct SetupPcsAuthorityDerivedDigestMessageV3<T> {
    pub kind: T,
    pub transition_index: [T; 2],
    pub digest: [T; DIGEST_SIZE],
}
define_typed_permutation_bus!(
    SetupPcsAuthorityDerivedDigestBusV3,
    SetupPcsAuthorityDerivedDigestMessageV3
);

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct SetupPcsAuthorityClaimValueMessageV3<T> {
    pub transition_index: [T; 2],
    pub claim_ordinal: [T; 2],
    pub current: [T; D_EF],
    pub rotated: [T; D_EF],
}
define_typed_permutation_bus!(
    SetupPcsAuthorityClaimValueBusV3,
    SetupPcsAuthorityClaimValueMessageV3
);

/// Setup-fixed cursors exported to the ordered-stacking assembly.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct SetupPcsAuthorityStackingScheduleMessageV3<T> {
    pub authority_transcript_proof_idx: T,
    pub transition_index: [T; 2],
    pub domain_prefix_tidx: T,
    pub first_claim_tidx: T,
    pub stacking_post_tidx: T,
    pub is_final: T,
    pub caller_prefix_tidx: T,
}
define_typed_permutation_bus!(
    SetupPcsAuthorityStackingScheduleBusV3,
    SetupPcsAuthorityStackingScheduleMessageV3
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetupPcsAuthorityStatementClaimIdentityV3 {
    pub setup_index: u32,
    pub air_id: u32,
    /// Physical part within the fixed-multi-AIR setup descriptor. This is
    /// serialized in the canonical SDK setup-authority statement.
    pub setup_part_index: u32,
    /// SWIRL source-proof coordinates. These identify the ColumnClaims bus
    /// occurrence and are a distinct namespace from `setup_part_index`.
    pub sort_idx: u32,
    pub part_index: u32,
    /// SDK `FixedMultiAirSetupTraceKind`: preprocessed = 1, cached-main = 2.
    pub kind_word: u32,
    pub column_index: u32,
    pub need_rot: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetupPcsAuthorityStackingScheduleV3 {
    pub domain_prefix_tidx: usize,
    pub first_claim_tidx: usize,
    pub stacking_post_tidx: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CanonicalFieldSourceV3 {
    Fixed(F),
    Variable {
        transition_index: u32,
        variable_index: u32,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DigestBlockPlanV3 {
    transition_index: u32,
    is_first: bool,
    is_last: bool,
    sources: [CanonicalFieldSourceV3; POSEIDON_RATE],
    active_slots: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct AuthorityDigestBlockPlanV3 {
    kind: u32,
    is_first: bool,
    is_last: bool,
    sources: [CanonicalFieldSourceV3; POSEIDON_RATE],
    active_slots: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct TranscriptObservationPlanV3 {
    tidx: usize,
    source: CanonicalFieldSourceV3,
    schedule: Option<(u32, SetupPcsAuthorityStackingScheduleV3, bool)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ClaimPlanV3 {
    transition_index: u32,
    identity: SetupPcsAuthorityStatementClaimIdentityV3,
    current_variable_index: u32,
    rotated_variable_index: Option<u32>,
}

/// Aggregation-VK-owned statement shape and exact transcript schedule.
///
/// `trusted_profile_fields` is the exact output of the SDK v3 `profile_fields` serialization.  It
/// is copied into the preprocessed trace; neither its length nor any element is proof data.
#[derive(Clone, Debug)]
pub struct SetupPcsAuthorityStatementProfileV3 {
    expected_app_vk_digest: Digest,
    expected_relation_digest: Digest,
    expected_aggregation_vk_digest: Digest,
    trusted_profile_fields: Arc<[F]>,
    transition_count: usize,
    code_class_index: u32,
    opening_point_len: usize,
    commitment_count: usize,
    initial_commitments: Arc<[MultiConstraintWhirInitialCommitment]>,
    authority_transcript_proof_idx: usize,
    claim_identities: Arc<[SetupPcsAuthorityStatementClaimIdentityV3]>,
    claim_plans: Arc<[ClaimPlanV3]>,
    transition_sources: Arc<[Vec<CanonicalFieldSourceV3>]>,
    variable_counts: Arc<[usize]>,
    digest_blocks: Arc<[DigestBlockPlanV3]>,
    authority_digest_blocks: Arc<[AuthorityDigestBlockPlanV3]>,
    transcript_observations: Arc<[TranscriptObservationPlanV3]>,
    schedules: Arc<[SetupPcsAuthorityStackingScheduleV3]>,
    statement_prefix_end_tidx: usize,
    final_stacking_post_tidx: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SetupPcsAuthorityStatementErrorV3 {
    TransitionCount,
    EmptyProfileFields,
    ProfileVersion,
    ZeroDigest,
    PointLength,
    CommitmentCount,
    InitialCommitment(usize),
    EmptyClaims,
    ClaimKind(usize),
    ClaimOrder(usize),
    ScheduleCount,
    ScheduleCursor(usize),
    TranscriptOverflow,
    VariableOverflow,
    RecordCount,
    ProtocolVersion(usize),
    TransitionIndex(usize),
    SourceProfile(usize),
    SourceDigest(usize),
    CheckpointEncoding(usize),
    OpeningPoint(usize),
    ClaimCount(usize),
    ClaimRotation(usize, usize),
}

impl core::fmt::Display for SetupPcsAuthorityStatementErrorV3 {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "invalid setup-PCS authority statement: {self:?}")
    }
}

impl std::error::Error for SetupPcsAuthorityStatementErrorV3 {}

fn digest_is_zero(digest: &Digest) -> bool {
    digest.iter().all(|&value| value == F::ZERO)
}

fn split_u32(value: u32) -> [F; 2] {
    [F::from_u16(value as u16), F::from_u16((value >> 16) as u16)]
}

fn push_u64_bytes(target: &mut Vec<CanonicalFieldSourceV3>, value: u64) {
    target.extend(
        value
            .to_le_bytes()
            .into_iter()
            .map(|byte| CanonicalFieldSourceV3::Fixed(F::from_u8(byte))),
    );
}

fn push_u32_bytes(target: &mut Vec<CanonicalFieldSourceV3>, value: u32) {
    push_u64_bytes(target, u64::from(value));
}

fn push_len_bytes(
    target: &mut Vec<CanonicalFieldSourceV3>,
    value: usize,
) -> Result<(), SetupPcsAuthorityStatementErrorV3> {
    push_u64_bytes(
        target,
        u64::try_from(value).map_err(|_| SetupPcsAuthorityStatementErrorV3::TranscriptOverflow)?,
    );
    Ok(())
}

fn push_variable_fields(
    target: &mut Vec<CanonicalFieldSourceV3>,
    transition_index: u32,
    cursor: &mut usize,
    count: usize,
) -> Result<u32, SetupPcsAuthorityStatementErrorV3> {
    let start =
        u32::try_from(*cursor).map_err(|_| SetupPcsAuthorityStatementErrorV3::VariableOverflow)?;
    for _ in 0..count {
        target.push(CanonicalFieldSourceV3::Variable {
            transition_index,
            variable_index: u32::try_from(*cursor)
                .map_err(|_| SetupPcsAuthorityStatementErrorV3::VariableOverflow)?,
        });
        *cursor = cursor
            .checked_add(1)
            .ok_or(SetupPcsAuthorityStatementErrorV3::VariableOverflow)?;
    }
    Ok(start)
}

fn validate_claim_identities(
    identities: &[SetupPcsAuthorityStatementClaimIdentityV3],
) -> Result<(), SetupPcsAuthorityStatementErrorV3> {
    if identities.is_empty() {
        return Err(SetupPcsAuthorityStatementErrorV3::EmptyClaims);
    }
    for (index, identity) in identities.iter().enumerate() {
        if !matches!(identity.kind_word, 1 | 2) {
            return Err(SetupPcsAuthorityStatementErrorV3::ClaimKind(index));
        }
        if index == 0 {
            if identity.setup_index != 0 || identity.column_index != 0 {
                return Err(SetupPcsAuthorityStatementErrorV3::ClaimOrder(index));
            }
            continue;
        }
        let previous = identities[index - 1];
        let valid_same_setup = identity.setup_index == previous.setup_index
            && identity.air_id == previous.air_id
            && identity.setup_part_index == previous.setup_part_index
            && identity.part_index == previous.part_index
            && identity.kind_word == previous.kind_word
            && identity.column_index == previous.column_index + 1;
        let valid_next_setup =
            identity.setup_index == previous.setup_index + 1 && identity.column_index == 0;
        if !valid_same_setup && !valid_next_setup {
            return Err(SetupPcsAuthorityStatementErrorV3::ClaimOrder(index));
        }
    }
    Ok(())
}

fn build_transition_sources(
    transition_index: u32,
    code_class_index: u32,
    opening_point_len: usize,
    claims: &[SetupPcsAuthorityStatementClaimIdentityV3],
) -> Result<(Vec<CanonicalFieldSourceV3>, usize, Vec<ClaimPlanV3>), SetupPcsAuthorityStatementErrorV3>
{
    let mut fields = Vec::new();
    let mut variable_cursor = 0usize;
    push_u32_bytes(&mut fields, transition_index);
    // source_root, source_instance_digest, source_forest_root, segment_openings_digest
    push_variable_fields(
        &mut fields,
        transition_index,
        &mut variable_cursor,
        4 * DIGEST_SIZE,
    )?;
    // Certified source checkpoint: state, tidx bytes, sample_count bytes.
    push_variable_fields(
        &mut fields,
        transition_index,
        &mut variable_cursor,
        POSEIDON_WIDTH + 16,
    )?;
    push_len_bytes(&mut fields, 1)?;
    push_u32_bytes(&mut fields, code_class_index);
    push_len_bytes(&mut fields, opening_point_len)?;
    push_variable_fields(
        &mut fields,
        transition_index,
        &mut variable_cursor,
        opening_point_len
            .checked_mul(D_EF)
            .ok_or(SetupPcsAuthorityStatementErrorV3::VariableOverflow)?,
    )?;
    push_len_bytes(&mut fields, claims.len())?;

    let mut plans = Vec::with_capacity(claims.len());
    for &identity in claims {
        push_u32_bytes(&mut fields, identity.setup_index);
        push_u32_bytes(&mut fields, identity.air_id);
        push_u32_bytes(&mut fields, identity.setup_part_index);
        push_u32_bytes(&mut fields, identity.kind_word);
        push_u32_bytes(&mut fields, identity.column_index);
        let current_variable_index =
            push_variable_fields(&mut fields, transition_index, &mut variable_cursor, D_EF)?;
        fields.push(CanonicalFieldSourceV3::Fixed(F::from_bool(
            identity.need_rot,
        )));
        let rotated_variable_index = identity
            .need_rot
            .then(|| {
                push_variable_fields(&mut fields, transition_index, &mut variable_cursor, D_EF)
            })
            .transpose()?;
        plans.push(ClaimPlanV3 {
            transition_index,
            identity,
            current_variable_index,
            rotated_variable_index,
        });
    }
    Ok((fields, variable_cursor, plans))
}

fn build_authority_digest_blocks(
    kind: u32,
    tag: u64,
    payload: &[CanonicalFieldSourceV3],
) -> Result<Vec<AuthorityDigestBlockPlanV3>, SetupPcsAuthorityStatementErrorV3> {
    let mut preimage = Vec::new();
    push_u64_bytes(&mut preimage, tag);
    push_len_bytes(&mut preimage, payload.len())?;
    preimage.extend_from_slice(payload);
    Ok((0..preimage.len().div_ceil(POSEIDON_RATE))
        .map(|block| {
            let start = block * POSEIDON_RATE;
            let active_slots = (preimage.len() - start).min(POSEIDON_RATE);
            AuthorityDigestBlockPlanV3 {
                kind,
                is_first: block == 0,
                is_last: start + active_slots == preimage.len(),
                sources: core::array::from_fn(|slot| {
                    preimage
                        .get(start + slot)
                        .copied()
                        .unwrap_or(CanonicalFieldSourceV3::Fixed(F::ZERO))
                }),
                active_slots,
            }
        })
        .collect())
}

impl SetupPcsAuthorityStatementProfileV3 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        expected_app_vk_digest: Digest,
        expected_relation_digest: Digest,
        expected_aggregation_vk_digest: Digest,
        trusted_profile_fields: Vec<F>,
        transition_count: usize,
        code_class_index: u32,
        opening_point_len: usize,
        initial_commitments: Vec<MultiConstraintWhirInitialCommitment>,
        authority_transcript_proof_idx: usize,
        claim_identities: Vec<SetupPcsAuthorityStatementClaimIdentityV3>,
        stacking_transcript_lengths: Vec<usize>,
    ) -> Result<Self, SetupPcsAuthorityStatementErrorV3> {
        if transition_count == 0 || transition_count > VERIFIER_WARP_HISTORY_MAX_TRANSITIONS_V2 {
            return Err(SetupPcsAuthorityStatementErrorV3::TransitionCount);
        }
        if trusted_profile_fields.len() < 8 {
            return Err(SetupPcsAuthorityStatementErrorV3::EmptyProfileFields);
        }
        let expected_version = SETUP_PCS_AUTHORITY_STATEMENT_PROTOCOL_V3
            .to_le_bytes()
            .into_iter()
            .chain([0u8; 4])
            .map(F::from_u8)
            .collect::<Vec<_>>();
        if trusted_profile_fields[..8] != expected_version {
            return Err(SetupPcsAuthorityStatementErrorV3::ProfileVersion);
        }
        if digest_is_zero(&expected_app_vk_digest)
            || digest_is_zero(&expected_relation_digest)
            || digest_is_zero(&expected_aggregation_vk_digest)
        {
            return Err(SetupPcsAuthorityStatementErrorV3::ZeroDigest);
        }
        if opening_point_len == 0 {
            return Err(SetupPcsAuthorityStatementErrorV3::PointLength);
        }
        if initial_commitments.is_empty() {
            return Err(SetupPcsAuthorityStatementErrorV3::CommitmentCount);
        }
        for (index, commitment) in initial_commitments.iter().enumerate() {
            if commitment.width == 0 || commitment.commitment.iter().all(|&value| value == F::ZERO)
            {
                return Err(SetupPcsAuthorityStatementErrorV3::InitialCommitment(index));
            }
        }
        let commitment_count = initial_commitments.len();
        validate_claim_identities(&claim_identities)?;
        if stacking_transcript_lengths.len() != transition_count {
            return Err(SetupPcsAuthorityStatementErrorV3::ScheduleCount);
        }

        let mut transition_sources = Vec::with_capacity(transition_count);
        let mut variable_counts = Vec::with_capacity(transition_count);
        let mut claim_plans = Vec::with_capacity(
            transition_count
                .checked_mul(claim_identities.len())
                .ok_or(SetupPcsAuthorityStatementErrorV3::VariableOverflow)?,
        );
        for transition in 0..transition_count {
            let transition_index = u32::try_from(transition)
                .map_err(|_| SetupPcsAuthorityStatementErrorV3::TransitionCount)?;
            let (sources, variable_count, plans) = build_transition_sources(
                transition_index,
                code_class_index,
                opening_point_len,
                &claim_identities,
            )?;
            transition_sources.push(sources);
            variable_counts.push(variable_count);
            claim_plans.extend(plans);
        }

        let mut batch_fields = trusted_profile_fields
            .iter()
            .copied()
            .map(CanonicalFieldSourceV3::Fixed)
            .collect::<Vec<_>>();
        push_len_bytes(&mut batch_fields, transition_count)?;
        for sources in &transition_sources {
            push_len_bytes(&mut batch_fields, sources.len())?;
            batch_fields.extend_from_slice(sources);
        }

        let mut transcript_observations = Vec::new();
        let mut prefix = Vec::new();
        push_u64_bytes(&mut prefix, SETUP_PCS_AUTHORITY_TRANSCRIPT_STATEMENT_TAG_V3);
        push_u64_bytes(
            &mut prefix,
            u64::from(SETUP_PCS_AUTHORITY_STATEMENT_PROTOCOL_V3),
        );
        push_len_bytes(&mut prefix, batch_fields.len())?;
        prefix.extend(batch_fields.iter().copied());
        for (tidx, source) in prefix.into_iter().enumerate() {
            transcript_observations.push(TranscriptObservationPlanV3 {
                tidx,
                source,
                schedule: None,
            });
        }
        let statement_prefix_end_tidx = transcript_observations.len();

        let profile_sources = trusted_profile_fields
            .iter()
            .copied()
            .map(CanonicalFieldSourceV3::Fixed)
            .collect::<Vec<_>>();
        let mut authority_digest_blocks = build_authority_digest_blocks(
            DERIVED_DIGEST_PROFILE_V3,
            SETUP_PCS_AUTHORITY_PROFILE_DIGEST_TAG_V3,
            &profile_sources,
        )?;
        authority_digest_blocks.extend(build_authority_digest_blocks(
            DERIVED_DIGEST_BATCH_V3,
            SETUP_PCS_AUTHORITY_BATCH_DIGEST_TAG_V3,
            &batch_fields,
        )?);

        let mut schedules = Vec::with_capacity(transition_count);
        let mut domain_prefix_tidx = statement_prefix_end_tidx;
        for (transition, &stacking_transcript_length) in
            stacking_transcript_lengths.iter().enumerate()
        {
            let first_claim_tidx = domain_prefix_tidx
                .checked_add(5)
                .ok_or(SetupPcsAuthorityStatementErrorV3::TranscriptOverflow)?;
            if stacking_transcript_length == 0 {
                return Err(SetupPcsAuthorityStatementErrorV3::ScheduleCursor(
                    transition,
                ));
            }
            let stacking_post_tidx = first_claim_tidx
                .checked_add(stacking_transcript_length)
                .ok_or(SetupPcsAuthorityStatementErrorV3::TranscriptOverflow)?;
            let schedule = SetupPcsAuthorityStackingScheduleV3 {
                domain_prefix_tidx,
                first_claim_tidx,
                stacking_post_tidx,
            };
            let transition_index = u32::try_from(transition)
                .map_err(|_| SetupPcsAuthorityStatementErrorV3::TransitionCount)?;
            let domain_values = [
                F::from_u64(SETUP_PCS_AUTHORITY_STACKING_DOMAIN_TAG_V3),
                F::from_u32(SETUP_PCS_AUTHORITY_STATEMENT_PROTOCOL_V3),
                F::from_u32(transition_index),
                F::from_u32(code_class_index),
                F::from_usize(commitment_count),
            ];
            for (offset, value) in domain_values.into_iter().enumerate() {
                transcript_observations.push(TranscriptObservationPlanV3 {
                    tidx: domain_prefix_tidx + offset,
                    source: CanonicalFieldSourceV3::Fixed(value),
                    schedule: (offset == 0).then_some((
                        transition_index,
                        schedule,
                        transition + 1 == transition_count,
                    )),
                });
            }
            schedules.push(schedule);
            domain_prefix_tidx = stacking_post_tidx;
        }
        let final_stacking_post_tidx = schedules
            .last()
            .ok_or(SetupPcsAuthorityStatementErrorV3::ScheduleCount)?
            .stacking_post_tidx;
        if final_stacking_post_tidx >= F::ORDER_U32 as usize {
            return Err(SetupPcsAuthorityStatementErrorV3::TranscriptOverflow);
        }

        let mut digest_blocks = Vec::new();
        for (transition, payload) in transition_sources.iter().enumerate() {
            let mut preimage = Vec::new();
            push_u64_bytes(&mut preimage, SETUP_PCS_AUTHORITY_TRANSITION_DIGEST_TAG_V3);
            push_len_bytes(&mut preimage, payload.len())?;
            preimage.extend_from_slice(payload);
            let block_count = preimage.len().div_ceil(POSEIDON_RATE);
            for block in 0..block_count {
                let start = block * POSEIDON_RATE;
                let active_slots = (preimage.len() - start).min(POSEIDON_RATE);
                let sources = core::array::from_fn(|slot| {
                    preimage
                        .get(start + slot)
                        .copied()
                        .unwrap_or(CanonicalFieldSourceV3::Fixed(F::ZERO))
                });
                digest_blocks.push(DigestBlockPlanV3 {
                    transition_index: transition as u32,
                    is_first: block == 0,
                    is_last: block + 1 == block_count,
                    sources,
                    active_slots,
                });
            }
        }

        Ok(Self {
            expected_app_vk_digest,
            expected_relation_digest,
            expected_aggregation_vk_digest,
            trusted_profile_fields: trusted_profile_fields.into(),
            transition_count,
            code_class_index,
            opening_point_len,
            commitment_count,
            initial_commitments: initial_commitments.into(),
            authority_transcript_proof_idx,
            claim_identities: claim_identities.into(),
            claim_plans: claim_plans.into(),
            transition_sources: transition_sources.into(),
            variable_counts: variable_counts.into(),
            digest_blocks: digest_blocks.into(),
            authority_digest_blocks: authority_digest_blocks.into(),
            transcript_observations: transcript_observations.into(),
            schedules: schedules.into(),
            statement_prefix_end_tidx,
            final_stacking_post_tidx,
        })
    }

    #[must_use]
    pub fn transition_count(&self) -> usize {
        self.transition_count
    }

    #[must_use]
    pub fn opening_point_len(&self) -> usize {
        self.opening_point_len
    }

    #[must_use]
    pub fn claim_count_per_transition(&self) -> usize {
        self.claim_identities.len()
    }

    #[must_use]
    pub fn trusted_profile_fields(&self) -> &[F] {
        &self.trusted_profile_fields
    }

    #[must_use]
    pub fn code_class_index(&self) -> u32 {
        self.code_class_index
    }

    #[must_use]
    pub fn commitment_count(&self) -> usize {
        self.commitment_count
    }

    #[must_use]
    pub fn statement_prefix_end_tidx(&self) -> usize {
        self.statement_prefix_end_tidx
    }

    #[must_use]
    pub fn first_claim_tidx(&self, transition: usize) -> Option<usize> {
        self.schedules
            .get(transition)
            .map(|value| value.first_claim_tidx)
    }

    #[must_use]
    pub fn stacking_post_tidx(&self, transition: usize) -> Option<usize> {
        self.schedules
            .get(transition)
            .map(|value| value.stacking_post_tidx)
    }

    /// Exact cursor consumed by `MultiConstraintCallerPrefixBus`.
    #[must_use]
    pub fn caller_prefix_tidx(&self) -> usize {
        self.final_stacking_post_tidx
    }

    #[must_use]
    pub fn source_point_lookup_demands(&self) -> Vec<(u32, u32, u32)> {
        (0..self.transition_count)
            .flat_map(|transition| {
                (0..self.opening_point_len)
                    .map(move |coordinate| (transition as u32, coordinate as u32, 1))
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SetupPcsAuthorityStatementBusesV3 {
    pub source_provenance: SetupPcsSourceProvenanceBusV3,
    pub source_opening_point: FixedSetupOpeningPointBusV2,
    pub column_claims: ColumnClaimsBus,
    pub canonical_fields: SetupPcsAuthorityCanonicalFieldBusV3,
    pub source_receipts: SetupPcsAuthoritySourceReceiptBusV3,
    pub derived_digests: SetupPcsAuthorityDerivedDigestBusV3,
    pub claim_values: SetupPcsAuthorityClaimValueBusV3,
    pub transition_statement: SetupPcsAuthorityTransitionStatementBusV3,
    pub bound_batch: SetupPcsAuthorityBoundBatchBusV3,
    pub bound_transition: SetupPcsAuthorityBoundTransitionBusV3,
    pub stacking_schedule: SetupPcsAuthorityStackingScheduleBusV3,
    pub initial_commitments: MultiConstraintInitialCommitmentBus,
    pub multi_whir_caller_prefix: MultiConstraintCallerPrefixBus,
    pub transcript: TranscriptBus,
    pub poseidon_permute: Poseidon2PermuteBus,
    pub poseidon_compress: Poseidon2CompressBus,
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct StatementHeaderPrepColsV3<T> {
    active: T,
    transition_index: [T; 2],
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct StatementHeaderColsV3<T> {
    active: T,
    provenance: SetupPcsSourceProvenanceMessageV3<T>,
    end_tidx_bytes: [T; 8],
    end_tidx_byte_bits: [[T; 8]; 8],
    sample_count_bytes: [T; 8],
    sample_count_byte_bits: [[T; 8]; 8],
    source_nonzero_selectors: [[T; DIGEST_SIZE]; 4],
    source_nonzero_inverses: [T; 4],
    end_tidx_nonzero_inverse: T,
    sample_count_selectors: [T; 8],
}

#[derive(Clone, Debug)]
pub struct SetupPcsAuthorityStatementHeaderAirV3 {
    pub profile: Arc<SetupPcsAuthorityStatementProfileV3>,
    pub buses: SetupPcsAuthorityStatementBusesV3,
}

impl BaseAir<F> for SetupPcsAuthorityStatementHeaderAirV3 {
    fn width(&self) -> usize {
        core::mem::size_of::<StatementHeaderColsV3<u8>>() + self.profile.opening_point_len * D_EF
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = core::mem::size_of::<StatementHeaderPrepColsV3<u8>>();
        let height = self.profile.transition_count.next_power_of_two().max(2);
        let mut values = F::zero_vec(width * height);
        for transition in 0..self.profile.transition_count {
            let cols: &mut StatementHeaderPrepColsV3<F> =
                values[transition * width..(transition + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.transition_index = split_u32(transition as u32);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for SetupPcsAuthorityStatementHeaderAirV3 {}
impl PartitionedBaseAir<F> for SetupPcsAuthorityStatementHeaderAirV3 {}

fn assert_equal_arrays<AB, const N: usize>(
    builder: &mut AB,
    enabled: impl Into<AB::Expr> + Clone,
    left: [impl Into<AB::Expr>; N],
    right: [impl Into<AB::Expr>; N],
) where
    AB: AirBuilder<F = F>,
{
    for (left, right) in left.into_iter().zip(right) {
        builder
            .when(enabled.clone())
            .assert_eq(left.into(), right.into());
    }
}

fn send_variable<AB>(
    bus: SetupPcsAuthorityCanonicalFieldBusV3,
    builder: &mut AB,
    transition_index: [AB::Expr; 2],
    variable_index: u32,
    value: AB::Expr,
    enabled: AB::Expr,
) where
    AB: InteractionBuilder,
    AB::Expr: Clone,
{
    bus.send(
        builder,
        SetupPcsAuthorityCanonicalFieldMessageV3 {
            transition_index,
            variable_index: [
                AB::Expr::from_u16(variable_index as u16),
                AB::Expr::from_u16((variable_index >> 16) as u16),
            ],
            value,
        },
        enabled * AB::Expr::from_u8(3),
    );
}

impl<AB> Air<AB> for SetupPcsAuthorityStatementHeaderAirV3
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let prep = builder.preprocessed();
        let prep_row = prep.row_slice(0).expect("setup statement header prep row");
        let prep: &StatementHeaderPrepColsV3<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("setup statement header row");
        let header_width = core::mem::size_of::<StatementHeaderColsV3<u8>>();
        let local: &StatementHeaderColsV3<AB::Var> = row[..header_width].borrow();
        let point = &row[header_width..];
        builder.assert_bool(prep.active);
        builder.assert_bool(local.active);
        builder.assert_eq(local.active, prep.active);
        let enabled = AB::Expr::from(prep.active);
        for &cell in row.iter() {
            builder
                .when(AB::Expr::ONE - enabled.clone())
                .assert_zero(cell);
        }
        builder.when(enabled.clone()).assert_eq(
            local.provenance.protocol_version,
            AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
        );
        assert_equal_arrays(
            builder,
            enabled.clone(),
            local.provenance.transition_index,
            prep.transition_index,
        );
        assert_equal_arrays(
            builder,
            enabled.clone(),
            local.provenance.app_vk_digest,
            self.profile.expected_app_vk_digest.map(AB::Expr::from),
        );
        assert_equal_arrays(
            builder,
            enabled.clone(),
            local.provenance.relation_digest,
            self.profile.expected_relation_digest.map(AB::Expr::from),
        );

        for (bytes, bits) in [
            (local.end_tidx_bytes, local.end_tidx_byte_bits),
            (local.sample_count_bytes, local.sample_count_byte_bits),
        ] {
            for (byte, bits) in bytes.into_iter().zip(bits) {
                let mut recomposed = AB::Expr::ZERO;
                for (bit_index, bit) in bits.into_iter().enumerate() {
                    builder.assert_bool(bit);
                    recomposed += AB::Expr::from(bit) * AB::Expr::from_usize(1 << bit_index);
                }
                builder.when(enabled.clone()).assert_eq(byte, recomposed);
            }
        }
        builder.when(enabled.clone()).assert_eq(
            local.provenance.end_tidx[0],
            AB::Expr::from(local.end_tidx_bytes[0])
                + AB::Expr::from_u32(256) * AB::Expr::from(local.end_tidx_bytes[1]),
        );
        builder.when(enabled.clone()).assert_eq(
            local.provenance.end_tidx[1],
            AB::Expr::from(local.end_tidx_bytes[2])
                + AB::Expr::from_u32(256) * AB::Expr::from(local.end_tidx_bytes[3]),
        );
        for byte in &local.end_tidx_bytes[4..] {
            builder.when(enabled.clone()).assert_zero(*byte);
        }
        let mut sample_count = AB::Expr::ZERO;
        for (index, byte) in local.sample_count_bytes.into_iter().enumerate().take(4) {
            sample_count += AB::Expr::from(byte) * AB::Expr::from_u64(1u64 << (8 * index));
        }
        builder
            .when(enabled.clone())
            .assert_eq(local.provenance.end_sample_count, sample_count);
        for byte in &local.sample_count_bytes[4..] {
            builder.when(enabled.clone()).assert_zero(*byte);
        }

        for (digest, (selectors, inverse)) in [
            local.provenance.source_root,
            local.provenance.source_instance_digest,
            local.provenance.source_forest_root,
            local.provenance.segment_openings_digest,
        ]
        .into_iter()
        .zip(
            local
                .source_nonzero_selectors
                .into_iter()
                .zip(local.source_nonzero_inverses),
        ) {
            let mut selector_sum = AB::Expr::ZERO;
            let mut selected = AB::Expr::ZERO;
            for (value, selector) in digest.into_iter().zip(selectors) {
                builder.assert_bool(selector);
                selector_sum += selector.into();
                selected += AB::Expr::from(value) * AB::Expr::from(selector);
            }
            builder.when(enabled.clone()).assert_one(selector_sum);
            builder.when(enabled.clone()).assert_one(selected * inverse);
        }
        let tidx_bit_sum = local
            .end_tidx_byte_bits
            .into_iter()
            .flatten()
            .fold(AB::Expr::ZERO, |sum, bit| sum + bit);
        builder
            .when(enabled.clone())
            .assert_one(tidx_bit_sum * local.end_tidx_nonzero_inverse);
        let mut sample_selector_sum = AB::Expr::ZERO;
        let mut selected_sample = AB::Expr::ZERO;
        for (index, selector) in local.sample_count_selectors.into_iter().enumerate() {
            builder.assert_bool(selector);
            sample_selector_sum += selector.into();
            selected_sample += AB::Expr::from(selector) * AB::Expr::from_usize(index + 1);
        }
        builder
            .when(enabled.clone())
            .assert_one(sample_selector_sum);
        builder
            .when(enabled.clone())
            .assert_eq(local.provenance.end_sample_count, selected_sample);

        self.buses
            .source_provenance
            .receive(builder, local.provenance.clone(), enabled.clone());

        let transition = prep.transition_index.map(Into::into);
        let mut variable_index = 0u32;
        for digest in [
            local.provenance.source_root,
            local.provenance.source_instance_digest,
            local.provenance.source_forest_root,
            local.provenance.segment_openings_digest,
        ] {
            for value in digest {
                send_variable(
                    self.buses.canonical_fields,
                    builder,
                    transition.clone(),
                    variable_index,
                    value.into(),
                    enabled.clone(),
                );
                variable_index += 1;
            }
        }
        for value in local.provenance.end_state {
            send_variable(
                self.buses.canonical_fields,
                builder,
                transition.clone(),
                variable_index,
                value.into(),
                enabled.clone(),
            );
            variable_index += 1;
        }
        for value in local
            .end_tidx_bytes
            .into_iter()
            .chain(local.sample_count_bytes)
        {
            send_variable(
                self.buses.canonical_fields,
                builder,
                transition.clone(),
                variable_index,
                value.into(),
                enabled.clone(),
            );
            variable_index += 1;
        }
        debug_assert_eq!(variable_index as usize, HEADER_VARIABLE_BASE_COUNT);
        for coordinate in 0..self.profile.opening_point_len {
            let value: [AB::Var; D_EF] = point[coordinate * D_EF..(coordinate + 1) * D_EF]
                .try_into()
                .expect("fixed setup point coordinate width");
            self.buses.source_opening_point.lookup_key(
                builder,
                FixedSetupOpeningPointMessageV2 {
                    proof_index: AB::Expr::from(prep.transition_index[0])
                        + AB::Expr::from_u32(1 << 16) * AB::Expr::from(prep.transition_index[1]),
                    point_index: AB::Expr::from_usize(coordinate),
                    value: value.map(Into::into),
                },
                enabled.clone(),
            );
            for limb in value {
                send_variable(
                    self.buses.canonical_fields,
                    builder,
                    transition.clone(),
                    variable_index,
                    limb.into(),
                    enabled.clone(),
                );
                variable_index += 1;
            }
        }
        self.buses.source_receipts.send(
            builder,
            SetupPcsAuthoritySourceReceiptMessageV3 {
                transition_index: transition,
                source_receipt_digest: local.provenance.source_receipt_digest.map(Into::into),
            },
            enabled,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct StatementClaimPrepColsV3<T> {
    active: T,
    transition_index: [T; 2],
    claim_ordinal: [T; 2],
    sort_idx: T,
    part_idx: T,
    col_idx: T,
    need_rot: T,
    current_variable_indices: [[T; 2]; D_EF],
    rotated_variable_indices: [[T; 2]; D_EF],
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct StatementClaimColsV3<T> {
    active: T,
    current: [T; D_EF],
    rotated: [T; D_EF],
}

#[derive(Clone, Debug)]
pub struct SetupPcsAuthorityStatementClaimAirV3 {
    pub profile: Arc<SetupPcsAuthorityStatementProfileV3>,
    pub buses: SetupPcsAuthorityStatementBusesV3,
}

impl BaseAir<F> for SetupPcsAuthorityStatementClaimAirV3 {
    fn width(&self) -> usize {
        core::mem::size_of::<StatementClaimColsV3<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = core::mem::size_of::<StatementClaimPrepColsV3<u8>>();
        let height = self.profile.claim_plans.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(width * height);
        for (row_index, plan) in self.profile.claim_plans.iter().enumerate() {
            let cols: &mut StatementClaimPrepColsV3<F> =
                values[row_index * width..(row_index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.transition_index = split_u32(plan.transition_index);
            cols.claim_ordinal =
                split_u32((row_index % self.profile.claim_identities.len()) as u32);
            cols.sort_idx = F::from_u32(plan.identity.sort_idx);
            cols.part_idx = F::from_u32(plan.identity.part_index);
            cols.col_idx = F::from_u32(plan.identity.column_index);
            cols.need_rot = F::from_bool(plan.identity.need_rot);
            cols.current_variable_indices =
                core::array::from_fn(|limb| split_u32(plan.current_variable_index + limb as u32));
            cols.rotated_variable_indices = core::array::from_fn(|limb| {
                split_u32(plan.rotated_variable_index.unwrap_or(0) + limb as u32)
            });
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for SetupPcsAuthorityStatementClaimAirV3 {}
impl PartitionedBaseAir<F> for SetupPcsAuthorityStatementClaimAirV3 {}

impl<AB> Air<AB> for SetupPcsAuthorityStatementClaimAirV3
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let prep = builder.preprocessed();
        let prep_row = prep.row_slice(0).expect("setup statement claim prep row");
        let prep: &StatementClaimPrepColsV3<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("setup statement claim row");
        let local: &StatementClaimColsV3<AB::Var> = (*row).borrow();
        builder.assert_bool(prep.active);
        builder.assert_bool(prep.need_rot);
        builder.assert_bool(local.active);
        builder.assert_eq(local.active, prep.active);
        let enabled = AB::Expr::from(prep.active);
        for &cell in row.iter() {
            builder
                .when(AB::Expr::ONE - enabled.clone())
                .assert_zero(cell);
        }
        for value in local.rotated {
            builder
                .when(enabled.clone() * (AB::Expr::ONE - AB::Expr::from(prep.need_rot)))
                .assert_zero(value);
        }
        let proof_idx = AB::Expr::from(prep.transition_index[0])
            + AB::Expr::from_u32(1 << 16) * AB::Expr::from(prep.transition_index[1]);
        self.buses.column_claims.receive(
            builder,
            proof_idx.clone(),
            ColumnClaimsMessage {
                sort_idx: prep.sort_idx.into(),
                part_idx: prep.part_idx.into(),
                col_idx: prep.col_idx.into(),
                claim: local.current.map(Into::into),
                is_rot: AB::Expr::ZERO,
            },
            enabled.clone(),
        );
        self.buses.column_claims.receive(
            builder,
            proof_idx,
            ColumnClaimsMessage {
                sort_idx: prep.sort_idx.into(),
                part_idx: prep.part_idx.into(),
                col_idx: prep.col_idx.into(),
                claim: local.rotated.map(Into::into),
                is_rot: AB::Expr::ONE,
            },
            enabled.clone() * prep.need_rot,
        );
        for (variable_index, value) in prep.current_variable_indices.into_iter().zip(local.current)
        {
            self.buses.canonical_fields.send(
                builder,
                SetupPcsAuthorityCanonicalFieldMessageV3 {
                    transition_index: prep.transition_index.map(Into::into),
                    variable_index: variable_index.map(Into::into),
                    value: value.into(),
                },
                enabled.clone() * AB::Expr::from_u8(3),
            );
        }
        for (variable_index, value) in prep.rotated_variable_indices.into_iter().zip(local.rotated)
        {
            self.buses.canonical_fields.send(
                builder,
                SetupPcsAuthorityCanonicalFieldMessageV3 {
                    transition_index: prep.transition_index.map(Into::into),
                    variable_index: variable_index.map(Into::into),
                    value: value.into(),
                },
                enabled.clone() * prep.need_rot * AB::Expr::from_u8(3),
            );
        }
        self.buses.claim_values.send(
            builder,
            SetupPcsAuthorityClaimValueMessageV3 {
                transition_index: prep.transition_index.map(Into::into),
                claim_ordinal: prep.claim_ordinal.map(Into::into),
                current: local.current.map(Into::into),
                rotated: local.rotated.map(Into::into),
            },
            enabled,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct StatementDigestPrepSlotV3<T> {
    active: T,
    is_variable: T,
    fixed_value: T,
    variable_transition_index: [T; 2],
    variable_index: [T; 2],
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct StatementDigestPrepColsV3<T> {
    active: T,
    is_first: T,
    is_last: T,
    transition_index: [T; 2],
    slots: [StatementDigestPrepSlotV3<T>; POSEIDON_RATE],
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct StatementDigestColsV3<T> {
    input: [T; POSEIDON_WIDTH],
    output: [T; POSEIDON_WIDTH],
    source_receipt_digest: [T; DIGEST_SIZE],
}

#[derive(Clone, Debug)]
pub struct SetupPcsAuthorityTransitionDigestAirV3 {
    pub profile: Arc<SetupPcsAuthorityStatementProfileV3>,
    pub buses: SetupPcsAuthorityStatementBusesV3,
}

impl BaseAir<F> for SetupPcsAuthorityTransitionDigestAirV3 {
    fn width(&self) -> usize {
        core::mem::size_of::<StatementDigestColsV3<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = core::mem::size_of::<StatementDigestPrepColsV3<u8>>();
        let height = self.profile.digest_blocks.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(width * height);
        for (row_index, plan) in self.profile.digest_blocks.iter().enumerate() {
            let cols: &mut StatementDigestPrepColsV3<F> =
                values[row_index * width..(row_index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.is_first = F::from_bool(plan.is_first);
            cols.is_last = F::from_bool(plan.is_last);
            cols.transition_index = split_u32(plan.transition_index);
            for (slot_index, source) in plan.sources.iter().enumerate() {
                let slot = &mut cols.slots[slot_index];
                slot.active = F::from_bool(slot_index < plan.active_slots);
                match *source {
                    CanonicalFieldSourceV3::Fixed(value) => slot.fixed_value = value,
                    CanonicalFieldSourceV3::Variable {
                        transition_index,
                        variable_index,
                    } => {
                        slot.is_variable = F::ONE;
                        slot.variable_transition_index = split_u32(transition_index);
                        slot.variable_index = split_u32(variable_index);
                    }
                }
            }
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for SetupPcsAuthorityTransitionDigestAirV3 {}
impl PartitionedBaseAir<F> for SetupPcsAuthorityTransitionDigestAirV3 {}

impl<AB> Air<AB> for SetupPcsAuthorityTransitionDigestAirV3
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let prep = builder.preprocessed();
        let prep_row = prep.row_slice(0).expect("setup statement digest prep row");
        let next_prep_row = prep
            .row_slice(1)
            .expect("setup statement digest next prep row");
        let prep: &StatementDigestPrepColsV3<AB::Var> = (*prep_row).borrow();
        let next_prep: &StatementDigestPrepColsV3<AB::Var> = (*next_prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("setup statement digest row");
        let next_row = main.row_slice(1).expect("setup statement digest next row");
        let local: &StatementDigestColsV3<AB::Var> = (*row).borrow();
        let next: &StatementDigestColsV3<AB::Var> = (*next_row).borrow();
        for value in [prep.active, prep.is_first, prep.is_last] {
            builder.assert_bool(value);
        }
        let enabled = AB::Expr::from(prep.active);
        for &cell in row.iter() {
            builder
                .when(AB::Expr::ONE - enabled.clone())
                .assert_zero(cell);
        }
        for (slot_index, slot) in prep.slots.iter().enumerate() {
            builder.assert_bool(slot.active);
            builder.assert_bool(slot.is_variable);
            builder
                .when(enabled.clone() * slot.active * (AB::Expr::ONE - slot.is_variable))
                .assert_eq(local.input[slot_index], slot.fixed_value);
            self.buses.canonical_fields.receive(
                builder,
                SetupPcsAuthorityCanonicalFieldMessageV3 {
                    transition_index: slot.variable_transition_index.map(Into::into),
                    variable_index: slot.variable_index.map(Into::into),
                    value: local.input[slot_index].into(),
                },
                enabled.clone() * slot.active * slot.is_variable,
            );
            builder
                .when(enabled.clone() * prep.is_first * (AB::Expr::ONE - slot.active))
                .assert_zero(local.input[slot_index]);
        }
        for lane in POSEIDON_RATE..POSEIDON_WIDTH {
            builder
                .when(enabled.clone() * prep.is_first)
                .assert_zero(local.input[lane]);
        }
        let continues = enabled.clone() * (AB::Expr::ONE - AB::Expr::from(prep.is_last));
        builder.when(continues.clone()).assert_one(next_prep.active);
        for lane in 0..POSEIDON_RATE {
            builder
                .when(continues.clone() * (AB::Expr::ONE - next_prep.slots[lane].active))
                .assert_eq(next.input[lane], local.output[lane]);
        }
        for lane in POSEIDON_RATE..POSEIDON_WIDTH {
            builder
                .when(continues.clone())
                .assert_eq(next.input[lane], local.output[lane]);
        }
        for value in local.source_receipt_digest {
            builder
                .when(enabled.clone() * (AB::Expr::ONE - AB::Expr::from(prep.is_last)))
                .assert_zero(value);
        }
        self.buses.poseidon_permute.lookup_key(
            builder,
            Poseidon2PermuteMessage {
                input: local.input.map(Into::into),
                output: local.output.map(Into::into),
            },
            enabled.clone(),
        );
        self.buses.source_receipts.receive(
            builder,
            SetupPcsAuthoritySourceReceiptMessageV3 {
                transition_index: prep.transition_index.map(Into::into),
                source_receipt_digest: local.source_receipt_digest.map(Into::into),
            },
            enabled.clone() * prep.is_last,
        );
        self.buses.transition_statement.send(
            builder,
            SetupPcsAuthorityTransitionStatementMessageV3 {
                protocol_version: AB::Expr::from_u32(SETUP_PCS_AUTHORITY_STATEMENT_PROTOCOL_V3),
                transition_index: prep.transition_index.map(Into::into),
                source_receipt_digest: local.source_receipt_digest.map(Into::into),
                transition_statement_digest: core::array::from_fn(|index| {
                    local.output[index].into()
                }),
            },
            enabled * prep.is_last,
        );
        self.buses.derived_digests.send(
            builder,
            SetupPcsAuthorityDerivedDigestMessageV3 {
                kind: AB::Expr::from_u32(DERIVED_DIGEST_TRANSITION_V3),
                transition_index: prep.transition_index.map(Into::into),
                digest: core::array::from_fn(|index| local.output[index].into()),
            },
            AB::Expr::from(prep.active) * prep.is_last,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct AuthorityDigestPrepColsV3<T> {
    active: T,
    kind: T,
    is_first: T,
    is_last: T,
    slots: [StatementDigestPrepSlotV3<T>; POSEIDON_RATE],
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct AuthorityDigestColsV3<T> {
    input: [T; POSEIDON_WIDTH],
    output: [T; POSEIDON_WIDTH],
}

/// Exact SDK padding-free Poseidon hashes of the setup profile and complete batch statement.
#[derive(Clone, Debug)]
pub struct SetupPcsAuthorityBatchDigestAirV3 {
    pub profile: Arc<SetupPcsAuthorityStatementProfileV3>,
    pub buses: SetupPcsAuthorityStatementBusesV3,
}

impl BaseAir<F> for SetupPcsAuthorityBatchDigestAirV3 {
    fn width(&self) -> usize {
        core::mem::size_of::<AuthorityDigestColsV3<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = core::mem::size_of::<AuthorityDigestPrepColsV3<u8>>();
        let height = self
            .profile
            .authority_digest_blocks
            .len()
            .next_power_of_two()
            .max(2);
        let mut values = F::zero_vec(width * height);
        for (row_index, plan) in self.profile.authority_digest_blocks.iter().enumerate() {
            let cols: &mut AuthorityDigestPrepColsV3<F> =
                values[row_index * width..(row_index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.kind = F::from_u32(plan.kind);
            cols.is_first = F::from_bool(plan.is_first);
            cols.is_last = F::from_bool(plan.is_last);
            for (slot_index, source) in plan.sources.iter().enumerate() {
                let slot = &mut cols.slots[slot_index];
                slot.active = F::from_bool(slot_index < plan.active_slots);
                match *source {
                    CanonicalFieldSourceV3::Fixed(value) => slot.fixed_value = value,
                    CanonicalFieldSourceV3::Variable {
                        transition_index,
                        variable_index,
                    } => {
                        slot.is_variable = F::ONE;
                        slot.variable_transition_index = split_u32(transition_index);
                        slot.variable_index = split_u32(variable_index);
                    }
                }
            }
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for SetupPcsAuthorityBatchDigestAirV3 {}
impl PartitionedBaseAir<F> for SetupPcsAuthorityBatchDigestAirV3 {}

impl<AB> Air<AB> for SetupPcsAuthorityBatchDigestAirV3
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let prep = builder.preprocessed();
        let prep_row = prep.row_slice(0).expect("authority digest prep row");
        let next_prep_row = prep.row_slice(1).expect("authority digest next prep row");
        let prep: &AuthorityDigestPrepColsV3<AB::Var> = (*prep_row).borrow();
        let next_prep: &AuthorityDigestPrepColsV3<AB::Var> = (*next_prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("authority digest row");
        let next_row = main.row_slice(1).expect("authority digest next row");
        let local: &AuthorityDigestColsV3<AB::Var> = (*row).borrow();
        let next: &AuthorityDigestColsV3<AB::Var> = (*next_row).borrow();
        for value in [prep.active, prep.is_first, prep.is_last] {
            builder.assert_bool(value);
        }
        let enabled = AB::Expr::from(prep.active);
        for &cell in row.iter() {
            builder
                .when(AB::Expr::ONE - enabled.clone())
                .assert_zero(cell);
        }
        for (slot_index, slot) in prep.slots.iter().enumerate() {
            builder.assert_bool(slot.active);
            builder.assert_bool(slot.is_variable);
            builder
                .when(enabled.clone() * slot.active * (AB::Expr::ONE - slot.is_variable))
                .assert_eq(local.input[slot_index], slot.fixed_value);
            self.buses.canonical_fields.receive(
                builder,
                SetupPcsAuthorityCanonicalFieldMessageV3 {
                    transition_index: slot.variable_transition_index.map(Into::into),
                    variable_index: slot.variable_index.map(Into::into),
                    value: local.input[slot_index].into(),
                },
                enabled.clone() * slot.active * slot.is_variable,
            );
            builder
                .when(enabled.clone() * prep.is_first * (AB::Expr::ONE - slot.active))
                .assert_zero(local.input[slot_index]);
        }
        for lane in POSEIDON_RATE..POSEIDON_WIDTH {
            builder
                .when(enabled.clone() * prep.is_first)
                .assert_zero(local.input[lane]);
        }
        let continues = enabled.clone() * (AB::Expr::ONE - AB::Expr::from(prep.is_last));
        builder.when(continues.clone()).assert_one(next_prep.active);
        builder
            .when(continues.clone())
            .assert_eq(next_prep.kind, prep.kind);
        for lane in 0..POSEIDON_RATE {
            builder
                .when(continues.clone() * (AB::Expr::ONE - next_prep.slots[lane].active))
                .assert_eq(next.input[lane], local.output[lane]);
        }
        for lane in POSEIDON_RATE..POSEIDON_WIDTH {
            builder
                .when(continues.clone())
                .assert_eq(next.input[lane], local.output[lane]);
        }
        self.buses.poseidon_permute.lookup_key(
            builder,
            Poseidon2PermuteMessage {
                input: local.input.map(Into::into),
                output: local.output.map(Into::into),
            },
            enabled.clone(),
        );
        self.buses.derived_digests.send(
            builder,
            SetupPcsAuthorityDerivedDigestMessageV3 {
                kind: prep.kind.into(),
                transition_index: [AB::Expr::ZERO, AB::Expr::ZERO],
                digest: core::array::from_fn(|index| local.output[index].into()),
            },
            enabled * prep.is_last,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct BoundOutputPrepColsV3<T> {
    active: T,
    is_global_first: T,
    is_global_last: T,
    is_transition_first: T,
    is_transition_last: T,
    transition_index: [T; 2],
    transition_count: [T; 2],
    claim_ordinal: [T; 2],
    claim_count: [T; 2],
    setup_index: [T; 2],
    air_id: [T; 2],
    sort_idx: [T; 2],
    part_index: [T; 2],
    column_index: [T; 2],
    need_rot: T,
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct BoundOutputColsV3<T> {
    current: [T; D_EF],
    rotated: [T; D_EF],
    profile_digest: [T; DIGEST_SIZE],
    batch_digest: [T; DIGEST_SIZE],
    transition_statement_digest: [T; DIGEST_SIZE],
    canonical_before: [T; DIGEST_SIZE],
    canonical_identity_header: [T; DIGEST_SIZE],
    canonical_identity_body: [T; DIGEST_SIZE],
    canonical_after: [T; DIGEST_SIZE],
    opening_before: [T; DIGEST_SIZE],
    opening_identity: [T; DIGEST_SIZE],
    opening_values: [T; DIGEST_SIZE],
    terminal_statement: [T; DIGEST_SIZE],
    terminal_relation: [T; DIGEST_SIZE],
    terminal_vk: [T; DIGEST_SIZE],
}

#[derive(Clone, Debug)]
pub struct SetupPcsAuthorityBoundOutputAirV3 {
    pub profile: Arc<SetupPcsAuthorityStatementProfileV3>,
    pub buses: SetupPcsAuthorityStatementBusesV3,
}

impl BaseAir<F> for SetupPcsAuthorityBoundOutputAirV3 {
    fn width(&self) -> usize {
        core::mem::size_of::<BoundOutputColsV3<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = core::mem::size_of::<BoundOutputPrepColsV3<u8>>();
        let height = self.profile.claim_plans.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(width * height);
        let claim_count = self.profile.claim_identities.len();
        for (row_index, plan) in self.profile.claim_plans.iter().enumerate() {
            let ordinal = row_index % claim_count;
            let transition = row_index / claim_count;
            let cols: &mut BoundOutputPrepColsV3<F> =
                values[row_index * width..(row_index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.is_global_first = F::from_bool(row_index == 0);
            cols.is_global_last = F::from_bool(row_index + 1 == self.profile.claim_plans.len());
            cols.is_transition_first = F::from_bool(ordinal == 0);
            cols.is_transition_last = F::from_bool(ordinal + 1 == claim_count);
            cols.transition_index = split_u32(transition as u32);
            cols.transition_count = split_u32(self.profile.transition_count as u32);
            cols.claim_ordinal = split_u32(ordinal as u32);
            cols.claim_count = split_u32(claim_count as u32);
            cols.setup_index = split_u32(plan.identity.setup_index);
            cols.air_id = split_u32(plan.identity.air_id);
            cols.sort_idx = split_u32(plan.identity.sort_idx);
            cols.part_index = split_u32(plan.identity.part_index);
            cols.column_index = split_u32(plan.identity.column_index);
            cols.need_rot = F::from_bool(plan.identity.need_rot);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for SetupPcsAuthorityBoundOutputAirV3 {}
impl PartitionedBaseAir<F> for SetupPcsAuthorityBoundOutputAirV3 {}

fn compress_lookup<AB: InteractionBuilder>(
    bus: Poseidon2CompressBus,
    builder: &mut AB,
    left: [impl Into<AB::Expr>; DIGEST_SIZE],
    right: [impl Into<AB::Expr>; DIGEST_SIZE],
    output: [impl Into<AB::Expr>; DIGEST_SIZE],
    enabled: impl Into<AB::Expr>,
) {
    let left = left.map(Into::into);
    let right = right.map(Into::into);
    let input = left
        .into_iter()
        .chain(right)
        .collect::<Vec<_>>()
        .try_into()
        .ok()
        .expect("two digest Poseidon input");
    bus.lookup_key(
        builder,
        Poseidon2CompressMessage {
            input,
            output: output.map(Into::into),
        },
        enabled,
    );
}

impl<AB> Air<AB> for SetupPcsAuthorityBoundOutputAirV3
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let prep = builder.preprocessed();
        let prep_row = prep.row_slice(0).expect("bound output prep row");
        let prep: &BoundOutputPrepColsV3<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("bound output row");
        let next_row = main.row_slice(1).expect("bound output next row");
        let local: &BoundOutputColsV3<AB::Var> = (*row).borrow();
        let next: &BoundOutputColsV3<AB::Var> = (*next_row).borrow();
        for value in [
            prep.active,
            prep.is_global_first,
            prep.is_global_last,
            prep.is_transition_first,
            prep.is_transition_last,
            prep.need_rot,
        ] {
            builder.assert_bool(value);
        }
        let enabled = AB::Expr::from(prep.active);
        for &cell in row.iter() {
            builder
                .when(AB::Expr::ONE - enabled.clone())
                .assert_zero(cell);
        }
        for value in local.rotated {
            builder
                .when(enabled.clone() * (AB::Expr::ONE - AB::Expr::from(prep.need_rot)))
                .assert_zero(value);
        }
        self.buses.claim_values.receive(
            builder,
            SetupPcsAuthorityClaimValueMessageV3 {
                transition_index: prep.transition_index.map(Into::into),
                claim_ordinal: prep.claim_ordinal.map(Into::into),
                current: local.current.map(Into::into),
                rotated: local.rotated.map(Into::into),
            },
            enabled.clone(),
        );
        for (kind, digest) in [
            (DERIVED_DIGEST_PROFILE_V3, local.profile_digest),
            (DERIVED_DIGEST_BATCH_V3, local.batch_digest),
        ] {
            self.buses.derived_digests.receive(
                builder,
                SetupPcsAuthorityDerivedDigestMessageV3 {
                    kind: AB::Expr::from_u32(kind),
                    transition_index: [AB::Expr::ZERO, AB::Expr::ZERO],
                    digest: digest.map(Into::into),
                },
                prep.is_global_first,
            );
        }
        self.buses.derived_digests.receive(
            builder,
            SetupPcsAuthorityDerivedDigestMessageV3 {
                kind: AB::Expr::from_u32(DERIVED_DIGEST_TRANSITION_V3),
                transition_index: prep.transition_index.map(Into::into),
                digest: local.transition_statement_digest.map(Into::into),
            },
            prep.is_transition_last,
        );
        let continues = enabled.clone() * (AB::Expr::ONE - AB::Expr::from(prep.is_global_last));
        for (actual, expected) in [
            (next.profile_digest, local.profile_digest),
            (next.batch_digest, local.batch_digest),
        ] {
            assert_equal_arrays(builder, continues.clone(), actual, expected);
        }
        for (canonical, opening) in local.canonical_before.into_iter().zip(local.opening_before) {
            builder
                .when(prep.is_transition_first)
                .assert_zero(canonical);
            builder.when(prep.is_transition_first).assert_zero(opening);
        }
        let inside_transition =
            enabled.clone() * (AB::Expr::ONE - AB::Expr::from(prep.is_transition_last));
        assert_equal_arrays(
            builder,
            inside_transition.clone(),
            next.canonical_before,
            local.canonical_after,
        );
        assert_equal_arrays(
            builder,
            inside_transition,
            next.opening_before,
            local.opening_values,
        );

        let identity_header = [
            AB::Expr::from_u32(super::SETUP_PCS_AUTHORITY_CLAIM_START_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_STATEMENT_PROTOCOL_V3),
            prep.transition_index[0].into(),
            prep.transition_index[1].into(),
            prep.claim_ordinal[0].into(),
            prep.claim_ordinal[1].into(),
            prep.claim_count[0].into(),
            prep.claim_count[1].into(),
        ];
        let identity_body = [
            prep.setup_index[0].into(),
            prep.setup_index[1].into(),
            prep.air_id[0].into(),
            prep.air_id[1].into(),
            prep.sort_idx[0].into(),
            prep.sort_idx[1].into(),
            prep.part_index[0].into(),
            prep.part_index[1].into(),
        ];
        let identity_column = [
            prep.column_index[0].into(),
            prep.column_index[1].into(),
            prep.need_rot.into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        compress_lookup(
            self.buses.poseidon_compress,
            builder,
            local.canonical_before,
            identity_header,
            local.canonical_identity_header,
            enabled.clone(),
        );
        compress_lookup(
            self.buses.poseidon_compress,
            builder,
            local.canonical_identity_header,
            identity_body,
            local.canonical_identity_body,
            enabled.clone(),
        );
        compress_lookup(
            self.buses.poseidon_compress,
            builder,
            local.canonical_identity_body,
            identity_column,
            local.canonical_after,
            enabled.clone(),
        );
        compress_lookup(
            self.buses.poseidon_compress,
            builder,
            local.opening_before,
            local.canonical_after,
            local.opening_identity,
            enabled.clone(),
        );
        let value_block = core::array::from_fn(|index| {
            if index < D_EF {
                AB::Expr::from(local.current[index])
            } else {
                AB::Expr::from(local.rotated[index - D_EF])
            }
        });
        compress_lookup(
            self.buses.poseidon_compress,
            builder,
            local.opening_identity,
            value_block,
            local.opening_values,
            enabled.clone(),
        );
        let transition_last = enabled.clone() * prep.is_transition_last;
        compress_lookup(
            self.buses.poseidon_compress,
            builder,
            local.opening_values,
            local.transition_statement_digest,
            local.terminal_statement,
            transition_last.clone(),
        );
        compress_lookup(
            self.buses.poseidon_compress,
            builder,
            local.terminal_statement,
            self.profile.expected_relation_digest.map(AB::Expr::from),
            local.terminal_relation,
            transition_last.clone(),
        );
        compress_lookup(
            self.buses.poseidon_compress,
            builder,
            local.terminal_relation,
            self.profile
                .expected_aggregation_vk_digest
                .map(AB::Expr::from),
            local.terminal_vk,
            transition_last.clone(),
        );
        for digest in [
            local.transition_statement_digest,
            local.terminal_statement,
            local.terminal_relation,
            local.terminal_vk,
        ] {
            for value in digest {
                builder
                    .when(
                        enabled.clone() * (AB::Expr::ONE - AB::Expr::from(prep.is_transition_last)),
                    )
                    .assert_zero(value);
            }
        }
        self.buses.bound_batch.send(
            builder,
            SetupPcsAuthorityBoundBatchMessageV3 {
                protocol_version: AB::Expr::from_u32(SETUP_PCS_AUTHORITY_STATEMENT_PROTOCOL_V3),
                profile_digest: local.profile_digest.map(Into::into),
                batch_digest: local.batch_digest.map(Into::into),
                relation_digest: self.profile.expected_relation_digest.map(AB::Expr::from),
                aggregation_vk_digest: self
                    .profile
                    .expected_aggregation_vk_digest
                    .map(AB::Expr::from),
                transition_count: prep.transition_count.map(Into::into),
            },
            prep.is_global_first,
        );
        self.buses.bound_transition.send(
            builder,
            SetupPcsAuthorityBoundTransitionMessageV3 {
                protocol_version: AB::Expr::from_u32(SETUP_PCS_AUTHORITY_STATEMENT_PROTOCOL_V3),
                profile_digest: local.profile_digest.map(Into::into),
                batch_digest: local.batch_digest.map(Into::into),
                relation_digest: self.profile.expected_relation_digest.map(AB::Expr::from),
                aggregation_vk_digest: self
                    .profile
                    .expected_aggregation_vk_digest
                    .map(AB::Expr::from),
                transition_index: prep.transition_index.map(Into::into),
                transition_count: prep.transition_count.map(Into::into),
                claim_count: prep.claim_count.map(Into::into),
                canonical_claims_digest: local.canonical_after.map(Into::into),
                transition_statement_digest: local.transition_statement_digest.map(Into::into),
                setup_openings_pre_global_digest: local.terminal_vk.map(Into::into),
            },
            transition_last,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct StatementTranscriptPrepColsV3<T> {
    active: T,
    is_variable: T,
    fixed_value: T,
    variable_transition_index: [T; 2],
    variable_index: [T; 2],
    tidx: T,
    emits_schedule: T,
    schedule_transition_index: [T; 2],
    domain_prefix_tidx: T,
    first_claim_tidx: T,
    stacking_post_tidx: T,
    is_final: T,
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct StatementTranscriptColsV3<T> {
    value: T,
}

#[derive(Clone, Debug)]
pub struct SetupPcsAuthorityStatementTranscriptAirV3 {
    pub profile: Arc<SetupPcsAuthorityStatementProfileV3>,
    pub buses: SetupPcsAuthorityStatementBusesV3,
}

impl BaseAir<F> for SetupPcsAuthorityStatementTranscriptAirV3 {
    fn width(&self) -> usize {
        core::mem::size_of::<StatementTranscriptColsV3<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = core::mem::size_of::<StatementTranscriptPrepColsV3<u8>>();
        let height = self
            .profile
            .transcript_observations
            .len()
            .next_power_of_two()
            .max(2);
        let mut values = F::zero_vec(width * height);
        for (row_index, plan) in self.profile.transcript_observations.iter().enumerate() {
            let cols: &mut StatementTranscriptPrepColsV3<F> =
                values[row_index * width..(row_index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.tidx = F::from_usize(plan.tidx);
            match plan.source {
                CanonicalFieldSourceV3::Fixed(value) => cols.fixed_value = value,
                CanonicalFieldSourceV3::Variable {
                    transition_index,
                    variable_index,
                } => {
                    cols.is_variable = F::ONE;
                    cols.variable_transition_index = split_u32(transition_index);
                    cols.variable_index = split_u32(variable_index);
                }
            }
            if let Some((transition, schedule, is_final)) = plan.schedule {
                cols.emits_schedule = F::ONE;
                cols.schedule_transition_index = split_u32(transition);
                cols.domain_prefix_tidx = F::from_usize(schedule.domain_prefix_tidx);
                cols.first_claim_tidx = F::from_usize(schedule.first_claim_tidx);
                cols.stacking_post_tidx = F::from_usize(schedule.stacking_post_tidx);
                cols.is_final = F::from_bool(is_final);
            }
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for SetupPcsAuthorityStatementTranscriptAirV3 {}
impl PartitionedBaseAir<F> for SetupPcsAuthorityStatementTranscriptAirV3 {}

impl<AB> Air<AB> for SetupPcsAuthorityStatementTranscriptAirV3
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let prep = builder.preprocessed();
        let prep_row = prep
            .row_slice(0)
            .expect("setup statement transcript prep row");
        let prep: &StatementTranscriptPrepColsV3<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("setup statement transcript row");
        let local: &StatementTranscriptColsV3<AB::Var> = (*row).borrow();
        for value in [
            prep.active,
            prep.is_variable,
            prep.emits_schedule,
            prep.is_final,
        ] {
            builder.assert_bool(value);
        }
        let enabled = AB::Expr::from(prep.active);
        builder
            .when(AB::Expr::ONE - enabled.clone())
            .assert_zero(local.value);
        builder
            .when(enabled.clone() * (AB::Expr::ONE - AB::Expr::from(prep.is_variable)))
            .assert_eq(local.value, prep.fixed_value);
        self.buses.canonical_fields.receive(
            builder,
            SetupPcsAuthorityCanonicalFieldMessageV3 {
                transition_index: prep.variable_transition_index.map(Into::into),
                variable_index: prep.variable_index.map(Into::into),
                value: local.value.into(),
            },
            enabled.clone() * prep.is_variable,
        );
        self.buses.transcript.observe(
            builder,
            AB::Expr::from_usize(self.profile.authority_transcript_proof_idx),
            prep.tidx,
            local.value,
            enabled.clone(),
        );
        let schedule_enabled = enabled.clone() * prep.emits_schedule;
        let caller_prefix = prep.stacking_post_tidx * prep.is_final;
        self.buses.stacking_schedule.send(
            builder,
            SetupPcsAuthorityStackingScheduleMessageV3 {
                authority_transcript_proof_idx: AB::Expr::from_usize(
                    self.profile.authority_transcript_proof_idx,
                ),
                transition_index: prep.schedule_transition_index.map(Into::into),
                domain_prefix_tidx: prep.domain_prefix_tidx.into(),
                first_claim_tidx: prep.first_claim_tidx.into(),
                stacking_post_tidx: prep.stacking_post_tidx.into(),
                is_final: prep.is_final.into(),
                caller_prefix_tidx: caller_prefix.into(),
            },
            schedule_enabled.clone(),
        );
        self.buses.multi_whir_caller_prefix.send(
            builder,
            AB::Expr::from_usize(self.profile.authority_transcript_proof_idx),
            MultiConstraintCallerPrefixMessage {
                tidx: prep.stacking_post_tidx.into(),
            },
            schedule_enabled * prep.is_final,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct FixedSenderColsV3<T> {
    active: T,
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct InitialCommitmentPrepColsV3<T> {
    active: T,
    commit_idx: T,
    width: T,
    commitment: [T; DIGEST_SIZE],
}

/// Verifier-fixed owner of every setup root and stacked width consumed by terminal multi-WHIR.
#[derive(Clone, Debug)]
pub struct SetupPcsAuthorityInitialCommitmentAirV3 {
    pub profile: Arc<SetupPcsAuthorityStatementProfileV3>,
    pub buses: SetupPcsAuthorityStatementBusesV3,
}

impl BaseAir<F> for SetupPcsAuthorityInitialCommitmentAirV3 {
    fn width(&self) -> usize {
        core::mem::size_of::<FixedSenderColsV3<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = core::mem::size_of::<InitialCommitmentPrepColsV3<u8>>();
        let height = self
            .profile
            .initial_commitments
            .len()
            .next_power_of_two()
            .max(2);
        let mut values = F::zero_vec(width * height);
        for (index, commitment) in self.profile.initial_commitments.iter().enumerate() {
            let cols: &mut InitialCommitmentPrepColsV3<F> =
                values[index * width..(index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.commit_idx = F::from_usize(index);
            cols.width = F::from_usize(commitment.width);
            cols.commitment = commitment.commitment;
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for SetupPcsAuthorityInitialCommitmentAirV3 {}
impl PartitionedBaseAir<F> for SetupPcsAuthorityInitialCommitmentAirV3 {}

impl<AB> Air<AB> for SetupPcsAuthorityInitialCommitmentAirV3
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let prep = builder.preprocessed();
        let prep_row = prep.row_slice(0).expect("initial commitment prep row");
        let prep: &InitialCommitmentPrepColsV3<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("initial commitment row");
        let local: &FixedSenderColsV3<AB::Var> = (*row).borrow();
        builder.assert_bool(prep.active);
        builder.assert_bool(local.active);
        builder.assert_eq(local.active, prep.active);
        self.buses.initial_commitments.send(
            builder,
            AB::Expr::from_usize(self.profile.authority_transcript_proof_idx),
            MultiConstraintInitialCommitmentMessage {
                commit_idx: prep.commit_idx.into(),
                width: prep.width.into(),
                commitment: prep.commitment.map(Into::into),
            },
            prep.active,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct ScheduleSinkPrepColsV3<T> {
    active: T,
    schedule: SetupPcsAuthorityStackingScheduleMessageV3<T>,
}

/// Closes the internal schedule edge and pins every ordered-stacking cursor in preprocessing.
#[derive(Clone, Debug)]
pub struct SetupPcsAuthorityScheduleSinkAirV3 {
    pub profile: Arc<SetupPcsAuthorityStatementProfileV3>,
    pub buses: SetupPcsAuthorityStatementBusesV3,
}

impl BaseAir<F> for SetupPcsAuthorityScheduleSinkAirV3 {
    fn width(&self) -> usize {
        core::mem::size_of::<FixedSenderColsV3<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = core::mem::size_of::<ScheduleSinkPrepColsV3<u8>>();
        let height = self.profile.schedules.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(width * height);
        for (index, schedule) in self.profile.schedules.iter().enumerate() {
            let cols: &mut ScheduleSinkPrepColsV3<F> =
                values[index * width..(index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            let is_final = index + 1 == self.profile.schedules.len();
            cols.schedule = SetupPcsAuthorityStackingScheduleMessageV3 {
                authority_transcript_proof_idx: F::from_usize(
                    self.profile.authority_transcript_proof_idx,
                ),
                transition_index: split_u32(index as u32),
                domain_prefix_tidx: F::from_usize(schedule.domain_prefix_tidx),
                first_claim_tidx: F::from_usize(schedule.first_claim_tidx),
                stacking_post_tidx: F::from_usize(schedule.stacking_post_tidx),
                is_final: F::from_bool(is_final),
                caller_prefix_tidx: if is_final {
                    F::from_usize(schedule.stacking_post_tidx)
                } else {
                    F::ZERO
                },
            };
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for SetupPcsAuthorityScheduleSinkAirV3 {}
impl PartitionedBaseAir<F> for SetupPcsAuthorityScheduleSinkAirV3 {}

impl<AB> Air<AB> for SetupPcsAuthorityScheduleSinkAirV3
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let prep = builder.preprocessed();
        let prep_row = prep.row_slice(0).expect("schedule sink prep row");
        let prep: &ScheduleSinkPrepColsV3<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("schedule sink row");
        let local: &FixedSenderColsV3<AB::Var> = (*row).borrow();
        builder.assert_bool(prep.active);
        builder.assert_bool(local.active);
        builder.assert_eq(local.active, prep.active);
        self.buses.stacking_schedule.receive(
            builder,
            SetupPcsAuthorityStackingScheduleMessageV3 {
                authority_transcript_proof_idx: prep.schedule.authority_transcript_proof_idx.into(),
                transition_index: prep.schedule.transition_index.map(Into::into),
                domain_prefix_tidx: prep.schedule.domain_prefix_tidx.into(),
                first_claim_tidx: prep.schedule.first_claim_tidx.into(),
                stacking_post_tidx: prep.schedule.stacking_post_tidx.into(),
                is_final: prep.schedule.is_final.into(),
                caller_prefix_tidx: prep.schedule.caller_prefix_tidx.into(),
            },
            prep.active,
        );
    }
}

#[derive(Clone, Debug)]
pub struct SetupPcsAuthorityStatementClaimRecordV3 {
    pub current: EF,
    pub rotated: Option<EF>,
}

#[derive(Clone, Debug)]
pub struct SetupPcsAuthorityStatementTransitionRecordV3 {
    pub provenance: SetupPcsSourceProvenanceMessageV3<F>,
    /// Exact setup PLE point, never the raw WARP one-shot point.
    pub setup_opening_point: Vec<EF>,
    pub claims: Vec<SetupPcsAuthorityStatementClaimRecordV3>,
}

#[derive(Debug)]
pub struct SetupPcsAuthorityStatementTraceV3 {
    pub header: RowMajorMatrix<F>,
    pub claims: RowMajorMatrix<F>,
    pub transition_digests: RowMajorMatrix<F>,
    pub authority_digests: RowMajorMatrix<F>,
    pub bound_outputs: RowMajorMatrix<F>,
    pub transcript: RowMajorMatrix<F>,
    pub schedule_sink: RowMajorMatrix<F>,
    pub initial_commitments: RowMajorMatrix<F>,
    pub poseidon_permute_inputs: Vec<[F; POSEIDON_WIDTH]>,
    pub poseidon_compress_inputs: Vec<[F; POSEIDON_WIDTH]>,
    pub transition_statements: Vec<SetupPcsAuthorityTransitionStatementMessageV3<F>>,
    pub bound_batch: SetupPcsAuthorityBoundBatchMessageV3<F>,
    pub bound_transitions: Vec<SetupPcsAuthorityBoundTransitionMessageV3<F>>,
    pub stacking_schedules: Vec<SetupPcsAuthorityStackingScheduleMessageV3<F>>,
    pub transcript_observations: Vec<(usize, F)>,
}

impl SetupPcsAuthorityStatementTraceV3 {
    /// AIR context order is a public integration contract.
    pub fn cpu_air_contexts(self) -> Vec<AirProvingContext<CpuBackend<crate::SC>>> {
        vec![
            AirProvingContext::simple_no_pis(self.header),
            AirProvingContext::simple_no_pis(self.claims),
            AirProvingContext::simple_no_pis(self.transition_digests),
            AirProvingContext::simple_no_pis(self.authority_digests),
            AirProvingContext::simple_no_pis(self.bound_outputs),
            AirProvingContext::simple_no_pis(self.transcript),
            AirProvingContext::simple_no_pis(self.schedule_sink),
            AirProvingContext::simple_no_pis(self.initial_commitments),
        ]
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum SetupPcsAuthorityStatementAirV3 {
    Header = 0,
    Claims = 1,
    TransitionDigest = 2,
    AuthorityDigest = 3,
    BoundOutput = 4,
    Transcript = 5,
    ScheduleSink = 6,
    InitialCommitments = 7,
}

impl SetupPcsAuthorityStatementAirV3 {
    pub const COUNT: usize = 8;
}

fn u32_limbs_to_bytes(
    limbs: [F; 2],
    transition: usize,
) -> Result<[u8; 8], SetupPcsAuthorityStatementErrorV3> {
    let lo = limbs[0].as_canonical_u32();
    let hi = limbs[1].as_canonical_u32();
    if lo > u16::MAX as u32 || hi > u16::MAX as u32 {
        return Err(SetupPcsAuthorityStatementErrorV3::CheckpointEncoding(
            transition,
        ));
    }
    let value = lo | (hi << 16);
    Ok(u64::from(value).to_le_bytes())
}

fn byte_bits(byte: u8) -> [F; 8] {
    core::array::from_fn(|bit| F::from_bool((byte >> bit) & 1 == 1))
}

fn nonzero_witness(digest: Digest) -> ([F; DIGEST_SIZE], F) {
    let index = digest
        .iter()
        .position(|&value| value != F::ZERO)
        .expect("validated nonzero source digest");
    let mut selectors = [F::ZERO; DIGEST_SIZE];
    selectors[index] = F::ONE;
    (selectors, digest[index].inverse())
}

fn validate_record(
    profile: &SetupPcsAuthorityStatementProfileV3,
    transition: usize,
    record: &SetupPcsAuthorityStatementTransitionRecordV3,
) -> Result<(), SetupPcsAuthorityStatementErrorV3> {
    if record.provenance.protocol_version != F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3) {
        return Err(SetupPcsAuthorityStatementErrorV3::ProtocolVersion(
            transition,
        ));
    }
    if record.provenance.transition_index != split_u32(transition as u32) {
        return Err(SetupPcsAuthorityStatementErrorV3::TransitionIndex(
            transition,
        ));
    }
    if record.provenance.app_vk_digest != profile.expected_app_vk_digest
        || record.provenance.relation_digest != profile.expected_relation_digest
    {
        return Err(SetupPcsAuthorityStatementErrorV3::SourceProfile(transition));
    }
    if [
        record.provenance.source_root,
        record.provenance.source_instance_digest,
        record.provenance.source_forest_root,
        record.provenance.segment_openings_digest,
    ]
    .iter()
    .any(digest_is_zero)
    {
        return Err(SetupPcsAuthorityStatementErrorV3::SourceDigest(transition));
    }
    let tidx_bytes = u32_limbs_to_bytes(record.provenance.end_tidx, transition)?;
    if tidx_bytes.iter().all(|&byte| byte == 0) {
        return Err(SetupPcsAuthorityStatementErrorV3::CheckpointEncoding(
            transition,
        ));
    }
    let sample = record.provenance.end_sample_count.as_canonical_u32();
    if !(1..=POSEIDON_RATE as u32).contains(&sample) {
        return Err(SetupPcsAuthorityStatementErrorV3::CheckpointEncoding(
            transition,
        ));
    }
    if record.setup_opening_point.len() != profile.opening_point_len {
        return Err(SetupPcsAuthorityStatementErrorV3::OpeningPoint(transition));
    }
    if record.claims.len() != profile.claim_identities.len() {
        return Err(SetupPcsAuthorityStatementErrorV3::ClaimCount(transition));
    }
    for (claim, identity) in record.claims.iter().zip(profile.claim_identities.iter()) {
        if claim.rotated.is_some() != identity.need_rot {
            return Err(SetupPcsAuthorityStatementErrorV3::ClaimRotation(
                transition,
                identity.column_index as usize,
            ));
        }
    }
    Ok(())
}

fn record_variable_values(
    profile: &SetupPcsAuthorityStatementProfileV3,
    transition: usize,
    record: &SetupPcsAuthorityStatementTransitionRecordV3,
) -> Result<Vec<F>, SetupPcsAuthorityStatementErrorV3> {
    let mut values = Vec::with_capacity(profile.variable_counts[transition]);
    values.extend_from_slice(&record.provenance.source_root);
    values.extend_from_slice(&record.provenance.source_instance_digest);
    values.extend_from_slice(&record.provenance.source_forest_root);
    values.extend_from_slice(&record.provenance.segment_openings_digest);
    values.extend_from_slice(&record.provenance.end_state);
    values.extend(u32_limbs_to_bytes(record.provenance.end_tidx, transition)?.map(F::from_u8));
    values.extend(
        u64::from(record.provenance.end_sample_count.as_canonical_u32())
            .to_le_bytes()
            .map(F::from_u8),
    );
    for &coordinate in &record.setup_opening_point {
        values.extend_from_slice(coordinate.as_basis_coefficients_slice());
    }
    for claim in &record.claims {
        values.extend_from_slice(claim.current.as_basis_coefficients_slice());
        if let Some(rotated) = claim.rotated {
            values.extend_from_slice(rotated.as_basis_coefficients_slice());
        }
    }
    if values.len() != profile.variable_counts[transition] {
        return Err(SetupPcsAuthorityStatementErrorV3::VariableOverflow);
    }
    Ok(values)
}

fn resolve_source(
    source: CanonicalFieldSourceV3,
    variable_values: &[Vec<F>],
) -> Result<F, SetupPcsAuthorityStatementErrorV3> {
    match source {
        CanonicalFieldSourceV3::Fixed(value) => Ok(value),
        CanonicalFieldSourceV3::Variable {
            transition_index,
            variable_index,
        } => variable_values
            .get(transition_index as usize)
            .and_then(|values| values.get(variable_index as usize))
            .copied()
            .ok_or(SetupPcsAuthorityStatementErrorV3::VariableOverflow),
    }
}

fn compress_host_v3(inputs: &mut Vec<[F; POSEIDON_WIDTH]>, left: Digest, right: Digest) -> Digest {
    let input = core::array::from_fn(|index| {
        if index < DIGEST_SIZE {
            left[index]
        } else {
            right[index - DIGEST_SIZE]
        }
    });
    inputs.push(input);
    poseidon2_compress_with_capacity(left, right).0
}

fn identity_blocks_v3(
    transition: u32,
    ordinal: u32,
    claim_count: u32,
    identity: SetupPcsAuthorityStatementClaimIdentityV3,
) -> [Digest; 3] {
    let transition = split_u32(transition);
    let ordinal = split_u32(ordinal);
    let count = split_u32(claim_count);
    let setup = split_u32(identity.setup_index);
    let air = split_u32(identity.air_id);
    let sort = split_u32(identity.sort_idx);
    let part = split_u32(identity.part_index);
    let column = split_u32(identity.column_index);
    [
        [
            F::from_u32(super::SETUP_PCS_AUTHORITY_CLAIM_START_TAG_V3),
            F::from_u32(SETUP_PCS_AUTHORITY_STATEMENT_PROTOCOL_V3),
            transition[0],
            transition[1],
            ordinal[0],
            ordinal[1],
            count[0],
            count[1],
        ],
        [
            setup[0], setup[1], air[0], air[1], sort[0], sort[1], part[0], part[1],
        ],
        [
            column[0],
            column[1],
            F::from_bool(identity.need_rot),
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
        ],
    ]
}

pub fn generate_setup_pcs_authority_statement_trace_v3(
    profile: &Arc<SetupPcsAuthorityStatementProfileV3>,
    records: &[SetupPcsAuthorityStatementTransitionRecordV3],
) -> Result<SetupPcsAuthorityStatementTraceV3, SetupPcsAuthorityStatementErrorV3> {
    if records.len() != profile.transition_count {
        return Err(SetupPcsAuthorityStatementErrorV3::RecordCount);
    }
    for (transition, record) in records.iter().enumerate() {
        validate_record(profile, transition, record)?;
    }
    let variable_values = records
        .iter()
        .enumerate()
        .map(|(transition, record)| record_variable_values(profile, transition, record))
        .collect::<Result<Vec<_>, _>>()?;

    let header_width =
        core::mem::size_of::<StatementHeaderColsV3<u8>>() + profile.opening_point_len * D_EF;
    let header_height = profile.transition_count.next_power_of_two().max(2);
    let mut header_values = F::zero_vec(header_width * header_height);
    for (transition, record) in records.iter().enumerate() {
        let row = &mut header_values[transition * header_width..(transition + 1) * header_width];
        let fixed_width = core::mem::size_of::<StatementHeaderColsV3<u8>>();
        let cols: &mut StatementHeaderColsV3<F> = row[..fixed_width].borrow_mut();
        cols.active = F::ONE;
        cols.provenance = record.provenance.clone();
        let tidx_bytes = u32_limbs_to_bytes(record.provenance.end_tidx, transition)?;
        cols.end_tidx_bytes = tidx_bytes.map(F::from_u8);
        cols.end_tidx_byte_bits = tidx_bytes.map(byte_bits);
        let sample = record.provenance.end_sample_count.as_canonical_u32();
        let sample_bytes = u64::from(sample).to_le_bytes();
        cols.sample_count_bytes = sample_bytes.map(F::from_u8);
        cols.sample_count_byte_bits = sample_bytes.map(byte_bits);
        for (index, digest) in [
            record.provenance.source_root,
            record.provenance.source_instance_digest,
            record.provenance.source_forest_root,
            record.provenance.segment_openings_digest,
        ]
        .into_iter()
        .enumerate()
        {
            let (selectors, inverse) = nonzero_witness(digest);
            cols.source_nonzero_selectors[index] = selectors;
            cols.source_nonzero_inverses[index] = inverse;
        }
        let bit_count = tidx_bytes.iter().map(|byte| byte.count_ones()).sum::<u32>();
        cols.end_tidx_nonzero_inverse = F::from_u32(bit_count).inverse();
        cols.sample_count_selectors[(sample - 1) as usize] = F::ONE;
        for (coordinate, &value) in record.setup_opening_point.iter().enumerate() {
            row[fixed_width + coordinate * D_EF..fixed_width + (coordinate + 1) * D_EF]
                .copy_from_slice(value.as_basis_coefficients_slice());
        }
    }

    let claim_width = core::mem::size_of::<StatementClaimColsV3<u8>>();
    let claim_height = profile.claim_plans.len().next_power_of_two().max(2);
    let mut claim_values = F::zero_vec(claim_width * claim_height);
    for (row_index, plan) in profile.claim_plans.iter().enumerate() {
        let claim = &records[plan.transition_index as usize].claims
            [row_index % profile.claim_identities.len()];
        let cols: &mut StatementClaimColsV3<F> =
            claim_values[row_index * claim_width..(row_index + 1) * claim_width].borrow_mut();
        cols.active = F::ONE;
        cols.current
            .copy_from_slice(claim.current.as_basis_coefficients_slice());
        if let Some(rotated) = claim.rotated {
            cols.rotated
                .copy_from_slice(rotated.as_basis_coefficients_slice());
        }
    }

    let digest_width = core::mem::size_of::<StatementDigestColsV3<u8>>();
    let digest_height = profile.digest_blocks.len().next_power_of_two().max(2);
    let mut digest_values = F::zero_vec(digest_width * digest_height);
    let mut permute_inputs = Vec::with_capacity(profile.digest_blocks.len());
    let mut transition_statements = Vec::with_capacity(profile.transition_count);
    let mut row_cursor = 0usize;
    for transition in 0..profile.transition_count {
        let payload = &profile.transition_sources[transition];
        let mut preimage = Vec::new();
        preimage.extend(
            SETUP_PCS_AUTHORITY_TRANSITION_DIGEST_TAG_V3
                .to_le_bytes()
                .map(F::from_u8),
        );
        preimage.extend((payload.len() as u64).to_le_bytes().map(F::from_u8));
        for &source in payload {
            preimage.push(resolve_source(source, &variable_values)?);
        }
        let (digest, pre_states, post_states) = poseidon2_hash_slice_with_states(&preimage);
        for (block, (&input, &output)) in pre_states.iter().zip(&post_states).enumerate() {
            let cols: &mut StatementDigestColsV3<F> = digest_values
                [row_cursor * digest_width..(row_cursor + 1) * digest_width]
                .borrow_mut();
            cols.input = input;
            cols.output = output;
            if block + 1 == pre_states.len() {
                cols.source_receipt_digest = records[transition].provenance.source_receipt_digest;
            }
            permute_inputs.push(input);
            row_cursor += 1;
        }
        transition_statements.push(SetupPcsAuthorityTransitionStatementMessageV3 {
            protocol_version: F::from_u32(SETUP_PCS_AUTHORITY_STATEMENT_PROTOCOL_V3),
            transition_index: split_u32(transition as u32),
            source_receipt_digest: records[transition].provenance.source_receipt_digest,
            transition_statement_digest: digest,
        });
    }
    debug_assert_eq!(row_cursor, profile.digest_blocks.len());

    let authority_digest_width = core::mem::size_of::<AuthorityDigestColsV3<u8>>();
    let authority_digest_height = profile
        .authority_digest_blocks
        .len()
        .next_power_of_two()
        .max(2);
    let mut authority_digest_values = F::zero_vec(authority_digest_width * authority_digest_height);
    let mut authority_digest_cursor = 0usize;
    let mut derived = Vec::new();
    for kind in [DERIVED_DIGEST_PROFILE_V3, DERIVED_DIGEST_BATCH_V3] {
        let plans = profile
            .authority_digest_blocks
            .iter()
            .filter(|plan| plan.kind == kind)
            .collect::<Vec<_>>();
        let mut preimage = Vec::new();
        for plan in &plans {
            for source in plan.sources.iter().take(plan.active_slots) {
                preimage.push(resolve_source(*source, &variable_values)?);
            }
        }
        let (digest, pre_states, post_states) = poseidon2_hash_slice_with_states(&preimage);
        for (&input, &output) in pre_states.iter().zip(&post_states) {
            let cols: &mut AuthorityDigestColsV3<F> = authority_digest_values
                [authority_digest_cursor * authority_digest_width
                    ..(authority_digest_cursor + 1) * authority_digest_width]
                .borrow_mut();
            cols.input = input;
            cols.output = output;
            permute_inputs.push(input);
            authority_digest_cursor += 1;
        }
        derived.push(digest);
    }
    debug_assert_eq!(
        authority_digest_cursor,
        profile.authority_digest_blocks.len()
    );
    let profile_digest = derived[0];
    let batch_digest = derived[1];

    let bound_width = core::mem::size_of::<BoundOutputColsV3<u8>>();
    let bound_height = profile.claim_plans.len().next_power_of_two().max(2);
    let mut bound_values = F::zero_vec(bound_width * bound_height);
    let mut compress_inputs = Vec::new();
    let mut bound_transitions = Vec::with_capacity(profile.transition_count);
    let claim_count = profile.claim_identities.len();
    for transition in 0..profile.transition_count {
        let mut canonical = [F::ZERO; DIGEST_SIZE];
        let mut opening = [F::ZERO; DIGEST_SIZE];
        for (ordinal, (&identity, claim)) in profile
            .claim_identities
            .iter()
            .zip(&records[transition].claims)
            .enumerate()
        {
            let row_index = transition * claim_count + ordinal;
            let cols: &mut BoundOutputColsV3<F> =
                bound_values[row_index * bound_width..(row_index + 1) * bound_width].borrow_mut();
            cols.current
                .copy_from_slice(claim.current.as_basis_coefficients_slice());
            if let Some(rotated) = claim.rotated {
                cols.rotated
                    .copy_from_slice(rotated.as_basis_coefficients_slice());
            }
            cols.profile_digest = profile_digest;
            cols.batch_digest = batch_digest;
            cols.canonical_before = canonical;
            cols.opening_before = opening;
            let blocks = identity_blocks_v3(
                transition as u32,
                ordinal as u32,
                claim_count as u32,
                identity,
            );
            cols.canonical_identity_header =
                compress_host_v3(&mut compress_inputs, canonical, blocks[0]);
            cols.canonical_identity_body = compress_host_v3(
                &mut compress_inputs,
                cols.canonical_identity_header,
                blocks[1],
            );
            cols.canonical_after = compress_host_v3(
                &mut compress_inputs,
                cols.canonical_identity_body,
                blocks[2],
            );
            cols.opening_identity =
                compress_host_v3(&mut compress_inputs, opening, cols.canonical_after);
            let value_block = core::array::from_fn(|index| {
                if index < D_EF {
                    cols.current[index]
                } else {
                    cols.rotated[index - D_EF]
                }
            });
            cols.opening_values =
                compress_host_v3(&mut compress_inputs, cols.opening_identity, value_block);
            canonical = cols.canonical_after;
            opening = cols.opening_values;
            if ordinal + 1 == claim_count {
                let statement = transition_statements[transition].transition_statement_digest;
                cols.transition_statement_digest = statement;
                cols.terminal_statement =
                    compress_host_v3(&mut compress_inputs, opening, statement);
                cols.terminal_relation = compress_host_v3(
                    &mut compress_inputs,
                    cols.terminal_statement,
                    profile.expected_relation_digest,
                );
                cols.terminal_vk = compress_host_v3(
                    &mut compress_inputs,
                    cols.terminal_relation,
                    profile.expected_aggregation_vk_digest,
                );
                bound_transitions.push(SetupPcsAuthorityBoundTransitionMessageV3 {
                    protocol_version: F::from_u32(SETUP_PCS_AUTHORITY_STATEMENT_PROTOCOL_V3),
                    profile_digest,
                    batch_digest,
                    relation_digest: profile.expected_relation_digest,
                    aggregation_vk_digest: profile.expected_aggregation_vk_digest,
                    transition_index: split_u32(transition as u32),
                    transition_count: split_u32(profile.transition_count as u32),
                    claim_count: split_u32(claim_count as u32),
                    canonical_claims_digest: canonical,
                    transition_statement_digest: statement,
                    setup_openings_pre_global_digest: cols.terminal_vk,
                });
            }
        }
    }
    let bound_batch = SetupPcsAuthorityBoundBatchMessageV3 {
        protocol_version: F::from_u32(SETUP_PCS_AUTHORITY_STATEMENT_PROTOCOL_V3),
        profile_digest,
        batch_digest,
        relation_digest: profile.expected_relation_digest,
        aggregation_vk_digest: profile.expected_aggregation_vk_digest,
        transition_count: split_u32(profile.transition_count as u32),
    };

    let transcript_width = core::mem::size_of::<StatementTranscriptColsV3<u8>>();
    let transcript_height = profile
        .transcript_observations
        .len()
        .next_power_of_two()
        .max(2);
    let mut transcript_values = F::zero_vec(transcript_width * transcript_height);
    let mut stacking_schedules = Vec::with_capacity(profile.transition_count);
    let mut transcript_observations = Vec::with_capacity(profile.transcript_observations.len());
    for (row_index, plan) in profile.transcript_observations.iter().enumerate() {
        let cols: &mut StatementTranscriptColsV3<F> = transcript_values
            [row_index * transcript_width..(row_index + 1) * transcript_width]
            .borrow_mut();
        cols.value = resolve_source(plan.source, &variable_values)?;
        transcript_observations.push((plan.tidx, cols.value));
        if let Some((transition, schedule, is_final)) = plan.schedule {
            stacking_schedules.push(SetupPcsAuthorityStackingScheduleMessageV3 {
                authority_transcript_proof_idx: F::from_usize(
                    profile.authority_transcript_proof_idx,
                ),
                transition_index: split_u32(transition),
                domain_prefix_tidx: F::from_usize(schedule.domain_prefix_tidx),
                first_claim_tidx: F::from_usize(schedule.first_claim_tidx),
                stacking_post_tidx: F::from_usize(schedule.stacking_post_tidx),
                is_final: F::from_bool(is_final),
                caller_prefix_tidx: if is_final {
                    F::from_usize(schedule.stacking_post_tidx)
                } else {
                    F::ZERO
                },
            });
        }
    }

    let fixed_sender_width = core::mem::size_of::<FixedSenderColsV3<u8>>();
    let schedule_sink_height = profile.schedules.len().next_power_of_two().max(2);
    let mut schedule_sink_values = F::zero_vec(fixed_sender_width * schedule_sink_height);
    for row_index in 0..profile.schedules.len() {
        let cols: &mut FixedSenderColsV3<F> = schedule_sink_values
            [row_index * fixed_sender_width..(row_index + 1) * fixed_sender_width]
            .borrow_mut();
        cols.active = F::ONE;
    }
    let initial_commitment_height = profile.initial_commitments.len().next_power_of_two().max(2);
    let mut initial_commitment_values = F::zero_vec(fixed_sender_width * initial_commitment_height);
    for row_index in 0..profile.initial_commitments.len() {
        let cols: &mut FixedSenderColsV3<F> = initial_commitment_values
            [row_index * fixed_sender_width..(row_index + 1) * fixed_sender_width]
            .borrow_mut();
        cols.active = F::ONE;
    }

    Ok(SetupPcsAuthorityStatementTraceV3 {
        header: RowMajorMatrix::new(header_values, header_width),
        claims: RowMajorMatrix::new(claim_values, claim_width),
        transition_digests: RowMajorMatrix::new(digest_values, digest_width),
        authority_digests: RowMajorMatrix::new(authority_digest_values, authority_digest_width),
        bound_outputs: RowMajorMatrix::new(bound_values, bound_width),
        transcript: RowMajorMatrix::new(transcript_values, transcript_width),
        schedule_sink: RowMajorMatrix::new(schedule_sink_values, fixed_sender_width),
        initial_commitments: RowMajorMatrix::new(initial_commitment_values, fixed_sender_width),
        poseidon_permute_inputs: permute_inputs,
        poseidon_compress_inputs: compress_inputs,
        transition_statements,
        bound_batch,
        bound_transitions,
        stacking_schedules,
        transcript_observations,
    })
}

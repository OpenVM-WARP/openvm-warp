//! Recursive certification of the fixed-verifier active-child-count functional.
//!
//! This module is deliberately separate from PESAT.  It augments the ordinary
//! verifier-derived SWIRL source functional with
//!
//! ```text
//!     rho * sum(VmPvsCols::is_valid) = rho * expected_active_child_count
//! ```
//!
//! before the existing one-shot source-opening reduction.  The source root and
//! the reduction point are joined on typed buses, so this module cannot be
//! satisfied by opening a second commitment or by reducing the count at a
//! second point.
//!
//! # Integration contract
//!
//! The enclosing fixed-verifier History module must provide exactly one sender
//! for each of the three input buses:
//!
//! - [`VerifierWarpActiveCountPhaseStartBusV2`] is sent by the transcript bridge at the cursor
//!   immediately after the source root and every ordinary SWIRL/LogUp opening have been absorbed.
//! - [`VerifierWarpOrdinarySourceFunctionalBusV2`] is sent by the exact mapped source evaluator. In
//!   particular, its target and weight-at-point are not host claims.
//! - [`VerifierWarpAugmentedSourceReductionBusV2`] is sent only by the genuine one-shot reduction
//!   verifier after it has consumed [`VerifierWarpAugmentedSourceClaimBusV2`].
//!
//! The producer bridge consumes [`VerifierWarpCertifiedActiveCountBusV2`].
//! The expected schedule is fixed in [`VerifierWarpActiveCountProfileV2`], but
//! it becomes certified only through the mapped source and reduction buses.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit::{bus::TranscriptBus, define_typed_lookup_bus};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, D_EF, EF, F};

use super::{
    CertifiedFixedMultiAirSourceBusV2, CertifiedFixedMultiAirSourceMessageV2,
    FIXED_MULTI_AIR_SOURCE_CAPACITY_V4, VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2,
};
use crate::circuit::native_warp_history_v19::{
    MappedAuxiliaryChallengeBusV19, MappedAuxiliaryChallengeMessageV19, OneShotStreamCursorBusV19,
    OneShotStreamCursorMessageV19, VerifiedOneShotRawOpeningBusV19,
    VerifiedOneShotRawOpeningMessageV19, MAX_RAW_MESSAGE_POINT_LEN_V19,
};

/// Must stay byte-for-byte equal to the native fixed-verifier implementation.
pub const VERIFIER_WARP_ACTIVE_COUNT_DOMAIN_V2: &[u8] =
    b"openvm-verifier-warp-active-child-count-v1";
pub const VERIFIER_WARP_ACTIVE_COUNT_BINDING_VERSION_V2: u32 = 1;
pub const VERIFIER_WARP_PROTOCOL_VERSION_V2: u32 = 2;
pub const VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2: usize = VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2;

const LIMB_BITS: usize = 16;
const LIMB_BASE: u32 = 1 << LIMB_BITS;

/// Setup-derived metadata for one fixed capacity-four verifier relation.
///
/// `profile_digest` is an additional typed-bus identity.  It is intentionally
/// not absorbed separately: the native transcript absorbs the expanded
/// canonical profile below, and transcript parity is the protocol anchor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierWarpActiveCountProfileV2 {
    pub relation_digest: Digest,
    pub profile_digest: Digest,
    pub profile_segment_count: u64,
    pub profile_batch_count: u64,
    pub batch_arity: u64,
    pub final_batch_active_count: u64,
    pub trace_heights: Arc<[u64]>,
    pub vm_pvs_air_id: u32,
    pub is_valid_common_main_column: u32,
    pub is_valid_message_block_start: u64,
    pub log_height: u8,
    pub log_message_len: u8,
}

impl VerifierWarpActiveCountProfileV2 {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.profile_segment_count == 0
            || self.profile_batch_count == 0
            || self.batch_arity != VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 as u64
            || self.final_batch_active_count == 0
            || self.final_batch_active_count > self.batch_arity
            || self.trace_heights.is_empty()
            || usize::from(self.log_message_len) > MAX_RAW_MESSAGE_POINT_LEN_V19
            || self.log_height > self.log_message_len
        {
            return Err("invalid verifier-WARP active-count profile");
        }
        let expected_batches = self
            .profile_segment_count
            .checked_add(self.batch_arity - 1)
            .ok_or("active-count batch arithmetic")?
            / self.batch_arity;
        let prefix = self
            .profile_batch_count
            .checked_sub(1)
            .and_then(|count| count.checked_mul(self.batch_arity))
            .ok_or("active-count batch arithmetic")?;
        let expected_final = self
            .profile_segment_count
            .checked_sub(prefix)
            .ok_or("active-count batch arithmetic")?;
        if expected_batches != self.profile_batch_count
            || expected_final != self.final_batch_active_count
        {
            return Err("noncanonical verifier-WARP active-count schedule");
        }

        let message_len = 1u64
            .checked_shl(u32::from(self.log_message_len))
            .ok_or("active-count message length")?;
        let height = 1u64
            .checked_shl(u32::from(self.log_height))
            .ok_or("active-count block height")?;
        if !self.is_valid_message_block_start.is_multiple_of(height)
            || self
                .is_valid_message_block_start
                .checked_add(height)
                .is_none_or(|end| end > message_len)
        {
            return Err("misaligned verifier-WARP is_valid message block");
        }
        Ok(())
    }

    pub fn expected_active_child_count(&self, batch_index: u64) -> Result<u8, &'static str> {
        self.validate()?;
        if batch_index >= self.profile_batch_count {
            return Err("active-count batch outside profile");
        }
        let count = if batch_index + 1 == self.profile_batch_count {
            self.final_batch_active_count
        } else {
            self.batch_arity
        };
        u8::try_from(count).map_err(|_| "active-count does not fit u8")
    }

    #[must_use]
    pub fn metadata_observation_count(&self) -> usize {
        // Domain bytes, protocol and binding versions, expanded metadata,
        // expanded profile, and the relation digest.  The EF4 sample and the
        // post-sample zero flag are accounted for separately.
        VERIFIER_WARP_ACTIVE_COUNT_DOMAIN_V2.len()
            + 2
            + 4 // batch index
            + 4 // metadata profile segment count
            + 4 // metadata profile batch count
            + 1 // expected count
            + 1 // AIR id
            + 1 // common-main column
            + 4 // block start
            + 1 // log height
            + 4 // profile segment count
            + 4 // batch arity
            + 4 // profile batch count
            + 4 // final batch count
            + 4 // trace-height vector length
            + 4 * self.trace_heights.len()
            + DIGEST_SIZE
    }

    #[must_use]
    pub fn transcript_span(&self) -> usize {
        self.metadata_observation_count() + D_EF + 1
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct VerifierWarpActiveCountPhaseStartMessageV2<T> {
    pub proof_index: T,
    pub batch_index: [T; 4],
    pub relation_digest: [T; DIGEST_SIZE],
    pub profile_digest: [T; DIGEST_SIZE],
    pub source_root: [T; DIGEST_SIZE],
    pub start_tidx: T,
}

define_typed_lookup_bus!(
    VerifierWarpActiveCountPhaseStartBusV2,
    VerifierWarpActiveCountPhaseStartMessageV2
);

/// Verifier-derived ordinary mapped source output at the eventual one-shot
/// point.  A host schedule is not an authorized sender for this bus.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct VerifierWarpOrdinarySourceFunctionalMessageV2<T> {
    pub proof_index: T,
    pub batch_index: [T; 4],
    pub relation_digest: [T; DIGEST_SIZE],
    pub profile_digest: [T; DIGEST_SIZE],
    pub source_root: [T; DIGEST_SIZE],
    pub log_message_len: T,
    pub ordinary_term_count: T,
    pub point_len: T,
    pub point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub ordinary_target: [T; D_EF],
    pub ordinary_weight_at_point: [T; D_EF],
}

define_typed_lookup_bus!(
    VerifierWarpOrdinarySourceFunctionalBusV2,
    VerifierWarpOrdinarySourceFunctionalMessageV2
);

/// Exact claim augmentation consumed by the one-shot claim transcript and
/// reduction verifier.  The count term is the final mapped-column term.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct VerifierWarpAugmentedSourceClaimMessageV2<T> {
    pub proof_index: T,
    pub batch_index: [T; 4],
    pub relation_digest: [T; DIGEST_SIZE],
    pub profile_digest: [T; DIGEST_SIZE],
    pub source_root: [T; DIGEST_SIZE],
    pub phase_end_tidx: T,
    pub log_message_len: T,
    pub ordinary_term_count: T,
    pub augmented_term_count: T,
    pub expected_active_child_count: T,
    pub block_start: [T; 4],
    pub log_height: T,
    pub l_skip: T,
    pub rotation: T,
    pub barycentric_len: T,
    pub barycentric_value: [T; D_EF],
    pub folded_row_eq_len: T,
    pub folded_row_eq_value: [T; D_EF],
    pub count_term_scale: [T; D_EF],
    pub batching_coefficient: [T; D_EF],
    pub combined_target: [T; D_EF],
}

define_typed_lookup_bus!(
    VerifierWarpAugmentedSourceClaimBusV2,
    VerifierWarpAugmentedSourceClaimMessageV2
);

/// Output of the genuine one-shot verifier after consuming the augmented
/// claim.  It repeats the descriptor so a different count term cannot be
/// substituted between claim binding and finalization.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct VerifierWarpAugmentedSourceReductionMessageV2<T> {
    pub proof_index: T,
    pub batch_index: [T; 4],
    pub relation_digest: [T; DIGEST_SIZE],
    pub profile_digest: [T; DIGEST_SIZE],
    pub source_root: [T; DIGEST_SIZE],
    pub phase_end_tidx: T,
    pub reduction_end_tidx: T,
    pub log_message_len: T,
    pub augmented_term_count: T,
    pub expected_active_child_count: T,
    pub block_start: [T; 4],
    pub log_height: T,
    pub count_term_scale: [T; D_EF],
    pub combined_target: [T; D_EF],
    pub point_len: T,
    pub point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub combined_weight_at_point: [T; D_EF],
    pub message_value: [T; D_EF],
}

define_typed_lookup_bus!(
    VerifierWarpAugmentedSourceReductionBusV2,
    VerifierWarpAugmentedSourceReductionMessageV2
);

/// Certified occupancy statement consumed by the v2 producer bridge.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct VerifierWarpCertifiedActiveCountMessageV2<T> {
    pub proof_index: T,
    pub batch_index: [T; 4],
    pub relation_digest: [T; DIGEST_SIZE],
    pub profile_digest: [T; DIGEST_SIZE],
    pub source_root: [T; DIGEST_SIZE],
    pub expected_active_child_count: T,
    pub batching_coefficient: [T; D_EF],
    pub point_len: T,
    pub point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub message_value: [T; D_EF],
    pub reduction_end_tidx: T,
}

define_typed_lookup_bus!(
    VerifierWarpCertifiedActiveCountBusV2,
    VerifierWarpCertifiedActiveCountMessageV2
);

/// Main row for the production C2 projection.
///
/// The v19 fixed-source path already performs the complete active-count
/// transcript, mapped-functional evaluation, and one-shot reduction. This row
/// joins their authenticated terminal messages to the same fixed source and
/// projects only the canonical C2 certificate consumed by History-v2.
#[repr(C)]
#[derive(AlignedBorrow)]
pub struct VerifierWarpActiveCountProjectionColsV2<T> {
    pub active: T,
    pub fixed_source: CertifiedFixedMultiAirSourceMessageV2<T>,
    pub batching_coefficient: [T; D_EF],
    pub reduction_end_tidx: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
struct VerifierWarpActiveCountProjectionPrepColsV2<T> {
    active: T,
    proof_index: T,
    batch_index: [T; 4],
    segment_index_lo: T,
    segment_index_hi: T,
    expected_active_child_count: T,
}

/// Degree-one authenticated projection from the real v19 source/LogUp
/// functional to C2. It does not replay or replace the active-count relation:
/// every input is a typed message emitted by the production verifier AIRs.
#[derive(Clone, Debug)]
pub struct VerifierWarpActiveCountProjectionAirV2 {
    pub profile: VerifierWarpActiveCountProfileV2,
    /// Number of consecutive transitions verified by this AIR instance.
    /// The profile remains block-wide; `segment_start` selects this bounded
    /// interval within it.
    pub transition_count: u32,
    pub segment_start: u32,
    pub source_log_codeword_len: u8,
    pub auxiliary_challenge_bus: MappedAuxiliaryChallengeBusV19,
    pub stream_cursor_bus: OneShotStreamCursorBusV19,
    pub raw_opening_bus: VerifiedOneShotRawOpeningBusV19,
    pub fixed_source_bus: CertifiedFixedMultiAirSourceBusV2,
    pub certified_count_bus: VerifierWarpCertifiedActiveCountBusV2,
}

impl VerifierWarpActiveCountProjectionAirV2 {
    fn validate(&self) -> Result<(), &'static str> {
        self.profile.validate()?;
        if self.transition_count == 0
            || u64::from(self.segment_start)
                .checked_add(u64::from(self.transition_count))
                .is_none_or(|end| end > self.profile.profile_batch_count)
            || self.source_log_codeword_len < self.profile.log_message_len
            || self.source_log_codeword_len > 31
        {
            return Err("invalid C2 projection source codeword length");
        }
        Ok(())
    }
}

impl BaseAir<F> for VerifierWarpActiveCountProjectionAirV2 {
    fn width(&self) -> usize {
        core::mem::size_of::<VerifierWarpActiveCountProjectionColsV2<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.validate().expect("invalid C2 projection profile");
        let logical_rows = usize::try_from(self.transition_count)
            .expect("C2 projection transition count does not fit usize");
        let width = VerifierWarpActiveCountProjectionPrepColsV2::<u8>::width();
        let height = logical_rows.next_power_of_two().max(2);
        let mut values = F::zero_vec(width * height);
        for batch_index in 0..logical_rows {
            let segment_index = self
                .segment_start
                .checked_add(u32::try_from(batch_index).expect("C2 projection batch index"))
                .expect("C2 projection segment index");
            let cols: &mut VerifierWarpActiveCountProjectionPrepColsV2<F> =
                values[batch_index * width..(batch_index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.proof_index = F::from_usize(batch_index);
            let global_batch_index = u64::from(segment_index);
            cols.batch_index = u64_limbs(global_batch_index).map(F::from_u16);
            cols.segment_index_lo = F::from_u32(segment_index & 0xffff);
            cols.segment_index_hi = F::from_u32(segment_index >> 16);
            cols.expected_active_child_count = F::from_u8(
                self.profile
                    .expected_active_child_count(global_batch_index)
                    .expect("validated C2 projection batch"),
            );
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for VerifierWarpActiveCountProjectionAirV2 {}
impl PartitionedBaseAir<F> for VerifierWarpActiveCountProjectionAirV2 {}

impl<AB> Air<AB> for VerifierWarpActiveCountProjectionAirV2
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.validate().is_ok(), "invalid C2 projection profile");
        let prep_matrix = builder.preprocessed();
        let prep_row = prep_matrix.row_slice(0).expect("C2 projection prep row");
        let prep: &VerifierWarpActiveCountProjectionPrepColsV2<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let main_row = main.row_slice(0).expect("C2 projection main row");
        let local: &VerifierWarpActiveCountProjectionColsV2<AB::Var> = (*main_row).borrow();
        let enabled = AB::Expr::from(prep.active);
        builder.assert_bool(prep.active);
        builder.assert_bool(local.active);
        builder.assert_eq(local.active, prep.active);
        builder
            .when(enabled.clone())
            .assert_eq(local.fixed_source.proof_index, prep.proof_index);
        builder
            .when(enabled.clone())
            .assert_eq(local.fixed_source.segment_index_lo, prep.segment_index_lo);
        builder
            .when(enabled.clone())
            .assert_eq(local.fixed_source.segment_index_hi, prep.segment_index_hi);
        builder.when(enabled.clone()).assert_eq(
            local.fixed_source.active_child_count,
            prep.expected_active_child_count,
        );
        builder.when(enabled.clone()).assert_eq(
            local.fixed_source.point_len,
            AB::Expr::from_u8(self.profile.log_message_len),
        );
        for limb in 0..DIGEST_SIZE {
            builder.when(enabled.clone()).assert_eq(
                local.fixed_source.relation_digest[limb],
                AB::Expr::from(self.profile.relation_digest[limb]),
            );
        }

        self.fixed_source_bus
            .receive(builder, local.fixed_source.clone(), enabled.clone());
        self.auxiliary_challenge_bus.receive(
            builder,
            MappedAuxiliaryChallengeMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                shard_ordinal: AB::Expr::ZERO,
                relation_digest: self.profile.relation_digest.map(Into::into),
                expected_value: prep.expected_active_child_count.into(),
                challenge: local.batching_coefficient.map(Into::into),
            },
            enabled.clone(),
        );
        self.raw_opening_bus.receive(
            builder,
            VerifiedOneShotRawOpeningMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                shard_ordinal: AB::Expr::ZERO,
                is_program: AB::Expr::ZERO,
                source_root: local.fixed_source.source_root.map(Into::into),
                range_start: AB::Expr::ZERO,
                range_end: AB::Expr::from_u32(1u32 << self.source_log_codeword_len),
                point_len: AB::Expr::from_u8(self.profile.log_message_len),
                point: local.fixed_source.point.map(|point| point.map(Into::into)),
                value: local.fixed_source.value.map(Into::into),
                program_fingerprint: core::array::from_fn(|_| AB::Expr::ZERO),
            },
            enabled.clone(),
        );
        self.stream_cursor_bus.receive(
            builder,
            OneShotStreamCursorMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                next_shard_ordinal: AB::Expr::ONE,
                tidx: local.reduction_end_tidx.into(),
            },
            enabled.clone(),
        );
        self.certified_count_bus.add_key_with_lookups(
            builder,
            VerifierWarpCertifiedActiveCountMessageV2 {
                proof_index: prep.proof_index.into(),
                batch_index: prep.batch_index.map(Into::into),
                relation_digest: self.profile.relation_digest.map(Into::into),
                profile_digest: self.profile.profile_digest.map(Into::into),
                source_root: local.fixed_source.source_root.map(Into::into),
                expected_active_child_count: prep.expected_active_child_count.into(),
                batching_coefficient: local.batching_coefficient.map(Into::into),
                point_len: AB::Expr::from_u8(self.profile.log_message_len),
                point: local.fixed_source.point.map(|point| point.map(Into::into)),
                message_value: local.fixed_source.value.map(Into::into),
                reduction_end_tidx: local.reduction_end_tidx.into(),
            },
            enabled,
        );
    }
}

#[derive(Clone, Debug)]
pub struct VerifierWarpActiveCountProjectionRecordV2 {
    pub fixed_source: CertifiedFixedMultiAirSourceMessageV2<F>,
    pub batching_coefficient: EF,
    pub reduction_end_tidx: u32,
}

/// Trace plus the exact typed messages emitted by its constrained active rows.
///
/// A downstream bridge must consume these messages instead of independently
/// rebuilding the lookup payload. This keeps the witness row and its
/// certificate on one canonical serialization path.
#[derive(Clone, Debug)]
pub struct VerifierWarpActiveCountProjectionTraceResultV2 {
    pub matrix: RowMajorMatrix<F>,
    pub messages: Vec<VerifierWarpCertifiedActiveCountMessageV2<F>>,
}

pub fn generate_verifier_warp_active_count_projection_trace_v2(
    air: &VerifierWarpActiveCountProjectionAirV2,
    records: &[VerifierWarpActiveCountProjectionRecordV2],
) -> Result<RowMajorMatrix<F>, &'static str> {
    Ok(generate_verifier_warp_active_count_projection_trace_and_messages_v2(air, records)?.matrix)
}

pub fn generate_verifier_warp_active_count_projection_trace_and_messages_v2(
    air: &VerifierWarpActiveCountProjectionAirV2,
    records: &[VerifierWarpActiveCountProjectionRecordV2],
) -> Result<VerifierWarpActiveCountProjectionTraceResultV2, &'static str> {
    air.validate()?;
    let logical_rows =
        usize::try_from(air.transition_count).map_err(|_| "C2 projection transition count")?;
    if records.len() != logical_rows {
        return Err("C2 projection record count");
    }
    let width = core::mem::size_of::<VerifierWarpActiveCountProjectionColsV2<u8>>();
    let height = logical_rows.next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    let mut messages = Vec::with_capacity(logical_rows);
    for (index, record) in records.iter().enumerate() {
        let segment_index = air
            .segment_start
            .checked_add(u32::try_from(index).map_err(|_| "C2 projection segment index")?)
            .ok_or("C2 projection segment index")?;
        let expected_count = air
            .profile
            .expected_active_child_count(u64::from(segment_index))?;
        if record.fixed_source.proof_index != F::from_usize(index)
            || record.fixed_source.segment_index_lo != F::from_u32(segment_index & 0xffff)
            || record.fixed_source.segment_index_hi != F::from_u32(segment_index >> 16)
            || record.fixed_source.active_child_count != F::from_u8(expected_count)
            || record.fixed_source.relation_digest != air.profile.relation_digest
            || record.fixed_source.point_len != F::from_u8(air.profile.log_message_len)
            || record.batching_coefficient == EF::ZERO
            || record.reduction_end_tidx == 0
        {
            return Err("C2 projection record identity");
        }
        let cols: &mut VerifierWarpActiveCountProjectionColsV2<F> =
            values[index * width..(index + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.fixed_source = record.fixed_source.clone();
        cols.batching_coefficient
            .copy_from_slice(record.batching_coefficient.as_basis_coefficients_slice());
        cols.reduction_end_tidx = F::from_u32(record.reduction_end_tidx);
        messages.push(VerifierWarpCertifiedActiveCountMessageV2 {
            proof_index: F::from_usize(index),
            batch_index: u64_limbs(u64::from(segment_index)).map(F::from_u16),
            relation_digest: air.profile.relation_digest,
            profile_digest: air.profile.profile_digest,
            source_root: record.fixed_source.source_root,
            expected_active_child_count: F::from_u8(expected_count),
            batching_coefficient: record
                .batching_coefficient
                .as_basis_coefficients_slice()
                .try_into()
                .map_err(|_| "C2 projection extension degree")?,
            point_len: F::from_u8(air.profile.log_message_len),
            point: record.fixed_source.point,
            message_value: record.fixed_source.value,
            reduction_end_tidx: F::from_u32(record.reduction_end_tidx),
        });
    }
    Ok(VerifierWarpActiveCountProjectionTraceResultV2 {
        matrix: RowMajorMatrix::new(values, width),
        messages,
    })
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct VerifierWarpActiveCountPrepColsV2<T> {
    pub active: T,
    pub batch_index: [T; 4],
    pub expected_active_child_count: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct VerifierWarpActiveCountColsV2<T> {
    pub active: T,
    pub proof_index: T,
    pub source_root: [T; DIGEST_SIZE],
    pub phase_start_tidx: T,
    pub phase_end_tidx: T,
    pub sampled_coefficient: [T; D_EF],
    pub sampled_coefficient_inverse: [T; D_EF],
    pub sampled_coefficient_is_zero: T,
    pub batching_coefficient: [T; D_EF],
    pub ordinary_term_count: T,
    pub point_len: T,
    pub point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub ordinary_target: [T; D_EF],
    pub ordinary_weight_at_point: [T; D_EF],
    pub count_prefix_products: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19 + 1],
    pub count_weight_at_point: [T; D_EF],
    pub count_term_scale: [T; D_EF],
    pub combined_target: [T; D_EF],
    pub combined_weight_at_point: [T; D_EF],
    pub reduction_end_tidx: T,
    pub message_value: [T; D_EF],
}

#[derive(Clone, ColumnsAir)]
#[columns_via(VerifierWarpActiveCountColsV2<u8>)]
pub struct VerifierWarpActiveCountFunctionalAirV2 {
    pub profile: VerifierWarpActiveCountProfileV2,
    pub transcript_bus: TranscriptBus,
    pub phase_start_bus: VerifierWarpActiveCountPhaseStartBusV2,
    pub ordinary_source_bus: VerifierWarpOrdinarySourceFunctionalBusV2,
    pub augmented_claim_bus: VerifierWarpAugmentedSourceClaimBusV2,
    pub augmented_reduction_bus: VerifierWarpAugmentedSourceReductionBusV2,
    pub certified_count_bus: VerifierWarpCertifiedActiveCountBusV2,
}

impl BaseAir<F> for VerifierWarpActiveCountFunctionalAirV2 {
    fn width(&self) -> usize {
        VerifierWarpActiveCountColsV2::<F>::width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.profile
            .validate()
            .expect("invalid active-count profile fixed in verifier key");
        let logical_rows = usize::try_from(self.profile.profile_batch_count)
            .expect("active-count profile batch count does not fit usize");
        let width = VerifierWarpActiveCountPrepColsV2::<F>::width();
        let height = logical_rows.next_power_of_two().max(2);
        let mut values = F::zero_vec(width * height);
        for batch_index in 0..logical_rows {
            let cols: &mut VerifierWarpActiveCountPrepColsV2<F> =
                values[batch_index * width..(batch_index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.batch_index = u64_limbs(batch_index as u64).map(F::from_u16);
            cols.expected_active_child_count = F::from_u8(
                self.profile
                    .expected_active_child_count(batch_index as u64)
                    .expect("validated active-count batch"),
            );
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for VerifierWarpActiveCountFunctionalAirV2 {}
impl PartitionedBaseAir<F> for VerifierWarpActiveCountFunctionalAirV2 {}

impl<AB> Air<AB> for VerifierWarpActiveCountFunctionalAirV2
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        assert!(
            self.profile.validate().is_ok(),
            "invalid active-count profile fixed in verifier key"
        );
        let prep_matrix = builder.preprocessed();
        let prep_row = prep_matrix
            .row_slice(0)
            .expect("active-count preprocessed row");
        let prep: &VerifierWarpActiveCountPrepColsV2<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let main_row = main.row_slice(0).expect("active-count main row");
        let local: &VerifierWarpActiveCountColsV2<AB::Var> = (*main_row).borrow();

        builder.assert_bool(prep.active);
        builder.assert_bool(local.active);
        builder.assert_bool(local.sampled_coefficient_is_zero);
        builder.assert_eq(local.active, prep.active);
        let enabled = AB::Expr::from(local.active);

        let batch_index_expr = u64_expr::<AB>(prep.batch_index);
        builder
            .when(enabled.clone())
            .assert_eq(local.proof_index, batch_index_expr);
        builder.when(enabled.clone()).assert_eq(
            local.point_len,
            AB::Expr::from_usize(usize::from(self.profile.log_message_len)),
        );
        for point in &local.point[usize::from(self.profile.log_message_len)..] {
            for limb in point {
                builder.when(enabled.clone()).assert_zero(*limb);
            }
        }

        let phase_message = VerifierWarpActiveCountPhaseStartMessageV2 {
            proof_index: local.proof_index.into(),
            batch_index: prep.batch_index.map(Into::into),
            relation_digest: self.profile.relation_digest.map(Into::into),
            profile_digest: self.profile.profile_digest.map(Into::into),
            source_root: local.source_root.map(Into::into),
            start_tidx: local.phase_start_tidx.into(),
        };
        self.phase_start_bus
            .lookup_key(builder, phase_message, enabled.clone());

        self.ordinary_source_bus.lookup_key(
            builder,
            VerifierWarpOrdinarySourceFunctionalMessageV2 {
                proof_index: local.proof_index.into(),
                batch_index: prep.batch_index.map(Into::into),
                relation_digest: self.profile.relation_digest.map(Into::into),
                profile_digest: self.profile.profile_digest.map(Into::into),
                source_root: local.source_root.map(Into::into),
                log_message_len: AB::Expr::from_u8(self.profile.log_message_len),
                ordinary_term_count: local.ordinary_term_count.into(),
                point_len: local.point_len.into(),
                point: local.point.map(|point| point.map(Into::into)),
                ordinary_target: local.ordinary_target.map(Into::into),
                ordinary_weight_at_point: local.ordinary_weight_at_point.map(Into::into),
            },
            enabled.clone(),
        );

        let mut tidx = AB::Expr::from(local.phase_start_tidx);
        for &byte in VERIFIER_WARP_ACTIVE_COUNT_DOMAIN_V2 {
            self.transcript_bus.observe(
                builder,
                local.proof_index,
                tidx.clone(),
                AB::Expr::from_u8(byte),
                enabled.clone(),
            );
            tidx += AB::Expr::ONE;
        }
        for value in [
            AB::Expr::from_u32(VERIFIER_WARP_PROTOCOL_VERSION_V2),
            AB::Expr::from_u32(VERIFIER_WARP_ACTIVE_COUNT_BINDING_VERSION_V2),
        ] {
            self.transcript_bus.observe(
                builder,
                local.proof_index,
                tidx.clone(),
                value,
                enabled.clone(),
            );
            tidx += AB::Expr::ONE;
        }
        observe_limbs(
            &self.transcript_bus,
            builder,
            local.proof_index,
            &mut tidx,
            prep.batch_index,
            enabled.clone(),
        );
        for value in [
            self.profile.profile_segment_count,
            self.profile.profile_batch_count,
        ] {
            observe_const_u64(
                &self.transcript_bus,
                builder,
                local.proof_index,
                &mut tidx,
                value,
                enabled.clone(),
            );
        }
        for value in [
            AB::Expr::from(prep.expected_active_child_count),
            AB::Expr::from_u32(self.profile.vm_pvs_air_id),
            AB::Expr::from_u32(self.profile.is_valid_common_main_column),
        ] {
            self.transcript_bus.observe(
                builder,
                local.proof_index,
                tidx.clone(),
                value,
                enabled.clone(),
            );
            tidx += AB::Expr::ONE;
        }
        observe_const_u64(
            &self.transcript_bus,
            builder,
            local.proof_index,
            &mut tidx,
            self.profile.is_valid_message_block_start,
            enabled.clone(),
        );
        self.transcript_bus.observe(
            builder,
            local.proof_index,
            tidx.clone(),
            AB::Expr::from_u8(self.profile.log_height),
            enabled.clone(),
        );
        tidx += AB::Expr::ONE;
        for value in [
            self.profile.profile_segment_count,
            self.profile.batch_arity,
            self.profile.profile_batch_count,
            self.profile.final_batch_active_count,
            self.profile.trace_heights.len() as u64,
        ] {
            observe_const_u64(
                &self.transcript_bus,
                builder,
                local.proof_index,
                &mut tidx,
                value,
                enabled.clone(),
            );
        }
        for &height in self.profile.trace_heights.iter() {
            observe_const_u64(
                &self.transcript_bus,
                builder,
                local.proof_index,
                &mut tidx,
                height,
                enabled.clone(),
            );
        }
        for value in self.profile.relation_digest {
            self.transcript_bus.observe(
                builder,
                local.proof_index,
                tidx.clone(),
                AB::Expr::from(value),
                enabled.clone(),
            );
            tidx += AB::Expr::ONE;
        }
        self.transcript_bus.sample_ext(
            builder,
            local.proof_index,
            tidx.clone(),
            local.sampled_coefficient,
            enabled.clone(),
        );
        tidx += AB::Expr::from_usize(D_EF);
        self.transcript_bus.observe(
            builder,
            local.proof_index,
            tidx.clone(),
            local.sampled_coefficient_is_zero,
            enabled.clone(),
        );
        tidx += AB::Expr::ONE;
        builder
            .when(enabled.clone())
            .assert_eq(local.phase_end_tidx, tidx);

        let sampled_times_inverse = ext_multiply::<AB::Expr>(
            local.sampled_coefficient.map(Into::into),
            local.sampled_coefficient_inverse.map(Into::into),
        );
        let expected_product = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE - AB::Expr::from(local.sampled_coefficient_is_zero)
            } else {
                AB::Expr::ZERO
            }
        });
        assert_ext_eq(
            builder,
            enabled.clone(),
            sampled_times_inverse,
            expected_product,
        );
        for limb in local.sampled_coefficient {
            builder.when(enabled.clone()).assert_zero(
                AB::Expr::from(limb) * AB::Expr::from(local.sampled_coefficient_is_zero),
            );
        }
        for limb in 0..D_EF {
            let expected = AB::Expr::from(local.sampled_coefficient[limb])
                + if limb == 0 {
                    AB::Expr::from(local.sampled_coefficient_is_zero)
                } else {
                    AB::Expr::ZERO
                };
            builder
                .when(enabled.clone())
                .assert_eq(local.batching_coefficient[limb], expected);
        }

        let prefix_len = usize::from(self.profile.log_message_len - self.profile.log_height);
        assert_ext_one(builder, enabled.clone(), local.count_prefix_products[0]);
        let block_index = self.profile.is_valid_message_block_start >> self.profile.log_height;
        for coordinate in 0..prefix_len {
            let point = local.point[coordinate].map(Into::into);
            let bit_shift = prefix_len - 1 - coordinate;
            let bit = ((block_index >> bit_shift) & 1) != 0;
            let factor = if bit {
                point
            } else {
                core::array::from_fn(|limb| {
                    if limb == 0 {
                        AB::Expr::ONE - point[limb].clone()
                    } else {
                        -point[limb].clone()
                    }
                })
            };
            let next = ext_multiply::<AB::Expr>(
                local.count_prefix_products[coordinate].map(Into::into),
                factor,
            );
            assert_ext_eq(
                builder,
                enabled.clone(),
                local.count_prefix_products[coordinate + 1].map(Into::into),
                next,
            );
        }
        for coordinate in prefix_len + 1..=MAX_RAW_MESSAGE_POINT_LEN_V19 {
            assert_ext_zero(
                builder,
                enabled.clone(),
                local.count_prefix_products[coordinate],
            );
        }
        assert_ext_eq(
            builder,
            enabled.clone(),
            local.count_weight_at_point.map(Into::into),
            local.count_prefix_products[prefix_len].map(Into::into),
        );

        let height = 1u64 << self.profile.log_height;
        let expected_scale = local
            .batching_coefficient
            .map(|limb| AB::Expr::from(limb) * AB::Expr::from_u64(height));
        assert_ext_eq(
            builder,
            enabled.clone(),
            local.count_term_scale.map(Into::into),
            expected_scale,
        );
        let expected_count = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::from(prep.expected_active_child_count)
            } else {
                AB::Expr::ZERO
            }
        });
        let count_target =
            ext_multiply::<AB::Expr>(local.batching_coefficient.map(Into::into), expected_count);
        let combined_target =
            ext_add::<AB::Expr>(local.ordinary_target.map(Into::into), count_target);
        assert_ext_eq(
            builder,
            enabled.clone(),
            local.combined_target.map(Into::into),
            combined_target,
        );
        let count_weight = ext_multiply::<AB::Expr>(
            local.batching_coefficient.map(Into::into),
            local.count_weight_at_point.map(Into::into),
        );
        let combined_weight =
            ext_add::<AB::Expr>(local.ordinary_weight_at_point.map(Into::into), count_weight);
        assert_ext_eq(
            builder,
            enabled.clone(),
            local.combined_weight_at_point.map(Into::into),
            combined_weight,
        );

        let block_start = u64_limbs(self.profile.is_valid_message_block_start).map(F::from_u16);
        let half = F::TWO.inverse();
        let base_one = core::array::from_fn(|limb| if limb == 0 { F::ONE } else { F::ZERO });
        let base_half = core::array::from_fn(|limb| if limb == 0 { half } else { F::ZERO });
        let augmented_term_count = AB::Expr::from(local.ordinary_term_count) + AB::Expr::ONE;
        self.augmented_claim_bus.add_key_with_lookups(
            builder,
            VerifierWarpAugmentedSourceClaimMessageV2 {
                proof_index: local.proof_index.into(),
                batch_index: prep.batch_index.map(Into::into),
                relation_digest: self.profile.relation_digest.map(Into::into),
                profile_digest: self.profile.profile_digest.map(Into::into),
                source_root: local.source_root.map(Into::into),
                phase_end_tidx: local.phase_end_tidx.into(),
                log_message_len: AB::Expr::from_u8(self.profile.log_message_len),
                ordinary_term_count: local.ordinary_term_count.into(),
                augmented_term_count: augmented_term_count.clone(),
                expected_active_child_count: prep.expected_active_child_count.into(),
                block_start: block_start.map(Into::into),
                log_height: AB::Expr::from_u8(self.profile.log_height),
                l_skip: AB::Expr::ZERO,
                rotation: AB::Expr::ZERO,
                barycentric_len: AB::Expr::ONE,
                barycentric_value: base_one.map(Into::into),
                folded_row_eq_len: AB::Expr::from_u8(self.profile.log_height),
                folded_row_eq_value: base_half.map(Into::into),
                count_term_scale: local.count_term_scale.map(Into::into),
                batching_coefficient: local.batching_coefficient.map(Into::into),
                combined_target: local.combined_target.map(Into::into),
            },
            enabled.clone(),
        );

        self.augmented_reduction_bus.lookup_key(
            builder,
            VerifierWarpAugmentedSourceReductionMessageV2 {
                proof_index: local.proof_index.into(),
                batch_index: prep.batch_index.map(Into::into),
                relation_digest: self.profile.relation_digest.map(Into::into),
                profile_digest: self.profile.profile_digest.map(Into::into),
                source_root: local.source_root.map(Into::into),
                phase_end_tidx: local.phase_end_tidx.into(),
                reduction_end_tidx: local.reduction_end_tidx.into(),
                log_message_len: AB::Expr::from_u8(self.profile.log_message_len),
                augmented_term_count,
                expected_active_child_count: prep.expected_active_child_count.into(),
                block_start: block_start.map(Into::into),
                log_height: AB::Expr::from_u8(self.profile.log_height),
                count_term_scale: local.count_term_scale.map(Into::into),
                combined_target: local.combined_target.map(Into::into),
                point_len: local.point_len.into(),
                point: local.point.map(|point| point.map(Into::into)),
                combined_weight_at_point: local.combined_weight_at_point.map(Into::into),
                message_value: local.message_value.map(Into::into),
            },
            enabled.clone(),
        );

        self.certified_count_bus.add_key_with_lookups(
            builder,
            VerifierWarpCertifiedActiveCountMessageV2 {
                proof_index: local.proof_index.into(),
                batch_index: prep.batch_index.map(Into::into),
                relation_digest: self.profile.relation_digest.map(Into::into),
                profile_digest: self.profile.profile_digest.map(Into::into),
                source_root: local.source_root.map(Into::into),
                expected_active_child_count: prep.expected_active_child_count.into(),
                batching_coefficient: local.batching_coefficient.map(Into::into),
                point_len: local.point_len.into(),
                point: local.point.map(|point| point.map(Into::into)),
                message_value: local.message_value.map(Into::into),
                reduction_end_tidx: local.reduction_end_tidx.into(),
            },
            enabled,
        );
    }
}

/// Verifier-derived values retained for one active-count row.  Fields repeated
/// by `ordinary` and `reduction` are checked before trace construction; they
/// are never silently normalized to the setup values.
#[derive(Clone, Debug)]
pub struct VerifierWarpActiveCountFunctionalRecordV2 {
    pub proof_index: u32,
    pub batch_index: u64,
    pub relation_digest: Digest,
    pub profile_digest: Digest,
    pub phase_source_root: Digest,
    pub ordinary_source_root: Digest,
    pub reduction_source_root: Digest,
    pub phase_start_tidx: u32,
    pub challenge_tidx: u32,
    pub phase_end_tidx: u32,
    pub sampled_coefficient: EF,
    pub ordinary_term_count: u32,
    pub ordinary_target: EF,
    pub ordinary_weight_at_point: EF,
    pub ordinary_point: Vec<EF>,
    pub reduction_point: Vec<EF>,
    pub reduction_end_tidx: u32,
    pub reduction_expected_active_child_count: u8,
    pub reduction_block_start: u64,
    pub reduction_log_height: u8,
    pub reduction_count_term_scale: EF,
    pub reduction_combined_target: EF,
    pub reduction_combined_weight_at_point: EF,
    pub message_value: EF,
}

pub fn generate_verifier_warp_active_count_functional_trace_v2(
    air: &VerifierWarpActiveCountFunctionalAirV2,
    records: &[VerifierWarpActiveCountFunctionalRecordV2],
) -> Result<RowMajorMatrix<F>, &'static str> {
    air.profile.validate()?;
    let batch_count = usize::try_from(air.profile.profile_batch_count)
        .map_err(|_| "active-count batch count does not fit usize")?;
    if records.len() != batch_count {
        return Err("active-count record/profile length");
    }
    let width = VerifierWarpActiveCountColsV2::<F>::width();
    let height = batch_count.next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    for (row_index, record) in records.iter().enumerate() {
        let batch_index = row_index as u64;
        let expected_count = air.profile.expected_active_child_count(batch_index)?;
        let challenge_tidx = record
            .phase_start_tidx
            .checked_add(
                u32::try_from(air.profile.metadata_observation_count())
                    .map_err(|_| "active-count transcript span")?,
            )
            .ok_or("active-count transcript cursor overflow")?;
        let phase_end_tidx = record
            .phase_start_tidx
            .checked_add(
                u32::try_from(air.profile.transcript_span())
                    .map_err(|_| "active-count transcript span")?,
            )
            .ok_or("active-count transcript cursor overflow")?;
        if record.proof_index as u64 != batch_index
            || record.batch_index != batch_index
            || record.relation_digest != air.profile.relation_digest
            || record.profile_digest != air.profile.profile_digest
            || record.phase_source_root != record.ordinary_source_root
            || record.phase_source_root != record.reduction_source_root
            || record.challenge_tidx != challenge_tidx
            || record.phase_end_tidx != phase_end_tidx
            || record.ordinary_point != record.reduction_point
            || record.ordinary_point.len() != usize::from(air.profile.log_message_len)
            || record.reduction_expected_active_child_count != expected_count
            || record.reduction_block_start != air.profile.is_valid_message_block_start
            || record.reduction_log_height != air.profile.log_height
            || record.reduction_end_tidx <= record.phase_end_tidx
        {
            return Err("active-count verifier-derived record mismatch");
        }
        let coefficient = nonzero_coefficient(record.sampled_coefficient);
        let expected_scale = coefficient * EF::from(F::from_u64(1u64 << air.profile.log_height));
        let expected_target =
            record.ordinary_target + coefficient * EF::from(F::from_u8(expected_count));
        let count_weight = active_count_weight_at_point(&air.profile, &record.ordinary_point)?;
        let expected_weight = record.ordinary_weight_at_point + coefficient * count_weight;
        if record.reduction_count_term_scale != expected_scale
            || record.reduction_combined_target != expected_target
            || record.reduction_combined_weight_at_point != expected_weight
        {
            return Err("active-count augmented reduction mismatch");
        }

        let row = &mut values[row_index * width..(row_index + 1) * width];
        let cols: &mut VerifierWarpActiveCountColsV2<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.proof_index = F::from_u32(record.proof_index);
        cols.source_root = record.phase_source_root;
        cols.phase_start_tidx = F::from_u32(record.phase_start_tidx);
        cols.phase_end_tidx = F::from_u32(record.phase_end_tidx);
        copy_ext(&mut cols.sampled_coefficient, record.sampled_coefficient);
        let inverse = if record.sampled_coefficient == EF::ZERO {
            EF::ZERO
        } else {
            record.sampled_coefficient.inverse()
        };
        copy_ext(&mut cols.sampled_coefficient_inverse, inverse);
        cols.sampled_coefficient_is_zero = F::from_bool(record.sampled_coefficient == EF::ZERO);
        copy_ext(&mut cols.batching_coefficient, coefficient);
        cols.ordinary_term_count = F::from_u32(record.ordinary_term_count);
        cols.point_len = F::from_u8(air.profile.log_message_len);
        for (target, &point) in cols.point.iter_mut().zip(&record.ordinary_point) {
            copy_ext(target, point);
        }
        copy_ext(&mut cols.ordinary_target, record.ordinary_target);
        copy_ext(
            &mut cols.ordinary_weight_at_point,
            record.ordinary_weight_at_point,
        );
        let prefix_len = usize::from(air.profile.log_message_len - air.profile.log_height);
        let block_index = air.profile.is_valid_message_block_start >> air.profile.log_height;
        let mut product = EF::ONE;
        copy_ext(&mut cols.count_prefix_products[0], product);
        for coordinate in 0..prefix_len {
            let bit = ((block_index >> (prefix_len - 1 - coordinate)) & 1) != 0;
            let point = record.ordinary_point[coordinate];
            product *= if bit { point } else { EF::ONE - point };
            copy_ext(&mut cols.count_prefix_products[coordinate + 1], product);
        }
        copy_ext(&mut cols.count_weight_at_point, product);
        copy_ext(&mut cols.count_term_scale, expected_scale);
        copy_ext(&mut cols.combined_target, expected_target);
        copy_ext(&mut cols.combined_weight_at_point, expected_weight);
        cols.reduction_end_tidx = F::from_u32(record.reduction_end_tidx);
        copy_ext(&mut cols.message_value, record.message_value);
    }
    Ok(RowMajorMatrix::new(values, width))
}

/// Canonical native transcript events for differential trace construction.
/// `true` denotes a sampled field element; all other values are observations.
pub fn verifier_warp_active_count_transcript_events_v2(
    profile: &VerifierWarpActiveCountProfileV2,
    batch_index: u64,
    sampled_coefficient: EF,
) -> Result<Vec<(F, bool)>, &'static str> {
    let expected_count = profile.expected_active_child_count(batch_index)?;
    let mut events = Vec::with_capacity(profile.transcript_span());
    events.extend(
        VERIFIER_WARP_ACTIVE_COUNT_DOMAIN_V2
            .iter()
            .copied()
            .map(|byte| (F::from_u8(byte), false)),
    );
    events.push((F::from_u32(VERIFIER_WARP_PROTOCOL_VERSION_V2), false));
    events.push((
        F::from_u32(VERIFIER_WARP_ACTIVE_COUNT_BINDING_VERSION_V2),
        false,
    ));
    for value in [
        batch_index,
        profile.profile_segment_count,
        profile.profile_batch_count,
    ] {
        extend_u64_events(&mut events, value);
    }
    events.extend([
        (F::from_u8(expected_count), false),
        (F::from_u32(profile.vm_pvs_air_id), false),
        (F::from_u32(profile.is_valid_common_main_column), false),
    ]);
    extend_u64_events(&mut events, profile.is_valid_message_block_start);
    events.push((F::from_u8(profile.log_height), false));
    for value in [
        profile.profile_segment_count,
        profile.batch_arity,
        profile.profile_batch_count,
        profile.final_batch_active_count,
        profile.trace_heights.len() as u64,
    ] {
        extend_u64_events(&mut events, value);
    }
    for &height in profile.trace_heights.iter() {
        extend_u64_events(&mut events, height);
    }
    events.extend(
        profile
            .relation_digest
            .iter()
            .copied()
            .map(|value| (value, false)),
    );
    events.extend(
        sampled_coefficient
            .as_basis_coefficients_slice()
            .iter()
            .copied()
            .map(|value| (value, true)),
    );
    events.push((F::from_bool(sampled_coefficient == EF::ZERO), false));
    debug_assert_eq!(events.len(), profile.transcript_span());
    Ok(events)
}

fn active_count_weight_at_point(
    profile: &VerifierWarpActiveCountProfileV2,
    point: &[EF],
) -> Result<EF, &'static str> {
    profile.validate()?;
    if point.len() != usize::from(profile.log_message_len) {
        return Err("active-count point length");
    }
    let prefix_len = usize::from(profile.log_message_len - profile.log_height);
    let block_index = profile.is_valid_message_block_start >> profile.log_height;
    Ok(point
        .iter()
        .take(prefix_len)
        .enumerate()
        .map(|(coordinate, &value)| {
            if ((block_index >> (prefix_len - 1 - coordinate)) & 1) != 0 {
                value
            } else {
                EF::ONE - value
            }
        })
        .product())
}

fn nonzero_coefficient(sampled: EF) -> EF {
    if sampled == EF::ZERO {
        EF::ONE
    } else {
        sampled
    }
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

fn u64_limbs(value: u64) -> [u16; 4] {
    core::array::from_fn(|limb| ((value >> (LIMB_BITS * limb)) & 0xffff) as u16)
}

fn extend_u64_events(events: &mut Vec<(F, bool)>, value: u64) {
    events.extend(
        u64_limbs(value)
            .into_iter()
            .map(|limb| (F::from_u16(limb), false)),
    );
}

fn u64_expr<AB: AirBuilder<F = F>>(limbs: [AB::Var; 4]) -> AB::Expr {
    let mut factor = AB::Expr::ONE;
    let mut value = AB::Expr::ZERO;
    for limb in limbs {
        value += AB::Expr::from(limb) * factor.clone();
        factor *= AB::Expr::from_u32(LIMB_BASE);
    }
    value
}

fn observe_limbs<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_index: AB::Var,
    tidx: &mut AB::Expr,
    limbs: [AB::Var; 4],
    enabled: AB::Expr,
) {
    for limb in limbs {
        bus.observe(builder, proof_index, tidx.clone(), limb, enabled.clone());
        *tidx += AB::Expr::ONE;
    }
}

fn observe_const_u64<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_index: AB::Var,
    tidx: &mut AB::Expr,
    value: u64,
    enabled: AB::Expr,
) {
    for limb in u64_limbs(value) {
        bus.observe(
            builder,
            proof_index,
            tidx.clone(),
            AB::Expr::from_u16(limb),
            enabled.clone(),
        );
        *tidx += AB::Expr::ONE;
    }
}

fn ext_add<FA>(left: [FA; D_EF], right: [FA; D_EF]) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
{
    core::array::from_fn(|limb| left[limb].clone() + right[limb].clone())
}

fn ext_multiply<FA>(left: [FA; D_EF], right: [FA; D_EF]) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
    FA::PrimeSubfield: BinomiallyExtendable<D_EF>,
{
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

fn assert_ext_eq<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    enabled: AB::Expr,
    left: [AB::Expr; D_EF],
    right: [AB::Expr; D_EF],
) {
    for limb in 0..D_EF {
        builder
            .when(enabled.clone())
            .assert_eq(left[limb].clone(), right[limb].clone());
    }
}

fn assert_ext_zero<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    enabled: AB::Expr,
    value: [AB::Var; D_EF],
) {
    for limb in value {
        builder.when(enabled.clone()).assert_zero(limb);
    }
}

fn assert_ext_one<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    enabled: AB::Expr,
    value: [AB::Var; D_EF],
) {
    builder.when(enabled.clone()).assert_one(value[0].clone());
    for limb in &value[1..] {
        builder.when(enabled.clone()).assert_zero(limb.clone());
    }
}

const _: () = assert!(D_EF == 4);

/// Setup identity for the fixed-capacity HLeaf active-count reduction.
/// Runtime occupancy and the terminal count are intentionally absent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierWarpActiveCountProfileV4 {
    pub relation_digest: Digest,
    pub profile_digest: Digest,
    pub batch_arity: u64,
    pub trace_heights: Arc<[u64]>,
    pub vm_pvs_air_id: u32,
    pub is_valid_common_main_column: u32,
    pub is_valid_message_block_start: u64,
    pub log_height: u8,
    pub log_message_len: u8,
}

impl VerifierWarpActiveCountProfileV4 {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.batch_arity != VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 as u64
            || self.trace_heights.is_empty()
            || usize::from(self.log_message_len) > MAX_RAW_MESSAGE_POINT_LEN_V19
            || self.log_height > self.log_message_len
        {
            return Err("invalid fixed-capacity active-count profile");
        }
        let message_len = 1u64
            .checked_shl(u32::from(self.log_message_len))
            .ok_or("active-count message length")?;
        let height = 1u64
            .checked_shl(u32::from(self.log_height))
            .ok_or("active-count block height")?;
        if !self.is_valid_message_block_start.is_multiple_of(height)
            || self
                .is_valid_message_block_start
                .checked_add(height)
                .is_none_or(|end| end > message_len)
        {
            return Err("misaligned fixed-capacity is_valid message block");
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
struct VerifierWarpActiveCountProjectionPrepColsV4<T> {
    capacity_active: T,
    is_first: T,
    is_last: T,
    proof_index: T,
    batch_index: [T; 4],
    segment_index_lo: T,
    segment_index_hi: T,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct VerifierWarpActiveCountProjectionColsV4<T> {
    pub active: T,
    pub active_child_count_flags: [T; VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2],
    pub fixed_source: CertifiedFixedMultiAirSourceMessageV2<T>,
    pub batching_coefficient: [T; D_EF],
    pub reduction_end_tidx: T,
}

/// Fixed-capacity projection from the genuine source/opening messages to the
/// History C2 certificate.  No setup column contains occupancy or count.
#[derive(Clone, Debug)]
pub struct VerifierWarpActiveCountProjectionAirV4 {
    pub profile: VerifierWarpActiveCountProfileV4,
    pub source_log_codeword_len: u8,
    pub auxiliary_challenge_bus: MappedAuxiliaryChallengeBusV19,
    pub stream_cursor_bus: OneShotStreamCursorBusV19,
    pub raw_opening_bus: VerifiedOneShotRawOpeningBusV19,
    pub fixed_source_bus: CertifiedFixedMultiAirSourceBusV2,
    pub certified_count_bus: VerifierWarpCertifiedActiveCountBusV2,
}

impl VerifierWarpActiveCountProjectionAirV4 {
    fn validate(&self) -> Result<(), &'static str> {
        self.profile.validate()?;
        if self.source_log_codeword_len < self.profile.log_message_len
            || self.source_log_codeword_len > 31
        {
            return Err("invalid fixed-capacity C2 source codeword length");
        }
        Ok(())
    }
}

impl BaseAir<F> for VerifierWarpActiveCountProjectionAirV4 {
    fn width(&self) -> usize {
        core::mem::size_of::<VerifierWarpActiveCountProjectionColsV4<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.validate().expect("invalid fixed-capacity C2 profile");
        let width = VerifierWarpActiveCountProjectionPrepColsV4::<u8>::width();
        let mut values = F::zero_vec(width * FIXED_MULTI_AIR_SOURCE_CAPACITY_V4);
        for slot in 0..FIXED_MULTI_AIR_SOURCE_CAPACITY_V4 {
            let prep: &mut VerifierWarpActiveCountProjectionPrepColsV4<F> =
                values[slot * width..(slot + 1) * width].borrow_mut();
            prep.capacity_active = F::ONE;
            prep.is_first = F::from_bool(slot == 0);
            prep.is_last = F::from_bool(slot + 1 == FIXED_MULTI_AIR_SOURCE_CAPACITY_V4);
            prep.proof_index = F::from_usize(slot);
            prep.batch_index = u64_limbs(slot as u64).map(F::from_u16);
            prep.segment_index_lo = F::from_usize(slot);
            prep.segment_index_hi = F::ZERO;
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for VerifierWarpActiveCountProjectionAirV4 {}
impl PartitionedBaseAir<F> for VerifierWarpActiveCountProjectionAirV4 {}

impl<AB> Air<AB> for VerifierWarpActiveCountProjectionAirV4
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.validate().is_ok());
        let preprocessed = builder.preprocessed();
        let prep_row = preprocessed
            .row_slice(0)
            .expect("fixed-capacity C2 prep row");
        let prep: &VerifierWarpActiveCountProjectionPrepColsV4<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("fixed-capacity C2 row");
        let next_row = main.row_slice(1).expect("fixed-capacity C2 next row");
        let local: &VerifierWarpActiveCountProjectionColsV4<AB::Var> = (*row).borrow();
        let next: &VerifierWarpActiveCountProjectionColsV4<AB::Var> = (*next_row).borrow();
        for bit in [
            prep.capacity_active,
            prep.is_first,
            prep.is_last,
            local.active,
        ] {
            builder.assert_bool(bit);
        }
        for flag in local.active_child_count_flags {
            builder.assert_bool(flag);
        }
        builder
            .when(AB::Expr::from(prep.is_first))
            .assert_one(local.active);
        let enabled = AB::Expr::from(local.active);
        let capacity_active = AB::Expr::from(prep.capacity_active);
        builder.assert_zero(enabled.clone() * (AB::Expr::ONE - capacity_active));
        let next_active = AB::Expr::from(next.active);
        let is_last = AB::Expr::from(prep.is_last);
        let terminal = enabled.clone()
            * (is_last.clone() + (AB::Expr::ONE - is_last) * (AB::Expr::ONE - next_active));
        let mut transition = builder.when_transition();
        transition.assert_zero((AB::Expr::ONE - enabled.clone()) * AB::Expr::from(next.active));
        let flag_sum = local
            .active_child_count_flags
            .iter()
            .fold(AB::Expr::ZERO, |sum, &flag| sum + AB::Expr::from(flag));
        builder.assert_eq(flag_sum, enabled.clone());
        let encoded_count = local
            .active_child_count_flags
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |sum, (index, &flag)| {
                sum + AB::Expr::from(flag) * AB::Expr::from_usize(index + 1)
            });
        builder.assert_eq(local.fixed_source.active_child_count, encoded_count);
        builder.when(enabled.clone() - terminal).assert_eq(
            local.fixed_source.active_child_count,
            AB::Expr::from_usize(VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2),
        );
        let inactive = AB::Expr::ONE - enabled.clone();
        for value in (*row).iter().skip(1) {
            builder.when(inactive.clone()).assert_zero((*value).into());
        }

        builder
            .when(enabled.clone())
            .assert_eq(local.fixed_source.proof_index, prep.proof_index);
        builder
            .when(enabled.clone())
            .assert_eq(local.fixed_source.segment_index_lo, prep.segment_index_lo);
        builder
            .when(enabled.clone())
            .assert_eq(local.fixed_source.segment_index_hi, prep.segment_index_hi);
        builder.when(enabled.clone()).assert_eq(
            local.fixed_source.point_len,
            AB::Expr::from_u8(self.profile.log_message_len),
        );
        for limb in 0..DIGEST_SIZE {
            builder.when(enabled.clone()).assert_eq(
                local.fixed_source.relation_digest[limb],
                AB::Expr::from(self.profile.relation_digest[limb]),
            );
        }

        self.fixed_source_bus
            .receive(builder, local.fixed_source.clone(), enabled.clone());
        self.auxiliary_challenge_bus.receive(
            builder,
            MappedAuxiliaryChallengeMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                shard_ordinal: AB::Expr::ZERO,
                relation_digest: self.profile.relation_digest.map(Into::into),
                expected_value: local.fixed_source.active_child_count.into(),
                challenge: local.batching_coefficient.map(Into::into),
            },
            enabled.clone(),
        );
        self.raw_opening_bus.receive(
            builder,
            VerifiedOneShotRawOpeningMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                shard_ordinal: AB::Expr::ZERO,
                is_program: AB::Expr::ZERO,
                source_root: local.fixed_source.source_root.map(Into::into),
                range_start: AB::Expr::ZERO,
                range_end: AB::Expr::from_u32(1u32 << self.source_log_codeword_len),
                point_len: AB::Expr::from_u8(self.profile.log_message_len),
                point: local.fixed_source.point.map(|point| point.map(Into::into)),
                value: local.fixed_source.value.map(Into::into),
                program_fingerprint: core::array::from_fn(|_| AB::Expr::ZERO),
            },
            enabled.clone(),
        );
        self.stream_cursor_bus.receive(
            builder,
            OneShotStreamCursorMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                next_shard_ordinal: AB::Expr::ONE,
                tidx: local.reduction_end_tidx.into(),
            },
            enabled.clone(),
        );
        self.certified_count_bus.add_key_with_lookups(
            builder,
            VerifierWarpCertifiedActiveCountMessageV2 {
                proof_index: prep.proof_index.into(),
                batch_index: prep.batch_index.map(Into::into),
                relation_digest: self.profile.relation_digest.map(Into::into),
                profile_digest: self.profile.profile_digest.map(Into::into),
                source_root: local.fixed_source.source_root.map(Into::into),
                expected_active_child_count: local.fixed_source.active_child_count.into(),
                batching_coefficient: local.batching_coefficient.map(Into::into),
                point_len: AB::Expr::from_u8(self.profile.log_message_len),
                point: local.fixed_source.point.map(|point| point.map(Into::into)),
                message_value: local.fixed_source.value.map(Into::into),
                reduction_end_tidx: local.reduction_end_tidx.into(),
            },
            enabled,
        );
    }
}

pub fn generate_verifier_warp_active_count_projection_trace_v4(
    air: &VerifierWarpActiveCountProjectionAirV4,
    records: &[VerifierWarpActiveCountProjectionRecordV2],
) -> Result<VerifierWarpActiveCountProjectionTraceResultV2, &'static str> {
    air.validate()?;
    if records.is_empty() || records.len() > FIXED_MULTI_AIR_SOURCE_CAPACITY_V4 {
        return Err("fixed-capacity C2 record count");
    }
    let width = core::mem::size_of::<VerifierWarpActiveCountProjectionColsV4<u8>>();
    let mut values = F::zero_vec(width * FIXED_MULTI_AIR_SOURCE_CAPACITY_V4);
    let mut messages = Vec::with_capacity(records.len());
    for (slot, record) in records.iter().enumerate() {
        let count = record.fixed_source.active_child_count;
        let count_usize = (1..=VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2)
            .find(|&candidate| count == F::from_usize(candidate))
            .ok_or("fixed-capacity C2 active count")?;
        if record.fixed_source.proof_index != F::from_usize(slot)
            || record.fixed_source.segment_index_lo != F::from_usize(slot)
            || record.fixed_source.segment_index_hi != F::ZERO
            || record.fixed_source.relation_digest != air.profile.relation_digest
            || record.fixed_source.point_len != F::from_u8(air.profile.log_message_len)
            || (slot + 1 != records.len() && count_usize != VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2)
            || record.batching_coefficient == EF::ZERO
            || record.reduction_end_tidx == 0
        {
            return Err("fixed-capacity C2 record identity");
        }
        let cols: &mut VerifierWarpActiveCountProjectionColsV4<F> =
            values[slot * width..(slot + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.active_child_count_flags[count_usize - 1] = F::ONE;
        cols.fixed_source = record.fixed_source.clone();
        cols.batching_coefficient
            .copy_from_slice(record.batching_coefficient.as_basis_coefficients_slice());
        cols.reduction_end_tidx = F::from_u32(record.reduction_end_tidx);
        messages.push(VerifierWarpCertifiedActiveCountMessageV2 {
            proof_index: F::from_usize(slot),
            batch_index: u64_limbs(slot as u64).map(F::from_u16),
            relation_digest: air.profile.relation_digest,
            profile_digest: air.profile.profile_digest,
            source_root: record.fixed_source.source_root,
            expected_active_child_count: count,
            batching_coefficient: record
                .batching_coefficient
                .as_basis_coefficients_slice()
                .try_into()
                .map_err(|_| "fixed-capacity C2 extension degree")?,
            point_len: F::from_u8(air.profile.log_message_len),
            point: record.fixed_source.point,
            message_value: record.fixed_source.value,
            reduction_end_tidx: F::from_u32(record.reduction_end_tidx),
        });
    }
    Ok(VerifierWarpActiveCountProjectionTraceResultV2 {
        matrix: RowMajorMatrix::new(values, width),
        messages,
    })
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;

    use openvm_recursion_circuit_derive::AlignedBorrow;
    use openvm_stark_backend::{
        air_builders::{
            debug::{check_constraints, check_logup},
            symbolic::get_symbolic_builder,
        },
        interaction::{BusIndex, SymbolicInteraction},
        keygen::types::TraceWidth,
        native_warp::prove_direct_message_opening_reduction,
        p3_air::AirBuilder,
        p3_field::PrimeCharacteristicRing,
        warp_pesat::{
            PrismalinearMappedColumnBlock, PrismalinearMappedColumnRotation,
            PrismalinearMappedColumnTerm, PrismalinearMappedColumnWeight,
            TerminalStructuredLinearClaim, TerminalWeightSpec,
        },
        AirRef, AnyAir, FiatShamirTranscript, StarkEngine,
    };
    use openvm_stark_sdk::config::{
        baby_bear_poseidon2::{
            default_duplex_sponge_recorder, BabyBearPoseidon2Config, BabyBearPoseidon2CpuEngine,
        },
        native_warp_history_params_with_100_bits_security,
    };

    use super::*;
    use crate::circuit::{
        native_warp_history_v19::{
            CertifiedLogUpOnlyEndpointBusV19, CertifiedSwirlRawOpeningBusV19,
            VerifiedLogUpArithmeticBusV19, VerifiedSourceForestLeafBusV19,
        },
        verifier_warp_history_v2::source_functional::{
            generate_fixed_multi_air_source_boundary_trace_v4, FixedMultiAirSourceBoundaryAirV4,
            FixedMultiAirSourceBoundaryColsV4, FixedMultiAirSourceBoundaryProfileV4,
            FixedMultiAirSourceBoundaryRecordV4,
        },
    };

    fn digest(seed: u32) -> Digest {
        core::array::from_fn(|limb| F::from_u32(seed + limb as u32))
    }

    fn ef(seed: u32) -> EF {
        let coefficients: [F; D_EF] = core::array::from_fn(|limb| F::from_u32(seed + limb as u32));
        EF::from_basis_coefficients_slice(&coefficients).unwrap()
    }

    fn profile(segment_count: u64) -> VerifierWarpActiveCountProfileV2 {
        VerifierWarpActiveCountProfileV2 {
            relation_digest: digest(10),
            profile_digest: digest(30),
            profile_segment_count: segment_count,
            profile_batch_count: segment_count
                .div_ceil(VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 as u64),
            batch_arity: VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 as u64,
            final_batch_active_count: {
                let remainder = segment_count % VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 as u64;
                if remainder == 0 {
                    VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 as u64
                } else {
                    remainder
                }
            },
            trace_heights: Arc::from([8, 16, 32]),
            vm_pvs_air_id: 17,
            is_valid_common_main_column: 1,
            is_valid_message_block_start: 16,
            log_height: 2,
            log_message_len: 6,
        }
    }

    fn functional_record(
        profile: &VerifierWarpActiveCountProfileV2,
        batch_index: u64,
    ) -> VerifierWarpActiveCountFunctionalRecordV2 {
        let sampled = ef(50 + batch_index as u32);
        let coefficient = nonzero_coefficient(sampled);
        let point = (0..profile.log_message_len)
            .map(|index| ef(80 + u32::from(index)))
            .collect::<Vec<_>>();
        let ordinary_target = ef(120);
        let ordinary_weight = ef(140);
        let expected_count = profile.expected_active_child_count(batch_index).unwrap();
        let count_weight = active_count_weight_at_point(profile, &point).unwrap();
        let start = 1000 + batch_index as u32 * 1000;
        let challenge = start + profile.metadata_observation_count() as u32;
        let end = start + profile.transcript_span() as u32;
        VerifierWarpActiveCountFunctionalRecordV2 {
            proof_index: batch_index as u32,
            batch_index,
            relation_digest: profile.relation_digest,
            profile_digest: profile.profile_digest,
            phase_source_root: digest(200 + batch_index as u32),
            ordinary_source_root: digest(200 + batch_index as u32),
            reduction_source_root: digest(200 + batch_index as u32),
            phase_start_tidx: start,
            challenge_tidx: challenge,
            phase_end_tidx: end,
            sampled_coefficient: sampled,
            ordinary_term_count: 9,
            ordinary_target,
            ordinary_weight_at_point: ordinary_weight,
            ordinary_point: point.clone(),
            reduction_point: point,
            reduction_end_tidx: end + 100,
            reduction_expected_active_child_count: expected_count,
            reduction_block_start: profile.is_valid_message_block_start,
            reduction_log_height: profile.log_height,
            reduction_count_term_scale: coefficient
                * EF::from(F::from_u64(1u64 << profile.log_height)),
            reduction_combined_target: ordinary_target
                + coefficient * EF::from(F::from_u8(expected_count)),
            reduction_combined_weight_at_point: ordinary_weight + coefficient * count_weight,
            message_value: ef(170),
        }
    }

    fn dummy_air(
        profile: VerifierWarpActiveCountProfileV2,
    ) -> VerifierWarpActiveCountFunctionalAirV2 {
        VerifierWarpActiveCountFunctionalAirV2 {
            profile,
            transcript_bus: TranscriptBus::new(400),
            phase_start_bus: VerifierWarpActiveCountPhaseStartBusV2::new(401),
            ordinary_source_bus: VerifierWarpOrdinarySourceFunctionalBusV2::new(402),
            augmented_claim_bus: VerifierWarpAugmentedSourceClaimBusV2::new(403),
            augmented_reduction_bus: VerifierWarpAugmentedSourceReductionBusV2::new(404),
            certified_count_bus: VerifierWarpCertifiedActiveCountBusV2::new(405),
        }
    }

    fn assert_air_constraints(
        air: &VerifierWarpActiveCountFunctionalAirV2,
        trace: &RowMajorMatrix<F>,
    ) {
        let preprocessed = air.preprocessed_trace().unwrap();
        check_constraints::<_, BabyBearPoseidon2Config>(
            air,
            "VerifierWarpActiveCountFunctionalAirV2",
            &Some(preprocessed.as_view()),
            &[trace.as_view()],
            &[],
        );
    }

    #[test]
    fn all_child_occupancies_have_canonical_traces_and_native_claims() {
        for occupancy in 1..=VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 as u64 {
            let profile = profile(occupancy);
            let air = dummy_air(profile.clone());
            let record = functional_record(&profile, 0);
            let trace = generate_verifier_warp_active_count_functional_trace_v2(
                &air,
                core::slice::from_ref(&record),
            )
            .unwrap();
            assert_eq!(trace.height(), 2);
            assert_air_constraints(&air, &trace);

            let coefficient = nonzero_coefficient(record.sampled_coefficient);
            let height = 1usize << profile.log_height;
            let term = PrismalinearMappedColumnTerm {
                block: PrismalinearMappedColumnBlock {
                    start: profile.is_valid_message_block_start as usize,
                    log_height: profile.log_height as usize,
                },
                l_skip: 0,
                barycentric_weights: vec![EF::ONE],
                folded_row_eq_point: vec![EF::TWO.inverse(); profile.log_height as usize],
                rotation: PrismalinearMappedColumnRotation::Current,
                scale: coefficient * EF::from(F::from_usize(height)),
            };
            let claim = TerminalStructuredLinearClaim::new(
                TerminalWeightSpec::PrismalinearMappedColumns(PrismalinearMappedColumnWeight {
                    log_message_len: profile.log_message_len as usize,
                    terms: vec![term],
                }),
                coefficient * EF::from(F::from_u64(occupancy)),
            );
            let mut message = vec![EF::ZERO; 1usize << profile.log_message_len];
            let start = profile.is_valid_message_block_start as usize;
            message[start..start + occupancy as usize].fill(EF::ONE);
            assert_eq!(claim.evaluate(&message), claim.target);
            let mut challenger = default_duplex_sponge_recorder();
            FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(
                &mut challenger,
                F::from_u32(1234 + occupancy as u32),
            );
            assert!(prove_direct_message_opening_reduction(
                &claim,
                &message,
                &mut openvm_stark_backend::native_warp::NativeWarpChallenger::<
                    BabyBearPoseidon2Config,
                    _,
                >::new(challenger),
            )
            .is_ok());

            message[start + occupancy as usize - 1] = EF::ZERO;
            assert_ne!(claim.evaluate(&message), claim.target, "mutated is_valid");
        }
    }

    #[test]
    fn transcript_events_match_native_metadata_order_and_zero_mapping() {
        let profile = profile(5);
        let events =
            verifier_warp_active_count_transcript_events_v2(&profile, 1, EF::ZERO).unwrap();
        assert_eq!(events.len(), profile.transcript_span());
        assert_eq!(
            &events[..VERIFIER_WARP_ACTIVE_COUNT_DOMAIN_V2.len()],
            &VERIFIER_WARP_ACTIVE_COUNT_DOMAIN_V2
                .iter()
                .map(|&byte| (F::from_u8(byte), false))
                .collect::<Vec<_>>()
        );
        assert_eq!(
            events[profile.metadata_observation_count()..][..D_EF],
            [(F::ZERO, true); D_EF]
        );
        assert_eq!(events.last(), Some(&(F::ONE, false)));
    }

    #[test]
    fn all_authority_and_second_opening_mutations_reject_before_trace_generation() {
        let profile = profile(5);
        let air = dummy_air(profile.clone());
        let records = vec![
            functional_record(&profile, 0),
            functional_record(&profile, 1),
        ];
        assert!(generate_verifier_warp_active_count_functional_trace_v2(&air, &records).is_ok());

        let reject = |mutate: fn(&mut VerifierWarpActiveCountFunctionalRecordV2)| {
            let mut changed = records.clone();
            mutate(&mut changed[1]);
            assert!(
                generate_verifier_warp_active_count_functional_trace_v2(&air, &changed).is_err()
            );
        };
        reject(|record| record.reduction_expected_active_child_count = 4); // expected count
        reject(|record| record.reduction_block_start += 4); // block offset
        reject(|record| record.reduction_log_height += 1); // height
        reject(|record| record.reduction_count_term_scale += EF::ONE); // scale
        reject(|record| record.batch_index = 0); // batch index
        reject(|record| record.challenge_tidx += 1); // challenge order
        reject(|record| record.ordinary_source_root[0] += F::ONE); // source root
        reject(|record| record.reduction_source_root[0] += F::ONE); // second root
        reject(|record| record.reduction_point[0] += EF::ONE); // second point
        reject(|record| record.reduction_combined_target += EF::ONE); // count/is_valid target
        reject(|record| record.reduction_combined_weight_at_point += EF::ONE);
        reject(|record| record.relation_digest[0] += F::ONE);
        reject(|record| record.profile_digest[0] += F::ONE);
    }

    #[test]
    fn arithmetic_trace_mutation_breaks_air_constraints() {
        let profile = profile(1);
        let air = dummy_air(profile.clone());
        let mut trace = generate_verifier_warp_active_count_functional_trace_v2(
            &air,
            &[functional_record(&profile, 0)],
        )
        .unwrap();
        assert_air_constraints(&air, &trace);
        let width = VerifierWarpActiveCountColsV2::<F>::width();
        let row: &mut VerifierWarpActiveCountColsV2<F> = trace.values[..width].borrow_mut();
        row.count_term_scale[0] += F::ONE;
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
            assert_air_constraints(&air, &trace)
        }))
        .is_err());
    }

    #[test]
    fn projection_uses_local_proof_slots_and_absolute_batch_indices() {
        let profile = profile(40);
        let air = VerifierWarpActiveCountProjectionAirV2 {
            profile: profile.clone(),
            transition_count: 2,
            segment_start: 7,
            source_log_codeword_len: profile.log_message_len,
            auxiliary_challenge_bus: MappedAuxiliaryChallengeBusV19::new(410),
            stream_cursor_bus: OneShotStreamCursorBusV19::new(411),
            raw_opening_bus: VerifiedOneShotRawOpeningBusV19::new(412),
            fixed_source_bus: CertifiedFixedMultiAirSourceBusV2::new(413),
            certified_count_bus: VerifierWarpCertifiedActiveCountBusV2::new(414),
        };
        let records = (0..2usize)
            .map(|local_index| {
                let global_index = 7 + local_index as u32;
                VerifierWarpActiveCountProjectionRecordV2 {
                    fixed_source: CertifiedFixedMultiAirSourceMessageV2 {
                        proof_index: F::from_usize(local_index),
                        segment_index_lo: F::from_u32(global_index & 0xffff),
                        segment_index_hi: F::from_u32(global_index >> 16),
                        active_child_count: F::from_u8(
                            profile
                                .expected_active_child_count(u64::from(global_index))
                                .unwrap(),
                        ),
                        app_vk_digest: digest(300),
                        relation_digest: profile.relation_digest,
                        source_forest_root: digest(320),
                        segment_openings_digest: digest(340),
                        source_root: digest(360 + global_index),
                        point_len: F::from_u8(profile.log_message_len),
                        point: [[F::ZERO; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
                        value: [F::ZERO; D_EF],
                        verifier_endpoint: [F::ZERO; D_EF],
                    },
                    batching_coefficient: ef(380 + global_index),
                    reduction_end_tidx: 400 + global_index,
                }
            })
            .collect::<Vec<_>>();
        let result =
            generate_verifier_warp_active_count_projection_trace_and_messages_v2(&air, &records)
                .unwrap();
        for (local_index, message) in result.messages.iter().enumerate() {
            let global_index = 7 + local_index as u64;
            assert_eq!(message.proof_index, F::from_usize(local_index));
            assert_eq!(
                message.batch_index,
                u64_limbs(global_index).map(F::from_u16)
            );
        }

        let mut wrong = records;
        wrong[1].fixed_source.proof_index = F::from_u32(8);
        assert!(
            generate_verifier_warp_active_count_projection_trace_and_messages_v2(&air, &wrong,)
                .is_err()
        );
    }

    fn profile_v4() -> VerifierWarpActiveCountProfileV4 {
        VerifierWarpActiveCountProfileV4 {
            relation_digest: digest(510),
            profile_digest: digest(520),
            batch_arity: VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 as u64,
            trace_heights: Arc::from([8, 16, 32]),
            vm_pvs_air_id: 17,
            is_valid_common_main_column: 1,
            is_valid_message_block_start: 16,
            log_height: 2,
            log_message_len: 6,
        }
    }

    fn projection_air_v4() -> VerifierWarpActiveCountProjectionAirV4 {
        VerifierWarpActiveCountProjectionAirV4 {
            profile: profile_v4(),
            source_log_codeword_len: 8,
            auxiliary_challenge_bus: MappedAuxiliaryChallengeBusV19::new(510),
            stream_cursor_bus: OneShotStreamCursorBusV19::new(511),
            raw_opening_bus: VerifiedOneShotRawOpeningBusV19::new(512),
            fixed_source_bus: CertifiedFixedMultiAirSourceBusV2::new(513),
            certified_count_bus: VerifierWarpCertifiedActiveCountBusV2::new(514),
        }
    }

    fn projection_records_v4(
        occupancy: usize,
        terminal_active_child_count: u8,
    ) -> Vec<VerifierWarpActiveCountProjectionRecordV2> {
        let profile = profile_v4();
        (0..occupancy)
            .map(|slot| {
                let count = if slot + 1 == occupancy {
                    terminal_active_child_count
                } else {
                    VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 as u8
                };
                VerifierWarpActiveCountProjectionRecordV2 {
                    fixed_source: CertifiedFixedMultiAirSourceMessageV2 {
                        proof_index: F::from_usize(slot),
                        segment_index_lo: F::from_usize(slot),
                        segment_index_hi: F::ZERO,
                        active_child_count: F::from_u8(count),
                        app_vk_digest: digest(530),
                        relation_digest: profile.relation_digest,
                        source_forest_root: digest(540 + slot as u32),
                        segment_openings_digest: digest(550 + slot as u32),
                        source_root: digest(560 + slot as u32),
                        point_len: F::from_u8(profile.log_message_len),
                        point: [[F::ZERO; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
                        value: [F::ZERO; D_EF],
                        verifier_endpoint: [F::ZERO; D_EF],
                    },
                    batching_coefficient: ef(570 + slot as u32),
                    reduction_end_tidx: 600 + slot as u32,
                }
            })
            .collect()
    }

    fn check_projection_v4(
        air: &VerifierWarpActiveCountProjectionAirV4,
        trace: &RowMajorMatrix<F>,
    ) {
        let prep = air.preprocessed_trace().unwrap();
        check_constraints::<_, BabyBearPoseidon2Config>(
            air,
            "VerifierWarpActiveCountProjectionAirV4",
            &Some(prep.as_view()),
            &[trace.as_view()],
            &[],
        );
    }

    #[test]
    fn fixed_capacity_projection_separates_leaf_occupancy_from_terminal_child_count() {
        let air = projection_air_v4();
        let prep = air.preprocessed_trace().unwrap();
        for occupancy in 1..=FIXED_MULTI_AIR_SOURCE_CAPACITY_V4 {
            let result = generate_verifier_warp_active_count_projection_trace_v4(
                &air,
                &projection_records_v4(occupancy, 2),
            )
            .unwrap();
            assert_eq!(result.matrix.height(), FIXED_MULTI_AIR_SOURCE_CAPACITY_V4);
            assert_eq!(air.preprocessed_trace().unwrap().values, prep.values);
            check_projection_v4(&air, &result.matrix);
            for (source, certificate) in projection_records_v4(occupancy, 2)
                .iter()
                .zip(&result.messages)
            {
                assert_eq!(certificate.source_root, source.fixed_source.source_root);
                assert_eq!(
                    certificate.expected_active_child_count,
                    source.fixed_source.active_child_count
                );
            }
        }

        let distinct = generate_verifier_warp_active_count_projection_trace_v4(
            &air,
            &projection_records_v4(3, 2),
        )
        .unwrap();
        assert_eq!(distinct.matrix.height(), FIXED_MULTI_AIR_SOURCE_CAPACITY_V4);
        check_projection_v4(&air, &distinct.matrix);
    }

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct FixedCapacityBusHarnessCols<T> {
        active: T,
        raw_opening: VerifiedOneShotRawOpeningMessageV19<T>,
        certified_count: VerifierWarpCertifiedActiveCountMessageV2<T>,
    }

    #[derive(Clone, Debug)]
    struct FixedCapacityBusHarnessAir {
        raw_opening_bus: VerifiedOneShotRawOpeningBusV19,
        certified_count_bus: VerifierWarpCertifiedActiveCountBusV2,
    }

    impl BaseAir<F> for FixedCapacityBusHarnessAir {
        fn width(&self) -> usize {
            core::mem::size_of::<FixedCapacityBusHarnessCols<u8>>()
        }
    }

    impl BaseAirWithPublicValues<F> for FixedCapacityBusHarnessAir {}
    impl PartitionedBaseAir<F> for FixedCapacityBusHarnessAir {}

    impl<AB> Air<AB> for FixedCapacityBusHarnessAir
    where
        AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("fixed-capacity bus harness row");
            let next_row = main
                .row_slice(1)
                .expect("fixed-capacity bus harness next row");
            let local: &FixedCapacityBusHarnessCols<AB::Var> = (*row).borrow();
            let next: &FixedCapacityBusHarnessCols<AB::Var> = (*next_row).borrow();
            builder.assert_bool(local.active);
            builder.when_first_row().assert_one(local.active);
            builder
                .when_transition()
                .assert_zero((AB::Expr::ONE - local.active.into()) * next.active.into());
            let enabled = AB::Expr::from(local.active);
            self.raw_opening_bus.send(
                builder,
                local.raw_opening.clone(),
                enabled.clone() * AB::Expr::TWO,
            );
            self.certified_count_bus
                .lookup_key(builder, local.certified_count.clone(), enabled);
        }
    }

    fn fixed_source_profile_v4() -> FixedMultiAirSourceBoundaryProfileV4 {
        FixedMultiAirSourceBoundaryProfileV4 {
            app_vk_digest: digest(530),
            relation_digest: profile_v4().relation_digest,
            log_message_len: profile_v4().log_message_len,
            log_codeword_len: 8,
        }
    }

    fn fixed_source_records_v4(
        occupancy: usize,
        terminal_active_child_count: u8,
    ) -> Vec<FixedMultiAirSourceBoundaryRecordV4> {
        projection_records_v4(occupancy, terminal_active_child_count)
            .into_iter()
            .map(|record| FixedMultiAirSourceBoundaryRecordV4 {
                active_child_count: if record.fixed_source.active_child_count
                    == F::from_usize(VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2)
                {
                    VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 as u8
                } else {
                    terminal_active_child_count
                },
                source_forest_root: record.fixed_source.source_forest_root,
                segment_openings_digest: record.fixed_source.segment_openings_digest,
                source_root: record.fixed_source.source_root,
                point: record.fixed_source.point[..profile_v4().log_message_len as usize].to_vec(),
                value: record.fixed_source.value,
                verifier_endpoint: record.fixed_source.verifier_endpoint,
            })
            .collect()
    }

    fn fixed_capacity_harness_trace(
        projection_records: &[VerifierWarpActiveCountProjectionRecordV2],
        count_messages: &[VerifierWarpCertifiedActiveCountMessageV2<F>],
        source_log_codeword_len: u8,
    ) -> RowMajorMatrix<F> {
        let width = core::mem::size_of::<FixedCapacityBusHarnessCols<u8>>();
        let mut values = F::zero_vec(width * FIXED_MULTI_AIR_SOURCE_CAPACITY_V4);
        for (slot, (record, count)) in projection_records.iter().zip(count_messages).enumerate() {
            let cols: &mut FixedCapacityBusHarnessCols<F> =
                values[slot * width..(slot + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.raw_opening = VerifiedOneShotRawOpeningMessageV19 {
                proof_index: F::from_usize(slot),
                segment_index_lo: F::from_usize(slot),
                segment_index_hi: F::ZERO,
                shard_ordinal: F::ZERO,
                is_program: F::ZERO,
                source_root: record.fixed_source.source_root,
                range_start: F::ZERO,
                range_end: F::from_u32(1u32 << source_log_codeword_len),
                point_len: record.fixed_source.point_len,
                point: record.fixed_source.point,
                value: record.fixed_source.value,
                program_fingerprint: [F::ZERO; D_EF],
            };
            cols.certified_count = count.clone();
        }
        RowMajorMatrix::new(values, width)
    }

    fn symbolic_interactions_v4(
        air: &dyn AnyAir<BabyBearPoseidon2Config>,
    ) -> Vec<SymbolicInteraction<F>> {
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

    fn check_fixed_capacity_bus_balance(
        airs: &[AirRef<BabyBearPoseidon2Config>],
        traces: &[&RowMajorMatrix<F>],
        selected_buses: &[BusIndex],
        check_air_constraints: bool,
    ) {
        let preprocessed_owned = airs
            .iter()
            .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
            .collect::<Vec<_>>();
        if check_air_constraints {
            for ((air, trace), prep) in airs.iter().zip(traces).zip(&preprocessed_owned) {
                check_constraints::<_, BabyBearPoseidon2Config>(
                    air.as_ref(),
                    &air.name(),
                    &prep.as_ref().map(RowMajorMatrix::as_view),
                    &[trace.as_view()],
                    &[],
                );
            }
        }
        let preprocessed = preprocessed_owned
            .iter()
            .map(|matrix| matrix.as_ref().map(RowMajorMatrix::as_view))
            .collect::<Vec<_>>();
        let interactions = airs
            .iter()
            .map(|air| {
                symbolic_interactions_v4(air.as_ref())
                    .into_iter()
                    .filter(|interaction| selected_buses.contains(&interaction.bus_index))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let views = traces
            .iter()
            .map(|trace| vec![trace.as_view()])
            .collect::<Vec<_>>();
        check_logup(
            &airs.iter().map(|air| air.name()).collect::<Vec<_>>(),
            &interactions,
            &preprocessed,
            &views,
            &vec![Vec::new(); airs.len()],
        );
    }

    #[test]
    fn fixed_capacity_source_and_c2_are_bus_bound_with_runtime_occupancy() {
        let raw_opening_bus_index = 902;
        let fixed_source_bus_index = 905;
        let certified_count_bus_index = 908;
        let source_air = FixedMultiAirSourceBoundaryAirV4 {
            profile: fixed_source_profile_v4(),
            arithmetic_bus: VerifiedLogUpArithmeticBusV19::new(900),
            source_leaf_bus: VerifiedSourceForestLeafBusV19::new(901),
            raw_opening_bus: VerifiedOneShotRawOpeningBusV19::new(raw_opening_bus_index),
            certified_endpoint_bus: CertifiedLogUpOnlyEndpointBusV19::new(903),
            certified_opening_bus: CertifiedSwirlRawOpeningBusV19::new(904),
            certified_source_bus: CertifiedFixedMultiAirSourceBusV2::new(fixed_source_bus_index),
            certified_source_lookup_count: 1,
        };
        let c2_air = VerifierWarpActiveCountProjectionAirV4 {
            profile: profile_v4(),
            source_log_codeword_len: source_air.profile.log_codeword_len,
            auxiliary_challenge_bus: MappedAuxiliaryChallengeBusV19::new(906),
            stream_cursor_bus: OneShotStreamCursorBusV19::new(907),
            raw_opening_bus: source_air.raw_opening_bus,
            fixed_source_bus: source_air.certified_source_bus,
            certified_count_bus: VerifierWarpCertifiedActiveCountBusV2::new(
                certified_count_bus_index,
            ),
        };
        let harness_air = FixedCapacityBusHarnessAir {
            raw_opening_bus: source_air.raw_opening_bus,
            certified_count_bus: c2_air.certified_count_bus,
        };
        let projection_records = projection_records_v4(3, 2);
        let source_records = fixed_source_records_v4(3, 2);
        let source_trace =
            generate_fixed_multi_air_source_boundary_trace_v4(&source_air, &source_records)
                .unwrap();
        let c2 =
            generate_verifier_warp_active_count_projection_trace_v4(&c2_air, &projection_records)
                .unwrap();
        let harness_trace = fixed_capacity_harness_trace(
            &projection_records,
            &c2.messages,
            source_air.profile.log_codeword_len,
        );
        let airs: Vec<AirRef<BabyBearPoseidon2Config>> = vec![
            Arc::new(source_air.clone()),
            Arc::new(c2_air.clone()),
            Arc::new(harness_air),
        ];
        let buses = [
            raw_opening_bus_index,
            fixed_source_bus_index,
            certified_count_bus_index,
        ];
        let mut params = native_warp_history_params_with_100_bits_security();
        params.max_constraint_degree = params.max_constraint_degree.max(2);
        let engine: BabyBearPoseidon2CpuEngine = BabyBearPoseidon2CpuEngine::new(params);
        let expected_key_fingerprint = engine.keygen(&airs).1.pre_hash;
        for occupancy in 1..=VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 {
            let occupancy_source = generate_fixed_multi_air_source_boundary_trace_v4(
                &source_air,
                &fixed_source_records_v4(occupancy, 2),
            )
            .unwrap();
            let occupancy_c2 = generate_verifier_warp_active_count_projection_trace_v4(
                &c2_air,
                &projection_records_v4(occupancy, 2),
            )
            .unwrap();
            assert_eq!(
                occupancy_source.height(),
                FIXED_MULTI_AIR_SOURCE_CAPACITY_V4
            );
            assert_eq!(
                occupancy_c2.matrix.height(),
                FIXED_MULTI_AIR_SOURCE_CAPACITY_V4
            );
            assert_eq!(engine.keygen(&airs).1.pre_hash, expected_key_fingerprint);
        }
        let honest_traces = [&source_trace, &c2.matrix, &harness_trace];
        check_fixed_capacity_bus_balance(&airs, &honest_traces, &buses, true);

        let rejects =
            |source: &RowMajorMatrix<F>, c2: &RowMajorMatrix<F>, harness: &RowMajorMatrix<F>| {
                assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
                    check_fixed_capacity_bus_balance(&airs, &[source, c2, harness], &buses, false)
                }))
                .is_err());
            };

        let mut wrong_root = source_trace.clone();
        let source_width = FixedMultiAirSourceBoundaryColsV4::<u8>::width();
        let row: &mut FixedMultiAirSourceBoundaryColsV4<F> =
            wrong_root.values[source_width..2 * source_width].borrow_mut();
        row.source_root[0] += F::ONE;
        rejects(&wrong_root, &c2.matrix, &harness_trace);

        let wrong_count_records = fixed_source_records_v4(3, 3);
        let wrong_count =
            generate_fixed_multi_air_source_boundary_trace_v4(&source_air, &wrong_count_records)
                .unwrap();
        rejects(&wrong_count, &c2.matrix, &harness_trace);

        let mut wrong_slot = c2.matrix.clone();
        let c2_width = VerifierWarpActiveCountProjectionColsV4::<u8>::width();
        let row: &mut VerifierWarpActiveCountProjectionColsV4<F> =
            wrong_slot.values[c2_width..2 * c2_width].borrow_mut();
        row.fixed_source.proof_index += F::ONE;
        rejects(&source_trace, &wrong_slot, &harness_trace);

        let mut inactive_multiplicity = source_trace.clone();
        let row: &mut FixedMultiAirSourceBoundaryColsV4<F> =
            inactive_multiplicity.values[3 * source_width..4 * source_width].borrow_mut();
        row.active = F::ONE;
        rejects(&inactive_multiplicity, &c2.matrix, &harness_trace);
    }
}

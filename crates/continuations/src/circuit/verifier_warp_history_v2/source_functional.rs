//! Fixed multi-AIR boundary for the recursive LogUp verifier and the single
//! same-root source-opening reduction.
//!
//! This AIR replaces the legacy per-AIR SWIRL boundary.  It accepts exactly
//! one selector-free source per verifier batch and exports an occupancy-bound
//! certificate only after the real LogUp arithmetic, source manifest, and
//! one-shot reduction buses agree.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, PermutationCheckBus},
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, D_EF, F};

use crate::circuit::{
    native_warp_history_v19::{
        CertifiedLogUpOnlyEndpointBusV19, CertifiedLogUpOnlyEndpointMessageV19,
        CertifiedSwirlRawOpeningBusV19, CertifiedSwirlRawOpeningMessageV19,
        VerifiedLogUpArithmeticBusV19, VerifiedLogUpArithmeticMessageV19,
        VerifiedOneShotRawOpeningBusV19, VerifiedOneShotRawOpeningMessageV19,
        VerifiedSourceForestLeafBusV19, VerifiedSourceForestLeafMessageV19,
        LOGUP_ONLY_MODE_TAG_V19, MAX_RAW_MESSAGE_POINT_LEN_V19, NATIVE_WARP_HISTORY_PROTOCOL_V19,
    },
    verifier_warp_history_v2::VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2,
};

macro_rules! define_permutation_bus {
    ($Bus:ident, $Message:ident) => {
        #[derive(Copy, Clone, Debug)]
        pub struct $Bus(PermutationCheckBus);

        impl $Bus {
            #[must_use]
            pub fn new(index: BusIndex) -> Self {
                Self(PermutationCheckBus::new(index))
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

/// Count-bound source result consumed by the v2 producer bridge.  The opening
/// point/value are retained so a splice cannot preserve only opaque digests.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct CertifiedFixedMultiAirSourceMessageV2<T> {
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub active_child_count: T,
    pub app_vk_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub source_root: [T; DIGEST_SIZE],
    pub point_len: T,
    pub point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub value: [T; D_EF],
    pub verifier_endpoint: [T; D_EF],
}

define_permutation_bus!(
    CertifiedFixedMultiAirSourceBusV2,
    CertifiedFixedMultiAirSourceMessageV2
);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedMultiAirSourceBoundaryProfileV2 {
    pub segment_start: u32,
    pub app_vk_digest: [F; DIGEST_SIZE],
    pub relation_digest: [F; DIGEST_SIZE],
    pub log_message_len: u8,
    pub log_codeword_len: u8,
    pub active_child_counts: Arc<[u8]>,
}

impl FixedMultiAirSourceBoundaryProfileV2 {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.log_message_len == 0
            || usize::from(self.log_message_len) > MAX_RAW_MESSAGE_POINT_LEN_V19
            || self.log_codeword_len < self.log_message_len
            || self.log_codeword_len >= 31
            || self.active_child_counts.is_empty()
        {
            return Err("invalid fixed multi-AIR source boundary profile");
        }
        for (index, &count) in self.active_child_counts.iter().enumerate() {
            let count = usize::from(count);
            if count == 0
                || count > VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2
                || (index + 1 != self.active_child_counts.len()
                    && count != VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2)
            {
                return Err("noncanonical fixed multi-AIR source schedule");
            }
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
struct FixedMultiAirSourceBoundaryPrepColsV2<T> {
    active: T,
    proof_index: T,
    segment_index_lo: T,
    segment_index_hi: T,
    active_child_count: T,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct FixedMultiAirSourceBoundaryColsV2<T> {
    pub active: T,
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub source_root: [T; DIGEST_SIZE],
    pub point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub value: [T; D_EF],
    pub verifier_endpoint: [T; D_EF],
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirSourceBoundaryAirV2 {
    pub profile: FixedMultiAirSourceBoundaryProfileV2,
    pub arithmetic_bus: VerifiedLogUpArithmeticBusV19,
    pub source_leaf_bus: VerifiedSourceForestLeafBusV19,
    pub raw_opening_bus: VerifiedOneShotRawOpeningBusV19,
    pub certified_endpoint_bus: CertifiedLogUpOnlyEndpointBusV19,
    pub certified_opening_bus: CertifiedSwirlRawOpeningBusV19,
    pub certified_source_bus: CertifiedFixedMultiAirSourceBusV2,
    /// Setup-owned fanout. Production source provenance requires
    /// `SETUP_PCS_SOURCE_FIXED_LOOKUP_COUNT_V3`: one copy each for the
    /// producer bridge, active-count functional check, and provenance AIR.
    /// Legacy compositions may retain their smaller explicitly configured
    /// fanout.
    pub certified_source_lookup_count: u32,
}

impl BaseAir<F> for FixedMultiAirSourceBoundaryAirV2 {
    fn width(&self) -> usize {
        core::mem::size_of::<FixedMultiAirSourceBoundaryColsV2<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.profile
            .validate()
            .expect("invalid fixed source profile");
        let width = FixedMultiAirSourceBoundaryPrepColsV2::<u8>::width();
        let height = self
            .profile
            .active_child_counts
            .len()
            .next_power_of_two()
            .max(2);
        let mut values = F::zero_vec(width * height);
        for (index, &count) in self.profile.active_child_counts.iter().enumerate() {
            let segment = self.profile.segment_start + index as u32;
            let cols: &mut FixedMultiAirSourceBoundaryPrepColsV2<F> =
                values[index * width..(index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.proof_index = F::from_usize(index);
            cols.segment_index_lo = F::from_u32(segment & 0xffff);
            cols.segment_index_hi = F::from_u32(segment >> 16);
            cols.active_child_count = F::from_u8(count);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for FixedMultiAirSourceBoundaryAirV2 {}
impl PartitionedBaseAir<F> for FixedMultiAirSourceBoundaryAirV2 {}

impl<AB> Air<AB> for FixedMultiAirSourceBoundaryAirV2
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.profile.validate().is_ok());
        let prep_matrix = builder.preprocessed();
        let prep_row = prep_matrix.row_slice(0).expect("fixed source prep row");
        let prep: &FixedMultiAirSourceBoundaryPrepColsV2<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("fixed source boundary row");
        let local: &FixedMultiAirSourceBoundaryColsV2<AB::Var> = (*row).borrow();
        let enabled = prep.active;
        builder.assert_bool(enabled);
        builder.assert_bool(local.active);
        builder.assert_eq(local.active, enabled);
        // Canonicalize the fixed-capacity raw-message point.  The one-shot
        // verifier consumes only `log_message_len` coordinates, while the
        // provenance receipt hashes the complete fixed array.  Zero padding
        // prevents unconstrained tail coordinates from making that receipt
        // malleable or being confused with an unrelated setup PLE point.
        for coordinate in local
            .point
            .iter()
            .skip(self.profile.log_message_len as usize)
        {
            for limb in coordinate {
                builder.when(enabled).assert_zero(*limb);
            }
        }
        // The recursive LogUp verifier's terminal GKR claim is generally
        // non-zero even when the authenticated segment sums cancel. It is
        // bound below into both typed outputs; only the before/after sums are
        // required to be zero.

        self.arithmetic_bus.receive(
            builder,
            VerifiedLogUpArithmeticMessageV19 {
                proof_index: prep.proof_index.into(),
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                mode_tag: AB::Expr::from_u32(LOGUP_ONLY_MODE_TAG_V19),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                app_vk_digest: self.profile.app_vk_digest.map(Into::into),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                verifier_endpoint: local.verifier_endpoint.map(Into::into),
                segment_sum_before: core::array::from_fn(|_| AB::Expr::ZERO),
                segment_sum_after: core::array::from_fn(|_| AB::Expr::ZERO),
                shard_count: AB::Expr::ONE,
            },
            enabled,
        );
        self.source_leaf_bus.receive(
            builder,
            VerifiedSourceForestLeafMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                shard_ordinal: AB::Expr::ZERO,
                air_id: AB::Expr::from_u32(u32::MAX),
                relation_digest: self.profile.relation_digest.map(Into::into),
                log_height: AB::Expr::from_u8(self.profile.log_message_len),
                cached_width: AB::Expr::ZERO,
                log_message_len: AB::Expr::from_u8(self.profile.log_message_len),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                source_root: local.source_root.map(Into::into),
                range_start: AB::Expr::ZERO,
                range_end: AB::Expr::from_u32(1u32 << self.profile.log_codeword_len),
            },
            enabled,
        );
        self.raw_opening_bus.receive(
            builder,
            VerifiedOneShotRawOpeningMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                shard_ordinal: AB::Expr::ZERO,
                is_program: AB::Expr::ZERO,
                source_root: local.source_root.map(Into::into),
                range_start: AB::Expr::ZERO,
                range_end: AB::Expr::from_u32(1u32 << self.profile.log_codeword_len),
                point_len: AB::Expr::from_u8(self.profile.log_message_len),
                point: local.point.map(|point| point.map(Into::into)),
                value: local.value.map(Into::into),
                program_fingerprint: core::array::from_fn(|_| AB::Expr::ZERO),
            },
            enabled,
        );

        self.certified_endpoint_bus.add_key_with_lookups(
            builder,
            CertifiedLogUpOnlyEndpointMessageV19 {
                proof_index: prep.proof_index.into(),
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                mode_tag: AB::Expr::from_u32(LOGUP_ONLY_MODE_TAG_V19),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                app_vk_digest: self.profile.app_vk_digest.map(Into::into),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                verifier_endpoint: local.verifier_endpoint.map(Into::into),
                segment_sum_before: core::array::from_fn(|_| AB::Expr::ZERO),
                segment_sum_after: core::array::from_fn(|_| AB::Expr::ZERO),
            },
            enabled,
        );
        self.certified_opening_bus.add_key_with_lookups(
            builder,
            CertifiedSwirlRawOpeningMessageV19 {
                proof_index: prep.proof_index.into(),
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                mode_tag: AB::Expr::from_u32(LOGUP_ONLY_MODE_TAG_V19),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                shard_ordinal: AB::Expr::ZERO,
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                root: local.source_root.map(Into::into),
                point_len: AB::Expr::from_u8(self.profile.log_message_len),
                point: local.point.map(|point| point.map(Into::into)),
                value: local.value.map(Into::into),
            },
            enabled,
        );
        self.certified_source_bus.send(
            builder,
            CertifiedFixedMultiAirSourceMessageV2 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                active_child_count: prep.active_child_count.into(),
                app_vk_digest: self.profile.app_vk_digest.map(Into::into),
                relation_digest: self.profile.relation_digest.map(Into::into),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                source_root: local.source_root.map(Into::into),
                point_len: AB::Expr::from_u8(self.profile.log_message_len),
                point: local.point.map(|point| point.map(Into::into)),
                value: local.value.map(Into::into),
                verifier_endpoint: local.verifier_endpoint.map(Into::into),
            },
            AB::Expr::from(enabled) * AB::Expr::from_u32(self.certified_source_lookup_count),
        );
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedMultiAirSourceBoundaryRecordV2 {
    pub source_forest_root: [F; DIGEST_SIZE],
    pub segment_openings_digest: [F; DIGEST_SIZE],
    pub source_root: [F; DIGEST_SIZE],
    pub point: Vec<[F; D_EF]>,
    pub value: [F; D_EF],
    pub verifier_endpoint: [F; D_EF],
}

pub fn generate_fixed_multi_air_source_boundary_trace_v2(
    air: &FixedMultiAirSourceBoundaryAirV2,
    records: &[FixedMultiAirSourceBoundaryRecordV2],
) -> Result<RowMajorMatrix<F>, &'static str> {
    air.profile.validate()?;
    if records.len() != air.profile.active_child_counts.len()
        || records
            .iter()
            .any(|record| record.point.len() != air.profile.log_message_len as usize)
    {
        return Err("fixed multi-AIR source boundary records");
    }
    let width = core::mem::size_of::<FixedMultiAirSourceBoundaryColsV2<u8>>();
    let height = records.len().next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    for (index, record) in records.iter().enumerate() {
        let cols: &mut FixedMultiAirSourceBoundaryColsV2<F> =
            values[index * width..(index + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.source_forest_root = record.source_forest_root;
        cols.segment_openings_digest = record.segment_openings_digest;
        cols.source_root = record.source_root;
        for (target, source) in cols.point.iter_mut().zip(&record.point) {
            *target = *source;
        }
        cols.value = record.value;
        cols.verifier_endpoint = record.verifier_endpoint;
    }
    Ok(RowMajorMatrix::new(values, width))
}

/// Capacity of the fixed HLeaf source interval.  This is relation data, not
/// witness-selected occupancy.
pub const FIXED_MULTI_AIR_SOURCE_CAPACITY_V4: usize = 4;

/// Setup-only identity for the fixed-capacity HLeaf source boundary.
///
/// Unlike [`FixedMultiAirSourceBoundaryProfileV2`], this profile deliberately
/// contains no interval length, final active count, or terminal slot.  Those
/// values are authenticated by [`FixedMultiAirSourceBoundaryAirV4`] as main
/// trace data, so one preprocessed trace and one key cover occupancies 1..=4.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedMultiAirSourceBoundaryProfileV4 {
    pub app_vk_digest: [F; DIGEST_SIZE],
    pub relation_digest: [F; DIGEST_SIZE],
    pub log_message_len: u8,
    pub log_codeword_len: u8,
}

impl FixedMultiAirSourceBoundaryProfileV4 {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.log_message_len == 0
            || usize::from(self.log_message_len) > MAX_RAW_MESSAGE_POINT_LEN_V19
            || self.log_codeword_len < self.log_message_len
            || self.log_codeword_len >= 31
        {
            return Err("invalid fixed-capacity source boundary profile");
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
struct FixedMultiAirSourceBoundaryPrepColsV4<T> {
    capacity_active: T,
    is_first: T,
    is_last: T,
    proof_index: T,
    segment_index_lo: T,
    segment_index_hi: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirSourceBoundaryColsV4<T> {
    /// Runtime nonempty-prefix selector.
    pub active: T,
    /// Exactly the last active row, derived in-circuit from `active`.
    pub is_terminal_active: T,
    /// One-hot encoding of the authenticated child count 1..=4.
    pub active_child_count_flags: [T; VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2],
    pub active_child_count: T,
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub source_root: [T; DIGEST_SIZE],
    pub point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub value: [T; D_EF],
    pub verifier_endpoint: [T; D_EF],
}

/// Fixed four-row source boundary with runtime occupancy.
///
/// Every lookup uses `active` as its multiplicity.  Consequently the inactive
/// suffix contributes no source, opening, transcript, or certificate message;
/// its complete witness row is constrained to canonical zero.  The certified
/// count and the opening are emitted with the exact same `source_root` field.
#[derive(Clone, Debug)]
pub struct FixedMultiAirSourceBoundaryAirV4 {
    pub profile: FixedMultiAirSourceBoundaryProfileV4,
    pub arithmetic_bus: VerifiedLogUpArithmeticBusV19,
    pub source_leaf_bus: VerifiedSourceForestLeafBusV19,
    pub raw_opening_bus: VerifiedOneShotRawOpeningBusV19,
    pub certified_endpoint_bus: CertifiedLogUpOnlyEndpointBusV19,
    pub certified_opening_bus: CertifiedSwirlRawOpeningBusV19,
    pub certified_source_bus: CertifiedFixedMultiAirSourceBusV2,
    pub certified_source_lookup_count: u32,
}

impl BaseAir<F> for FixedMultiAirSourceBoundaryAirV4 {
    fn width(&self) -> usize {
        FixedMultiAirSourceBoundaryColsV4::<u8>::width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.profile
            .validate()
            .expect("invalid fixed-capacity source profile");
        let width = FixedMultiAirSourceBoundaryPrepColsV4::<u8>::width();
        let mut values = F::zero_vec(width * FIXED_MULTI_AIR_SOURCE_CAPACITY_V4);
        for slot in 0..FIXED_MULTI_AIR_SOURCE_CAPACITY_V4 {
            let cols: &mut FixedMultiAirSourceBoundaryPrepColsV4<F> =
                values[slot * width..(slot + 1) * width].borrow_mut();
            cols.capacity_active = F::ONE;
            cols.is_first = F::from_bool(slot == 0);
            cols.is_last = F::from_bool(slot + 1 == FIXED_MULTI_AIR_SOURCE_CAPACITY_V4);
            cols.proof_index = F::from_usize(slot);
            cols.segment_index_lo = F::from_usize(slot);
            cols.segment_index_hi = F::ZERO;
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for FixedMultiAirSourceBoundaryAirV4 {}
impl PartitionedBaseAir<F> for FixedMultiAirSourceBoundaryAirV4 {}

impl<AB> Air<AB> for FixedMultiAirSourceBoundaryAirV4
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.profile.validate().is_ok());
        let preprocessed = builder.preprocessed();
        let prep_row = preprocessed
            .row_slice(0)
            .expect("fixed-capacity source prep row");
        let prep: &FixedMultiAirSourceBoundaryPrepColsV4<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("fixed-capacity source row");
        let next_row = main.row_slice(1).expect("fixed-capacity source next row");
        let local: &FixedMultiAirSourceBoundaryColsV4<AB::Var> = (*row).borrow();
        let next: &FixedMultiAirSourceBoundaryColsV4<AB::Var> = (*next_row).borrow();

        for bit in [
            prep.capacity_active,
            prep.is_first,
            prep.is_last,
            local.active,
            local.is_terminal_active,
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
        let is_last = AB::Expr::from(prep.is_last);
        let next_active = AB::Expr::from(next.active);
        let expected_terminal = enabled.clone()
            * (is_last.clone() + (AB::Expr::ONE - is_last) * (AB::Expr::ONE - next_active));
        builder.assert_eq(local.is_terminal_active, expected_terminal.clone());
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
        builder.assert_eq(local.active_child_count, encoded_count);
        builder.when(enabled.clone() - expected_terminal).assert_eq(
            local.active_child_count,
            AB::Expr::from_usize(VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2),
        );

        // Every inactive witness coordinate is canonical zero.  This both
        // removes lookup multiplicity and prevents an unused row from carrying
        // an alternative root/count that a later adapter could accidentally
        // treat as authority.
        let inactive = AB::Expr::from(prep.capacity_active) - enabled.clone();
        for value in (*row).iter().skip(1) {
            builder.when(inactive.clone()).assert_zero((*value).into());
        }
        for point in local
            .point
            .iter()
            .skip(self.profile.log_message_len as usize)
        {
            for limb in point {
                builder.when(enabled.clone()).assert_zero(*limb);
            }
        }

        self.arithmetic_bus.receive(
            builder,
            VerifiedLogUpArithmeticMessageV19 {
                proof_index: prep.proof_index.into(),
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                mode_tag: AB::Expr::from_u32(LOGUP_ONLY_MODE_TAG_V19),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                app_vk_digest: self.profile.app_vk_digest.map(Into::into),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                verifier_endpoint: local.verifier_endpoint.map(Into::into),
                segment_sum_before: core::array::from_fn(|_| AB::Expr::ZERO),
                segment_sum_after: core::array::from_fn(|_| AB::Expr::ZERO),
                shard_count: AB::Expr::ONE,
            },
            enabled.clone(),
        );
        self.source_leaf_bus.receive(
            builder,
            VerifiedSourceForestLeafMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                shard_ordinal: AB::Expr::ZERO,
                air_id: AB::Expr::from_u32(u32::MAX),
                relation_digest: self.profile.relation_digest.map(Into::into),
                log_height: AB::Expr::from_u8(self.profile.log_message_len),
                cached_width: AB::Expr::ZERO,
                log_message_len: AB::Expr::from_u8(self.profile.log_message_len),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                source_root: local.source_root.map(Into::into),
                range_start: AB::Expr::ZERO,
                range_end: AB::Expr::from_u32(1u32 << self.profile.log_codeword_len),
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
                source_root: local.source_root.map(Into::into),
                range_start: AB::Expr::ZERO,
                range_end: AB::Expr::from_u32(1u32 << self.profile.log_codeword_len),
                point_len: AB::Expr::from_u8(self.profile.log_message_len),
                point: local.point.map(|point| point.map(Into::into)),
                value: local.value.map(Into::into),
                program_fingerprint: core::array::from_fn(|_| AB::Expr::ZERO),
            },
            enabled.clone(),
        );

        self.certified_endpoint_bus.add_key_with_lookups(
            builder,
            CertifiedLogUpOnlyEndpointMessageV19 {
                proof_index: prep.proof_index.into(),
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                mode_tag: AB::Expr::from_u32(LOGUP_ONLY_MODE_TAG_V19),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                app_vk_digest: self.profile.app_vk_digest.map(Into::into),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                verifier_endpoint: local.verifier_endpoint.map(Into::into),
                segment_sum_before: core::array::from_fn(|_| AB::Expr::ZERO),
                segment_sum_after: core::array::from_fn(|_| AB::Expr::ZERO),
            },
            enabled.clone(),
        );
        self.certified_opening_bus.add_key_with_lookups(
            builder,
            CertifiedSwirlRawOpeningMessageV19 {
                proof_index: prep.proof_index.into(),
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                mode_tag: AB::Expr::from_u32(LOGUP_ONLY_MODE_TAG_V19),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                shard_ordinal: AB::Expr::ZERO,
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                root: local.source_root.map(Into::into),
                point_len: AB::Expr::from_u8(self.profile.log_message_len),
                point: local.point.map(|point| point.map(Into::into)),
                value: local.value.map(Into::into),
            },
            enabled.clone(),
        );
        self.certified_source_bus.send(
            builder,
            CertifiedFixedMultiAirSourceMessageV2 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                active_child_count: local.active_child_count.into(),
                app_vk_digest: self.profile.app_vk_digest.map(Into::into),
                relation_digest: self.profile.relation_digest.map(Into::into),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                source_root: local.source_root.map(Into::into),
                point_len: AB::Expr::from_u8(self.profile.log_message_len),
                point: local.point.map(|point| point.map(Into::into)),
                value: local.value.map(Into::into),
                verifier_endpoint: local.verifier_endpoint.map(Into::into),
            },
            enabled * AB::Expr::from_u32(self.certified_source_lookup_count),
        );
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedMultiAirSourceBoundaryRecordV4 {
    pub active_child_count: u8,
    pub source_forest_root: [F; DIGEST_SIZE],
    pub segment_openings_digest: [F; DIGEST_SIZE],
    pub source_root: [F; DIGEST_SIZE],
    pub point: Vec<[F; D_EF]>,
    pub value: [F; D_EF],
    pub verifier_endpoint: [F; D_EF],
}

pub fn generate_fixed_multi_air_source_boundary_trace_v4(
    air: &FixedMultiAirSourceBoundaryAirV4,
    records: &[FixedMultiAirSourceBoundaryRecordV4],
) -> Result<RowMajorMatrix<F>, &'static str> {
    air.profile.validate()?;
    if records.is_empty()
        || records.len() > FIXED_MULTI_AIR_SOURCE_CAPACITY_V4
        || records.iter().any(|record| {
            record.active_child_count == 0
                || usize::from(record.active_child_count) > VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2
                || record.point.len() != air.profile.log_message_len as usize
        })
        || records[..records.len() - 1].iter().any(|record| {
            usize::from(record.active_child_count) != VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2
        })
    {
        return Err("noncanonical fixed-capacity source records");
    }
    let width = FixedMultiAirSourceBoundaryColsV4::<u8>::width();
    let mut values = F::zero_vec(width * FIXED_MULTI_AIR_SOURCE_CAPACITY_V4);
    for (slot, record) in records.iter().enumerate() {
        let cols: &mut FixedMultiAirSourceBoundaryColsV4<F> =
            values[slot * width..(slot + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.is_terminal_active = F::from_bool(slot + 1 == records.len());
        cols.active_child_count_flags[usize::from(record.active_child_count) - 1] = F::ONE;
        cols.active_child_count = F::from_u8(record.active_child_count);
        cols.source_forest_root = record.source_forest_root;
        cols.segment_openings_digest = record.segment_openings_digest;
        cols.source_root = record.source_root;
        for (target, source) in cols.point.iter_mut().zip(&record.point) {
            *target = *source;
        }
        cols.value = record.value;
        cols.verifier_endpoint = record.verifier_endpoint;
    }
    Ok(RowMajorMatrix::new(values, width))
}

#[cfg(test)]
mod fixed_capacity_v4_tests {
    use std::panic::AssertUnwindSafe;

    use openvm_stark_backend::{air_builders::debug::check_constraints, p3_matrix::Matrix};
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

    use super::*;

    fn digest(seed: u32) -> [F; DIGEST_SIZE] {
        core::array::from_fn(|index| F::from_u32(seed + index as u32))
    }

    fn air() -> FixedMultiAirSourceBoundaryAirV4 {
        FixedMultiAirSourceBoundaryAirV4 {
            profile: FixedMultiAirSourceBoundaryProfileV4 {
                app_vk_digest: digest(10),
                relation_digest: digest(20),
                log_message_len: 3,
                log_codeword_len: 5,
            },
            arithmetic_bus: VerifiedLogUpArithmeticBusV19::new(100),
            source_leaf_bus: VerifiedSourceForestLeafBusV19::new(101),
            raw_opening_bus: VerifiedOneShotRawOpeningBusV19::new(102),
            certified_endpoint_bus: CertifiedLogUpOnlyEndpointBusV19::new(103),
            certified_opening_bus: CertifiedSwirlRawOpeningBusV19::new(104),
            certified_source_bus: CertifiedFixedMultiAirSourceBusV2::new(105),
            certified_source_lookup_count: 1,
        }
    }

    fn records_with_terminal_count(
        occupancy: usize,
        terminal_active_child_count: u8,
    ) -> Vec<FixedMultiAirSourceBoundaryRecordV4> {
        (0..occupancy)
            .map(|slot| FixedMultiAirSourceBoundaryRecordV4 {
                active_child_count: if slot + 1 == occupancy {
                    terminal_active_child_count
                } else {
                    VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2 as u8
                },
                source_forest_root: digest(100 + slot as u32 * 10),
                segment_openings_digest: digest(200 + slot as u32 * 10),
                source_root: digest(300 + slot as u32 * 10),
                point: vec![[F::from_usize(slot + 1); D_EF]; 3],
                value: [F::from_usize(slot + 2); D_EF],
                verifier_endpoint: [F::from_usize(slot + 3); D_EF],
            })
            .collect()
    }

    fn records(occupancy: usize) -> Vec<FixedMultiAirSourceBoundaryRecordV4> {
        records_with_terminal_count(occupancy, 2)
    }

    fn check(air: &FixedMultiAirSourceBoundaryAirV4, trace: &RowMajorMatrix<F>) {
        let prep = air.preprocessed_trace().unwrap();
        check_constraints::<_, BabyBearPoseidon2Config>(
            air,
            "FixedMultiAirSourceBoundaryAirV4",
            &Some(prep.as_view()),
            &[trace.as_view()],
            &[],
        );
    }

    #[test]
    fn occupancy_one_through_four_uses_one_fixed_profile_and_four_rows() {
        let air = air();
        let prep = air.preprocessed_trace().unwrap();
        let width = air.width();
        for occupancy in 1..=FIXED_MULTI_AIR_SOURCE_CAPACITY_V4 {
            let trace =
                generate_fixed_multi_air_source_boundary_trace_v4(&air, &records(occupancy))
                    .unwrap();
            assert_eq!(trace.height(), FIXED_MULTI_AIR_SOURCE_CAPACITY_V4);
            assert_eq!(trace.width(), width);
            assert_eq!(air.preprocessed_trace().unwrap(), prep);
            check(&air, &trace);
        }

        // HLeaf occupancy counts WARP transitions; the terminal source count
        // counts segment proofs in that source. They are intentionally
        // independent runtime values under the same key.
        let occupancy_three_terminal_two = generate_fixed_multi_air_source_boundary_trace_v4(
            &air,
            &records_with_terminal_count(3, 2),
        )
        .unwrap();
        assert_eq!(air.preprocessed_trace().unwrap(), prep);
        check(&air, &occupancy_three_terminal_two);
    }

    #[test]
    fn prefix_terminal_count_and_inactive_suffix_mutations_reject() {
        let air = air();
        let width = FixedMultiAirSourceBoundaryColsV4::<u8>::width();
        let honest = generate_fixed_multi_air_source_boundary_trace_v4(&air, &records(2)).unwrap();
        check(&air, &honest);

        let rejects = |mutate: &dyn Fn(&mut RowMajorMatrix<F>)| {
            let mut changed = honest.clone();
            mutate(&mut changed);
            assert!(std::panic::catch_unwind(AssertUnwindSafe(|| check(&air, &changed))).is_err());
        };
        rejects(&|trace| {
            let row: &mut FixedMultiAirSourceBoundaryColsV4<F> =
                trace.values[width..2 * width].borrow_mut();
            row.is_terminal_active = F::ZERO;
        });
        rejects(&|trace| {
            let row: &mut FixedMultiAirSourceBoundaryColsV4<F> = trace.values[..width].borrow_mut();
            row.active_child_count_flags = [F::ZERO; VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2];
            row.active_child_count_flags[0] = F::ONE;
            row.active_child_count = F::ONE;
        });
        rejects(&|trace| {
            let row: &mut FixedMultiAirSourceBoundaryColsV4<F> =
                trace.values[2 * width..3 * width].borrow_mut();
            row.active = F::ONE;
        });
        rejects(&|trace| {
            let row: &mut FixedMultiAirSourceBoundaryColsV4<F> =
                trace.values[2 * width..3 * width].borrow_mut();
            row.source_root[0] = F::ONE;
        });
    }
}

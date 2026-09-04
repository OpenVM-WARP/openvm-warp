//! Production source-side transcript plumbing for protocol-v19 LogUp/SWIRL.
//!
//! This file owns only the phases surrounding the recursive LogUp-only
//! verifier. The recursive GKR/batch-constraint module supplies the cursor at
//! which per-source claim derivation begins. Claim derivation for every active
//! source is completed first, in canonical shard order. The one-shot opening
//! reductions then run on the same transcript, also in canonical shard order.
//! No AIR in this module can authorize its own mapped functional.

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit::bus::TranscriptBus;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, PermutationCheckBus},
    native_warp::direct_message_opening_reduction_claim_observations,
    warp_pesat::TerminalStructuredLinearClaim,
    BaseAirWithPublicValues, FiatShamirTranscript, PartitionedBaseAir, TranscriptHistory,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config, DuplexSpongeRecorder, DIGEST_SIZE, D_EF, EF, F,
};
use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

#[cfg(test)]
use super::{
    OneShotRoundStartBusV19, OneShotRoundStartMessageV19, VerifiedOneShotClaimObservationBusV19,
    VerifiedOneShotClaimObservationMessageV19,
};
use super::{
    OneShotStreamCursorBusV19, OneShotStreamCursorMessageV19, NATIVE_WARP_HISTORY_PROTOCOL_V19,
};
use crate::circuit::verifier_warp_history_v2::{
    VerifierWarpActiveCountProfileV4, FIXED_MULTI_AIR_SOURCE_CAPACITY_V4,
    VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2,
};

/// Exact tag used by `derive_direct_air_logup_claims` before each source's
/// ordinary mapped functional challenge.
pub const DIRECT_LOGUP_CLAIM_TAG_V19: u64 = 0x4e57_4c43_4c4d_0013;
/// Exact tag used before the Program cached-column target and mixing sample.
pub const PROGRAM_FINGERPRINT_TARGET_TAG_V19: u64 = 0x4e57_5052_4654_0013;

macro_rules! define_permutation_bus {
    ($Bus:ident, $Message:ident) => {
        #[derive(Copy, Clone, Debug)]
        pub struct $Bus(PermutationCheckBus);

        impl $Bus {
            #[must_use]
            pub fn new(bus_index: BusIndex) -> Self {
                Self(PermutationCheckBus::new(bus_index))
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

/// Cursor exported by the recursive LogUp-only verifier immediately before
/// the dynamic column-opening vector is absorbed.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct LogUpOpeningPhaseStartMessageV19<T> {
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub opening_start_tidx: T,
}

define_permutation_bus!(
    LogUpOpeningPhaseStartBusV19,
    LogUpOpeningPhaseStartMessageV19
);

/// Cursor exported after the authenticated dynamic column-opening vector has
/// been absorbed. The claim-derivation AIR is the sole receiver.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct LogUpClaimPhaseStartMessageV19<T> {
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub first_claim_tidx: T,
}

define_permutation_bus!(LogUpClaimPhaseStartBusV19, LogUpClaimPhaseStartMessageV19);

/// Fiat--Shamir challenges consumed by the mapped-functional evaluator.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct MappedFunctionalChallengeMessageV19<T> {
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub shard_ordinal: T,
    pub shard_id: T,
    pub relation_digest: [T; DIGEST_SIZE],
    pub is_program: T,
    pub batching_challenge: [T; D_EF],
    pub program_fingerprint: [T; D_EF],
    pub program_mix_challenge: [T; D_EF],
}

define_permutation_bus!(
    MappedFunctionalChallengeBusV19,
    MappedFunctionalChallengeMessageV19
);

/// Verifier-derived non-zero coefficient for an auxiliary obligation folded
/// into the same mapped source functional.  Protocol v2 uses this for the
/// canonical `VmPvsCols::is_valid` occupancy term.  Keeping it on a distinct
/// typed bus prevents a host witness from choosing the count coefficient or
/// silently omitting the term when the ordinary LogUp challenge is zero.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct MappedAuxiliaryChallengeMessageV19<T> {
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub shard_ordinal: T,
    pub relation_digest: [T; DIGEST_SIZE],
    pub expected_value: T,
    pub challenge: [T; D_EF],
}

define_permutation_bus!(
    MappedAuxiliaryChallengeBusV19,
    MappedAuxiliaryChallengeMessageV19
);

pub const ACTIVE_CHILD_COUNT_BINDING_VERSION_V19: u32 = 1;
pub const ACTIVE_CHILD_COUNT_PROTOCOL_VERSION_V19: u32 = 2;
const ACTIVE_CHILD_COUNT_DOMAIN_V19: &[u8] = b"openvm-verifier-warp-active-child-count-v1";

/// Complete setup-owned transcript material for the canonical occupancy term.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActiveChildCountTranscriptProfileV19 {
    pub segment_start: u32,
    pub relation_digest: [F; DIGEST_SIZE],
    pub active_child_counts: std::sync::Arc<[u8]>,
    pub vm_pvs_air_id: u32,
    pub is_valid_common_main_column: u32,
    pub is_valid_message_block_start: u64,
    pub vm_pvs_log_height: u8,
    pub trace_heights: std::sync::Arc<[u32]>,
}

impl ActiveChildCountTranscriptProfileV19 {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.active_child_counts.is_empty()
            || self.trace_heights.is_empty()
            || self.trace_heights.iter().any(|&height| height == 0)
        {
            return Err("empty active-child-count transcript profile");
        }
        for (index, &count) in self.active_child_counts.iter().enumerate() {
            if count == 0
                || usize::from(count) > VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2
                || (index + 1 != self.active_child_counts.len()
                    && usize::from(count) != VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2)
            {
                return Err("noncanonical active-child-count schedule");
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn segment_count(&self) -> u64 {
        self.active_child_counts
            .iter()
            .map(|&count| u64::from(count))
            .sum()
    }

    #[must_use]
    pub fn observation_count(&self) -> usize {
        ACTIVE_CHILD_COUNT_DOMAIN_V19.len()
            + 2
            + 4 * 3
            + 1
            + 1
            + 1
            + 4
            + 1
            + 4 * 5
            + self.trace_heights.len() * 4
            + DIGEST_SIZE
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
struct ActiveChildCountTranscriptPrepColsV19<T> {
    active: T,
    proof_index: T,
    batch_index_limbs: [T; 4],
    segment_index_lo: T,
    segment_index_hi: T,
    expected_count: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ActiveChildCountTranscriptColsV19<T> {
    pub active: T,
    pub start_tidx: T,
    pub sampled: [T; D_EF],
    pub sampled_inverse: [T; D_EF],
    pub sampled_is_zero: T,
    pub challenge: [T; D_EF],
    pub end_tidx: T,
}

/// Replays the native active-count Fiat--Shamir phase and emits the sole
/// coefficient accepted by the mapped source evaluator.
#[derive(Clone, ColumnsAir)]
#[columns_via(ActiveChildCountTranscriptColsV19<u8>)]
pub struct ActiveChildCountTranscriptAirV19 {
    pub profile: ActiveChildCountTranscriptProfileV19,
    pub transcript_bus: TranscriptBus,
    pub input_cursor_bus: OneShotStreamCursorBusV19,
    pub output_cursor_bus: OneShotStreamCursorBusV19,
    pub challenge_bus: MappedAuxiliaryChallengeBusV19,
    /// Verifier-key-owned fanout. One lookup is consumed by the mapped
    /// functional and an optional second lookup is consumed by History-v2's
    /// C2 projection. Witness-selected multiplicities are forbidden.
    pub challenge_lookup_count: u32,
}

impl BaseAir<F> for ActiveChildCountTranscriptAirV19 {
    fn width(&self) -> usize {
        ActiveChildCountTranscriptColsV19::<F>::width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.profile
            .validate()
            .expect("invalid active-child-count transcript profile");
        let width = ActiveChildCountTranscriptPrepColsV19::<F>::width();
        let height = self
            .profile
            .active_child_counts
            .len()
            .next_power_of_two()
            .max(2);
        let mut values = F::zero_vec(width * height);
        for (index, &count) in self.profile.active_child_counts.iter().enumerate() {
            let cols: &mut ActiveChildCountTranscriptPrepColsV19<F> =
                values[index * width..(index + 1) * width].borrow_mut();
            let segment_index = self.profile.segment_start + index as u32;
            cols.active = F::ONE;
            cols.proof_index = F::from_usize(index);
            cols.batch_index_limbs = (index as u64)
                .to_le_bytes()
                .chunks_exact(2)
                .map(|bytes| F::from_u16(u16::from_le_bytes([bytes[0], bytes[1]])))
                .collect::<Vec<_>>()
                .try_into()
                .expect("four batch-index limbs");
            cols.segment_index_lo = F::from_u32(segment_index & 0xffff);
            cols.segment_index_hi = F::from_u32(segment_index >> 16);
            cols.expected_count = F::from_u8(count);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for ActiveChildCountTranscriptAirV19 {}
impl PartitionedBaseAir<F> for ActiveChildCountTranscriptAirV19 {}

impl<AB> Air<AB> for ActiveChildCountTranscriptAirV19
where
    AB: AirBuilder<F = F> + InteractionBuilder + PairBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.profile.validate().is_ok());
        let preprocessed = builder.preprocessed();
        let prep_row = preprocessed
            .row_slice(0)
            .expect("active-count preprocessed row");
        let prep: &ActiveChildCountTranscriptPrepColsV19<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("active-count transcript row");
        let local: &ActiveChildCountTranscriptColsV19<AB::Var> = (*row).borrow();
        let enabled = AB::Expr::from(prep.active);
        builder.assert_bool(prep.active);
        builder.assert_bool(local.active);
        builder.assert_eq(local.active, prep.active);
        builder.assert_bool(local.sampled_is_zero);

        self.input_cursor_bus.receive(
            builder,
            OneShotStreamCursorMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                next_shard_ordinal: AB::Expr::ZERO,
                tidx: local.start_tidx.into(),
            },
            enabled.clone(),
        );

        let mut tidx = AB::Expr::from(local.start_tidx);
        {
            let mut observe = |builder: &mut AB, value: AB::Expr| {
                self.transcript_bus.observe(
                    builder,
                    prep.proof_index,
                    tidx.clone(),
                    value,
                    enabled.clone(),
                );
                tidx += AB::Expr::ONE;
            };
            for &byte in ACTIVE_CHILD_COUNT_DOMAIN_V19 {
                observe(builder, AB::Expr::from_u8(byte));
            }
            observe(
                builder,
                AB::Expr::from_u32(ACTIVE_CHILD_COUNT_PROTOCOL_VERSION_V19),
            );
            observe(
                builder,
                AB::Expr::from_u32(ACTIVE_CHILD_COUNT_BINDING_VERSION_V19),
            );
            for limb in prep.batch_index_limbs {
                observe(builder, limb.into());
            }
            observe_u64_const_v19(&mut observe, builder, self.profile.segment_count());
            observe_u64_const_v19(
                &mut observe,
                builder,
                self.profile.active_child_counts.len() as u64,
            );
            observe(builder, prep.expected_count.into());
            observe(builder, AB::Expr::from_u32(self.profile.vm_pvs_air_id));
            observe(
                builder,
                AB::Expr::from_u32(self.profile.is_valid_common_main_column),
            );
            observe_u64_const_v19(
                &mut observe,
                builder,
                self.profile.is_valid_message_block_start,
            );
            observe(builder, AB::Expr::from_u8(self.profile.vm_pvs_log_height));
            for value in [
                self.profile.segment_count(),
                4,
                self.profile.active_child_counts.len() as u64,
                u64::from(*self.profile.active_child_counts.last().unwrap()),
                self.profile.trace_heights.len() as u64,
            ] {
                observe_u64_const_v19(&mut observe, builder, value);
            }
            for &height in self.profile.trace_heights.iter() {
                observe_u64_const_v19(&mut observe, builder, u64::from(height));
            }
        }
        self.transcript_bus.observe_commit(
            builder,
            prep.proof_index,
            tidx.clone(),
            self.profile.relation_digest,
            enabled.clone(),
        );
        tidx += AB::Expr::from_usize(DIGEST_SIZE);
        self.transcript_bus.sample_ext(
            builder,
            prep.proof_index,
            tidx.clone(),
            local.sampled,
            enabled.clone(),
        );
        tidx += AB::Expr::from_usize(D_EF);

        let product = ext_mul_active_count_v19::<AB::Expr>(local.sampled, local.sampled_inverse);
        for (limb, value) in product.into_iter().enumerate() {
            builder.when(enabled.clone()).assert_eq(
                value,
                if limb == 0 {
                    AB::Expr::ONE - AB::Expr::from(local.sampled_is_zero)
                } else {
                    AB::Expr::ZERO
                },
            );
            builder
                .when(enabled.clone())
                .assert_zero(AB::Expr::from(local.sampled[limb]) * local.sampled_is_zero);
            builder.when(enabled.clone()).assert_eq(
                local.challenge[limb],
                AB::Expr::from(local.sampled[limb])
                    + AB::Expr::from_bool(limb == 0) * local.sampled_is_zero,
            );
        }
        self.transcript_bus.observe(
            builder,
            prep.proof_index,
            tidx.clone(),
            local.sampled_is_zero,
            enabled.clone(),
        );
        tidx += AB::Expr::ONE;
        builder
            .when(enabled.clone())
            .assert_eq(local.end_tidx, tidx.clone());

        self.challenge_bus.send(
            builder,
            MappedAuxiliaryChallengeMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                shard_ordinal: AB::Expr::ZERO,
                relation_digest: self.profile.relation_digest.map(Into::into),
                expected_value: prep.expected_count.into(),
                challenge: local.challenge.map(Into::into),
            },
            enabled.clone() * AB::Expr::from_u32(self.challenge_lookup_count),
        );
        self.output_cursor_bus.send(
            builder,
            OneShotStreamCursorMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                next_shard_ordinal: AB::Expr::ZERO,
                tidx,
            },
            enabled,
        );
    }
}

fn observe_u64_const_v19<AB, O>(observe: &mut O, builder: &mut AB, value: u64)
where
    AB: AirBuilder<F = F>,
    O: FnMut(&mut AB, AB::Expr),
{
    for limb in 0..4 {
        observe(
            builder,
            AB::Expr::from_u16(((value >> (16 * limb)) & 0xffff) as u16),
        );
    }
}

fn ext_mul_active_count_v19<FA>(
    left: [impl Into<FA>; D_EF],
    right: [impl Into<FA>; D_EF],
) -> [FA; D_EF]
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

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActiveChildCountTranscriptRecordV19 {
    pub proof_index: u32,
    pub segment_index: u32,
    pub start_tidx: u32,
    pub sampled: EF,
}

/// Replay the native occupancy phase on the host and return the exact record
/// consumed by [`ActiveChildCountTranscriptAirV19`].  Keeping this beside the
/// AIR prevents the context generator from silently drifting to a different
/// observation order or domain separator.
pub fn observe_active_child_count_transcript_v19(
    profile: &ActiveChildCountTranscriptProfileV19,
    proof_index: usize,
    transcript: &mut DuplexSpongeRecorder,
) -> Result<ActiveChildCountTranscriptRecordV19, &'static str> {
    profile.validate()?;
    let expected_count = *profile
        .active_child_counts
        .get(proof_index)
        .ok_or("active-child-count proof index")?;
    let segment_index = profile
        .segment_start
        .checked_add(u32::try_from(proof_index).map_err(|_| "active-child-count proof index")?)
        .ok_or("active-child-count segment index")?;
    let start_tidx =
        u32::try_from(transcript.len()).map_err(|_| "active-child-count transcript cursor")?;

    let observe = |transcript: &mut DuplexSpongeRecorder, value: F| {
        FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(transcript, value);
    };
    let observe_u64 = |transcript: &mut DuplexSpongeRecorder, value: u64| {
        for limb in 0..4 {
            observe(
                transcript,
                F::from_u16(((value >> (16 * limb)) & 0xffff) as u16),
            );
        }
    };

    for &byte in ACTIVE_CHILD_COUNT_DOMAIN_V19 {
        observe(transcript, F::from_u8(byte));
    }
    observe(
        transcript,
        F::from_u32(ACTIVE_CHILD_COUNT_PROTOCOL_VERSION_V19),
    );
    observe(
        transcript,
        F::from_u32(ACTIVE_CHILD_COUNT_BINDING_VERSION_V19),
    );
    observe_u64(transcript, proof_index as u64);
    observe_u64(transcript, profile.segment_count());
    observe_u64(transcript, profile.active_child_counts.len() as u64);
    observe(transcript, F::from_u8(expected_count));
    observe(transcript, F::from_u32(profile.vm_pvs_air_id));
    observe(transcript, F::from_u32(profile.is_valid_common_main_column));
    observe_u64(transcript, profile.is_valid_message_block_start);
    observe(transcript, F::from_u8(profile.vm_pvs_log_height));
    for value in [
        profile.segment_count(),
        4,
        profile.active_child_counts.len() as u64,
        u64::from(*profile.active_child_counts.last().unwrap()),
        profile.trace_heights.len() as u64,
    ] {
        observe_u64(transcript, value);
    }
    for &height in profile.trace_heights.iter() {
        observe_u64(transcript, u64::from(height));
    }
    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_commit(
        transcript,
        profile.relation_digest,
    );
    let sampled = FiatShamirTranscript::<BabyBearPoseidon2Config>::sample_ext(transcript);
    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(
        transcript,
        F::from_bool(sampled == EF::ZERO),
    );
    let expected_end = usize::try_from(start_tidx)
        .map_err(|_| "active-child-count transcript cursor")?
        + profile.observation_count()
        + D_EF
        + 1;
    if transcript.len() != expected_end {
        return Err("active-child-count observation schedule");
    }
    Ok(ActiveChildCountTranscriptRecordV19 {
        proof_index: u32::try_from(proof_index).map_err(|_| "active-child-count proof index")?,
        segment_index,
        start_tidx,
        sampled,
    })
}

pub fn generate_active_child_count_transcript_trace_v19(
    air: &ActiveChildCountTranscriptAirV19,
    records: &[ActiveChildCountTranscriptRecordV19],
) -> Result<RowMajorMatrix<F>, &'static str> {
    air.profile.validate()?;
    if records.len() != air.profile.active_child_counts.len() {
        return Err("active-child-count transcript record count");
    }
    let width = ActiveChildCountTranscriptColsV19::<F>::width();
    let height = records.len().next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    for (index, record) in records.iter().enumerate() {
        if record.proof_index as usize != index
            || record.segment_index != air.profile.segment_start + index as u32
        {
            return Err("active-child-count transcript record order");
        }
        let cols: &mut ActiveChildCountTranscriptColsV19<F> =
            values[index * width..(index + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.start_tidx = F::from_u32(record.start_tidx);
        copy_ext(&mut cols.sampled, record.sampled);
        let is_zero = record.sampled == EF::ZERO;
        cols.sampled_is_zero = F::from_bool(is_zero);
        copy_ext(
            &mut cols.sampled_inverse,
            if is_zero {
                EF::ZERO
            } else {
                record.sampled.inverse()
            },
        );
        copy_ext(
            &mut cols.challenge,
            if is_zero { EF::ONE } else { record.sampled },
        );
        cols.end_tidx =
            F::from_usize(record.start_tidx as usize + air.profile.observation_count() + D_EF + 1);
    }
    Ok(RowMajorMatrix::new(values, width))
}

/// Domain-separated binding version for the fixed-capacity runtime schedule.
pub const ACTIVE_CHILD_COUNT_BINDING_VERSION_V4: u32 = 2;

/// Row domain for the runtime active-count AIR: one row per WARP transition
/// in a History leaf. This is deliberately independent of
/// [`VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2`], which counts the segment proofs
/// inside each individual WARP source.
pub const ACTIVE_CHILD_COUNT_TRACE_HEIGHT_V4: usize = FIXED_MULTI_AIR_SOURCE_CAPACITY_V4;

fn active_child_count_observation_count_v4(profile: &VerifierWarpActiveCountProfileV4) -> usize {
    ACTIVE_CHILD_COUNT_DOMAIN_V19.len()
        + 2
        + 4 // local slot
        + 4 // fixed HLeaf capacity
        + 1 // runtime active child count
        + 1 // runtime terminal-active flag
        + 1 // VmPvs AIR id
        + 1 // is_valid column
        + 4 // message block start
        + 1 // VmPvs log height
        + 4 // fixed verifier batch arity
        + 4 // trace-height vector length
        + 4 * profile.trace_heights.len()
        + DIGEST_SIZE
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
struct ActiveChildCountTranscriptPrepColsV4<T> {
    capacity_active: T,
    is_first: T,
    is_last: T,
    proof_index: T,
    batch_index_limbs: [T; 4],
    segment_index_lo: T,
    segment_index_hi: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ActiveChildCountTranscriptColsV4<T> {
    pub active: T,
    pub is_terminal_active: T,
    pub active_child_count_flags: [T; VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2],
    pub active_child_count: T,
    pub start_tidx: T,
    pub sampled: [T; D_EF],
    pub sampled_inverse: [T; D_EF],
    pub sampled_is_zero: T,
    pub challenge: [T; D_EF],
    pub end_tidx: T,
}

/// Fixed-capacity active-count transcript. Occupancy and the terminal source's
/// child count are witness data constrained by this AIR; only structural
/// verifier metadata remains in the profile/key. The physical trace is padded
/// to [`ACTIVE_CHILD_COUNT_TRACE_HEIGHT_V4`] rows for the PCS.
#[derive(Clone, ColumnsAir)]
#[columns_via(ActiveChildCountTranscriptColsV4<u8>)]
pub struct ActiveChildCountTranscriptAirV4 {
    pub profile: VerifierWarpActiveCountProfileV4,
    pub transcript_bus: TranscriptBus,
    pub input_cursor_bus: OneShotStreamCursorBusV19,
    pub output_cursor_bus: OneShotStreamCursorBusV19,
    pub challenge_bus: MappedAuxiliaryChallengeBusV19,
    pub challenge_lookup_count: u32,
}

impl BaseAir<F> for ActiveChildCountTranscriptAirV4 {
    fn width(&self) -> usize {
        ActiveChildCountTranscriptColsV4::<F>::width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.profile
            .validate()
            .expect("invalid fixed-capacity active-count transcript profile");
        let width = ActiveChildCountTranscriptPrepColsV4::<F>::width();
        let mut values = F::zero_vec(width * ACTIVE_CHILD_COUNT_TRACE_HEIGHT_V4);
        for slot in 0..ACTIVE_CHILD_COUNT_TRACE_HEIGHT_V4 {
            let prep: &mut ActiveChildCountTranscriptPrepColsV4<F> =
                values[slot * width..(slot + 1) * width].borrow_mut();
            prep.capacity_active = F::ONE;
            prep.is_first = F::from_bool(slot == 0);
            prep.is_last = F::from_bool(slot + 1 == ACTIVE_CHILD_COUNT_TRACE_HEIGHT_V4);
            prep.proof_index = F::from_usize(slot);
            prep.batch_index_limbs = (slot as u64)
                .to_le_bytes()
                .chunks_exact(2)
                .map(|bytes| F::from_u16(u16::from_le_bytes([bytes[0], bytes[1]])))
                .collect::<Vec<_>>()
                .try_into()
                .expect("four fixed-capacity batch-index limbs");
            prep.segment_index_lo = F::from_usize(slot);
            prep.segment_index_hi = F::ZERO;
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for ActiveChildCountTranscriptAirV4 {}
impl PartitionedBaseAir<F> for ActiveChildCountTranscriptAirV4 {}

impl<AB> Air<AB> for ActiveChildCountTranscriptAirV4
where
    AB: AirBuilder<F = F> + InteractionBuilder + PairBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.profile.validate().is_ok());
        let preprocessed = builder.preprocessed();
        let prep_row = preprocessed
            .row_slice(0)
            .expect("fixed-capacity active-count prep row");
        let prep: &ActiveChildCountTranscriptPrepColsV4<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("fixed-capacity active-count row");
        let next_row = main
            .row_slice(1)
            .expect("fixed-capacity active-count next row");
        let local: &ActiveChildCountTranscriptColsV4<AB::Var> = (*row).borrow();
        let next: &ActiveChildCountTranscriptColsV4<AB::Var> = (*next_row).borrow();
        for bit in [
            prep.capacity_active,
            prep.is_first,
            prep.is_last,
            local.active,
            local.is_terminal_active,
            local.sampled_is_zero,
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
        // A row outside the setup-fixed History-leaf schedule cannot become
        // an unauthenticated extra transition.
        builder.assert_zero(enabled.clone() * (AB::Expr::ONE - capacity_active));
        let is_last = AB::Expr::from(prep.is_last);
        let terminal = enabled.clone()
            * (is_last.clone()
                + (AB::Expr::ONE - is_last) * (AB::Expr::ONE - AB::Expr::from(next.active)));
        builder.assert_eq(local.is_terminal_active, terminal.clone());
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
        builder.when(enabled.clone() - terminal).assert_eq(
            local.active_child_count,
            AB::Expr::from_usize(VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2),
        );
        // Disabled semantic slots and physical padding rows are both
        // canonical zero rows. This also prevents unused witness columns on a
        // padding row from carrying unconstrained data.
        let inactive = AB::Expr::ONE - enabled.clone();
        for value in (*row).iter().skip(1) {
            builder.when(inactive.clone()).assert_zero((*value).into());
        }

        self.input_cursor_bus.receive(
            builder,
            OneShotStreamCursorMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                next_shard_ordinal: AB::Expr::ZERO,
                tidx: local.start_tidx.into(),
            },
            enabled.clone(),
        );

        let mut tidx = AB::Expr::from(local.start_tidx);
        let mut observe = |builder: &mut AB, value: AB::Expr| {
            self.transcript_bus.observe(
                builder,
                prep.proof_index,
                tidx.clone(),
                value,
                enabled.clone(),
            );
            tidx += AB::Expr::ONE;
        };
        for &byte in ACTIVE_CHILD_COUNT_DOMAIN_V19 {
            observe(builder, AB::Expr::from_u8(byte));
        }
        observe(
            builder,
            AB::Expr::from_u32(ACTIVE_CHILD_COUNT_PROTOCOL_VERSION_V19),
        );
        observe(
            builder,
            AB::Expr::from_u32(ACTIVE_CHILD_COUNT_BINDING_VERSION_V4),
        );
        for limb in prep.batch_index_limbs {
            observe(builder, limb.into());
        }
        observe_u64_const_v19(
            &mut observe,
            builder,
            VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 as u64,
        );
        observe(builder, local.active_child_count.into());
        observe(builder, local.is_terminal_active.into());
        observe(builder, AB::Expr::from_u32(self.profile.vm_pvs_air_id));
        observe(
            builder,
            AB::Expr::from_u32(self.profile.is_valid_common_main_column),
        );
        observe_u64_const_v19(
            &mut observe,
            builder,
            self.profile.is_valid_message_block_start,
        );
        observe(builder, AB::Expr::from_u8(self.profile.log_height));
        observe_u64_const_v19(&mut observe, builder, self.profile.batch_arity);
        observe_u64_const_v19(
            &mut observe,
            builder,
            self.profile.trace_heights.len() as u64,
        );
        for &height in self.profile.trace_heights.iter() {
            observe_u64_const_v19(&mut observe, builder, height);
        }
        drop(observe);
        self.transcript_bus.observe_commit(
            builder,
            prep.proof_index,
            tidx.clone(),
            self.profile.relation_digest,
            enabled.clone(),
        );
        tidx += AB::Expr::from_usize(DIGEST_SIZE);
        self.transcript_bus.sample_ext(
            builder,
            prep.proof_index,
            tidx.clone(),
            local.sampled,
            enabled.clone(),
        );
        tidx += AB::Expr::from_usize(D_EF);

        let product = ext_mul_active_count_v19::<AB::Expr>(local.sampled, local.sampled_inverse);
        for (limb, value) in product.into_iter().enumerate() {
            builder.when(enabled.clone()).assert_eq(
                value,
                if limb == 0 {
                    AB::Expr::ONE - AB::Expr::from(local.sampled_is_zero)
                } else {
                    AB::Expr::ZERO
                },
            );
            builder
                .when(enabled.clone())
                .assert_zero(AB::Expr::from(local.sampled[limb]) * local.sampled_is_zero);
            builder.when(enabled.clone()).assert_eq(
                local.challenge[limb],
                AB::Expr::from(local.sampled[limb])
                    + AB::Expr::from_bool(limb == 0) * local.sampled_is_zero,
            );
        }
        self.transcript_bus.observe(
            builder,
            prep.proof_index,
            tidx.clone(),
            local.sampled_is_zero,
            enabled.clone(),
        );
        tidx += AB::Expr::ONE;
        builder
            .when(enabled.clone())
            .assert_eq(local.end_tidx, tidx.clone());
        self.challenge_bus.send(
            builder,
            MappedAuxiliaryChallengeMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                shard_ordinal: AB::Expr::ZERO,
                relation_digest: self.profile.relation_digest.map(Into::into),
                expected_value: local.active_child_count.into(),
                challenge: local.challenge.map(Into::into),
            },
            enabled.clone() * AB::Expr::from_u32(self.challenge_lookup_count),
        );
        self.output_cursor_bus.send(
            builder,
            OneShotStreamCursorMessageV19 {
                proof_index: prep.proof_index.into(),
                segment_index_lo: prep.segment_index_lo.into(),
                segment_index_hi: prep.segment_index_hi.into(),
                next_shard_ordinal: AB::Expr::ZERO,
                tidx,
            },
            enabled,
        );
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ActiveChildCountTranscriptRecordV4 {
    pub proof_index: u32,
    pub active_child_count: u8,
    pub start_tidx: u32,
    pub sampled: EF,
}

pub fn observe_active_child_count_transcript_v4(
    profile: &VerifierWarpActiveCountProfileV4,
    proof_index: usize,
    active_child_count: u8,
    is_terminal_active: bool,
    transcript: &mut DuplexSpongeRecorder,
) -> Result<ActiveChildCountTranscriptRecordV4, &'static str> {
    profile.validate()?;
    if proof_index >= ACTIVE_CHILD_COUNT_TRACE_HEIGHT_V4
        || active_child_count == 0
        || usize::from(active_child_count) > VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2
        || (!is_terminal_active
            && usize::from(active_child_count) != VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2)
    {
        return Err("invalid fixed-capacity active-count transcript record");
    }
    let start_tidx =
        u32::try_from(transcript.len()).map_err(|_| "active-count transcript cursor")?;
    let observe = |transcript: &mut DuplexSpongeRecorder, value: F| {
        FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(transcript, value);
    };
    let observe_u64 = |transcript: &mut DuplexSpongeRecorder, value: u64| {
        for limb in 0..4 {
            observe(
                transcript,
                F::from_u16(((value >> (16 * limb)) & 0xffff) as u16),
            );
        }
    };
    for &byte in ACTIVE_CHILD_COUNT_DOMAIN_V19 {
        observe(transcript, F::from_u8(byte));
    }
    observe(
        transcript,
        F::from_u32(ACTIVE_CHILD_COUNT_PROTOCOL_VERSION_V19),
    );
    observe(
        transcript,
        F::from_u32(ACTIVE_CHILD_COUNT_BINDING_VERSION_V4),
    );
    observe_u64(transcript, proof_index as u64);
    observe_u64(transcript, VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 as u64);
    observe(transcript, F::from_u8(active_child_count));
    observe(transcript, F::from_bool(is_terminal_active));
    observe(transcript, F::from_u32(profile.vm_pvs_air_id));
    observe(transcript, F::from_u32(profile.is_valid_common_main_column));
    observe_u64(transcript, profile.is_valid_message_block_start);
    observe(transcript, F::from_u8(profile.log_height));
    observe_u64(transcript, profile.batch_arity);
    observe_u64(transcript, profile.trace_heights.len() as u64);
    for &height in profile.trace_heights.iter() {
        observe_u64(transcript, height);
    }
    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe_commit(
        transcript,
        profile.relation_digest,
    );
    let sampled = FiatShamirTranscript::<BabyBearPoseidon2Config>::sample_ext(transcript);
    FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(
        transcript,
        F::from_bool(sampled == EF::ZERO),
    );
    let expected_end = usize::try_from(start_tidx).map_err(|_| "active-count transcript cursor")?
        + active_child_count_observation_count_v4(profile)
        + D_EF
        + 1;
    if transcript.len() != expected_end {
        return Err("fixed-capacity active-count observation schedule");
    }
    Ok(ActiveChildCountTranscriptRecordV4 {
        proof_index: u32::try_from(proof_index).map_err(|_| "active-count proof index")?,
        active_child_count,
        start_tidx,
        sampled,
    })
}

pub fn generate_active_child_count_transcript_trace_v4(
    air: &ActiveChildCountTranscriptAirV4,
    records: &[ActiveChildCountTranscriptRecordV4],
) -> Result<RowMajorMatrix<F>, &'static str> {
    air.profile.validate()?;
    if records.is_empty() || records.len() > ACTIVE_CHILD_COUNT_TRACE_HEIGHT_V4 {
        return Err("fixed-capacity active-count transcript record count");
    }
    let width = ActiveChildCountTranscriptColsV4::<F>::width();
    let mut values = F::zero_vec(width * ACTIVE_CHILD_COUNT_TRACE_HEIGHT_V4);
    for (slot, record) in records.iter().enumerate() {
        if record.proof_index as usize != slot
            || record.active_child_count == 0
            || usize::from(record.active_child_count) > VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2
            || (slot + 1 != records.len()
                && usize::from(record.active_child_count) != VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2)
        {
            return Err("fixed-capacity active-count transcript record order");
        }
        let cols: &mut ActiveChildCountTranscriptColsV4<F> =
            values[slot * width..(slot + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.is_terminal_active = F::from_bool(slot + 1 == records.len());
        cols.active_child_count_flags[usize::from(record.active_child_count) - 1] = F::ONE;
        cols.active_child_count = F::from_u8(record.active_child_count);
        cols.start_tidx = F::from_u32(record.start_tidx);
        copy_ext(&mut cols.sampled, record.sampled);
        let is_zero = record.sampled == EF::ZERO;
        cols.sampled_is_zero = F::from_bool(is_zero);
        copy_ext(
            &mut cols.sampled_inverse,
            if is_zero {
                EF::ZERO
            } else {
                record.sampled.inverse()
            },
        );
        copy_ext(
            &mut cols.challenge,
            if is_zero { EF::ONE } else { record.sampled },
        );
        cols.end_tidx = F::from_usize(
            record.start_tidx as usize
                + active_child_count_observation_count_v4(&air.profile)
                + D_EF
                + 1,
        );
    }
    Ok(RowMajorMatrix::new(values, width))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LogUpClaimDerivationRecordV19 {
    pub proof_index: u32,
    pub segment_index: u32,
    pub shard_ordinal: u16,
    pub shard_count: u16,
    pub shard_id: u32,
    pub relation_digest: [F; DIGEST_SIZE],
    pub is_program: bool,
    pub batching_challenge: EF,
    pub program_fingerprint: EF,
    pub program_mix_challenge: EF,
    pub start_tidx: u32,
}

impl LogUpClaimDerivationRecordV19 {
    #[must_use]
    pub const fn transcript_slots(&self) -> u32 {
        // tag, version, segment, shard id, relation digest, batching sample
        let ordinary = 4 + DIGEST_SIZE as u32 + D_EF as u32;
        // Program target tag, target limbs, and extension sample.
        ordinary
            + if self.is_program {
                1 + 2 * D_EF as u32
            } else {
                0
            }
    }

    #[must_use]
    pub const fn end_tidx(&self) -> u32 {
        self.start_tidx + self.transcript_slots()
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct LogUpClaimDerivationColsV19<T> {
    pub active: T,
    pub is_segment_first: T,
    pub is_segment_last: T,
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub shard_ordinal: T,
    pub shard_count: T,
    pub shard_id: T,
    pub relation_digest: [T; DIGEST_SIZE],
    pub is_program: T,
    pub batching_challenge: [T; D_EF],
    pub program_fingerprint: [T; D_EF],
    pub program_mix_challenge: [T; D_EF],
    pub start_tidx: T,
    pub end_tidx: T,
}

#[derive(Clone, ColumnsAir)]
#[columns_via(LogUpClaimDerivationColsV19<u8>)]
pub struct LogUpClaimDerivationAirV19 {
    pub segment_start: u32,
    pub transcript_bus: TranscriptBus,
    pub phase_start_bus: LogUpClaimPhaseStartBusV19,
    pub challenge_bus: MappedFunctionalChallengeBusV19,
    pub stream_cursor_bus: OneShotStreamCursorBusV19,
}

impl BaseAir<F> for LogUpClaimDerivationAirV19 {
    fn width(&self) -> usize {
        LogUpClaimDerivationColsV19::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for LogUpClaimDerivationAirV19 {}
impl PartitionedBaseAir<F> for LogUpClaimDerivationAirV19 {}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for LogUpClaimDerivationAirV19 {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("LogUp claim row");
        let next_row = main.row_slice(1).expect("LogUp claim next row");
        let local: &LogUpClaimDerivationColsV19<AB::Var> = (*local_row).borrow();
        let next: &LogUpClaimDerivationColsV19<AB::Var> = (*next_row).borrow();

        for bit in [
            local.active,
            local.is_segment_first,
            local.is_segment_last,
            local.is_program,
        ] {
            builder.assert_bool(bit);
        }
        builder.when_first_row().assert_one(local.active);
        builder
            .when(local.active * local.is_segment_first)
            .assert_zero(local.shard_ordinal);
        builder.when(local.active).assert_eq(
            AB::Expr::from(local.proof_index) + AB::Expr::from_u32(self.segment_start),
            AB::Expr::from(local.segment_index_lo)
                + AB::Expr::from_u32(1 << 16) * AB::Expr::from(local.segment_index_hi),
        );
        builder
            .when(local.active * local.is_segment_last)
            .assert_eq(
                AB::Expr::from(local.shard_ordinal) + AB::Expr::ONE,
                local.shard_count,
            );

        let continuing =
            AB::Expr::from(next.active) * (AB::Expr::ONE - AB::Expr::from(local.is_segment_last));
        let mut transition_builder = builder.when_transition();
        let mut transition = transition_builder.when(continuing);
        transition.assert_zero(next.is_segment_first);
        transition.assert_eq(next.proof_index, local.proof_index);
        transition.assert_eq(next.segment_index_lo, local.segment_index_lo);
        transition.assert_eq(next.segment_index_hi, local.segment_index_hi);
        transition.assert_eq(next.shard_count, local.shard_count);
        transition.assert_eq(
            next.shard_ordinal,
            AB::Expr::from(local.shard_ordinal) + AB::Expr::ONE,
        );
        transition.assert_eq(next.start_tidx, local.end_tidx);

        let enabled = AB::Expr::from(local.active);
        let first = enabled.clone() * AB::Expr::from(local.is_segment_first);
        self.phase_start_bus.receive(
            builder,
            LogUpClaimPhaseStartMessageV19 {
                proof_index: local.proof_index.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                first_claim_tidx: local.start_tidx.into(),
            },
            first,
        );

        let mut tidx = AB::Expr::from(local.start_tidx);
        for value in [
            AB::Expr::from_u64(DIRECT_LOGUP_CLAIM_TAG_V19),
            AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            AB::Expr::from(local.segment_index_lo)
                + AB::Expr::from_u32(1 << 16) * AB::Expr::from(local.segment_index_hi),
            AB::Expr::from(local.shard_id),
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
        self.transcript_bus.observe_commit(
            builder,
            local.proof_index,
            tidx.clone(),
            local.relation_digest,
            enabled.clone(),
        );
        tidx += AB::Expr::from_usize(DIGEST_SIZE);
        self.transcript_bus.sample_ext(
            builder,
            local.proof_index,
            tidx.clone(),
            local.batching_challenge,
            enabled.clone(),
        );
        tidx += AB::Expr::from_usize(D_EF);

        let program = enabled.clone() * AB::Expr::from(local.is_program);
        self.transcript_bus.observe(
            builder,
            local.proof_index,
            tidx.clone(),
            AB::Expr::from_u64(PROGRAM_FINGERPRINT_TARGET_TAG_V19),
            program.clone(),
        );
        tidx += AB::Expr::from(local.is_program);
        self.transcript_bus.observe_ext(
            builder,
            local.proof_index,
            tidx.clone(),
            local.program_fingerprint,
            program.clone(),
        );
        tidx += AB::Expr::from(local.is_program) * AB::Expr::from_usize(D_EF);
        self.transcript_bus.sample_ext(
            builder,
            local.proof_index,
            tidx.clone(),
            local.program_mix_challenge,
            program,
        );
        tidx += AB::Expr::from(local.is_program) * AB::Expr::from_usize(D_EF);
        builder
            .when(enabled.clone())
            .assert_eq(local.end_tidx, tidx);
        for value in local
            .program_fingerprint
            .iter()
            .chain(local.program_mix_challenge.iter())
        {
            builder
                .when(enabled.clone() * (AB::Expr::ONE - AB::Expr::from(local.is_program)))
                .assert_zero(*value);
        }

        self.challenge_bus.send(
            builder,
            MappedFunctionalChallengeMessageV19 {
                proof_index: local.proof_index.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                shard_ordinal: local.shard_ordinal.into(),
                shard_id: local.shard_id.into(),
                relation_digest: local.relation_digest.map(Into::into),
                is_program: local.is_program.into(),
                batching_challenge: local.batching_challenge.map(Into::into),
                program_fingerprint: local.program_fingerprint.map(Into::into),
                program_mix_challenge: local.program_mix_challenge.map(Into::into),
            },
            enabled.clone(),
        );
        self.stream_cursor_bus.send(
            builder,
            OneShotStreamCursorMessageV19 {
                proof_index: local.proof_index.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                next_shard_ordinal: AB::Expr::ZERO,
                tidx: local.end_tidx.into(),
            },
            enabled * AB::Expr::from(local.is_segment_last),
        );
    }
}

pub fn generate_logup_claim_derivation_trace_v19(
    records: &[LogUpClaimDerivationRecordV19],
) -> Result<RowMajorMatrix<F>, &'static str> {
    if records.is_empty() {
        return Err("empty LogUp claim derivation batch");
    }
    for (index, record) in records.iter().enumerate() {
        if record.shard_count == 0
            || record.shard_ordinal >= record.shard_count
            || (!record.is_program
                && (record.program_fingerprint != EF::ZERO
                    || record.program_mix_challenge != EF::ZERO))
        {
            return Err("invalid LogUp claim derivation record");
        }
        if let Some(previous) = index.checked_sub(1).map(|i| &records[i]) {
            let same_segment = previous.segment_index == record.segment_index;
            if same_segment
                && (record.shard_ordinal != previous.shard_ordinal + 1
                    || record.start_tidx != previous.end_tidx())
            {
                return Err("noncanonical LogUp claim order");
            }
            if !same_segment
                && (record.shard_ordinal != 0 || previous.shard_ordinal + 1 != previous.shard_count)
            {
                return Err("incomplete LogUp claim segment");
            }
        } else if record.shard_ordinal != 0 {
            return Err("LogUp claim batch must start at ordinal zero");
        }
    }
    if records.last().unwrap().shard_ordinal + 1 != records.last().unwrap().shard_count {
        return Err("incomplete final LogUp claim segment");
    }

    let width = LogUpClaimDerivationColsV19::<F>::width();
    let height = records.len().next_power_of_two().max(2);
    let mut values = F::zero_vec(height * width);
    for (row, record) in records.iter().enumerate() {
        let cols: &mut LogUpClaimDerivationColsV19<F> =
            values[row * width..(row + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.is_segment_first = F::from_bool(record.shard_ordinal == 0);
        cols.is_segment_last = F::from_bool(record.shard_ordinal + 1 == record.shard_count);
        cols.proof_index = F::from_u32(record.proof_index);
        cols.segment_index_lo = F::from_u32(record.segment_index & 0xffff);
        cols.segment_index_hi = F::from_u32(record.segment_index >> 16);
        cols.shard_ordinal = F::from_u16(record.shard_ordinal);
        cols.shard_count = F::from_u16(record.shard_count);
        cols.shard_id = F::from_u32(record.shard_id);
        cols.relation_digest = record.relation_digest;
        cols.is_program = F::from_bool(record.is_program);
        copy_ext(&mut cols.batching_challenge, record.batching_challenge);
        copy_ext(&mut cols.program_fingerprint, record.program_fingerprint);
        copy_ext(
            &mut cols.program_mix_challenge,
            record.program_mix_challenge,
        );
        cols.start_tidx = F::from_u32(record.start_tidx);
        cols.end_tidx = F::from_u32(record.end_tidx());
    }
    Ok(RowMajorMatrix::new(values, width))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OneShotClaimTranscriptRecordV19 {
    pub proof_index: u32,
    pub segment_index: u32,
    pub shard_ordinal: u16,
    pub start_tidx: u32,
    pub observations: Vec<EF>,
}

impl OneShotClaimTranscriptRecordV19 {
    pub fn from_claim(
        proof_index: u32,
        segment_index: u32,
        shard_ordinal: u16,
        start_tidx: u32,
        claim: &TerminalStructuredLinearClaim<EF>,
    ) -> Result<Self, &'static str> {
        let observations = direct_message_opening_reduction_claim_observations(claim)
            .map_err(|_| "invalid one-shot claim")?;
        if observations.is_empty() {
            return Err("invalid one-shot transcript record");
        }
        Ok(Self {
            proof_index,
            segment_index,
            shard_ordinal,
            start_tidx,
            observations,
        })
    }

    #[must_use]
    pub fn round_start_tidx(&self) -> u32 {
        self.start_tidx + (self.observations.len() * D_EF) as u32
    }
}

/// Test-only unfused oracle retained for differential transcript checks. The
/// production v19 composition emits these observations directly from the
/// mapped-functional AIR and cannot instantiate this relay table.
#[cfg(test)]
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct OneShotClaimTranscriptColsV19<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub shard_ordinal: T,
    pub observation_index: T,
    pub observation_count: T,
    pub tidx: T,
    pub value: [T; D_EF],
}

#[cfg(test)]
#[derive(Clone, ColumnsAir)]
#[columns_via(OneShotClaimTranscriptColsV19<u8>)]
pub struct OneShotClaimTranscriptAirV19 {
    pub segment_start: u32,
    pub transcript_bus: TranscriptBus,
    pub observation_bus: VerifiedOneShotClaimObservationBusV19,
    pub round_start_bus: OneShotRoundStartBusV19,
    pub stream_cursor_bus: OneShotStreamCursorBusV19,
}

#[cfg(test)]
impl BaseAir<F> for OneShotClaimTranscriptAirV19 {
    fn width(&self) -> usize {
        OneShotClaimTranscriptColsV19::<F>::width()
    }
}
#[cfg(test)]
impl BaseAirWithPublicValues<F> for OneShotClaimTranscriptAirV19 {}
#[cfg(test)]
impl PartitionedBaseAir<F> for OneShotClaimTranscriptAirV19 {}

#[cfg(test)]
impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for OneShotClaimTranscriptAirV19 {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("one-shot claim transcript row");
        let next_row = main
            .row_slice(1)
            .expect("one-shot claim transcript next row");
        let local: &OneShotClaimTranscriptColsV19<AB::Var> = (*local_row).borrow();
        let next: &OneShotClaimTranscriptColsV19<AB::Var> = (*next_row).borrow();
        for bit in [local.active, local.is_first, local.is_last] {
            builder.assert_bool(bit);
        }
        builder.when_first_row().assert_one(local.active);
        builder
            .when(local.active * local.is_first)
            .assert_zero(local.observation_index);
        builder.when(local.active * local.is_last).assert_eq(
            AB::Expr::from(local.observation_index) + AB::Expr::ONE,
            local.observation_count,
        );
        builder.when(local.active).assert_eq(
            AB::Expr::from(local.proof_index) + AB::Expr::from_u32(self.segment_start),
            AB::Expr::from(local.segment_index_lo)
                + AB::Expr::from_u32(1 << 16) * AB::Expr::from(local.segment_index_hi),
        );

        let continuing =
            AB::Expr::from(next.active) * (AB::Expr::ONE - AB::Expr::from(local.is_last));
        let mut transition_builder = builder.when_transition();
        let mut transition = transition_builder.when(continuing);
        transition.assert_zero(next.is_first);
        transition.assert_eq(next.proof_index, local.proof_index);
        transition.assert_eq(next.segment_index_lo, local.segment_index_lo);
        transition.assert_eq(next.segment_index_hi, local.segment_index_hi);
        transition.assert_eq(next.shard_ordinal, local.shard_ordinal);
        transition.assert_eq(next.observation_count, local.observation_count);
        transition.assert_eq(
            next.observation_index,
            AB::Expr::from(local.observation_index) + AB::Expr::ONE,
        );
        transition.assert_eq(
            next.tidx,
            AB::Expr::from(local.tidx) + AB::Expr::from_usize(D_EF),
        );

        let enabled = AB::Expr::from(local.active);
        let first = enabled.clone() * AB::Expr::from(local.is_first);
        let last = enabled.clone() * AB::Expr::from(local.is_last);
        self.stream_cursor_bus.receive(
            builder,
            OneShotStreamCursorMessageV19 {
                proof_index: local.proof_index.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                next_shard_ordinal: local.shard_ordinal.into(),
                tidx: local.tidx.into(),
            },
            first,
        );
        self.observation_bus.receive(
            builder,
            VerifiedOneShotClaimObservationMessageV19 {
                proof_index: local.proof_index.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                shard_ordinal: local.shard_ordinal.into(),
                observation_index: local.observation_index.into(),
                observation_count: local.observation_count.into(),
                value: local.value.map(Into::into),
            },
            enabled.clone(),
        );
        self.transcript_bus.observe_ext(
            builder,
            local.proof_index,
            local.tidx,
            local.value,
            enabled,
        );
        self.round_start_bus.send(
            builder,
            OneShotRoundStartMessageV19 {
                proof_index: local.proof_index.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                shard_ordinal: local.shard_ordinal.into(),
                round_start_tidx: AB::Expr::from(local.tidx) + AB::Expr::from_usize(D_EF),
            },
            last,
        );
    }
}

#[cfg(test)]
pub fn generate_one_shot_claim_transcript_trace_v19(
    records: &[OneShotClaimTranscriptRecordV19],
) -> Result<RowMajorMatrix<F>, &'static str> {
    if records.is_empty() || records.iter().any(|record| record.observations.is_empty()) {
        return Err("empty one-shot claim transcript batch");
    }
    let rows = records
        .iter()
        .map(|record| record.observations.len())
        .sum::<usize>();
    let width = OneShotClaimTranscriptColsV19::<F>::width();
    let height = rows.next_power_of_two().max(2);
    let mut values = F::zero_vec(height * width);
    let mut row = 0;
    for record in records {
        for (index, observation) in record.observations.iter().copied().enumerate() {
            let cols: &mut OneShotClaimTranscriptColsV19<F> =
                values[row * width..(row + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.is_first = F::from_bool(index == 0);
            cols.is_last = F::from_bool(index + 1 == record.observations.len());
            cols.proof_index = F::from_u32(record.proof_index);
            cols.segment_index_lo = F::from_u32(record.segment_index & 0xffff);
            cols.segment_index_hi = F::from_u32(record.segment_index >> 16);
            cols.shard_ordinal = F::from_u16(record.shard_ordinal);
            cols.observation_index = F::from_usize(index);
            cols.observation_count = F::from_usize(record.observations.len());
            cols.tidx = F::from_u32(record.start_tidx + (index * D_EF) as u32);
            copy_ext(&mut cols.value, observation);
            row += 1;
        }
    }
    Ok(RowMajorMatrix::new(values, width))
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;

    use openvm_stark_backend::{
        air_builders::debug::check_constraints,
        native_warp::{
            bind_direct_message_opening_reduction_claim,
            direct_message_opening_reduction_claim_observations,
        },
        warp_pesat::{
            AlgebraicChallenger, PrismalinearMappedColumnBlock, PrismalinearMappedColumnRotation,
            PrismalinearMappedColumnTerm, PrismalinearMappedColumnWeight,
            TerminalStructuredLinearClaim, TerminalWeightSpec,
        },
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

    use super::*;

    #[derive(Default)]
    struct RecordingChallenger {
        observations: Vec<EF>,
    }

    impl AlgebraicChallenger<EF> for RecordingChallenger {
        fn observe(&mut self, value: EF) {
            self.observations.push(value);
        }

        fn sample(&mut self) -> EF {
            EF::ONE
        }
    }

    fn claim() -> TerminalStructuredLinearClaim<EF> {
        TerminalStructuredLinearClaim::new(
            TerminalWeightSpec::PrismalinearMappedColumns(PrismalinearMappedColumnWeight {
                log_message_len: 3,
                terms: vec![PrismalinearMappedColumnTerm {
                    block: PrismalinearMappedColumnBlock {
                        start: 0,
                        log_height: 2,
                    },
                    l_skip: 1,
                    barycentric_weights: vec![EF::from_u32(3), EF::from_u32(5)],
                    folded_row_eq_point: vec![EF::from_u32(7)],
                    rotation: PrismalinearMappedColumnRotation::Next,
                    scale: EF::from_u32(11),
                }],
            }),
            EF::from_u32(13),
        )
    }

    #[test]
    fn claim_observation_rows_are_byte_for_byte_backend_parity() {
        let claim = claim();
        let expected = direct_message_opening_reduction_claim_observations(&claim).unwrap();
        let mut challenger = RecordingChallenger::default();
        bind_direct_message_opening_reduction_claim(&mut challenger, &claim).unwrap();
        assert_eq!(challenger.observations, expected);
        let record = OneShotClaimTranscriptRecordV19::from_claim(4, 4, 2, 91, &claim).unwrap();
        assert_eq!(record.observations, expected);
        assert_eq!(
            record.round_start_tidx(),
            91 + (expected.len() * D_EF) as u32
        );
        let trace = generate_one_shot_claim_transcript_trace_v19(&[record]).unwrap();
        for (index, expected) in expected.iter().enumerate() {
            let row = trace.row_slice(index).unwrap();
            let row: &OneShotClaimTranscriptColsV19<F> = (*row).borrow();
            assert_eq!(row.tidx, F::from_u32(91 + (index * D_EF) as u32));
            assert_eq!(&row.value, expected.as_basis_coefficients_slice());
        }
    }

    #[test]
    fn claim_derivation_schedule_is_segment_ordered_and_program_only_mix_is_enforced() {
        let records = vec![
            LogUpClaimDerivationRecordV19 {
                proof_index: 7,
                segment_index: 7,
                shard_ordinal: 0,
                shard_count: 2,
                shard_id: 3,
                relation_digest: [F::ONE; DIGEST_SIZE],
                is_program: true,
                batching_challenge: EF::from_u32(17),
                program_fingerprint: EF::from_u32(19),
                program_mix_challenge: EF::from_u32(23),
                start_tidx: 40,
            },
            LogUpClaimDerivationRecordV19 {
                proof_index: 7,
                segment_index: 7,
                shard_ordinal: 1,
                shard_count: 2,
                shard_id: 8,
                relation_digest: [F::TWO; DIGEST_SIZE],
                is_program: false,
                batching_challenge: EF::from_u32(29),
                program_fingerprint: EF::ZERO,
                program_mix_challenge: EF::ZERO,
                start_tidx: 40 + (4 + DIGEST_SIZE + D_EF + 1 + 2 * D_EF) as u32,
            },
        ];
        let trace = generate_logup_claim_derivation_trace_v19(&records).unwrap();
        let first_row = trace.row_slice(0).unwrap();
        let first: &LogUpClaimDerivationColsV19<F> = (*first_row).borrow();
        let second_row = trace.row_slice(1).unwrap();
        let second: &LogUpClaimDerivationColsV19<F> = (*second_row).borrow();
        assert_eq!(first.end_tidx, second.start_tidx);

        let mut bad = records;
        bad[1].program_mix_challenge = EF::ONE;
        assert!(generate_logup_claim_derivation_trace_v19(&bad).is_err());
    }

    fn active_count_profile_v4() -> VerifierWarpActiveCountProfileV4 {
        VerifierWarpActiveCountProfileV4 {
            relation_digest: [F::from_u32(701); DIGEST_SIZE],
            profile_digest: [F::from_u32(702); DIGEST_SIZE],
            batch_arity: VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 as u64,
            trace_heights: std::sync::Arc::from([8, 16, 32]),
            vm_pvs_air_id: 3,
            is_valid_common_main_column: 1,
            is_valid_message_block_start: 16,
            log_height: 2,
            log_message_len: 6,
        }
    }

    fn active_count_air_v4() -> ActiveChildCountTranscriptAirV4 {
        ActiveChildCountTranscriptAirV4 {
            profile: active_count_profile_v4(),
            transcript_bus: TranscriptBus::new(700),
            input_cursor_bus: OneShotStreamCursorBusV19::new(701),
            output_cursor_bus: OneShotStreamCursorBusV19::new(702),
            challenge_bus: MappedAuxiliaryChallengeBusV19::new(703),
            challenge_lookup_count: 2,
        }
    }

    fn active_count_records_v4(
        occupancy: usize,
        terminal_active_child_count: u8,
    ) -> Vec<ActiveChildCountTranscriptRecordV4> {
        (0..occupancy)
            .map(|slot| ActiveChildCountTranscriptRecordV4 {
                proof_index: slot as u32,
                active_child_count: if slot + 1 == occupancy {
                    terminal_active_child_count
                } else {
                    VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2 as u8
                },
                start_tidx: 1000 + slot as u32 * 100,
                sampled: EF::from_u32(710 + slot as u32),
            })
            .collect()
    }

    fn check_active_count_v4(air: &ActiveChildCountTranscriptAirV4, trace: &RowMajorMatrix<F>) {
        let prep = air.preprocessed_trace().unwrap();
        check_constraints::<_, BabyBearPoseidon2Config>(
            air,
            "ActiveChildCountTranscriptAirV4",
            &Some(prep.as_view()),
            &[trace.as_view()],
            &[],
        );
    }

    #[test]
    fn fixed_capacity_active_count_transcript_has_runtime_prefix_and_stable_key_shape() {
        let air = active_count_air_v4();
        let prep = air.preprocessed_trace().unwrap();
        for occupancy in 1..=ACTIVE_CHILD_COUNT_TRACE_HEIGHT_V4 {
            let trace = generate_active_child_count_transcript_trace_v4(
                &air,
                &active_count_records_v4(occupancy, 2),
            )
            .unwrap();
            assert_eq!(trace.height(), ACTIVE_CHILD_COUNT_TRACE_HEIGHT_V4);
            assert_eq!(air.preprocessed_trace().unwrap().values, prep.values);
            check_active_count_v4(&air, &trace);
        }
        let distinct =
            generate_active_child_count_transcript_trace_v4(&air, &active_count_records_v4(3, 2))
                .unwrap();
        check_active_count_v4(&air, &distinct);
    }

    #[test]
    fn fixed_capacity_active_count_terminal_and_inactive_mutations_reject() {
        let air = active_count_air_v4();
        let honest =
            generate_active_child_count_transcript_trace_v4(&air, &active_count_records_v4(3, 2))
                .unwrap();
        check_active_count_v4(&air, &honest);
        let width = ActiveChildCountTranscriptColsV4::<F>::width();
        let rejects = |mutate: &dyn Fn(&mut RowMajorMatrix<F>)| {
            let mut changed = honest.clone();
            mutate(&mut changed);
            assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
                check_active_count_v4(&air, &changed)
            }))
            .is_err());
        };
        rejects(&|trace| {
            let row: &mut ActiveChildCountTranscriptColsV4<F> =
                trace.values[2 * width..3 * width].borrow_mut();
            row.is_terminal_active = F::ZERO;
        });
        rejects(&|trace| {
            let row: &mut ActiveChildCountTranscriptColsV4<F> =
                trace.values[3 * width..4 * width].borrow_mut();
            row.challenge[0] = F::ONE;
        });
        rejects(&|trace| {
            let row: &mut ActiveChildCountTranscriptColsV4<F> =
                trace.values[width..2 * width].borrow_mut();
            row.active_child_count_flags = [F::ZERO; VERIFIER_WARP_ACTIVE_CHILD_CAPACITY_V2];
            row.active_child_count_flags[0] = F::ONE;
            row.active_child_count = F::ONE;
        });
    }
}

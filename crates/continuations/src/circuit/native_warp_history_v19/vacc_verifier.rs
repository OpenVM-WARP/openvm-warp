//! Protocol-v19 verifier AIR composition for one ordinary direct-AIR WARP
//! VACC step.
//!
//! This module consumes the verifier-derived [`WarpVaccStepVerification`] and
//! exact [`TranscriptLog`].  It authenticates fresh and prior codeword
//! openings, certifies both sumchecks and their Fiat--Shamir challenges, and
//! publishes the exact typed statements consumed by
//! [`WarpReplayProducerAirV19`].  The direct AIR relation itself is deliberately
//! absent: terminal Decide evaluates the fixed homogeneous PESAT index.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit::{
    bus::{
        CertifiedTranscriptCheckpointBus, CertifiedTranscriptCheckpointMessage,
        ResumeTranscriptStateBus, ResumeTranscriptStateMessage, TranscriptBusMessage,
    },
    define_typed_lookup_bus,
    native_warp::*,
    primitives::{
        bus::{ExpBitsLenMessage, RightShiftMessage},
        exp_bits_len::{ExpBitsLenAir, ExpBitsLenCols, ExpBitsLenCpuTraceGenerator},
    },
    system::BusInventory,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    native_warp::NativeWarpFamilyParams,
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    warp_accum::{
        twin_degree, BinaryMerkleMultiproofRecord, FiniteStackedFreshBatchOpeningProof,
        FiniteStackedFreshCommitment, FiniteStackedFreshSourceOpeningVerification,
        MerkleBatchOpeningProof, MerkleBatchOpeningVerification, NativeSumcheckKind,
        NativeTranscriptPhase, NativeTranscriptPhaseSpan, ReducedWarpVaccStepProof,
        StackedRsBatchOpeningProof, StackedRsBatchOpeningVerification, StackedRsFreshCommitment,
        WarpStepInputKind, WarpVaccStepProof, WarpVaccStepVerification,
        EXACT_FINITE_WARP_CALL_END_TAG, EXACT_FINITE_WARP_TRANSCRIPT_VERSION,
        NATIVE_WARP_VACC_APPENDIX_D_PROTOCOL_TAG, NATIVE_WARP_VACC_PROTOCOL_TAG,
    },
    warp_pesat::{AccumulatorInstance, FreshPesatClaim},
    AirRef, AnyAir, StarkProtocolConfig, SystemParams, TranscriptCheckpoint, TranscriptEvent,
    TranscriptEventKind, TranscriptLog,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as NativeSC, Digest, CHUNK, DIGEST_SIZE, D_EF, EF, F,
};

// Re-export the fixed-capacity route through this wildcard-exported module;
// `producer` itself has an explicit legacy export list in `mod.rs`.
pub use super::producer::{
    generate_fixed_hleaf_warp_replay_producer_trace_v4, generate_fixed_hleaf_warp_replay_trace_v4,
    FixedHLeafWarpReplayAirV4, FixedHLeafWarpReplayColsV4, FixedHLeafWarpReplayErrorV4,
    FixedHLeafWarpReplayPrepColsV4, FixedHLeafWarpReplayProducerAirV4,
};
use super::{
    CertifiedDirectAirVaccInputBusV19, CertifiedDirectAirVaccInputMessageV19,
    DirectAirVaccContextBusV19, DirectAirVaccContextMessageV19, DirectAirVaccProducerRecordV19,
    TranscriptCheckpointRecordV19, WarpReplayProducerColsV19, MAX_FRESH_BETA_LEN_V19,
    MAX_RAW_MESSAGE_POINT_LEN_V19, NATIVE_WARP_HISTORY_PROTOCOL_V19,
};
use crate::circuit::{
    finite_warp_v3::{
        FiniteWarpV3ExactVaccAuthorityBus, FiniteWarpV3ExactVaccAuthorityMessage,
        FreshInstanceLinkConsumerRequirements,
    },
    native_warp_accumulator::{
        generate_native_accumulator_digest_traces, NativeAccumulatorBindingMode,
        NativeAccumulatorDigestTraces, NativeAccumulatorHashAir, NativeAccumulatorHashCols,
        NativeAccumulatorRootDigestCols, NativeAccumulatorValueCols,
        NativePrivateAccumulatorLayout,
    },
};

pub type DirectAirVaccVerificationV19 = WarpVaccStepVerification<
    EF,
    Digest,
    MerkleBatchOpeningVerification<EF, Digest>,
    MerkleBatchOpeningVerification<EF, Digest>,
>;

pub type DirectAirAppendixDVaccVerificationV19 = WarpVaccStepVerification<
    EF,
    Digest,
    NativeAppendixDFreshOpeningVerification,
    MerkleBatchOpeningVerification<EF, Digest>,
>;

/// One exact-finite invocation proof as emitted by the production stacked
/// BabyBear source lane. The recursive verifier never accepts a reduced replay
/// object: it consumes the original claims and both authenticated opening
/// proofs used by ordinary WARP `Verify`.
pub type DirectAirExactFiniteVaccProofV19 = WarpVaccStepProof<
    EF,
    Digest,
    FiniteStackedFreshBatchOpeningProof<F, Digest>,
    MerkleBatchOpeningProof<EF, Digest>,
    FiniteStackedFreshCommitment<Digest>,
>;

pub type DirectAirExactFiniteVaccVerificationV19 = WarpVaccStepVerification<
    EF,
    Digest,
    FiniteStackedFreshSourceOpeningVerification<EF, Digest>,
    MerkleBatchOpeningVerification<EF, Digest>,
>;

pub type DirectAirReducedSwirlVaccProofV19 = ReducedWarpVaccStepProof<
    EF,
    Digest,
    StackedRsBatchOpeningProof<F, Digest>,
    MerkleBatchOpeningProof<EF, Digest>,
    StackedRsFreshCommitment<EF, Digest>,
>;

pub type DirectAirReducedSwirlVaccVerificationV19 = WarpVaccStepVerification<
    EF,
    Digest,
    StackedRsBatchOpeningVerification<F, EF, Digest>,
    MerkleBatchOpeningVerification<EF, Digest>,
>;

/// One genuine reduced native transition qualified by its authoritative
/// transcript interval. `proof_idx` is the canonical chain step; callers do
/// not provide any verifier-success bit.
pub struct DirectAirReducedSwirlVaccVerifierRecordV19<'a> {
    /// Canonical block-wide WARP call index.  This value is serialized in the
    /// native transcript and therefore must never be renumbered by a
    /// recursive wrapper.
    pub proof_idx: usize,
    /// Dense transcript/AIR namespace inside the current bounded wrapper
    /// leaf.  Whole-block batches set this equal to `proof_idx`; recursive
    /// transition leaves restart it at zero without changing any transcript
    /// value.
    pub local_proof_idx: usize,
    /// Whether this is the final call of the complete block schedule.  Only
    /// that call owns the manifest-footer and terminal-Decide transcript
    /// suffix; the final call of an intermediate physical leaf does not.
    pub is_final_call: bool,
    pub proof: &'a DirectAirReducedSwirlVaccProofV19,
    pub verification: &'a DirectAirReducedSwirlVaccVerificationV19,
    pub transcript: &'a TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    pub prior: Option<&'a AccumulatorInstance<EF, Digest>>,
    pub batch_start_tidx: usize,
    pub vacc_start_tidx: usize,
    pub vacc_end_tidx: usize,
    /// First transcript operation of every canonical source commitment
    /// descriptor, in source order.
    pub commitment_tidxs: Vec<usize>,
}

/// Borrowed native-verifier output for one setup-fixed exact-finite call.
/// `start_checkpoint` is immediately before the call's Appendix-D prefix;
/// call zero therefore follows the separately constrained schedule prefix.
pub struct DirectAirExactFiniteVaccVerifierRecordV19<'a> {
    pub proof: &'a DirectAirExactFiniteVaccProofV19,
    pub verification: &'a DirectAirExactFiniteVaccVerificationV19,
    pub transcript: &'a TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    pub prior: Option<&'a AccumulatorInstance<EF, Digest>>,
    pub start_checkpoint: TranscriptCheckpointRecordV19,
    pub end_checkpoint: TranscriptCheckpointRecordV19,
}

impl<'a> DirectAirExactFiniteVaccVerifierRecordV19<'a> {
    /// Construct the only accepted exact-finite verifier record from native
    /// proof material and its authenticated transcript. Transcript cursors,
    /// sample counts, and sponge states are derived here; callers cannot
    /// supply checkpoint metadata independently.
    pub fn from_authenticated_transcript(
        module: &DirectAirVaccVerifierModuleV19,
        proof: &'a DirectAirExactFiniteVaccProofV19,
        verification: &'a DirectAirExactFiniteVaccVerificationV19,
        transcript: &'a TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
        prior: Option<&'a AccumulatorInstance<EF, Digest>>,
    ) -> Result<Self, DirectAirVaccVerifierErrorV19> {
        module.exact_finite_record_from_authenticated_transcript(
            proof,
            verification,
            transcript,
            prior,
        )
    }
}

/// Setup-fixed LogUp multiplicities for composing ordinary exact WARP Verify
/// with the finite-v3 fresh-instance linker. Producer counts are totals; the
/// fresh-slot count is the additional per-fresh-source lookup introduced by
/// the linker on top of the ordinary claim/projection consumers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DirectAirExactFiniteVaccMultiplicityProfileV19 {
    pub fresh_claim_consumer_count: usize,
    pub exact_schedule_lookup_count: usize,
    pub exact_call_protocol_lookup_count_per_call: usize,
    pub fresh_root_lookup_count_per_call: usize,
    pub extra_fresh_slot_lookup_count_per_source: usize,
}

impl DirectAirExactFiniteVaccMultiplicityProfileV19 {
    pub const MAX_COUNT: usize = 16;

    pub fn from_fresh_instance_link_requirements(
        requirements: FreshInstanceLinkConsumerRequirements,
    ) -> Result<Self, DirectAirVaccVerifierErrorV19> {
        let profile = Self {
            fresh_claim_consumer_count: requirements.fresh_claim_consumer_count_with_exact_vacc(),
            exact_schedule_lookup_count: requirements.exact_schedule_producer_count(),
            exact_call_protocol_lookup_count_per_call: requirements
                .exact_call_protocol_producer_count_per_call(),
            fresh_root_lookup_count_per_call: requirements.fresh_root_producer_count_per_call(),
            extra_fresh_slot_lookup_count_per_source: requirements
                .extra_fresh_slot_lookups_per_source,
        };
        profile.validate()?;
        Ok(profile)
    }

    pub fn validate(self) -> Result<(), DirectAirVaccVerifierErrorV19> {
        for count in [
            self.fresh_claim_consumer_count,
            self.exact_schedule_lookup_count,
            self.exact_call_protocol_lookup_count_per_call,
            self.fresh_root_lookup_count_per_call,
            self.extra_fresh_slot_lookup_count_per_source,
        ] {
            if count == 0 || count > Self::MAX_COUNT {
                return Err(DirectAirVaccVerifierErrorV19::InvalidProfile);
            }
        }
        Ok(())
    }
}

/// Verifier-key mode for one call of the exact finite schedule. All values are
/// fixed before key generation. In particular, `call_index` is not selected
/// by a witness and determines both arity and whether a prior exists.
#[derive(Clone, Debug)]
pub struct DirectAirExactFiniteVaccConfigV19 {
    pub transcript_profile: NativeExactFiniteVaccTranscriptProfile,
    pub call_index: usize,
    pub protocol_digest: Digest,
    pub multiplicities: DirectAirExactFiniteVaccMultiplicityProfileV19,
    pub schedule_bus: NativeExactFiniteVaccScheduleBus,
    pub call_protocol_bus: NativeExactFiniteVaccCallProtocolBus,
    pub authority_bus: FiniteWarpV3ExactVaccAuthorityBus,
}

impl DirectAirExactFiniteVaccConfigV19 {
    fn validate(&self) -> Result<(), DirectAirVaccVerifierErrorV19> {
        self.transcript_profile
            .validate()
            .map_err(|_| DirectAirVaccVerifierErrorV19::InvalidProfile)?;
        self.multiplicities.validate()?;
        if self.call_index >= self.transcript_profile.active_call_count()
            || self.protocol_digest.iter().all(|value| *value == F::ZERO)
        {
            return Err(DirectAirVaccVerifierErrorV19::InvalidProfile);
        }
        Ok(())
    }

    fn call(&self) -> NativeExactFiniteVaccCallProfile {
        self.transcript_profile.calls[self.call_index]
    }

    fn source_start(&self) -> usize {
        self.transcript_profile.calls[..self.call_index]
            .iter()
            .map(|call| call.fresh_count)
            .sum()
    }
}

/// Fresh-opening authentication understood by the direct-AIR VACC History
/// verifier.  The CUDA shared-forest path deliberately returns `()` here:
/// its rows and paths are authenticated by the dedicated forest AIR, never by
/// synthesizing an ordinary scalar Merkle verification record.
pub trait DirectAirFreshAuthenticationV19 {
    fn scalar_merkle(&self) -> Option<&MerkleBatchOpeningVerification<EF, Digest>>;

    fn appendix_d_base(&self) -> Option<&NativeAppendixDFreshOpeningVerification> {
        None
    }
}

impl DirectAirFreshAuthenticationV19 for MerkleBatchOpeningVerification<EF, Digest> {
    fn scalar_merkle(&self) -> Option<&MerkleBatchOpeningVerification<EF, Digest>> {
        Some(self)
    }
}

impl DirectAirFreshAuthenticationV19 for () {
    fn scalar_merkle(&self) -> Option<&MerkleBatchOpeningVerification<EF, Digest>> {
        None
    }
}

impl DirectAirFreshAuthenticationV19 for NativeAppendixDFreshOpeningVerification {
    fn scalar_merkle(&self) -> Option<&MerkleBatchOpeningVerification<EF, Digest>> {
        None
    }

    fn appendix_d_base(&self) -> Option<&NativeAppendixDFreshOpeningVerification> {
        Some(self)
    }
}

/// Verifier-key choice for the fresh commitment lane.  Both modes execute the
/// same standard WARP algebra after the commitment phase; only authentication
/// of source zero-shift queries differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectAirVaccFreshCommitmentModeV19 {
    ScalarMerkle,
    AppendixDBase,
    /// Construction-10.4 base-field columns under one invocation-wide
    /// stacked Merkle tree. This is the production exact-finite lane.
    FiniteStackedBase,
    CudaSharedForest,
}

impl DirectAirVaccFreshCommitmentModeV19 {
    const fn transcript_extra_elements(self) -> usize {
        match self {
            Self::ScalarMerkle | Self::AppendixDBase | Self::FiniteStackedBase => 0,
            // column_start, column_width, forest_width, rows_per_query
            Self::CudaSharedForest => 4,
        }
    }

    const fn uses_scalar_merkle(self) -> bool {
        matches!(self, Self::ScalarMerkle)
    }

    const fn uses_merkle_authentication(self) -> bool {
        matches!(
            self,
            Self::ScalarMerkle | Self::AppendixDBase | Self::FiniteStackedBase
        )
    }

    const fn protocol_tag(self) -> &'static [u8] {
        match self {
            Self::AppendixDBase | Self::FiniteStackedBase => {
                NATIVE_WARP_VACC_APPENDIX_D_PROTOCOL_TAG
            }
            Self::ScalarMerkle | Self::CudaSharedForest => NATIVE_WARP_VACC_PROTOCOL_TAG,
        }
    }
}

/// Verifier-key choice for the transcript-bound backend WARP step.
/// Existing v19/v2/v3 compositions retain the historical segment-index
/// schedule. Seeded HLeaf V4 admits only prior-bearing transitions and uses
/// the canonical positive step one independently of local slot/History index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectAirVaccWarpStepModeV19 {
    LegacySegmentIndex,
    FixedHLeafV4,
}

/// Borrowed verifier output accepted by the trace generator.  SDK's
/// `VerifiedDirectAirVaccV19` maps to this without copying proof data.
pub struct DirectAirVaccVerifierRecordV19<
    'a,
    FreshVerification = MerkleBatchOpeningVerification<EF, Digest>,
> {
    pub producer: &'a DirectAirVaccProducerRecordV19,
    pub verification: &'a WarpVaccStepVerification<
        EF,
        Digest,
        FreshVerification,
        MerkleBatchOpeningVerification<EF, Digest>,
    >,
    pub transcript: &'a TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    /// The verifier-derived preceding output for this homogeneous shard.
    /// Bootstrap records carry `None`; continuation records carry the exact
    /// prior instance whose root was authenticated by the VACC proof.
    pub prior: Option<&'a AccumulatorInstance<EF, Digest>>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DirectAirVaccVerifierErrorV19 {
    InvalidProfile,
    RecordShape(&'static str),
    Transcript(&'static str),
    Merkle(&'static str),
    Algebra(&'static str),
    PriorChain,
    AirTraceCount,
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeStandardVaccPrefixEventMessageV19<T> {
    pub relation_digest: [T; DIGEST_SIZE],
    pub ordinal: T,
    /// Verifier-fixed byte. Dynamic step-byte rows use zero here and are
    /// identified by exactly one `step_flags` selector.
    pub catalog_value: T,
    pub is_step: T,
    pub step_flags: [T; 8],
    pub is_last: T,
    pub event_count: T,
}

define_typed_lookup_bus!(
    NativeStandardVaccPrefixEventBusV19,
    NativeStandardVaccPrefixEventMessageV19
);

#[derive(Clone, Debug, PartialEq, Eq)]
struct NativeStandardVaccPrefixEventV19 {
    relation_digest: Digest,
    ordinal: u32,
    catalog_value: u8,
    step_index: Option<u8>,
    is_last: bool,
    event_count: u32,
}

fn standard_vacc_prefix_events_v19(
    shape: &NativeStandardVaccShapeProfile,
    relation: &NativeStandardVaccRelationProfile,
    include_prior: bool,
    fresh_commitment_mode: DirectAirVaccFreshCommitmentModeV19,
) -> Result<Vec<NativeStandardVaccPrefixEventV19>, DirectAirVaccVerifierErrorV19> {
    let mut values = Vec::new();
    values.extend(
        fresh_commitment_mode
            .protocol_tag()
            .iter()
            .copied()
            .map(|value| (value, None)),
    );
    values.extend(0u64.to_le_bytes().into_iter().map(|value| (value, None)));
    values.extend(
        (relation.relation_description.len() as u64)
            .to_le_bytes()
            .into_iter()
            .map(|value| (value, None)),
    );
    values.extend(
        relation
            .relation_description
            .iter()
            .copied()
            .map(|value| (value, None)),
    );
    for dimension in [
        STANDARD_DIRECT_VACC_INPUT_ARITY,
        shape.num_ood,
        shape.num_shift_queries,
        shape.batching_arity,
        shape.log_message_len,
        shape.log_codeword_len,
    ] {
        values.extend(
            (dimension as u64)
                .to_le_bytes()
                .into_iter()
                .map(|value| (value, None)),
        );
    }
    values.extend((0u8..8).map(|step_index| (0, Some(step_index))));
    for suffix in [1u64, u64::from(include_prior)] {
        values.extend(suffix.to_le_bytes().into_iter().map(|value| (value, None)));
    }

    if values.len() >= F::ORDER_U32 as usize {
        return Err(DirectAirVaccVerifierErrorV19::InvalidProfile);
    }
    let event_count =
        u32::try_from(values.len()).map_err(|_| DirectAirVaccVerifierErrorV19::InvalidProfile)?;
    Ok(values
        .into_iter()
        .enumerate()
        .map(
            |(ordinal, (catalog_value, step_index))| NativeStandardVaccPrefixEventV19 {
                relation_digest: relation.relation_digest,
                ordinal: ordinal as u32,
                catalog_value,
                step_index,
                is_last: ordinal + 1 == event_count as usize,
                event_count,
            },
        )
        .collect())
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeStandardVaccPrefixCatalogPrepColsV19<T> {
    pub active: T,
    pub relation_digest: [T; DIGEST_SIZE],
    pub ordinal: T,
    pub catalog_value: T,
    pub is_step: T,
    pub step_flags: [T; 8],
    pub is_last: T,
    pub event_count: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeStandardVaccPrefixCatalogColsV19<T> {
    pub multiplicity: T,
}

/// Verifier-key-owned table of every canonical prefix event for every
/// relation admitted by one numeric VACC shape group.
#[derive(Clone, ColumnsAir)]
#[columns_via(NativeStandardVaccPrefixCatalogColsV19<u8>)]
pub struct NativeStandardVaccPrefixCatalogAirV19 {
    events: Arc<[NativeStandardVaccPrefixEventV19]>,
    bus: NativeStandardVaccPrefixEventBusV19,
}

impl BaseAir<F> for NativeStandardVaccPrefixCatalogAirV19 {
    fn width(&self) -> usize {
        NativeStandardVaccPrefixCatalogColsV19::<F>::width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = NativeStandardVaccPrefixCatalogPrepColsV19::<F>::width();
        let height = self.events.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(width * height);
        for (row, event) in self.events.iter().enumerate() {
            let cols: &mut NativeStandardVaccPrefixCatalogPrepColsV19<F> =
                values[row * width..(row + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.relation_digest = event.relation_digest;
            cols.ordinal = F::from_u32(event.ordinal);
            cols.catalog_value = F::from_u8(event.catalog_value);
            cols.is_step = F::from_bool(event.step_index.is_some());
            if let Some(step_index) = event.step_index {
                cols.step_flags[step_index as usize] = F::ONE;
            }
            cols.is_last = F::from_bool(event.is_last);
            cols.event_count = F::from_u32(event.event_count);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl openvm_stark_backend::BaseAirWithPublicValues<F> for NativeStandardVaccPrefixCatalogAirV19 {}
impl openvm_stark_backend::PartitionedBaseAir<F> for NativeStandardVaccPrefixCatalogAirV19 {}

impl<AB> Air<AB> for NativeStandardVaccPrefixCatalogAirV19
where
    AB: AirBuilder<F = F> + InteractionBuilder + PairBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder.preprocessed();
        let prep_row = preprocessed
            .row_slice(0)
            .expect("standard VACC prefix catalog preprocessed row");
        let prep: &NativeStandardVaccPrefixCatalogPrepColsV19<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("standard VACC prefix catalog row");
        let local: &NativeStandardVaccPrefixCatalogColsV19<AB::Var> = (*row).borrow();

        builder.assert_bool(prep.active);
        builder.assert_bool(prep.is_step);
        builder.assert_bool(prep.is_last);
        let mut step_sum = AB::Expr::ZERO;
        for flag in prep.step_flags {
            builder.assert_bool(flag);
            step_sum += flag;
        }
        builder.assert_eq(step_sum, prep.is_step);
        builder
            .when(AB::Expr::ONE - prep.active)
            .assert_zero(local.multiplicity);
        self.bus.add_key_with_lookups(
            builder,
            NativeStandardVaccPrefixEventMessageV19 {
                relation_digest: prep.relation_digest.map(Into::into),
                ordinal: prep.ordinal.into(),
                catalog_value: prep.catalog_value.into(),
                is_step: prep.is_step.into(),
                step_flags: prep.step_flags.map(Into::into),
                is_last: prep.is_last.into(),
                event_count: prep.event_count.into(),
            },
            prep.active * local.multiplicity,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeStandardVaccPrefixStreamColsV19<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub proof_idx: T,
    pub start_tidx: T,
    pub start_nonzero: T,
    pub start_inverse: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub has_prior: T,
    pub relation_digest: [T; DIGEST_SIZE],
    pub ordinal: T,
    pub event_count: T,
    pub catalog_value: T,
    pub is_step: T,
    pub step_flags: [T; 8],
    pub event_value: T,
    pub event_bits: [T; 8],
    pub step_lo_sum: T,
    pub step_hi_sum: T,
}

/// Ragged proof-event stream. It authenticates each event against the setup
/// catalog and emits the exact transcript/protocol/cursor interactions that
/// the former relation-per-AIR prefix emitted.
#[derive(Clone, ColumnsAir)]
#[columns_via(NativeStandardVaccPrefixStreamColsV19<u8>)]
pub struct NativeStandardVaccPrefixStreamAirV19 {
    transcript_bus: openvm_recursion_circuit::bus::TranscriptBus,
    phase_cursor_bus: NativeVaccPhaseCursorBus,
    protocol_bus: NativeStandardVaccProtocolBus,
    event_bus: NativeStandardVaccPrefixEventBusV19,
    include_prior: bool,
    warp_step_mode: DirectAirVaccWarpStepModeV19,
    /// The legacy full-transcript verifier closes boundary zero with a prefix
    /// remainder AIR. A resumed verifier starts at an authenticated transcript
    /// checkpoint instead, so emitting that unconsumed boundary would leave a
    /// free phase-cursor obligation.
    emit_start_boundary: bool,
}

impl BaseAir<F> for NativeStandardVaccPrefixStreamAirV19 {
    fn width(&self) -> usize {
        NativeStandardVaccPrefixStreamColsV19::<F>::width()
    }
}
impl openvm_stark_backend::BaseAirWithPublicValues<F> for NativeStandardVaccPrefixStreamAirV19 {}
impl openvm_stark_backend::PartitionedBaseAir<F> for NativeStandardVaccPrefixStreamAirV19 {}

impl<AB> Air<AB> for NativeStandardVaccPrefixStreamAirV19
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("standard VACC prefix stream row");
        let next_row = main
            .row_slice(1)
            .expect("standard VACC prefix stream next row");
        let local: &NativeStandardVaccPrefixStreamColsV19<AB::Var> = (*local_row).borrow();
        let next: &NativeStandardVaccPrefixStreamColsV19<AB::Var> = (*next_row).borrow();

        for flag in [
            local.active,
            local.is_first,
            local.is_last,
            local.start_nonzero,
            local.has_prior,
            local.is_step,
        ] {
            builder.assert_bool(flag);
        }
        builder.assert_zero(local.is_first * (AB::Expr::ONE - local.active));
        builder.assert_zero(local.is_last * (AB::Expr::ONE - local.active));
        builder
            .when(local.active)
            .assert_eq(local.has_prior, AB::Expr::from_bool(self.include_prior));
        builder
            .when(local.active * local.start_nonzero)
            .assert_one(local.start_tidx * local.start_inverse);
        builder
            .when(local.active * (AB::Expr::ONE - local.start_nonzero))
            .assert_zero(local.start_tidx);
        builder
            .when(local.active * (AB::Expr::ONE - local.start_nonzero))
            .assert_zero(local.start_inverse);

        let mut step_sum = AB::Expr::ZERO;
        for flag in local.step_flags {
            builder.assert_bool(flag);
            step_sum += flag;
        }
        builder
            .when(local.active)
            .assert_eq(step_sum, local.is_step);
        let mut byte = AB::Expr::ZERO;
        for (bit_index, bit) in local.event_bits.into_iter().enumerate() {
            builder.assert_bool(bit);
            byte += bit * AB::Expr::from_usize(1 << bit_index);
            builder
                .when(local.active * (AB::Expr::ONE - local.is_step))
                .assert_zero(bit);
        }
        builder
            .when(local.active * local.is_step)
            .assert_eq(local.event_value, byte);
        builder
            .when(local.active * (AB::Expr::ONE - local.is_step))
            .assert_eq(local.event_value, local.catalog_value);
        builder.when(local.active).assert_zero(
            local.event_value
                * (local.step_flags[4]
                    + local.step_flags[5]
                    + local.step_flags[6]
                    + local.step_flags[7]),
        );
        builder
            .when(local.active)
            .assert_zero(local.event_value * local.is_step * (AB::Expr::ONE - local.has_prior));

        let lo_contribution = local.event_value
            * (local.step_flags[0] + local.step_flags[1] * AB::Expr::from_u32(256));
        let hi_contribution = local.event_value
            * (local.step_flags[2] + local.step_flags[3] * AB::Expr::from_u32(256));
        builder
            .when(local.active * local.is_first)
            .assert_zero(local.ordinal);
        builder
            .when(local.active * local.is_first)
            .assert_eq(local.step_lo_sum, lo_contribution.clone());
        builder
            .when(local.active * local.is_first)
            .assert_eq(local.step_hi_sum, hi_contribution.clone());
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when_first_row().assert_zero(local.proof_idx);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);

        let local_continues = local.active * (AB::Expr::ONE - local.is_last);
        let next_starts = next.active * next.is_first;
        let next_continues = next.active * (AB::Expr::ONE - next.is_first);
        builder
            .when_transition()
            .when(AB::Expr::ONE - local.active)
            .assert_zero(next.active);
        builder
            .when_transition()
            .when(local_continues)
            .assert_one(next.active);
        builder
            .when_transition()
            .when(next.active)
            .assert_eq(next.is_first, local.is_last);
        builder
            .when_transition()
            .when(next_starts.clone())
            .assert_eq(next.proof_idx, local.proof_idx + AB::Expr::ONE);
        builder
            .when_transition()
            .when(next_starts.clone())
            .assert_zero(next.ordinal);
        builder
            .when_transition()
            .when(next_starts.clone())
            .assert_eq(
                next.step_lo_sum,
                next.event_value
                    * (next.step_flags[0] + next.step_flags[1] * AB::Expr::from_u32(256)),
            );
        builder.when_transition().when(next_starts).assert_eq(
            next.step_hi_sum,
            next.event_value * (next.step_flags[2] + next.step_flags[3] * AB::Expr::from_u32(256)),
        );
        builder
            .when_transition()
            .when(next_continues.clone())
            .assert_eq(next.proof_idx, local.proof_idx);
        builder
            .when_transition()
            .when(next_continues.clone())
            .assert_eq(next.ordinal, local.ordinal + AB::Expr::ONE);
        for (next_value, local_value) in next.relation_digest.into_iter().zip(local.relation_digest)
        {
            builder
                .when_transition()
                .when(next_continues.clone())
                .assert_eq(next_value, local_value);
        }
        for (next_value, local_value) in [
            (next.start_tidx, local.start_tidx),
            (next.start_nonzero, local.start_nonzero),
            (next.start_inverse, local.start_inverse),
            (next.segment_index_lo, local.segment_index_lo),
            (next.segment_index_hi, local.segment_index_hi),
            (next.has_prior, local.has_prior),
            (next.event_count, local.event_count),
        ] {
            builder
                .when_transition()
                .when(next_continues.clone())
                .assert_eq(next_value, local_value);
        }
        builder
            .when_transition()
            .when(next_continues.clone())
            .assert_eq(
                next.step_lo_sum,
                local.step_lo_sum
                    + next.event_value
                        * (next.step_flags[0] + next.step_flags[1] * AB::Expr::from_u32(256)),
            );
        builder.when_transition().when(next_continues).assert_eq(
            next.step_hi_sum,
            local.step_hi_sum
                + next.event_value
                    * (next.step_flags[2] + next.step_flags[3] * AB::Expr::from_u32(256)),
        );
        builder
            .when(local.active * local.is_last)
            .assert_eq(local.ordinal + AB::Expr::ONE, local.event_count);
        let expected_step_lo = match self.warp_step_mode {
            DirectAirVaccWarpStepModeV19::LegacySegmentIndex => {
                AB::Expr::from(local.segment_index_lo) * local.has_prior
            }
            DirectAirVaccWarpStepModeV19::FixedHLeafV4 => {
                AB::Expr::from(local.has_prior)
                    * AB::Expr::from_u32(FIXED_HLEAF_CONTINUATION_WARP_STEP_V4)
            }
        };
        let expected_step_hi = match self.warp_step_mode {
            DirectAirVaccWarpStepModeV19::LegacySegmentIndex => {
                AB::Expr::from(local.segment_index_hi) * local.has_prior
            }
            DirectAirVaccWarpStepModeV19::FixedHLeafV4 => AB::Expr::ZERO,
        };
        builder
            .when(local.active * local.is_last)
            .assert_eq(local.step_lo_sum, expected_step_lo);
        builder
            .when(local.active * local.is_last)
            .assert_eq(local.step_hi_sum, expected_step_hi);

        self.event_bus.lookup_key(
            builder,
            NativeStandardVaccPrefixEventMessageV19 {
                relation_digest: local.relation_digest.map(Into::into),
                ordinal: local.ordinal.into(),
                catalog_value: local.catalog_value.into(),
                is_step: local.is_step.into(),
                step_flags: local.step_flags.map(Into::into),
                is_last: local.is_last.into(),
                event_count: local.event_count.into(),
            },
            local.active,
        );
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            AB::Expr::from(local.start_tidx)
                + AB::Expr::from_usize(D_EF) * AB::Expr::from(local.ordinal),
            [
                local.event_value.into(),
                AB::Expr::ZERO,
                AB::Expr::ZERO,
                AB::Expr::ZERO,
            ],
            local.active,
        );
        self.protocol_bus.add_key_with_lookups(
            builder,
            NativeStandardVaccProtocolMessage {
                proof_idx: local.proof_idx.into(),
                start_tidx: local.start_tidx.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                has_prior: local.has_prior.into(),
                relation_digest: local.relation_digest.map(Into::into),
            },
            local.active * local.is_first,
        );
        self.phase_cursor_bus.send(
            builder,
            NativeVaccPhaseCursorMessage {
                proof_idx: local.proof_idx.into(),
                boundary: AB::Expr::ZERO,
                tidx: local.start_tidx.into(),
            },
            local.active
                * local.is_first
                * local.start_nonzero
                * AB::Expr::from_bool(self.emit_start_boundary),
        );
        self.phase_cursor_bus.send(
            builder,
            NativeVaccPhaseCursorMessage {
                proof_idx: local.proof_idx.into(),
                boundary: AB::Expr::ONE,
                tidx: AB::Expr::from(local.start_tidx)
                    + AB::Expr::from_usize(D_EF) * (AB::Expr::from(local.ordinal) + AB::Expr::ONE),
            },
            local.active * local.is_last,
        );
    }
}

fn vacc_cursor_role_lengths_v19(
    profile: &NativeStandardVaccShapeProfile,
    include_prior: bool,
    fresh_commitment_mode: DirectAirVaccFreshCommitmentModeV19,
) -> [usize; VACC_ROLE_COUNT] {
    let mut lengths = [0usize; VACC_ROLE_COUNT];
    lengths[VACC_ROLE_FRESH_ROOT] = DIGEST_SIZE + fresh_commitment_mode.transcript_extra_elements();
    lengths[VACC_ROLE_FRESH_ALPHA] = profile.log_codeword_len;
    lengths[VACC_ROLE_FRESH_MU] = 1;
    lengths[VACC_ROLE_FRESH_BETA_TAIL] = profile.beta_len - profile.log_constraints;
    if include_prior {
        lengths[VACC_ROLE_PRIOR_ROOT] = DIGEST_SIZE;
        lengths[VACC_ROLE_PRIOR_ALPHA] = profile.log_codeword_len;
        lengths[VACC_ROLE_PRIOR_MU] = 1;
        lengths[VACC_ROLE_PRIOR_BETA] = profile.beta_len;
        lengths[VACC_ROLE_PRIOR_ETA] = 1;
    }
    lengths[VACC_ROLE_FRESH_TAU] = profile.log_constraints;
    lengths[VACC_ROLE_OMEGA] = 1;
    lengths[VACC_ROLE_SELECTOR] = 1;
    lengths[VACC_ROLE_TWIN_SUMCHECK] = twin_degree(
        profile.log_codeword_len,
        profile.log_constraints,
        profile.max_degree,
    ) + 2;
    lengths[VACC_ROLE_OUTPUT_ROOT] = DIGEST_SIZE;
    lengths[VACC_ROLE_NU] = 1;
    lengths[VACC_ROLE_ETA] = 1;
    lengths[VACC_ROLE_OOD_POINT] = profile.num_ood * profile.log_codeword_len;
    lengths[VACC_ROLE_OOD_ANSWER] = profile.num_ood;
    lengths[VACC_ROLE_SHIFT] = profile.num_shift_queries;
    lengths[VACC_ROLE_XI] = profile.batching_arity.ilog2() as usize;
    lengths[VACC_ROLE_BATCHING_SUMCHECK] = profile.log_codeword_len * 4;
    lengths[VACC_ROLE_MU] = 1;
    lengths
}

fn vacc_cursor_role_lengths_exact_v19(
    profile: &NativeStandardVaccShapeProfile,
    input_arity: usize,
    fresh_count: usize,
    include_prior: bool,
) -> [usize; VACC_ROLE_COUNT] {
    let mut lengths = [0usize; VACC_ROLE_COUNT];
    // Each finite-stacked descriptor observes six domain/layout values,
    // source ordinal, two code dimensions, and the root limbs.
    lengths[VACC_ROLE_FRESH_ROOT] = fresh_count * (9 + DIGEST_SIZE);
    lengths[VACC_ROLE_FRESH_ALPHA] = fresh_count * profile.log_codeword_len;
    lengths[VACC_ROLE_FRESH_MU] = fresh_count;
    lengths[VACC_ROLE_FRESH_BETA_TAIL] = fresh_count * (profile.beta_len - profile.log_constraints);
    if include_prior {
        lengths[VACC_ROLE_PRIOR_ROOT] = DIGEST_SIZE;
        lengths[VACC_ROLE_PRIOR_ALPHA] = profile.log_codeword_len;
        lengths[VACC_ROLE_PRIOR_MU] = 1;
        lengths[VACC_ROLE_PRIOR_BETA] = profile.beta_len;
        lengths[VACC_ROLE_PRIOR_ETA] = 1;
    }
    lengths[VACC_ROLE_FRESH_TAU] = fresh_count * profile.log_constraints;
    lengths[VACC_ROLE_OMEGA] = 1;
    lengths[VACC_ROLE_SELECTOR] = input_arity.ilog2() as usize;
    lengths[VACC_ROLE_TWIN_SUMCHECK] = (input_arity.ilog2() as usize)
        * (twin_degree(
            profile.log_codeword_len,
            profile.log_constraints,
            profile.max_degree,
        ) + 2);
    lengths[VACC_ROLE_OUTPUT_ROOT] = DIGEST_SIZE;
    lengths[VACC_ROLE_NU] = 1;
    lengths[VACC_ROLE_ETA] = 1;
    lengths[VACC_ROLE_OOD_POINT] = profile.num_ood * profile.log_codeword_len;
    lengths[VACC_ROLE_OOD_ANSWER] = profile.num_ood;
    lengths[VACC_ROLE_SHIFT] = profile.num_shift_queries;
    lengths[VACC_ROLE_XI] = profile.batching_arity.ilog2() as usize;
    lengths[VACC_ROLE_BATCHING_SUMCHECK] = profile.log_codeword_len * 4;
    lengths[VACC_ROLE_MU] = 1;
    lengths
}

fn next_nonempty_vacc_roles_v19(lengths: &[usize; VACC_ROLE_COUNT]) -> [usize; VACC_ROLE_COUNT] {
    core::array::from_fn(|role| {
        ((role + 1)..VACC_ROLE_COUNT)
            .find(|&candidate| lengths[candidate] != 0)
            .unwrap_or(VACC_ROLE_COUNT)
    })
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct DirectAirVaccTranscriptCursorColsV19<T> {
    pub active: T,
    pub proof_idx: T,
    pub tidx: T,
    pub role: T,
    pub ordinal: T,
    pub role_flags: [T; VACC_ROLE_COUNT],
    pub is_ext: T,
    pub is_sample: T,
    pub is_first_ordinal: T,
    pub is_last_ordinal: T,
    pub ordinal_inverse: T,
    pub last_ordinal_inverse: T,
    pub sumcheck_round: T,
    pub sumcheck_step: T,
    pub is_last_sumcheck_step: T,
    pub sumcheck_last_step_inverse: T,
    /// Prepared products keep the exact cursor state machine within the
    /// recursive degree-four envelope.  Every helper is constrained below;
    /// none is a host acceptance bit.
    pub is_sumcheck: T,
    pub sumcheck_step_continues: T,
    pub continues_proof: T,
    pub continues_sumcheck_phase: T,
    pub starts_proof: T,
    pub value: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(DirectAirVaccTranscriptCursorColsV19<u8>)]
pub struct DirectAirVaccTranscriptCursorAirV19 {
    pub transcript_bus: openvm_recursion_circuit::bus::TranscriptBus,
    pub semantic_bus: openvm_recursion_circuit::bus::TranscriptBus,
    pub phase_cursor_bus: NativeVaccPhaseCursorBus,
    pub role_bus: NativeVaccTranscriptRoleBus,
    pub lengths: [usize; VACC_ROLE_COUNT],
    pub next_roles: [usize; VACC_ROLE_COUNT],
    pub twin_event_count: usize,
    /// Reduced-SWIRL has setup-fixed arity but a runtime-constrained active
    /// fresh prefix and variable original-root tuples. In this mode role ends
    /// are proved by typed consumers and the cursor transition itself rather
    /// than by verifier-key constant role lengths.
    pub dynamic_role_lengths: bool,
    /// Bootstrap omits the entire prior-accumulator role interval; a
    /// continuation must include it because its typed prior consumers are
    /// active. This permits exactly that one protocol-prescribed skip.
    pub dynamic_optional_prior: bool,
    /// Scalar and Appendix-D fresh commitments have dedicated consumers on
    /// `role_bus`.  The reduced stacked-RS path instead authenticates every
    /// descriptor field and original root directly at its exact transcript
    /// index on `semantic_bus`; emitting a second role lookup there would be
    /// an unconsumed duplicate statement.
    pub emit_fresh_root_role: bool,
}

impl openvm_stark_backend::BaseAirWithPublicValues<F> for DirectAirVaccTranscriptCursorAirV19 {}
impl openvm_stark_backend::PartitionedBaseAir<F> for DirectAirVaccTranscriptCursorAirV19 {}
impl BaseAir<F> for DirectAirVaccTranscriptCursorAirV19 {
    fn width(&self) -> usize {
        DirectAirVaccTranscriptCursorColsV19::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for DirectAirVaccTranscriptCursorAirV19 {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("direct VACC cursor row");
        let next_row = main.row_slice(1).expect("direct VACC cursor next row");
        let local: &DirectAirVaccTranscriptCursorColsV19<AB::Var> = (*local_row).borrow();
        let next: &DirectAirVaccTranscriptCursorColsV19<AB::Var> = (*next_row).borrow();

        for flag in [
            local.active,
            local.is_ext,
            local.is_sample,
            local.is_first_ordinal,
            local.is_last_ordinal,
            local.is_last_sumcheck_step,
            local.is_sumcheck,
            local.sumcheck_step_continues,
            local.continues_proof,
            local.continues_sumcheck_phase,
            local.starts_proof,
        ] {
            builder.assert_bool(flag);
        }
        for flag in local.role_flags {
            builder.assert_bool(flag);
        }
        builder.assert_eq(
            local.active,
            local
                .role_flags
                .into_iter()
                .map(AB::Expr::from)
                .sum::<AB::Expr>(),
        );
        builder.when_first_row().assert_one(local.active);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);

        let role = local
            .role_flags
            .into_iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (index, flag)| {
                acc + flag * AB::Expr::from_usize(index)
            });
        builder.when(local.active).assert_eq(local.role, role);
        for (role, &length) in self.lengths.iter().enumerate() {
            if length == 0 {
                builder.assert_zero(local.role_flags[role]);
            }
        }
        let nominal_len = local
            .role_flags
            .into_iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (role, flag)| {
                acc + flag * AB::Expr::from_usize(self.lengths[role])
            });
        let distance = nominal_len - AB::Expr::ONE - local.ordinal;
        builder
            .when(local.active * local.is_first_ordinal)
            .assert_zero(local.ordinal);
        builder
            .when(local.active * (AB::Expr::ONE - local.is_first_ordinal))
            .assert_one(local.ordinal * local.ordinal_inverse);
        if !self.dynamic_role_lengths {
            builder
                .when(local.active * local.is_last_ordinal)
                .assert_zero(distance.clone());
            builder
                .when(local.active * (AB::Expr::ONE - local.is_last_ordinal))
                .assert_one(distance * local.last_ordinal_inverse);
        }

        let is_shift = local.role_flags[VACC_ROLE_SHIFT];
        builder
            .when(local.active)
            .assert_eq(local.is_ext, AB::Expr::ONE - is_shift);
        for limb in &local.value[1..] {
            builder.when(local.active * is_shift).assert_zero(*limb);
        }

        let twin_sumcheck = local.role_flags[VACC_ROLE_TWIN_SUMCHECK];
        let batching_sumcheck = local.role_flags[VACC_ROLE_BATCHING_SUMCHECK];
        let is_sumcheck = twin_sumcheck + batching_sumcheck;
        builder.assert_eq(local.is_sumcheck, is_sumcheck.clone());
        let event_count = twin_sumcheck * AB::Expr::from_usize(self.twin_event_count)
            + batching_sumcheck * AB::Expr::from_usize(4);
        builder.when(local.active * is_sumcheck.clone()).assert_eq(
            local.ordinal,
            local.sumcheck_round * event_count.clone() + local.sumcheck_step,
        );
        builder
            .when(local.active * is_sumcheck.clone() * local.is_first_ordinal)
            .assert_zero(local.sumcheck_round);
        builder
            .when(local.active * is_sumcheck.clone() * local.is_first_ordinal)
            .assert_zero(local.sumcheck_step);
        let step_distance = event_count - AB::Expr::ONE - local.sumcheck_step;
        builder
            .when(local.active * is_sumcheck.clone() * local.is_last_sumcheck_step)
            .assert_zero(step_distance.clone());
        builder.assert_eq(
            local.sumcheck_step_continues,
            local.active * local.is_sumcheck * (AB::Expr::ONE - local.is_last_sumcheck_step),
        );
        builder
            .when(local.sumcheck_step_continues)
            .assert_one(step_distance * local.sumcheck_last_step_inverse);
        for witness in [
            local.sumcheck_round,
            local.sumcheck_step,
            local.sumcheck_last_step_inverse,
        ] {
            builder
                .when(local.active * (AB::Expr::ONE - is_sumcheck.clone()))
                .assert_zero(witness);
        }
        builder
            .when(local.active * (AB::Expr::ONE - is_sumcheck.clone()))
            .assert_zero(local.is_last_sumcheck_step);
        let always_sample = local.role_flags[VACC_ROLE_FRESH_TAU]
            + local.role_flags[VACC_ROLE_OMEGA]
            + local.role_flags[VACC_ROLE_SELECTOR]
            + local.role_flags[VACC_ROLE_OOD_POINT]
            + local.role_flags[VACC_ROLE_SHIFT]
            + local.role_flags[VACC_ROLE_XI];
        builder.when(local.active).assert_eq(
            local.is_sample,
            always_sample + is_sumcheck.clone() * local.is_last_sumcheck_step,
        );

        let width = AB::Expr::ONE + local.is_ext * AB::Expr::from_usize(D_EF - 1);
        let phase_end = AB::Expr::from(local.is_last_ordinal);
        let proof_end = local.role_flags[VACC_ROLE_MU] * phase_end.clone();
        let starts_proof = local.role_flags[VACC_ROLE_FRESH_ROOT] * local.is_first_ordinal;
        builder.assert_eq(local.starts_proof, local.active * starts_proof.clone());
        builder.assert_eq(local.continues_proof, local.active - local.starts_proof);
        builder.assert_eq(
            local.continues_sumcheck_phase,
            local.active * local.is_sumcheck * (AB::Expr::ONE - local.is_first_ordinal),
        );
        self.phase_cursor_bus.receive(
            builder,
            NativeVaccPhaseCursorMessage {
                proof_idx: local.proof_idx.into(),
                boundary: AB::Expr::ONE,
                tidx: local.tidx.into(),
            },
            local.active * starts_proof,
        );
        self.phase_cursor_bus.send(
            builder,
            NativeVaccPhaseCursorMessage {
                proof_idx: local.proof_idx.into(),
                boundary: AB::Expr::from_usize(2),
                tidx: local.tidx + width.clone(),
            },
            local.active * proof_end.clone(),
        );

        let expected_next_role = local
            .role_flags
            .into_iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |acc, (role, flag)| {
                acc + flag * AB::Expr::from_usize(self.next_roles[role])
            });
        let same_phase = AB::Expr::ONE - phase_end.clone();
        {
            let mut transition = builder.when_transition();
            let mut continuing = transition.when(next.continues_proof);
            continuing.assert_eq(next.proof_idx, local.proof_idx);
            continuing.assert_eq(next.tidx, local.tidx + width.clone());
            if self.dynamic_optional_prior {
                continuing
                    .when(same_phase.clone())
                    .assert_eq(next.role, local.role);
            } else {
                continuing.assert_eq(
                    next.role,
                    same_phase.clone() * local.role + phase_end.clone() * expected_next_role,
                );
            }
            continuing.assert_eq(
                next.ordinal,
                same_phase.clone() * (local.ordinal + AB::Expr::ONE),
            );
        }
        if self.dynamic_optional_prior {
            let after_prior = self.next_roles[VACC_ROLE_PRIOR_ETA];
            let optional_skip =
                local.role_flags[VACC_ROLE_FRESH_BETA_TAIL] * next.role_flags[after_prior];
            let ordinary_advance = local.role_flags.into_iter().enumerate().fold(
                AB::Expr::ZERO,
                |sum, (role, flag)| {
                    let next_role = self.next_roles[role];
                    if next_role < VACC_ROLE_COUNT {
                        sum + flag * next.role_flags[next_role]
                    } else {
                        // `VACC_ROLE_COUNT` is the sentinel for the terminal
                        // role. The surrounding constraint is disabled when
                        // no row continues this proof, but Rust indexing is
                        // eager, so represent the sentinel contribution as
                        // zero rather than indexing one past the flag array.
                        sum
                    }
                },
            );
            builder
                .when_transition()
                .when(next.continues_proof * phase_end.clone())
                .assert_one(ordinary_advance + optional_skip);
        }
        {
            let mut transition = builder.when_transition();
            let mut sumcheck_transition = transition.when(next.continues_sumcheck_phase);
            sumcheck_transition.assert_eq(
                next.sumcheck_round,
                local.sumcheck_round + local.is_last_sumcheck_step,
            );
            sumcheck_transition.assert_eq(
                next.sumcheck_step,
                (AB::Expr::ONE - local.is_last_sumcheck_step) * (local.sumcheck_step + AB::F::ONE),
            );
        }
        {
            let mut transition = builder.when_transition();
            let mut next_proof = transition.when(next.starts_proof);
            next_proof.assert_one(proof_end.clone());
            next_proof.assert_eq(next.proof_idx, local.proof_idx + AB::F::ONE);
            next_proof.assert_eq(next.role, AB::Expr::from_usize(VACC_ROLE_FRESH_ROOT));
            next_proof.assert_zero(next.ordinal);
        }
        builder
            .when_transition()
            .when(local.active - next.active)
            .assert_one(proof_end.clone());
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(proof_end);

        let message = NativeVaccTranscriptRoleMessage {
            proof_idx: local.proof_idx.into(),
            role: local.role.into(),
            ordinal: local.ordinal.into(),
            tidx: local.tidx.into(),
            value: local.value.map(Into::into),
            is_ext: local.is_ext.into(),
            is_sample: local.is_sample.into(),
        };
        self.role_bus.send(
            builder,
            message,
            local.active
                * (AB::Expr::ONE
                    - local.role_flags[VACC_ROLE_FRESH_ROOT]
                        * AB::Expr::from_bool(!self.emit_fresh_root_role)),
        );
        for (limb, value) in local.value.into_iter().enumerate() {
            let enabled = local.active
                * (local.is_ext + (AB::Expr::ONE - local.is_ext) * AB::Expr::from_bool(limb == 0));
            let transcript_message = TranscriptBusMessage {
                tidx: local.tidx + AB::Expr::from_usize(limb),
                value: value.into(),
                is_sample: local.is_sample.into(),
            };
            self.transcript_bus.receive(
                builder,
                local.proof_idx,
                transcript_message.clone(),
                enabled.clone(),
            );
            self.semantic_bus
                .send(builder, local.proof_idx, transcript_message, enabled);
        }
    }
}

pub struct DirectAirVaccVectorCoordinateAirV19 {
    pub inner: NativeVectorCoordinateAir,
    pub role_bus: NativeVaccTranscriptRoleBus,
    pub selector_vector: usize,
    pub xi_vector: usize,
    /// Fixed local HLeaf proof slot for this physical table. Legacy unbounded
    /// batches retain one unslotted table.
    pub fixed_proof_slot: Option<usize>,
}

impl openvm_stark_backend::BaseAirWithPublicValues<F> for DirectAirVaccVectorCoordinateAirV19 {}
impl openvm_stark_backend::PartitionedBaseAir<F> for DirectAirVaccVectorCoordinateAirV19 {}
impl BaseAir<F> for DirectAirVaccVectorCoordinateAirV19 {
    fn width(&self) -> usize {
        NativeVectorCoordinateCols::<F>::width()
    }
}
impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for DirectAirVaccVectorCoordinateAirV19 {
    fn eval(&self, builder: &mut AB) {
        self.inner.eval(builder);
        let main = builder.main();
        let row = main.row_slice(0).expect("direct VACC vector role row");
        let local: &NativeVectorCoordinateCols<AB::Var> = (*row).borrow();
        if let Some(proof_slot) = self.fixed_proof_slot {
            builder
                .when(local.active)
                .assert_eq(local.proof_idx, AB::Expr::from_usize(proof_slot));
        }
        let is_transcript = local.source_kind[VECTOR_SOURCE_TRANSCRIPT];
        let selector = AB::Expr::from_usize(self.selector_vector);
        let xi = AB::Expr::from_usize(self.xi_vector);
        builder
            .when(local.active * is_transcript)
            .assert_zero((local.vector - selector.clone()) * (local.vector - xi.clone()));
        let is_xi = (local.vector - selector.clone())
            * (AB::F::from_usize(self.xi_vector) - AB::F::from_usize(self.selector_vector))
                .inverse();
        builder
            .when(local.active * is_transcript)
            .assert_bool(is_xi.clone());
        self.role_bus.receive(
            builder,
            NativeVaccTranscriptRoleMessage {
                proof_idx: local.proof_idx.into(),
                role: AB::Expr::from_usize(VACC_ROLE_SELECTOR)
                    + is_xi * AB::Expr::from_usize(VACC_ROLE_XI - VACC_ROLE_SELECTOR),
                ordinal: local.coordinate.into(),
                tidx: local.tidx.into(),
                value: local.value.map(Into::into),
                is_ext: AB::Expr::ONE,
                is_sample: AB::Expr::ONE,
            },
            local.active * is_transcript,
        );
    }
}

pub struct DirectAirVaccTwinOmegaAirV19 {
    pub inner: NativeTwinOmegaAir,
    pub role_bus: NativeVaccTranscriptRoleBus,
}

impl openvm_stark_backend::BaseAirWithPublicValues<F> for DirectAirVaccTwinOmegaAirV19 {}
impl openvm_stark_backend::PartitionedBaseAir<F> for DirectAirVaccTwinOmegaAirV19 {}
impl BaseAir<F> for DirectAirVaccTwinOmegaAirV19 {
    fn width(&self) -> usize {
        NativeTwinOmegaCols::<F>::width()
    }
}

pub struct DirectAirVaccCoefficientSumcheckAirV19 {
    pub inner: NativeCoefficientSumcheckAir,
    pub role_bus: NativeVaccTranscriptRoleBus,
}

impl openvm_stark_backend::BaseAirWithPublicValues<F> for DirectAirVaccCoefficientSumcheckAirV19 {}
impl openvm_stark_backend::PartitionedBaseAir<F> for DirectAirVaccCoefficientSumcheckAirV19 {}
impl BaseAir<F> for DirectAirVaccCoefficientSumcheckAirV19 {
    fn width(&self) -> usize {
        NativeCoefficientSumcheckCols::<F>::width()
    }
}
impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for DirectAirVaccCoefficientSumcheckAirV19
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield:
        openvm_stark_backend::p3_field::extension::BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        self.inner.eval(builder);
        let main = builder.main();
        let row = main.row_slice(0).expect("direct VACC sumcheck role row");
        let local: &NativeCoefficientSumcheckCols<AB::Var> = (*row).borrow();
        let degree = AB::Expr::from_usize(self.inner.twin_degree)
            + local.kind
                * (AB::Expr::from_usize(self.inner.batching_degree)
                    - AB::Expr::from_usize(self.inner.twin_degree));
        let role = AB::Expr::from_usize(VACC_ROLE_TWIN_SUMCHECK)
            + local.kind
                * AB::Expr::from_usize(VACC_ROLE_BATCHING_SUMCHECK - VACC_ROLE_TWIN_SUMCHECK);
        let event_count = degree.clone() + AB::Expr::from_usize(2);
        self.role_bus.receive(
            builder,
            NativeVaccTranscriptRoleMessage {
                proof_idx: local.proof_idx.into(),
                role: role.clone(),
                ordinal: local.round * event_count.clone() + local.coefficient_index,
                tidx: local.tidx + local.coefficient_index * AB::Expr::from_usize(D_EF),
                value: local.coefficient.map(Into::into),
                is_ext: AB::Expr::ONE,
                is_sample: AB::Expr::ZERO,
            },
            local.active,
        );
        self.role_bus.receive(
            builder,
            NativeVaccTranscriptRoleMessage {
                proof_idx: local.proof_idx.into(),
                role,
                ordinal: local.round * event_count + degree.clone() + AB::Expr::ONE,
                tidx: local.tidx + (degree + AB::Expr::ONE) * AB::Expr::from_usize(D_EF),
                value: local.challenge.map(Into::into),
                is_ext: AB::Expr::ONE,
                is_sample: AB::Expr::ONE,
            },
            local.active * local.is_last_coefficient,
        );
    }
}
impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for DirectAirVaccTwinOmegaAirV19 {
    fn eval(&self, builder: &mut AB) {
        self.inner.eval(builder);
        let main = builder.main();
        let row = main.row_slice(0).expect("direct VACC omega role row");
        let local: &NativeTwinOmegaCols<AB::Var> = (*row).borrow();
        self.role_bus.receive(
            builder,
            NativeVaccTranscriptRoleMessage {
                proof_idx: local.proof_idx.into(),
                role: AB::Expr::from_usize(VACC_ROLE_OMEGA),
                ordinal: AB::Expr::ZERO,
                tidx: local.tidx.into(),
                value: local.value.map(Into::into),
                is_ext: AB::Expr::ONE,
                is_sample: AB::Expr::ONE,
            },
            local.active,
        );
    }
}

pub struct DirectAirVaccTwinFinalAirV19 {
    pub inner: NativeTwinFinalAir,
    pub role_bus: NativeVaccTranscriptRoleBus,
}

impl openvm_stark_backend::BaseAirWithPublicValues<F> for DirectAirVaccTwinFinalAirV19 {}
impl openvm_stark_backend::PartitionedBaseAir<F> for DirectAirVaccTwinFinalAirV19 {}
impl BaseAir<F> for DirectAirVaccTwinFinalAirV19 {
    fn width(&self) -> usize {
        NativeTwinFinalCols::<F>::width()
    }
}
impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for DirectAirVaccTwinFinalAirV19
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield:
        openvm_stark_backend::p3_field::extension::BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        self.inner.eval(builder);
        let main = builder.main();
        let row = main.row_slice(0).expect("direct VACC twin-final role row");
        let local: &NativeTwinFinalCols<AB::Var> = (*row).borrow();
        for (role, tidx, value) in [
            (VACC_ROLE_NU, local.nu_tidx, local.nu_0),
            (VACC_ROLE_ETA, local.eta_tidx, local.eta),
        ] {
            self.role_bus.receive(
                builder,
                NativeVaccTranscriptRoleMessage {
                    proof_idx: local.proof_idx.into(),
                    role: AB::Expr::from_usize(role),
                    ordinal: AB::Expr::ZERO,
                    tidx: tidx.into(),
                    value: value.map(Into::into),
                    is_ext: AB::Expr::ONE,
                    is_sample: AB::Expr::ZERO,
                },
                local.active,
            );
        }
    }
}

pub struct DirectAirVaccOodClaimAirV19 {
    pub inner: NativeOodClaimAir,
    pub role_bus: NativeVaccTranscriptRoleBus,
    pub dimension: usize,
}

impl openvm_stark_backend::BaseAirWithPublicValues<F> for DirectAirVaccOodClaimAirV19 {}
impl openvm_stark_backend::PartitionedBaseAir<F> for DirectAirVaccOodClaimAirV19 {}
impl BaseAir<F> for DirectAirVaccOodClaimAirV19 {
    fn width(&self) -> usize {
        NativeOodClaimCols::<F>::width()
    }
}
impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for DirectAirVaccOodClaimAirV19 {
    fn eval(&self, builder: &mut AB) {
        self.inner.eval(builder);
        let main = builder.main();
        let row = main.row_slice(0).expect("direct VACC OOD role row");
        let local: &NativeOodClaimCols<AB::Var> = (*row).borrow();
        self.role_bus.receive(
            builder,
            NativeVaccTranscriptRoleMessage {
                proof_idx: local.proof_idx.into(),
                role: AB::Expr::from_usize(VACC_ROLE_OOD_POINT)
                    + local.is_target
                        * AB::Expr::from_usize(VACC_ROLE_OOD_ANSWER - VACC_ROLE_OOD_POINT),
                ordinal: (AB::Expr::ONE - local.is_target)
                    * (local.ood * AB::Expr::from_usize(self.dimension) + local.coordinate)
                    + local.is_target * local.ood,
                tidx: local.tidx.into(),
                value: local.value.map(Into::into),
                is_ext: AB::Expr::ONE,
                is_sample: AB::Expr::ONE - local.is_target,
            },
            local.active,
        );
    }
}

pub struct DirectAirVaccShiftScheduleAirV19 {
    pub inner: NativeShiftScheduleAir,
    pub role_bus: NativeVaccTranscriptRoleBus,
}

fn generate_direct_air_vacc_shift_schedule_trace_v19(
    proof_idx: usize,
    samples: &[F],
    sample_tidx: &[usize],
    indices: &[u32],
    index_lookup_count: u32,
    log_codeword_len: usize,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if samples.is_empty()
        || samples.len() != sample_tidx.len()
        || samples.len() != indices.len()
        || log_codeword_len == 0
        // The recorded backend samples one canonical BabyBear element and
        // masks its low bits. Unlike the legacy exact-uniform shift AIR, v19
        // intentionally permits the backend's biased `sample_bits` rule and
        // therefore supports every bit length accepted by that backend:
        // `2^bits < p`, i.e. at most 30 bits for BabyBear. The shared
        // ExpBitsLen/RightShift tables authenticate the full canonical
        // 31-bit decomposition, including the quotient above bit 27.
        || log_codeword_len > 30
    {
        return None;
    }
    let valid_rows = samples.len() * log_codeword_len;
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeShiftScheduleCols::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    let mask = (1u32 << log_codeword_len) - 1;
    for shift in 0..samples.len() {
        let canonical = samples[shift].as_canonical_u32();
        if canonical & mask != indices[shift] {
            return None;
        }
        let quotient = canonical >> log_codeword_len;
        let mut reconstructed = 0u32;
        for coordinate in 0..log_codeword_len {
            let bit_index = log_codeword_len - 1 - coordinate;
            let bit = (indices[shift] >> bit_index) & 1;
            let before = reconstructed;
            reconstructed += bit << bit_index;
            let row_index = shift * log_codeword_len + coordinate;
            let cols: &mut NativeShiftScheduleCols<F> =
                trace[row_index * width..(row_index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.shift = F::from_usize(shift);
            cols.coordinate = F::from_usize(coordinate);
            cols.is_first = F::from_bool(coordinate == 0);
            cols.is_last = F::from_bool(coordinate + 1 == log_codeword_len);
            cols.is_first_shift = F::from_bool(shift == 0 && coordinate == 0);
            cols.proof_idx = F::from_usize(proof_idx);
            cols.tidx = F::from_usize(sample_tidx[shift]);
            cols.sample = samples[shift];
            cols.accepted_inverse = F::ZERO;
            cols.quotient = F::from_u32(quotient);
            cols.index = F::from_u32(indices[shift]);
            cols.bit = F::from_u32(bit);
            cols.power = F::from_u32(1u32 << bit_index);
            cols.reconstructed_before = F::from_u32(before);
            cols.reconstructed_after = F::from_u32(reconstructed);
            cols.value[0] = F::from_u32(bit);
            cols.index_lookup_count = if coordinate == 0 {
                F::from_u32(index_lookup_count)
            } else {
                F::ZERO
            };
        }
    }
    Some(RowMajorMatrix::new(trace, width))
}

impl openvm_stark_backend::BaseAirWithPublicValues<F> for DirectAirVaccShiftScheduleAirV19 {}
impl openvm_stark_backend::PartitionedBaseAir<F> for DirectAirVaccShiftScheduleAirV19 {}
impl BaseAir<F> for DirectAirVaccShiftScheduleAirV19 {
    fn width(&self) -> usize {
        NativeShiftScheduleCols::<F>::width()
    }
}
impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for DirectAirVaccShiftScheduleAirV19 {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("direct VACC shift row");
        let next_row = main.row_slice(1).expect("direct VACC next shift row");
        let local: &NativeShiftScheduleCols<AB::Var> = (*local_row).borrow();
        let next: &NativeShiftScheduleCols<AB::Var> = (*next_row).borrow();
        for flag in [
            local.active,
            local.is_first,
            local.is_last,
            local.is_first_shift,
            local.bit,
        ] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first_shift);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .when(local.active - next.active)
            .assert_one(local.is_last);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);
        builder
            .when(local.active * local.is_first_shift)
            .assert_one(local.is_first);
        builder
            .when(local.active * local.is_first_shift)
            .assert_zero(local.shift);
        builder
            .when(local.active * local.is_first)
            .assert_zero(local.coordinate);
        builder.when(local.active * local.is_last).assert_eq(
            local.coordinate,
            AB::Expr::from_usize(self.inner.log_codeword_len - 1),
        );
        builder
            .when(local.active * local.is_first)
            .assert_zero(local.reconstructed_before);
        builder.when(local.active * local.is_first).assert_eq(
            local.power,
            AB::Expr::from_u32(1u32 << (self.inner.log_codeword_len - 1)),
        );
        builder.when(local.active).assert_eq(
            local.reconstructed_after,
            local.reconstructed_before + local.bit * local.power,
        );
        // This legacy column used to witness `(sample + 1)^{-1}`. Protocol
        // v19 follows the backend's biased `sample_bits_raw`, so `-1` is a
        // valid raw sample and the column is fixed to zero.
        builder
            .when(local.active)
            .assert_zero(local.accepted_inverse);

        let same_shift = next.active * (AB::Expr::ONE - next.is_first);
        let mut transition = builder.when_transition();
        let mut same = transition.when(same_shift);
        same.assert_eq(next.proof_idx, local.proof_idx);
        same.assert_zero(next.is_first_shift);
        same.assert_eq(next.shift, local.shift);
        same.assert_eq(next.coordinate, local.coordinate + AB::F::ONE);
        same.assert_eq(next.tidx, local.tidx);
        same.assert_eq(next.sample, local.sample);
        same.assert_eq(next.quotient, local.quotient);
        same.assert_eq(next.index, local.index);
        same.assert_eq(local.power, next.power * AB::F::TWO);
        same.assert_eq(next.reconstructed_before, local.reconstructed_after);
        let mut transition = builder.when_transition();
        let mut next_shift = transition.when(next.active * next.is_first);
        next_shift.assert_one(local.is_last);
        next_shift
            .when(next.is_first_shift)
            .assert_eq(next.proof_idx, local.proof_idx + AB::F::ONE);
        next_shift
            .when(AB::Expr::ONE - next.is_first_shift)
            .assert_eq(next.proof_idx, local.proof_idx);
        next_shift
            .when(AB::Expr::ONE - next.is_first_shift)
            .assert_eq(next.shift, local.shift + AB::F::ONE);
        builder
            .when(local.active * local.is_last)
            .assert_eq(local.reconstructed_after, local.index);

        self.inner.transcript_bus.sample(
            builder,
            local.proof_idx,
            local.tidx,
            local.sample,
            local.active * local.is_first,
        );
        self.inner.right_shift_bus.lookup_key(
            builder,
            RightShiftMessage {
                input: local.sample.into(),
                shift_bits: AB::Expr::from_usize(self.inner.log_codeword_len),
                result: local.quotient.into(),
            },
            local.active * local.is_first,
        );
        self.inner.exp_bits_len_bus.lookup_key(
            builder,
            ExpBitsLenMessage {
                base: AB::Expr::ONE,
                bit_src: local.sample.into(),
                num_bits: AB::Expr::ZERO,
                result: AB::Expr::ONE,
            },
            local.active * local.is_first,
        );
        builder.when(local.active * local.is_first).assert_eq(
            local.sample,
            local.index + local.quotient * AB::Expr::from_u32(1u32 << self.inner.log_codeword_len),
        );
        self.inner.shift_index_bus.add_key_with_lookups(
            builder,
            NativeShiftIndexMessage {
                proof_idx: local.proof_idx.into(),
                shift: local.shift.into(),
                index: local.index.into(),
            },
            local.index_lookup_count,
        );
        builder
            .when(local.active)
            .assert_eq(local.value[0], local.bit);
        for limb in &local.value[1..] {
            builder.when(local.active).assert_zero(*limb);
        }
        if let Some(opening_bus) = self.inner.opening_bus {
            opening_bus.send(
                builder,
                NativeOpeningClaimMessage {
                    proof_idx: local.proof_idx.into(),
                    claim: AB::Expr::from_usize(self.inner.opening_claim_offset) + local.shift,
                    section: AB::Expr::from_usize(OPENING_SECTION_POINT),
                    coordinate: local.coordinate.into(),
                    value: local.value.map(Into::into),
                },
                local.active,
            );
        }
        self.role_bus.receive(
            builder,
            NativeVaccTranscriptRoleMessage {
                proof_idx: local.proof_idx.into(),
                role: AB::Expr::from_usize(VACC_ROLE_SHIFT),
                ordinal: local.shift.into(),
                tidx: local.tidx.into(),
                value: [
                    local.sample.into(),
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                ],
                is_ext: AB::Expr::ZERO,
                is_sample: AB::Expr::ONE,
            },
            local.active * local.is_first,
        );
    }
}

pub struct DirectAirVaccBatchingFinalAirV19 {
    pub inner: NativeBatchingFinalAir,
    pub role_bus: NativeVaccTranscriptRoleBus,
}

impl openvm_stark_backend::BaseAirWithPublicValues<F> for DirectAirVaccBatchingFinalAirV19 {}
impl openvm_stark_backend::PartitionedBaseAir<F> for DirectAirVaccBatchingFinalAirV19 {}
impl BaseAir<F> for DirectAirVaccBatchingFinalAirV19 {
    fn width(&self) -> usize {
        NativeBatchingFinalCols::<F>::width()
    }
}
impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for DirectAirVaccBatchingFinalAirV19
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield:
        openvm_stark_backend::p3_field::extension::BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        self.inner.eval(builder);
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("direct VACC batching-final role row");
        let local: &NativeBatchingFinalCols<AB::Var> = (*row).borrow();
        self.role_bus.receive(
            builder,
            NativeVaccTranscriptRoleMessage {
                proof_idx: local.proof_idx.into(),
                role: AB::Expr::from_usize(VACC_ROLE_MU),
                ordinal: AB::Expr::ZERO,
                tidx: local.mu_tidx.into(),
                value: local.mu.map(Into::into),
                is_ext: AB::Expr::ONE,
                is_sample: AB::Expr::ZERO,
            },
            local.active * local.is_last,
        );
    }
}

/// Caller-owned buses which connect this verifier to the v19 History
/// producer. For a multi-group composition these buses are group-private
/// adapter inputs: they must not be shared across groups until a certified
/// `(group_id, local_proof_idx) -> history_proof_idx` remap has been applied.
#[derive(Clone, Copy, Debug)]
pub struct DirectAirVaccHistoryBusesV19 {
    pub context: DirectAirVaccContextBusV19,
    pub input: CertifiedDirectAirVaccInputBusV19,
}

/// Tree identifiers used by the ordinary scalar-codeword authentication.
/// Fresh and prior codewords have independent outer trees and disjoint inner
/// row-tree ranges; no stacked-PCS descriptor is involved.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectAirVaccTreeLayoutV19 {
    pub fresh_outer: usize,
    pub fresh_rows: usize,
    pub prior_outer: usize,
    pub prior_rows: usize,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct DirectAirExactFiniteVaccAuthorityColsV19<T> {
    pub active: T,
    pub start_tidx: T,
    pub start_sample_count: T,
    pub start_state: [T; POSEIDON2_WIDTH],
    pub end_tidx: T,
    pub end_sample_count: T,
    pub end_state: [T; POSEIDON2_WIDTH],
    pub fresh_root: [T; DIGEST_SIZE],
    pub prior_root: [T; DIGEST_SIZE],
    pub output_root: [T; DIGEST_SIZE],
    pub prior_digest: [T; DIGEST_SIZE],
    pub output_digest: [T; DIGEST_SIZE],
}

/// Final adapter inside the complete exact-finite verifier package. Every
/// dynamic value is consumed from a lower verifier AIR before one authority
/// tuple is published to the minimal finite-v3 wrapper.
#[derive(ColumnsAir)]
#[columns_via(DirectAirExactFiniteVaccAuthorityColsV19<u8>)]
pub struct DirectAirExactFiniteVaccAuthorityAirV19 {
    pub exact: DirectAirExactFiniteVaccConfigV19,
    pub protocol_bus: NativeStandardVaccProtocolBus,
    pub end_bus: NativeStandardVaccEndBus,
    pub root_bus: NativeStandardVaccRootBus,
    pub digest_bus: NativeStandardVaccDigestBus,
    pub checkpoint_bus: CertifiedTranscriptCheckpointBus,
    pub resume_bus: ResumeTranscriptStateBus,
}

impl BaseAir<F> for DirectAirExactFiniteVaccAuthorityAirV19 {
    fn width(&self) -> usize {
        DirectAirExactFiniteVaccAuthorityColsV19::<F>::width()
    }
}
impl openvm_stark_backend::BaseAirWithPublicValues<F> for DirectAirExactFiniteVaccAuthorityAirV19 {}
impl openvm_stark_backend::PartitionedBaseAir<F> for DirectAirExactFiniteVaccAuthorityAirV19 {}

impl<AB> Air<AB> for DirectAirExactFiniteVaccAuthorityAirV19
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(
            self.exact.validate().is_ok(),
            "invalid exact-finite authority key"
        );
        let call = self.exact.call();
        let proof_idx = AB::Expr::from_usize(self.exact.call_index);
        let has_prior = AB::Expr::from_usize(call.prior_count);
        let main = builder.main();
        let row = main.row_slice(0).expect("exact-finite VACC authority row");
        let local: &DirectAirExactFiniteVaccAuthorityColsV19<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        let enabled = local.active;

        self.exact.call_protocol_bus.lookup_key(
            builder,
            NativeExactFiniteVaccCallProtocolMessage {
                proof_idx: proof_idx.clone(),
                start_tidx: local.start_tidx.into(),
                call_index: AB::Expr::from_usize(self.exact.call_index),
                input_arity: AB::Expr::from_usize(call.input_arity),
                fresh_count: AB::Expr::from_usize(call.fresh_count),
                prior_count: has_prior.clone(),
                relation_digest: self
                    .exact
                    .transcript_profile
                    .relation_digest
                    .map(AB::Expr::from),
                index_digest: self
                    .exact
                    .transcript_profile
                    .index_digest
                    .map(AB::Expr::from),
                setup_digest: self
                    .exact
                    .transcript_profile
                    .setup_digest
                    .map(AB::Expr::from),
                schedule_digest: self
                    .exact
                    .transcript_profile
                    .schedule_digest
                    .map(AB::Expr::from),
            },
            enabled,
        );
        self.exact.schedule_bus.lookup_key(
            builder,
            NativeExactFiniteVaccScheduleMessage {
                proof_idx: AB::Expr::ZERO,
                end_tidx: local.start_tidx.into(),
                call_count: AB::Expr::from_usize(self.exact.transcript_profile.active_call_count()),
                total_fresh: AB::Expr::from_usize(self.exact.transcript_profile.total_fresh()),
                relation_digest: self
                    .exact
                    .transcript_profile
                    .relation_digest
                    .map(AB::Expr::from),
                index_digest: self
                    .exact
                    .transcript_profile
                    .index_digest
                    .map(AB::Expr::from),
                setup_digest: self
                    .exact
                    .transcript_profile
                    .setup_digest
                    .map(AB::Expr::from),
                schedule_digest: self
                    .exact
                    .transcript_profile
                    .schedule_digest
                    .map(AB::Expr::from),
            },
            AB::Expr::from(enabled) * AB::Expr::from_bool(self.exact.call_index == 0),
        );
        self.protocol_bus.lookup_key(
            builder,
            NativeStandardVaccProtocolMessage {
                proof_idx: proof_idx.clone(),
                start_tidx: local.start_tidx.into(),
                segment_index_lo: AB::Expr::from_usize(self.exact.call_index),
                segment_index_hi: AB::Expr::ZERO,
                has_prior: has_prior.clone(),
                relation_digest: self
                    .exact
                    .transcript_profile
                    .relation_digest
                    .map(AB::Expr::from),
            },
            enabled,
        );
        self.end_bus.lookup_key(
            builder,
            NativeStandardVaccEndMessage {
                proof_idx: proof_idx.clone(),
                end_tidx: local.end_tidx.into(),
            },
            enabled,
        );
        for (kind, root, multiplicity) in [
            (0usize, local.fresh_root, AB::Expr::from(enabled)),
            (
                1usize,
                local.prior_root,
                AB::Expr::from(enabled) * has_prior.clone(),
            ),
            (2usize, local.output_root, AB::Expr::from(enabled)),
        ] {
            self.root_bus.lookup_key(
                builder,
                NativeStandardVaccRootMessage {
                    proof_idx: proof_idx.clone(),
                    kind: AB::Expr::from_usize(kind),
                    root: root.map(Into::into),
                },
                multiplicity,
            );
        }
        // The stacked commitment AIR produces the ordinary fresh-root key.
        // Republish only the setup-fixed extra copies after this authority
        // row has consumed and authenticated that ordinary key.
        self.root_bus.add_key_with_lookups(
            builder,
            NativeStandardVaccRootMessage {
                proof_idx: proof_idx.clone(),
                kind: AB::Expr::ZERO,
                root: local.fresh_root.map(Into::into),
            },
            AB::Expr::from(enabled)
                * AB::Expr::from_usize(
                    self.exact
                        .multiplicities
                        .fresh_root_lookup_count_per_call
                        .saturating_sub(1),
                ),
        );
        for (state, digest, multiplicity) in [
            (
                0usize,
                local.prior_digest,
                AB::Expr::from(enabled) * has_prior.clone(),
            ),
            (1usize, local.output_digest, AB::Expr::from(enabled)),
        ] {
            self.digest_bus.lookup_key(
                builder,
                NativeStandardVaccDigestMessage {
                    proof_idx: proof_idx.clone(),
                    state: AB::Expr::from_usize(state),
                    digest: digest.map(Into::into),
                },
                multiplicity,
            );
        }
        self.checkpoint_bus.receive(
            builder,
            proof_idx.clone(),
            CertifiedTranscriptCheckpointMessage {
                kind: AB::Expr::ONE,
                tidx: local.end_tidx.into(),
                sample_count: local.end_sample_count.into(),
                state: local.end_state.map(Into::into),
            },
            enabled,
        );
        self.resume_bus.send(
            builder,
            proof_idx.clone(),
            ResumeTranscriptStateMessage {
                tidx: local.start_tidx.into(),
                state: local.start_state.map(Into::into),
            },
            AB::Expr::from(enabled) * AB::Expr::from_bool(self.exact.call_index != 0),
        );
        if self.exact.call_index == 0 {
            builder.when(enabled).assert_zero(local.start_sample_count);
            for limb in local.start_state {
                builder.when(enabled).assert_zero(limb);
            }
        }
        for limb in 0..DIGEST_SIZE {
            builder
                .when(AB::Expr::from(enabled) * (AB::Expr::ONE - has_prior.clone()))
                .assert_zero(local.prior_root[limb]);
            builder
                .when(AB::Expr::from(enabled) * (AB::Expr::ONE - has_prior.clone()))
                .assert_zero(local.prior_digest[limb]);
        }

        self.exact.authority_bus.add_key_with_lookups(
            builder,
            FiniteWarpV3ExactVaccAuthorityMessage {
                proof_idx: proof_idx.clone(),
                call_index: AB::Expr::from_usize(self.exact.call_index),
                source_start: AB::Expr::from_usize(self.exact.source_start()),
                source_count: AB::Expr::from_usize(call.fresh_count),
                input_arity: AB::Expr::from_usize(call.input_arity),
                has_prior,
                transcript_version: AB::Expr::from_u64(EXACT_FINITE_WARP_TRANSCRIPT_VERSION),
                protocol_digest: self.exact.protocol_digest.map(AB::Expr::from),
                relation_digest: self
                    .exact
                    .transcript_profile
                    .relation_digest
                    .map(AB::Expr::from),
                warp_index_digest: self
                    .exact
                    .transcript_profile
                    .index_digest
                    .map(AB::Expr::from),
                setup_digest: self
                    .exact
                    .transcript_profile
                    .setup_digest
                    .map(AB::Expr::from),
                schedule_digest: self
                    .exact
                    .transcript_profile
                    .schedule_digest
                    .map(AB::Expr::from),
                start_tidx: local.start_tidx.into(),
                start_sample_count: local.start_sample_count.into(),
                start_state: local.start_state.map(Into::into),
                end_tidx: local.end_tidx.into(),
                end_sample_count: local.end_sample_count.into(),
                end_state: local.end_state.map(Into::into),
                fresh_stacked_root: local.fresh_root.map(Into::into),
                prior_accumulator_root: local.prior_root.map(Into::into),
                output_accumulator_root: local.output_root.map(Into::into),
                prior_accumulator_digest: local.prior_digest.map(Into::into),
                output_accumulator_digest: local.output_digest.map(Into::into),
            },
            enabled,
        );
    }
}

fn generate_direct_air_exact_finite_vacc_authority_trace_v19(
    record: &DirectAirExactFiniteVaccVerifierRecordV19<'_>,
    fresh_root: Digest,
    prior_digest: Option<Digest>,
    output_digest: Digest,
) -> RowMajorMatrix<F> {
    let width = DirectAirExactFiniteVaccAuthorityColsV19::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut DirectAirExactFiniteVaccAuthorityColsV19<F> = values.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.start_tidx = F::from_u32(record.start_checkpoint.operation_index);
    cols.start_sample_count = F::from_u8(record.start_checkpoint.sample_count);
    cols.start_state = record.start_checkpoint.state;
    cols.end_tidx = F::from_u32(record.end_checkpoint.operation_index);
    cols.end_sample_count = F::from_u8(record.end_checkpoint.sample_count);
    cols.end_state = record.end_checkpoint.state;
    cols.fresh_root = fresh_root;
    cols.prior_root = record
        .prior
        .map_or([F::ZERO; DIGEST_SIZE], |prior| prior.rt);
    cols.output_root = record.verification.output_instance.rt;
    cols.prior_digest = prior_digest.unwrap_or([F::ZERO; DIGEST_SIZE]);
    cols.output_digest = output_digest;
    RowMajorMatrix::new(values, width)
}

/// Verifier-key shape for the reduced-SWIRL original-root source lane. Every
/// native call uses this one fixed arity. Runtime `fresh_count` is constrained
/// by the outer reduced schedule and may vary only through the active prefix;
/// the relation never becomes keyed by an individual call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectAirReducedSwirlVaccConfigV19 {
    pub input_arity: usize,
    pub max_roots_per_source: usize,
    pub projection_sources_per_shard: usize,
    pub max_projection_height: usize,
}

impl DirectAirReducedSwirlVaccConfigV19 {
    pub fn validate(&self) -> Result<(), DirectAirVaccVerifierErrorV19> {
        if self.input_arity < 2
            || self.input_arity > 64
            || !self.input_arity.is_power_of_two()
            || self.max_roots_per_source == 0
            || self.projection_sources_per_shard == 0
            || self.projection_sources_per_shard > self.input_arity
            || self.max_projection_height == 0
            || !self.max_projection_height.is_power_of_two()
        {
            return Err(DirectAirVaccVerifierErrorV19::InvalidProfile);
        }
        Ok(())
    }
}

/// Complete verifier-key composition for every standard direct-AIR VACC step
/// with one numeric shape and prior-state mode. Relation-specific canonical
/// descriptions live in one fixed event catalog consumed by one ragged proof
/// stream; the expensive algebra/Merkle verifier bundle is instantiated once
/// for the whole shape.
pub struct DirectAirVaccVerifierModuleV19 {
    pub profile: NativeStandardVaccShapeProfile,
    pub relations: Vec<NativeStandardVaccRelationProfile>,
    prefix_events: Arc<[NativeStandardVaccPrefixEventV19]>,
    prefix_event_bus: NativeStandardVaccPrefixEventBusV19,
    pub include_prior: bool,
    pub warp_step_mode: DirectAirVaccWarpStepModeV19,
    pub fresh_commitment_mode: DirectAirVaccFreshCommitmentModeV19,
    pub shared: BusInventory,
    pub buses: NativeWarpPcdBusInventory,
    pub history_buses: DirectAirVaccHistoryBusesV19,
    pub protocol_bus: NativeStandardVaccProtocolBus,
    pub end_bus: NativeStandardVaccEndBus,
    pub statement_root_bus: NativeStandardVaccRootBus,
    pub statement_digest_bus: NativeStandardVaccDigestBus,
    pub algebra: NativeWarpAlgebraLayout,
    pub private_accumulator: NativePrivateAccumulatorLayout,
    pub trees: DirectAirVaccTreeLayoutV19,
    pub transcript: NativeWarpTranscriptModule,
    /// Setup-fixed mode used by recursive History. The source verifier
    /// authenticates the start state on a private resume bus, so this module
    /// proves only the VACC suffix and does not replay the source prefix.
    pub resume_from_start_checkpoint: bool,
    /// Number of rows in the vector-alphabet RS oracle. The flattened WARP
    /// codeword length remains `2^profile.log_codeword_len`.
    pub oracle_height: usize,
    pub query_count: usize,
    /// Number of setup-fixed consumers of the certified fresh VACC input.
    /// Ordinary v19 replay uses one; verifier-WARP v2 adds one beta-layout
    /// adapter and therefore sets this to two before keygen.
    pub vacc_input_lookup_count: usize,
    /// Present only for the exact finite verifier-WARP wrapper. Legacy v19
    /// History modules keep this `None` and retain byte-for-byte AIR order.
    pub exact_finite: Option<DirectAirExactFiniteVaccConfigV19>,
    /// Setup-fixed reduced-SWIRL mode. This is mutually exclusive with the
    /// exact-finite per-call specialization.
    pub reduced_swirl: Option<DirectAirReducedSwirlVaccConfigV19>,
}

impl DirectAirVaccVerifierModuleV19 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        profile: NativeStandardVaccProfile,
        include_prior: bool,
        shared: BusInventory,
        buses: NativeWarpPcdBusInventory,
        history_buses: DirectAirVaccHistoryBusesV19,
        protocol_bus: NativeStandardVaccProtocolBus,
        end_bus: NativeStandardVaccEndBus,
        statement_root_bus: NativeStandardVaccRootBus,
        statement_digest_bus: NativeStandardVaccDigestBus,
        system_params: SystemParams,
    ) -> Result<Self, DirectAirVaccVerifierErrorV19> {
        profile
            .validate()
            .map_err(|_| DirectAirVaccVerifierErrorV19::InvalidProfile)?;
        Self::new_shape_batched(
            vec![profile],
            include_prior,
            shared,
            buses,
            history_buses,
            protocol_bus,
            end_bus,
            statement_root_bus,
            statement_digest_bus,
            system_params,
        )
    }

    /// Build one heavy verifier bundle for a numeric WARP shape. Every
    /// relation contributes verifier-fixed rows to one prefix catalog; one
    /// proof-event stream absorbs the selected relation's exact canonical
    /// bytes and exports its digest on the proof-indexed protocol bus.
    #[allow(clippy::too_many_arguments)]
    pub fn new_shape_batched(
        profiles: Vec<NativeStandardVaccProfile>,
        include_prior: bool,
        shared: BusInventory,
        buses: NativeWarpPcdBusInventory,
        history_buses: DirectAirVaccHistoryBusesV19,
        protocol_bus: NativeStandardVaccProtocolBus,
        end_bus: NativeStandardVaccEndBus,
        statement_root_bus: NativeStandardVaccRootBus,
        statement_digest_bus: NativeStandardVaccDigestBus,
        system_params: SystemParams,
    ) -> Result<Self, DirectAirVaccVerifierErrorV19> {
        Self::new_shape_batched_with_fresh_commitment(
            profiles,
            include_prior,
            DirectAirVaccFreshCommitmentModeV19::ScalarMerkle,
            false,
            shared,
            buses,
            history_buses,
            protocol_bus,
            end_bus,
            statement_root_bus,
            statement_digest_bus,
            system_params,
        )
    }

    /// Standard all-extension VACC verifier resumed from the source verifier's
    /// certified transcript checkpoint. This is the fixed-HLeaf production
    /// mode: it changes only how the initial transcript state is supplied and
    /// retains the ordinary scalar-EF Merkle commitment alphabet and protocol
    /// tag.
    #[allow(clippy::too_many_arguments)]
    pub fn new_shape_batched_resumed(
        profiles: Vec<NativeStandardVaccProfile>,
        include_prior: bool,
        shared: BusInventory,
        buses: NativeWarpPcdBusInventory,
        history_buses: DirectAirVaccHistoryBusesV19,
        protocol_bus: NativeStandardVaccProtocolBus,
        end_bus: NativeStandardVaccEndBus,
        statement_root_bus: NativeStandardVaccRootBus,
        statement_digest_bus: NativeStandardVaccDigestBus,
        system_params: SystemParams,
    ) -> Result<Self, DirectAirVaccVerifierErrorV19> {
        Self::new_shape_batched_with_fresh_commitment(
            profiles,
            include_prior,
            DirectAirVaccFreshCommitmentModeV19::ScalarMerkle,
            true,
            shared,
            buses,
            history_buses,
            protocol_bus,
            end_bus,
            statement_root_bus,
            statement_digest_bus,
            system_params,
        )
    }

    /// Build the standard VACC algebra with CUDA shared-forest source
    /// authentication supplied by a sibling verifier module.  This changes
    /// only the verifier-key commitment lane: all WARP challenges, sumchecks,
    /// prior authentication and output binding remain the standard protocol.
    #[allow(clippy::too_many_arguments)]
    pub fn new_shape_batched_cuda_shared_forest(
        profiles: Vec<NativeStandardVaccProfile>,
        include_prior: bool,
        shared: BusInventory,
        buses: NativeWarpPcdBusInventory,
        history_buses: DirectAirVaccHistoryBusesV19,
        protocol_bus: NativeStandardVaccProtocolBus,
        end_bus: NativeStandardVaccEndBus,
        statement_root_bus: NativeStandardVaccRootBus,
        statement_digest_bus: NativeStandardVaccDigestBus,
        system_params: SystemParams,
    ) -> Result<Self, DirectAirVaccVerifierErrorV19> {
        Self::new_shape_batched_with_fresh_commitment(
            profiles,
            include_prior,
            DirectAirVaccFreshCommitmentModeV19::CudaSharedForest,
            false,
            shared,
            buses,
            history_buses,
            protocol_bus,
            end_bus,
            statement_root_bus,
            statement_digest_bus,
            system_params,
        )
    }

    /// Build the standard EF4 VACC algebra with a genuine BabyBear fresh
    /// commitment lane as specified by Appendix D. Prior and output
    /// accumulators remain ordinary EF4 Merkle codewords.
    #[allow(clippy::too_many_arguments)]
    pub fn new_shape_batched_appendix_d(
        profiles: Vec<NativeStandardVaccProfile>,
        include_prior: bool,
        shared: BusInventory,
        buses: NativeWarpPcdBusInventory,
        history_buses: DirectAirVaccHistoryBusesV19,
        protocol_bus: NativeStandardVaccProtocolBus,
        end_bus: NativeStandardVaccEndBus,
        statement_root_bus: NativeStandardVaccRootBus,
        statement_digest_bus: NativeStandardVaccDigestBus,
        system_params: SystemParams,
    ) -> Result<Self, DirectAirVaccVerifierErrorV19> {
        Self::new_shape_batched_with_fresh_commitment(
            profiles,
            include_prior,
            DirectAirVaccFreshCommitmentModeV19::AppendixDBase,
            false,
            shared,
            buses,
            history_buses,
            protocol_bus,
            end_bus,
            statement_root_bus,
            statement_digest_bus,
            system_params,
        )
    }

    /// Appendix-D verifier whose transcript starts from a checkpoint
    /// authenticated by a sibling complete-source verifier AIR.
    #[allow(clippy::too_many_arguments)]
    pub fn new_shape_batched_appendix_d_resumed(
        profiles: Vec<NativeStandardVaccProfile>,
        include_prior: bool,
        shared: BusInventory,
        buses: NativeWarpPcdBusInventory,
        history_buses: DirectAirVaccHistoryBusesV19,
        protocol_bus: NativeStandardVaccProtocolBus,
        end_bus: NativeStandardVaccEndBus,
        statement_root_bus: NativeStandardVaccRootBus,
        statement_digest_bus: NativeStandardVaccDigestBus,
        system_params: SystemParams,
    ) -> Result<Self, DirectAirVaccVerifierErrorV19> {
        Self::new_shape_batched_with_fresh_commitment(
            profiles,
            include_prior,
            DirectAirVaccFreshCommitmentModeV19::AppendixDBase,
            true,
            shared,
            buses,
            history_buses,
            protocol_bus,
            end_bus,
            statement_root_bus,
            statement_digest_bus,
            system_params,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_shape_batched_with_fresh_commitment(
        profiles: Vec<NativeStandardVaccProfile>,
        include_prior: bool,
        fresh_commitment_mode: DirectAirVaccFreshCommitmentModeV19,
        resume_from_start_checkpoint: bool,
        shared: BusInventory,
        buses: NativeWarpPcdBusInventory,
        history_buses: DirectAirVaccHistoryBusesV19,
        protocol_bus: NativeStandardVaccProtocolBus,
        end_bus: NativeStandardVaccEndBus,
        statement_root_bus: NativeStandardVaccRootBus,
        statement_digest_bus: NativeStandardVaccDigestBus,
        system_params: SystemParams,
    ) -> Result<Self, DirectAirVaccVerifierErrorV19> {
        let profile = profiles
            .first()
            .ok_or(DirectAirVaccVerifierErrorV19::InvalidProfile)?
            .shape_profile();
        if profiles
            .iter()
            .any(|candidate| candidate.validate().is_err() || candidate.shape_profile() != profile)
            || profiles.iter().enumerate().any(|(index, candidate)| {
                profiles[..index]
                    .iter()
                    .any(|prior| prior.relation_digest() == candidate.relation_digest())
            })
        {
            return Err(DirectAirVaccVerifierErrorV19::InvalidProfile);
        }
        let relations = profiles
            .iter()
            .map(NativeStandardVaccProfile::relation_profile)
            .collect::<Vec<_>>();
        let prefix_events = relations
            .iter()
            .map(|relation| {
                standard_vacc_prefix_events_v19(
                    &profile,
                    relation,
                    include_prior,
                    fresh_commitment_mode,
                )
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();
        if profile.log_codeword_len > MAX_RAW_MESSAGE_POINT_LEN_V19
            || profile.beta_len > MAX_FRESH_BETA_LEN_V19
        {
            return Err(DirectAirVaccVerifierErrorV19::InvalidProfile);
        }
        let codeword_len = 1usize
            .checked_shl(profile.log_codeword_len as u32)
            .ok_or(DirectAirVaccVerifierErrorV19::InvalidProfile)?;
        let alphabet_width = 1usize
            .checked_shl(profile.initial_folding_factor as u32)
            .ok_or(DirectAirVaccVerifierErrorV19::InvalidProfile)?;
        let oracle_height = codeword_len
            .checked_div(alphabet_width)
            .ok_or(DirectAirVaccVerifierErrorV19::InvalidProfile)?;
        let query_count = oracle_height
            .checked_div(profile.rows_per_query)
            .filter(|count| count.is_power_of_two())
            .ok_or(DirectAirVaccVerifierErrorV19::InvalidProfile)?;
        let fresh_outer = 0;
        let fresh_rows = fresh_outer + 1;
        let prior_outer = fresh_rows + query_count;
        let prior_rows = prior_outer + 1;
        let family = family_from_profile(&profile);
        let algebra = NativeWarpAlgebraLayout::new(&family);
        let private_accumulator =
            NativePrivateAccumulatorLayout::new(0, profile.log_codeword_len, profile.beta_len);
        let transcript = NativeWarpTranscriptModule::new_with_certified_checkpoints(
            &shared,
            &buses,
            system_params,
            false,
            resume_from_start_checkpoint,
        );
        Ok(Self {
            profile,
            relations,
            prefix_events: prefix_events.into(),
            // `NativeWarpPcdBusInventory` owns a contiguous namespace. Group
            // allocation reserves its first free index for this catalog bus.
            prefix_event_bus: NativeStandardVaccPrefixEventBusV19::new(buses.next_bus_idx()),
            include_prior,
            warp_step_mode: DirectAirVaccWarpStepModeV19::LegacySegmentIndex,
            fresh_commitment_mode,
            shared,
            buses,
            history_buses,
            protocol_bus,
            end_bus,
            statement_root_bus,
            statement_digest_bus,
            algebra,
            private_accumulator,
            trees: DirectAirVaccTreeLayoutV19 {
                fresh_outer,
                fresh_rows,
                prior_outer,
                prior_rows,
            },
            transcript,
            resume_from_start_checkpoint,
            oracle_height,
            query_count,
            vacc_input_lookup_count: 1,
            exact_finite: None,
            reduced_swirl: None,
        })
    }

    /// Construct one complete ordinary WARP `Verify` authority for a
    /// setup-fixed call of the exact-finite schedule. Call zero verifies the
    /// schedule prefix and the first invocation from the canonical transcript
    /// state; later calls resume from the preceding certified checkpoint.
    ///
    /// This is not a History replay and does not run terminal `Decide`.
    #[allow(clippy::too_many_arguments)]
    pub fn new_exact_finite_call(
        exact: DirectAirExactFiniteVaccConfigV19,
        shared: BusInventory,
        buses: NativeWarpPcdBusInventory,
        history_buses: DirectAirVaccHistoryBusesV19,
        protocol_bus: NativeStandardVaccProtocolBus,
        end_bus: NativeStandardVaccEndBus,
        statement_root_bus: NativeStandardVaccRootBus,
        statement_digest_bus: NativeStandardVaccDigestBus,
        system_params: SystemParams,
    ) -> Result<Self, DirectAirVaccVerifierErrorV19> {
        exact.validate()?;
        let call = exact.call();
        let standard = NativeStandardVaccProfile::from_exact_finite_transcript_profile(
            &exact.transcript_profile,
        )
        .map_err(|_| DirectAirVaccVerifierErrorV19::InvalidProfile)?;
        let include_prior = call.prior_count == 1;
        let mut module = Self::new_shape_batched_with_fresh_commitment(
            vec![standard],
            include_prior,
            DirectAirVaccFreshCommitmentModeV19::AppendixDBase,
            exact.call_index != 0,
            shared,
            buses,
            history_buses,
            protocol_bus,
            end_bus,
            statement_root_bus,
            statement_digest_bus,
            system_params,
        )?;
        module.fresh_commitment_mode = DirectAirVaccFreshCommitmentModeV19::FiniteStackedBase;
        let family = family_from_shape_profile(&module.profile, call.input_arity, call.fresh_count);
        module.algebra = NativeWarpAlgebraLayout::new(&family);

        // The shared source tree uses one outer tree and one small row tree per
        // sampled shift. Keep the prior accumulator forest disjoint.
        module.trees = DirectAirVaccTreeLayoutV19 {
            fresh_outer: 0,
            fresh_rows: 1,
            prior_outer: 1 + module.profile.num_shift_queries,
            prior_rows: 2 + module.profile.num_shift_queries,
        };
        module.exact_finite = Some(exact);
        Ok(module)
    }

    /// Construct one verifier-key-stable reduced-SWIRL VACC verifier. Unlike
    /// `new_exact_finite_call`, this constructor is independent of call index,
    /// fresh count, and bootstrap/continuation mode. Those values are supplied
    /// by constrained active-prefix records on shared typed buses.
    #[allow(clippy::too_many_arguments)]
    pub fn new_reduced_swirl_batched(
        profile: NativeStandardVaccProfile,
        reduced: DirectAirReducedSwirlVaccConfigV19,
        shared: BusInventory,
        buses: NativeWarpPcdBusInventory,
        history_buses: DirectAirVaccHistoryBusesV19,
        protocol_bus: NativeStandardVaccProtocolBus,
        end_bus: NativeStandardVaccEndBus,
        statement_root_bus: NativeStandardVaccRootBus,
        statement_digest_bus: NativeStandardVaccDigestBus,
        system_params: SystemParams,
    ) -> Result<Self, DirectAirVaccVerifierErrorV19> {
        reduced.validate()?;
        let mut module = Self::new_shape_batched_with_fresh_commitment(
            vec![profile],
            true,
            DirectAirVaccFreshCommitmentModeV19::CudaSharedForest,
            true,
            shared,
            buses,
            history_buses,
            protocol_bus,
            end_bus,
            statement_root_bus,
            statement_digest_bus,
            system_params,
        )?;
        let family =
            family_from_shape_profile(&module.profile, reduced.input_arity, reduced.input_arity);
        module.algebra = NativeWarpAlgebraLayout::new(&family);
        let fresh_tree_count = reduced
            .input_arity
            .checked_mul(reduced.max_roots_per_source)
            .and_then(|count| count.checked_mul(1 + module.profile.num_shift_queries))
            .ok_or(DirectAirVaccVerifierErrorV19::InvalidProfile)?;
        module.trees = DirectAirVaccTreeLayoutV19 {
            fresh_outer: 0,
            fresh_rows: 1,
            prior_outer: fresh_tree_count,
            prior_rows: fresh_tree_count + 1,
        };
        module.reduced_swirl = Some(reduced);
        Ok(module)
    }

    fn exact_config(
        &self,
    ) -> Result<&DirectAirExactFiniteVaccConfigV19, DirectAirVaccVerifierErrorV19> {
        self.exact_finite
            .as_ref()
            .ok_or(DirectAirVaccVerifierErrorV19::InvalidProfile)
    }

    fn input_arity(&self) -> usize {
        self.reduced_swirl.as_ref().map_or_else(
            || {
                self.exact_finite
                    .as_ref()
                    .map_or(STANDARD_DIRECT_VACC_INPUT_ARITY, |exact| {
                        exact.call().input_arity
                    })
            },
            |reduced| reduced.input_arity,
        )
    }

    fn prior_claim_source(&self) -> usize {
        self.input_arity() - 1
    }

    fn fresh_count(&self) -> usize {
        if let Some(reduced) = self.reduced_swirl.as_ref() {
            return reduced.input_arity;
        }
        self.exact_finite
            .as_ref()
            .map_or(1, |exact| exact.call().fresh_count)
    }

    /// Configure the exact number of consumers of the certified fresh input.
    /// This is verification-key material, not a runtime witness choice.
    pub fn set_vacc_input_lookup_count(&mut self, count: usize) {
        assert!(count != 0, "certified VACC input consumer count is zero");
        self.vacc_input_lookup_count = count;
    }

    /// Select the fixed-HLeaf backend schedule. This is verifier-key state and
    /// must be called before AIR construction/keygen. It does not reinterpret
    /// old v19 records or create a neutral prior.
    pub fn enable_fixed_hleaf_warp_step_v4(&mut self) {
        assert!(
            self.include_prior,
            "seeded fixed HLeaf requires the prior-bearing standard VACC relation"
        );
        self.warp_step_mode = DirectAirVaccWarpStepModeV19::FixedHLeafV4;
    }

    /// AIRs in the exact order returned by the trace generator below.
    #[must_use]
    pub fn airs<PCS: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<PCS>> {
        if self.reduced_swirl.is_some() {
            return self.reduced_swirl_airs::<PCS>();
        }
        if self.exact_finite.is_some() {
            return self.exact_finite_airs::<PCS>();
        }
        assert!(
            self.warp_step_mode != DirectAirVaccWarpStepModeV19::FixedHLeafV4 || self.include_prior,
            "seeded fixed HLeaf cannot instantiate a bootstrap VACC package"
        );
        let family = family_from_profile(&self.profile);
        let claim_count = NativeWarpAlgebraLayout::claim_count(&family);
        let authenticated_claim_count = 1 + self.profile.num_ood + self.profile.num_shift_queries;
        let opening_padding_count = claim_count - authenticated_claim_count;
        let log_claim_count = claim_count.ilog2() as usize;
        let constraint_degree = twin_degree(
            self.profile.log_codeword_len,
            self.profile.log_constraints,
            self.profile.max_degree,
        );
        let outer_depth = self.query_count.ilog2() as usize;

        let mut airs = self.transcript.airs::<PCS>();
        add_air(
            &mut airs,
            NativeStandardVaccPrefixCatalogAirV19 {
                events: Arc::clone(&self.prefix_events),
                bus: self.prefix_event_bus,
            },
        );
        add_air(
            &mut airs,
            NativeStandardVaccPrefixStreamAirV19 {
                transcript_bus: self.buses.transcript,
                phase_cursor_bus: self.buses.vacc_phase_cursor,
                protocol_bus: self.protocol_bus,
                event_bus: self.prefix_event_bus,
                include_prior: self.include_prior,
                warp_step_mode: self.warp_step_mode,
                emit_start_boundary: !self.resume_from_start_checkpoint,
            },
        );
        let cursor_lengths = vacc_cursor_role_lengths_v19(
            &self.profile,
            self.include_prior,
            self.fresh_commitment_mode,
        );
        add_air(
            &mut airs,
            DirectAirVaccTranscriptCursorAirV19 {
                transcript_bus: self.buses.transcript,
                semantic_bus: self.buses.vacc_semantic_transcript,
                phase_cursor_bus: self.buses.vacc_phase_cursor,
                role_bus: self.buses.vacc_transcript_role,
                next_roles: next_nonempty_vacc_roles_v19(&cursor_lengths),
                lengths: cursor_lengths,
                twin_event_count: constraint_degree + 2,
                dynamic_role_lengths: false,
                dynamic_optional_prior: false,
                emit_fresh_root_role: true,
            },
        );
        add_air(
            &mut airs,
            NativeStandardVaccEndAir {
                transcript_bus: self.buses.transcript,
                transcript_end_index_bus: self
                    .resume_from_start_checkpoint
                    .then_some(self.shared.transcript_end_index_bus),
                phase_cursor_bus: self.buses.vacc_phase_cursor,
                end_bus: self.end_bus,
                end_tag: SHARD_TRANSCRIPT_VACC_END_TAG_V19,
                extension_tag: false,
            },
        );
        add_air(
            &mut airs,
            DirectAirVaccStatementAirV19 {
                profile: self.profile.clone(),
                protocol_bus: self.protocol_bus,
                end_bus: self.end_bus,
                root_bus: self.statement_root_bus,
                digest_bus: self.statement_digest_bus,
                claim_bus: self.buses.claim_value,
                context_bus: self.history_buses.context,
                vacc_input_bus: self.history_buses.input,
                vacc_input_lookup_count: self.vacc_input_lookup_count,
            },
        );
        add_air(
            &mut airs,
            NativeStandardClaimValueAir {
                claim_bus: self.buses.claim_value,
                transcript_bus: self.buses.vacc_semantic_transcript,
                transcript_role_bus: self.buses.vacc_transcript_role,
                layout_bus: self.buses.claim_layout,
                slot_bus: self.buses.input_slot_layout,
                fresh_claim_consumer_count: 2,
                prior_claim_consumer_count: usize::from(self.include_prior) + 1,
                fresh_alpha_len: self.profile.log_codeword_len,
                fresh_tau_len: self.profile.log_constraints,
                fresh_beta_tail_len: self.profile.beta_len - self.profile.log_constraints,
            },
        );
        add_air(
            &mut airs,
            NativeClaimLayoutAir {
                layout_bus: self.buses.claim_layout,
            },
        );
        add_air(
            &mut airs,
            NativeStandardInputSlotLayoutAir {
                bus: self.buses.input_slot_layout,
                has_prior: self.include_prior,
            },
        );
        if self.fresh_commitment_mode.uses_scalar_merkle() {
            add_air(
                &mut airs,
                NativeStandardCodewordProjectionAir {
                    shift_index_bus: self.buses.shift_index,
                    leaf_value_bus: self.buses.leaf_value,
                    authenticated_bus: self.buses.authenticated_shift,
                    slot_bus: self.buses.input_slot_layout,
                    oracle_height: self.oracle_height,
                    query_count: self.query_count,
                    row_tree_id_offset: self.trees.fresh_rows,
                    source: 0,
                    variant: 1 + usize::from(self.include_prior) * 3,
                    is_fresh: true,
                },
            );
        } else if self.fresh_commitment_mode == DirectAirVaccFreshCommitmentModeV19::AppendixDBase {
            add_air(
                &mut airs,
                NativeAppendixDCodewordProjectionAir {
                    shift_index_bus: self.buses.shift_index,
                    leaf_value_bus: self.buses.leaf_value,
                    authenticated_bus: self.buses.authenticated_shift,
                    slot_bus: self.buses.input_slot_layout,
                    oracle_height: self.oracle_height,
                    query_count: self.query_count,
                    row_tree_id_offset: self.trees.fresh_rows,
                    source: 0,
                    variant: 1 + usize::from(self.include_prior) * 3,
                },
            );
        }
        if self.include_prior {
            add_air(
                &mut airs,
                NativeStandardCodewordProjectionAir {
                    shift_index_bus: self.buses.shift_index,
                    leaf_value_bus: self.buses.leaf_value,
                    authenticated_bus: self.buses.authenticated_shift,
                    slot_bus: self.buses.input_slot_layout,
                    oracle_height: self.oracle_height,
                    query_count: self.query_count,
                    row_tree_id_offset: self.trees.prior_rows,
                    source: 1,
                    variant: 4,
                    is_fresh: false,
                },
            );
        }
        if self.fresh_commitment_mode.uses_merkle_authentication() {
            self.add_root_air(
                &mut airs,
                0,
                self.trees.fresh_outer,
                outer_depth,
                true,
                false,
            );
        }
        if self.include_prior {
            self.add_root_air(
                &mut airs,
                1,
                self.trees.prior_outer,
                outer_depth,
                true,
                true,
            );
        }
        if self.fresh_commitment_mode.uses_merkle_authentication() || self.include_prior {
            add_air(
                &mut airs,
                NativeLeafHashAir {
                    permute_bus: self.shared.poseidon2_permute_bus,
                    value_bus: self.buses.leaf_value,
                    leaf_bus: self.buses.opening_leaf,
                },
            );
            add_air(
                &mut airs,
                NativeMerkleMultiproofAir {
                    compress_bus: self.shared.poseidon2_compress_bus,
                    leaf_bus: self.buses.opening_leaf,
                    node_bus: self.buses.merkle_node,
                    root_bus: self.buses.merkle_root,
                },
            );
            add_air(
                &mut airs,
                NativeMerkleLeafAdapterAir {
                    inner_depth: self.profile.rows_per_query.ilog2() as usize,
                    leaf_bus: self.buses.opening_leaf,
                    root_bus: self.buses.merkle_root,
                },
            );
        }
        let vector_shard_count = self.vector_coordinate_shard_count_v19();
        for proof_slot in 0..vector_shard_count {
            add_air(
                &mut airs,
                DirectAirVaccVectorCoordinateAirV19 {
                    inner: NativeVectorCoordinateAir {
                        vector_bus: self.buses.vector_coordinate,
                        sumcheck_bus: self.buses.sumcheck_challenge,
                        transcript_bus: self.buses.vacc_semantic_transcript,
                        folded_bus: self.buses.folded_claim,
                        opening_bus: self.buses.opening_claim,
                    },
                    role_bus: self.buses.vacc_transcript_role,
                    selector_vector: self.algebra.selector_tau_vector as usize,
                    xi_vector: self.algebra.xi_vector as usize,
                    fixed_proof_slot: (vector_shard_count > 1).then_some(proof_slot),
                },
            );
        }
        for (dimensions, group_offset) in [
            (1usize, 0usize),
            (log_claim_count, 2 * STANDARD_DIRECT_VACC_INPUT_ARITY + 1),
            (
                self.profile.log_codeword_len,
                2 * STANDARD_DIRECT_VACC_INPUT_ARITY + 1 + claim_count,
            ),
        ] {
            add_air(
                &mut airs,
                NativeEqEvaluationAir {
                    result_bus: self.buses.eq_result,
                    vector_bus: self.buses.vector_coordinate,
                    dimensions,
                    group_offset,
                },
            );
        }
        add_air(
            &mut airs,
            DirectAirVaccTwinOmegaAirV19 {
                inner: NativeTwinOmegaAir {
                    transcript_bus: self.buses.vacc_semantic_transcript,
                    omega_bus: self.buses.twin_omega,
                },
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            NativeTwinSigmaAir {
                claim_bus: self.buses.claim_value,
                eq_bus: self.buses.eq_result,
                sumcheck_initial_bus: self.buses.sumcheck_initial,
                input_arity: STANDARD_DIRECT_VACC_INPUT_ARITY,
                omega_bus: self.buses.twin_omega,
            },
        );
        add_air(
            &mut airs,
            DirectAirVaccCoefficientSumcheckAirV19 {
                inner: NativeCoefficientSumcheckAir {
                    transcript_bus: self.buses.vacc_semantic_transcript,
                    round_bus: self.buses.sumcheck_round,
                    initial_bus: self.buses.sumcheck_initial,
                    challenge_bus: self.buses.sumcheck_challenge,
                    twin_degree: constraint_degree,
                    batching_degree: 2,
                    twin_rounds: 1,
                    batching_rounds: self.profile.log_codeword_len,
                },
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            NativeTwinFoldAir {
                claim_bus: self.buses.claim_value,
                eq_bus: self.buses.eq_result,
                folded_bus: self.buses.folded_claim,
                input_arity: STANDARD_DIRECT_VACC_INPUT_ARITY,
                weight_group_offset: STANDARD_DIRECT_VACC_INPUT_ARITY,
            },
        );
        add_air(
            &mut airs,
            DirectAirVaccTwinFinalAirV19 {
                inner: NativeTwinFinalAir {
                    sumcheck_round_bus: self.buses.sumcheck_round,
                    eq_bus: self.buses.eq_result,
                    scalar_bus: self.buses.twin_scalar,
                    last_round: 0,
                    selector_eq_group: self.algebra.selector_at_gamma_group as usize,
                    omega_bus: self.buses.twin_omega,
                    transcript_bus: self.buses.vacc_semantic_transcript,
                },
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            NativeInitialOpeningPointAir {
                folded_bus: self.buses.folded_claim,
                opening_bus: self.buses.opening_claim,
                copies: opening_padding_count + 1,
            },
        );
        add_air(
            &mut airs,
            NativeInitialOpeningTargetAir {
                twin_bus: self.buses.twin_scalar,
                opening_bus: self.buses.opening_claim,
                copies: opening_padding_count + 1,
            },
        );
        add_air(
            &mut airs,
            NativeOpeningPaddingAir {
                opening_bus: self.buses.opening_claim,
            },
        );
        add_air(
            &mut airs,
            DirectAirVaccOodClaimAirV19 {
                inner: NativeOodClaimAir {
                    transcript_bus: self.buses.vacc_semantic_transcript,
                    opening_bus: self.buses.opening_claim,
                },
                role_bus: self.buses.vacc_transcript_role,
                dimension: self.profile.log_codeword_len,
            },
        );
        add_air(
            &mut airs,
            DirectAirVaccShiftScheduleAirV19 {
                inner: NativeShiftScheduleAir {
                    transcript_bus: self.buses.vacc_semantic_transcript,
                    exp_bits_len_bus: self.shared.exp_bits_len_bus,
                    right_shift_bus: self.shared.right_shift_bus,
                    shift_index_bus: self.buses.shift_index,
                    opening_bus: Some(self.buses.opening_claim),
                    log_codeword_len: self.profile.log_codeword_len,
                    opening_claim_offset: 1 + self.profile.num_ood,
                },
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            NativeShiftMergeAir {
                authenticated_bus: self.buses.authenticated_shift,
                eq_bus: self.buses.eq_result,
                opening_bus: self.buses.opening_claim,
                slot_bus: self.buses.input_slot_layout,
                input_arity: STANDARD_DIRECT_VACC_INPUT_ARITY,
                gamma_eq_group_offset: STANDARD_DIRECT_VACC_INPUT_ARITY,
                opening_claim_offset: 1 + self.profile.num_ood,
            },
        );
        add_air(
            &mut airs,
            NativeBatchingSigmaAir {
                eq_bus: self.buses.eq_result,
                opening_bus: self.buses.opening_claim,
                sumcheck_initial_bus: Some(self.buses.sumcheck_initial),
                certified_claim_bus: Some(self.buses.certified_batching_claim),
                claim_count,
                xi_eq_group_offset: self.algebra.xi_weight_groups[0] as usize,
            },
        );
        add_air(
            &mut airs,
            DirectAirVaccBatchingFinalAirV19 {
                inner: NativeBatchingFinalAir {
                    eq_bus: self.buses.eq_result,
                    sumcheck_round_bus: self.buses.sumcheck_round,
                    output_bus: self.buses.batching_output,
                    claim_count,
                    xi_eq_group_offset: self.algebra.xi_weight_groups[0] as usize,
                    point_eq_group_offset: self.algebra.opening_at_alpha_groups[0] as usize,
                    last_round: self.profile.log_codeword_len - 1,
                    transcript_bus: self.buses.vacc_semantic_transcript,
                },
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            NativeBatchingAlphaAir {
                challenge_bus: self.buses.sumcheck_challenge,
                output_bus: self.buses.batching_output,
            },
        );
        self.add_root_air(&mut airs, 2, 0, 0, false, true);
        if self.include_prior {
            self.add_accumulator_airs(&mut airs, NativeAccumulatorBindingMode::Prior);
        }
        self.add_accumulator_airs(&mut airs, NativeAccumulatorBindingMode::Output);
        if !self.resume_from_start_checkpoint {
            add_air(
                &mut airs,
                NativeTranscriptPrefixRemainderAir {
                    transcript_bus: self.buses.transcript,
                    phase_cursor_bus: self.buses.vacc_phase_cursor,
                },
            );
        }
        add_air(
            &mut airs,
            ExpBitsLenAir::new(self.shared.exp_bits_len_bus, self.shared.right_shift_bus),
        );
        airs
    }

    fn exact_finite_fresh_profile(&self) -> NativeFiniteStackedFreshProfile {
        let exact = self
            .exact_finite
            .as_ref()
            .expect("exact-finite profile is verifier-key data");
        let call = exact.call();
        NativeFiniteStackedFreshProfile {
            input_arity: call.input_arity,
            fresh_count: call.fresh_count,
            prior_count: call.prior_count,
            log_message_len: self.profile.log_message_len,
            log_codeword_len: self.profile.log_codeword_len,
            rows_per_leaf: self.profile.rows_per_query,
            outer_tree_id: self.trees.fresh_outer as u32,
            row_tree_id_offset: self.trees.fresh_rows as u32,
        }
    }

    /// Complete AIR inventory for one exact-finite call. It is deliberately
    /// assembled from the same v19 transcript/Merkle/twin/constraint/batching
    /// verifier components as the legacy path, with only setup-fixed arity and
    /// the Construction-10.4 stacked fresh commitment substituted.
    fn exact_finite_airs<PCS: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<PCS>> {
        let exact = self
            .exact_config()
            .expect("invalid exact-finite verifier module");
        let call = exact.call();
        let algebra_profile = exact
            .transcript_profile
            .call_algebra_profile(exact.call_index)
            .expect("invalid exact-finite algebra profile");
        let family = family_from_shape_profile(&self.profile, call.input_arity, call.fresh_count);
        let claim_count = NativeWarpAlgebraLayout::claim_count(&family);
        let authenticated_claim_count = 1 + self.profile.num_ood + self.profile.num_shift_queries;
        let opening_padding_count = claim_count - authenticated_claim_count;
        let log_claim_count = claim_count.ilog2() as usize;
        let outer_depth = self.query_count.ilog2() as usize;
        let fresh = self.exact_finite_fresh_profile();

        let mut airs = self.transcript.airs::<PCS>();
        if exact.call_index == 0 {
            add_air(
                &mut airs,
                NativeExactFiniteVaccSchedulePrefixAir {
                    transcript_bus: self.buses.transcript,
                    schedule_bus: exact.schedule_bus,
                    profile: exact.transcript_profile.clone(),
                    lookup_count: exact.multiplicities.exact_schedule_lookup_count,
                },
            );
        }
        add_air(
            &mut airs,
            NativeExactFiniteVaccCallPrefixAir {
                transcript_bus: self.buses.transcript,
                phase_cursor_bus: self.buses.vacc_phase_cursor,
                protocol_bus: exact.call_protocol_bus,
                legacy_protocol_bus: self.protocol_bus,
                profile: exact.transcript_profile.clone(),
                call_index: exact.call_index,
                lookup_count: exact
                    .multiplicities
                    .exact_call_protocol_lookup_count_per_call,
            },
        );
        let cursor_lengths = vacc_cursor_role_lengths_exact_v19(
            &self.profile,
            call.input_arity,
            call.fresh_count,
            call.prior_count == 1,
        );
        add_air(
            &mut airs,
            DirectAirVaccTranscriptCursorAirV19 {
                transcript_bus: self.buses.transcript,
                semantic_bus: self.buses.vacc_semantic_transcript,
                phase_cursor_bus: self.buses.vacc_phase_cursor,
                role_bus: self.buses.vacc_transcript_role,
                next_roles: next_nonempty_vacc_roles_v19(&cursor_lengths),
                lengths: cursor_lengths,
                twin_event_count: algebra_profile.twin_degree + 2,
                dynamic_role_lengths: false,
                dynamic_optional_prior: false,
                emit_fresh_root_role: true,
            },
        );
        add_air(
            &mut airs,
            NativeStandardVaccEndAir {
                transcript_bus: self.buses.transcript,
                transcript_end_index_bus: (exact.call_index != 0)
                    .then_some(self.shared.transcript_end_index_bus),
                phase_cursor_bus: self.buses.vacc_phase_cursor,
                end_bus: self.end_bus,
                end_tag: EXACT_FINITE_WARP_CALL_END_TAG,
                extension_tag: true,
            },
        );
        add_air(
            &mut airs,
            NativeStandardClaimValueAir {
                claim_bus: self.buses.claim_value,
                transcript_bus: self.buses.vacc_semantic_transcript,
                transcript_role_bus: self.buses.vacc_transcript_role,
                layout_bus: self.buses.claim_layout,
                slot_bus: self.buses.input_slot_layout,
                fresh_claim_consumer_count: exact.multiplicities.fresh_claim_consumer_count,
                prior_claim_consumer_count: usize::from(self.include_prior) + 1,
                fresh_alpha_len: self.profile.log_codeword_len,
                fresh_tau_len: self.profile.log_constraints,
                fresh_beta_tail_len: self.profile.beta_len - self.profile.log_constraints,
            },
        );
        add_air(
            &mut airs,
            NativeClaimLayoutAir {
                layout_bus: self.buses.claim_layout,
            },
        );
        add_air(
            &mut airs,
            NativeExactFiniteInputSlotLayoutAir {
                bus: self.buses.input_slot_layout,
                input_arity: call.input_arity,
                fresh_count: call.fresh_count,
                prior_count: call.prior_count,
                ordinary_lookup_count: self.profile.log_codeword_len
                    + self.profile.beta_len
                    + 2
                    + 2 * self.profile.num_shift_queries,
                extra_fresh_lookup_count: exact
                    .multiplicities
                    .extra_fresh_slot_lookup_count_per_source,
            },
        );
        add_air(
            &mut airs,
            NativeFiniteStackedFreshCommitmentAir {
                transcript_bus: self.buses.vacc_semantic_transcript,
                transcript_role_bus: self.buses.vacc_transcript_role,
                phase_cursor_bus: self.buses.vacc_phase_cursor,
                merkle_root_bus: self.buses.merkle_root,
                statement_root_bus: self.statement_root_bus,
                profile: fresh,
            },
        );
        add_air(
            &mut airs,
            NativeFiniteStackedFreshProjectionAir {
                shift_index_bus: self.buses.shift_index,
                leaf_value_bus: self.buses.leaf_value,
                authenticated_bus: self.buses.authenticated_shift,
                slot_bus: self.buses.input_slot_layout,
                profile: fresh,
                shift_count: self.profile.num_shift_queries,
            },
        );
        if self.include_prior {
            add_air(
                &mut airs,
                NativeStandardCodewordProjectionAir {
                    shift_index_bus: self.buses.shift_index,
                    leaf_value_bus: self.buses.leaf_value,
                    authenticated_bus: self.buses.authenticated_shift,
                    slot_bus: self.buses.input_slot_layout,
                    oracle_height: self.oracle_height,
                    query_count: self.query_count,
                    row_tree_id_offset: self.trees.prior_rows,
                    source: call.input_arity - 1,
                    variant: fresh.variant(),
                    is_fresh: false,
                },
            );
            self.add_root_air(
                &mut airs,
                1,
                self.trees.prior_outer,
                outer_depth,
                true,
                true,
            );
        }
        add_air(
            &mut airs,
            NativeLeafHashAir {
                permute_bus: self.shared.poseidon2_permute_bus,
                value_bus: self.buses.leaf_value,
                leaf_bus: self.buses.opening_leaf,
            },
        );
        add_air(
            &mut airs,
            NativeMerkleMultiproofAir {
                compress_bus: self.shared.poseidon2_compress_bus,
                leaf_bus: self.buses.opening_leaf,
                node_bus: self.buses.merkle_node,
                root_bus: self.buses.merkle_root,
            },
        );
        add_air(
            &mut airs,
            NativeMerkleLeafAdapterAir {
                inner_depth: self.profile.rows_per_query.ilog2() as usize,
                leaf_bus: self.buses.opening_leaf,
                root_bus: self.buses.merkle_root,
            },
        );
        add_air(
            &mut airs,
            DirectAirVaccVectorCoordinateAirV19 {
                inner: NativeVectorCoordinateAir {
                    vector_bus: self.buses.vector_coordinate,
                    sumcheck_bus: self.buses.sumcheck_challenge,
                    transcript_bus: self.buses.vacc_semantic_transcript,
                    folded_bus: self.buses.folded_claim,
                    opening_bus: self.buses.opening_claim,
                },
                role_bus: self.buses.vacc_transcript_role,
                selector_vector: self.algebra.selector_tau_vector as usize,
                xi_vector: self.algebra.xi_vector as usize,
                fixed_proof_slot: None,
            },
        );
        for (dimensions, group_offset) in [
            (algebra_profile.twin_rounds, 0usize),
            (log_claim_count, 2 * call.input_arity + 1),
            (
                self.profile.log_codeword_len,
                2 * call.input_arity + 1 + claim_count,
            ),
        ] {
            add_air(
                &mut airs,
                NativeEqEvaluationAir {
                    result_bus: self.buses.eq_result,
                    vector_bus: self.buses.vector_coordinate,
                    dimensions,
                    group_offset,
                },
            );
        }
        add_air(
            &mut airs,
            DirectAirVaccTwinOmegaAirV19 {
                inner: NativeTwinOmegaAir {
                    transcript_bus: self.buses.vacc_semantic_transcript,
                    omega_bus: self.buses.twin_omega,
                },
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            algebra_profile.twin_sigma_air(
                self.buses.claim_value,
                self.buses.eq_result,
                self.buses.sumcheck_initial,
                self.buses.twin_omega,
            ),
        );
        add_air(
            &mut airs,
            DirectAirVaccCoefficientSumcheckAirV19 {
                inner: algebra_profile.coefficient_sumcheck_air(
                    self.buses.vacc_semantic_transcript,
                    self.buses.sumcheck_round,
                    self.buses.sumcheck_initial,
                    self.buses.sumcheck_challenge,
                ),
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            algebra_profile.twin_fold_air(
                self.buses.claim_value,
                self.buses.eq_result,
                self.buses.folded_claim,
            ),
        );
        add_air(
            &mut airs,
            DirectAirVaccTwinFinalAirV19 {
                inner: algebra_profile.twin_final_air(
                    self.buses.sumcheck_round,
                    self.buses.eq_result,
                    self.buses.twin_scalar,
                    self.algebra.selector_at_gamma_group as usize,
                    self.buses.twin_omega,
                    self.buses.vacc_semantic_transcript,
                ),
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            NativeInitialOpeningPointAir {
                folded_bus: self.buses.folded_claim,
                opening_bus: self.buses.opening_claim,
                copies: opening_padding_count + 1,
            },
        );
        add_air(
            &mut airs,
            NativeInitialOpeningTargetAir {
                twin_bus: self.buses.twin_scalar,
                opening_bus: self.buses.opening_claim,
                copies: opening_padding_count + 1,
            },
        );
        add_air(
            &mut airs,
            NativeOpeningPaddingAir {
                opening_bus: self.buses.opening_claim,
            },
        );
        add_air(
            &mut airs,
            DirectAirVaccOodClaimAirV19 {
                inner: NativeOodClaimAir {
                    transcript_bus: self.buses.vacc_semantic_transcript,
                    opening_bus: self.buses.opening_claim,
                },
                role_bus: self.buses.vacc_transcript_role,
                dimension: self.profile.log_codeword_len,
            },
        );
        add_air(
            &mut airs,
            DirectAirVaccShiftScheduleAirV19 {
                inner: NativeShiftScheduleAir {
                    transcript_bus: self.buses.vacc_semantic_transcript,
                    exp_bits_len_bus: self.shared.exp_bits_len_bus,
                    right_shift_bus: self.shared.right_shift_bus,
                    shift_index_bus: self.buses.shift_index,
                    opening_bus: Some(self.buses.opening_claim),
                    log_codeword_len: self.profile.log_codeword_len,
                    opening_claim_offset: 1 + self.profile.num_ood,
                },
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            NativeShiftMergeAir {
                authenticated_bus: self.buses.authenticated_shift,
                eq_bus: self.buses.eq_result,
                opening_bus: self.buses.opening_claim,
                slot_bus: self.buses.input_slot_layout,
                input_arity: call.input_arity,
                gamma_eq_group_offset: call.input_arity,
                opening_claim_offset: 1 + self.profile.num_ood,
            },
        );
        add_air(
            &mut airs,
            NativeBatchingSigmaAir {
                eq_bus: self.buses.eq_result,
                opening_bus: self.buses.opening_claim,
                sumcheck_initial_bus: Some(self.buses.sumcheck_initial),
                // The exact finite wrapper proves the complete VACC algebra
                // directly and has no History replay seal. Keep the ordinary
                // sumcheck link above, but do not emit the legacy History-only
                // batching certificate into an unowned bus namespace.
                certified_claim_bus: None,
                claim_count,
                xi_eq_group_offset: self.algebra.xi_weight_groups[0] as usize,
            },
        );
        add_air(
            &mut airs,
            DirectAirVaccBatchingFinalAirV19 {
                inner: NativeBatchingFinalAir {
                    eq_bus: self.buses.eq_result,
                    sumcheck_round_bus: self.buses.sumcheck_round,
                    output_bus: self.buses.batching_output,
                    claim_count,
                    xi_eq_group_offset: self.algebra.xi_weight_groups[0] as usize,
                    point_eq_group_offset: self.algebra.opening_at_alpha_groups[0] as usize,
                    last_round: self.profile.log_codeword_len - 1,
                    transcript_bus: self.buses.vacc_semantic_transcript,
                },
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            NativeBatchingAlphaAir {
                challenge_bus: self.buses.sumcheck_challenge,
                output_bus: self.buses.batching_output,
            },
        );
        self.add_root_air(&mut airs, 2, 0, 0, false, true);
        if self.include_prior {
            self.add_accumulator_airs(&mut airs, NativeAccumulatorBindingMode::Prior);
        }
        self.add_accumulator_airs(&mut airs, NativeAccumulatorBindingMode::Output);
        add_air(
            &mut airs,
            DirectAirExactFiniteVaccAuthorityAirV19 {
                exact: exact.clone(),
                protocol_bus: self.protocol_bus,
                end_bus: self.end_bus,
                root_bus: self.statement_root_bus,
                digest_bus: self.statement_digest_bus,
                checkpoint_bus: self.buses.transcript_checkpoint,
                resume_bus: self.shared.resume_state_bus,
            },
        );
        add_air(
            &mut airs,
            ExpBitsLenAir::new(self.shared.exp_bits_len_bus, self.shared.right_shift_bus),
        );
        airs
    }

    /// One fixed AIR inventory for every reduced-SWIRL call. Runtime calls
    /// are rows, never verifier-key-specialized AIR clones.
    fn reduced_swirl_airs<PCS: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<PCS>> {
        let reduced = self
            .reduced_swirl
            .as_ref()
            .expect("reduced-SWIRL verifier configuration");
        let input_arity = reduced.input_arity;
        let twin_rounds = input_arity.ilog2() as usize;
        let constraint_degree = twin_degree(
            self.profile.log_codeword_len,
            self.profile.log_constraints,
            self.profile.max_degree,
        );
        let algebra_profile = NativeExactFiniteVaccCallAlgebraProfile {
            input_arity,
            twin_rounds,
            twin_last_round: twin_rounds - 1,
            twin_degree: constraint_degree,
            batching_rounds: self.profile.log_codeword_len,
            batching_degree: 2,
        };
        let family = family_from_shape_profile(&self.profile, input_arity, input_arity);
        let claim_count = NativeWarpAlgebraLayout::claim_count(&family);
        let authenticated_claim_count = 1 + self.profile.num_ood + self.profile.num_shift_queries;
        let opening_padding_count = claim_count - authenticated_claim_count;
        let log_claim_count = claim_count.ilog2() as usize;
        let query_stride = (1usize << self.profile.log_codeword_len) / self.profile.rows_per_query;
        let outer_depth = query_stride.ilog2() as usize;
        let root_tree_stride = 1 + self.profile.num_shift_queries;
        let tree_source_stride = reduced.max_roots_per_source * root_tree_stride;

        let mut airs = self.transcript.airs::<PCS>();
        let mut role_lengths =
            vacc_cursor_role_lengths_exact_v19(&self.profile, input_arity, input_arity, true);
        role_lengths[VACC_ROLE_FRESH_ROOT] =
            input_arity * (9 + reduced.max_roots_per_source * (1 + DIGEST_SIZE));
        add_air(
            &mut airs,
            DirectAirVaccTranscriptCursorAirV19 {
                transcript_bus: self.buses.transcript,
                semantic_bus: self.buses.vacc_semantic_transcript,
                phase_cursor_bus: self.buses.vacc_phase_cursor,
                role_bus: self.buses.vacc_transcript_role,
                next_roles: next_nonempty_vacc_roles_v19(&role_lengths),
                lengths: role_lengths,
                twin_event_count: constraint_degree + 2,
                dynamic_role_lengths: true,
                dynamic_optional_prior: true,
                emit_fresh_root_role: false,
            },
        );
        add_air(
            &mut airs,
            DirectAirReducedVaccEndAirV19 {
                transcript_bus: self.buses.transcript,
                phase_cursor_bus: self.buses.vacc_phase_cursor,
                end_bus: self.end_bus,
                end_tag: EXACT_FINITE_WARP_CALL_END_TAG,
            },
        );
        add_air(
            &mut airs,
            NativeStandardClaimValueAir {
                claim_bus: self.buses.claim_value,
                transcript_bus: self.buses.vacc_semantic_transcript,
                transcript_role_bus: self.buses.vacc_transcript_role,
                layout_bus: self.buses.claim_layout,
                slot_bus: self.buses.input_slot_layout,
                // Algebra plus the authoritative reduced-source linker.
                fresh_claim_consumer_count: 2,
                // Algebra plus the dynamic-prior adapter.
                prior_claim_consumer_count: 2,
                fresh_alpha_len: self.profile.log_codeword_len,
                fresh_tau_len: self.profile.log_constraints,
                fresh_beta_tail_len: self.profile.beta_len - self.profile.log_constraints,
            },
        );
        add_air(
            &mut airs,
            NativeClaimLayoutAir {
                layout_bus: self.buses.claim_layout,
            },
        );
        add_air(
            &mut airs,
            NativeDirectFreshCommitmentAir {
                transcript_bus: self.buses.vacc_semantic_transcript,
                source_bus: self.buses.direct_fresh_source,
                activity_bus: self.buses.fresh_source_activity,
                fresh_count_bus: self.buses.fresh_count,
                digest_element_bus: None,
                max_fresh: input_arity,
                max_roots: reduced.max_roots_per_source,
                shift_count: self.profile.num_shift_queries,
                expected_log_message_len: self.profile.log_message_len,
                expected_log_codeword_len: self.profile.log_codeword_len,
                expected_rows_per_query: self.profile.rows_per_query,
                tree_source_stride,
                first_tree_id: 0,
                digest_metadata_len: 0,
                digest_source_width: 0,
                digest_alpha_len: 0,
                digest_beta_len: 0,
                claim_rows_per_source: 0,
            },
        );
        add_air(
            &mut airs,
            NativeDirectFreshRootAir {
                transcript_bus: self.buses.vacc_semantic_transcript,
                source_bus: self.buses.direct_fresh_source,
                root_bus: self.buses.direct_fresh_root,
                merkle_root_bus: self.buses.merkle_root,
                digest_element_bus: None,
                max_fresh: input_arity,
                max_roots: reduced.max_roots_per_source,
                shift_count: self.profile.num_shift_queries,
                root_tree_stride,
                outer_depth,
                digest_metadata_len: 0,
                digest_source_width: 0,
            },
        );
        for first_source in (0..input_arity).step_by(reduced.projection_sources_per_shard) {
            add_air(
                &mut airs,
                NativeDirectFreshProjectionAir {
                    source_bus: self.buses.direct_fresh_source,
                    root_bus: self.buses.direct_fresh_root,
                    shift_index_bus: self.buses.shift_index,
                    leaf_value_bus: self.buses.leaf_value,
                    authenticated_bus: self.buses.authenticated_shift,
                    shift_count: self.profile.num_shift_queries,
                    query_stride,
                    root_tree_stride,
                    first_source,
                    allow_empty: first_source != 0,
                },
            );
        }
        add_air(
            &mut airs,
            NativeAccumulatorProjectionAir {
                shift_index_bus: self.buses.shift_index,
                leaf_value_bus: self.buses.leaf_value,
                authenticated_bus: self.buses.authenticated_shift,
                slot_bus: self.buses.input_slot_layout,
                oracle_height: self.oracle_height,
                query_count: self.query_count,
                row_tree_id_offset: self.trees.prior_rows,
            },
        );
        self.add_root_air(
            &mut airs,
            1,
            self.trees.prior_outer,
            outer_depth,
            true,
            true,
        );
        add_air(
            &mut airs,
            NativeLeafHashAir {
                permute_bus: self.shared.poseidon2_permute_bus,
                value_bus: self.buses.leaf_value,
                leaf_bus: self.buses.opening_leaf,
            },
        );
        add_air(
            &mut airs,
            NativeMerkleMultiproofAir {
                compress_bus: self.shared.poseidon2_compress_bus,
                leaf_bus: self.buses.opening_leaf,
                node_bus: self.buses.merkle_node,
                root_bus: self.buses.merkle_root,
            },
        );
        add_air(
            &mut airs,
            NativeMerkleLeafAdapterAir {
                inner_depth: self.profile.rows_per_query.ilog2() as usize,
                leaf_bus: self.buses.opening_leaf,
                root_bus: self.buses.merkle_root,
            },
        );
        add_air(
            &mut airs,
            DirectAirVaccVectorCoordinateAirV19 {
                inner: NativeVectorCoordinateAir {
                    vector_bus: self.buses.vector_coordinate,
                    sumcheck_bus: self.buses.sumcheck_challenge,
                    transcript_bus: self.buses.vacc_semantic_transcript,
                    folded_bus: self.buses.folded_claim,
                    opening_bus: self.buses.opening_claim,
                },
                role_bus: self.buses.vacc_transcript_role,
                selector_vector: self.algebra.selector_tau_vector as usize,
                xi_vector: self.algebra.xi_vector as usize,
                fixed_proof_slot: None,
            },
        );
        for (dimensions, group_offset) in [
            (twin_rounds, 0usize),
            (log_claim_count, 2 * input_arity + 1),
            (
                self.profile.log_codeword_len,
                2 * input_arity + 1 + claim_count,
            ),
        ] {
            add_air(
                &mut airs,
                NativeEqEvaluationAir {
                    result_bus: self.buses.eq_result,
                    vector_bus: self.buses.vector_coordinate,
                    dimensions,
                    group_offset,
                },
            );
        }
        add_air(
            &mut airs,
            DirectAirVaccTwinOmegaAirV19 {
                inner: NativeTwinOmegaAir {
                    transcript_bus: self.buses.vacc_semantic_transcript,
                    omega_bus: self.buses.twin_omega,
                },
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            algebra_profile.twin_sigma_air(
                self.buses.claim_value,
                self.buses.eq_result,
                self.buses.sumcheck_initial,
                self.buses.twin_omega,
            ),
        );
        add_air(
            &mut airs,
            DirectAirVaccCoefficientSumcheckAirV19 {
                inner: algebra_profile.coefficient_sumcheck_air(
                    self.buses.vacc_semantic_transcript,
                    self.buses.sumcheck_round,
                    self.buses.sumcheck_initial,
                    self.buses.sumcheck_challenge,
                ),
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            algebra_profile.twin_fold_air(
                self.buses.claim_value,
                self.buses.eq_result,
                self.buses.folded_claim,
            ),
        );
        add_air(
            &mut airs,
            DirectAirVaccTwinFinalAirV19 {
                inner: algebra_profile.twin_final_air(
                    self.buses.sumcheck_round,
                    self.buses.eq_result,
                    self.buses.twin_scalar,
                    self.algebra.selector_at_gamma_group as usize,
                    self.buses.twin_omega,
                    self.buses.vacc_semantic_transcript,
                ),
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            NativeInitialOpeningPointAir {
                folded_bus: self.buses.folded_claim,
                opening_bus: self.buses.opening_claim,
                copies: opening_padding_count + 1,
            },
        );
        add_air(
            &mut airs,
            NativeInitialOpeningTargetAir {
                twin_bus: self.buses.twin_scalar,
                opening_bus: self.buses.opening_claim,
                copies: opening_padding_count + 1,
            },
        );
        add_air(
            &mut airs,
            NativeOpeningPaddingAir {
                opening_bus: self.buses.opening_claim,
            },
        );
        add_air(
            &mut airs,
            DirectAirVaccOodClaimAirV19 {
                inner: NativeOodClaimAir {
                    transcript_bus: self.buses.vacc_semantic_transcript,
                    opening_bus: self.buses.opening_claim,
                },
                role_bus: self.buses.vacc_transcript_role,
                dimension: self.profile.log_codeword_len,
            },
        );
        add_air(
            &mut airs,
            DirectAirVaccShiftScheduleAirV19 {
                inner: NativeShiftScheduleAir {
                    transcript_bus: self.buses.vacc_semantic_transcript,
                    exp_bits_len_bus: self.shared.exp_bits_len_bus,
                    right_shift_bus: self.shared.right_shift_bus,
                    shift_index_bus: self.buses.shift_index,
                    opening_bus: Some(self.buses.opening_claim),
                    log_codeword_len: self.profile.log_codeword_len,
                    opening_claim_offset: 1 + self.profile.num_ood,
                },
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            NativeShiftMergeAir {
                authenticated_bus: self.buses.authenticated_shift,
                eq_bus: self.buses.eq_result,
                opening_bus: self.buses.opening_claim,
                slot_bus: self.buses.input_slot_layout,
                input_arity,
                gamma_eq_group_offset: input_arity,
                opening_claim_offset: 1 + self.profile.num_ood,
            },
        );
        add_air(
            &mut airs,
            NativeBatchingSigmaAir {
                eq_bus: self.buses.eq_result,
                opening_bus: self.buses.opening_claim,
                sumcheck_initial_bus: Some(self.buses.sumcheck_initial),
                certified_claim_bus: None,
                claim_count,
                xi_eq_group_offset: self.algebra.xi_weight_groups[0] as usize,
            },
        );
        add_air(
            &mut airs,
            DirectAirVaccBatchingFinalAirV19 {
                inner: NativeBatchingFinalAir {
                    eq_bus: self.buses.eq_result,
                    sumcheck_round_bus: self.buses.sumcheck_round,
                    output_bus: self.buses.batching_output,
                    claim_count,
                    xi_eq_group_offset: self.algebra.xi_weight_groups[0] as usize,
                    point_eq_group_offset: self.algebra.opening_at_alpha_groups[0] as usize,
                    last_round: self.profile.log_codeword_len - 1,
                    transcript_bus: self.buses.vacc_semantic_transcript,
                },
                role_bus: self.buses.vacc_transcript_role,
            },
        );
        add_air(
            &mut airs,
            NativeBatchingAlphaAir {
                challenge_bus: self.buses.sumcheck_challenge,
                output_bus: self.buses.batching_output,
            },
        );
        self.add_root_air(&mut airs, 2, 0, 0, false, true);
        add_air(
            &mut airs,
            DirectAirReducedPriorClaimAdapterAirV19 {
                claim_bus: self.buses.claim_value,
                slot_bus: self.buses.input_slot_layout,
                virtual_source: input_arity - 1,
            },
        );
        self.add_reduced_accumulator_airs(&mut airs, NativeAccumulatorBindingMode::Prior, true);
        self.add_reduced_accumulator_airs(&mut airs, NativeAccumulatorBindingMode::Output, false);
        add_air(
            &mut airs,
            ExpBitsLenAir::new(self.shared.exp_bits_len_bus, self.shared.right_shift_bus),
        );
        airs
    }

    fn add_reduced_accumulator_airs<PCS: StarkProtocolConfig<F = F>>(
        &self,
        airs: &mut Vec<AirRef<PCS>>,
        mode: NativeAccumulatorBindingMode,
        allow_empty: bool,
    ) {
        let state = usize::from(mode == NativeAccumulatorBindingMode::Output);
        add_air(
            airs,
            DirectAirAccumulatorValueAirV19 {
                mode,
                allow_empty,
                layout: self.private_accumulator.clone(),
                claim_source: self.input_arity() - 1,
                digest_element_bus: self.buses.accumulator_digest_element,
                claim_bus: self.buses.claim_value,
                batching_bus: self.buses.batching_output,
                folded_bus: self.buses.folded_claim,
                twin_bus: self.buses.twin_scalar,
            },
        );
        add_air(
            airs,
            NativeAccumulatorHashAir {
                state,
                allow_empty,
                alpha_len: self.private_accumulator.alpha_len,
                beta_len: self.private_accumulator.beta_len,
                digest_element_bus: self.buses.accumulator_digest_element,
                algebraic_digest_bus: self.buses.accumulator_algebraic_digest,
                permute_bus: self.shared.poseidon2_permute_bus,
                compress_bus: self.shared.poseidon2_compress_bus,
            },
        );
        add_air(
            airs,
            DirectAirAccumulatorRootDigestAirV19 {
                mode,
                allow_empty,
                root_bus: self.buses.accumulator_root,
                algebraic_digest_bus: self.buses.accumulator_algebraic_digest,
                compress_bus: self.shared.poseidon2_compress_bus,
                digest_bus: self.statement_digest_bus,
                certified_digest_bus: None,
            },
        );
    }

    /// Keep the large row-local vector table under the History PCS height
    /// without changing WHIR parameters. Fixed HLeaves have exactly four
    /// authenticated local transition slots, so each slot receives its own
    /// table. Legacy batches preserve their historical single-table layout.
    fn vector_coordinate_shard_count_v19(&self) -> usize {
        if self.warp_step_mode == DirectAirVaccWarpStepModeV19::FixedHLeafV4 {
            FIXED_HLEAF_VACC_CAPACITY_V4
        } else {
            1
        }
    }

    /// AIRs for a parent composition that owns the physical Poseidon table.
    ///
    /// The verifier's transcript AIR and every algebra/authentication AIR are
    /// retained in their ordinary order; only the second AIR, the local
    /// `Poseidon2Air`, is omitted. The caller must install one Poseidon AIR
    /// covering this module's distinct permutation/compression buses and feed
    /// it the packet returned by
    /// [`Self::generate_traces_for_shared_poseidon`].
    #[must_use]
    pub fn airs_without_poseidon<PCS: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<PCS>> {
        let mut airs = self.airs::<PCS>();
        assert!(airs.len() >= 2, "direct VACC verifier AIR prefix");
        airs.remove(1);
        airs
    }

    fn add_root_air<PCS: StarkProtocolConfig<F = F>>(
        &self,
        airs: &mut Vec<AirRef<PCS>>,
        kind: usize,
        tree_id: usize,
        depth: usize,
        authenticate_merkle: bool,
        bind_accumulator: bool,
    ) {
        add_air(
            airs,
            NativeStandardVaccCommitmentRootAir {
                transcript_bus: self.buses.vacc_semantic_transcript,
                transcript_role_bus: self.buses.vacc_transcript_role,
                merkle_root_bus: authenticate_merkle.then_some(self.buses.merkle_root),
                accumulator_root_bus: bind_accumulator.then_some(self.buses.accumulator_root),
                statement_root_bus: self.statement_root_bus,
                proof_kind: kind,
                transcript_role: match kind {
                    0 => VACC_ROLE_FRESH_ROOT,
                    1 => VACC_ROLE_PRIOR_ROOT,
                    2 => VACC_ROLE_OUTPUT_ROOT,
                    _ => unreachable!("standard VACC root kind is verifier-key data"),
                },
                expected_tree_id: tree_id,
                expected_depth: depth,
            },
        );
    }

    fn add_accumulator_airs<PCS: StarkProtocolConfig<F = F>>(
        &self,
        airs: &mut Vec<AirRef<PCS>>,
        mode: NativeAccumulatorBindingMode,
    ) {
        let state = usize::from(mode == NativeAccumulatorBindingMode::Output);
        add_air(
            airs,
            DirectAirAccumulatorValueAirV19 {
                mode,
                allow_empty: false,
                layout: self.private_accumulator.clone(),
                claim_source: self.prior_claim_source(),
                digest_element_bus: self.buses.accumulator_digest_element,
                claim_bus: self.buses.claim_value,
                batching_bus: self.buses.batching_output,
                folded_bus: self.buses.folded_claim,
                twin_bus: self.buses.twin_scalar,
            },
        );
        add_air(
            airs,
            NativeAccumulatorHashAir {
                state,
                allow_empty: false,
                alpha_len: self.private_accumulator.alpha_len,
                beta_len: self.private_accumulator.beta_len,
                digest_element_bus: self.buses.accumulator_digest_element,
                algebraic_digest_bus: self.buses.accumulator_algebraic_digest,
                permute_bus: self.shared.poseidon2_permute_bus,
                compress_bus: self.shared.poseidon2_compress_bus,
            },
        );
        add_air(
            airs,
            DirectAirAccumulatorRootDigestAirV19 {
                mode,
                allow_empty: false,
                root_bus: self.buses.accumulator_root,
                algebraic_digest_bus: self.buses.accumulator_algebraic_digest,
                compress_bus: self.shared.poseidon2_compress_bus,
                digest_bus: self.statement_digest_bus,
                certified_digest_bus: self
                    .exact_finite
                    .is_none()
                    .then_some(self.buses.certified_accumulator_digest),
            },
        );
    }
}

fn family_from_profile(profile: &NativeStandardVaccShapeProfile) -> NativeWarpFamilyParams {
    family_from_shape_profile(profile, STANDARD_DIRECT_VACC_INPUT_ARITY, 1)
}

fn family_from_shape_profile(
    profile: &NativeStandardVaccShapeProfile,
    input_arity: usize,
    fresh_count: usize,
) -> NativeWarpFamilyParams {
    NativeWarpFamilyParams {
        max_shape_slots: 1,
        input_arity,
        max_fresh_per_step: fresh_count,
        num_ood: profile.num_ood,
        num_shift_queries: profile.num_shift_queries,
        max_stacked_roots: 1,
        max_stacked_width: 1 << profile.initial_folding_factor,
        max_public_values: 0,
        log_message_height: profile.log_message_len - profile.initial_folding_factor,
        accumulator_rows_per_query: profile.rows_per_query,
        source_rows_per_query: profile.rows_per_query,
        log_codeword_len: profile.log_codeword_len,
        log_constraints: profile.log_constraints,
        explicit_len: profile.beta_len - profile.log_constraints,
        beta_len: profile.beta_len,
    }
}

fn add_air<PCS, A>(airs: &mut Vec<AirRef<PCS>>, air: A)
where
    PCS: StarkProtocolConfig,
    A: AnyAir<PCS> + 'static,
{
    airs.push(Arc::new(air));
}

/// Transcript terminator emitted by `NativeV19TranscriptFactory` after VACC.
pub const SHARD_TRANSCRIPT_VACC_END_TAG_V19: u64 = 0x4e57_5645_4e44_0013;

#[derive(Clone, Debug, PartialEq, Eq)]
struct DirectAirPriorTranscriptScheduleV19 {
    root_tidx: usize,
    alpha_tidx: Vec<usize>,
    mu_tidx: usize,
    beta_tidx: Vec<usize>,
    eta_tidx: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DirectAirSampleBitsScheduleV19 {
    operation_range: core::ops::Range<usize>,
    result: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct DirectAirVaccTranscriptScheduleV19 {
    proof_idx: usize,
    start_tidx: usize,
    fresh_root_tidx: usize,
    fresh_commitment_extra_tidx: Vec<usize>,
    fresh_alpha_tidx: Vec<usize>,
    fresh_mu_tidx: Vec<usize>,
    fresh_beta_tail_tidx: Vec<usize>,
    fresh_tau_tidx: Vec<usize>,
    prior: Option<DirectAirPriorTranscriptScheduleV19>,
    omega_tidx: usize,
    selector_tidx: Vec<usize>,
    gamma_tidx: Vec<usize>,
    output_root_tidx: usize,
    nu_tidx: usize,
    eta_tidx: usize,
    ood_point_tidx: Vec<usize>,
    ood_answer_tidx: Vec<usize>,
    shifts: Vec<DirectAirSampleBitsScheduleV19>,
    xi_tidx: Vec<usize>,
    mu_tidx: usize,
    end_tag_tidx: usize,
    end_tidx: usize,
    discarded_end_sample: EF,
    claimed: Vec<bool>,
}

impl DirectAirVaccTranscriptScheduleV19 {
    fn from_record<FreshVerification: DirectAirFreshAuthenticationV19>(
        module: &DirectAirVaccVerifierModuleV19,
        record: &DirectAirVaccVerifierRecordV19<'_, FreshVerification>,
    ) -> Result<Self, DirectAirVaccVerifierErrorV19> {
        let producer = record.producer;
        let verification = record.verification;
        let log = record.transcript;
        let start = producer.start_checkpoint.operation_index as usize;
        let end = producer.end_checkpoint.operation_index as usize;
        if producer.has_prior != module.include_prior
            || record.prior.is_some() != module.include_prior
            || verification.fresh_authentication.len() != 1
            || verification.prior_authentication.is_some() != module.include_prior
            || producer.fresh_alpha.len() != module.profile.log_codeword_len
            || producer.fresh_beta.len() != module.profile.beta_len
            || verification.twin.fresh_taus.len() != 1
            || verification.twin.fresh_taus[0].len() != module.profile.log_constraints
            || verification.twin.selector_tau.len() != 1
            || verification.twin.gamma.len() != 1
            || verification.twin.zeta_0.len() != module.profile.log_codeword_len
            || verification.twin.output_beta.len() != module.profile.beta_len
            || verification.ood.len() != module.profile.num_ood
            || verification.shifts.len() != module.profile.num_shift_queries
            || verification.batching.alpha.len() != module.profile.log_codeword_len
            || verification.batching.xi.len() != module.profile.batching_arity.ilog2() as usize
        {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "standard VACC dimensions",
            ));
        }
        validate_v19_phase_ranges(log, &verification.transcript_phases, start, end)?;
        let gamma_tidx = validate_v19_sumcheck_spans(
            log,
            &verification.transcript_phases,
            &verification.twin.sumcheck,
            NativeSumcheckKind::TwinConstraint,
        )?;
        let batching_alpha_tidx = validate_v19_sumcheck_spans(
            log,
            &verification.transcript_phases,
            &verification.batching.sumcheck,
            NativeSumcheckKind::MultilinearBatching,
        )?;
        if gamma_tidx.len() != 1 || batching_alpha_tidx.len() != module.profile.log_codeword_len {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "sumcheck round count",
            ));
        }

        let protocol = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::VaccProtocolPrefix,
        )?;
        if protocol.operation_range.start != start {
            return Err(DirectAirVaccVerifierErrorV19::Transcript(
                "VACC prefix start",
            ));
        }
        let commitments = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::FreshCommitments,
        )?;
        let mut events = v19_phase_events(log, commitments)?.iter();
        let fresh_root_tidx = take_v19_lifted_digest(&mut events)?;
        let fresh_commitment_extra_tidx = take_v19_ext_events(
            &mut events,
            module.fresh_commitment_mode.transcript_extra_elements(),
            false,
        )?;
        ensure_no_v19_events(events)?;

        let fresh_claims = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::FreshClaims,
        )?;
        let mut events = v19_phase_events(log, fresh_claims)?.iter();
        let fresh_alpha_tidx =
            take_v19_ext_events(&mut events, module.profile.log_codeword_len, false)?;
        let fresh_mu_tidx = vec![take_v19_ext_event(&mut events, false)?];
        let fresh_beta_tail_tidx = take_v19_ext_events(
            &mut events,
            module.profile.beta_len - module.profile.log_constraints,
            false,
        )?;
        ensure_no_v19_events(events)?;

        let prior_phase = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::PriorAccumulator,
        )?;
        let mut events = v19_phase_events(log, prior_phase)?.iter();
        let prior = if module.include_prior {
            let prior = DirectAirPriorTranscriptScheduleV19 {
                root_tidx: take_v19_lifted_digest(&mut events)?,
                alpha_tidx: take_v19_ext_events(
                    &mut events,
                    module.profile.log_codeword_len,
                    false,
                )?,
                mu_tidx: take_v19_ext_event(&mut events, false)?,
                beta_tidx: take_v19_ext_events(&mut events, module.profile.beta_len, false)?,
                eta_tidx: take_v19_ext_event(&mut events, false)?,
            };
            Some(prior)
        } else {
            None
        };
        ensure_no_v19_events(events)?;

        let tau_phase = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::FreshZeroCheckPoints,
        )?;
        let mut events = v19_phase_events(log, tau_phase)?.iter();
        let fresh_tau_tidx =
            take_v19_ext_events(&mut events, module.profile.log_constraints, true)?;
        ensure_no_v19_events(events)?;

        let twin = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::TwinChallenges,
        )?;
        let mut events = v19_phase_events(log, twin)?.iter();
        let omega_tidx = take_v19_ext_event(&mut events, true)?;
        let selector_tidx = take_v19_ext_events(&mut events, 1, true)?;
        ensure_no_v19_events(events)?;

        let output = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::OutputAccumulatorCommitment,
        )?;
        let mut events = v19_phase_events(log, output)?.iter();
        let output_root_tidx = take_v19_lifted_digest(&mut events)?;
        let nu_tidx = take_v19_ext_event(&mut events, false)?;
        let eta_tidx = take_v19_ext_event(&mut events, false)?;
        ensure_no_v19_events(events)?;

        let ood_points = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::OodPoints,
        )?;
        let mut events = v19_phase_events(log, ood_points)?.iter();
        let ood_point_tidx = (0..module.profile.num_ood)
            .map(|_| {
                take_v19_ext_events(&mut events, module.profile.log_codeword_len, true)?
                    .first()
                    .copied()
                    .ok_or(DirectAirVaccVerifierErrorV19::Transcript("empty OOD point"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        ensure_no_v19_events(events)?;
        let ood_answers = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::OodAnswers,
        )?;
        let mut events = v19_phase_events(log, ood_answers)?.iter();
        let ood_answer_tidx = take_v19_ext_events(&mut events, module.profile.num_ood, false)?;
        ensure_no_v19_events(events)?;

        let shift_phase = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::ShiftIndices,
        )?;
        let shift_events = v19_phase_events(log, shift_phase)?;
        if shift_events.len() != module.profile.num_shift_queries {
            return Err(DirectAirVaccVerifierErrorV19::Transcript(
                "shift event count",
            ));
        }
        let shifts = shift_events
            .iter()
            .map(|event| match event.kind {
                TranscriptEventKind::SampleBits { bits, result }
                    if bits == module.profile.log_codeword_len
                        && event.operation_range.end
                            == event.operation_range.start.saturating_add(1) =>
                {
                    Ok(DirectAirSampleBitsScheduleV19 {
                        operation_range: event.operation_range.clone(),
                        result,
                    })
                }
                _ => Err(DirectAirVaccVerifierErrorV19::Transcript(
                    "shift sampling event",
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let batching = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::BatchingChallenge,
        )?;
        let mut events = v19_phase_events(log, batching)?.iter();
        let xi_tidx = take_v19_ext_events(
            &mut events,
            module.profile.batching_arity.ilog2() as usize,
            true,
        )?;
        ensure_no_v19_events(events)?;
        let target = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::FinalTarget,
        )?;
        let mut events = v19_phase_events(log, target)?.iter();
        let mu_tidx = take_v19_ext_event(&mut events, false)?;
        ensure_no_v19_events(events)?;
        let end_tag_tidx = target.operation_range.end;
        if end_tag_tidx.checked_add(1 + D_EF) != Some(end)
            || log.values().get(end_tag_tidx).copied()
                != Some(F::from_u64(SHARD_TRANSCRIPT_VACC_END_TAG_V19))
            || log.samples().get(end_tag_tidx).copied() != Some(false)
            || log
                .samples()
                .get(end_tag_tidx + 1..end)
                .is_none_or(|samples| samples.iter().any(|&sample| !sample))
        {
            return Err(DirectAirVaccVerifierErrorV19::Transcript(
                "VACC transcript terminator",
            ));
        }
        let discarded_end_sample = EF::from_basis_coefficients_slice(
            log.values()
                .get(end_tag_tidx + 1..end)
                .ok_or(DirectAirVaccVerifierErrorV19::Transcript("VACC end sample"))?,
        )
        .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
            "VACC end sample degree",
        ))?;

        let mut claimed = vec![false; log.len()];
        for phase in &verification.transcript_phases {
            mark_v19_claimed(&mut claimed, phase.operation_range.clone())?;
        }
        mark_v19_claimed(&mut claimed, end_tag_tidx..end)?;
        let schedule = Self {
            proof_idx: producer.proof_index as usize,
            start_tidx: start,
            fresh_root_tidx,
            fresh_commitment_extra_tidx,
            fresh_alpha_tidx,
            fresh_mu_tidx,
            fresh_beta_tail_tidx,
            fresh_tau_tidx,
            prior,
            omega_tidx,
            selector_tidx,
            gamma_tidx,
            output_root_tidx,
            nu_tidx,
            eta_tidx,
            ood_point_tidx,
            ood_answer_tidx,
            shifts,
            xi_tidx,
            mu_tidx,
            end_tag_tidx,
            end_tidx: end,
            discarded_end_sample,
            claimed,
        };
        schedule.validate_values(module, record)?;
        Ok(schedule)
    }

    fn from_exact_record(
        module: &DirectAirVaccVerifierModuleV19,
        record: &DirectAirExactFiniteVaccVerifierRecordV19<'_>,
    ) -> Result<Self, DirectAirVaccVerifierErrorV19> {
        let exact = module.exact_config()?;
        let call = exact.call();
        let proof_idx = exact.call_index;
        let verification = record.verification;
        let log = record.transcript;
        let start = record.start_checkpoint.operation_index as usize;
        let end = record.end_checkpoint.operation_index as usize;
        let selector_rounds = call.input_arity.ilog2() as usize;
        if record.proof.fresh_claims.len() != call.fresh_count
            || verification.fresh_authentication.len() != call.fresh_count
            || record.prior.is_some() != (call.prior_count == 1)
            || verification.prior_authentication.is_some() != (call.prior_count == 1)
            || verification.twin.fresh_taus.len() != call.fresh_count
            || verification
                .twin
                .fresh_taus
                .iter()
                .any(|tau| tau.len() != module.profile.log_constraints)
            || verification.twin.selector_tau.len() != selector_rounds
            || verification.twin.gamma.len() != selector_rounds
            || verification.twin.zeta_0.len() != module.profile.log_codeword_len
            || verification.twin.output_beta.len() != module.profile.beta_len
            || verification.ood.len() != module.profile.num_ood
            || verification.shifts.len() != module.profile.num_shift_queries
            || verification.batching.alpha.len() != module.profile.log_codeword_len
            || verification.batching.xi.len() != module.profile.batching_arity.ilog2() as usize
        {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "exact-finite VACC dimensions",
            ));
        }
        validate_v19_phase_ranges(log, &verification.transcript_phases, start, end)?;
        let gamma_tidx = validate_v19_sumcheck_spans(
            log,
            &verification.transcript_phases,
            &verification.twin.sumcheck,
            NativeSumcheckKind::TwinConstraint,
        )?;
        let batching_alpha_tidx = validate_v19_sumcheck_spans(
            log,
            &verification.transcript_phases,
            &verification.batching.sumcheck,
            NativeSumcheckKind::MultilinearBatching,
        )?;
        if gamma_tidx.len() != selector_rounds
            || batching_alpha_tidx.len() != module.profile.log_codeword_len
        {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "exact-finite sumcheck round count",
            ));
        }

        let protocol = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::VaccProtocolPrefix,
        )?;
        if protocol.operation_range.start != start {
            return Err(DirectAirVaccVerifierErrorV19::Transcript(
                "exact-finite VACC prefix start",
            ));
        }
        let commitments = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::FreshCommitments,
        )?;
        let mut events = v19_phase_events(log, commitments)?.iter();
        let commitment_tidx =
            take_v19_ext_events(&mut events, call.fresh_count * (9 + DIGEST_SIZE), false)?;
        ensure_no_v19_events(events)?;
        let fresh_root_tidx =
            *commitment_tidx
                .first()
                .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
                    "empty stacked commitment phase",
                ))?;
        let fresh_commitment_extra_tidx = commitment_tidx[1..].to_vec();

        let fresh_claims = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::FreshClaims,
        )?;
        let mut events = v19_phase_events(log, fresh_claims)?.iter();
        let fresh_alpha_tidx = take_v19_ext_events(
            &mut events,
            call.fresh_count * module.profile.log_codeword_len,
            false,
        )?;
        let fresh_mu_tidx = take_v19_ext_events(&mut events, call.fresh_count, false)?;
        let fresh_beta_tail_tidx = take_v19_ext_events(
            &mut events,
            call.fresh_count * (module.profile.beta_len - module.profile.log_constraints),
            false,
        )?;
        ensure_no_v19_events(events)?;

        let prior_phase = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::PriorAccumulator,
        )?;
        let mut events = v19_phase_events(log, prior_phase)?.iter();
        let prior = if call.prior_count == 1 {
            Some(DirectAirPriorTranscriptScheduleV19 {
                root_tidx: take_v19_lifted_digest(&mut events)?,
                alpha_tidx: take_v19_ext_events(
                    &mut events,
                    module.profile.log_codeword_len,
                    false,
                )?,
                mu_tidx: take_v19_ext_event(&mut events, false)?,
                beta_tidx: take_v19_ext_events(&mut events, module.profile.beta_len, false)?,
                eta_tidx: take_v19_ext_event(&mut events, false)?,
            })
        } else {
            None
        };
        ensure_no_v19_events(events)?;

        let tau_phase = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::FreshZeroCheckPoints,
        )?;
        let mut events = v19_phase_events(log, tau_phase)?.iter();
        let fresh_tau_tidx = take_v19_ext_events(
            &mut events,
            call.fresh_count * module.profile.log_constraints,
            true,
        )?;
        ensure_no_v19_events(events)?;

        let twin = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::TwinChallenges,
        )?;
        let mut events = v19_phase_events(log, twin)?.iter();
        let omega_tidx = take_v19_ext_event(&mut events, true)?;
        let selector_tidx = take_v19_ext_events(&mut events, selector_rounds, true)?;
        ensure_no_v19_events(events)?;

        let output = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::OutputAccumulatorCommitment,
        )?;
        let mut events = v19_phase_events(log, output)?.iter();
        let output_root_tidx = take_v19_lifted_digest(&mut events)?;
        let nu_tidx = take_v19_ext_event(&mut events, false)?;
        let eta_tidx = take_v19_ext_event(&mut events, false)?;
        ensure_no_v19_events(events)?;

        let ood_points = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::OodPoints,
        )?;
        let mut events = v19_phase_events(log, ood_points)?.iter();
        let ood_point_tidx = (0..module.profile.num_ood)
            .map(|_| {
                take_v19_ext_events(&mut events, module.profile.log_codeword_len, true)?
                    .first()
                    .copied()
                    .ok_or(DirectAirVaccVerifierErrorV19::Transcript("empty OOD point"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        ensure_no_v19_events(events)?;
        let ood_answers = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::OodAnswers,
        )?;
        let mut events = v19_phase_events(log, ood_answers)?.iter();
        let ood_answer_tidx = take_v19_ext_events(&mut events, module.profile.num_ood, false)?;
        ensure_no_v19_events(events)?;

        let shift_phase = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::ShiftIndices,
        )?;
        let shift_events = v19_phase_events(log, shift_phase)?;
        if shift_events.len() != module.profile.num_shift_queries {
            return Err(DirectAirVaccVerifierErrorV19::Transcript(
                "shift event count",
            ));
        }
        let shifts = shift_events
            .iter()
            .map(|event| match event.kind {
                TranscriptEventKind::SampleBits { bits, result }
                    if bits == module.profile.log_codeword_len
                        && event.operation_range.end
                            == event.operation_range.start.saturating_add(1) =>
                {
                    Ok(DirectAirSampleBitsScheduleV19 {
                        operation_range: event.operation_range.clone(),
                        result,
                    })
                }
                _ => Err(DirectAirVaccVerifierErrorV19::Transcript(
                    "shift sampling event",
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let batching = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::BatchingChallenge,
        )?;
        let mut events = v19_phase_events(log, batching)?.iter();
        let xi_tidx = take_v19_ext_events(
            &mut events,
            module.profile.batching_arity.ilog2() as usize,
            true,
        )?;
        ensure_no_v19_events(events)?;
        let target = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::FinalTarget,
        )?;
        let mut events = v19_phase_events(log, target)?.iter();
        let mu_tidx = take_v19_ext_event(&mut events, false)?;
        ensure_no_v19_events(events)?;
        let boundary = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::ExactFiniteCallBoundary {
                call: proof_idx.try_into().map_err(|_| {
                    DirectAirVaccVerifierErrorV19::Transcript(
                        "exact-finite boundary call-index width",
                    )
                })?,
            },
        )?;
        let end_tag_tidx = boundary.operation_range.start;
        let sample_tidx =
            end_tag_tidx
                .checked_add(D_EF)
                .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
                    "exact-finite end-tag overflow",
                ))?;
        let expected_tag = EF::from_u64(EXACT_FINITE_WARP_CALL_END_TAG);
        if end_tag_tidx != target.operation_range.end
            || boundary.operation_range.end != end
            || sample_tidx.checked_add(D_EF) != Some(end)
            || log.values().get(end_tag_tidx..sample_tidx)
                != Some(expected_tag.as_basis_coefficients_slice())
            || log
                .samples()
                .get(end_tag_tidx..sample_tidx)
                .is_none_or(|samples| samples.iter().any(|&sample| sample))
            || log
                .samples()
                .get(sample_tidx..end)
                .is_none_or(|samples| samples.iter().any(|&sample| !sample))
        {
            return Err(DirectAirVaccVerifierErrorV19::Transcript(
                "exact-finite VACC transcript terminator",
            ));
        }
        let discarded_end_sample =
            EF::from_basis_coefficients_slice(log.values().get(sample_tidx..end).ok_or(
                DirectAirVaccVerifierErrorV19::Transcript("exact-finite VACC end sample"),
            )?)
            .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
                "exact-finite VACC end sample degree",
            ))?;
        let mut claimed = vec![false; log.len()];
        for phase in &verification.transcript_phases {
            mark_v19_claimed(&mut claimed, phase.operation_range.clone())?;
        }
        let schedule = Self {
            proof_idx,
            start_tidx: start,
            fresh_root_tidx,
            fresh_commitment_extra_tidx,
            fresh_alpha_tidx,
            fresh_mu_tidx,
            fresh_beta_tail_tidx,
            fresh_tau_tidx,
            prior,
            omega_tidx,
            selector_tidx,
            gamma_tidx,
            output_root_tidx,
            nu_tidx,
            eta_tidx,
            ood_point_tidx,
            ood_answer_tidx,
            shifts,
            xi_tidx,
            mu_tidx,
            end_tag_tidx,
            end_tidx: end,
            discarded_end_sample,
            claimed,
        };
        schedule.validate_exact_values(module, record)?;
        Ok(schedule)
    }

    fn from_reduced_swirl_record(
        module: &DirectAirVaccVerifierModuleV19,
        record: &DirectAirReducedSwirlVaccVerifierRecordV19<'_>,
    ) -> Result<Self, DirectAirVaccVerifierErrorV19> {
        let reduced = module
            .reduced_swirl
            .as_ref()
            .ok_or(DirectAirVaccVerifierErrorV19::InvalidProfile)?;
        let verification = record.verification;
        let log = record.transcript;
        let fresh_count = record.proof.inner().fresh_claims.len();
        let has_prior = record.prior.is_some();
        let selector_rounds = reduced.input_arity.ilog2() as usize;
        if fresh_count == 0
            || fresh_count + usize::from(has_prior) > reduced.input_arity
            || verification.fresh_authentication.len() != fresh_count
            || verification.prior_authentication.is_some() != has_prior
            || verification.twin.fresh_taus.len() != fresh_count
            || verification
                .twin
                .fresh_taus
                .iter()
                .any(|tau| tau.len() != module.profile.log_constraints)
            || verification.twin.selector_tau.len() != selector_rounds
            || verification.twin.gamma.len() != selector_rounds
            || verification.twin.zeta_0.len() != module.profile.log_codeword_len
            || verification.twin.output_beta.len() != module.profile.beta_len
            || verification.ood.len() != module.profile.num_ood
            || verification.shifts.len() != module.profile.num_shift_queries
            || verification.batching.alpha.len() != module.profile.log_codeword_len
            || verification.batching.xi.len() != module.profile.batching_arity.ilog2() as usize
            || record.commitment_tidxs.len() != fresh_count
            || has_prior != (record.proof_idx != 0)
        {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "reduced-SWIRL VACC dimensions",
            ));
        }
        validate_v19_phase_ranges(
            log,
            &verification.transcript_phases,
            record.vacc_start_tidx,
            record.vacc_end_tidx,
        )?;
        let gamma_tidx = validate_v19_sumcheck_spans(
            log,
            &verification.transcript_phases,
            &verification.twin.sumcheck,
            NativeSumcheckKind::TwinConstraint,
        )?;
        let batching_alpha_tidx = validate_v19_sumcheck_spans(
            log,
            &verification.transcript_phases,
            &verification.batching.sumcheck,
            NativeSumcheckKind::MultilinearBatching,
        )?;
        if gamma_tidx.len() != selector_rounds
            || batching_alpha_tidx.len() != module.profile.log_codeword_len
        {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "reduced-SWIRL sumcheck rounds",
            ));
        }
        let protocol = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::VaccProtocolPrefix,
        )?;
        if protocol.operation_range.start != record.vacc_start_tidx {
            return Err(DirectAirVaccVerifierErrorV19::Transcript(
                "reduced-SWIRL VACC prefix start",
            ));
        }
        let commitments = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::FreshCommitments,
        )?;
        let commitment_events = v19_phase_events(log, commitments)?;
        let expected_commitment_events = record
            .proof
            .inner()
            .fresh_claims
            .iter()
            .try_fold(0usize, |total, claim| {
                total.checked_add(9 + claim.commitment.roots.len() * (1 + DIGEST_SIZE))
            })
            .ok_or(DirectAirVaccVerifierErrorV19::RecordShape(
                "reduced-SWIRL commitment event count",
            ))?;
        if commitment_events.len() != expected_commitment_events {
            return Err(DirectAirVaccVerifierErrorV19::Transcript(
                "reduced-SWIRL commitment events",
            ));
        }
        let commitment_tidx = commitment_events
            .iter()
            .map(|event| match event.kind {
                TranscriptEventKind::ObserveExt { degree: D_EF }
                    if event.operation_range.len() == D_EF =>
                {
                    Ok(event.operation_range.start)
                }
                _ => Err(DirectAirVaccVerifierErrorV19::Transcript(
                    "reduced-SWIRL commitment event kind",
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut offset = 0usize;
        for (source, claim) in record.proof.inner().fresh_claims.iter().enumerate() {
            if commitment_tidx[offset] != record.commitment_tidxs[source] {
                return Err(DirectAirVaccVerifierErrorV19::Transcript(
                    "reduced-SWIRL commitment descriptor boundary",
                ));
            }
            offset += 9 + claim.commitment.roots.len() * (1 + DIGEST_SIZE);
        }
        let fresh_root_tidx = commitment_tidx[0];
        let fresh_commitment_extra_tidx = commitment_tidx[1..].to_vec();

        let fresh_claims = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::FreshClaims,
        )?;
        let mut events = v19_phase_events(log, fresh_claims)?.iter();
        let fresh_alpha_tidx = take_v19_ext_events(
            &mut events,
            fresh_count * module.profile.log_codeword_len,
            false,
        )?;
        let fresh_mu_tidx = take_v19_ext_events(&mut events, fresh_count, false)?;
        let fresh_beta_tail_tidx = take_v19_ext_events(
            &mut events,
            fresh_count * (module.profile.beta_len - module.profile.log_constraints),
            false,
        )?;
        ensure_no_v19_events(events)?;
        let prior_phase = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::PriorAccumulator,
        )?;
        let mut events = v19_phase_events(log, prior_phase)?.iter();
        let prior = has_prior
            .then(|| {
                Ok(DirectAirPriorTranscriptScheduleV19 {
                    root_tidx: take_v19_lifted_digest(&mut events)?,
                    alpha_tidx: take_v19_ext_events(
                        &mut events,
                        module.profile.log_codeword_len,
                        false,
                    )?,
                    mu_tidx: take_v19_ext_event(&mut events, false)?,
                    beta_tidx: take_v19_ext_events(&mut events, module.profile.beta_len, false)?,
                    eta_tidx: take_v19_ext_event(&mut events, false)?,
                })
            })
            .transpose()?;
        ensure_no_v19_events(events)?;
        let tau_phase = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::FreshZeroCheckPoints,
        )?;
        let mut events = v19_phase_events(log, tau_phase)?.iter();
        let fresh_tau_tidx = take_v19_ext_events(
            &mut events,
            fresh_count * module.profile.log_constraints,
            true,
        )?;
        ensure_no_v19_events(events)?;
        let twin = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::TwinChallenges,
        )?;
        let mut events = v19_phase_events(log, twin)?.iter();
        let omega_tidx = take_v19_ext_event(&mut events, true)?;
        let selector_tidx = take_v19_ext_events(&mut events, selector_rounds, true)?;
        ensure_no_v19_events(events)?;
        let output = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::OutputAccumulatorCommitment,
        )?;
        let mut events = v19_phase_events(log, output)?.iter();
        let output_root_tidx = take_v19_lifted_digest(&mut events)?;
        let nu_tidx = take_v19_ext_event(&mut events, false)?;
        let eta_tidx = take_v19_ext_event(&mut events, false)?;
        ensure_no_v19_events(events)?;
        let ood_points = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::OodPoints,
        )?;
        let mut events = v19_phase_events(log, ood_points)?.iter();
        let ood_point_tidx = (0..module.profile.num_ood)
            .map(|_| {
                take_v19_ext_events(&mut events, module.profile.log_codeword_len, true)?
                    .first()
                    .copied()
                    .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
                        "reduced-SWIRL empty OOD point",
                    ))
            })
            .collect::<Result<Vec<_>, _>>()?;
        ensure_no_v19_events(events)?;
        let ood_answers = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::OodAnswers,
        )?;
        let mut events = v19_phase_events(log, ood_answers)?.iter();
        let ood_answer_tidx = take_v19_ext_events(&mut events, module.profile.num_ood, false)?;
        ensure_no_v19_events(events)?;
        let shift_phase = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::ShiftIndices,
        )?;
        let shift_events = v19_phase_events(log, shift_phase)?;
        if shift_events.len() != module.profile.num_shift_queries {
            return Err(DirectAirVaccVerifierErrorV19::Transcript(
                "reduced-SWIRL shift event count",
            ));
        }
        let shifts = shift_events
            .iter()
            .map(|event| match event.kind {
                TranscriptEventKind::SampleBits { bits, result }
                    if bits == module.profile.log_codeword_len
                        && event.operation_range.len() == 1 =>
                {
                    Ok(DirectAirSampleBitsScheduleV19 {
                        operation_range: event.operation_range.clone(),
                        result,
                    })
                }
                _ => Err(DirectAirVaccVerifierErrorV19::Transcript(
                    "reduced-SWIRL shift sampling event",
                )),
            })
            .collect::<Result<Vec<_>, _>>()?;
        let batching = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::BatchingChallenge,
        )?;
        let mut events = v19_phase_events(log, batching)?.iter();
        let xi_tidx = take_v19_ext_events(
            &mut events,
            module.profile.batching_arity.ilog2() as usize,
            true,
        )?;
        ensure_no_v19_events(events)?;
        let target = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::FinalTarget,
        )?;
        let mut events = v19_phase_events(log, target)?.iter();
        let mu_tidx = take_v19_ext_event(&mut events, false)?;
        ensure_no_v19_events(events)?;
        let boundary = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::ExactFiniteCallBoundary {
                call: record.proof_idx.try_into().map_err(|_| {
                    DirectAirVaccVerifierErrorV19::Transcript(
                        "reduced-SWIRL boundary call-index width",
                    )
                })?,
            },
        )?;
        let end_tag_tidx = boundary.operation_range.start;
        let sample_tidx =
            end_tag_tidx
                .checked_add(D_EF)
                .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
                    "reduced-SWIRL end-tag overflow",
                ))?;
        let expected_tag = EF::from_u64(EXACT_FINITE_WARP_CALL_END_TAG);
        if end_tag_tidx != target.operation_range.end
            || boundary.operation_range.end != record.vacc_end_tidx
            || sample_tidx.checked_add(D_EF) != Some(record.vacc_end_tidx)
            || log.values().get(end_tag_tidx..sample_tidx)
                != Some(expected_tag.as_basis_coefficients_slice())
            || log
                .samples()
                .get(end_tag_tidx..sample_tidx)
                .is_none_or(|samples| samples.iter().any(|&sample| sample))
            || log
                .samples()
                .get(sample_tidx..record.vacc_end_tidx)
                .is_none_or(|samples| samples.iter().any(|&sample| !sample))
        {
            return Err(DirectAirVaccVerifierErrorV19::Transcript(
                "reduced-SWIRL VACC transcript terminator",
            ));
        }
        let discarded_end_sample = EF::from_basis_coefficients_slice(
            log.values().get(sample_tidx..record.vacc_end_tidx).ok_or(
                DirectAirVaccVerifierErrorV19::Transcript("reduced-SWIRL VACC end sample"),
            )?,
        )
        .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
            "reduced-SWIRL VACC end sample degree",
        ))?;
        let mut claimed = vec![false; log.len()];
        for phase in &verification.transcript_phases {
            mark_v19_claimed(&mut claimed, phase.operation_range.clone())?;
        }
        let schedule = Self {
            proof_idx: record.local_proof_idx,
            start_tidx: record.vacc_start_tidx,
            fresh_root_tidx,
            fresh_commitment_extra_tidx,
            fresh_alpha_tidx,
            fresh_mu_tidx,
            fresh_beta_tail_tidx,
            fresh_tau_tidx,
            prior,
            omega_tidx,
            selector_tidx,
            gamma_tidx,
            output_root_tidx,
            nu_tidx,
            eta_tidx,
            ood_point_tidx,
            ood_answer_tidx,
            shifts,
            xi_tidx,
            mu_tidx,
            end_tag_tidx,
            end_tidx: record.vacc_end_tidx,
            discarded_end_sample,
            claimed,
        };
        schedule.validate_reduced_values(module, record)?;
        Ok(schedule)
    }

    fn validate_reduced_values(
        &self,
        module: &DirectAirVaccVerifierModuleV19,
        record: &DirectAirReducedSwirlVaccVerifierRecordV19<'_>,
    ) -> Result<(), DirectAirVaccVerifierErrorV19> {
        let proof = record.proof.inner();
        let verification = record.verification;
        let alpha_len = module.profile.log_codeword_len;
        let tau_len = module.profile.log_constraints;
        let beta_tail_len = module.profile.beta_len - tau_len;
        for (source, claim) in proof.fresh_claims.iter().enumerate() {
            if claim.alpha.len() != alpha_len
                || claim.beta.len() != module.profile.beta_len
                || claim.eta != EF::ZERO
            {
                return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                    "reduced-SWIRL fresh claim dimensions",
                ));
            }
            for (coordinate, &value) in claim.alpha.iter().enumerate() {
                expect_v19_ext(
                    record.transcript,
                    self.fresh_alpha_tidx[source * alpha_len + coordinate],
                    value,
                )?;
            }
            expect_v19_ext(record.transcript, self.fresh_mu_tidx[source], claim.mu)?;
            for (coordinate, &value) in claim.beta[tau_len..].iter().enumerate() {
                expect_v19_ext(
                    record.transcript,
                    self.fresh_beta_tail_tidx[source * beta_tail_len + coordinate],
                    value,
                )?;
            }
            for (coordinate, (&sampled, &retained)) in verification.twin.fresh_taus[source]
                .iter()
                .zip(&claim.beta[..tau_len])
                .enumerate()
            {
                expect_v19_ext(
                    record.transcript,
                    self.fresh_tau_tidx[source * tau_len + coordinate],
                    sampled,
                )?;
                if sampled != retained {
                    return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                        "reduced-SWIRL fresh beta/tau prefix",
                    ));
                }
            }
        }
        expect_v19_ext(record.transcript, self.omega_tidx, verification.twin.omega)?;
        for (tidx, &value) in self
            .selector_tidx
            .iter()
            .zip(&verification.twin.selector_tau)
        {
            expect_v19_ext(record.transcript, *tidx, value)?;
        }
        for (tidx, &value) in self.gamma_tidx.iter().zip(&verification.twin.gamma) {
            expect_v19_ext(record.transcript, *tidx, value)?;
        }
        expect_v19_digest(
            record.transcript,
            self.output_root_tidx,
            &verification.output_instance.rt,
        )?;
        expect_v19_ext(record.transcript, self.nu_tidx, verification.twin.nu_0)?;
        expect_v19_ext(record.transcript, self.eta_tidx, verification.twin.eta)?;
        for (ordinal, ood) in verification.ood.iter().enumerate() {
            for (coordinate, &value) in ood.point.iter().enumerate() {
                expect_v19_ext(
                    record.transcript,
                    self.ood_point_tidx[ordinal] + coordinate * D_EF,
                    value,
                )?;
            }
            expect_v19_ext(record.transcript, self.ood_answer_tidx[ordinal], ood.answer)?;
        }
        for (sample, shift) in self.shifts.iter().zip(&verification.shifts) {
            if sample.result != u64::from(shift.index) {
                return Err(DirectAirVaccVerifierErrorV19::Transcript(
                    "reduced-SWIRL shift result",
                ));
            }
        }
        for (tidx, &value) in self.xi_tidx.iter().zip(&verification.batching.xi) {
            expect_v19_ext(record.transcript, *tidx, value)?;
        }
        expect_v19_ext(
            record.transcript,
            self.mu_tidx,
            verification.batching.mu_final,
        )?;
        if let (Some(prior_schedule), Some(prior)) = (self.prior.as_ref(), record.prior) {
            expect_v19_digest(record.transcript, prior_schedule.root_tidx, &prior.rt)?;
            for (tidx, &value) in prior_schedule.alpha_tidx.iter().zip(&prior.alpha) {
                expect_v19_ext(record.transcript, *tidx, value)?;
            }
            expect_v19_ext(record.transcript, prior_schedule.mu_tidx, prior.mu)?;
            for (tidx, &value) in prior_schedule.beta_tidx.iter().zip(&prior.beta) {
                expect_v19_ext(record.transcript, *tidx, value)?;
            }
            expect_v19_ext(record.transcript, prior_schedule.eta_tidx, prior.eta)?;
        }
        Ok(())
    }

    fn validate_values<FreshVerification: DirectAirFreshAuthenticationV19>(
        &self,
        module: &DirectAirVaccVerifierModuleV19,
        record: &DirectAirVaccVerifierRecordV19<'_, FreshVerification>,
    ) -> Result<(), DirectAirVaccVerifierErrorV19> {
        let p = record.producer;
        let v = record.verification;
        expect_v19_digest(record.transcript, self.fresh_root_tidx, &p.fresh_root)?;
        for (tidx, value) in self.fresh_alpha_tidx.iter().zip(&p.fresh_alpha) {
            expect_v19_ext_array(record.transcript, *tidx, value)?;
        }
        expect_v19_ext_array(record.transcript, self.fresh_mu_tidx[0], &p.fresh_mu)?;
        for (tidx, value) in self
            .fresh_beta_tail_tidx
            .iter()
            .zip(&p.fresh_beta[module.profile.log_constraints..])
        {
            expect_v19_ext_array(record.transcript, *tidx, value)?;
        }
        for ((tidx, expected), retained) in self
            .fresh_tau_tidx
            .iter()
            .zip(&v.twin.fresh_taus[0])
            .zip(&p.fresh_beta[..module.profile.log_constraints])
        {
            expect_v19_ext(record.transcript, *tidx, *expected)?;
            if EF::from_basis_coefficients_slice(retained) != Some(*expected) {
                return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                    "fresh beta tau prefix",
                ));
            }
        }
        expect_v19_ext(record.transcript, self.omega_tidx, v.twin.omega)?;
        expect_v19_ext(
            record.transcript,
            self.selector_tidx[0],
            v.twin.selector_tau[0],
        )?;
        expect_v19_ext(record.transcript, self.gamma_tidx[0], v.twin.gamma[0])?;
        expect_v19_digest(
            record.transcript,
            self.output_root_tidx,
            &v.output_instance.rt,
        )?;
        expect_v19_ext(record.transcript, self.nu_tidx, v.twin.nu_0)?;
        expect_v19_ext(record.transcript, self.eta_tidx, v.twin.eta)?;
        for (ordinal, ood) in v.ood.iter().enumerate() {
            for (coordinate, &value) in ood.point.iter().enumerate() {
                expect_v19_ext(
                    record.transcript,
                    self.ood_point_tidx[ordinal] + coordinate * D_EF,
                    value,
                )?;
            }
            expect_v19_ext(record.transcript, self.ood_answer_tidx[ordinal], ood.answer)?;
        }
        for (sample, shift) in self.shifts.iter().zip(&v.shifts) {
            if sample.result != u64::from(shift.index) {
                return Err(DirectAirVaccVerifierErrorV19::Transcript("shift result"));
            }
        }
        for (tidx, &value) in self.xi_tidx.iter().zip(&v.batching.xi) {
            expect_v19_ext(record.transcript, *tidx, value)?;
        }
        expect_v19_ext(record.transcript, self.mu_tidx, v.batching.mu_final)?;
        if let (Some(prior_schedule), Some(prior)) = (self.prior.as_ref(), record.prior) {
            expect_v19_digest(record.transcript, prior_schedule.root_tidx, &prior.rt)?;
            for (tidx, &value) in prior_schedule.alpha_tidx.iter().zip(&prior.alpha) {
                expect_v19_ext(record.transcript, *tidx, value)?;
            }
            expect_v19_ext(record.transcript, prior_schedule.mu_tidx, prior.mu)?;
            for (tidx, &value) in prior_schedule.beta_tidx.iter().zip(&prior.beta) {
                expect_v19_ext(record.transcript, *tidx, value)?;
            }
            expect_v19_ext(record.transcript, prior_schedule.eta_tidx, prior.eta)?;
        }
        Ok(())
    }

    fn validate_exact_values(
        &self,
        module: &DirectAirVaccVerifierModuleV19,
        record: &DirectAirExactFiniteVaccVerifierRecordV19<'_>,
    ) -> Result<(), DirectAirVaccVerifierErrorV19> {
        let proof = record.proof;
        let verification = record.verification;
        let fresh_count = module.fresh_count();
        let alpha_len = module.profile.log_codeword_len;
        let beta_tail_len = module.profile.beta_len - module.profile.log_constraints;
        for (source, claim) in proof.fresh_claims.iter().enumerate() {
            for (coordinate, &value) in claim.alpha.iter().enumerate() {
                expect_v19_ext(
                    record.transcript,
                    self.fresh_alpha_tidx[source * alpha_len + coordinate],
                    value,
                )?;
            }
            expect_v19_ext(record.transcript, self.fresh_mu_tidx[source], claim.mu)?;
            for (coordinate, &value) in claim.beta[module.profile.log_constraints..]
                .iter()
                .enumerate()
            {
                expect_v19_ext(
                    record.transcript,
                    self.fresh_beta_tail_tidx[source * beta_tail_len + coordinate],
                    value,
                )?;
            }
            for (coordinate, (&expected, &retained)) in verification.twin.fresh_taus[source]
                .iter()
                .zip(&claim.beta[..module.profile.log_constraints])
                .enumerate()
            {
                expect_v19_ext(
                    record.transcript,
                    self.fresh_tau_tidx[source * module.profile.log_constraints + coordinate],
                    expected,
                )?;
                if retained != expected {
                    return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                        "fresh beta tau prefix",
                    ));
                }
            }
        }
        if proof.fresh_claims.len() != fresh_count {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "fresh claim count",
            ));
        }
        expect_v19_ext(record.transcript, self.omega_tidx, verification.twin.omega)?;
        for (tidx, &value) in self
            .selector_tidx
            .iter()
            .zip(&verification.twin.selector_tau)
        {
            expect_v19_ext(record.transcript, *tidx, value)?;
        }
        for (tidx, &value) in self.gamma_tidx.iter().zip(&verification.twin.gamma) {
            expect_v19_ext(record.transcript, *tidx, value)?;
        }
        expect_v19_digest(
            record.transcript,
            self.output_root_tidx,
            &verification.output_instance.rt,
        )?;
        expect_v19_ext(record.transcript, self.nu_tidx, verification.twin.nu_0)?;
        expect_v19_ext(record.transcript, self.eta_tidx, verification.twin.eta)?;
        for (ordinal, ood) in verification.ood.iter().enumerate() {
            for (coordinate, &value) in ood.point.iter().enumerate() {
                expect_v19_ext(
                    record.transcript,
                    self.ood_point_tidx[ordinal] + coordinate * D_EF,
                    value,
                )?;
            }
            expect_v19_ext(record.transcript, self.ood_answer_tidx[ordinal], ood.answer)?;
        }
        for (sample, shift) in self.shifts.iter().zip(&verification.shifts) {
            if sample.result != u64::from(shift.index) {
                return Err(DirectAirVaccVerifierErrorV19::Transcript("shift result"));
            }
        }
        for (tidx, &value) in self.xi_tidx.iter().zip(&verification.batching.xi) {
            expect_v19_ext(record.transcript, *tidx, value)?;
        }
        expect_v19_ext(
            record.transcript,
            self.mu_tidx,
            verification.batching.mu_final,
        )?;
        if let (Some(prior_schedule), Some(prior)) = (self.prior.as_ref(), record.prior) {
            expect_v19_digest(record.transcript, prior_schedule.root_tidx, &prior.rt)?;
            for (tidx, &value) in prior_schedule.alpha_tidx.iter().zip(&prior.alpha) {
                expect_v19_ext(record.transcript, *tidx, value)?;
            }
            expect_v19_ext(record.transcript, prior_schedule.mu_tidx, prior.mu)?;
            for (tidx, &value) in prior_schedule.beta_tidx.iter().zip(&prior.beta) {
                expect_v19_ext(record.transcript, *tidx, value)?;
            }
            expect_v19_ext(record.transcript, prior_schedule.eta_tidx, prior.eta)?;
        }
        Ok(())
    }
}

fn validate_v19_phase_ranges(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    phases: &[NativeTranscriptPhaseSpan],
    start: usize,
    end: usize,
) -> Result<(), DirectAirVaccVerifierErrorV19> {
    let mut previous_event = phases.first().map_or(0, |phase| phase.event_range.start);
    let mut previous_operation = start;
    let mut previous_permutation = phases
        .first()
        .map_or(0, |phase| phase.permutation_range.start);
    for phase in phases {
        if phase.event_range.start != previous_event
            || phase.operation_range.start != previous_operation
            || phase.permutation_range.start != previous_permutation
            || phase.event_range.end > log.events().len()
            || phase.operation_range.end > log.len()
            || phase.operation_range.end > end
            || phase.permutation_range.end > log.permutation_transitions().len()
            || phase.event_range.start > phase.event_range.end
            || phase.operation_range.start > phase.operation_range.end
            || phase.permutation_range.start > phase.permutation_range.end
        {
            return Err(DirectAirVaccVerifierErrorV19::Transcript(
                "non-canonical phase ranges",
            ));
        }
        let events = v19_phase_events(log, phase)?;
        if events.first().map(|event| event.operation_range.start)
            != (!events.is_empty()).then_some(phase.operation_range.start)
            || events.last().map(|event| event.operation_range.end)
                != (!events.is_empty()).then_some(phase.operation_range.end)
        {
            return Err(DirectAirVaccVerifierErrorV19::Transcript(
                "phase event boundaries",
            ));
        }
        previous_event = phase.event_range.end;
        previous_operation = phase.operation_range.end;
        previous_permutation = phase.permutation_range.end;
    }
    Ok(())
}

fn validate_v19_sumcheck_spans(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    phases: &[NativeTranscriptPhaseSpan],
    verification: &openvm_stark_backend::warp_accum::CoefficientSumcheckVerification<EF>,
    kind: NativeSumcheckKind,
) -> Result<Vec<usize>, DirectAirVaccVerifierErrorV19> {
    if verification.kind != kind {
        return Err(DirectAirVaccVerifierErrorV19::Algebra("sumcheck kind"));
    }
    verification
        .rounds
        .iter()
        .map(|round| {
            let phase_kind = match kind {
                NativeSumcheckKind::TwinConstraint => {
                    NativeTranscriptPhase::TwinSumcheckRound { round: round.round }
                }
                NativeSumcheckKind::MultilinearBatching => {
                    NativeTranscriptPhase::BatchingSumcheckRound { round: round.round }
                }
            };
            let phase = unique_v19_phase(phases, &phase_kind)?;
            if phase != &round.transcript_span {
                return Err(DirectAirVaccVerifierErrorV19::Transcript(
                    "sumcheck phase span",
                ));
            }
            let events = v19_phase_events(log, phase)?;
            if events.len() != round.coefficients.len() + 1 {
                return Err(DirectAirVaccVerifierErrorV19::Transcript(
                    "sumcheck event count",
                ));
            }
            for event in &events[..round.coefficients.len()] {
                v19_ext_event_tidx(event, false)?;
            }
            v19_ext_event_tidx(&events[round.coefficients.len()], true)
        })
        .collect()
}

fn unique_v19_phase<'a>(
    phases: &'a [NativeTranscriptPhaseSpan],
    wanted: &NativeTranscriptPhase,
) -> Result<&'a NativeTranscriptPhaseSpan, DirectAirVaccVerifierErrorV19> {
    let mut matches = phases.iter().filter(|phase| &phase.phase == wanted);
    let phase = matches
        .next()
        .ok_or(DirectAirVaccVerifierErrorV19::Transcript("missing phase"))?;
    if matches.next().is_some() {
        return Err(DirectAirVaccVerifierErrorV19::Transcript("duplicate phase"));
    }
    Ok(phase)
}

fn v19_phase_events<'a>(
    log: &'a TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    phase: &NativeTranscriptPhaseSpan,
) -> Result<&'a [TranscriptEvent], DirectAirVaccVerifierErrorV19> {
    log.events()
        .get(phase.event_range.clone())
        .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
            "phase event range",
        ))
}

fn take_v19_ext_events<'a>(
    events: &mut impl Iterator<Item = &'a TranscriptEvent>,
    count: usize,
    sampled: bool,
) -> Result<Vec<usize>, DirectAirVaccVerifierErrorV19> {
    (0..count)
        .map(|_| take_v19_ext_event(events, sampled))
        .collect()
}

fn take_v19_ext_event<'a>(
    events: &mut impl Iterator<Item = &'a TranscriptEvent>,
    sampled: bool,
) -> Result<usize, DirectAirVaccVerifierErrorV19> {
    v19_ext_event_tidx(
        events
            .next()
            .ok_or(DirectAirVaccVerifierErrorV19::Transcript("missing event"))?,
        sampled,
    )
}

fn take_v19_lifted_digest<'a>(
    events: &mut impl Iterator<Item = &'a TranscriptEvent>,
) -> Result<usize, DirectAirVaccVerifierErrorV19> {
    let first = take_v19_ext_event(events, false)?;
    for limb in 1..DIGEST_SIZE {
        if take_v19_ext_event(events, false)? != first + limb * D_EF {
            return Err(DirectAirVaccVerifierErrorV19::Transcript(
                "lifted digest layout",
            ));
        }
    }
    Ok(first)
}

fn ensure_no_v19_events<'a>(
    mut events: impl Iterator<Item = &'a TranscriptEvent>,
) -> Result<(), DirectAirVaccVerifierErrorV19> {
    if events.next().is_some() {
        Err(DirectAirVaccVerifierErrorV19::Transcript("phase tail"))
    } else {
        Ok(())
    }
}

fn v19_ext_event_tidx(
    event: &TranscriptEvent,
    sampled: bool,
) -> Result<usize, DirectAirVaccVerifierErrorV19> {
    let expected = if sampled {
        TranscriptEventKind::SampleExt { degree: D_EF }
    } else {
        TranscriptEventKind::ObserveExt { degree: D_EF }
    };
    if event.kind != expected || event.operation_range.len() != D_EF {
        return Err(DirectAirVaccVerifierErrorV19::Transcript("extension event"));
    }
    Ok(event.operation_range.start)
}

fn mark_v19_claimed(
    claimed: &mut [bool],
    range: core::ops::Range<usize>,
) -> Result<(), DirectAirVaccVerifierErrorV19> {
    let values = claimed
        .get_mut(range)
        .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
            "claimed operation range",
        ))?;
    if values.iter().any(|&value| value) {
        return Err(DirectAirVaccVerifierErrorV19::Transcript(
            "duplicate claimed operation",
        ));
    }
    values.fill(true);
    Ok(())
}

fn expect_v19_ext(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    tidx: usize,
    expected: EF,
) -> Result<(), DirectAirVaccVerifierErrorV19> {
    if log.values().get(tidx..tidx + D_EF) != Some(expected.as_basis_coefficients_slice()) {
        return Err(DirectAirVaccVerifierErrorV19::Transcript(
            "extension value mismatch",
        ));
    }
    Ok(())
}

fn expect_v19_ext_array(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    tidx: usize,
    expected: &[F; D_EF],
) -> Result<(), DirectAirVaccVerifierErrorV19> {
    if log.values().get(tidx..tidx + D_EF) != Some(expected) {
        return Err(DirectAirVaccVerifierErrorV19::Transcript(
            "retained extension mismatch",
        ));
    }
    Ok(())
}

fn expect_v19_digest(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    tidx: usize,
    digest: &Digest,
) -> Result<(), DirectAirVaccVerifierErrorV19> {
    for (limb, &value) in digest.iter().enumerate() {
        expect_v19_ext(
            log,
            tidx + limb * D_EF,
            EF::from_basis_coefficients_slice(&[value, F::ZERO, F::ZERO, F::ZERO])
                .expect("EF4 basis"),
        )?;
    }
    Ok(())
}

struct DirectAirVaccAlgebraTracesV19 {
    claim_values: RowMajorMatrix<F>,
    claim_layout: RowMajorMatrix<F>,
    slot_layout: RowMajorMatrix<F>,
    vector: RowMajorMatrix<F>,
    selector_eq: RowMajorMatrix<F>,
    xi_eq: RowMajorMatrix<F>,
    opening_eq: RowMajorMatrix<F>,
    omega: RowMajorMatrix<F>,
    twin_sigma: RowMajorMatrix<F>,
    sumcheck: RowMajorMatrix<F>,
    twin_fold: RowMajorMatrix<F>,
    twin_final: RowMajorMatrix<F>,
    initial_point: RowMajorMatrix<F>,
    initial_target: RowMajorMatrix<F>,
    opening_padding: RowMajorMatrix<F>,
    ood_claims: RowMajorMatrix<F>,
    shift_schedule: RowMajorMatrix<F>,
    shift_merge: RowMajorMatrix<F>,
    batching_sigma: RowMajorMatrix<F>,
    batching_final: RowMajorMatrix<F>,
    batching_alpha: RowMajorMatrix<F>,
}

#[derive(Clone, Debug)]
struct DirectAirVaccFreshClaimDataV19 {
    alpha: Vec<EF>,
    beta: Vec<EF>,
    mu: EF,
    eta: EF,
}

fn shared_twin_lookup_count_v19(input_arity: usize) -> Result<u32, DirectAirVaccVerifierErrorV19> {
    u32::try_from(
        input_arity
            .checked_add(1)
            .ok_or(DirectAirVaccVerifierErrorV19::InvalidProfile)?,
    )
    .map_err(|_| DirectAirVaccVerifierErrorV19::InvalidProfile)
}

fn direct_vacc_eq_group_offsets_v19(
    module: &DirectAirVaccVerifierModuleV19,
) -> Result<(usize, usize), DirectAirVaccVerifierErrorV19> {
    let xi = *module
        .algebra
        .xi_weight_groups
        .first()
        .ok_or(DirectAirVaccVerifierErrorV19::InvalidProfile)? as usize;
    let opening = *module
        .algebra
        .opening_at_alpha_groups
        .first()
        .ok_or(DirectAirVaccVerifierErrorV19::InvalidProfile)? as usize;
    Ok((xi, opening))
}

impl<Comm> From<&FreshPesatClaim<EF, Comm>> for DirectAirVaccFreshClaimDataV19 {
    fn from(claim: &FreshPesatClaim<EF, Comm>) -> Self {
        Self {
            alpha: claim.alpha.clone(),
            beta: claim.beta.clone(),
            mu: claim.mu,
            eta: claim.eta,
        }
    }
}

fn generate_direct_air_vacc_algebra_traces_v19<
    FreshVerification: DirectAirFreshAuthenticationV19,
>(
    module: &DirectAirVaccVerifierModuleV19,
    proof_idx: usize,
    record: &DirectAirVaccVerifierRecordV19<'_, FreshVerification>,
    schedule: &DirectAirVaccTranscriptScheduleV19,
) -> Result<DirectAirVaccAlgebraTracesV19, DirectAirVaccVerifierErrorV19> {
    let fresh_claims = vec![DirectAirVaccFreshClaimDataV19 {
        alpha: record
            .producer
            .fresh_alpha
            .iter()
            .map(ext_from_v19_array)
            .collect::<Result<Vec<_>, _>>()?,
        beta: record
            .producer
            .fresh_beta
            .iter()
            .map(ext_from_v19_array)
            .collect::<Result<Vec<_>, _>>()?,
        mu: ext_from_v19_array(&record.producer.fresh_mu)?,
        eta: ext_from_v19_array(&record.producer.fresh_eta)?,
    }];
    generate_direct_air_vacc_algebra_traces_from_parts_v19(
        module,
        proof_idx,
        &fresh_claims,
        record.verification,
        record.transcript,
        record.prior,
        schedule,
    )
}

#[allow(clippy::too_many_arguments)]
fn generate_direct_air_vacc_algebra_traces_from_parts_v19<FreshVerification>(
    module: &DirectAirVaccVerifierModuleV19,
    proof_idx: usize,
    fresh_claims: &[DirectAirVaccFreshClaimDataV19],
    v: &WarpVaccStepVerification<
        EF,
        Digest,
        FreshVerification,
        MerkleBatchOpeningVerification<EF, Digest>,
    >,
    transcript: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    prior: Option<&AccumulatorInstance<EF, Digest>>,
    schedule: &DirectAirVaccTranscriptScheduleV19,
) -> Result<DirectAirVaccAlgebraTracesV19, DirectAirVaccVerifierErrorV19> {
    macro_rules! algebra {
        ($value:expr, $stage:literal) => {
            $value.ok_or(DirectAirVaccVerifierErrorV19::Algebra($stage))?
        };
    }
    let profile = &module.profile;
    // `include_prior` is a verifier-key capability for the setup-fixed
    // reduced-SWIRL module.  Bootstrap deliberately exercises that same AIR
    // inventory without a prior accumulator, so trace generation must use
    // the authenticated record's actual mode.  Treating the capability bit
    // as runtime presence creates input_arity + 1 slots on a full bootstrap
    // batch and incorrectly includes a prior in the shift reduction.
    let has_prior = prior.is_some();
    let zero_alpha = vec![EF::ZERO; profile.log_codeword_len];
    let zero_beta = vec![EF::ZERO; profile.beta_len];
    let mut alphas = fresh_claims
        .iter()
        .map(|claim| claim.alpha.clone())
        .collect::<Vec<_>>();
    let mut betas = fresh_claims
        .iter()
        .map(|claim| claim.beta.clone())
        .collect::<Vec<_>>();
    let mut mus = fresh_claims
        .iter()
        .map(|claim| claim.mu)
        .collect::<Vec<_>>();
    let mut etas = fresh_claims
        .iter()
        .map(|claim| claim.eta)
        .collect::<Vec<_>>();
    let mut kinds = vec![(true, false, false); fresh_claims.len()];
    let alpha_len = profile.log_codeword_len;
    let tau_len = profile.log_constraints;
    let beta_tail_len = profile.beta_len - tau_len;
    let mut transcript_bindings = (0..fresh_claims.len())
        .map(|source| {
            let mut bindings = Vec::with_capacity(alpha_len + profile.beta_len + 2);
            bindings.extend(
                schedule.fresh_alpha_tidx[source * alpha_len..(source + 1) * alpha_len]
                    .iter()
                    .map(|&tidx| Some((proof_idx, tidx, false))),
            );
            bindings.extend(
                schedule.fresh_tau_tidx[source * tau_len..(source + 1) * tau_len]
                    .iter()
                    .map(|&tidx| Some((proof_idx, tidx, true))),
            );
            bindings.extend(
                schedule.fresh_beta_tail_tidx[source * beta_tail_len..(source + 1) * beta_tail_len]
                    .iter()
                    .map(|&tidx| Some((proof_idx, tidx, false))),
            );
            bindings.push(Some((proof_idx, schedule.fresh_mu_tidx[source], false)));
            bindings.push(None);
            bindings
        })
        .collect::<Vec<_>>();
    if let (Some(prior), Some(prior_schedule)) = (prior, schedule.prior.as_ref()) {
        if prior.alpha.len() != profile.log_codeword_len || prior.beta.len() != profile.beta_len {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "prior accumulator dimensions",
            ));
        }
        alphas.push(prior.alpha.clone());
        betas.push(prior.beta.clone());
        mus.push(prior.mu);
        etas.push(prior.eta);
        kinds.push((false, true, false));
        let mut bindings = Vec::with_capacity(profile.log_codeword_len + profile.beta_len + 2);
        bindings.extend(
            prior_schedule
                .alpha_tidx
                .iter()
                .map(|&tidx| Some((proof_idx, tidx, false))),
        );
        bindings.extend(
            prior_schedule
                .beta_tidx
                .iter()
                .map(|&tidx| Some((proof_idx, tidx, false))),
        );
        bindings.push(Some((proof_idx, prior_schedule.mu_tidx, false)));
        bindings.push(Some((proof_idx, prior_schedule.eta_tidx, false)));
        transcript_bindings.push(bindings);
    }
    // Padding is independent of prior presence.  A continuation's final
    // fresh batch may be partial, so it contains fresh claims, one prior, and
    // canonical zero slots up to the setup-fixed input arity.
    while alphas.len() < module.input_arity() {
        alphas.push(zero_alpha.clone());
        betas.push(zero_beta.clone());
        mus.push(EF::ZERO);
        etas.push(EF::ZERO);
        kinds.push((false, false, true));
        transcript_bindings.push(vec![None; profile.log_codeword_len + profile.beta_len + 2]);
    }
    let input_arity = module.input_arity();
    if alphas.len() != input_arity {
        return Err(DirectAirVaccVerifierErrorV19::RecordShape(
            "exact input claim count",
        ));
    }
    let claim_inputs = (0..input_arity)
        .map(|source| NativeClaimTraceInput {
            proof_idx,
            alpha: &alphas[source],
            beta: &betas[source],
            mu: mus[source],
            eta: etas[source],
            is_fresh: kinds[source].0,
            is_prior: kinds[source].1,
            is_dummy: kinds[source].2,
            transcript_bindings: &transcript_bindings[source],
        })
        .collect::<Vec<_>>();
    let claim_values = algebra!(
        generate_native_claim_value_trace(&claim_inputs, profile.log_constraints, None),
        "claim values"
    );
    let claim_layout = algebra!(
        generate_native_claim_layout_trace(
            profile.log_codeword_len,
            profile.beta_len,
            profile.log_constraints,
            input_arity,
        ),
        "claim layout"
    );
    let claim_rows = profile.log_codeword_len + profile.beta_len + 2;
    let shift_count = v.shifts.len();
    let mut slot_lookups = vec![(claim_rows + 2 * shift_count) as u32; fresh_claims.len()];
    let extra_fresh_slot_lookups = module.exact_finite.as_ref().map_or(0, |exact| {
        exact
            .multiplicities
            .extra_fresh_slot_lookup_count_per_source
    });
    for lookup_count in &mut slot_lookups {
        *lookup_count = lookup_count
            .checked_add(
                u32::try_from(extra_fresh_slot_lookups)
                    .map_err(|_| DirectAirVaccVerifierErrorV19::InvalidProfile)?,
            )
            .ok_or(DirectAirVaccVerifierErrorV19::InvalidProfile)?;
    }
    if has_prior {
        slot_lookups.push((claim_rows + 2 * shift_count) as u32);
    }
    slot_lookups.resize(input_arity, (claim_rows + shift_count) as u32);
    let slot_layout = algebra!(
        generate_native_exact_finite_input_slot_layout_trace(
            input_arity,
            fresh_claims.len(),
            usize::from(has_prior),
            &slot_lookups,
        ),
        "slot layout"
    );

    let claim_count = module.profile.batching_arity;
    let selector_rounds = input_arity.ilog2() as usize;
    let selector_indices = (0..input_arity)
        .map(|index| boolean_v19_point(index, selector_rounds))
        .collect::<Vec<_>>();
    let claim_indices = (0..claim_count)
        .map(|index| boolean_v19_point(index, claim_count.ilog2() as usize))
        .collect::<Vec<_>>();
    let mut opening_points = direct_v19_opening_points(v);
    let mut opening_targets = direct_v19_opening_targets(v);
    opening_points.resize(claim_count, opening_points[0].clone());
    opening_targets.resize(claim_count, opening_targets[0]);

    let mut vector_values = Vec::<Vec<EF>>::new();
    let mut vector_sources = Vec::<NativeVectorSource>::new();
    let mut vector_ids = Vec::<u32>::new();
    let mut vector_counts = Vec::<Vec<u32>>::new();
    let shared_twin_lookup_count = shared_twin_lookup_count_v19(input_arity)?;
    push_v19_vector(
        &mut vector_values,
        &mut vector_sources,
        &mut vector_ids,
        &mut vector_counts,
        module.algebra.selector_tau_vector,
        v.twin.selector_tau.clone(),
        NativeVectorSource::Transcript {
            proof_idx: proof_idx as u32,
            tidx: schedule.selector_tidx.iter().map(|&x| x as u32).collect(),
        },
        shared_twin_lookup_count,
    );
    push_v19_vector(
        &mut vector_values,
        &mut vector_sources,
        &mut vector_ids,
        &mut vector_counts,
        module.algebra.gamma_vector,
        v.twin.gamma.clone(),
        NativeVectorSource::Sumcheck { kind: 0 },
        shared_twin_lookup_count,
    );
    push_v19_vector(
        &mut vector_values,
        &mut vector_sources,
        &mut vector_ids,
        &mut vector_counts,
        module.algebra.xi_vector,
        v.batching.xi.clone(),
        NativeVectorSource::Transcript {
            proof_idx: proof_idx as u32,
            tidx: schedule.xi_tidx.iter().map(|&x| x as u32).collect(),
        },
        claim_count as u32,
    );
    push_v19_vector(
        &mut vector_values,
        &mut vector_sources,
        &mut vector_ids,
        &mut vector_counts,
        module.algebra.alpha_vector,
        v.batching.alpha.clone(),
        NativeVectorSource::Sumcheck { kind: 1 },
        claim_count as u32,
    );
    for (id, values) in module
        .algebra
        .input_index_vectors
        .iter()
        .zip(&selector_indices)
    {
        push_v19_vector(
            &mut vector_values,
            &mut vector_sources,
            &mut vector_ids,
            &mut vector_counts,
            *id,
            values.clone(),
            NativeVectorSource::Boolean,
            2,
        );
    }
    for (id, values) in module
        .algebra
        .claim_index_vectors
        .iter()
        .zip(&claim_indices)
    {
        push_v19_vector(
            &mut vector_values,
            &mut vector_sources,
            &mut vector_ids,
            &mut vector_counts,
            *id,
            values.clone(),
            NativeVectorSource::Boolean,
            1,
        );
    }
    for (claim, (id, values)) in module
        .algebra
        .opening_point_vectors
        .iter()
        .zip(&opening_points)
        .enumerate()
    {
        push_v19_vector(
            &mut vector_values,
            &mut vector_sources,
            &mut vector_ids,
            &mut vector_counts,
            *id,
            values.clone(),
            NativeVectorSource::OpeningPoint {
                claim: claim as u32,
            },
            1,
        );
    }
    let vector_inputs = (0..vector_values.len())
        .map(|index| NativeVectorTraceInput {
            proof_idx: proof_idx as u32,
            vector: vector_ids[index],
            values: &vector_values[index],
            source: vector_sources[index].clone(),
            lookup_counts: &vector_counts[index],
        })
        .collect::<Vec<_>>();
    let vector = algebra!(
        generate_native_vector_coordinate_trace(&vector_inputs, None),
        "vector coordinates"
    );

    let selector_result_lookups =
        (profile.log_codeword_len + profile.beta_len + shift_count) as u32;
    let mut selector_pairs = Vec::with_capacity(5);
    for (index, point) in selector_indices.iter().enumerate() {
        selector_pairs.push(NativeEqTraceInput {
            left_vector: module.algebra.selector_tau_vector,
            right_vector: module.algebra.input_index_vectors[index],
            left: &v.twin.selector_tau,
            right: point,
            lookup_count: 1,
        });
    }
    for (index, point) in selector_indices.iter().enumerate() {
        selector_pairs.push(NativeEqTraceInput {
            left_vector: module.algebra.gamma_vector,
            right_vector: module.algebra.input_index_vectors[index],
            left: &v.twin.gamma,
            right: point,
            lookup_count: selector_result_lookups,
        });
    }
    selector_pairs.push(NativeEqTraceInput {
        left_vector: module.algebra.selector_tau_vector,
        right_vector: module.algebra.gamma_vector,
        left: &v.twin.selector_tau,
        right: &v.twin.gamma,
        lookup_count: 1,
    });
    let selector_eq = algebra!(
        generate_native_eq_trace(proof_idx, &selector_pairs, 0, None),
        "selector equality"
    );
    let xi_pairs = claim_indices
        .iter()
        .enumerate()
        .map(|(claim, point)| NativeEqTraceInput {
            left_vector: module.algebra.xi_vector,
            right_vector: module.algebra.claim_index_vectors[claim],
            left: &v.batching.xi,
            right: point,
            lookup_count: 2,
        })
        .collect::<Vec<_>>();
    let (xi_eq_group_offset, opening_eq_group_offset) = direct_vacc_eq_group_offsets_v19(module)?;
    let xi_eq = algebra!(
        generate_native_eq_trace(proof_idx, &xi_pairs, xi_eq_group_offset, None,),
        "batch selector equality"
    );
    let opening_pairs = opening_points
        .iter()
        .enumerate()
        .map(|(claim, point)| NativeEqTraceInput {
            left_vector: module.algebra.opening_point_vectors[claim],
            right_vector: module.algebra.alpha_vector,
            left: point,
            right: &v.batching.alpha,
            lookup_count: 1,
        })
        .collect::<Vec<_>>();
    let opening_eq = algebra!(
        generate_native_eq_trace(proof_idx, &opening_pairs, opening_eq_group_offset, None,),
        "opening equality"
    );
    let selector_weights = selector_indices
        .iter()
        .map(|point| direct_v19_eq(&v.twin.selector_tau, point))
        .collect::<Vec<_>>();
    let gamma_weights = selector_indices
        .iter()
        .map(|point| direct_v19_eq(&v.twin.gamma, point))
        .collect::<Vec<_>>();
    let xi_weights = claim_indices
        .iter()
        .map(|point| direct_v19_eq(&v.batching.xi, point))
        .collect::<Vec<_>>();
    let point_eq_alpha = opening_points
        .iter()
        .map(|point| direct_v19_eq(point, &v.batching.alpha))
        .collect::<Vec<_>>();

    let omega = generate_native_twin_omega_trace(
        proof_idx,
        schedule.omega_tidx,
        v.twin.omega,
        shared_twin_lookup_count as usize,
    );
    let twin_sigma = algebra!(
        generate_native_twin_sigma_trace(
            proof_idx,
            &selector_weights,
            &mus.iter()
                .copied()
                .zip(etas.iter().copied())
                .collect::<Vec<_>>(),
            v.twin.omega,
            None,
        ),
        "twin sigma"
    );
    let sumcheck = algebra!(
        generate_native_coefficient_sumcheck_trace(
            proof_idx,
            proof_idx,
            &[&v.twin.sumcheck, &v.batching.sumcheck],
            None,
        ),
        "sumchecks"
    );
    let twin_fold = algebra!(
        generate_native_twin_fold_trace(
            proof_idx,
            &[
                (CLAIM_SECTION_ALPHA, alphas.as_slice()),
                (CLAIM_SECTION_BETA, betas.as_slice()),
            ],
            &gamma_weights,
            None,
        ),
        "twin fold"
    );
    let twin_last = v.twin.sumcheck.rounds.len().checked_sub(1).ok_or(
        DirectAirVaccVerifierErrorV19::Algebra("empty twin sumcheck"),
    )?;
    let twin_pre_claim = v
        .twin
        .sumcheck
        .pre_claim_at_round(twin_last)
        .ok_or(DirectAirVaccVerifierErrorV19::Algebra("twin pre-claim"))?;
    let twin_challenge = *v
        .twin
        .sumcheck
        .point
        .get(twin_last)
        .ok_or(DirectAirVaccVerifierErrorV19::Algebra("twin challenge"))?;
    let twin_final = generate_native_twin_final_trace(
        twin_pre_claim,
        v.twin.sumcheck.final_claim,
        twin_challenge,
        direct_v19_eq(&v.twin.selector_tau, &v.twin.gamma),
        v.twin.omega,
        v.twin.nu_0,
        v.twin.eta,
        proof_idx,
        schedule.nu_tidx,
        schedule.eta_tidx,
    );
    let initial_point = algebra!(
        generate_native_initial_opening_point_trace(proof_idx, &v.twin.zeta_0),
        "initial point"
    );
    let initial_target = generate_native_initial_opening_target_trace(proof_idx, v.twin.nu_0);
    let authenticated_claim_count = 1 + v.ood.len() + v.shifts.len();
    let opening_padding = algebra!(
        generate_native_opening_padding_trace(
            proof_idx,
            authenticated_claim_count,
            claim_count,
            &opening_points[0],
            opening_targets[0],
            None,
        ),
        "opening padding"
    );
    let ood_claims = algebra!(
        generate_native_ood_claim_trace(
            proof_idx,
            &v.ood
                .iter()
                .map(|ood| ood.point.clone())
                .collect::<Vec<_>>(),
            &schedule.ood_point_tidx,
            &v.ood.iter().map(|ood| ood.answer).collect::<Vec<_>>(),
            &schedule.ood_answer_tidx,
        ),
        "OOD claims"
    );
    let shift_samples = schedule
        .shifts
        .iter()
        .map(|sample| {
            transcript
                .values()
                .get(sample.operation_range.end.checked_sub(1)?)
                .copied()
        })
        .collect::<Option<Vec<_>>>()
        .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
            "accepted shift samples",
        ))?;
    let shift_tidx = schedule
        .shifts
        .iter()
        .map(|sample| sample.operation_range.end - 1)
        .collect::<Vec<_>>();
    let shift_schedule = algebra!(
        generate_direct_air_vacc_shift_schedule_trace_v19(
            proof_idx,
            &shift_samples,
            &shift_tidx,
            &v.shifts.iter().map(|shift| shift.index).collect::<Vec<_>>(),
            (fresh_claims.len() + usize::from(has_prior)) as u32,
            profile.log_codeword_len,
            None,
        ),
        "shift schedule"
    );
    let shift_answers = v
        .shifts
        .iter()
        .map(|shift| {
            let mut answers = shift.fresh_answers.clone();
            if let Some(prior) = shift.prior_answer {
                answers.push(prior);
            }
            answers.resize(input_arity, EF::ZERO);
            answers
        })
        .collect::<Vec<_>>();
    let shift_merge = algebra!(
        generate_native_shift_merge_trace(
            proof_idx,
            &gamma_weights,
            &shift_answers,
            fresh_claims.len(),
            has_prior,
            None,
        ),
        "shift merge"
    );
    let batching_sigma = algebra!(
        generate_native_batching_sigma_trace(proof_idx, &xi_weights, &opening_targets, None,),
        "batching sigma"
    );
    let batching_last = v.batching.sumcheck.rounds.len().checked_sub(1).ok_or(
        DirectAirVaccVerifierErrorV19::Algebra("empty batching sumcheck"),
    )?;
    let batching_pre_claim = v
        .batching
        .sumcheck
        .pre_claim_at_round(batching_last)
        .ok_or(DirectAirVaccVerifierErrorV19::Algebra("batching pre-claim"))?;
    let batching_challenge = *v
        .batching
        .sumcheck
        .point
        .get(batching_last)
        .ok_or(DirectAirVaccVerifierErrorV19::Algebra("batching challenge"))?;
    let batching_final = algebra!(
        generate_native_batching_final_trace(
            &xi_weights,
            &point_eq_alpha,
            batching_pre_claim,
            v.batching.sumcheck.final_claim,
            batching_challenge,
            v.batching.mu_final,
            proof_idx,
            schedule.mu_tidx,
            None,
        ),
        "batching final"
    );
    let batching_alpha = algebra!(
        generate_native_batching_alpha_trace(proof_idx, &v.batching.alpha),
        "batching alpha"
    );
    Ok(DirectAirVaccAlgebraTracesV19 {
        claim_values,
        claim_layout,
        slot_layout,
        vector,
        selector_eq,
        xi_eq,
        opening_eq,
        omega,
        twin_sigma,
        sumcheck,
        twin_fold,
        twin_final,
        initial_point,
        initial_target,
        opening_padding,
        ood_claims,
        shift_schedule,
        shift_merge,
        batching_sigma,
        batching_final,
        batching_alpha,
    })
}

#[allow(clippy::too_many_arguments)]
fn push_v19_vector(
    values: &mut Vec<Vec<EF>>,
    sources: &mut Vec<NativeVectorSource>,
    ids: &mut Vec<u32>,
    counts: &mut Vec<Vec<u32>>,
    id: u32,
    vector: Vec<EF>,
    source: NativeVectorSource,
    lookup_count: u32,
) {
    counts.push(vec![lookup_count; vector.len()]);
    values.push(vector);
    sources.push(source);
    ids.push(id);
}

fn boolean_v19_point(index: usize, dimensions: usize) -> Vec<EF> {
    (0..dimensions)
        .map(|coordinate| EF::from_bool(((index >> (dimensions - 1 - coordinate)) & 1) == 1))
        .collect()
}

fn direct_v19_eq(left: &[EF], right: &[EF]) -> EF {
    openvm_stark_backend::warp_pesat::eval_eq_points(left, right)
}

fn direct_v19_opening_points<FreshVerification>(
    v: &WarpVaccStepVerification<
        EF,
        Digest,
        FreshVerification,
        MerkleBatchOpeningVerification<EF, Digest>,
    >,
) -> Vec<Vec<EF>> {
    core::iter::once(v.twin.zeta_0.clone())
        .chain(v.ood.iter().map(|ood| ood.point.clone()))
        .chain(v.shifts.iter().map(|shift| shift.boolean_point.clone()))
        .collect()
}

fn direct_v19_opening_targets<FreshVerification>(
    v: &WarpVaccStepVerification<
        EF,
        Digest,
        FreshVerification,
        MerkleBatchOpeningVerification<EF, Digest>,
    >,
) -> Vec<EF> {
    core::iter::once(v.twin.nu_0)
        .chain(v.ood.iter().map(|ood| ood.answer))
        .chain(v.shifts.iter().map(|shift| shift.merged_answer))
        .collect()
}

fn ext_from_v19_array(value: &[F; D_EF]) -> Result<EF, DirectAirVaccVerifierErrorV19> {
    EF::from_basis_coefficients_slice(value)
        .ok_or(DirectAirVaccVerifierErrorV19::RecordShape("EF4 value"))
}

#[derive(Clone, Debug)]
struct DirectAirVaccCursorEventV19 {
    role: usize,
    ordinal: usize,
    tidx: usize,
    value: [F; D_EF],
    is_ext: bool,
    is_sample: bool,
}

fn push_v19_cursor_ext(
    events: &mut Vec<DirectAirVaccCursorEventV19>,
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    role: usize,
    ordinal: usize,
    tidx: usize,
    is_sample: bool,
) -> Result<(), DirectAirVaccVerifierErrorV19> {
    let values =
        log.values()
            .get(tidx..tidx + D_EF)
            .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
                "cursor extension",
            ))?;
    let samples =
        log.samples()
            .get(tidx..tidx + D_EF)
            .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
                "cursor extension",
            ))?;
    if samples.iter().any(|&sample| sample != is_sample) {
        return Err(DirectAirVaccVerifierErrorV19::Transcript(
            "cursor extension kind",
        ));
    }
    events.push(DirectAirVaccCursorEventV19 {
        role,
        ordinal,
        tidx,
        value: values
            .try_into()
            .map_err(|_| DirectAirVaccVerifierErrorV19::Transcript("cursor extension"))?,
        is_ext: true,
        is_sample,
    });
    Ok(())
}

fn generate_direct_air_vacc_cursor_trace_v19<FreshVerification: DirectAirFreshAuthenticationV19>(
    module: &DirectAirVaccVerifierModuleV19,
    record: &DirectAirVaccVerifierRecordV19<'_, FreshVerification>,
    schedule: &DirectAirVaccTranscriptScheduleV19,
) -> Result<RowMajorMatrix<F>, DirectAirVaccVerifierErrorV19> {
    generate_direct_air_vacc_cursor_trace_from_parts_v19(
        module,
        record.transcript,
        record.verification,
        schedule,
    )
}

fn generate_direct_air_vacc_cursor_trace_from_parts_v19<FreshVerification>(
    module: &DirectAirVaccVerifierModuleV19,
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    verification: &WarpVaccStepVerification<
        EF,
        Digest,
        FreshVerification,
        MerkleBatchOpeningVerification<EF, Digest>,
    >,
    schedule: &DirectAirVaccTranscriptScheduleV19,
) -> Result<RowMajorMatrix<F>, DirectAirVaccVerifierErrorV19> {
    let mut events = Vec::<DirectAirVaccCursorEventV19>::new();
    if module.exact_finite.is_some() || module.reduced_swirl.is_some() {
        for (ordinal, tidx) in core::iter::once(schedule.fresh_root_tidx)
            .chain(schedule.fresh_commitment_extra_tidx.iter().copied())
            .enumerate()
        {
            push_v19_cursor_ext(&mut events, log, VACC_ROLE_FRESH_ROOT, ordinal, tidx, false)?;
        }
    } else {
        for limb in 0..DIGEST_SIZE {
            push_v19_cursor_ext(
                &mut events,
                log,
                VACC_ROLE_FRESH_ROOT,
                limb,
                schedule.fresh_root_tidx + limb * D_EF,
                false,
            )?;
        }
        for (offset, &tidx) in schedule.fresh_commitment_extra_tidx.iter().enumerate() {
            push_v19_cursor_ext(
                &mut events,
                log,
                VACC_ROLE_FRESH_ROOT,
                DIGEST_SIZE + offset,
                tidx,
                false,
            )?;
        }
    }
    for (ordinal, &tidx) in schedule.fresh_alpha_tidx.iter().enumerate() {
        push_v19_cursor_ext(
            &mut events,
            log,
            VACC_ROLE_FRESH_ALPHA,
            ordinal,
            tidx,
            false,
        )?;
    }
    for (ordinal, &tidx) in schedule.fresh_mu_tidx.iter().enumerate() {
        push_v19_cursor_ext(&mut events, log, VACC_ROLE_FRESH_MU, ordinal, tidx, false)?;
    }
    for (ordinal, &tidx) in schedule.fresh_beta_tail_tidx.iter().enumerate() {
        push_v19_cursor_ext(
            &mut events,
            log,
            VACC_ROLE_FRESH_BETA_TAIL,
            ordinal,
            tidx,
            false,
        )?;
    }
    if let Some(prior) = schedule.prior.as_ref() {
        for limb in 0..DIGEST_SIZE {
            push_v19_cursor_ext(
                &mut events,
                log,
                VACC_ROLE_PRIOR_ROOT,
                limb,
                prior.root_tidx + limb * D_EF,
                false,
            )?;
        }
        for (ordinal, &tidx) in prior.alpha_tidx.iter().enumerate() {
            push_v19_cursor_ext(
                &mut events,
                log,
                VACC_ROLE_PRIOR_ALPHA,
                ordinal,
                tidx,
                false,
            )?;
        }
        push_v19_cursor_ext(
            &mut events,
            log,
            VACC_ROLE_PRIOR_MU,
            0,
            prior.mu_tidx,
            false,
        )?;
        for (ordinal, &tidx) in prior.beta_tidx.iter().enumerate() {
            push_v19_cursor_ext(&mut events, log, VACC_ROLE_PRIOR_BETA, ordinal, tidx, false)?;
        }
        push_v19_cursor_ext(
            &mut events,
            log,
            VACC_ROLE_PRIOR_ETA,
            0,
            prior.eta_tidx,
            false,
        )?;
    }
    for (ordinal, &tidx) in schedule.fresh_tau_tidx.iter().enumerate() {
        push_v19_cursor_ext(&mut events, log, VACC_ROLE_FRESH_TAU, ordinal, tidx, true)?;
    }
    push_v19_cursor_ext(
        &mut events,
        log,
        VACC_ROLE_OMEGA,
        0,
        schedule.omega_tidx,
        true,
    )?;
    for (ordinal, &tidx) in schedule.selector_tidx.iter().enumerate() {
        push_v19_cursor_ext(&mut events, log, VACC_ROLE_SELECTOR, ordinal, tidx, true)?;
    }
    for round in &verification.twin.sumcheck.rounds {
        let event_count = round.coefficients.len() + 1;
        for coefficient in 0..round.coefficients.len() {
            push_v19_cursor_ext(
                &mut events,
                log,
                VACC_ROLE_TWIN_SUMCHECK,
                round.round as usize * event_count + coefficient,
                round.transcript_span.operation_range.start + coefficient * D_EF,
                false,
            )?;
        }
        push_v19_cursor_ext(
            &mut events,
            log,
            VACC_ROLE_TWIN_SUMCHECK,
            round.round as usize * event_count + round.coefficients.len(),
            round.transcript_span.operation_range.end - D_EF,
            true,
        )?;
    }
    for limb in 0..DIGEST_SIZE {
        push_v19_cursor_ext(
            &mut events,
            log,
            VACC_ROLE_OUTPUT_ROOT,
            limb,
            schedule.output_root_tidx + limb * D_EF,
            false,
        )?;
    }
    for (role, tidx) in [
        (VACC_ROLE_NU, schedule.nu_tidx),
        (VACC_ROLE_ETA, schedule.eta_tidx),
    ] {
        push_v19_cursor_ext(&mut events, log, role, 0, tidx, false)?;
    }
    for (ood, &start) in schedule.ood_point_tidx.iter().enumerate() {
        for coordinate in 0..module.profile.log_codeword_len {
            push_v19_cursor_ext(
                &mut events,
                log,
                VACC_ROLE_OOD_POINT,
                ood * module.profile.log_codeword_len + coordinate,
                start + coordinate * D_EF,
                true,
            )?;
        }
    }
    for (ordinal, &tidx) in schedule.ood_answer_tidx.iter().enumerate() {
        push_v19_cursor_ext(&mut events, log, VACC_ROLE_OOD_ANSWER, ordinal, tidx, false)?;
    }
    for (shift, sample) in schedule.shifts.iter().enumerate() {
        let tidx = sample.operation_range.start;
        let value = *log
            .values()
            .get(tidx)
            .ok_or(DirectAirVaccVerifierErrorV19::Transcript("shift cursor"))?;
        if sample.operation_range.end != tidx + 1 || log.samples().get(tidx).copied() != Some(true)
        {
            return Err(DirectAirVaccVerifierErrorV19::Transcript(
                "shift cursor kind",
            ));
        }
        events.push(DirectAirVaccCursorEventV19 {
            role: VACC_ROLE_SHIFT,
            ordinal: shift,
            tidx,
            value: [value, F::ZERO, F::ZERO, F::ZERO],
            is_ext: false,
            is_sample: true,
        });
    }
    for (ordinal, &tidx) in schedule.xi_tidx.iter().enumerate() {
        push_v19_cursor_ext(&mut events, log, VACC_ROLE_XI, ordinal, tidx, true)?;
    }
    for round in &verification.batching.sumcheck.rounds {
        let event_count = round.coefficients.len() + 1;
        for coefficient in 0..round.coefficients.len() {
            push_v19_cursor_ext(
                &mut events,
                log,
                VACC_ROLE_BATCHING_SUMCHECK,
                round.round as usize * event_count + coefficient,
                round.transcript_span.operation_range.start + coefficient * D_EF,
                false,
            )?;
        }
        push_v19_cursor_ext(
            &mut events,
            log,
            VACC_ROLE_BATCHING_SUMCHECK,
            round.round as usize * event_count + round.coefficients.len(),
            round.transcript_span.operation_range.end - D_EF,
            true,
        )?;
    }
    push_v19_cursor_ext(&mut events, log, VACC_ROLE_MU, 0, schedule.mu_tidx, false)?;

    let first = events
        .first()
        .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
            "empty VACC cursor",
        ))?;
    let last = events
        .last()
        .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
            "empty VACC cursor",
        ))?;
    if first.tidx != schedule.fresh_root_tidx
        || last.tidx + D_EF != schedule.end_tag_tidx
        || events
            .windows(2)
            .any(|pair| pair[1].tidx != pair[0].tidx + if pair[0].is_ext { D_EF } else { 1 })
    {
        return Err(DirectAirVaccVerifierErrorV19::Transcript(
            "non-contiguous VACC cursor",
        ));
    }
    let lengths = if module.reduced_swirl.is_some() {
        let mut lengths = [0usize; VACC_ROLE_COUNT];
        for event in &events {
            lengths[event.role] = lengths[event.role].max(event.ordinal + 1);
        }
        lengths
    } else if module.exact_finite.is_some() {
        vacc_cursor_role_lengths_exact_v19(
            &module.profile,
            module.input_arity(),
            module.fresh_count(),
            module.include_prior,
        )
    } else {
        vacc_cursor_role_lengths_v19(
            &module.profile,
            module.include_prior,
            module.fresh_commitment_mode,
        )
    };
    let height = events.len().next_power_of_two();
    let width = DirectAirVaccTranscriptCursorColsV19::<F>::width();
    let mut trace = vec![F::ZERO; height * width];
    let twin_event_count = twin_degree(
        module.profile.log_codeword_len,
        module.profile.log_constraints,
        module.profile.max_degree,
    ) + 2;
    for (row, event) in events.iter().enumerate() {
        let cols: &mut DirectAirVaccTranscriptCursorColsV19<F> =
            trace[row * width..(row + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.proof_idx = F::from_usize(schedule.proof_idx);
        cols.tidx = F::from_usize(event.tidx);
        cols.role = F::from_usize(event.role);
        cols.ordinal = F::from_usize(event.ordinal);
        cols.role_flags[event.role] = F::ONE;
        cols.is_ext = F::from_bool(event.is_ext);
        cols.is_sample = F::from_bool(event.is_sample);
        cols.is_first_ordinal = F::from_bool(event.ordinal == 0);
        cols.is_last_ordinal = F::from_bool(
            events
                .get(row + 1)
                .is_none_or(|next| next.role != event.role),
        );
        cols.ordinal_inverse = if event.ordinal == 0 {
            F::ZERO
        } else {
            F::from_usize(event.ordinal).inverse()
        };
        let distance = lengths[event.role] - 1 - event.ordinal;
        cols.last_ordinal_inverse = if distance == 0 {
            F::ZERO
        } else {
            F::from_usize(distance).inverse()
        };
        let event_count = if event.role == VACC_ROLE_TWIN_SUMCHECK {
            twin_event_count
        } else if event.role == VACC_ROLE_BATCHING_SUMCHECK {
            4
        } else {
            0
        };
        cols.is_sumcheck = F::from_bool(event_count != 0);
        if event_count != 0 {
            cols.sumcheck_round = F::from_usize(event.ordinal / event_count);
            cols.sumcheck_step = F::from_usize(event.ordinal % event_count);
            let step = event.ordinal % event_count;
            let step_distance = event_count - 1 - step;
            cols.is_last_sumcheck_step = F::from_bool(step_distance == 0);
            cols.sumcheck_last_step_inverse = if step_distance == 0 {
                F::ZERO
            } else {
                F::from_usize(step_distance).inverse()
            };
            cols.sumcheck_step_continues = F::from_bool(step_distance != 0);
        }
        let starts_proof = event.role == VACC_ROLE_FRESH_ROOT && event.ordinal == 0;
        cols.starts_proof = F::from_bool(starts_proof);
        cols.continues_proof = F::from_bool(!starts_proof);
        cols.continues_sumcheck_phase = F::from_bool(event_count != 0 && event.ordinal != 0);
        cols.value = event.value;
    }
    Ok(RowMajorMatrix::new(trace, width))
}

/// Complete row-major witness in the same order as
/// [`DirectAirVaccVerifierModuleV19::airs`].
pub struct DirectAirVaccVerifierTraceV19 {
    /// One matrix per AIR, in exactly the order returned by
    /// [`DirectAirVaccVerifierModuleV19::airs`].
    pub traces: Vec<RowMajorMatrix<F>>,
    /// Complete Poseidon permutation inputs represented by the module's
    /// Poseidon AIR matrix. Exposed so a caller can merge this verifier into a
    /// shared MultiSTARK Poseidon table without regenerating authentication or
    /// transcript witnesses.
    pub poseidon_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    /// Complete Poseidon compression inputs represented by the module's
    /// Poseidon AIR matrix.
    pub poseidon_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub output_instance: AccumulatorInstance<EF, Digest>,
    pub output_instance_digest: Digest,
}

/// Complete production witness for one setup-fixed exact-finite WARP Verify
/// call. The final checkpoint is verifier-derived and is the only transcript
/// continuation token a later terminal component may consume.
pub struct DirectAirExactFiniteVaccVerifierTraceV19 {
    /// One matrix per AIR in [`DirectAirVaccVerifierModuleV19::airs`] order.
    pub traces: Vec<RowMajorMatrix<F>>,
    pub poseidon_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub poseidon_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub output_instance: AccumulatorInstance<EF, Digest>,
    pub output_instance_digest: Digest,
    pub final_transcript_checkpoint: TranscriptCheckpointRecordV19,
}

impl DirectAirExactFiniteVaccVerifierTraceV19 {
    #[must_use]
    pub fn air_matrices(&self) -> &[RowMajorMatrix<F>] {
        &self.traces
    }

    pub fn into_air_matrices(self) -> Vec<RowMajorMatrix<F>> {
        self.traces
    }
}

/// Exact-finite witness packet for a composition that owns the physical
/// Poseidon table. It exposes every hashing request generated by transcript,
/// stacked-fresh Merkle authentication, prior authentication, and output
/// reconstruction, while preserving their distinct bus namespace.
pub struct DirectAirExactFiniteVaccSharedPoseidonTraceV19 {
    pub traces: Vec<RowMajorMatrix<F>>,
    pub poseidon_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub poseidon_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub output_instance: AccumulatorInstance<EF, Digest>,
    pub output_instance_digest: Digest,
    pub final_transcript_checkpoint: TranscriptCheckpointRecordV19,
}

impl DirectAirExactFiniteVaccSharedPoseidonTraceV19 {
    fn with_local_poseidon_table(
        self,
        module: &DirectAirVaccVerifierModuleV19,
    ) -> Result<DirectAirExactFiniteVaccVerifierTraceV19, DirectAirVaccVerifierErrorV19> {
        let poseidon = module
            .transcript
            .build_poseidon2_trace(
                self.poseidon_permutation_inputs.clone(),
                self.poseidon_compression_inputs.clone(),
                None,
            )
            .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
                "exact-finite Poseidon trace generation",
            ))?;
        let mut traces = self.traces;
        if traces.is_empty() {
            return Err(DirectAirVaccVerifierErrorV19::AirTraceCount);
        }
        traces.insert(1, poseidon);
        if traces.len() != module.airs::<NativeSC>().len() {
            return Err(DirectAirVaccVerifierErrorV19::AirTraceCount);
        }
        Ok(DirectAirExactFiniteVaccVerifierTraceV19 {
            traces,
            poseidon_permutation_inputs: self.poseidon_permutation_inputs,
            poseidon_compression_inputs: self.poseidon_compression_inputs,
            output_instance: self.output_instance,
            output_instance_digest: self.output_instance_digest,
            final_transcript_checkpoint: self.final_transcript_checkpoint,
        })
    }
}

/// Batched witness for one numeric-shape/prior-mode verifier module. Proof
/// indices are private, dense, group-local identifiers in `0..records.len()`.
/// They are not global History record identifiers. A multi-group History
/// composition must isolate each group's internal buses and certify an
/// explicit `(group_id, local_proof_idx) -> history_proof_idx` mapping before
/// forwarding statements to shared History buses. This module intentionally
/// does not claim that such a remapping has happened.
pub struct DirectAirVaccVerifierBatchTraceV19 {
    /// One matrix per AIR, in exactly the order returned by
    /// [`DirectAirVaccVerifierModuleV19::airs`].
    pub traces: Vec<RowMajorMatrix<F>>,
    pub poseidon_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub poseidon_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub output_instances: Vec<AccumulatorInstance<EF, Digest>>,
    pub output_instance_digests: Vec<Digest>,
}

/// Verifier witness packet for a parent-owned Poseidon table.
///
/// `traces` matches
/// [`DirectAirVaccVerifierModuleV19::airs_without_poseidon`] exactly. The two
/// input vectors are complete for this verifier bus owner, including transcript,
/// Merkle, accumulator-digest, and optional CUDA shared-forest requests. They
/// must be inserted as one owner entry in a multi-bus Poseidon table; combining
/// their multiplicities with another owner's bus would erase namespace
/// separation.
pub struct DirectAirVaccSharedPoseidonBatchTraceV19 {
    pub traces: Vec<RowMajorMatrix<F>>,
    pub poseidon_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub poseidon_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub output_instances: Vec<AccumulatorInstance<EF, Digest>>,
    pub output_instance_digests: Vec<Digest>,
}

impl DirectAirVaccSharedPoseidonBatchTraceV19 {
    fn with_local_poseidon_table(
        self,
        module: &DirectAirVaccVerifierModuleV19,
    ) -> Result<DirectAirVaccVerifierBatchTraceV19, DirectAirVaccVerifierErrorV19> {
        let poseidon2_trace = module
            .transcript
            .build_poseidon2_trace(
                self.poseidon_permutation_inputs.clone(),
                self.poseidon_compression_inputs.clone(),
                None,
            )
            .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
                "batched Poseidon trace generation",
            ))?;
        let mut traces = self.traces;
        if traces.is_empty() {
            return Err(DirectAirVaccVerifierErrorV19::AirTraceCount);
        }
        // Full verifier order is TranscriptAir, Poseidon2Air, then the
        // algebra/authentication AIRs. The shared packet omits only index 1.
        traces.insert(1, poseidon2_trace);
        if traces.len() != module.airs::<NativeSC>().len() {
            return Err(DirectAirVaccVerifierErrorV19::AirTraceCount);
        }
        Ok(DirectAirVaccVerifierBatchTraceV19 {
            traces,
            poseidon_permutation_inputs: self.poseidon_permutation_inputs,
            poseidon_compression_inputs: self.poseidon_compression_inputs,
            output_instances: self.output_instances,
            output_instance_digests: self.output_instance_digests,
        })
    }
}

impl DirectAirVaccVerifierBatchTraceV19 {
    #[must_use]
    pub fn air_matrices(&self) -> &[RowMajorMatrix<F>] {
        &self.traces
    }

    pub fn into_air_matrices(self) -> Vec<RowMajorMatrix<F>> {
        self.traces
    }
}

struct DirectAirVaccRecordTraceV19 {
    /// Matrices after TranscriptAir and Poseidon2Air, in verifier AIR order.
    traces: Vec<RowMajorMatrix<F>>,
    schedule: DirectAirVaccTranscriptScheduleV19,
    external_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    external_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    output_instance: AccumulatorInstance<EF, Digest>,
    output_instance_digest: Digest,
}

struct DirectAirReducedSwirlRecordTraceV19 {
    /// Matrices after TranscriptAir, in reduced verifier AIR order. The
    /// parent-owned Poseidon table is omitted.
    traces: Vec<RowMajorMatrix<F>>,
    schedule: DirectAirVaccTranscriptScheduleV19,
    external_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    external_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    output_instance: AccumulatorInstance<EF, Digest>,
    output_instance_digest: Digest,
    end_state: [F; POSEIDON2_WIDTH],
}

/// Split at the exact source/VACC event boundary and prove that resetting the
/// duplex bookkeeping there is faithful. The VACC suffix starts with an
/// absorb event and the source checkpoint has no pending absorb lanes; those
/// are precisely the conditions required by `TranscriptLog` resumption.
fn exact_v19_resumable_suffix(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    start_tidx: usize,
    expected_state: [F; POSEIDON2_WIDTH],
) -> Result<TranscriptLog<F, [F; POSEIDON2_WIDTH]>, DirectAirVaccVerifierErrorV19> {
    let (event_index, event) = log
        .events()
        .iter()
        .enumerate()
        .find(|(_, event)| event.operation_range.start == start_tidx)
        .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
            "VACC resume event boundary",
        ))?;
    let checkpoint = TranscriptCheckpoint {
        operations: start_tidx,
        events: event_index,
        permutations: event.permutation_range.start,
    };
    if log.resumable_checkpoint_before::<CHUNK>(checkpoint) != Some(checkpoint) {
        return Err(DirectAirVaccVerifierErrorV19::Transcript(
            "non-resumable VACC checkpoint",
        ));
    }
    let suffix = log
        .suffix(checkpoint)
        .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
            "VACC transcript suffix",
        ))?;
    if suffix.is_empty()
        || suffix.perm_results().first().copied() != Some(expected_state)
        || suffix
            .events()
            .first()
            .is_none_or(|event| event.operation_range.start != 0)
    {
        return Err(DirectAirVaccVerifierErrorV19::Transcript(
            "VACC resume state",
        ));
    }
    Ok(suffix)
}

fn exact_v19_prefix_through(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    end_tidx: usize,
) -> Result<TranscriptLog<F, [F; POSEIDON2_WIDTH]>, DirectAirVaccVerifierErrorV19> {
    let (event_index, event) = log
        .events()
        .iter()
        .enumerate()
        .find(|(_, event)| event.operation_range.end == end_tidx)
        .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
            "exact-finite call end boundary",
        ))?;
    log.prefix(TranscriptCheckpoint {
        operations: end_tidx,
        events: event_index + 1,
        permutations: event.permutation_range.end,
    })
    .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
        "exact-finite call prefix",
    ))
}

fn exact_v19_checkpoint_at(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    operation_index: usize,
) -> Result<TranscriptCheckpointRecordV19, DirectAirVaccVerifierErrorV19> {
    if operation_index == 0 || operation_index > log.len() {
        return Err(DirectAirVaccVerifierErrorV19::Transcript(
            "exact-finite checkpoint operation index",
        ));
    }
    let prefix = exact_v19_prefix_through(log, operation_index)?;
    let sample_count = prefix
        .samples()
        .iter()
        .rev()
        .take_while(|&&is_sample| is_sample)
        .count();
    if sample_count == 0 || sample_count > CHUNK {
        return Err(DirectAirVaccVerifierErrorV19::Transcript(
            "exact-finite checkpoint sample count",
        ));
    }
    let state =
        prefix
            .perm_results()
            .last()
            .copied()
            .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
                "exact-finite checkpoint sponge state",
            ))?;
    Ok(TranscriptCheckpointRecordV19 {
        operation_index: operation_index.try_into().map_err(|_| {
            DirectAirVaccVerifierErrorV19::Transcript("exact-finite checkpoint index width")
        })?,
        sample_count: sample_count.try_into().map_err(|_| {
            DirectAirVaccVerifierErrorV19::Transcript("exact-finite checkpoint sample width")
        })?,
        state,
    })
}

fn validate_exact_v19_checkpoint(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    checkpoint: TranscriptCheckpointRecordV19,
) -> Result<(), DirectAirVaccVerifierErrorV19> {
    if exact_v19_checkpoint_at(log, checkpoint.operation_index as usize)? != checkpoint {
        return Err(DirectAirVaccVerifierErrorV19::Transcript(
            "non-canonical exact-finite checkpoint",
        ));
    }
    Ok(())
}

fn exact_v19_resumable_interval(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    start_tidx: usize,
    end_tidx: usize,
    expected_state: [F; POSEIDON2_WIDTH],
) -> Result<TranscriptLog<F, [F; POSEIDON2_WIDTH]>, DirectAirVaccVerifierErrorV19> {
    let prefix = exact_v19_prefix_through(log, end_tidx)?;
    exact_v19_resumable_suffix(&prefix, start_tidx, expected_state)
}

impl DirectAirVaccVerifierTraceV19 {
    #[must_use]
    pub fn air_matrices(&self) -> &[RowMajorMatrix<F>] {
        &self.traces
    }

    pub fn into_air_matrices(self) -> Vec<RowMajorMatrix<F>> {
        self.traces
    }
}

impl DirectAirVaccVerifierModuleV19 {
    /// Derive a proof-qualified exact-finite record without accepting host
    /// checkpoint metadata. The native verifier's phase spans determine the
    /// call interval; the authenticated transcript determines terminal sample
    /// counts and sponge states. Call zero uses the canonical no-resume state
    /// while its operation index remains the derived end of the schedule
    /// prefix proved in the same transcript AIR.
    pub fn exact_finite_record_from_authenticated_transcript<'a>(
        &self,
        proof: &'a DirectAirExactFiniteVaccProofV19,
        verification: &'a DirectAirExactFiniteVaccVerificationV19,
        transcript: &'a TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
        prior: Option<&'a AccumulatorInstance<EF, Digest>>,
    ) -> Result<DirectAirExactFiniteVaccVerifierRecordV19<'a>, DirectAirVaccVerifierErrorV19> {
        let exact = self.exact_config()?;
        let protocol = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::VaccProtocolPrefix,
        )?;
        let target = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::FinalTarget,
        )?;
        let start = protocol.operation_range.start;
        let boundary = unique_v19_phase(
            &verification.transcript_phases,
            &NativeTranscriptPhase::ExactFiniteCallBoundary {
                call: exact.call_index.try_into().map_err(|_| {
                    DirectAirVaccVerifierErrorV19::Transcript(
                        "exact-finite boundary call-index width",
                    )
                })?,
            },
        )?;
        if boundary.operation_range.start != target.operation_range.end
            || boundary.operation_range.len() != 2 * D_EF
        {
            return Err(DirectAirVaccVerifierErrorV19::Transcript(
                "exact-finite authenticated call boundary",
            ));
        }
        let end = boundary.operation_range.end;
        let start_checkpoint = if exact.call_index == 0 {
            let schedule_end =
                native_exact_finite_vacc_schedule_prefix_elements(&exact.transcript_profile)
                    .map_err(|_| DirectAirVaccVerifierErrorV19::InvalidProfile)?
                    .len()
                    * D_EF;
            if start != schedule_end {
                return Err(DirectAirVaccVerifierErrorV19::Transcript(
                    "exact-finite schedule/call boundary",
                ));
            }
            TranscriptCheckpointRecordV19 {
                operation_index: start.try_into().map_err(|_| {
                    DirectAirVaccVerifierErrorV19::Transcript(
                        "exact-finite start checkpoint index width",
                    )
                })?,
                sample_count: 0,
                state: [F::ZERO; POSEIDON2_WIDTH],
            }
        } else {
            exact_v19_checkpoint_at(transcript, start)?
        };
        let end_checkpoint = exact_v19_checkpoint_at(transcript, end)?;
        let record = DirectAirExactFiniteVaccVerifierRecordV19 {
            proof,
            verification,
            transcript,
            prior,
            start_checkpoint,
            end_checkpoint,
        };
        let _authenticated_fresh_root = self.validate_exact_finite_record(&record)?;
        DirectAirVaccTranscriptScheduleV19::from_exact_record(self, &record)?;
        Ok(record)
    }

    /// Generate the complete recursive authority for one native exact-finite
    /// WARP call, including its local Poseidon table. This consumes the real
    /// native proof and verifier record; there is no success bit or replay
    /// receipt that can substitute for the authenticated contexts below.
    pub fn generate_exact_finite_trace(
        &self,
        config: &NativeSC,
        record: &DirectAirExactFiniteVaccVerifierRecordV19<'_>,
    ) -> Result<DirectAirExactFiniteVaccVerifierTraceV19, DirectAirVaccVerifierErrorV19> {
        self.generate_exact_finite_trace_for_shared_poseidon(config, record)?
            .with_local_poseidon_table(self)
    }

    /// Shared-Poseidon production API for one exact-finite call. The returned
    /// packet covers schedule and call prefixes, stacked-fresh and prior
    /// authentication, both sumchecks, OOD/shift/batching bridges, transcript
    /// checkpoints, and output-accumulator reconstruction.
    pub fn generate_exact_finite_trace_for_shared_poseidon(
        &self,
        config: &NativeSC,
        record: &DirectAirExactFiniteVaccVerifierRecordV19<'_>,
    ) -> Result<DirectAirExactFiniteVaccSharedPoseidonTraceV19, DirectAirVaccVerifierErrorV19> {
        if self.fresh_commitment_mode != DirectAirVaccFreshCommitmentModeV19::FiniteStackedBase {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "exact-finite record supplied to legacy verifier",
            ));
        }
        let exact = self.exact_config()?;
        let call = exact.call();
        let proof_idx = exact.call_index;
        let fresh_root = self.validate_exact_finite_record(record)?;
        let schedule = DirectAirVaccTranscriptScheduleV19::from_exact_record(self, record)?;
        let cursor = generate_direct_air_vacc_cursor_trace_from_parts_v19(
            self,
            record.transcript,
            record.verification,
            &schedule,
        )?;
        let fresh_claims = record
            .proof
            .fresh_claims
            .iter()
            .map(DirectAirVaccFreshClaimDataV19::from)
            .collect::<Vec<_>>();
        let algebra = generate_direct_air_vacc_algebra_traces_from_parts_v19(
            self,
            proof_idx,
            &fresh_claims,
            record.verification,
            record.transcript,
            record.prior,
            &schedule,
        )?;

        let fresh_profile = self.exact_finite_fresh_profile();
        let descriptors = record
            .proof
            .fresh_claims
            .iter()
            .map(|claim| claim.commitment.clone())
            .collect::<Vec<_>>();
        let fresh_commitment = generate_native_finite_stacked_fresh_commitment_trace(
            proof_idx,
            schedule.fresh_root_tidx,
            &descriptors,
            fresh_profile,
        )
        .ok_or(DirectAirVaccVerifierErrorV19::Merkle(
            "stacked fresh commitment trace",
        ))?;
        let flat_indices = record
            .verification
            .shifts
            .iter()
            .map(|shift| shift.index as usize)
            .collect::<Vec<_>>();
        let fresh_proof = record.proof.openings.fresh_opening_proofs.first().ok_or(
            DirectAirVaccVerifierErrorV19::Merkle("stacked fresh opening proof"),
        )?;
        let shared_fresh = record.verification.fresh_authentication[0].shared.as_ref();
        let fresh_opening = generate_native_finite_stacked_fresh_opening_traces(
            config.hasher(),
            proof_idx,
            &flat_indices,
            fresh_proof,
            shared_fresh,
            fresh_profile,
        )
        .ok_or(DirectAirVaccVerifierErrorV19::Merkle(
            "stacked fresh opening traces",
        ))?;

        let prior_projection =
            if let Some(authentication) = record.verification.prior_authentication.as_ref() {
                Some(
                    generate_native_exact_finite_codeword_projection_trace_checked(
                        config.hasher(),
                        proof_idx,
                        authentication,
                        self.profile.log_codeword_len,
                        self.profile.rows_per_query,
                        self.trees.prior_rows,
                        u32::try_from(self.trees.prior_outer)
                            .map_err(|_| DirectAirVaccVerifierErrorV19::Merkle("prior tree id"))?,
                        call.input_arity - 1,
                        fresh_profile.variant(),
                        call.input_arity,
                        None,
                    )
                    .map_err(DirectAirVaccVerifierErrorV19::Merkle)?,
                )
            } else {
                None
            };

        let mut leaf_hash_matrices = vec![fresh_opening.leaf_hash.matrix];
        let mut merkle_matrices = vec![fresh_opening.merkle];
        let mut adapter_matrices = vec![fresh_opening.leaf_adapter];
        let mut permutation_inputs = fresh_opening.leaf_hash.permutation_inputs;
        let mut compression_inputs = fresh_opening.compression_inputs;
        if let (Some(projection), Some(authentication)) = (
            prior_projection.as_ref(),
            record.verification.prior_authentication.as_ref(),
        ) {
            self.collect_projection_authentication(
                config,
                proof_idx,
                projection,
                authentication,
                self.trees.prior_outer,
                &mut leaf_hash_matrices,
                &mut merkle_matrices,
                &mut adapter_matrices,
                &mut permutation_inputs,
                &mut compression_inputs,
            )?;
        }
        let leaf_hash =
            merge_v19_active_rows(&leaf_hash_matrices, NativeLeafHashCols::<F>::width()).ok_or(
                DirectAirVaccVerifierErrorV19::Merkle("exact-finite leaf hash merge"),
            )?;
        let merkle =
            merge_v19_active_rows(&merkle_matrices, NativeMerkleCompressionCols::<F>::width())
                .ok_or(DirectAirVaccVerifierErrorV19::Merkle(
                    "exact-finite Merkle merge",
                ))?;
        let leaf_adapter =
            merge_v19_active_rows(&adapter_matrices, NativeMerkleLeafAdapterCols::<F>::width())
                .ok_or(DirectAirVaccVerifierErrorV19::Merkle(
                    "exact-finite leaf adapter merge",
                ))?;

        let prior_digest = record
            .prior
            .map(|prior| {
                generate_native_accumulator_digest_traces(
                    proof_idx,
                    prior,
                    &self.private_accumulator,
                )
                .ok_or(DirectAirVaccVerifierErrorV19::RecordShape(
                    "exact-finite prior accumulator digest",
                ))
            })
            .transpose()?;
        let output_digest = generate_native_accumulator_digest_traces(
            proof_idx,
            &record.verification.output_instance,
            &self.private_accumulator,
        )
        .ok_or(DirectAirVaccVerifierErrorV19::RecordShape(
            "exact-finite output accumulator digest",
        ))?;
        append_v19_accumulator_poseidon(
            prior_digest.as_ref(),
            &mut permutation_inputs,
            &mut compression_inputs,
        );
        append_v19_accumulator_poseidon(
            Some(&output_digest),
            &mut permutation_inputs,
            &mut compression_inputs,
        );

        let schedule_prefix = (exact.call_index == 0)
            .then(|| generate_native_exact_finite_vacc_schedule_prefix_trace(0, 0));
        let call_prefix =
            generate_native_exact_finite_vacc_call_prefix_trace(proof_idx, schedule.start_tidx);
        let end = generate_native_standard_vacc_end_trace(
            proof_idx,
            schedule.end_tag_tidx,
            schedule.end_tidx,
            schedule.discarded_end_sample,
        );
        let prior_root = schedule.prior.as_ref().map(|prior| {
            generate_native_standard_vacc_commitment_root_trace(
                proof_idx,
                prior.root_tidx,
                record.prior.expect("validated exact prior").rt,
            )
        });
        let output_root = generate_native_standard_vacc_commitment_root_trace(
            proof_idx,
            schedule.output_root_tidx,
            record.verification.output_instance.rt,
        );
        let authority = generate_direct_air_exact_finite_vacc_authority_trace_v19(
            record,
            fresh_root,
            prior_digest.as_ref().map(|digest| digest.instance_digest),
            output_digest.instance_digest,
        );
        let exp_bits = generate_direct_v19_exp_bits_trace_from_log(
            record.transcript,
            &schedule,
            self.profile.log_codeword_len,
        )?;

        let mut tail = Vec::new();
        if let Some(schedule_prefix) = schedule_prefix {
            tail.push(schedule_prefix);
        }
        tail.extend([call_prefix, cursor, end]);
        tail.extend([
            algebra.claim_values,
            algebra.claim_layout,
            algebra.slot_layout,
            fresh_commitment,
            fresh_opening.projection,
        ]);
        if let Some(projection) = prior_projection {
            tail.push(projection.matrix);
        }
        if let Some(prior_root) = prior_root {
            tail.push(prior_root);
        }
        tail.extend([leaf_hash, merkle, leaf_adapter]);
        tail.extend([
            algebra.vector,
            algebra.selector_eq,
            algebra.xi_eq,
            algebra.opening_eq,
            algebra.omega,
            algebra.twin_sigma,
            algebra.sumcheck,
            algebra.twin_fold,
            algebra.twin_final,
            algebra.initial_point,
            algebra.initial_target,
            algebra.opening_padding,
            algebra.ood_claims,
            algebra.shift_schedule,
            algebra.shift_merge,
            algebra.batching_sigma,
            algebra.batching_final,
            algebra.batching_alpha,
            output_root,
        ]);
        if let Some(prior) = prior_digest {
            tail.extend([prior.values, prior.hash, prior.root]);
        }
        tail.extend([
            output_digest.values,
            output_digest.hash,
            output_digest.root,
            authority,
            exp_bits,
        ]);

        let owned_log;
        let (logs, resumes, checkpoints) = if exact.call_index == 0 {
            owned_log = exact_v19_prefix_through(record.transcript, schedule.end_tidx)?;
            (
                vec![&owned_log],
                Vec::new(),
                vec![Some([None, Some(schedule.end_tidx)])],
            )
        } else {
            owned_log = exact_v19_resumable_interval(
                record.transcript,
                schedule.start_tidx,
                schedule.end_tidx,
                record.start_checkpoint.state,
            )?;
            (
                vec![&owned_log],
                vec![Some((schedule.start_tidx, record.start_checkpoint.state))],
                vec![Some([None, Some(schedule.end_tidx)])],
            )
        };
        let transcript = self
            .transcript
            .generate_trace_inputs_with_external_resumed_and_optional_checkpoints(
                &logs,
                &resumes,
                &checkpoints,
                permutation_inputs,
                compression_inputs,
                None,
            )
            .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
                "exact-finite transcript trace generation",
            ))?;
        let mut traces = Vec::with_capacity(tail.len() + 1);
        traces.push(transcript.trace);
        traces.extend(tail);
        if traces.len() != self.airs_without_poseidon::<NativeSC>().len() {
            return Err(DirectAirVaccVerifierErrorV19::AirTraceCount);
        }
        Ok(DirectAirExactFiniteVaccSharedPoseidonTraceV19 {
            traces,
            poseidon_permutation_inputs: transcript.permutation_inputs,
            poseidon_compression_inputs: transcript.compression_inputs,
            output_instance: record.verification.output_instance.clone(),
            output_instance_digest: output_digest.instance_digest,
            final_transcript_checkpoint: record.end_checkpoint,
        })
    }

    /// Generate one setup-fixed reduced-SWIRL verifier batch. Every native
    /// transition remains an ordinary WARP Verify record; only the physical
    /// AIR tables are merged. Original application roots are authenticated by
    /// the direct stacked-RS opening tables and are never replaced by a scalar
    /// commitment.
    pub fn generate_reduced_swirl_traces_for_shared_poseidon(
        &self,
        config: &NativeSC,
        records: &[DirectAirReducedSwirlVaccVerifierRecordV19<'_>],
    ) -> Result<DirectAirVaccSharedPoseidonBatchTraceV19, DirectAirVaccVerifierErrorV19> {
        self.reduced_swirl
            .as_ref()
            .ok_or(DirectAirVaccVerifierErrorV19::InvalidProfile)?;
        if records.is_empty() {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "empty reduced-SWIRL VACC batch",
            ));
        }
        // Merge each verifier record as soon as it has been generated.  A
        // reduced-SWIRL record contains several independently padded tables;
        // retaining all records and then merging them made peak host memory
        // proportional to the sum of those padded tables (more than 40 GiB
        // for a 38-call real block).  The merged AIRs need only the active
        // rows, while transcript construction needs compact checkpoints and
        // Poseidon requests.  This is the same consume-after-parent pattern
        // used by OpenVM's recursive proof tree.
        let mut merged_values: Vec<Vec<F>> = Vec::new();
        let mut merged_widths = Vec::new();
        let mut claim_layout = None;
        let mut schedule_end_tidxs = Vec::with_capacity(records.len());
        let mut end_states = Vec::with_capacity(records.len());
        let mut external_permutation_inputs = Vec::new();
        let mut external_compression_inputs = Vec::new();
        let mut output_instances = Vec::with_capacity(records.len());
        let mut output_instance_digests = Vec::with_capacity(records.len());
        let mut start_states = Vec::with_capacity(records.len());
        let has_any_prior = records.iter().any(|record| record.prior.is_some());
        // Per-record tail order begins cursor, end, claim values, then the
        // setup-fixed claim-layout table. Dynamic slot rows are owned by the
        // reduced aggregate and are intentionally absent here.
        let claim_layout_index = 3;
        for (proof_idx, record) in records.iter().enumerate() {
            if record.local_proof_idx != proof_idx
                || record.prior.is_some() != (record.proof_idx != 0)
                || (proof_idx != 0
                    && (record.batch_start_tidx != records[proof_idx - 1].vacc_end_tidx
                        || record.prior
                            != Some(&records[proof_idx - 1].verification.output_instance)))
            {
                return Err(DirectAirVaccVerifierErrorV19::PriorChain);
            }
            let start_state = if record.proof_idx == 0 {
                [F::ZERO; POSEIDON2_WIDTH]
            } else {
                exact_v19_checkpoint_at(record.transcript, record.batch_start_tidx)?.state
            };
            if proof_idx != 0 && start_state != end_states[proof_idx - 1] {
                return Err(DirectAirVaccVerifierErrorV19::PriorChain);
            }
            let generated = self.generate_reduced_swirl_record_trace(config, record)?;
            if proof_idx == 0 {
                merged_values.reserve_exact(generated.traces.len());
                merged_widths.reserve_exact(generated.traces.len());
                for matrix in &generated.traces {
                    let width = matrix.width();
                    let capacity = matrix
                        .values
                        .len()
                        .checked_mul(records.len())
                        .ok_or(DirectAirVaccVerifierErrorV19::AirTraceCount)?;
                    merged_values.push(Vec::with_capacity(capacity));
                    merged_widths.push(width);
                }
            } else if generated.traces.len() != merged_values.len() {
                return Err(DirectAirVaccVerifierErrorV19::AirTraceCount);
            }
            // The final nine record-local tables are, in order: output root,
            // prior adapter, prior value/hash/root, output value/hash/root,
            // and ExpBitsLen.  Validate this fixed inventory before using the
            // generic active-row compactor below.  In particular, an output
            // accumulator is never optional.
            let trailing = generated
                .traces
                .len()
                .checked_sub(9)
                .ok_or(DirectAirVaccVerifierErrorV19::AirTraceCount)?;
            let prior_value_index = trailing + 2;
            let prior_hash_index = trailing + 3;
            let output_value_index = trailing + 5;
            let output_hash_index = trailing + 6;
            let output_root_index = trailing + 7;
            if generated.traces[prior_value_index].width()
                != NativeAccumulatorValueCols::<F>::width()
                || generated.traces[prior_hash_index].width()
                    != NativeAccumulatorHashCols::<F>::width()
                || generated.traces[output_value_index].width()
                    != NativeAccumulatorValueCols::<F>::width()
                || generated.traces[output_hash_index].width()
                    != NativeAccumulatorHashCols::<F>::width()
                || generated.traces[output_root_index].width()
                    != NativeAccumulatorRootDigestCols::<F>::width()
                || generated.traces[output_value_index].values.first() != Some(&F::ONE)
                || generated.traces[output_hash_index].values.first() != Some(&F::ONE)
                || generated.traces[output_root_index].values.first() != Some(&F::ONE)
            {
                return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                    "reduced-SWIRL accumulator trace inventory",
                ));
            }
            for (matrix_index, matrix) in generated.traces.iter().enumerate() {
                let width = *merged_widths
                    .get(matrix_index)
                    .ok_or(DirectAirVaccVerifierErrorV19::AirTraceCount)?;
                if width == 0 || matrix.width() != width {
                    return Err(DirectAirVaccVerifierErrorV19::AirTraceCount);
                }
                if matrix_index == claim_layout_index {
                    if proof_idx == 0 {
                        claim_layout = Some(matrix.clone());
                    }
                    continue;
                }
                let destination = merged_values
                    .get_mut(matrix_index)
                    .ok_or(DirectAirVaccVerifierErrorV19::AirTraceCount)?;
                for (row_index, row) in matrix.values.chunks_exact(width).enumerate() {
                    // `NativeAccumulator{Value,Hash}Air` require a canonical
                    // first-row marker even when an optional prior table is
                    // wholly empty.  Keep that marker only when the complete
                    // physical batch has no prior.  If later records do have
                    // priors, retaining an inactive bootstrap row before
                    // their active rows would violate the table's monotone
                    // activity constraint, so the marker must be compacted
                    // away in that case.
                    let keep_empty_prior_marker = !has_any_prior
                        && row_index == 0
                        && (matrix_index == prior_value_index || matrix_index == prior_hash_index);
                    if row[0] != F::ZERO || keep_empty_prior_marker {
                        destination.extend_from_slice(row);
                    }
                }
            }
            schedule_end_tidxs.push(generated.schedule.end_tidx);
            end_states.push(generated.end_state);
            external_permutation_inputs.extend(generated.external_permutation_inputs);
            external_compression_inputs.extend(generated.external_compression_inputs);
            output_instances.push(generated.output_instance);
            output_instance_digests.push(generated.output_instance_digest);
            start_states.push(start_state);
            // `generated` and all of its padded per-record matrices are
            // released here instead of remaining live for the whole batch.
        }

        let mut owned_logs = Vec::with_capacity(records.len());
        let mut resumes = Vec::with_capacity(records.len());
        for (proof_idx, record) in records.iter().enumerate() {
            // The final physical transcript row set owns the one canonical
            // suffix after VACC as well: the authenticated manifest footer
            // followed immediately by terminal Decide/WHIR.  Footer and
            // terminal AIRs consume that suffix from the final call's proof
            // namespace.  Stopping every log at `vacc_end_tidx` would leave
            // those consumers unauthenticated (and unbalanced) even though
            // the native verifier recorded the complete transcript.
            let transcript_end = if record.is_final_call {
                record.transcript.len()
            } else {
                record.vacc_end_tidx
            };
            if record.proof_idx == 0 {
                owned_logs.push(exact_v19_prefix_through(record.transcript, transcript_end)?);
                resumes.push(None);
            } else {
                owned_logs.push(exact_v19_resumable_interval(
                    record.transcript,
                    record.batch_start_tidx,
                    transcript_end,
                    start_states[proof_idx],
                )?);
                resumes.push(Some((record.batch_start_tidx, start_states[proof_idx])));
            }
        }
        let logs = owned_logs.iter().collect::<Vec<_>>();
        let checkpoints = schedule_end_tidxs
            .iter()
            .copied()
            .map(|end_tidx| Some([None, Some(end_tidx)]))
            .collect::<Vec<_>>();
        let transcript = self
            .transcript
            .generate_trace_inputs_with_external_resumed_and_optional_checkpoints(
                &logs,
                &resumes,
                &checkpoints,
                external_permutation_inputs,
                external_compression_inputs,
                None,
            )
            .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
                "reduced-SWIRL transcript trace generation",
            ))?;

        let matrix_count = merged_values.len();
        if matrix_count == 0 || merged_widths.len() != matrix_count {
            return Err(DirectAirVaccVerifierErrorV19::AirTraceCount);
        }
        let mut traces = Vec::with_capacity(matrix_count + 1);
        traces.push(transcript.trace);
        for matrix_index in 0..matrix_count {
            let merged = if matrix_index == claim_layout_index {
                scale_v19_claim_layout_lookups(
                    claim_layout
                        .take()
                        .ok_or(DirectAirVaccVerifierErrorV19::AirTraceCount)?,
                    records.len(),
                )
            } else {
                let width = merged_widths[matrix_index];
                let mut values = core::mem::take(&mut merged_values[matrix_index]);
                let active_rows = values.len() / width;
                let height = active_rows.max(1).next_power_of_two();
                values.resize(
                    height
                        .checked_mul(width)
                        .ok_or(DirectAirVaccVerifierErrorV19::AirTraceCount)?,
                    F::ZERO,
                );
                if matrix_index + 1 == matrix_count {
                    // `ExpBitsLenAir` has a multiplicative neutral inactive
                    // row instead of the generic all-zero inactive row.
                    let exp_width = ExpBitsLenCols::<F>::width();
                    if width != exp_width {
                        return Err(DirectAirVaccVerifierErrorV19::AirTraceCount);
                    }
                    for row in values[active_rows * width..].chunks_exact_mut(width) {
                        let cols: &mut ExpBitsLenCols<F> = row.borrow_mut();
                        cols.result = F::ONE;
                        cols.result_multiplier = F::ONE;
                    }
                }
                RowMajorMatrix::new(values, width)
            };
            traces.push(merged);
        }
        let airs = self.airs_without_poseidon::<NativeSC>();
        if traces.len() != airs.len()
            || traces
                .iter()
                .zip(&airs)
                .any(|(trace, air)| trace.width() != BaseAir::<F>::width(air.as_ref()))
        {
            return Err(DirectAirVaccVerifierErrorV19::AirTraceCount);
        }
        let output_value_index = traces
            .len()
            .checked_sub(4)
            .ok_or(DirectAirVaccVerifierErrorV19::AirTraceCount)?;
        let output_hash_index = traces.len() - 3;
        let output_root_index = traces.len() - 2;
        if traces[output_value_index].values.first() != Some(&F::ONE)
            || traces[output_hash_index].values.first() != Some(&F::ONE)
            || traces[output_root_index].values.first() != Some(&F::ONE)
        {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "empty reduced-SWIRL output accumulator batch",
            ));
        }
        Ok(DirectAirVaccSharedPoseidonBatchTraceV19 {
            traces,
            poseidon_permutation_inputs: transcript.permutation_inputs,
            poseidon_compression_inputs: transcript.compression_inputs,
            output_instances,
            output_instance_digests,
        })
    }

    fn generate_reduced_swirl_record_trace(
        &self,
        config: &NativeSC,
        record: &DirectAirReducedSwirlVaccVerifierRecordV19<'_>,
    ) -> Result<DirectAirReducedSwirlRecordTraceV19, DirectAirVaccVerifierErrorV19> {
        let reduced = self
            .reduced_swirl
            .as_ref()
            .ok_or(DirectAirVaccVerifierErrorV19::InvalidProfile)?;
        let proof = record.proof.inner();
        let local_proof_idx = record.local_proof_idx;
        let fresh_count = proof.fresh_claims.len();
        let expected_kinds = (0..reduced.input_arity)
            .map(|slot| {
                if slot < fresh_count {
                    WarpStepInputKind::Fresh
                } else if record.prior.is_some() && slot == fresh_count {
                    WarpStepInputKind::PriorAccumulator
                } else {
                    WarpStepInputKind::Padding
                }
            })
            .collect::<Vec<_>>();
        if proof.input_kinds != expected_kinds
            || proof.output_instance != record.verification.output_instance
            || proof.openings.fresh_opening_proofs.len() != fresh_count
            || proof.openings.acc_opening_proofs.len() != usize::from(record.prior.is_some())
            || record.verification.prior_authentication.is_some() != record.prior.is_some()
            || record.vacc_start_tidx <= record.batch_start_tidx
            || record.vacc_end_tidx <= record.vacc_start_tidx
        {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "reduced-SWIRL proof envelope",
            ));
        }
        let schedule = DirectAirVaccTranscriptScheduleV19::from_reduced_swirl_record(self, record)?;
        let cursor = generate_direct_air_vacc_cursor_trace_from_parts_v19(
            self,
            record.transcript,
            record.verification,
            &schedule,
        )?;
        let fresh_claims = proof
            .fresh_claims
            .iter()
            .map(DirectAirVaccFreshClaimDataV19::from)
            .collect::<Vec<_>>();
        let algebra = generate_direct_air_vacc_algebra_traces_from_parts_v19(
            self,
            local_proof_idx,
            &fresh_claims,
            record.verification,
            record.transcript,
            record.prior,
            &schedule,
        )?;
        let root_tree_stride = 1 + self.profile.num_shift_queries;
        let fresh_opening = generate_native_direct_fresh_opening_traces(
            config.hasher(),
            &proof
                .fresh_claims
                .iter()
                .map(|claim| claim.commitment.clone())
                .collect::<Vec<_>>(),
            &record.verification.fresh_authentication,
            &record.commitment_tidxs,
            local_proof_idx,
            reduced.input_arity,
            reduced.max_roots_per_source,
            0,
            u32::try_from(root_tree_stride)
                .map_err(|_| DirectAirVaccVerifierErrorV19::InvalidProfile)?,
            reduced.projection_sources_per_shard,
            reduced.max_projection_height,
        )
        .ok_or(DirectAirVaccVerifierErrorV19::Merkle(
            "reduced-SWIRL original-root openings",
        ))?;
        let prior_projection = record
            .verification
            .prior_authentication
            .as_ref()
            .map(|authentication| {
                generate_native_accumulator_projection_trace_checked(
                    config.hasher(),
                    local_proof_idx,
                    authentication,
                    self.profile.log_codeword_len,
                    self.profile.rows_per_query,
                    self.trees.prior_rows,
                    u32::try_from(self.trees.prior_outer)
                        .map_err(|_| DirectAirVaccVerifierErrorV19::InvalidProfile)?,
                    fresh_count,
                    reduced.input_arity,
                    None,
                )
                .map_err(DirectAirVaccVerifierErrorV19::Merkle)
            })
            .transpose()?;

        let mut leaf_hash_matrices = vec![fresh_opening.leaf_hash.matrix];
        let mut merkle_matrices = vec![fresh_opening.merkle];
        let mut adapter_matrices = vec![fresh_opening.leaf_adapter];
        let mut permutation_inputs = fresh_opening.leaf_hash.permutation_inputs;
        let mut compression_inputs = fresh_opening.compression_inputs;
        if let (Some(projection), Some(authentication)) = (
            prior_projection.as_ref(),
            record.verification.prior_authentication.as_ref(),
        ) {
            self.collect_projection_authentication(
                config,
                local_proof_idx,
                projection,
                authentication,
                self.trees.prior_outer,
                &mut leaf_hash_matrices,
                &mut merkle_matrices,
                &mut adapter_matrices,
                &mut permutation_inputs,
                &mut compression_inputs,
            )?;
        }
        let leaf_hash =
            merge_v19_active_rows(&leaf_hash_matrices, NativeLeafHashCols::<F>::width()).ok_or(
                DirectAirVaccVerifierErrorV19::Merkle("reduced-SWIRL leaf hash merge"),
            )?;
        let merkle =
            merge_v19_active_rows(&merkle_matrices, NativeMerkleCompressionCols::<F>::width())
                .ok_or(DirectAirVaccVerifierErrorV19::Merkle(
                    "reduced-SWIRL Merkle merge",
                ))?;
        let leaf_adapter =
            merge_v19_active_rows(&adapter_matrices, NativeMerkleLeafAdapterCols::<F>::width())
                .ok_or(DirectAirVaccVerifierErrorV19::Merkle(
                    "reduced-SWIRL leaf-adapter merge",
                ))?;

        let prior_digest = record
            .prior
            .map(|prior| {
                generate_native_accumulator_digest_traces(
                    local_proof_idx,
                    prior,
                    &self.private_accumulator,
                )
                .ok_or(DirectAirVaccVerifierErrorV19::RecordShape(
                    "reduced-SWIRL prior accumulator digest",
                ))
            })
            .transpose()?;
        let output_digest = generate_native_accumulator_digest_traces(
            local_proof_idx,
            &record.verification.output_instance,
            &self.private_accumulator,
        )
        .ok_or(DirectAirVaccVerifierErrorV19::RecordShape(
            "reduced-SWIRL output accumulator digest",
        ))?;
        append_v19_accumulator_poseidon(
            prior_digest.as_ref(),
            &mut permutation_inputs,
            &mut compression_inputs,
        );
        append_v19_accumulator_poseidon(
            Some(&output_digest),
            &mut permutation_inputs,
            &mut compression_inputs,
        );

        let end = generate_direct_air_reduced_vacc_end_trace_v19(
            local_proof_idx,
            schedule.end_tag_tidx,
            schedule.end_tidx,
            schedule.discarded_end_sample,
        );
        let prior_root = schedule.prior.as_ref().map(|prior| {
            generate_native_standard_vacc_commitment_root_trace(
                local_proof_idx,
                prior.root_tidx,
                record.prior.expect("validated reduced prior").rt,
            )
        });
        let output_root = generate_native_standard_vacc_commitment_root_trace(
            local_proof_idx,
            schedule.output_root_tidx,
            record.verification.output_instance.rt,
        );
        let prior_adapter = generate_direct_air_reduced_prior_claim_adapter_trace_v19(
            &record
                .prior
                .map(|prior| vec![(local_proof_idx, fresh_count, prior)])
                .unwrap_or_default(),
            reduced.input_arity,
            &self.private_accumulator,
        )?;
        let exp_bits = generate_direct_v19_exp_bits_trace_from_log(
            record.transcript,
            &schedule,
            self.profile.log_codeword_len,
        )?;

        let empty_prior = empty_v19_accumulator_digest_traces(&self.private_accumulator);
        let prior_digest = prior_digest.unwrap_or(empty_prior);
        let empty_projection = RowMajorMatrix::new(
            F::zero_vec(NativeAccumulatorProjectionCols::<F>::width() * 2),
            NativeAccumulatorProjectionCols::<F>::width(),
        );
        let empty_root = RowMajorMatrix::new(
            F::zero_vec(NativeStandardVaccCommitmentRootCols::<F>::width() * 2),
            NativeStandardVaccCommitmentRootCols::<F>::width(),
        );
        let mut traces = vec![
            cursor,
            end,
            algebra.claim_values,
            algebra.claim_layout,
            fresh_opening.commitment,
            fresh_opening.roots,
        ];
        traces.extend(fresh_opening.projections);
        traces.push(prior_projection.map_or(empty_projection, |projection| projection.matrix));
        traces.push(prior_root.unwrap_or(empty_root));
        traces.extend([leaf_hash, merkle, leaf_adapter]);
        traces.extend([
            algebra.vector,
            algebra.selector_eq,
            algebra.xi_eq,
            algebra.opening_eq,
            algebra.omega,
            algebra.twin_sigma,
            algebra.sumcheck,
            algebra.twin_fold,
            algebra.twin_final,
            algebra.initial_point,
            algebra.initial_target,
            algebra.opening_padding,
            algebra.ood_claims,
            algebra.shift_schedule,
            algebra.shift_merge,
            algebra.batching_sigma,
            algebra.batching_final,
            algebra.batching_alpha,
            output_root,
            prior_adapter,
            prior_digest.values,
            prior_digest.hash,
            prior_digest.root,
            output_digest.values,
            output_digest.hash,
            output_digest.root,
            exp_bits,
        ]);
        if traces.len() + 1 != self.airs_without_poseidon::<NativeSC>().len() {
            return Err(DirectAirVaccVerifierErrorV19::AirTraceCount);
        }
        let end_state = exact_v19_checkpoint_at(record.transcript, record.vacc_end_tidx)?.state;
        Ok(DirectAirReducedSwirlRecordTraceV19 {
            traces,
            schedule,
            external_permutation_inputs: permutation_inputs,
            external_compression_inputs: compression_inputs,
            output_instance: record.verification.output_instance.clone(),
            output_instance_digest: output_digest.instance_digest,
            end_state,
        })
    }

    /// Compatibility wrapper for the obsolete one-proof path.  The record
    /// must use group-local `proof_index = 0`.
    pub fn generate_trace(
        &self,
        config: &NativeSC,
        record: &DirectAirVaccVerifierRecordV19<'_>,
    ) -> Result<DirectAirVaccVerifierTraceV19, DirectAirVaccVerifierErrorV19> {
        let batch = self.generate_traces(config, core::slice::from_ref(record))?;
        let output_instance = batch.output_instances.into_iter().next().ok_or(
            DirectAirVaccVerifierErrorV19::RecordShape("empty VACC batch"),
        )?;
        let output_instance_digest = batch.output_instance_digests[0];
        Ok(DirectAirVaccVerifierTraceV19 {
            traces: batch.traces,
            poseidon_permutation_inputs: batch.poseidon_permutation_inputs,
            poseidon_compression_inputs: batch.poseidon_compression_inputs,
            output_instance,
            output_instance_digest,
        })
    }

    /// Generate one physical verifier witness for many records sharing this
    /// numeric shape and prior mode.  `TranscriptAir` defines proof IDs by log
    /// order, so records must carry the dense group-local indices `0..n`.
    pub fn generate_traces(
        &self,
        config: &NativeSC,
        records: &[DirectAirVaccVerifierRecordV19<'_>],
    ) -> Result<DirectAirVaccVerifierBatchTraceV19, DirectAirVaccVerifierErrorV19> {
        if self.fresh_commitment_mode != DirectAirVaccFreshCommitmentModeV19::ScalarMerkle {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "scalar record supplied to CUDA shared-forest module",
            ));
        }
        self.generate_traces_for_shared_poseidon(config, records)?
            .with_local_poseidon_table(self)
    }

    /// Generate every verifier witness except the physical Poseidon table.
    /// The returned request packet is complete and is intended for one entry
    /// of [`NativeWarpTranscriptModule::multi_bus_poseidon_air`]. No Poseidon
    /// table is constructed along this path.
    pub fn generate_traces_for_shared_poseidon(
        &self,
        config: &NativeSC,
        records: &[DirectAirVaccVerifierRecordV19<'_>],
    ) -> Result<DirectAirVaccSharedPoseidonBatchTraceV19, DirectAirVaccVerifierErrorV19> {
        if self.fresh_commitment_mode != DirectAirVaccFreshCommitmentModeV19::ScalarMerkle {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "scalar record supplied to CUDA shared-forest module",
            ));
        }
        self.generate_trace_inputs_with_external_poseidon(config, records, Vec::new(), Vec::new())
    }

    /// Generate recursive verifier traces for the production Appendix-D lane:
    /// BabyBear fresh leaves and EF4 prior/output accumulators.
    pub fn generate_appendix_d_traces(
        &self,
        config: &NativeSC,
        records: &[DirectAirVaccVerifierRecordV19<'_, NativeAppendixDFreshOpeningVerification>],
    ) -> Result<DirectAirVaccVerifierBatchTraceV19, DirectAirVaccVerifierErrorV19> {
        if self.fresh_commitment_mode != DirectAirVaccFreshCommitmentModeV19::AppendixDBase {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "Appendix-D record supplied to non-Appendix-D module",
            ));
        }
        self.generate_appendix_d_traces_for_shared_poseidon(config, records)?
            .with_local_poseidon_table(self)
    }

    /// Appendix-D counterpart of [`Self::generate_traces_for_shared_poseidon`].
    pub fn generate_appendix_d_traces_for_shared_poseidon(
        &self,
        config: &NativeSC,
        records: &[DirectAirVaccVerifierRecordV19<'_, NativeAppendixDFreshOpeningVerification>],
    ) -> Result<DirectAirVaccSharedPoseidonBatchTraceV19, DirectAirVaccVerifierErrorV19> {
        if self.fresh_commitment_mode != DirectAirVaccFreshCommitmentModeV19::AppendixDBase {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "Appendix-D record supplied to non-Appendix-D module",
            ));
        }
        self.generate_trace_inputs_with_external_poseidon(config, records, Vec::new(), Vec::new())
    }

    /// Generate the standard VACC verifier traces when fresh authentication
    /// is provided by `CudaSharedForestVerifierModuleV19`.  The additional
    /// Poseidon requests are folded into this module's one shared permutation
    /// table; no second table or CPU authentication fallback is introduced.
    pub fn generate_cuda_shared_forest_traces(
        &self,
        config: &NativeSC,
        records: &[DirectAirVaccVerifierRecordV19<'_, ()>],
        forest_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        forest_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    ) -> Result<DirectAirVaccVerifierBatchTraceV19, DirectAirVaccVerifierErrorV19> {
        if self.fresh_commitment_mode != DirectAirVaccFreshCommitmentModeV19::CudaSharedForest {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "CUDA forest record supplied to scalar VACC module",
            ));
        }
        self.generate_cuda_shared_forest_traces_for_shared_poseidon(
            config,
            records,
            forest_permutation_inputs,
            forest_compression_inputs,
        )?
        .with_local_poseidon_table(self)
    }

    /// CUDA shared-forest counterpart of
    /// [`Self::generate_traces_for_shared_poseidon`]. Forest requests are
    /// included in the same owner packet and no local Poseidon table is built.
    pub fn generate_cuda_shared_forest_traces_for_shared_poseidon(
        &self,
        config: &NativeSC,
        records: &[DirectAirVaccVerifierRecordV19<'_, ()>],
        forest_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        forest_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    ) -> Result<DirectAirVaccSharedPoseidonBatchTraceV19, DirectAirVaccVerifierErrorV19> {
        if self.fresh_commitment_mode != DirectAirVaccFreshCommitmentModeV19::CudaSharedForest {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "CUDA forest record supplied to scalar VACC module",
            ));
        }
        self.generate_trace_inputs_with_external_poseidon(
            config,
            records,
            forest_permutation_inputs,
            forest_compression_inputs,
        )
    }

    fn generate_trace_inputs_with_external_poseidon<
        FreshVerification: DirectAirFreshAuthenticationV19,
    >(
        &self,
        config: &NativeSC,
        records: &[DirectAirVaccVerifierRecordV19<'_, FreshVerification>],
        mut additional_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        mut additional_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    ) -> Result<DirectAirVaccSharedPoseidonBatchTraceV19, DirectAirVaccVerifierErrorV19> {
        if records.is_empty() {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "empty VACC batch",
            ));
        }
        let mut generated = Vec::with_capacity(records.len());
        for (proof_idx, record) in records.iter().enumerate() {
            if record.producer.proof_index as usize != proof_idx {
                return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                    "non-canonical group-local proof index",
                ));
            }
            generated.push(self.generate_record_trace(config, record)?);
        }

        let resumed_logs = self
            .resume_from_start_checkpoint
            .then(|| {
                records
                    .iter()
                    .map(|record| {
                        exact_v19_resumable_suffix(
                            record.transcript,
                            record.producer.start_checkpoint.operation_index as usize,
                            record.producer.start_checkpoint.state,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?;
        let logs = match resumed_logs.as_ref() {
            Some(logs) => logs.iter().collect::<Vec<_>>(),
            None => records
                .iter()
                .map(|record| record.transcript)
                .collect::<Vec<_>>(),
        };
        let resumes = if self.resume_from_start_checkpoint {
            records
                .iter()
                .map(|record| {
                    Some((
                        record.producer.start_checkpoint.operation_index as usize,
                        record.producer.start_checkpoint.state,
                    ))
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let checkpoints = generated
            .iter()
            .map(|record| {
                Some(if self.resume_from_start_checkpoint {
                    [None, Some(record.schedule.end_tidx)]
                } else {
                    [
                        Some(record.schedule.start_tidx),
                        Some(record.schedule.end_tidx),
                    ]
                })
            })
            .collect::<Vec<_>>();
        let mut external_permutation_inputs = generated
            .iter()
            .flat_map(|record| record.external_permutation_inputs.iter().copied())
            .collect::<Vec<_>>();
        let mut external_compression_inputs = generated
            .iter()
            .flat_map(|record| record.external_compression_inputs.iter().copied())
            .collect::<Vec<_>>();
        external_permutation_inputs.append(&mut additional_permutation_inputs);
        external_compression_inputs.append(&mut additional_compression_inputs);
        let transcript = self
            .transcript
            .generate_trace_inputs_with_external_resumed_and_optional_checkpoints(
                &logs,
                &resumes,
                &checkpoints,
                external_permutation_inputs,
                external_compression_inputs,
                None,
            )
            .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
                "batched transcript trace generation",
            ))?;
        let poseidon_permutation_inputs = transcript.permutation_inputs;
        let poseidon_compression_inputs = transcript.compression_inputs;

        let matrix_count = generated[0].traces.len();
        if generated
            .iter()
            .any(|record| record.traces.len() != matrix_count)
        {
            return Err(DirectAirVaccVerifierErrorV19::AirTraceCount);
        }
        let claim_layout_index = 4;
        let slot_layout_index = 5;
        let (prefix_catalog, prefix_stream) =
            self.generate_prefix_compaction_traces(records, &generated)?;
        let mut traces = Vec::with_capacity(matrix_count + 3);
        traces.push(transcript.trace);
        traces.push(prefix_catalog);
        traces.push(prefix_stream);
        for matrix_index in 0..matrix_count {
            let first = &generated[0].traces[matrix_index];
            let merged = if matrix_index == claim_layout_index {
                scale_v19_claim_layout_lookups(first.clone(), records.len())
            } else if matrix_index == slot_layout_index {
                scale_v19_slot_layout_lookups(first.clone(), records.len())
            } else if matrix_index + 1 == matrix_count {
                // `ExpBitsLenAir` is the final verifier AIR.  Its inactive
                // rows are not the all-zero row: the reverse running-product
                // recurrence requires both multiplicative columns to be one.
                // Generic shape batching used to erase that neutral tail,
                // which only surfaced when the merged request count was not
                // already a power of two.
                merge_v19_exp_bits_rows(generated.iter().map(|record| &record.traces[matrix_index]))
                    .ok_or(DirectAirVaccVerifierErrorV19::AirTraceCount)?
            } else {
                merge_v19_active_row_refs(
                    generated.iter().map(|record| &record.traces[matrix_index]),
                    first.width(),
                )
                .ok_or(DirectAirVaccVerifierErrorV19::AirTraceCount)?
            };
            traces.push(merged);
        }
        if traces.len() != self.airs_without_poseidon::<NativeSC>().len() {
            return Err(DirectAirVaccVerifierErrorV19::AirTraceCount);
        }

        Ok(DirectAirVaccSharedPoseidonBatchTraceV19 {
            traces,
            poseidon_permutation_inputs,
            poseidon_compression_inputs,
            output_instances: generated
                .iter()
                .map(|record| record.output_instance.clone())
                .collect(),
            output_instance_digests: generated
                .iter()
                .map(|record| record.output_instance_digest)
                .collect(),
        })
    }

    fn generate_prefix_compaction_traces<FreshVerification: DirectAirFreshAuthenticationV19>(
        &self,
        records: &[DirectAirVaccVerifierRecordV19<'_, FreshVerification>],
        generated: &[DirectAirVaccRecordTraceV19],
    ) -> Result<(RowMajorMatrix<F>, RowMajorMatrix<F>), DirectAirVaccVerifierErrorV19> {
        if records.len() != generated.len() || records.is_empty() {
            return Err(DirectAirVaccVerifierErrorV19::AirTraceCount);
        }

        let catalog_width = NativeStandardVaccPrefixCatalogColsV19::<F>::width();
        let catalog_height = self.prefix_events.len().next_power_of_two().max(2);
        let mut catalog_values = F::zero_vec(catalog_width * catalog_height);
        for (event_index, event) in self.prefix_events.iter().enumerate() {
            let multiplicity = records
                .iter()
                .filter(|record| record.producer.relation_digest == event.relation_digest)
                .count();
            let cols: &mut NativeStandardVaccPrefixCatalogColsV19<F> = catalog_values
                [event_index * catalog_width..(event_index + 1) * catalog_width]
                .borrow_mut();
            cols.multiplicity = F::from_usize(multiplicity);
        }

        let stream_width = NativeStandardVaccPrefixStreamColsV19::<F>::width();
        let stream_len = records
            .iter()
            .map(|record| {
                self.prefix_events
                    .iter()
                    .filter(|event| event.relation_digest == record.producer.relation_digest)
                    .count()
            })
            .sum::<usize>();
        if stream_len == 0 {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "unknown VACC prefix relation",
            ));
        }
        let stream_height = stream_len.next_power_of_two().max(2);
        let mut stream_values = F::zero_vec(stream_width * stream_height);
        let mut stream_row = 0usize;
        for (proof_idx, (record, generated_record)) in
            records.iter().zip(generated.iter()).enumerate()
        {
            let events = self
                .prefix_events
                .iter()
                .filter(|event| event.relation_digest == record.producer.relation_digest)
                .collect::<Vec<_>>();
            if events.is_empty() {
                return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                    "unknown VACC prefix relation",
                ));
            }
            let step = match (self.include_prior, self.warp_step_mode) {
                (false, DirectAirVaccWarpStepModeV19::LegacySegmentIndex) => 0,
                (false, DirectAirVaccWarpStepModeV19::FixedHLeafV4) => {
                    return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                        "seeded fixed HLeaf without prior",
                    ));
                }
                (true, DirectAirVaccWarpStepModeV19::LegacySegmentIndex) => {
                    u64::from(record.producer.segment_index)
                }
                (true, DirectAirVaccWarpStepModeV19::FixedHLeafV4) => {
                    u64::from(FIXED_HLEAF_CONTINUATION_WARP_STEP_V4)
                }
            };
            let step_bytes = step.to_le_bytes();
            let mut step_lo_sum = 0u32;
            let mut step_hi_sum = 0u32;
            for event in events {
                let row =
                    &mut stream_values[stream_row * stream_width..(stream_row + 1) * stream_width];
                let cols: &mut NativeStandardVaccPrefixStreamColsV19<F> = row.borrow_mut();
                cols.active = F::ONE;
                cols.is_first = F::from_bool(event.ordinal == 0);
                cols.is_last = F::from_bool(event.is_last);
                cols.proof_idx = F::from_usize(proof_idx);
                cols.start_tidx = F::from_usize(generated_record.schedule.start_tidx);
                cols.start_nonzero = F::from_bool(generated_record.schedule.start_tidx != 0);
                cols.start_inverse = if generated_record.schedule.start_tidx == 0 {
                    F::ZERO
                } else {
                    cols.start_tidx.inverse()
                };
                cols.segment_index_lo = F::from_u32(record.producer.segment_index & 0xffff);
                cols.segment_index_hi = F::from_u32(record.producer.segment_index >> 16);
                cols.has_prior = F::from_bool(self.include_prior);
                cols.relation_digest = event.relation_digest;
                cols.ordinal = F::from_u32(event.ordinal);
                cols.event_count = F::from_u32(event.event_count);
                cols.catalog_value = F::from_u8(event.catalog_value);
                cols.is_step = F::from_bool(event.step_index.is_some());
                let event_value = if let Some(step_index) = event.step_index {
                    cols.step_flags[step_index as usize] = F::ONE;
                    let value = step_bytes[step_index as usize];
                    for bit in 0..8 {
                        cols.event_bits[bit] = F::from_bool(((value >> bit) & 1) == 1);
                    }
                    match step_index {
                        0 => step_lo_sum += u32::from(value),
                        1 => step_lo_sum += u32::from(value) << 8,
                        2 => step_hi_sum += u32::from(value),
                        3 => step_hi_sum += u32::from(value) << 8,
                        _ => {}
                    }
                    value
                } else {
                    event.catalog_value
                };
                cols.event_value = F::from_u8(event_value);
                cols.step_lo_sum = F::from_u32(step_lo_sum);
                cols.step_hi_sum = F::from_u32(step_hi_sum);
                stream_row += 1;
            }
        }
        if stream_row != stream_len {
            return Err(DirectAirVaccVerifierErrorV19::AirTraceCount);
        }
        Ok((
            RowMajorMatrix::new(catalog_values, catalog_width),
            RowMajorMatrix::new(stream_values, stream_width),
        ))
    }

    fn generate_record_trace<FreshVerification: DirectAirFreshAuthenticationV19>(
        &self,
        config: &NativeSC,
        record: &DirectAirVaccVerifierRecordV19<'_, FreshVerification>,
    ) -> Result<DirectAirVaccRecordTraceV19, DirectAirVaccVerifierErrorV19> {
        let proof_idx = record.producer.proof_index as usize;
        let schedule = DirectAirVaccTranscriptScheduleV19::from_record(self, record)?;
        self.validate_record(record)?;
        let cursor = generate_direct_air_vacc_cursor_trace_v19(self, record, &schedule)?;
        let algebra =
            generate_direct_air_vacc_algebra_traces_v19(self, proof_idx, record, &schedule)?;
        let fresh_authentication = record
            .verification
            .fresh_authentication
            .first()
            .and_then(DirectAirFreshAuthenticationV19::scalar_merkle);
        let appendix_d_authentication = record
            .verification
            .fresh_authentication
            .first()
            .and_then(DirectAirFreshAuthenticationV19::appendix_d_base);
        let fresh_projection = if self.fresh_commitment_mode.uses_scalar_merkle() {
            let authentication = fresh_authentication.ok_or(
                DirectAirVaccVerifierErrorV19::Merkle("missing scalar fresh authentication"),
            )?;
            Some(
                generate_native_standard_codeword_projection_trace(
                    config.hasher(),
                    proof_idx,
                    authentication,
                    self.profile.log_codeword_len,
                    self.profile.rows_per_query,
                    self.trees.fresh_rows,
                    u32::try_from(self.trees.fresh_outer)
                        .map_err(|_| DirectAirVaccVerifierErrorV19::Merkle("fresh tree id"))?,
                    0,
                    1 + usize::from(self.include_prior) * 3,
                    None,
                )
                .ok_or(DirectAirVaccVerifierErrorV19::Merkle(
                    "fresh codeword projection",
                ))?,
            )
        } else if self.fresh_commitment_mode == DirectAirVaccFreshCommitmentModeV19::AppendixDBase {
            let authentication =
                appendix_d_authentication.ok_or(DirectAirVaccVerifierErrorV19::Merkle(
                    "missing Appendix-D base fresh authentication",
                ))?;
            Some(
                generate_native_appendix_d_codeword_projection_trace(
                    config.hasher(),
                    proof_idx,
                    authentication,
                    self.profile.log_codeword_len,
                    self.profile.rows_per_query,
                    self.trees.fresh_rows,
                    u32::try_from(self.trees.fresh_outer)
                        .map_err(|_| DirectAirVaccVerifierErrorV19::Merkle("fresh tree id"))?,
                    0,
                    1 + usize::from(self.include_prior) * 3,
                    None,
                )
                .ok_or(DirectAirVaccVerifierErrorV19::Merkle(
                    "Appendix-D fresh codeword projection",
                ))?,
            )
        } else {
            if fresh_authentication.is_some() || appendix_d_authentication.is_some() {
                return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                    "standalone fresh authentication in CUDA forest mode",
                ));
            }
            None
        };
        let prior_projection =
            if let Some(authentication) = record.verification.prior_authentication.as_ref() {
                Some(
                    generate_native_standard_codeword_projection_trace(
                        config.hasher(),
                        proof_idx,
                        authentication,
                        self.profile.log_codeword_len,
                        self.profile.rows_per_query,
                        self.trees.prior_rows,
                        u32::try_from(self.trees.prior_outer)
                            .map_err(|_| DirectAirVaccVerifierErrorV19::Merkle("prior tree id"))?,
                        1,
                        4,
                        None,
                    )
                    .ok_or(DirectAirVaccVerifierErrorV19::Merkle(
                        "prior codeword projection",
                    ))?,
                )
            } else {
                None
            };

        let mut leaf_hash_matrices = Vec::new();
        let mut merkle_matrices = Vec::new();
        let mut adapter_matrices = Vec::new();
        let mut permutation_inputs = Vec::new();
        let mut compression_inputs = Vec::new();
        if let (Some(projection), Some(authentication)) =
            (fresh_projection.as_ref(), fresh_authentication)
        {
            self.collect_projection_authentication(
                config,
                proof_idx,
                projection,
                authentication,
                self.trees.fresh_outer,
                &mut leaf_hash_matrices,
                &mut merkle_matrices,
                &mut adapter_matrices,
                &mut permutation_inputs,
                &mut compression_inputs,
            )?;
        }
        if let (Some(projection), Some(authentication)) =
            (fresh_projection.as_ref(), appendix_d_authentication)
        {
            self.collect_projection_authentication_record(
                config,
                proof_idx,
                projection,
                &authentication.inner.multiproof,
                self.trees.fresh_outer,
                &mut leaf_hash_matrices,
                &mut merkle_matrices,
                &mut adapter_matrices,
                &mut permutation_inputs,
                &mut compression_inputs,
            )?;
        }
        if let (Some(projection), Some(authentication)) = (
            prior_projection.as_ref(),
            record.verification.prior_authentication.as_ref(),
        ) {
            self.collect_projection_authentication(
                config,
                proof_idx,
                projection,
                authentication,
                self.trees.prior_outer,
                &mut leaf_hash_matrices,
                &mut merkle_matrices,
                &mut adapter_matrices,
                &mut permutation_inputs,
                &mut compression_inputs,
            )?;
        }
        let scalar_authentication_traces = if self
            .fresh_commitment_mode
            .uses_merkle_authentication()
            || self.include_prior
        {
            Some((
                merge_v19_active_rows(&leaf_hash_matrices, NativeLeafHashCols::<F>::width())
                    .ok_or(DirectAirVaccVerifierErrorV19::Merkle("leaf hash merge"))?,
                merge_v19_active_rows(&merkle_matrices, NativeMerkleCompressionCols::<F>::width())
                    .ok_or(DirectAirVaccVerifierErrorV19::Merkle("multiproof merge"))?,
                merge_v19_active_rows(&adapter_matrices, NativeMerkleLeafAdapterCols::<F>::width())
                    .ok_or(DirectAirVaccVerifierErrorV19::Merkle("leaf adapter merge"))?,
            ))
        } else {
            None
        };

        let prior_digest = record
            .prior
            .map(|prior| {
                generate_native_accumulator_digest_traces(
                    proof_idx,
                    prior,
                    &self.private_accumulator,
                )
                .ok_or(DirectAirVaccVerifierErrorV19::RecordShape(
                    "prior accumulator digest",
                ))
            })
            .transpose()?;
        let output_digest = generate_native_accumulator_digest_traces(
            proof_idx,
            &record.verification.output_instance,
            &self.private_accumulator,
        )
        .ok_or(DirectAirVaccVerifierErrorV19::RecordShape(
            "output accumulator digest",
        ))?;
        append_v19_accumulator_poseidon(
            prior_digest.as_ref(),
            &mut permutation_inputs,
            &mut compression_inputs,
        );
        append_v19_accumulator_poseidon(
            Some(&output_digest),
            &mut permutation_inputs,
            &mut compression_inputs,
        );
        if prior_digest.as_ref().is_some_and(|prior| {
            prior.instance_digest != record.producer.previous_accumulator_digest
        }) || output_digest.instance_digest != record.producer.next_accumulator_digest
        {
            return Err(DirectAirVaccVerifierErrorV19::PriorChain);
        }

        let remainder = (!self.resume_from_start_checkpoint)
            .then(|| {
                generate_native_transcript_prefix_remainder_trace(
                    record.transcript,
                    proof_idx,
                    schedule.start_tidx,
                    None,
                )
                .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
                    "transcript remainder",
                ))
            })
            .transpose()?;
        let exp_bits =
            generate_direct_v19_exp_bits_trace(record, &schedule, self.profile.log_codeword_len)?;
        let statement = generate_direct_v19_statement_trace(record.producer)?;
        let end = generate_native_standard_vacc_end_trace(
            proof_idx,
            schedule.end_tag_tidx,
            schedule.end_tidx,
            schedule.discarded_end_sample,
        );
        let fresh_root = self
            .fresh_commitment_mode
            .uses_merkle_authentication()
            .then(|| {
                generate_native_standard_vacc_commitment_root_trace(
                    proof_idx,
                    schedule.fresh_root_tidx,
                    record.producer.fresh_root,
                )
            });
        let prior_root = schedule.prior.as_ref().map(|prior| {
            generate_native_standard_vacc_commitment_root_trace(
                proof_idx,
                prior.root_tidx,
                record.producer.prior_root,
            )
        });
        let output_root = generate_native_standard_vacc_commitment_root_trace(
            proof_idx,
            schedule.output_root_tidx,
            record.verification.output_instance.rt,
        );

        let mut traces = vec![
            cursor,
            end,
            statement,
            algebra.claim_values,
            algebra.claim_layout,
            algebra.slot_layout,
        ];
        if let Some(fresh_projection) = fresh_projection {
            traces.push(fresh_projection.matrix);
        }
        if let Some(prior) = prior_projection {
            traces.push(prior.matrix);
        }
        if let Some(fresh_root) = fresh_root {
            traces.push(fresh_root);
        }
        if let Some(prior_root) = prior_root {
            traces.push(prior_root);
        }
        if let Some((leaf_hash, merkle, leaf_adapter)) = scalar_authentication_traces {
            traces.extend([leaf_hash, merkle, leaf_adapter]);
        }
        let vector_shard_count = self.vector_coordinate_shard_count_v19();
        let vector_shard = if vector_shard_count == 1 {
            0
        } else if proof_idx < vector_shard_count {
            proof_idx
        } else {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "fixed HLeaf VACC proof index exceeds vector-table slots",
            ));
        };
        let vector_width = algebra.vector.width();
        let mut vector_traces = (0..vector_shard_count)
            .map(|_| RowMajorMatrix::new(F::zero_vec(vector_width * 2), vector_width))
            .collect::<Vec<_>>();
        vector_traces[vector_shard] = algebra.vector;
        traces.extend(vector_traces);
        traces.extend([
            algebra.selector_eq,
            algebra.xi_eq,
            algebra.opening_eq,
            algebra.omega,
            algebra.twin_sigma,
            algebra.sumcheck,
            algebra.twin_fold,
            algebra.twin_final,
            algebra.initial_point,
            algebra.initial_target,
            algebra.opening_padding,
            algebra.ood_claims,
            algebra.shift_schedule,
            algebra.shift_merge,
            algebra.batching_sigma,
            algebra.batching_final,
            algebra.batching_alpha,
            output_root,
        ]);
        if let Some(prior) = prior_digest {
            traces.extend([prior.values, prior.hash, prior.root]);
        }
        traces.extend([output_digest.values, output_digest.hash, output_digest.root]);
        if let Some(remainder) = remainder {
            traces.push(remainder);
        }
        traces.push(exp_bits);
        // TranscriptAir, Poseidon2Air, prefix catalog and prefix stream are
        // generated once for the whole group batch.
        let expected = self.airs::<NativeSC>().len() - 4;
        if traces.len() != expected {
            return Err(DirectAirVaccVerifierErrorV19::AirTraceCount);
        }
        Ok(DirectAirVaccRecordTraceV19 {
            traces,
            schedule,
            external_permutation_inputs: permutation_inputs,
            external_compression_inputs: compression_inputs,
            output_instance: record.verification.output_instance.clone(),
            output_instance_digest: output_digest.instance_digest,
        })
    }

    fn validate_record<FreshVerification: DirectAirFreshAuthenticationV19>(
        &self,
        record: &DirectAirVaccVerifierRecordV19<'_, FreshVerification>,
    ) -> Result<(), DirectAirVaccVerifierErrorV19> {
        let p = record.producer;
        let v = record.verification;
        if v.fresh_authentication.len() != 1
            || !self
                .relations
                .iter()
                .any(|relation| relation.relation_digest == p.relation_digest)
            || p.next_root != v.output_instance.rt
            || v.batching.sigma_2 != ext_from_v19_array(&p.authenticated_batching_claim)?
            || ext_from_v19_array(&p.fresh_mu)? != ext_from_v19_array(&p.opening_value)?
            || ext_from_v19_array(&p.fresh_eta)? != EF::ZERO
            || p.opening_point.len() != self.profile.log_message_len
            || p.fresh_alpha[..self.profile.log_message_len] != p.opening_point
            || p.fresh_alpha[self.profile.log_message_len..]
                .iter()
                .any(|value| *value != [F::ZERO; D_EF])
            || p.start_checkpoint.operation_index as usize >= record.transcript.len()
            || p.end_checkpoint.operation_index as usize > record.transcript.len()
            || (self.warp_step_mode == DirectAirVaccWarpStepModeV19::FixedHLeafV4
                && (!self.include_prior || p.update_index != FIXED_HLEAF_CONTINUATION_WARP_STEP_V4))
        {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "certified direct-AIR VACC statement",
            ));
        }
        match self.fresh_commitment_mode {
            DirectAirVaccFreshCommitmentModeV19::ScalarMerkle => {
                let authentication = v.fresh_authentication[0].scalar_merkle().ok_or(
                    DirectAirVaccVerifierErrorV19::RecordShape(
                        "missing scalar fresh authentication",
                    ),
                )?;
                if p.fresh_root != authentication.multiproof.expected_root {
                    return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                        "scalar fresh root",
                    ));
                }
            }
            DirectAirVaccFreshCommitmentModeV19::AppendixDBase => {
                let authentication = v.fresh_authentication[0].appendix_d_base().ok_or(
                    DirectAirVaccVerifierErrorV19::RecordShape(
                        "missing Appendix-D base fresh authentication",
                    ),
                )?;
                if p.fresh_root != authentication.commitment
                    || p.fresh_root != authentication.inner.multiproof.expected_root
                    || authentication.log_codeword_len as usize != self.profile.log_codeword_len
                    || !authentication.validates_systematic_embedding()
                    || authentication.indices != authentication.inner.flat_indices
                    || authentication.base_values != authentication.inner.values
                {
                    return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                        "Appendix-D fresh authentication",
                    ));
                }
            }
            DirectAirVaccFreshCommitmentModeV19::FiniteStackedBase => {
                return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                    "exact-finite record supplied to legacy generator",
                ));
            }
            DirectAirVaccFreshCommitmentModeV19::CudaSharedForest => {
                if v.fresh_authentication[0].scalar_merkle().is_some()
                    || v.fresh_authentication[0].appendix_d_base().is_some()
                {
                    return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                        "unexpected scalar fresh authentication",
                    ));
                }
            }
        }
        if let Some(prior) = record.prior {
            if prior.rt != p.prior_root
                || prior.rt
                    != v.prior_authentication
                        .as_ref()
                        .ok_or(DirectAirVaccVerifierErrorV19::PriorChain)?
                        .multiproof
                        .expected_root
            {
                return Err(DirectAirVaccVerifierErrorV19::PriorChain);
            }
        } else if p.prior_root != [F::ZERO; DIGEST_SIZE]
            || p.previous_accumulator_digest != [F::ZERO; DIGEST_SIZE]
        {
            return Err(DirectAirVaccVerifierErrorV19::PriorChain);
        }
        Ok(())
    }

    fn validate_exact_finite_record(
        &self,
        record: &DirectAirExactFiniteVaccVerifierRecordV19<'_>,
    ) -> Result<Digest, DirectAirVaccVerifierErrorV19> {
        let exact = self.exact_config()?;
        let call = exact.call();
        let proof = record.proof;
        let verification = record.verification;
        let expected_kinds = (0..call.input_arity)
            .map(|source| {
                if source < call.fresh_count {
                    WarpStepInputKind::Fresh
                } else {
                    WarpStepInputKind::PriorAccumulator
                }
            })
            .collect::<Vec<_>>();
        if proof.input_kinds != expected_kinds
            || proof.fresh_claims.len() != call.fresh_count
            || proof.output_instance != verification.output_instance
            || record.prior.is_some() != (call.prior_count == 1)
            || verification.prior_authentication.is_some() != (call.prior_count == 1)
            || proof.openings.fresh_opening_proofs.len() != 1
            || proof.openings.acc_opening_proofs.len() != call.prior_count
            || proof.openings.acc_shift_answers.len() != call.prior_count
            || record.start_checkpoint.operation_index >= record.end_checkpoint.operation_index
            || record.end_checkpoint.operation_index as usize > record.transcript.len()
        {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "exact-finite proof envelope",
            ));
        }
        if exact.call_index == 0 {
            let schedule_end =
                native_exact_finite_vacc_schedule_prefix_elements(&exact.transcript_profile)
                    .map_err(|_| DirectAirVaccVerifierErrorV19::InvalidProfile)?
                    .len()
                    * D_EF;
            if record.start_checkpoint.operation_index as usize != schedule_end
                || record.start_checkpoint.sample_count != 0
                || record.start_checkpoint.state != [F::ZERO; POSEIDON2_WIDTH]
            {
                return Err(DirectAirVaccVerifierErrorV19::Transcript(
                    "exact-finite schedule checkpoint",
                ));
            }
        } else {
            validate_exact_v19_checkpoint(record.transcript, record.start_checkpoint)?;
        }
        validate_exact_v19_checkpoint(record.transcript, record.end_checkpoint)?;

        for claim in &proof.fresh_claims {
            if claim.alpha.len() != self.profile.log_codeword_len
                || claim.beta.len() != self.profile.beta_len
                || claim.eta != EF::ZERO
            {
                return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                    "exact-finite fresh claim",
                ));
            }
        }
        let descriptors = proof
            .fresh_claims
            .iter()
            .map(|claim| &claim.commitment)
            .collect::<Vec<_>>();
        let first = descriptors
            .first()
            .ok_or(DirectAirVaccVerifierErrorV19::RecordShape(
                "empty stacked commitment",
            ))?;
        let fresh_root = first.root;
        if descriptors.iter().enumerate().any(|(source, descriptor)| {
            descriptor.root != fresh_root
                || descriptor.source_ordinal as usize != source
                || descriptor.fresh_count as usize != call.fresh_count
                || descriptor.log_message_len as usize != self.profile.log_message_len
                || descriptor.log_codeword_len as usize != self.profile.log_codeword_len
                || descriptor.rows_per_leaf as usize != self.profile.rows_per_query
        }) {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "stacked commitment descriptors",
            ));
        }

        let shared = verification
            .fresh_authentication
            .first()
            .ok_or(DirectAirVaccVerifierErrorV19::Merkle(
                "missing stacked fresh verification",
            ))?
            .shared
            .as_ref();
        if verification.fresh_authentication.len() != call.fresh_count
            || !shared.recorded
            || shared.multiproof.is_none()
            || shared.values_by_source.len() != call.fresh_count
            || verification.fresh_authentication.iter().enumerate().any(
                |(source, authentication)| {
                    authentication.source_ordinal as usize != source
                        || authentication.shared.as_ref() != shared
                        || authentication.values != shared.values_by_source[source]
                },
            )
            || shared
                .values_by_source
                .iter()
                .any(|values| values.len() != self.profile.num_shift_queries)
            || shared
                .multiproof
                .as_ref()
                .is_none_or(|multiproof| multiproof.expected_root != fresh_root)
        {
            return Err(DirectAirVaccVerifierErrorV19::Merkle(
                "stacked fresh authentication",
            ));
        }
        for (shift, verification_shift) in verification.shifts.iter().enumerate() {
            if verification_shift.fresh_answers.len() != call.fresh_count
                || verification_shift
                    .fresh_answers
                    .iter()
                    .enumerate()
                    .any(|(source, answer)| shared.values_by_source[source][shift] != *answer)
                || verification_shift.prior_answer.is_some() != (call.prior_count == 1)
            {
                return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                    "exact-finite authenticated shifts",
                ));
            }
        }
        if let Some(prior) = record.prior {
            let authentication = verification
                .prior_authentication
                .as_ref()
                .ok_or(DirectAirVaccVerifierErrorV19::PriorChain)?;
            if prior.alpha.len() != self.profile.log_codeword_len
                || prior.beta.len() != self.profile.beta_len
                || authentication.multiproof.expected_root != prior.rt
            {
                return Err(DirectAirVaccVerifierErrorV19::PriorChain);
            }
        }
        Ok(fresh_root)
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_projection_authentication(
        &self,
        config: &NativeSC,
        proof_idx: usize,
        projection: &NativeAccumulatorProjectionTrace,
        authentication: &MerkleBatchOpeningVerification<EF, Digest>,
        outer_tree_id: usize,
        leaf_hash_matrices: &mut Vec<RowMajorMatrix<F>>,
        merkle_matrices: &mut Vec<RowMajorMatrix<F>>,
        adapter_matrices: &mut Vec<RowMajorMatrix<F>>,
        permutation_inputs: &mut Vec<[F; POSEIDON2_WIDTH]>,
        compression_inputs: &mut Vec<[F; POSEIDON2_WIDTH]>,
    ) -> Result<(), DirectAirVaccVerifierErrorV19> {
        self.collect_projection_authentication_record(
            config,
            proof_idx,
            projection,
            &authentication.multiproof,
            outer_tree_id,
            leaf_hash_matrices,
            merkle_matrices,
            adapter_matrices,
            permutation_inputs,
            compression_inputs,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn collect_projection_authentication_record(
        &self,
        config: &NativeSC,
        proof_idx: usize,
        projection: &NativeAccumulatorProjectionTrace,
        outer_multiproof: &BinaryMerkleMultiproofRecord<Digest>,
        outer_tree_id: usize,
        leaf_hash_matrices: &mut Vec<RowMajorMatrix<F>>,
        merkle_matrices: &mut Vec<RowMajorMatrix<F>>,
        adapter_matrices: &mut Vec<RowMajorMatrix<F>>,
        permutation_inputs: &mut Vec<[F; POSEIDON2_WIDTH]>,
        compression_inputs: &mut Vec<[F; POSEIDON2_WIDTH]>,
    ) -> Result<(), DirectAirVaccVerifierErrorV19> {
        let leaf = generate_native_leaf_hash_trace(&projection.leaf_hash_inputs(), None)
            .ok_or(DirectAirVaccVerifierErrorV19::Merkle("leaf hash"))?;
        permutation_inputs.extend(leaf.permutation_inputs.iter().copied());
        leaf_hash_matrices.push(leaf.matrix);
        let mut records = projection
            .inner_merkle
            .iter()
            .map(|(tree, proof)| (*tree, proof))
            .collect::<Vec<_>>();
        records.push((
            u32::try_from(outer_tree_id)
                .map_err(|_| DirectAirVaccVerifierErrorV19::Merkle("outer tree id"))?,
            outer_multiproof,
        ));
        compression_inputs.extend(v19_merkle_compression_inputs(&records));
        if records
            .iter()
            .any(|(_, proof)| !proof.compressions.is_empty())
        {
            merkle_matrices.push(
                generate_native_merkle_multiproof_trace(proof_idx, &records, None)
                    .ok_or(DirectAirVaccVerifierErrorV19::Merkle("Merkle multiproof"))?,
            );
        }
        if !projection.leaf_adapters.is_empty() {
            adapter_matrices.push(
                generate_native_merkle_leaf_adapter_trace(
                    &projection.leaf_adapters,
                    self.profile.rows_per_query.ilog2() as usize,
                    None,
                )
                .ok_or(DirectAirVaccVerifierErrorV19::Merkle("leaf adapter"))?,
            );
        }
        let _ = config;
        Ok(())
    }
}

fn generate_direct_v19_statement_trace(
    record: &DirectAirVaccProducerRecordV19,
) -> Result<RowMajorMatrix<F>, DirectAirVaccVerifierErrorV19> {
    let width = DirectAirVaccStatementColsV19::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut DirectAirVaccStatementColsV19<F> = values.as_mut_slice().borrow_mut();
    let inner = &mut cols.inner;
    inner.active = F::ONE;
    set_v19_u32(
        record.proof_index,
        &mut inner.proof_index_lo,
        &mut inner.proof_index_hi,
    );
    set_v19_u32(
        record.segment_index,
        &mut inner.segment_index_lo,
        &mut inner.segment_index_hi,
    );
    set_v19_u32(
        record.update_index,
        &mut inner.update_index_lo,
        &mut inner.update_index_hi,
    );
    inner.shard_ordinal = F::from_u16(record.shard_ordinal);
    inner.has_prior = F::from_bool(record.has_prior);
    inner.key_digest = record.key_digest;
    inner.relation_digest = record.relation_digest;
    inner.source_forest_root = record.source_forest_root;
    inner.segment_openings_digest = record.segment_openings_digest;
    inner.prior_root = record.prior_root;
    inner.fresh_root = record.fresh_root;
    inner.next_root = record.next_root;
    inner.previous_accumulator_digest = record.previous_accumulator_digest;
    inner.next_accumulator_digest = record.next_accumulator_digest;
    inner.previous_checkpoint_digest = record.previous_checkpoint_digest;
    inner.next_checkpoint_digest = record.next_checkpoint_digest;
    for (target, source) in inner.fresh_alpha.iter_mut().zip(&record.fresh_alpha) {
        *target = *source;
    }
    inner.fresh_mu = record.fresh_mu;
    for (target, source) in inner.fresh_beta.iter_mut().zip(&record.fresh_beta) {
        *target = *source;
    }
    inner.fresh_eta = record.fresh_eta;
    set_v19_u32(
        record.start_checkpoint.operation_index,
        &mut inner.start_tidx_lo,
        &mut inner.start_tidx_hi,
    );
    set_v19_u32(
        record.end_checkpoint.operation_index,
        &mut inner.end_tidx_lo,
        &mut inner.end_tidx_hi,
    );
    Ok(RowMajorMatrix::new(values, width))
}

fn set_v19_u32(value: u32, lo: &mut F, hi: &mut F) {
    *lo = F::from_u32(value & 0xffff);
    *hi = F::from_u32(value >> 16);
}

fn generate_direct_v19_exp_bits_trace<FreshVerification: DirectAirFreshAuthenticationV19>(
    record: &DirectAirVaccVerifierRecordV19<'_, FreshVerification>,
    schedule: &DirectAirVaccTranscriptScheduleV19,
    log_codeword_len: usize,
) -> Result<RowMajorMatrix<F>, DirectAirVaccVerifierErrorV19> {
    generate_direct_v19_exp_bits_trace_from_log(record.transcript, schedule, log_codeword_len)
}

fn generate_direct_v19_exp_bits_trace_from_log(
    transcript: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    schedule: &DirectAirVaccTranscriptScheduleV19,
    log_codeword_len: usize,
) -> Result<RowMajorMatrix<F>, DirectAirVaccVerifierErrorV19> {
    let generator = ExpBitsLenCpuTraceGenerator::default();
    for sample in &schedule.shifts {
        let accepted = *transcript
            .values()
            .get(sample.operation_range.end - 1)
            .ok_or(DirectAirVaccVerifierErrorV19::Transcript(
                "shift exponent request",
            ))?;
        generator.add_requests_with_shift([(F::ONE, accepted, 0, log_codeword_len, 1)]);
    }
    generator
        .generate_trace_row_major(None)
        .ok_or(DirectAirVaccVerifierErrorV19::Algebra(
            "shift exponent table",
        ))
}

fn append_v19_accumulator_poseidon(
    traces: Option<&NativeAccumulatorDigestTraces>,
    permutations: &mut Vec<[F; POSEIDON2_WIDTH]>,
    compressions: &mut Vec<[F; POSEIDON2_WIDTH]>,
) {
    if let Some(traces) = traces {
        permutations.extend(traces.poseidon2_permute_inputs.iter().copied());
        compressions.extend(traces.poseidon2_compress_inputs.iter().copied());
    }
}

fn v19_merkle_compression_inputs(
    records: &[(u32, &BinaryMerkleMultiproofRecord<Digest>)],
) -> Vec<[F; POSEIDON2_WIDTH]> {
    records
        .iter()
        .flat_map(|(_, record)| {
            record.compressions.iter().map(|compression| {
                core::array::from_fn(|index| {
                    if index < DIGEST_SIZE {
                        compression.left[index]
                    } else {
                        compression.right[index - DIGEST_SIZE]
                    }
                })
            })
        })
        .collect()
}

fn merge_v19_active_rows(
    matrices: &[RowMajorMatrix<F>],
    width: usize,
) -> Option<RowMajorMatrix<F>> {
    merge_v19_active_row_refs(matrices.iter(), width)
}

fn merge_v19_active_row_refs<'a>(
    matrices: impl IntoIterator<Item = &'a RowMajorMatrix<F>>,
    width: usize,
) -> Option<RowMajorMatrix<F>> {
    let matrices = matrices.into_iter().collect::<Vec<_>>();
    if width == 0 || matrices.is_empty() || matrices.iter().any(|matrix| matrix.width() != width) {
        return None;
    }
    let active = matrices
        .iter()
        .flat_map(|matrix| matrix.values.chunks_exact(width))
        .filter(|row| row[0] != F::ZERO)
        .count();
    let height = active.max(1).next_power_of_two();
    let mut values = Vec::with_capacity(height * width);
    for matrix in matrices {
        for row in matrix.values.chunks_exact(width) {
            if row[0] != F::ZERO {
                values.extend_from_slice(row);
            }
        }
    }
    values.resize(height * width, F::ZERO);
    Some(RowMajorMatrix::new(values, width))
}

fn merge_v19_exp_bits_rows<'a>(
    matrices: impl IntoIterator<Item = &'a RowMajorMatrix<F>>,
) -> Option<RowMajorMatrix<F>> {
    let width = ExpBitsLenCols::<F>::width();
    let mut matrix = merge_v19_active_row_refs(matrices, width)?;
    for row in matrix.values.chunks_exact_mut(width) {
        let cols: &mut ExpBitsLenCols<F> = row.borrow_mut();
        if cols.is_valid == F::ZERO {
            cols.result = F::ONE;
            cols.result_multiplier = F::ONE;
        }
    }
    Some(matrix)
}

fn scale_v19_claim_layout_lookups(
    mut matrix: RowMajorMatrix<F>,
    factor: usize,
) -> RowMajorMatrix<F> {
    let width = NativeClaimLayoutCols::<F>::width();
    let factor = F::from_usize(factor);
    for row in matrix.values.chunks_exact_mut(width) {
        let cols: &mut NativeClaimLayoutCols<F> = row.borrow_mut();
        cols.lookup_count *= factor;
    }
    matrix
}

fn scale_v19_slot_layout_lookups(
    mut matrix: RowMajorMatrix<F>,
    factor: usize,
) -> RowMajorMatrix<F> {
    let width = NativeInputSlotLayoutCols::<F>::width();
    let factor = F::from_usize(factor);
    for row in matrix.values.chunks_exact_mut(width) {
        let cols: &mut NativeInputSlotLayoutCols<F> = row.borrow_mut();
        cols.lookup_count *= factor;
    }
    matrix
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct DirectAirVaccStatementColsV19<T> {
    pub inner: WarpReplayProducerColsV19<T>,
}

/// Adapter from the generic standard-VACC buses to the protocol-v19 History
/// buses.  Every cryptographic value is received from a verifier AIR; only
/// cross-component manifest metadata is forwarded as witness data for the
/// History/SWIRL modules to bind later.
#[derive(Clone, Debug)]
pub struct DirectAirVaccStatementAirV19 {
    pub profile: NativeStandardVaccShapeProfile,
    pub protocol_bus: NativeStandardVaccProtocolBus,
    pub end_bus: NativeStandardVaccEndBus,
    pub root_bus: NativeStandardVaccRootBus,
    pub digest_bus: NativeStandardVaccDigestBus,
    pub claim_bus: NativeClaimValueBus,
    pub context_bus: DirectAirVaccContextBusV19,
    pub vacc_input_bus: CertifiedDirectAirVaccInputBusV19,
    pub vacc_input_lookup_count: usize,
}

impl BaseAir<F> for DirectAirVaccStatementAirV19 {
    fn width(&self) -> usize {
        DirectAirVaccStatementColsV19::<F>::width()
    }
}
impl openvm_stark_backend::BaseAirWithPublicValues<F> for DirectAirVaccStatementAirV19 {}
impl openvm_stark_backend::PartitionedBaseAir<F> for DirectAirVaccStatementAirV19 {}

impl<AB> Air<AB> for DirectAirVaccStatementAirV19
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(
            self.vacc_input_lookup_count != 0,
            "certified VACC input has no consumer"
        );
        assert!(
            self.profile.validate().is_ok(),
            "invalid standard VACC profile"
        );
        assert!(
            self.profile.log_codeword_len <= MAX_RAW_MESSAGE_POINT_LEN_V19
                && self.profile.beta_len <= MAX_FRESH_BETA_LEN_V19,
            "v19 statement capacity"
        );
        let main = builder.main();
        let row = main.row_slice(0).expect("direct VACC statement row");
        let local: &DirectAirVaccStatementColsV19<AB::Var> = (*row).borrow();
        let local = &local.inner;
        let enabled = local.active;
        builder.assert_bool(enabled);
        builder.when_first_row().assert_one(enabled);
        builder.when(enabled).assert_bool(local.has_prior);
        let proof_idx = join_u32::<AB>(local.proof_index_lo, local.proof_index_hi);
        let start_tidx = join_u32::<AB>(local.start_tidx_lo, local.start_tidx_hi);
        let end_tidx = join_u32::<AB>(local.end_tidx_lo, local.end_tidx_hi);
        self.protocol_bus.lookup_key(
            builder,
            NativeStandardVaccProtocolMessage {
                proof_idx: proof_idx.clone(),
                start_tidx,
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                has_prior: local.has_prior.into(),
                relation_digest: local.relation_digest.map(Into::into),
            },
            enabled,
        );
        self.end_bus.lookup_key(
            builder,
            NativeStandardVaccEndMessage {
                proof_idx: proof_idx.clone(),
                end_tidx,
            },
            enabled,
        );
        for (kind, root, multiplicity) in [
            (0usize, local.fresh_root, AB::Expr::from(enabled)),
            (
                1usize,
                local.prior_root,
                AB::Expr::from(enabled) * local.has_prior,
            ),
            (2usize, local.next_root, AB::Expr::from(enabled)),
        ] {
            self.root_bus.lookup_key(
                builder,
                NativeStandardVaccRootMessage {
                    proof_idx: proof_idx.clone(),
                    kind: AB::Expr::from_usize(kind),
                    root: root.map(Into::into),
                },
                multiplicity,
            );
        }
        for (state, digest, multiplicity) in [
            (
                0usize,
                local.previous_accumulator_digest,
                AB::Expr::from(enabled) * local.has_prior,
            ),
            (
                1usize,
                local.next_accumulator_digest,
                AB::Expr::from(enabled),
            ),
        ] {
            self.digest_bus.lookup_key(
                builder,
                NativeStandardVaccDigestMessage {
                    proof_idx: proof_idx.clone(),
                    state: AB::Expr::from_usize(state),
                    digest: digest.map(Into::into),
                },
                multiplicity,
            );
        }
        for limb in 0..DIGEST_SIZE {
            builder
                .when(AB::Expr::from(enabled) * (AB::Expr::ONE - local.has_prior))
                .assert_zero(local.prior_root[limb]);
            builder
                .when(AB::Expr::from(enabled) * (AB::Expr::ONE - local.has_prior))
                .assert_zero(local.previous_accumulator_digest[limb]);
        }
        for coordinate in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
            for limb in 0..D_EF {
                if coordinate >= self.profile.log_codeword_len {
                    builder
                        .when(enabled)
                        .assert_zero(local.fresh_alpha[coordinate][limb]);
                }
                if coordinate >= self.profile.log_message_len
                    && coordinate < self.profile.log_codeword_len
                {
                    builder
                        .when(enabled)
                        .assert_zero(local.fresh_alpha[coordinate][limb]);
                }
            }
            if coordinate < self.profile.log_codeword_len {
                self.claim_bus.receive(
                    builder,
                    NativeClaimValueMessage {
                        proof_idx: proof_idx.clone(),
                        source: AB::Expr::ZERO,
                        section: AB::Expr::from_usize(CLAIM_SECTION_ALPHA),
                        coordinate: AB::Expr::from_usize(coordinate),
                        value: local.fresh_alpha[coordinate].map(Into::into),
                    },
                    enabled,
                );
            }
        }
        for coordinate in 0..MAX_FRESH_BETA_LEN_V19 {
            if coordinate < self.profile.beta_len {
                self.claim_bus.receive(
                    builder,
                    NativeClaimValueMessage {
                        proof_idx: proof_idx.clone(),
                        source: AB::Expr::ZERO,
                        section: AB::Expr::from_usize(CLAIM_SECTION_BETA),
                        coordinate: AB::Expr::from_usize(coordinate),
                        value: local.fresh_beta[coordinate].map(Into::into),
                    },
                    enabled,
                );
            } else {
                for limb in 0..D_EF {
                    builder
                        .when(enabled)
                        .assert_zero(local.fresh_beta[coordinate][limb]);
                }
            }
        }
        for (section, value) in [
            (CLAIM_SECTION_MU, local.fresh_mu),
            (CLAIM_SECTION_ETA, local.fresh_eta),
        ] {
            self.claim_bus.receive(
                builder,
                NativeClaimValueMessage {
                    proof_idx: proof_idx.clone(),
                    source: AB::Expr::ZERO,
                    section: AB::Expr::from_usize(section),
                    coordinate: AB::Expr::ZERO,
                    value: value.map(Into::into),
                },
                enabled,
            );
        }
        for limb in local.fresh_eta {
            builder.when(enabled).assert_zero(limb);
        }

        self.context_bus.add_key_with_lookups(
            builder,
            DirectAirVaccContextMessageV19 {
                proof_index: proof_idx.clone(),
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
        self.vacc_input_bus.add_key_with_lookups(
            builder,
            CertifiedDirectAirVaccInputMessageV19 {
                proof_index: proof_idx,
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                update_index_lo: local.update_index_lo.into(),
                update_index_hi: local.update_index_hi.into(),
                shard_ordinal: local.shard_ordinal.into(),
                relation_digest: local.relation_digest.map(Into::into),
                root: local.fresh_root.map(Into::into),
                alpha_len: AB::Expr::from_usize(self.profile.log_codeword_len),
                alpha: local.fresh_alpha.map(|value| value.map(Into::into)),
                mu: local.fresh_mu.map(Into::into),
                beta_len: AB::Expr::from_usize(self.profile.beta_len),
                beta: local.fresh_beta.map(|value| value.map(Into::into)),
                eta: local.fresh_eta.map(Into::into),
            },
            AB::Expr::from(enabled) * AB::Expr::from_usize(self.vacc_input_lookup_count),
        );
    }
}

fn join_u32<AB: AirBuilder<F = F>>(lo: AB::Var, hi: AB::Var) -> AB::Expr {
    AB::Expr::from(lo) + AB::Expr::from(hi) * AB::Expr::from_u32(1 << 16)
}

/// Constrained renaming of the runtime prior slot to the verifier-key fixed
/// virtual prior source consumed by the ordinary v19 accumulator binder.
/// The source is authenticated as `kind=prior` by the schedule-produced slot
/// table, so this does not introduce a host-selected prior position.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct DirectAirReducedPriorClaimAdapterColsV19<T> {
    pub active: T,
    pub proof_idx: T,
    pub source: T,
    pub variant: T,
    pub section: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(DirectAirReducedPriorClaimAdapterColsV19<u8>)]
pub struct DirectAirReducedPriorClaimAdapterAirV19 {
    pub claim_bus: NativeClaimValueBus,
    pub slot_bus: NativeInputSlotLayoutBus,
    pub virtual_source: usize,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct DirectAirReducedVaccEndColsV19<T> {
    pub active: T,
    pub proof_idx: T,
    pub cursor_end_tidx: T,
    pub end_tidx: T,
    pub discarded_sample: [T; D_EF],
}

/// The semantic VACC cursor ends at the final target; the authenticated call
/// endpoint additionally includes the resumable observe-and-sample boundary.
#[derive(ColumnsAir)]
#[columns_via(DirectAirReducedVaccEndColsV19<u8>)]
pub struct DirectAirReducedVaccEndAirV19 {
    pub transcript_bus: openvm_recursion_circuit::bus::TranscriptBus,
    pub phase_cursor_bus: NativeVaccPhaseCursorBus,
    pub end_bus: NativeStandardVaccEndBus,
    pub end_tag: u64,
}

impl BaseAir<F> for DirectAirReducedVaccEndAirV19 {
    fn width(&self) -> usize {
        DirectAirReducedVaccEndColsV19::<F>::width()
    }
}
impl openvm_stark_backend::BaseAirWithPublicValues<F> for DirectAirReducedVaccEndAirV19 {}
impl openvm_stark_backend::PartitionedBaseAir<F> for DirectAirReducedVaccEndAirV19 {}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for DirectAirReducedVaccEndAirV19 {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("reduced VACC end row");
        let local: &DirectAirReducedVaccEndColsV19<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        self.phase_cursor_bus.receive(
            builder,
            NativeVaccPhaseCursorMessage {
                proof_idx: local.proof_idx.into(),
                boundary: AB::Expr::from_usize(2),
                tidx: local.cursor_end_tidx.into(),
            },
            local.active,
        );
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            local.cursor_end_tidx,
            core::array::from_fn(|limb| {
                if limb == 0 {
                    AB::Expr::from_u64(self.end_tag)
                } else {
                    AB::Expr::ZERO
                }
            }),
            local.active,
        );
        let sample_tidx = AB::Expr::from(local.cursor_end_tidx) + AB::Expr::from_usize(D_EF);
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            sample_tidx.clone(),
            local.discarded_sample,
            local.active,
        );
        builder
            .when(local.active)
            .assert_eq(local.end_tidx, sample_tidx + AB::Expr::from_usize(D_EF));
        self.end_bus.add_key_with_lookups(
            builder,
            NativeStandardVaccEndMessage {
                proof_idx: local.proof_idx.into(),
                end_tidx: local.end_tidx.into(),
            },
            local.active,
        );
    }
}

impl BaseAir<F> for DirectAirReducedPriorClaimAdapterAirV19 {
    fn width(&self) -> usize {
        DirectAirReducedPriorClaimAdapterColsV19::<F>::width()
    }
}
impl openvm_stark_backend::BaseAirWithPublicValues<F> for DirectAirReducedPriorClaimAdapterAirV19 {}
impl openvm_stark_backend::PartitionedBaseAir<F> for DirectAirReducedPriorClaimAdapterAirV19 {}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB>
    for DirectAirReducedPriorClaimAdapterAirV19
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("reduced prior claim adapter row");
        let local: &DirectAirReducedPriorClaimAdapterColsV19<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        self.slot_bus.lookup_key(
            builder,
            NativeInputSlotLayoutMessage {
                variant: local.variant.into(),
                source: local.source.into(),
                kind: [AB::Expr::ZERO, AB::Expr::ONE, AB::Expr::ZERO],
            },
            local.active,
        );
        let dynamic = NativeClaimValueMessage {
            proof_idx: local.proof_idx.into(),
            source: local.source.into(),
            section: local.section.into(),
            coordinate: local.coordinate.into(),
            value: local.value.map(Into::into),
        };
        self.claim_bus.receive(builder, dynamic, local.active);
        self.claim_bus.send(
            builder,
            NativeClaimValueMessage {
                proof_idx: local.proof_idx.into(),
                source: AB::Expr::from_usize(self.virtual_source),
                section: local.section.into(),
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active,
        );
    }
}

fn generate_direct_air_reduced_prior_claim_adapter_trace_v19(
    records: &[(usize, usize, &AccumulatorInstance<EF, Digest>)],
    input_arity: usize,
    layout: &NativePrivateAccumulatorLayout,
) -> Result<RowMajorMatrix<F>, DirectAirVaccVerifierErrorV19> {
    let width = DirectAirReducedPriorClaimAdapterColsV19::<F>::width();
    let rows_per_prior = layout.alpha_len + layout.beta_len + 2;
    let active_rows = records
        .len()
        .checked_mul(rows_per_prior)
        .ok_or(DirectAirVaccVerifierErrorV19::AirTraceCount)?;
    let height = active_rows.max(1).next_power_of_two();
    let mut values = F::zero_vec(height * width);
    let mut output_row = 0usize;
    for &(proof_idx, fresh_count, prior) in records {
        if fresh_count + 1 > input_arity {
            return Err(DirectAirVaccVerifierErrorV19::RecordShape(
                "reduced prior slot",
            ));
        }
        let digest = generate_native_accumulator_digest_traces(proof_idx, prior, layout).ok_or(
            DirectAirVaccVerifierErrorV19::RecordShape("reduced prior accumulator dimensions"),
        )?;
        for row in 0..digest.values.height() {
            let source = digest
                .values
                .row_slice(row)
                .ok_or(DirectAirVaccVerifierErrorV19::AirTraceCount)?;
            let source: &NativeAccumulatorValueCols<F> = (*source).borrow();
            if source.active == F::ZERO {
                continue;
            }
            let target: &mut DirectAirReducedPriorClaimAdapterColsV19<F> =
                values[output_row * width..(output_row + 1) * width].borrow_mut();
            target.active = F::ONE;
            target.proof_idx = F::from_usize(proof_idx);
            target.source = F::from_usize(fresh_count);
            target.variant = F::from_usize(fresh_count + input_arity + 1);
            target.section = source.section[2]
                + source.section[1] * F::from_usize(CLAIM_SECTION_MU)
                + source.section[3] * F::from_usize(CLAIM_SECTION_ETA);
            target.coordinate = source.coordinate;
            target.value = source.value;
            output_row += 1;
        }
    }
    if output_row != active_rows {
        return Err(DirectAirVaccVerifierErrorV19::AirTraceCount);
    }
    Ok(RowMajorMatrix::new(values, width))
}

fn generate_direct_air_reduced_vacc_end_trace_v19(
    proof_idx: usize,
    cursor_end_tidx: usize,
    end_tidx: usize,
    discarded_sample: EF,
) -> RowMajorMatrix<F> {
    let width = DirectAirReducedVaccEndColsV19::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut DirectAirReducedVaccEndColsV19<F> = values.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.proof_idx = F::from_usize(proof_idx);
    cols.cursor_end_tidx = F::from_usize(cursor_end_tidx);
    cols.end_tidx = F::from_usize(end_tidx);
    cols.discarded_sample
        .copy_from_slice(discarded_sample.as_basis_coefficients_slice());
    RowMajorMatrix::new(values, width)
}

/// Canonical inactive marker for the optional prior tables. The marker is not
/// an unconstrained all-zero witness: the value/hash AIRs still require their
/// first-row framing columns, while every lookup-producing row stays inactive.
/// Output tables never use this helper and remain non-empty by construction.
fn empty_v19_accumulator_digest_traces(
    layout: &NativePrivateAccumulatorLayout,
) -> NativeAccumulatorDigestTraces {
    let value_width = NativeAccumulatorValueCols::<F>::width();
    let mut value_rows = F::zero_vec(value_width * 2);
    let value: &mut NativeAccumulatorValueCols<F> = value_rows[..value_width].borrow_mut();
    value.is_first = F::ONE;
    value.section[0] = F::ONE;

    let hash_width = NativeAccumulatorHashCols::<F>::width();
    let mut hash_rows = F::zero_vec(hash_width * 2);
    let hash: &mut NativeAccumulatorHashCols<F> = hash_rows[..hash_width].borrow_mut();
    hash.is_first = F::ONE;

    let root_width = NativeAccumulatorRootDigestCols::<F>::width();
    let _ = layout;
    NativeAccumulatorDigestTraces {
        values: RowMajorMatrix::new(value_rows, value_width),
        hash: RowMajorMatrix::new(hash_rows, hash_width),
        root: RowMajorMatrix::new(F::zero_vec(root_width * 2), root_width),
        poseidon2_permute_inputs: Vec::new(),
        poseidon2_compress_inputs: Vec::new(),
        instance_digest: [F::ZERO; DIGEST_SIZE],
    }
}

/// Value binder for one prior/output accumulator instance.  It uses the same
/// canonical digest preimage as terminal Decide, but replaces legacy PCD-state
/// reads with the actual standard-VACC claim/algebra buses.
pub struct DirectAirAccumulatorValueAirV19 {
    pub mode: NativeAccumulatorBindingMode,
    pub allow_empty: bool,
    pub layout: NativePrivateAccumulatorLayout,
    /// Prior accumulators occupy the final input slot. This is `1` for the
    /// legacy arity-two verifier but is setup-dependent for exact finite calls.
    pub claim_source: usize,
    pub digest_element_bus: NativeAccumulatorDigestElementBus,
    pub claim_bus: NativeClaimValueBus,
    pub batching_bus: NativeBatchingOutputBus,
    pub folded_bus: NativeFoldedClaimBus,
    pub twin_bus: NativeTwinScalarBus,
}

impl BaseAir<F> for DirectAirAccumulatorValueAirV19 {
    fn width(&self) -> usize {
        NativeAccumulatorValueCols::<F>::width()
    }
}
impl openvm_stark_backend::BaseAirWithPublicValues<F> for DirectAirAccumulatorValueAirV19 {}
impl openvm_stark_backend::PartitionedBaseAir<F> for DirectAirAccumulatorValueAirV19 {}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for DirectAirAccumulatorValueAirV19 {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("direct accumulator value row");
        let next_row = main
            .row_slice(1)
            .expect("direct accumulator value next row");
        let local: &NativeAccumulatorValueCols<AB::Var> = (*local_row).borrow();
        let next: &NativeAccumulatorValueCols<AB::Var> = (*next_row).borrow();
        for flag in [
            local.active,
            local.is_first,
            local.is_last,
            local.section_last,
        ] {
            builder.assert_bool(flag);
        }
        for flag in local.section {
            builder.assert_bool(flag);
        }
        let [is_alpha, is_mu, is_beta, is_eta] = local.section.map(AB::Expr::from);
        builder
            .when(local.active)
            .assert_one(is_alpha.clone() + is_mu.clone() + is_beta.clone() + is_eta.clone());
        let section_sum = is_alpha.clone() + is_mu.clone() + is_beta.clone() + is_eta.clone();
        if self.allow_empty {
            // The canonical empty bootstrap marker retains only the mandatory
            // first-row framing bit. Every later inactive padding row is all
            // zero. This is verifier-key-selected, not a host acceptance bit.
            builder
                .when(AB::Expr::ONE - local.active)
                .assert_eq(section_sum, local.is_first);
        } else {
            builder
                .when(AB::Expr::ONE - local.active)
                .assert_zero(section_sum);
        }
        let section_len = is_alpha.clone() * AB::Expr::from_usize(self.layout.alpha_len)
            + is_mu.clone()
            + is_beta.clone() * AB::Expr::from_usize(self.layout.beta_len)
            + is_eta.clone();
        builder
            .when(local.active)
            .assert_eq(local.coordinate + local.remaining, section_len);
        let remaining_minus_one = local.remaining - AB::Expr::ONE;
        builder
            .when(local.active * local.section_last)
            .assert_zero(remaining_minus_one.clone());
        builder
            .when(local.active * (AB::Expr::ONE - local.section_last))
            .assert_one(remaining_minus_one * local.section_last_inverse);
        builder
            .when(local.active)
            .assert_eq(local.is_last, is_eta.clone() * local.section_last);
        if !self.allow_empty {
            builder.when_first_row().assert_one(local.active);
        }
        builder.when_first_row().assert_one(local.is_first);
        builder.when_first_row().assert_one(local.section[0]);
        builder.when_first_row().assert_zero(local.coordinate);
        builder.when_first_row().assert_zero(local.ordinal);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder.when_transition().assert_eq(
            local.active - next.active,
            local.is_last * (AB::Expr::ONE - next.active),
        );
        let mut transition = builder.when_transition();
        let mut same_proof = transition.when(next.active * (AB::Expr::ONE - next.is_first));
        same_proof.assert_eq(next.proof_idx, local.proof_idx);
        same_proof.assert_eq(next.ordinal, local.ordinal + AB::F::ONE);
        let mut transition = builder.when_transition();
        let mut next_proof = transition.when(next.active * next.is_first);
        next_proof.assert_one(local.is_last);
        next_proof.assert_eq(next.proof_idx, local.proof_idx + AB::F::ONE);
        next_proof.assert_zero(next.ordinal);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);
        let continue_section =
            next.active * (AB::Expr::ONE - next.is_first) * (AB::Expr::ONE - local.section_last);
        let mut transition = builder.when_transition();
        let mut same = transition.when(continue_section);
        for index in 0..4 {
            same.assert_eq(next.section[index], local.section[index]);
        }
        same.assert_eq(next.coordinate, local.coordinate + AB::F::ONE);
        same.assert_eq(next.remaining, local.remaining - AB::F::ONE);
        let advance_section = next.active * (AB::Expr::ONE - next.is_first) * local.section_last;
        let mut transition = builder.when_transition();
        let mut advance = transition.when(advance_section);
        advance.assert_zero(next.coordinate);
        advance.assert_eq(next.section[0], AB::Expr::ZERO);
        advance.assert_eq(next.section[1], is_alpha.clone());
        advance.assert_eq(next.section[2], is_mu.clone());
        advance.assert_eq(next.section[3], is_beta.clone());

        let state = match self.mode {
            NativeAccumulatorBindingMode::Prior => 0,
            NativeAccumulatorBindingMode::Output => 1,
        };
        for limb in 0..D_EF {
            self.digest_element_bus.send(
                builder,
                NativeAccumulatorDigestElementMessage {
                    proof_idx: local.proof_idx.into(),
                    state: AB::Expr::from_usize(state),
                    index: AB::Expr::from_usize(3)
                        + local.ordinal * AB::Expr::from_usize(D_EF)
                        + AB::Expr::from_usize(limb),
                    value: local.value[limb].into(),
                },
                local.active,
            );
        }
        let section = is_beta.clone()
            + is_mu.clone() * AB::Expr::from_usize(CLAIM_SECTION_MU)
            + is_eta.clone() * AB::Expr::from_usize(CLAIM_SECTION_ETA);
        match self.mode {
            NativeAccumulatorBindingMode::Prior => self.claim_bus.receive(
                builder,
                NativeClaimValueMessage {
                    proof_idx: local.proof_idx.into(),
                    source: AB::Expr::from_usize(self.claim_source),
                    section,
                    coordinate: local.coordinate.into(),
                    value: local.value.map(Into::into),
                },
                local.active,
            ),
            NativeAccumulatorBindingMode::Output => {
                self.batching_bus.receive(
                    builder,
                    NativeBatchingOutputMessage {
                        proof_idx: local.proof_idx.into(),
                        section: AB::Expr::from_usize(BATCHING_OUTPUT_ALPHA),
                        coordinate: local.coordinate.into(),
                        value: local.value.map(Into::into),
                    },
                    local.active * is_alpha.clone(),
                );
                self.batching_bus.receive(
                    builder,
                    NativeBatchingOutputMessage {
                        proof_idx: local.proof_idx.into(),
                        section: AB::Expr::from_usize(BATCHING_OUTPUT_MU),
                        coordinate: AB::Expr::ZERO,
                        value: local.value.map(Into::into),
                    },
                    local.active * is_mu.clone(),
                );
                self.folded_bus.receive(
                    builder,
                    NativeFoldedClaimMessage {
                        proof_idx: local.proof_idx.into(),
                        section: AB::Expr::from_usize(CLAIM_SECTION_BETA),
                        coordinate: local.coordinate.into(),
                        value: local.value.map(Into::into),
                    },
                    local.active * is_beta.clone(),
                );
                self.twin_bus.receive(
                    builder,
                    NativeTwinScalarMessage {
                        proof_idx: local.proof_idx.into(),
                        kind: AB::Expr::from_usize(TWIN_SCALAR_ETA),
                        value: local.value.map(Into::into),
                    },
                    local.active * is_eta.clone(),
                );
            }
        }
    }
}

pub struct DirectAirAccumulatorRootDigestAirV19 {
    pub mode: NativeAccumulatorBindingMode,
    pub allow_empty: bool,
    pub root_bus: NativeAccumulatorRootBus,
    pub algebraic_digest_bus: NativeAccumulatorAlgebraicDigestBus,
    pub compress_bus: openvm_recursion_circuit::bus::Poseidon2CompressBus,
    pub digest_bus: NativeStandardVaccDigestBus,
    /// Optional History-seal export. Exact finite WARP proves this digest in
    /// the complete verifier relation and deliberately has no History replay.
    pub certified_digest_bus: Option<NativeCertifiedAccumulatorDigestBus>,
}

impl BaseAir<F> for DirectAirAccumulatorRootDigestAirV19 {
    fn width(&self) -> usize {
        NativeAccumulatorRootDigestCols::<F>::width()
    }
}
impl openvm_stark_backend::BaseAirWithPublicValues<F> for DirectAirAccumulatorRootDigestAirV19 {}
impl openvm_stark_backend::PartitionedBaseAir<F> for DirectAirAccumulatorRootDigestAirV19 {}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for DirectAirAccumulatorRootDigestAirV19 {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("direct accumulator digest row");
        let local: &NativeAccumulatorRootDigestCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        if !self.allow_empty {
            builder.when_first_row().assert_one(local.active);
        }
        let state = match self.mode {
            NativeAccumulatorBindingMode::Prior => 0,
            NativeAccumulatorBindingMode::Output => 1,
        };
        self.algebraic_digest_bus.receive(
            builder,
            NativeAccumulatorAlgebraicDigestMessage {
                proof_idx: local.proof_idx.into(),
                state: AB::Expr::from_usize(state),
                digest: local.algebraic_digest.map(Into::into),
            },
            local.active,
        );
        self.root_bus.receive(
            builder,
            NativeAccumulatorRootMessage {
                proof_idx: local.proof_idx.into(),
                state: AB::Expr::from_usize(state),
                digest: local.root.map(Into::into),
            },
            local.active,
        );
        self.compress_bus.lookup_key(
            builder,
            openvm_recursion_circuit::bus::Poseidon2CompressMessage {
                input: core::array::from_fn(|index| {
                    if index < DIGEST_SIZE {
                        local.root[index].into()
                    } else {
                        local.algebraic_digest[index - DIGEST_SIZE].into()
                    }
                }),
                output: local.instance_digest.map(Into::into),
            },
            local.active,
        );
        self.digest_bus.add_key_with_lookups(
            builder,
            NativeStandardVaccDigestMessage {
                proof_idx: local.proof_idx.into(),
                state: AB::Expr::from_usize(state),
                digest: local.instance_digest.map(Into::into),
            },
            local.active,
        );
        if let (NativeAccumulatorBindingMode::Output, Some(certified_digest_bus)) =
            (self.mode, self.certified_digest_bus)
        {
            certified_digest_bus.send(
                builder,
                NativeCertifiedAccumulatorDigestMessage {
                    proof_idx: local.proof_idx.into(),
                    digest: local.instance_digest.map(Into::into),
                },
                local.active,
            );
        }
    }
}

const _: () = assert!(POSEIDON2_WIDTH == 2 * DIGEST_SIZE);

#[cfg(test)]
pub(crate) mod tests {
    use std::{convert::Infallible, sync::Arc};

    use openvm_recursion_circuit::{
        primitives::bus::{ExpBitsLenBus, RightShiftBus},
        system::BusIndexManager,
    };
    use openvm_stark_backend::{
        air_builders::{
            debug::{check_constraints, check_logup},
            symbolic::{get_symbolic_builder, SymbolicConstraintsDag, SymbolicRapBuilder},
            PartitionedAirBuilder,
        },
        interaction::SymbolicInteraction,
        keygen::types::{StarkVerifyingKey, StarkVerifyingParams, TraceWidth},
        native_warp::{
            CpuDirectAirProductExecutor, CpuDirectAirShard, DirectAirCodeClass,
            DirectAirFreshInput, DirectAirPesatIndex, DirectAirPesatInstance,
            DirectAirPublicSchema, DirectAirTranscriptFactory, FixedMultiAirPesatIndex,
            LocalWarpAccumulator, LocalWarpTransitionExecutor, LocalWarpTransitionJob,
            NativeWarpChallenger, PerShardTranscript,
        },
        p3_air::BaseAirWithPublicValues,
        p3_matrix::dense::RowMajorMatrixView,
        warp_accum::{
            AppendixDBaseFreshOpeningProver, AppendixDBaseFreshOpeningVerifier,
            AppendixDBaseFreshSource, BaseFreshCodewordBatchOpeningProver,
            ExternalCommitmentObserver, ExternalCommittedPesatSource,
            FreshCodewordBatchOpeningProver, MerkleBatchOpeningProof, MerkleOpeningBackend,
            WarpAccumError, WarpParams, WarpRootProver, WhirInitialRsWarpCode,
            WhirRsCodeProverData, FINITE_STACKED_FRESH_BASE_ALPHABET_TAG,
            FINITE_STACKED_FRESH_ROW_LAYOUT_VERSION, FINITE_STACKED_FRESH_TRANSCRIPT_VERSION,
        },
        warp_pesat::{AlgebraicChallenger, FreshPesatSource, LinearChainSchedule},
        FiatShamirTranscript, PartitionedBaseAir,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::{
        default_duplex_sponge_recorder, poseidon2_compress_with_capacity, DuplexSpongeRecorder,
    };
    use p3_air::AirBuilderWithPublicValues;
    use p3_field::ExtensionField;

    use super::*;
    use crate::circuit::native_warp_history_v19::{
        generate_warp_replay_producer_trace_v19, CertifiedFreshExplicitDigestBusV19,
        CertifiedSwirlRawOpeningBusV19, CertifiedSwirlRawOpeningMessageV19,
        CertifiedWarpReplayBusV19, CertifiedWarpReplayMessageV19, HistoryPoseidon2CompressBusV19,
        HistoryPoseidon2CompressMessageV19, TranscriptCheckpointRecordV19,
        WarpReplayProducerAirV19, LOGUP_ONLY_MODE_TAG_V19,
    };

    const SHARD_TRANSCRIPT_TAG: u64 = 0x4e57_5452_5348_0013;
    const SHARD_TRANSCRIPT_VACC_START_TAG: u64 = 0x4e57_5653_5441_0013;

    struct InteractionOnlyAir {
        num_public_values: usize,
    }

    impl BaseAir<F> for InteractionOnlyAir {
        fn width(&self) -> usize {
            2
        }
    }
    impl BaseAirWithPublicValues<F> for InteractionOnlyAir {
        fn num_public_values(&self) -> usize {
            self.num_public_values
        }
    }
    impl PartitionedBaseAir<F> for InteractionOnlyAir {
        fn cached_main_widths(&self) -> Vec<usize> {
            vec![1]
        }

        fn common_main_width(&self) -> usize {
            1
        }
    }
    impl Air<SymbolicRapBuilder<F>> for InteractionOnlyAir {
        fn eval(&self, builder: &mut SymbolicRapBuilder<F>) {
            let cached = builder.cached_mains()[0].clone();
            let common = builder.common_main().clone();
            builder.push_interaction(
                11,
                [
                    cached.row_slice(0).expect("cached row")[0],
                    builder.public_values()[0],
                ],
                common.row_slice(0).expect("common row")[0],
                1,
            );
        }
    }

    struct ExactFreshLinkMultiplicitySinkAir {
        profile: NativeExactFiniteVaccTranscriptProfile,
        schedule_bus: NativeExactFiniteVaccScheduleBus,
        call_protocol_bus: NativeExactFiniteVaccCallProtocolBus,
        root_bus: NativeStandardVaccRootBus,
        fresh_root: Digest,
        call_start_tidx: usize,
    }

    impl BaseAir<F> for ExactFreshLinkMultiplicitySinkAir {
        fn width(&self) -> usize {
            1
        }
    }
    impl BaseAirWithPublicValues<F> for ExactFreshLinkMultiplicitySinkAir {}
    impl PartitionedBaseAir<F> for ExactFreshLinkMultiplicitySinkAir {}
    impl<AB> Air<AB> for ExactFreshLinkMultiplicitySinkAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let enabled = builder.main().row_slice(0).expect("exact link sink row")[0];
            let call = self.profile.calls[0];
            let schedule_end = native_exact_finite_vacc_schedule_prefix_elements(&self.profile)
                .expect("valid exact profile")
                .len()
                * D_EF;
            self.schedule_bus.lookup_key(
                builder,
                NativeExactFiniteVaccScheduleMessage {
                    proof_idx: AB::Expr::ZERO,
                    end_tidx: AB::Expr::from_usize(schedule_end),
                    call_count: AB::Expr::ONE,
                    total_fresh: AB::Expr::from_usize(call.fresh_count),
                    relation_digest: self.profile.relation_digest.map(AB::Expr::from),
                    index_digest: self.profile.index_digest.map(AB::Expr::from),
                    setup_digest: self.profile.setup_digest.map(AB::Expr::from),
                    schedule_digest: self.profile.schedule_digest.map(AB::Expr::from),
                },
                enabled,
            );
            self.call_protocol_bus.lookup_key(
                builder,
                NativeExactFiniteVaccCallProtocolMessage {
                    proof_idx: AB::Expr::ZERO,
                    start_tidx: AB::Expr::from_usize(self.call_start_tidx),
                    call_index: AB::Expr::ZERO,
                    input_arity: AB::Expr::from_usize(call.input_arity),
                    fresh_count: AB::Expr::from_usize(call.fresh_count),
                    prior_count: AB::Expr::ZERO,
                    relation_digest: self.profile.relation_digest.map(AB::Expr::from),
                    index_digest: self.profile.index_digest.map(AB::Expr::from),
                    setup_digest: self.profile.setup_digest.map(AB::Expr::from),
                    schedule_digest: self.profile.schedule_digest.map(AB::Expr::from),
                },
                enabled,
            );
            self.root_bus.lookup_key(
                builder,
                NativeStandardVaccRootMessage {
                    proof_idx: AB::Expr::ZERO,
                    kind: AB::Expr::ZERO,
                    root: self.fresh_root.map(AB::Expr::from),
                },
                enabled,
            );
        }
    }

    struct ExactFreshClaimMultiplicitySinkAir {
        claim_bus: NativeClaimValueBus,
        lookup_count: usize,
    }

    impl BaseAir<F> for ExactFreshClaimMultiplicitySinkAir {
        fn width(&self) -> usize {
            NativeClaimValueCols::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for ExactFreshClaimMultiplicitySinkAir {}
    impl PartitionedBaseAir<F> for ExactFreshClaimMultiplicitySinkAir {}
    impl<AB> Air<AB> for ExactFreshClaimMultiplicitySinkAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("exact claim sink row");
            let local: &NativeClaimValueCols<AB::Var> = (*row).borrow();
            self.claim_bus.receive(
                builder,
                NativeClaimValueMessage {
                    proof_idx: local.proof_idx.into(),
                    source: local.source.into(),
                    section: local.section.into(),
                    coordinate: local.coordinate.into(),
                    value: local.value.map(Into::into),
                },
                AB::Expr::from(local.active) * AB::Expr::from_usize(self.lookup_count),
            );
        }
    }

    struct ExactFreshSlotRemainderSinkAir {
        slot_bus: NativeInputSlotLayoutBus,
        lookup_count: usize,
    }

    impl BaseAir<F> for ExactFreshSlotRemainderSinkAir {
        fn width(&self) -> usize {
            NativeInputSlotLayoutCols::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for ExactFreshSlotRemainderSinkAir {}
    impl PartitionedBaseAir<F> for ExactFreshSlotRemainderSinkAir {}
    impl<AB> Air<AB> for ExactFreshSlotRemainderSinkAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("exact slot sink row");
            let local: &NativeInputSlotLayoutCols<AB::Var> = (*row).borrow();
            self.slot_bus.lookup_key(
                builder,
                NativeInputSlotLayoutMessage {
                    variant: local.variant.into(),
                    source: local.source.into(),
                    kind: local.kind.map(Into::into),
                },
                AB::Expr::from(local.active) * AB::Expr::from_usize(self.lookup_count),
            );
        }
    }

    #[derive(Clone)]
    struct TestTranscriptFactory;

    impl
        DirectAirTranscriptFactory<
            openvm_stark_backend::native_warp::DirectAirPesatShardKey<Digest>,
            Digest,
            Digest,
        > for TestTranscriptFactory
    {
        type Transcript = NativeWarpChallenger<NativeSC, DuplexSpongeRecorder>;
        type FinalCheckpoint = Digest;
        type Error = Infallible;

        fn create(
            &self,
            key: &openvm_stark_backend::native_warp::DirectAirPesatShardKey<Digest>,
            seed: &PerShardTranscript<Digest, Digest>,
        ) -> Result<Self::Transcript, Self::Error> {
            Ok(test_challenger(key, seed).0)
        }

        fn finish(&self, _transcript: Self::Transcript) -> Result<Digest, Self::Error> {
            Ok([F::ZERO; DIGEST_SIZE])
        }
    }

    pub(crate) struct Fixture {
        pub(crate) config: NativeSC,
        pub(crate) profile: NativeStandardVaccProfile,
        pub(crate) module: DirectAirVaccVerifierModuleV19,
        pub(crate) producer_air: WarpReplayProducerAirV19,
        pub(crate) history_bus: CertifiedWarpReplayBusV19,
        pub(crate) swirl_opening_bus: CertifiedSwirlRawOpeningBusV19,
        pub(crate) history_compress_bus: HistoryPoseidon2CompressBusV19,
        pub(crate) producer: DirectAirVaccProducerRecordV19,
        pub(crate) verification: DirectAirVaccVerificationV19,
        pub(crate) transcript: TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
        pub(crate) prior: Option<AccumulatorInstance<EF, Digest>>,
    }

    pub(crate) struct AppendixDFixture {
        pub(crate) config: NativeSC,
        pub(crate) profile: NativeStandardVaccProfile,
        pub(crate) producer: DirectAirVaccProducerRecordV19,
        pub(crate) verification: DirectAirAppendixDVaccVerificationV19,
        pub(crate) transcript: TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    }

    #[derive(Clone)]
    struct HostAppendixDSource {
        root: Digest,
        base_prover_data: WhirRsCodeProverData<F, Digest>,
        base_codeword: Vec<F>,
        base_message: Vec<F>,
        explicit: Vec<EF>,
        alpha: Vec<EF>,
        mu: EF,
    }

    impl FreshPesatSource<EF> for HostAppendixDSource {
        type Commitment = Digest;

        fn commitment(&self) -> Self::Commitment {
            self.root
        }

        fn explicit_instance(&self) -> &[EF] {
            &self.explicit
        }

        fn witness_len(&self) -> usize {
            self.base_message.len()
        }

        fn witness_value(&self, index: usize) -> EF {
            EF::from(self.base_message[index])
        }

        fn claimed_mu(&self) -> EF {
            self.mu
        }

        fn claimed_alpha(&self, _log_codeword_len: usize) -> Vec<EF> {
            self.alpha.clone()
        }
    }

    impl ExternalCommittedPesatSource<EF> for HostAppendixDSource {
        type ProverData = WhirRsCodeProverData<F, Digest>;

        fn codeword_len(&self) -> usize {
            self.base_codeword.len()
        }

        fn read_codeword_chunk(&self, start: usize, output: &mut [EF]) {
            let end = start + output.len();
            for (target, &value) in output.iter_mut().zip(&self.base_codeword[start..end]) {
                *target = EF::from(value);
            }
        }

        fn prover_data(&self) -> Self::ProverData {
            self.base_prover_data.clone()
        }

        fn release_dense_data(&mut self) {}
    }

    impl AppendixDBaseFreshSource<F, EF> for HostAppendixDSource {
        type BaseProverData = WhirRsCodeProverData<F, Digest>;

        fn base_prover_data(&self) -> Self::BaseProverData {
            self.base_prover_data.clone()
        }
    }

    impl ExternalCommitmentObserver<EF> for HostAppendixDSource {
        type Commitment = Digest;

        fn observe_external_commitment<Ch>(&self, commitment: &Digest, challenger: &mut Ch)
        where
            Ch: AlgebraicChallenger<EF>,
        {
            for &limb in commitment {
                challenger.observe(EF::from(limb));
            }
        }
    }

    #[derive(Clone)]
    struct HostBaseOpeningSource {
        root: Digest,
        prover_data: WhirRsCodeProverData<F, Digest>,
        codeword: Vec<F>,
    }

    impl FreshPesatSource<F> for HostBaseOpeningSource {
        type Commitment = Digest;

        fn commitment(&self) -> Self::Commitment {
            self.root
        }

        fn explicit_instance(&self) -> &[F] {
            &[]
        }

        fn witness_len(&self) -> usize {
            0
        }

        fn witness_value(&self, _index: usize) -> F {
            F::ZERO
        }

        fn claimed_mu(&self) -> F {
            F::ZERO
        }
    }

    impl ExternalCommittedPesatSource<F> for HostBaseOpeningSource {
        type ProverData = WhirRsCodeProverData<F, Digest>;

        fn codeword_len(&self) -> usize {
            self.codeword.len()
        }

        fn read_codeword_chunk(&self, start: usize, output: &mut [F]) {
            output.copy_from_slice(&self.codeword[start..start + output.len()]);
        }

        fn prover_data(&self) -> Self::ProverData {
            self.prover_data.clone()
        }
    }

    #[derive(Clone)]
    struct HostBaseOpeningBackend {
        inner:
            MerkleOpeningBackend<<NativeSC as openvm_stark_backend::StarkProtocolConfig>::Hasher>,
    }

    impl BaseFreshCodewordBatchOpeningProver<F, EF, HostAppendixDSource> for HostBaseOpeningBackend {
        type BatchProof = MerkleBatchOpeningProof<F, Digest>;
        type Error = WarpAccumError;

        fn supports_released_dense_data(&self) -> bool {
            true
        }

        fn open_base_batch(
            &self,
            source: &HostAppendixDSource,
            indices: &[usize],
        ) -> Result<(Vec<F>, Self::BatchProof), Self::Error> {
            let view = HostBaseOpeningSource {
                root: source.root,
                prover_data: source.base_prover_data.clone(),
                codeword: source.base_codeword.clone(),
            };
            self.inner.open_batch(&view, indices)
        }
    }

    fn fixture() -> Fixture {
        fixture_with_prior(false)
    }

    fn fixture_with_prior(with_prior: bool) -> Fixture {
        fixture_for_air_id(with_prior, 9)
    }

    fn exact_digest(seed: u32) -> Digest {
        core::array::from_fn(|limb| F::from_u32(seed + limb as u32))
    }

    fn exact_64_then_8_profile() -> NativeExactFiniteVaccTranscriptProfile {
        NativeExactFiniteVaccTranscriptProfile {
            relation_description: b"exact-finite-64-then-8-test-relation".to_vec(),
            external_index_binding: vec![EF::from_u32(17), EF::from_u32(19)],
            relation_digest: exact_digest(10),
            index_digest: exact_digest(20),
            setup_digest: exact_digest(30),
            schedule_digest: exact_digest(40),
            shape: NativeStandardVaccShapeProfile {
                num_ood: 1,
                num_shift_queries: 1,
                batching_arity: 4,
                log_message_len: 3,
                log_codeword_len: 4,
                initial_folding_factor: 1,
                log_constraints: 1,
                beta_len: 3,
                max_degree: 2,
                rows_per_query: 2,
            },
            max_input_arity: 64,
            calls: [
                NativeExactFiniteVaccCallProfile {
                    active: true,
                    input_arity: 64,
                    fresh_count: 64,
                    prior_count: 0,
                },
                NativeExactFiniteVaccCallProfile {
                    active: true,
                    input_arity: 8,
                    fresh_count: 7,
                    prior_count: 1,
                },
                NativeExactFiniteVaccCallProfile::inactive(),
            ],
        }
    }

    fn exact_module_for_call(
        profile: NativeExactFiniteVaccTranscriptProfile,
        call_index: usize,
    ) -> DirectAirVaccVerifierModuleV19 {
        let mut manager = BusIndexManager::new();
        let shared = BusInventory::new(&mut manager);
        let buses = NativeWarpPcdBusInventory::new(manager.next_bus_idx());
        let mut extra = BusIndexManager::from_next_bus_idx(buses.next_bus_idx());
        let exact = DirectAirExactFiniteVaccConfigV19 {
            transcript_profile: profile,
            call_index,
            protocol_digest: exact_digest(50),
            multiplicities:
                DirectAirExactFiniteVaccMultiplicityProfileV19::from_fresh_instance_link_requirements(
                    FreshInstanceLinkConsumerRequirements::EXACT,
                )
                .unwrap(),
            schedule_bus: NativeExactFiniteVaccScheduleBus::new(extra.new_bus_idx()),
            call_protocol_bus: NativeExactFiniteVaccCallProtocolBus::new(extra.new_bus_idx()),
            authority_bus: FiniteWarpV3ExactVaccAuthorityBus::new(extra.new_bus_idx()),
        };
        let history_buses = DirectAirVaccHistoryBusesV19 {
            context: DirectAirVaccContextBusV19::new(extra.new_bus_idx()),
            input: CertifiedDirectAirVaccInputBusV19::new(extra.new_bus_idx()),
        };
        DirectAirVaccVerifierModuleV19::new_exact_finite_call(
            exact,
            shared,
            buses,
            history_buses,
            NativeStandardVaccProtocolBus::new(extra.new_bus_idx()),
            NativeStandardVaccEndBus::new(extra.new_bus_idx()),
            NativeStandardVaccRootBus::new(extra.new_bus_idx()),
            NativeStandardVaccDigestBus::new(extra.new_bus_idx()),
            SystemParams::new_for_testing(10),
        )
        .unwrap()
    }

    #[test]
    fn exact_64_then_8_production_modules_use_key_fixed_call_algebra() {
        let profile = exact_64_then_8_profile();
        profile.validate().unwrap();
        for (call_index, input_arity, fresh_count, twin_rounds) in [(0, 64, 64, 6), (1, 8, 7, 3)] {
            let module = exact_module_for_call(profile.clone(), call_index);
            assert_eq!(module.input_arity(), input_arity);
            assert_eq!(module.fresh_count(), fresh_count);
            assert_eq!(
                module
                    .exact_config()
                    .unwrap()
                    .transcript_profile
                    .call_algebra_profile(call_index)
                    .unwrap()
                    .twin_rounds,
                twin_rounds,
            );
            assert_eq!(
                shared_twin_lookup_count_v19(input_arity).unwrap(),
                u32::try_from(input_arity + 1).unwrap()
            );
            assert_eq!(module.prior_claim_source(), input_arity - 1);
            let claim_count = module.profile.batching_arity;
            assert_eq!(
                direct_vacc_eq_group_offsets_v19(&module).unwrap(),
                (2 * input_arity + 1, 2 * input_arity + 1 + claim_count)
            );
            assert!(!module.airs::<NativeSC>().is_empty());
            assert!(!module.airs_without_poseidon::<NativeSC>().is_empty());
        }
    }

    #[test]
    fn exact_multiplicity_profile_rejects_zero_and_unbounded_counts() {
        let canonical =
            DirectAirExactFiniteVaccMultiplicityProfileV19::from_fresh_instance_link_requirements(
                FreshInstanceLinkConsumerRequirements::EXACT,
            )
            .unwrap();
        canonical.validate().unwrap();

        let mut invalid = canonical;
        invalid.fresh_claim_consumer_count = 0;
        assert!(invalid.validate().is_err());

        let mut invalid = canonical;
        invalid.exact_schedule_lookup_count =
            DirectAirExactFiniteVaccMultiplicityProfileV19::MAX_COUNT + 1;
        assert!(invalid.validate().is_err());

        let mut invalid = canonical;
        invalid.extra_fresh_slot_lookup_count_per_source = 0;
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn exact_fresh_link_multiplicities_balance_the_real_typed_buses() {
        let requirements = FreshInstanceLinkConsumerRequirements::EXACT;
        let multiplicities =
            DirectAirExactFiniteVaccMultiplicityProfileV19::from_fresh_instance_link_requirements(
                requirements,
            )
            .unwrap();
        assert_eq!(multiplicities.fresh_claim_consumer_count, 2);
        assert_eq!(multiplicities.exact_schedule_lookup_count, 2);
        assert_eq!(multiplicities.exact_call_protocol_lookup_count_per_call, 2);
        assert_eq!(multiplicities.fresh_root_lookup_count_per_call, 2);
        assert_eq!(multiplicities.extra_fresh_slot_lookup_count_per_source, 1);

        let mut profile = exact_64_then_8_profile();
        profile.calls = [
            NativeExactFiniteVaccCallProfile {
                active: true,
                input_arity: 2,
                fresh_count: 2,
                prior_count: 0,
            },
            NativeExactFiniteVaccCallProfile::inactive(),
            NativeExactFiniteVaccCallProfile::inactive(),
        ];
        let module = exact_module_for_call(profile.clone(), 0);
        let exact = module.exact_config().unwrap().clone();
        let call_start_tidx = native_exact_finite_vacc_schedule_prefix_elements(&profile)
            .unwrap()
            .len()
            * D_EF;
        let fresh_root = exact_digest(700);
        let output_root = exact_digest(800);

        let schedule_air = NativeExactFiniteVaccSchedulePrefixAir {
            transcript_bus: module.buses.transcript,
            schedule_bus: exact.schedule_bus,
            profile: profile.clone(),
            lookup_count: multiplicities.exact_schedule_lookup_count,
        };
        let call_air = NativeExactFiniteVaccCallPrefixAir {
            transcript_bus: module.buses.transcript,
            phase_cursor_bus: module.buses.vacc_phase_cursor,
            protocol_bus: exact.call_protocol_bus,
            legacy_protocol_bus: module.protocol_bus,
            profile: profile.clone(),
            call_index: 0,
            lookup_count: multiplicities.exact_call_protocol_lookup_count_per_call,
        };
        let claim_air = NativeStandardClaimValueAir {
            claim_bus: module.buses.claim_value,
            transcript_bus: module.buses.vacc_semantic_transcript,
            transcript_role_bus: module.buses.vacc_transcript_role,
            layout_bus: module.buses.claim_layout,
            slot_bus: module.buses.input_slot_layout,
            fresh_claim_consumer_count: multiplicities.fresh_claim_consumer_count,
            prior_claim_consumer_count: 1,
            fresh_alpha_len: profile.shape.log_codeword_len,
            fresh_tau_len: profile.shape.log_constraints,
            fresh_beta_tail_len: profile.shape.beta_len - profile.shape.log_constraints,
        };
        let claim_rows = profile.shape.log_codeword_len + profile.shape.beta_len + 2;
        let ordinary_slot_lookups = claim_rows + 2 * profile.shape.num_shift_queries;
        let slot_air = NativeExactFiniteInputSlotLayoutAir {
            bus: module.buses.input_slot_layout,
            input_arity: 2,
            fresh_count: 2,
            prior_count: 0,
            ordinary_lookup_count: ordinary_slot_lookups,
            extra_fresh_lookup_count: multiplicities.extra_fresh_slot_lookup_count_per_source,
        };
        let fresh_profile = module.exact_finite_fresh_profile();
        let commitment_air = NativeFiniteStackedFreshCommitmentAir {
            transcript_bus: module.buses.vacc_semantic_transcript,
            transcript_role_bus: module.buses.vacc_transcript_role,
            phase_cursor_bus: module.buses.vacc_phase_cursor,
            merkle_root_bus: module.buses.merkle_root,
            statement_root_bus: module.statement_root_bus,
            profile: fresh_profile,
        };
        let output_root_air = NativeStandardVaccCommitmentRootAir {
            transcript_bus: module.buses.vacc_semantic_transcript,
            transcript_role_bus: module.buses.vacc_transcript_role,
            merkle_root_bus: None,
            accumulator_root_bus: None,
            statement_root_bus: module.statement_root_bus,
            proof_kind: 2,
            transcript_role: 0,
            expected_tree_id: 0,
            expected_depth: 0,
        };
        let authority_air = DirectAirExactFiniteVaccAuthorityAirV19 {
            exact: exact.clone(),
            protocol_bus: module.protocol_bus,
            end_bus: module.end_bus,
            root_bus: module.statement_root_bus,
            digest_bus: module.statement_digest_bus,
            checkpoint_bus: module.buses.transcript_checkpoint,
            resume_bus: module.shared.resume_state_bus,
        };
        let fresh_link_sink = ExactFreshLinkMultiplicitySinkAir {
            profile: profile.clone(),
            schedule_bus: exact.schedule_bus,
            call_protocol_bus: exact.call_protocol_bus,
            root_bus: module.statement_root_bus,
            fresh_root,
            call_start_tidx,
        };
        let claim_sink = ExactFreshClaimMultiplicitySinkAir {
            claim_bus: module.buses.claim_value,
            lookup_count: multiplicities.fresh_claim_consumer_count,
        };
        let slot_sink = ExactFreshSlotRemainderSinkAir {
            slot_bus: module.buses.input_slot_layout,
            lookup_count: 2 * profile.shape.num_shift_queries
                + multiplicities.extra_fresh_slot_lookup_count_per_source,
        };

        let schedule_trace = generate_native_exact_finite_vacc_schedule_prefix_trace(0, 0);
        let call_trace = generate_native_exact_finite_vacc_call_prefix_trace(0, call_start_tidx);
        let alpha = vec![EF::from_u32(3); profile.shape.log_codeword_len];
        let beta = vec![EF::from_u32(5); profile.shape.beta_len];
        let bindings = vec![None; alpha.len() + beta.len() + 2];
        let claims = (0..2)
            .map(|source| NativeClaimTraceInput {
                proof_idx: 0,
                alpha: &alpha,
                beta: &beta,
                mu: EF::from_u32(11 + source),
                eta: EF::ZERO,
                is_fresh: true,
                is_prior: false,
                is_dummy: false,
                transcript_bindings: &bindings,
            })
            .collect::<Vec<_>>();
        let claim_trace =
            generate_native_claim_value_trace(&claims, profile.shape.log_constraints, None)
                .unwrap();
        let slot_count = u32::try_from(
            ordinary_slot_lookups + multiplicities.extra_fresh_slot_lookup_count_per_source,
        )
        .unwrap();
        let slot_trace = generate_native_exact_finite_input_slot_layout_trace(
            2,
            2,
            0,
            &[slot_count, slot_count],
        )
        .unwrap();
        let descriptors = (0..2)
            .map(|source| FiniteStackedFreshCommitment {
                root: fresh_root,
                source_ordinal: source,
                fresh_count: 2,
                log_message_len: profile.shape.log_message_len as u32,
                log_codeword_len: profile.shape.log_codeword_len as u32,
                rows_per_leaf: profile.shape.rows_per_query as u32,
                alphabet_tag: FINITE_STACKED_FRESH_BASE_ALPHABET_TAG,
                row_layout_version: FINITE_STACKED_FRESH_ROW_LAYOUT_VERSION,
                transcript_version: FINITE_STACKED_FRESH_TRANSCRIPT_VERSION,
            })
            .collect::<Vec<_>>();
        let commitment_trace = generate_native_finite_stacked_fresh_commitment_trace(
            0,
            call_start_tidx + 1,
            &descriptors,
            fresh_profile,
        )
        .unwrap();
        let output_root_trace =
            generate_native_standard_vacc_commitment_root_trace(0, 900, output_root);
        let authority_width = DirectAirExactFiniteVaccAuthorityColsV19::<F>::width();
        let mut authority_values = F::zero_vec(authority_width);
        let authority: &mut DirectAirExactFiniteVaccAuthorityColsV19<F> =
            authority_values.as_mut_slice().borrow_mut();
        authority.active = F::ONE;
        authority.start_tidx = F::from_usize(call_start_tidx);
        authority.end_tidx = F::from_u32(1_000);
        authority.fresh_root = fresh_root;
        authority.output_root = output_root;
        authority.output_digest = exact_digest(900);
        let authority_trace = RowMajorMatrix::new(authority_values, authority_width);
        let sink_trace = RowMajorMatrix::new(vec![F::ONE], 1);

        let mut target_buses = symbolic_interactions(&fresh_link_sink)
            .into_iter()
            .chain(symbolic_interactions(&claim_sink))
            .chain(symbolic_interactions(&slot_sink))
            .map(|interaction| interaction.bus_index)
            .collect::<Vec<_>>();
        target_buses.sort_unstable();
        target_buses.dedup();
        let filter = |interactions: Vec<SymbolicInteraction<F>>| {
            interactions
                .into_iter()
                .filter(|interaction| target_buses.contains(&interaction.bus_index))
                .collect::<Vec<_>>()
        };
        let interactions = vec![
            filter(symbolic_interactions(&schedule_air)),
            filter(symbolic_interactions(&call_air)),
            filter(symbolic_interactions(&claim_air)),
            filter(symbolic_interactions(&slot_air)),
            filter(symbolic_interactions(&commitment_air)),
            filter(symbolic_interactions(&output_root_air)),
            filter(symbolic_interactions(&authority_air)),
            filter(symbolic_interactions(&fresh_link_sink)),
            filter(symbolic_interactions(&claim_sink)),
            filter(symbolic_interactions(&slot_sink)),
        ];
        let matrices = vec![
            vec![schedule_trace.as_view()],
            vec![call_trace.as_view()],
            vec![claim_trace.as_view()],
            vec![slot_trace.as_view()],
            vec![commitment_trace.as_view()],
            vec![output_root_trace.as_view()],
            vec![authority_trace.as_view()],
            vec![sink_trace.as_view()],
            vec![claim_trace.as_view()],
            vec![slot_trace.as_view()],
        ];
        let names = (0..matrices.len())
            .map(|index| format!("exact-multiplicity-{index}"))
            .collect::<Vec<_>>();
        let empty = (0..matrices.len()).map(|_| None).collect::<Vec<_>>();
        let public = (0..matrices.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        check_logup(&names, &interactions, &empty, &matrices, &public);
    }

    #[test]
    fn authenticated_checkpoint_derivation_rejects_cursor_sample_and_state_mutations() {
        let fixture = fixture();
        let canonical = exact_v19_checkpoint_at(
            &fixture.transcript,
            fixture.producer.end_checkpoint.operation_index as usize,
        )
        .unwrap();
        assert_eq!(canonical, fixture.producer.end_checkpoint);
        validate_exact_v19_checkpoint(&fixture.transcript, canonical).unwrap();

        let mut mutated = canonical;
        mutated.operation_index = mutated.operation_index.saturating_add(1);
        assert!(validate_exact_v19_checkpoint(&fixture.transcript, mutated).is_err());

        let mut mutated = canonical;
        mutated.sample_count = mutated.sample_count.saturating_add(1);
        assert!(validate_exact_v19_checkpoint(&fixture.transcript, mutated).is_err());

        let mut mutated = canonical;
        mutated.state[0] += F::ONE;
        assert!(validate_exact_v19_checkpoint(&fixture.transcript, mutated).is_err());
    }

    #[test]
    fn fixed_hleaf_v4_uses_one_prior_bearing_step_one_module() {
        let mut fixture = fixture_with_prior(true);
        fixture.module.enable_fixed_hleaf_warp_step_v4();
        assert!(fixture.module.include_prior);
        assert_eq!(
            fixture.module.warp_step_mode,
            DirectAirVaccWarpStepModeV19::FixedHLeafV4
        );
    }

    #[test]
    #[should_panic(
        expected = "seeded fixed HLeaf requires the prior-bearing standard VACC relation"
    )]
    fn fixed_hleaf_v4_rejects_bootstrap_module() {
        let mut fixture = fixture_with_prior(false);
        fixture.module.enable_fixed_hleaf_warp_step_v4();
    }

    pub(crate) fn fixture_for_air_id(with_prior: bool, air_id: usize) -> Fixture {
        fixture_for_air_id_and_height(with_prior, air_id, 2)
    }

    pub(crate) fn fixture_for_air_id_and_height(
        with_prior: bool,
        air_id: usize,
        log_height: usize,
    ) -> Fixture {
        fixture_for_air_id_height_and_public_values(
            with_prior,
            air_id,
            log_height,
            vec![F::from_u32(7)],
        )
    }

    pub(crate) fn fixture_for_air_id_height_and_public_values(
        with_prior: bool,
        air_id: usize,
        log_height: usize,
        public_values: Vec<F>,
    ) -> Fixture {
        assert!(!public_values.is_empty());
        let params = SystemParams::new_for_testing(10);
        let config = NativeSC::default_from_params(params.clone());
        let relation = Arc::new(interaction_only_relation_with_air_id_and_public_values(
            &config,
            log_height,
            air_id,
            public_values.len(),
        ));
        assert_eq!(relation.exact_max_degree(), 1);
        assert_eq!(relation.warp_degree_envelope(), 5);
        let key = relation.description().shard_key.clone();
        let code = WhirInitialRsWarpCode::new(
            config.hasher().clone(),
            key.code_class.log_message_len as usize,
            key.code_class.log_blowup as usize,
            key.code_class.initial_folding_factor as usize,
            key.code_class.rows_per_query as usize,
        );
        let warp_params = WarpParams::with_input_arity(2, 1, 1).unwrap();
        let shard = CpuDirectAirShard::new(Arc::clone(&relation), code, warp_params.clone())
            .expect("direct shard");
        let mut executor = CpuDirectAirProductExecutor::new(TestTranscriptFactory);
        executor
            .register_shard(shard)
            .expect("register direct shard");
        let fresh = interaction_only_fresh(&relation, &public_values);
        let mut prepared = executor.prepare_fresh(&key, fresh).expect("prepare fresh");
        let raw_point = (0..key.code_class.log_message_len)
            .map(|coordinate| EF::from_u32(17 + u32::from(coordinate)))
            .collect::<Vec<_>>();
        let mut opening_value = evaluate_mle(prepared.message(), &raw_point);
        prepared = prepared
            .with_raw_message_opening(raw_point.clone(), opening_value)
            .expect("systematic raw opening");
        let seed = PerShardTranscript {
            segment_index: 0,
            domain_seed: [F::from_u32(31); DIGEST_SIZE],
            start_checkpoint: [F::from_u32(47); DIGEST_SIZE],
        };
        let mut output = executor
            .transition(LocalWarpTransitionJob {
                segment_index: 0,
                logical_index: 0,
                key: &key,
                code_class: &key.code_class,
                prior: None,
                fresh: prepared,
                transcript: &seed,
            })
            .expect("bootstrap transition");
        let mut receipt = executor.drain_receipts().pop().expect("transition receipt");
        let mut prior_instance = None;
        let mut selected_seed = seed;
        if with_prior {
            prior_instance = Some(receipt.prover_record.output_instance.clone());
            let prior = LocalWarpAccumulator {
                accumulator: output.accumulator,
                accumulator_root: output.accumulator_root,
                state_leaf_root: output.accumulator_root,
                code_class: key.code_class,
                last_updated_segment: 0,
            };
            selected_seed = PerShardTranscript {
                segment_index: 1,
                domain_seed: [F::from_u32(97); DIGEST_SIZE],
                start_checkpoint: [F::from_u32(101); DIGEST_SIZE],
            };
            let mut next_fresh = executor
                .prepare_fresh(&key, interaction_only_fresh(&relation, &public_values))
                .expect("prepare continuation fresh");
            opening_value = evaluate_mle(next_fresh.message(), &raw_point);
            next_fresh = next_fresh
                .with_raw_message_opening(raw_point.clone(), opening_value)
                .expect("continuation systematic raw opening");
            output = executor
                .transition(LocalWarpTransitionJob {
                    segment_index: 1,
                    logical_index: 0,
                    key: &key,
                    code_class: &key.code_class,
                    prior: Some(&prior),
                    fresh: next_fresh,
                    transcript: &selected_seed,
                })
                .expect("continuation transition");
            receipt = executor
                .drain_receipts()
                .pop()
                .expect("continuation receipt");
        }
        let replay_code = WhirInitialRsWarpCode::new(
            config.hasher().clone(),
            key.code_class.log_message_len as usize,
            key.code_class.log_blowup as usize,
            key.code_class.initial_folding_factor as usize,
            key.code_class.rows_per_query as usize,
        );
        let replay = WarpRootProver::new(
            replay_code,
            openvm_stark_backend::warp_pesat::LinearChainSchedule::new(2).unwrap(),
            warp_params,
        )
        .unwrap();
        let openings = MerkleOpeningBackend::new_recording(config.hasher().clone());
        let (mut challenger, start_checkpoint) = test_challenger(&key, &selected_seed);
        let verification = replay
            .verify_vacc_step_recorded::<F, EF, _, _, _, _>(
                relation.as_ref(),
                None,
                usize::from(with_prior),
                prior_instance.as_ref(),
                &receipt.proof,
                &mut challenger,
                &openings,
                &openings,
            )
            .expect("recorded VACC replay");
        let (transcript, end_checkpoint) = finish_test_challenger(challenger);
        assert_eq!(output.accumulator_root, verification.output_instance.rt);
        let fresh_claim = &receipt.proof.fresh_claims[0];
        let relation_digest = key.relation_digest;
        let output_digest = openvm_stark_backend::native_warp::native_accumulator_instance_digest(
            &config,
            &verification.output_instance,
        );
        let producer = DirectAirVaccProducerRecordV19 {
            proof_index: 0,
            segment_index: u32::from(with_prior),
            update_index: u32::from(with_prior),
            shard_ordinal: 0,
            has_prior: with_prior,
            key_digest: [F::from_u32(61); DIGEST_SIZE],
            relation_digest,
            source_forest_root: [F::from_u32(67); DIGEST_SIZE],
            segment_openings_digest: [F::from_u32(71); DIGEST_SIZE],
            prior_root: receipt.previous_root.unwrap_or([F::ZERO; DIGEST_SIZE]),
            fresh_root: receipt.fresh_root,
            next_root: receipt.next_root,
            opening_point: raw_point.iter().copied().map(ext_array).collect(),
            opening_value: ext_array(opening_value),
            fresh_alpha: fresh_claim.alpha.iter().copied().map(ext_array).collect(),
            fresh_mu: ext_array(fresh_claim.mu),
            fresh_beta: fresh_claim.beta.iter().copied().map(ext_array).collect(),
            fresh_eta: ext_array(fresh_claim.eta),
            previous_accumulator_digest: prior_instance.as_ref().map_or(
                [F::ZERO; DIGEST_SIZE],
                |prior| {
                    openvm_stark_backend::native_warp::native_accumulator_instance_digest(
                        &config, prior,
                    )
                },
            ),
            next_accumulator_digest: output_digest,
            previous_checkpoint_digest: [F::from_u32(73); DIGEST_SIZE],
            next_checkpoint_digest: [F::from_u32(79); DIGEST_SIZE],
            authenticated_batching_claim: ext_array(verification.batching.sigma_2),
            start_checkpoint,
            end_checkpoint,
        };
        let shape = relation.pesat_shape();
        let profile = NativeStandardVaccProfile::from_direct_air_relation(
            relation.as_ref(),
            WarpParams::with_input_arity(2, 1, 1).unwrap(),
        )
        .unwrap();
        assert_eq!(profile.log_constraints, shape.log_constraints);
        assert_eq!(
            profile,
            NativeStandardVaccProfile::from_direct_air_description(
                relation.description(),
                WarpParams::with_input_arity(2, 1, 1).unwrap(),
            )
            .unwrap()
        );
        let mut manager = BusIndexManager::new();
        let shared = BusInventory::new(&mut manager);
        let buses = NativeWarpPcdBusInventory::new(manager.next_bus_idx());
        let mut extra = BusIndexManager::from_next_bus_idx(buses.next_bus_idx());
        let prefix_event_bus_index = extra.new_bus_idx();
        debug_assert_eq!(prefix_event_bus_index, buses.next_bus_idx());
        let history_buses = DirectAirVaccHistoryBusesV19 {
            context: DirectAirVaccContextBusV19::new(extra.new_bus_idx()),
            input: CertifiedDirectAirVaccInputBusV19::new(extra.new_bus_idx()),
        };
        let protocol_bus = NativeStandardVaccProtocolBus::new(extra.new_bus_idx());
        let end_bus = NativeStandardVaccEndBus::new(extra.new_bus_idx());
        let root_bus = NativeStandardVaccRootBus::new(extra.new_bus_idx());
        let digest_bus = NativeStandardVaccDigestBus::new(extra.new_bus_idx());
        let history_bus = CertifiedWarpReplayBusV19::new(extra.new_bus_idx());
        let swirl_opening_bus = CertifiedSwirlRawOpeningBusV19::new(extra.new_bus_idx());
        let fresh_explicit_bus = CertifiedFreshExplicitDigestBusV19::new(extra.new_bus_idx());
        let history_compress_bus = HistoryPoseidon2CompressBusV19::new(extra.new_bus_idx());
        let module = DirectAirVaccVerifierModuleV19::new(
            profile.clone(),
            with_prior,
            shared,
            buses,
            history_buses,
            protocol_bus,
            end_bus,
            root_bus,
            digest_bus,
            params,
        )
        .unwrap();
        let producer_air = WarpReplayProducerAirV19 {
            log_message_len: profile.log_message_len,
            log_codeword_len: profile.log_codeword_len,
            beta_len: profile.beta_len,
            compress_bus: history_compress_bus,
            history_bus,
            context_bus: history_buses.context,
            swirl_opening_bus,
            vacc_input_bus: history_buses.input,
            canonical_vacc_input_bus: None,
            fresh_explicit_bus,
            fresh_explicit_lookup_count: 0,
            checkpoint_bus: module.buses.transcript_checkpoint,
            batching_claim_bus: module.buses.certified_batching_claim,
            next_accumulator_digest_bus: module.buses.certified_accumulator_digest,
            setup_schedule: None,
        };
        Fixture {
            config,
            profile,
            module,
            producer_air,
            history_bus,
            swirl_opening_bus,
            history_compress_bus,
            producer,
            verification,
            transcript,
            prior: prior_instance,
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn appendix_d_fixture_for_fixed_source(
        relation: Arc<FixedMultiAirPesatIndex<F, Digest>>,
        message: Vec<EF>,
        explicit: Vec<EF>,
        root: Digest,
        base_prover_data: WhirRsCodeProverData<F, Digest>,
        base_codeword: Vec<F>,
        opening_point: Vec<EF>,
        opening_value: EF,
        source_forest_root: Digest,
        segment_openings_digest: Digest,
    ) -> AppendixDFixture {
        let params = SystemParams::new_for_testing(10);
        let config = NativeSC::default_from_params(params);
        let description = relation.description();
        let code_class = description.code_class;
        assert_eq!(message.len(), 1usize << code_class.log_message_len);
        assert_eq!(opening_point.len(), code_class.log_message_len as usize);
        assert_eq!(evaluate_mle(&message, &opening_point), opening_value);
        let base_message = message
            .iter()
            .copied()
            .map(|value| value.as_base().expect("Appendix-D base source"))
            .collect::<Vec<_>>();
        let code = WhirInitialRsWarpCode::new(
            config.hasher().clone(),
            code_class.log_message_len as usize,
            code_class.log_blowup as usize,
            code_class.initial_folding_factor as usize,
            code_class.rows_per_query as usize,
        );
        let alpha = code
            .systematic_message_opening_point(&opening_point)
            .expect("Appendix-D systematic point");
        let source = HostAppendixDSource {
            root,
            base_prover_data,
            base_codeword,
            base_message,
            explicit,
            alpha,
            mu: opening_value,
        };
        let warp_params = WarpParams::with_input_arity(2, 1, 1).unwrap();
        let prover = WarpRootProver::new(
            code.clone(),
            LinearChainSchedule::new(2).unwrap(),
            warp_params.clone(),
        )
        .unwrap()
        .with_appendix_d_base_fresh();
        let fresh_openings = AppendixDBaseFreshOpeningProver::new(HostBaseOpeningBackend {
            inner: MerkleOpeningBackend::new(config.hasher().clone()),
        });
        let accumulator_openings = MerkleOpeningBackend::new(config.hasher().clone());
        let (mut challenger, start_checkpoint) = fixed_test_challenger(description.relation_digest);
        let (accumulator, proof, prover_record) = prover
            .prove_vacc_step_recorded::<F, EF, _, _, _, _, _, _>(
                relation.as_ref(),
                None,
                0,
                vec![source],
                None,
                &mut challenger,
                &fresh_openings,
                &accumulator_openings,
                &(),
            )
            .expect("Appendix-D VACC transition");
        let (transcript, end_checkpoint) = finish_test_challenger(challenger);

        let replay = WarpRootProver::new(
            code,
            LinearChainSchedule::new(2).unwrap(),
            warp_params.clone(),
        )
        .unwrap()
        .with_appendix_d_base_fresh();
        let fresh_verifier = AppendixDBaseFreshOpeningVerifier::new(
            MerkleOpeningBackend::new_recording(config.hasher().clone()),
        );
        let accumulator_verifier = MerkleOpeningBackend::new_recording(config.hasher().clone());
        let (mut verifier_challenger, verifier_start) =
            fixed_test_challenger(description.relation_digest);
        assert_eq!(verifier_start, start_checkpoint);
        let verification = replay
            .verify_vacc_step_recorded::<F, EF, _, _, _, _>(
                relation.as_ref(),
                None,
                0,
                None,
                &proof,
                &mut verifier_challenger,
                &fresh_verifier,
                &accumulator_verifier,
            )
            .expect("Appendix-D VACC replay");
        let (verifier_transcript, verifier_end) = finish_test_challenger(verifier_challenger);
        assert_eq!(verifier_end, end_checkpoint);
        assert_eq!(verifier_transcript.values(), transcript.values());
        assert_eq!(verifier_transcript.samples(), transcript.samples());
        assert_eq!(
            verifier_transcript.perm_results(),
            transcript.perm_results()
        );
        assert_eq!(verifier_transcript.events(), transcript.events());
        assert_eq!(
            verifier_transcript.permutation_transitions(),
            transcript.permutation_transitions()
        );
        assert_eq!(verification.output_instance, accumulator.instance);
        assert_eq!(verification.output_instance, prover_record.output_instance);

        let fresh_claim = &proof.fresh_claims[0];
        let output_digest = openvm_stark_backend::native_warp::native_accumulator_instance_digest(
            &config,
            &verification.output_instance,
        );
        let producer = DirectAirVaccProducerRecordV19 {
            proof_index: 0,
            segment_index: 0,
            update_index: 0,
            shard_ordinal: 0,
            has_prior: false,
            key_digest: [F::from_u32(61); DIGEST_SIZE],
            relation_digest: description.relation_digest,
            source_forest_root,
            segment_openings_digest,
            prior_root: [F::ZERO; DIGEST_SIZE],
            fresh_root: root,
            next_root: verification.output_instance.rt,
            opening_point: opening_point.iter().copied().map(ext_array).collect(),
            opening_value: ext_array(opening_value),
            fresh_alpha: fresh_claim.alpha.iter().copied().map(ext_array).collect(),
            fresh_mu: ext_array(fresh_claim.mu),
            fresh_beta: fresh_claim.beta.iter().copied().map(ext_array).collect(),
            fresh_eta: ext_array(fresh_claim.eta),
            previous_accumulator_digest: [F::ZERO; DIGEST_SIZE],
            next_accumulator_digest: output_digest,
            previous_checkpoint_digest: [F::from_u32(73); DIGEST_SIZE],
            next_checkpoint_digest: [F::from_u32(79); DIGEST_SIZE],
            authenticated_batching_claim: ext_array(verification.batching.sigma_2),
            start_checkpoint,
            end_checkpoint,
        };
        let profile = NativeStandardVaccProfile::from_fixed_multi_air_relation(
            relation.as_ref(),
            warp_params,
        )
        .unwrap();
        AppendixDFixture {
            config,
            profile,
            producer,
            verification,
            transcript,
        }
    }

    fn fixed_test_challenger(
        relation_digest: Digest,
    ) -> (
        NativeWarpChallenger<NativeSC, DuplexSpongeRecorder>,
        TranscriptCheckpointRecordV19,
    ) {
        let mut transcript = default_duplex_sponge_recorder();
        for value in [
            F::from_u64(SHARD_TRANSCRIPT_TAG),
            F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            F::ZERO,
            F::ZERO,
            F::ZERO,
        ] {
            FiatShamirTranscript::<NativeSC>::observe(&mut transcript, value);
        }
        FiatShamirTranscript::<NativeSC>::observe_commit(&mut transcript, relation_digest);
        FiatShamirTranscript::<NativeSC>::observe_commit(
            &mut transcript,
            [F::from_u32(31); DIGEST_SIZE],
        );
        FiatShamirTranscript::<NativeSC>::observe_commit(
            &mut transcript,
            [F::from_u32(47); DIGEST_SIZE],
        );
        FiatShamirTranscript::<NativeSC>::observe(
            &mut transcript,
            F::from_u64(SHARD_TRANSCRIPT_VACC_START_TAG),
        );
        let _ = FiatShamirTranscript::<NativeSC>::sample_ext(&mut transcript);
        let checkpoint = checkpoint_record(&transcript);
        (NativeWarpChallenger::new(transcript), checkpoint)
    }

    fn interaction_only_relation_with_air_id(
        config: &NativeSC,
        log_height: usize,
        air_id: usize,
    ) -> DirectAirPesatIndex<F, Digest> {
        interaction_only_relation_with_air_id_and_public_values(config, log_height, air_id, 1)
    }

    fn interaction_only_relation_with_air_id_and_public_values(
        config: &NativeSC,
        log_height: usize,
        air_id: usize,
        num_public_values: usize,
    ) -> DirectAirPesatIndex<F, Digest> {
        let width = TraceWidth {
            preprocessed: None,
            cached_mains: vec![1],
            common_main: 1,
        };
        let symbolic =
            get_symbolic_builder(&InteractionOnlyAir { num_public_values }, &width).constraints();
        assert!(symbolic.constraints.is_empty());
        let vk = StarkVerifyingKey {
            preprocessed_data: None,
            params: StarkVerifyingParams {
                width,
                num_public_values,
                need_rot: false,
            },
            symbolic_constraints: Arc::new(SymbolicConstraintsDag::from(symbolic)),
            max_constraint_degree: 0,
            is_required: true,
            unused_variables: Vec::new(),
        };
        let raw_len = (1usize << log_height) * 2;
        let log_message_len = raw_len.next_power_of_two().ilog2() as u8;
        DirectAirPesatIndex::from_verifying_key(
            config.hasher(),
            [F::from_u32(83); DIGEST_SIZE],
            air_id,
            log_height,
            &vk,
            None,
            DirectAirPublicSchema {
                public_values_len: num_public_values as u32,
                boundary_values_len: 0,
                schema_digest: [F::from_u32(89); DIGEST_SIZE],
            },
            DirectAirCodeClass {
                log_message_len,
                log_blowup: 1,
                log_codeword_len: log_message_len + 1,
                initial_folding_factor: 0,
                rows_per_query: 2,
            },
        )
        .expect("canonical interaction-only relation")
    }

    fn install_relation_catalog(fixture: &mut Fixture, air_ids: &[usize]) {
        let warp = WarpParams::with_input_arity(2, 1, 1).unwrap();
        let profiles = air_ids
            .iter()
            .map(|&air_id| {
                let relation = interaction_only_relation_with_air_id(&fixture.config, 2, air_id);
                NativeStandardVaccProfile::from_direct_air_relation(&relation, warp.clone())
                    .unwrap()
            })
            .collect();
        let old = &fixture.module;
        fixture.module = DirectAirVaccVerifierModuleV19::new_shape_batched(
            profiles,
            old.include_prior,
            old.shared.clone(),
            old.buses.clone(),
            old.history_buses,
            old.protocol_bus,
            old.end_bus,
            old.statement_root_bus,
            old.statement_digest_bus,
            SystemParams::new_for_testing(10),
        )
        .unwrap();
    }

    fn interaction_only_fresh(
        relation: &DirectAirPesatIndex<F, Digest>,
        public_values: &[F],
    ) -> DirectAirFreshInput<F> {
        let height = relation.height();
        let cached =
            RowMajorMatrix::new((0..height).map(|row| F::from_usize(3 + row)).collect(), 1);
        let common =
            RowMajorMatrix::new((0..height).map(|row| F::from_usize(13 + row)).collect(), 1);
        let witness = relation
            .witness_from_row_major_parts(&[cached], Some(&common))
            .unwrap();
        DirectAirFreshInput::new(
            DirectAirPesatInstance {
                public_values: public_values.to_vec(),
                boundary_values: Vec::new(),
            },
            witness,
        )
    }

    fn test_challenger(
        key: &openvm_stark_backend::native_warp::DirectAirPesatShardKey<Digest>,
        seed: &PerShardTranscript<Digest, Digest>,
    ) -> (
        NativeWarpChallenger<NativeSC, DuplexSpongeRecorder>,
        TranscriptCheckpointRecordV19,
    ) {
        let mut transcript = default_duplex_sponge_recorder();
        FiatShamirTranscript::<NativeSC>::observe(
            &mut transcript,
            F::from_u64(SHARD_TRANSCRIPT_TAG),
        );
        FiatShamirTranscript::<NativeSC>::observe(
            &mut transcript,
            F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
        );
        FiatShamirTranscript::<NativeSC>::observe(&mut transcript, F::from_u64(seed.segment_index));
        FiatShamirTranscript::<NativeSC>::observe(&mut transcript, F::from_u32(key.air_id));
        FiatShamirTranscript::<NativeSC>::observe(&mut transcript, F::from_u8(key.log_height));
        FiatShamirTranscript::<NativeSC>::observe_commit(&mut transcript, key.relation_digest);
        FiatShamirTranscript::<NativeSC>::observe_commit(&mut transcript, seed.domain_seed);
        FiatShamirTranscript::<NativeSC>::observe_commit(&mut transcript, seed.start_checkpoint);
        FiatShamirTranscript::<NativeSC>::observe(
            &mut transcript,
            F::from_u64(SHARD_TRANSCRIPT_VACC_START_TAG),
        );
        let _ = FiatShamirTranscript::<NativeSC>::sample_ext(&mut transcript);
        let checkpoint = checkpoint_record(&transcript);
        (NativeWarpChallenger::new(transcript), checkpoint)
    }

    fn finish_test_challenger(
        challenger: NativeWarpChallenger<NativeSC, DuplexSpongeRecorder>,
    ) -> (
        TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
        TranscriptCheckpointRecordV19,
    ) {
        let mut transcript = challenger.into_inner();
        FiatShamirTranscript::<NativeSC>::observe(
            &mut transcript,
            F::from_u64(SHARD_TRANSCRIPT_VACC_END_TAG_V19),
        );
        let _ = FiatShamirTranscript::<NativeSC>::sample_ext(&mut transcript);
        let checkpoint = checkpoint_record(&transcript);
        (transcript.log, checkpoint)
    }

    fn checkpoint_record(transcript: &DuplexSpongeRecorder) -> TranscriptCheckpointRecordV19 {
        let checkpoint = transcript.inner.checkpoint();
        assert_eq!(checkpoint.absorb_idx, 0);
        TranscriptCheckpointRecordV19 {
            operation_index: transcript.log.len().try_into().unwrap(),
            sample_count: (DIGEST_SIZE - checkpoint.sample_idx).try_into().unwrap(),
            state: checkpoint.state,
        }
    }

    fn evaluate_mle(values: &[EF], point: &[EF]) -> EF {
        assert_eq!(values.len(), 1usize << point.len());
        let mut layer = values.to_vec();
        for &coordinate in point {
            let half = layer.len() / 2;
            for index in 0..half {
                layer[index] = layer[index] + coordinate * (layer[index + half] - layer[index]);
            }
            layer.truncate(half);
        }
        layer[0]
    }

    fn ext_array(value: EF) -> [F; D_EF] {
        value.as_basis_coefficients_slice().try_into().unwrap()
    }

    fn check_air_ref(air: &AirRef<NativeSC>, trace: &RowMajorMatrix<F>) {
        let preprocessed = BaseAir::<F>::preprocessed_trace(air.as_ref());
        let preprocessed = preprocessed.as_ref().map(RowMajorMatrix::as_view);
        check_constraints::<_, NativeSC>(
            air.as_ref(),
            &air.name(),
            &preprocessed,
            &[RowMajorMatrixView::new(&trace.values, trace.width())],
            &[],
        );
    }

    #[test]
    fn grouped_exp_bits_uses_multiplicative_identity_padding() {
        let matrices = (0..3)
            .map(|index| {
                let generator = ExpBitsLenCpuTraceGenerator::default();
                generator.add_request(F::ONE, F::from_usize(index + 1), 1);
                generator
                    .generate_trace_row_major(None)
                    .expect("one ExpBits request")
            })
            .collect::<Vec<_>>();
        let merged = merge_v19_exp_bits_rows(matrices.iter()).expect("merged ExpBits trace");
        assert_eq!(merged.height(), 128);
        let padding_row = merged
            .row_slice(96)
            .expect("first non-power-of-two padding row");
        let padding: &ExpBitsLenCols<F> = (*padding_row).borrow();
        assert_eq!(padding.is_valid, F::ZERO);
        assert_eq!(padding.result, F::ONE);
        assert_eq!(padding.result_multiplier, F::ONE);

        let air: AirRef<NativeSC> = Arc::new(ExpBitsLenAir::new(
            ExpBitsLenBus::new(0),
            RightShiftBus::new(1),
        ));
        check_air_ref(&air, &merged);
    }

    #[derive(Clone, Debug)]
    struct VaccIntegrationBoundaryAir {
        opening_bus: CertifiedSwirlRawOpeningBusV19,
        history_bus: CertifiedWarpReplayBusV19,
        log_message_len: usize,
    }

    impl BaseAir<F> for VaccIntegrationBoundaryAir {
        fn width(&self) -> usize {
            WarpReplayProducerColsV19::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for VaccIntegrationBoundaryAir {}
    impl PartitionedBaseAir<F> for VaccIntegrationBoundaryAir {}

    impl<AB> Air<AB> for VaccIntegrationBoundaryAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("VACC integration boundary row");
            let local: &WarpReplayProducerColsV19<AB::Var> = (*row).borrow();
            let active = local.active;
            let proof_index = AB::Expr::from(local.proof_index_lo)
                + AB::Expr::from(local.proof_index_hi) * AB::Expr::from_u32(1 << 16);
            self.opening_bus.add_key_with_lookups(
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
                active,
            );
            self.history_bus.lookup_key(
                builder,
                CertifiedWarpReplayMessageV19 {
                    protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                    segment_index_lo: local.segment_index_lo.into(),
                    segment_index_hi: local.segment_index_hi.into(),
                    update_index_lo: local.update_index_lo.into(),
                    update_index_hi: local.update_index_hi.into(),
                    shard_ordinal: local.shard_ordinal.into(),
                    source_forest_root: local.source_forest_root.map(Into::into),
                    key_digest: local.key_digest.map(Into::into),
                    relation_digest: local.relation_digest.map(Into::into),
                    opening_claim_digest: local.opening_claim_digest.map(Into::into),
                    fresh_instance_digest: local.fresh_instance_digest.map(Into::into),
                    segment_openings_digest: local.segment_openings_digest.map(Into::into),
                    prior_root: local.prior_root.map(Into::into),
                    fresh_root: local.fresh_root.map(Into::into),
                    next_root: local.next_root.map(Into::into),
                    previous_accumulator_digest: local.previous_accumulator_digest.map(Into::into),
                    next_accumulator_digest: local.next_accumulator_digest.map(Into::into),
                    authenticated_batching_claim: local
                        .authenticated_batching_claim
                        .map(Into::into),
                    previous_checkpoint_digest: local.previous_checkpoint_digest.map(Into::into),
                    next_checkpoint_digest: local.next_checkpoint_digest.map(Into::into),
                    replay_endpoint_digest: local.replay_endpoint_digest.map(Into::into),
                    replay_binding_digest: local.replay_binding_digest.map(Into::into),
                },
                active,
            );
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct HistoryCompressionSourceAir(HistoryPoseidon2CompressBusV19);

    impl BaseAir<F> for HistoryCompressionSourceAir {
        fn width(&self) -> usize {
            1 + 3 * DIGEST_SIZE
        }
    }
    impl BaseAirWithPublicValues<F> for HistoryCompressionSourceAir {}
    impl PartitionedBaseAir<F> for HistoryCompressionSourceAir {}

    impl<AB> Air<AB> for HistoryCompressionSourceAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("history compression source row");
            let next = main
                .row_slice(1)
                .expect("history compression source next row");
            let active = row[0];
            builder.assert_bool(active);
            builder.when_transition().when(next[0]).assert_one(active);
            self.0.add_key_with_lookups(
                builder,
                HistoryPoseidon2CompressMessageV19 {
                    input: core::array::from_fn(|index| row[1 + index].into()),
                    output: core::array::from_fn(|index| row[1 + 2 * DIGEST_SIZE + index].into()),
                },
                active,
            );
        }
    }

    fn history_compression_trace(inputs: &[[F; 2 * DIGEST_SIZE]]) -> RowMajorMatrix<F> {
        let width = 1 + 3 * DIGEST_SIZE;
        let height = (inputs.len() + 1).next_power_of_two();
        let mut values = F::zero_vec(width * height);
        for (index, input) in inputs.iter().enumerate() {
            let row = &mut values[index * width..(index + 1) * width];
            row[0] = F::ONE;
            row[1..1 + 2 * DIGEST_SIZE].copy_from_slice(input);
            let left = input[..DIGEST_SIZE].try_into().unwrap();
            let right = input[DIGEST_SIZE..].try_into().unwrap();
            row[1 + 2 * DIGEST_SIZE..]
                .copy_from_slice(&poseidon2_compress_with_capacity(left, right).0);
        }
        RowMajorMatrix::new(values, width)
    }

    fn symbolic_interactions_dyn(air: &dyn AnyAir<NativeSC>) -> Vec<SymbolicInteraction<F>> {
        let preprocessed = BaseAir::<F>::preprocessed_trace(air);
        get_symbolic_builder(
            air,
            &TraceWidth {
                preprocessed: preprocessed.as_ref().map(Matrix::width),
                cached_mains: air.cached_main_widths(),
                common_main: air.common_main_width(),
            },
        )
        .constraints()
        .interactions
    }

    fn symbolic_interactions<R>(air: &R) -> Vec<SymbolicInteraction<F>>
    where
        R: Air<SymbolicRapBuilder<F>>
            + BaseAir<F>
            + BaseAirWithPublicValues<F>
            + PartitionedBaseAir<F>,
    {
        get_symbolic_builder(
            air,
            &TraceWidth {
                preprocessed: None,
                cached_mains: air.cached_main_widths(),
                common_main: air.common_main_width(),
            },
        )
        .constraints()
        .interactions
    }

    fn check_composed_vacc(
        fixture: &Fixture,
        producer: &DirectAirVaccProducerRecordV19,
        verification: &DirectAirVaccVerificationV19,
        transcript: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    ) -> Result<(), DirectAirVaccVerifierErrorV19> {
        let record = DirectAirVaccVerifierRecordV19 {
            producer,
            verification,
            transcript,
            prior: fixture.prior.as_ref(),
        };
        let verifier_trace = fixture.module.generate_trace(&fixture.config, &record)?;
        let verifier_airs = fixture.module.airs::<NativeSC>();
        for (air, matrix) in verifier_airs.iter().zip(verifier_trace.air_matrices()) {
            check_air_ref(air, matrix);
        }
        let producer_trace =
            generate_warp_replay_producer_trace_v19(&fixture.producer_air, &[producer.clone()], 1)
                .map_err(|_| DirectAirVaccVerifierErrorV19::RecordShape("positive producer"))?;
        let boundary = VaccIntegrationBoundaryAir {
            opening_bus: fixture.swirl_opening_bus,
            history_bus: fixture.history_bus,
            log_message_len: fixture.module.profile.log_message_len,
        };
        let compression_source = HistoryCompressionSourceAir(fixture.history_compress_bus);
        let compression = history_compression_trace(&producer_trace.compression_inputs);

        check_constraints::<_, NativeSC>(
            &fixture.producer_air,
            "WarpReplayProducerAirV19",
            &None,
            &[producer_trace.matrix.as_view()],
            &[],
        );
        check_constraints::<_, NativeSC>(
            &boundary,
            "VaccIntegrationBoundaryAir",
            &None,
            &[producer_trace.matrix.as_view()],
            &[],
        );
        check_constraints::<_, NativeSC>(
            &compression_source,
            "HistoryCompressionSourceAir",
            &None,
            &[compression.as_view()],
            &[],
        );

        let mut names = verifier_airs
            .iter()
            .map(|air| air.name())
            .collect::<Vec<_>>();
        names.extend([
            "WarpReplayProducerAirV19".to_string(),
            "VaccIntegrationBoundaryAir".to_string(),
            "HistoryCompressionSourceAir".to_string(),
        ]);
        let mut interactions = verifier_airs
            .iter()
            .map(|air| symbolic_interactions_dyn(air.as_ref()))
            .collect::<Vec<_>>();
        interactions.extend([
            symbolic_interactions(&fixture.producer_air),
            symbolic_interactions(&boundary),
            symbolic_interactions(&compression_source),
        ]);
        let mut matrices = verifier_trace
            .air_matrices()
            .iter()
            .map(|matrix| vec![matrix.as_view()])
            .collect::<Vec<_>>();
        matrices.extend([
            vec![producer_trace.matrix.as_view()],
            vec![producer_trace.matrix.as_view()],
            vec![compression.as_view()],
        ]);
        let public_values = (0..matrices.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        let mut preprocessed_owned = verifier_airs
            .iter()
            .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
            .collect::<Vec<_>>();
        preprocessed_owned.extend((0..3).map(|_| None));
        let preprocessed = preprocessed_owned
            .iter()
            .map(|matrix| matrix.as_ref().map(RowMajorMatrix::as_view))
            .collect::<Vec<_>>();
        check_logup(
            &names,
            &interactions,
            &preprocessed,
            &matrices,
            &public_values,
        );
        Ok(())
    }

    fn check_composed_vacc_batch_trace(
        fixture: &Fixture,
        verifier_trace: &DirectAirVaccVerifierBatchTraceV19,
        producers: &[DirectAirVaccProducerRecordV19],
    ) {
        let verifier_airs = fixture.module.airs::<NativeSC>();
        for (air, matrix) in verifier_airs.iter().zip(verifier_trace.air_matrices()) {
            check_air_ref(air, matrix);
        }
        let producer_trace = generate_warp_replay_producer_trace_v19(
            &fixture.producer_air,
            producers,
            producers.len().next_power_of_two(),
        )
        .expect("batched positive producer");
        let boundary = VaccIntegrationBoundaryAir {
            opening_bus: fixture.swirl_opening_bus,
            history_bus: fixture.history_bus,
            log_message_len: fixture.module.profile.log_message_len,
        };
        let compression_source = HistoryCompressionSourceAir(fixture.history_compress_bus);
        let compression = history_compression_trace(&producer_trace.compression_inputs);
        check_constraints::<_, NativeSC>(
            &fixture.producer_air,
            "WarpReplayProducerAirV19",
            &None,
            &[producer_trace.matrix.as_view()],
            &[],
        );
        check_constraints::<_, NativeSC>(
            &boundary,
            "VaccIntegrationBoundaryAir",
            &None,
            &[producer_trace.matrix.as_view()],
            &[],
        );
        check_constraints::<_, NativeSC>(
            &compression_source,
            "HistoryCompressionSourceAir",
            &None,
            &[compression.as_view()],
            &[],
        );

        let mut names = verifier_airs
            .iter()
            .map(|air| air.name())
            .collect::<Vec<_>>();
        names.extend([
            "WarpReplayProducerAirV19".to_string(),
            "VaccIntegrationBoundaryAir".to_string(),
            "HistoryCompressionSourceAir".to_string(),
        ]);
        let mut interactions = verifier_airs
            .iter()
            .map(|air| symbolic_interactions_dyn(air.as_ref()))
            .collect::<Vec<_>>();
        interactions.extend([
            symbolic_interactions(&fixture.producer_air),
            symbolic_interactions(&boundary),
            symbolic_interactions(&compression_source),
        ]);
        let mut matrices = verifier_trace
            .air_matrices()
            .iter()
            .map(|matrix| vec![matrix.as_view()])
            .collect::<Vec<_>>();
        matrices.extend([
            vec![producer_trace.matrix.as_view()],
            vec![producer_trace.matrix.as_view()],
            vec![compression.as_view()],
        ]);
        let public_values = (0..matrices.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        let mut preprocessed_owned = verifier_airs
            .iter()
            .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
            .collect::<Vec<_>>();
        preprocessed_owned.extend((0..3).map(|_| None));
        let preprocessed = preprocessed_owned
            .iter()
            .map(|matrix| matrix.as_ref().map(RowMajorMatrix::as_view))
            .collect::<Vec<_>>();
        check_logup(
            &names,
            &interactions,
            &preprocessed,
            &matrices,
            &public_values,
        );
    }

    fn two_record_batch(
        fixture: &Fixture,
        second: &Fixture,
    ) -> (
        Vec<DirectAirVaccProducerRecordV19>,
        DirectAirVaccVerifierBatchTraceV19,
    ) {
        let first_producer = fixture.producer.clone();
        let mut second_producer = second.producer.clone();
        second_producer.proof_index = 1;
        let producers = vec![first_producer, second_producer];
        let records = [
            DirectAirVaccVerifierRecordV19 {
                producer: &producers[0],
                verification: &fixture.verification,
                transcript: &fixture.transcript,
                prior: fixture.prior.as_ref(),
            },
            DirectAirVaccVerifierRecordV19 {
                producer: &producers[1],
                verification: &second.verification,
                transcript: &second.transcript,
                prior: second.prior.as_ref(),
            },
        ];
        let trace = fixture
            .module
            .generate_traces(&fixture.config, &records)
            .expect("honest two-record VACC batch");
        (producers, trace)
    }

    #[test]
    fn honest_two_record_same_shape_vacc_batch_is_constrained() {
        let first = fixture();
        let second = fixture();
        let (producers, trace) = two_record_batch(&first, &second);
        assert_eq!(trace.output_instances.len(), 2);
        assert_eq!(trace.traces.len(), first.module.airs::<NativeSC>().len());
        check_composed_vacc_batch_trace(&first, &trace, &producers);
    }

    #[test]
    fn prior_group_replay_certifies_local_zero_as_global_one() {
        let mut fixture = fixture_with_prior(true);
        fixture.producer_air.setup_schedule = Some(
            crate::circuit::native_warp_history_v19::WarpReplayProducerScheduleV19::new(1, 1, true)
                .expect("one-record prior schedule"),
        );

        // The heavy verifier remains in its setup-defined dense local
        // namespace. Only the positive History producer carries the global
        // batch index and performs the constrained remap at its input buses.
        let local = fixture.producer.clone();
        assert_eq!(local.proof_index, 0);
        let verifier_record = DirectAirVaccVerifierRecordV19 {
            producer: &local,
            verification: &fixture.verification,
            transcript: &fixture.transcript,
            prior: fixture.prior.as_ref(),
        };
        let verifier_trace = fixture
            .module
            .generate_traces(&fixture.config, &[verifier_record])
            .expect("local prior verifier trace");

        let mut global = local;
        global.proof_index = 1;
        check_composed_vacc_batch_trace(&fixture, &verifier_trace, &[global]);
    }

    #[test]
    fn statement_allows_all_zero_systematic_opening_point() {
        let fixture = fixture();
        let mut producer = fixture.producer.clone();
        producer.opening_point.fill([F::ZERO; D_EF]);
        producer.fresh_alpha.fill([F::ZERO; D_EF]);
        let trace = generate_direct_v19_statement_trace(&producer)
            .expect("all-zero systematic opening statement");
        let air = DirectAirVaccStatementAirV19 {
            profile: fixture.module.profile.clone(),
            protocol_bus: fixture.module.protocol_bus,
            end_bus: fixture.module.end_bus,
            root_bus: fixture.module.statement_root_bus,
            digest_bus: fixture.module.statement_digest_bus,
            claim_bus: fixture.module.buses.claim_value,
            context_bus: fixture.module.history_buses.context,
            vacc_input_bus: fixture.module.history_buses.input,
            vacc_input_lookup_count: 1,
        };
        check_constraints::<_, NativeSC>(
            &air,
            "DirectAirVaccStatementAirV19",
            &None,
            &[trace.as_view()],
            &[],
        );
    }

    #[test]
    fn shift_schedule_accepts_minus_one_raw_sample() {
        let fixture = fixture();
        let log_codeword_len = fixture.module.profile.log_codeword_len;
        let mask = (1u32 << log_codeword_len) - 1;
        let sample = -F::ONE;
        let index = sample.as_canonical_u32() & mask;
        let trace = generate_direct_air_vacc_shift_schedule_trace_v19(
            0,
            &[sample],
            &[17],
            &[index],
            1,
            log_codeword_len,
            None,
        )
        .expect("masked -1 shift sample");
        let air = DirectAirVaccShiftScheduleAirV19 {
            inner: NativeShiftScheduleAir {
                transcript_bus: fixture.module.buses.vacc_semantic_transcript,
                exp_bits_len_bus: fixture.module.shared.exp_bits_len_bus,
                right_shift_bus: fixture.module.shared.right_shift_bus,
                shift_index_bus: fixture.module.buses.shift_index,
                opening_bus: Some(fixture.module.buses.opening_claim),
                log_codeword_len,
                opening_claim_offset: 1 + fixture.module.profile.num_ood,
            },
            role_bus: fixture.module.buses.vacc_transcript_role,
        };
        check_constraints::<_, NativeSC>(
            &air,
            "DirectAirVaccShiftScheduleAirV19",
            &None,
            &[trace.as_view()],
            &[],
        );
    }

    #[test]
    fn direct_shift_schedule_accepts_log29_backend_mask() {
        let fixture = fixture();
        let log_codeword_len = 29;
        let sample = F::from_u32(1_500_000_000);
        let index = sample.as_canonical_u32() & ((1u32 << log_codeword_len) - 1);
        let trace = generate_direct_air_vacc_shift_schedule_trace_v19(
            0,
            &[sample],
            &[23],
            &[index],
            1,
            log_codeword_len,
            None,
        )
        .expect("log-29 backend shift sample");
        let air = DirectAirVaccShiftScheduleAirV19 {
            inner: NativeShiftScheduleAir {
                transcript_bus: fixture.module.buses.vacc_semantic_transcript,
                exp_bits_len_bus: fixture.module.shared.exp_bits_len_bus,
                right_shift_bus: fixture.module.shared.right_shift_bus,
                shift_index_bus: fixture.module.buses.shift_index,
                opening_bus: Some(fixture.module.buses.opening_claim),
                log_codeword_len,
                opening_claim_offset: 1 + fixture.module.profile.num_ood,
            },
            role_bus: fixture.module.buses.vacc_transcript_role,
        };
        check_constraints::<_, NativeSC>(
            &air,
            "DirectAirVaccShiftScheduleAirV19",
            &None,
            &[trace.as_view()],
            &[],
        );
    }

    #[test]
    fn cross_spliced_dynamic_proof_index_is_rejected() {
        let first = fixture();
        let second = fixture();
        let (producers, mut trace) = two_record_batch(&first, &second);
        let airs = first.module.airs::<NativeSC>();
        let omega_index = airs
            .iter()
            .position(|air| air.name() == "DirectAirVaccTwinOmegaAirV19")
            .expect("twin omega AIR");
        let width = NativeTwinOmegaCols::<F>::width();
        let second_row = &mut trace.traces[omega_index].values[width..2 * width];
        let cols: &mut NativeTwinOmegaCols<F> = second_row.borrow_mut();
        assert_eq!(cols.proof_idx, F::ONE);
        cols.proof_idx = F::ZERO;
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_composed_vacc_batch_trace(&first, &trace, &producers);
        }))
        .is_err());
    }

    #[test]
    fn cross_spliced_claim_transcript_id_is_rejected() {
        let first = fixture();
        let second = fixture();
        let (producers, mut trace) = two_record_batch(&first, &second);
        let airs = first.module.airs::<NativeSC>();
        let claim_index = airs
            .iter()
            .position(|air| air.name() == "NativeStandardClaimValueAir")
            .expect("standard claim AIR");
        let width = NativeClaimValueCols::<F>::width();
        let mut selected = None;
        for row in trace.traces[claim_index].values.chunks_exact_mut(width) {
            let cols: &NativeClaimValueCols<F> = (&*row).borrow();
            if cols.active == F::ONE && cols.proof_idx == F::ONE && cols.is_transcript == F::ONE {
                selected = Some(row);
                break;
            }
        }
        let row: &mut NativeClaimValueCols<F> = selected
            .expect("second proof transcript-backed claim")
            .borrow_mut();
        assert_eq!(row.transcript_id, F::ONE);
        row.transcript_id = F::ZERO;
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_composed_vacc_batch_trace(&first, &trace, &producers);
        }))
        .is_err());
    }

    #[test]
    fn cursor_rejects_a_locally_well_formed_phase_skip() {
        let first = fixture();
        let second = fixture();
        let (producers, mut trace) = two_record_batch(&first, &second);
        let airs = first.module.airs::<NativeSC>();
        let cursor_index = airs
            .iter()
            .position(|air| air.name() == "DirectAirVaccTranscriptCursorAirV19")
            .expect("VACC cursor AIR");
        let width = DirectAirVaccTranscriptCursorColsV19::<F>::width();
        let row = trace.traces[cursor_index]
            .values
            .chunks_exact_mut(width)
            .nth(DIGEST_SIZE)
            .expect("first fresh-alpha cursor row");
        let cols: &mut DirectAirVaccTranscriptCursorColsV19<F> = row.borrow_mut();
        assert_eq!(cols.role, F::from_usize(VACC_ROLE_FRESH_ALPHA));
        cols.role_flags[VACC_ROLE_FRESH_ALPHA] = F::ZERO;
        cols.role_flags[VACC_ROLE_FRESH_MU] = F::ONE;
        cols.role = F::from_usize(VACC_ROLE_FRESH_MU);
        cols.ordinal = F::ZERO;
        cols.is_first_ordinal = F::ONE;
        cols.is_last_ordinal = F::ONE;
        cols.ordinal_inverse = F::ZERO;
        cols.last_ordinal_inverse = F::ZERO;
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_composed_vacc_batch_trace(&first, &trace, &producers);
        }))
        .is_err());
    }

    #[test]
    fn every_dynamic_cursor_role_is_demanded_by_the_real_vacc_airs() {
        // A continuation contains every optional-prior role as well as every
        // fresh, sumcheck, opening, output-root, and final-mu role.  Restrict
        // LogUp to the semantic role bus so this test specifically audits the
        // typed consumers on which dynamic role lengths rely.
        let fixture = fixture_with_prior(true);
        let record = DirectAirVaccVerifierRecordV19 {
            producer: &fixture.producer,
            verification: &fixture.verification,
            transcript: &fixture.transcript,
            prior: fixture.prior.as_ref(),
        };
        let trace = fixture
            .module
            .generate_trace(&fixture.config, &record)
            .expect("honest prior-bearing VACC trace");
        let airs = fixture.module.airs::<NativeSC>();
        let cursor_index = airs
            .iter()
            .position(|air| air.name() == "DirectAirVaccTranscriptCursorAirV19")
            .expect("VACC cursor AIR");
        let role_bus = fixture.module.buses.vacc_transcript_role.index();
        let names = airs.iter().map(|air| air.name()).collect::<Vec<_>>();
        let interactions = airs
            .iter()
            .map(|air| {
                symbolic_interactions_dyn(air.as_ref())
                    .into_iter()
                    .filter(|interaction| interaction.bus_index == role_bus)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let preprocessed_owned = airs
            .iter()
            .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
            .collect::<Vec<_>>();
        let preprocessed = preprocessed_owned
            .iter()
            .map(|matrix| matrix.as_ref().map(RowMajorMatrix::as_view))
            .collect::<Vec<_>>();
        let public_values = (0..airs.len()).map(|_| Vec::new()).collect::<Vec<_>>();
        let honest_matrices = trace
            .traces
            .iter()
            .map(|matrix| vec![matrix.as_view()])
            .collect::<Vec<_>>();
        check_logup(
            &names,
            &interactions,
            &preprocessed,
            &honest_matrices,
            &public_values,
        );

        let cursor_width = DirectAirVaccTranscriptCursorColsV19::<F>::width();
        for role in 0..VACC_ROLE_COUNT {
            let mut mutated = trace
                .traces
                .iter()
                .map(|matrix| matrix.clone())
                .collect::<Vec<_>>();
            let row = mutated[cursor_index]
                .values
                .chunks_exact_mut(cursor_width)
                .find(|row| {
                    let cols: &DirectAirVaccTranscriptCursorColsV19<F> = (&**row).borrow();
                    cols.active == F::ONE && cols.role == F::from_usize(role)
                })
                .unwrap_or_else(|| panic!("continuation fixture omitted VACC role {role}"));
            let cols: &mut DirectAirVaccTranscriptCursorColsV19<F> = row.borrow_mut();
            // This is the adversarial dynamic-length move: end/omit a role
            // while leaving all verifier algebra tables unchanged.  The real
            // typed consumer for that role must make the permutation fail.
            cols.active = F::ZERO;
            let matrices = mutated
                .iter()
                .map(|matrix| vec![matrix.as_view()])
                .collect::<Vec<_>>();
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    check_logup(
                        &names,
                        &interactions,
                        &preprocessed,
                        &matrices,
                        &public_values,
                    );
                }))
                .is_err(),
                "truncating dynamic VACC role {role} was not rejected"
            );
        }

        // In particular, continuation cannot take the bootstrap-only skip of
        // the complete prior interval.
        let mut skipped_prior = trace.traces.clone();
        for row in skipped_prior[cursor_index]
            .values
            .chunks_exact_mut(cursor_width)
        {
            let cols: &mut DirectAirVaccTranscriptCursorColsV19<F> = row.borrow_mut();
            if cols.active == F::ONE
                && (VACC_ROLE_PRIOR_ROOT..=VACC_ROLE_PRIOR_ETA)
                    .contains(&(cols.role.as_canonical_u32() as usize))
            {
                cols.active = F::ZERO;
            }
        }
        let skipped_matrices = skipped_prior
            .iter()
            .map(|matrix| vec![matrix.as_view()])
            .collect::<Vec<_>>();
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_logup(
                &names,
                &interactions,
                &preprocessed,
                &skipped_matrices,
                &public_values,
            );
        }))
        .is_err());
    }

    #[test]
    fn reduced_optional_prior_has_a_constrained_empty_marker_but_output_cannot_be_empty() {
        let fixture = fixture();
        let layout = fixture.module.private_accumulator.clone();
        let traces = empty_v19_accumulator_digest_traces(&layout);
        let prior_value_air = DirectAirAccumulatorValueAirV19 {
            mode: NativeAccumulatorBindingMode::Prior,
            allow_empty: true,
            layout: layout.clone(),
            claim_source: fixture.module.input_arity() - 1,
            digest_element_bus: fixture.module.buses.accumulator_digest_element,
            claim_bus: fixture.module.buses.claim_value,
            batching_bus: fixture.module.buses.batching_output,
            folded_bus: fixture.module.buses.folded_claim,
            twin_bus: fixture.module.buses.twin_scalar,
        };
        let prior_hash_air = NativeAccumulatorHashAir {
            state: 0,
            allow_empty: true,
            alpha_len: layout.alpha_len,
            beta_len: layout.beta_len,
            digest_element_bus: fixture.module.buses.accumulator_digest_element,
            algebraic_digest_bus: fixture.module.buses.accumulator_algebraic_digest,
            permute_bus: fixture.module.shared.poseidon2_permute_bus,
            compress_bus: fixture.module.shared.poseidon2_compress_bus,
        };
        let prior_root_air = DirectAirAccumulatorRootDigestAirV19 {
            mode: NativeAccumulatorBindingMode::Prior,
            allow_empty: true,
            root_bus: fixture.module.buses.accumulator_root,
            algebraic_digest_bus: fixture.module.buses.accumulator_algebraic_digest,
            compress_bus: fixture.module.shared.poseidon2_compress_bus,
            digest_bus: fixture.module.statement_digest_bus,
            certified_digest_bus: None,
        };
        check_constraints::<_, NativeSC>(
            &prior_value_air,
            "reduced optional prior value marker",
            &None,
            &[traces.values.as_view()],
            &[],
        );
        check_constraints::<_, NativeSC>(
            &prior_hash_air,
            "reduced optional prior hash marker",
            &None,
            &[traces.hash.as_view()],
            &[],
        );
        check_constraints::<_, NativeSC>(
            &prior_root_air,
            "reduced optional prior root marker",
            &None,
            &[traces.root.as_view()],
            &[],
        );

        let mut malformed = traces.values.clone();
        let first: &mut NativeAccumulatorValueCols<F> =
            malformed.values[..NativeAccumulatorValueCols::<F>::width()].borrow_mut();
        first.section[0] = F::ZERO;
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_constraints::<_, NativeSC>(
                &prior_value_air,
                "malformed reduced optional prior marker",
                &None,
                &[malformed.as_view()],
                &[],
            );
        }))
        .is_err());

        let output_value_air = DirectAirAccumulatorValueAirV19 {
            mode: NativeAccumulatorBindingMode::Output,
            allow_empty: false,
            layout,
            claim_source: fixture.module.input_arity() - 1,
            digest_element_bus: fixture.module.buses.accumulator_digest_element,
            claim_bus: fixture.module.buses.claim_value,
            batching_bus: fixture.module.buses.batching_output,
            folded_bus: fixture.module.buses.folded_claim,
            twin_bus: fixture.module.buses.twin_scalar,
        };
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_constraints::<_, NativeSC>(
                &output_value_air,
                "empty reduced output accumulator",
                &None,
                &[traces.values.as_view()],
                &[],
            );
        }))
        .is_err());
    }

    #[test]
    fn relation_catalog_counts_only_the_selected_relation() {
        let mut a = fixture_for_air_id(false, 9);
        install_relation_catalog(&mut a, &[9, 10]);
        let record = DirectAirVaccVerifierRecordV19 {
            producer: &a.producer,
            verification: &a.verification,
            transcript: &a.transcript,
            prior: None,
        };
        let trace = a
            .module
            .generate_trace(&a.config, &record)
            .expect("catalog A-only trace");
        let airs = a.module.airs::<NativeSC>();
        let catalog_index = airs
            .iter()
            .position(|air| air.name() == "NativeStandardVaccPrefixCatalogAirV19")
            .expect("prefix catalog AIR");
        let active_multiplicities = trace.traces[catalog_index]
            .values
            .iter()
            .filter(|&&multiplicity| multiplicity == F::ONE)
            .count();
        let a_event_count = a
            .module
            .prefix_events
            .iter()
            .filter(|event| event.relation_digest == a.producer.relation_digest)
            .count();
        assert_eq!(active_multiplicities, a_event_count);
        check_composed_vacc(&a, &a.producer, &a.verification, &a.transcript)
            .expect("catalog A-only composition");
    }

    #[test]
    fn relation_catalog_batches_a_and_b() {
        let mut a = fixture_for_air_id(false, 9);
        let one_relation_air_count = a.module.airs::<NativeSC>().len();
        install_relation_catalog(&mut a, &[9, 10]);
        assert_eq!(
            a.module.airs::<NativeSC>().len(),
            one_relation_air_count,
            "relation count must not change the verifier AIR count"
        );
        let b = fixture_for_air_id(false, 10);
        let (producers, trace) = two_record_batch(&a, &b);
        let airs = a.module.airs::<NativeSC>();
        assert_eq!(
            airs.iter()
                .filter(|air| {
                    matches!(
                        air.name().as_str(),
                        "NativeStandardVaccPrefixCatalogAirV19"
                            | "NativeStandardVaccPrefixStreamAirV19"
                    )
                })
                .count(),
            2,
            "prefix AIR count is constant in the relation count"
        );
        let stream_index = airs
            .iter()
            .position(|air| air.name() == "NativeStandardVaccPrefixStreamAirV19")
            .unwrap();
        let width = NativeStandardVaccPrefixStreamColsV19::<F>::width();
        let active_proofs = trace.traces[stream_index]
            .values
            .chunks_exact(width)
            .filter(|row| row[0] == F::ONE && row[1] == F::ONE)
            .count();
        assert_eq!(active_proofs, 2);
        check_composed_vacc_batch_trace(&a, &trace, &producers);
    }

    fn assert_prefix_stream_matches_legacy_and_recorded_transcript(
        module: &DirectAirVaccVerifierModuleV19,
        trace: &DirectAirVaccVerifierBatchTraceV19,
        records: &[DirectAirVaccVerifierRecordV19<'_>],
    ) {
        let airs = module.airs::<NativeSC>();
        let stream_index = airs
            .iter()
            .position(|air| air.name() == "NativeStandardVaccPrefixStreamAirV19")
            .expect("prefix stream AIR");
        let width = NativeStandardVaccPrefixStreamColsV19::<F>::width();
        let rows = trace.traces[stream_index]
            .values
            .chunks_exact(width)
            .map(|row| row.borrow())
            .collect::<Vec<&NativeStandardVaccPrefixStreamColsV19<F>>>();

        for (proof_idx, record) in records.iter().enumerate() {
            let schedule = DirectAirVaccTranscriptScheduleV19::from_record(module, record)
                .expect("canonical legacy VACC schedule");
            let legacy = generate_native_standard_vacc_prefix_trace(
                proof_idx,
                schedule.start_tidx,
                record.producer.segment_index,
                module.include_prior,
            );
            let legacy: &NativeStandardVaccPrefixCols<F> = legacy.values.as_slice().borrow();
            let proof_rows = rows
                .iter()
                .copied()
                .filter(|row| row.active == F::ONE && row.proof_idx == F::from_usize(proof_idx))
                .collect::<Vec<_>>();
            assert!(!proof_rows.is_empty());
            let first = proof_rows[0];
            assert_eq!(first.start_tidx, legacy.start_tidx);
            assert_eq!(first.segment_index_lo, legacy.segment_index_lo);
            assert_eq!(first.segment_index_hi, legacy.segment_index_hi);
            assert_eq!(first.has_prior, legacy.has_prior);
            assert_eq!(first.relation_digest, record.producer.relation_digest);

            for (ordinal, row) in proof_rows.iter().enumerate() {
                assert_eq!(row.ordinal, F::from_usize(ordinal));
                let tidx = schedule.start_tidx + D_EF * ordinal;
                assert_eq!(record.transcript.values()[tidx], row.event_value);
                assert_eq!(
                    &record.transcript.values()[tidx + 1..tidx + D_EF],
                    &[F::ZERO; D_EF - 1],
                    "every legacy prefix byte is observed as [byte, 0, 0, 0]"
                );
            }
            assert_eq!(
                schedule.start_tidx + D_EF * proof_rows.len(),
                schedule.fresh_root_tidx,
                "phase-cursor boundary 1 is byte-for-byte identical to the legacy prefix end"
            );
            assert_eq!(proof_rows.last().unwrap().is_last, F::ONE);
        }
    }

    #[test]
    fn compact_prefix_is_differentially_identical_for_relations_and_prior_modes() {
        let mut a = fixture_for_air_id(false, 9);
        install_relation_catalog(&mut a, &[9, 10]);
        let b = fixture_for_air_id(false, 10);
        let (producers, trace) = two_record_batch(&a, &b);
        let records = [
            DirectAirVaccVerifierRecordV19 {
                producer: &producers[0],
                verification: &a.verification,
                transcript: &a.transcript,
                prior: None,
            },
            DirectAirVaccVerifierRecordV19 {
                producer: &producers[1],
                verification: &b.verification,
                transcript: &b.transcript,
                prior: None,
            },
        ];
        assert_prefix_stream_matches_legacy_and_recorded_transcript(&a.module, &trace, &records);

        let prior = fixture_for_air_id(true, 9);
        let prior_record = DirectAirVaccVerifierRecordV19 {
            producer: &prior.producer,
            verification: &prior.verification,
            transcript: &prior.transcript,
            prior: prior.prior.as_ref(),
        };
        let prior_trace = prior
            .module
            .generate_traces(&prior.config, core::slice::from_ref(&prior_record))
            .expect("prior compact prefix trace");
        assert_prefix_stream_matches_legacy_and_recorded_transcript(
            &prior.module,
            &prior_trace,
            core::slice::from_ref(&prior_record),
        );
    }

    #[test]
    fn relation_catalog_rejects_unknown_digest_in_host_and_air() {
        let mut a = fixture_for_air_id(false, 9);
        install_relation_catalog(&mut a, &[9, 10]);

        let mut unknown = a.producer.clone();
        unknown.relation_digest = [F::from_u32(211); DIGEST_SIZE];
        let unknown_relation_digest = unknown.relation_digest;
        let unknown_record = DirectAirVaccVerifierRecordV19 {
            producer: &unknown,
            verification: &a.verification,
            transcript: &a.transcript,
            prior: None,
        };
        assert!(a.module.generate_trace(&a.config, &unknown_record).is_err());

        let producers = vec![a.producer.clone()];
        let records = [DirectAirVaccVerifierRecordV19 {
            producer: &producers[0],
            verification: &a.verification,
            transcript: &a.transcript,
            prior: None,
        }];
        let mut trace = a.module.generate_traces(&a.config, &records).unwrap();
        let airs = a.module.airs::<NativeSC>();
        let statement_index = airs
            .iter()
            .position(|air| air.name() == "DirectAirVaccStatementAirV19")
            .expect("VACC statement AIR");
        let row: &mut DirectAirVaccStatementColsV19<F> = trace.traces[statement_index]
            .values
            .as_mut_slice()
            .borrow_mut();
        row.inner.relation_digest = unknown.relation_digest;
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_composed_vacc_batch_trace(&a, &trace, &[unknown]);
        }))
        .is_err());

        let producers = vec![a.producer.clone()];
        let records = [DirectAirVaccVerifierRecordV19 {
            producer: &producers[0],
            verification: &a.verification,
            transcript: &a.transcript,
            prior: None,
        }];
        let mut trace = a.module.generate_traces(&a.config, &records).unwrap();
        let stream_index = airs
            .iter()
            .position(|air| air.name() == "NativeStandardVaccPrefixStreamAirV19")
            .unwrap();
        let row: &mut NativeStandardVaccPrefixStreamColsV19<F> = trace.traces[stream_index]
            .values
            .chunks_exact_mut(NativeStandardVaccPrefixStreamColsV19::<F>::width())
            .next()
            .unwrap()
            .borrow_mut();
        row.relation_digest = unknown_relation_digest;
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_composed_vacc_batch_trace(&a, &trace, &producers);
        }))
        .is_err());
    }

    #[test]
    fn relation_catalog_rejects_prefix_event_tampering() {
        let fixture = fixture();
        let producers = vec![fixture.producer.clone()];
        let records = [DirectAirVaccVerifierRecordV19 {
            producer: &producers[0],
            verification: &fixture.verification,
            transcript: &fixture.transcript,
            prior: None,
        }];
        let mut trace = fixture
            .module
            .generate_traces(&fixture.config, &records)
            .unwrap();
        let airs = fixture.module.airs::<NativeSC>();
        let stream_index = airs
            .iter()
            .position(|air| air.name() == "NativeStandardVaccPrefixStreamAirV19")
            .unwrap();
        let row: &mut NativeStandardVaccPrefixStreamColsV19<F> = trace.traces[stream_index]
            .values
            .chunks_exact_mut(NativeStandardVaccPrefixStreamColsV19::<F>::width())
            .next()
            .unwrap()
            .borrow_mut();
        assert_eq!(row.is_step, F::ZERO);
        row.event_value += F::ONE;
        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_composed_vacc_batch_trace(&fixture, &trace, &producers);
        }))
        .is_err());
    }

    #[test]
    fn honest_degree_one_canonical_zero_vacc_trace_is_constrained() {
        let fixture = fixture();
        let record = DirectAirVaccVerifierRecordV19 {
            producer: &fixture.producer,
            verification: &fixture.verification,
            transcript: &fixture.transcript,
            prior: fixture.prior.as_ref(),
        };
        let trace = fixture
            .module
            .generate_trace(&fixture.config, &record)
            .expect("real recorded VACC trace");
        let airs = fixture.module.airs::<NativeSC>();
        assert_eq!(airs.len(), trace.traces.len());
        for (air, matrix) in airs.iter().zip(&trace.traces) {
            check_air_ref(air, matrix);
        }
        assert!(!trace.poseidon_permutation_inputs.is_empty());
        assert!(!trace.poseidon_compression_inputs.is_empty());
        check_composed_vacc(
            &fixture,
            &fixture.producer,
            &fixture.verification,
            &fixture.transcript,
        )
        .expect("balanced real VACC verifier and producer composition");
    }

    #[test]
    fn exact_vacc_suffix_resume_preserves_the_transcript_anchor() {
        let fixture = fixture();
        let start = fixture.producer.start_checkpoint.operation_index as usize;
        let suffix = exact_v19_resumable_suffix(
            &fixture.transcript,
            start,
            fixture.producer.start_checkpoint.state,
        )
        .expect("exact source/VACC checkpoint must be resumable");
        assert_eq!(suffix.values(), &fixture.transcript.values()[start..]);
        assert_eq!(suffix.samples(), &fixture.transcript.samples()[start..]);
        assert_eq!(
            suffix.perm_results().first().copied(),
            Some(fixture.producer.start_checkpoint.state)
        );
        assert_eq!(
            suffix
                .events()
                .first()
                .map(|event| event.operation_range.start),
            Some(0)
        );
        assert!(
            !suffix.samples()[0],
            "VACC suffix must begin with an absorb"
        );

        let mut wrong_state = fixture.producer.start_checkpoint.state;
        wrong_state[0] += F::ONE;
        assert!(exact_v19_resumable_suffix(&fixture.transcript, start, wrong_state).is_err());
        assert!(exact_v19_resumable_suffix(
            &fixture.transcript,
            start + 1,
            fixture.producer.start_checkpoint.state,
        )
        .is_err());

        let mut starts_with_sample = fixture.transcript.clone();
        starts_with_sample.samples_mut()[start] = true;
        assert!(exact_v19_resumable_suffix(
            &starts_with_sample,
            start,
            fixture.producer.start_checkpoint.state,
        )
        .is_err());
    }

    fn composed_rejects(
        fixture: &Fixture,
        producer: &DirectAirVaccProducerRecordV19,
        verification: &DirectAirVaccVerificationV19,
        transcript: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    ) -> bool {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            check_composed_vacc(fixture, producer, verification, transcript)
        }))
        .map_or(true, |result| result.is_err())
    }

    #[test]
    fn tampered_vacc_algebra_and_fresh_merkle_path_are_rejected() {
        let fixture = fixture();

        let mut twin = fixture.verification.clone();
        twin.twin.sumcheck.rounds[0].coefficients[0] += EF::ONE;
        assert!(composed_rejects(
            &fixture,
            &fixture.producer,
            &twin,
            &fixture.transcript
        ));

        let mut batching = fixture.verification.clone();
        batching.batching.sumcheck.rounds[0].coefficients[0] += EF::ONE;
        assert!(composed_rejects(
            &fixture,
            &fixture.producer,
            &batching,
            &fixture.transcript
        ));

        let mut merkle = fixture.verification.clone();
        let compression = merkle.fresh_authentication[0]
            .multiproof
            .compressions
            .first_mut()
            .expect("nontrivial fresh authentication path");
        compression.left[0] += F::ONE;
        assert!(composed_rejects(
            &fixture,
            &fixture.producer,
            &merkle,
            &fixture.transcript
        ));
    }

    #[test]
    fn honest_continuation_and_tampered_prior_merkle_path() {
        let fixture = fixture_with_prior(true);
        check_composed_vacc(
            &fixture,
            &fixture.producer,
            &fixture.verification,
            &fixture.transcript,
        )
        .expect("balanced continuation VACC composition");

        let mut merkle = fixture.verification.clone();
        let compression = merkle
            .prior_authentication
            .as_mut()
            .expect("continuation prior authentication")
            .multiproof
            .compressions
            .first_mut()
            .expect("nontrivial prior authentication path");
        compression.right[0] += F::ONE;
        assert!(composed_rejects(
            &fixture,
            &fixture.producer,
            &merkle,
            &fixture.transcript
        ));
    }

    #[test]
    fn tampered_vacc_statement_and_checkpoint_are_rejected() {
        let fixture = fixture();

        let mut alpha_prefix = fixture.producer.clone();
        alpha_prefix.fresh_alpha[0][0] += F::ONE;
        assert!(composed_rejects(
            &fixture,
            &alpha_prefix,
            &fixture.verification,
            &fixture.transcript
        ));

        let mut alpha_suffix = fixture.producer.clone();
        alpha_suffix.fresh_alpha[fixture.module.profile.log_message_len][0] = F::ONE;
        assert!(composed_rejects(
            &fixture,
            &alpha_suffix,
            &fixture.verification,
            &fixture.transcript
        ));

        let mut mu = fixture.producer.clone();
        mu.fresh_mu[0] += F::ONE;
        assert!(composed_rejects(
            &fixture,
            &mu,
            &fixture.verification,
            &fixture.transcript
        ));

        let mut eta = fixture.producer.clone();
        eta.fresh_eta[0] = F::ONE;
        assert!(composed_rejects(
            &fixture,
            &eta,
            &fixture.verification,
            &fixture.transcript
        ));

        let mut relation = fixture.producer.clone();
        relation.relation_digest[0] += F::ONE;
        assert!(composed_rejects(
            &fixture,
            &relation,
            &fixture.verification,
            &fixture.transcript
        ));

        let mut checkpoint = fixture.producer.clone();
        checkpoint.start_checkpoint.state[0] += F::ONE;
        assert!(composed_rejects(
            &fixture,
            &checkpoint,
            &fixture.verification,
            &fixture.transcript
        ));

        let mut output_root = fixture.producer.clone();
        output_root.next_root[0] += F::ONE;
        assert!(composed_rejects(
            &fixture,
            &output_root,
            &fixture.verification,
            &fixture.transcript
        ));
    }

    #[test]
    fn appendix_d_prefix_catalog_is_cross_tagged_and_rejects_all_ef_records() {
        let fixture = fixture();
        let shape = fixture.module.profile.clone();
        let relation = fixture.module.relations[0].clone();
        let all_ef = standard_vacc_prefix_events_v19(
            &shape,
            &relation,
            false,
            DirectAirVaccFreshCommitmentModeV19::ScalarMerkle,
        )
        .unwrap();
        let appendix_d = standard_vacc_prefix_events_v19(
            &shape,
            &relation,
            false,
            DirectAirVaccFreshCommitmentModeV19::AppendixDBase,
        )
        .unwrap();
        let all_ef_bytes = all_ef
            .iter()
            .map(|event| event.catalog_value)
            .collect::<Vec<_>>();
        let appendix_d_bytes = appendix_d
            .iter()
            .map(|event| event.catalog_value)
            .collect::<Vec<_>>();
        assert!(all_ef_bytes.starts_with(NATIVE_WARP_VACC_PROTOCOL_TAG));
        assert!(appendix_d_bytes.starts_with(NATIVE_WARP_VACC_APPENDIX_D_PROTOCOL_TAG));
        assert_ne!(all_ef_bytes, appendix_d_bytes);

        let appendix_module = DirectAirVaccVerifierModuleV19::new_shape_batched_appendix_d(
            vec![fixture.profile.clone()],
            false,
            fixture.module.shared.clone(),
            fixture.module.buses.clone(),
            fixture.module.history_buses,
            fixture.module.protocol_bus,
            fixture.module.end_bus,
            fixture.module.statement_root_bus,
            fixture.module.statement_digest_bus,
            SystemParams::new_for_testing(10),
        )
        .unwrap();
        let all_ef_record = DirectAirVaccVerifierRecordV19 {
            producer: &fixture.producer,
            verification: &fixture.verification,
            transcript: &fixture.transcript,
            prior: None,
        };
        assert!(appendix_module
            .generate_traces(&fixture.config, core::slice::from_ref(&all_ef_record))
            .is_err());
        assert!(appendix_module
            .airs::<NativeSC>()
            .iter()
            .any(|air| air.name() == "NativeAppendixDCodewordProjectionAir"));
        assert!(!appendix_module
            .airs::<NativeSC>()
            .iter()
            .any(|air| air.name() == "NativeStandardCodewordProjectionAir"));
    }
}

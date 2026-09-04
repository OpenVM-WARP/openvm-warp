//! Fail-closed protocol-v19 LogUp/SWIRL verifier boundary.
//!
//! This module deliberately keeps three statements separate:
//!
//! 1. the segment-wide `LogUpOnly` GKR/batch-constraint equation;
//! 2. the ephemeral mapped source functional and its one-shot degree-two reduction to one raw
//!    systematic-message opening; and
//! 3. the VM semantics read from verifier-authenticated direct-AIR public values.
//!
//! In particular, `PROGRAM_AIR_ID` has no public values.  Stable program
//! identity is a fingerprint of the cached Program columns in the *same raw
//! Program source*.  A transcript-derived row point and column challenge fix
//! that fingerprint functional.  A fresh transcript challenge combines it
//! with the ordinary SWIRL functional before the one-shot reduction.  The
//! resulting single raw opening is what WARP carries and terminal WHIR opens.
//! No source root, beta challenge, or host `VmSegmentMetadata` is accepted as
//! a substitute for this statement.
//!
//! The input buses below are permutation buses on purpose.  They must be sent
//! by the real recursive GKR/batch endpoint evaluator, source-forest verifier,
//! and mapped-functional evaluator in the same multi-AIR proof.  Omitting one
//! of those providers leaves LogUp unbalanced and makes the composition fail
//! closed; there is no positive copy-source AIR in production code.

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit::{
    arch::{BOUNDARY_AIR_ID, CONNECTOR_AIR_ID, MERKLE_AIR_ID, PROGRAM_AIR_ID},
    system::connector::DEFAULT_SUSPEND_EXIT_CODE,
};
use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit::bus::TranscriptBus;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, LookupBus, PermutationCheckBus},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, DIGEST_SIZE, D_EF, EF, F,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::{
    CertifiedLogUpOnlyEndpointBusV19, CertifiedLogUpOnlyEndpointMessageV19,
    CertifiedSwirlRawOpeningBusV19, CertifiedSwirlRawOpeningMessageV19,
    HistoryPoseidon2CompressBusV19, HistoryPoseidon2CompressMessageV19, LOGUP_ONLY_MODE_TAG_V19,
    MAX_RAW_MESSAGE_POINT_LEN_V19, NATIVE_WARP_HISTORY_PROTOCOL_V19,
};

const LIMB_BITS: usize = 16;
/// Domain separator used by the SDK's execution-wide Program fingerprint
/// challenge transcript.
pub const PROGRAM_FINGERPRINT_DOMAIN_TAG_V19: u64 = 0x4e57_5052_4644_0013;
/// Domain separator used by the canonical Program fingerprint digest.
pub const PROGRAM_FINGERPRINT_DIGEST_TAG_V19: u64 = 0x4e57_5052_4648_0013;
/// Canonical domain separator for the v19 VM-state digest.  SDK history
/// material must use this value rather than defining a second local tag.
pub const VM_STATE_HASH_TAG_V19: u32 = 0x19_56_4d;
const REQUIRED_VM_AIR_COUNT_V19: usize = 4;

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

macro_rules! define_lookup_bus {
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
                message: $Message<impl Into<AB::Expr> + Clone>,
                enabled: impl Into<AB::Expr>,
            ) {
                self.0.lookup_key(builder, message.to_vec(), enabled);
            }

            pub fn add_key_with_lookups<AB: InteractionBuilder>(
                &self,
                builder: &mut AB,
                message: $Message<impl Into<AB::Expr> + Clone>,
                count: impl Into<AB::Expr>,
            ) {
                self.0
                    .add_key_with_lookups(builder, message.to_vec(), count);
            }
        }
    };
}

/// Exact output expected from the mode-aware recursive GKR and interaction
/// endpoint evaluator.  `shard_count` binds omission at the boundary table.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct VerifiedLogUpArithmeticMessageV19<T> {
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
    pub shard_count: T,
}

define_permutation_bus!(
    VerifiedLogUpArithmeticBusV19,
    VerifiedLogUpArithmeticMessageV19
);

/// One canonical source-forest leaf/range.  The real forest verifier is the
/// only production sender.  Contiguous ranges plus canonical ordinals prevent
/// omission, overlap, and reordering.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct VerifiedSourceForestLeafMessageV19<T> {
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub shard_ordinal: T,
    pub air_id: T,
    pub relation_digest: [T; DIGEST_SIZE],
    pub log_height: T,
    pub cached_width: T,
    /// `log2` of this shard's padded systematic message.  This is part of
    /// the fixed source/relation descriptor and authenticates the ragged raw
    /// opening width used at the SWIRL boundary.
    pub log_message_len: T,
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub source_root: [T; DIGEST_SIZE],
    pub range_start: T,
    pub range_end: T,
}

define_permutation_bus!(
    VerifiedSourceForestLeafBusV19,
    VerifiedSourceForestLeafMessageV19
);

/// Verifier-authenticated direct-AIR public value.  These messages must be
/// emitted by the LogUp endpoint's real column/public-value evaluator, never
/// reconstructed from host segment metadata.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct VerifiedDirectAirPublicValueMessageV19<T> {
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub air_id: T,
    pub public_value_index: T,
    pub value: T,
}

define_permutation_bus!(
    VerifiedDirectAirPublicValueBusV19,
    VerifiedDirectAirPublicValueMessageV19
);

/// Execution-wide fingerprint challenges derived from the application VK and
/// homogeneous-shard registry.  The challenge bus is a lookup bus because the
/// same fixed challenge is consumed by every Program source in the chunk.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct ProgramFingerprintChallengeMessageV19<T> {
    pub protocol_version: T,
    pub app_vk_digest: [T; DIGEST_SIZE],
    pub registry_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub log_height: T,
    pub cached_width: T,
    pub row_point_len: T,
    pub row_point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub column_challenge: [T; D_EF],
}

define_lookup_bus!(
    ProgramFingerprintChallengeBusV19,
    ProgramFingerprintChallengeMessageV19
);

/// Result of evaluating the verifier-derived ordinary mapped functional and,
/// for the Program shard, the cached-only fingerprint functional at the
/// one-shot sumcheck point.  Its production provider is the structured mapped
/// functional evaluator AIR.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct VerifiedMappedFunctionalMessageV19<T> {
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub shard_ordinal: T,
    pub is_program: T,
    pub source_root: [T; DIGEST_SIZE],
    pub range_start: T,
    pub range_end: T,
    pub functional_digest: [T; DIGEST_SIZE],
    pub point_len: T,
    pub point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub ordinary_target: [T; D_EF],
    pub ordinary_weight_at_point: [T; D_EF],
    pub program_fingerprint: [T; D_EF],
    pub fingerprint_weight_at_point: [T; D_EF],
}

define_permutation_bus!(
    VerifiedMappedFunctionalBusV19,
    VerifiedMappedFunctionalMessageV19
);

/// One extension-field observation in the backend's canonical encoding of a
/// [`TerminalWeightSpec`](openvm_stark_backend::warp_pesat::TerminalWeightSpec)
/// and target.  The mapped-functional provider derives these rows from the
/// authenticated descriptor; the claim-transcript AIR consumes them in order.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct VerifiedOneShotClaimObservationMessageV19<T> {
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub shard_ordinal: T,
    pub observation_index: T,
    pub observation_count: T,
    pub value: [T; D_EF],
}

define_permutation_bus!(
    VerifiedOneShotClaimObservationBusV19,
    VerifiedOneShotClaimObservationMessageV19
);

/// Phase-cursor handoff from exact claim binding to reduction round zero.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct OneShotRoundStartMessageV19<T> {
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub shard_ordinal: T,
    pub round_start_tidx: T,
}

define_permutation_bus!(OneShotRoundStartBusV19, OneShotRoundStartMessageV19);

/// Ordered segment stream cursor. Claim derivation emits ordinal zero and
/// each completed reduction emits the next ordinal. This makes a per-shard
/// transcript fork impossible while the claim and sumcheck use separate AIRs.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct OneShotStreamCursorMessageV19<T> {
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub next_shard_ordinal: T,
    pub tidx: T,
}

define_permutation_bus!(OneShotStreamCursorBusV19, OneShotStreamCursorMessageV19);

/// Degree-two one-shot reduction output.  The mapped descriptor is absent:
/// it is consumed by this module and cannot enter History or terminal state.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct VerifiedOneShotRawOpeningMessageV19<T> {
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub shard_ordinal: T,
    pub is_program: T,
    pub source_root: [T; DIGEST_SIZE],
    pub range_start: T,
    pub range_end: T,
    pub point_len: T,
    pub point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub value: [T; D_EF],
    pub program_fingerprint: [T; D_EF],
}

define_permutation_bus!(
    VerifiedOneShotRawOpeningBusV19,
    VerifiedOneShotRawOpeningMessageV19
);

/// VM boundary data certified from direct-AIR public values plus the Program
/// cached-column fingerprint.  There is intentionally no cached PCS
/// commitment field here.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct CertifiedVmSegmentMetadataMessageV19<T> {
    pub protocol_version: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub initial_pc: T,
    pub final_pc: T,
    pub exit_code: T,
    pub is_terminate: T,
    pub initial_memory_root: [T; DIGEST_SIZE],
    pub final_memory_root: [T; DIGEST_SIZE],
    pub program_fingerprint: [T; D_EF],
    pub program_fingerprint_digest: [T; DIGEST_SIZE],
    pub from_vm_state: [T; DIGEST_SIZE],
    pub to_vm_state: [T; DIGEST_SIZE],
}

define_permutation_bus!(
    CertifiedVmSegmentMetadataBusV19,
    CertifiedVmSegmentMetadataMessageV19
);

/// Separate exact fingerprint statement for History/chunk-boundary wiring.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct CertifiedProgramFingerprintMessageV19<T> {
    pub protocol_version: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub app_vk_digest: [T; DIGEST_SIZE],
    pub registry_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub log_height: T,
    pub cached_width: T,
    pub value: [T; D_EF],
    pub digest: [T; DIGEST_SIZE],
}

define_permutation_bus!(
    CertifiedProgramFingerprintBusV19,
    CertifiedProgramFingerprintMessageV19
);

#[derive(Clone, Debug)]
pub struct ProgramFingerprintChallengeRecordV19 {
    pub proof_index: u32,
    pub start_tidx: u32,
    pub row_point: Vec<EF>,
    pub column_challenge: EF,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ProgramFingerprintChallengeColsV19<T> {
    pub active: T,
    pub proof_index: T,
    pub start_tidx: T,
    pub row_point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub column_challenge: [T; D_EF],
}

/// Fiat--Shamir derivation of the execution-wide cached Program fingerprint
/// point and column batching challenge.
#[derive(Clone, ColumnsAir)]
#[columns_via(ProgramFingerprintChallengeColsV19<u8>)]
pub struct ProgramFingerprintChallengeAirV19 {
    app_vk_digest: [F; DIGEST_SIZE],
    registry_digest: [F; DIGEST_SIZE],
    relation_digest: [F; DIGEST_SIZE],
    log_height: usize,
    cached_width: u32,
    /// Exact number of Program-source consumers in this bounded chunk.
    consumer_count: usize,
    transcript_bus: TranscriptBus,
    challenge_bus: ProgramFingerprintChallengeBusV19,
}

impl BaseAir<F> for ProgramFingerprintChallengeAirV19 {
    fn width(&self) -> usize {
        ProgramFingerprintChallengeColsV19::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for ProgramFingerprintChallengeAirV19 {}
impl PartitionedBaseAir<F> for ProgramFingerprintChallengeAirV19 {}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for ProgramFingerprintChallengeAirV19 {
    fn eval(&self, builder: &mut AB) {
        debug_assert!(self.log_height <= MAX_RAW_MESSAGE_POINT_LEN_V19);
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("program fingerprint challenge row");
        let local: &ProgramFingerprintChallengeColsV19<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        let next_row = main
            .row_slice(1)
            .expect("program fingerprint challenge next row");
        let next: &ProgramFingerprintChallengeColsV19<AB::Var> = (*next_row).borrow();
        builder.when_transition().assert_zero(next.active);

        let enabled = AB::Expr::from(local.active);
        let mut tidx = AB::Expr::from(local.start_tidx);
        self.transcript_bus.observe(
            builder,
            local.proof_index,
            tidx.clone(),
            AB::Expr::from_u64(PROGRAM_FINGERPRINT_DOMAIN_TAG_V19),
            enabled.clone(),
        );
        tidx += AB::Expr::ONE;
        self.transcript_bus.observe(
            builder,
            local.proof_index,
            tidx.clone(),
            AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            enabled.clone(),
        );
        tidx += AB::Expr::ONE;
        for value in self.app_vk_digest {
            self.transcript_bus.observe(
                builder,
                local.proof_index,
                tidx.clone(),
                value,
                enabled.clone(),
            );
            tidx += AB::Expr::ONE;
        }
        for value in self.registry_digest {
            self.transcript_bus.observe(
                builder,
                local.proof_index,
                tidx.clone(),
                value,
                enabled.clone(),
            );
            tidx += AB::Expr::ONE;
        }
        for value in self.relation_digest {
            self.transcript_bus.observe(
                builder,
                local.proof_index,
                tidx.clone(),
                value,
                enabled.clone(),
            );
            tidx += AB::Expr::ONE;
        }
        self.transcript_bus.observe(
            builder,
            local.proof_index,
            tidx.clone(),
            AB::Expr::from_usize(self.log_height),
            enabled.clone(),
        );
        tidx += AB::Expr::ONE;
        self.transcript_bus.observe(
            builder,
            local.proof_index,
            tidx.clone(),
            AB::Expr::from_u32(self.cached_width),
            enabled.clone(),
        );
        tidx += AB::Expr::ONE;
        for (index, point) in local.row_point.iter().enumerate() {
            let used = enabled.clone() * AB::Expr::from_bool(index < self.log_height);
            self.transcript_bus
                .sample_ext(builder, local.proof_index, tidx.clone(), *point, used);
            if index < self.log_height {
                tidx += AB::Expr::from_usize(D_EF);
            } else {
                for limb in point {
                    builder.when(enabled.clone()).assert_zero(*limb);
                }
            }
        }
        self.transcript_bus.sample_ext(
            builder,
            local.proof_index,
            tidx,
            local.column_challenge,
            enabled.clone(),
        );
        self.challenge_bus.add_key_with_lookups(
            builder,
            ProgramFingerprintChallengeMessageV19 {
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                app_vk_digest: self.app_vk_digest.map(Into::into),
                registry_digest: self.registry_digest.map(Into::into),
                relation_digest: self.relation_digest.map(Into::into),
                log_height: AB::Expr::from_usize(self.log_height),
                cached_width: AB::Expr::from_u32(self.cached_width),
                row_point_len: AB::Expr::from_usize(self.log_height),
                row_point: local.row_point.map(|point| point.map(Into::into)),
                column_challenge: local.column_challenge.map(Into::into),
            },
            enabled * AB::Expr::from_usize(self.consumer_count),
        );
    }
}

pub fn generate_program_fingerprint_challenge_trace_v19(
    air: &ProgramFingerprintChallengeAirV19,
    record: &ProgramFingerprintChallengeRecordV19,
) -> Result<RowMajorMatrix<F>, &'static str> {
    if record.row_point.len() != air.log_height || air.log_height > MAX_RAW_MESSAGE_POINT_LEN_V19 {
        return Err("program fingerprint challenge shape");
    }
    let width = ProgramFingerprintChallengeColsV19::<F>::width();
    let mut values = F::zero_vec(2 * width);
    let cols: &mut ProgramFingerprintChallengeColsV19<F> = values[..width].borrow_mut();
    cols.active = F::ONE;
    cols.proof_index = F::from_u32(record.proof_index);
    cols.start_tidx = F::from_u32(record.start_tidx);
    for (target, value) in cols.row_point.iter_mut().zip(&record.row_point) {
        target.copy_from_slice(value.as_basis_coefficients_slice());
    }
    cols.column_challenge
        .copy_from_slice(record.column_challenge.as_basis_coefficients_slice());
    Ok(RowMajorMatrix::new(values, width))
}

#[derive(Clone, Debug)]
pub struct OneShotReductionRecordV19 {
    pub proof_index: u32,
    pub segment_index: u32,
    pub shard_ordinal: u16,
    pub is_program: bool,
    pub source_root: [F; DIGEST_SIZE],
    pub range_start: u32,
    pub range_end: u32,
    pub functional_digest: [F; DIGEST_SIZE],
    pub ordinary_target: EF,
    pub ordinary_weight_at_point: EF,
    pub program_fingerprint: EF,
    pub fingerprint_weight_at_point: EF,
    pub mix_challenge: EF,
    pub start_tidx: u32,
    pub round_polynomials: Vec<[EF; 3]>,
    pub point: Vec<EF>,
    pub message_value: EF,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct OneShotReductionColsV19<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub shard_ordinal: T,
    pub is_program: T,
    pub source_root: [T; DIGEST_SIZE],
    pub range_start: T,
    pub range_end: T,
    pub functional_digest: [T; DIGEST_SIZE],
    pub round: T,
    pub round_tidx: T,
    pub ordinary_target: [T; D_EF],
    pub ordinary_weight_at_point: [T; D_EF],
    pub program_fingerprint: [T; D_EF],
    pub fingerprint_weight_at_point: [T; D_EF],
    pub mix_challenge: [T; D_EF],
    pub combined_target: [T; D_EF],
    pub combined_weight_at_point: [T; D_EF],
    pub pre_claim: [T; D_EF],
    pub coefficients: [[T; D_EF]; 3],
    pub challenge: [T; D_EF],
    pub post_claim: [T; D_EF],
    pub round_selector: [T; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub message_value: [T; D_EF],
}

/// Verifies the one-shot degree-two product sumcheck.  The ordinary and
/// fingerprint targets are bound before `mix_challenge` is sampled.
#[derive(Clone, ColumnsAir)]
#[columns_via(OneShotReductionColsV19<u8>)]
pub struct OneShotReductionAirV19 {
    log_message_len: usize,
    transcript_bus: TranscriptBus,
    round_start_bus: OneShotRoundStartBusV19,
    stream_cursor_bus: OneShotStreamCursorBusV19,
    mapped_functional_bus: VerifiedMappedFunctionalBusV19,
    output_bus: VerifiedOneShotRawOpeningBusV19,
    /// Setup-owned fanout counts. History-v2 C2 consumes a second copy of
    /// each terminal cursor and raw opening; ordinary v19 keeps both at one.
    stream_cursor_lookup_count: u32,
    output_lookup_count: u32,
}

impl OneShotReductionAirV19 {
    /// Production constructor. All authority buses are supplied by the
    /// enclosing protocol-v19 composite; no standalone/copy provider is
    /// allocated here.
    #[must_use]
    pub const fn new(
        log_message_len: usize,
        transcript_bus: TranscriptBus,
        round_start_bus: OneShotRoundStartBusV19,
        stream_cursor_bus: OneShotStreamCursorBusV19,
        mapped_functional_bus: VerifiedMappedFunctionalBusV19,
        output_bus: VerifiedOneShotRawOpeningBusV19,
    ) -> Self {
        Self {
            log_message_len,
            transcript_bus,
            round_start_bus,
            stream_cursor_bus,
            mapped_functional_bus,
            output_bus,
            stream_cursor_lookup_count: 1,
            output_lookup_count: 1,
        }
    }

    #[must_use]
    pub const fn log_message_len(&self) -> usize {
        self.log_message_len
    }

    /// Attach one additional verifier-owned consumer for the terminal cursor
    /// and raw opening. This is a key-generation operation, never witness
    /// data, and may be performed only once.
    pub fn attach_history_v2_active_count_projection(&mut self) -> Result<(), &'static str> {
        if self.stream_cursor_lookup_count != 1 || self.output_lookup_count != 1 {
            return Err("active-count projection already attached");
        }
        self.stream_cursor_lookup_count = 2;
        self.output_lookup_count = 2;
        Ok(())
    }
}

impl BaseAir<F> for OneShotReductionAirV19 {
    fn width(&self) -> usize {
        OneShotReductionColsV19::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for OneShotReductionAirV19 {}
impl PartitionedBaseAir<F> for OneShotReductionAirV19 {}

impl<AB> Air<AB> for OneShotReductionAirV19
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        debug_assert!(self.log_message_len <= MAX_RAW_MESSAGE_POINT_LEN_V19);
        let main = builder.main();
        let local_row = main.row_slice(0).expect("one-shot row");
        let next_row = main.row_slice(1).expect("one-shot next row");
        let local: &OneShotReductionColsV19<AB::Var> = (*local_row).borrow();
        let next: &OneShotReductionColsV19<AB::Var> = (*next_row).borrow();

        for flag in [
            local.active,
            local.is_first,
            local.is_last,
            local.is_program,
        ] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when_first_row().assert_zero(local.round);
        builder.when(local.active * local.is_last).assert_eq(
            local.round + AB::Expr::ONE,
            AB::Expr::from_usize(self.log_message_len),
        );
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .when(local.active - next.active)
            .assert_one(local.is_last);
        let continuing = next.active * (AB::Expr::ONE - AB::Expr::from(local.is_last));
        let starting = next.active * local.is_last;
        let mut transition_builder = builder.when_transition();
        let mut transition = transition_builder.when(continuing.clone());
        transition.assert_zero(next.is_first);
        transition.assert_eq(next.round, local.round + AB::Expr::ONE);
        transition.assert_eq(
            next.round_tidx,
            local.round_tidx + AB::Expr::from_usize(4 * D_EF),
        );
        transition.assert_eq(next.proof_index, local.proof_index);
        transition.assert_eq(next.segment_index_lo, local.segment_index_lo);
        transition.assert_eq(next.segment_index_hi, local.segment_index_hi);
        transition.assert_eq(next.shard_ordinal, local.shard_ordinal);
        transition.assert_eq(next.is_program, local.is_program);
        assert_array_eq(&mut transition, next.source_root, local.source_root);
        transition.assert_eq(next.range_start, local.range_start);
        transition.assert_eq(next.range_end, local.range_end);
        assert_array_eq(
            &mut transition,
            next.functional_digest,
            local.functional_digest,
        );
        assert_array_eq(&mut transition, next.ordinary_target, local.ordinary_target);
        assert_array_eq(
            &mut transition,
            next.ordinary_weight_at_point,
            local.ordinary_weight_at_point,
        );
        assert_array_eq(
            &mut transition,
            next.program_fingerprint,
            local.program_fingerprint,
        );
        assert_array_eq(
            &mut transition,
            next.fingerprint_weight_at_point,
            local.fingerprint_weight_at_point,
        );
        assert_array_eq(&mut transition, next.mix_challenge, local.mix_challenge);
        assert_array_eq(&mut transition, next.combined_target, local.combined_target);
        assert_array_eq(
            &mut transition,
            next.combined_weight_at_point,
            local.combined_weight_at_point,
        );
        assert_array_eq(&mut transition, next.pre_claim, local.post_claim);
        assert_array_eq(&mut transition, next.message_value, local.message_value);
        let mut start_builder = builder.when_transition();
        let mut start = start_builder.when(starting);
        start.assert_one(next.is_first);
        start.assert_zero(next.round);

        let mut selector_sum = AB::Expr::ZERO;
        let mut selected_round = AB::Expr::ZERO;
        for index in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
            builder.assert_bool(local.round_selector[index]);
            let used = AB::Expr::from_bool(index < self.log_message_len);
            builder.assert_zero(
                (AB::Expr::ONE - used.clone()) * AB::Expr::from(local.round_selector[index]),
            );
            selector_sum += AB::Expr::from(local.round_selector[index]);
            selected_round +=
                AB::Expr::from(local.round_selector[index]) * AB::Expr::from_usize(index);
            if index < self.log_message_len {
                for limb in 0..D_EF {
                    builder.assert_zero(
                        AB::Expr::from(local.round_selector[index])
                            * (AB::Expr::from(local.point[index][limb])
                                - AB::Expr::from(local.challenge[limb])),
                    );
                }
            } else {
                for limb in local.point[index] {
                    builder.when(local.active).assert_zero(limb);
                }
            }
            let mut point_transition = builder.when_transition();
            let mut when_continuing = point_transition.when(continuing.clone());
            assert_array_eq(
                &mut when_continuing,
                next.point[index],
                local.point[index].map(Into::into),
            );
        }
        builder.assert_eq(selector_sum, local.active);
        builder
            .when(local.active)
            .assert_eq(selected_round, local.round);

        let first = local.active * local.is_first;
        self.round_start_bus.receive(
            builder,
            OneShotRoundStartMessageV19 {
                proof_index: local.proof_index.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                shard_ordinal: local.shard_ordinal.into(),
                round_start_tidx: local.round_tidx.into(),
            },
            first.clone(),
        );
        for (index, coefficient) in local.coefficients.iter().enumerate() {
            self.transcript_bus.observe_ext(
                builder,
                local.proof_index,
                local.round_tidx + AB::Expr::from_usize(index * D_EF),
                *coefficient,
                local.active,
            );
        }
        self.transcript_bus.sample_ext(
            builder,
            local.proof_index,
            local.round_tidx + AB::Expr::from_usize(3 * D_EF),
            local.challenge,
            local.active,
        );

        let combined_target = ext_field_add_local::<AB::Expr>(
            local.ordinary_target,
            ext_field_multiply_local::<AB::Expr>(local.mix_challenge, local.program_fingerprint),
        );
        let combined_weight = ext_field_add_local::<AB::Expr>(
            local.ordinary_weight_at_point,
            ext_field_multiply_local::<AB::Expr>(
                local.mix_challenge,
                local.fingerprint_weight_at_point,
            ),
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.combined_target,
            combined_target,
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.combined_weight_at_point,
            combined_weight,
        );
        for limb in local.program_fingerprint {
            builder
                .when(
                    AB::Expr::from(local.active)
                        * (AB::Expr::ONE - AB::Expr::from(local.is_program)),
                )
                .assert_zero(limb);
        }
        for limb in local.fingerprint_weight_at_point {
            builder
                .when(
                    AB::Expr::from(local.active)
                        * (AB::Expr::ONE - AB::Expr::from(local.is_program)),
                )
                .assert_zero(limb);
        }
        for limb in local.mix_challenge {
            builder
                .when(
                    AB::Expr::from(local.active)
                        * (AB::Expr::ONE - AB::Expr::from(local.is_program)),
                )
                .assert_zero(limb);
        }
        assert_array_eq(
            &mut builder.when(first.clone()),
            local.pre_claim,
            local.combined_target,
        );
        let at_one = ext_field_add_local::<AB::Expr>(
            ext_field_add_local::<AB::Expr>(local.coefficients[0], local.coefficients[1]),
            local.coefficients[2],
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.pre_claim,
            ext_field_add_local::<AB::Expr>(local.coefficients[0], at_one),
        );
        let post_claim = ext_field_add_local::<AB::Expr>(
            local.coefficients[0],
            ext_field_multiply_local::<AB::Expr>(
                local.challenge,
                ext_field_add_local::<AB::Expr>(
                    local.coefficients[1],
                    ext_field_multiply_local::<AB::Expr>(local.challenge, local.coefficients[2]),
                ),
            ),
        );
        assert_array_eq(
            &mut builder.when(local.active),
            local.post_claim,
            post_claim,
        );

        let last = local.active * local.is_last;
        assert_array_eq(
            &mut builder.when(last.clone()),
            local.post_claim,
            ext_field_multiply_local::<AB::Expr>(
                local.message_value,
                local.combined_weight_at_point,
            ),
        );
        self.transcript_bus.observe_ext(
            builder,
            local.proof_index,
            local.round_tidx + AB::Expr::from_usize(4 * D_EF),
            local.message_value,
            last.clone(),
        );
        self.stream_cursor_bus.send(
            builder,
            OneShotStreamCursorMessageV19 {
                proof_index: local.proof_index.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                next_shard_ordinal: AB::Expr::from(local.shard_ordinal) + AB::Expr::ONE,
                tidx: AB::Expr::from(local.round_tidx) + AB::Expr::from_usize(5 * D_EF),
            },
            last.clone() * AB::Expr::from_u32(self.stream_cursor_lookup_count),
        );

        // The execution-wide fingerprint challenge is consumed by the mapped
        // functional evaluator, not by this reduction.  The one-shot's `rho`
        // above is deliberately a separate challenge.

        self.mapped_functional_bus.receive(
            builder,
            VerifiedMappedFunctionalMessageV19 {
                proof_index: local.proof_index.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                shard_ordinal: local.shard_ordinal.into(),
                is_program: local.is_program.into(),
                source_root: local.source_root.map(Into::into),
                range_start: local.range_start.into(),
                range_end: local.range_end.into(),
                functional_digest: local.functional_digest.map(Into::into),
                point_len: AB::Expr::from_usize(self.log_message_len),
                point: local.point.map(|point| point.map(Into::into)),
                ordinary_target: local.ordinary_target.map(Into::into),
                ordinary_weight_at_point: local.ordinary_weight_at_point.map(Into::into),
                program_fingerprint: local.program_fingerprint.map(Into::into),
                fingerprint_weight_at_point: local.fingerprint_weight_at_point.map(Into::into),
            },
            last.clone(),
        );
        self.output_bus.send(
            builder,
            VerifiedOneShotRawOpeningMessageV19 {
                proof_index: local.proof_index.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                shard_ordinal: local.shard_ordinal.into(),
                is_program: local.is_program.into(),
                source_root: local.source_root.map(Into::into),
                range_start: local.range_start.into(),
                range_end: local.range_end.into(),
                point_len: AB::Expr::from_usize(self.log_message_len),
                point: local.point.map(|point| point.map(Into::into)),
                value: local.message_value.map(Into::into),
                program_fingerprint: local.program_fingerprint.map(Into::into),
            },
            last * AB::Expr::from_u32(self.output_lookup_count),
        );
    }
}

pub fn generate_one_shot_reduction_trace_v19(
    air: &OneShotReductionAirV19,
    record: &OneShotReductionRecordV19,
) -> Result<RowMajorMatrix<F>, &'static str> {
    generate_grouped_one_shot_reduction_trace_v19(air, core::slice::from_ref(record))
}

/// Concatenate every reduction with the same numeric shape into one physical
/// AIR trace. Logical transcript order is still fixed by the proof-indexed
/// round-start/stream-cursor buses, so grouping does not create a transcript
/// fork or permit reordering.
pub fn generate_grouped_one_shot_reduction_trace_v19(
    air: &OneShotReductionAirV19,
    records: &[OneShotReductionRecordV19],
) -> Result<RowMajorMatrix<F>, &'static str> {
    if records.is_empty()
        || air.log_message_len == 0
        || air.log_message_len > MAX_RAW_MESSAGE_POINT_LEN_V19
        || records.iter().any(|record| {
            record.round_polynomials.len() != air.log_message_len
                || record.point.len() != air.log_message_len
                || record.range_start >= record.range_end
                || (!record.is_program
                    && (record.program_fingerprint != EF::ZERO
                        || record.fingerprint_weight_at_point != EF::ZERO
                        || record.mix_challenge != EF::ZERO))
                || (record.is_program && record.mix_challenge == EF::ZERO)
        })
    {
        return Err("one-shot reduction shape");
    }
    let record_rows = air.log_message_len;
    let rows = record_rows
        .checked_mul(records.len())
        .ok_or("one-shot reduction row count")?;
    let height = rows.next_power_of_two().max(2);
    let width = OneShotReductionColsV19::<F>::width();
    let mut values = F::zero_vec(height * width);
    for (record_index, record) in records.iter().enumerate() {
        let combined_target =
            record.ordinary_target + record.mix_challenge * record.program_fingerprint;
        let combined_weight = record.ordinary_weight_at_point
            + record.mix_challenge * record.fingerprint_weight_at_point;
        let mut running = combined_target;
        let mut round_tidx = record.start_tidx as usize;
        for round in 0..record_rows {
            let coefficients = record.round_polynomials[round];
            if coefficients[0] + coefficients.iter().copied().sum::<EF>() != running {
                return Err("one-shot reduction polynomial");
            }
            let challenge = record.point[round];
            let post =
                coefficients[0] + challenge * (coefficients[1] + challenge * coefficients[2]);
            let row = record_index * record_rows + round;
            let cols: &mut OneShotReductionColsV19<F> =
                values[row * width..(row + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.is_first = F::from_bool(round == 0);
            cols.is_last = F::from_bool(round + 1 == record_rows);
            cols.proof_index = F::from_u32(record.proof_index);
            cols.segment_index_lo = F::from_u32(record.segment_index & 0xffff);
            cols.segment_index_hi = F::from_u32(record.segment_index >> 16);
            cols.shard_ordinal = F::from_u16(record.shard_ordinal);
            cols.is_program = F::from_bool(record.is_program);
            cols.source_root = record.source_root;
            cols.range_start = F::from_u32(record.range_start);
            cols.range_end = F::from_u32(record.range_end);
            cols.functional_digest = record.functional_digest;
            cols.round = F::from_usize(round);
            cols.round_tidx = F::from_usize(round_tidx);
            copy_ext(&mut cols.ordinary_target, record.ordinary_target);
            copy_ext(
                &mut cols.ordinary_weight_at_point,
                record.ordinary_weight_at_point,
            );
            copy_ext(&mut cols.program_fingerprint, record.program_fingerprint);
            copy_ext(
                &mut cols.fingerprint_weight_at_point,
                record.fingerprint_weight_at_point,
            );
            copy_ext(&mut cols.mix_challenge, record.mix_challenge);
            copy_ext(&mut cols.combined_target, combined_target);
            copy_ext(&mut cols.combined_weight_at_point, combined_weight);
            copy_ext(&mut cols.pre_claim, running);
            for (target, value) in cols.coefficients.iter_mut().zip(coefficients) {
                copy_ext(target, value);
            }
            copy_ext(&mut cols.challenge, challenge);
            copy_ext(&mut cols.post_claim, post);
            cols.round_selector[round] = F::ONE;
            for (target, value) in cols.point.iter_mut().zip(&record.point) {
                copy_ext(target, *value);
            }
            copy_ext(&mut cols.message_value, record.message_value);
            running = post;
            round_tidx += 4 * D_EF;
        }
        if running != record.message_value * combined_weight {
            return Err("one-shot reduction endpoint");
        }
    }
    Ok(RowMajorMatrix::new(values, width))
}

#[derive(Clone, Debug)]
pub struct LogUpSwirlBoundaryShardRecordV19 {
    pub proof_index: u32,
    pub segment_index: u32,
    pub shard_ordinal: u16,
    pub shard_count: u16,
    pub air_id: u32,
    pub is_program: bool,
    pub relation_digest: [F; DIGEST_SIZE],
    pub log_height: u8,
    pub cached_width: u32,
    pub log_message_len: u8,
    pub app_vk_digest: [F; DIGEST_SIZE],
    pub registry_digest: [F; DIGEST_SIZE],
    pub source_forest_root: [F; DIGEST_SIZE],
    pub segment_openings_digest: [F; DIGEST_SIZE],
    pub source_root: [F; DIGEST_SIZE],
    pub range_start: u32,
    pub range_end: u32,
    pub opening_point: Vec<EF>,
    pub opening_value: EF,
    pub verifier_endpoint: EF,
    pub segment_sum_before: EF,
    pub segment_sum_after: EF,
    pub initial_pc: F,
    pub final_pc: F,
    pub exit_code: F,
    pub is_terminate: bool,
    pub initial_memory_root: [F; DIGEST_SIZE],
    pub final_memory_root: [F; DIGEST_SIZE],
    pub program_fingerprint: EF,
    pub program_fingerprint_digest: [F; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct LogUpSwirlBoundaryColsV19<T> {
    pub active: T,
    pub is_segment_first: T,
    pub is_segment_last: T,
    pub is_program: T,
    pub proof_index: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub segment_index_increment_carry: T,
    pub shard_ordinal: T,
    pub shard_count: T,
    pub air_id: T,
    pub relation_digest: [T; DIGEST_SIZE],
    pub log_height: T,
    pub cached_width: T,
    pub log_message_len: T,
    /// Exact equality indicators for Program, Connector, Boundary, and Merkle.
    pub required_air_flags: [T; REQUIRED_VM_AIR_COUNT_V19],
    pub required_air_diff_inverses: [T; REQUIRED_VM_AIR_COUNT_V19],
    pub required_air_counts: [T; REQUIRED_VM_AIR_COUNT_V19],
    pub app_vk_digest: [T; DIGEST_SIZE],
    pub registry_digest: [T; DIGEST_SIZE],
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub source_root: [T; DIGEST_SIZE],
    pub range_start: T,
    pub range_end: T,
    pub range_nonzero_inverse: T,
    pub point_len: T,
    /// One-hot authentication of `point_len` in `1..=MAX`.  Keeping this in
    /// the same segment-wide AIR permits heterogeneous shard message sizes
    /// without splitting required-VM cardinality across independent AIRs.
    pub point_len_selector: [T; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub opening_point: [[T; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
    pub opening_value: [T; D_EF],
    pub verifier_endpoint: [T; D_EF],
    pub segment_sum_before: [T; D_EF],
    pub segment_sum_after: [T; D_EF],
    pub initial_pc: T,
    pub final_pc: T,
    pub exit_code: T,
    pub is_terminate: T,
    pub initial_memory_root: [T; DIGEST_SIZE],
    pub final_memory_root: [T; DIGEST_SIZE],
    pub program_fingerprint: [T; D_EF],
    pub stable_program_fingerprint: [T; D_EF],
    pub stable_program_relation_digest: [T; DIGEST_SIZE],
    pub stable_program_log_height: T,
    pub stable_program_cached_width: T,
    pub program_fingerprint_digest: [T; DIGEST_SIZE],
    pub from_vm_inner_digest: [T; DIGEST_SIZE],
    pub to_vm_inner_digest: [T; DIGEST_SIZE],
    pub from_vm_state: [T; DIGEST_SIZE],
    pub to_vm_state: [T; DIGEST_SIZE],
    pub program_count: T,
}

/// Joins the cryptographic verifier outputs, enforces canonical segment/shard
/// structure and stable Program identity, and publishes the exact v19 buses.
#[derive(Clone, ColumnsAir)]
#[columns_via(LogUpSwirlBoundaryColsV19<u8>)]
pub struct LogUpSwirlBoundaryAirV19 {
    segment_start: u32,
    arithmetic_bus: VerifiedLogUpArithmeticBusV19,
    forest_bus: VerifiedSourceForestLeafBusV19,
    public_value_bus: VerifiedDirectAirPublicValueBusV19,
    one_shot_bus: VerifiedOneShotRawOpeningBusV19,
    certified_endpoint_bus: CertifiedLogUpOnlyEndpointBusV19,
    certified_opening_bus: CertifiedSwirlRawOpeningBusV19,
    certified_vm_bus: CertifiedVmSegmentMetadataBusV19,
    certified_program_bus: CertifiedProgramFingerprintBusV19,
    compress_bus: HistoryPoseidon2CompressBusV19,
}

impl LogUpSwirlBoundaryAirV19 {
    /// Production constructor joining only verifier-authenticated providers.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub const fn new(
        segment_start: u32,
        arithmetic_bus: VerifiedLogUpArithmeticBusV19,
        forest_bus: VerifiedSourceForestLeafBusV19,
        public_value_bus: VerifiedDirectAirPublicValueBusV19,
        one_shot_bus: VerifiedOneShotRawOpeningBusV19,
        certified_endpoint_bus: CertifiedLogUpOnlyEndpointBusV19,
        certified_opening_bus: CertifiedSwirlRawOpeningBusV19,
        certified_vm_bus: CertifiedVmSegmentMetadataBusV19,
        certified_program_bus: CertifiedProgramFingerprintBusV19,
        compress_bus: HistoryPoseidon2CompressBusV19,
    ) -> Self {
        Self {
            segment_start,
            arithmetic_bus,
            forest_bus,
            public_value_bus,
            one_shot_bus,
            certified_endpoint_bus,
            certified_opening_bus,
            certified_vm_bus,
            certified_program_bus,
            compress_bus,
        }
    }
}

impl BaseAir<F> for LogUpSwirlBoundaryAirV19 {
    fn width(&self) -> usize {
        LogUpSwirlBoundaryColsV19::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for LogUpSwirlBoundaryAirV19 {}
impl PartitionedBaseAir<F> for LogUpSwirlBoundaryAirV19 {}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for LogUpSwirlBoundaryAirV19 {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("LogUp/SWIRL boundary row");
        let next_row = main.row_slice(1).expect("LogUp/SWIRL boundary next row");
        let local: &LogUpSwirlBoundaryColsV19<AB::Var> = (*local_row).borrow();
        let next: &LogUpSwirlBoundaryColsV19<AB::Var> = (*next_row).borrow();
        for flag in [
            local.active,
            local.is_segment_first,
            local.is_segment_last,
            local.is_program,
            local.is_terminate,
            local.segment_index_increment_carry,
        ] {
            builder.assert_bool(flag);
        }
        let required_air_ids = [
            PROGRAM_AIR_ID,
            CONNECTOR_AIR_ID,
            BOUNDARY_AIR_ID,
            MERKLE_AIR_ID,
        ];
        for (index, required_air_id) in required_air_ids.into_iter().enumerate() {
            let flag = local.required_air_flags[index];
            builder.assert_bool(flag);
            let difference = AB::Expr::from(local.air_id) - AB::Expr::from_usize(required_air_id);
            builder
                .when(local.active)
                .assert_zero(difference.clone() * AB::Expr::from(flag));
            builder.when(local.active).assert_eq(
                difference * AB::Expr::from(local.required_air_diff_inverses[index]),
                AB::Expr::ONE - AB::Expr::from(flag),
            );
        }
        builder
            .when(local.active)
            .assert_eq(local.is_program, local.required_air_flags[0]);
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_segment_first);
        builder.when_first_row().assert_zero(local.shard_ordinal);
        builder.when_first_row().assert_zero(local.range_start);
        builder.when(local.active).assert_eq(
            AB::Expr::from(local.proof_index) + AB::Expr::from_u32(self.segment_start),
            AB::Expr::from(local.segment_index_lo)
                + AB::Expr::from_u32(1 << LIMB_BITS) * AB::Expr::from(local.segment_index_hi),
        );
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .when(local.active - next.active)
            .assert_one(local.is_segment_last);
        builder
            .when(local.active * local.is_segment_last)
            .assert_eq(
                AB::Expr::from(local.shard_ordinal) + AB::Expr::ONE,
                local.shard_count,
            );
        builder
            .when(local.active)
            .assert_one((local.range_end - local.range_start) * local.range_nonzero_inverse);

        // Authenticate a non-zero per-row opening dimension.  The selected
        // value is tied below to the source forest's fixed relation
        // descriptor (`log_message_len`), and every unused coordinate is
        // forced to canonical zero.  This keeps heterogeneous shards in one
        // segment-wide cardinality trace.
        let mut selector_sum = AB::Expr::ZERO;
        let mut selected_len = AB::Expr::ZERO;
        for index in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
            let selected = AB::Expr::from(local.point_len_selector[index]);
            builder.assert_bool(local.point_len_selector[index]);
            selector_sum += selected.clone();
            selected_len += selected.clone() * AB::Expr::from_usize(index + 1);
            let mut coordinate_is_used = AB::Expr::ZERO;
            for selected_len_minus_one in index..MAX_RAW_MESSAGE_POINT_LEN_V19 {
                coordinate_is_used +=
                    AB::Expr::from(local.point_len_selector[selected_len_minus_one]);
            }
            for limb in local.opening_point[index] {
                builder.when(local.active).assert_zero(
                    (AB::Expr::ONE - coordinate_is_used.clone()) * AB::Expr::from(limb),
                );
            }
        }
        builder.assert_eq(selector_sum, local.active);
        builder
            .when(local.active)
            .assert_eq(local.point_len, selected_len);
        builder
            .when(local.active)
            .assert_eq(local.point_len, local.log_message_len);

        let same_segment =
            AB::Expr::from(next.active) * (AB::Expr::ONE - AB::Expr::from(local.is_segment_last));
        let next_segment = AB::Expr::from(next.active) * AB::Expr::from(local.is_segment_last);
        let mut transition_builder = builder.when_transition();
        let mut same = transition_builder.when(same_segment.clone());
        same.assert_zero(next.is_segment_first);
        same.assert_eq(next.proof_index, local.proof_index);
        same.assert_eq(next.segment_index_lo, local.segment_index_lo);
        same.assert_eq(next.segment_index_hi, local.segment_index_hi);
        same.assert_eq(
            next.shard_ordinal,
            AB::Expr::from(local.shard_ordinal) + AB::Expr::ONE,
        );
        same.assert_eq(next.shard_count, local.shard_count);
        same.assert_eq(next.range_start, local.range_end);
        assert_array_eq(&mut same, next.app_vk_digest, local.app_vk_digest);
        assert_array_eq(&mut same, next.registry_digest, local.registry_digest);
        assert_array_eq(&mut same, next.source_forest_root, local.source_forest_root);
        assert_array_eq(
            &mut same,
            next.segment_openings_digest,
            local.segment_openings_digest,
        );
        assert_array_eq(&mut same, next.verifier_endpoint, local.verifier_endpoint);
        assert_array_eq(&mut same, next.segment_sum_before, local.segment_sum_before);
        assert_array_eq(&mut same, next.segment_sum_after, local.segment_sum_after);
        same.assert_eq(next.initial_pc, local.initial_pc);
        same.assert_eq(next.final_pc, local.final_pc);
        same.assert_eq(next.exit_code, local.exit_code);
        same.assert_eq(next.is_terminate, local.is_terminate);
        assert_array_eq(
            &mut same,
            next.initial_memory_root,
            local.initial_memory_root,
        );
        assert_array_eq(&mut same, next.final_memory_root, local.final_memory_root);
        assert_array_eq(
            &mut same,
            next.stable_program_fingerprint,
            local.stable_program_fingerprint,
        );
        assert_array_eq(
            &mut same,
            next.stable_program_relation_digest,
            local.stable_program_relation_digest,
        );
        same.assert_eq(
            next.stable_program_log_height,
            local.stable_program_log_height,
        );
        same.assert_eq(
            next.stable_program_cached_width,
            local.stable_program_cached_width,
        );
        assert_array_eq(
            &mut same,
            next.program_fingerprint_digest,
            local.program_fingerprint_digest,
        );
        same.assert_eq(next.program_count, local.program_count + next.is_program);
        for index in 0..REQUIRED_VM_AIR_COUNT_V19 {
            same.assert_eq(
                next.required_air_counts[index],
                AB::Expr::from(local.required_air_counts[index])
                    + AB::Expr::from(next.required_air_flags[index]),
            );
        }

        let mut transition_builder = builder.when_transition();
        let mut advance = transition_builder.when(next_segment);
        advance.assert_one(next.is_segment_first);
        advance.assert_zero(next.shard_ordinal);
        advance.assert_zero(next.range_start);
        advance.assert_eq(
            next.segment_index_lo,
            AB::Expr::from(local.segment_index_lo) + AB::Expr::ONE
                - AB::Expr::from(local.segment_index_increment_carry)
                    * AB::Expr::from_u32(1 << LIMB_BITS),
        );
        advance.assert_eq(
            next.segment_index_hi,
            AB::Expr::from(local.segment_index_hi)
                + AB::Expr::from(local.segment_index_increment_carry),
        );
        advance.assert_eq(local.program_count, AB::Expr::ONE);
        assert_array_eq(
            &mut advance,
            next.stable_program_fingerprint,
            local.stable_program_fingerprint,
        );
        assert_array_eq(
            &mut advance,
            next.stable_program_relation_digest,
            local.stable_program_relation_digest,
        );
        advance.assert_eq(
            next.stable_program_log_height,
            local.stable_program_log_height,
        );
        advance.assert_eq(
            next.stable_program_cached_width,
            local.stable_program_cached_width,
        );
        assert_array_eq(&mut advance, next.app_vk_digest, local.app_vk_digest);
        assert_array_eq(&mut advance, next.registry_digest, local.registry_digest);
        assert_array_eq(
            &mut advance,
            next.program_fingerprint_digest,
            local.program_fingerprint_digest,
        );
        advance.assert_eq(next.program_count, next.is_program);
        for index in 0..REQUIRED_VM_AIR_COUNT_V19 {
            advance.assert_eq(local.required_air_counts[index], AB::Expr::ONE);
            advance.assert_eq(
                next.required_air_counts[index],
                next.required_air_flags[index],
            );
        }

        builder
            .when(local.active * local.is_segment_first)
            .assert_eq(local.program_count, local.is_program);
        for index in 0..REQUIRED_VM_AIR_COUNT_V19 {
            builder
                .when(local.active * local.is_segment_first)
                .assert_eq(
                    local.required_air_counts[index],
                    local.required_air_flags[index],
                );
            builder
                .when(local.active * local.is_segment_last)
                .assert_eq(local.required_air_counts[index], AB::Expr::ONE);
        }
        builder
            .when(local.active * local.is_segment_last)
            .assert_eq(local.program_count, AB::Expr::ONE);
        assert_array_eq(
            &mut builder.when(local.active * local.is_program),
            local.program_fingerprint,
            local.stable_program_fingerprint,
        );
        assert_array_eq(
            &mut builder.when(local.active * local.is_program),
            local.relation_digest,
            local.stable_program_relation_digest,
        );
        builder
            .when(local.active * local.is_program)
            .assert_eq(local.log_height, local.stable_program_log_height);
        builder
            .when(local.active * local.is_program)
            .assert_eq(local.cached_width, local.stable_program_cached_width);
        for limb in local.program_fingerprint {
            builder
                .when(
                    AB::Expr::from(local.active)
                        * (AB::Expr::ONE - AB::Expr::from(local.is_program)),
                )
                .assert_zero(limb);
        }
        for limb in local.segment_sum_before {
            builder.when(local.active).assert_zero(limb);
        }
        for limb in local.segment_sum_after {
            builder.when(local.active).assert_zero(limb);
        }
        builder
            .when(local.active * local.is_terminate)
            .assert_zero(local.exit_code);
        builder
            .when(
                AB::Expr::from(local.active) * (AB::Expr::ONE - AB::Expr::from(local.is_terminate)),
            )
            .assert_eq(
                local.exit_code,
                AB::Expr::from_u32(DEFAULT_SUSPEND_EXIT_CODE),
            );

        self.forest_bus.receive(
            builder,
            VerifiedSourceForestLeafMessageV19 {
                proof_index: local.proof_index.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                shard_ordinal: local.shard_ordinal.into(),
                air_id: local.air_id.into(),
                relation_digest: local.relation_digest.map(Into::into),
                log_height: local.log_height.into(),
                cached_width: local.cached_width.into(),
                log_message_len: local.log_message_len.into(),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                source_root: local.source_root.map(Into::into),
                range_start: local.range_start.into(),
                range_end: local.range_end.into(),
            },
            local.active,
        );
        self.one_shot_bus.receive(
            builder,
            VerifiedOneShotRawOpeningMessageV19 {
                proof_index: local.proof_index.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                shard_ordinal: local.shard_ordinal.into(),
                is_program: local.is_program.into(),
                source_root: local.source_root.map(Into::into),
                range_start: local.range_start.into(),
                range_end: local.range_end.into(),
                point_len: local.point_len.into(),
                point: local.opening_point.map(|point| point.map(Into::into)),
                value: local.opening_value.map(Into::into),
                program_fingerprint: local.program_fingerprint.map(Into::into),
            },
            local.active,
        );

        let first = local.active * local.is_segment_first;
        self.arithmetic_bus.receive(
            builder,
            VerifiedLogUpArithmeticMessageV19 {
                proof_index: local.proof_index.into(),
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                mode_tag: AB::Expr::from_u32(LOGUP_ONLY_MODE_TAG_V19),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                app_vk_digest: local.app_vk_digest.map(Into::into),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                verifier_endpoint: local.verifier_endpoint.map(Into::into),
                segment_sum_before: local.segment_sum_before.map(Into::into),
                segment_sum_after: local.segment_sum_after.map(Into::into),
                shard_count: local.shard_count.into(),
            },
            first.clone(),
        );
        self.certified_endpoint_bus.add_key_with_lookups(
            builder,
            CertifiedLogUpOnlyEndpointMessageV19 {
                proof_index: local.proof_index.into(),
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                mode_tag: AB::Expr::from_u32(LOGUP_ONLY_MODE_TAG_V19),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                app_vk_digest: local.app_vk_digest.map(Into::into),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                verifier_endpoint: local.verifier_endpoint.map(Into::into),
                segment_sum_before: local.segment_sum_before.map(Into::into),
                segment_sum_after: local.segment_sum_after.map(Into::into),
            },
            first.clone(),
        );
        self.certified_opening_bus.add_key_with_lookups(
            builder,
            CertifiedSwirlRawOpeningMessageV19 {
                proof_index: local.proof_index.into(),
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                mode_tag: AB::Expr::from_u32(LOGUP_ONLY_MODE_TAG_V19),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                shard_ordinal: local.shard_ordinal.into(),
                source_forest_root: local.source_forest_root.map(Into::into),
                segment_openings_digest: local.segment_openings_digest.map(Into::into),
                root: local.source_root.map(Into::into),
                point_len: local.point_len.into(),
                point: local.opening_point.map(|point| point.map(Into::into)),
                value: local.opening_value.map(Into::into),
            },
            local.active,
        );

        for (index, value) in [
            local.initial_pc,
            local.final_pc,
            local.exit_code,
            local.is_terminate,
        ]
        .into_iter()
        .enumerate()
        {
            self.public_value_bus.receive(
                builder,
                VerifiedDirectAirPublicValueMessageV19 {
                    proof_index: local.proof_index.into(),
                    segment_index_lo: local.segment_index_lo.into(),
                    segment_index_hi: local.segment_index_hi.into(),
                    air_id: AB::Expr::from_usize(CONNECTOR_AIR_ID),
                    public_value_index: AB::Expr::from_usize(index),
                    value: value.into(),
                },
                first.clone(),
            );
        }
        for (index, value) in local
            .initial_memory_root
            .iter()
            .chain(local.final_memory_root.iter())
            .enumerate()
        {
            self.public_value_bus.receive(
                builder,
                VerifiedDirectAirPublicValueMessageV19 {
                    proof_index: local.proof_index.into(),
                    segment_index_lo: local.segment_index_lo.into(),
                    segment_index_hi: local.segment_index_hi.into(),
                    air_id: AB::Expr::from_usize(MERKLE_AIR_ID),
                    public_value_index: AB::Expr::from_usize(index),
                    value: (*value).into(),
                },
                first.clone(),
            );
        }

        let fingerprint_hash_left: [AB::Expr; DIGEST_SIZE] =
            core::array::from_fn(|index| match index {
                0 => AB::Expr::from_u64(PROGRAM_FINGERPRINT_DIGEST_TAG_V19),
                1 => AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                2 => local.stable_program_log_height.into(),
                3 => local.stable_program_cached_width.into(),
                _ => local.stable_program_fingerprint[index - 4].into(),
            });
        let fingerprint_hash_input: [AB::Expr; 2 * DIGEST_SIZE] = core::array::from_fn(|index| {
            if index < DIGEST_SIZE {
                fingerprint_hash_left[index].clone()
            } else {
                local.stable_program_relation_digest[index - DIGEST_SIZE].into()
            }
        });
        self.compress_bus.lookup_key(
            builder,
            HistoryPoseidon2CompressMessageV19 {
                input: fingerprint_hash_input,
                output: local.program_fingerprint_digest.map(Into::into),
            },
            first.clone(),
        );
        let from_meta: [AB::Expr; DIGEST_SIZE] = core::array::from_fn(|index| match index {
            0 => AB::Expr::from_u32(VM_STATE_HASH_TAG_V19),
            1 => AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            2 => local.initial_pc.into(),
            _ => AB::Expr::ZERO,
        });
        let to_meta: [AB::Expr; DIGEST_SIZE] = core::array::from_fn(|index| match index {
            0 => AB::Expr::from_u32(VM_STATE_HASH_TAG_V19),
            1 => AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            2 => local.final_pc.into(),
            3 => local.is_terminate.into(),
            _ => AB::Expr::ZERO,
        });
        self.compress_bus.lookup_key(
            builder,
            HistoryPoseidon2CompressMessageV19 {
                input: join_digest_expr(from_meta, local.initial_memory_root.map(Into::into)),
                output: local.from_vm_inner_digest.map(Into::into),
            },
            first.clone(),
        );
        self.compress_bus.lookup_key(
            builder,
            HistoryPoseidon2CompressMessageV19 {
                input: join_digest_expr(to_meta, local.final_memory_root.map(Into::into)),
                output: local.to_vm_inner_digest.map(Into::into),
            },
            first.clone(),
        );
        self.compress_bus.lookup_key(
            builder,
            HistoryPoseidon2CompressMessageV19 {
                input: join_digest_expr(
                    local.program_fingerprint_digest.map(Into::into),
                    local.from_vm_inner_digest.map(Into::into),
                ),
                output: local.from_vm_state.map(Into::into),
            },
            first.clone(),
        );
        self.compress_bus.lookup_key(
            builder,
            HistoryPoseidon2CompressMessageV19 {
                input: join_digest_expr(
                    local.program_fingerprint_digest.map(Into::into),
                    local.to_vm_inner_digest.map(Into::into),
                ),
                output: local.to_vm_state.map(Into::into),
            },
            first.clone(),
        );
        self.certified_vm_bus.send(
            builder,
            CertifiedVmSegmentMetadataMessageV19 {
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                initial_pc: local.initial_pc.into(),
                final_pc: local.final_pc.into(),
                exit_code: local.exit_code.into(),
                is_terminate: local.is_terminate.into(),
                initial_memory_root: local.initial_memory_root.map(Into::into),
                final_memory_root: local.final_memory_root.map(Into::into),
                program_fingerprint: local.stable_program_fingerprint.map(Into::into),
                program_fingerprint_digest: local.program_fingerprint_digest.map(Into::into),
                from_vm_state: local.from_vm_state.map(Into::into),
                to_vm_state: local.to_vm_state.map(Into::into),
            },
            first.clone(),
        );
        self.certified_program_bus.send(
            builder,
            CertifiedProgramFingerprintMessageV19 {
                protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                app_vk_digest: local.app_vk_digest.map(Into::into),
                registry_digest: local.registry_digest.map(Into::into),
                relation_digest: local.stable_program_relation_digest.map(Into::into),
                log_height: local.stable_program_log_height.into(),
                cached_width: local.stable_program_cached_width.into(),
                value: local.stable_program_fingerprint.map(Into::into),
                digest: local.program_fingerprint_digest.map(Into::into),
            },
            first,
        );

        builder
            .when(local.active * local.is_program)
            .assert_eq(local.air_id, AB::Expr::from_usize(PROGRAM_AIR_ID));
    }
}

pub fn generate_logup_swirl_boundary_trace_v19(
    _air: &LogUpSwirlBoundaryAirV19,
    records: &[LogUpSwirlBoundaryShardRecordV19],
) -> Result<RowMajorMatrix<F>, &'static str> {
    if records.is_empty() {
        return Err("LogUp/SWIRL boundary shape");
    }
    let height = records.len().next_power_of_two().max(2);
    let width = LogUpSwirlBoundaryColsV19::<F>::width();
    let mut values = F::zero_vec(height * width);
    let stable = records
        .iter()
        .find(|record| record.is_program)
        .ok_or("missing Program shard")?
        .program_fingerprint;
    let stable_digest = records
        .iter()
        .find(|record| record.is_program)
        .ok_or("missing Program shard")?
        .program_fingerprint_digest;
    let stable_relation_digest = records
        .iter()
        .find(|record| record.is_program)
        .ok_or("missing Program shard")?
        .relation_digest;
    let stable_log_height = records
        .iter()
        .find(|record| record.is_program)
        .ok_or("missing Program shard")?
        .log_height;
    let stable_cached_width = records
        .iter()
        .find(|record| record.is_program)
        .ok_or("missing Program shard")?
        .cached_width;
    if stable_digest
        != compute_program_fingerprint_digest_v19(
            stable_relation_digest,
            stable_log_height,
            stable_cached_width,
            stable,
        )
    {
        return Err("non-canonical Program fingerprint digest");
    }
    let mut program_count = 0usize;
    let mut required_air_counts = [0usize; REQUIRED_VM_AIR_COUNT_V19];
    for (index, record) in records.iter().enumerate() {
        if record.opening_point.is_empty()
            || record.opening_point.len() > MAX_RAW_MESSAGE_POINT_LEN_V19
            || record.opening_point.len() != usize::from(record.log_message_len)
            || record.range_start >= record.range_end
            || usize::from(record.shard_ordinal) + 1 > usize::from(record.shard_count)
            || (record.is_program
                && (record.relation_digest != stable_relation_digest
                    || record.log_height != stable_log_height
                    || record.cached_width != stable_cached_width
                    || record.program_fingerprint != stable
                    || record.program_fingerprint_digest != stable_digest))
        {
            return Err("LogUp/SWIRL boundary record");
        }
        let is_first = record.shard_ordinal == 0;
        let is_last = record.shard_ordinal + 1 == record.shard_count;
        if is_first {
            program_count = 0;
            required_air_counts = [0; REQUIRED_VM_AIR_COUNT_V19];
        }
        program_count += usize::from(record.is_program);
        let required_air_ids = [
            PROGRAM_AIR_ID,
            CONNECTOR_AIR_ID,
            BOUNDARY_AIR_ID,
            MERKLE_AIR_ID,
        ];
        for (required, &air_id) in required_air_ids.iter().enumerate() {
            required_air_counts[required] += usize::from(record.air_id as usize == air_id);
        }
        if is_last
            && (program_count != 1
                || required_air_counts
                    .iter()
                    .any(|&required_count| required_count != 1))
        {
            return Err("required VM shard cardinality");
        }
        let cols: &mut LogUpSwirlBoundaryColsV19<F> =
            values[index * width..(index + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.is_segment_first = F::from_bool(is_first);
        cols.is_segment_last = F::from_bool(is_last);
        cols.is_program = F::from_bool(record.is_program);
        cols.proof_index = F::from_u32(record.proof_index);
        cols.segment_index_lo = F::from_u32(record.segment_index & 0xffff);
        cols.segment_index_hi = F::from_u32(record.segment_index >> 16);
        cols.segment_index_increment_carry = F::from_bool(
            record.shard_ordinal + 1 == record.shard_count
                && (record.segment_index & 0xffff) == 0xffff,
        );
        cols.shard_ordinal = F::from_u16(record.shard_ordinal);
        cols.shard_count = F::from_u16(record.shard_count);
        cols.air_id = F::from_u32(record.air_id);
        cols.relation_digest = record.relation_digest;
        cols.log_height = F::from_u8(record.log_height);
        cols.cached_width = F::from_u32(record.cached_width);
        cols.log_message_len = F::from_u8(record.log_message_len);
        for (required, &air_id) in required_air_ids.iter().enumerate() {
            let difference = F::from_u32(record.air_id) - F::from_usize(air_id);
            let flag = difference == F::ZERO;
            cols.required_air_flags[required] = F::from_bool(flag);
            cols.required_air_diff_inverses[required] =
                if flag { F::ZERO } else { difference.inverse() };
            cols.required_air_counts[required] = F::from_usize(required_air_counts[required]);
        }
        cols.app_vk_digest = record.app_vk_digest;
        cols.registry_digest = record.registry_digest;
        cols.source_forest_root = record.source_forest_root;
        cols.segment_openings_digest = record.segment_openings_digest;
        cols.source_root = record.source_root;
        cols.range_start = F::from_u32(record.range_start);
        cols.range_end = F::from_u32(record.range_end);
        cols.range_nonzero_inverse =
            (F::from_u32(record.range_end) - F::from_u32(record.range_start)).inverse();
        cols.point_len = F::from_usize(record.opening_point.len());
        cols.point_len_selector[record.opening_point.len() - 1] = F::ONE;
        for (target, value) in cols.opening_point.iter_mut().zip(&record.opening_point) {
            copy_ext(target, *value);
        }
        copy_ext(&mut cols.opening_value, record.opening_value);
        copy_ext(&mut cols.verifier_endpoint, record.verifier_endpoint);
        copy_ext(&mut cols.segment_sum_before, record.segment_sum_before);
        copy_ext(&mut cols.segment_sum_after, record.segment_sum_after);
        cols.initial_pc = record.initial_pc;
        cols.final_pc = record.final_pc;
        cols.exit_code = record.exit_code;
        cols.is_terminate = F::from_bool(record.is_terminate);
        cols.initial_memory_root = record.initial_memory_root;
        cols.final_memory_root = record.final_memory_root;
        copy_ext(&mut cols.program_fingerprint, record.program_fingerprint);
        copy_ext(&mut cols.stable_program_fingerprint, stable);
        cols.stable_program_relation_digest = stable_relation_digest;
        cols.stable_program_log_height = F::from_u8(stable_log_height);
        cols.stable_program_cached_width = F::from_u32(stable_cached_width);
        cols.program_fingerprint_digest = stable_digest;
        let from_state = compute_vm_state_digest_v19(
            stable_digest,
            record.initial_pc,
            false,
            record.initial_memory_root,
        );
        let to_state = compute_vm_state_digest_v19(
            stable_digest,
            record.final_pc,
            record.is_terminate,
            record.final_memory_root,
        );
        cols.from_vm_inner_digest = from_state.inner_digest;
        cols.to_vm_inner_digest = to_state.inner_digest;
        cols.from_vm_state = from_state.state_digest;
        cols.to_vm_state = to_state.state_digest;
        cols.program_count = F::from_usize(program_count);
    }
    Ok(RowMajorMatrix::new(values, width))
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

/// Canonical host/circuit-identical digest of the stable cached Program
/// fingerprint.
///
/// This is exactly
/// `H([tag, v19, log_height, cached_width, target EF4], relation_digest)`.
/// Both SDK claim construction and the boundary AIR use this helper/framing;
/// changing it is a protocol change.
#[must_use]
pub fn compute_program_fingerprint_digest_v19(
    relation_digest: [F; DIGEST_SIZE],
    log_height: u8,
    cached_width: u32,
    target: EF,
) -> [F; DIGEST_SIZE] {
    let mut left = [F::ZERO; DIGEST_SIZE];
    left[0] = F::from_u64(PROGRAM_FINGERPRINT_DIGEST_TAG_V19);
    left[1] = F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19);
    left[2] = F::from_u8(log_height);
    left[3] = F::from_u32(cached_width);
    left[4..].copy_from_slice(target.as_basis_coefficients_slice());
    poseidon2_compress_with_capacity(left, relation_digest).0
}

/// Two compression outputs used by the v19 VM-state definition.  Keeping the
/// inner digest explicit lets SDK trace construction populate the exact
/// witness columns without reimplementing the framing convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VmStateDigestRecordV19 {
    pub inner_digest: [F; DIGEST_SIZE],
    pub state_digest: [F; DIGEST_SIZE],
}

/// Compute
/// `H(program_fingerprint_digest, H(tag/version/pc/terminated, memory_root))`.
/// Initial segment states pass `terminated = false`; final states use the
/// certified Connector termination bit.
#[must_use]
pub fn compute_vm_state_digest_v19(
    program_fingerprint_digest: [F; DIGEST_SIZE],
    pc: F,
    terminated: bool,
    memory_root: [F; DIGEST_SIZE],
) -> VmStateDigestRecordV19 {
    let mut metadata = [F::ZERO; DIGEST_SIZE];
    metadata[0] = F::from_u32(VM_STATE_HASH_TAG_V19);
    metadata[1] = F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19);
    metadata[2] = pc;
    metadata[3] = F::from_bool(terminated);
    let inner_digest = poseidon2_compress_with_capacity(metadata, memory_root).0;
    let state_digest = poseidon2_compress_with_capacity(program_fingerprint_digest, inner_digest).0;
    VmStateDigestRecordV19 {
        inner_digest,
        state_digest,
    }
}

fn join_digest_expr<E: Clone>(
    left: [E; DIGEST_SIZE],
    right: [E; DIGEST_SIZE],
) -> [E; 2 * DIGEST_SIZE] {
    core::array::from_fn(|index| {
        if index < DIGEST_SIZE {
            left[index].clone()
        } else {
            right[index - DIGEST_SIZE].clone()
        }
    })
}

fn ext_field_add_local<FA>(left: [impl Into<FA>; D_EF], right: [impl Into<FA>; D_EF]) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
{
    let left = left.map(Into::into);
    let right = right.map(Into::into);
    core::array::from_fn(|index| left[index].clone() + right[index].clone())
}

fn ext_field_multiply_local<FA>(
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
            let coordinate = degree % D_EF;
            output[coordinate] = output[coordinate].clone() + term;
        }
    }
    output
}

const _: () = assert!(D_EF == 4);
const _: () = assert!(2 * DIGEST_SIZE == 16);

#[cfg(test)]
mod tests {
    use openvm_recursion_circuit::bus::{TranscriptBus, TranscriptBusMessage};
    use openvm_stark_backend::{
        air_builders::{
            debug::{check_constraints, check_logup},
            symbolic::{get_symbolic_builder, SymbolicRapBuilder},
        },
        interaction::SymbolicInteraction,
        keygen::types::TraceWidth,
        p3_matrix::Matrix,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::{
        poseidon2_compress_with_capacity, BabyBearPoseidon2Config,
    };

    use super::*;

    const TRANSCRIPT_BUS: u16 = 300;
    const MAPPED_BUS: u16 = 301;
    const ONE_SHOT_BUS: u16 = 302;
    const ARITHMETIC_BUS: u16 = 303;
    const FOREST_BUS: u16 = 304;
    const PV_BUS: u16 = 305;
    const ENDPOINT_BUS: u16 = 306;
    const OPENING_BUS: u16 = 307;
    const VM_BUS: u16 = 308;
    const PROGRAM_BUS: u16 = 309;
    const COMPRESS_BUS: u16 = 310;

    fn digest(seed: u32) -> [F; DIGEST_SIZE] {
        core::array::from_fn(|index| F::from_u32(seed + index as u32))
    }

    fn ef(seed: u32) -> EF {
        let coefficients: [F; D_EF] =
            core::array::from_fn(|index| F::from_u32(seed + index as u32));
        EF::from_basis_coefficients_slice(&coefficients).unwrap()
    }

    #[derive(Clone, ColumnsAir)]
    #[columns_via(ProgramFingerprintChallengeColsV19<u8>)]
    struct TestOnlyProgramFingerprintChallengeConsumerAir {
        app_vk_digest: [F; DIGEST_SIZE],
        registry_digest: [F; DIGEST_SIZE],
        relation_digest: [F; DIGEST_SIZE],
        log_height: usize,
        cached_width: u32,
        challenge_bus: ProgramFingerprintChallengeBusV19,
    }

    impl BaseAir<F> for TestOnlyProgramFingerprintChallengeConsumerAir {
        fn width(&self) -> usize {
            ProgramFingerprintChallengeColsV19::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for TestOnlyProgramFingerprintChallengeConsumerAir {}
    impl PartitionedBaseAir<F> for TestOnlyProgramFingerprintChallengeConsumerAir {}

    impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB>
        for TestOnlyProgramFingerprintChallengeConsumerAir
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).unwrap();
            let local: &ProgramFingerprintChallengeColsV19<AB::Var> = (*row).borrow();
            self.challenge_bus.lookup_key(
                builder,
                ProgramFingerprintChallengeMessageV19 {
                    protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                    app_vk_digest: self.app_vk_digest.map(Into::into),
                    registry_digest: self.registry_digest.map(Into::into),
                    relation_digest: self.relation_digest.map(Into::into),
                    log_height: AB::Expr::from_usize(self.log_height),
                    cached_width: AB::Expr::from_u32(self.cached_width),
                    row_point_len: AB::Expr::from_usize(self.log_height),
                    row_point: local.row_point.map(|point| point.map(Into::into)),
                    column_challenge: local.column_challenge.map(Into::into),
                },
                local.active,
            );
        }
    }

    fn fingerprint_challenge_air() -> ProgramFingerprintChallengeAirV19 {
        ProgramFingerprintChallengeAirV19 {
            app_vk_digest: digest(1),
            registry_digest: digest(20),
            relation_digest: digest(40),
            log_height: 2,
            cached_width: 6,
            consumer_count: 1,
            transcript_bus: TranscriptBus::new(TRANSCRIPT_BUS),
            challenge_bus: ProgramFingerprintChallengeBusV19::new(PROGRAM_BUS),
        }
    }

    fn fingerprint_challenge_record() -> ProgramFingerprintChallengeRecordV19 {
        ProgramFingerprintChallengeRecordV19 {
            proof_index: 3,
            start_tidx: 5,
            row_point: vec![ef(60), ef(80)],
            column_challenge: ef(100),
        }
    }

    fn fingerprint_challenge_transcript_trace(
        air: &ProgramFingerprintChallengeAirV19,
        record: &ProgramFingerprintChallengeRecordV19,
    ) -> RowMajorMatrix<F> {
        let mut operations = vec![
            (F::from_u64(PROGRAM_FINGERPRINT_DOMAIN_TAG_V19), false),
            (F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19), false),
        ];
        operations.extend(air.app_vk_digest.map(|value| (value, false)));
        operations.extend(air.registry_digest.map(|value| (value, false)));
        operations.extend(air.relation_digest.map(|value| (value, false)));
        operations.push((F::from_usize(air.log_height), false));
        operations.push((F::from_u32(air.cached_width), false));
        for challenge in record
            .row_point
            .iter()
            .chain(core::iter::once(&record.column_challenge))
        {
            operations.extend(
                challenge
                    .as_basis_coefficients_slice()
                    .iter()
                    .copied()
                    .map(|value| (value, true)),
            );
        }
        let width = TranscriptSourceCols::<F>::width();
        let height = operations.len().next_power_of_two();
        let mut values = F::zero_vec(width * height);
        for (offset, (value, is_sample)) in operations.into_iter().enumerate() {
            let row: &mut TranscriptSourceCols<F> =
                values[offset * width..(offset + 1) * width].borrow_mut();
            row.active = F::ONE;
            row.proof_index = F::from_u32(record.proof_index);
            row.tidx = F::from_usize(record.start_tidx as usize + offset);
            row.value = value;
            row.is_sample = F::from_bool(is_sample);
        }
        RowMajorMatrix::new(values, width)
    }

    fn one_shot_air() -> OneShotReductionAirV19 {
        OneShotReductionAirV19 {
            log_message_len: 2,
            transcript_bus: TranscriptBus::new(TRANSCRIPT_BUS),
            round_start_bus: OneShotRoundStartBusV19::new(ARITHMETIC_BUS),
            stream_cursor_bus: OneShotStreamCursorBusV19::new(FOREST_BUS),
            mapped_functional_bus: VerifiedMappedFunctionalBusV19::new(MAPPED_BUS),
            output_bus: VerifiedOneShotRawOpeningBusV19::new(ONE_SHOT_BUS),
            stream_cursor_lookup_count: 1,
            output_lookup_count: 1,
        }
    }

    fn one_shot_record() -> OneShotReductionRecordV19 {
        let ordinary_target = EF::from_u32(10);
        let fingerprint = EF::from_u32(2);
        let rho = EF::from_u32(3);
        let combined = ordinary_target + rho * fingerprint;
        let first = [combined * F::TWO.inverse(), EF::ZERO, EF::ZERO];
        let second_running = first[0];
        let second = [second_running * F::TWO.inverse(), EF::ZERO, EF::ZERO];
        OneShotReductionRecordV19 {
            proof_index: 7,
            segment_index: 7,
            shard_ordinal: 0,
            is_program: true,
            source_root: digest(50),
            range_start: 0,
            range_end: 64,
            functional_digest: digest(70),
            ordinary_target,
            ordinary_weight_at_point: EF::ONE,
            program_fingerprint: fingerprint,
            fingerprint_weight_at_point: EF::ZERO,
            mix_challenge: rho,
            start_tidx: 9,
            round_polynomials: vec![first, second],
            point: vec![ef(100), ef(120)],
            message_value: second[0],
        }
    }

    fn check_air<A>(air: &A, name: &str, matrix: &RowMajorMatrix<F>)
    where
        A: for<'a> Air<
                openvm_stark_backend::air_builders::debug::DebugConstraintBuilder<
                    'a,
                    BabyBearPoseidon2Config,
                >,
            > + BaseAir<F>
            + PartitionedBaseAir<F>,
    {
        check_constraints::<_, BabyBearPoseidon2Config>(air, name, &None, &[matrix.as_view()], &[]);
    }

    fn symbolic_interactions<A>(air: &A) -> Vec<SymbolicInteraction<F>>
    where
        A: Air<SymbolicRapBuilder<F>>
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

    #[derive(Clone, ColumnsAir)]
    #[columns_via(OneShotReductionColsV19<u8>)]
    struct TestOnlyOneShotEchoAuthorityAir {
        round_start_bus: OneShotRoundStartBusV19,
        stream_cursor_bus: OneShotStreamCursorBusV19,
        mapped_bus: VerifiedMappedFunctionalBusV19,
        output_bus: VerifiedOneShotRawOpeningBusV19,
        log_message_len: usize,
    }

    impl BaseAir<F> for TestOnlyOneShotEchoAuthorityAir {
        fn width(&self) -> usize {
            OneShotReductionColsV19::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for TestOnlyOneShotEchoAuthorityAir {}
    impl PartitionedBaseAir<F> for TestOnlyOneShotEchoAuthorityAir {}

    impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for TestOnlyOneShotEchoAuthorityAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).unwrap();
            let local: &OneShotReductionColsV19<AB::Var> = (*row).borrow();
            let first = local.active * local.is_first;
            let last = local.active * local.is_last;
            self.round_start_bus.send(
                builder,
                OneShotRoundStartMessageV19 {
                    proof_index: local.proof_index.into(),
                    segment_index_lo: local.segment_index_lo.into(),
                    segment_index_hi: local.segment_index_hi.into(),
                    shard_ordinal: local.shard_ordinal.into(),
                    round_start_tidx: local.round_tidx.into(),
                },
                first,
            );
            self.stream_cursor_bus.receive(
                builder,
                OneShotStreamCursorMessageV19 {
                    proof_index: local.proof_index.into(),
                    segment_index_lo: local.segment_index_lo.into(),
                    segment_index_hi: local.segment_index_hi.into(),
                    next_shard_ordinal: AB::Expr::from(local.shard_ordinal) + AB::Expr::ONE,
                    tidx: AB::Expr::from(local.round_tidx) + AB::Expr::from_usize(5 * D_EF),
                },
                last.clone(),
            );
            self.mapped_bus.send(
                builder,
                VerifiedMappedFunctionalMessageV19 {
                    proof_index: local.proof_index.into(),
                    segment_index_lo: local.segment_index_lo.into(),
                    segment_index_hi: local.segment_index_hi.into(),
                    shard_ordinal: local.shard_ordinal.into(),
                    is_program: local.is_program.into(),
                    source_root: local.source_root.map(Into::into),
                    range_start: local.range_start.into(),
                    range_end: local.range_end.into(),
                    functional_digest: local.functional_digest.map(Into::into),
                    point_len: AB::Expr::from_usize(self.log_message_len),
                    point: local.point.map(|point| point.map(Into::into)),
                    ordinary_target: local.ordinary_target.map(Into::into),
                    ordinary_weight_at_point: local.ordinary_weight_at_point.map(Into::into),
                    program_fingerprint: local.program_fingerprint.map(Into::into),
                    fingerprint_weight_at_point: local.fingerprint_weight_at_point.map(Into::into),
                },
                last.clone(),
            );
            self.output_bus.receive(
                builder,
                VerifiedOneShotRawOpeningMessageV19 {
                    proof_index: local.proof_index.into(),
                    segment_index_lo: local.segment_index_lo.into(),
                    segment_index_hi: local.segment_index_hi.into(),
                    shard_ordinal: local.shard_ordinal.into(),
                    is_program: local.is_program.into(),
                    source_root: local.source_root.map(Into::into),
                    range_start: local.range_start.into(),
                    range_end: local.range_end.into(),
                    point_len: AB::Expr::from_usize(self.log_message_len),
                    point: local.point.map(|point| point.map(Into::into)),
                    value: local.message_value.map(Into::into),
                    program_fingerprint: local.program_fingerprint.map(Into::into),
                },
                last,
            );
        }
    }

    #[derive(Clone, ColumnsAir)]
    #[columns_via(TranscriptSourceCols<u8>)]
    struct TranscriptSourceAir(TranscriptBus);

    #[repr(C)]
    #[derive(AlignedBorrow, StructReflection)]
    struct TranscriptSourceCols<T> {
        active: T,
        proof_index: T,
        tidx: T,
        value: T,
        is_sample: T,
    }

    impl BaseAir<F> for TranscriptSourceAir {
        fn width(&self) -> usize {
            TranscriptSourceCols::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for TranscriptSourceAir {}
    impl PartitionedBaseAir<F> for TranscriptSourceAir {}

    impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for TranscriptSourceAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).unwrap();
            let local: &TranscriptSourceCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.active);
            builder.assert_bool(local.is_sample);
            self.0.send(
                builder,
                local.proof_index,
                TranscriptBusMessage {
                    tidx: local.tidx.into(),
                    value: local.value.into(),
                    is_sample: local.is_sample.into(),
                },
                local.active,
            );
        }
    }

    fn one_shot_transcript_trace(record: &OneShotReductionRecordV19) -> RowMajorMatrix<F> {
        grouped_one_shot_transcript_trace(core::slice::from_ref(record))
    }

    fn grouped_one_shot_transcript_trace(
        records: &[OneShotReductionRecordV19],
    ) -> RowMajorMatrix<F> {
        let mut operations = Vec::<(F, bool)>::new();
        let start_tidx = records[0].start_tidx;
        for record in records {
            assert_eq!(
                record.start_tidx as usize,
                start_tidx as usize + operations.len()
            );
            for (polynomial, challenge) in record.round_polynomials.iter().zip(&record.point) {
                for value in polynomial {
                    operations.extend(
                        value
                            .as_basis_coefficients_slice()
                            .iter()
                            .copied()
                            .map(|value| (value, false)),
                    );
                }
                operations.extend(
                    challenge
                        .as_basis_coefficients_slice()
                        .iter()
                        .copied()
                        .map(|value| (value, true)),
                );
            }
            operations.extend(
                record
                    .message_value
                    .as_basis_coefficients_slice()
                    .iter()
                    .copied()
                    .map(|value| (value, false)),
            );
        }
        let width = TranscriptSourceCols::<F>::width();
        let height = operations.len().next_power_of_two();
        let mut values = F::zero_vec(width * height);
        for (offset, (value, sample)) in operations.into_iter().enumerate() {
            let row: &mut TranscriptSourceCols<F> =
                values[offset * width..(offset + 1) * width].borrow_mut();
            row.active = F::ONE;
            row.proof_index = F::from_u32(records[0].proof_index);
            row.tidx = F::from_usize(start_tidx as usize + offset);
            row.value = value;
            row.is_sample = F::from_bool(sample);
        }
        RowMajorMatrix::new(values, width)
    }

    fn check_one_shot_composed(
        verifier_trace: &RowMajorMatrix<F>,
        authority_trace: &RowMajorMatrix<F>,
        transcript_trace: &RowMajorMatrix<F>,
    ) {
        let verifier = one_shot_air();
        let authority = TestOnlyOneShotEchoAuthorityAir {
            round_start_bus: OneShotRoundStartBusV19::new(ARITHMETIC_BUS),
            stream_cursor_bus: OneShotStreamCursorBusV19::new(FOREST_BUS),
            mapped_bus: VerifiedMappedFunctionalBusV19::new(MAPPED_BUS),
            output_bus: VerifiedOneShotRawOpeningBusV19::new(ONE_SHOT_BUS),
            log_message_len: 2,
        };
        let transcript = TranscriptSourceAir(TranscriptBus::new(TRANSCRIPT_BUS));
        check_logup(
            &[
                "one-shot".into(),
                "mapped authority".into(),
                "transcript".into(),
            ],
            &[
                symbolic_interactions(&verifier),
                symbolic_interactions(&authority),
                symbolic_interactions(&transcript),
            ],
            &[None, None, None],
            &[
                vec![verifier_trace.as_view()],
                vec![authority_trace.as_view()],
                vec![transcript_trace.as_view()],
            ],
            &[vec![], vec![], vec![]],
        );
    }

    #[test]
    fn program_fingerprint_schedule_and_digest_match_the_sdk_protocol() {
        let air = fingerprint_challenge_air();
        let record = fingerprint_challenge_record();
        let trace = generate_program_fingerprint_challenge_trace_v19(&air, &record).unwrap();
        let transcript_trace = fingerprint_challenge_transcript_trace(&air, &record);
        let consumer = TestOnlyProgramFingerprintChallengeConsumerAir {
            app_vk_digest: air.app_vk_digest,
            registry_digest: air.registry_digest,
            relation_digest: air.relation_digest,
            log_height: air.log_height,
            cached_width: air.cached_width,
            challenge_bus: ProgramFingerprintChallengeBusV19::new(PROGRAM_BUS),
        };
        let transcript = TranscriptSourceAir(TranscriptBus::new(TRANSCRIPT_BUS));
        check_air(&air, "Program fingerprint challenge", &trace);
        check_logup(
            &[
                "Program fingerprint challenge".into(),
                "challenge consumer".into(),
                "transcript".into(),
            ],
            &[
                symbolic_interactions(&air),
                symbolic_interactions(&consumer),
                symbolic_interactions(&transcript),
            ],
            &[None, None, None],
            &[
                vec![trace.as_view()],
                vec![trace.as_view()],
                vec![transcript_trace.as_view()],
            ],
            &[vec![], vec![], vec![]],
        );

        let target = ef(120);
        let mut exact_left = [F::ZERO; DIGEST_SIZE];
        exact_left[0] = F::from_u64(PROGRAM_FINGERPRINT_DIGEST_TAG_V19);
        exact_left[1] = F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19);
        exact_left[2] = F::from_usize(air.log_height);
        exact_left[3] = F::from_u32(air.cached_width);
        exact_left[4..].copy_from_slice(target.as_basis_coefficients_slice());
        assert_eq!(
            compute_program_fingerprint_digest_v19(
                air.relation_digest,
                air.log_height as u8,
                air.cached_width,
                target,
            ),
            poseidon2_compress_with_capacity(exact_left, air.relation_digest).0,
        );

        let mut tampered_transcript = transcript_trace.clone();
        let relation_start = 2 + 2 * DIGEST_SIZE;
        let row: &mut TranscriptSourceCols<F> =
            tampered_transcript.row_mut(relation_start).borrow_mut();
        row.value += F::ONE;
        assert!(std::panic::catch_unwind(|| {
            check_logup(
                &[
                    "Program fingerprint challenge".into(),
                    "challenge consumer".into(),
                    "tampered transcript".into(),
                ],
                &[
                    symbolic_interactions(&air),
                    symbolic_interactions(&consumer),
                    symbolic_interactions(&transcript),
                ],
                &[None, None, None],
                &[
                    vec![trace.as_view()],
                    vec![trace.as_view()],
                    vec![tampered_transcript.as_view()],
                ],
                &[vec![], vec![], vec![]],
            )
        })
        .is_err());
    }

    #[test]
    fn honest_degree_two_one_shot_with_program_fingerprint_is_constrained() {
        let air = one_shot_air();
        let record = one_shot_record();
        let trace = generate_one_shot_reduction_trace_v19(&air, &record).unwrap();
        let transcript = one_shot_transcript_trace(&record);
        check_air(&air, "one-shot", &trace);
        check_one_shot_composed(&trace, &trace, &transcript);
    }

    #[test]
    fn grouped_one_shot_same_shape_checks_last_to_first_and_provider_multisets() {
        let air = one_shot_air();
        let first = one_shot_record();
        let mut second = one_shot_record();
        second.shard_ordinal = 1;
        second.source_root = digest(150);
        second.range_start = first.range_end;
        second.range_end = first.range_end + 64;
        second.functional_digest = digest(170);
        second.start_tidx = first.start_tidx + ((4 * air.log_message_len + 1) * D_EF) as u32;
        let records = vec![first, second];
        let trace = generate_grouped_one_shot_reduction_trace_v19(&air, &records).unwrap();
        let transcript = grouped_one_shot_transcript_trace(&records);
        check_air(&air, "grouped one-shot", &trace);
        check_one_shot_composed(&trace, &trace, &transcript);

        let second_start = air.log_message_len;
        let mut bad_reset = trace.clone();
        let row: &mut OneShotReductionColsV19<F> = bad_reset.row_mut(second_start).borrow_mut();
        row.is_first = F::ZERO;
        assert!(
            std::panic::catch_unwind(|| check_air(&air, "bad grouped reset", &bad_reset)).is_err()
        );

        let mut bad_cursor = trace.clone();
        let row: &mut OneShotReductionColsV19<F> = bad_cursor.row_mut(second_start).borrow_mut();
        row.round_tidx += F::ONE;
        assert!(std::panic::catch_unwind(|| {
            check_one_shot_composed(&bad_cursor, &trace, &transcript)
        })
        .is_err());
    }

    #[test]
    fn one_shot_round_functional_point_value_and_transcript_tampering_is_rejected() {
        let air = one_shot_air();
        let record = one_shot_record();
        let honest = generate_one_shot_reduction_trace_v19(&air, &record).unwrap();
        let transcript = one_shot_transcript_trace(&record);
        for mutation in 0..7 {
            let mut trace = honest.clone();
            let row: &mut OneShotReductionColsV19<F> = trace.row_mut(0).borrow_mut();
            match mutation {
                0 => row.coefficients[0][0] += F::ONE,
                1 => row.functional_digest[0] += F::ONE,
                2 => row.ordinary_weight_at_point[0] += F::ONE,
                3 => row.program_fingerprint[0] += F::ONE,
                4 => row.point[0][0] += F::ONE,
                5 => row.message_value[0] += F::ONE,
                6 => row.mix_challenge[0] += F::ONE,
                _ => unreachable!(),
            }
            assert!(
                std::panic::catch_unwind(|| {
                    check_air(&air, "tampered one-shot", &trace);
                    check_one_shot_composed(&trace, &honest, &transcript)
                })
                .is_err(),
                "test-only one-shot mutation {mutation} survived"
            );
        }
        let mut bad_transcript = transcript.clone();
        let row: &mut TranscriptSourceCols<F> = bad_transcript.row_mut(0).borrow_mut();
        row.value += F::ONE;
        assert!(std::panic::catch_unwind(|| {
            check_one_shot_composed(&honest, &honest, &bad_transcript)
        })
        .is_err());
    }

    fn boundary_air() -> LogUpSwirlBoundaryAirV19 {
        LogUpSwirlBoundaryAirV19 {
            segment_start: 0,
            arithmetic_bus: VerifiedLogUpArithmeticBusV19::new(ARITHMETIC_BUS),
            forest_bus: VerifiedSourceForestLeafBusV19::new(FOREST_BUS),
            public_value_bus: VerifiedDirectAirPublicValueBusV19::new(PV_BUS),
            one_shot_bus: VerifiedOneShotRawOpeningBusV19::new(ONE_SHOT_BUS),
            certified_endpoint_bus: CertifiedLogUpOnlyEndpointBusV19::new(ENDPOINT_BUS),
            certified_opening_bus: CertifiedSwirlRawOpeningBusV19::new(OPENING_BUS),
            certified_vm_bus: CertifiedVmSegmentMetadataBusV19::new(VM_BUS),
            certified_program_bus: CertifiedProgramFingerprintBusV19::new(PROGRAM_BUS),
            compress_bus: HistoryPoseidon2CompressBusV19::new(COMPRESS_BUS),
        }
    }

    fn boundary_records() -> Vec<LogUpSwirlBoundaryShardRecordV19> {
        let app_vk = digest(400);
        let registry = digest(420);
        let program_relation_digest = digest(430);
        let program_log_height = 2;
        let program_cached_width = 6;
        let fingerprint = ef(440);
        let fingerprint_digest = compute_program_fingerprint_digest_v19(
            program_relation_digest,
            program_log_height,
            program_cached_width,
            fingerprint,
        );
        let mut records = Vec::new();
        for segment in 0..2u32 {
            for ordinal in 0..4u16 {
                records.push(LogUpSwirlBoundaryShardRecordV19 {
                    proof_index: 7 + segment,
                    segment_index: 7 + segment,
                    shard_ordinal: ordinal,
                    shard_count: 4,
                    air_id: if ordinal == 0 {
                        PROGRAM_AIR_ID as u32
                    } else if ordinal == 1 {
                        CONNECTOR_AIR_ID as u32
                    } else if ordinal == 2 {
                        BOUNDARY_AIR_ID as u32
                    } else {
                        MERKLE_AIR_ID as u32
                    },
                    is_program: ordinal == 0,
                    relation_digest: if ordinal == 0 {
                        program_relation_digest
                    } else {
                        digest(450 + u32::from(ordinal) * 10)
                    },
                    log_height: if ordinal == 0 {
                        program_log_height
                    } else {
                        2 + ordinal as u8
                    },
                    cached_width: if ordinal == 0 {
                        program_cached_width
                    } else {
                        0
                    },
                    log_message_len: if ordinal % 2 == 0 { 2 } else { 3 },
                    app_vk_digest: app_vk,
                    registry_digest: registry,
                    source_forest_root: digest(500 + segment * 20),
                    segment_openings_digest: digest(520 + segment * 20),
                    source_root: digest(600 + segment * 100 + u32::from(ordinal) * 10),
                    range_start: u32::from(ordinal) * 64,
                    range_end: (u32::from(ordinal) + 1) * 64,
                    opening_point: (0..if ordinal % 2 == 0 { 2 } else { 3 })
                        .map(|point| ef(700 + u32::from(ordinal) * 10 + point + segment))
                        .collect(),
                    opening_value: ef(740 + u32::from(ordinal)),
                    verifier_endpoint: ef(760 + segment),
                    segment_sum_before: EF::ZERO,
                    segment_sum_after: EF::ZERO,
                    initial_pc: F::from_u32(100 + segment * 4),
                    final_pc: F::from_u32(104 + segment * 4),
                    exit_code: F::from_u32(if segment == 0 {
                        DEFAULT_SUSPEND_EXIT_CODE
                    } else {
                        0
                    }),
                    is_terminate: segment == 1,
                    initial_memory_root: digest(800 + segment * 20),
                    final_memory_root: digest(820 + segment * 20),
                    program_fingerprint: if ordinal == 0 { fingerprint } else { EF::ZERO },
                    program_fingerprint_digest: fingerprint_digest,
                });
            }
        }
        records
    }

    #[derive(Clone, ColumnsAir)]
    #[columns_via(LogUpSwirlBoundaryColsV19<u8>)]
    struct TestOnlyBoundaryEchoAuthorityAir {
        mode_tag: u32,
        arithmetic_bus: VerifiedLogUpArithmeticBusV19,
        forest_bus: VerifiedSourceForestLeafBusV19,
        pv_bus: VerifiedDirectAirPublicValueBusV19,
        one_shot_bus: VerifiedOneShotRawOpeningBusV19,
        endpoint_bus: CertifiedLogUpOnlyEndpointBusV19,
        opening_bus: CertifiedSwirlRawOpeningBusV19,
        vm_bus: CertifiedVmSegmentMetadataBusV19,
        program_bus: CertifiedProgramFingerprintBusV19,
        compress_bus: HistoryPoseidon2CompressBusV19,
    }

    impl BaseAir<F> for TestOnlyBoundaryEchoAuthorityAir {
        fn width(&self) -> usize {
            LogUpSwirlBoundaryColsV19::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for TestOnlyBoundaryEchoAuthorityAir {}
    impl PartitionedBaseAir<F> for TestOnlyBoundaryEchoAuthorityAir {}

    impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for TestOnlyBoundaryEchoAuthorityAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).unwrap();
            let local: &LogUpSwirlBoundaryColsV19<AB::Var> = (*row).borrow();
            let enabled = AB::Expr::from(local.active);
            let first = enabled.clone() * AB::Expr::from(local.is_segment_first);
            self.arithmetic_bus.send(
                builder,
                VerifiedLogUpArithmeticMessageV19 {
                    proof_index: local.proof_index.into(),
                    protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                    mode_tag: AB::Expr::from_u32(self.mode_tag),
                    segment_index_lo: local.segment_index_lo.into(),
                    segment_index_hi: local.segment_index_hi.into(),
                    app_vk_digest: local.app_vk_digest.map(Into::into),
                    source_forest_root: local.source_forest_root.map(Into::into),
                    segment_openings_digest: local.segment_openings_digest.map(Into::into),
                    verifier_endpoint: local.verifier_endpoint.map(Into::into),
                    segment_sum_before: local.segment_sum_before.map(Into::into),
                    segment_sum_after: local.segment_sum_after.map(Into::into),
                    shard_count: local.shard_count.into(),
                },
                first.clone(),
            );
            self.forest_bus.send(
                builder,
                VerifiedSourceForestLeafMessageV19 {
                    proof_index: local.proof_index.into(),
                    segment_index_lo: local.segment_index_lo.into(),
                    segment_index_hi: local.segment_index_hi.into(),
                    shard_ordinal: local.shard_ordinal.into(),
                    air_id: local.air_id.into(),
                    relation_digest: local.relation_digest.map(Into::into),
                    log_height: local.log_height.into(),
                    cached_width: local.cached_width.into(),
                    log_message_len: local.log_message_len.into(),
                    source_forest_root: local.source_forest_root.map(Into::into),
                    segment_openings_digest: local.segment_openings_digest.map(Into::into),
                    source_root: local.source_root.map(Into::into),
                    range_start: local.range_start.into(),
                    range_end: local.range_end.into(),
                },
                enabled.clone(),
            );
            self.one_shot_bus.send(
                builder,
                VerifiedOneShotRawOpeningMessageV19 {
                    proof_index: local.proof_index.into(),
                    segment_index_lo: local.segment_index_lo.into(),
                    segment_index_hi: local.segment_index_hi.into(),
                    shard_ordinal: local.shard_ordinal.into(),
                    is_program: local.is_program.into(),
                    source_root: local.source_root.map(Into::into),
                    range_start: local.range_start.into(),
                    range_end: local.range_end.into(),
                    point_len: local.point_len.into(),
                    point: local.opening_point.map(|point| point.map(Into::into)),
                    value: local.opening_value.map(Into::into),
                    program_fingerprint: local.program_fingerprint.map(Into::into),
                },
                enabled.clone(),
            );
            self.endpoint_bus.lookup_key(
                builder,
                CertifiedLogUpOnlyEndpointMessageV19 {
                    proof_index: local.proof_index.into(),
                    protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                    mode_tag: AB::Expr::from_u32(LOGUP_ONLY_MODE_TAG_V19),
                    segment_index_lo: local.segment_index_lo.into(),
                    segment_index_hi: local.segment_index_hi.into(),
                    app_vk_digest: local.app_vk_digest.map(Into::into),
                    source_forest_root: local.source_forest_root.map(Into::into),
                    segment_openings_digest: local.segment_openings_digest.map(Into::into),
                    verifier_endpoint: local.verifier_endpoint.map(Into::into),
                    segment_sum_before: local.segment_sum_before.map(Into::into),
                    segment_sum_after: local.segment_sum_after.map(Into::into),
                },
                first.clone(),
            );
            self.opening_bus.lookup_key(
                builder,
                CertifiedSwirlRawOpeningMessageV19 {
                    proof_index: local.proof_index.into(),
                    protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                    mode_tag: AB::Expr::from_u32(LOGUP_ONLY_MODE_TAG_V19),
                    segment_index_lo: local.segment_index_lo.into(),
                    segment_index_hi: local.segment_index_hi.into(),
                    shard_ordinal: local.shard_ordinal.into(),
                    source_forest_root: local.source_forest_root.map(Into::into),
                    segment_openings_digest: local.segment_openings_digest.map(Into::into),
                    root: local.source_root.map(Into::into),
                    point_len: local.point_len.into(),
                    point: local.opening_point.map(|point| point.map(Into::into)),
                    value: local.opening_value.map(Into::into),
                },
                enabled.clone(),
            );
            for (index, value) in [
                local.initial_pc,
                local.final_pc,
                local.exit_code,
                local.is_terminate,
            ]
            .into_iter()
            .enumerate()
            {
                self.pv_bus.send(
                    builder,
                    VerifiedDirectAirPublicValueMessageV19 {
                        proof_index: local.proof_index.into(),
                        segment_index_lo: local.segment_index_lo.into(),
                        segment_index_hi: local.segment_index_hi.into(),
                        air_id: AB::Expr::from_usize(CONNECTOR_AIR_ID),
                        public_value_index: AB::Expr::from_usize(index),
                        value: value.into(),
                    },
                    first.clone(),
                );
            }
            for (index, value) in local
                .initial_memory_root
                .iter()
                .chain(local.final_memory_root.iter())
                .enumerate()
            {
                self.pv_bus.send(
                    builder,
                    VerifiedDirectAirPublicValueMessageV19 {
                        proof_index: local.proof_index.into(),
                        segment_index_lo: local.segment_index_lo.into(),
                        segment_index_hi: local.segment_index_hi.into(),
                        air_id: AB::Expr::from_usize(MERKLE_AIR_ID),
                        public_value_index: AB::Expr::from_usize(index),
                        value: (*value).into(),
                    },
                    first.clone(),
                );
            }
            self.vm_bus.receive(
                builder,
                CertifiedVmSegmentMetadataMessageV19 {
                    protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                    segment_index_lo: local.segment_index_lo.into(),
                    segment_index_hi: local.segment_index_hi.into(),
                    initial_pc: local.initial_pc.into(),
                    final_pc: local.final_pc.into(),
                    exit_code: local.exit_code.into(),
                    is_terminate: local.is_terminate.into(),
                    initial_memory_root: local.initial_memory_root.map(Into::into),
                    final_memory_root: local.final_memory_root.map(Into::into),
                    program_fingerprint: local.stable_program_fingerprint.map(Into::into),
                    program_fingerprint_digest: local.program_fingerprint_digest.map(Into::into),
                    from_vm_state: local.from_vm_state.map(Into::into),
                    to_vm_state: local.to_vm_state.map(Into::into),
                },
                first.clone(),
            );
            self.program_bus.receive(
                builder,
                CertifiedProgramFingerprintMessageV19 {
                    protocol_version: AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                    segment_index_lo: local.segment_index_lo.into(),
                    segment_index_hi: local.segment_index_hi.into(),
                    app_vk_digest: local.app_vk_digest.map(Into::into),
                    registry_digest: local.registry_digest.map(Into::into),
                    relation_digest: local.stable_program_relation_digest.map(Into::into),
                    log_height: local.stable_program_log_height.into(),
                    cached_width: local.stable_program_cached_width.into(),
                    value: local.stable_program_fingerprint.map(Into::into),
                    digest: local.program_fingerprint_digest.map(Into::into),
                },
                first.clone(),
            );
            let left: [AB::Expr; DIGEST_SIZE] = core::array::from_fn(|index| match index {
                0 => AB::Expr::from_u64(PROGRAM_FINGERPRINT_DIGEST_TAG_V19),
                1 => AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                2 => local.stable_program_log_height.into(),
                3 => local.stable_program_cached_width.into(),
                _ => local.stable_program_fingerprint[index - 4].into(),
            });
            self.compress_bus.add_key_with_lookups(
                builder,
                HistoryPoseidon2CompressMessageV19 {
                    input: core::array::from_fn(|index| {
                        if index < DIGEST_SIZE {
                            left[index].clone()
                        } else {
                            local.stable_program_relation_digest[index - DIGEST_SIZE].into()
                        }
                    }),
                    output: local.program_fingerprint_digest.map(Into::into),
                },
                first.clone(),
            );
            let from_meta: [AB::Expr; DIGEST_SIZE] = core::array::from_fn(|index| match index {
                0 => AB::Expr::from_u32(VM_STATE_HASH_TAG_V19),
                1 => AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                2 => local.initial_pc.into(),
                _ => AB::Expr::ZERO,
            });
            let to_meta: [AB::Expr; DIGEST_SIZE] = core::array::from_fn(|index| match index {
                0 => AB::Expr::from_u32(VM_STATE_HASH_TAG_V19),
                1 => AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                2 => local.final_pc.into(),
                3 => local.is_terminate.into(),
                _ => AB::Expr::ZERO,
            });
            for (left, right, output) in [
                (
                    from_meta,
                    local.initial_memory_root.map(Into::into),
                    local.from_vm_inner_digest.map(Into::into),
                ),
                (
                    to_meta,
                    local.final_memory_root.map(Into::into),
                    local.to_vm_inner_digest.map(Into::into),
                ),
                (
                    local.program_fingerprint_digest.map(Into::into),
                    local.from_vm_inner_digest.map(Into::into),
                    local.from_vm_state.map(Into::into),
                ),
                (
                    local.program_fingerprint_digest.map(Into::into),
                    local.to_vm_inner_digest.map(Into::into),
                    local.to_vm_state.map(Into::into),
                ),
            ] {
                self.compress_bus.add_key_with_lookups(
                    builder,
                    HistoryPoseidon2CompressMessageV19 {
                        input: join_digest_expr(left, right),
                        output,
                    },
                    first.clone(),
                );
            }
        }
    }

    fn boundary_authority(mode_tag: u32) -> TestOnlyBoundaryEchoAuthorityAir {
        TestOnlyBoundaryEchoAuthorityAir {
            mode_tag,
            arithmetic_bus: VerifiedLogUpArithmeticBusV19::new(ARITHMETIC_BUS),
            forest_bus: VerifiedSourceForestLeafBusV19::new(FOREST_BUS),
            pv_bus: VerifiedDirectAirPublicValueBusV19::new(PV_BUS),
            one_shot_bus: VerifiedOneShotRawOpeningBusV19::new(ONE_SHOT_BUS),
            endpoint_bus: CertifiedLogUpOnlyEndpointBusV19::new(ENDPOINT_BUS),
            opening_bus: CertifiedSwirlRawOpeningBusV19::new(OPENING_BUS),
            vm_bus: CertifiedVmSegmentMetadataBusV19::new(VM_BUS),
            program_bus: CertifiedProgramFingerprintBusV19::new(PROGRAM_BUS),
            compress_bus: HistoryPoseidon2CompressBusV19::new(COMPRESS_BUS),
        }
    }

    fn check_boundary_composed(
        verifier_trace: &RowMajorMatrix<F>,
        authority_trace: &RowMajorMatrix<F>,
        mode_tag: u32,
    ) {
        let verifier = boundary_air();
        let authority = boundary_authority(mode_tag);
        check_logup(
            &["boundary".into(), "cryptographic authorities".into()],
            &[
                symbolic_interactions(&verifier),
                symbolic_interactions(&authority),
            ],
            &[None, None],
            &[
                vec![verifier_trace.as_view()],
                vec![authority_trace.as_view()],
            ],
            &[vec![], vec![]],
        );
    }

    #[test]
    fn honest_multi_air_semantics_and_stable_program_fingerprint_are_constrained() {
        let records = boundary_records();
        let air = boundary_air();
        let trace = generate_logup_swirl_boundary_trace_v19(&air, &records).unwrap();
        check_air(&air, "LogUp/SWIRL boundary", &trace);
        check_boundary_composed(&trace, &trace, LOGUP_ONLY_MODE_TAG_V19);
    }

    #[test]
    fn inactive_power_of_two_padding_rows_satisfy_boundary_constraints() {
        let mut records = boundary_records();
        records.truncate(4);
        for record in &mut records {
            record.shard_count = 5;
        }
        let mut extra = records[3].clone();
        extra.shard_ordinal = 4;
        extra.air_id = 99;
        extra.relation_digest = digest(999);
        extra.range_start = records[3].range_end;
        extra.range_end = extra.range_start + 64;
        records.push(extra);

        let air = boundary_air();
        let trace = generate_logup_swirl_boundary_trace_v19(&air, &records).unwrap();
        assert_eq!(trace.height(), 8);
        check_air(&air, "LogUp/SWIRL boundary with padding", &trace);
        check_boundary_composed(&trace, &trace, LOGUP_ONLY_MODE_TAG_V19);
    }

    #[test]
    fn mode_endpoint_pvs_fingerprint_forest_omission_and_reorder_tampering_is_rejected() {
        let records = boundary_records();
        let air = boundary_air();
        let honest = generate_logup_swirl_boundary_trace_v19(&air, &records).unwrap();
        assert!(std::panic::catch_unwind(|| {
            check_boundary_composed(&honest, &honest, LOGUP_ONLY_MODE_TAG_V19 ^ 1)
        })
        .is_err());

        for mutation in 0..18 {
            let mut trace = honest.clone();
            let row_index = if mutation == 11 { 1 } else { 0 };
            let row: &mut LogUpSwirlBoundaryColsV19<F> = trace.row_mut(row_index).borrow_mut();
            match mutation {
                0 => row.verifier_endpoint[0] += F::ONE,
                1 => row.initial_pc += F::ONE,
                2 => row.final_pc += F::ONE,
                3 => row.initial_memory_root[0] += F::ONE,
                4 => row.final_memory_root[0] += F::ONE,
                5 => row.is_terminate = F::ONE - row.is_terminate,
                6 => row.stable_program_fingerprint[0] += F::ONE,
                7 => row.program_fingerprint_digest[0] += F::ONE,
                8 => row.source_forest_root[0] += F::ONE,
                9 => row.source_root[0] += F::ONE,
                10 => row.range_end += F::ONE,
                11 => row.shard_ordinal += F::ONE,
                12 => row.stable_program_relation_digest[0] += F::ONE,
                13 => row.stable_program_log_height += F::ONE,
                14 => row.stable_program_cached_width += F::ONE,
                15 => {
                    row.point_len += F::ONE;
                    row.point_len_selector.swap(1, 2);
                }
                16 => row.log_message_len += F::ONE,
                17 => row.opening_point[MAX_RAW_MESSAGE_POINT_LEN_V19 - 1][0] = F::ONE,
                _ => unreachable!(),
            }
            assert!(std::panic::catch_unwind(|| {
                check_boundary_composed(&trace, &honest, LOGUP_ONLY_MODE_TAG_V19)
            })
            .is_err());
        }

        let mut omitted = honest.clone();
        let last_of_first: &mut LogUpSwirlBoundaryColsV19<F> = omitted.row_mut(3).borrow_mut();
        last_of_first.active = F::ZERO;
        assert!(std::panic::catch_unwind(|| {
            check_boundary_composed(&omitted, &honest, LOGUP_ONLY_MODE_TAG_V19)
        })
        .is_err());
    }
}

//! Canonical 108-field interval record and host-side merge oracle.

use core::ops::Range;

use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, F};

use crate::circuit::verifier_warp_history_v2::VerifierWarpHistoryChunkPublicValuesV3;

/// Number of field elements in one authenticated History-v3 interval.
pub const VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3: usize = 108;

// Canonical offsets in `VerifierWarpHistoryChunkPublicValuesV3::to_vec()`.
// Keep these explicit: recursive composition is a security boundary, and a
// reordered or silently omitted field must change either a constraint or a
// typed-bus key.
pub const CHUNK_PROTOCOL_VERSION_V3: usize = 0;
pub const SOURCE_HISTORY_PROTOCOL_VERSION_V3: usize = 1;
pub const SOURCE_CHILD_CAPACITY_V3: usize = 2;
pub const VACC_INPUT_ARITY_V3: usize = 3;
pub const PROTOCOL_DIGEST_V3: Range<usize> = 4..12;
pub const RELATION_DIGEST_V3: Range<usize> = 12..20;
pub const PROGRAM_COMMITMENT_V3: Range<usize> = 20..28;
pub const FIXED_BINDINGS_V3: Range<usize> = 0..28;
pub const TOTAL_BATCH_COUNT_V3: Range<usize> = 28..30;
pub const CHUNK_INDEX_V3: Range<usize> = 30..32;
pub const START_BATCH_INDEX_V3: Range<usize> = 32..34;
pub const END_BATCH_INDEX_V3: Range<usize> = 34..36;
pub const ACTIVE_CHILD_COUNT_BEFORE_V3: Range<usize> = 36..40;
pub const ACTIVE_CHILD_COUNT_AFTER_V3: Range<usize> = 40..44;
pub const INITIAL_PC_V3: usize = 44;
pub const INITIAL_MEMORY_ROOT_V3: Range<usize> = 45..53;
pub const FINAL_PC_V3: usize = 53;
pub const FINAL_MEMORY_ROOT_V3: Range<usize> = 54..62;
pub const INITIAL_ACCUMULATOR_DIGEST_V3: Range<usize> = 62..70;
pub const FINAL_ACCUMULATOR_DIGEST_V3: Range<usize> = 70..78;
pub const HISTORY_HASH_BEFORE_V3: Range<usize> = 78..86;
pub const HISTORY_HASH_AFTER_V3: Range<usize> = 86..94;
pub const OBSERVATION_COUNT_BEFORE_V3: Range<usize> = 94..96;
pub const OBSERVATION_COUNT_AFTER_V3: Range<usize> = 96..98;
pub const IS_GENESIS_V3: usize = 98;
pub const IS_TERMINAL_V3: usize = 99;
pub const PUBLIC_VALUES_DIGEST_V3: Range<usize> = 100..108;

/// Typed key carried on the recursive interval buses.
///
/// The array representation is intentional: its length is the protocol
/// width, so adding a field to the public statement cannot compile without
/// updating this module's width assertion and mapping tests.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug, PartialEq, Eq)]
pub struct VerifierWarpHistoryChunkIntervalMessageV3<T> {
    pub fields: [T; VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3],
}

impl VerifierWarpHistoryChunkIntervalMessageV3<F> {
    #[must_use]
    pub fn from_public_values(values: VerifierWarpHistoryChunkPublicValuesV3) -> Self {
        let fields: [F; VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3] = values
            .to_vec()
            .try_into()
            .expect("History-v3 public-value width is a compile-time protocol constant");
        Self { fields }
    }
}

/// Two ordered, already-authenticated child interval statements.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifierWarpHistoryChunkCompositionRecordV3 {
    pub left: VerifierWarpHistoryChunkPublicValuesV3,
    pub right: VerifierWarpHistoryChunkPublicValuesV3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifierWarpHistoryChunkCompositionErrorV3 {
    FixedBinding(usize),
    TotalBatchCount,
    BatchBoundary,
    VmBoundary,
    AccumulatorBoundary,
    HistoryHashBoundary,
    ActiveChildCountBoundary,
    ObservationCountBoundary,
    ChunkIndexOverflow,
    ChunkIndexOrder,
    RightIsGenesis,
    LeftIsTerminal,
    LeftPublicValues(usize),
    RightNonterminalPublicValues(usize),
}

impl VerifierWarpHistoryChunkCompositionRecordV3 {
    /// Slow host oracle for the AIR's interval merge.
    ///
    /// `chunk_index` denotes the rightmost leaf chunk in a composed interval.
    /// Consequently this core supports a recursive left fold: the right child
    /// is the next direct chunk, and the merged statement can be used as the
    /// next left child without resetting ordered History state.
    pub fn merged(
        &self,
    ) -> Result<VerifierWarpHistoryChunkPublicValuesV3, VerifierWarpHistoryChunkCompositionErrorV3>
    {
        let left_fields = self.left.to_vec();
        let right_fields = self.right.to_vec();
        for index in FIXED_BINDINGS_V3 {
            if left_fields[index] != right_fields[index] {
                return Err(VerifierWarpHistoryChunkCompositionErrorV3::FixedBinding(
                    index,
                ));
            }
        }
        if self.left.total_batch_count != self.right.total_batch_count {
            return Err(VerifierWarpHistoryChunkCompositionErrorV3::TotalBatchCount);
        }
        if self.left.end_batch_index != self.right.start_batch_index {
            return Err(VerifierWarpHistoryChunkCompositionErrorV3::BatchBoundary);
        }
        if self.left.final_state != self.right.initial_state {
            return Err(VerifierWarpHistoryChunkCompositionErrorV3::VmBoundary);
        }
        if self.left.final_accumulator_digest != self.right.initial_accumulator_digest {
            return Err(VerifierWarpHistoryChunkCompositionErrorV3::AccumulatorBoundary);
        }
        if self.left.history_hash_after != self.right.history_hash_before {
            return Err(VerifierWarpHistoryChunkCompositionErrorV3::HistoryHashBoundary);
        }
        if self.left.active_child_count_after != self.right.active_child_count_before {
            return Err(VerifierWarpHistoryChunkCompositionErrorV3::ActiveChildCountBoundary);
        }
        if self.left.observation_count_after != self.right.observation_count_before {
            return Err(VerifierWarpHistoryChunkCompositionErrorV3::ObservationCountBoundary);
        }
        let next_chunk_index = self
            .left
            .chunk_index
            .checked_add(1)
            .ok_or(VerifierWarpHistoryChunkCompositionErrorV3::ChunkIndexOverflow)?;
        if self.right.chunk_index != next_chunk_index {
            return Err(VerifierWarpHistoryChunkCompositionErrorV3::ChunkIndexOrder);
        }
        if self.right.is_genesis {
            return Err(VerifierWarpHistoryChunkCompositionErrorV3::RightIsGenesis);
        }
        if self.left.is_terminal {
            return Err(VerifierWarpHistoryChunkCompositionErrorV3::LeftIsTerminal);
        }
        for (index, value) in self.left.public_values_digest.iter().enumerate() {
            if *value != F::ZERO {
                return Err(VerifierWarpHistoryChunkCompositionErrorV3::LeftPublicValues(index));
            }
        }
        if !self.right.is_terminal {
            for (index, value) in self.right.public_values_digest.iter().enumerate() {
                if *value != F::ZERO {
                    return Err(
                        VerifierWarpHistoryChunkCompositionErrorV3::RightNonterminalPublicValues(
                            index,
                        ),
                    );
                }
            }
        }

        Ok(VerifierWarpHistoryChunkPublicValuesV3 {
            chunk_protocol_version: self.left.chunk_protocol_version,
            source_history_protocol_version: self.left.source_history_protocol_version,
            source_child_capacity: self.left.source_child_capacity,
            vacc_input_arity: self.left.vacc_input_arity,
            protocol_digest: self.left.protocol_digest,
            relation_digest: self.left.relation_digest,
            program_commitment: self.left.program_commitment,
            total_batch_count: self.left.total_batch_count,
            chunk_index: self.right.chunk_index,
            start_batch_index: self.left.start_batch_index,
            end_batch_index: self.right.end_batch_index,
            active_child_count_before: self.left.active_child_count_before,
            active_child_count_after: self.right.active_child_count_after,
            initial_state: self.left.initial_state,
            final_state: self.right.final_state,
            initial_accumulator_digest: self.left.initial_accumulator_digest,
            final_accumulator_digest: self.right.final_accumulator_digest,
            history_hash_before: self.left.history_hash_before,
            history_hash_after: self.right.history_hash_after,
            observation_count_before: self.left.observation_count_before,
            observation_count_after: self.right.observation_count_after,
            is_genesis: self.left.is_genesis,
            is_terminal: self.right.is_terminal,
            public_values_digest: self.right.public_values_digest,
        })
    }
}

const _: () = assert!(DIGEST_SIZE == 8);
const _: () = assert!(
    VerifierWarpHistoryChunkPublicValuesV3::WIDTH == VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3
);
const _: () = assert!(PUBLIC_VALUES_DIGEST_V3.end == VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3);

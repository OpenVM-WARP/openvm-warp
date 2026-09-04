use openvm_stark_backend::p3_field::{PrimeCharacteristicRing, PrimeField32};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, F};

use super::{
    VERIFIER_WARP_HISTORY_CHUNK_PROTOCOL_V3, VERIFIER_WARP_HISTORY_PROTOCOL_V2,
    VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2, VERIFIER_WARP_VACC_INPUT_ARITY_V2,
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifierWarpVmStateV2 {
    pub pc: F,
    pub memory_root: Digest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifierWarpChildRecordV2 {
    pub occupied: bool,
    pub input: VerifierWarpVmStateV2,
    pub output: VerifierWarpVmStateV2,
    pub exit_code: F,
    pub terminates: bool,
}

impl VerifierWarpChildRecordV2 {
    #[must_use]
    pub const fn padding() -> Self {
        Self {
            occupied: false,
            input: VerifierWarpVmStateV2 {
                pc: F::ZERO,
                memory_root: [F::ZERO; DIGEST_SIZE],
            },
            output: VerifierWarpVmStateV2 {
                pc: F::ZERO,
                memory_root: [F::ZERO; DIGEST_SIZE],
            },
            exit_code: F::ZERO,
            terminates: false,
        }
    }
}

/// One completely authenticated source/VACC transition statement.
///
/// This is witness data for the History AIR, not an authority object.  The
/// source and VACC fields are also consumed on typed buses whose only senders
/// in the final MultiSTARK VK are the real verifier compositions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierWarpHistoryTransitionRecordV2 {
    pub protocol_version: u32,
    pub vacc_input_arity: u8,
    pub protocol_digest: Digest,
    pub relation_digest: Digest,
    pub batch_index: u32,
    pub active_child_count: u8,
    pub children: [VerifierWarpChildRecordV2; VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2],
    pub program_commitment: Digest,
    pub prior_accumulator_digest: Digest,
    pub source_accumulator_digest: Digest,
    pub output_accumulator_digest: Digest,
    pub source_commitment_root: Digest,
    pub external_logup_gkr_digest: Digest,
    pub source_functional_digest: Digest,
    pub setup_openings_digest: Digest,
    pub source_statement_digest: Digest,
    pub transition_transcript_digest: Digest,
    pub public_values_digest: Digest,
}

impl VerifierWarpHistoryTransitionRecordV2 {
    #[must_use]
    pub fn has_fixed_protocol_dimensions(&self) -> bool {
        self.protocol_version == VERIFIER_WARP_HISTORY_PROTOCOL_V2
            && usize::from(self.vacc_input_arity) == VERIFIER_WARP_VACC_INPUT_ARITY_V2
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifierWarpHistoryPublicValuesV2 {
    pub protocol_version: u32,
    pub source_child_capacity: u32,
    pub vacc_input_arity: u32,
    pub protocol_digest: Digest,
    pub relation_digest: Digest,
    pub program_commitment: Digest,
    pub batch_count: u32,
    pub active_child_count: u64,
    pub initial_state: VerifierWarpVmStateV2,
    pub final_state: VerifierWarpVmStateV2,
    pub genesis_accumulator_digest: Digest,
    pub final_accumulator_digest: Digest,
    pub public_values_digest: Digest,
    pub history_statement_digest: Digest,
}

impl VerifierWarpHistoryPublicValuesV2 {
    pub const WIDTH: usize = 83;

    #[must_use]
    pub fn to_vec(self) -> Vec<F> {
        let mut values = Vec::with_capacity(Self::WIDTH);
        values.extend([
            F::from_u32(self.protocol_version),
            F::from_u32(self.source_child_capacity),
            F::from_u32(self.vacc_input_arity),
        ]);
        values.extend(self.protocol_digest);
        values.extend(self.relation_digest);
        values.extend(self.program_commitment);
        values.extend([
            F::from_u16(self.batch_count as u16),
            F::from_u16((self.batch_count >> 16) as u16),
        ]);
        for shift in [0, 16, 32, 48] {
            values.push(F::from_u16((self.active_child_count >> shift) as u16));
        }
        values.push(self.initial_state.pc);
        values.extend(self.initial_state.memory_root);
        values.push(self.final_state.pc);
        values.extend(self.final_state.memory_root);
        values.extend(self.genesis_accumulator_digest);
        values.extend(self.final_accumulator_digest);
        values.extend(self.public_values_digest);
        values.extend(self.history_statement_digest);
        debug_assert_eq!(values.len(), Self::WIDTH);
        values
    }
}

impl Default for VerifierWarpHistoryPublicValuesV2 {
    fn default() -> Self {
        Self {
            protocol_version: VERIFIER_WARP_HISTORY_PROTOCOL_V2,
            source_child_capacity: VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2 as u32,
            vacc_input_arity: VERIFIER_WARP_VACC_INPUT_ARITY_V2 as u32,
            protocol_digest: [F::ZERO; DIGEST_SIZE],
            relation_digest: [F::ZERO; DIGEST_SIZE],
            program_commitment: [F::ZERO; DIGEST_SIZE],
            batch_count: 0,
            active_child_count: 0,
            initial_state: VerifierWarpVmStateV2 {
                pc: F::ZERO,
                memory_root: [F::ZERO; DIGEST_SIZE],
            },
            final_state: VerifierWarpVmStateV2 {
                pc: F::ZERO,
                memory_root: [F::ZERO; DIGEST_SIZE],
            },
            genesis_accumulator_digest: [F::ZERO; DIGEST_SIZE],
            final_accumulator_digest: [F::ZERO; DIGEST_SIZE],
            public_values_digest: [F::ZERO; DIGEST_SIZE],
            history_statement_digest: [F::ZERO; DIGEST_SIZE],
        }
    }
}

/// Incoming state of one bounded History interval.
///
/// Every field is constrained by the History AIR and is re-exposed in the
/// chunk public values.  A recursive composition AIR must equate these fields
/// to the preceding chunk's outgoing fields; they are never host authority.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifierWarpHistoryChunkContextV3 {
    pub total_batch_count: u32,
    pub chunk_index: u32,
    pub start_batch_index: u32,
    pub active_child_count_before: u64,
    pub observation_count_before: u32,
    pub history_hash_before: Digest,
    pub is_genesis: bool,
    pub is_terminal: bool,
}

/// Public interval statement of one independently proved bounded History
/// chunk.  It contains both endpoints needed for exhaustive recursive
/// composition; no opaque payload or host-computed endpoint is accepted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifierWarpHistoryChunkPublicValuesV3 {
    pub chunk_protocol_version: u32,
    pub source_history_protocol_version: u32,
    pub source_child_capacity: u32,
    pub vacc_input_arity: u32,
    pub protocol_digest: Digest,
    pub relation_digest: Digest,
    pub program_commitment: Digest,
    pub total_batch_count: u32,
    pub chunk_index: u32,
    pub start_batch_index: u32,
    pub end_batch_index: u32,
    pub active_child_count_before: u64,
    pub active_child_count_after: u64,
    pub initial_state: VerifierWarpVmStateV2,
    pub final_state: VerifierWarpVmStateV2,
    pub initial_accumulator_digest: Digest,
    pub final_accumulator_digest: Digest,
    pub history_hash_before: Digest,
    pub history_hash_after: Digest,
    pub observation_count_before: u32,
    pub observation_count_after: u32,
    pub is_genesis: bool,
    pub is_terminal: bool,
    pub public_values_digest: Digest,
}

impl VerifierWarpHistoryChunkPublicValuesV3 {
    pub const WIDTH: usize = 108;

    #[must_use]
    pub fn to_vec(self) -> Vec<F> {
        let mut values = Vec::with_capacity(Self::WIDTH);
        values.extend([
            F::from_u32(self.chunk_protocol_version),
            F::from_u32(self.source_history_protocol_version),
            F::from_u32(self.source_child_capacity),
            F::from_u32(self.vacc_input_arity),
        ]);
        values.extend(self.protocol_digest);
        values.extend(self.relation_digest);
        values.extend(self.program_commitment);
        for value in [
            self.total_batch_count,
            self.chunk_index,
            self.start_batch_index,
            self.end_batch_index,
        ] {
            values.extend([F::from_u16(value as u16), F::from_u16((value >> 16) as u16)]);
        }
        for value in [
            self.active_child_count_before,
            self.active_child_count_after,
        ] {
            for shift in [0, 16, 32, 48] {
                values.push(F::from_u16((value >> shift) as u16));
            }
        }
        values.push(self.initial_state.pc);
        values.extend(self.initial_state.memory_root);
        values.push(self.final_state.pc);
        values.extend(self.final_state.memory_root);
        values.extend(self.initial_accumulator_digest);
        values.extend(self.final_accumulator_digest);
        values.extend(self.history_hash_before);
        values.extend(self.history_hash_after);
        for value in [self.observation_count_before, self.observation_count_after] {
            values.extend([F::from_u16(value as u16), F::from_u16((value >> 16) as u16)]);
        }
        values.extend([
            F::from_bool(self.is_genesis),
            F::from_bool(self.is_terminal),
        ]);
        values.extend(self.public_values_digest);
        debug_assert_eq!(values.len(), Self::WIDTH);
        values
    }

    /// Parse untrusted proof public values without panicking or accepting
    /// noncanonical 16-bit limb/boolean encodings.
    pub fn try_from_slice(values: &[F]) -> Result<Self, VerifierWarpHistoryChunkPvsErrorV3> {
        if values.len() != Self::WIDTH {
            return Err(VerifierWarpHistoryChunkPvsErrorV3::Width {
                expected: Self::WIDTH,
                actual: values.len(),
            });
        }
        let mut cursor = 0usize;
        let chunk_protocol_version = take_at(values, &mut cursor).as_canonical_u32();
        let source_history_protocol_version = take_at(values, &mut cursor).as_canonical_u32();
        let source_child_capacity = take_at(values, &mut cursor).as_canonical_u32();
        let vacc_input_arity = take_at(values, &mut cursor).as_canonical_u32();
        if chunk_protocol_version != VERIFIER_WARP_HISTORY_CHUNK_PROTOCOL_V3
            || source_history_protocol_version != VERIFIER_WARP_HISTORY_PROTOCOL_V2
            || source_child_capacity != VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2 as u32
            || vacc_input_arity != VERIFIER_WARP_VACC_INPUT_ARITY_V2 as u32
        {
            return Err(VerifierWarpHistoryChunkPvsErrorV3::Protocol);
        }
        let protocol_digest = take_digest(values, &mut cursor);
        let relation_digest = take_digest(values, &mut cursor);
        let program_commitment = take_digest(values, &mut cursor);
        let total_batch_count = take_u32(values, &mut cursor)?;
        let chunk_index = take_u32(values, &mut cursor)?;
        let start_batch_index = take_u32(values, &mut cursor)?;
        let end_batch_index = take_u32(values, &mut cursor)?;
        let active_child_count_before = take_u64(values, &mut cursor)?;
        let active_child_count_after = take_u64(values, &mut cursor)?;
        let initial_state = VerifierWarpVmStateV2 {
            pc: take_at(values, &mut cursor),
            memory_root: take_digest(values, &mut cursor),
        };
        let final_state = VerifierWarpVmStateV2 {
            pc: take_at(values, &mut cursor),
            memory_root: take_digest(values, &mut cursor),
        };
        let initial_accumulator_digest = take_digest(values, &mut cursor);
        let final_accumulator_digest = take_digest(values, &mut cursor);
        let history_hash_before = take_digest(values, &mut cursor);
        let history_hash_after = take_digest(values, &mut cursor);
        let observation_count_before = take_u32(values, &mut cursor)?;
        let observation_count_after = take_u32(values, &mut cursor)?;
        let is_genesis = take_bool(values, &mut cursor)?;
        let is_terminal = take_bool(values, &mut cursor)?;
        let public_values_digest = take_digest(values, &mut cursor);
        debug_assert_eq!(cursor, Self::WIDTH);
        Ok(Self {
            chunk_protocol_version,
            source_history_protocol_version,
            source_child_capacity,
            vacc_input_arity,
            protocol_digest,
            relation_digest,
            program_commitment,
            total_batch_count,
            chunk_index,
            start_batch_index,
            end_batch_index,
            active_child_count_before,
            active_child_count_after,
            initial_state,
            final_state,
            initial_accumulator_digest,
            final_accumulator_digest,
            history_hash_before,
            history_hash_after,
            observation_count_before,
            observation_count_after,
            is_genesis,
            is_terminal,
            public_values_digest,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifierWarpHistoryChunkPvsErrorV3 {
    Width { expected: usize, actual: usize },
    Protocol,
    NonCanonicalLimb { index: usize },
    NonBoolean { index: usize },
}

fn take_at(values: &[F], cursor: &mut usize) -> F {
    let value = values[*cursor];
    *cursor += 1;
    value
}

fn take_digest(values: &[F], cursor: &mut usize) -> Digest {
    core::array::from_fn(|_| take_at(values, cursor))
}

fn take_u16(values: &[F], cursor: &mut usize) -> Result<u16, VerifierWarpHistoryChunkPvsErrorV3> {
    let index = *cursor;
    let value = take_at(values, cursor).as_canonical_u32();
    u16::try_from(value).map_err(|_| VerifierWarpHistoryChunkPvsErrorV3::NonCanonicalLimb { index })
}

fn take_u32(values: &[F], cursor: &mut usize) -> Result<u32, VerifierWarpHistoryChunkPvsErrorV3> {
    let lo = u32::from(take_u16(values, cursor)?);
    let hi = u32::from(take_u16(values, cursor)?);
    Ok(lo | (hi << 16))
}

fn take_u64(values: &[F], cursor: &mut usize) -> Result<u64, VerifierWarpHistoryChunkPvsErrorV3> {
    let mut value = 0u64;
    for shift in [0, 16, 32, 48] {
        value |= u64::from(take_u16(values, cursor)?) << shift;
    }
    Ok(value)
}

fn take_bool(values: &[F], cursor: &mut usize) -> Result<bool, VerifierWarpHistoryChunkPvsErrorV3> {
    let index = *cursor;
    match take_at(values, cursor).as_canonical_u32() {
        0 => Ok(false),
        1 => Ok(true),
        _ => Err(VerifierWarpHistoryChunkPvsErrorV3::NonBoolean { index }),
    }
}

impl Default for VerifierWarpHistoryChunkPublicValuesV3 {
    fn default() -> Self {
        Self {
            chunk_protocol_version: VERIFIER_WARP_HISTORY_CHUNK_PROTOCOL_V3,
            source_history_protocol_version: VERIFIER_WARP_HISTORY_PROTOCOL_V2,
            source_child_capacity: VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2 as u32,
            vacc_input_arity: VERIFIER_WARP_VACC_INPUT_ARITY_V2 as u32,
            protocol_digest: [F::ZERO; DIGEST_SIZE],
            relation_digest: [F::ZERO; DIGEST_SIZE],
            program_commitment: [F::ZERO; DIGEST_SIZE],
            total_batch_count: 0,
            chunk_index: 0,
            start_batch_index: 0,
            end_batch_index: 0,
            active_child_count_before: 0,
            active_child_count_after: 0,
            initial_state: VerifierWarpVmStateV2 {
                pc: F::ZERO,
                memory_root: [F::ZERO; DIGEST_SIZE],
            },
            final_state: VerifierWarpVmStateV2 {
                pc: F::ZERO,
                memory_root: [F::ZERO; DIGEST_SIZE],
            },
            initial_accumulator_digest: [F::ZERO; DIGEST_SIZE],
            final_accumulator_digest: [F::ZERO; DIGEST_SIZE],
            history_hash_before: [F::ZERO; DIGEST_SIZE],
            history_hash_after: [F::ZERO; DIGEST_SIZE],
            observation_count_before: 0,
            observation_count_after: 0,
            is_genesis: false,
            is_terminal: false,
            public_values_digest: [F::ZERO; DIGEST_SIZE],
        }
    }
}

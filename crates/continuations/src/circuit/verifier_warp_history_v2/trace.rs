use core::borrow::BorrowMut;

use openvm_circuit::system::connector::DEFAULT_SUSPEND_EXIT_CODE;
use openvm_stark_backend::{
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_matrix::dense::RowMajorMatrix,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, Digest, DIGEST_SIZE, F,
};

use super::{
    VerifierWarpHistoryChunkContextV3, VerifierWarpHistoryChunkPublicValuesV3,
    VerifierWarpHistoryPublicValuesV2, VerifierWarpHistoryRowColsV2,
    VerifierWarpHistoryTransitionRecordV2, HISTORY_END_TAG_V2, HISTORY_ROW_HASH_STEP_COUNT_V2,
    HISTORY_START_TAG_V2, MANIFEST_ACCUMULATORS_TAG_V2, MANIFEST_CHILDREN_TAG_V2,
    MANIFEST_CHILD_TAG_V2, MANIFEST_END_TAG_V2, MANIFEST_HEADER_TAG_V2,
    MANIFEST_OBSERVATION_COUNT_V2, MANIFEST_PUBLIC_VALUES_TAG_V2, MANIFEST_SOURCE_AUTH_TAG_V2,
    MANIFEST_SOURCE_TAG_V2, MANIFEST_START_TAG_V2, STATEMENT_DIGEST_COMMITMENT_TAG_V2,
    STATEMENT_DIGEST_END_TAG_V2, STATEMENT_DIGEST_FIELD_TAG_V2, STATEMENT_DIGEST_START_TAG_V2,
    VERIFIER_WARP_HISTORY_CHUNK_PROTOCOL_V3, VERIFIER_WARP_HISTORY_PROTOCOL_V2,
    VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2, VERIFIER_WARP_VACC_INPUT_ARITY_V2,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VerifierWarpHistoryTraceErrorV2 {
    Empty,
    Protocol(usize),
    BatchIndex(usize),
    ActivePrefix(usize),
    ChildBoundary(usize),
    FixedBinding(usize),
    Continuity(usize),
    PrematureFinal(usize),
    MissingTermination,
    ChunkContext,
    Overflow,
}

pub struct VerifierWarpHistoryTraceV2 {
    pub matrix: RowMajorMatrix<F>,
    pub public_values: VerifierWarpHistoryPublicValuesV2,
    pub compression_inputs: Vec<[F; 2 * DIGEST_SIZE]>,
}

pub struct VerifierWarpHistoryChunkTraceV3 {
    pub matrix: RowMajorMatrix<F>,
    pub public_values: VerifierWarpHistoryChunkPublicValuesV3,
    pub compression_inputs: Vec<[F; 2 * DIGEST_SIZE]>,
}

#[derive(Clone, Copy)]
enum ObservationV2 {
    Field(F),
    Digest(Digest),
}

pub fn generate_verifier_warp_history_trace_v2(
    records: &[VerifierWarpHistoryTransitionRecordV2],
) -> Result<VerifierWarpHistoryTraceV2, VerifierWarpHistoryTraceErrorV2> {
    let total_batch_count =
        u32::try_from(records.len()).map_err(|_| VerifierWarpHistoryTraceErrorV2::Overflow)?;
    let context = VerifierWarpHistoryChunkContextV3 {
        total_batch_count,
        chunk_index: 0,
        start_batch_index: 0,
        active_child_count_before: 0,
        observation_count_before: 0,
        history_hash_before: verifier_warp_history_initial_hash_v2(),
        is_genesis: true,
        is_terminal: true,
    };
    let chunk = generate_verifier_warp_history_chunk_trace_v3(records, context)?;
    let public_values = VerifierWarpHistoryPublicValuesV2 {
        protocol_version: VERIFIER_WARP_HISTORY_PROTOCOL_V2,
        source_child_capacity: VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2 as u32,
        vacc_input_arity: VERIFIER_WARP_VACC_INPUT_ARITY_V2 as u32,
        protocol_digest: chunk.public_values.protocol_digest,
        relation_digest: chunk.public_values.relation_digest,
        program_commitment: chunk.public_values.program_commitment,
        batch_count: chunk.public_values.total_batch_count,
        active_child_count: chunk.public_values.active_child_count_after,
        initial_state: chunk.public_values.initial_state,
        final_state: chunk.public_values.final_state,
        genesis_accumulator_digest: chunk.public_values.initial_accumulator_digest,
        final_accumulator_digest: chunk.public_values.final_accumulator_digest,
        public_values_digest: chunk.public_values.public_values_digest,
        history_statement_digest: chunk.public_values.history_hash_after,
    };
    Ok(VerifierWarpHistoryTraceV2 {
        matrix: chunk.matrix,
        public_values,
        compression_inputs: chunk.compression_inputs,
    })
}

#[must_use]
pub fn verifier_warp_history_initial_hash_v2() -> Digest {
    let mut state = [F::ZERO; DIGEST_SIZE];
    state[0] = F::from_u32(STATEMENT_DIGEST_START_TAG_V2);
    state[1] = F::from_u32(VERIFIER_WARP_HISTORY_PROTOCOL_V2);
    state
}

/// Generate one independently provable bounded interval without resetting the
/// ordered manifest state.  The returned public values are the sole authority
/// for recursive composition of adjacent intervals.
pub fn generate_verifier_warp_history_chunk_trace_v3(
    records: &[VerifierWarpHistoryTransitionRecordV2],
    context: VerifierWarpHistoryChunkContextV3,
) -> Result<VerifierWarpHistoryChunkTraceV3, VerifierWarpHistoryTraceErrorV2> {
    validate_records(records, &context)?;
    let first = records
        .first()
        .ok_or(VerifierWarpHistoryTraceErrorV2::Empty)?;
    let last = records
        .last()
        .ok_or(VerifierWarpHistoryTraceErrorV2::Empty)?;
    let first_child = first.children[0];
    let last_child = last.children[0];
    let chunk_len =
        u32::try_from(records.len()).map_err(|_| VerifierWarpHistoryTraceErrorV2::Overflow)?;
    let end_batch_index = context
        .start_batch_index
        .checked_add(chunk_len)
        .ok_or(VerifierWarpHistoryTraceErrorV2::Overflow)?;

    let width = core::mem::size_of::<VerifierWarpHistoryRowColsV2<u8>>();
    let height = records.len().next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    let mut compression_inputs = Vec::new();
    let mut history_state = context.history_hash_before;
    let mut active_total = context.active_child_count_before;
    let mut observation_count = context.observation_count_before;

    for (row_index, record) in records.iter().enumerate() {
        let is_first = row_index == 0;
        let is_last = row_index + 1 == records.len();
        let row = &mut values[row_index * width..(row_index + 1) * width];
        let cols: &mut VerifierWarpHistoryRowColsV2<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.is_first = F::from_bool(is_first);
        cols.is_last = F::from_bool(is_last);
        cols.protocol_version = F::from_u32(record.protocol_version);
        cols.vacc_input_arity = F::from_u8(record.vacc_input_arity);
        cols.protocol_digest = record.protocol_digest;
        cols.relation_digest = record.relation_digest;
        cols.batch_index = u32_limbs(record.batch_index);
        set_limb_bits(&mut cols.batch_index_bits, &cols.batch_index);
        cols.batch_index_carry = F::from_bool((record.batch_index as u16) == u16::MAX);
        cols.active_child_count = F::from_u8(record.active_child_count);
        cols.active_count_flags[usize::from(record.active_child_count - 1)] = F::ONE;
        for (child_cols, child) in cols.children.iter_mut().zip(record.children) {
            child_cols.occupied = F::from_bool(child.occupied);
            child_cols.input.pc = child.input.pc;
            child_cols.input.memory_root = child.input.memory_root;
            child_cols.output.pc = child.output.pc;
            child_cols.output.memory_root = child.output.memory_root;
            child_cols.exit_code = child.exit_code;
            child_cols.terminates = F::from_bool(child.terminates);
        }
        cols.program_commitment = record.program_commitment;
        cols.prior_accumulator_digest = record.prior_accumulator_digest;
        cols.source_accumulator_digest = record.source_accumulator_digest;
        cols.output_accumulator_digest = record.output_accumulator_digest;
        cols.source_commitment_root = record.source_commitment_root;
        cols.external_logup_gkr_digest = record.external_logup_gkr_digest;
        cols.source_functional_digest = record.source_functional_digest;
        cols.setup_openings_digest = record.setup_openings_digest;
        cols.source_statement_digest = record.source_statement_digest;
        cols.transition_transcript_digest = record.transition_transcript_digest;
        cols.public_values_digest = record.public_values_digest;

        cols.active_total_before = u64_limbs(active_total);
        let next_active_total = active_total + u64::from(record.active_child_count);
        cols.active_total_after = u64_limbs(next_active_total);
        set_limb_bits(
            &mut cols.active_total_before_bits,
            &cols.active_total_before,
        );
        set_limb_bits(&mut cols.active_total_after_bits, &cols.active_total_after);
        set_add_carries(
            &mut cols.active_total_carries,
            &cols.active_total_before,
            u64::from(record.active_child_count),
        );
        active_total = next_active_total;

        cols.observation_count_before = u32_limbs(observation_count);
        let observation_delta = u32::try_from(MANIFEST_OBSERVATION_COUNT_V2)
            .expect("manifest count fits u32")
            + if is_first && context.is_genesis { 6 } else { 0 }
            + u32::from(is_last && context.is_terminal);
        let next_observation_count = observation_count
            .checked_add(observation_delta)
            .ok_or(VerifierWarpHistoryTraceErrorV2::Overflow)?;
        cols.observation_count_after = u32_limbs(next_observation_count);
        set_limb_bits(
            &mut cols.observation_count_before_bits,
            &cols.observation_count_before,
        );
        set_limb_bits(
            &mut cols.observation_count_after_bits,
            &cols.observation_count_after,
        );
        set_add_carries(
            &mut cols.observation_count_carries,
            &cols.observation_count_before,
            u64::from(observation_delta),
        );
        observation_count = next_observation_count;

        cols.history_hash_before = history_state;
        let mut step = 0usize;
        if is_first && context.is_genesis {
            for observation in history_header_observations(context.total_batch_count) {
                absorb_observation(
                    observation,
                    &mut history_state,
                    &mut cols.hash_outputs,
                    &mut step,
                    &mut compression_inputs,
                );
            }
        } else {
            for _ in 0..6 {
                cols.hash_outputs[step] = history_state;
                step += 1;
            }
        }
        for observation in manifest_observations(record) {
            absorb_observation(
                observation,
                &mut history_state,
                &mut cols.hash_outputs,
                &mut step,
                &mut compression_inputs,
            );
        }
        if is_last && context.is_terminal {
            absorb_observation(
                ObservationV2::Field(F::from_u32(HISTORY_END_TAG_V2)),
                &mut history_state,
                &mut cols.hash_outputs,
                &mut step,
                &mut compression_inputs,
            );
            let mut end = [F::ZERO; DIGEST_SIZE];
            end[0] = F::from_u32(STATEMENT_DIGEST_END_TAG_V2);
            end[1] = F::from_u16(observation_count as u16);
            end[2] = F::from_u16((observation_count >> 16) as u16);
            history_state = compress_recorded(history_state, end, &mut compression_inputs);
            cols.hash_outputs[step] = history_state;
            step += 1;
        } else {
            for _ in 0..2 {
                cols.hash_outputs[step] = history_state;
                step += 1;
            }
        }
        debug_assert_eq!(step, HISTORY_ROW_HASH_STEP_COUNT_V2);
        cols.history_hash_after = history_state;
        cols.endpoint_batch_count = u32_limbs(context.total_batch_count);
        set_limb_bits(
            &mut cols.endpoint_batch_count_bits,
            &cols.endpoint_batch_count,
        );
        cols.endpoint_chunk_index = u32_limbs(context.chunk_index);
        set_limb_bits(
            &mut cols.endpoint_chunk_index_bits,
            &cols.endpoint_chunk_index,
        );
        cols.endpoint_batch_start = u32_limbs(context.start_batch_index);
        cols.endpoint_batch_end = u32_limbs(end_batch_index);
        set_limb_bits(&mut cols.endpoint_batch_end_bits, &cols.endpoint_batch_end);
        cols.endpoint_active_total_start = u64_limbs(context.active_child_count_before);
        cols.endpoint_observation_count_start = u32_limbs(context.observation_count_before);
        cols.endpoint_history_hash_start = context.history_hash_before;
        cols.endpoint_is_genesis = F::from_bool(context.is_genesis);
        cols.endpoint_is_terminal = F::from_bool(context.is_terminal);
        cols.endpoint_protocol_digest = first.protocol_digest;
        cols.endpoint_relation_digest = first.relation_digest;
        cols.endpoint_program_commitment = first.program_commitment;
        cols.endpoint_initial_pc = first_child.input.pc;
        cols.endpoint_initial_memory_root = first_child.input.memory_root;
        cols.endpoint_genesis_accumulator_digest = first.prior_accumulator_digest;
    }

    let public_values = VerifierWarpHistoryChunkPublicValuesV3 {
        chunk_protocol_version: VERIFIER_WARP_HISTORY_CHUNK_PROTOCOL_V3,
        source_history_protocol_version: VERIFIER_WARP_HISTORY_PROTOCOL_V2,
        source_child_capacity: VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2 as u32,
        vacc_input_arity: VERIFIER_WARP_VACC_INPUT_ARITY_V2 as u32,
        protocol_digest: first.protocol_digest,
        relation_digest: first.relation_digest,
        program_commitment: first.program_commitment,
        total_batch_count: context.total_batch_count,
        chunk_index: context.chunk_index,
        start_batch_index: context.start_batch_index,
        end_batch_index,
        active_child_count_before: context.active_child_count_before,
        active_child_count_after: active_total,
        initial_state: first_child.input,
        final_state: last_child.output,
        initial_accumulator_digest: first.prior_accumulator_digest,
        final_accumulator_digest: last.output_accumulator_digest,
        history_hash_before: context.history_hash_before,
        history_hash_after: history_state,
        observation_count_before: context.observation_count_before,
        observation_count_after: observation_count,
        is_genesis: context.is_genesis,
        is_terminal: context.is_terminal,
        public_values_digest: last.public_values_digest,
    };
    Ok(VerifierWarpHistoryChunkTraceV3 {
        matrix: RowMajorMatrix::new(values, width),
        public_values,
        compression_inputs,
    })
}

fn validate_records(
    records: &[VerifierWarpHistoryTransitionRecordV2],
    context: &VerifierWarpHistoryChunkContextV3,
) -> Result<(), VerifierWarpHistoryTraceErrorV2> {
    let first = records
        .first()
        .ok_or(VerifierWarpHistoryTraceErrorV2::Empty)?;
    let chunk_len =
        u32::try_from(records.len()).map_err(|_| VerifierWarpHistoryTraceErrorV2::Overflow)?;
    let end = context
        .start_batch_index
        .checked_add(chunk_len)
        .ok_or(VerifierWarpHistoryTraceErrorV2::Overflow)?;
    if context.total_batch_count == 0
        || end > context.total_batch_count
        || context.is_genesis
            && (context.chunk_index != 0
                || context.start_batch_index != 0
                || context.active_child_count_before != 0
                || context.observation_count_before != 0
                || context.history_hash_before != verifier_warp_history_initial_hash_v2())
        || !context.is_genesis && context.start_batch_index == 0
        || context.is_terminal && end != context.total_batch_count
        || !context.is_terminal && end == context.total_batch_count
    {
        return Err(VerifierWarpHistoryTraceErrorV2::ChunkContext);
    }
    for (index, record) in records.iter().enumerate() {
        if !record.has_fixed_protocol_dimensions() {
            return Err(VerifierWarpHistoryTraceErrorV2::Protocol(index));
        }
        let expected_batch = context
            .start_batch_index
            .checked_add(
                u32::try_from(index).map_err(|_| VerifierWarpHistoryTraceErrorV2::Overflow)?,
            )
            .ok_or(VerifierWarpHistoryTraceErrorV2::Overflow)?;
        if record.batch_index != expected_batch {
            return Err(VerifierWarpHistoryTraceErrorV2::BatchIndex(index));
        }
        let count = usize::from(record.active_child_count);
        if !(1..=VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2).contains(&count)
            || !record.children[0].occupied
            || record.children[1..]
                .iter()
                .any(|child| *child != super::VerifierWarpChildRecordV2::padding())
        {
            return Err(VerifierWarpHistoryTraceErrorV2::ActivePrefix(index));
        }
        if record.protocol_digest != first.protocol_digest
            || record.relation_digest != first.relation_digest
            || record.program_commitment != first.program_commitment
        {
            return Err(VerifierWarpHistoryTraceErrorV2::FixedBinding(index));
        }
        let aggregate = record.children[0];
        if (aggregate.terminates && aggregate.exit_code != F::ZERO)
            || (!aggregate.terminates
                && aggregate.exit_code != F::from_u32(DEFAULT_SUSPEND_EXIT_CODE))
        {
            return Err(VerifierWarpHistoryTraceErrorV2::ChildBoundary(index));
        }
        let is_chunk_last = index + 1 == records.len();
        let is_final = is_chunk_last && context.is_terminal;
        let last_child = aggregate;
        if !is_final {
            if count != VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2
                || last_child.terminates
                || record.public_values_digest != [F::ZERO; DIGEST_SIZE]
            {
                return Err(VerifierWarpHistoryTraceErrorV2::PrematureFinal(index));
            }
        } else if !last_child.terminates {
            return Err(VerifierWarpHistoryTraceErrorV2::MissingTermination);
        }
        if index != 0 {
            let previous = &records[index - 1];
            let previous_last = previous.children[0];
            if previous_last.output != record.children[0].input
                || previous.output_accumulator_digest != record.prior_accumulator_digest
            {
                return Err(VerifierWarpHistoryTraceErrorV2::Continuity(index));
            }
        }
    }
    Ok(())
}

fn history_header_observations(batch_count: u32) -> [ObservationV2; 6] {
    [
        ObservationV2::Field(F::from_u32(HISTORY_START_TAG_V2)),
        ObservationV2::Field(F::from_u32(VERIFIER_WARP_HISTORY_PROTOCOL_V2)),
        ObservationV2::Field(F::from_usize(VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2)),
        ObservationV2::Field(F::from_usize(VERIFIER_WARP_VACC_INPUT_ARITY_V2)),
        ObservationV2::Field(F::from_u16(batch_count as u16)),
        ObservationV2::Field(F::from_u16((batch_count >> 16) as u16)),
    ]
}

fn manifest_observations(record: &VerifierWarpHistoryTransitionRecordV2) -> Vec<ObservationV2> {
    let mut observations = Vec::with_capacity(MANIFEST_OBSERVATION_COUNT_V2);
    macro_rules! field {
        ($value:expr) => {
            observations.push(ObservationV2::Field($value))
        };
    }
    field!(F::from_u32(MANIFEST_START_TAG_V2));
    field!(F::from_u32(MANIFEST_HEADER_TAG_V2));
    field!(F::from_u32(record.protocol_version));
    field!(F::from_usize(VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2));
    field!(F::from_u8(record.vacc_input_arity));
    observations.push(ObservationV2::Digest(record.protocol_digest));
    observations.push(ObservationV2::Digest(record.relation_digest));
    field!(F::from_u16(record.batch_index as u16));
    field!(F::from_u16((record.batch_index >> 16) as u16));
    field!(F::from_u8(record.active_child_count));
    observations.push(ObservationV2::Digest(record.program_commitment));
    field!(F::from_u32(MANIFEST_CHILDREN_TAG_V2));
    for (slot, child) in record.children.iter().enumerate() {
        field!(F::from_u32(MANIFEST_CHILD_TAG_V2));
        field!(F::from_usize(slot));
        field!(F::from_bool(child.occupied));
        field!(child.input.pc);
        observations.push(ObservationV2::Digest(child.input.memory_root));
        field!(child.output.pc);
        observations.push(ObservationV2::Digest(child.output.memory_root));
        field!(child.exit_code);
        field!(F::from_bool(child.terminates));
    }
    field!(F::from_u32(MANIFEST_ACCUMULATORS_TAG_V2));
    observations.push(ObservationV2::Digest(record.prior_accumulator_digest));
    observations.push(ObservationV2::Digest(record.source_accumulator_digest));
    observations.push(ObservationV2::Digest(record.output_accumulator_digest));
    field!(F::from_u32(MANIFEST_SOURCE_AUTH_TAG_V2));
    observations.push(ObservationV2::Digest(record.source_commitment_root));
    observations.push(ObservationV2::Digest(record.external_logup_gkr_digest));
    observations.push(ObservationV2::Digest(record.source_functional_digest));
    observations.push(ObservationV2::Digest(record.setup_openings_digest));
    field!(F::from_u32(MANIFEST_SOURCE_TAG_V2));
    observations.push(ObservationV2::Digest(record.source_statement_digest));
    observations.push(ObservationV2::Digest(record.transition_transcript_digest));
    field!(F::from_u32(MANIFEST_PUBLIC_VALUES_TAG_V2));
    observations.push(ObservationV2::Digest(record.public_values_digest));
    field!(F::from_u32(MANIFEST_END_TAG_V2));
    debug_assert_eq!(observations.len(), MANIFEST_OBSERVATION_COUNT_V2);
    observations
}

fn absorb_observation(
    observation: ObservationV2,
    state: &mut Digest,
    outputs: &mut [[F; DIGEST_SIZE]; HISTORY_ROW_HASH_STEP_COUNT_V2],
    step: &mut usize,
    inputs: &mut Vec<[F; 2 * DIGEST_SIZE]>,
) {
    match observation {
        ObservationV2::Field(value) => {
            let mut block = [F::ZERO; DIGEST_SIZE];
            block[0] = F::from_u32(STATEMENT_DIGEST_FIELD_TAG_V2);
            block[1] = value;
            *state = compress_recorded(*state, block, inputs);
            outputs[*step] = *state;
            *step += 1;
        }
        ObservationV2::Digest(digest) => {
            let mut tag = [F::ZERO; DIGEST_SIZE];
            tag[0] = F::from_u32(STATEMENT_DIGEST_COMMITMENT_TAG_V2);
            *state = compress_recorded(*state, tag, inputs);
            outputs[*step] = *state;
            *step += 1;
            *state = compress_recorded(*state, digest, inputs);
            outputs[*step] = *state;
            *step += 1;
        }
    }
}

fn compress_recorded(
    left: Digest,
    right: Digest,
    inputs: &mut Vec<[F; 2 * DIGEST_SIZE]>,
) -> Digest {
    inputs.push(core::array::from_fn(|index| {
        if index < DIGEST_SIZE {
            left[index]
        } else {
            right[index - DIGEST_SIZE]
        }
    }));
    poseidon2_compress_with_capacity(left, right).0
}

fn u32_limbs(value: u32) -> [F; 2] {
    [F::from_u16(value as u16), F::from_u16((value >> 16) as u16)]
}

fn u64_limbs(value: u64) -> [F; 4] {
    core::array::from_fn(|limb| F::from_u16((value >> (16 * limb)) as u16))
}

fn set_limb_bits<const N: usize>(bits: &mut [[F; 16]; N], limbs: &[F; N]) {
    for (limb_bits, limb) in bits.iter_mut().zip(limbs) {
        let value = limb.as_canonical_u32();
        for (bit, bit_cell) in limb_bits.iter_mut().enumerate() {
            *bit_cell = F::from_bool(((value >> bit) & 1) == 1);
        }
    }
}

fn set_add_carries<const N: usize>(carries: &mut [F; N], before: &[F; N], addend: u64) {
    let mut carry = addend;
    for (limb, carry_cell) in before.iter().zip(carries) {
        let total = u64::from(limb.as_canonical_u32()) + carry;
        carry = total >> 16;
        *carry_cell = F::from_bool(carry != 0);
    }
}

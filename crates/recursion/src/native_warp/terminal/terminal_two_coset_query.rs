//! Auxiliary lookup generation for coefficient-two-coset terminal queries.
//!
//! The initial
//! [`CoefficientTwoCosetGrs`](openvm_stark_backend::warp_accum::WhirInitialRsLayout::CoefficientTwoCosetGrs)
//! oracle is a scalar codeword over two interleaved cosets. A physical query
//! index `2t + b` therefore has pre-fold root `omega^t * g^b`; later WHIR
//! rounds use an ordinary scalar-RS root. The terminal query AIR emits one
//! `RightShift` lookup per query and either one or two `ExpBitsLen` lookups.
//! This module constructs the exact matching request stream for the shared
//! auxiliary owner.

use openvm_stark_backend::{
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32, TwoAdicField},
    transcript::TranscriptLog,
    warp_accum::{
        terminal_whir::{terminal_whir_query_root, TerminalWhirLayout},
        TerminalWhirVerification,
    },
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};

use crate::primitives::exp_bits_len::ExpBitsLenCpuTraceGenerator;

/// The descriptor-bound production folding arity: sixteen scalar leaves.
pub const COEFFICIENT_TWO_COSET_QUERY_K: usize = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoefficientTwoCosetQueryAuxError {
    UnsupportedProfile,
    VerificationShape,
    Transcript,
    QueryIndex,
    QueryRoot,
    ScalarLeafGeometry,
}

impl core::fmt::Display for CoefficientTwoCosetQueryAuxError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "coefficient-two-coset query auxiliaries: {self:?}"
        )
    }
}

impl std::error::Error for CoefficientTwoCosetQueryAuxError {}

/// The semantic exponentiation lookup emitted by one query row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoefficientTwoCosetExpBitsLenRequest {
    pub base: F,
    pub bit_src: F,
    pub num_bits: usize,
    pub result: F,
}

/// The semantic right-shift lookup emitted by one query row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoefficientTwoCosetRightShiftRequest {
    pub input: F,
    pub shift_bits: usize,
    pub result: F,
}

/// Why an auxiliary record exists. Keeping this tag in the host record makes
/// layout mistakes visible in differential tests and owner-level diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoefficientTwoCosetQueryAuxKind {
    /// `1^0 = 1`, paired with the initial query-index right shift.
    InitialShiftIdentity,
    /// `omega^(index >> 1)`, before selecting the second coset with `g`.
    InitialSubgroupRoot,
    /// The ordinary scalar-RS root and query-index shift used after round zero.
    OrdinaryRootAndShift,
}

/// One record accepted by the shared `ExpBitsLenAir`. A right-shift request is
/// attached only when its input equals the exponentiation record's `bit_src`,
/// which is the physical interface implemented by `ExpBitsLenCpuTraceGenerator`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoefficientTwoCosetQueryAuxRecord {
    pub round: u32,
    pub query: u32,
    pub merkle_index: u32,
    pub kind: CoefficientTwoCosetQueryAuxKind,
    pub exponentiation: CoefficientTwoCosetExpBitsLenRequest,
    pub right_shift: Option<CoefficientTwoCosetRightShiftRequest>,
}

/// Exact requests generated in query order and ready to merge into the one
/// shared `ExpBitsLenAir` trace.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CoefficientTwoCosetQueryAuxRequests {
    records: Vec<CoefficientTwoCosetQueryAuxRecord>,
}

impl CoefficientTwoCosetQueryAuxRequests {
    #[must_use]
    pub fn records(&self) -> &[CoefficientTwoCosetQueryAuxRecord] {
        &self.records
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Append these requests to an existing owner without regenerating or
    /// reordering any other module's requests.
    pub fn populate(&self, generator: &ExpBitsLenCpuTraceGenerator) {
        generator.add_requests_with_shift(self.records.iter().map(|record| {
            let (shift_bits, shift_mult) = match record.right_shift {
                Some(shift) => (shift.shift_bits, 1),
                None => (0, 0),
            };
            (
                record.exponentiation.base,
                record.exponentiation.bit_src,
                record.exponentiation.num_bits,
                shift_bits,
                shift_mult,
            )
        }));
    }
}

fn low_bits(value: u32, bits: usize) -> Result<u32, CoefficientTwoCosetQueryAuxError> {
    if bits >= u32::BITS as usize {
        return Err(CoefficientTwoCosetQueryAuxError::UnsupportedProfile);
    }
    if bits == 0 {
        return Ok(0);
    }
    Ok(value & ((1u32 << bits) - 1))
}

fn checked_query_sample(
    transcript: &TranscriptLog<F, [F; 16]>,
    tidx: usize,
    query_bits: usize,
    expected_index: u32,
) -> Result<F, CoefficientTwoCosetQueryAuxError> {
    if query_bits == 0 || query_bits >= u32::BITS as usize {
        return Err(CoefficientTwoCosetQueryAuxError::UnsupportedProfile);
    }
    let sample = *transcript
        .values()
        .get(tidx)
        .ok_or(CoefficientTwoCosetQueryAuxError::Transcript)?;
    if transcript.samples().get(tidx) != Some(&true) {
        return Err(CoefficientTwoCosetQueryAuxError::Transcript);
    }
    if low_bits(sample.as_canonical_u32(), query_bits)? != expected_index {
        return Err(CoefficientTwoCosetQueryAuxError::QueryIndex);
    }
    Ok(sample)
}

fn validate_scalar_leaf_geometry(
    opened_rows: &[Vec<EF>],
) -> Result<(), CoefficientTwoCosetQueryAuxError> {
    let expected_rows = 1usize << COEFFICIENT_TWO_COSET_QUERY_K;
    if opened_rows.len() != expected_rows || opened_rows.iter().any(|row| row.len() != 1) {
        return Err(CoefficientTwoCosetQueryAuxError::ScalarLeafGeometry);
    }
    Ok(())
}

fn push_record(
    records: &mut Vec<CoefficientTwoCosetQueryAuxRecord>,
    round: u32,
    query: usize,
    merkle_index: u32,
    kind: CoefficientTwoCosetQueryAuxKind,
    exponentiation: CoefficientTwoCosetExpBitsLenRequest,
    right_shift: Option<CoefficientTwoCosetRightShiftRequest>,
) -> Result<(), CoefficientTwoCosetQueryAuxError> {
    if right_shift
        .as_ref()
        .is_some_and(|shift| shift.input != exponentiation.bit_src)
    {
        return Err(CoefficientTwoCosetQueryAuxError::VerificationShape);
    }
    let exponent = low_bits(
        exponentiation.bit_src.as_canonical_u32(),
        exponentiation.num_bits,
    )?;
    if exponentiation.base.exp_u64(exponent as u64) != exponentiation.result {
        return Err(CoefficientTwoCosetQueryAuxError::QueryRoot);
    }
    records.push(CoefficientTwoCosetQueryAuxRecord {
        round,
        query: u32::try_from(query)
            .map_err(|_| CoefficientTwoCosetQueryAuxError::VerificationShape)?,
        merkle_index,
        kind,
        exponentiation,
        right_shift,
    });
    Ok(())
}

/// Derive all query-owned `ExpBitsLen` and `RightShift` requests from a native
/// terminal verification record.
///
/// The output order exactly matches the query AIR:
///
/// - round zero: shift identity, then initial subgroup root;
/// - later rounds: ordinary root and shift in one request.
///
/// `verification` must already come from native terminal verification, but this
/// function still validates every shape and root it consumes. It therefore
/// fails closed if a caller accidentally supplies vector leaves, stale query
/// roots, non-sample transcript cells, or indices outside the sampled domain.
pub fn generate_coefficient_two_coset_query_aux_requests(
    verification: &TerminalWhirVerification<F, EF, Digest>,
    transcript: &TranscriptLog<F, [F; 16]>,
    initial_log_domain_size: usize,
) -> Result<CoefficientTwoCosetQueryAuxRequests, CoefficientTwoCosetQueryAuxError> {
    if verification.rounds.is_empty()
        || initial_log_domain_size <= COEFFICIENT_TWO_COSET_QUERY_K
        || initial_log_domain_size - 1 > <F as TwoAdicField>::TWO_ADICITY
    {
        return Err(CoefficientTwoCosetQueryAuxError::UnsupportedProfile);
    }

    let mut records = Vec::new();
    for (round_index, round) in verification.rounds.iter().enumerate() {
        let log_rs_domain_size = initial_log_domain_size
            .checked_sub(round_index)
            .ok_or(CoefficientTwoCosetQueryAuxError::UnsupportedProfile)?;
        let query_bits = log_rs_domain_size
            .checked_sub(COEFFICIENT_TWO_COSET_QUERY_K)
            .ok_or(CoefficientTwoCosetQueryAuxError::UnsupportedProfile)?;
        if query_bits == 0
            || query_bits >= u32::BITS as usize
            || round.round as usize != round_index
            || round.log_rs_domain_size as usize != log_rs_domain_size
            || round.alphas.len() != COEFFICIENT_TWO_COSET_QUERY_K
            || round.query_indices.is_empty()
            || round.query_indices.len() != round.query_roots.len()
            || round.query_indices.len() != round.folded_values.len()
            || round.query_indices.len() != round.opened_rows.len()
            || round.query_indices.len() != round.query_digests.len()
        {
            return Err(CoefficientTwoCosetQueryAuxError::VerificationShape);
        }
        if round_index != 0 && log_rs_domain_size > <F as TwoAdicField>::TWO_ADICITY {
            return Err(CoefficientTwoCosetQueryAuxError::UnsupportedProfile);
        }

        let query_tidx = round
            .transcript_span
            .end
            .operations
            .checked_sub(D_EF + round.query_indices.len())
            .ok_or(CoefficientTwoCosetQueryAuxError::Transcript)?;
        if query_tidx < round.transcript_span.start.operations {
            return Err(CoefficientTwoCosetQueryAuxError::Transcript);
        }
        let query_domain_size = 1u32
            .checked_shl(query_bits as u32)
            .ok_or(CoefficientTwoCosetQueryAuxError::UnsupportedProfile)?;

        for (query, (((&merkle_index, &recorded_root), opened_rows), _query_digest)) in round
            .query_indices
            .iter()
            .zip(&round.query_roots)
            .zip(&round.opened_rows)
            .zip(&round.query_digests)
            .enumerate()
        {
            if merkle_index >= query_domain_size {
                return Err(CoefficientTwoCosetQueryAuxError::QueryIndex);
            }
            validate_scalar_leaf_geometry(opened_rows)?;
            let tidx = query_tidx
                .checked_add(query)
                .ok_or(CoefficientTwoCosetQueryAuxError::Transcript)?;
            let sample = checked_query_sample(transcript, tidx, query_bits, merkle_index)?;
            let raw_root = terminal_whir_query_root::<F>(
                TerminalWhirLayout::ScalarCoefficientTwoCoset,
                round_index == 0,
                merkle_index as usize,
                log_rs_domain_size,
                COEFFICIENT_TWO_COSET_QUERY_K,
            )
            .map_err(|_| CoefficientTwoCosetQueryAuxError::UnsupportedProfile)?;
            let mut post_fold_root = raw_root;
            for _ in 0..COEFFICIENT_TWO_COSET_QUERY_K {
                post_fold_root *= post_fold_root;
            }
            if post_fold_root != recorded_root {
                return Err(CoefficientTwoCosetQueryAuxError::QueryRoot);
            }

            let shift = CoefficientTwoCosetRightShiftRequest {
                input: sample,
                shift_bits: query_bits,
                result: F::from_u32(sample.as_canonical_u32() >> query_bits),
            };
            if round_index == 0 {
                push_record(
                    &mut records,
                    round.round,
                    query,
                    merkle_index,
                    CoefficientTwoCosetQueryAuxKind::InitialShiftIdentity,
                    CoefficientTwoCosetExpBitsLenRequest {
                        base: F::ONE,
                        bit_src: sample,
                        num_bits: 0,
                        result: F::ONE,
                    },
                    Some(shift),
                )?;

                let subgroup_index = merkle_index >> 1;
                let subgroup_bits = query_bits
                    .checked_sub(1)
                    .ok_or(CoefficientTwoCosetQueryAuxError::UnsupportedProfile)?;
                let root_omega = F::two_adic_generator(log_rs_domain_size - 1);
                let subgroup_root = root_omega.exp_u64(subgroup_index as u64);
                let expected_raw_root = if merkle_index & 1 == 0 {
                    subgroup_root
                } else {
                    subgroup_root * F::GENERATOR
                };
                if expected_raw_root != raw_root {
                    return Err(CoefficientTwoCosetQueryAuxError::QueryRoot);
                }
                push_record(
                    &mut records,
                    round.round,
                    query,
                    merkle_index,
                    CoefficientTwoCosetQueryAuxKind::InitialSubgroupRoot,
                    CoefficientTwoCosetExpBitsLenRequest {
                        base: root_omega,
                        bit_src: F::from_u32(subgroup_index),
                        num_bits: subgroup_bits,
                        result: subgroup_root,
                    },
                    None,
                )?;
            } else {
                let root_omega = F::two_adic_generator(log_rs_domain_size);
                push_record(
                    &mut records,
                    round.round,
                    query,
                    merkle_index,
                    CoefficientTwoCosetQueryAuxKind::OrdinaryRootAndShift,
                    CoefficientTwoCosetExpBitsLenRequest {
                        base: root_omega,
                        bit_src: sample,
                        num_bits: query_bits,
                        result: raw_root,
                    },
                    Some(shift),
                )?;
            }
        }
    }

    Ok(CoefficientTwoCosetQueryAuxRequests { records })
}

/// Validate, derive, and append the exact query requests to a shared auxiliary
/// trace owner. The returned count is useful for owner-level height planning.
pub fn populate_coefficient_two_coset_query_aux_requests(
    verification: &TerminalWhirVerification<F, EF, Digest>,
    transcript: &TranscriptLog<F, [F; 16]>,
    initial_log_domain_size: usize,
    generator: &ExpBitsLenCpuTraceGenerator,
) -> Result<usize, CoefficientTwoCosetQueryAuxError> {
    let requests = generate_coefficient_two_coset_query_aux_requests(
        verification,
        transcript,
        initial_log_domain_size,
    )?;
    let count = requests.len();
    requests.populate(generator);
    Ok(count)
}

#[cfg(test)]
mod tests {
    use core::borrow::Borrow;
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use openvm_stark_backend::{
        p3_field::PrimeCharacteristicRing,
        warp_accum::{
            BinaryMerkleMultiproofRecord, RsAdjointEvalVerification, TerminalWhirRoundVerification,
            TerminalWhirTranscriptPhase, TerminalWhirTranscriptPhaseSpan,
        },
        TranscriptCheckpoint,
    };
    use p3_matrix::Matrix;

    use super::*;
    use crate::primitives::exp_bits_len::{ExpBitsLenCols, ExpBitsLenCpuTraceGenerator};

    const INITIAL_LOG_DOMAIN_SIZE: usize = 9;

    fn digest(seed: u32) -> Digest {
        core::array::from_fn(|index| F::from_u32(seed + index as u32))
    }

    fn span(
        round: u32,
        start_operations: usize,
        end_operations: usize,
    ) -> TerminalWhirTranscriptPhaseSpan {
        TerminalWhirTranscriptPhaseSpan {
            phase: TerminalWhirTranscriptPhase::QueryPhase { round },
            start: TranscriptCheckpoint {
                operations: start_operations,
                ..Default::default()
            },
            end: TranscriptCheckpoint {
                operations: end_operations,
                ..Default::default()
            },
        }
    }

    fn scalar_opening(seed: u32) -> Vec<Vec<EF>> {
        (0..1usize << COEFFICIENT_TWO_COSET_QUERY_K)
            .map(|row| vec![EF::from(F::from_u32(seed + row as u32 + 1))])
            .collect()
    }

    fn round(
        round_index: usize,
        query_indices: Vec<u32>,
        query_tidx: usize,
    ) -> TerminalWhirRoundVerification<F, EF, Digest> {
        let log_rs_domain_size = INITIAL_LOG_DOMAIN_SIZE - round_index;
        let query_roots = query_indices
            .iter()
            .map(|&index| {
                let mut root = terminal_whir_query_root::<F>(
                    TerminalWhirLayout::ScalarCoefficientTwoCoset,
                    round_index == 0,
                    index as usize,
                    log_rs_domain_size,
                    COEFFICIENT_TWO_COSET_QUERY_K,
                )
                .unwrap();
                for _ in 0..COEFFICIENT_TWO_COSET_QUERY_K {
                    root *= root;
                }
                root
            })
            .collect::<Vec<_>>();
        let query_count = query_indices.len();
        let root = digest(50 + round_index as u32);
        TerminalWhirRoundVerification {
            round: round_index as u32,
            transcript_span: span(
                round_index as u32,
                query_tidx,
                query_tidx + query_count + D_EF,
            ),
            commitment: root,
            log_rs_domain_size: log_rs_domain_size as u32,
            alphas: vec![EF::ONE; COEFFICIENT_TWO_COSET_QUERY_K],
            sumcheck_rounds: Vec::new(),
            query_indices,
            query_roots,
            folded_values: vec![EF::ZERO; query_count],
            opened_rows: (0..query_count)
                .map(|query| scalar_opening(100 * round_index as u32 + 20 * query as u32))
                .collect(),
            query_digests: (0..query_count)
                .map(|query| digest(100 + query as u32))
                .collect(),
            multiproof: BinaryMerkleMultiproofRecord {
                expected_root: root,
                depth: (log_rs_domain_size - COEFFICIENT_TWO_COSET_QUERY_K) as u32,
                leaf_indices: Vec::new(),
                leaf_digests: Vec::new(),
                compressions: Vec::new(),
                consumed_siblings: 0,
            },
            ood_point: None,
            ood_value: None,
            gamma: EF::ONE,
            pre_round_claim: EF::ZERO,
            post_round_claim: EF::ZERO,
        }
    }

    fn fixture() -> (
        TerminalWhirVerification<F, EF, Digest>,
        TranscriptLog<F, [F; 16]>,
    ) {
        // Initial indices 2 and 3 share `t = 1` and exercise both cosets.
        let initial_indices = vec![2, 3];
        let later_indices = vec![1, 4];
        let round0_tidx = 0;
        let round1_tidx = initial_indices.len() + D_EF;
        let rounds = vec![
            round(0, initial_indices.clone(), round0_tidx),
            round(1, later_indices.clone(), round1_tidx),
        ];
        let transcript_len = round1_tidx + later_indices.len() + D_EF;
        let mut values = vec![F::ZERO; transcript_len];
        let mut samples = vec![false; transcript_len];
        let initial_bits = INITIAL_LOG_DOMAIN_SIZE - COEFFICIENT_TWO_COSET_QUERY_K;
        let later_bits = initial_bits - 1;
        for (offset, index) in initial_indices.into_iter().enumerate() {
            values[round0_tidx + offset] = F::from_u32(index + (5 << initial_bits));
            samples[round0_tidx + offset] = true;
        }
        for (offset, index) in later_indices.into_iter().enumerate() {
            values[round1_tidx + offset] = F::from_u32(index + (7 << later_bits));
            samples[round1_tidx + offset] = true;
        }
        let adjoint = RsAdjointEvalVerification {
            transcript_start: TranscriptCheckpoint::default(),
            transcript_end: TranscriptCheckpoint::default(),
            claimed_value: EF::ZERO,
            degree: 0,
            rounds: Vec::new(),
            point: Vec::new(),
            final_claim: EF::ZERO,
            expected_final: EF::ZERO,
        };
        let empty_span = TerminalWhirTranscriptPhaseSpan {
            phase: TerminalWhirTranscriptPhase::Descriptor,
            start: TranscriptCheckpoint::default(),
            end: TranscriptCheckpoint::default(),
        };
        (
            TerminalWhirVerification {
                transcript_start: TranscriptCheckpoint::default(),
                descriptor_span: empty_span.clone(),
                batching_challenge_span: TerminalWhirTranscriptPhaseSpan {
                    phase: TerminalWhirTranscriptPhase::BatchingChallenge,
                    ..empty_span.clone()
                },
                transcript_end: TranscriptCheckpoint {
                    operations: transcript_len,
                    ..Default::default()
                },
                root: digest(1),
                batching_challenge: EF::ONE,
                initial_claim: EF::ZERO,
                rounds,
                final_poly: vec![EF::ZERO],
                final_weight_evals: vec![EF::ZERO],
                final_weight_span: TerminalWhirTranscriptPhaseSpan {
                    phase: TerminalWhirTranscriptPhase::FinalWeight,
                    ..empty_span
                },
                suffix_point: Vec::new(),
                accumulator_adjoint: adjoint,
                expected_weight: EF::ZERO,
                actual_weight: EF::ZERO,
                final_inner_product: EF::ZERO,
                final_claim: EF::ZERO,
            },
            TranscriptLog::new(values, samples),
        )
    }

    #[test]
    fn exact_requests_match_native_roots_for_both_cosets_and_later_rounds() {
        let (verification, transcript) = fixture();
        let requests = generate_coefficient_two_coset_query_aux_requests(
            &verification,
            &transcript,
            INITIAL_LOG_DOMAIN_SIZE,
        )
        .unwrap();
        assert_eq!(requests.len(), 6);

        let records = requests.records();
        for query in 0..2 {
            let shift = records[2 * query];
            let subgroup = records[2 * query + 1];
            assert_eq!(
                shift.kind,
                CoefficientTwoCosetQueryAuxKind::InitialShiftIdentity
            );
            assert_eq!(shift.exponentiation.result, F::ONE);
            assert_eq!(shift.right_shift.unwrap().shift_bits, 5);
            assert_eq!(shift.right_shift.unwrap().result, F::from_u32(5));

            let index = verification.rounds[0].query_indices[query];
            let omega = F::two_adic_generator(INITIAL_LOG_DOMAIN_SIZE - 1);
            let subgroup_root = omega.exp_u64((index >> 1) as u64);
            assert_eq!(
                subgroup.kind,
                CoefficientTwoCosetQueryAuxKind::InitialSubgroupRoot
            );
            assert_eq!(subgroup.exponentiation.bit_src, F::from_u32(index >> 1));
            assert_eq!(subgroup.exponentiation.num_bits, 4);
            assert_eq!(subgroup.exponentiation.result, subgroup_root);
            assert!(subgroup.right_shift.is_none());

            let expected_raw = if index & 1 == 0 {
                subgroup_root
            } else {
                subgroup_root * F::GENERATOR
            };
            let native = terminal_whir_query_root::<F>(
                TerminalWhirLayout::ScalarCoefficientTwoCoset,
                true,
                index as usize,
                INITIAL_LOG_DOMAIN_SIZE,
                COEFFICIENT_TWO_COSET_QUERY_K,
            )
            .unwrap();
            assert_eq!(expected_raw, native);
        }

        for (record, &index) in records[4..]
            .iter()
            .zip(&verification.rounds[1].query_indices)
        {
            assert_eq!(
                record.kind,
                CoefficientTwoCosetQueryAuxKind::OrdinaryRootAndShift
            );
            assert_eq!(record.exponentiation.num_bits, 4);
            assert_eq!(
                low_bits(record.exponentiation.bit_src.as_canonical_u32(), 4).unwrap(),
                index
            );
            assert_eq!(record.right_shift.unwrap().result, F::from_u32(7));
            let native = terminal_whir_query_root::<F>(
                TerminalWhirLayout::ScalarCoefficientTwoCoset,
                false,
                index as usize,
                INITIAL_LOG_DOMAIN_SIZE - 1,
                COEFFICIENT_TWO_COSET_QUERY_K,
            )
            .unwrap();
            assert_eq!(record.exponentiation.result, native);
        }
    }

    #[test]
    fn populated_exp_trace_emits_exact_exponents_and_right_shifts() {
        let (verification, transcript) = fixture();
        let requests = generate_coefficient_two_coset_query_aux_requests(
            &verification,
            &transcript,
            INITIAL_LOG_DOMAIN_SIZE,
        )
        .unwrap();
        let generator = ExpBitsLenCpuTraceGenerator::default();
        requests.populate(&generator);
        let row_offsets = generator
            .requests
            .lock()
            .unwrap()
            .iter()
            .map(|record| record.row_offset as usize)
            .collect::<Vec<_>>();
        let trace = generator.generate_trace_row_major(None).unwrap();
        let width = ExpBitsLenCols::<F>::width();

        for (&row_offset, request) in row_offsets.iter().zip(requests.records()) {
            let first_row = trace.row_slice(row_offset).unwrap();
            let first: &ExpBitsLenCols<F> = (*first_row).borrow();
            assert_eq!(first.base, request.exponentiation.base);
            assert_eq!(first.bit_src, request.exponentiation.bit_src);
            assert_eq!(
                first.num_bits,
                F::from_usize(request.exponentiation.num_bits)
            );
            assert_eq!(first.result, request.exponentiation.result);
            if let Some(shift) = request.right_shift {
                let shift_row = trace.row_slice(row_offset + shift.shift_bits).unwrap();
                let shift_row: &ExpBitsLenCols<F> = (*shift_row).borrow();
                assert_eq!(shift_row.bit_src, shift.result);
                assert_eq!(shift_row.shift_mult, F::ONE);
            }
        }
        assert_eq!(trace.width(), width);
    }

    #[test]
    fn vector_or_malformed_scalar_leaf_geometry_is_rejected() {
        let (verification, transcript) = fixture();
        for mutate in [0usize, 1] {
            let mut malformed = verification.clone();
            if mutate == 0 {
                malformed.rounds[0].opened_rows[0] = vec![vec![EF::ZERO; 16]];
            } else {
                malformed.rounds[0].opened_rows[0].pop();
            }
            assert_eq!(
                generate_coefficient_two_coset_query_aux_requests(
                    &malformed,
                    &transcript,
                    INITIAL_LOG_DOMAIN_SIZE,
                ),
                Err(CoefficientTwoCosetQueryAuxError::ScalarLeafGeometry)
            );
        }
    }

    #[test]
    fn malformed_indices_transcript_roots_and_shapes_fail_closed_without_panicking() {
        let (verification, transcript) = fixture();
        let mut cases = Vec::new();

        let mut wrong_index = verification.clone();
        wrong_index.rounds[0].query_indices[0] = 1 << 5;
        cases.push((
            wrong_index,
            transcript.clone(),
            CoefficientTwoCosetQueryAuxError::QueryIndex,
        ));

        let mut wrong_sample = transcript.clone();
        wrong_sample.values_mut()[0] += F::ONE;
        cases.push((
            verification.clone(),
            wrong_sample,
            CoefficientTwoCosetQueryAuxError::QueryIndex,
        ));

        let mut not_a_sample = transcript.clone();
        not_a_sample.samples_mut()[0] = false;
        cases.push((
            verification.clone(),
            not_a_sample,
            CoefficientTwoCosetQueryAuxError::Transcript,
        ));

        let mut wrong_root = verification.clone();
        wrong_root.rounds[0].query_roots[0] += F::ONE;
        cases.push((
            wrong_root,
            transcript.clone(),
            CoefficientTwoCosetQueryAuxError::QueryRoot,
        ));

        let mut wrong_shape = verification.clone();
        wrong_shape.rounds[1].query_digests.pop();
        cases.push((
            wrong_shape,
            transcript,
            CoefficientTwoCosetQueryAuxError::VerificationShape,
        ));

        for (malformed, log, expected) in cases {
            let result = catch_unwind(AssertUnwindSafe(|| {
                generate_coefficient_two_coset_query_aux_requests(
                    &malformed,
                    &log,
                    INITIAL_LOG_DOMAIN_SIZE,
                )
            }));
            assert_eq!(result.unwrap(), Err(expected));
        }
    }
}

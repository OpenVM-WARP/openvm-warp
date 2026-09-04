//! Recursive verifier boundary for the finite-prefix PCS used by finite WARP v3.
//!
//! The SDK crate is above continuations in the dependency graph, so this
//! module cannot name its FinitePrefixLinkProof directly. The witness types
//! below are a field-for-field backend-neutral copy of that wire object. No
//! digest or host-verification bit replaces the concrete proof.
//!
//! This module owns the prefix-specific statement and quadratic range-mask
//! sumcheck. It exports the two exact WHIR opening claims and publishes
//! OrderedManifestPrefixAuthorityBus only after consuming a genuine recursive
//! multi-WHIR completion. The companion query AIR implements the sole WHIR
//! difference: the first oracle is interleaved [H_m, g H_m], while every
//! continuation round is ordinary WHIR.

use core::{
    borrow::{Borrow, BorrowMut},
    ops::Range,
};
use std::sync::Arc;

use openvm_circuit_primitives::{encoder::Encoder, StructReflection, StructReflectionHelper};
use openvm_cpu_backend::CpuBackend;
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_recursion_circuit::{
    bus::{CertifiedTranscriptCheckpointBus, TranscriptBus},
    primitives::{
        bus::{ExpBitsLenBus, ExpBitsLenMessage, RightShiftBus, RightShiftMessage},
        exp_bits_len::ExpBitsLenCpuTraceGenerator,
    },
    system::{AirModule, BusIndexManager, BusInventory},
    utils::{ext_field_add, ext_field_multiply, interpolate_quadratic},
    whir::{
        folding::{FoldRecord, WhirFoldingCols},
        multi_constraint::{
            air::{
                MultiConstraintCallerPrefixMessage, MultiConstraintInitialCommitmentBus,
                MultiConstraintInitialCommitmentMessage, MultiConstraintOpeningMessage,
                MultiConstraintPointMessage, MultiConstraintStatementBuses,
                MultiConstraintSumcheckCols,
            },
            derive_batching_coefficients_preflight, derive_multi_constraint_whir_data,
            module::{
                MultiConstraintWhirAir, MultiConstraintWhirCompletionBus,
                MultiConstraintWhirCompletionMessage, MultiConstraintWhirDirectCarrier,
                MultiConstraintWhirModule, MultiConstraintWhirModuleError,
                PreparedMultiConstraintWhirCarrier,
            },
            run_multi_constraint_whir_preflight,
            trace::{
                generate_multi_constraint_completion_trace, MultiConstraintWhirTerminalCheckpoint,
            },
            MultiConstraintWhirInitialCommitment, MultiConstraintWhirProfile,
            MultiConstraintWhirStatement, MultiConstraintWhirTranscriptPreflight,
        },
    },
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, PermutationCheckBus},
    keygen::types::MultiStarkVerifyingKey,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{
        extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing,
        PrimeField32, TwoAdicField,
    },
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    poly_common::interpolate_quadratic_at_012,
    proof::WhirProof,
    prover::AirProvingContext,
    transcript::TranscriptLog,
    AirRef, BaseAirWithPublicValues, FiatShamirTranscript, PartitionedBaseAir, StarkProtocolConfig,
    SystemParams, TranscriptHistory,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, BabyBearPoseidon2Config, Digest, CHUNK, DIGEST_SIZE, D_EF, EF,
    F,
};

use super::{
    OrderedManifestPrefixAuthorityBus, OrderedManifestPrefixReceiptMessage,
    ORDERED_MANIFEST_PREFIX_SLOTS,
};

pub const FINITE_PREFIX_LINK_PROTOCOL_VERSION: u32 = 4;
pub const FINITE_PREFIX_TWO_COSET_RS_LAYOUT_VERSION: u32 = 1;
pub const FINITE_PREFIX_TWO_COSET_INTERLEAVED_ORDERING: u32 = 1;
pub const FINITE_PREFIX_LINK_ROWS_PER_LEAF: u32 = 16;

const PREFIX_LINK_TRANSCRIPT_TAG: u64 = 0x4650_4c4b_5452_0002;
const PREFIX_LINK_TWO_COSET_RS_TAG: u64 = 0x4650_5253_3243_0001;
const PREFIX_LINK_COEFFICIENT_MESSAGE_TAG: u64 = 0x4650_434f_4546_0001;
const PREFIX_LINK_BASE_PHASE_TAG: u64 = 0x4650_4c4b_4241_5345;
const PREFIX_LINK_FULL_PHASE_TAG: u64 = 0x4650_4c4b_4655_4c4c;
const PREFIX_LINK_SUMCHECK_TAG: u64 = 0x4650_4c4b_5355_4d43;
const PREFIX_LINK_OPENINGS_TAG: u64 = 0x4650_4c4b_4f50_454e;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinitePrefixCircuitContext {
    pub relation_digest: Digest,
    pub index_digest: Digest,
    pub schedule_digest: Digest,
    pub source_order_digest: Digest,
    pub call_index: u32,
    pub source_start: u32,
    pub source_count: u32,
    pub expected_active_child_counts: [u8; ORDERED_MANIFEST_PREFIX_SLOTS],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinitePrefixCircuitLayout {
    pub l_skip: u32,
    pub log_message_len: u32,
    pub log_blowup: u32,
    pub log_codeword_len: u32,
    pub rows_per_leaf: u32,
    pub trace_prefix_len: u64,
    pub active_count_block_start: u64,
    pub active_count_log_height: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinitePrefixCircuitStatement {
    pub protocol_version: u32,
    pub context: FinitePrefixCircuitContext,
    pub layout: FinitePrefixCircuitLayout,
    pub base_root: Digest,
    pub full_root: Digest,
    pub logup_alpha: EF,
    pub logup_beta: EF,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinitePrefixCircuitSumcheckProof {
    pub round_evaluations: Vec<[EF; 3]>,
    pub base_openings: Vec<EF>,
    pub full_openings: Vec<EF>,
}

/// Isomorphic to the SDK proof; the required SDK adapter only copies fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinitePrefixCircuitProof {
    pub statement: FinitePrefixCircuitStatement,
    pub sumcheck: FinitePrefixCircuitSumcheckProof,
    pub whir: WhirProof<BabyBearPoseidon2Config>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FinitePrefixVerifierProfile {
    pub relation_digest: Digest,
    pub index_digest: Digest,
    pub layout: FinitePrefixCircuitLayout,
    pub source_count: usize,
    pub class_index: usize,
    /// Number of WHIR sumcheck rounds in the setup-fixed child key.
    pub whir_sumcheck_rounds: usize,
}

impl FinitePrefixVerifierProfile {
    pub fn validate(&self, params: &SystemParams) -> Result<(), FinitePrefixVerifyError> {
        if self.source_count == 0 || self.source_count > ORDERED_MANIFEST_PREFIX_SLOTS {
            return Err(FinitePrefixVerifyError::Profile("source count"));
        }
        if self.relation_digest == [F::ZERO; DIGEST_SIZE]
            || self.index_digest == [F::ZERO; DIGEST_SIZE]
        {
            return Err(FinitePrefixVerifyError::Profile("zero setup digest"));
        }
        let l = self.layout;
        if l.l_skip != 0
            || l.rows_per_leaf != FINITE_PREFIX_LINK_ROWS_PER_LEAF
            || l.log_codeword_len != l.log_message_len.saturating_add(l.log_blowup)
            || l.log_message_len as usize != params.log_stacked_height()
            || l.log_blowup as usize != params.log_blowup
            || self.whir_sumcheck_rounds != params.num_whir_sumcheck_rounds()
            || params.commit_rows_per_query() != l.rows_per_leaf as usize
            || params.fold_rows_per_query() != l.rows_per_leaf as usize
            || !params.whir_leaf_layout_matches_fold()
        {
            return Err(FinitePrefixVerifyError::Profile("WHIR/layout mismatch"));
        }
        let message_len = 1u64
            .checked_shl(l.log_message_len)
            .ok_or(FinitePrefixVerifyError::Profile("message dimension"))?;
        let active_height = 1u64
            .checked_shl(l.active_count_log_height)
            .ok_or(FinitePrefixVerifyError::Profile("active-count dimension"))?;
        if l.trace_prefix_len > message_len
            || l.active_count_block_start % active_height != 0
            || l.active_count_block_start
                .checked_add(active_height)
                .is_none_or(|end| end > l.trace_prefix_len)
        {
            return Err(FinitePrefixVerifyError::Profile("prefix range"));
        }
        if params.l_skip != 0
            || params.log_blowup != 1
            || params.k_whir() != 4
            || F::GENERATOR.exp_power_of_2(l.log_message_len as usize) == F::ONE
        {
            return Err(FinitePrefixVerifyError::Profile("two-coset parameters"));
        }
        Ok(())
    }

    pub fn multi_whir_profile(
        &self,
    ) -> Result<MultiConstraintWhirProfile, FinitePrefixVerifyError> {
        MultiConstraintWhirProfile::new(
            2,
            self.layout.log_message_len as usize,
            vec![self.source_count, self.source_count],
        )
        .map_err(|_| FinitePrefixVerifyError::Profile("multi-WHIR profile"))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FinitePrefixVerifyError {
    Profile(&'static str),
    Statement(&'static str),
    ProofShape(&'static str),
    Sumcheck,
    Transcript,
    TraceHeight,
    RecursiveWhir(String),
}

impl core::fmt::Display for FinitePrefixVerifyError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "finite-prefix recursive verifier: {self:?}")
    }
}
impl std::error::Error for FinitePrefixVerifyError {}

impl From<MultiConstraintWhirModuleError> for FinitePrefixVerifyError {
    fn from(value: MultiConstraintWhirModuleError) -> Self {
        Self::RecursiveWhir(value.to_string())
    }
}

#[derive(Clone, Debug)]
struct FinitePrefixAuthorityWitness {
    source_batching: EF,
    mask_point: Vec<EF>,
    sumcheck_point: Vec<EF>,
    claims: Vec<EF>,
    source_powers: Vec<EF>,
    difference_accs: Vec<EF>,
    range_less: Vec<EF>,
    range_equal: Vec<EF>,
}

/// Fully replayed, backend-neutral input to the recursive verifier.
///
/// Construction succeeds only after the exact finite-prefix transcript and
/// quadratic terminal identity have been checked.  This host validation is
/// not authority: all values are copied into AIR columns, while `transcript`,
/// `multi_preflight`, and the concrete `whir` proof feed the ordinary
/// recursive transcript/Merkle/WHIR owners.
#[derive(Clone, Debug)]
pub struct PreparedFinitePrefixCircuitProof {
    pub proof: FinitePrefixCircuitProof,
    pub transcript: TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    pub multi_statement: MultiConstraintWhirStatement,
    pub multi_preflight: MultiConstraintWhirTranscriptPreflight,
    pub initial_commitments: Vec<MultiConstraintWhirInitialCommitment>,
    pub terminal_checkpoint: MultiConstraintWhirTerminalCheckpoint,
    authority: FinitePrefixAuthorityWitness,
}

fn transcript_sample_ext<TS>(transcript: &mut TS) -> EF
where
    TS: FiatShamirTranscript<BabyBearPoseidon2Config>,
{
    <TS as FiatShamirTranscript<BabyBearPoseidon2Config>>::sample_ext(transcript)
}

fn transcript_observe<TS>(transcript: &mut TS, value: F)
where
    TS: FiatShamirTranscript<BabyBearPoseidon2Config>,
{
    <TS as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(transcript, value);
}

fn transcript_observe_ext<TS>(transcript: &mut TS, value: EF)
where
    TS: FiatShamirTranscript<BabyBearPoseidon2Config>,
{
    <TS as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe_ext(transcript, value);
}

fn transcript_observe_digest<TS>(transcript: &mut TS, digest: Digest)
where
    TS: FiatShamirTranscript<BabyBearPoseidon2Config>,
{
    for value in digest {
        transcript_observe(transcript, value);
    }
}

fn replay_prefix_header<TS>(transcript: &mut TS, statement: &FinitePrefixCircuitStatement)
where
    TS: FiatShamirTranscript<BabyBearPoseidon2Config>,
{
    let context = statement.context;
    let layout = statement.layout;
    for value in [
        F::from_u64(PREFIX_LINK_TRANSCRIPT_TAG),
        F::from_u32(FINITE_PREFIX_LINK_PROTOCOL_VERSION),
        F::from_u64(PREFIX_LINK_TWO_COSET_RS_TAG),
        F::from_u32(FINITE_PREFIX_TWO_COSET_RS_LAYOUT_VERSION),
        F::from_u32(FINITE_PREFIX_TWO_COSET_INTERLEAVED_ORDERING),
        F::from_u64(PREFIX_LINK_COEFFICIENT_MESSAGE_TAG),
        F::GENERATOR,
    ] {
        transcript_observe(transcript, value);
    }
    for digest in [
        context.relation_digest,
        context.index_digest,
        context.schedule_digest,
        context.source_order_digest,
    ] {
        transcript_observe_digest(transcript, digest);
    }
    for value in [
        context.call_index,
        context.source_start,
        context.source_count,
    ] {
        transcript_observe(transcript, F::from_u32(value));
    }
    for count in context.expected_active_child_counts {
        transcript_observe(transcript, F::from_u8(count));
    }
    for value in [
        layout.l_skip,
        layout.log_message_len,
        layout.log_blowup,
        layout.log_codeword_len,
        layout.rows_per_leaf,
    ] {
        transcript_observe(transcript, F::from_u32(value));
    }
    transcript_observe(transcript, F::from_u64(layout.trace_prefix_len));
    transcript_observe(transcript, F::from_u64(layout.active_count_block_start));
    transcript_observe(transcript, F::from_u32(layout.active_count_log_height));
    transcript_observe(transcript, F::from_u64(PREFIX_LINK_BASE_PHASE_TAG));
    transcript_observe_digest(transcript, statement.base_root);
}

fn native_range_eq_states(a: &[EF], b: &[EF], prefix_len: u64) -> (Vec<EF>, Vec<EF>) {
    let mut less = Vec::with_capacity(a.len() + 1);
    let mut equal = Vec::with_capacity(a.len() + 1);
    less.push(EF::ZERO);
    equal.push(EF::ONE);
    let full_domain = prefix_len == (1u64 << a.len());
    for bit in 0..a.len() {
        let zero = (EF::ONE - a[bit]) * (EF::ONE - b[bit]);
        let one = a[bit] * b[bit];
        let eq_bit = zero + one;
        let (next_less, next_equal) = if full_domain {
            (EF::ZERO, equal[bit] * eq_bit)
        } else if (prefix_len >> (a.len() - 1 - bit)) & 1 == 0 {
            (less[bit] * eq_bit, equal[bit] * zero)
        } else {
            (less[bit] * eq_bit + equal[bit] * zero, equal[bit] * one)
        };
        less.push(next_less);
        equal.push(next_equal);
    }
    (less, equal)
}

/// Replay the SDK verifier byte-for-byte up to and including ordinary WHIR.
/// The first-round domain distinction is enforced by
/// [`FinitePrefixTwoCosetWhirQueryAir`] during recursive verification.
pub fn prepare_finite_prefix_circuit_proof(
    params: &SystemParams,
    profile: &FinitePrefixVerifierProfile,
    proof: &FinitePrefixCircuitProof,
) -> Result<PreparedFinitePrefixCircuitProof, FinitePrefixVerifyError> {
    profile.validate(params)?;
    let statement = &proof.statement;
    if statement.protocol_version != FINITE_PREFIX_LINK_PROTOCOL_VERSION
        || statement.context.relation_digest != profile.relation_digest
        || statement.context.index_digest != profile.index_digest
        || statement.context.source_count as usize != profile.source_count
        || statement.layout != profile.layout
    {
        return Err(FinitePrefixVerifyError::Statement("fixed statement"));
    }
    let height = 1usize
        .checked_shl(statement.layout.active_count_log_height)
        .ok_or(FinitePrefixVerifyError::Statement("active-count height"))?;
    if statement.context.expected_active_child_counts[..profile.source_count]
        .iter()
        .any(|&count| count == 0 || usize::from(count) > height)
        || statement.context.expected_active_child_counts[profile.source_count..]
            .iter()
            .any(|&count| count != 0)
    {
        return Err(FinitePrefixVerifyError::Statement("active-child counts"));
    }
    let m = profile.layout.log_message_len as usize;
    let n = profile.source_count;
    if proof.sumcheck.round_evaluations.len() != m
        || proof.sumcheck.base_openings.len() != n
        || proof.sumcheck.full_openings.len() != n
    {
        return Err(FinitePrefixVerifyError::ProofShape("prefix sumcheck"));
    }

    let mut transcript = default_duplex_sponge_recorder();
    replay_prefix_header(&mut transcript, statement);
    let alpha = transcript_sample_ext(&mut transcript);
    let beta = transcript_sample_ext(&mut transcript);
    if (alpha, beta) != (statement.logup_alpha, statement.logup_beta) {
        return Err(FinitePrefixVerifyError::Transcript);
    }
    transcript_observe(&mut transcript, F::from_u64(PREFIX_LINK_FULL_PHASE_TAG));
    transcript_observe_digest(&mut transcript, statement.full_root);
    let source_batching = transcript_sample_ext(&mut transcript);
    let mask_point = (0..m)
        .map(|_| transcript_sample_ext(&mut transcript))
        .collect::<Vec<_>>();

    let mut claim = EF::ZERO;
    let mut claims = vec![claim];
    let mut sumcheck_point = Vec::with_capacity(m);
    for (round, evaluations) in proof.sumcheck.round_evaluations.iter().enumerate() {
        if evaluations[0] + evaluations[1] != claim {
            return Err(FinitePrefixVerifyError::Sumcheck);
        }
        transcript_observe(&mut transcript, F::from_u64(PREFIX_LINK_SUMCHECK_TAG));
        transcript_observe(&mut transcript, F::from_usize(round));
        for &evaluation in evaluations {
            transcript_observe_ext(&mut transcript, evaluation);
        }
        let challenge = transcript_sample_ext(&mut transcript);
        claim = interpolate_quadratic_at_012(evaluations, challenge);
        claims.push(claim);
        sumcheck_point.push(challenge);
    }
    transcript_observe(&mut transcript, F::from_u64(PREFIX_LINK_OPENINGS_TAG));
    transcript_observe(&mut transcript, F::from_usize(m));
    for &coordinate in &sumcheck_point {
        transcript_observe_ext(&mut transcript, coordinate);
    }
    for openings in [&proof.sumcheck.base_openings, &proof.sumcheck.full_openings] {
        transcript_observe(&mut transcript, F::from_usize(openings.len()));
        for &opening in openings {
            transcript_observe_ext(&mut transcript, opening);
        }
    }

    let mut source_powers = Vec::with_capacity(n + 1);
    let mut difference_accs = Vec::with_capacity(n + 1);
    let mut power = EF::ONE;
    let mut difference = EF::ZERO;
    source_powers.push(power);
    difference_accs.push(difference);
    for (&base, &full) in proof
        .sumcheck
        .base_openings
        .iter()
        .zip(&proof.sumcheck.full_openings)
    {
        difference += power * (full - base);
        power *= source_batching;
        source_powers.push(power);
        difference_accs.push(difference);
    }
    let (range_less, range_equal) = native_range_eq_states(
        &mask_point,
        &sumcheck_point,
        statement.layout.trace_prefix_len,
    );
    let mask = if statement.layout.trace_prefix_len == (1u64 << m) {
        range_equal[m]
    } else {
        range_less[m]
    };
    if claim != difference * mask {
        return Err(FinitePrefixVerifyError::Sumcheck);
    }

    let (batching_prefix_tidx, batching_gamma, batching_coefficients) =
        derive_batching_coefficients_preflight(&mut transcript, 2)
            .map_err(|_| FinitePrefixVerifyError::Transcript)?;
    let multi_preflight = run_multi_constraint_whir_preflight(
        &mut transcript,
        params,
        &proof.whir,
        2,
        batching_prefix_tidx,
        batching_gamma,
    )
    .map_err(|_| FinitePrefixVerifyError::ProofShape("WHIR"))?;

    let active_openings = statement.context.expected_active_child_counts[..profile.source_count]
        .iter()
        .map(|&count| EF::from(F::from_u8(count)) / EF::from(F::from_usize(height)))
        .collect::<Vec<_>>();
    let multi_statement = MultiConstraintWhirStatement {
        points: vec![
            sumcheck_point.iter().rev().copied().collect(),
            active_count_point(profile),
        ],
        openings: vec![
            vec![
                proof.sumcheck.base_openings.clone(),
                proof.sumcheck.full_openings.clone(),
            ],
            vec![active_openings.clone(), active_openings],
        ],
        batching_coefficients,
    };
    multi_statement
        .validate(&profile.multi_whir_profile()?)
        .map_err(|_| FinitePrefixVerifyError::ProofShape("WHIR statement"))?;

    let log = TranscriptHistory::into_log(transcript);
    let trailing_samples = log
        .samples()
        .iter()
        .rev()
        .take_while(|&&sample| sample)
        .count();
    let sample_count = match trailing_samples % 8 {
        0 => 8,
        remainder => remainder,
    };
    let terminal_checkpoint = MultiConstraintWhirTerminalCheckpoint {
        end_tidx: log.len(),
        sample_count,
        state: *log
            .perm_results()
            .last()
            .ok_or(FinitePrefixVerifyError::Transcript)?,
    };
    Ok(PreparedFinitePrefixCircuitProof {
        proof: proof.clone(),
        transcript: log,
        multi_statement,
        multi_preflight,
        initial_commitments: vec![
            MultiConstraintWhirInitialCommitment {
                commitment: statement.base_root,
                width: n,
            },
            MultiConstraintWhirInitialCommitment {
                commitment: statement.full_root,
                width: n,
            },
        ],
        terminal_checkpoint,
        authority: FinitePrefixAuthorityWitness {
            source_batching,
            mask_point,
            sumcheck_point,
            claims,
            source_powers,
            difference_accs,
            range_less,
            range_equal,
        },
    })
}

#[derive(Clone, Copy, Debug)]
pub struct FinitePrefixVerifierBuses {
    pub transcript: TranscriptBus,
    pub statement: MultiConstraintStatementBuses,
    pub initial_commitment: MultiConstraintInitialCommitmentBus,
    pub whir_completion: MultiConstraintWhirCompletionBus,
    pub authority: OrderedManifestPrefixAuthorityBus,
}

#[derive(Clone, Debug)]
struct PrefixRowLayout {
    active: usize,
    statement: Range<usize>,
    count_minus_one_bits: Range<usize>,
    source_batching: Range<usize>,
    mask_point: Range<usize>,
    sumcheck_point: Range<usize>,
    round_evaluations: Range<usize>,
    claims: Range<usize>,
    source_powers: Range<usize>,
    difference_accs: Range<usize>,
    base_openings: Range<usize>,
    full_openings: Range<usize>,
    range_less: Range<usize>,
    range_equal: Range<usize>,
    completion: Range<usize>,
    width: usize,
}

fn take(cursor: &mut usize, len: usize) -> Range<usize> {
    let start = *cursor;
    *cursor += len;
    start..*cursor
}

impl PrefixRowLayout {
    fn new(profile: &FinitePrefixVerifierProfile) -> Self {
        let m = profile.layout.log_message_len as usize;
        let n = profile.source_count;
        let count_bits = profile.layout.active_count_log_height as usize;
        let mut c = 0;
        let active = take(&mut c, 1).start;
        let statement = take(&mut c, 134);
        let count_minus_one_bits = take(&mut c, ORDERED_MANIFEST_PREFIX_SLOTS * count_bits);
        let source_batching = take(&mut c, D_EF);
        let mask_point = take(&mut c, m * D_EF);
        let sumcheck_point = take(&mut c, m * D_EF);
        let round_evaluations = take(&mut c, m * 3 * D_EF);
        let claims = take(&mut c, (m + 1) * D_EF);
        let source_powers = take(&mut c, (n + 1) * D_EF);
        let difference_accs = take(&mut c, (n + 1) * D_EF);
        let base_openings = take(&mut c, n * D_EF);
        let full_openings = take(&mut c, n * D_EF);
        let range_less = take(&mut c, (m + 1) * D_EF);
        let range_equal = take(&mut c, (m + 1) * D_EF);
        let completion = take(&mut c, 2 + POSEIDON2_WIDTH + 2 * D_EF);
        Self {
            active,
            statement,
            count_minus_one_bits,
            source_batching,
            mask_point,
            sumcheck_point,
            round_evaluations,
            claims,
            source_powers,
            difference_accs,
            base_openings,
            full_openings,
            range_less,
            range_equal,
            completion,
            width: c,
        }
    }

    fn ext<T: Copy>(&self, row: &[T], range: &Range<usize>, index: usize) -> [T; D_EF] {
        row[range.start + index * D_EF..range.start + (index + 1) * D_EF]
            .try_into()
            .expect("extension slice")
    }
}

fn push_digest(fields: &mut Vec<F>, digest: Digest) {
    fields.extend(digest);
}

fn push_ext(fields: &mut Vec<F>, value: EF) {
    fields.extend_from_slice(value.as_basis_coefficients_slice());
}

fn finite_prefix_statement_fields(statement: &FinitePrefixCircuitStatement) -> Vec<F> {
    let mut fields = Vec::with_capacity(134);
    fields.push(F::from_u32(statement.protocol_version));
    push_digest(&mut fields, statement.context.relation_digest);
    push_digest(&mut fields, statement.context.index_digest);
    push_digest(&mut fields, statement.context.schedule_digest);
    push_digest(&mut fields, statement.context.source_order_digest);
    fields.push(F::from_u32(statement.context.call_index));
    fields.push(F::from_u32(statement.context.source_start));
    fields.push(F::from_u32(statement.context.source_count));
    fields.extend(
        statement
            .context
            .expected_active_child_counts
            .map(F::from_u8),
    );
    for value in [
        statement.layout.l_skip,
        statement.layout.log_message_len,
        statement.layout.log_blowup,
        statement.layout.log_codeword_len,
        statement.layout.rows_per_leaf,
    ] {
        fields.push(F::from_u32(value));
    }
    fields.push(F::from_u32(statement.layout.trace_prefix_len as u32));
    fields.push(F::from_u32(
        (statement.layout.trace_prefix_len >> 32) as u32,
    ));
    fields.push(F::from_u32(
        statement.layout.active_count_block_start as u32,
    ));
    fields.push(F::from_u32(
        (statement.layout.active_count_block_start >> 32) as u32,
    ));
    fields.push(F::from_u32(statement.layout.active_count_log_height));
    push_digest(&mut fields, statement.base_root);
    push_digest(&mut fields, statement.full_root);
    push_ext(&mut fields, statement.logup_alpha);
    push_ext(&mut fields, statement.logup_beta);
    debug_assert_eq!(fields.len(), 134);
    fields
}

fn copy_ext_to_range(row: &mut [F], range: &Range<usize>, index: usize, value: EF) {
    row[range.start + index * D_EF..range.start + (index + 1) * D_EF]
        .copy_from_slice(value.as_basis_coefficients_slice());
}

impl PreparedFinitePrefixCircuitProof {
    /// Build the authority trace. `completion` must be projected from the
    /// recursively constrained two-coset WHIR path, never from a host verdict.
    pub fn authority_trace(
        &self,
        profile: &FinitePrefixVerifierProfile,
        completion: &MultiConstraintWhirCompletionMessage<F>,
    ) -> Result<RowMajorMatrix<F>, FinitePrefixVerifyError> {
        let layout = PrefixRowLayout::new(profile);
        let mut values = F::zero_vec(2 * layout.width);
        let row = &mut values[..layout.width];
        row[layout.active] = F::ONE;
        let statement_fields = finite_prefix_statement_fields(&self.proof.statement);
        row[layout.statement.clone()].copy_from_slice(&statement_fields);

        let count_bits = profile.layout.active_count_log_height as usize;
        for (source, &count) in self
            .proof
            .statement
            .context
            .expected_active_child_counts
            .iter()
            .enumerate()
        {
            let value = u32::from(count).saturating_sub(1);
            for bit in 0..count_bits {
                row[layout.count_minus_one_bits.start + source * count_bits + bit] =
                    F::from_bool(((value >> bit) & 1) != 0);
            }
        }
        copy_ext_to_range(
            row,
            &layout.source_batching,
            0,
            self.authority.source_batching,
        );
        for (index, &value) in self.authority.mask_point.iter().enumerate() {
            copy_ext_to_range(row, &layout.mask_point, index, value);
        }
        for (index, &value) in self.authority.sumcheck_point.iter().enumerate() {
            copy_ext_to_range(row, &layout.sumcheck_point, index, value);
        }
        for (round, evaluations) in self.proof.sumcheck.round_evaluations.iter().enumerate() {
            for (at, &value) in evaluations.iter().enumerate() {
                copy_ext_to_range(row, &layout.round_evaluations, round * 3 + at, value);
            }
        }
        for (index, &value) in self.authority.claims.iter().enumerate() {
            copy_ext_to_range(row, &layout.claims, index, value);
        }
        for (index, &value) in self.authority.source_powers.iter().enumerate() {
            copy_ext_to_range(row, &layout.source_powers, index, value);
        }
        for (index, &value) in self.authority.difference_accs.iter().enumerate() {
            copy_ext_to_range(row, &layout.difference_accs, index, value);
        }
        for (index, &value) in self.proof.sumcheck.base_openings.iter().enumerate() {
            copy_ext_to_range(row, &layout.base_openings, index, value);
        }
        for (index, &value) in self.proof.sumcheck.full_openings.iter().enumerate() {
            copy_ext_to_range(row, &layout.full_openings, index, value);
        }
        for (index, &value) in self.authority.range_less.iter().enumerate() {
            copy_ext_to_range(row, &layout.range_less, index, value);
        }
        for (index, &value) in self.authority.range_equal.iter().enumerate() {
            copy_ext_to_range(row, &layout.range_equal, index, value);
        }
        let c = layout.completion.start;
        row[c] = completion.end_tidx;
        row[c + 1] = completion.sample_count;
        row[c + 2..c + 2 + POSEIDON2_WIDTH].copy_from_slice(&completion.state);
        row[c + 2 + POSEIDON2_WIDTH..c + 2 + POSEIDON2_WIDTH + D_EF]
            .copy_from_slice(&completion.final_aggregate);
        row[c + 2 + POSEIDON2_WIDTH + D_EF..c + 2 + POSEIDON2_WIDTH + 2 * D_EF]
            .copy_from_slice(&completion.final_claim);
        Ok(RowMajorMatrix::new(values, layout.width))
    }
}

#[derive(Clone, Debug)]
pub struct FinitePrefixAuthorityAir {
    pub profile: FinitePrefixVerifierProfile,
    pub buses: FinitePrefixVerifierBuses,
}

impl FinitePrefixAuthorityAir {
    pub fn new(
        params: &SystemParams,
        profile: FinitePrefixVerifierProfile,
        buses: FinitePrefixVerifierBuses,
    ) -> Result<Self, FinitePrefixVerifyError> {
        profile.validate(params)?;
        Ok(Self { profile, buses })
    }

    fn layout(&self) -> PrefixRowLayout {
        PrefixRowLayout::new(&self.profile)
    }
}

impl BaseAir<F> for FinitePrefixAuthorityAir {
    fn width(&self) -> usize {
        self.layout().width
    }
}
impl BaseAirWithPublicValues<F> for FinitePrefixAuthorityAir {}
impl PartitionedBaseAir<F> for FinitePrefixAuthorityAir {}

fn ext_from_slice<E: Clone>(slice: &[E]) -> [E; D_EF] {
    assert_eq!(slice.len(), D_EF, "extension slice");
    core::array::from_fn(|i| slice[i].clone())
}

fn assert_ext_eq<AB: AirBuilder>(
    builder: &mut AB,
    left: [impl Into<AB::Expr>; D_EF],
    right: [impl Into<AB::Expr>; D_EF],
) {
    for (left, right) in left.into_iter().zip(right) {
        builder.assert_eq(left, right);
    }
}

fn ext_one<E: PrimeCharacteristicRing>() -> [E; D_EF] {
    [E::ONE, E::ZERO, E::ZERO, E::ZERO]
}

// Exact offsets in OrderedManifestPrefixReceiptMessage::to_vec.
const REL: usize = 1;
const IDX: usize = REL + DIGEST_SIZE;
const SCHEDULE: usize = IDX + DIGEST_SIZE;
const SOURCE_ORDER: usize = SCHEDULE + DIGEST_SIZE;
const CALL: usize = SOURCE_ORDER + DIGEST_SIZE;
const SOURCE_START: usize = CALL + 1;
const SOURCE_COUNT: usize = SOURCE_START + 1;
const COUNTS: usize = SOURCE_COUNT + 1;
const L_SKIP: usize = COUNTS + ORDERED_MANIFEST_PREFIX_SLOTS;
const LOG_M: usize = L_SKIP + 1;
const LOG_B: usize = LOG_M + 1;
const LOG_C: usize = LOG_B + 1;
const ROWS: usize = LOG_C + 1;
const PREFIX_LO: usize = ROWS + 1;
const PREFIX_HI: usize = PREFIX_LO + 1;
const COUNT_START_LO: usize = PREFIX_HI + 1;
const COUNT_START_HI: usize = COUNT_START_LO + 1;
const COUNT_LOG: usize = COUNT_START_HI + 1;
const BASE_ROOT: usize = COUNT_LOG + 1;
const FULL_ROOT: usize = BASE_ROOT + DIGEST_SIZE;
const LOGUP_ALPHA: usize = FULL_ROOT + DIGEST_SIZE;
const LOGUP_BETA: usize = LOGUP_ALPHA + D_EF;
const _: () = assert!(LOGUP_BETA + D_EF == 134);

fn ts_observe<AB>(
    bus: TranscriptBus,
    builder: &mut AB,
    tidx: &mut usize,
    value: impl Into<AB::Expr>,
    enabled: AB::Expr,
) where
    AB: AirBuilder<F = F> + InteractionBuilder,
{
    bus.observe(
        builder,
        AB::Expr::ZERO,
        AB::Expr::from_usize(*tidx),
        value,
        enabled,
    );
    *tidx += 1;
}

fn ts_observe_ext<AB>(
    bus: TranscriptBus,
    builder: &mut AB,
    tidx: &mut usize,
    value: [impl Into<AB::Expr>; D_EF],
    enabled: AB::Expr,
) where
    AB: AirBuilder<F = F> + InteractionBuilder,
{
    bus.observe_ext(
        builder,
        AB::Expr::ZERO,
        AB::Expr::from_usize(*tidx),
        value,
        enabled,
    );
    *tidx += D_EF;
}

fn ts_sample_ext<AB>(
    bus: TranscriptBus,
    builder: &mut AB,
    tidx: &mut usize,
    value: [impl Into<AB::Expr>; D_EF],
    enabled: AB::Expr,
) where
    AB: AirBuilder<F = F> + InteractionBuilder,
{
    bus.sample_ext(
        builder,
        AB::Expr::ZERO,
        AB::Expr::from_usize(*tidx),
        value,
        enabled,
    );
    *tidx += D_EF;
}

fn ts_observe_digest<AB>(
    bus: TranscriptBus,
    builder: &mut AB,
    tidx: &mut usize,
    digest: [impl Into<AB::Expr>; DIGEST_SIZE],
    enabled: AB::Expr,
) where
    AB: AirBuilder<F = F> + InteractionBuilder,
{
    for value in digest {
        ts_observe(bus, builder, tidx, value, enabled.clone());
    }
}

impl<AB> Air<AB> for FinitePrefixAuthorityAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield:
        BinomiallyExtendable<{ D_EF }> + TwoAdicField,
{
    fn eval(&self, builder: &mut AB) {
        let layout = self.layout();
        let main = builder.main();
        let local = main.row_slice(0).expect("finite-prefix authority row");
        let next = main.row_slice(1).expect("finite-prefix padding row");
        let active = local[layout.active];
        builder.assert_bool(active);
        builder.when_first_row().assert_one(active);
        builder.when_last_row().assert_zero(active);
        builder
            .when_transition()
            .assert_eq(active - next[layout.active], active);
        let enabled: AB::Expr = active.into();
        for value in local.iter().skip(1) {
            builder
                .when(AB::Expr::ONE - enabled.clone())
                .assert_zero(*value);
        }

        let s = &local[layout.statement.clone()];
        let at = |i: usize| s[i];
        let digest = |start: usize| core::array::from_fn(|i| at(start + i));
        let expect = |builder: &mut AB, value: AB::Var, expected: u32| {
            builder
                .when(enabled.clone())
                .assert_eq(value, AB::Expr::from_u32(expected));
        };
        expect(builder, at(0), FINITE_PREFIX_LINK_PROTOCOL_VERSION);
        for (actual, expected) in digest(REL).into_iter().zip(self.profile.relation_digest) {
            expect(builder, actual, expected.as_canonical_u32());
        }
        for (actual, expected) in digest(IDX).into_iter().zip(self.profile.index_digest) {
            expect(builder, actual, expected.as_canonical_u32());
        }
        expect(builder, at(SOURCE_COUNT), self.profile.source_count as u32);
        let p = self.profile.layout;
        for (offset, expected) in [
            (L_SKIP, p.l_skip),
            (LOG_M, p.log_message_len),
            (LOG_B, p.log_blowup),
            (LOG_C, p.log_codeword_len),
            (ROWS, p.rows_per_leaf),
            (PREFIX_LO, p.trace_prefix_len as u32),
            (PREFIX_HI, (p.trace_prefix_len >> 32) as u32),
            (COUNT_START_LO, p.active_count_block_start as u32),
            (COUNT_START_HI, (p.active_count_block_start >> 32) as u32),
            (COUNT_LOG, p.active_count_log_height),
        ] {
            expect(builder, at(offset), expected);
        }

        // The SDK accepts counts 1..=2^active_count_log_height in active
        // slots and requires a canonical zero tail. Encode count - 1.
        let count_bits = p.active_count_log_height as usize;
        for source in 0..ORDERED_MANIFEST_PREFIX_SLOTS {
            let count = at(COUNTS + source);
            if source < self.profile.source_count {
                let mut reconstructed = AB::Expr::ONE;
                for bit in 0..count_bits {
                    let value =
                        local[layout.count_minus_one_bits.start + source * count_bits + bit];
                    builder.assert_bool(value);
                    reconstructed += value * AB::Expr::from_u32(1u32 << bit);
                }
                builder
                    .when(enabled.clone())
                    .assert_eq(count, reconstructed);
            } else {
                builder.when(enabled.clone()).assert_zero(count);
                for bit in 0..count_bits {
                    builder.when(enabled.clone()).assert_zero(
                        local[layout.count_minus_one_bits.start + source * count_bits + bit],
                    );
                }
            }
        }

        let mut tidx = 0usize;
        for value in [
            F::from_u64(PREFIX_LINK_TRANSCRIPT_TAG),
            F::from_u32(FINITE_PREFIX_LINK_PROTOCOL_VERSION),
            F::from_u64(PREFIX_LINK_TWO_COSET_RS_TAG),
            F::from_u32(FINITE_PREFIX_TWO_COSET_RS_LAYOUT_VERSION),
            F::from_u32(FINITE_PREFIX_TWO_COSET_INTERLEAVED_ORDERING),
            F::from_u64(PREFIX_LINK_COEFFICIENT_MESSAGE_TAG),
            F::GENERATOR,
        ] {
            ts_observe(
                self.buses.transcript,
                builder,
                &mut tidx,
                AB::Expr::from_u32(value.as_canonical_u32()),
                enabled.clone(),
            );
        }
        for value in [
            digest(REL),
            digest(IDX),
            digest(SCHEDULE),
            digest(SOURCE_ORDER),
        ] {
            ts_observe_digest(
                self.buses.transcript,
                builder,
                &mut tidx,
                value,
                enabled.clone(),
            );
        }
        for offset in [CALL, SOURCE_START, SOURCE_COUNT] {
            ts_observe(
                self.buses.transcript,
                builder,
                &mut tidx,
                at(offset),
                enabled.clone(),
            );
        }
        for source in 0..ORDERED_MANIFEST_PREFIX_SLOTS {
            ts_observe(
                self.buses.transcript,
                builder,
                &mut tidx,
                at(COUNTS + source),
                enabled.clone(),
            );
        }
        for offset in [L_SKIP, LOG_M, LOG_B, LOG_C, ROWS] {
            ts_observe(
                self.buses.transcript,
                builder,
                &mut tidx,
                at(offset),
                enabled.clone(),
            );
        }
        for value in [
            F::from_u64(p.trace_prefix_len),
            F::from_u64(p.active_count_block_start),
            F::from_u32(p.active_count_log_height),
            F::from_u64(PREFIX_LINK_BASE_PHASE_TAG),
        ] {
            ts_observe(
                self.buses.transcript,
                builder,
                &mut tidx,
                AB::Expr::from_u32(value.as_canonical_u32()),
                enabled.clone(),
            );
        }
        ts_observe_digest(
            self.buses.transcript,
            builder,
            &mut tidx,
            digest(BASE_ROOT),
            enabled.clone(),
        );
        ts_sample_ext(
            self.buses.transcript,
            builder,
            &mut tidx,
            ext_from_slice(&s[LOGUP_ALPHA..LOGUP_ALPHA + D_EF]),
            enabled.clone(),
        );
        ts_sample_ext(
            self.buses.transcript,
            builder,
            &mut tidx,
            ext_from_slice(&s[LOGUP_BETA..LOGUP_BETA + D_EF]),
            enabled.clone(),
        );
        ts_observe(
            self.buses.transcript,
            builder,
            &mut tidx,
            AB::Expr::from_u32(F::from_u64(PREFIX_LINK_FULL_PHASE_TAG).as_canonical_u32()),
            enabled.clone(),
        );
        ts_observe_digest(
            self.buses.transcript,
            builder,
            &mut tidx,
            digest(FULL_ROOT),
            enabled.clone(),
        );

        let source_batching = layout.ext(&local, &layout.source_batching, 0);
        ts_sample_ext(
            self.buses.transcript,
            builder,
            &mut tidx,
            source_batching,
            enabled.clone(),
        );
        let m = p.log_message_len as usize;
        let mask_point = (0..m)
            .map(|i| layout.ext(&local, &layout.mask_point, i))
            .collect::<Vec<_>>();
        for value in &mask_point {
            ts_sample_ext(
                self.buses.transcript,
                builder,
                &mut tidx,
                *value,
                enabled.clone(),
            );
        }

        let sumcheck_point = (0..m)
            .map(|i| layout.ext(&local, &layout.sumcheck_point, i))
            .collect::<Vec<_>>();
        let claims = (0..=m)
            .map(|i| layout.ext(&local, &layout.claims, i))
            .collect::<Vec<_>>();
        assert_ext_eq(
            &mut builder.when(enabled.clone()),
            claims[0],
            [AB::Expr::ZERO; D_EF],
        );
        for round in 0..m {
            let evals: [[AB::Var; D_EF]; 3] = core::array::from_fn(|j| {
                layout.ext(&local, &layout.round_evaluations, round * 3 + j)
            });
            assert_ext_eq(
                &mut builder.when(enabled.clone()),
                ext_field_add::<AB::Expr>(evals[0], evals[1]),
                claims[round],
            );
            ts_observe(
                self.buses.transcript,
                builder,
                &mut tidx,
                AB::Expr::from_u32(F::from_u64(PREFIX_LINK_SUMCHECK_TAG).as_canonical_u32()),
                enabled.clone(),
            );
            ts_observe(
                self.buses.transcript,
                builder,
                &mut tidx,
                AB::Expr::from_usize(round),
                enabled.clone(),
            );
            for value in evals {
                ts_observe_ext(
                    self.buses.transcript,
                    builder,
                    &mut tidx,
                    value,
                    enabled.clone(),
                );
            }
            ts_sample_ext(
                self.buses.transcript,
                builder,
                &mut tidx,
                sumcheck_point[round],
                enabled.clone(),
            );
            assert_ext_eq(
                &mut builder.when(enabled.clone()),
                claims[round + 1],
                interpolate_quadratic::<AB::Expr>(
                    claims[round].map(Into::into),
                    evals[1].map(Into::into),
                    evals[2].map(Into::into),
                    sumcheck_point[round].map(Into::into),
                ),
            );
        }

        self.eval_openings_and_publish(
            builder,
            &local,
            &layout,
            &mut tidx,
            enabled,
            s,
            &mask_point,
            &sumcheck_point,
            &claims,
            digest(BASE_ROOT),
            digest(FULL_ROOT),
        );
    }
}

impl FinitePrefixAuthorityAir {
    #[allow(clippy::too_many_arguments)]
    fn eval_openings_and_publish<AB>(
        &self,
        builder: &mut AB,
        local: &[AB::Var],
        layout: &PrefixRowLayout,
        tidx: &mut usize,
        enabled: AB::Expr,
        statement: &[AB::Var],
        mask_point: &[[AB::Var; D_EF]],
        sumcheck_point: &[[AB::Var; D_EF]],
        claims: &[[AB::Var; D_EF]],
        base_root: [AB::Var; DIGEST_SIZE],
        full_root: [AB::Var; DIGEST_SIZE],
    ) where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
        AB::Expr: From<AB::Var>,
        <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield:
            BinomiallyExtendable<{ D_EF }> + TwoAdicField,
    {
        let p = self.profile.layout;
        let m = p.log_message_len as usize;
        let n = self.profile.source_count;
        let base = (0..n)
            .map(|i| layout.ext(local, &layout.base_openings, i))
            .collect::<Vec<_>>();
        let full = (0..n)
            .map(|i| layout.ext(local, &layout.full_openings, i))
            .collect::<Vec<_>>();

        ts_observe(
            self.buses.transcript,
            builder,
            tidx,
            AB::Expr::from_u32(F::from_u64(PREFIX_LINK_OPENINGS_TAG).as_canonical_u32()),
            enabled.clone(),
        );
        ts_observe(
            self.buses.transcript,
            builder,
            tidx,
            AB::Expr::from_usize(m),
            enabled.clone(),
        );
        for value in sumcheck_point {
            ts_observe_ext(
                self.buses.transcript,
                builder,
                tidx,
                *value,
                enabled.clone(),
            );
        }
        for openings in [&base, &full] {
            ts_observe(
                self.buses.transcript,
                builder,
                tidx,
                AB::Expr::from_usize(n),
                enabled.clone(),
            );
            for value in openings.iter() {
                ts_observe_ext(
                    self.buses.transcript,
                    builder,
                    tidx,
                    *value,
                    enabled.clone(),
                );
            }
        }

        let source_batching = layout.ext(local, &layout.source_batching, 0);
        let powers = (0..=n)
            .map(|i| layout.ext(local, &layout.source_powers, i))
            .collect::<Vec<_>>();
        let accs = (0..=n)
            .map(|i| layout.ext(local, &layout.difference_accs, i))
            .collect::<Vec<_>>();
        assert_ext_eq(
            &mut builder.when(enabled.clone()),
            powers[0],
            ext_one::<AB::Expr>(),
        );
        assert_ext_eq(
            &mut builder.when(enabled.clone()),
            accs[0],
            [AB::Expr::ZERO; D_EF],
        );
        for source in 0..n {
            assert_ext_eq(
                &mut builder.when(enabled.clone()),
                powers[source + 1],
                ext_field_multiply::<AB::Expr>(powers[source], source_batching),
            );
            let delta = core::array::from_fn(|limb| {
                AB::Expr::from(full[source][limb]) - AB::Expr::from(base[source][limb])
            });
            assert_ext_eq(
                &mut builder.when(enabled.clone()),
                accs[source + 1],
                ext_field_add::<AB::Expr>(
                    accs[source],
                    ext_field_multiply::<AB::Expr>(powers[source], delta),
                ),
            );
        }

        // SDK range_eq_inner_product, with MSB-first coordinates.
        let less = (0..=m)
            .map(|i| layout.ext(local, &layout.range_less, i))
            .collect::<Vec<_>>();
        let equal = (0..=m)
            .map(|i| layout.ext(local, &layout.range_equal, i))
            .collect::<Vec<_>>();
        assert_ext_eq(
            &mut builder.when(enabled.clone()),
            less[0],
            [AB::Expr::ZERO; D_EF],
        );
        assert_ext_eq(
            &mut builder.when(enabled.clone()),
            equal[0],
            ext_one::<AB::Expr>(),
        );
        let full_domain = p.trace_prefix_len == (1u64 << p.log_message_len);
        for bit in 0..m {
            let a = mask_point[bit].map(Into::into);
            let b = sumcheck_point[bit].map(Into::into);
            let one_minus_a = core::array::from_fn(|i| {
                if i == 0 {
                    AB::Expr::ONE - a[0].clone()
                } else {
                    -a[i].clone()
                }
            });
            let one_minus_b = core::array::from_fn(|i| {
                if i == 0 {
                    AB::Expr::ONE - b[0].clone()
                } else {
                    -b[i].clone()
                }
            });
            let zero = ext_field_multiply::<AB::Expr>(one_minus_a, one_minus_b);
            let one = ext_field_multiply::<AB::Expr>(a, b);
            let eq_bit = ext_field_add::<AB::Expr>(zero.clone(), one.clone());
            if full_domain {
                assert_ext_eq(
                    &mut builder.when(enabled.clone()),
                    less[bit + 1],
                    [AB::Expr::ZERO; D_EF],
                );
                assert_ext_eq(
                    &mut builder.when(enabled.clone()),
                    equal[bit + 1],
                    ext_field_multiply::<AB::Expr>(equal[bit], eq_bit),
                );
            } else if (p.trace_prefix_len >> (m - 1 - bit)) & 1 == 0 {
                assert_ext_eq(
                    &mut builder.when(enabled.clone()),
                    less[bit + 1],
                    ext_field_multiply::<AB::Expr>(less[bit], eq_bit),
                );
                assert_ext_eq(
                    &mut builder.when(enabled.clone()),
                    equal[bit + 1],
                    ext_field_multiply::<AB::Expr>(equal[bit], zero),
                );
            } else {
                assert_ext_eq(
                    &mut builder.when(enabled.clone()),
                    less[bit + 1],
                    ext_field_add::<AB::Expr>(
                        ext_field_multiply::<AB::Expr>(less[bit], eq_bit),
                        ext_field_multiply::<AB::Expr>(equal[bit], zero),
                    ),
                );
                assert_ext_eq(
                    &mut builder.when(enabled.clone()),
                    equal[bit + 1],
                    ext_field_multiply::<AB::Expr>(equal[bit], one),
                );
            }
        }
        let mask = if full_domain { equal[m] } else { less[m] };
        assert_ext_eq(
            &mut builder.when(enabled.clone()),
            claims[m],
            ext_field_multiply::<AB::Expr>(accs[n], mask),
        );

        // The ordinary recursive multi-WHIR prefix owns MCWB and MCWP from
        // this exact next transcript index onward.
        self.buses.statement.caller_prefix.send(
            builder,
            AB::Expr::ZERO,
            MultiConstraintCallerPrefixMessage {
                tidx: AB::Expr::from_usize(*tidx),
            },
            enabled.clone(),
        );

        let active_point = active_count_point(&self.profile);
        let multi = self
            .profile
            .multi_whir_profile()
            .expect("validated multi-WHIR profile");
        for constraint in 0..2 {
            for coordinate in 0..m {
                let value: [AB::Expr; D_EF] = if constraint == 0 {
                    sumcheck_point[m - 1 - coordinate].map(Into::into)
                } else {
                    let coefficients: &[F] = active_point[coordinate].as_basis_coefficients_slice();
                    core::array::from_fn(|limb| {
                        AB::Expr::from_u32(coefficients[limb].as_canonical_u32())
                    })
                };
                let multiplicity = multi
                    .point_lookup_multiplicity(coordinate, self.profile.whir_sumcheck_rounds)
                    .expect("validated point multiplicity");
                self.buses.statement.point.add_key_with_lookups(
                    builder,
                    AB::Expr::ZERO,
                    MultiConstraintPointMessage {
                        constraint_idx: AB::Expr::from_usize(constraint),
                        coordinate_idx: AB::Expr::from_usize(coordinate),
                        value,
                    },
                    enabled.clone() * AB::Expr::from_usize(multiplicity),
                );
            }
        }
        let inv_height = AB::Expr::from_u32(
            F::from_u32(1u32 << p.active_count_log_height)
                .inverse()
                .as_canonical_u32(),
        );
        for constraint in 0..2 {
            for commitment in 0..2 {
                for source in 0..n {
                    let value = if constraint == 0 {
                        if commitment == 0 {
                            base[source].map(Into::into)
                        } else {
                            full[source].map(Into::into)
                        }
                    } else {
                        [
                            AB::Expr::from(statement[COUNTS + source]) * inv_height.clone(),
                            AB::Expr::ZERO,
                            AB::Expr::ZERO,
                            AB::Expr::ZERO,
                        ]
                    };
                    self.buses.statement.opening.add_key_with_lookups(
                        builder,
                        AB::Expr::ZERO,
                        MultiConstraintOpeningMessage {
                            constraint_idx: AB::Expr::from_usize(constraint),
                            opening_idx: AB::Expr::from_usize(commitment * n + source),
                            value,
                        },
                        enabled.clone(),
                    );
                }
            }
        }
        for (commit_idx, root) in [base_root, full_root].into_iter().enumerate() {
            self.buses.initial_commitment.send(
                builder,
                AB::Expr::ZERO,
                MultiConstraintInitialCommitmentMessage {
                    commit_idx: AB::Expr::from_usize(commit_idx),
                    width: AB::Expr::from_usize(n),
                    commitment: root.map(Into::into),
                },
                enabled.clone(),
            );
        }

        let c = &local[layout.completion.clone()];
        let state: [AB::Var; POSEIDON2_WIDTH] = c[2..2 + POSEIDON2_WIDTH]
            .try_into()
            .expect("checkpoint state");
        let final_aggregate: [AB::Var; D_EF] = c[2 + POSEIDON2_WIDTH..2 + POSEIDON2_WIDTH + D_EF]
            .try_into()
            .expect("final aggregate");
        let final_claim: [AB::Var; D_EF] = c
            [2 + POSEIDON2_WIDTH + D_EF..2 + POSEIDON2_WIDTH + 2 * D_EF]
            .try_into()
            .expect("final claim");
        self.buses.whir_completion.receive(
            builder,
            AB::Expr::ZERO,
            MultiConstraintWhirCompletionMessage {
                proof_idx: AB::Expr::ZERO,
                class_index: AB::Expr::from_usize(self.profile.class_index),
                end_tidx: c[0].into(),
                sample_count: c[1].into(),
                state: state.map(Into::into),
                final_aggregate: final_aggregate.map(Into::into),
                final_claim: final_claim.map(Into::into),
            },
            enabled.clone(),
        );

        let d = |start: usize| core::array::from_fn(|i| statement[start + i]);
        self.buses.authority.add_key_with_lookups(
            builder,
            OrderedManifestPrefixReceiptMessage {
                protocol_version: statement[0],
                relation_digest: d(REL),
                index_digest: d(IDX),
                schedule_digest: d(SCHEDULE),
                source_order_digest: d(SOURCE_ORDER),
                call_index: statement[CALL],
                source_start: statement[SOURCE_START],
                source_count: statement[SOURCE_COUNT],
                expected_active_child_counts: core::array::from_fn(|i| statement[COUNTS + i]),
                l_skip: statement[L_SKIP],
                log_message_len: statement[LOG_M],
                log_blowup: statement[LOG_B],
                log_codeword_len: statement[LOG_C],
                rows_per_leaf: statement[ROWS],
                trace_prefix_len_lo: statement[PREFIX_LO],
                trace_prefix_len_hi: statement[PREFIX_HI],
                active_count_block_start_lo: statement[COUNT_START_LO],
                active_count_block_start_hi: statement[COUNT_START_HI],
                active_count_log_height: statement[COUNT_LOG],
                base_root: d(BASE_ROOT),
                full_root: d(FULL_ROOT),
                logup_alpha: ext_from_slice(&statement[LOGUP_ALPHA..LOGUP_ALPHA + D_EF]),
                logup_beta: ext_from_slice(&statement[LOGUP_BETA..LOGUP_BETA + D_EF]),
            },
            enabled,
        );
    }
}

fn active_count_point(profile: &FinitePrefixVerifierProfile) -> Vec<EF> {
    let m = profile.layout.log_message_len as usize;
    let row_dim = profile.layout.active_count_log_height as usize;
    let height = 1usize << row_dim;
    let block = profile.layout.active_count_block_start as usize / height;
    let prefix_dim = m - row_dim;
    let mut point = Vec::with_capacity(m);
    for coordinate in 0..prefix_dim {
        point.push(EF::from(F::from_bool(
            ((block >> (prefix_dim - coordinate - 1)) & 1) != 0,
        )));
    }
    point.extend(core::iter::repeat_n(EF::TWO.inverse(), row_dim));
    point.reverse();
    point
}

/// Replacement for only the first-domain-sensitive recursive WHIR AIR.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FinitePrefixTwoCosetWhirQueryCols<T> {
    pub is_enabled: T,
    pub proof_idx: T,
    pub whir_round: T,
    pub query_idx: T,
    pub is_first_in_proof: T,
    pub is_first_in_round: T,
    pub tidx: T,
    pub num_queries: T,
    pub omega: T,
    pub sample: T,
    pub zi_root: T,
    pub zi: T,
    pub yi: [T; D_EF],
    pub gamma: [T; D_EF],
    pub gamma_pow: [T; D_EF],
    pub pre_claim: [T; D_EF],
    pub post_claim: [T; D_EF],
    pub is_initial: T,
    pub round_inverse: T,
    pub quotient: T,
    pub parity: T,
    pub subgroup_root: T,
}

#[derive(Clone, Debug)]
pub struct FinitePrefixTwoCosetWhirQueryAir {
    pub transcript_bus: TranscriptBus,
    pub exp_bits_len_bus: ExpBitsLenBus,
    pub right_shift_bus: RightShiftBus,
    pub query_bus_idx: BusIndex,
    pub verify_queries_bus_idx: BusIndex,
    pub verify_query_bus_idx: BusIndex,
    pub k: usize,
    /// m + 1 for the rate-half initial codeword.
    pub initial_log_domain_size: usize,
    pub num_rounds: usize,
}

impl BaseAir<F> for FinitePrefixTwoCosetWhirQueryAir {
    fn width(&self) -> usize {
        FinitePrefixTwoCosetWhirQueryCols::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for FinitePrefixTwoCosetWhirQueryAir {}
impl PartitionedBaseAir<F> for FinitePrefixTwoCosetWhirQueryAir {}

fn permutation_send<AB: InteractionBuilder>(
    bus_idx: BusIndex,
    builder: &mut AB,
    values: Vec<AB::Expr>,
    enabled: impl Into<AB::Expr>,
) {
    PermutationCheckBus::new(bus_idx).send(builder, values, enabled);
}

fn permutation_receive<AB: InteractionBuilder>(
    bus_idx: BusIndex,
    builder: &mut AB,
    values: Vec<AB::Expr>,
    enabled: impl Into<AB::Expr>,
) {
    PermutationCheckBus::new(bus_idx).receive(builder, values, enabled);
}

impl<AB> Air<AB> for FinitePrefixTwoCosetWhirQueryAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield:
        BinomiallyExtendable<{ D_EF }> + TwoAdicField,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("two-coset query row");
        let next_row = main.row_slice(1).expect("two-coset query next row");
        let local: &FinitePrefixTwoCosetWhirQueryCols<AB::Var> = (*local_row).borrow();
        let next: &FinitePrefixTwoCosetWhirQueryCols<AB::Var> = (*next_row).borrow();
        for flag in [
            local.is_enabled,
            local.is_first_in_proof,
            local.is_first_in_round,
            local.is_initial,
            local.parity,
        ] {
            builder.assert_bool(flag);
        }
        builder
            .when_first_row()
            .assert_eq(local.is_enabled, local.is_first_in_proof);
        builder
            .when(local.is_first_in_proof)
            .assert_one(local.is_first_in_round);
        builder
            .when(local.is_first_in_proof)
            .assert_zero(local.proof_idx);
        builder
            .when(local.is_first_in_proof)
            .assert_zero(local.whir_round);
        builder
            .when(local.is_first_in_round)
            .assert_zero(local.query_idx);
        let enabled: AB::Expr = local.is_enabled.into();
        builder
            .when(enabled.clone() * local.is_initial)
            .assert_zero(local.whir_round);
        builder
            .when(enabled.clone() * (AB::Expr::ONE - local.is_initial))
            .assert_one(local.whir_round * local.round_inverse);

        let same_proof = next.is_enabled - next.is_first_in_proof;
        let same_round = next.is_enabled - next.is_first_in_round;
        builder
            .when(same_proof.clone())
            .assert_eq(next.proof_idx, local.proof_idx);
        builder
            .when(same_round.clone())
            .assert_eq(next.whir_round, local.whir_round);
        builder
            .when(same_round.clone())
            .assert_eq(next.query_idx, local.query_idx + AB::F::ONE);
        builder
            .when(same_round.clone())
            .assert_eq(next.tidx, local.tidx + AB::F::ONE);
        builder
            .when(same_round.clone())
            .assert_eq(next.num_queries, local.num_queries);
        builder
            .when(same_round.clone())
            .assert_eq(next.omega, local.omega);
        let next_round = same_proof.clone() - same_round.clone();
        builder
            .when(next_round.clone())
            .assert_one(local.is_enabled);
        builder
            .when(next_round.clone())
            .assert_eq(next.whir_round, local.whir_round + AB::F::ONE);
        builder.when(next_round).assert_one(next.is_first_in_round);
        builder
            .when(enabled.clone() - same_round.clone())
            .assert_eq(local.query_idx + AB::F::ONE, local.num_queries);
        builder
            .when(enabled.clone() - same_proof.clone())
            .assert_eq(local.whir_round, AB::Expr::from_usize(self.num_rounds - 1));

        assert_ext_eq(
            &mut builder.when(local.is_first_in_round),
            local.gamma_pow,
            ext_field_multiply::<AB::Expr>(local.gamma, local.gamma),
        );
        assert_ext_eq(
            &mut builder.when(same_round.clone()),
            next.gamma,
            local.gamma,
        );
        assert_ext_eq(
            &mut builder.when(same_round.clone()),
            next.gamma_pow,
            ext_field_multiply::<AB::Expr>(local.gamma, local.gamma_pow),
        );
        assert_ext_eq(
            &mut builder.when(same_round.clone()),
            next.pre_claim,
            ext_field_add::<AB::Expr>(
                local.pre_claim,
                ext_field_multiply::<AB::Expr>(local.gamma_pow, local.yi),
            ),
        );
        assert_ext_eq(
            &mut builder.when(same_round.clone()),
            next.post_claim,
            local.post_claim,
        );
        assert_ext_eq(
            &mut builder.when(enabled.clone() - same_round.clone()),
            local.post_claim,
            ext_field_add::<AB::Expr>(
                local.pre_claim,
                ext_field_multiply::<AB::Expr>(local.gamma_pow, local.yi),
            ),
        );

        let mut verify_queries = vec![
            local.proof_idx.into(),
            local.tidx.into(),
            local.whir_round.into(),
            local.num_queries.into(),
            local.omega.into(),
        ];
        verify_queries.extend(local.gamma.map(Into::into));
        verify_queries.extend(local.pre_claim.map(Into::into));
        verify_queries.extend(local.post_claim.map(Into::into));
        permutation_receive(
            self.verify_queries_bus_idx,
            builder,
            verify_queries,
            local.is_first_in_round,
        );
        self.transcript_bus.sample(
            builder,
            local.proof_idx,
            local.tidx,
            local.sample,
            enabled.clone(),
        );

        let query_bits =
            AB::Expr::from_usize(self.initial_log_domain_size - self.k) - local.whir_round;
        self.right_shift_bus.lookup_key(
            builder,
            RightShiftMessage {
                input: local.sample.into(),
                shift_bits: AB::Expr::ONE,
                result: local.quotient.into(),
            },
            enabled.clone() * local.is_initial,
        );
        builder
            .when(enabled.clone() * local.is_initial)
            .assert_eq(local.sample, local.quotient * AB::F::TWO + local.parity);
        self.exp_bits_len_bus.lookup_key(
            builder,
            ExpBitsLenMessage {
                // In the physical `H union gH` layout `omega` is already the
                // generator of H.  There is no unavailable 2^(m+1)-root to
                // square, including at BabyBear's maximum m = 27 geometry.
                base: local.omega.into(),
                bit_src: local.quotient.into(),
                num_bits: query_bits.clone() - AB::Expr::ONE,
                result: local.subgroup_root.into(),
            },
            enabled.clone() * local.is_initial,
        );
        // The shared ExpBitsLen table also owns the two right-shift rows used
        // only by the two-coset quotient and the initial Merkle path. Their
        // exponentiation side is the canonical no-op 1^0 = 1; consuming both
        // keys here prevents an unconstrained table multiplicity.
        for _ in 0..2 {
            self.exp_bits_len_bus.lookup_key(
                builder,
                ExpBitsLenMessage {
                    base: AB::Expr::ONE,
                    bit_src: local.sample.into(),
                    num_bits: AB::Expr::ZERO,
                    result: AB::Expr::ONE,
                },
                enabled.clone() * local.is_initial,
            );
        }
        let coset = AB::Expr::ONE
            + local.parity * (AB::Expr::from_u32(F::GENERATOR.as_canonical_u32()) - AB::Expr::ONE);
        builder
            .when(enabled.clone() * local.is_initial)
            .assert_eq(local.zi_root, local.subgroup_root * coset);
        self.exp_bits_len_bus.lookup_key(
            builder,
            ExpBitsLenMessage {
                base: local.omega.into(),
                bit_src: local.sample.into(),
                num_bits: query_bits,
                result: local.zi_root.into(),
            },
            enabled.clone() * (AB::Expr::ONE - local.is_initial),
        );

        let mut query = vec![
            local.proof_idx.into(),
            local.whir_round.into(),
            AB::Expr::from(local.query_idx) + AB::Expr::ONE,
            local.zi.into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        permutation_send(
            self.query_bus_idx,
            builder,
            core::mem::take(&mut query),
            enabled.clone(),
        );
        let mut verify_query = vec![
            local.proof_idx.into(),
            local.whir_round.into(),
            local.query_idx.into(),
            local.sample.into(),
            local.zi_root.into(),
            local.zi.into(),
        ];
        verify_query.extend(local.yi.map(Into::into));
        permutation_send(self.verify_query_bus_idx, builder, verify_query, enabled);
    }
}

/// Internal bus order allocated by MultiConstraintWhirModule::new.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinitePrefixWhirInternalBusIndices {
    pub sumcheck: BusIndex,
    pub alpha: BusIndex,
    pub gamma: BusIndex,
    pub query: BusIndex,
    pub verify_queries: BusIndex,
    pub verify_query: BusIndex,
}

impl FinitePrefixWhirInternalBusIndices {
    #[must_use]
    pub const fn from_first_allocated(first: BusIndex) -> Self {
        Self {
            sumcheck: first,
            alpha: first + 1,
            gamma: first + 2,
            query: first + 3,
            verify_queries: first + 4,
            verify_query: first + 5,
        }
    }
}

/// Native SDK first-round point map for the physical interleaving.
pub fn finite_prefix_initial_query_root(
    message_log: usize,
    k: usize,
    query_index: usize,
) -> Result<F, FinitePrefixVerifyError> {
    let query_bits = message_log
        .checked_add(1)
        .and_then(|value| value.checked_sub(k))
        .ok_or(FinitePrefixVerifyError::Profile("initial query dimension"))?;
    if query_index >= 1usize << query_bits {
        return Err(FinitePrefixVerifyError::ProofShape("query index"));
    }
    let mut root = F::two_adic_generator(message_log).exp_u64((query_index >> 1) as u64);
    if query_index & 1 == 1 {
        root *= F::GENERATOR;
    }
    Ok(root)
}

#[derive(Clone, Debug)]
struct FinitePrefixTwoCosetDerived {
    round_offsets: Vec<usize>,
    zi_roots: Vec<F>,
    zis: Vec<F>,
    yis: Vec<EF>,
    fold_records: Vec<FoldRecord>,
    initial_claims: Vec<EF>,
    post_sumcheck_claims: Vec<EF>,
    pre_query_claims: Vec<EF>,
    final_aggregate: EF,
    final_claim: EF,
}

impl FinitePrefixTwoCosetDerived {
    fn query(&self, round: usize, query: usize) -> usize {
        self.round_offsets[round] + query
    }
}

fn prefix_binary_k_fold(
    mut values: Vec<EF>,
    alphas: &[EF],
    base_coset_shift: F,
    whir_round: usize,
    query_idx: usize,
    records: &mut Vec<FoldRecord>,
) -> EF {
    debug_assert_eq!(values.len(), 1usize << alphas.len());
    let n = values.len();
    let k = alphas.len();
    let omega = F::two_adic_generator(k);
    let omega_inv = omega.inverse();
    let twiddles = omega.powers().take(1 << (k - 1)).collect();
    let inverse_twiddles = omega_inv.powers().take(1 << (k - 1)).collect();
    let mut shift = base_coset_shift;
    let mut shift_inverse = base_coset_shift.inverse();
    for (height, &alpha) in alphas.iter().enumerate() {
        let half = n >> (height + 1);
        let (low, high) = values.split_at_mut(half);
        for index in 0..half {
            let twiddle = twiddles[index << height];
            let eval_point = twiddle * shift;
            let eval_point_inverse = inverse_twiddles[index << height] * shift_inverse;
            let left = low[index];
            let right = high[index];
            let value = left
                + (alpha - EF::from(eval_point))
                    * (left - right)
                    * EF::from(eval_point_inverse * F::TWO.inverse());
            records.push(FoldRecord::new(
                whir_round,
                query_idx,
                twiddle,
                shift,
                half,
                index,
                height + 1,
                left,
                right,
                value,
                alpha,
            ));
            low[index] = value;
        }
        shift *= shift;
        shift_inverse *= shift_inverse;
    }
    values[0]
}

fn derive_two_coset_whir(
    params: &SystemParams,
    profile: &FinitePrefixVerifierProfile,
    input: &PreparedFinitePrefixCircuitProof,
) -> Result<FinitePrefixTwoCosetDerived, FinitePrefixVerifyError> {
    let proof = &input.proof.whir;
    let preflight = &input.multi_preflight;
    let multi_profile = profile.multi_whir_profile()?;
    let multi_derived = derive_multi_constraint_whir_data(
        &multi_profile,
        &input.multi_statement,
        preflight.mu,
        &preflight.alphas,
        &proof.final_poly,
    )
    .map_err(|_| FinitePrefixVerifyError::ProofShape("derived WHIR statement"))?;
    let rounds = params.num_whir_rounds();
    let k = params.k_whir();
    let message_log = profile.layout.log_message_len as usize;
    let mut round_offsets = Vec::with_capacity(rounds + 1);
    round_offsets.push(0);
    for round in &params.whir.rounds {
        round_offsets.push(round_offsets.last().copied().unwrap() + round.num_queries);
    }
    let mut zi_roots = Vec::with_capacity(*round_offsets.last().unwrap());
    let mut zis = Vec::with_capacity(*round_offsets.last().unwrap());
    let mut yis = Vec::with_capacity(*round_offsets.last().unwrap());
    let mut fold_records =
        Vec::with_capacity(round_offsets.last().copied().unwrap_or(0) * ((1usize << k) - 1));
    let total_width = 2 * profile.source_count;
    let mu_powers = preflight.mu.powers().take(total_width).collect();
    let mut initial_claims = Vec::with_capacity(rounds + 1);
    let mut post_sumcheck_claims = Vec::with_capacity(params.num_whir_sumcheck_rounds());
    let mut pre_query_claims = Vec::with_capacity(rounds);
    let mut claim = multi_derived.initial_target;

    for round in 0..rounds {
        initial_claims.push(claim);
        let alphas = &preflight.alphas[round * k..(round + 1) * k];
        for subround in 0..k {
            let index = round * k + subround;
            let [at_one, at_two] = proof.whir_sumcheck_polys[index];
            claim = interpolate_quadratic_at_012(
                &[claim - at_one, at_one, at_two],
                preflight.alphas[index],
            );
            post_sumcheck_claims.push(claim);
        }
        let gamma = preflight.gammas[round];
        if round + 1 < rounds {
            claim += gamma * proof.ood_values[round];
        }
        pre_query_claims.push(claim);
        let query_start = round_offsets[round];
        let query_end = round_offsets[round + 1];
        let log_domain = message_log + 1 - round;
        for (query_idx, &sample) in preflight.queries[query_start..query_end].iter().enumerate() {
            let query_bits = log_domain - k;
            let index = sample.as_canonical_u32() as usize & ((1usize << query_bits) - 1);
            let root = if round == 0 {
                finite_prefix_initial_query_root(message_log, k, index)?
            } else {
                F::two_adic_generator(log_domain).exp_u64(index as u64)
            };
            let zi = root.exp_power_of_2(k);
            let opened = if round == 0 {
                let mut values = EF::zero_vec(1usize << k);
                let mut global_column = 0usize;
                for commitment in &proof.initial_round_opened_rows {
                    let rows = &commitment[query_idx];
                    for column in 0..rows[0].len() {
                        let coefficient = mu_powers[global_column];
                        global_column += 1;
                        for (value, row) in values.iter_mut().zip(rows) {
                            *value += coefficient * row[column];
                        }
                    }
                }
                if global_column != total_width {
                    return Err(FinitePrefixVerifyError::ProofShape("initial columns"));
                }
                values
            } else {
                proof.codeword_opened_values[round - 1][query_idx].clone()
            };
            if opened.len() != 1usize << k {
                return Err(FinitePrefixVerifyError::ProofShape("opened WHIR leaf"));
            }
            let record_start = fold_records.len();
            let yi =
                prefix_binary_k_fold(opened, alphas, root, round, query_idx, &mut fold_records);
            for record in &mut fold_records[record_start..] {
                record.set_final_values(zi, yi);
            }
            zi_roots.push(root);
            zis.push(zi);
            yis.push(yi);
            claim += gamma.exp_u64(query_idx as u64 + 2) * yi;
        }
    }
    initial_claims.push(claim);
    Ok(FinitePrefixTwoCosetDerived {
        round_offsets,
        zi_roots,
        zis,
        yis,
        fold_records,
        initial_claims,
        post_sumcheck_claims,
        pre_query_claims,
        final_aggregate: multi_derived.final_weighted_evaluation,
        final_claim: claim,
    })
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
struct PrefixInitialOpenedValuesCols<T> {
    proof_idx: T,
    query_idx: T,
    commit_idx: T,
    coset_idx: T,
    col_chunk_idx: T,
    is_first_in_proof: T,
    is_first_in_query: T,
    is_first_in_commit: T,
    is_first_in_coset: T,
    flags: [T; CHUNK],
    codeword_value_acc: [T; D_EF],
    codeword_value_next_acc: [T; D_EF],
    mu_pows_even_clamped: [[T; D_EF]; CHUNK / 2],
    mu_pow_last_clamped: [T; D_EF],
    mu: [T; D_EF],
    pre_state: [T; POSEIDON2_WIDTH],
    post_state: [T; POSEIDON2_WIDTH],
    twiddle: T,
    zi_root: T,
    zi: T,
    yi: [T; D_EF],
    merkle_idx_bit_src: T,
}

// Field-for-field mirror of recursion's private-module `WhirRoundCols`.
// The AIR itself remains the public, reused `WhirRoundAir`; this mirror is
// used only to replace the three claim witnesses affected by the two-coset
// first query map without widening recursion's crate API.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
struct PrefixWhirRoundCols<T, const ENC_WIDTH: usize> {
    is_enabled: T,
    proof_idx: T,
    whir_round: T,
    is_first_in_proof: T,
    tidx: T,
    num_queries: T,
    omega: T,
    z0: [T; D_EF],
    y0: [T; D_EF],
    commit: [T; DIGEST_SIZE],
    final_poly_mle_eval: [T; D_EF],
    query_pow_witness: T,
    query_pow_sample: T,
    gamma: [T; D_EF],
    claim: [T; D_EF],
    next_claim: [T; D_EF],
    post_sumcheck_claim: [T; D_EF],
    whir_round_enc: [T; ENC_WIDTH],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
struct PrefixFinalPolyQueryEvalCols<T> {
    is_enabled: T,
    proof_idx: T,
    whir_round: T,
    query_idx: T,
    phase_idx: T,
    eval_idx: T,
    is_first_in_proof: T,
    is_first_in_round: T,
    is_first_in_query: T,
    is_first_in_phase: T,
    is_last_round: T,
    is_query_zero: T,
    query_pow: [T; D_EF],
    alpha: [T; D_EF],
    gamma: [T; D_EF],
    gamma_pow: [T; D_EF],
    final_poly_coeff: [T; D_EF],
    final_value_acc: [T; D_EF],
    gamma_eq_acc: [T; D_EF],
    horner_acc: [T; D_EF],
    do_carry: T,
}

fn copy_ext_array(output: &mut [F; D_EF], value: EF) {
    output.copy_from_slice(value.as_basis_coefficients_slice());
}

fn generate_two_coset_query_trace(
    params: &SystemParams,
    input: &PreparedFinitePrefixCircuitProof,
    derived: &FinitePrefixTwoCosetDerived,
    height: usize,
) -> Result<RowMajorMatrix<F>, FinitePrefixVerifyError> {
    let valid_rows = derived.yis.len();
    if height < valid_rows {
        return Err(FinitePrefixVerifyError::TraceHeight);
    }
    let width = FinitePrefixTwoCosetWhirQueryCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    let initial_log_domain_size = input.proof.statement.layout.log_message_len as usize + 1;
    for round in 0..params.num_whir_rounds() {
        let start = derived.round_offsets[round];
        let end = derived.round_offsets[round + 1];
        let num_queries = end - start;
        let gamma = input.multi_preflight.gammas[round];
        let mut claim = derived.pre_query_claims[round];
        for query_idx in 0..num_queries {
            let row_idx = start + query_idx;
            let row = &mut values[row_idx * width..(row_idx + 1) * width];
            let cols: &mut FinitePrefixTwoCosetWhirQueryCols<F> = row.borrow_mut();
            let sample = input.multi_preflight.queries[row_idx];
            let query_bits = initial_log_domain_size - params.k_whir() - round;
            let index = sample.as_canonical_u32() as usize & ((1usize << query_bits) - 1);
            cols.is_enabled = F::ONE;
            cols.proof_idx = F::ZERO;
            cols.whir_round = F::from_usize(round);
            cols.query_idx = F::from_usize(query_idx);
            cols.is_first_in_proof = F::from_bool(round == 0 && query_idx == 0);
            cols.is_first_in_round = F::from_bool(query_idx == 0);
            cols.tidx =
                F::from_usize(input.multi_preflight.query_tidx_per_round[round] + query_idx);
            cols.num_queries = F::from_usize(num_queries);
            let omega_log = initial_log_domain_size - if round == 0 { 1 } else { round };
            cols.omega = F::two_adic_generator(omega_log);
            cols.sample = sample;
            cols.zi_root = derived.zi_roots[row_idx];
            cols.zi = derived.zis[row_idx];
            copy_ext_array(&mut cols.yi, derived.yis[row_idx]);
            copy_ext_array(&mut cols.gamma, gamma);
            let gamma_power = gamma.exp_u64(query_idx as u64 + 2);
            copy_ext_array(&mut cols.gamma_pow, gamma_power);
            copy_ext_array(&mut cols.pre_claim, claim);
            copy_ext_array(&mut cols.post_claim, derived.initial_claims[round + 1]);
            cols.is_initial = F::from_bool(round == 0);
            cols.round_inverse = if round == 0 {
                F::ZERO
            } else {
                F::from_usize(round).inverse()
            };
            if round == 0 {
                let sample_u32 = sample.as_canonical_u32();
                cols.quotient = F::from_u32(sample_u32 >> 1);
                cols.parity = F::from_u32(sample_u32 & 1);
                cols.subgroup_root =
                    F::two_adic_generator(input.proof.statement.layout.log_message_len as usize)
                        .exp_u64((index >> 1) as u64);
            }
            claim += gamma_power * derived.yis[row_idx];
        }
    }
    Ok(RowMajorMatrix::new(values, width))
}

fn patch_initial_opened_values_trace(
    trace: &mut RowMajorMatrix<F>,
    derived: &FinitePrefixTwoCosetDerived,
) -> Result<(), FinitePrefixVerifyError> {
    let width = PrefixInitialOpenedValuesCols::<F>::width();
    if trace.width() != width {
        return Err(FinitePrefixVerifyError::ProofShape(
            "recursive initial-opening layout",
        ));
    }
    for row in trace.values.chunks_exact_mut(width) {
        let cols: &mut PrefixInitialOpenedValuesCols<F> = row.borrow_mut();
        if cols.flags[0] == F::ZERO {
            continue;
        }
        let query = cols.query_idx.as_canonical_u32() as usize;
        let index = derived.query(0, query);
        cols.zi_root = derived.zi_roots[index];
        cols.zi = derived.zis[index];
        copy_ext_array(&mut cols.yi, derived.yis[index]);
    }
    Ok(())
}

fn generate_two_coset_folding_trace(
    derived: &FinitePrefixTwoCosetDerived,
    height: usize,
) -> Result<RowMajorMatrix<F>, FinitePrefixVerifyError> {
    if height < derived.fold_records.len() {
        return Err(FinitePrefixVerifyError::TraceHeight);
    }
    let width = WhirFoldingCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    for (row, record) in values.chunks_exact_mut(width).zip(&derived.fold_records) {
        let cols: &mut WhirFoldingCols<F> = row.borrow_mut();
        cols.is_valid = F::ONE;
        cols.proof_idx = F::ZERO;
        cols.whir_round = F::from_u32(record.whir_round);
        cols.query_idx = F::from_u32(record.query_idx);
        cols.is_root = F::from_bool(record.coset_size == 1);
        cols.coset_shift = record.coset_shift;
        cols.coset_idx = F::from_u32(record.coset_idx);
        cols.height = F::from_u32(record.height);
        cols.twiddle = record.twiddle;
        cols.coset_size = F::from_u32(record.coset_size);
        cols.z_final = record.z_final;
        copy_ext_array(&mut cols.value, record.value);
        copy_ext_array(&mut cols.left_value, record.left_value);
        copy_ext_array(&mut cols.right_value, record.right_value);
        copy_ext_array(&mut cols.y_final, record.y_final);
        copy_ext_array(&mut cols.alpha, record.alpha);
    }
    Ok(RowMajorMatrix::new(values, width))
}

fn patch_whir_round_trace_impl<const ENC_WIDTH: usize>(
    trace: &mut RowMajorMatrix<F>,
    params: &SystemParams,
    derived: &FinitePrefixTwoCosetDerived,
) -> Result<(), FinitePrefixVerifyError> {
    let width = PrefixWhirRoundCols::<F, ENC_WIDTH>::width();
    if trace.width() != width || trace.height() < params.num_whir_rounds() {
        return Err(FinitePrefixVerifyError::ProofShape(
            "recursive WHIR-round layout",
        ));
    }
    let k = params.k_whir();
    for round in 0..params.num_whir_rounds() {
        let row = &mut trace.values[round * width..(round + 1) * width];
        let cols: &mut PrefixWhirRoundCols<F, ENC_WIDTH> = row.borrow_mut();
        copy_ext_array(&mut cols.claim, derived.initial_claims[round]);
        copy_ext_array(&mut cols.next_claim, derived.initial_claims[round + 1]);
        copy_ext_array(
            &mut cols.post_sumcheck_claim,
            derived.post_sumcheck_claims[(round + 1) * k - 1],
        );
        copy_ext_array(&mut cols.final_poly_mle_eval, derived.final_aggregate);
    }
    Ok(())
}

fn patch_whir_round_trace(
    trace: &mut RowMajorMatrix<F>,
    params: &SystemParams,
    derived: &FinitePrefixTwoCosetDerived,
) -> Result<(), FinitePrefixVerifyError> {
    let encoder = Encoder::new(params.num_whir_rounds().max(2), 2, false);
    match encoder.width() {
        1 => patch_whir_round_trace_impl::<1>(trace, params, derived),
        2 => patch_whir_round_trace_impl::<2>(trace, params, derived),
        3 => patch_whir_round_trace_impl::<3>(trace, params, derived),
        _ => Err(FinitePrefixVerifyError::ProofShape("WHIR encoder width")),
    }
}

fn patch_multi_sumcheck_trace(
    trace: &mut RowMajorMatrix<F>,
    params: &SystemParams,
    derived: &FinitePrefixTwoCosetDerived,
) -> Result<(), FinitePrefixVerifyError> {
    let width = MultiConstraintSumcheckCols::<F>::width();
    let rounds = params.num_whir_sumcheck_rounds();
    if trace.width() != width || trace.height() < rounds {
        return Err(FinitePrefixVerifyError::ProofShape(
            "recursive sumcheck layout",
        ));
    }
    let k = params.k_whir();
    for index in 0..rounds {
        let row = &mut trace.values[index * width..(index + 1) * width];
        let cols: &mut MultiConstraintSumcheckCols<F> = row.borrow_mut();
        let round = index / k;
        let subround = index % k;
        let pre_claim = if subround == 0 {
            derived.initial_claims[round]
        } else {
            derived.post_sumcheck_claims[index - 1]
        };
        copy_ext_array(&mut cols.pre_claim, pre_claim);
        copy_ext_array(
            &mut cols.post_group_claim,
            derived.post_sumcheck_claims[(round + 1) * k - 1],
        );
    }
    Ok(())
}

fn generate_two_coset_final_query_trace(
    params: &SystemParams,
    input: &PreparedFinitePrefixCircuitProof,
    derived: &FinitePrefixTwoCosetDerived,
    height: usize,
) -> Result<RowMajorMatrix<F>, FinitePrefixVerifyError> {
    let k = params.k_whir();
    let rounds = params.num_whir_rounds();
    let final_poly_len = 1usize << params.log_final_poly_len();
    let mut valid_rows = 0usize;
    for round in 0..rounds {
        valid_rows += (params.whir.rounds[round].num_queries + 1)
            * (k * (rounds - round - 1) + final_poly_len);
    }
    if height < valid_rows {
        return Err(FinitePrefixVerifyError::TraceHeight);
    }
    let width = PrefixFinalPolyQueryEvalCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    let mut row_idx = 0usize;
    let mut final_value_acc = EF::ZERO;
    for round in 0..rounds {
        let eq_len = k * (rounds - round - 1);
        let gamma = input.multi_preflight.gammas[round];
        let mut gamma_power = gamma;
        let query_count = params.whir.rounds[round].num_queries + 1;
        for query_idx in 0..query_count {
            let mut gamma_eq_acc = gamma_power;
            let mut horner_acc = EF::ZERO;
            let mut query_power = if query_idx == 0 {
                if round + 1 < rounds {
                    input.multi_preflight.z0s[round]
                } else {
                    EF::ZERO
                }
            } else {
                EF::from(derived.zis[derived.query(round, query_idx - 1)])
            };
            for eval_idx in 0..eq_len {
                let alpha = input.multi_preflight.alphas[(round + 1) * k + eval_idx];
                fill_final_query_row(
                    &mut values,
                    width,
                    row_idx,
                    rounds,
                    final_poly_len,
                    params.whir.rounds[round].num_queries,
                    round,
                    query_idx,
                    0,
                    eval_idx,
                    eq_len,
                    alpha,
                    query_power,
                    gamma,
                    gamma_power,
                    EF::ZERO,
                    final_value_acc,
                    gamma_eq_acc,
                    horner_acc,
                );
                row_idx += 1;
                gamma_eq_acc *= EF::ONE - query_power - alpha + query_power * alpha.double();
                query_power *= query_power;
            }
            for (eval_idx, &coefficient) in input.proof.whir.final_poly.iter().rev().enumerate() {
                fill_final_query_row(
                    &mut values,
                    width,
                    row_idx,
                    rounds,
                    final_poly_len,
                    params.whir.rounds[round].num_queries,
                    round,
                    query_idx,
                    1,
                    eval_idx,
                    eq_len,
                    EF::ZERO,
                    query_power,
                    gamma,
                    gamma_power,
                    coefficient,
                    final_value_acc,
                    gamma_eq_acc,
                    horner_acc,
                );
                row_idx += 1;
                horner_acc = horner_acc * query_power + coefficient;
            }
            if query_idx != 0 || round + 1 != rounds {
                final_value_acc += gamma_eq_acc * horner_acc;
            }
            gamma_power *= gamma;
        }
    }
    debug_assert_eq!(row_idx, valid_rows);
    Ok(RowMajorMatrix::new(values, width))
}

#[allow(clippy::too_many_arguments)]
fn fill_final_query_row(
    values: &mut [F],
    width: usize,
    row_idx: usize,
    rounds: usize,
    final_poly_len: usize,
    in_domain_queries: usize,
    round: usize,
    query_idx: usize,
    phase: usize,
    eval_idx: usize,
    eq_len: usize,
    alpha: EF,
    query_power: EF,
    gamma: EF,
    gamma_power: EF,
    coefficient: EF,
    final_value_acc: EF,
    gamma_eq_acc: EF,
    horner_acc: EF,
) {
    let row = &mut values[row_idx * width..(row_idx + 1) * width];
    let cols: &mut PrefixFinalPolyQueryEvalCols<F> = row.borrow_mut();
    let first_phase = eval_idx == 0;
    let first_query = first_phase && (phase == 0 || eq_len == 0);
    let first_round = first_query && query_idx == 0;
    let first_proof = first_round && round == 0;
    let same_phase = if phase == 0 {
        eval_idx + 1 < eq_len
    } else {
        eval_idx + 1 < final_poly_len
    };
    let same_query = same_phase || phase == 0;
    let same_round = same_query || query_idx < in_domain_queries;
    let same_proof = same_round || round + 1 < rounds;
    let is_q0_last = query_idx == 0 && round + 1 == rounds;
    cols.is_enabled = F::ONE;
    cols.proof_idx = F::ZERO;
    cols.whir_round = F::from_usize(round);
    cols.query_idx = F::from_usize(query_idx);
    cols.phase_idx = F::from_usize(phase);
    cols.eval_idx = F::from_usize(eval_idx);
    cols.is_first_in_proof = F::from_bool(first_proof);
    cols.is_first_in_round = F::from_bool(first_round);
    cols.is_first_in_query = F::from_bool(first_query);
    cols.is_first_in_phase = F::from_bool(first_phase);
    cols.is_last_round = F::from_bool(round + 1 == rounds);
    cols.is_query_zero = F::from_bool(query_idx == 0);
    cols.do_carry = F::from_bool(!same_query && same_proof && !is_q0_last);
    copy_ext_array(&mut cols.query_pow, query_power);
    copy_ext_array(&mut cols.alpha, alpha);
    copy_ext_array(&mut cols.gamma, gamma);
    copy_ext_array(&mut cols.gamma_pow, gamma_power);
    copy_ext_array(&mut cols.final_poly_coeff, coefficient);
    copy_ext_array(&mut cols.final_value_acc, final_value_acc);
    copy_ext_array(&mut cols.gamma_eq_acc, gamma_eq_acc);
    copy_ext_array(&mut cols.horner_acc, horner_acc);
}

fn enqueue_two_coset_exp_requests(
    params: &SystemParams,
    input: &PreparedFinitePrefixCircuitProof,
    generator: &ExpBitsLenCpuTraceGenerator,
) {
    if params.whir.mu_pow_bits > 0 {
        generator.add_request(
            F::GENERATOR,
            input.multi_preflight.mu_pow_sample,
            params.whir.mu_pow_bits,
        );
    }
    if params.whir.folding_pow_bits > 0 {
        generator.add_requests(
            input
                .multi_preflight
                .folding_pow_samples
                .iter()
                .copied()
                .map(|sample| (F::GENERATOR, sample, params.whir.folding_pow_bits)),
        );
    }
    if params.whir.query_phase_pow_bits > 0 {
        generator.add_requests(
            input
                .multi_preflight
                .query_pow_samples
                .iter()
                .copied()
                .map(|sample| (F::GENERATOR, sample, params.whir.query_phase_pow_bits)),
        );
    }
    let initial_log_domain = input.proof.statement.layout.log_message_len as usize + 1;
    for round in 0..params.num_whir_rounds() {
        let query_offset = params.whir.rounds[..round]
            .iter()
            .map(|config| config.num_queries)
            .sum::<usize>();
        let queries = &input.multi_preflight.queries
            [query_offset..query_offset + params.whir.rounds[round].num_queries];
        let log_domain = initial_log_domain - round;
        if round == 0 {
            for &sample in queries {
                let query_bits = log_domain - params.k_whir();
                let quotient = F::from_u32(sample.as_canonical_u32() >> 1);
                generator.add_request(
                    F::two_adic_generator(input.proof.statement.layout.log_message_len as usize),
                    quotient,
                    query_bits - 1,
                );
                generator.add_requests_with_shift([
                    (F::ONE, sample, 0, 1, 1),
                    (
                        F::ONE,
                        sample,
                        0,
                        initial_log_domain - params.k_whir() + 1,
                        2,
                    ),
                ]);
            }
        } else {
            generator.add_requests_with_shift(queries.iter().copied().map(|sample| {
                (
                    F::two_adic_generator(log_domain),
                    sample,
                    log_domain - params.k_whir(),
                    initial_log_domain - params.k_whir() + 1 - round,
                    1,
                )
            }));
        }
    }
}

/// Complete finite-prefix PCS verifier owner, excluding the wrapper-shared
/// Transcript/Merkle/Poseidon and ExpBitsLen physical tables.
///
/// AIR order is `Authority` followed by the stable fourteen AIRs of
/// [`MultiConstraintWhirModule`]. The ordinary query AIR is replaced in place
/// by [`FinitePrefixTwoCosetWhirQueryAir`]; all later-round AIR identities stay
/// unchanged.
pub struct FinitePrefixVerifierModule {
    params: SystemParams,
    profile: FinitePrefixVerifierProfile,
    authority: Arc<FinitePrefixAuthorityAir>,
    multi_whir: MultiConstraintWhirModule,
    query_air: Arc<FinitePrefixTwoCosetWhirQueryAir>,
    buses: FinitePrefixVerifierBuses,
}

impl FinitePrefixVerifierModule {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        profile: FinitePrefixVerifierProfile,
        authority_bus: OrderedManifestPrefixAuthorityBus,
        terminal_checkpoint_bus: CertifiedTranscriptCheckpointBus,
        terminal_checkpoint_kind: usize,
        bus_indices: &mut BusIndexManager,
        bus_inventory: BusInventory,
    ) -> Result<Self, FinitePrefixVerifyError> {
        profile.validate(&child_vk.inner.params)?;
        let statement = MultiConstraintStatementBuses::new(
            bus_indices.new_bus_idx(),
            bus_indices.new_bus_idx(),
            bus_indices.new_bus_idx(),
        );
        let initial_commitment =
            MultiConstraintInitialCommitmentBus::new(bus_indices.new_bus_idx());
        let first_internal = bus_indices.next_bus_idx();
        let multi_whir = MultiConstraintWhirModule::new_with_external_coefficient_two_coset_query(
            child_vk,
            profile.multi_whir_profile()?,
            statement,
            initial_commitment,
            terminal_checkpoint_bus,
            terminal_checkpoint_kind,
            profile.class_index,
            bus_indices,
            bus_inventory.clone(),
        )?;
        let internal = FinitePrefixWhirInternalBusIndices::from_first_allocated(first_internal);
        let buses = FinitePrefixVerifierBuses {
            transcript: bus_inventory.transcript_bus,
            statement,
            initial_commitment,
            whir_completion: multi_whir.completion_bus(),
            authority: authority_bus,
        };
        let authority = Arc::new(FinitePrefixAuthorityAir::new(
            &child_vk.inner.params,
            profile.clone(),
            buses,
        )?);
        let query_air = Arc::new(FinitePrefixTwoCosetWhirQueryAir {
            transcript_bus: bus_inventory.transcript_bus,
            exp_bits_len_bus: bus_inventory.exp_bits_len_bus,
            right_shift_bus: bus_inventory.right_shift_bus,
            query_bus_idx: internal.query,
            verify_queries_bus_idx: internal.verify_queries,
            verify_query_bus_idx: internal.verify_query,
            k: child_vk.inner.params.k_whir(),
            initial_log_domain_size: profile.layout.log_message_len as usize + 1,
            num_rounds: child_vk.inner.params.num_whir_rounds(),
        });
        Ok(Self {
            params: child_vk.inner.params.clone(),
            profile,
            authority,
            multi_whir,
            query_air,
            buses,
        })
    }

    #[must_use]
    pub const fn buses(&self) -> FinitePrefixVerifierBuses {
        self.buses
    }

    #[must_use]
    pub fn profile(&self) -> &FinitePrefixVerifierProfile {
        &self.profile
    }

    /// Generate the concrete authority and WHIR traces for one invocation.
    /// The returned prepared carrier must also be passed to the wrapper's
    /// existing shared TranscriptModule so transcript and Merkle interactions
    /// are balanced by the ordinary recursive owners.
    pub fn generate_air_contexts<SC: StarkProtocolConfig<F = F>>(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proof: &FinitePrefixCircuitProof,
        exp_bits_len: &ExpBitsLenCpuTraceGenerator,
        required_multi_heights: Option<&[usize]>,
    ) -> Result<FinitePrefixCircuitTraceBundle<SC>, FinitePrefixVerifyError> {
        if child_vk.inner.params != self.params {
            return Err(FinitePrefixVerifyError::Profile("child verifying key"));
        }
        let input = prepare_finite_prefix_circuit_proof(&self.params, &self.profile, proof)?;
        let prepared_whir =
            self.multi_whir
                .prepare_direct_carriers(&[MultiConstraintWhirDirectCarrier {
                    whir_proof: &input.proof.whir,
                    initial_commitments: &input.initial_commitments,
                    statement: &input.multi_statement,
                    multi_preflight: &input.multi_preflight,
                    transcript: &input.transcript,
                    terminal_checkpoint: input.terminal_checkpoint,
                }])?;
        let derived = derive_two_coset_whir(&self.params, &self.profile, &input)?;

        // Generate the stable ordinary contexts using an isolated lookup
        // recorder. Every first-domain-dependent matrix is replaced below;
        // only the exact two-coset requests enter the enclosing table.
        let scratch_exp = ExpBitsLenCpuTraceGenerator::default();
        let query_height = required_multi_heights
            .map(|heights| heights[MultiConstraintWhirAir::Query as usize])
            .unwrap_or_else(|| derived.yis.len().next_power_of_two());
        let query_trace =
            generate_two_coset_query_trace(&self.params, &input, &derived, query_height)?;
        let mut inner = self
            .multi_whir
            .generate_air_contexts_with_external_two_coset_query::<SC>(
                child_vk,
                &prepared_whir,
                &scratch_exp,
                required_multi_heights,
                query_trace,
            )?;
        patch_whir_round_trace(
            &mut inner[MultiConstraintWhirAir::WhirRound as usize].common_main,
            &self.params,
            &derived,
        )?;
        patch_initial_opened_values_trace(
            &mut inner[MultiConstraintWhirAir::InitialOpenedValues as usize].common_main,
            &derived,
        )?;
        let folding_height = inner[MultiConstraintWhirAir::Folding as usize]
            .common_main
            .height();
        inner[MultiConstraintWhirAir::Folding as usize].common_main =
            generate_two_coset_folding_trace(&derived, folding_height)?;
        let final_query_height = inner[MultiConstraintWhirAir::FinalPolyQueryEval as usize]
            .common_main
            .height();
        inner[MultiConstraintWhirAir::FinalPolyQueryEval as usize].common_main =
            generate_two_coset_final_query_trace(
                &self.params,
                &input,
                &derived,
                final_query_height,
            )?;
        patch_multi_sumcheck_trace(
            &mut inner[MultiConstraintWhirAir::Sumcheck as usize].common_main,
            &self.params,
            &derived,
        )?;

        let completion = MultiConstraintWhirCompletionMessage {
            proof_idx: F::ZERO,
            class_index: F::from_usize(self.profile.class_index),
            end_tidx: F::from_usize(input.terminal_checkpoint.end_tidx),
            sample_count: F::from_usize(input.terminal_checkpoint.sample_count),
            state: input.terminal_checkpoint.state,
            final_aggregate: derived
                .final_aggregate
                .as_basis_coefficients_slice()
                .try_into()
                .expect("EF basis width"),
            final_claim: derived
                .final_claim
                .as_basis_coefficients_slice()
                .try_into()
                .expect("EF basis width"),
        };
        let completion_height = inner[MultiConstraintWhirAir::Completion as usize]
            .common_main
            .height();
        inner[MultiConstraintWhirAir::Completion as usize].common_main =
            generate_multi_constraint_completion_trace(
                &[input.terminal_checkpoint],
                &[derived.final_aggregate],
                &[derived.final_claim],
                self.profile.class_index,
                Some(completion_height),
            )
            .map_err(|_| FinitePrefixVerifyError::ProofShape("WHIR completion"))?
            .ok_or(FinitePrefixVerifyError::TraceHeight)?;

        enqueue_two_coset_exp_requests(&self.params, &input, exp_bits_len);
        let authority =
            AirProvingContext::simple_no_pis(input.authority_trace(&self.profile, &completion)?);
        let mut contexts = Vec::with_capacity(1 + inner.len());
        contexts.push(authority);
        contexts.extend(inner);
        Ok(FinitePrefixCircuitTraceBundle {
            contexts,
            prepared_whir,
            replayed: input,
        })
    }
}

impl AirModule for FinitePrefixVerifierModule {
    fn num_airs(&self) -> usize {
        1 + MultiConstraintWhirAir::COUNT
    }

    fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        let mut inner = self.multi_whir.airs::<SC>();
        inner[MultiConstraintWhirAir::Query as usize] = self.query_air.clone();
        let mut airs = Vec::with_capacity(1 + inner.len());
        airs.push(self.authority.clone() as AirRef<SC>);
        airs.extend(inner);
        airs
    }
}

pub struct FinitePrefixCircuitTraceBundle<SC: StarkProtocolConfig<F = F>> {
    pub contexts: Vec<AirProvingContext<CpuBackend<SC>>>,
    pub prepared_whir: Vec<PreparedMultiConstraintWhirCarrier>,
    pub replayed: PreparedFinitePrefixCircuitProof,
}

//! Recursive-verifier support for WHIR section 5.2 multi-constraint openings.
//!
//! The legacy [`super::WhirModule`] verifies one MLE opening point and must keep
//! its transcript and AIR schedule unchanged.  This module contains the
//! additional, fixed-profile data needed when the same committed matrices are
//! opened at several points:
//!
//! ```text
//! target(mu) = sum_t rho_t sum_j mu^j y[t][j]
//! W(X)       = sum_t rho_t mobius_eq(point[t], X).
//! ```
//!
//! In particular, the points are never collapsed to a synthetic point.  The
//! prefix product for every constraint is retained independently until the
//! final polynomial is evaluated at that constraint's suffix.

use itertools::Itertools;
use openvm_stark_backend::{
    poly_common::{eval_mle_evals_at_point, interpolate_quadratic_at_012},
    proof::WhirProof,
    whir::MULTI_CONSTRAINT_WHIR_PROTOCOL_VERSION,
    FiatShamirTranscript, SystemParams, TranscriptHistory,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, EF, F};
use p3_field::PrimeCharacteristicRing;

use crate::{
    transcript::merkle_verify::MerkleInitialCommitment,
    utils::{pow_observe_sample, FlattenedLayout},
};

pub mod air;
pub mod module;
pub mod trace;

#[cfg(test)]
mod tests;

/// Backend tag used by `derive_multi_constraint_batching_coefficients`.
///
/// The backend deliberately keeps its raw transcript tags private.  The
/// recursion circuit has to constrain the literal field elements, so the
/// values are mirrored here and covered by transcript-differential tests.
pub const MULTI_CONSTRAINT_BATCHING_DOMAIN_TAG: u64 = 0x4d43_5742; // "MCWB"

/// Backend tag used by `prove_whir_opening_multi` / `verify_whir_multi`.
pub const MULTI_CONSTRAINT_PROOF_DOMAIN_TAG: u64 = 0x4d43_5750; // "MCWP"

/// Root and column width of one retained initial setup commitment.
///
/// These values are part of the authority statement.  The complete module
/// consumes them through `MultiConstraintInitialCommitmentBus`; they are not
/// accepted as unconstrained host metadata.
pub type MultiConstraintWhirInitialCommitment = MerkleInitialCommitment;

/// A fixed recursive-circuit profile.  Counts and dimensions are key material,
/// not prover-selected proof fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiConstraintWhirProfile {
    pub constraint_count: usize,
    pub point_dimension: usize,
    pub commitment_widths: Vec<usize>,
}

impl MultiConstraintWhirProfile {
    pub fn new(
        constraint_count: usize,
        point_dimension: usize,
        commitment_widths: Vec<usize>,
    ) -> Result<Self, MultiConstraintWhirError> {
        let profile = Self {
            constraint_count,
            point_dimension,
            commitment_widths,
        };
        profile.validate()?;
        Ok(profile)
    }

    pub fn validate(&self) -> Result<(), MultiConstraintWhirError> {
        if self.constraint_count == 0 {
            return Err(MultiConstraintWhirError::EmptyConstraints);
        }
        if self.point_dimension == 0 {
            return Err(MultiConstraintWhirError::ZeroPointDimension);
        }
        if self.commitment_widths.is_empty() {
            return Err(MultiConstraintWhirError::EmptyCommitments);
        }
        if let Some((commitment, _)) = self
            .commitment_widths
            .iter()
            .enumerate()
            .find(|(_, width)| **width == 0)
        {
            return Err(MultiConstraintWhirError::ZeroCommitmentWidth { commitment });
        }
        let _ = self.total_width()?;
        Ok(())
    }

    pub fn total_width(&self) -> Result<usize, MultiConstraintWhirError> {
        self.commitment_widths
            .iter()
            .try_fold(0usize, |total, &width| {
                total
                    .checked_add(width)
                    .ok_or(MultiConstraintWhirError::TotalWidthOverflow)
            })
    }

    /// Number of point lookups expected from the caller-owned statement AIR.
    ///
    /// Prefix coordinates are consumed once by the per-constraint weight AIR.
    /// A suffix coordinate is consumed by every node in its final-polynomial
    /// folding layer.  The caller can use this value as the multiplicity on the
    /// shared statement lookup bus.
    pub fn point_lookup_multiplicity(
        &self,
        coordinate: usize,
        num_sumcheck_rounds: usize,
    ) -> Result<usize, MultiConstraintWhirError> {
        if coordinate >= self.point_dimension {
            return Err(MultiConstraintWhirError::CoordinateOutOfRange {
                coordinate,
                dimension: self.point_dimension,
            });
        }
        if num_sumcheck_rounds > self.point_dimension {
            return Err(MultiConstraintWhirError::TooManySumcheckRounds {
                rounds: num_sumcheck_rounds,
                dimension: self.point_dimension,
            });
        }
        if coordinate < num_sumcheck_rounds {
            Ok(1)
        } else {
            // The first suffix variable occurs at the root layer and is needed
            // once; the next is needed by two nodes, etc.
            1usize
                .checked_shl((coordinate - num_sumcheck_rounds) as u32)
                .ok_or(MultiConstraintWhirError::TraceSizeOverflow)
        }
    }
}

/// Owned point/opening statement used by CPU preflight and trace generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiConstraintWhirStatement {
    /// `[constraint][coordinate]`.
    pub points: Vec<Vec<EF>>,
    /// `[constraint][commitment][column]`.
    pub openings: Vec<Vec<Vec<EF>>>,
    /// Caller-derived `1, gamma, gamma^2, ...`.
    pub batching_coefficients: Vec<EF>,
}

impl MultiConstraintWhirStatement {
    pub fn validate(
        &self,
        profile: &MultiConstraintWhirProfile,
    ) -> Result<(), MultiConstraintWhirError> {
        profile.validate()?;
        if self.points.len() != profile.constraint_count {
            return Err(MultiConstraintWhirError::ConstraintCount {
                actual: self.points.len(),
                expected: profile.constraint_count,
            });
        }
        if self.openings.len() != profile.constraint_count {
            return Err(MultiConstraintWhirError::OpeningConstraintCount {
                actual: self.openings.len(),
                expected: profile.constraint_count,
            });
        }
        if self.batching_coefficients.len() != profile.constraint_count {
            return Err(MultiConstraintWhirError::CoefficientCount {
                actual: self.batching_coefficients.len(),
                expected: profile.constraint_count,
            });
        }
        for constraint in 0..profile.constraint_count {
            if self.points[constraint].len() != profile.point_dimension {
                return Err(MultiConstraintWhirError::PointDimension {
                    constraint,
                    actual: self.points[constraint].len(),
                    expected: profile.point_dimension,
                });
            }
            if self.openings[constraint].len() != profile.commitment_widths.len() {
                return Err(MultiConstraintWhirError::CommitmentCount {
                    constraint,
                    actual: self.openings[constraint].len(),
                    expected: profile.commitment_widths.len(),
                });
            }
            for (commitment, (&expected, openings)) in profile
                .commitment_widths
                .iter()
                .zip(&self.openings[constraint])
                .enumerate()
            {
                if openings.len() != expected {
                    return Err(MultiConstraintWhirError::OpeningWidth {
                        constraint,
                        commitment,
                        actual: openings.len(),
                        expected,
                    });
                }
            }
        }
        let gamma = self
            .batching_coefficients
            .get(1)
            .copied()
            .unwrap_or(EF::ONE);
        let mut expected = EF::ONE;
        for (constraint, &rho) in self.batching_coefficients.iter().enumerate() {
            if rho != expected {
                return Err(MultiConstraintWhirError::InvalidCoefficientPower { constraint });
            }
            expected *= gamma;
        }
        Ok(())
    }
}

/// Values derived natively and then copied into the small recursive traces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiConstraintWhirDerived {
    pub initial_target: EF,
    /// `[constraint][sumcheck_round]`, before multiplication by `rho_t`.
    pub mobius_prefix_partials: Vec<Vec<EF>>,
    /// `rho_t * mobius_eq(point_t[..r], alpha[..r])` for each constraint.
    pub final_prefix_weights: Vec<EF>,
    /// MLE evaluation of the final coefficient table at each point suffix.
    pub final_suffix_evaluations: Vec<EF>,
    /// Exact generalized contribution consumed by the final WHIR check.
    pub final_weighted_evaluation: EF,
}

/// Strict validation errors.  Public multi-proof entry points return these
/// errors before any trace generator reaches an indexing operation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MultiConstraintWhirError {
    EmptyConstraints,
    ZeroPointDimension,
    EmptyCommitments,
    ZeroCommitmentWidth {
        commitment: usize,
    },
    TotalWidthOverflow,
    TraceSizeOverflow,
    ConstraintCount {
        actual: usize,
        expected: usize,
    },
    OpeningConstraintCount {
        actual: usize,
        expected: usize,
    },
    CoefficientCount {
        actual: usize,
        expected: usize,
    },
    PointDimension {
        constraint: usize,
        actual: usize,
        expected: usize,
    },
    CommitmentCount {
        constraint: usize,
        actual: usize,
        expected: usize,
    },
    OpeningWidth {
        constraint: usize,
        commitment: usize,
        actual: usize,
        expected: usize,
    },
    InvalidCoefficientPower {
        constraint: usize,
    },
    CoordinateOutOfRange {
        coordinate: usize,
        dimension: usize,
    },
    TooManySumcheckRounds {
        rounds: usize,
        dimension: usize,
    },
    AlphaCount {
        actual: usize,
        expected: usize,
    },
    FinalPolynomialLength {
        actual: usize,
        expected: usize,
    },
    ProofShape(&'static str),
}

impl core::fmt::Display for MultiConstraintWhirError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        use MultiConstraintWhirError::*;
        match self {
            EmptyConstraints => write!(f, "multi-constraint WHIR requires at least one constraint"),
            ZeroPointDimension => write!(f, "multi-constraint WHIR point dimension must be nonzero"),
            EmptyCommitments => write!(f, "multi-constraint WHIR requires at least one commitment"),
            ZeroCommitmentWidth { commitment } => {
                write!(f, "multi-constraint WHIR commitment {commitment} has zero width")
            }
            TotalWidthOverflow => write!(f, "multi-constraint WHIR total commitment width overflow"),
            TraceSizeOverflow => write!(f, "multi-constraint WHIR trace size overflow"),
            ConstraintCount { actual, expected } => {
                write!(f, "constraint count {actual}, expected {expected}")
            }
            OpeningConstraintCount { actual, expected } => {
                write!(f, "opening constraint count {actual}, expected {expected}")
            }
            CoefficientCount { actual, expected } => {
                write!(f, "batching coefficient count {actual}, expected {expected}")
            }
            PointDimension { constraint, actual, expected } => write!(
                f,
                "point {constraint} has dimension {actual}, expected {expected}"
            ),
            CommitmentCount { constraint, actual, expected } => write!(
                f,
                "constraint {constraint} has {actual} commitments, expected {expected}"
            ),
            OpeningWidth { constraint, commitment, actual, expected } => write!(
                f,
                "constraint {constraint}, commitment {commitment} has width {actual}, expected {expected}"
            ),
            InvalidCoefficientPower { constraint } => write!(
                f,
                "batching coefficient {constraint} is not the expected gamma power"
            ),
            CoordinateOutOfRange { coordinate, dimension } => {
                write!(f, "coordinate {coordinate} is outside point dimension {dimension}")
            }
            TooManySumcheckRounds { rounds, dimension } => {
                write!(f, "{rounds} sumcheck rounds exceed point dimension {dimension}")
            }
            AlphaCount { actual, expected } => {
                write!(f, "alpha count {actual}, expected {expected}")
            }
            FinalPolynomialLength { actual, expected } => {
                write!(f, "final polynomial length {actual}, expected {expected}")
            }
            ProofShape(message) => write!(f, "WHIR proof shape mismatch: {message}"),
        }
    }
}

impl std::error::Error for MultiConstraintWhirError {}

/// Compute the backend's combined initial target without borrowing backend
/// statement wrappers.  Commitment-major/column-major ordering is mandatory.
pub fn combined_initial_target(
    profile: &MultiConstraintWhirProfile,
    statement: &MultiConstraintWhirStatement,
    mu: EF,
) -> Result<EF, MultiConstraintWhirError> {
    statement.validate(profile)?;
    let total_width = profile.total_width()?;
    let mu_powers = mu.powers().take(total_width).collect_vec();
    Ok(statement
        .openings
        .iter()
        .zip(&statement.batching_coefficients)
        .fold(EF::ZERO, |combined, (constraint, &rho)| {
            let target = constraint
                .iter()
                .flatten()
                .zip(&mu_powers)
                .fold(EF::ZERO, |acc, (&opening, &mu_power)| {
                    acc + mu_power * opening
                });
            combined + rho * target
        }))
}

/// Check the witness coefficients against the challenge sampled by the
/// backend `MCWB` transcript phase.  The AIR enforces the same recurrence; this
/// native check makes malformed inputs fail before trace generation.
pub fn validate_batching_coefficients_against_gamma(
    statement: &MultiConstraintWhirStatement,
    transcript_gamma: EF,
) -> Result<(), MultiConstraintWhirError> {
    let mut expected = EF::ONE;
    for (constraint, &rho) in statement.batching_coefficients.iter().enumerate() {
        if rho != expected {
            return Err(MultiConstraintWhirError::InvalidCoefficientPower { constraint });
        }
        expected *= transcript_gamma;
    }
    Ok(())
}

#[inline]
pub fn mobius_eq_1_native(point: EF, alpha: EF) -> EF {
    EF::ONE - alpha - point.double() + EF::from_u8(3) * point * alpha
}

/// Native reference for the exact multi-constraint weight used by the final
/// WHIR polynomial check.
pub fn derive_multi_constraint_whir_data(
    profile: &MultiConstraintWhirProfile,
    statement: &MultiConstraintWhirStatement,
    mu: EF,
    alphas: &[EF],
    final_poly: &[EF],
) -> Result<MultiConstraintWhirDerived, MultiConstraintWhirError> {
    statement.validate(profile)?;
    if alphas.len() > profile.point_dimension {
        return Err(MultiConstraintWhirError::TooManySumcheckRounds {
            rounds: alphas.len(),
            dimension: profile.point_dimension,
        });
    }
    let suffix_dimension = profile.point_dimension - alphas.len();
    let expected_final_len = 1usize
        .checked_shl(suffix_dimension as u32)
        .ok_or(MultiConstraintWhirError::TraceSizeOverflow)?;
    if final_poly.len() != expected_final_len {
        return Err(MultiConstraintWhirError::FinalPolynomialLength {
            actual: final_poly.len(),
            expected: expected_final_len,
        });
    }

    let initial_target = combined_initial_target(profile, statement, mu)?;
    let mut mobius_prefix_partials = Vec::with_capacity(profile.constraint_count);
    let mut final_prefix_weights = Vec::with_capacity(profile.constraint_count);
    let mut final_suffix_evaluations = Vec::with_capacity(profile.constraint_count);
    let mut final_weighted_evaluation = EF::ZERO;

    for constraint in 0..profile.constraint_count {
        let point = &statement.points[constraint];
        let mut partial = EF::ONE;
        let mut partials = Vec::with_capacity(alphas.len());
        for (&u, &alpha) in point.iter().zip(alphas) {
            partial *= mobius_eq_1_native(u, alpha);
            partials.push(partial);
        }
        let weighted_prefix = statement.batching_coefficients[constraint] * partial;
        let mut final_poly_buf = final_poly.to_vec();
        let suffix_eval = eval_mle_evals_at_point(&mut final_poly_buf, &point[alphas.len()..]);
        final_weighted_evaluation += weighted_prefix * suffix_eval;
        mobius_prefix_partials.push(partials);
        final_prefix_weights.push(weighted_prefix);
        final_suffix_evaluations.push(suffix_eval);
    }

    Ok(MultiConstraintWhirDerived {
        initial_target,
        mobius_prefix_partials,
        final_prefix_weights,
        final_suffix_evaluations,
        final_weighted_evaluation,
    })
}

/// Transcript values needed by recursive trace generation.  `batching_gamma`
/// is sampled by the caller-owned multi-statement phase; all remaining values
/// start at the backend `MCWP` proof-domain prefix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MultiConstraintWhirTranscriptPreflight {
    pub batching_prefix_tidx: usize,
    pub batching_gamma: EF,
    pub proof_prefix_tidx: usize,
    pub mu_pow_sample: F,
    pub mu: EF,
    pub post_mu_tidx: usize,
    pub whir_round_tidx_per_round: Vec<usize>,
    pub query_tidx_per_round: Vec<usize>,
    pub alphas: Vec<EF>,
    pub z0s: Vec<EF>,
    pub gammas: Vec<EF>,
    pub folding_pow_samples: Vec<F>,
    pub query_pow_samples: Vec<F>,
    pub queries: Vec<F>,
}

/// Replay the backend's `MCWB` coefficient derivation.  The caller must have
/// already absorbed roots, layouts, identifiers, points and claimed openings.
pub fn derive_batching_coefficients_preflight<TS>(
    transcript: &mut TS,
    constraint_count: usize,
) -> Result<(usize, EF, Vec<EF>), MultiConstraintWhirError>
where
    TS: FiatShamirTranscript<BabyBearPoseidon2Config> + TranscriptHistory,
{
    if constraint_count == 0 {
        return Err(MultiConstraintWhirError::EmptyConstraints);
    }
    let start = transcript.len();
    transcript.observe(F::from_u64(MULTI_CONSTRAINT_BATCHING_DOMAIN_TAG));
    transcript.observe(F::from_u32(MULTI_CONSTRAINT_WHIR_PROTOCOL_VERSION));
    transcript.observe(F::from_usize(constraint_count));
    let gamma = transcript.sample_ext();
    Ok((
        start,
        gamma,
        gamma.powers().take(constraint_count).collect(),
    ))
}

/// Replay the backend multi-proof domain and ordinary WHIR transcript exactly.
///
/// This is deliberately separate from [`super::WhirModule::run_preflight`]:
/// the legacy path observes no `MCWP` prefix and samples μ in the stacking
/// module.  Existing callers therefore retain their old transcript byte for
/// byte.
pub fn run_multi_constraint_whir_preflight<TS>(
    transcript: &mut TS,
    params: &SystemParams,
    whir_proof: &WhirProof<BabyBearPoseidon2Config>,
    constraint_count: usize,
    batching_prefix_tidx: usize,
    batching_gamma: EF,
) -> Result<MultiConstraintWhirTranscriptPreflight, MultiConstraintWhirError>
where
    TS: FiatShamirTranscript<BabyBearPoseidon2Config> + TranscriptHistory,
{
    validate_whir_proof_shape(params, whir_proof)?;
    if constraint_count == 0 {
        return Err(MultiConstraintWhirError::EmptyConstraints);
    }

    let proof_prefix_tidx = transcript.len();
    transcript.observe(F::from_u64(MULTI_CONSTRAINT_PROOF_DOMAIN_TAG));
    transcript.observe(F::from_u32(MULTI_CONSTRAINT_WHIR_PROTOCOL_VERSION));
    transcript.observe(F::from_usize(constraint_count));

    let mu_pow_sample = pow_observe_sample(
        transcript,
        params.whir.mu_pow_bits,
        whir_proof.mu_pow_witness,
    );
    let mu = transcript.sample_ext();
    let post_mu_tidx = transcript.len();

    let k_whir = params.k_whir();
    let num_whir_rounds = params.num_whir_rounds();
    let mut whir_round_tidx_per_round = Vec::with_capacity(num_whir_rounds);
    let mut query_tidx_per_round = Vec::with_capacity(num_whir_rounds);
    let mut alphas = Vec::with_capacity(params.num_whir_sumcheck_rounds());
    let mut z0s = Vec::with_capacity(num_whir_rounds.saturating_sub(1));
    let mut gammas = Vec::with_capacity(num_whir_rounds);
    let mut folding_pow_samples = Vec::with_capacity(params.num_whir_sumcheck_rounds());
    let mut query_pow_samples = Vec::with_capacity(num_whir_rounds);
    let total_queries = params.whir.rounds.iter().map(|r| r.num_queries).sum();
    let mut queries = Vec::with_capacity(total_queries);

    for round in 0..num_whir_rounds {
        whir_round_tidx_per_round.push(transcript.len());
        for subround in 0..k_whir {
            let [ev1, ev2] = whir_proof.whir_sumcheck_polys[round * k_whir + subround];
            transcript.observe_ext(ev1);
            transcript.observe_ext(ev2);
            folding_pow_samples.push(pow_observe_sample(
                transcript,
                params.whir.folding_pow_bits,
                whir_proof.folding_pow_witnesses[round * k_whir + subround],
            ));
            alphas.push(transcript.sample_ext());
        }

        if round + 1 < num_whir_rounds {
            transcript.observe_commit(whir_proof.codeword_commits[round]);
            z0s.push(transcript.sample_ext());
            transcript.observe_ext(whir_proof.ood_values[round]);
        } else {
            for &coefficient in &whir_proof.final_poly {
                transcript.observe_ext(coefficient);
            }
        }

        query_pow_samples.push(pow_observe_sample(
            transcript,
            params.whir.query_phase_pow_bits,
            whir_proof.query_phase_pow_witnesses[round],
        ));
        query_tidx_per_round.push(transcript.len());
        for _ in 0..params.whir.rounds[round].num_queries {
            queries.push(transcript.sample());
        }
        gammas.push(transcript.sample_ext());
    }

    Ok(MultiConstraintWhirTranscriptPreflight {
        batching_prefix_tidx,
        batching_gamma,
        proof_prefix_tidx,
        mu_pow_sample,
        mu,
        post_mu_tidx,
        whir_round_tidx_per_round,
        query_tidx_per_round,
        alphas,
        z0s,
        gammas,
        folding_pow_samples,
        query_pow_samples,
        queries,
    })
}

/// Validate every proof vector indexed by recursive trace generation.
pub fn validate_whir_proof_shape(
    params: &SystemParams,
    proof: &WhirProof<BabyBearPoseidon2Config>,
) -> Result<(), MultiConstraintWhirError> {
    let rounds = params.num_whir_rounds();
    let sumcheck_rounds = params.num_whir_sumcheck_rounds();
    if rounds == 0 {
        return Err(MultiConstraintWhirError::ProofShape("zero WHIR rounds"));
    }
    if proof.whir_sumcheck_polys.len() != sumcheck_rounds {
        return Err(MultiConstraintWhirError::ProofShape(
            "sumcheck polynomial count",
        ));
    }
    if proof.folding_pow_witnesses.len() != sumcheck_rounds {
        return Err(MultiConstraintWhirError::ProofShape(
            "folding PoW witness count",
        ));
    }
    if proof.query_phase_pow_witnesses.len() != rounds {
        return Err(MultiConstraintWhirError::ProofShape(
            "query PoW witness count",
        ));
    }
    if proof.codeword_commits.len() != rounds - 1 || proof.ood_values.len() != rounds - 1 {
        return Err(MultiConstraintWhirError::ProofShape(
            "non-final round commitment/OOD count",
        ));
    }
    if proof.codeword_opened_values.len() != rounds - 1
        || proof.codeword_merkle_proofs.len() != rounds - 1
    {
        return Err(MultiConstraintWhirError::ProofShape(
            "non-initial opening round count",
        ));
    }
    if proof.final_poly.len() != 1usize << params.log_final_poly_len() {
        return Err(MultiConstraintWhirError::FinalPolynomialLength {
            actual: proof.final_poly.len(),
            expected: 1usize << params.log_final_poly_len(),
        });
    }
    for (round, cfg) in params.whir.rounds.iter().enumerate() {
        if round == 0 {
            if proof
                .initial_round_opened_rows
                .iter()
                .any(|rows| rows.len() != cfg.num_queries)
                || proof
                    .initial_round_merkle_proofs
                    .iter()
                    .any(|paths| paths.len() != cfg.num_queries)
            {
                return Err(MultiConstraintWhirError::ProofShape("initial query count"));
            }
        } else if proof.codeword_opened_values[round - 1].len() != cfg.num_queries
            || proof.codeword_merkle_proofs[round - 1].len() != cfg.num_queries
        {
            return Err(MultiConstraintWhirError::ProofShape("codeword query count"));
        }
    }
    Ok(())
}

/// Recompute the scalar WHIR claim evolution.  This is shared by the trace
/// generator and native differential tests; it does not perform Merkle checks.
pub fn evolve_whir_claims(
    initial_target: EF,
    params: &SystemParams,
    proof: &WhirProof<BabyBearPoseidon2Config>,
    preflight: &MultiConstraintWhirTranscriptPreflight,
    query_evaluations: &[EF],
) -> Result<Vec<EF>, MultiConstraintWhirError> {
    validate_whir_proof_shape(params, proof)?;
    let expected_queries: usize = params.whir.rounds.iter().map(|r| r.num_queries).sum();
    if query_evaluations.len() != expected_queries {
        return Err(MultiConstraintWhirError::ProofShape(
            "query evaluation count",
        ));
    }
    if preflight.alphas.len() != params.num_whir_sumcheck_rounds() {
        return Err(MultiConstraintWhirError::AlphaCount {
            actual: preflight.alphas.len(),
            expected: params.num_whir_sumcheck_rounds(),
        });
    }

    let mut claims = Vec::with_capacity(params.num_whir_rounds() + 1);
    let mut claim = initial_target;
    claims.push(claim);
    let mut query_offset = 0;
    for round in 0..params.num_whir_rounds() {
        for subround in 0..params.k_whir() {
            let [ev1, ev2] = proof.whir_sumcheck_polys[round * params.k_whir() + subround];
            let ev0 = claim - ev1;
            claim = interpolate_quadratic_at_012(
                &[ev0, ev1, ev2],
                preflight.alphas[round * params.k_whir() + subround],
            );
        }
        let gamma = preflight.gammas[round];
        if round + 1 < params.num_whir_rounds() {
            claim += gamma * proof.ood_values[round];
        }
        for (&yi, gamma_power) in query_evaluations
            [query_offset..query_offset + params.whir.rounds[round].num_queries]
            .iter()
            .zip(gamma.powers().skip(2))
        {
            claim += yi * gamma_power;
        }
        query_offset += params.whir.rounds[round].num_queries;
        claims.push(claim);
    }
    Ok(claims)
}

/// Fixed row layout used by the small weight trace.
#[derive(Clone, Debug)]
pub struct MultiConstraintWeightLayout {
    pub num_proofs: usize,
    pub constraint_count: usize,
    pub num_sumcheck_rounds: usize,
}

impl MultiConstraintWeightLayout {
    pub fn new(
        num_proofs: usize,
        constraint_count: usize,
        num_sumcheck_rounds: usize,
    ) -> Result<Self, MultiConstraintWhirError> {
        if constraint_count == 0 {
            return Err(MultiConstraintWhirError::EmptyConstraints);
        }
        if num_sumcheck_rounds == 0 {
            return Err(MultiConstraintWhirError::ProofShape(
                "zero WHIR sumcheck rounds",
            ));
        }
        let _ = num_proofs
            .checked_mul(constraint_count)
            .and_then(|x| x.checked_mul(num_sumcheck_rounds))
            .ok_or(MultiConstraintWhirError::TraceSizeOverflow)?;
        Ok(Self {
            num_proofs,
            constraint_count,
            num_sumcheck_rounds,
        })
    }
}

impl FlattenedLayout for MultiConstraintWeightLayout {
    type Index = (usize, usize, usize);

    fn len(&self) -> usize {
        self.num_proofs * self.constraint_count * self.num_sumcheck_rounds
    }

    fn offset(&self, (proof, constraint, round): Self::Index) -> usize {
        debug_assert!(proof < self.num_proofs);
        debug_assert!(constraint < self.constraint_count);
        debug_assert!(round < self.num_sumcheck_rounds);
        (proof * self.constraint_count + constraint) * self.num_sumcheck_rounds + round
    }
}

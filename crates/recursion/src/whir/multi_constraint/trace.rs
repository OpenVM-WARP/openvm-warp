use core::borrow::BorrowMut;

use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{proof::WhirProof, SystemParams};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, D_EF, EF, F};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

use super::{
    air::{
        MultiConstraintCompletionCols, MultiConstraintFinalAggregateCols,
        MultiConstraintFinalPolyCols, MultiConstraintInitialCommitmentCols,
        MultiConstraintPrefixCols, MultiConstraintSumcheckCols, MultiConstraintTargetCols,
        MultiConstraintWeightCols,
    },
    derive_multi_constraint_whir_data, mobius_eq_1_native,
    validate_batching_coefficients_against_gamma, MultiConstraintWeightLayout,
    MultiConstraintWhirDerived, MultiConstraintWhirError, MultiConstraintWhirInitialCommitment,
    MultiConstraintWhirProfile, MultiConstraintWhirStatement,
    MultiConstraintWhirTranscriptPreflight,
};
use crate::utils::{pow_tidx_count, FlattenedLayout};

/// Row-aligned checkpoint emitted by `TranscriptModule` at the end of WHIR.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MultiConstraintWhirTerminalCheckpoint {
    pub end_tidx: usize,
    /// Number of squeeze operations in the selected transcript row. Together
    /// with `end_tidx` this is the canonical constrained duplex cursor.
    pub sample_count: usize,
    pub state: [F; POSEIDON2_WIDTH],
}

#[inline]
fn copy_ext(dst: &mut [F; D_EF], value: EF) {
    dst.copy_from_slice(value.as_basis_coefficients_slice());
}

fn padded_height(valid_rows: usize, required_height: Option<usize>) -> Option<usize> {
    let minimum = valid_rows.max(1).next_power_of_two();
    match required_height {
        Some(height) if height < valid_rows || !height.is_power_of_two() => None,
        Some(height) => Some(height),
        None => Some(minimum),
    }
}

pub fn generate_multi_constraint_initial_commitment_trace(
    commitments: &[Vec<MultiConstraintWhirInitialCommitment>],
    required_height: Option<usize>,
) -> Result<Option<RowMajorMatrix<F>>, MultiConstraintWhirError> {
    if commitments.is_empty() || commitments.iter().any(Vec::is_empty) {
        return Err(MultiConstraintWhirError::EmptyCommitments);
    }
    let valid_rows = commitments.iter().try_fold(0usize, |total, proof| {
        proof.iter().try_fold(total, |total, commitment| {
            if commitment.width == 0 {
                return Err(MultiConstraintWhirError::ZeroCommitmentWidth { commitment: 0 });
            }
            total
                .checked_add(commitment.width)
                .ok_or(MultiConstraintWhirError::TraceSizeOverflow)
        })
    })?;
    let Some(height) = padded_height(valid_rows, required_height) else {
        return Ok(None);
    };
    let width = MultiConstraintInitialCommitmentCols::<F>::width();
    let mut values = vec![F::ZERO; height * width];
    let mut row_idx = 0;
    for (proof_idx, proof_commitments) in commitments.iter().enumerate() {
        for (commit_idx, commitment) in proof_commitments.iter().enumerate() {
            for col_idx in 0..commitment.width {
                let cols: &mut MultiConstraintInitialCommitmentCols<F> =
                    values[row_idx * width..(row_idx + 1) * width].borrow_mut();
                cols.is_enabled = F::ONE;
                cols.proof_idx = F::from_usize(proof_idx);
                cols.commit_idx = F::from_usize(commit_idx);
                cols.col_idx = F::from_usize(col_idx);
                cols.is_first_in_proof = F::from_bool(commit_idx == 0 && col_idx == 0);
                cols.is_first_in_commit = F::from_bool(col_idx == 0);
                cols.width = F::from_usize(commitment.width);
                cols.commitment = commitment.commitment;
                row_idx += 1;
            }
        }
    }
    Ok(Some(RowMajorMatrix::new(values, width)))
}

pub fn generate_multi_constraint_prefix_trace(
    preflights: &[MultiConstraintWhirTranscriptPreflight],
    statements: &[MultiConstraintWhirStatement],
    derived: &[MultiConstraintWhirDerived],
    proofs: &[&WhirProof<BabyBearPoseidon2Config>],
    constraint_count: usize,
    required_height: Option<usize>,
) -> Result<Option<RowMajorMatrix<F>>, MultiConstraintWhirError> {
    if preflights.len() != statements.len()
        || preflights.len() != derived.len()
        || preflights.len() != proofs.len()
    {
        return Err(MultiConstraintWhirError::ProofShape(
            "prefix batch length mismatch",
        ));
    }
    if constraint_count == 0 {
        return Err(MultiConstraintWhirError::EmptyConstraints);
    }
    let valid_rows = preflights
        .len()
        .checked_mul(constraint_count)
        .ok_or(MultiConstraintWhirError::TraceSizeOverflow)?;
    let Some(height) = padded_height(valid_rows, required_height) else {
        return Ok(None);
    };
    let width = MultiConstraintPrefixCols::<F>::width();
    let mut values = vec![F::ZERO; height * width];
    for proof_idx in 0..preflights.len() {
        let preflight = &preflights[proof_idx];
        let statement = &statements[proof_idx];
        validate_batching_coefficients_against_gamma(statement, preflight.batching_gamma)?;
        for constraint_idx in 0..constraint_count {
            let row_idx = proof_idx * constraint_count + constraint_idx;
            let cols: &mut MultiConstraintPrefixCols<F> =
                values[row_idx * width..(row_idx + 1) * width].borrow_mut();
            cols.is_enabled = F::ONE;
            cols.proof_idx = F::from_usize(proof_idx);
            cols.constraint_idx = F::from_usize(constraint_idx);
            cols.is_first_in_proof = F::from_bool(constraint_idx == 0);
            cols.prefix_tidx = F::from_usize(preflight.batching_prefix_tidx);
            copy_ext(&mut cols.batching_gamma, preflight.batching_gamma);
            copy_ext(
                &mut cols.rho,
                statement.batching_coefficients[constraint_idx],
            );
            cols.mu_pow_witness = proofs[proof_idx].mu_pow_witness;
            cols.mu_pow_sample = preflight.mu_pow_sample;
            copy_ext(&mut cols.mu, preflight.mu);
            copy_ext(&mut cols.initial_target, derived[proof_idx].initial_target);
        }
    }
    Ok(Some(RowMajorMatrix::new(values, width)))
}

pub fn generate_multi_constraint_target_trace(
    profile: &MultiConstraintWhirProfile,
    statements: &[MultiConstraintWhirStatement],
    mus: &[EF],
    required_height: Option<usize>,
) -> Result<Option<RowMajorMatrix<F>>, MultiConstraintWhirError> {
    if statements.len() != mus.len() {
        return Err(MultiConstraintWhirError::ProofShape(
            "target batch length mismatch",
        ));
    }
    let total_width = profile.total_width()?;
    for statement in statements {
        statement.validate(profile)?;
    }
    let rows_per_proof = profile
        .constraint_count
        .checked_mul(total_width)
        .ok_or(MultiConstraintWhirError::TraceSizeOverflow)?;
    let valid_rows = statements
        .len()
        .checked_mul(rows_per_proof)
        .ok_or(MultiConstraintWhirError::TraceSizeOverflow)?;
    let Some(height) = padded_height(valid_rows, required_height) else {
        return Ok(None);
    };
    let width = MultiConstraintTargetCols::<F>::width();
    let mut values = vec![F::ZERO; height * width];
    for (proof_idx, (statement, &mu)) in statements.iter().zip(mus).enumerate() {
        let mut accumulator = EF::ZERO;
        for constraint_idx in 0..profile.constraint_count {
            let rho = statement.batching_coefficients[constraint_idx];
            let flat_openings = statement.openings[constraint_idx]
                .iter()
                .flatten()
                .copied()
                .collect::<Vec<_>>();
            let mut mu_power = EF::ONE;
            for (opening_idx, &opening) in flat_openings.iter().enumerate() {
                let row_idx =
                    proof_idx * rows_per_proof + constraint_idx * total_width + opening_idx;
                let cols: &mut MultiConstraintTargetCols<F> =
                    values[row_idx * width..(row_idx + 1) * width].borrow_mut();
                cols.is_enabled = F::ONE;
                cols.proof_idx = F::from_usize(proof_idx);
                cols.constraint_idx = F::from_usize(constraint_idx);
                cols.opening_idx = F::from_usize(opening_idx);
                cols.is_first_in_proof = F::from_bool(constraint_idx == 0 && opening_idx == 0);
                cols.is_first_in_constraint = F::from_bool(opening_idx == 0);
                copy_ext(&mut cols.rho, rho);
                copy_ext(&mut cols.mu, mu);
                copy_ext(&mut cols.mu_power, mu_power);
                copy_ext(&mut cols.opening, opening);
                copy_ext(&mut cols.accumulator, accumulator);
                accumulator += rho * mu_power * opening;
                copy_ext(&mut cols.next_accumulator, accumulator);
                mu_power *= mu;
            }
        }
    }
    Ok(Some(RowMajorMatrix::new(values, width)))
}

/// Inputs already derived by the ordinary WHIR query/folding path.
pub struct MultiConstraintSumcheckTraceInput<'a> {
    pub proof: &'a WhirProof<BabyBearPoseidon2Config>,
    pub preflight: &'a MultiConstraintWhirTranscriptPreflight,
    /// One claim at the start of every WHIR round.
    pub initial_claim_per_round: &'a [EF],
    /// One post-sumcheck claim per alpha round.
    pub post_sumcheck_claims: &'a [EF],
}

pub fn generate_multi_constraint_sumcheck_trace(
    params: &SystemParams,
    inputs: &[MultiConstraintSumcheckTraceInput<'_>],
    constraint_count: usize,
    alpha_lookup_counts_without_weights: &[usize],
    required_height: Option<usize>,
) -> Result<Option<RowMajorMatrix<F>>, MultiConstraintWhirError> {
    let rows_per_proof = params.num_whir_sumcheck_rounds();
    if rows_per_proof == 0 {
        return Err(MultiConstraintWhirError::ProofShape(
            "zero WHIR sumcheck rounds",
        ));
    }
    if alpha_lookup_counts_without_weights.len() != rows_per_proof {
        return Err(MultiConstraintWhirError::AlphaCount {
            actual: alpha_lookup_counts_without_weights.len(),
            expected: rows_per_proof,
        });
    }
    let valid_rows = inputs
        .len()
        .checked_mul(rows_per_proof)
        .ok_or(MultiConstraintWhirError::TraceSizeOverflow)?;
    let Some(height) = padded_height(valid_rows, required_height) else {
        return Ok(None);
    };
    let width = MultiConstraintSumcheckCols::<F>::width();
    let mut values = vec![F::ZERO; height * width];
    let k = params.k_whir();
    let pow_offset = pow_tidx_count(params.whir.folding_pow_bits);
    for (proof_idx, input) in inputs.iter().enumerate() {
        if input.preflight.alphas.len() != rows_per_proof
            || input.proof.whir_sumcheck_polys.len() != rows_per_proof
            || input.proof.folding_pow_witnesses.len() != rows_per_proof
            || input.preflight.folding_pow_samples.len() != rows_per_proof
            || input.initial_claim_per_round.len() != params.num_whir_rounds()
            || input.post_sumcheck_claims.len() != rows_per_proof
        {
            return Err(MultiConstraintWhirError::ProofShape(
                "sumcheck trace input dimensions",
            ));
        }
        for i in 0..rows_per_proof {
            let whir_round = i / k;
            let subidx = i % k;
            let row_idx = proof_idx * rows_per_proof + i;
            let cols: &mut MultiConstraintSumcheckCols<F> =
                values[row_idx * width..(row_idx + 1) * width].borrow_mut();
            cols.is_enabled = F::ONE;
            cols.proof_idx = F::from_usize(proof_idx);
            cols.whir_round = F::from_usize(whir_round);
            cols.subidx = F::from_usize(subidx);
            cols.is_first_in_proof = F::from_bool(i == 0);
            cols.is_first_in_round = F::from_bool(subidx == 0);
            cols.tidx = F::from_usize(
                input.preflight.whir_round_tidx_per_round[whir_round]
                    + subidx * (3 * D_EF + pow_offset),
            );
            copy_ext(&mut cols.ev1, input.proof.whir_sumcheck_polys[i][0]);
            copy_ext(&mut cols.ev2, input.proof.whir_sumcheck_polys[i][1]);
            cols.folding_pow_witness = input.proof.folding_pow_witnesses[i];
            cols.folding_pow_sample = input.preflight.folding_pow_samples[i];
            copy_ext(&mut cols.alpha, input.preflight.alphas[i]);
            copy_ext(
                &mut cols.pre_claim,
                if subidx == 0 {
                    input.initial_claim_per_round[whir_round]
                } else {
                    input.post_sumcheck_claims[i - 1]
                },
            );
            copy_ext(
                &mut cols.post_group_claim,
                input.post_sumcheck_claims[(whir_round + 1) * k - 1],
            );
            cols.alpha_lookup_count = F::from_usize(
                alpha_lookup_counts_without_weights[i]
                    .checked_add(constraint_count)
                    .ok_or(MultiConstraintWhirError::TraceSizeOverflow)?,
            );
        }
    }
    Ok(Some(RowMajorMatrix::new(values, width)))
}

pub fn generate_multi_constraint_weight_trace(
    profile: &MultiConstraintWhirProfile,
    statements: &[MultiConstraintWhirStatement],
    alphas: &[Vec<EF>],
    required_height: Option<usize>,
) -> Result<Option<RowMajorMatrix<F>>, MultiConstraintWhirError> {
    if statements.len() != alphas.len() {
        return Err(MultiConstraintWhirError::ProofShape(
            "weight batch length mismatch",
        ));
    }
    let num_sumcheck_rounds = alphas.first().map_or(0, Vec::len);
    let layout = MultiConstraintWeightLayout::new(
        statements.len(),
        profile.constraint_count,
        num_sumcheck_rounds,
    )?;
    for (statement, alpha) in statements.iter().zip(alphas) {
        statement.validate(profile)?;
        if alpha.len() != num_sumcheck_rounds {
            return Err(MultiConstraintWhirError::AlphaCount {
                actual: alpha.len(),
                expected: num_sumcheck_rounds,
            });
        }
        if alpha.len() > profile.point_dimension {
            return Err(MultiConstraintWhirError::TooManySumcheckRounds {
                rounds: alpha.len(),
                dimension: profile.point_dimension,
            });
        }
    }
    let Some(height) = padded_height(layout.len(), required_height) else {
        return Ok(None);
    };
    let width = MultiConstraintWeightCols::<F>::width();
    let mut values = vec![F::ZERO; height * width];
    for proof_idx in 0..statements.len() {
        for constraint_idx in 0..profile.constraint_count {
            let rho = statements[proof_idx].batching_coefficients[constraint_idx];
            let mut partial = EF::ONE;
            for round_idx in 0..num_sumcheck_rounds {
                let row_idx = layout.offset((proof_idx, constraint_idx, round_idx));
                let cols: &mut MultiConstraintWeightCols<F> =
                    values[row_idx * width..(row_idx + 1) * width].borrow_mut();
                cols.is_enabled = F::ONE;
                cols.proof_idx = F::from_usize(proof_idx);
                cols.constraint_idx = F::from_usize(constraint_idx);
                cols.round_idx = F::from_usize(round_idx);
                cols.is_first_in_proof = F::from_bool(constraint_idx == 0 && round_idx == 0);
                cols.is_first_in_constraint = F::from_bool(round_idx == 0);
                copy_ext(&mut cols.rho, rho);
                let point = statements[proof_idx].points[constraint_idx][round_idx];
                let alpha = alphas[proof_idx][round_idx];
                copy_ext(&mut cols.point, point);
                copy_ext(&mut cols.alpha, alpha);
                copy_ext(&mut cols.partial_before, partial);
                partial *= mobius_eq_1_native(point, alpha);
                copy_ext(&mut cols.partial_after, partial);
            }
        }
    }
    Ok(Some(RowMajorMatrix::new(values, width)))
}

pub fn generate_multi_constraint_final_poly_trace(
    profile: &MultiConstraintWhirProfile,
    statements: &[MultiConstraintWhirStatement],
    derived: &[MultiConstraintWhirDerived],
    final_polys: &[Vec<EF>],
    num_sumcheck_rounds: usize,
    tidx_final_poly_starts: &[usize],
    required_height: Option<usize>,
) -> Result<Option<RowMajorMatrix<F>>, MultiConstraintWhirError> {
    let batch = statements.len();
    if derived.len() != batch || final_polys.len() != batch || tidx_final_poly_starts.len() != batch
    {
        return Err(MultiConstraintWhirError::ProofShape(
            "final polynomial batch length mismatch",
        ));
    }
    if num_sumcheck_rounds > profile.point_dimension {
        return Err(MultiConstraintWhirError::TooManySumcheckRounds {
            rounds: num_sumcheck_rounds,
            dimension: profile.point_dimension,
        });
    }
    let num_vars = profile.point_dimension - num_sumcheck_rounds;
    let rows_per_tree = (1usize
        .checked_shl((num_vars + 1) as u32)
        .ok_or(MultiConstraintWhirError::TraceSizeOverflow)?)
    .checked_sub(1)
    .ok_or(MultiConstraintWhirError::TraceSizeOverflow)?;
    let rows_per_proof = profile
        .constraint_count
        .checked_mul(rows_per_tree)
        .ok_or(MultiConstraintWhirError::TraceSizeOverflow)?;
    let valid_rows = batch
        .checked_mul(rows_per_proof)
        .ok_or(MultiConstraintWhirError::TraceSizeOverflow)?;
    let Some(height) = padded_height(valid_rows, required_height) else {
        return Ok(None);
    };
    let width = MultiConstraintFinalPolyCols::<F>::width();
    let mut values = vec![F::ZERO; height * width];

    for proof_idx in 0..batch {
        statements[proof_idx].validate(profile)?;
        if final_polys[proof_idx].len() != 1usize << num_vars {
            return Err(MultiConstraintWhirError::FinalPolynomialLength {
                actual: final_polys[proof_idx].len(),
                expected: 1usize << num_vars,
            });
        }
        for constraint_idx in 0..profile.constraint_count {
            let mut buffer = final_polys[proof_idx].clone();
            let mut row_in_tree = 0usize;
            for layer in 0..=num_vars {
                let len = 1usize << (num_vars - layer);
                let point = if layer == 0 {
                    EF::ZERO
                } else {
                    statements[proof_idx].points[constraint_idx]
                        [num_sumcheck_rounds + num_vars - layer]
                };
                let (left, right) = buffer.split_at_mut(len);
                for node_idx in 0..len {
                    let (left_value, right_value, value) = if layer == 0 {
                        (left[node_idx], EF::ZERO, left[node_idx])
                    } else {
                        let l = left[node_idx];
                        let r = right[node_idx];
                        let value = l + (r - l) * point;
                        left[node_idx] = value;
                        (l, r, value)
                    };
                    let row_idx =
                        proof_idx * rows_per_proof + constraint_idx * rows_per_tree + row_in_tree;
                    let cols: &mut MultiConstraintFinalPolyCols<F> =
                        values[row_idx * width..(row_idx + 1) * width].borrow_mut();
                    cols.is_enabled = F::ONE;
                    cols.proof_idx = F::from_usize(proof_idx);
                    cols.constraint_idx = F::from_usize(constraint_idx);
                    cols.is_first_in_proof = F::from_bool(constraint_idx == 0 && row_in_tree == 0);
                    // This flag marks the transcript-owning constraint, not the
                    // first row of every constraint.
                    cols.is_first_in_constraint = F::from_bool(constraint_idx == 0);
                    cols.is_root = F::from_bool(layer == num_vars);
                    cols.layer = F::from_usize(layer);
                    cols.layer_inv = cols.layer.try_inverse().unwrap_or_default();
                    cols.node_idx = F::from_usize(node_idx);
                    cols.node_idx_inv = cols.node_idx.try_inverse().unwrap_or_default();
                    cols.num_nodes_in_layer = F::from_usize(len);
                    cols.tidx_final_poly_start = F::from_usize(tidx_final_poly_starts[proof_idx]);
                    copy_ext(&mut cols.point, point);
                    copy_ext(&mut cols.left_value, left_value);
                    copy_ext(&mut cols.right_value, right_value);
                    copy_ext(&mut cols.value, value);
                    if layer == num_vars {
                        copy_ext(
                            &mut cols.final_weight,
                            derived[proof_idx].final_prefix_weights[constraint_idx],
                        );
                    }
                    row_in_tree += 1;
                }
            }
            debug_assert_eq!(row_in_tree, rows_per_tree);
        }
    }
    Ok(Some(RowMajorMatrix::new(values, width)))
}

pub fn generate_multi_constraint_final_aggregate_trace(
    derived: &[MultiConstraintWhirDerived],
    tidx_final_poly_starts: &[usize],
    constraint_count: usize,
    required_height: Option<usize>,
) -> Result<Option<RowMajorMatrix<F>>, MultiConstraintWhirError> {
    if derived.len() != tidx_final_poly_starts.len() {
        return Err(MultiConstraintWhirError::ProofShape(
            "final aggregate batch length mismatch",
        ));
    }
    if constraint_count == 0 {
        return Err(MultiConstraintWhirError::EmptyConstraints);
    }
    let valid_rows = derived
        .len()
        .checked_mul(constraint_count)
        .ok_or(MultiConstraintWhirError::TraceSizeOverflow)?;
    let Some(height) = padded_height(valid_rows, required_height) else {
        return Ok(None);
    };
    let width = MultiConstraintFinalAggregateCols::<F>::width();
    let mut values = vec![F::ZERO; height * width];
    for proof_idx in 0..derived.len() {
        if derived[proof_idx].final_prefix_weights.len() != constraint_count
            || derived[proof_idx].final_suffix_evaluations.len() != constraint_count
        {
            return Err(MultiConstraintWhirError::ConstraintCount {
                actual: derived[proof_idx].final_prefix_weights.len(),
                expected: constraint_count,
            });
        }
        let mut accumulator = EF::ZERO;
        for constraint_idx in 0..constraint_count {
            let contribution = derived[proof_idx].final_prefix_weights[constraint_idx]
                * derived[proof_idx].final_suffix_evaluations[constraint_idx];
            let row_idx = proof_idx * constraint_count + constraint_idx;
            let cols: &mut MultiConstraintFinalAggregateCols<F> =
                values[row_idx * width..(row_idx + 1) * width].borrow_mut();
            cols.is_enabled = F::ONE;
            cols.proof_idx = F::from_usize(proof_idx);
            cols.constraint_idx = F::from_usize(constraint_idx);
            cols.is_first_in_proof = F::from_bool(constraint_idx == 0);
            copy_ext(&mut cols.contribution, contribution);
            copy_ext(&mut cols.accumulator, accumulator);
            accumulator += contribution;
            copy_ext(&mut cols.next_accumulator, accumulator);
            cols.tidx_final_poly_start = F::from_usize(tidx_final_poly_starts[proof_idx]);
        }
    }
    Ok(Some(RowMajorMatrix::new(values, width)))
}

pub fn generate_multi_constraint_completion_trace(
    checkpoints: &[MultiConstraintWhirTerminalCheckpoint],
    final_aggregates: &[EF],
    final_claims: &[EF],
    class_index: usize,
    required_height: Option<usize>,
) -> Result<Option<RowMajorMatrix<F>>, MultiConstraintWhirError> {
    if checkpoints.len() != final_aggregates.len() || checkpoints.len() != final_claims.len() {
        return Err(MultiConstraintWhirError::ProofShape(
            "completion batch length mismatch",
        ));
    }
    let valid_rows = checkpoints.len();
    let Some(height) = padded_height(valid_rows, required_height) else {
        return Ok(None);
    };
    let width = MultiConstraintCompletionCols::<F>::width();
    let mut values = vec![F::ZERO; height * width];
    for proof_idx in 0..valid_rows {
        let checkpoint = checkpoints[proof_idx];
        if checkpoint.sample_count == 0 {
            return Err(MultiConstraintWhirError::ProofShape(
                "terminal checkpoint is not a squeeze row",
            ));
        }
        let cols: &mut MultiConstraintCompletionCols<F> =
            values[proof_idx * width..(proof_idx + 1) * width].borrow_mut();
        cols.is_enabled = F::ONE;
        cols.proof_idx = F::from_usize(proof_idx);
        cols.is_first_in_proof = F::ONE;
        cols.class_index = F::from_usize(class_index);
        cols.end_tidx = F::from_usize(checkpoint.end_tidx);
        cols.sample_count = F::from_usize(checkpoint.sample_count);
        cols.state = checkpoint.state;
        copy_ext(&mut cols.final_aggregate, final_aggregates[proof_idx]);
        copy_ext(&mut cols.final_claim, final_claims[proof_idx]);
    }
    Ok(Some(RowMajorMatrix::new(values, width)))
}

/// Build all CPU-generated small traces.  Merkle, opened-value, query and
/// folding traces remain the existing WHIR chips and are intentionally absent.
pub fn generate_multi_constraint_small_traces(
    params: &SystemParams,
    profile: &MultiConstraintWhirProfile,
    statements: &[MultiConstraintWhirStatement],
    preflights: &[MultiConstraintWhirTranscriptPreflight],
    proofs: &[&WhirProof<BabyBearPoseidon2Config>],
    required_heights: Option<[usize; 5]>,
) -> Result<[RowMajorMatrix<F>; 5], MultiConstraintWhirError> {
    if statements.len() != preflights.len() || statements.len() != proofs.len() {
        return Err(MultiConstraintWhirError::ProofShape(
            "small trace batch length mismatch",
        ));
    }
    let mut derived = Vec::with_capacity(statements.len());
    let mut alphas = Vec::with_capacity(statements.len());
    let mut mus = Vec::with_capacity(statements.len());
    let mut final_polys = Vec::with_capacity(statements.len());
    let mut final_tidx = Vec::with_capacity(statements.len());
    let folding_pow_offset = pow_tidx_count(params.whir.folding_pow_bits);
    let final_round = params
        .num_whir_rounds()
        .checked_sub(1)
        .ok_or(MultiConstraintWhirError::ProofShape("zero WHIR rounds"))?;
    for i in 0..statements.len() {
        let data = derive_multi_constraint_whir_data(
            profile,
            &statements[i],
            preflights[i].mu,
            &preflights[i].alphas,
            &proofs[i].final_poly,
        )?;
        final_tidx.push(
            preflights[i].whir_round_tidx_per_round[final_round]
                + params.k_whir() * (3 * D_EF + folding_pow_offset),
        );
        derived.push(data);
        alphas.push(preflights[i].alphas.clone());
        mus.push(preflights[i].mu);
        final_polys.push(proofs[i].final_poly.clone());
    }
    let h = required_heights.unwrap_or([0; 5]);
    let req = |i: usize| required_heights.map(|_| h[i]);
    let prefix = generate_multi_constraint_prefix_trace(
        preflights,
        statements,
        &derived,
        proofs,
        profile.constraint_count,
        req(0),
    )?
    .ok_or(MultiConstraintWhirError::TraceSizeOverflow)?;
    let target = generate_multi_constraint_target_trace(profile, statements, &mus, req(1))?
        .ok_or(MultiConstraintWhirError::TraceSizeOverflow)?;
    let weight = generate_multi_constraint_weight_trace(profile, statements, &alphas, req(2))?
        .ok_or(MultiConstraintWhirError::TraceSizeOverflow)?;
    let final_poly = generate_multi_constraint_final_poly_trace(
        profile,
        statements,
        &derived,
        &final_polys,
        params.num_whir_sumcheck_rounds(),
        &final_tidx,
        req(3),
    )?
    .ok_or(MultiConstraintWhirError::TraceSizeOverflow)?;
    let aggregate = generate_multi_constraint_final_aggregate_trace(
        &derived,
        &final_tidx,
        profile.constraint_count,
        req(4),
    )?
    .ok_or(MultiConstraintWhirError::TraceSizeOverflow)?;
    Ok([prefix, target, weight, final_poly, aggregate])
}

#[cfg(feature = "cuda")]
pub fn transport_multi_constraint_small_traces_to_device(
    traces: &[RowMajorMatrix<F>],
    device_ctx: &openvm_cuda_common::stream::GpuDeviceCtx,
) -> Result<
    Vec<openvm_stark_backend::prover::AirProvingContext<openvm_cuda_backend::GpuBackend>>,
    openvm_cuda_common::error::MemCopyError,
> {
    use openvm_cuda_backend::data_transporter::transport_matrix_h2d_row;
    traces
        .iter()
        .map(|trace| {
            transport_matrix_h2d_row(trace, device_ctx).map(|matrix| {
                openvm_stark_backend::prover::AirProvingContext::simple_no_pis(matrix)
            })
        })
        .collect()
}

use core::borrow::BorrowMut;

use openvm_stark_backend::warp_accum::{CoefficientSumcheckVerification, NativeSumcheckKind};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::dense::RowMajorMatrix;

use super::NativeCoefficientSumcheckCols;

pub fn generate_native_coefficient_sumcheck_trace(
    proof_idx: usize,
    transcript_id: usize,
    verifications: &[&CoefficientSumcheckVerification<EF>],
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if verifications
        .iter()
        .any(|verification| !verification.has_valid_claim_chain())
    {
        return None;
    }
    let valid_rows = verifications
        .iter()
        .map(|verification| {
            verification
                .rounds
                .iter()
                .map(|round| round.coefficients.len())
                .sum::<usize>()
        })
        .sum::<usize>();
    if valid_rows == 0 {
        return None;
    }
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeCoefficientSumcheckCols::<F>::width();
    let mut trace = F::zero_vec(height * width);
    let mut row_index = 0usize;
    for verification in verifications {
        let kind = match verification.kind {
            NativeSumcheckKind::TwinConstraint => 0,
            NativeSumcheckKind::MultilinearBatching => 1,
        };
        let mut pre_claim = verification.initial_claim;
        for (round, &challenge) in verification.rounds.iter().zip(&verification.point) {
            let mut at_one_acc = EF::ZERO;
            let mut challenge_power = EF::ONE;
            let mut evaluation_acc = EF::ZERO;
            let at_zero = round.coefficients[0];
            for (coefficient_index, &coefficient) in round.coefficients.iter().enumerate() {
                at_one_acc += coefficient;
                if coefficient_index == 0 {
                    evaluation_acc = coefficient;
                } else {
                    challenge_power *= challenge;
                    evaluation_acc += coefficient * challenge_power;
                }
                let row = &mut trace[row_index * width..(row_index + 1) * width];
                let cols: &mut NativeCoefficientSumcheckCols<F> = row.borrow_mut();
                cols.active = F::ONE;
                if proof_idx != transcript_id {
                    return None;
                }
                cols.proof_idx = F::from_usize(proof_idx);
                cols.kind = F::from_usize(kind);
                cols.round = F::from_u32(round.round);
                cols.coefficient_index = F::from_usize(coefficient_index);
                cols.is_first_coefficient = F::from_bool(coefficient_index == 0);
                cols.is_last_coefficient =
                    F::from_bool(coefficient_index + 1 == round.coefficients.len());
                cols.is_initial_round = F::from_bool(round.round == 0);
                cols.is_final_round =
                    F::from_bool(round.round as usize + 1 == verification.rounds.len());
                cols.first_inverse = if coefficient_index == 0 {
                    F::ZERO
                } else {
                    F::from_usize(coefficient_index).inverse()
                };
                let distance = round.coefficients.len() - 1 - coefficient_index;
                cols.last_inverse = if distance == 0 {
                    F::ZERO
                } else {
                    F::from_usize(distance).inverse()
                };
                cols.round_inverse = if round.round == 0 {
                    F::ZERO
                } else {
                    F::from_u32(round.round).inverse()
                };
                let final_distance = verification.rounds.len() - 1 - round.round as usize;
                cols.final_round_inverse = if final_distance == 0 {
                    F::ZERO
                } else {
                    F::from_usize(final_distance).inverse()
                };
                cols.tidx = F::from_usize(round.transcript_span.operation_range.start);
                cols.coefficient
                    .copy_from_slice(coefficient.as_basis_coefficients_slice());
                cols.challenge
                    .copy_from_slice(challenge.as_basis_coefficients_slice());
                cols.pre_claim
                    .copy_from_slice(pre_claim.as_basis_coefficients_slice());
                cols.at_zero
                    .copy_from_slice(at_zero.as_basis_coefficients_slice());
                cols.at_one_acc
                    .copy_from_slice(at_one_acc.as_basis_coefficients_slice());
                cols.challenge_power
                    .copy_from_slice(challenge_power.as_basis_coefficients_slice());
                cols.evaluation_acc
                    .copy_from_slice(evaluation_acc.as_basis_coefficients_slice());
                row_index += 1;
            }
            pre_claim = evaluation_acc;
        }
        debug_assert_eq!(pre_claim, verification.final_claim);
    }
    debug_assert_eq!(row_index, valid_rows);
    Some(RowMajorMatrix::new(trace, width))
}

const _: () = assert!(D_EF == 4);

#[cfg(test)]
mod tests {
    use openvm_stark_backend::warp_accum::{
        CoefficientSumcheckRoundRecord, NativeTranscriptPhase, NativeTranscriptPhaseSpan,
    };

    use super::*;

    fn span(round: u32) -> NativeTranscriptPhaseSpan {
        NativeTranscriptPhaseSpan {
            phase: NativeTranscriptPhase::TwinSumcheckRound { round },
            event_range: 0..0,
            operation_range: (round as usize * 16)..(round as usize * 16 + 16),
            permutation_range: 0..0,
        }
    }

    fn compact_verification() -> CoefficientSumcheckVerification<EF> {
        CoefficientSumcheckVerification {
            kind: NativeSumcheckKind::TwinConstraint,
            degree: 2,
            initial_claim: EF::from_u32(7),
            rounds: vec![
                CoefficientSumcheckRoundRecord {
                    round: 0,
                    transcript_span: span(0),
                    coefficients: [1, 2, 3].map(EF::from_u32).to_vec(),
                },
                CoefficientSumcheckRoundRecord {
                    round: 1,
                    transcript_span: span(1),
                    // 4 + (4 + 5 + 21) = 34, the first round's value at 3.
                    coefficients: [4, 5, 21].map(EF::from_u32).to_vec(),
                },
            ],
            point: [3, 2].map(EF::from_u32).to_vec(),
            final_claim: EF::from_u32(98),
        }
    }

    #[test]
    fn compact_sumcheck_wire_record_derives_and_checks_claim_chain() {
        let verification = compact_verification();
        assert!(verification.has_valid_claim_chain());
        assert_eq!(verification.pre_claim_at_round(0), Some(EF::from_u32(7)));
        assert_eq!(verification.pre_claim_at_round(1), Some(EF::from_u32(34)));
        assert!(generate_native_coefficient_sumcheck_trace(0, 0, &[&verification], None).is_some());

        let mut malformed = verification.clone();
        malformed.final_claim += EF::ONE;
        assert!(!malformed.has_valid_claim_chain());
        assert!(generate_native_coefficient_sumcheck_trace(0, 0, &[&malformed], None).is_none());

        let mut malformed = verification.clone();
        malformed.rounds[1].coefficients[2] += EF::ONE;
        assert!(!malformed.has_valid_claim_chain());

        let mut malformed = verification.clone();
        malformed.rounds[1].round = 0;
        assert!(!malformed.has_valid_claim_chain());

        let mut malformed = verification;
        malformed.point.pop();
        assert!(!malformed.has_valid_claim_chain());
    }
}

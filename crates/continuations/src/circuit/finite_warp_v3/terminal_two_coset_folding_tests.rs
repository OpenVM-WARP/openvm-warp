use core::borrow::{Borrow, BorrowMut};
use std::panic::{catch_unwind, AssertUnwindSafe};

use openvm_recursion_circuit::native_warp::terminal::{
    NativeTerminalWhirAlphaBus, NativeTerminalWhirFoldingBus, NativeTerminalWhirFoldingCols,
};
use openvm_stark_backend::{
    air_builders::debug::check_constraints,
    p3_field::PrimeCharacteristicRing,
    verifier::whir::binary_k_fold,
    warp_accum::{
        terminal_whir::{terminal_whir_query_root, TerminalWhirLayout},
        BinaryMerkleMultiproofRecord, RsAdjointEvalVerification, TerminalDescriptor,
        TerminalWhirRoundVerification, TerminalWhirTranscriptPhase,
        TerminalWhirTranscriptPhaseSpan, WhirInitialRsWarpCode,
    },
    StarkProtocolConfig, SystemParams, TranscriptCheckpoint, WhirConfig, WhirProximityStrategy,
    WhirRoundConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config as SC, D_EF};
use p3_matrix::Matrix;

use super::*;

type Code = WhirInitialRsWarpCode<
    <SC as StarkProtocolConfig>::Hasher,
    openvm_stark_backend::warp_accum::FieldElementDigestObserver,
>;

fn digest(seed: u32) -> Digest {
    core::array::from_fn(|index| F::from_u32(seed + index as u32))
}

fn fixture() -> FiniteWarpV3TwoCosetTerminalProfile {
    let config = SC::default_from_params(SystemParams::new_for_testing(10));
    let code: Code =
        WhirInitialRsWarpCode::try_new_coefficient_two_coset(config.hasher().clone(), 8, 1, 0, 16)
            .unwrap();
    let whir = WhirConfig {
        k: FINITE_WARP_V3_TWO_COSET_K,
        rounds: vec![WhirRoundConfig { num_queries: 1 }],
        mu_pow_bits: 0,
        query_phase_pow_bits: 0,
        folding_pow_bits: 0,
        proximity: WhirProximityStrategy::UniqueDecoding,
    };
    let descriptor = TerminalDescriptor::from_whir_initial_rs(digest(10), &code, &whir, D_EF);
    FiniteWarpV3TwoCosetTerminalProfile::new(&descriptor, &code, &whir).unwrap()
}

fn span(phase: TerminalWhirTranscriptPhase) -> TerminalWhirTranscriptPhaseSpan {
    TerminalWhirTranscriptPhaseSpan {
        phase,
        start: TranscriptCheckpoint::default(),
        end: TranscriptCheckpoint::default(),
    }
}

fn verification(
    profile: &FiniteWarpV3TwoCosetTerminalProfile,
    seed: u32,
) -> TerminalWhirVerification<F, EF, Digest> {
    let merkle_index = 3u32;
    let raw_root = terminal_whir_query_root::<F>(
        TerminalWhirLayout::ScalarCoefficientTwoCoset,
        true,
        merkle_index as usize,
        profile.alpha_len(),
        FINITE_WARP_V3_TWO_COSET_K,
    )
    .unwrap();
    let values = (0..16)
        .map(|index| EF::from(F::from_u32(seed + index as u32 * 13 + 1)))
        .collect::<Vec<_>>();
    let alphas = [3, 7, 19, 31]
        .into_iter()
        .map(|value| EF::from(F::from_u32(seed + value)))
        .collect::<Vec<_>>();
    let folded = binary_k_fold::<F, EF>(values.clone(), &alphas, raw_root);
    let query_digest = digest(70);
    let round = TerminalWhirRoundVerification {
        round: 0,
        transcript_span: span(TerminalWhirTranscriptPhase::QueryPhase { round: 0 }),
        commitment: digest(50),
        log_rs_domain_size: profile.alpha_len() as u32,
        alphas,
        sumcheck_rounds: Vec::new(),
        query_indices: vec![merkle_index],
        query_roots: vec![raw_root.exp_power_of_2(FINITE_WARP_V3_TWO_COSET_K)],
        folded_values: vec![folded],
        opened_rows: vec![values.into_iter().map(|value| vec![value]).collect()],
        query_digests: vec![query_digest],
        multiproof: BinaryMerkleMultiproofRecord {
            expected_root: digest(50),
            depth: (profile.alpha_len() - FINITE_WARP_V3_TWO_COSET_K) as u32,
            leaf_indices: vec![merkle_index],
            leaf_digests: vec![query_digest],
            compressions: Vec::new(),
            consumed_siblings: 0,
        },
        ood_point: None,
        ood_value: None,
        gamma: EF::ONE,
        pre_round_claim: EF::ZERO,
        post_round_claim: folded,
    };
    TerminalWhirVerification {
        transcript_start: TranscriptCheckpoint::default(),
        descriptor_span: span(TerminalWhirTranscriptPhase::Descriptor),
        batching_challenge_span: span(TerminalWhirTranscriptPhase::BatchingChallenge),
        transcript_end: TranscriptCheckpoint::default(),
        root: digest(50),
        batching_challenge: EF::ONE,
        initial_claim: EF::ZERO,
        rounds: vec![round],
        final_poly: vec![EF::ZERO],
        final_weight_evals: vec![EF::ZERO],
        final_weight_span: span(TerminalWhirTranscriptPhase::FinalWeight),
        suffix_point: Vec::new(),
        accumulator_adjoint: RsAdjointEvalVerification {
            transcript_start: TranscriptCheckpoint::default(),
            transcript_end: TranscriptCheckpoint::default(),
            claimed_value: EF::ZERO,
            degree: 0,
            rounds: Vec::new(),
            point: Vec::new(),
            final_claim: EF::ZERO,
            expected_final: EF::ZERO,
        },
        expected_weight: EF::ZERO,
        actual_weight: EF::ZERO,
        final_inner_product: EF::ZERO,
        final_claim: EF::ZERO,
    }
}

fn air(
    profile: &FiniteWarpV3TwoCosetTerminalProfile,
) -> FiniteWarpV3TwoCosetTerminalWhirFoldingAir {
    FiniteWarpV3TwoCosetTerminalWhirFoldingAir::new(
        profile,
        NativeTerminalWhirAlphaBus::new(900),
        NativeTerminalWhirFoldingBus::new(901),
    )
    .unwrap()
}

#[test]
fn recorded_binary_coset_folds_match_native_reference_for_fresh_cases() {
    for seed in 1..=8u32 {
        let values = (0..16)
            .map(|index| EF::from(F::from_u32(seed * 101 + index as u32 * 17)))
            .collect::<Vec<_>>();
        let alphas = (0..4)
            .map(|index| EF::from(F::from_u32(seed * 37 + index as u32 * 11 + 1)))
            .collect::<Vec<_>>();
        let root = F::GENERATOR * F::from_u32(seed + 1);
        let expected = binary_k_fold::<F, EF>(values.clone(), &alphas, root);
        let mut actual_values = values;
        let mut records = Vec::new();
        let actual =
            record_binary_k_fold(&mut actual_values, &alphas, root, 0, 0, &mut records).unwrap();
        assert_eq!(actual, expected, "seed {seed}");
        assert_eq!(records.len(), 15);
    }
}

#[test]
fn folding_trace_satisfies_binary_air_and_mutation_is_rejected() {
    let profile = fixture();
    let verification = verification(&profile, 41);
    let trace = generate_finite_warp_v3_two_coset_terminal_whir_folding_trace(
        &profile,
        &verification,
        None,
    )
    .unwrap();
    assert_eq!(trace.height(), 16);
    let root_rows = trace
        .values
        .chunks_exact(trace.width())
        .filter(|row| {
            let cols: &NativeTerminalWhirFoldingCols<F> = (*row).borrow();
            cols.is_root == F::ONE
        })
        .count();
    assert_eq!(root_rows, 1);
    check_constraints::<_, SC>(
        &air(&profile),
        "FiniteWarpV3TwoCosetTerminalWhirFoldingAir",
        &None,
        &[trace.as_view()],
        &[],
    );

    let mut mutated = trace;
    let width = mutated.width();
    let first: &mut NativeTerminalWhirFoldingCols<F> = mutated.values[..width].borrow_mut();
    first.value[0] += F::ONE;
    let rejected = catch_unwind(AssertUnwindSafe(|| {
        check_constraints::<_, SC>(
            &air(&profile),
            "FiniteWarpV3TwoCosetTerminalWhirFoldingAir",
            &None,
            &[mutated.as_view()],
            &[],
        );
    }));
    assert!(rejected.is_err(), "mutated binary fold was accepted");
}

#[test]
fn malformed_folding_inputs_return_errors_without_panicking() {
    let profile = fixture();
    let valid = verification(&profile, 47);
    let mut cases = Vec::new();
    let mut short_alphas = valid.clone();
    short_alphas.rounds[0].alphas.pop();
    cases.push((
        short_alphas,
        FiniteWarpV3TwoCosetFoldingError::VerificationShape,
    ));
    let mut short_opening = valid.clone();
    short_opening.rounds[0].opened_rows[0].pop();
    cases.push((
        short_opening,
        FiniteWarpV3TwoCosetFoldingError::OpeningShape,
    ));
    let mut wide_scalar = valid.clone();
    wide_scalar.rounds[0].opened_rows[0][0].push(EF::ONE);
    cases.push((wide_scalar, FiniteWarpV3TwoCosetFoldingError::OpeningShape));
    let mut wrong_root = valid.clone();
    wrong_root.rounds[0].query_roots[0] += F::ONE;
    cases.push((wrong_root, FiniteWarpV3TwoCosetFoldingError::QueryRoot));
    let mut wrong_fold = valid.clone();
    wrong_fold.rounds[0].folded_values[0] += EF::ONE;
    cases.push((wrong_fold, FiniteWarpV3TwoCosetFoldingError::FoldedClaim));

    for (verification, expected) in cases {
        let result = catch_unwind(AssertUnwindSafe(|| {
            generate_finite_warp_v3_two_coset_terminal_whir_folding_trace(
                &profile,
                &verification,
                None,
            )
        }));
        assert_eq!(result.unwrap().unwrap_err(), expected);
    }
    let too_short = catch_unwind(AssertUnwindSafe(|| {
        generate_finite_warp_v3_two_coset_terminal_whir_folding_trace(&profile, &valid, Some(14))
    }));
    assert_eq!(
        too_short.unwrap().unwrap_err(),
        FiniteWarpV3TwoCosetFoldingError::TraceHeight
    );
}

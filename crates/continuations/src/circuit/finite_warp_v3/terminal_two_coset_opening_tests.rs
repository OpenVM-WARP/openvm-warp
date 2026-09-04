use core::borrow::BorrowMut;
use std::panic::{catch_unwind, AssertUnwindSafe};

use openvm_recursion_circuit::native_warp::{
    NativeLeafValueBus, NativeMerkleRootBus, NativeOpeningLeafBus, NativeTerminalWhirFoldingBus,
    NativeTerminalWhirOpenedCols, NativeTerminalWhirQueryBus,
};
use openvm_stark_backend::{
    air_builders::debug::check_constraints,
    hasher::MerkleHasher,
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

use super::*;

type Code = WhirInitialRsWarpCode<
    <SC as StarkProtocolConfig>::Hasher,
    openvm_stark_backend::warp_accum::FieldElementDigestObserver,
>;

fn digest(seed: u32) -> Digest {
    core::array::from_fn(|index| F::from_u32(seed + index as u32))
}

fn fixture() -> (SC, FiniteWarpV3TwoCosetTerminalProfile) {
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
    let profile = FiniteWarpV3TwoCosetTerminalProfile::new(&descriptor, &code, &whir).unwrap();
    (config, profile)
}

fn span(phase: TerminalWhirTranscriptPhase, operations: usize) -> TerminalWhirTranscriptPhaseSpan {
    TerminalWhirTranscriptPhaseSpan {
        phase,
        start: TranscriptCheckpoint::default(),
        end: TranscriptCheckpoint {
            operations,
            ..Default::default()
        },
    }
}

fn verification(
    config: &SC,
    profile: &FiniteWarpV3TwoCosetTerminalProfile,
    seed: u32,
) -> (
    TerminalWhirVerification<F, EF, Digest>,
    TranscriptLog<F, [F; 16]>,
) {
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
        .map(|index| EF::from(F::from_u32(seed + index as u32 * 7 + 1)))
        .collect::<Vec<_>>();
    let alphas = [2, 5, 11, 17]
        .into_iter()
        .map(|value| EF::from(F::from_u32(seed + value)))
        .collect::<Vec<_>>();
    let folded = binary_k_fold::<F, EF>(values.clone(), &alphas, raw_root);
    let leaf_digests = values
        .iter()
        .map(|value| {
            config
                .hasher()
                .hash_slice(value.as_basis_coefficients_slice())
        })
        .collect::<Vec<_>>();
    let query_digest = config.hasher().tree_compress(leaf_digests);
    let query_span = span(
        TerminalWhirTranscriptPhase::QueryPhase { round: 0 },
        D_EF + 1,
    );
    let round = TerminalWhirRoundVerification {
        round: 0,
        transcript_span: query_span.clone(),
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
    let verification = TerminalWhirVerification {
        transcript_start: TranscriptCheckpoint::default(),
        descriptor_span: span(TerminalWhirTranscriptPhase::Descriptor, 0),
        batching_challenge_span: span(TerminalWhirTranscriptPhase::BatchingChallenge, 0),
        transcript_end: query_span.end,
        root: digest(50),
        batching_challenge: EF::ONE,
        initial_claim: EF::ZERO,
        rounds: vec![round],
        final_poly: vec![EF::ZERO],
        final_weight_evals: vec![EF::ZERO],
        final_weight_span: span(TerminalWhirTranscriptPhase::FinalWeight, 0),
        suffix_point: Vec::new(),
        accumulator_adjoint: adjoint,
        expected_weight: EF::ZERO,
        actual_weight: EF::ZERO,
        final_inner_product: EF::ZERO,
        final_claim: EF::ZERO,
    };
    let mut transcript_values = vec![F::ZERO; D_EF + 1];
    transcript_values[0] = F::from_u32(merkle_index);
    let mut samples = vec![false; D_EF + 1];
    samples[0] = true;
    (verification, TranscriptLog::new(transcript_values, samples))
}

fn air(profile: &FiniteWarpV3TwoCosetTerminalProfile) -> FiniteWarpV3TwoCosetTerminalWhirOpenedAir {
    FiniteWarpV3TwoCosetTerminalWhirOpenedAir::new(
        profile,
        NativeTerminalWhirQueryBus::new(800),
        NativeTerminalWhirFoldingBus::new(801),
        NativeLeafValueBus::new(802),
        NativeOpeningLeafBus::new(803),
        NativeMerkleRootBus::new(804),
    )
    .unwrap()
}

#[test]
fn scalar_opening_uses_sixteen_individual_leaves_and_depth_four_root() {
    let (config, profile) = fixture();
    let (verification, transcript) = verification(&config, &profile, 19);
    let generated = generate_finite_warp_v3_two_coset_terminal_whir_opened_trace(
        &profile,
        config.hasher(),
        &verification,
        &transcript,
        10,
        20,
        None,
    )
    .unwrap();
    assert_eq!(generated.matrix.height(), 16);
    assert_eq!(generated.leaves.len(), 16);
    assert!(generated
        .leaves
        .iter()
        .enumerate()
        .all(|(index, leaf)| leaf.leaf_index as usize == index
            && leaf.values.len() == D_EF
            && leaf.lookup_counts == vec![1; D_EF]));
    assert_eq!(generated.inner_merkle.len(), 1);
    assert_eq!(generated.inner_merkle[0].1.depth, 4);
    assert_eq!(
        generated.inner_merkle[0].1.expected_root,
        verification.rounds[0].query_digests[0]
    );

    check_constraints::<_, SC>(
        &air(&profile),
        "FiniteWarpV3TwoCosetTerminalWhirOpenedAir",
        &None,
        &[generated.matrix.as_view()],
        &[],
    );
}

#[test]
fn opening_air_rejects_mutated_scalar_twiddle_recurrence() {
    let (config, profile) = fixture();
    let (verification, transcript) = verification(&config, &profile, 23);
    let mut matrix = generate_finite_warp_v3_two_coset_terminal_whir_opened_trace(
        &profile,
        config.hasher(),
        &verification,
        &transcript,
        10,
        20,
        None,
    )
    .unwrap()
    .matrix;
    let width = matrix.width();
    let second: &mut NativeTerminalWhirOpenedCols<F> = matrix.values[width..2 * width].borrow_mut();
    second.twiddle += F::ONE;
    let rejected = catch_unwind(AssertUnwindSafe(|| {
        check_constraints::<_, SC>(
            &air(&profile),
            "FiniteWarpV3TwoCosetTerminalWhirOpenedAir",
            &None,
            &[matrix.as_view()],
            &[],
        );
    }));
    assert!(rejected.is_err(), "mutated scalar-leaf order was accepted");
}

#[test]
fn malformed_openings_transcript_and_roots_return_errors_without_panicking() {
    let (config, profile) = fixture();
    let (verification, transcript) = verification(&config, &profile, 29);

    let mut cases = Vec::new();
    let mut short_coset = verification.clone();
    short_coset.rounds[0].opened_rows[0].pop();
    cases.push((
        short_coset,
        transcript.clone(),
        FiniteWarpV3TwoCosetOpeningError::OpeningShape,
    ));
    let mut wide_scalar = verification.clone();
    wide_scalar.rounds[0].opened_rows[0][0].push(EF::ONE);
    cases.push((
        wide_scalar,
        transcript.clone(),
        FiniteWarpV3TwoCosetOpeningError::OpeningShape,
    ));
    let mut wrong_root = verification.clone();
    wrong_root.rounds[0].query_roots[0] += F::ONE;
    cases.push((
        wrong_root,
        transcript.clone(),
        FiniteWarpV3TwoCosetOpeningError::QueryRoot,
    ));
    let mut wrong_digest = verification.clone();
    wrong_digest.rounds[0].query_digests[0][0] += F::ONE;
    cases.push((
        wrong_digest,
        transcript.clone(),
        FiniteWarpV3TwoCosetOpeningError::MerkleRoot,
    ));
    cases.push((
        verification.clone(),
        TranscriptLog::new(Vec::new(), Vec::new()),
        FiniteWarpV3TwoCosetOpeningError::Transcript,
    ));
    let mut wrong_kind = transcript.clone();
    wrong_kind.samples_mut()[0] = false;
    cases.push((
        verification.clone(),
        wrong_kind,
        FiniteWarpV3TwoCosetOpeningError::Transcript,
    ));

    for (verification, transcript, expected) in cases {
        let result = catch_unwind(AssertUnwindSafe(|| {
            generate_finite_warp_v3_two_coset_terminal_whir_opened_trace(
                &profile,
                config.hasher(),
                &verification,
                &transcript,
                10,
                20,
                None,
            )
        }));
        assert_eq!(result.unwrap().unwrap_err(), expected);
    }

    let too_short = catch_unwind(AssertUnwindSafe(|| {
        generate_finite_warp_v3_two_coset_terminal_whir_opened_trace(
            &profile,
            config.hasher(),
            &verification,
            &transcript,
            10,
            20,
            Some(15),
        )
    }));
    assert_eq!(
        too_short.unwrap().unwrap_err(),
        FiniteWarpV3TwoCosetOpeningError::TraceHeight
    );
}

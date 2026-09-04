use std::panic::{catch_unwind, AssertUnwindSafe};

use openvm_stark_backend::{
    p3_field::PrimeCharacteristicRing,
    warp_accum::{
        terminal_whir::{terminal_whir_query_root, TerminalWhirLayout},
        TerminalDescriptor, WhirInitialRsWarpCode,
    },
    StarkProtocolConfig, SystemParams, WhirConfig, WhirProximityStrategy, WhirRoundConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config as SC;

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
        rounds: vec![
            WhirRoundConfig { num_queries: 2 },
            WhirRoundConfig { num_queries: 2 },
        ],
        mu_pow_bits: 0,
        query_phase_pow_bits: 0,
        folding_pow_bits: 0,
        proximity: WhirProximityStrategy::UniqueDecoding,
    };
    let descriptor = TerminalDescriptor::from_whir_initial_rs(digest(10), &code, &whir, D_EF);
    FiniteWarpV3TwoCosetTerminalProfile::new(&descriptor, &code, &whir).unwrap()
}

#[test]
fn terminal_query_roots_differentially_match_native_initial_and_later_rounds() {
    let profile = fixture();
    for round in 0..2 {
        let query_bits = profile.alpha_len() - FINITE_WARP_V3_TWO_COSET_K - round;
        for index in [0, 1, 2, 3, (1usize << query_bits) - 1] {
            let (root, squares) =
                two_coset_query_root_and_squares(&profile, round, index as u32).unwrap();
            let native = terminal_whir_query_root::<F>(
                TerminalWhirLayout::ScalarCoefficientTwoCoset,
                round == 0,
                index,
                profile.alpha_len() - round,
                FINITE_WARP_V3_TWO_COSET_K,
            )
            .unwrap();
            assert_eq!(root, native);
            assert_eq!(
                squares[FINITE_WARP_V3_TWO_COSET_K - 1],
                native.exp_power_of_2(FINITE_WARP_V3_TWO_COSET_K)
            );
        }
    }

    let (even, _) = two_coset_query_root_and_squares(&profile, 0, 0).unwrap();
    let (odd, _) = two_coset_query_root_and_squares(&profile, 0, 1).unwrap();
    assert_eq!(even, F::ONE);
    assert_eq!(odd, F::GENERATOR);
}

#[test]
fn mutated_query_sample_is_rejected() {
    let profile = fixture();
    let bits = profile.alpha_len() - FINITE_WARP_V3_TWO_COSET_K;
    let mut log = TranscriptLog::new(vec![F::from_u32(3)], vec![true]);
    assert_eq!(checked_query_sample(&log, 0, bits, 3), Ok(F::from_u32(3)));
    log.values_mut()[0] += F::ONE;
    assert_eq!(
        checked_query_sample(&log, 0, bits, 3),
        Err(FiniteWarpV3TwoCosetQueryError::QueryIndex)
    );
}

#[test]
fn truncated_or_wrong_kind_query_sample_is_rejected_without_panicking() {
    let profile = fixture();
    let bits = profile.alpha_len() - FINITE_WARP_V3_TWO_COSET_K;
    for log in [
        TranscriptLog::new(Vec::new(), Vec::new()),
        TranscriptLog::new(vec![F::from_u32(3)], vec![false]),
    ] {
        let result = catch_unwind(AssertUnwindSafe(|| checked_query_sample(&log, 0, bits, 3)));
        assert_eq!(
            result.unwrap(),
            Err(FiniteWarpV3TwoCosetQueryError::Transcript)
        );
    }
}

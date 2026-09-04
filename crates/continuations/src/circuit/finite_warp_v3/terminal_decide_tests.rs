use core::{
    any::TypeId,
    borrow::{Borrow, BorrowMut},
};
use std::panic::AssertUnwindSafe;

use openvm_stark_backend::{
    air_builders::debug::check_constraints,
    native_warp::{
        native_accumulator_instance_digest, FixedMultiAirCompletePesatIndex,
        FixedMultiAirCompleteTerminalProof, FixedMultiAirPesatIndex, FixedMultiAirTerminalProof,
    },
    p3_field::PrimeCharacteristicRing,
    p3_matrix::Matrix,
    warp_accum::{
        RsAdjointEvalVerification, RsAdjointSumcheckRoundRecord,
        TerminalConstrainedLinearizerProof, TerminalWhirTranscriptPhase,
        TerminalWhirTranscriptPhaseSpan,
    },
    SystemParams, TranscriptCheckpoint,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as NativeSC, DIGEST_SIZE,
};

use super::*;

fn digest(seed: u32) -> Digest {
    core::array::from_fn(|index| F::from_u32(seed + index as u32))
}

fn instance(alpha_len: usize, beta_len: usize) -> AccumulatorInstance<EF, Digest> {
    AccumulatorInstance {
        rt: digest(100),
        alpha: (0..alpha_len)
            .map(|index| EF::from(F::from_usize(index + 1)))
            .collect(),
        mu: EF::from(F::from_u32(200)),
        beta: (0..beta_len)
            .map(|index| EF::from(F::from_usize(index + 300)))
            .collect(),
        eta: EF::from(F::from_u32(400)),
    }
}

#[test]
fn production_complete_terminal_types_cannot_alias_the_legacy_owner_types() {
    assert_ne!(
        TypeId::of::<FixedMultiAirCompleteTerminalProof<EF>>(),
        TypeId::of::<FixedMultiAirTerminalProof<EF>>()
    );
    assert_ne!(
        TypeId::of::<FixedMultiAirCompletePesatIndex<F, Digest>>(),
        TypeId::of::<FixedMultiAirPesatIndex<F, Digest>>()
    );
    assert_eq!(
        TypeId::of::<FiniteWarpV3CompleteNonlinearProof>(),
        TypeId::of::<TerminalConstrainedLinearizerProof<FixedMultiAirCompleteTerminalProof<EF>>>()
    );
}

#[test]
fn terminal_index_digest_binds_every_fixed_metadata_word() {
    let relation = digest(10);
    let metadata = [1, 2, 3, u64::MAX, 0xfeed_beef_dead_cafe];
    let expected = finite_warp_v3_terminal_index_digest(relation, &metadata);
    assert_ne!(expected, [F::ZERO; DIGEST_SIZE]);
    assert_eq!(
        expected,
        finite_warp_v3_terminal_index_digest(relation, &metadata)
    );

    let mut changed_relation = relation;
    changed_relation[3] += F::ONE;
    assert_ne!(
        expected,
        finite_warp_v3_terminal_index_digest(changed_relation, &metadata)
    );
    for index in 0..metadata.len() {
        let mut changed = metadata;
        changed[index] ^= 1;
        assert_ne!(
            expected,
            finite_warp_v3_terminal_index_digest(relation, &changed),
            "metadata word {index} was not bound"
        );
    }
    assert_ne!(
        expected,
        finite_warp_v3_terminal_index_digest(relation, &metadata[..metadata.len() - 1])
    );
}

#[test]
fn receipt_trace_uses_the_canonical_accumulator_digest() {
    let instance = instance(5, 7);
    let (trace, receipt) = finite_warp_v3_terminal_receipt_trace(&instance).unwrap();
    let config = NativeSC::default_from_params(SystemParams::new_for_testing(10));
    assert_eq!(
        receipt.final_accumulator_digest,
        native_accumulator_instance_digest(&config, &instance)
    );
    assert_eq!(receipt.final_accumulator_root, instance.rt);
    let row = trace.row_slice(0).unwrap();
    let cols: &NativeAccumulatorRootDigestCols<F> = (*row).borrow();
    assert_eq!(cols.instance_digest, receipt.final_accumulator_digest);
    assert_eq!(cols.root, receipt.final_accumulator_root);
    assert_eq!(cols.proof_idx, F::ZERO);
    assert_eq!(cols.active, F::ONE);
}

#[test]
fn receipt_trace_rejects_empty_accumulator_dimensions_without_panicking() {
    for (alpha_len, beta_len) in [(0, 3), (3, 0), (0, 0)] {
        assert!(matches!(
            finite_warp_v3_terminal_receipt_trace(&instance(alpha_len, beta_len)),
            Err(FiniteWarpV3TerminalDecideError::InvalidAccumulatorShape)
        ));
    }
}

#[test]
fn terminal_instance_state_machine_accepts_exact_order_and_rejects_mutations() {
    let instance = instance(5, 7);
    let layout = NativePrivateAccumulatorLayout::new(0, 5, 7);
    let digest_traces = generate_native_accumulator_digest_traces(0, &instance, &layout).unwrap();
    let air = FiniteWarpV3TerminalInstanceAir {
        layout,
        terminal_instance_bus: FixedMultiAirTerminalInstanceValueBus::new(700),
        digest_element_bus: NativeAccumulatorDigestElementBus::new(701),
    };
    check_constraints::<_, NativeSC>(
        &air,
        "FiniteWarpV3TerminalInstanceAir",
        &None,
        &[digest_traces.values.as_view()],
        &[],
    );

    let mut wrong_section = digest_traces.values.clone();
    let width = wrong_section.width();
    let second: &mut NativeAccumulatorValueCols<F> =
        wrong_section.values[width..2 * width].borrow_mut();
    second.section = [F::ZERO, F::ONE, F::ZERO, F::ZERO];
    let rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        check_constraints::<_, NativeSC>(
            &air,
            "FiniteWarpV3TerminalInstanceAir",
            &None,
            &[wrong_section.as_view()],
            &[],
        );
    }));
    assert!(
        rejected.is_err(),
        "out-of-order terminal section was accepted"
    );

    let mut wrong_ordinal = digest_traces.values;
    let width = wrong_ordinal.width();
    let second: &mut NativeAccumulatorValueCols<F> =
        wrong_ordinal.values[width..2 * width].borrow_mut();
    second.ordinal += F::ONE;
    let rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        check_constraints::<_, NativeSC>(
            &air,
            "FiniteWarpV3TerminalInstanceAir",
            &None,
            &[wrong_ordinal.as_view()],
            &[],
        );
    }));
    assert!(
        rejected.is_err(),
        "non-canonical digest ordinal was accepted"
    );
}

#[test]
fn receipt_air_rejects_noncanonical_activity_and_proof_index() {
    let instance = instance(5, 7);
    let (trace, _) = finite_warp_v3_terminal_receipt_trace(&instance).unwrap();
    let binding = FiniteWarpV3TerminalDecideBinding {
        protocol_digest: digest(10),
        terminal_index_digest: digest(20),
        verifier_component_digest: digest(30),
        warp_index_digest: digest(31),
        vacc_setup_digest: digest(32),
        schedule_digest: digest(33),
        warp_call_count: 3,
    };
    let air = FiniteWarpV3TerminalDecideReceiptAir {
        binding,
        relation_digest: digest(40),
        alpha_len: 5,
        beta_len: 7,
        receipt_bus: FiniteWarpV3TerminalReceiptBus::new(710),
        terminal_binding_bus: FixedMultiAirTerminalBindingBus::new(711),
        terminal_whir_root_bus: NativeTerminalAccumulatorRootBus::new(712),
        algebraic_digest_bus: NativeAccumulatorAlgebraicDigestBus::new(713),
        compress_bus: Poseidon2CompressBus::new(714),
        terminal_accumulator_link_bus: FiniteWarpV3TerminalAccumulatorLinkBus::new(715),
    };
    check_constraints::<_, NativeSC>(
        &air,
        "FiniteWarpV3TerminalDecideReceiptAir",
        &None,
        &[trace.as_view()],
        &[],
    );

    let mut wrong_index = trace.clone();
    let cols: &mut NativeAccumulatorRootDigestCols<F> =
        wrong_index.values.as_mut_slice().borrow_mut();
    cols.proof_idx = F::ONE;
    let rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        check_constraints::<_, NativeSC>(
            &air,
            "FiniteWarpV3TerminalDecideReceiptAir",
            &None,
            &[wrong_index.as_view()],
            &[],
        );
    }));
    assert!(
        rejected.is_err(),
        "nonzero terminal proof index was accepted"
    );

    let mut inactive = trace;
    let cols: &mut NativeAccumulatorRootDigestCols<F> = inactive.values.as_mut_slice().borrow_mut();
    cols.active = F::ZERO;
    let rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        check_constraints::<_, NativeSC>(
            &air,
            "FiniteWarpV3TerminalDecideReceiptAir",
            &None,
            &[inactive.as_view()],
            &[],
        );
    }));
    assert!(
        rejected.is_err(),
        "missing terminal receipt row was accepted"
    );
}

fn empty_span(phase: TerminalWhirTranscriptPhase) -> TerminalWhirTranscriptPhaseSpan {
    TerminalWhirTranscriptPhaseSpan {
        phase,
        start: TranscriptCheckpoint::default(),
        end: TranscriptCheckpoint::default(),
    }
}

fn exact_two_coset_adjoint_fixture(
    log_message_len: usize,
) -> (TerminalWhirVerification<F, EF, Digest>, Vec<EF>) {
    let alpha = (0..=log_message_len)
        .map(|index| EF::from(F::from_usize(2 * index + 3)))
        .collect::<Vec<_>>();
    let adjoint_point = (0..=log_message_len)
        .map(|index| EF::from(F::from_usize(3 * index + 11)))
        .collect::<Vec<_>>();
    let whir_point = (0..log_message_len)
        .map(|index| EF::from(F::from_usize(5 * index + 17)))
        .collect::<Vec<_>>();

    let ordered_alpha = core::iter::once(alpha[log_message_len])
        .chain(alpha[..log_message_len].iter().copied())
        .collect::<Vec<_>>();
    let q = ordered_alpha
        .iter()
        .zip(&adjoint_point)
        .map(|(&a, &r)| (EF::ONE - a) * (EF::ONE - r) + a * r)
        .product::<EF>();
    let mut y_powers = vec![EF::ONE; log_message_len];
    for &(message_bit, round, root) in &two_coset_adjoint_root_schedule(log_message_len).unwrap() {
        let r = adjoint_point[round];
        y_powers[message_bit] *= (EF::ONE - r) + r * EF::from(root);
    }
    let selector = whir_point
        .iter()
        .zip(&y_powers)
        .map(|(&z, &y)| (EF::ONE - z) + (z.double() - EF::ONE) * y)
        .product::<EF>();
    let final_claim = q * selector;
    let rounds = adjoint_point
        .iter()
        .enumerate()
        .map(|(round, &challenge)| RsAdjointSumcheckRoundRecord {
            round: round.try_into().unwrap(),
            transcript_start: TranscriptCheckpoint::default(),
            transcript_end: TranscriptCheckpoint::default(),
            pre_claim: EF::ZERO,
            evaluations: Vec::new(),
            challenge,
            post_claim: EF::ZERO,
        })
        .collect();
    let verification = TerminalWhirVerification {
        transcript_start: TranscriptCheckpoint::default(),
        descriptor_span: empty_span(TerminalWhirTranscriptPhase::Descriptor),
        batching_challenge_span: empty_span(TerminalWhirTranscriptPhase::BatchingChallenge),
        transcript_end: TranscriptCheckpoint::default(),
        root: digest(900),
        batching_challenge: EF::ONE,
        initial_claim: EF::ZERO,
        rounds: Vec::new(),
        final_poly: vec![EF::ZERO],
        final_weight_evals: vec![EF::ZERO],
        final_weight_span: empty_span(TerminalWhirTranscriptPhase::FinalWeight),
        suffix_point: whir_point,
        accumulator_adjoint: RsAdjointEvalVerification {
            transcript_start: TranscriptCheckpoint::default(),
            transcript_end: TranscriptCheckpoint::default(),
            claimed_value: EF::from(F::from_u32(123)),
            degree: (log_message_len + 1).try_into().unwrap(),
            rounds,
            point: adjoint_point,
            final_claim,
            expected_final: final_claim,
        },
        expected_weight: EF::ZERO,
        actual_weight: EF::ZERO,
        final_inner_product: EF::ZERO,
        final_claim: EF::ZERO,
    };
    (verification, alpha)
}

#[test]
fn coefficient_native_two_coset_adjoint_matches_the_exact_unaligned_schedule() {
    let log_message_len = 3;
    let (verification, alpha) = exact_two_coset_adjoint_fixture(log_message_len);
    let q_air = FiniteWarpV3TwoCosetRsAdjointQAir {
        round_bus: NativeTerminalRsAdjointRoundBus::new(800),
        accumulator_value_bus: NativeTerminalAccumulatorValueBus::new(801),
        q_bus: NativeTerminalRsAdjointQBus::new(802),
        log_message_len,
    };
    let y_air = FiniteWarpV3TwoCosetRsAdjointYAir::new(
        NativeTerminalRsAdjointRoundBus::new(800),
        NativeTerminalRsAdjointYBus::new(803),
        log_message_len,
    )
    .unwrap();
    let selector_air = FiniteWarpV3TwoCosetRsAdjointSelectorAir {
        point_bus: NativeTerminalWhirPointBus::new(804),
        q_bus: NativeTerminalRsAdjointQBus::new(802),
        y_bus: NativeTerminalRsAdjointYBus::new(803),
        claim_bus: NativeTerminalRsAdjointClaimBus::new(805),
        value_bus: NativeTerminalRsAdjointValueBus::new(806),
        log_message_len,
    };
    let q = generate_two_coset_adjoint_q_trace(&verification, &alpha, log_message_len).unwrap();
    let y = generate_two_coset_adjoint_y_trace(&y_air, &verification).unwrap();
    let selector =
        generate_two_coset_adjoint_selector_trace(&verification, &alpha, log_message_len).unwrap();

    check_constraints::<_, NativeSC>(
        &q_air,
        "coefficient-native two-coset adjoint q",
        &None,
        &[q.as_view()],
        &[],
    );
    check_constraints::<_, NativeSC>(
        &y_air,
        "coefficient-native two-coset adjoint y",
        &None,
        &[y.as_view()],
        &[],
    );
    check_constraints::<_, NativeSC>(
        &selector_air,
        "coefficient-native two-coset adjoint selector",
        &None,
        &[selector.as_view()],
        &[],
    );

    let selector_last_row = selector.row_slice(log_message_len - 1).unwrap();
    let selector_last: &NativeTerminalRsAdjointSelectorCols<F> = (*selector_last_row).borrow();
    assert_eq!(
        EF::from_basis_coefficients_slice(&selector_last.final_claim).unwrap(),
        verification.accumulator_adjoint.expected_final
    );
}

fn mutate_two_coset_y_root<const WIDTH: usize>(matrix: &mut RowMajorMatrix<F>, row: usize) {
    let width = NativeTerminalRsAdjointYCols::<F, WIDTH>::width();
    let cols: &mut NativeTerminalRsAdjointYCols<F, WIDTH> =
        matrix.values[row * width..(row + 1) * width].borrow_mut();
    cols.root += F::ONE;
}

#[test]
fn coefficient_native_two_coset_adjoint_rejects_coordinate_root_and_endpoint_mutations() {
    let log_message_len = 3;
    let (verification, alpha) = exact_two_coset_adjoint_fixture(log_message_len);
    let q_air = FiniteWarpV3TwoCosetRsAdjointQAir {
        round_bus: NativeTerminalRsAdjointRoundBus::new(810),
        accumulator_value_bus: NativeTerminalAccumulatorValueBus::new(811),
        q_bus: NativeTerminalRsAdjointQBus::new(812),
        log_message_len,
    };
    let y_air = FiniteWarpV3TwoCosetRsAdjointYAir::new(
        NativeTerminalRsAdjointRoundBus::new(810),
        NativeTerminalRsAdjointYBus::new(813),
        log_message_len,
    )
    .unwrap();
    let selector_air = FiniteWarpV3TwoCosetRsAdjointSelectorAir {
        point_bus: NativeTerminalWhirPointBus::new(814),
        q_bus: NativeTerminalRsAdjointQBus::new(812),
        y_bus: NativeTerminalRsAdjointYBus::new(813),
        claim_bus: NativeTerminalRsAdjointClaimBus::new(815),
        value_bus: NativeTerminalRsAdjointValueBus::new(816),
        log_message_len,
    };

    let mut wrong_coordinate =
        generate_two_coset_adjoint_q_trace(&verification, &alpha, log_message_len).unwrap();
    let q_width = NativeTerminalRsAdjointQCols::<F>::width();
    let second: &mut NativeTerminalRsAdjointQCols<F> =
        wrong_coordinate.values[q_width..2 * q_width].borrow_mut();
    second.alpha[0] += F::ONE;
    assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
        check_constraints::<_, NativeSC>(
            &q_air,
            "mutated two-coset adjoint coordinate",
            &None,
            &[wrong_coordinate.as_view()],
            &[],
        );
    }))
    .is_err());

    let mut wrong_root = generate_two_coset_adjoint_y_trace(&y_air, &verification).unwrap();
    match y_air.encoder.width() {
        1 => mutate_two_coset_y_root::<1>(&mut wrong_root, 1),
        2 => mutate_two_coset_y_root::<2>(&mut wrong_root, 1),
        3 => mutate_two_coset_y_root::<3>(&mut wrong_root, 1),
        4 => mutate_two_coset_y_root::<4>(&mut wrong_root, 1),
        5 => mutate_two_coset_y_root::<5>(&mut wrong_root, 1),
        6 => mutate_two_coset_y_root::<6>(&mut wrong_root, 1),
        7 => mutate_two_coset_y_root::<7>(&mut wrong_root, 1),
        8 => mutate_two_coset_y_root::<8>(&mut wrong_root, 1),
        9 => mutate_two_coset_y_root::<9>(&mut wrong_root, 1),
        10 => mutate_two_coset_y_root::<10>(&mut wrong_root, 1),
        11 => mutate_two_coset_y_root::<11>(&mut wrong_root, 1),
        _ => unreachable!(),
    }
    assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
        check_constraints::<_, NativeSC>(
            &y_air,
            "mutated two-coset adjoint root schedule",
            &None,
            &[wrong_root.as_view()],
            &[],
        );
    }))
    .is_err());

    let mut wrong_endpoint =
        generate_two_coset_adjoint_selector_trace(&verification, &alpha, log_message_len).unwrap();
    let selector_width = NativeTerminalRsAdjointSelectorCols::<F>::width();
    let last: &mut NativeTerminalRsAdjointSelectorCols<F> = wrong_endpoint.values
        [(log_message_len - 1) * selector_width..log_message_len * selector_width]
        .borrow_mut();
    last.final_claim[0] += F::ONE;
    assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
        check_constraints::<_, NativeSC>(
            &selector_air,
            "mutated two-coset adjoint endpoint",
            &None,
            &[wrong_endpoint.as_view()],
            &[],
        );
    }))
    .is_err());
}

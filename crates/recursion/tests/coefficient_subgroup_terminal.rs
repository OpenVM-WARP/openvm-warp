use core::borrow::{Borrow, BorrowMut};
use std::panic::{catch_unwind, AssertUnwindSafe};

use openvm_recursion_circuit::native_warp::terminal::{
    generate_native_terminal_rs_adjoint_q_trace,
    generate_native_terminal_rs_adjoint_selector_trace,
    generate_native_terminal_rs_adjoint_y_trace, generate_native_terminal_whir_statement_trace,
    validate_native_terminal_eq_statement, NativeTerminalEqClaimError,
    NativeTerminalRsAdjointClaimBus, NativeTerminalRsAdjointQAir, NativeTerminalRsAdjointQBus,
    NativeTerminalRsAdjointQCols, NativeTerminalRsAdjointRoundBus,
    NativeTerminalRsAdjointSelectorAir, NativeTerminalRsAdjointValueBus,
    NativeTerminalRsAdjointYAir, NativeTerminalRsAdjointYBus, NativeTerminalTranscriptBinding,
    NativeTerminalTranscriptBindingError, NativeTerminalWhirPointBus,
    NATIVE_TERMINAL_DESCRIPTOR_DOMAIN_TAG,
};
use openvm_stark_backend::{
    air_builders::debug::{check_constraints, DebugConstraintBuilder},
    transcript::{TranscriptCheckpoint, TranscriptLog},
    warp_accum::{
        terminal_whir::{TerminalWhirTranscriptPhase, TerminalWhirTranscriptPhaseSpan},
        RsAdjointEvalVerification, TerminalConstrainedCodeLayout, TerminalConstrainedRsStatement,
        TerminalDescriptor, TerminalWhirVerification, WhirInitialRsLayout, WhirInitialRsWarpCode,
    },
    warp_pesat::{TerminalStructuredLinearClaim, TerminalWeightSpec},
    BaseAirWithPublicValues, PartitionedBaseAir, StarkProtocolConfig, SystemParams, WhirConfig,
    WhirProximityStrategy, WhirRoundConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as SC, Digest, DIGEST_SIZE, D_EF, EF, F,
};
use p3_air::{Air, BaseAir};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

fn whir() -> WhirConfig {
    WhirConfig {
        k: 4,
        rounds: vec![WhirRoundConfig { num_queries: 1 }],
        mu_pow_bits: 0,
        query_phase_pow_bits: 0,
        folding_pow_bits: 0,
        proximity: WhirProximityStrategy::UniqueDecoding,
    }
}

fn coefficient_code(
    log_message_len: usize,
) -> WhirInitialRsWarpCode<<SC as StarkProtocolConfig>::Hasher> {
    let config = SC::default_from_params(SystemParams::new_for_testing(8));
    WhirInitialRsWarpCode::try_new_coefficient_subgroup(
        config.hasher().clone(),
        log_message_len,
        1,
        0,
        16,
    )
    .expect("coefficient-subgroup fixture")
}

fn checkpoint(operations: usize) -> TranscriptCheckpoint {
    TranscriptCheckpoint {
        operations,
        events: 0,
        permutations: 0,
    }
}

fn span(
    phase: TerminalWhirTranscriptPhase,
    start: usize,
    end: usize,
) -> TerminalWhirTranscriptPhaseSpan {
    TerminalWhirTranscriptPhaseSpan {
        phase,
        start: checkpoint(start),
        end: checkpoint(end),
    }
}

#[allow(clippy::too_many_arguments)]
fn fake_verification(
    root: Digest,
    start: usize,
    descriptor_end: usize,
    batching_end: usize,
    xi: EF,
    initial_claim: EF,
    suffix_point: Vec<EF>,
    adjoint: RsAdjointEvalVerification<EF>,
) -> TerminalWhirVerification<F, EF, Digest> {
    TerminalWhirVerification {
        transcript_start: checkpoint(start),
        descriptor_span: span(
            TerminalWhirTranscriptPhase::Descriptor,
            start,
            descriptor_end,
        ),
        batching_challenge_span: span(
            TerminalWhirTranscriptPhase::BatchingChallenge,
            descriptor_end,
            batching_end,
        ),
        transcript_end: checkpoint(batching_end),
        root,
        batching_challenge: xi,
        initial_claim,
        rounds: Vec::new(),
        final_poly: Vec::new(),
        final_weight_evals: Vec::new(),
        final_weight_span: span(
            TerminalWhirTranscriptPhase::FinalWeight,
            batching_end,
            batching_end,
        ),
        suffix_point,
        accumulator_adjoint: adjoint,
        expected_weight: EF::ZERO,
        actual_weight: EF::ZERO,
        final_inner_product: EF::ZERO,
        final_claim: EF::ZERO,
    }
}

fn empty_adjoint() -> RsAdjointEvalVerification<EF> {
    RsAdjointEvalVerification {
        transcript_start: checkpoint(0),
        transcript_end: checkpoint(0),
        claimed_value: EF::ZERO,
        degree: 0,
        rounds: Vec::new(),
        point: Vec::new(),
        final_claim: EF::ZERO,
        expected_final: EF::ZERO,
    }
}

fn check_air<A>(air: &A, trace: &RowMajorMatrix<F>)
where
    A: for<'a> Air<DebugConstraintBuilder<'a, SC>>
        + BaseAir<F>
        + BaseAirWithPublicValues<F>
        + PartitionedBaseAir<F>,
{
    check_constraints::<_, SC>(
        air,
        core::any::type_name::<A>(),
        &None,
        &[trace.as_view()],
        &[],
    );
}

#[test]
fn coefficient_binding_and_eq_relation_fail_closed() {
    let code = coefficient_code(8);
    let whir = whir();
    let root = core::array::from_fn(|index| F::from_usize(100 + index));
    let descriptor = TerminalDescriptor::from_whir_initial_rs(root, &code, &whir, D_EF);
    let binding = NativeTerminalTranscriptBinding::coefficient_subgroup(&descriptor, &code, &whir)
        .expect("exact coefficient-subgroup binding");
    assert_eq!(binding.layout(), WhirInitialRsLayout::CoefficientSubgroup);
    assert!(!binding.descriptor_post_root().is_empty());
    assert!(!binding.whir_layout().is_empty());

    let mut wrong_binding = descriptor.clone();
    wrong_binding.code_layout_version ^= 1;
    assert_eq!(
        NativeTerminalTranscriptBinding::coefficient_subgroup(&wrong_binding, &code, &whir),
        Err(NativeTerminalTranscriptBindingError::Descriptor),
    );
    let mut wrong_folding = descriptor.clone();
    wrong_folding.initial_folding_factor = 1;
    assert_eq!(
        NativeTerminalTranscriptBinding::coefficient_subgroup(&wrong_folding, &code, &whir),
        Err(NativeTerminalTranscriptBindingError::Descriptor),
    );
    let config = SC::default_from_params(SystemParams::new_for_testing(8));
    let ordinary = WhirInitialRsWarpCode::new(config.hasher().clone(), 8, 1, 0, 16);
    assert_eq!(
        NativeTerminalTranscriptBinding::coefficient_subgroup(&descriptor, &ordinary, &whir),
        Err(NativeTerminalTranscriptBindingError::Layout),
    );

    let point = vec![EF::from_u64(3), EF::from_u64(5), EF::from_u64(7)];
    let source_target = EF::from_u64(11);
    let eta = EF::from_u64(13);
    let beta = point
        .iter()
        .copied()
        .chain([source_target])
        .collect::<Vec<_>>();
    let statement = TerminalConstrainedRsStatement {
        linearizer_claims: vec![TerminalStructuredLinearClaim::new(
            TerminalWeightSpec::Eq {
                point: point.clone(),
            },
            source_target + eta,
        )],
    };
    validate_native_terminal_eq_statement(&beta, eta, &statement).expect("exact Eq terminal");
    let mut wrong_point = statement.clone();
    if let TerminalWeightSpec::Eq { point } = &mut wrong_point.linearizer_claims[0].weight {
        point[0] += EF::ONE;
    }
    assert_eq!(
        validate_native_terminal_eq_statement(&beta, eta, &wrong_point),
        Err(NativeTerminalEqClaimError::Relation),
    );
    let mut wrong_target = statement;
    wrong_target.linearizer_claims[0].target += EF::ONE;
    assert_eq!(
        validate_native_terminal_eq_statement(&beta, eta, &wrong_target),
        Err(NativeTerminalEqClaimError::Relation),
    );
}

#[test]
fn coefficient_statement_trace_rejects_root_binding_coordinate_and_target() {
    let code = coefficient_code(8);
    let whir = whir();
    let root = core::array::from_fn(|index| F::from_usize(300 + index));
    let descriptor = TerminalDescriptor::from_whir_initial_rs(root, &code, &whir, D_EF);
    let binding = NativeTerminalTranscriptBinding::coefficient_subgroup(&descriptor, &code, &whir)
        .expect("exact coefficient-subgroup binding");
    let start = 3;
    let post_root = binding
        .descriptor_post_root()
        .iter()
        .chain(binding.whir_layout())
        .copied()
        .collect::<Vec<_>>();
    let descriptor_end = start + 1 + DIGEST_SIZE + post_root.len();
    let batching_end = descriptor_end + D_EF;
    let xi = EF::from_u64(17);
    let mu = EF::from_u64(19);
    let beta_last = EF::from_u64(23);
    let eta = EF::from_u64(29);
    let verification = fake_verification(
        root,
        start,
        descriptor_end,
        batching_end,
        xi,
        mu + xi * (eta + beta_last),
        Vec::new(),
        empty_adjoint(),
    );
    let mut transcript = TranscriptLog::default();
    for index in 0..start {
        transcript.push_observe(F::from_usize(500 + index));
    }
    transcript.push_observe(F::from_u64(NATIVE_TERMINAL_DESCRIPTOR_DOMAIN_TAG));
    transcript.extend_observe(&root);
    transcript.extend_observe(&post_root);
    transcript.extend_sample(xi.as_basis_coefficients_slice());
    assert!(generate_native_terminal_whir_statement_trace(
        &descriptor,
        &binding,
        mu,
        beta_last,
        eta,
        &verification,
        &transcript,
    )
    .is_some());

    let mut wrong_root = verification.clone();
    wrong_root.root[0] += F::ONE;
    assert!(generate_native_terminal_whir_statement_trace(
        &descriptor,
        &binding,
        mu,
        beta_last,
        eta,
        &wrong_root,
        &transcript,
    )
    .is_none());
    assert!(generate_native_terminal_whir_statement_trace(
        &descriptor,
        &binding,
        mu,
        beta_last,
        eta + EF::ONE,
        &verification,
        &transcript,
    )
    .is_none());
    let mut wrong_binding = transcript.clone();
    wrong_binding.values_mut()[start + 1 + DIGEST_SIZE] += F::ONE;
    assert!(generate_native_terminal_whir_statement_trace(
        &descriptor,
        &binding,
        mu,
        beta_last,
        eta,
        &verification,
        &wrong_binding,
    )
    .is_none());
    let mut wrong_coordinate = transcript;
    wrong_coordinate.values_mut()[descriptor_end - 1] += F::ONE;
    assert!(generate_native_terminal_whir_statement_trace(
        &descriptor,
        &binding,
        mu,
        beta_last,
        eta,
        &verification,
        &wrong_coordinate,
    )
    .is_none());
}

#[test]
fn zero_fold_coefficient_adjoint_matches_backend_and_rejects_mutations() {
    // Rows-per-query is terminal metadata only for this adjoint component, so
    // the production value 16 remains valid for this small dynamic oracle.
    let code = coefficient_code(3);
    let alpha = (0..code.log_codeword_len())
        .map(|index| EF::from_usize(37 + 7 * index))
        .collect::<Vec<_>>();
    let whir_point = (0..code.log_message_len())
        .map(|index| EF::from_usize(71 + 11 * index))
        .collect::<Vec<_>>();
    let adjoint_point = (0..code.log_codeword_len())
        .map(|index| EF::from_usize(109 + 13 * index))
        .collect::<Vec<_>>();
    let q = alpha
        .iter()
        .zip(&adjoint_point)
        .map(|(&alpha, &challenge)| (EF::ONE - alpha) * (EF::ONE - challenge) + alpha * challenge)
        .product::<EF>();
    let omega = F::two_adic_generator(code.log_codeword_len());
    let selector = whir_point
        .iter()
        .enumerate()
        .map(|(message_bit, &coordinate)| {
            let y = adjoint_point
                .iter()
                .enumerate()
                .map(|(round, &challenge)| {
                    let exponent_power = code.log_codeword_len() - 1 - round + message_bit;
                    let root = if exponent_power >= code.log_codeword_len() {
                        F::ONE
                    } else {
                        omega.exp_power_of_2(exponent_power)
                    };
                    (EF::ONE - challenge) + challenge * EF::from(root)
                })
                .product::<EF>();
            (EF::ONE - coordinate) + (coordinate.double() - EF::ONE) * y
        })
        .product::<EF>();
    let claimed_value = code.accumulator_weight_eval_at(&alpha, &whir_point);
    let adjoint = RsAdjointEvalVerification {
        transcript_start: checkpoint(0),
        transcript_end: checkpoint(0),
        claimed_value,
        degree: (code.log_message_len() + 1) as u32,
        rounds: Vec::new(),
        point: adjoint_point,
        final_claim: q * selector,
        expected_final: q * selector,
    };
    let mut verification = fake_verification(
        [F::ZERO; 8],
        0,
        0,
        0,
        EF::ZERO,
        EF::ZERO,
        whir_point,
        adjoint,
    );

    let q_air = NativeTerminalRsAdjointQAir {
        round_bus: NativeTerminalRsAdjointRoundBus::new(1),
        accumulator_value_bus:
            openvm_recursion_circuit::native_warp::terminal::NativeTerminalAccumulatorValueBus::new(
                2,
            ),
        point_bus: NativeTerminalWhirPointBus::new(3),
        q_bus: NativeTerminalRsAdjointQBus::new(4),
        round_count: alpha.len(),
        initial_folding_factor: 0,
    };
    let q_trace = generate_native_terminal_rs_adjoint_q_trace(&verification, &alpha, 0, None)
        .expect("zero-fold q trace");
    check_air(&q_air, &q_trace);
    for row in q_trace.values.chunks_exact(q_trace.width()) {
        let cols: &NativeTerminalRsAdjointQCols<F> = row.borrow();
        if cols.active == F::ONE {
            assert_eq!(cols.is_column, F::ZERO);
        }
    }

    let y_air = NativeTerminalRsAdjointYAir::new(
        NativeTerminalRsAdjointRoundBus::new(1),
        NativeTerminalRsAdjointYBus::new(5),
        code.log_message_len(),
        code.log_codeword_len(),
    );
    let y_trace = generate_native_terminal_rs_adjoint_y_trace(&y_air, &verification, None)
        .expect("coefficient-subgroup y trace");
    check_air(&y_air, &y_trace);
    let selector_air = NativeTerminalRsAdjointSelectorAir {
        point_bus: NativeTerminalWhirPointBus::new(3),
        q_bus: NativeTerminalRsAdjointQBus::new(4),
        y_bus: NativeTerminalRsAdjointYBus::new(5),
        claim_bus: NativeTerminalRsAdjointClaimBus::new(6),
        value_bus: NativeTerminalRsAdjointValueBus::new(7),
        log_message_len: code.log_message_len(),
        point_coordinate_offset: 0,
    };
    let selector_trace =
        generate_native_terminal_rs_adjoint_selector_trace(&verification, &alpha, 0, None)
            .expect("zero-fold selector trace");
    check_air(&selector_air, &selector_trace);
    assert_eq!(
        verification.accumulator_adjoint.claimed_value,
        code.accumulator_weight_eval_at(&alpha, &verification.suffix_point),
    );

    assert!(generate_native_terminal_rs_adjoint_q_trace(&verification, &alpha, 1, None).is_none());
    assert!(
        generate_native_terminal_rs_adjoint_selector_trace(&verification, &alpha, 1, None)
            .is_none()
    );
    let mut wrong_coordinate = q_trace;
    let width = wrong_coordinate.width();
    let first: &mut NativeTerminalRsAdjointQCols<F> = wrong_coordinate.values[..width].borrow_mut();
    first.alpha[0] += F::ONE;
    assert!(catch_unwind(AssertUnwindSafe(|| {
        check_air(&q_air, &wrong_coordinate);
    }))
    .is_err());
    verification.accumulator_adjoint.final_claim += EF::ONE;
    assert!(
        generate_native_terminal_rs_adjoint_selector_trace(&verification, &alpha, 0, None)
            .is_none()
    );
}

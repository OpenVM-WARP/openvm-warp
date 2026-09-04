use core::borrow::Borrow;

use openvm_stark_backend::{
    poly_common::{eval_mle_evals_at_point, eval_mobius_eq_mle, Squarable},
    proof::WhirProof,
    prover::{
        poly::Ple, stacked_pcs::stacked_commit, whir::prove_whir_opening_multi, ColMajorMatrix,
    },
    test_utils::{default_test_params_small, FibFixture, TestFixture},
    verifier::whir::verify_whir_multi,
    whir::{
        combined_initial_target as backend_combined_initial_target,
        derive_multi_constraint_batching_coefficients, WhirOpeningConstraint,
    },
    FiatShamirTranscript, StarkEngine, StarkProtocolConfig, TranscriptHistory,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge, default_duplex_sponge_recorder, BabyBearPoseidon2Config,
    BabyBearPoseidon2RefEngine, DuplexSponge, EF, F,
};
use p3_field::{PrimeCharacteristicRing, PrimeField32};
use p3_matrix::Matrix;

use super::{
    air::{MultiConstraintCompletionCols, MultiConstraintStatementBuses},
    combined_initial_target, derive_batching_coefficients_preflight,
    derive_multi_constraint_whir_data, run_multi_constraint_whir_preflight,
    trace::generate_multi_constraint_prefix_trace,
    validate_batching_coefficients_against_gamma, MultiConstraintWhirDerived,
    MultiConstraintWhirError, MultiConstraintWhirProfile, MultiConstraintWhirStatement,
    MultiConstraintWhirTranscriptPreflight,
};
use crate::{
    bus::CertifiedTranscriptCheckpointBus,
    primitives::exp_bits_len::ExpBitsLenCpuTraceGenerator,
    system::{AirModule, BusIndexManager, BusInventory},
    transcript::{
        merkle_verify::{
            generate_trace_with_initial_commitments, MerkleVerifyCols, MerkleVerifyTraceError,
        },
        TranscriptModule,
    },
    whir::multi_constraint::{
        air::MultiConstraintInitialCommitmentBus,
        module::{
            MultiConstraintWhirAir, MultiConstraintWhirDirectCarrier, MultiConstraintWhirModule,
        },
        trace::MultiConstraintWhirTerminalCheckpoint,
        MultiConstraintWhirInitialCommitment,
    },
};

fn ef(value: u64) -> EF {
    EF::from_u64(value)
}

fn bind_native_statement<T: FiatShamirTranscript<BabyBearPoseidon2Config>>(
    transcript: &mut T,
    roots: &[[F; 8]],
    points: &[Vec<EF>],
    openings: &[Vec<Vec<EF>>],
) {
    transcript.observe(F::from_u64(0x5354_4154));
    for &root in roots {
        transcript.observe_commit(root);
    }
    transcript.observe(F::from_usize(points.len()));
    for (constraint, (point, per_commitment)) in points.iter().zip(openings).enumerate() {
        transcript.observe(F::from_usize(constraint));
        transcript.observe(F::from_usize(point.len()));
        for &coordinate in point {
            transcript.observe_ext(coordinate);
        }
        transcript.observe(F::from_usize(per_commitment.len()));
        for commitment in per_commitment {
            transcript.observe(F::from_usize(commitment.len()));
            for &opening in commitment {
                transcript.observe_ext(opening);
            }
        }
    }
}

#[test]
fn complete_native_transcript_and_completion_message_are_recursive_differential() {
    const CONSTRAINTS: usize = 3;
    let engine = BabyBearPoseidon2RefEngine::<DuplexSponge>::new(default_test_params_small());
    let config = engine.config().clone();
    let params = config.params();
    let height = 1usize << params.log_stacked_height();
    let widths = [2usize, 1usize];
    let traces = widths
        .iter()
        .enumerate()
        .map(|(commitment, &width)| {
            ColMajorMatrix::new(
                (0..height * width)
                    .map(|index| F::from_usize(index * 17 + commitment * 101 + 9))
                    .collect(),
                width,
            )
        })
        .collect::<Vec<_>>();
    let committed = traces
        .iter()
        .map(|trace| {
            stacked_commit(
                config.hasher(),
                params.l_skip,
                params.n_stack,
                params.log_blowup,
                params.log_commit_rows_per_query,
                &[trace],
            )
            .unwrap()
        })
        .collect::<Vec<_>>();
    let roots = committed.iter().map(|(root, _)| *root).collect::<Vec<_>>();

    let mut points = Vec::with_capacity(CONSTRAINTS);
    let mut openings = Vec::with_capacity(CONSTRAINTS);
    for constraint in 0..CONSTRAINTS {
        let prism = (0..=params.n_stack)
            .map(|coordinate| EF::from_usize(31 * constraint + coordinate + 2))
            .collect::<Vec<_>>();
        let cube = prism[0]
            .exp_powers_of_2()
            .take(params.l_skip)
            .chain(prism[1..].iter().copied())
            .collect::<Vec<_>>();
        let per_commitment = committed
            .iter()
            .map(|(_, committed)| {
                committed
                    .matrix
                    .columns()
                    .map(|column| {
                        Ple::from_evaluations(params.l_skip, column).eval_at_point(
                            params.l_skip,
                            prism[0],
                            &prism[1..],
                        )
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        points.push(cube);
        openings.push(per_commitment);
    }

    let mut native_transcript = default_duplex_sponge_recorder();
    bind_native_statement(&mut native_transcript, &roots, &points, &openings);
    let native_coefficients = derive_multi_constraint_batching_coefficients::<
        BabyBearPoseidon2Config,
        _,
    >(&mut native_transcript, CONSTRAINTS)
    .unwrap();
    let native_constraints = points
        .iter()
        .zip(&openings)
        .zip(&native_coefficients)
        .map(|((point, openings), &rho)| WhirOpeningConstraint::new(point, openings, rho))
        .collect::<Vec<_>>();
    let committed_refs = committed
        .iter()
        .map(|(_, committed)| (&committed.matrix, &committed.tree))
        .collect::<Vec<_>>();
    let proof = prove_whir_opening_multi::<BabyBearPoseidon2Config, _>(
        &mut native_transcript,
        config.hasher(),
        params.l_skip,
        params.log_blowup,
        params.whir(),
        &committed_refs,
        &native_constraints,
    )
    .unwrap();
    let native_log = native_transcript.into_log();

    let mut recursive_transcript = default_duplex_sponge_recorder();
    bind_native_statement(&mut recursive_transcript, &roots, &points, &openings);
    let (batching_prefix_tidx, batching_gamma, recursive_coefficients) =
        derive_batching_coefficients_preflight(&mut recursive_transcript, CONSTRAINTS).unwrap();
    assert_eq!(recursive_coefficients, native_coefficients);
    let recursive_preflight = run_multi_constraint_whir_preflight(
        &mut recursive_transcript,
        params,
        &proof,
        CONSTRAINTS,
        batching_prefix_tidx,
        batching_gamma,
    )
    .unwrap();
    let recursive_log = recursive_transcript.into_log();
    assert_eq!(recursive_log.values(), native_log.values());
    assert_eq!(recursive_log.samples(), native_log.samples());
    // The backend recorder labels PoW/query helpers as CheckWitness/SampleBits,
    // while recursion replays their primitive observe/sample operations.  AIR
    // semantics are the operation stream, which is exactly equal above.

    let recursive_statement = MultiConstraintWhirStatement {
        points: points.clone(),
        openings: openings.clone(),
        batching_coefficients: recursive_coefficients,
    };
    let profile =
        MultiConstraintWhirProfile::new(CONSTRAINTS, points[0].len(), widths.to_vec()).unwrap();
    let derived = derive_multi_constraint_whir_data(
        &profile,
        &recursive_statement,
        recursive_preflight.mu,
        &recursive_preflight.alphas,
        &proof.final_poly,
    )
    .unwrap();
    assert_eq!(
        derived.initial_target,
        backend_combined_initial_target(
            &native_constraints,
            recursive_preflight.mu,
            profile.total_width().unwrap(),
        )
    );

    // Exercise the complete owner with a genuine PCS-level carrier.  There is
    // deliberately no full child Proof and no synthetic single stacking
    // opening in the public API.
    let (child_vk, _) = FibFixture::new(0, 1, 1 << params.l_skip).keygen_and_prove(&engine);
    let initial_commitments = roots
        .iter()
        .zip(widths)
        .map(|(&commitment, width)| MultiConstraintWhirInitialCommitment { commitment, width })
        .collect::<Vec<_>>();

    let trailing_samples = recursive_log
        .samples()
        .iter()
        .rev()
        .take_while(|&&is_sample| is_sample)
        .count();
    let sample_count = match trailing_samples % 8 {
        0 => 8,
        remainder => remainder,
    };
    let terminal_checkpoint = MultiConstraintWhirTerminalCheckpoint {
        end_tidx: recursive_log.len(),
        sample_count,
        state: *recursive_log.perm_results().last().unwrap(),
    };
    let mut bus_indices = BusIndexManager::new();
    let inventory = BusInventory::new(&mut bus_indices);
    let transcript_module =
        TranscriptModule::<1>::new(inventory.clone(), params.clone(), false, false);
    let statement_buses = MultiConstraintStatementBuses::new(
        bus_indices.new_bus_idx(),
        bus_indices.new_bus_idx(),
        bus_indices.new_bus_idx(),
    );
    let initial_commitment_bus =
        MultiConstraintInitialCommitmentBus::new(bus_indices.new_bus_idx());
    let checkpoint_bus = CertifiedTranscriptCheckpointBus::new(bus_indices.new_bus_idx());
    let module = MultiConstraintWhirModule::new(
        &child_vk,
        profile.clone(),
        statement_buses,
        initial_commitment_bus,
        checkpoint_bus,
        1,
        7,
        &mut bus_indices,
        inventory,
    )
    .unwrap();
    let prepared = module
        .prepare_direct_carriers(&[MultiConstraintWhirDirectCarrier {
            whir_proof: &proof,
            initial_commitments: &initial_commitments,
            statement: &recursive_statement,
            multi_preflight: &recursive_preflight,
            transcript: &recursive_log,
            terminal_checkpoint,
        }])
        .unwrap();
    assert_eq!(prepared[0].whir_proof(), &proof);
    assert_eq!(
        prepared[0].transcript_preflight().transcript.values(),
        recursive_log.values()
    );
    assert!(
        !prepared[0]
            .transcript_preflight()
            .poseidon2_perm_inputs
            .is_empty(),
        "initial WHIR row hashes must reach the shared Poseidon owner"
    );
    assert!(
        !prepared[0]
            .transcript_preflight()
            .poseidon2_compress_inputs
            .is_empty(),
        "folded-codeword leaf hashes must reach the shared Poseidon owner"
    );
    let contexts = module
        .generate_air_contexts::<BabyBearPoseidon2Config>(
            &child_vk,
            &prepared,
            &ExpBitsLenCpuTraceGenerator::default(),
            None,
        )
        .unwrap();
    assert_eq!(
        module.airs::<BabyBearPoseidon2Config>().len(),
        MultiConstraintWhirAir::COUNT
    );
    assert_eq!(contexts.len(), MultiConstraintWhirAir::COUNT);

    // The direct-carrier Merkle path must use both retained roots in their
    // exact `commit_minor` order. The compatibility proof intentionally has a
    // zero common-main root, so this also catches accidental fallback to the
    // legacy first-root path.
    let transcript_proofs = prepared
        .iter()
        .map(|carrier| carrier.transcript_proof_adapter().clone())
        .collect::<Vec<_>>();
    let transcript_preflights = prepared
        .iter()
        .map(|carrier| carrier.transcript_preflight().clone())
        .collect::<Vec<_>>();
    let direct_commitments = prepared
        .iter()
        .map(|carrier| carrier.initial_commitments().to_vec())
        .collect::<Vec<_>>();
    let transcript_packet = transcript_module
        .generate_cpu_contexts_for_shared_poseidon_with_checkpoints_and_initial_commitments::<
            BabyBearPoseidon2Config,
        >(
            &child_vk,
            &transcript_proofs,
            &transcript_preflights,
            &[],
            &[],
            &[],
            &direct_commitments,
            None,
        )
        .unwrap()
        .unwrap();
    let merkle_trace = &transcript_packet.contexts[1].common_main.values;
    let final_initial_rows = merkle_trace
        .chunks(MerkleVerifyCols::<F>::width())
        .map(|row| row.borrow())
        .filter(|cols: &&MerkleVerifyCols<F>| {
            cols.is_valid == F::ONE && cols.is_last_merkle == F::ONE && cols.commit_major == F::ZERO
        })
        .collect::<Vec<_>>();
    assert_eq!(
        final_initial_rows.len(),
        params.whir.rounds[0].num_queries * widths.len()
    );
    for cols in &final_initial_rows {
        let commit_minor = cols.commit_minor.as_canonical_u32() as usize;
        assert_eq!(cols.left, roots[commit_minor]);
        assert_eq!(cols.right, roots[commit_minor]);
    }

    let mut mutated_root = direct_commitments.clone();
    mutated_root[0][1].commitment[0] += F::ONE;
    let (mutated_trace, _) = generate_trace_with_initial_commitments(
        &child_vk,
        &transcript_proofs,
        &transcript_preflights,
        params,
        &mutated_root,
        None,
    )
    .unwrap()
    .unwrap();
    assert!(mutated_trace
        .chunks(MerkleVerifyCols::<F>::width())
        .map(|row| row.borrow())
        .any(|cols: &MerkleVerifyCols<F>| {
            cols.is_valid == F::ONE
                && cols.is_last_merkle == F::ONE
                && cols.commit_major == F::ZERO
                && cols.commit_minor == F::ONE
                && cols.left != cols.right
        }));

    let mut reordered = direct_commitments.clone();
    reordered[0].swap(0, 1);
    assert!(matches!(
        generate_trace_with_initial_commitments(
            &child_vk,
            &transcript_proofs,
            &transcript_preflights,
            params,
            &reordered,
            None,
        ),
        Err(MerkleVerifyTraceError::InitialCommitmentWidth {
            proof_idx: 0,
            commit_minor: 0,
            ..
        })
    ));

    let missing_root = vec![direct_commitments[0][..1].to_vec()];
    assert!(matches!(
        generate_trace_with_initial_commitments(
            &child_vk,
            &transcript_proofs,
            &transcript_preflights,
            params,
            &missing_root,
            None,
        ),
        Err(MerkleVerifyTraceError::InitialCommitmentCount {
            proof_idx: 0,
            actual: 1,
            expected: 2,
        })
    ));

    #[cfg(feature = "cuda")]
    {
        use openvm_cuda_common::stream::GpuDeviceCtx;

        use crate::{
            cuda::{preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu},
            transcript::cuda_tracegen::TranscriptBlob,
        };

        let device_ctx = GpuDeviceCtx::for_current_device().expect("CUDA device");
        let vk_gpu = VerifyingKeyGpu::new(&child_vk, &device_ctx);
        let proofs_gpu = transcript_proofs
            .iter()
            .map(|proof| ProofGpu::new(&child_vk, proof, &device_ctx))
            .collect::<Vec<_>>();
        let preflights_gpu = transcript_proofs
            .iter()
            .zip(&transcript_preflights)
            .map(|(proof, preflight)| PreflightGpu::new(&child_vk, proof, preflight, &device_ctx))
            .collect::<Vec<_>>();

        let external_permutations = Vec::new();
        let external_compressions = Vec::new();
        let cuda_ctx = (&external_permutations, &external_compressions, &device_ctx);
        let blob = TranscriptBlob::new_with_initial_commitments(
            &vk_gpu,
            &proofs_gpu,
            &preflights_gpu,
            &cuda_ctx,
            &direct_commitments,
        )
        .unwrap();
        let merkle_blob = &blob.merkle_verify_blob;
        for record in merkle_blob
            .records
            .iter()
            .filter(|record| record.commit_major == 0)
        {
            let commit_minor = usize::from(record.commit_minor);
            let root_offset = record.siblings_offset as usize
                + usize::from(record.depth) * roots[commit_minor].len();
            assert_eq!(
                &merkle_blob.sibling_hashes[root_offset..root_offset + roots[commit_minor].len()],
                &roots[commit_minor]
            );
        }

        let mutated_blob = TranscriptBlob::new_with_initial_commitments(
            &vk_gpu,
            &proofs_gpu,
            &preflights_gpu,
            &cuda_ctx,
            &mutated_root,
        )
        .unwrap();
        let mutated_merkle_blob = &mutated_blob.merkle_verify_blob;
        assert!(mutated_merkle_blob
            .records
            .iter()
            .filter(|record| record.commit_major == 0 && record.commit_minor == 1)
            .all(|record| {
                let root_offset =
                    record.siblings_offset as usize + usize::from(record.depth) * roots[1].len();
                mutated_merkle_blob.sibling_hashes[root_offset] == mutated_root[0][1].commitment[0]
                    && mutated_merkle_blob.sibling_hashes[root_offset] != roots[1][0]
            }));
    }

    // The integration helper must project exactly the active row emitted by
    // the genuine completion trace. Neither terminal scalar is supplied by
    // this test (or by an enclosing setup-authority circuit).
    let completion_messages = module.completion_messages(&child_vk, &prepared).unwrap();
    assert_eq!(completion_messages.len(), 1);
    let completion_trace = &contexts[MultiConstraintWhirAir::Completion as usize].common_main;
    let active_row = &completion_trace.values[..completion_trace.width()];
    let active: &MultiConstraintCompletionCols<F> = active_row.borrow();
    let projected = &completion_messages[0];
    assert_eq!(active.is_enabled, F::ONE);
    assert_eq!(active.proof_idx, projected.proof_idx);
    assert_eq!(active.class_index, projected.class_index);
    assert_eq!(active.end_tidx, projected.end_tidx);
    assert_eq!(active.sample_count, projected.sample_count);
    assert_eq!(active.state, projected.state);
    assert_eq!(active.final_aggregate, projected.final_aggregate);
    assert_eq!(active.final_claim, projected.final_claim);

    let mut wrong_width = initial_commitments.clone();
    wrong_width[0].width += 1;
    assert!(module
        .prepare_direct_carriers(&[MultiConstraintWhirDirectCarrier {
            whir_proof: &proof,
            initial_commitments: &wrong_width,
            statement: &recursive_statement,
            multi_preflight: &recursive_preflight,
            transcript: &recursive_log,
            terminal_checkpoint,
        }])
        .is_err());
    let mut missing_constraint = recursive_statement.clone();
    missing_constraint.points.pop();
    missing_constraint.openings.pop();
    missing_constraint.batching_coefficients.pop();
    assert!(module
        .prepare_direct_carriers(&[MultiConstraintWhirDirectCarrier {
            whir_proof: &proof,
            initial_commitments: &initial_commitments,
            statement: &missing_constraint,
            multi_preflight: &recursive_preflight,
            transcript: &recursive_log,
            terminal_checkpoint,
        }])
        .is_err());

    let verify = |points: &[Vec<EF>],
                  openings: &[Vec<Vec<EF>>],
                  coefficient_mutation: Option<usize>,
                  proof: &WhirProof<BabyBearPoseidon2Config>| {
        let mut transcript = default_duplex_sponge();
        bind_native_statement(&mut transcript, &roots, points, openings);
        let mut coefficients = derive_multi_constraint_batching_coefficients::<
            BabyBearPoseidon2Config,
            _,
        >(&mut transcript, CONSTRAINTS)
        .unwrap();
        if let Some(index) = coefficient_mutation {
            coefficients[index] += EF::ONE;
        }
        let constraints = points
            .iter()
            .zip(openings)
            .zip(&coefficients)
            .map(|((point, openings), &rho)| WhirOpeningConstraint::new(point, openings, rho))
            .collect::<Vec<_>>();
        verify_whir_multi(&mut transcript, &config, proof, &roots, &constraints)
    };
    verify(&points, &openings, None, &proof).unwrap();

    let mut bad_points = points.clone();
    bad_points[1][0] += EF::ONE;
    assert!(verify(&bad_points, &openings, None, &proof).is_err());

    let mut bad_openings = openings.clone();
    bad_openings[2][0][1] += EF::ONE;
    assert!(verify(&points, &bad_openings, None, &proof).is_err());
    assert!(verify(&points, &openings, Some(1), &proof).is_err());

    let mut bad_query = proof.clone();
    bad_query.initial_round_opened_rows[0][0][0][0] += F::ONE;
    assert!(verify(&points, &openings, None, &bad_query).is_err());

    let mut bad_final_poly = proof.clone();
    bad_final_poly.final_poly[0] += EF::ONE;
    assert!(verify(&points, &openings, None, &bad_final_poly).is_err());
}

fn fixture() -> (
    MultiConstraintWhirProfile,
    MultiConstraintWhirStatement,
    EF,
    Vec<EF>,
    Vec<EF>,
) {
    let profile = MultiConstraintWhirProfile::new(3, 4, vec![2, 1]).unwrap();
    let gamma = ef(7);
    let statement = MultiConstraintWhirStatement {
        points: vec![
            vec![ef(2), ef(3), ef(5), ef(11)],
            vec![ef(13), ef(17), ef(19), ef(23)],
            vec![ef(29), ef(31), ef(37), ef(41)],
        ],
        openings: vec![
            vec![vec![ef(43), ef(47)], vec![ef(53)]],
            vec![vec![ef(59), ef(61)], vec![ef(67)]],
            vec![vec![ef(71), ef(73)], vec![ef(79)]],
        ],
        batching_coefficients: vec![EF::ONE, gamma, gamma * gamma],
    };
    let mu = ef(83);
    let alphas = vec![ef(89), ef(97)];
    let final_poly = vec![ef(101), ef(103), ef(107), ef(109)];
    (profile, statement, mu, alphas, final_poly)
}

fn derived_fixture() -> (
    MultiConstraintWhirProfile,
    MultiConstraintWhirStatement,
    MultiConstraintWhirDerived,
) {
    let (profile, statement, mu, alphas, final_poly) = fixture();
    let derived =
        derive_multi_constraint_whir_data(&profile, &statement, mu, &alphas, &final_poly).unwrap();
    (profile, statement, derived)
}

#[test]
fn initial_target_matches_backend_reference() {
    let (profile, statement, mu, _, _) = fixture();
    let constraints = statement
        .points
        .iter()
        .zip(&statement.openings)
        .zip(&statement.batching_coefficients)
        .map(|((point, openings), &rho)| WhirOpeningConstraint::new(point, openings, rho))
        .collect::<Vec<_>>();
    let expected =
        backend_combined_initial_target(&constraints, mu, profile.total_width().unwrap());
    assert_eq!(
        combined_initial_target(&profile, &statement, mu).unwrap(),
        expected
    );
}

#[test]
fn generalized_weight_matches_direct_native_reference() {
    let (profile, statement, mu, alphas, final_poly) = fixture();
    let derived =
        derive_multi_constraint_whir_data(&profile, &statement, mu, &alphas, &final_poly).unwrap();
    let expected = statement
        .points
        .iter()
        .zip(&statement.batching_coefficients)
        .map(|(point, &rho)| {
            let prefix = eval_mobius_eq_mle(&point[..alphas.len()], &alphas);
            let mut table = final_poly.clone();
            let suffix = eval_mle_evals_at_point(&mut table, &point[alphas.len()..]);
            rho * prefix * suffix
        })
        .sum::<EF>();
    assert_eq!(derived.final_weighted_evaluation, expected);
    assert_eq!(derived.final_prefix_weights.len(), profile.constraint_count);
}

#[test]
fn batching_transcript_is_backend_differential() {
    let mut backend = default_duplex_sponge_recorder();
    let mut recursion = default_duplex_sponge_recorder();
    // Model an identical caller-owned statement prefix.
    for value in [F::from_u32(123), F::from_u32(456)] {
        FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(&mut backend, value);
        FiatShamirTranscript::<BabyBearPoseidon2Config>::observe(&mut recursion, value);
    }
    let expected = derive_multi_constraint_batching_coefficients::<BabyBearPoseidon2Config, _>(
        &mut backend,
        3,
    )
    .unwrap();
    let (start, gamma, actual) = derive_batching_coefficients_preflight(&mut recursion, 3).unwrap();
    assert_eq!(start, 2);
    assert_eq!(actual, expected);
    assert_eq!(gamma, expected[1]);
    let backend_log = TranscriptHistory::into_log(backend);
    let recursion_log = TranscriptHistory::into_log(recursion);
    assert_eq!(backend_log.values(), recursion_log.values());
    assert_eq!(backend_log.samples(), recursion_log.samples());
}

#[test]
fn transcript_gamma_not_caller_geometric_sequence_is_authoritative() {
    let (_, mut statement, _, _, _) = fixture();
    let transcript_gamma = ef(7);
    validate_batching_coefficients_against_gamma(&statement, transcript_gamma).unwrap();

    // This remains a geometric sequence, but for a prover-selected gamma.  A
    // mere `statement.validate()` accepts its shape; the transcript check and
    // Prefix AIR recurrence reject it.
    let malicious_gamma = ef(17);
    statement.batching_coefficients =
        vec![EF::ONE, malicious_gamma, malicious_gamma * malicious_gamma];
    statement
        .validate(&MultiConstraintWhirProfile::new(3, 4, vec![2, 1]).unwrap())
        .unwrap();
    assert_eq!(
        validate_batching_coefficients_against_gamma(&statement, transcript_gamma),
        Err(MultiConstraintWhirError::InvalidCoefficientPower { constraint: 1 })
    );
}

#[test]
fn prefix_trace_rejects_coefficients_for_wrong_gamma_before_air() {
    let (profile, mut statement, derived) = derived_fixture();
    let malicious_gamma = ef(17);
    statement.batching_coefficients =
        vec![EF::ONE, malicious_gamma, malicious_gamma * malicious_gamma];
    let preflight = MultiConstraintWhirTranscriptPreflight {
        batching_prefix_tidx: 9,
        batching_gamma: ef(7),
        proof_prefix_tidx: 16,
        mu_pow_sample: F::ZERO,
        mu: ef(83),
        post_mu_tidx: 23,
        whir_round_tidx_per_round: vec![],
        query_tidx_per_round: vec![],
        alphas: vec![],
        z0s: vec![],
        gammas: vec![],
        folding_pow_samples: vec![],
        query_pow_samples: vec![],
        queries: vec![],
    };
    let proof = WhirProof::<BabyBearPoseidon2Config> {
        mu_pow_witness: F::ZERO,
        whir_sumcheck_polys: vec![],
        codeword_commits: vec![],
        ood_values: vec![],
        initial_round_opened_rows: vec![],
        initial_round_merkle_proofs: vec![],
        codeword_opened_values: vec![],
        codeword_merkle_proofs: vec![],
        folding_pow_witnesses: vec![],
        query_phase_pow_witnesses: vec![],
        final_poly: vec![],
    };
    assert_eq!(profile.constraint_count, 3);
    assert_eq!(
        generate_multi_constraint_prefix_trace(
            &[preflight],
            &[statement],
            &[derived],
            &[&proof],
            3,
            None,
        ),
        Err(MultiConstraintWhirError::InvalidCoefficientPower { constraint: 1 })
    );
}

#[test]
fn mutations_change_or_reject_the_statement() {
    let (profile, statement, mu, alphas, final_poly) = fixture();
    let baseline =
        derive_multi_constraint_whir_data(&profile, &statement, mu, &alphas, &final_poly).unwrap();

    let mut point = statement.clone();
    point.points[1][3] += EF::ONE;
    assert_ne!(
        derive_multi_constraint_whir_data(&profile, &point, mu, &alphas, &final_poly)
            .unwrap()
            .final_weighted_evaluation,
        baseline.final_weighted_evaluation
    );

    let mut opening = statement.clone();
    opening.openings[2][0][1] += EF::ONE;
    assert_ne!(
        derive_multi_constraint_whir_data(&profile, &opening, mu, &alphas, &final_poly)
            .unwrap()
            .initial_target,
        baseline.initial_target
    );

    let mut order = statement.clone();
    order.points.swap(0, 2);
    order.openings.swap(0, 2);
    assert_ne!(
        derive_multi_constraint_whir_data(&profile, &order, mu, &alphas, &final_poly)
            .unwrap()
            .final_weighted_evaluation,
        baseline.final_weighted_evaluation
    );

    let mut count = statement.clone();
    count.points.pop();
    assert_eq!(
        count.validate(&profile),
        Err(MultiConstraintWhirError::ConstraintCount {
            actual: 2,
            expected: 3
        })
    );

    // The final polynomial is part of the WHIR proof. Mutating it changes the
    // final generalized check, while malformed length is rejected.
    let mut proof_mutation = final_poly.clone();
    proof_mutation[0] += EF::ONE;
    assert_ne!(
        derive_multi_constraint_whir_data(&profile, &statement, mu, &alphas, &proof_mutation)
            .unwrap()
            .final_weighted_evaluation,
        baseline.final_weighted_evaluation
    );
    assert_eq!(
        derive_multi_constraint_whir_data(&profile, &statement, mu, &alphas, &final_poly[..3]),
        Err(MultiConstraintWhirError::FinalPolynomialLength {
            actual: 3,
            expected: 4
        })
    );
}

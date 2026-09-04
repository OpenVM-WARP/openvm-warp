use openvm_cpu_backend::CpuReducedSwirlSource;
use openvm_sdk::prover::reduced_swirl_native::{
    prove_reduced_swirl_native_cpu_streaming, reduced_swirl_manifest_digest,
    reduced_swirl_source_challenger, verify_reduced_swirl_native_recorded,
    ReducedSwirlNativeCpuStream, ReducedSwirlNativeSetup, ReducedSwirlNativeStatement,
    REDUCED_SWIRL_NATIVE_PROTOCOL_VERSION,
};
use openvm_stark_backend::{
    p3_field::PrimeCharacteristicRing,
    prover::{DeviceDataTransporter, NativeStackingReduction},
    test_utils::{default_test_params_small, PreprocessedFibFixture, TestFixture},
    StarkEngine, WhirProximityStrategy,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2CpuEngine, Digest, DuplexSponge, F,
};

type Engine = BabyBearPoseidon2CpuEngine<DuplexSponge>;

fn reduced_swirl_test_params() -> openvm_stark_backend::SystemParams {
    let mut params = default_test_params_small();
    params.whir.proximity = WhirProximityStrategy::UniqueDecoding;
    params
}

fn assert_native_proof_rejected_without_panic(
    setup: &ReducedSwirlNativeSetup,
    statement: &ReducedSwirlNativeStatement,
    claims: &[openvm_stark_backend::warp_accum::ReducedConstrainedCodeClaim<
        openvm_stark_sdk::config::baby_bear_poseidon2::EF,
        openvm_sdk::prover::reduced_swirl_native::ReducedSwirlFreshCommitment,
    >],
    proof: &openvm_sdk::prover::reduced_swirl_native::ReducedSwirlNativeProof,
) {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        verify_reduced_swirl_native_recorded(setup, statement, claims, proof)
    }));
    assert!(matches!(result, Ok(Err(_))), "malformed proof must reject");
}

fn digest(seed: u32) -> Digest {
    core::array::from_fn(|index| F::from_u32(seed + index as u32 + 1))
}

#[test]
fn genuine_reduced_swirl_sources_produce_one_verified_native_proof() -> eyre::Result<()> {
    let params = reduced_swirl_test_params();
    let engine = Engine::new(params.clone());
    let selectors = vec![true; 1 << 5];
    let key_fixture = PreprocessedFibFixture::new(0, 1, selectors.clone());
    let (host_pk, _vk) = key_fixture.keygen(&engine);
    let device_pk = engine.device().transport_pk_to_device(&host_pk);
    let mut sources = Vec::new();

    for (a, b) in [(0, 1), (2, 3)] {
        let fixture = PreprocessedFibFixture::new(a, b, selectors.clone());
        let proving_context = engine
            .device()
            .transport_proving_ctx_to_device(&fixture.generate_proving_ctx());
        let mut prefix_prover = engine.prover();
        let NativeStackingReduction {
            stacking_proof,
            pending_witness,
            ..
        } = prefix_prover.prove_native_stacking_reduction(&device_pk, proving_context)?;
        let mut source_challenger = reduced_swirl_source_challenger();
        let source = CpuReducedSwirlSource::try_from_pending(
            pending_witness,
            &stacking_proof.stacking_openings,
            &mut source_challenger,
        )?;
        source.validate_message_opening_reference()?;
        source.validate_codeword_opening_reference()?;
        sources.push(source);
    }

    let setup = ReducedSwirlNativeSetup::new(engine.config(), &params, 2, 16, sources.len())?;
    let source_bindings = vec![digest(200), digest(300)];
    let statement = ReducedSwirlNativeStatement {
        protocol_version: REDUCED_SWIRL_NATIVE_PROTOCOL_VERSION,
        block_manifest_digest: reduced_swirl_manifest_digest(&source_bindings)?,
        source_bindings,
    };
    let mut supplied = Some(sources);
    let output = prove_reduced_swirl_native_cpu_streaming(
        &setup,
        statement.clone(),
        |range| -> Result<Vec<_>, &'static str> {
            let sources = supplied.take().ok_or("sources requested twice")?;
            if range != (0..sources.len()) {
                return Err("unexpected source range");
            }
            Ok(sources)
        },
    )?;
    assert_eq!(output.proof.vacc.steps.len(), 1);
    assert_eq!(output.authoritative_claims.len(), 2);
    assert!(output
        .authoritative_claims
        .iter()
        .all(|claim| claim.eta != openvm_stark_sdk::config::baby_bear_poseidon2::EF::ZERO));

    let verification = verify_reduced_swirl_native_recorded(
        &setup,
        &statement,
        &output.authoritative_claims,
        &output.proof,
    )?;
    assert_eq!(
        verification.final_instance,
        output.proof.vacc.final_instance
    );
    assert_eq!(verification.transition_records.len(), 1);
    assert_eq!(verification.terminal.root, verification.final_instance.rt);
    assert_eq!(
        verification.swirl_power_batch_security,
        output.swirl_power_batch_security
    );
    assert_eq!(output.swirl_power_batch_security.source_count, 2);
    assert!(
        output
            .swirl_power_batch_security
            .conservative_security_bits_floor
            <= output.swirl_power_batch_security.available_field_bits_floor
    );

    // Verification enforces the setup family capacity before accepting a
    // transcript. Prover-side bounds alone would not protect the union-bound
    // security accounting of a verifier configured for fewer sources.
    let undersized_setup = ReducedSwirlNativeSetup::new(engine.config(), &params, 2, 16, 1)?;
    assert!(verify_reduced_swirl_native_recorded(
        &undersized_setup,
        &statement,
        &output.authoritative_claims,
        &output.proof,
    )
    .is_err());

    let mut substituted_claims = output.authoritative_claims.clone();
    substituted_claims[0].eta += openvm_stark_sdk::config::baby_bear_poseidon2::EF::ONE;
    assert!(verify_reduced_swirl_native_recorded(
        &setup,
        &statement,
        &substituted_claims,
        &output.proof,
    )
    .is_err());

    let mut reordered_statement = statement.clone();
    reordered_statement.source_bindings.swap(0, 1);
    reordered_statement.block_manifest_digest =
        reduced_swirl_manifest_digest(&reordered_statement.source_bindings)?;
    assert!(verify_reduced_swirl_native_recorded(
        &setup,
        &reordered_statement,
        &output.authoritative_claims,
        &output.proof,
    )
    .is_err());

    let mut duplicated_statement = statement.clone();
    duplicated_statement.source_bindings[1] = duplicated_statement.source_bindings[0];
    duplicated_statement.block_manifest_digest =
        reduced_swirl_manifest_digest(&duplicated_statement.source_bindings)?;
    assert!(verify_reduced_swirl_native_recorded(
        &setup,
        &duplicated_statement,
        &output.authoritative_claims,
        &output.proof,
    )
    .is_err());
    Ok(())
}

#[test]
fn cpu_stream_flushes_exact_linear_schedule_with_bounded_pending_sources() -> eyre::Result<()> {
    let params = reduced_swirl_test_params();
    let engine = Engine::new(params.clone());
    let selectors = vec![true; 1 << 5];
    let key_fixture = PreprocessedFibFixture::new(0, 1, selectors.clone());
    let (host_pk, _vk) = key_fixture.keygen(&engine);
    let device_pk = engine.device().transport_pk_to_device(&host_pk);
    let source_count = 5;
    let setup = ReducedSwirlNativeSetup::new(engine.config(), &params, 2, 16, source_count)?;
    let mut stream = ReducedSwirlNativeCpuStream::new(&setup, source_count)?;

    for source_index in 0..source_count {
        let fixture = PreprocessedFibFixture::new(
            source_index as u64,
            source_index as u64 + 1,
            selectors.clone(),
        );
        let proving_context = engine
            .device()
            .transport_proving_ctx_to_device(&fixture.generate_proving_ctx());
        let mut prefix_prover = engine.prover();
        let NativeStackingReduction {
            stacking_proof,
            pending_witness,
            ..
        } = prefix_prover.prove_native_stacking_reduction(&device_pk, proving_context)?;
        let mut source_challenger = reduced_swirl_source_challenger();
        let source = CpuReducedSwirlSource::try_from_pending(
            pending_witness,
            &stacking_proof.stacking_openings,
            &mut source_challenger,
        )?;
        stream.push_source(digest(1000 + source_index as u32 * 10), source)?;
        assert!(stream.pending_source_count() <= 1);
        if source_index != 0 {
            assert_eq!(stream.pending_source_count(), 0);
        }
    }

    let output = stream.finish()?;
    assert_eq!(output.proof.vacc.steps.len(), 4);
    let verification = verify_reduced_swirl_native_recorded(
        &setup,
        &output.proof.statement,
        &output.authoritative_claims,
        &output.proof,
    )?;
    assert_eq!(verification.transition_records.len(), 4);
    assert_eq!(verification.terminal.root, verification.final_instance.rt);

    // The verifier owns the exact linear WARP schedule. A malformed history
    // must return an error rather than relying on vector indexing or accepting
    // a host-provided final instance.
    let mut reordered = output.proof.clone();
    reordered.vacc.steps.swap(1, 2);
    assert_native_proof_rejected_without_panic(
        &setup,
        &output.proof.statement,
        &output.authoritative_claims,
        &reordered,
    );

    let mut dropped = output.proof.clone();
    dropped.vacc.steps.remove(1);
    assert_native_proof_rejected_without_panic(
        &setup,
        &output.proof.statement,
        &output.authoritative_claims,
        &dropped,
    );

    let mut duplicated = output.proof.clone();
    duplicated.vacc.steps[2] = duplicated.vacc.steps[1].clone();
    assert_native_proof_rejected_without_panic(
        &setup,
        &output.proof.statement,
        &output.authoritative_claims,
        &duplicated,
    );

    let mut substituted_root = output.proof.clone();
    substituted_root.vacc.final_instance.rt[0] += F::ONE;
    assert_native_proof_rejected_without_panic(
        &setup,
        &output.proof.statement,
        &output.authoritative_claims,
        &substituted_root,
    );
    Ok(())
}

#[test]
fn stream_rejects_runtime_source_count_above_fixed_key_capacity() -> eyre::Result<()> {
    let params = reduced_swirl_test_params();
    let engine = Engine::new(params.clone());
    let setup = ReducedSwirlNativeSetup::new(engine.config(), &params, 2, 16, 8)?;
    assert_eq!(setup.maximum_source_count(), 8);
    assert!(ReducedSwirlNativeCpuStream::new(&setup, 9).is_err());
    Ok(())
}

#[test]
fn setup_rejects_whir_regimes_not_covered_by_the_swirl_power_batch_bound() {
    let mut params = default_test_params_small();
    params.whir.proximity = WhirProximityStrategy::ListDecoding { m: 2 };
    let engine = Engine::new(params.clone());
    assert!(ReducedSwirlNativeSetup::new(engine.config(), &params, 2, 16, 8).is_err());

    params.whir.proximity = WhirProximityStrategy::SplitUniqueList {
        m: 2,
        list_start_round: 1,
    };
    let engine = Engine::new(params.clone());
    assert!(ReducedSwirlNativeSetup::new(engine.config(), &params, 2, 16, 8).is_err());
}

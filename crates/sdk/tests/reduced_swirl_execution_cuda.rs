#![cfg(feature = "cuda")]

use std::sync::Arc;

use eyre::Result;
use openvm_circuit::{
    arch::{
        instructions::{
            exe::VmExe, instruction::Instruction, program::Program, LocalOpcode, SystemOpcode,
        },
        MemoryConfig, SystemConfig,
    },
    system::cuda::extensions::SystemGpuBuilder,
};
use openvm_continuations::circuit::reduced_swirl_transition_leaf::{
    reduced_swirl_transition_chain_append, reduced_swirl_transition_chain_genesis,
    REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY,
};
use openvm_recursion_circuit::native_warp::{
    generate_reduced_swirl_manifest_reconciliation_trace, reduced_swirl_vacc_schedule,
    ReducedSwirlVaccTransitionReceiptMessage,
};
use openvm_sdk::{
    config::AppConfig,
    keygen::AppProvingKey,
    prover::{
        reduced_swirl_execution_cuda::{
            new_reduced_swirl_cuda_app_prover, prove_reduced_swirl_cuda_execution,
            prove_reduced_swirl_cuda_transition_execution,
        },
        reduced_swirl_native::{
            reduced_swirl_manifest_digest, verify_reduced_swirl_native_recorded,
        },
        reduced_swirl_production_cuda::prove_reduced_swirl_production_cuda,
        reduced_swirl_source_leaf::{
            reduced_swirl_source_tree_params, ReducedSwirlSourceTreeCudaProver,
        },
    },
    StdIn, F,
};
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::config::{
    app_params_with_100_bits_security, internal_params_with_100_bits_security,
    leaf_params_with_100_bits_security, params_with_100_bits_security,
};

const TEST_LOG_STACKED_HEIGHT: usize = 13;

fn transition_wrapper_params(
    native: &openvm_stark_backend::SystemParams,
) -> openvm_stark_backend::SystemParams {
    params_with_100_bits_security(
        native.log_blowup,
        native.l_skip,
        native.n_stack,
        native.w_stack,
        native.whir.folding_pow_bits,
        native.whir.mu_pow_bits,
        native.whir.proximity,
        8,
        native.whir.query_phase_pow_bits,
        native.whir.k,
        native.log_commit_rows_per_query,
    )
}

fn segmented_config() -> SystemConfig {
    let address_spaces = MemoryConfig::empty_address_space_configs(5);
    let mut config = SystemConfig::new(3, MemoryConfig::new(2, address_spaces, 8, 20, 10), 32);
    config.set_segmentation_max_memory(1);
    config
}

fn segmented_exe() -> Arc<VmExe<F>> {
    segmented_exe_with_nops(2_100)
}

fn segmented_exe_with_nops(nops: usize) -> Arc<VmExe<F>> {
    let nop = Instruction::from_isize(SystemOpcode::PHANTOM.global_opcode(), 0, 0, 0, 0, 0);
    let terminate = Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0);
    let mut instructions = vec![nop; nops];
    instructions.push(terminate);
    Arc::new(VmExe::new(Program::from_instructions(&instructions)))
}

#[test]
fn segmented_cuda_vm_reuses_original_stacking_codewords_for_every_source() -> Result<()> {
    let app_config = AppConfig::new(
        segmented_config(),
        app_params_with_100_bits_security(TEST_LOG_STACKED_HEIGHT),
    );
    let app_pk = AppProvingKey::keygen(app_config)?;
    let mut app =
        new_reduced_swirl_cuda_app_prover(SystemGpuBuilder, &app_pk.app_vm_pk, segmented_exe())?;
    let execution = prove_reduced_swirl_cuda_execution(&mut app, StdIn::default(), 2, 16)?;

    let source_count = execution.retained_prefixes.len();
    assert!(
        source_count > 1,
        "fixture must exercise a continuation chain"
    );
    assert_eq!(execution.authoritative_wrapper_claims.len(), source_count);
    assert_eq!(
        execution.native.native.authoritative_claims.len(),
        source_count
    );
    assert_eq!(execution.segment_metadata.len(), source_count);
    assert_eq!(
        execution
            .native
            .native
            .proof
            .statement
            .block_manifest_digest,
        reduced_swirl_manifest_digest(&execution.native.native.proof.statement.source_bindings)?
    );
    execution
        .native
        .telemetry
        .assert_no_duplicate_payload_pipeline()
        .map_err(eyre::Report::msg)?;
    assert_eq!(execution.native.telemetry.fresh_reencodes, 0);
    assert_eq!(execution.native.telemetry.fresh_recommits, 0);
    assert_eq!(execution.native.telemetry.accumulator_spill_count, 0);
    assert_eq!(execution.native.telemetry.accumulator_restore_count, 0);
    assert_eq!(execution.native.telemetry.terminal_reused_initial_roots, 1);
    assert_eq!(execution.native.telemetry.terminal_accumulator_reencodes, 0);
    assert_eq!(execution.native.telemetry.terminal_accumulator_recommits, 0);
    assert_eq!(
        execution.native.telemetry.terminal_full_message_d2h_bytes,
        0
    );

    let verification = verify_reduced_swirl_native_recorded(
        execution.setup.cpu_setup(),
        &execution.native.native.proof.statement,
        &execution.native.native.authoritative_claims,
        &execution.native.native.proof,
    )?;
    assert_eq!(
        verification.final_instance,
        execution.native.native.proof.vacc.final_instance
    );
    assert_eq!(verification.terminal.root, verification.final_instance.rt);
    assert_eq!(
        verification.transition_records.len(),
        execution.native.native.proof.vacc.steps.len()
    );
    assert_eq!(
        verification.swirl_power_batch_security,
        execution.native.native.swirl_power_batch_security
    );
    assert!(execution
        .native
        .native
        .proof
        .statement
        .source_bindings
        .iter()
        .all(|binding| binding.iter().any(|limb| *limb != F::ZERO)));
    Ok(())
}

#[test]
fn segmented_cuda_vm_streams_bounded_source_and_vacc_transition_leaves() -> Result<()> {
    // The combined transition leaf includes the bounded VACC verifier and its
    // Poseidon table in addition to the application traces.
    let app_params = app_params_with_100_bits_security(TEST_LOG_STACKED_HEIGHT + 2);
    let app_config = AppConfig::new(segmented_config(), app_params.clone());
    let app_pk = AppProvingKey::keygen(app_config)?;
    let mut app = new_reduced_swirl_cuda_app_prover(
        SystemGpuBuilder,
        &app_pk.app_vm_pk,
        segmented_exe_with_nops(10_500),
    )?;
    let wrapper_params = transition_wrapper_params(&app_params);
    let execution = prove_reduced_swirl_cuda_transition_execution(
        &mut app,
        StdIn::default(),
        8,
        16,
        wrapper_params,
    )?;

    assert!(
        execution
            .native
            .native
            .proof
            .statement
            .source_bindings
            .len()
            > 8
    );
    assert_eq!(
        execution.transition_leaf_proofs.len(),
        execution.native.native.proof.vacc.steps.len()
    );
    assert_eq!(execution.initial_transition_state.call_cursor, F::ZERO);
    assert_eq!(
        execution.final_transition_state.call_cursor,
        F::from_usize(execution.transition_leaf_proofs.len())
    );
    assert_eq!(
        execution.final_transition_state.source_cursor,
        F::from_usize(
            execution
                .native
                .native
                .proof
                .statement
                .source_bindings
                .len()
        )
    );
    assert_eq!(
        execution.final_transition_state.manifest_chain,
        execution.transition_chain_endpoint
    );
    assert_eq!(
        execution.final_transition_state.accumulator_root,
        execution.native.native.proof.vacc.final_instance.rt
    );
    Ok(())
}

#[test]
fn manifest_reconciliation_matches_transition_chain_for_production_schedule() -> Result<()> {
    const INPUT_ARITY: usize = 8;
    let source_protocol_digest = core::array::from_fn(|limb| F::from_usize(700 + limb));

    for source_count in [1usize, 8, 9, 100, 568, 1024] {
        let entry_digests = (0..source_count)
            .map(|source| core::array::from_fn(|limb| F::from_usize((source + 1) * 31 + limb + 1)))
            .collect::<Vec<_>>();
        let reconciliation = generate_reduced_swirl_manifest_reconciliation_trace(
            INPUT_ARITY,
            source_protocol_digest,
            &entry_digests,
        )
        .map_err(eyre::Report::msg)?;
        assert_eq!(
            reconciliation.receipt.flat_manifest_digest,
            reduced_swirl_manifest_digest(&entry_digests)?
        );

        let calls =
            reduced_swirl_vacc_schedule(source_count, INPUT_ARITY).map_err(eyre::Report::msg)?;
        let mut chain = reduced_swirl_transition_chain_genesis(source_protocol_digest);
        for call in &calls {
            let source_end = call.source_start + call.fresh_count;
            let local_manifest_digest =
                reduced_swirl_manifest_digest(&entry_digests[call.source_start..source_end])?;
            let transition = ReducedSwirlVaccTransitionReceiptMessage {
                protocol_digest: source_protocol_digest,
                relation_digest: [F::ONE; 8],
                warp_index_digest: [F::ONE; 8],
                schedule_digest: [F::ONE; 8],
                total_source_count: F::from_usize(source_count),
                call_index: F::from_usize(call.step),
                source_start: F::from_usize(call.source_start),
                fresh_count: F::from_usize(call.fresh_count),
                prior_count: F::from_usize(call.prior_count),
                batch_start_tidx: F::ZERO,
                vacc_start_tidx: F::ZERO,
                vacc_end_tidx: F::ZERO,
                start_sample_count: F::ZERO,
                start_state: [F::ZERO; 16],
                end_sample_count: F::ZERO,
                end_state: [F::ZERO; 16],
                prior_root: [F::ZERO; 8],
                output_root: [F::ZERO; 8],
                prior_digest: [F::ZERO; 8],
                output_digest: [F::ZERO; 8],
                manifest_digest: local_manifest_digest,
                is_final: F::from_bool(source_end == source_count),
            };
            chain =
                reduced_swirl_transition_chain_append(chain, &transition, local_manifest_digest);
        }

        assert_eq!(
            reconciliation.receipt.source_count,
            F::from_usize(source_count)
        );
        assert_eq!(
            reconciliation.receipt.call_count,
            F::from_usize(calls.len())
        );
        assert_eq!(reconciliation.receipt.rolling_chain_endpoint, chain);
    }
    Ok(())
}

#[test]
fn segmented_cuda_vm_emits_one_recursively_normalized_reduced_warp_proof() -> Result<()> {
    let app_config = AppConfig::new(
        segmented_config(),
        // The combined transition/finalizer AIR inventory has a log-height-14
        // trace in this tiny fixture. Keep the native constrained-RS geometry
        // exact while giving the fixture the same two-bit headroom as the
        // dedicated transition lifecycle test above.
        app_params_with_100_bits_security(TEST_LOG_STACKED_HEIGHT + 2),
    );
    let app_pk = AppProvingKey::keygen(app_config)?;
    let app =
        new_reduced_swirl_cuda_app_prover(SystemGpuBuilder, &app_pk.app_vm_pk, segmented_exe())?;
    let output = prove_reduced_swirl_production_cuda(
        app,
        StdIn::default(),
        REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY,
        16,
        leaf_params_with_100_bits_security(),
        internal_params_with_100_bits_security(),
    )?;

    assert!(output.telemetry.source_count > 1);
    assert_eq!(
        output.telemetry.vacc_call_count,
        output.native.proof.vacc.steps.len()
    );
    assert_eq!(
        output.native_verification.final_instance,
        output.native.proof.vacc.final_instance
    );
    assert_eq!(
        output.succinct_proof.inner.public_values.len(),
        output.succinct_vk.mvk.inner.per_air.len()
    );
    Ok(())
}

/// Exercise a long source-tree input through the setup-fixed wide prefix and
/// internal layers. This regression used to enter the shape-unstable generic
/// `RecursiveSelf` route after 28 transition leaves.
#[test]
fn segmented_cuda_vm_source_tree_accepts_long_fixed_fan_in() -> Result<()> {
    // This fixture deliberately emits enough VM rows to create at least 28
    // transition leaves.  Give those rows a setup-fixed application envelope
    // large enough to reach the source-tree code under test instead of
    // failing earlier in the application PCS layout.
    let app_params = app_params_with_100_bits_security(TEST_LOG_STACKED_HEIGHT + 5);
    let app_config = AppConfig::new(segmented_config(), app_params.clone());
    let app_pk = AppProvingKey::keygen(app_config)?;
    let mut app = new_reduced_swirl_cuda_app_prover(
        SystemGpuBuilder,
        &app_pk.app_vm_pk,
        segmented_exe_with_nops(240_000),
    )?;
    let wrapper_params = transition_wrapper_params(&app_params);
    let execution = prove_reduced_swirl_cuda_transition_execution(
        &mut app,
        StdIn::default(),
        REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY,
        16,
        wrapper_params,
    )?;
    assert!(
        execution.transition_leaf_proofs.len() >= 28,
        "fixture must exceed the former ternary source-tree capacity"
    );

    let tree_params = reduced_swirl_source_tree_params(&leaf_params_with_100_bits_security())?;
    ReducedSwirlSourceTreeCudaProver::new(execution.transition_leaf_vk, tree_params).prove(
        execution.transition_leaf_proofs,
        execution.recursive_app_vk_commit,
    )?;
    Ok(())
}

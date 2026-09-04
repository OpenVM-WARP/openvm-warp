#![cfg(not(feature = "cuda"))]

use std::sync::Arc;

use eyre::Result;
use openvm_circuit::{
    arch::{
        instructions::{
            exe::VmExe, instruction::Instruction, program::Program, LocalOpcode, SystemOpcode,
        },
        MemoryConfig, SystemConfig,
    },
    system::SystemCpuBuilder,
};
use openvm_sdk::{
    config::AppConfig,
    keygen::AppProvingKey,
    prover::{
        reduced_swirl_execution_cpu::prove_reduced_swirl_cpu_execution,
        reduced_swirl_native::{
            reduced_swirl_manifest_digest, verify_reduced_swirl_native_recorded,
        },
        AppProver,
    },
    StdIn, F,
};
use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
use openvm_stark_sdk::config::{
    app_params_with_100_bits_security, baby_bear_poseidon2::BabyBearPoseidon2CpuEngine,
};

const TEST_LOG_STACKED_HEIGHT: usize = 13;

fn segmented_config() -> SystemConfig {
    let address_spaces = MemoryConfig::empty_address_space_configs(5);
    let mut config = SystemConfig::new(3, MemoryConfig::new(2, address_spaces, 8, 20, 10), 32);
    config.set_segmentation_max_memory(1);
    config
}

fn segmented_exe() -> Arc<VmExe<F>> {
    let nop = Instruction::from_isize(SystemOpcode::PHANTOM.global_opcode(), 0, 0, 0, 0, 0);
    let terminate = Instruction::from_isize(SystemOpcode::TERMINATE.global_opcode(), 0, 0, 0, 0, 0);
    let mut instructions = vec![nop; 2_100];
    instructions.push(terminate);
    Arc::new(VmExe::new(Program::from_instructions(&instructions)))
}

#[test]
fn segmented_vm_stops_before_whir_and_accumulates_every_reduced_source() -> Result<()> {
    let app_config = AppConfig::new(
        segmented_config(),
        app_params_with_100_bits_security(TEST_LOG_STACKED_HEIGHT),
    );
    let app_pk = AppProvingKey::keygen(app_config)?;
    let mut app = AppProver::<BabyBearPoseidon2CpuEngine, SystemCpuBuilder>::new(
        SystemCpuBuilder,
        &app_pk.app_vm_pk,
        segmented_exe(),
    )?;
    let execution = prove_reduced_swirl_cpu_execution(&mut app, StdIn::default(), 2, 16)?;

    let source_count = execution.retained_prefixes.len();
    assert!(
        source_count > 1,
        "fixture must exercise a continuation chain"
    );
    assert_eq!(execution.authoritative_wrapper_claims.len(), source_count);
    assert_eq!(execution.native.authoritative_claims.len(), source_count);
    assert_eq!(execution.segment_metadata.len(), source_count);
    assert!(execution
        .native
        .proof
        .statement
        .source_bindings
        .iter()
        .all(|binding| binding.iter().any(|limb| *limb != F::ZERO)));
    assert_eq!(
        execution.native.proof.statement.block_manifest_digest,
        reduced_swirl_manifest_digest(&execution.native.proof.statement.source_bindings)?
    );

    let verification = verify_reduced_swirl_native_recorded(
        &execution.setup,
        &execution.native.proof.statement,
        &execution.native.authoritative_claims,
        &execution.native.proof,
    )?;
    assert_eq!(
        verification.final_instance,
        execution.native.proof.vacc.final_instance
    );
    assert_eq!(verification.terminal.root, verification.final_instance.rt);
    assert_eq!(
        verification.transition_records.len(),
        execution.native.proof.vacc.steps.len()
    );
    Ok(())
}

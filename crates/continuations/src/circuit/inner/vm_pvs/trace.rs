use std::borrow::{Borrow, BorrowMut};

use openvm_circuit::system::{connector::VmConnectorPvs, memory::merkle::MemoryMerklePvs};
use openvm_cpu_backend::CpuBackend;
use openvm_stark_backend::{proof::Proof, prover::AirProvingContext};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, DIGEST_SIZE, F};
use openvm_verify_stark_host::pvs::{VmPvs, VM_PVS_AIR_ID};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::dense::RowMajorMatrix;

use crate::circuit::inner::{app::*, vm_pvs::air::VmPvsCols, ProofsType};

#[derive(Clone, Copy)]
enum LeafProgramSource {
    CachedProgram,
    AuthenticatedVmPvs { vm_pvs_air_id: usize },
}

pub fn generate_proving_ctx(
    proofs: &[Proof<BabyBearPoseidon2Config>],
    proofs_type: ProofsType,
    child_is_app: bool,
    deferral_enabled: bool,
    required_height: Option<usize>,
) -> AirProvingContext<CpuBackend<BabyBearPoseidon2Config>> {
    generate_proving_ctx_with_leaf_source(
        proofs,
        proofs_type,
        child_is_app,
        deferral_enabled,
        required_height,
        LeafProgramSource::CachedProgram,
    )
}

/// Generate an app-style recursive prefix from a child proof which already
/// exposes one complete, proof-authenticated [`VmPvs`] vector. A companion
/// aggregation sub-circuit must consume that vector from `PublicValuesBus`
/// and bind it to `VerifierExecutionIdentityBus`; this function only builds
/// the ordinary `VmPvsAir` witness.
pub fn generate_proving_ctx_from_authenticated_vm_pvs(
    proofs: &[Proof<BabyBearPoseidon2Config>],
    proofs_type: ProofsType,
    deferral_enabled: bool,
    required_height: Option<usize>,
    vm_pvs_air_id: usize,
) -> AirProvingContext<CpuBackend<BabyBearPoseidon2Config>> {
    generate_proving_ctx_with_leaf_source(
        proofs,
        proofs_type,
        true,
        deferral_enabled,
        required_height,
        LeafProgramSource::AuthenticatedVmPvs { vm_pvs_air_id },
    )
}

fn generate_proving_ctx_with_leaf_source(
    proofs: &[Proof<BabyBearPoseidon2Config>],
    proofs_type: ProofsType,
    child_is_app: bool,
    deferral_enabled: bool,
    required_height: Option<usize>,
    leaf_program_source: LeafProgramSource,
) -> AirProvingContext<CpuBackend<BabyBearPoseidon2Config>> {
    debug_assert!(!proofs.is_empty());

    let num_vm_proofs = match proofs_type {
        ProofsType::Vm => proofs.len(),
        ProofsType::Deferral => 0,
        ProofsType::Mix | ProofsType::Combined => 1,
    };

    let natural_height = num_vm_proofs.next_power_of_two();
    let height = required_height.unwrap_or(natural_height);
    assert!(
        height.is_power_of_two() && height >= natural_height,
        "fixed VM-PVS height must be a power of two at least the natural height"
    );
    let base_width = VmPvsCols::<u8>::width();
    let width = base_width + deferral_enabled as usize;

    let mut trace = vec![F::ZERO; height * width];
    for (proof_idx, (proof, chunk)) in proofs[0..num_vm_proofs.max(1)]
        .iter()
        .zip(trace.chunks_exact_mut(width))
        .enumerate()
    {
        let (base_chunk, def_chunk) = chunk.split_at_mut(base_width);
        let cols: &mut VmPvsCols<F> = base_chunk.borrow_mut();
        cols.proof_idx = F::from_usize(proof_idx);

        if deferral_enabled {
            def_chunk[0] = match proofs_type {
                ProofsType::Vm | ProofsType::Mix => F::ZERO,
                ProofsType::Deferral => F::ONE,
                ProofsType::Combined => F::TWO,
            };
            if def_chunk[0] == F::ONE {
                continue;
            }
        }

        cols.is_valid = F::ONE;
        cols.is_last = F::from_bool(proof_idx + 1 == num_vm_proofs);

        if child_is_app {
            match leaf_program_source {
                LeafProgramSource::CachedProgram => {
                    cols.child_pvs.program_commit = proof.trace_vdata[PROGRAM_AIR_ID]
                        .as_ref()
                        .expect("program trace vdata must be present for app children")
                        .cached_commitments[PROGRAM_CACHED_TRACE_INDEX];

                    let &VmConnectorPvs {
                        initial_pc,
                        final_pc,
                        exit_code,
                        is_terminate,
                    } = proof.public_values[CONNECTOR_AIR_ID].as_slice().borrow();
                    cols.child_pvs.initial_pc = initial_pc;
                    cols.child_pvs.final_pc = final_pc;
                    cols.child_pvs.exit_code = exit_code;
                    cols.child_pvs.is_terminate = is_terminate;

                    let &MemoryMerklePvs::<_, DIGEST_SIZE> {
                        initial_root,
                        final_root,
                    } = proof.public_values[MERKLE_AIR_ID].as_slice().borrow();
                    cols.child_pvs.initial_root = initial_root;
                    cols.child_pvs.final_root = final_root;
                }
                LeafProgramSource::AuthenticatedVmPvs { vm_pvs_air_id } => {
                    let fields = proof
                        .public_values
                        .get(vm_pvs_air_id)
                        .expect("authenticated VmPvs AIR must be present");
                    assert_eq!(
                        fields.len(),
                        VmPvs::<u8>::width(),
                        "authenticated VmPvs width"
                    );
                    cols.child_pvs = *fields.as_slice().borrow();
                }
            }
        } else {
            cols.has_verifier_pvs = F::ONE;
            let child_pvs: &VmPvs<F> = proof.public_values[VM_PVS_AIR_ID].as_slice().borrow();
            cols.child_pvs = *child_pvs;
        }
    }

    let mut public_values = vec![F::ZERO; VmPvs::<u8>::width()];
    let pvs: &mut VmPvs<F> = public_values.as_mut_slice().borrow_mut();

    if num_vm_proofs > 0 {
        let first_row: &VmPvsCols<F> = trace[..base_width].borrow();
        let last_row: &VmPvsCols<F> =
            trace[(num_vm_proofs - 1) * width..(num_vm_proofs - 1) * width + base_width].borrow();

        pvs.program_commit = first_row.child_pvs.program_commit;
        pvs.initial_pc = first_row.child_pvs.initial_pc;
        pvs.initial_root = first_row.child_pvs.initial_root;

        pvs.final_pc = last_row.child_pvs.final_pc;
        pvs.exit_code = last_row.child_pvs.exit_code;
        pvs.is_terminate = last_row.child_pvs.is_terminate;
        pvs.final_root = last_row.child_pvs.final_root;
    }

    AirProvingContext {
        cached_mains: vec![],
        common_main: RowMajorMatrix::new(trace, width),
        public_values,
    }
}

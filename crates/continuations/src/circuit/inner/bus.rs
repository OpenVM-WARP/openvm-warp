use openvm_recursion_circuit::define_typed_per_proof_lookup_bus;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct PvsAirConsistencyMessage<T> {
    pub deferral_flag: T,
    pub has_verifier_pvs: T,
}

define_typed_per_proof_lookup_bus!(PvsAirConsistencyBus, PvsAirConsistencyMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct VerifierLayerIdentityMessage<T> {
    pub internal_flag: T,
    pub recursion_depth: T,
}

define_typed_per_proof_lookup_bus!(VerifierLayerIdentityBus, VerifierLayerIdentityMessage);

/// Complete VM execution identity read from an authenticated child proof.
/// History-v4 projections consume this beside their custom interval statement
/// so the recursive key chain and the History chain cannot describe different
/// programs or state boundaries.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct VerifierExecutionIdentityMessage<T> {
    pub program_commit: [T; DIGEST_SIZE],
    pub initial_pc: T,
    pub final_pc: T,
    pub exit_code: T,
    pub is_terminate: T,
    pub initial_root: [T; DIGEST_SIZE],
    pub final_root: [T; DIGEST_SIZE],
}

define_typed_per_proof_lookup_bus!(
    VerifierExecutionIdentityBus,
    VerifierExecutionIdentityMessage
);

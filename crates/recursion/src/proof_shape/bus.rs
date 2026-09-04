use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;

use crate::{define_typed_per_proof_permutation_bus, define_typed_permutation_bus};

pub const PROOF_SHAPE_METADATA_NUM_LIMBS: usize = 4;

/// Verifier-key metadata selected by one proof-shape row.
///
/// The large-key verifier uses a committed preprocessed table rather than a
/// degree-two selector over every AIR in the child key.  Keeping the complete
/// selected tuple on one permutation bus is important: splitting it across
/// buses would let a malicious witness combine fields from different AIRs.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ProofShapeMetadataMessage<T> {
    pub air_idx: T,
    pub is_required: T,
    pub need_rot: T,
    pub num_public_values: T,
    pub has_public_values: T,
    pub num_interactions: T,
    pub num_interactions_limbs: [T; PROOF_SHAPE_METADATA_NUM_LIMBS],
    pub main_width: T,
    pub is_min_cached: T,
    pub has_preprocessed: T,
    pub preprocessed_log_height: T,
    pub preprocessed_width: T,
    pub preprocessed_commit: [T; DIGEST_SIZE],
}

define_typed_permutation_bus!(ProofShapeMetadataBus, ProofShapeMetadataMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct ProofShapePermutationMessage<T> {
    pub idx: T,
}

define_typed_per_proof_permutation_bus!(ProofShapePermutationBus, ProofShapePermutationMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct StartingTidxMessage<T> {
    pub air_idx: T,
    pub tidx: T,
}

define_typed_per_proof_permutation_bus!(StartingTidxBus, StartingTidxMessage);

/// Caller-certified transcript checkpoint from which a partial verifier starts.
///
/// The companion rebasing AIR forwards the exact same `(tidx, state)` tuple to
/// `ResumeTranscriptStateBus`, so the transcript AIR and the protocol schedule
/// cannot be rebased independently.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct RebasedTranscriptStartMessage<T> {
    pub tidx: T,
    pub state: [T; POSEIDON2_WIDTH],
}

define_typed_per_proof_permutation_bus!(RebasedTranscriptStartBus, RebasedTranscriptStartMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NumPublicValuesMessage<T> {
    pub air_idx: T,
    pub tidx: T,
    pub num_pvs: T,
}

define_typed_per_proof_permutation_bus!(NumPublicValuesBus, NumPublicValuesMessage);

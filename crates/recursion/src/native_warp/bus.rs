use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::interaction::BusIndex;
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, D_EF};

use crate::{
    bus::{CertifiedTranscriptCheckpointBus, TranscriptBus},
    define_typed_lookup_bus, define_typed_permutation_bus,
    system::BusIndexManager,
};

#[derive(Clone, Debug)]
pub struct NativeWarpVerifierBusInventory {
    pub transcript: TranscriptBus,
    /// Transcript operations routed by the proof-indexed VACC cursor only
    /// after their canonical protocol role has been certified.
    pub vacc_semantic_transcript: TranscriptBus,
    pub vacc_phase_cursor: NativeVaccPhaseCursorBus,
    pub vacc_transcript_role: NativeVaccTranscriptRoleBus,
    pub transcript_checkpoint: CertifiedTranscriptCheckpointBus,
    pub sumcheck_round: NativeSumcheckRoundBus,
    pub sumcheck_initial: NativeSumcheckInitialBus,
    pub sumcheck_challenge: NativeSumcheckChallengeBus,
    pub claim_value: NativeClaimValueBus,
    pub fresh_source_activity: NativeFreshSourceActivityBus,
    pub claim_layout: NativeClaimLayoutBus,
    pub folded_claim: NativeFoldedClaimBus,
    pub twin_scalar: NativeTwinScalarBus,
    pub twin_omega: NativeTwinOmegaBus,
    pub opening_claim: NativeOpeningClaimBus,
    pub batching_output: NativeBatchingOutputBus,
    pub opening_leaf: NativeOpeningLeafBus,
    pub leaf_value: NativeLeafValueBus,
    pub merkle_node: NativeMerkleNodeBus,
    pub merkle_root: NativeMerkleRootBus,
    pub eq_result: NativeEqResultBus,
    pub vector_coordinate: NativeVectorCoordinateBus,
    pub authenticated_shift: NativeAuthenticatedShiftBus,
    pub direct_fresh_source: NativeDirectFreshSourceBus,
    pub direct_fresh_root: NativeDirectFreshRootBus,
    pub shift_index: NativeShiftIndexBus,
    pub fresh_count: NativeFreshCountBus,
    pub input_slot_layout: NativeInputSlotLayoutBus,
    pub accumulator_root: NativeAccumulatorRootBus,
    pub accumulator_digest_element: NativeAccumulatorDigestElementBus,
    pub accumulator_algebraic_digest: NativeAccumulatorAlgebraicDigestBus,
    next_bus_idx: BusIndex,
}

impl NativeWarpVerifierBusInventory {
    #[must_use]
    pub fn new(first_bus_idx: BusIndex) -> Self {
        let mut manager = BusIndexManager::from_next_bus_idx(first_bus_idx);
        let transcript = TranscriptBus::new(manager.new_bus_idx());
        let vacc_semantic_transcript = TranscriptBus::new(manager.new_bus_idx());
        let vacc_phase_cursor = NativeVaccPhaseCursorBus::new(manager.new_bus_idx());
        let vacc_transcript_role = NativeVaccTranscriptRoleBus::new(manager.new_bus_idx());
        let transcript_checkpoint = CertifiedTranscriptCheckpointBus::new(manager.new_bus_idx());
        let sumcheck_round = NativeSumcheckRoundBus::new(manager.new_bus_idx());
        let sumcheck_initial = NativeSumcheckInitialBus::new(manager.new_bus_idx());
        let sumcheck_challenge = NativeSumcheckChallengeBus::new(manager.new_bus_idx());
        let claim_value = NativeClaimValueBus::new(manager.new_bus_idx());
        let fresh_source_activity = NativeFreshSourceActivityBus::new(manager.new_bus_idx());
        let claim_layout = NativeClaimLayoutBus::new(manager.new_bus_idx());
        let folded_claim = NativeFoldedClaimBus::new(manager.new_bus_idx());
        let twin_scalar = NativeTwinScalarBus::new(manager.new_bus_idx());
        let twin_omega = NativeTwinOmegaBus::new(manager.new_bus_idx());
        let opening_claim = NativeOpeningClaimBus::new(manager.new_bus_idx());
        let batching_output = NativeBatchingOutputBus::new(manager.new_bus_idx());
        let opening_leaf = NativeOpeningLeafBus::new(manager.new_bus_idx());
        let leaf_value = NativeLeafValueBus::new(manager.new_bus_idx());
        let merkle_node = NativeMerkleNodeBus::new(manager.new_bus_idx());
        let merkle_root = NativeMerkleRootBus::new(manager.new_bus_idx());
        let eq_result = NativeEqResultBus::new(manager.new_bus_idx());
        let vector_coordinate = NativeVectorCoordinateBus::new(manager.new_bus_idx());
        let authenticated_shift = NativeAuthenticatedShiftBus::new(manager.new_bus_idx());
        let direct_fresh_source = NativeDirectFreshSourceBus::new(manager.new_bus_idx());
        let direct_fresh_root = NativeDirectFreshRootBus::new(manager.new_bus_idx());
        let shift_index = NativeShiftIndexBus::new(manager.new_bus_idx());
        let fresh_count = NativeFreshCountBus::new(manager.new_bus_idx());
        let input_slot_layout = NativeInputSlotLayoutBus::new(manager.new_bus_idx());
        let accumulator_root = NativeAccumulatorRootBus::new(manager.new_bus_idx());
        let accumulator_digest_element =
            NativeAccumulatorDigestElementBus::new(manager.new_bus_idx());
        let accumulator_algebraic_digest =
            NativeAccumulatorAlgebraicDigestBus::new(manager.new_bus_idx());
        let next_bus_idx = manager.new_bus_idx();
        Self {
            transcript,
            vacc_semantic_transcript,
            vacc_phase_cursor,
            vacc_transcript_role,
            transcript_checkpoint,
            sumcheck_round,
            sumcheck_initial,
            sumcheck_challenge,
            claim_value,
            fresh_source_activity,
            claim_layout,
            folded_claim,
            twin_scalar,
            twin_omega,
            opening_claim,
            batching_output,
            opening_leaf,
            leaf_value,
            merkle_node,
            merkle_root,
            eq_result,
            vector_coordinate,
            authenticated_shift,
            direct_fresh_source,
            direct_fresh_root,
            shift_index,
            fresh_count,
            input_slot_layout,
            accumulator_root,
            accumulator_digest_element,
            accumulator_algebraic_digest,
            next_bus_idx,
        }
    }

    #[must_use]
    pub const fn next_bus_idx(&self) -> BusIndex {
        self.next_bus_idx
    }
}

/// One verifier-key boundary in a proof-indexed standard VACC transcript.
/// Boundary 0 is the beginning of the VACC prefix, boundary 1 is the first
/// post-prefix semantic event, and boundary 2 is the terminator tag.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeVaccPhaseCursorMessage<T> {
    pub proof_idx: T,
    pub boundary: T,
    pub tidx: T,
}

define_typed_permutation_bus!(NativeVaccPhaseCursorBus, NativeVaccPhaseCursorMessage);

/// Exact semantic role of one transcript event after the protocol-order
/// cursor has consumed it from the real Fiat--Shamir transcript.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeVaccTranscriptRoleMessage<T> {
    pub proof_idx: T,
    pub role: T,
    pub ordinal: T,
    pub tidx: T,
    pub value: [T; D_EF],
    pub is_ext: T,
    pub is_sample: T,
}

define_typed_permutation_bus!(NativeVaccTranscriptRoleBus, NativeVaccTranscriptRoleMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeSumcheckRoundMessage<T> {
    pub proof_idx: T,
    pub kind: T,
    pub round: T,
    pub pre_claim: [T; D_EF],
    pub post_claim: [T; D_EF],
    pub challenge: [T; D_EF],
}

define_typed_permutation_bus!(NativeSumcheckRoundBus, NativeSumcheckRoundMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeSumcheckInitialMessage<T> {
    pub proof_idx: T,
    pub kind: T,
    pub claim: [T; D_EF],
}

define_typed_permutation_bus!(NativeSumcheckInitialBus, NativeSumcheckInitialMessage);

/// The batching sumcheck's initial claim authenticated by original-root and
/// prior-accumulator openings inside the native history relation.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeCertifiedBatchingClaimMessage<T> {
    pub proof_idx: T,
    pub claim: [T; D_EF],
}

define_typed_permutation_bus!(
    NativeCertifiedBatchingClaimBus,
    NativeCertifiedBatchingClaimMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeSumcheckChallengeMessage<T> {
    pub proof_idx: T,
    pub kind: T,
    pub round: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(NativeSumcheckChallengeBus, NativeSumcheckChallengeMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeClaimValueMessage<T> {
    pub proof_idx: T,
    pub source: T,
    pub section: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(NativeClaimValueBus, NativeClaimValueMessage);

/// Fixed source-slot activity used by the row-oriented fresh-claim binding.
///
/// The source descriptor provides one lookup-table key with a compile-time
/// multiplicity equal to the number of claim rows for that source. Every claim
/// row looks up the same `(source, active)` pair.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeFreshSourceActivityMessage<T> {
    pub source: T,
    pub active: T,
}

define_typed_lookup_bus!(
    NativeFreshSourceActivityBus,
    NativeFreshSourceActivityMessage
);

/// One base-field element of the canonical, fixed-width fresh-input digest
/// preimage.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeFreshDigestElementMessage<T> {
    pub index: T,
    pub value: T,
}

define_typed_permutation_bus!(NativeFreshDigestElementBus, NativeFreshDigestElementMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeClaimLayoutMessage<T> {
    pub section: T,
    pub coordinate: T,
    pub is_beta_tail: T,
    pub tail_coordinate: T,
}

define_typed_lookup_bus!(NativeClaimLayoutBus, NativeClaimLayoutMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeFoldedClaimMessage<T> {
    pub proof_idx: T,
    pub section: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(NativeFoldedClaimBus, NativeFoldedClaimMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTwinScalarMessage<T> {
    pub proof_idx: T,
    pub kind: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(NativeTwinScalarBus, NativeTwinScalarMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTwinOmegaMessage<T> {
    pub proof_idx: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(NativeTwinOmegaBus, NativeTwinOmegaMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeOpeningClaimMessage<T> {
    pub proof_idx: T,
    pub claim: T,
    /// 0 = point coordinate, 1 = target scalar.
    pub section: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(NativeOpeningClaimBus, NativeOpeningClaimMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeBatchingOutputMessage<T> {
    pub proof_idx: T,
    /// 0 = alpha coordinate, 1 = mu.
    pub section: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(NativeBatchingOutputBus, NativeBatchingOutputMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeAuthenticatedShiftMessage<T> {
    pub proof_idx: T,
    pub shift: T,
    pub source: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(NativeAuthenticatedShiftBus, NativeAuthenticatedShiftMessage);

/// Descriptor shared by the original-root commitment and its on-demand
/// extension-field projections. The commitment table provides one key per
/// source with enough multiplicity for every root row and shift query.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeDirectFreshSourceMessage<T> {
    pub source: T,
    pub active: T,
    pub root_count: T,
    pub first_tree_id: T,
    pub commitment_tidx: T,
    pub theta: [T; D_EF],
}

define_typed_lookup_bus!(NativeDirectFreshSourceBus, NativeDirectFreshSourceMessage);

/// One original base-field root in a direct fresh-source descriptor.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeDirectFreshRootMessage<T> {
    pub source: T,
    pub root_ordinal: T,
    pub tree_id: T,
    pub width: T,
    pub root: [T; DIGEST_SIZE],
}

define_typed_lookup_bus!(NativeDirectFreshRootBus, NativeDirectFreshRootMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeShiftIndexMessage<T> {
    pub proof_idx: T,
    pub shift: T,
    pub index: T,
}

define_typed_lookup_bus!(NativeShiftIndexBus, NativeShiftIndexMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeFreshCountMessage<T> {
    pub count: T,
}

define_typed_lookup_bus!(NativeFreshCountBus, NativeFreshCountMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeInputSlotLayoutMessage<T> {
    pub variant: T,
    pub source: T,
    /// Fresh, prior accumulator, dummy.
    pub kind: [T; 3],
}

define_typed_lookup_bus!(NativeInputSlotLayoutBus, NativeInputSlotLayoutMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeAccumulatorRootMessage<T> {
    pub proof_idx: T,
    /// 0 = prior input, 1 = output accumulator.
    pub state: T,
    pub digest: [T; DIGEST_SIZE],
}

define_typed_permutation_bus!(NativeAccumulatorRootBus, NativeAccumulatorRootMessage);

/// One base-field element in the canonical accumulator-instance hash preimage.
/// `state = 0` identifies the private prior accumulator and `state = 1` the
/// VACC output accumulator.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeAccumulatorDigestElementMessage<T> {
    pub proof_idx: T,
    pub state: T,
    pub index: T,
    pub value: T,
}

define_typed_permutation_bus!(
    NativeAccumulatorDigestElementBus,
    NativeAccumulatorDigestElementMessage
);

/// Poseidon sponge output for the algebraic portion
/// `(alpha, mu, beta, eta)` of one accumulator instance.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeAccumulatorAlgebraicDigestMessage<T> {
    pub proof_idx: T,
    pub state: T,
    pub digest: [T; DIGEST_SIZE],
}

define_typed_permutation_bus!(
    NativeAccumulatorAlgebraicDigestBus,
    NativeAccumulatorAlgebraicDigestMessage
);

/// Verifier endpoint input exported by the transcript-facing reduction AIRs.
///
/// This is a lookup bus because the same authenticated opening or challenge
/// can be used by both the symbolic AIR endpoint and the stacking endpoint.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeReductionEndpointInputMessage<T> {
    pub reduction: T,
    pub kind: T,
    pub index: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    NativeReductionEndpointInputBus,
    NativeReductionEndpointInputMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeOpeningLeafMessage<T> {
    pub proof_idx: T,
    pub tree_id: T,
    pub index: T,
    pub digest: [T; DIGEST_SIZE],
}

define_typed_permutation_bus!(NativeOpeningLeafBus, NativeOpeningLeafMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeLeafValueMessage<T> {
    pub proof_idx: T,
    pub tree_id: T,
    pub index: T,
    pub position: T,
    pub value: T,
}

define_typed_lookup_bus!(NativeLeafValueBus, NativeLeafValueMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeMerkleNodeMessage<T> {
    pub proof_idx: T,
    pub tree_id: T,
    pub level: T,
    pub index: T,
    pub digest: [T; DIGEST_SIZE],
}

define_typed_permutation_bus!(NativeMerkleNodeBus, NativeMerkleNodeMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeMerkleRootMessage<T> {
    pub proof_idx: T,
    pub tree_id: T,
    pub depth: T,
    pub digest: [T; DIGEST_SIZE],
}

define_typed_permutation_bus!(NativeMerkleRootBus, NativeMerkleRootMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeEqResultMessage<T> {
    pub proof_idx: T,
    pub group: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(NativeEqResultBus, NativeEqResultMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeVectorCoordinateMessage<T> {
    pub proof_idx: T,
    pub vector: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(NativeVectorCoordinateBus, NativeVectorCoordinateMessage);

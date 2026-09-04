use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, D_EF};

use crate::{define_typed_lookup_bus, define_typed_permutation_bus};

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirBetaCoordinateMessage<T> {
    pub coordinate: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirBetaCoordinateBus,
    FixedMultiAirBetaCoordinateMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirTerminalBindingMessage<T> {
    pub relation_digest: [T; DIGEST_SIZE],
    pub root: [T; DIGEST_SIZE],
    pub alpha_len: T,
    pub beta_len: T,
}

define_typed_permutation_bus!(
    FixedMultiAirTerminalBindingBus,
    FixedMultiAirTerminalBindingMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirTerminalInstanceValueMessage<T> {
    /// 0 = alpha, 1 = mu, 2 = beta, 3 = eta.
    pub section: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirTerminalInstanceValueBus,
    FixedMultiAirTerminalInstanceValueMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirGlobalClaimMessage<T> {
    pub eta: [T; D_EF],
    pub one: [T; D_EF],
}

define_typed_lookup_bus!(FixedMultiAirGlobalClaimBus, FixedMultiAirGlobalClaimMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirRegionClaimMessage<T> {
    pub region: T,
    pub claim: [T; D_EF],
}

define_typed_lookup_bus!(FixedMultiAirRegionClaimBus, FixedMultiAirRegionClaimMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirPaddingClaimMessage<T> {
    pub present: T,
    pub claim: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirPaddingClaimBus,
    FixedMultiAirPaddingClaimMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirPaddingPointMessage<T> {
    pub coordinate: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirPaddingPointBus,
    FixedMultiAirPaddingPointMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirPaddingSumcheckFinalMessage<T> {
    /// Transcript index immediately before the padding-opening tag.
    pub tidx: T,
    pub claim: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirPaddingSumcheckFinalBus,
    FixedMultiAirPaddingSumcheckFinalMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirPaddingOpeningMessage<T> {
    /// Transcript index after the padding-opening observation.
    pub tidx: T,
    pub claim: [T; D_EF],
    pub opening: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirPaddingOpeningBus,
    FixedMultiAirPaddingOpeningMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirRegionStartMessage<T> {
    pub region: T,
    pub tidx: T,
    pub claim: [T; D_EF],
}

define_typed_permutation_bus!(FixedMultiAirRegionStartBus, FixedMultiAirRegionStartMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirRegionPointMessage<T> {
    pub region: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(FixedMultiAirRegionPointBus, FixedMultiAirRegionPointMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirRegionSumcheckFinalMessage<T> {
    pub region: T,
    /// Transcript index immediately before the opened-column tag.
    pub tidx: T,
    pub claim: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirRegionSumcheckFinalBus,
    FixedMultiAirRegionSumcheckFinalMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirDecompositionContributionMessage<T> {
    pub component: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirDecompositionContributionBus,
    FixedMultiAirDecompositionContributionMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirRegionOpeningMessage<T> {
    pub region: T,
    pub opening: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirRegionOpeningBus,
    FixedMultiAirRegionOpeningMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirRegionRhoMessage<T> {
    pub region: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(FixedMultiAirRegionRhoBus, FixedMultiAirRegionRhoMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirRegionLocalPaddingPointMessage<T> {
    pub region: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirRegionLocalPaddingPointBus,
    FixedMultiAirRegionLocalPaddingPointMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirRegionFinalEvaluationMessage<T> {
    pub region: T,
    pub claim: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirRegionFinalEvaluationBus,
    FixedMultiAirRegionFinalEvaluationMessage
);

/// Header of one exact generic `TerminalStructuredLinearClaim`.
/// `kind = 0` denotes `PrismalinearMappedColumns`; `kind = 1` denotes `Eq`.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirStructuredClaimHeaderMessage<T> {
    pub claim: T,
    pub kind: T,
    pub log_message_len: T,
    pub term_count: T,
    pub point_len: T,
    pub target: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirStructuredClaimHeaderBus,
    FixedMultiAirStructuredClaimHeaderMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirMappedTermMessage<T> {
    pub claim: T,
    pub term: T,
    pub block_start: T,
    pub log_height: T,
    pub l_skip: T,
    pub rotation: T,
    pub scale: [T; D_EF],
}

define_typed_permutation_bus!(FixedMultiAirMappedTermBus, FixedMultiAirMappedTermMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirStructuredPointMessage<T> {
    pub claim: T,
    pub term: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirStructuredPointBus,
    FixedMultiAirStructuredPointMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirWhirStartMessage<T> {
    pub tidx: T,
    pub root: [T; DIGEST_SIZE],
    pub batching_challenge: [T; D_EF],
    pub mu: [T; D_EF],
}

define_typed_permutation_bus!(FixedMultiAirWhirStartBus, FixedMultiAirWhirStartMessage);

/// One verifier-derived structured claim after assigning its exact `xi`
/// power.  This is consumed by the structured RS-dual endpoint verifier.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirBatchedClaimMessage<T> {
    pub claim: T,
    pub kind: T,
    pub log_message_len: T,
    pub term_count: T,
    pub point_len: T,
    pub target: [T; D_EF],
    pub batching_scale: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirBatchedClaimBus,
    FixedMultiAirBatchedClaimMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirLinearizerAuxPointMessage<T> {
    pub coordinate: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirLinearizerAuxPointBus,
    FixedMultiAirLinearizerAuxPointMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirLinearizerSumcheckFinalMessage<T> {
    pub initial_claim: [T; D_EF],
    pub final_claim: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirLinearizerSumcheckFinalBus,
    FixedMultiAirLinearizerSumcheckFinalMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirLinearizerYMessage<T> {
    pub message_bit: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(FixedMultiAirLinearizerYBus, FixedMultiAirLinearizerYMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirLinearizerSelectorMessage<T> {
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirLinearizerSelectorBus,
    FixedMultiAirLinearizerSelectorMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirLinearizerRawTermMessage<T> {
    pub ordinal: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirLinearizerRawTermBus,
    FixedMultiAirLinearizerRawTermMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirLinearizerRawWeightMessage<T> {
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirLinearizerRawWeightBus,
    FixedMultiAirLinearizerRawWeightMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirEndpointNodeMessage<T> {
    pub region: T,
    pub node: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirEndpointNodeBus,
    FixedMultiAirEndpointNodeMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirEndpointFixedValueMessage<T> {
    pub region: T,
    pub source: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirEndpointFixedValueBus,
    FixedMultiAirEndpointFixedValueMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirEndpointFixedFoldStateMessage<T> {
    pub region: T,
    pub source: T,
    pub layer: T,
    pub index: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirEndpointFixedFoldStateBus,
    FixedMultiAirEndpointFixedFoldStateMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirConstraintWeightMessage<T> {
    pub region: T,
    pub constraint: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirConstraintWeightBus,
    FixedMultiAirConstraintWeightMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirConstraintWeightStateMessage<T> {
    pub region: T,
    pub layer: T,
    pub carry: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirConstraintWeightStateBus,
    FixedMultiAirConstraintWeightStateMessage
);

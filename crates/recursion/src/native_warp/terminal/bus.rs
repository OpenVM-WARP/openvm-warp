use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, D_EF};

use crate::{define_typed_lookup_bus, define_typed_permutation_bus};

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalWhirRoundMessage<T> {
    pub round: T,
    pub tidx: T,
    pub pre_claim: [T; D_EF],
    pub post_sumcheck_claim: [T; D_EF],
}

define_typed_permutation_bus!(NativeTerminalWhirRoundBus, NativeTerminalWhirRoundMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalWhirAlphaMessage<T> {
    pub round: T,
    pub fold: T,
    pub challenge: [T; D_EF],
}

define_typed_lookup_bus!(NativeTerminalWhirAlphaBus, NativeTerminalWhirAlphaMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalWhirVerifyQueriesMessage<T> {
    pub round: T,
    pub tidx: T,
    pub num_queries: T,
    pub omega: T,
    pub gamma: [T; D_EF],
    pub pre_claim: [T; D_EF],
    pub post_claim: [T; D_EF],
}

define_typed_permutation_bus!(
    NativeTerminalWhirVerifyQueriesBus,
    NativeTerminalWhirVerifyQueriesMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalWhirQueryMessage<T> {
    pub round: T,
    pub query: T,
    pub global_query: T,
    pub inner_tree_id: T,
    pub outer_tree_id: T,
    pub merkle_index_sample: T,
    pub merkle_index: T,
    pub zi_root: T,
    pub zi: T,
    pub yi: [T; D_EF],
}

define_typed_permutation_bus!(NativeTerminalWhirQueryBus, NativeTerminalWhirQueryMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalWhirFoldingMessage<T> {
    pub round: T,
    pub query: T,
    pub height: T,
    pub coset_shift: T,
    pub coset_size: T,
    pub coset_index: T,
    pub twiddle: T,
    pub value: [T; D_EF],
    pub z_final: T,
    pub y_final: [T; D_EF],
}

define_typed_permutation_bus!(
    NativeTerminalWhirFoldingBus,
    NativeTerminalWhirFoldingMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalWhirStatementMessage<T> {
    pub tidx: T,
    pub root: [T; DIGEST_SIZE],
    pub batching_challenge: [T; D_EF],
    pub initial_claim: [T; D_EF],
}

define_typed_lookup_bus!(
    NativeTerminalWhirStatementBus,
    NativeTerminalWhirStatementMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalWhirFinalClaimMessage<T> {
    pub final_poly_tidx: T,
    pub tidx: T,
    pub claim: [T; D_EF],
}

define_typed_permutation_bus!(
    NativeTerminalWhirFinalClaimBus,
    NativeTerminalWhirFinalClaimMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalWhirFinalContextMessage<T> {
    pub final_weight_tidx: T,
    pub claim: [T; D_EF],
}

define_typed_lookup_bus!(
    NativeTerminalWhirFinalContextBus,
    NativeTerminalWhirFinalContextMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalWhirPointMessage<T> {
    pub coordinate: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(NativeTerminalWhirPointBus, NativeTerminalWhirPointMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalWhirFinalPolyMessage<T> {
    pub layer: T,
    pub index: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(
    NativeTerminalWhirFinalPolyBus,
    NativeTerminalWhirFinalPolyMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalWhirFinalWeightMessage<T> {
    pub index: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(
    NativeTerminalWhirFinalWeightBus,
    NativeTerminalWhirFinalWeightMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalWhirActualWeightMessage<T> {
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(
    NativeTerminalWhirActualWeightBus,
    NativeTerminalWhirActualWeightMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalRsAdjointRoundMessage<T> {
    pub round: T,
    pub challenge: [T; D_EF],
}

define_typed_lookup_bus!(
    NativeTerminalRsAdjointRoundBus,
    NativeTerminalRsAdjointRoundMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalRsAdjointClaimMessage<T> {
    pub claimed_value: [T; D_EF],
    pub final_claim: [T; D_EF],
}

define_typed_permutation_bus!(
    NativeTerminalRsAdjointClaimBus,
    NativeTerminalRsAdjointClaimMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalAccumulatorValueMessage<T> {
    /// 0 = alpha, 1 = mu, 2 = beta, 3 = eta.
    pub section: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    NativeTerminalAccumulatorValueBus,
    NativeTerminalAccumulatorValueMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalAccumulatorRootMessage<T> {
    pub root: [T; DIGEST_SIZE],
}

define_typed_permutation_bus!(
    NativeTerminalAccumulatorRootBus,
    NativeTerminalAccumulatorRootMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalRsAdjointQMessage<T> {
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(NativeTerminalRsAdjointQBus, NativeTerminalRsAdjointQMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalRsAdjointYMessage<T> {
    pub message_bit: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(NativeTerminalRsAdjointYBus, NativeTerminalRsAdjointYMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalRsAdjointValueMessage<T> {
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(
    NativeTerminalRsAdjointValueBus,
    NativeTerminalRsAdjointValueMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalWhirWeightTermMessage<T> {
    pub ordinal: T,
    pub after_folds: T,
    pub length: T,
    pub generator: [T; D_EF],
    pub scale: [T; D_EF],
}

define_typed_permutation_bus!(
    NativeTerminalWhirWeightTermBus,
    NativeTerminalWhirWeightTermMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalWhirWeightTermResultMessage<T> {
    pub ordinal: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(
    NativeTerminalWhirWeightTermResultBus,
    NativeTerminalWhirWeightTermResultMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalWhirLinearizerWeightMessage<T> {
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(
    NativeTerminalWhirLinearizerWeightBus,
    NativeTerminalWhirLinearizerWeightMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalLinearizerDescriptorMessage<T> {
    pub tidx: T,
    pub root: [T; DIGEST_SIZE],
}

define_typed_permutation_bus!(
    NativeTerminalLinearizerDescriptorBus,
    NativeTerminalLinearizerDescriptorMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeTerminalLinearizerEndMessage<T> {
    pub tidx: T,
    pub root: [T; DIGEST_SIZE],
    pub mu: [T; D_EF],
    pub beta_last: [T; D_EF],
    pub eta: [T; D_EF],
}

define_typed_permutation_bus!(
    NativeTerminalLinearizerEndBus,
    NativeTerminalLinearizerEndMessage
);

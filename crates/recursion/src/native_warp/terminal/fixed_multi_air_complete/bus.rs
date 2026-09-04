use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_sdk::config::baby_bear_poseidon2::D_EF;

use crate::{define_typed_lookup_bus, define_typed_permutation_bus};

/// Authenticated entry into one setup-fixed complete-terminal region.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteRegionStartMessage<T> {
    pub region: T,
    /// Native transcript index of the local/interaction region domain tag.
    pub tidx: T,
    pub claim: [T; D_EF],
}

// Local and interaction obligations intentionally use distinct typed buses.
// A witness cannot select a component kind and cannot route a local claim into
// an interaction-region verifier (or conversely).
define_typed_permutation_bus!(
    FixedMultiAirCompleteLocalRegionStartBus,
    FixedMultiAirCompleteRegionStartMessage
);
define_typed_permutation_bus!(
    FixedMultiAirCompleteInteractionRegionStartBus,
    FixedMultiAirCompleteRegionStartMessage
);

/// One Boolean-cube coordinate sampled by a complete regional sumcheck.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteRegionPointMessage<T> {
    pub region: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirCompleteLocalRegionPointBus,
    FixedMultiAirCompleteRegionPointMessage
);
define_typed_lookup_bus!(
    FixedMultiAirCompleteInteractionRegionPointBus,
    FixedMultiAirCompleteRegionPointMessage
);

/// Authenticated output of one complete regional degree-seven sumcheck.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteRegionSumcheckFinalMessage<T> {
    pub region: T,
    /// Native transcript index immediately before the openings domain tag.
    pub tidx: T,
    pub claim: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirCompleteLocalRegionSumcheckFinalBus,
    FixedMultiAirCompleteRegionSumcheckFinalMessage
);
define_typed_permutation_bus!(
    FixedMultiAirCompleteInteractionRegionSumcheckFinalBus,
    FixedMultiAirCompleteRegionSumcheckFinalMessage
);

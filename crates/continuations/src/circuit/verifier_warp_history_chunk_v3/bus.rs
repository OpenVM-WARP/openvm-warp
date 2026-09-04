//! Typed buses for exhaustive History-v3 interval statements.

use openvm_stark_backend::interaction::{BusIndex, InteractionBuilder, LookupBus};

use super::VerifierWarpHistoryChunkIntervalMessageV3;

/// A typed 108-coordinate lookup bus.
///
/// Left input, right input, and merged output use distinct instances of this
/// type.  Distinct indices make child order part of the fixed circuit wiring
/// instead of relying on an unordered multiset to infer orientation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VerifierWarpHistoryChunkIntervalBusV3(LookupBus);

impl VerifierWarpHistoryChunkIntervalBusV3 {
    #[must_use]
    pub const fn new(index: BusIndex) -> Self {
        Self(LookupBus::new(index))
    }

    #[must_use]
    pub const fn index(self) -> BusIndex {
        self.0.index
    }

    pub fn lookup_key<AB>(
        &self,
        builder: &mut AB,
        message: VerifierWarpHistoryChunkIntervalMessageV3<impl Into<AB::Expr> + Clone>,
        enabled: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
    {
        self.0.lookup_key(builder, message.to_vec(), enabled);
    }

    pub fn add_key_with_lookups<AB>(
        &self,
        builder: &mut AB,
        message: VerifierWarpHistoryChunkIntervalMessageV3<impl Into<AB::Expr> + Clone>,
        lookups: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
    {
        self.0
            .add_key_with_lookups(builder, message.to_vec(), lookups);
    }
}

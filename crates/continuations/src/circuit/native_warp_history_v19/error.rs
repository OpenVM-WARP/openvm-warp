use core::fmt;

/// Deterministic witness-generation and verifier-profile failures.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum HistoryV19Error {
    InvalidProfile(&'static str),
    EmptyHistory,
    TooManySegments {
        maximum: usize,
        actual: usize,
    },
    TooManyShards {
        segment: usize,
        maximum: usize,
        actual: usize,
    },
    SegmentIndexMismatch {
        segment: usize,
    },
    VmBoundaryDiscontinuity {
        segment: usize,
    },
    SegmentAfterTermination {
        segment: usize,
    },
    ProductRootMismatch {
        segment: usize,
        shard: Option<usize>,
    },
    LogUpModeMismatch {
        segment: usize,
    },
    NonCanonicalShardOrder {
        segment: usize,
        shard: usize,
    },
    DuplicateShardKey {
        segment: usize,
        shard: usize,
    },
    ShardOrdinalOutOfRange {
        segment: usize,
        shard: usize,
    },
    ShardAppVkMismatch {
        segment: usize,
        shard: usize,
    },
    MerklePathLengthMismatch {
        segment: usize,
        shard: usize,
    },
    ShardCatalogAuthenticationFailed {
        segment: usize,
        shard: usize,
    },
    ReplayBindingMismatch {
        segment: usize,
        shard: usize,
    },
    ManifestDigestMismatch {
        segment: usize,
    },
    FinalBoundaryMismatch,
    TerminalStateRequired,
    TraceCapacityExceeded {
        maximum: usize,
        actual: usize,
    },
    IntegerOverflow,
}

impl fmt::Display for HistoryV19Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for HistoryV19Error {}

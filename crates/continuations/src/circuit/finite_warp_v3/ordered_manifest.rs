//! Ordered finite-source manifest and prefix-link receipt producer.
//!
//! This module is deliberately a receipt *consumer* at its two cryptographic
//! inputs and a receipt producer at the fixed WARP-v3 boundary:
//!
//! - `source_authority` must be populated by the complete fixed-PESAT source verifier, never by a
//!   host boolean;
//! - `prefix_authority` must be populated by the recursive verifier for the concrete
//!   prefix-equality PCS proof;
//! - the AIRs below validate ordering, occupancy, VM continuity, and the exact native manifest
//!   transcript before publishing `manifest` and `execution` receipts consumed by
//!   `FiniteWarpV3StatementAir`.
//!
//! The manifest transcript constants and observation order are copied from
//! `sdk::prover::native_warp::finite_manifest`. They intentionally remain
//! protocol-versioned here: changing either side is a protocol change and the
//! dynamic transcript differential tests in `ordered_manifest_tests.rs` must
//! fail until both are reviewed together.

use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_recursion_circuit::{
    bus::TranscriptBus,
    native_warp::{NativeWarpTranscriptArtifacts, NativeWarpTranscriptModule},
    system::{BusIndexManager, BusInventory},
};
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, LookupBus},
    p3_field::PrimeCharacteristicRing,
    transcript::{TranscriptHistory, TranscriptLog},
    AirRef, FiatShamirTranscript, StarkProtocolConfig, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, Digest, DIGEST_SIZE, D_EF, EF, F,
};

use super::{
    FiniteWarpV3ReceiptBuses, FINITE_WARP_V3_MAX_CALLS, FINITE_WARP_V3_MAX_INPUT_ARITY,
    FINITE_WARP_V3_MAX_SOURCES,
};

#[path = "ordered_manifest_air.rs"]
mod ordered_manifest_air;
pub use ordered_manifest_air::*;

#[cfg(test)]
#[path = "ordered_manifest_tests.rs"]
mod ordered_manifest_tests;

/// Native finite-manifest transcript version.
pub const ORDERED_MANIFEST_PROTOCOL_VERSION: u32 = 3;
/// Native prefix-link transcript version accepted by this wrapper profile.
pub const ORDERED_MANIFEST_PREFIX_PROTOCOL_VERSION: u32 = 4;
/// Maximum number of normalized recursive proofs bound by one wrapper.
pub const ORDERED_MANIFEST_MAX_NORMALIZED_LEAVES: u32 = 1024;
/// Prefix-link occupancy vector width.
pub const ORDERED_MANIFEST_PREFIX_SLOTS: usize = 64;

// Exact native finite-manifest domain separators. All are canonical BabyBear
// values and therefore can be observed directly as one field element.
pub(crate) const SCHEDULE_DIGEST_TAG: u32 = 0x4653_0101;
pub(crate) const SCHEDULE_START_TAG: u32 = 0x4653_0102;
pub(crate) const SCHEDULE_CALL_TAG: u32 = 0x4653_0103;
pub(crate) const SCHEDULE_END_TAG: u32 = 0x4653_0104;
pub(crate) const MANIFEST_DIGEST_TAG: u32 = 0x4653_0201;
pub(crate) const MANIFEST_START_TAG: u32 = 0x4653_0202;
pub(crate) const MANIFEST_INVOCATION_TAG: u32 = 0x4653_0203;
pub(crate) const MANIFEST_SOURCE_TAG: u32 = 0x4653_0204;
pub(crate) const MANIFEST_FINAL_TAG: u32 = 0x4653_0205;
pub(crate) const MANIFEST_END_TAG: u32 = 0x4653_0206;
pub(crate) const SOURCE_ORDER_DIGEST_TAG: u32 = 0x4653_0207;
pub(crate) const OBSERVATION_FIELD_TAG: u32 = 0x4653_0301;
pub(crate) const OBSERVATION_DIGEST_TAG: u32 = 0x4653_0302;
pub(crate) const OBSERVATION_END_TAG: u32 = 0x4653_0303;

/// Exact setup-owned prefix statement fields needed to authenticate the
/// source manifest against the prefix-equality PCS verifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrderedManifestPrefixProfile {
    pub index_digest: Digest,
    pub l_skip: u32,
    pub log_message_len: u32,
    pub log_blowup: u32,
    pub log_codeword_len: u32,
    pub rows_per_leaf: u32,
    pub trace_prefix_len: u64,
    pub active_count_block_start: u64,
    pub active_count_log_height: u32,
}

/// VK-owned manifest profile. Counts remain proof-dependent but bounded; the
/// normalized-leaf total is setup-owned so a witness cannot silently shorten
/// the supplied recursive-proof sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderedManifestProfile {
    pub protocol_digest: Digest,
    pub relation_digest: Digest,
    pub warp_index_digest: Digest,
    pub verifier_component_digest: Digest,
    pub prefix: OrderedManifestPrefixProfile,
    pub expected_normalized_leaf_count: u32,
    /// Supported values are 1, 2, 4, or 8. The current complete verifier
    /// relation uses two normalized children per source.
    pub max_active_children_per_source: u8,
    pub suspend_exit_code: u32,
}

impl OrderedManifestProfile {
    pub fn validate(&self) -> Result<(), OrderedManifestError> {
        for (name, digest) in [
            ("protocol digest", self.protocol_digest),
            ("relation digest", self.relation_digest),
            ("WARP index digest", self.warp_index_digest),
            ("component digest", self.verifier_component_digest),
            ("prefix index digest", self.prefix.index_digest),
        ] {
            if digest.iter().all(|value| *value == F::ZERO) {
                return Err(OrderedManifestError::InvalidProfile(name));
            }
        }
        if self.expected_normalized_leaf_count == 0
            || self.expected_normalized_leaf_count > ORDERED_MANIFEST_MAX_NORMALIZED_LEAVES
            || !matches!(self.max_active_children_per_source, 1 | 2 | 4 | 8)
            || self.prefix.l_skip != 0
            || self.prefix.log_codeword_len
                != self
                    .prefix
                    .log_message_len
                    .checked_add(self.prefix.log_blowup)
                    .ok_or(OrderedManifestError::IntegerOverflow)?
            || self.prefix.rows_per_leaf != 16
            || self.prefix.trace_prefix_len == 0
            || self.prefix.active_count_log_height >= 63
        {
            return Err(OrderedManifestError::InvalidProfile("manifest bounds"));
        }
        let message_len = 1u64.checked_shl(self.prefix.log_message_len).ok_or(
            OrderedManifestError::InvalidProfile("prefix message dimension"),
        )?;
        let codeword_len = 1u64.checked_shl(self.prefix.log_codeword_len).ok_or(
            OrderedManifestError::InvalidProfile("prefix codeword dimension"),
        )?;
        let active_height = 1u64 << self.prefix.active_count_log_height;
        if self.prefix.trace_prefix_len > message_len
            || codeword_len % u64::from(self.prefix.rows_per_leaf) != 0
            || u64::from(self.max_active_children_per_source) > active_height
            || self.prefix.active_count_block_start % active_height != 0
            || self
                .prefix
                .active_count_block_start
                .checked_add(active_height)
                .is_none_or(|end| end > self.prefix.trace_prefix_len)
        {
            return Err(OrderedManifestError::InvalidProfile(
                "active-count prefix layout",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OrderedManifestVmState {
    pub pc: F,
    pub memory_root: Digest,
}

/// One source receipt derived from the genuine complete fixed-PESAT verifier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderedManifestSourceRecord {
    pub source_index: u32,
    pub call_index: u32,
    pub fresh_index_in_call: u32,
    pub normalized_leaf_start: u32,
    pub active_child_count: u8,
    pub program_commitment: Digest,
    pub initial_state: OrderedManifestVmState,
    pub final_state: OrderedManifestVmState,
    pub exit_code: F,
    pub is_terminate: F,
    pub source_instance_digest: Digest,
}

/// One prefix-link receipt derived from a concrete recursive prefix PCS
/// verifier. Its `full_root` is Construction 10.4's stacked fresh root.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderedManifestCallRecord {
    pub call_index: u32,
    pub input_arity: u32,
    pub source_start: u32,
    pub source_count: u32,
    pub base_root: Digest,
    pub full_root: Digest,
    pub logup_alpha: EF,
    pub logup_beta: EF,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderedManifestRecord {
    pub calls: Vec<OrderedManifestCallRecord>,
    pub sources: Vec<OrderedManifestSourceRecord>,
    pub final_accumulator_digest: Digest,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OrderedManifestDerivedRecord {
    pub record: OrderedManifestRecord,
    pub schedule_digest: Digest,
    pub source_order_digests: Vec<Digest>,
    pub manifest_digest: Digest,
}

#[derive(Clone, Debug)]
pub struct OrderedManifestTranscriptBundle {
    /// Proof 0 is the schedule, proofs `1..=call_count` are source-order
    /// digests, and proof `1 + call_count` is the complete manifest.
    pub logs: Vec<TranscriptLog<F, [F; 16]>>,
    pub schedule_digest: Digest,
    pub source_order_digests: Vec<Digest>,
    pub manifest_digest: Digest,
}

#[derive(Clone, Debug)]
pub struct OrderedManifestSourceReceiptMessage<T> {
    pub protocol_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub warp_index_digest: [T; DIGEST_SIZE],
    pub source_index: T,
    pub call_index: T,
    pub fresh_index_in_call: T,
    pub normalized_leaf_start: T,
    pub active_child_count: T,
    pub program_commitment: [T; DIGEST_SIZE],
    pub initial_pc: T,
    pub initial_root: [T; DIGEST_SIZE],
    pub final_pc: T,
    pub final_root: [T; DIGEST_SIZE],
    pub exit_code: T,
    pub is_terminate: T,
    pub source_instance_digest: [T; DIGEST_SIZE],
}

impl<T: Clone> OrderedManifestSourceReceiptMessage<T> {
    fn to_vec(&self) -> Vec<T> {
        let mut values = Vec::with_capacity(64);
        values.extend_from_slice(&self.protocol_digest);
        values.extend_from_slice(&self.relation_digest);
        values.extend_from_slice(&self.warp_index_digest);
        values.extend([
            self.source_index.clone(),
            self.call_index.clone(),
            self.fresh_index_in_call.clone(),
            self.normalized_leaf_start.clone(),
            self.active_child_count.clone(),
        ]);
        values.extend_from_slice(&self.program_commitment);
        values.push(self.initial_pc.clone());
        values.extend_from_slice(&self.initial_root);
        values.push(self.final_pc.clone());
        values.extend_from_slice(&self.final_root);
        values.extend([self.exit_code.clone(), self.is_terminate.clone()]);
        values.extend_from_slice(&self.source_instance_digest);
        values
    }
}

#[derive(Clone, Debug)]
pub struct OrderedManifestPrefixReceiptMessage<T> {
    pub protocol_version: T,
    pub relation_digest: [T; DIGEST_SIZE],
    pub index_digest: [T; DIGEST_SIZE],
    pub schedule_digest: [T; DIGEST_SIZE],
    pub source_order_digest: [T; DIGEST_SIZE],
    pub call_index: T,
    pub source_start: T,
    pub source_count: T,
    pub expected_active_child_counts: [T; ORDERED_MANIFEST_PREFIX_SLOTS],
    pub l_skip: T,
    pub log_message_len: T,
    pub log_blowup: T,
    pub log_codeword_len: T,
    pub rows_per_leaf: T,
    pub trace_prefix_len_lo: T,
    pub trace_prefix_len_hi: T,
    pub active_count_block_start_lo: T,
    pub active_count_block_start_hi: T,
    pub active_count_log_height: T,
    pub base_root: [T; DIGEST_SIZE],
    pub full_root: [T; DIGEST_SIZE],
    pub logup_alpha: [T; D_EF],
    pub logup_beta: [T; D_EF],
}

impl<T: Clone> OrderedManifestPrefixReceiptMessage<T> {
    fn to_vec(&self) -> Vec<T> {
        let mut values = Vec::with_capacity(132);
        values.push(self.protocol_version.clone());
        values.extend_from_slice(&self.relation_digest);
        values.extend_from_slice(&self.index_digest);
        values.extend_from_slice(&self.schedule_digest);
        values.extend_from_slice(&self.source_order_digest);
        values.extend([
            self.call_index.clone(),
            self.source_start.clone(),
            self.source_count.clone(),
        ]);
        values.extend_from_slice(&self.expected_active_child_counts);
        values.extend([
            self.l_skip.clone(),
            self.log_message_len.clone(),
            self.log_blowup.clone(),
            self.log_codeword_len.clone(),
            self.rows_per_leaf.clone(),
            self.trace_prefix_len_lo.clone(),
            self.trace_prefix_len_hi.clone(),
            self.active_count_block_start_lo.clone(),
            self.active_count_block_start_hi.clone(),
            self.active_count_log_height.clone(),
        ]);
        values.extend_from_slice(&self.base_root);
        values.extend_from_slice(&self.full_root);
        values.extend_from_slice(&self.logup_alpha);
        values.extend_from_slice(&self.logup_beta);
        values
    }
}

#[derive(Clone, Debug)]
pub struct OrderedManifestOccupancyMessage<T> {
    pub source_index: T,
    pub call_index: T,
    pub fresh_index_in_call: T,
    pub active_child_count: T,
}

impl<T: Clone> OrderedManifestOccupancyMessage<T> {
    fn to_vec(&self) -> Vec<T> {
        vec![
            self.source_index.clone(),
            self.call_index.clone(),
            self.fresh_index_in_call.clone(),
            self.active_child_count.clone(),
        ]
    }
}

#[derive(Clone, Debug)]
pub struct OrderedManifestCallSummaryMessage<T> {
    pub schedule_digest: [T; DIGEST_SIZE],
    pub call_count: T,
    pub total_source_count: T,
    pub call_index: T,
    pub source_start: T,
    pub source_count: T,
    pub input_arity: T,
    pub full_root: [T; DIGEST_SIZE],
}

impl<T: Clone> OrderedManifestCallSummaryMessage<T> {
    fn to_vec(&self) -> Vec<T> {
        let mut values = Vec::with_capacity(22);
        values.extend_from_slice(&self.schedule_digest);
        values.extend([
            self.call_count.clone(),
            self.total_source_count.clone(),
            self.call_index.clone(),
            self.source_start.clone(),
            self.source_count.clone(),
            self.input_arity.clone(),
        ]);
        values.extend_from_slice(&self.full_root);
        values
    }
}

#[derive(Clone, Debug)]
pub struct OrderedManifestSourceSummaryMessage<T> {
    pub source_count: T,
    pub call_count: T,
    pub normalized_leaf_count: T,
    pub program_commitment: [T; DIGEST_SIZE],
    pub initial_pc: T,
    pub initial_root: [T; DIGEST_SIZE],
    pub final_pc: T,
    pub final_root: [T; DIGEST_SIZE],
}

impl<T: Clone> OrderedManifestSourceSummaryMessage<T> {
    fn to_vec(&self) -> Vec<T> {
        let mut values = vec![
            self.source_count.clone(),
            self.call_count.clone(),
            self.normalized_leaf_count.clone(),
        ];
        values.extend_from_slice(&self.program_commitment);
        values.push(self.initial_pc.clone());
        values.extend_from_slice(&self.initial_root);
        values.push(self.final_pc.clone());
        values.extend_from_slice(&self.final_root);
        values
    }
}

macro_rules! ordered_lookup_bus {
    ($name:ident, $message:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name(LookupBus);

        impl $name {
            #[must_use]
            pub const fn new(index: BusIndex) -> Self {
                Self(LookupBus::new(index))
            }

            #[must_use]
            pub const fn index(self) -> BusIndex {
                self.0.index
            }

            pub fn lookup_key<AB, T>(
                &self,
                builder: &mut AB,
                message: $message<T>,
                enabled: impl Into<AB::Expr>,
            ) where
                AB: InteractionBuilder,
                T: Into<AB::Expr> + Clone,
            {
                self.0.lookup_key(builder, message.to_vec(), enabled);
            }

            pub fn add_key_with_lookups<AB, T>(
                &self,
                builder: &mut AB,
                message: $message<T>,
                count: impl Into<AB::Expr>,
            ) where
                AB: InteractionBuilder,
                T: Into<AB::Expr> + Clone,
            {
                self.0
                    .add_key_with_lookups(builder, message.to_vec(), count);
            }
        }
    };
}

ordered_lookup_bus!(
    OrderedManifestSourceAuthorityBus,
    OrderedManifestSourceReceiptMessage
);
ordered_lookup_bus!(
    OrderedManifestPrefixAuthorityBus,
    OrderedManifestPrefixReceiptMessage
);
ordered_lookup_bus!(
    OrderedManifestPrefixBindingBus,
    OrderedManifestPrefixReceiptMessage
);
ordered_lookup_bus!(OrderedManifestOccupancyBus, OrderedManifestOccupancyMessage);
ordered_lookup_bus!(
    OrderedManifestCallSummaryBus,
    OrderedManifestCallSummaryMessage
);
ordered_lookup_bus!(
    OrderedManifestSourceSummaryBus,
    OrderedManifestSourceSummaryMessage
);

#[derive(Clone, Copy, Debug)]
pub struct OrderedManifestBuses {
    pub transcript: TranscriptBus,
    pub source_authority: OrderedManifestSourceAuthorityBus,
    pub prefix_authority: OrderedManifestPrefixAuthorityBus,
    /// Republished only after manifest/prefix linkage; the WARP-Verify receipt
    /// producer must consume this exact bus.
    pub prefix_binding: OrderedManifestPrefixBindingBus,
    pub occupancy: OrderedManifestOccupancyBus,
    pub call_summary: OrderedManifestCallSummaryBus,
    pub source_summary: OrderedManifestSourceSummaryBus,
}

impl OrderedManifestBuses {
    #[must_use]
    pub fn new(indices: &mut BusIndexManager) -> Self {
        Self {
            transcript: TranscriptBus::new(indices.new_bus_idx()),
            source_authority: OrderedManifestSourceAuthorityBus::new(indices.new_bus_idx()),
            prefix_authority: OrderedManifestPrefixAuthorityBus::new(indices.new_bus_idx()),
            prefix_binding: OrderedManifestPrefixBindingBus::new(indices.new_bus_idx()),
            occupancy: OrderedManifestOccupancyBus::new(indices.new_bus_idx()),
            call_summary: OrderedManifestCallSummaryBus::new(indices.new_bus_idx()),
            source_summary: OrderedManifestSourceSummaryBus::new(indices.new_bus_idx()),
        }
    }
}

/// Stable AIR/context identity used by the production wrapper owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderedManifestAirKind {
    Header,
    Calls,
    Sources,
    Transcript,
    Poseidon2,
}

/// Complete manifest receipt producer, including its own transcript namespace.
/// The Poseidon AIR can either be returned privately or merged with the other
/// wrapper verifier components through `poseidon2_bus_owner`.
pub struct OrderedManifestReceiptProducer {
    pub profile: OrderedManifestProfile,
    pub wrapper_buses: FiniteWarpV3ReceiptBuses,
    pub buses: OrderedManifestBuses,
    pub header_air: OrderedManifestHeaderAir,
    pub call_air: OrderedManifestCallAir,
    pub source_air: OrderedManifestSourceAir,
    transcript: NativeWarpTranscriptModule,
}

impl OrderedManifestReceiptProducer {
    pub fn new(
        profile: OrderedManifestProfile,
        wrapper_buses: FiniteWarpV3ReceiptBuses,
        buses: OrderedManifestBuses,
        shared: &BusInventory,
        params: SystemParams,
    ) -> Result<Self, OrderedManifestError> {
        profile.validate()?;
        let transcript = NativeWarpTranscriptModule::new_for_bus(shared, buses.transcript, params);
        Ok(Self {
            header_air: OrderedManifestHeaderAir::new(
                profile.clone(),
                wrapper_buses,
                buses,
                buses.transcript,
            ),
            call_air: OrderedManifestCallAir::new(profile.clone(), buses, buses.transcript),
            source_air: OrderedManifestSourceAir::new(profile.clone(), buses, buses.transcript),
            profile,
            wrapper_buses,
            buses,
            transcript,
        })
    }

    #[must_use]
    pub fn poseidon2_bus_owner(&self) -> openvm_recursion_circuit::transcript::Poseidon2BusOwner {
        self.transcript.poseidon2_bus_owner()
    }

    #[must_use]
    pub const fn air_order(&self) -> [OrderedManifestAirKind; 5] {
        [
            OrderedManifestAirKind::Header,
            OrderedManifestAirKind::Calls,
            OrderedManifestAirKind::Sources,
            OrderedManifestAirKind::Transcript,
            OrderedManifestAirKind::Poseidon2,
        ]
    }

    /// AIR order used by the CPU contexts: header, calls, sources, transcript,
    /// Poseidon. Parent assemblies sharing Poseidon should drop the last AIR.
    #[must_use]
    pub fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        let mut airs: Vec<AirRef<SC>> = vec![
            std::sync::Arc::new(self.header_air.clone()),
            std::sync::Arc::new(self.call_air.clone()),
            std::sync::Arc::new(self.source_air.clone()),
        ];
        airs.extend(self.transcript.airs::<SC>());
        debug_assert_eq!(airs.len(), self.air_order().len());
        airs
    }

    /// AIRs aligned with
    /// [`OrderedManifestTraces::into_shared_poseidon_traces`] when the wrapper
    /// owns one shared Poseidon table.
    #[must_use]
    pub fn airs_without_poseidon<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        let mut airs = self.airs::<SC>();
        let removed = airs.pop();
        debug_assert!(removed.is_some());
        airs
    }

    pub fn generate_traces(
        &self,
        record: &OrderedManifestRecord,
    ) -> Result<OrderedManifestTraces, OrderedManifestError> {
        let derived = derive_ordered_manifest(&self.profile, record)?;
        let transcript_bundle = ordered_manifest_transcript_bundle(&self.profile, &derived)?;
        let logs = transcript_bundle.logs.iter().collect::<Vec<_>>();
        let transcript = self
            .transcript
            .generate_trace(&logs, None)
            .ok_or(OrderedManifestError::TranscriptTrace)?;
        let (header, calls, sources) =
            generate_ordered_manifest_main_traces(&self.profile, &derived)?;
        Ok(OrderedManifestTraces {
            derived,
            header,
            calls,
            sources,
            transcript,
        })
    }
}

pub struct OrderedManifestTraces {
    pub derived: OrderedManifestDerivedRecord,
    pub header: openvm_stark_backend::p3_matrix::dense::RowMajorMatrix<F>,
    pub calls: openvm_stark_backend::p3_matrix::dense::RowMajorMatrix<F>,
    pub sources: openvm_stark_backend::p3_matrix::dense::RowMajorMatrix<F>,
    pub transcript: NativeWarpTranscriptArtifacts,
}

/// AIR-aligned manifest traces, retaining the native transcript-derived
/// statement used by the wrapper boundary.
pub struct OrderedManifestOrderedTraces {
    pub derived: OrderedManifestDerivedRecord,
    pub ordered_traces: Vec<openvm_stark_backend::p3_matrix::dense::RowMajorMatrix<F>>,
}

/// Manifest contexts and hash inputs for one shared production Poseidon AIR.
pub struct OrderedManifestSharedPoseidonTraces {
    pub derived: OrderedManifestDerivedRecord,
    pub ordered_traces_without_poseidon:
        Vec<openvm_stark_backend::p3_matrix::dense::RowMajorMatrix<F>>,
    pub permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
}

impl OrderedManifestTraces {
    /// Consume traces in [`OrderedManifestReceiptProducer::air_order`].
    #[must_use]
    pub fn into_ordered_traces(self) -> OrderedManifestOrderedTraces {
        let Self {
            derived,
            header,
            calls,
            sources,
            transcript,
        } = self;
        OrderedManifestOrderedTraces {
            derived,
            ordered_traces: vec![
                header,
                calls,
                sources,
                transcript.trace,
                transcript.poseidon2_trace,
            ],
        }
    }

    /// Consume traces while preserving the inputs required by a shared
    /// Poseidon owner. The trace order is `air_order()` without `Poseidon2`.
    #[must_use]
    pub fn into_shared_poseidon_traces(self) -> OrderedManifestSharedPoseidonTraces {
        let Self {
            derived,
            header,
            calls,
            sources,
            transcript,
        } = self;
        let NativeWarpTranscriptArtifacts {
            trace,
            poseidon2_trace: _,
            permutation_inputs,
            compression_inputs,
        } = transcript;
        OrderedManifestSharedPoseidonTraces {
            derived,
            ordered_traces_without_poseidon: vec![header, calls, sources, trace],
            permutation_inputs,
            compression_inputs,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum OrderedManifestError {
    InvalidProfile(&'static str),
    EmptyCalls,
    EmptySources,
    TooManyCalls(usize),
    TooManySources(usize),
    CallShape(usize),
    SourceShape(usize),
    SourceOrder(usize),
    Occupancy(usize),
    StateContinuity(usize),
    Termination(usize),
    NormalizedLeafCount { expected: u32, actual: u32 },
    ZeroDigest(&'static str),
    DuplicateRoot { first: usize, second: usize },
    IntegerOverflow,
    TranscriptTrace,
    TranscriptMismatch,
}

impl core::fmt::Display for OrderedManifestError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "invalid ordered WARP-v3 manifest: {self:?}")
    }
}

impl std::error::Error for OrderedManifestError {}

pub fn derive_ordered_manifest(
    profile: &OrderedManifestProfile,
    record: &OrderedManifestRecord,
) -> Result<OrderedManifestDerivedRecord, OrderedManifestError> {
    profile.validate()?;
    validate_record(profile, record)?;
    let schedule_digest = schedule_transcript(record)?.0;
    let mut source_order_digests = Vec::with_capacity(record.calls.len());
    for call in &record.calls {
        let start = call.source_start as usize;
        let end = start
            .checked_add(call.source_count as usize)
            .ok_or(OrderedManifestError::IntegerOverflow)?;
        source_order_digests.push(
            source_order_transcript(
                call.call_index,
                record
                    .sources
                    .get(start..end)
                    .ok_or(OrderedManifestError::CallShape(call.call_index as usize))?,
            )?
            .0,
        );
    }
    let manifest_digest = manifest_transcript(record, schedule_digest)?.0;
    Ok(OrderedManifestDerivedRecord {
        record: record.clone(),
        schedule_digest,
        source_order_digests,
        manifest_digest,
    })
}

pub fn ordered_manifest_transcript_bundle(
    profile: &OrderedManifestProfile,
    derived: &OrderedManifestDerivedRecord,
) -> Result<OrderedManifestTranscriptBundle, OrderedManifestError> {
    profile.validate()?;
    validate_record(profile, &derived.record)?;
    let (schedule_digest, schedule_log) = schedule_transcript(&derived.record)?;
    let mut logs = vec![schedule_log];
    let mut source_order_digests = Vec::with_capacity(derived.record.calls.len());
    for call in &derived.record.calls {
        let start = call.source_start as usize;
        let end = start
            .checked_add(call.source_count as usize)
            .ok_or(OrderedManifestError::IntegerOverflow)?;
        let (digest, log) = source_order_transcript(
            call.call_index,
            derived
                .record
                .sources
                .get(start..end)
                .ok_or(OrderedManifestError::CallShape(call.call_index as usize))?,
        )?;
        source_order_digests.push(digest);
        logs.push(log);
    }
    let (manifest_digest, manifest_log) = manifest_transcript(&derived.record, schedule_digest)?;
    logs.push(manifest_log);
    if schedule_digest != derived.schedule_digest
        || source_order_digests != derived.source_order_digests
        || manifest_digest != derived.manifest_digest
    {
        return Err(OrderedManifestError::TranscriptMismatch);
    }
    Ok(OrderedManifestTranscriptBundle {
        logs,
        schedule_digest,
        source_order_digests,
        manifest_digest,
    })
}

fn validate_record(
    profile: &OrderedManifestProfile,
    record: &OrderedManifestRecord,
) -> Result<(), OrderedManifestError> {
    if record.calls.is_empty() {
        return Err(OrderedManifestError::EmptyCalls);
    }
    if record.sources.is_empty() {
        return Err(OrderedManifestError::EmptySources);
    }
    if record.calls.len() > FINITE_WARP_V3_MAX_CALLS {
        return Err(OrderedManifestError::TooManyCalls(record.calls.len()));
    }
    if record.sources.len() > FINITE_WARP_V3_MAX_SOURCES as usize {
        return Err(OrderedManifestError::TooManySources(record.sources.len()));
    }
    if record
        .final_accumulator_digest
        .iter()
        .all(|value| *value == F::ZERO)
    {
        return Err(OrderedManifestError::ZeroDigest("final accumulator"));
    }
    let mut source_start = 0u32;
    for (index, call) in record.calls.iter().enumerate() {
        let prior = u32::from(index != 0);
        if call.call_index != index as u32
            || call.source_start != source_start
            || call.source_count == 0
            || call.input_arity != call.source_count + prior
            || call.input_arity < 2
            || call.input_arity > FINITE_WARP_V3_MAX_INPUT_ARITY
            || !call.input_arity.is_power_of_two()
            || call.base_root.iter().all(|value| *value == F::ZERO)
            || call.full_root.iter().all(|value| *value == F::ZERO)
        {
            return Err(OrderedManifestError::CallShape(index));
        }
        if let Some(first) = record.calls[..index]
            .iter()
            .position(|earlier| earlier.full_root == call.full_root)
        {
            return Err(OrderedManifestError::DuplicateRoot {
                first,
                second: index,
            });
        }
        source_start = source_start
            .checked_add(call.source_count)
            .ok_or(OrderedManifestError::IntegerOverflow)?;
    }
    if source_start as usize != record.sources.len() {
        return Err(OrderedManifestError::CallShape(record.calls.len() - 1));
    }

    let mut normalized_start = 0u32;
    for (index, source) in record.sources.iter().enumerate() {
        if source.source_index != index as u32
            || source.normalized_leaf_start != normalized_start
            || source.active_child_count == 0
            || source.active_child_count > profile.max_active_children_per_source
            || source
                .program_commitment
                .iter()
                .all(|value| *value == F::ZERO)
            || source
                .source_instance_digest
                .iter()
                .all(|value| *value == F::ZERO)
        {
            return Err(OrderedManifestError::SourceShape(index));
        }
        let call = record
            .calls
            .get(source.call_index as usize)
            .ok_or(OrderedManifestError::SourceOrder(index))?;
        if source.source_index < call.source_start
            || source.source_index >= call.source_start + call.source_count
            || source.fresh_index_in_call != source.source_index - call.source_start
        {
            return Err(OrderedManifestError::SourceOrder(index));
        }
        if let Some(previous) = index.checked_sub(1).and_then(|i| record.sources.get(i)) {
            if previous.program_commitment != source.program_commitment
                || previous.final_state != source.initial_state
            {
                return Err(OrderedManifestError::StateContinuity(index));
            }
        }
        let final_source = index + 1 == record.sources.len();
        if final_source {
            if source.exit_code != F::ZERO || source.is_terminate != F::ONE {
                return Err(OrderedManifestError::Termination(index));
            }
        } else if source.exit_code != F::from_u32(profile.suspend_exit_code)
            || source.is_terminate != F::ZERO
        {
            return Err(OrderedManifestError::Termination(index));
        }
        normalized_start = normalized_start
            .checked_add(u32::from(source.active_child_count))
            .ok_or(OrderedManifestError::IntegerOverflow)?;
    }
    if normalized_start != profile.expected_normalized_leaf_count {
        return Err(OrderedManifestError::NormalizedLeafCount {
            expected: profile.expected_normalized_leaf_count,
            actual: normalized_start,
        });
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum Observation {
    Field(F),
    Digest(Digest),
}

fn schedule_transcript(
    record: &OrderedManifestRecord,
) -> Result<(Digest, TranscriptLog<F, [F; 16]>), OrderedManifestError> {
    let mut observations = Vec::with_capacity(8 + 9 * record.calls.len());
    observations.push(Observation::Field(F::from_u32(SCHEDULE_START_TAG)));
    push_u32(&mut observations, ORDERED_MANIFEST_PROTOCOL_VERSION);
    push_u32(&mut observations, record.sources.len() as u32);
    push_u32(&mut observations, record.calls.len() as u32);
    for (position, call) in record.calls.iter().enumerate() {
        observations.push(Observation::Field(F::from_u32(SCHEDULE_CALL_TAG)));
        for value in [
            position as u32,
            call.input_arity,
            call.source_count,
            u32::from(position != 0),
        ] {
            push_u32(&mut observations, value);
        }
    }
    observations.push(Observation::Field(F::from_u32(SCHEDULE_END_TAG)));
    digest_observations(SCHEDULE_DIGEST_TAG, &observations)
}

fn source_order_transcript(
    call_index: u32,
    sources: &[OrderedManifestSourceRecord],
) -> Result<(Digest, TranscriptLog<F, [F; 16]>), OrderedManifestError> {
    if sources.is_empty() {
        return Err(OrderedManifestError::EmptySources);
    }
    let mut observations = Vec::with_capacity(7 + 16 * sources.len());
    observations.push(Observation::Field(F::from_u32(SOURCE_ORDER_DIGEST_TAG)));
    push_u32(&mut observations, ORDERED_MANIFEST_PROTOCOL_VERSION);
    push_u32(&mut observations, call_index);
    push_u32(&mut observations, sources.len() as u32);
    for source in sources {
        for value in [
            source.source_index,
            source.call_index,
            source.fresh_index_in_call,
            source.normalized_leaf_start,
        ] {
            push_u32(&mut observations, value);
        }
        observations.push(Observation::Field(F::from_u8(source.active_child_count)));
        observations.extend([
            Observation::Digest(source.program_commitment),
            Observation::Field(source.initial_state.pc),
            Observation::Digest(source.initial_state.memory_root),
            Observation::Field(source.final_state.pc),
            Observation::Digest(source.final_state.memory_root),
            Observation::Field(source.exit_code),
            Observation::Field(source.is_terminate),
        ]);
    }
    digest_observations(SOURCE_ORDER_DIGEST_TAG, &observations)
}

fn manifest_transcript(
    record: &OrderedManifestRecord,
    schedule_digest: Digest,
) -> Result<(Digest, TranscriptLog<F, [F; 16]>), OrderedManifestError> {
    let mut observations =
        Vec::with_capacity(11 + 8 * record.calls.len() + 18 * record.sources.len());
    observations.push(Observation::Field(F::from_u32(MANIFEST_START_TAG)));
    push_u32(&mut observations, ORDERED_MANIFEST_PROTOCOL_VERSION);
    observations.push(Observation::Digest(schedule_digest));
    push_u32(&mut observations, record.calls.len() as u32);
    for call in &record.calls {
        observations.push(Observation::Field(F::from_u32(MANIFEST_INVOCATION_TAG)));
        for value in [call.call_index, call.input_arity, call.source_count] {
            push_u32(&mut observations, value);
        }
        observations.push(Observation::Digest(call.full_root));
    }
    push_u32(&mut observations, record.sources.len() as u32);
    for source in &record.sources {
        observations.push(Observation::Field(F::from_u32(MANIFEST_SOURCE_TAG)));
        for value in [
            source.source_index,
            source.call_index,
            source.fresh_index_in_call,
            source.normalized_leaf_start,
        ] {
            push_u32(&mut observations, value);
        }
        observations.push(Observation::Field(F::from_u8(source.active_child_count)));
        observations.extend([
            Observation::Digest(source.program_commitment),
            Observation::Field(source.initial_state.pc),
            Observation::Digest(source.initial_state.memory_root),
            Observation::Field(source.final_state.pc),
            Observation::Digest(source.final_state.memory_root),
            Observation::Field(source.exit_code),
            Observation::Field(source.is_terminate),
            Observation::Digest(source.source_instance_digest),
        ]);
    }
    observations.push(Observation::Field(F::from_u32(MANIFEST_FINAL_TAG)));
    observations.push(Observation::Digest(record.final_accumulator_digest));
    observations.push(Observation::Field(F::from_u32(MANIFEST_END_TAG)));
    digest_observations(MANIFEST_DIGEST_TAG, &observations)
}

fn digest_observations(
    domain: u32,
    observations: &[Observation],
) -> Result<(Digest, TranscriptLog<F, [F; 16]>), OrderedManifestError> {
    let mut transcript = default_duplex_sponge_recorder();
    <_ as FiatShamirTranscript<crate::SC>>::observe(&mut transcript, F::from_u32(domain));
    <_ as FiatShamirTranscript<crate::SC>>::observe(
        &mut transcript,
        F::from_u32(ORDERED_MANIFEST_PROTOCOL_VERSION),
    );
    let observation_count =
        u32::try_from(observations.len()).map_err(|_| OrderedManifestError::IntegerOverflow)?;
    observe_u32(&mut transcript, observation_count);
    for observation in observations {
        match observation {
            Observation::Field(value) => {
                <_ as FiatShamirTranscript<crate::SC>>::observe(
                    &mut transcript,
                    F::from_u32(OBSERVATION_FIELD_TAG),
                );
                <_ as FiatShamirTranscript<crate::SC>>::observe(&mut transcript, *value);
            }
            Observation::Digest(digest) => {
                <_ as FiatShamirTranscript<crate::SC>>::observe(
                    &mut transcript,
                    F::from_u32(OBSERVATION_DIGEST_TAG),
                );
                for value in digest {
                    <_ as FiatShamirTranscript<crate::SC>>::observe(&mut transcript, *value);
                }
            }
        }
    }
    <_ as FiatShamirTranscript<crate::SC>>::observe(
        &mut transcript,
        F::from_u32(OBSERVATION_END_TAG),
    );
    let digest =
        core::array::from_fn(|_| <_ as FiatShamirTranscript<crate::SC>>::sample(&mut transcript));
    Ok((digest, TranscriptHistory::into_log(transcript)))
}

fn push_u32(observations: &mut Vec<Observation>, value: u32) {
    observations.push(Observation::Field(F::from_u32(value & 0xffff)));
    observations.push(Observation::Field(F::from_u32(value >> 16)));
}

fn observe_u32<T: FiatShamirTranscript<crate::SC>>(transcript: &mut T, value: u32) {
    <T as FiatShamirTranscript<crate::SC>>::observe(transcript, F::from_u32(value & 0xffff));
    <T as FiatShamirTranscript<crate::SC>>::observe(transcript, F::from_u32(value >> 16));
}

const _: () = assert!(FINITE_WARP_V3_MAX_CALLS == 3);
const _: () = assert!(FINITE_WARP_V3_MAX_INPUT_ARITY == 64);
const _: () = assert!(FINITE_WARP_V3_MAX_SOURCES == 128);
const _: () = assert!(ORDERED_MANIFEST_PREFIX_SLOTS == FINITE_WARP_V3_MAX_INPUT_ARITY as usize);
const _: () = assert!(D_EF == 4);

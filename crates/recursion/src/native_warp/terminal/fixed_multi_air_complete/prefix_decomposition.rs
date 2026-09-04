//! Statement, beta-power, and decomposition owner for the complete fixed
//! multi-AIR terminal relation.
//!
//! This module deliberately does not reuse the legacy
//! `fixed_multi_air::{prefix,decomposition}` pair.  The complete relation has
//! two regional claim families and a relation-native beta-power contribution.
//! Its exact transcript prefix is:
//!
//! 1. `COMPLETE_TERMINAL_STATEMENT_TAG`, terminal version, canonical metadata, `beta`, and `eta`;
//! 2. `COMPLETE_TERMINAL_DECOMPOSITION_TAG`, every local claim, then every interaction claim in
//!    setup order.
//!
//! `alpha`, `mu`, the commitment root, and the relation digest are published
//! on typed buses but are not spuriously absorbed inside the complete-terminal
//! statement domain.  The surrounding terminal package owns their earlier
//! transcript binding.
//!
//! Integration (owned by `fixed_multi_air_complete::mod`/circuit assembly)
//! needs to re-export this file, allocate all buses in
//! [`FixedMultiAirCompletePrefixBuses`], install the four AIRs returned by
//! [`FixedMultiAirCompletePrefixDecompositionAirs::new`], and place each
//! generated cached/common pair in the same order.  The transcript AIR must
//! use the same proof index and transcript log.  A downstream cursor-chain
//! owner must consume the claim catalogs plus the one sequence-start message,
//! then emit distinct local/interaction starts only after advancing through
//! every preceding region's rounds and opening tail.  The final-instance/root
//! bridge must consume the binding and every externally requested instance
//! lookup.

use core::{
    borrow::{Borrow, BorrowMut},
    fmt,
};
use std::sync::Arc;

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder,
    interaction::InteractionBuilder,
    native_warp::{
        FixedMultiAirCompleteConstraintLayout, FixedMultiAirCompletePesatIndex,
        FixedMultiAirCompleteTerminalMetadata, FixedMultiAirCompleteTerminalProof,
        FixedMultiAirCompleteTerminalRecordLayout, FixedMultiAirCompleteTranscriptBinding,
        FixedMultiAirCompleteWitnessLayout, COMPLETE_TERMINAL_DECOMPOSITION_TAG,
        COMPLETE_TERMINAL_STATEMENT_TAG, FIXED_MULTI_AIR_COMPLETE_PESAT_VERSION,
        FIXED_MULTI_AIR_COMPLETE_TERMINAL_ROUND_DEGREE, FIXED_MULTI_AIR_COMPLETE_TERMINAL_VERSION,
        FIXED_MULTI_AIR_COMPLETE_WARP_DEGREE,
    },
    transcript::TranscriptLog,
    warp_pesat::AccumulatorInstance,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing, PrimeField32,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    define_typed_lookup_bus, define_typed_permutation_bus,
    utils::{ext_field_add, ext_field_multiply, ext_field_one_minus, ext_field_subtract},
};

const INSTANCE_SECTION_ALPHA: usize = 0;
const INSTANCE_SECTION_MU: usize = 1;
const INSTANCE_SECTION_BETA: usize = 2;
const INSTANCE_SECTION_ETA: usize = 3;
const INSTANCE_SECTION_COUNT: usize = 4;

const PREFIX_SOURCE_CONSTANT: usize = 0;
const PREFIX_SOURCE_BETA: usize = 1;
const PREFIX_SOURCE_ETA: usize = 2;
const PREFIX_SOURCE_LOCAL_CLAIM: usize = 3;
const PREFIX_SOURCE_INTERACTION_CLAIM: usize = 4;
const PREFIX_SOURCE_COUNT: usize = 5;

const DECOMPOSITION_SOURCE_BETA: usize = 0;
const DECOMPOSITION_SOURCE_LOCAL: usize = 1;
const DECOMPOSITION_SOURCE_INTERACTION: usize = 2;
const DECOMPOSITION_SOURCE_COUNT: usize = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedMultiAirCompletePrefixError {
    Metadata(&'static str),
    Shape(&'static str),
    Overflow(&'static str),
    Transcript(&'static str),
    Root,
    Claim,
    InteractionPresence { region: usize },
}

impl fmt::Display for FixedMultiAirCompletePrefixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Metadata(message) => write!(formatter, "complete terminal metadata: {message}"),
            Self::Shape(message) => write!(formatter, "complete terminal shape: {message}"),
            Self::Overflow(message) => write!(formatter, "complete terminal overflow: {message}"),
            Self::Transcript(message) => {
                write!(formatter, "complete terminal transcript: {message}")
            }
            Self::Root => formatter.write_str("complete terminal accumulator root mismatch"),
            Self::Claim => formatter.write_str("complete terminal decomposition claim mismatch"),
            Self::InteractionPresence { region } => write!(
                formatter,
                "complete terminal interaction proof presence mismatch in region {region}"
            ),
        }
    }
}

impl std::error::Error for FixedMultiAirCompletePrefixError {}

type PrefixResult<T> = Result<T, FixedMultiAirCompletePrefixError>;

/// Setup-owned consumer counts.  These are part of the verifier component,
/// never proof data.  Counts name consumers outside this module; the profile
/// adds its own exact transcript/beta/decomposition fanout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedMultiAirCompleteExternalLookupCounts {
    pub binding: u32,
    pub alpha: Vec<u32>,
    pub mu: u32,
    pub beta: Vec<u32>,
    pub eta: u32,
    pub local_claims: Vec<u32>,
    pub interaction_claims: Vec<u32>,
}

impl FixedMultiAirCompleteExternalLookupCounts {
    /// Minimum sound outer binding: one consumer authenticates the root and
    /// relation, and one consumer authenticates every codeword point
    /// coordinate plus `mu`.  Beta/eta and regional claims already have
    /// consumers inside this module.
    #[must_use]
    pub fn semantic_minimum(alpha_len: usize, beta_len: usize, region_count: usize) -> Self {
        Self {
            binding: 1,
            alpha: vec![1; alpha_len],
            mu: 1,
            beta: vec![0; beta_len],
            eta: 0,
            // The cursor-chain owner consumes each catalog entry before it
            // emits an actual transcript-positioned regional start.
            local_claims: vec![1; region_count],
            interaction_claims: vec![1; region_count],
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteBindingMessage<T> {
    pub relation_digest: [T; DIGEST_SIZE],
    pub root: [T; DIGEST_SIZE],
    pub alpha_len: T,
    pub beta_len: T,
}

define_typed_lookup_bus!(
    FixedMultiAirCompleteBindingBus,
    FixedMultiAirCompleteBindingMessage
);

/// Algebraic handoff from the enclosing terminal transcript owner to the
/// complete nonlinear relation. The prefix witness still carries this cursor
/// so trace generation can replay the native transcript, but the value is no
/// longer trusted as an unconstrained common-trace cell.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteStatementStartCursorMessage<T> {
    pub start_tidx: T,
}

define_typed_permutation_bus!(
    FixedMultiAirCompleteStatementStartCursorBus,
    FixedMultiAirCompleteStatementStartCursorMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteInstanceValueMessage<T> {
    /// 0 = codeword point `alpha`, 1 = `mu`, 2 = PESAT point `beta`, 3 = `eta`.
    pub section: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirCompleteInstanceValueBus,
    FixedMultiAirCompleteInstanceValueMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteRegionalClaimMessage<T> {
    pub region: T,
    pub claim: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirCompleteLocalClaimBus,
    FixedMultiAirCompleteRegionalClaimMessage
);
define_typed_lookup_bus!(
    FixedMultiAirCompleteInteractionClaimBus,
    FixedMultiAirCompleteRegionalClaimMessage
);

/// Starts the downstream transcript cursor chain.  Later region owners must
/// chain their end cursor to the next setup ordinal; they must not all restart
/// at `decomposition_end_tidx`.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteRegionSequenceStartMessage<T> {
    pub decomposition_end_tidx: T,
    pub protocol_component_count: T,
}

define_typed_permutation_bus!(
    FixedMultiAirCompleteRegionSequenceStartBus,
    FixedMultiAirCompleteRegionSequenceStartMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteBetaClaimMessage<T> {
    pub claim: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirCompleteBetaClaimBus,
    FixedMultiAirCompleteBetaClaimMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteDecompositionReceiptMessage<T> {
    pub eta: [T; D_EF],
    pub beta_claim: [T; D_EF],
    pub local_count: T,
    pub interaction_count: T,
}

define_typed_permutation_bus!(
    FixedMultiAirCompleteDecompositionReceiptBus,
    FixedMultiAirCompleteDecompositionReceiptMessage
);

#[derive(Clone, Copy, Debug)]
pub struct FixedMultiAirCompletePrefixBuses {
    pub transcript: TranscriptBus,
    pub statement_start_cursor: FixedMultiAirCompleteStatementStartCursorBus,
    pub binding: FixedMultiAirCompleteBindingBus,
    pub instance: FixedMultiAirCompleteInstanceValueBus,
    pub local_claim: FixedMultiAirCompleteLocalClaimBus,
    pub interaction_claim: FixedMultiAirCompleteInteractionClaimBus,
    pub region_sequence_start: FixedMultiAirCompleteRegionSequenceStartBus,
    pub beta_claim: FixedMultiAirCompleteBetaClaimBus,
    pub decomposition_receipt: FixedMultiAirCompleteDecompositionReceiptBus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefixSource {
    Constant(EF),
    Beta(usize),
    Eta,
    LocalClaim(usize),
    InteractionClaim(usize),
}

#[derive(Clone, Debug)]
struct PrefixScheduleEntry {
    source: PrefixSource,
    interaction_present: bool,
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompletePrefixProfile {
    pub metadata: FixedMultiAirCompleteTerminalMetadata<Digest>,
    pub relation_digest: Digest,
    pub metadata_bytes: Vec<u8>,
    pub alpha_len: usize,
    pub beta_len: usize,
    pub region_count: usize,
    pub interaction_present: Vec<bool>,
    pub exact_relation_degree: usize,
    pub log_constraints: usize,
    pub beta_explicit_coordinate: usize,
    pub one_explicit_coordinate: usize,
    pub beta_power_explicit_start: usize,
    pub beta_power_count: usize,
    pub beta_constraint_start: usize,
    pub protocol_component_count: usize,
    pub binding_lookup_count: u32,
    pub instance_lookup_counts: Vec<Vec<u32>>,
    pub local_claim_lookup_counts: Vec<u32>,
    pub interaction_claim_lookup_counts: Vec<u32>,
    transcript_schedule: Vec<PrefixScheduleEntry>,
}

impl FixedMultiAirCompletePrefixProfile {
    pub fn from_relation(
        relation: &FixedMultiAirCompletePesatIndex<F, Digest>,
        external: FixedMultiAirCompleteExternalLookupCounts,
    ) -> PrefixResult<Self> {
        let metadata = relation
            .terminal_linearization_metadata()
            .map_err(|_| FixedMultiAirCompletePrefixError::Metadata("backend linearization"))?;
        Self::from_metadata(metadata, external)
    }

    /// Build only from verifier-owned metadata.  Callers must obtain this
    /// value from the admitted relation, never from proof bytes.
    pub(crate) fn from_metadata(
        metadata: FixedMultiAirCompleteTerminalMetadata<Digest>,
        external: FixedMultiAirCompleteExternalLookupCounts,
    ) -> PrefixResult<Self> {
        validate_complete_metadata(&metadata)?;
        let relation_digest = metadata.relation_digest;
        let metadata_bytes = metadata
            .canonical_bytes()
            .map_err(|_| FixedMultiAirCompletePrefixError::Metadata("canonical encoding"))?;
        let alpha_len = usize::from(metadata.code_class.log_codeword_len);
        let log_constraints = usize::from(metadata.log_constraints);
        let explicit_len = checked_usize(metadata.explicit_len, "explicit length")?;
        let beta_len = log_constraints
            .checked_add(explicit_len)
            .ok_or(FixedMultiAirCompletePrefixError::Overflow("beta length"))?;
        let region_count = metadata.regions.len();
        let beta_power_count = usize::try_from(metadata.beta_power_count)
            .map_err(|_| FixedMultiAirCompletePrefixError::Overflow("beta power count"))?;
        let one_explicit_coordinate = log_constraints;
        let beta_explicit_coordinate = log_constraints
            .checked_add(checked_usize(
                metadata.beta_explicit_offset,
                "beta explicit offset",
            )?)
            .ok_or(FixedMultiAirCompletePrefixError::Overflow(
                "beta explicit coordinate",
            ))?;
        let beta_power_explicit_start = log_constraints
            .checked_add(checked_usize(
                metadata.beta_power_explicit_offset,
                "beta-power explicit offset",
            )?)
            .ok_or(FixedMultiAirCompletePrefixError::Overflow(
                "beta-power explicit coordinate",
            ))?;
        let beta_constraint_start =
            checked_usize(metadata.beta_constraints.start, "beta constraint offset")?;
        let interaction_present = metadata
            .regions
            .iter()
            .map(|region| !region.interactions.is_empty())
            .collect::<Vec<_>>();
        let protocol_component_count = region_count
            .checked_add(
                interaction_present
                    .iter()
                    .filter(|&&present| present)
                    .count(),
            )
            .ok_or(FixedMultiAirCompletePrefixError::Overflow(
                "regional component count",
            ))?;
        for (value, context) in [
            (metadata_bytes.len(), "metadata transcript length"),
            (alpha_len, "alpha field length"),
            (beta_len, "beta field length"),
            (region_count, "region field count"),
            (protocol_component_count, "regional component field count"),
        ] {
            validate_field_usize(value, context)?;
        }

        validate_external_counts(&external, alpha_len, beta_len, region_count)?;

        let mut instance_lookup_counts = vec![Vec::new(); INSTANCE_SECTION_COUNT];
        instance_lookup_counts[INSTANCE_SECTION_ALPHA] = external.alpha;
        instance_lookup_counts[INSTANCE_SECTION_MU] = vec![external.mu];
        instance_lookup_counts[INSTANCE_SECTION_BETA] = external.beta;
        instance_lookup_counts[INSTANCE_SECTION_ETA] = vec![external.eta];

        // Exact statement replay consumes every beta coordinate once and eta
        // once.  The decomposition AIR consumes eta once more.
        for count in &mut instance_lookup_counts[INSTANCE_SECTION_BETA] {
            checked_add_count(count, 1, "statement beta lookup count")?;
        }
        checked_add_count(
            &mut instance_lookup_counts[INSTANCE_SECTION_ETA][0],
            2,
            "statement/decomposition eta lookup count",
        )?;

        // The beta AIR reads every tau coordinate for every explicit power.
        for count in &mut instance_lookup_counts[INSTANCE_SECTION_BETA][..log_constraints] {
            checked_add_count(
                count,
                u32::try_from(beta_power_count).map_err(|_| {
                    FixedMultiAirCompletePrefixError::Overflow("tau lookup multiplicity")
                })?,
                "tau lookup count",
            )?;
        }
        // It reads one, beta, and the current power on the first row of every
        // power block, plus the preceding power for powers 1..m-1.
        checked_add_count(
            &mut instance_lookup_counts[INSTANCE_SECTION_BETA][one_explicit_coordinate],
            u32::try_from(beta_power_count).map_err(|_| {
                FixedMultiAirCompletePrefixError::Overflow("one lookup multiplicity")
            })?,
            "one lookup count",
        )?;
        checked_add_count(
            &mut instance_lookup_counts[INSTANCE_SECTION_BETA][beta_explicit_coordinate],
            u32::try_from(beta_power_count).map_err(|_| {
                FixedMultiAirCompletePrefixError::Overflow("beta lookup multiplicity")
            })?,
            "beta lookup count",
        )?;
        for power in 0..beta_power_count {
            checked_add_count(
                &mut instance_lookup_counts[INSTANCE_SECTION_BETA]
                    [beta_power_explicit_start + power],
                1,
                "current beta-power lookup count",
            )?;
            if power != 0 {
                checked_add_count(
                    &mut instance_lookup_counts[INSTANCE_SECTION_BETA]
                        [beta_power_explicit_start + power - 1],
                    1,
                    "previous beta-power lookup count",
                )?;
            }
        }

        let mut local_claim_lookup_counts = external.local_claims;
        let mut interaction_claim_lookup_counts = external.interaction_claims;
        for count in &mut local_claim_lookup_counts {
            checked_add_count(count, 1, "decomposition local-claim lookup count")?;
        }
        for count in &mut interaction_claim_lookup_counts {
            checked_add_count(count, 1, "decomposition interaction-claim lookup count")?;
        }

        let transcript_schedule =
            build_transcript_schedule(&metadata_bytes, beta_len, &interaction_present)?;
        let exact_relation_degree = usize::from(metadata.exact_relation_degree);

        Ok(Self {
            metadata,
            relation_digest,
            metadata_bytes,
            alpha_len,
            beta_len,
            region_count,
            interaction_present,
            exact_relation_degree,
            log_constraints,
            beta_explicit_coordinate,
            one_explicit_coordinate,
            beta_power_explicit_start,
            beta_power_count,
            beta_constraint_start,
            protocol_component_count,
            binding_lookup_count: external.binding,
            instance_lookup_counts,
            local_claim_lookup_counts,
            interaction_claim_lookup_counts,
            transcript_schedule,
        })
    }

    #[must_use]
    pub fn transcript_observation_count(&self) -> usize {
        self.transcript_schedule.len()
    }

    #[must_use]
    pub fn instance_row_count(&self) -> usize {
        self.alpha_len + 1 + self.beta_len + 1
    }

    #[must_use]
    pub fn beta_row_count(&self) -> usize {
        self.beta_power_count * self.log_constraints
    }

    #[must_use]
    pub fn decomposition_row_count(&self) -> usize {
        1 + 2 * self.region_count
    }
}

fn validate_complete_metadata(
    metadata: &FixedMultiAirCompleteTerminalMetadata<Digest>,
) -> PrefixResult<()> {
    if metadata.terminal_version != FIXED_MULTI_AIR_COMPLETE_TERMINAL_VERSION
        || metadata.relation_version != FIXED_MULTI_AIR_COMPLETE_PESAT_VERSION
        || metadata.transcript != FixedMultiAirCompleteTranscriptBinding::V4
        || metadata.constraint_layout
            != FixedMultiAirCompleteConstraintLayout::LocalAirRegionConstraintRowThenBetaThenRegionInteractionRowThenGlobalV2
        || metadata.witness_layout
            != FixedMultiAirCompleteWitnessLayout::TracePrefixThenRegionInteractionCoordinateRowV2
        || metadata.record_layout
            != FixedMultiAirCompleteTerminalRecordLayout::RegionInteractionRowV1
    {
        return Err(FixedMultiAirCompletePrefixError::Metadata(
            "version or canonical layout",
        ));
    }
    if metadata.regions.is_empty()
        || metadata.log_constraints == 0
        || metadata.beta_power_count == 0
        || metadata.beta_constraints.len != u64::from(metadata.beta_power_count)
        || usize::from(metadata.terminal_round_degree)
            != FIXED_MULTI_AIR_COMPLETE_TERMINAL_ROUND_DEGREE
        || usize::from(metadata.exact_relation_degree) < 2
        || usize::from(metadata.exact_relation_degree) > FIXED_MULTI_AIR_COMPLETE_WARP_DEGREE
    {
        return Err(FixedMultiAirCompletePrefixError::Metadata(
            "complete relation dimensions or degree",
        ));
    }
    let explicit_len = checked_usize(metadata.explicit_len, "explicit length")?;
    let beta_offset = checked_usize(metadata.beta_explicit_offset, "beta explicit offset")?;
    let powers_start = checked_usize(
        metadata.beta_power_explicit_offset,
        "beta-power explicit offset",
    )?;
    let powers_end = powers_start
        .checked_add(usize::try_from(metadata.beta_power_count).map_err(|_| {
            FixedMultiAirCompletePrefixError::Overflow("beta-power explicit interval")
        })?)
        .ok_or(FixedMultiAirCompletePrefixError::Overflow(
            "beta-power explicit interval",
        ))?;
    if metadata.alpha_explicit_offset >= metadata.explicit_len
        || beta_offset >= explicit_len
        || powers_end > explicit_len
    {
        return Err(FixedMultiAirCompletePrefixError::Metadata(
            "explicit or constraint interval",
        ));
    }
    let padded_constraints = metadata
        .zero_constraint_tail
        .end()
        .max(metadata.global_constraint.end());
    let constraint_domain = 1u64
        .checked_shl(u32::from(metadata.log_constraints))
        .ok_or(FixedMultiAirCompletePrefixError::Overflow(
            "constraint domain",
        ))?;
    if padded_constraints != constraint_domain
        || metadata.beta_constraints.end() > constraint_domain
    {
        return Err(FixedMultiAirCompletePrefixError::Metadata(
            "constraint-domain coverage",
        ));
    }
    for (index, region) in metadata.regions.iter().enumerate() {
        if usize::try_from(region.setup_ordinal).ok() != Some(index) {
            return Err(FixedMultiAirCompletePrefixError::Metadata(
                "region setup order",
            ));
        }
        for (interaction, entry) in region.interactions.iter().enumerate() {
            if usize::try_from(entry.interaction_ordinal).ok() != Some(interaction) {
                return Err(FixedMultiAirCompletePrefixError::Metadata(
                    "interaction setup order",
                ));
            }
        }
    }
    Ok(())
}

fn validate_external_counts(
    external: &FixedMultiAirCompleteExternalLookupCounts,
    alpha_len: usize,
    beta_len: usize,
    region_count: usize,
) -> PrefixResult<()> {
    if external.binding == 0
        || external.alpha.len() != alpha_len
        || external.alpha.iter().any(|&count| count == 0)
        || external.mu == 0
        || external.beta.len() != beta_len
        || external.local_claims.len() != region_count
        || external.interaction_claims.len() != region_count
        || external.local_claims.iter().any(|&count| count == 0)
        || external.interaction_claims.iter().any(|&count| count == 0)
    {
        return Err(FixedMultiAirCompletePrefixError::Shape(
            "external lookup counts",
        ));
    }
    for count in core::iter::once(external.binding)
        .chain(external.alpha.iter().copied())
        .chain(core::iter::once(external.mu))
        .chain(external.beta.iter().copied())
        .chain(core::iter::once(external.eta))
        .chain(external.local_claims.iter().copied())
        .chain(external.interaction_claims.iter().copied())
    {
        validate_count(count, "external lookup count")?;
    }
    Ok(())
}

fn validate_count(count: u32, context: &'static str) -> PrefixResult<()> {
    if count >= F::ORDER_U32 {
        return Err(FixedMultiAirCompletePrefixError::Overflow(context));
    }
    Ok(())
}

fn validate_field_usize(value: usize, context: &'static str) -> PrefixResult<()> {
    if value >= F::ORDER_U32 as usize {
        return Err(FixedMultiAirCompletePrefixError::Overflow(context));
    }
    Ok(())
}

fn checked_add_count(count: &mut u32, amount: u32, context: &'static str) -> PrefixResult<()> {
    *count = count
        .checked_add(amount)
        .ok_or(FixedMultiAirCompletePrefixError::Overflow(context))?;
    validate_count(*count, context)
}

fn checked_usize(value: u64, context: &'static str) -> PrefixResult<usize> {
    usize::try_from(value).map_err(|_| FixedMultiAirCompletePrefixError::Overflow(context))
}

fn build_transcript_schedule(
    metadata_bytes: &[u8],
    beta_len: usize,
    interaction_present: &[bool],
) -> PrefixResult<Vec<PrefixScheduleEntry>> {
    let region_count = interaction_present.len();
    let capacity = 7usize
        .checked_add(metadata_bytes.len())
        .and_then(|value| value.checked_add(beta_len))
        .and_then(|value| value.checked_add(2 * region_count))
        .ok_or(FixedMultiAirCompletePrefixError::Overflow(
            "transcript schedule",
        ))?;
    let mut schedule = Vec::with_capacity(capacity);
    push_constant(&mut schedule, COMPLETE_TERMINAL_STATEMENT_TAG);
    push_constant(
        &mut schedule,
        u64::from(FIXED_MULTI_AIR_COMPLETE_TERMINAL_VERSION),
    );
    push_constant(
        &mut schedule,
        u64::try_from(metadata_bytes.len()).map_err(|_| {
            FixedMultiAirCompletePrefixError::Overflow("metadata transcript length")
        })?,
    );
    for &byte in metadata_bytes {
        push_constant(&mut schedule, u64::from(byte));
    }
    push_constant(
        &mut schedule,
        u64::try_from(beta_len)
            .map_err(|_| FixedMultiAirCompletePrefixError::Overflow("beta transcript length"))?,
    );
    for coordinate in 0..beta_len {
        schedule.push(PrefixScheduleEntry {
            source: PrefixSource::Beta(coordinate),
            interaction_present: false,
        });
    }
    schedule.push(PrefixScheduleEntry {
        source: PrefixSource::Eta,
        interaction_present: false,
    });

    push_constant(&mut schedule, COMPLETE_TERMINAL_DECOMPOSITION_TAG);
    push_constant(
        &mut schedule,
        u64::try_from(region_count).map_err(|_| {
            FixedMultiAirCompletePrefixError::Overflow("local claim transcript length")
        })?,
    );
    for region in 0..region_count {
        schedule.push(PrefixScheduleEntry {
            source: PrefixSource::LocalClaim(region),
            interaction_present: false,
        });
    }
    push_constant(
        &mut schedule,
        u64::try_from(region_count).map_err(|_| {
            FixedMultiAirCompletePrefixError::Overflow("interaction claim transcript length")
        })?,
    );
    for (region, &present) in interaction_present.iter().enumerate() {
        schedule.push(PrefixScheduleEntry {
            source: PrefixSource::InteractionClaim(region),
            interaction_present: present,
        });
    }
    Ok(schedule)
}

fn push_constant(schedule: &mut Vec<PrefixScheduleEntry>, value: u64) {
    schedule.push(PrefixScheduleEntry {
        source: PrefixSource::Constant(EF::from_u64(value)),
        interaction_present: false,
    });
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteInstanceScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub section_flags: [T; INSTANCE_SECTION_COUNT],
    pub coordinate: T,
    pub lookup_count: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteInstanceCols<T> {
    pub value: [T; D_EF],
    pub root: [T; DIGEST_SIZE],
}

/// Publishes the complete final accumulator instance exactly once.  The
/// catalog's multiplicities are verifier configuration and therefore live in
/// cached rows.
pub struct FixedMultiAirCompleteInstanceAir {
    pub profile: Arc<FixedMultiAirCompletePrefixProfile>,
    pub binding_bus: FixedMultiAirCompleteBindingBus,
    pub instance_bus: FixedMultiAirCompleteInstanceValueBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteInstanceAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteInstanceAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirCompleteInstanceScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteInstanceCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteInstanceAir {}
impl BaseAir<F> for FixedMultiAirCompleteInstanceAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteInstanceScheduleCols::<F>::width()
            + FixedMultiAirCompleteInstanceCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteInstanceAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let cached_row = builder.cached_mains()[0]
            .row_slice(0)
            .expect("complete terminal instance cached row")
            .to_vec();
        let next_cached_row = builder.cached_mains()[0]
            .row_slice(1)
            .expect("complete terminal instance next cached row")
            .to_vec();
        let main_row = builder
            .common_main()
            .row_slice(0)
            .expect("complete terminal instance row")
            .to_vec();
        let next_main_row = builder
            .common_main()
            .row_slice(1)
            .expect("complete terminal instance next row")
            .to_vec();
        let schedule: &FixedMultiAirCompleteInstanceScheduleCols<AB::Var> =
            cached_row.as_slice().borrow();
        let next_schedule: &FixedMultiAirCompleteInstanceScheduleCols<AB::Var> =
            next_cached_row.as_slice().borrow();
        let local: &FixedMultiAirCompleteInstanceCols<AB::Var> = main_row.as_slice().borrow();
        let next: &FixedMultiAirCompleteInstanceCols<AB::Var> = next_main_row.as_slice().borrow();

        for flag in [schedule.active, schedule.is_first, schedule.is_last]
            .into_iter()
            .chain(schedule.section_flags)
        {
            builder.assert_bool(flag);
        }
        let section_sum = schedule
            .section_flags
            .into_iter()
            .fold(AB::Expr::ZERO, |sum, flag| sum + flag);
        builder.assert_eq(section_sum, schedule.active);
        builder.when_first_row().assert_one(schedule.active);
        builder.when_first_row().assert_one(schedule.is_first);
        builder
            .when_transition()
            .assert_bool(schedule.active - next_schedule.active);
        builder
            .when_transition()
            .assert_eq(schedule.active - next_schedule.active, schedule.is_last);
        builder
            .when_last_row()
            .assert_eq(schedule.is_last, schedule.active);
        assert_array_eq(
            &mut builder.when_transition().when(next_schedule.active),
            next.root,
            local.root.map(Into::into),
        );

        let section = schedule.section_flags[INSTANCE_SECTION_MU]
            + schedule.section_flags[INSTANCE_SECTION_BETA] * AB::Expr::from_usize(2)
            + schedule.section_flags[INSTANCE_SECTION_ETA] * AB::Expr::from_usize(3);
        self.instance_bus.add_key_with_lookups(
            builder,
            FixedMultiAirCompleteInstanceValueMessage {
                section,
                coordinate: schedule.coordinate.into(),
                value: local.value.map(Into::into),
            },
            schedule.lookup_count,
        );
        self.binding_bus.add_key_with_lookups(
            builder,
            FixedMultiAirCompleteBindingMessage {
                relation_digest: self.profile.relation_digest.map(Into::into),
                root: local.root.map(Into::into),
                alpha_len: AB::Expr::from_usize(self.profile.alpha_len),
                beta_len: AB::Expr::from_usize(self.profile.beta_len),
            },
            schedule.is_first * AB::Expr::from_u32(self.profile.binding_lookup_count),
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompletePrefixScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub source_flags: [T; PREFIX_SOURCE_COUNT],
    pub source_index: T,
    pub interaction_present: T,
    pub expected_proof_present: T,
    pub local_claim_lookup_count: T,
    pub interaction_claim_lookup_count: T,
    pub constant: [T; D_EF],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompletePrefixCols<T> {
    pub tidx: T,
    pub decomposition_end_tidx: T,
    pub value: [T; D_EF],
    pub interaction_proof_present: T,
}

/// Exact complete-terminal statement and decomposition transcript replay.
pub struct FixedMultiAirCompleteTranscriptPrefixAir {
    pub profile: Arc<FixedMultiAirCompletePrefixProfile>,
    pub proof_idx: usize,
    pub transcript_bus: TranscriptBus,
    pub statement_start_cursor_bus: FixedMultiAirCompleteStatementStartCursorBus,
    pub instance_bus: FixedMultiAirCompleteInstanceValueBus,
    pub local_claim_bus: FixedMultiAirCompleteLocalClaimBus,
    pub interaction_claim_bus: FixedMultiAirCompleteInteractionClaimBus,
    pub region_sequence_start_bus: FixedMultiAirCompleteRegionSequenceStartBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteTranscriptPrefixAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteTranscriptPrefixAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirCompletePrefixScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirCompletePrefixCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteTranscriptPrefixAir {}
impl BaseAir<F> for FixedMultiAirCompleteTranscriptPrefixAir {
    fn width(&self) -> usize {
        FixedMultiAirCompletePrefixScheduleCols::<F>::width()
            + FixedMultiAirCompletePrefixCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteTranscriptPrefixAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let cached_row = builder.cached_mains()[0]
            .row_slice(0)
            .expect("complete terminal prefix cached row")
            .to_vec();
        let next_cached_row = builder.cached_mains()[0]
            .row_slice(1)
            .expect("complete terminal prefix next cached row")
            .to_vec();
        let main_row = builder
            .common_main()
            .row_slice(0)
            .expect("complete terminal prefix row")
            .to_vec();
        let next_main_row = builder
            .common_main()
            .row_slice(1)
            .expect("complete terminal prefix next row")
            .to_vec();
        let schedule: &FixedMultiAirCompletePrefixScheduleCols<AB::Var> =
            cached_row.as_slice().borrow();
        let next_schedule: &FixedMultiAirCompletePrefixScheduleCols<AB::Var> =
            next_cached_row.as_slice().borrow();
        let local: &FixedMultiAirCompletePrefixCols<AB::Var> = main_row.as_slice().borrow();
        let next: &FixedMultiAirCompletePrefixCols<AB::Var> = next_main_row.as_slice().borrow();

        for flag in [
            schedule.active,
            schedule.is_first,
            schedule.is_last,
            schedule.interaction_present,
            schedule.expected_proof_present,
            local.interaction_proof_present,
        ]
        .into_iter()
        .chain(schedule.source_flags)
        {
            builder.assert_bool(flag);
        }
        let source_sum = schedule
            .source_flags
            .into_iter()
            .fold(AB::Expr::ZERO, |sum, flag| sum + flag);
        builder.assert_eq(source_sum, schedule.active);
        builder.when_first_row().assert_one(schedule.active);
        builder.when_first_row().assert_one(schedule.is_first);
        builder
            .when_transition()
            .assert_bool(schedule.active - next_schedule.active);
        builder
            .when_transition()
            .assert_eq(schedule.active - next_schedule.active, schedule.is_last);
        builder
            .when_last_row()
            .assert_eq(schedule.is_last, schedule.active);
        let mut transition_builder = builder.when_transition();
        let mut transition = transition_builder.when(next_schedule.active);
        transition.assert_eq(
            next.tidx,
            AB::Expr::from(local.tidx) + AB::Expr::from_usize(D_EF),
        );
        transition.assert_eq(next.decomposition_end_tidx, local.decomposition_end_tidx);
        builder.when(schedule.is_last).assert_eq(
            local.decomposition_end_tidx,
            AB::Expr::from(local.tidx) + AB::Expr::from_usize(D_EF),
        );
        builder.assert_eq(
            local.interaction_proof_present,
            schedule.expected_proof_present,
        );

        self.statement_start_cursor_bus.receive(
            builder,
            FixedMultiAirCompleteStatementStartCursorMessage {
                start_tidx: local.tidx.into(),
            },
            schedule.is_first,
        );

        assert_array_eq(
            &mut builder.when(schedule.source_flags[PREFIX_SOURCE_CONSTANT]),
            local.value,
            schedule.constant.map(Into::into),
        );
        let zero = [
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        assert_array_eq(
            &mut builder.when(
                schedule.source_flags[PREFIX_SOURCE_INTERACTION_CLAIM]
                    * (AB::Expr::ONE - schedule.interaction_present),
            ),
            local.value,
            zero,
        );

        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::from_usize(self.proof_idx),
            local.tidx,
            local.value,
            schedule.active,
        );
        self.instance_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInstanceValueMessage {
                section: AB::Expr::from_usize(INSTANCE_SECTION_BETA),
                coordinate: schedule.source_index.into(),
                value: local.value.map(Into::into),
            },
            schedule.source_flags[PREFIX_SOURCE_BETA],
        );
        self.instance_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInstanceValueMessage {
                section: AB::Expr::from_usize(INSTANCE_SECTION_ETA),
                coordinate: AB::Expr::ZERO,
                value: local.value.map(Into::into),
            },
            schedule.source_flags[PREFIX_SOURCE_ETA],
        );
        self.local_claim_bus.add_key_with_lookups(
            builder,
            FixedMultiAirCompleteRegionalClaimMessage {
                region: schedule.source_index.into(),
                claim: local.value.map(Into::into),
            },
            schedule.local_claim_lookup_count,
        );
        self.interaction_claim_bus.add_key_with_lookups(
            builder,
            FixedMultiAirCompleteRegionalClaimMessage {
                region: schedule.source_index.into(),
                claim: local.value.map(Into::into),
            },
            schedule.interaction_claim_lookup_count,
        );

        self.region_sequence_start_bus.send(
            builder,
            FixedMultiAirCompleteRegionSequenceStartMessage {
                decomposition_end_tidx: local.decomposition_end_tidx.into(),
                protocol_component_count: AB::Expr::from_usize(
                    self.profile.protocol_component_count,
                ),
            },
            schedule.is_last,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteBetaScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub is_power_first: T,
    pub is_power_last: T,
    pub is_power_zero: T,
    pub power: T,
    pub tau_coordinate: T,
    pub tau_bit: T,
    pub current_power_coordinate: T,
    pub previous_power_coordinate: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteBetaCols<T> {
    pub one: [T; D_EF],
    pub beta: [T; D_EF],
    pub current_power: [T; D_EF],
    pub previous_power: [T; D_EF],
    pub tau: [T; D_EF],
    pub eq_before: [T; D_EF],
    pub eq_after: [T; D_EF],
    pub equation: [T; D_EF],
    pub weighted: [T; D_EF],
    pub running_before: [T; D_EF],
    pub running_after: [T; D_EF],
}

/// Constant-degree Eq automaton for the exact backend beta contribution.
/// Each beta equation receives one row per global `tau` coordinate, avoiding
/// a degree-`log_constraints` product inside one AIR row.
pub struct FixedMultiAirCompleteBetaAir {
    pub profile: Arc<FixedMultiAirCompletePrefixProfile>,
    pub instance_bus: FixedMultiAirCompleteInstanceValueBus,
    pub beta_claim_bus: FixedMultiAirCompleteBetaClaimBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteBetaAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteBetaAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirCompleteBetaScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteBetaCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteBetaAir {}
impl BaseAir<F> for FixedMultiAirCompleteBetaAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteBetaScheduleCols::<F>::width()
            + FixedMultiAirCompleteBetaCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteBetaAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached_row = builder.cached_mains()[0]
            .row_slice(0)
            .expect("complete terminal beta cached row")
            .to_vec();
        let next_cached_row = builder.cached_mains()[0]
            .row_slice(1)
            .expect("complete terminal beta next cached row")
            .to_vec();
        let main_row = builder
            .common_main()
            .row_slice(0)
            .expect("complete terminal beta row")
            .to_vec();
        let next_main_row = builder
            .common_main()
            .row_slice(1)
            .expect("complete terminal beta next row")
            .to_vec();
        let schedule: &FixedMultiAirCompleteBetaScheduleCols<AB::Var> =
            cached_row.as_slice().borrow();
        let next_schedule: &FixedMultiAirCompleteBetaScheduleCols<AB::Var> =
            next_cached_row.as_slice().borrow();
        let local: &FixedMultiAirCompleteBetaCols<AB::Var> = main_row.as_slice().borrow();
        let next: &FixedMultiAirCompleteBetaCols<AB::Var> = next_main_row.as_slice().borrow();

        for flag in [
            schedule.active,
            schedule.is_first,
            schedule.is_last,
            schedule.is_power_first,
            schedule.is_power_last,
            schedule.is_power_zero,
            schedule.tau_bit,
        ] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(schedule.active);
        builder.when_first_row().assert_one(schedule.is_first);
        builder
            .when_transition()
            .assert_bool(schedule.active - next_schedule.active);
        builder
            .when_transition()
            .assert_eq(schedule.active - next_schedule.active, schedule.is_last);
        builder
            .when_last_row()
            .assert_eq(schedule.is_last, schedule.active);
        builder
            .when(schedule.is_power_first)
            .assert_eq(schedule.tau_coordinate, AB::Expr::ZERO);
        builder.when(schedule.is_power_last).assert_eq(
            schedule.tau_coordinate,
            AB::Expr::from_usize(self.profile.log_constraints - 1),
        );

        let mut transition_builder = builder.when_transition();
        let mut same_power = transition_builder
            .when(next_schedule.active * (AB::Expr::ONE - next_schedule.is_power_first));
        same_power.assert_eq(next_schedule.power, schedule.power);
        assert_array_eq(&mut same_power, next.one, local.one.map(Into::into));
        assert_array_eq(&mut same_power, next.beta, local.beta.map(Into::into));
        assert_array_eq(
            &mut same_power,
            next.current_power,
            local.current_power.map(Into::into),
        );
        assert_array_eq(
            &mut same_power,
            next.previous_power,
            local.previous_power.map(Into::into),
        );
        assert_array_eq(
            &mut same_power,
            next.equation,
            local.equation.map(Into::into),
        );
        assert_array_eq(
            &mut same_power,
            next.eq_before,
            local.eq_after.map(Into::into),
        );

        let field_one = [
            AB::Expr::ONE,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        let zero = [
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        assert_array_eq(
            &mut builder.when(schedule.is_power_first),
            local.eq_before,
            field_one.clone(),
        );
        assert_array_eq(
            &mut builder.when(schedule.is_power_first * schedule.is_power_zero),
            local.previous_power,
            zero.clone(),
        );

        let tau = local.tau.map(Into::into);
        let one_minus_tau = ext_field_one_minus::<AB::Expr>(tau.clone());
        let eq_factor = core::array::from_fn(|limb| {
            schedule.tau_bit * tau[limb].clone()
                + (AB::Expr::ONE - schedule.tau_bit) * one_minus_tau[limb].clone()
        });
        let expected_eq_after =
            ext_field_multiply::<AB::Expr>(local.eq_before.map(Into::into), eq_factor);
        assert_array_eq(builder, local.eq_after, expected_eq_after);

        let current_minus_one = ext_field_subtract::<AB::Expr>(
            local.current_power.map(Into::into),
            local.one.map(Into::into),
        );
        let current_times_one = ext_field_multiply::<AB::Expr>(
            local.current_power.map(Into::into),
            local.one.map(Into::into),
        );
        let previous_times_beta = ext_field_multiply::<AB::Expr>(
            local.previous_power.map(Into::into),
            local.beta.map(Into::into),
        );
        let recurrence = ext_field_subtract::<AB::Expr>(current_times_one, previous_times_beta);
        let zero_equation = ext_field_scale_by_power(
            current_minus_one,
            local.one.map(Into::into),
            self.profile.exact_relation_degree.saturating_sub(1),
        );
        let recurrence_equation = ext_field_scale_by_power(
            recurrence,
            local.one.map(Into::into),
            self.profile.exact_relation_degree.saturating_sub(2),
        );
        let expected_equation = core::array::from_fn(|limb| {
            schedule.is_power_zero * zero_equation[limb].clone()
                + (AB::Expr::ONE - schedule.is_power_zero) * recurrence_equation[limb].clone()
        });
        assert_array_eq(
            &mut builder.when(schedule.is_power_first),
            local.equation,
            expected_equation,
        );

        let expected_weighted = ext_field_multiply::<AB::Expr>(
            local.equation.map(Into::into),
            local.eq_after.map(Into::into),
        );
        assert_array_eq(
            &mut builder.when(schedule.is_power_last),
            local.weighted,
            expected_weighted,
        );
        assert_array_eq(
            &mut builder.when(AB::Expr::ONE - schedule.is_power_last),
            local.weighted,
            zero,
        );
        assert_array_eq(
            &mut builder.when(schedule.is_first),
            local.running_before,
            [
                AB::Expr::ZERO,
                AB::Expr::ZERO,
                AB::Expr::ZERO,
                AB::Expr::ZERO,
            ],
        );
        let expected_running_after = ext_field_add::<AB::Expr>(
            local.running_before.map(Into::into),
            local.weighted.map(Into::into),
        );
        assert_array_eq(builder, local.running_after, expected_running_after);
        assert_array_eq(
            &mut builder.when_transition().when(next_schedule.active),
            next.running_before,
            local.running_after.map(Into::into),
        );

        self.instance_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInstanceValueMessage {
                section: AB::Expr::from_usize(INSTANCE_SECTION_BETA),
                coordinate: schedule.tau_coordinate.into(),
                value: local.tau.map(Into::into),
            },
            schedule.active,
        );
        for (coordinate, value, enabled) in [
            (
                self.profile.one_explicit_coordinate,
                local.one,
                schedule.is_power_first,
            ),
            (
                self.profile.beta_explicit_coordinate,
                local.beta,
                schedule.is_power_first,
            ),
        ] {
            self.instance_bus.lookup_key(
                builder,
                FixedMultiAirCompleteInstanceValueMessage {
                    section: AB::Expr::from_usize(INSTANCE_SECTION_BETA),
                    coordinate: AB::Expr::from_usize(coordinate),
                    value: value.map(Into::into),
                },
                enabled,
            );
        }
        self.instance_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInstanceValueMessage {
                section: AB::Expr::from_usize(INSTANCE_SECTION_BETA),
                coordinate: schedule.current_power_coordinate.into(),
                value: local.current_power.map(Into::into),
            },
            schedule.is_power_first,
        );
        self.instance_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInstanceValueMessage {
                section: AB::Expr::from_usize(INSTANCE_SECTION_BETA),
                coordinate: schedule.previous_power_coordinate.into(),
                value: local.previous_power.map(Into::into),
            },
            schedule.is_power_first * (AB::Expr::ONE - schedule.is_power_zero),
        );
        self.beta_claim_bus.send(
            builder,
            FixedMultiAirCompleteBetaClaimMessage {
                claim: local.running_after.map(Into::into),
            },
            schedule.is_last,
        );
    }
}

fn ext_field_scale_by_power<FA>(
    mut value: [FA; D_EF],
    base: [FA; D_EF],
    exponent: usize,
) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
    FA::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    for _ in 0..exponent {
        value = ext_field_multiply::<FA>(value, base.clone());
    }
    value
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteDecompositionScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub source_flags: [T; DECOMPOSITION_SOURCE_COUNT],
    pub source_index: T,
    pub interaction_present: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteDecompositionCols<T> {
    pub value: [T; D_EF],
    pub eta: [T; D_EF],
    pub beta_claim: [T; D_EF],
    pub running: [T; D_EF],
}

/// Enforces `beta_claim + sum(local) + sum(interaction) = eta` in the exact
/// setup-fixed order used by the backend verifier.
pub struct FixedMultiAirCompleteDecompositionAir {
    pub profile: Arc<FixedMultiAirCompletePrefixProfile>,
    pub instance_bus: FixedMultiAirCompleteInstanceValueBus,
    pub local_claim_bus: FixedMultiAirCompleteLocalClaimBus,
    pub interaction_claim_bus: FixedMultiAirCompleteInteractionClaimBus,
    pub beta_claim_bus: FixedMultiAirCompleteBetaClaimBus,
    pub receipt_bus: FixedMultiAirCompleteDecompositionReceiptBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteDecompositionAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteDecompositionAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirCompleteDecompositionScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteDecompositionCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteDecompositionAir {}
impl BaseAir<F> for FixedMultiAirCompleteDecompositionAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteDecompositionScheduleCols::<F>::width()
            + FixedMultiAirCompleteDecompositionCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteDecompositionAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let cached_row = builder.cached_mains()[0]
            .row_slice(0)
            .expect("complete terminal decomposition cached row")
            .to_vec();
        let next_cached_row = builder.cached_mains()[0]
            .row_slice(1)
            .expect("complete terminal decomposition next cached row")
            .to_vec();
        let main_row = builder
            .common_main()
            .row_slice(0)
            .expect("complete terminal decomposition row")
            .to_vec();
        let next_main_row = builder
            .common_main()
            .row_slice(1)
            .expect("complete terminal decomposition next row")
            .to_vec();
        let schedule: &FixedMultiAirCompleteDecompositionScheduleCols<AB::Var> =
            cached_row.as_slice().borrow();
        let next_schedule: &FixedMultiAirCompleteDecompositionScheduleCols<AB::Var> =
            next_cached_row.as_slice().borrow();
        let local: &FixedMultiAirCompleteDecompositionCols<AB::Var> = main_row.as_slice().borrow();
        let next: &FixedMultiAirCompleteDecompositionCols<AB::Var> =
            next_main_row.as_slice().borrow();

        for flag in [
            schedule.active,
            schedule.is_first,
            schedule.is_last,
            schedule.interaction_present,
        ]
        .into_iter()
        .chain(schedule.source_flags)
        {
            builder.assert_bool(flag);
        }
        let source_sum = schedule
            .source_flags
            .into_iter()
            .fold(AB::Expr::ZERO, |sum, flag| sum + flag);
        builder.assert_eq(source_sum, schedule.active);
        builder.when_first_row().assert_one(schedule.active);
        builder.when_first_row().assert_one(schedule.is_first);
        builder
            .when_transition()
            .assert_bool(schedule.active - next_schedule.active);
        builder
            .when_transition()
            .assert_eq(schedule.active - next_schedule.active, schedule.is_last);
        builder
            .when_last_row()
            .assert_eq(schedule.is_last, schedule.active);

        let mut transition_builder = builder.when_transition();
        let mut transition = transition_builder.when(next_schedule.active);
        assert_array_eq(&mut transition, next.eta, local.eta.map(Into::into));
        assert_array_eq(
            &mut transition,
            next.beta_claim,
            local.beta_claim.map(Into::into),
        );
        let expected_next_running =
            ext_field_add::<AB::Expr>(local.running.map(Into::into), next.value.map(Into::into));
        assert_array_eq(&mut transition, next.running, expected_next_running);
        assert_array_eq(
            &mut builder.when(schedule.is_first),
            local.running,
            local.value.map(Into::into),
        );
        assert_array_eq(
            &mut builder.when(schedule.source_flags[DECOMPOSITION_SOURCE_BETA]),
            local.value,
            local.beta_claim.map(Into::into),
        );
        assert_array_eq(
            &mut builder.when(
                schedule.source_flags[DECOMPOSITION_SOURCE_INTERACTION]
                    * (AB::Expr::ONE - schedule.interaction_present),
            ),
            local.value,
            [
                AB::Expr::ZERO,
                AB::Expr::ZERO,
                AB::Expr::ZERO,
                AB::Expr::ZERO,
            ],
        );
        assert_array_eq(
            &mut builder.when(schedule.is_last),
            local.running,
            local.eta.map(Into::into),
        );

        self.instance_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInstanceValueMessage {
                section: AB::Expr::from_usize(INSTANCE_SECTION_ETA),
                coordinate: AB::Expr::ZERO,
                value: local.eta.map(Into::into),
            },
            schedule.is_first,
        );
        self.local_claim_bus.lookup_key(
            builder,
            FixedMultiAirCompleteRegionalClaimMessage {
                region: schedule.source_index.into(),
                claim: local.value.map(Into::into),
            },
            schedule.source_flags[DECOMPOSITION_SOURCE_LOCAL],
        );
        self.interaction_claim_bus.lookup_key(
            builder,
            FixedMultiAirCompleteRegionalClaimMessage {
                region: schedule.source_index.into(),
                claim: local.value.map(Into::into),
            },
            schedule.source_flags[DECOMPOSITION_SOURCE_INTERACTION],
        );
        self.beta_claim_bus.receive(
            builder,
            FixedMultiAirCompleteBetaClaimMessage {
                claim: local.beta_claim.map(Into::into),
            },
            schedule.is_first,
        );
        self.receipt_bus.send(
            builder,
            FixedMultiAirCompleteDecompositionReceiptMessage {
                eta: local.eta.map(Into::into),
                beta_claim: local.beta_claim.map(Into::into),
                local_count: AB::Expr::from_usize(self.profile.region_count),
                interaction_count: AB::Expr::from_usize(self.profile.region_count),
            },
            schedule.is_last,
        );
    }
}

pub struct FixedMultiAirCompletePrefixDecompositionAirs {
    pub instance: FixedMultiAirCompleteInstanceAir,
    pub transcript_prefix: FixedMultiAirCompleteTranscriptPrefixAir,
    pub beta: FixedMultiAirCompleteBetaAir,
    pub decomposition: FixedMultiAirCompleteDecompositionAir,
}

impl FixedMultiAirCompletePrefixDecompositionAirs {
    #[must_use]
    pub fn new(
        profile: Arc<FixedMultiAirCompletePrefixProfile>,
        proof_idx: usize,
        buses: FixedMultiAirCompletePrefixBuses,
    ) -> Self {
        Self {
            instance: FixedMultiAirCompleteInstanceAir {
                profile: profile.clone(),
                binding_bus: buses.binding,
                instance_bus: buses.instance,
            },
            transcript_prefix: FixedMultiAirCompleteTranscriptPrefixAir {
                profile: profile.clone(),
                proof_idx,
                transcript_bus: buses.transcript,
                statement_start_cursor_bus: buses.statement_start_cursor,
                instance_bus: buses.instance,
                local_claim_bus: buses.local_claim,
                interaction_claim_bus: buses.interaction_claim,
                region_sequence_start_bus: buses.region_sequence_start,
            },
            beta: FixedMultiAirCompleteBetaAir {
                profile: profile.clone(),
                instance_bus: buses.instance,
                beta_claim_bus: buses.beta_claim,
            },
            decomposition: FixedMultiAirCompleteDecompositionAir {
                profile,
                instance_bus: buses.instance,
                local_claim_bus: buses.local_claim,
                interaction_claim_bus: buses.interaction_claim,
                beta_claim_bus: buses.beta_claim,
                receipt_bus: buses.decomposition_receipt,
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompletePartitionedTrace {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompletePrefixDecompositionTraceData {
    /// Same order as [`FixedMultiAirCompletePrefixDecompositionAirs`].
    pub instance: FixedMultiAirCompletePartitionedTrace,
    pub transcript_prefix: FixedMultiAirCompletePartitionedTrace,
    pub beta: FixedMultiAirCompletePartitionedTrace,
    pub decomposition: FixedMultiAirCompletePartitionedTrace,
    pub decomposition_end_tidx: usize,
    pub beta_claim: EF,
}

pub struct FixedMultiAirCompletePrefixWitness<'a> {
    /// Root authenticated by the preceding final-VACC handoff.
    pub authenticated_root: Digest,
    pub instance: &'a AccumulatorInstance<EF, Digest>,
    pub proof: &'a FixedMultiAirCompleteTerminalProof<EF>,
    pub transcript: &'a TranscriptLog<F, [F; 16]>,
    pub statement_start_tidx: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FixedMultiAirCompletePrefixRequiredHeights {
    pub instance: Option<usize>,
    pub transcript_prefix: Option<usize>,
    pub beta: Option<usize>,
    pub decomposition: Option<usize>,
}

impl FixedMultiAirCompletePrefixDecompositionAirs {
    pub fn generate_traces(
        &self,
        witness: FixedMultiAirCompletePrefixWitness<'_>,
        required: FixedMultiAirCompletePrefixRequiredHeights,
    ) -> PrefixResult<FixedMultiAirCompletePrefixDecompositionTraceData> {
        let profile = self.instance.profile.as_ref();
        validate_witness(profile, &witness)?;
        let beta_claim = reference_beta_claim(profile, &witness.instance.beta)?;
        let decomposition_sum = beta_claim
            + witness.proof.local_claims.iter().copied().sum::<EF>()
            + witness.proof.interaction_claims.iter().copied().sum::<EF>();
        if decomposition_sum != witness.instance.eta {
            return Err(FixedMultiAirCompletePrefixError::Claim);
        }

        let instance = generate_instance_trace(profile, witness.instance, required.instance)?;
        let (transcript_prefix, decomposition_end_tidx) = generate_transcript_prefix_trace(
            profile,
            witness.instance,
            witness.proof,
            witness.transcript,
            witness.statement_start_tidx,
            required.transcript_prefix,
        )?;
        let beta = generate_beta_trace(profile, witness.instance, required.beta)?;
        let decomposition = generate_decomposition_trace(
            profile,
            witness.instance,
            witness.proof,
            beta_claim,
            required.decomposition,
        )?;
        Ok(FixedMultiAirCompletePrefixDecompositionTraceData {
            instance,
            transcript_prefix,
            beta,
            decomposition,
            decomposition_end_tidx,
            beta_claim,
        })
    }
}

fn validate_witness(
    profile: &FixedMultiAirCompletePrefixProfile,
    witness: &FixedMultiAirCompletePrefixWitness<'_>,
) -> PrefixResult<()> {
    let instance = witness.instance;
    let proof = witness.proof;
    if witness.authenticated_root != instance.rt {
        return Err(FixedMultiAirCompletePrefixError::Root);
    }
    if instance.alpha.len() != profile.alpha_len
        || instance.beta.len() != profile.beta_len
        || proof.local_claims.len() != profile.region_count
        || proof.interaction_claims.len() != profile.region_count
        || proof.local_proofs.len() != profile.region_count
        || proof.interaction_proofs.len() != profile.region_count
    {
        return Err(FixedMultiAirCompletePrefixError::Shape(
            "instance or regional proof dimensions",
        ));
    }
    for region in 0..profile.region_count {
        let actual = proof.interaction_proofs[region].is_some();
        let expected = profile.interaction_present[region];
        if actual != expected {
            return Err(FixedMultiAirCompletePrefixError::InteractionPresence { region });
        }
        if !expected && proof.interaction_claims[region] != EF::ZERO {
            return Err(FixedMultiAirCompletePrefixError::Claim);
        }
    }
    Ok(())
}

fn generate_instance_trace(
    profile: &FixedMultiAirCompletePrefixProfile,
    instance: &AccumulatorInstance<EF, Digest>,
    required_height: Option<usize>,
) -> PrefixResult<FixedMultiAirCompletePartitionedTrace> {
    let rows = profile.instance_row_count();
    let height = admitted_height(rows, required_height, "instance trace height")?;
    let cached_width = FixedMultiAirCompleteInstanceScheduleCols::<F>::width();
    let common_width = FixedMultiAirCompleteInstanceCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    let mut entries = Vec::with_capacity(rows);
    for (coordinate, &value) in instance.alpha.iter().enumerate() {
        entries.push((INSTANCE_SECTION_ALPHA, coordinate, value));
    }
    entries.push((INSTANCE_SECTION_MU, 0, instance.mu));
    for (coordinate, &value) in instance.beta.iter().enumerate() {
        entries.push((INSTANCE_SECTION_BETA, coordinate, value));
    }
    entries.push((INSTANCE_SECTION_ETA, 0, instance.eta));
    if entries.len() != rows {
        return Err(FixedMultiAirCompletePrefixError::Shape("instance schedule"));
    }
    for (row, &(section, coordinate, value)) in entries.iter().enumerate() {
        let schedule: &mut FixedMultiAirCompleteInstanceScheduleCols<F> =
            cached[row * cached_width..(row + 1) * cached_width].borrow_mut();
        schedule.active = F::ONE;
        schedule.is_first = F::from_bool(row == 0);
        schedule.is_last = F::from_bool(row + 1 == rows);
        schedule.section_flags[section] = F::ONE;
        schedule.coordinate = F::from_usize(coordinate);
        schedule.lookup_count = F::from_u32(profile.instance_lookup_counts[section][coordinate]);
        let cols: &mut FixedMultiAirCompleteInstanceCols<F> =
            common[row * common_width..(row + 1) * common_width].borrow_mut();
        copy_ext(&mut cols.value, value);
        cols.root.copy_from_slice(&instance.rt);
    }
    Ok(FixedMultiAirCompletePartitionedTrace {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
    })
}

fn generate_transcript_prefix_trace(
    profile: &FixedMultiAirCompletePrefixProfile,
    instance: &AccumulatorInstance<EF, Digest>,
    proof: &FixedMultiAirCompleteTerminalProof<EF>,
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
    required_height: Option<usize>,
) -> PrefixResult<(FixedMultiAirCompletePartitionedTrace, usize)> {
    let rows = profile.transcript_schedule.len();
    let height = admitted_height(rows, required_height, "transcript-prefix trace height")?;
    let transcript_slots =
        rows.checked_mul(D_EF)
            .ok_or(FixedMultiAirCompletePrefixError::Overflow(
                "transcript-prefix slots",
            ))?;
    let end_tidx = start_tidx.checked_add(transcript_slots).ok_or(
        FixedMultiAirCompletePrefixError::Overflow("transcript-prefix cursor"),
    )?;
    if end_tidx >= F::ORDER_U32 as usize {
        return Err(FixedMultiAirCompletePrefixError::Overflow(
            "transcript-prefix field cursor",
        ));
    }
    let cached_width = FixedMultiAirCompletePrefixScheduleCols::<F>::width();
    let common_width = FixedMultiAirCompletePrefixCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    let mut tidx = start_tidx;
    for (row, entry) in profile.transcript_schedule.iter().enumerate() {
        let value = prefix_source_value(entry.source, instance, proof)?;
        expect_ext(transcript, tidx, value)?;
        let schedule: &mut FixedMultiAirCompletePrefixScheduleCols<F> =
            cached[row * cached_width..(row + 1) * cached_width].borrow_mut();
        schedule.active = F::ONE;
        schedule.is_first = F::from_bool(row == 0);
        schedule.is_last = F::from_bool(row + 1 == rows);
        schedule.interaction_present = F::from_bool(entry.interaction_present);
        match entry.source {
            PrefixSource::Constant(constant) => {
                schedule.source_flags[PREFIX_SOURCE_CONSTANT] = F::ONE;
                copy_ext(&mut schedule.constant, constant);
            }
            PrefixSource::Beta(coordinate) => {
                schedule.source_flags[PREFIX_SOURCE_BETA] = F::ONE;
                schedule.source_index = F::from_usize(coordinate);
            }
            PrefixSource::Eta => {
                schedule.source_flags[PREFIX_SOURCE_ETA] = F::ONE;
            }
            PrefixSource::LocalClaim(region) => {
                schedule.source_flags[PREFIX_SOURCE_LOCAL_CLAIM] = F::ONE;
                schedule.source_index = F::from_usize(region);
                schedule.local_claim_lookup_count =
                    F::from_u32(profile.local_claim_lookup_counts[region]);
            }
            PrefixSource::InteractionClaim(region) => {
                schedule.source_flags[PREFIX_SOURCE_INTERACTION_CLAIM] = F::ONE;
                schedule.source_index = F::from_usize(region);
                schedule.expected_proof_present = F::from_bool(entry.interaction_present);
                schedule.interaction_claim_lookup_count =
                    F::from_u32(profile.interaction_claim_lookup_counts[region]);
            }
        }
        let cols: &mut FixedMultiAirCompletePrefixCols<F> =
            common[row * common_width..(row + 1) * common_width].borrow_mut();
        cols.tidx = F::from_usize(tidx);
        cols.decomposition_end_tidx = F::from_usize(end_tidx);
        copy_ext(&mut cols.value, value);
        if let PrefixSource::InteractionClaim(region) = entry.source {
            cols.interaction_proof_present =
                F::from_bool(proof.interaction_proofs[region].is_some());
        }
        tidx = tidx
            .checked_add(D_EF)
            .ok_or(FixedMultiAirCompletePrefixError::Overflow(
                "transcript-prefix row cursor",
            ))?;
    }
    if tidx != end_tidx {
        return Err(FixedMultiAirCompletePrefixError::Transcript(
            "prefix cursor schedule",
        ));
    }
    Ok((
        FixedMultiAirCompletePartitionedTrace {
            cached: RowMajorMatrix::new(cached, cached_width),
            common: RowMajorMatrix::new(common, common_width),
        },
        end_tidx,
    ))
}

fn prefix_source_value(
    source: PrefixSource,
    instance: &AccumulatorInstance<EF, Digest>,
    proof: &FixedMultiAirCompleteTerminalProof<EF>,
) -> PrefixResult<EF> {
    match source {
        PrefixSource::Constant(value) => Ok(value),
        PrefixSource::Beta(coordinate) => {
            instance
                .beta
                .get(coordinate)
                .copied()
                .ok_or(FixedMultiAirCompletePrefixError::Shape(
                    "beta transcript coordinate",
                ))
        }
        PrefixSource::Eta => Ok(instance.eta),
        PrefixSource::LocalClaim(region) => {
            proof
                .local_claims
                .get(region)
                .copied()
                .ok_or(FixedMultiAirCompletePrefixError::Shape(
                    "local claim transcript coordinate",
                ))
        }
        PrefixSource::InteractionClaim(region) => {
            proof.interaction_claims.get(region).copied().ok_or(
                FixedMultiAirCompletePrefixError::Shape("interaction claim transcript coordinate"),
            )
        }
    }
}

fn generate_beta_trace(
    profile: &FixedMultiAirCompletePrefixProfile,
    instance: &AccumulatorInstance<EF, Digest>,
    required_height: Option<usize>,
) -> PrefixResult<FixedMultiAirCompletePartitionedTrace> {
    let rows = profile.beta_row_count();
    let height = admitted_height(rows, required_height, "beta trace height")?;
    let cached_width = FixedMultiAirCompleteBetaScheduleCols::<F>::width();
    let common_width = FixedMultiAirCompleteBetaCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    let tau = instance
        .beta
        .get(..profile.log_constraints)
        .ok_or(FixedMultiAirCompletePrefixError::Shape("tau coordinates"))?;
    let one = instance.beta[profile.one_explicit_coordinate];
    let beta = instance.beta[profile.beta_explicit_coordinate];
    let mut running = EF::ZERO;
    let mut row = 0usize;
    for power in 0..profile.beta_power_count {
        let current_coordinate = profile.beta_power_explicit_start + power;
        let previous_coordinate = if power == 0 {
            0
        } else {
            current_coordinate - 1
        };
        let current = instance.beta[current_coordinate];
        let previous = if power == 0 {
            EF::ZERO
        } else {
            instance.beta[previous_coordinate]
        };
        let equation = beta_power_equation(
            one,
            beta,
            current,
            previous,
            power,
            profile.exact_relation_degree,
        );
        let constraint_index = profile.beta_constraint_start.checked_add(power).ok_or(
            FixedMultiAirCompletePrefixError::Overflow("beta constraint index"),
        )?;
        let mut eq = EF::ONE;
        for (coordinate, &tau_value) in tau.iter().enumerate() {
            let bit = ((constraint_index >> (profile.log_constraints - 1 - coordinate)) & 1) != 0;
            let eq_before = eq;
            eq *= if bit { tau_value } else { EF::ONE - tau_value };
            let weighted = if coordinate + 1 == profile.log_constraints {
                equation * eq
            } else {
                EF::ZERO
            };
            let running_before = running;
            running += weighted;

            let schedule: &mut FixedMultiAirCompleteBetaScheduleCols<F> =
                cached[row * cached_width..(row + 1) * cached_width].borrow_mut();
            schedule.active = F::ONE;
            schedule.is_first = F::from_bool(row == 0);
            schedule.is_last = F::from_bool(row + 1 == rows);
            schedule.is_power_first = F::from_bool(coordinate == 0);
            schedule.is_power_last = F::from_bool(coordinate + 1 == profile.log_constraints);
            schedule.is_power_zero = F::from_bool(power == 0);
            schedule.power = F::from_usize(power);
            schedule.tau_coordinate = F::from_usize(coordinate);
            schedule.tau_bit = F::from_bool(bit);
            schedule.current_power_coordinate = F::from_usize(current_coordinate);
            schedule.previous_power_coordinate = F::from_usize(previous_coordinate);

            let cols: &mut FixedMultiAirCompleteBetaCols<F> =
                common[row * common_width..(row + 1) * common_width].borrow_mut();
            copy_ext(&mut cols.one, one);
            copy_ext(&mut cols.beta, beta);
            copy_ext(&mut cols.current_power, current);
            copy_ext(&mut cols.previous_power, previous);
            copy_ext(&mut cols.tau, tau_value);
            copy_ext(&mut cols.eq_before, eq_before);
            copy_ext(&mut cols.eq_after, eq);
            copy_ext(&mut cols.equation, equation);
            copy_ext(&mut cols.weighted, weighted);
            copy_ext(&mut cols.running_before, running_before);
            copy_ext(&mut cols.running_after, running);
            row += 1;
        }
    }
    if row != rows || running != reference_beta_claim(profile, &instance.beta)? {
        return Err(FixedMultiAirCompletePrefixError::Claim);
    }
    Ok(FixedMultiAirCompletePartitionedTrace {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
    })
}

fn generate_decomposition_trace(
    profile: &FixedMultiAirCompletePrefixProfile,
    instance: &AccumulatorInstance<EF, Digest>,
    proof: &FixedMultiAirCompleteTerminalProof<EF>,
    beta_claim: EF,
    required_height: Option<usize>,
) -> PrefixResult<FixedMultiAirCompletePartitionedTrace> {
    let rows = profile.decomposition_row_count();
    let height = admitted_height(rows, required_height, "decomposition trace height")?;
    let cached_width = FixedMultiAirCompleteDecompositionScheduleCols::<F>::width();
    let common_width = FixedMultiAirCompleteDecompositionCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    let mut values = Vec::with_capacity(rows);
    values.push((DECOMPOSITION_SOURCE_BETA, 0usize, true, beta_claim));
    values.extend(
        proof
            .local_claims
            .iter()
            .copied()
            .enumerate()
            .map(|(region, value)| (DECOMPOSITION_SOURCE_LOCAL, region, true, value)),
    );
    values.extend(
        proof
            .interaction_claims
            .iter()
            .copied()
            .enumerate()
            .map(|(region, value)| {
                (
                    DECOMPOSITION_SOURCE_INTERACTION,
                    region,
                    profile.interaction_present[region],
                    value,
                )
            }),
    );
    let mut running = EF::ZERO;
    for (row, &(source, source_index, present, value)) in values.iter().enumerate() {
        running += value;
        let schedule: &mut FixedMultiAirCompleteDecompositionScheduleCols<F> =
            cached[row * cached_width..(row + 1) * cached_width].borrow_mut();
        schedule.active = F::ONE;
        schedule.is_first = F::from_bool(row == 0);
        schedule.is_last = F::from_bool(row + 1 == rows);
        schedule.source_flags[source] = F::ONE;
        schedule.source_index = F::from_usize(source_index);
        schedule.interaction_present = F::from_bool(present);
        let cols: &mut FixedMultiAirCompleteDecompositionCols<F> =
            common[row * common_width..(row + 1) * common_width].borrow_mut();
        copy_ext(&mut cols.value, value);
        copy_ext(&mut cols.eta, instance.eta);
        copy_ext(&mut cols.beta_claim, beta_claim);
        copy_ext(&mut cols.running, running);
    }
    if running != instance.eta {
        return Err(FixedMultiAirCompletePrefixError::Claim);
    }
    Ok(FixedMultiAirCompletePartitionedTrace {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
    })
}

pub fn reference_beta_claim(
    profile: &FixedMultiAirCompletePrefixProfile,
    beta_point: &[EF],
) -> PrefixResult<EF> {
    if beta_point.len() != profile.beta_len {
        return Err(FixedMultiAirCompletePrefixError::Shape(
            "beta-claim point length",
        ));
    }
    let tau = &beta_point[..profile.log_constraints];
    let one = beta_point[profile.one_explicit_coordinate];
    let beta = beta_point[profile.beta_explicit_coordinate];
    let mut claim = EF::ZERO;
    for power in 0..profile.beta_power_count {
        let current = beta_point[profile.beta_power_explicit_start + power];
        let previous = if power == 0 {
            EF::ZERO
        } else {
            beta_point[profile.beta_power_explicit_start + power - 1]
        };
        let equation = beta_power_equation(
            one,
            beta,
            current,
            previous,
            power,
            profile.exact_relation_degree,
        );
        let index = profile.beta_constraint_start.checked_add(power).ok_or(
            FixedMultiAirCompletePrefixError::Overflow("beta-claim constraint index"),
        )?;
        claim += equation * eval_eq_index_checked(tau, index)?;
    }
    Ok(claim)
}

fn beta_power_equation(
    one: EF,
    beta: EF,
    current: EF,
    previous: EF,
    power: usize,
    degree: usize,
) -> EF {
    if power == 0 {
        scale_by_power(current - one, one, degree.saturating_sub(1))
    } else {
        scale_by_power(
            current * one - previous * beta,
            one,
            degree.saturating_sub(2),
        )
    }
}

fn scale_by_power(mut value: EF, base: EF, exponent: usize) -> EF {
    for _ in 0..exponent {
        value *= base;
    }
    value
}

fn eval_eq_index_checked(point: &[EF], index: usize) -> PrefixResult<EF> {
    let domain = 1usize
        .checked_shl(
            u32::try_from(point.len())
                .map_err(|_| FixedMultiAirCompletePrefixError::Overflow("Eq point dimension"))?,
        )
        .ok_or(FixedMultiAirCompletePrefixError::Overflow(
            "Eq point domain",
        ))?;
    if index >= domain {
        return Err(FixedMultiAirCompletePrefixError::Shape(
            "Eq index outside constraint domain",
        ));
    }
    Ok(point
        .iter()
        .enumerate()
        .fold(EF::ONE, |weight, (coordinate, &value)| {
            let bit = (index >> (point.len() - 1 - coordinate)) & 1;
            weight * if bit == 0 { EF::ONE - value } else { value }
        }))
}

fn admitted_height(
    rows: usize,
    required_height: Option<usize>,
    context: &'static str,
) -> PrefixResult<usize> {
    if rows == 0 {
        return Err(FixedMultiAirCompletePrefixError::Shape(context));
    }
    let minimum = rows
        .checked_next_power_of_two()
        .ok_or(FixedMultiAirCompletePrefixError::Overflow(context))?;
    match required_height {
        Some(height) if height.is_power_of_two() && height >= rows => Ok(height),
        Some(_) => Err(FixedMultiAirCompletePrefixError::Shape(context)),
        None => Ok(minimum),
    }
}

fn expect_ext(
    transcript: &TranscriptLog<F, [F; 16]>,
    tidx: usize,
    expected: EF,
) -> PrefixResult<()> {
    let end = tidx
        .checked_add(D_EF)
        .ok_or(FixedMultiAirCompletePrefixError::Overflow(
            "transcript observation",
        ))?;
    let values =
        transcript
            .values()
            .get(tidx..end)
            .ok_or(FixedMultiAirCompletePrefixError::Transcript(
                "observation length",
            ))?;
    let sample_flags =
        transcript
            .samples()
            .get(tidx..end)
            .ok_or(FixedMultiAirCompletePrefixError::Transcript(
                "sample-flag length",
            ))?;
    if sample_flags.iter().any(|&is_sample| is_sample)
        || EF::from_basis_coefficients_slice(values) != Some(expected)
    {
        return Err(FixedMultiAirCompletePrefixError::Transcript(
            "observation value or kind",
        ));
    }
    Ok(())
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use openvm_stark_backend::{
        air_builders::debug::{check_constraints, DebugConstraintBuilder},
        native_warp::{
            DirectAirCodeClass, FixedMultiAirCompleteRegionSumcheckProof,
            FixedMultiAirCompleteTerminalInteraction, FixedMultiAirCompleteTerminalInterval,
            FixedMultiAirCompleteTerminalQColumn, FixedMultiAirCompleteTerminalRegion,
        },
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config as SC;

    use super::*;

    fn ef4(seed: u64) -> EF {
        EF::from_basis_coefficients_slice(&[
            F::from_u64(seed),
            F::from_u64(seed + 1),
            F::from_u64(seed + 2),
            F::from_u64(seed + 3),
        ])
        .expect("four EF4 coordinates")
    }

    fn interval(start: u64, len: u64) -> FixedMultiAirCompleteTerminalInterval {
        FixedMultiAirCompleteTerminalInterval { start, len }
    }

    fn digest(seed: u32) -> Digest {
        core::array::from_fn(|index| F::from_u32(seed + index as u32))
    }

    fn metadata() -> FixedMultiAirCompleteTerminalMetadata<Digest> {
        let q_columns = core::array::from_fn(|coordinate| FixedMultiAirCompleteTerminalQColumn {
            coordinate: coordinate as u8,
            message: interval(8 + 4 * coordinate as u64, 4),
        });
        let interaction = FixedMultiAirCompleteTerminalInteraction {
            interaction_ordinal: 0,
            bus_index: 7,
            message_len: 2,
            count_degree: 1,
            denominator_degree: 3,
            inverse_constraints: interval(7, 4),
            q_columns,
        };
        FixedMultiAirCompleteTerminalMetadata {
            terminal_version: FIXED_MULTI_AIR_COMPLETE_TERMINAL_VERSION,
            relation_version: FIXED_MULTI_AIR_COMPLETE_PESAT_VERSION,
            relation_digest: digest(100),
            transcript: FixedMultiAirCompleteTranscriptBinding::V4,
            constraint_layout:
                FixedMultiAirCompleteConstraintLayout::LocalAirRegionConstraintRowThenBetaThenRegionInteractionRowThenGlobalV2,
            witness_layout:
                FixedMultiAirCompleteWitnessLayout::TracePrefixThenRegionInteractionCoordinateRowV2,
            record_layout:
                FixedMultiAirCompleteTerminalRecordLayout::RegionInteractionRowV1,
            code_class: DirectAirCodeClass {
                log_message_len: 5,
                log_blowup: 1,
                log_codeword_len: 6,
                initial_folding_factor: 0,
                rows_per_query: 1,
            },
            exact_relation_degree: FIXED_MULTI_AIR_COMPLETE_WARP_DEGREE as u8,
            terminal_round_degree: FIXED_MULTI_AIR_COMPLETE_TERMINAL_ROUND_DEGREE as u8,
            explicit_len: 7,
            log_constraints: 4,
            trace_message: interval(0, 8),
            inverse_coordinates: interval(8, 16),
            raw_message: interval(0, 24),
            padded_message: interval(0, 32),
            local_constraints: interval(0, 4),
            beta_constraints: interval(4, 3),
            inverse_constraints: interval(7, 4),
            global_constraint: interval(11, 1),
            zero_constraint_tail: interval(12, 4),
            alpha_explicit_offset: 1,
            beta_explicit_offset: 2,
            beta_power_explicit_offset: 3,
            beta_power_count: 3,
            public_explicit_offset: 6,
            regions: vec![
                FixedMultiAirCompleteTerminalRegion {
                    setup_ordinal: 0,
                    air_id: 3,
                    log_height: 2,
                    constraints_per_row: 1,
                    trace_message: interval(0, 4),
                    local_constraints: interval(0, 2),
                    interaction_record_offset: 0,
                    interactions: Vec::new(),
                },
                FixedMultiAirCompleteTerminalRegion {
                    setup_ordinal: 1,
                    air_id: 9,
                    log_height: 2,
                    constraints_per_row: 1,
                    trace_message: interval(4, 4),
                    local_constraints: interval(2, 2),
                    interaction_record_offset: 0,
                    interactions: vec![interaction],
                },
            ],
        }
    }

    fn profile() -> Arc<FixedMultiAirCompletePrefixProfile> {
        let metadata = metadata();
        let beta_len = usize::from(metadata.log_constraints) + metadata.explicit_len as usize;
        let counts = FixedMultiAirCompleteExternalLookupCounts::semantic_minimum(
            usize::from(metadata.code_class.log_codeword_len),
            beta_len,
            metadata.regions.len(),
        );
        Arc::new(
            FixedMultiAirCompletePrefixProfile::from_metadata(metadata, counts)
                .expect("complete prefix profile"),
        )
    }

    fn buses() -> FixedMultiAirCompletePrefixBuses {
        FixedMultiAirCompletePrefixBuses {
            transcript: TranscriptBus::new(101),
            statement_start_cursor: FixedMultiAirCompleteStatementStartCursorBus::new(102),
            binding: FixedMultiAirCompleteBindingBus::new(103),
            instance: FixedMultiAirCompleteInstanceValueBus::new(104),
            local_claim: FixedMultiAirCompleteLocalClaimBus::new(105),
            interaction_claim: FixedMultiAirCompleteInteractionClaimBus::new(106),
            region_sequence_start: FixedMultiAirCompleteRegionSequenceStartBus::new(107),
            beta_claim: FixedMultiAirCompleteBetaClaimBus::new(108),
            decomposition_receipt: FixedMultiAirCompleteDecompositionReceiptBus::new(109),
        }
    }

    fn instance_and_proof(
        profile: &FixedMultiAirCompletePrefixProfile,
    ) -> (
        AccumulatorInstance<EF, Digest>,
        FixedMultiAirCompleteTerminalProof<EF>,
    ) {
        let logup_beta = ef4(19);
        let mut beta = vec![ef4(2), ef4(5), ef4(8), ef4(11)];
        beta.extend([
            EF::ONE,
            ef4(17),
            logup_beta,
            EF::ONE,
            logup_beta,
            logup_beta * logup_beta,
            ef4(23),
        ]);
        assert_eq!(beta.len(), profile.beta_len);
        let local_claims = vec![ef4(29), ef4(31)];
        let interaction_claims = vec![EF::ZERO, ef4(37)];
        let beta_claim = reference_beta_claim(profile, &beta).expect("beta claim");
        assert_eq!(beta_claim, EF::ZERO);
        let eta = beta_claim
            + local_claims.iter().copied().sum::<EF>()
            + interaction_claims.iter().copied().sum::<EF>();
        let shell = FixedMultiAirCompleteRegionSumcheckProof {
            round_evaluations: Vec::new(),
            opened_columns: Vec::new(),
        };
        (
            AccumulatorInstance {
                rt: digest(200),
                alpha: (0..profile.alpha_len)
                    .map(|index| ef4(41 + 4 * index as u64))
                    .collect(),
                mu: ef4(71),
                beta,
                eta,
            },
            FixedMultiAirCompleteTerminalProof {
                local_claims,
                interaction_claims,
                local_proofs: vec![shell.clone(), shell.clone()],
                interaction_proofs: vec![None, Some(shell)],
            },
        )
    }

    fn transcript_for(
        profile: &FixedMultiAirCompletePrefixProfile,
        instance: &AccumulatorInstance<EF, Digest>,
        proof: &FixedMultiAirCompleteTerminalProof<EF>,
        start_tidx: usize,
    ) -> TranscriptLog<F, [F; 16]> {
        let mut values = vec![F::from_u32(999); start_tidx];
        for entry in &profile.transcript_schedule {
            let value = prefix_source_value(entry.source, instance, proof)
                .expect("scheduled transcript value");
            values.extend_from_slice(value.as_basis_coefficients_slice());
        }
        TranscriptLog::new(values.clone(), vec![false; values.len()])
    }

    fn airs_and_witness() -> (
        FixedMultiAirCompletePrefixDecompositionAirs,
        AccumulatorInstance<EF, Digest>,
        FixedMultiAirCompleteTerminalProof<EF>,
        TranscriptLog<F, [F; 16]>,
        usize,
    ) {
        let profile = profile();
        let (instance, proof) = instance_and_proof(&profile);
        let start_tidx = 8;
        let transcript = transcript_for(&profile, &instance, &proof, start_tidx);
        (
            FixedMultiAirCompletePrefixDecompositionAirs::new(profile, 0, buses()),
            instance,
            proof,
            transcript,
            start_tidx,
        )
    }

    fn generate(
        airs: &FixedMultiAirCompletePrefixDecompositionAirs,
        instance: &AccumulatorInstance<EF, Digest>,
        proof: &FixedMultiAirCompleteTerminalProof<EF>,
        transcript: &TranscriptLog<F, [F; 16]>,
        start_tidx: usize,
    ) -> PrefixResult<FixedMultiAirCompletePrefixDecompositionTraceData> {
        airs.generate_traces(
            FixedMultiAirCompletePrefixWitness {
                authenticated_root: instance.rt,
                instance,
                proof,
                transcript,
                statement_start_tidx: start_tidx,
            },
            FixedMultiAirCompletePrefixRequiredHeights::default(),
        )
    }

    fn check_partitioned<A>(air: &A, trace: &FixedMultiAirCompletePartitionedTrace)
    where
        A: for<'a> Air<DebugConstraintBuilder<'a, SC>>
            + BaseAir<F>
            + BaseAirWithPublicValues<F>
            + PartitionedBaseAir<F>,
    {
        check_constraints::<_, SC>(
            air,
            core::any::type_name::<A>(),
            &None,
            &[trace.cached.as_view(), trace.common.as_view()],
            &[],
        );
    }

    #[test]
    fn honest_ef4_statement_beta_and_decomposition_constraints_hold() {
        let (airs, instance, proof, transcript, start_tidx) = airs_and_witness();
        let traces = generate(&airs, &instance, &proof, &transcript, start_tidx)
            .expect("honest complete prefix traces");
        assert_eq!(traces.beta_claim, EF::ZERO);
        assert_eq!(
            traces.decomposition_end_tidx,
            start_tidx + airs.instance.profile.transcript_observation_count() * D_EF
        );
        check_partitioned(&airs.instance, &traces.instance);
        check_partitioned(&airs.transcript_prefix, &traces.transcript_prefix);
        check_partitioned(&airs.beta, &traces.beta);
        check_partitioned(&airs.decomposition, &traces.decomposition);
    }

    #[test]
    fn beta_claim_matches_direct_eq_reference_for_nonzero_equation() {
        let profile = profile();
        let (mut instance, _) = instance_and_proof(&profile);
        instance.beta[profile.beta_power_explicit_start + 2] += ef4(83);
        let claim = reference_beta_claim(&profile, &instance.beta).expect("nonzero beta claim");
        let tau = &instance.beta[..profile.log_constraints];
        let one = instance.beta[profile.one_explicit_coordinate];
        let beta = instance.beta[profile.beta_explicit_coordinate];
        let expected = (instance.beta[profile.beta_power_explicit_start + 2] * one
            - instance.beta[profile.beta_power_explicit_start + 1] * beta)
            * scale_by_power(
                EF::ONE,
                one,
                profile.exact_relation_degree.saturating_sub(2),
            )
            * eval_eq_index_checked(tau, profile.beta_constraint_start + 2).expect("Eq weight");
        assert_eq!(claim, expected);
        assert_ne!(claim, EF::ZERO);
    }

    #[test]
    fn beta_claim_order_and_transcript_mutations_are_rejected() {
        let (airs, mut instance, proof, transcript, start_tidx) = airs_and_witness();
        instance.beta[0] += EF::ONE;
        assert!(matches!(
            generate(&airs, &instance, &proof, &transcript, start_tidx),
            Err(FixedMultiAirCompletePrefixError::Transcript(_))
        ));

        let (airs, instance, mut proof, transcript, start_tidx) = airs_and_witness();
        proof.local_claims.swap(0, 1);
        assert!(matches!(
            generate(&airs, &instance, &proof, &transcript, start_tidx),
            Err(FixedMultiAirCompletePrefixError::Transcript(_))
        ));

        let (airs, instance, proof, mut transcript, start_tidx) = airs_and_witness();
        transcript.values_mut()[start_tidx] += F::ONE;
        assert!(matches!(
            generate(&airs, &instance, &proof, &transcript, start_tidx),
            Err(FixedMultiAirCompletePrefixError::Transcript(_))
        ));
    }

    #[test]
    fn root_claim_and_malformed_lengths_fail_without_panicking() {
        let (airs, instance, proof, transcript, start_tidx) = airs_and_witness();
        let wrong_root = digest(201);
        assert!(matches!(
            airs.generate_traces(
                FixedMultiAirCompletePrefixWitness {
                    authenticated_root: wrong_root,
                    instance: &instance,
                    proof: &proof,
                    transcript: &transcript,
                    statement_start_tidx: start_tidx,
                },
                FixedMultiAirCompletePrefixRequiredHeights::default(),
            ),
            Err(FixedMultiAirCompletePrefixError::Root)
        ));

        let mut malformed = instance.clone();
        malformed.beta.pop();
        let result = catch_unwind(AssertUnwindSafe(|| {
            generate(&airs, &malformed, &proof, &transcript, start_tidx)
        }));
        assert!(matches!(
            result,
            Ok(Err(FixedMultiAirCompletePrefixError::Shape(_)))
        ));

        let mut forged = proof.clone();
        forged.local_claims[0] += EF::ONE;
        assert!(matches!(
            generate(&airs, &instance, &forged, &transcript, start_tidx),
            Err(FixedMultiAirCompletePrefixError::Claim)
                | Err(FixedMultiAirCompletePrefixError::Transcript(_))
        ));
    }

    #[test]
    fn empty_interaction_claim_and_presence_are_setup_forced() {
        let (airs, instance, mut proof, transcript, start_tidx) = airs_and_witness();
        proof.interaction_claims[0] = EF::ONE;
        assert!(matches!(
            generate(&airs, &instance, &proof, &transcript, start_tidx),
            Err(FixedMultiAirCompletePrefixError::Claim)
        ));

        let (airs, instance, mut proof, transcript, start_tidx) = airs_and_witness();
        proof.interaction_proofs[0] = proof.interaction_proofs[1].clone();
        assert!(matches!(
            generate(&airs, &instance, &proof, &transcript, start_tidx),
            Err(FixedMultiAirCompletePrefixError::InteractionPresence { region: 0 })
        ));

        let (airs, instance, mut proof, transcript, start_tidx) = airs_and_witness();
        proof.interaction_proofs[1] = None;
        assert!(matches!(
            generate(&airs, &instance, &proof, &transcript, start_tidx),
            Err(FixedMultiAirCompletePrefixError::InteractionPresence { region: 1 })
        ));
    }

    #[test]
    fn proof_presence_trace_mutation_breaks_constraints() {
        let (airs, instance, proof, transcript, start_tidx) = airs_and_witness();
        let mut traces = generate(&airs, &instance, &proof, &transcript, start_tidx)
            .expect("honest complete prefix traces");
        let common_width = traces.transcript_prefix.common.width();
        let empty_row = airs
            .instance
            .profile
            .transcript_schedule
            .iter()
            .position(|entry| matches!(entry.source, PrefixSource::InteractionClaim(0)))
            .expect("empty interaction transcript row");
        let cols: &mut FixedMultiAirCompletePrefixCols<F> = traces.transcript_prefix.common.values
            [empty_row * common_width..(empty_row + 1) * common_width]
            .borrow_mut();
        cols.interaction_proof_present = F::ONE;
        let rejected = catch_unwind(AssertUnwindSafe(|| {
            check_partitioned(&airs.transcript_prefix, &traces.transcript_prefix)
        }));
        assert!(rejected.is_err(), "forged proof presence was accepted");
    }

    #[test]
    fn beta_eq_and_running_trace_mutations_break_constraints() {
        let (airs, instance, proof, transcript, start_tidx) = airs_and_witness();
        let mut traces = generate(&airs, &instance, &proof, &transcript, start_tidx)
            .expect("honest complete prefix traces");
        let width = traces.beta.common.width();
        let cols: &mut FixedMultiAirCompleteBetaCols<F> =
            traces.beta.common.values[..width].borrow_mut();
        cols.eq_after[0] += F::ONE;
        let rejected = catch_unwind(AssertUnwindSafe(|| {
            check_partitioned(&airs.beta, &traces.beta)
        }));
        assert!(rejected.is_err(), "forged Eq recurrence was accepted");

        let mut traces = generate(&airs, &instance, &proof, &transcript, start_tidx)
            .expect("honest complete prefix traces");
        let last_row = airs.instance.profile.beta_row_count() - 1;
        let cols: &mut FixedMultiAirCompleteBetaCols<F> =
            traces.beta.common.values[last_row * width..(last_row + 1) * width].borrow_mut();
        cols.running_after[1] += F::ONE;
        let rejected = catch_unwind(AssertUnwindSafe(|| {
            check_partitioned(&airs.beta, &traces.beta)
        }));
        assert!(rejected.is_err(), "forged beta running claim was accepted");
    }

    #[test]
    fn exactly_one_sequence_start_and_typed_claim_catalog_order_are_fixed() {
        let (airs, instance, proof, transcript, start_tidx) = airs_and_witness();
        let traces = generate(&airs, &instance, &proof, &transcript, start_tidx)
            .expect("honest complete prefix traces");
        let cached_width = traces.transcript_prefix.cached.width();
        let mut sequence_start_rows = 0usize;
        let mut local_regions = Vec::new();
        let mut interaction_regions = Vec::new();
        for row in 0..airs.instance.profile.transcript_observation_count() {
            let schedule: &FixedMultiAirCompletePrefixScheduleCols<F> =
                traces.transcript_prefix.cached.values
                    [row * cached_width..(row + 1) * cached_width]
                    .borrow();
            sequence_start_rows += usize::from(schedule.is_last == F::ONE);
            if schedule.source_flags[PREFIX_SOURCE_LOCAL_CLAIM] == F::ONE {
                local_regions.push(schedule.source_index.as_canonical_u32() as usize);
            }
            if schedule.source_flags[PREFIX_SOURCE_INTERACTION_CLAIM] == F::ONE {
                interaction_regions.push(schedule.source_index.as_canonical_u32() as usize);
            }
        }
        assert_eq!(sequence_start_rows, 1);
        assert_eq!(local_regions, vec![0, 1]);
        assert_eq!(interaction_regions, vec![0, 1]);
        assert_eq!(airs.instance.profile.protocol_component_count, 3);
    }

    #[test]
    fn transcript_schedule_uses_exact_backend_tags_and_order() {
        let profile = profile();
        let (instance, proof) = instance_and_proof(&profile);
        let values = profile
            .transcript_schedule
            .iter()
            .map(|entry| prefix_source_value(entry.source, &instance, &proof).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(values[0], EF::from_u64(COMPLETE_TERMINAL_STATEMENT_TAG));
        assert_eq!(
            values[1],
            EF::from_u64(u64::from(FIXED_MULTI_AIR_COMPLETE_TERMINAL_VERSION))
        );
        let decomposition_tag = values
            .iter()
            .position(|&value| value == EF::from_u64(COMPLETE_TERMINAL_DECOMPOSITION_TAG))
            .expect("decomposition tag");
        let expected_statement_len =
            1 + 1 + 1 + profile.metadata_bytes.len() + 1 + profile.beta_len + 1;
        assert_eq!(decomposition_tag, expected_statement_len);
        assert_eq!(
            &values[decomposition_tag + 2..decomposition_tag + 2 + profile.region_count],
            proof.local_claims.as_slice()
        );
        assert_eq!(
            &values[decomposition_tag + 3 + profile.region_count..],
            proof.interaction_claims.as_slice()
        );
    }

    #[test]
    fn lookup_catalog_multiplicities_balance_module_and_cursor_consumers() {
        let profile = profile();
        let external = FixedMultiAirCompleteExternalLookupCounts::semantic_minimum(
            profile.alpha_len,
            profile.beta_len,
            profile.region_count,
        );
        for (coordinate, &producer) in profile.instance_lookup_counts[INSTANCE_SECTION_ALPHA]
            .iter()
            .enumerate()
        {
            assert_eq!(producer, external.alpha[coordinate]);
        }
        assert_eq!(
            profile.instance_lookup_counts[INSTANCE_SECTION_MU][0],
            external.mu
        );

        let mut expected_beta = external.beta.clone();
        for count in &mut expected_beta {
            *count += 1; // exact statement transcript
        }
        expected_beta[profile.one_explicit_coordinate] += profile.beta_power_count as u32;
        expected_beta[profile.beta_explicit_coordinate] += profile.beta_power_count as u32;
        for coordinate in 0..profile.log_constraints {
            expected_beta[coordinate] += profile.beta_power_count as u32;
        }
        for power in 0..profile.beta_power_count {
            expected_beta[profile.beta_power_explicit_start + power] += 1;
            if power != 0 {
                expected_beta[profile.beta_power_explicit_start + power - 1] += 1;
            }
        }
        assert_eq!(
            profile.instance_lookup_counts[INSTANCE_SECTION_BETA],
            expected_beta
        );
        assert_eq!(
            profile.instance_lookup_counts[INSTANCE_SECTION_ETA][0],
            external.eta + 2
        );
        for region in 0..profile.region_count {
            assert_eq!(
                profile.local_claim_lookup_counts[region],
                external.local_claims[region] + 1
            );
            assert_eq!(
                profile.interaction_claim_lookup_counts[region],
                external.interaction_claims[region] + 1
            );
        }
    }

    #[test]
    fn metadata_geometry_and_field_overflow_are_rejected() {
        let mut bad = metadata();
        bad.regions[1].setup_ordinal = 0;
        let counts = FixedMultiAirCompleteExternalLookupCounts::semantic_minimum(6, 11, 2);
        assert!(matches!(
            FixedMultiAirCompletePrefixProfile::from_metadata(bad, counts.clone()),
            Err(FixedMultiAirCompletePrefixError::Metadata(_))
        ));

        let mut bad = metadata();
        bad.zero_constraint_tail.len = 3;
        assert!(matches!(
            FixedMultiAirCompletePrefixProfile::from_metadata(bad, counts.clone()),
            Err(FixedMultiAirCompletePrefixError::Metadata(_))
        ));

        let mut bad_counts = counts;
        bad_counts.binding = F::ORDER_U32;
        assert!(matches!(
            FixedMultiAirCompletePrefixProfile::from_metadata(metadata(), bad_counts),
            Err(FixedMultiAirCompletePrefixError::Overflow(_))
        ));

        let mut missing_cursor_owner =
            FixedMultiAirCompleteExternalLookupCounts::semantic_minimum(6, 11, 2);
        missing_cursor_owner.local_claims[0] = 0;
        assert!(matches!(
            FixedMultiAirCompletePrefixProfile::from_metadata(metadata(), missing_cursor_owner),
            Err(FixedMultiAirCompletePrefixError::Shape(_))
        ));
    }
}

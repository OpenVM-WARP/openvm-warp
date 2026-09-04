//! Canonical region cursor and dynamic-opening owner for the complete
//! fixed-multi-AIR terminal verifier.
//!
//! The native complete terminal transcript is deliberately not a collection
//! of independently restartable regional proofs.  It is one sequence:
//!
//! ```text
//! decomposition
//!   -> every local region
//!   -> every proof-present interaction region
//!   -> one global opening batch
//! ```
//!
//! This module makes that sequence an AIR invariant.  A setup-fixed sequence
//! bridge consumes the single decomposition endpoint and creates one typed
//! cursor.  Each component consumes that cursor, authenticates its catalogued
//! claim, hands the exact start index to the corresponding degree-seven
//! sumcheck, consumes the sumcheck endpoint, observes the opening tail, and
//! only then advances the cursor.  The final component emits the unique batch
//! start.  Empty interaction components consume and constrain their zero
//! claims in the sequence bridge, but never receive a cursor or a proof.
//!
//! Regional point authority remains owned by `sumcheck.rs`: the local and
//! interaction sumcheck AIRs publish their challenges on the distinct point
//! buses bundled by [`FixedMultiAirCompleteRegionalAuthorityBuses`].  This
//! module publishes the corresponding opening and final-evaluation
//! authorities.  Keeping all three buses in one setup bundle prevents a
//! circuit assembler from accidentally cross-routing component kinds.

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
    air_builders::{
        symbolic::{symbolic_variable::Entry, SymbolicExpressionNode},
        PartitionedAirBuilder,
    },
    interaction::InteractionBuilder,
    native_warp::{
        FixedMultiAirCompleteRegionSumcheckProof,
        FixedMultiAirCompleteTerminalCircuitComponentKind,
        FixedMultiAirCompleteTerminalCircuitNodeSource, FixedMultiAirCompleteTerminalCircuitPlan,
        FixedMultiAirCompleteTerminalProof, FixedMultiAirCompleteTerminalTranscriptStep,
        COMPLETE_TERMINAL_BATCH_TAG, COMPLETE_TERMINAL_INTERACTION_REGION_TAG,
        COMPLETE_TERMINAL_LOCAL_REGION_TAG, COMPLETE_TERMINAL_OPENINGS_TAG,
        COMPLETE_TERMINAL_ROUND_TAG, FIXED_MULTI_AIR_COMPLETE_TERMINAL_CIRCUIT_PLAN_VERSION,
        FIXED_MULTI_AIR_COMPLETE_TERMINAL_ROUND_DEGREE,
    },
    transcript::TranscriptLog,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField32};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    define_typed_lookup_bus, define_typed_permutation_bus,
    native_warp::terminal::fixed_multi_air_complete::{
        FixedMultiAirCompleteDecompositionReceiptBus,
        FixedMultiAirCompleteDecompositionReceiptMessage, FixedMultiAirCompleteInteractionClaimBus,
        FixedMultiAirCompleteInteractionRegionPointBus,
        FixedMultiAirCompleteInteractionRegionStartBus,
        FixedMultiAirCompleteInteractionRegionSumcheckFinalBus, FixedMultiAirCompleteLocalClaimBus,
        FixedMultiAirCompleteLocalRegionPointBus, FixedMultiAirCompleteLocalRegionStartBus,
        FixedMultiAirCompleteLocalRegionSumcheckFinalBus,
        FixedMultiAirCompleteRegionSequenceStartBus,
        FixedMultiAirCompleteRegionSequenceStartMessage, FixedMultiAirCompleteRegionStartMessage,
        FixedMultiAirCompleteRegionSumcheckFinalMessage, FixedMultiAirCompleteRegionalClaimMessage,
        FIXED_MULTI_AIR_COMPLETE_REGION_PREFIX_BASE_VALUES,
        FIXED_MULTI_AIR_COMPLETE_ROUND_TRANSCRIPT_BASE_VALUES,
        FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS,
    },
};

const TAIL_SOURCE_COUNT: usize = 2;
const TAIL_SOURCE_CONSTANT: usize = 0;
const TAIL_SOURCE_OPENING: usize = 1;

const _: () = assert!(D_EF == 4);
const _: () = assert!(FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS == 8);
const _: () = assert!(FIXED_MULTI_AIR_COMPLETE_TERMINAL_ROUND_DEGREE == 7);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedMultiAirCompleteRegionCursorOpeningError {
    Plan(&'static str),
    Shape(&'static str),
    Overflow(&'static str),
    Transcript(&'static str),
    Claim {
        component: FixedMultiAirCompleteTerminalCircuitComponentKind,
        region: usize,
        round: usize,
    },
    InteractionPresence {
        region: usize,
    },
    TraceHeight {
        component: Option<usize>,
    },
    Allocation,
}

impl fmt::Display for FixedMultiAirCompleteRegionCursorOpeningError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Plan(message) => write!(formatter, "complete region cursor plan: {message}"),
            Self::Shape(message) => write!(formatter, "complete region cursor shape: {message}"),
            Self::Overflow(message) => {
                write!(formatter, "complete region cursor overflow: {message}")
            }
            Self::Transcript(message) => {
                write!(formatter, "complete region cursor transcript: {message}")
            }
            Self::Claim {
                component,
                region,
                round,
            } => write!(
                formatter,
                "complete {component:?} region {region} sumcheck claim mismatch in round {round}"
            ),
            Self::InteractionPresence { region } => write!(
                formatter,
                "complete interaction proof presence mismatch in region {region}"
            ),
            Self::TraceHeight { component: None } => {
                formatter.write_str("complete region sequence trace height")
            }
            Self::TraceHeight {
                component: Some(component),
            } => write!(
                formatter,
                "complete region opening trace height for component {component}"
            ),
            Self::Allocation => formatter.write_str("complete region cursor allocation"),
        }
    }
}

impl std::error::Error for FixedMultiAirCompleteRegionCursorOpeningError {}

type CursorResult<T> = Result<T, FixedMultiAirCompleteRegionCursorOpeningError>;

/// The one-use cursor threaded through the setup-fixed regional sequence.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteRegionCursorMessage<T> {
    pub component: T,
    /// Index of this component's local/interaction domain tag.
    pub tidx: T,
    pub protocol_component_count: T,
    /// Number of dynamic openings belonging to all preceding components.
    pub global_opening_cursor: T,
}

define_typed_permutation_bus!(
    FixedMultiAirCompleteRegionCursorBus,
    FixedMultiAirCompleteRegionCursorMessage
);

/// One authenticated dynamic opening.  Component kind is fixed by the bus
/// type; both regional and global ordinals are setup-fixed.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteRegionOpeningMessage<T> {
    pub region: T,
    pub regional_opening: T,
    pub global_opening: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirCompleteLocalRegionOpeningBus,
    FixedMultiAirCompleteRegionOpeningMessage
);
define_typed_lookup_bus!(
    FixedMultiAirCompleteInteractionRegionOpeningBus,
    FixedMultiAirCompleteRegionOpeningMessage
);

/// Final value of one regional sumcheck at its sampled Boolean-cube point.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteRegionFinalEvaluationMessage<T> {
    pub region: T,
    pub claim: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirCompleteLocalRegionFinalEvaluationBus,
    FixedMultiAirCompleteRegionFinalEvaluationMessage
);
define_typed_permutation_bus!(
    FixedMultiAirCompleteInteractionRegionFinalEvaluationBus,
    FixedMultiAirCompleteRegionFinalEvaluationMessage
);

/// Exact handoff to the one global mapped-opening batch owner.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteOpeningBatchStartMessage<T> {
    /// Index of `COMPLETE_TERMINAL_BATCH_TAG`.
    pub tidx: T,
    pub protocol_component_count: T,
    pub global_opening_count: T,
}

define_typed_permutation_bus!(
    FixedMultiAirCompleteOpeningBatchStartBus,
    FixedMultiAirCompleteOpeningBatchStartMessage
);

/// Setup bundle shared by sumchecks, opening tails, endpoint owners, and the
/// global mapper.  Sumchecks publish points; opening tails publish openings
/// and final evaluations.
#[derive(Clone, Copy, Debug)]
pub struct FixedMultiAirCompleteRegionalAuthorityBuses {
    pub local_point: FixedMultiAirCompleteLocalRegionPointBus,
    pub interaction_point: FixedMultiAirCompleteInteractionRegionPointBus,
    pub local_opening: FixedMultiAirCompleteLocalRegionOpeningBus,
    pub interaction_opening: FixedMultiAirCompleteInteractionRegionOpeningBus,
    pub local_final_evaluation: FixedMultiAirCompleteLocalRegionFinalEvaluationBus,
    pub interaction_final_evaluation: FixedMultiAirCompleteInteractionRegionFinalEvaluationBus,
}

#[derive(Clone, Copy, Debug)]
pub struct FixedMultiAirCompleteRegionCursorOpeningBuses {
    pub transcript: TranscriptBus,
    pub sequence_start: FixedMultiAirCompleteRegionSequenceStartBus,
    pub decomposition_receipt: FixedMultiAirCompleteDecompositionReceiptBus,
    pub cursor: FixedMultiAirCompleteRegionCursorBus,
    pub local_claim: FixedMultiAirCompleteLocalClaimBus,
    pub interaction_claim: FixedMultiAirCompleteInteractionClaimBus,
    pub local_start: FixedMultiAirCompleteLocalRegionStartBus,
    pub interaction_start: FixedMultiAirCompleteInteractionRegionStartBus,
    pub local_sumcheck_final: FixedMultiAirCompleteLocalRegionSumcheckFinalBus,
    pub interaction_sumcheck_final: FixedMultiAirCompleteInteractionRegionSumcheckFinalBus,
    pub authorities: FixedMultiAirCompleteRegionalAuthorityBuses,
    pub batch_start: FixedMultiAirCompleteOpeningBatchStartBus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedMultiAirCompleteRegionCursorComponentProfile {
    pub kind: FixedMultiAirCompleteTerminalCircuitComponentKind,
    pub component_ordinal: usize,
    pub region: usize,
    pub claim_ordinal: usize,
    pub proof_ordinal: usize,
    pub round_count: usize,
    pub opening_ordinal_start: usize,
    pub opening_count: usize,
    /// Exact setup-derived fanout: expression variables and interaction-q
    /// users, plus the one global mapped-opening consumer.
    pub opening_lookup_counts: Vec<u32>,
    pub point_fixed_source_count: usize,
    pub point_common_count: usize,
    pub is_last: bool,
}

impl FixedMultiAirCompleteRegionCursorComponentProfile {
    fn region_tag(&self) -> u64 {
        match self.kind {
            FixedMultiAirCompleteTerminalCircuitComponentKind::Local => {
                COMPLETE_TERMINAL_LOCAL_REGION_TAG
            }
            FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction => {
                COMPLETE_TERMINAL_INTERACTION_REGION_TAG
            }
        }
    }

    fn opening_end(&self) -> CursorResult<usize> {
        self.opening_ordinal_start
            .checked_add(self.opening_count)
            .ok_or(FixedMultiAirCompleteRegionCursorOpeningError::Overflow(
                "component opening endpoint",
            ))
    }

    fn regional_transcript_span(&self) -> CursorResult<usize> {
        self.round_count
            .checked_mul(FIXED_MULTI_AIR_COMPLETE_ROUND_TRANSCRIPT_BASE_VALUES)
            .and_then(|rounds| {
                FIXED_MULTI_AIR_COMPLETE_REGION_PREFIX_BASE_VALUES.checked_add(rounds)
            })
            .ok_or(FixedMultiAirCompleteRegionCursorOpeningError::Overflow(
                "regional transcript span",
            ))
    }

    fn opening_tail_rows(&self) -> CursorResult<usize> {
        2usize.checked_add(self.opening_count).ok_or(
            FixedMultiAirCompleteRegionCursorOpeningError::Overflow("opening tail rows"),
        )
    }
}

/// Setup-fixed projection of the canonical backend plan needed by this owner.
#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteRegionCursorOpeningProfile {
    pub plan: Arc<FixedMultiAirCompleteTerminalCircuitPlan<F, Digest>>,
    pub components: Vec<Arc<FixedMultiAirCompleteRegionCursorComponentProfile>>,
    pub empty_interaction_regions: Vec<usize>,
    pub global_opening_count: usize,
}

impl FixedMultiAirCompleteRegionCursorOpeningProfile {
    pub fn from_plan(
        plan: Arc<FixedMultiAirCompleteTerminalCircuitPlan<F, Digest>>,
    ) -> CursorResult<Self> {
        validate_field_word(plan.regions.len(), "region count")?;
        if plan.version != FIXED_MULTI_AIR_COMPLETE_TERMINAL_CIRCUIT_PLAN_VERSION {
            return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                "circuit-plan version",
            ));
        }
        plan.canonical_bytes().map_err(|_| {
            FixedMultiAirCompleteRegionCursorOpeningError::Plan("canonical encoding")
        })?;
        if plan.regions.is_empty() {
            return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                "empty region inventory",
            ));
        }

        let mut components = Vec::new();
        let mut empty_interaction_regions = Vec::new();
        let mut opening_cursor = 0usize;

        for kind in [
            FixedMultiAirCompleteTerminalCircuitComponentKind::Local,
            FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction,
        ] {
            for (region_index, region) in plan.regions.iter().enumerate() {
                let setup_ordinal = checked_usize(region.setup_ordinal, "region setup ordinal")?;
                if setup_ordinal != region_index {
                    return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                        "region setup order",
                    ));
                }
                let component = match kind {
                    FixedMultiAirCompleteTerminalCircuitComponentKind::Local => &region.local,
                    FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction => {
                        &region.interaction
                    }
                };
                if component.identity.kind != kind
                    || checked_usize(
                        component.identity.region_ordinal,
                        "component region ordinal",
                    )? != region_index
                    || checked_usize(component.identity.claim_ordinal, "claim ordinal")?
                        != region_index
                {
                    return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                        "component identity",
                    ));
                }

                let proof_present = component.identity.proof_present;
                if kind == FixedMultiAirCompleteTerminalCircuitComponentKind::Local
                    && !proof_present
                {
                    return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                        "local proof omitted",
                    ));
                }
                if kind == FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction
                    && !proof_present
                {
                    if component.identity.proof_ordinal.is_some()
                        || component.identity.opening_count != 0
                        || !component.interactions.is_empty()
                    {
                        return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                            "empty interaction proof geometry",
                        ));
                    }
                    empty_interaction_regions.push(region_index);
                    continue;
                }

                let proof_ordinal = checked_usize(
                    component.identity.proof_ordinal.ok_or(
                        FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                            "proof-present component ordinal",
                        ),
                    )?,
                    "proof ordinal",
                )?;
                if proof_ordinal != region_index {
                    return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                        "native proof ordinal",
                    ));
                }
                let opening_start = checked_usize(
                    component.identity.opening_ordinal_start,
                    "opening ordinal start",
                )?;
                let opening_count =
                    checked_usize(component.identity.opening_count, "opening count")?;
                if opening_start != opening_cursor
                    || opening_count != component.expression.dynamic_columns.len()
                {
                    return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                        "component opening geometry",
                    ));
                }
                let round_count = usize::from(region.log_height);
                validate_field_word(round_count, "round count")?;
                validate_field_word(opening_start, "opening ordinal start")?;
                validate_field_word(opening_count, "opening count")?;
                let point_fixed_source_count = component.expression.fixed_columns.len();
                validate_field_word(point_fixed_source_count, "fixed point-source count")?;
                let opening_lookup_counts = opening_lookup_counts(component)?;
                // One endpoint owner consumes each challenge. The global
                // mapper additionally consumes the same regional point once
                // for every mapped dynamic opening of this component. Fixed
                // column folds are accounted for separately by the sumcheck
                // point multiplicity recurrence.
                let point_common_count = opening_count.checked_add(1).ok_or(
                    FixedMultiAirCompleteRegionCursorOpeningError::Overflow(
                        "common point-source count",
                    ),
                )?;
                validate_field_word(point_common_count, "common point-source count")?;
                let component_ordinal = components.len();
                validate_field_word(component_ordinal, "component ordinal")?;
                opening_cursor = opening_cursor.checked_add(opening_count).ok_or(
                    FixedMultiAirCompleteRegionCursorOpeningError::Overflow("global opening count"),
                )?;
                components.push(Arc::new(
                    FixedMultiAirCompleteRegionCursorComponentProfile {
                        kind,
                        component_ordinal,
                        region: region_index,
                        claim_ordinal: region_index,
                        proof_ordinal,
                        round_count,
                        opening_ordinal_start: opening_start,
                        opening_count,
                        opening_lookup_counts,
                        point_fixed_source_count,
                        point_common_count,
                        is_last: false,
                    },
                ));
            }
        }
        if components.is_empty() {
            return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                "empty component inventory",
            ));
        }
        if let Some(last) = components.last_mut() {
            Arc::make_mut(last).is_last = true;
        }
        validate_field_word(components.len(), "protocol component count")?;
        validate_field_word(opening_cursor, "global opening count")?;
        validate_mapped_openings(&plan, &components, opening_cursor)?;
        validate_transcript_plan(&plan, &components, opening_cursor)?;

        Ok(Self {
            plan,
            components,
            empty_interaction_regions,
            global_opening_count: opening_cursor,
        })
    }

    #[must_use]
    pub fn protocol_component_count(&self) -> usize {
        self.components.len()
    }

    #[must_use]
    pub fn sequence_row_count(&self) -> usize {
        // One row consumes each empty interaction claim and the final row
        // performs the unique sequence-start -> cursor bridge.
        self.empty_interaction_regions.len() + 1
    }
}

/// Derive the exact producer multiplicity for every dynamic opening from the
/// canonical component plan. Each opening is consumed once by the global
/// mapper, once per dynamic expression-variable node that names it, and once
/// per interaction `q` coordinate that names it.
fn opening_lookup_counts(
    component: &openvm_stark_backend::native_warp::FixedMultiAirCompleteTerminalCircuitComponentPlan<
        F,
    >,
) -> CursorResult<Vec<u32>> {
    let expression = &component.expression;
    if expression.nodes.len() != expression.node_sources.len()
        || checked_usize(component.identity.opening_count, "opening lookup count")?
            != expression.dynamic_columns.len()
    {
        return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
            "opening lookup geometry",
        ));
    }

    // The global mapped-opening owner consumes every dynamic opening exactly
    // once, independently of endpoint expression and interaction-q fanout.
    let mut counts = vec![1usize; expression.dynamic_columns.len()];
    for (node, source) in expression.nodes.iter().zip(&expression.node_sources) {
        match node {
            SymbolicExpressionNode::Variable(variable) => match variable.entry {
                Entry::Main { .. } | Entry::Preprocessed { .. } => match source {
                    Some(FixedMultiAirCompleteTerminalCircuitNodeSource::Dynamic(source)) => {
                        increment_opening_lookup_count(
                            &mut counts,
                            *source,
                            "dynamic expression opening",
                        )?;
                    }
                    Some(FixedMultiAirCompleteTerminalCircuitNodeSource::Fixed(_)) => {}
                    None => {
                        return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                            "missing expression node source",
                        ));
                    }
                },
                Entry::Public if source.is_none() => {}
                Entry::Public => {
                    return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                        "public expression node source",
                    ));
                }
                Entry::Challenge => {
                    return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                        "challenge expression node",
                    ));
                }
            },
            _ if source.is_some() => {
                return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                    "non-variable expression node source",
                ));
            }
            _ => {}
        }
    }
    for interaction in &component.interactions {
        for &source in &interaction.q_source_ordinals {
            increment_opening_lookup_count(&mut counts, source, "interaction q opening")?;
        }
    }

    counts
        .into_iter()
        .map(|count| {
            validate_field_word(count, "opening lookup multiplicity")?;
            u32::try_from(count).map_err(|_| {
                FixedMultiAirCompleteRegionCursorOpeningError::Overflow(
                    "opening lookup multiplicity",
                )
            })
        })
        .collect()
}

fn increment_opening_lookup_count(
    counts: &mut [usize],
    source: u32,
    context: &'static str,
) -> CursorResult<()> {
    let source = checked_usize(source, context)?;
    let count = counts
        .get_mut(source)
        .ok_or(FixedMultiAirCompleteRegionCursorOpeningError::Plan(context))?;
    *count =
        count
            .checked_add(1)
            .ok_or(FixedMultiAirCompleteRegionCursorOpeningError::Overflow(
                context,
            ))?;
    Ok(())
}

fn validate_mapped_openings(
    plan: &FixedMultiAirCompleteTerminalCircuitPlan<F, Digest>,
    components: &[Arc<FixedMultiAirCompleteRegionCursorComponentProfile>],
    opening_count: usize,
) -> CursorResult<()> {
    if plan.mapped_openings.len() != opening_count {
        return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
            "mapped opening coverage",
        ));
    }
    for component in components {
        let planned = match component.kind {
            FixedMultiAirCompleteTerminalCircuitComponentKind::Local => {
                &plan.regions[component.region].local
            }
            FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction => {
                &plan.regions[component.region].interaction
            }
        };
        for regional_opening in 0..component.opening_count {
            let global_opening = component
                .opening_ordinal_start
                .checked_add(regional_opening)
                .ok_or(FixedMultiAirCompleteRegionCursorOpeningError::Overflow(
                    "mapped opening ordinal",
                ))?;
            let mapped = plan.mapped_openings.get(global_opening).ok_or(
                FixedMultiAirCompleteRegionCursorOpeningError::Plan("mapped opening ordinal"),
            )?;
            if mapped.component != component.kind
                || checked_usize(mapped.region_ordinal, "mapped region ordinal")?
                    != component.region
                || checked_usize(
                    mapped.regional_opening_ordinal,
                    "mapped regional opening ordinal",
                )? != regional_opening
                || checked_usize(
                    mapped.global_opening_ordinal,
                    "mapped global opening ordinal",
                )? != global_opening
                || checked_usize(mapped.rho_ordinal, "mapped rho ordinal")? != global_opening
                || planned.expression.dynamic_columns.get(regional_opening) != Some(&mapped.source)
            {
                return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                    "mapped opening order",
                ));
            }
        }
    }
    Ok(())
}

fn validate_transcript_plan(
    plan: &FixedMultiAirCompleteTerminalCircuitPlan<F, Digest>,
    components: &[Arc<FixedMultiAirCompleteRegionCursorComponentProfile>],
    opening_count: usize,
) -> CursorResult<()> {
    let mut step = 0usize;
    match plan.transcript.get(step) {
        Some(FixedMultiAirCompleteTerminalTranscriptStep::ObserveStatement { .. }) => step += 1,
        _ => {
            return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                "statement transcript step",
            ))
        }
    }
    match plan.transcript.get(step) {
        Some(FixedMultiAirCompleteTerminalTranscriptStep::ObserveDecomposition { .. }) => step += 1,
        _ => {
            return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                "decomposition transcript step",
            ))
        }
    }
    let mut challenge_ordinal = 0usize;
    for component in components {
        match plan.transcript.get(step) {
            Some(FixedMultiAirCompleteTerminalTranscriptStep::ObserveRegion {
                component: kind,
                region_ordinal,
                tag,
                observed_extension_elements,
            }) if *kind == component.kind
                && checked_usize(*region_ordinal, "transcript region ordinal")?
                    == component.region
                && *tag == component.region_tag()
                && *observed_extension_elements == 2 =>
            {
                step += 1;
            }
            _ => {
                return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                    "regional transcript order",
                ))
            }
        }
        for round in 0..component.round_count {
            match plan.transcript.get(step) {
                Some(FixedMultiAirCompleteTerminalTranscriptStep::ObserveRound {
                    component: kind,
                    region_ordinal,
                    round: planned_round,
                    tag,
                    evaluations,
                    observed_extension_elements,
                }) if *kind == component.kind
                    && checked_usize(*region_ordinal, "round region ordinal")?
                        == component.region
                    && usize::from(*planned_round) == round
                    && *tag == COMPLETE_TERMINAL_ROUND_TAG
                    && usize::from(*evaluations)
                        == FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS
                    && *observed_extension_elements
                        == u64::try_from(3 + FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS)
                            .map_err(|_| {
                                FixedMultiAirCompleteRegionCursorOpeningError::Overflow(
                                    "round observation count",
                                )
                            })? =>
                {
                    step += 1;
                }
                _ => {
                    return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                        "round transcript order",
                    ))
                }
            }
            match plan.transcript.get(step) {
                Some(FixedMultiAirCompleteTerminalTranscriptStep::SampleRoundChallenge {
                    component: kind,
                    region_ordinal,
                    round: planned_round,
                    challenge_ordinal: planned_challenge,
                }) if *kind == component.kind
                    && checked_usize(*region_ordinal, "challenge region ordinal")?
                        == component.region
                    && usize::from(*planned_round) == round
                    && checked_usize(*planned_challenge, "round challenge ordinal")?
                        == challenge_ordinal =>
                {
                    step += 1;
                    challenge_ordinal = challenge_ordinal.checked_add(1).ok_or(
                        FixedMultiAirCompleteRegionCursorOpeningError::Overflow(
                            "round challenge count",
                        ),
                    )?;
                }
                _ => {
                    return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                        "round challenge order",
                    ))
                }
            }
        }
        match plan.transcript.get(step) {
            Some(FixedMultiAirCompleteTerminalTranscriptStep::ObserveOpenings {
                component: kind,
                region_ordinal,
                tag,
                opening_count: planned_count,
                observed_extension_elements,
            }) if *kind == component.kind
                && checked_usize(*region_ordinal, "opening region ordinal")?
                    == component.region
                && *tag == COMPLETE_TERMINAL_OPENINGS_TAG
                && checked_usize(*planned_count, "transcript opening count")?
                    == component.opening_count
                && *observed_extension_elements
                    == u64::try_from(2 + component.opening_count).map_err(|_| {
                        FixedMultiAirCompleteRegionCursorOpeningError::Overflow(
                            "opening observation count",
                        )
                    })? =>
            {
                step += 1;
            }
            _ => {
                return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                    "opening transcript order",
                ))
            }
        }
    }
    match plan.transcript.get(step) {
        Some(FixedMultiAirCompleteTerminalTranscriptStep::ObserveBatch {
            tag,
            component_count,
            observed_extension_elements,
        }) if *tag == COMPLETE_TERMINAL_BATCH_TAG
            && checked_usize(*component_count, "batch opening count")? == opening_count
            && *observed_extension_elements == 2 =>
        {
            step += 1;
        }
        _ => {
            return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                "batch transcript step",
            ))
        }
    }
    match plan.transcript.get(step) {
        Some(FixedMultiAirCompleteTerminalTranscriptStep::SampleGlobalRho {
            challenge_ordinal: planned_challenge,
        }) if checked_usize(*planned_challenge, "global rho ordinal")? == challenge_ordinal => {
            step += 1;
        }
        _ => {
            return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                "global rho transcript step",
            ))
        }
    }
    if step != plan.transcript.len()
        || checked_usize(plan.round_challenge_count, "round challenge count")? != challenge_ordinal
        || checked_usize(
            plan.global_rho_challenge_ordinal,
            "global rho challenge ordinal",
        )? != challenge_ordinal
    {
        return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
            "transcript plan coverage",
        ));
    }
    Ok(())
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteRegionSequenceScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub is_empty_interaction: T,
    pub region: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteRegionSequenceCols<T> {
    pub decomposition_end_tidx: T,
    pub eta: [T; D_EF],
    pub beta_claim: [T; D_EF],
    pub empty_interaction_claim: [T; D_EF],
}

/// Consumes the one decomposition endpoint, constrains every omitted
/// interaction claim to zero, and creates exactly one regional cursor.
pub struct FixedMultiAirCompleteRegionSequenceAir {
    pub profile: Arc<FixedMultiAirCompleteRegionCursorOpeningProfile>,
    pub sequence_start_bus: FixedMultiAirCompleteRegionSequenceStartBus,
    pub decomposition_receipt_bus: FixedMultiAirCompleteDecompositionReceiptBus,
    pub interaction_claim_bus: FixedMultiAirCompleteInteractionClaimBus,
    pub cursor_bus: FixedMultiAirCompleteRegionCursorBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteRegionSequenceAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteRegionSequenceAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirCompleteRegionSequenceScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteRegionSequenceCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteRegionSequenceAir {}
impl BaseAir<F> for FixedMultiAirCompleteRegionSequenceAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteRegionSequenceScheduleCols::<F>::width()
            + FixedMultiAirCompleteRegionSequenceCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteRegionSequenceAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let cached_row = builder.cached_mains()[0]
            .row_slice(0)
            .expect("complete region sequence cached row")
            .to_vec();
        let next_cached_row = builder.cached_mains()[0]
            .row_slice(1)
            .expect("complete region sequence next cached row")
            .to_vec();
        let common_row = builder
            .common_main()
            .row_slice(0)
            .expect("complete region sequence row")
            .to_vec();
        let next_common_row = builder
            .common_main()
            .row_slice(1)
            .expect("complete region sequence next row")
            .to_vec();
        let schedule: &FixedMultiAirCompleteRegionSequenceScheduleCols<AB::Var> =
            cached_row.as_slice().borrow();
        let next_schedule: &FixedMultiAirCompleteRegionSequenceScheduleCols<AB::Var> =
            next_cached_row.as_slice().borrow();
        let local: &FixedMultiAirCompleteRegionSequenceCols<AB::Var> =
            common_row.as_slice().borrow();
        let next: &FixedMultiAirCompleteRegionSequenceCols<AB::Var> =
            next_common_row.as_slice().borrow();

        for flag in [
            schedule.active,
            schedule.is_first,
            schedule.is_last,
            schedule.is_empty_interaction,
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
        builder.when(schedule.active).assert_eq(
            schedule.is_last,
            AB::Expr::ONE - schedule.is_empty_interaction,
        );

        let mut transition_builder = builder.when_transition();
        let mut transition = transition_builder.when(next_schedule.active);
        transition.assert_eq(next.decomposition_end_tidx, local.decomposition_end_tidx);

        assert_array_eq(
            &mut builder.when(schedule.is_empty_interaction),
            local.empty_interaction_claim,
            [AB::Expr::ZERO; D_EF],
        );
        self.interaction_claim_bus.lookup_key(
            builder,
            FixedMultiAirCompleteRegionalClaimMessage {
                region: schedule.region.into(),
                claim: local.empty_interaction_claim.map(Into::into),
            },
            schedule.is_empty_interaction,
        );

        self.sequence_start_bus.receive(
            builder,
            FixedMultiAirCompleteRegionSequenceStartMessage {
                decomposition_end_tidx: local.decomposition_end_tidx.into(),
                protocol_component_count: AB::Expr::from_usize(
                    self.profile.protocol_component_count(),
                ),
            },
            schedule.is_last,
        );
        self.decomposition_receipt_bus.receive(
            builder,
            FixedMultiAirCompleteDecompositionReceiptMessage {
                eta: local.eta.map(Into::into),
                beta_claim: local.beta_claim.map(Into::into),
                local_count: AB::Expr::from_usize(self.profile.plan.regions.len()),
                interaction_count: AB::Expr::from_usize(self.profile.plan.regions.len()),
            },
            schedule.is_last,
        );
        self.cursor_bus.send(
            builder,
            FixedMultiAirCompleteRegionCursorMessage {
                component: AB::Expr::ZERO,
                tidx: local.decomposition_end_tidx.into(),
                protocol_component_count: AB::Expr::from_usize(
                    self.profile.protocol_component_count(),
                ),
                global_opening_cursor: AB::Expr::ZERO,
            },
            schedule.is_last,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteRegionOpeningScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub source_flags: [T; TAIL_SOURCE_COUNT],
    pub source_index: T,
    pub constant: [T; D_EF],
    pub opening_lookup_count: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteRegionOpeningCols<T> {
    /// Cursor position at the component's region-domain tag.
    pub component_start_tidx: T,
    /// Position of this opening-tail observation.
    pub tidx: T,
    pub value: [T; D_EF],
    pub initial_claim: [T; D_EF],
    pub final_claim: [T; D_EF],
}

#[derive(Clone, Copy)]
enum ComponentBuses {
    Local {
        claim: FixedMultiAirCompleteLocalClaimBus,
        start: FixedMultiAirCompleteLocalRegionStartBus,
        sumcheck_final: FixedMultiAirCompleteLocalRegionSumcheckFinalBus,
        opening: FixedMultiAirCompleteLocalRegionOpeningBus,
        final_evaluation: FixedMultiAirCompleteLocalRegionFinalEvaluationBus,
    },
    Interaction {
        claim: FixedMultiAirCompleteInteractionClaimBus,
        start: FixedMultiAirCompleteInteractionRegionStartBus,
        sumcheck_final: FixedMultiAirCompleteInteractionRegionSumcheckFinalBus,
        opening: FixedMultiAirCompleteInteractionRegionOpeningBus,
        final_evaluation: FixedMultiAirCompleteInteractionRegionFinalEvaluationBus,
    },
}

impl ComponentBuses {
    fn lookup_claim<AB: InteractionBuilder>(
        self,
        builder: &mut AB,
        message: FixedMultiAirCompleteRegionalClaimMessage<impl Into<AB::Expr> + Clone>,
        enabled: impl Into<AB::Expr>,
    ) {
        match self {
            Self::Local { claim, .. } => claim.lookup_key(builder, message, enabled),
            Self::Interaction { claim, .. } => claim.lookup_key(builder, message, enabled),
        }
    }

    fn send_start<AB: InteractionBuilder>(
        self,
        builder: &mut AB,
        message: FixedMultiAirCompleteRegionStartMessage<impl Into<AB::Expr> + Clone>,
        enabled: impl Into<AB::Expr>,
    ) {
        match self {
            Self::Local { start, .. } => start.send(builder, message, enabled),
            Self::Interaction { start, .. } => start.send(builder, message, enabled),
        }
    }

    fn receive_sumcheck_final<AB: InteractionBuilder>(
        self,
        builder: &mut AB,
        message: FixedMultiAirCompleteRegionSumcheckFinalMessage<impl Into<AB::Expr> + Clone>,
        enabled: impl Into<AB::Expr>,
    ) {
        match self {
            Self::Local { sumcheck_final, .. } => sumcheck_final.receive(builder, message, enabled),
            Self::Interaction { sumcheck_final, .. } => {
                sumcheck_final.receive(builder, message, enabled)
            }
        }
    }

    fn publish_opening<AB: InteractionBuilder>(
        self,
        builder: &mut AB,
        message: FixedMultiAirCompleteRegionOpeningMessage<impl Into<AB::Expr> + Clone>,
        count: impl Into<AB::Expr>,
    ) {
        match self {
            Self::Local { opening, .. } => opening.add_key_with_lookups(builder, message, count),
            Self::Interaction { opening, .. } => {
                opening.add_key_with_lookups(builder, message, count)
            }
        }
    }

    fn send_final_evaluation<AB: InteractionBuilder>(
        self,
        builder: &mut AB,
        message: FixedMultiAirCompleteRegionFinalEvaluationMessage<impl Into<AB::Expr> + Clone>,
        enabled: impl Into<AB::Expr>,
    ) {
        match self {
            Self::Local {
                final_evaluation, ..
            } => final_evaluation.send(builder, message, enabled),
            Self::Interaction {
                final_evaluation, ..
            } => final_evaluation.send(builder, message, enabled),
        }
    }
}

/// Opening-tail owner for one setup-fixed local or interaction component.
/// The component kind is selected in the verifying key, never by a witness.
pub struct FixedMultiAirCompleteRegionOpeningAir {
    pub profile: Arc<FixedMultiAirCompleteRegionCursorComponentProfile>,
    pub transcript_bus: TranscriptBus,
    pub cursor_bus: FixedMultiAirCompleteRegionCursorBus,
    buses: ComponentBuses,
    pub batch_start_bus: FixedMultiAirCompleteOpeningBatchStartBus,
    pub protocol_component_count: usize,
    pub global_opening_count: usize,
}

impl FixedMultiAirCompleteRegionOpeningAir {
    fn new(
        profile: Arc<FixedMultiAirCompleteRegionCursorComponentProfile>,
        buses: FixedMultiAirCompleteRegionCursorOpeningBuses,
        protocol_component_count: usize,
        global_opening_count: usize,
    ) -> Self {
        let component_buses = match profile.kind {
            FixedMultiAirCompleteTerminalCircuitComponentKind::Local => ComponentBuses::Local {
                claim: buses.local_claim,
                start: buses.local_start,
                sumcheck_final: buses.local_sumcheck_final,
                opening: buses.authorities.local_opening,
                final_evaluation: buses.authorities.local_final_evaluation,
            },
            FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction => {
                ComponentBuses::Interaction {
                    claim: buses.interaction_claim,
                    start: buses.interaction_start,
                    sumcheck_final: buses.interaction_sumcheck_final,
                    opening: buses.authorities.interaction_opening,
                    final_evaluation: buses.authorities.interaction_final_evaluation,
                }
            }
        };
        Self {
            profile,
            transcript_bus: buses.transcript,
            cursor_bus: buses.cursor,
            buses: component_buses,
            batch_start_bus: buses.batch_start,
            protocol_component_count,
            global_opening_count,
        }
    }
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteRegionOpeningAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteRegionOpeningAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirCompleteRegionOpeningScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteRegionOpeningCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteRegionOpeningAir {}
impl BaseAir<F> for FixedMultiAirCompleteRegionOpeningAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteRegionOpeningScheduleCols::<F>::width()
            + FixedMultiAirCompleteRegionOpeningCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteRegionOpeningAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let cached_row = builder.cached_mains()[0]
            .row_slice(0)
            .expect("complete opening-tail cached row")
            .to_vec();
        let next_cached_row = builder.cached_mains()[0]
            .row_slice(1)
            .expect("complete opening-tail next cached row")
            .to_vec();
        let common_row = builder
            .common_main()
            .row_slice(0)
            .expect("complete opening-tail row")
            .to_vec();
        let next_common_row = builder
            .common_main()
            .row_slice(1)
            .expect("complete opening-tail next row")
            .to_vec();
        let schedule: &FixedMultiAirCompleteRegionOpeningScheduleCols<AB::Var> =
            cached_row.as_slice().borrow();
        let next_schedule: &FixedMultiAirCompleteRegionOpeningScheduleCols<AB::Var> =
            next_cached_row.as_slice().borrow();
        let local: &FixedMultiAirCompleteRegionOpeningCols<AB::Var> =
            common_row.as_slice().borrow();
        let next: &FixedMultiAirCompleteRegionOpeningCols<AB::Var> =
            next_common_row.as_slice().borrow();

        for flag in [schedule.active, schedule.is_first, schedule.is_last]
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
        transition.assert_eq(next.component_start_tidx, local.component_start_tidx);
        transition.assert_eq(next.tidx, local.tidx + AB::Expr::from_usize(D_EF));
        assert_array_eq(
            &mut transition,
            next.initial_claim,
            local.initial_claim.map(Into::into),
        );
        assert_array_eq(
            &mut transition,
            next.final_claim,
            local.final_claim.map(Into::into),
        );

        assert_array_eq(
            &mut builder.when(schedule.source_flags[TAIL_SOURCE_CONSTANT]),
            local.value,
            schedule.constant.map(Into::into),
        );
        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            local.tidx,
            local.value,
            schedule.active,
        );

        self.cursor_bus.receive(
            builder,
            FixedMultiAirCompleteRegionCursorMessage {
                component: AB::Expr::from_usize(self.profile.component_ordinal),
                tidx: local.component_start_tidx.into(),
                protocol_component_count: AB::Expr::from_usize(self.protocol_component_count),
                global_opening_cursor: AB::Expr::from_usize(self.profile.opening_ordinal_start),
            },
            schedule.is_first,
        );
        self.buses.lookup_claim(
            builder,
            FixedMultiAirCompleteRegionalClaimMessage {
                region: AB::Expr::from_usize(self.profile.claim_ordinal),
                claim: local.initial_claim.map(Into::into),
            },
            schedule.is_first,
        );
        self.buses.send_start(
            builder,
            FixedMultiAirCompleteRegionStartMessage {
                region: AB::Expr::from_usize(self.profile.region),
                tidx: local.component_start_tidx.into(),
                claim: local.initial_claim.map(Into::into),
            },
            schedule.is_first,
        );
        self.buses.receive_sumcheck_final(
            builder,
            FixedMultiAirCompleteRegionSumcheckFinalMessage {
                region: AB::Expr::from_usize(self.profile.region),
                tidx: local.tidx.into(),
                claim: local.final_claim.map(Into::into),
            },
            schedule.is_first,
        );

        self.buses.publish_opening(
            builder,
            FixedMultiAirCompleteRegionOpeningMessage {
                region: AB::Expr::from_usize(self.profile.region),
                regional_opening: schedule.source_index.into(),
                global_opening: AB::Expr::from_usize(self.profile.opening_ordinal_start)
                    + AB::Expr::from(schedule.source_index),
                value: local.value.map(Into::into),
            },
            schedule.opening_lookup_count,
        );
        self.buses.send_final_evaluation(
            builder,
            FixedMultiAirCompleteRegionFinalEvaluationMessage {
                region: AB::Expr::from_usize(self.profile.region),
                claim: local.final_claim.map(Into::into),
            },
            schedule.is_last,
        );

        let end_tidx = AB::Expr::from(local.tidx) + AB::Expr::from_usize(D_EF);
        if self.profile.is_last {
            self.batch_start_bus.send(
                builder,
                FixedMultiAirCompleteOpeningBatchStartMessage {
                    tidx: end_tidx,
                    protocol_component_count: AB::Expr::from_usize(self.protocol_component_count),
                    global_opening_count: AB::Expr::from_usize(self.global_opening_count),
                },
                schedule.is_last,
            );
        } else {
            self.cursor_bus.send(
                builder,
                FixedMultiAirCompleteRegionCursorMessage {
                    component: AB::Expr::from_usize(self.profile.component_ordinal + 1),
                    tidx: end_tidx,
                    protocol_component_count: AB::Expr::from_usize(self.protocol_component_count),
                    global_opening_cursor: AB::Expr::from_usize(
                        self.profile
                            .opening_end()
                            .expect("setup-validated component opening endpoint"),
                    ),
                },
                schedule.is_last,
            );
        }
    }
}

/// AIR inventory in exact native component order.  Parent integration must
/// interleave each entry with the matching local/interaction sumcheck AIR
/// using the same start, point, and final buses.
pub struct FixedMultiAirCompleteRegionCursorOpeningAirs {
    pub profile: Arc<FixedMultiAirCompleteRegionCursorOpeningProfile>,
    pub sequence: FixedMultiAirCompleteRegionSequenceAir,
    pub components: Vec<FixedMultiAirCompleteRegionOpeningAir>,
}

impl FixedMultiAirCompleteRegionCursorOpeningAirs {
    pub fn new(
        profile: Arc<FixedMultiAirCompleteRegionCursorOpeningProfile>,
        buses: FixedMultiAirCompleteRegionCursorOpeningBuses,
    ) -> CursorResult<Self> {
        if profile.components.is_empty() {
            return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                "empty component inventory",
            ));
        }
        let sequence = FixedMultiAirCompleteRegionSequenceAir {
            profile: profile.clone(),
            sequence_start_bus: buses.sequence_start,
            decomposition_receipt_bus: buses.decomposition_receipt,
            interaction_claim_bus: buses.interaction_claim,
            cursor_bus: buses.cursor,
        };
        let components = profile
            .components
            .iter()
            .cloned()
            .map(|component| {
                FixedMultiAirCompleteRegionOpeningAir::new(
                    component,
                    buses,
                    profile.protocol_component_count(),
                    profile.global_opening_count,
                )
            })
            .collect();
        Ok(Self {
            profile,
            sequence,
            components,
        })
    }

    pub fn generate_traces(
        &self,
        proof: &FixedMultiAirCompleteTerminalProof<EF>,
        transcript: &TranscriptLog<F, [F; 16]>,
        decomposition_end_tidx: usize,
        eta: EF,
        beta_claim: EF,
        required: FixedMultiAirCompleteRegionCursorOpeningRequiredHeights,
    ) -> CursorResult<FixedMultiAirCompleteRegionCursorOpeningTraceData> {
        validate_complete_proof_shape(&self.profile, proof)?;
        if transcript.values().len() != transcript.samples().len() {
            return Err(FixedMultiAirCompleteRegionCursorOpeningError::Transcript(
                "value/sample length mismatch",
            ));
        }
        validate_transcript_index(decomposition_end_tidx, "decomposition endpoint")?;
        if required.components.len() != self.components.len() {
            return Err(FixedMultiAirCompleteRegionCursorOpeningError::Shape(
                "required component heights",
            ));
        }

        let sequence = generate_sequence_trace(
            &self.profile,
            proof,
            decomposition_end_tidx,
            eta,
            beta_claim,
            required.sequence,
        )?;
        let mut component_traces = Vec::new();
        component_traces
            .try_reserve_exact(self.components.len())
            .map_err(|_| FixedMultiAirCompleteRegionCursorOpeningError::Allocation)?;
        let mut boundaries = Vec::new();
        boundaries
            .try_reserve_exact(self.components.len())
            .map_err(|_| FixedMultiAirCompleteRegionCursorOpeningError::Allocation)?;
        let mut cursor = decomposition_end_tidx;
        for ((air, &required_height), profile) in self
            .components
            .iter()
            .zip(&required.components)
            .zip(&self.profile.components)
        {
            let (claim, regional_proof) = component_proof(proof, profile)?;
            let (trace, boundary) = generate_component_opening_trace(
                air,
                claim,
                regional_proof,
                transcript,
                cursor,
                required_height,
            )?;
            if boundary.component_start_tidx != cursor {
                return Err(FixedMultiAirCompleteRegionCursorOpeningError::Transcript(
                    "component cursor chain",
                ));
            }
            cursor = boundary.opening_end_tidx;
            component_traces.push(trace);
            boundaries.push(boundary);
        }
        if boundaries.len() != self.profile.protocol_component_count()
            || self.profile.global_opening_count
                != self
                    .profile
                    .components
                    .last()
                    .ok_or(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                        "missing final component",
                    ))?
                    .opening_end()?
        {
            return Err(FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                "final cursor coverage",
            ));
        }
        Ok(FixedMultiAirCompleteRegionCursorOpeningTraceData {
            sequence,
            components: component_traces,
            boundaries,
            batch_start_tidx: cursor,
            global_opening_count: self.profile.global_opening_count,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedMultiAirCompleteRegionCursorOpeningRequiredHeights {
    pub sequence: Option<usize>,
    /// Exact native component order.
    pub components: Vec<Option<usize>>,
}

impl FixedMultiAirCompleteRegionCursorOpeningRequiredHeights {
    #[must_use]
    pub fn minimal(component_count: usize) -> Self {
        Self {
            sequence: None,
            components: vec![None; component_count],
        }
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteRegionCursorOpeningPartitionedTrace {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedMultiAirCompleteRegionCursorBoundary {
    pub kind: FixedMultiAirCompleteTerminalCircuitComponentKind,
    pub region: usize,
    pub component_start_tidx: usize,
    pub opening_start_tidx: usize,
    pub opening_end_tidx: usize,
    pub final_evaluation: EF,
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteRegionCursorOpeningTraceData {
    pub sequence: FixedMultiAirCompleteRegionCursorOpeningPartitionedTrace,
    pub components: Vec<FixedMultiAirCompleteRegionCursorOpeningPartitionedTrace>,
    pub boundaries: Vec<FixedMultiAirCompleteRegionCursorBoundary>,
    pub batch_start_tidx: usize,
    pub global_opening_count: usize,
}

fn validate_complete_proof_shape(
    profile: &FixedMultiAirCompleteRegionCursorOpeningProfile,
    proof: &FixedMultiAirCompleteTerminalProof<EF>,
) -> CursorResult<()> {
    let region_count = profile.plan.regions.len();
    if proof.local_claims.len() != region_count
        || proof.interaction_claims.len() != region_count
        || proof.local_proofs.len() != region_count
        || proof.interaction_proofs.len() != region_count
    {
        return Err(FixedMultiAirCompleteRegionCursorOpeningError::Shape(
            "regional proof vectors",
        ));
    }
    for region in 0..region_count {
        let expected = profile.plan.regions[region]
            .interaction
            .identity
            .proof_present;
        let actual = proof
            .interaction_proofs
            .get(region)
            .ok_or(FixedMultiAirCompleteRegionCursorOpeningError::Shape(
                "interaction proof vector",
            ))?
            .is_some();
        if actual != expected {
            return Err(
                FixedMultiAirCompleteRegionCursorOpeningError::InteractionPresence { region },
            );
        }
        if !expected
            && proof.interaction_claims.get(region).copied().ok_or(
                FixedMultiAirCompleteRegionCursorOpeningError::Shape("interaction claim vector"),
            )? != EF::ZERO
        {
            return Err(FixedMultiAirCompleteRegionCursorOpeningError::Shape(
                "empty interaction claim",
            ));
        }
    }
    Ok(())
}

fn component_proof<'a>(
    proof: &'a FixedMultiAirCompleteTerminalProof<EF>,
    profile: &FixedMultiAirCompleteRegionCursorComponentProfile,
) -> CursorResult<(EF, &'a FixedMultiAirCompleteRegionSumcheckProof<EF>)> {
    match profile.kind {
        FixedMultiAirCompleteTerminalCircuitComponentKind::Local => {
            let claim = proof
                .local_claims
                .get(profile.claim_ordinal)
                .copied()
                .ok_or(FixedMultiAirCompleteRegionCursorOpeningError::Shape(
                    "local claim ordinal",
                ))?;
            let regional = proof.local_proofs.get(profile.proof_ordinal).ok_or(
                FixedMultiAirCompleteRegionCursorOpeningError::Shape("local proof ordinal"),
            )?;
            Ok((claim, regional))
        }
        FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction => {
            let claim = proof
                .interaction_claims
                .get(profile.claim_ordinal)
                .copied()
                .ok_or(FixedMultiAirCompleteRegionCursorOpeningError::Shape(
                    "interaction claim ordinal",
                ))?;
            let regional = proof
                .interaction_proofs
                .get(profile.proof_ordinal)
                .ok_or(FixedMultiAirCompleteRegionCursorOpeningError::Shape(
                    "interaction proof ordinal",
                ))?
                .as_ref()
                .ok_or(
                    FixedMultiAirCompleteRegionCursorOpeningError::InteractionPresence {
                        region: profile.region,
                    },
                )?;
            Ok((claim, regional))
        }
    }
}

fn generate_sequence_trace(
    profile: &FixedMultiAirCompleteRegionCursorOpeningProfile,
    proof: &FixedMultiAirCompleteTerminalProof<EF>,
    decomposition_end_tidx: usize,
    eta: EF,
    beta_claim: EF,
    required_height: Option<usize>,
) -> CursorResult<FixedMultiAirCompleteRegionCursorOpeningPartitionedTrace> {
    let rows = profile.sequence_row_count();
    let height = admitted_height(rows, required_height, None)?;
    let cached_width = FixedMultiAirCompleteRegionSequenceScheduleCols::<F>::width();
    let common_width = FixedMultiAirCompleteRegionSequenceCols::<F>::width();
    let cached_cells = checked_cells(height, cached_width)?;
    let common_cells = checked_cells(height, common_width)?;
    let mut cached = try_zero_vec(cached_cells)?;
    let mut common = try_zero_vec(common_cells)?;

    for (row, region) in profile
        .empty_interaction_regions
        .iter()
        .copied()
        .map(Some)
        .chain(core::iter::once(None))
        .enumerate()
    {
        let schedule: &mut FixedMultiAirCompleteRegionSequenceScheduleCols<F> =
            cached[row * cached_width..(row + 1) * cached_width].borrow_mut();
        schedule.active = F::ONE;
        schedule.is_first = F::from_bool(row == 0);
        schedule.is_last = F::from_bool(row + 1 == rows);
        if let Some(region) = region {
            schedule.is_empty_interaction = F::ONE;
            schedule.region = F::from_usize(region);
        }
        let cols: &mut FixedMultiAirCompleteRegionSequenceCols<F> =
            common[row * common_width..(row + 1) * common_width].borrow_mut();
        cols.decomposition_end_tidx = F::from_usize(decomposition_end_tidx);
        copy_ext(&mut cols.eta, eta);
        copy_ext(&mut cols.beta_claim, beta_claim);
        if let Some(region) = region {
            let claim = proof.interaction_claims.get(region).copied().ok_or(
                FixedMultiAirCompleteRegionCursorOpeningError::Shape(
                    "empty interaction claim ordinal",
                ),
            )?;
            if claim != EF::ZERO {
                return Err(FixedMultiAirCompleteRegionCursorOpeningError::Shape(
                    "nonzero empty interaction claim",
                ));
            }
            copy_ext(&mut cols.empty_interaction_claim, claim);
        }
    }
    Ok(FixedMultiAirCompleteRegionCursorOpeningPartitionedTrace {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
    })
}

fn generate_component_opening_trace(
    air: &FixedMultiAirCompleteRegionOpeningAir,
    initial_claim: EF,
    proof: &FixedMultiAirCompleteRegionSumcheckProof<EF>,
    transcript: &TranscriptLog<F, [F; 16]>,
    component_start_tidx: usize,
    required_height: Option<usize>,
) -> CursorResult<(
    FixedMultiAirCompleteRegionCursorOpeningPartitionedTrace,
    FixedMultiAirCompleteRegionCursorBoundary,
)> {
    let profile = air.profile.as_ref();
    if proof.round_evaluations.len() != profile.round_count
        || proof.opened_columns.len() != profile.opening_count
        || profile.opening_lookup_counts.len() != profile.opening_count
        || proof
            .round_evaluations
            .iter()
            .any(|evaluations| evaluations.len() != FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS)
    {
        return Err(FixedMultiAirCompleteRegionCursorOpeningError::Shape(
            "regional proof dimensions",
        ));
    }
    let (final_claim, opening_start_tidx) = replay_regional_sumcheck(
        profile,
        initial_claim,
        proof,
        transcript,
        component_start_tidx,
    )?;
    let rows = profile.opening_tail_rows()?;
    let height = admitted_height(rows, required_height, Some(profile.component_ordinal))?;
    let cached_width = FixedMultiAirCompleteRegionOpeningScheduleCols::<F>::width();
    let common_width = FixedMultiAirCompleteRegionOpeningCols::<F>::width();
    let mut cached = try_zero_vec(checked_cells(height, cached_width)?)?;
    let mut common = try_zero_vec(checked_cells(height, common_width)?)?;
    let opening_end_tidx = rows
        .checked_mul(D_EF)
        .and_then(|span| opening_start_tidx.checked_add(span))
        .ok_or(FixedMultiAirCompleteRegionCursorOpeningError::Overflow(
            "opening transcript endpoint",
        ))?;
    validate_transcript_index(opening_end_tidx, "opening transcript endpoint")?;

    let mut tidx = opening_start_tidx;
    for row in 0..rows {
        let (source, source_index, expected) = match row {
            0 => (
                TAIL_SOURCE_CONSTANT,
                0,
                EF::from_u64(COMPLETE_TERMINAL_OPENINGS_TAG),
            ),
            1 => (
                TAIL_SOURCE_CONSTANT,
                0,
                EF::from_usize(profile.opening_count),
            ),
            _ => {
                let opening = row - 2;
                let value = proof.opened_columns.get(opening).copied().ok_or(
                    FixedMultiAirCompleteRegionCursorOpeningError::Shape("opening ordinal"),
                )?;
                (TAIL_SOURCE_OPENING, opening, value)
            }
        };
        expect_ext(transcript, tidx, expected, false)?;
        let schedule: &mut FixedMultiAirCompleteRegionOpeningScheduleCols<F> =
            cached[row * cached_width..(row + 1) * cached_width].borrow_mut();
        schedule.active = F::ONE;
        schedule.is_first = F::from_bool(row == 0);
        schedule.is_last = F::from_bool(row + 1 == rows);
        schedule.source_flags[source] = F::ONE;
        schedule.source_index = F::from_usize(source_index);
        if source == TAIL_SOURCE_CONSTANT {
            copy_ext(&mut schedule.constant, expected);
        } else {
            schedule.opening_lookup_count =
                F::from_u32(*profile.opening_lookup_counts.get(source_index).ok_or(
                    FixedMultiAirCompleteRegionCursorOpeningError::Plan(
                        "opening lookup multiplicity ordinal",
                    ),
                )?);
        }

        let cols: &mut FixedMultiAirCompleteRegionOpeningCols<F> =
            common[row * common_width..(row + 1) * common_width].borrow_mut();
        cols.component_start_tidx = F::from_usize(component_start_tidx);
        cols.tidx = F::from_usize(tidx);
        copy_ext(&mut cols.value, expected);
        copy_ext(&mut cols.initial_claim, initial_claim);
        copy_ext(&mut cols.final_claim, final_claim);
        tidx = tidx.checked_add(D_EF).ok_or(
            FixedMultiAirCompleteRegionCursorOpeningError::Overflow("opening transcript cursor"),
        )?;
    }
    if tidx != opening_end_tidx {
        return Err(FixedMultiAirCompleteRegionCursorOpeningError::Transcript(
            "opening tail coverage",
        ));
    }
    Ok((
        FixedMultiAirCompleteRegionCursorOpeningPartitionedTrace {
            cached: RowMajorMatrix::new(cached, cached_width),
            common: RowMajorMatrix::new(common, common_width),
        },
        FixedMultiAirCompleteRegionCursorBoundary {
            kind: profile.kind,
            region: profile.region,
            component_start_tidx,
            opening_start_tidx,
            opening_end_tidx,
            final_evaluation: final_claim,
        },
    ))
}

fn replay_regional_sumcheck(
    profile: &FixedMultiAirCompleteRegionCursorComponentProfile,
    initial_claim: EF,
    proof: &FixedMultiAirCompleteRegionSumcheckProof<EF>,
    transcript: &TranscriptLog<F, [F; 16]>,
    component_start_tidx: usize,
) -> CursorResult<(EF, usize)> {
    expect_ext(
        transcript,
        component_start_tidx,
        EF::from_u64(profile.region_tag()),
        false,
    )?;
    let region_tidx = component_start_tidx.checked_add(D_EF).ok_or(
        FixedMultiAirCompleteRegionCursorOpeningError::Overflow("region ordinal cursor"),
    )?;
    expect_ext(
        transcript,
        region_tidx,
        EF::from_usize(profile.region),
        false,
    )?;
    let mut round_tidx = component_start_tidx
        .checked_add(FIXED_MULTI_AIR_COMPLETE_REGION_PREFIX_BASE_VALUES)
        .ok_or(FixedMultiAirCompleteRegionCursorOpeningError::Overflow(
            "first round cursor",
        ))?;
    let mut claim = initial_claim;
    for (round, evaluations) in proof.round_evaluations.iter().enumerate() {
        let evaluations: &[EF; FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS] =
            evaluations.as_slice().try_into().map_err(|_| {
                FixedMultiAirCompleteRegionCursorOpeningError::Shape("round evaluation count")
            })?;
        expect_ext(
            transcript,
            round_tidx,
            EF::from_u64(COMPLETE_TERMINAL_ROUND_TAG),
            false,
        )?;
        expect_ext(
            transcript,
            checked_add_slots(round_tidx, 1, "round ordinal cursor")?,
            EF::from_usize(round),
            false,
        )?;
        expect_ext(
            transcript,
            checked_add_slots(round_tidx, 2, "round count cursor")?,
            EF::from_usize(FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS),
            false,
        )?;
        for (evaluation, &value) in evaluations.iter().enumerate() {
            expect_ext(
                transcript,
                checked_add_slots(round_tidx, 3 + evaluation, "round evaluation cursor")?,
                value,
                false,
            )?;
        }
        if evaluations[0] + evaluations[1] != claim {
            return Err(FixedMultiAirCompleteRegionCursorOpeningError::Claim {
                component: profile.kind,
                region: profile.region,
                round,
            });
        }
        let challenge = read_ext(
            transcript,
            checked_add_slots(
                round_tidx,
                3 + FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS,
                "round challenge cursor",
            )?,
            true,
        )?;
        claim = interpolate(evaluations, challenge);
        round_tidx = round_tidx
            .checked_add(FIXED_MULTI_AIR_COMPLETE_ROUND_TRANSCRIPT_BASE_VALUES)
            .ok_or(FixedMultiAirCompleteRegionCursorOpeningError::Overflow(
                "round transcript cursor",
            ))?;
    }
    let expected_opening_start = component_start_tidx
        .checked_add(profile.regional_transcript_span()?)
        .ok_or(FixedMultiAirCompleteRegionCursorOpeningError::Overflow(
            "opening start cursor",
        ))?;
    if round_tidx != expected_opening_start {
        return Err(FixedMultiAirCompleteRegionCursorOpeningError::Transcript(
            "regional transcript coverage",
        ));
    }
    Ok((claim, round_tidx))
}

fn interpolate(
    evaluations: &[EF; FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS],
    challenge: EF,
) -> EF {
    evaluations
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            let numerator = (0..FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS)
                .filter(|&other| other != index)
                .map(|other| challenge - EF::from_usize(other))
                .product::<EF>();
            value * numerator * EF::from(lagrange_denominator(index).inverse())
        })
        .sum()
}

fn lagrange_denominator(index: usize) -> F {
    (0..FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS)
        .filter(|&other| other != index)
        .map(|other| F::from_usize(index) - F::from_usize(other))
        .product()
}

fn expect_ext(
    transcript: &TranscriptLog<F, [F; 16]>,
    tidx: usize,
    expected: EF,
    is_sample: bool,
) -> CursorResult<()> {
    let actual = read_ext(transcript, tidx, is_sample)?;
    if actual != expected {
        return Err(FixedMultiAirCompleteRegionCursorOpeningError::Transcript(
            "operation value",
        ));
    }
    Ok(())
}

fn read_ext(
    transcript: &TranscriptLog<F, [F; 16]>,
    tidx: usize,
    is_sample: bool,
) -> CursorResult<EF> {
    let end =
        tidx.checked_add(D_EF)
            .ok_or(FixedMultiAirCompleteRegionCursorOpeningError::Overflow(
                "transcript operation",
            ))?;
    let values = transcript.values().get(tidx..end).ok_or(
        FixedMultiAirCompleteRegionCursorOpeningError::Transcript("operation range"),
    )?;
    let samples = transcript.samples().get(tidx..end).ok_or(
        FixedMultiAirCompleteRegionCursorOpeningError::Transcript("sample range"),
    )?;
    if samples.iter().any(|&sample| sample != is_sample) {
        return Err(FixedMultiAirCompleteRegionCursorOpeningError::Transcript(
            "operation kind",
        ));
    }
    EF::from_basis_coefficients_slice(values).ok_or(
        FixedMultiAirCompleteRegionCursorOpeningError::Transcript("extension-field operation"),
    )
}

fn checked_add_slots(tidx: usize, slots: usize, context: &'static str) -> CursorResult<usize> {
    slots
        .checked_mul(D_EF)
        .and_then(|offset| tidx.checked_add(offset))
        .ok_or(FixedMultiAirCompleteRegionCursorOpeningError::Overflow(
            context,
        ))
}

fn validate_transcript_index(index: usize, context: &'static str) -> CursorResult<()> {
    if index >= F::ORDER_U32 as usize {
        return Err(FixedMultiAirCompleteRegionCursorOpeningError::Overflow(
            context,
        ));
    }
    Ok(())
}

fn validate_field_word(value: usize, context: &'static str) -> CursorResult<()> {
    if value >= F::ORDER_U32 as usize {
        return Err(FixedMultiAirCompleteRegionCursorOpeningError::Overflow(
            context,
        ));
    }
    Ok(())
}

fn checked_usize(value: u32, context: &'static str) -> CursorResult<usize> {
    usize::try_from(value)
        .map_err(|_| FixedMultiAirCompleteRegionCursorOpeningError::Overflow(context))
}

fn admitted_height(
    rows: usize,
    required: Option<usize>,
    component: Option<usize>,
) -> CursorResult<usize> {
    if rows == 0 {
        return Err(FixedMultiAirCompleteRegionCursorOpeningError::Shape(
            "zero-row trace",
        ));
    }
    let minimum = rows.checked_next_power_of_two().ok_or(
        FixedMultiAirCompleteRegionCursorOpeningError::Overflow("trace height"),
    )?;
    match required {
        Some(height) if height.is_power_of_two() && height >= rows => Ok(height),
        Some(_) => Err(FixedMultiAirCompleteRegionCursorOpeningError::TraceHeight { component }),
        None => Ok(minimum),
    }
}

fn checked_cells(height: usize, width: usize) -> CursorResult<usize> {
    height
        .checked_mul(width)
        .ok_or(FixedMultiAirCompleteRegionCursorOpeningError::Overflow(
            "trace cell count",
        ))
}

fn try_zero_vec(len: usize) -> CursorResult<Vec<F>> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| FixedMultiAirCompleteRegionCursorOpeningError::Allocation)?;
    values.resize(len, F::ZERO);
    Ok(values)
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

#[cfg(test)]
mod tests {
    use core::any::TypeId;
    use std::{
        panic::{catch_unwind, AssertUnwindSafe},
        sync::Arc,
    };

    use openvm_stark_backend::{
        air_builders::{
            debug::{check_constraints, check_logup, DebugConstraintBuilder},
            symbolic::{
                get_symbolic_builder, symbolic_expression::SymbolicExpression,
                SymbolicConstraintsDag, SymbolicRapBuilder,
            },
        },
        hasher::MerkleHasher,
        interaction::{InteractionBuilder, SymbolicInteraction},
        keygen::types::{StarkVerifyingKey, StarkVerifyingParams, TraceWidth},
        native_warp::{
            DirectAirCodeClass, DirectAirPesatIndex, DirectAirPesatInstance, DirectAirPublicSchema,
            FixedMultiAirCompletePesatIndex, FixedMultiAirCompletePesatInstance,
            FixedMultiAirCompleteSourceRegion, FixedMultiAirCompleteTerminalLinearizer,
            NativeWarpChallenger,
        },
        transcript::TranscriptHistory,
        warp_pesat::{evaluate_mle, AccumulatorInstance, StructuredTerminalPesatLinearizer},
        StarkProtocolConfig, SystemParams,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::{
        default_duplex_sponge_recorder, BabyBearPoseidon2Config as SC, DIGEST_SIZE,
    };
    use p3_air::AirBuilderWithPublicValues;

    use super::*;
    use crate::{
        bus::TranscriptBusMessage,
        native_warp::terminal::fixed_multi_air_complete::{
            generate_fixed_multi_air_complete_interaction_region_sumcheck_trace,
            generate_fixed_multi_air_complete_local_region_sumcheck_trace,
            FixedMultiAirCompleteInteractionRegionSumcheckAir,
            FixedMultiAirCompleteLocalRegionSumcheckAir, FixedMultiAirCompleteRegionPointMessage,
        },
    };

    type TestDigest = [F; DIGEST_SIZE];

    #[derive(Clone, Copy)]
    struct TestAir {
        /// `1` sends, `-1` receives, and `0` has no interaction.
        interaction: i8,
        bus: u16,
    }

    impl BaseAir<F> for TestAir {
        fn width(&self) -> usize {
            2
        }
    }

    impl BaseAirWithPublicValues<F> for TestAir {
        fn num_public_values(&self) -> usize {
            1
        }
    }

    impl PartitionedBaseAir<F> for TestAir {}

    impl Air<SymbolicRapBuilder<F>> for TestAir {
        fn eval(&self, builder: &mut SymbolicRapBuilder<F>) {
            let main = builder.common_main().clone();
            let local = main.row_slice(0).expect("test complete-terminal row");
            let value = local[0];
            let multiplicity = local[1];
            builder.assert_zero(value - builder.public_values()[0]);
            if self.interaction != 0 {
                let count = SymbolicExpression::from(multiplicity);
                let count = if self.interaction > 0 { count } else { -count };
                builder.push_interaction(self.bus, [value, value], count, 1);
            }
        }
    }

    fn digest(value: u32) -> TestDigest {
        [F::from_u32(value); DIGEST_SIZE]
    }

    fn verifying_key(air: &TestAir) -> StarkVerifyingKey<F, TestDigest> {
        let width = TraceWidth {
            preprocessed: None,
            cached_mains: Vec::new(),
            common_main: 2,
        };
        let symbolic = get_symbolic_builder(air, &width).constraints();
        StarkVerifyingKey {
            preprocessed_data: None,
            params: StarkVerifyingParams {
                width,
                num_public_values: 1,
                need_rot: false,
            },
            max_constraint_degree: symbolic.max_constraint_degree() as u8,
            symbolic_constraints: Arc::new(SymbolicConstraintsDag::from(symbolic)),
            is_required: true,
            unused_variables: Vec::new(),
        }
    }

    fn direct_relation<H>(
        hasher: &H,
        air: &TestAir,
        air_id: usize,
        log_height: usize,
    ) -> DirectAirPesatIndex<F, TestDigest>
    where
        H: MerkleHasher<F = F, Digest = TestDigest>,
    {
        DirectAirPesatIndex::from_verifying_key(
            hasher,
            digest(1),
            air_id,
            log_height,
            &verifying_key(air),
            None,
            DirectAirPublicSchema {
                public_values_len: 1,
                boundary_values_len: 0,
                schema_digest: digest(100 + air_id as u32),
            },
            DirectAirCodeClass {
                log_message_len: u8::try_from(log_height + 1).expect("small direct message"),
                log_blowup: 1,
                log_codeword_len: u8::try_from(log_height + 2).expect("small direct codeword"),
                initial_folding_factor: 0,
                rows_per_query: 2,
            },
        )
        .expect("direct complete-terminal relation")
    }

    struct BackendFixture {
        plan: Arc<FixedMultiAirCompleteTerminalCircuitPlan<F, TestDigest>>,
        proof: FixedMultiAirCompleteTerminalProof<EF>,
        transcript: TranscriptLog<F, [F; 16]>,
        decomposition_end_tidx: usize,
        eta: EF,
        beta_claim: EF,
    }

    fn backend_fixture() -> BackendFixture {
        let config = SC::default_from_params(SystemParams::new_for_testing(8));
        let hasher = config.hasher();
        // The complete relation requires descending, trace-aligned AIR
        // heights. Regions 0 and 1 carry matching send/receive interactions
        // over the same two-row domain; region 2 is zero-round and
        // interaction-empty.
        let shapes = [(3usize, 1usize, 1i8), (9, 1, -1), (12, 0, 0)];
        let relations = shapes
            .iter()
            .map(|&(air_id, log_height, interaction)| {
                direct_relation(
                    hasher,
                    &TestAir {
                        interaction,
                        bus: 7,
                    },
                    air_id,
                    log_height,
                )
            })
            .collect::<Vec<_>>();
        // Two (4 trace + 8 inverse-coordinate) regions plus one two-cell
        // trace contain 26 raw cells, canonically padded to 32.
        let complete_log_message_len = 5u8;
        let relation = FixedMultiAirCompletePesatIndex::from_direct_air_regions(
            hasher,
            digest(1),
            relations,
            DirectAirCodeClass {
                log_message_len: complete_log_message_len,
                log_blowup: 1,
                log_codeword_len: complete_log_message_len + 1,
                initial_folding_factor: 0,
                rows_per_query: 2,
            },
        )
        .expect("fixed complete relation");
        let sources = shapes
            .iter()
            .map(|&(air_id, log_height, interaction)| {
                let height = 1usize << log_height;
                FixedMultiAirCompleteSourceRegion {
                    air_id: u32::try_from(air_id).expect("small fixture AIR id"),
                    public_values: vec![F::from_u32(7)],
                    boundary_values: Vec::new(),
                    // Source cells are column-major: value column, then
                    // multiplicity column.
                    common_trace_cells: (0..height)
                        .map(|_| F::from_u32(7))
                        .chain((0..height).map(|_| F::from_bool(interaction != 0)))
                        .collect(),
                }
            })
            .collect::<Vec<_>>();
        let direct_instances = shapes
            .iter()
            .map(|_| DirectAirPesatInstance {
                public_values: vec![F::from_u32(7)],
                boundary_values: Vec::new(),
            })
            .collect::<Vec<_>>();
        let public = FixedMultiAirCompletePesatInstance::from_alpha_beta(
            EF::from_u32(11),
            EF::from_u32(5),
            direct_instances,
            2,
        );
        let witness = relation
            .synthesize_witness(&public, &sources)
            .expect("complete witness");
        let message = relation
            .padded_witness::<EF>(&witness)
            .expect("padded complete witness");
        let explicit = relation
            .explicit_assignment(&public)
            .expect("complete explicit assignment");
        let constraints = relation
            .evaluate_reference(&public, &witness)
            .expect("complete relation evaluation");
        if let Some((index, value)) = constraints
            .iter()
            .copied()
            .enumerate()
            .find(|(_, value)| *value != EF::ZERO)
        {
            panic!("fixture constraint {index} is nonzero: {value:?}");
        }
        let tau = (0..relation.pesat_shape().log_constraints)
            .map(|index| EF::from_usize(17 + index))
            .collect::<Vec<_>>();
        let eta = evaluate_mle(&constraints, &tau);
        let mut beta = tau;
        beta.extend(explicit.into_iter().map(EF::from));
        let instance = AccumulatorInstance {
            rt: digest(77),
            alpha: (0..usize::from(complete_log_message_len + 1))
                .map(|index| EF::from_usize(31 + index))
                .collect(),
            mu: EF::from_u32(41),
            beta,
            eta,
        };
        let linearizer =
            FixedMultiAirCompleteTerminalLinearizer::new(&relation).expect("linearizer");
        let plan = Arc::new(linearizer.circuit_plan().expect("canonical circuit plan"));
        let mut challenger = NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
        let (proof, claims) = linearizer
            .prove_from_reader(&instance, &message, &mut challenger)
            .expect("backend complete terminal proof");
        let transcript = TranscriptHistory::into_log(challenger.into_inner());
        let mut verifier = NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
        let verified = linearizer
            .verify_structured_terminal_claims(&instance, &proof, &mut verifier)
            .expect("backend complete terminal verification");
        assert_eq!(verified, claims);
        let verifier_log = TranscriptHistory::into_log(verifier.into_inner());
        assert_eq!(transcript.values(), verifier_log.values());
        assert_eq!(transcript.samples(), verifier_log.samples());
        let decomposition_end_tidx =
            find_region_start(&transcript, COMPLETE_TERMINAL_LOCAL_REGION_TAG, 0);
        let eta = instance.eta;
        let beta_claim = eta
            - proof.local_claims.iter().copied().sum::<EF>()
            - proof.interaction_claims.iter().copied().sum::<EF>();
        BackendFixture {
            plan,
            proof,
            transcript,
            decomposition_end_tidx,
            eta,
            beta_claim,
        }
    }

    fn find_region_start(transcript: &TranscriptLog<F, [F; 16]>, tag: u64, region: usize) -> usize {
        let tag = EF::from_u64(tag);
        let region = EF::from_usize(region);
        let tag_limbs = tag.as_basis_coefficients_slice();
        let region_limbs = region.as_basis_coefficients_slice();
        transcript
            .values()
            .windows(2 * D_EF)
            .enumerate()
            .find_map(|(start, values)| {
                let flags = transcript.samples().get(start..start + 2 * D_EF)?;
                (values[..D_EF] == *tag_limbs
                    && values[D_EF..] == *region_limbs
                    && flags.iter().all(|&flag| !flag))
                .then_some(start)
            })
            .expect("backend regional domain")
    }

    fn buses() -> FixedMultiAirCompleteRegionCursorOpeningBuses {
        FixedMultiAirCompleteRegionCursorOpeningBuses {
            transcript: TranscriptBus::new(301),
            sequence_start: FixedMultiAirCompleteRegionSequenceStartBus::new(302),
            decomposition_receipt: FixedMultiAirCompleteDecompositionReceiptBus::new(317),
            cursor: FixedMultiAirCompleteRegionCursorBus::new(303),
            local_claim: FixedMultiAirCompleteLocalClaimBus::new(304),
            interaction_claim: FixedMultiAirCompleteInteractionClaimBus::new(305),
            local_start: FixedMultiAirCompleteLocalRegionStartBus::new(306),
            interaction_start: FixedMultiAirCompleteInteractionRegionStartBus::new(307),
            local_sumcheck_final: FixedMultiAirCompleteLocalRegionSumcheckFinalBus::new(308),
            interaction_sumcheck_final: FixedMultiAirCompleteInteractionRegionSumcheckFinalBus::new(
                309,
            ),
            authorities: FixedMultiAirCompleteRegionalAuthorityBuses {
                local_point: FixedMultiAirCompleteLocalRegionPointBus::new(310),
                interaction_point: FixedMultiAirCompleteInteractionRegionPointBus::new(311),
                local_opening: FixedMultiAirCompleteLocalRegionOpeningBus::new(312),
                interaction_opening: FixedMultiAirCompleteInteractionRegionOpeningBus::new(313),
                local_final_evaluation: FixedMultiAirCompleteLocalRegionFinalEvaluationBus::new(
                    314,
                ),
                interaction_final_evaluation:
                    FixedMultiAirCompleteInteractionRegionFinalEvaluationBus::new(315),
            },
            batch_start: FixedMultiAirCompleteOpeningBatchStartBus::new(316),
        }
    }

    fn owner(fixture: &BackendFixture) -> FixedMultiAirCompleteRegionCursorOpeningAirs {
        let profile = Arc::new(
            FixedMultiAirCompleteRegionCursorOpeningProfile::from_plan(fixture.plan.clone())
                .expect("canonical cursor profile"),
        );
        FixedMultiAirCompleteRegionCursorOpeningAirs::new(profile, buses())
            .expect("cursor/opening AIRs")
    }

    fn traces(
        fixture: &BackendFixture,
        owner: &FixedMultiAirCompleteRegionCursorOpeningAirs,
    ) -> FixedMultiAirCompleteRegionCursorOpeningTraceData {
        owner
            .generate_traces(
                &fixture.proof,
                &fixture.transcript,
                fixture.decomposition_end_tidx,
                fixture.eta,
                fixture.beta_claim,
                FixedMultiAirCompleteRegionCursorOpeningRequiredHeights::minimal(
                    owner.profile.protocol_component_count(),
                ),
            )
            .expect("complete cursor/opening traces")
    }

    fn check_partitioned<A>(
        air: &A,
        trace: &FixedMultiAirCompleteRegionCursorOpeningPartitionedTrace,
    ) where
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
    fn native_local_then_interaction_transcript_and_zero_round_match() {
        let fixture = backend_fixture();
        let owner = owner(&fixture);
        let traces = traces(&fixture, &owner);
        let order = owner
            .profile
            .components
            .iter()
            .map(|component| (component.kind, component.region))
            .collect::<Vec<_>>();
        assert_eq!(
            order,
            vec![
                (FixedMultiAirCompleteTerminalCircuitComponentKind::Local, 0),
                (FixedMultiAirCompleteTerminalCircuitComponentKind::Local, 1),
                (FixedMultiAirCompleteTerminalCircuitComponentKind::Local, 2),
                (
                    FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction,
                    0,
                ),
                (
                    FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction,
                    1,
                ),
            ]
        );
        assert_eq!(owner.profile.empty_interaction_regions, vec![2]);
        assert_eq!(
            traces.boundaries[0].component_start_tidx,
            fixture.decomposition_end_tidx
        );
        assert_eq!(
            traces.boundaries[0].opening_start_tidx,
            fixture.decomposition_end_tidx
                + FIXED_MULTI_AIR_COMPLETE_REGION_PREFIX_BASE_VALUES
                + FIXED_MULTI_AIR_COMPLETE_ROUND_TRANSCRIPT_BASE_VALUES
        );
        assert_eq!(owner.profile.components[0].round_count, 1);
        assert_eq!(owner.profile.components[2].round_count, 0);
        for pair in traces.boundaries.windows(2) {
            assert_eq!(pair[0].opening_end_tidx, pair[1].component_start_tidx);
        }
        assert_eq!(
            traces.batch_start_tidx,
            traces
                .boundaries
                .last()
                .expect("final boundary")
                .opening_end_tidx
        );
        assert_eq!(
            read_ext(&fixture.transcript, traces.batch_start_tidx, false,)
                .expect("native batch tag"),
            EF::from_u64(COMPLETE_TERMINAL_BATCH_TAG)
        );
        assert_eq!(
            read_ext(&fixture.transcript, traces.batch_start_tidx + D_EF, false,)
                .expect("native batch count"),
            EF::from_usize(traces.global_opening_count)
        );
        check_partitioned(&owner.sequence, &traces.sequence);
        for (air, trace) in owner.components.iter().zip(&traces.components) {
            check_partitioned(air, trace);
        }
    }

    #[test]
    fn setup_derived_opening_multiplicities_match_endpoint_consumers() {
        let fixture = backend_fixture();
        let owner = owner(&fixture);
        let traces = traces(&fixture, &owner);

        for ((profile, air), trace) in owner
            .profile
            .components
            .iter()
            .zip(&owner.components)
            .zip(&traces.components)
        {
            let component = match profile.kind {
                FixedMultiAirCompleteTerminalCircuitComponentKind::Local => {
                    &fixture.plan.regions[profile.region].local
                }
                FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction => {
                    &fixture.plan.regions[profile.region].interaction
                }
            };
            // Independent setup oracle: enumerate the actual expression and
            // q consumers instead of reading the cursor profile or trace.
            let mut endpoint_counts = vec![1u32; component.expression.dynamic_columns.len()];
            for (node, source) in component
                .expression
                .nodes
                .iter()
                .zip(&component.expression.node_sources)
            {
                if matches!(
                    node,
                    SymbolicExpressionNode::Variable(variable)
                        if matches!(variable.entry, Entry::Main { .. } | Entry::Preprocessed { .. })
                ) {
                    if let Some(FixedMultiAirCompleteTerminalCircuitNodeSource::Dynamic(source)) =
                        source
                    {
                        endpoint_counts[*source as usize] += 1;
                    }
                }
            }
            for interaction in &component.interactions {
                for &source in &interaction.q_source_ordinals {
                    endpoint_counts[source as usize] += 1;
                }
            }
            assert_eq!(profile.opening_lookup_counts, endpoint_counts);
            assert_eq!(
                profile.point_fixed_source_count,
                component.expression.fixed_columns.len()
            );
            assert_eq!(
                profile.point_common_count,
                1 + component.expression.dynamic_columns.len(),
                "one endpoint point use plus one global-mapper use per mapped opening"
            );

            let width = FixedMultiAirCompleteRegionOpeningScheduleCols::<F>::width();
            assert_eq!(air.cached_main_widths(), vec![width]);
            for (opening, &count) in profile.opening_lookup_counts.iter().enumerate() {
                let row = opening + 2;
                let schedule: &FixedMultiAirCompleteRegionOpeningScheduleCols<F> =
                    trace.cached.values[row * width..(row + 1) * width].borrow();
                assert_eq!(schedule.opening_lookup_count, F::from_u32(count));
            }
        }

        // The minimal honest fixture happens to use every opening exactly
        // twice. Force a setup-only q fanout variant to ensure derivation is
        // genuinely per opening rather than a disguised uniform constant.
        let mut nonuniform = fixture.plan.regions[0].interaction.clone();
        let replaced = nonuniform.interactions[0].q_source_ordinals[1];
        let repeated = nonuniform.interactions[0].q_source_ordinals[0];
        assert_ne!(repeated, replaced, "fixture q coordinates must be distinct");
        let baseline = opening_lookup_counts(&fixture.plan.regions[0].interaction)
            .expect("honest interaction multiplicities");
        nonuniform.interactions[0].q_source_ordinals[1] = repeated;
        let derived = opening_lookup_counts(&nonuniform).expect("nonuniform setup multiplicities");
        assert_eq!(derived[repeated as usize], baseline[repeated as usize] + 1);
        assert_eq!(derived[replaced as usize] + 1, baseline[replaced as usize]);

        let mut dynamic_source = (*fixture.plan).clone();
        let source = dynamic_source.regions[0]
            .local
            .expression
            .node_sources
            .iter_mut()
            .find(|source| {
                matches!(
                    source,
                    Some(FixedMultiAirCompleteTerminalCircuitNodeSource::Dynamic(_))
                )
            })
            .expect("fixture dynamic expression source");
        *source = Some(FixedMultiAirCompleteTerminalCircuitNodeSource::Dynamic(
            u32::MAX,
        ));
        assert_error_without_panic(|| {
            FixedMultiAirCompleteRegionCursorOpeningProfile::from_plan(Arc::new(dynamic_source))
        });

        let mut q_source = (*fixture.plan).clone();
        q_source.regions[0].interaction.interactions[0].q_source_ordinals[0] = u32::MAX;
        assert_error_without_panic(|| {
            FixedMultiAirCompleteRegionCursorOpeningProfile::from_plan(Arc::new(q_source))
        });
    }

    fn assert_error_without_panic<T>(operation: impl FnOnce() -> CursorResult<T>) {
        let result = catch_unwind(AssertUnwindSafe(operation));
        assert!(
            matches!(result, Ok(Err(_))),
            "malformed input did not return an error"
        );
    }

    #[test]
    fn opening_count_tidx_presence_and_proof_order_mutations_reject_without_panic() {
        let fixture = backend_fixture();
        let owner = owner(&fixture);
        let honest = traces(&fixture, &owner);
        let first = &honest.boundaries[0];

        let mut opening = fixture.transcript.clone();
        opening.values_mut()[first.opening_start_tidx + 2 * D_EF] += F::ONE;
        assert_error_without_panic(|| {
            owner.generate_traces(
                &fixture.proof,
                &opening,
                fixture.decomposition_end_tidx,
                fixture.eta,
                fixture.beta_claim,
                FixedMultiAirCompleteRegionCursorOpeningRequiredHeights::minimal(
                    owner.profile.protocol_component_count(),
                ),
            )
        });

        let mut count = fixture.transcript.clone();
        count.values_mut()[first.opening_start_tidx + D_EF] += F::ONE;
        assert_error_without_panic(|| {
            owner.generate_traces(
                &fixture.proof,
                &count,
                fixture.decomposition_end_tidx,
                fixture.eta,
                fixture.beta_claim,
                FixedMultiAirCompleteRegionCursorOpeningRequiredHeights::minimal(
                    owner.profile.protocol_component_count(),
                ),
            )
        });
        assert_error_without_panic(|| {
            owner.generate_traces(
                &fixture.proof,
                &fixture.transcript,
                fixture.decomposition_end_tidx + D_EF,
                fixture.eta,
                fixture.beta_claim,
                FixedMultiAirCompleteRegionCursorOpeningRequiredHeights::minimal(
                    owner.profile.protocol_component_count(),
                ),
            )
        });

        let mut dropped = fixture.proof.clone();
        dropped.local_proofs.pop();
        assert_error_without_panic(|| {
            owner.generate_traces(
                &dropped,
                &fixture.transcript,
                fixture.decomposition_end_tidx,
                fixture.eta,
                fixture.beta_claim,
                FixedMultiAirCompleteRegionCursorOpeningRequiredHeights::minimal(
                    owner.profile.protocol_component_count(),
                ),
            )
        });
        let mut reordered = fixture.proof.clone();
        reordered.local_proofs.swap(0, 2);
        assert_error_without_panic(|| {
            owner.generate_traces(
                &reordered,
                &fixture.transcript,
                fixture.decomposition_end_tidx,
                fixture.eta,
                fixture.beta_claim,
                FixedMultiAirCompleteRegionCursorOpeningRequiredHeights::minimal(
                    owner.profile.protocol_component_count(),
                ),
            )
        });
        let mut duplicated = fixture.proof.clone();
        duplicated.local_proofs[2] = duplicated.local_proofs[0].clone();
        assert_error_without_panic(|| {
            owner.generate_traces(
                &duplicated,
                &fixture.transcript,
                fixture.decomposition_end_tidx,
                fixture.eta,
                fixture.beta_claim,
                FixedMultiAirCompleteRegionCursorOpeningRequiredHeights::minimal(
                    owner.profile.protocol_component_count(),
                ),
            )
        });
        let mut forged_presence = fixture.proof.clone();
        forged_presence.interaction_proofs[2] = forged_presence.interaction_proofs[0].clone();
        assert_error_without_panic(|| {
            owner.generate_traces(
                &forged_presence,
                &fixture.transcript,
                fixture.decomposition_end_tidx,
                fixture.eta,
                fixture.beta_claim,
                FixedMultiAirCompleteRegionCursorOpeningRequiredHeights::minimal(
                    owner.profile.protocol_component_count(),
                ),
            )
        });
    }

    #[test]
    fn dropped_reordered_duplicated_and_cross_kind_plan_steps_are_rejected() {
        let fixture = backend_fixture();
        let mut dropped = (*fixture.plan).clone();
        dropped.transcript.remove(2);
        assert_error_without_panic(|| {
            FixedMultiAirCompleteRegionCursorOpeningProfile::from_plan(Arc::new(dropped))
        });

        let mut reordered = (*fixture.plan).clone();
        reordered.regions.swap(0, 1);
        assert_error_without_panic(|| {
            FixedMultiAirCompleteRegionCursorOpeningProfile::from_plan(Arc::new(reordered))
        });

        let mut duplicated = (*fixture.plan).clone();
        let duplicate = duplicated.transcript[2].clone();
        duplicated.transcript.insert(2, duplicate);
        assert_error_without_panic(|| {
            FixedMultiAirCompleteRegionCursorOpeningProfile::from_plan(Arc::new(duplicated))
        });

        let mut cross_kind = (*fixture.plan).clone();
        if let FixedMultiAirCompleteTerminalTranscriptStep::ObserveRegion {
            component, tag, ..
        } = &mut cross_kind.transcript[2]
        {
            *component = FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction;
            *tag = COMPLETE_TERMINAL_INTERACTION_REGION_TAG;
        } else {
            panic!("first regional transcript step");
        }
        assert_error_without_panic(|| {
            FixedMultiAirCompleteRegionCursorOpeningProfile::from_plan(Arc::new(cross_kind))
        });
    }

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct TranscriptOracleCols<T> {
        active: T,
        tidx: T,
        value: T,
        is_sample: T,
    }

    struct TranscriptOracleAir {
        bus: TranscriptBus,
    }

    impl BaseAir<F> for TranscriptOracleAir {
        fn width(&self) -> usize {
            TranscriptOracleCols::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for TranscriptOracleAir {}
    impl PartitionedBaseAir<F> for TranscriptOracleAir {}
    impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for TranscriptOracleAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("transcript oracle row");
            let local: &TranscriptOracleCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.active);
            builder.when(local.active).assert_bool(local.is_sample);
            self.bus.send(
                builder,
                AB::Expr::ZERO,
                TranscriptBusMessage {
                    tidx: local.tidx.into(),
                    value: local.value.into(),
                    is_sample: local.is_sample.into(),
                },
                local.active,
            );
        }
    }

    fn transcript_oracle_trace(
        transcript: &TranscriptLog<F, [F; 16]>,
        start: usize,
        end: usize,
    ) -> RowMajorMatrix<F> {
        let len = end.checked_sub(start).expect("ordered transcript range");
        let width = TranscriptOracleCols::<F>::width();
        let height = len.max(1).next_power_of_two();
        let mut values = vec![F::ZERO; height * width];
        for offset in 0..len {
            let cols: &mut TranscriptOracleCols<F> =
                values[offset * width..(offset + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.tidx = F::from_usize(start + offset);
            cols.value = transcript.values()[start + offset];
            cols.is_sample = F::from_bool(transcript.samples()[start + offset]);
        }
        RowMajorMatrix::new(values, width)
    }

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct SequenceOracleCols<T> {
        active: T,
        tidx: T,
        component_count: T,
    }

    struct SequenceOracleAir {
        bus: FixedMultiAirCompleteRegionSequenceStartBus,
    }

    impl BaseAir<F> for SequenceOracleAir {
        fn width(&self) -> usize {
            SequenceOracleCols::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for SequenceOracleAir {}
    impl PartitionedBaseAir<F> for SequenceOracleAir {}
    impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for SequenceOracleAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("sequence oracle row");
            let local: &SequenceOracleCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.active);
            self.bus.send(
                builder,
                FixedMultiAirCompleteRegionSequenceStartMessage {
                    decomposition_end_tidx: local.tidx.into(),
                    protocol_component_count: local.component_count.into(),
                },
                local.active,
            );
        }
    }

    fn sequence_oracle_trace(tidx: usize, component_count: usize) -> RowMajorMatrix<F> {
        let width = SequenceOracleCols::<F>::width();
        let mut values = vec![F::ZERO; width];
        let cols: &mut SequenceOracleCols<F> = values.as_mut_slice().borrow_mut();
        cols.active = F::ONE;
        cols.tidx = F::from_usize(tidx);
        cols.component_count = F::from_usize(component_count);
        RowMajorMatrix::new(values, width)
    }

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct DecompositionReceiptOracleCols<T> {
        active: T,
        eta: [T; D_EF],
        beta_claim: [T; D_EF],
        region_count: T,
    }

    struct DecompositionReceiptOracleAir {
        bus: FixedMultiAirCompleteDecompositionReceiptBus,
    }

    impl BaseAir<F> for DecompositionReceiptOracleAir {
        fn width(&self) -> usize {
            DecompositionReceiptOracleCols::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for DecompositionReceiptOracleAir {}
    impl PartitionedBaseAir<F> for DecompositionReceiptOracleAir {}
    impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for DecompositionReceiptOracleAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("decomposition receipt oracle row");
            let local: &DecompositionReceiptOracleCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.active);
            self.bus.send(
                builder,
                FixedMultiAirCompleteDecompositionReceiptMessage {
                    eta: local.eta.map(Into::into),
                    beta_claim: local.beta_claim.map(Into::into),
                    local_count: local.region_count.into(),
                    interaction_count: local.region_count.into(),
                },
                local.active,
            );
        }
    }

    fn decomposition_receipt_oracle_trace(
        eta: EF,
        beta_claim: EF,
        region_count: usize,
    ) -> RowMajorMatrix<F> {
        let width = DecompositionReceiptOracleCols::<F>::width();
        let mut values = vec![F::ZERO; width];
        let cols: &mut DecompositionReceiptOracleCols<F> = values.as_mut_slice().borrow_mut();
        cols.active = F::ONE;
        copy_ext(&mut cols.eta, eta);
        copy_ext(&mut cols.beta_claim, beta_claim);
        cols.region_count = F::from_usize(region_count);
        RowMajorMatrix::new(values, width)
    }

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct ClaimOracleCols<T> {
        active: T,
        is_interaction: T,
        region: T,
        claim: [T; D_EF],
    }

    struct ClaimOracleAir {
        local: FixedMultiAirCompleteLocalClaimBus,
        interaction: FixedMultiAirCompleteInteractionClaimBus,
    }

    impl BaseAir<F> for ClaimOracleAir {
        fn width(&self) -> usize {
            ClaimOracleCols::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for ClaimOracleAir {}
    impl PartitionedBaseAir<F> for ClaimOracleAir {}
    impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for ClaimOracleAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("claim oracle row");
            let local: &ClaimOracleCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.active);
            builder.when(local.active).assert_bool(local.is_interaction);
            let message = FixedMultiAirCompleteRegionalClaimMessage {
                region: local.region.into(),
                claim: local.claim.map(Into::into),
            };
            self.local.add_key_with_lookups(
                builder,
                message.clone(),
                local.active * (AB::Expr::ONE - local.is_interaction),
            );
            self.interaction.add_key_with_lookups(
                builder,
                message,
                local.active * local.is_interaction,
            );
        }
    }

    fn claim_oracle_trace(proof: &FixedMultiAirCompleteTerminalProof<EF>) -> RowMajorMatrix<F> {
        let rows = proof.local_claims.len() + proof.interaction_claims.len();
        let width = ClaimOracleCols::<F>::width();
        let height = rows.next_power_of_two();
        let mut values = vec![F::ZERO; height * width];
        for (row, (is_interaction, region, claim)) in proof
            .local_claims
            .iter()
            .copied()
            .enumerate()
            .map(|(region, claim)| (false, region, claim))
            .chain(
                proof
                    .interaction_claims
                    .iter()
                    .copied()
                    .enumerate()
                    .map(|(region, claim)| (true, region, claim)),
            )
            .enumerate()
        {
            let cols: &mut ClaimOracleCols<F> = values[row * width..(row + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.is_interaction = F::from_bool(is_interaction);
            cols.region = F::from_usize(region);
            copy_ext(&mut cols.claim, claim);
        }
        RowMajorMatrix::new(values, width)
    }

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct PointOracleCols<T> {
        active: T,
        region: T,
        coordinate: T,
        value: [T; D_EF],
        count: T,
    }

    #[derive(Clone, Copy)]
    enum PointOracleBus {
        Local(FixedMultiAirCompleteLocalRegionPointBus),
        Interaction(FixedMultiAirCompleteInteractionRegionPointBus),
    }

    struct PointOracleAir {
        bus: PointOracleBus,
    }

    impl BaseAir<F> for PointOracleAir {
        fn width(&self) -> usize {
            PointOracleCols::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for PointOracleAir {}
    impl PartitionedBaseAir<F> for PointOracleAir {}
    impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for PointOracleAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("point oracle row");
            let local: &PointOracleCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.active);
            let message = FixedMultiAirCompleteRegionPointMessage {
                region: local.region.into(),
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            };
            match self.bus {
                PointOracleBus::Local(bus) => {
                    bus.lookup_key(builder, message, local.active * local.count)
                }
                PointOracleBus::Interaction(bus) => {
                    bus.lookup_key(builder, message, local.active * local.count)
                }
            }
        }
    }

    fn point_lookup_count(
        profile: &FixedMultiAirCompleteRegionCursorComponentProfile,
        round: usize,
    ) -> usize {
        let exponent = profile.round_count - round - 1;
        profile.point_fixed_source_count * (1usize << exponent) + profile.point_common_count
    }

    fn round_challenge(transcript: &TranscriptLog<F, [F; 16]>, start: usize, round: usize) -> EF {
        let tidx = start
            + FIXED_MULTI_AIR_COMPLETE_REGION_PREFIX_BASE_VALUES
            + round * FIXED_MULTI_AIR_COMPLETE_ROUND_TRANSCRIPT_BASE_VALUES
            + (3 + FIXED_MULTI_AIR_COMPLETE_SUMCHECK_EVALUATIONS) * D_EF;
        read_ext(transcript, tidx, true).expect("regional challenge")
    }

    fn point_oracle_trace(
        profiles: &[Arc<FixedMultiAirCompleteRegionCursorComponentProfile>],
        boundaries: &[FixedMultiAirCompleteRegionCursorBoundary],
        transcript: &TranscriptLog<F, [F; 16]>,
        kind: FixedMultiAirCompleteTerminalCircuitComponentKind,
    ) -> RowMajorMatrix<F> {
        let rows = profiles
            .iter()
            .filter(|profile| profile.kind == kind)
            .map(|profile| profile.round_count)
            .sum::<usize>();
        let width = PointOracleCols::<F>::width();
        let height = rows.max(1).next_power_of_two();
        let mut values = vec![F::ZERO; height * width];
        let mut row = 0usize;
        for (profile, boundary) in profiles.iter().zip(boundaries) {
            if profile.kind != kind {
                continue;
            }
            for round in 0..profile.round_count {
                let cols: &mut PointOracleCols<F> =
                    values[row * width..(row + 1) * width].borrow_mut();
                cols.active = F::ONE;
                cols.region = F::from_usize(profile.region);
                cols.coordinate = F::from_usize(round);
                copy_ext(
                    &mut cols.value,
                    round_challenge(transcript, boundary.component_start_tidx, round),
                );
                cols.count = F::from_usize(point_lookup_count(profile, round));
                row += 1;
            }
        }
        RowMajorMatrix::new(values, width)
    }

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct OpeningOracleCols<T> {
        active: T,
        region: T,
        regional_opening: T,
        global_opening: T,
        value: [T; D_EF],
        count: T,
    }

    #[derive(Clone, Copy)]
    enum OpeningOracleBus {
        Local(FixedMultiAirCompleteLocalRegionOpeningBus),
        Interaction(FixedMultiAirCompleteInteractionRegionOpeningBus),
    }

    struct OpeningOracleAir {
        bus: OpeningOracleBus,
    }

    impl BaseAir<F> for OpeningOracleAir {
        fn width(&self) -> usize {
            OpeningOracleCols::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for OpeningOracleAir {}
    impl PartitionedBaseAir<F> for OpeningOracleAir {}
    impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for OpeningOracleAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("opening oracle row");
            let local: &OpeningOracleCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.active);
            let message = FixedMultiAirCompleteRegionOpeningMessage {
                region: local.region.into(),
                regional_opening: local.regional_opening.into(),
                global_opening: local.global_opening.into(),
                value: local.value.map(Into::into),
            };
            match self.bus {
                OpeningOracleBus::Local(bus) => {
                    bus.lookup_key(builder, message, local.active * local.count)
                }
                OpeningOracleBus::Interaction(bus) => {
                    bus.lookup_key(builder, message, local.active * local.count)
                }
            }
        }
    }

    fn opening_oracle_trace(
        profiles: &[Arc<FixedMultiAirCompleteRegionCursorComponentProfile>],
        proof: &FixedMultiAirCompleteTerminalProof<EF>,
        kind: FixedMultiAirCompleteTerminalCircuitComponentKind,
    ) -> RowMajorMatrix<F> {
        let rows = profiles
            .iter()
            .filter(|profile| profile.kind == kind)
            .map(|profile| profile.opening_count)
            .sum::<usize>();
        let width = OpeningOracleCols::<F>::width();
        let height = rows.max(1).next_power_of_two();
        let mut values = vec![F::ZERO; height * width];
        let mut row = 0usize;
        for profile in profiles.iter().filter(|profile| profile.kind == kind) {
            let (_, regional) = component_proof(proof, profile).expect("component proof");
            for (opening, &value) in regional.opened_columns.iter().enumerate() {
                let cols: &mut OpeningOracleCols<F> =
                    values[row * width..(row + 1) * width].borrow_mut();
                cols.active = F::ONE;
                cols.region = F::from_usize(profile.region);
                cols.regional_opening = F::from_usize(opening);
                cols.global_opening = F::from_usize(profile.opening_ordinal_start + opening);
                copy_ext(&mut cols.value, value);
                cols.count = F::from_u32(profile.opening_lookup_counts[opening]);
                row += 1;
            }
        }
        RowMajorMatrix::new(values, width)
    }

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct FinalOracleCols<T> {
        active: T,
        region: T,
        claim: [T; D_EF],
    }

    #[derive(Clone, Copy)]
    enum FinalOracleBus {
        Local(FixedMultiAirCompleteLocalRegionFinalEvaluationBus),
        Interaction(FixedMultiAirCompleteInteractionRegionFinalEvaluationBus),
    }

    struct FinalOracleAir {
        bus: FinalOracleBus,
    }

    impl BaseAir<F> for FinalOracleAir {
        fn width(&self) -> usize {
            FinalOracleCols::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for FinalOracleAir {}
    impl PartitionedBaseAir<F> for FinalOracleAir {}
    impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for FinalOracleAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("final oracle row");
            let local: &FinalOracleCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.active);
            let message = FixedMultiAirCompleteRegionFinalEvaluationMessage {
                region: local.region.into(),
                claim: local.claim.map(Into::into),
            };
            match self.bus {
                FinalOracleBus::Local(bus) => bus.receive(builder, message, local.active),
                FinalOracleBus::Interaction(bus) => bus.receive(builder, message, local.active),
            }
        }
    }

    fn final_oracle_trace(
        boundaries: &[FixedMultiAirCompleteRegionCursorBoundary],
        kind: FixedMultiAirCompleteTerminalCircuitComponentKind,
    ) -> RowMajorMatrix<F> {
        let selected = boundaries
            .iter()
            .filter(|boundary| boundary.kind == kind)
            .collect::<Vec<_>>();
        let width = FinalOracleCols::<F>::width();
        let height = selected.len().max(1).next_power_of_two();
        let mut values = vec![F::ZERO; height * width];
        for (row, boundary) in selected.into_iter().enumerate() {
            let cols: &mut FinalOracleCols<F> = values[row * width..(row + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.region = F::from_usize(boundary.region);
            copy_ext(&mut cols.claim, boundary.final_evaluation);
        }
        RowMajorMatrix::new(values, width)
    }

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct BatchOracleCols<T> {
        active: T,
        tidx: T,
        component_count: T,
        opening_count: T,
    }

    struct BatchOracleAir {
        bus: FixedMultiAirCompleteOpeningBatchStartBus,
    }

    impl BaseAir<F> for BatchOracleAir {
        fn width(&self) -> usize {
            BatchOracleCols::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for BatchOracleAir {}
    impl PartitionedBaseAir<F> for BatchOracleAir {}
    impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for BatchOracleAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("batch oracle row");
            let local: &BatchOracleCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.active);
            self.bus.receive(
                builder,
                FixedMultiAirCompleteOpeningBatchStartMessage {
                    tidx: local.tidx.into(),
                    protocol_component_count: local.component_count.into(),
                    global_opening_count: local.opening_count.into(),
                },
                local.active,
            );
        }
    }

    fn batch_oracle_trace(
        tidx: usize,
        component_count: usize,
        opening_count: usize,
    ) -> RowMajorMatrix<F> {
        let width = BatchOracleCols::<F>::width();
        let mut values = vec![F::ZERO; width];
        let cols: &mut BatchOracleCols<F> = values.as_mut_slice().borrow_mut();
        cols.active = F::ONE;
        cols.tidx = F::from_usize(tidx);
        cols.component_count = F::from_usize(component_count);
        cols.opening_count = F::from_usize(opening_count);
        RowMajorMatrix::new(values, width)
    }

    fn symbolic_interactions<A>(air: &A) -> Vec<SymbolicInteraction<F>>
    where
        A: Air<SymbolicRapBuilder<F>>
            + BaseAir<F>
            + BaseAirWithPublicValues<F>
            + PartitionedBaseAir<F>,
    {
        get_symbolic_builder(
            air,
            &TraceWidth {
                preprocessed: None,
                cached_mains: air.cached_main_widths(),
                common_main: air.common_main_width(),
            },
        )
        .constraints()
        .interactions
    }

    struct OwnedInteractionCase<'a> {
        name: String,
        interactions: Vec<SymbolicInteraction<F>>,
        matrices: Vec<p3_matrix::dense::RowMajorMatrixView<'a, F>>,
    }

    fn check_cases(cases: &[OwnedInteractionCase<'_>]) {
        let names = cases
            .iter()
            .map(|case| case.name.clone())
            .collect::<Vec<_>>();
        let interactions = cases
            .iter()
            .map(|case| case.interactions.clone())
            .collect::<Vec<_>>();
        let matrices = cases
            .iter()
            .map(|case| case.matrices.clone())
            .collect::<Vec<_>>();
        let preprocessed = vec![None; cases.len()];
        let publics = vec![Vec::new(); cases.len()];
        check_logup(&names, &interactions, &preprocessed, &matrices, &publics);
    }

    #[test]
    fn full_typed_interaction_balance_and_cross_routing_rejection() {
        assert_ne!(
            TypeId::of::<FixedMultiAirCompleteLocalRegionOpeningBus>(),
            TypeId::of::<FixedMultiAirCompleteInteractionRegionOpeningBus>()
        );
        let fixture = backend_fixture();
        let owner = owner(&fixture);
        let traces = traces(&fixture, &owner);
        let buses = buses();

        let transcript_air = TranscriptOracleAir {
            bus: buses.transcript,
        };
        let transcript_trace = transcript_oracle_trace(
            &fixture.transcript,
            fixture.decomposition_end_tidx,
            traces.batch_start_tidx,
        );
        let sequence_air = SequenceOracleAir {
            bus: buses.sequence_start,
        };
        let sequence_trace = sequence_oracle_trace(
            fixture.decomposition_end_tidx,
            owner.profile.protocol_component_count(),
        );
        let decomposition_receipt_air = DecompositionReceiptOracleAir {
            bus: buses.decomposition_receipt,
        };
        let decomposition_receipt_trace = decomposition_receipt_oracle_trace(
            fixture.eta,
            fixture.beta_claim,
            owner.profile.plan.regions.len(),
        );
        let claim_air = ClaimOracleAir {
            local: buses.local_claim,
            interaction: buses.interaction_claim,
        };
        let claim_trace = claim_oracle_trace(&fixture.proof);
        let local_point_air = PointOracleAir {
            bus: PointOracleBus::Local(buses.authorities.local_point),
        };
        let interaction_point_air = PointOracleAir {
            bus: PointOracleBus::Interaction(buses.authorities.interaction_point),
        };
        let local_point_trace = point_oracle_trace(
            &owner.profile.components,
            &traces.boundaries,
            &fixture.transcript,
            FixedMultiAirCompleteTerminalCircuitComponentKind::Local,
        );
        let interaction_point_trace = point_oracle_trace(
            &owner.profile.components,
            &traces.boundaries,
            &fixture.transcript,
            FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction,
        );
        let local_opening_air = OpeningOracleAir {
            bus: OpeningOracleBus::Local(buses.authorities.local_opening),
        };
        let interaction_opening_air = OpeningOracleAir {
            bus: OpeningOracleBus::Interaction(buses.authorities.interaction_opening),
        };
        let local_opening_trace = opening_oracle_trace(
            &owner.profile.components,
            &fixture.proof,
            FixedMultiAirCompleteTerminalCircuitComponentKind::Local,
        );
        let interaction_opening_trace = opening_oracle_trace(
            &owner.profile.components,
            &fixture.proof,
            FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction,
        );
        let local_final_air = FinalOracleAir {
            bus: FinalOracleBus::Local(buses.authorities.local_final_evaluation),
        };
        let interaction_final_air = FinalOracleAir {
            bus: FinalOracleBus::Interaction(buses.authorities.interaction_final_evaluation),
        };
        let local_final_trace = final_oracle_trace(
            &traces.boundaries,
            FixedMultiAirCompleteTerminalCircuitComponentKind::Local,
        );
        let interaction_final_trace = final_oracle_trace(
            &traces.boundaries,
            FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction,
        );
        let batch_air = BatchOracleAir {
            bus: buses.batch_start,
        };
        let batch_trace = batch_oracle_trace(
            traces.batch_start_tidx,
            owner.profile.protocol_component_count(),
            traces.global_opening_count,
        );

        let mut local_sumchecks = Vec::new();
        let mut interaction_sumchecks = Vec::new();
        let mut sumcheck_traces = Vec::new();
        for (profile, boundary) in owner.profile.components.iter().zip(&traces.boundaries) {
            let (claim, proof) =
                component_proof(&fixture.proof, profile).expect("regional sumcheck witness");
            match profile.kind {
                FixedMultiAirCompleteTerminalCircuitComponentKind::Local => {
                    let air = FixedMultiAirCompleteLocalRegionSumcheckAir::new(
                        buses.transcript,
                        buses.local_start,
                        buses.authorities.local_point,
                        buses.local_sumcheck_final,
                        profile.region,
                        profile.round_count,
                        profile.opening_count,
                        profile.point_fixed_source_count,
                        profile.point_common_count,
                    )
                    .expect("local sumcheck AIR");
                    let trace = generate_fixed_multi_air_complete_local_region_sumcheck_trace(
                        &air,
                        claim,
                        proof,
                        &fixture.transcript,
                        boundary.component_start_tidx,
                        None,
                    )
                    .expect("local sumcheck trace");
                    local_sumchecks.push(air);
                    sumcheck_traces.push(trace);
                }
                FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction => {
                    let air = FixedMultiAirCompleteInteractionRegionSumcheckAir::new(
                        buses.transcript,
                        buses.interaction_start,
                        buses.authorities.interaction_point,
                        buses.interaction_sumcheck_final,
                        profile.region,
                        profile.round_count,
                        profile.opening_count,
                        profile.point_fixed_source_count,
                        profile.point_common_count,
                    )
                    .expect("interaction sumcheck AIR");
                    let trace =
                        generate_fixed_multi_air_complete_interaction_region_sumcheck_trace(
                            &air,
                            claim,
                            proof,
                            &fixture.transcript,
                            boundary.component_start_tidx,
                            None,
                        )
                        .expect("interaction sumcheck trace");
                    interaction_sumchecks.push(air);
                    sumcheck_traces.push(trace);
                }
            }
        }

        let mut wrong_count_cached = traces.components[0].cached.clone();
        let opening_width = FixedMultiAirCompleteRegionOpeningScheduleCols::<F>::width();
        let wrong_count_schedule: &mut FixedMultiAirCompleteRegionOpeningScheduleCols<F> =
            wrong_count_cached.values[2 * opening_width..3 * opening_width].borrow_mut();
        wrong_count_schedule.opening_lookup_count += F::ONE;

        // Keep every trace and AIR owner alive while `check_logup` borrows its
        // matrix views. The order of local/interaction sumcheck vectors does
        // not carry protocol authority; the typed cursor/start/final buses do.
        let mut cases = Vec::new();
        cases.push(OwnedInteractionCase {
            name: "sequence-owner".to_string(),
            interactions: symbolic_interactions(&owner.sequence),
            matrices: vec![
                traces.sequence.cached.as_view(),
                traces.sequence.common.as_view(),
            ],
        });
        for (index, (air, trace)) in owner.components.iter().zip(&traces.components).enumerate() {
            cases.push(OwnedInteractionCase {
                name: format!("opening-tail-{index}"),
                interactions: symbolic_interactions(air),
                matrices: vec![trace.cached.as_view(), trace.common.as_view()],
            });
        }
        let mut local_index = 0usize;
        let mut interaction_index = 0usize;
        for (profile, trace) in owner.profile.components.iter().zip(&sumcheck_traces) {
            match profile.kind {
                FixedMultiAirCompleteTerminalCircuitComponentKind::Local => {
                    let air = &local_sumchecks[local_index];
                    local_index += 1;
                    cases.push(OwnedInteractionCase {
                        name: format!("local-sumcheck-{}", profile.region),
                        interactions: symbolic_interactions(air),
                        matrices: vec![trace.as_view()],
                    });
                }
                FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction => {
                    let air = &interaction_sumchecks[interaction_index];
                    interaction_index += 1;
                    cases.push(OwnedInteractionCase {
                        name: format!("interaction-sumcheck-{}", profile.region),
                        interactions: symbolic_interactions(air),
                        matrices: vec![trace.as_view()],
                    });
                }
            }
        }
        for (name, interactions, matrix) in [
            (
                "transcript-oracle",
                symbolic_interactions(&transcript_air),
                transcript_trace.as_view(),
            ),
            (
                "sequence-oracle",
                symbolic_interactions(&sequence_air),
                sequence_trace.as_view(),
            ),
            (
                "decomposition-receipt-oracle",
                symbolic_interactions(&decomposition_receipt_air),
                decomposition_receipt_trace.as_view(),
            ),
            (
                "claim-oracle",
                symbolic_interactions(&claim_air),
                claim_trace.as_view(),
            ),
            (
                "local-point-oracle",
                symbolic_interactions(&local_point_air),
                local_point_trace.as_view(),
            ),
            (
                "interaction-point-oracle",
                symbolic_interactions(&interaction_point_air),
                interaction_point_trace.as_view(),
            ),
            (
                "local-opening-oracle",
                symbolic_interactions(&local_opening_air),
                local_opening_trace.as_view(),
            ),
            (
                "interaction-opening-oracle",
                symbolic_interactions(&interaction_opening_air),
                interaction_opening_trace.as_view(),
            ),
            (
                "local-final-oracle",
                symbolic_interactions(&local_final_air),
                local_final_trace.as_view(),
            ),
            (
                "interaction-final-oracle",
                symbolic_interactions(&interaction_final_air),
                interaction_final_trace.as_view(),
            ),
            (
                "batch-oracle",
                symbolic_interactions(&batch_air),
                batch_trace.as_view(),
            ),
        ] {
            cases.push(OwnedInteractionCase {
                name: name.to_string(),
                interactions,
                matrices: vec![matrix],
            });
        }
        check_cases(&cases);

        // A setup-row multiplicity mutation must unbalance the independently
        // derived endpoint/global consumers even though all opening values
        // and transcript indices remain honest.
        let opening_tail = cases
            .iter()
            .position(|case| case.name == "opening-tail-0")
            .expect("first opening-tail case");
        let honest_cached = cases[opening_tail].matrices[0].clone();
        cases[opening_tail].matrices[0] = wrong_count_cached.as_view();
        assert!(catch_unwind(AssertUnwindSafe(|| check_cases(&cases))).is_err());
        cases[opening_tail].matrices[0] = honest_cached;

        // A valid local opening routed onto the interaction bus must leave
        // both typed multisets unbalanced.
        let cross_opening_air = OpeningOracleAir {
            bus: OpeningOracleBus::Interaction(buses.authorities.interaction_opening),
        };
        let original = cases
            .iter()
            .position(|case| case.name == "local-opening-oracle")
            .expect("local opening oracle case");
        cases[original].interactions = symbolic_interactions(&cross_opening_air);
        assert!(catch_unwind(AssertUnwindSafe(|| check_cases(&cases))).is_err());
    }
}

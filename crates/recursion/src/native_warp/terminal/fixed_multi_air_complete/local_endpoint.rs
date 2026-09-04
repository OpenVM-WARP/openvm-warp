//! Recursive endpoint for one genuine complete-terminal local AIR region.
//!
//! This module deliberately consumes the canonical setup-only circuit plan
//! emitted by `FixedMultiAirCompleteTerminalLinearizer::circuit_plan`.  It
//! never reconstructs an AIR layout from a verifying key and never treats a
//! PCS opening as a PESAT predicate.  The proof-controlled values are linked
//! through typed buses; all schedules and fixed tables are proving-key data.

use core::borrow::{Borrow, BorrowMut};
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
        DirectAirFixedTrace, FixedMultiAirCompleteCachedTrace,
        FixedMultiAirCompleteTerminalCircuitComponentKind,
        FixedMultiAirCompleteTerminalCircuitDynamicKind,
        FixedMultiAirCompleteTerminalCircuitFixedColumn,
        FixedMultiAirCompleteTerminalCircuitNodeSource, FixedMultiAirCompleteTerminalCircuitPlan,
        FixedMultiAirCompleteTerminalConstraintPlan, FixedMultiAirCompleteTerminalEqBlockPlan,
        FixedMultiAirCompleteTerminalEqRole,
    },
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing, PrimeField32,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    define_typed_lookup_bus, define_typed_permutation_bus,
    native_warp::terminal::fixed_multi_air_complete::{
        FixedMultiAirCompleteEqWeightBus, FixedMultiAirCompleteEqWeightMessage,
        FixedMultiAirCompleteInstanceValueBus, FixedMultiAirCompleteInstanceValueMessage,
        FixedMultiAirCompleteLocalRegionFinalEvaluationBus,
        FixedMultiAirCompleteLocalRegionOpeningBus, FixedMultiAirCompleteLocalRegionPointBus,
        FixedMultiAirCompleteRegionFinalEvaluationMessage,
        FixedMultiAirCompleteRegionOpeningMessage, FixedMultiAirCompleteRegionPointMessage,
        FixedMultiAirCompleteRegionalPointBus, FixedMultiAirCompleteRegionalPointMessage,
    },
    utils::{ext_field_add, ext_field_multiply, ext_field_subtract},
};

const INSTANCE_SECTION_BETA: usize = 2;
const LOCAL_EQ_ROLE: usize = 0;
const SELECTOR_COUNT: usize = 3;
const SELECTOR_FIRST: usize = 0;
const SELECTOR_LAST: usize = 1;
const SELECTOR_TRANSITION: usize = 2;

// The complete relation supports exact degree up to six. The backend uses a
// fixed degree-seven regional sumcheck envelope; the plan's exact relation
// degree remains the homogeneous DAG scaling exponent.
const MAX_RELATION_DEGREE: usize = 6;

const NODE_KIND_COUNT: usize = 6;
const NODE_SOURCE: usize = 0;
const NODE_CONSTANT: usize = 1;
const NODE_ADD: usize = 2;
const NODE_SUB: usize = 3;
const NODE_NEG: usize = 4;
const NODE_MUL: usize = 5;

const SOURCE_KIND_COUNT: usize = 3;
const SOURCE_OPENING: usize = 0;
const SOURCE_FIXED: usize = 1;
const SOURCE_INSTANCE: usize = 2;

const FIXED_KIND_COUNT: usize = 3;
const FIXED_INITIAL: usize = 0;
const FIXED_FOLD: usize = 1;
const FIXED_FINAL: usize = 2;

const _: () = assert!(D_EF == 4);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedMultiAirCompleteLocalEndpointError {
    Plan(&'static str),
    Setup(&'static str),
    Proof(&'static str),
    Overflow(&'static str),
    Allocation,
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteLocalFixedValueMessage<T> {
    pub region: T,
    pub source: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirCompleteLocalFixedValueBus,
    FixedMultiAirCompleteLocalFixedValueMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteLocalFixedStateMessage<T> {
    pub region: T,
    pub source: T,
    pub layer: T,
    pub index: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirCompleteLocalFixedStateBus,
    FixedMultiAirCompleteLocalFixedStateMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteLocalNodeMessage<T> {
    pub region: T,
    pub node: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirCompleteLocalNodeBus,
    FixedMultiAirCompleteLocalNodeMessage
);

#[derive(Clone, Copy, Debug)]
pub struct FixedMultiAirCompleteLocalEndpointBuses {
    pub point: FixedMultiAirCompleteLocalRegionPointBus,
    /// Internal fanout from the authenticated local point to every local
    /// constraint BlockEq owner.
    pub block_eq_point: FixedMultiAirCompleteRegionalPointBus,
    pub opening: FixedMultiAirCompleteLocalRegionOpeningBus,
    pub final_evaluation: FixedMultiAirCompleteLocalRegionFinalEvaluationBus,
    pub instance: FixedMultiAirCompleteInstanceValueBus,
    pub eq_weight: FixedMultiAirCompleteEqWeightBus,
    pub fixed_state: FixedMultiAirCompleteLocalFixedStateBus,
    pub fixed_value: FixedMultiAirCompleteLocalFixedValueBus,
    pub node: FixedMultiAirCompleteLocalNodeBus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum LocalNodeSource {
    Opening(usize),
    Fixed(usize),
    Instance(usize),
}

/// Setup-only profile for one local region. Every ordinal is copied from the
/// canonical complete-terminal circuit plan or derived from its fixed
/// metadata by a checked formula used by the backend.
#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteLocalEndpointProfile {
    pub region: usize,
    pub setup_ordinal: usize,
    pub log_height: usize,
    pub exact_relation_degree: usize,
    pub one_beta_coordinate: usize,
    pub component:
        openvm_stark_backend::native_warp::FixedMultiAirCompleteTerminalCircuitComponentPlan<F>,
    node_sources: Vec<Option<LocalNodeSource>>,
    node_fanout: Vec<usize>,
    block_eq_point_blocks: Vec<usize>,
    opening_lookup_counts: Vec<usize>,
    fixed_lookup_counts: Vec<usize>,
    instance_beta_lookup_counts: Vec<usize>,
}

impl FixedMultiAirCompleteLocalEndpointProfile {
    pub fn from_circuit_plan(
        plan: &FixedMultiAirCompleteTerminalCircuitPlan<F, Digest>,
        region: usize,
    ) -> Result<Self, FixedMultiAirCompleteLocalEndpointError> {
        let region_plan =
            plan.regions
                .get(region)
                .ok_or(FixedMultiAirCompleteLocalEndpointError::Plan(
                    "local region ordinal",
                ))?;
        let component = &region_plan.local;
        if component.identity.kind != FixedMultiAirCompleteTerminalCircuitComponentKind::Local
            || usize::try_from(component.identity.region_ordinal).ok() != Some(region)
            || usize::try_from(component.identity.claim_ordinal).ok() != Some(region)
            || component
                .identity
                .proof_ordinal
                .and_then(|x| usize::try_from(x).ok())
                != Some(region)
            || !component.identity.proof_present
            || !component.interactions.is_empty()
            || usize::try_from(region_plan.setup_ordinal).ok() != Some(region)
        {
            return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                "local component identity",
            ));
        }

        let log_height = usize::from(region_plan.log_height);
        let height = checked_pow2(log_height, "local region height")?;
        let exact_relation_degree = usize::from(plan.metadata.exact_relation_degree);
        if exact_relation_degree == 0 || exact_relation_degree > MAX_RELATION_DEGREE {
            return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                "complete relation degree",
            ));
        }
        if usize::from(plan.metadata.terminal_round_degree) != MAX_RELATION_DEGREE + 1 {
            return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                "complete regional degree",
            ));
        }
        let expression = &component.expression;
        if expression.nodes.is_empty()
            || expression.nodes.len() != expression.homogeneous_degrees.len()
            || expression.nodes.len() != expression.node_sources.len()
            || component.constraints.is_empty()
            || usize::try_from(component.identity.opening_count).ok()
                != Some(expression.dynamic_columns.len())
        {
            return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                "local expression dimensions",
            ));
        }
        validate_sorted_unique(&expression.dynamic_columns, "dynamic source order")?;
        validate_sorted_unique(&expression.fixed_columns, "fixed source order")?;

        let metadata_region = plan.metadata.regions.get(region).ok_or(
            FixedMultiAirCompleteLocalEndpointError::Plan("local metadata region"),
        )?;
        if metadata_region.setup_ordinal != region_plan.setup_ordinal
            || metadata_region.air_id != region_plan.air_id
            || metadata_region.log_height != region_plan.log_height
        {
            return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                "local metadata identity",
            ));
        }
        let trace_start = usize::try_from(metadata_region.trace_message.start)
            .map_err(|_| FixedMultiAirCompleteLocalEndpointError::Overflow("trace start"))?;

        let beta_len =
            usize::from(plan.metadata.log_constraints)
                .checked_add(usize::try_from(plan.metadata.explicit_len).map_err(|_| {
                    FixedMultiAirCompleteLocalEndpointError::Overflow("beta length")
                })?)
                .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(
                    "beta length",
                ))?;
        let one_beta_coordinate = usize::from(plan.metadata.log_constraints)
            .checked_add(
                usize::try_from(plan.scalars.distinguished_one_explicit_offset).map_err(|_| {
                    FixedMultiAirCompleteLocalEndpointError::Overflow("one coordinate")
                })?,
            )
            .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(
                "one coordinate",
            ))?;
        if one_beta_coordinate >= beta_len {
            return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                "distinguished-one coordinate",
            ));
        }

        let mut node_sources = Vec::with_capacity(expression.nodes.len());
        let mut opening_lookup_counts = vec![0usize; expression.dynamic_columns.len()];
        let fixed_source_count = expression
            .fixed_columns
            .len()
            .checked_add(SELECTOR_COUNT)
            .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(
                "fixed source count",
            ))?;
        let mut fixed_lookup_counts = vec![0usize; fixed_source_count];
        let mut instance_beta_lookup_counts = vec![0usize; beta_len];

        for (node_index, node) in expression.nodes.iter().enumerate() {
            let degree = usize::from(expression.homogeneous_degrees[node_index]);
            if degree > exact_relation_degree {
                return Err(FixedMultiAirCompleteLocalEndpointError::Plan("node degree"));
            }
            let plan_source = expression.node_sources[node_index];
            let source = match node {
                SymbolicExpressionNode::Variable(variable) => match variable.entry {
                    Entry::Main { part_index, offset } => {
                        if offset > 1 {
                            return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                                "main rotation",
                            ));
                        }
                        match plan_source {
                            Some(FixedMultiAirCompleteTerminalCircuitNodeSource::Dynamic(
                                ordinal,
                            )) => {
                                let ordinal = checked_u32(ordinal, "dynamic source ordinal")?;
                                let descriptor = expression.dynamic_columns.get(ordinal).ok_or(
                                    FixedMultiAirCompleteLocalEndpointError::Plan(
                                        "dynamic source ordinal",
                                    ),
                                )?;
                                let expected_start = trace_start
                                    .checked_add(variable.index.checked_mul(height).ok_or(
                                        FixedMultiAirCompleteLocalEndpointError::Overflow(
                                            "dynamic source start",
                                        ),
                                    )?)
                                    .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(
                                        "dynamic source start",
                                    ))?;
                                if descriptor.kind
                                    != FixedMultiAirCompleteTerminalCircuitDynamicKind::Trace
                                    || usize::try_from(descriptor.start).ok()
                                        != Some(expected_start)
                                    || usize::from(descriptor.log_height) != log_height
                                    || usize::from(descriptor.rotation) != offset
                                {
                                    return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                                        "dynamic main source",
                                    ));
                                }
                                opening_lookup_counts[ordinal] = opening_lookup_counts[ordinal]
                                    .checked_add(1)
                                    .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(
                                        "opening lookup count",
                                    ))?;
                                Some(LocalNodeSource::Opening(ordinal))
                            }
                            Some(FixedMultiAirCompleteTerminalCircuitNodeSource::Fixed(
                                ordinal,
                            )) => {
                                let ordinal = checked_u32(ordinal, "cached source ordinal")?;
                                let descriptor = expression.fixed_columns.get(ordinal).ok_or(
                                    FixedMultiAirCompleteLocalEndpointError::Plan(
                                        "cached source ordinal",
                                    ),
                                )?;
                                if *descriptor
                                    != (FixedMultiAirCompleteTerminalCircuitFixedColumn::Cached {
                                        part: u32::try_from(part_index).map_err(|_| {
                                            FixedMultiAirCompleteLocalEndpointError::Overflow(
                                                "cached part",
                                            )
                                        })?,
                                        column: u32::try_from(variable.index).map_err(|_| {
                                            FixedMultiAirCompleteLocalEndpointError::Overflow(
                                                "cached column",
                                            )
                                        })?,
                                        rotation: u8::try_from(offset).map_err(|_| {
                                            FixedMultiAirCompleteLocalEndpointError::Overflow(
                                                "cached rotation",
                                            )
                                        })?,
                                    })
                                {
                                    return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                                        "cached main source",
                                    ));
                                }
                                fixed_lookup_counts[ordinal] = fixed_lookup_counts[ordinal]
                                    .checked_add(1)
                                    .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(
                                        "fixed lookup count",
                                    ))?;
                                Some(LocalNodeSource::Fixed(ordinal))
                            }
                            None => {
                                return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                                    "missing main source",
                                ))
                            }
                        }
                    }
                    Entry::Preprocessed { offset } => {
                        if offset > 1 {
                            return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                                "preprocessed rotation",
                            ));
                        }
                        let Some(FixedMultiAirCompleteTerminalCircuitNodeSource::Fixed(ordinal)) =
                            plan_source
                        else {
                            return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                                "preprocessed source kind",
                            ));
                        };
                        let ordinal = checked_u32(ordinal, "preprocessed source ordinal")?;
                        let descriptor = expression.fixed_columns.get(ordinal).ok_or(
                            FixedMultiAirCompleteLocalEndpointError::Plan(
                                "preprocessed source ordinal",
                            ),
                        )?;
                        if *descriptor
                            != (FixedMultiAirCompleteTerminalCircuitFixedColumn::Preprocessed {
                                column: u32::try_from(variable.index).map_err(|_| {
                                    FixedMultiAirCompleteLocalEndpointError::Overflow(
                                        "preprocessed column",
                                    )
                                })?,
                                rotation: u8::try_from(offset).map_err(|_| {
                                    FixedMultiAirCompleteLocalEndpointError::Overflow(
                                        "preprocessed rotation",
                                    )
                                })?,
                            })
                        {
                            return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                                "preprocessed source",
                            ));
                        }
                        fixed_lookup_counts[ordinal] = fixed_lookup_counts[ordinal]
                            .checked_add(1)
                            .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(
                                "fixed lookup count",
                            ))?;
                        Some(LocalNodeSource::Fixed(ordinal))
                    }
                    Entry::Public => {
                        if plan_source.is_some()
                            || variable.index
                                >= usize::try_from(region_plan.local_explicit_len).map_err(
                                    |_| {
                                        FixedMultiAirCompleteLocalEndpointError::Overflow(
                                            "local public length",
                                        )
                                    },
                                )?
                        {
                            return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                                "public source",
                            ));
                        }
                        let coordinate = usize::from(plan.metadata.log_constraints)
                            .checked_add(
                                usize::try_from(region_plan.local_explicit_offset).map_err(
                                    |_| {
                                        FixedMultiAirCompleteLocalEndpointError::Overflow(
                                            "local public offset",
                                        )
                                    },
                                )?,
                            )
                            .and_then(|x| x.checked_add(variable.index))
                            .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(
                                "public coordinate",
                            ))?;
                        if coordinate >= beta_len {
                            return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                                "public coordinate",
                            ));
                        }
                        instance_beta_lookup_counts[coordinate] = instance_beta_lookup_counts
                            [coordinate]
                            .checked_add(1)
                            .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(
                                "public lookup count",
                            ))?;
                        Some(LocalNodeSource::Instance(coordinate))
                    }
                    Entry::Challenge => {
                        return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                            "challenge variable",
                        ))
                    }
                },
                SymbolicExpressionNode::IsFirstRow => {
                    if plan_source.is_some() {
                        return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                            "first-row source",
                        ));
                    }
                    let ordinal = expression.fixed_columns.len() + SELECTOR_FIRST;
                    fixed_lookup_counts[ordinal] += 1;
                    Some(LocalNodeSource::Fixed(ordinal))
                }
                SymbolicExpressionNode::IsLastRow => {
                    if plan_source.is_some() {
                        return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                            "last-row source",
                        ));
                    }
                    let ordinal = expression.fixed_columns.len() + SELECTOR_LAST;
                    fixed_lookup_counts[ordinal] += 1;
                    Some(LocalNodeSource::Fixed(ordinal))
                }
                SymbolicExpressionNode::IsTransition => {
                    if plan_source.is_some() {
                        return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                            "transition source",
                        ));
                    }
                    let ordinal = expression.fixed_columns.len() + SELECTOR_TRANSITION;
                    fixed_lookup_counts[ordinal] += 1;
                    Some(LocalNodeSource::Fixed(ordinal))
                }
                SymbolicExpressionNode::Constant(_)
                | SymbolicExpressionNode::Add { .. }
                | SymbolicExpressionNode::Sub { .. }
                | SymbolicExpressionNode::Neg { .. }
                | SymbolicExpressionNode::Mul { .. } => {
                    if plan_source.is_some() {
                        return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                            "non-source DAG node",
                        ));
                    }
                    None
                }
            };
            node_sources.push(source);
        }

        let mut node_fanout = vec![0usize; expression.nodes.len()];
        let mut symbolic_degrees = Vec::with_capacity(expression.nodes.len());
        for (node_index, node) in expression.nodes.iter().enumerate() {
            let symbolic_degree = validate_node_topology_and_degree(
                node_index,
                node,
                &expression.homogeneous_degrees,
                &symbolic_degrees,
                exact_relation_degree,
            )?;
            symbolic_degrees.push(symbolic_degree);
            match node {
                SymbolicExpressionNode::Add {
                    left_idx,
                    right_idx,
                    ..
                }
                | SymbolicExpressionNode::Sub {
                    left_idx,
                    right_idx,
                    ..
                }
                | SymbolicExpressionNode::Mul {
                    left_idx,
                    right_idx,
                    ..
                } => {
                    checked_increment(&mut node_fanout, *left_idx, "left node fanout")?;
                    checked_increment(&mut node_fanout, *right_idx, "right node fanout")?;
                }
                SymbolicExpressionNode::Neg { idx, .. } => {
                    checked_increment(&mut node_fanout, *idx, "negated node fanout")?;
                }
                _ => {}
            }
        }

        let global_log_height = usize::from(plan.metadata.log_constraints);
        let mut block_eq_point_blocks = Vec::with_capacity(component.constraints.len());
        for (constraint_ordinal, constraint) in component.constraints.iter().enumerate() {
            validate_local_constraint(
                constraint,
                constraint_ordinal,
                metadata_region.local_constraints.start,
                height,
                global_log_height,
                log_height,
                &expression.homogeneous_degrees,
            )?;
            checked_increment(
                &mut node_fanout,
                checked_option_u32(constraint.root_node, "constraint root")?,
                "constraint root fanout",
            )?;
            block_eq_point_blocks.push(usize::try_from(constraint.constraint_index).map_err(
                |_| FixedMultiAirCompleteLocalEndpointError::Overflow("constraint block"),
            )?);
        }
        if block_eq_point_blocks
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                "local Eq block order",
            ));
        }

        for &count in &opening_lookup_counts {
            if count == 0 {
                return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                    "unused dynamic opening",
                ));
            }
        }
        for &count in &fixed_lookup_counts[..expression.fixed_columns.len()] {
            if count == 0 {
                return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                    "unused fixed source",
                ));
            }
        }

        // The distinguished one is consumed by every active DAG row and once
        // per folded constraint, exactly as in backend homogeneous scaling.
        instance_beta_lookup_counts[one_beta_coordinate] = instance_beta_lookup_counts
            [one_beta_coordinate]
            .checked_add(expression.nodes.len())
            .and_then(|x| x.checked_add(component.constraints.len()))
            .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(
                "distinguished-one lookup count",
            ))?;

        validate_mapped_openings(plan, region, component)?;
        validate_field_word(region, "region")?;
        validate_field_word(expression.nodes.len(), "node count")?;
        validate_field_word(component.constraints.len(), "constraint count")?;
        for constraint in &component.constraints {
            validate_field_u64(constraint.constraint_index, "constraint block")?;
        }

        Ok(Self {
            region,
            setup_ordinal: region,
            log_height,
            exact_relation_degree,
            one_beta_coordinate,
            component: component.clone(),
            node_sources,
            node_fanout,
            block_eq_point_blocks,
            opening_lookup_counts,
            fixed_lookup_counts,
            instance_beta_lookup_counts,
        })
    }

    #[must_use]
    pub fn opening_lookup_counts(&self) -> &[usize] {
        &self.opening_lookup_counts
    }

    /// Multiplicities the cursor/openings owner must publish: every local
    /// endpoint variable use plus the one canonical global mapped-opening
    /// consumer. No constant fanout or host-provided count is permitted.
    pub fn cursor_opening_lookup_counts(
        &self,
    ) -> Result<Vec<u32>, FixedMultiAirCompleteLocalEndpointError> {
        self.opening_lookup_counts
            .iter()
            .map(|&endpoint_count| {
                endpoint_count
                    .checked_add(1)
                    .and_then(|count| u32::try_from(count).ok())
                    .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(
                        "cursor opening lookup count",
                    ))
            })
            .collect()
    }

    #[must_use]
    pub fn instance_beta_lookup_counts(&self) -> &[usize] {
        &self.instance_beta_lookup_counts
    }

    /// Number of endpoint fixed-source folds rooted at each local sumcheck
    /// point coordinate. Pass this as `point_fixed_source_count` to the
    /// regional cursor/sumcheck owner; its round recurrence expands this
    /// source count to the exact number of fold-row lookups.
    #[must_use]
    pub fn point_lookup_count_per_coordinate(&self) -> usize {
        self.component.expression.fixed_columns.len()
    }

    #[must_use]
    /// Owner-level common point multiplicity. The local endpoint consumes one
    /// lookup per coordinate to evaluate all three selectors together. The
    /// global mapper independently consumes one lookup per mapped dynamic
    /// opening, so the cursor must publish `1 + opening_count`.
    pub fn point_common_lookup_count_per_coordinate(
        &self,
    ) -> Result<usize, FixedMultiAirCompleteLocalEndpointError> {
        self.component
            .expression
            .dynamic_columns
            .len()
            .checked_add(1)
            .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(
                "cursor common point lookup count",
            ))
    }

    #[must_use]
    pub fn fixed_source_count(&self) -> usize {
        self.fixed_lookup_counts.len()
    }

    /// Exact setup-fixed BlockEq identities which consume this region's local
    /// point. One internal regional-point key is published per block and per
    /// coordinate; these lookups are downstream of the selector's single
    /// authenticated local-point lookup.
    #[must_use]
    pub fn block_eq_point_blocks(&self) -> &[usize] {
        &self.block_eq_point_blocks
    }

    /// Canonical block identity used by `block_eq`: the globally unique start
    /// of this constraint block in the complete relation table.
    pub fn eq_block_id(
        &self,
        constraint_ordinal: usize,
    ) -> Result<usize, FixedMultiAirCompleteLocalEndpointError> {
        let constraint = self.component.constraints.get(constraint_ordinal).ok_or(
            FixedMultiAirCompleteLocalEndpointError::Plan("constraint ordinal"),
        )?;
        usize::try_from(constraint.constraint_index)
            .map_err(|_| FixedMultiAirCompleteLocalEndpointError::Overflow("Eq block id"))
    }
}

/// Immutable source tables in the exact `component.expression.fixed_columns`
/// order. Their cells become cached-main columns of the endpoint AIR and must
/// be included in the complete relation/VK digest by the owner.
#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteLocalFixedValues {
    pub region: usize,
    pub log_height: usize,
    descriptors: Vec<FixedMultiAirCompleteTerminalCircuitFixedColumn>,
    tables: Vec<Vec<F>>,
}

impl FixedMultiAirCompleteLocalFixedValues {
    pub fn from_relation_setup(
        profile: &FixedMultiAirCompleteLocalEndpointProfile,
        cached_parts: &[FixedMultiAirCompleteCachedTrace<F>],
        preprocessed: Option<&DirectAirFixedTrace<F>>,
    ) -> Result<Self, FixedMultiAirCompleteLocalEndpointError> {
        let height = checked_pow2(profile.log_height, "fixed-table height")?;
        let mut tables = Vec::with_capacity(profile.component.expression.fixed_columns.len());
        for descriptor in &profile.component.expression.fixed_columns {
            let (width, values, column, rotation) = match *descriptor {
                FixedMultiAirCompleteTerminalCircuitFixedColumn::Cached {
                    part,
                    column,
                    rotation,
                } => {
                    let part = checked_u32(part, "cached part")?;
                    let trace = cached_parts.get(part).ok_or(
                        FixedMultiAirCompleteLocalEndpointError::Setup("cached part"),
                    )?;
                    (
                        usize::try_from(trace.width).map_err(|_| {
                            FixedMultiAirCompleteLocalEndpointError::Overflow("cached width")
                        })?,
                        trace.values.as_slice(),
                        checked_u32(column, "cached column")?,
                        usize::from(rotation),
                    )
                }
                FixedMultiAirCompleteTerminalCircuitFixedColumn::Preprocessed {
                    column,
                    rotation,
                } => {
                    let trace = preprocessed.ok_or(
                        FixedMultiAirCompleteLocalEndpointError::Setup("preprocessed trace"),
                    )?;
                    (
                        usize::try_from(trace.width).map_err(|_| {
                            FixedMultiAirCompleteLocalEndpointError::Overflow("preprocessed width")
                        })?,
                        trace.values.as_slice(),
                        checked_u32(column, "preprocessed column")?,
                        usize::from(rotation),
                    )
                }
            };
            if rotation > 1
                || column >= width
                || values.len()
                    != height.checked_mul(width).ok_or(
                        FixedMultiAirCompleteLocalEndpointError::Overflow("fixed table cells"),
                    )?
            {
                return Err(FixedMultiAirCompleteLocalEndpointError::Setup(
                    "fixed table shape",
                ));
            }
            let mut table = try_zero_vec(height)?;
            for (row, output) in table.iter_mut().enumerate() {
                let source_row = (row + rotation) & (height - 1);
                *output = *values.get(source_row * width + column).ok_or(
                    FixedMultiAirCompleteLocalEndpointError::Setup("fixed table cell"),
                )?;
            }
            tables.push(table);
        }
        Ok(Self {
            region: profile.region,
            log_height: profile.log_height,
            descriptors: profile.component.expression.fixed_columns.clone(),
            tables,
        })
    }

    fn validate(
        &self,
        profile: &FixedMultiAirCompleteLocalEndpointProfile,
    ) -> Result<(), FixedMultiAirCompleteLocalEndpointError> {
        let height = checked_pow2(profile.log_height, "fixed-table height")?;
        if self.region != profile.region
            || self.log_height != profile.log_height
            || self.descriptors != profile.component.expression.fixed_columns
            || self.tables.len() != self.descriptors.len()
            || self.tables.iter().any(|table| table.len() != height)
        {
            return Err(FixedMultiAirCompleteLocalEndpointError::Setup(
                "fixed setup substitution",
            ));
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteLocalFixedScheduleCols<T> {
    pub active: T,
    pub kind_flags: [T; FIXED_KIND_COUNT],
    pub source: T,
    pub layer: T,
    pub index: T,
    pub low_index: T,
    pub high_index: T,
    pub point_coordinate: T,
    pub initial_value: [T; D_EF],
    pub fixed_value_lookup_count: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteLocalFixedCols<T> {
    pub low: [T; D_EF],
    pub high: [T; D_EF],
    pub point: [T; D_EF],
    pub value: [T; D_EF],
}

/// Evaluates setup-fixed cached/preprocessed columns and the three analytic
/// row selectors at the authenticated regional point.
pub struct FixedMultiAirCompleteLocalFixedAir {
    pub profile: Arc<FixedMultiAirCompleteLocalEndpointProfile>,
    pub point_bus: FixedMultiAirCompleteLocalRegionPointBus,
    pub state_bus: FixedMultiAirCompleteLocalFixedStateBus,
    pub value_bus: FixedMultiAirCompleteLocalFixedValueBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteLocalFixedAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteLocalFixedAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirCompleteLocalFixedScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteLocalFixedCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteLocalFixedAir {}
impl BaseAir<F> for FixedMultiAirCompleteLocalFixedAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteLocalFixedScheduleCols::<F>::width()
            + FixedMultiAirCompleteLocalFixedCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteLocalFixedAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    AB::Expr: PrimeCharacteristicRing,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("complete local fixed schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("complete local fixed row")
            .to_vec();
        let schedule: &FixedMultiAirCompleteLocalFixedScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let local: &FixedMultiAirCompleteLocalFixedCols<AB::Var> = common.as_slice().borrow();

        for flag in schedule.kind_flags {
            builder.assert_bool(flag);
        }
        let active = schedule
            .kind_flags
            .iter()
            .fold(AB::Expr::ZERO, |sum, &flag| sum + AB::Expr::from(flag));
        builder.assert_eq(active, schedule.active);
        builder.assert_bool(schedule.active);

        assert_array_eq(
            &mut builder.when(schedule.kind_flags[FIXED_INITIAL]),
            local.value,
            schedule.initial_value.map(Into::into),
        );
        let folded = ext_field_add::<AB::Expr>(
            local.low.map(Into::into),
            ext_field_multiply::<AB::Expr>(
                local.point,
                ext_field_subtract::<AB::Expr>(local.high, local.low),
            ),
        );
        assert_array_eq(
            &mut builder.when(schedule.kind_flags[FIXED_FOLD]),
            local.value,
            folded,
        );

        assert_array_eq(
            &mut builder.when(schedule.kind_flags[FIXED_FINAL]),
            local.value,
            local.low.map(Into::into),
        );

        self.point_bus.lookup_key(
            builder,
            FixedMultiAirCompleteRegionPointMessage {
                region: AB::Expr::from_usize(self.profile.region),
                coordinate: schedule.point_coordinate.into(),
                value: local.point.map(Into::into),
            },
            schedule.kind_flags[FIXED_FOLD],
        );
        self.state_bus.receive(
            builder,
            FixedMultiAirCompleteLocalFixedStateMessage {
                region: AB::Expr::from_usize(self.profile.region),
                source: schedule.source.into(),
                layer: AB::Expr::from(schedule.layer) - AB::Expr::ONE,
                index: schedule.low_index.into(),
                value: local.low.map(Into::into),
            },
            schedule.kind_flags[FIXED_FOLD],
        );
        self.state_bus.receive(
            builder,
            FixedMultiAirCompleteLocalFixedStateMessage {
                region: AB::Expr::from_usize(self.profile.region),
                source: schedule.source.into(),
                layer: AB::Expr::from(schedule.layer) - AB::Expr::ONE,
                index: schedule.high_index.into(),
                value: local.high.map(Into::into),
            },
            schedule.kind_flags[FIXED_FOLD],
        );
        self.state_bus.send(
            builder,
            FixedMultiAirCompleteLocalFixedStateMessage {
                region: AB::Expr::from_usize(self.profile.region),
                source: schedule.source.into(),
                layer: schedule.layer.into(),
                index: schedule.index.into(),
                value: local.value.map(Into::into),
            },
            AB::Expr::from(schedule.kind_flags[FIXED_INITIAL])
                + AB::Expr::from(schedule.kind_flags[FIXED_FOLD]),
        );
        self.state_bus.receive(
            builder,
            FixedMultiAirCompleteLocalFixedStateMessage {
                region: AB::Expr::from_usize(self.profile.region),
                source: schedule.source.into(),
                layer: schedule.layer.into(),
                index: AB::Expr::ZERO,
                value: local.low.map(Into::into),
            },
            schedule.kind_flags[FIXED_FINAL],
        );
        self.value_bus.add_key_with_lookups(
            builder,
            FixedMultiAirCompleteLocalFixedValueMessage {
                region: AB::Expr::from_usize(self.profile.region),
                source: schedule.source.into(),
                value: local.value.map(Into::into),
            },
            schedule.fixed_value_lookup_count,
        );
    }
}

#[derive(Clone, Debug)]
struct FixedScheduleEntry {
    kind: usize,
    source: usize,
    layer: usize,
    index: usize,
    low_index: usize,
    high_index: usize,
    point_coordinate: usize,
    initial_value: EF,
    fixed_value_lookup_count: usize,
    low: EF,
    high: EF,
    point: EF,
    value: EF,
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteLocalFixedTrace {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub values: Vec<EF>,
}

pub fn generate_fixed_multi_air_complete_local_fixed_trace(
    profile: &FixedMultiAirCompleteLocalEndpointProfile,
    setup: &FixedMultiAirCompleteLocalFixedValues,
    point: &[EF],
    required_height: Option<usize>,
) -> Result<FixedMultiAirCompleteLocalFixedTrace, FixedMultiAirCompleteLocalEndpointError> {
    setup.validate(profile)?;
    if point.len() != profile.log_height {
        return Err(FixedMultiAirCompleteLocalEndpointError::Proof(
            "local point dimension",
        ));
    }
    let dense_count = setup.tables.len();
    let mut schedule = Vec::new();
    let mut final_values = Vec::with_capacity(dense_count);
    for source in 0..dense_count {
        let mut layer_values = setup.tables[source]
            .iter()
            .copied()
            .map(EF::from)
            .collect::<Vec<_>>();
        for (index, &value) in layer_values.iter().enumerate() {
            schedule.push(FixedScheduleEntry {
                kind: FIXED_INITIAL,
                source,
                layer: 0,
                index,
                low_index: 0,
                high_index: 0,
                point_coordinate: 0,
                initial_value: value,
                fixed_value_lookup_count: 0,
                low: EF::ZERO,
                high: EF::ZERO,
                point: EF::ZERO,
                value,
            });
        }
        for (coordinate, &challenge) in point.iter().enumerate() {
            let half = layer_values.len() / 2;
            if half == 0 || layer_values.len() != 2 * half {
                return Err(FixedMultiAirCompleteLocalEndpointError::Setup(
                    "fixed fold layer",
                ));
            }
            let mut next = Vec::new();
            next.try_reserve_exact(half)
                .map_err(|_| FixedMultiAirCompleteLocalEndpointError::Allocation)?;
            for index in 0..half {
                let low = layer_values[index];
                let high = layer_values[index + half];
                let value = low + challenge * (high - low);
                next.push(value);
                schedule.push(FixedScheduleEntry {
                    kind: FIXED_FOLD,
                    source,
                    layer: coordinate + 1,
                    index,
                    low_index: index,
                    high_index: index + half,
                    point_coordinate: coordinate,
                    initial_value: EF::ZERO,
                    fixed_value_lookup_count: 0,
                    low,
                    high,
                    point: challenge,
                    value,
                });
            }
            layer_values = next;
        }
        if layer_values.len() != 1 {
            return Err(FixedMultiAirCompleteLocalEndpointError::Setup(
                "fixed fold final",
            ));
        }
        let value = layer_values[0];
        final_values.push(value);
        schedule.push(FixedScheduleEntry {
            kind: FIXED_FINAL,
            source,
            layer: profile.log_height,
            index: 0,
            low_index: 0,
            high_index: 0,
            point_coordinate: 0,
            initial_value: EF::ZERO,
            fixed_value_lookup_count: profile.fixed_lookup_counts[source],
            low: value,
            high: EF::ZERO,
            point: EF::ZERO,
            value,
        });
    }

    let trace_height = if schedule.is_empty() {
        let height = required_height.unwrap_or(1);
        if !height.is_power_of_two() {
            return Err(FixedMultiAirCompleteLocalEndpointError::Proof(
                "fixed trace",
            ));
        }
        height
    } else {
        checked_trace_height(schedule.len(), required_height, "fixed trace")?
    };
    let cached_width = FixedMultiAirCompleteLocalFixedScheduleCols::<F>::width();
    let common_width = FixedMultiAirCompleteLocalFixedCols::<F>::width();
    let mut cached = try_zero_vec(checked_cells(trace_height, cached_width, "fixed cached")?)?;
    let mut common = try_zero_vec(checked_cells(trace_height, common_width, "fixed common")?)?;
    for (row, entry) in schedule.iter().enumerate() {
        let schedule_cols: &mut FixedMultiAirCompleteLocalFixedScheduleCols<F> =
            cached[row * cached_width..(row + 1) * cached_width].borrow_mut();
        schedule_cols.active = F::ONE;
        schedule_cols.kind_flags[entry.kind] = F::ONE;
        schedule_cols.source = field_usize(entry.source, "fixed source")?;
        schedule_cols.layer = field_usize(entry.layer, "fixed layer")?;
        schedule_cols.index = field_usize(entry.index, "fixed index")?;
        schedule_cols.low_index = field_usize(entry.low_index, "fixed low index")?;
        schedule_cols.high_index = field_usize(entry.high_index, "fixed high index")?;
        schedule_cols.point_coordinate = field_usize(entry.point_coordinate, "point coordinate")?;
        copy_ext(&mut schedule_cols.initial_value, entry.initial_value);
        schedule_cols.fixed_value_lookup_count =
            field_usize(entry.fixed_value_lookup_count, "fixed lookup count")?;
        let cols: &mut FixedMultiAirCompleteLocalFixedCols<F> =
            common[row * common_width..(row + 1) * common_width].borrow_mut();
        copy_ext(&mut cols.low, entry.low);
        copy_ext(&mut cols.high, entry.high);
        copy_ext(&mut cols.point, entry.point);
        copy_ext(&mut cols.value, entry.value);
    }
    Ok(FixedMultiAirCompleteLocalFixedTrace {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        values: final_values,
    })
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteLocalSelectorScheduleCols<T> {
    pub active: T,
    pub has_coordinate: T,
    pub is_first: T,
    pub is_last: T,
    pub coordinate: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteLocalSelectorCols<T> {
    pub point: [T; D_EF],
    pub first_before: [T; D_EF],
    pub first_after: [T; D_EF],
    pub last_before: [T; D_EF],
    pub last_after: [T; D_EF],
    pub selectors: [[T; D_EF]; SELECTOR_COUNT],
}

/// Evaluates IsFirst, IsLast and IsTransition together. It consumes each
/// authenticated local point coordinate exactly once, then republishes that
/// value under every setup-fixed local constraint BlockEq identity.
pub struct FixedMultiAirCompleteLocalSelectorAir {
    pub profile: Arc<FixedMultiAirCompleteLocalEndpointProfile>,
    pub point_bus: FixedMultiAirCompleteLocalRegionPointBus,
    pub block_eq_point_bus: FixedMultiAirCompleteRegionalPointBus,
    pub value_bus: FixedMultiAirCompleteLocalFixedValueBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteLocalSelectorAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteLocalSelectorAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirCompleteLocalSelectorScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteLocalSelectorCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteLocalSelectorAir {}
impl BaseAir<F> for FixedMultiAirCompleteLocalSelectorAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteLocalSelectorScheduleCols::<F>::width()
            + FixedMultiAirCompleteLocalSelectorCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteLocalSelectorAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("complete local selector schedule")
            .to_vec();
        let next_cached = builder.cached_mains()[0]
            .row_slice(1)
            .expect("next complete local selector schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("complete local selector row")
            .to_vec();
        let next_common = builder
            .common_main()
            .row_slice(1)
            .expect("next complete local selector row")
            .to_vec();
        let schedule: &FixedMultiAirCompleteLocalSelectorScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let next_schedule: &FixedMultiAirCompleteLocalSelectorScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FixedMultiAirCompleteLocalSelectorCols<AB::Var> = common.as_slice().borrow();
        let next: &FixedMultiAirCompleteLocalSelectorCols<AB::Var> =
            next_common.as_slice().borrow();

        for flag in [
            schedule.active,
            schedule.has_coordinate,
            schedule.is_first,
            schedule.is_last,
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

        let ext_one = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            }
        });
        assert_array_eq(
            &mut builder.when(schedule.is_first),
            local.first_before,
            ext_one.clone(),
        );
        assert_array_eq(
            &mut builder.when(schedule.is_first),
            local.last_before,
            ext_one,
        );
        let first_factor = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE - AB::Expr::from(local.point[limb])
            } else {
                -AB::Expr::from(local.point[limb])
            }
        });
        let expected_first = ext_field_multiply::<AB::Expr>(local.first_before, first_factor);
        let expected_last = ext_field_multiply::<AB::Expr>(local.last_before, local.point);
        assert_array_eq(
            &mut builder.when(schedule.has_coordinate),
            local.first_after,
            expected_first,
        );
        assert_array_eq(
            &mut builder.when(schedule.has_coordinate),
            local.last_after,
            expected_last,
        );
        assert_array_eq(
            &mut builder.when(schedule.active - schedule.has_coordinate),
            local.first_after,
            local.first_before.map(Into::into),
        );
        assert_array_eq(
            &mut builder.when(schedule.active - schedule.has_coordinate),
            local.last_after,
            local.last_before.map(Into::into),
        );
        assert_array_eq(
            &mut builder.when_transition().when(next_schedule.active),
            next.first_before,
            local.first_after.map(Into::into),
        );
        assert_array_eq(
            &mut builder.when_transition().when(next_schedule.active),
            next.last_before,
            local.last_after.map(Into::into),
        );
        assert_array_eq(
            &mut builder.when(schedule.is_last),
            local.selectors[SELECTOR_FIRST],
            local.first_after.map(Into::into),
        );
        assert_array_eq(
            &mut builder.when(schedule.is_last),
            local.selectors[SELECTOR_LAST],
            local.last_after.map(Into::into),
        );
        let transition = core::array::from_fn(|limb| {
            let one = if limb == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            one - AB::Expr::from(local.last_after[limb])
        });
        assert_array_eq(
            &mut builder.when(schedule.is_last),
            local.selectors[SELECTOR_TRANSITION],
            transition,
        );

        self.point_bus.lookup_key(
            builder,
            FixedMultiAirCompleteRegionPointMessage {
                region: AB::Expr::from_usize(self.profile.region),
                coordinate: schedule.coordinate.into(),
                value: local.point.map(Into::into),
            },
            schedule.has_coordinate,
        );
        for &block in self.profile.block_eq_point_blocks() {
            self.block_eq_point_bus.add_key_with_lookups(
                builder,
                FixedMultiAirCompleteRegionalPointMessage {
                    block: AB::Expr::from_usize(block),
                    coordinate: schedule.coordinate.into(),
                    value: local.point.map(Into::into),
                },
                schedule.has_coordinate,
            );
        }
        let dense_count = self.profile.component.expression.fixed_columns.len();
        for selector in 0..SELECTOR_COUNT {
            self.value_bus.add_key_with_lookups(
                builder,
                FixedMultiAirCompleteLocalFixedValueMessage {
                    region: AB::Expr::from_usize(self.profile.region),
                    source: AB::Expr::from_usize(dense_count + selector),
                    value: local.selectors[selector].map(Into::into),
                },
                schedule.is_last
                    * AB::Expr::from_usize(
                        self.profile.fixed_lookup_counts[dense_count + selector],
                    ),
            );
        }
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteLocalSelectorTrace {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub selectors: [EF; SELECTOR_COUNT],
}

pub fn generate_fixed_multi_air_complete_local_selector_trace(
    profile: &FixedMultiAirCompleteLocalEndpointProfile,
    point: &[EF],
    required_height: Option<usize>,
) -> Result<FixedMultiAirCompleteLocalSelectorTrace, FixedMultiAirCompleteLocalEndpointError> {
    if point.len() != profile.log_height {
        return Err(FixedMultiAirCompleteLocalEndpointError::Proof(
            "local selector point",
        ));
    }
    let rows = point.len().max(1);
    let height = checked_trace_height(rows, required_height, "selector trace")?;
    let cached_width = FixedMultiAirCompleteLocalSelectorScheduleCols::<F>::width();
    let common_width = FixedMultiAirCompleteLocalSelectorCols::<F>::width();
    let mut cached = try_zero_vec(checked_cells(height, cached_width, "selector cached")?)?;
    let mut common = try_zero_vec(checked_cells(height, common_width, "selector common")?)?;
    let mut first = EF::ONE;
    let mut last = EF::ONE;
    for row in 0..rows {
        let schedule: &mut FixedMultiAirCompleteLocalSelectorScheduleCols<F> =
            cached[row * cached_width..(row + 1) * cached_width].borrow_mut();
        schedule.active = F::ONE;
        schedule.has_coordinate = F::from_bool(row < point.len());
        schedule.is_first = F::from_bool(row == 0);
        schedule.is_last = F::from_bool(row + 1 == rows);
        schedule.coordinate = field_usize(row, "selector coordinate")?;
        let cols: &mut FixedMultiAirCompleteLocalSelectorCols<F> =
            common[row * common_width..(row + 1) * common_width].borrow_mut();
        let value = point.get(row).copied().unwrap_or(EF::ZERO);
        copy_ext(&mut cols.point, value);
        copy_ext(&mut cols.first_before, first);
        copy_ext(&mut cols.last_before, last);
        if row < point.len() {
            first *= EF::ONE - value;
            last *= value;
        }
        copy_ext(&mut cols.first_after, first);
        copy_ext(&mut cols.last_after, last);
        if row + 1 == rows {
            copy_ext(&mut cols.selectors[SELECTOR_FIRST], first);
            copy_ext(&mut cols.selectors[SELECTOR_LAST], last);
            copy_ext(&mut cols.selectors[SELECTOR_TRANSITION], EF::ONE - last);
        }
    }
    Ok(FixedMultiAirCompleteLocalSelectorTrace {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        selectors: [first, last, EF::ONE - last],
    })
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteLocalNodeScheduleCols<T> {
    pub active: T,
    pub kind_flags: [T; NODE_KIND_COUNT],
    pub source_flags: [T; SOURCE_KIND_COUNT],
    pub node: T,
    pub arg0_node: T,
    pub arg1_node: T,
    pub source_index: T,
    pub fanout: T,
    pub constant: [T; D_EF],
    pub left_power_flags: [T; MAX_RELATION_DEGREE + 1],
    pub right_power_flags: [T; MAX_RELATION_DEGREE + 1],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteLocalNodeCols<T> {
    pub source_value: [T; D_EF],
    pub arg0: [T; D_EF],
    pub arg1: [T; D_EF],
    pub value: [T; D_EF],
    pub one: [T; D_EF],
    pub one_powers: [[T; D_EF]; MAX_RELATION_DEGREE + 1],
}

/// Evaluates the canonical local expression DAG. The cached schedule fixes
/// every node kind, predecessor, source ordinal, degree delta, and fanout.
pub struct FixedMultiAirCompleteLocalNodeAir {
    pub profile: Arc<FixedMultiAirCompleteLocalEndpointProfile>,
    pub opening_bus: FixedMultiAirCompleteLocalRegionOpeningBus,
    pub fixed_value_bus: FixedMultiAirCompleteLocalFixedValueBus,
    pub instance_bus: FixedMultiAirCompleteInstanceValueBus,
    pub node_bus: FixedMultiAirCompleteLocalNodeBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteLocalNodeAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteLocalNodeAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirCompleteLocalNodeScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteLocalNodeCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteLocalNodeAir {}
impl BaseAir<F> for FixedMultiAirCompleteLocalNodeAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteLocalNodeScheduleCols::<F>::width()
            + FixedMultiAirCompleteLocalNodeCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteLocalNodeAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("complete local node schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("complete local node row")
            .to_vec();
        let schedule: &FixedMultiAirCompleteLocalNodeScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let local: &FixedMultiAirCompleteLocalNodeCols<AB::Var> = common.as_slice().borrow();

        let active = schedule
            .kind_flags
            .iter()
            .fold(AB::Expr::ZERO, |sum, &flag| sum + flag);
        builder.assert_eq(active.clone(), schedule.active);
        builder.assert_bool(schedule.active);
        let source_sum = schedule
            .source_flags
            .iter()
            .fold(AB::Expr::ZERO, |sum, &flag| sum + flag);
        builder.assert_eq(source_sum, schedule.kind_flags[NODE_SOURCE]);
        for flag in schedule.kind_flags.into_iter().chain(schedule.source_flags) {
            builder.assert_bool(flag);
        }
        for flag in schedule
            .left_power_flags
            .into_iter()
            .chain(schedule.right_power_flags)
        {
            builder.assert_bool(flag);
        }
        builder.assert_eq(
            schedule
                .left_power_flags
                .iter()
                .fold(AB::Expr::ZERO, |sum, &flag| sum + flag),
            schedule.active,
        );
        builder.assert_eq(
            schedule
                .right_power_flags
                .iter()
                .fold(AB::Expr::ZERO, |sum, &flag| sum + flag),
            schedule.active,
        );

        let one = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            }
        });
        assert_array_eq(&mut builder.when(active.clone()), local.one_powers[0], one);
        assert_array_eq(
            &mut builder.when(active.clone()),
            local.one_powers[1],
            local.one.map(Into::into),
        );
        for power in 1..MAX_RELATION_DEGREE {
            assert_array_eq(
                &mut builder.when(active.clone()),
                local.one_powers[power + 1],
                ext_field_multiply::<AB::Expr>(local.one_powers[power], local.one_powers[1]),
            );
        }
        let left_power = select_power::<AB>(local.one_powers, schedule.left_power_flags);
        let right_power = select_power::<AB>(local.one_powers, schedule.right_power_flags);
        let left_scaled = ext_field_multiply::<AB::Expr>(local.arg0, left_power);
        let right_scaled = ext_field_multiply::<AB::Expr>(local.arg1, right_power);
        let add = ext_field_add::<AB::Expr>(left_scaled.clone(), right_scaled.clone());
        let sub = ext_field_subtract::<AB::Expr>(left_scaled, right_scaled);
        let neg = local.arg0.map(|value| -AB::Expr::from(value));
        let mul = ext_field_multiply::<AB::Expr>(local.arg0, local.arg1);
        let expected = core::array::from_fn(|limb| {
            schedule.kind_flags[NODE_SOURCE] * local.source_value[limb]
                + schedule.kind_flags[NODE_CONSTANT] * schedule.constant[limb]
                + schedule.kind_flags[NODE_ADD] * add[limb].clone()
                + schedule.kind_flags[NODE_SUB] * sub[limb].clone()
                + schedule.kind_flags[NODE_NEG] * neg[limb].clone()
                + schedule.kind_flags[NODE_MUL] * mul[limb].clone()
        });
        assert_array_eq(builder, local.value, expected);

        self.opening_bus.lookup_key(
            builder,
            FixedMultiAirCompleteRegionOpeningMessage {
                region: AB::Expr::from_usize(self.profile.region),
                regional_opening: schedule.source_index.into(),
                global_opening: AB::Expr::from_usize(
                    self.profile.component.identity.opening_ordinal_start as usize,
                ) + AB::Expr::from(schedule.source_index),
                value: local.source_value.map(Into::into),
            },
            schedule.source_flags[SOURCE_OPENING],
        );
        self.fixed_value_bus.lookup_key(
            builder,
            FixedMultiAirCompleteLocalFixedValueMessage {
                region: AB::Expr::from_usize(self.profile.region),
                source: schedule.source_index.into(),
                value: local.source_value.map(Into::into),
            },
            schedule.source_flags[SOURCE_FIXED],
        );
        self.instance_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInstanceValueMessage {
                section: AB::Expr::from_usize(INSTANCE_SECTION_BETA),
                coordinate: schedule.source_index.into(),
                value: local.source_value.map(Into::into),
            },
            schedule.source_flags[SOURCE_INSTANCE],
        );
        self.instance_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInstanceValueMessage {
                section: AB::Expr::from_usize(INSTANCE_SECTION_BETA),
                coordinate: AB::Expr::from_usize(self.profile.one_beta_coordinate),
                value: local.one.map(Into::into),
            },
            schedule.active,
        );
        let arg0_enabled = schedule.kind_flags[NODE_ADD]
            + schedule.kind_flags[NODE_SUB]
            + schedule.kind_flags[NODE_NEG]
            + schedule.kind_flags[NODE_MUL];
        self.node_bus.lookup_key(
            builder,
            FixedMultiAirCompleteLocalNodeMessage {
                region: AB::Expr::from_usize(self.profile.region),
                node: schedule.arg0_node.into(),
                value: local.arg0.map(Into::into),
            },
            arg0_enabled,
        );
        let arg1_enabled = schedule.kind_flags[NODE_ADD]
            + schedule.kind_flags[NODE_SUB]
            + schedule.kind_flags[NODE_MUL];
        self.node_bus.lookup_key(
            builder,
            FixedMultiAirCompleteLocalNodeMessage {
                region: AB::Expr::from_usize(self.profile.region),
                node: schedule.arg1_node.into(),
                value: local.arg1.map(Into::into),
            },
            arg1_enabled,
        );
        self.node_bus.add_key_with_lookups(
            builder,
            FixedMultiAirCompleteLocalNodeMessage {
                region: AB::Expr::from_usize(self.profile.region),
                node: schedule.node.into(),
                value: local.value.map(Into::into),
            },
            schedule.fanout,
        );
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteLocalNodeTrace {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub values: Vec<EF>,
}

pub fn generate_fixed_multi_air_complete_local_node_trace(
    profile: &FixedMultiAirCompleteLocalEndpointProfile,
    beta: &[EF],
    openings: &[EF],
    fixed_values: &[EF],
    required_height: Option<usize>,
) -> Result<FixedMultiAirCompleteLocalNodeTrace, FixedMultiAirCompleteLocalEndpointError> {
    let expression = &profile.component.expression;
    if beta.len() != profile.instance_beta_lookup_counts.len()
        || openings.len() != expression.dynamic_columns.len()
        || fixed_values.len() != profile.fixed_source_count()
        || expression.nodes.len() != profile.node_sources.len()
        || expression.nodes.len() != profile.node_fanout.len()
    {
        return Err(FixedMultiAirCompleteLocalEndpointError::Proof(
            "local endpoint dimensions",
        ));
    }
    let one = *beta.get(profile.one_beta_coordinate).ok_or(
        FixedMultiAirCompleteLocalEndpointError::Proof("distinguished one"),
    )?;
    let one_powers = powers(one);
    let valid_rows = expression.nodes.len();
    let trace_height = checked_trace_height(valid_rows, required_height, "node trace")?;
    let cached_width = FixedMultiAirCompleteLocalNodeScheduleCols::<F>::width();
    let common_width = FixedMultiAirCompleteLocalNodeCols::<F>::width();
    let mut cached = try_zero_vec(checked_cells(trace_height, cached_width, "node cached")?)?;
    let mut common = try_zero_vec(checked_cells(trace_height, common_width, "node common")?)?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(valid_rows)
        .map_err(|_| FixedMultiAirCompleteLocalEndpointError::Allocation)?;

    for (node_index, node) in expression.nodes.iter().enumerate() {
        let mut kind = NODE_SOURCE;
        let mut source_kind = None;
        let mut source_index = 0usize;
        let mut arg0_node = 0usize;
        let mut arg1_node = 0usize;
        let mut left_delta = 0usize;
        let mut right_delta = 0usize;
        let mut constant = EF::ZERO;
        let mut source_value = EF::ZERO;
        let mut arg0 = EF::ZERO;
        let mut arg1 = EF::ZERO;
        let target_degree = usize::from(expression.homogeneous_degrees[node_index]);
        let value =
            match node {
                SymbolicExpressionNode::Variable(_)
                | SymbolicExpressionNode::IsFirstRow
                | SymbolicExpressionNode::IsLastRow
                | SymbolicExpressionNode::IsTransition => {
                    let source = profile.node_sources[node_index].ok_or(
                        FixedMultiAirCompleteLocalEndpointError::Plan("local node source"),
                    )?;
                    source_value = match source {
                        LocalNodeSource::Opening(index) => {
                            source_kind = Some(SOURCE_OPENING);
                            source_index = index;
                            *openings.get(index).ok_or(
                                FixedMultiAirCompleteLocalEndpointError::Proof("dynamic opening"),
                            )?
                        }
                        LocalNodeSource::Fixed(index) => {
                            source_kind = Some(SOURCE_FIXED);
                            source_index = index;
                            *fixed_values.get(index).ok_or(
                                FixedMultiAirCompleteLocalEndpointError::Proof("fixed value"),
                            )?
                        }
                        LocalNodeSource::Instance(coordinate) => {
                            source_kind = Some(SOURCE_INSTANCE);
                            source_index = coordinate;
                            *beta.get(coordinate).ok_or(
                                FixedMultiAirCompleteLocalEndpointError::Proof("public value"),
                            )?
                        }
                    };
                    source_value
                }
                SymbolicExpressionNode::Constant(value) => {
                    kind = NODE_CONSTANT;
                    constant = EF::from(*value);
                    constant
                }
                SymbolicExpressionNode::Add {
                    left_idx,
                    right_idx,
                    ..
                } => {
                    kind = NODE_ADD;
                    arg0_node = *left_idx;
                    arg1_node = *right_idx;
                    arg0 = checked_node_value(&values, *left_idx)?;
                    arg1 = checked_node_value(&values, *right_idx)?;
                    left_delta = target_degree
                        .checked_sub(usize::from(
                            *expression.homogeneous_degrees.get(*left_idx).ok_or(
                                FixedMultiAirCompleteLocalEndpointError::Plan("left degree"),
                            )?,
                        ))
                        .ok_or(FixedMultiAirCompleteLocalEndpointError::Plan(
                            "left degree delta",
                        ))?;
                    right_delta = target_degree
                        .checked_sub(usize::from(
                            *expression.homogeneous_degrees.get(*right_idx).ok_or(
                                FixedMultiAirCompleteLocalEndpointError::Plan("right degree"),
                            )?,
                        ))
                        .ok_or(FixedMultiAirCompleteLocalEndpointError::Plan(
                            "right degree delta",
                        ))?;
                    arg0 * one_powers[left_delta] + arg1 * one_powers[right_delta]
                }
                SymbolicExpressionNode::Sub {
                    left_idx,
                    right_idx,
                    ..
                } => {
                    kind = NODE_SUB;
                    arg0_node = *left_idx;
                    arg1_node = *right_idx;
                    arg0 = checked_node_value(&values, *left_idx)?;
                    arg1 = checked_node_value(&values, *right_idx)?;
                    left_delta = target_degree
                        .checked_sub(usize::from(
                            *expression.homogeneous_degrees.get(*left_idx).ok_or(
                                FixedMultiAirCompleteLocalEndpointError::Plan("left degree"),
                            )?,
                        ))
                        .ok_or(FixedMultiAirCompleteLocalEndpointError::Plan(
                            "left degree delta",
                        ))?;
                    right_delta = target_degree
                        .checked_sub(usize::from(
                            *expression.homogeneous_degrees.get(*right_idx).ok_or(
                                FixedMultiAirCompleteLocalEndpointError::Plan("right degree"),
                            )?,
                        ))
                        .ok_or(FixedMultiAirCompleteLocalEndpointError::Plan(
                            "right degree delta",
                        ))?;
                    arg0 * one_powers[left_delta] - arg1 * one_powers[right_delta]
                }
                SymbolicExpressionNode::Neg { idx, .. } => {
                    kind = NODE_NEG;
                    arg0_node = *idx;
                    arg0 = checked_node_value(&values, *idx)?;
                    -arg0
                }
                SymbolicExpressionNode::Mul {
                    left_idx,
                    right_idx,
                    ..
                } => {
                    kind = NODE_MUL;
                    arg0_node = *left_idx;
                    arg1_node = *right_idx;
                    arg0 = checked_node_value(&values, *left_idx)?;
                    arg1 = checked_node_value(&values, *right_idx)?;
                    arg0 * arg1
                }
            };
        if left_delta > MAX_RELATION_DEGREE || right_delta > MAX_RELATION_DEGREE {
            return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                "homogeneous degree delta",
            ));
        }
        values.push(value);

        let schedule: &mut FixedMultiAirCompleteLocalNodeScheduleCols<F> =
            cached[node_index * cached_width..(node_index + 1) * cached_width].borrow_mut();
        schedule.active = F::ONE;
        schedule.kind_flags[kind] = F::ONE;
        if let Some(source_kind) = source_kind {
            schedule.source_flags[source_kind] = F::ONE;
        }
        schedule.node = field_usize(node_index, "node ordinal")?;
        schedule.arg0_node = field_usize(arg0_node, "arg0 node")?;
        schedule.arg1_node = field_usize(arg1_node, "arg1 node")?;
        schedule.source_index = field_usize(source_index, "node source")?;
        schedule.fanout = field_usize(profile.node_fanout[node_index], "node fanout")?;
        copy_ext(&mut schedule.constant, constant);
        schedule.left_power_flags[left_delta] = F::ONE;
        schedule.right_power_flags[right_delta] = F::ONE;

        let cols: &mut FixedMultiAirCompleteLocalNodeCols<F> =
            common[node_index * common_width..(node_index + 1) * common_width].borrow_mut();
        copy_ext(&mut cols.source_value, source_value);
        copy_ext(&mut cols.arg0, arg0);
        copy_ext(&mut cols.arg1, arg1);
        copy_ext(&mut cols.value, value);
        copy_ext(&mut cols.one, one);
        for (target, &power) in cols.one_powers.iter_mut().zip(&one_powers) {
            copy_ext(target, power);
        }
    }
    Ok(FixedMultiAirCompleteLocalNodeTrace {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        values,
    })
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteLocalFoldScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub constraint_ordinal: T,
    pub constraint_block: T,
    pub root_node: T,
    pub root_power_flags: [T; MAX_RELATION_DEGREE + 1],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteLocalFoldCols<T> {
    pub root_value: [T; D_EF],
    pub weight: [T; D_EF],
    pub one: [T; D_EF],
    pub one_powers: [[T; D_EF]; MAX_RELATION_DEGREE + 1],
    pub term: [T; D_EF],
    pub sum_before: [T; D_EF],
    pub sum_after: [T; D_EF],
}

/// Folds local constraint roots in canonical constraint order, after exact
/// homogeneous scaling, under the Eq weights supplied by `block_eq`.
pub struct FixedMultiAirCompleteLocalFoldAir {
    pub profile: Arc<FixedMultiAirCompleteLocalEndpointProfile>,
    pub node_bus: FixedMultiAirCompleteLocalNodeBus,
    pub instance_bus: FixedMultiAirCompleteInstanceValueBus,
    pub eq_weight_bus: FixedMultiAirCompleteEqWeightBus,
    pub final_evaluation_bus: FixedMultiAirCompleteLocalRegionFinalEvaluationBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteLocalFoldAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteLocalFoldAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirCompleteLocalFoldScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteLocalFoldCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteLocalFoldAir {}
impl BaseAir<F> for FixedMultiAirCompleteLocalFoldAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteLocalFoldScheduleCols::<F>::width()
            + FixedMultiAirCompleteLocalFoldCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteLocalFoldAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("complete local fold schedule")
            .to_vec();
        let next_cached = builder.cached_mains()[0]
            .row_slice(1)
            .expect("next complete local fold schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("complete local fold row")
            .to_vec();
        let next_common = builder
            .common_main()
            .row_slice(1)
            .expect("next complete local fold row")
            .to_vec();
        let schedule: &FixedMultiAirCompleteLocalFoldScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let next_schedule: &FixedMultiAirCompleteLocalFoldScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FixedMultiAirCompleteLocalFoldCols<AB::Var> = common.as_slice().borrow();
        let next: &FixedMultiAirCompleteLocalFoldCols<AB::Var> = next_common.as_slice().borrow();

        for flag in [schedule.active, schedule.is_first, schedule.is_last] {
            builder.assert_bool(flag);
        }
        for flag in schedule.root_power_flags {
            builder.assert_bool(flag);
        }
        builder.assert_eq(
            schedule
                .root_power_flags
                .iter()
                .fold(AB::Expr::ZERO, |sum, &flag| sum + flag),
            schedule.active,
        );
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
            next.sum_before,
            local.sum_after.map(Into::into),
        );
        let ext_one = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            }
        });
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.one_powers[0],
            ext_one,
        );
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.one_powers[1],
            local.one.map(Into::into),
        );
        for power in 1..MAX_RELATION_DEGREE {
            assert_array_eq(
                &mut builder.when(schedule.active),
                local.one_powers[power + 1],
                ext_field_multiply::<AB::Expr>(local.one_powers[power], local.one_powers[1]),
            );
        }
        let root_power = select_power::<AB>(local.one_powers, schedule.root_power_flags);
        let scaled_root = ext_field_multiply::<AB::Expr>(local.root_value, root_power);
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.term,
            ext_field_multiply::<AB::Expr>(local.weight, scaled_root),
        );
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.sum_after,
            ext_field_add::<AB::Expr>(local.sum_before, local.term),
        );
        assert_array_eq(
            &mut builder.when(schedule.is_first),
            local.sum_before,
            [AB::Expr::ZERO; D_EF],
        );

        self.node_bus.lookup_key(
            builder,
            FixedMultiAirCompleteLocalNodeMessage {
                region: AB::Expr::from_usize(self.profile.region),
                node: schedule.root_node.into(),
                value: local.root_value.map(Into::into),
            },
            schedule.active,
        );
        self.instance_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInstanceValueMessage {
                section: AB::Expr::from_usize(INSTANCE_SECTION_BETA),
                coordinate: AB::Expr::from_usize(self.profile.one_beta_coordinate),
                value: local.one.map(Into::into),
            },
            schedule.active,
        );
        self.eq_weight_bus.receive(
            builder,
            FixedMultiAirCompleteEqWeightMessage {
                block: schedule.constraint_block.into(),
                role: AB::Expr::from_usize(LOCAL_EQ_ROLE),
                role_ordinal: schedule.constraint_ordinal.into(),
                value: local.weight.map(Into::into),
            },
            schedule.active,
        );
        self.final_evaluation_bus.receive(
            builder,
            FixedMultiAirCompleteRegionFinalEvaluationMessage {
                region: AB::Expr::from_usize(self.profile.region),
                claim: local.sum_after.map(Into::into),
            },
            schedule.is_last,
        );
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteLocalFoldTrace {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub final_claim: EF,
}

pub fn generate_fixed_multi_air_complete_local_fold_trace(
    profile: &FixedMultiAirCompleteLocalEndpointProfile,
    node_values: &[EF],
    eq_weights: &[EF],
    one: EF,
    required_height: Option<usize>,
) -> Result<FixedMultiAirCompleteLocalFoldTrace, FixedMultiAirCompleteLocalEndpointError> {
    let constraints = &profile.component.constraints;
    if node_values.len() != profile.component.expression.nodes.len()
        || eq_weights.len() != constraints.len()
    {
        return Err(FixedMultiAirCompleteLocalEndpointError::Proof(
            "local fold dimensions",
        ));
    }
    let trace_height = checked_trace_height(constraints.len(), required_height, "fold trace")?;
    let cached_width = FixedMultiAirCompleteLocalFoldScheduleCols::<F>::width();
    let common_width = FixedMultiAirCompleteLocalFoldCols::<F>::width();
    let mut cached = try_zero_vec(checked_cells(trace_height, cached_width, "fold cached")?)?;
    let mut common = try_zero_vec(checked_cells(trace_height, common_width, "fold common")?)?;
    let one_powers = powers(one);
    let mut sum = EF::ZERO;

    for (ordinal, (constraint, &weight)) in constraints.iter().zip(eq_weights).enumerate() {
        let root = checked_option_u32(constraint.root_node, "constraint root")?;
        let root_value = checked_node_value(node_values, root)?;
        let root_degree = usize::from(constraint.root_degree.ok_or(
            FixedMultiAirCompleteLocalEndpointError::Plan("constraint root degree"),
        )?);
        let delta = profile
            .exact_relation_degree
            .checked_sub(root_degree)
            .ok_or(FixedMultiAirCompleteLocalEndpointError::Plan(
                "constraint degree delta",
            ))?;
        if delta > MAX_RELATION_DEGREE {
            return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                "constraint degree delta",
            ));
        }
        let term = weight * root_value * one_powers[delta];
        let before = sum;
        sum += term;

        let schedule: &mut FixedMultiAirCompleteLocalFoldScheduleCols<F> =
            cached[ordinal * cached_width..(ordinal + 1) * cached_width].borrow_mut();
        schedule.active = F::ONE;
        schedule.is_first = F::from_bool(ordinal == 0);
        schedule.is_last = F::from_bool(ordinal + 1 == constraints.len());
        schedule.constraint_ordinal = field_usize(ordinal, "constraint ordinal")?;
        schedule.constraint_block = field_u64(constraint.constraint_index, "constraint block")?;
        schedule.root_node = field_usize(root, "constraint root")?;
        schedule.root_power_flags[delta] = F::ONE;

        let cols: &mut FixedMultiAirCompleteLocalFoldCols<F> =
            common[ordinal * common_width..(ordinal + 1) * common_width].borrow_mut();
        copy_ext(&mut cols.root_value, root_value);
        copy_ext(&mut cols.weight, weight);
        copy_ext(&mut cols.one, one);
        for (target, &power) in cols.one_powers.iter_mut().zip(&one_powers) {
            copy_ext(target, power);
        }
        copy_ext(&mut cols.term, term);
        copy_ext(&mut cols.sum_before, before);
        copy_ext(&mut cols.sum_after, sum);
    }
    Ok(FixedMultiAirCompleteLocalFoldTrace {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        final_claim: sum,
    })
}

pub struct FixedMultiAirCompleteLocalEndpointAirs {
    pub fixed: FixedMultiAirCompleteLocalFixedAir,
    pub selector: FixedMultiAirCompleteLocalSelectorAir,
    pub nodes: FixedMultiAirCompleteLocalNodeAir,
    pub fold: FixedMultiAirCompleteLocalFoldAir,
}

impl FixedMultiAirCompleteLocalEndpointAirs {
    #[must_use]
    pub fn new(
        profile: Arc<FixedMultiAirCompleteLocalEndpointProfile>,
        buses: FixedMultiAirCompleteLocalEndpointBuses,
    ) -> Self {
        Self {
            fixed: FixedMultiAirCompleteLocalFixedAir {
                profile: profile.clone(),
                point_bus: buses.point,
                state_bus: buses.fixed_state,
                value_bus: buses.fixed_value,
            },
            selector: FixedMultiAirCompleteLocalSelectorAir {
                profile: profile.clone(),
                point_bus: buses.point,
                block_eq_point_bus: buses.block_eq_point,
                value_bus: buses.fixed_value,
            },
            nodes: FixedMultiAirCompleteLocalNodeAir {
                profile: profile.clone(),
                opening_bus: buses.opening,
                fixed_value_bus: buses.fixed_value,
                instance_bus: buses.instance,
                node_bus: buses.node,
            },
            fold: FixedMultiAirCompleteLocalFoldAir {
                profile,
                node_bus: buses.node,
                instance_bus: buses.instance,
                eq_weight_bus: buses.eq_weight,
                final_evaluation_bus: buses.final_evaluation,
            },
        }
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteLocalEndpointTrace {
    pub fixed: FixedMultiAirCompleteLocalFixedTrace,
    pub selector: FixedMultiAirCompleteLocalSelectorTrace,
    pub nodes: FixedMultiAirCompleteLocalNodeTrace,
    pub fold: FixedMultiAirCompleteLocalFoldTrace,
}

#[allow(clippy::too_many_arguments)]
pub fn generate_fixed_multi_air_complete_local_endpoint_trace(
    profile: &FixedMultiAirCompleteLocalEndpointProfile,
    setup: &FixedMultiAirCompleteLocalFixedValues,
    beta: &[EF],
    point: &[EF],
    openings: &[EF],
    eq_weights: &[EF],
    fixed_height: Option<usize>,
    selector_height: Option<usize>,
    node_height: Option<usize>,
    fold_height: Option<usize>,
) -> Result<FixedMultiAirCompleteLocalEndpointTrace, FixedMultiAirCompleteLocalEndpointError> {
    let fixed =
        generate_fixed_multi_air_complete_local_fixed_trace(profile, setup, point, fixed_height)?;
    let selector =
        generate_fixed_multi_air_complete_local_selector_trace(profile, point, selector_height)?;
    let mut fixed_and_selectors = Vec::new();
    fixed_and_selectors
        .try_reserve_exact(profile.fixed_source_count())
        .map_err(|_| FixedMultiAirCompleteLocalEndpointError::Allocation)?;
    fixed_and_selectors.extend_from_slice(&fixed.values);
    fixed_and_selectors.extend_from_slice(&selector.selectors);
    let nodes = generate_fixed_multi_air_complete_local_node_trace(
        profile,
        beta,
        openings,
        &fixed_and_selectors,
        node_height,
    )?;
    let one = *beta.get(profile.one_beta_coordinate).ok_or(
        FixedMultiAirCompleteLocalEndpointError::Proof("distinguished one"),
    )?;
    let fold = generate_fixed_multi_air_complete_local_fold_trace(
        profile,
        &nodes.values,
        eq_weights,
        one,
        fold_height,
    )?;
    Ok(FixedMultiAirCompleteLocalEndpointTrace {
        fixed,
        selector,
        nodes,
        fold,
    })
}

fn validate_mapped_openings(
    plan: &FixedMultiAirCompleteTerminalCircuitPlan<F, Digest>,
    region: usize,
    component: &openvm_stark_backend::native_warp::FixedMultiAirCompleteTerminalCircuitComponentPlan<F>,
) -> Result<(), FixedMultiAirCompleteLocalEndpointError> {
    let start = checked_u32(
        component.identity.opening_ordinal_start,
        "opening ordinal start",
    )?;
    let count = checked_u32(component.identity.opening_count, "opening count")?;
    let end = start
        .checked_add(count)
        .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(
            "mapped opening range",
        ))?;
    let mapped = plan.mapped_openings.get(start..end).ok_or(
        FixedMultiAirCompleteLocalEndpointError::Plan("mapped opening range"),
    )?;
    if mapped.len() != component.expression.dynamic_columns.len() {
        return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
            "mapped opening coverage",
        ));
    }
    for (ordinal, (opening, source)) in mapped
        .iter()
        .zip(&component.expression.dynamic_columns)
        .enumerate()
    {
        let global =
            start
                .checked_add(ordinal)
                .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(
                    "global opening ordinal",
                ))?;
        let canonical_eq = FixedMultiAirCompleteTerminalEqBlockPlan::new(
            usize::from(plan.metadata.code_class.log_message_len),
            source.start,
            usize::from(source.log_height),
            usize::from(source.rotation),
        )
        .map_err(|_| FixedMultiAirCompleteLocalEndpointError::Plan("mapped opening Eq"))?;
        if opening.component != FixedMultiAirCompleteTerminalCircuitComponentKind::Local
            || usize::try_from(opening.region_ordinal).ok() != Some(region)
            || usize::try_from(opening.regional_opening_ordinal).ok() != Some(ordinal)
            || usize::try_from(opening.global_opening_ordinal).ok() != Some(global)
            || usize::try_from(opening.rho_ordinal).ok() != Some(global)
            || opening.source != *source
            || opening.block.start
                != usize::try_from(source.start).map_err(|_| {
                    FixedMultiAirCompleteLocalEndpointError::Overflow("mapped block start")
                })?
            || opening.block.log_height != usize::from(source.log_height)
            || opening.eq != canonical_eq
        {
            return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                "mapped opening identity",
            ));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_local_constraint(
    constraint: &FixedMultiAirCompleteTerminalConstraintPlan,
    ordinal: usize,
    local_constraint_start: u64,
    height: usize,
    global_log_height: usize,
    local_log_height: usize,
    degrees: &[u8],
) -> Result<(), FixedMultiAirCompleteLocalEndpointError> {
    let expected_index = local_constraint_start
        .checked_add(
            u64::try_from(ordinal)
                .ok()
                .and_then(|x| x.checked_mul(u64::try_from(height).ok()?))
                .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(
                    "constraint index",
                ))?,
        )
        .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(
            "constraint index",
        ))?;
    let root = checked_option_u32(constraint.root_node, "constraint root")?;
    let root_degree = *degrees
        .get(root)
        .ok_or(FixedMultiAirCompleteLocalEndpointError::Plan(
            "constraint root",
        ))?;
    let canonical_eq = FixedMultiAirCompleteTerminalEqBlockPlan::new(
        global_log_height,
        expected_index,
        local_log_height,
        0,
    )
    .map_err(|_| FixedMultiAirCompleteLocalEndpointError::Plan("local constraint Eq"))?;
    if constraint.role
        != (FixedMultiAirCompleteTerminalEqRole::LocalConstraint {
            constraint_ordinal: u32::try_from(ordinal).map_err(|_| {
                FixedMultiAirCompleteLocalEndpointError::Overflow("constraint ordinal")
            })?,
        })
        || constraint.constraint_index != expected_index
        || constraint.root_degree != Some(root_degree)
        || constraint.eq != canonical_eq
    {
        return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
            "local constraint plan",
        ));
    }
    Ok(())
}

fn validate_node_topology_and_degree(
    node_index: usize,
    node: &SymbolicExpressionNode<F>,
    degrees: &[u8],
    symbolic_degrees: &[usize],
    max_degree: usize,
) -> Result<usize, FixedMultiAirCompleteLocalEndpointError> {
    let target = usize::from(*degrees.get(node_index).ok_or(
        FixedMultiAirCompleteLocalEndpointError::Plan("node degree ordinal"),
    )?);
    let operand_degree = |index: usize| {
        if index >= node_index {
            return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                "DAG topological order",
            ));
        }
        degrees.get(index).copied().map(usize::from).ok_or(
            FixedMultiAirCompleteLocalEndpointError::Plan("DAG operand degree"),
        )
    };
    let symbolic_operand_degree =
        |index: usize| {
            if index >= node_index {
                return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                    "DAG topological order",
                ));
            }
            symbolic_degrees.get(index).copied().ok_or(
                FixedMultiAirCompleteLocalEndpointError::Plan("symbolic DAG operand degree"),
            )
        };
    let (expected, symbolic_expected) = match node {
        SymbolicExpressionNode::Variable(variable) => match variable.entry {
            Entry::Main { .. } => (1, 1),
            Entry::Preprocessed { .. } => (0, 0),
            Entry::Public => (1, 0),
            Entry::Challenge => {
                return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                    "challenge variable",
                ))
            }
        },
        SymbolicExpressionNode::IsFirstRow
        | SymbolicExpressionNode::IsLastRow
        | SymbolicExpressionNode::IsTransition => (0, 1),
        SymbolicExpressionNode::Constant(_) => (0, 0),
        SymbolicExpressionNode::Add {
            left_idx,
            right_idx,
            degree_multiple,
        }
        | SymbolicExpressionNode::Sub {
            left_idx,
            right_idx,
            degree_multiple,
        } => {
            let expected = operand_degree(*left_idx)?.max(operand_degree(*right_idx)?);
            let symbolic_expected =
                symbolic_operand_degree(*left_idx)?.max(symbolic_operand_degree(*right_idx)?);
            if *degree_multiple != symbolic_expected {
                return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                    "additive degree metadata",
                ));
            }
            (expected, symbolic_expected)
        }
        SymbolicExpressionNode::Neg {
            idx,
            degree_multiple,
        } => {
            let expected = operand_degree(*idx)?;
            let symbolic_expected = symbolic_operand_degree(*idx)?;
            if *degree_multiple != symbolic_expected {
                return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                    "negation degree metadata",
                ));
            }
            (expected, symbolic_expected)
        }
        SymbolicExpressionNode::Mul {
            left_idx,
            right_idx,
            degree_multiple,
        } => {
            let expected = operand_degree(*left_idx)?
                .checked_add(operand_degree(*right_idx)?)
                .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(
                    "multiplicative degree",
                ))?;
            let symbolic_expected = symbolic_operand_degree(*left_idx)?
                .checked_add(symbolic_operand_degree(*right_idx)?)
                .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(
                    "symbolic multiplicative degree",
                ))?;
            if *degree_multiple != symbolic_expected {
                return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
                    "multiplicative degree metadata",
                ));
            }
            (expected, symbolic_expected)
        }
    };
    if expected != target || target > max_degree {
        return Err(FixedMultiAirCompleteLocalEndpointError::Plan(
            "homogeneous node degree",
        ));
    }
    Ok(symbolic_expected)
}

fn validate_sorted_unique<T: Ord>(
    values: &[T],
    context: &'static str,
) -> Result<(), FixedMultiAirCompleteLocalEndpointError> {
    if values.windows(2).any(|pair| pair[0] >= pair[1]) {
        return Err(FixedMultiAirCompleteLocalEndpointError::Plan(context));
    }
    Ok(())
}

fn checked_increment(
    values: &mut [usize],
    index: usize,
    context: &'static str,
) -> Result<(), FixedMultiAirCompleteLocalEndpointError> {
    let value = values
        .get_mut(index)
        .ok_or(FixedMultiAirCompleteLocalEndpointError::Plan(context))?;
    *value = value
        .checked_add(1)
        .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(context))?;
    Ok(())
}

fn checked_node_value(
    values: &[EF],
    index: usize,
) -> Result<EF, FixedMultiAirCompleteLocalEndpointError> {
    values
        .get(index)
        .copied()
        .ok_or(FixedMultiAirCompleteLocalEndpointError::Plan(
            "DAG node value",
        ))
}

fn checked_option_u32(
    value: Option<u32>,
    context: &'static str,
) -> Result<usize, FixedMultiAirCompleteLocalEndpointError> {
    checked_u32(
        value.ok_or(FixedMultiAirCompleteLocalEndpointError::Plan(context))?,
        context,
    )
}

fn checked_u32(
    value: u32,
    context: &'static str,
) -> Result<usize, FixedMultiAirCompleteLocalEndpointError> {
    usize::try_from(value).map_err(|_| FixedMultiAirCompleteLocalEndpointError::Overflow(context))
}

fn checked_pow2(
    log_height: usize,
    context: &'static str,
) -> Result<usize, FixedMultiAirCompleteLocalEndpointError> {
    1usize
        .checked_shl(
            u32::try_from(log_height)
                .map_err(|_| FixedMultiAirCompleteLocalEndpointError::Overflow(context))?,
        )
        .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(context))
}

fn checked_trace_height(
    valid_rows: usize,
    required_height: Option<usize>,
    context: &'static str,
) -> Result<usize, FixedMultiAirCompleteLocalEndpointError> {
    if valid_rows == 0 {
        return Err(FixedMultiAirCompleteLocalEndpointError::Plan(context));
    }
    let minimum = valid_rows
        .checked_next_power_of_two()
        .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(context))?;
    let height = required_height.unwrap_or(minimum);
    if height < valid_rows || !height.is_power_of_two() {
        return Err(FixedMultiAirCompleteLocalEndpointError::Proof(context));
    }
    Ok(height)
}

fn checked_cells(
    height: usize,
    width: usize,
    context: &'static str,
) -> Result<usize, FixedMultiAirCompleteLocalEndpointError> {
    height
        .checked_mul(width)
        .filter(|&cells| cells <= isize::MAX as usize / core::mem::size_of::<F>())
        .ok_or(FixedMultiAirCompleteLocalEndpointError::Overflow(context))
}

fn try_zero_vec(len: usize) -> Result<Vec<F>, FixedMultiAirCompleteLocalEndpointError> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(len)
        .map_err(|_| FixedMultiAirCompleteLocalEndpointError::Allocation)?;
    output.resize(len, F::ZERO);
    Ok(output)
}

fn validate_field_word(
    value: usize,
    context: &'static str,
) -> Result<(), FixedMultiAirCompleteLocalEndpointError> {
    if u64::try_from(value)
        .ok()
        .is_none_or(|x| x >= u64::from(F::ORDER_U32))
    {
        return Err(FixedMultiAirCompleteLocalEndpointError::Overflow(context));
    }
    Ok(())
}

fn validate_field_u64(
    value: u64,
    context: &'static str,
) -> Result<(), FixedMultiAirCompleteLocalEndpointError> {
    if value >= u64::from(F::ORDER_U32) {
        return Err(FixedMultiAirCompleteLocalEndpointError::Overflow(context));
    }
    Ok(())
}

fn field_usize(
    value: usize,
    context: &'static str,
) -> Result<F, FixedMultiAirCompleteLocalEndpointError> {
    validate_field_word(value, context)?;
    Ok(F::from_usize(value))
}

fn field_u64(
    value: u64,
    context: &'static str,
) -> Result<F, FixedMultiAirCompleteLocalEndpointError> {
    validate_field_u64(value, context)?;
    Ok(F::from_u64(value))
}

fn powers(one: EF) -> [EF; MAX_RELATION_DEGREE + 1] {
    let mut result = [EF::ONE; MAX_RELATION_DEGREE + 1];
    for power in 0..MAX_RELATION_DEGREE {
        result[power + 1] = result[power] * one;
    }
    result
}

fn select_power<AB: AirBuilder<F = F>>(
    powers: [[AB::Var; D_EF]; MAX_RELATION_DEGREE + 1],
    flags: [AB::Var; MAX_RELATION_DEGREE + 1],
) -> [AB::Expr; D_EF] {
    core::array::from_fn(|limb| {
        powers
            .iter()
            .zip(flags.iter())
            .fold(AB::Expr::ZERO, |sum, (power, flag)| {
                sum + AB::Expr::from(flag.clone()) * AB::Expr::from(power[limb].clone())
            })
    })
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
            debug::{check_constraints, check_logup},
            symbolic::{get_symbolic_builder, SymbolicConstraintsDag, SymbolicRapBuilder},
        },
        hasher::MerkleHasher,
        interaction::{InteractionBuilder, SymbolicInteraction},
        keygen::types::{
            StarkVerifyingKey, StarkVerifyingParams, TraceWidth, VerifierSinglePreprocessedData,
        },
        native_warp::{
            DirectAirCodeClass, DirectAirFixedTrace, DirectAirPesatIndex, DirectAirPublicSchema,
            FixedMultiAirCompleteCachedTrace, FixedMultiAirCompletePesatIndex,
            FixedMultiAirCompleteTerminalLinearizer,
        },
        StarkProtocolConfig, SystemParams,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::{
        BabyBearPoseidon2Config as SC, DIGEST_SIZE,
    };
    use p3_air::{AirBuilderWithPublicValues, PairBuilder};

    use super::*;
    use crate::native_warp::terminal::fixed_multi_air_complete::{
        FixedMultiAirCompleteBlockEqAir, FixedMultiAirCompleteBlockEqProfile,
        FixedMultiAirCompleteGlobalPointBus, FixedMultiAirCompleteGlobalPointMessage,
        FixedMultiAirCompleteInteractionRegionFinalEvaluationBus,
        FixedMultiAirCompleteInteractionRegionOpeningBus,
        FixedMultiAirCompleteRegionCursorOpeningProfile,
    };

    #[derive(Clone, Copy)]
    struct CompleteLocalTestAir;

    impl BaseAir<F> for CompleteLocalTestAir {
        fn width(&self) -> usize {
            3
        }
    }

    impl BaseAirWithPublicValues<F> for CompleteLocalTestAir {
        fn num_public_values(&self) -> usize {
            1
        }
    }

    impl PartitionedBaseAir<F> for CompleteLocalTestAir {
        fn cached_main_widths(&self) -> Vec<usize> {
            vec![1]
        }

        fn common_main_width(&self) -> usize {
            2
        }
    }

    impl Air<SymbolicRapBuilder<F>> for CompleteLocalTestAir {
        fn eval(&self, builder: &mut SymbolicRapBuilder<F>) {
            let main = builder.common_main().clone();
            let cached = builder.cached_mains()[0].clone();
            let preprocessed = builder.preprocessed();
            let local = main.row_slice(0).expect("local fixture row");
            let next = main.row_slice(1).expect("next local fixture row");
            let cached_local = cached.row_slice(0).expect("cached fixture row")[0];
            let cached_next = cached.row_slice(1).expect("next cached fixture row")[0];
            let fixed_local = preprocessed.row_slice(0).expect("fixed fixture row")[0];
            let fixed_next = preprocessed.row_slice(1).expect("next fixed fixture row")[0];
            let value = local[0];
            let auxiliary = local[1];
            let public = builder.public_values()[0];
            builder.assert_zero(builder.is_first_row() * (value - public));
            builder.assert_zero(builder.is_last_row() * (value - public));
            builder.assert_zero(builder.is_transition() * (next[0] - value));
            builder.assert_zero(cached_local - fixed_local);
            builder.assert_zero(builder.is_transition() * (cached_next - fixed_next));
            let auxiliary_square = auxiliary.clone() * auxiliary.clone();
            let value_square = value.clone() * value.clone();
            builder.assert_zero(
                auxiliary_square.clone() * auxiliary_square * auxiliary
                    - value_square.clone() * value_square * value,
            );
        }
    }

    fn digest(value: u32) -> Digest {
        [F::from_u32(value); DIGEST_SIZE]
    }

    fn direct_relation<H>(hasher: &H, log_height: usize) -> DirectAirPesatIndex<F, Digest>
    where
        H: MerkleHasher<F = F, Digest = Digest>,
    {
        let air = CompleteLocalTestAir;
        let width = TraceWidth {
            preprocessed: Some(1),
            cached_mains: vec![1],
            common_main: 2,
        };
        let symbolic = get_symbolic_builder(&air, &width).constraints();
        let verifying_key = StarkVerifyingKey {
            preprocessed_data: Some(VerifierSinglePreprocessedData {
                commit: digest(19),
                hypercube_dim: log_height as isize,
                stacking_width: 1,
            }),
            params: StarkVerifyingParams {
                width,
                num_public_values: 1,
                need_rot: true,
            },
            max_constraint_degree: symbolic.max_constraint_degree() as u8,
            symbolic_constraints: Arc::new(SymbolicConstraintsDag::from(symbolic)),
            is_required: true,
            unused_variables: Vec::new(),
        };
        let log_message_len = u8::try_from(log_height + 2).expect("small fixture");
        DirectAirPesatIndex::from_verifying_key(
            hasher,
            digest(1),
            17,
            log_height,
            &verifying_key,
            Some(DirectAirFixedTrace {
                width: 1,
                values: vec![F::from_u32(9); 1usize << log_height],
            }),
            DirectAirPublicSchema {
                public_values_len: 1,
                boundary_values_len: 0,
                schema_digest: digest(2),
            },
            DirectAirCodeClass {
                log_message_len,
                log_blowup: 1,
                log_codeword_len: log_message_len + 1,
                initial_folding_factor: 0,
                rows_per_query: 1,
            },
        )
        .expect("direct local fixture relation")
    }

    fn backend_plan(log_height: usize) -> Arc<FixedMultiAirCompleteTerminalCircuitPlan<F, Digest>> {
        let config = SC::default_from_params(SystemParams::new_for_testing(8));
        let hasher = config.hasher();
        let log_message_len = u8::try_from(log_height + 1).expect("small fixture");
        let relation = FixedMultiAirCompletePesatIndex::from_direct_air_regions_with_fixed_cached(
            hasher,
            digest(1),
            vec![direct_relation(hasher, log_height)],
            vec![vec![FixedMultiAirCompleteCachedTrace {
                width: 1,
                values: vec![F::from_u32(9); 1usize << log_height],
            }]],
            DirectAirCodeClass {
                log_message_len,
                log_blowup: 1,
                log_codeword_len: log_message_len + 1,
                initial_folding_factor: 0,
                rows_per_query: 1,
            },
        )
        .expect("complete local fixture relation");
        Arc::new(
            FixedMultiAirCompleteTerminalLinearizer::new(&relation)
                .expect("complete local linearizer")
                .circuit_plan()
                .expect("backend complete circuit plan"),
        )
    }

    fn ef(seed: usize) -> EF {
        EF::from_basis_coefficients_fn(|coordinate| F::from_usize(seed + 13 * coordinate + 1))
    }

    fn selectors(point: &[EF]) -> [EF; SELECTOR_COUNT] {
        let first = point
            .iter()
            .copied()
            .map(|coordinate| EF::ONE - coordinate)
            .product();
        let last = point.iter().copied().product();
        [first, last, EF::ONE - last]
    }

    fn fixed_setup(
        profile: &FixedMultiAirCompleteLocalEndpointProfile,
    ) -> FixedMultiAirCompleteLocalFixedValues {
        let height = 1usize << profile.log_height;
        FixedMultiAirCompleteLocalFixedValues::from_relation_setup(
            profile,
            &[FixedMultiAirCompleteCachedTrace {
                width: 1,
                values: vec![F::from_u32(9); height],
            }],
            Some(&DirectAirFixedTrace {
                width: 1,
                values: vec![F::from_u32(9); height],
            }),
        )
        .expect("canonical fixed setup")
    }

    fn independent_endpoint(
        profile: &FixedMultiAirCompleteLocalEndpointProfile,
        beta: &[EF],
        openings: &[EF],
        fixed: &[EF],
        eq_weights: &[EF],
    ) -> (Vec<EF>, EF) {
        let mut nodes: Vec<EF> = Vec::with_capacity(profile.component.expression.nodes.len());
        let one = beta[profile.one_beta_coordinate];
        let one_powers = powers(one);
        for (index, node) in profile.component.expression.nodes.iter().enumerate() {
            let value = match node {
                SymbolicExpressionNode::Variable { .. }
                | SymbolicExpressionNode::IsFirstRow
                | SymbolicExpressionNode::IsLastRow
                | SymbolicExpressionNode::IsTransition => {
                    match profile.node_sources[index].expect("canonical source node") {
                        LocalNodeSource::Opening(source) => openings[source],
                        LocalNodeSource::Fixed(source) => fixed[source],
                        LocalNodeSource::Instance(source) => beta[source],
                    }
                }
                SymbolicExpressionNode::Constant(value) => EF::from(*value),
                SymbolicExpressionNode::Add {
                    left_idx,
                    right_idx,
                    ..
                } => {
                    let target =
                        usize::from(profile.component.expression.homogeneous_degrees[index]);
                    let left =
                        usize::from(profile.component.expression.homogeneous_degrees[*left_idx]);
                    let right =
                        usize::from(profile.component.expression.homogeneous_degrees[*right_idx]);
                    nodes[*left_idx] * one_powers[target - left]
                        + nodes[*right_idx] * one_powers[target - right]
                }
                SymbolicExpressionNode::Sub {
                    left_idx,
                    right_idx,
                    ..
                } => {
                    let target =
                        usize::from(profile.component.expression.homogeneous_degrees[index]);
                    let left =
                        usize::from(profile.component.expression.homogeneous_degrees[*left_idx]);
                    let right =
                        usize::from(profile.component.expression.homogeneous_degrees[*right_idx]);
                    nodes[*left_idx] * one_powers[target - left]
                        - nodes[*right_idx] * one_powers[target - right]
                }
                SymbolicExpressionNode::Neg { idx, .. } => -nodes[*idx],
                SymbolicExpressionNode::Mul {
                    left_idx,
                    right_idx,
                    ..
                } => nodes[*left_idx] * nodes[*right_idx],
            };
            nodes.push(value);
        }
        let final_claim = profile
            .component
            .constraints
            .iter()
            .zip(eq_weights)
            .map(|(constraint, &weight)| {
                let root = usize::try_from(constraint.root_node.expect("local root"))
                    .expect("small local root");
                let degree = usize::from(constraint.root_degree.expect("local root degree"));
                weight * nodes[root] * one_powers[profile.exact_relation_degree - degree]
            })
            .sum();
        (nodes, final_claim)
    }

    fn assert_endpoint_constraints(
        airs: &FixedMultiAirCompleteLocalEndpointAirs,
        trace: &FixedMultiAirCompleteLocalEndpointTrace,
    ) {
        for (name, air, cached, common) in [
            (
                "local fixed",
                &airs.fixed as &dyn EndpointDebugAir,
                &trace.fixed.cached,
                &trace.fixed.common,
            ),
            (
                "local selector",
                &airs.selector as &dyn EndpointDebugAir,
                &trace.selector.cached,
                &trace.selector.common,
            ),
            (
                "local nodes",
                &airs.nodes as &dyn EndpointDebugAir,
                &trace.nodes.cached,
                &trace.nodes.common,
            ),
            (
                "local fold",
                &airs.fold as &dyn EndpointDebugAir,
                &trace.fold.cached,
                &trace.fold.common,
            ),
        ] {
            air.check(name, cached, common);
        }
    }

    trait EndpointDebugAir {
        fn check(&self, name: &str, cached: &RowMajorMatrix<F>, common: &RowMajorMatrix<F>);
    }

    impl<A> EndpointDebugAir for A
    where
        A: for<'a> Air<openvm_stark_backend::air_builders::debug::DebugConstraintBuilder<'a, SC>>
            + BaseAir<F>
            + BaseAirWithPublicValues<F>
            + PartitionedBaseAir<F>,
    {
        fn check(&self, name: &str, cached: &RowMajorMatrix<F>, common: &RowMajorMatrix<F>) {
            check_constraints::<_, SC>(
                self,
                name,
                &None,
                &[cached.as_view(), common.as_view()],
                &[],
            );
        }
    }

    fn test_buses() -> FixedMultiAirCompleteLocalEndpointBuses {
        FixedMultiAirCompleteLocalEndpointBuses {
            point: FixedMultiAirCompleteLocalRegionPointBus::new(901),
            block_eq_point: FixedMultiAirCompleteRegionalPointBus::new(902),
            opening: FixedMultiAirCompleteLocalRegionOpeningBus::new(903),
            final_evaluation: FixedMultiAirCompleteLocalRegionFinalEvaluationBus::new(904),
            instance: FixedMultiAirCompleteInstanceValueBus::new(905),
            eq_weight: FixedMultiAirCompleteEqWeightBus::new(906),
            fixed_state: FixedMultiAirCompleteLocalFixedStateBus::new(907),
            fixed_value: FixedMultiAirCompleteLocalFixedValueBus::new(908),
            node: FixedMultiAirCompleteLocalNodeBus::new(909),
        }
    }

    #[test]
    fn backend_plans_match_local_endpoint_at_zero_and_nonzero_heights() {
        for log_height in 0..=3 {
            let plan = backend_plan(log_height);
            let profile = Arc::new(
                FixedMultiAirCompleteLocalEndpointProfile::from_circuit_plan(&plan, 0)
                    .expect("canonical local profile"),
            );
            let cursor = FixedMultiAirCompleteRegionCursorOpeningProfile::from_plan(plan.clone())
                .expect("canonical cursor profile");
            let cursor_local = cursor
                .components
                .iter()
                .find(|component| {
                    component.kind == FixedMultiAirCompleteTerminalCircuitComponentKind::Local
                        && component.region == 0
                })
                .expect("local cursor component");
            assert_eq!(
                profile.cursor_opening_lookup_counts().unwrap(),
                cursor_local.opening_lookup_counts
            );
            assert_eq!(
                profile.point_lookup_count_per_coordinate(),
                cursor_local.point_fixed_source_count
            );
            assert_eq!(
                profile.point_common_lookup_count_per_coordinate().unwrap(),
                cursor_local.point_common_count
            );
            assert_eq!(
                profile.block_eq_point_blocks().len(),
                profile.component.constraints.len()
            );

            let beta = (0..profile.instance_beta_lookup_counts().len())
                .map(|index| ef(100 + index))
                .collect::<Vec<_>>();
            let point = (0..log_height)
                .map(|index| ef(300 + index))
                .collect::<Vec<_>>();
            let openings = (0..profile.component.expression.dynamic_columns.len())
                .map(|index| ef(500 + index))
                .collect::<Vec<_>>();
            let eq_weights = (0..profile.component.constraints.len())
                .map(|index| ef(700 + index))
                .collect::<Vec<_>>();
            let setup = fixed_setup(&profile);
            let trace = generate_fixed_multi_air_complete_local_endpoint_trace(
                &profile,
                &setup,
                &beta,
                &point,
                &openings,
                &eq_weights,
                None,
                None,
                None,
                None,
            )
            .expect("local endpoint trace");
            let mut fixed = trace.fixed.values.clone();
            fixed.extend_from_slice(&selectors(&point));
            let (expected_nodes, expected_final) =
                independent_endpoint(&profile, &beta, &openings, &fixed, &eq_weights);
            assert_eq!(trace.nodes.values, expected_nodes);
            assert_eq!(trace.fold.final_claim, expected_final);
            let airs = FixedMultiAirCompleteLocalEndpointAirs::new(profile, test_buses());
            assert_endpoint_constraints(&airs, &trace);
        }
    }

    #[test]
    fn malformed_backend_plan_sources_fail_without_panicking() {
        let plan = backend_plan(2);
        let mut malformed = (*plan).clone();
        malformed.regions[0].local.expression.node_sources.pop();
        let result = catch_unwind(AssertUnwindSafe(|| {
            FixedMultiAirCompleteLocalEndpointProfile::from_circuit_plan(&malformed, 0)
        }));
        assert!(result.is_ok());
        assert!(result.unwrap().is_err());

        let mut malformed = (*plan).clone();
        malformed.regions[0].local.expression.dynamic_columns[0].rotation = 2;
        let result = catch_unwind(AssertUnwindSafe(|| {
            FixedMultiAirCompleteLocalEndpointProfile::from_circuit_plan(&malformed, 0)
        }));
        assert!(result.is_ok());
        assert!(result.unwrap().is_err());
    }

    const AUTH_POINT_PUBLISH: usize = 0;
    const AUTH_POINT_MAPPER: usize = 1;
    const AUTH_OPENING_PUBLISH: usize = 2;
    const AUTH_OPENING_MAPPER: usize = 3;
    const AUTH_INSTANCE_PUBLISH: usize = 4;
    const AUTH_GLOBAL_POINT_PUBLISH: usize = 5;
    const AUTH_FINAL_PUBLISH: usize = 6;
    const AUTH_KIND_COUNT: usize = 7;

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct AuthorityCols<T> {
        active: T,
        kind_flags: [T; AUTH_KIND_COUNT],
        region: T,
        coordinate: T,
        regional_opening: T,
        global_opening: T,
        section: T,
        value: [T; D_EF],
        count: T,
    }

    #[derive(Clone, Copy)]
    enum TestOpeningBus {
        Local(FixedMultiAirCompleteLocalRegionOpeningBus),
        Interaction(FixedMultiAirCompleteInteractionRegionOpeningBus),
    }

    #[derive(Clone, Copy)]
    enum TestFinalBus {
        Local(FixedMultiAirCompleteLocalRegionFinalEvaluationBus),
        Interaction(FixedMultiAirCompleteInteractionRegionFinalEvaluationBus),
    }

    struct AuthorityAir {
        point: FixedMultiAirCompleteLocalRegionPointBus,
        opening: TestOpeningBus,
        final_evaluation: TestFinalBus,
        instance: FixedMultiAirCompleteInstanceValueBus,
        global_point: FixedMultiAirCompleteGlobalPointBus,
    }

    impl BaseAir<F> for AuthorityAir {
        fn width(&self) -> usize {
            AuthorityCols::<F>::width()
        }
    }

    impl BaseAirWithPublicValues<F> for AuthorityAir {}
    impl PartitionedBaseAir<F> for AuthorityAir {}

    impl<AB> Air<AB> for AuthorityAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("authority row");
            let local: &AuthorityCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.active);
            for flag in local.kind_flags {
                builder.assert_bool(flag);
            }
            let active = local
                .kind_flags
                .iter()
                .fold(AB::Expr::ZERO, |sum, &flag| sum + AB::Expr::from(flag));
            builder.assert_eq(active, local.active);

            self.point.add_key_with_lookups(
                builder,
                FixedMultiAirCompleteRegionPointMessage {
                    region: local.region.into(),
                    coordinate: local.coordinate.into(),
                    value: local.value.map(Into::into),
                },
                local.kind_flags[AUTH_POINT_PUBLISH] * local.count,
            );
            self.point.lookup_key(
                builder,
                FixedMultiAirCompleteRegionPointMessage {
                    region: local.region.into(),
                    coordinate: local.coordinate.into(),
                    value: local.value.map(Into::into),
                },
                local.kind_flags[AUTH_POINT_MAPPER] * local.count,
            );

            let opening = FixedMultiAirCompleteRegionOpeningMessage {
                region: local.region.into(),
                regional_opening: local.regional_opening.into(),
                global_opening: local.global_opening.into(),
                value: local.value.map(Into::into),
            };
            match self.opening {
                TestOpeningBus::Local(bus) => {
                    bus.add_key_with_lookups(
                        builder,
                        opening.clone(),
                        local.kind_flags[AUTH_OPENING_PUBLISH] * local.count,
                    );
                    bus.lookup_key(
                        builder,
                        opening,
                        local.kind_flags[AUTH_OPENING_MAPPER] * local.count,
                    );
                }
                TestOpeningBus::Interaction(bus) => {
                    bus.add_key_with_lookups(
                        builder,
                        opening.clone(),
                        local.kind_flags[AUTH_OPENING_PUBLISH] * local.count,
                    );
                    bus.lookup_key(
                        builder,
                        opening,
                        local.kind_flags[AUTH_OPENING_MAPPER] * local.count,
                    );
                }
            }
            self.instance.add_key_with_lookups(
                builder,
                FixedMultiAirCompleteInstanceValueMessage {
                    section: local.section.into(),
                    coordinate: local.coordinate.into(),
                    value: local.value.map(Into::into),
                },
                local.kind_flags[AUTH_INSTANCE_PUBLISH] * local.count,
            );
            self.global_point.add_key_with_lookups(
                builder,
                FixedMultiAirCompleteGlobalPointMessage {
                    coordinate: local.coordinate.into(),
                    value: local.value.map(Into::into),
                },
                local.kind_flags[AUTH_GLOBAL_POINT_PUBLISH] * local.count,
            );
            let final_message = FixedMultiAirCompleteRegionFinalEvaluationMessage {
                region: local.region.into(),
                claim: local.value.map(Into::into),
            };
            match self.final_evaluation {
                TestFinalBus::Local(bus) => {
                    bus.send(builder, final_message, local.kind_flags[AUTH_FINAL_PUBLISH])
                }
                TestFinalBus::Interaction(bus) => {
                    bus.send(builder, final_message, local.kind_flags[AUTH_FINAL_PUBLISH])
                }
            }
        }
    }

    #[derive(Clone, Copy)]
    struct AuthorityRow {
        kind: usize,
        region: usize,
        coordinate: usize,
        regional_opening: usize,
        global_opening: usize,
        section: usize,
        value: EF,
        count: usize,
    }

    fn authority_trace(
        profile: &FixedMultiAirCompleteLocalEndpointProfile,
        beta: &[EF],
        point: &[EF],
        openings: &[EF],
        global_point: &[EF],
        final_claim: EF,
    ) -> RowMajorMatrix<F> {
        let opening_start =
            usize::try_from(profile.component.identity.opening_ordinal_start).unwrap();
        let mut rows = Vec::new();
        for (coordinate, &value) in point.iter().enumerate() {
            let fixed_count = profile.point_lookup_count_per_coordinate()
                * (1usize << (profile.log_height - coordinate - 1));
            rows.push(AuthorityRow {
                kind: AUTH_POINT_PUBLISH,
                region: profile.region,
                coordinate,
                regional_opening: 0,
                global_opening: 0,
                section: 0,
                value,
                count: fixed_count + profile.point_common_lookup_count_per_coordinate().unwrap(),
            });
            if !openings.is_empty() {
                rows.push(AuthorityRow {
                    kind: AUTH_POINT_MAPPER,
                    region: profile.region,
                    coordinate,
                    regional_opening: 0,
                    global_opening: 0,
                    section: 0,
                    value,
                    count: openings.len(),
                });
            }
        }
        let cursor_counts = profile.cursor_opening_lookup_counts().unwrap();
        for (opening, (&value, &count)) in openings.iter().zip(&cursor_counts).enumerate() {
            for (kind, count) in [
                (AUTH_OPENING_PUBLISH, count as usize),
                (AUTH_OPENING_MAPPER, 1),
            ] {
                rows.push(AuthorityRow {
                    kind,
                    region: profile.region,
                    coordinate: 0,
                    regional_opening: opening,
                    global_opening: opening_start + opening,
                    section: 0,
                    value,
                    count,
                });
            }
        }
        for (coordinate, (&value, &count)) in beta
            .iter()
            .zip(profile.instance_beta_lookup_counts())
            .enumerate()
        {
            if count != 0 {
                rows.push(AuthorityRow {
                    kind: AUTH_INSTANCE_PUBLISH,
                    region: profile.region,
                    coordinate,
                    regional_opening: 0,
                    global_opening: 0,
                    section: INSTANCE_SECTION_BETA,
                    value,
                    count,
                });
            }
        }
        for (coordinate, &value) in global_point.iter().enumerate() {
            rows.push(AuthorityRow {
                kind: AUTH_GLOBAL_POINT_PUBLISH,
                region: profile.region,
                coordinate,
                regional_opening: 0,
                global_opening: 0,
                section: 0,
                value,
                count: profile.block_eq_point_blocks().len(),
            });
        }
        rows.push(AuthorityRow {
            kind: AUTH_FINAL_PUBLISH,
            region: profile.region,
            coordinate: 0,
            regional_opening: 0,
            global_opening: 0,
            section: 0,
            value: final_claim,
            count: 1,
        });

        let width = AuthorityCols::<F>::width();
        let height = rows.len().max(1).next_power_of_two();
        let mut values = vec![F::ZERO; height * width];
        for (row, entry) in rows.iter().enumerate() {
            let cols: &mut AuthorityCols<F> = values[row * width..(row + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.kind_flags[entry.kind] = F::ONE;
            cols.region = F::from_usize(entry.region);
            cols.coordinate = F::from_usize(entry.coordinate);
            cols.regional_opening = F::from_usize(entry.regional_opening);
            cols.global_opening = F::from_usize(entry.global_opening);
            cols.section = F::from_usize(entry.section);
            copy_ext(&mut cols.value, entry.value);
            cols.count = F::from_usize(entry.count);
        }
        RowMajorMatrix::new(values, width)
    }

    fn authority_air(
        buses: FixedMultiAirCompleteLocalEndpointBuses,
        global_point: FixedMultiAirCompleteGlobalPointBus,
        cross_opening: bool,
        cross_final: bool,
    ) -> AuthorityAir {
        AuthorityAir {
            point: buses.point,
            opening: if cross_opening {
                TestOpeningBus::Interaction(FixedMultiAirCompleteInteractionRegionOpeningBus::new(
                    911,
                ))
            } else {
                TestOpeningBus::Local(buses.opening)
            },
            final_evaluation: if cross_final {
                TestFinalBus::Interaction(
                    FixedMultiAirCompleteInteractionRegionFinalEvaluationBus::new(912),
                )
            } else {
                TestFinalBus::Local(buses.final_evaluation)
            },
            instance: buses.instance,
            global_point,
        }
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

    fn check_full_balance(
        endpoint_airs: &FixedMultiAirCompleteLocalEndpointAirs,
        endpoint_trace: &FixedMultiAirCompleteLocalEndpointTrace,
        block_airs: &[FixedMultiAirCompleteBlockEqAir],
        block_traces: &[crate::native_warp::terminal::fixed_multi_air_complete::FixedMultiAirCompleteBlockEqTrace],
        authority_air: &AuthorityAir,
        authority_trace: &RowMajorMatrix<F>,
    ) {
        let mut names = Vec::new();
        let mut interactions = Vec::new();
        let mut matrices = Vec::new();
        for (name, air, cached, common) in [
            (
                "local-fixed",
                &endpoint_airs.fixed as &dyn SymbolicEndpointAir,
                &endpoint_trace.fixed.cached,
                &endpoint_trace.fixed.common,
            ),
            (
                "local-selector",
                &endpoint_airs.selector as &dyn SymbolicEndpointAir,
                &endpoint_trace.selector.cached,
                &endpoint_trace.selector.common,
            ),
            (
                "local-nodes",
                &endpoint_airs.nodes as &dyn SymbolicEndpointAir,
                &endpoint_trace.nodes.cached,
                &endpoint_trace.nodes.common,
            ),
            (
                "local-fold",
                &endpoint_airs.fold as &dyn SymbolicEndpointAir,
                &endpoint_trace.fold.cached,
                &endpoint_trace.fold.common,
            ),
        ] {
            names.push(name.to_string());
            interactions.push(air.interactions());
            matrices.push(vec![cached.as_view(), common.as_view()]);
        }
        for (index, (air, trace)) in block_airs.iter().zip(block_traces).enumerate() {
            names.push(format!("block-eq-{index}"));
            interactions.push(symbolic_interactions(air));
            matrices.push(vec![trace.cached.as_view(), trace.common.as_view()]);
        }
        names.push("authority".to_string());
        interactions.push(symbolic_interactions(authority_air));
        matrices.push(vec![authority_trace.as_view()]);
        let preprocessed = vec![None; names.len()];
        let publics = vec![Vec::new(); names.len()];
        check_logup(&names, &interactions, &preprocessed, &matrices, &publics);
    }

    trait SymbolicEndpointAir {
        fn interactions(&self) -> Vec<SymbolicInteraction<F>>;
    }

    impl<A> SymbolicEndpointAir for A
    where
        A: Air<SymbolicRapBuilder<F>>
            + BaseAir<F>
            + BaseAirWithPublicValues<F>
            + PartitionedBaseAir<F>,
    {
        fn interactions(&self) -> Vec<SymbolicInteraction<F>> {
            symbolic_interactions(self)
        }
    }

    #[test]
    fn full_typed_balance_includes_block_eq_bridge_and_rejects_cross_routing() {
        let plan = backend_plan(2);
        let profile = Arc::new(
            FixedMultiAirCompleteLocalEndpointProfile::from_circuit_plan(&plan, 0).unwrap(),
        );
        let beta = (0..profile.instance_beta_lookup_counts().len())
            .map(|index| ef(100 + index))
            .collect::<Vec<_>>();
        let point = (0..profile.log_height)
            .map(|index| ef(300 + index))
            .collect::<Vec<_>>();
        let openings = (0..profile.component.expression.dynamic_columns.len())
            .map(|index| ef(500 + index))
            .collect::<Vec<_>>();
        let global_point = beta[..usize::from(plan.metadata.log_constraints)].to_vec();
        let buses = test_buses();
        let global_point_bus = FixedMultiAirCompleteGlobalPointBus::new(910);
        let mut block_airs = Vec::new();
        let mut block_traces = Vec::new();
        let mut eq_weights = Vec::new();
        for (ordinal, constraint) in profile.component.constraints.iter().enumerate() {
            let block = profile.eq_block_id(ordinal).unwrap();
            let block_profile = FixedMultiAirCompleteBlockEqProfile::from_constraint(
                block,
                profile.region,
                constraint,
            )
            .unwrap();
            let trace =
                crate::native_warp::terminal::fixed_multi_air_complete::generate_fixed_multi_air_complete_block_eq_trace(
                    &block_profile,
                    &global_point,
                    &point,
                    None,
                )
                .unwrap();
            eq_weights.push(trace.weight);
            block_airs.push(FixedMultiAirCompleteBlockEqAir {
                profile: block_profile,
                global_point_bus,
                regional_point_bus: buses.block_eq_point,
                weight_bus: buses.eq_weight,
            });
            block_traces.push(trace);
        }
        let setup = fixed_setup(&profile);
        let endpoint_trace = generate_fixed_multi_air_complete_local_endpoint_trace(
            &profile,
            &setup,
            &beta,
            &point,
            &openings,
            &eq_weights,
            None,
            None,
            None,
            None,
        )
        .unwrap();
        let endpoint_airs = FixedMultiAirCompleteLocalEndpointAirs::new(profile.clone(), buses);
        assert_endpoint_constraints(&endpoint_airs, &endpoint_trace);
        for (air, trace) in block_airs.iter().zip(&block_traces) {
            check_constraints::<_, SC>(
                air,
                "local block Eq",
                &None,
                &[trace.cached.as_view(), trace.common.as_view()],
                &[],
            );
        }
        let authority_trace = authority_trace(
            &profile,
            &beta,
            &point,
            &openings,
            &global_point,
            endpoint_trace.fold.final_claim,
        );
        let honest = authority_air(buses, global_point_bus, false, false);
        check_constraints::<_, SC>(
            &honest,
            "local endpoint authority",
            &None,
            &[authority_trace.as_view()],
            &[],
        );
        check_full_balance(
            &endpoint_airs,
            &endpoint_trace,
            &block_airs,
            &block_traces,
            &honest,
            &authority_trace,
        );

        // Every authenticated point coordinate is constrained before it is
        // republished to BlockEq.
        for row in 0..point.len() {
            let mut bad = endpoint_trace.selector.common.clone();
            let width = FixedMultiAirCompleteLocalSelectorCols::<F>::width();
            let cols: &mut FixedMultiAirCompleteLocalSelectorCols<F> =
                bad.values[row * width..(row + 1) * width].borrow_mut();
            cols.point[0] += F::ONE;
            assert!(catch_unwind(AssertUnwindSafe(|| {
                check_constraints::<_, SC>(
                    &endpoint_airs.selector,
                    "mutated local selector point",
                    &None,
                    &[endpoint_trace.selector.cached.as_view(), bad.as_view()],
                    &[],
                )
            }))
            .is_err());
        }

        // Mutating any dynamic opening at its canonical source row breaks the
        // local node equation, while mutating its setup ordinal breaks the
        // cursor/endpoint typed multiset.
        for opening in 0..openings.len() {
            let row = profile
                .node_sources
                .iter()
                .position(|source| *source == Some(LocalNodeSource::Opening(opening)))
                .unwrap();
            let mut bad_common = endpoint_trace.nodes.common.clone();
            let common_width = FixedMultiAirCompleteLocalNodeCols::<F>::width();
            let cols: &mut FixedMultiAirCompleteLocalNodeCols<F> =
                bad_common.values[row * common_width..(row + 1) * common_width].borrow_mut();
            cols.source_value[0] += F::ONE;
            assert!(catch_unwind(AssertUnwindSafe(|| {
                check_constraints::<_, SC>(
                    &endpoint_airs.nodes,
                    "mutated local opening value",
                    &None,
                    &[endpoint_trace.nodes.cached.as_view(), bad_common.as_view()],
                    &[],
                )
            }))
            .is_err());

            let mut bad_trace = endpoint_trace.clone();
            let cached_width = FixedMultiAirCompleteLocalNodeScheduleCols::<F>::width();
            let schedule: &mut FixedMultiAirCompleteLocalNodeScheduleCols<F> =
                bad_trace.nodes.cached.values[row * cached_width..(row + 1) * cached_width]
                    .borrow_mut();
            schedule.source_index += F::ONE;
            assert!(catch_unwind(AssertUnwindSafe(|| {
                check_full_balance(
                    &endpoint_airs,
                    &bad_trace,
                    &block_airs,
                    &block_traces,
                    &honest,
                    &authority_trace,
                )
            }))
            .is_err());
        }

        for row in 0..profile.component.expression.nodes.len() {
            let mut bad = endpoint_trace.nodes.common.clone();
            let width = FixedMultiAirCompleteLocalNodeCols::<F>::width();
            let cols: &mut FixedMultiAirCompleteLocalNodeCols<F> =
                bad.values[row * width..(row + 1) * width].borrow_mut();
            cols.value[0] += F::ONE;
            assert!(catch_unwind(AssertUnwindSafe(|| {
                check_constraints::<_, SC>(
                    &endpoint_airs.nodes,
                    "mutated local DAG node",
                    &None,
                    &[endpoint_trace.nodes.cached.as_view(), bad.as_view()],
                    &[],
                )
            }))
            .is_err());
        }

        for row in 0..profile.component.constraints.len() {
            let mut bad = endpoint_trace.fold.common.clone();
            let width = FixedMultiAirCompleteLocalFoldCols::<F>::width();
            let cols: &mut FixedMultiAirCompleteLocalFoldCols<F> =
                bad.values[row * width..(row + 1) * width].borrow_mut();
            cols.weight[0] += F::ONE;
            assert!(catch_unwind(AssertUnwindSafe(|| {
                check_constraints::<_, SC>(
                    &endpoint_airs.fold,
                    "mutated local Eq weight",
                    &None,
                    &[endpoint_trace.fold.cached.as_view(), bad.as_view()],
                    &[],
                )
            }))
            .is_err());
        }

        let mut bad_final = authority_trace.clone();
        let width = AuthorityCols::<F>::width();
        let final_row = bad_final
            .values
            .chunks_exact(width)
            .position(|row| {
                let cols: &AuthorityCols<F> = row.borrow();
                cols.kind_flags[AUTH_FINAL_PUBLISH] == F::ONE
            })
            .unwrap();
        let cols: &mut AuthorityCols<F> =
            bad_final.values[final_row * width..(final_row + 1) * width].borrow_mut();
        cols.value[0] += F::ONE;
        assert!(catch_unwind(AssertUnwindSafe(|| {
            check_full_balance(
                &endpoint_airs,
                &endpoint_trace,
                &block_airs,
                &block_traces,
                &honest,
                &bad_final,
            )
        }))
        .is_err());

        let wrong_opening = authority_air(buses, global_point_bus, true, false);
        assert!(catch_unwind(AssertUnwindSafe(|| {
            check_full_balance(
                &endpoint_airs,
                &endpoint_trace,
                &block_airs,
                &block_traces,
                &wrong_opening,
                &authority_trace,
            )
        }))
        .is_err());
        let wrong_final = authority_air(buses, global_point_bus, false, true);
        assert!(catch_unwind(AssertUnwindSafe(|| {
            check_full_balance(
                &endpoint_airs,
                &endpoint_trace,
                &block_airs,
                &block_traces,
                &wrong_final,
                &authority_trace,
            )
        }))
        .is_err());
    }

    #[test]
    fn local_and_interaction_authority_types_cannot_cross_route() {
        assert_ne!(
            TypeId::of::<FixedMultiAirCompleteLocalRegionOpeningBus>(),
            TypeId::of::<FixedMultiAirCompleteInteractionRegionOpeningBus>()
        );
        assert_ne!(
            TypeId::of::<FixedMultiAirCompleteLocalRegionFinalEvaluationBus>(),
            TypeId::of::<FixedMultiAirCompleteInteractionRegionFinalEvaluationBus>()
        );
    }
}

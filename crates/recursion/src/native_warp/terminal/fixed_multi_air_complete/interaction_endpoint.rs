//! Nonlinear endpoint verifier for complete-terminal LogUp interactions.
//!
//! Integration dependencies:
//! - the cursor/openings owner must publish `interaction_opening_bus`, `interaction_point_bus`, and
//!   `final_evaluation_bus` with the exact multiplicities returned by the profile;
//! - the prefix owner must include this module's instance-value and empty-claim lookups in its
//!   setup-fixed producer counts;
//! - one `block_eq` AIR must produce every inverse weight and the single global scalar weight named
//!   by this profile.
//!
//! Local and interaction authorities are different Rust bus types. There is
//! no witness component selector and an empty interaction region has a
//! setup-fixed zero-claim AIR instead of an optional proof path.

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
        FixedMultiAirCompleteTerminalCircuitComponentPlan,
        FixedMultiAirCompleteTerminalCircuitDynamicKind,
        FixedMultiAirCompleteTerminalCircuitFixedColumn,
        FixedMultiAirCompleteTerminalCircuitNodeSource,
        FixedMultiAirCompleteTerminalCircuitRegionPlan, FixedMultiAirCompleteTerminalScalarPlan,
        FIXED_MULTI_AIR_COMPLETE_EXTENSION_DEGREE,
    },
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::{
    block_eq::{
        FixedMultiAirCompleteEqWeightBus, FixedMultiAirCompleteEqWeightMessage,
        FixedMultiAirCompleteRegionalPointBus, FixedMultiAirCompleteRegionalPointMessage,
    },
    FixedMultiAirCompleteInstanceValueBus, FixedMultiAirCompleteInstanceValueMessage,
    FixedMultiAirCompleteInteractionClaimBus,
    FixedMultiAirCompleteInteractionRegionFinalEvaluationBus,
    FixedMultiAirCompleteInteractionRegionOpeningBus,
    FixedMultiAirCompleteInteractionRegionPointBus,
    FixedMultiAirCompleteRegionFinalEvaluationMessage, FixedMultiAirCompleteRegionOpeningMessage,
    FixedMultiAirCompleteRegionPointMessage, FixedMultiAirCompleteRegionalClaimMessage,
};
use crate::{
    define_typed_lookup_bus, define_typed_permutation_bus,
    utils::{ext_field_add, ext_field_multiply, ext_field_subtract},
};

const INSTANCE_SECTION_BETA: usize = 2;
const MAX_EXACT_DEGREE: usize = 6;
const NODE_KIND_COUNT: usize = 6;
const NODE_SOURCE_KIND_COUNT: usize = 4;
const NODE_SOURCE_DYNAMIC: usize = 0;
const NODE_SOURCE_FIXED: usize = 1;
const NODE_SOURCE_PUBLIC: usize = 2;
const NODE_SOURCE_SELECTOR: usize = 3;
const NODE_KIND_SOURCE: usize = 0;
const NODE_KIND_CONSTANT: usize = 1;
const NODE_KIND_ADD: usize = 2;
const NODE_KIND_SUB: usize = 3;
const NODE_KIND_NEG: usize = 4;
const NODE_KIND_MUL: usize = 5;
const DENOMINATOR_KIND_COUNT: usize = 3;
const DENOMINATOR_ALPHA: usize = 0;
const DENOMINATOR_BUS: usize = 1;
const DENOMINATOR_MESSAGE: usize = 2;
const FIXED_KIND_COUNT: usize = 3;
const FIXED_INITIAL: usize = 0;
const FIXED_FOLD: usize = 1;
const FIXED_FINAL: usize = 2;

const _: () = assert!(D_EF == 4);
const _: () = assert!(FIXED_MULTI_AIR_COMPLETE_EXTENSION_DEGREE == D_EF);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteInteractionFixedOpeningMessage<T> {
    pub region: T,
    pub opening: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirCompleteInteractionFixedOpeningBus,
    FixedMultiAirCompleteInteractionFixedOpeningMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteInteractionFixedStateMessage<T> {
    pub region: T,
    pub opening: T,
    pub layer: T,
    pub index: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirCompleteInteractionFixedStateBus,
    FixedMultiAirCompleteInteractionFixedStateMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteInteractionSelectorMessage<T> {
    pub region: T,
    /// `0 = first`, `1 = last`, `2 = transition`.
    pub selector: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirCompleteInteractionSelectorBus,
    FixedMultiAirCompleteInteractionSelectorMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteInteractionNodeMessage<T> {
    pub region: T,
    pub node: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirCompleteInteractionNodeBus,
    FixedMultiAirCompleteInteractionNodeMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteInteractionDenominatorMessage<T> {
    pub region: T,
    pub interaction: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirCompleteInteractionDenominatorBus,
    FixedMultiAirCompleteInteractionDenominatorMessage
);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedMultiAirCompleteInteractionEndpointError {
    Configuration(&'static str),
    Shape(&'static str),
    Arithmetic(&'static str),
    Witness(&'static str),
    EmptyRegion,
    NonEmptyRegion,
}

/// Narrow setup-only input used until the complete circuit owner exposes a
/// single constructor over its full circuit plan.
#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteInteractionEndpointPlanInput<'a> {
    pub global_log_constraints: usize,
    pub global_explicit_len: usize,
    pub exact_relation_degree: usize,
    pub region: &'a FixedMultiAirCompleteTerminalCircuitRegionPlan<F>,
    pub scalars: &'a FixedMultiAirCompleteTerminalScalarPlan,
    pub inverse_weight_blocks: &'a [usize],
    pub global_weight_block: usize,
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteInteractionEndpointProfile {
    pub region: usize,
    pub log_height: usize,
    pub exact_relation_degree: usize,
    pub component: FixedMultiAirCompleteTerminalCircuitComponentPlan<F>,
    pub one_beta_coordinate: usize,
    pub alpha_beta_coordinate: usize,
    pub beta_power_beta_start: usize,
    pub beta_power_count: usize,
    pub public_beta_start: usize,
    pub public_count: usize,
    pub inverse_weight_blocks: Vec<usize>,
    pub global_weight_block: usize,
    node_fanout: Vec<usize>,
    selector_lookup_counts: [usize; 3],
    opening_lookup_counts: Vec<usize>,
    fixed_opening_lookup_counts: Vec<usize>,
    instance_beta_lookup_counts: Vec<usize>,
}

impl FixedMultiAirCompleteInteractionEndpointProfile {
    pub fn new(
        input: FixedMultiAirCompleteInteractionEndpointPlanInput<'_>,
    ) -> Result<Self, FixedMultiAirCompleteInteractionEndpointError> {
        if input.exact_relation_degree == 0 || input.exact_relation_degree > MAX_EXACT_DEGREE {
            return Err(
                FixedMultiAirCompleteInteractionEndpointError::Configuration(
                    "exact relation degree",
                ),
            );
        }
        let region = usize::try_from(input.region.interaction.identity.region_ordinal)
            .map_err(|_| FixedMultiAirCompleteInteractionEndpointError::Arithmetic("region"))?;
        if input.region.setup_ordinal as usize != region
            || input.region.local.identity.region_ordinal as usize != region
            || input.region.interaction.identity.kind
                != FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction
        {
            return Err(
                FixedMultiAirCompleteInteractionEndpointError::Configuration("regional identity"),
            );
        }
        let component = input.region.interaction.clone();
        let interaction_count = component.interactions.len();
        if component.identity.proof_present != (interaction_count != 0)
            || component.identity.proof_ordinal.is_some() != component.identity.proof_present
            || component.identity.opening_count as usize
                != component.expression.dynamic_columns.len()
        {
            return Err(
                FixedMultiAirCompleteInteractionEndpointError::Configuration(
                    "interaction proof presence",
                ),
            );
        }
        if input.inverse_weight_blocks.len() != interaction_count {
            return Err(
                FixedMultiAirCompleteInteractionEndpointError::Configuration(
                    "inverse weight block count",
                ),
            );
        }
        let log_height = usize::from(input.region.log_height);
        if interaction_count == 0 {
            if !component.expression.nodes.is_empty()
                || !component.expression.homogeneous_degrees.is_empty()
                || !component.expression.dynamic_columns.is_empty()
                || !component.expression.fixed_columns.is_empty()
                || !component.expression.node_sources.is_empty()
                || !component.constraints.is_empty()
            {
                return Err(
                    FixedMultiAirCompleteInteractionEndpointError::Configuration(
                        "empty interaction expression",
                    ),
                );
            }
        } else {
            validate_expression(&component, log_height, input.exact_relation_degree)?;
        }
        validate_interactions(
            &component,
            log_height,
            input.exact_relation_degree,
            input.scalars.beta_power_count as usize,
        )?;

        let one_beta_coordinate = checked_coordinate(
            input.global_log_constraints,
            input.scalars.distinguished_one_explicit_offset,
            "one beta coordinate",
        )?;
        let alpha_beta_coordinate = checked_coordinate(
            input.global_log_constraints,
            input.scalars.alpha_explicit_offset,
            "alpha beta coordinate",
        )?;
        let beta_power_beta_start = checked_coordinate(
            input.global_log_constraints,
            input.scalars.beta_power_explicit_offset,
            "beta power coordinate",
        )?;
        let public_beta_start = checked_coordinate(
            input.global_log_constraints,
            input.region.local_explicit_offset,
            "public beta coordinate",
        )?;
        let public_count = usize::try_from(input.region.local_explicit_len).map_err(|_| {
            FixedMultiAirCompleteInteractionEndpointError::Arithmetic("public count")
        })?;
        let beta_power_count = usize::try_from(input.scalars.beta_power_count).map_err(|_| {
            FixedMultiAirCompleteInteractionEndpointError::Arithmetic("beta power count")
        })?;
        let beta_len = input
            .global_log_constraints
            .checked_add(input.global_explicit_len)
            .ok_or(FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                "interaction beta length",
            ))?;
        let beta_power_end = beta_power_beta_start.checked_add(beta_power_count).ok_or(
            FixedMultiAirCompleteInteractionEndpointError::Arithmetic("interaction beta-power end"),
        )?;
        let public_end = public_beta_start.checked_add(public_count).ok_or(
            FixedMultiAirCompleteInteractionEndpointError::Arithmetic("interaction public end"),
        )?;
        if one_beta_coordinate >= beta_len
            || alpha_beta_coordinate >= beta_len
            || beta_power_end > beta_len
            || public_end > beta_len
        {
            return Err(
                FixedMultiAirCompleteInteractionEndpointError::Configuration(
                    "interaction beta coordinates",
                ),
            );
        }

        let mut node_fanout = vec![0usize; component.expression.nodes.len()];
        let mut selector_lookup_counts = [0usize; 3];
        let mut opening_lookup_counts = vec![0usize; component.expression.dynamic_columns.len()];
        let mut fixed_opening_lookup_counts =
            vec![0usize; component.expression.fixed_columns.len()];
        let mut instance_beta_lookup_counts = vec![0usize; beta_len];
        for (node_index, node) in component.expression.nodes.iter().enumerate() {
            checked_increment(
                &mut instance_beta_lookup_counts,
                one_beta_coordinate,
                "node distinguished-one lookup",
            )?;
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
                    checked_increment(&mut node_fanout, *idx, "negative node fanout")?;
                }
                SymbolicExpressionNode::Variable(variable) => match variable.entry {
                    Entry::Main { .. } | Entry::Preprocessed { .. } => {
                        match component.expression.node_sources[node_index] {
                            Some(FixedMultiAirCompleteTerminalCircuitNodeSource::Dynamic(
                                source,
                            )) => {
                                checked_increment(
                                    &mut opening_lookup_counts,
                                    source as usize,
                                    "dynamic opening lookup",
                                )?;
                            }
                            Some(FixedMultiAirCompleteTerminalCircuitNodeSource::Fixed(source)) => {
                                checked_increment(
                                    &mut fixed_opening_lookup_counts,
                                    source as usize,
                                    "fixed opening lookup",
                                )?;
                            }
                            None => {
                                return Err(
                                    FixedMultiAirCompleteInteractionEndpointError::Configuration(
                                        "missing variable source",
                                    ),
                                )
                            }
                        }
                    }
                    Entry::Public => {
                        if variable.index >= public_count {
                            return Err(
                                FixedMultiAirCompleteInteractionEndpointError::Configuration(
                                    "public variable range",
                                ),
                            );
                        }
                        let coordinate = public_beta_start.checked_add(variable.index).ok_or(
                            FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                                "public instance coordinate",
                            ),
                        )?;
                        checked_increment(
                            &mut instance_beta_lookup_counts,
                            coordinate,
                            "public instance lookup",
                        )?;
                    }
                    Entry::Challenge => {
                        return Err(
                            FixedMultiAirCompleteInteractionEndpointError::Configuration(
                                "challenge variable",
                            ),
                        )
                    }
                },
                SymbolicExpressionNode::IsFirstRow => selector_lookup_counts[0] += 1,
                SymbolicExpressionNode::IsLastRow => selector_lookup_counts[1] += 1,
                SymbolicExpressionNode::IsTransition => selector_lookup_counts[2] += 1,
                SymbolicExpressionNode::Constant(_) => {}
            }
        }
        for interaction in &component.interactions {
            for &root in &interaction.message_roots {
                checked_increment(&mut node_fanout, root as usize, "message root fanout")?;
            }
            checked_increment(
                &mut node_fanout,
                interaction.count_root as usize,
                "count root fanout",
            )?;
            for &source in &interaction.q_source_ordinals {
                checked_increment(
                    &mut opening_lookup_counts,
                    source as usize,
                    "q opening lookup",
                )?;
            }
            let denominator_terms = interaction.message_roots.len().checked_add(2).ok_or(
                FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                    "denominator instance terms",
                ),
            )?;
            checked_add_to_count(
                &mut instance_beta_lookup_counts,
                one_beta_coordinate,
                denominator_terms,
                "denominator distinguished-one lookups",
            )?;
            checked_increment(
                &mut instance_beta_lookup_counts,
                alpha_beta_coordinate,
                "denominator alpha lookup",
            )?;
            let bus_coordinate = beta_power_beta_start
                .checked_add(interaction.bus_beta_power_ordinal as usize)
                .ok_or(FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                    "denominator bus coordinate",
                ))?;
            checked_increment(
                &mut instance_beta_lookup_counts,
                bus_coordinate,
                "denominator bus lookup",
            )?;
            for message in 0..interaction.message_roots.len() {
                let coordinate = beta_power_beta_start.checked_add(message).ok_or(
                    FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                        "denominator message coordinate",
                    ),
                )?;
                checked_increment(
                    &mut instance_beta_lookup_counts,
                    coordinate,
                    "denominator message lookup",
                )?;
            }
            checked_increment(
                &mut instance_beta_lookup_counts,
                one_beta_coordinate,
                "endpoint distinguished-one lookup",
            )?;
        }
        Ok(Self {
            region,
            log_height,
            exact_relation_degree: input.exact_relation_degree,
            component,
            one_beta_coordinate,
            alpha_beta_coordinate,
            beta_power_beta_start,
            beta_power_count,
            public_beta_start,
            public_count,
            inverse_weight_blocks: input.inverse_weight_blocks.to_vec(),
            global_weight_block: input.global_weight_block,
            node_fanout,
            selector_lookup_counts,
            opening_lookup_counts,
            fixed_opening_lookup_counts,
            instance_beta_lookup_counts,
        })
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.component.interactions.is_empty()
    }

    #[must_use]
    pub fn interaction_count(&self) -> usize {
        self.component.interactions.len()
    }

    #[must_use]
    pub fn selector_lookup_counts(&self) -> [usize; 3] {
        self.selector_lookup_counts
    }

    #[must_use]
    pub fn opening_lookup_counts(&self) -> &[usize] {
        &self.opening_lookup_counts
    }

    #[must_use]
    pub fn fixed_opening_lookup_counts(&self) -> &[usize] {
        &self.fixed_opening_lookup_counts
    }

    /// Exact aggregate producer multiplicity for every coordinate in the
    /// global beta/explicit instance section. The inventory covers all node,
    /// denominator, and nonlinear-endpoint lookups; fixed and selector AIRs
    /// intentionally contribute none.
    #[must_use]
    pub fn instance_beta_lookup_counts(&self) -> &[usize] {
        &self.instance_beta_lookup_counts
    }

    /// One endpoint selector fold plus any downstream global-mapping uses
    /// must be included by the owner when configuring the point producer.
    #[must_use]
    pub const fn endpoint_point_lookup_count(&self) -> usize {
        1
    }

    /// Number of independent setup-fixed column folds rooted at every
    /// interaction sumcheck point coordinate. This is the cursor profile's
    /// `point_fixed_source_count`; it is deliberately separate from the
    /// selector/global-mapper `point_common_count`.
    #[must_use]
    pub fn point_fixed_source_count(&self) -> usize {
        self.component.expression.fixed_columns.len()
    }
}

fn checked_coordinate(
    prefix: usize,
    offset: u64,
    context: &'static str,
) -> Result<usize, FixedMultiAirCompleteInteractionEndpointError> {
    prefix
        .checked_add(
            usize::try_from(offset)
                .map_err(|_| FixedMultiAirCompleteInteractionEndpointError::Arithmetic(context))?,
        )
        .ok_or(FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
            context,
        ))
}

fn checked_increment(
    counts: &mut [usize],
    index: usize,
    context: &'static str,
) -> Result<(), FixedMultiAirCompleteInteractionEndpointError> {
    checked_add_to_count(counts, index, 1, context)
}

fn checked_add_to_count(
    counts: &mut [usize],
    index: usize,
    addend: usize,
    context: &'static str,
) -> Result<(), FixedMultiAirCompleteInteractionEndpointError> {
    let count = counts
        .get_mut(index)
        .ok_or(FixedMultiAirCompleteInteractionEndpointError::Configuration(context))?;
    *count = count.checked_add(addend).ok_or(
        FixedMultiAirCompleteInteractionEndpointError::Arithmetic(context),
    )?;
    Ok(())
}

fn validate_expression(
    component: &FixedMultiAirCompleteTerminalCircuitComponentPlan<F>,
    log_height: usize,
    exact_degree: usize,
) -> Result<(), FixedMultiAirCompleteInteractionEndpointError> {
    let expression = &component.expression;
    if expression.nodes.is_empty()
        || expression.nodes.len() != expression.homogeneous_degrees.len()
        || expression.nodes.len() != expression.node_sources.len()
        || expression
            .homogeneous_degrees
            .iter()
            .any(|&degree| usize::from(degree) > exact_degree)
        || expression.dynamic_columns.iter().any(|column| {
            column.log_height as usize != log_height
                || column.rotation > 1
                || !matches!(
                    column.kind,
                    FixedMultiAirCompleteTerminalCircuitDynamicKind::Trace
                        | FixedMultiAirCompleteTerminalCircuitDynamicKind::Inverse
                )
        })
        || expression.fixed_columns.iter().any(|column| match column {
            openvm_stark_backend::native_warp::FixedMultiAirCompleteTerminalCircuitFixedColumn::Cached {
                rotation,
                ..
            }
            | openvm_stark_backend::native_warp::FixedMultiAirCompleteTerminalCircuitFixedColumn::Preprocessed {
                rotation,
                ..
            } => *rotation > 1,
        })
    {
        return Err(FixedMultiAirCompleteInteractionEndpointError::Configuration(
            "interaction expression dimensions",
        ));
    }
    for (index, node) in expression.nodes.iter().enumerate() {
        let source = expression.node_sources[index];
        match node {
            SymbolicExpressionNode::Variable(variable) => match variable.entry {
                Entry::Main { .. } | Entry::Preprocessed { .. } if source.is_none() => {
                    return Err(
                        FixedMultiAirCompleteInteractionEndpointError::Configuration(
                            "missing interaction node source",
                        ),
                    )
                }
                Entry::Public if source.is_some() => {
                    return Err(
                        FixedMultiAirCompleteInteractionEndpointError::Configuration(
                            "public node has column source",
                        ),
                    )
                }
                Entry::Challenge => {
                    return Err(
                        FixedMultiAirCompleteInteractionEndpointError::Configuration(
                            "interaction challenge node",
                        ),
                    )
                }
                _ => {}
            },
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
                if *left_idx >= index || *right_idx >= index || source.is_some() {
                    return Err(
                        FixedMultiAirCompleteInteractionEndpointError::Configuration(
                            "interaction DAG topology",
                        ),
                    );
                }
            }
            SymbolicExpressionNode::Neg { idx, .. } => {
                if *idx >= index || source.is_some() {
                    return Err(
                        FixedMultiAirCompleteInteractionEndpointError::Configuration(
                            "interaction DAG topology",
                        ),
                    );
                }
            }
            _ if source.is_some() => {
                return Err(
                    FixedMultiAirCompleteInteractionEndpointError::Configuration(
                        "non-variable node source",
                    ),
                )
            }
            _ => {}
        }
    }
    Ok(())
}

fn validate_interactions(
    component: &FixedMultiAirCompleteTerminalCircuitComponentPlan<F>,
    log_height: usize,
    exact_degree: usize,
    beta_power_count: usize,
) -> Result<(), FixedMultiAirCompleteInteractionEndpointError> {
    for (ordinal, interaction) in component.interactions.iter().enumerate() {
        let expected_inverse_degree = interaction.denominator_degree.checked_add(1).ok_or(
            FixedMultiAirCompleteInteractionEndpointError::Arithmetic("inverse degree"),
        )?;
        let expected_global_term_degree = interaction.count_degree.checked_add(1).ok_or(
            FixedMultiAirCompleteInteractionEndpointError::Arithmetic("global term degree"),
        )?;
        if interaction.interaction_ordinal as usize != ordinal
            || interaction.message_roots.len() >= beta_power_count
            || interaction.bus_beta_power_ordinal as usize != interaction.message_roots.len()
            || interaction.bus_index_plus_one != u32::from(interaction.bus_index) + 1
            || interaction.denominator_degree == 0
            || interaction.inverse_degree != expected_inverse_degree
            || interaction.inverse_degree as usize > exact_degree
            || interaction.global_term_degree != expected_global_term_degree
            || interaction.global_term_degree as usize > exact_degree
            || interaction.inverse_constraint.eq.log_height as usize != log_height
            || interaction.inverse_constraint.eq.rotation != 0
            || interaction
                .q_source_ordinals
                .iter()
                .any(|&source| source as usize >= component.expression.dynamic_columns.len())
        {
            return Err(
                FixedMultiAirCompleteInteractionEndpointError::Configuration(
                    "canonical interaction plan",
                ),
            );
        }
        if !matches!(
            interaction.inverse_constraint.role,
            openvm_stark_backend::native_warp::FixedMultiAirCompleteTerminalEqRole::InverseConstraint {
                interaction_ordinal
            } if interaction_ordinal as usize == ordinal
        ) {
            return Err(
                FixedMultiAirCompleteInteractionEndpointError::Configuration("inverse Eq role"),
            );
        }
        for &root in interaction
            .message_roots
            .iter()
            .chain(core::iter::once(&interaction.count_root))
        {
            if root as usize >= component.expression.nodes.len() {
                return Err(
                    FixedMultiAirCompleteInteractionEndpointError::Configuration(
                        "interaction node root",
                    ),
                );
            }
        }
    }
    Ok(())
}

/// Immutable setup tables in canonical interaction
/// `component.expression.fixed_columns` order. These values are proving-key
/// data: they are folded by an AIR at the authenticated interaction point and
/// are never accepted from a witness as an opening authority.
#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteInteractionFixedValues {
    pub region: usize,
    pub log_height: usize,
    descriptors: Vec<FixedMultiAirCompleteTerminalCircuitFixedColumn>,
    tables: Vec<Vec<F>>,
}

impl FixedMultiAirCompleteInteractionFixedValues {
    pub fn from_relation_setup(
        profile: &FixedMultiAirCompleteInteractionEndpointProfile,
        cached_parts: &[FixedMultiAirCompleteCachedTrace<F>],
        preprocessed: Option<&DirectAirFixedTrace<F>>,
    ) -> Result<Self, FixedMultiAirCompleteInteractionEndpointError> {
        let height = checked_pow2(profile.log_height, "interaction fixed-table height")?;
        let mut tables = Vec::new();
        tables
            .try_reserve_exact(profile.component.expression.fixed_columns.len())
            .map_err(|_| {
                FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                    "interaction fixed-table allocation",
                )
            })?;
        for descriptor in &profile.component.expression.fixed_columns {
            let (width, values, column, rotation) = match *descriptor {
                FixedMultiAirCompleteTerminalCircuitFixedColumn::Cached {
                    part,
                    column,
                    rotation,
                } => {
                    let part = usize::try_from(part).map_err(|_| {
                        FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                            "interaction cached part",
                        )
                    })?;
                    let trace = cached_parts.get(part).ok_or(
                        FixedMultiAirCompleteInteractionEndpointError::Configuration(
                            "interaction cached part",
                        ),
                    )?;
                    (
                        usize::try_from(trace.width).map_err(|_| {
                            FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                                "interaction cached width",
                            )
                        })?,
                        trace.values.as_slice(),
                        usize::try_from(column).map_err(|_| {
                            FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                                "interaction cached column",
                            )
                        })?,
                        usize::from(rotation),
                    )
                }
                FixedMultiAirCompleteTerminalCircuitFixedColumn::Preprocessed {
                    column,
                    rotation,
                } => {
                    let trace = preprocessed.ok_or(
                        FixedMultiAirCompleteInteractionEndpointError::Configuration(
                            "interaction preprocessed trace",
                        ),
                    )?;
                    (
                        usize::try_from(trace.width).map_err(|_| {
                            FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                                "interaction preprocessed width",
                            )
                        })?,
                        trace.values.as_slice(),
                        usize::try_from(column).map_err(|_| {
                            FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                                "interaction preprocessed column",
                            )
                        })?,
                        usize::from(rotation),
                    )
                }
            };
            let expected_cells = height.checked_mul(width).ok_or(
                FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                    "interaction fixed-table cells",
                ),
            )?;
            if width == 0 || column >= width || rotation > 1 || values.len() != expected_cells {
                return Err(
                    FixedMultiAirCompleteInteractionEndpointError::Configuration(
                        "interaction fixed-table shape",
                    ),
                );
            }
            let mut table = try_zero_vec(height, "interaction fixed table")?;
            for (row, output) in table.iter_mut().enumerate() {
                let source_row = row.checked_add(rotation).ok_or(
                    FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                        "interaction fixed rotation",
                    ),
                )? & (height - 1);
                let cell = source_row
                    .checked_mul(width)
                    .and_then(|start| start.checked_add(column))
                    .ok_or(FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                        "interaction fixed cell",
                    ))?;
                *output = *values.get(cell).ok_or(
                    FixedMultiAirCompleteInteractionEndpointError::Configuration(
                        "interaction fixed cell",
                    ),
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
        profile: &FixedMultiAirCompleteInteractionEndpointProfile,
    ) -> Result<(), FixedMultiAirCompleteInteractionEndpointError> {
        let height = checked_pow2(profile.log_height, "interaction fixed-table height")?;
        if self.region != profile.region
            || self.log_height != profile.log_height
            || self.descriptors != profile.component.expression.fixed_columns
            || self.tables.len() != self.descriptors.len()
            || self.tables.iter().any(|table| table.len() != height)
        {
            return Err(
                FixedMultiAirCompleteInteractionEndpointError::Configuration(
                    "interaction fixed setup substitution",
                ),
            );
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteInteractionFixedScheduleLaneCols<T> {
    pub active: T,
    pub kind_flags: [T; FIXED_KIND_COUNT],
    pub opening: T,
    pub layer: T,
    pub index: T,
    pub low_index: T,
    pub high_index: T,
    pub point_coordinate: T,
    pub initial_value: [T; D_EF],
    pub opening_lookup_count: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteInteractionFixedLaneCols<T> {
    pub low: [T; D_EF],
    pub high: [T; D_EF],
    pub point: [T; D_EF],
    pub value: [T; D_EF],
}

/// Folds every setup-fixed interaction column at the exact regional point.
/// The cached schedule contains all table cells and routing ordinals, while
/// the common trace contains only algebraically constrained fold state.
pub struct FixedMultiAirCompleteInteractionFixedAir {
    pub profile: Arc<FixedMultiAirCompleteInteractionEndpointProfile>,
    pub setup: Arc<FixedMultiAirCompleteInteractionFixedValues>,
    pub point_bus: FixedMultiAirCompleteInteractionRegionPointBus,
    pub state_bus: FixedMultiAirCompleteInteractionFixedStateBus,
    pub opening_bus: FixedMultiAirCompleteInteractionFixedOpeningBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteInteractionFixedAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteInteractionFixedAir {
    fn common_main_width(&self) -> usize {
        INTERACTION_FIXED_LANES * FixedMultiAirCompleteInteractionFixedLaneCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteInteractionFixedAir {}
impl BaseAir<F> for FixedMultiAirCompleteInteractionFixedAir {
    fn width(&self) -> usize {
        self.common_main_width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        Some(
            generate_fixed_multi_air_complete_interaction_fixed_preprocessed_trace(
                &self.profile,
                &self.setup,
                None,
            )
            .expect("validated interaction fixed preprocessed trace"),
        )
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteInteractionFixedAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder + PairBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let preprocessed = builder
            .preprocessed()
            .row_slice(0)
            .expect("interaction fixed schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("interaction fixed row")
            .to_vec();
        let schedule_width =
            FixedMultiAirCompleteInteractionFixedScheduleLaneCols::<AB::Var>::width();
        let common_width = FixedMultiAirCompleteInteractionFixedLaneCols::<AB::Var>::width();
        for lane in 0..INTERACTION_FIXED_LANES {
            let schedule: &FixedMultiAirCompleteInteractionFixedScheduleLaneCols<AB::Var> =
                preprocessed[lane * schedule_width..(lane + 1) * schedule_width].borrow();
            let local: &FixedMultiAirCompleteInteractionFixedLaneCols<AB::Var> =
                common[lane * common_width..(lane + 1) * common_width].borrow();
            eval_interaction_fixed_lane(self, builder, schedule, local);
        }
    }
}

const INTERACTION_FIXED_LANES: usize = 2;

fn eval_interaction_fixed_lane<AB>(
    air: &FixedMultiAirCompleteInteractionFixedAir,
    builder: &mut AB,
    schedule: &FixedMultiAirCompleteInteractionFixedScheduleLaneCols<AB::Var>,
    local: &FixedMultiAirCompleteInteractionFixedLaneCols<AB::Var>,
) where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    for flag in schedule.kind_flags {
        builder.assert_bool(flag);
    }
    let active = schedule
        .kind_flags
        .iter()
        .fold(AB::Expr::ZERO, |sum, &flag| sum + flag);
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

    air.point_bus.lookup_key(
        builder,
        FixedMultiAirCompleteRegionPointMessage {
            region: AB::Expr::from_usize(air.profile.region),
            coordinate: schedule.point_coordinate.into(),
            value: local.point.map(Into::into),
        },
        schedule.kind_flags[FIXED_FOLD],
    );
    air.state_bus.receive(
        builder,
        FixedMultiAirCompleteInteractionFixedStateMessage {
            region: AB::Expr::from_usize(air.profile.region),
            opening: schedule.opening.into(),
            layer: AB::Expr::from(schedule.layer) - AB::Expr::ONE,
            index: schedule.low_index.into(),
            value: local.low.map(Into::into),
        },
        schedule.kind_flags[FIXED_FOLD],
    );
    air.state_bus.receive(
        builder,
        FixedMultiAirCompleteInteractionFixedStateMessage {
            region: AB::Expr::from_usize(air.profile.region),
            opening: schedule.opening.into(),
            layer: AB::Expr::from(schedule.layer) - AB::Expr::ONE,
            index: schedule.high_index.into(),
            value: local.high.map(Into::into),
        },
        schedule.kind_flags[FIXED_FOLD],
    );
    air.state_bus.send(
        builder,
        FixedMultiAirCompleteInteractionFixedStateMessage {
            region: AB::Expr::from_usize(air.profile.region),
            opening: schedule.opening.into(),
            layer: schedule.layer.into(),
            index: schedule.index.into(),
            value: local.value.map(Into::into),
        },
        AB::Expr::from(schedule.kind_flags[FIXED_INITIAL])
            + AB::Expr::from(schedule.kind_flags[FIXED_FOLD]),
    );
    air.state_bus.receive(
        builder,
        FixedMultiAirCompleteInteractionFixedStateMessage {
            region: AB::Expr::from_usize(air.profile.region),
            opening: schedule.opening.into(),
            layer: schedule.layer.into(),
            index: AB::Expr::ZERO,
            value: local.low.map(Into::into),
        },
        schedule.kind_flags[FIXED_FINAL],
    );
    air.opening_bus.add_key_with_lookups(
        builder,
        FixedMultiAirCompleteInteractionFixedOpeningMessage {
            region: AB::Expr::from_usize(air.profile.region),
            opening: schedule.opening.into(),
            value: local.value.map(Into::into),
        },
        schedule.opening_lookup_count,
    );
}

#[derive(Clone, Debug)]
struct InteractionFixedScheduleEntry {
    kind: usize,
    opening: usize,
    layer: usize,
    index: usize,
    low_index: usize,
    high_index: usize,
    point_coordinate: usize,
    initial_value: EF,
    opening_lookup_count: usize,
    low: EF,
    high: EF,
    point: EF,
    value: EF,
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteInteractionFixedTrace {
    pub preprocessed: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub values: Vec<EF>,
}

fn interaction_fixed_schedule_entries(
    profile: &FixedMultiAirCompleteInteractionEndpointProfile,
    setup: &FixedMultiAirCompleteInteractionFixedValues,
) -> Result<Vec<InteractionFixedScheduleEntry>, FixedMultiAirCompleteInteractionEndpointError> {
    setup.validate(profile)?;
    let table_height = checked_pow2(profile.log_height, "interaction fixed-table height")?;
    let row_capacity = setup
        .tables
        .len()
        .checked_mul(table_height)
        .and_then(|cells| cells.checked_mul(2))
        .ok_or(FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
            "interaction fixed schedule rows",
        ))?;
    let mut entries = Vec::new();
    entries.try_reserve_exact(row_capacity).map_err(|_| {
        FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
            "interaction fixed schedule allocation",
        )
    })?;
    for (opening, table) in setup.tables.iter().enumerate() {
        for (index, &value) in table.iter().enumerate() {
            entries.push(InteractionFixedScheduleEntry {
                kind: FIXED_INITIAL,
                opening,
                layer: 0,
                index,
                low_index: 0,
                high_index: 0,
                point_coordinate: 0,
                initial_value: EF::from(value),
                opening_lookup_count: 0,
                low: EF::ZERO,
                high: EF::ZERO,
                point: EF::ZERO,
                value: EF::ZERO,
            });
        }
        let mut layer_len = table.len();
        for coordinate in 0..profile.log_height {
            let half = layer_len / 2;
            if half == 0 || layer_len != 2 * half {
                return Err(
                    FixedMultiAirCompleteInteractionEndpointError::Configuration(
                        "interaction fixed fold layer",
                    ),
                );
            }
            for index in 0..half {
                entries.push(InteractionFixedScheduleEntry {
                    kind: FIXED_FOLD,
                    opening,
                    layer: coordinate + 1,
                    index,
                    low_index: index,
                    high_index: index + half,
                    point_coordinate: coordinate,
                    initial_value: EF::ZERO,
                    opening_lookup_count: 0,
                    low: EF::ZERO,
                    high: EF::ZERO,
                    point: EF::ZERO,
                    value: EF::ZERO,
                });
            }
            layer_len = half;
        }
        if layer_len != 1 {
            return Err(
                FixedMultiAirCompleteInteractionEndpointError::Configuration(
                    "interaction fixed final",
                ),
            );
        }
        entries.push(InteractionFixedScheduleEntry {
            kind: FIXED_FINAL,
            opening,
            layer: profile.log_height,
            index: 0,
            low_index: 0,
            high_index: 0,
            point_coordinate: 0,
            initial_value: EF::ZERO,
            opening_lookup_count: profile.fixed_opening_lookup_counts[opening],
            low: EF::ZERO,
            high: EF::ZERO,
            point: EF::ZERO,
            value: EF::ZERO,
        });
    }
    Ok(entries)
}

pub fn generate_fixed_multi_air_complete_interaction_fixed_preprocessed_trace(
    profile: &FixedMultiAirCompleteInteractionEndpointProfile,
    setup: &FixedMultiAirCompleteInteractionFixedValues,
    required_height: Option<usize>,
) -> Result<RowMajorMatrix<F>, FixedMultiAirCompleteInteractionEndpointError> {
    let entries = interaction_fixed_schedule_entries(profile, setup)?;
    let packed_rows = entries.len().div_ceil(INTERACTION_FIXED_LANES);
    let height = if packed_rows == 0 {
        match required_height {
            Some(height) if height.is_power_of_two() => height,
            Some(_) => {
                return Err(FixedMultiAirCompleteInteractionEndpointError::Shape(
                    "interaction fixed trace height",
                ))
            }
            None => 1,
        }
    } else {
        checked_trace_height(packed_rows, required_height)?
    };
    let lane_width = FixedMultiAirCompleteInteractionFixedScheduleLaneCols::<F>::width();
    let width = INTERACTION_FIXED_LANES * lane_width;
    let mut values = try_zero_cells(height, width, "interaction fixed preprocessed")?;
    for (entry_index, entry) in entries.iter().enumerate() {
        let row = entry_index / INTERACTION_FIXED_LANES;
        let lane = entry_index % INTERACTION_FIXED_LANES;
        let start = row * width + lane * lane_width;
        let schedule: &mut FixedMultiAirCompleteInteractionFixedScheduleLaneCols<F> =
            values[start..start + lane_width].borrow_mut();
        schedule.active = F::ONE;
        schedule.kind_flags[entry.kind] = F::ONE;
        schedule.opening = F::from_usize(entry.opening);
        schedule.layer = F::from_usize(entry.layer);
        schedule.index = F::from_usize(entry.index);
        schedule.low_index = F::from_usize(entry.low_index);
        schedule.high_index = F::from_usize(entry.high_index);
        schedule.point_coordinate = F::from_usize(entry.point_coordinate);
        copy_ext(&mut schedule.initial_value, entry.initial_value);
        schedule.opening_lookup_count = F::from_usize(entry.opening_lookup_count);
    }
    Ok(RowMajorMatrix::new(values, width))
}

pub fn generate_fixed_multi_air_complete_interaction_fixed_trace(
    profile: &FixedMultiAirCompleteInteractionEndpointProfile,
    setup: &FixedMultiAirCompleteInteractionFixedValues,
    point: &[EF],
    required_height: Option<usize>,
) -> Result<FixedMultiAirCompleteInteractionFixedTrace, FixedMultiAirCompleteInteractionEndpointError>
{
    setup.validate(profile)?;
    if point.len() != profile.log_height {
        return Err(FixedMultiAirCompleteInteractionEndpointError::Shape(
            "interaction fixed point",
        ));
    }
    let table_height = checked_pow2(profile.log_height, "interaction fixed-table height")?;
    let row_capacity = setup
        .tables
        .len()
        .checked_mul(table_height)
        .and_then(|cells| cells.checked_mul(2))
        .ok_or(FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
            "interaction fixed schedule rows",
        ))?;
    let mut entries = Vec::new();
    entries.try_reserve_exact(row_capacity).map_err(|_| {
        FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
            "interaction fixed schedule allocation",
        )
    })?;
    let mut final_values = Vec::new();
    final_values
        .try_reserve_exact(setup.tables.len())
        .map_err(|_| {
            FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                "interaction fixed values allocation",
            )
        })?;

    for (opening, table) in setup.tables.iter().enumerate() {
        let mut layer_values = Vec::new();
        layer_values.try_reserve_exact(table.len()).map_err(|_| {
            FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                "interaction fixed layer allocation",
            )
        })?;
        layer_values.extend(table.iter().copied().map(EF::from));
        for (index, &value) in layer_values.iter().enumerate() {
            entries.push(InteractionFixedScheduleEntry {
                kind: FIXED_INITIAL,
                opening,
                layer: 0,
                index,
                low_index: 0,
                high_index: 0,
                point_coordinate: 0,
                initial_value: value,
                opening_lookup_count: 0,
                low: EF::ZERO,
                high: EF::ZERO,
                point: EF::ZERO,
                value,
            });
        }
        for (coordinate, &challenge) in point.iter().enumerate() {
            let half = layer_values.len() / 2;
            if half == 0 || layer_values.len() != 2 * half {
                return Err(
                    FixedMultiAirCompleteInteractionEndpointError::Configuration(
                        "interaction fixed fold layer",
                    ),
                );
            }
            let mut next = Vec::new();
            next.try_reserve_exact(half).map_err(|_| {
                FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                    "interaction fixed fold allocation",
                )
            })?;
            for index in 0..half {
                let low = layer_values[index];
                let high = layer_values[index + half];
                let value = low + challenge * (high - low);
                next.push(value);
                entries.push(InteractionFixedScheduleEntry {
                    kind: FIXED_FOLD,
                    opening,
                    layer: coordinate + 1,
                    index,
                    low_index: index,
                    high_index: index + half,
                    point_coordinate: coordinate,
                    initial_value: EF::ZERO,
                    opening_lookup_count: 0,
                    low,
                    high,
                    point: challenge,
                    value,
                });
            }
            layer_values = next;
        }
        let value = *layer_values.first().ok_or(
            FixedMultiAirCompleteInteractionEndpointError::Configuration("interaction fixed final"),
        )?;
        if layer_values.len() != 1 {
            return Err(
                FixedMultiAirCompleteInteractionEndpointError::Configuration(
                    "interaction fixed final",
                ),
            );
        }
        final_values.push(value);
        entries.push(InteractionFixedScheduleEntry {
            kind: FIXED_FINAL,
            opening,
            layer: profile.log_height,
            index: 0,
            low_index: 0,
            high_index: 0,
            point_coordinate: 0,
            initial_value: EF::ZERO,
            opening_lookup_count: profile.fixed_opening_lookup_counts[opening],
            low: value,
            high: EF::ZERO,
            point: EF::ZERO,
            value,
        });
    }

    let packed_rows = entries.len().div_ceil(INTERACTION_FIXED_LANES);
    let height = if packed_rows == 0 {
        match required_height {
            Some(height) if height.is_power_of_two() => height,
            Some(_) => {
                return Err(FixedMultiAirCompleteInteractionEndpointError::Shape(
                    "interaction fixed trace height",
                ))
            }
            None => 1,
        }
    } else {
        checked_trace_height(packed_rows, required_height)?
    };
    let preprocessed = generate_fixed_multi_air_complete_interaction_fixed_preprocessed_trace(
        profile,
        setup,
        Some(height),
    )?;
    let lane_width = FixedMultiAirCompleteInteractionFixedLaneCols::<F>::width();
    let common_width = INTERACTION_FIXED_LANES * lane_width;
    let mut common = try_zero_cells(height, common_width, "interaction fixed common")?;
    for (entry_index, entry) in entries.iter().enumerate() {
        let row = entry_index / INTERACTION_FIXED_LANES;
        let lane = entry_index % INTERACTION_FIXED_LANES;
        let start = row * common_width + lane * lane_width;
        let cols: &mut FixedMultiAirCompleteInteractionFixedLaneCols<F> =
            common[start..start + lane_width].borrow_mut();
        copy_ext(&mut cols.low, entry.low);
        copy_ext(&mut cols.high, entry.high);
        copy_ext(&mut cols.point, entry.point);
        copy_ext(&mut cols.value, entry.value);
    }
    Ok(FixedMultiAirCompleteInteractionFixedTrace {
        preprocessed,
        common: RowMajorMatrix::new(common, common_width),
        values: final_values,
    })
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteInteractionSelectorScheduleCols<T> {
    pub active: T,
    pub has_coordinate: T,
    pub is_first: T,
    pub is_last: T,
    pub coordinate: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteInteractionSelectorCols<T> {
    pub point: [T; D_EF],
    pub first_before: [T; D_EF],
    pub first_after: [T; D_EF],
    pub last_before: [T; D_EF],
    pub last_after: [T; D_EF],
    pub selectors: [[T; D_EF]; 3],
}

pub struct FixedMultiAirCompleteInteractionSelectorAir {
    pub profile: Arc<FixedMultiAirCompleteInteractionEndpointProfile>,
    pub point_bus: FixedMultiAirCompleteInteractionRegionPointBus,
    pub block_eq_point_bus: FixedMultiAirCompleteRegionalPointBus,
    pub selector_bus: FixedMultiAirCompleteInteractionSelectorBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteInteractionSelectorAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteInteractionSelectorAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirCompleteInteractionSelectorScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteInteractionSelectorCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteInteractionSelectorAir {}
impl BaseAir<F> for FixedMultiAirCompleteInteractionSelectorAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteInteractionSelectorScheduleCols::<F>::width()
            + FixedMultiAirCompleteInteractionSelectorCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteInteractionSelectorAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("interaction selector schedule")
            .to_vec();
        let next_cached = builder.cached_mains()[0]
            .row_slice(1)
            .expect("next interaction selector schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("interaction selector row")
            .to_vec();
        let next_common = builder
            .common_main()
            .row_slice(1)
            .expect("next interaction selector row")
            .to_vec();
        let schedule: &FixedMultiAirCompleteInteractionSelectorScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let next_schedule: &FixedMultiAirCompleteInteractionSelectorScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FixedMultiAirCompleteInteractionSelectorCols<AB::Var> =
            common.as_slice().borrow();
        let next: &FixedMultiAirCompleteInteractionSelectorCols<AB::Var> =
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
        let one = ext_one_expr::<AB>();
        assert_array_eq(
            &mut builder.when(schedule.is_first),
            local.first_before,
            one.clone(),
        );
        assert_array_eq(
            &mut builder.when(schedule.is_first),
            local.last_before,
            one.clone(),
        );
        let first_factor = ext_field_subtract::<AB::Expr>(one.clone(), local.point);
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
            &mut builder.when(schedule.active * (AB::Expr::ONE - schedule.has_coordinate)),
            local.first_after,
            local.first_before,
        );
        assert_array_eq(
            &mut builder.when(schedule.active * (AB::Expr::ONE - schedule.has_coordinate)),
            local.last_after,
            local.last_before,
        );
        let mut transition = builder.when_transition();
        let mut continuation = transition.when(next_schedule.active);
        assert_array_eq(&mut continuation, next.first_before, local.first_after);
        assert_array_eq(&mut continuation, next.last_before, local.last_after);
        assert_array_eq(
            &mut builder.when(schedule.is_last),
            local.selectors[0],
            local.first_after,
        );
        assert_array_eq(
            &mut builder.when(schedule.is_last),
            local.selectors[1],
            local.last_after,
        );
        assert_array_eq(
            &mut builder.when(schedule.is_last),
            local.selectors[2],
            ext_field_subtract::<AB::Expr>(one, local.last_after),
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
        // Every interaction inverse block uses this same authenticated point.
        // The block ID is setup-fixed in each block-Eq profile, so the owner
        // publishes one copy per block without a component selector.
        for &block in &self.profile.inverse_weight_blocks {
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
        for selector in 0..3 {
            self.selector_bus.add_key_with_lookups(
                builder,
                FixedMultiAirCompleteInteractionSelectorMessage {
                    region: AB::Expr::from_usize(self.profile.region),
                    selector: AB::Expr::from_usize(selector),
                    value: local.selectors[selector].map(Into::into),
                },
                schedule.is_last
                    * AB::Expr::from_usize(self.profile.selector_lookup_counts[selector]),
            );
        }
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteInteractionSelectorTrace {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub selectors: [EF; 3],
}

pub fn generate_fixed_multi_air_complete_interaction_selector_trace(
    profile: &FixedMultiAirCompleteInteractionEndpointProfile,
    point: &[EF],
    required_height: Option<usize>,
) -> Result<
    FixedMultiAirCompleteInteractionSelectorTrace,
    FixedMultiAirCompleteInteractionEndpointError,
> {
    if point.len() != profile.log_height {
        return Err(FixedMultiAirCompleteInteractionEndpointError::Shape(
            "interaction point",
        ));
    }
    let valid_rows = point.len().max(1);
    let height = checked_trace_height(valid_rows, required_height)?;
    let cached_width = FixedMultiAirCompleteInteractionSelectorScheduleCols::<F>::width();
    let common_width = FixedMultiAirCompleteInteractionSelectorCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    let mut first = EF::ONE;
    let mut last = EF::ONE;
    for row in 0..valid_rows {
        let schedule: &mut FixedMultiAirCompleteInteractionSelectorScheduleCols<F> =
            cached[row * cached_width..(row + 1) * cached_width].borrow_mut();
        schedule.active = F::ONE;
        schedule.has_coordinate = F::from_bool(row < point.len());
        schedule.is_first = F::from_bool(row == 0);
        schedule.is_last = F::from_bool(row + 1 == valid_rows);
        schedule.coordinate = F::from_usize(row);
        let cols: &mut FixedMultiAirCompleteInteractionSelectorCols<F> =
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
        if row + 1 == valid_rows {
            copy_ext(&mut cols.selectors[0], first);
            copy_ext(&mut cols.selectors[1], last);
            copy_ext(&mut cols.selectors[2], EF::ONE - last);
        }
    }
    Ok(FixedMultiAirCompleteInteractionSelectorTrace {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        selectors: [first, last, EF::ONE - last],
    })
}

// The node, denominator, endpoint, and empty-region AIRs follow below. They
// intentionally share only interaction-specific authorities defined above.

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteInteractionNodeScheduleCols<T> {
    pub active: T,
    pub kind_flags: [T; NODE_KIND_COUNT],
    pub source_flags: [T; NODE_SOURCE_KIND_COUNT],
    pub node: T,
    pub arg0_node: T,
    pub arg1_node: T,
    pub source_index: T,
    pub selector: T,
    pub fanout: T,
    pub constant: [T; D_EF],
    pub left_power_flags: [T; MAX_EXACT_DEGREE + 1],
    pub right_power_flags: [T; MAX_EXACT_DEGREE + 1],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteInteractionNodeCols<T> {
    pub source_value: [T; D_EF],
    pub arg0: [T; D_EF],
    pub arg1: [T; D_EF],
    pub value: [T; D_EF],
    pub one: [T; D_EF],
    pub one_powers: [[T; D_EF]; MAX_EXACT_DEGREE + 1],
}

pub struct FixedMultiAirCompleteInteractionNodeAir {
    pub profile: Arc<FixedMultiAirCompleteInteractionEndpointProfile>,
    pub opening_bus: FixedMultiAirCompleteInteractionRegionOpeningBus,
    pub fixed_opening_bus: FixedMultiAirCompleteInteractionFixedOpeningBus,
    pub instance_bus: FixedMultiAirCompleteInstanceValueBus,
    pub selector_bus: FixedMultiAirCompleteInteractionSelectorBus,
    pub node_bus: FixedMultiAirCompleteInteractionNodeBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteInteractionNodeAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteInteractionNodeAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirCompleteInteractionNodeScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteInteractionNodeCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteInteractionNodeAir {}
impl BaseAir<F> for FixedMultiAirCompleteInteractionNodeAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteInteractionNodeScheduleCols::<F>::width()
            + FixedMultiAirCompleteInteractionNodeCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteInteractionNodeAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("interaction node schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("interaction node row")
            .to_vec();
        let schedule: &FixedMultiAirCompleteInteractionNodeScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let local: &FixedMultiAirCompleteInteractionNodeCols<AB::Var> = common.as_slice().borrow();
        for flag in schedule
            .kind_flags
            .into_iter()
            .chain(schedule.source_flags)
            .chain(schedule.left_power_flags)
            .chain(schedule.right_power_flags)
            .chain(core::iter::once(schedule.active))
        {
            builder.assert_bool(flag);
        }
        let kind_sum = schedule
            .kind_flags
            .iter()
            .fold(AB::Expr::ZERO, |sum, &flag| sum + flag);
        let source_sum = schedule
            .source_flags
            .iter()
            .fold(AB::Expr::ZERO, |sum, &flag| sum + flag);
        builder.assert_eq(kind_sum, schedule.active);
        builder.assert_eq(source_sum, schedule.kind_flags[NODE_KIND_SOURCE]);
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

        let one = ext_one_expr::<AB>();
        assert_array_eq(&mut builder.when(schedule.active), local.one_powers[0], one);
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.one_powers[1],
            local.one,
        );
        for power in 1..MAX_EXACT_DEGREE {
            assert_array_eq(
                &mut builder.when(schedule.active),
                local.one_powers[power + 1],
                ext_field_multiply::<AB::Expr>(local.one_powers[power], local.one_powers[1]),
            );
        }
        let left_power = select_power::<AB>(local.one_powers, schedule.left_power_flags);
        let right_power = select_power::<AB>(local.one_powers, schedule.right_power_flags);
        let scaled_left = ext_field_multiply::<AB::Expr>(local.arg0, left_power);
        let scaled_right = ext_field_multiply::<AB::Expr>(local.arg1, right_power);
        let add = ext_field_add::<AB::Expr>(scaled_left.clone(), scaled_right.clone());
        let sub = ext_field_subtract::<AB::Expr>(scaled_left, scaled_right);
        let neg = local.arg0.map(|limb| -AB::Expr::from(limb));
        let mul = ext_field_multiply::<AB::Expr>(local.arg0, local.arg1);
        let expected = core::array::from_fn(|limb| {
            schedule.kind_flags[NODE_KIND_SOURCE] * AB::Expr::from(local.source_value[limb])
                + schedule.kind_flags[NODE_KIND_CONSTANT] * AB::Expr::from(schedule.constant[limb])
                + schedule.kind_flags[NODE_KIND_ADD] * add[limb].clone()
                + schedule.kind_flags[NODE_KIND_SUB] * sub[limb].clone()
                + schedule.kind_flags[NODE_KIND_NEG] * neg[limb].clone()
                + schedule.kind_flags[NODE_KIND_MUL] * mul[limb].clone()
        });
        assert_array_eq(&mut builder.when(schedule.active), local.value, expected);

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
            schedule.source_flags[NODE_SOURCE_DYNAMIC],
        );
        self.fixed_opening_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInteractionFixedOpeningMessage {
                region: AB::Expr::from_usize(self.profile.region),
                opening: schedule.source_index.into(),
                value: local.source_value.map(Into::into),
            },
            schedule.source_flags[NODE_SOURCE_FIXED],
        );
        self.instance_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInstanceValueMessage {
                section: AB::Expr::from_usize(INSTANCE_SECTION_BETA),
                coordinate: schedule.source_index.into(),
                value: local.source_value.map(Into::into),
            },
            schedule.source_flags[NODE_SOURCE_PUBLIC],
        );
        self.selector_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInteractionSelectorMessage {
                region: AB::Expr::from_usize(self.profile.region),
                selector: schedule.selector.into(),
                value: local.source_value.map(Into::into),
            },
            schedule.source_flags[NODE_SOURCE_SELECTOR],
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
        let binary = schedule.kind_flags[NODE_KIND_ADD]
            + schedule.kind_flags[NODE_KIND_SUB]
            + schedule.kind_flags[NODE_KIND_MUL];
        self.node_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInteractionNodeMessage {
                region: AB::Expr::from_usize(self.profile.region),
                node: schedule.arg0_node.into(),
                value: local.arg0.map(Into::into),
            },
            binary.clone() + schedule.kind_flags[NODE_KIND_NEG],
        );
        self.node_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInteractionNodeMessage {
                region: AB::Expr::from_usize(self.profile.region),
                node: schedule.arg1_node.into(),
                value: local.arg1.map(Into::into),
            },
            binary,
        );
        self.node_bus.add_key_with_lookups(
            builder,
            FixedMultiAirCompleteInteractionNodeMessage {
                region: AB::Expr::from_usize(self.profile.region),
                node: schedule.node.into(),
                value: local.value.map(Into::into),
            },
            schedule.fanout,
        );
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteInteractionNodeTrace {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub values: Vec<EF>,
}

pub fn generate_fixed_multi_air_complete_interaction_node_trace(
    profile: &FixedMultiAirCompleteInteractionEndpointProfile,
    beta: &[EF],
    openings: &[EF],
    fixed_openings: &[EF],
    selectors: [EF; 3],
    required_height: Option<usize>,
) -> Result<FixedMultiAirCompleteInteractionNodeTrace, FixedMultiAirCompleteInteractionEndpointError>
{
    let expression = &profile.component.expression;
    if openings.len() != expression.dynamic_columns.len()
        || fixed_openings.len() != expression.fixed_columns.len()
        || profile.one_beta_coordinate >= beta.len()
        || profile
            .public_beta_start
            .checked_add(profile.public_count)
            .is_none_or(|end| end > beta.len())
    {
        return Err(FixedMultiAirCompleteInteractionEndpointError::Shape(
            "interaction node inputs",
        ));
    }
    let valid_rows = expression.nodes.len();
    let height = checked_trace_height(valid_rows, required_height)?;
    let cached_width = FixedMultiAirCompleteInteractionNodeScheduleCols::<F>::width();
    let common_width = FixedMultiAirCompleteInteractionNodeCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    let one = beta[profile.one_beta_coordinate];
    let one_powers = powers(one);
    let mut values = Vec::with_capacity(valid_rows);
    for (node_index, node) in expression.nodes.iter().enumerate() {
        let schedule: &mut FixedMultiAirCompleteInteractionNodeScheduleCols<F> =
            cached[node_index * cached_width..(node_index + 1) * cached_width].borrow_mut();
        schedule.active = F::ONE;
        schedule.node = F::from_usize(node_index);
        schedule.fanout = F::from_usize(profile.node_fanout[node_index]);
        let cols: &mut FixedMultiAirCompleteInteractionNodeCols<F> =
            common[node_index * common_width..(node_index + 1) * common_width].borrow_mut();
        copy_ext(&mut cols.one, one);
        for (target, &power) in cols.one_powers.iter_mut().zip(&one_powers) {
            copy_ext(target, power);
        }
        let mut left_delta = 0usize;
        let mut right_delta = 0usize;
        let value = match node {
            SymbolicExpressionNode::Variable(variable) => {
                schedule.kind_flags[NODE_KIND_SOURCE] = F::ONE;
                match variable.entry {
                    Entry::Main { .. } | Entry::Preprocessed { .. } => {
                        match expression.node_sources[node_index].ok_or(
                            FixedMultiAirCompleteInteractionEndpointError::Configuration(
                                "interaction variable source",
                            ),
                        )? {
                            FixedMultiAirCompleteTerminalCircuitNodeSource::Dynamic(source) => {
                                let source = source as usize;
                                schedule.source_flags[NODE_SOURCE_DYNAMIC] = F::ONE;
                                schedule.source_index = F::from_usize(source);
                                *openings.get(source).ok_or(
                                    FixedMultiAirCompleteInteractionEndpointError::Shape(
                                        "dynamic source",
                                    ),
                                )?
                            }
                            FixedMultiAirCompleteTerminalCircuitNodeSource::Fixed(source) => {
                                let source = source as usize;
                                schedule.source_flags[NODE_SOURCE_FIXED] = F::ONE;
                                schedule.source_index = F::from_usize(source);
                                *fixed_openings.get(source).ok_or(
                                    FixedMultiAirCompleteInteractionEndpointError::Shape(
                                        "fixed source",
                                    ),
                                )?
                            }
                        }
                    }
                    Entry::Public => {
                        let coordinate = profile
                            .public_beta_start
                            .checked_add(variable.index)
                            .ok_or(FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                                "public coordinate",
                            ))?;
                        schedule.source_flags[NODE_SOURCE_PUBLIC] = F::ONE;
                        schedule.source_index = F::from_usize(coordinate);
                        *beta.get(coordinate).ok_or(
                            FixedMultiAirCompleteInteractionEndpointError::Shape(
                                "public coordinate",
                            ),
                        )?
                    }
                    Entry::Challenge => {
                        return Err(
                            FixedMultiAirCompleteInteractionEndpointError::Configuration(
                                "challenge variable",
                            ),
                        )
                    }
                }
            }
            SymbolicExpressionNode::IsFirstRow => {
                schedule.kind_flags[NODE_KIND_SOURCE] = F::ONE;
                schedule.source_flags[NODE_SOURCE_SELECTOR] = F::ONE;
                schedule.selector = F::ZERO;
                selectors[0]
            }
            SymbolicExpressionNode::IsLastRow => {
                schedule.kind_flags[NODE_KIND_SOURCE] = F::ONE;
                schedule.source_flags[NODE_SOURCE_SELECTOR] = F::ONE;
                schedule.selector = F::ONE;
                selectors[1]
            }
            SymbolicExpressionNode::IsTransition => {
                schedule.kind_flags[NODE_KIND_SOURCE] = F::ONE;
                schedule.source_flags[NODE_SOURCE_SELECTOR] = F::ONE;
                schedule.selector = F::TWO;
                selectors[2]
            }
            SymbolicExpressionNode::Constant(value) => {
                schedule.kind_flags[NODE_KIND_CONSTANT] = F::ONE;
                let value = EF::from(*value);
                copy_ext(&mut schedule.constant, value);
                value
            }
            SymbolicExpressionNode::Add {
                left_idx,
                right_idx,
                ..
            }
            | SymbolicExpressionNode::Sub {
                left_idx,
                right_idx,
                ..
            } => {
                let is_add = matches!(node, SymbolicExpressionNode::Add { .. });
                schedule.kind_flags[if is_add { NODE_KIND_ADD } else { NODE_KIND_SUB }] = F::ONE;
                schedule.arg0_node = F::from_usize(*left_idx);
                schedule.arg1_node = F::from_usize(*right_idx);
                let left = *values.get(*left_idx).ok_or(
                    FixedMultiAirCompleteInteractionEndpointError::Configuration("left node"),
                )?;
                let right = *values.get(*right_idx).ok_or(
                    FixedMultiAirCompleteInteractionEndpointError::Configuration("right node"),
                )?;
                copy_ext(&mut cols.arg0, left);
                copy_ext(&mut cols.arg1, right);
                let degree = expression.homogeneous_degrees[node_index] as usize;
                left_delta = degree
                    .checked_sub(expression.homogeneous_degrees[*left_idx] as usize)
                    .ok_or(
                        FixedMultiAirCompleteInteractionEndpointError::Configuration("left degree"),
                    )?;
                right_delta = degree
                    .checked_sub(expression.homogeneous_degrees[*right_idx] as usize)
                    .ok_or(
                        FixedMultiAirCompleteInteractionEndpointError::Configuration(
                            "right degree",
                        ),
                    )?;
                if is_add {
                    left * one_powers[left_delta] + right * one_powers[right_delta]
                } else {
                    left * one_powers[left_delta] - right * one_powers[right_delta]
                }
            }
            SymbolicExpressionNode::Neg { idx, .. } => {
                schedule.kind_flags[NODE_KIND_NEG] = F::ONE;
                schedule.arg0_node = F::from_usize(*idx);
                let value = *values.get(*idx).ok_or(
                    FixedMultiAirCompleteInteractionEndpointError::Configuration("neg node"),
                )?;
                copy_ext(&mut cols.arg0, value);
                -value
            }
            SymbolicExpressionNode::Mul {
                left_idx,
                right_idx,
                ..
            } => {
                schedule.kind_flags[NODE_KIND_MUL] = F::ONE;
                schedule.arg0_node = F::from_usize(*left_idx);
                schedule.arg1_node = F::from_usize(*right_idx);
                let left = *values.get(*left_idx).ok_or(
                    FixedMultiAirCompleteInteractionEndpointError::Configuration("left node"),
                )?;
                let right = *values.get(*right_idx).ok_or(
                    FixedMultiAirCompleteInteractionEndpointError::Configuration("right node"),
                )?;
                copy_ext(&mut cols.arg0, left);
                copy_ext(&mut cols.arg1, right);
                left * right
            }
        };
        if left_delta > MAX_EXACT_DEGREE || right_delta > MAX_EXACT_DEGREE {
            return Err(
                FixedMultiAirCompleteInteractionEndpointError::Configuration("node degree delta"),
            );
        }
        schedule.left_power_flags[left_delta] = F::ONE;
        schedule.right_power_flags[right_delta] = F::ONE;
        if schedule.kind_flags[NODE_KIND_SOURCE] == F::ONE {
            copy_ext(&mut cols.source_value, value);
        }
        copy_ext(&mut cols.value, value);
        values.push(value);
    }
    Ok(FixedMultiAirCompleteInteractionNodeTrace {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        values,
    })
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteInteractionDenominatorScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub is_trace_last: T,
    pub kind_flags: [T; DENOMINATOR_KIND_COUNT],
    pub interaction: T,
    pub node: T,
    pub beta_coordinate: T,
    pub scalar: [T; D_EF],
    pub power_flags: [T; MAX_EXACT_DEGREE + 1],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteInteractionDenominatorCols<T> {
    pub one: [T; D_EF],
    pub one_powers: [[T; D_EF]; MAX_EXACT_DEGREE + 1],
    pub beta_value: [T; D_EF],
    pub node_value: [T; D_EF],
    pub term: [T; D_EF],
    pub running_before: [T; D_EF],
    pub running_after: [T; D_EF],
}

pub struct FixedMultiAirCompleteInteractionDenominatorAir {
    pub profile: Arc<FixedMultiAirCompleteInteractionEndpointProfile>,
    pub instance_bus: FixedMultiAirCompleteInstanceValueBus,
    pub node_bus: FixedMultiAirCompleteInteractionNodeBus,
    pub denominator_bus: FixedMultiAirCompleteInteractionDenominatorBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteInteractionDenominatorAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteInteractionDenominatorAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirCompleteInteractionDenominatorScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteInteractionDenominatorCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteInteractionDenominatorAir {}
impl BaseAir<F> for FixedMultiAirCompleteInteractionDenominatorAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteInteractionDenominatorScheduleCols::<F>::width()
            + FixedMultiAirCompleteInteractionDenominatorCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteInteractionDenominatorAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("interaction denominator schedule")
            .to_vec();
        let next_cached = builder.cached_mains()[0]
            .row_slice(1)
            .expect("next interaction denominator schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("interaction denominator row")
            .to_vec();
        let next_common = builder
            .common_main()
            .row_slice(1)
            .expect("next interaction denominator row")
            .to_vec();
        let schedule: &FixedMultiAirCompleteInteractionDenominatorScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let next_schedule: &FixedMultiAirCompleteInteractionDenominatorScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FixedMultiAirCompleteInteractionDenominatorCols<AB::Var> =
            common.as_slice().borrow();
        let next: &FixedMultiAirCompleteInteractionDenominatorCols<AB::Var> =
            next_common.as_slice().borrow();

        for flag in schedule
            .kind_flags
            .into_iter()
            .chain(schedule.power_flags)
            .chain([
                schedule.active,
                schedule.is_first,
                schedule.is_last,
                schedule.is_trace_last,
            ])
        {
            builder.assert_bool(flag);
        }
        builder.assert_eq(
            schedule
                .kind_flags
                .iter()
                .fold(AB::Expr::ZERO, |sum, &flag| sum + flag),
            schedule.active,
        );
        builder.assert_eq(
            schedule
                .power_flags
                .iter()
                .fold(AB::Expr::ZERO, |sum, &flag| sum + flag),
            schedule.active,
        );
        builder.when_first_row().assert_one(schedule.active);
        builder.when_first_row().assert_one(schedule.is_first);
        builder
            .when_transition()
            .assert_bool(schedule.active - next_schedule.active);
        builder.when_transition().assert_eq(
            schedule.active - next_schedule.active,
            schedule.is_trace_last,
        );
        builder
            .when_last_row()
            .assert_eq(schedule.is_trace_last, schedule.active);

        assert_array_eq(
            &mut builder.when(schedule.active),
            local.one_powers[0],
            ext_one_expr::<AB>(),
        );
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.one_powers[1],
            local.one,
        );
        for power in 1..MAX_EXACT_DEGREE {
            assert_array_eq(
                &mut builder.when(schedule.active),
                local.one_powers[power + 1],
                ext_field_multiply::<AB::Expr>(local.one_powers[power], local.one),
            );
        }
        let selected_power = select_power::<AB>(local.one_powers, schedule.power_flags);
        let fixed_one = ext_one_expr::<AB>();
        let factor = core::array::from_fn(|limb| {
            schedule.kind_flags[DENOMINATOR_ALPHA] * fixed_one[limb].clone()
                + schedule.kind_flags[DENOMINATOR_BUS] * AB::Expr::from(schedule.scalar[limb])
                + schedule.kind_flags[DENOMINATOR_MESSAGE] * AB::Expr::from(local.node_value[limb])
        });
        let expected_term = ext_field_multiply::<AB::Expr>(
            ext_field_multiply::<AB::Expr>(local.beta_value, factor),
            selected_power,
        );
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.term,
            expected_term,
        );
        assert_array_eq(
            &mut builder.when(schedule.is_first),
            local.running_before,
            core::array::from_fn(|_| AB::Expr::ZERO),
        );
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.running_after,
            ext_field_add::<AB::Expr>(local.running_before, local.term),
        );
        let mut transition = builder.when_transition();
        let mut continuation = transition
            .when(schedule.active * (AB::Expr::ONE - schedule.is_last) * next_schedule.active);
        assert_array_eq(&mut continuation, next.running_before, local.running_after);

        self.instance_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInstanceValueMessage {
                section: AB::Expr::from_usize(INSTANCE_SECTION_BETA),
                coordinate: AB::Expr::from_usize(self.profile.one_beta_coordinate),
                value: local.one.map(Into::into),
            },
            schedule.active,
        );
        self.instance_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInstanceValueMessage {
                section: AB::Expr::from_usize(INSTANCE_SECTION_BETA),
                coordinate: schedule.beta_coordinate.into(),
                value: local.beta_value.map(Into::into),
            },
            schedule.active,
        );
        self.node_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInteractionNodeMessage {
                region: AB::Expr::from_usize(self.profile.region),
                node: schedule.node.into(),
                value: local.node_value.map(Into::into),
            },
            schedule.kind_flags[DENOMINATOR_MESSAGE],
        );
        self.denominator_bus.send(
            builder,
            FixedMultiAirCompleteInteractionDenominatorMessage {
                region: AB::Expr::from_usize(self.profile.region),
                interaction: schedule.interaction.into(),
                value: local.running_after.map(Into::into),
            },
            schedule.is_last,
        );
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteInteractionDenominatorTrace {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub denominators: Vec<EF>,
}

pub fn generate_fixed_multi_air_complete_interaction_denominator_trace(
    profile: &FixedMultiAirCompleteInteractionEndpointProfile,
    beta: &[EF],
    nodes: &[EF],
    required_height: Option<usize>,
) -> Result<
    FixedMultiAirCompleteInteractionDenominatorTrace,
    FixedMultiAirCompleteInteractionEndpointError,
> {
    if profile.is_empty() {
        return Err(FixedMultiAirCompleteInteractionEndpointError::EmptyRegion);
    }
    if profile.one_beta_coordinate >= beta.len()
        || profile.alpha_beta_coordinate >= beta.len()
        || profile
            .beta_power_beta_start
            .checked_add(profile.beta_power_count)
            .is_none_or(|end| end > beta.len())
        || nodes.len() != profile.component.expression.nodes.len()
    {
        return Err(FixedMultiAirCompleteInteractionEndpointError::Shape(
            "interaction denominator inputs",
        ));
    }
    let rows = profile
        .component
        .interactions
        .iter()
        .try_fold(0usize, |rows, interaction| {
            rows.checked_add(2usize.checked_add(interaction.message_roots.len())?)
        })
        .ok_or(FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
            "interaction denominator rows",
        ))?;
    let height = checked_trace_height(rows, required_height)?;
    let cached_width = FixedMultiAirCompleteInteractionDenominatorScheduleCols::<F>::width();
    let common_width = FixedMultiAirCompleteInteractionDenominatorCols::<F>::width();
    let cells = height.checked_mul(cached_width).ok_or(
        FixedMultiAirCompleteInteractionEndpointError::Arithmetic("denominator cached cells"),
    )?;
    let common_cells = height.checked_mul(common_width).ok_or(
        FixedMultiAirCompleteInteractionEndpointError::Arithmetic("denominator common cells"),
    )?;
    let mut cached = F::zero_vec(cells);
    let mut common = F::zero_vec(common_cells);
    let one = beta[profile.one_beta_coordinate];
    let one_powers = powers(one);
    let mut denominators = Vec::with_capacity(profile.interaction_count());
    let mut row = 0usize;
    for (interaction_ordinal, interaction) in profile.component.interactions.iter().enumerate() {
        let denominator_degree = usize::from(interaction.denominator_degree);
        let mut running = EF::ZERO;
        let terms = 2usize.checked_add(interaction.message_roots.len()).ok_or(
            FixedMultiAirCompleteInteractionEndpointError::Arithmetic("denominator term count"),
        )?;
        for term_ordinal in 0..terms {
            let schedule: &mut FixedMultiAirCompleteInteractionDenominatorScheduleCols<F> =
                cached[row * cached_width..(row + 1) * cached_width].borrow_mut();
            schedule.active = F::ONE;
            schedule.is_first = F::from_bool(term_ordinal == 0);
            schedule.is_last = F::from_bool(term_ordinal + 1 == terms);
            schedule.is_trace_last = F::from_bool(row + 1 == rows);
            schedule.interaction = F::from_usize(interaction_ordinal);
            let cols: &mut FixedMultiAirCompleteInteractionDenominatorCols<F> =
                common[row * common_width..(row + 1) * common_width].borrow_mut();
            copy_ext(&mut cols.one, one);
            for (target, &power) in cols.one_powers.iter_mut().zip(&one_powers) {
                copy_ext(target, power);
            }
            copy_ext(&mut cols.running_before, running);
            let (kind, beta_coordinate, factor, power) = if term_ordinal == 0 {
                (
                    DENOMINATOR_ALPHA,
                    profile.alpha_beta_coordinate,
                    EF::ONE,
                    denominator_degree.checked_sub(1).ok_or(
                        FixedMultiAirCompleteInteractionEndpointError::Configuration(
                            "alpha denominator degree",
                        ),
                    )?,
                )
            } else if term_ordinal == 1 {
                let beta_coordinate = profile
                    .beta_power_beta_start
                    .checked_add(interaction.bus_beta_power_ordinal as usize)
                    .ok_or(FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                        "bus beta coordinate",
                    ))?;
                (
                    DENOMINATOR_BUS,
                    beta_coordinate,
                    EF::from(F::from_u32(interaction.bus_index_plus_one)),
                    denominator_degree.checked_sub(1).ok_or(
                        FixedMultiAirCompleteInteractionEndpointError::Configuration(
                            "bus denominator degree",
                        ),
                    )?,
                )
            } else {
                let message = term_ordinal - 2;
                let root = interaction.message_roots[message] as usize;
                let root_degree = usize::from(
                    *profile
                        .component
                        .expression
                        .homogeneous_degrees
                        .get(root)
                        .ok_or(
                            FixedMultiAirCompleteInteractionEndpointError::Configuration(
                                "message root degree",
                            ),
                        )?,
                );
                schedule.node = F::from_usize(root);
                let node = *nodes.get(root).ok_or(
                    FixedMultiAirCompleteInteractionEndpointError::Shape("message root value"),
                )?;
                copy_ext(&mut cols.node_value, node);
                (
                    DENOMINATOR_MESSAGE,
                    profile.beta_power_beta_start.checked_add(message).ok_or(
                        FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                            "message beta coordinate",
                        ),
                    )?,
                    node,
                    denominator_degree
                        .checked_sub(root_degree.checked_add(1).ok_or(
                            FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                                "message term degree",
                            ),
                        )?)
                        .ok_or(
                            FixedMultiAirCompleteInteractionEndpointError::Configuration(
                                "message denominator degree",
                            ),
                        )?,
                )
            };
            if power > MAX_EXACT_DEGREE {
                return Err(
                    FixedMultiAirCompleteInteractionEndpointError::Configuration(
                        "denominator one power",
                    ),
                );
            }
            schedule.kind_flags[kind] = F::ONE;
            schedule.beta_coordinate = F::from_usize(beta_coordinate);
            schedule.power_flags[power] = F::ONE;
            if kind == DENOMINATOR_BUS {
                copy_ext(&mut schedule.scalar, factor);
            }
            let beta_value = *beta.get(beta_coordinate).ok_or(
                FixedMultiAirCompleteInteractionEndpointError::Shape("denominator beta value"),
            )?;
            let term = beta_value * factor * one_powers[power];
            running += term;
            copy_ext(&mut cols.beta_value, beta_value);
            copy_ext(&mut cols.term, term);
            copy_ext(&mut cols.running_after, running);
            row += 1;
        }
        denominators.push(running);
    }
    if row != rows {
        return Err(FixedMultiAirCompleteInteractionEndpointError::Witness(
            "denominator row coverage",
        ));
    }
    Ok(FixedMultiAirCompleteInteractionDenominatorTrace {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        denominators,
    })
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteInteractionEndpointScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub interaction: T,
    pub count_node: T,
    pub q_regional_openings: [T; D_EF],
    pub q_global_openings: [T; D_EF],
    pub inverse_weight_block: T,
    pub inverse_degree_flags: [T; MAX_EXACT_DEGREE + 1],
    pub inverse_tail_power_flags: [T; MAX_EXACT_DEGREE + 1],
    pub global_tail_power_flags: [T; MAX_EXACT_DEGREE + 1],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteInteractionEndpointCols<T> {
    pub one: [T; D_EF],
    pub one_powers: [[T; D_EF]; MAX_EXACT_DEGREE + 1],
    pub denominator: [T; D_EF],
    pub q_coordinates: [[T; D_EF]; D_EF],
    pub q: [T; D_EF],
    pub count: [T; D_EF],
    pub inverse_weight: [T; D_EF],
    pub global_weight: [T; D_EF],
    pub inverse_residual: [T; D_EF],
    pub global_term: [T; D_EF],
    pub contribution: [T; D_EF],
    pub running_before: [T; D_EF],
    pub running_after: [T; D_EF],
    pub final_evaluation: [T; D_EF],
}

pub struct FixedMultiAirCompleteInteractionEndpointAir {
    pub profile: Arc<FixedMultiAirCompleteInteractionEndpointProfile>,
    pub opening_bus: FixedMultiAirCompleteInteractionRegionOpeningBus,
    pub final_evaluation_bus: FixedMultiAirCompleteInteractionRegionFinalEvaluationBus,
    pub instance_bus: FixedMultiAirCompleteInstanceValueBus,
    pub node_bus: FixedMultiAirCompleteInteractionNodeBus,
    pub denominator_bus: FixedMultiAirCompleteInteractionDenominatorBus,
    pub eq_weight_bus: FixedMultiAirCompleteEqWeightBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteInteractionEndpointAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteInteractionEndpointAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirCompleteInteractionEndpointScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteInteractionEndpointCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteInteractionEndpointAir {}
impl BaseAir<F> for FixedMultiAirCompleteInteractionEndpointAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteInteractionEndpointScheduleCols::<F>::width()
            + FixedMultiAirCompleteInteractionEndpointCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteInteractionEndpointAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("interaction endpoint schedule")
            .to_vec();
        let next_cached = builder.cached_mains()[0]
            .row_slice(1)
            .expect("next interaction endpoint schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("interaction endpoint row")
            .to_vec();
        let next_common = builder
            .common_main()
            .row_slice(1)
            .expect("next interaction endpoint row")
            .to_vec();
        let schedule: &FixedMultiAirCompleteInteractionEndpointScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let next_schedule: &FixedMultiAirCompleteInteractionEndpointScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FixedMultiAirCompleteInteractionEndpointCols<AB::Var> =
            common.as_slice().borrow();
        let next: &FixedMultiAirCompleteInteractionEndpointCols<AB::Var> =
            next_common.as_slice().borrow();

        for flag in schedule
            .inverse_degree_flags
            .into_iter()
            .chain(schedule.inverse_tail_power_flags)
            .chain(schedule.global_tail_power_flags)
            .chain([schedule.active, schedule.is_first, schedule.is_last])
        {
            builder.assert_bool(flag);
        }
        builder.assert_eq(
            schedule
                .inverse_degree_flags
                .iter()
                .fold(AB::Expr::ZERO, |sum, &flag| sum + flag),
            schedule.active,
        );
        builder.assert_eq(
            schedule
                .inverse_tail_power_flags
                .iter()
                .fold(AB::Expr::ZERO, |sum, &flag| sum + flag),
            schedule.active,
        );
        builder.assert_eq(
            schedule
                .global_tail_power_flags
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
            &mut builder.when(schedule.active),
            local.one_powers[0],
            ext_one_expr::<AB>(),
        );
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.one_powers[1],
            local.one,
        );
        for power in 1..MAX_EXACT_DEGREE {
            assert_array_eq(
                &mut builder.when(schedule.active),
                local.one_powers[power + 1],
                ext_field_multiply::<AB::Expr>(local.one_powers[power], local.one),
            );
        }

        let mut reconstructed_q = core::array::from_fn(|_| AB::Expr::ZERO);
        for coordinate in 0..D_EF {
            let basis = core::array::from_fn(|limb| {
                if limb == coordinate {
                    AB::Expr::ONE
                } else {
                    AB::Expr::ZERO
                }
            });
            reconstructed_q = ext_field_add::<AB::Expr>(
                reconstructed_q,
                ext_field_multiply::<AB::Expr>(local.q_coordinates[coordinate], basis),
            );
        }
        assert_array_eq(&mut builder.when(schedule.active), local.q, reconstructed_q);
        let denominator_times_q = ext_field_multiply::<AB::Expr>(local.denominator, local.q);
        let inverse_target = select_power::<AB>(local.one_powers, schedule.inverse_degree_flags);
        let inverse_tail = select_power::<AB>(local.one_powers, schedule.inverse_tail_power_flags);
        let global_tail = select_power::<AB>(local.one_powers, schedule.global_tail_power_flags);
        let expected_inverse = ext_field_multiply::<AB::Expr>(
            ext_field_subtract::<AB::Expr>(denominator_times_q, inverse_target),
            inverse_tail,
        );
        let expected_global = ext_field_multiply::<AB::Expr>(
            ext_field_multiply::<AB::Expr>(local.count, local.q),
            global_tail,
        );
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.inverse_residual,
            expected_inverse,
        );
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.global_term,
            expected_global,
        );
        let expected_contribution = ext_field_add::<AB::Expr>(
            ext_field_multiply::<AB::Expr>(local.inverse_weight, local.inverse_residual),
            ext_field_multiply::<AB::Expr>(local.global_weight, local.global_term),
        );
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.contribution,
            expected_contribution,
        );
        assert_array_eq(
            &mut builder.when(schedule.is_first),
            local.running_before,
            core::array::from_fn(|_| AB::Expr::ZERO),
        );
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.running_after,
            ext_field_add::<AB::Expr>(local.running_before, local.contribution),
        );
        let mut transition = builder.when_transition();
        let mut continuation = transition
            .when(schedule.active * (AB::Expr::ONE - schedule.is_last) * next_schedule.active);
        assert_array_eq(&mut continuation, next.running_before, local.running_after);
        assert_array_eq(&mut continuation, next.global_weight, local.global_weight);
        assert_array_eq(
            &mut builder.when(schedule.is_last),
            local.final_evaluation,
            local.running_after,
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
        self.denominator_bus.receive(
            builder,
            FixedMultiAirCompleteInteractionDenominatorMessage {
                region: AB::Expr::from_usize(self.profile.region),
                interaction: schedule.interaction.into(),
                value: local.denominator.map(Into::into),
            },
            schedule.active,
        );
        for coordinate in 0..D_EF {
            self.opening_bus.lookup_key(
                builder,
                FixedMultiAirCompleteRegionOpeningMessage {
                    region: AB::Expr::from_usize(self.profile.region),
                    regional_opening: schedule.q_regional_openings[coordinate].into(),
                    global_opening: schedule.q_global_openings[coordinate].into(),
                    value: local.q_coordinates[coordinate].map(Into::into),
                },
                schedule.active,
            );
        }
        self.node_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInteractionNodeMessage {
                region: AB::Expr::from_usize(self.profile.region),
                node: schedule.count_node.into(),
                value: local.count.map(Into::into),
            },
            schedule.active,
        );
        self.eq_weight_bus.receive(
            builder,
            FixedMultiAirCompleteEqWeightMessage {
                block: schedule.inverse_weight_block.into(),
                role: AB::Expr::from_usize(2),
                role_ordinal: schedule.interaction.into(),
                value: local.inverse_weight.map(Into::into),
            },
            schedule.active,
        );
        self.eq_weight_bus.receive(
            builder,
            FixedMultiAirCompleteEqWeightMessage {
                block: AB::Expr::from_usize(self.profile.global_weight_block),
                role: AB::Expr::from_usize(3),
                role_ordinal: AB::Expr::ZERO,
                value: local.global_weight.map(Into::into),
            },
            schedule.is_first,
        );
        self.final_evaluation_bus.receive(
            builder,
            FixedMultiAirCompleteRegionFinalEvaluationMessage {
                region: AB::Expr::from_usize(self.profile.region),
                claim: local.final_evaluation.map(Into::into),
            },
            schedule.is_last,
        );
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteInteractionEndpointTrace {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub evaluation: EF,
}

#[allow(clippy::too_many_arguments)]
pub fn generate_fixed_multi_air_complete_interaction_endpoint_trace(
    profile: &FixedMultiAirCompleteInteractionEndpointProfile,
    beta: &[EF],
    openings: &[EF],
    nodes: &[EF],
    denominators: &[EF],
    inverse_weights: &[EF],
    global_weight: EF,
    final_evaluation: EF,
    required_height: Option<usize>,
) -> Result<
    FixedMultiAirCompleteInteractionEndpointTrace,
    FixedMultiAirCompleteInteractionEndpointError,
> {
    if profile.is_empty() {
        return Err(FixedMultiAirCompleteInteractionEndpointError::EmptyRegion);
    }
    let interaction_count = profile.interaction_count();
    if profile.one_beta_coordinate >= beta.len()
        || openings.len() != profile.component.expression.dynamic_columns.len()
        || nodes.len() != profile.component.expression.nodes.len()
        || denominators.len() != interaction_count
        || inverse_weights.len() != interaction_count
    {
        return Err(FixedMultiAirCompleteInteractionEndpointError::Shape(
            "interaction endpoint inputs",
        ));
    }
    let height = checked_trace_height(interaction_count, required_height)?;
    let cached_width = FixedMultiAirCompleteInteractionEndpointScheduleCols::<F>::width();
    let common_width = FixedMultiAirCompleteInteractionEndpointCols::<F>::width();
    let cached_cells = height.checked_mul(cached_width).ok_or(
        FixedMultiAirCompleteInteractionEndpointError::Arithmetic("endpoint cached cells"),
    )?;
    let common_cells = height.checked_mul(common_width).ok_or(
        FixedMultiAirCompleteInteractionEndpointError::Arithmetic("endpoint common cells"),
    )?;
    let mut cached = F::zero_vec(cached_cells);
    let mut common = F::zero_vec(common_cells);
    let one = beta[profile.one_beta_coordinate];
    let one_powers = powers(one);
    let opening_start =
        usize::try_from(profile.component.identity.opening_ordinal_start).map_err(|_| {
            FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                "interaction global opening start",
            )
        })?;
    let mut running = EF::ZERO;
    for (ordinal, interaction) in profile.component.interactions.iter().enumerate() {
        let inverse_degree = usize::from(interaction.inverse_degree);
        let global_degree = usize::from(interaction.global_term_degree);
        let inverse_tail = profile
            .exact_relation_degree
            .checked_sub(inverse_degree)
            .ok_or(
                FixedMultiAirCompleteInteractionEndpointError::Configuration(
                    "inverse endpoint degree",
                ),
            )?;
        let global_tail = profile
            .exact_relation_degree
            .checked_sub(global_degree)
            .ok_or(
                FixedMultiAirCompleteInteractionEndpointError::Configuration(
                    "global endpoint degree",
                ),
            )?;
        if inverse_degree > MAX_EXACT_DEGREE
            || inverse_tail > MAX_EXACT_DEGREE
            || global_tail > MAX_EXACT_DEGREE
        {
            return Err(
                FixedMultiAirCompleteInteractionEndpointError::Configuration("endpoint power"),
            );
        }
        let schedule: &mut FixedMultiAirCompleteInteractionEndpointScheduleCols<F> =
            cached[ordinal * cached_width..(ordinal + 1) * cached_width].borrow_mut();
        schedule.active = F::ONE;
        schedule.is_first = F::from_bool(ordinal == 0);
        schedule.is_last = F::from_bool(ordinal + 1 == interaction_count);
        schedule.interaction = F::from_usize(ordinal);
        schedule.count_node = F::from_u32(interaction.count_root);
        schedule.inverse_weight_block = F::from_usize(profile.inverse_weight_blocks[ordinal]);
        schedule.inverse_degree_flags[inverse_degree] = F::ONE;
        schedule.inverse_tail_power_flags[inverse_tail] = F::ONE;
        schedule.global_tail_power_flags[global_tail] = F::ONE;

        let cols: &mut FixedMultiAirCompleteInteractionEndpointCols<F> =
            common[ordinal * common_width..(ordinal + 1) * common_width].borrow_mut();
        copy_ext(&mut cols.one, one);
        for (target, &power) in cols.one_powers.iter_mut().zip(&one_powers) {
            copy_ext(target, power);
        }
        let denominator = denominators[ordinal];
        copy_ext(&mut cols.denominator, denominator);
        let mut q = EF::ZERO;
        for coordinate in 0..D_EF {
            let source =
                usize::try_from(interaction.q_source_ordinals[coordinate]).map_err(|_| {
                    FixedMultiAirCompleteInteractionEndpointError::Arithmetic("q source ordinal")
                })?;
            let global = opening_start.checked_add(source).ok_or(
                FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
                    "q global opening ordinal",
                ),
            )?;
            let q_coordinate = *openings.get(source).ok_or(
                FixedMultiAirCompleteInteractionEndpointError::Shape("q source opening"),
            )?;
            schedule.q_regional_openings[coordinate] = F::from_usize(source);
            schedule.q_global_openings[coordinate] = F::from_usize(global);
            copy_ext(&mut cols.q_coordinates[coordinate], q_coordinate);
            q += basis_element(coordinate) * q_coordinate;
        }
        copy_ext(&mut cols.q, q);
        let count = *nodes.get(interaction.count_root as usize).ok_or(
            FixedMultiAirCompleteInteractionEndpointError::Shape("interaction count node"),
        )?;
        copy_ext(&mut cols.count, count);
        copy_ext(&mut cols.inverse_weight, inverse_weights[ordinal]);
        copy_ext(&mut cols.global_weight, global_weight);
        let inverse_residual =
            (denominator * q - one_powers[inverse_degree]) * one_powers[inverse_tail];
        let global_term = count * q * one_powers[global_tail];
        let contribution =
            inverse_weights[ordinal] * inverse_residual + global_weight * global_term;
        copy_ext(&mut cols.inverse_residual, inverse_residual);
        copy_ext(&mut cols.global_term, global_term);
        copy_ext(&mut cols.contribution, contribution);
        copy_ext(&mut cols.running_before, running);
        running += contribution;
        copy_ext(&mut cols.running_after, running);
        if ordinal + 1 == interaction_count {
            copy_ext(&mut cols.final_evaluation, final_evaluation);
        }
    }
    if running != final_evaluation {
        return Err(FixedMultiAirCompleteInteractionEndpointError::Witness(
            "interaction final evaluation",
        ));
    }
    Ok(FixedMultiAirCompleteInteractionEndpointTrace {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        evaluation: running,
    })
}

/// Setup-fixed zero endpoint for interaction-free regions. There is no
/// proof-presence witness or selector: this AIR can only be instantiated from
/// a profile whose canonical interaction list is empty.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteEmptyInteractionCols<T> {
    pub claim: [T; D_EF],
}

pub struct FixedMultiAirCompleteEmptyInteractionAir {
    pub profile: Arc<FixedMultiAirCompleteInteractionEndpointProfile>,
    pub interaction_claim_bus: FixedMultiAirCompleteInteractionClaimBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteEmptyInteractionAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteEmptyInteractionAir {
    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteEmptyInteractionCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteEmptyInteractionAir {}
impl BaseAir<F> for FixedMultiAirCompleteEmptyInteractionAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteEmptyInteractionCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteEmptyInteractionAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("empty interaction endpoint")
            .to_vec();
        let local: &FixedMultiAirCompleteEmptyInteractionCols<AB::Var> = common.as_slice().borrow();
        for limb in local.claim {
            builder.assert_zero(limb);
        }
        self.interaction_claim_bus.lookup_key(
            builder,
            FixedMultiAirCompleteRegionalClaimMessage {
                region: AB::Expr::from_usize(self.profile.region),
                claim: local.claim.map(Into::into),
            },
            AB::Expr::ONE,
        );
    }
}

pub fn generate_fixed_multi_air_complete_empty_interaction_trace(
    profile: &FixedMultiAirCompleteInteractionEndpointProfile,
    claim: EF,
    required_height: Option<usize>,
) -> Result<RowMajorMatrix<F>, FixedMultiAirCompleteInteractionEndpointError> {
    if !profile.is_empty() {
        return Err(FixedMultiAirCompleteInteractionEndpointError::NonEmptyRegion);
    }
    if claim != EF::ZERO {
        return Err(FixedMultiAirCompleteInteractionEndpointError::Witness(
            "empty interaction claim",
        ));
    }
    let height = checked_trace_height(1, required_height)?;
    let width = FixedMultiAirCompleteEmptyInteractionCols::<F>::width();
    let cells = height.checked_mul(width).ok_or(
        FixedMultiAirCompleteInteractionEndpointError::Arithmetic("empty interaction cells"),
    )?;
    Ok(RowMajorMatrix::new(F::zero_vec(cells), width))
}

fn checked_trace_height(
    rows: usize,
    required: Option<usize>,
) -> Result<usize, FixedMultiAirCompleteInteractionEndpointError> {
    if rows == 0 {
        return Err(FixedMultiAirCompleteInteractionEndpointError::Shape(
            "zero-row interaction trace",
        ));
    }
    let minimum = rows.checked_next_power_of_two().ok_or(
        FixedMultiAirCompleteInteractionEndpointError::Arithmetic("interaction trace height"),
    )?;
    match required {
        Some(height) if height.is_power_of_two() && height >= rows => Ok(height),
        Some(_) => Err(FixedMultiAirCompleteInteractionEndpointError::Shape(
            "interaction trace height",
        )),
        None => Ok(minimum),
    }
}

fn checked_pow2(
    exponent: usize,
    context: &'static str,
) -> Result<usize, FixedMultiAirCompleteInteractionEndpointError> {
    1usize
        .checked_shl(
            u32::try_from(exponent)
                .map_err(|_| FixedMultiAirCompleteInteractionEndpointError::Arithmetic(context))?,
        )
        .ok_or(FixedMultiAirCompleteInteractionEndpointError::Arithmetic(
            context,
        ))
}

fn try_zero_cells(
    height: usize,
    width: usize,
    context: &'static str,
) -> Result<Vec<F>, FixedMultiAirCompleteInteractionEndpointError> {
    let cells = height.checked_mul(width).ok_or(
        FixedMultiAirCompleteInteractionEndpointError::Arithmetic(context),
    )?;
    try_zero_vec(cells, context)
}

fn try_zero_vec(
    len: usize,
    context: &'static str,
) -> Result<Vec<F>, FixedMultiAirCompleteInteractionEndpointError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| FixedMultiAirCompleteInteractionEndpointError::Arithmetic(context))?;
    values.resize(len, F::ZERO);
    Ok(values)
}

fn ext_one_expr<AB: AirBuilder>() -> [AB::Expr; D_EF] {
    core::array::from_fn(|coordinate| {
        if coordinate == 0 {
            AB::Expr::ONE
        } else {
            AB::Expr::ZERO
        }
    })
}

fn select_power<AB: AirBuilder>(
    powers: [[AB::Var; D_EF]; MAX_EXACT_DEGREE + 1],
    flags: [AB::Var; MAX_EXACT_DEGREE + 1],
) -> [AB::Expr; D_EF] {
    core::array::from_fn(|limb| {
        (0..=MAX_EXACT_DEGREE).fold(AB::Expr::ZERO, |sum, power| {
            sum + AB::Expr::from(flags[power].clone()) * AB::Expr::from(powers[power][limb].clone())
        })
    })
}

fn powers(value: EF) -> [EF; MAX_EXACT_DEGREE + 1] {
    let mut powers = [EF::ONE; MAX_EXACT_DEGREE + 1];
    for exponent in 1..=MAX_EXACT_DEGREE {
        powers[exponent] = powers[exponent - 1] * value;
    }
    powers
}

fn basis_element(coordinate: usize) -> EF {
    EF::from_basis_coefficients_fn(|index| F::from_bool(index == coordinate))
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

#[cfg(test)]
pub(super) mod tests {
    use std::{
        panic::{catch_unwind, AssertUnwindSafe},
        sync::Arc,
    };

    use openvm_stark_backend::{
        air_builders::{
            debug::check_constraints,
            symbolic::{
                get_symbolic_builder, symbolic_expression::SymbolicExpression,
                SymbolicConstraintsDag, SymbolicRapBuilder,
            },
        },
        hasher::MerkleHasher,
        keygen::types::{StarkVerifyingKey, StarkVerifyingParams, TraceWidth},
        native_warp::{
            DirectAirCodeClass, DirectAirPesatIndex, DirectAirPublicSchema,
            FixedMultiAirCompletePesatIndex, FixedMultiAirCompleteTerminalCircuitPlan,
            FixedMultiAirCompleteTerminalLinearizer,
        },
        StarkProtocolConfig, SystemParams,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::{
        BabyBearPoseidon2Config as SC, Digest, DIGEST_SIZE,
    };
    use p3_air::{AirBuilder, AirBuilderWithPublicValues};

    use super::*;
    use crate::native_warp::terminal::fixed_multi_air_complete::region_cursor_openings::FixedMultiAirCompleteRegionCursorOpeningProfile;

    #[derive(Clone, Copy)]
    struct InteractionTestAir {
        sign: i8,
    }

    impl BaseAir<F> for InteractionTestAir {
        fn width(&self) -> usize {
            2
        }
    }

    impl BaseAirWithPublicValues<F> for InteractionTestAir {
        fn num_public_values(&self) -> usize {
            1
        }
    }

    impl PartitionedBaseAir<F> for InteractionTestAir {}

    impl Air<SymbolicRapBuilder<F>> for InteractionTestAir {
        fn eval(&self, builder: &mut SymbolicRapBuilder<F>) {
            let main = builder.common_main().clone();
            let row = main.row_slice(0).expect("interaction fixture row");
            let value = row[0];
            let multiplicity = row[1];
            builder.assert_zero(value - builder.public_values()[0]);
            if self.sign != 0 {
                let count = if self.sign > 0 {
                    SymbolicExpression::from(multiplicity)
                } else {
                    -SymbolicExpression::from(multiplicity)
                };
                builder.push_interaction(9, [value, value], count, 1);
            }
        }
    }

    fn digest(value: u32) -> Digest {
        [F::from_u32(value); DIGEST_SIZE]
    }

    fn verifying_key(air: &InteractionTestAir) -> StarkVerifyingKey<F, Digest> {
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
        air: &InteractionTestAir,
        air_id: usize,
    ) -> DirectAirPesatIndex<F, Digest>
    where
        H: MerkleHasher<F = F, Digest = Digest>,
    {
        DirectAirPesatIndex::from_verifying_key(
            hasher,
            digest(1),
            air_id,
            1,
            &verifying_key(air),
            None,
            DirectAirPublicSchema {
                public_values_len: 1,
                boundary_values_len: 0,
                schema_digest: digest(100 + air_id as u32),
            },
            DirectAirCodeClass {
                log_message_len: 2,
                log_blowup: 1,
                log_codeword_len: 3,
                initial_folding_factor: 0,
                rows_per_query: 2,
            },
        )
        .expect("direct relation")
    }

    pub(crate) fn backend_plan() -> Arc<FixedMultiAirCompleteTerminalCircuitPlan<F, Digest>> {
        let config = SC::default_from_params(SystemParams::new_for_testing(8));
        let hasher = config.hasher();
        let relations = [1i8, -1, 0]
            .into_iter()
            .enumerate()
            .map(|(index, sign)| direct_relation(hasher, &InteractionTestAir { sign }, 20 + index))
            .collect();
        let relation = FixedMultiAirCompletePesatIndex::from_direct_air_regions(
            hasher,
            digest(1),
            relations,
            DirectAirCodeClass {
                log_message_len: 5,
                log_blowup: 1,
                log_codeword_len: 6,
                initial_folding_factor: 0,
                rows_per_query: 2,
            },
        )
        .expect("complete relation");
        Arc::new(
            FixedMultiAirCompleteTerminalLinearizer::new(&relation)
                .expect("complete linearizer")
                .circuit_plan()
                .expect("backend-generated complete plan"),
        )
    }

    fn profile() -> Arc<FixedMultiAirCompleteInteractionEndpointProfile> {
        let plan = backend_plan();
        let region = &plan.regions[0];
        assert!(!region.interaction.interactions.is_empty());
        for interaction in &region.interaction.interactions {
            assert_eq!(
                interaction.global_term_degree,
                interaction
                    .count_degree
                    .checked_add(1)
                    .expect("small degree")
            );
        }
        Arc::new(
            FixedMultiAirCompleteInteractionEndpointProfile::new(
                FixedMultiAirCompleteInteractionEndpointPlanInput {
                    global_log_constraints: usize::from(plan.metadata.log_constraints),
                    global_explicit_len: usize::try_from(plan.metadata.explicit_len)
                        .expect("explicit length"),
                    exact_relation_degree: usize::from(plan.metadata.exact_relation_degree),
                    region,
                    scalars: &plan.scalars,
                    inverse_weight_blocks: &(0..region.interaction.interactions.len())
                        .map(|index| 70 + index)
                        .collect::<Vec<_>>(),
                    global_weight_block: 90,
                },
            )
            .expect("backend-generated interaction profile"),
        )
    }

    fn ef(seed: u64) -> EF {
        EF::from_basis_coefficients_fn(|coordinate| F::from_u64(seed + coordinate as u64 * 13))
    }

    fn endpoint_reference(
        profile: &FixedMultiAirCompleteInteractionEndpointProfile,
        one: EF,
        openings: &[EF],
        nodes: &[EF],
        denominators: &[EF],
        inverse_weights: &[EF],
        global_weight: EF,
    ) -> EF {
        let one_powers = powers(one);
        profile
            .component
            .interactions
            .iter()
            .enumerate()
            .map(|(ordinal, interaction)| {
                let q = interaction
                    .q_source_ordinals
                    .iter()
                    .enumerate()
                    .map(|(coordinate, &source)| {
                        basis_element(coordinate) * openings[source as usize]
                    })
                    .sum::<EF>();
                let inverse_degree = interaction.inverse_degree as usize;
                let global_degree = interaction.global_term_degree as usize;
                inverse_weights[ordinal]
                    * (denominators[ordinal] * q - one_powers[inverse_degree])
                    * one_powers[profile.exact_relation_degree - inverse_degree]
                    + global_weight
                        * nodes[interaction.count_root as usize]
                        * q
                        * one_powers[profile.exact_relation_degree - global_degree]
            })
            .sum()
    }

    fn assert_constraint_failure(operation: impl FnOnce()) {
        assert!(catch_unwind(AssertUnwindSafe(operation)).is_err());
    }

    #[test]
    fn backend_generated_nonempty_profile_and_ef4_endpoint_match() {
        let profile = profile();
        let cursor = FixedMultiAirCompleteRegionCursorOpeningProfile::from_plan(backend_plan())
            .expect("cursor profile");
        let cursor_component = cursor
            .components
            .iter()
            .find(|component| {
                component.kind == FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction
                    && component.region == profile.region
            })
            .expect("interaction cursor component");
        assert_eq!(
            cursor_component.point_common_count,
            1 + cursor_component.opening_count,
            "one endpoint selector plus one mapper point lookup per opening"
        );
        assert_eq!(
            cursor_component.point_fixed_source_count,
            profile.point_fixed_source_count(),
            "one independent fixed-column fold per fixed source"
        );
        assert_eq!(
            cursor_component.opening_lookup_counts,
            profile
                .opening_lookup_counts()
                .iter()
                .map(|count| u32::try_from(count + 1).expect("small lookup count"))
                .collect::<Vec<_>>(),
            "endpoint expression/q fanout plus one global mapper"
        );
        let mut expected_instance = vec![0usize; profile.instance_beta_lookup_counts().len()];
        for node in &profile.component.expression.nodes {
            expected_instance[profile.one_beta_coordinate] += 1;
            if let SymbolicExpressionNode::Variable(variable) = node {
                if variable.entry == Entry::Public {
                    expected_instance[profile.public_beta_start + variable.index] += 1;
                }
            }
        }
        let mut denominator_rows = 0usize;
        for interaction in &profile.component.interactions {
            let terms = 2 + interaction.message_roots.len();
            denominator_rows += terms;
            expected_instance[profile.one_beta_coordinate] += terms + 1;
            expected_instance[profile.alpha_beta_coordinate] += 1;
            expected_instance
                [profile.beta_power_beta_start + interaction.bus_beta_power_ordinal as usize] += 1;
            for message in 0..interaction.message_roots.len() {
                expected_instance[profile.beta_power_beta_start + message] += 1;
            }
        }
        assert_eq!(profile.instance_beta_lookup_counts(), expected_instance);
        let public_node_uses = profile
            .component
            .expression
            .nodes
            .iter()
            .filter(|node| {
                matches!(
                    node,
                    SymbolicExpressionNode::Variable(variable) if variable.entry == Entry::Public
                )
            })
            .count();
        assert_eq!(
            profile.instance_beta_lookup_counts().iter().sum::<usize>(),
            profile.component.expression.nodes.len()
                + public_node_uses
                + 2 * denominator_rows
                + profile.interaction_count(),
            "node, denominator, and endpoint instance lookups balance exactly"
        );
        let beta_len = profile
            .beta_power_beta_start
            .checked_add(profile.beta_power_count)
            .expect("beta length")
            .max(profile.public_beta_start + profile.public_count)
            .max(profile.alpha_beta_coordinate + 1)
            .max(profile.one_beta_coordinate + 1);
        let beta = (0..beta_len)
            .map(|index| ef(100 + index as u64))
            .collect::<Vec<_>>();
        let openings = (0..profile.component.expression.dynamic_columns.len())
            .map(|index| ef(300 + index as u64))
            .collect::<Vec<_>>();
        let fixed = (0..profile.component.expression.fixed_columns.len())
            .map(|index| ef(500 + index as u64))
            .collect::<Vec<_>>();
        let selectors = [ef(601), ef(602), ef(603)];
        let node_trace = generate_fixed_multi_air_complete_interaction_node_trace(
            &profile, &beta, &openings, &fixed, selectors, None,
        )
        .expect("node trace");
        let denominator_trace = generate_fixed_multi_air_complete_interaction_denominator_trace(
            &profile,
            &beta,
            &node_trace.values,
            None,
        )
        .expect("denominator trace");
        let inverse_weights = (0..profile.interaction_count())
            .map(|index| ef(700 + index as u64))
            .collect::<Vec<_>>();
        let global_weight = ef(800);
        let expected = endpoint_reference(
            &profile,
            beta[profile.one_beta_coordinate],
            &openings,
            &node_trace.values,
            &denominator_trace.denominators,
            &inverse_weights,
            global_weight,
        );
        let endpoint = generate_fixed_multi_air_complete_interaction_endpoint_trace(
            &profile,
            &beta,
            &openings,
            &node_trace.values,
            &denominator_trace.denominators,
            &inverse_weights,
            global_weight,
            expected,
            None,
        )
        .expect("endpoint trace");
        assert_eq!(endpoint.evaluation, expected);

        let endpoint_air = FixedMultiAirCompleteInteractionEndpointAir {
            profile: profile.clone(),
            opening_bus: FixedMultiAirCompleteInteractionRegionOpeningBus::new(1001),
            final_evaluation_bus: FixedMultiAirCompleteInteractionRegionFinalEvaluationBus::new(
                1002,
            ),
            instance_bus: FixedMultiAirCompleteInstanceValueBus::new(1003),
            node_bus: FixedMultiAirCompleteInteractionNodeBus::new(1004),
            denominator_bus: FixedMultiAirCompleteInteractionDenominatorBus::new(1005),
            eq_weight_bus: FixedMultiAirCompleteEqWeightBus::new(1006),
        };
        check_constraints::<_, SC>(
            &endpoint_air,
            "complete interaction endpoint",
            &None,
            &[endpoint.cached.as_view(), endpoint.common.as_view()],
            &[],
        );

        for mutate in 0..4 {
            let mut bad = endpoint.common.clone();
            let width = FixedMultiAirCompleteInteractionEndpointCols::<F>::width();
            let cols: &mut FixedMultiAirCompleteInteractionEndpointCols<F> =
                bad.values[..width].borrow_mut();
            match mutate {
                0 => cols.q_coordinates[0][0] += F::ONE,
                1 => cols.denominator[0] += F::ONE,
                2 => cols.inverse_weight[0] += F::ONE,
                _ => {
                    let last = profile.interaction_count() - 1;
                    let cols: &mut FixedMultiAirCompleteInteractionEndpointCols<F> =
                        bad.values[last * width..(last + 1) * width].borrow_mut();
                    cols.final_evaluation[0] += F::ONE;
                }
            }
            assert_constraint_failure(|| {
                check_constraints::<_, SC>(
                    &endpoint_air,
                    "mutated complete interaction endpoint",
                    &None,
                    &[endpoint.cached.as_view(), bad.as_view()],
                    &[],
                );
            });
        }
    }

    #[test]
    fn setup_fixed_interaction_columns_fold_at_exact_point_and_reject_mutations() {
        let mut owned_profile = (*profile()).clone();
        owned_profile.component.expression.fixed_columns = vec![
            FixedMultiAirCompleteTerminalCircuitFixedColumn::Cached {
                part: 0,
                column: 1,
                rotation: 0,
            },
            FixedMultiAirCompleteTerminalCircuitFixedColumn::Preprocessed {
                column: 0,
                rotation: 1,
            },
        ];
        owned_profile.fixed_opening_lookup_counts = vec![2, 1];
        let profile = Arc::new(owned_profile);
        assert_eq!(profile.point_fixed_source_count(), 2);

        let cached = [FixedMultiAirCompleteCachedTrace {
            width: 2,
            values: vec![
                F::from_u32(3),
                F::from_u32(5),
                F::from_u32(7),
                F::from_u32(11),
            ],
        }];
        let preprocessed = DirectAirFixedTrace {
            width: 1,
            values: vec![F::from_u32(13), F::from_u32(17)],
        };
        let setup = FixedMultiAirCompleteInteractionFixedValues::from_relation_setup(
            &profile,
            &cached,
            Some(&preprocessed),
        )
        .expect("fixed relation setup");
        let point = [ef(41)];
        let trace = generate_fixed_multi_air_complete_interaction_fixed_trace(
            &profile, &setup, &point, None,
        )
        .expect("fixed interaction fold");
        let expected_cached = EF::from(F::from_u32(5))
            + point[0] * (EF::from(F::from_u32(11)) - EF::from(F::from_u32(5)));
        // Rotation one changes [13, 17] into [17, 13] before the MLE fold.
        let expected_preprocessed = EF::from(F::from_u32(17))
            + point[0] * (EF::from(F::from_u32(13)) - EF::from(F::from_u32(17)));
        assert_eq!(trace.values, vec![expected_cached, expected_preprocessed]);

        let air = FixedMultiAirCompleteInteractionFixedAir {
            profile: profile.clone(),
            setup: Arc::new(setup.clone()),
            point_bus: FixedMultiAirCompleteInteractionRegionPointBus::new(1601),
            state_bus: FixedMultiAirCompleteInteractionFixedStateBus::new(1602),
            opening_bus: FixedMultiAirCompleteInteractionFixedOpeningBus::new(1603),
        };
        check_constraints::<_, SC>(
            &air,
            "interaction fixed fold",
            &Some(trace.preprocessed.as_view()),
            &[trace.common.as_view()],
            &[],
        );

        let lane_width = FixedMultiAirCompleteInteractionFixedLaneCols::<F>::width();
        let common_width = INTERACTION_FIXED_LANES * lane_width;
        let mut bad_fold = trace.common.clone();
        // Entry two is the first fold and occupies lane zero of packed row one.
        let fold_row = 1;
        let cols: &mut FixedMultiAirCompleteInteractionFixedLaneCols<F> = bad_fold.values
            [fold_row * common_width..fold_row * common_width + lane_width]
            .borrow_mut();
        cols.value[0] += F::ONE;
        assert_constraint_failure(|| {
            check_constraints::<_, SC>(
                &air,
                "mutated interaction fixed fold",
                &Some(trace.preprocessed.as_view()),
                &[bad_fold.as_view()],
                &[],
            );
        });

        let mut wrong_profile = (*profile).clone();
        wrong_profile.component.expression.fixed_columns[1] =
            FixedMultiAirCompleteTerminalCircuitFixedColumn::Preprocessed {
                column: 0,
                rotation: 0,
            };
        assert!(generate_fixed_multi_air_complete_interaction_fixed_trace(
            &wrong_profile,
            &setup,
            &point,
            None,
        )
        .is_err());
        assert!(
            FixedMultiAirCompleteInteractionFixedValues::from_relation_setup(
                &profile, &cached, None,
            )
            .is_err()
        );
    }

    #[test]
    fn final_target_and_empty_profile_are_not_witness_selectable() {
        let profile = profile();
        let beta_len = profile.instance_beta_lookup_counts().len();
        let beta = vec![EF::ONE; beta_len];
        let openings = vec![EF::ONE; profile.component.expression.dynamic_columns.len()];
        let fixed = vec![EF::ONE; profile.component.expression.fixed_columns.len()];
        let nodes = generate_fixed_multi_air_complete_interaction_node_trace(
            &profile,
            &beta,
            &openings,
            &fixed,
            [EF::ONE; 3],
            None,
        )
        .expect("nodes");
        let denominators = generate_fixed_multi_air_complete_interaction_denominator_trace(
            &profile,
            &beta,
            &nodes.values,
            None,
        )
        .expect("denominators");
        assert!(
            generate_fixed_multi_air_complete_interaction_endpoint_trace(
                &profile,
                &beta,
                &openings,
                &nodes.values,
                &denominators.denominators,
                &vec![EF::ONE; profile.interaction_count()],
                EF::ONE,
                EF::from_u32(123),
                None,
            )
            .is_err()
        );

        let plan = backend_plan();
        let empty_region = plan
            .regions
            .iter()
            .find(|region| !region.interaction.identity.proof_present)
            .expect("backend-generated empty interaction region");
        let empty = FixedMultiAirCompleteInteractionEndpointProfile::new(
            FixedMultiAirCompleteInteractionEndpointPlanInput {
                global_log_constraints: usize::from(plan.metadata.log_constraints),
                global_explicit_len: usize::try_from(plan.metadata.explicit_len)
                    .expect("explicit length"),
                exact_relation_degree: usize::from(plan.metadata.exact_relation_degree),
                region: empty_region,
                scalars: &plan.scalars,
                inverse_weight_blocks: &[],
                global_weight_block: 90,
            },
        )
        .expect("setup-fixed empty profile");
        assert!(
            generate_fixed_multi_air_complete_empty_interaction_trace(&empty, EF::ZERO, None)
                .is_ok()
        );
        assert!(
            generate_fixed_multi_air_complete_empty_interaction_trace(&empty, EF::ONE, None)
                .is_err()
        );
    }
}

use core::borrow::{Borrow, BorrowMut};
use std::collections::{BTreeMap, BTreeSet};

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
    native_warp::{DirectAirMappedRotation, DirectAirPesatIndex, FixedMultiAirPesatIndex},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    native_warp::terminal::fixed_multi_air::{
        FixedMultiAirBetaCoordinateBus, FixedMultiAirBetaCoordinateMessage,
        FixedMultiAirConstraintWeightBus, FixedMultiAirConstraintWeightMessage,
        FixedMultiAirEndpointFixedValueBus, FixedMultiAirEndpointFixedValueMessage,
        FixedMultiAirEndpointNodeBus, FixedMultiAirEndpointNodeMessage,
        FixedMultiAirRegionFinalEvaluationBus, FixedMultiAirRegionFinalEvaluationMessage,
        FixedMultiAirRegionOpeningBus, FixedMultiAirRegionOpeningMessage,
    },
    utils::{ext_field_add, ext_field_multiply, ext_field_subtract},
};

const KIND_COUNT: usize = 6;
const KIND_SOURCE: usize = 0;
const KIND_CONSTANT: usize = 1;
const KIND_ADD: usize = 2;
const KIND_SUB: usize = 3;
const KIND_NEG: usize = 4;
const KIND_MUL: usize = 5;

const SOURCE_KIND_COUNT: usize = 3;
const SOURCE_OPENING: usize = 0;
const SOURCE_FIXED: usize = 1;
const SOURCE_BETA: usize = 2;

/// Direct terminal degree envelope.  It is protocol-fixed at five; the extra
/// equality factor belongs to the sumcheck, not to this DAG evaluation.
const MAX_NODE_DEGREE: usize = 5;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedMultiAirEndpointTraceError {
    Shape,
    Relation,
    Evaluation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FixedMultiAirDynamicColumnPlan {
    pub part: usize,
    pub column: usize,
    pub rotation: DirectAirMappedRotation,
    pub local_block_start: usize,
    pub global_block_start: usize,
    pub log_height: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct FixedMultiAirFixedColumnPlan {
    pub column: usize,
    pub rotation: DirectAirMappedRotation,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedMultiAirEndpointSource {
    Opening(usize),
    Fixed(usize),
    Beta(usize),
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirEndpointPlan {
    pub region: usize,
    pub log_height: usize,
    /// Coordinate of the distinguished constant-one value in the complete
    /// global beta vector.  This is fixed by the admitted global relation,
    /// not by the local regional PESAT shape.
    pub one_coordinate: usize,
    pub dynamic: Vec<FixedMultiAirDynamicColumnPlan>,
    pub fixed: Vec<FixedMultiAirFixedColumnPlan>,
    /// Three synthetic fixed tables follow the preprocessed sources:
    /// is-first, is-last, and is-transition.
    pub first_source: usize,
    pub last_source: usize,
    pub transition_source: usize,
    pub node_sources: Vec<Option<FixedMultiAirEndpointSource>>,
    pub fanout: Vec<usize>,
}

impl FixedMultiAirEndpointPlan {
    pub fn from_relation(
        relation: &FixedMultiAirPesatIndex<F, Digest>,
        region: usize,
    ) -> Result<Self, FixedMultiAirEndpointTraceError> {
        let local = relation
            .region_relation(region)
            .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
        let region_description = relation
            .description()
            .regions
            .get(region)
            .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
        let layout = &local.description().shard_key.trace_layout;
        let widths = layout.part_widths();
        let height = local.height();
        let log_height = local.description().shard_key.log_height as usize;
        let mut part_starts = Vec::with_capacity(widths.len());
        let mut offset = 0usize;
        for &width in &widths {
            part_starts.push(offset);
            offset = offset
                .checked_add(width.saturating_mul(height))
                .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
        }
        if offset != local.raw_witness_len() {
            return Err(FixedMultiAirEndpointTraceError::Relation);
        }

        let mut dynamic_set = BTreeSet::new();
        let mut fixed_set = BTreeSet::new();
        for node in &local.constraint_dag().nodes {
            if let SymbolicExpressionNode::Variable(variable) = node {
                let rotation = rotation_from_entry(variable.entry)?;
                match variable.entry {
                    Entry::Main { part_index, .. } => {
                        let width = *widths
                            .get(part_index)
                            .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
                        if variable.index >= width {
                            return Err(FixedMultiAirEndpointTraceError::Relation);
                        }
                        let local_block_start = part_starts[part_index]
                            .checked_add(variable.index.saturating_mul(height))
                            .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
                        let global_block_start = usize::try_from(region_description.message_offset)
                            .ok()
                            .and_then(|start| start.checked_add(local_block_start))
                            .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
                        dynamic_set.insert(FixedMultiAirDynamicColumnPlan {
                            part: part_index,
                            column: variable.index,
                            rotation,
                            local_block_start,
                            global_block_start,
                            log_height,
                        });
                    }
                    Entry::Preprocessed { .. } => {
                        fixed_set.insert(FixedMultiAirFixedColumnPlan {
                            column: variable.index,
                            rotation,
                        });
                    }
                    Entry::Public => {}
                    Entry::Challenge => return Err(FixedMultiAirEndpointTraceError::Relation),
                }
            }
        }
        let dynamic = dynamic_set.into_iter().collect::<Vec<_>>();
        let fixed = fixed_set.into_iter().collect::<Vec<_>>();
        let dynamic_indices = dynamic
            .iter()
            .copied()
            .enumerate()
            .map(|(index, source)| (source, index))
            .collect::<BTreeMap<_, _>>();
        let fixed_indices = fixed
            .iter()
            .copied()
            .enumerate()
            .map(|(index, source)| (source, index))
            .collect::<BTreeMap<_, _>>();
        let first_source = fixed.len();
        let last_source = first_source + 1;
        let transition_source = first_source + 2;
        let global_log = relation.pesat_shape().log_constraints;
        let public_start = usize::try_from(region_description.explicit_offset)
            .map_err(|_| FixedMultiAirEndpointTraceError::Relation)?;
        let mut node_sources = Vec::with_capacity(local.constraint_dag().nodes.len());
        for node in &local.constraint_dag().nodes {
            let source = match node {
                SymbolicExpressionNode::Variable(variable) => match variable.entry {
                    Entry::Main { part_index, .. } => {
                        let rotation = rotation_from_entry(variable.entry)?;
                        let local_block_start = part_starts[part_index]
                            .checked_add(variable.index.saturating_mul(height))
                            .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
                        let global_block_start = usize::try_from(region_description.message_offset)
                            .ok()
                            .and_then(|start| start.checked_add(local_block_start))
                            .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
                        let key = FixedMultiAirDynamicColumnPlan {
                            part: part_index,
                            column: variable.index,
                            rotation,
                            local_block_start,
                            global_block_start,
                            log_height,
                        };
                        Some(FixedMultiAirEndpointSource::Opening(
                            *dynamic_indices
                                .get(&key)
                                .ok_or(FixedMultiAirEndpointTraceError::Relation)?,
                        ))
                    }
                    Entry::Preprocessed { .. } => {
                        let key = FixedMultiAirFixedColumnPlan {
                            column: variable.index,
                            rotation: rotation_from_entry(variable.entry)?,
                        };
                        Some(FixedMultiAirEndpointSource::Fixed(
                            *fixed_indices
                                .get(&key)
                                .ok_or(FixedMultiAirEndpointTraceError::Relation)?,
                        ))
                    }
                    Entry::Public => Some(FixedMultiAirEndpointSource::Beta(
                        global_log
                            .checked_add(public_start)
                            .and_then(|x| x.checked_add(variable.index))
                            .ok_or(FixedMultiAirEndpointTraceError::Relation)?,
                    )),
                    Entry::Challenge => return Err(FixedMultiAirEndpointTraceError::Relation),
                },
                SymbolicExpressionNode::IsFirstRow => {
                    Some(FixedMultiAirEndpointSource::Fixed(first_source))
                }
                SymbolicExpressionNode::IsLastRow => {
                    Some(FixedMultiAirEndpointSource::Fixed(last_source))
                }
                SymbolicExpressionNode::IsTransition => {
                    Some(FixedMultiAirEndpointSource::Fixed(transition_source))
                }
                _ => None,
            };
            node_sources.push(source);
        }
        let mut fanout = vec![0usize; local.constraint_dag().nodes.len()];
        for node in &local.constraint_dag().nodes {
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
                    *fanout
                        .get_mut(*left_idx)
                        .ok_or(FixedMultiAirEndpointTraceError::Relation)? += 1;
                    *fanout
                        .get_mut(*right_idx)
                        .ok_or(FixedMultiAirEndpointTraceError::Relation)? += 1;
                }
                SymbolicExpressionNode::Neg { idx, .. } => {
                    *fanout
                        .get_mut(*idx)
                        .ok_or(FixedMultiAirEndpointTraceError::Relation)? += 1;
                }
                _ => {}
            }
        }
        for &root in &local.constraint_dag().constraint_idx {
            *fanout
                .get_mut(root)
                .ok_or(FixedMultiAirEndpointTraceError::Relation)? += 1;
        }
        Ok(Self {
            region,
            log_height,
            one_coordinate: global_log,
            dynamic,
            fixed,
            first_source,
            last_source,
            transition_source,
            node_sources,
            fanout,
        })
    }

    #[must_use]
    pub fn fixed_source_count(&self) -> usize {
        self.fixed.len() + 3
    }

    /// Fixed columns represented by a dense table and folded through all
    /// nodes at each sumcheck coordinate.
    #[must_use]
    pub fn dense_fixed_source_count(&self) -> usize {
        self.fixed.len()
    }

    /// Canonical row selectors evaluated by one analytic product step per
    /// coordinate instead of a dense fold tree.
    #[must_use]
    pub fn analytic_fixed_source_count(&self) -> usize {
        self.fixed_source_count() - self.dense_fixed_source_count()
    }

    #[must_use]
    pub fn opening_lookup_counts(&self) -> Vec<usize> {
        let mut counts = vec![1usize; self.dynamic.len()]; // mapped target recurrence
        for source in self.node_sources.iter().flatten() {
            if let FixedMultiAirEndpointSource::Opening(index) = *source {
                counts[index] += 1;
            }
        }
        counts
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirEndpointNodeScheduleCols<T> {
    pub active: T,
    pub kind_flags: [T; KIND_COUNT],
    pub source_flags: [T; SOURCE_KIND_COUNT],
    pub node: T,
    pub arg0_node: T,
    pub arg1_node: T,
    pub source_index: T,
    pub fanout: T,
    pub constant: [T; D_EF],
    pub left_power_flags: [T; MAX_NODE_DEGREE + 1],
    pub right_power_flags: [T; MAX_NODE_DEGREE + 1],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirEndpointNodeCols<T> {
    pub source_value: [T; D_EF],
    pub arg0: [T; D_EF],
    pub arg1: [T; D_EF],
    pub value: [T; D_EF],
    pub one: [T; D_EF],
    pub one_powers: [[T; D_EF]; MAX_NODE_DEGREE + 1],
}

pub struct FixedMultiAirEndpointNodeAir {
    pub opening_bus: FixedMultiAirRegionOpeningBus,
    pub fixed_value_bus: FixedMultiAirEndpointFixedValueBus,
    pub beta_bus: FixedMultiAirBetaCoordinateBus,
    pub node_bus: FixedMultiAirEndpointNodeBus,
    pub region: usize,
    pub one_coordinate: usize,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirEndpointNodeAir {}
impl PartitionedBaseAir<F> for FixedMultiAirEndpointNodeAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirEndpointNodeScheduleCols::<F>::width()]
    }
    fn common_main_width(&self) -> usize {
        FixedMultiAirEndpointNodeCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirEndpointNodeAir {}
impl BaseAir<F> for FixedMultiAirEndpointNodeAir {
    fn width(&self) -> usize {
        FixedMultiAirEndpointNodeScheduleCols::<F>::width()
            + FixedMultiAirEndpointNodeCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirEndpointNodeAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached_row = builder.cached_mains()[0]
            .row_slice(0)
            .expect("endpoint DAG schedule")
            .to_vec();
        let common_row = builder
            .common_main()
            .row_slice(0)
            .expect("endpoint DAG row")
            .to_vec();
        let schedule: &FixedMultiAirEndpointNodeScheduleCols<AB::Var> =
            cached_row.as_slice().borrow();
        let local: &FixedMultiAirEndpointNodeCols<AB::Var> = common_row.as_slice().borrow();
        let active = schedule
            .kind_flags
            .iter()
            .fold(AB::Expr::ZERO, |sum, &x| sum + x);
        builder.assert_eq(active.clone(), schedule.active);
        builder.assert_bool(schedule.active);
        let source_sum = schedule
            .source_flags
            .iter()
            .fold(AB::Expr::ZERO, |sum, &x| sum + x);
        builder.assert_eq(source_sum, schedule.kind_flags[KIND_SOURCE]);
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
                .fold(AB::Expr::ZERO, |sum, &x| sum + x),
            schedule.active,
        );
        builder.assert_eq(
            schedule
                .right_power_flags
                .iter()
                .fold(AB::Expr::ZERO, |sum, &x| sum + x),
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
        for power in 0..MAX_NODE_DEGREE {
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
        let neg = local.arg0.map(|x| -AB::Expr::from(x));
        let mul = ext_field_multiply::<AB::Expr>(local.arg0, local.arg1);
        let expected = core::array::from_fn(|limb| {
            schedule.kind_flags[KIND_SOURCE] * local.source_value[limb]
                + schedule.kind_flags[KIND_CONSTANT] * schedule.constant[limb]
                + schedule.kind_flags[KIND_ADD] * add[limb].clone()
                + schedule.kind_flags[KIND_SUB] * sub[limb].clone()
                + schedule.kind_flags[KIND_NEG] * neg[limb].clone()
                + schedule.kind_flags[KIND_MUL] * mul[limb].clone()
        });
        assert_array_eq(builder, local.value, expected);

        self.opening_bus.lookup_key(
            builder,
            FixedMultiAirRegionOpeningMessage {
                region: AB::Expr::from_usize(self.region),
                opening: schedule.source_index.into(),
                value: local.source_value.map(Into::into),
            },
            schedule.source_flags[SOURCE_OPENING],
        );
        self.fixed_value_bus.lookup_key(
            builder,
            FixedMultiAirEndpointFixedValueMessage {
                region: AB::Expr::from_usize(self.region),
                source: schedule.source_index.into(),
                value: local.source_value.map(Into::into),
            },
            schedule.source_flags[SOURCE_FIXED],
        );
        self.beta_bus.lookup_key(
            builder,
            FixedMultiAirBetaCoordinateMessage {
                coordinate: schedule.source_index.into(),
                value: local.source_value.map(Into::into),
            },
            schedule.source_flags[SOURCE_BETA],
        );
        self.beta_bus.lookup_key(
            builder,
            FixedMultiAirBetaCoordinateMessage {
                coordinate: AB::Expr::from_usize(self.one_coordinate),
                value: local.one.map(Into::into),
            },
            schedule.active,
        );
        self.node_bus.lookup_key(
            builder,
            FixedMultiAirEndpointNodeMessage {
                region: AB::Expr::from_usize(self.region),
                node: schedule.arg0_node.into(),
                value: local.arg0.map(Into::into),
            },
            schedule.kind_flags[KIND_ADD]
                + schedule.kind_flags[KIND_SUB]
                + schedule.kind_flags[KIND_NEG]
                + schedule.kind_flags[KIND_MUL],
        );
        self.node_bus.lookup_key(
            builder,
            FixedMultiAirEndpointNodeMessage {
                region: AB::Expr::from_usize(self.region),
                node: schedule.arg1_node.into(),
                value: local.arg1.map(Into::into),
            },
            schedule.kind_flags[KIND_ADD]
                + schedule.kind_flags[KIND_SUB]
                + schedule.kind_flags[KIND_MUL],
        );
        self.node_bus.add_key_with_lookups(
            builder,
            FixedMultiAirEndpointNodeMessage {
                region: AB::Expr::from_usize(self.region),
                node: schedule.node.into(),
                value: local.value.map(Into::into),
            },
            schedule.fanout,
        );
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirEndpointNodeTraceOutput {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub values: Vec<EF>,
}

pub fn generate_fixed_multi_air_endpoint_node_traces(
    relation: &DirectAirPesatIndex<F, Digest>,
    plan: &FixedMultiAirEndpointPlan,
    global_beta: &[EF],
    openings: &[EF],
    fixed_values: &[EF],
    required_height: Option<usize>,
) -> Result<FixedMultiAirEndpointNodeTraceOutput, FixedMultiAirEndpointTraceError> {
    let dag = relation.constraint_dag();
    let degrees = relation.homogeneous_node_degrees();
    if dag.nodes.len() != plan.node_sources.len()
        || dag.nodes.len() != plan.fanout.len()
        || dag.nodes.len() != degrees.len()
        || openings.len() != plan.dynamic.len()
        || fixed_values.len() != plan.fixed_source_count()
    {
        return Err(FixedMultiAirEndpointTraceError::Shape);
    }
    let one = *global_beta
        .get(plan.one_coordinate)
        .ok_or(FixedMultiAirEndpointTraceError::Shape)?;
    let valid_rows = dag.nodes.len();
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if valid_rows == 0 || height < valid_rows {
        return Err(FixedMultiAirEndpointTraceError::Shape);
    }
    let cached_width = FixedMultiAirEndpointNodeScheduleCols::<F>::width();
    let common_width = FixedMultiAirEndpointNodeCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    let mut values = Vec::with_capacity(valid_rows);
    let one_powers = powers(one);
    for (node_index, node) in dag.nodes.iter().enumerate() {
        let mut kind = KIND_SOURCE;
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
        let target_degree = usize::from(degrees[node_index]);
        let value = match node {
            SymbolicExpressionNode::Variable(_)
            | SymbolicExpressionNode::IsFirstRow
            | SymbolicExpressionNode::IsLastRow
            | SymbolicExpressionNode::IsTransition => {
                let source = plan.node_sources[node_index]
                    .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
                source_value = match source {
                    FixedMultiAirEndpointSource::Opening(index) => {
                        source_kind = Some(SOURCE_OPENING);
                        source_index = index;
                        *openings
                            .get(index)
                            .ok_or(FixedMultiAirEndpointTraceError::Shape)?
                    }
                    FixedMultiAirEndpointSource::Fixed(index) => {
                        source_kind = Some(SOURCE_FIXED);
                        source_index = index;
                        *fixed_values
                            .get(index)
                            .ok_or(FixedMultiAirEndpointTraceError::Shape)?
                    }
                    FixedMultiAirEndpointSource::Beta(coordinate) => {
                        source_kind = Some(SOURCE_BETA);
                        source_index = coordinate;
                        *global_beta
                            .get(coordinate)
                            .ok_or(FixedMultiAirEndpointTraceError::Shape)?
                    }
                };
                source_value
            }
            SymbolicExpressionNode::Constant(value) => {
                kind = KIND_CONSTANT;
                constant = EF::from(*value);
                constant
            }
            SymbolicExpressionNode::Add {
                left_idx,
                right_idx,
                ..
            } => {
                kind = KIND_ADD;
                arg0_node = *left_idx;
                arg1_node = *right_idx;
                arg0 = *values
                    .get(*left_idx)
                    .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
                arg1 = *values
                    .get(*right_idx)
                    .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
                left_delta = target_degree
                    .checked_sub(usize::from(degrees[*left_idx]))
                    .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
                right_delta = target_degree
                    .checked_sub(usize::from(degrees[*right_idx]))
                    .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
                arg0 * one_powers[left_delta] + arg1 * one_powers[right_delta]
            }
            SymbolicExpressionNode::Sub {
                left_idx,
                right_idx,
                ..
            } => {
                kind = KIND_SUB;
                arg0_node = *left_idx;
                arg1_node = *right_idx;
                arg0 = *values
                    .get(*left_idx)
                    .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
                arg1 = *values
                    .get(*right_idx)
                    .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
                left_delta = target_degree
                    .checked_sub(usize::from(degrees[*left_idx]))
                    .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
                right_delta = target_degree
                    .checked_sub(usize::from(degrees[*right_idx]))
                    .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
                arg0 * one_powers[left_delta] - arg1 * one_powers[right_delta]
            }
            SymbolicExpressionNode::Neg { idx, .. } => {
                kind = KIND_NEG;
                arg0_node = *idx;
                arg0 = *values
                    .get(*idx)
                    .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
                -arg0
            }
            SymbolicExpressionNode::Mul {
                left_idx,
                right_idx,
                ..
            } => {
                kind = KIND_MUL;
                arg0_node = *left_idx;
                arg1_node = *right_idx;
                arg0 = *values
                    .get(*left_idx)
                    .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
                arg1 = *values
                    .get(*right_idx)
                    .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
                arg0 * arg1
            }
        };
        if left_delta > MAX_NODE_DEGREE || right_delta > MAX_NODE_DEGREE {
            return Err(FixedMultiAirEndpointTraceError::Relation);
        }
        values.push(value);
        let cached_row = &mut cached[node_index * cached_width..(node_index + 1) * cached_width];
        let schedule: &mut FixedMultiAirEndpointNodeScheduleCols<F> = cached_row.borrow_mut();
        schedule.active = F::ONE;
        schedule.kind_flags[kind] = F::ONE;
        if let Some(source_kind) = source_kind {
            schedule.source_flags[source_kind] = F::ONE;
        }
        schedule.node = F::from_usize(node_index);
        schedule.arg0_node = F::from_usize(arg0_node);
        schedule.arg1_node = F::from_usize(arg1_node);
        schedule.source_index = F::from_usize(source_index);
        schedule.fanout = F::from_usize(plan.fanout[node_index]);
        copy_ext(&mut schedule.constant, constant);
        schedule.left_power_flags[left_delta] = F::ONE;
        schedule.right_power_flags[right_delta] = F::ONE;

        let common_row = &mut common[node_index * common_width..(node_index + 1) * common_width];
        let cols: &mut FixedMultiAirEndpointNodeCols<F> = common_row.borrow_mut();
        copy_ext(&mut cols.source_value, source_value);
        copy_ext(&mut cols.arg0, arg0);
        copy_ext(&mut cols.arg1, arg1);
        copy_ext(&mut cols.value, value);
        copy_ext(&mut cols.one, one);
        for (target, &power) in cols.one_powers.iter_mut().zip(&one_powers) {
            copy_ext(target, power);
        }
    }
    Ok(FixedMultiAirEndpointNodeTraceOutput {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        values,
    })
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirEndpointFoldScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub constraint: T,
    pub root_node: T,
    pub root_power_flags: [T; MAX_NODE_DEGREE + 1],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirEndpointFoldCols<T> {
    pub root_value: [T; D_EF],
    pub weight: [T; D_EF],
    pub one: [T; D_EF],
    pub one_powers: [[T; D_EF]; MAX_NODE_DEGREE + 1],
    pub term: [T; D_EF],
    pub sum_before: [T; D_EF],
    pub sum_after: [T; D_EF],
}

pub struct FixedMultiAirEndpointFoldAir {
    pub node_bus: FixedMultiAirEndpointNodeBus,
    pub weight_bus: FixedMultiAirConstraintWeightBus,
    pub beta_bus: FixedMultiAirBetaCoordinateBus,
    pub final_evaluation_bus: FixedMultiAirRegionFinalEvaluationBus,
    pub region: usize,
    pub one_coordinate: usize,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirEndpointFoldAir {}
impl PartitionedBaseAir<F> for FixedMultiAirEndpointFoldAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirEndpointFoldScheduleCols::<F>::width()]
    }
    fn common_main_width(&self) -> usize {
        FixedMultiAirEndpointFoldCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirEndpointFoldAir {}
impl BaseAir<F> for FixedMultiAirEndpointFoldAir {
    fn width(&self) -> usize {
        FixedMultiAirEndpointFoldScheduleCols::<F>::width()
            + FixedMultiAirEndpointFoldCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirEndpointFoldAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("endpoint fold schedule")
            .to_vec();
        let next_cached = builder.cached_mains()[0]
            .row_slice(1)
            .expect("next endpoint fold schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("endpoint fold row")
            .to_vec();
        let next_common = builder
            .common_main()
            .row_slice(1)
            .expect("next endpoint fold row")
            .to_vec();
        let schedule: &FixedMultiAirEndpointFoldScheduleCols<AB::Var> = cached.as_slice().borrow();
        let next_schedule: &FixedMultiAirEndpointFoldScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FixedMultiAirEndpointFoldCols<AB::Var> = common.as_slice().borrow();
        let next: &FixedMultiAirEndpointFoldCols<AB::Var> = next_common.as_slice().borrow();
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
                .fold(AB::Expr::ZERO, |sum, &x| sum + x),
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
        let mut transition = builder.when_transition();
        let mut same = transition.when(next_schedule.active);
        assert_array_eq(&mut same, next.sum_before, local.sum_after);
        let one = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            }
        });
        assert_array_eq(&mut builder.when(schedule.active), local.one_powers[0], one);
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.one_powers[1],
            local.one.map(Into::into),
        );
        for power in 0..MAX_NODE_DEGREE {
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
            FixedMultiAirEndpointNodeMessage {
                region: AB::Expr::from_usize(self.region),
                node: schedule.root_node.into(),
                value: local.root_value.map(Into::into),
            },
            schedule.active,
        );
        self.weight_bus.lookup_key(
            builder,
            FixedMultiAirConstraintWeightMessage {
                region: AB::Expr::from_usize(self.region),
                constraint: schedule.constraint.into(),
                value: local.weight.map(Into::into),
            },
            schedule.active,
        );
        self.beta_bus.lookup_key(
            builder,
            FixedMultiAirBetaCoordinateMessage {
                coordinate: AB::Expr::from_usize(self.one_coordinate),
                value: local.one.map(Into::into),
            },
            schedule.active,
        );
        self.final_evaluation_bus.receive(
            builder,
            FixedMultiAirRegionFinalEvaluationMessage {
                region: AB::Expr::from_usize(self.region),
                claim: local.sum_after.map(Into::into),
            },
            schedule.is_last,
        );
    }
}

pub fn generate_fixed_multi_air_endpoint_fold_traces(
    relation: &DirectAirPesatIndex<F, Digest>,
    node_values: &[EF],
    weights: &[EF],
    one: EF,
    degree_delta: usize,
    required_height: Option<usize>,
) -> Result<(RowMajorMatrix<F>, RowMajorMatrix<F>, EF), FixedMultiAirEndpointTraceError> {
    let roots = &relation.constraint_dag().constraint_idx;
    let degrees = relation.homogeneous_node_degrees();
    if roots.is_empty() || roots.len() != weights.len() {
        return Err(FixedMultiAirEndpointTraceError::Shape);
    }
    let height = required_height.unwrap_or_else(|| roots.len().next_power_of_two());
    if height < roots.len() {
        return Err(FixedMultiAirEndpointTraceError::Shape);
    }
    let cached_width = FixedMultiAirEndpointFoldScheduleCols::<F>::width();
    let common_width = FixedMultiAirEndpointFoldCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    let one_powers = powers(one);
    let mut sum = EF::ZERO;
    for (constraint, (&root, &weight)) in roots.iter().zip(weights).enumerate() {
        let root_value = *node_values
            .get(root)
            .ok_or(FixedMultiAirEndpointTraceError::Shape)?;
        let delta = relation
            .exact_max_degree()
            .checked_sub(usize::from(
                *degrees
                    .get(root)
                    .ok_or(FixedMultiAirEndpointTraceError::Relation)?,
            ))
            .ok_or(FixedMultiAirEndpointTraceError::Relation)?
            .checked_add(degree_delta)
            .ok_or(FixedMultiAirEndpointTraceError::Relation)?;
        if delta > MAX_NODE_DEGREE {
            return Err(FixedMultiAirEndpointTraceError::Relation);
        }
        let term = weight * root_value * one_powers[delta];
        let before = sum;
        sum += term;
        let cached_row = &mut cached[constraint * cached_width..(constraint + 1) * cached_width];
        let schedule: &mut FixedMultiAirEndpointFoldScheduleCols<F> = cached_row.borrow_mut();
        schedule.active = F::ONE;
        schedule.is_first = F::from_bool(constraint == 0);
        schedule.is_last = F::from_bool(constraint + 1 == roots.len());
        schedule.constraint = F::from_usize(constraint);
        schedule.root_node = F::from_usize(root);
        schedule.root_power_flags[delta] = F::ONE;
        let common_row = &mut common[constraint * common_width..(constraint + 1) * common_width];
        let cols: &mut FixedMultiAirEndpointFoldCols<F> = common_row.borrow_mut();
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
    Ok((
        RowMajorMatrix::new(cached, cached_width),
        RowMajorMatrix::new(common, common_width),
        sum,
    ))
}

fn rotation_from_entry(
    entry: Entry,
) -> Result<DirectAirMappedRotation, FixedMultiAirEndpointTraceError> {
    match entry.offset() {
        Some(0) => Ok(DirectAirMappedRotation::Current),
        Some(1) => Ok(DirectAirMappedRotation::Next),
        None if matches!(entry, Entry::Public) => Ok(DirectAirMappedRotation::Current),
        _ => Err(FixedMultiAirEndpointTraceError::Relation),
    }
}

fn powers(one: EF) -> [EF; MAX_NODE_DEGREE + 1] {
    let mut result = [EF::ONE; MAX_NODE_DEGREE + 1];
    for power in 0..MAX_NODE_DEGREE {
        result[power + 1] = result[power] * one;
    }
    result
}

fn select_power<AB: AirBuilder<F = F>>(
    powers: [[AB::Var; D_EF]; MAX_NODE_DEGREE + 1],
    flags: [AB::Var; MAX_NODE_DEGREE + 1],
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

const _: () = assert!(D_EF == 4);

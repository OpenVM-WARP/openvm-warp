use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder, interaction::InteractionBuilder,
    native_warp::DirectAirPesatIndex, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_air::{Air, BaseAir};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    native_warp::terminal::fixed_multi_air::{
        FixedMultiAirEndpointFixedFoldStateBus, FixedMultiAirEndpointFixedFoldStateMessage,
        FixedMultiAirEndpointFixedValueBus, FixedMultiAirEndpointFixedValueMessage,
        FixedMultiAirEndpointPlan, FixedMultiAirEndpointSource, FixedMultiAirRegionPointBus,
        FixedMultiAirRegionPointMessage,
    },
    utils::{ext_field_add, ext_field_multiply, ext_field_subtract},
};

const KIND_COUNT: usize = 4;
const KIND_INITIAL: usize = 0;
const KIND_FOLD: usize = 1;
const KIND_FINAL: usize = 2;
const KIND_SELECTOR_STEP: usize = 3;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedMultiAirEndpointFixedTraceError {
    Shape,
    Relation,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirEndpointFixedScheduleCols<T> {
    pub active: T,
    pub kind_flags: [T; KIND_COUNT],
    pub source: T,
    pub layer: T,
    pub index: T,
    pub low_index: T,
    pub high_index: T,
    pub point_coordinate: T,
    pub initial_value: [T; D_EF],
    pub fixed_value_lookup_count: T,
    /// Selects `point` instead of `1-point` in an analytic row-selector
    /// product. This is fixed in the cached schedule.
    pub selector_uses_point: T,
    /// Complements the final product for `is_transition = 1-is_last`.
    pub selector_complement: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirEndpointFixedCols<T> {
    pub low: [T; D_EF],
    pub high: [T; D_EF],
    pub point: [T; D_EF],
    pub value: [T; D_EF],
}

/// Verifier-key fixed MLE evaluation. Preprocessed-table cells live only in
/// cached rows and are folded normally. The three canonical row selectors use
/// their closed-form products, avoiding an O(trace height) recursive witness.
pub struct FixedMultiAirEndpointFixedAir {
    pub point_bus: FixedMultiAirRegionPointBus,
    pub state_bus: FixedMultiAirEndpointFixedFoldStateBus,
    pub fixed_value_bus: FixedMultiAirEndpointFixedValueBus,
    pub region: usize,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirEndpointFixedAir {}
impl PartitionedBaseAir<F> for FixedMultiAirEndpointFixedAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirEndpointFixedScheduleCols::<F>::width()]
    }
    fn common_main_width(&self) -> usize {
        FixedMultiAirEndpointFixedCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirEndpointFixedAir {}
impl BaseAir<F> for FixedMultiAirEndpointFixedAir {
    fn width(&self) -> usize {
        FixedMultiAirEndpointFixedScheduleCols::<F>::width()
            + FixedMultiAirEndpointFixedCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirEndpointFixedAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    AB::Expr: PrimeCharacteristicRing,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("fixed endpoint table schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("fixed endpoint table row")
            .to_vec();
        let schedule: &FixedMultiAirEndpointFixedScheduleCols<AB::Var> = cached.as_slice().borrow();
        let local: &FixedMultiAirEndpointFixedCols<AB::Var> = common.as_slice().borrow();
        for flag in schedule.kind_flags {
            builder.assert_bool(flag);
        }
        builder.assert_bool(schedule.selector_uses_point);
        builder.assert_bool(schedule.selector_complement);
        let active = schedule
            .kind_flags
            .iter()
            .fold(AB::Expr::ZERO, |sum, &flag| sum + AB::Expr::from(flag));
        builder.assert_eq(active, schedule.active);
        builder.assert_bool(schedule.active);

        assert_array_eq(
            &mut builder.when(schedule.kind_flags[KIND_INITIAL]),
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
            &mut builder.when(schedule.kind_flags[KIND_FOLD]),
            local.value,
            folded,
        );
        let selector_uses_point = AB::Expr::from(schedule.selector_uses_point);
        let selector_factor = core::array::from_fn(|limb| {
            let point = AB::Expr::from(local.point[limb]);
            if limb == 0 {
                selector_uses_point.clone() * point.clone()
                    + (AB::Expr::ONE - selector_uses_point.clone()) * (AB::Expr::ONE - point)
            } else {
                selector_uses_point.clone() * point.clone()
                    - (AB::Expr::ONE - selector_uses_point.clone()) * point
            }
        });
        assert_array_eq(
            &mut builder.when(schedule.kind_flags[KIND_SELECTOR_STEP]),
            local.value,
            ext_field_multiply::<AB::Expr>(local.low, selector_factor),
        );
        let selector_complement = AB::Expr::from(schedule.selector_complement);
        let complemented = core::array::from_fn(|limb| {
            let one = if limb == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            let low = AB::Expr::from(local.low[limb]);
            selector_complement.clone() * (one - low.clone())
                + (AB::Expr::ONE - selector_complement.clone()) * low
        });
        assert_array_eq(
            &mut builder.when(schedule.kind_flags[KIND_FINAL]),
            local.value,
            complemented,
        );

        self.point_bus.lookup_key(
            builder,
            FixedMultiAirRegionPointMessage {
                region: AB::Expr::from_usize(self.region),
                coordinate: schedule.point_coordinate.into(),
                value: local.point.map(Into::into),
            },
            schedule.kind_flags[KIND_FOLD] + schedule.kind_flags[KIND_SELECTOR_STEP],
        );
        self.state_bus.receive(
            builder,
            FixedMultiAirEndpointFixedFoldStateMessage {
                region: AB::Expr::from_usize(self.region),
                source: schedule.source.into(),
                layer: (AB::Expr::from(schedule.layer) - AB::Expr::ONE),
                index: schedule.low_index.into(),
                value: local.low.map(Into::into),
            },
            schedule.kind_flags[KIND_FOLD],
        );
        self.state_bus.receive(
            builder,
            FixedMultiAirEndpointFixedFoldStateMessage {
                region: AB::Expr::from_usize(self.region),
                source: schedule.source.into(),
                layer: (AB::Expr::from(schedule.layer) - AB::Expr::ONE),
                index: schedule.high_index.into(),
                value: local.high.map(Into::into),
            },
            schedule.kind_flags[KIND_FOLD],
        );
        self.state_bus.receive(
            builder,
            FixedMultiAirEndpointFixedFoldStateMessage {
                region: AB::Expr::from_usize(self.region),
                source: schedule.source.into(),
                layer: (AB::Expr::from(schedule.layer) - AB::Expr::ONE),
                index: AB::Expr::ZERO,
                value: local.low.map(Into::into),
            },
            schedule.kind_flags[KIND_SELECTOR_STEP],
        );
        self.state_bus.send(
            builder,
            FixedMultiAirEndpointFixedFoldStateMessage {
                region: AB::Expr::from_usize(self.region),
                source: schedule.source.into(),
                layer: schedule.layer.into(),
                index: schedule.index.into(),
                value: local.value.map(Into::into),
            },
            AB::Expr::from(schedule.kind_flags[KIND_INITIAL])
                + AB::Expr::from(schedule.kind_flags[KIND_FOLD])
                + AB::Expr::from(schedule.kind_flags[KIND_SELECTOR_STEP]),
        );
        self.state_bus.receive(
            builder,
            FixedMultiAirEndpointFixedFoldStateMessage {
                region: AB::Expr::from_usize(self.region),
                source: schedule.source.into(),
                layer: schedule.layer.into(),
                index: AB::Expr::ZERO,
                value: local.low.map(Into::into),
            },
            schedule.kind_flags[KIND_FINAL],
        );
        self.fixed_value_bus.add_key_with_lookups(
            builder,
            FixedMultiAirEndpointFixedValueMessage {
                region: AB::Expr::from_usize(self.region),
                source: schedule.source.into(),
                value: local.value.map(Into::into),
            },
            schedule.fixed_value_lookup_count,
        );
    }
}

#[derive(Clone, Debug)]
struct ScheduleEntry {
    kind: usize,
    source: usize,
    layer: usize,
    index: usize,
    low_index: usize,
    high_index: usize,
    point_coordinate: usize,
    initial_value: EF,
    fixed_value_lookup_count: usize,
    selector_uses_point: bool,
    selector_complement: bool,
    low: EF,
    high: EF,
    point: EF,
    value: EF,
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirEndpointFixedTraceOutput {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub values: Vec<EF>,
}

pub fn generate_fixed_multi_air_endpoint_fixed_traces(
    relation: &DirectAirPesatIndex<F, Digest>,
    plan: &FixedMultiAirEndpointPlan,
    point: &[EF],
    required_height: Option<usize>,
) -> Result<FixedMultiAirEndpointFixedTraceOutput, FixedMultiAirEndpointFixedTraceError> {
    let height = relation.height();
    if point.len() != plan.log_height || height != 1usize << plan.log_height {
        return Err(FixedMultiAirEndpointFixedTraceError::Shape);
    }
    let fixed_trace = relation.fixed_trace();
    let fixed_width = fixed_trace.map_or(0, |trace| trace.width as usize);
    if plan.fixed.iter().any(|source| source.column >= fixed_width) {
        return Err(FixedMultiAirEndpointFixedTraceError::Relation);
    }
    if let Some(trace) = fixed_trace {
        if trace.values.len() != height.saturating_mul(fixed_width) {
            return Err(FixedMultiAirEndpointFixedTraceError::Relation);
        }
    }
    let mut lookup_counts = vec![0usize; plan.fixed_source_count()];
    for source in plan.node_sources.iter().flatten() {
        if let FixedMultiAirEndpointSource::Fixed(index) = *source {
            *lookup_counts
                .get_mut(index)
                .ok_or(FixedMultiAirEndpointFixedTraceError::Relation)? += 1;
        }
    }
    let mut schedule = Vec::new();
    let mut final_values = Vec::with_capacity(plan.fixed_source_count());
    for source_index in 0..plan.fixed_source_count() {
        if source_index >= plan.fixed.len() {
            let selector_uses_point = source_index != plan.first_source;
            let selector_complement = source_index == plan.transition_source;
            let mut product = EF::ONE;
            schedule.push(ScheduleEntry {
                kind: KIND_INITIAL,
                source: source_index,
                layer: 0,
                index: 0,
                low_index: 0,
                high_index: 0,
                point_coordinate: 0,
                initial_value: EF::ONE,
                fixed_value_lookup_count: 0,
                selector_uses_point,
                selector_complement,
                low: EF::ZERO,
                high: EF::ZERO,
                point: EF::ZERO,
                value: EF::ONE,
            });
            for (coordinate, &challenge) in point.iter().enumerate() {
                let factor = if selector_uses_point {
                    challenge
                } else {
                    EF::ONE - challenge
                };
                let value = product * factor;
                schedule.push(ScheduleEntry {
                    kind: KIND_SELECTOR_STEP,
                    source: source_index,
                    layer: coordinate + 1,
                    index: 0,
                    low_index: 0,
                    high_index: 0,
                    point_coordinate: coordinate,
                    initial_value: EF::ZERO,
                    fixed_value_lookup_count: 0,
                    selector_uses_point,
                    selector_complement,
                    low: product,
                    high: EF::ZERO,
                    point: challenge,
                    value,
                });
                product = value;
            }
            let value = if selector_complement {
                EF::ONE - product
            } else {
                product
            };
            final_values.push(value);
            schedule.push(ScheduleEntry {
                kind: KIND_FINAL,
                source: source_index,
                layer: plan.log_height,
                index: 0,
                low_index: 0,
                high_index: 0,
                point_coordinate: 0,
                initial_value: EF::ZERO,
                fixed_value_lookup_count: lookup_counts[source_index],
                selector_uses_point,
                selector_complement,
                low: product,
                high: EF::ZERO,
                point: EF::ZERO,
                value,
            });
            continue;
        }
        let mut layer_values = initial_table(relation, plan, source_index)?;
        for (index, &value) in layer_values.iter().enumerate() {
            schedule.push(ScheduleEntry {
                kind: KIND_INITIAL,
                source: source_index,
                layer: 0,
                index,
                low_index: 0,
                high_index: 0,
                point_coordinate: 0,
                initial_value: value,
                fixed_value_lookup_count: 0,
                selector_uses_point: false,
                selector_complement: false,
                low: EF::ZERO,
                high: EF::ZERO,
                point: EF::ZERO,
                value,
            });
        }
        for (coordinate, &challenge) in point.iter().enumerate() {
            let half = layer_values.len() / 2;
            if half == 0 {
                return Err(FixedMultiAirEndpointFixedTraceError::Relation);
            }
            let mut next = Vec::with_capacity(half);
            for index in 0..half {
                let low = layer_values[index];
                let high = layer_values[index + half];
                let value = low + challenge * (high - low);
                next.push(value);
                schedule.push(ScheduleEntry {
                    kind: KIND_FOLD,
                    source: source_index,
                    layer: coordinate + 1,
                    index,
                    low_index: index,
                    high_index: index + half,
                    point_coordinate: coordinate,
                    initial_value: EF::ZERO,
                    fixed_value_lookup_count: 0,
                    selector_uses_point: false,
                    selector_complement: false,
                    low,
                    high,
                    point: challenge,
                    value,
                });
            }
            layer_values = next;
        }
        if layer_values.len() != 1 {
            return Err(FixedMultiAirEndpointFixedTraceError::Relation);
        }
        let value = layer_values[0];
        final_values.push(value);
        schedule.push(ScheduleEntry {
            kind: KIND_FINAL,
            source: source_index,
            layer: plan.log_height,
            index: 0,
            low_index: 0,
            high_index: 0,
            point_coordinate: 0,
            initial_value: EF::ZERO,
            fixed_value_lookup_count: lookup_counts[source_index],
            selector_uses_point: false,
            selector_complement: false,
            low: value,
            high: EF::ZERO,
            point: EF::ZERO,
            value,
        });
    }
    let trace_height = required_height.unwrap_or_else(|| schedule.len().next_power_of_two());
    if trace_height < schedule.len() {
        return Err(FixedMultiAirEndpointFixedTraceError::Shape);
    }
    let cached_width = FixedMultiAirEndpointFixedScheduleCols::<F>::width();
    let common_width = FixedMultiAirEndpointFixedCols::<F>::width();
    let mut cached = F::zero_vec(trace_height * cached_width);
    let mut common = F::zero_vec(trace_height * common_width);
    for (row, entry) in schedule.iter().enumerate() {
        let cached_row = &mut cached[row * cached_width..(row + 1) * cached_width];
        let schedule_cols: &mut FixedMultiAirEndpointFixedScheduleCols<F> = cached_row.borrow_mut();
        schedule_cols.active = F::ONE;
        schedule_cols.kind_flags[entry.kind] = F::ONE;
        schedule_cols.source = F::from_usize(entry.source);
        schedule_cols.layer = F::from_usize(entry.layer);
        schedule_cols.index = F::from_usize(entry.index);
        schedule_cols.low_index = F::from_usize(entry.low_index);
        schedule_cols.high_index = F::from_usize(entry.high_index);
        schedule_cols.point_coordinate = F::from_usize(entry.point_coordinate);
        copy_ext(&mut schedule_cols.initial_value, entry.initial_value);
        schedule_cols.fixed_value_lookup_count = F::from_usize(entry.fixed_value_lookup_count);
        schedule_cols.selector_uses_point = F::from_bool(entry.selector_uses_point);
        schedule_cols.selector_complement = F::from_bool(entry.selector_complement);
        let common_row = &mut common[row * common_width..(row + 1) * common_width];
        let cols: &mut FixedMultiAirEndpointFixedCols<F> = common_row.borrow_mut();
        copy_ext(&mut cols.low, entry.low);
        copy_ext(&mut cols.high, entry.high);
        copy_ext(&mut cols.point, entry.point);
        copy_ext(&mut cols.value, entry.value);
    }
    Ok(FixedMultiAirEndpointFixedTraceOutput {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        values: final_values,
    })
}

fn initial_table(
    relation: &DirectAirPesatIndex<F, Digest>,
    plan: &FixedMultiAirEndpointPlan,
    source: usize,
) -> Result<Vec<EF>, FixedMultiAirEndpointFixedTraceError> {
    let height = relation.height();
    if source < plan.fixed.len() {
        let descriptor = plan.fixed[source];
        let trace = relation
            .fixed_trace()
            .ok_or(FixedMultiAirEndpointFixedTraceError::Relation)?;
        let width = trace.width as usize;
        let rotation = match descriptor.rotation {
            openvm_stark_backend::native_warp::DirectAirMappedRotation::Current => 0,
            openvm_stark_backend::native_warp::DirectAirMappedRotation::Next => 1,
        };
        return (0..height)
            .map(|row| {
                trace
                    .values
                    .get(((row + rotation) & (height - 1)) * width + descriptor.column)
                    .copied()
                    .map(EF::from)
                    .ok_or(FixedMultiAirEndpointFixedTraceError::Relation)
            })
            .collect();
    }
    if source == plan.first_source {
        Ok((0..height).map(|row| EF::from_bool(row == 0)).collect())
    } else if source == plan.last_source {
        Ok((0..height)
            .map(|row| EF::from_bool(row + 1 == height))
            .collect())
    } else if source == plan.transition_source {
        Ok((0..height)
            .map(|row| EF::from_bool(row + 1 != height))
            .collect())
    } else {
        Err(FixedMultiAirEndpointFixedTraceError::Relation)
    }
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

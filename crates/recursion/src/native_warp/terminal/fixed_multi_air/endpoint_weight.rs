use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder, interaction::InteractionBuilder, BaseAirWithPublicValues,
    PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    native_warp::terminal::fixed_multi_air::{
        FixedMultiAirBetaCoordinateBus, FixedMultiAirBetaCoordinateMessage,
        FixedMultiAirConstraintWeightBus, FixedMultiAirConstraintWeightMessage,
        FixedMultiAirConstraintWeightStateBus, FixedMultiAirConstraintWeightStateMessage,
        FixedMultiAirRegionPointBus, FixedMultiAirRegionPointMessage,
    },
    utils::{ext_field_add, ext_field_multiply, ext_field_subtract},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedMultiAirConstraintWeightTraceError {
    Shape,
    Relation,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirConstraintWeightInitScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub is_step: T,
    pub constraint: T,
    pub initial_carry: T,
    pub carry_before: T,
    pub carry_after: T,
    pub tau_coordinate: T,
    pub carry_bit: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirConstraintWeightInitCols<T> {
    pub tau: [T; D_EF],
    pub weight_before: [T; D_EF],
    pub weight_after: [T; D_EF],
}

/// Initial high-bit carry products for the backend flattened-weight DP.
pub struct FixedMultiAirConstraintWeightInitAir {
    pub beta_bus: FixedMultiAirBetaCoordinateBus,
    pub state_bus: FixedMultiAirConstraintWeightStateBus,
    pub weight_bus: FixedMultiAirConstraintWeightBus,
    pub region: usize,
    /// A height-one region has no row-variable DP rounds.  Its high-bit
    /// product is already the final flattened constraint weight, so the init
    /// AIR must publish it directly instead of leaving an unmatched state.
    pub zero_log_height: bool,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirConstraintWeightInitAir {}
impl PartitionedBaseAir<F> for FixedMultiAirConstraintWeightInitAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirConstraintWeightInitScheduleCols::<F>::width()]
    }
    fn common_main_width(&self) -> usize {
        FixedMultiAirConstraintWeightInitCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirConstraintWeightInitAir {}
impl BaseAir<F> for FixedMultiAirConstraintWeightInitAir {
    fn width(&self) -> usize {
        FixedMultiAirConstraintWeightInitScheduleCols::<F>::width()
            + FixedMultiAirConstraintWeightInitCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirConstraintWeightInitAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("constraint-weight init schedule")
            .to_vec();
        let next_cached = builder.cached_mains()[0]
            .row_slice(1)
            .expect("next constraint-weight init schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("constraint-weight init row")
            .to_vec();
        let next_common = builder
            .common_main()
            .row_slice(1)
            .expect("next constraint-weight init row")
            .to_vec();
        let schedule: &FixedMultiAirConstraintWeightInitScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let next_schedule: &FixedMultiAirConstraintWeightInitScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FixedMultiAirConstraintWeightInitCols<AB::Var> = common.as_slice().borrow();
        let next: &FixedMultiAirConstraintWeightInitCols<AB::Var> = next_common.as_slice().borrow();
        for flag in [
            schedule.active,
            schedule.is_first,
            schedule.is_last,
            schedule.is_step,
            schedule.carry_bit,
        ] {
            builder.assert_bool(flag);
        }
        builder
            .when(schedule.is_first)
            .assert_eq(local.weight_before[0], AB::Expr::ONE);
        for limb in 1..D_EF {
            builder
                .when(schedule.is_first)
                .assert_zero(local.weight_before[limb]);
        }
        let factor = core::array::from_fn(|limb| {
            if limb == 0 {
                schedule.carry_bit * local.tau[limb]
                    + (AB::Expr::ONE - schedule.carry_bit) * (AB::Expr::ONE - local.tau[limb])
            } else {
                schedule.carry_bit * local.tau[limb]
                    - (AB::Expr::ONE - schedule.carry_bit) * local.tau[limb]
            }
        });
        let stepped = ext_field_multiply::<AB::Expr>(local.weight_before, factor);
        let expected = core::array::from_fn(|limb| {
            schedule.is_step * stepped[limb].clone()
                + (AB::Expr::ONE - schedule.is_step) * local.weight_before[limb]
        });
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.weight_after,
            expected,
        );
        builder.when(schedule.is_step).assert_eq(
            schedule.carry_before,
            schedule.carry_after * AB::Expr::TWO + schedule.carry_bit,
        );
        builder
            .when(schedule.is_last)
            .assert_zero(schedule.carry_after);
        let continue_same = next_schedule.active * (AB::Expr::ONE - next_schedule.is_first);
        let mut transition = builder.when_transition();
        let mut same = transition.when(continue_same);
        same.assert_eq(next_schedule.initial_carry, schedule.initial_carry);
        same.assert_eq(next_schedule.carry_before, schedule.carry_after);
        assert_array_eq(&mut same, next.weight_before, local.weight_after);

        self.beta_bus.lookup_key(
            builder,
            FixedMultiAirBetaCoordinateMessage {
                coordinate: schedule.tau_coordinate.into(),
                value: local.tau.map(Into::into),
            },
            schedule.is_step,
        );
        if self.zero_log_height {
            self.weight_bus.add_key_with_lookups(
                builder,
                FixedMultiAirConstraintWeightMessage {
                    region: AB::Expr::from_usize(self.region),
                    constraint: schedule.constraint.into(),
                    value: local.weight_after.map(Into::into),
                },
                schedule.is_last,
            );
        } else {
            self.state_bus.send(
                builder,
                FixedMultiAirConstraintWeightStateMessage {
                    region: AB::Expr::from_usize(self.region),
                    layer: AB::Expr::ZERO,
                    carry: schedule.constraint.into(),
                    value: local.weight_after.map(Into::into),
                },
                schedule.is_last,
            );
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirConstraintWeightDpScheduleCols<T> {
    pub active: T,
    pub layer: T,
    pub carry: T,
    pub previous0: T,
    pub previous1: T,
    pub output_bit0: T,
    pub output_bit1: T,
    pub point_coordinate: T,
    pub tau_coordinate: T,
    pub is_final_layer: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirConstraintWeightDpCols<T> {
    pub previous_value0: [T; D_EF],
    pub previous_value1: [T; D_EF],
    pub point: [T; D_EF],
    pub tau: [T; D_EF],
    pub value: [T; D_EF],
}

pub struct FixedMultiAirConstraintWeightDpAir {
    pub beta_bus: FixedMultiAirBetaCoordinateBus,
    pub point_bus: FixedMultiAirRegionPointBus,
    pub state_bus: FixedMultiAirConstraintWeightStateBus,
    pub weight_bus: FixedMultiAirConstraintWeightBus,
    pub region: usize,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirConstraintWeightDpAir {}
impl PartitionedBaseAir<F> for FixedMultiAirConstraintWeightDpAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirConstraintWeightDpScheduleCols::<F>::width()]
    }
    fn common_main_width(&self) -> usize {
        FixedMultiAirConstraintWeightDpCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirConstraintWeightDpAir {}
impl BaseAir<F> for FixedMultiAirConstraintWeightDpAir {
    fn width(&self) -> usize {
        FixedMultiAirConstraintWeightDpScheduleCols::<F>::width()
            + FixedMultiAirConstraintWeightDpCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirConstraintWeightDpAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("constraint-weight DP schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("constraint-weight DP row")
            .to_vec();
        let schedule: &FixedMultiAirConstraintWeightDpScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let local: &FixedMultiAirConstraintWeightDpCols<AB::Var> = common.as_slice().borrow();
        for flag in [
            schedule.active,
            schedule.output_bit0,
            schedule.output_bit1,
            schedule.is_final_layer,
        ] {
            builder.assert_bool(flag);
        }
        let one = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            }
        });
        let equality = ext_field_add::<AB::Expr>(
            ext_field_multiply::<AB::Expr>(
                ext_field_subtract::<AB::Expr>(one.clone(), local.point),
                ext_field_subtract::<AB::Expr>(one, local.tau),
            ),
            ext_field_multiply::<AB::Expr>(local.point, local.tau),
        );
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.value,
            ext_field_multiply::<AB::Expr>(local.previous_value0, equality),
        );
        self.state_bus.receive(
            builder,
            FixedMultiAirConstraintWeightStateMessage {
                region: AB::Expr::from_usize(self.region),
                layer: AB::Expr::from(schedule.layer) - AB::Expr::ONE,
                carry: schedule.previous0.into(),
                value: local.previous_value0.map(Into::into),
            },
            schedule.active,
        );
        self.state_bus.send(
            builder,
            FixedMultiAirConstraintWeightStateMessage {
                region: AB::Expr::from_usize(self.region),
                layer: schedule.layer.into(),
                carry: schedule.carry.into(),
                value: local.value.map(Into::into),
            },
            schedule.active * (AB::Expr::ONE - schedule.is_final_layer),
        );
        self.beta_bus.lookup_key(
            builder,
            FixedMultiAirBetaCoordinateMessage {
                coordinate: schedule.tau_coordinate.into(),
                value: local.tau.map(Into::into),
            },
            schedule.active,
        );
        self.point_bus.lookup_key(
            builder,
            FixedMultiAirRegionPointMessage {
                region: AB::Expr::from_usize(self.region),
                coordinate: schedule.point_coordinate.into(),
                value: local.point.map(Into::into),
            },
            schedule.active,
        );
        self.weight_bus.add_key_with_lookups(
            builder,
            FixedMultiAirConstraintWeightMessage {
                region: AB::Expr::from_usize(self.region),
                constraint: schedule.carry.into(),
                value: local.value.map(Into::into),
            },
            schedule.is_final_layer,
        );
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirConstraintWeightTraceOutput {
    pub init_cached: RowMajorMatrix<F>,
    pub init_common: RowMajorMatrix<F>,
    pub dp_cached: RowMajorMatrix<F>,
    pub dp_common: RowMajorMatrix<F>,
    pub weights: Vec<EF>,
}

pub fn generate_fixed_multi_air_constraint_weight_traces(
    tau: &[EF],
    point: &[EF],
    constraints_per_row: usize,
    constraint_offset: u64,
    init_required_height: Option<usize>,
    dp_required_height: Option<usize>,
) -> Result<FixedMultiAirConstraintWeightTraceOutput, FixedMultiAirConstraintWeightTraceError> {
    if constraints_per_row == 0 || tau.len() < point.len() || tau.len() >= u64::BITS as usize {
        return Err(FixedMultiAirConstraintWeightTraceError::Shape);
    }
    let log_height = point.len();
    let high_steps = tau.len() - log_height;
    let height = 1u64
        .checked_shl(log_height as u32)
        .ok_or(FixedMultiAirConstraintWeightTraceError::Relation)?;
    if !constraint_offset.is_multiple_of(height) {
        return Err(FixedMultiAirConstraintWeightTraceError::Relation);
    }
    let constraint_base = constraint_offset / height;
    let mut init_records = Vec::new();
    let mut suffix = EF::zero_vec(constraints_per_row);
    for constraint in 0..constraints_per_row {
        let initial_carry = constraint_base
            .checked_add(constraint as u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or(FixedMultiAirConstraintWeightTraceError::Relation)?;
        let mut carry = initial_carry;
        let mut weight = EF::ONE;
        if high_steps == 0 {
            init_records.push((
                constraint,
                initial_carry,
                carry,
                carry,
                0usize,
                0usize,
                true,
                true,
                false,
                EF::ZERO,
                EF::ONE,
                EF::ONE,
            ));
        } else {
            for (step, output_bit_from_lsb) in (log_height..tau.len()).enumerate() {
                let coordinate = tau.len() - 1 - output_bit_from_lsb;
                let bit = carry & 1;
                let next_carry = carry >> 1;
                let before = weight;
                let tau_value = tau[coordinate];
                weight *= if bit == 0 {
                    EF::ONE - tau_value
                } else {
                    tau_value
                };
                init_records.push((
                    constraint,
                    initial_carry,
                    carry,
                    next_carry,
                    coordinate,
                    bit,
                    step == 0,
                    step + 1 == high_steps,
                    true,
                    tau_value,
                    before,
                    weight,
                ));
                carry = next_carry;
            }
        }
        if carry != 0 {
            return Err(FixedMultiAirConstraintWeightTraceError::Relation);
        }
        suffix[constraint] = weight;
    }
    let init_height =
        init_required_height.unwrap_or_else(|| init_records.len().next_power_of_two());
    if init_height < init_records.len() {
        return Err(FixedMultiAirConstraintWeightTraceError::Shape);
    }
    let init_cached_width = FixedMultiAirConstraintWeightInitScheduleCols::<F>::width();
    let init_common_width = FixedMultiAirConstraintWeightInitCols::<F>::width();
    let mut init_cached = F::zero_vec(init_height * init_cached_width);
    let mut init_common = F::zero_vec(init_height * init_common_width);
    for (
        row,
        &(
            initial,
            initial_carry,
            before_carry,
            after_carry,
            coordinate,
            bit,
            first,
            last,
            is_step,
            tau_value,
            before,
            after,
        ),
    ) in init_records.iter().enumerate()
    {
        let cached_row = &mut init_cached[row * init_cached_width..(row + 1) * init_cached_width];
        let schedule: &mut FixedMultiAirConstraintWeightInitScheduleCols<F> =
            cached_row.borrow_mut();
        schedule.active = F::ONE;
        schedule.is_first = F::from_bool(first);
        schedule.is_last = F::from_bool(last);
        schedule.is_step = F::from_bool(is_step);
        schedule.constraint = F::from_usize(initial);
        schedule.initial_carry = F::from_usize(initial_carry);
        schedule.carry_before = F::from_usize(before_carry);
        schedule.carry_after = F::from_usize(after_carry);
        schedule.tau_coordinate = F::from_usize(coordinate);
        schedule.carry_bit = F::from_usize(bit);
        let common_row = &mut init_common[row * init_common_width..(row + 1) * init_common_width];
        let cols: &mut FixedMultiAirConstraintWeightInitCols<F> = common_row.borrow_mut();
        copy_ext(&mut cols.tau, tau_value);
        copy_ext(&mut cols.weight_before, before);
        copy_ext(&mut cols.weight_after, after);
    }

    let mut dp_records = Vec::with_capacity(log_height.saturating_mul(constraints_per_row));
    for layer in 1..=log_height {
        let row_coordinate = layer - 1;
        let tau_coordinate = tau.len() - log_height + row_coordinate;
        let row_value = point[row_coordinate];
        let tau_value = tau[tau_coordinate];
        let mut values = EF::zero_vec(constraints_per_row);
        for constraint in 0..constraints_per_row {
            let equality = (EF::ONE - row_value) * (EF::ONE - tau_value) + row_value * tau_value;
            let value = suffix[constraint] * equality;
            values[constraint] = value;
            dp_records.push((
                layer,
                constraint,
                constraint,
                constraint,
                0,
                0,
                row_coordinate,
                tau_coordinate,
                row_value,
                tau_value,
                suffix[constraint],
                suffix[constraint],
                value,
            ));
        }
        suffix = values;
    }
    let dp_height = dp_required_height.unwrap_or_else(|| dp_records.len().next_power_of_two());
    if dp_height < dp_records.len() {
        return Err(FixedMultiAirConstraintWeightTraceError::Shape);
    }
    let dp_cached_width = FixedMultiAirConstraintWeightDpScheduleCols::<F>::width();
    let dp_common_width = FixedMultiAirConstraintWeightDpCols::<F>::width();
    let mut dp_cached = F::zero_vec(dp_height * dp_cached_width);
    let mut dp_common = F::zero_vec(dp_height * dp_common_width);
    for (
        row,
        &(
            layer,
            carry,
            prev0,
            prev1,
            out0,
            out1,
            point_coordinate,
            tau_coordinate,
            point_value,
            tau_value,
            previous0,
            previous1,
            value,
        ),
    ) in dp_records.iter().enumerate()
    {
        let cached_row = &mut dp_cached[row * dp_cached_width..(row + 1) * dp_cached_width];
        let schedule: &mut FixedMultiAirConstraintWeightDpScheduleCols<F> = cached_row.borrow_mut();
        schedule.active = F::ONE;
        schedule.layer = F::from_usize(layer);
        schedule.carry = F::from_usize(carry);
        schedule.previous0 = F::from_usize(prev0);
        schedule.previous1 = F::from_usize(prev1);
        schedule.output_bit0 = F::from_usize(out0);
        schedule.output_bit1 = F::from_usize(out1);
        schedule.point_coordinate = F::from_usize(point_coordinate);
        schedule.tau_coordinate = F::from_usize(tau_coordinate);
        schedule.is_final_layer = F::from_bool(layer == log_height);
        let common_row = &mut dp_common[row * dp_common_width..(row + 1) * dp_common_width];
        let cols: &mut FixedMultiAirConstraintWeightDpCols<F> = common_row.borrow_mut();
        copy_ext(&mut cols.previous_value0, previous0);
        copy_ext(&mut cols.previous_value1, previous1);
        copy_ext(&mut cols.point, point_value);
        copy_ext(&mut cols.tau, tau_value);
        copy_ext(&mut cols.value, value);
    }
    Ok(FixedMultiAirConstraintWeightTraceOutput {
        init_cached: RowMajorMatrix::new(init_cached, init_cached_width),
        init_common: RowMajorMatrix::new(init_common, init_common_width),
        dp_cached: RowMajorMatrix::new(dp_cached, dp_cached_width),
        dp_common: RowMajorMatrix::new(dp_common, dp_common_width),
        weights: suffix,
    })
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

#[cfg(test)]
mod tests {
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    fn eq_at_index(point: &[EF], index: usize) -> EF {
        point
            .iter()
            .enumerate()
            .fold(EF::ONE, |weight, (coordinate, &x)| {
                weight
                    * if (index >> (point.len() - 1 - coordinate)) & 1 == 0 {
                        EF::ONE - x
                    } else {
                        x
                    }
            })
    }

    #[test]
    fn height_one_region_publishes_high_bit_products_as_final_weights() {
        let tau = [2, 3, 5, 7]
            .into_iter()
            .map(EF::from_u32)
            .collect::<Vec<_>>();
        let constraint_offset = 8u64;
        let constraints_per_row = 3;
        let generated = generate_fixed_multi_air_constraint_weight_traces(
            &tau,
            &[],
            constraints_per_row,
            constraint_offset,
            None,
            None,
        )
        .expect("height-one regional weights");

        assert_eq!(
            generated.weights,
            (0..constraints_per_row)
                .map(|index| eq_at_index(&tau, constraint_offset as usize + index))
                .collect::<Vec<_>>()
        );
        assert_eq!(generated.dp_cached.height(), 1);
        assert_eq!(generated.dp_common.height(), 1);
        assert!(generated
            .dp_cached
            .values
            .iter()
            .all(|&value| value == F::ZERO));
        assert!(generated
            .dp_common
            .values
            .iter()
            .all(|&value| value == F::ZERO));
    }
}

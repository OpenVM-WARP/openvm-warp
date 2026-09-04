use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder, interaction::InteractionBuilder,
    native_warp::FixedMultiAirPesatIndex, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    native_warp::terminal::fixed_multi_air::{
        FixedMultiAirBetaCoordinateBus, FixedMultiAirBetaCoordinateMessage,
        FixedMultiAirDecompositionContributionBus, FixedMultiAirDecompositionContributionMessage,
        FixedMultiAirGlobalClaimBus, FixedMultiAirGlobalClaimMessage, FixedMultiAirPaddingClaimBus,
        FixedMultiAirPaddingClaimMessage, FixedMultiAirRegionClaimBus,
        FixedMultiAirRegionClaimMessage,
    },
    utils::ext_field_multiply,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedMultiAirDecompositionTraceError {
    Shape,
    Claim,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedMultiAirDecompositionComponentKind {
    Region { region: usize },
    Padding,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedMultiAirDecompositionComponentPlan {
    pub component: usize,
    pub kind: FixedMultiAirDecompositionComponentKind,
    pub prefix_len: usize,
    pub prefix_tag: u64,
    pub degree_delta: usize,
}

impl FixedMultiAirDecompositionComponentPlan {
    pub fn from_relation(
        relation: &FixedMultiAirPesatIndex<F, Digest>,
    ) -> Result<Vec<Self>, FixedMultiAirDecompositionTraceError> {
        let description = relation.description();
        let mut plans = Vec::with_capacity(
            relation.region_count() + usize::from(description.padding_constraint_count != 0),
        );
        for (region, region_description) in description.regions.iter().enumerate() {
            let height = 1u64
                .checked_shl(region_description.log_height as u32)
                .ok_or(FixedMultiAirDecompositionTraceError::Shape)?;
            let expected_count = height
                .checked_mul(region_description.constraints_per_row as u64)
                .ok_or(FixedMultiAirDecompositionTraceError::Shape)?;
            if region_description.constraints_per_row == 0
                || region_description.constraint_count != expected_count
                || !region_description.constraint_offset.is_multiple_of(height)
                || region_description.exact_max_degree > description.exact_max_degree
            {
                return Err(FixedMultiAirDecompositionTraceError::Shape);
            }
            // `FixedMultiAirTerminalLinearizer::region_claim` already uses
            // global `eq(tau, offset + constraint * height + row)` weights and
            // applies degree homogenization. The transcript-bound regional
            // claim is therefore a complete global contribution, not the old
            // local dyadic-block claim that required prefix/degree scaling in
            // this AIR.
            plans.push(Self {
                component: plans.len(),
                kind: FixedMultiAirDecompositionComponentKind::Region { region },
                prefix_len: 0,
                prefix_tag: 0,
                degree_delta: 0,
            });
        }
        if description.padding_constraint_count != 0 {
            if description.padding_constraint_offset
                != description
                    .regions
                    .last()
                    .and_then(|region| {
                        region
                            .constraint_offset
                            .checked_add(region.constraint_count)
                    })
                    .ok_or(FixedMultiAirDecompositionTraceError::Shape)?
                || description
                    .padding_constraint_offset
                    .checked_add(description.padding_constraint_count)
                    != Some(description.real_constraint_count)
            {
                return Err(FixedMultiAirDecompositionTraceError::Shape);
            }
            // The padding reduction likewise proves its globally weighted,
            // already-homogenized contribution directly.
            plans.push(Self {
                component: plans.len(),
                kind: FixedMultiAirDecompositionComponentKind::Padding,
                prefix_len: 0,
                prefix_tag: 0,
                degree_delta: 0,
            });
        }
        Ok(plans)
    }

    #[must_use]
    pub fn valid_rows(&self) -> usize {
        self.prefix_len + self.degree_delta + 1
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirDecompositionScheduleCols<T> {
    pub active: T,
    pub is_prefix: T,
    pub is_degree_scale: T,
    pub is_final: T,
    pub is_first: T,
    pub step: T,
    pub beta_coordinate: T,
    pub prefix_bit: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirDecompositionCols<T> {
    pub claim: [T; D_EF],
    pub eta: [T; D_EF],
    pub one: [T; D_EF],
    pub beta_value: [T; D_EF],
    pub state_before: [T; D_EF],
    pub state_after: [T; D_EF],
}

/// One setup-fixed, already globally weighted region or padding contribution
/// to the global terminal eta. Compact-layout regional and padding sumchecks
/// include all prefix equality weights and homogenization before emitting the
/// claims consumed here.
pub struct FixedMultiAirDecompositionComponentAir {
    pub beta_bus: FixedMultiAirBetaCoordinateBus,
    pub global_bus: FixedMultiAirGlobalClaimBus,
    pub region_claim_bus: FixedMultiAirRegionClaimBus,
    pub padding_claim_bus: FixedMultiAirPaddingClaimBus,
    pub contribution_bus: FixedMultiAirDecompositionContributionBus,
    pub plan: FixedMultiAirDecompositionComponentPlan,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirDecompositionComponentAir {}
impl PartitionedBaseAir<F> for FixedMultiAirDecompositionComponentAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirDecompositionScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirDecompositionCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirDecompositionComponentAir {}
impl BaseAir<F> for FixedMultiAirDecompositionComponentAir {
    fn width(&self) -> usize {
        FixedMultiAirDecompositionScheduleCols::<F>::width()
            + FixedMultiAirDecompositionCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirDecompositionComponentAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let schedule_row = builder.cached_mains()[0]
            .row_slice(0)
            .expect("fixed decomposition schedule row")
            .to_vec();
        let next_schedule_row = builder.cached_mains()[0]
            .row_slice(1)
            .expect("fixed next decomposition schedule row")
            .to_vec();
        let common_row = builder
            .common_main()
            .row_slice(0)
            .expect("fixed decomposition row")
            .to_vec();
        let next_common_row = builder
            .common_main()
            .row_slice(1)
            .expect("fixed next decomposition row")
            .to_vec();
        let schedule: &FixedMultiAirDecompositionScheduleCols<AB::Var> =
            schedule_row.as_slice().borrow();
        let next_schedule: &FixedMultiAirDecompositionScheduleCols<AB::Var> =
            next_schedule_row.as_slice().borrow();
        let local: &FixedMultiAirDecompositionCols<AB::Var> = common_row.as_slice().borrow();
        let next: &FixedMultiAirDecompositionCols<AB::Var> = next_common_row.as_slice().borrow();

        for flag in [
            schedule.active,
            schedule.is_prefix,
            schedule.is_degree_scale,
            schedule.is_final,
            schedule.is_first,
            schedule.prefix_bit,
        ] {
            builder.assert_bool(flag);
        }
        builder.assert_eq(
            schedule.is_prefix + schedule.is_degree_scale + schedule.is_final,
            schedule.active,
        );
        builder.when_first_row().assert_one(schedule.active);
        builder.when_first_row().assert_one(schedule.is_first);
        builder.when_first_row().assert_zero(schedule.step);
        builder
            .when_transition()
            .assert_bool(schedule.active - next_schedule.active);
        builder
            .when_transition()
            .assert_eq(schedule.active - next_schedule.active, schedule.is_final);
        builder
            .when_transition()
            .when(next_schedule.active)
            .assert_eq(next_schedule.step, schedule.step + AB::F::ONE);
        builder
            .when_last_row()
            .when(schedule.active)
            .assert_one(schedule.is_final);

        let factor = core::array::from_fn(|limb| {
            let value = AB::Expr::from(local.beta_value[limb]);
            if limb == 0 {
                schedule.prefix_bit * value.clone()
                    + (AB::Expr::ONE - schedule.prefix_bit) * (AB::Expr::ONE - value)
            } else {
                schedule.prefix_bit * value.clone() - (AB::Expr::ONE - schedule.prefix_bit) * value
            }
        });
        let prefixed = ext_field_multiply::<AB::Expr>(local.state_before, factor);
        let scaled = ext_field_multiply::<AB::Expr>(local.state_before, local.one);
        let expected_after = core::array::from_fn(|limb| {
            AB::Expr::from(schedule.is_prefix) * prefixed[limb].clone()
                + AB::Expr::from(schedule.is_degree_scale) * scaled[limb].clone()
                + AB::Expr::from(schedule.is_final) * AB::Expr::from(local.state_before[limb])
        });
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.state_after,
            expected_after,
        );
        assert_array_eq(
            &mut builder.when(schedule.is_first),
            local.state_before,
            local.claim.map(Into::into),
        );
        let mut transition = builder.when_transition();
        let mut active_next = transition.when(next_schedule.active);
        assert_array_eq(&mut active_next, next.claim, local.claim);
        assert_array_eq(&mut active_next, next.eta, local.eta);
        assert_array_eq(&mut active_next, next.one, local.one);
        assert_array_eq(&mut active_next, next.state_before, local.state_after);

        self.global_bus.lookup_key(
            builder,
            FixedMultiAirGlobalClaimMessage {
                eta: local.eta.map(Into::into),
                one: local.one.map(Into::into),
            },
            schedule.is_first,
        );
        match self.plan.kind {
            FixedMultiAirDecompositionComponentKind::Region { region } => {
                self.region_claim_bus.lookup_key(
                    builder,
                    FixedMultiAirRegionClaimMessage {
                        region: AB::Expr::from_usize(region),
                        claim: local.claim.map(Into::into),
                    },
                    schedule.is_first,
                );
            }
            FixedMultiAirDecompositionComponentKind::Padding => {
                self.padding_claim_bus.lookup_key(
                    builder,
                    FixedMultiAirPaddingClaimMessage {
                        present: AB::Expr::ONE,
                        claim: local.claim.map(Into::into),
                    },
                    schedule.is_first,
                );
            }
        }
        self.beta_bus.lookup_key(
            builder,
            FixedMultiAirBetaCoordinateMessage {
                coordinate: schedule.beta_coordinate.into(),
                value: local.beta_value.map(Into::into),
            },
            schedule.is_prefix,
        );
        self.contribution_bus.send(
            builder,
            FixedMultiAirDecompositionContributionMessage {
                component: AB::Expr::from_usize(self.plan.component),
                value: local.state_after.map(Into::into),
            },
            schedule.is_final,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirDecompositionSumCols<T> {
    pub active: T,
    pub component: T,
    pub is_first: T,
    pub is_last: T,
    pub contribution: [T; D_EF],
    pub sum_before: [T; D_EF],
    pub sum_after: [T; D_EF],
    pub eta: [T; D_EF],
    pub one: [T; D_EF],
}

/// Adds all setup-fixed regional contributions and constrains the result to
/// the complete final accumulator `eta`.
#[derive(ColumnsAir)]
#[columns_via(FixedMultiAirDecompositionSumCols<u8>)]
pub struct FixedMultiAirDecompositionSumAir {
    pub global_bus: FixedMultiAirGlobalClaimBus,
    pub contribution_bus: FixedMultiAirDecompositionContributionBus,
    pub component_count: usize,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirDecompositionSumAir {}
impl PartitionedBaseAir<F> for FixedMultiAirDecompositionSumAir {}
impl BaseAir<F> for FixedMultiAirDecompositionSumAir {
    fn width(&self) -> usize {
        FixedMultiAirDecompositionSumCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for FixedMultiAirDecompositionSumAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("fixed decomposition sum row");
        let next_row = main.row_slice(1).expect("fixed next decomposition sum row");
        let local: &FixedMultiAirDecompositionSumCols<AB::Var> = (*row).borrow();
        let next: &FixedMultiAirDecompositionSumCols<AB::Var> = (*next_row).borrow();
        for flag in [local.active, local.is_first, local.is_last] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when_first_row().assert_zero(local.component);
        builder.when(local.active * local.is_last).assert_eq(
            local.component,
            AB::Expr::from_usize(self.component_count - 1),
        );
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.is_last);
        let mut transition = builder.when_transition();
        let mut same = transition.when(next.active);
        same.assert_eq(next.component, local.component + AB::F::ONE);
        same.assert_zero(next.is_first);
        assert_array_eq(&mut same, next.sum_before, local.sum_after);
        assert_array_eq(&mut same, next.eta, local.eta);
        assert_array_eq(&mut same, next.one, local.one);
        builder
            .when_last_row()
            .assert_eq(local.is_last, local.active);
        assert_array_eq(
            &mut builder.when(local.is_first),
            local.sum_before,
            [AB::Expr::ZERO; D_EF],
        );
        let expected_after = core::array::from_fn(|limb| {
            AB::Expr::from(local.sum_before[limb]) + AB::Expr::from(local.contribution[limb])
        });
        assert_array_eq(
            &mut builder.when(local.active),
            local.sum_after,
            expected_after,
        );
        assert_array_eq(
            &mut builder.when(local.active * local.is_last),
            local.sum_after,
            local.eta.map(Into::into),
        );
        self.global_bus.lookup_key(
            builder,
            FixedMultiAirGlobalClaimMessage {
                eta: local.eta.map(Into::into),
                one: local.one.map(Into::into),
            },
            local.is_first,
        );
        self.contribution_bus.receive(
            builder,
            FixedMultiAirDecompositionContributionMessage {
                component: local.component.into(),
                value: local.contribution.map(Into::into),
            },
            local.active,
        );
    }
}

pub fn generate_fixed_multi_air_decomposition_schedule_trace(
    plan: &FixedMultiAirDecompositionComponentPlan,
    required_height: Option<usize>,
) -> Result<RowMajorMatrix<F>, FixedMultiAirDecompositionTraceError> {
    let valid_rows = plan.valid_rows();
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return Err(FixedMultiAirDecompositionTraceError::Shape);
    }
    let width = FixedMultiAirDecompositionScheduleCols::<F>::width();
    let mut trace = F::zero_vec(height * width);
    for row_index in 0..valid_rows {
        let row = &mut trace[row_index * width..(row_index + 1) * width];
        let cols: &mut FixedMultiAirDecompositionScheduleCols<F> = row.borrow_mut();
        let is_prefix = row_index < plan.prefix_len;
        let is_degree_scale =
            row_index >= plan.prefix_len && row_index < plan.prefix_len + plan.degree_delta;
        let is_final = row_index + 1 == valid_rows;
        cols.active = F::ONE;
        cols.is_prefix = F::from_bool(is_prefix);
        cols.is_degree_scale = F::from_bool(is_degree_scale);
        cols.is_final = F::from_bool(is_final);
        cols.is_first = F::from_bool(row_index == 0);
        cols.step = F::from_usize(row_index);
        if is_prefix {
            cols.beta_coordinate = F::from_usize(row_index);
            let shift = plan.prefix_len - 1 - row_index;
            cols.prefix_bit = F::from_bool((plan.prefix_tag >> shift) & 1 == 1);
        }
    }
    Ok(RowMajorMatrix::new(trace, width))
}

pub fn generate_fixed_multi_air_decomposition_component_trace(
    plan: &FixedMultiAirDecompositionComponentPlan,
    beta: &[EF],
    eta: EF,
    one: EF,
    claim: EF,
    required_height: Option<usize>,
) -> Result<(RowMajorMatrix<F>, EF), FixedMultiAirDecompositionTraceError> {
    if beta.len() < plan.prefix_len {
        return Err(FixedMultiAirDecompositionTraceError::Shape);
    }
    let valid_rows = plan.valid_rows();
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return Err(FixedMultiAirDecompositionTraceError::Shape);
    }
    let width = FixedMultiAirDecompositionCols::<F>::width();
    let mut trace = F::zero_vec(height * width);
    let mut state = claim;
    for row_index in 0..valid_rows {
        let row = &mut trace[row_index * width..(row_index + 1) * width];
        let cols: &mut FixedMultiAirDecompositionCols<F> = row.borrow_mut();
        copy_ext(&mut cols.claim, claim);
        copy_ext(&mut cols.eta, eta);
        copy_ext(&mut cols.one, one);
        copy_ext(&mut cols.state_before, state);
        if row_index < plan.prefix_len {
            let value = beta[row_index];
            copy_ext(&mut cols.beta_value, value);
            let shift = plan.prefix_len - 1 - row_index;
            let bit = (plan.prefix_tag >> shift) & 1;
            state *= if bit == 0 { EF::ONE - value } else { value };
        } else if row_index < plan.prefix_len + plan.degree_delta {
            state *= one;
        }
        copy_ext(&mut cols.state_after, state);
    }
    Ok((RowMajorMatrix::new(trace, width), state))
}

pub fn generate_fixed_multi_air_decomposition_sum_trace(
    contributions: &[EF],
    eta: EF,
    one: EF,
    required_height: Option<usize>,
) -> Result<RowMajorMatrix<F>, FixedMultiAirDecompositionTraceError> {
    if contributions.is_empty() {
        return Err(FixedMultiAirDecompositionTraceError::Shape);
    }
    let height = required_height.unwrap_or_else(|| contributions.len().next_power_of_two());
    if height < contributions.len() {
        return Err(FixedMultiAirDecompositionTraceError::Shape);
    }
    let width = FixedMultiAirDecompositionSumCols::<F>::width();
    let mut trace = F::zero_vec(height * width);
    let mut sum = EF::ZERO;
    for (component, &contribution) in contributions.iter().enumerate() {
        let row = &mut trace[component * width..(component + 1) * width];
        let cols: &mut FixedMultiAirDecompositionSumCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.component = F::from_usize(component);
        cols.is_first = F::from_bool(component == 0);
        cols.is_last = F::from_bool(component + 1 == contributions.len());
        copy_ext(&mut cols.contribution, contribution);
        copy_ext(&mut cols.sum_before, sum);
        sum += contribution;
        copy_ext(&mut cols.sum_after, sum);
        copy_ext(&mut cols.eta, eta);
        copy_ext(&mut cols.one, one);
    }
    if sum != eta {
        return Err(FixedMultiAirDecompositionTraceError::Claim);
    }
    Ok(RowMajorMatrix::new(trace, width))
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

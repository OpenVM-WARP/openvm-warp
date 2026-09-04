use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder, interaction::InteractionBuilder,
    native_warp::DirectAirMappedRotation, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    native_warp::terminal::fixed_multi_air::{
        FixedMultiAirEndpointPlan, FixedMultiAirMappedTermBus, FixedMultiAirMappedTermMessage,
        FixedMultiAirRegionOpeningBus, FixedMultiAirRegionOpeningMessage,
        FixedMultiAirRegionPointBus, FixedMultiAirRegionPointMessage, FixedMultiAirRegionRhoBus,
        FixedMultiAirRegionRhoMessage, FixedMultiAirStructuredClaimHeaderBus,
        FixedMultiAirStructuredClaimHeaderMessage, FixedMultiAirStructuredPointBus,
        FixedMultiAirStructuredPointMessage,
    },
    utils::ext_field_multiply,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedMultiAirMappedTraceError {
    Shape,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirMappedTermScheduleCols<T> {
    pub active: T,
    pub has_term: T,
    pub is_first: T,
    pub is_last: T,
    pub term: T,
    pub block_start: T,
    pub log_height: T,
    pub rotation: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirMappedTermCols<T> {
    pub opening: [T; D_EF],
    pub rho: [T; D_EF],
    pub scale: [T; D_EF],
    pub term_value: [T; D_EF],
    pub target_before: [T; D_EF],
    pub target_after: [T; D_EF],
}

/// Exact `rho^i` target and mapped-term descriptor producer.
pub struct FixedMultiAirMappedTermAir {
    pub opening_bus: FixedMultiAirRegionOpeningBus,
    pub rho_bus: FixedMultiAirRegionRhoBus,
    pub header_bus: FixedMultiAirStructuredClaimHeaderBus,
    pub term_bus: FixedMultiAirMappedTermBus,
    pub region: usize,
    pub global_log_message: usize,
    pub term_count: usize,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirMappedTermAir {}
impl PartitionedBaseAir<F> for FixedMultiAirMappedTermAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirMappedTermScheduleCols::<F>::width()]
    }
    fn common_main_width(&self) -> usize {
        FixedMultiAirMappedTermCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirMappedTermAir {}
impl BaseAir<F> for FixedMultiAirMappedTermAir {
    fn width(&self) -> usize {
        FixedMultiAirMappedTermScheduleCols::<F>::width()
            + FixedMultiAirMappedTermCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirMappedTermAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("mapped-term schedule")
            .to_vec();
        let next_cached = builder.cached_mains()[0]
            .row_slice(1)
            .expect("next mapped-term schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("mapped-term row")
            .to_vec();
        let next_common = builder
            .common_main()
            .row_slice(1)
            .expect("next mapped-term row")
            .to_vec();
        let schedule: &FixedMultiAirMappedTermScheduleCols<AB::Var> = cached.as_slice().borrow();
        let next_schedule: &FixedMultiAirMappedTermScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FixedMultiAirMappedTermCols<AB::Var> = common.as_slice().borrow();
        let next: &FixedMultiAirMappedTermCols<AB::Var> = next_common.as_slice().borrow();
        for flag in [
            schedule.active,
            schedule.has_term,
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
        let one = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            }
        });
        assert_array_eq(&mut builder.when(schedule.is_first), local.scale, one);
        assert_array_eq(
            &mut builder.when(schedule.is_first),
            local.target_before,
            [AB::Expr::ZERO; D_EF],
        );
        let term_value = ext_field_multiply::<AB::Expr>(local.opening, local.scale);
        assert_array_eq(
            &mut builder.when(schedule.has_term),
            local.term_value,
            term_value,
        );
        let expected_after = core::array::from_fn(|limb| {
            local.target_before[limb] + schedule.has_term * local.term_value[limb]
        });
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.target_after,
            expected_after,
        );
        let mut transition = builder.when_transition();
        let mut same = transition.when(next_schedule.active);
        assert_array_eq(&mut same, next.rho, local.rho);
        assert_array_eq(&mut same, next.target_before, local.target_after);
        assert_array_eq(
            &mut same,
            next.scale,
            ext_field_multiply::<AB::Expr>(local.scale, local.rho),
        );
        self.rho_bus.lookup_key(
            builder,
            FixedMultiAirRegionRhoMessage {
                region: AB::Expr::from_usize(self.region),
                value: local.rho.map(Into::into),
            },
            schedule.is_first,
        );
        self.opening_bus.lookup_key(
            builder,
            FixedMultiAirRegionOpeningMessage {
                region: AB::Expr::from_usize(self.region),
                opening: schedule.term.into(),
                value: local.opening.map(Into::into),
            },
            schedule.has_term,
        );
        self.term_bus.send(
            builder,
            FixedMultiAirMappedTermMessage {
                claim: AB::Expr::from_usize(self.region),
                term: schedule.term.into(),
                block_start: schedule.block_start.into(),
                log_height: schedule.log_height.into(),
                l_skip: AB::Expr::ZERO,
                rotation: schedule.rotation.into(),
                scale: local.scale.map(Into::into),
            },
            schedule.has_term,
        );
        self.header_bus.send(
            builder,
            FixedMultiAirStructuredClaimHeaderMessage {
                claim: AB::Expr::from_usize(self.region),
                kind: AB::Expr::ZERO,
                log_message_len: AB::Expr::from_usize(self.global_log_message),
                term_count: AB::Expr::from_usize(self.term_count),
                point_len: AB::Expr::ZERO,
                target: local.target_after.map(Into::into),
            },
            schedule.is_last,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirMappedPointScheduleCols<T> {
    pub active: T,
    pub term: T,
    pub coordinate: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirMappedPointCols<T> {
    pub value: [T; D_EF],
}

pub struct FixedMultiAirMappedPointAir {
    pub point_bus: FixedMultiAirRegionPointBus,
    pub structured_point_bus: FixedMultiAirStructuredPointBus,
    pub region: usize,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirMappedPointAir {}
impl PartitionedBaseAir<F> for FixedMultiAirMappedPointAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirMappedPointScheduleCols::<F>::width()]
    }
    fn common_main_width(&self) -> usize {
        FixedMultiAirMappedPointCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirMappedPointAir {}
impl BaseAir<F> for FixedMultiAirMappedPointAir {
    fn width(&self) -> usize {
        FixedMultiAirMappedPointScheduleCols::<F>::width()
            + FixedMultiAirMappedPointCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirMappedPointAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("mapped-point schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("mapped-point row")
            .to_vec();
        let schedule: &FixedMultiAirMappedPointScheduleCols<AB::Var> = cached.as_slice().borrow();
        let local: &FixedMultiAirMappedPointCols<AB::Var> = common.as_slice().borrow();
        builder.assert_bool(schedule.active);
        self.point_bus.lookup_key(
            builder,
            FixedMultiAirRegionPointMessage {
                region: AB::Expr::from_usize(self.region),
                coordinate: schedule.coordinate.into(),
                value: local.value.map(Into::into),
            },
            schedule.active,
        );
        self.structured_point_bus.send(
            builder,
            FixedMultiAirStructuredPointMessage {
                claim: AB::Expr::from_usize(self.region),
                term: schedule.term.into(),
                coordinate: schedule.coordinate.into(),
                value: local.value.map(Into::into),
            },
            schedule.active,
        );
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirMappedTraceOutput {
    pub term_cached: RowMajorMatrix<F>,
    pub term_common: RowMajorMatrix<F>,
    pub point_cached: RowMajorMatrix<F>,
    pub point_common: RowMajorMatrix<F>,
    pub target: EF,
}

pub fn generate_fixed_multi_air_mapped_traces(
    plan: &FixedMultiAirEndpointPlan,
    _global_log_message: usize,
    openings: &[EF],
    point: &[EF],
    rho: EF,
    term_required_height: Option<usize>,
    point_required_height: Option<usize>,
) -> Result<FixedMultiAirMappedTraceOutput, FixedMultiAirMappedTraceError> {
    if openings.len() != plan.dynamic.len() || point.len() != plan.log_height {
        return Err(FixedMultiAirMappedTraceError::Shape);
    }
    let valid_term_rows = openings.len().max(1);
    let term_height = term_required_height.unwrap_or_else(|| valid_term_rows.next_power_of_two());
    if term_height < valid_term_rows {
        return Err(FixedMultiAirMappedTraceError::Shape);
    }
    let term_cached_width = FixedMultiAirMappedTermScheduleCols::<F>::width();
    let term_common_width = FixedMultiAirMappedTermCols::<F>::width();
    let mut term_cached = F::zero_vec(term_height * term_cached_width);
    let mut term_common = F::zero_vec(term_height * term_common_width);
    let mut scale = EF::ONE;
    let mut target = EF::ZERO;
    for row in 0..valid_term_rows {
        let has_term = row < openings.len();
        let opening = openings.get(row).copied().unwrap_or(EF::ZERO);
        let term_value = if has_term { scale * opening } else { EF::ZERO };
        let before = target;
        target += term_value;
        let cached_row = &mut term_cached[row * term_cached_width..(row + 1) * term_cached_width];
        let schedule: &mut FixedMultiAirMappedTermScheduleCols<F> = cached_row.borrow_mut();
        schedule.active = F::ONE;
        schedule.has_term = F::from_bool(has_term);
        schedule.is_first = F::from_bool(row == 0);
        schedule.is_last = F::from_bool(row + 1 == valid_term_rows);
        schedule.term = F::from_usize(row);
        if let Some(term) = plan.dynamic.get(row) {
            schedule.block_start = F::from_usize(term.global_block_start);
            schedule.log_height = F::from_usize(term.log_height);
            schedule.rotation = F::from_usize(match term.rotation {
                DirectAirMappedRotation::Current => 0,
                DirectAirMappedRotation::Next => 1,
            });
        }
        let common_row = &mut term_common[row * term_common_width..(row + 1) * term_common_width];
        let cols: &mut FixedMultiAirMappedTermCols<F> = common_row.borrow_mut();
        copy_ext(&mut cols.opening, opening);
        copy_ext(&mut cols.rho, rho);
        copy_ext(&mut cols.scale, scale);
        copy_ext(&mut cols.term_value, term_value);
        copy_ext(&mut cols.target_before, before);
        copy_ext(&mut cols.target_after, target);
        scale *= rho;
    }

    let valid_point_rows = openings.len().saturating_mul(point.len());
    let point_height =
        point_required_height.unwrap_or_else(|| valid_point_rows.max(1).next_power_of_two());
    if point_height < valid_point_rows {
        return Err(FixedMultiAirMappedTraceError::Shape);
    }
    let point_cached_width = FixedMultiAirMappedPointScheduleCols::<F>::width();
    let point_common_width = FixedMultiAirMappedPointCols::<F>::width();
    let mut point_cached = F::zero_vec(point_height * point_cached_width);
    let mut point_common = F::zero_vec(point_height * point_common_width);
    let mut row = 0usize;
    for term in 0..openings.len() {
        for (coordinate, &value) in point.iter().enumerate() {
            let cached_row =
                &mut point_cached[row * point_cached_width..(row + 1) * point_cached_width];
            let schedule: &mut FixedMultiAirMappedPointScheduleCols<F> = cached_row.borrow_mut();
            schedule.active = F::ONE;
            schedule.term = F::from_usize(term);
            schedule.coordinate = F::from_usize(coordinate);
            let common_row =
                &mut point_common[row * point_common_width..(row + 1) * point_common_width];
            let cols: &mut FixedMultiAirMappedPointCols<F> = common_row.borrow_mut();
            copy_ext(&mut cols.value, value);
            row += 1;
        }
    }
    Ok(FixedMultiAirMappedTraceOutput {
        term_cached: RowMajorMatrix::new(term_cached, term_cached_width),
        term_common: RowMajorMatrix::new(term_common, term_common_width),
        point_cached: RowMajorMatrix::new(point_cached, point_cached_width),
        point_common: RowMajorMatrix::new(point_common, point_common_width),
        target,
    })
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

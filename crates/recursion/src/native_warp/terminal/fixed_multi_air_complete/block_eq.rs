//! Recursive AIR for the exact unaligned block-Eq functional used by the
//! complete fixed-multi-AIR terminal relation.
//!
//! Integration note: this file deliberately defines its own narrow point and
//! output buses.  The complete terminal owner must wire `global_point_bus` to
//! the authenticated PESAT `tau`, `regional_point_bus` to the corresponding
//! local/interaction sumcheck point, and consume every `weight_bus` message
//! exactly once.  Geometry is setup-only and must come from
//! [`FixedMultiAirCompleteTerminalEqBlockPlan`].

use core::borrow::{Borrow, BorrowMut};
use std::{collections::BTreeSet, sync::Arc};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder,
    interaction::InteractionBuilder,
    native_warp::{
        FixedMultiAirCompleteTerminalConstraintPlan, FixedMultiAirCompleteTerminalEqBlockPlan,
        FixedMultiAirCompleteTerminalEqRole,
    },
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir, PairBuilder};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::prefix_decomposition::{
    FixedMultiAirCompleteInstanceValueBus, FixedMultiAirCompleteInstanceValueMessage,
};
use crate::{
    define_typed_lookup_bus, define_typed_permutation_bus,
    utils::{ext_field_add, ext_field_multiply, ext_field_one_minus},
};

const CARRY_STATES: usize = 4;
const LOCAL_CHOICES: usize = 2;
const TRANSITION_SLOTS: usize = CARRY_STATES * LOCAL_CHOICES;
/// Canonical v3 wrapper stacked-PCS column ceiling. Packing is greedy and
/// setup-only, so this affects neither Eq semantics nor proof-dependent data.
pub const FIXED_MULTI_AIR_COMPLETE_PACKED_BLOCK_EQ_MAX_ROWS: usize = 1 << 19;

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteGlobalPointMessage<T> {
    pub coordinate: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirCompleteGlobalPointBus,
    FixedMultiAirCompleteGlobalPointMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteRegionalPointMessage<T> {
    /// Setup-fixed Eq-block identity. Local and interaction owners publish to
    /// different block IDs; no witness component selector exists.
    pub block: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

define_typed_lookup_bus!(
    FixedMultiAirCompleteRegionalPointBus,
    FixedMultiAirCompleteRegionalPointMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteEqWeightMessage<T> {
    pub block: T,
    pub role: T,
    pub role_ordinal: T,
    pub value: [T; D_EF],
}

define_typed_permutation_bus!(
    FixedMultiAirCompleteEqWeightBus,
    FixedMultiAirCompleteEqWeightMessage
);

const INSTANCE_SECTION_BETA: usize = 2;

/// Setup-fixed bridge from the authenticated PESAT point `tau` to every
/// nonlinear Eq block. It consumes each instance coordinate once and
/// publishes the same typed global-point key with exactly `block_count`
/// lookups. No proof value controls either count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedMultiAirCompleteGlobalPointBridgeProfile {
    pub log_constraints: usize,
    pub block_count: usize,
}

impl FixedMultiAirCompleteGlobalPointBridgeProfile {
    pub fn new(
        log_constraints: usize,
        block_count: usize,
    ) -> Result<Self, FixedMultiAirCompleteBlockEqError> {
        if log_constraints == 0
            || block_count == 0
            || u32::try_from(log_constraints).is_err()
            || u32::try_from(block_count).is_err()
        {
            return Err(FixedMultiAirCompleteBlockEqError::Geometry(
                "global-point bridge dimensions",
            ));
        }
        Ok(Self {
            log_constraints,
            block_count,
        })
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteGlobalPointBridgeScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub coordinate: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteGlobalPointBridgeCols<T> {
    pub value: [T; D_EF],
}

pub struct FixedMultiAirCompleteGlobalPointBridgeAir {
    pub profile: FixedMultiAirCompleteGlobalPointBridgeProfile,
    pub instance_bus: FixedMultiAirCompleteInstanceValueBus,
    pub global_point_bus: FixedMultiAirCompleteGlobalPointBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteGlobalPointBridgeAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteGlobalPointBridgeAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirCompleteGlobalPointBridgeScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteGlobalPointBridgeCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteGlobalPointBridgeAir {}
impl BaseAir<F> for FixedMultiAirCompleteGlobalPointBridgeAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteGlobalPointBridgeScheduleCols::<F>::width()
            + FixedMultiAirCompleteGlobalPointBridgeCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteGlobalPointBridgeAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("complete global-point bridge schedule")
            .to_vec();
        let next_cached = builder.cached_mains()[0]
            .row_slice(1)
            .expect("next complete global-point bridge schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("complete global-point bridge row")
            .to_vec();
        let schedule: &FixedMultiAirCompleteGlobalPointBridgeScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let next_schedule: &FixedMultiAirCompleteGlobalPointBridgeScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FixedMultiAirCompleteGlobalPointBridgeCols<AB::Var> =
            common.as_slice().borrow();

        for flag in [schedule.active, schedule.is_first, schedule.is_last] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(schedule.active);
        builder.when_first_row().assert_one(schedule.is_first);
        builder.when_first_row().assert_zero(schedule.coordinate);
        builder.when(schedule.active * schedule.is_last).assert_eq(
            schedule.coordinate,
            AB::Expr::from_usize(self.profile.log_constraints - 1),
        );
        builder
            .when_transition()
            .assert_bool(schedule.active - next_schedule.active);
        builder
            .when_transition()
            .assert_eq(schedule.active - next_schedule.active, schedule.is_last);
        let mut transition = builder.when_transition();
        let mut continuation = transition.when(next_schedule.active);
        continuation.assert_zero(next_schedule.is_first);
        continuation.assert_eq(
            next_schedule.coordinate,
            schedule.coordinate + AB::Expr::ONE,
        );
        builder
            .when_last_row()
            .assert_eq(schedule.is_last, schedule.active);

        self.instance_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInstanceValueMessage {
                section: AB::Expr::from_usize(INSTANCE_SECTION_BETA),
                coordinate: schedule.coordinate.into(),
                value: local.value.map(Into::into),
            },
            schedule.active,
        );
        self.global_point_bus.add_key_with_lookups(
            builder,
            FixedMultiAirCompleteGlobalPointMessage {
                coordinate: schedule.coordinate.into(),
                value: local.value.map(Into::into),
            },
            schedule.active * AB::Expr::from_usize(self.profile.block_count),
        );
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteGlobalPointBridgeTrace {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
}

pub fn generate_fixed_multi_air_complete_global_point_bridge_trace(
    profile: FixedMultiAirCompleteGlobalPointBridgeProfile,
    beta: &[EF],
    required_height: Option<usize>,
) -> Result<FixedMultiAirCompleteGlobalPointBridgeTrace, FixedMultiAirCompleteBlockEqError> {
    let profile = FixedMultiAirCompleteGlobalPointBridgeProfile::new(
        profile.log_constraints,
        profile.block_count,
    )?;
    if beta.len() < profile.log_constraints {
        return Err(FixedMultiAirCompleteBlockEqError::Shape(
            "global-point bridge beta",
        ));
    }
    let minimum = profile.log_constraints.next_power_of_two();
    let height = required_height.unwrap_or(minimum);
    if !height.is_power_of_two() || height < profile.log_constraints {
        return Err(FixedMultiAirCompleteBlockEqError::Shape(
            "global-point bridge height",
        ));
    }
    let cached_width = FixedMultiAirCompleteGlobalPointBridgeScheduleCols::<F>::width();
    let common_width = FixedMultiAirCompleteGlobalPointBridgeCols::<F>::width();
    let cached_cells =
        height
            .checked_mul(cached_width)
            .ok_or(FixedMultiAirCompleteBlockEqError::Shape(
                "global-point bridge cached cells",
            ))?;
    let common_cells =
        height
            .checked_mul(common_width)
            .ok_or(FixedMultiAirCompleteBlockEqError::Shape(
                "global-point bridge common cells",
            ))?;
    let mut cached = F::zero_vec(cached_cells);
    let mut common = F::zero_vec(common_cells);
    for (coordinate, &value) in beta[..profile.log_constraints].iter().enumerate() {
        let schedule: &mut FixedMultiAirCompleteGlobalPointBridgeScheduleCols<F> =
            cached[coordinate * cached_width..(coordinate + 1) * cached_width].borrow_mut();
        schedule.active = F::ONE;
        schedule.is_first = F::from_bool(coordinate == 0);
        schedule.is_last = F::from_bool(coordinate + 1 == profile.log_constraints);
        schedule.coordinate = F::from_usize(coordinate);
        let cols: &mut FixedMultiAirCompleteGlobalPointBridgeCols<F> =
            common[coordinate * common_width..(coordinate + 1) * common_width].borrow_mut();
        copy_ext(&mut cols.value, value);
    }
    Ok(FixedMultiAirCompleteGlobalPointBridgeTrace {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedMultiAirCompleteBlockEqError {
    Geometry(&'static str),
    Shape(&'static str),
    Witness(&'static str),
}

/// Circuit-local identity for every use of the two-carry Eq primitive.
///
/// The backend's [`FixedMultiAirCompleteTerminalEqRole`] deliberately covers
/// nonlinear relation constraints only. A mapped opening is a terminal
/// linearizer term, not a relation constraint, so it has a distinct local
/// variant here and is never converted back into the backend enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FixedMultiAirCompleteBlockEqRole {
    LocalConstraint { constraint_ordinal: u32 },
    BetaConstraint { power_ordinal: u32 },
    InverseConstraint { interaction_ordinal: u32 },
    GlobalConstraint,
    MappedOpening { global_opening_ordinal: u32 },
}

impl From<FixedMultiAirCompleteTerminalEqRole> for FixedMultiAirCompleteBlockEqRole {
    fn from(role: FixedMultiAirCompleteTerminalEqRole) -> Self {
        match role {
            FixedMultiAirCompleteTerminalEqRole::LocalConstraint { constraint_ordinal } => {
                Self::LocalConstraint { constraint_ordinal }
            }
            FixedMultiAirCompleteTerminalEqRole::BetaConstraint { power_ordinal } => {
                Self::BetaConstraint { power_ordinal }
            }
            FixedMultiAirCompleteTerminalEqRole::InverseConstraint {
                interaction_ordinal,
            } => Self::InverseConstraint {
                interaction_ordinal,
            },
            FixedMultiAirCompleteTerminalEqRole::GlobalConstraint => Self::GlobalConstraint,
        }
    }
}

/// Setup identity of one Eq block. The role fixes whether a regional point is
/// consumed; this is never selected by witness data.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedMultiAirCompleteBlockEqProfile {
    pub block: usize,
    pub role: FixedMultiAirCompleteBlockEqRole,
    pub region: usize,
    pub geometry: FixedMultiAirCompleteTerminalEqBlockPlan,
    /// Setup-fixed number of consumers of the final Eq weight. Every local
    /// and inverse block has one consumer. The single global-constraint block
    /// is shared by every nonempty interaction region and therefore publishes
    /// one identical permutation message per such region.
    pub weight_multiplicity: usize,
}

impl FixedMultiAirCompleteBlockEqProfile {
    pub fn from_constraint(
        block: usize,
        region: usize,
        constraint: &FixedMultiAirCompleteTerminalConstraintPlan,
    ) -> Result<Self, FixedMultiAirCompleteBlockEqError> {
        Self::new(block, constraint.role.into(), region, constraint.eq.clone())
    }

    pub fn new(
        block: usize,
        role: FixedMultiAirCompleteBlockEqRole,
        region: usize,
        geometry: FixedMultiAirCompleteTerminalEqBlockPlan,
    ) -> Result<Self, FixedMultiAirCompleteBlockEqError> {
        Self::new_with_weight_multiplicity(block, role, region, geometry, 1)
    }

    pub fn new_with_weight_multiplicity(
        block: usize,
        role: FixedMultiAirCompleteBlockEqRole,
        region: usize,
        geometry: FixedMultiAirCompleteTerminalEqBlockPlan,
        weight_multiplicity: usize,
    ) -> Result<Self, FixedMultiAirCompleteBlockEqError> {
        let canonical = FixedMultiAirCompleteTerminalEqBlockPlan::new(
            usize::from(geometry.global_log_height),
            geometry.start,
            usize::from(geometry.log_height),
            usize::from(geometry.rotation),
        )
        .map_err(|_| FixedMultiAirCompleteBlockEqError::Geometry("invalid Eq geometry"))?;
        if canonical != geometry {
            return Err(FixedMultiAirCompleteBlockEqError::Geometry(
                "non-canonical two-carry schedule",
            ));
        }
        match role {
            FixedMultiAirCompleteBlockEqRole::BetaConstraint { .. }
            | FixedMultiAirCompleteBlockEqRole::GlobalConstraint
                if geometry.log_height != 0 =>
            {
                return Err(FixedMultiAirCompleteBlockEqError::Geometry(
                    "scalar role has regional geometry",
                ));
            }
            _ => {}
        }
        if weight_multiplicity == 0 || u32::try_from(weight_multiplicity).is_err() {
            return Err(FixedMultiAirCompleteBlockEqError::Geometry(
                "Eq weight multiplicity",
            ));
        }
        if weight_multiplicity != 1
            && !matches!(role, FixedMultiAirCompleteBlockEqRole::GlobalConstraint)
        {
            return Err(FixedMultiAirCompleteBlockEqError::Geometry(
                "only the global Eq weight may have multiple consumers",
            ));
        }
        Ok(Self {
            block,
            role,
            region,
            geometry,
            weight_multiplicity,
        })
    }

    #[must_use]
    pub const fn role_wire(&self) -> (usize, usize) {
        match self.role {
            FixedMultiAirCompleteBlockEqRole::LocalConstraint { constraint_ordinal } => {
                (0, constraint_ordinal as usize)
            }
            FixedMultiAirCompleteBlockEqRole::BetaConstraint { power_ordinal } => {
                (1, power_ordinal as usize)
            }
            FixedMultiAirCompleteBlockEqRole::InverseConstraint {
                interaction_ordinal,
            } => (2, interaction_ordinal as usize),
            FixedMultiAirCompleteBlockEqRole::GlobalConstraint => (3, 0),
            FixedMultiAirCompleteBlockEqRole::MappedOpening {
                global_opening_ordinal,
            } => (4, global_opening_ordinal as usize),
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteBlockEqScheduleCols<T> {
    pub active: T,
    pub has_round: T,
    pub is_first: T,
    pub is_last: T,
    pub has_local_bit: T,
    pub global_coordinate: T,
    pub local_coordinate: T,
    pub transition_enabled: [T; TRANSITION_SLOTS],
    pub transition_global_bit: [T; TRANSITION_SLOTS],
    pub transition_destination: [[T; CARRY_STATES]; TRANSITION_SLOTS],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteBlockEqCols<T> {
    pub global_point: [T; D_EF],
    pub local_point: [T; D_EF],
    pub state_before: [[T; D_EF]; CARRY_STATES],
    pub state_after: [[T; D_EF]; CARRY_STATES],
    pub weight: [T; D_EF],
}

/// Setup-fixed schedule for a forest of Eq blocks packed into one trace.
///
/// All descriptor columns live in cached main. They are therefore committed
/// by the verifying key and cannot be selected by the private witness. The
/// common trace contains only the same four-carry computation as the legacy
/// one-block AIR.
#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompletePackedBlockEqScheduleCols<T> {
    pub active: T,
    pub has_round: T,
    pub is_first: T,
    pub is_last: T,
    pub has_local_bit: T,
    pub initial_rotation: T,
    pub block: T,
    pub role: T,
    pub role_ordinal: T,
    pub weight_multiplicity: T,
    pub global_coordinate: T,
    pub local_coordinate: T,
    pub transition_enabled: [T; TRANSITION_SLOTS],
    pub transition_global_bit: [T; TRANSITION_SLOTS],
    pub transition_destination: [[T; CARRY_STATES]; TRANSITION_SLOTS],
}

/// Canonical setup inventory for the packed Eq table. Block identity remains
/// unique and the block order is fixed before any proof witness is known.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedMultiAirCompletePackedBlockEqProfile {
    pub blocks: Vec<FixedMultiAirCompleteBlockEqProfile>,
    pub valid_rows: usize,
}

impl FixedMultiAirCompletePackedBlockEqProfile {
    pub fn new(
        blocks: Vec<FixedMultiAirCompleteBlockEqProfile>,
    ) -> Result<Self, FixedMultiAirCompleteBlockEqError> {
        if blocks.is_empty() {
            return Err(FixedMultiAirCompleteBlockEqError::Geometry(
                "empty packed Eq inventory",
            ));
        }
        let mut block_ids = BTreeSet::new();
        let mut valid_rows = 0usize;
        for block in &blocks {
            let canonical = FixedMultiAirCompleteBlockEqProfile::new_with_weight_multiplicity(
                block.block,
                block.role,
                block.region,
                block.geometry.clone(),
                block.weight_multiplicity,
            )?;
            if &canonical != block || !block_ids.insert(block.block) {
                return Err(FixedMultiAirCompleteBlockEqError::Geometry(
                    "non-canonical packed Eq inventory",
                ));
            }
            valid_rows = valid_rows
                .checked_add(block.geometry.rounds.len().max(1))
                .ok_or(FixedMultiAirCompleteBlockEqError::Geometry(
                    "packed Eq row count",
                ))?;
        }
        Ok(Self { blocks, valid_rows })
    }

    #[must_use]
    pub fn height(&self) -> usize {
        self.valid_rows.next_power_of_two()
    }

    pub fn partition(
        blocks: Vec<FixedMultiAirCompleteBlockEqProfile>,
        max_rows: usize,
    ) -> Result<Vec<Self>, FixedMultiAirCompleteBlockEqError> {
        if max_rows == 0 || !max_rows.is_power_of_two() {
            return Err(FixedMultiAirCompleteBlockEqError::Geometry(
                "packed Eq partition height",
            ));
        }
        let canonical = Self::new(blocks)?;
        let mut partitions = Vec::new();
        let mut current = Vec::new();
        let mut current_rows = 0usize;
        for block in canonical.blocks {
            let rows = block.geometry.rounds.len().max(1);
            if rows > max_rows {
                return Err(FixedMultiAirCompleteBlockEqError::Geometry(
                    "Eq block exceeds packed partition",
                ));
            }
            if !current.is_empty() && current_rows + rows > max_rows {
                partitions.push(Self::new(core::mem::take(&mut current))?);
                current_rows = 0;
            }
            current_rows += rows;
            current.push(block);
        }
        if !current.is_empty() {
            partitions.push(Self::new(current)?);
        }
        Ok(partitions)
    }
}

/// One AIR program for all setup-fixed Eq blocks. This changes only physical
/// batching: each row emits the exact same typed point and weight messages as
/// the corresponding one-block AIR.
pub struct FixedMultiAirCompletePackedBlockEqAir {
    pub profile: Arc<FixedMultiAirCompletePackedBlockEqProfile>,
    pub global_point_bus: FixedMultiAirCompleteGlobalPointBus,
    pub regional_point_bus: FixedMultiAirCompleteRegionalPointBus,
    pub weight_bus: FixedMultiAirCompleteEqWeightBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompletePackedBlockEqAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompletePackedBlockEqAir {
    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteBlockEqCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompletePackedBlockEqAir {}
impl BaseAir<F> for FixedMultiAirCompletePackedBlockEqAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteBlockEqCols::<F>::width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        Some(
            generate_fixed_multi_air_complete_packed_block_eq_preprocessed(&self.profile)
                .expect("validated complete packed block-Eq preprocessed trace"),
        )
    }
}

impl<AB> Air<AB> for FixedMultiAirCompletePackedBlockEqAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder + PairBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder
            .preprocessed()
            .row_slice(0)
            .expect("complete packed block-Eq schedule")
            .to_vec();
        let next_cached = builder
            .preprocessed()
            .row_slice(1)
            .expect("next complete packed block-Eq schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("complete packed block-Eq row")
            .to_vec();
        let next_common = builder
            .common_main()
            .row_slice(1)
            .expect("next complete packed block-Eq row")
            .to_vec();
        let schedule: &FixedMultiAirCompletePackedBlockEqScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let next_schedule: &FixedMultiAirCompletePackedBlockEqScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FixedMultiAirCompleteBlockEqCols<AB::Var> = common.as_slice().borrow();
        let next: &FixedMultiAirCompleteBlockEqCols<AB::Var> = next_common.as_slice().borrow();

        for flag in [
            schedule.active,
            schedule.has_round,
            schedule.is_first,
            schedule.is_last,
            schedule.has_local_bit,
            schedule.initial_rotation,
        ]
        .into_iter()
        .chain(schedule.transition_enabled)
        .chain(schedule.transition_global_bit)
        .chain(schedule.transition_destination.into_iter().flatten())
        {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(schedule.active);
        builder.when_first_row().assert_one(schedule.is_first);
        builder
            .when(AB::Expr::ONE - schedule.active)
            .assert_zero(schedule.is_first + schedule.is_last + schedule.has_round);
        builder
            .when_transition()
            .assert_bool(schedule.active - next_schedule.active);
        let mut transition = builder.when_transition();
        transition
            .when(next_schedule.active)
            .assert_eq(next_schedule.is_first, schedule.is_last);
        transition
            .when(schedule.active - next_schedule.active)
            .assert_one(schedule.is_last);
        builder
            .when_last_row()
            .when(schedule.active)
            .assert_one(schedule.is_last);

        let mut transition = builder.when_transition();
        let mut continuation =
            transition.when(next_schedule.active * (AB::Expr::ONE - next_schedule.is_first));
        continuation.assert_eq(next_schedule.block, schedule.block);
        continuation.assert_eq(next_schedule.role, schedule.role);
        continuation.assert_eq(next_schedule.role_ordinal, schedule.role_ordinal);
        continuation.assert_eq(
            next_schedule.weight_multiplicity,
            schedule.weight_multiplicity,
        );
        continuation.assert_eq(next_schedule.initial_rotation, schedule.initial_rotation);
        for state in 0..CARRY_STATES {
            assert_array_eq(
                &mut continuation,
                next.state_before[state],
                local.state_after[state],
            );
        }

        for input in 0..CARRY_STATES {
            let first_slot = input * LOCAL_CHOICES;
            builder.assert_eq(schedule.transition_enabled[first_slot], schedule.has_round);
            builder.assert_eq(
                schedule.transition_enabled[first_slot + 1],
                schedule.has_round * schedule.has_local_bit,
            );
            for choice in 0..LOCAL_CHOICES {
                let slot = first_slot + choice;
                builder.assert_eq(
                    schedule.transition_destination[slot]
                        .iter()
                        .fold(AB::Expr::ZERO, |sum, &flag| sum + flag),
                    schedule.transition_enabled[slot],
                );
            }
        }

        let one: [AB::Expr; D_EF] = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            }
        });
        for state in 0..CARRY_STATES {
            let selected = if state == 0 {
                AB::Expr::ONE - schedule.initial_rotation
            } else if state == 2 {
                schedule.initial_rotation.into()
            } else {
                AB::Expr::ZERO
            };
            let expected = core::array::from_fn(|limb| selected.clone() * one[limb].clone());
            assert_array_eq(
                &mut builder.when(schedule.active * schedule.is_first),
                local.state_before[state],
                expected,
            );
        }

        let local_zero = ext_field_one_minus::<AB::Expr>(local.local_point);
        let global_zero = ext_field_one_minus::<AB::Expr>(local.global_point);
        let mut expected_after: [[AB::Expr; D_EF]; CARRY_STATES] =
            core::array::from_fn(|_| core::array::from_fn(|_| AB::Expr::ZERO));
        for input in 0..CARRY_STATES {
            for choice in 0..LOCAL_CHOICES {
                let slot = input * LOCAL_CHOICES + choice;
                let local_factor = if choice == 0 {
                    core::array::from_fn(|limb| {
                        schedule.has_local_bit * local_zero[limb].clone()
                            + (AB::Expr::ONE - schedule.has_local_bit) * one[limb].clone()
                    })
                } else {
                    local.local_point.map(Into::into)
                };
                let global_factor = core::array::from_fn(|limb| {
                    schedule.transition_global_bit[slot] * AB::Expr::from(local.global_point[limb])
                        + (AB::Expr::ONE - schedule.transition_global_bit[slot])
                            * global_zero[limb].clone()
                });
                let contribution = ext_field_multiply::<AB::Expr>(
                    ext_field_multiply::<AB::Expr>(local.state_before[input], local_factor),
                    global_factor,
                );
                for output in 0..CARRY_STATES {
                    for limb in 0..D_EF {
                        expected_after[output][limb] = expected_after[output][limb].clone()
                            + schedule.transition_destination[slot][output]
                                * contribution[limb].clone();
                    }
                }
            }
        }
        for state in 0..CARRY_STATES {
            let expected = core::array::from_fn(|limb| {
                schedule.has_round * expected_after[state][limb].clone()
                    + (AB::Expr::ONE - schedule.has_round)
                        * AB::Expr::from(local.state_before[state][limb])
            });
            assert_array_eq(
                &mut builder.when(schedule.active),
                local.state_after[state],
                expected,
            );
        }

        let final_weight = ext_field_add::<AB::Expr>(local.state_after[0], local.state_after[2]);
        assert_array_eq(
            &mut builder.when(schedule.active * schedule.is_last),
            local.weight,
            final_weight,
        );
        assert_array_eq(
            &mut builder.when(schedule.active * (AB::Expr::ONE - schedule.is_last)),
            local.weight,
            [AB::Expr::ZERO; D_EF],
        );

        self.global_point_bus.lookup_key(
            builder,
            FixedMultiAirCompleteGlobalPointMessage {
                coordinate: schedule.global_coordinate.into(),
                value: local.global_point.map(Into::into),
            },
            schedule.has_round,
        );
        self.regional_point_bus.lookup_key(
            builder,
            FixedMultiAirCompleteRegionalPointMessage {
                block: schedule.block.into(),
                coordinate: schedule.local_coordinate.into(),
                value: local.local_point.map(Into::into),
            },
            schedule.has_round * schedule.has_local_bit,
        );
        self.weight_bus.send(
            builder,
            FixedMultiAirCompleteEqWeightMessage {
                block: schedule.block.into(),
                role: schedule.role.into(),
                role_ordinal: schedule.role_ordinal.into(),
                value: local.weight.map(Into::into),
            },
            schedule.is_last * schedule.weight_multiplicity,
        );
    }
}

pub struct FixedMultiAirCompleteBlockEqAir {
    pub profile: FixedMultiAirCompleteBlockEqProfile,
    pub global_point_bus: FixedMultiAirCompleteGlobalPointBus,
    pub regional_point_bus: FixedMultiAirCompleteRegionalPointBus,
    pub weight_bus: FixedMultiAirCompleteEqWeightBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteBlockEqAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteBlockEqAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirCompleteBlockEqScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteBlockEqCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteBlockEqAir {}
impl BaseAir<F> for FixedMultiAirCompleteBlockEqAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteBlockEqScheduleCols::<F>::width()
            + FixedMultiAirCompleteBlockEqCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteBlockEqAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("complete block-Eq schedule")
            .to_vec();
        let next_cached = builder.cached_mains()[0]
            .row_slice(1)
            .expect("next complete block-Eq schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("complete block-Eq row")
            .to_vec();
        let next_common = builder
            .common_main()
            .row_slice(1)
            .expect("next complete block-Eq row")
            .to_vec();
        let schedule: &FixedMultiAirCompleteBlockEqScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let next_schedule: &FixedMultiAirCompleteBlockEqScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FixedMultiAirCompleteBlockEqCols<AB::Var> = common.as_slice().borrow();
        let next: &FixedMultiAirCompleteBlockEqCols<AB::Var> = next_common.as_slice().borrow();

        for flag in [
            schedule.active,
            schedule.has_round,
            schedule.is_first,
            schedule.is_last,
            schedule.has_local_bit,
        ]
        .into_iter()
        .chain(schedule.transition_enabled)
        .chain(schedule.transition_global_bit)
        .chain(schedule.transition_destination.into_iter().flatten())
        {
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

        let mut transition = builder.when_transition();
        let mut continuation =
            transition.when(next_schedule.active * (AB::Expr::ONE - next_schedule.is_first));
        for state in 0..CARRY_STATES {
            assert_array_eq(
                &mut continuation,
                next.state_before[state],
                local.state_after[state],
            );
        }

        for input in 0..CARRY_STATES {
            let first_slot = input * LOCAL_CHOICES;
            builder.assert_eq(schedule.transition_enabled[first_slot], schedule.has_round);
            builder.assert_eq(
                schedule.transition_enabled[first_slot + 1],
                schedule.has_round * schedule.has_local_bit,
            );
            for choice in 0..LOCAL_CHOICES {
                let slot = first_slot + choice;
                builder.assert_eq(
                    schedule.transition_destination[slot]
                        .iter()
                        .fold(AB::Expr::ZERO, |sum, &flag| sum + flag),
                    schedule.transition_enabled[slot],
                );
            }
        }

        let one = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            }
        });
        for state in 0..CARRY_STATES {
            let expected = if state == usize::from(self.profile.geometry.rotation) * 2 {
                one.clone()
            } else {
                [AB::Expr::ZERO; D_EF]
            };
            assert_array_eq(
                &mut builder.when(schedule.is_first),
                local.state_before[state],
                expected,
            );
        }

        let local_zero = ext_field_one_minus::<AB::Expr>(local.local_point);
        let global_zero = ext_field_one_minus::<AB::Expr>(local.global_point);
        let mut expected_after: [[AB::Expr; D_EF]; CARRY_STATES] =
            core::array::from_fn(|_| core::array::from_fn(|_| AB::Expr::ZERO));
        for input in 0..CARRY_STATES {
            for choice in 0..LOCAL_CHOICES {
                let slot = input * LOCAL_CHOICES + choice;
                let local_factor = if choice == 0 {
                    core::array::from_fn(|limb| {
                        schedule.has_local_bit * local_zero[limb].clone()
                            + (AB::Expr::ONE - schedule.has_local_bit) * one[limb].clone()
                    })
                } else {
                    local.local_point.map(Into::into)
                };
                let global_factor = core::array::from_fn(|limb| {
                    schedule.transition_global_bit[slot] * AB::Expr::from(local.global_point[limb])
                        + (AB::Expr::ONE - schedule.transition_global_bit[slot])
                            * global_zero[limb].clone()
                });
                let contribution = ext_field_multiply::<AB::Expr>(
                    ext_field_multiply::<AB::Expr>(local.state_before[input], local_factor),
                    global_factor,
                );
                for output in 0..CARRY_STATES {
                    for limb in 0..D_EF {
                        expected_after[output][limb] = expected_after[output][limb].clone()
                            + schedule.transition_destination[slot][output]
                                * contribution[limb].clone();
                    }
                }
            }
        }
        for state in 0..CARRY_STATES {
            let expected = core::array::from_fn(|limb| {
                schedule.has_round * expected_after[state][limb].clone()
                    + (AB::Expr::ONE - schedule.has_round)
                        * AB::Expr::from(local.state_before[state][limb])
            });
            assert_array_eq(
                &mut builder.when(schedule.active),
                local.state_after[state],
                expected,
            );
        }

        let final_weight = ext_field_add::<AB::Expr>(local.state_after[0], local.state_after[2]);
        assert_array_eq(
            &mut builder.when(schedule.is_last),
            local.weight,
            final_weight,
        );
        assert_array_eq(
            &mut builder.when(schedule.active * (AB::Expr::ONE - schedule.is_last)),
            local.weight,
            [AB::Expr::ZERO; D_EF],
        );

        self.global_point_bus.lookup_key(
            builder,
            FixedMultiAirCompleteGlobalPointMessage {
                coordinate: schedule.global_coordinate.into(),
                value: local.global_point.map(Into::into),
            },
            schedule.has_round,
        );
        self.regional_point_bus.lookup_key(
            builder,
            FixedMultiAirCompleteRegionalPointMessage {
                block: AB::Expr::from_usize(self.profile.block),
                coordinate: schedule.local_coordinate.into(),
                value: local.local_point.map(Into::into),
            },
            schedule.has_round * schedule.has_local_bit,
        );
        let (role, role_ordinal) = self.profile.role_wire();
        self.weight_bus.send(
            builder,
            FixedMultiAirCompleteEqWeightMessage {
                block: AB::Expr::from_usize(self.profile.block),
                role: AB::Expr::from_usize(role),
                role_ordinal: AB::Expr::from_usize(role_ordinal),
                value: local.weight.map(Into::into),
            },
            schedule.is_last * AB::Expr::from_usize(self.profile.weight_multiplicity),
        );
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteBlockEqTrace {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub weight: EF,
}

pub fn generate_fixed_multi_air_complete_block_eq_trace(
    profile: &FixedMultiAirCompleteBlockEqProfile,
    global_point: &[EF],
    regional_point: &[EF],
    required_height: Option<usize>,
) -> Result<FixedMultiAirCompleteBlockEqTrace, FixedMultiAirCompleteBlockEqError> {
    // Revalidate because profiles can be deserialized or assembled outside the
    // canonical backend accessor before integration.
    let canonical = FixedMultiAirCompleteBlockEqProfile::new_with_weight_multiplicity(
        profile.block,
        profile.role,
        profile.region,
        profile.geometry.clone(),
        profile.weight_multiplicity,
    )?;
    if &canonical != profile {
        return Err(FixedMultiAirCompleteBlockEqError::Geometry(
            "non-canonical Eq profile",
        ));
    }
    let global_log_height = usize::from(profile.geometry.global_log_height);
    let local_log_height = usize::from(profile.geometry.log_height);
    if global_point.len() != global_log_height || regional_point.len() != local_log_height {
        return Err(FixedMultiAirCompleteBlockEqError::Shape(
            "Eq point dimension",
        ));
    }
    let valid_rows = profile.geometry.rounds.len().max(1);
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows || !height.is_power_of_two() {
        return Err(FixedMultiAirCompleteBlockEqError::Shape("Eq trace height"));
    }
    let cached_width = FixedMultiAirCompleteBlockEqScheduleCols::<F>::width();
    let common_width = FixedMultiAirCompleteBlockEqCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    let mut state = [[EF::ZERO; 2]; 2];
    state[usize::from(profile.geometry.rotation)][0] = EF::ONE;
    let rounds = if profile.geometry.rounds.is_empty() {
        1
    } else {
        profile.geometry.rounds.len()
    };
    for row in 0..rounds {
        let schedule: &mut FixedMultiAirCompleteBlockEqScheduleCols<F> =
            cached[row * cached_width..(row + 1) * cached_width].borrow_mut();
        schedule.active = F::ONE;
        schedule.is_first = F::from_bool(row == 0);
        schedule.is_last = F::from_bool(row + 1 == rounds);
        let cols: &mut FixedMultiAirCompleteBlockEqCols<F> =
            common[row * common_width..(row + 1) * common_width].borrow_mut();
        copy_state(&mut cols.state_before, &state);
        if let Some(round) = profile.geometry.rounds.get(row) {
            schedule.has_round = F::ONE;
            schedule.has_local_bit = F::from_bool(round.has_local_bit);
            let bit = usize::from(round.bit_ordinal);
            let global_coordinate = global_log_height.checked_sub(1 + bit).ok_or(
                FixedMultiAirCompleteBlockEqError::Geometry("global coordinate underflow"),
            )?;
            schedule.global_coordinate = F::from_usize(global_coordinate);
            copy_ext(&mut cols.global_point, global_point[global_coordinate]);
            if round.has_local_bit {
                let local_coordinate = local_log_height.checked_sub(1 + bit).ok_or(
                    FixedMultiAirCompleteBlockEqError::Geometry("local coordinate underflow"),
                )?;
                schedule.local_coordinate = F::from_usize(local_coordinate);
                copy_ext(&mut cols.local_point, regional_point[local_coordinate]);
            }
            let mut seen = [false; TRANSITION_SLOTS];
            let mut next = [[EF::ZERO; 2]; 2];
            for transition in &round.transitions {
                let rotation_in = usize::from(transition.rotation_carry_in);
                let start_in = usize::from(transition.start_carry_in);
                let local_bit = transition.local_bit.map_or(0, usize::from);
                let rotation_out = usize::from(transition.rotation_carry_out);
                let start_out = usize::from(transition.start_carry_out);
                if rotation_in > 1
                    || start_in > 1
                    || local_bit > 1
                    || rotation_out > 1
                    || start_out > 1
                    || transition.source_bit > 1
                    || transition.global_bit > 1
                    || transition.local_bit.is_some() != round.has_local_bit
                {
                    return Err(FixedMultiAirCompleteBlockEqError::Geometry(
                        "malformed two-carry transition",
                    ));
                }
                let slot = (rotation_in * 2 + start_in) * LOCAL_CHOICES + local_bit;
                if seen[slot] {
                    return Err(FixedMultiAirCompleteBlockEqError::Geometry(
                        "duplicate two-carry transition",
                    ));
                }
                seen[slot] = true;
                schedule.transition_enabled[slot] = F::ONE;
                schedule.transition_global_bit[slot] = F::from_u8(transition.global_bit);
                schedule.transition_destination[slot][rotation_out * 2 + start_out] = F::ONE;
                let local_factor = if round.has_local_bit {
                    let coordinate = regional_point[local_log_height - 1 - bit];
                    if local_bit == 1 {
                        coordinate
                    } else {
                        EF::ONE - coordinate
                    }
                } else {
                    EF::ONE
                };
                let coordinate = global_point[global_coordinate];
                let global_factor = if transition.global_bit == 1 {
                    coordinate
                } else {
                    EF::ONE - coordinate
                };
                next[rotation_out][start_out] +=
                    state[rotation_in][start_in] * local_factor * global_factor;
            }
            let expected_slots = if round.has_local_bit { 8 } else { 4 };
            if seen.iter().filter(|&&present| present).count() != expected_slots {
                return Err(FixedMultiAirCompleteBlockEqError::Geometry(
                    "incomplete two-carry round",
                ));
            }
            state = next;
        }
        copy_state(&mut cols.state_after, &state);
        if row + 1 == rounds {
            let weight = state[0][0] + state[1][0];
            copy_ext(&mut cols.weight, weight);
        }
    }
    let weight = state[0][0] + state[1][0];
    Ok(FixedMultiAirCompleteBlockEqTrace {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        weight,
    })
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompletePackedBlockEqTrace {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    /// Differential anchor in the setup-fixed packed block order.
    pub weights: Vec<EF>,
}

/// Streaming assembler used by the complete terminal trace generator. It
/// copies only active rows from each reference block and drops the temporary
/// per-block matrices immediately, avoiding both independent power-of-two
/// padding and retention of thousands of allocations.
pub struct FixedMultiAirCompletePackedBlockEqTraceBuilder<'a> {
    profile: &'a FixedMultiAirCompletePackedBlockEqProfile,
    cached: Vec<F>,
    common: Vec<F>,
    next_block: usize,
    next_row: usize,
    weights: Vec<EF>,
}

impl<'a> FixedMultiAirCompletePackedBlockEqTraceBuilder<'a> {
    pub fn new(
        profile: &'a FixedMultiAirCompletePackedBlockEqProfile,
        required_height: Option<usize>,
    ) -> Result<Self, FixedMultiAirCompleteBlockEqError> {
        let minimum = profile.height();
        let height = required_height.unwrap_or(minimum);
        if !height.is_power_of_two() || height < profile.valid_rows {
            return Err(FixedMultiAirCompleteBlockEqError::Shape(
                "packed Eq trace height",
            ));
        }
        let cached_width = FixedMultiAirCompletePackedBlockEqScheduleCols::<F>::width();
        let common_width = FixedMultiAirCompleteBlockEqCols::<F>::width();
        let cached_cells =
            height
                .checked_mul(cached_width)
                .ok_or(FixedMultiAirCompleteBlockEqError::Shape(
                    "packed Eq cached cells",
                ))?;
        let common_cells =
            height
                .checked_mul(common_width)
                .ok_or(FixedMultiAirCompleteBlockEqError::Shape(
                    "packed Eq common cells",
                ))?;
        Ok(Self {
            profile,
            cached: F::zero_vec(cached_cells),
            common: F::zero_vec(common_cells),
            next_block: 0,
            next_row: 0,
            weights: Vec::with_capacity(profile.blocks.len()),
        })
    }

    pub fn push(
        &mut self,
        profile: &FixedMultiAirCompleteBlockEqProfile,
        trace: &FixedMultiAirCompleteBlockEqTrace,
    ) -> Result<(), FixedMultiAirCompleteBlockEqError> {
        let expected = self.profile.blocks.get(self.next_block).ok_or(
            FixedMultiAirCompleteBlockEqError::Witness("too many packed Eq blocks"),
        )?;
        if expected != profile {
            return Err(FixedMultiAirCompleteBlockEqError::Witness(
                "packed Eq block order",
            ));
        }
        let valid_rows = profile.geometry.rounds.len().max(1);
        let source_cached_width = FixedMultiAirCompleteBlockEqScheduleCols::<F>::width();
        let packed_cached_width = FixedMultiAirCompletePackedBlockEqScheduleCols::<F>::width();
        let common_width = FixedMultiAirCompleteBlockEqCols::<F>::width();
        if trace.cached.width() != source_cached_width
            || trace.common.width() != common_width
            || trace.cached.height() < valid_rows
            || trace.common.height() < valid_rows
            || self.next_row.checked_add(valid_rows).is_none()
            || self.next_row + valid_rows > self.profile.valid_rows
        {
            return Err(FixedMultiAirCompleteBlockEqError::Shape(
                "packed Eq source trace",
            ));
        }
        let (role, role_ordinal) = profile.role_wire();
        for row in 0..valid_rows {
            let source: &FixedMultiAirCompleteBlockEqScheduleCols<F> = trace.cached.values
                [row * source_cached_width..(row + 1) * source_cached_width]
                .borrow();
            let destination_row = self.next_row + row;
            let destination: &mut FixedMultiAirCompletePackedBlockEqScheduleCols<F> = self.cached
                [destination_row * packed_cached_width
                    ..(destination_row + 1) * packed_cached_width]
                .borrow_mut();
            destination.active = source.active;
            destination.has_round = source.has_round;
            destination.is_first = source.is_first;
            destination.is_last = source.is_last;
            destination.has_local_bit = source.has_local_bit;
            destination.initial_rotation = F::from_u8(profile.geometry.rotation);
            destination.block = F::from_usize(profile.block);
            destination.role = F::from_usize(role);
            destination.role_ordinal = F::from_usize(role_ordinal);
            destination.weight_multiplicity = F::from_usize(profile.weight_multiplicity);
            destination.global_coordinate = source.global_coordinate;
            destination.local_coordinate = source.local_coordinate;
            destination.transition_enabled = source.transition_enabled;
            destination.transition_global_bit = source.transition_global_bit;
            destination.transition_destination = source.transition_destination;

            let source_common = &trace.common.values[row * common_width..(row + 1) * common_width];
            self.common[destination_row * common_width..(destination_row + 1) * common_width]
                .copy_from_slice(source_common);
        }
        self.next_row += valid_rows;
        self.next_block += 1;
        self.weights.push(trace.weight);
        Ok(())
    }

    pub fn finish(
        self,
    ) -> Result<FixedMultiAirCompletePackedBlockEqTrace, FixedMultiAirCompleteBlockEqError> {
        if self.next_block != self.profile.blocks.len() || self.next_row != self.profile.valid_rows
        {
            return Err(FixedMultiAirCompleteBlockEqError::Witness(
                "incomplete packed Eq inventory",
            ));
        }
        Ok(FixedMultiAirCompletePackedBlockEqTrace {
            cached: RowMajorMatrix::new(
                self.cached,
                FixedMultiAirCompletePackedBlockEqScheduleCols::<F>::width(),
            ),
            common: RowMajorMatrix::new(
                self.common,
                FixedMultiAirCompleteBlockEqCols::<F>::width(),
            ),
            weights: self.weights,
        })
    }
}

/// Deterministic setup trace committed in the verifying key. Eq transition
/// schedules and descriptor identities do not depend on point values, so zero
/// points produce exactly the cached schedule used by every honest witness.
pub fn generate_fixed_multi_air_complete_packed_block_eq_preprocessed(
    profile: &FixedMultiAirCompletePackedBlockEqProfile,
) -> Result<RowMajorMatrix<F>, FixedMultiAirCompleteBlockEqError> {
    let mut builder = FixedMultiAirCompletePackedBlockEqTraceBuilder::new(profile, None)?;
    for block in &profile.blocks {
        let global = vec![EF::ZERO; usize::from(block.geometry.global_log_height)];
        let regional = vec![EF::ZERO; usize::from(block.geometry.log_height)];
        let trace =
            generate_fixed_multi_air_complete_block_eq_trace(block, &global, &regional, None)?;
        builder.push(block, &trace)?;
    }
    Ok(builder.finish()?.cached)
}

/// Streaming owner for a small forest of packed tables. It preserves the
/// canonical block order while keeping every column within the stacked-PCS
/// height admitted by the wrapper parameters.
pub struct FixedMultiAirCompletePackedBlockEqForestTraceBuilder<'a> {
    tables: Vec<FixedMultiAirCompletePackedBlockEqTraceBuilder<'a>>,
    current: usize,
}

impl<'a> FixedMultiAirCompletePackedBlockEqForestTraceBuilder<'a> {
    pub fn new(
        profiles: &'a [Arc<FixedMultiAirCompletePackedBlockEqProfile>],
    ) -> Result<Self, FixedMultiAirCompleteBlockEqError> {
        if profiles.is_empty() {
            return Err(FixedMultiAirCompleteBlockEqError::Geometry(
                "empty packed Eq forest",
            ));
        }
        let tables = profiles
            .iter()
            .map(|profile| {
                FixedMultiAirCompletePackedBlockEqTraceBuilder::new(profile.as_ref(), None)
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self { tables, current: 0 })
    }

    pub fn push(
        &mut self,
        profile: &FixedMultiAirCompleteBlockEqProfile,
        trace: &FixedMultiAirCompleteBlockEqTrace,
    ) -> Result<(), FixedMultiAirCompleteBlockEqError> {
        while self
            .tables
            .get(self.current)
            .is_some_and(|table| table.next_block == table.profile.blocks.len())
        {
            self.current += 1;
        }
        self.tables
            .get_mut(self.current)
            .ok_or(FixedMultiAirCompleteBlockEqError::Witness(
                "too many packed Eq forest blocks",
            ))?
            .push(profile, trace)
    }

    pub fn finish(
        self,
    ) -> Result<Vec<FixedMultiAirCompletePackedBlockEqTrace>, FixedMultiAirCompleteBlockEqError>
    {
        self.tables
            .into_iter()
            .map(|table| table.finish())
            .collect()
    }
}

fn copy_state(target: &mut [[F; D_EF]; CARRY_STATES], state: &[[EF; 2]; 2]) {
    for rotation in 0..2 {
        for carry in 0..2 {
            copy_ext(&mut target[rotation * 2 + carry], state[rotation][carry]);
        }
    }
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use openvm_stark_backend::{
        air_builders::debug::check_constraints,
        warp_pesat::{
            PrismalinearMappedColumnBlock, PrismalinearMappedColumnRotation,
            PrismalinearMappedColumnTerm, PrismalinearMappedColumnWeight,
        },
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config as SC, EF, F};
    use p3_field::PrimeCharacteristicRing;
    use p3_matrix::Matrix;

    use super::*;

    fn ef4(seed: u64) -> EF {
        EF::from_basis_coefficients_slice(&[
            F::from_u64(seed),
            F::from_u64(seed + 1),
            F::from_u64(seed + 2),
            F::from_u64(seed + 3),
        ])
        .expect("EF4")
    }

    fn air(profile: FixedMultiAirCompleteBlockEqProfile) -> FixedMultiAirCompleteBlockEqAir {
        FixedMultiAirCompleteBlockEqAir {
            profile,
            global_point_bus: FixedMultiAirCompleteGlobalPointBus::new(401),
            regional_point_bus: FixedMultiAirCompleteRegionalPointBus::new(402),
            weight_bus: FixedMultiAirCompleteEqWeightBus::new(403),
        }
    }

    fn packed_air(
        profile: &FixedMultiAirCompletePackedBlockEqProfile,
    ) -> FixedMultiAirCompletePackedBlockEqAir {
        FixedMultiAirCompletePackedBlockEqAir {
            profile: Arc::new(profile.clone()),
            global_point_bus: FixedMultiAirCompleteGlobalPointBus::new(411),
            regional_point_bus: FixedMultiAirCompleteRegionalPointBus::new(412),
            weight_bus: FixedMultiAirCompleteEqWeightBus::new(413),
        }
    }

    #[test]
    fn packed_table_matches_individual_blocks_and_resets_carries() {
        let global = (0..6).map(|i| ef4(17 + 7 * i)).collect::<Vec<_>>();
        let local0 = (0..3).map(|i| ef4(101 + 11 * i)).collect::<Vec<_>>();
        let local1 = (0..2).map(|i| ef4(211 + 13 * i)).collect::<Vec<_>>();
        let profile0 = FixedMultiAirCompleteBlockEqProfile::new(
            9,
            FixedMultiAirCompleteBlockEqRole::LocalConstraint {
                constraint_ordinal: 2,
            },
            0,
            FixedMultiAirCompleteTerminalEqBlockPlan::new(global.len(), 3, local0.len(), 0)
                .unwrap(),
        )
        .unwrap();
        let profile1 = FixedMultiAirCompleteBlockEqProfile::new(
            14,
            FixedMultiAirCompleteBlockEqRole::InverseConstraint {
                interaction_ordinal: 1,
            },
            1,
            FixedMultiAirCompleteTerminalEqBlockPlan::new(global.len(), 5, local1.len(), 1)
                .unwrap(),
        )
        .unwrap();
        let packed_profile = FixedMultiAirCompletePackedBlockEqProfile::new(vec![
            profile0.clone(),
            profile1.clone(),
        ])
        .unwrap();
        let trace0 =
            generate_fixed_multi_air_complete_block_eq_trace(&profile0, &global, &local0, None)
                .unwrap();
        let trace1 =
            generate_fixed_multi_air_complete_block_eq_trace(&profile1, &global, &local1, None)
                .unwrap();
        let expected_weights = vec![trace0.weight, trace1.weight];
        let first_rows = profile0.geometry.rounds.len().max(1);
        let mut builder =
            FixedMultiAirCompletePackedBlockEqTraceBuilder::new(&packed_profile, None).unwrap();
        builder.push(&profile0, &trace0).unwrap();
        builder.push(&profile1, &trace1).unwrap();
        let packed = builder.finish().unwrap();
        assert_eq!(packed.weights, expected_weights);
        assert_eq!(packed.cached.height(), packed_profile.height());
        check_constraints::<_, SC>(
            &packed_air(&packed_profile),
            "FixedMultiAirCompletePackedBlockEqAir",
            &Some(packed.cached.as_view()),
            &[packed.common.as_view()],
            &[],
        );

        let mut cross_block = packed.common.clone();
        let width = FixedMultiAirCompleteBlockEqCols::<F>::width();
        let row: &mut FixedMultiAirCompleteBlockEqCols<F> =
            cross_block.values[first_rows * width..(first_rows + 1) * width].borrow_mut();
        row.state_before[0][0] += F::ONE;
        let rejected = catch_unwind(AssertUnwindSafe(|| {
            check_constraints::<_, SC>(
                &packed_air(&packed_profile),
                "mutated packed Eq block reset",
                &Some(packed.cached.as_view()),
                &[cross_block.as_view()],
                &[],
            );
        }));
        assert!(rejected.is_err());

        let mut descriptor = packed.cached.clone();
        let schedule_width = FixedMultiAirCompletePackedBlockEqScheduleCols::<F>::width();
        let row: &mut FixedMultiAirCompletePackedBlockEqScheduleCols<F> = descriptor.values
            [(first_rows + 1) * schedule_width..(first_rows + 2) * schedule_width]
            .borrow_mut();
        row.block += F::ONE;
        let rejected = catch_unwind(AssertUnwindSafe(|| {
            check_constraints::<_, SC>(
                &packed_air(&packed_profile),
                "mutated packed Eq descriptor",
                &Some(descriptor.as_view()),
                &[packed.common.as_view()],
                &[],
            );
        }));
        assert!(rejected.is_err());
    }

    #[test]
    fn packed_inventory_rejects_duplicate_drop_and_reorder() {
        let global = (0..5).map(|i| ef4(31 + 5 * i)).collect::<Vec<_>>();
        let local = (0..2).map(|i| ef4(71 + 3 * i)).collect::<Vec<_>>();
        let first = FixedMultiAirCompleteBlockEqProfile::new(
            3,
            FixedMultiAirCompleteBlockEqRole::LocalConstraint {
                constraint_ordinal: 0,
            },
            0,
            FixedMultiAirCompleteTerminalEqBlockPlan::new(5, 3, 2, 0).unwrap(),
        )
        .unwrap();
        let second = FixedMultiAirCompleteBlockEqProfile::new(
            4,
            FixedMultiAirCompleteBlockEqRole::InverseConstraint {
                interaction_ordinal: 0,
            },
            0,
            FixedMultiAirCompleteTerminalEqBlockPlan::new(5, 7, 2, 1).unwrap(),
        )
        .unwrap();
        assert!(
            FixedMultiAirCompletePackedBlockEqProfile::new(vec![first.clone(), first.clone()])
                .is_err()
        );
        let packed =
            FixedMultiAirCompletePackedBlockEqProfile::new(vec![first.clone(), second.clone()])
                .unwrap();
        let first_trace =
            generate_fixed_multi_air_complete_block_eq_trace(&first, &global, &local, None)
                .unwrap();
        let second_trace =
            generate_fixed_multi_air_complete_block_eq_trace(&second, &global, &local, None)
                .unwrap();
        let mut reordered =
            FixedMultiAirCompletePackedBlockEqTraceBuilder::new(&packed, None).unwrap();
        assert!(reordered.push(&second, &second_trace).is_err());
        let mut dropped =
            FixedMultiAirCompletePackedBlockEqTraceBuilder::new(&packed, None).unwrap();
        dropped.push(&first, &first_trace).unwrap();
        assert!(dropped.finish().is_err());
    }

    #[test]
    fn global_point_bridge_is_fixed_to_tau_and_setup_block_count() {
        let profile = FixedMultiAirCompleteGlobalPointBridgeProfile::new(6, 3).unwrap();
        let beta = (0..6)
            .map(|index| ef4(17 + index as u64))
            .collect::<Vec<_>>();
        let trace =
            generate_fixed_multi_air_complete_global_point_bridge_trace(profile, &beta, None)
                .unwrap();
        let air = FixedMultiAirCompleteGlobalPointBridgeAir {
            profile,
            instance_bus: FixedMultiAirCompleteInstanceValueBus::new(399),
            global_point_bus: FixedMultiAirCompleteGlobalPointBus::new(400),
        };
        check_constraints::<_, SC>(
            &air,
            "FixedMultiAirCompleteGlobalPointBridgeAir",
            &None,
            &[trace.cached.as_view(), trace.common.as_view()],
            &[],
        );

        let mut mutated = trace.cached.clone();
        let width = FixedMultiAirCompleteGlobalPointBridgeScheduleCols::<F>::width();
        let row: &mut FixedMultiAirCompleteGlobalPointBridgeScheduleCols<F> =
            mutated.values[2 * width..3 * width].borrow_mut();
        row.coordinate += F::ONE;
        let rejected = catch_unwind(AssertUnwindSafe(|| {
            check_constraints::<_, SC>(
                &air,
                "mutated complete global-point bridge",
                &None,
                &[mutated.as_view(), trace.common.as_view()],
                &[],
            );
        }));
        assert!(rejected.is_err());
        assert!(FixedMultiAirCompleteGlobalPointBridgeProfile::new(0, 3).is_err());
        assert!(FixedMultiAirCompleteGlobalPointBridgeProfile::new(6, 0).is_err());
    }

    #[test]
    fn unaligned_two_carry_matches_backend_mapped_weight_helper() {
        for (start, log_height, rotation) in [(3, 3, 0), (5, 3, 1), (17, 2, 1), (31, 0, 0)] {
            let global = (0..6).map(|i| ef4(17 + 7 * i)).collect::<Vec<_>>();
            let local = (0..log_height)
                .map(|i| ef4(101 + 11 * i as u64))
                .collect::<Vec<_>>();
            let geometry = FixedMultiAirCompleteTerminalEqBlockPlan::new(
                global.len(),
                start,
                log_height,
                rotation,
            )
            .unwrap();
            let profile = FixedMultiAirCompleteBlockEqProfile::new(
                9,
                FixedMultiAirCompleteBlockEqRole::MappedOpening {
                    global_opening_ordinal: 4,
                },
                2,
                geometry,
            )
            .unwrap();
            let trace =
                generate_fixed_multi_air_complete_block_eq_trace(&profile, &global, &local, None)
                    .unwrap();
            let backend = PrismalinearMappedColumnWeight {
                log_message_len: global.len(),
                terms: vec![PrismalinearMappedColumnTerm {
                    block: PrismalinearMappedColumnBlock {
                        start: start as usize,
                        log_height,
                    },
                    l_skip: 0,
                    barycentric_weights: vec![EF::ONE],
                    folded_row_eq_point: local,
                    rotation: if rotation == 0 {
                        PrismalinearMappedColumnRotation::Current
                    } else {
                        PrismalinearMappedColumnRotation::Next
                    },
                    scale: EF::ONE,
                }],
            }
            .evaluate_mle_at(&global)
            .unwrap();
            assert_eq!(trace.weight, backend);
            check_constraints::<_, SC>(
                &air(profile),
                "FixedMultiAirCompleteBlockEqAir",
                &None,
                &[trace.cached.as_view(), trace.common.as_view()],
                &[],
            );
        }
    }

    #[test]
    fn malformed_geometry_and_point_dimensions_are_errors() {
        let mut geometry = FixedMultiAirCompleteTerminalEqBlockPlan::new(5, 3, 2, 1).unwrap();
        geometry.rounds[0].transitions[0].global_bit ^= 1;
        assert!(FixedMultiAirCompleteBlockEqProfile::new(
            0,
            FixedMultiAirCompleteBlockEqRole::InverseConstraint {
                interaction_ordinal: 0,
            },
            0,
            geometry,
        )
        .is_err());

        let geometry = FixedMultiAirCompleteTerminalEqBlockPlan::new(5, 3, 2, 0).unwrap();
        let profile = FixedMultiAirCompleteBlockEqProfile::new(
            0,
            FixedMultiAirCompleteBlockEqRole::LocalConstraint {
                constraint_ordinal: 0,
            },
            0,
            geometry,
        )
        .unwrap();
        assert!(generate_fixed_multi_air_complete_block_eq_trace(
            &profile,
            &[EF::ONE; 4],
            &[EF::ONE; 2],
            None,
        )
        .is_err());

        let scalar = FixedMultiAirCompleteTerminalEqBlockPlan::new(5, 0, 0, 0).unwrap();
        let global = FixedMultiAirCompleteBlockEqProfile::new_with_weight_multiplicity(
            7,
            FixedMultiAirCompleteBlockEqRole::GlobalConstraint,
            0,
            scalar.clone(),
            3,
        )
        .unwrap();
        assert_eq!(global.weight_multiplicity, 3);
        assert!(
            FixedMultiAirCompleteBlockEqProfile::new_with_weight_multiplicity(
                8,
                FixedMultiAirCompleteBlockEqRole::InverseConstraint {
                    interaction_ordinal: 0,
                },
                0,
                scalar.clone(),
                2,
            )
            .is_err()
        );
        assert!(
            FixedMultiAirCompleteBlockEqProfile::new_with_weight_multiplicity(
                9,
                FixedMultiAirCompleteBlockEqRole::GlobalConstraint,
                0,
                scalar,
                0,
            )
            .is_err()
        );
    }

    #[test]
    fn state_or_weight_mutation_breaks_air_constraints() {
        let geometry = FixedMultiAirCompleteTerminalEqBlockPlan::new(6, 5, 3, 1).unwrap();
        let profile = FixedMultiAirCompleteBlockEqProfile::new(
            12,
            FixedMultiAirCompleteBlockEqRole::MappedOpening {
                global_opening_ordinal: 7,
            },
            3,
            geometry,
        )
        .unwrap();
        let global = (0..6).map(|i| ef4(3 + i)).collect::<Vec<_>>();
        let local = (0..3).map(|i| ef4(41 + i)).collect::<Vec<_>>();
        let mut trace =
            generate_fixed_multi_air_complete_block_eq_trace(&profile, &global, &local, None)
                .unwrap();
        let width = trace.common.width();
        let final_row = usize::from(profile.geometry.global_log_height) - 1;
        let cols: &mut FixedMultiAirCompleteBlockEqCols<F> =
            trace.common.values[final_row * width..(final_row + 1) * width].borrow_mut();
        cols.weight[0] += F::ONE;
        let rejected = catch_unwind(AssertUnwindSafe(|| {
            check_constraints::<_, SC>(
                &air(profile),
                "mutated complete block Eq",
                &None,
                &[trace.cached.as_view(), trace.common.as_view()],
                &[],
            );
        }));
        assert!(rejected.is_err());
    }
}

//! One canonical mapped-opening claim for the complete terminal verifier.
//!
//! This is the only owner of the native global `rho`: it consumes Ampere's
//! opening-batch receipt after every regional opening has been transcript
//! bound, replays the backend batch tag/count, and samples `rho` once. Local
//! openings precede all proof-present interaction openings exactly as listed
//! by `FixedMultiAirCompleteTerminalCircuitPlan::mapped_openings`.
//!
//! Integration dependencies:
//! - consume the exact cursor-owned local/interaction opening buses; no bridge opening table is
//!   permitted;
//! - configure each cursor opening's producer multiplicity as endpoint fanout plus this module's
//!   one lookup (the cursor profile derives that count);
//! - feed the emitted `FixedMultiAirStructuredClaimHeaderBus`, `FixedMultiAirMappedTermBus`, and
//!   `FixedMultiAirStructuredPointBus` directly into the generic structured-target and exact
//!   two-carry raw linearizer tail. `FixedMultiAirLinearizerRawComponentPlan::from_complete_plan`
//!   is the setup authority for that downstream schedule;
//! - do not restart a PCS/WHIR transcript here. The existing WHIR-prefix AIR remains the sole owner
//!   of the terminal-WHIR transcript.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder,
    interaction::InteractionBuilder,
    native_warp::{
        FixedMultiAirCompleteTerminalCircuitComponentKind,
        FixedMultiAirCompleteTerminalCircuitPlan, COMPLETE_TERMINAL_BATCH_TAG,
    },
    transcript::TranscriptLog,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{
    extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing, PrimeField32,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::{
    FixedMultiAirCompleteInteractionRegionOpeningBus,
    FixedMultiAirCompleteInteractionRegionPointBus, FixedMultiAirCompleteLocalRegionOpeningBus,
    FixedMultiAirCompleteLocalRegionPointBus, FixedMultiAirCompleteOpeningBatchStartBus,
    FixedMultiAirCompleteOpeningBatchStartMessage, FixedMultiAirCompleteRegionOpeningMessage,
    FixedMultiAirCompleteRegionPointMessage,
};
use crate::{
    bus::TranscriptBus,
    define_typed_permutation_bus,
    native_warp::terminal::fixed_multi_air::{
        FixedMultiAirMappedTermBus, FixedMultiAirMappedTermMessage,
        FixedMultiAirStructuredClaimHeaderBus, FixedMultiAirStructuredClaimHeaderMessage,
        FixedMultiAirStructuredPointBus, FixedMultiAirStructuredPointMessage,
    },
    utils::{ext_field_add, ext_field_multiply},
};

const SOURCE_KIND_COUNT: usize = 2;
const SOURCE_LOCAL: usize = 0;
const SOURCE_INTERACTION: usize = 1;
const STRUCTURED_CLAIM_ORDINAL: usize = 0;
const STRUCTURED_CLAIM_KIND_MAPPED: usize = 0;

const _: () = assert!(D_EF == 4);

/// Unique transcript seam between complete mapped-opening batching and the
/// complete terminal WHIR prefix. The receiver is owned by the aggregate
/// complete-prefix AIR; no descriptor or PCS statement is mirrored here.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FixedMultiAirCompleteWhirPrefixCursorMessage<T> {
    pub tidx: T,
}

define_typed_permutation_bus!(
    FixedMultiAirCompleteWhirPrefixCursorBus,
    FixedMultiAirCompleteWhirPrefixCursorMessage
);

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedMultiAirCompleteGlobalMappingError {
    Plan(&'static str),
    Shape(&'static str),
    Arithmetic(&'static str),
    Transcript(&'static str),
    Witness(&'static str),
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteGlobalMappingProfile {
    pub plan: Arc<FixedMultiAirCompleteTerminalCircuitPlan<F, Digest>>,
    pub log_message_len: usize,
    pub protocol_component_count: usize,
    pub opening_count: usize,
    pub point_row_count: usize,
}

impl FixedMultiAirCompleteGlobalMappingProfile {
    pub fn from_plan(
        plan: Arc<FixedMultiAirCompleteTerminalCircuitPlan<F, Digest>>,
    ) -> Result<Self, FixedMultiAirCompleteGlobalMappingError> {
        let log_message_len = usize::from(plan.metadata.code_class.log_message_len);
        if log_message_len == 0 || plan.mapped_openings.is_empty() {
            return Err(FixedMultiAirCompleteGlobalMappingError::Plan(
                "global mapping dimensions",
            ));
        }
        let message_len = 1usize.checked_shl(log_message_len as u32).ok_or(
            FixedMultiAirCompleteGlobalMappingError::Arithmetic("global message length"),
        )?;
        if usize::try_from(plan.metadata.padded_message.len).ok() != Some(message_len) {
            return Err(FixedMultiAirCompleteGlobalMappingError::Plan(
                "padded message power of two",
            ));
        }
        let mut opening_cursor = 0usize;
        let mut component_count = 0usize;
        let mut seen_interaction = false;
        let mut point_row_count = 0usize;
        for kind in [
            FixedMultiAirCompleteTerminalCircuitComponentKind::Local,
            FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction,
        ] {
            for (region, region_plan) in plan.regions.iter().enumerate() {
                let component = match kind {
                    FixedMultiAirCompleteTerminalCircuitComponentKind::Local => &region_plan.local,
                    FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction => {
                        &region_plan.interaction
                    }
                };
                if kind == FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction
                    && !component.identity.proof_present
                {
                    if component.identity.opening_count != 0
                        || component.identity.proof_ordinal.is_some()
                        || !component.expression.dynamic_columns.is_empty()
                        || !component.interactions.is_empty()
                    {
                        return Err(FixedMultiAirCompleteGlobalMappingError::Plan(
                            "empty interaction mapping",
                        ));
                    }
                    continue;
                }
                component_count = component_count.checked_add(1).ok_or(
                    FixedMultiAirCompleteGlobalMappingError::Arithmetic("component count"),
                )?;
                let component_openings = usize::try_from(component.identity.opening_count)
                    .map_err(|_| {
                        FixedMultiAirCompleteGlobalMappingError::Arithmetic(
                            "component opening count",
                        )
                    })?;
                if usize::try_from(component.identity.region_ordinal).ok() != Some(region)
                    || usize::try_from(component.identity.opening_ordinal_start).ok()
                        != Some(opening_cursor)
                    || component_openings != component.expression.dynamic_columns.len()
                {
                    return Err(FixedMultiAirCompleteGlobalMappingError::Plan(
                        "component mapped-opening identity",
                    ));
                }
                for regional_opening in 0..component_openings {
                    let mapped = plan.mapped_openings.get(opening_cursor).ok_or(
                        FixedMultiAirCompleteGlobalMappingError::Plan("mapped opening coverage"),
                    )?;
                    if mapped.component != kind
                        || usize::try_from(mapped.region_ordinal).ok() != Some(region)
                        || usize::try_from(mapped.regional_opening_ordinal).ok()
                            != Some(regional_opening)
                        || usize::try_from(mapped.global_opening_ordinal).ok()
                            != Some(opening_cursor)
                        || usize::try_from(mapped.rho_ordinal).ok() != Some(opening_cursor)
                        || component.expression.dynamic_columns.get(regional_opening)
                            != Some(&mapped.source)
                        || mapped.block.start
                            != usize::try_from(mapped.source.start).map_err(|_| {
                                FixedMultiAirCompleteGlobalMappingError::Arithmetic(
                                    "mapped block start",
                                )
                            })?
                        || mapped.block.log_height != usize::from(mapped.source.log_height)
                        || mapped.rotation.offset() != usize::from(mapped.source.rotation)
                        || mapped.eq.global_log_height as usize != log_message_len
                        || mapped.eq.start != mapped.source.start
                        || mapped.eq.log_height != mapped.source.log_height
                        || mapped.eq.rotation != mapped.source.rotation
                    {
                        return Err(FixedMultiAirCompleteGlobalMappingError::Plan(
                            "canonical mapped opening",
                        ));
                    }
                    if mapped.component
                        == FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction
                    {
                        seen_interaction = true;
                    } else if seen_interaction {
                        return Err(FixedMultiAirCompleteGlobalMappingError::Plan(
                            "local-before-interaction order",
                        ));
                    }
                    let canonical_eq = openvm_stark_backend::native_warp::FixedMultiAirCompleteTerminalEqBlockPlan::new(
                        log_message_len,
                        mapped.source.start,
                        usize::from(mapped.source.log_height),
                        usize::from(mapped.source.rotation),
                    )
                    .map_err(|_| {
                        FixedMultiAirCompleteGlobalMappingError::Plan("mapped two-carry Eq")
                    })?;
                    if canonical_eq != mapped.eq {
                        return Err(FixedMultiAirCompleteGlobalMappingError::Plan(
                            "non-canonical mapped two-carry Eq",
                        ));
                    }
                    point_row_count = point_row_count
                        .checked_add(usize::from(mapped.source.log_height))
                        .ok_or(FixedMultiAirCompleteGlobalMappingError::Arithmetic(
                            "mapped point rows",
                        ))?;
                    opening_cursor = opening_cursor.checked_add(1).ok_or(
                        FixedMultiAirCompleteGlobalMappingError::Arithmetic("opening cursor"),
                    )?;
                }
            }
        }
        if opening_cursor != plan.mapped_openings.len() {
            return Err(FixedMultiAirCompleteGlobalMappingError::Plan(
                "mapped opening tail",
            ));
        }
        validate_field_word(opening_cursor, "opening count")?;
        validate_field_word(component_count, "component count")?;
        validate_field_word(point_row_count, "point row count")?;
        Ok(Self {
            plan,
            log_message_len,
            protocol_component_count: component_count,
            opening_count: opening_cursor,
            point_row_count,
        })
    }

    /// The direct cursor producer must reserve exactly one lookup for this
    /// mapper in addition to endpoint fanout, for every mapped opening.
    #[must_use]
    pub const fn mapper_opening_lookup_count(&self) -> u32 {
        1
    }
}

fn validate_field_word(
    value: usize,
    context: &'static str,
) -> Result<(), FixedMultiAirCompleteGlobalMappingError> {
    if u32::try_from(value).is_err() || value as u64 >= u64::from(F::ORDER_U32) {
        return Err(FixedMultiAirCompleteGlobalMappingError::Arithmetic(context));
    }
    Ok(())
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteGlobalMappingScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub source_flags: [T; SOURCE_KIND_COUNT],
    pub global_opening: T,
    pub region: T,
    pub regional_opening: T,
    pub block_start: T,
    pub log_height: T,
    pub rotation: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteGlobalMappingCols<T> {
    pub batch_tidx: T,
    pub opening: [T; D_EF],
    pub rho: [T; D_EF],
    pub scale: [T; D_EF],
    pub scaled_opening: [T; D_EF],
    pub target_before: [T; D_EF],
    pub target_after: [T; D_EF],
}

/// Binds all canonical openings into one mapped-columns structured claim.
pub struct FixedMultiAirCompleteGlobalMappingAir {
    pub profile: Arc<FixedMultiAirCompleteGlobalMappingProfile>,
    pub transcript_bus: TranscriptBus,
    pub batch_start_bus: FixedMultiAirCompleteOpeningBatchStartBus,
    pub local_opening_bus: FixedMultiAirCompleteLocalRegionOpeningBus,
    pub interaction_opening_bus: FixedMultiAirCompleteInteractionRegionOpeningBus,
    pub header_bus: FixedMultiAirStructuredClaimHeaderBus,
    pub term_bus: FixedMultiAirMappedTermBus,
    pub whir_prefix_cursor_bus: FixedMultiAirCompleteWhirPrefixCursorBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteGlobalMappingAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteGlobalMappingAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirCompleteGlobalMappingScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteGlobalMappingCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteGlobalMappingAir {}
impl BaseAir<F> for FixedMultiAirCompleteGlobalMappingAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteGlobalMappingScheduleCols::<F>::width()
            + FixedMultiAirCompleteGlobalMappingCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteGlobalMappingAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("complete global-mapping schedule")
            .to_vec();
        let next_cached = builder.cached_mains()[0]
            .row_slice(1)
            .expect("next complete global-mapping schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("complete global-mapping row")
            .to_vec();
        let next_common = builder
            .common_main()
            .row_slice(1)
            .expect("next complete global-mapping row")
            .to_vec();
        let schedule: &FixedMultiAirCompleteGlobalMappingScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let next_schedule: &FixedMultiAirCompleteGlobalMappingScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FixedMultiAirCompleteGlobalMappingCols<AB::Var> = common.as_slice().borrow();
        let next: &FixedMultiAirCompleteGlobalMappingCols<AB::Var> =
            next_common.as_slice().borrow();

        for flag in schedule.source_flags.into_iter().chain([
            schedule.active,
            schedule.is_first,
            schedule.is_last,
        ]) {
            builder.assert_bool(flag);
        }
        builder.assert_eq(
            schedule
                .source_flags
                .iter()
                .fold(AB::Expr::ZERO, |sum, &flag| sum + flag),
            schedule.active,
        );
        builder.when_first_row().assert_one(schedule.active);
        builder.when_first_row().assert_one(schedule.is_first);
        builder
            .when_first_row()
            .assert_zero(schedule.global_opening);
        builder.when(schedule.active * schedule.is_last).assert_eq(
            schedule.global_opening,
            AB::Expr::from_usize(self.profile.opening_count - 1),
        );
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
        let mut continuation = transition.when(next_schedule.active);
        continuation.assert_eq(
            next_schedule.global_opening,
            schedule.global_opening + AB::Expr::ONE,
        );
        continuation.assert_eq(next.batch_tidx, local.batch_tidx);
        assert_array_eq(&mut continuation, next.rho, local.rho);
        assert_array_eq(
            &mut continuation,
            next.scale,
            ext_field_multiply::<AB::Expr>(local.scale, local.rho),
        );
        assert_array_eq(&mut continuation, next.target_before, local.target_after);
        assert_array_eq(
            &mut builder.when(schedule.is_first),
            local.scale,
            ext_one_expr::<AB>(),
        );
        assert_array_eq(
            &mut builder.when(schedule.is_first),
            local.target_before,
            core::array::from_fn(|_| AB::Expr::ZERO),
        );
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.scaled_opening,
            ext_field_multiply::<AB::Expr>(local.opening, local.scale),
        );
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.target_after,
            ext_field_add::<AB::Expr>(local.target_before, local.scaled_opening),
        );

        self.batch_start_bus.receive(
            builder,
            FixedMultiAirCompleteOpeningBatchStartMessage {
                tidx: local.batch_tidx.into(),
                protocol_component_count: AB::Expr::from_usize(
                    self.profile.protocol_component_count,
                ),
                global_opening_count: AB::Expr::from_usize(self.profile.opening_count),
            },
            schedule.is_first,
        );
        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            local.batch_tidx,
            ext_from_u64::<AB>(COMPLETE_TERMINAL_BATCH_TAG),
            schedule.is_first,
        );
        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.batch_tidx) + AB::Expr::from_usize(D_EF),
            ext_from_usize::<AB>(self.profile.opening_count),
            schedule.is_first,
        );
        self.transcript_bus.sample_ext(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.batch_tidx) + AB::Expr::from_usize(2 * D_EF),
            local.rho,
            schedule.is_first,
        );

        let opening_message = FixedMultiAirCompleteRegionOpeningMessage {
            region: schedule.region.into(),
            regional_opening: schedule.regional_opening.into(),
            global_opening: schedule.global_opening.into(),
            value: local.opening.map(Into::into),
        };
        self.local_opening_bus.lookup_key(
            builder,
            opening_message.clone(),
            schedule.source_flags[SOURCE_LOCAL],
        );
        self.interaction_opening_bus.lookup_key(
            builder,
            opening_message,
            schedule.source_flags[SOURCE_INTERACTION],
        );
        self.term_bus.send(
            builder,
            FixedMultiAirMappedTermMessage {
                claim: AB::Expr::from_usize(STRUCTURED_CLAIM_ORDINAL),
                term: schedule.global_opening.into(),
                block_start: schedule.block_start.into(),
                log_height: schedule.log_height.into(),
                l_skip: AB::Expr::ZERO,
                rotation: schedule.rotation.into(),
                scale: local.scale.map(Into::into),
            },
            schedule.active,
        );
        self.header_bus.send(
            builder,
            FixedMultiAirStructuredClaimHeaderMessage {
                claim: AB::Expr::from_usize(STRUCTURED_CLAIM_ORDINAL),
                kind: AB::Expr::from_usize(STRUCTURED_CLAIM_KIND_MAPPED),
                log_message_len: AB::Expr::from_usize(self.profile.log_message_len),
                term_count: AB::Expr::from_usize(self.profile.opening_count),
                point_len: AB::Expr::ZERO,
                target: local.target_after.map(Into::into),
            },
            schedule.is_last,
        );
        self.whir_prefix_cursor_bus.send(
            builder,
            FixedMultiAirCompleteWhirPrefixCursorMessage {
                tidx: AB::Expr::from(local.batch_tidx) + AB::Expr::from_usize(3 * D_EF),
            },
            schedule.is_last,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteGlobalMappingPointScheduleCols<T> {
    pub active: T,
    pub source_flags: [T; SOURCE_KIND_COUNT],
    pub term: T,
    pub region: T,
    pub coordinate: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirCompleteGlobalMappingPointCols<T> {
    pub value: [T; D_EF],
}

/// Routes each setup-fixed mapped term point directly from the corresponding
/// typed regional point authority into the generic structured claim.
pub struct FixedMultiAirCompleteGlobalMappingPointAir {
    pub profile: Arc<FixedMultiAirCompleteGlobalMappingProfile>,
    pub local_point_bus: FixedMultiAirCompleteLocalRegionPointBus,
    pub interaction_point_bus: FixedMultiAirCompleteInteractionRegionPointBus,
    pub structured_point_bus: FixedMultiAirStructuredPointBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirCompleteGlobalMappingPointAir {}
impl PartitionedBaseAir<F> for FixedMultiAirCompleteGlobalMappingPointAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirCompleteGlobalMappingPointScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirCompleteGlobalMappingPointCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirCompleteGlobalMappingPointAir {}
impl BaseAir<F> for FixedMultiAirCompleteGlobalMappingPointAir {
    fn width(&self) -> usize {
        FixedMultiAirCompleteGlobalMappingPointScheduleCols::<F>::width()
            + FixedMultiAirCompleteGlobalMappingPointCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirCompleteGlobalMappingPointAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("complete global-mapping point schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("complete global-mapping point row")
            .to_vec();
        let schedule: &FixedMultiAirCompleteGlobalMappingPointScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let local: &FixedMultiAirCompleteGlobalMappingPointCols<AB::Var> =
            common.as_slice().borrow();
        for flag in schedule.source_flags.into_iter().chain([schedule.active]) {
            builder.assert_bool(flag);
        }
        builder.assert_eq(
            schedule
                .source_flags
                .iter()
                .fold(AB::Expr::ZERO, |sum, &flag| sum + flag),
            schedule.active,
        );
        let point_message = FixedMultiAirCompleteRegionPointMessage {
            region: schedule.region.into(),
            coordinate: schedule.coordinate.into(),
            value: local.value.map(Into::into),
        };
        self.local_point_bus.lookup_key(
            builder,
            point_message.clone(),
            schedule.source_flags[SOURCE_LOCAL],
        );
        self.interaction_point_bus.lookup_key(
            builder,
            point_message,
            schedule.source_flags[SOURCE_INTERACTION],
        );
        self.structured_point_bus.send(
            builder,
            FixedMultiAirStructuredPointMessage {
                claim: AB::Expr::from_usize(STRUCTURED_CLAIM_ORDINAL),
                term: schedule.term.into(),
                coordinate: schedule.coordinate.into(),
                value: local.value.map(Into::into),
            },
            schedule.active,
        );
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteGlobalMappingWitness<'a> {
    pub local_openings: &'a [Vec<EF>],
    /// Empty interaction regions must be `None`; proof-present regions must be
    /// `Some` with exactly their canonical opening count.
    pub interaction_openings: &'a [Option<Vec<EF>>],
    pub local_points: &'a [Vec<EF>],
    pub interaction_points: &'a [Option<Vec<EF>>],
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirCompleteGlobalMappingTrace {
    pub mapping_cached: RowMajorMatrix<F>,
    pub mapping_common: RowMajorMatrix<F>,
    pub point_cached: RowMajorMatrix<F>,
    pub point_common: RowMajorMatrix<F>,
    pub rho: EF,
    pub target: EF,
    pub batch_end_tidx: usize,
}

pub fn generate_fixed_multi_air_complete_global_mapping_trace(
    profile: &FixedMultiAirCompleteGlobalMappingProfile,
    witness: FixedMultiAirCompleteGlobalMappingWitness<'_>,
    transcript: &TranscriptLog<F, [F; 16]>,
    batch_tidx: usize,
    mapping_required_height: Option<usize>,
    point_required_height: Option<usize>,
) -> Result<FixedMultiAirCompleteGlobalMappingTrace, FixedMultiAirCompleteGlobalMappingError> {
    if witness.local_openings.len() != profile.plan.regions.len()
        || witness.interaction_openings.len() != profile.plan.regions.len()
        || witness.local_points.len() != profile.plan.regions.len()
        || witness.interaction_points.len() != profile.plan.regions.len()
    {
        return Err(FixedMultiAirCompleteGlobalMappingError::Shape(
            "global mapping regional inputs",
        ));
    }
    expect_ext(
        transcript,
        batch_tidx,
        EF::from_u64(COMPLETE_TERMINAL_BATCH_TAG),
        false,
    )?;
    expect_ext(
        transcript,
        batch_tidx
            .checked_add(D_EF)
            .ok_or(FixedMultiAirCompleteGlobalMappingError::Arithmetic(
                "batch count transcript index",
            ))?,
        EF::from_usize(profile.opening_count),
        false,
    )?;
    let rho_tidx = batch_tidx.checked_add(2 * D_EF).ok_or(
        FixedMultiAirCompleteGlobalMappingError::Arithmetic("rho transcript index"),
    )?;
    let rho = read_ext(transcript, rho_tidx, true)?;
    let batch_end_tidx =
        rho_tidx
            .checked_add(D_EF)
            .ok_or(FixedMultiAirCompleteGlobalMappingError::Arithmetic(
                "rho transcript end",
            ))?;

    let mapping_height = admitted_height(
        profile.opening_count,
        mapping_required_height,
        "global mapping height",
    )?;
    let point_height = admitted_height(
        profile.point_row_count.max(1),
        point_required_height,
        "global mapping point height",
    )?;
    let mapping_cached_width = FixedMultiAirCompleteGlobalMappingScheduleCols::<F>::width();
    let mapping_common_width = FixedMultiAirCompleteGlobalMappingCols::<F>::width();
    let point_cached_width = FixedMultiAirCompleteGlobalMappingPointScheduleCols::<F>::width();
    let point_common_width = FixedMultiAirCompleteGlobalMappingPointCols::<F>::width();
    let mut mapping_cached = zero_cells(mapping_height, mapping_cached_width)?;
    let mut mapping_common = zero_cells(mapping_height, mapping_common_width)?;
    let mut point_cached = zero_cells(point_height, point_cached_width)?;
    let mut point_common = zero_cells(point_height, point_common_width)?;

    validate_regional_witness(profile, &witness)?;
    let mut scale = EF::ONE;
    let mut target = EF::ZERO;
    let mut point_row = 0usize;
    for (global_opening, mapped) in profile.plan.mapped_openings.iter().enumerate() {
        let region = mapped.region_ordinal as usize;
        let regional_opening = mapped.regional_opening_ordinal as usize;
        let (source, openings, point) = match mapped.component {
            FixedMultiAirCompleteTerminalCircuitComponentKind::Local => (
                SOURCE_LOCAL,
                witness.local_openings.get(region).ok_or(
                    FixedMultiAirCompleteGlobalMappingError::Shape("local opening region"),
                )?,
                witness.local_points.get(region).ok_or(
                    FixedMultiAirCompleteGlobalMappingError::Shape("local point region"),
                )?,
            ),
            FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction => (
                SOURCE_INTERACTION,
                witness
                    .interaction_openings
                    .get(region)
                    .and_then(Option::as_ref)
                    .ok_or(FixedMultiAirCompleteGlobalMappingError::Shape(
                        "interaction opening region",
                    ))?,
                witness
                    .interaction_points
                    .get(region)
                    .and_then(Option::as_ref)
                    .ok_or(FixedMultiAirCompleteGlobalMappingError::Shape(
                        "interaction point region",
                    ))?,
            ),
        };
        let opening = *openings.get(regional_opening).ok_or(
            FixedMultiAirCompleteGlobalMappingError::Shape("regional opening ordinal"),
        )?;
        if point.len() != mapped.block.log_height {
            return Err(FixedMultiAirCompleteGlobalMappingError::Shape(
                "mapped regional point",
            ));
        }
        let schedule: &mut FixedMultiAirCompleteGlobalMappingScheduleCols<F> = mapping_cached
            [global_opening * mapping_cached_width..(global_opening + 1) * mapping_cached_width]
            .borrow_mut();
        schedule.active = F::ONE;
        schedule.is_first = F::from_bool(global_opening == 0);
        schedule.is_last = F::from_bool(global_opening + 1 == profile.opening_count);
        schedule.source_flags[source] = F::ONE;
        schedule.global_opening = F::from_usize(global_opening);
        schedule.region = F::from_usize(region);
        schedule.regional_opening = F::from_usize(regional_opening);
        schedule.block_start = F::from_usize(mapped.block.start);
        schedule.log_height = F::from_usize(mapped.block.log_height);
        schedule.rotation = F::from_usize(mapped.rotation.offset());
        let cols: &mut FixedMultiAirCompleteGlobalMappingCols<F> = mapping_common
            [global_opening * mapping_common_width..(global_opening + 1) * mapping_common_width]
            .borrow_mut();
        cols.batch_tidx = F::from_usize(batch_tidx);
        copy_ext(&mut cols.opening, opening);
        copy_ext(&mut cols.rho, rho);
        copy_ext(&mut cols.scale, scale);
        let scaled = scale * opening;
        copy_ext(&mut cols.scaled_opening, scaled);
        copy_ext(&mut cols.target_before, target);
        target += scaled;
        copy_ext(&mut cols.target_after, target);

        for (coordinate, &value) in point.iter().enumerate() {
            let point_schedule: &mut FixedMultiAirCompleteGlobalMappingPointScheduleCols<F> =
                point_cached[point_row * point_cached_width..(point_row + 1) * point_cached_width]
                    .borrow_mut();
            point_schedule.active = F::ONE;
            point_schedule.source_flags[source] = F::ONE;
            point_schedule.term = F::from_usize(global_opening);
            point_schedule.region = F::from_usize(region);
            point_schedule.coordinate = F::from_usize(coordinate);
            let point_cols: &mut FixedMultiAirCompleteGlobalMappingPointCols<F> = point_common
                [point_row * point_common_width..(point_row + 1) * point_common_width]
                .borrow_mut();
            copy_ext(&mut point_cols.value, value);
            point_row += 1;
        }
        scale *= rho;
    }
    if point_row != profile.point_row_count {
        return Err(FixedMultiAirCompleteGlobalMappingError::Witness(
            "global mapped point coverage",
        ));
    }
    Ok(FixedMultiAirCompleteGlobalMappingTrace {
        mapping_cached: RowMajorMatrix::new(mapping_cached, mapping_cached_width),
        mapping_common: RowMajorMatrix::new(mapping_common, mapping_common_width),
        point_cached: RowMajorMatrix::new(point_cached, point_cached_width),
        point_common: RowMajorMatrix::new(point_common, point_common_width),
        rho,
        target,
        batch_end_tidx,
    })
}

fn validate_regional_witness(
    profile: &FixedMultiAirCompleteGlobalMappingProfile,
    witness: &FixedMultiAirCompleteGlobalMappingWitness<'_>,
) -> Result<(), FixedMultiAirCompleteGlobalMappingError> {
    for (region, plan) in profile.plan.regions.iter().enumerate() {
        if witness.local_openings[region].len() != plan.local.expression.dynamic_columns.len()
            || witness.local_points[region].len() != usize::from(plan.log_height)
        {
            return Err(FixedMultiAirCompleteGlobalMappingError::Shape(
                "local mapped witness",
            ));
        }
        match (
            plan.interaction.identity.proof_present,
            &witness.interaction_openings[region],
            &witness.interaction_points[region],
        ) {
            (true, Some(openings), Some(point))
                if openings.len() == plan.interaction.expression.dynamic_columns.len()
                    && point.len() == usize::from(plan.log_height) => {}
            (false, None, None)
                if plan.interaction.identity.opening_count == 0
                    && plan.interaction.interactions.is_empty() => {}
            _ => {
                return Err(FixedMultiAirCompleteGlobalMappingError::Shape(
                    "interaction mapped witness presence",
                ))
            }
        }
    }
    Ok(())
}

fn admitted_height(
    rows: usize,
    required: Option<usize>,
    context: &'static str,
) -> Result<usize, FixedMultiAirCompleteGlobalMappingError> {
    if rows == 0 {
        return Err(FixedMultiAirCompleteGlobalMappingError::Shape(context));
    }
    let minimum = rows
        .checked_next_power_of_two()
        .ok_or(FixedMultiAirCompleteGlobalMappingError::Arithmetic(context))?;
    match required {
        Some(height) if height.is_power_of_two() && height >= rows => Ok(height),
        Some(_) => Err(FixedMultiAirCompleteGlobalMappingError::Shape(context)),
        None => Ok(minimum),
    }
}

fn zero_cells(
    height: usize,
    width: usize,
) -> Result<Vec<F>, FixedMultiAirCompleteGlobalMappingError> {
    let len =
        height
            .checked_mul(width)
            .ok_or(FixedMultiAirCompleteGlobalMappingError::Arithmetic(
                "global mapping trace cells",
            ))?;
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| FixedMultiAirCompleteGlobalMappingError::Arithmetic("trace allocation"))?;
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

fn ext_from_u64<AB: AirBuilder>(value: u64) -> [AB::Expr; D_EF] {
    core::array::from_fn(|coordinate| {
        if coordinate == 0 {
            AB::Expr::from_u64(value)
        } else {
            AB::Expr::ZERO
        }
    })
}

fn ext_from_usize<AB: AirBuilder>(value: usize) -> [AB::Expr; D_EF] {
    core::array::from_fn(|coordinate| {
        if coordinate == 0 {
            AB::Expr::from_usize(value)
        } else {
            AB::Expr::ZERO
        }
    })
}

fn read_ext(
    transcript: &TranscriptLog<F, [F; 16]>,
    tidx: usize,
    sampled: bool,
) -> Result<EF, FixedMultiAirCompleteGlobalMappingError> {
    let end = tidx
        .checked_add(D_EF)
        .ok_or(FixedMultiAirCompleteGlobalMappingError::Arithmetic(
            "transcript extension endpoint",
        ))?;
    let values = transcript.values().get(tidx..end).ok_or(
        FixedMultiAirCompleteGlobalMappingError::Transcript("extension value"),
    )?;
    let flags = transcript.samples().get(tidx..end).ok_or(
        FixedMultiAirCompleteGlobalMappingError::Transcript("extension sample flags"),
    )?;
    if flags.iter().any(|&flag| flag != sampled) {
        return Err(FixedMultiAirCompleteGlobalMappingError::Transcript(
            "extension sample kind",
        ));
    }
    EF::from_basis_coefficients_slice(values).ok_or(
        FixedMultiAirCompleteGlobalMappingError::Transcript("extension coordinates"),
    )
}

fn expect_ext(
    transcript: &TranscriptLog<F, [F; 16]>,
    tidx: usize,
    expected: EF,
    sampled: bool,
) -> Result<(), FixedMultiAirCompleteGlobalMappingError> {
    if read_ext(transcript, tidx, sampled)? != expected {
        return Err(FixedMultiAirCompleteGlobalMappingError::Transcript(
            "extension mismatch",
        ));
    }
    Ok(())
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

#[cfg(test)]
mod tests {
    use std::{
        panic::{catch_unwind, AssertUnwindSafe},
        sync::Arc,
    };

    use openvm_stark_backend::{
        air_builders::{
            debug::{check_constraints, check_logup},
            symbolic::{get_symbolic_builder, SymbolicRapBuilder},
        },
        interaction::{InteractionBuilder, SymbolicInteraction},
        keygen::types::TraceWidth,
        warp_pesat::PrismalinearMappedColumnRotation,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config as SC;

    use super::*;
    use crate::native_warp::terminal::fixed_multi_air_complete::{
        interaction_endpoint::tests::backend_plan,
        region_cursor_openings::FixedMultiAirCompleteRegionCursorOpeningProfile,
    };

    fn ef(seed: u64) -> EF {
        EF::from_basis_coefficients_fn(|coordinate| F::from_u64(seed + coordinate as u64 * 17))
    }

    struct OwnedWitness {
        local_openings: Vec<Vec<EF>>,
        interaction_openings: Vec<Option<Vec<EF>>>,
        local_points: Vec<Vec<EF>>,
        interaction_points: Vec<Option<Vec<EF>>>,
    }

    impl OwnedWitness {
        fn view(&self) -> FixedMultiAirCompleteGlobalMappingWitness<'_> {
            FixedMultiAirCompleteGlobalMappingWitness {
                local_openings: &self.local_openings,
                interaction_openings: &self.interaction_openings,
                local_points: &self.local_points,
                interaction_points: &self.interaction_points,
            }
        }
    }

    fn witness(profile: &FixedMultiAirCompleteGlobalMappingProfile) -> OwnedWitness {
        let mut local_openings = Vec::new();
        let mut interaction_openings = Vec::new();
        let mut local_points = Vec::new();
        let mut interaction_points = Vec::new();
        for (region, plan) in profile.plan.regions.iter().enumerate() {
            local_openings.push(
                (0..plan.local.expression.dynamic_columns.len())
                    .map(|opening| ef(1000 + region as u64 * 100 + opening as u64))
                    .collect(),
            );
            local_points.push(
                (0..usize::from(plan.log_height))
                    .map(|coordinate| ef(2000 + region as u64 * 100 + coordinate as u64))
                    .collect(),
            );
            if plan.interaction.identity.proof_present {
                interaction_openings.push(Some(
                    (0..plan.interaction.expression.dynamic_columns.len())
                        .map(|opening| ef(3000 + region as u64 * 100 + opening as u64))
                        .collect(),
                ));
                interaction_points.push(Some(
                    (0..usize::from(plan.log_height))
                        .map(|coordinate| ef(4000 + region as u64 * 100 + coordinate as u64))
                        .collect(),
                ));
            } else {
                interaction_openings.push(None);
                interaction_points.push(None);
            }
        }
        OwnedWitness {
            local_openings,
            interaction_openings,
            local_points,
            interaction_points,
        }
    }

    fn transcript(opening_count: usize, rho: EF) -> TranscriptLog<F, [F; 16]> {
        let mut values = Vec::new();
        let mut samples = Vec::new();
        for (value, sampled) in [
            (EF::from_u64(COMPLETE_TERMINAL_BATCH_TAG), false),
            (EF::from_usize(opening_count), false),
            (rho, true),
        ] {
            values.extend_from_slice(value.as_basis_coefficients_slice());
            samples.extend(core::iter::repeat_n(sampled, D_EF));
        }
        TranscriptLog::new(values, samples)
    }

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct WhirPrefixCursorReceiverCols<T> {
        active: T,
        tidx: T,
    }

    struct WhirPrefixCursorReceiverAir {
        bus: FixedMultiAirCompleteWhirPrefixCursorBus,
    }

    impl BaseAir<F> for WhirPrefixCursorReceiverAir {
        fn width(&self) -> usize {
            WhirPrefixCursorReceiverCols::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for WhirPrefixCursorReceiverAir {}
    impl PartitionedBaseAir<F> for WhirPrefixCursorReceiverAir {}
    impl<AB> Air<AB> for WhirPrefixCursorReceiverAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("WHIR-prefix cursor receiver");
            let local: &WhirPrefixCursorReceiverCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.active);
            self.bus.receive(
                builder,
                FixedMultiAirCompleteWhirPrefixCursorMessage {
                    tidx: local.tidx.into(),
                },
                local.active,
            );
        }
    }

    fn receiver_trace(tidx: usize) -> RowMajorMatrix<F> {
        let width = WhirPrefixCursorReceiverCols::<F>::width();
        let mut values = vec![F::ZERO; width];
        let cols: &mut WhirPrefixCursorReceiverCols<F> = values.as_mut_slice().borrow_mut();
        cols.active = F::ONE;
        cols.tidx = F::from_usize(tidx);
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

    fn check_whir_cursor_balance(
        mapping_air: &FixedMultiAirCompleteGlobalMappingAir,
        trace: &FixedMultiAirCompleteGlobalMappingTrace,
        receiver_air: &WhirPrefixCursorReceiverAir,
        receiver: &RowMajorMatrix<F>,
    ) {
        const WHIR_CURSOR_BUS_INDEX: u16 = 1107;
        let mapping_interactions = symbolic_interactions(mapping_air)
            .into_iter()
            .filter(|interaction| interaction.bus_index == WHIR_CURSOR_BUS_INDEX)
            .collect::<Vec<_>>();
        let receiver_interactions = symbolic_interactions(receiver_air);
        assert_eq!(mapping_interactions.len(), 1);
        assert_eq!(receiver_interactions.len(), 1);
        let names = vec![
            "global mapping cursor".to_string(),
            "complete WHIR prefix receiver".to_string(),
        ];
        check_logup(
            &names,
            &[mapping_interactions, receiver_interactions],
            &[None, None],
            &[
                vec![
                    trace.mapping_cached.as_view(),
                    trace.mapping_common.as_view(),
                ],
                vec![receiver.as_view()],
            ],
            &[Vec::new(), Vec::new()],
        );
    }

    #[test]
    fn canonical_local_then_nonempty_interaction_order_and_single_rho_match() {
        let profile = Arc::new(
            FixedMultiAirCompleteGlobalMappingProfile::from_plan(backend_plan())
                .expect("global mapping profile"),
        );
        let cursor =
            FixedMultiAirCompleteRegionCursorOpeningProfile::from_plan(profile.plan.clone())
                .expect("cursor profile");
        assert_eq!(cursor.global_opening_count, profile.opening_count);
        let mut mapped_counts = vec![0usize; profile.opening_count];
        for mapped in &profile.plan.mapped_openings {
            mapped_counts[mapped.global_opening_ordinal as usize] += 1;
        }
        assert!(mapped_counts.iter().all(|&count| count == 1));
        for component in &cursor.components {
            let mapper_point_uses = profile
                .plan
                .mapped_openings
                .iter()
                .filter(|mapped| {
                    mapped.component == component.kind
                        && mapped.region_ordinal as usize == component.region
                })
                .count();
            assert_eq!(mapper_point_uses, component.opening_count);
            assert_eq!(
                component.point_common_count,
                1 + mapper_point_uses,
                "one endpoint selector and one mapper point lookup per opening"
            );
            assert!(component
                .opening_lookup_counts
                .iter()
                .all(|&count| count >= profile.mapper_opening_lookup_count()));
        }
        assert!(profile
            .plan
            .regions
            .iter()
            .any(|region| !region.interaction.identity.proof_present));
        let first_interaction = profile
            .plan
            .mapped_openings
            .iter()
            .position(|mapped| {
                mapped.component == FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction
            })
            .expect("interaction mappings");
        assert!(profile.plan.mapped_openings[..first_interaction]
            .iter()
            .all(|mapped| {
                mapped.component == FixedMultiAirCompleteTerminalCircuitComponentKind::Local
            }));
        assert!(profile.plan.mapped_openings[first_interaction..]
            .iter()
            .all(|mapped| {
                mapped.component == FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction
            }));

        let witness = witness(&profile);
        let rho = ef(77);
        let transcript = transcript(profile.opening_count, rho);
        let trace = generate_fixed_multi_air_complete_global_mapping_trace(
            &profile,
            witness.view(),
            &transcript,
            0,
            None,
            None,
        )
        .expect("global mapping trace");
        let expected = profile.plan.mapped_openings.iter().fold(
            (EF::ONE, EF::ZERO),
            |(scale, target), mapped| {
                let opening = match mapped.component {
                    FixedMultiAirCompleteTerminalCircuitComponentKind::Local => {
                        witness.local_openings[mapped.region_ordinal as usize]
                            [mapped.regional_opening_ordinal as usize]
                    }
                    FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction => witness
                        .interaction_openings[mapped.region_ordinal as usize]
                        .as_ref()
                        .expect("present interaction")
                        [mapped.regional_opening_ordinal as usize],
                };
                (scale * rho, target + scale * opening)
            },
        );
        assert_eq!(trace.rho, rho);
        assert_eq!(trace.target, expected.1);
        assert_eq!(trace.batch_end_tidx, 3 * D_EF);

        let air = FixedMultiAirCompleteGlobalMappingAir {
            profile: profile.clone(),
            transcript_bus: TranscriptBus::new(1101),
            batch_start_bus: FixedMultiAirCompleteOpeningBatchStartBus::new(1102),
            local_opening_bus: FixedMultiAirCompleteLocalRegionOpeningBus::new(1103),
            interaction_opening_bus: FixedMultiAirCompleteInteractionRegionOpeningBus::new(1104),
            header_bus: FixedMultiAirStructuredClaimHeaderBus::new(1105),
            term_bus: FixedMultiAirMappedTermBus::new(1106),
            whir_prefix_cursor_bus: FixedMultiAirCompleteWhirPrefixCursorBus::new(1107),
        };
        check_constraints::<_, SC>(
            &air,
            "complete global mapping",
            &None,
            &[
                trace.mapping_cached.as_view(),
                trace.mapping_common.as_view(),
            ],
            &[],
        );
        let point_air = FixedMultiAirCompleteGlobalMappingPointAir {
            profile: profile.clone(),
            local_point_bus: FixedMultiAirCompleteLocalRegionPointBus::new(1108),
            interaction_point_bus: FixedMultiAirCompleteInteractionRegionPointBus::new(1109),
            structured_point_bus: FixedMultiAirStructuredPointBus::new(1110),
        };
        check_constraints::<_, SC>(
            &point_air,
            "complete global mapping points",
            &None,
            &[trace.point_cached.as_view(), trace.point_common.as_view()],
            &[],
        );

        let receiver_air = WhirPrefixCursorReceiverAir {
            bus: FixedMultiAirCompleteWhirPrefixCursorBus::new(1107),
        };
        let receiver = receiver_trace(trace.batch_end_tidx);
        check_whir_cursor_balance(&air, &trace, &receiver_air, &receiver);
        let mut wrong_cursor = receiver.clone();
        let cursor: &mut WhirPrefixCursorReceiverCols<F> =
            wrong_cursor.values.as_mut_slice().borrow_mut();
        cursor.tidx += F::ONE;
        assert!(catch_unwind(AssertUnwindSafe(|| {
            check_whir_cursor_balance(&air, &trace, &receiver_air, &wrong_cursor);
        }))
        .is_err());

        let mut wrong_target = trace.mapping_common.clone();
        let width = FixedMultiAirCompleteGlobalMappingCols::<F>::width();
        let final_row = profile.opening_count - 1;
        let cols: &mut FixedMultiAirCompleteGlobalMappingCols<F> =
            wrong_target.values[final_row * width..(final_row + 1) * width].borrow_mut();
        cols.target_after[0] += F::ONE;
        assert!(catch_unwind(AssertUnwindSafe(|| {
            check_constraints::<_, SC>(
                &air,
                "mutated complete global mapping",
                &None,
                &[trace.mapping_cached.as_view(), wrong_target.as_view()],
                &[],
            );
        }))
        .is_err());
    }

    #[test]
    fn cross_kind_order_rho_rotation_and_empty_presence_mutations_reject() {
        let honest = backend_plan();
        let mut cross_kind = (*honest).clone();
        cross_kind.mapped_openings[0].component =
            FixedMultiAirCompleteTerminalCircuitComponentKind::Interaction;
        assert!(
            FixedMultiAirCompleteGlobalMappingProfile::from_plan(Arc::new(cross_kind)).is_err()
        );

        let mut order = (*honest).clone();
        order.mapped_openings.swap(0, 1);
        assert!(FixedMultiAirCompleteGlobalMappingProfile::from_plan(Arc::new(order)).is_err());

        let mut rho = (*honest).clone();
        rho.mapped_openings[1].rho_ordinal = 0;
        assert!(FixedMultiAirCompleteGlobalMappingProfile::from_plan(Arc::new(rho)).is_err());

        let mut rotation = (*honest).clone();
        rotation.mapped_openings[0].rotation = match rotation.mapped_openings[0].rotation {
            PrismalinearMappedColumnRotation::Current => PrismalinearMappedColumnRotation::Next,
            PrismalinearMappedColumnRotation::Next => PrismalinearMappedColumnRotation::Current,
        };
        assert!(FixedMultiAirCompleteGlobalMappingProfile::from_plan(Arc::new(rotation)).is_err());

        let profile =
            FixedMultiAirCompleteGlobalMappingProfile::from_plan(honest).expect("honest profile");
        let mut bad_witness = witness(&profile);
        let empty_region = profile
            .plan
            .regions
            .iter()
            .position(|region| !region.interaction.identity.proof_present)
            .expect("empty interaction region");
        bad_witness.interaction_openings[empty_region] = Some(Vec::new());
        assert!(generate_fixed_multi_air_complete_global_mapping_trace(
            &profile,
            bad_witness.view(),
            &transcript(profile.opening_count, ef(99)),
            0,
            None,
            None,
        )
        .is_err());
    }
}

//! Exact serialized constrained-RS statement between the complete nonlinear
//! relation and the coefficient-two-coset WHIR prefix.
//!
//! The native terminal verifier reconstructs one compact mapped-column claim,
//! absorbs every field of that claim, and only then starts terminal WHIR.  This
//! AIR is the algebraic relay for exactly that boundary: it consumes the typed
//! claim emitted by the complete relation, observes its canonical wire image,
//! and republishes the same typed values to the generic structured-adjoint
//! tail.  No proof-supplied descriptor or host verdict can bypass the relay.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_recursion_circuit::{
    bus::TranscriptBus,
    define_typed_permutation_bus,
    native_warp::terminal::{
        FixedMultiAirCompleteWhirPrefixCursorBus, FixedMultiAirCompleteWhirPrefixCursorMessage,
        FixedMultiAirMappedTermBus, FixedMultiAirMappedTermMessage,
        FixedMultiAirStructuredClaimHeaderBus, FixedMultiAirStructuredClaimHeaderMessage,
        FixedMultiAirStructuredPointBus, FixedMultiAirStructuredPointMessage,
    },
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder,
    interaction::InteractionBuilder,
    native_warp::{
        FixedMultiAirCompleteTerminalCircuitPlan, FINITE_COMPLETE_TERMINAL_RS_STATEMENT_TAG,
    },
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField32},
    transcript::TranscriptLog,
    warp_accum::TerminalConstrainedRsStatement,
    warp_pesat::TerminalWeightSpec,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

#[cfg(test)]
#[path = "terminal_rs_statement_tests.rs"]
mod tests;

const SOURCE_COUNT: usize = 4;
const SOURCE_CONSTANT: usize = 0;
const SOURCE_TARGET: usize = 1;
const SOURCE_SCALE: usize = 2;
const SOURCE_POINT: usize = 3;
const CLAIM_ORDINAL: usize = 0;
/// Internal complete-terminal typed-bus discriminator.
const CLAIM_BUS_KIND_MAPPED: usize = 0;
/// Canonical SDK transcript discriminator for
/// `TerminalWeightSpec::PrismalinearMappedColumns` (`Eq` is zero).
const CLAIM_TRANSCRIPT_KIND_MAPPED: usize = 1;

const _: () = assert!(D_EF == 4);

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct FiniteWarpV3RsStatementEndMessage<T> {
    pub tidx: T,
}

define_typed_permutation_bus!(
    FiniteWarpV3RsStatementEndBus,
    FiniteWarpV3RsStatementEndMessage
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RsSource {
    Constant(F),
    Target {
        limb: usize,
    },
    Scale {
        term: usize,
        limb: usize,
    },
    Point {
        term: usize,
        coordinate: usize,
        limb: usize,
    },
}

impl RsSource {
    const fn source(self) -> usize {
        match self {
            Self::Constant(_) => SOURCE_CONSTANT,
            Self::Target { .. } => SOURCE_TARGET,
            Self::Scale { .. } => SOURCE_SCALE,
            Self::Point { .. } => SOURCE_POINT,
        }
    }

    const fn limb(self) -> usize {
        match self {
            Self::Constant(_) => 0,
            Self::Target { limb } | Self::Scale { limb, .. } | Self::Point { limb, .. } => limb,
        }
    }

    const fn term(self) -> usize {
        match self {
            Self::Scale { term, .. } | Self::Point { term, .. } => term,
            Self::Constant(_) | Self::Target { .. } => 0,
        }
    }

    const fn coordinate(self) -> usize {
        match self {
            Self::Point { coordinate, .. } => coordinate,
            Self::Constant(_) | Self::Target { .. } | Self::Scale { .. } => 0,
        }
    }

    const fn is_group_start(self) -> bool {
        !matches!(self, Self::Constant(_)) && self.limb() == 0
    }

    const fn carries_to_next(self) -> bool {
        !matches!(self, Self::Constant(_)) && self.limb() + 1 < D_EF
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FiniteWarpV3RsStatementError {
    Profile(&'static str),
    Statement(&'static str),
    Transcript,
    Height,
    Arithmetic,
}

#[derive(Clone, Debug)]
pub struct FiniteWarpV3RsStatementProfile {
    pub plan: Arc<FixedMultiAirCompleteTerminalCircuitPlan<F, Digest>>,
    schedule: Arc<[RsSource]>,
}

impl FiniteWarpV3RsStatementProfile {
    pub fn new(
        plan: Arc<FixedMultiAirCompleteTerminalCircuitPlan<F, Digest>>,
    ) -> Result<Self, FiniteWarpV3RsStatementError> {
        let log_message_len = usize::from(plan.metadata.code_class.log_message_len);
        if log_message_len == 0 || plan.mapped_openings.is_empty() {
            return Err(FiniteWarpV3RsStatementError::Profile("mapped claim shape"));
        }
        let mut schedule = Vec::new();
        push_u64(&mut schedule, FINITE_COMPLETE_TERMINAL_RS_STATEMENT_TAG);
        push_u64(&mut schedule, 1); // exactly one complete-relation claim
        push_u64(&mut schedule, CLAIM_TRANSCRIPT_KIND_MAPPED as u64);
        push_u64(&mut schedule, log_message_len as u64);
        push_u64(&mut schedule, plan.mapped_openings.len() as u64);
        for (term, mapped) in plan.mapped_openings.iter().enumerate() {
            if mapped.block.start
                != usize::try_from(mapped.source.start)
                    .map_err(|_| FiniteWarpV3RsStatementError::Profile("mapped block start"))?
                || mapped.block.log_height != usize::from(mapped.source.log_height)
                || mapped.rotation.offset() != usize::from(mapped.source.rotation)
            {
                return Err(FiniteWarpV3RsStatementError::Profile(
                    "non-canonical mapped term",
                ));
            }
            push_u64(&mut schedule, mapped.block.start as u64);
            push_u64(&mut schedule, mapped.block.log_height as u64);
            push_u64(&mut schedule, 0); // direct row-opening form: l_skip = 0
            push_u64(&mut schedule, mapped.rotation.offset() as u64);
            push_u64(&mut schedule, 1); // one barycentric weight
            push_ext_constant(&mut schedule, EF::ONE);
            push_u64(&mut schedule, mapped.block.log_height as u64);
            for coordinate in 0..mapped.block.log_height {
                push_point(&mut schedule, term, coordinate);
            }
            push_scale(&mut schedule, term);
        }
        push_target(&mut schedule);
        if schedule.is_empty() || schedule.len() > u32::MAX as usize {
            return Err(FiniteWarpV3RsStatementError::Arithmetic);
        }
        Ok(Self {
            plan,
            schedule: schedule.into(),
        })
    }

    #[must_use]
    pub fn observation_len(&self) -> usize {
        self.schedule.len()
    }

    #[must_use]
    pub fn log_message_len(&self) -> usize {
        usize::from(self.plan.metadata.code_class.log_message_len)
    }

    #[must_use]
    pub fn term_count(&self) -> usize {
        self.plan.mapped_openings.len()
    }
}

fn push_u64(schedule: &mut Vec<RsSource>, value: u64) {
    for shift in [0, 16, 32, 48] {
        schedule.push(RsSource::Constant(F::from_u32(
            ((value >> shift) & 0xffff) as u32,
        )));
    }
}

fn push_ext_constant(schedule: &mut Vec<RsSource>, value: EF) {
    schedule.extend(
        value
            .as_basis_coefficients_slice()
            .iter()
            .copied()
            .map(RsSource::Constant),
    );
}

fn push_target(schedule: &mut Vec<RsSource>) {
    schedule.extend((0..D_EF).map(|limb| RsSource::Target { limb }));
}

fn push_scale(schedule: &mut Vec<RsSource>, term: usize) {
    schedule.extend((0..D_EF).map(|limb| RsSource::Scale { term, limb }));
}

fn push_point(schedule: &mut Vec<RsSource>, term: usize, coordinate: usize) {
    schedule.extend((0..D_EF).map(|limb| RsSource::Point {
        term,
        coordinate,
        limb,
    }));
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FiniteWarpV3RsStatementScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub source_flags: [T; SOURCE_COUNT],
    pub limb_flags: [T; D_EF],
    pub group_start: T,
    pub carries_to_next: T,
    pub term: T,
    pub coordinate: T,
    pub limb: T,
    pub block_start: T,
    pub log_height: T,
    pub rotation: T,
    pub constant: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FiniteWarpV3RsStatementCols<T> {
    pub tidx: T,
    pub value: [T; D_EF],
}

pub struct FiniteWarpV3RsStatementAir {
    pub profile: Arc<FiniteWarpV3RsStatementProfile>,
    pub transcript_bus: TranscriptBus,
    pub source_cursor_bus: FixedMultiAirCompleteWhirPrefixCursorBus,
    pub end_cursor_bus: FiniteWarpV3RsStatementEndBus,
    pub source_header_bus: FixedMultiAirStructuredClaimHeaderBus,
    pub source_term_bus: FixedMultiAirMappedTermBus,
    pub source_point_bus: FixedMultiAirStructuredPointBus,
    pub output_header_bus: FixedMultiAirStructuredClaimHeaderBus,
    pub output_term_bus: FixedMultiAirMappedTermBus,
    pub output_point_bus: FixedMultiAirStructuredPointBus,
}

impl BaseAirWithPublicValues<F> for FiniteWarpV3RsStatementAir {}
impl PartitionedBaseAir<F> for FiniteWarpV3RsStatementAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FiniteWarpV3RsStatementScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FiniteWarpV3RsStatementCols::<F>::width()
    }
}
impl BaseAir<F> for FiniteWarpV3RsStatementAir {
    fn width(&self) -> usize {
        FiniteWarpV3RsStatementScheduleCols::<F>::width()
            + FiniteWarpV3RsStatementCols::<F>::width()
    }
}

impl<AB> Air<AB> for FiniteWarpV3RsStatementAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("RS statement schedule")
            .to_vec();
        let next_cached = builder.cached_mains()[0]
            .row_slice(1)
            .expect("next RS statement schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("RS statement row")
            .to_vec();
        let next_common = builder
            .common_main()
            .row_slice(1)
            .expect("next RS statement row")
            .to_vec();
        let schedule: &FiniteWarpV3RsStatementScheduleCols<AB::Var> = cached.as_slice().borrow();
        let next_schedule: &FiniteWarpV3RsStatementScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FiniteWarpV3RsStatementCols<AB::Var> = common.as_slice().borrow();
        let next: &FiniteWarpV3RsStatementCols<AB::Var> = next_common.as_slice().borrow();

        for flag in schedule
            .source_flags
            .into_iter()
            .chain([
                schedule.active,
                schedule.is_first,
                schedule.is_last,
                schedule.group_start,
                schedule.carries_to_next,
            ])
            .chain(schedule.limb_flags)
        {
            builder.assert_bool(flag);
        }
        builder.assert_eq(
            schedule
                .source_flags
                .iter()
                .fold(AB::Expr::ZERO, |sum, &flag| sum + flag),
            schedule.active,
        );
        builder.assert_eq(
            schedule
                .limb_flags
                .into_iter()
                .fold(AB::Expr::ZERO, |sum, flag| sum + flag),
            schedule.source_flags[SOURCE_TARGET]
                + schedule.source_flags[SOURCE_SCALE]
                + schedule.source_flags[SOURCE_POINT],
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
            .when_transition()
            .when(next_schedule.active)
            .assert_eq(next.tidx, local.tidx + AB::F::ONE);
        let mut transition = builder.when_transition();
        let mut carry = transition.when(schedule.carries_to_next);
        for limb in 0..D_EF {
            carry.assert_eq(next.value[limb], local.value[limb]);
        }

        let [is_constant, is_target, is_scale, is_point] =
            schedule.source_flags.map(AB::Expr::from);
        let selected = (0..D_EF)
            .map(|limb| schedule.limb_flags[limb] * local.value[limb])
            .fold(AB::Expr::ZERO, |sum, value| sum + value);
        self.transcript_bus.observe(
            builder,
            AB::Expr::ZERO,
            local.tidx,
            is_constant * schedule.constant
                + (is_target.clone() + is_scale.clone() + is_point.clone()) * selected,
            schedule.active,
        );

        self.source_cursor_bus.receive(
            builder,
            FixedMultiAirCompleteWhirPrefixCursorMessage {
                tidx: local.tidx.into(),
            },
            schedule.is_first,
        );
        self.end_cursor_bus.send(
            builder,
            FiniteWarpV3RsStatementEndMessage {
                tidx: AB::Expr::from(local.tidx) + AB::Expr::ONE,
            },
            schedule.is_last,
        );

        let target_enabled = schedule.group_start * schedule.source_flags[SOURCE_TARGET];
        let target_message = FixedMultiAirStructuredClaimHeaderMessage {
            claim: AB::Expr::from_usize(CLAIM_ORDINAL),
            kind: AB::Expr::from_usize(CLAIM_BUS_KIND_MAPPED),
            log_message_len: AB::Expr::from_usize(self.profile.log_message_len()),
            term_count: AB::Expr::from_usize(self.profile.term_count()),
            point_len: AB::Expr::ZERO,
            target: local.value.map(Into::into),
        };
        self.source_header_bus
            .receive(builder, target_message.clone(), target_enabled.clone());
        self.output_header_bus
            .send(builder, target_message, target_enabled);

        let scale_enabled = schedule.group_start * schedule.source_flags[SOURCE_SCALE];
        let scale_message = FixedMultiAirMappedTermMessage {
            claim: AB::Expr::from_usize(CLAIM_ORDINAL),
            term: schedule.term.into(),
            block_start: schedule.block_start.into(),
            log_height: schedule.log_height.into(),
            l_skip: AB::Expr::ZERO,
            rotation: schedule.rotation.into(),
            scale: local.value.map(Into::into),
        };
        self.source_term_bus
            .receive(builder, scale_message.clone(), scale_enabled.clone());
        self.output_term_bus
            .send(builder, scale_message, scale_enabled);

        let point_enabled = schedule.group_start * schedule.source_flags[SOURCE_POINT];
        let point_message = FixedMultiAirStructuredPointMessage {
            claim: AB::Expr::from_usize(CLAIM_ORDINAL),
            term: schedule.term.into(),
            coordinate: schedule.coordinate.into(),
            value: local.value.map(Into::into),
        };
        self.source_point_bus
            .receive(builder, point_message.clone(), point_enabled.clone());
        self.output_point_bus
            .send(builder, point_message, point_enabled);
    }
}

#[derive(Clone, Debug)]
pub struct FiniteWarpV3RsStatementTraceData {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub end_tidx: usize,
}

pub fn generate_finite_warp_v3_rs_statement_trace(
    profile: &FiniteWarpV3RsStatementProfile,
    statement: &TerminalConstrainedRsStatement<EF>,
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
    required_height: Option<usize>,
) -> Result<FiniteWarpV3RsStatementTraceData, FiniteWarpV3RsStatementError> {
    let claim = statement
        .linearizer_claims
        .as_slice()
        .first()
        .filter(|_| statement.linearizer_claims.len() == 1)
        .ok_or(FiniteWarpV3RsStatementError::Statement("claim count"))?;
    let TerminalWeightSpec::PrismalinearMappedColumns(mapped) = &claim.weight else {
        return Err(FiniteWarpV3RsStatementError::Statement("claim kind"));
    };
    if mapped.log_message_len != profile.log_message_len()
        || mapped.terms.len() != profile.term_count()
    {
        return Err(FiniteWarpV3RsStatementError::Statement("mapped shape"));
    }
    for (term, (actual, expected)) in mapped
        .terms
        .iter()
        .zip(profile.plan.mapped_openings.iter())
        .enumerate()
    {
        if actual.block != expected.block
            || actual.l_skip != 0
            || actual.rotation != expected.rotation
            || actual.barycentric_weights.as_slice() != [EF::ONE]
            || actual.folded_row_eq_point.len() != expected.block.log_height
            || term >= profile.term_count()
        {
            return Err(FiniteWarpV3RsStatementError::Statement("mapped term"));
        }
    }

    let end_tidx = start_tidx
        .checked_add(profile.schedule.len())
        .ok_or(FiniteWarpV3RsStatementError::Arithmetic)?;
    if end_tidx >= F::ORDER_U32 as usize {
        return Err(FiniteWarpV3RsStatementError::Arithmetic);
    }
    let transcript_values = transcript
        .values()
        .get(start_tidx..end_tidx)
        .ok_or(FiniteWarpV3RsStatementError::Transcript)?;
    if transcript
        .samples()
        .get(start_tidx..end_tidx)
        .is_none_or(|samples| samples.iter().any(|&sample| sample))
    {
        return Err(FiniteWarpV3RsStatementError::Transcript);
    }
    let valid_rows = profile.schedule.len();
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows || !height.is_power_of_two() {
        return Err(FiniteWarpV3RsStatementError::Height);
    }
    let cached_width = FiniteWarpV3RsStatementScheduleCols::<F>::width();
    let common_width = FiniteWarpV3RsStatementCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    for (row, source) in profile.schedule.iter().copied().enumerate() {
        let schedule: &mut FiniteWarpV3RsStatementScheduleCols<F> =
            cached[row * cached_width..(row + 1) * cached_width].borrow_mut();
        schedule.active = F::ONE;
        schedule.is_first = F::from_bool(row == 0);
        schedule.is_last = F::from_bool(row + 1 == valid_rows);
        schedule.source_flags[source.source()] = F::ONE;
        schedule.group_start = F::from_bool(source.is_group_start());
        schedule.carries_to_next = F::from_bool(source.carries_to_next());
        schedule.term = F::from_usize(source.term());
        schedule.coordinate = F::from_usize(source.coordinate());
        schedule.limb = F::from_usize(source.limb());
        if !matches!(source, RsSource::Constant(_)) {
            schedule.limb_flags[source.limb()] = F::ONE;
        }
        if let RsSource::Constant(value) = source {
            schedule.constant = value;
        }
        if matches!(source, RsSource::Scale { .. }) {
            let expected = &profile.plan.mapped_openings[source.term()];
            schedule.block_start = F::from_usize(expected.block.start);
            schedule.log_height = F::from_usize(expected.block.log_height);
            schedule.rotation = F::from_usize(expected.rotation.offset());
        }

        let local: &mut FiniteWarpV3RsStatementCols<F> =
            common[row * common_width..(row + 1) * common_width].borrow_mut();
        local.tidx = F::from_usize(start_tidx + row);
        let value = match source {
            RsSource::Constant(value) => {
                if transcript_values[row] != value {
                    return Err(FiniteWarpV3RsStatementError::Transcript);
                }
                continue;
            }
            RsSource::Target { .. } => claim.target,
            RsSource::Scale { term, .. } => mapped.terms[term].scale,
            RsSource::Point {
                term, coordinate, ..
            } => mapped.terms[term].folded_row_eq_point[coordinate],
        };
        local
            .value
            .copy_from_slice(value.as_basis_coefficients_slice());
        if transcript_values[row] != local.value[source.limb()] {
            return Err(FiniteWarpV3RsStatementError::Transcript);
        }
    }
    Ok(FiniteWarpV3RsStatementTraceData {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        end_tidx,
    })
}

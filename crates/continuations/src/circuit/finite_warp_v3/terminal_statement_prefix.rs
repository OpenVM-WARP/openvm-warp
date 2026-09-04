//! Exact native finite-terminal statement and constrained-prefix transcript.
//!
//! This is the typed bridge between the final authenticated VACC checkpoint
//! and the complete nonlinear verifier PESAT.  It replays both native
//! statement observations before emitting the only admissible start cursor
//! for `FixedMultiAirCompleteTerminalCircuit`.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_recursion_circuit::{
    bus::{ResumeTranscriptStateBus, ResumeTranscriptStateMessage, TranscriptBus},
    define_typed_permutation_bus,
    native_warp::terminal::{
        FixedMultiAirCompleteBindingBus, FixedMultiAirCompleteBindingMessage,
        FixedMultiAirCompleteInstanceValueBus, FixedMultiAirCompleteInstanceValueMessage,
        FixedMultiAirCompleteStatementStartCursorBus,
        FixedMultiAirCompleteStatementStartCursorMessage,
    },
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    native_warp::{
        FINITE_COMPLETE_TERMINAL_PACKAGE_VERSION, FINITE_COMPLETE_TERMINAL_STATEMENT_TAG,
    },
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField32},
    transcript::TranscriptLog,
    warp_pesat::AccumulatorInstance,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, D_EF, EF, F};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::{
    FiniteWarpV3FinalVaccCheckpointBus, FiniteWarpV3FinalVaccCheckpointMessage,
    FiniteWarpV3TerminalAccumulatorLinkBus, FiniteWarpV3TerminalAccumulatorLinkMessage,
    FiniteWarpV3TerminalDecideBinding, FiniteWarpV3TerminalTranscriptCheckpoint,
    FiniteWarpV3TerminalTranscriptSeamCols, FiniteWarpV3TwoCosetTerminalProfile,
};

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct FiniteWarpV3TerminalStatementStartMessage<T> {
    pub tidx: T,
    pub root: [T; DIGEST_SIZE],
    pub accumulator_digest: [T; DIGEST_SIZE],
}

define_typed_permutation_bus!(
    FiniteWarpV3TerminalStatementStartBus,
    FiniteWarpV3TerminalStatementStartMessage
);

/// Complete seam variant. It is the sole consumer of the final VACC
/// checkpoint and additionally publishes the exact statement start cursor.
pub struct FiniteWarpV3CompleteTerminalTranscriptSeamAir {
    pub final_vacc_checkpoint_bus: FiniteWarpV3FinalVaccCheckpointBus,
    pub terminal_resume_bus: ResumeTranscriptStateBus,
    pub terminal_accumulator_link_bus: FiniteWarpV3TerminalAccumulatorLinkBus,
    pub statement_start_bus: FiniteWarpV3TerminalStatementStartBus,
    pub binding: FiniteWarpV3TerminalDecideBinding,
    pub relation_digest: Digest,
}

impl BaseAirWithPublicValues<F> for FiniteWarpV3CompleteTerminalTranscriptSeamAir {}
impl PartitionedBaseAir<F> for FiniteWarpV3CompleteTerminalTranscriptSeamAir {}
impl BaseAir<F> for FiniteWarpV3CompleteTerminalTranscriptSeamAir {
    fn width(&self) -> usize {
        FiniteWarpV3TerminalTranscriptSeamCols::<F>::width()
    }
}

impl<AB> Air<AB> for FiniteWarpV3CompleteTerminalTranscriptSeamAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("complete terminal seam row");
        let local: &FiniteWarpV3TerminalTranscriptSeamCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_transition().assert_zero(local.active);
        self.final_vacc_checkpoint_bus.lookup_key(
            builder,
            FiniteWarpV3FinalVaccCheckpointMessage {
                protocol_digest: self.binding.protocol_digest.map(Into::into),
                relation_digest: self.relation_digest.map(Into::into),
                warp_index_digest: self.binding.warp_index_digest.map(Into::into),
                setup_digest: self.binding.vacc_setup_digest.map(Into::into),
                schedule_digest: self.binding.schedule_digest.map(Into::into),
                call_count: AB::Expr::from_usize(self.binding.warp_call_count),
                end_tidx: local.tidx.into(),
                end_sample_count: local.sample_count.into(),
                end_state: local.state.map(Into::into),
                output_accumulator_root: local.output_accumulator_root.map(Into::into),
                output_accumulator_digest: local.output_accumulator_digest.map(Into::into),
            },
            local.active,
        );
        self.terminal_resume_bus.send(
            builder,
            AB::Expr::ZERO,
            ResumeTranscriptStateMessage {
                tidx: local.tidx.into(),
                state: local.state.map(Into::into),
            },
            local.active,
        );
        self.terminal_accumulator_link_bus.add_key_with_lookups(
            builder,
            FiniteWarpV3TerminalAccumulatorLinkMessage {
                root: local.output_accumulator_root.map(Into::into),
                digest: local.output_accumulator_digest.map(Into::into),
            },
            local.active,
        );
        self.statement_start_bus.send(
            builder,
            FiniteWarpV3TerminalStatementStartMessage {
                tidx: local.tidx.into(),
                root: local.output_accumulator_root.map(Into::into),
                accumulator_digest: local.output_accumulator_digest.map(Into::into),
            },
            local.active,
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PrefixSource {
    Constant(F),
    Root {
        coordinate: usize,
    },
    Instance {
        section: usize,
        coordinate: usize,
        limb: usize,
    },
}

#[derive(Clone, Debug)]
pub struct FiniteWarpV3TerminalStatementPrefixProfile {
    relation_digest: Digest,
    alpha_len: usize,
    beta_len: usize,
    schedule: Arc<[PrefixSource]>,
}

impl FiniteWarpV3TerminalStatementPrefixProfile {
    pub fn new(
        two_coset: &FiniteWarpV3TwoCosetTerminalProfile,
        relation_digest: Digest,
        terminal_index_digest: Digest,
        beta_len: usize,
    ) -> Result<Self, FiniteWarpV3TerminalStatementPrefixError> {
        let alpha_len = two_coset.alpha_len();
        if relation_digest == [F::ZERO; DIGEST_SIZE]
            || terminal_index_digest == [F::ZERO; DIGEST_SIZE]
            || alpha_len == 0
            || beta_len == 0
        {
            return Err(FiniteWarpV3TerminalStatementPrefixError::Profile);
        }
        let mut schedule = Vec::new();
        push_u64(&mut schedule, FINITE_COMPLETE_TERMINAL_STATEMENT_TAG);
        push_u64(
            &mut schedule,
            u64::from(FINITE_COMPLETE_TERMINAL_PACKAGE_VERSION),
        );
        schedule.extend(relation_digest.into_iter().map(PrefixSource::Constant));
        schedule.extend(
            terminal_index_digest
                .into_iter()
                .map(PrefixSource::Constant),
        );

        let descriptor = two_coset.descriptor_fiat_shamir_values([F::ZERO; DIGEST_SIZE]);
        schedule.push(PrefixSource::Constant(descriptor[0]));
        schedule.extend((0..DIGEST_SIZE).map(|coordinate| PrefixSource::Root { coordinate }));
        schedule.extend(
            descriptor[1 + DIGEST_SIZE..]
                .iter()
                .copied()
                .map(PrefixSource::Constant),
        );
        schedule.extend((0..DIGEST_SIZE).map(|coordinate| PrefixSource::Root { coordinate }));
        push_u64(&mut schedule, alpha_len as u64);
        for coordinate in 0..alpha_len {
            push_instance(&mut schedule, 0, coordinate);
        }
        push_instance(&mut schedule, 1, 0);
        push_u64(&mut schedule, beta_len as u64);
        for coordinate in 0..beta_len {
            push_instance(&mut schedule, 2, coordinate);
        }
        push_instance(&mut schedule, 3, 0);

        // `descriptor.observe_algebraic`: every semantic base-field value is
        // embedded into EF4 before reaching the physical transcript.
        let algebraic = two_coset.descriptor_algebraic_values([F::ZERO; DIGEST_SIZE]);
        push_ext_constant(&mut schedule, algebraic[0]);
        for coordinate in 0..DIGEST_SIZE {
            schedule.push(PrefixSource::Root { coordinate });
            schedule.extend((1..D_EF).map(|_| PrefixSource::Constant(F::ZERO)));
        }
        for value in &algebraic[1 + DIGEST_SIZE..] {
            push_ext_constant(&mut schedule, *value);
        }
        for coordinate in 0..alpha_len {
            push_instance(&mut schedule, 0, coordinate);
        }
        push_instance(&mut schedule, 1, 0);
        for coordinate in 0..beta_len {
            push_instance(&mut schedule, 2, coordinate);
        }
        push_instance(&mut schedule, 3, 0);
        if schedule.is_empty() || schedule.len() > u32::MAX as usize {
            return Err(FiniteWarpV3TerminalStatementPrefixError::Profile);
        }
        Ok(Self {
            relation_digest,
            alpha_len,
            beta_len,
            schedule: schedule.into(),
        })
    }

    #[must_use]
    pub fn observation_len(&self) -> usize {
        self.schedule.len()
    }
}

fn push_u64(schedule: &mut Vec<PrefixSource>, value: u64) {
    for shift in [0, 16, 32, 48] {
        schedule.push(PrefixSource::Constant(F::from_u32(
            ((value >> shift) & 0xffff) as u32,
        )));
    }
}

fn push_ext_constant(schedule: &mut Vec<PrefixSource>, value: EF) {
    schedule.extend(
        value
            .as_basis_coefficients_slice()
            .iter()
            .copied()
            .map(PrefixSource::Constant),
    );
}

fn push_instance(schedule: &mut Vec<PrefixSource>, section: usize, coordinate: usize) {
    schedule.extend((0..D_EF).map(|limb| PrefixSource::Instance {
        section,
        coordinate,
        limb,
    }));
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiniteWarpV3TerminalStatementPrefixError {
    Profile,
    Shape,
    Transcript,
    Height,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FiniteWarpV3TerminalStatementPrefixScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub source_flags: [T; 3],
    pub root_flags: [T; DIGEST_SIZE],
    pub limb_flags: [T; D_EF],
    pub group_start: T,
    pub carries_to_next: T,
    pub section: T,
    pub coordinate: T,
    pub limb: T,
    pub instance_lookup: T,
    pub constant: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FiniteWarpV3TerminalStatementPrefixCols<T> {
    pub tidx: T,
    pub root: [T; DIGEST_SIZE],
    pub accumulator_digest: [T; DIGEST_SIZE],
    pub instance_value: [T; D_EF],
    pub observed: T,
}

pub struct FiniteWarpV3TerminalStatementPrefixAir {
    pub profile: Arc<FiniteWarpV3TerminalStatementPrefixProfile>,
    pub transcript_bus: TranscriptBus,
    pub start_bus: FiniteWarpV3TerminalStatementStartBus,
    pub complete_start_bus: FixedMultiAirCompleteStatementStartCursorBus,
    pub binding_bus: FixedMultiAirCompleteBindingBus,
    pub instance_bus: FixedMultiAirCompleteInstanceValueBus,
}

impl BaseAirWithPublicValues<F> for FiniteWarpV3TerminalStatementPrefixAir {}
impl PartitionedBaseAir<F> for FiniteWarpV3TerminalStatementPrefixAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FiniteWarpV3TerminalStatementPrefixScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FiniteWarpV3TerminalStatementPrefixCols::<F>::width()
    }
}
impl BaseAir<F> for FiniteWarpV3TerminalStatementPrefixAir {
    fn width(&self) -> usize {
        FiniteWarpV3TerminalStatementPrefixScheduleCols::<F>::width()
            + FiniteWarpV3TerminalStatementPrefixCols::<F>::width()
    }
}

impl<AB> Air<AB> for FiniteWarpV3TerminalStatementPrefixAir
where
    AB: openvm_stark_backend::air_builders::PartitionedAirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("terminal statement prefix schedule")
            .to_vec();
        let next_cached = builder.cached_mains()[0]
            .row_slice(1)
            .expect("terminal statement prefix next schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("terminal statement prefix row")
            .to_vec();
        let next_common = builder
            .common_main()
            .row_slice(1)
            .expect("terminal statement prefix next row")
            .to_vec();
        let schedule: &FiniteWarpV3TerminalStatementPrefixScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let next_schedule: &FiniteWarpV3TerminalStatementPrefixScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FiniteWarpV3TerminalStatementPrefixCols<AB::Var> = common.as_slice().borrow();
        let next: &FiniteWarpV3TerminalStatementPrefixCols<AB::Var> =
            next_common.as_slice().borrow();
        for flag in [
            schedule.active,
            schedule.is_first,
            schedule.is_last,
            schedule.instance_lookup,
            schedule.group_start,
            schedule.carries_to_next,
        ]
        .into_iter()
        .chain(schedule.source_flags)
        .chain(schedule.root_flags)
        .chain(schedule.limb_flags)
        {
            builder.assert_bool(flag);
        }
        let source_sum = schedule
            .source_flags
            .into_iter()
            .fold(AB::Expr::ZERO, |sum, flag| sum + flag);
        builder.assert_eq(source_sum, schedule.active);
        builder.assert_eq(
            schedule
                .root_flags
                .into_iter()
                .fold(AB::Expr::ZERO, |sum, flag| sum + flag),
            schedule.source_flags[1],
        );
        builder.assert_eq(
            schedule
                .limb_flags
                .into_iter()
                .fold(AB::Expr::ZERO, |sum, flag| sum + flag),
            schedule.source_flags[2],
        );
        builder.assert_eq(schedule.group_start, schedule.instance_lookup);
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
        for coordinate in 0..DIGEST_SIZE {
            builder
                .when_transition()
                .when(next_schedule.active)
                .assert_eq(next.root[coordinate], local.root[coordinate]);
            builder
                .when_transition()
                .when(next_schedule.active)
                .assert_eq(
                    next.accumulator_digest[coordinate],
                    local.accumulator_digest[coordinate],
                );
        }
        let mut transition = builder.when_transition();
        let mut carry = transition.when(schedule.carries_to_next);
        for limb in 0..D_EF {
            carry.assert_eq(next.instance_value[limb], local.instance_value[limb]);
        }
        let [is_constant, is_root, is_instance] = schedule.source_flags.map(AB::Expr::from);
        let selected_root = (0..DIGEST_SIZE)
            .map(|coordinate| schedule.root_flags[coordinate] * local.root[coordinate])
            .fold(AB::Expr::ZERO, |sum, value| sum + value);
        let selected_instance = (0..D_EF)
            .map(|limb| schedule.limb_flags[limb] * local.instance_value[limb])
            .fold(AB::Expr::ZERO, |sum, value| sum + value);
        builder.when(schedule.active).assert_eq(
            local.observed,
            is_constant * schedule.constant
                + is_root * selected_root
                + is_instance * selected_instance,
        );
        self.transcript_bus.observe(
            builder,
            AB::Expr::ZERO,
            local.tidx,
            local.observed,
            schedule.active,
        );
        self.start_bus.receive(
            builder,
            FiniteWarpV3TerminalStatementStartMessage {
                tidx: local.tidx.into(),
                root: local.root.map(Into::into),
                accumulator_digest: local.accumulator_digest.map(Into::into),
            },
            schedule.is_first,
        );
        self.binding_bus.lookup_key(
            builder,
            FixedMultiAirCompleteBindingMessage {
                relation_digest: self.profile.relation_digest.map(Into::into),
                root: local.root.map(Into::into),
                alpha_len: AB::Expr::from_usize(self.profile.alpha_len),
                beta_len: AB::Expr::from_usize(self.profile.beta_len),
            },
            schedule.is_first,
        );
        self.instance_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInstanceValueMessage {
                section: schedule.section.into(),
                coordinate: schedule.coordinate.into(),
                value: local.instance_value.map(Into::into),
            },
            schedule.instance_lookup,
        );
        self.complete_start_bus.send(
            builder,
            FixedMultiAirCompleteStatementStartCursorMessage {
                start_tidx: AB::Expr::from(local.tidx) + AB::Expr::ONE,
            },
            schedule.is_last,
        );
    }
}

pub struct FiniteWarpV3TerminalStatementPrefixTraceData {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub end_tidx: usize,
}

pub fn generate_finite_warp_v3_terminal_statement_prefix_trace(
    profile: &FiniteWarpV3TerminalStatementPrefixProfile,
    instance: &AccumulatorInstance<EF, Digest>,
    accumulator_digest: Digest,
    transcript: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    start: FiniteWarpV3TerminalTranscriptCheckpoint,
    required_height: Option<usize>,
) -> Result<FiniteWarpV3TerminalStatementPrefixTraceData, FiniteWarpV3TerminalStatementPrefixError>
{
    if instance.alpha.len() != profile.alpha_len || instance.beta.len() != profile.beta_len {
        return Err(FiniteWarpV3TerminalStatementPrefixError::Shape);
    }
    let start_tidx =
        usize::try_from(start.tidx).map_err(|_| FiniteWarpV3TerminalStatementPrefixError::Shape)?;
    let end_tidx = start_tidx
        .checked_add(profile.schedule.len())
        .ok_or(FiniteWarpV3TerminalStatementPrefixError::Shape)?;
    if end_tidx >= F::ORDER_U32 as usize {
        return Err(FiniteWarpV3TerminalStatementPrefixError::Shape);
    }
    let values = transcript
        .values()
        .get(start_tidx..end_tidx)
        .ok_or(FiniteWarpV3TerminalStatementPrefixError::Transcript)?;
    if transcript
        .samples()
        .get(start_tidx..end_tidx)
        .is_none_or(|samples| samples.iter().any(|&sample| sample))
    {
        return Err(FiniteWarpV3TerminalStatementPrefixError::Transcript);
    }
    let valid_rows = profile.schedule.len();
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows || !height.is_power_of_two() {
        return Err(FiniteWarpV3TerminalStatementPrefixError::Height);
    }
    let cached_width = FiniteWarpV3TerminalStatementPrefixScheduleCols::<F>::width();
    let common_width = FiniteWarpV3TerminalStatementPrefixCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    for (row, source) in profile.schedule.iter().copied().enumerate() {
        let cached_row = &mut cached[row * cached_width..(row + 1) * cached_width];
        let schedule: &mut FiniteWarpV3TerminalStatementPrefixScheduleCols<F> =
            cached_row.borrow_mut();
        schedule.active = F::ONE;
        schedule.is_first = F::from_bool(row == 0);
        schedule.is_last = F::from_bool(row + 1 == valid_rows);
        let common_row = &mut common[row * common_width..(row + 1) * common_width];
        let local: &mut FiniteWarpV3TerminalStatementPrefixCols<F> = common_row.borrow_mut();
        local.tidx = F::from_usize(start_tidx + row);
        local.root = instance.rt;
        local.accumulator_digest = accumulator_digest;
        match source {
            PrefixSource::Constant(value) => {
                schedule.source_flags[0] = F::ONE;
                schedule.constant = value;
                local.observed = value;
            }
            PrefixSource::Root { coordinate } => {
                schedule.source_flags[1] = F::ONE;
                schedule.root_flags[coordinate] = F::ONE;
                local.observed = instance.rt[coordinate];
            }
            PrefixSource::Instance {
                section,
                coordinate,
                limb,
            } => {
                let value = instance_value(instance, section, coordinate)?;
                schedule.source_flags[2] = F::ONE;
                schedule.section = F::from_usize(section);
                schedule.coordinate = F::from_usize(coordinate);
                schedule.limb = F::from_usize(limb);
                schedule.limb_flags[limb] = F::ONE;
                schedule.group_start = F::from_bool(limb == 0);
                schedule.carries_to_next = F::from_bool(limb + 1 < D_EF);
                schedule.instance_lookup = F::from_bool(limb == 0);
                local
                    .instance_value
                    .copy_from_slice(value.as_basis_coefficients_slice());
                local.observed = local.instance_value[limb];
            }
        }
        if values[row] != local.observed {
            return Err(FiniteWarpV3TerminalStatementPrefixError::Transcript);
        }
    }
    Ok(FiniteWarpV3TerminalStatementPrefixTraceData {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        end_tidx,
    })
}

fn instance_value(
    instance: &AccumulatorInstance<EF, Digest>,
    section: usize,
    coordinate: usize,
) -> Result<EF, FiniteWarpV3TerminalStatementPrefixError> {
    match section {
        0 => instance.alpha.get(coordinate).copied(),
        1 => (coordinate == 0).then_some(instance.mu),
        2 => instance.beta.get(coordinate).copied(),
        3 => (coordinate == 0).then_some(instance.eta),
        _ => None,
    }
    .ok_or(FiniteWarpV3TerminalStatementPrefixError::Shape)
}

#[cfg(test)]
#[path = "terminal_statement_prefix_tests.rs"]
mod terminal_statement_prefix_tests;

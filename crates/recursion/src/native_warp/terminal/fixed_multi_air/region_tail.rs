use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder,
    interaction::InteractionBuilder,
    native_warp::{DirectAirConstraintSumcheckProof, DirectAirPesatIndex},
    transcript::TranscriptLog,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    native_warp::terminal::fixed_multi_air::{
        FixedMultiAirRegionFinalEvaluationBus, FixedMultiAirRegionFinalEvaluationMessage,
        FixedMultiAirRegionLocalPaddingPointBus, FixedMultiAirRegionLocalPaddingPointMessage,
        FixedMultiAirRegionOpeningBus, FixedMultiAirRegionOpeningMessage,
        FixedMultiAirRegionRhoBus, FixedMultiAirRegionRhoMessage,
        FixedMultiAirRegionSumcheckFinalBus, FixedMultiAirRegionSumcheckFinalMessage,
    },
};

const DIRECT_TERMINAL_OPENINGS_TAG: u64 = 0x4e57_4441_4343_0003;
const DIRECT_TERMINAL_MAPPED_BATCH_TAG: u64 = 0x4e57_4441_5442_0001;
const DIRECT_TERMINAL_PADDING_TAG: u64 = 0x4e57_4441_5450_0001;

const SOURCE_COUNT: usize = 3;
const SOURCE_CONSTANT: usize = 0;
const SOURCE_OPENING: usize = 1;
const SOURCE_SAMPLE: usize = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedMultiAirRegionTailTraceError {
    Shape,
    Transcript,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirRegionTailScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub source_flags: [T; SOURCE_COUNT],
    pub source_index: T,
    pub constant: [T; D_EF],
    pub opening_lookup_count: T,
    pub rho_lookup_count: T,
    pub padding_point_lookup_count: T,
    pub emit_final_evaluation: T,
    pub is_rho: T,
    pub is_padding_point: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirRegionTailCols<T> {
    pub tidx: T,
    pub value: [T; D_EF],
    pub final_claim: [T; D_EF],
}

/// Exact post-sumcheck transcript producer for one fixed AIR region.
///
/// The row schedule is key data.  It has one row per observed opening and one
/// row per sampled local-padding coordinate, so both width and AIR degree are
/// independent of trace height and dynamic-column count.
pub struct FixedMultiAirRegionTailAir {
    pub transcript_bus: TranscriptBus,
    pub sumcheck_final_bus: FixedMultiAirRegionSumcheckFinalBus,
    pub final_evaluation_bus: FixedMultiAirRegionFinalEvaluationBus,
    pub opening_bus: FixedMultiAirRegionOpeningBus,
    pub rho_bus: FixedMultiAirRegionRhoBus,
    pub local_padding_point_bus: FixedMultiAirRegionLocalPaddingPointBus,
    pub region: usize,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirRegionTailAir {}
impl PartitionedBaseAir<F> for FixedMultiAirRegionTailAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirRegionTailScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirRegionTailCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirRegionTailAir {}
impl BaseAir<F> for FixedMultiAirRegionTailAir {
    fn width(&self) -> usize {
        FixedMultiAirRegionTailScheduleCols::<F>::width()
            + FixedMultiAirRegionTailCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirRegionTailAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    AB::Expr: PrimeCharacteristicRing,
{
    fn eval(&self, builder: &mut AB) {
        let cached_row = builder.cached_mains()[0]
            .row_slice(0)
            .expect("fixed region-tail schedule row")
            .to_vec();
        let next_cached_row = builder.cached_mains()[0]
            .row_slice(1)
            .expect("fixed next region-tail schedule row")
            .to_vec();
        let common_row = builder
            .common_main()
            .row_slice(0)
            .expect("fixed region-tail row")
            .to_vec();
        let next_common_row = builder
            .common_main()
            .row_slice(1)
            .expect("fixed next region-tail row")
            .to_vec();
        let schedule: &FixedMultiAirRegionTailScheduleCols<AB::Var> =
            cached_row.as_slice().borrow();
        let next_schedule: &FixedMultiAirRegionTailScheduleCols<AB::Var> =
            next_cached_row.as_slice().borrow();
        let local: &FixedMultiAirRegionTailCols<AB::Var> = common_row.as_slice().borrow();
        let next: &FixedMultiAirRegionTailCols<AB::Var> = next_common_row.as_slice().borrow();

        for flag in [
            schedule.active,
            schedule.is_first,
            schedule.is_last,
            schedule.emit_final_evaluation,
            schedule.is_rho,
            schedule.is_padding_point,
        ]
        .into_iter()
        .chain(schedule.source_flags)
        {
            builder.assert_bool(flag);
        }
        let source_sum = schedule
            .source_flags
            .iter()
            .fold(AB::Expr::ZERO, |sum, &flag| sum + AB::Expr::from(flag));
        builder.assert_eq(source_sum, schedule.active);
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
        same.assert_eq(next.tidx, local.tidx + AB::Expr::from_usize(D_EF));
        assert_array_eq(&mut same, next.final_claim, local.final_claim);

        assert_array_eq(
            &mut builder.when(schedule.source_flags[SOURCE_CONSTANT]),
            local.value,
            schedule.constant.map(Into::into),
        );
        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            local.tidx,
            local.value,
            schedule.source_flags[SOURCE_CONSTANT] + schedule.source_flags[SOURCE_OPENING],
        );
        self.transcript_bus.sample_ext(
            builder,
            AB::Expr::ZERO,
            local.tidx,
            local.value,
            schedule.source_flags[SOURCE_SAMPLE],
        );

        self.sumcheck_final_bus.receive(
            builder,
            FixedMultiAirRegionSumcheckFinalMessage {
                region: AB::Expr::from_usize(self.region),
                tidx: local.tidx.into(),
                claim: local.final_claim.map(Into::into),
            },
            schedule.is_first,
        );
        self.opening_bus.add_key_with_lookups(
            builder,
            FixedMultiAirRegionOpeningMessage {
                region: AB::Expr::from_usize(self.region),
                opening: schedule.source_index.into(),
                value: local.value.map(Into::into),
            },
            schedule.opening_lookup_count,
        );
        self.rho_bus.add_key_with_lookups(
            builder,
            FixedMultiAirRegionRhoMessage {
                region: AB::Expr::from_usize(self.region),
                value: local.value.map(Into::into),
            },
            schedule.rho_lookup_count,
        );
        self.local_padding_point_bus.add_key_with_lookups(
            builder,
            FixedMultiAirRegionLocalPaddingPointMessage {
                region: AB::Expr::from_usize(self.region),
                coordinate: schedule.source_index.into(),
                value: local.value.map(Into::into),
            },
            schedule.padding_point_lookup_count,
        );
        self.final_evaluation_bus.send(
            builder,
            FixedMultiAirRegionFinalEvaluationMessage {
                region: AB::Expr::from_usize(self.region),
                claim: local.final_claim.map(Into::into),
            },
            schedule.emit_final_evaluation,
        );
    }
}

#[derive(Clone, Debug)]
struct TailScheduleEntry {
    source: usize,
    source_index: usize,
    constant: EF,
    opening_lookup_count: usize,
    rho_lookup_count: usize,
    padding_point_lookup_count: usize,
    emit_final_evaluation: bool,
    is_rho: bool,
    is_padding_point: bool,
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirRegionTailTraceOutput {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub rho: EF,
    pub local_padding_point: Vec<EF>,
    pub end_tidx: usize,
}

#[allow(clippy::too_many_arguments)]
pub fn generate_fixed_multi_air_region_tail_traces(
    relation: &DirectAirPesatIndex<F, Digest>,
    final_claim: EF,
    proof: &DirectAirConstraintSumcheckProof<EF>,
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
    opening_lookup_counts: &[usize],
    rho_lookup_count: usize,
    padding_point_lookup_count: usize,
    required_height: Option<usize>,
) -> Result<FixedMultiAirRegionTailTraceOutput, FixedMultiAirRegionTailTraceError> {
    if proof.opened_columns.len() != opening_lookup_counts.len() {
        return Err(FixedMultiAirRegionTailTraceError::Shape);
    }
    let padded_message_len = relation.pesat_shape().witness_len();
    let has_local_padding = relation.raw_witness_len() < padded_message_len;
    let local_padding_len = if has_local_padding {
        relation.pesat_shape().log_witness
    } else {
        0
    };

    let mut schedule = Vec::new();
    push_const(&mut schedule, DIRECT_TERMINAL_OPENINGS_TAG);
    push_const(&mut schedule, proof.opened_columns.len() as u64);
    let final_eval_row = if proof.opened_columns.is_empty() {
        1
    } else {
        1 + proof.opened_columns.len()
    };
    for (opening, &lookup_count) in opening_lookup_counts.iter().enumerate() {
        let mut entry = source(SOURCE_OPENING, opening);
        entry.opening_lookup_count = lookup_count;
        schedule.push(entry);
    }
    schedule[final_eval_row].emit_final_evaluation = true;
    push_const(&mut schedule, DIRECT_TERMINAL_MAPPED_BATCH_TAG);
    let mut rho_entry = source(SOURCE_SAMPLE, 0);
    rho_entry.rho_lookup_count = rho_lookup_count;
    rho_entry.is_rho = true;
    schedule.push(rho_entry);
    push_const(&mut schedule, DIRECT_TERMINAL_PADDING_TAG);
    push_const(&mut schedule, u64::from(has_local_padding));
    for coordinate in 0..local_padding_len {
        let mut entry = source(SOURCE_SAMPLE, coordinate);
        entry.padding_point_lookup_count = padding_point_lookup_count;
        entry.is_padding_point = true;
        schedule.push(entry);
    }

    let height = required_height.unwrap_or_else(|| schedule.len().next_power_of_two());
    if height < schedule.len() {
        return Err(FixedMultiAirRegionTailTraceError::Shape);
    }
    let cached_width = FixedMultiAirRegionTailScheduleCols::<F>::width();
    let common_width = FixedMultiAirRegionTailCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    let mut tidx = start_tidx;
    let mut rho = None;
    let mut local_padding_point = Vec::with_capacity(local_padding_len);
    for (row_index, entry) in schedule.iter().enumerate() {
        let expected = match entry.source {
            SOURCE_CONSTANT => Some(entry.constant),
            SOURCE_OPENING => proof.opened_columns.get(entry.source_index).copied(),
            SOURCE_SAMPLE => None,
            _ => return Err(FixedMultiAirRegionTailTraceError::Shape),
        };
        let value = read_ext(transcript, tidx, entry.source == SOURCE_SAMPLE)?;
        if expected.is_some_and(|expected| expected != value) {
            return Err(FixedMultiAirRegionTailTraceError::Transcript);
        }
        if entry.is_rho {
            rho = Some(value);
        }
        if entry.is_padding_point {
            local_padding_point.push(value);
        }

        let cached_row = &mut cached[row_index * cached_width..(row_index + 1) * cached_width];
        let schedule_cols: &mut FixedMultiAirRegionTailScheduleCols<F> = cached_row.borrow_mut();
        schedule_cols.active = F::ONE;
        schedule_cols.is_first = F::from_bool(row_index == 0);
        schedule_cols.is_last = F::from_bool(row_index + 1 == schedule.len());
        schedule_cols.source_flags[entry.source] = F::ONE;
        schedule_cols.source_index = F::from_usize(entry.source_index);
        copy_ext(&mut schedule_cols.constant, entry.constant);
        schedule_cols.opening_lookup_count = F::from_usize(entry.opening_lookup_count);
        schedule_cols.rho_lookup_count = F::from_usize(entry.rho_lookup_count);
        schedule_cols.padding_point_lookup_count = F::from_usize(entry.padding_point_lookup_count);
        schedule_cols.emit_final_evaluation = F::from_bool(entry.emit_final_evaluation);
        schedule_cols.is_rho = F::from_bool(entry.is_rho);
        schedule_cols.is_padding_point = F::from_bool(entry.is_padding_point);

        let common_row = &mut common[row_index * common_width..(row_index + 1) * common_width];
        let cols: &mut FixedMultiAirRegionTailCols<F> = common_row.borrow_mut();
        cols.tidx = F::from_usize(tidx);
        copy_ext(&mut cols.value, value);
        copy_ext(&mut cols.final_claim, final_claim);
        tidx += D_EF;
    }
    Ok(FixedMultiAirRegionTailTraceOutput {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        rho: rho.ok_or(FixedMultiAirRegionTailTraceError::Transcript)?,
        local_padding_point,
        end_tidx: tidx,
    })
}

fn source(source: usize, source_index: usize) -> TailScheduleEntry {
    TailScheduleEntry {
        source,
        source_index,
        constant: EF::ZERO,
        opening_lookup_count: 0,
        rho_lookup_count: 0,
        padding_point_lookup_count: 0,
        emit_final_evaluation: false,
        is_rho: false,
        is_padding_point: false,
    }
}

fn push_const(schedule: &mut Vec<TailScheduleEntry>, value: u64) {
    let mut entry = source(SOURCE_CONSTANT, 0);
    entry.constant = EF::from_u64(value);
    schedule.push(entry);
}

fn read_ext(
    transcript: &TranscriptLog<F, [F; 16]>,
    tidx: usize,
    is_sample: bool,
) -> Result<EF, FixedMultiAirRegionTailTraceError> {
    let end = tidx
        .checked_add(D_EF)
        .ok_or(FixedMultiAirRegionTailTraceError::Transcript)?;
    let values = transcript
        .values()
        .get(tidx..end)
        .ok_or(FixedMultiAirRegionTailTraceError::Transcript)?;
    let samples = transcript
        .samples()
        .get(tidx..end)
        .ok_or(FixedMultiAirRegionTailTraceError::Transcript)?;
    if samples.iter().any(|&sample| sample != is_sample) {
        return Err(FixedMultiAirRegionTailTraceError::Transcript);
    }
    EF::from_basis_coefficients_slice(values).ok_or(FixedMultiAirRegionTailTraceError::Transcript)
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder, interaction::InteractionBuilder,
    transcript::TranscriptLog, warp_accum::TerminalDescriptor, warp_pesat::AccumulatorInstance,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    native_warp::terminal::{
        fixed_multi_air::{
            FixedMultiAirBatchedClaimBus, FixedMultiAirBatchedClaimMessage,
            FixedMultiAirStructuredClaimHeaderBus, FixedMultiAirStructuredClaimHeaderMessage,
            FixedMultiAirTerminalBindingBus, FixedMultiAirTerminalBindingMessage,
            FixedMultiAirTerminalInstanceValueBus, FixedMultiAirTerminalInstanceValueMessage,
            FixedMultiAirWhirStartBus, FixedMultiAirWhirStartMessage,
        },
        NativeTerminalAccumulatorRootBus, NativeTerminalAccumulatorRootMessage,
        NativeTerminalAccumulatorValueBus, NativeTerminalAccumulatorValueMessage,
        NativeTerminalWhirStatementBus, NativeTerminalWhirStatementMessage,
        NATIVE_TERMINAL_DESCRIPTOR_DOMAIN_TAG,
    },
    utils::{ext_field_add, ext_field_multiply},
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedMultiAirStatementTraceError {
    Shape,
    Transcript,
    Claim,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirWhirPrefixCols<T> {
    pub active: T,
    pub tidx: T,
    pub root: [T; DIGEST_SIZE],
    pub mu: [T; D_EF],
    pub batching_challenge: [T; D_EF],
}

/// Starts the backend terminal-WHIR transcript from the already authenticated
/// terminal descriptor/root and samples the exact backend batching challenge.
#[derive(ColumnsAir)]
#[columns_via(FixedMultiAirWhirPrefixCols<u8>)]
pub struct FixedMultiAirWhirPrefixAir {
    pub transcript_bus: TranscriptBus,
    pub binding_bus: FixedMultiAirTerminalBindingBus,
    pub instance_bus: FixedMultiAirTerminalInstanceValueBus,
    pub start_bus: FixedMultiAirWhirStartBus,
    /// Same-root output consumed by the outer statement AIR. This is emitted
    /// from the value already authenticated by `binding_bus`, not from host
    /// metadata.
    pub root_bus: NativeTerminalAccumulatorRootBus,
    pub relation_digest: Digest,
    pub metadata_words: Vec<u64>,
    pub alpha_len: usize,
    pub beta_len: usize,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirWhirPrefixAir {}
impl PartitionedBaseAir<F> for FixedMultiAirWhirPrefixAir {}
impl BaseAir<F> for FixedMultiAirWhirPrefixAir {
    fn width(&self) -> usize {
        FixedMultiAirWhirPrefixCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for FixedMultiAirWhirPrefixAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("fixed multi-AIR WHIR prefix row");
        let local: &FixedMultiAirWhirPrefixCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_transition().assert_zero(local.active);

        self.binding_bus.receive(
            builder,
            FixedMultiAirTerminalBindingMessage {
                relation_digest: self.relation_digest.map(Into::into),
                root: local.root.map(Into::into),
                alpha_len: AB::Expr::from_usize(self.alpha_len),
                beta_len: AB::Expr::from_usize(self.beta_len),
            },
            local.active,
        );
        self.instance_bus.lookup_key(
            builder,
            FixedMultiAirTerminalInstanceValueMessage {
                section: AB::Expr::ONE,
                coordinate: AB::Expr::ZERO,
                value: local.mu.map(Into::into),
            },
            local.active,
        );
        self.transcript_bus.observe(
            builder,
            AB::Expr::ZERO,
            local.tidx.into(),
            AB::Expr::from_u64(NATIVE_TERMINAL_DESCRIPTOR_DOMAIN_TAG),
            local.active,
        );
        self.transcript_bus.observe_commit(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.tidx) + AB::Expr::ONE,
            local.root,
            local.active,
        );
        let metadata_tidx = AB::Expr::from(local.tidx) + AB::Expr::from_usize(1 + DIGEST_SIZE);
        for (index, &word) in self.metadata_words.iter().enumerate() {
            self.transcript_bus.observe(
                builder,
                AB::Expr::ZERO,
                metadata_tidx.clone() + AB::Expr::from_usize(index),
                AB::Expr::from_u64(word),
                local.active,
            );
        }
        let batching_tidx = metadata_tidx + AB::Expr::from_usize(self.metadata_words.len());
        self.transcript_bus.sample_ext(
            builder,
            AB::Expr::ZERO,
            batching_tidx,
            local.batching_challenge,
            local.active,
        );
        self.start_bus.send(
            builder,
            FixedMultiAirWhirStartMessage {
                tidx: AB::Expr::from(local.tidx)
                    + AB::Expr::from_usize(1 + DIGEST_SIZE + self.metadata_words.len() + D_EF),
                root: local.root.map(Into::into),
                batching_challenge: local.batching_challenge.map(Into::into),
                mu: local.mu.map(Into::into),
            },
            local.active,
        );
        self.root_bus.send(
            builder,
            NativeTerminalAccumulatorRootMessage {
                root: local.root.map(Into::into),
            },
            local.active,
        );
    }
}

pub fn generate_fixed_multi_air_whir_prefix_trace(
    descriptor: &TerminalDescriptor<Digest>,
    instance: &AccumulatorInstance<EF, Digest>,
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
) -> Result<RowMajorMatrix<F>, FixedMultiAirStatementTraceError> {
    if descriptor.root != instance.rt {
        return Err(FixedMultiAirStatementTraceError::Shape);
    }
    expect_base(
        transcript,
        start_tidx,
        F::from_u64(NATIVE_TERMINAL_DESCRIPTOR_DOMAIN_TAG),
        false,
    )?;
    for (limb, &root) in descriptor.root.iter().enumerate() {
        expect_base(transcript, start_tidx + 1 + limb, root, false)?;
    }
    let metadata_start = start_tidx + 1 + DIGEST_SIZE;
    for (index, word) in descriptor.metadata_words().enumerate() {
        expect_base(transcript, metadata_start + index, F::from_u64(word), false)?;
    }
    let batching_tidx = metadata_start + descriptor.metadata_words().count();
    let batching_challenge = read_ext(transcript, batching_tidx, true)?;
    let width = FixedMultiAirWhirPrefixCols::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut FixedMultiAirWhirPrefixCols<F> = values.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.tidx = F::from_usize(start_tidx);
    cols.root = descriptor.root;
    copy_ext(&mut cols.mu, instance.mu);
    copy_ext(&mut cols.batching_challenge, batching_challenge);
    Ok(RowMajorMatrix::new(values, width))
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirTargetScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub claim: T,
    pub endpoint_lookup_count: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirTargetCols<T> {
    pub statement_tidx: T,
    pub root: [T; DIGEST_SIZE],
    pub kind: T,
    pub log_message_len: T,
    pub term_count: T,
    pub point_len: T,
    pub target: [T; D_EF],
    pub batching_challenge: [T; D_EF],
    pub batching_scale: [T; D_EF],
    pub sum_before: [T; D_EF],
    pub sum_after: [T; D_EF],
    pub mu: [T; D_EF],
}

/// Computes exactly `mu + sum_i xi^(i+1) target_i` in canonical claim order,
/// and forwards each complete structured descriptor with the same xi power to
/// the RS-dual verifier.
pub struct FixedMultiAirStructuredTargetAir {
    pub start_bus: FixedMultiAirWhirStartBus,
    pub header_bus: FixedMultiAirStructuredClaimHeaderBus,
    pub batched_claim_bus: FixedMultiAirBatchedClaimBus,
    pub statement_bus: NativeTerminalWhirStatementBus,
    pub claim_count: usize,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirStructuredTargetAir {}
impl PartitionedBaseAir<F> for FixedMultiAirStructuredTargetAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirTargetScheduleCols::<F>::width()]
    }
    fn common_main_width(&self) -> usize {
        FixedMultiAirTargetCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirStructuredTargetAir {}
impl BaseAir<F> for FixedMultiAirStructuredTargetAir {
    fn width(&self) -> usize {
        FixedMultiAirTargetScheduleCols::<F>::width() + FixedMultiAirTargetCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirStructuredTargetAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("structured target schedule")
            .to_vec();
        let next_cached = builder.cached_mains()[0]
            .row_slice(1)
            .expect("next structured target schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("structured target row")
            .to_vec();
        let next_common = builder
            .common_main()
            .row_slice(1)
            .expect("next structured target row")
            .to_vec();
        let schedule: &FixedMultiAirTargetScheduleCols<AB::Var> = cached.as_slice().borrow();
        let next_schedule: &FixedMultiAirTargetScheduleCols<AB::Var> =
            next_cached.as_slice().borrow();
        let local: &FixedMultiAirTargetCols<AB::Var> = common.as_slice().borrow();
        let next: &FixedMultiAirTargetCols<AB::Var> = next_common.as_slice().borrow();
        for flag in [schedule.active, schedule.is_first, schedule.is_last] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(schedule.active);
        builder.when_first_row().assert_one(schedule.is_first);
        builder.when_first_row().assert_zero(schedule.claim);
        builder
            .when(schedule.active * schedule.is_last)
            .assert_eq(schedule.claim, AB::Expr::from_usize(self.claim_count - 1));
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
        same.assert_eq(next_schedule.claim, schedule.claim + AB::Expr::ONE);
        same.assert_zero(next_schedule.is_first);
        same.assert_eq(next.statement_tidx, local.statement_tidx);
        assert_array_eq(&mut same, next.root, local.root);
        assert_array_eq(&mut same, next.batching_challenge, local.batching_challenge);
        assert_array_eq(
            &mut same,
            next.batching_scale,
            ext_field_multiply::<AB::Expr>(local.batching_scale, local.batching_challenge),
        );
        assert_array_eq(&mut same, next.sum_before, local.sum_after);
        assert_array_eq(&mut same, next.mu, local.mu);
        assert_array_eq(
            &mut builder.when(schedule.is_first),
            local.batching_scale,
            local.batching_challenge.map(Into::into),
        );
        assert_array_eq(
            &mut builder.when(schedule.is_first),
            local.sum_before,
            local.mu.map(Into::into),
        );
        let contribution = ext_field_multiply::<AB::Expr>(local.batching_scale, local.target);
        assert_array_eq(
            &mut builder.when(schedule.active),
            local.sum_after,
            ext_field_add::<AB::Expr>(local.sum_before, contribution),
        );
        self.start_bus.receive(
            builder,
            FixedMultiAirWhirStartMessage {
                tidx: local.statement_tidx.into(),
                root: local.root.map(Into::into),
                batching_challenge: local.batching_challenge.map(Into::into),
                mu: local.mu.map(Into::into),
            },
            schedule.is_first,
        );
        self.header_bus.receive(
            builder,
            FixedMultiAirStructuredClaimHeaderMessage {
                claim: schedule.claim.into(),
                kind: local.kind.into(),
                log_message_len: local.log_message_len.into(),
                term_count: local.term_count.into(),
                point_len: local.point_len.into(),
                target: local.target.map(Into::into),
            },
            schedule.active,
        );
        self.batched_claim_bus.add_key_with_lookups(
            builder,
            FixedMultiAirBatchedClaimMessage {
                claim: schedule.claim.into(),
                kind: local.kind.into(),
                log_message_len: local.log_message_len.into(),
                term_count: local.term_count.into(),
                point_len: local.point_len.into(),
                target: local.target.map(Into::into),
                batching_scale: local.batching_scale.map(Into::into),
            },
            schedule.endpoint_lookup_count,
        );
        self.statement_bus.add_key_with_lookups(
            builder,
            NativeTerminalWhirStatementMessage {
                tidx: local.statement_tidx.into(),
                root: local.root.map(Into::into),
                batching_challenge: local.batching_challenge.map(Into::into),
                initial_claim: local.sum_after.map(Into::into),
            },
            schedule.is_last * AB::Expr::TWO,
        );
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirStructuredClaimRecord {
    pub kind: usize,
    pub log_message_len: usize,
    pub term_count: usize,
    pub point_len: usize,
    pub target: EF,
    pub endpoint_lookup_count: usize,
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirStructuredTargetTraceOutput {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub initial_claim: EF,
}

#[allow(clippy::too_many_arguments)]
pub fn generate_fixed_multi_air_structured_target_traces(
    descriptor: &TerminalDescriptor<Digest>,
    statement_tidx: usize,
    mu: EF,
    batching_challenge: EF,
    claims: &[FixedMultiAirStructuredClaimRecord],
    required_height: Option<usize>,
) -> Result<FixedMultiAirStructuredTargetTraceOutput, FixedMultiAirStatementTraceError> {
    if claims.is_empty() {
        return Err(FixedMultiAirStatementTraceError::Shape);
    }
    let height = required_height.unwrap_or_else(|| claims.len().next_power_of_two());
    if height < claims.len() {
        return Err(FixedMultiAirStatementTraceError::Shape);
    }
    let cached_width = FixedMultiAirTargetScheduleCols::<F>::width();
    let common_width = FixedMultiAirTargetCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    let mut scale = batching_challenge;
    let mut sum = mu;
    for (claim, record) in claims.iter().enumerate() {
        let before = sum;
        sum += scale * record.target;
        let cached_row = &mut cached[claim * cached_width..(claim + 1) * cached_width];
        let schedule: &mut FixedMultiAirTargetScheduleCols<F> = cached_row.borrow_mut();
        schedule.active = F::ONE;
        schedule.is_first = F::from_bool(claim == 0);
        schedule.is_last = F::from_bool(claim + 1 == claims.len());
        schedule.claim = F::from_usize(claim);
        schedule.endpoint_lookup_count = F::from_usize(record.endpoint_lookup_count);
        let common_row = &mut common[claim * common_width..(claim + 1) * common_width];
        let cols: &mut FixedMultiAirTargetCols<F> = common_row.borrow_mut();
        cols.statement_tidx = F::from_usize(statement_tidx);
        cols.root = descriptor.root;
        cols.kind = F::from_usize(record.kind);
        cols.log_message_len = F::from_usize(record.log_message_len);
        cols.term_count = F::from_usize(record.term_count);
        cols.point_len = F::from_usize(record.point_len);
        copy_ext(&mut cols.target, record.target);
        copy_ext(&mut cols.batching_challenge, batching_challenge);
        copy_ext(&mut cols.batching_scale, scale);
        copy_ext(&mut cols.sum_before, before);
        copy_ext(&mut cols.sum_after, sum);
        copy_ext(&mut cols.mu, mu);
        scale *= batching_challenge;
    }
    Ok(FixedMultiAirStructuredTargetTraceOutput {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        initial_claim: sum,
    })
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirAccumulatorBridgeScheduleCols<T> {
    pub active: T,
    pub section: T,
    pub coordinate: T,
    pub native_lookup_count: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirAccumulatorBridgeCols<T> {
    pub value: [T; D_EF],
}

/// Typed bridge from the complete fixed-multi-AIR instance catalog to the
/// existing exact RS-adjoint producers.  No value is asserted by the host.
pub struct FixedMultiAirAccumulatorBridgeAir {
    pub fixed_bus: FixedMultiAirTerminalInstanceValueBus,
    pub native_bus: NativeTerminalAccumulatorValueBus,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirAccumulatorBridgeAir {}
impl PartitionedBaseAir<F> for FixedMultiAirAccumulatorBridgeAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirAccumulatorBridgeScheduleCols::<F>::width()]
    }
    fn common_main_width(&self) -> usize {
        FixedMultiAirAccumulatorBridgeCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirAccumulatorBridgeAir {}
impl BaseAir<F> for FixedMultiAirAccumulatorBridgeAir {
    fn width(&self) -> usize {
        FixedMultiAirAccumulatorBridgeScheduleCols::<F>::width()
            + FixedMultiAirAccumulatorBridgeCols::<F>::width()
    }
}

impl<AB: PartitionedAirBuilder<F = F> + InteractionBuilder> Air<AB>
    for FixedMultiAirAccumulatorBridgeAir
{
    fn eval(&self, builder: &mut AB) {
        let cached = builder.cached_mains()[0]
            .row_slice(0)
            .expect("accumulator bridge schedule")
            .to_vec();
        let common = builder
            .common_main()
            .row_slice(0)
            .expect("accumulator bridge row")
            .to_vec();
        let schedule: &FixedMultiAirAccumulatorBridgeScheduleCols<AB::Var> =
            cached.as_slice().borrow();
        let local: &FixedMultiAirAccumulatorBridgeCols<AB::Var> = common.as_slice().borrow();
        builder.assert_bool(schedule.active);
        self.fixed_bus.lookup_key(
            builder,
            FixedMultiAirTerminalInstanceValueMessage {
                section: schedule.section.into(),
                coordinate: schedule.coordinate.into(),
                value: local.value.map(Into::into),
            },
            schedule.active,
        );
        self.native_bus.add_key_with_lookups(
            builder,
            NativeTerminalAccumulatorValueMessage {
                section: schedule.section.into(),
                coordinate: schedule.coordinate.into(),
                value: local.value.map(Into::into),
            },
            schedule.native_lookup_count,
        );
    }
}

pub fn generate_fixed_multi_air_accumulator_bridge_traces(
    instance: &AccumulatorInstance<EF, Digest>,
    native_lookup_counts: &[Vec<usize>; 4],
    required_height: Option<usize>,
) -> Result<(RowMajorMatrix<F>, RowMajorMatrix<F>), FixedMultiAirStatementTraceError> {
    let sections = [
        instance.alpha.as_slice(),
        core::slice::from_ref(&instance.mu),
        instance.beta.as_slice(),
        core::slice::from_ref(&instance.eta),
    ];
    if sections
        .iter()
        .zip(native_lookup_counts)
        .any(|(values, counts)| values.len() != counts.len())
    {
        return Err(FixedMultiAirStatementTraceError::Shape);
    }
    let valid_rows = sections.iter().map(|values| values.len()).sum::<usize>();
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return Err(FixedMultiAirStatementTraceError::Shape);
    }
    let cached_width = FixedMultiAirAccumulatorBridgeScheduleCols::<F>::width();
    let common_width = FixedMultiAirAccumulatorBridgeCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    let mut row = 0usize;
    for (section, (values, counts)) in sections.iter().zip(native_lookup_counts).enumerate() {
        for (coordinate, (&value, &count)) in values.iter().zip(counts).enumerate() {
            let cached_row = &mut cached[row * cached_width..(row + 1) * cached_width];
            let schedule: &mut FixedMultiAirAccumulatorBridgeScheduleCols<F> =
                cached_row.borrow_mut();
            schedule.active = F::ONE;
            schedule.section = F::from_usize(section);
            schedule.coordinate = F::from_usize(coordinate);
            schedule.native_lookup_count = F::from_usize(count);
            let common_row = &mut common[row * common_width..(row + 1) * common_width];
            let cols: &mut FixedMultiAirAccumulatorBridgeCols<F> = common_row.borrow_mut();
            copy_ext(&mut cols.value, value);
            row += 1;
        }
    }
    Ok((
        RowMajorMatrix::new(cached, cached_width),
        RowMajorMatrix::new(common, common_width),
    ))
}

fn read_ext(
    transcript: &TranscriptLog<F, [F; 16]>,
    tidx: usize,
    is_sample: bool,
) -> Result<EF, FixedMultiAirStatementTraceError> {
    let end = tidx
        .checked_add(D_EF)
        .ok_or(FixedMultiAirStatementTraceError::Transcript)?;
    let values = transcript
        .values()
        .get(tidx..end)
        .ok_or(FixedMultiAirStatementTraceError::Transcript)?;
    let samples = transcript
        .samples()
        .get(tidx..end)
        .ok_or(FixedMultiAirStatementTraceError::Transcript)?;
    if samples.iter().any(|&sample| sample != is_sample) {
        return Err(FixedMultiAirStatementTraceError::Transcript);
    }
    EF::from_basis_coefficients_slice(values).ok_or(FixedMultiAirStatementTraceError::Transcript)
}

fn expect_base(
    transcript: &TranscriptLog<F, [F; 16]>,
    tidx: usize,
    expected: F,
    is_sample: bool,
) -> Result<(), FixedMultiAirStatementTraceError> {
    if transcript.values().get(tidx) != Some(&expected)
        || transcript.samples().get(tidx) != Some(&is_sample)
    {
        return Err(FixedMultiAirStatementTraceError::Transcript);
    }
    Ok(())
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

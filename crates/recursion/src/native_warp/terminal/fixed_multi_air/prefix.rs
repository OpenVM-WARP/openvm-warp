use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::PartitionedAirBuilder,
    interaction::InteractionBuilder,
    native_warp::{
        FixedMultiAirPesatIndex, FixedMultiAirTerminalProof, FIXED_MULTI_AIR_TERMINAL_VERSION,
    },
    transcript::TranscriptLog,
    warp_accum::TerminalDescriptor,
    warp_pesat::AccumulatorInstance,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    native_warp::terminal::fixed_multi_air::{
        FixedMultiAirBetaCoordinateBus, FixedMultiAirBetaCoordinateMessage,
        FixedMultiAirGlobalClaimBus, FixedMultiAirGlobalClaimMessage, FixedMultiAirPaddingClaimBus,
        FixedMultiAirPaddingClaimMessage, FixedMultiAirRegionClaimBus,
        FixedMultiAirRegionClaimMessage, FixedMultiAirRegionStartBus,
        FixedMultiAirRegionStartMessage, FixedMultiAirTerminalBindingBus,
        FixedMultiAirTerminalBindingMessage, FixedMultiAirTerminalInstanceValueBus,
        FixedMultiAirTerminalInstanceValueMessage,
    },
};

const TERMINAL_DESCRIPTOR_DOMAIN_TAG: u64 = 917_001;
const FIXED_TERMINAL_STATEMENT_TAG: u64 = 0x4e57_4d41_544d_0002;
const FIXED_TERMINAL_REGION_CLAIMS_TAG: u64 = 0x4e57_4d41_5443_0002;
const FIXED_TERMINAL_REGION_TAG: u64 = 0x4e57_4d41_5452_0002;
const COMPACT_COLUMNS_TRANSCRIPT_TAG: u64 = 0x4e57_4441_4343_0001;
const COMPACT_COLUMNS_TRANSCRIPT_VERSION: u32 = 1;

const SOURCE_COUNT: usize = 8;
const SOURCE_CONSTANT: usize = 0;
const SOURCE_ROOT: usize = 1;
const SOURCE_ALPHA: usize = 2;
const SOURCE_MU: usize = 3;
const SOURCE_BETA: usize = 4;
const SOURCE_ETA: usize = 5;
const SOURCE_REGION_CLAIM: usize = 6;
const SOURCE_PADDING_CLAIM: usize = 7;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedMultiAirTerminalPrefixTraceError {
    Shape,
    Transcript,
    Root,
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirTerminalPrefixLookupCounts {
    pub global_claim: usize,
    pub alpha: Vec<usize>,
    pub mu: usize,
    pub beta: Vec<usize>,
    pub eta: usize,
    pub region_claims: Vec<usize>,
    pub padding_claim: usize,
}

impl FixedMultiAirTerminalPrefixLookupCounts {
    #[must_use]
    pub fn semantic_minimum(alpha_len: usize, beta_len: usize, region_count: usize) -> Self {
        Self {
            global_claim: region_count + 1,
            alpha: vec![1; alpha_len],
            mu: 1,
            // These are fanouts on the typed beta-coordinate bus only. The
            // accumulator bridge consumes the separate instance-value bus.
            beta: vec![0; beta_len],
            eta: 1,
            region_claims: vec![2; region_count],
            padding_claim: 2,
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirPrefixScheduleCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub source_flags: [T; SOURCE_COUNT],
    pub source_index: T,
    /// Fixed one-hot selection of a descriptor-root limb.  Keeping this in
    /// cached rows avoids an index interpolation whose degree grew with the
    /// digest width.
    pub root_flags: [T; DIGEST_SIZE],
    /// Fixed lookup multiplicities.  These are key data, not proof witness
    /// selectors, and avoid variable-degree interpolation over coordinates.
    pub global_lookup_count: T,
    pub instance_lookup_count: T,
    pub beta_lookup_count: T,
    pub region_lookup_count: T,
    pub padding_lookup_count: T,
    pub constant: [T; D_EF],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirPrefixCols<T> {
    pub tidx: T,
    pub value: [T; D_EF],
    pub descriptor_root: [T; DIGEST_SIZE],
    pub instance_root: [T; DIGEST_SIZE],
    pub eta: [T; D_EF],
    pub one: [T; D_EF],
}

/// Exact descriptor/instance/global-relation/decomposition-claim prefix.
///
/// The schedule is a cached main trace generated from the admitted relation
/// and terminal descriptor.  Main rows cannot select a different source kind.
pub struct FixedMultiAirTerminalPrefixAir {
    pub transcript_bus: TranscriptBus,
    pub binding_bus: FixedMultiAirTerminalBindingBus,
    pub instance_bus: FixedMultiAirTerminalInstanceValueBus,
    pub beta_bus: FixedMultiAirBetaCoordinateBus,
    pub global_bus: FixedMultiAirGlobalClaimBus,
    pub region_claim_bus: FixedMultiAirRegionClaimBus,
    pub padding_claim_bus: FixedMultiAirPaddingClaimBus,
    pub relation_digest: Digest,
    pub alpha_len: usize,
    pub beta_len: usize,
    /// Setup-fixed number of authenticated consumers of the complete
    /// relation/root binding. The aggregate terminal circuit uses two: the
    /// exact WHIR prefix and the outer public statement.
    pub binding_lookup_count: usize,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirTerminalPrefixAir {}
impl PartitionedBaseAir<F> for FixedMultiAirTerminalPrefixAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirPrefixScheduleCols::<F>::width()]
    }

    fn common_main_width(&self) -> usize {
        FixedMultiAirPrefixCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirTerminalPrefixAir {}
impl BaseAir<F> for FixedMultiAirTerminalPrefixAir {
    fn width(&self) -> usize {
        FixedMultiAirPrefixScheduleCols::<F>::width() + FixedMultiAirPrefixCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirTerminalPrefixAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        let cached_row = builder.cached_mains()[0]
            .row_slice(0)
            .expect("fixed terminal prefix schedule row")
            .to_vec();
        let next_cached_row = builder.cached_mains()[0]
            .row_slice(1)
            .expect("fixed next terminal prefix schedule row")
            .to_vec();
        let main_row = builder
            .common_main()
            .row_slice(0)
            .expect("fixed terminal prefix row")
            .to_vec();
        let next_main_row = builder
            .common_main()
            .row_slice(1)
            .expect("fixed next terminal prefix row")
            .to_vec();
        let schedule: &FixedMultiAirPrefixScheduleCols<AB::Var> = cached_row.as_slice().borrow();
        let next_schedule: &FixedMultiAirPrefixScheduleCols<AB::Var> =
            next_cached_row.as_slice().borrow();
        let local: &FixedMultiAirPrefixCols<AB::Var> = main_row.as_slice().borrow();
        let next: &FixedMultiAirPrefixCols<AB::Var> = next_main_row.as_slice().borrow();

        for flag in [schedule.active, schedule.is_first, schedule.is_last]
            .into_iter()
            .chain(schedule.source_flags)
            .chain(schedule.root_flags)
        {
            builder.assert_bool(flag);
        }
        let source_sum = schedule
            .source_flags
            .iter()
            .fold(AB::Expr::ZERO, |sum, &flag| sum + flag);
        builder.assert_eq(source_sum, schedule.active);
        let root_sum = schedule
            .root_flags
            .iter()
            .fold(AB::Expr::ZERO, |sum, &flag| sum + flag);
        builder.assert_eq(root_sum, schedule.source_flags[SOURCE_ROOT]);
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
        same.assert_zero(next_schedule.is_first);
        same.assert_eq(next.tidx, local.tidx + AB::Expr::from_usize(D_EF));
        assert_array_eq(&mut same, next.descriptor_root, local.descriptor_root);
        assert_array_eq(&mut same, next.instance_root, local.instance_root);
        assert_array_eq(&mut same, next.eta, local.eta);
        assert_array_eq(&mut same, next.one, local.one);

        assert_array_eq(
            &mut builder.when(schedule.is_first),
            local.descriptor_root,
            local.instance_root.map(Into::into),
        );
        let constant = schedule.constant.map(Into::into);
        assert_array_eq(
            &mut builder.when(schedule.source_flags[SOURCE_CONSTANT]),
            local.value,
            constant,
        );
        let selected_root_limb = schedule
            .root_flags
            .iter()
            .zip(local.descriptor_root)
            .fold(AB::Expr::ZERO, |sum, (&flag, limb)| sum + flag * limb);
        let root_value = core::array::from_fn(|limb| {
            if limb == 0 {
                selected_root_limb.clone()
            } else {
                AB::Expr::ZERO
            }
        });
        assert_array_eq(
            &mut builder.when(schedule.source_flags[SOURCE_ROOT]),
            local.value,
            root_value,
        );
        assert_array_eq(
            &mut builder.when(schedule.source_flags[SOURCE_ETA]),
            local.value,
            local.eta.map(Into::into),
        );

        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            local.tidx,
            local.value,
            schedule.active,
        );
        self.binding_bus.send(
            builder,
            FixedMultiAirTerminalBindingMessage {
                relation_digest: self.relation_digest.map(Into::into),
                root: local.instance_root.map(Into::into),
                alpha_len: AB::Expr::from_usize(self.alpha_len),
                beta_len: AB::Expr::from_usize(self.beta_len),
            },
            schedule.is_first * AB::Expr::from_usize(self.binding_lookup_count),
        );
        self.global_bus.add_key_with_lookups(
            builder,
            FixedMultiAirGlobalClaimMessage {
                eta: local.eta.map(Into::into),
                one: local.one.map(Into::into),
            },
            schedule.global_lookup_count,
        );

        let source_section = schedule.source_flags[SOURCE_MU]
            + schedule.source_flags[SOURCE_BETA] * AB::Expr::from_usize(2)
            + schedule.source_flags[SOURCE_ETA] * AB::Expr::from_usize(3);
        self.instance_bus.add_key_with_lookups(
            builder,
            FixedMultiAirTerminalInstanceValueMessage {
                section: source_section,
                coordinate: schedule.source_index.into(),
                value: local.value.map(Into::into),
            },
            schedule.instance_lookup_count,
        );
        self.beta_bus.add_key_with_lookups(
            builder,
            FixedMultiAirBetaCoordinateMessage {
                coordinate: schedule.source_index.into(),
                value: local.value.map(Into::into),
            },
            schedule.beta_lookup_count,
        );
        self.region_claim_bus.add_key_with_lookups(
            builder,
            FixedMultiAirRegionClaimMessage {
                region: schedule.source_index.into(),
                claim: local.value.map(Into::into),
            },
            schedule.region_lookup_count,
        );
        self.padding_claim_bus.add_key_with_lookups(
            builder,
            FixedMultiAirPaddingClaimMessage {
                present: AB::Expr::ONE,
                claim: local.value.map(Into::into),
            },
            schedule.padding_lookup_count,
        );
    }
}

#[derive(Clone, Debug)]
struct PrefixScheduleEntry {
    source: usize,
    source_index: usize,
    constant: EF,
    global_lookup_count: usize,
    instance_lookup_count: usize,
    beta_lookup_count: usize,
    region_lookup_count: usize,
    padding_lookup_count: usize,
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirTerminalPrefixTraceOutput {
    pub cached: RowMajorMatrix<F>,
    pub common: RowMajorMatrix<F>,
    pub end_tidx: usize,
}

pub fn generate_fixed_multi_air_terminal_prefix_traces(
    descriptor: &TerminalDescriptor<Digest>,
    relation: &FixedMultiAirPesatIndex<F, Digest>,
    instance: &AccumulatorInstance<EF, Digest>,
    proof: &FixedMultiAirTerminalProof<EF>,
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
    lookup_counts: &FixedMultiAirTerminalPrefixLookupCounts,
    required_height: Option<usize>,
) -> Result<FixedMultiAirTerminalPrefixTraceOutput, FixedMultiAirTerminalPrefixTraceError> {
    generate_fixed_multi_air_terminal_prefix_traces_inner(
        descriptor,
        relation,
        instance,
        proof,
        transcript,
        start_tidx,
        lookup_counts,
        0,
        required_height,
    )
}

/// Final-wrapper-only prefix generation. The direct-final statement performs
/// one additional lookup of every accumulator-instance coordinate. This
/// private embedding API keeps that fixed multiplicity out of proof witness
/// data and does not alter the independent beta-coordinate bus counts.
pub(super) fn generate_fixed_multi_air_terminal_prefix_traces_for_final_wrapper(
    descriptor: &TerminalDescriptor<Digest>,
    relation: &FixedMultiAirPesatIndex<F, Digest>,
    instance: &AccumulatorInstance<EF, Digest>,
    proof: &FixedMultiAirTerminalProof<EF>,
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
    lookup_counts: &FixedMultiAirTerminalPrefixLookupCounts,
    required_height: Option<usize>,
) -> Result<FixedMultiAirTerminalPrefixTraceOutput, FixedMultiAirTerminalPrefixTraceError> {
    generate_fixed_multi_air_terminal_prefix_traces_inner(
        descriptor,
        relation,
        instance,
        proof,
        transcript,
        start_tidx,
        lookup_counts,
        1,
        required_height,
    )
}

#[allow(clippy::too_many_arguments)]
fn generate_fixed_multi_air_terminal_prefix_traces_inner(
    descriptor: &TerminalDescriptor<Digest>,
    relation: &FixedMultiAirPesatIndex<F, Digest>,
    instance: &AccumulatorInstance<EF, Digest>,
    proof: &FixedMultiAirTerminalProof<EF>,
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
    lookup_counts: &FixedMultiAirTerminalPrefixLookupCounts,
    external_instance_lookup_count: usize,
    required_height: Option<usize>,
) -> Result<FixedMultiAirTerminalPrefixTraceOutput, FixedMultiAirTerminalPrefixTraceError> {
    if descriptor.root != instance.rt
        || instance.alpha.len() != descriptor.log_codeword_len as usize
        || instance.beta.len() != relation.pesat_shape().beta_len()
        || proof.region_claims.len() != relation.region_count()
        || proof.region_proofs.len() != relation.region_count()
        || proof.padding_claim.is_some() != (relation.description().padding_constraint_count != 0)
        || lookup_counts.alpha.len() != instance.alpha.len()
        || lookup_counts.beta.len() != instance.beta.len()
        || lookup_counts.region_claims.len() != proof.region_claims.len()
    {
        return Err(FixedMultiAirTerminalPrefixTraceError::Shape);
    }
    let mut schedule = Vec::new();
    push_const(&mut schedule, TERMINAL_DESCRIPTOR_DOMAIN_TAG);
    schedule[0].global_lookup_count = lookup_counts.global_claim;
    for index in 0..DIGEST_SIZE {
        push_source(&mut schedule, SOURCE_ROOT, index);
    }
    for word in descriptor.metadata_words() {
        push_const(&mut schedule, word);
    }
    for index in 0..instance.alpha.len() {
        let mut entry = source(SOURCE_ALPHA, index);
        entry.instance_lookup_count = lookup_counts.alpha[index]
            .checked_add(external_instance_lookup_count)
            .ok_or(FixedMultiAirTerminalPrefixTraceError::Shape)?;
        schedule.push(entry);
    }
    let mut mu = source(SOURCE_MU, 0);
    mu.instance_lookup_count = lookup_counts
        .mu
        .checked_add(external_instance_lookup_count)
        .ok_or(FixedMultiAirTerminalPrefixTraceError::Shape)?;
    schedule.push(mu);
    for index in 0..instance.beta.len() {
        let mut entry = source(SOURCE_BETA, index);
        // Every accumulator coordinate is consumed once by the fixed/native
        // bridge. Typed beta-coordinate consumers are accounted independently
        // by `beta_lookup_count`; conflating the two fanouts overproduces the
        // instance catalog by every internal beta lookup.
        entry.instance_lookup_count = 1usize
            .checked_add(external_instance_lookup_count)
            .ok_or(FixedMultiAirTerminalPrefixTraceError::Shape)?;
        entry.beta_lookup_count = lookup_counts.beta[index];
        schedule.push(entry);
    }
    let mut eta = source(SOURCE_ETA, 0);
    eta.instance_lookup_count = lookup_counts
        .eta
        .checked_add(external_instance_lookup_count)
        .ok_or(FixedMultiAirTerminalPrefixTraceError::Shape)?;
    schedule.push(eta);

    push_const(&mut schedule, FIXED_TERMINAL_STATEMENT_TAG);
    push_const(&mut schedule, FIXED_MULTI_AIR_TERMINAL_VERSION as u64);
    let description = relation.canonical_description_bytes();
    push_const(&mut schedule, description.len() as u64);
    for &byte in description {
        push_const(&mut schedule, byte as u64);
    }
    push_const(&mut schedule, instance.beta.len() as u64);
    for index in 0..instance.beta.len() {
        push_source(&mut schedule, SOURCE_BETA, index);
    }
    push_source(&mut schedule, SOURCE_ETA, 0);

    push_const(&mut schedule, FIXED_TERMINAL_REGION_CLAIMS_TAG);
    push_const(&mut schedule, proof.region_claims.len() as u64);
    for index in 0..proof.region_claims.len() {
        let mut entry = source(SOURCE_REGION_CLAIM, index);
        entry.region_lookup_count = lookup_counts.region_claims[index];
        schedule.push(entry);
    }
    push_const(&mut schedule, u64::from(proof.padding_claim.is_some()));
    if proof.padding_claim.is_some() {
        let mut entry = source(SOURCE_PADDING_CLAIM, 0);
        entry.padding_lookup_count = lookup_counts.padding_claim;
        schedule.push(entry);
    }

    let height = required_height.unwrap_or_else(|| schedule.len().next_power_of_two());
    if height < schedule.len() {
        return Err(FixedMultiAirTerminalPrefixTraceError::Shape);
    }
    let cached_width = FixedMultiAirPrefixScheduleCols::<F>::width();
    let common_width = FixedMultiAirPrefixCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    let one = *instance
        .beta
        .get(relation.pesat_shape().log_constraints)
        .ok_or(FixedMultiAirTerminalPrefixTraceError::Shape)?;
    let mut tidx = start_tidx;
    for (row_index, entry) in schedule.iter().enumerate() {
        let value = source_value(entry, instance, proof, descriptor)?;
        expect_ext(transcript, tidx, value)?;
        let cached_row = &mut cached[row_index * cached_width..(row_index + 1) * cached_width];
        let schedule_cols: &mut FixedMultiAirPrefixScheduleCols<F> = cached_row.borrow_mut();
        schedule_cols.active = F::ONE;
        schedule_cols.is_first = F::from_bool(row_index == 0);
        schedule_cols.is_last = F::from_bool(row_index + 1 == schedule.len());
        schedule_cols.source_flags[entry.source] = F::ONE;
        schedule_cols.source_index = F::from_usize(entry.source_index);
        if entry.source == SOURCE_ROOT {
            schedule_cols.root_flags[entry.source_index] = F::ONE;
        }
        schedule_cols.global_lookup_count = F::from_usize(entry.global_lookup_count);
        schedule_cols.instance_lookup_count = F::from_usize(entry.instance_lookup_count);
        schedule_cols.beta_lookup_count = F::from_usize(entry.beta_lookup_count);
        schedule_cols.region_lookup_count = F::from_usize(entry.region_lookup_count);
        schedule_cols.padding_lookup_count = F::from_usize(entry.padding_lookup_count);
        copy_ext(&mut schedule_cols.constant, entry.constant);

        let common_row = &mut common[row_index * common_width..(row_index + 1) * common_width];
        let cols: &mut FixedMultiAirPrefixCols<F> = common_row.borrow_mut();
        cols.tidx = F::from_usize(tidx);
        copy_ext(&mut cols.value, value);
        cols.descriptor_root.copy_from_slice(&descriptor.root);
        cols.instance_root.copy_from_slice(&instance.rt);
        copy_ext(&mut cols.eta, instance.eta);
        copy_ext(&mut cols.one, one);
        tidx += D_EF;
    }
    Ok(FixedMultiAirTerminalPrefixTraceOutput {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        end_tidx: tidx,
    })
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FixedMultiAirRegionPrefixCols<T> {
    pub tidx: T,
    pub value: [T; D_EF],
    pub claim: [T; D_EF],
}

pub struct FixedMultiAirRegionPrefixAir {
    pub transcript_bus: TranscriptBus,
    pub beta_bus: FixedMultiAirBetaCoordinateBus,
    pub region_claim_bus: FixedMultiAirRegionClaimBus,
    pub start_bus: FixedMultiAirRegionStartBus,
    pub region: usize,
}

impl BaseAirWithPublicValues<F> for FixedMultiAirRegionPrefixAir {}
impl PartitionedBaseAir<F> for FixedMultiAirRegionPrefixAir {
    fn cached_main_widths(&self) -> Vec<usize> {
        vec![FixedMultiAirPrefixScheduleCols::<F>::width()]
    }
    fn common_main_width(&self) -> usize {
        FixedMultiAirRegionPrefixCols::<F>::width()
    }
}
impl ColumnsAir for FixedMultiAirRegionPrefixAir {}
impl BaseAir<F> for FixedMultiAirRegionPrefixAir {
    fn width(&self) -> usize {
        FixedMultiAirPrefixScheduleCols::<F>::width() + FixedMultiAirRegionPrefixCols::<F>::width()
    }
}

impl<AB> Air<AB> for FixedMultiAirRegionPrefixAir
where
    AB: PartitionedAirBuilder<F = F> + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let cached_row = builder.cached_mains()[0].row_slice(0).unwrap().to_vec();
        let next_cached_row = builder.cached_mains()[0].row_slice(1).unwrap().to_vec();
        let main_row = builder.common_main().row_slice(0).unwrap().to_vec();
        let next_main_row = builder.common_main().row_slice(1).unwrap().to_vec();
        let schedule: &FixedMultiAirPrefixScheduleCols<AB::Var> = cached_row.as_slice().borrow();
        let next_schedule: &FixedMultiAirPrefixScheduleCols<AB::Var> =
            next_cached_row.as_slice().borrow();
        let local: &FixedMultiAirRegionPrefixCols<AB::Var> = main_row.as_slice().borrow();
        let next: &FixedMultiAirRegionPrefixCols<AB::Var> = next_main_row.as_slice().borrow();
        let source_sum = schedule
            .source_flags
            .iter()
            .fold(AB::Expr::ZERO, |sum, &x| sum + x);
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
        assert_array_eq(&mut same, next.claim, local.claim);
        assert_array_eq(
            &mut builder.when(schedule.source_flags[SOURCE_CONSTANT]),
            local.value,
            schedule.constant.map(Into::into),
        );
        assert_array_eq(
            &mut builder.when(schedule.source_flags[SOURCE_REGION_CLAIM]),
            local.value,
            local.claim.map(Into::into),
        );
        self.transcript_bus.observe_ext(
            builder,
            AB::Expr::ZERO,
            local.tidx,
            local.value,
            schedule.active,
        );
        self.beta_bus.lookup_key(
            builder,
            FixedMultiAirBetaCoordinateMessage {
                coordinate: schedule.source_index.into(),
                value: local.value.map(Into::into),
            },
            schedule.source_flags[SOURCE_BETA],
        );
        self.region_claim_bus.lookup_key(
            builder,
            FixedMultiAirRegionClaimMessage {
                region: AB::Expr::from_usize(self.region),
                claim: local.claim.map(Into::into),
            },
            schedule.source_flags[SOURCE_REGION_CLAIM],
        );
        self.start_bus.send(
            builder,
            FixedMultiAirRegionStartMessage {
                region: AB::Expr::from_usize(self.region),
                tidx: AB::Expr::from(local.tidx) + AB::Expr::from_usize(D_EF),
                claim: local.claim.map(Into::into),
            },
            schedule.is_last,
        );
    }
}

pub fn generate_fixed_multi_air_region_prefix_traces(
    region: usize,
    relation: &FixedMultiAirPesatIndex<F, Digest>,
    beta: &[EF],
    claim: EF,
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
    required_height: Option<usize>,
) -> Result<FixedMultiAirTerminalPrefixTraceOutput, FixedMultiAirTerminalPrefixTraceError> {
    let local = relation
        .region_relation(region)
        .ok_or(FixedMultiAirTerminalPrefixTraceError::Shape)?;
    let description = relation
        .description()
        .regions
        .get(region)
        .ok_or(FixedMultiAirTerminalPrefixTraceError::Shape)?;
    let local_log = local.pesat_shape().log_constraints;
    let global_log = relation.pesat_shape().log_constraints;
    let explicit_start = description.explicit_offset as usize;
    let explicit_end = explicit_start + description.explicit_len as usize;
    if beta.len() != relation.pesat_shape().beta_len()
        || local_log > global_log
        || explicit_end > beta.len()
    {
        return Err(FixedMultiAirTerminalPrefixTraceError::Shape);
    }
    let mut schedule = Vec::new();
    push_const(&mut schedule, FIXED_TERMINAL_REGION_TAG);
    push_const(&mut schedule, region as u64);
    push_const(&mut schedule, COMPACT_COLUMNS_TRANSCRIPT_TAG);
    push_const(&mut schedule, COMPACT_COLUMNS_TRANSCRIPT_VERSION as u64);
    push_const(
        &mut schedule,
        local.canonical_description_bytes().len() as u64,
    );
    for &byte in local.canonical_description_bytes() {
        push_const(&mut schedule, byte as u64);
    }
    push_const(&mut schedule, global_log as u64);
    for coordinate in 0..global_log {
        push_source(&mut schedule, SOURCE_BETA, coordinate);
    }
    push_const(&mut schedule, description.constraint_offset);
    push_const(&mut schedule, description.constraint_count);
    push_const(
        &mut schedule,
        u64::from(relation.description().exact_max_degree - description.exact_max_degree),
    );
    push_const(&mut schedule, u64::from(1 + description.explicit_len));
    push_source(&mut schedule, SOURCE_BETA, global_log);
    for coordinate in explicit_start..explicit_end {
        push_source(&mut schedule, SOURCE_BETA, global_log + coordinate);
    }
    push_source(&mut schedule, SOURCE_REGION_CLAIM, region);
    push_const(&mut schedule, local.pesat_shape().witness_len() as u64);

    let height = required_height.unwrap_or_else(|| schedule.len().next_power_of_two());
    if height < schedule.len() {
        return Err(FixedMultiAirTerminalPrefixTraceError::Shape);
    }
    let cached_width = FixedMultiAirPrefixScheduleCols::<F>::width();
    let common_width = FixedMultiAirRegionPrefixCols::<F>::width();
    let mut cached = F::zero_vec(height * cached_width);
    let mut common = F::zero_vec(height * common_width);
    let mut tidx = start_tidx;
    for (row_index, entry) in schedule.iter().enumerate() {
        let value = match entry.source {
            SOURCE_CONSTANT => entry.constant,
            SOURCE_BETA => beta[entry.source_index],
            SOURCE_REGION_CLAIM => claim,
            _ => return Err(FixedMultiAirTerminalPrefixTraceError::Shape),
        };
        expect_ext(transcript, tidx, value)?;
        let cached_row = &mut cached[row_index * cached_width..(row_index + 1) * cached_width];
        let schedule_cols: &mut FixedMultiAirPrefixScheduleCols<F> = cached_row.borrow_mut();
        schedule_cols.active = F::ONE;
        schedule_cols.is_first = F::from_bool(row_index == 0);
        schedule_cols.is_last = F::from_bool(row_index + 1 == schedule.len());
        schedule_cols.source_flags[entry.source] = F::ONE;
        schedule_cols.source_index = F::from_usize(entry.source_index);
        copy_ext(&mut schedule_cols.constant, entry.constant);
        let common_row = &mut common[row_index * common_width..(row_index + 1) * common_width];
        let cols: &mut FixedMultiAirRegionPrefixCols<F> = common_row.borrow_mut();
        cols.tidx = F::from_usize(tidx);
        copy_ext(&mut cols.value, value);
        copy_ext(&mut cols.claim, claim);
        tidx += D_EF;
    }
    Ok(FixedMultiAirTerminalPrefixTraceOutput {
        cached: RowMajorMatrix::new(cached, cached_width),
        common: RowMajorMatrix::new(common, common_width),
        end_tidx: tidx,
    })
}

fn source(source: usize, source_index: usize) -> PrefixScheduleEntry {
    PrefixScheduleEntry {
        source,
        source_index,
        constant: EF::ZERO,
        global_lookup_count: 0,
        instance_lookup_count: 0,
        beta_lookup_count: 0,
        region_lookup_count: 0,
        padding_lookup_count: 0,
    }
}

fn push_source(schedule: &mut Vec<PrefixScheduleEntry>, source_kind: usize, index: usize) {
    schedule.push(source(source_kind, index));
}

fn push_const(schedule: &mut Vec<PrefixScheduleEntry>, value: u64) {
    let mut entry = source(SOURCE_CONSTANT, 0);
    entry.constant = EF::from_u64(value);
    schedule.push(entry);
}

fn source_value(
    entry: &PrefixScheduleEntry,
    instance: &AccumulatorInstance<EF, Digest>,
    proof: &FixedMultiAirTerminalProof<EF>,
    descriptor: &TerminalDescriptor<Digest>,
) -> Result<EF, FixedMultiAirTerminalPrefixTraceError> {
    Ok(match entry.source {
        SOURCE_CONSTANT => entry.constant,
        SOURCE_ROOT => EF::from(
            *descriptor
                .root
                .get(entry.source_index)
                .ok_or(FixedMultiAirTerminalPrefixTraceError::Shape)?,
        ),
        SOURCE_ALPHA => *instance
            .alpha
            .get(entry.source_index)
            .ok_or(FixedMultiAirTerminalPrefixTraceError::Shape)?,
        SOURCE_MU => instance.mu,
        SOURCE_BETA => *instance
            .beta
            .get(entry.source_index)
            .ok_or(FixedMultiAirTerminalPrefixTraceError::Shape)?,
        SOURCE_ETA => instance.eta,
        SOURCE_REGION_CLAIM => *proof
            .region_claims
            .get(entry.source_index)
            .ok_or(FixedMultiAirTerminalPrefixTraceError::Shape)?,
        SOURCE_PADDING_CLAIM => proof
            .padding_claim
            .ok_or(FixedMultiAirTerminalPrefixTraceError::Shape)?,
        _ => return Err(FixedMultiAirTerminalPrefixTraceError::Shape),
    })
}

fn expect_ext(
    transcript: &TranscriptLog<F, [F; 16]>,
    tidx: usize,
    expected: EF,
) -> Result<(), FixedMultiAirTerminalPrefixTraceError> {
    let end = tidx
        .checked_add(D_EF)
        .ok_or(FixedMultiAirTerminalPrefixTraceError::Transcript)?;
    let values = transcript
        .values()
        .get(tidx..end)
        .ok_or(FixedMultiAirTerminalPrefixTraceError::Transcript)?;
    let samples = transcript
        .samples()
        .get(tidx..end)
        .ok_or(FixedMultiAirTerminalPrefixTraceError::Transcript)?;
    if samples.iter().any(|&sample| sample)
        || EF::from_basis_coefficients_slice(values) != Some(expected)
    {
        return Err(FixedMultiAirTerminalPrefixTraceError::Transcript);
    }
    Ok(())
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

const _: () = assert!(D_EF == 4);

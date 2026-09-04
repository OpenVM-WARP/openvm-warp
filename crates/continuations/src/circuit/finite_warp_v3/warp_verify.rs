//! Exact-finite WARP `Verify` receipt producer for the bounded v3 wrapper.
//!
//! The complete recursive VACC verifier is the sole producer of
//! [`FiniteWarpV3ExactVaccAuthorityBus`]. This module only consumes that
//! authority, constrains cross-call accumulator/checkpoint continuity, and
//! emits the v3 wrapper receipt. It cannot manufacture authority from host
//! metadata, does not replay the source PESAT, and does not run terminal
//! Decide.

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, LookupBus},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    warp_accum::EXACT_FINITE_WARP_TRANSCRIPT_VERSION,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, F};

use super::{
    FiniteWarpV3CallReceiptBus, FiniteWarpV3CallReceiptMessage, FiniteWarpV3FinalVaccCheckpointBus,
    FiniteWarpV3FinalVaccCheckpointMessage, FINITE_WARP_V3_MAX_CALLS,
    FINITE_WARP_V3_MAX_INPUT_ARITY,
};

const _: () = assert!(FINITE_WARP_V3_MAX_CALLS == 3);
const _: () = assert!(FINITE_WARP_V3_MAX_INPUT_ARITY == 64);

/// Complete authority exported by the generalized exact-finite recursive
/// verifier after it has checked one ordinary WARP invocation.
///
/// The existing v19 protocol bus intentionally omits high-arity and external
/// index metadata because it serves the legacy arity-two path. Requiring this
/// additional typed lookup prevents that verifier from being accidentally
/// treated as an exact-finite verifier. No receipt can balance unless a lower
/// AIR has authenticated every field below against the exact transcript,
/// sumchecks, openings, and output instance.
#[derive(Clone, Debug)]
pub struct FiniteWarpV3ExactVaccAuthorityMessage<T> {
    pub proof_idx: T,
    pub call_index: T,
    pub source_start: T,
    pub source_count: T,
    pub input_arity: T,
    pub has_prior: T,
    pub transcript_version: T,
    pub protocol_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub warp_index_digest: [T; DIGEST_SIZE],
    pub setup_digest: [T; DIGEST_SIZE],
    pub schedule_digest: [T; DIGEST_SIZE],
    pub start_tidx: T,
    pub start_sample_count: T,
    pub start_state: [T; POSEIDON2_WIDTH],
    pub end_tidx: T,
    pub end_sample_count: T,
    pub end_state: [T; POSEIDON2_WIDTH],
    pub fresh_stacked_root: [T; DIGEST_SIZE],
    pub prior_accumulator_root: [T; DIGEST_SIZE],
    pub output_accumulator_root: [T; DIGEST_SIZE],
    pub prior_accumulator_digest: [T; DIGEST_SIZE],
    pub output_accumulator_digest: [T; DIGEST_SIZE],
}

impl<T: Clone> FiniteWarpV3ExactVaccAuthorityMessage<T> {
    #[must_use]
    pub fn to_vec(&self) -> Vec<T> {
        let mut values = Vec::with_capacity(10 + 9 * DIGEST_SIZE + 2 * POSEIDON2_WIDTH);
        values.extend([
            self.proof_idx.clone(),
            self.call_index.clone(),
            self.source_start.clone(),
            self.source_count.clone(),
            self.input_arity.clone(),
            self.has_prior.clone(),
            self.transcript_version.clone(),
        ]);
        values.extend_from_slice(&self.protocol_digest);
        values.extend_from_slice(&self.relation_digest);
        values.extend_from_slice(&self.warp_index_digest);
        values.extend_from_slice(&self.setup_digest);
        values.extend_from_slice(&self.schedule_digest);
        values.extend([self.start_tidx.clone(), self.start_sample_count.clone()]);
        values.extend_from_slice(&self.start_state);
        values.extend([self.end_tidx.clone(), self.end_sample_count.clone()]);
        values.extend_from_slice(&self.end_state);
        values.extend_from_slice(&self.fresh_stacked_root);
        values.extend_from_slice(&self.prior_accumulator_root);
        values.extend_from_slice(&self.output_accumulator_root);
        values.extend_from_slice(&self.prior_accumulator_digest);
        values.extend_from_slice(&self.output_accumulator_digest);
        values
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FiniteWarpV3ExactVaccAuthorityBus(LookupBus);

impl FiniteWarpV3ExactVaccAuthorityBus {
    #[must_use]
    pub const fn new(index: BusIndex) -> Self {
        Self(LookupBus::new(index))
    }

    #[must_use]
    pub const fn index(self) -> BusIndex {
        self.0.index
    }

    pub fn lookup_key<AB, T>(
        &self,
        builder: &mut AB,
        message: FiniteWarpV3ExactVaccAuthorityMessage<T>,
        enabled: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
        T: Into<AB::Expr> + Clone,
    {
        self.0.lookup_key(builder, message.to_vec(), enabled);
    }

    pub fn add_key_with_lookups<AB, T>(
        &self,
        builder: &mut AB,
        message: FiniteWarpV3ExactVaccAuthorityMessage<T>,
        lookups: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
        T: Into<AB::Expr> + Clone,
    {
        self.0
            .add_key_with_lookups(builder, message.to_vec(), lookups);
    }
}

/// Setup-fixed shape of one exact-finite WARP invocation.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FiniteWarpV3WarpVerifyCallProfile {
    pub active: bool,
    pub source_start: u32,
    pub source_count: u32,
    pub input_arity: u32,
}

impl FiniteWarpV3WarpVerifyCallProfile {
    #[must_use]
    pub const fn inactive() -> Self {
        Self {
            active: false,
            source_start: 0,
            source_count: 0,
            input_arity: 0,
        }
    }
}

/// Verifier-key data for the receipt producer.
///
/// `schedule_digest` commits to the complete exact-finite schedule, including
/// all inactive slots. `first_call_start.tidx` is the boundary immediately
/// after the separately verified schedule prefix. Its state/sample fields are
/// canonical zero because call zero verifies that prefix in the same
/// transcript AIR and therefore does not resume from a host-provided sponge
/// state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteWarpV3WarpVerifyProfile {
    pub transcript_version: u64,
    pub protocol_digest: Digest,
    pub relation_digest: Digest,
    pub warp_index_digest: Digest,
    /// Exact-finite transcript setup/index digest. This is compared verbatim
    /// with `NativeExactFiniteVaccTranscriptProfile::setup_digest`; it is not
    /// an alias for the wrapper component digest.
    pub setup_digest: Digest,
    pub verifier_component_digest: Digest,
    pub schedule_digest: Digest,
    pub first_call_start: FiniteWarpV3TranscriptCheckpoint,
    pub calls: [FiniteWarpV3WarpVerifyCallProfile; FINITE_WARP_V3_MAX_CALLS],
}

impl FiniteWarpV3WarpVerifyProfile {
    pub fn validate(&self) -> Result<(), FiniteWarpV3WarpVerifyError> {
        if self.transcript_version != EXACT_FINITE_WARP_TRANSCRIPT_VERSION {
            return Err(FiniteWarpV3WarpVerifyError::TranscriptVersion);
        }
        for (kind, digest) in [
            ("protocol", self.protocol_digest),
            ("relation", self.relation_digest),
            ("WARP index", self.warp_index_digest),
            ("WARP setup", self.setup_digest),
            ("component", self.verifier_component_digest),
            ("schedule", self.schedule_digest),
        ] {
            if digest.iter().all(|value| *value == F::ZERO) {
                return Err(FiniteWarpV3WarpVerifyError::UnsetDigest(kind));
            }
        }
        if self.first_call_start.tidx == 0
            || self.first_call_start.sample_count != 0
            || self.first_call_start.state != [F::ZERO; POSEIDON2_WIDTH]
        {
            return Err(FiniteWarpV3WarpVerifyError::InvalidPrefixCheckpoint);
        }

        let mut saw_inactive = false;
        let mut expected_source_start = 0u32;
        let mut active_count = 0usize;
        for (call_index, call) in self.calls.iter().copied().enumerate() {
            if !call.active {
                saw_inactive = true;
                if call != FiniteWarpV3WarpVerifyCallProfile::inactive() {
                    return Err(FiniteWarpV3WarpVerifyError::NonCanonicalInactive(
                        call_index,
                    ));
                }
                continue;
            }
            if saw_inactive {
                return Err(FiniteWarpV3WarpVerifyError::NonPrefixCalls);
            }
            active_count += 1;
            let prior_count = u32::from(call_index != 0);
            if call.source_start != expected_source_start
                || call.source_count == 0
                || call.input_arity < 2
                || call.input_arity > FINITE_WARP_V3_MAX_INPUT_ARITY
                || !call.input_arity.is_power_of_two()
                || call.source_count.checked_add(prior_count) != Some(call.input_arity)
            {
                return Err(FiniteWarpV3WarpVerifyError::InvalidCallShape(call_index));
            }
            expected_source_start = expected_source_start
                .checked_add(call.source_count)
                .ok_or(FiniteWarpV3WarpVerifyError::InvalidCallShape(call_index))?;
        }
        if active_count == 0 {
            return Err(FiniteWarpV3WarpVerifyError::EmptySchedule);
        }
        Ok(())
    }

    #[must_use]
    pub fn active_call_count(&self) -> usize {
        self.calls.iter().take_while(|call| call.active).count()
    }
}

/// Exact duplex checkpoint exported by `TranscriptAir`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FiniteWarpV3TranscriptCheckpoint {
    pub tidx: u32,
    pub sample_count: u32,
    pub state: [F; POSEIDON2_WIDTH],
}

/// Verifier-derived evidence for one ordinary WARP call.
///
/// This is private witness data. The AIR authenticates every field against a
/// recursive verifier bus before publishing a v3 receipt.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FiniteWarpV3WarpVerifyCallRecord {
    pub fresh_stacked_root: Digest,
    pub prior_accumulator_root: Digest,
    pub output_accumulator_root: Digest,
    pub prior_accumulator_digest: Digest,
    pub output_accumulator_digest: Digest,
    pub start: FiniteWarpV3TranscriptCheckpoint,
    pub end: FiniteWarpV3TranscriptCheckpoint,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteWarpV3WarpVerifyRecord {
    pub manifest_digest: Digest,
    pub calls: [FiniteWarpV3WarpVerifyCallRecord; FINITE_WARP_V3_MAX_CALLS],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FiniteWarpV3WarpVerifyError {
    TranscriptVersion,
    UnsetDigest(&'static str),
    InvalidPrefixCheckpoint,
    EmptySchedule,
    NonPrefixCalls,
    NonCanonicalInactive(usize),
    InvalidCallShape(usize),
    RecordShape(usize),
    AccumulatorChain(usize),
    TranscriptChain(usize),
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FiniteWarpV3WarpVerifyCols<T> {
    pub row_present: T,
    pub active: T,
    pub slot_flags: [T; FINITE_WARP_V3_MAX_CALLS],
    pub call_index: T,
    pub source_start: T,
    pub source_count: T,
    pub input_arity: T,
    pub has_prior: T,
    pub manifest_digest: [T; DIGEST_SIZE],
    pub fresh_stacked_root: [T; DIGEST_SIZE],
    pub prior_accumulator_root: [T; DIGEST_SIZE],
    pub output_accumulator_root: [T; DIGEST_SIZE],
    pub prior_accumulator_digest: [T; DIGEST_SIZE],
    pub output_accumulator_digest: [T; DIGEST_SIZE],
    pub start_tidx: T,
    pub start_sample_count: T,
    pub start_state: [T; POSEIDON2_WIDTH],
    pub end_tidx: T,
    pub end_sample_count: T,
    pub end_state: [T; POSEIDON2_WIDTH],
}

/// Bus bundle shared with the exact recursive WARP verifier.
#[derive(Clone, Copy, Debug)]
pub struct FiniteWarpV3WarpVerifyBuses {
    pub exact: FiniteWarpV3ExactVaccAuthorityBus,
    pub receipt: FiniteWarpV3CallReceiptBus,
    /// Complete dedicated handoff to the terminal transcript seam.  The
    /// message includes the authenticated output root and accumulator digest,
    /// so no host-side bridge can attach those values to a bare transcript
    /// checkpoint after WARP verification.
    pub final_vacc_checkpoint: FiniteWarpV3FinalVaccCheckpointBus,
}

/// Receipt adapter for authority emitted by the complete exact-finite VACC
/// verifier. This AIR is intentionally unable to produce authority itself.
#[derive(Clone, Debug)]
pub struct FiniteWarpV3WarpVerifyReceiptAir {
    pub profile: FiniteWarpV3WarpVerifyProfile,
    pub buses: FiniteWarpV3WarpVerifyBuses,
}

impl FiniteWarpV3WarpVerifyReceiptAir {
    pub fn new(
        profile: FiniteWarpV3WarpVerifyProfile,
        buses: FiniteWarpV3WarpVerifyBuses,
    ) -> Result<Self, FiniteWarpV3WarpVerifyError> {
        profile.validate()?;
        Ok(Self { profile, buses })
    }

    pub fn generate_trace(
        &self,
        record: &FiniteWarpV3WarpVerifyRecord,
    ) -> Result<RowMajorMatrix<F>, FiniteWarpV3WarpVerifyError> {
        self.profile.validate()?;
        if record.manifest_digest.iter().all(|value| *value == F::ZERO) {
            return Err(FiniteWarpV3WarpVerifyError::UnsetDigest("manifest"));
        }

        let active_count = self.profile.active_call_count();
        for call_index in 0..FINITE_WARP_V3_MAX_CALLS {
            let call = record.calls[call_index];
            if call_index >= active_count {
                if call != FiniteWarpV3WarpVerifyCallRecord::default() {
                    return Err(FiniteWarpV3WarpVerifyError::RecordShape(call_index));
                }
                continue;
            }
            if call.start.tidx >= call.end.tidx
                || call
                    .fresh_stacked_root
                    .iter()
                    .all(|value| *value == F::ZERO)
                || call
                    .output_accumulator_root
                    .iter()
                    .all(|value| *value == F::ZERO)
                || call
                    .output_accumulator_digest
                    .iter()
                    .all(|value| *value == F::ZERO)
            {
                return Err(FiniteWarpV3WarpVerifyError::RecordShape(call_index));
            }
            if call_index == 0 {
                if call.prior_accumulator_root != [F::ZERO; DIGEST_SIZE]
                    || call.prior_accumulator_digest != [F::ZERO; DIGEST_SIZE]
                {
                    return Err(FiniteWarpV3WarpVerifyError::AccumulatorChain(call_index));
                }
                if call.start != self.profile.first_call_start {
                    return Err(FiniteWarpV3WarpVerifyError::TranscriptChain(call_index));
                }
            } else {
                let previous = record.calls[call_index - 1];
                if call.prior_accumulator_root != previous.output_accumulator_root
                    || call.prior_accumulator_digest != previous.output_accumulator_digest
                {
                    return Err(FiniteWarpV3WarpVerifyError::AccumulatorChain(call_index));
                }
                if call.start != previous.end {
                    return Err(FiniteWarpV3WarpVerifyError::TranscriptChain(call_index));
                }
            }
        }

        let width = self.width();
        let mut values = F::zero_vec(4 * width);
        for call_index in 0..FINITE_WARP_V3_MAX_CALLS {
            let profile = self.profile.calls[call_index];
            let record_call = record.calls[call_index];
            let cols: &mut FiniteWarpV3WarpVerifyCols<F> =
                values[call_index * width..(call_index + 1) * width].borrow_mut();
            cols.row_present = F::ONE;
            cols.slot_flags[call_index] = F::ONE;
            if !profile.active {
                continue;
            }
            cols.active = F::ONE;
            cols.call_index = F::from_usize(call_index);
            cols.source_start = F::from_u32(profile.source_start);
            cols.source_count = F::from_u32(profile.source_count);
            cols.input_arity = F::from_u32(profile.input_arity);
            cols.has_prior = F::from_bool(call_index != 0);
            cols.manifest_digest = record.manifest_digest;
            cols.fresh_stacked_root = record_call.fresh_stacked_root;
            cols.prior_accumulator_root = record_call.prior_accumulator_root;
            cols.output_accumulator_root = record_call.output_accumulator_root;
            cols.prior_accumulator_digest = record_call.prior_accumulator_digest;
            cols.output_accumulator_digest = record_call.output_accumulator_digest;
            write_checkpoint(
                record_call.start,
                &mut cols.start_tidx,
                &mut cols.start_sample_count,
                &mut cols.start_state,
            );
            write_checkpoint(
                record_call.end,
                &mut cols.end_tidx,
                &mut cols.end_sample_count,
                &mut cols.end_state,
            );
        }
        Ok(RowMajorMatrix::new(values, width))
    }
}

impl BaseAir<F> for FiniteWarpV3WarpVerifyReceiptAir {
    fn width(&self) -> usize {
        FiniteWarpV3WarpVerifyCols::<F>::width()
    }
}

impl BaseAirWithPublicValues<F> for FiniteWarpV3WarpVerifyReceiptAir {}
impl PartitionedBaseAir<F> for FiniteWarpV3WarpVerifyReceiptAir {}

impl<AB> Air<AB> for FiniteWarpV3WarpVerifyReceiptAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(
            self.profile.validate().is_ok(),
            "invalid finite WARP v3 Verify profile"
        );
        let main = builder.main();
        let local_row = main.row_slice(0).expect("finite WARP v3 Verify row");
        let next_row = main.row_slice(1).expect("finite WARP v3 Verify next row");
        let local: &FiniteWarpV3WarpVerifyCols<AB::Var> = (*local_row).borrow();
        let next: &FiniteWarpV3WarpVerifyCols<AB::Var> = (*next_row).borrow();

        builder.assert_bool(local.row_present);
        builder.assert_bool(local.active);
        let mut slot_sum = AB::Expr::ZERO;
        for flag in local.slot_flags {
            builder.assert_bool(flag);
            slot_sum += flag;
        }
        builder.assert_eq(slot_sum, local.row_present);
        builder.when_first_row().assert_one(local.row_present);
        builder.when_first_row().assert_one(local.slot_flags[0]);
        builder.when_last_row().assert_zero(local.row_present);
        builder
            .when_transition()
            .assert_eq(next.slot_flags[0], AB::Expr::ZERO);
        builder
            .when_transition()
            .assert_eq(next.slot_flags[1], local.slot_flags[0]);
        builder
            .when_transition()
            .assert_eq(next.slot_flags[2], local.slot_flags[1]);
        builder
            .when_transition()
            .assert_eq(next.row_present, local.slot_flags[0] + local.slot_flags[1]);

        let expected_active = selected_u32::<AB>(
            &local.slot_flags,
            self.profile.calls.map(|call| u32::from(call.active)),
        );
        builder.assert_eq(local.active, expected_active);
        for (actual, expected) in [
            (
                local.call_index,
                selected_u32::<AB>(
                    &local.slot_flags,
                    core::array::from_fn(|index| {
                        if self.profile.calls[index].active {
                            index as u32
                        } else {
                            0
                        }
                    }),
                ),
            ),
            (
                local.source_start,
                selected_u32::<AB>(
                    &local.slot_flags,
                    self.profile.calls.map(|call| call.source_start),
                ),
            ),
            (
                local.source_count,
                selected_u32::<AB>(
                    &local.slot_flags,
                    self.profile.calls.map(|call| call.source_count),
                ),
            ),
            (
                local.input_arity,
                selected_u32::<AB>(
                    &local.slot_flags,
                    self.profile.calls.map(|call| call.input_arity),
                ),
            ),
            (
                local.has_prior,
                selected_u32::<AB>(
                    &local.slot_flags,
                    core::array::from_fn(|index| {
                        u32::from(index != 0 && self.profile.calls[index].active)
                    }),
                ),
            ),
        ] {
            builder.assert_eq(actual, expected);
        }
        builder.assert_bool(local.has_prior);

        let inactive = AB::Expr::ONE - Into::<AB::Expr>::into(local.active);
        // Slot selectors remain live on inactive configured slots; every
        // proof-derived field is nevertheless canonical zero.
        for value in local.as_slice().iter().skip(2 + FINITE_WARP_V3_MAX_CALLS) {
            builder.when(inactive.clone()).assert_zero(*value);
        }

        let enabled = local.active;
        let enabled_expr = Into::<AB::Expr>::into(enabled);
        let bootstrap =
            enabled_expr.clone() * (AB::Expr::ONE - Into::<AB::Expr>::into(local.has_prior));
        let proof_idx = Into::<AB::Expr>::into(local.call_index);
        let authority = FiniteWarpV3ExactVaccAuthorityMessage {
            proof_idx: proof_idx.clone(),
            call_index: local.call_index.into(),
            source_start: local.source_start.into(),
            source_count: local.source_count.into(),
            input_arity: local.input_arity.into(),
            has_prior: local.has_prior.into(),
            transcript_version: AB::Expr::from_u64(self.profile.transcript_version),
            protocol_digest: self.profile.protocol_digest.map(AB::Expr::from),
            relation_digest: self.profile.relation_digest.map(AB::Expr::from),
            warp_index_digest: self.profile.warp_index_digest.map(AB::Expr::from),
            setup_digest: self.profile.setup_digest.map(AB::Expr::from),
            schedule_digest: self.profile.schedule_digest.map(AB::Expr::from),
            start_tidx: local.start_tidx.into(),
            start_sample_count: local.start_sample_count.into(),
            start_state: local.start_state.map(Into::into),
            end_tidx: local.end_tidx.into(),
            end_sample_count: local.end_sample_count.into(),
            end_state: local.end_state.map(Into::into),
            fresh_stacked_root: local.fresh_stacked_root.map(Into::into),
            prior_accumulator_root: local.prior_accumulator_root.map(Into::into),
            output_accumulator_root: local.output_accumulator_root.map(Into::into),
            prior_accumulator_digest: local.prior_accumulator_digest.map(Into::into),
            output_accumulator_digest: local.output_accumulator_digest.map(Into::into),
        };
        self.buses.exact.lookup_key(builder, authority, enabled);

        let next_enabled = next.active;
        builder.when(bootstrap.clone()).assert_eq(
            local.start_tidx,
            AB::Expr::from_u32(self.profile.first_call_start.tidx),
        );
        builder.when(bootstrap.clone()).assert_eq(
            local.start_sample_count,
            AB::Expr::from_u32(self.profile.first_call_start.sample_count),
        );
        for (actual, expected) in local
            .start_state
            .iter()
            .zip(self.profile.first_call_start.state)
        {
            builder
                .when(bootstrap.clone())
                .assert_eq(*actual, AB::Expr::from(expected));
        }
        builder
            .when_transition()
            .when(next_enabled)
            .assert_eq(next.start_tidx, local.end_tidx);
        builder
            .when_transition()
            .when(next_enabled)
            .assert_eq(next.start_sample_count, local.end_sample_count);
        for limb in 0..POSEIDON2_WIDTH {
            builder
                .when_transition()
                .when(next_enabled)
                .assert_eq(next.start_state[limb], local.end_state[limb]);
        }
        for limb in 0..DIGEST_SIZE {
            builder
                .when_transition()
                .when(next_enabled)
                .assert_eq(next.manifest_digest[limb], local.manifest_digest[limb]);
            builder
                .when(bootstrap.clone())
                .assert_zero(local.prior_accumulator_root[limb]);
            builder
                .when(bootstrap.clone())
                .assert_zero(local.prior_accumulator_digest[limb]);
            builder.when_transition().when(next_enabled).assert_eq(
                next.prior_accumulator_root[limb],
                local.output_accumulator_root[limb],
            );
            builder.when_transition().when(next_enabled).assert_eq(
                next.prior_accumulator_digest[limb],
                local.output_accumulator_digest[limb],
            );
        }

        self.buses.receipt.add_key_with_lookups(
            builder,
            FiniteWarpV3CallReceiptMessage {
                protocol_digest: self.profile.protocol_digest.map(AB::Expr::from),
                relation_digest: self.profile.relation_digest.map(AB::Expr::from),
                warp_index_digest: self.profile.warp_index_digest.map(AB::Expr::from),
                verifier_component_digest: self
                    .profile
                    .verifier_component_digest
                    .map(AB::Expr::from),
                schedule_digest: self.profile.schedule_digest.map(AB::Expr::from),
                manifest_digest: local.manifest_digest.map(Into::into),
                call_index: local.call_index.into(),
                source_start: local.source_start.into(),
                source_count: local.source_count.into(),
                input_arity: local.input_arity.into(),
                fresh_stacked_root: local.fresh_stacked_root.map(Into::into),
                prior_accumulator_digest: local.prior_accumulator_digest.map(Into::into),
                output_accumulator_digest: local.output_accumulator_digest.map(Into::into),
            },
            enabled,
        );
        self.buses.final_vacc_checkpoint.add_key_with_lookups(
            builder,
            FiniteWarpV3FinalVaccCheckpointMessage {
                protocol_digest: self.profile.protocol_digest.map(AB::Expr::from),
                relation_digest: self.profile.relation_digest.map(AB::Expr::from),
                warp_index_digest: self.profile.warp_index_digest.map(AB::Expr::from),
                setup_digest: self.profile.setup_digest.map(AB::Expr::from),
                schedule_digest: self.profile.schedule_digest.map(AB::Expr::from),
                call_count: AB::Expr::from_usize(self.profile.active_call_count()),
                end_tidx: local.end_tidx.into(),
                end_sample_count: local.end_sample_count.into(),
                end_state: local.end_state.map(Into::into),
                output_accumulator_root: local.output_accumulator_root.map(Into::into),
                output_accumulator_digest: local.output_accumulator_digest.map(Into::into),
            },
            local.slot_flags[self.profile.active_call_count() - 1],
        );
    }
}

fn write_checkpoint(
    checkpoint: FiniteWarpV3TranscriptCheckpoint,
    tidx: &mut F,
    sample_count: &mut F,
    state: &mut [F; POSEIDON2_WIDTH],
) {
    *tidx = F::from_u32(checkpoint.tidx);
    *sample_count = F::from_u32(checkpoint.sample_count);
    *state = checkpoint.state;
}

fn selected_u32<AB: AirBuilder<F = F>>(
    selectors: &[AB::Var; FINITE_WARP_V3_MAX_CALLS],
    values: [u32; FINITE_WARP_V3_MAX_CALLS],
) -> AB::Expr
where
    AB::Var: Copy,
{
    selectors
        .iter()
        .zip(values)
        .fold(AB::Expr::ZERO, |sum, (selector, value)| {
            sum + Into::<AB::Expr>::into(*selector) * AB::Expr::from_u32(value)
        })
}

#[cfg(test)]
#[path = "warp_verify_tests.rs"]
mod tests;

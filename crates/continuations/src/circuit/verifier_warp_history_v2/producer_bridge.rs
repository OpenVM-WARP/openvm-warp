//! One-way adapters from the real v19 verifier producers to History v2.
//!
//! The fixed source is the ordinary `InnerCircuit<VerifierSubCircuit<4>>`.
//! Its aggregate `VmPvsAir` public values occur at setup-fixed coordinates of
//! the direct relation's explicit vector. Standard VACC exports the complete
//! authenticated fresh beta; this AIR consumes that exact message and binds
//! the fixed beta coordinates to the History batch boundary.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, D_EF, F};
use openvm_verify_stark_host::pvs::VmPvs;

use super::{
    CertifiedFixedMultiAirSourceBusV2, CertifiedFixedMultiAirSourceMessageV2,
    FixedSetupOpeningCertificateBusV2, FixedSetupOpeningCertificateMessageV2,
    VerifierWarpCertifiedActiveCountBusV2, VerifierWarpCertifiedActiveCountMessageV2,
    VerifierWarpCertifiedChildMessageV2, VerifierWarpSourceCertificateBusV2,
    VerifierWarpSourceCertificateMessageV2, VerifierWarpVaccCertificateBusV2,
    VerifierWarpVaccCertificateMessageV2, FIXED_SETUP_OPENING_PROTOCOL_V2,
    VERIFIER_WARP_HISTORY_PROTOCOL_V2, VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2,
    VERIFIER_WARP_VACC_INPUT_ARITY_V2,
};
use crate::circuit::native_warp_history_v19::{
    CertifiedDirectAirVaccInputBusV19, CertifiedDirectAirVaccInputMessageV19,
    CertifiedFreshExplicitDigestBusV19, CertifiedFreshExplicitDigestMessageV19,
    CertifiedWarpReplayBusV19, CertifiedWarpReplayMessageV19, FixedSourcePublicValueBusV19,
    FixedSourcePublicValueMessageV19, LogUpOnlyHistoryBusV19, LogUpOnlyHistoryMessageV19,
    MAX_FRESH_BETA_LEN_V19, MAX_RAW_MESSAGE_POINT_LEN_V19, NATIVE_WARP_HISTORY_PROTOCOL_V19,
};

/// Origin of one coordinate in the fixed relation's complete explicit vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifierWarpFixedPublicValueSourceV2 {
    /// Setup-derived value, including coordinate zero's distinguished one and
    /// the child verifier's exact `DagCommitPvs` digest.
    TrustedConstant(F),
    /// Dynamic aggregate VM public-value coordinate constrained by this AIR.
    VmPvsCoordinate(usize),
}

/// Exhaustive setup-fixed adapter for the relation's explicit vector. There
/// is one source entry per global explicit coordinate; omission, duplication,
/// and local-vs-global index confusion therefore fail during key assembly.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierWarpFixedPublicValuesProfileV2 {
    pub log_constraints: usize,
    pub sources: Arc<[VerifierWarpFixedPublicValueSourceV2]>,
    /// Number of complete fixed-verifier proofs in this History statement.
    ///
    /// Occupancy is deliberately absent from this setup-owned layout.  The
    /// exact count is certified from the committed source message by
    /// `VerifierWarpActiveCountFunctionalAirV2`.
    pub batch_count: usize,
}

impl VerifierWarpFixedPublicValuesProfileV2 {
    pub fn validate(&self) -> Result<(), &'static str> {
        let explicit_len = self.sources.len();
        let beta_len = self
            .log_constraints
            .checked_add(explicit_len)
            .ok_or("verifier-WARP beta length overflow")?;
        if explicit_len == 0 || beta_len > MAX_FRESH_BETA_LEN_V19 || self.batch_count == 0 {
            return Err("invalid verifier-WARP fixed public-value profile");
        }
        if self.sources[0] != VerifierWarpFixedPublicValueSourceV2::TrustedConstant(F::ONE) {
            return Err("verifier-WARP distinguished-one coordinate");
        }
        let mut vm_seen = vec![false; VmPvs::<u8>::width()];
        for source in self.sources.iter() {
            if let VerifierWarpFixedPublicValueSourceV2::VmPvsCoordinate(offset) = *source {
                let seen = vm_seen
                    .get_mut(offset)
                    .ok_or("verifier-WARP VmPvs coordinate")?;
                if *seen {
                    return Err("duplicate verifier-WARP VmPvs coordinate");
                }
                *seen = true;
            }
        }
        if vm_seen.iter().any(|seen| !seen) {
            return Err("incomplete verifier-WARP VmPvs coordinates");
        }
        Ok(())
    }

    #[must_use]
    pub fn beta_len(&self) -> usize {
        self.log_constraints + self.sources.len()
    }

    #[must_use]
    pub fn explicit_len(&self) -> usize {
        self.sources.len()
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct VerifierWarpProducerBridgePrepColsV2<T> {
    pub active: T,
    pub proof_index: T,
    pub batch_index_lo: T,
    pub batch_index_hi: T,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct VerifierWarpProducerBridgeColsV2<T> {
    pub active: T,
    pub vm_pvs: VmPvs<T>,
    pub logup: LogUpOnlyHistoryMessageV19<T>,
    pub fixed_source: CertifiedFixedMultiAirSourceMessageV2<T>,
    pub fresh_input: CertifiedDirectAirVaccInputMessageV19<T>,
    pub fresh_explicit: CertifiedFreshExplicitDigestMessageV19<T>,
    pub replay: CertifiedWarpReplayMessageV19<T>,
    /// Narrow V3 setup-authority compatibility certificate. This is copied
    /// into the bridge trace only so the typed permutation bus can authenticate
    /// it; none of its values are reconstructed from the legacy digest.
    pub fixed_setup_opening: FixedSetupOpeningCertificateMessageV2<T>,
    /// Exact C2 certificate for the same committed source message.
    pub active_count: VerifierWarpCertifiedActiveCountMessageV2<T>,
}

#[derive(Clone, Debug)]
pub struct VerifierWarpProducerBridgeAirV2 {
    pub fixed_public_values: VerifierWarpFixedPublicValuesProfileV2,
    /// Absolute block transition represented by local proof slot zero.
    pub segment_start: u32,
    pub protocol_digest: Digest,
    /// Canonical relation/index digest admitted by the fixed aggregation VK.
    pub admitted_relation_digest: Digest,
    /// Canonical VACC protocol/index key admitted by the fixed aggregation VK.
    pub expected_key_digest: Digest,
    pub source_relation_vk_digest: Digest,
    pub source_log_message_len: usize,
    pub source_log_codeword_len: usize,
    /// Digest of the setup-owned C2 profile admitted by this History VK.
    pub active_count_profile_digest: Digest,
    pub logup_bus: LogUpOnlyHistoryBusV19,
    pub fixed_source_bus: CertifiedFixedMultiAirSourceBusV2,
    pub vacc_input_bus: CertifiedDirectAirVaccInputBusV19,
    pub fresh_explicit_bus: CertifiedFreshExplicitDigestBusV19,
    pub replay_bus: CertifiedWarpReplayBusV19,
    pub fixed_public_values_bus: FixedSourcePublicValueBusV19,
    /// Present only for a nonempty setup-matrix profile. In canonical
    /// no-setup mode the certificate columns are constrained to one exact
    /// empty value and no compatibility bus exists.
    pub fixed_setup_opening_bus: Option<FixedSetupOpeningCertificateBusV2>,
    pub active_count_bus: VerifierWarpCertifiedActiveCountBusV2,
    pub source_bus: VerifierWarpSourceCertificateBusV2,
    pub vacc_bus: VerifierWarpVaccCertificateBusV2,
}

impl BaseAir<F> for VerifierWarpProducerBridgeAirV2 {
    fn width(&self) -> usize {
        core::mem::size_of::<VerifierWarpProducerBridgeColsV2<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.fixed_public_values
            .validate()
            .expect("invalid verifier-WARP fixed public-value profile in VK");
        let width = VerifierWarpProducerBridgePrepColsV2::<u8>::width();
        let height = self
            .fixed_public_values
            .batch_count
            .next_power_of_two()
            .max(2);
        let mut values = F::zero_vec(width * height);
        for index in 0..self.fixed_public_values.batch_count {
            let global_index = self
                .segment_start
                .checked_add(u32::try_from(index).expect("verifier-WARP bridge proof index"))
                .expect("verifier-WARP bridge global index");
            let row: &mut VerifierWarpProducerBridgePrepColsV2<F> =
                values[index * width..(index + 1) * width].borrow_mut();
            row.active = F::ONE;
            row.proof_index = F::from_usize(index);
            row.batch_index_lo = F::from_u16(global_index as u16);
            row.batch_index_hi = F::from_u16((global_index >> 16) as u16);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for VerifierWarpProducerBridgeAirV2 {}
impl PartitionedBaseAir<F> for VerifierWarpProducerBridgeAirV2 {}

impl<AB> Air<AB> for VerifierWarpProducerBridgeAirV2
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(
            self.fixed_public_values.validate().is_ok(),
            "invalid fixed public-value profile"
        );
        let prep_matrix = builder.preprocessed();
        let prep_row = prep_matrix
            .row_slice(0)
            .expect("verifier-WARP bridge prep row");
        let prep: &VerifierWarpProducerBridgePrepColsV2<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("verifier-WARP producer bridge row");
        let local: &VerifierWarpProducerBridgeColsV2<AB::Var> = (*row).borrow();
        let enabled = prep.active;
        builder.assert_bool(enabled);
        builder.assert_bool(local.active);
        builder.assert_eq(local.active, enabled);

        let batch_index = AB::Expr::from(prep.batch_index_lo)
            + AB::Expr::from_u32(1 << 16) * AB::Expr::from(prep.batch_index_hi);
        for protocol_version in [
            local.logup.protocol_version,
            local.fresh_input.protocol_version,
            local.replay.protocol_version,
        ] {
            builder.when(enabled).assert_eq(
                protocol_version,
                AB::Expr::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            );
        }
        builder
            .when(enabled)
            .assert_eq(local.fixed_source.proof_index, prep.proof_index);
        builder
            .when(enabled)
            .assert_eq(local.fixed_source.segment_index_lo, prep.batch_index_lo);
        builder
            .when(enabled)
            .assert_eq(local.fixed_source.segment_index_hi, prep.batch_index_hi);
        builder.when(enabled).assert_eq(
            local.fixed_source.active_child_count,
            local.active_count.expected_active_child_count,
        );
        for (lo, hi) in [
            (local.logup.segment_index_lo, local.logup.segment_index_hi),
            (
                local.fresh_input.segment_index_lo,
                local.fresh_input.segment_index_hi,
            ),
            (
                local.fresh_input.update_index_lo,
                local.fresh_input.update_index_hi,
            ),
            (local.replay.segment_index_lo, local.replay.segment_index_hi),
            (local.replay.update_index_lo, local.replay.update_index_hi),
        ] {
            builder.when(enabled).assert_eq(lo, prep.batch_index_lo);
            builder.when(enabled).assert_eq(hi, prep.batch_index_hi);
        }
        builder
            .when(enabled)
            .assert_eq(local.fresh_input.proof_index, batch_index.clone());
        builder
            .when(enabled)
            .assert_eq(local.fresh_explicit.proof_index, batch_index.clone());
        builder
            .when(enabled)
            .assert_zero(local.fresh_input.shard_ordinal);
        builder
            .when(enabled)
            .assert_zero(local.replay.shard_ordinal);
        builder.when(enabled).assert_eq(
            local.fresh_input.beta_len,
            AB::Expr::from_usize(self.fixed_public_values.beta_len()),
        );

        // The V3 retained setup PCS path is the sole authority for setup-fixed
        // SWIRL openings. `segment_openings_digest` remains a legacy SWIRL/VACC
        // binding and is never reinterpreted as this compatibility digest.
        builder.when(enabled).assert_eq(
            local.fixed_setup_opening.protocol_version,
            AB::Expr::from_u32(FIXED_SETUP_OPENING_PROTOCOL_V2),
        );
        builder
            .when(enabled)
            .assert_eq(local.fixed_setup_opening.proof_index, prep.proof_index);
        if self.fixed_setup_opening_bus.is_none() {
            builder
                .when(enabled)
                .assert_zero(local.fixed_setup_opening.canonical_claim_count);
            for limb in 0..DIGEST_SIZE {
                builder
                    .when(enabled)
                    .assert_zero(local.fixed_setup_opening.setup_openings_digest[limb]);
            }
        }

        // C2 is the sole authority for occupancy.  Bind every identity field
        // that selects the same-message source reduction.  The complete
        // opening metadata remains present in the typed lookup key below, so
        // point/value/cursor substitutions cannot be projected away by the
        // bridge.
        builder
            .when(enabled)
            .assert_eq(local.active_count.proof_index, prep.proof_index);
        for (limb, expected) in local.active_count.batch_index.iter().zip([
            AB::Expr::from(prep.batch_index_lo),
            AB::Expr::from(prep.batch_index_hi),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ]) {
            builder.when(enabled).assert_eq(*limb, expected);
        }

        for limb in 0..DIGEST_SIZE {
            builder.when(enabled).assert_eq(
                local.logup.app_vk_digest[limb],
                AB::Expr::from(self.source_relation_vk_digest[limb]),
            );
            builder.when(enabled).assert_eq(
                local.fixed_source.app_vk_digest[limb],
                AB::Expr::from(self.source_relation_vk_digest[limb]),
            );
            builder.when(enabled).assert_eq(
                local.fixed_source.relation_digest[limb],
                AB::Expr::from(self.admitted_relation_digest[limb]),
            );
            builder.when(enabled).assert_eq(
                local.fixed_setup_opening.source_relation_vk_digest[limb],
                AB::Expr::from(self.source_relation_vk_digest[limb]),
            );
            builder.when(enabled).assert_eq(
                local.replay.key_digest[limb],
                AB::Expr::from(self.expected_key_digest[limb]),
            );
            for digest in [
                local.fresh_input.relation_digest[limb],
                local.fresh_explicit.relation_digest[limb],
                local.replay.relation_digest[limb],
                local.active_count.relation_digest[limb],
            ] {
                builder
                    .when(enabled)
                    .assert_eq(digest, AB::Expr::from(self.admitted_relation_digest[limb]));
            }
            builder
                .when(enabled)
                .assert_eq(local.fresh_input.root[limb], local.replay.fresh_root[limb]);
            builder.when(enabled).assert_eq(
                local.fixed_source.source_root[limb],
                local.fresh_input.root[limb],
            );
            builder.when(enabled).assert_eq(
                local.fixed_source.source_forest_root[limb],
                local.logup.source_forest_root[limb],
            );
            builder.when(enabled).assert_eq(
                local.fixed_source.segment_openings_digest[limb],
                local.logup.segment_openings_digest[limb],
            );
            builder.when(enabled).assert_eq(
                local.active_count.source_root[limb],
                local.fresh_input.root[limb],
            );
            builder.when(enabled).assert_eq(
                local.active_count.profile_digest[limb],
                AB::Expr::from(self.active_count_profile_digest[limb]),
            );
            builder.when(enabled).assert_eq(
                local.logup.source_forest_root[limb],
                local.replay.source_forest_root[limb],
            );
            builder.when(enabled).assert_eq(
                local.logup.segment_openings_digest[limb],
                local.replay.segment_openings_digest[limb],
            );
        }
        assert!(
            self.source_log_message_len > 0
                && self.source_log_codeword_len >= self.source_log_message_len
                && self.source_log_codeword_len <= MAX_RAW_MESSAGE_POINT_LEN_V19
        );
        builder.when(enabled).assert_eq(
            local.fixed_source.point_len,
            AB::Expr::from_usize(self.source_log_message_len),
        );
        builder.when(enabled).assert_eq(
            local.fresh_input.alpha_len,
            AB::Expr::from_usize(self.source_log_codeword_len),
        );
        // `alpha_len` is the codeword point length; the source point occupies
        // its setup-fixed prefix and the RS blowup suffix is canonical zero.
        for index in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
            for limb in 0..D_EF {
                if index < self.source_log_message_len {
                    builder.when(enabled).assert_eq(
                        local.fixed_source.point[index][limb],
                        local.fresh_input.alpha[index][limb],
                    );
                } else {
                    builder
                        .when(enabled)
                        .assert_zero(local.fixed_source.point[index][limb]);
                    if index < self.source_log_codeword_len {
                        builder
                            .when(enabled)
                            .assert_zero(local.fresh_input.alpha[index][limb]);
                    }
                }
            }
        }
        for limb in 0..D_EF {
            builder
                .when(enabled)
                .assert_eq(local.fixed_source.value[limb], local.fresh_input.mu[limb]);
            builder.when(enabled).assert_eq(
                local.fixed_source.verifier_endpoint[limb],
                local.logup.verifier_endpoint[limb],
            );
        }

        // Explicit beta is a base-field vector. Coordinate zero is the fixed
        // constant one; the aggregate VmPvs range is setup-derived.
        let explicit_start = self.fixed_public_values.log_constraints;
        for coordinate in 0..self.fixed_public_values.explicit_len() {
            let beta_coordinate = explicit_start + coordinate;
            for limb in 1..D_EF {
                builder
                    .when(enabled)
                    .assert_zero(local.fresh_input.beta[beta_coordinate][limb]);
            }
            let value: AB::Expr = match self.fixed_public_values.sources[coordinate] {
                VerifierWarpFixedPublicValueSourceV2::TrustedConstant(value) => {
                    builder.when(enabled).assert_eq(
                        local.fresh_input.beta[beta_coordinate][0],
                        AB::Expr::from(value),
                    );
                    value.into()
                }
                VerifierWarpFixedPublicValueSourceV2::VmPvsCoordinate(offset) => {
                    let value = local.vm_pvs.as_slice()[offset];
                    builder
                        .when(enabled)
                        .assert_eq(local.fresh_input.beta[beta_coordinate][0], value);
                    value.into()
                }
            };
            // The fixed-source instance digest contains the concatenated AIR
            // public values, while PESAT explicit coordinate zero is the
            // distinguished one. Constrain coordinate zero above, but emit
            // only coordinates 1.. using their canonical source-vector index.
            if coordinate > 0 {
                self.fixed_public_values_bus.send(
                    builder,
                    FixedSourcePublicValueMessageV19 {
                        proof_index: prep.proof_index.into(),
                        segment_index_lo: local.fresh_input.segment_index_lo.into(),
                        segment_index_hi: local.fresh_input.segment_index_hi.into(),
                        public_value_index: AB::Expr::from_usize(coordinate - 1),
                        value,
                    },
                    enabled,
                );
            }
        }

        // Independent positive statements from the actual producer AIRs.
        self.logup_bus
            .lookup_key(builder, local.logup.clone(), enabled);
        self.fixed_source_bus
            .receive(builder, local.fixed_source.clone(), enabled);
        self.vacc_input_bus
            .lookup_key(builder, local.fresh_input.clone(), enabled);
        self.fresh_explicit_bus
            .lookup_key(builder, local.fresh_explicit.clone(), enabled);
        self.replay_bus
            .lookup_key(builder, local.replay.clone(), enabled);
        for (lo, hi) in [(
            local.fresh_explicit.segment_index_lo,
            local.fresh_explicit.segment_index_hi,
        )] {
            builder.when(enabled).assert_eq(lo, prep.batch_index_lo);
            builder.when(enabled).assert_eq(hi, prep.batch_index_hi);
        }
        if let Some(bus) = self.fixed_setup_opening_bus {
            bus.receive(builder, local.fixed_setup_opening.clone(), enabled);
        }
        self.active_count_bus
            .lookup_key(builder, local.active_count.clone(), enabled);
        let zero_digest = core::array::from_fn(|_| AB::Expr::ZERO);
        let children = core::array::from_fn(|slot| {
            if slot == 0 {
                VerifierWarpCertifiedChildMessageV2 {
                    occupied: AB::Expr::ONE,
                    input_pc: local.vm_pvs.initial_pc.into(),
                    input_memory_root: local.vm_pvs.initial_root.map(Into::into),
                    output_pc: local.vm_pvs.final_pc.into(),
                    output_memory_root: local.vm_pvs.final_root.map(Into::into),
                    exit_code: local.vm_pvs.exit_code.into(),
                    terminates: local.vm_pvs.is_terminate.into(),
                }
            } else {
                VerifierWarpCertifiedChildMessageV2 {
                    occupied: AB::Expr::ZERO,
                    input_pc: AB::Expr::ZERO,
                    input_memory_root: zero_digest.clone(),
                    output_pc: AB::Expr::ZERO,
                    output_memory_root: zero_digest.clone(),
                    exit_code: AB::Expr::ZERO,
                    terminates: AB::Expr::ZERO,
                }
            }
        });
        self.source_bus.add_key_with_lookups(
            builder,
            VerifierWarpSourceCertificateMessageV2 {
                protocol_version: AB::Expr::from_u32(VERIFIER_WARP_HISTORY_PROTOCOL_V2),
                source_child_capacity: AB::Expr::from_usize(VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2),
                batch_index_lo: prep.batch_index_lo.into(),
                batch_index_hi: prep.batch_index_hi.into(),
                active_child_count: local.active_count.expected_active_child_count.into(),
                protocol_digest: self.protocol_digest.map(Into::into),
                relation_digest: self.admitted_relation_digest.map(Into::into),
                program_commitment: local.vm_pvs.program_commit.map(Into::into),
                children,
                source_accumulator_digest: local.replay.fresh_instance_digest.map(Into::into),
                // This is the root of the exact systematic WARP source
                // message checked by C2, not the separate SWIRL forest root.
                source_commitment_root: local.active_count.source_root.map(Into::into),
                external_logup_gkr_digest: local.logup.checkpoint_digest.map(Into::into),
                source_functional_digest: local.replay.opening_claim_digest.map(Into::into),
                setup_openings_digest: local
                    .fixed_setup_opening
                    .setup_openings_digest
                    .map(Into::into),
                source_statement_digest: local.replay.replay_endpoint_digest.map(Into::into),
            },
            enabled,
        );
        self.vacc_bus.add_key_with_lookups(
            builder,
            VerifierWarpVaccCertificateMessageV2 {
                protocol_version: AB::Expr::from_u32(VERIFIER_WARP_HISTORY_PROTOCOL_V2),
                input_arity: AB::Expr::from_usize(VERIFIER_WARP_VACC_INPUT_ARITY_V2),
                batch_index_lo: prep.batch_index_lo.into(),
                batch_index_hi: prep.batch_index_hi.into(),
                relation_digest: self.admitted_relation_digest.map(Into::into),
                prior_accumulator_digest: local.replay.previous_accumulator_digest.map(Into::into),
                source_accumulator_digest: local.replay.fresh_instance_digest.map(Into::into),
                output_accumulator_digest: local.replay.next_accumulator_digest.map(Into::into),
                source_commitment_root: local.active_count.source_root.map(Into::into),
                transition_transcript_digest: local.replay.replay_binding_digest.map(Into::into),
            },
            enabled,
        );
    }
}

#[derive(Clone)]
pub struct VerifierWarpProducerBridgeRecordV2 {
    pub vm_pvs: VmPvs<F>,
    pub logup: LogUpOnlyHistoryMessageV19<F>,
    pub fixed_source: CertifiedFixedMultiAirSourceMessageV2<F>,
    pub fresh_input: CertifiedDirectAirVaccInputMessageV19<F>,
    pub fresh_explicit: CertifiedFreshExplicitDigestMessageV19<F>,
    pub replay: CertifiedWarpReplayMessageV19<F>,
    pub fixed_setup_opening: FixedSetupOpeningCertificateMessageV2<F>,
    pub active_count: VerifierWarpCertifiedActiveCountMessageV2<F>,
}

/// Trace-generation output for the producer bridge.
///
/// The certificates are derived from the populated bridge rows and
/// setup-owned AIR constants.  They are therefore outputs of trace
/// generation, not caller-provided authority objects.  All three collections
/// have exactly the setup-fixed `batch_count` length (the matrix may include
/// additional zero padding rows).
pub struct VerifierWarpProducerBridgeTraceResultV2 {
    pub matrix: RowMajorMatrix<F>,
    pub source_certificates: Box<[VerifierWarpSourceCertificateMessageV2<F>]>,
    pub vacc_certificates: Box<[VerifierWarpVaccCertificateMessageV2<F>]>,
}

fn source_certificate_from_bridge_row_v2(
    air: &VerifierWarpProducerBridgeAirV2,
    batch_index: u32,
    local: &VerifierWarpProducerBridgeColsV2<F>,
) -> VerifierWarpSourceCertificateMessageV2<F> {
    let children = core::array::from_fn(|slot| {
        if slot == 0 {
            VerifierWarpCertifiedChildMessageV2 {
                occupied: F::ONE,
                input_pc: local.vm_pvs.initial_pc,
                input_memory_root: local.vm_pvs.initial_root,
                output_pc: local.vm_pvs.final_pc,
                output_memory_root: local.vm_pvs.final_root,
                exit_code: local.vm_pvs.exit_code,
                terminates: local.vm_pvs.is_terminate,
            }
        } else {
            VerifierWarpCertifiedChildMessageV2 {
                occupied: F::ZERO,
                input_pc: F::ZERO,
                input_memory_root: [F::ZERO; DIGEST_SIZE],
                output_pc: F::ZERO,
                output_memory_root: [F::ZERO; DIGEST_SIZE],
                exit_code: F::ZERO,
                terminates: F::ZERO,
            }
        }
    });
    VerifierWarpSourceCertificateMessageV2 {
        protocol_version: F::from_u32(VERIFIER_WARP_HISTORY_PROTOCOL_V2),
        source_child_capacity: F::from_usize(VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2),
        batch_index_lo: F::from_u16(batch_index as u16),
        batch_index_hi: F::from_u16((batch_index >> 16) as u16),
        active_child_count: local.active_count.expected_active_child_count,
        protocol_digest: air.protocol_digest,
        relation_digest: air.admitted_relation_digest,
        program_commitment: local.vm_pvs.program_commit,
        children,
        source_accumulator_digest: local.replay.fresh_instance_digest,
        source_commitment_root: local.active_count.source_root,
        external_logup_gkr_digest: local.logup.checkpoint_digest,
        source_functional_digest: local.replay.opening_claim_digest,
        setup_openings_digest: local.fixed_setup_opening.setup_openings_digest,
        source_statement_digest: local.replay.replay_endpoint_digest,
    }
}

fn vacc_certificate_from_bridge_row_v2(
    air: &VerifierWarpProducerBridgeAirV2,
    batch_index: u32,
    local: &VerifierWarpProducerBridgeColsV2<F>,
) -> VerifierWarpVaccCertificateMessageV2<F> {
    VerifierWarpVaccCertificateMessageV2 {
        protocol_version: F::from_u32(VERIFIER_WARP_HISTORY_PROTOCOL_V2),
        input_arity: F::from_usize(VERIFIER_WARP_VACC_INPUT_ARITY_V2),
        batch_index_lo: F::from_u16(batch_index as u16),
        batch_index_hi: F::from_u16((batch_index >> 16) as u16),
        relation_digest: air.admitted_relation_digest,
        prior_accumulator_digest: local.replay.previous_accumulator_digest,
        source_accumulator_digest: local.replay.fresh_instance_digest,
        output_accumulator_digest: local.replay.next_accumulator_digest,
        source_commitment_root: local.active_count.source_root,
        transition_transcript_digest: local.replay.replay_binding_digest,
    }
}

/// Generates the producer-bridge trace and its exact source/VACC certificate
/// messages.
///
/// The setup-fixed record count is checked before allocation, and all size
/// arithmetic is checked.  Certificate formulas deliberately mirror the AIR
/// sends above; local differential tests compare them through the typed buses
/// so any future drift fails closed.
pub fn generate_verifier_warp_producer_bridge_trace_and_certificates_v2(
    air: &VerifierWarpProducerBridgeAirV2,
    records: &[VerifierWarpProducerBridgeRecordV2],
) -> Result<VerifierWarpProducerBridgeTraceResultV2, &'static str> {
    air.fixed_public_values.validate()?;
    if records.len() != air.fixed_public_values.batch_count {
        return Err("verifier-WARP bridge record/profile length");
    }
    let _batch_count =
        u32::try_from(records.len()).map_err(|_| "verifier-WARP bridge batch count exceeds u32")?;
    let width = core::mem::size_of::<VerifierWarpProducerBridgeColsV2<u8>>();
    let height = records
        .len()
        .checked_next_power_of_two()
        .ok_or("verifier-WARP bridge trace height overflow")?
        .max(2);
    let value_count = width
        .checked_mul(height)
        .ok_or("verifier-WARP bridge trace allocation overflow")?;
    let mut values = F::zero_vec(value_count);
    let mut source_certificates = Vec::with_capacity(records.len());
    let mut vacc_certificates = Vec::with_capacity(records.len());
    for (row_index, record) in records.iter().enumerate() {
        let local_index =
            u32::try_from(row_index).map_err(|_| "verifier-WARP bridge proof index exceeds u32")?;
        let batch_index = air
            .segment_start
            .checked_add(local_index)
            .ok_or("verifier-WARP bridge batch index overflow")?;
        let row_start = row_index
            .checked_mul(width)
            .ok_or("verifier-WARP bridge row offset overflow")?;
        let row_end = row_start
            .checked_add(width)
            .ok_or("verifier-WARP bridge row end overflow")?;
        let row = values
            .get_mut(row_start..row_end)
            .ok_or("verifier-WARP bridge row outside trace")?;
        let cols: &mut VerifierWarpProducerBridgeColsV2<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.vm_pvs = record.vm_pvs;
        cols.logup = record.logup.clone();
        cols.fixed_source = record.fixed_source.clone();
        cols.fresh_input = record.fresh_input.clone();
        cols.fresh_explicit = record.fresh_explicit.clone();
        cols.replay = record.replay.clone();
        cols.fixed_setup_opening = record.fixed_setup_opening.clone();
        cols.active_count = record.active_count.clone();
        source_certificates.push(source_certificate_from_bridge_row_v2(
            air,
            batch_index,
            cols,
        ));
        vacc_certificates.push(vacc_certificate_from_bridge_row_v2(air, batch_index, cols));
    }
    Ok(VerifierWarpProducerBridgeTraceResultV2 {
        matrix: RowMajorMatrix::new(values, width),
        source_certificates: source_certificates.into_boxed_slice(),
        vacc_certificates: vacc_certificates.into_boxed_slice(),
    })
}

/// Compatibility wrapper returning only the bridge matrix.
pub fn generate_verifier_warp_producer_bridge_trace_v2(
    air: &VerifierWarpProducerBridgeAirV2,
    records: &[VerifierWarpProducerBridgeRecordV2],
) -> Result<RowMajorMatrix<F>, &'static str> {
    Ok(generate_verifier_warp_producer_bridge_trace_and_certificates_v2(air, records)?.matrix)
}

#[cfg(test)]
mod tests {
    use std::{panic::AssertUnwindSafe, sync::Arc};

    use openvm_recursion_circuit::system::BusIndexManager;
    use openvm_stark_backend::{
        air_builders::{
            debug::{check_constraints, check_logup},
            symbolic::get_symbolic_builder,
        },
        interaction::{BusIndex, SymbolicInteraction},
        keygen::types::TraceWidth,
        AirRef, AnyAir,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::{
        BabyBearPoseidon2Config as NativeSC, D_EF,
    };

    use super::*;
    use crate::circuit::native_warp_history_v19::{
        CertifiedDirectAirVaccInputBusV19, CertifiedFreshExplicitDigestBusV19,
        CertifiedWarpReplayBusV19, FixedSourcePublicValueBusV19, LogUpOnlyHistoryBusV19,
        MAX_FRESH_BETA_LEN_V19, MAX_RAW_MESSAGE_POINT_LEN_V19,
    };

    fn digest(seed: u32) -> Digest {
        core::array::from_fn(|limb| F::from_u32(seed + limb as u32))
    }

    #[test]
    fn complete_capacity_four_verifier_beta_fits_history_inventory() {
        // Current production relation: one distinguished coordinate, 66
        // VerifierBasePvs coordinates, eight DAG-commit coordinates, and 28
        // aggregate VmPvs coordinates.  Only the latter are dynamic here;
        // setup reconstructs every other explicit value.
        let mut sources = vec![VerifierWarpFixedPublicValueSourceV2::TrustedConstant(F::ZERO); 103];
        sources[0] = VerifierWarpFixedPublicValueSourceV2::TrustedConstant(F::ONE);
        for (offset, source) in sources[75..103].iter_mut().enumerate() {
            *source = VerifierWarpFixedPublicValueSourceV2::VmPvsCoordinate(offset);
        }
        let profile = VerifierWarpFixedPublicValuesProfileV2 {
            log_constraints: 29,
            sources: sources.into(),
            batch_count: 128,
        };
        assert_eq!(profile.beta_len(), 132);
        assert!(profile.beta_len() <= MAX_FRESH_BETA_LEN_V19);
        assert_eq!(profile.validate(), Ok(()));
    }

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct CertificateSinkCols<T> {
        active: T,
        source: VerifierWarpSourceCertificateMessageV2<T>,
        vacc: VerifierWarpVaccCertificateMessageV2<T>,
    }

    #[derive(Clone)]
    struct CertificateSinkAir {
        source_bus: VerifierWarpSourceCertificateBusV2,
        vacc_bus: VerifierWarpVaccCertificateBusV2,
    }

    impl BaseAir<F> for CertificateSinkAir {
        fn width(&self) -> usize {
            core::mem::size_of::<CertificateSinkCols<u8>>()
        }
    }

    impl BaseAirWithPublicValues<F> for CertificateSinkAir {}
    impl PartitionedBaseAir<F> for CertificateSinkAir {}

    impl<AB> Air<AB> for CertificateSinkAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("certificate sink row");
            let local: &CertificateSinkCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.active);
            self.source_bus
                .lookup_key(builder, local.source.clone(), local.active);
            self.vacc_bus
                .lookup_key(builder, local.vacc.clone(), local.active);
        }
    }

    fn certificate_sink_trace(
        source: VerifierWarpSourceCertificateMessageV2<F>,
        vacc: VerifierWarpVaccCertificateMessageV2<F>,
    ) -> RowMajorMatrix<F> {
        let width = core::mem::size_of::<CertificateSinkCols<u8>>();
        let mut values = F::zero_vec(2 * width);
        let local: &mut CertificateSinkCols<F> = values[..width].borrow_mut();
        local.active = F::ONE;
        local.source = source;
        local.vacc = vacc;
        RowMajorMatrix::new(values, width)
    }

    fn symbolic_interactions(air: &dyn AnyAir<NativeSC>) -> Vec<SymbolicInteraction<F>> {
        let preprocessed = BaseAir::<F>::preprocessed_trace(air).map(|trace| trace.width());
        get_symbolic_builder(
            air,
            &TraceWidth {
                preprocessed,
                cached_mains: air.cached_main_widths(),
                common_main: air.common_main_width(),
            },
        )
        .constraints()
        .interactions
    }

    fn check_air_constraints(air: &dyn AnyAir<NativeSC>, trace: &RowMajorMatrix<F>) {
        let preprocessed = BaseAir::<F>::preprocessed_trace(air);
        check_constraints::<_, NativeSC>(
            air,
            &air.name(),
            &preprocessed.as_ref().map(RowMajorMatrix::as_view),
            &[trace.as_view()],
            &[],
        );
    }

    fn check_certificate_bus_balance(
        air: &VerifierWarpProducerBridgeAirV2,
        bridge_trace: &RowMajorMatrix<F>,
        source: VerifierWarpSourceCertificateMessageV2<F>,
        vacc: VerifierWarpVaccCertificateMessageV2<F>,
        source_bus_index: BusIndex,
        vacc_bus_index: BusIndex,
    ) {
        let sink_air = CertificateSinkAir {
            source_bus: air.source_bus,
            vacc_bus: air.vacc_bus,
        };
        let sink_trace = certificate_sink_trace(source, vacc);
        let airs: Vec<AirRef<NativeSC>> = vec![Arc::new(air.clone()), Arc::new(sink_air)];
        let matrices = [bridge_trace, &sink_trace];
        for (air, matrix) in airs.iter().zip(matrices) {
            check_air_constraints(air.as_ref(), matrix);
        }
        let preprocessed_owned = airs
            .iter()
            .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
            .collect::<Vec<_>>();
        let preprocessed = preprocessed_owned
            .iter()
            .map(|matrix| matrix.as_ref().map(RowMajorMatrix::as_view))
            .collect::<Vec<_>>();
        let selected = [source_bus_index, vacc_bus_index];
        let interactions = airs
            .iter()
            .map(|air| {
                symbolic_interactions(air.as_ref())
                    .into_iter()
                    .filter(|interaction| selected.contains(&interaction.bus_index))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let views = vec![vec![bridge_trace.as_view()], vec![sink_trace.as_view()]];
        let public_values = vec![Vec::new(), Vec::new()];
        check_logup(
            &airs.iter().map(|air| air.name()).collect::<Vec<_>>(),
            &interactions,
            &preprocessed,
            &views,
            &public_values,
        );
    }

    fn fixture() -> (
        VerifierWarpProducerBridgeAirV2,
        VerifierWarpProducerBridgeRecordV2,
        BusIndex,
        BusIndex,
    ) {
        let mut manager = BusIndexManager::new();
        let source_bus_index = manager.new_bus_idx();
        let vacc_bus_index = manager.new_bus_idx();
        let relation_digest = digest(100);
        let key_digest = digest(120);
        let source_relation_vk_digest = digest(140);
        let source_root = digest(160);
        let source_forest_root = digest(180);
        let segment_openings_digest = digest(200);
        let active_count_profile_digest = digest(220);
        let protocol_digest = digest(240);
        let vm_pvs = VmPvs {
            program_commit: digest(260),
            initial_pc: F::from_u32(17),
            final_pc: F::from_u32(29),
            exit_code: F::ZERO,
            is_terminate: F::ONE,
            initial_root: digest(280),
            final_root: digest(300),
        };
        let fixed_public_values = VerifierWarpFixedPublicValuesProfileV2 {
            log_constraints: 1,
            sources: core::iter::once(VerifierWarpFixedPublicValueSourceV2::TrustedConstant(
                F::ONE,
            ))
            .chain(
                (0..VmPvs::<u8>::width())
                    .map(VerifierWarpFixedPublicValueSourceV2::VmPvsCoordinate),
            )
            .collect::<Vec<_>>()
            .into(),
            batch_count: 1,
        };
        let air = VerifierWarpProducerBridgeAirV2 {
            fixed_public_values: fixed_public_values.clone(),
            segment_start: 0,
            protocol_digest,
            admitted_relation_digest: relation_digest,
            expected_key_digest: key_digest,
            source_relation_vk_digest,
            source_log_message_len: 1,
            source_log_codeword_len: 1,
            active_count_profile_digest,
            logup_bus: LogUpOnlyHistoryBusV19::new(manager.new_bus_idx()),
            fixed_source_bus: CertifiedFixedMultiAirSourceBusV2::new(manager.new_bus_idx()),
            vacc_input_bus: CertifiedDirectAirVaccInputBusV19::new(manager.new_bus_idx()),
            fresh_explicit_bus: CertifiedFreshExplicitDigestBusV19::new(manager.new_bus_idx()),
            replay_bus: CertifiedWarpReplayBusV19::new(manager.new_bus_idx()),
            fixed_public_values_bus: FixedSourcePublicValueBusV19::new(manager.new_bus_idx()),
            fixed_setup_opening_bus: None,
            active_count_bus: VerifierWarpCertifiedActiveCountBusV2::new(manager.new_bus_idx()),
            source_bus: VerifierWarpSourceCertificateBusV2::new(source_bus_index),
            vacc_bus: VerifierWarpVaccCertificateBusV2::new(vacc_bus_index),
        };
        let mut point = [[F::ZERO; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19];
        point[0][0] = F::from_u32(320);
        let mut beta = [[F::ZERO; D_EF]; MAX_FRESH_BETA_LEN_V19];
        beta[1][0] = F::ONE;
        for (offset, &value) in vm_pvs.as_slice().iter().enumerate() {
            beta[2 + offset][0] = value;
        }
        let active_count = VerifierWarpCertifiedActiveCountMessageV2 {
            proof_index: F::ZERO,
            batch_index: [F::ZERO; 4],
            relation_digest,
            profile_digest: active_count_profile_digest,
            source_root,
            expected_active_child_count: F::ONE,
            batching_coefficient: [F::ZERO; D_EF],
            point_len: F::ONE,
            point,
            message_value: [F::ZERO; D_EF],
            reduction_end_tidx: F::from_u32(340),
        };
        let record = VerifierWarpProducerBridgeRecordV2 {
            vm_pvs,
            logup: LogUpOnlyHistoryMessageV19 {
                protocol_version: F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                mode_tag: F::ONE,
                segment_index_lo: F::ZERO,
                segment_index_hi: F::ZERO,
                app_vk_digest: source_relation_vk_digest,
                source_forest_root,
                segment_openings_digest,
                verifier_endpoint: [F::ZERO; D_EF],
                checkpoint_digest: digest(360),
            },
            fixed_source: CertifiedFixedMultiAirSourceMessageV2 {
                proof_index: F::ZERO,
                segment_index_lo: F::ZERO,
                segment_index_hi: F::ZERO,
                active_child_count: F::ONE,
                app_vk_digest: source_relation_vk_digest,
                relation_digest,
                source_forest_root,
                segment_openings_digest,
                source_root,
                point_len: F::ONE,
                point,
                value: [F::ZERO; D_EF],
                verifier_endpoint: [F::ZERO; D_EF],
            },
            fresh_input: CertifiedDirectAirVaccInputMessageV19 {
                proof_index: F::ZERO,
                protocol_version: F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                segment_index_lo: F::ZERO,
                segment_index_hi: F::ZERO,
                update_index_lo: F::ZERO,
                update_index_hi: F::ZERO,
                shard_ordinal: F::ZERO,
                relation_digest,
                root: source_root,
                alpha_len: F::ONE,
                alpha: point,
                mu: [F::ZERO; D_EF],
                beta_len: F::from_usize(fixed_public_values.beta_len()),
                beta,
                eta: [F::ZERO; D_EF],
            },
            fresh_explicit: CertifiedFreshExplicitDigestMessageV19 {
                proof_index: F::ZERO,
                segment_index_lo: F::ZERO,
                segment_index_hi: F::ZERO,
                relation_digest,
                digest: digest(380),
            },
            replay: CertifiedWarpReplayMessageV19 {
                protocol_version: F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
                segment_index_lo: F::ZERO,
                segment_index_hi: F::ZERO,
                update_index_lo: F::ZERO,
                update_index_hi: F::ZERO,
                shard_ordinal: F::ZERO,
                source_forest_root,
                key_digest,
                relation_digest,
                opening_claim_digest: digest(400),
                fresh_instance_digest: digest(420),
                segment_openings_digest,
                prior_root: digest(440),
                fresh_root: source_root,
                next_root: digest(460),
                previous_accumulator_digest: digest(480),
                next_accumulator_digest: digest(500),
                authenticated_batching_claim: [F::ZERO; D_EF],
                previous_checkpoint_digest: digest(520),
                next_checkpoint_digest: digest(540),
                replay_endpoint_digest: digest(560),
                replay_binding_digest: digest(580),
            },
            fixed_setup_opening: FixedSetupOpeningCertificateMessageV2 {
                protocol_version: F::from_u32(FIXED_SETUP_OPENING_PROTOCOL_V2),
                proof_index: F::ZERO,
                canonical_claim_count: F::ZERO,
                source_relation_vk_digest,
                setup_openings_digest: [F::ZERO; DIGEST_SIZE],
            },
            active_count,
        };
        (air, record, source_bus_index, vacc_bus_index)
    }

    #[test]
    fn returned_certificates_match_exact_air_bus_messages_and_compatibility_trace() {
        let (air, record, source_bus_index, vacc_bus_index) = fixture();
        let result = generate_verifier_warp_producer_bridge_trace_and_certificates_v2(
            &air,
            core::slice::from_ref(&record),
        )
        .unwrap();
        assert_eq!(result.source_certificates.len(), 1);
        assert_eq!(result.vacc_certificates.len(), 1);
        assert_eq!(result.matrix.height(), 2);
        let local: &VerifierWarpProducerBridgeColsV2<F> =
            result.matrix.values[..result.matrix.width()].borrow();
        assert_eq!(local.vm_pvs.as_slice(), record.vm_pvs.as_slice());
        assert_eq!(local.replay, record.replay);
        assert_eq!(result.source_certificates[0].batch_index_lo, F::ZERO);
        assert_eq!(
            result.source_certificates[0].source_commitment_root,
            local.active_count.source_root
        );
        assert_eq!(
            result.vacc_certificates[0].transition_transcript_digest,
            local.replay.replay_binding_digest
        );
        check_certificate_bus_balance(
            &air,
            &result.matrix,
            result.source_certificates[0].clone(),
            result.vacc_certificates[0].clone(),
            source_bus_index,
            vacc_bus_index,
        );

        let compatibility = generate_verifier_warp_producer_bridge_trace_v2(&air, &[record])
            .expect("compatibility trace");
        assert_eq!(compatibility.width(), result.matrix.width());
        assert_eq!(compatibility.values, result.matrix.values);
    }

    #[test]
    fn nonzero_interval_keeps_local_authority_slots_and_absolute_transition_ids() {
        let (mut air, mut record, source_bus_index, vacc_bus_index) = fixture();
        let global_index = 70_003u32;
        air.segment_start = global_index;
        let lo = F::from_u16(global_index as u16);
        let hi = F::from_u16((global_index >> 16) as u16);
        for (segment_lo, segment_hi) in [
            (
                &mut record.logup.segment_index_lo,
                &mut record.logup.segment_index_hi,
            ),
            (
                &mut record.fixed_source.segment_index_lo,
                &mut record.fixed_source.segment_index_hi,
            ),
            (
                &mut record.fresh_input.segment_index_lo,
                &mut record.fresh_input.segment_index_hi,
            ),
            (
                &mut record.fresh_explicit.segment_index_lo,
                &mut record.fresh_explicit.segment_index_hi,
            ),
            (
                &mut record.replay.segment_index_lo,
                &mut record.replay.segment_index_hi,
            ),
        ] {
            *segment_lo = lo;
            *segment_hi = hi;
        }
        record.fresh_input.update_index_lo = lo;
        record.fresh_input.update_index_hi = hi;
        record.replay.update_index_lo = lo;
        record.replay.update_index_hi = hi;
        record.fresh_input.proof_index = F::from_u32(global_index);
        record.fresh_explicit.proof_index = F::from_u32(global_index);
        record.active_count.batch_index = [lo, hi, F::ZERO, F::ZERO];

        // These authority namespaces deliberately remain local to this leaf.
        assert_eq!(record.fixed_source.proof_index, F::ZERO);
        assert_eq!(record.fixed_setup_opening.proof_index, F::ZERO);
        assert_eq!(record.active_count.proof_index, F::ZERO);

        let result = generate_verifier_warp_producer_bridge_trace_and_certificates_v2(
            &air,
            core::slice::from_ref(&record),
        )
        .unwrap();
        check_air_constraints(&air, &result.matrix);
        assert_eq!(result.source_certificates[0].batch_index_lo, lo);
        assert_eq!(result.source_certificates[0].batch_index_hi, hi);
        assert_eq!(result.vacc_certificates[0].batch_index_lo, lo);
        assert_eq!(result.vacc_certificates[0].batch_index_hi, hi);
        check_certificate_bus_balance(
            &air,
            &result.matrix,
            result.source_certificates[0].clone(),
            result.vacc_certificates[0].clone(),
            source_bus_index,
            vacc_bus_index,
        );

        let mut wrong = record;
        wrong.fixed_source.proof_index = F::from_u32(global_index);
        let trace = generate_verifier_warp_producer_bridge_trace_and_certificates_v2(
            &air,
            core::slice::from_ref(&wrong),
        )
        .unwrap();
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
            check_air_constraints(&air, &trace.matrix)
        }))
        .is_err());
    }

    fn assert_mutated_certificate_rejects(
        source_case: Option<usize>,
        vacc_case: Option<usize>,
        label: &str,
    ) {
        let (air, record, source_bus_index, vacc_bus_index) = fixture();
        let result = generate_verifier_warp_producer_bridge_trace_and_certificates_v2(
            &air,
            core::slice::from_ref(&record),
        )
        .unwrap();
        let mut source = result.source_certificates[0].clone();
        let mut vacc = result.vacc_certificates[0].clone();
        if let Some(case) = source_case {
            match case {
                0 => source.protocol_version += F::ONE,
                1 => source.source_child_capacity += F::ONE,
                2 => source.batch_index_lo += F::ONE,
                3 => source.batch_index_hi += F::ONE,
                4 => source.active_child_count += F::ONE,
                5 => source.protocol_digest[0] += F::ONE,
                6 => source.relation_digest[0] += F::ONE,
                7 => source.program_commitment[0] += F::ONE,
                8 => source.children[0].occupied += F::ONE,
                9 => source.children[0].input_pc += F::ONE,
                10 => source.children[0].input_memory_root[0] += F::ONE,
                11 => source.children[0].output_pc += F::ONE,
                12 => source.children[0].output_memory_root[0] += F::ONE,
                13 => source.children[0].exit_code += F::ONE,
                14 => source.children[0].terminates += F::ONE,
                15 => source.children[1].occupied += F::ONE,
                16 => source.source_accumulator_digest[0] += F::ONE,
                17 => source.source_commitment_root[0] += F::ONE,
                18 => source.external_logup_gkr_digest[0] += F::ONE,
                19 => source.source_functional_digest[0] += F::ONE,
                20 => source.setup_openings_digest[0] += F::ONE,
                21 => source.source_statement_digest[0] += F::ONE,
                _ => unreachable!("bounded source mutation case"),
            }
        }
        if let Some(case) = vacc_case {
            match case {
                0 => vacc.protocol_version += F::ONE,
                1 => vacc.input_arity += F::ONE,
                2 => vacc.batch_index_lo += F::ONE,
                3 => vacc.batch_index_hi += F::ONE,
                4 => vacc.relation_digest[0] += F::ONE,
                5 => vacc.prior_accumulator_digest[0] += F::ONE,
                6 => vacc.source_accumulator_digest[0] += F::ONE,
                7 => vacc.output_accumulator_digest[0] += F::ONE,
                8 => vacc.source_commitment_root[0] += F::ONE,
                9 => vacc.transition_transcript_digest[0] += F::ONE,
                _ => unreachable!("bounded VACC mutation case"),
            }
        }
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                check_certificate_bus_balance(
                    &air,
                    &result.matrix,
                    source,
                    vacc,
                    source_bus_index,
                    vacc_bus_index,
                );
            }))
            .is_err(),
            "mutated certificate authority unexpectedly balanced: {label}"
        );
    }

    #[test]
    fn every_source_certificate_authority_field_mutation_rejects() {
        let labels = [
            "protocol_version",
            "source_child_capacity",
            "batch_index_lo",
            "batch_index_hi",
            "active_child_count",
            "protocol_digest",
            "relation_digest",
            "program_commitment",
            "child_occupied",
            "child_input_pc",
            "child_input_memory_root",
            "child_output_pc",
            "child_output_memory_root",
            "child_exit_code",
            "child_terminates",
            "padding_child",
            "source_accumulator_digest",
            "source_commitment_root",
            "external_logup_gkr_digest",
            "source_functional_digest",
            "setup_openings_digest",
            "source_statement_digest",
        ];
        for (case, label) in labels.into_iter().enumerate() {
            assert_mutated_certificate_rejects(Some(case), None, label);
        }
    }

    #[test]
    fn every_vacc_certificate_authority_field_mutation_rejects() {
        let labels = [
            "protocol_version",
            "input_arity",
            "batch_index_lo",
            "batch_index_hi",
            "relation_digest",
            "prior_accumulator_digest",
            "source_accumulator_digest",
            "output_accumulator_digest",
            "source_commitment_root",
            "transition_transcript_digest",
        ];
        for (case, label) in labels.into_iter().enumerate() {
            assert_mutated_certificate_rejects(None, Some(case), label);
        }
    }

    #[test]
    fn replay_key_digest_is_setup_owned_and_mutation_rejects() {
        let (air, record, _, _) = fixture();
        let honest = generate_verifier_warp_producer_bridge_trace_and_certificates_v2(
            &air,
            core::slice::from_ref(&record),
        )
        .unwrap();
        check_air_constraints(&air, &honest.matrix);

        let mut mutated = honest.matrix;
        let width = mutated.width();
        let local: &mut VerifierWarpProducerBridgeColsV2<F> = mutated.values[..width].borrow_mut();
        local.replay.key_digest[0] += F::ONE;
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                check_air_constraints(&air, &mutated);
            }))
            .is_err(),
            "proof-carried replay key unexpectedly escaped the setup binding"
        );
    }

    #[test]
    fn generator_rejects_non_setup_record_count() {
        let (air, _, _, _) = fixture();
        assert_eq!(
            generate_verifier_warp_producer_bridge_trace_and_certificates_v2(&air, &[]).err(),
            Some("verifier-WARP bridge record/profile length")
        );
    }
}

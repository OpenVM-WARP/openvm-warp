//! Production-only authority join for History-v2 source certificates.
//!
//! The ordinary producer bridge authenticates the fixed verifier source and
//! Appendix-D VACC statement.  This AIR prevents either certificate from
//! reaching History until the V3 setup-PCS authority and same-message occupancy
//! (C2) certificates for the same batch have also been consumed.  It is a
//! one-way join: no digest or host boolean can stand in for any input bus.

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
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, F};

use super::{
    FixedSetupOpeningCertificateBusV2, FixedSetupOpeningCertificateMessageV2,
    VerifierWarpCertifiedActiveCountBusV2, VerifierWarpCertifiedActiveCountMessageV2,
    VerifierWarpSourceCertificateBusV2, VerifierWarpSourceCertificateMessageV2,
    VerifierWarpVaccCertificateBusV2, VerifierWarpVaccCertificateMessageV2,
    VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2,
};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierWarpProducerAuthorityProfileV2 {
    pub relation_digest: Digest,
    pub profile_digest: Digest,
    pub source_relation_vk_digest: Digest,
    pub active_child_counts: Arc<[u8]>,
    pub fixed_claim_counts: Arc<[u32]>,
}

impl VerifierWarpProducerAuthorityProfileV2 {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.active_child_counts.is_empty()
            || self.active_child_counts.len() != self.fixed_claim_counts.len()
            || self
                .active_child_counts
                .iter()
                .enumerate()
                .any(|(index, &count)| {
                    let count = usize::from(count);
                    count == 0
                        || count > VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2
                        || (index + 1 != self.active_child_counts.len()
                            && count != VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2)
                })
        {
            return Err("invalid History-v2 producer-authority profile");
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct VerifierWarpProducerAuthorityPrepColsV2<T> {
    pub active: T,
    pub batch_index_lo: T,
    pub batch_index_hi: T,
    pub active_child_count: T,
    pub fixed_claim_count: T,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct VerifierWarpProducerAuthorityColsV2<T> {
    pub active: T,
    pub source: VerifierWarpSourceCertificateMessageV2<T>,
    pub vacc: VerifierWarpVaccCertificateMessageV2<T>,
    pub fixed: FixedSetupOpeningCertificateMessageV2<T>,
    pub count: VerifierWarpCertifiedActiveCountMessageV2<T>,
}

/// Exact four-way authority join used only by the production History-v2
/// inventory. `unchecked_source_bus` and `unchecked_vacc_bus` are private to
/// the production circuit; only the output buses are consumed by History.
#[derive(Clone, Debug)]
pub struct VerifierWarpProducerAuthorityAirV2 {
    pub profile: VerifierWarpProducerAuthorityProfileV2,
    pub unchecked_source_bus: VerifierWarpSourceCertificateBusV2,
    pub unchecked_vacc_bus: VerifierWarpVaccCertificateBusV2,
    pub fixed_bus: FixedSetupOpeningCertificateBusV2,
    pub count_bus: VerifierWarpCertifiedActiveCountBusV2,
    pub certified_source_bus: VerifierWarpSourceCertificateBusV2,
    pub certified_vacc_bus: VerifierWarpVaccCertificateBusV2,
}

impl BaseAir<F> for VerifierWarpProducerAuthorityAirV2 {
    fn width(&self) -> usize {
        core::mem::size_of::<VerifierWarpProducerAuthorityColsV2<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.profile
            .validate()
            .expect("invalid History-v2 producer-authority profile in VK");
        let width = VerifierWarpProducerAuthorityPrepColsV2::<u8>::width();
        let height = self
            .profile
            .active_child_counts
            .len()
            .next_power_of_two()
            .max(2);
        let mut values = F::zero_vec(width * height);
        for (index, (&active_count, &fixed_count)) in self
            .profile
            .active_child_counts
            .iter()
            .zip(self.profile.fixed_claim_counts.iter())
            .enumerate()
        {
            let cols: &mut VerifierWarpProducerAuthorityPrepColsV2<F> =
                values[index * width..(index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.batch_index_lo = F::from_u16(index as u16);
            cols.batch_index_hi = F::from_u16((index >> 16) as u16);
            cols.active_child_count = F::from_u8(active_count);
            cols.fixed_claim_count = F::from_u32(fixed_count);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for VerifierWarpProducerAuthorityAirV2 {}
impl PartitionedBaseAir<F> for VerifierWarpProducerAuthorityAirV2 {}

fn assert_values_eq<AB, const N: usize>(
    builder: &mut AB,
    enabled: AB::Expr,
    a: [AB::Var; N],
    b: [AB::Var; N],
) where
    AB: AirBuilder<F = F>,
    AB::Var: Copy,
{
    for (left, right) in a.into_iter().zip(b) {
        builder.when(enabled.clone()).assert_eq(left, right);
    }
}

impl<AB> Air<AB> for VerifierWarpProducerAuthorityAirV2
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.profile.validate().is_ok());
        let prep = builder.preprocessed();
        let prep_row = prep.row_slice(0).expect("producer-authority prep row");
        let prep: &VerifierWarpProducerAuthorityPrepColsV2<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("producer-authority row");
        let local: &VerifierWarpProducerAuthorityColsV2<AB::Var> = (*row).borrow();
        builder.assert_bool(prep.active);
        builder.assert_bool(local.active);
        builder.assert_eq(local.active, prep.active);
        let enabled = AB::Expr::from(local.active);

        for value in [local.source.batch_index_lo, local.vacc.batch_index_lo] {
            builder
                .when(enabled.clone())
                .assert_eq(value, prep.batch_index_lo);
        }
        for value in [local.source.batch_index_hi, local.vacc.batch_index_hi] {
            builder
                .when(enabled.clone())
                .assert_eq(value, prep.batch_index_hi);
        }
        builder.when(enabled.clone()).assert_eq(
            local.fixed.proof_index,
            AB::Expr::from(prep.batch_index_lo)
                + AB::Expr::from_u32(1 << 16) * AB::Expr::from(prep.batch_index_hi),
        );
        builder
            .when(enabled.clone())
            .assert_eq(local.count.proof_index, local.fixed.proof_index);
        builder
            .when(enabled.clone())
            .assert_eq(local.count.batch_index[0], prep.batch_index_lo);
        builder
            .when(enabled.clone())
            .assert_eq(local.count.batch_index[1], prep.batch_index_hi);
        builder
            .when(enabled.clone())
            .assert_zero(local.count.batch_index[2]);
        builder
            .when(enabled.clone())
            .assert_zero(local.count.batch_index[3]);
        builder
            .when(enabled.clone())
            .assert_eq(local.source.active_child_count, prep.active_child_count);
        builder.when(enabled.clone()).assert_eq(
            local.count.expected_active_child_count,
            prep.active_child_count,
        );
        builder
            .when(enabled.clone())
            .assert_eq(local.fixed.canonical_claim_count, prep.fixed_claim_count);

        assert_values_eq(
            builder,
            enabled.clone(),
            local.source.relation_digest,
            local.vacc.relation_digest,
        );
        assert_values_eq(
            builder,
            enabled.clone(),
            local.source.relation_digest,
            local.count.relation_digest,
        );
        for (actual, expected) in local
            .source
            .relation_digest
            .into_iter()
            .zip(self.profile.relation_digest)
        {
            builder
                .when(enabled.clone())
                .assert_eq(actual, AB::Expr::from(expected));
        }
        for (actual, expected) in local
            .count
            .profile_digest
            .into_iter()
            .zip(self.profile.profile_digest)
        {
            builder
                .when(enabled.clone())
                .assert_eq(actual, AB::Expr::from(expected));
        }
        for (actual, expected) in local
            .fixed
            .source_relation_vk_digest
            .into_iter()
            .zip(self.profile.source_relation_vk_digest)
        {
            builder
                .when(enabled.clone())
                .assert_eq(actual, AB::Expr::from(expected));
        }
        assert_values_eq(
            builder,
            enabled.clone(),
            local.source.setup_openings_digest,
            local.fixed.setup_openings_digest,
        );
        assert_values_eq(
            builder,
            enabled.clone(),
            local.source.source_commitment_root,
            local.count.source_root,
        );
        assert_values_eq(
            builder,
            enabled.clone(),
            local.source.source_commitment_root,
            local.vacc.source_commitment_root,
        );
        assert_values_eq(
            builder,
            enabled.clone(),
            local.source.source_accumulator_digest,
            local.vacc.source_accumulator_digest,
        );

        self.unchecked_source_bus
            .lookup_key(builder, local.source.clone(), enabled.clone());
        self.unchecked_vacc_bus
            .lookup_key(builder, local.vacc.clone(), enabled.clone());
        self.fixed_bus
            .receive(builder, local.fixed.clone(), enabled.clone());
        self.count_bus
            .lookup_key(builder, local.count.clone(), enabled.clone());
        self.certified_source_bus.add_key_with_lookups(
            builder,
            local.source.clone(),
            enabled.clone(),
        );
        self.certified_vacc_bus
            .add_key_with_lookups(builder, local.vacc.clone(), enabled);
    }
}

#[derive(Clone, Debug)]
pub struct VerifierWarpProducerAuthorityRecordV2 {
    pub source: VerifierWarpSourceCertificateMessageV2<F>,
    pub vacc: VerifierWarpVaccCertificateMessageV2<F>,
    pub fixed: FixedSetupOpeningCertificateMessageV2<F>,
    pub count: VerifierWarpCertifiedActiveCountMessageV2<F>,
}

pub fn generate_verifier_warp_producer_authority_trace_v2(
    air: &VerifierWarpProducerAuthorityAirV2,
    records: &[VerifierWarpProducerAuthorityRecordV2],
) -> Result<RowMajorMatrix<F>, &'static str> {
    air.profile.validate()?;
    if records.len() != air.profile.active_child_counts.len() {
        return Err("producer-authority record/profile length");
    }
    let width = core::mem::size_of::<VerifierWarpProducerAuthorityColsV2<u8>>();
    let height = records.len().next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    for (index, record) in records.iter().enumerate() {
        let cols: &mut VerifierWarpProducerAuthorityColsV2<F> =
            values[index * width..(index + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.source = record.source.clone();
        cols.vacc = record.vacc.clone();
        cols.fixed = record.fixed.clone();
        cols.count = record.count.clone();
    }
    Ok(RowMajorMatrix::new(values, width))
}

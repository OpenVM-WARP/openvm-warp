//! Fixed-capacity boundary for the History-v4 setup-PCS authority.
//!
//! SWIRL first reduces heterogeneous trace claims to ordered stacked-opening
//! claims, then WHIR authenticates the resulting constraints under the setup
//! commitment.  Runtime occupancy must therefore change neither the stacked
//! layout nor the multi-constraint WHIR verifier key.  This adapter presents
//! exactly four transitions to the ordinary V3 stacking/WHIR owners:
//!
//! - active-prefix slots copy the genuine source provenance, setup point and mapped setup claims
//!   exactly;
//! - the inactive external suffix is constrained to canonical zero; and
//! - internally, inactive slots repeat slot zero's already-authenticated setup opening.  The
//!   complete V3 statement transcript observes all four slots and the ordinary four-constraint WHIR
//!   verifier checks them.
//!
//! Repeating slot zero is padding, not a source statement.  No certificate or
//! host Boolean authorizes it: slot zero is constrained active, every active
//! bit is tied to the source-provenance permutation bus, and the padding
//! values are copied through private typed buses.  Inactive certificates and
//! preprocessed claims are never exported to History or the source verifier.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_recursion_circuit::{
    bus::{ColumnClaimsBus, ColumnClaimsMessage},
    define_typed_lookup_bus, define_typed_permutation_bus,
    system::BusIndexManager,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::{BasedVectorSpace, PrimeCharacteristicRing},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, D_EF, EF, F};

use super::{
    FixedSetupOpeningCertificateBusV2, FixedSetupOpeningCertificateMessageV2,
    FixedSetupOpeningPointBusV2, FixedSetupOpeningPointMessageV2,
};
use crate::circuit::native_warp_history_v19::{
    SetupPcsSourceProvenanceBusV3, SetupPcsSourceProvenanceMessageV3,
    SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3, TRANSCRIPT_WIDTH_V19,
};

/// The setup authority has one setup-fixed relation and one VK for all
/// nonempty occupancies of a capacity-four HLeaf.
pub const FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4: usize = 4;

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Debug)]
pub struct FixedCapacitySetupActivationMessageV4<T> {
    pub slot: T,
    pub active: T,
}
define_typed_lookup_bus!(
    FixedCapacitySetupActivationBusV4,
    FixedCapacitySetupActivationMessageV4
);
impl FixedCapacitySetupActivationBusV4 {
    #[must_use]
    pub fn index(&self) -> openvm_stark_backend::interaction::BusIndex {
        self.0.index
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Debug)]
struct FixedCapacitySetupPointPaddingMessageV4<T> {
    point_index: T,
    value: [T; D_EF],
}
define_typed_permutation_bus!(
    FixedCapacitySetupPointPaddingBusV4,
    FixedCapacitySetupPointPaddingMessageV4
);

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Debug)]
struct FixedCapacitySetupClaimPaddingMessageV4<T> {
    claim_ordinal: T,
    current: [T; D_EF],
    rotated: [T; D_EF],
}
define_typed_permutation_bus!(
    FixedCapacitySetupClaimPaddingBusV4,
    FixedCapacitySetupClaimPaddingMessageV4
);

#[repr(C)]
#[derive(AlignedBorrow, StructReflection, Clone, Debug)]
struct FixedCapacitySetupProvenancePaddingMessageV4<T> {
    protocol_version: T,
    segment_index: [T; 2],
    app_vk_digest: [T; DIGEST_SIZE],
    relation_digest: [T; DIGEST_SIZE],
    source_root: [T; DIGEST_SIZE],
    source_instance_digest: [T; DIGEST_SIZE],
    source_forest_root: [T; DIGEST_SIZE],
    segment_openings_digest: [T; DIGEST_SIZE],
    source_checkpoint_digest: [T; DIGEST_SIZE],
    source_manifest_digest: [T; DIGEST_SIZE],
    source_receipt_digest: [T; DIGEST_SIZE],
    end_tidx: [T; 2],
    end_sample_count: T,
    end_state: [T; TRANSCRIPT_WIDTH_V19],
}
define_typed_permutation_bus!(
    FixedCapacitySetupProvenancePaddingBusV4,
    FixedCapacitySetupProvenancePaddingMessageV4
);

/// One setup-column identity in the exact V3 claim-bridge order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedCapacitySetupClaimIdentityV4 {
    pub sort_idx: u32,
    pub part_idx: u32,
    pub col_idx: u32,
    pub need_rot: bool,
    pub is_cached_main: bool,
}

/// Occupancy-independent profile for the four boundary AIRs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedCapacitySetupBoundaryProfileV4 {
    pub app_vk_digest: Digest,
    pub relation_digest: Digest,
    pub opening_point_len: usize,
    pub claims: Arc<[FixedCapacitySetupClaimIdentityV4]>,
}

impl FixedCapacitySetupBoundaryProfileV4 {
    pub fn new(
        app_vk_digest: Digest,
        relation_digest: Digest,
        opening_point_len: usize,
        claims: Vec<FixedCapacitySetupClaimIdentityV4>,
    ) -> Result<Self, FixedCapacitySetupAuthorityErrorV4> {
        if app_vk_digest == Digest::default()
            || relation_digest == Digest::default()
            || opening_point_len == 0
            || claims.is_empty()
        {
            return Err(FixedCapacitySetupAuthorityErrorV4::Profile);
        }
        Ok(Self {
            app_vk_digest,
            relation_digest,
            opening_point_len,
            claims: claims.into(),
        })
    }

    fn activation_consumer_count(&self) -> Result<u32, FixedCapacitySetupAuthorityErrorV4> {
        self.opening_point_len
            .checked_add(self.claims.len())
            .and_then(|count| count.checked_add(1))
            .and_then(|count| u32::try_from(count).ok())
            .ok_or(FixedCapacitySetupAuthorityErrorV4::Overflow)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FixedCapacitySetupBoundaryBusesV4 {
    pub external_provenance: SetupPcsSourceProvenanceBusV3,
    pub external_opening_point: FixedSetupOpeningPointBusV2,
    pub external_column_claims: ColumnClaimsBus,
    pub external_certificate: FixedSetupOpeningCertificateBusV2,
    pub internal_provenance: SetupPcsSourceProvenanceBusV3,
    pub internal_opening_point: FixedSetupOpeningPointBusV2,
    pub internal_column_claims: ColumnClaimsBus,
    pub internal_certificate: FixedSetupOpeningCertificateBusV2,
    pub activation: FixedCapacitySetupActivationBusV4,
    provenance_padding: FixedCapacitySetupProvenancePaddingBusV4,
    point_padding: FixedCapacitySetupPointPaddingBusV4,
    claim_padding: FixedCapacitySetupClaimPaddingBusV4,
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct FixedCapacitySetupSlotPrepColsV4<T> {
    used: T,
    slot: T,
    is_slot_zero: T,
    transition_index: [T; 2],
    activation_consumer_count: T,
}

#[repr(C)]
#[derive(AlignedBorrow, Clone)]
pub struct FixedCapacitySetupSlotColsV4<T> {
    pub active: T,
    pub external: SetupPcsSourceProvenanceMessageV3<T>,
    slot_zero: FixedCapacitySetupProvenancePaddingMessageV4<T>,
}

#[derive(Clone, Debug)]
pub struct FixedCapacitySetupSlotAirV4 {
    pub profile: Arc<FixedCapacitySetupBoundaryProfileV4>,
    pub buses: FixedCapacitySetupBoundaryBusesV4,
}

impl BaseAir<F> for FixedCapacitySetupSlotAirV4 {
    fn width(&self) -> usize {
        core::mem::size_of::<FixedCapacitySetupSlotColsV4<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = core::mem::size_of::<FixedCapacitySetupSlotPrepColsV4<u8>>();
        let mut values = F::zero_vec(width * FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4);
        let consumers = self
            .profile
            .activation_consumer_count()
            .expect("validated fixed-capacity setup profile");
        for slot in 0..FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4 {
            let cols: &mut FixedCapacitySetupSlotPrepColsV4<F> =
                values[slot * width..(slot + 1) * width].borrow_mut();
            cols.used = F::ONE;
            cols.slot = F::from_usize(slot);
            cols.is_slot_zero = F::from_bool(slot == 0);
            cols.transition_index = split_u32_v4(slot as u32);
            cols.activation_consumer_count = F::from_u32(consumers);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}
impl BaseAirWithPublicValues<F> for FixedCapacitySetupSlotAirV4 {}
impl PartitionedBaseAir<F> for FixedCapacitySetupSlotAirV4 {}

impl<AB> Air<AB> for FixedCapacitySetupSlotAirV4
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let prep = builder.preprocessed();
        let prep_row = prep.row_slice(0).expect("fixed setup slot prep row");
        let prep: &FixedCapacitySetupSlotPrepColsV4<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("fixed setup slot row");
        let next_row = main.row_slice(1).expect("fixed setup next slot row");
        let local: &FixedCapacitySetupSlotColsV4<AB::Var> = (*row).borrow();
        let next: &FixedCapacitySetupSlotColsV4<AB::Var> = (*next_row).borrow();

        builder.assert_bool(prep.used);
        builder.assert_bool(prep.is_slot_zero);
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder
            .when_transition()
            .assert_zero((AB::Expr::ONE - AB::Expr::from(local.active)) * next.active);
        let active = AB::Expr::from(local.active);
        let inactive = AB::Expr::ONE - active.clone();
        for value in local.external.clone().to_vec() {
            builder.when(inactive.clone()).assert_zero(value);
        }
        builder.when(active.clone()).assert_eq(
            local.external.protocol_version,
            AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
        );
        assert_array_eq_v4(
            builder,
            active.clone(),
            local.external.transition_index,
            prep.transition_index,
        );
        assert_array_eq_v4(
            builder,
            active.clone(),
            local.external.app_vk_digest,
            self.profile.app_vk_digest.map(AB::Expr::from),
        );
        assert_array_eq_v4(
            builder,
            active.clone(),
            local.external.relation_digest,
            self.profile.relation_digest.map(AB::Expr::from),
        );

        self.buses
            .external_provenance
            .receive(builder, local.external.clone(), active.clone());
        self.buses.activation.add_key_with_lookups(
            builder,
            FixedCapacitySetupActivationMessageV4 {
                slot: prep.slot.into(),
                active: local.active.into(),
            },
            prep.activation_consumer_count,
        );

        let is_slot_zero = AB::Expr::from(prep.is_slot_zero);
        let later_slot = AB::Expr::ONE - is_slot_zero.clone();
        let external_padding = provenance_padding_v4(&local.external);
        assert_padding_eq_v4(
            builder,
            is_slot_zero.clone(),
            &local.slot_zero,
            &external_padding,
        );
        self.buses.provenance_padding.send(
            builder,
            external_padding,
            is_slot_zero.clone() * AB::Expr::from_usize(3),
        );
        self.buses
            .provenance_padding
            .receive(builder, local.slot_zero.clone(), later_slot);

        let selected = select_provenance_v4::<AB>(
            prep.transition_index,
            active,
            inactive,
            &local.external,
            &local.slot_zero,
        );
        // Header and claim bridge are the two ordinary V3 consumers.
        self.buses
            .internal_provenance
            .send(builder, selected.clone(), AB::Expr::TWO);
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct FixedCapacitySetupPointPrepColsV4<T> {
    used: T,
    slot: T,
    point_index: T,
    is_slot_zero: T,
}

#[repr(C)]
#[derive(AlignedBorrow, Clone)]
pub struct FixedCapacitySetupPointColsV4<T> {
    pub active: T,
    pub external_value: [T; D_EF],
    slot_zero_value: [T; D_EF],
}

#[derive(Clone, Debug)]
pub struct FixedCapacitySetupPointAirV4 {
    pub profile: Arc<FixedCapacitySetupBoundaryProfileV4>,
    pub buses: FixedCapacitySetupBoundaryBusesV4,
}

impl BaseAir<F> for FixedCapacitySetupPointAirV4 {
    fn width(&self) -> usize {
        core::mem::size_of::<FixedCapacitySetupPointColsV4<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let row_count = self
            .profile
            .opening_point_len
            .checked_mul(FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4)
            .expect("validated fixed setup point rows");
        let height = row_count.next_power_of_two().max(2);
        let width = core::mem::size_of::<FixedCapacitySetupPointPrepColsV4<u8>>();
        let mut values = F::zero_vec(width * height);
        for slot in 0..FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4 {
            for point_index in 0..self.profile.opening_point_len {
                let row = slot * self.profile.opening_point_len + point_index;
                let cols: &mut FixedCapacitySetupPointPrepColsV4<F> =
                    values[row * width..(row + 1) * width].borrow_mut();
                cols.used = F::ONE;
                cols.slot = F::from_usize(slot);
                cols.point_index = F::from_usize(point_index);
                cols.is_slot_zero = F::from_bool(slot == 0);
            }
        }
        Some(RowMajorMatrix::new(values, width))
    }
}
impl BaseAirWithPublicValues<F> for FixedCapacitySetupPointAirV4 {}
impl PartitionedBaseAir<F> for FixedCapacitySetupPointAirV4 {}

impl<AB> Air<AB> for FixedCapacitySetupPointAirV4
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let prep = builder.preprocessed();
        let prep_row = prep.row_slice(0).expect("fixed setup point prep row");
        let prep: &FixedCapacitySetupPointPrepColsV4<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("fixed setup point row");
        let local: &FixedCapacitySetupPointColsV4<AB::Var> = (*row).borrow();
        builder.assert_bool(prep.used);
        builder.assert_bool(prep.is_slot_zero);
        builder.assert_bool(local.active);
        let used = AB::Expr::from(prep.used);
        let active = AB::Expr::from(local.active);
        builder
            .when(AB::Expr::ONE - used.clone())
            .assert_zero(local.active);
        let inactive = used.clone() - active.clone();
        for value in local
            .external_value
            .into_iter()
            .chain(local.slot_zero_value)
        {
            builder
                .when(AB::Expr::ONE - used.clone())
                .assert_zero(value);
        }
        for value in local.external_value {
            builder.when(inactive.clone()).assert_zero(value);
        }
        self.buses.activation.lookup_key(
            builder,
            FixedCapacitySetupActivationMessageV4 {
                slot: prep.slot.into(),
                active: local.active.into(),
            },
            used.clone(),
        );
        self.buses.external_opening_point.lookup_key(
            builder,
            FixedSetupOpeningPointMessageV2 {
                proof_index: prep.slot.into(),
                point_index: prep.point_index.into(),
                value: local.external_value.map(Into::into),
            },
            active.clone(),
        );
        assert_array_eq_v4(
            builder,
            AB::Expr::from(prep.is_slot_zero),
            local.slot_zero_value,
            local.external_value,
        );
        self.buses.point_padding.send(
            builder,
            FixedCapacitySetupPointPaddingMessageV4 {
                point_index: prep.point_index.into(),
                value: local.external_value.map(Into::into),
            },
            AB::Expr::from(prep.is_slot_zero) * AB::Expr::from_usize(3),
        );
        self.buses.point_padding.receive(
            builder,
            FixedCapacitySetupPointPaddingMessageV4 {
                point_index: prep.point_index.into(),
                value: local.slot_zero_value.map(Into::into),
            },
            used.clone() - AB::Expr::from(prep.is_slot_zero),
        );
        let selected = core::array::from_fn(|limb| {
            active.clone() * AB::Expr::from(local.external_value[limb])
                + inactive.clone() * AB::Expr::from(local.slot_zero_value[limb])
        });
        // Statement header and ordered-stacking source-point adapter each
        // consume the same exact coordinate.
        self.buses.internal_opening_point.add_key_with_lookups(
            builder,
            FixedSetupOpeningPointMessageV2 {
                proof_index: prep.slot.into(),
                point_index: prep.point_index.into(),
                value: selected,
            },
            AB::Expr::TWO * used,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct FixedCapacitySetupClaimPrepColsV4<T> {
    used: T,
    slot: T,
    claim_ordinal: T,
    sort_idx: T,
    part_idx: T,
    col_idx: T,
    need_rot: T,
    is_cached_main: T,
    is_slot_zero: T,
}

#[repr(C)]
#[derive(AlignedBorrow, Clone)]
pub struct FixedCapacitySetupClaimColsV4<T> {
    pub active: T,
    pub external_current: [T; D_EF],
    pub external_rotated: [T; D_EF],
    slot_zero_current: [T; D_EF],
    slot_zero_rotated: [T; D_EF],
}

#[derive(Clone, Debug)]
pub struct FixedCapacitySetupClaimAirV4 {
    pub profile: Arc<FixedCapacitySetupBoundaryProfileV4>,
    pub buses: FixedCapacitySetupBoundaryBusesV4,
}

impl BaseAir<F> for FixedCapacitySetupClaimAirV4 {
    fn width(&self) -> usize {
        core::mem::size_of::<FixedCapacitySetupClaimColsV4<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let row_count = self
            .profile
            .claims
            .len()
            .checked_mul(FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4)
            .expect("validated fixed setup claim rows");
        let height = row_count.next_power_of_two().max(2);
        let width = core::mem::size_of::<FixedCapacitySetupClaimPrepColsV4<u8>>();
        let mut values = F::zero_vec(width * height);
        for slot in 0..FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4 {
            for (claim_ordinal, identity) in self.profile.claims.iter().enumerate() {
                let row = slot * self.profile.claims.len() + claim_ordinal;
                let cols: &mut FixedCapacitySetupClaimPrepColsV4<F> =
                    values[row * width..(row + 1) * width].borrow_mut();
                cols.used = F::ONE;
                cols.slot = F::from_usize(slot);
                cols.claim_ordinal = F::from_usize(claim_ordinal);
                cols.sort_idx = F::from_u32(identity.sort_idx);
                cols.part_idx = F::from_u32(identity.part_idx);
                cols.col_idx = F::from_u32(identity.col_idx);
                cols.need_rot = F::from_bool(identity.need_rot);
                cols.is_cached_main = F::from_bool(identity.is_cached_main);
                cols.is_slot_zero = F::from_bool(slot == 0);
            }
        }
        Some(RowMajorMatrix::new(values, width))
    }
}
impl BaseAirWithPublicValues<F> for FixedCapacitySetupClaimAirV4 {}
impl PartitionedBaseAir<F> for FixedCapacitySetupClaimAirV4 {}

impl<AB> Air<AB> for FixedCapacitySetupClaimAirV4
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let prep = builder.preprocessed();
        let prep_row = prep.row_slice(0).expect("fixed setup claim prep row");
        let prep: &FixedCapacitySetupClaimPrepColsV4<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("fixed setup claim row");
        let local: &FixedCapacitySetupClaimColsV4<AB::Var> = (*row).borrow();
        for bit in [
            prep.used,
            prep.need_rot,
            prep.is_cached_main,
            prep.is_slot_zero,
            local.active,
        ] {
            builder.assert_bool(bit);
        }
        let used = AB::Expr::from(prep.used);
        let active = AB::Expr::from(local.active);
        builder
            .when(AB::Expr::ONE - used.clone())
            .assert_zero(local.active);
        let inactive = used.clone() - active.clone();
        for value in local
            .external_current
            .into_iter()
            .chain(local.external_rotated)
            .chain(local.slot_zero_current)
            .chain(local.slot_zero_rotated)
        {
            builder
                .when(AB::Expr::ONE - used.clone())
                .assert_zero(value);
        }
        for value in local
            .external_current
            .into_iter()
            .chain(local.external_rotated)
        {
            builder.when(inactive.clone()).assert_zero(value);
        }
        for value in local
            .external_rotated
            .into_iter()
            .chain(local.slot_zero_rotated)
        {
            builder
                .when(used.clone() * (AB::Expr::ONE - AB::Expr::from(prep.need_rot)))
                .assert_zero(value);
        }
        self.buses.activation.lookup_key(
            builder,
            FixedCapacitySetupActivationMessageV4 {
                slot: prep.slot.into(),
                active: local.active.into(),
            },
            used.clone(),
        );
        assert_array_eq_v4(
            builder,
            AB::Expr::from(prep.is_slot_zero),
            local.slot_zero_current,
            local.external_current,
        );
        assert_array_eq_v4(
            builder,
            AB::Expr::from(prep.is_slot_zero),
            local.slot_zero_rotated,
            local.external_rotated,
        );
        let padding = FixedCapacitySetupClaimPaddingMessageV4 {
            claim_ordinal: prep.claim_ordinal.into(),
            current: local.external_current.map(Into::into),
            rotated: local.external_rotated.map(Into::into),
        };
        self.buses.claim_padding.send(
            builder,
            padding,
            AB::Expr::from(prep.is_slot_zero) * AB::Expr::from_usize(3),
        );
        self.buses.claim_padding.receive(
            builder,
            FixedCapacitySetupClaimPaddingMessageV4 {
                claim_ordinal: prep.claim_ordinal.into(),
                current: local.slot_zero_current.map(Into::into),
                rotated: local.slot_zero_rotated.map(Into::into),
            },
            used.clone() - AB::Expr::from(prep.is_slot_zero),
        );
        let current = core::array::from_fn(|limb| {
            active.clone() * AB::Expr::from(local.external_current[limb])
                + inactive.clone() * AB::Expr::from(local.slot_zero_current[limb])
        });
        let rotated = core::array::from_fn(|limb| {
            active.clone() * AB::Expr::from(local.external_rotated[limb])
                + inactive.clone() * AB::Expr::from(local.slot_zero_rotated[limb])
        });
        let current_message = ColumnClaimsMessage {
            sort_idx: prep.sort_idx.into(),
            part_idx: prep.part_idx.into(),
            col_idx: prep.col_idx.into(),
            claim: current,
            is_rot: AB::Expr::ZERO,
        };
        let rotated_message = ColumnClaimsMessage {
            sort_idx: prep.sort_idx.into(),
            part_idx: prep.part_idx.into(),
            col_idx: prep.col_idx.into(),
            claim: rotated,
            is_rot: AB::Expr::ONE,
        };
        let cached = AB::Expr::from(prep.is_cached_main);
        let preprocessed = used.clone() - cached.clone();
        self.buses.external_column_claims.receive(
            builder,
            prep.slot,
            current_message.clone(),
            active.clone() * cached.clone(),
        );
        self.buses.external_column_claims.receive(
            builder,
            prep.slot,
            rotated_message.clone(),
            active.clone() * cached.clone() * AB::Expr::from(prep.need_rot),
        );
        self.buses.internal_column_claims.send(
            builder,
            prep.slot,
            current_message.clone(),
            cached.clone(),
        );
        self.buses.internal_column_claims.send(
            builder,
            prep.slot,
            rotated_message.clone(),
            cached.clone() * AB::Expr::from(prep.need_rot),
        );
        self.buses.internal_column_claims.receive(
            builder,
            prep.slot,
            current_message.clone(),
            preprocessed.clone(),
        );
        self.buses.internal_column_claims.receive(
            builder,
            prep.slot,
            rotated_message.clone(),
            preprocessed.clone() * AB::Expr::from(prep.need_rot),
        );
        self.buses.external_column_claims.send(
            builder,
            prep.slot,
            current_message,
            active.clone() * preprocessed.clone(),
        );
        self.buses.external_column_claims.send(
            builder,
            prep.slot,
            rotated_message,
            active * preprocessed * AB::Expr::from(prep.need_rot),
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct FixedCapacitySetupCertificatePrepColsV4<T> {
    used: T,
    slot: T,
}

#[repr(C)]
#[derive(AlignedBorrow, Clone)]
pub struct FixedCapacitySetupCertificateColsV4<T> {
    pub active: T,
    pub internal: FixedSetupOpeningCertificateMessageV2<T>,
    pub external: FixedSetupOpeningCertificateMessageV2<T>,
}

#[derive(Clone, Debug)]
pub struct FixedCapacitySetupCertificateAirV4 {
    pub profile: Arc<FixedCapacitySetupBoundaryProfileV4>,
    pub buses: FixedCapacitySetupBoundaryBusesV4,
}

impl BaseAir<F> for FixedCapacitySetupCertificateAirV4 {
    fn width(&self) -> usize {
        core::mem::size_of::<FixedCapacitySetupCertificateColsV4<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = core::mem::size_of::<FixedCapacitySetupCertificatePrepColsV4<u8>>();
        let mut values = F::zero_vec(width * FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4);
        for slot in 0..FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4 {
            let cols: &mut FixedCapacitySetupCertificatePrepColsV4<F> =
                values[slot * width..(slot + 1) * width].borrow_mut();
            cols.used = F::ONE;
            cols.slot = F::from_usize(slot);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}
impl BaseAirWithPublicValues<F> for FixedCapacitySetupCertificateAirV4 {}
impl PartitionedBaseAir<F> for FixedCapacitySetupCertificateAirV4 {}

impl<AB> Air<AB> for FixedCapacitySetupCertificateAirV4
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let prep = builder.preprocessed();
        let prep_row = prep.row_slice(0).expect("fixed setup certificate prep row");
        let prep: &FixedCapacitySetupCertificatePrepColsV4<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("fixed setup certificate row");
        let local: &FixedCapacitySetupCertificateColsV4<AB::Var> = (*row).borrow();
        builder.assert_bool(prep.used);
        builder.assert_bool(local.active);
        let active = AB::Expr::from(local.active);
        let inactive = AB::Expr::from(prep.used) - active.clone();
        self.buses.activation.lookup_key(
            builder,
            FixedCapacitySetupActivationMessageV4 {
                slot: prep.slot.into(),
                active: local.active.into(),
            },
            prep.used,
        );
        self.buses
            .internal_certificate
            .receive(builder, local.internal.clone(), prep.used);
        for (external, internal) in local
            .external
            .clone()
            .to_vec()
            .into_iter()
            .zip(local.internal.clone().to_vec())
        {
            builder.when(active.clone()).assert_eq(external, internal);
        }
        for value in local.external.clone().to_vec() {
            builder.when(inactive.clone()).assert_zero(value);
        }
        self.buses
            .external_certificate
            .send(builder, local.external.clone(), active);
    }
}

/// Four AIRs inserted between genuine HLeaf sources and the unchanged V3
/// recursive setup authority.
#[derive(Clone, Debug)]
pub struct FixedCapacitySetupBoundaryAirsV4 {
    pub profile: Arc<FixedCapacitySetupBoundaryProfileV4>,
    pub slot: FixedCapacitySetupSlotAirV4,
    pub point: FixedCapacitySetupPointAirV4,
    pub claim: FixedCapacitySetupClaimAirV4,
    pub certificate: FixedCapacitySetupCertificateAirV4,
}

impl FixedCapacitySetupBoundaryAirsV4 {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        profile: FixedCapacitySetupBoundaryProfileV4,
        external_provenance: SetupPcsSourceProvenanceBusV3,
        external_opening_point: FixedSetupOpeningPointBusV2,
        external_column_claims: ColumnClaimsBus,
        external_certificate: FixedSetupOpeningCertificateBusV2,
        internal_provenance: SetupPcsSourceProvenanceBusV3,
        internal_opening_point: FixedSetupOpeningPointBusV2,
        internal_column_claims: ColumnClaimsBus,
        internal_certificate: FixedSetupOpeningCertificateBusV2,
        manager: &mut BusIndexManager,
    ) -> Result<Self, FixedCapacitySetupAuthorityErrorV4> {
        profile.activation_consumer_count()?;
        let profile = Arc::new(profile);
        let buses = FixedCapacitySetupBoundaryBusesV4 {
            external_provenance,
            external_opening_point,
            external_column_claims,
            external_certificate,
            internal_provenance,
            internal_opening_point,
            internal_column_claims,
            internal_certificate,
            activation: FixedCapacitySetupActivationBusV4::new(manager.new_bus_idx()),
            provenance_padding: FixedCapacitySetupProvenancePaddingBusV4::new(
                manager.new_bus_idx(),
            ),
            point_padding: FixedCapacitySetupPointPaddingBusV4::new(manager.new_bus_idx()),
            claim_padding: FixedCapacitySetupClaimPaddingBusV4::new(manager.new_bus_idx()),
        };
        Ok(Self {
            profile: Arc::clone(&profile),
            slot: FixedCapacitySetupSlotAirV4 {
                profile: Arc::clone(&profile),
                buses,
            },
            point: FixedCapacitySetupPointAirV4 {
                profile: Arc::clone(&profile),
                buses,
            },
            claim: FixedCapacitySetupClaimAirV4 {
                profile: Arc::clone(&profile),
                buses,
            },
            certificate: FixedCapacitySetupCertificateAirV4 { profile, buses },
        })
    }

    #[must_use]
    pub fn source_point_demands(&self) -> Vec<(u32, u32, u32)> {
        (0..FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4)
            .flat_map(|slot| {
                (0..self.profile.opening_point_len).map(move |point| (slot as u32, point as u32, 1))
            })
            .collect()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedCapacitySetupTransitionRecordV4 {
    pub provenance: SetupPcsSourceProvenanceMessageV3<F>,
    pub opening_point: Vec<EF>,
    pub claims: Vec<FixedCapacitySetupClaimRecordV4>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FixedCapacitySetupClaimRecordV4 {
    pub current: EF,
    pub rotated: Option<EF>,
}

#[derive(Clone, Debug)]
pub struct FixedCapacitySetupBoundaryTracesV4 {
    pub slot: RowMajorMatrix<F>,
    pub point: RowMajorMatrix<F>,
    pub claim: RowMajorMatrix<F>,
    pub certificate: RowMajorMatrix<F>,
}

/// Generate the adapter witnesses. `active` contains only genuine source
/// records. `internal_certificates` contains the four certificates emitted by
/// the padded V3 authority and grants no authority by itself.
pub fn generate_fixed_capacity_setup_boundary_traces_v4(
    airs: &FixedCapacitySetupBoundaryAirsV4,
    active: &[FixedCapacitySetupTransitionRecordV4],
    internal_certificates: &[FixedSetupOpeningCertificateMessageV2<F>],
) -> Result<FixedCapacitySetupBoundaryTracesV4, FixedCapacitySetupAuthorityErrorV4> {
    let occupancy = active.len();
    if !(1..=FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4).contains(&occupancy)
        || internal_certificates.len() != FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4
    {
        return Err(FixedCapacitySetupAuthorityErrorV4::Occupancy);
    }
    for (slot, record) in active.iter().enumerate() {
        if record.provenance.protocol_version
            != F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3)
            || record.provenance.transition_index != split_u32_v4(slot as u32)
            || record.provenance.app_vk_digest != airs.profile.app_vk_digest
            || record.provenance.relation_digest != airs.profile.relation_digest
            || record.opening_point.len() != airs.profile.opening_point_len
            || record.claims.len() != airs.profile.claims.len()
            || record
                .claims
                .iter()
                .zip(airs.profile.claims.iter())
                .any(|(claim, identity)| claim.rotated.is_some() != identity.need_rot)
        {
            return Err(FixedCapacitySetupAuthorityErrorV4::Record(slot));
        }
    }
    for (slot, certificate) in internal_certificates.iter().enumerate() {
        if certificate.proof_index != F::from_usize(slot) {
            return Err(FixedCapacitySetupAuthorityErrorV4::Certificate(slot));
        }
    }

    let first = active
        .first()
        .ok_or(FixedCapacitySetupAuthorityErrorV4::Occupancy)?;
    let slot_width = airs.slot.width();
    let mut slot_values = F::zero_vec(
        slot_width
            .checked_mul(FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4)
            .ok_or(FixedCapacitySetupAuthorityErrorV4::Overflow)?,
    );
    for slot in 0..FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4 {
        let cols: &mut FixedCapacitySetupSlotColsV4<F> =
            slot_values[slot * slot_width..(slot + 1) * slot_width].borrow_mut();
        cols.active = F::from_bool(slot < occupancy);
        if let Some(record) = active.get(slot) {
            cols.external = record.provenance.clone();
        }
        cols.slot_zero = provenance_padding_v4(&first.provenance);
    }

    let point_rows = airs
        .profile
        .opening_point_len
        .checked_mul(FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4)
        .ok_or(FixedCapacitySetupAuthorityErrorV4::Overflow)?;
    let point_width = airs.point.width();
    let point_height = point_rows.next_power_of_two().max(2);
    let mut point_values = F::zero_vec(
        point_width
            .checked_mul(point_height)
            .ok_or(FixedCapacitySetupAuthorityErrorV4::Overflow)?,
    );
    for slot in 0..FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4 {
        for point_index in 0..airs.profile.opening_point_len {
            let row = slot * airs.profile.opening_point_len + point_index;
            let cols: &mut FixedCapacitySetupPointColsV4<F> =
                point_values[row * point_width..(row + 1) * point_width].borrow_mut();
            cols.active = F::from_bool(slot < occupancy);
            if let Some(record) = active.get(slot) {
                copy_ext_v4(&mut cols.external_value, record.opening_point[point_index]);
            }
            copy_ext_v4(&mut cols.slot_zero_value, first.opening_point[point_index]);
        }
    }

    let claim_rows = airs
        .profile
        .claims
        .len()
        .checked_mul(FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4)
        .ok_or(FixedCapacitySetupAuthorityErrorV4::Overflow)?;
    let claim_width = airs.claim.width();
    let claim_height = claim_rows.next_power_of_two().max(2);
    let mut claim_values = F::zero_vec(
        claim_width
            .checked_mul(claim_height)
            .ok_or(FixedCapacitySetupAuthorityErrorV4::Overflow)?,
    );
    for slot in 0..FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4 {
        for claim_ordinal in 0..airs.profile.claims.len() {
            let row = slot * airs.profile.claims.len() + claim_ordinal;
            let cols: &mut FixedCapacitySetupClaimColsV4<F> =
                claim_values[row * claim_width..(row + 1) * claim_width].borrow_mut();
            cols.active = F::from_bool(slot < occupancy);
            if let Some(record) = active.get(slot) {
                let claim = &record.claims[claim_ordinal];
                copy_ext_v4(&mut cols.external_current, claim.current);
                if let Some(rotated) = claim.rotated {
                    copy_ext_v4(&mut cols.external_rotated, rotated);
                }
            }
            let padding = &first.claims[claim_ordinal];
            copy_ext_v4(&mut cols.slot_zero_current, padding.current);
            if let Some(rotated) = padding.rotated {
                copy_ext_v4(&mut cols.slot_zero_rotated, rotated);
            }
        }
    }

    let certificate_width = airs.certificate.width();
    let mut certificate_values = F::zero_vec(
        certificate_width
            .checked_mul(FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4)
            .ok_or(FixedCapacitySetupAuthorityErrorV4::Overflow)?,
    );
    for slot in 0..FIXED_CAPACITY_SETUP_AUTHORITY_CAPACITY_V4 {
        let cols: &mut FixedCapacitySetupCertificateColsV4<F> = certificate_values
            [slot * certificate_width..(slot + 1) * certificate_width]
            .borrow_mut();
        cols.active = F::from_bool(slot < occupancy);
        cols.internal = internal_certificates[slot].clone();
        if slot < occupancy {
            cols.external = internal_certificates[slot].clone();
        }
    }

    Ok(FixedCapacitySetupBoundaryTracesV4 {
        slot: RowMajorMatrix::new(slot_values, slot_width),
        point: RowMajorMatrix::new(point_values, point_width),
        claim: RowMajorMatrix::new(claim_values, claim_width),
        certificate: RowMajorMatrix::new(certificate_values, certificate_width),
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FixedCapacitySetupAuthorityErrorV4 {
    Profile,
    Occupancy,
    Record(usize),
    Certificate(usize),
    Overflow,
}

impl core::fmt::Display for FixedCapacitySetupAuthorityErrorV4 {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Profile => {
                formatter.write_str("fixed-capacity setup authority profile is malformed")
            }
            Self::Occupancy => {
                formatter.write_str("fixed-capacity setup authority occupancy must be in 1..=4")
            }
            Self::Record(slot) => write!(
                formatter,
                "fixed-capacity setup authority source record {slot} is malformed"
            ),
            Self::Certificate(slot) => write!(
                formatter,
                "fixed-capacity setup authority certificate {slot} is malformed"
            ),
            Self::Overflow => {
                formatter.write_str("fixed-capacity setup authority size arithmetic overflow")
            }
        }
    }
}

impl std::error::Error for FixedCapacitySetupAuthorityErrorV4 {}

fn split_u32_v4(value: u32) -> [F; 2] {
    [F::from_u32(value & 0xffff), F::from_u32(value >> 16)]
}

fn copy_ext_v4(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

fn provenance_padding_v4<T: Copy>(
    source: &SetupPcsSourceProvenanceMessageV3<T>,
) -> FixedCapacitySetupProvenancePaddingMessageV4<T> {
    FixedCapacitySetupProvenancePaddingMessageV4 {
        protocol_version: source.protocol_version,
        segment_index: source.segment_index,
        app_vk_digest: source.app_vk_digest,
        relation_digest: source.relation_digest,
        source_root: source.source_root,
        source_instance_digest: source.source_instance_digest,
        source_forest_root: source.source_forest_root,
        segment_openings_digest: source.segment_openings_digest,
        source_checkpoint_digest: source.source_checkpoint_digest,
        source_manifest_digest: source.source_manifest_digest,
        source_receipt_digest: source.source_receipt_digest,
        end_tidx: source.end_tidx,
        end_sample_count: source.end_sample_count,
        end_state: source.end_state,
    }
}

fn select_provenance_v4<AB>(
    transition_index: [AB::Var; 2],
    active: AB::Expr,
    inactive: AB::Expr,
    external: &SetupPcsSourceProvenanceMessageV3<AB::Var>,
    padding: &FixedCapacitySetupProvenancePaddingMessageV4<AB::Var>,
) -> SetupPcsSourceProvenanceMessageV3<AB::Expr>
where
    AB: AirBuilder<F = F>,
    AB::Var: Copy,
{
    let select = |actual: AB::Var, pad: AB::Var| {
        active.clone() * AB::Expr::from(actual) + inactive.clone() * AB::Expr::from(pad)
    };
    SetupPcsSourceProvenanceMessageV3 {
        protocol_version: select(external.protocol_version, padding.protocol_version),
        transition_index: transition_index.map(Into::into),
        segment_index: core::array::from_fn(|i| {
            select(external.segment_index[i], padding.segment_index[i])
        }),
        app_vk_digest: core::array::from_fn(|i| {
            select(external.app_vk_digest[i], padding.app_vk_digest[i])
        }),
        relation_digest: core::array::from_fn(|i| {
            select(external.relation_digest[i], padding.relation_digest[i])
        }),
        source_root: core::array::from_fn(|i| {
            select(external.source_root[i], padding.source_root[i])
        }),
        source_instance_digest: core::array::from_fn(|i| {
            select(
                external.source_instance_digest[i],
                padding.source_instance_digest[i],
            )
        }),
        source_forest_root: core::array::from_fn(|i| {
            select(
                external.source_forest_root[i],
                padding.source_forest_root[i],
            )
        }),
        segment_openings_digest: core::array::from_fn(|i| {
            select(
                external.segment_openings_digest[i],
                padding.segment_openings_digest[i],
            )
        }),
        source_checkpoint_digest: core::array::from_fn(|i| {
            select(
                external.source_checkpoint_digest[i],
                padding.source_checkpoint_digest[i],
            )
        }),
        source_manifest_digest: core::array::from_fn(|i| {
            select(
                external.source_manifest_digest[i],
                padding.source_manifest_digest[i],
            )
        }),
        source_receipt_digest: core::array::from_fn(|i| {
            select(
                external.source_receipt_digest[i],
                padding.source_receipt_digest[i],
            )
        }),
        end_tidx: core::array::from_fn(|i| select(external.end_tidx[i], padding.end_tidx[i])),
        end_sample_count: select(external.end_sample_count, padding.end_sample_count),
        end_state: core::array::from_fn(|i| select(external.end_state[i], padding.end_state[i])),
    }
}

fn assert_array_eq_v4<AB, L, R, const N: usize>(
    builder: &mut AB,
    enabled: impl Into<AB::Expr> + Clone,
    left: [L; N],
    right: [R; N],
) where
    AB: AirBuilder<F = F>,
    L: Into<AB::Expr>,
    R: Into<AB::Expr>,
{
    for (left, right) in left.into_iter().zip(right) {
        builder
            .when(enabled.clone())
            .assert_eq(left.into(), right.into());
    }
}

fn assert_padding_eq_v4<AB>(
    builder: &mut AB,
    enabled: impl Into<AB::Expr> + Clone,
    left: &FixedCapacitySetupProvenancePaddingMessageV4<AB::Var>,
    right: &FixedCapacitySetupProvenancePaddingMessageV4<AB::Var>,
) where
    AB: AirBuilder<F = F>,
    AB::Var: Copy,
{
    for (left, right) in left
        .clone()
        .to_vec()
        .into_iter()
        .zip(right.clone().to_vec())
    {
        builder.when(enabled.clone()).assert_eq(left, right);
    }
}

#[cfg(test)]
mod tests {
    use std::{panic::AssertUnwindSafe, sync::Arc};

    use openvm_recursion_circuit::bus::{ColumnClaimsBus, ColumnClaimsMessage};
    use openvm_stark_backend::{
        air_builders::{
            debug::{check_constraints, check_logup},
            symbolic::get_symbolic_builder,
        },
        interaction::SymbolicInteraction,
        keygen::types::TraceWidth,
        p3_matrix::Matrix,
        AirRef, AnyAir,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config as SC;

    use super::*;
    use crate::circuit::verifier_warp_history_v2::FIXED_SETUP_OPENING_PROTOCOL_V2;

    fn digest(value: u32) -> Digest {
        [F::from_u32(value); DIGEST_SIZE]
    }

    fn fixture() -> FixedCapacitySetupBoundaryAirsV4 {
        let external = FixedCapacitySetupBoundaryBusesV4 {
            external_provenance: SetupPcsSourceProvenanceBusV3::new(0),
            external_opening_point: FixedSetupOpeningPointBusV2::new(1),
            external_column_claims: ColumnClaimsBus::new(2),
            external_certificate: FixedSetupOpeningCertificateBusV2::new(3),
            internal_provenance: SetupPcsSourceProvenanceBusV3::new(4),
            internal_opening_point: FixedSetupOpeningPointBusV2::new(5),
            internal_column_claims: ColumnClaimsBus::new(6),
            internal_certificate: FixedSetupOpeningCertificateBusV2::new(7),
            activation: FixedCapacitySetupActivationBusV4::new(8),
            provenance_padding: FixedCapacitySetupProvenancePaddingBusV4::new(9),
            point_padding: FixedCapacitySetupPointPaddingBusV4::new(10),
            claim_padding: FixedCapacitySetupClaimPaddingBusV4::new(11),
        };
        let mut manager = BusIndexManager::from_next_bus_idx(8);
        let airs = FixedCapacitySetupBoundaryAirsV4::new(
            FixedCapacitySetupBoundaryProfileV4::new(
                digest(1),
                digest(2),
                2,
                vec![
                    FixedCapacitySetupClaimIdentityV4 {
                        sort_idx: 0,
                        part_idx: 1,
                        col_idx: 3,
                        need_rot: true,
                        is_cached_main: true,
                    },
                    FixedCapacitySetupClaimIdentityV4 {
                        sort_idx: 1,
                        part_idx: 0,
                        col_idx: 5,
                        need_rot: false,
                        is_cached_main: false,
                    },
                ],
            )
            .unwrap(),
            external.external_provenance,
            external.external_opening_point,
            external.external_column_claims,
            external.external_certificate,
            external.internal_provenance,
            external.internal_opening_point,
            external.internal_column_claims,
            external.internal_certificate,
            &mut manager,
        )
        .unwrap();
        assert_eq!(airs.slot.buses.activation.index(), 8);
        airs
    }

    fn provenance(slot: usize) -> SetupPcsSourceProvenanceMessageV3<F> {
        SetupPcsSourceProvenanceMessageV3 {
            protocol_version: F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition_index: split_u32_v4(slot as u32),
            segment_index: split_u32_v4(100 + slot as u32),
            app_vk_digest: digest(1),
            relation_digest: digest(2),
            source_root: digest(10 + slot as u32),
            source_instance_digest: digest(20 + slot as u32),
            source_forest_root: digest(30 + slot as u32),
            segment_openings_digest: digest(40 + slot as u32),
            source_checkpoint_digest: digest(50 + slot as u32),
            source_manifest_digest: digest(60 + slot as u32),
            source_receipt_digest: digest(70 + slot as u32),
            end_tidx: split_u32_v4(200 + slot as u32),
            end_sample_count: F::from_usize(300 + slot),
            end_state: [F::from_usize(400 + slot); TRANSCRIPT_WIDTH_V19],
        }
    }

    fn record(slot: usize) -> FixedCapacitySetupTransitionRecordV4 {
        FixedCapacitySetupTransitionRecordV4 {
            provenance: provenance(slot),
            opening_point: vec![EF::from_usize(500 + slot), EF::from_usize(510 + slot)],
            claims: vec![
                FixedCapacitySetupClaimRecordV4 {
                    current: EF::from_usize(600 + slot),
                    rotated: Some(EF::from_usize(610 + slot)),
                },
                FixedCapacitySetupClaimRecordV4 {
                    current: EF::from_usize(620 + slot),
                    rotated: None,
                },
            ],
        }
    }

    fn certificate(slot: usize) -> FixedSetupOpeningCertificateMessageV2<F> {
        FixedSetupOpeningCertificateMessageV2 {
            protocol_version: F::from_u32(FIXED_SETUP_OPENING_PROTOCOL_V2),
            proof_index: F::from_usize(slot),
            canonical_claim_count: F::from_usize(2),
            source_relation_vk_digest: digest(1),
            setup_openings_digest: digest(80 + slot as u32),
        }
    }

    fn symbolic_interactions(air: &dyn AnyAir<SC>) -> Vec<SymbolicInteraction<F>> {
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

    fn check_boundary(
        airs: &FixedCapacitySetupBoundaryAirsV4,
        traces: &FixedCapacitySetupBoundaryTracesV4,
    ) {
        let air_refs: Vec<AirRef<SC>> = vec![
            Arc::new(airs.slot.clone()),
            Arc::new(airs.point.clone()),
            Arc::new(airs.claim.clone()),
            Arc::new(airs.certificate.clone()),
        ];
        let matrices = [
            &traces.slot,
            &traces.point,
            &traces.claim,
            &traces.certificate,
        ];
        let preprocessed_owned = air_refs
            .iter()
            .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
            .collect::<Vec<_>>();
        for ((air, matrix), preprocessed) in air_refs.iter().zip(matrices).zip(&preprocessed_owned)
        {
            check_constraints::<_, SC>(
                air.as_ref(),
                &air.name(),
                &preprocessed.as_ref().map(RowMajorMatrix::as_view),
                &[matrix.as_view()],
                &[],
            );
        }

        let selected_buses = [
            airs.slot.buses.activation.index(),
            airs.slot.buses.provenance_padding.index(),
            airs.slot.buses.point_padding.index(),
            airs.slot.buses.claim_padding.index(),
        ];
        let interactions = air_refs
            .iter()
            .map(|air| {
                symbolic_interactions(air.as_ref())
                    .into_iter()
                    .filter(|interaction| selected_buses.contains(&interaction.bus_index))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let preprocessed = preprocessed_owned
            .iter()
            .map(|matrix| matrix.as_ref().map(RowMajorMatrix::as_view))
            .collect::<Vec<_>>();
        let views = matrices
            .iter()
            .map(|matrix| vec![matrix.as_view()])
            .collect::<Vec<_>>();
        check_logup(
            &air_refs.iter().map(|air| air.name()).collect::<Vec<_>>(),
            &interactions,
            &preprocessed,
            &views,
            &vec![Vec::new(); air_refs.len()],
        );
    }

    #[test]
    fn one_key_accepts_every_nonempty_occupancy_and_balances_padding() {
        let airs = fixture();
        let certificates = (0..4).map(certificate).collect::<Vec<_>>();
        for occupancy in 1..=4 {
            let records = (0..occupancy).map(record).collect::<Vec<_>>();
            let traces =
                generate_fixed_capacity_setup_boundary_traces_v4(&airs, &records, &certificates)
                    .unwrap();
            check_boundary(&airs, &traces);
        }
    }

    #[test]
    fn inactive_suffix_is_canonical_zero() {
        let airs = fixture();
        let certificates = (0..4).map(certificate).collect::<Vec<_>>();
        let mut traces =
            generate_fixed_capacity_setup_boundary_traces_v4(&airs, &[record(0)], &certificates)
                .unwrap();
        let width = airs.slot.width();
        let cols: &mut FixedCapacitySetupSlotColsV4<F> =
            traces.slot.values[width..2 * width].borrow_mut();
        cols.external.source_root[0] = F::ONE;
        let preprocessed = airs.slot.preprocessed_trace().unwrap();
        let rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
            check_constraints::<_, SC>(
                &airs.slot,
                "fixed-capacity setup slots",
                &Some(preprocessed.as_view()),
                &[traces.slot.as_view()],
                &[],
            );
        }));
        assert!(rejected.is_err());
    }

    #[test]
    fn malformed_inputs_return_errors_without_panicking() {
        let airs = fixture();
        let certificates = (0..4).map(certificate).collect::<Vec<_>>();
        for records in [Vec::new(), (0..5).map(record).collect::<Vec<_>>()] {
            let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
                generate_fixed_capacity_setup_boundary_traces_v4(&airs, &records, &certificates)
            }));
            assert!(matches!(
                result,
                Ok(Err(FixedCapacitySetupAuthorityErrorV4::Occupancy))
            ));
        }

        let mut wrong_record = record(0);
        wrong_record.provenance.transition_index = split_u32_v4(3);
        assert!(matches!(
            generate_fixed_capacity_setup_boundary_traces_v4(&airs, &[wrong_record], &certificates,),
            Err(FixedCapacitySetupAuthorityErrorV4::Record(0))
        ));

        let mut wrong_certificates = certificates;
        wrong_certificates[2].proof_index = F::from_usize(3);
        assert!(matches!(
            generate_fixed_capacity_setup_boundary_traces_v4(
                &airs,
                &[record(0)],
                &wrong_certificates,
            ),
            Err(FixedCapacitySetupAuthorityErrorV4::Certificate(2))
        ));
    }

    // Keep the typed claim message in this focused test module's compile path;
    // this detects accidental divergence from the recursive lane bus schema.
    const _: usize = core::mem::size_of::<ColumnClaimsMessage<F>>();
}

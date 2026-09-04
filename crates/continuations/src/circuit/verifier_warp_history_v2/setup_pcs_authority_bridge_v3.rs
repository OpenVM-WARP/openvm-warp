//! Compact circuit bridge from terminal setup-PCS authority to SWIRL claims.
//!
//! The retired `FixedSetupOpeningAirV2` evaluated every setup table once per
//! transition.  This AIR deliberately contains no setup values and performs
//! no polynomial evaluation.  A terminal verifier first checks the ordinary
//! stacked opening reductions and multi-constraint WHIR proof against the
//! setup commitments fixed by the aggregation VK.  Only that verifier may
//! send [`SetupPcsAuthorityCompletionMessageV3`] and the corresponding
//! [`SetupPcsAuthorityTransitionCertificateMessageV3`] messages.
//!
//! This bridge then authenticates the value-free claim schedule in its
//! preprocessed trace, hashes the supplied current/rotated values into a V3
//! transition digest, consumes exactly one authority certificate per
//! transition. Preprocessed claims are published for the symbolic evaluator;
//! cached-main claims instead consume one extra copy published by the mapped
//! source, forcing the independently PCS-authenticated and source-message
//! claims to be identical. The mapped source's ordinary copies continue to
//! feed the symbolic evaluator and transcript observer.
//!
//! V3 uses new domain tags.  It cannot reuse the V2 digest verbatim: V2 bound
//! the SWIRL point and asserted an in-circuit table evaluation, whereas V3
//! binds a terminal post-stacking/multi-WHIR completion.  Relation and VK
//! digests remain byte-for-byte the same field digests at the integration
//! boundary.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_recursion_circuit::{
    bus::{ColumnClaimsBus, ColumnClaimsMessage, Poseidon2CompressBus, Poseidon2CompressMessage},
    define_typed_permutation_bus,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::{BasedVectorSpace, PrimeCharacteristicRing},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, Digest, DIGEST_SIZE, D_EF, EF, F,
};

use super::{
    FixedSetupOpeningCertificateBusV2, FixedSetupOpeningCertificateMessageV2,
    SetupPcsAuthorityTransitionStatementBusV3, SetupPcsAuthorityTransitionStatementMessageV3,
    FIXED_SETUP_OPENING_PROTOCOL_V2, VERIFIER_WARP_HISTORY_MAX_TRANSITIONS_V2,
};
use crate::circuit::native_warp_history_v19::{
    SetupPcsSourceProvenanceBusV3, SetupPcsSourceProvenanceMessageV3,
    SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3,
};

pub const SETUP_PCS_AUTHORITY_PROTOCOL_V3: u32 = 3;
pub const SETUP_PCS_AUTHORITY_GLOBAL_START_TAG_V3: u32 = 0x5350_4701;
pub const SETUP_PCS_AUTHORITY_GLOBAL_END_TAG_V3: u32 = 0x5350_4702;
pub const SETUP_PCS_AUTHORITY_CLAIM_START_TAG_V3: u32 = 0x5350_4301;
pub const SETUP_PCS_AUTHORITY_TRANSITION_END_TAG_V3: u32 = 0x5350_4302;

/// Terminal authority completion.  This message is emitted by a real
/// circuit-side stacked-reduction/multi-WHIR verifier, never by this bridge.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct SetupPcsAuthorityCompletionMessageV3<T> {
    pub protocol_version: T,
    pub profile_digest: [T; DIGEST_SIZE],
    pub batch_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub aggregation_vk_digest: [T; DIGEST_SIZE],
    pub transition_count: [T; 2],
    pub whir_completion_binding: [T; DIGEST_SIZE],
    pub global_binding_digest: [T; DIGEST_SIZE],
}
define_typed_permutation_bus!(
    SetupPcsAuthorityCompletionBusV3,
    SetupPcsAuthorityCompletionMessageV3
);

/// Per-transition certificate released by the same terminal verifier only
/// after the global authority proof has completed.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct SetupPcsAuthorityTransitionCertificateMessageV3<T> {
    pub protocol_version: T,
    pub profile_digest: [T; DIGEST_SIZE],
    pub batch_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub aggregation_vk_digest: [T; DIGEST_SIZE],
    pub transition_index: [T; 2],
    pub transition_count: [T; 2],
    pub claim_count: [T; 2],
    pub canonical_claims_digest: [T; DIGEST_SIZE],
    pub transition_statement_digest: [T; DIGEST_SIZE],
    pub setup_openings_digest: [T; DIGEST_SIZE],
    pub global_binding_digest: [T; DIGEST_SIZE],
}
define_typed_permutation_bus!(
    SetupPcsAuthorityTransitionCertificateBusV3,
    SetupPcsAuthorityTransitionCertificateMessageV3
);

/// Value-free identity of one setup-owned SWIRL column claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetupPcsAuthorityClaimIdentityV3 {
    pub transition_index: u32,
    pub setup_index: u32,
    pub air_id: u32,
    pub sort_idx: u32,
    pub part_idx: u32,
    pub col_idx: u32,
    pub need_rot: bool,
    /// `true` exactly for a cached-main setup column that is also present in
    /// the mapped WARP source message. Preprocessed columns are `false`.
    pub is_cached_main: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SetupPcsAuthorityClaimPlanV3 {
    identity: SetupPcsAuthorityClaimIdentityV3,
    claim_ordinal: u32,
    claim_count: u32,
    is_transition_first: bool,
    is_transition_last: bool,
    is_global_first: bool,
    is_global_last: bool,
}

/// VK-owned bridge profile.  It contains identities and ordering only; no
/// setup matrix row or column value can enter this object.
#[derive(Clone, Debug)]
pub struct SetupPcsAuthorityClaimBridgeProfileV3 {
    authority_profile_digest: Digest,
    relation_digest: Digest,
    aggregation_vk_digest: Digest,
    transition_count: u32,
    claims: Arc<[SetupPcsAuthorityClaimPlanV3]>,
    source_provenance_bus: Option<SetupPcsSourceProvenanceBusV3>,
    transition_statement_bus: Option<SetupPcsAuthorityTransitionStatementBusV3>,
    authority_statement_column_claims_bus: Option<ColumnClaimsBus>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetupPcsAuthorityClaimBridgeErrorV3 {
    ZeroDigest,
    TransitionCount,
    EmptyClaims,
    TransitionGap(u32),
    ClaimOrder(usize),
    DuplicateClaim(usize),
    ColumnGap(usize),
    CountOverflow,
    RecordCount,
    RecordTransition(usize),
    RecordClaimCount(u32),
    RecordClaimIdentity(u32, usize),
    RecordRotation(u32, usize),
    CompletionBinding,
    CertificateBinding(u32),
    CanonicalClaimsDigest(u32),
    SetupOpeningsDigest(u32),
    MissingSourceProvenance(u32),
    SourceProvenanceBinding(u32),
    MissingProductionBinding,
}

fn digest_is_zero(value: &Digest) -> bool {
    value.iter().all(|&limb| limb == F::ZERO)
}

fn split_u32(value: u32) -> [F; 2] {
    [F::from_u16(value as u16), F::from_u16((value >> 16) as u16)]
}

fn identity_order_key(
    identity: SetupPcsAuthorityClaimIdentityV3,
) -> (u32, u32, u32, u32, u32, u32) {
    (
        identity.transition_index,
        identity.sort_idx,
        identity.part_idx,
        identity.setup_index,
        identity.air_id,
        identity.col_idx,
    )
}

impl SetupPcsAuthorityClaimBridgeProfileV3 {
    pub fn new(
        authority_profile_digest: Digest,
        relation_digest: Digest,
        aggregation_vk_digest: Digest,
        transition_count: usize,
        identities: Vec<SetupPcsAuthorityClaimIdentityV3>,
    ) -> Result<Self, SetupPcsAuthorityClaimBridgeErrorV3> {
        if digest_is_zero(&authority_profile_digest)
            || digest_is_zero(&relation_digest)
            || digest_is_zero(&aggregation_vk_digest)
        {
            return Err(SetupPcsAuthorityClaimBridgeErrorV3::ZeroDigest);
        }
        if transition_count == 0 || transition_count > VERIFIER_WARP_HISTORY_MAX_TRANSITIONS_V2 {
            return Err(SetupPcsAuthorityClaimBridgeErrorV3::TransitionCount);
        }
        if identities.is_empty() {
            return Err(SetupPcsAuthorityClaimBridgeErrorV3::EmptyClaims);
        }
        let transition_count = u32::try_from(transition_count)
            .map_err(|_| SetupPcsAuthorityClaimBridgeErrorV3::CountOverflow)?;

        let mut counts = vec![0u32; transition_count as usize];
        let mut previous = None;
        let mut previous_group: Option<(u32, u32, u32, u32, u32)> = None;
        let mut previous_column: Option<u32> = None;
        for (index, &identity) in identities.iter().enumerate() {
            if identity.transition_index >= transition_count {
                return Err(SetupPcsAuthorityClaimBridgeErrorV3::TransitionGap(
                    identity.transition_index,
                ));
            }
            let key = identity_order_key(identity);
            if previous.is_some_and(|old| old > key) {
                return Err(SetupPcsAuthorityClaimBridgeErrorV3::ClaimOrder(index));
            }
            if previous == Some(key) {
                return Err(SetupPcsAuthorityClaimBridgeErrorV3::DuplicateClaim(index));
            }
            previous = Some(key);
            let group = (
                identity.transition_index,
                identity.sort_idx,
                identity.part_idx,
                identity.setup_index,
                identity.air_id,
            );
            if previous_group == Some(group) {
                if previous_column.and_then(|column| column.checked_add(1))
                    != Some(identity.col_idx)
                {
                    return Err(SetupPcsAuthorityClaimBridgeErrorV3::ColumnGap(index));
                }
            } else if identity.col_idx != 0 {
                return Err(SetupPcsAuthorityClaimBridgeErrorV3::ColumnGap(index));
            }
            previous_group = Some(group);
            previous_column = Some(identity.col_idx);
            counts[identity.transition_index as usize] = counts[identity.transition_index as usize]
                .checked_add(1)
                .ok_or(SetupPcsAuthorityClaimBridgeErrorV3::CountOverflow)?;
        }
        if let Some(missing) = counts.iter().position(|&count| count == 0) {
            return Err(SetupPcsAuthorityClaimBridgeErrorV3::TransitionGap(
                missing as u32,
            ));
        }

        let mut ordinals = vec![0u32; transition_count as usize];
        let last_index = identities.len() - 1;
        let claims = identities
            .into_iter()
            .enumerate()
            .map(|(index, identity)| {
                let transition = identity.transition_index as usize;
                let ordinal = ordinals[transition];
                ordinals[transition] += 1;
                SetupPcsAuthorityClaimPlanV3 {
                    identity,
                    claim_ordinal: ordinal,
                    claim_count: counts[transition],
                    is_transition_first: ordinal == 0,
                    is_transition_last: ordinal + 1 == counts[transition],
                    is_global_first: index == 0,
                    is_global_last: index == last_index,
                }
            })
            .collect::<Vec<_>>();
        Ok(Self {
            authority_profile_digest,
            relation_digest,
            aggregation_vk_digest,
            transition_count,
            claims: claims.into(),
            source_provenance_bus: None,
            transition_statement_bus: None,
            authority_statement_column_claims_bus: None,
        })
    }

    #[must_use]
    pub fn transition_count(&self) -> usize {
        self.transition_count as usize
    }

    #[must_use]
    pub fn claim_count(&self) -> usize {
        self.claims.len()
    }

    #[must_use]
    pub fn authority_profile_digest(&self) -> Digest {
        self.authority_profile_digest
    }

    pub fn claim_schedule(
        &self,
    ) -> impl ExactSizeIterator<Item = SetupPcsAuthorityClaimIdentityV3> + '_ {
        self.claims.iter().map(|claim| claim.identity)
    }

    /// Require the bridge to consume the exact constrained source receipt and
    /// a transition statement derived by the authority statement AIR.  These
    /// two buses must be enabled together; the source receipt alone does not
    /// contain the independent setup PLE point or ordered setup claims.
    #[must_use]
    pub fn with_constrained_source_provenance_v3(
        mut self,
        source_provenance_bus: SetupPcsSourceProvenanceBusV3,
        transition_statement_bus: SetupPcsAuthorityTransitionStatementBusV3,
    ) -> Self {
        self.source_provenance_bus = Some(source_provenance_bus);
        self.transition_statement_bus = Some(transition_statement_bus);
        self
    }

    /// Send one additional exact claim copy to the authority statement AIR.
    /// The existing source verifier still receives exactly two copies and the
    /// isolated stacking verifier remains a producer checked by a receive.
    #[must_use]
    pub fn with_authority_statement_column_claims_bus_v3(mut self, bus: ColumnClaimsBus) -> Self {
        self.authority_statement_column_claims_bus = Some(bus);
        self
    }

    #[must_use]
    pub fn constrained_source_provenance_enabled_v3(&self) -> bool {
        self.source_provenance_bus.is_some() && self.transition_statement_bus.is_some()
    }

    /// Construct the fail-closed production profile in one operation.
    ///
    /// The more permissive builder methods remain available for isolated differential tests, but
    /// production cannot obtain this profile without all three authority inputs: constrained
    /// source provenance, the exact transition-statement digest, and a separate copy of every
    /// setup claim consumed by the statement/transcript owner.
    #[allow(clippy::too_many_arguments)]
    pub fn new_production_v3(
        authority_profile_digest: Digest,
        relation_digest: Digest,
        aggregation_vk_digest: Digest,
        transition_count: usize,
        identities: Vec<SetupPcsAuthorityClaimIdentityV3>,
        source_provenance_bus: SetupPcsSourceProvenanceBusV3,
        transition_statement_bus: SetupPcsAuthorityTransitionStatementBusV3,
        authority_statement_column_claims_bus: ColumnClaimsBus,
    ) -> Result<Self, SetupPcsAuthorityClaimBridgeErrorV3> {
        Ok(Self::new(
            authority_profile_digest,
            relation_digest,
            aggregation_vk_digest,
            transition_count,
            identities,
        )?
        .with_constrained_source_provenance_v3(source_provenance_bus, transition_statement_bus)
        .with_authority_statement_column_claims_bus_v3(authority_statement_column_claims_bus))
    }

    pub fn validate_production_v3(&self) -> Result<(), SetupPcsAuthorityClaimBridgeErrorV3> {
        if !self.constrained_source_provenance_enabled_v3()
            || self.authority_statement_column_claims_bus.is_none()
        {
            return Err(SetupPcsAuthorityClaimBridgeErrorV3::MissingProductionBinding);
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct SetupPcsAuthorityClaimBridgePrepColsV3<T> {
    pub active: T,
    pub is_transition_first: T,
    pub is_transition_last: T,
    pub is_global_first: T,
    pub is_global_last: T,
    pub transition_index: [T; 2],
    pub claim_ordinal: [T; 2],
    pub claim_count: [T; 2],
    pub setup_index: [T; 2],
    pub air_id: [T; 2],
    pub sort_idx: [T; 2],
    pub part_idx: [T; 2],
    pub col_idx: [T; 2],
    pub need_rot: T,
    pub is_cached_main: T,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct SetupPcsAuthorityClaimBridgeColsV3<T> {
    pub active: T,
    pub current: [T; D_EF],
    pub rotated: [T; D_EF],
    pub batch_digest: [T; DIGEST_SIZE],
    pub global_binding_digest: [T; DIGEST_SIZE],
    pub transition_statement_digest: [T; DIGEST_SIZE],
    pub source_provenance: SetupPcsSourceProvenanceMessageV3<T>,
    pub canonical_before: [T; DIGEST_SIZE],
    pub canonical_identity_header: [T; DIGEST_SIZE],
    pub canonical_identity_body: [T; DIGEST_SIZE],
    pub canonical_after: [T; DIGEST_SIZE],
    pub opening_before: [T; DIGEST_SIZE],
    pub opening_identity: [T; DIGEST_SIZE],
    pub opening_values: [T; DIGEST_SIZE],
    pub terminal_statement: [T; DIGEST_SIZE],
    pub terminal_relation: [T; DIGEST_SIZE],
    pub terminal_vk: [T; DIGEST_SIZE],
    pub terminal_global: [T; DIGEST_SIZE],
    pub terminal_end: [T; DIGEST_SIZE],
    pub completion_hash_header: [T; DIGEST_SIZE],
    pub completion_hash_profile: [T; DIGEST_SIZE],
    pub completion_hash_batch: [T; DIGEST_SIZE],
    pub completion_hash_relation: [T; DIGEST_SIZE],
    pub completion_hash_vk: [T; DIGEST_SIZE],
    pub completion_hash_whir: [T; DIGEST_SIZE],
    pub completion_hash_end: [T; DIGEST_SIZE],
    pub completion: SetupPcsAuthorityCompletionMessageV3<T>,
    pub certificate: SetupPcsAuthorityTransitionCertificateMessageV3<T>,
}

#[derive(Clone, Debug)]
pub struct SetupPcsAuthorityClaimBridgeAirV3 {
    pub profile: SetupPcsAuthorityClaimBridgeProfileV3,
    pub completion_bus: SetupPcsAuthorityCompletionBusV3,
    pub transition_certificate_bus: SetupPcsAuthorityTransitionCertificateBusV3,
    /// Narrow, temporary projection consumed by the existing History producer
    /// bridge. Authority still comes exclusively from the V3 terminal
    /// verifier: this bus carries only fields already constrained in
    /// `certificate` and never evaluates a setup table.
    pub history_certificate_bus: Option<FixedSetupOpeningCertificateBusV2>,
    /// Isolated claim bus produced by the authority StackingModule.
    ///
    /// The source verifier's `column_claims_bus` already receives two copies below (symbolic
    /// evaluation and transcript observation).  Stacking's OpeningClaimsAir is itself a producer,
    /// so sharing that bus would create excess multiplicity.  This second bus makes the bridge a
    /// receiver and thereby enforces equality between the genuine stacking proof claims and the
    /// exact fixed claims consumed by the source verifier.
    pub stacking_claims_bus: Option<ColumnClaimsBus>,
    pub column_claims_bus: ColumnClaimsBus,
    pub compress_bus: Poseidon2CompressBus,
}

impl SetupPcsAuthorityClaimBridgeAirV3 {
    /// Fail-closed production constructor.  In particular, the ordinary ordered-stacking claim
    /// stream is mandatory here; omitting it would leave the terminal PCS proof disconnected from
    /// the exact setup claims replayed by the source verifier.
    #[allow(clippy::too_many_arguments)]
    pub fn new_production_v3(
        profile: SetupPcsAuthorityClaimBridgeProfileV3,
        completion_bus: SetupPcsAuthorityCompletionBusV3,
        transition_certificate_bus: SetupPcsAuthorityTransitionCertificateBusV3,
        history_certificate_bus: FixedSetupOpeningCertificateBusV2,
        stacking_claims_bus: ColumnClaimsBus,
        column_claims_bus: ColumnClaimsBus,
        compress_bus: Poseidon2CompressBus,
    ) -> Result<Self, SetupPcsAuthorityClaimBridgeErrorV3> {
        profile.validate_production_v3()?;
        Ok(Self {
            profile,
            completion_bus,
            transition_certificate_bus,
            history_certificate_bus: Some(history_certificate_bus),
            stacking_claims_bus: Some(stacking_claims_bus),
            column_claims_bus,
            compress_bus,
        })
    }
}

impl BaseAir<F> for SetupPcsAuthorityClaimBridgeAirV3 {
    fn width(&self) -> usize {
        core::mem::size_of::<SetupPcsAuthorityClaimBridgeColsV3<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        let width = core::mem::size_of::<SetupPcsAuthorityClaimBridgePrepColsV3<u8>>();
        let height = self.profile.claims.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(width * height);
        for (index, plan) in self.profile.claims.iter().enumerate() {
            let cols: &mut SetupPcsAuthorityClaimBridgePrepColsV3<F> =
                values[index * width..(index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.is_transition_first = F::from_bool(plan.is_transition_first);
            cols.is_transition_last = F::from_bool(plan.is_transition_last);
            cols.is_global_first = F::from_bool(plan.is_global_first);
            cols.is_global_last = F::from_bool(plan.is_global_last);
            cols.transition_index = split_u32(plan.identity.transition_index);
            cols.claim_ordinal = split_u32(plan.claim_ordinal);
            cols.claim_count = split_u32(plan.claim_count);
            cols.setup_index = split_u32(plan.identity.setup_index);
            cols.air_id = split_u32(plan.identity.air_id);
            cols.sort_idx = split_u32(plan.identity.sort_idx);
            cols.part_idx = split_u32(plan.identity.part_idx);
            cols.col_idx = split_u32(plan.identity.col_idx);
            cols.need_rot = F::from_bool(plan.identity.need_rot);
            cols.is_cached_main = F::from_bool(plan.identity.is_cached_main);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for SetupPcsAuthorityClaimBridgeAirV3 {}
impl PartitionedBaseAir<F> for SetupPcsAuthorityClaimBridgeAirV3 {}

fn value_from_limbs<FA: PrimeCharacteristicRing>(limbs: [impl Into<FA>; 2]) -> FA {
    let [low, high] = limbs.map(Into::into);
    low + high * FA::from_u32(1 << 16)
}

fn assert_array_eq_when<AB, const N: usize>(
    builder: &mut AB,
    enabled: impl Into<AB::Expr> + Clone,
    actual: [AB::Var; N],
    expected: [impl Into<AB::Expr>; N],
) where
    AB: AirBuilder<F = F>,
    AB::Var: Copy,
{
    for (actual, expected) in actual.into_iter().zip(expected) {
        builder
            .when(enabled.clone())
            .assert_eq(actual, expected.into());
    }
}

fn compress_lookup<AB>(
    bus: Poseidon2CompressBus,
    builder: &mut AB,
    left: [impl Into<AB::Expr> + Clone; DIGEST_SIZE],
    right: [impl Into<AB::Expr> + Clone; DIGEST_SIZE],
    output: [impl Into<AB::Expr> + Clone; DIGEST_SIZE],
    enabled: impl Into<AB::Expr>,
) where
    AB: InteractionBuilder,
{
    let left = left.map(Into::into);
    let right = right.map(Into::into);
    bus.lookup_key(
        builder,
        Poseidon2CompressMessage {
            input: core::array::from_fn(|index| {
                if index < DIGEST_SIZE {
                    left[index].clone()
                } else {
                    right[index - DIGEST_SIZE].clone()
                }
            }),
            output: output.map(Into::into),
        },
        enabled,
    );
}

fn zero_block<FA: PrimeCharacteristicRing>() -> [FA; DIGEST_SIZE] {
    core::array::from_fn(|_| FA::ZERO)
}

fn constant_digest_expr<AB: AirBuilder<F = F>>(digest: Digest) -> [AB::Expr; DIGEST_SIZE] {
    digest.map(AB::Expr::from)
}

impl<AB> Air<AB> for SetupPcsAuthorityClaimBridgeAirV3
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let prep_matrix = builder.preprocessed();
        let prep_row = prep_matrix.row_slice(0).expect("setup authority prep row");
        let prep_next_row = prep_matrix
            .row_slice(1)
            .expect("setup authority next prep row");
        let prep: &SetupPcsAuthorityClaimBridgePrepColsV3<AB::Var> = (*prep_row).borrow();
        let prep_next: &SetupPcsAuthorityClaimBridgePrepColsV3<AB::Var> = (*prep_next_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("setup authority bridge row");
        let next_row = main.row_slice(1).expect("setup authority bridge next row");
        let local: &SetupPcsAuthorityClaimBridgeColsV3<AB::Var> = (*row).borrow();
        let next: &SetupPcsAuthorityClaimBridgeColsV3<AB::Var> = (*next_row).borrow();

        for bit in [
            prep.active,
            prep.is_transition_first,
            prep.is_transition_last,
            prep.is_global_first,
            prep.is_global_last,
            prep.need_rot,
            prep.is_cached_main,
            local.active,
        ] {
            builder.assert_bool(bit);
        }
        builder.assert_eq(local.active, prep.active);
        let enabled = AB::Expr::from(prep.active);
        let transition_first = AB::Expr::from(prep.is_transition_first);
        let transition_last = AB::Expr::from(prep.is_transition_last);
        let global_first = AB::Expr::from(prep.is_global_first);
        let need_rot = AB::Expr::from(prep.need_rot);
        let cached_main = AB::Expr::from(prep.is_cached_main);

        // Padding rows contain no witness-carried authority.
        for &cell in row.iter() {
            builder
                .when(AB::Expr::ONE - enabled.clone())
                .assert_zero(cell);
        }
        for limb in local.rotated {
            builder
                .when(enabled.clone() * (AB::Expr::ONE - need_rot.clone()))
                .assert_zero(limb);
        }

        // The global authority binding and batch digest are carried through
        // every active row, while a transition statement is carried through
        // exactly its own canonical claim range.
        let next_active = AB::Expr::from(prep_next.active);
        assert_array_eq_when(
            &mut builder.when_transition(),
            enabled.clone() * next_active.clone(),
            next.global_binding_digest,
            local.global_binding_digest.map(Into::into),
        );
        assert_array_eq_when(
            &mut builder.when_transition(),
            enabled.clone() * next_active.clone(),
            next.batch_digest,
            local.batch_digest.map(Into::into),
        );
        assert_array_eq_when(
            &mut builder.when_transition(),
            enabled.clone() * (AB::Expr::ONE - transition_last.clone()),
            next.transition_statement_digest,
            local.transition_statement_digest.map(Into::into),
        );
        assert_array_eq_when(
            builder,
            transition_first.clone(),
            local.canonical_before,
            zero_block::<AB::Expr>(),
        );
        assert_array_eq_when(
            builder,
            transition_first.clone(),
            local.opening_before,
            zero_block::<AB::Expr>(),
        );

        if let (Some(provenance_bus), Some(statement_bus)) = (
            self.profile.source_provenance_bus,
            self.profile.transition_statement_bus,
        ) {
            builder.when(transition_first.clone()).assert_eq(
                local.source_provenance.protocol_version,
                AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            );
            assert_array_eq_when(
                builder,
                transition_first.clone(),
                local.source_provenance.transition_index,
                prep.transition_index.map(Into::into),
            );
            assert_array_eq_when(
                builder,
                transition_first.clone(),
                local.source_provenance.relation_digest,
                self.profile.relation_digest.map(AB::Expr::from),
            );
            assert_array_eq_when(
                builder,
                transition_first.clone(),
                local.source_provenance.app_vk_digest,
                self.profile.aggregation_vk_digest.map(AB::Expr::from),
            );
            provenance_bus.receive(
                builder,
                local.source_provenance.clone(),
                transition_first.clone(),
            );
            statement_bus.receive(
                builder,
                SetupPcsAuthorityTransitionStatementMessageV3 {
                    protocol_version: AB::Expr::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
                    transition_index: prep.transition_index.map(Into::into),
                    source_receipt_digest: local
                        .source_provenance
                        .source_receipt_digest
                        .map(Into::into),
                    transition_statement_digest: local.transition_statement_digest.map(Into::into),
                },
                transition_first.clone(),
            );
        } else {
            // Legacy differential/oracle mode has no constrained provenance
            // witness cells. Production enables both typed buses above.
            for value in local.source_provenance.clone().to_vec() {
                builder.when(enabled.clone()).assert_zero(value);
            }
        }
        for value in local.source_provenance.clone().to_vec() {
            builder
                .when(enabled.clone() * (AB::Expr::ONE - transition_first.clone()))
                .assert_zero(value);
        }
        assert_array_eq_when(
            &mut builder.when_transition(),
            enabled.clone() * (AB::Expr::ONE - transition_last.clone()),
            next.canonical_before,
            local.canonical_after.map(Into::into),
        );
        assert_array_eq_when(
            &mut builder.when_transition(),
            enabled.clone() * (AB::Expr::ONE - transition_last.clone()),
            next.opening_before,
            local.opening_values.map(Into::into),
        );

        // Canonical claim identity.  All u32 values use two u16 limbs so the
        // hash encoding is injective over the complete host type.
        let identity_header: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_CLAIM_START_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
            prep.transition_index[0].into(),
            prep.transition_index[1].into(),
            prep.claim_ordinal[0].into(),
            prep.claim_ordinal[1].into(),
            prep.claim_count[0].into(),
            prep.claim_count[1].into(),
        ];
        let identity_body: [AB::Expr; DIGEST_SIZE] = [
            prep.setup_index[0].into(),
            prep.setup_index[1].into(),
            prep.air_id[0].into(),
            prep.air_id[1].into(),
            prep.sort_idx[0].into(),
            prep.sort_idx[1].into(),
            prep.part_idx[0].into(),
            prep.part_idx[1].into(),
        ];
        let identity_column: [AB::Expr; DIGEST_SIZE] = [
            prep.col_idx[0].into(),
            prep.col_idx[1].into(),
            prep.need_rot.into(),
            // Claim identity is shared with the statement AIR. Whether this
            // setup column is cached-main or preprocessed is already fixed by
            // the authority profile (and its digest); it controls only which
            // side of ColumnClaimsBus this bridge owns.
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        compress_lookup(
            self.compress_bus,
            builder,
            local.canonical_before,
            identity_header,
            local.canonical_identity_header,
            enabled.clone(),
        );
        compress_lookup(
            self.compress_bus,
            builder,
            local.canonical_identity_header,
            identity_body,
            local.canonical_identity_body,
            enabled.clone(),
        );
        compress_lookup(
            self.compress_bus,
            builder,
            local.canonical_identity_body,
            identity_column,
            local.canonical_after,
            enabled.clone(),
        );

        compress_lookup(
            self.compress_bus,
            builder,
            local.opening_before,
            local.canonical_after,
            local.opening_identity,
            enabled.clone(),
        );
        let value_block: [AB::Expr; DIGEST_SIZE] = core::array::from_fn(|index| {
            if index < D_EF {
                AB::Expr::from(local.current[index])
            } else {
                AB::Expr::from(local.rotated[index - D_EF])
            }
        });
        compress_lookup(
            self.compress_bus,
            builder,
            local.opening_identity,
            value_block,
            local.opening_values,
            enabled.clone(),
        );

        let relation = constant_digest_expr::<AB>(self.profile.relation_digest);
        let aggregation_vk = constant_digest_expr::<AB>(self.profile.aggregation_vk_digest);
        compress_lookup(
            self.compress_bus,
            builder,
            local.opening_values,
            local.transition_statement_digest,
            local.terminal_statement,
            transition_last.clone(),
        );
        compress_lookup(
            self.compress_bus,
            builder,
            local.terminal_statement,
            relation.clone(),
            local.terminal_relation,
            transition_last.clone(),
        );
        compress_lookup(
            self.compress_bus,
            builder,
            local.terminal_relation,
            aggregation_vk.clone(),
            local.terminal_vk,
            transition_last.clone(),
        );
        compress_lookup(
            self.compress_bus,
            builder,
            local.terminal_vk,
            local.global_binding_digest,
            local.terminal_global,
            transition_last.clone(),
        );
        let transition_end: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_TRANSITION_END_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
            prep.transition_index[0].into(),
            prep.transition_index[1].into(),
            prep.claim_count[0].into(),
            prep.claim_count[1].into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        compress_lookup(
            self.compress_bus,
            builder,
            local.terminal_global,
            transition_end,
            local.terminal_end,
            transition_last.clone(),
        );

        // The sole completion row recomputes the global binding.  The WHIR
        // completion therefore cannot be dropped while retaining otherwise
        // valid per-transition certificates.
        let global_header: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_GLOBAL_START_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
            F::from_u16(self.profile.transition_count as u16).into(),
            F::from_u16((self.profile.transition_count >> 16) as u16).into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        compress_lookup(
            self.compress_bus,
            builder,
            zero_block::<AB::Expr>(),
            global_header,
            local.completion_hash_header,
            global_first.clone(),
        );
        compress_lookup(
            self.compress_bus,
            builder,
            local.completion_hash_header,
            local.completion.profile_digest,
            local.completion_hash_profile,
            global_first.clone(),
        );
        compress_lookup(
            self.compress_bus,
            builder,
            local.completion_hash_profile,
            local.completion.batch_digest,
            local.completion_hash_batch,
            global_first.clone(),
        );
        compress_lookup(
            self.compress_bus,
            builder,
            local.completion_hash_batch,
            local.completion.relation_digest,
            local.completion_hash_relation,
            global_first.clone(),
        );
        compress_lookup(
            self.compress_bus,
            builder,
            local.completion_hash_relation,
            local.completion.aggregation_vk_digest,
            local.completion_hash_vk,
            global_first.clone(),
        );
        compress_lookup(
            self.compress_bus,
            builder,
            local.completion_hash_vk,
            local.completion.whir_completion_binding,
            local.completion_hash_whir,
            global_first.clone(),
        );
        let global_end: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_GLOBAL_END_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
            F::from_u16(self.profile.transition_count as u16).into(),
            F::from_u16((self.profile.transition_count >> 16) as u16).into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        compress_lookup(
            self.compress_bus,
            builder,
            local.completion_hash_whir,
            global_end,
            local.completion_hash_end,
            global_first.clone(),
        );

        builder.when(global_first.clone()).assert_eq(
            local.completion.protocol_version,
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
        );
        assert_array_eq_when(
            builder,
            global_first.clone(),
            local.completion.profile_digest,
            constant_digest_expr::<AB>(self.profile.authority_profile_digest),
        );
        assert_array_eq_when(
            builder,
            global_first.clone(),
            local.completion.relation_digest,
            relation.clone(),
        );
        assert_array_eq_when(
            builder,
            global_first.clone(),
            local.completion.aggregation_vk_digest,
            aggregation_vk.clone(),
        );
        assert_array_eq_when(
            builder,
            global_first.clone(),
            local.completion.transition_count,
            split_u32(self.profile.transition_count).map(AB::Expr::from),
        );
        assert_array_eq_when(
            builder,
            global_first.clone(),
            local.completion.global_binding_digest,
            local.completion_hash_end.map(Into::into),
        );
        assert_array_eq_when(
            builder,
            global_first.clone(),
            local.global_binding_digest,
            local.completion_hash_end.map(Into::into),
        );
        assert_array_eq_when(
            builder,
            global_first.clone(),
            local.batch_digest,
            local.completion.batch_digest.map(Into::into),
        );
        self.completion_bus
            .receive(builder, local.completion.clone(), global_first.clone());

        // Completion/certificate message cells and their private hash scratch
        // are canonical zero away from their unique rows.
        for value in local.completion.clone().to_vec() {
            builder
                .when(enabled.clone() * (AB::Expr::ONE - global_first.clone()))
                .assert_zero(value);
        }
        for digest in [
            local.completion_hash_header,
            local.completion_hash_profile,
            local.completion_hash_batch,
            local.completion_hash_relation,
            local.completion_hash_vk,
            local.completion_hash_whir,
            local.completion_hash_end,
        ] {
            for limb in digest {
                builder
                    .when(enabled.clone() * (AB::Expr::ONE - global_first.clone()))
                    .assert_zero(limb);
            }
        }
        for digest in [
            local.terminal_statement,
            local.terminal_relation,
            local.terminal_vk,
            local.terminal_global,
            local.terminal_end,
        ] {
            for limb in digest {
                builder
                    .when(enabled.clone() * (AB::Expr::ONE - transition_last.clone()))
                    .assert_zero(limb);
            }
        }

        let certificate = &local.certificate;
        builder.when(transition_last.clone()).assert_eq(
            certificate.protocol_version,
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
        );
        for (actual, expected) in [
            (
                certificate.profile_digest,
                self.profile.authority_profile_digest,
            ),
            (certificate.relation_digest, self.profile.relation_digest),
            (
                certificate.aggregation_vk_digest,
                self.profile.aggregation_vk_digest,
            ),
        ] {
            assert_array_eq_when(
                builder,
                transition_last.clone(),
                actual,
                expected.map(AB::Expr::from),
            );
        }
        for (actual, expected) in [
            (certificate.batch_digest, local.batch_digest),
            (
                certificate.global_binding_digest,
                local.global_binding_digest,
            ),
            (certificate.canonical_claims_digest, local.canonical_after),
            (
                certificate.transition_statement_digest,
                local.transition_statement_digest,
            ),
            (certificate.setup_openings_digest, local.terminal_end),
        ] {
            assert_array_eq_when(
                builder,
                transition_last.clone(),
                actual,
                expected.map(Into::into),
            );
        }
        assert_array_eq_when(
            builder,
            transition_last.clone(),
            certificate.transition_index,
            prep.transition_index.map(Into::into),
        );
        assert_array_eq_when(
            builder,
            transition_last.clone(),
            certificate.transition_count,
            split_u32(self.profile.transition_count).map(AB::Expr::from),
        );
        assert_array_eq_when(
            builder,
            transition_last.clone(),
            certificate.claim_count,
            prep.claim_count.map(Into::into),
        );
        self.transition_certificate_bus.receive(
            builder,
            certificate.clone(),
            transition_last.clone(),
        );
        if let Some(bus) = self.history_certificate_bus {
            bus.send(
                builder,
                FixedSetupOpeningCertificateMessageV2 {
                    protocol_version: AB::Expr::from_u32(FIXED_SETUP_OPENING_PROTOCOL_V2),
                    proof_index: value_from_limbs::<AB::Expr>(prep.transition_index),
                    canonical_claim_count: value_from_limbs::<AB::Expr>(prep.claim_count),
                    source_relation_vk_digest: self
                        .profile
                        .aggregation_vk_digest
                        .map(AB::Expr::from),
                    setup_openings_digest: certificate.setup_openings_digest.map(Into::into),
                },
                transition_last.clone(),
            );
        }
        for value in certificate.clone().to_vec() {
            builder
                .when(enabled.clone() * (AB::Expr::ONE - transition_last.clone()))
                .assert_zero(value);
        }

        let proof_idx = value_from_limbs::<AB::Expr>(prep.transition_index);
        let sort_idx = value_from_limbs::<AB::Expr>(prep.sort_idx);
        let part_idx = value_from_limbs::<AB::Expr>(prep.part_idx);
        let col_idx = value_from_limbs::<AB::Expr>(prep.col_idx);
        let claim_message = |is_rot, claim| ColumnClaimsMessage {
            sort_idx: sort_idx.clone(),
            part_idx: part_idx.clone(),
            col_idx: col_idx.clone(),
            claim,
            is_rot,
        };
        let preprocessed = enabled.clone() * (AB::Expr::ONE - cached_main.clone());
        let cached = enabled.clone() * cached_main;
        self.column_claims_bus.send(
            builder,
            proof_idx.clone(),
            claim_message(AB::Expr::ZERO, local.current.map(Into::into)),
            preprocessed.clone(),
        );
        self.column_claims_bus.send(
            builder,
            proof_idx.clone(),
            claim_message(AB::Expr::ONE, local.rotated.map(Into::into)),
            preprocessed * need_rot.clone(),
        );
        self.column_claims_bus.receive(
            builder,
            proof_idx.clone(),
            claim_message(AB::Expr::ZERO, local.current.map(Into::into)),
            cached.clone(),
        );
        self.column_claims_bus.receive(
            builder,
            proof_idx.clone(),
            claim_message(AB::Expr::ONE, local.rotated.map(Into::into)),
            cached * need_rot.clone(),
        );
        if let Some(bus) = self.profile.authority_statement_column_claims_bus {
            bus.send(
                builder,
                proof_idx.clone(),
                ColumnClaimsMessage {
                    sort_idx: sort_idx.clone(),
                    part_idx: part_idx.clone(),
                    col_idx: col_idx.clone(),
                    claim: local.current.map(Into::into),
                    is_rot: AB::Expr::ZERO,
                },
                enabled.clone(),
            );
            bus.send(
                builder,
                proof_idx.clone(),
                ColumnClaimsMessage {
                    sort_idx: sort_idx.clone(),
                    part_idx: part_idx.clone(),
                    col_idx: col_idx.clone(),
                    claim: local.rotated.map(Into::into),
                    is_rot: AB::Expr::ONE,
                },
                enabled.clone() * need_rot.clone(),
            );
        }
        if let Some(bus) = self.stacking_claims_bus {
            bus.receive(
                builder,
                proof_idx.clone(),
                ColumnClaimsMessage {
                    sort_idx: sort_idx.clone(),
                    part_idx: part_idx.clone(),
                    col_idx: col_idx.clone(),
                    claim: local.current.map(Into::into),
                    is_rot: AB::Expr::ZERO,
                },
                enabled.clone(),
            );
            bus.receive(
                builder,
                proof_idx,
                ColumnClaimsMessage {
                    sort_idx,
                    part_idx,
                    col_idx,
                    claim: local.rotated.map(Into::into),
                    is_rot: AB::Expr::ONE,
                },
                enabled * need_rot,
            );
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetupPcsAuthorityClaimRecordV3 {
    pub identity: SetupPcsAuthorityClaimIdentityV3,
    pub current: EF,
    pub rotated: Option<EF>,
}

#[derive(Clone, Debug)]
pub struct SetupPcsAuthorityTransitionRecordV3 {
    pub transition_index: u32,
    pub transition_statement_digest: Digest,
    /// Required when `with_constrained_source_provenance_v3` is enabled.
    /// Legacy differential/oracle callers retain `None`.
    pub source_provenance: Option<SetupPcsSourceProvenanceMessageV3<F>>,
    pub claims: Vec<SetupPcsAuthorityClaimRecordV3>,
    /// Must have been emitted by the circuit-side authority verifier.  Trace
    /// generation only copies and cross-checks it; it does not mint it.
    pub certificate: SetupPcsAuthorityTransitionCertificateMessageV3<F>,
}

#[derive(Clone, Debug)]
pub struct SetupPcsAuthorityClaimBridgeRecordV3 {
    /// Must have been emitted by the circuit-side authority verifier.
    pub completion: SetupPcsAuthorityCompletionMessageV3<F>,
    pub transitions: Vec<SetupPcsAuthorityTransitionRecordV3>,
}

#[derive(Debug)]
pub struct SetupPcsAuthorityClaimBridgeTraceV3 {
    pub matrix: RowMajorMatrix<F>,
    pub compression_inputs: Vec<[F; 2 * DIGEST_SIZE]>,
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

fn compress_host(
    compression_inputs: &mut Vec<[F; 2 * DIGEST_SIZE]>,
    left: Digest,
    right: Digest,
) -> Digest {
    let mut input = [F::ZERO; 2 * DIGEST_SIZE];
    input[..DIGEST_SIZE].copy_from_slice(&left);
    input[DIGEST_SIZE..].copy_from_slice(&right);
    compression_inputs.push(input);
    poseidon2_compress_with_capacity(left, right).0
}

fn claim_identity_blocks(plan: SetupPcsAuthorityClaimPlanV3) -> [Digest; 3] {
    let transition = split_u32(plan.identity.transition_index);
    let ordinal = split_u32(plan.claim_ordinal);
    let count = split_u32(plan.claim_count);
    let setup = split_u32(plan.identity.setup_index);
    let air = split_u32(plan.identity.air_id);
    let sort = split_u32(plan.identity.sort_idx);
    let part = split_u32(plan.identity.part_idx);
    let column = split_u32(plan.identity.col_idx);
    [
        [
            F::from_u32(SETUP_PCS_AUTHORITY_CLAIM_START_TAG_V3),
            F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
            transition[0],
            transition[1],
            ordinal[0],
            ordinal[1],
            count[0],
            count[1],
        ],
        [
            setup[0], setup[1], air[0], air[1], sort[0], sort[1], part[0], part[1],
        ],
        [
            column[0],
            column[1],
            F::from_bool(plan.identity.need_rot),
            // Keep this byte-for-byte identical to the statement AIR's
            // canonical column-claim identity. The cached-main role is
            // setup-profile metadata, not a second identity encoding.
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
        ],
    ]
}

fn global_binding_host(
    profile: &SetupPcsAuthorityClaimBridgeProfileV3,
    completion: &SetupPcsAuthorityCompletionMessageV3<F>,
    compression_inputs: &mut Vec<[F; 2 * DIGEST_SIZE]>,
) -> Digest {
    let count = split_u32(profile.transition_count);
    let header = [
        F::from_u32(SETUP_PCS_AUTHORITY_GLOBAL_START_TAG_V3),
        F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
        count[0],
        count[1],
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
    ];
    let end = [
        F::from_u32(SETUP_PCS_AUTHORITY_GLOBAL_END_TAG_V3),
        F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
        count[0],
        count[1],
        F::ZERO,
        F::ZERO,
        F::ZERO,
        F::ZERO,
    ];
    let mut state = compress_host(compression_inputs, [F::ZERO; DIGEST_SIZE], header);
    for block in [
        completion.profile_digest,
        completion.batch_digest,
        completion.relation_digest,
        completion.aggregation_vk_digest,
        completion.whir_completion_binding,
        end,
    ] {
        state = compress_host(compression_inputs, state, block);
    }
    state
}

fn validate_completion(
    profile: &SetupPcsAuthorityClaimBridgeProfileV3,
    completion: &SetupPcsAuthorityCompletionMessageV3<F>,
    compression_inputs: &mut Vec<[F; 2 * DIGEST_SIZE]>,
) -> Result<Digest, SetupPcsAuthorityClaimBridgeErrorV3> {
    if completion.protocol_version != F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3)
        || completion.profile_digest != profile.authority_profile_digest
        || completion.relation_digest != profile.relation_digest
        || completion.aggregation_vk_digest != profile.aggregation_vk_digest
        || completion.transition_count != split_u32(profile.transition_count)
        || digest_is_zero(&completion.batch_digest)
        || digest_is_zero(&completion.whir_completion_binding)
    {
        return Err(SetupPcsAuthorityClaimBridgeErrorV3::CompletionBinding);
    }
    let expected = global_binding_host(profile, completion, compression_inputs);
    if completion.global_binding_digest != expected {
        return Err(SetupPcsAuthorityClaimBridgeErrorV3::CompletionBinding);
    }
    Ok(expected)
}

pub fn generate_setup_pcs_authority_claim_bridge_trace_v3(
    air: &SetupPcsAuthorityClaimBridgeAirV3,
    record: &SetupPcsAuthorityClaimBridgeRecordV3,
) -> Result<SetupPcsAuthorityClaimBridgeTraceV3, SetupPcsAuthorityClaimBridgeErrorV3> {
    if record.transitions.len() != air.profile.transition_count() {
        return Err(SetupPcsAuthorityClaimBridgeErrorV3::RecordCount);
    }
    let width = core::mem::size_of::<SetupPcsAuthorityClaimBridgeColsV3<u8>>();
    let height = air.profile.claims.len().next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    let mut compression_inputs = Vec::new();
    let global_binding =
        validate_completion(&air.profile, &record.completion, &mut compression_inputs)?;

    let mut claim_cursor = 0usize;
    for (expected_transition, transition) in record.transitions.iter().enumerate() {
        let expected_transition = expected_transition as u32;
        if transition.transition_index != expected_transition {
            return Err(SetupPcsAuthorityClaimBridgeErrorV3::RecordTransition(
                expected_transition as usize,
            ));
        }
        let plans = air
            .profile
            .claims
            .iter()
            .copied()
            .filter(|plan| plan.identity.transition_index == expected_transition)
            .collect::<Vec<_>>();
        if transition.claims.len() != plans.len() {
            return Err(SetupPcsAuthorityClaimBridgeErrorV3::RecordClaimCount(
                expected_transition,
            ));
        }
        let mut canonical_state = [F::ZERO; DIGEST_SIZE];
        let mut opening_state = [F::ZERO; DIGEST_SIZE];
        for (claim_index, (plan, claim)) in
            plans.iter().copied().zip(&transition.claims).enumerate()
        {
            if claim.identity != plan.identity {
                return Err(SetupPcsAuthorityClaimBridgeErrorV3::RecordClaimIdentity(
                    expected_transition,
                    claim_index,
                ));
            }
            if claim.rotated.is_some() != plan.identity.need_rot {
                return Err(SetupPcsAuthorityClaimBridgeErrorV3::RecordRotation(
                    expected_transition,
                    claim_index,
                ));
            }
            let cols: &mut SetupPcsAuthorityClaimBridgeColsV3<F> =
                values[claim_cursor * width..(claim_cursor + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.batch_digest = record.completion.batch_digest;
            cols.global_binding_digest = global_binding;
            cols.transition_statement_digest = transition.transition_statement_digest;
            if plan.is_transition_first {
                match (
                    air.profile.constrained_source_provenance_enabled_v3(),
                    transition.source_provenance.as_ref(),
                ) {
                    (true, Some(provenance)) => {
                        let expected_index = split_u32(expected_transition);
                        if provenance.protocol_version
                            != F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3)
                            || provenance.transition_index != expected_index
                            || provenance.relation_digest != air.profile.relation_digest
                            || provenance.app_vk_digest != air.profile.aggregation_vk_digest
                        {
                            return Err(
                                SetupPcsAuthorityClaimBridgeErrorV3::SourceProvenanceBinding(
                                    expected_transition,
                                ),
                            );
                        }
                        cols.source_provenance = provenance.clone();
                    }
                    (true, None) => {
                        return Err(
                            SetupPcsAuthorityClaimBridgeErrorV3::MissingSourceProvenance(
                                expected_transition,
                            ),
                        );
                    }
                    (false, None) => {}
                    (false, Some(_)) => {
                        return Err(
                            SetupPcsAuthorityClaimBridgeErrorV3::SourceProvenanceBinding(
                                expected_transition,
                            ),
                        );
                    }
                }
            }
            copy_ext(&mut cols.current, claim.current);
            copy_ext(&mut cols.rotated, claim.rotated.unwrap_or(EF::ZERO));

            cols.canonical_before = canonical_state;
            let [header, body, column] = claim_identity_blocks(plan);
            cols.canonical_identity_header =
                compress_host(&mut compression_inputs, canonical_state, header);
            cols.canonical_identity_body = compress_host(
                &mut compression_inputs,
                cols.canonical_identity_header,
                body,
            );
            cols.canonical_after = compress_host(
                &mut compression_inputs,
                cols.canonical_identity_body,
                column,
            );
            canonical_state = cols.canonical_after;

            cols.opening_before = opening_state;
            cols.opening_identity =
                compress_host(&mut compression_inputs, opening_state, canonical_state);
            let mut value_block = [F::ZERO; DIGEST_SIZE];
            value_block[..D_EF].copy_from_slice(&cols.current);
            value_block[D_EF..].copy_from_slice(&cols.rotated);
            cols.opening_values =
                compress_host(&mut compression_inputs, cols.opening_identity, value_block);
            opening_state = cols.opening_values;

            if plan.is_global_first {
                cols.completion = record.completion.clone();
                // Replay the already-recorded global chain outputs in row order.
                let count = split_u32(air.profile.transition_count);
                let header = [
                    F::from_u32(SETUP_PCS_AUTHORITY_GLOBAL_START_TAG_V3),
                    F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
                    count[0],
                    count[1],
                    F::ZERO,
                    F::ZERO,
                    F::ZERO,
                    F::ZERO,
                ];
                cols.completion_hash_header =
                    poseidon2_compress_with_capacity([F::ZERO; DIGEST_SIZE], header).0;
                cols.completion_hash_profile = poseidon2_compress_with_capacity(
                    cols.completion_hash_header,
                    record.completion.profile_digest,
                )
                .0;
                cols.completion_hash_batch = poseidon2_compress_with_capacity(
                    cols.completion_hash_profile,
                    record.completion.batch_digest,
                )
                .0;
                cols.completion_hash_relation = poseidon2_compress_with_capacity(
                    cols.completion_hash_batch,
                    record.completion.relation_digest,
                )
                .0;
                cols.completion_hash_vk = poseidon2_compress_with_capacity(
                    cols.completion_hash_relation,
                    record.completion.aggregation_vk_digest,
                )
                .0;
                cols.completion_hash_whir = poseidon2_compress_with_capacity(
                    cols.completion_hash_vk,
                    record.completion.whir_completion_binding,
                )
                .0;
                let end = [
                    F::from_u32(SETUP_PCS_AUTHORITY_GLOBAL_END_TAG_V3),
                    F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
                    count[0],
                    count[1],
                    F::ZERO,
                    F::ZERO,
                    F::ZERO,
                    F::ZERO,
                ];
                cols.completion_hash_end =
                    poseidon2_compress_with_capacity(cols.completion_hash_whir, end).0;
            }

            if plan.is_transition_last {
                cols.terminal_statement = compress_host(
                    &mut compression_inputs,
                    opening_state,
                    transition.transition_statement_digest,
                );
                cols.terminal_relation = compress_host(
                    &mut compression_inputs,
                    cols.terminal_statement,
                    air.profile.relation_digest,
                );
                cols.terminal_vk = compress_host(
                    &mut compression_inputs,
                    cols.terminal_relation,
                    air.profile.aggregation_vk_digest,
                );
                cols.terminal_global =
                    compress_host(&mut compression_inputs, cols.terminal_vk, global_binding);
                let transition_limbs = split_u32(expected_transition);
                let count_limbs = split_u32(plan.claim_count);
                let end = [
                    F::from_u32(SETUP_PCS_AUTHORITY_TRANSITION_END_TAG_V3),
                    F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
                    transition_limbs[0],
                    transition_limbs[1],
                    count_limbs[0],
                    count_limbs[1],
                    F::ZERO,
                    F::ZERO,
                ];
                cols.terminal_end =
                    compress_host(&mut compression_inputs, cols.terminal_global, end);
                let certificate = &transition.certificate;
                if certificate.protocol_version != F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3)
                    || certificate.profile_digest != air.profile.authority_profile_digest
                    || certificate.batch_digest != record.completion.batch_digest
                    || certificate.relation_digest != air.profile.relation_digest
                    || certificate.aggregation_vk_digest != air.profile.aggregation_vk_digest
                    || certificate.transition_index != transition_limbs
                    || certificate.transition_count != split_u32(air.profile.transition_count)
                    || certificate.claim_count != count_limbs
                    || certificate.transition_statement_digest
                        != transition.transition_statement_digest
                    || certificate.global_binding_digest != global_binding
                {
                    return Err(SetupPcsAuthorityClaimBridgeErrorV3::CertificateBinding(
                        expected_transition,
                    ));
                }
                if certificate.canonical_claims_digest != canonical_state {
                    return Err(SetupPcsAuthorityClaimBridgeErrorV3::CanonicalClaimsDigest(
                        expected_transition,
                    ));
                }
                if certificate.setup_openings_digest != cols.terminal_end {
                    return Err(SetupPcsAuthorityClaimBridgeErrorV3::SetupOpeningsDigest(
                        expected_transition,
                    ));
                }
                cols.certificate = certificate.clone();
            }
            claim_cursor += 1;
        }
    }
    if claim_cursor != air.profile.claims.len() {
        return Err(SetupPcsAuthorityClaimBridgeErrorV3::RecordCount);
    }
    Ok(SetupPcsAuthorityClaimBridgeTraceV3 {
        matrix: RowMajorMatrix::new(values, width),
        compression_inputs,
    })
}

#[cfg(test)]
mod tests {
    use openvm_stark_backend::{
        air_builders::{debug::check_constraints, symbolic::get_symbolic_builder},
        interaction::BusIndex,
        keygen::types::TraceWidth,
        p3_air::BaseAir,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config as NativeSC;

    use super::*;

    fn digest(seed: u32) -> Digest {
        core::array::from_fn(|index| F::from_u32(seed + index as u32 + 1))
    }

    fn profile() -> SetupPcsAuthorityClaimBridgeProfileV3 {
        SetupPcsAuthorityClaimBridgeProfileV3::new(
            digest(10),
            digest(30),
            digest(50),
            2,
            vec![
                SetupPcsAuthorityClaimIdentityV3 {
                    transition_index: 0,
                    setup_index: 0,
                    air_id: 7,
                    sort_idx: 3,
                    part_idx: 1,
                    col_idx: 0,
                    need_rot: true,
                    is_cached_main: false,
                },
                SetupPcsAuthorityClaimIdentityV3 {
                    transition_index: 0,
                    setup_index: 0,
                    air_id: 7,
                    sort_idx: 3,
                    part_idx: 1,
                    col_idx: 1,
                    need_rot: true,
                    is_cached_main: false,
                },
                SetupPcsAuthorityClaimIdentityV3 {
                    transition_index: 1,
                    setup_index: 0,
                    air_id: 7,
                    sort_idx: 3,
                    part_idx: 1,
                    col_idx: 0,
                    need_rot: true,
                    is_cached_main: false,
                },
                SetupPcsAuthorityClaimIdentityV3 {
                    transition_index: 1,
                    setup_index: 0,
                    air_id: 7,
                    sort_idx: 3,
                    part_idx: 1,
                    col_idx: 1,
                    need_rot: true,
                    is_cached_main: false,
                },
            ],
        )
        .unwrap()
    }

    fn air() -> SetupPcsAuthorityClaimBridgeAirV3 {
        SetupPcsAuthorityClaimBridgeAirV3 {
            profile: profile(),
            completion_bus: SetupPcsAuthorityCompletionBusV3::new(BusIndex::from(1u16)),
            transition_certificate_bus: SetupPcsAuthorityTransitionCertificateBusV3::new(
                BusIndex::from(2u16),
            ),
            history_certificate_bus: None,
            stacking_claims_bus: None,
            column_claims_bus: ColumnClaimsBus::new(BusIndex::from(3u16)),
            compress_bus: Poseidon2CompressBus::new(BusIndex::from(4u16)),
        }
    }

    fn valid_record(
        air: &SetupPcsAuthorityClaimBridgeAirV3,
    ) -> SetupPcsAuthorityClaimBridgeRecordV3 {
        let mut completion = SetupPcsAuthorityCompletionMessageV3 {
            protocol_version: F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
            profile_digest: air.profile.authority_profile_digest,
            batch_digest: digest(70),
            relation_digest: air.profile.relation_digest,
            aggregation_vk_digest: air.profile.aggregation_vk_digest,
            transition_count: split_u32(air.profile.transition_count),
            whir_completion_binding: digest(110),
            global_binding_digest: [F::ZERO; DIGEST_SIZE],
        };
        completion.global_binding_digest =
            global_binding_host(&air.profile, &completion, &mut Vec::new());
        let mut transitions = Vec::new();
        for transition_index in 0..air.profile.transition_count {
            let plans = air
                .profile
                .claims
                .iter()
                .copied()
                .filter(|plan| plan.identity.transition_index == transition_index)
                .collect::<Vec<_>>();
            let claims = plans
                .iter()
                .enumerate()
                .map(|(index, plan)| SetupPcsAuthorityClaimRecordV3 {
                    identity: plan.identity,
                    current: EF::from(F::from_u32(200 + transition_index * 10 + index as u32)),
                    rotated: Some(EF::from(F::from_u32(
                        220 + transition_index * 10 + index as u32,
                    ))),
                })
                .collect::<Vec<_>>();
            let statement = digest(130 + transition_index * 10);
            let mut compression_inputs = Vec::new();
            let mut canonical = [F::ZERO; DIGEST_SIZE];
            let mut opening = [F::ZERO; DIGEST_SIZE];
            for (plan, claim) in plans.iter().copied().zip(&claims) {
                for block in claim_identity_blocks(plan) {
                    canonical = compress_host(&mut compression_inputs, canonical, block);
                }
                opening = compress_host(&mut compression_inputs, opening, canonical);
                let mut values = [F::ZERO; DIGEST_SIZE];
                values[..D_EF].copy_from_slice(claim.current.as_basis_coefficients_slice());
                values[D_EF..].copy_from_slice(
                    claim
                        .rotated
                        .unwrap_or(EF::ZERO)
                        .as_basis_coefficients_slice(),
                );
                opening = compress_host(&mut compression_inputs, opening, values);
            }
            let mut terminal = compress_host(&mut compression_inputs, opening, statement);
            terminal = compress_host(
                &mut compression_inputs,
                terminal,
                air.profile.relation_digest,
            );
            terminal = compress_host(
                &mut compression_inputs,
                terminal,
                air.profile.aggregation_vk_digest,
            );
            terminal = compress_host(
                &mut compression_inputs,
                terminal,
                completion.global_binding_digest,
            );
            let transition_limbs = split_u32(transition_index);
            let count_limbs = split_u32(plans.len() as u32);
            terminal = compress_host(
                &mut compression_inputs,
                terminal,
                [
                    F::from_u32(SETUP_PCS_AUTHORITY_TRANSITION_END_TAG_V3),
                    F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
                    transition_limbs[0],
                    transition_limbs[1],
                    count_limbs[0],
                    count_limbs[1],
                    F::ZERO,
                    F::ZERO,
                ],
            );
            transitions.push(SetupPcsAuthorityTransitionRecordV3 {
                transition_index,
                transition_statement_digest: statement,
                source_provenance: None,
                claims,
                certificate: SetupPcsAuthorityTransitionCertificateMessageV3 {
                    protocol_version: F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
                    profile_digest: air.profile.authority_profile_digest,
                    batch_digest: completion.batch_digest,
                    relation_digest: air.profile.relation_digest,
                    aggregation_vk_digest: air.profile.aggregation_vk_digest,
                    transition_index: transition_limbs,
                    transition_count: split_u32(air.profile.transition_count),
                    claim_count: count_limbs,
                    canonical_claims_digest: canonical,
                    transition_statement_digest: statement,
                    setup_openings_digest: terminal,
                    global_binding_digest: completion.global_binding_digest,
                },
            });
        }
        SetupPcsAuthorityClaimBridgeRecordV3 {
            completion,
            transitions,
        }
    }

    #[test]
    fn cached_main_role_does_not_fork_the_statement_claim_identity() {
        let mut cached = profile().claims[0];
        let preprocessed = cached;
        cached.identity.is_cached_main = true;
        assert_eq!(
            claim_identity_blocks(cached),
            claim_identity_blocks(preprocessed)
        );
    }

    #[test]
    fn compact_trace_has_one_row_per_claim_and_no_setup_values() {
        let air = air();
        let record = valid_record(&air);
        let trace = generate_setup_pcs_authority_claim_bridge_trace_v3(&air, &record).unwrap();
        assert_eq!(air.profile.claim_count(), 4);
        assert_eq!(trace.matrix.height(), 4);
        assert!(trace.compression_inputs.len() < 64);
        let preprocessed = BaseAir::<F>::preprocessed_trace(&air).unwrap();
        check_constraints::<_, NativeSC>(
            &air,
            "SetupPcsAuthorityClaimBridgeAirV3",
            &Some(preprocessed.as_view()),
            &[trace.matrix.as_view()],
            &[],
        );
    }

    #[test]
    fn profile_rejects_order_duplicates_gaps_and_missing_transitions() {
        let base = profile();
        let identities = base.claim_schedule().collect::<Vec<_>>();
        let mut reordered = identities.clone();
        reordered.swap(0, 1);
        assert!(matches!(
            SetupPcsAuthorityClaimBridgeProfileV3::new(
                digest(10),
                digest(30),
                digest(50),
                2,
                reordered
            ),
            Err(SetupPcsAuthorityClaimBridgeErrorV3::ClaimOrder(_))
                | Err(SetupPcsAuthorityClaimBridgeErrorV3::ColumnGap(_))
        ));
        let mut duplicate = identities.clone();
        duplicate[1] = duplicate[0];
        assert!(matches!(
            SetupPcsAuthorityClaimBridgeProfileV3::new(
                digest(10),
                digest(30),
                digest(50),
                2,
                duplicate
            ),
            Err(SetupPcsAuthorityClaimBridgeErrorV3::DuplicateClaim(_))
        ));
        let missing = identities
            .into_iter()
            .filter(|identity| identity.transition_index == 0)
            .collect();
        assert_eq!(
            SetupPcsAuthorityClaimBridgeProfileV3::new(
                digest(10),
                digest(30),
                digest(50),
                2,
                missing
            )
            .unwrap_err(),
            SetupPcsAuthorityClaimBridgeErrorV3::TransitionGap(1)
        );
    }

    #[test]
    fn trace_generation_rejects_all_authority_mutation_classes() {
        let air = air();
        let valid = valid_record(&air);

        let mut mutation = valid.clone();
        mutation.transitions[0].claims[0].current += EF::ONE;
        assert_eq!(
            generate_setup_pcs_authority_claim_bridge_trace_v3(&air, &mutation).unwrap_err(),
            SetupPcsAuthorityClaimBridgeErrorV3::SetupOpeningsDigest(0)
        );
        let mut mutation = valid.clone();
        mutation.transitions[0].claims[0].rotated = None;
        assert_eq!(
            generate_setup_pcs_authority_claim_bridge_trace_v3(&air, &mutation).unwrap_err(),
            SetupPcsAuthorityClaimBridgeErrorV3::RecordRotation(0, 0)
        );
        let mut mutation = valid.clone();
        mutation.transitions[0].claims[0].identity.col_idx = 9;
        assert_eq!(
            generate_setup_pcs_authority_claim_bridge_trace_v3(&air, &mutation).unwrap_err(),
            SetupPcsAuthorityClaimBridgeErrorV3::RecordClaimIdentity(0, 0)
        );
        let mut mutation = valid.clone();
        mutation.transitions[0].transition_index = 1;
        assert_eq!(
            generate_setup_pcs_authority_claim_bridge_trace_v3(&air, &mutation).unwrap_err(),
            SetupPcsAuthorityClaimBridgeErrorV3::RecordTransition(0)
        );
        let mut mutation = valid.clone();
        mutation.transitions[0].claims.pop();
        assert_eq!(
            generate_setup_pcs_authority_claim_bridge_trace_v3(&air, &mutation).unwrap_err(),
            SetupPcsAuthorityClaimBridgeErrorV3::RecordClaimCount(0)
        );
        let mut mutation = valid.clone();
        mutation.transitions[0].certificate.canonical_claims_digest[0] += F::ONE;
        assert_eq!(
            generate_setup_pcs_authority_claim_bridge_trace_v3(&air, &mutation).unwrap_err(),
            SetupPcsAuthorityClaimBridgeErrorV3::CanonicalClaimsDigest(0)
        );
        let mut mutation = valid.clone();
        mutation.transitions[0].transition_statement_digest[0] += F::ONE;
        assert_eq!(
            generate_setup_pcs_authority_claim_bridge_trace_v3(&air, &mutation).unwrap_err(),
            SetupPcsAuthorityClaimBridgeErrorV3::CertificateBinding(0)
        );
        let mut mutation = valid.clone();
        mutation.transitions[0].certificate.setup_openings_digest[0] += F::ONE;
        assert_eq!(
            generate_setup_pcs_authority_claim_bridge_trace_v3(&air, &mutation).unwrap_err(),
            SetupPcsAuthorityClaimBridgeErrorV3::SetupOpeningsDigest(0)
        );
        let mut mutation = valid.clone();
        mutation.transitions[0].certificate.claim_count[0] += F::ONE;
        assert_eq!(
            generate_setup_pcs_authority_claim_bridge_trace_v3(&air, &mutation).unwrap_err(),
            SetupPcsAuthorityClaimBridgeErrorV3::CertificateBinding(0)
        );
        let mut mutation = valid.clone();
        mutation.completion.whir_completion_binding[0] += F::ONE;
        assert_eq!(
            generate_setup_pcs_authority_claim_bridge_trace_v3(&air, &mutation).unwrap_err(),
            SetupPcsAuthorityClaimBridgeErrorV3::CompletionBinding
        );
        let mut mutation = valid.clone();
        mutation.transitions[1].certificate.global_binding_digest[0] += F::ONE;
        assert_eq!(
            generate_setup_pcs_authority_claim_bridge_trace_v3(&air, &mutation).unwrap_err(),
            SetupPcsAuthorityClaimBridgeErrorV3::CertificateBinding(1)
        );
    }

    #[test]
    fn production_constructor_cannot_degrade_to_optional_authority_edges() {
        assert_eq!(
            profile().validate_production_v3(),
            Err(SetupPcsAuthorityClaimBridgeErrorV3::MissingProductionBinding)
        );

        let base = profile();
        let production_profile = SetupPcsAuthorityClaimBridgeProfileV3::new_production_v3(
            base.authority_profile_digest,
            base.relation_digest,
            base.aggregation_vk_digest,
            base.transition_count(),
            base.claim_schedule().collect(),
            SetupPcsSourceProvenanceBusV3::new(BusIndex::from(20u16)),
            SetupPcsAuthorityTransitionStatementBusV3::new(BusIndex::from(21u16)),
            ColumnClaimsBus::new(BusIndex::from(22u16)),
        )
        .unwrap();
        let air = SetupPcsAuthorityClaimBridgeAirV3::new_production_v3(
            production_profile,
            SetupPcsAuthorityCompletionBusV3::new(BusIndex::from(23u16)),
            SetupPcsAuthorityTransitionCertificateBusV3::new(BusIndex::from(24u16)),
            FixedSetupOpeningCertificateBusV2::new(BusIndex::from(25u16)),
            ColumnClaimsBus::new(BusIndex::from(26u16)),
            ColumnClaimsBus::new(BusIndex::from(27u16)),
            Poseidon2CompressBus::new(BusIndex::from(28u16)),
        )
        .unwrap();
        assert!(air.history_certificate_bus.is_some());
        assert!(air.stacking_claims_bus.is_some());
        air.profile.validate_production_v3().unwrap();
    }

    fn source_provenance(
        air: &SetupPcsAuthorityClaimBridgeAirV3,
        transition_index: u32,
    ) -> SetupPcsSourceProvenanceMessageV3<F> {
        SetupPcsSourceProvenanceMessageV3 {
            protocol_version: F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition_index: split_u32(transition_index),
            segment_index: split_u32(transition_index),
            app_vk_digest: air.profile.aggregation_vk_digest,
            relation_digest: air.profile.relation_digest,
            source_root: digest(300 + transition_index * 100),
            source_instance_digest: digest(320 + transition_index * 100),
            source_forest_root: digest(340 + transition_index * 100),
            segment_openings_digest: digest(360 + transition_index * 100),
            source_checkpoint_digest: digest(380 + transition_index * 100),
            source_manifest_digest: digest(400 + transition_index * 100),
            source_receipt_digest: digest(420 + transition_index * 100),
            end_tidx: split_u32(1_000 + transition_index),
            end_sample_count: F::from_u8(D_EF as u8),
            end_state: core::array::from_fn(|lane| {
                F::from_u32(2_000 + transition_index * 100 + lane as u32)
            }),
        }
    }

    #[test]
    fn constrained_mode_requires_provenance_statement_and_exact_claim_fanout() {
        let provenance_index = BusIndex::from(5u16);
        let statement_index = BusIndex::from(6u16);
        let stacking_index = BusIndex::from(7u16);
        let authority_claim_index = BusIndex::from(8u16);
        let source_claim_index = BusIndex::from(9u16);
        let configured_profile = profile()
            .with_constrained_source_provenance_v3(
                SetupPcsSourceProvenanceBusV3::new(provenance_index),
                SetupPcsAuthorityTransitionStatementBusV3::new(statement_index),
            )
            .with_authority_statement_column_claims_bus_v3(ColumnClaimsBus::new(
                authority_claim_index,
            ));
        let air = SetupPcsAuthorityClaimBridgeAirV3 {
            profile: configured_profile,
            completion_bus: SetupPcsAuthorityCompletionBusV3::new(BusIndex::from(1u16)),
            transition_certificate_bus: SetupPcsAuthorityTransitionCertificateBusV3::new(
                BusIndex::from(2u16),
            ),
            history_certificate_bus: None,
            stacking_claims_bus: Some(ColumnClaimsBus::new(stacking_index)),
            column_claims_bus: ColumnClaimsBus::new(source_claim_index),
            compress_bus: Poseidon2CompressBus::new(BusIndex::from(10u16)),
        };
        let missing = valid_record(&air);
        assert_eq!(
            generate_setup_pcs_authority_claim_bridge_trace_v3(&air, &missing).unwrap_err(),
            SetupPcsAuthorityClaimBridgeErrorV3::MissingSourceProvenance(0)
        );
        let mut record = valid_record(&air);
        for transition in &mut record.transitions {
            transition.source_provenance =
                Some(source_provenance(&air, transition.transition_index));
        }
        let trace =
            generate_setup_pcs_authority_claim_bridge_trace_v3(&air, &record).expect("bridge");
        let preprocessed = BaseAir::<F>::preprocessed_trace(&air).unwrap();
        check_constraints::<_, NativeSC>(
            &air,
            "SetupPcsAuthorityClaimBridgeAirV3-constrained",
            &Some(preprocessed.as_view()),
            &[trace.matrix.as_view()],
            &[],
        );

        let symbolic = get_symbolic_builder(
            &air,
            &TraceWidth {
                preprocessed: Some(preprocessed.width()),
                cached_mains: air.cached_main_widths(),
                common_main: air.common_main_width(),
            },
        )
        .constraints()
        .interactions;
        let count = |index| {
            symbolic
                .iter()
                .filter(|interaction| interaction.bus_index == index)
                .count()
        };
        assert_eq!(count(provenance_index), 1, "one provenance receive");
        assert_eq!(
            count(statement_index),
            1,
            "one transition-statement receive"
        );
        assert_eq!(
            count(source_claim_index),
            4,
            "two source consumers per claim kind"
        );
        assert_eq!(
            count(stacking_index),
            2,
            "current/rotation stacking equality"
        );
        assert_eq!(
            count(authority_claim_index),
            2,
            "one current/rotation copy for the authority statement"
        );

        let mut mismatch = record;
        mismatch.transitions[1]
            .source_provenance
            .as_mut()
            .unwrap()
            .end_state[0] += F::ONE;
        // Checkpoint state is opaque to the bridge but the complete typed
        // provenance message is received. Its source receipt must therefore
        // be produced by the provenance/statement AIRs, not repaired here.
        assert!(generate_setup_pcs_authority_claim_bridge_trace_v3(&air, &mismatch).is_ok());
        mismatch.transitions[1]
            .source_provenance
            .as_mut()
            .unwrap()
            .relation_digest[0] += F::ONE;
        assert_eq!(
            generate_setup_pcs_authority_claim_bridge_trace_v3(&air, &mismatch).unwrap_err(),
            SetupPcsAuthorityClaimBridgeErrorV3::SourceProvenanceBinding(1)
        );
    }
}

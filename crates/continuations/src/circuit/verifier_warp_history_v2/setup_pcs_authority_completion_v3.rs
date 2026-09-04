//! Fail-closed completion owner for setup-PCS authority v3.
//!
//! This AIR is the only adapter allowed to turn a constrained terminal
//! multi-constraint WHIR completion into the messages consumed by History.
//! It receives exactly one WHIR completion under authority `(proof_idx,
//! class_index) = (0, 0)`, hashes the complete constrained endpoint, and only
//! then sends one global completion and one certificate for every
//! setup-fixed transition.
//!
//! There is deliberately no `post_stacking_digest`.  Ordered stacking and the
//! multi-opening statement precede WHIR in one authority transcript, so the
//! constrained terminal transcript state already binds that protocol history.
//! Adding a native digest here would create a second, host-derived authority
//! path.
//!
//! The two caller buses are receivers in this file.  Production must connect
//! them to the verifier-owned statement AIR:
//!
//! - [`SetupPcsAuthorityBoundBatchBusV3`] authenticates the setup-fixed batch statement digest;
//! - [`SetupPcsAuthorityBoundTransitionBusV3`] authenticates each canonical transition statement
//!   and the pre-global setup-opening digest produced from the constrained ordered-stacking
//!   outputs.
//!
//! No host record can release authority: without the WHIR sender, either
//! caller sender, or the Poseidon owners, the interaction multiset is
//! unbalanced.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_recursion_circuit::{
    bus::{Poseidon2CompressBus, Poseidon2CompressMessage},
    define_typed_permutation_bus,
    whir::multi_constraint::module::{
        MultiConstraintWhirCompletionBus, MultiConstraintWhirCompletionMessage,
    },
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, Digest, DIGEST_SIZE, D_EF, F,
};

use super::{
    SetupPcsAuthorityCompletionBusV3, SetupPcsAuthorityCompletionMessageV3,
    SetupPcsAuthorityTransitionCertificateBusV3, SetupPcsAuthorityTransitionCertificateMessageV3,
    SETUP_PCS_AUTHORITY_GLOBAL_END_TAG_V3, SETUP_PCS_AUTHORITY_GLOBAL_START_TAG_V3,
    SETUP_PCS_AUTHORITY_PROTOCOL_V3, SETUP_PCS_AUTHORITY_TRANSITION_END_TAG_V3,
    VERIFIER_WARP_HISTORY_MAX_TRANSITIONS_V2,
};

pub const SETUP_PCS_AUTHORITY_WHIR_PROOF_IDX_V3: u32 = 0;
pub const SETUP_PCS_AUTHORITY_WHIR_CLASS_IDX_V3: u32 = 0;
pub const SETUP_PCS_AUTHORITY_WHIR_COMPLETION_START_TAG_V3: u32 = 0x5350_5701;
pub const SETUP_PCS_AUTHORITY_WHIR_STATE_TAG_V3: u32 = 0x5350_5702;
pub const SETUP_PCS_AUTHORITY_WHIR_AGGREGATE_TAG_V3: u32 = 0x5350_5703;
pub const SETUP_PCS_AUTHORITY_WHIR_CLAIM_TAG_V3: u32 = 0x5350_5704;
pub const SETUP_PCS_AUTHORITY_WHIR_COMPLETION_END_TAG_V3: u32 = 0x5350_5705;

const WHIR_HASH_STEPS_V3: usize = 7;
const GLOBAL_HASH_STEPS_V3: usize = 7;

// The binding below covers the entire canonical 16-lane recursive transcript
// state in three tagged blocks.  A parameter change must deliberately version
// this protocol instead of silently dropping or reordering state lanes.
const _: [(); 2 * DIGEST_SIZE] = [(); POSEIDON2_WIDTH];

/// Batch statement produced by the constrained authority statement owner.
/// All profile fields are repeated so mismatched key material cannot share a
/// batch digest at this boundary.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug, PartialEq, Eq)]
pub struct SetupPcsAuthorityBoundBatchMessageV3<T> {
    pub protocol_version: T,
    pub profile_digest: [T; DIGEST_SIZE],
    pub batch_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub aggregation_vk_digest: [T; DIGEST_SIZE],
    pub transition_count: [T; 2],
}
define_typed_permutation_bus!(
    SetupPcsAuthorityBoundBatchBusV3,
    SetupPcsAuthorityBoundBatchMessageV3
);

/// One transition statement produced by the constrained statement and
/// ordered-stacking composition.
///
/// `setup_openings_pre_global_digest` is the canonical digest immediately
/// after absorbing the transition statement, relation, and aggregation VK.
/// This AIR appends the WHIR-derived global binding and transition end block;
/// consequently a transition certificate cannot be replayed under another
/// WHIR completion.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug, PartialEq, Eq)]
pub struct SetupPcsAuthorityBoundTransitionMessageV3<T> {
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
    pub setup_openings_pre_global_digest: [T; DIGEST_SIZE],
}
define_typed_permutation_bus!(
    SetupPcsAuthorityBoundTransitionBusV3,
    SetupPcsAuthorityBoundTransitionMessageV3
);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SetupPcsAuthorityCompletionProfileV3 {
    profile_digest: Digest,
    relation_digest: Digest,
    aggregation_vk_digest: Digest,
    claim_counts: Arc<[u32]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetupPcsAuthorityCompletionErrorV3 {
    EmptyProfile,
    TooManyTransitions,
    ZeroProfileDigest,
    ZeroRelationDigest,
    ZeroAggregationVkDigest,
    ZeroClaimCount(usize),
    RecordTransitionCount,
    BatchBinding,
    TransitionBinding(usize),
    WhirBinding,
}

impl core::fmt::Display for SetupPcsAuthorityCompletionErrorV3 {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "invalid setup-PCS authority completion: {self:?}"
        )
    }
}

impl std::error::Error for SetupPcsAuthorityCompletionErrorV3 {}

impl SetupPcsAuthorityCompletionProfileV3 {
    pub fn new(
        profile_digest: Digest,
        relation_digest: Digest,
        aggregation_vk_digest: Digest,
        claim_counts: Vec<u32>,
    ) -> Result<Self, SetupPcsAuthorityCompletionErrorV3> {
        let profile = Self {
            profile_digest,
            relation_digest,
            aggregation_vk_digest,
            claim_counts: claim_counts.into(),
        };
        profile.validate()?;
        Ok(profile)
    }

    pub fn validate(&self) -> Result<(), SetupPcsAuthorityCompletionErrorV3> {
        if self.claim_counts.is_empty() {
            return Err(SetupPcsAuthorityCompletionErrorV3::EmptyProfile);
        }
        if self.claim_counts.len() > VERIFIER_WARP_HISTORY_MAX_TRANSITIONS_V2 {
            return Err(SetupPcsAuthorityCompletionErrorV3::TooManyTransitions);
        }
        if digest_is_zero(&self.profile_digest) {
            return Err(SetupPcsAuthorityCompletionErrorV3::ZeroProfileDigest);
        }
        if digest_is_zero(&self.relation_digest) {
            return Err(SetupPcsAuthorityCompletionErrorV3::ZeroRelationDigest);
        }
        if digest_is_zero(&self.aggregation_vk_digest) {
            return Err(SetupPcsAuthorityCompletionErrorV3::ZeroAggregationVkDigest);
        }
        if let Some((index, _)) = self
            .claim_counts
            .iter()
            .enumerate()
            .find(|(_, count)| **count == 0)
        {
            return Err(SetupPcsAuthorityCompletionErrorV3::ZeroClaimCount(index));
        }
        Ok(())
    }

    #[must_use]
    pub fn transition_count(&self) -> usize {
        self.claim_counts.len()
    }

    #[must_use]
    pub fn profile_digest(&self) -> Digest {
        self.profile_digest
    }

    #[must_use]
    pub fn relation_digest(&self) -> Digest {
        self.relation_digest
    }

    #[must_use]
    pub fn aggregation_vk_digest(&self) -> Digest {
        self.aggregation_vk_digest
    }

    #[must_use]
    pub fn claim_counts(&self) -> &[u32] {
        &self.claim_counts
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct SetupPcsAuthorityCompletionPrepColsV3<T> {
    active: T,
    is_first: T,
    is_last: T,
    transition_index: [T; 2],
    transition_count: [T; 2],
    claim_count: [T; 2],
    profile_digest: [T; DIGEST_SIZE],
    relation_digest: [T; DIGEST_SIZE],
    aggregation_vk_digest: [T; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct SetupPcsAuthorityCompletionColsV3<T> {
    pub active: T,
    pub batch: SetupPcsAuthorityBoundBatchMessageV3<T>,
    pub transition: SetupPcsAuthorityBoundTransitionMessageV3<T>,
    pub whir: MultiConstraintWhirCompletionMessage<T>,
    pub whir_hashes: [[T; DIGEST_SIZE]; WHIR_HASH_STEPS_V3],
    pub global_binding_digest: [T; DIGEST_SIZE],
    pub global_hashes: [[T; DIGEST_SIZE]; GLOBAL_HASH_STEPS_V3],
    pub setup_openings_with_global: [T; DIGEST_SIZE],
    pub setup_openings_final: [T; DIGEST_SIZE],
}

#[derive(Clone, Debug)]
pub struct SetupPcsAuthorityCompletionAirV3 {
    pub profile: SetupPcsAuthorityCompletionProfileV3,
    pub batch_statement_bus: SetupPcsAuthorityBoundBatchBusV3,
    pub transition_statement_bus: SetupPcsAuthorityBoundTransitionBusV3,
    pub whir_completion_bus: MultiConstraintWhirCompletionBus,
    pub completion_bus: SetupPcsAuthorityCompletionBusV3,
    pub transition_certificate_bus: SetupPcsAuthorityTransitionCertificateBusV3,
    pub compress_bus: Poseidon2CompressBus,
}

impl BaseAir<F> for SetupPcsAuthorityCompletionAirV3 {
    fn width(&self) -> usize {
        core::mem::size_of::<SetupPcsAuthorityCompletionColsV3<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.profile
            .validate()
            .expect("invalid setup-PCS authority completion profile in VK");
        let width = core::mem::size_of::<SetupPcsAuthorityCompletionPrepColsV3<u8>>();
        let height = self.profile.transition_count().next_power_of_two().max(2);
        let mut values = F::zero_vec(width * height);
        let transition_count = split_u32(self.profile.transition_count() as u32);
        for (index, &claim_count) in self.profile.claim_counts.iter().enumerate() {
            let cols: &mut SetupPcsAuthorityCompletionPrepColsV3<F> =
                values[index * width..(index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.is_first = F::from_bool(index == 0);
            cols.is_last = F::from_bool(index + 1 == self.profile.transition_count());
            cols.transition_index = split_u32(index as u32);
            cols.transition_count = transition_count;
            cols.claim_count = split_u32(claim_count);
            cols.profile_digest = self.profile.profile_digest;
            cols.relation_digest = self.profile.relation_digest;
            cols.aggregation_vk_digest = self.profile.aggregation_vk_digest;
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for SetupPcsAuthorityCompletionAirV3 {}
impl PartitionedBaseAir<F> for SetupPcsAuthorityCompletionAirV3 {}

fn digest_is_zero(value: &Digest) -> bool {
    value.iter().all(|&limb| limb == F::ZERO)
}

fn split_u32(value: u32) -> [F; 2] {
    [F::from_u16(value as u16), F::from_u16((value >> 16) as u16)]
}

fn zero_block<FA: PrimeCharacteristicRing>() -> [FA; DIGEST_SIZE] {
    core::array::from_fn(|_| FA::ZERO)
}

fn assert_array_eq_when<AB, const N: usize>(
    builder: &mut AB,
    enabled: impl Into<AB::Expr> + Clone,
    actual: [impl Into<AB::Expr>; N],
    expected: [impl Into<AB::Expr>; N],
) where
    AB: AirBuilder<F = F>,
{
    for (actual, expected) in actual.into_iter().zip(expected) {
        builder
            .when(enabled.clone())
            .assert_eq(actual.into(), expected.into());
    }
}

fn assert_message_eq_when<AB>(
    builder: &mut AB,
    enabled: impl Into<AB::Expr> + Clone,
    actual: Vec<AB::Var>,
    expected: Vec<AB::Var>,
) where
    AB: AirBuilder<F = F>,
    AB::Var: Copy,
{
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.into_iter().zip(expected) {
        builder.when(enabled.clone()).assert_eq(actual, expected);
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

impl<AB> Air<AB> for SetupPcsAuthorityCompletionAirV3
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let prep_matrix = builder.preprocessed();
        let prep_row = prep_matrix
            .row_slice(0)
            .expect("setup-PCS completion prep row");
        let prep: &SetupPcsAuthorityCompletionPrepColsV3<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("setup-PCS completion row");
        let next_row = main.row_slice(1).expect("setup-PCS completion next row");
        let local: &SetupPcsAuthorityCompletionColsV3<AB::Var> = (*row).borrow();
        let next: &SetupPcsAuthorityCompletionColsV3<AB::Var> = (*next_row).borrow();

        for bit in [prep.active, prep.is_first, prep.is_last, local.active] {
            builder.assert_bool(bit);
        }
        builder.assert_eq(local.active, prep.active);
        let enabled = AB::Expr::from(prep.active);
        let first = AB::Expr::from(prep.is_first);
        let last = AB::Expr::from(prep.is_last);
        let not_last = enabled.clone() * (AB::Expr::ONE - last.clone());

        for &cell in row.iter() {
            builder
                .when(AB::Expr::ONE - enabled.clone())
                .assert_zero(cell);
        }

        // The unique batch statement enters on row zero, then is constrained
        // unchanged across the fixed active prefix.
        builder.when(enabled.clone()).assert_eq(
            local.batch.protocol_version,
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
        );
        for (actual, expected) in [
            (local.batch.profile_digest, prep.profile_digest),
            (local.batch.relation_digest, prep.relation_digest),
            (
                local.batch.aggregation_vk_digest,
                prep.aggregation_vk_digest,
            ),
        ] {
            assert_array_eq_when(builder, enabled.clone(), actual, expected);
        }
        assert_array_eq_when(
            builder,
            enabled.clone(),
            local.batch.transition_count,
            prep.transition_count,
        );
        self.batch_statement_bus
            .receive(builder, local.batch.clone(), first.clone());
        assert_message_eq_when(
            &mut builder.when_transition(),
            not_last.clone(),
            next.batch.clone().to_vec(),
            local.batch.clone().to_vec(),
        );

        // One verifier-owned transition statement is required at every
        // active, setup-fixed index.
        builder.when(enabled.clone()).assert_eq(
            local.transition.protocol_version,
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
        );
        for (actual, expected) in [
            (local.transition.profile_digest, prep.profile_digest),
            (local.transition.batch_digest, local.batch.batch_digest),
            (local.transition.relation_digest, prep.relation_digest),
            (
                local.transition.aggregation_vk_digest,
                prep.aggregation_vk_digest,
            ),
        ] {
            assert_array_eq_when(builder, enabled.clone(), actual, expected);
        }
        for (actual, expected) in [
            (local.transition.transition_index, prep.transition_index),
            (local.transition.transition_count, prep.transition_count),
            (local.transition.claim_count, prep.claim_count),
        ] {
            assert_array_eq_when(builder, enabled.clone(), actual, expected);
        }
        self.transition_statement_bus
            .receive(builder, local.transition.clone(), enabled.clone());

        // Bind the complete transition opening digest to the same global
        // WHIR completion used by every other transition certificate.
        compress_lookup(
            self.compress_bus,
            builder,
            local.transition.setup_openings_pre_global_digest,
            local.global_binding_digest,
            local.setup_openings_with_global,
            enabled.clone(),
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
            local.setup_openings_with_global,
            transition_end,
            local.setup_openings_final,
            enabled.clone(),
        );

        // Global binding is copied backwards algebraically: all active rows
        // must agree, and the unique last row constrains it to the hash of the
        // accepted WHIR endpoint.
        assert_array_eq_when(
            &mut builder.when_transition(),
            not_last.clone(),
            next.global_binding_digest,
            local.global_binding_digest,
        );

        builder
            .when(last.clone())
            .assert_eq(local.whir.proof_idx, AB::Expr::ZERO);
        builder
            .when(last.clone())
            .assert_eq(local.whir.class_index, AB::Expr::ZERO);
        self.whir_completion_bus
            .receive(builder, AB::Expr::ZERO, local.whir.clone(), last.clone());

        let whir_header: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_WHIR_COMPLETION_START_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
            local.whir.proof_idx.into(),
            local.whir.class_index.into(),
            local.whir.end_tidx.into(),
            local.whir.sample_count.into(),
            AB::Expr::from_usize(POSEIDON2_WIDTH),
            AB::Expr::from_usize(D_EF),
        ];
        let whir_state_0: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_WHIR_STATE_TAG_V3),
            AB::Expr::ZERO,
            local.whir.state[0].into(),
            local.whir.state[1].into(),
            local.whir.state[2].into(),
            local.whir.state[3].into(),
            local.whir.state[4].into(),
            local.whir.state[5].into(),
        ];
        let whir_state_1: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_WHIR_STATE_TAG_V3),
            AB::Expr::ONE,
            local.whir.state[6].into(),
            local.whir.state[7].into(),
            local.whir.state[8].into(),
            local.whir.state[9].into(),
            local.whir.state[10].into(),
            local.whir.state[11].into(),
        ];
        let whir_state_2: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_WHIR_STATE_TAG_V3),
            AB::Expr::from_u32(2),
            local.whir.state[12].into(),
            local.whir.state[13].into(),
            local.whir.state[14].into(),
            local.whir.state[15].into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        let whir_aggregate: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_WHIR_AGGREGATE_TAG_V3),
            local.whir.final_aggregate[0].into(),
            local.whir.final_aggregate[1].into(),
            local.whir.final_aggregate[2].into(),
            local.whir.final_aggregate[3].into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        let whir_claim: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_WHIR_CLAIM_TAG_V3),
            local.whir.final_claim[0].into(),
            local.whir.final_claim[1].into(),
            local.whir.final_claim[2].into(),
            local.whir.final_claim[3].into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        let whir_end: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_WHIR_COMPLETION_END_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
            local.whir.proof_idx.into(),
            local.whir.class_index.into(),
            local.whir.end_tidx.into(),
            local.whir.sample_count.into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        let whir_blocks = [
            whir_header,
            whir_state_0,
            whir_state_1,
            whir_state_2,
            whir_aggregate,
            whir_claim,
            whir_end,
        ];
        let mut whir_before = zero_block::<AB::Expr>();
        for (block, after) in whir_blocks.into_iter().zip(local.whir_hashes) {
            compress_lookup(
                self.compress_bus,
                builder,
                whir_before,
                block,
                after,
                last.clone(),
            );
            whir_before = after.map(Into::into);
        }
        let whir_completion_binding = local.whir_hashes[WHIR_HASH_STEPS_V3 - 1];

        let global_header: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_GLOBAL_START_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
            prep.transition_count[0].into(),
            prep.transition_count[1].into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        let global_end: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_GLOBAL_END_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
            prep.transition_count[0].into(),
            prep.transition_count[1].into(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        let global_blocks: [[AB::Expr; DIGEST_SIZE]; GLOBAL_HASH_STEPS_V3] = [
            global_header,
            prep.profile_digest.map(Into::into),
            local.batch.batch_digest.map(Into::into),
            prep.relation_digest.map(Into::into),
            prep.aggregation_vk_digest.map(Into::into),
            whir_completion_binding.map(Into::into),
            global_end,
        ];
        let mut global_before = zero_block::<AB::Expr>();
        for (block, after) in global_blocks.into_iter().zip(local.global_hashes) {
            compress_lookup(
                self.compress_bus,
                builder,
                global_before,
                block,
                after,
                last.clone(),
            );
            global_before = after.map(Into::into);
        }
        let constrained_global = local.global_hashes[GLOBAL_HASH_STEPS_V3 - 1];
        assert_array_eq_when(
            builder,
            last.clone(),
            local.global_binding_digest,
            constrained_global,
        );

        // WHIR and its hash scratch exist only on the unique completion row.
        for value in local.whir.clone().to_vec() {
            builder
                .when(enabled.clone() * (AB::Expr::ONE - last.clone()))
                .assert_zero(value);
        }
        for digest in local.whir_hashes.into_iter().chain(local.global_hashes) {
            for value in digest {
                builder
                    .when(enabled.clone() * (AB::Expr::ONE - last.clone()))
                    .assert_zero(value);
            }
        }

        self.completion_bus.send(
            builder,
            SetupPcsAuthorityCompletionMessageV3 {
                protocol_version: AB::Expr::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
                profile_digest: prep.profile_digest.map(Into::into),
                batch_digest: local.batch.batch_digest.map(Into::into),
                relation_digest: prep.relation_digest.map(Into::into),
                aggregation_vk_digest: prep.aggregation_vk_digest.map(Into::into),
                transition_count: prep.transition_count.map(Into::into),
                whir_completion_binding: whir_completion_binding.map(Into::into),
                global_binding_digest: constrained_global.map(Into::into),
            },
            last.clone(),
        );
        self.transition_certificate_bus.send(
            builder,
            SetupPcsAuthorityTransitionCertificateMessageV3 {
                protocol_version: AB::Expr::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
                profile_digest: prep.profile_digest.map(Into::into),
                batch_digest: local.batch.batch_digest.map(Into::into),
                relation_digest: prep.relation_digest.map(Into::into),
                aggregation_vk_digest: prep.aggregation_vk_digest.map(Into::into),
                transition_index: prep.transition_index.map(Into::into),
                transition_count: prep.transition_count.map(Into::into),
                claim_count: prep.claim_count.map(Into::into),
                canonical_claims_digest: local.transition.canonical_claims_digest.map(Into::into),
                transition_statement_digest: local
                    .transition
                    .transition_statement_digest
                    .map(Into::into),
                setup_openings_digest: local.setup_openings_final.map(Into::into),
                global_binding_digest: local.global_binding_digest.map(Into::into),
            },
            enabled,
        );
    }
}

#[derive(Clone, Debug)]
pub struct SetupPcsAuthorityCompletionRecordV3 {
    pub batch: SetupPcsAuthorityBoundBatchMessageV3<F>,
    pub transitions: Vec<SetupPcsAuthorityBoundTransitionMessageV3<F>>,
    pub whir: MultiConstraintWhirCompletionMessage<F>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetupPcsAuthorityPoseidonRecordV3 {
    pub input: [F; POSEIDON2_WIDTH],
    pub output: Digest,
}

#[derive(Clone, Debug)]
pub struct SetupPcsAuthorityCompletionTraceV3 {
    pub matrix: RowMajorMatrix<F>,
    pub poseidon: Vec<SetupPcsAuthorityPoseidonRecordV3>,
    pub completion: SetupPcsAuthorityCompletionMessageV3<F>,
    pub certificates: Vec<SetupPcsAuthorityTransitionCertificateMessageV3<F>>,
}

fn compress_host(
    records: &mut Vec<SetupPcsAuthorityPoseidonRecordV3>,
    left: Digest,
    right: Digest,
) -> Digest {
    let input = core::array::from_fn(|index| {
        if index < DIGEST_SIZE {
            left[index]
        } else {
            right[index - DIGEST_SIZE]
        }
    });
    let output = poseidon2_compress_with_capacity(left, right).0;
    records.push(SetupPcsAuthorityPoseidonRecordV3 { input, output });
    output
}

fn whir_blocks_host(
    whir: &MultiConstraintWhirCompletionMessage<F>,
) -> [[F; DIGEST_SIZE]; WHIR_HASH_STEPS_V3] {
    [
        [
            F::from_u32(SETUP_PCS_AUTHORITY_WHIR_COMPLETION_START_TAG_V3),
            F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
            whir.proof_idx,
            whir.class_index,
            whir.end_tidx,
            whir.sample_count,
            F::from_usize(POSEIDON2_WIDTH),
            F::from_usize(D_EF),
        ],
        [
            F::from_u32(SETUP_PCS_AUTHORITY_WHIR_STATE_TAG_V3),
            F::ZERO,
            whir.state[0],
            whir.state[1],
            whir.state[2],
            whir.state[3],
            whir.state[4],
            whir.state[5],
        ],
        [
            F::from_u32(SETUP_PCS_AUTHORITY_WHIR_STATE_TAG_V3),
            F::ONE,
            whir.state[6],
            whir.state[7],
            whir.state[8],
            whir.state[9],
            whir.state[10],
            whir.state[11],
        ],
        [
            F::from_u32(SETUP_PCS_AUTHORITY_WHIR_STATE_TAG_V3),
            F::from_u32(2),
            whir.state[12],
            whir.state[13],
            whir.state[14],
            whir.state[15],
            F::ZERO,
            F::ZERO,
        ],
        [
            F::from_u32(SETUP_PCS_AUTHORITY_WHIR_AGGREGATE_TAG_V3),
            whir.final_aggregate[0],
            whir.final_aggregate[1],
            whir.final_aggregate[2],
            whir.final_aggregate[3],
            F::ZERO,
            F::ZERO,
            F::ZERO,
        ],
        [
            F::from_u32(SETUP_PCS_AUTHORITY_WHIR_CLAIM_TAG_V3),
            whir.final_claim[0],
            whir.final_claim[1],
            whir.final_claim[2],
            whir.final_claim[3],
            F::ZERO,
            F::ZERO,
            F::ZERO,
        ],
        [
            F::from_u32(SETUP_PCS_AUTHORITY_WHIR_COMPLETION_END_TAG_V3),
            F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
            whir.proof_idx,
            whir.class_index,
            whir.end_tidx,
            whir.sample_count,
            F::ZERO,
            F::ZERO,
        ],
    ]
}

fn hash_whir_completion_host(
    records: &mut Vec<SetupPcsAuthorityPoseidonRecordV3>,
    whir: &MultiConstraintWhirCompletionMessage<F>,
) -> ([Digest; WHIR_HASH_STEPS_V3], Digest) {
    let mut hashes = [[F::ZERO; DIGEST_SIZE]; WHIR_HASH_STEPS_V3];
    let mut state = [F::ZERO; DIGEST_SIZE];
    for (index, block) in whir_blocks_host(whir).into_iter().enumerate() {
        state = compress_host(records, state, block);
        hashes[index] = state;
    }
    (hashes, state)
}

fn hash_global_binding_host(
    records: &mut Vec<SetupPcsAuthorityPoseidonRecordV3>,
    profile: &SetupPcsAuthorityCompletionProfileV3,
    batch_digest: Digest,
    whir_completion_binding: Digest,
) -> ([Digest; GLOBAL_HASH_STEPS_V3], Digest) {
    let count = split_u32(profile.transition_count() as u32);
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
    let blocks = [
        header,
        profile.profile_digest,
        batch_digest,
        profile.relation_digest,
        profile.aggregation_vk_digest,
        whir_completion_binding,
        end,
    ];
    let mut hashes = [[F::ZERO; DIGEST_SIZE]; GLOBAL_HASH_STEPS_V3];
    let mut state = [F::ZERO; DIGEST_SIZE];
    for (index, block) in blocks.into_iter().enumerate() {
        state = compress_host(records, state, block);
        hashes[index] = state;
    }
    (hashes, state)
}

pub fn generate_setup_pcs_authority_completion_trace_v3(
    air: &SetupPcsAuthorityCompletionAirV3,
    record: &SetupPcsAuthorityCompletionRecordV3,
) -> Result<SetupPcsAuthorityCompletionTraceV3, SetupPcsAuthorityCompletionErrorV3> {
    air.profile.validate()?;
    if record.transitions.len() != air.profile.transition_count() {
        return Err(SetupPcsAuthorityCompletionErrorV3::RecordTransitionCount);
    }
    let expected_count = split_u32(air.profile.transition_count() as u32);
    if record.batch.protocol_version != F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3)
        || record.batch.profile_digest != air.profile.profile_digest
        || record.batch.relation_digest != air.profile.relation_digest
        || record.batch.aggregation_vk_digest != air.profile.aggregation_vk_digest
        || record.batch.transition_count != expected_count
        || digest_is_zero(&record.batch.batch_digest)
    {
        return Err(SetupPcsAuthorityCompletionErrorV3::BatchBinding);
    }
    for (index, (transition, &claim_count)) in record
        .transitions
        .iter()
        .zip(air.profile.claim_counts.iter())
        .enumerate()
    {
        if transition.protocol_version != F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3)
            || transition.profile_digest != air.profile.profile_digest
            || transition.batch_digest != record.batch.batch_digest
            || transition.relation_digest != air.profile.relation_digest
            || transition.aggregation_vk_digest != air.profile.aggregation_vk_digest
            || transition.transition_index != split_u32(index as u32)
            || transition.transition_count != expected_count
            || transition.claim_count != split_u32(claim_count)
            || digest_is_zero(&transition.canonical_claims_digest)
            || digest_is_zero(&transition.transition_statement_digest)
            || digest_is_zero(&transition.setup_openings_pre_global_digest)
        {
            return Err(SetupPcsAuthorityCompletionErrorV3::TransitionBinding(index));
        }
    }
    if record.whir.proof_idx != F::ZERO
        || record.whir.class_index != F::ZERO
        || record.whir.sample_count == F::ZERO
    {
        return Err(SetupPcsAuthorityCompletionErrorV3::WhirBinding);
    }

    let mut poseidon = Vec::new();
    let (whir_hashes, whir_completion_binding) =
        hash_whir_completion_host(&mut poseidon, &record.whir);
    let (global_hashes, global_binding_digest) = hash_global_binding_host(
        &mut poseidon,
        &air.profile,
        record.batch.batch_digest,
        whir_completion_binding,
    );
    let completion = SetupPcsAuthorityCompletionMessageV3 {
        protocol_version: F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
        profile_digest: air.profile.profile_digest,
        batch_digest: record.batch.batch_digest,
        relation_digest: air.profile.relation_digest,
        aggregation_vk_digest: air.profile.aggregation_vk_digest,
        transition_count: expected_count,
        whir_completion_binding,
        global_binding_digest,
    };

    let width = core::mem::size_of::<SetupPcsAuthorityCompletionColsV3<u8>>();
    let height = air.profile.transition_count().next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    let mut certificates = Vec::with_capacity(record.transitions.len());
    for (index, transition) in record.transitions.iter().enumerate() {
        let cols: &mut SetupPcsAuthorityCompletionColsV3<F> =
            values[index * width..(index + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.batch = record.batch.clone();
        cols.transition = transition.clone();
        cols.global_binding_digest = global_binding_digest;
        if index + 1 == air.profile.transition_count() {
            cols.whir = record.whir.clone();
            cols.whir_hashes = whir_hashes;
            cols.global_hashes = global_hashes;
        }
        cols.setup_openings_with_global = compress_host(
            &mut poseidon,
            transition.setup_openings_pre_global_digest,
            global_binding_digest,
        );
        let end = [
            F::from_u32(SETUP_PCS_AUTHORITY_TRANSITION_END_TAG_V3),
            F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
            transition.transition_index[0],
            transition.transition_index[1],
            transition.claim_count[0],
            transition.claim_count[1],
            F::ZERO,
            F::ZERO,
        ];
        cols.setup_openings_final =
            compress_host(&mut poseidon, cols.setup_openings_with_global, end);
        certificates.push(SetupPcsAuthorityTransitionCertificateMessageV3 {
            protocol_version: F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
            profile_digest: air.profile.profile_digest,
            batch_digest: record.batch.batch_digest,
            relation_digest: air.profile.relation_digest,
            aggregation_vk_digest: air.profile.aggregation_vk_digest,
            transition_index: transition.transition_index,
            transition_count: expected_count,
            claim_count: transition.claim_count,
            canonical_claims_digest: transition.canonical_claims_digest,
            transition_statement_digest: transition.transition_statement_digest,
            setup_openings_digest: cols.setup_openings_final,
            global_binding_digest,
        });
    }

    Ok(SetupPcsAuthorityCompletionTraceV3 {
        matrix: RowMajorMatrix::new(values, width),
        poseidon,
        completion,
        certificates,
    })
}

#[cfg(test)]
mod tests {
    use std::{panic::AssertUnwindSafe, sync::Arc};

    use openvm_recursion_circuit::whir::multi_constraint::module::MultiConstraintWhirCompletionBus;
    use openvm_stark_backend::{
        air_builders::{
            debug::{check_constraints, check_logup},
            symbolic::get_symbolic_builder,
        },
        interaction::{BusIndex, SymbolicInteraction},
        keygen::types::TraceWidth,
        p3_air::BaseAir,
        p3_matrix::Matrix,
        AirRef, AnyAir,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config as SC;

    use super::*;

    const BATCH_BUS: u16 = 7_100;
    const TRANSITION_BUS: u16 = 7_101;
    const WHIR_BUS: u16 = 7_102;
    const COMPLETION_BUS: u16 = 7_103;
    const CERTIFICATE_BUS: u16 = 7_104;
    const COMPRESS_BUS: u16 = 7_105;

    fn digest(seed: u32) -> Digest {
        core::array::from_fn(|index| F::from_u32(seed + index as u32 + 1))
    }

    fn profile() -> SetupPcsAuthorityCompletionProfileV3 {
        SetupPcsAuthorityCompletionProfileV3::new(digest(10), digest(30), digest(50), vec![2, 3, 4])
            .expect("profile")
    }

    fn completion_air() -> SetupPcsAuthorityCompletionAirV3 {
        SetupPcsAuthorityCompletionAirV3 {
            profile: profile(),
            batch_statement_bus: SetupPcsAuthorityBoundBatchBusV3::new(BusIndex::from(BATCH_BUS)),
            transition_statement_bus: SetupPcsAuthorityBoundTransitionBusV3::new(BusIndex::from(
                TRANSITION_BUS,
            )),
            whir_completion_bus: MultiConstraintWhirCompletionBus::new(BusIndex::from(WHIR_BUS)),
            completion_bus: SetupPcsAuthorityCompletionBusV3::new(BusIndex::from(COMPLETION_BUS)),
            transition_certificate_bus: SetupPcsAuthorityTransitionCertificateBusV3::new(
                BusIndex::from(CERTIFICATE_BUS),
            ),
            compress_bus: Poseidon2CompressBus::new(BusIndex::from(COMPRESS_BUS)),
        }
    }

    fn record(air: &SetupPcsAuthorityCompletionAirV3) -> SetupPcsAuthorityCompletionRecordV3 {
        let transition_count = split_u32(air.profile.transition_count() as u32);
        let batch_digest = digest(70);
        let batch = SetupPcsAuthorityBoundBatchMessageV3 {
            protocol_version: F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
            profile_digest: air.profile.profile_digest,
            batch_digest,
            relation_digest: air.profile.relation_digest,
            aggregation_vk_digest: air.profile.aggregation_vk_digest,
            transition_count,
        };
        let transitions = air
            .profile
            .claim_counts
            .iter()
            .enumerate()
            .map(
                |(index, &claim_count)| SetupPcsAuthorityBoundTransitionMessageV3 {
                    protocol_version: F::from_u32(SETUP_PCS_AUTHORITY_PROTOCOL_V3),
                    profile_digest: air.profile.profile_digest,
                    batch_digest,
                    relation_digest: air.profile.relation_digest,
                    aggregation_vk_digest: air.profile.aggregation_vk_digest,
                    transition_index: split_u32(index as u32),
                    transition_count,
                    claim_count: split_u32(claim_count),
                    canonical_claims_digest: digest(100 + index as u32 * 30),
                    transition_statement_digest: digest(110 + index as u32 * 30),
                    setup_openings_pre_global_digest: digest(120 + index as u32 * 30),
                },
            )
            .collect();
        let whir = MultiConstraintWhirCompletionMessage {
            proof_idx: F::ZERO,
            class_index: F::ZERO,
            end_tidx: F::from_u32(50_000),
            sample_count: F::from_u32(3),
            state: core::array::from_fn(|lane| F::from_u32(60_000 + lane as u32)),
            final_aggregate: core::array::from_fn(|limb| F::from_u32(70_000 + limb as u32)),
            final_claim: core::array::from_fn(|limb| F::from_u32(80_000 + limb as u32)),
        };
        SetupPcsAuthorityCompletionRecordV3 {
            batch,
            transitions,
            whir,
        }
    }

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct AuthorityFixtureCols<T> {
        batch_enabled: T,
        transition_enabled: T,
        whir_enabled: T,
        completion_enabled: T,
        certificate_enabled: T,
        whir_outer_proof_idx: T,
        batch: SetupPcsAuthorityBoundBatchMessageV3<T>,
        transition: SetupPcsAuthorityBoundTransitionMessageV3<T>,
        whir: MultiConstraintWhirCompletionMessage<T>,
        completion: SetupPcsAuthorityCompletionMessageV3<T>,
        certificate: SetupPcsAuthorityTransitionCertificateMessageV3<T>,
    }

    #[derive(Clone, Debug)]
    struct AuthorityFixtureAir {
        batch_bus: SetupPcsAuthorityBoundBatchBusV3,
        transition_bus: SetupPcsAuthorityBoundTransitionBusV3,
        whir_bus: MultiConstraintWhirCompletionBus,
        completion_bus: SetupPcsAuthorityCompletionBusV3,
        certificate_bus: SetupPcsAuthorityTransitionCertificateBusV3,
    }

    impl BaseAir<F> for AuthorityFixtureAir {
        fn width(&self) -> usize {
            core::mem::size_of::<AuthorityFixtureCols<u8>>()
        }
    }
    impl BaseAirWithPublicValues<F> for AuthorityFixtureAir {}
    impl PartitionedBaseAir<F> for AuthorityFixtureAir {}

    impl<AB> Air<AB> for AuthorityFixtureAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("authority fixture row");
            let local: &AuthorityFixtureCols<AB::Var> = (*row).borrow();
            for bit in [
                local.batch_enabled,
                local.transition_enabled,
                local.whir_enabled,
                local.completion_enabled,
                local.certificate_enabled,
            ] {
                builder.assert_bool(bit);
            }
            self.batch_bus
                .send(builder, local.batch.clone(), local.batch_enabled);
            self.transition_bus
                .send(builder, local.transition.clone(), local.transition_enabled);
            self.whir_bus.send(
                builder,
                local.whir_outer_proof_idx,
                local.whir.clone(),
                local.whir_enabled,
            );
            self.completion_bus.receive(
                builder,
                local.completion.clone(),
                local.completion_enabled,
            );
            self.certificate_bus.receive(
                builder,
                local.certificate.clone(),
                local.certificate_enabled,
            );
        }
    }

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct PoseidonFixtureCols<T> {
        active: T,
        input: [T; POSEIDON2_WIDTH],
        output: [T; DIGEST_SIZE],
    }

    #[derive(Clone, Debug)]
    struct PoseidonFixtureAir(Poseidon2CompressBus);

    impl BaseAir<F> for PoseidonFixtureAir {
        fn width(&self) -> usize {
            core::mem::size_of::<PoseidonFixtureCols<u8>>()
        }
    }
    impl BaseAirWithPublicValues<F> for PoseidonFixtureAir {}
    impl PartitionedBaseAir<F> for PoseidonFixtureAir {}

    impl<AB> Air<AB> for PoseidonFixtureAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("Poseidon fixture row");
            let local: &PoseidonFixtureCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.active);
            self.0.add_key_with_lookups(
                builder,
                Poseidon2CompressMessage {
                    input: local.input.map(Into::into),
                    output: local.output.map(Into::into),
                },
                local.active,
            );
        }
    }

    fn fixture_air() -> AuthorityFixtureAir {
        AuthorityFixtureAir {
            batch_bus: SetupPcsAuthorityBoundBatchBusV3::new(BusIndex::from(BATCH_BUS)),
            transition_bus: SetupPcsAuthorityBoundTransitionBusV3::new(BusIndex::from(
                TRANSITION_BUS,
            )),
            whir_bus: MultiConstraintWhirCompletionBus::new(BusIndex::from(WHIR_BUS)),
            completion_bus: SetupPcsAuthorityCompletionBusV3::new(BusIndex::from(COMPLETION_BUS)),
            certificate_bus: SetupPcsAuthorityTransitionCertificateBusV3::new(BusIndex::from(
                CERTIFICATE_BUS,
            )),
        }
    }

    fn fixture_trace(
        record: &SetupPcsAuthorityCompletionRecordV3,
        authority: &SetupPcsAuthorityCompletionTraceV3,
    ) -> RowMajorMatrix<F> {
        let width = core::mem::size_of::<AuthorityFixtureCols<u8>>();
        let height = record.transitions.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(width * height);
        for index in 0..record.transitions.len() {
            let cols: &mut AuthorityFixtureCols<F> =
                values[index * width..(index + 1) * width].borrow_mut();
            cols.batch_enabled = F::from_bool(index == 0);
            cols.transition_enabled = F::ONE;
            cols.whir_enabled = F::from_bool(index + 1 == record.transitions.len());
            cols.completion_enabled = cols.whir_enabled;
            cols.certificate_enabled = F::ONE;
            cols.whir_outer_proof_idx = F::ZERO;
            if index == 0 {
                cols.batch = record.batch.clone();
            }
            cols.transition = record.transitions[index].clone();
            if index + 1 == record.transitions.len() {
                cols.whir = record.whir.clone();
                cols.completion = authority.completion.clone();
            }
            cols.certificate = authority.certificates[index].clone();
        }
        RowMajorMatrix::new(values, width)
    }

    fn poseidon_trace(records: &[SetupPcsAuthorityPoseidonRecordV3]) -> RowMajorMatrix<F> {
        let width = core::mem::size_of::<PoseidonFixtureCols<u8>>();
        let height = records.len().next_power_of_two().max(2);
        let mut values = F::zero_vec(width * height);
        for (index, record) in records.iter().enumerate() {
            let cols: &mut PoseidonFixtureCols<F> =
                values[index * width..(index + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.input = record.input;
            cols.output = record.output;
        }
        RowMajorMatrix::new(values, width)
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

    struct Composition {
        airs: Vec<AirRef<SC>>,
        matrices: Vec<RowMajorMatrix<F>>,
    }

    fn composition() -> Composition {
        let completion = completion_air();
        let record = record(&completion);
        let trace = generate_setup_pcs_authority_completion_trace_v3(&completion, &record)
            .expect("completion trace");
        Composition {
            airs: vec![
                Arc::new(completion),
                Arc::new(fixture_air()),
                Arc::new(PoseidonFixtureAir(Poseidon2CompressBus::new(
                    BusIndex::from(COMPRESS_BUS),
                ))),
            ],
            matrices: vec![
                trace.matrix.clone(),
                fixture_trace(&record, &trace),
                poseidon_trace(&trace.poseidon),
            ],
        }
    }

    fn check_composition(composition: &Composition) {
        let preprocessed_owned = composition
            .airs
            .iter()
            .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
            .collect::<Vec<_>>();
        for ((air, matrix), preprocessed) in composition
            .airs
            .iter()
            .zip(&composition.matrices)
            .zip(&preprocessed_owned)
        {
            check_constraints::<_, SC>(
                air.as_ref(),
                &air.name(),
                &preprocessed.as_ref().map(RowMajorMatrix::as_view),
                &[matrix.as_view()],
                &[],
            );
        }
        let interactions = composition
            .airs
            .iter()
            .map(|air| symbolic_interactions(air.as_ref()))
            .collect::<Vec<_>>();
        let preprocessed = preprocessed_owned
            .iter()
            .map(|matrix| matrix.as_ref().map(RowMajorMatrix::as_view))
            .collect::<Vec<_>>();
        let mains = composition
            .matrices
            .iter()
            .map(|matrix| vec![matrix.as_view()])
            .collect::<Vec<_>>();
        check_logup(
            &composition
                .airs
                .iter()
                .map(|air| air.name())
                .collect::<Vec<_>>(),
            &interactions,
            &preprocessed,
            &mains,
            &vec![Vec::new(); composition.airs.len()],
        );
    }

    fn fixture_row_mut(
        composition: &mut Composition,
        row_index: usize,
    ) -> &mut AuthorityFixtureCols<F> {
        let width = core::mem::size_of::<AuthorityFixtureCols<u8>>();
        composition.matrices[1].values[row_index * width..(row_index + 1) * width].borrow_mut()
    }

    fn assert_unbalanced(mutator: impl FnOnce(&mut Composition)) {
        let mut composition = composition();
        mutator(&mut composition);
        assert!(
            std::panic::catch_unwind(AssertUnwindSafe(|| check_composition(&composition))).is_err(),
            "mutation unexpectedly preserved all authority interactions"
        );
    }

    #[test]
    fn completion_is_balanced_only_with_whir_and_all_statements() {
        let composition = composition();
        check_composition(&composition);

        let completion_interactions = symbolic_interactions(composition.airs[0].as_ref());
        let count = |bus: u16| {
            completion_interactions
                .iter()
                .filter(|interaction| interaction.bus_index == bus)
                .count()
        };
        assert_eq!(count(BATCH_BUS), 1, "one batch-statement receive");
        assert_eq!(count(TRANSITION_BUS), 1, "one transition-stream receive");
        assert_eq!(count(WHIR_BUS), 1, "one terminal WHIR receive");
        assert_eq!(count(COMPLETION_BUS), 1, "one History completion send");
        assert_eq!(count(CERTIFICATE_BUS), 1, "one certificate-stream send");
    }

    #[test]
    fn whir_completion_mutations_break_authority() {
        let last = profile().transition_count() - 1;
        assert_unbalanced(|composition| fixture_row_mut(composition, last).whir_enabled = F::ZERO);
        assert_unbalanced(|composition| {
            fixture_row_mut(composition, last).whir_outer_proof_idx = F::ONE;
        });
        assert_unbalanced(|composition| {
            fixture_row_mut(composition, last).whir.proof_idx = F::ONE;
        });
        assert_unbalanced(|composition| {
            fixture_row_mut(composition, last).whir.class_index = F::ONE;
        });
        assert_unbalanced(|composition| {
            fixture_row_mut(composition, last).whir.end_tidx += F::ONE;
        });
        assert_unbalanced(|composition| {
            fixture_row_mut(composition, last).whir.sample_count += F::ONE;
        });
        assert_unbalanced(|composition| {
            fixture_row_mut(composition, last).whir.state[7] += F::ONE;
        });
        assert_unbalanced(|composition| {
            fixture_row_mut(composition, last).whir.final_aggregate[2] += F::ONE;
        });
        assert_unbalanced(|composition| {
            fixture_row_mut(composition, last).whir.final_claim[1] += F::ONE;
        });
    }

    #[test]
    fn statement_order_and_fixed_key_mutations_break_authority() {
        assert_unbalanced(|composition| {
            let width = core::mem::size_of::<AuthorityFixtureCols<u8>>();
            let (first, rest) = composition.matrices[1].values.split_at_mut(width);
            let row0: &mut AuthorityFixtureCols<F> = first.borrow_mut();
            let row1: &mut AuthorityFixtureCols<F> = rest[..width].borrow_mut();
            // Permutation-bus row order is intentionally irrelevant.  A
            // protocol reorder means assigning transition payload 1 to the
            // setup-fixed index 0 while retaining the indexed envelope.
            core::mem::swap(
                &mut row0.transition.canonical_claims_digest,
                &mut row1.transition.canonical_claims_digest,
            );
            core::mem::swap(
                &mut row0.transition.transition_statement_digest,
                &mut row1.transition.transition_statement_digest,
            );
            core::mem::swap(
                &mut row0.transition.setup_openings_pre_global_digest,
                &mut row1.transition.setup_openings_pre_global_digest,
            );
        });
        assert_unbalanced(|composition| {
            fixture_row_mut(composition, 1).transition_enabled = F::ZERO;
        });
        assert_unbalanced(|composition| {
            fixture_row_mut(composition, 0).batch.profile_digest[0] += F::ONE;
        });
        assert_unbalanced(|composition| {
            fixture_row_mut(composition, 0).batch.aggregation_vk_digest[0] += F::ONE;
        });
        assert_unbalanced(|composition| {
            fixture_row_mut(composition, 0).batch.relation_digest[0] += F::ONE;
        });
        assert_unbalanced(|composition| {
            fixture_row_mut(composition, 0).batch.batch_digest[0] += F::ONE;
        });
    }

    #[test]
    fn output_global_binding_mutation_breaks_authority() {
        let last = profile().transition_count() - 1;
        assert_unbalanced(|composition| {
            fixture_row_mut(composition, last)
                .completion
                .global_binding_digest[0] += F::ONE;
        });
        assert_unbalanced(|composition| {
            fixture_row_mut(composition, 0)
                .certificate
                .global_binding_digest[0] += F::ONE;
        });
    }
}

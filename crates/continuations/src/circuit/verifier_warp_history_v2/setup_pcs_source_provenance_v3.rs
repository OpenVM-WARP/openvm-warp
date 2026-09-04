//! Constrained source provenance for terminal setup-PCS authority.
//!
//! This AIR joins three independently constrained statements:
//!
//! - the direct SWIRL source manifest;
//! - the already-checked final LogUp transcript checkpoint;
//! - the fixed multi-AIR source boundary, including its one-shot systematic WARP message opening.
//!
//! The one-shot point is hashed into the source receipt.  It is not and cannot
//! be substituted for the setup PLE point, which remains on the independent
//! `FixedSetupOpeningPointBusV2` path.

use core::borrow::{Borrow, BorrowMut};

use openvm_recursion_circuit::{
    bus::{Poseidon2CompressBus, Poseidon2CompressMessage},
    define_typed_permutation_bus,
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
    CertifiedFixedMultiAirSourceBusV2, CertifiedFixedMultiAirSourceMessageV2,
    FIXED_MULTI_AIR_SOURCE_CAPACITY_V4, VERIFIER_WARP_HISTORY_MAX_TRANSITIONS_V2,
};
use crate::circuit::native_warp_history_v19::{
    SetupPcsSourceCheckpointBusV3, SetupPcsSourceCheckpointMessageV3, SetupPcsSourceManifestBusV3,
    SetupPcsSourceManifestMessageV3, SetupPcsSourceProvenanceBusV3,
    SetupPcsSourceProvenanceMessageV3, MAX_RAW_MESSAGE_POINT_LEN_V19,
    SETUP_PCS_SOURCE_PROVENANCE_MULTIPLICITY_V3, SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3,
    TRANSCRIPT_WIDTH_V19,
};

pub const SETUP_PCS_SOURCE_MANIFEST_START_TAG_V3: u32 = 0x5350_4d01;
pub const SETUP_PCS_SOURCE_MANIFEST_END_TAG_V3: u32 = 0x5350_4d02;
pub const SETUP_PCS_SOURCE_CHECKPOINT_START_TAG_V3: u32 = 0x5350_4b01;
pub const SETUP_PCS_SOURCE_CHECKPOINT_END_TAG_V3: u32 = 0x5350_4b02;
pub const SETUP_PCS_SOURCE_RECEIPT_START_TAG_V3: u32 = 0x5350_5201;
pub const SETUP_PCS_SOURCE_RECEIPT_POINT_TAG_V3: u32 = 0x5350_5202;
pub const SETUP_PCS_SOURCE_RECEIPT_COORD_TAG_V3: u32 = 0x5350_5203;
pub const SETUP_PCS_SOURCE_RECEIPT_VALUE_TAG_V3: u32 = 0x5350_5204;
pub const SETUP_PCS_SOURCE_RECEIPT_ENDPOINT_TAG_V3: u32 = 0x5350_5205;
pub const SETUP_PCS_SOURCE_RECEIPT_END_TAG_V3: u32 = 0x5350_5206;
/// The fixed-capacity HLeaf boundary is one external consumer for the two
/// internal V3 setup-authority consumers; the source-resume bridge is the
/// second external consumer.  The legacy direct V3 topology exposes all
/// three consumers directly and therefore retains multiplicity three.
pub const SETUP_PCS_SOURCE_PROVENANCE_MULTIPLICITY_V4: u32 = 2;

const MANIFEST_HASH_SLOTS_V3: usize = 8;
const CHECKPOINT_HASH_SLOTS_V3: usize = 10;
const RECEIPT_HASH_SLOTS_V3: usize = MAX_RAW_MESSAGE_POINT_LEN_V19 + 8;

/// Statement emitted by the setup-authority statement verifier after it has
/// consumed one provenance receipt, the genuine setup PLE point, and the
/// exact ordered setup claims.  The bridge consumes this edge instead of
/// trusting a host-copied transition digest.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug, PartialEq, Eq)]
pub struct SetupPcsAuthorityTransitionStatementMessageV3<T> {
    pub protocol_version: T,
    pub transition_index: [T; 2],
    pub source_receipt_digest: [T; DIGEST_SIZE],
    pub transition_statement_digest: [T; DIGEST_SIZE],
}
define_typed_permutation_bus!(
    SetupPcsAuthorityTransitionStatementBusV3,
    SetupPcsAuthorityTransitionStatementMessageV3
);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetupPcsSourceProvenanceProfileV3 {
    pub transition_count: usize,
    pub output_multiplicity: u32,
}

impl SetupPcsSourceProvenanceProfileV3 {
    pub fn validate(&self) -> Result<(), SetupPcsSourceProvenanceErrorV3> {
        if self.transition_count == 0
            || self.transition_count > VERIFIER_WARP_HISTORY_MAX_TRANSITIONS_V2
        {
            return Err(SetupPcsSourceProvenanceErrorV3::TransitionCount);
        }
        // V3 has exactly two independent consumers: the terminal claim bridge and the
        // authority statement/transcript owner.  Allowing an arbitrary nonzero fanout would let
        // a production composition silently omit either obligation or introduce an unowned
        // authority edge while still passing profile validation.
        if self.output_multiplicity != SETUP_PCS_SOURCE_PROVENANCE_MULTIPLICITY_V3 {
            return Err(SetupPcsSourceProvenanceErrorV3::OutputMultiplicity);
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct SetupPcsSourceProvenancePrepColsV3<T> {
    active: T,
    transition_index: [T; 2],
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct SetupPcsSourceProvenanceColsV3<T> {
    pub active: T,
    pub manifest: SetupPcsSourceManifestMessageV3<T>,
    pub checkpoint: SetupPcsSourceCheckpointMessageV3<T>,
    pub fixed_source: CertifiedFixedMultiAirSourceMessageV2<T>,
    pub manifest_hashes: [[T; DIGEST_SIZE]; MANIFEST_HASH_SLOTS_V3],
    pub checkpoint_hashes: [[T; DIGEST_SIZE]; CHECKPOINT_HASH_SLOTS_V3],
    pub receipt_hashes: [[T; DIGEST_SIZE]; RECEIPT_HASH_SLOTS_V3],
}

#[derive(Clone, Debug)]
pub struct SetupPcsSourceProvenanceAirV3 {
    pub profile: SetupPcsSourceProvenanceProfileV3,
    pub manifest_bus: SetupPcsSourceManifestBusV3,
    pub checkpoint_bus: SetupPcsSourceCheckpointBusV3,
    pub fixed_source_bus: CertifiedFixedMultiAirSourceBusV2,
    pub provenance_bus: SetupPcsSourceProvenanceBusV3,
    pub compress_bus: Poseidon2CompressBus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SetupPcsSourceProvenanceErrorV3 {
    TransitionCount,
    OutputMultiplicity,
    RecordCount,
    TransitionIndex(usize),
    ProtocolVersion(usize),
    ManifestCheckpointMismatch(usize),
    ManifestSourceMismatch(usize),
    CheckpointSourceMismatch(usize),
    SeedRelationMismatch(usize),
}

impl BaseAir<F> for SetupPcsSourceProvenanceAirV3 {
    fn width(&self) -> usize {
        core::mem::size_of::<SetupPcsSourceProvenanceColsV3<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.profile
            .validate()
            .expect("invalid setup PCS source provenance profile");
        let width = core::mem::size_of::<SetupPcsSourceProvenancePrepColsV3<u8>>();
        let height = self.profile.transition_count.next_power_of_two().max(2);
        let mut values = F::zero_vec(width * height);
        for transition in 0..self.profile.transition_count {
            let cols: &mut SetupPcsSourceProvenancePrepColsV3<F> =
                values[transition * width..(transition + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.transition_index = split_u32(transition as u32);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for SetupPcsSourceProvenanceAirV3 {}
impl PartitionedBaseAir<F> for SetupPcsSourceProvenanceAirV3 {}

fn split_u32(value: u32) -> [F; 2] {
    [F::from_u16(value as u16), F::from_u16((value >> 16) as u16)]
}

fn zero_expr<AB: AirBuilder<F = F>>() -> [AB::Expr; DIGEST_SIZE] {
    core::array::from_fn(|_| AB::Expr::ZERO)
}

fn assert_array_eq<AB, const N: usize>(
    builder: &mut AB,
    enabled: impl Into<AB::Expr> + Clone,
    left: [impl Into<AB::Expr>; N],
    right: [impl Into<AB::Expr>; N],
) where
    AB: AirBuilder<F = F>,
{
    for (left, right) in left.into_iter().zip(right) {
        builder
            .when(enabled.clone())
            .assert_eq(left.into(), right.into());
    }
}

fn compress<AB>(
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

fn tagged_ext_expr<AB: AirBuilder<F = F>>(
    tag: u32,
    value: &[AB::Var; D_EF],
) -> [AB::Expr; DIGEST_SIZE]
where
    AB::Var: Copy,
{
    core::array::from_fn(|index| match index {
        0 => AB::Expr::from_u32(tag),
        1..=D_EF => value[index - 1].into(),
        _ => AB::Expr::ZERO,
    })
}

impl<AB> Air<AB> for SetupPcsSourceProvenanceAirV3
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.profile.validate().is_ok());
        let prep = builder.preprocessed();
        let prep_row = prep.row_slice(0).expect("source provenance prep row");
        let prep: &SetupPcsSourceProvenancePrepColsV3<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("source provenance row");
        let local: &SetupPcsSourceProvenanceColsV3<AB::Var> = (*row).borrow();
        builder.assert_bool(prep.active);
        builder.assert_bool(local.active);
        builder.assert_eq(local.active, prep.active);
        let enabled = AB::Expr::from(prep.active);
        for &cell in row.iter() {
            builder
                .when(AB::Expr::ONE - enabled.clone())
                .assert_zero(cell);
        }

        for protocol in [
            local.manifest.protocol_version,
            local.checkpoint.protocol_version,
        ] {
            builder.when(enabled.clone()).assert_eq(
                protocol,
                AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            );
        }
        assert_array_eq(
            builder,
            enabled.clone(),
            local.manifest.transition_index,
            prep.transition_index,
        );
        assert_array_eq(
            builder,
            enabled.clone(),
            local.checkpoint.transition_index,
            prep.transition_index,
        );
        builder.when(enabled.clone()).assert_eq(
            local.fixed_source.proof_index,
            AB::Expr::from(prep.transition_index[0])
                + AB::Expr::from_u32(1 << 16) * AB::Expr::from(prep.transition_index[1]),
        );
        assert_array_eq(
            builder,
            enabled.clone(),
            local.manifest.segment_index,
            local.checkpoint.segment_index,
        );
        assert_array_eq(
            builder,
            enabled.clone(),
            local.manifest.segment_index,
            [
                local.fixed_source.segment_index_lo,
                local.fixed_source.segment_index_hi,
            ],
        );

        for (left, right) in [
            (
                &local.manifest.app_vk_digest,
                &local.checkpoint.app_vk_digest,
            ),
            (
                &local.manifest.source_forest_root,
                &local.checkpoint.source_forest_root,
            ),
            (
                &local.manifest.segment_openings_digest,
                &local.checkpoint.segment_openings_digest,
            ),
        ] {
            assert_array_eq(builder, enabled.clone(), *left, *right);
        }
        for (left, right) in [
            (
                &local.manifest.app_vk_digest,
                &local.fixed_source.app_vk_digest,
            ),
            (
                &local.manifest.relation_digest,
                &local.fixed_source.relation_digest,
            ),
            (&local.manifest.source_root, &local.fixed_source.source_root),
            (
                &local.manifest.source_forest_root,
                &local.fixed_source.source_forest_root,
            ),
            (
                &local.manifest.segment_openings_digest,
                &local.fixed_source.segment_openings_digest,
            ),
        ] {
            assert_array_eq(builder, enabled.clone(), *left, *right);
        }
        for (left, right) in [
            (
                &local.checkpoint.app_vk_digest,
                &local.fixed_source.app_vk_digest,
            ),
            (
                &local.checkpoint.source_forest_root,
                &local.fixed_source.source_forest_root,
            ),
            (
                &local.checkpoint.segment_openings_digest,
                &local.fixed_source.segment_openings_digest,
            ),
        ] {
            assert_array_eq(builder, enabled.clone(), *left, *right);
        }
        assert_array_eq(
            builder,
            enabled.clone(),
            local.checkpoint.verifier_endpoint,
            local.fixed_source.verifier_endpoint,
        );

        let transition = prep.transition_index.map(Into::into);
        let segment = local.manifest.segment_index.map(Into::into);
        let manifest_header: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_SOURCE_MANIFEST_START_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition[0].clone(),
            transition[1].clone(),
            segment[0].clone(),
            segment[1].clone(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        compress(
            self.compress_bus,
            builder,
            zero_expr::<AB>(),
            manifest_header,
            local.manifest_hashes[0],
            enabled.clone(),
        );
        let manifest_blocks = [
            local.manifest.app_vk_digest,
            local.manifest.relation_digest,
            local.manifest.source_root,
            local.manifest.source_instance_digest,
            local.manifest.source_forest_root,
            local.manifest.segment_openings_digest,
        ];
        for (index, block) in manifest_blocks.into_iter().enumerate() {
            compress(
                self.compress_bus,
                builder,
                local.manifest_hashes[index],
                block,
                local.manifest_hashes[index + 1],
                enabled.clone(),
            );
        }
        let manifest_end: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_SOURCE_MANIFEST_END_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition[0].clone(),
            transition[1].clone(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        compress(
            self.compress_bus,
            builder,
            local.manifest_hashes[6],
            manifest_end,
            local.manifest_hashes[7],
            enabled.clone(),
        );

        let checkpoint_header: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_SOURCE_CHECKPOINT_START_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition[0].clone(),
            transition[1].clone(),
            local.checkpoint.end_tidx[0].into(),
            local.checkpoint.end_tidx[1].into(),
            local.checkpoint.end_sample_count.into(),
            AB::Expr::ZERO,
        ];
        compress(
            self.compress_bus,
            builder,
            zero_expr::<AB>(),
            checkpoint_header,
            local.checkpoint_hashes[0],
            enabled.clone(),
        );
        let checkpoint_blocks: [[AB::Expr; DIGEST_SIZE]; 8] = [
            [
                segment[0].clone(),
                segment[1].clone(),
                AB::Expr::ZERO,
                AB::Expr::ZERO,
                AB::Expr::ZERO,
                AB::Expr::ZERO,
                AB::Expr::ZERO,
                AB::Expr::ZERO,
            ],
            local.checkpoint.app_vk_digest.map(Into::into),
            local.checkpoint.source_forest_root.map(Into::into),
            local.checkpoint.segment_openings_digest.map(Into::into),
            tagged_ext_expr::<AB>(
                SETUP_PCS_SOURCE_RECEIPT_ENDPOINT_TAG_V3,
                &local.checkpoint.verifier_endpoint,
            ),
            core::array::from_fn(|index| local.checkpoint.end_state[index].into()),
            core::array::from_fn(|index| local.checkpoint.end_state[index + DIGEST_SIZE].into()),
            local.checkpoint.logup_history_digest.map(Into::into),
        ];
        for (index, block) in checkpoint_blocks.into_iter().enumerate() {
            compress(
                self.compress_bus,
                builder,
                local.checkpoint_hashes[index],
                block,
                local.checkpoint_hashes[index + 1],
                enabled.clone(),
            );
        }
        let checkpoint_end: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_SOURCE_CHECKPOINT_END_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition[0].clone(),
            transition[1].clone(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        compress(
            self.compress_bus,
            builder,
            local.checkpoint_hashes[8],
            checkpoint_end,
            local.checkpoint_hashes[9],
            enabled.clone(),
        );

        let receipt_header: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_SOURCE_RECEIPT_START_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition[0].clone(),
            transition[1].clone(),
            segment[0].clone(),
            segment[1].clone(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        compress(
            self.compress_bus,
            builder,
            zero_expr::<AB>(),
            receipt_header,
            local.receipt_hashes[0],
            enabled.clone(),
        );
        for (index, block) in [
            local.manifest_hashes[7],
            local.checkpoint_hashes[9],
            local.checkpoint.logup_history_digest,
        ]
        .into_iter()
        .enumerate()
        {
            compress(
                self.compress_bus,
                builder,
                local.receipt_hashes[index],
                block,
                local.receipt_hashes[index + 1],
                enabled.clone(),
            );
        }
        let point_meta: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_SOURCE_RECEIPT_POINT_TAG_V3),
            local.fixed_source.point_len.into(),
            AB::Expr::from_usize(MAX_RAW_MESSAGE_POINT_LEN_V19),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        compress(
            self.compress_bus,
            builder,
            local.receipt_hashes[3],
            point_meta,
            local.receipt_hashes[4],
            enabled.clone(),
        );
        for coordinate in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
            let block: [AB::Expr; DIGEST_SIZE] = core::array::from_fn(|index| {
                if index == 0 {
                    AB::Expr::from_u32(SETUP_PCS_SOURCE_RECEIPT_COORD_TAG_V3)
                } else if index == 1 {
                    AB::Expr::from_usize(coordinate)
                } else if index < 2 + D_EF {
                    local.fixed_source.point[coordinate][index - 2].into()
                } else {
                    AB::Expr::ZERO
                }
            });
            compress(
                self.compress_bus,
                builder,
                local.receipt_hashes[4 + coordinate],
                block,
                local.receipt_hashes[5 + coordinate],
                enabled.clone(),
            );
        }
        let value_index = 5 + MAX_RAW_MESSAGE_POINT_LEN_V19;
        compress(
            self.compress_bus,
            builder,
            local.receipt_hashes[value_index - 1],
            tagged_ext_expr::<AB>(
                SETUP_PCS_SOURCE_RECEIPT_VALUE_TAG_V3,
                &local.fixed_source.value,
            ),
            local.receipt_hashes[value_index],
            enabled.clone(),
        );
        compress(
            self.compress_bus,
            builder,
            local.receipt_hashes[value_index],
            tagged_ext_expr::<AB>(
                SETUP_PCS_SOURCE_RECEIPT_ENDPOINT_TAG_V3,
                &local.fixed_source.verifier_endpoint,
            ),
            local.receipt_hashes[value_index + 1],
            enabled.clone(),
        );
        let receipt_end: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_SOURCE_RECEIPT_END_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition[0].clone(),
            transition[1].clone(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        compress(
            self.compress_bus,
            builder,
            local.receipt_hashes[value_index + 1],
            receipt_end,
            local.receipt_hashes[value_index + 2],
            enabled.clone(),
        );

        self.manifest_bus
            .receive(builder, local.manifest.clone(), enabled.clone());
        self.checkpoint_bus
            .receive(builder, local.checkpoint.clone(), enabled.clone());
        self.fixed_source_bus
            .receive(builder, local.fixed_source.clone(), enabled.clone());
        self.provenance_bus.send(
            builder,
            SetupPcsSourceProvenanceMessageV3 {
                protocol_version: AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
                transition_index: prep.transition_index.map(Into::into),
                segment_index: local.manifest.segment_index.map(Into::into),
                app_vk_digest: local.manifest.app_vk_digest.map(Into::into),
                relation_digest: local.manifest.relation_digest.map(Into::into),
                source_root: local.manifest.source_root.map(Into::into),
                source_instance_digest: local.manifest.source_instance_digest.map(Into::into),
                source_forest_root: local.manifest.source_forest_root.map(Into::into),
                segment_openings_digest: local.manifest.segment_openings_digest.map(Into::into),
                source_checkpoint_digest: local.checkpoint_hashes[9].map(Into::into),
                source_manifest_digest: local.manifest_hashes[7].map(Into::into),
                source_receipt_digest: local.receipt_hashes[value_index + 2].map(Into::into),
                end_tidx: local.checkpoint.end_tidx.map(Into::into),
                end_sample_count: local.checkpoint.end_sample_count.into(),
                end_state: local.checkpoint.end_state.map(Into::into),
            },
            enabled * AB::Expr::from_u32(self.profile.output_multiplicity),
        );
    }
}

/// Fixed-capacity provenance profile. Runtime occupancy is intentionally
/// absent from verifier-key material.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SetupPcsSourceProvenanceProfileV4 {
    pub output_multiplicity: u32,
    /// Relation/index authenticated by the valid setup seed and shared by the
    /// one batched, prior-bearing VACC package.
    pub seed_relation_digest: Digest,
}

impl SetupPcsSourceProvenanceProfileV4 {
    pub fn validate(self) -> Result<(), SetupPcsSourceProvenanceErrorV3> {
        if self.output_multiplicity != SETUP_PCS_SOURCE_PROVENANCE_MULTIPLICITY_V4 {
            return Err(SetupPcsSourceProvenanceErrorV3::OutputMultiplicity);
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct SetupPcsSourceProvenancePrepColsV4<T> {
    pub local_slot: T,
}

/// V4 retains the exact V3 receipt/hash columns; only activation moves from
/// setup to a constrained runtime prefix.
pub type SetupPcsSourceProvenanceColsV4<T> = SetupPcsSourceProvenanceColsV3<T>;

#[derive(Clone, Debug)]
pub struct SetupPcsSourceProvenanceAirV4 {
    pub profile: SetupPcsSourceProvenanceProfileV4,
    pub manifest_bus: SetupPcsSourceManifestBusV3,
    pub checkpoint_bus: SetupPcsSourceCheckpointBusV3,
    pub fixed_source_bus: CertifiedFixedMultiAirSourceBusV2,
    pub provenance_bus: SetupPcsSourceProvenanceBusV3,
    pub compress_bus: Poseidon2CompressBus,
}

impl BaseAir<F> for SetupPcsSourceProvenanceAirV4 {
    fn width(&self) -> usize {
        core::mem::size_of::<SetupPcsSourceProvenanceColsV4<u8>>()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        self.profile
            .validate()
            .expect("invalid fixed HLeaf source provenance profile");
        let width = core::mem::size_of::<SetupPcsSourceProvenancePrepColsV4<u8>>();
        let mut values = F::zero_vec(width * FIXED_MULTI_AIR_SOURCE_CAPACITY_V4);
        for slot in 0..FIXED_MULTI_AIR_SOURCE_CAPACITY_V4 {
            let cols: &mut SetupPcsSourceProvenancePrepColsV4<F> =
                values[slot * width..(slot + 1) * width].borrow_mut();
            cols.local_slot = F::from_usize(slot);
        }
        Some(RowMajorMatrix::new(values, width))
    }
}
impl BaseAirWithPublicValues<F> for SetupPcsSourceProvenanceAirV4 {}
impl PartitionedBaseAir<F> for SetupPcsSourceProvenanceAirV4 {}

impl<AB> Air<AB> for SetupPcsSourceProvenanceAirV4
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.profile.validate().is_ok());
        let prep = builder.preprocessed();
        let prep_row = prep.row_slice(0).expect("source provenance v4 prep row");
        let prep: &SetupPcsSourceProvenancePrepColsV4<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("source provenance v4 row");
        let next_row = main.row_slice(1).expect("source provenance v4 next row");
        let local: &SetupPcsSourceProvenanceColsV4<AB::Var> = (*row).borrow();
        let next: &SetupPcsSourceProvenanceColsV4<AB::Var> = (*next_row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder
            .when_transition()
            .assert_zero(next.active * (AB::Expr::ONE - local.active));
        let enabled = AB::Expr::from(local.active);
        for &cell in row.iter().skip(1) {
            builder
                .when(AB::Expr::ONE - enabled.clone())
                .assert_zero(cell);
        }

        let transition = [AB::Expr::from(prep.local_slot), AB::Expr::ZERO];
        for limb in 0..DIGEST_SIZE {
            builder.when(enabled.clone()).assert_eq(
                local.manifest.relation_digest[limb],
                AB::Expr::from(self.profile.seed_relation_digest[limb]),
            );
        }
        for protocol in [
            local.manifest.protocol_version,
            local.checkpoint.protocol_version,
        ] {
            builder.when(enabled.clone()).assert_eq(
                protocol,
                AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            );
        }
        for transition_index in [
            local.manifest.transition_index,
            local.checkpoint.transition_index,
        ] {
            builder
                .when(enabled.clone())
                .assert_eq(transition_index[0], prep.local_slot);
            builder
                .when(enabled.clone())
                .assert_zero(transition_index[1]);
        }
        builder
            .when(enabled.clone())
            .assert_eq(local.fixed_source.proof_index, prep.local_slot);
        assert_array_eq(
            builder,
            enabled.clone(),
            local.manifest.segment_index,
            local.checkpoint.segment_index,
        );
        assert_array_eq(
            builder,
            enabled.clone(),
            local.manifest.segment_index,
            [
                local.fixed_source.segment_index_lo,
                local.fixed_source.segment_index_hi,
            ],
        );
        builder
            .when(enabled.clone())
            .assert_eq(local.manifest.segment_index[0], prep.local_slot);
        builder
            .when(enabled.clone())
            .assert_zero(local.manifest.segment_index[1]);

        for (left, right) in [
            (
                &local.manifest.app_vk_digest,
                &local.checkpoint.app_vk_digest,
            ),
            (
                &local.manifest.source_forest_root,
                &local.checkpoint.source_forest_root,
            ),
            (
                &local.manifest.segment_openings_digest,
                &local.checkpoint.segment_openings_digest,
            ),
        ] {
            assert_array_eq(builder, enabled.clone(), *left, *right);
        }
        for (left, right) in [
            (
                &local.manifest.app_vk_digest,
                &local.fixed_source.app_vk_digest,
            ),
            (
                &local.manifest.relation_digest,
                &local.fixed_source.relation_digest,
            ),
            (&local.manifest.source_root, &local.fixed_source.source_root),
            (
                &local.manifest.source_forest_root,
                &local.fixed_source.source_forest_root,
            ),
            (
                &local.manifest.segment_openings_digest,
                &local.fixed_source.segment_openings_digest,
            ),
        ] {
            assert_array_eq(builder, enabled.clone(), *left, *right);
        }
        for (left, right) in [
            (
                &local.checkpoint.app_vk_digest,
                &local.fixed_source.app_vk_digest,
            ),
            (
                &local.checkpoint.source_forest_root,
                &local.fixed_source.source_forest_root,
            ),
            (
                &local.checkpoint.segment_openings_digest,
                &local.fixed_source.segment_openings_digest,
            ),
        ] {
            assert_array_eq(builder, enabled.clone(), *left, *right);
        }
        assert_array_eq(
            builder,
            enabled.clone(),
            local.checkpoint.verifier_endpoint,
            local.fixed_source.verifier_endpoint,
        );

        let segment = local.manifest.segment_index.map(Into::into);
        let manifest_header: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_SOURCE_MANIFEST_START_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition[0].clone(),
            transition[1].clone(),
            segment[0].clone(),
            segment[1].clone(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        compress(
            self.compress_bus,
            builder,
            zero_expr::<AB>(),
            manifest_header,
            local.manifest_hashes[0],
            enabled.clone(),
        );
        for (index, block) in [
            local.manifest.app_vk_digest,
            local.manifest.relation_digest,
            local.manifest.source_root,
            local.manifest.source_instance_digest,
            local.manifest.source_forest_root,
            local.manifest.segment_openings_digest,
        ]
        .into_iter()
        .enumerate()
        {
            compress(
                self.compress_bus,
                builder,
                local.manifest_hashes[index],
                block,
                local.manifest_hashes[index + 1],
                enabled.clone(),
            );
        }
        let manifest_end: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_SOURCE_MANIFEST_END_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition[0].clone(),
            transition[1].clone(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        compress(
            self.compress_bus,
            builder,
            local.manifest_hashes[6],
            manifest_end,
            local.manifest_hashes[7],
            enabled.clone(),
        );

        let checkpoint_header: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_SOURCE_CHECKPOINT_START_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition[0].clone(),
            transition[1].clone(),
            local.checkpoint.end_tidx[0].into(),
            local.checkpoint.end_tidx[1].into(),
            local.checkpoint.end_sample_count.into(),
            AB::Expr::ZERO,
        ];
        compress(
            self.compress_bus,
            builder,
            zero_expr::<AB>(),
            checkpoint_header,
            local.checkpoint_hashes[0],
            enabled.clone(),
        );
        let checkpoint_blocks: [[AB::Expr; DIGEST_SIZE]; 8] = [
            [
                segment[0].clone(),
                segment[1].clone(),
                AB::Expr::ZERO,
                AB::Expr::ZERO,
                AB::Expr::ZERO,
                AB::Expr::ZERO,
                AB::Expr::ZERO,
                AB::Expr::ZERO,
            ],
            local.checkpoint.app_vk_digest.map(Into::into),
            local.checkpoint.source_forest_root.map(Into::into),
            local.checkpoint.segment_openings_digest.map(Into::into),
            tagged_ext_expr::<AB>(
                SETUP_PCS_SOURCE_RECEIPT_ENDPOINT_TAG_V3,
                &local.checkpoint.verifier_endpoint,
            ),
            core::array::from_fn(|index| local.checkpoint.end_state[index].into()),
            core::array::from_fn(|index| local.checkpoint.end_state[index + DIGEST_SIZE].into()),
            local.checkpoint.logup_history_digest.map(Into::into),
        ];
        for (index, block) in checkpoint_blocks.into_iter().enumerate() {
            compress(
                self.compress_bus,
                builder,
                local.checkpoint_hashes[index],
                block,
                local.checkpoint_hashes[index + 1],
                enabled.clone(),
            );
        }
        let checkpoint_end: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_SOURCE_CHECKPOINT_END_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition[0].clone(),
            transition[1].clone(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        compress(
            self.compress_bus,
            builder,
            local.checkpoint_hashes[8],
            checkpoint_end,
            local.checkpoint_hashes[9],
            enabled.clone(),
        );

        let receipt_header: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_SOURCE_RECEIPT_START_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition[0].clone(),
            transition[1].clone(),
            segment[0].clone(),
            segment[1].clone(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        compress(
            self.compress_bus,
            builder,
            zero_expr::<AB>(),
            receipt_header,
            local.receipt_hashes[0],
            enabled.clone(),
        );
        for (index, block) in [
            local.manifest_hashes[7],
            local.checkpoint_hashes[9],
            local.checkpoint.logup_history_digest,
        ]
        .into_iter()
        .enumerate()
        {
            compress(
                self.compress_bus,
                builder,
                local.receipt_hashes[index],
                block,
                local.receipt_hashes[index + 1],
                enabled.clone(),
            );
        }
        let point_meta: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_SOURCE_RECEIPT_POINT_TAG_V3),
            local.fixed_source.point_len.into(),
            AB::Expr::from_usize(MAX_RAW_MESSAGE_POINT_LEN_V19),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        compress(
            self.compress_bus,
            builder,
            local.receipt_hashes[3],
            point_meta,
            local.receipt_hashes[4],
            enabled.clone(),
        );
        for coordinate in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
            let block: [AB::Expr; DIGEST_SIZE] = core::array::from_fn(|index| {
                if index == 0 {
                    AB::Expr::from_u32(SETUP_PCS_SOURCE_RECEIPT_COORD_TAG_V3)
                } else if index == 1 {
                    AB::Expr::from_usize(coordinate)
                } else if index < 2 + D_EF {
                    local.fixed_source.point[coordinate][index - 2].into()
                } else {
                    AB::Expr::ZERO
                }
            });
            compress(
                self.compress_bus,
                builder,
                local.receipt_hashes[4 + coordinate],
                block,
                local.receipt_hashes[5 + coordinate],
                enabled.clone(),
            );
        }
        let value_index = 5 + MAX_RAW_MESSAGE_POINT_LEN_V19;
        compress(
            self.compress_bus,
            builder,
            local.receipt_hashes[value_index - 1],
            tagged_ext_expr::<AB>(
                SETUP_PCS_SOURCE_RECEIPT_VALUE_TAG_V3,
                &local.fixed_source.value,
            ),
            local.receipt_hashes[value_index],
            enabled.clone(),
        );
        compress(
            self.compress_bus,
            builder,
            local.receipt_hashes[value_index],
            tagged_ext_expr::<AB>(
                SETUP_PCS_SOURCE_RECEIPT_ENDPOINT_TAG_V3,
                &local.fixed_source.verifier_endpoint,
            ),
            local.receipt_hashes[value_index + 1],
            enabled.clone(),
        );
        let receipt_end: [AB::Expr; DIGEST_SIZE] = [
            AB::Expr::from_u32(SETUP_PCS_SOURCE_RECEIPT_END_TAG_V3),
            AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition[0].clone(),
            transition[1].clone(),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ];
        compress(
            self.compress_bus,
            builder,
            local.receipt_hashes[value_index + 1],
            receipt_end,
            local.receipt_hashes[value_index + 2],
            enabled.clone(),
        );

        self.manifest_bus
            .receive(builder, local.manifest.clone(), enabled.clone());
        self.checkpoint_bus
            .receive(builder, local.checkpoint.clone(), enabled.clone());
        self.fixed_source_bus
            .receive(builder, local.fixed_source.clone(), enabled.clone());
        self.provenance_bus.send(
            builder,
            SetupPcsSourceProvenanceMessageV3 {
                protocol_version: AB::Expr::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
                transition_index: transition,
                segment_index: local.manifest.segment_index.map(Into::into),
                app_vk_digest: local.manifest.app_vk_digest.map(Into::into),
                relation_digest: local.manifest.relation_digest.map(Into::into),
                source_root: local.manifest.source_root.map(Into::into),
                source_instance_digest: local.manifest.source_instance_digest.map(Into::into),
                source_forest_root: local.manifest.source_forest_root.map(Into::into),
                segment_openings_digest: local.manifest.segment_openings_digest.map(Into::into),
                source_checkpoint_digest: local.checkpoint_hashes[9].map(Into::into),
                source_manifest_digest: local.manifest_hashes[7].map(Into::into),
                source_receipt_digest: local.receipt_hashes[value_index + 2].map(Into::into),
                end_tidx: local.checkpoint.end_tidx.map(Into::into),
                end_sample_count: local.checkpoint.end_sample_count.into(),
                end_state: local.checkpoint.end_state.map(Into::into),
            },
            enabled * AB::Expr::from_u32(self.profile.output_multiplicity),
        );
    }
}

#[derive(Clone, Debug)]
pub struct SetupPcsSourceProvenanceRecordV3 {
    pub manifest: SetupPcsSourceManifestMessageV3<F>,
    pub checkpoint: SetupPcsSourceCheckpointMessageV3<F>,
    pub fixed_source: CertifiedFixedMultiAirSourceMessageV2<F>,
}

#[derive(Debug)]
pub struct SetupPcsSourceProvenanceTraceV3 {
    pub matrix: RowMajorMatrix<F>,
    pub compression_inputs: Vec<[F; 2 * DIGEST_SIZE]>,
    pub provenances: Vec<SetupPcsSourceProvenanceMessageV3<F>>,
}

fn compress_host(inputs: &mut Vec<[F; 2 * DIGEST_SIZE]>, left: Digest, right: Digest) -> Digest {
    let mut input = [F::ZERO; 2 * DIGEST_SIZE];
    input[..DIGEST_SIZE].copy_from_slice(&left);
    input[DIGEST_SIZE..].copy_from_slice(&right);
    inputs.push(input);
    poseidon2_compress_with_capacity(left, right).0
}

fn tagged_ext(tag: u32, value: [F; D_EF]) -> Digest {
    let mut block = [F::ZERO; DIGEST_SIZE];
    block[0] = F::from_u32(tag);
    block[1..1 + D_EF].copy_from_slice(&value);
    block
}

fn validate_record(
    transition: usize,
    record: &SetupPcsSourceProvenanceRecordV3,
) -> Result<(), SetupPcsSourceProvenanceErrorV3> {
    let expected = split_u32(transition as u32);
    if record.manifest.protocol_version != F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3)
        || record.checkpoint.protocol_version
            != F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3)
    {
        return Err(SetupPcsSourceProvenanceErrorV3::ProtocolVersion(transition));
    }
    if record.manifest.transition_index != expected
        || record.checkpoint.transition_index != expected
        || record.fixed_source.proof_index != F::from_usize(transition)
    {
        return Err(SetupPcsSourceProvenanceErrorV3::TransitionIndex(transition));
    }
    if record.manifest.app_vk_digest != record.checkpoint.app_vk_digest
        || record.manifest.segment_index != record.checkpoint.segment_index
        || record.manifest.source_forest_root != record.checkpoint.source_forest_root
        || record.manifest.segment_openings_digest != record.checkpoint.segment_openings_digest
    {
        return Err(SetupPcsSourceProvenanceErrorV3::ManifestCheckpointMismatch(
            transition,
        ));
    }
    if record.manifest.app_vk_digest != record.fixed_source.app_vk_digest
        || record.manifest.segment_index
            != [
                record.fixed_source.segment_index_lo,
                record.fixed_source.segment_index_hi,
            ]
        || record.manifest.relation_digest != record.fixed_source.relation_digest
        || record.manifest.source_root != record.fixed_source.source_root
        || record.manifest.source_forest_root != record.fixed_source.source_forest_root
        || record.manifest.segment_openings_digest != record.fixed_source.segment_openings_digest
    {
        return Err(SetupPcsSourceProvenanceErrorV3::ManifestSourceMismatch(
            transition,
        ));
    }
    if record.checkpoint.app_vk_digest != record.fixed_source.app_vk_digest
        || record.checkpoint.source_forest_root != record.fixed_source.source_forest_root
        || record.checkpoint.segment_openings_digest != record.fixed_source.segment_openings_digest
        || record.checkpoint.verifier_endpoint != record.fixed_source.verifier_endpoint
    {
        return Err(SetupPcsSourceProvenanceErrorV3::CheckpointSourceMismatch(
            transition,
        ));
    }
    Ok(())
}

pub fn generate_setup_pcs_source_provenance_trace_v3(
    air: &SetupPcsSourceProvenanceAirV3,
    records: &[SetupPcsSourceProvenanceRecordV3],
) -> Result<SetupPcsSourceProvenanceTraceV3, SetupPcsSourceProvenanceErrorV3> {
    air.profile.validate()?;
    if records.len() != air.profile.transition_count {
        return Err(SetupPcsSourceProvenanceErrorV3::RecordCount);
    }
    let width = core::mem::size_of::<SetupPcsSourceProvenanceColsV3<u8>>();
    let height = records.len().next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    let mut compression_inputs = Vec::new();
    let mut provenances = Vec::with_capacity(records.len());
    for (transition, record) in records.iter().enumerate() {
        validate_record(transition, record)?;
        let cols: &mut SetupPcsSourceProvenanceColsV3<F> =
            values[transition * width..(transition + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.manifest = record.manifest.clone();
        cols.checkpoint = record.checkpoint.clone();
        cols.fixed_source = record.fixed_source.clone();
        let transition_limbs = split_u32(transition as u32);
        let segment_limbs = record.manifest.segment_index;

        let manifest_header = [
            F::from_u32(SETUP_PCS_SOURCE_MANIFEST_START_TAG_V3),
            F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition_limbs[0],
            transition_limbs[1],
            segment_limbs[0],
            segment_limbs[1],
            F::ZERO,
            F::ZERO,
        ];
        cols.manifest_hashes[0] = compress_host(
            &mut compression_inputs,
            [F::ZERO; DIGEST_SIZE],
            manifest_header,
        );
        for (index, block) in [
            record.manifest.app_vk_digest,
            record.manifest.relation_digest,
            record.manifest.source_root,
            record.manifest.source_instance_digest,
            record.manifest.source_forest_root,
            record.manifest.segment_openings_digest,
        ]
        .into_iter()
        .enumerate()
        {
            cols.manifest_hashes[index + 1] =
                compress_host(&mut compression_inputs, cols.manifest_hashes[index], block);
        }
        let manifest_end = [
            F::from_u32(SETUP_PCS_SOURCE_MANIFEST_END_TAG_V3),
            F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition_limbs[0],
            transition_limbs[1],
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
        ];
        cols.manifest_hashes[7] = compress_host(
            &mut compression_inputs,
            cols.manifest_hashes[6],
            manifest_end,
        );

        let checkpoint_header = [
            F::from_u32(SETUP_PCS_SOURCE_CHECKPOINT_START_TAG_V3),
            F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition_limbs[0],
            transition_limbs[1],
            record.checkpoint.end_tidx[0],
            record.checkpoint.end_tidx[1],
            record.checkpoint.end_sample_count,
            F::ZERO,
        ];
        cols.checkpoint_hashes[0] = compress_host(
            &mut compression_inputs,
            [F::ZERO; DIGEST_SIZE],
            checkpoint_header,
        );
        let state_left: Digest = record.checkpoint.end_state[..DIGEST_SIZE]
            .try_into()
            .expect("checkpoint left half");
        let state_right: Digest = record.checkpoint.end_state[DIGEST_SIZE..]
            .try_into()
            .expect("checkpoint right half");
        let segment_block = [
            segment_limbs[0],
            segment_limbs[1],
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
        ];
        for (index, block) in [
            segment_block,
            record.checkpoint.app_vk_digest,
            record.checkpoint.source_forest_root,
            record.checkpoint.segment_openings_digest,
            tagged_ext(
                SETUP_PCS_SOURCE_RECEIPT_ENDPOINT_TAG_V3,
                record.checkpoint.verifier_endpoint,
            ),
            state_left,
            state_right,
            record.checkpoint.logup_history_digest,
        ]
        .into_iter()
        .enumerate()
        {
            cols.checkpoint_hashes[index + 1] = compress_host(
                &mut compression_inputs,
                cols.checkpoint_hashes[index],
                block,
            );
        }
        let checkpoint_end = [
            F::from_u32(SETUP_PCS_SOURCE_CHECKPOINT_END_TAG_V3),
            F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition_limbs[0],
            transition_limbs[1],
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
        ];
        cols.checkpoint_hashes[9] = compress_host(
            &mut compression_inputs,
            cols.checkpoint_hashes[8],
            checkpoint_end,
        );

        let receipt_header = [
            F::from_u32(SETUP_PCS_SOURCE_RECEIPT_START_TAG_V3),
            F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition_limbs[0],
            transition_limbs[1],
            segment_limbs[0],
            segment_limbs[1],
            F::ZERO,
            F::ZERO,
        ];
        cols.receipt_hashes[0] = compress_host(
            &mut compression_inputs,
            [F::ZERO; DIGEST_SIZE],
            receipt_header,
        );
        for (index, block) in [
            cols.manifest_hashes[7],
            cols.checkpoint_hashes[9],
            record.checkpoint.logup_history_digest,
        ]
        .into_iter()
        .enumerate()
        {
            cols.receipt_hashes[index + 1] =
                compress_host(&mut compression_inputs, cols.receipt_hashes[index], block);
        }
        let point_meta = [
            F::from_u32(SETUP_PCS_SOURCE_RECEIPT_POINT_TAG_V3),
            record.fixed_source.point_len,
            F::from_usize(MAX_RAW_MESSAGE_POINT_LEN_V19),
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
        ];
        cols.receipt_hashes[4] =
            compress_host(&mut compression_inputs, cols.receipt_hashes[3], point_meta);
        for coordinate in 0..MAX_RAW_MESSAGE_POINT_LEN_V19 {
            let mut block = [F::ZERO; DIGEST_SIZE];
            block[0] = F::from_u32(SETUP_PCS_SOURCE_RECEIPT_COORD_TAG_V3);
            block[1] = F::from_usize(coordinate);
            block[2..2 + D_EF].copy_from_slice(&record.fixed_source.point[coordinate]);
            cols.receipt_hashes[5 + coordinate] = compress_host(
                &mut compression_inputs,
                cols.receipt_hashes[4 + coordinate],
                block,
            );
        }
        let value_index = 5 + MAX_RAW_MESSAGE_POINT_LEN_V19;
        cols.receipt_hashes[value_index] = compress_host(
            &mut compression_inputs,
            cols.receipt_hashes[value_index - 1],
            tagged_ext(
                SETUP_PCS_SOURCE_RECEIPT_VALUE_TAG_V3,
                record.fixed_source.value,
            ),
        );
        cols.receipt_hashes[value_index + 1] = compress_host(
            &mut compression_inputs,
            cols.receipt_hashes[value_index],
            tagged_ext(
                SETUP_PCS_SOURCE_RECEIPT_ENDPOINT_TAG_V3,
                record.fixed_source.verifier_endpoint,
            ),
        );
        let receipt_end = [
            F::from_u32(SETUP_PCS_SOURCE_RECEIPT_END_TAG_V3),
            F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition_limbs[0],
            transition_limbs[1],
            F::ZERO,
            F::ZERO,
            F::ZERO,
            F::ZERO,
        ];
        cols.receipt_hashes[value_index + 2] = compress_host(
            &mut compression_inputs,
            cols.receipt_hashes[value_index + 1],
            receipt_end,
        );
        provenances.push(SetupPcsSourceProvenanceMessageV3 {
            protocol_version: F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition_index: transition_limbs,
            segment_index: segment_limbs,
            app_vk_digest: record.manifest.app_vk_digest,
            relation_digest: record.manifest.relation_digest,
            source_root: record.manifest.source_root,
            source_instance_digest: record.manifest.source_instance_digest,
            source_forest_root: record.manifest.source_forest_root,
            segment_openings_digest: record.manifest.segment_openings_digest,
            source_checkpoint_digest: cols.checkpoint_hashes[9],
            source_manifest_digest: cols.manifest_hashes[7],
            source_receipt_digest: cols.receipt_hashes[value_index + 2],
            end_tidx: record.checkpoint.end_tidx,
            end_sample_count: record.checkpoint.end_sample_count,
            end_state: record.checkpoint.end_state,
        });
    }
    Ok(SetupPcsSourceProvenanceTraceV3 {
        matrix: RowMajorMatrix::new(values, width),
        compression_inputs,
        provenances,
    })
}

/// Generate the V4 fixed-height witness while reusing the exact V3 receipt
/// encoding. Only the active prefix is materialized; all remaining rows are
/// canonical zero.
pub fn generate_setup_pcs_source_provenance_trace_v4(
    air: &SetupPcsSourceProvenanceAirV4,
    records: &[SetupPcsSourceProvenanceRecordV3],
) -> Result<SetupPcsSourceProvenanceTraceV3, SetupPcsSourceProvenanceErrorV3> {
    air.profile.validate()?;
    if records.is_empty() || records.len() > FIXED_MULTI_AIR_SOURCE_CAPACITY_V4 {
        return Err(SetupPcsSourceProvenanceErrorV3::RecordCount);
    }
    if let Some((slot, _)) = records.iter().enumerate().find(|(_, record)| {
        record.manifest.relation_digest != air.profile.seed_relation_digest
            || record.fixed_source.relation_digest != air.profile.seed_relation_digest
    }) {
        return Err(SetupPcsSourceProvenanceErrorV3::SeedRelationMismatch(slot));
    }
    let legacy = SetupPcsSourceProvenanceAirV3 {
        profile: SetupPcsSourceProvenanceProfileV3 {
            transition_count: records.len(),
            // The V3 helper is used only to materialize the shared receipt and
            // Poseidon columns.  Those columns are independent of the
            // provenance-bus fanout.  Keep the helper under its own exact V3
            // protocol profile; the enclosing V4 AIR has already validated
            // and enforces its distinct two-consumer fanout above.
            output_multiplicity: SETUP_PCS_SOURCE_PROVENANCE_MULTIPLICITY_V3,
        },
        manifest_bus: air.manifest_bus,
        checkpoint_bus: air.checkpoint_bus,
        fixed_source_bus: air.fixed_source_bus,
        provenance_bus: air.provenance_bus,
        compress_bus: air.compress_bus,
    };
    let mut trace = generate_setup_pcs_source_provenance_trace_v3(&legacy, records)?;
    let width = trace.matrix.width();
    trace
        .matrix
        .values
        .resize(width * FIXED_MULTI_AIR_SOURCE_CAPACITY_V4, F::ZERO);
    trace.matrix = RowMajorMatrix::new(trace.matrix.values, width);
    Ok(trace)
}

const _: () = assert!(TRANSCRIPT_WIDTH_V19 == 2 * DIGEST_SIZE);

#[cfg(test)]
mod fixed_capacity_v4_tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use openvm_recursion_circuit::bus::Poseidon2CompressBus;
    use openvm_stark_backend::{
        air_builders::debug::check_constraints, interaction::BusIndex, p3_matrix::Matrix,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config as NativeSC;

    use super::*;

    fn record(slot: usize) -> SetupPcsSourceProvenanceRecordV3 {
        let transition = [F::from_usize(slot), F::ZERO];
        let manifest = SetupPcsSourceManifestMessageV3 {
            protocol_version: F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition_index: transition,
            segment_index: transition,
            app_vk_digest: [F::ZERO; DIGEST_SIZE],
            relation_digest: [F::ZERO; DIGEST_SIZE],
            source_root: [F::ZERO; DIGEST_SIZE],
            source_instance_digest: [F::ZERO; DIGEST_SIZE],
            source_forest_root: [F::ZERO; DIGEST_SIZE],
            segment_openings_digest: [F::ZERO; DIGEST_SIZE],
        };
        let checkpoint = SetupPcsSourceCheckpointMessageV3 {
            protocol_version: F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition_index: transition,
            segment_index: transition,
            app_vk_digest: [F::ZERO; DIGEST_SIZE],
            source_forest_root: [F::ZERO; DIGEST_SIZE],
            segment_openings_digest: [F::ZERO; DIGEST_SIZE],
            verifier_endpoint: [F::ZERO; D_EF],
            end_tidx: [F::from_usize(20 + slot), F::ZERO],
            end_sample_count: F::ZERO,
            end_state: [F::ZERO; TRANSCRIPT_WIDTH_V19],
            logup_history_digest: [F::ZERO; DIGEST_SIZE],
        };
        let fixed_source = CertifiedFixedMultiAirSourceMessageV2 {
            proof_index: F::from_usize(slot),
            segment_index_lo: F::from_usize(slot),
            segment_index_hi: F::ZERO,
            active_child_count: F::ONE,
            app_vk_digest: [F::ZERO; DIGEST_SIZE],
            relation_digest: [F::ZERO; DIGEST_SIZE],
            source_forest_root: [F::ZERO; DIGEST_SIZE],
            segment_openings_digest: [F::ZERO; DIGEST_SIZE],
            source_root: [F::ZERO; DIGEST_SIZE],
            point_len: F::ZERO,
            point: [[F::ZERO; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19],
            value: [F::ZERO; D_EF],
            verifier_endpoint: [F::ZERO; D_EF],
        };
        SetupPcsSourceProvenanceRecordV3 {
            manifest,
            checkpoint,
            fixed_source,
        }
    }

    fn air() -> SetupPcsSourceProvenanceAirV4 {
        SetupPcsSourceProvenanceAirV4 {
            profile: SetupPcsSourceProvenanceProfileV4 {
                output_multiplicity: SETUP_PCS_SOURCE_PROVENANCE_MULTIPLICITY_V4,
                seed_relation_digest: [F::ZERO; DIGEST_SIZE],
            },
            manifest_bus: SetupPcsSourceManifestBusV3::new(BusIndex::from(1u16)),
            checkpoint_bus: SetupPcsSourceCheckpointBusV3::new(BusIndex::from(2u16)),
            fixed_source_bus: CertifiedFixedMultiAirSourceBusV2::new(BusIndex::from(3u16)),
            provenance_bus: SetupPcsSourceProvenanceBusV3::new(BusIndex::from(4u16)),
            compress_bus: Poseidon2CompressBus::new(BusIndex::from(5u16)),
        }
    }

    fn check(air: &SetupPcsSourceProvenanceAirV4, matrix: &RowMajorMatrix<F>) {
        let prep = air.preprocessed_trace().expect("fixed provenance prep");
        check_constraints::<_, NativeSC>(
            air,
            "SetupPcsSourceProvenanceAirV4",
            &Some(prep.as_view()),
            &[matrix.as_view()],
            &[],
        );
    }

    #[test]
    fn every_history_leaf_occupancy_has_one_key_shape_and_zero_suffix() {
        let air = air();
        let prep = air.preprocessed_trace().unwrap();
        for occupancy in 1..=FIXED_MULTI_AIR_SOURCE_CAPACITY_V4 {
            let records = (0..occupancy).map(record).collect::<Vec<_>>();
            let trace = generate_setup_pcs_source_provenance_trace_v4(&air, &records).unwrap();
            assert_eq!(trace.matrix.width(), air.width());
            assert_eq!(trace.matrix.height(), FIXED_MULTI_AIR_SOURCE_CAPACITY_V4);
            assert!(trace.matrix.values[occupancy * air.width()..]
                .iter()
                .all(|value| *value == F::ZERO));
            assert_eq!(air.preprocessed_trace().unwrap().values, prep.values);
            check(&air, &trace.matrix);
        }
    }

    #[test]
    fn inactive_and_transition_selector_mutations_fail() {
        let air = air();
        let mut trace = generate_setup_pcs_source_provenance_trace_v4(&air, &[record(0)]).unwrap();
        let width = trace.matrix.width();
        let inactive: &mut SetupPcsSourceProvenanceColsV4<F> =
            trace.matrix.values[width..2 * width].borrow_mut();
        inactive.active = F::ONE;
        assert!(catch_unwind(AssertUnwindSafe(|| check(&air, &trace.matrix))).is_err());

        let mut wrong = record(0);
        wrong.manifest.transition_index[0] = F::ONE;
        assert!(matches!(
            generate_setup_pcs_source_provenance_trace_v4(&air, &[wrong]),
            Err(SetupPcsSourceProvenanceErrorV3::TransitionIndex(0))
        ));

        let mut wrong_seed = record(0);
        wrong_seed.manifest.relation_digest[0] = F::ONE;
        wrong_seed.fixed_source.relation_digest[0] = F::ONE;
        assert!(matches!(
            generate_setup_pcs_source_provenance_trace_v4(&air, &[wrong_seed]),
            Err(SetupPcsSourceProvenanceErrorV3::SeedRelationMismatch(0))
        ));
    }
}

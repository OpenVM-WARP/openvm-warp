//! Ordered receipt for deferred-SWIRL sources consumed by native WARP.
//!
//! This is an adapter around the ordinary recursive verifier.  It consumes
//! the verifier's authenticated proof-shape, public-value, checkpoint and
//! [`ReducedSwirlSourceAir`] exports, recomputes the SDK's canonical source
//! entry and ordered-manifest Poseidon transcripts, and publishes exactly one
//! [`ReducedSwirlSourceReceiptMessage`].  It never verifies the child WHIR
//! tail and it never treats a PCS opening as PESAT.

use std::sync::Arc;

use openvm_circuit::arch::{
    CONNECTOR_AIR_ID, MERKLE_AIR_ID, PROGRAM_AIR_ID, PROGRAM_CACHED_TRACE_INDEX,
};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_recursion_circuit::{
    bus::{
        AirPresenceBusMessage, CachedCommitBusMessage, FinalTranscriptStateMessage,
        HyperdimBusMessage, PreHashMessage, PublicValuesBusMessage, TranscriptBus,
        TranscriptBusMessage, TranscriptEndIndexMessage,
    },
    native_warp::{
        NativeWarpTranscriptArtifacts, NativeWarpTranscriptModule, RecursiveReducedSwirlClaim,
        ReducedSwirlSourceAir, ReducedSwirlSourceAuthorityBus, ReducedSwirlSourceAuthorityMessage,
        ReducedSwirlSourceBetaBus, ReducedSwirlSourceBetaMessage, ReducedSwirlSourceClaimBus,
        ReducedSwirlSourceClaimMessage, ReducedSwirlSourceOpeningBus,
        ReducedSwirlSourceOpeningMessage, ReducedSwirlSourcePointBus,
        ReducedSwirlSourcePointMessage, ReducedSwirlSourceProfile, ReducedSwirlSourceRecord,
        ReducedSwirlSourceRootBus, ReducedSwirlSourceRootMessage, ReducedSwirlSourceRootWidthBus,
        ReducedSwirlSourceRootWidthMessage,
    },
    system::{
        AggregationSubCircuit, BusIndexManager, BusInventory, VerifierConfig, VerifierSubCircuit,
        VerifierTailMode,
    },
    transcript::Poseidon2BusOwner,
};
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder},
    keygen::types::MultiStarkVerifyingKey,
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    transcript::{TranscriptHistory, TranscriptLog},
    AirRef, BaseAirWithPublicValues, FiatShamirTranscript, PartitionedBaseAir, StarkProtocolConfig,
    SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, BabyBearPoseidon2Config, Digest, DIGEST_SIZE, D_EF, EF, F,
};

use super::reduced_swirl_warp::{
    ReducedSwirlSourceReceiptBus, ReducedSwirlSourceReceiptMessage,
    REDUCED_SWIRL_WRAPPER_MAX_SOURCES, REDUCED_SWIRL_WRAPPER_PROTOCOL_VERSION,
};

/// Exact SDK domain used by `ReducedSwirlSourceManifestPrefix::digest_with_claim`.
pub const REDUCED_SWIRL_SOURCE_DIGEST_TAG: u32 = 0x5253_0101;
const OBS_FIELD_TAG: u32 = 0x5253_0201;
const OBS_DIGEST_TAG: u32 = 0x5253_0202;
const OBS_EXTENSION_TAG: u32 = 0x5253_0203;
const OBS_END_TAG: u32 = 0x5253_02ff;
const ENTRY_DOMAIN: u32 = REDUCED_SWIRL_SOURCE_DIGEST_TAG ^ 0x70;
const LAYOUT_DOMAIN: u32 = REDUCED_SWIRL_SOURCE_DIGEST_TAG ^ 0x10;
const PENDING_DOMAIN: u32 = REDUCED_SWIRL_SOURCE_DIGEST_TAG ^ 0x30;
const CLAIM_DOMAIN: u32 = REDUCED_SWIRL_SOURCE_DIGEST_TAG ^ 0x40;
const MANIFEST_TAG: &[u8] = b"openvm.native-warp.swirl-reduced-source.manifest.v3";

const SOURCE_HASHES_PER_SOURCE: usize = 4;
const LAYOUT_HASH_SLOT: usize = 0;
const PENDING_HASH_SLOT: usize = 1;
const CLAIM_HASH_SLOT: usize = 2;
const ENTRY_HASH_SLOT: usize = 3;

const ACTIVE: usize = 0;
const IS_LAST: usize = 1;
const IS_FIRST: usize = 2;
const SOURCE: usize = 3;
const SEGMENT: usize = 4;
const SOURCE_COUNT: usize = 5;
const ROOT_COUNT: usize = 6;
const OPENING_COUNT: usize = 7;
const CHECKPOINT_TIDX: usize = 8;
const CHECKPOINT_SAMPLES: usize = 9;
const CHECKPOINT_STATE: usize = CHECKPOINT_SAMPLES + D_EF;
const PROGRAM: usize = CHECKPOINT_STATE + POSEIDON2_WIDTH;
const INITIAL_PC: usize = PROGRAM + DIGEST_SIZE;
const INITIAL_ROOT: usize = INITIAL_PC + 1;
const FINAL_PC: usize = INITIAL_ROOT + DIGEST_SIZE;
const FINAL_ROOT: usize = FINAL_PC + 1;
const EXIT_CODE: usize = FINAL_ROOT + DIGEST_SIZE;
const IS_TERMINATE: usize = EXIT_CODE + 1;
const CHAIN_INITIAL_PC: usize = IS_TERMINATE + 1;
const CHAIN_INITIAL_ROOT: usize = CHAIN_INITIAL_PC + 1;
const LAYOUT_DIGEST: usize = CHAIN_INITIAL_ROOT + DIGEST_SIZE;
const PENDING_DIGEST: usize = LAYOUT_DIGEST + DIGEST_SIZE;
const CLAIM_DIGEST: usize = PENDING_DIGEST + DIGEST_SIZE;
const ENTRY_DIGEST: usize = CLAIM_DIGEST + DIGEST_SIZE;
const MANIFEST_DIGEST: usize = ENTRY_DIGEST + DIGEST_SIZE;
const THETA: usize = MANIFEST_DIGEST + DIGEST_SIZE;
const MU: usize = THETA + D_EF;
const ETA: usize = MU + D_EF;
const HEADER_WIDTH: usize = ETA + D_EF;

const LAYOUT_PRESENT: usize = 0;
const LAYOUT_SORT_INDEX: usize = 1;
const LAYOUT_N_ABS: usize = 2;
const LAYOUT_N_SIGN: usize = 3;
const LAYOUT_HEADER_WIDTH: usize = 4;

const ROOT_ACTIVE: usize = 0;
const ROOT_WIDTH: usize = 1;
const ROOT_DIGEST: usize = 2;
const ROOT_SLOT_WIDTH: usize = ROOT_DIGEST + DIGEST_SIZE;

const OPENING_ACTIVE: usize = 0;
const OPENING_ROOT: usize = 1;
const OPENING_COLUMN: usize = 2;
const OPENING_FIRST: usize = 3;
const OPENING_COLUMN_INVERSE: usize = 4;
const OPENING_GROUP_WIDTH: usize = 5;
const OPENING_VALUE: usize = 6;
const OPENING_SLOT_WIDTH: usize = OPENING_VALUE + D_EF;

/// VK-owned layout needed to authenticate the exact SDK `trace_vdata` hash.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlSourceReceiptProfile {
    pub source: ReducedSwirlSourceProfile,
    pub protocol_digest: Digest,
    pub child_vk_pre_hash: Digest,
    pub child_air_count: usize,
    pub child_l_skip: usize,
    /// Global cached indices, in each AIR's local cached-commit order.
    pub cached_global_indices: Vec<Vec<usize>>,
    pub suspend_exit_code: u32,
}

impl ReducedSwirlSourceReceiptProfile {
    pub fn validate(&self) -> Result<(), &'static str> {
        self.source.validate()?;
        if self.source.maximum_sources == 0
            || self.source.maximum_sources > REDUCED_SWIRL_WRAPPER_MAX_SOURCES as usize
            || self.child_air_count == 0
            || self.child_air_count > u16::MAX as usize
            || self.child_l_skip > u16::MAX as usize
            || self.cached_global_indices.len() != self.child_air_count
            || self.protocol_digest.iter().all(|value| *value == F::ZERO)
            || self.child_vk_pre_hash.iter().all(|value| *value == F::ZERO)
            || [
                self.source.maximum_roots_per_source,
                self.source.maximum_openings_per_source,
                self.source.l_skip,
                self.source.n_stack,
                self.source.log_blowup,
                self.source.log_commit_rows_per_query,
                self.source.log_message_len(),
                self.source.log_codeword_len(),
                self.source.rows_per_query(),
            ]
            .into_iter()
            .any(|value| value > u16::MAX as usize)
        {
            return Err("reduced-SWIRL source-receipt profile");
        }
        let mut seen = std::collections::BTreeSet::new();
        for indices in &self.cached_global_indices {
            if indices.len() > u16::MAX as usize {
                return Err("cached-commit count exceeds canonical u16");
            }
            for &index in indices {
                if index > u16::MAX as usize {
                    return Err("cached-commit global index exceeds canonical u16");
                }
                if !seen.insert(index) {
                    return Err("duplicate cached-commit global index");
                }
            }
        }
        Ok(())
    }

    fn layout_slot_width(&self, air: usize) -> usize {
        LAYOUT_HEADER_WIDTH + self.cached_global_indices[air].len() * DIGEST_SIZE
    }

    fn layout_offset(&self, air: usize) -> usize {
        HEADER_WIDTH
            + self.cached_global_indices[..air]
                .iter()
                .map(|indices| LAYOUT_HEADER_WIDTH + indices.len() * DIGEST_SIZE)
                .sum::<usize>()
    }

    fn roots_offset(&self) -> usize {
        self.layout_offset(self.child_air_count)
    }

    fn points_offset(&self) -> usize {
        self.roots_offset() + self.source.maximum_roots_per_source * ROOT_SLOT_WIDTH
    }

    fn betas_offset(&self) -> usize {
        self.points_offset() + self.source.log_message_len() * D_EF
    }

    fn openings_offset(&self) -> usize {
        self.betas_offset() + self.source.log_message_len() * D_EF
    }

    #[must_use]
    pub fn trace_width(&self) -> usize {
        self.openings_offset() + self.source.maximum_openings_per_source * OPENING_SLOT_WIDTH
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlReceiptLayoutEntry {
    pub log_height: Option<usize>,
    pub cached_commitments: Vec<Digest>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReducedSwirlReceiptVmBoundary {
    pub program_commitment: Digest,
    pub initial_pc: F,
    pub initial_root: Digest,
    pub final_pc: F,
    pub final_root: Digest,
    pub exit_code: F,
    pub is_terminate: F,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlSourceReceiptRecord {
    pub segment_index: u32,
    pub checkpoint_tidx: usize,
    pub checkpoint_samples: EF,
    pub checkpoint_state: [F; POSEIDON2_WIDTH],
    pub layout: Vec<ReducedSwirlReceiptLayoutEntry>,
    pub roots: Vec<Digest>,
    pub widths: Vec<usize>,
    pub stacking_point: Vec<EF>,
    pub stacking_openings: Vec<Vec<EF>>,
    pub theta: EF,
    pub mu: EF,
    pub beta: Vec<EF>,
    pub eta: EF,
    pub vm: ReducedSwirlReceiptVmBoundary,
    pub layout_digest: Digest,
    pub pending_digest: Digest,
    pub claim_digest: Digest,
    pub entry_digest: Digest,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlSourceReceiptBlock {
    /// Global source index of `sources[0]`. A complete block uses zero; a
    /// bounded recursive source leaf uses its canonical interval start.
    pub source_offset: u32,
    pub sources: Vec<ReducedSwirlSourceReceiptRecord>,
    pub manifest_digest: Digest,
}

/// Setup-fixed wiring of the source receipt component.
///
/// Inline mode exports source authority to the sibling VACC component and
/// therefore uses fanout two for source exports. Detached mode is used by a
/// bounded source leaf: the receipt is the sole consumer, so every source
/// export has fanout one and no authority record is emitted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u8)]
pub enum ReducedSwirlSourceReceiptMode {
    #[default]
    Inline = 0,
    Detached = 1,
}

impl ReducedSwirlSourceReceiptMode {
    #[must_use]
    pub const fn export_lookup_count(self) -> u32 {
        match self {
            Self::Inline => 2,
            Self::Detached => 1,
        }
    }

    #[must_use]
    pub const fn emits_authority(self) -> bool {
        matches!(self, Self::Inline)
    }

    #[must_use]
    pub const fn protocol_tag(self) -> usize {
        self as u8 as usize
    }
}

/// Fill every canonical subdigest, source-entry digest and the ordered block
/// manifest from authenticated scalar data. SDK adapters use this before
/// comparing against `ReducedSwirlSourceManifestPrefix::digest_with_claim`.
pub fn canonicalize_reduced_swirl_source_receipt_block(
    profile: &ReducedSwirlSourceReceiptProfile,
    block: &mut ReducedSwirlSourceReceiptBlock,
) -> Result<(), &'static str> {
    // Structural validation intentionally runs before hashing. Digest fields
    // are ignored here and overwritten below.
    validate_block(profile, block)?;
    for source in &mut block.sources {
        source.layout_digest = hash_observations(LAYOUT_DOMAIN, &layout_observations(source)?)?.0;
        source.pending_digest =
            hash_observations(PENDING_DOMAIN, &pending_observations(profile, source)?)?.0;
        source.claim_digest =
            hash_observations(CLAIM_DOMAIN, &claim_observations(profile, source)?)?.0;
        source.entry_digest = hash_observations(
            ENTRY_DOMAIN,
            &entry_observations(source.segment_index as usize, source),
        )?
        .0;
    }
    block.manifest_digest = manifest_transcript(
        &block
            .sources
            .iter()
            .map(|source| source.entry_digest)
            .collect::<Vec<_>>(),
    )?
    .0;
    Ok(())
}

/// One fixed-height active-prefix AIR.  The existing deferred verifier and
/// `ReducedSwirlSourceAir` remain the sole authorities for all witness data.
#[derive(Clone, Debug)]
pub struct ReducedSwirlSourceReceiptAir {
    pub profile: ReducedSwirlSourceReceiptProfile,
    pub mode: ReducedSwirlSourceReceiptMode,
    pub transcript_bus: TranscriptBus,
    pub verifier_buses: BusInventory,
    pub source_air: ReducedSwirlSourceAir,
    pub authority_bus: ReducedSwirlSourceAuthorityBus,
    pub receipt_bus: ReducedSwirlSourceReceiptBus,
}

impl BaseAir<F> for ReducedSwirlSourceReceiptAir {
    fn width(&self) -> usize {
        self.profile.trace_width()
    }
}
impl BaseAirWithPublicValues<F> for ReducedSwirlSourceReceiptAir {}
impl PartitionedBaseAir<F> for ReducedSwirlSourceReceiptAir {}

impl<AB> Air<AB> for ReducedSwirlSourceReceiptAir
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = &*main.row_slice(0).expect("reduced-SWIRL receipt row");
        let next = &*main.row_slice(1).expect("reduced-SWIRL receipt next row");
        let active = expr::<AB>(local[ACTIVE]);
        let next_active = expr::<AB>(next[ACTIVE]);
        let is_last = expr::<AB>(local[IS_LAST]);
        builder.assert_bool(local[ACTIVE]);
        builder.assert_bool(local[IS_LAST]);
        builder.assert_bool(local[IS_FIRST]);
        builder.when_first_row().assert_one(local[ACTIVE]);
        builder.when_first_row().assert_one(local[IS_FIRST]);
        builder.when_first_row().assert_zero(local[SOURCE]);
        builder
            .when_transition()
            .when(next_active.clone())
            .assert_zero(next[IS_FIRST]);
        builder
            .when_transition()
            .assert_bool(local[ACTIVE] - next[ACTIVE]);
        builder
            .when_transition()
            .assert_eq(local[IS_LAST], local[ACTIVE] - next[ACTIVE]);
        builder
            .when_last_row()
            .assert_eq(local[IS_LAST], local[ACTIVE]);
        builder
            .when_transition()
            .when(next_active.clone())
            .assert_eq(next[SOURCE], expr::<AB>(local[SOURCE]) + AB::Expr::ONE);
        builder
            .when_transition()
            .when(next_active.clone())
            .assert_eq(next[SEGMENT], expr::<AB>(local[SEGMENT]) + AB::Expr::ONE);
        builder.when(is_last.clone()).assert_eq(
            local[SOURCE_COUNT],
            expr::<AB>(local[SOURCE]) + AB::Expr::ONE,
        );
        for value in local.iter().skip(1) {
            builder
                .when(AB::Expr::ONE - active.clone())
                .assert_zero(*value);
        }

        // Shared program and exact adjacent VM state continuity.
        builder
            .when_first_row()
            .when(active.clone())
            .assert_eq(local[CHAIN_INITIAL_PC], local[INITIAL_PC]);
        for limb in 0..DIGEST_SIZE {
            builder
                .when_first_row()
                .when(active.clone())
                .assert_eq(local[CHAIN_INITIAL_ROOT + limb], local[INITIAL_ROOT + limb]);
        }
        let mut transition = builder.when_transition();
        let mut both = transition.when(next_active.clone());
        both.assert_eq(next[PROGRAM], local[PROGRAM]);
        for limb in 1..DIGEST_SIZE {
            both.assert_eq(next[PROGRAM + limb], local[PROGRAM + limb]);
        }
        both.assert_eq(next[INITIAL_PC], local[FINAL_PC]);
        for limb in 0..DIGEST_SIZE {
            both.assert_eq(next[INITIAL_ROOT + limb], local[FINAL_ROOT + limb]);
            both.assert_eq(
                next[CHAIN_INITIAL_ROOT + limb],
                local[CHAIN_INITIAL_ROOT + limb],
            );
        }
        both.assert_eq(next[CHAIN_INITIAL_PC], local[CHAIN_INITIAL_PC]);
        for limb in 0..DIGEST_SIZE {
            both.assert_eq(next[MANIFEST_DIGEST + limb], local[MANIFEST_DIGEST + limb]);
        }
        both.assert_eq(next[SOURCE_COUNT], local[SOURCE_COUNT]);
        builder.assert_bool(local[IS_TERMINATE]);
        builder
            .when(active.clone() - is_last.clone())
            .assert_zero(local[IS_TERMINATE]);
        builder
            .when(active.clone() - expr::<AB>(local[IS_TERMINATE]))
            .assert_eq(
                local[EXIT_CODE],
                AB::Expr::from_u32(self.profile.suspend_exit_code),
            );
        builder
            .when(expr::<AB>(local[IS_TERMINATE]))
            .assert_eq(local[IS_LAST], AB::Expr::ONE);
        builder
            .when(expr::<AB>(local[IS_TERMINATE]))
            .assert_zero(local[EXIT_CODE]);

        let proof = expr::<AB>(local[SOURCE]);
        self.consume_checkpoint(builder, local, proof.clone(), active.clone());
        self.consume_layout(builder, local, proof.clone(), active.clone());
        self.consume_vm(builder, local, proof.clone(), active.clone());
        self.consume_source_claim(builder, local, proof.clone(), active.clone());

        self.emit_layout_hash(builder, local, proof.clone(), active.clone());
        self.emit_pending_hash(builder, local, proof.clone(), active.clone());
        self.emit_claim_hash(builder, local, proof.clone(), active.clone());
        self.emit_entry_hash(builder, local, proof.clone(), active.clone());
        self.emit_manifest(
            builder,
            local,
            proof.clone(),
            active.clone(),
            is_last.clone(),
        );

        if self.mode.emits_authority() {
            self.authority_bus.send(
                builder,
                ReducedSwirlSourceAuthorityMessage {
                    source: proof.clone(),
                    segment_index: local[SEGMENT].into(),
                    common_main_root: array::<AB, DIGEST_SIZE>(
                        local,
                        self.profile.roots_offset() + ROOT_DIGEST,
                    ),
                    trace_layout_digest: array::<AB, DIGEST_SIZE>(local, LAYOUT_DIGEST),
                    pending_claim_digest: array::<AB, DIGEST_SIZE>(local, PENDING_DIGEST),
                    checkpoint_tidx: local[CHECKPOINT_TIDX].into(),
                    checkpoint_state: array::<AB, POSEIDON2_WIDTH>(local, CHECKPOINT_STATE),
                    program_commitment: array::<AB, DIGEST_SIZE>(local, PROGRAM),
                    initial_pc: local[INITIAL_PC].into(),
                    initial_root: array::<AB, DIGEST_SIZE>(local, INITIAL_ROOT),
                    final_pc: local[FINAL_PC].into(),
                    final_root: array::<AB, DIGEST_SIZE>(local, FINAL_ROOT),
                    exit_code: local[EXIT_CODE].into(),
                    is_terminate: local[IS_TERMINATE].into(),
                },
                active.clone(),
            );
        }
        self.receipt_bus.add_key_with_lookups(
            builder,
            ReducedSwirlSourceReceiptMessage {
                protocol_digest: constant_digest::<AB>(self.profile.protocol_digest),
                manifest_digest: array::<AB, DIGEST_SIZE>(local, MANIFEST_DIGEST),
                source_offset: expr::<AB>(local[SEGMENT]) - proof.clone(),
                source_count: local[SOURCE_COUNT].into(),
                program_commitment: array::<AB, DIGEST_SIZE>(local, PROGRAM),
                initial_pc: local[CHAIN_INITIAL_PC].into(),
                initial_root: array::<AB, DIGEST_SIZE>(local, CHAIN_INITIAL_ROOT),
                final_pc: local[FINAL_PC].into(),
                final_root: array::<AB, DIGEST_SIZE>(local, FINAL_ROOT),
                exit_code: local[EXIT_CODE].into(),
                is_terminate: local[IS_TERMINATE].into(),
            },
            is_last,
        );
    }
}

impl ReducedSwirlSourceReceiptAir {
    fn consume_checkpoint<AB>(
        &self,
        builder: &mut AB,
        row: &[AB::Var],
        proof: AB::Expr,
        enabled: AB::Expr,
    ) where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        for limb in 0..D_EF {
            self.verifier_buses.transcript_bus.receive(
                builder,
                proof.clone(),
                TranscriptBusMessage {
                    tidx: expr::<AB>(row[CHECKPOINT_TIDX]) + AB::Expr::from_usize(limb),
                    value: row[CHECKPOINT_SAMPLES + limb].into(),
                    is_sample: AB::Expr::ONE,
                },
                enabled.clone(),
            );
        }
        self.verifier_buses.transcript_end_index_bus.receive(
            builder,
            proof.clone(),
            TranscriptEndIndexMessage {
                tidx: expr::<AB>(row[CHECKPOINT_TIDX]) + AB::Expr::from_usize(D_EF),
            },
            enabled.clone(),
        );
        self.verifier_buses.final_state_bus.receive(
            builder,
            proof,
            FinalTranscriptStateMessage {
                state: array::<AB, POSEIDON2_WIDTH>(row, CHECKPOINT_STATE),
            },
            enabled,
        );
    }

    fn consume_layout<AB>(
        &self,
        builder: &mut AB,
        row: &[AB::Var],
        proof: AB::Expr,
        enabled: AB::Expr,
    ) where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        for air in 0..self.profile.child_air_count {
            let offset = self.profile.layout_offset(air);
            let present = expr::<AB>(row[offset + LAYOUT_PRESENT]);
            builder.assert_bool(row[offset + LAYOUT_PRESENT]);
            self.verifier_buses.air_presence_bus.lookup_key(
                builder,
                proof.clone(),
                AirPresenceBusMessage {
                    air_idx: AB::Expr::from_usize(air),
                    is_present: present.clone(),
                },
                enabled.clone(),
            );
            self.verifier_buses.air_shape_bus.lookup_air_id(
                builder,
                proof.clone(),
                row[offset + LAYOUT_SORT_INDEX],
                AB::Expr::from_usize(air),
                enabled.clone() * present.clone(),
            );
            self.verifier_buses.hyperdim_bus.lookup_key(
                builder,
                proof.clone(),
                HyperdimBusMessage {
                    sort_idx: row[offset + LAYOUT_SORT_INDEX].into(),
                    n_abs: row[offset + LAYOUT_N_ABS].into(),
                    n_sign_bit: row[offset + LAYOUT_N_SIGN].into(),
                },
                enabled.clone() * present.clone(),
            );
            builder.assert_bool(row[offset + LAYOUT_N_SIGN]);
            for value in &row[offset + 1..offset + self.profile.layout_slot_width(air)] {
                builder
                    .when(AB::Expr::ONE - present.clone())
                    .assert_zero(*value);
            }
            for (cached, &global) in self.profile.cached_global_indices[air].iter().enumerate() {
                let digest_offset = offset + LAYOUT_HEADER_WIDTH + cached * DIGEST_SIZE;
                self.verifier_buses.cached_commit_bus.receive(
                    builder,
                    proof.clone(),
                    CachedCommitBusMessage {
                        air_idx: AB::Expr::from_usize(air),
                        cached_idx: AB::Expr::from_usize(cached),
                        global_cached_idx: AB::Expr::from_usize(global),
                        cached_commit: array::<AB, DIGEST_SIZE>(row, digest_offset),
                    },
                    enabled.clone() * present.clone(),
                );
                if air == PROGRAM_AIR_ID && cached == PROGRAM_CACHED_TRACE_INDEX {
                    for limb in 0..DIGEST_SIZE {
                        builder
                            .when(enabled.clone() * present.clone())
                            .assert_eq(row[PROGRAM + limb], row[digest_offset + limb]);
                    }
                }
            }
        }
        self.verifier_buses.pre_hash_bus.receive(
            builder,
            proof,
            PreHashMessage {
                vk_pre_hash: constant_digest::<AB>(self.profile.child_vk_pre_hash),
            },
            enabled,
        );
    }

    fn consume_vm<AB>(&self, builder: &mut AB, row: &[AB::Var], proof: AB::Expr, enabled: AB::Expr)
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        for (pv, value) in [INITIAL_PC, FINAL_PC, EXIT_CODE, IS_TERMINATE]
            .into_iter()
            .enumerate()
        {
            self.verifier_buses.public_values_bus.receive(
                builder,
                proof.clone(),
                PublicValuesBusMessage {
                    air_idx: AB::Expr::from_usize(CONNECTOR_AIR_ID),
                    pv_idx: AB::Expr::from_usize(pv),
                    value: row[value].into(),
                },
                enabled.clone(),
            );
        }
        for limb in 0..DIGEST_SIZE {
            self.verifier_buses.public_values_bus.receive(
                builder,
                proof.clone(),
                PublicValuesBusMessage {
                    air_idx: AB::Expr::from_usize(MERKLE_AIR_ID),
                    pv_idx: AB::Expr::from_usize(limb),
                    value: row[INITIAL_ROOT + limb].into(),
                },
                enabled.clone(),
            );
            self.verifier_buses.public_values_bus.receive(
                builder,
                proof.clone(),
                PublicValuesBusMessage {
                    air_idx: AB::Expr::from_usize(MERKLE_AIR_ID),
                    pv_idx: AB::Expr::from_usize(DIGEST_SIZE + limb),
                    value: row[FINAL_ROOT + limb].into(),
                },
                enabled.clone(),
            );
        }
    }

    fn consume_source_claim<AB>(
        &self,
        builder: &mut AB,
        row: &[AB::Var],
        proof: AB::Expr,
        enabled: AB::Expr,
    ) where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        let roots = self.profile.roots_offset();
        let mut root_sum = AB::Expr::ZERO;
        for ordinal in 0..self.profile.source.maximum_roots_per_source {
            let offset = roots + ordinal * ROOT_SLOT_WIDTH;
            let slot_active = expr::<AB>(row[offset + ROOT_ACTIVE]);
            builder.assert_bool(row[offset + ROOT_ACTIVE]);
            if ordinal == 0 {
                builder
                    .when(enabled.clone())
                    .assert_eq(row[offset + ROOT_ACTIVE], row[ACTIVE]);
            } else {
                builder.assert_zero(
                    slot_active.clone()
                        * (AB::Expr::ONE - expr::<AB>(row[offset - ROOT_SLOT_WIDTH + ROOT_ACTIVE])),
                );
            }
            root_sum += slot_active.clone();
            self.source_air.root_export_bus.lookup_key(
                builder,
                ReducedSwirlSourceRootMessage {
                    source: proof.clone(),
                    root_ordinal: AB::Expr::from_usize(ordinal),
                    width: row[offset + ROOT_WIDTH].into(),
                    root: array::<AB, DIGEST_SIZE>(row, offset + ROOT_DIGEST),
                },
                enabled.clone() * slot_active.clone(),
            );
            // Re-key the already authenticated root width locally for the
            // first opening in this root's contiguous column group. This
            // replaces the former openings-by-roots one-hot table: the source
            // AIR already constrains opening order, root ordinals, and group
            // widths, while this balanced local bus ties the digest width to
            // the exact root row consumed above.
            self.source_air.root_width_bus.send(
                builder,
                ReducedSwirlSourceRootWidthMessage {
                    source: proof.clone(),
                    root_ordinal: AB::Expr::from_usize(ordinal),
                    width: row[offset + ROOT_WIDTH].into(),
                },
                enabled.clone() * slot_active.clone(),
            );
            for value in &row[offset + 1..offset + ROOT_SLOT_WIDTH] {
                builder
                    .when(AB::Expr::ONE - slot_active.clone())
                    .assert_zero(*value);
            }
        }
        builder
            .when(enabled.clone())
            .assert_eq(root_sum, row[ROOT_COUNT]);
        for coordinate in 0..self.profile.source.log_message_len() {
            let point = self.profile.points_offset() + coordinate * D_EF;
            let beta = self.profile.betas_offset() + coordinate * D_EF;
            self.source_air.point_export_bus.lookup_key(
                builder,
                ReducedSwirlSourcePointMessage {
                    source: proof.clone(),
                    coordinate: AB::Expr::from_usize(coordinate),
                    value: array::<AB, D_EF>(row, point),
                },
                enabled.clone(),
            );
            // Inline VACC consumes roots/beta/claim but not point/opening.
            // Balance that setup-fixed second export lookup only in inline
            // mode. Detached leaves use fanout one and must not duplicate it.
            if self.mode == ReducedSwirlSourceReceiptMode::Inline {
                self.source_air.point_export_bus.lookup_key(
                    builder,
                    ReducedSwirlSourcePointMessage {
                        source: proof.clone(),
                        coordinate: AB::Expr::from_usize(coordinate),
                        value: array::<AB, D_EF>(row, point),
                    },
                    enabled.clone(),
                );
            }
            self.source_air.beta_export_bus.lookup_key(
                builder,
                ReducedSwirlSourceBetaMessage {
                    source: proof.clone(),
                    coordinate: AB::Expr::from_usize(coordinate),
                    value: array::<AB, D_EF>(row, beta),
                },
                enabled.clone(),
            );
            let reverse_point = self.profile.points_offset()
                + (self.profile.source.log_message_len() - 1 - coordinate) * D_EF;
            for limb in 0..D_EF {
                builder
                    .when(enabled.clone())
                    .assert_eq(row[beta + limb], row[reverse_point + limb]);
            }
        }
        let openings = self.profile.openings_offset();
        let mut opening_sum = AB::Expr::ZERO;
        for index in 0..self.profile.source.maximum_openings_per_source {
            let offset = openings + index * OPENING_SLOT_WIDTH;
            let slot_active = expr::<AB>(row[offset + OPENING_ACTIVE]);
            let first = expr::<AB>(row[offset + OPENING_FIRST]);
            let column = expr::<AB>(row[offset + OPENING_COLUMN]);
            builder.assert_bool(row[offset + OPENING_ACTIVE]);
            builder.assert_bool(row[offset + OPENING_FIRST]);
            if index > 0 {
                builder.assert_zero(
                    slot_active.clone()
                        * (AB::Expr::ONE
                            - expr::<AB>(row[offset - OPENING_SLOT_WIDTH + OPENING_ACTIVE])),
                );
            }
            opening_sum += slot_active.clone();
            builder.assert_zero(slot_active.clone() * first.clone() * column.clone());
            builder
                .when(slot_active.clone() * (AB::Expr::ONE - first.clone()))
                .assert_one(column * row[offset + OPENING_COLUMN_INVERSE]);
            builder
                .when(slot_active.clone() * first.clone())
                .assert_zero(row[offset + OPENING_COLUMN_INVERSE]);
            builder
                .when(slot_active.clone() * (AB::Expr::ONE - first.clone()))
                .assert_zero(row[offset + OPENING_GROUP_WIDTH]);
            self.source_air.root_width_bus.receive(
                builder,
                ReducedSwirlSourceRootWidthMessage {
                    source: proof.clone(),
                    root_ordinal: row[offset + OPENING_ROOT].into(),
                    width: row[offset + OPENING_GROUP_WIDTH].into(),
                },
                enabled.clone() * slot_active.clone() * first.clone(),
            );
            if self.mode == ReducedSwirlSourceReceiptMode::Inline {
                self.source_air.opening_export_bus.lookup_key(
                    builder,
                    ReducedSwirlSourceOpeningMessage {
                        source: proof.clone(),
                        opening_index: AB::Expr::from_usize(index),
                        root_ordinal: row[offset + OPENING_ROOT].into(),
                        column: row[offset + OPENING_COLUMN].into(),
                        value: array::<AB, D_EF>(row, offset + OPENING_VALUE),
                    },
                    enabled.clone() * slot_active.clone(),
                );
            }
            self.source_air.opening_export_bus.lookup_key(
                builder,
                ReducedSwirlSourceOpeningMessage {
                    source: proof.clone(),
                    opening_index: AB::Expr::from_usize(index),
                    root_ordinal: row[offset + OPENING_ROOT].into(),
                    column: row[offset + OPENING_COLUMN].into(),
                    value: array::<AB, D_EF>(row, offset + OPENING_VALUE),
                },
                enabled.clone() * slot_active.clone(),
            );
            for value in &row[offset + 1..offset + OPENING_SLOT_WIDTH] {
                builder
                    .when(AB::Expr::ONE - slot_active.clone())
                    .assert_zero(*value);
            }
        }
        builder
            .when(enabled.clone())
            .assert_eq(opening_sum, row[OPENING_COUNT]);
        self.source_air.claim_export_bus.lookup_key(builder, ReducedSwirlSourceClaimMessage {
            source: proof,
            root_count: row[ROOT_COUNT].into(), opening_count: row[OPENING_COUNT].into(),
            l_skip: AB::Expr::from_usize(self.profile.source.l_skip), n_stack: AB::Expr::from_usize(self.profile.source.n_stack),
            log_blowup: AB::Expr::from_usize(self.profile.source.log_blowup), log_commit_rows_per_query: AB::Expr::from_usize(self.profile.source.log_commit_rows_per_query),
            log_message_len: AB::Expr::from_usize(self.profile.source.log_message_len()), log_codeword_len: AB::Expr::from_usize(self.profile.source.log_codeword_len()), rows_per_query: AB::Expr::from_usize(self.profile.source.rows_per_query()),
            coefficient_layout_tag: AB::Expr::from_u64(openvm_recursion_circuit::native_warp::REDUCED_SWIRL_COEFFICIENT_SUBGROUP_TAG), coefficient_layout_version: AB::Expr::from_u32(openvm_recursion_circuit::native_warp::REDUCED_SWIRL_COEFFICIENT_SUBGROUP_VERSION), alpha_is_zero: AB::Expr::ONE,
            theta: array::<AB, D_EF>(row, THETA), mu: array::<AB, D_EF>(row, MU), eta: array::<AB, D_EF>(row, ETA),
        }, enabled);
    }

    fn emit_layout_hash<AB>(
        &self,
        builder: &mut AB,
        row: &[AB::Var],
        proof: AB::Expr,
        enabled: AB::Expr,
    ) where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        let hash_proof = proof.clone() * AB::Expr::from_usize(SOURCE_HASHES_PER_SOURCE)
            + AB::Expr::from_usize(LAYOUT_HASH_SLOT);
        let mut count = AB::Expr::from_usize(2 + 3 * self.profile.child_air_count);
        for air in 0..self.profile.child_air_count {
            let o = self.profile.layout_offset(air);
            count += expr::<AB>(row[o + LAYOUT_PRESENT])
                * AB::Expr::from_usize(4 + self.profile.cached_global_indices[air].len());
        }
        let mut t = preamble::<AB>(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            LAYOUT_DOMAIN,
            count,
            enabled.clone(),
        );
        typed_u32(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            AB::Expr::from_usize(self.profile.child_air_count),
            enabled.clone(),
        );
        for air in 0..self.profile.child_air_count {
            let o = self.profile.layout_offset(air);
            let present = expr::<AB>(row[o + LAYOUT_PRESENT]);
            typed_u32(
                builder,
                self.transcript_bus,
                hash_proof.clone(),
                &mut t,
                AB::Expr::from_usize(air),
                enabled.clone(),
            );
            typed_field(
                builder,
                self.transcript_bus,
                hash_proof.clone(),
                &mut t,
                present.clone(),
                enabled.clone(),
            );
            let conditional = enabled.clone() * present.clone();
            let signed = expr::<AB>(row[o + LAYOUT_N_ABS]);
            let log_height = AB::Expr::from_usize(self.profile.child_l_skip) + signed.clone()
                - AB::Expr::TWO * signed * row[o + LAYOUT_N_SIGN];
            typed_u32(
                builder,
                self.transcript_bus,
                hash_proof.clone(),
                &mut t,
                log_height,
                conditional.clone(),
            );
            typed_u32(
                builder,
                self.transcript_bus,
                hash_proof.clone(),
                &mut t,
                AB::Expr::from_usize(self.profile.cached_global_indices[air].len()),
                conditional.clone(),
            );
            for cached in 0..self.profile.cached_global_indices[air].len() {
                typed_digest(
                    builder,
                    self.transcript_bus,
                    hash_proof.clone(),
                    &mut t,
                    array::<AB, DIGEST_SIZE>(row, o + LAYOUT_HEADER_WIDTH + cached * DIGEST_SIZE),
                    conditional.clone(),
                );
            }
        }
        finish_hash(
            builder,
            self.transcript_bus,
            hash_proof,
            t,
            &row[LAYOUT_DIGEST..LAYOUT_DIGEST + DIGEST_SIZE],
            enabled,
        );
    }

    fn emit_pending_hash<AB>(
        &self,
        builder: &mut AB,
        row: &[AB::Var],
        proof: AB::Expr,
        enabled: AB::Expr,
    ) where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        let hash_proof = proof.clone() * AB::Expr::from_usize(SOURCE_HASHES_PER_SOURCE)
            + AB::Expr::from_usize(PENDING_HASH_SLOT);
        let root_count = expr::<AB>(row[ROOT_COUNT]);
        let opening_count = expr::<AB>(row[OPENING_COUNT]);
        let count = AB::Expr::from_usize(18 + self.profile.source.log_message_len())
            + AB::Expr::from_u32(5) * root_count.clone()
            + opening_count;
        let mut t = preamble::<AB>(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            PENDING_DOMAIN,
            count,
            enabled.clone(),
        );
        typed_u32(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            root_count.clone(),
            enabled.clone(),
        );
        for ordinal in 0..self.profile.source.maximum_roots_per_source {
            let o = self.profile.roots_offset() + ordinal * ROOT_SLOT_WIDTH;
            typed_digest(
                builder,
                self.transcript_bus,
                hash_proof.clone(),
                &mut t,
                array::<AB, DIGEST_SIZE>(row, o + ROOT_DIGEST),
                enabled.clone() * row[o + ROOT_ACTIVE],
            );
        }
        typed_u32(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            root_count.clone(),
            enabled.clone(),
        );
        for ordinal in 0..self.profile.source.maximum_roots_per_source {
            let o = self.profile.roots_offset() + ordinal * ROOT_SLOT_WIDTH;
            typed_u32(
                builder,
                self.transcript_bus,
                hash_proof.clone(),
                &mut t,
                row[o + ROOT_WIDTH].into(),
                enabled.clone() * row[o + ROOT_ACTIVE],
            );
        }
        for value in [
            self.profile.source.l_skip,
            self.profile.source.n_stack,
            self.profile.source.log_blowup,
            self.profile.source.log_commit_rows_per_query,
        ] {
            typed_u32(
                builder,
                self.transcript_bus,
                hash_proof.clone(),
                &mut t,
                AB::Expr::from_usize(value),
                enabled.clone(),
            );
        }
        let total_width = (0..self.profile.source.maximum_roots_per_source).fold(
            AB::Expr::ZERO,
            |acc, ordinal| {
                acc + row[self.profile.roots_offset() + ordinal * ROOT_SLOT_WIDTH + ROOT_WIDTH]
            },
        );
        typed_u32(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            total_width,
            enabled.clone(),
        );
        typed_u32(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            AB::Expr::from_usize(self.profile.source.log_message_len()),
            enabled.clone(),
        );
        for coordinate in 0..self.profile.source.log_message_len() {
            typed_extension(
                builder,
                self.transcript_bus,
                hash_proof.clone(),
                &mut t,
                array::<AB, D_EF>(row, self.profile.points_offset() + coordinate * D_EF),
                enabled.clone(),
            );
        }
        typed_u32(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            root_count,
            enabled.clone(),
        );
        // Openings are exported by `ReducedSwirlSourceAir` in canonical
        // root-major, column-major order. Emit each group's authenticated
        // width at column zero, then its values. This is byte-for-byte the
        // pending-claim transcript without a quadratic one-hot selector
        // matrix.
        for index in 0..self.profile.source.maximum_openings_per_source {
            let o = self.profile.openings_offset() + index * OPENING_SLOT_WIDTH;
            let opening_active = expr::<AB>(row[o + OPENING_ACTIVE]);
            let first = expr::<AB>(row[o + OPENING_FIRST]);
            typed_u32(
                builder,
                self.transcript_bus,
                hash_proof.clone(),
                &mut t,
                row[o + OPENING_GROUP_WIDTH].into(),
                enabled.clone() * opening_active.clone() * first,
            );
            typed_extension(
                builder,
                self.transcript_bus,
                hash_proof.clone(),
                &mut t,
                array::<AB, D_EF>(row, o + OPENING_VALUE),
                enabled.clone() * opening_active,
            );
        }
        finish_hash(
            builder,
            self.transcript_bus,
            hash_proof,
            t,
            &row[PENDING_DIGEST..PENDING_DIGEST + DIGEST_SIZE],
            enabled,
        );
    }

    fn emit_claim_hash<AB>(
        &self,
        builder: &mut AB,
        row: &[AB::Var],
        proof: AB::Expr,
        enabled: AB::Expr,
    ) where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        let hash_proof = proof.clone() * AB::Expr::from_usize(SOURCE_HASHES_PER_SOURCE)
            + AB::Expr::from_usize(CLAIM_HASH_SLOT);
        let root_count = expr::<AB>(row[ROOT_COUNT]);
        let count = AB::Expr::from_usize(
            19 + self.profile.source.log_codeword_len() + self.profile.source.log_message_len(),
        ) + AB::Expr::from_u32(3) * root_count.clone();
        let mut t = preamble::<AB>(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            CLAIM_DOMAIN,
            count,
            enabled.clone(),
        );
        typed_u32(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            root_count.clone(),
            enabled.clone(),
        );
        for ordinal in 0..self.profile.source.maximum_roots_per_source {
            let o = self.profile.roots_offset() + ordinal * ROOT_SLOT_WIDTH;
            typed_digest(
                builder,
                self.transcript_bus,
                hash_proof.clone(),
                &mut t,
                array::<AB, DIGEST_SIZE>(row, o + ROOT_DIGEST),
                enabled.clone() * row[o + ROOT_ACTIVE],
            );
        }
        typed_u32(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            root_count,
            enabled.clone(),
        );
        for ordinal in 0..self.profile.source.maximum_roots_per_source {
            let o = self.profile.roots_offset() + ordinal * ROOT_SLOT_WIDTH;
            typed_u32(
                builder,
                self.transcript_bus,
                hash_proof.clone(),
                &mut t,
                row[o + ROOT_WIDTH].into(),
                enabled.clone() * row[o + ROOT_ACTIVE],
            );
        }
        for value in [
            self.profile.source.l_skip,
            self.profile.source.n_stack,
            self.profile.source.log_blowup,
            self.profile.source.log_commit_rows_per_query,
        ] {
            typed_u32(
                builder,
                self.transcript_bus,
                hash_proof.clone(),
                &mut t,
                AB::Expr::from_usize(value),
                enabled.clone(),
            );
        }
        typed_extension(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            array::<AB, D_EF>(row, THETA),
            enabled.clone(),
        );
        typed_u32(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            AB::Expr::from_usize(self.profile.source.log_codeword_len()),
            enabled.clone(),
        );
        for _ in 0..self.profile.source.log_codeword_len() {
            typed_extension(
                builder,
                self.transcript_bus,
                hash_proof.clone(),
                &mut t,
                [AB::Expr::ZERO; D_EF],
                enabled.clone(),
            );
        }
        typed_extension(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            array::<AB, D_EF>(row, MU),
            enabled.clone(),
        );
        typed_u32(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            AB::Expr::from_usize(self.profile.source.log_message_len()),
            enabled.clone(),
        );
        for coordinate in 0..self.profile.source.log_message_len() {
            typed_extension(
                builder,
                self.transcript_bus,
                hash_proof.clone(),
                &mut t,
                array::<AB, D_EF>(row, self.profile.betas_offset() + coordinate * D_EF),
                enabled.clone(),
            );
        }
        typed_extension(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            array::<AB, D_EF>(row, ETA),
            enabled.clone(),
        );
        finish_hash(
            builder,
            self.transcript_bus,
            hash_proof,
            t,
            &row[CLAIM_DIGEST..CLAIM_DIGEST + DIGEST_SIZE],
            enabled,
        );
    }

    fn emit_entry_hash<AB>(
        &self,
        builder: &mut AB,
        row: &[AB::Var],
        proof: AB::Expr,
        enabled: AB::Expr,
    ) where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        let hash_proof = proof.clone() * AB::Expr::from_usize(SOURCE_HASHES_PER_SOURCE)
            + AB::Expr::from_usize(ENTRY_HASH_SLOT);
        let mut t = preamble::<AB>(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            ENTRY_DOMAIN,
            AB::Expr::from_usize(15),
            enabled.clone(),
        );
        typed_u32(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            // Entry digests are globally indexed even inside a bounded leaf.
            // `SOURCE` remains the verifier-local proof index used by buses.
            row[SEGMENT].into(),
            enabled.clone(),
        );
        typed_u32(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            row[SEGMENT].into(),
            enabled.clone(),
        );
        typed_digest(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            array::<AB, DIGEST_SIZE>(row, self.profile.roots_offset() + ROOT_DIGEST),
            enabled.clone(),
        );
        for offset in [LAYOUT_DIGEST, PENDING_DIGEST, CLAIM_DIGEST, PROGRAM] {
            typed_digest(
                builder,
                self.transcript_bus,
                hash_proof.clone(),
                &mut t,
                array::<AB, DIGEST_SIZE>(row, offset),
                enabled.clone(),
            );
        }
        typed_field(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            row[INITIAL_PC].into(),
            enabled.clone(),
        );
        typed_digest(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            array::<AB, DIGEST_SIZE>(row, INITIAL_ROOT),
            enabled.clone(),
        );
        typed_field(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            row[FINAL_PC].into(),
            enabled.clone(),
        );
        typed_digest(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            array::<AB, DIGEST_SIZE>(row, FINAL_ROOT),
            enabled.clone(),
        );
        typed_field(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            row[EXIT_CODE].into(),
            enabled.clone(),
        );
        typed_field(
            builder,
            self.transcript_bus,
            hash_proof.clone(),
            &mut t,
            row[IS_TERMINATE].into(),
            enabled.clone(),
        );
        finish_hash(
            builder,
            self.transcript_bus,
            hash_proof,
            t,
            &row[ENTRY_DIGEST..ENTRY_DIGEST + DIGEST_SIZE],
            enabled,
        );
    }

    fn emit_manifest<AB>(
        &self,
        builder: &mut AB,
        row: &[AB::Var],
        proof: AB::Expr,
        enabled: AB::Expr,
        is_last: AB::Expr,
    ) where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        let hash_proof =
            expr::<AB>(row[SOURCE_COUNT]) * AB::Expr::from_usize(SOURCE_HASHES_PER_SOURCE);
        let base = AB::Expr::from_usize(MANIFEST_TAG.len() + 2)
            + proof.clone() * AB::Expr::from_usize(1 + DIGEST_SIZE);
        for (index, byte) in MANIFEST_TAG.iter().copied().enumerate() {
            self.transcript_bus.observe(
                builder,
                hash_proof.clone(),
                AB::Expr::from_usize(index),
                AB::Expr::from_u8(byte),
                enabled.clone() * row[IS_FIRST],
            );
        }
        self.transcript_bus.observe(
            builder,
            hash_proof.clone(),
            AB::Expr::from_usize(MANIFEST_TAG.len()),
            AB::Expr::from_u32(REDUCED_SWIRL_WRAPPER_PROTOCOL_VERSION),
            enabled.clone() * row[IS_FIRST],
        );
        self.transcript_bus.observe(
            builder,
            hash_proof.clone(),
            AB::Expr::from_usize(MANIFEST_TAG.len() + 1),
            row[SOURCE_COUNT],
            enabled.clone() * row[IS_FIRST],
        );
        self.transcript_bus.observe(
            builder,
            hash_proof.clone(),
            base.clone(),
            row[SOURCE],
            enabled.clone(),
        );
        for limb in 0..DIGEST_SIZE {
            self.transcript_bus.observe(
                builder,
                hash_proof.clone(),
                base.clone() + AB::Expr::from_usize(1 + limb),
                row[ENTRY_DIGEST + limb],
                enabled.clone(),
            );
        }
        let sample_start = AB::Expr::from_usize(MANIFEST_TAG.len() + 2)
            + expr::<AB>(row[SOURCE_COUNT]) * AB::Expr::from_usize(1 + DIGEST_SIZE);
        for limb in 0..DIGEST_SIZE {
            self.transcript_bus.sample(
                builder,
                hash_proof.clone(),
                sample_start.clone() + AB::Expr::from_usize(limb),
                row[MANIFEST_DIGEST + limb],
                is_last.clone(),
            );
        }
    }
}

#[derive(Clone, Copy)]
enum CanonicalObservation {
    Field(F),
    Digest(Digest),
    Extension(EF),
}

/// AIR-aligned receipt and canonical transcript traces.
pub struct ReducedSwirlSourceReceiptTraces {
    pub receipt: RowMajorMatrix<F>,
    pub transcript: NativeWarpTranscriptArtifacts,
    pub entry_digests: Vec<Digest>,
    pub manifest_digest: Digest,
}

/// Owns only the canonical source/manifest transcript.  The ordinary
/// deferred verifier and [`ReducedSwirlSourceAir`] are assembled beside it.
pub struct ReducedSwirlSourceReceiptProducer {
    pub air: ReducedSwirlSourceReceiptAir,
    transcript: NativeWarpTranscriptModule,
}

impl ReducedSwirlSourceReceiptProducer {
    pub fn new(
        air: ReducedSwirlSourceReceiptAir,
        shared: &BusInventory,
        params: SystemParams,
    ) -> Result<Self, &'static str> {
        air.profile.validate()?;
        let transcript =
            NativeWarpTranscriptModule::new_for_bus(shared, air.transcript_bus, params);
        Ok(Self { air, transcript })
    }

    #[must_use]
    pub fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        let mut airs = vec![std::sync::Arc::new(self.air.clone()) as AirRef<SC>];
        airs.extend(self.transcript.airs::<SC>());
        airs
    }

    /// Poseidon lookup owner shared with a detached source-leaf boundary AIR.
    /// The boundary AIR must use these buses, and pass its compression
    /// pre-states to [`Self::generate_traces_with_external_compressions`], so
    /// no second Poseidon table is introduced.
    #[must_use]
    pub fn poseidon2_bus_owner(&self) -> Poseidon2BusOwner {
        self.transcript.poseidon2_bus_owner()
    }

    pub fn generate_traces(
        &self,
        block: &ReducedSwirlSourceReceiptBlock,
    ) -> Result<ReducedSwirlSourceReceiptTraces, &'static str> {
        self.generate_traces_with_external_compressions(block, &[])
    }

    /// Generate the ordinary receipt traces while servicing an additional,
    /// caller-owned set of compression lookups in this producer's existing
    /// Poseidon table. Inputs are full Poseidon pre-states in canonical lookup
    /// order. Existing inline callers use [`Self::generate_traces`].
    pub fn generate_traces_with_external_compressions(
        &self,
        block: &ReducedSwirlSourceReceiptBlock,
        external_poseidon2_compression_inputs: &[[F; POSEIDON2_WIDTH]],
    ) -> Result<ReducedSwirlSourceReceiptTraces, &'static str> {
        let logs = canonical_transcript_logs(&self.air.profile, block)?;
        let log_refs = logs.iter().collect::<Vec<_>>();
        let transcript = self
            .transcript
            .generate_trace_with_external(
                &log_refs,
                Vec::new(),
                external_poseidon2_compression_inputs.to_vec(),
                None,
                None,
            )
            .ok_or("reduced-SWIRL receipt transcript trace")?;
        let receipt = generate_receipt_trace(&self.air.profile, block)?;
        Ok(ReducedSwirlSourceReceiptTraces {
            receipt,
            transcript,
            entry_digests: block
                .sources
                .iter()
                .map(|source| source.entry_digest)
                .collect(),
            manifest_digest: block.manifest_digest,
        })
    }
}

/// Complete source-side component below the fixed-capacity wrapper.
///
/// It embeds OpenVM's ordinary `DeferredWhir` verifier, removes only its
/// public checkpoint AIR, and routes that checkpoint privately into the
/// receipt AIR. The established [`ReducedSwirlSourceAir`] remains the sole
/// post-stacking claim reducer.
pub struct ReducedSwirlSourceReceiptComponent<
    const MAX_SOURCES: usize = { REDUCED_SWIRL_WRAPPER_MAX_SOURCES as usize },
> {
    verifier: VerifierSubCircuit<MAX_SOURCES, 0>,
    params: SystemParams,
    checkpoint_air_index: usize,
    source_air: ReducedSwirlSourceAir,
    source_transcript: NativeWarpTranscriptModule,
    receipt: ReducedSwirlSourceReceiptProducer,
    mode: ReducedSwirlSourceReceiptMode,
    next_bus_idx: BusIndex,
}

/// Source and receipt traces in the exact order appended after the filtered
/// ordinary verifier AIRs.
pub struct ReducedSwirlSourceReceiptComponentTraces {
    pub source: RowMajorMatrix<F>,
    pub source_transcript: NativeWarpTranscriptArtifacts,
    pub receipt: ReducedSwirlSourceReceiptTraces,
    pub claims: Vec<RecursiveReducedSwirlClaim>,
}

impl<const MAX_SOURCES: usize> ReducedSwirlSourceReceiptComponent<MAX_SOURCES> {
    /// Allocate the verifier and source auxiliaries from one bus namespace.
    /// `receipt_bus` and `authority_bus` must already have been allocated from
    /// `bus_idx_manager`; [`Self::next_bus_idx`] is safe for later components.
    pub fn new(
        child_vk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
        profile: ReducedSwirlSourceReceiptProfile,
        params: SystemParams,
        receipt_bus: ReducedSwirlSourceReceiptBus,
        authority_bus: ReducedSwirlSourceAuthorityBus,
        bus_idx_manager: BusIndexManager,
    ) -> Result<Self, &'static str> {
        Self::new_with_mode(
            child_vk,
            profile,
            params,
            receipt_bus,
            authority_bus,
            bus_idx_manager,
            ReducedSwirlSourceReceiptMode::Inline,
        )
    }

    /// Construct a bounded source-leaf component whose sole output is the
    /// receipt. No per-source authority lookup escapes this component.
    pub fn new_detached(
        child_vk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
        profile: ReducedSwirlSourceReceiptProfile,
        params: SystemParams,
        receipt_bus: ReducedSwirlSourceReceiptBus,
        authority_bus: ReducedSwirlSourceAuthorityBus,
        bus_idx_manager: BusIndexManager,
    ) -> Result<Self, &'static str> {
        Self::new_with_mode(
            child_vk,
            profile,
            params,
            receipt_bus,
            authority_bus,
            bus_idx_manager,
            ReducedSwirlSourceReceiptMode::Detached,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_with_mode(
        child_vk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
        profile: ReducedSwirlSourceReceiptProfile,
        params: SystemParams,
        receipt_bus: ReducedSwirlSourceReceiptBus,
        authority_bus: ReducedSwirlSourceAuthorityBus,
        bus_idx_manager: BusIndexManager,
        mode: ReducedSwirlSourceReceiptMode,
    ) -> Result<Self, &'static str> {
        profile.validate()?;
        if profile.source.maximum_sources != MAX_SOURCES {
            return Err("reduced-SWIRL source component capacity mismatch");
        }
        let child_params = &child_vk.inner.params;
        if profile.child_air_count != child_vk.inner.per_air.len()
            || profile.child_vk_pre_hash != child_vk.pre_hash
            || profile.child_l_skip != child_params.l_skip
            || profile.source.l_skip != child_params.l_skip
            || profile.source.n_stack != child_params.n_stack
            || profile.source.log_blowup != child_params.log_blowup
            || profile.source.log_commit_rows_per_query != child_params.log_commit_rows_per_query
        {
            return Err("reduced-SWIRL source profile/child VK mismatch");
        }
        let mut verifier =
            VerifierSubCircuit::<MAX_SOURCES, 0>::new_with_options_from_bus_idx_manager(
                Arc::clone(&child_vk),
                VerifierConfig {
                    continuations_enabled: true,
                    final_state_bus_enabled: true,
                    // Authenticate the exact child-VK symbolic table as the
                    // first verifier AIR's cached main. Besides avoiding dynamic
                    // replay, this is the standard OpenVM key-lineage seam: an
                    // ordinary recursive parent reads this commitment at
                    // CONSTRAINT_EVAL_AIR_ID from the wrapper proof.
                    has_cached: true,
                    tail_mode: VerifierTailMode::DeferredWhir,
                },
                bus_idx_manager,
            );
        // In cached mode the complete symbolic table is authenticated by the
        // wrapper PCS itself. `dag_commit_info` exists only for the alternate
        // record/replay mode and must not be substituted for this commitment.
        let stacking_endpoint_bus = verifier.configure_deferred_swirl_source_exports()?;
        let checkpoint_air_index = verifier
            .deferred_opening_checkpoint_air_index()
            .ok_or("deferred-SWIRL checkpoint AIR")?;
        let verifier_buses = verifier.bus_inventory().clone();
        let mut buses = BusIndexManager::from_next_bus_idx(verifier.next_bus_idx());

        let source_transcript_bus = TranscriptBus::new(buses.new_bus_idx());
        let source_air = ReducedSwirlSourceAir {
            profile: profile.source,
            transcript_bus: source_transcript_bus,
            commitments_bus: verifier_buses.commitments_bus,
            stacking_endpoint_bus,
            root_width_bus: ReducedSwirlSourceRootWidthBus::new(buses.new_bus_idx()),
            root_export_bus: ReducedSwirlSourceRootBus::new(buses.new_bus_idx()),
            point_export_bus: ReducedSwirlSourcePointBus::new(buses.new_bus_idx()),
            opening_export_bus: ReducedSwirlSourceOpeningBus::new(buses.new_bus_idx()),
            beta_export_bus: ReducedSwirlSourceBetaBus::new(buses.new_bus_idx()),
            claim_export_bus: ReducedSwirlSourceClaimBus::new(buses.new_bus_idx()),
            // Inline mode has a second VACC consumer. A detached source leaf
            // has only the receipt consumer and therefore exact fanout one.
            export_lookup_count: mode.export_lookup_count(),
        };
        let source_transcript = NativeWarpTranscriptModule::new_for_bus(
            &verifier_buses,
            source_transcript_bus,
            params.clone(),
        );
        let receipt_air = ReducedSwirlSourceReceiptAir {
            profile,
            mode,
            transcript_bus: TranscriptBus::new(buses.new_bus_idx()),
            verifier_buses: verifier_buses.clone(),
            source_air,
            authority_bus,
            receipt_bus,
        };
        let receipt =
            ReducedSwirlSourceReceiptProducer::new(receipt_air, &verifier_buses, params.clone())?;
        Ok(Self {
            verifier,
            params,
            checkpoint_air_index,
            source_air,
            source_transcript,
            receipt,
            mode,
            next_bus_idx: buses.next_bus_idx(),
        })
    }

    #[must_use]
    pub const fn verifier(&self) -> &VerifierSubCircuit<MAX_SOURCES, 0> {
        &self.verifier
    }

    #[must_use]
    pub const fn params(&self) -> &SystemParams {
        &self.params
    }

    #[must_use]
    pub const fn checkpoint_air_index(&self) -> usize {
        self.checkpoint_air_index
    }

    #[must_use]
    pub const fn source_air(&self) -> &ReducedSwirlSourceAir {
        &self.source_air
    }

    #[must_use]
    pub const fn receipt_air(&self) -> &ReducedSwirlSourceReceiptAir {
        &self.receipt.air
    }

    #[must_use]
    pub const fn mode(&self) -> ReducedSwirlSourceReceiptMode {
        self.mode
    }

    /// Poseidon owner to be used by a detached source-leaf boundary AIR.
    #[must_use]
    pub fn receipt_poseidon2_bus_owner(&self) -> Poseidon2BusOwner {
        self.receipt.poseidon2_bus_owner()
    }

    #[must_use]
    pub const fn next_bus_idx(&self) -> BusIndex {
        self.next_bus_idx
    }

    #[must_use]
    pub fn verifier_airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        self.verifier
            .airs::<SC>()
            .into_iter()
            .enumerate()
            .filter_map(|(index, air)| (index != self.checkpoint_air_index).then_some(air))
            .collect()
    }

    #[must_use]
    pub fn auxiliary_airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        let mut airs = vec![Arc::new(self.source_air) as AirRef<SC>];
        airs.extend(self.source_transcript.airs::<SC>());
        airs.extend(self.receipt.airs::<SC>());
        airs
    }

    pub fn generate_auxiliary_traces(
        &self,
        source_records: &[ReducedSwirlSourceRecord],
        block: &ReducedSwirlSourceReceiptBlock,
    ) -> Result<ReducedSwirlSourceReceiptComponentTraces, &'static str> {
        self.generate_auxiliary_traces_with_external_compressions(source_records, block, &[])
    }

    /// As [`Self::generate_auxiliary_traces`], with extra compression
    /// pre-states supplied by a boundary AIR sharing the receipt Poseidon bus.
    pub fn generate_auxiliary_traces_with_external_compressions(
        &self,
        source_records: &[ReducedSwirlSourceRecord],
        block: &ReducedSwirlSourceReceiptBlock,
        external_poseidon2_compression_inputs: &[[F; POSEIDON2_WIDTH]],
    ) -> Result<ReducedSwirlSourceReceiptComponentTraces, &'static str> {
        if source_records.len() != block.sources.len() {
            return Err("reduced-SWIRL source/receipt count");
        }
        let source = openvm_recursion_circuit::native_warp::generate_reduced_swirl_source_trace(
            self.source_air.profile,
            source_records,
        )?;
        for (claim, receipt) in source.claims.iter().zip(&block.sources) {
            for (matches, field) in [
                (claim.roots == receipt.roots, "roots"),
                (claim.widths == receipt.widths, "widths"),
                (claim.theta == receipt.theta, "theta"),
                (
                    claim.alpha.len() == self.source_air.profile.log_codeword_len()
                        && claim.alpha.iter().all(|value| *value == EF::ZERO),
                    "alpha",
                ),
                (claim.mu == receipt.mu, "mu"),
                (claim.beta == receipt.beta, "beta"),
                (claim.eta == receipt.eta, "eta"),
            ] {
                if !matches {
                    return Err(field);
                }
            }
        }
        let source_logs = source.transcript_logs.iter().collect::<Vec<_>>();
        let source_transcript = self
            .source_transcript
            .generate_trace(&source_logs, None)
            .ok_or("reduced-SWIRL source transcript trace")?;
        let receipt = self.receipt.generate_traces_with_external_compressions(
            block,
            external_poseidon2_compression_inputs,
        )?;
        Ok(ReducedSwirlSourceReceiptComponentTraces {
            source: source.trace,
            source_transcript,
            receipt,
            claims: source.claims,
        })
    }
}

impl<const MAX_SOURCES: usize> AggregationSubCircuit
    for ReducedSwirlSourceReceiptComponent<MAX_SOURCES>
{
    fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        self.verifier_airs::<SC>()
            .into_iter()
            .chain(self.auxiliary_airs::<SC>())
            .collect()
    }

    fn bus_inventory(&self) -> &BusInventory {
        self.verifier.bus_inventory()
    }

    fn next_bus_idx(&self) -> BusIndex {
        self.next_bus_idx
    }

    fn max_num_proofs(&self) -> usize {
        MAX_SOURCES
    }
}

fn validate_block(
    profile: &ReducedSwirlSourceReceiptProfile,
    block: &ReducedSwirlSourceReceiptBlock,
) -> Result<(), &'static str> {
    profile.validate()?;
    if block.sources.is_empty() || block.sources.len() > profile.source.maximum_sources {
        return Err("reduced-SWIRL receipt source count");
    }
    let mut previous: Option<&ReducedSwirlSourceReceiptRecord> = None;
    for (source_index, source) in block.sources.iter().enumerate() {
        if source.segment_index as usize != block.source_offset as usize + source_index
            || F::from_usize(source.checkpoint_tidx).as_canonical_u32() as usize
                != source.checkpoint_tidx
            || source.layout.len() != profile.child_air_count
            || source.roots.is_empty()
            || source.roots.len() > profile.source.maximum_roots_per_source
            || source.roots.len() != source.widths.len()
            || source.stacking_point.len() != profile.source.log_message_len()
            || source.beta.len() != profile.source.log_message_len()
            || source.beta
                != source
                    .stacking_point
                    .iter()
                    .rev()
                    .copied()
                    .collect::<Vec<_>>()
            || source.stacking_openings.len() != source.roots.len()
            || source
                .stacking_openings
                .iter()
                .zip(&source.widths)
                .any(|(opening, width)| opening.len() != *width)
            || source.widths.iter().sum::<usize>() > profile.source.maximum_openings_per_source
            || source.widths.iter().any(|width| *width > u16::MAX as usize)
        {
            return Err("reduced-SWIRL receipt source shape");
        }
        for (air, layout) in source.layout.iter().enumerate() {
            if layout
                .log_height
                .is_some_and(|height| height > u16::MAX as usize)
            {
                return Err("reduced-SWIRL receipt trace height exceeds canonical u16");
            }
            let expected_cached = if layout.log_height.is_some() {
                profile.cached_global_indices[air].len()
            } else {
                0
            };
            if layout.cached_commitments.len() != expected_cached {
                return Err("reduced-SWIRL receipt trace layout");
            }
        }
        let program = source
            .layout
            .get(PROGRAM_AIR_ID)
            .and_then(|layout| layout.cached_commitments.get(PROGRAM_CACHED_TRACE_INDEX))
            .ok_or("reduced-SWIRL receipt program commitment")?;
        if *program != source.vm.program_commitment {
            return Err("reduced-SWIRL receipt program commitment mismatch");
        }
        if let Some(previous) = previous {
            if previous.vm.program_commitment != source.vm.program_commitment
                || previous.vm.final_pc != source.vm.initial_pc
                || previous.vm.final_root != source.vm.initial_root
            {
                return Err("reduced-SWIRL receipt VM continuity");
            }
        }
        let final_source = source_index + 1 == block.sources.len();
        if !final_source
            && (source.vm.exit_code != F::from_u32(profile.suspend_exit_code)
                || source.vm.is_terminate != F::ZERO)
        {
            return Err("reduced-SWIRL receipt intermediate termination");
        }
        if final_source
            && !((source.vm.exit_code == F::ZERO && source.vm.is_terminate == F::ONE)
                || (source.vm.exit_code == F::from_u32(profile.suspend_exit_code)
                    && source.vm.is_terminate == F::ZERO))
        {
            return Err("reduced-SWIRL receipt final status");
        }
        previous = Some(source);
    }
    Ok(())
}

fn canonical_transcript_logs(
    profile: &ReducedSwirlSourceReceiptProfile,
    block: &ReducedSwirlSourceReceiptBlock,
) -> Result<Vec<TranscriptLog<F, [F; POSEIDON2_WIDTH]>>, &'static str> {
    validate_block(profile, block)?;
    let mut logs = Vec::with_capacity(SOURCE_HASHES_PER_SOURCE * block.sources.len() + 1);
    for source in &block.sources {
        let layout = layout_observations(source)?;
        let pending = pending_observations(profile, source)?;
        let claim = claim_observations(profile, source)?;
        let layout_result = hash_observations(LAYOUT_DOMAIN, &layout)?;
        let pending_result = hash_observations(PENDING_DOMAIN, &pending)?;
        let claim_result = hash_observations(CLAIM_DOMAIN, &claim)?;
        if layout_result.0 != source.layout_digest
            || pending_result.0 != source.pending_digest
            || claim_result.0 != source.claim_digest
        {
            return Err("reduced-SWIRL canonical subdigest mismatch");
        }
        let entry = entry_observations(source.segment_index as usize, source);
        let entry_result = hash_observations(ENTRY_DOMAIN, &entry)?;
        if entry_result.0 != source.entry_digest {
            return Err("reduced-SWIRL canonical entry digest mismatch");
        }
        logs.extend([
            layout_result.1,
            pending_result.1,
            claim_result.1,
            entry_result.1,
        ]);
    }
    let manifest = manifest_transcript(
        &block
            .sources
            .iter()
            .map(|source| source.entry_digest)
            .collect::<Vec<_>>(),
    )?;
    if manifest.0 != block.manifest_digest {
        return Err("reduced-SWIRL canonical manifest mismatch");
    }
    logs.push(manifest.1);
    Ok(logs)
}

fn layout_observations(
    source: &ReducedSwirlSourceReceiptRecord,
) -> Result<Vec<CanonicalObservation>, &'static str> {
    let mut out = Vec::new();
    push_u32_obs(&mut out, source.layout.len())?;
    for (air, layout) in source.layout.iter().enumerate() {
        push_u32_obs(&mut out, air)?;
        out.push(CanonicalObservation::Field(F::from_bool(
            layout.log_height.is_some(),
        )));
        if let Some(log_height) = layout.log_height {
            push_u32_obs(&mut out, log_height)?;
            push_u32_obs(&mut out, layout.cached_commitments.len())?;
            out.extend(
                layout
                    .cached_commitments
                    .iter()
                    .copied()
                    .map(CanonicalObservation::Digest),
            );
        }
    }
    Ok(out)
}

fn pending_observations(
    profile: &ReducedSwirlSourceReceiptProfile,
    source: &ReducedSwirlSourceReceiptRecord,
) -> Result<Vec<CanonicalObservation>, &'static str> {
    let mut out = Vec::new();
    push_u32_obs(&mut out, source.roots.len())?;
    out.extend(
        source
            .roots
            .iter()
            .copied()
            .map(CanonicalObservation::Digest),
    );
    push_u32_obs(&mut out, source.widths.len())?;
    for &width in &source.widths {
        push_u32_obs(&mut out, width)?;
    }
    for value in [
        profile.source.l_skip,
        profile.source.n_stack,
        profile.source.log_blowup,
        profile.source.log_commit_rows_per_query,
        source.widths.iter().sum(),
    ] {
        push_u32_obs(&mut out, value)?;
    }
    push_u32_obs(&mut out, source.stacking_point.len())?;
    out.extend(
        source
            .stacking_point
            .iter()
            .copied()
            .map(CanonicalObservation::Extension),
    );
    push_u32_obs(&mut out, source.stacking_openings.len())?;
    for opening in &source.stacking_openings {
        push_u32_obs(&mut out, opening.len())?;
        out.extend(opening.iter().copied().map(CanonicalObservation::Extension));
    }
    Ok(out)
}

fn claim_observations(
    profile: &ReducedSwirlSourceReceiptProfile,
    source: &ReducedSwirlSourceReceiptRecord,
) -> Result<Vec<CanonicalObservation>, &'static str> {
    let mut out = Vec::new();
    push_u32_obs(&mut out, source.roots.len())?;
    out.extend(
        source
            .roots
            .iter()
            .copied()
            .map(CanonicalObservation::Digest),
    );
    push_u32_obs(&mut out, source.widths.len())?;
    for &width in &source.widths {
        push_u32_obs(&mut out, width)?;
    }
    for value in [
        profile.source.l_skip,
        profile.source.n_stack,
        profile.source.log_blowup,
        profile.source.log_commit_rows_per_query,
    ] {
        push_u32_obs(&mut out, value)?;
    }
    out.push(CanonicalObservation::Extension(source.theta));
    push_u32_obs(&mut out, profile.source.log_codeword_len())?;
    out.extend(std::iter::repeat_n(
        CanonicalObservation::Extension(EF::ZERO),
        profile.source.log_codeword_len(),
    ));
    out.push(CanonicalObservation::Extension(source.mu));
    push_u32_obs(&mut out, source.beta.len())?;
    out.extend(
        source
            .beta
            .iter()
            .copied()
            .map(CanonicalObservation::Extension),
    );
    out.push(CanonicalObservation::Extension(source.eta));
    Ok(out)
}

fn entry_observations(
    source_index: usize,
    source: &ReducedSwirlSourceReceiptRecord,
) -> Vec<CanonicalObservation> {
    let mut out = Vec::with_capacity(15);
    push_u32_obs(&mut out, source_index).expect("source capacity is u32");
    push_u32_obs(&mut out, source.segment_index as usize).expect("segment index is u32");
    out.extend([
        CanonicalObservation::Digest(source.roots[0]),
        CanonicalObservation::Digest(source.layout_digest),
        CanonicalObservation::Digest(source.pending_digest),
        CanonicalObservation::Digest(source.claim_digest),
        CanonicalObservation::Digest(source.vm.program_commitment),
        CanonicalObservation::Field(source.vm.initial_pc),
        CanonicalObservation::Digest(source.vm.initial_root),
        CanonicalObservation::Field(source.vm.final_pc),
        CanonicalObservation::Digest(source.vm.final_root),
        CanonicalObservation::Field(source.vm.exit_code),
        CanonicalObservation::Field(source.vm.is_terminate),
    ]);
    out
}

fn push_u32_obs(out: &mut Vec<CanonicalObservation>, value: usize) -> Result<(), &'static str> {
    let value = u32::try_from(value).map_err(|_| "reduced-SWIRL canonical u32")?;
    out.push(CanonicalObservation::Field(F::from_u32(value & 0xffff)));
    out.push(CanonicalObservation::Field(F::from_u32(value >> 16)));
    Ok(())
}

fn hash_observations(
    domain: u32,
    observations: &[CanonicalObservation],
) -> Result<(Digest, TranscriptLog<F, [F; POSEIDON2_WIDTH]>), &'static str> {
    let count = u32::try_from(observations.len()).map_err(|_| "reduced-SWIRL observation count")?;
    let mut transcript = default_duplex_sponge_recorder();
    for value in [
        F::from_u32(domain),
        F::from_u32(REDUCED_SWIRL_WRAPPER_PROTOCOL_VERSION),
        F::from_u32(count & 0xffff),
        F::from_u32(count >> 16),
    ] {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(&mut transcript, value);
    }
    for observation in observations {
        match observation {
            CanonicalObservation::Field(value) => {
                for value in [F::from_u32(OBS_FIELD_TAG), *value] {
                    <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
                        &mut transcript,
                        value,
                    );
                }
            }
            CanonicalObservation::Digest(digest) => {
                <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
                    &mut transcript,
                    F::from_u32(OBS_DIGEST_TAG),
                );
                for &value in digest {
                    <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
                        &mut transcript,
                        value,
                    );
                }
            }
            CanonicalObservation::Extension(value) => {
                <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
                    &mut transcript,
                    F::from_u32(OBS_EXTENSION_TAG),
                );
                for &limb in value.as_basis_coefficients_slice() {
                    <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
                        &mut transcript,
                        limb,
                    );
                }
            }
        }
    }
    <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
        &mut transcript,
        F::from_u32(OBS_END_TAG),
    );
    let digest = core::array::from_fn(|_| {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::sample(&mut transcript)
    });
    Ok((digest, transcript.into_log()))
}

fn manifest_transcript(
    entries: &[Digest],
) -> Result<(Digest, TranscriptLog<F, [F; POSEIDON2_WIDTH]>), &'static str> {
    if entries.is_empty()
        || entries
            .iter()
            .any(|digest| digest.iter().all(|x| *x == F::ZERO))
    {
        return Err("reduced-SWIRL manifest entries");
    }
    let mut transcript = default_duplex_sponge_recorder();
    for &byte in MANIFEST_TAG {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
            &mut transcript,
            F::from_u8(byte),
        );
    }
    <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
        &mut transcript,
        F::from_u32(REDUCED_SWIRL_WRAPPER_PROTOCOL_VERSION),
    );
    <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
        &mut transcript,
        F::from_usize(entries.len()),
    );
    for (source, digest) in entries.iter().enumerate() {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
            &mut transcript,
            F::from_usize(source),
        );
        for &limb in digest {
            <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(&mut transcript, limb);
        }
    }
    let digest = core::array::from_fn(|_| {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::sample(&mut transcript)
    });
    Ok((digest, transcript.into_log()))
}

fn generate_receipt_trace(
    profile: &ReducedSwirlSourceReceiptProfile,
    block: &ReducedSwirlSourceReceiptBlock,
) -> Result<RowMajorMatrix<F>, &'static str> {
    validate_block(profile, block)?;
    // Re-run all hashes before constructing an honest witness. This catches
    // SDK/circuit transcript drift before proving.
    let _ = canonical_transcript_logs(profile, block)?;
    let width = profile.trace_width();
    let mut values = F::zero_vec(width * profile.source.maximum_sources);
    let first = &block.sources[0];
    for (source_index, source) in block.sources.iter().enumerate() {
        let row = &mut values[source_index * width..(source_index + 1) * width];
        row[ACTIVE] = F::ONE;
        row[IS_LAST] = F::from_bool(source_index + 1 == block.sources.len());
        row[IS_FIRST] = F::from_bool(source_index == 0);
        row[SOURCE] = F::from_usize(source_index);
        row[SEGMENT] = F::from_u32(source.segment_index);
        row[SOURCE_COUNT] = F::from_usize(block.sources.len());
        row[ROOT_COUNT] = F::from_usize(source.roots.len());
        row[OPENING_COUNT] = F::from_usize(source.widths.iter().sum());
        row[CHECKPOINT_TIDX] = F::from_usize(source.checkpoint_tidx);
        row[CHECKPOINT_SAMPLES..CHECKPOINT_STATE]
            .copy_from_slice(source.checkpoint_samples.as_basis_coefficients_slice());
        row[CHECKPOINT_STATE..PROGRAM].copy_from_slice(&source.checkpoint_state);
        row[PROGRAM..INITIAL_PC].copy_from_slice(&source.vm.program_commitment);
        row[INITIAL_PC] = source.vm.initial_pc;
        row[INITIAL_ROOT..FINAL_PC].copy_from_slice(&source.vm.initial_root);
        row[FINAL_PC] = source.vm.final_pc;
        row[FINAL_ROOT..EXIT_CODE].copy_from_slice(&source.vm.final_root);
        row[EXIT_CODE] = source.vm.exit_code;
        row[IS_TERMINATE] = source.vm.is_terminate;
        row[CHAIN_INITIAL_PC] = first.vm.initial_pc;
        row[CHAIN_INITIAL_ROOT..LAYOUT_DIGEST].copy_from_slice(&first.vm.initial_root);
        row[LAYOUT_DIGEST..PENDING_DIGEST].copy_from_slice(&source.layout_digest);
        row[PENDING_DIGEST..CLAIM_DIGEST].copy_from_slice(&source.pending_digest);
        row[CLAIM_DIGEST..ENTRY_DIGEST].copy_from_slice(&source.claim_digest);
        row[ENTRY_DIGEST..MANIFEST_DIGEST].copy_from_slice(&source.entry_digest);
        row[MANIFEST_DIGEST..THETA].copy_from_slice(&block.manifest_digest);
        row[THETA..MU].copy_from_slice(source.theta.as_basis_coefficients_slice());
        row[MU..ETA].copy_from_slice(source.mu.as_basis_coefficients_slice());
        row[ETA..HEADER_WIDTH].copy_from_slice(source.eta.as_basis_coefficients_slice());

        let mut present = source
            .layout
            .iter()
            .enumerate()
            .filter_map(|(air, layout)| layout.log_height.map(|height| (air, height)))
            .collect::<Vec<_>>();
        present.sort_by_key(|(air, height)| (std::cmp::Reverse(*height), *air));
        let sorted = present
            .iter()
            .enumerate()
            .map(|(sort, (air, _))| (*air, sort))
            .collect::<std::collections::BTreeMap<_, _>>();
        for (air, layout) in source.layout.iter().enumerate() {
            let Some(log_height) = layout.log_height else {
                continue;
            };
            let offset = profile.layout_offset(air);
            row[offset + LAYOUT_PRESENT] = F::ONE;
            row[offset + LAYOUT_SORT_INDEX] = F::from_usize(sorted[&air]);
            let (n_sign, n_abs) = if log_height < profile.child_l_skip {
                (true, profile.child_l_skip - log_height)
            } else {
                (false, log_height - profile.child_l_skip)
            };
            row[offset + LAYOUT_N_ABS] = F::from_usize(n_abs);
            row[offset + LAYOUT_N_SIGN] = F::from_bool(n_sign);
            for (cached, digest) in layout.cached_commitments.iter().enumerate() {
                let start = offset + LAYOUT_HEADER_WIDTH + cached * DIGEST_SIZE;
                row[start..start + DIGEST_SIZE].copy_from_slice(digest);
            }
        }
        for ordinal in 0..source.roots.len() {
            let offset = profile.roots_offset() + ordinal * ROOT_SLOT_WIDTH;
            row[offset + ROOT_ACTIVE] = F::ONE;
            row[offset + ROOT_WIDTH] = F::from_usize(source.widths[ordinal]);
            row[offset + ROOT_DIGEST..offset + ROOT_SLOT_WIDTH]
                .copy_from_slice(&source.roots[ordinal]);
        }
        for coordinate in 0..profile.source.log_message_len() {
            let point = profile.points_offset() + coordinate * D_EF;
            let beta = profile.betas_offset() + coordinate * D_EF;
            row[point..point + D_EF]
                .copy_from_slice(source.stacking_point[coordinate].as_basis_coefficients_slice());
            row[beta..beta + D_EF]
                .copy_from_slice(source.beta[coordinate].as_basis_coefficients_slice());
        }
        let mut opening_index = 0usize;
        for (root, opening) in source.stacking_openings.iter().enumerate() {
            for (column, value) in opening.iter().enumerate() {
                let offset = profile.openings_offset() + opening_index * OPENING_SLOT_WIDTH;
                row[offset + OPENING_ACTIVE] = F::ONE;
                row[offset + OPENING_ROOT] = F::from_usize(root);
                row[offset + OPENING_COLUMN] = F::from_usize(column);
                row[offset + OPENING_FIRST] = F::from_bool(column == 0);
                row[offset + OPENING_COLUMN_INVERSE] = if column == 0 {
                    F::ZERO
                } else {
                    F::from_usize(column).inverse()
                };
                if column == 0 {
                    row[offset + OPENING_GROUP_WIDTH] = F::from_usize(source.widths[root]);
                }
                row[offset + OPENING_VALUE..offset + OPENING_SLOT_WIDTH]
                    .copy_from_slice(value.as_basis_coefficients_slice());
                opening_index += 1;
            }
        }
    }
    Ok(RowMajorMatrix::new(values, width))
}

// Transcript helpers match SDK `digest_observations` exactly. `tidx` is the
// concrete transcript operation index, while `count` counts typed observations.
fn preamble<AB: AirBuilder<F = F> + InteractionBuilder>(
    builder: &mut AB,
    bus: TranscriptBus,
    proof: AB::Expr,
    domain: u32,
    count: AB::Expr,
    enabled: AB::Expr,
) -> AB::Expr {
    for (index, value) in [
        AB::Expr::from_u32(domain),
        AB::Expr::from_u32(REDUCED_SWIRL_WRAPPER_PROTOCOL_VERSION),
        count.clone(),
        AB::Expr::ZERO,
    ]
    .into_iter()
    .enumerate()
    {
        bus.observe(
            builder,
            proof.clone(),
            AB::Expr::from_usize(index),
            value,
            enabled.clone(),
        );
    }
    AB::Expr::from_usize(4)
}
fn typed_field<AB: AirBuilder<F = F> + InteractionBuilder>(
    builder: &mut AB,
    bus: TranscriptBus,
    proof: AB::Expr,
    t: &mut AB::Expr,
    value: AB::Expr,
    enabled: AB::Expr,
) {
    bus.observe(
        builder,
        proof.clone(),
        t.clone(),
        AB::Expr::from_u32(OBS_FIELD_TAG),
        enabled.clone(),
    );
    bus.observe(
        builder,
        proof,
        t.clone() + AB::Expr::ONE,
        value,
        enabled.clone(),
    );
    // `tidx` indexes operations that actually occur in the canonical
    // transcript. Optional observations are omitted by the host recorder,
    // so a disabled typed field must not leave a hole in the AIR cursor.
    *t = t.clone() + AB::Expr::from_u32(2) * enabled;
}
fn typed_u32<AB: AirBuilder<F = F> + InteractionBuilder>(
    builder: &mut AB,
    bus: TranscriptBus,
    proof: AB::Expr,
    t: &mut AB::Expr,
    value: AB::Expr,
    enabled: AB::Expr,
) {
    typed_field(builder, bus, proof.clone(), t, value, enabled.clone());
    typed_field(builder, bus, proof, t, AB::Expr::ZERO, enabled);
}
fn typed_digest<AB: AirBuilder<F = F> + InteractionBuilder>(
    builder: &mut AB,
    bus: TranscriptBus,
    proof: AB::Expr,
    t: &mut AB::Expr,
    digest: [AB::Expr; DIGEST_SIZE],
    enabled: AB::Expr,
) {
    bus.observe(
        builder,
        proof.clone(),
        t.clone(),
        AB::Expr::from_u32(OBS_DIGEST_TAG),
        enabled.clone(),
    );
    for (i, v) in digest.into_iter().enumerate() {
        bus.observe(
            builder,
            proof.clone(),
            t.clone() + AB::Expr::from_usize(1 + i),
            v,
            enabled.clone(),
        );
    }
    *t = t.clone() + AB::Expr::from_usize(1 + DIGEST_SIZE) * enabled;
}
fn typed_extension<AB: AirBuilder<F = F> + InteractionBuilder>(
    builder: &mut AB,
    bus: TranscriptBus,
    proof: AB::Expr,
    t: &mut AB::Expr,
    value: [AB::Expr; D_EF],
    enabled: AB::Expr,
) {
    bus.observe(
        builder,
        proof.clone(),
        t.clone(),
        AB::Expr::from_u32(OBS_EXTENSION_TAG),
        enabled.clone(),
    );
    for (i, v) in value.into_iter().enumerate() {
        bus.observe(
            builder,
            proof.clone(),
            t.clone() + AB::Expr::from_usize(1 + i),
            v,
            enabled.clone(),
        );
    }
    *t = t.clone() + AB::Expr::from_usize(1 + D_EF) * enabled;
}
fn finish_hash<AB: AirBuilder<F = F> + InteractionBuilder>(
    builder: &mut AB,
    bus: TranscriptBus,
    proof: AB::Expr,
    t: AB::Expr,
    digest: &[AB::Var],
    enabled: AB::Expr,
) where
    AB::Var: Copy,
{
    bus.observe(
        builder,
        proof.clone(),
        t.clone(),
        AB::Expr::from_u32(OBS_END_TAG),
        enabled.clone(),
    );
    for (i, v) in digest.iter().enumerate() {
        bus.sample(
            builder,
            proof.clone(),
            t.clone() + AB::Expr::from_usize(1 + i),
            *v,
            enabled.clone(),
        );
    }
}
fn expr<AB: AirBuilder>(value: AB::Var) -> AB::Expr
where
    AB::Var: Copy,
{
    value.into()
}
fn array<AB: AirBuilder, const N: usize>(row: &[AB::Var], offset: usize) -> [AB::Expr; N]
where
    AB::Var: Copy,
{
    core::array::from_fn(|i| row[offset + i].into())
}
fn constant_digest<AB: AirBuilder<F = F>>(digest: Digest) -> [AB::Expr; DIGEST_SIZE] {
    digest.map(|v| AB::Expr::from_u32(v.as_canonical_u32()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: u32) -> Digest {
        core::array::from_fn(|limb| F::from_u32(seed + limb as u32 + 1))
    }

    fn profile() -> ReducedSwirlSourceReceiptProfile {
        let child_air_count = MERKLE_AIR_ID + 1;
        let mut cached_global_indices = vec![Vec::new(); child_air_count];
        cached_global_indices[PROGRAM_AIR_ID] = vec![0];
        ReducedSwirlSourceReceiptProfile {
            source: ReducedSwirlSourceProfile {
                maximum_sources: REDUCED_SWIRL_WRAPPER_MAX_SOURCES as usize,
                maximum_roots_per_source: 2,
                maximum_openings_per_source: 3,
                l_skip: 1,
                n_stack: 1,
                log_blowup: 1,
                log_commit_rows_per_query: 0,
            },
            protocol_digest: digest(100),
            child_vk_pre_hash: digest(120),
            child_air_count,
            child_l_skip: 1,
            cached_global_indices,
            suspend_exit_code: 2,
        }
    }

    fn source(
        index: usize,
        program: Digest,
        initial_pc: F,
        initial_root: Digest,
        final_pc: F,
        final_root: Digest,
        is_last: bool,
    ) -> ReducedSwirlSourceReceiptRecord {
        let mut layout = vec![
            ReducedSwirlReceiptLayoutEntry {
                log_height: None,
                cached_commitments: Vec::new(),
            };
            MERKLE_AIR_ID + 1
        ];
        layout[PROGRAM_AIR_ID] = ReducedSwirlReceiptLayoutEntry {
            log_height: Some(4),
            cached_commitments: vec![program],
        };
        let point = vec![
            EF::from_u32(11 + index as u32),
            EF::from_u32(21 + index as u32),
        ];
        ReducedSwirlSourceReceiptRecord {
            segment_index: index as u32,
            checkpoint_tidx: 64 + index * 4,
            checkpoint_samples: EF::from_u32(31 + index as u32),
            checkpoint_state: core::array::from_fn(|limb| {
                F::from_u32(200 + 20 * index as u32 + limb as u32)
            }),
            layout,
            roots: vec![digest(300 + index as u32 * 20)],
            widths: vec![1],
            stacking_point: point.clone(),
            stacking_openings: vec![vec![EF::from_u32(41 + index as u32)]],
            theta: EF::from_u32(51 + index as u32),
            mu: EF::from_u32(61 + index as u32),
            beta: point.into_iter().rev().collect(),
            eta: EF::from_u32(71 + index as u32),
            vm: ReducedSwirlReceiptVmBoundary {
                program_commitment: program,
                initial_pc,
                initial_root,
                final_pc,
                final_root,
                exit_code: F::from_u32(if is_last { 0 } else { 2 }),
                is_terminate: F::from_bool(is_last),
            },
            layout_digest: [F::ZERO; DIGEST_SIZE],
            pending_digest: [F::ZERO; DIGEST_SIZE],
            claim_digest: [F::ZERO; DIGEST_SIZE],
            entry_digest: [F::ZERO; DIGEST_SIZE],
        }
    }

    fn canonical_block() -> ReducedSwirlSourceReceiptBlock {
        let program = digest(10);
        let first_initial = digest(20);
        let middle = digest(30);
        let final_root = digest(40);
        let mut block = ReducedSwirlSourceReceiptBlock {
            source_offset: 0,
            sources: vec![
                source(
                    0,
                    program,
                    F::from_u32(3),
                    first_initial,
                    F::from_u32(7),
                    middle,
                    false,
                ),
                source(
                    1,
                    program,
                    F::from_u32(7),
                    middle,
                    F::from_u32(9),
                    final_root,
                    true,
                ),
            ],
            manifest_digest: [F::ZERO; DIGEST_SIZE],
        };
        canonicalize_reduced_swirl_source_receipt_block(&profile(), &mut block).unwrap();
        block
    }

    #[test]
    fn canonical_active_prefix_has_fixed_capacity_and_stable_manifest() {
        let block = canonical_block();
        let traces = generate_receipt_trace(&profile(), &block).unwrap();
        assert_eq!(traces.height(), REDUCED_SWIRL_WRAPPER_MAX_SOURCES as usize);
        assert_eq!(traces.width(), profile().trace_width());
        assert_eq!(
            manifest_transcript(
                &block
                    .sources
                    .iter()
                    .map(|source| source.entry_digest)
                    .collect::<Vec<_>>()
            )
            .unwrap()
            .0,
            block.manifest_digest
        );
    }

    #[test]
    fn omitted_reordered_and_duplicated_sources_are_rejected() {
        let original = canonical_block();

        let mut omitted = original.clone();
        omitted.sources.pop();
        assert!(canonical_transcript_logs(&profile(), &omitted).is_err());

        let mut reordered = original.clone();
        reordered.sources.swap(0, 1);
        assert!(canonical_transcript_logs(&profile(), &reordered).is_err());

        let mut duplicated = original.clone();
        duplicated.sources[1] = duplicated.sources[0].clone();
        assert!(canonical_transcript_logs(&profile(), &duplicated).is_err());
    }

    #[test]
    fn boundary_and_claim_mutations_are_rejected() {
        let original = canonical_block();

        let mut boundary = original.clone();
        boundary.sources[1].vm.initial_pc += F::ONE;
        assert!(canonical_transcript_logs(&profile(), &boundary).is_err());

        let mut claim = original.clone();
        claim.sources[0].theta += EF::ONE;
        assert!(canonical_transcript_logs(&profile(), &claim).is_err());

        let mut opening = original;
        opening.sources[0].stacking_openings[0][0] += EF::ONE;
        assert!(canonical_transcript_logs(&profile(), &opening).is_err());
    }

    #[test]
    fn nonzero_offset_nonterminal_chunk_is_canonical() {
        let program = digest(10);
        let initial_root = digest(20);
        let middle = digest(30);
        let final_root = digest(40);
        let mut block = ReducedSwirlSourceReceiptBlock {
            source_offset: 7,
            sources: vec![
                source(
                    7,
                    program,
                    F::from_u32(3),
                    initial_root,
                    F::from_u32(7),
                    middle,
                    false,
                ),
                source(
                    8,
                    program,
                    F::from_u32(7),
                    middle,
                    F::from_u32(9),
                    final_root,
                    false,
                ),
            ],
            manifest_digest: [F::ZERO; DIGEST_SIZE],
        };
        canonicalize_reduced_swirl_source_receipt_block(&profile(), &mut block).unwrap();
        assert_eq!(block.sources[0].segment_index, 7);
        assert_eq!(block.sources[1].segment_index, 8);
        assert_eq!(block.sources[1].vm.is_terminate, F::ZERO);
        assert_eq!(
            block.sources[1].vm.exit_code,
            F::from_u32(profile().suspend_exit_code)
        );
        canonical_transcript_logs(&profile(), &block).unwrap();
        let trace = generate_receipt_trace(&profile(), &block).unwrap();
        let first = trace.row_slice(0).unwrap();
        assert_eq!(first[SOURCE], F::ZERO);
        assert_eq!(first[SEGMENT], F::from_u32(7));
    }

    #[test]
    fn source_offset_and_global_range_mismatches_are_rejected() {
        let mut block = canonical_block();
        block.source_offset = 1;
        assert!(canonicalize_reduced_swirl_source_receipt_block(&profile(), &mut block).is_err());

        let mut block = canonical_block();
        block.sources[1].segment_index = 3;
        assert!(canonicalize_reduced_swirl_source_receipt_block(&profile(), &mut block).is_err());
    }
}

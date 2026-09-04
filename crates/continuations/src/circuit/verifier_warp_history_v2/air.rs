use core::borrow::Borrow;

use openvm_circuit::system::connector::DEFAULT_SUSPEND_EXIT_CODE;
use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit::{
    bus::{Poseidon2CompressBus, Poseidon2CompressMessage},
    define_typed_lookup_bus,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, F};

use super::{
    VerifierWarpHistoryChunkPublicValuesV3, VerifierWarpHistoryPublicValuesV2, HISTORY_END_TAG_V2,
    HISTORY_ROW_HASH_STEP_COUNT_V2, HISTORY_START_TAG_V2, MANIFEST_ACCUMULATORS_TAG_V2,
    MANIFEST_CHILDREN_TAG_V2, MANIFEST_CHILD_TAG_V2, MANIFEST_END_TAG_V2, MANIFEST_HEADER_TAG_V2,
    MANIFEST_PUBLIC_VALUES_TAG_V2, MANIFEST_SOURCE_AUTH_TAG_V2, MANIFEST_SOURCE_TAG_V2,
    MANIFEST_START_TAG_V2, STATEMENT_DIGEST_COMMITMENT_TAG_V2, STATEMENT_DIGEST_END_TAG_V2,
    STATEMENT_DIGEST_FIELD_TAG_V2, STATEMENT_DIGEST_START_TAG_V2,
    VERIFIER_WARP_HISTORY_CHUNK_PROTOCOL_V3, VERIFIER_WARP_HISTORY_PROTOCOL_V2,
    VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2, VERIFIER_WARP_VACC_INPUT_ARITY_V2,
};
use crate::circuit::verifier_warp_history_chunk_v3::{
    VerifierWarpHistoryChunkIntervalBusV3, VerifierWarpHistoryChunkIntervalMessageV3,
    VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3,
};

const LIMB_BITS: usize = 16;
const LIMB_BASE: u32 = 1 << LIMB_BITS;

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug, PartialEq, Eq)]
pub struct VerifierWarpCertifiedChildMessageV2<T> {
    pub occupied: T,
    pub input_pc: T,
    pub input_memory_root: [T; DIGEST_SIZE],
    pub output_pc: T,
    pub output_memory_root: [T; DIGEST_SIZE],
    pub exit_code: T,
    pub terminates: T,
}

/// Statement emitted only by the fixed multi-AIR source verifier bridge.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct VerifierWarpSourceCertificateMessageV2<T> {
    pub protocol_version: T,
    pub source_child_capacity: T,
    pub batch_index_lo: T,
    pub batch_index_hi: T,
    pub active_child_count: T,
    pub protocol_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub program_commitment: [T; DIGEST_SIZE],
    pub children: [VerifierWarpCertifiedChildMessageV2<T>; VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2],
    pub source_accumulator_digest: [T; DIGEST_SIZE],
    pub source_commitment_root: [T; DIGEST_SIZE],
    pub external_logup_gkr_digest: [T; DIGEST_SIZE],
    pub source_functional_digest: [T; DIGEST_SIZE],
    pub setup_openings_digest: [T; DIGEST_SIZE],
    pub source_statement_digest: [T; DIGEST_SIZE],
}

define_typed_lookup_bus!(
    VerifierWarpSourceCertificateBusV2,
    VerifierWarpSourceCertificateMessageV2
);

/// Statement emitted only by the standard arity-two VACC verifier bridge.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct VerifierWarpVaccCertificateMessageV2<T> {
    pub protocol_version: T,
    pub input_arity: T,
    pub batch_index_lo: T,
    pub batch_index_hi: T,
    pub relation_digest: [T; DIGEST_SIZE],
    pub prior_accumulator_digest: [T; DIGEST_SIZE],
    pub source_accumulator_digest: [T; DIGEST_SIZE],
    pub output_accumulator_digest: [T; DIGEST_SIZE],
    pub source_commitment_root: [T; DIGEST_SIZE],
    pub transition_transcript_digest: [T; DIGEST_SIZE],
}

define_typed_lookup_bus!(
    VerifierWarpVaccCertificateBusV2,
    VerifierWarpVaccCertificateMessageV2
);

/// Terminal user-public-values statement.  Its production AIR verifies the
/// ordinary OpenVM public-values Merkle proof against `final_memory_root`.
/// History only consumes this message on its final active row.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct VerifierWarpBlockPublicValuesMessageV2<T> {
    pub final_memory_root: [T; DIGEST_SIZE],
    pub public_values_digest: [T; DIGEST_SIZE],
}

define_typed_lookup_bus!(
    VerifierWarpBlockPublicValuesBusV2,
    VerifierWarpBlockPublicValuesMessageV2
);

/// Complete aggregate History endpoint. Its field order is exactly the former
/// 83-cell standalone History public row, but direct-final production obtains
/// it only from [`VerifierWarpHistoryAirV2`] over this typed provider bus.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct VerifierWarpHistoryEndpointMessageV2<T> {
    pub protocol_version: T,
    pub source_child_capacity: T,
    pub vacc_input_arity: T,
    pub protocol_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub program_commitment: [T; DIGEST_SIZE],
    pub batch_count: [T; 2],
    pub active_child_count: [T; 4],
    pub initial_pc: T,
    pub initial_memory_root: [T; DIGEST_SIZE],
    pub final_pc: T,
    pub final_memory_root: [T; DIGEST_SIZE],
    pub genesis_accumulator_digest: [T; DIGEST_SIZE],
    pub final_accumulator_digest: [T; DIGEST_SIZE],
    pub public_values_digest: [T; DIGEST_SIZE],
    pub history_statement_digest: [T; DIGEST_SIZE],
}

define_typed_lookup_bus!(
    VerifierWarpHistoryEndpointBusV2,
    VerifierWarpHistoryEndpointMessageV2
);

impl<T: Clone> VerifierWarpHistoryEndpointMessageV2<T> {
    /// Canonical former-PV field order. Keeping this conversion next to the
    /// typed message prevents standalone and direct-final encodings drifting.
    #[must_use]
    pub fn flatten(&self) -> Vec<T> {
        let mut values = Vec::with_capacity(VerifierWarpHistoryPublicValuesV2::WIDTH);
        values.extend([
            self.protocol_version.clone(),
            self.source_child_capacity.clone(),
            self.vacc_input_arity.clone(),
        ]);
        values.extend(self.protocol_digest.iter().cloned());
        values.extend(self.relation_digest.iter().cloned());
        values.extend(self.program_commitment.iter().cloned());
        values.extend(self.batch_count.iter().cloned());
        values.extend(self.active_child_count.iter().cloned());
        values.push(self.initial_pc.clone());
        values.extend(self.initial_memory_root.iter().cloned());
        values.push(self.final_pc.clone());
        values.extend(self.final_memory_root.iter().cloned());
        values.extend(self.genesis_accumulator_digest.iter().cloned());
        values.extend(self.final_accumulator_digest.iter().cloned());
        values.extend(self.public_values_digest.iter().cloned());
        values.extend(self.history_statement_digest.iter().cloned());
        debug_assert_eq!(values.len(), VerifierWarpHistoryPublicValuesV2::WIDTH);
        values
    }
}

/// Complete public boundary of one bounded History interval.  Recursive
/// composition consumes every field and republishes the merged interval; the
/// verifier never accepts only a hash of this message.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct VerifierWarpHistoryChunkEndpointMessageV3<T> {
    pub chunk_protocol_version: T,
    pub source_history_protocol_version: T,
    pub source_child_capacity: T,
    pub vacc_input_arity: T,
    pub protocol_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub program_commitment: [T; DIGEST_SIZE],
    pub total_batch_count: [T; 2],
    pub chunk_index: [T; 2],
    pub start_batch_index: [T; 2],
    pub end_batch_index: [T; 2],
    pub active_child_count_before: [T; 4],
    pub active_child_count_after: [T; 4],
    pub initial_pc: T,
    pub initial_memory_root: [T; DIGEST_SIZE],
    pub final_pc: T,
    pub final_memory_root: [T; DIGEST_SIZE],
    pub initial_accumulator_digest: [T; DIGEST_SIZE],
    pub final_accumulator_digest: [T; DIGEST_SIZE],
    pub history_hash_before: [T; DIGEST_SIZE],
    pub history_hash_after: [T; DIGEST_SIZE],
    pub observation_count_before: [T; 2],
    pub observation_count_after: [T; 2],
    pub is_genesis: T,
    pub is_terminal: T,
    pub public_values_digest: [T; DIGEST_SIZE],
}

impl<T: Clone> VerifierWarpHistoryChunkEndpointMessageV3<T> {
    #[must_use]
    pub fn flatten(&self) -> Vec<T> {
        let mut values = Vec::with_capacity(VerifierWarpHistoryChunkPublicValuesV3::WIDTH);
        values.extend([
            self.chunk_protocol_version.clone(),
            self.source_history_protocol_version.clone(),
            self.source_child_capacity.clone(),
            self.vacc_input_arity.clone(),
        ]);
        values.extend(self.protocol_digest.iter().cloned());
        values.extend(self.relation_digest.iter().cloned());
        values.extend(self.program_commitment.iter().cloned());
        values.extend(self.total_batch_count.iter().cloned());
        values.extend(self.chunk_index.iter().cloned());
        values.extend(self.start_batch_index.iter().cloned());
        values.extend(self.end_batch_index.iter().cloned());
        values.extend(self.active_child_count_before.iter().cloned());
        values.extend(self.active_child_count_after.iter().cloned());
        values.push(self.initial_pc.clone());
        values.extend(self.initial_memory_root.iter().cloned());
        values.push(self.final_pc.clone());
        values.extend(self.final_memory_root.iter().cloned());
        values.extend(self.initial_accumulator_digest.iter().cloned());
        values.extend(self.final_accumulator_digest.iter().cloned());
        values.extend(self.history_hash_before.iter().cloned());
        values.extend(self.history_hash_after.iter().cloned());
        values.extend(self.observation_count_before.iter().cloned());
        values.extend(self.observation_count_after.iter().cloned());
        values.extend([self.is_genesis.clone(), self.is_terminal.clone()]);
        values.extend(self.public_values_digest.iter().cloned());
        debug_assert_eq!(values.len(), VerifierWarpHistoryChunkPublicValuesV3::WIDTH);
        values
    }
}

/// Setup-fixed History output architecture.
#[derive(Clone, Copy, Debug)]
pub enum VerifierWarpHistoryOutputModeV2 {
    /// Reference-only standalone History STARK with the canonical 83-cell PV
    /// row. No endpoint lookup is emitted.
    StandalonePublicValues,
    /// Independently proved bounded interval.  Its 108 public cells are
    /// exhaustively consumed by the recursive History composition circuit.
    ChunkPublicValuesV3,
    /// Current bounded interval embedded in a recursive composition stage.
    /// The complete 108-field statement is emitted on a typed bus and the AIR
    /// has no standalone public values; the composition AIR owns the stage's
    /// merged public statement.
    ChunkTypedProviderV3 {
        interval_bus: VerifierWarpHistoryChunkIntervalBusV3,
        lookup_count: u32,
    },
    /// Production direct-final mode. History has zero PVs and is the sole
    /// positive provider for the complete endpoint message.
    DirectFinalProvider {
        endpoint_bus: VerifierWarpHistoryEndpointBusV2,
        lookup_count: u32,
    },
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct VerifierWarpVmStateColsV2<T> {
    pub pc: T,
    pub memory_root: [T; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct VerifierWarpChildColsV2<T> {
    pub occupied: T,
    pub input: VerifierWarpVmStateColsV2<T>,
    pub output: VerifierWarpVmStateColsV2<T>,
    pub exit_code: T,
    pub terminates: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct VerifierWarpHistoryRowColsV2<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub protocol_version: T,
    pub vacc_input_arity: T,
    pub protocol_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub batch_index: [T; 2],
    pub batch_index_bits: [[T; LIMB_BITS]; 2],
    pub batch_index_carry: T,
    pub active_child_count: T,
    pub active_count_flags: [T; VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2],
    pub children: [VerifierWarpChildColsV2<T>; VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2],
    pub program_commitment: [T; DIGEST_SIZE],
    pub prior_accumulator_digest: [T; DIGEST_SIZE],
    pub source_accumulator_digest: [T; DIGEST_SIZE],
    pub output_accumulator_digest: [T; DIGEST_SIZE],
    pub source_commitment_root: [T; DIGEST_SIZE],
    pub external_logup_gkr_digest: [T; DIGEST_SIZE],
    pub source_functional_digest: [T; DIGEST_SIZE],
    pub setup_openings_digest: [T; DIGEST_SIZE],
    pub source_statement_digest: [T; DIGEST_SIZE],
    pub transition_transcript_digest: [T; DIGEST_SIZE],
    pub public_values_digest: [T; DIGEST_SIZE],
    pub active_total_before: [T; 4],
    pub active_total_after: [T; 4],
    pub active_total_before_bits: [[T; LIMB_BITS]; 4],
    pub active_total_after_bits: [[T; LIMB_BITS]; 4],
    pub active_total_carries: [T; 4],
    pub observation_count_before: [T; 2],
    pub observation_count_after: [T; 2],
    pub observation_count_before_bits: [[T; LIMB_BITS]; 2],
    pub observation_count_after_bits: [[T; LIMB_BITS]; 2],
    pub observation_count_carries: [T; 2],
    pub history_hash_before: [T; DIGEST_SIZE],
    pub history_hash_after: [T; DIGEST_SIZE],
    pub hash_outputs: [[T; DIGEST_SIZE]; HISTORY_ROW_HASH_STEP_COUNT_V2],
    /// First-row and total-history values carried to the final provider row.
    /// These replace the standalone public row as authority in direct mode.
    pub endpoint_batch_count: [T; 2],
    pub endpoint_batch_count_bits: [[T; LIMB_BITS]; 2],
    pub endpoint_chunk_index: [T; 2],
    pub endpoint_chunk_index_bits: [[T; LIMB_BITS]; 2],
    pub endpoint_batch_start: [T; 2],
    pub endpoint_batch_end: [T; 2],
    pub endpoint_batch_end_bits: [[T; LIMB_BITS]; 2],
    pub endpoint_active_total_start: [T; 4],
    pub endpoint_observation_count_start: [T; 2],
    pub endpoint_history_hash_start: [T; DIGEST_SIZE],
    pub endpoint_is_genesis: T,
    pub endpoint_is_terminal: T,
    pub endpoint_protocol_digest: [T; DIGEST_SIZE],
    pub endpoint_relation_digest: [T; DIGEST_SIZE],
    pub endpoint_program_commitment: [T; DIGEST_SIZE],
    pub endpoint_initial_pc: T,
    pub endpoint_initial_memory_root: [T; DIGEST_SIZE],
    pub endpoint_genesis_accumulator_digest: [T; DIGEST_SIZE],
}

#[derive(Clone, ColumnsAir)]
#[columns_via(VerifierWarpHistoryRowColsV2<u8>)]
pub struct VerifierWarpHistoryAirV2 {
    pub source_bus: VerifierWarpSourceCertificateBusV2,
    pub vacc_bus: VerifierWarpVaccCertificateBusV2,
    pub block_public_values_bus: VerifierWarpBlockPublicValuesBusV2,
    pub compress_bus: Poseidon2CompressBus,
    pub output_mode: VerifierWarpHistoryOutputModeV2,
}

impl BaseAir<F> for VerifierWarpHistoryAirV2 {
    fn width(&self) -> usize {
        VerifierWarpHistoryRowColsV2::<F>::width()
    }
}

impl BaseAirWithPublicValues<F> for VerifierWarpHistoryAirV2 {
    fn num_public_values(&self) -> usize {
        match self.output_mode {
            VerifierWarpHistoryOutputModeV2::StandalonePublicValues => {
                VerifierWarpHistoryPublicValuesV2::WIDTH
            }
            VerifierWarpHistoryOutputModeV2::ChunkPublicValuesV3 => {
                VerifierWarpHistoryChunkPublicValuesV3::WIDTH
            }
            VerifierWarpHistoryOutputModeV2::ChunkTypedProviderV3 { .. } => 0,
            VerifierWarpHistoryOutputModeV2::DirectFinalProvider { .. } => 0,
        }
    }
}

impl PartitionedBaseAir<F> for VerifierWarpHistoryAirV2 {}

impl<AB> Air<AB> for VerifierWarpHistoryAirV2
where
    AB: AirBuilder<F = F> + AirBuilderWithPublicValues + InteractionBuilder,
    AB::Var: Copy,
    AB::PublicVar: Copy,
    AB::Expr: From<AB::PublicVar>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("verifier-WARP History row");
        let next_row = main.row_slice(1).expect("verifier-WARP History next row");
        let local: &VerifierWarpHistoryRowColsV2<AB::Var> = (*local_row).borrow();
        let next: &VerifierWarpHistoryRowColsV2<AB::Var> = (*next_row).borrow();
        let public = builder.public_values().to_vec();

        self.eval_activity(builder, local, next);
        self.eval_endpoint_carry(builder, local, next);
        self.eval_protocol(builder, local, next);
        self.eval_children(builder, local);
        self.eval_continuity(builder, local, next);
        self.eval_counters(builder, local, next);
        self.eval_certificates(builder, local);
        self.eval_history_hash(builder, local, next);
        self.eval_output(builder, local, &public);
    }
}

impl VerifierWarpHistoryAirV2 {
    fn eval_endpoint_carry<AB: AirBuilder<F = F>>(
        &self,
        builder: &mut AB,
        local: &VerifierWarpHistoryRowColsV2<AB::Var>,
        next: &VerifierWarpHistoryRowColsV2<AB::Var>,
    ) where
        AB::Var: Copy,
    {
        builder
            .when(local.active)
            .assert_bool(local.endpoint_is_genesis);
        builder
            .when(local.active)
            .assert_bool(local.endpoint_is_terminal);
        for (value, bits) in local
            .endpoint_batch_count
            .iter()
            .zip(&local.endpoint_batch_count_bits)
            .chain(
                local
                    .endpoint_chunk_index
                    .iter()
                    .zip(&local.endpoint_chunk_index_bits),
            )
            .chain(
                local
                    .endpoint_batch_end
                    .iter()
                    .zip(&local.endpoint_batch_end_bits),
            )
        {
            self.assert_limb(builder, local.active, *value, bits);
        }
        builder
            .when_first_row()
            .assert_eq(local.endpoint_initial_pc, local.children[0].input.pc);
        for limb in 0..2 {
            builder
                .when_first_row()
                .assert_eq(local.endpoint_batch_start[limb], local.batch_index[limb]);
            builder.when_first_row().assert_eq(
                local.endpoint_observation_count_start[limb],
                local.observation_count_before[limb],
            );
            builder
                .when(local.endpoint_is_genesis)
                .assert_zero(local.endpoint_batch_start[limb]);
            builder
                .when(local.endpoint_is_genesis)
                .assert_zero(local.endpoint_observation_count_start[limb]);
            builder.when(local.endpoint_is_terminal).assert_eq(
                local.endpoint_batch_end[limb],
                local.endpoint_batch_count[limb],
            );
        }
        for limb in 0..4 {
            builder.when_first_row().assert_eq(
                local.endpoint_active_total_start[limb],
                local.active_total_before[limb],
            );
            builder
                .when(local.endpoint_is_genesis)
                .assert_zero(local.endpoint_active_total_start[limb]);
        }
        for limb in 0..2 {
            builder
                .when(local.endpoint_is_genesis)
                .assert_zero(local.endpoint_chunk_index[limb]);
        }
        for limb in 0..DIGEST_SIZE {
            builder.when_first_row().assert_eq(
                local.endpoint_protocol_digest[limb],
                local.protocol_digest[limb],
            );
            builder.when_first_row().assert_eq(
                local.endpoint_relation_digest[limb],
                local.relation_digest[limb],
            );
            builder.when_first_row().assert_eq(
                local.endpoint_program_commitment[limb],
                local.program_commitment[limb],
            );
            builder.when_first_row().assert_eq(
                local.endpoint_initial_memory_root[limb],
                local.children[0].input.memory_root[limb],
            );
            builder.when_first_row().assert_eq(
                local.endpoint_genesis_accumulator_digest[limb],
                local.prior_accumulator_digest[limb],
            );
            builder.when_first_row().assert_eq(
                local.endpoint_history_hash_start[limb],
                local.history_hash_before[limb],
            );
        }
        let carry = next.active;
        builder
            .when_transition()
            .when(carry)
            .assert_eq(next.endpoint_initial_pc, local.endpoint_initial_pc);
        for (next_digest, local_digest) in [
            (
                &next.endpoint_protocol_digest,
                &local.endpoint_protocol_digest,
            ),
            (
                &next.endpoint_relation_digest,
                &local.endpoint_relation_digest,
            ),
            (
                &next.endpoint_program_commitment,
                &local.endpoint_program_commitment,
            ),
            (
                &next.endpoint_initial_memory_root,
                &local.endpoint_initial_memory_root,
            ),
            (
                &next.endpoint_genesis_accumulator_digest,
                &local.endpoint_genesis_accumulator_digest,
            ),
        ] {
            for limb in 0..DIGEST_SIZE {
                builder
                    .when_transition()
                    .when(carry)
                    .assert_eq(next_digest[limb], local_digest[limb]);
            }
        }
        for limb in 0..2 {
            for (next_value, local_value) in [
                (&next.endpoint_batch_count, &local.endpoint_batch_count),
                (&next.endpoint_chunk_index, &local.endpoint_chunk_index),
                (&next.endpoint_batch_start, &local.endpoint_batch_start),
                (&next.endpoint_batch_end, &local.endpoint_batch_end),
                (
                    &next.endpoint_observation_count_start,
                    &local.endpoint_observation_count_start,
                ),
            ] {
                builder
                    .when_transition()
                    .when(carry)
                    .assert_eq(next_value[limb], local_value[limb]);
            }
        }
        for limb in 0..4 {
            builder.when_transition().when(carry).assert_eq(
                next.endpoint_active_total_start[limb],
                local.endpoint_active_total_start[limb],
            );
        }
        for limb in 0..DIGEST_SIZE {
            builder.when_transition().when(carry).assert_eq(
                next.endpoint_history_hash_start[limb],
                local.endpoint_history_hash_start[limb],
            );
        }
        builder
            .when_transition()
            .when(carry)
            .assert_eq(next.endpoint_is_genesis, local.endpoint_is_genesis);
        builder
            .when_transition()
            .when(carry)
            .assert_eq(next.endpoint_is_terminal, local.endpoint_is_terminal);
    }

    fn eval_activity<AB: AirBuilder<F = F>>(
        &self,
        builder: &mut AB,
        local: &VerifierWarpHistoryRowColsV2<AB::Var>,
        next: &VerifierWarpHistoryRowColsV2<AB::Var>,
    ) where
        AB::Var: Copy,
    {
        for flag in [local.active, local.is_first, local.is_last] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when_transition().assert_zero(next.is_first);
        builder
            .when_transition()
            .assert_zero(next.active * (AB::Expr::ONE - local.active));
        builder
            .when_transition()
            .assert_eq(local.is_last, local.active * (AB::Expr::ONE - next.active));
        builder
            .when_last_row()
            .assert_eq(local.is_last, local.active);
    }

    fn eval_protocol<AB: AirBuilder<F = F>>(
        &self,
        builder: &mut AB,
        local: &VerifierWarpHistoryRowColsV2<AB::Var>,
        next: &VerifierWarpHistoryRowColsV2<AB::Var>,
    ) where
        AB::Var: Copy,
    {
        let enabled = local.active;
        builder.when(enabled).assert_eq(
            local.protocol_version,
            AB::Expr::from_u32(VERIFIER_WARP_HISTORY_PROTOCOL_V2),
        );
        builder.when(enabled).assert_eq(
            local.vacc_input_arity,
            AB::Expr::from_usize(VERIFIER_WARP_VACC_INPUT_ARITY_V2),
        );
        for (digest, endpoint) in [
            (&local.protocol_digest, &local.endpoint_protocol_digest),
            (&local.relation_digest, &local.endpoint_relation_digest),
            (
                &local.program_commitment,
                &local.endpoint_program_commitment,
            ),
        ] {
            for limb in 0..DIGEST_SIZE {
                builder
                    .when(enabled)
                    .assert_eq(digest[limb], endpoint[limb]);
            }
        }
        let link = next.active;
        builder
            .when_transition()
            .when(link)
            .assert_eq(next.protocol_version, local.protocol_version);
        builder
            .when_transition()
            .when(link)
            .assert_eq(next.vacc_input_arity, local.vacc_input_arity);
        for (next_digest, local_digest) in [
            (&next.protocol_digest, &local.protocol_digest),
            (&next.relation_digest, &local.relation_digest),
            (&next.program_commitment, &local.program_commitment),
        ] {
            for limb in 0..DIGEST_SIZE {
                builder
                    .when_transition()
                    .when(link)
                    .assert_eq(next_digest[limb], local_digest[limb]);
            }
        }
    }

    fn eval_children<AB: AirBuilder<F = F>>(
        &self,
        builder: &mut AB,
        local: &VerifierWarpHistoryRowColsV2<AB::Var>,
    ) where
        AB::Var: Copy,
    {
        let enabled = local.active;
        let mut flag_sum = AB::Expr::ZERO;
        let mut count = AB::Expr::ZERO;
        for (index, flag) in local.active_count_flags.iter().enumerate() {
            builder.when(enabled).assert_bool(*flag);
            flag_sum += *flag;
            count += *flag * AB::Expr::from_usize(index + 1);
        }
        builder.when(enabled).assert_one(flag_sum);
        builder
            .when(enabled)
            .assert_eq(local.active_child_count, count);

        // The unchanged capacity-four recursive source exposes one aggregate
        // VmPvs statement. Child ordering and within-batch continuity are
        // constraints of that fixed relation, so History stores the aggregate
        // in slot zero and requires every other slot to be canonical padding.
        for (slot, child) in local.children.iter().enumerate() {
            let expected_occupied = if slot == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            };
            builder.when(enabled).assert_bool(child.occupied);
            builder
                .when(enabled)
                .assert_eq(child.occupied, expected_occupied);
            builder.when(enabled).assert_bool(child.terminates);
            let inactive = enabled * (AB::Expr::ONE - child.occupied);
            builder.when(inactive.clone()).assert_zero(child.input.pc);
            builder.when(inactive.clone()).assert_zero(child.output.pc);
            builder.when(inactive.clone()).assert_zero(child.exit_code);
            builder.when(inactive.clone()).assert_zero(child.terminates);
            for digest in [&child.input.memory_root, &child.output.memory_root] {
                for &limb in digest {
                    builder.when(inactive.clone()).assert_zero(limb);
                }
            }

            let occupied = enabled * child.occupied;
            builder
                .when(occupied.clone() * (AB::Expr::ONE - child.terminates))
                .assert_eq(
                    child.exit_code,
                    AB::Expr::from_u32(DEFAULT_SUSPEND_EXIT_CODE),
                );
            builder
                .when(occupied * child.terminates)
                .assert_zero(child.exit_code);
        }

        let last_terminates = local.children[0].terminates;
        builder
            .when(enabled * (AB::Expr::ONE - local.is_last))
            .assert_zero(last_terminates.clone());
        builder
            .when(local.is_last)
            .assert_eq(last_terminates, local.endpoint_is_terminal);
        let requires_full_batch = enabled * (AB::Expr::ONE - local.is_last)
            + local.is_last * (AB::Expr::ONE - local.endpoint_is_terminal);
        builder
            .when(requires_full_batch)
            .assert_one(local.active_count_flags[VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2 - 1]);
        let terminal_row = local.is_last * local.endpoint_is_terminal;
        for limb in 0..DIGEST_SIZE {
            builder
                .when(enabled - terminal_row.clone())
                .assert_zero(local.public_values_digest[limb]);
        }
    }

    fn eval_continuity<AB: AirBuilder<F = F>>(
        &self,
        builder: &mut AB,
        local: &VerifierWarpHistoryRowColsV2<AB::Var>,
        next: &VerifierWarpHistoryRowColsV2<AB::Var>,
    ) where
        AB::Var: Copy,
    {
        let first_child = &local.children[0];
        builder
            .when_first_row()
            .assert_eq(first_child.input.pc, local.endpoint_initial_pc);
        for limb in 0..DIGEST_SIZE {
            builder.when_first_row().assert_eq(
                first_child.input.memory_root[limb],
                local.endpoint_initial_memory_root[limb],
            );
            builder.when_first_row().assert_eq(
                local.prior_accumulator_digest[limb],
                local.endpoint_genesis_accumulator_digest[limb],
            );
        }
        let link = next.active;
        builder
            .when_transition()
            .when(link)
            .assert_eq(local.children[0].output.pc, next.children[0].input.pc);
        for limb in 0..DIGEST_SIZE {
            builder.when_transition().when(link).assert_eq(
                local.children[0].output.memory_root[limb],
                next.children[0].input.memory_root[limb],
            );
        }
        for limb in 0..DIGEST_SIZE {
            builder.when_transition().when(link).assert_eq(
                local.output_accumulator_digest[limb],
                next.prior_accumulator_digest[limb],
            );
        }
    }

    fn eval_counters<AB: AirBuilder<F = F>>(
        &self,
        builder: &mut AB,
        local: &VerifierWarpHistoryRowColsV2<AB::Var>,
        next: &VerifierWarpHistoryRowColsV2<AB::Var>,
    ) where
        AB::Var: Copy,
    {
        for (value, bits) in local.batch_index.iter().zip(&local.batch_index_bits) {
            self.assert_limb(builder, local.active, *value, bits);
        }
        builder
            .when(local.active)
            .assert_bool(local.batch_index_carry);
        builder
            .when_first_row()
            .assert_eq(local.batch_index[0], local.endpoint_batch_start[0]);
        builder
            .when_first_row()
            .assert_eq(local.batch_index[1], local.endpoint_batch_start[1]);
        builder.when_transition().when(next.active).assert_eq(
            next.batch_index[0],
            local.batch_index[0] + AB::Expr::ONE
                - local.batch_index_carry * AB::Expr::from_u32(LIMB_BASE),
        );
        builder.when_transition().when(next.active).assert_eq(
            next.batch_index[1],
            local.batch_index[1] + local.batch_index_carry,
        );
        builder.when(local.is_last).assert_eq(
            local.batch_index[0] + AB::Expr::ONE
                - local.batch_index_carry * AB::Expr::from_u32(LIMB_BASE),
            local.endpoint_batch_end[0],
        );
        builder.when(local.is_last).assert_eq(
            local.batch_index[1] + local.batch_index_carry,
            local.endpoint_batch_end[1],
        );

        for limb in 0..4 {
            self.assert_limb(
                builder,
                local.active,
                local.active_total_before[limb],
                &local.active_total_before_bits[limb],
            );
            self.assert_limb(
                builder,
                local.active,
                local.active_total_after[limb],
                &local.active_total_after_bits[limb],
            );
            builder
                .when(local.active)
                .assert_bool(local.active_total_carries[limb]);
            let addend = if limb == 0 {
                AB::Expr::from(local.active_child_count)
            } else {
                AB::Expr::from(local.active_total_carries[limb - 1])
            };
            builder.when(local.active).assert_eq(
                local.active_total_after[limb],
                local.active_total_before[limb] + addend
                    - local.active_total_carries[limb] * AB::Expr::from_u32(LIMB_BASE),
            );
            builder.when_first_row().assert_eq(
                local.active_total_before[limb],
                local.endpoint_active_total_start[limb],
            );
            builder.when_transition().when(next.active).assert_eq(
                next.active_total_before[limb],
                local.active_total_after[limb],
            );
        }

        for limb in 0..2 {
            self.assert_limb(
                builder,
                local.active,
                local.observation_count_before[limb],
                &local.observation_count_before_bits[limb],
            );
            self.assert_limb(
                builder,
                local.active,
                local.observation_count_after[limb],
                &local.observation_count_after_bits[limb],
            );
            builder
                .when(local.active)
                .assert_bool(local.observation_count_carries[limb]);
            let first_addend = AB::Expr::from_usize(super::MANIFEST_OBSERVATION_COUNT_V2)
                + local.is_first
                    * local.endpoint_is_genesis
                    * AB::Expr::from_usize(super::HISTORY_HEADER_OBSERVATION_COUNT_V2)
                + local.is_last * local.endpoint_is_terminal;
            let addend = if limb == 0 {
                first_addend
            } else {
                AB::Expr::from(local.observation_count_carries[limb - 1])
            };
            builder.when(local.active).assert_eq(
                local.observation_count_after[limb],
                local.observation_count_before[limb] + addend
                    - local.observation_count_carries[limb] * AB::Expr::from_u32(LIMB_BASE),
            );
            builder.when_first_row().assert_eq(
                local.observation_count_before[limb],
                local.endpoint_observation_count_start[limb],
            );
            builder.when_transition().when(next.active).assert_eq(
                next.observation_count_before[limb],
                local.observation_count_after[limb],
            );
        }
    }

    fn eval_certificates<AB: AirBuilder<F = F> + InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &VerifierWarpHistoryRowColsV2<AB::Var>,
    ) where
        AB::Var: Copy,
    {
        let children = core::array::from_fn(|slot| {
            let child = &local.children[slot];
            VerifierWarpCertifiedChildMessageV2 {
                occupied: child.occupied.into(),
                input_pc: child.input.pc.into(),
                input_memory_root: child.input.memory_root.map(Into::into),
                output_pc: child.output.pc.into(),
                output_memory_root: child.output.memory_root.map(Into::into),
                exit_code: child.exit_code.into(),
                terminates: child.terminates.into(),
            }
        });
        self.source_bus.lookup_key(
            builder,
            VerifierWarpSourceCertificateMessageV2 {
                protocol_version: local.protocol_version.into(),
                source_child_capacity: AB::Expr::from_usize(VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2),
                batch_index_lo: local.batch_index[0].into(),
                batch_index_hi: local.batch_index[1].into(),
                active_child_count: local.active_child_count.into(),
                protocol_digest: local.protocol_digest.map(Into::into),
                relation_digest: local.relation_digest.map(Into::into),
                program_commitment: local.program_commitment.map(Into::into),
                children,
                source_accumulator_digest: local.source_accumulator_digest.map(Into::into),
                source_commitment_root: local.source_commitment_root.map(Into::into),
                external_logup_gkr_digest: local.external_logup_gkr_digest.map(Into::into),
                source_functional_digest: local.source_functional_digest.map(Into::into),
                setup_openings_digest: local.setup_openings_digest.map(Into::into),
                source_statement_digest: local.source_statement_digest.map(Into::into),
            },
            local.active,
        );
        self.vacc_bus.lookup_key(
            builder,
            VerifierWarpVaccCertificateMessageV2 {
                protocol_version: local.protocol_version.into(),
                input_arity: local.vacc_input_arity.into(),
                batch_index_lo: local.batch_index[0].into(),
                batch_index_hi: local.batch_index[1].into(),
                relation_digest: local.relation_digest.map(Into::into),
                prior_accumulator_digest: local.prior_accumulator_digest.map(Into::into),
                source_accumulator_digest: local.source_accumulator_digest.map(Into::into),
                output_accumulator_digest: local.output_accumulator_digest.map(Into::into),
                source_commitment_root: local.source_commitment_root.map(Into::into),
                transition_transcript_digest: local.transition_transcript_digest.map(Into::into),
            },
            local.active,
        );
        let final_memory_root = local.children[0].output.memory_root.map(Into::into);
        // A direct-final History proof owns the terminal block-PV check.  A
        // bounded chunk deliberately exports the digest as a public endpoint
        // instead: the recursive root verifies the block-PV path exactly once
        // and binds it to the unique terminal child.  Requiring this lookup in
        // every chunk would either leave nonterminal buses unbalanced or
        // duplicate the expensive terminal Merkle proof.
        if !matches!(
            self.output_mode,
            VerifierWarpHistoryOutputModeV2::ChunkPublicValuesV3
                | VerifierWarpHistoryOutputModeV2::ChunkTypedProviderV3 { .. }
        ) {
            self.block_public_values_bus.lookup_key(
                builder,
                VerifierWarpBlockPublicValuesMessageV2 {
                    final_memory_root,
                    public_values_digest: local.public_values_digest.map(Into::into),
                },
                local.is_last * local.endpoint_is_terminal,
            );
        }
    }

    #[allow(clippy::too_many_lines)]
    fn eval_history_hash<AB: AirBuilder<F = F> + InteractionBuilder>(
        &self,
        builder: &mut AB,
        local: &VerifierWarpHistoryRowColsV2<AB::Var>,
        next: &VerifierWarpHistoryRowColsV2<AB::Var>,
    ) where
        AB::Var: Copy,
    {
        let initial: [AB::Expr; DIGEST_SIZE] = core::array::from_fn(|limb| match limb {
            0 => AB::Expr::from_u32(STATEMENT_DIGEST_START_TAG_V2),
            1 => AB::Expr::from_u32(VERIFIER_WARP_HISTORY_PROTOCOL_V2),
            _ => AB::Expr::ZERO,
        });
        for limb in 0..DIGEST_SIZE {
            builder.when_first_row().assert_eq(
                local.history_hash_before[limb],
                local.endpoint_history_hash_start[limb],
            );
            builder
                .when(AB::Expr::from(local.is_first) * AB::Expr::from(local.endpoint_is_genesis))
                .assert_eq(local.history_hash_before[limb], initial[limb].clone());
            builder.when_transition().when(next.active).assert_eq(
                next.history_hash_before[limb],
                local.history_hash_after[limb],
            );
        }
        let mut state = local.history_hash_before.map(Into::into);
        let mut step = 0usize;
        let header_enabled =
            AB::Expr::from(local.is_first) * AB::Expr::from(local.endpoint_is_genesis);
        for value in [
            AB::Expr::from_u32(HISTORY_START_TAG_V2),
            AB::Expr::from_u32(VERIFIER_WARP_HISTORY_PROTOCOL_V2),
            AB::Expr::from_usize(VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2),
            AB::Expr::from_usize(VERIFIER_WARP_VACC_INPUT_ARITY_V2),
            local.endpoint_batch_count[0].into(),
            local.endpoint_batch_count[1].into(),
        ] {
            state = self.absorb_field(
                builder,
                state,
                value,
                local.hash_outputs[step],
                header_enabled.clone(),
                local.active.into(),
            );
            step += 1;
        }

        let enabled = AB::Expr::from(local.active);
        macro_rules! field {
            ($value:expr) => {{
                state = self.absorb_field(
                    builder,
                    state,
                    $value,
                    local.hash_outputs[step],
                    enabled.clone(),
                    local.active.into(),
                );
                step += 1;
            }};
        }
        macro_rules! digest {
            ($value:expr) => {{
                state = self.absorb_digest(
                    builder,
                    state,
                    $value,
                    local.hash_outputs[step],
                    local.hash_outputs[step + 1],
                    enabled.clone(),
                    local.active.into(),
                );
                step += 2;
            }};
        }
        field!(AB::Expr::from_u32(MANIFEST_START_TAG_V2));
        field!(AB::Expr::from_u32(MANIFEST_HEADER_TAG_V2));
        field!(local.protocol_version.into());
        field!(AB::Expr::from_usize(VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2));
        field!(local.vacc_input_arity.into());
        digest!(local.protocol_digest.map(Into::into));
        digest!(local.relation_digest.map(Into::into));
        field!(local.batch_index[0].into());
        field!(local.batch_index[1].into());
        field!(local.active_child_count.into());
        digest!(local.program_commitment.map(Into::into));
        field!(AB::Expr::from_u32(MANIFEST_CHILDREN_TAG_V2));
        for (slot, child) in local.children.iter().enumerate() {
            field!(AB::Expr::from_u32(MANIFEST_CHILD_TAG_V2));
            field!(AB::Expr::from_usize(slot));
            field!(child.occupied.into());
            field!(child.input.pc.into());
            digest!(child.input.memory_root.map(Into::into));
            field!(child.output.pc.into());
            digest!(child.output.memory_root.map(Into::into));
            field!(child.exit_code.into());
            field!(child.terminates.into());
        }
        field!(AB::Expr::from_u32(MANIFEST_ACCUMULATORS_TAG_V2));
        digest!(local.prior_accumulator_digest.map(Into::into));
        digest!(local.source_accumulator_digest.map(Into::into));
        digest!(local.output_accumulator_digest.map(Into::into));
        field!(AB::Expr::from_u32(MANIFEST_SOURCE_AUTH_TAG_V2));
        digest!(local.source_commitment_root.map(Into::into));
        digest!(local.external_logup_gkr_digest.map(Into::into));
        digest!(local.source_functional_digest.map(Into::into));
        digest!(local.setup_openings_digest.map(Into::into));
        field!(AB::Expr::from_u32(MANIFEST_SOURCE_TAG_V2));
        digest!(local.source_statement_digest.map(Into::into));
        digest!(local.transition_transcript_digest.map(Into::into));
        field!(AB::Expr::from_u32(MANIFEST_PUBLIC_VALUES_TAG_V2));
        digest!(local.public_values_digest.map(Into::into));
        field!(AB::Expr::from_u32(MANIFEST_END_TAG_V2));

        let trailer_enabled =
            AB::Expr::from(local.is_last) * AB::Expr::from(local.endpoint_is_terminal);
        state = self.absorb_field(
            builder,
            state,
            AB::Expr::from_u32(HISTORY_END_TAG_V2),
            local.hash_outputs[step],
            trailer_enabled.clone(),
            local.active.into(),
        );
        step += 1;
        let mut end = core::array::from_fn(|_| AB::Expr::ZERO);
        end[0] = AB::Expr::from_u32(STATEMENT_DIGEST_END_TAG_V2);
        end[1] = local.observation_count_after[0].into();
        end[2] = local.observation_count_after[1].into();
        state = self.absorb_block(
            builder,
            state,
            end,
            local.hash_outputs[step],
            trailer_enabled,
            local.active.into(),
        );
        step += 1;
        debug_assert_eq!(step, HISTORY_ROW_HASH_STEP_COUNT_V2);
        for limb in 0..DIGEST_SIZE {
            builder
                .when(local.active)
                .assert_eq(local.history_hash_after[limb], state[limb].clone());
        }
    }

    fn endpoint_message<AB: AirBuilder<F = F>>(
        &self,
        local: &VerifierWarpHistoryRowColsV2<AB::Var>,
    ) -> VerifierWarpHistoryEndpointMessageV2<AB::Expr>
    where
        AB::Var: Copy,
    {
        VerifierWarpHistoryEndpointMessageV2 {
            protocol_version: AB::Expr::from_u32(VERIFIER_WARP_HISTORY_PROTOCOL_V2),
            source_child_capacity: AB::Expr::from_usize(VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2),
            vacc_input_arity: AB::Expr::from_usize(VERIFIER_WARP_VACC_INPUT_ARITY_V2),
            protocol_digest: local.endpoint_protocol_digest.map(Into::into),
            relation_digest: local.endpoint_relation_digest.map(Into::into),
            program_commitment: local.endpoint_program_commitment.map(Into::into),
            batch_count: local.endpoint_batch_count.map(Into::into),
            active_child_count: local.active_total_after.map(Into::into),
            initial_pc: local.endpoint_initial_pc.into(),
            initial_memory_root: local.endpoint_initial_memory_root.map(Into::into),
            final_pc: local.children[0].output.pc.into(),
            final_memory_root: local.children[0].output.memory_root.map(Into::into),
            genesis_accumulator_digest: local.endpoint_genesis_accumulator_digest.map(Into::into),
            final_accumulator_digest: local.output_accumulator_digest.map(Into::into),
            public_values_digest: local.public_values_digest.map(Into::into),
            history_statement_digest: local.history_hash_after.map(Into::into),
        }
    }

    fn chunk_endpoint_message<AB: AirBuilder<F = F>>(
        &self,
        local: &VerifierWarpHistoryRowColsV2<AB::Var>,
    ) -> VerifierWarpHistoryChunkEndpointMessageV3<AB::Expr>
    where
        AB::Var: Copy,
    {
        VerifierWarpHistoryChunkEndpointMessageV3 {
            chunk_protocol_version: AB::Expr::from_u32(VERIFIER_WARP_HISTORY_CHUNK_PROTOCOL_V3),
            source_history_protocol_version: AB::Expr::from_u32(VERIFIER_WARP_HISTORY_PROTOCOL_V2),
            source_child_capacity: AB::Expr::from_usize(VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2),
            vacc_input_arity: AB::Expr::from_usize(VERIFIER_WARP_VACC_INPUT_ARITY_V2),
            protocol_digest: local.endpoint_protocol_digest.map(Into::into),
            relation_digest: local.endpoint_relation_digest.map(Into::into),
            program_commitment: local.endpoint_program_commitment.map(Into::into),
            total_batch_count: local.endpoint_batch_count.map(Into::into),
            chunk_index: local.endpoint_chunk_index.map(Into::into),
            start_batch_index: local.endpoint_batch_start.map(Into::into),
            end_batch_index: local.endpoint_batch_end.map(Into::into),
            active_child_count_before: local.endpoint_active_total_start.map(Into::into),
            active_child_count_after: local.active_total_after.map(Into::into),
            initial_pc: local.endpoint_initial_pc.into(),
            initial_memory_root: local.endpoint_initial_memory_root.map(Into::into),
            final_pc: local.children[0].output.pc.into(),
            final_memory_root: local.children[0].output.memory_root.map(Into::into),
            initial_accumulator_digest: local.endpoint_genesis_accumulator_digest.map(Into::into),
            final_accumulator_digest: local.output_accumulator_digest.map(Into::into),
            history_hash_before: local.endpoint_history_hash_start.map(Into::into),
            history_hash_after: local.history_hash_after.map(Into::into),
            observation_count_before: local.endpoint_observation_count_start.map(Into::into),
            observation_count_after: local.observation_count_after.map(Into::into),
            is_genesis: local.endpoint_is_genesis.into(),
            is_terminal: local.endpoint_is_terminal.into(),
            public_values_digest: local.public_values_digest.map(Into::into),
        }
    }

    fn eval_output<AB>(
        &self,
        builder: &mut AB,
        local: &VerifierWarpHistoryRowColsV2<AB::Var>,
        public: &[AB::PublicVar],
    ) where
        AB: AirBuilder<F = F> + AirBuilderWithPublicValues + InteractionBuilder,
        AB::Var: Copy,
        AB::PublicVar: Copy,
        AB::Expr: From<AB::PublicVar>,
    {
        let endpoint = self.endpoint_message::<AB>(local);
        match self.output_mode {
            VerifierWarpHistoryOutputModeV2::StandalonePublicValues => {
                builder
                    .when_first_row()
                    .assert_one(local.endpoint_is_genesis);
                builder
                    .when_first_row()
                    .assert_one(local.endpoint_is_terminal);
                assert_eq!(
                    public.len(),
                    VerifierWarpHistoryPublicValuesV2::WIDTH,
                    "standalone History-v2 PV width"
                );
                let values = endpoint.flatten();
                debug_assert_eq!(values.len(), VerifierWarpHistoryPublicValuesV2::WIDTH);
                for (value, public_value) in values.into_iter().zip(public.iter().copied()) {
                    builder
                        .when(local.is_last)
                        .assert_eq(value, AB::Expr::from(public_value));
                }
            }
            VerifierWarpHistoryOutputModeV2::ChunkPublicValuesV3 => {
                assert_eq!(
                    public.len(),
                    VerifierWarpHistoryChunkPublicValuesV3::WIDTH,
                    "chunk History-v3 PV width"
                );
                let values = self.chunk_endpoint_message::<AB>(local).flatten();
                debug_assert_eq!(values.len(), VerifierWarpHistoryChunkPublicValuesV3::WIDTH);
                for (value, public_value) in values.into_iter().zip(public.iter().copied()) {
                    builder
                        .when(local.is_last)
                        .assert_eq(value, AB::Expr::from(public_value));
                }
            }
            VerifierWarpHistoryOutputModeV2::ChunkTypedProviderV3 {
                interval_bus,
                lookup_count,
            } => {
                assert!(lookup_count > 0, "zero chunk endpoint multiplicity");
                assert!(
                    public.is_empty(),
                    "embedded chunk History must have zero PVs"
                );
                let fields: [_; VERIFIER_WARP_HISTORY_CHUNK_MESSAGE_WIDTH_V3] = self
                    .chunk_endpoint_message::<AB>(local)
                    .flatten()
                    .try_into()
                    .expect("History-v3 endpoint width is a protocol constant");
                interval_bus.add_key_with_lookups(
                    builder,
                    VerifierWarpHistoryChunkIntervalMessageV3 { fields },
                    Into::<AB::Expr>::into(local.is_last) * AB::Expr::from_u32(lookup_count),
                );
            }
            VerifierWarpHistoryOutputModeV2::DirectFinalProvider {
                endpoint_bus,
                lookup_count,
            } => {
                builder
                    .when_first_row()
                    .assert_one(local.endpoint_is_genesis);
                builder
                    .when_first_row()
                    .assert_one(local.endpoint_is_terminal);
                assert!(lookup_count > 0, "zero History endpoint multiplicity");
                assert!(public.is_empty(), "direct History-v2 must have zero PVs");
                endpoint_bus.add_key_with_lookups(
                    builder,
                    endpoint,
                    Into::<AB::Expr>::into(local.is_last) * AB::Expr::from_u32(lookup_count),
                );
            }
        }
    }

    fn absorb_field<AB: AirBuilder<F = F> + InteractionBuilder>(
        &self,
        builder: &mut AB,
        state: [AB::Expr; DIGEST_SIZE],
        value: AB::Expr,
        output: [AB::Var; DIGEST_SIZE],
        enabled: AB::Expr,
        row_active: AB::Expr,
    ) -> [AB::Expr; DIGEST_SIZE]
    where
        AB::Var: Copy,
    {
        let mut block = core::array::from_fn(|_| AB::Expr::ZERO);
        block[0] = AB::Expr::from_u32(STATEMENT_DIGEST_FIELD_TAG_V2);
        block[1] = value;
        self.absorb_block(builder, state, block, output, enabled, row_active)
    }

    fn absorb_digest<AB: AirBuilder<F = F> + InteractionBuilder>(
        &self,
        builder: &mut AB,
        state: [AB::Expr; DIGEST_SIZE],
        digest: [AB::Expr; DIGEST_SIZE],
        tag_output: [AB::Var; DIGEST_SIZE],
        digest_output: [AB::Var; DIGEST_SIZE],
        enabled: AB::Expr,
        row_active: AB::Expr,
    ) -> [AB::Expr; DIGEST_SIZE]
    where
        AB::Var: Copy,
    {
        let mut tag = core::array::from_fn(|_| AB::Expr::ZERO);
        tag[0] = AB::Expr::from_u32(STATEMENT_DIGEST_COMMITMENT_TAG_V2);
        let state = self.absorb_block(
            builder,
            state,
            tag,
            tag_output,
            enabled.clone(),
            row_active.clone(),
        );
        self.absorb_block(builder, state, digest, digest_output, enabled, row_active)
    }

    fn absorb_block<AB: AirBuilder<F = F> + InteractionBuilder>(
        &self,
        builder: &mut AB,
        state: [AB::Expr; DIGEST_SIZE],
        block: [AB::Expr; DIGEST_SIZE],
        output: [AB::Var; DIGEST_SIZE],
        enabled: AB::Expr,
        row_active: AB::Expr,
    ) -> [AB::Expr; DIGEST_SIZE]
    where
        AB::Var: Copy,
    {
        self.compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: core::array::from_fn(|index| {
                    if index < DIGEST_SIZE {
                        state[index].clone()
                    } else {
                        block[index - DIGEST_SIZE].clone()
                    }
                }),
                output: output.map(Into::into),
            },
            enabled.clone(),
        );
        // Every hash step has its own trace column.  On an active row where a
        // conditional step is disabled, copy the prior state into that
        // column.  Returning the column directly prevents the conditional
        // expression from being nested through all 94 history observations
        // (which previously made the symbolic AIR degree grow to 94).
        let copy = row_active - enabled;
        for limb in 0..DIGEST_SIZE {
            builder
                .when(copy.clone())
                .assert_eq(output[limb], state[limb].clone());
        }
        output.map(Into::into)
    }

    fn assert_limb<AB: AirBuilder<F = F>>(
        &self,
        builder: &mut AB,
        enabled: AB::Var,
        value: AB::Var,
        bits: &[AB::Var; LIMB_BITS],
    ) where
        AB::Var: Copy,
    {
        let recomposed = bits
            .iter()
            .enumerate()
            .fold(AB::Expr::ZERO, |sum, (bit, value)| {
                builder.when(enabled).assert_bool(*value);
                sum + *value * AB::Expr::from_u32(1 << bit)
            });
        builder.when(enabled).assert_eq(value, recomposed);
    }
}

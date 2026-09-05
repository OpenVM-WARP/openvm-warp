//! Complete transition-tree verifier and terminal hand-off for reduced-SWIRL WARP.
//!
//! This circuit verifies exactly one ordinary OpenVM recursive tree root whose
//! leaves are [`ReducedSwirlTransitionLeaf`](super::reduced_swirl_transition_leaf)
//! proofs.  It opens only the tree's initial and final boundary digests.  It
//! does not replay transitions and it does not treat a PCS opening as PESAT.
//!
//! The rolling per-call manifest authenticated by the tree and the canonical
//! flat terminal manifest are intentionally distinct commitments.  Their sole
//! authority is `ReducedSwirlManifestReconciliationAir`; this finalizer
//! consumes its typed receipt and exports the flat digest to the existing VACC
//! footer.  All host checks in the trace generator are duplicated by AIR
//! constraints and are therefore only early error reporting.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_recursion_circuit::{
    bus::{
        CachedCommitBus, CachedCommitBusMessage, Poseidon2CompressBus, Poseidon2CompressMessage,
        PreHashBus, PreHashMessage, PublicValuesBus, PublicValuesBusMessage,
        ResumeTranscriptStateBus, ResumeTranscriptStateMessage,
    },
    native_warp::{
        ReducedSwirlLocalTerminalReceiptBus, ReducedSwirlLocalTerminalReceiptMessage,
        ReducedSwirlManifestDigestBus, ReducedSwirlManifestDigestMessage,
        ReducedSwirlManifestReconciliationReceiptBus,
        ReducedSwirlManifestReconciliationReceiptMessage, ReducedSwirlVaccChainEndBus,
        ReducedSwirlVaccChainEndMessage, ReducedSwirlVaccChainReceiptBus,
        ReducedSwirlVaccChainReceiptMessage,
    },
    system::{
        AggregationSubCircuit, BusIndexManager, VerifierConfig, VerifierSubCircuit,
        VerifierTailMode,
    },
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder},
    keygen::types::MultiStarkVerifyingKey,
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::{Field, PrimeCharacteristicRing},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    AirRef, BaseAirWithPublicValues, PartitionedBaseAir, StarkProtocolConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, BabyBearPoseidon2Config, Digest, DIGEST_SIZE, F,
};
use openvm_verify_stark_host::pvs::{
    VerifierBasePvs, VkCommit, VmPvs, CONSTRAINT_EVAL_AIR_ID, CONSTRAINT_EVAL_CACHED_INDEX,
    VERIFIER_PVS_AIR_ID, VM_PVS_AIR_ID,
};

use super::{
    reduced_swirl_transition_leaf::{
        reduced_swirl_transition_chain_genesis, reduced_swirl_transition_chain_genesis_metadata,
        reduced_swirl_transition_state_digest, state_digest, ReducedSwirlTransitionState,
        StateHashWitness, REDUCED_SWIRL_TRANSITION_LEAF_PROTOCOL_VERSION,
        REDUCED_SWIRL_TRANSITION_STATE_METADATA_TAG,
    },
    reduced_swirl_warp::{ReducedSwirlExecutionBus, ReducedSwirlExecutionMessage},
    Circuit,
};

const TRANSITION_TREE_PROOF_INDEX: usize = 0;
const TRANSITION_TERMINAL_PROOF_INDEX: usize = 0;

/// Setup authority propagated by the ordinary OpenVM transition-proof tree.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReducedSwirlTransitionTreeTrustedVkCommits {
    pub app_vk_commit: VkCommit<F>,
    pub transition_leaf_vk_commit: VkCommit<F>,
    pub internal_for_leaf_vk_commit: VkCommit<F>,
    pub recursive_vk_commit: VkCommit<F>,
}

/// Setup-fixed identity of the transition tree and the WARP family it binds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReducedSwirlTransitionFinalizerBinding {
    pub source_protocol_digest: Digest,
    pub warp_protocol_digest: Digest,
    pub relation_digest: Digest,
    pub warp_index_digest: Digest,
    pub schedule_digest: Digest,
    pub trusted_vk_commits: ReducedSwirlTransitionTreeTrustedVkCommits,
}

impl ReducedSwirlTransitionFinalizerBinding {
    pub fn validate(&self) -> Result<(), ReducedSwirlTransitionFinalizerError> {
        if [
            self.source_protocol_digest,
            self.warp_protocol_digest,
            self.relation_digest,
            self.warp_index_digest,
            self.schedule_digest,
        ]
        .iter()
        .any(is_zero_digest)
            || [
                self.trusted_vk_commits.app_vk_commit,
                self.trusted_vk_commits.transition_leaf_vk_commit,
                self.trusted_vk_commits.internal_for_leaf_vk_commit,
                self.trusted_vk_commits.recursive_vk_commit,
            ]
            .into_iter()
            .any(is_unset_vk_commit)
        {
            return Err(ReducedSwirlTransitionFinalizerError::UnsetBinding);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReducedSwirlTransitionFinalizerError {
    UnsetBinding,
    ChildVkPreHash,
    ChildPublicValueProfile,
    ChildCachedProfile,
    ReusedBusIndex,
    SymbolicDagAuthority,
    PublicValueShape,
    OrdinaryPvsMismatch,
    EmptyExecution,
    SetupMismatch,
    GenesisMismatch,
    CoverageMismatch,
    TreeBoundaryMismatch,
    ReconciliationMismatch,
    ProgramMismatch,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlTransitionFinalizerRecord {
    pub initial_state: ReducedSwirlTransitionState,
    pub final_state: ReducedSwirlTransitionState,
    pub reconciliation: ReducedSwirlManifestReconciliationReceiptMessage<F>,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ReducedSwirlTransitionStateCols<T> {
    pub source_protocol_digest: [T; DIGEST_SIZE],
    pub warp_protocol_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub warp_index_digest: [T; DIGEST_SIZE],
    pub schedule_digest: [T; DIGEST_SIZE],
    pub total_source_count: T,
    pub call_cursor: T,
    pub source_cursor: T,
    pub transcript_tidx: T,
    pub transcript_sample_count: T,
    pub transcript_state: [T; 16],
    pub accumulator_root: [T; DIGEST_SIZE],
    pub accumulator_digest: [T; DIGEST_SIZE],
    pub vm_pc: T,
    pub vm_root: [T; DIGEST_SIZE],
    pub manifest_chain: [T; DIGEST_SIZE],
    pub program_commitment: [T; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ReducedSwirlTransitionStateHashCols<T> {
    pub setup_left: [T; DIGEST_SIZE],
    pub setup_right: [T; DIGEST_SIZE],
    pub setup: [T; DIGEST_SIZE],
    pub protocol: [T; DIGEST_SIZE],
    pub control: [T; DIGEST_SIZE],
    pub transcript: [T; DIGEST_SIZE],
    pub accumulator: [T; DIGEST_SIZE],
    pub transcript_accumulator: [T; DIGEST_SIZE],
    pub vm_chain: [T; DIGEST_SIZE],
    pub vm_program: [T; DIGEST_SIZE],
    pub left: [T; DIGEST_SIZE],
    pub boundary: [T; DIGEST_SIZE],
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ReducedSwirlTransitionFinalizerCols<T> {
    pub active: T,
    pub depth_inv: T,
    pub depth_minus_one_inv: T,
    pub is_depth_one: T,
    pub source_count_inv: T,
    pub call_count_inv: T,
    pub child_verifier_pvs: VerifierBasePvs<T>,
    pub child_vm_pvs: VmPvs<T>,
    pub reconciliation: ReducedSwirlManifestReconciliationReceiptMessage<T>,
    pub chain_genesis: [T; DIGEST_SIZE],
    pub initial: ReducedSwirlTransitionStateCols<T>,
    pub initial_hash: ReducedSwirlTransitionStateHashCols<T>,
    pub final_state: ReducedSwirlTransitionStateCols<T>,
    pub final_hash: ReducedSwirlTransitionStateHashCols<T>,
}

#[derive(Clone)]
pub struct ReducedSwirlTransitionFinalizerTrace {
    pub matrix: RowMajorMatrix<F>,
    pub compression_inputs: Vec<[F; 2 * DIGEST_SIZE]>,
    pub real_vm_pvs: VmPvs<F>,
    pub resume: ResumeTranscriptStateMessage<F>,
    pub chain_end: ReducedSwirlVaccChainEndMessage<F>,
    pub manifest: ReducedSwirlManifestDigestMessage<F>,
}

/// Fixed terminal identity used to join the transition-tree endpoint to the
/// existing terminal Decide component.
///
/// The terminal component and the VACC footer deliberately live on separate
/// typed buses.  This binding gives their sole consumer the constants needed
/// to prove that both receipts describe the same accumulated instance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReducedSwirlTransitionTerminalJoinBinding {
    pub protocol_digest: Digest,
    pub relation_digest: Digest,
    pub warp_index_digest: Digest,
    pub schedule_digest: Digest,
    pub terminal_index_digest: Digest,
    pub verifier_component_digest: Digest,
}

impl ReducedSwirlTransitionTerminalJoinBinding {
    pub fn validate(&self) -> Result<(), ReducedSwirlTransitionFinalizerError> {
        if [
            self.protocol_digest,
            self.relation_digest,
            self.warp_index_digest,
            self.schedule_digest,
            self.terminal_index_digest,
            self.verifier_component_digest,
        ]
        .iter()
        .any(is_zero_digest)
        {
            return Err(ReducedSwirlTransitionFinalizerError::UnsetBinding);
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ReducedSwirlTransitionTerminalJoinCols<T> {
    pub active: T,
    pub source_count: T,
    pub call_count: T,
    pub manifest_digest: [T; DIGEST_SIZE],
    pub final_accumulator_digest: [T; DIGEST_SIZE],
    pub final_accumulator_root: [T; DIGEST_SIZE],
}

/// Typed equality join between the final VACC footer and terminal Decide.
///
/// Without this AIR, the footer's chain receipt and the terminal component's
/// receipt would each be valid but unrelated lookup keys.  The two-row trace
/// consumes both under one witness row, forcing equal root and instance
/// digest while retaining the distinct protocol constants carried by each
/// producer.
#[derive(Clone, Copy, Debug)]
pub struct ReducedSwirlTransitionTerminalJoinAir {
    pub binding: ReducedSwirlTransitionTerminalJoinBinding,
    pub chain_receipt_bus: ReducedSwirlVaccChainReceiptBus,
    pub terminal_receipt_bus: ReducedSwirlLocalTerminalReceiptBus,
}

impl ReducedSwirlTransitionTerminalJoinAir {
    pub fn new(
        binding: ReducedSwirlTransitionTerminalJoinBinding,
        chain_receipt_bus: ReducedSwirlVaccChainReceiptBus,
        terminal_receipt_bus: ReducedSwirlLocalTerminalReceiptBus,
    ) -> Result<Self, ReducedSwirlTransitionFinalizerError> {
        binding.validate()?;
        Ok(Self {
            binding,
            chain_receipt_bus,
            terminal_receipt_bus,
        })
    }

    pub fn generate_trace(
        &self,
        record: &ReducedSwirlTransitionFinalizerRecord,
    ) -> Result<RowMajorMatrix<F>, ReducedSwirlTransitionFinalizerError> {
        self.binding.validate()?;
        if record.reconciliation.source_count == F::ZERO
            || record.reconciliation.call_count == F::ZERO
            || record
                .final_state
                .accumulator_digest
                .iter()
                .all(|value| *value == F::ZERO)
            || record
                .final_state
                .accumulator_root
                .iter()
                .all(|value| *value == F::ZERO)
        {
            return Err(ReducedSwirlTransitionFinalizerError::EmptyExecution);
        }
        let width = self.width();
        let mut values = F::zero_vec(2 * width);
        let local: &mut ReducedSwirlTransitionTerminalJoinCols<F> = values[..width].borrow_mut();
        local.active = F::ONE;
        local.source_count = record.reconciliation.source_count;
        local.call_count = record.reconciliation.call_count;
        local.manifest_digest = record.reconciliation.flat_manifest_digest;
        local.final_accumulator_digest = record.final_state.accumulator_digest;
        local.final_accumulator_root = record.final_state.accumulator_root;
        Ok(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for ReducedSwirlTransitionTerminalJoinAir {}
impl PartitionedBaseAir<F> for ReducedSwirlTransitionTerminalJoinAir {}
impl BaseAir<F> for ReducedSwirlTransitionTerminalJoinAir {
    fn width(&self) -> usize {
        ReducedSwirlTransitionTerminalJoinCols::<F>::width()
    }
}

impl<AB> Air<AB> for ReducedSwirlTransitionTerminalJoinAir
where
    AB: AirBuilder<F = F> + InteractionBuilder + PairBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("terminal join row");
        let next_row = main.row_slice(1).expect("terminal join padding row");
        let local: &ReducedSwirlTransitionTerminalJoinCols<AB::Var> = (*local_row).borrow();
        let next: &ReducedSwirlTransitionTerminalJoinCols<AB::Var> = (*next_row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_last_row().assert_zero(local.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.active);
        let active: AB::Expr = local.active.into();

        self.chain_receipt_bus.lookup_key(
            builder,
            ReducedSwirlVaccChainReceiptMessage {
                protocol_digest: self.binding.protocol_digest.map(AB::Expr::from),
                relation_digest: self.binding.relation_digest.map(AB::Expr::from),
                warp_index_digest: self.binding.warp_index_digest.map(AB::Expr::from),
                schedule_digest: self.binding.schedule_digest.map(AB::Expr::from),
                manifest_digest: local.manifest_digest.map(Into::into),
                source_count: local.source_count.into(),
                call_count: local.call_count.into(),
                final_accumulator_digest: local.final_accumulator_digest.map(Into::into),
                final_accumulator_root: local.final_accumulator_root.map(Into::into),
            },
            active.clone(),
        );
        self.terminal_receipt_bus.lookup_key(
            builder,
            ReducedSwirlLocalTerminalReceiptMessage {
                protocol_digest: self.binding.protocol_digest.map(AB::Expr::from),
                relation_digest: self.binding.relation_digest.map(AB::Expr::from),
                terminal_index_digest: self.binding.terminal_index_digest.map(AB::Expr::from),
                verifier_component_digest: self
                    .binding
                    .verifier_component_digest
                    .map(AB::Expr::from),
                final_accumulator_digest: local.final_accumulator_digest.map(Into::into),
                final_accumulator_root: local.final_accumulator_root.map(Into::into),
            },
            active,
        );
    }
}

#[derive(Clone, Debug)]
pub struct ReducedSwirlTransitionFinalizerAir {
    binding: ReducedSwirlTransitionFinalizerBinding,
    child_air_count: usize,
    public_values_bus: PublicValuesBus,
    cached_commit_bus: CachedCommitBus,
    pre_hash_bus: PreHashBus,
    compress_bus: Poseidon2CompressBus,
    reconciliation_bus: ReducedSwirlManifestReconciliationReceiptBus,
    resume_bus: ResumeTranscriptStateBus,
    chain_end_bus: ReducedSwirlVaccChainEndBus,
    manifest_digest_bus: ReducedSwirlManifestDigestBus,
    execution_bus: ReducedSwirlExecutionBus,
    chain_genesis: Digest,
}

impl ReducedSwirlTransitionFinalizerAir {
    #[allow(clippy::too_many_arguments)]
    fn new(
        binding: ReducedSwirlTransitionFinalizerBinding,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        public_values_bus: PublicValuesBus,
        cached_commit_bus: CachedCommitBus,
        pre_hash_bus: PreHashBus,
        compress_bus: Poseidon2CompressBus,
        reconciliation_bus: ReducedSwirlManifestReconciliationReceiptBus,
        resume_bus: ResumeTranscriptStateBus,
        chain_end_bus: ReducedSwirlVaccChainEndBus,
        manifest_digest_bus: ReducedSwirlManifestDigestBus,
        execution_bus: ReducedSwirlExecutionBus,
    ) -> Result<Self, ReducedSwirlTransitionFinalizerError> {
        binding.validate()?;
        validate_child_vk_profile(child_vk)?;
        if child_vk.pre_hash != binding.trusted_vk_commits.recursive_vk_commit.vk_pre_hash {
            return Err(ReducedSwirlTransitionFinalizerError::ChildVkPreHash);
        }
        let used = [
            public_values_bus.index(),
            cached_commit_bus.index(),
            pre_hash_bus.index(),
            compress_bus.index(),
            resume_bus.index(),
            chain_end_bus.index(),
            manifest_digest_bus.index(),
            execution_bus.index(),
        ];
        if used
            .iter()
            .enumerate()
            .any(|(index, bus)| used[..index].contains(bus))
        {
            return Err(ReducedSwirlTransitionFinalizerError::ReusedBusIndex);
        }
        Ok(Self {
            child_air_count: child_vk.inner.per_air.len(),
            chain_genesis: reduced_swirl_transition_chain_genesis(binding.source_protocol_digest),
            binding,
            public_values_bus,
            cached_commit_bus,
            pre_hash_bus,
            compress_bus,
            reconciliation_bus,
            resume_bus,
            chain_end_bus,
            manifest_digest_bus,
            execution_bus,
        })
    }

    #[must_use]
    pub const fn binding(&self) -> ReducedSwirlTransitionFinalizerBinding {
        self.binding
    }

    #[must_use]
    pub const fn execution_bus(&self) -> ReducedSwirlExecutionBus {
        self.execution_bus
    }

    pub fn generate_trace(
        &self,
        child_public_values: &[Vec<F>],
        record: &ReducedSwirlTransitionFinalizerRecord,
    ) -> Result<ReducedSwirlTransitionFinalizerTrace, ReducedSwirlTransitionFinalizerError> {
        if child_public_values.len() != self.child_air_count
            || child_public_values
                .get(VERIFIER_PVS_AIR_ID)
                .is_none_or(|values| values.len() != VerifierBasePvs::<u8>::width())
            || child_public_values
                .get(VM_PVS_AIR_ID)
                .is_none_or(|values| values.len() != VmPvs::<u8>::width())
            || child_public_values
                .iter()
                .enumerate()
                .any(|(air_id, values)| {
                    air_id != VERIFIER_PVS_AIR_ID && air_id != VM_PVS_AIR_ID && !values.is_empty()
                })
        {
            return Err(ReducedSwirlTransitionFinalizerError::PublicValueShape);
        }
        let child_verifier_pvs: &VerifierBasePvs<F> =
            child_public_values[VERIFIER_PVS_AIR_ID].as_slice().borrow();
        let child_vm_pvs: &VmPvs<F> = child_public_values[VM_PVS_AIR_ID].as_slice().borrow();
        self.validate_verifier_pvs(child_verifier_pvs)?;
        self.validate_record(child_vm_pvs, record)?;

        let mut compression_inputs = Vec::with_capacity(25);
        let chain_genesis = record_compression(
            self.binding.source_protocol_digest,
            reduced_swirl_transition_chain_genesis_metadata(),
            &mut compression_inputs,
        );
        let initial_hash = state_digest(&record.initial_state, &mut compression_inputs);
        let final_hash = state_digest(&record.final_state, &mut compression_inputs);
        if initial_hash.boundary != child_vm_pvs.initial_root
            || final_hash.boundary != child_vm_pvs.final_root
            || initial_hash.boundary != reduced_swirl_transition_state_digest(&record.initial_state)
            || final_hash.boundary != reduced_swirl_transition_state_digest(&record.final_state)
        {
            return Err(ReducedSwirlTransitionFinalizerError::TreeBoundaryMismatch);
        }

        let width = self.width();
        let mut values = F::zero_vec(2 * width);
        let local: &mut ReducedSwirlTransitionFinalizerCols<F> = values[..width].borrow_mut();
        local.active = F::ONE;
        let depth = child_verifier_pvs.recursion_depth;
        local.depth_inv = depth.inverse();
        local.is_depth_one = F::from_bool(depth == F::ONE);
        local.depth_minus_one_inv = if depth == F::ONE {
            F::ZERO
        } else {
            (depth - F::ONE).inverse()
        };
        local.source_count_inv = record.reconciliation.source_count.inverse();
        local.call_count_inv = record.reconciliation.call_count.inverse();
        local.child_verifier_pvs = *child_verifier_pvs;
        local.child_vm_pvs = *child_vm_pvs;
        write_reconciliation(&mut local.reconciliation, &record.reconciliation);
        local.chain_genesis = chain_genesis;
        write_state(&mut local.initial, &record.initial_state);
        write_state_hash(&mut local.initial_hash, &initial_hash);
        write_state(&mut local.final_state, &record.final_state);
        write_state_hash(&mut local.final_hash, &final_hash);

        let real_vm_pvs = real_vm_pvs(record);
        let resume = ResumeTranscriptStateMessage {
            tidx: record.final_state.transcript_tidx,
            state: record.final_state.transcript_state,
        };
        let chain_end = ReducedSwirlVaccChainEndMessage {
            source_count: record.reconciliation.source_count,
            call_count: record.reconciliation.call_count,
            proof_idx: record.reconciliation.call_count - F::ONE,
            footer_start_tidx: record.final_state.transcript_tidx,
            final_accumulator_digest: record.final_state.accumulator_digest,
            final_accumulator_root: record.final_state.accumulator_root,
        };
        let manifest = ReducedSwirlManifestDigestMessage {
            source_count: record.reconciliation.source_count,
            digest: record.reconciliation.flat_manifest_digest,
        };
        Ok(ReducedSwirlTransitionFinalizerTrace {
            matrix: RowMajorMatrix::new(values, width),
            compression_inputs,
            real_vm_pvs,
            resume,
            chain_end,
            manifest,
        })
    }

    fn validate_verifier_pvs(
        &self,
        pvs: &VerifierBasePvs<F>,
    ) -> Result<(), ReducedSwirlTransitionFinalizerError> {
        let trusted = self.binding.trusted_vk_commits;
        let recursive_ok = if pvs.recursion_depth == F::ONE {
            is_unset_vk_commit(pvs.internal_recursive_vk_commit)
        } else {
            pvs.internal_recursive_vk_commit == trusted.recursive_vk_commit
        };
        if pvs.internal_flag != F::TWO
            || pvs.recursion_depth == F::ZERO
            || pvs.app_vk_commit != trusted.app_vk_commit
            || pvs.leaf_vk_commit != trusted.transition_leaf_vk_commit
            || pvs.internal_for_leaf_vk_commit != trusted.internal_for_leaf_vk_commit
            || !recursive_ok
        {
            return Err(ReducedSwirlTransitionFinalizerError::OrdinaryPvsMismatch);
        }
        Ok(())
    }

    fn validate_record(
        &self,
        child: &VmPvs<F>,
        record: &ReducedSwirlTransitionFinalizerRecord,
    ) -> Result<(), ReducedSwirlTransitionFinalizerError> {
        let initial = &record.initial_state;
        let final_state = &record.final_state;
        let reconciliation = &record.reconciliation;
        if reconciliation.source_count == F::ZERO || reconciliation.call_count == F::ZERO {
            return Err(ReducedSwirlTransitionFinalizerError::EmptyExecution);
        }
        for state in [initial, final_state] {
            if state.source_protocol_digest != self.binding.source_protocol_digest
                || state.warp_protocol_digest != self.binding.warp_protocol_digest
                || state.relation_digest != self.binding.relation_digest
                || state.warp_index_digest != self.binding.warp_index_digest
                || state.schedule_digest != self.binding.schedule_digest
            {
                return Err(ReducedSwirlTransitionFinalizerError::SetupMismatch);
            }
        }
        if initial.call_cursor != F::ZERO
            || initial.source_cursor != F::ZERO
            || initial.transcript_sample_count != F::ZERO
            || initial.transcript_state != [F::ZERO; 16]
            || initial.accumulator_root != [F::ZERO; DIGEST_SIZE]
            || initial.accumulator_digest != [F::ZERO; DIGEST_SIZE]
            || initial.manifest_chain != self.chain_genesis
        {
            return Err(ReducedSwirlTransitionFinalizerError::GenesisMismatch);
        }
        if initial.total_source_count != reconciliation.source_count
            || final_state.total_source_count != reconciliation.source_count
            || final_state.call_cursor != reconciliation.call_count
            || final_state.source_cursor != reconciliation.source_count
        {
            return Err(ReducedSwirlTransitionFinalizerError::CoverageMismatch);
        }
        if final_state.manifest_chain != reconciliation.rolling_chain_endpoint {
            return Err(ReducedSwirlTransitionFinalizerError::ReconciliationMismatch);
        }
        if initial.program_commitment != final_state.program_commitment
            || child.program_commit != initial.program_commitment
        {
            return Err(ReducedSwirlTransitionFinalizerError::ProgramMismatch);
        }
        let initial_boundary = reduced_swirl_transition_state_digest(initial);
        let final_boundary = reduced_swirl_transition_state_digest(final_state);
        if child.initial_pc != F::ZERO
            || child.final_pc != reconciliation.call_count
            || child.exit_code != F::ZERO
            || child.is_terminate != F::ONE
            || child.initial_root != initial_boundary
            || child.final_root != final_boundary
        {
            return Err(ReducedSwirlTransitionFinalizerError::TreeBoundaryMismatch);
        }
        Ok(())
    }

    fn compress<AB: AirBuilder<F = F> + InteractionBuilder>(
        &self,
        builder: &mut AB,
        left: [AB::Expr; DIGEST_SIZE],
        right: [AB::Expr; DIGEST_SIZE],
        output: [AB::Expr; DIGEST_SIZE],
        enabled: AB::Expr,
    ) {
        self.compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: core::array::from_fn(|index| {
                    if index < DIGEST_SIZE {
                        left[index].clone()
                    } else {
                        right[index - DIGEST_SIZE].clone()
                    }
                }),
                output,
            },
            enabled,
        );
    }

    fn eval_state_hash<AB>(
        &self,
        builder: &mut AB,
        state: &ReducedSwirlTransitionStateCols<AB::Var>,
        hash: &ReducedSwirlTransitionStateHashCols<AB::Var>,
        enabled: AB::Expr,
    ) where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        self.compress(
            builder,
            state.warp_protocol_digest.map(Into::into),
            state.relation_digest.map(Into::into),
            hash.setup_left.map(Into::into),
            enabled.clone(),
        );
        self.compress(
            builder,
            state.warp_index_digest.map(Into::into),
            state.schedule_digest.map(Into::into),
            hash.setup_right.map(Into::into),
            enabled.clone(),
        );
        self.compress(
            builder,
            hash.setup_left.map(Into::into),
            hash.setup_right.map(Into::into),
            hash.setup.map(Into::into),
            enabled.clone(),
        );
        self.compress(
            builder,
            state.source_protocol_digest.map(Into::into),
            hash.setup.map(Into::into),
            hash.protocol.map(Into::into),
            enabled.clone(),
        );
        let metadata = [
            AB::Expr::from_u32(REDUCED_SWIRL_TRANSITION_STATE_METADATA_TAG),
            AB::Expr::from_u32(REDUCED_SWIRL_TRANSITION_LEAF_PROTOCOL_VERSION),
            state.total_source_count.into(),
            state.call_cursor.into(),
            state.source_cursor.into(),
            state.transcript_tidx.into(),
            state.transcript_sample_count.into(),
            state.vm_pc.into(),
        ];
        self.compress(
            builder,
            hash.protocol.map(Into::into),
            metadata,
            hash.control.map(Into::into),
            enabled.clone(),
        );
        self.compress(
            builder,
            core::array::from_fn(|index| state.transcript_state[index].into()),
            core::array::from_fn(|index| state.transcript_state[DIGEST_SIZE + index].into()),
            hash.transcript.map(Into::into),
            enabled.clone(),
        );
        self.compress(
            builder,
            state.accumulator_root.map(Into::into),
            state.accumulator_digest.map(Into::into),
            hash.accumulator.map(Into::into),
            enabled.clone(),
        );
        self.compress(
            builder,
            hash.transcript.map(Into::into),
            hash.accumulator.map(Into::into),
            hash.transcript_accumulator.map(Into::into),
            enabled.clone(),
        );
        self.compress(
            builder,
            state.vm_root.map(Into::into),
            state.manifest_chain.map(Into::into),
            hash.vm_chain.map(Into::into),
            enabled.clone(),
        );
        self.compress(
            builder,
            hash.vm_chain.map(Into::into),
            state.program_commitment.map(Into::into),
            hash.vm_program.map(Into::into),
            enabled.clone(),
        );
        self.compress(
            builder,
            hash.control.map(Into::into),
            hash.transcript_accumulator.map(Into::into),
            hash.left.map(Into::into),
            enabled.clone(),
        );
        self.compress(
            builder,
            hash.left.map(Into::into),
            hash.vm_program.map(Into::into),
            hash.boundary.map(Into::into),
            enabled,
        );
    }
}

impl BaseAir<F> for ReducedSwirlTransitionFinalizerAir {
    fn width(&self) -> usize {
        core::mem::size_of::<ReducedSwirlTransitionFinalizerCols<u8>>()
    }
}

impl BaseAirWithPublicValues<F> for ReducedSwirlTransitionFinalizerAir {}
impl PartitionedBaseAir<F> for ReducedSwirlTransitionFinalizerAir {}

impl<AB> Air<AB> for ReducedSwirlTransitionFinalizerAir
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("transition finalizer row");
        let next_row = main.row_slice(1).expect("transition finalizer padding row");
        let local: &ReducedSwirlTransitionFinalizerCols<AB::Var> = (*row).borrow();
        let next: &ReducedSwirlTransitionFinalizerCols<AB::Var> = (*next_row).borrow();
        let active = AB::Expr::from(local.active);
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_last_row().assert_zero(local.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.active);
        for value in next.as_slice().iter().skip(1) {
            builder.when(active.clone()).assert_zero(*value);
        }

        builder.assert_bool(local.is_depth_one);
        let depth = AB::Expr::from(local.child_verifier_pvs.recursion_depth);
        let depth_minus_one = depth.clone() - AB::Expr::ONE;
        builder
            .when(active.clone())
            .assert_one(depth.clone() * local.depth_inv);
        builder.when(active.clone()).assert_eq(
            depth_minus_one.clone() * local.depth_minus_one_inv,
            AB::Expr::ONE - local.is_depth_one,
        );
        builder
            .when(active.clone())
            .assert_zero(depth_minus_one * local.is_depth_one);
        builder
            .when(active.clone())
            .assert_one(AB::Expr::from(local.reconciliation.source_count) * local.source_count_inv);
        builder
            .when(active.clone())
            .assert_one(AB::Expr::from(local.reconciliation.call_count) * local.call_count_inv);

        let trusted = self.binding.trusted_vk_commits;
        builder
            .when(active.clone())
            .assert_eq(local.child_verifier_pvs.internal_flag, AB::Expr::TWO);
        assert_vk_commit_eq(
            builder,
            active.clone(),
            local.child_verifier_pvs.app_vk_commit,
            trusted.app_vk_commit,
        );
        assert_vk_commit_eq(
            builder,
            active.clone(),
            local.child_verifier_pvs.leaf_vk_commit,
            trusted.transition_leaf_vk_commit,
        );
        assert_vk_commit_eq(
            builder,
            active.clone(),
            local.child_verifier_pvs.internal_for_leaf_vk_commit,
            trusted.internal_for_leaf_vk_commit,
        );
        for limb in 0..DIGEST_SIZE {
            for (actual, expected) in [
                (
                    local
                        .child_verifier_pvs
                        .internal_recursive_vk_commit
                        .cached_commit[limb],
                    trusted.recursive_vk_commit.cached_commit[limb],
                ),
                (
                    local
                        .child_verifier_pvs
                        .internal_recursive_vk_commit
                        .vk_pre_hash[limb],
                    trusted.recursive_vk_commit.vk_pre_hash[limb],
                ),
            ] {
                builder.when(active.clone()).assert_eq(
                    actual,
                    (AB::Expr::ONE - local.is_depth_one) * AB::Expr::from(expected),
                );
            }
        }

        for (pv_idx, value) in local
            .child_verifier_pvs
            .as_slice()
            .iter()
            .copied()
            .enumerate()
        {
            self.public_values_bus.receive(
                builder,
                AB::Expr::from_usize(TRANSITION_TREE_PROOF_INDEX),
                PublicValuesBusMessage {
                    air_idx: AB::Expr::from_usize(VERIFIER_PVS_AIR_ID),
                    pv_idx: AB::Expr::from_usize(pv_idx),
                    value: value.into(),
                },
                active.clone(),
            );
        }
        for (pv_idx, value) in local.child_vm_pvs.as_slice().iter().copied().enumerate() {
            self.public_values_bus.receive(
                builder,
                AB::Expr::from_usize(TRANSITION_TREE_PROOF_INDEX),
                PublicValuesBusMessage {
                    air_idx: AB::Expr::from_usize(VM_PVS_AIR_ID),
                    pv_idx: AB::Expr::from_usize(pv_idx),
                    value: value.into(),
                },
                active.clone(),
            );
        }
        self.cached_commit_bus.receive(
            builder,
            AB::Expr::from_usize(TRANSITION_TREE_PROOF_INDEX),
            CachedCommitBusMessage {
                air_idx: AB::Expr::from_usize(CONSTRAINT_EVAL_AIR_ID),
                cached_idx: AB::Expr::from_usize(CONSTRAINT_EVAL_CACHED_INDEX),
                global_cached_idx: AB::Expr::ZERO,
                cached_commit: core::array::from_fn(|limb| {
                    AB::Expr::from(local.is_depth_one)
                        * AB::Expr::from(trusted.internal_for_leaf_vk_commit.cached_commit[limb])
                        + (AB::Expr::ONE - local.is_depth_one)
                            * AB::Expr::from(trusted.recursive_vk_commit.cached_commit[limb])
                }),
            },
            active.clone(),
        );
        self.pre_hash_bus.receive(
            builder,
            AB::Expr::from_usize(TRANSITION_TREE_PROOF_INDEX),
            PreHashMessage {
                vk_pre_hash: trusted.recursive_vk_commit.vk_pre_hash.map(AB::Expr::from),
            },
            active.clone(),
        );

        for state in [&local.initial, &local.final_state] {
            for (actual, expected) in [
                (
                    state.source_protocol_digest,
                    self.binding.source_protocol_digest,
                ),
                (
                    state.warp_protocol_digest,
                    self.binding.warp_protocol_digest,
                ),
                (state.relation_digest, self.binding.relation_digest),
                (state.warp_index_digest, self.binding.warp_index_digest),
                (state.schedule_digest, self.binding.schedule_digest),
            ] {
                assert_digest_eq_const(builder, active.clone(), actual, expected);
            }
            builder
                .when(active.clone())
                .assert_eq(state.total_source_count, local.reconciliation.source_count);
        }
        for value in [
            local.initial.call_cursor,
            local.initial.source_cursor,
            local.initial.transcript_sample_count,
        ] {
            builder.when(active.clone()).assert_zero(value);
        }
        for value in local
            .initial
            .transcript_state
            .iter()
            .chain(&local.initial.accumulator_root)
            .chain(&local.initial.accumulator_digest)
        {
            builder.when(active.clone()).assert_zero(*value);
        }
        assert_digest_eq(
            builder,
            active.clone(),
            local.initial.manifest_chain,
            local.chain_genesis,
        );
        builder.when(active.clone()).assert_eq(
            local.final_state.call_cursor,
            local.reconciliation.call_count,
        );
        builder.when(active.clone()).assert_eq(
            local.final_state.source_cursor,
            local.reconciliation.source_count,
        );
        assert_digest_eq(
            builder,
            active.clone(),
            local.final_state.manifest_chain,
            local.reconciliation.rolling_chain_endpoint,
        );
        assert_digest_eq(
            builder,
            active.clone(),
            local.initial.program_commitment,
            local.final_state.program_commitment,
        );

        self.compress(
            builder,
            self.binding.source_protocol_digest.map(AB::Expr::from),
            reduced_swirl_transition_chain_genesis_metadata().map(AB::Expr::from),
            local.chain_genesis.map(Into::into),
            active.clone(),
        );
        self.eval_state_hash(builder, &local.initial, &local.initial_hash, active.clone());
        self.eval_state_hash(
            builder,
            &local.final_state,
            &local.final_hash,
            active.clone(),
        );

        assert_digest_eq(
            builder,
            active.clone(),
            local.child_vm_pvs.program_commit,
            local.initial.program_commitment,
        );
        assert_digest_eq(
            builder,
            active.clone(),
            local.child_vm_pvs.initial_root,
            local.initial_hash.boundary,
        );
        assert_digest_eq(
            builder,
            active.clone(),
            local.child_vm_pvs.final_root,
            local.final_hash.boundary,
        );
        builder
            .when(active.clone())
            .assert_zero(local.child_vm_pvs.initial_pc);
        builder
            .when(active.clone())
            .assert_eq(local.child_vm_pvs.final_pc, local.reconciliation.call_count);
        builder
            .when(active.clone())
            .assert_zero(local.child_vm_pvs.exit_code);
        builder
            .when(active.clone())
            .assert_one(local.child_vm_pvs.is_terminate);

        self.reconciliation_bus.lookup_key(
            builder,
            ReducedSwirlManifestReconciliationReceiptMessage {
                source_count: local.reconciliation.source_count.into(),
                call_count: local.reconciliation.call_count.into(),
                flat_manifest_digest: local.reconciliation.flat_manifest_digest.map(Into::into),
                rolling_chain_endpoint: local.reconciliation.rolling_chain_endpoint.map(Into::into),
            },
            active.clone(),
        );
        self.resume_bus.send(
            builder,
            AB::Expr::from_usize(TRANSITION_TERMINAL_PROOF_INDEX),
            ResumeTranscriptStateMessage {
                tidx: local.final_state.transcript_tidx.into(),
                state: local.final_state.transcript_state.map(Into::into),
            },
            active.clone(),
        );
        self.chain_end_bus.send(
            builder,
            ReducedSwirlVaccChainEndMessage {
                source_count: local.reconciliation.source_count.into(),
                call_count: local.reconciliation.call_count.into(),
                proof_idx: AB::Expr::from(local.reconciliation.call_count) - AB::Expr::ONE,
                footer_start_tidx: local.final_state.transcript_tidx.into(),
                final_accumulator_digest: local.final_state.accumulator_digest.map(Into::into),
                final_accumulator_root: local.final_state.accumulator_root.map(Into::into),
            },
            active.clone(),
        );
        self.manifest_digest_bus.send(
            builder,
            ReducedSwirlManifestDigestMessage {
                source_count: local.reconciliation.source_count.into(),
                digest: local.reconciliation.flat_manifest_digest.map(Into::into),
            },
            active.clone(),
        );
        self.execution_bus.lookup_key(
            builder,
            ReducedSwirlExecutionMessage {
                vm_pvs: VmPvs {
                    program_commit: local.initial.program_commitment.map(Into::into),
                    initial_pc: local.initial.vm_pc.into(),
                    final_pc: local.final_state.vm_pc.into(),
                    exit_code: AB::Expr::ZERO,
                    is_terminate: AB::Expr::ONE,
                    initial_root: local.initial.vm_root.map(Into::into),
                    final_root: local.final_state.vm_root.map(Into::into),
                },
            },
            active,
        );
    }
}

/// Complete one-child recursive verifier plus the transition endpoint bridge.
/// The terminal footer, transcript, Decide, and public `VmPvs` AIR are composed
/// around this inventory by the caller using the exported typed buses.
#[derive(Clone)]
pub struct ReducedSwirlTransitionFinalizerCircuit {
    child_vk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
    verifier: Arc<VerifierSubCircuit<1>>,
    finalizer: Arc<ReducedSwirlTransitionFinalizerAir>,
    next_bus_idx: BusIndex,
}

impl ReducedSwirlTransitionFinalizerCircuit {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        child_vk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
        binding: ReducedSwirlTransitionFinalizerBinding,
        reconciliation_bus_index: BusIndex,
        resume_bus: ResumeTranscriptStateBus,
        chain_end_bus: ReducedSwirlVaccChainEndBus,
        manifest_digest_bus: ReducedSwirlManifestDigestBus,
        execution_bus: ReducedSwirlExecutionBus,
        bus_idx_manager: BusIndexManager,
    ) -> Result<Self, ReducedSwirlTransitionFinalizerError> {
        let external = [
            reconciliation_bus_index,
            resume_bus.index(),
            chain_end_bus.index(),
            manifest_digest_bus.index(),
            execution_bus.index(),
        ];
        if external
            .iter()
            .enumerate()
            .any(|(index, bus)| external[..index].contains(bus))
            || external
                .iter()
                .any(|bus| *bus >= bus_idx_manager.next_bus_idx())
        {
            return Err(ReducedSwirlTransitionFinalizerError::ReusedBusIndex);
        }
        let mut verifier = VerifierSubCircuit::<1>::new_with_options_from_bus_idx_manager(
            child_vk.clone(),
            VerifierConfig {
                continuations_enabled: true,
                final_state_bus_enabled: false,
                has_cached: false,
                tail_mode: VerifierTailMode::Complete,
            },
            bus_idx_manager,
        );
        let expected_dag_commit = verifier
            .cached_trace_record_for_child(child_vk.as_ref())
            .dag_commit_info
            .ok_or(ReducedSwirlTransitionFinalizerError::SymbolicDagAuthority)?
            .commit;
        verifier
            .bind_fixed_dag_commit(expected_dag_commit)
            .map_err(|_| ReducedSwirlTransitionFinalizerError::SymbolicDagAuthority)?;
        if verifier.max_num_proofs() != 1 {
            return Err(ReducedSwirlTransitionFinalizerError::ChildPublicValueProfile);
        }
        let buses = verifier.bus_inventory();
        let finalizer = ReducedSwirlTransitionFinalizerAir::new(
            binding,
            child_vk.as_ref(),
            buses.public_values_bus,
            buses.cached_commit_bus,
            buses.pre_hash_bus,
            buses.poseidon2_compress_bus,
            ReducedSwirlManifestReconciliationReceiptBus::new(reconciliation_bus_index),
            resume_bus,
            chain_end_bus,
            manifest_digest_bus,
            execution_bus,
        )?;
        let next_bus_idx = verifier.next_bus_idx();
        Ok(Self {
            child_vk,
            verifier: Arc::new(verifier),
            finalizer: Arc::new(finalizer),
            next_bus_idx,
        })
    }

    #[must_use]
    pub fn child_vk(&self) -> &Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>> {
        &self.child_vk
    }

    #[must_use]
    pub fn verifier(&self) -> &Arc<VerifierSubCircuit<1>> {
        &self.verifier
    }

    #[must_use]
    pub fn finalizer(&self) -> &Arc<ReducedSwirlTransitionFinalizerAir> {
        &self.finalizer
    }

    #[must_use]
    pub const fn next_bus_idx(&self) -> BusIndex {
        self.next_bus_idx
    }
}

impl<SC: StarkProtocolConfig<F = F>> Circuit<SC> for ReducedSwirlTransitionFinalizerCircuit {
    fn airs(&self) -> Vec<AirRef<SC>> {
        core::iter::once(self.finalizer.clone() as AirRef<SC>)
            .chain(self.verifier.airs::<SC>())
            .collect()
    }
}

fn validate_child_vk_profile(
    child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
) -> Result<(), ReducedSwirlTransitionFinalizerError> {
    if child_vk
        .inner
        .per_air
        .get(VERIFIER_PVS_AIR_ID)
        .is_none_or(|air| air.params.num_public_values != VerifierBasePvs::<u8>::width())
        || child_vk
            .inner
            .per_air
            .get(VM_PVS_AIR_ID)
            .is_none_or(|air| air.params.num_public_values != VmPvs::<u8>::width())
        || child_vk
            .inner
            .per_air
            .iter()
            .enumerate()
            .any(|(air_id, air)| {
                air_id != VERIFIER_PVS_AIR_ID
                    && air_id != VM_PVS_AIR_ID
                    && air.params.num_public_values != 0
            })
    {
        return Err(ReducedSwirlTransitionFinalizerError::ChildPublicValueProfile);
    }
    if child_vk
        .inner
        .per_air
        .get(CONSTRAINT_EVAL_AIR_ID)
        .is_none_or(|air| air.params.width.cached_mains.len() != 1)
        || child_vk
            .inner
            .per_air
            .iter()
            .enumerate()
            .any(|(air_id, air)| {
                air_id != CONSTRAINT_EVAL_AIR_ID && !air.params.width.cached_mains.is_empty()
            })
    {
        return Err(ReducedSwirlTransitionFinalizerError::ChildCachedProfile);
    }
    Ok(())
}

fn write_state(dst: &mut ReducedSwirlTransitionStateCols<F>, src: &ReducedSwirlTransitionState) {
    dst.source_protocol_digest = src.source_protocol_digest;
    dst.warp_protocol_digest = src.warp_protocol_digest;
    dst.relation_digest = src.relation_digest;
    dst.warp_index_digest = src.warp_index_digest;
    dst.schedule_digest = src.schedule_digest;
    dst.total_source_count = src.total_source_count;
    dst.call_cursor = src.call_cursor;
    dst.source_cursor = src.source_cursor;
    dst.transcript_tidx = src.transcript_tidx;
    dst.transcript_sample_count = src.transcript_sample_count;
    dst.transcript_state = src.transcript_state;
    dst.accumulator_root = src.accumulator_root;
    dst.accumulator_digest = src.accumulator_digest;
    dst.vm_pc = src.vm_pc;
    dst.vm_root = src.vm_root;
    dst.manifest_chain = src.manifest_chain;
    dst.program_commitment = src.program_commitment;
}

fn write_state_hash(dst: &mut ReducedSwirlTransitionStateHashCols<F>, src: &StateHashWitness) {
    dst.setup_left = src.setup_left;
    dst.setup_right = src.setup_right;
    dst.setup = src.setup;
    dst.protocol = src.protocol;
    dst.control = src.control;
    dst.transcript = src.transcript;
    dst.accumulator = src.accumulator;
    dst.transcript_accumulator = src.transcript_accumulator;
    dst.vm_chain = src.vm_chain;
    dst.vm_program = src.vm_program;
    dst.left = src.left;
    dst.boundary = src.boundary;
}

fn write_reconciliation(
    dst: &mut ReducedSwirlManifestReconciliationReceiptMessage<F>,
    src: &ReducedSwirlManifestReconciliationReceiptMessage<F>,
) {
    dst.source_count = src.source_count;
    dst.call_count = src.call_count;
    dst.flat_manifest_digest = src.flat_manifest_digest;
    dst.rolling_chain_endpoint = src.rolling_chain_endpoint;
}

fn real_vm_pvs(record: &ReducedSwirlTransitionFinalizerRecord) -> VmPvs<F> {
    VmPvs {
        program_commit: record.initial_state.program_commitment,
        initial_pc: record.initial_state.vm_pc,
        final_pc: record.final_state.vm_pc,
        exit_code: F::ZERO,
        is_terminate: F::ONE,
        initial_root: record.initial_state.vm_root,
        final_root: record.final_state.vm_root,
    }
}

fn record_compression(
    left: Digest,
    right: Digest,
    inputs: &mut Vec<[F; 2 * DIGEST_SIZE]>,
) -> Digest {
    inputs.push(core::array::from_fn(|index| {
        if index < DIGEST_SIZE {
            left[index]
        } else {
            right[index - DIGEST_SIZE]
        }
    }));
    poseidon2_compress_with_capacity(left, right).0
}

fn assert_vk_commit_eq<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    enabled: AB::Expr,
    actual: VkCommit<AB::Var>,
    expected: VkCommit<F>,
) where
    AB::Var: Copy,
{
    assert_digest_eq_const(
        builder,
        enabled.clone(),
        actual.cached_commit,
        expected.cached_commit,
    );
    assert_digest_eq_const(builder, enabled, actual.vk_pre_hash, expected.vk_pre_hash);
}

fn assert_digest_eq<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    enabled: AB::Expr,
    actual: [AB::Var; DIGEST_SIZE],
    expected: [AB::Var; DIGEST_SIZE],
) where
    AB::Var: Copy,
{
    for limb in 0..DIGEST_SIZE {
        builder
            .when(enabled.clone())
            .assert_eq(actual[limb], expected[limb]);
    }
}

fn assert_digest_eq_const<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    enabled: AB::Expr,
    actual: [AB::Var; DIGEST_SIZE],
    expected: Digest,
) where
    AB::Var: Copy,
{
    for limb in 0..DIGEST_SIZE {
        builder
            .when(enabled.clone())
            .assert_eq(actual[limb], AB::Expr::from(expected[limb]));
    }
}

fn is_zero_digest(digest: &Digest) -> bool {
    digest.iter().all(|value| *value == F::ZERO)
}

fn is_unset_vk_commit(commit: VkCommit<F>) -> bool {
    is_zero_digest(&commit.cached_commit) && is_zero_digest(&commit.vk_pre_hash)
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use openvm_stark_backend::{
        air_builders::{
            debug::{check_constraints, check_logup},
            symbolic::get_symbolic_builder,
        },
        interaction::SymbolicInteraction,
        keygen::types::TraceWidth,
        p3_air::AirBuilder,
        p3_matrix::Matrix,
        AirRef, AnyAir, StarkEngine,
    };
    use openvm_stark_sdk::config::{
        baby_bear_poseidon2::{
            BabyBearPoseidon2Config as NativeSC, BabyBearPoseidon2CpuEngine, DuplexSponge,
        },
        internal_params_with_100_bits_security,
    };

    use super::*;

    fn digest(seed: u32) -> Digest {
        core::array::from_fn(|index| F::from_u32(seed + index as u32 + 1))
    }

    fn vk_commit(seed: u32) -> VkCommit<F> {
        VkCommit {
            cached_commit: digest(seed),
            vk_pre_hash: digest(seed + 100),
        }
    }

    #[derive(Clone, Debug)]
    struct TestAir {
        pvs: usize,
        cached: bool,
    }

    impl BaseAir<F> for TestAir {
        fn width(&self) -> usize {
            1 + usize::from(self.cached)
        }
    }

    impl BaseAirWithPublicValues<F> for TestAir {
        fn num_public_values(&self) -> usize {
            self.pvs
        }
    }

    impl PartitionedBaseAir<F> for TestAir {
        fn cached_main_widths(&self) -> Vec<usize> {
            self.cached.then_some(1).into_iter().collect()
        }

        fn common_main_width(&self) -> usize {
            1
        }
    }

    impl<AB: openvm_stark_backend::air_builders::PartitionedAirBuilder<F = F>> Air<AB> for TestAir {
        fn eval(&self, builder: &mut AB) {
            let value = {
                let main = builder.common_main();
                let row = main.row_slice(0).expect("test AIR row");
                row[0].clone()
            };
            builder.assert_zero(value);
        }
    }

    fn test_vk() -> MultiStarkVerifyingKey<NativeSC> {
        let engine: BabyBearPoseidon2CpuEngine<DuplexSponge> =
            BabyBearPoseidon2CpuEngine::new(internal_params_with_100_bits_security());
        let airs: Vec<AirRef<NativeSC>> = (0..=CONSTRAINT_EVAL_AIR_ID)
            .map(|air_id| {
                Arc::new(TestAir {
                    pvs: match air_id {
                        VERIFIER_PVS_AIR_ID => VerifierBasePvs::<u8>::width(),
                        VM_PVS_AIR_ID => VmPvs::<u8>::width(),
                        _ => 0,
                    },
                    cached: air_id == CONSTRAINT_EVAL_AIR_ID,
                }) as AirRef<NativeSC>
            })
            .collect();
        engine.keygen(&airs).1
    }

    fn binding(root_pre_hash: Digest) -> ReducedSwirlTransitionFinalizerBinding {
        let mut recursive_vk_commit = vk_commit(180);
        recursive_vk_commit.vk_pre_hash = root_pre_hash;
        ReducedSwirlTransitionFinalizerBinding {
            source_protocol_digest: digest(1),
            warp_protocol_digest: digest(20),
            relation_digest: digest(40),
            warp_index_digest: digest(60),
            schedule_digest: digest(80),
            trusted_vk_commits: ReducedSwirlTransitionTreeTrustedVkCommits {
                app_vk_commit: vk_commit(100),
                transition_leaf_vk_commit: vk_commit(120),
                internal_for_leaf_vk_commit: vk_commit(140),
                recursive_vk_commit,
            },
        }
    }

    fn verifier_pvs(
        binding: ReducedSwirlTransitionFinalizerBinding,
        depth: u32,
    ) -> VerifierBasePvs<F> {
        VerifierBasePvs {
            internal_flag: F::TWO,
            app_vk_commit: binding.trusted_vk_commits.app_vk_commit,
            leaf_vk_commit: binding.trusted_vk_commits.transition_leaf_vk_commit,
            internal_for_leaf_vk_commit: binding.trusted_vk_commits.internal_for_leaf_vk_commit,
            recursion_depth: F::from_u32(depth),
            internal_recursive_vk_commit: if depth == 1 {
                VkCommit {
                    cached_commit: [F::ZERO; DIGEST_SIZE],
                    vk_pre_hash: [F::ZERO; DIGEST_SIZE],
                }
            } else {
                binding.trusted_vk_commits.recursive_vk_commit
            },
        }
    }

    fn record(
        binding: ReducedSwirlTransitionFinalizerBinding,
    ) -> ReducedSwirlTransitionFinalizerRecord {
        let source_count = F::from_u32(15);
        let call_count = F::from_u32(2);
        let program_commitment = digest(220);
        let rolling_chain_endpoint = digest(240);
        let initial_state = ReducedSwirlTransitionState {
            source_protocol_digest: binding.source_protocol_digest,
            warp_protocol_digest: binding.warp_protocol_digest,
            relation_digest: binding.relation_digest,
            warp_index_digest: binding.warp_index_digest,
            schedule_digest: binding.schedule_digest,
            total_source_count: source_count,
            call_cursor: F::ZERO,
            source_cursor: F::ZERO,
            // The first VACC call starts after the setup-fixed native
            // statement header. Its operation cursor is therefore nonzero,
            // while the starting sponge and sample cursor are still the
            // canonical zero-transcript genesis.
            transcript_tidx: F::from_u32(197),
            transcript_sample_count: F::ZERO,
            transcript_state: [F::ZERO; 16],
            accumulator_root: [F::ZERO; DIGEST_SIZE],
            accumulator_digest: [F::ZERO; DIGEST_SIZE],
            vm_pc: F::from_u32(7),
            vm_root: digest(260),
            manifest_chain: reduced_swirl_transition_chain_genesis(binding.source_protocol_digest),
            program_commitment,
        };
        let final_state = ReducedSwirlTransitionState {
            source_protocol_digest: binding.source_protocol_digest,
            warp_protocol_digest: binding.warp_protocol_digest,
            relation_digest: binding.relation_digest,
            warp_index_digest: binding.warp_index_digest,
            schedule_digest: binding.schedule_digest,
            total_source_count: source_count,
            call_cursor: call_count,
            source_cursor: source_count,
            transcript_tidx: F::from_u32(2_900),
            transcript_sample_count: F::ONE,
            transcript_state: core::array::from_fn(|index| F::from_u32(300 + index as u32)),
            accumulator_root: digest(340),
            accumulator_digest: digest(360),
            vm_pc: F::from_u32(99),
            vm_root: digest(380),
            manifest_chain: rolling_chain_endpoint,
            program_commitment,
        };
        ReducedSwirlTransitionFinalizerRecord {
            initial_state,
            final_state,
            reconciliation: ReducedSwirlManifestReconciliationReceiptMessage {
                source_count,
                call_count,
                flat_manifest_digest: digest(400),
                rolling_chain_endpoint,
            },
        }
    }

    fn child_vm(record: &ReducedSwirlTransitionFinalizerRecord) -> VmPvs<F> {
        VmPvs {
            program_commit: record.initial_state.program_commitment,
            initial_pc: F::ZERO,
            final_pc: record.reconciliation.call_count,
            exit_code: F::ZERO,
            is_terminate: F::ONE,
            initial_root: reduced_swirl_transition_state_digest(&record.initial_state),
            final_root: reduced_swirl_transition_state_digest(&record.final_state),
        }
    }

    fn public_values(
        binding: ReducedSwirlTransitionFinalizerBinding,
        record: &ReducedSwirlTransitionFinalizerRecord,
    ) -> Vec<Vec<F>> {
        let mut values = vec![Vec::new(); CONSTRAINT_EVAL_AIR_ID + 1];
        values[VERIFIER_PVS_AIR_ID] = verifier_pvs(binding, 2).as_slice().to_vec();
        values[VM_PVS_AIR_ID] = child_vm(record).as_slice().to_vec();
        values
    }

    fn test_finalizer() -> (
        ReducedSwirlTransitionFinalizerAir,
        ReducedSwirlTransitionFinalizerBinding,
    ) {
        let vk = test_vk();
        let binding = binding(vk.pre_hash);
        let air = ReducedSwirlTransitionFinalizerAir::new(
            binding,
            &vk,
            PublicValuesBus::new(10),
            CachedCommitBus::new(11),
            PreHashBus::new(12),
            Poseidon2CompressBus::new(13),
            ReducedSwirlManifestReconciliationReceiptBus::new(14),
            ResumeTranscriptStateBus::new(15),
            ReducedSwirlVaccChainEndBus::new(16),
            ReducedSwirlManifestDigestBus::new(17),
            ReducedSwirlExecutionBus::new(18),
        )
        .expect("valid transition finalizer");
        (air, binding)
    }

    #[test]
    fn exact_state_openings_and_terminal_exports_are_constrained() {
        let (air, binding) = test_finalizer();
        let record = record(binding);
        let public_values = public_values(binding, &record);
        let trace = air
            .generate_trace(&public_values, &record)
            .expect("valid transition-tree endpoint");
        assert_eq!(trace.compression_inputs.len(), 25);
        assert_eq!(
            trace.real_vm_pvs.as_slice(),
            real_vm_pvs(&record).as_slice()
        );
        assert_eq!(trace.resume.tidx, record.final_state.transcript_tidx);
        assert_eq!(trace.resume.state, record.final_state.transcript_state);
        assert_eq!(trace.chain_end.proof_idx, F::ONE);
        assert_eq!(
            trace.chain_end.final_accumulator_root,
            record.final_state.accumulator_root
        );
        assert_eq!(
            trace.manifest.digest,
            record.reconciliation.flat_manifest_digest
        );
        check_constraints::<_, NativeSC>(
            &air,
            "ReducedSwirlTransitionFinalizerAir",
            &None,
            &[trace.matrix.as_view()],
            &[],
        );
    }

    fn assert_record_rejected(
        mutate: impl FnOnce(&mut ReducedSwirlTransitionFinalizerRecord),
        expected: ReducedSwirlTransitionFinalizerError,
    ) {
        let (air, binding) = test_finalizer();
        let mut record = record(binding);
        let public_values = public_values(binding, &record);
        mutate(&mut record);
        assert_eq!(
            air.generate_trace(&public_values, &record).err(),
            Some(expected)
        );
    }

    #[test]
    fn wrong_genesis_and_final_coverage_are_rejected() {
        assert_record_rejected(
            |record| record.initial_state.manifest_chain[0] += F::ONE,
            ReducedSwirlTransitionFinalizerError::GenesisMismatch,
        );
        assert_record_rejected(
            |record| record.initial_state.transcript_state[7] += F::ONE,
            ReducedSwirlTransitionFinalizerError::GenesisMismatch,
        );
        assert_record_rejected(
            |record| record.initial_state.accumulator_root[1] += F::ONE,
            ReducedSwirlTransitionFinalizerError::GenesisMismatch,
        );
        assert_record_rejected(
            |record| record.initial_state.transcript_tidx += F::ONE,
            ReducedSwirlTransitionFinalizerError::TreeBoundaryMismatch,
        );
        assert_record_rejected(
            |record| record.final_state.call_cursor += F::ONE,
            ReducedSwirlTransitionFinalizerError::CoverageMismatch,
        );
        assert_record_rejected(
            |record| record.final_state.source_cursor -= F::ONE,
            ReducedSwirlTransitionFinalizerError::CoverageMismatch,
        );
        assert_record_rejected(
            |record| record.reconciliation.source_count += F::ONE,
            ReducedSwirlTransitionFinalizerError::CoverageMismatch,
        );
        assert_record_rejected(
            |record| record.reconciliation.call_count += F::ONE,
            ReducedSwirlTransitionFinalizerError::CoverageMismatch,
        );
    }

    #[test]
    fn wrong_tree_checkpoint_accumulator_and_reconciliation_are_rejected() {
        let (air, binding) = test_finalizer();
        let record = record(binding);
        let mut initial_root_values = public_values(binding, &record);
        initial_root_values[VM_PVS_AIR_ID][DIGEST_SIZE + 4] += F::ONE;
        assert_eq!(
            air.generate_trace(&initial_root_values, &record).err(),
            Some(ReducedSwirlTransitionFinalizerError::TreeBoundaryMismatch)
        );
        let mut final_root_values = public_values(binding, &record);
        final_root_values[VM_PVS_AIR_ID][2 * DIGEST_SIZE + 4] += F::ONE;
        assert_eq!(
            air.generate_trace(&final_root_values, &record).err(),
            Some(ReducedSwirlTransitionFinalizerError::TreeBoundaryMismatch)
        );

        assert_record_rejected(
            |record| record.reconciliation.rolling_chain_endpoint[0] += F::ONE,
            ReducedSwirlTransitionFinalizerError::ReconciliationMismatch,
        );
        assert_record_rejected(
            |record| record.final_state.transcript_tidx += F::ONE,
            ReducedSwirlTransitionFinalizerError::TreeBoundaryMismatch,
        );
        assert_record_rejected(
            |record| record.final_state.transcript_state[3] += F::ONE,
            ReducedSwirlTransitionFinalizerError::TreeBoundaryMismatch,
        );
        assert_record_rejected(
            |record| record.final_state.accumulator_root[2] += F::ONE,
            ReducedSwirlTransitionFinalizerError::TreeBoundaryMismatch,
        );
        assert_record_rejected(
            |record| record.final_state.accumulator_digest[4] += F::ONE,
            ReducedSwirlTransitionFinalizerError::TreeBoundaryMismatch,
        );
    }

    #[test]
    fn vk_lineage_and_child_success_are_not_host_assumptions() {
        let (air, finalizer_binding) = test_finalizer();
        let record = record(finalizer_binding);
        let mut child_public_values = public_values(finalizer_binding, &record);
        child_public_values[VERIFIER_PVS_AIR_ID][DIGEST_SIZE * 2] += F::ONE;
        assert_eq!(
            air.generate_trace(&child_public_values, &record).err(),
            Some(ReducedSwirlTransitionFinalizerError::OrdinaryPvsMismatch)
        );
        let mut child_public_values = public_values(finalizer_binding, &record);
        child_public_values[VM_PVS_AIR_ID][DIGEST_SIZE + 2] = F::ONE;
        assert_eq!(
            air.generate_trace(&child_public_values, &record).err(),
            Some(ReducedSwirlTransitionFinalizerError::TreeBoundaryMismatch)
        );

        let vk = test_vk();
        let bad_binding = binding(digest(9_999));
        assert_eq!(
            ReducedSwirlTransitionFinalizerAir::new(
                bad_binding,
                &vk,
                PublicValuesBus::new(20),
                CachedCommitBus::new(21),
                PreHashBus::new(22),
                Poseidon2CompressBus::new(23),
                ReducedSwirlManifestReconciliationReceiptBus::new(24),
                ResumeTranscriptStateBus::new(25),
                ReducedSwirlVaccChainEndBus::new(26),
                ReducedSwirlManifestDigestBus::new(27),
                ReducedSwirlExecutionBus::new(28),
            )
            .err(),
            Some(ReducedSwirlTransitionFinalizerError::ChildVkPreHash)
        );
    }

    #[derive(Clone, Debug)]
    struct TestFinalizerAuthorityAir {
        finalizer: ReducedSwirlTransitionFinalizerAir,
    }

    impl BaseAir<F> for TestFinalizerAuthorityAir {
        fn width(&self) -> usize {
            core::mem::size_of::<ReducedSwirlTransitionFinalizerCols<u8>>()
        }
    }
    impl BaseAirWithPublicValues<F> for TestFinalizerAuthorityAir {}
    impl PartitionedBaseAir<F> for TestFinalizerAuthorityAir {}

    impl<AB> Air<AB> for TestFinalizerAuthorityAir
    where
        AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("finalizer authority row");
            let local: &ReducedSwirlTransitionFinalizerCols<AB::Var> = (*row).borrow();
            let active = AB::Expr::from(local.active);
            for (pv_idx, value) in local
                .child_verifier_pvs
                .as_slice()
                .iter()
                .copied()
                .enumerate()
            {
                self.finalizer.public_values_bus.send(
                    builder,
                    AB::Expr::ZERO,
                    PublicValuesBusMessage {
                        air_idx: AB::Expr::from_usize(VERIFIER_PVS_AIR_ID),
                        pv_idx: AB::Expr::from_usize(pv_idx),
                        value: value.into(),
                    },
                    active.clone(),
                );
            }
            for (pv_idx, value) in local.child_vm_pvs.as_slice().iter().copied().enumerate() {
                self.finalizer.public_values_bus.send(
                    builder,
                    AB::Expr::ZERO,
                    PublicValuesBusMessage {
                        air_idx: AB::Expr::from_usize(VM_PVS_AIR_ID),
                        pv_idx: AB::Expr::from_usize(pv_idx),
                        value: value.into(),
                    },
                    active.clone(),
                );
            }
            let trusted = self.finalizer.binding.trusted_vk_commits;
            self.finalizer.cached_commit_bus.send(
                builder,
                AB::Expr::ZERO,
                CachedCommitBusMessage {
                    air_idx: AB::Expr::from_usize(CONSTRAINT_EVAL_AIR_ID),
                    cached_idx: AB::Expr::from_usize(CONSTRAINT_EVAL_CACHED_INDEX),
                    global_cached_idx: AB::Expr::ZERO,
                    cached_commit: core::array::from_fn(|limb| {
                        AB::Expr::from(local.is_depth_one)
                            * AB::Expr::from(
                                trusted.internal_for_leaf_vk_commit.cached_commit[limb],
                            )
                            + (AB::Expr::ONE - local.is_depth_one)
                                * AB::Expr::from(trusted.recursive_vk_commit.cached_commit[limb])
                    }),
                },
                active.clone(),
            );
            self.finalizer.pre_hash_bus.send(
                builder,
                AB::Expr::ZERO,
                PreHashMessage {
                    vk_pre_hash: trusted.recursive_vk_commit.vk_pre_hash.map(AB::Expr::from),
                },
                active.clone(),
            );
            self.finalizer.reconciliation_bus.add_key_with_lookups(
                builder,
                ReducedSwirlManifestReconciliationReceiptMessage {
                    source_count: local.reconciliation.source_count.into(),
                    call_count: local.reconciliation.call_count.into(),
                    flat_manifest_digest: local.reconciliation.flat_manifest_digest.map(Into::into),
                    rolling_chain_endpoint: local
                        .reconciliation
                        .rolling_chain_endpoint
                        .map(Into::into),
                },
                active.clone(),
            );
            self.finalizer.resume_bus.receive(
                builder,
                AB::Expr::ZERO,
                ResumeTranscriptStateMessage {
                    tidx: local.final_state.transcript_tidx.into(),
                    state: local.final_state.transcript_state.map(Into::into),
                },
                active.clone(),
            );
            self.finalizer.chain_end_bus.receive(
                builder,
                ReducedSwirlVaccChainEndMessage {
                    source_count: local.reconciliation.source_count.into(),
                    call_count: local.reconciliation.call_count.into(),
                    proof_idx: AB::Expr::from(local.reconciliation.call_count) - AB::Expr::ONE,
                    footer_start_tidx: local.final_state.transcript_tidx.into(),
                    final_accumulator_digest: local.final_state.accumulator_digest.map(Into::into),
                    final_accumulator_root: local.final_state.accumulator_root.map(Into::into),
                },
                active.clone(),
            );
            self.finalizer.manifest_digest_bus.receive(
                builder,
                ReducedSwirlManifestDigestMessage {
                    source_count: local.reconciliation.source_count.into(),
                    digest: local.reconciliation.flat_manifest_digest.map(Into::into),
                },
                active.clone(),
            );
            self.finalizer.execution_bus.add_key_with_lookups(
                builder,
                ReducedSwirlExecutionMessage {
                    vm_pvs: VmPvs {
                        program_commit: local.initial.program_commitment.map(Into::into),
                        initial_pc: local.initial.vm_pc.into(),
                        final_pc: local.final_state.vm_pc.into(),
                        exit_code: AB::Expr::ZERO,
                        is_terminate: AB::Expr::ONE,
                        initial_root: local.initial.vm_root.map(Into::into),
                        final_root: local.final_state.vm_root.map(Into::into),
                    },
                },
                active,
            );
        }
    }

    #[derive(Clone, Copy, Debug)]
    struct TestCompressionAuthorityAir(Poseidon2CompressBus);

    impl BaseAir<F> for TestCompressionAuthorityAir {
        fn width(&self) -> usize {
            1 + 3 * DIGEST_SIZE
        }
    }
    impl BaseAirWithPublicValues<F> for TestCompressionAuthorityAir {}
    impl PartitionedBaseAir<F> for TestCompressionAuthorityAir {}

    impl<AB> Air<AB> for TestCompressionAuthorityAir
    where
        AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("compression authority row");
            let active = AB::Expr::from(row[0]);
            builder.assert_bool(row[0]);
            self.0.add_key_with_lookups(
                builder,
                Poseidon2CompressMessage {
                    input: core::array::from_fn(|index| row[1 + index].into()),
                    output: core::array::from_fn(|index| row[1 + 2 * DIGEST_SIZE + index].into()),
                },
                active,
            );
        }
    }

    fn compression_trace(inputs: &[[F; 2 * DIGEST_SIZE]]) -> RowMajorMatrix<F> {
        let width = 1 + 3 * DIGEST_SIZE;
        let height = (inputs.len() + 1).next_power_of_two();
        let mut values = F::zero_vec(width * height);
        for (row_index, input) in inputs.iter().enumerate() {
            let row = &mut values[row_index * width..(row_index + 1) * width];
            row[0] = F::ONE;
            row[1..1 + 2 * DIGEST_SIZE].copy_from_slice(input);
            let left = input[..DIGEST_SIZE].try_into().expect("left digest");
            let right = input[DIGEST_SIZE..].try_into().expect("right digest");
            row[1 + 2 * DIGEST_SIZE..]
                .copy_from_slice(&poseidon2_compress_with_capacity(left, right).0);
        }
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

    fn check_composed_logup(
        finalizer: &ReducedSwirlTransitionFinalizerAir,
        finalizer_trace: &RowMajorMatrix<F>,
        authority_trace: &RowMajorMatrix<F>,
        compression: &RowMajorMatrix<F>,
    ) {
        let airs: Vec<AirRef<NativeSC>> = vec![
            Arc::new(finalizer.clone()),
            Arc::new(TestFinalizerAuthorityAir {
                finalizer: finalizer.clone(),
            }),
            Arc::new(TestCompressionAuthorityAir(finalizer.compress_bus)),
        ];
        let matrices = [finalizer_trace, authority_trace, compression];
        let preprocessed_owned = airs
            .iter()
            .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
            .collect::<Vec<_>>();
        let preprocessed = preprocessed_owned
            .iter()
            .map(|matrix| matrix.as_ref().map(RowMajorMatrix::as_view))
            .collect::<Vec<_>>();
        let interactions = airs
            .iter()
            .map(|air| symbolic_interactions(air.as_ref()))
            .collect::<Vec<_>>();
        let views = matrices
            .iter()
            .map(|matrix| vec![matrix.as_view()])
            .collect::<Vec<_>>();
        let public_values = vec![Vec::new(); airs.len()];
        check_logup(
            &airs.iter().map(|air| air.name()).collect::<Vec<_>>(),
            &interactions,
            &preprocessed,
            &views,
            &public_values,
        );
    }

    #[test]
    fn flat_manifest_digest_is_bound_only_through_the_typed_reconciliation_authority() {
        let (air, binding) = test_finalizer();
        let record = record(binding);
        let public_values = public_values(binding, &record);
        let trace = air.generate_trace(&public_values, &record).unwrap();
        let authority_trace = trace.matrix.clone();
        let compression = compression_trace(&trace.compression_inputs);
        check_composed_logup(&air, &trace.matrix, &authority_trace, &compression);

        let mut mutated = trace.matrix.clone();
        let width = mutated.width();
        let local: &mut ReducedSwirlTransitionFinalizerCols<F> =
            mutated.values[..width].borrow_mut();
        local.reconciliation.flat_manifest_digest[0] += F::ONE;
        check_constraints::<_, NativeSC>(
            &air,
            "flat digest uses reconciliation lookup",
            &None,
            &[mutated.as_view()],
            &[],
        );
        let rejected = catch_unwind(AssertUnwindSafe(|| {
            check_composed_logup(&air, &mutated, &authority_trace, &compression);
        }));
        assert!(rejected.is_err(), "mutated flat manifest balanced LogUp");
    }

    #[derive(Clone, Copy, Debug)]
    struct TestTerminalJoinAuthorityAir {
        binding: ReducedSwirlTransitionTerminalJoinBinding,
        chain_receipt_bus: ReducedSwirlVaccChainReceiptBus,
        terminal_receipt_bus: ReducedSwirlLocalTerminalReceiptBus,
        terminal_root_delta: F,
    }

    impl BaseAir<F> for TestTerminalJoinAuthorityAir {
        fn width(&self) -> usize {
            ReducedSwirlTransitionTerminalJoinCols::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for TestTerminalJoinAuthorityAir {}
    impl PartitionedBaseAir<F> for TestTerminalJoinAuthorityAir {}

    impl<AB> Air<AB> for TestTerminalJoinAuthorityAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder + PairBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("terminal authority row");
            let local: &ReducedSwirlTransitionTerminalJoinCols<AB::Var> = (*row).borrow();
            let active: AB::Expr = local.active.into();
            self.chain_receipt_bus.add_key_with_lookups(
                builder,
                ReducedSwirlVaccChainReceiptMessage {
                    protocol_digest: self.binding.protocol_digest.map(AB::Expr::from),
                    relation_digest: self.binding.relation_digest.map(AB::Expr::from),
                    warp_index_digest: self.binding.warp_index_digest.map(AB::Expr::from),
                    schedule_digest: self.binding.schedule_digest.map(AB::Expr::from),
                    manifest_digest: local.manifest_digest.map(Into::into),
                    source_count: local.source_count.into(),
                    call_count: local.call_count.into(),
                    final_accumulator_digest: local.final_accumulator_digest.map(Into::into),
                    final_accumulator_root: local.final_accumulator_root.map(Into::into),
                },
                active.clone(),
            );
            self.terminal_receipt_bus.add_key_with_lookups(
                builder,
                ReducedSwirlLocalTerminalReceiptMessage {
                    protocol_digest: self.binding.protocol_digest.map(AB::Expr::from),
                    relation_digest: self.binding.relation_digest.map(AB::Expr::from),
                    terminal_index_digest: self.binding.terminal_index_digest.map(AB::Expr::from),
                    verifier_component_digest: self
                        .binding
                        .verifier_component_digest
                        .map(AB::Expr::from),
                    final_accumulator_digest: local.final_accumulator_digest.map(Into::into),
                    final_accumulator_root: core::array::from_fn(|limb| {
                        AB::Expr::from(local.final_accumulator_root[limb])
                            + if limb == 0 {
                                AB::Expr::from(self.terminal_root_delta)
                            } else {
                                AB::Expr::ZERO
                            }
                    }),
                },
                active,
            );
        }
    }

    fn check_terminal_join_logup(
        join: &ReducedSwirlTransitionTerminalJoinAir,
        trace: &RowMajorMatrix<F>,
        authority: TestTerminalJoinAuthorityAir,
    ) {
        let airs: Vec<AirRef<NativeSC>> = vec![Arc::new(*join), Arc::new(authority)];
        let preprocessed_owned = airs
            .iter()
            .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
            .collect::<Vec<_>>();
        let preprocessed = preprocessed_owned
            .iter()
            .map(|matrix| matrix.as_ref().map(RowMajorMatrix::as_view))
            .collect::<Vec<_>>();
        let interactions = airs
            .iter()
            .map(|air| symbolic_interactions(air.as_ref()))
            .collect::<Vec<_>>();
        let views = vec![vec![trace.as_view()], vec![trace.as_view()]];
        check_logup(
            &airs.iter().map(|air| air.name()).collect::<Vec<_>>(),
            &interactions,
            &preprocessed,
            &views,
            &[Vec::new(), Vec::new()],
        );
    }

    #[test]
    fn terminal_join_requires_decide_and_vacc_to_open_the_same_accumulator() {
        let (_, finalizer_binding) = test_finalizer();
        let record = record(finalizer_binding);
        let binding = ReducedSwirlTransitionTerminalJoinBinding {
            protocol_digest: finalizer_binding.warp_protocol_digest,
            relation_digest: finalizer_binding.relation_digest,
            warp_index_digest: finalizer_binding.warp_index_digest,
            schedule_digest: finalizer_binding.schedule_digest,
            terminal_index_digest: digest(400),
            verifier_component_digest: digest(420),
        };
        let chain_receipt_bus = ReducedSwirlVaccChainReceiptBus::new(30);
        let terminal_receipt_bus = ReducedSwirlLocalTerminalReceiptBus::new(31);
        let join = ReducedSwirlTransitionTerminalJoinAir::new(
            binding,
            chain_receipt_bus,
            terminal_receipt_bus,
        )
        .unwrap();
        let trace = join.generate_trace(&record).unwrap();
        check_constraints::<_, NativeSC>(
            &join,
            "ReducedSwirlTransitionTerminalJoinAir",
            &None,
            &[trace.as_view()],
            &[],
        );
        let authority = TestTerminalJoinAuthorityAir {
            binding,
            chain_receipt_bus,
            terminal_receipt_bus,
            terminal_root_delta: F::ZERO,
        };
        check_terminal_join_logup(&join, &trace, authority);

        let rejected = catch_unwind(AssertUnwindSafe(|| {
            check_terminal_join_logup(
                &join,
                &trace,
                TestTerminalJoinAuthorityAir {
                    terminal_root_delta: F::ONE,
                    ..authority
                },
            );
        }));
        assert!(
            rejected.is_err(),
            "terminal Decide root was not joined to the VACC footer root"
        );
    }
}

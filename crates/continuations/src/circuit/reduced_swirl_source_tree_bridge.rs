//! Complete one-child verifier and detached-VACC bridge for a reduced-SWIRL source tree.
//!
//! The bridge accepts exactly one proof under a setup-fixed recursive root VK.
//! Every child public-value coordinate and the child VK commitment are consumed
//! from the complete [`VerifierSubCircuit`]. The authenticated synthetic
//! [`VmPvs`] are then opened only with the source summary emitted by detached
//! VACC. No host success bit, point-opening PESAT, or unverified source-tree
//! digest is accepted at this boundary.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_recursion_circuit::{
    bus::{
        CachedCommitBus, CachedCommitBusMessage, Poseidon2CompressBus, Poseidon2CompressMessage,
        PreHashBus, PreHashMessage, PublicValuesBus, PublicValuesBusMessage,
    },
    native_warp::{
        ReducedSwirlVaccSourceSummaryBus, ReducedSwirlVaccSourceSummaryMessage,
        REDUCED_SWIRL_VACC_SOURCE_LEAF_CAPACITY,
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
    p3_field::{Field, PrimeCharacteristicRing, PrimeField32},
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
    reduced_swirl_source_leaf::{
        reduced_swirl_source_boundary_metadata, reduced_swirl_source_boundary_metadata_expr,
        reduced_swirl_source_chain_genesis, REDUCED_SWIRL_SOURCE_LEAF_CAPACITY,
    },
    reduced_swirl_warp::{ReducedSwirlSourceReceiptBus, ReducedSwirlSourceReceiptMessage},
    Circuit,
};

const SOURCE_TREE_PROOF_INDEX: usize = 0;

/// Setup authority propagated by the fixed OpenVM source-tree recursion.
///
/// `recursive_vk_commit` is both the VK used by the complete one-child
/// verifier and the key propagated by a recursive-self child. A depth-one
/// root has the latter field unset, as required by the ordinary OpenVM PVS
/// convention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReducedSwirlSourceTreeTrustedVkCommits {
    /// Commitment to the custom transition-leaf VK.  It is the application
    /// key of this independent recursion ladder; the transition-leaf AIR in
    /// turn fixes the original execution/SWIRL application VK.
    pub app_vk_commit: VkCommit<F>,
    /// Commitment to the first ordinary leaf-prefix VK.
    pub leaf_vk_commit: VkCommit<F>,
    /// Commitment to the InternalForLeaf VK, propagated by the first proof
    /// under the stable recursive relation.
    pub internal_for_leaf_vk_commit: VkCommit<F>,
    /// Commitment to the stable recursive source-tree VK. This is always the
    /// VK which verifies the supplied root proof; it occurs inside child PVS
    /// only when `recursion_depth > 1`.
    pub recursive_vk_commit: VkCommit<F>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReducedSwirlSourceTreeBridgeBinding {
    pub protocol_digest: Digest,
    pub source_leaf_capacity: u32,
    pub trusted_vk_commits: ReducedSwirlSourceTreeTrustedVkCommits,
}

impl ReducedSwirlSourceTreeBridgeBinding {
    pub fn validate(&self) -> Result<(), ReducedSwirlSourceTreeBridgeError> {
        if self.source_leaf_capacity as usize != REDUCED_SWIRL_SOURCE_LEAF_CAPACITY
            || REDUCED_SWIRL_VACC_SOURCE_LEAF_CAPACITY != REDUCED_SWIRL_SOURCE_LEAF_CAPACITY
        {
            return Err(ReducedSwirlSourceTreeBridgeError::CapacityMismatch);
        }
        if is_zero_digest(&self.protocol_digest)
            || [
                self.trusted_vk_commits.app_vk_commit,
                self.trusted_vk_commits.leaf_vk_commit,
                self.trusted_vk_commits.internal_for_leaf_vk_commit,
                self.trusted_vk_commits.recursive_vk_commit,
            ]
            .into_iter()
            .any(is_unset_vk_commit)
        {
            return Err(ReducedSwirlSourceTreeBridgeError::UnsetBinding);
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReducedSwirlSourceTreeBridgeError {
    CapacityMismatch,
    UnsetBinding,
    ChildVkPreHash,
    ChildPublicValueProfile,
    ChildCachedProfile,
    ReusedBusIndex,
    SymbolicDagAuthority,
    PublicValueShape,
    OrdinaryPvsMismatch,
    SourceSummaryMismatch,
    SyntheticVmPvsMismatch,
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub struct ReducedSwirlSourceTreeBridgeCols<T> {
    pub active: T,
    pub depth_inv: T,
    pub depth_minus_one_inv: T,
    pub is_depth_one: T,
    pub source_count_inv: T,
    pub child_verifier_pvs: VerifierBasePvs<T>,
    pub child_vm_pvs: VmPvs<T>,
    pub summary: ReducedSwirlVaccSourceSummaryMessage<T>,
    pub initial_protocol: [T; DIGEST_SIZE],
    pub initial_state: [T; DIGEST_SIZE],
    pub initial_boundary: [T; DIGEST_SIZE],
    pub final_protocol: [T; DIGEST_SIZE],
    pub final_state: [T; DIGEST_SIZE],
    pub final_boundary: [T; DIGEST_SIZE],
}

#[derive(Clone, Debug)]
pub struct ReducedSwirlSourceTreeBridgeTrace {
    pub matrix: RowMajorMatrix<F>,
    pub compression_inputs: Vec<[F; 2 * DIGEST_SIZE]>,
    pub receipt: ReducedSwirlSourceReceiptMessage<F>,
}

#[derive(Clone, Debug)]
pub struct ReducedSwirlSourceTreeBridgeAir {
    binding: ReducedSwirlSourceTreeBridgeBinding,
    child_air_count: usize,
    public_values_bus: PublicValuesBus,
    cached_commit_bus: CachedCommitBus,
    pre_hash_bus: PreHashBus,
    compress_bus: Poseidon2CompressBus,
    summary_bus: ReducedSwirlVaccSourceSummaryBus,
    summary_bus_index: BusIndex,
    receipt_bus: ReducedSwirlSourceReceiptBus,
    genesis: Digest,
}

impl ReducedSwirlSourceTreeBridgeAir {
    #[allow(clippy::too_many_arguments)]
    fn new(
        binding: ReducedSwirlSourceTreeBridgeBinding,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        public_values_bus: PublicValuesBus,
        cached_commit_bus: CachedCommitBus,
        pre_hash_bus: PreHashBus,
        compress_bus: Poseidon2CompressBus,
        summary_bus: ReducedSwirlVaccSourceSummaryBus,
        summary_bus_index: BusIndex,
        receipt_bus: ReducedSwirlSourceReceiptBus,
    ) -> Result<Self, ReducedSwirlSourceTreeBridgeError> {
        binding.validate()?;
        validate_child_vk_profile(child_vk)?;
        if child_vk.pre_hash != binding.trusted_vk_commits.recursive_vk_commit.vk_pre_hash {
            return Err(ReducedSwirlSourceTreeBridgeError::ChildVkPreHash);
        }
        let used = [
            public_values_bus.index(),
            cached_commit_bus.index(),
            pre_hash_bus.index(),
            compress_bus.index(),
            summary_bus_index,
            receipt_bus.index(),
        ];
        if used
            .iter()
            .enumerate()
            .any(|(index, bus)| used[..index].contains(bus))
        {
            return Err(ReducedSwirlSourceTreeBridgeError::ReusedBusIndex);
        }
        Ok(Self {
            binding,
            child_air_count: child_vk.inner.per_air.len(),
            public_values_bus,
            cached_commit_bus,
            pre_hash_bus,
            compress_bus,
            summary_bus,
            summary_bus_index,
            receipt_bus,
            // The leaf's domain preimage is intentionally private. Computing
            // this setup constant through the public helper prevents a second
            // copy of that protocol tag from drifting.
            genesis: reduced_swirl_source_chain_genesis(binding.protocol_digest),
        })
    }

    #[must_use]
    pub const fn binding(&self) -> ReducedSwirlSourceTreeBridgeBinding {
        self.binding
    }

    #[must_use]
    pub const fn trusted_vk_commits(&self) -> ReducedSwirlSourceTreeTrustedVkCommits {
        self.binding.trusted_vk_commits
    }

    #[must_use]
    pub const fn receipt_bus(&self) -> ReducedSwirlSourceReceiptBus {
        self.receipt_bus
    }

    #[must_use]
    pub const fn summary_bus(&self) -> ReducedSwirlVaccSourceSummaryBus {
        self.summary_bus
    }

    #[must_use]
    pub const fn summary_bus_index(&self) -> BusIndex {
        self.summary_bus_index
    }

    pub fn generate_trace(
        &self,
        child_public_values: &[Vec<F>],
        summary: ReducedSwirlVaccSourceSummaryMessage<F>,
    ) -> Result<ReducedSwirlSourceTreeBridgeTrace, ReducedSwirlSourceTreeBridgeError> {
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
            return Err(ReducedSwirlSourceTreeBridgeError::PublicValueShape);
        }
        let verifier_fields = &child_public_values[VERIFIER_PVS_AIR_ID];
        let vm_fields = &child_public_values[VM_PVS_AIR_ID];
        let verifier_pvs: &VerifierBasePvs<F> = verifier_fields.as_slice().borrow();
        let vm_pvs: &VmPvs<F> = vm_fields.as_slice().borrow();
        self.validate_verifier_pvs(verifier_pvs)?;
        self.validate_summary(&summary)?;

        let source_count = summary.source_count.as_canonical_u32();
        let mut compression_inputs = Vec::with_capacity(6);
        let initial_protocol = record_compression(
            summary.protocol_digest,
            reduced_swirl_source_boundary_metadata(0, summary.initial_pc),
            &mut compression_inputs,
        );
        let initial_state =
            record_compression(summary.initial_root, self.genesis, &mut compression_inputs);
        let initial_boundary =
            record_compression(initial_protocol, initial_state, &mut compression_inputs);
        let final_protocol = record_compression(
            summary.protocol_digest,
            reduced_swirl_source_boundary_metadata(source_count, summary.final_pc),
            &mut compression_inputs,
        );
        let final_state = record_compression(
            summary.final_root,
            summary.chunk_chain_endpoint,
            &mut compression_inputs,
        );
        let final_boundary =
            record_compression(final_protocol, final_state, &mut compression_inputs);
        let expected_vm_pvs = VmPvs {
            program_commit: summary.program_commitment,
            initial_pc: F::ZERO,
            final_pc: summary.source_count,
            exit_code: F::ZERO,
            is_terminate: F::ONE,
            initial_root: initial_boundary,
            final_root: final_boundary,
        };
        if vm_pvs.as_slice() != expected_vm_pvs.as_slice() {
            return Err(ReducedSwirlSourceTreeBridgeError::SyntheticVmPvsMismatch);
        }

        let width = self.width();
        let mut values = F::zero_vec(2 * width);
        let local: &mut ReducedSwirlSourceTreeBridgeCols<F> = values[..width].borrow_mut();
        local.active = F::ONE;
        let depth = verifier_pvs.recursion_depth;
        local.depth_inv = depth.inverse();
        local.is_depth_one = F::from_bool(depth == F::ONE);
        local.depth_minus_one_inv = if depth == F::ONE {
            F::ZERO
        } else {
            (depth - F::ONE).inverse()
        };
        local.source_count_inv = summary.source_count.inverse();
        local.child_verifier_pvs = *verifier_pvs;
        local.child_vm_pvs = *vm_pvs;
        local.summary = summary.clone();
        local.initial_protocol = initial_protocol;
        local.initial_state = initial_state;
        local.initial_boundary = initial_boundary;
        local.final_protocol = final_protocol;
        local.final_state = final_state;
        local.final_boundary = final_boundary;

        Ok(ReducedSwirlSourceTreeBridgeTrace {
            matrix: RowMajorMatrix::new(values, width),
            compression_inputs,
            receipt: source_receipt(&summary),
        })
    }

    fn validate_verifier_pvs(
        &self,
        pvs: &VerifierBasePvs<F>,
    ) -> Result<(), ReducedSwirlSourceTreeBridgeError> {
        let trusted = self.binding.trusted_vk_commits;
        let recursive_ok = if pvs.recursion_depth == F::ONE {
            is_unset_vk_commit(pvs.internal_recursive_vk_commit)
        } else {
            pvs.internal_recursive_vk_commit == trusted.recursive_vk_commit
        };
        if pvs.internal_flag != F::TWO
            || pvs.recursion_depth == F::ZERO
            || pvs.app_vk_commit != trusted.app_vk_commit
            || pvs.leaf_vk_commit != trusted.leaf_vk_commit
            || pvs.internal_for_leaf_vk_commit != trusted.internal_for_leaf_vk_commit
            || !recursive_ok
        {
            return Err(ReducedSwirlSourceTreeBridgeError::OrdinaryPvsMismatch);
        }
        Ok(())
    }

    fn validate_summary(
        &self,
        summary: &ReducedSwirlVaccSourceSummaryMessage<F>,
    ) -> Result<(), ReducedSwirlSourceTreeBridgeError> {
        if summary.protocol_digest != self.binding.protocol_digest
            || summary.source_count == F::ZERO
            || summary.source_leaf_capacity != F::from_u32(self.binding.source_leaf_capacity)
            || is_zero_digest(&summary.manifest_digest)
            || is_zero_digest(&summary.program_commitment)
            || summary.exit_code != F::ZERO
            || summary.is_terminate != F::ONE
        {
            return Err(ReducedSwirlSourceTreeBridgeError::SourceSummaryMismatch);
        }
        Ok(())
    }

    fn lookup_compression<AB: AirBuilder<F = F> + InteractionBuilder>(
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
}

impl BaseAir<F> for ReducedSwirlSourceTreeBridgeAir {
    fn width(&self) -> usize {
        core::mem::size_of::<ReducedSwirlSourceTreeBridgeCols<u8>>()
    }
}

impl BaseAirWithPublicValues<F> for ReducedSwirlSourceTreeBridgeAir {}
impl PartitionedBaseAir<F> for ReducedSwirlSourceTreeBridgeAir {}

impl<AB> Air<AB> for ReducedSwirlSourceTreeBridgeAir
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("source-tree bridge row");
        let next_row = main.row_slice(1).expect("source-tree bridge padding row");
        let local: &ReducedSwirlSourceTreeBridgeCols<AB::Var> = (*row).borrow();
        let next: &ReducedSwirlSourceTreeBridgeCols<AB::Var> = (*next_row).borrow();
        let active = AB::Expr::from(local.active);
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_last_row().assert_zero(local.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.active);
        for value in next.as_slice().iter().skip(1) {
            // On the last row the cyclic `next` view is the active first row.
            // Only the active row may require its following padding row to be
            // zero.
            builder.when(local.active).assert_zero(*value);
        }

        let depth = AB::Expr::from(local.child_verifier_pvs.recursion_depth);
        let depth_minus_one = depth.clone() - AB::Expr::ONE;
        builder.assert_bool(local.is_depth_one);
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
            .assert_one(AB::Expr::from(local.summary.source_count) * local.source_count_inv);

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
            trusted.leaf_vk_commit,
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
                AB::Expr::from_usize(SOURCE_TREE_PROOF_INDEX),
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
                AB::Expr::from_usize(SOURCE_TREE_PROOF_INDEX),
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
            AB::Expr::from_usize(SOURCE_TREE_PROOF_INDEX),
            CachedCommitBusMessage {
                air_idx: AB::Expr::from_usize(CONSTRAINT_EVAL_AIR_ID),
                cached_idx: AB::Expr::from_usize(CONSTRAINT_EVAL_CACHED_INDEX),
                global_cached_idx: AB::Expr::ZERO,
                // A depth-one root verifies an internal-for-leaf proof, while
                // deeper roots verify the self-recursive relation.  This is
                // the same conditional lineage already enforced for the
                // child verifier public values above.
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
            AB::Expr::from_usize(SOURCE_TREE_PROOF_INDEX),
            PreHashMessage {
                vk_pre_hash: trusted.recursive_vk_commit.vk_pre_hash.map(AB::Expr::from),
            },
            active.clone(),
        );

        let summary = ReducedSwirlVaccSourceSummaryMessage {
            protocol_digest: local.summary.protocol_digest.map(Into::into),
            manifest_digest: local.summary.manifest_digest.map(Into::into),
            source_count: local.summary.source_count.into(),
            source_leaf_capacity: local.summary.source_leaf_capacity.into(),
            program_commitment: local.summary.program_commitment.map(Into::into),
            initial_pc: local.summary.initial_pc.into(),
            initial_root: local.summary.initial_root.map(Into::into),
            final_pc: local.summary.final_pc.into(),
            final_root: local.summary.final_root.map(Into::into),
            exit_code: local.summary.exit_code.into(),
            is_terminate: local.summary.is_terminate.into(),
            chunk_chain_endpoint: local.summary.chunk_chain_endpoint.map(Into::into),
        };
        self.summary_bus
            .lookup_key(builder, summary.clone(), active.clone());
        for limb in 0..DIGEST_SIZE {
            builder.when(active.clone()).assert_eq(
                local.summary.protocol_digest[limb],
                AB::Expr::from(self.binding.protocol_digest[limb]),
            );
        }
        builder.when(active.clone()).assert_eq(
            local.summary.source_leaf_capacity,
            AB::Expr::from_u32(self.binding.source_leaf_capacity),
        );
        builder
            .when(active.clone())
            .assert_zero(local.summary.exit_code);
        builder
            .when(active.clone())
            .assert_one(local.summary.is_terminate);

        self.lookup_compression(
            builder,
            local.summary.protocol_digest.map(Into::into),
            reduced_swirl_source_boundary_metadata_expr::<AB>(
                AB::Expr::ZERO,
                local.summary.initial_pc.into(),
            ),
            local.initial_protocol.map(Into::into),
            active.clone(),
        );
        self.lookup_compression(
            builder,
            local.summary.initial_root.map(Into::into),
            self.genesis.map(AB::Expr::from),
            local.initial_state.map(Into::into),
            active.clone(),
        );
        self.lookup_compression(
            builder,
            local.initial_protocol.map(Into::into),
            local.initial_state.map(Into::into),
            local.initial_boundary.map(Into::into),
            active.clone(),
        );
        self.lookup_compression(
            builder,
            local.summary.protocol_digest.map(Into::into),
            reduced_swirl_source_boundary_metadata_expr::<AB>(
                local.summary.source_count.into(),
                local.summary.final_pc.into(),
            ),
            local.final_protocol.map(Into::into),
            active.clone(),
        );
        self.lookup_compression(
            builder,
            local.summary.final_root.map(Into::into),
            local.summary.chunk_chain_endpoint.map(Into::into),
            local.final_state.map(Into::into),
            active.clone(),
        );
        self.lookup_compression(
            builder,
            local.final_protocol.map(Into::into),
            local.final_state.map(Into::into),
            local.final_boundary.map(Into::into),
            active.clone(),
        );

        for limb in 0..DIGEST_SIZE {
            builder.when(active.clone()).assert_eq(
                local.child_vm_pvs.program_commit[limb],
                local.summary.program_commitment[limb],
            );
            builder.when(active.clone()).assert_eq(
                local.child_vm_pvs.initial_root[limb],
                local.initial_boundary[limb],
            );
            builder.when(active.clone()).assert_eq(
                local.child_vm_pvs.final_root[limb],
                local.final_boundary[limb],
            );
        }
        builder
            .when(active.clone())
            .assert_zero(local.child_vm_pvs.initial_pc);
        builder
            .when(active.clone())
            .assert_eq(local.child_vm_pvs.final_pc, local.summary.source_count);
        builder
            .when(active.clone())
            .assert_zero(local.child_vm_pvs.exit_code);
        builder
            .when(active.clone())
            .assert_one(local.child_vm_pvs.is_terminate);

        self.receipt_bus.add_key_with_lookups(
            builder,
            ReducedSwirlSourceReceiptMessage {
                protocol_digest: local.summary.protocol_digest.map(Into::into),
                manifest_digest: local.summary.manifest_digest.map(Into::into),
                source_offset: AB::Expr::ZERO,
                source_count: local.summary.source_count.into(),
                program_commitment: local.summary.program_commitment.map(Into::into),
                initial_pc: local.summary.initial_pc.into(),
                initial_root: local.summary.initial_root.map(Into::into),
                final_pc: local.summary.final_pc.into(),
                final_root: local.summary.final_root.map(Into::into),
                exit_code: local.summary.exit_code.into(),
                is_terminate: local.summary.is_terminate.into(),
            },
            active,
        );
    }
}

/// Circuit inventory used by the compact tail wrapper. The verifier is always
/// complete; changing it to a deferred tail would be a protocol violation.
#[derive(Clone)]
pub struct ReducedSwirlSourceTreeBridgeCircuit {
    child_vk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
    verifier: Arc<VerifierSubCircuit<1>>,
    bridge: Arc<ReducedSwirlSourceTreeBridgeAir>,
    next_bus_idx: BusIndex,
}

impl ReducedSwirlSourceTreeBridgeCircuit {
    pub fn new(
        child_vk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
        binding: ReducedSwirlSourceTreeBridgeBinding,
        summary_bus_index: BusIndex,
        receipt_bus: ReducedSwirlSourceReceiptBus,
        bus_idx_manager: BusIndexManager,
    ) -> Result<Self, ReducedSwirlSourceTreeBridgeError> {
        if summary_bus_index == receipt_bus.index()
            || summary_bus_index >= bus_idx_manager.next_bus_idx()
            || receipt_bus.index() >= bus_idx_manager.next_bus_idx()
        {
            return Err(ReducedSwirlSourceTreeBridgeError::ReusedBusIndex);
        }
        let summary_bus = ReducedSwirlVaccSourceSummaryBus::new(summary_bus_index);
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
            .ok_or(ReducedSwirlSourceTreeBridgeError::SymbolicDagAuthority)?
            .commit;
        verifier
            .bind_fixed_dag_commit(expected_dag_commit)
            .map_err(|_| ReducedSwirlSourceTreeBridgeError::SymbolicDagAuthority)?;
        if verifier.max_num_proofs() != 1 {
            return Err(ReducedSwirlSourceTreeBridgeError::ChildPublicValueProfile);
        }
        let buses = verifier.bus_inventory();
        let bridge = ReducedSwirlSourceTreeBridgeAir::new(
            binding,
            child_vk.as_ref(),
            buses.public_values_bus,
            buses.cached_commit_bus,
            buses.pre_hash_bus,
            buses.poseidon2_compress_bus,
            summary_bus,
            summary_bus_index,
            receipt_bus,
        )?;
        let next_bus_idx = verifier.next_bus_idx();
        Ok(Self {
            child_vk,
            verifier: Arc::new(verifier),
            bridge: Arc::new(bridge),
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
    pub fn bridge(&self) -> &Arc<ReducedSwirlSourceTreeBridgeAir> {
        &self.bridge
    }

    #[must_use]
    pub const fn next_bus_idx(&self) -> BusIndex {
        self.next_bus_idx
    }
}

impl<SC: StarkProtocolConfig<F = F>> Circuit<SC> for ReducedSwirlSourceTreeBridgeCircuit {
    fn airs(&self) -> Vec<AirRef<SC>> {
        core::iter::once(self.bridge.clone() as AirRef<SC>)
            .chain(self.verifier.airs::<SC>())
            .collect()
    }
}

fn validate_child_vk_profile(
    child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
) -> Result<(), ReducedSwirlSourceTreeBridgeError> {
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
        return Err(ReducedSwirlSourceTreeBridgeError::ChildPublicValueProfile);
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
        return Err(ReducedSwirlSourceTreeBridgeError::ChildCachedProfile);
    }
    Ok(())
}

fn source_receipt(
    summary: &ReducedSwirlVaccSourceSummaryMessage<F>,
) -> ReducedSwirlSourceReceiptMessage<F> {
    ReducedSwirlSourceReceiptMessage {
        protocol_digest: summary.protocol_digest,
        manifest_digest: summary.manifest_digest,
        source_offset: F::ZERO,
        source_count: summary.source_count,
        program_commitment: summary.program_commitment,
        initial_pc: summary.initial_pc,
        initial_root: summary.initial_root,
        final_pc: summary.final_pc,
        final_root: summary.final_root,
        exit_code: summary.exit_code,
        is_terminate: summary.is_terminate,
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
    for limb in 0..DIGEST_SIZE {
        builder.when(enabled.clone()).assert_eq(
            actual.cached_commit[limb],
            AB::Expr::from(expected.cached_commit[limb]),
        );
        builder.when(enabled.clone()).assert_eq(
            actual.vk_pre_hash[limb],
            AB::Expr::from(expected.vk_pre_hash[limb]),
        );
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
        air_builders::PartitionedAirBuilder,
        p3_air::{Air, BaseAir},
        p3_field::PrimeCharacteristicRing,
        AirRef, BaseAirWithPublicValues, PartitionedBaseAir, StarkEngine,
    };
    use openvm_stark_sdk::config::{
        baby_bear_poseidon2::{BabyBearPoseidon2CpuEngine, DuplexSponge},
        native_warp_history_params_with_100_bits_security,
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

    fn binding_with_root_pre_hash(root_pre_hash: Digest) -> ReducedSwirlSourceTreeBridgeBinding {
        let mut recursive_vk_commit = vk_commit(80);
        recursive_vk_commit.vk_pre_hash = root_pre_hash;
        ReducedSwirlSourceTreeBridgeBinding {
            protocol_digest: digest(1),
            source_leaf_capacity: REDUCED_SWIRL_SOURCE_LEAF_CAPACITY as u32,
            trusted_vk_commits: ReducedSwirlSourceTreeTrustedVkCommits {
                app_vk_commit: vk_commit(20),
                leaf_vk_commit: vk_commit(40),
                internal_for_leaf_vk_commit: vk_commit(60),
                recursive_vk_commit,
            },
        }
    }

    fn verifier_pvs(
        binding: ReducedSwirlSourceTreeBridgeBinding,
        depth: u32,
    ) -> VerifierBasePvs<F> {
        let trusted = binding.trusted_vk_commits;
        VerifierBasePvs {
            internal_flag: F::TWO,
            app_vk_commit: trusted.app_vk_commit,
            leaf_vk_commit: trusted.leaf_vk_commit,
            internal_for_leaf_vk_commit: trusted.internal_for_leaf_vk_commit,
            recursion_depth: F::from_u32(depth),
            internal_recursive_vk_commit: if depth == 1 {
                VkCommit {
                    cached_commit: [F::ZERO; DIGEST_SIZE],
                    vk_pre_hash: [F::ZERO; DIGEST_SIZE],
                }
            } else {
                trusted.recursive_vk_commit
            },
        }
    }

    fn summary(
        binding: ReducedSwirlSourceTreeBridgeBinding,
    ) -> ReducedSwirlVaccSourceSummaryMessage<F> {
        ReducedSwirlVaccSourceSummaryMessage {
            protocol_digest: binding.protocol_digest,
            manifest_digest: digest(300),
            source_count: F::from_u32(129),
            source_leaf_capacity: F::from_usize(REDUCED_SWIRL_SOURCE_LEAF_CAPACITY),
            program_commitment: digest(400),
            initial_pc: F::from_u32(7),
            initial_root: digest(500),
            final_pc: F::from_u32(99),
            final_root: digest(600),
            exit_code: F::ZERO,
            is_terminate: F::ONE,
            chunk_chain_endpoint: digest(700),
        }
    }

    fn synthetic_vm(summary: &ReducedSwirlVaccSourceSummaryMessage<F>) -> VmPvs<F> {
        let genesis = reduced_swirl_source_chain_genesis(summary.protocol_digest);
        let initial_protocol = poseidon2_compress_with_capacity(
            summary.protocol_digest,
            reduced_swirl_source_boundary_metadata(0, summary.initial_pc),
        )
        .0;
        let initial_state = poseidon2_compress_with_capacity(summary.initial_root, genesis).0;
        let final_protocol = poseidon2_compress_with_capacity(
            summary.protocol_digest,
            reduced_swirl_source_boundary_metadata(
                summary.source_count.as_canonical_u32(),
                summary.final_pc,
            ),
        )
        .0;
        let final_state =
            poseidon2_compress_with_capacity(summary.final_root, summary.chunk_chain_endpoint).0;
        VmPvs {
            program_commit: summary.program_commitment,
            initial_pc: F::ZERO,
            final_pc: summary.source_count,
            exit_code: F::ZERO,
            is_terminate: F::ONE,
            initial_root: poseidon2_compress_with_capacity(initial_protocol, initial_state).0,
            final_root: poseidon2_compress_with_capacity(final_protocol, final_state).0,
        }
    }

    #[test]
    fn binding_rejects_capacity_and_vk_mutations() {
        let mut bad = binding_with_root_pre_hash(digest(900));
        bad.source_leaf_capacity /= 2;
        assert_eq!(
            bad.validate(),
            Err(ReducedSwirlSourceTreeBridgeError::CapacityMismatch)
        );
        let mut bad = binding_with_root_pre_hash(digest(900));
        bad.trusted_vk_commits.recursive_vk_commit = VkCommit {
            cached_commit: [F::ZERO; DIGEST_SIZE],
            vk_pre_hash: [F::ZERO; DIGEST_SIZE],
        };
        assert_eq!(
            bad.validate(),
            Err(ReducedSwirlSourceTreeBridgeError::UnsetBinding)
        );

        let vk = test_vk();
        let mismatched = binding_with_root_pre_hash(digest(9_000));
        assert_eq!(
            ReducedSwirlSourceTreeBridgeAir::new(
                mismatched,
                &vk,
                PublicValuesBus::new(10),
                CachedCommitBus::new(11),
                PreHashBus::new(12),
                Poseidon2CompressBus::new(13),
                ReducedSwirlVaccSourceSummaryBus::new(14),
                14,
                ReducedSwirlSourceReceiptBus::new(15),
            )
            .err(),
            Some(ReducedSwirlSourceTreeBridgeError::ChildVkPreHash)
        );
    }

    #[test]
    fn synthetic_boundaries_bind_summary_root_chain_and_status() {
        let summary = summary(binding_with_root_pre_hash(digest(900)));
        let expected = synthetic_vm(&summary);
        let mutations: [fn(&mut ReducedSwirlVaccSourceSummaryMessage<F>); 5] = [
            |value: &mut ReducedSwirlVaccSourceSummaryMessage<F>| value.initial_root[0] += F::ONE,
            |value: &mut ReducedSwirlVaccSourceSummaryMessage<F>| value.final_root[0] += F::ONE,
            |value: &mut ReducedSwirlVaccSourceSummaryMessage<F>| {
                value.chunk_chain_endpoint[0] += F::ONE
            },
            |value: &mut ReducedSwirlVaccSourceSummaryMessage<F>| {
                value.program_commitment[0] += F::ONE
            },
            |value: &mut ReducedSwirlVaccSourceSummaryMessage<F>| value.source_count += F::ONE,
        ];
        for mutation in mutations {
            let mut changed = summary.clone();
            mutation(&mut changed);
            assert_ne!(synthetic_vm(&changed).as_slice(), expected.as_slice());
        }
        let mut bad_status = summary;
        bad_status.is_terminate = F::ZERO;
        assert_ne!(bad_status.is_terminate, F::ONE);
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

    impl<AB: PartitionedAirBuilder<F = F>> Air<AB> for TestAir {
        fn eval(&self, builder: &mut AB) {
            let value = {
                let main = builder.common_main();
                let row = main.row_slice(0).expect("test AIR row");
                row[0].clone()
            };
            builder.assert_zero(value);
        }
    }

    fn test_vk() -> MultiStarkVerifyingKey<BabyBearPoseidon2Config> {
        let params = native_warp_history_params_with_100_bits_security();
        let engine: BabyBearPoseidon2CpuEngine<DuplexSponge> =
            BabyBearPoseidon2CpuEngine::new(params);
        let airs: Vec<AirRef<BabyBearPoseidon2Config>> = (0..=CONSTRAINT_EVAL_AIR_ID)
            .map(|air_id| {
                Arc::new(TestAir {
                    pvs: match air_id {
                        VERIFIER_PVS_AIR_ID => VerifierBasePvs::<u8>::width(),
                        VM_PVS_AIR_ID => VmPvs::<u8>::width(),
                        _ => 0,
                    },
                    cached: air_id == CONSTRAINT_EVAL_AIR_ID,
                }) as AirRef<BabyBearPoseidon2Config>
            })
            .collect();
        engine.keygen(&airs).1
    }

    fn test_bridge() -> (
        ReducedSwirlSourceTreeBridgeAir,
        ReducedSwirlSourceTreeBridgeBinding,
    ) {
        let vk = test_vk();
        let binding = binding_with_root_pre_hash(vk.pre_hash);
        let bridge = ReducedSwirlSourceTreeBridgeAir::new(
            binding,
            &vk,
            PublicValuesBus::new(10),
            CachedCommitBus::new(11),
            PreHashBus::new(12),
            Poseidon2CompressBus::new(13),
            ReducedSwirlVaccSourceSummaryBus::new(14),
            14,
            ReducedSwirlSourceReceiptBus::new(15),
        )
        .expect("valid source-tree bridge");
        (bridge, binding)
    }

    fn public_values(
        binding: ReducedSwirlSourceTreeBridgeBinding,
        summary: &ReducedSwirlVaccSourceSummaryMessage<F>,
    ) -> Vec<Vec<F>> {
        let mut values = vec![Vec::new(); CONSTRAINT_EVAL_AIR_ID + 1];
        values[VERIFIER_PVS_AIR_ID] = verifier_pvs(binding, 2).as_slice().to_vec();
        values[VM_PVS_AIR_ID] = synthetic_vm(summary).as_slice().to_vec();
        values
    }

    #[test]
    fn bridge_accepts_exact_tree_and_summary_and_rejects_mutations() {
        let (bridge, binding) = test_bridge();
        let summary = summary(binding);
        let public_values = public_values(binding, &summary);
        let honest = bridge
            .generate_trace(&public_values, summary.clone())
            .expect("exact source tree and VACC summary");
        assert_eq!(honest.compression_inputs.len(), 6);
        assert_eq!(honest.receipt.source_offset, F::ZERO);

        let mut bad_vk = public_values.clone();
        bad_vk[VERIFIER_PVS_AIR_ID][1] += F::ONE;
        assert_eq!(
            bridge.generate_trace(&bad_vk, summary.clone()).err(),
            Some(ReducedSwirlSourceTreeBridgeError::OrdinaryPvsMismatch)
        );
        let mut bad_vm = public_values.clone();
        bad_vm[VM_PVS_AIR_ID][0] += F::ONE;
        assert_eq!(
            bridge.generate_trace(&bad_vm, summary.clone()).err(),
            Some(ReducedSwirlSourceTreeBridgeError::SyntheticVmPvsMismatch)
        );
        for mutation in [
            |value: &mut ReducedSwirlVaccSourceSummaryMessage<F>| value.initial_root[0] += F::ONE,
            |value: &mut ReducedSwirlVaccSourceSummaryMessage<F>| value.final_root[0] += F::ONE,
            |value: &mut ReducedSwirlVaccSourceSummaryMessage<F>| {
                value.chunk_chain_endpoint[0] += F::ONE
            },
        ] {
            let mut bad_summary = summary.clone();
            mutation(&mut bad_summary);
            assert_eq!(
                bridge.generate_trace(&public_values, bad_summary).err(),
                Some(ReducedSwirlSourceTreeBridgeError::SyntheticVmPvsMismatch)
            );
        }
        for mutation in [
            |value: &mut ReducedSwirlVaccSourceSummaryMessage<F>| value.exit_code = F::ONE,
            |value: &mut ReducedSwirlVaccSourceSummaryMessage<F>| value.is_terminate = F::ZERO,
            |value: &mut ReducedSwirlVaccSourceSummaryMessage<F>| {
                value.source_leaf_capacity -= F::ONE
            },
        ] {
            let mut bad_summary = summary.clone();
            mutation(&mut bad_summary);
            assert_eq!(
                bridge.generate_trace(&public_values, bad_summary).err(),
                Some(ReducedSwirlSourceTreeBridgeError::SourceSummaryMismatch)
            );
        }
    }

    #[test]
    fn malformed_public_value_shapes_are_rejected_without_panicking() {
        let (bridge, binding) = test_bridge();
        let summary = summary(binding);
        let result = catch_unwind(AssertUnwindSafe(|| {
            bridge.generate_trace(&[vec![F::ZERO]], summary)
        }));
        assert!(matches!(
            result,
            Ok(Err(ReducedSwirlSourceTreeBridgeError::PublicValueShape))
        ));
    }
}

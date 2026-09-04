//! Fixed-key recursive leaf for one reduced-SWIRL WARP transition.
//!
//! The leaf combines the existing deferred-SWIRL verifier prefix with the
//! genuine ordinary-WARP VACC verifier.  Their typed receipts are joined in
//! one AIR, so no host equality or detached certificate stands between the
//! source claim and the fresh input consumed by WARP.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit::{
    bus::{Poseidon2CompressBus, Poseidon2CompressMessage},
    native_warp::{ReducedSwirlVaccTransitionReceiptBus, ReducedSwirlVaccTransitionReceiptMessage},
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder},
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    AirRef, BaseAirWithPublicValues, PartitionedBaseAir, StarkProtocolConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, Digest, DIGEST_SIZE, F,
};
use openvm_verify_stark_host::pvs::{VerifierBasePvs, VkCommit, VmPvs};

use super::{
    reduced_swirl_warp::{
        ReducedSwirlExecutionBus, ReducedSwirlExecutionMessage, ReducedSwirlSourceReceiptBus,
        ReducedSwirlSourceReceiptMessage, ReducedSwirlWrapperVmPvsAir,
    },
    Circuit,
};

pub const REDUCED_SWIRL_TRANSITION_LEAF_PROTOCOL_VERSION: u32 = 2;
pub const REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY: usize = 8;
pub const REDUCED_SWIRL_TRANSITION_CHAIN_METADATA_TAG: u32 = 0x5254_4c01;
pub const REDUCED_SWIRL_TRANSITION_STATE_METADATA_TAG: u32 = 0x5254_4c02;
pub const REDUCED_SWIRL_TRANSITION_CHAIN_GENESIS_TAG: u32 = 0x5254_4c03;

#[derive(Clone, Debug, PartialEq)]
pub struct ReducedSwirlTransitionLeafBinding {
    pub protocol_version: u32,
    pub source_capacity: u32,
    pub source_component_digest: Digest,
    pub vacc_component_digest: Digest,
    pub recursive_app_vk_commit: VkCommit<F>,
}

impl ReducedSwirlTransitionLeafBinding {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.protocol_version != REDUCED_SWIRL_TRANSITION_LEAF_PROTOCOL_VERSION
            || self.source_capacity == 0
            || self.source_capacity as usize > REDUCED_SWIRL_TRANSITION_LEAF_CAPACITY
            || !self.source_capacity.is_power_of_two()
            || is_zero_digest(&self.source_component_digest)
            || is_zero_digest(&self.vacc_component_digest)
            || is_zero_digest(&self.recursive_app_vk_commit.cached_commit)
            || is_zero_digest(&self.recursive_app_vk_commit.vk_pre_hash)
        {
            return Err("reduced-SWIRL transition-leaf binding");
        }
        Ok(())
    }

    #[must_use]
    pub fn verifier_pvs(&self) -> VerifierBasePvs<F> {
        let unset = VkCommit {
            cached_commit: [F::ZERO; DIGEST_SIZE],
            vk_pre_hash: [F::ZERO; DIGEST_SIZE],
        };
        VerifierBasePvs {
            internal_flag: F::ZERO,
            app_vk_commit: self.recursive_app_vk_commit,
            leaf_vk_commit: unset,
            internal_for_leaf_vk_commit: unset,
            recursion_depth: F::ZERO,
            internal_recursive_vk_commit: unset,
        }
    }
}

/// Complete preimage of one transition-tree boundary root.  Internal OpenVM
/// recursion carries only the resulting digest, while the finalizer opens the
/// initial and terminal preimages and proves that they are the canonical
/// genesis and final WARP states.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlTransitionState {
    pub source_protocol_digest: Digest,
    pub warp_protocol_digest: Digest,
    pub relation_digest: Digest,
    pub warp_index_digest: Digest,
    pub schedule_digest: Digest,
    pub total_source_count: F,
    pub call_cursor: F,
    pub source_cursor: F,
    pub transcript_tidx: F,
    pub transcript_sample_count: F,
    pub transcript_state: [F; 16],
    pub accumulator_root: Digest,
    pub accumulator_digest: Digest,
    pub vm_pc: F,
    pub vm_root: Digest,
    pub manifest_chain: Digest,
    pub program_commitment: Digest,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ReducedSwirlTransitionLeafRecord {
    pub source: ReducedSwirlSourceReceiptMessage<F>,
    pub transition: ReducedSwirlVaccTransitionReceiptMessage<F>,
    pub chain_before: Digest,
}

pub struct ReducedSwirlTransitionLeafDerived {
    pub source_end: F,
    pub call_end: F,
    pub chunk: Digest,
    pub chain_after: Digest,
    pub genesis_chain: Digest,
    pub initial_boundary: Digest,
    pub final_boundary: Digest,
    pub vm_pvs: VmPvs<F>,
    pub compression_inputs: Vec<[F; 2 * DIGEST_SIZE]>,
    initial_hashes: StateHashWitness,
    final_hashes: StateHashWitness,
}

#[derive(Clone, Debug)]
pub(crate) struct StateHashWitness {
    pub(crate) setup_left: Digest,
    pub(crate) setup_right: Digest,
    pub(crate) setup: Digest,
    pub(crate) protocol: Digest,
    pub(crate) control: Digest,
    pub(crate) transcript: Digest,
    pub(crate) accumulator: Digest,
    pub(crate) transcript_accumulator: Digest,
    pub(crate) vm_chain: Digest,
    pub(crate) vm_program: Digest,
    pub(crate) left: Digest,
    pub(crate) boundary: Digest,
}

impl ReducedSwirlTransitionLeafRecord {
    #[must_use]
    pub fn initial_state(&self) -> ReducedSwirlTransitionState {
        ReducedSwirlTransitionState {
            source_protocol_digest: self.source.protocol_digest,
            warp_protocol_digest: self.transition.protocol_digest,
            relation_digest: self.transition.relation_digest,
            warp_index_digest: self.transition.warp_index_digest,
            schedule_digest: self.transition.schedule_digest,
            total_source_count: self.transition.total_source_count,
            call_cursor: self.transition.call_index,
            source_cursor: self.transition.source_start,
            transcript_tidx: self.transition.batch_start_tidx,
            transcript_sample_count: self.transition.start_sample_count,
            transcript_state: self.transition.start_state,
            accumulator_root: self.transition.prior_root,
            accumulator_digest: self.transition.prior_digest,
            vm_pc: self.source.initial_pc,
            vm_root: self.source.initial_root,
            manifest_chain: self.chain_before,
            program_commitment: self.source.program_commitment,
        }
    }

    #[must_use]
    pub fn final_state(&self, chain_after: Digest) -> ReducedSwirlTransitionState {
        ReducedSwirlTransitionState {
            source_protocol_digest: self.source.protocol_digest,
            warp_protocol_digest: self.transition.protocol_digest,
            relation_digest: self.transition.relation_digest,
            warp_index_digest: self.transition.warp_index_digest,
            schedule_digest: self.transition.schedule_digest,
            total_source_count: self.transition.total_source_count,
            call_cursor: self.transition.call_index + F::ONE,
            source_cursor: self.transition.source_start + self.transition.fresh_count,
            transcript_tidx: self.transition.vacc_end_tidx,
            transcript_sample_count: self.transition.end_sample_count,
            transcript_state: self.transition.end_state,
            accumulator_root: self.transition.output_root,
            accumulator_digest: self.transition.output_digest,
            vm_pc: self.source.final_pc,
            vm_root: self.source.final_root,
            manifest_chain: chain_after,
            program_commitment: self.source.program_commitment,
        }
    }

    pub fn derive(&self) -> Result<ReducedSwirlTransitionLeafDerived, &'static str> {
        if self.source.manifest_digest != self.transition.manifest_digest
            || self.source.source_offset != self.transition.source_start
            || self.source.source_count != self.transition.fresh_count
            || self.source.source_count == F::ZERO
            || is_zero_digest(&self.source.protocol_digest)
            || is_zero_digest(&self.transition.protocol_digest)
            || is_zero_digest(&self.source.program_commitment)
        {
            return Err("reduced-SWIRL transition-leaf receipt mismatch");
        }
        let source_end = self.source.source_offset + self.source.source_count;
        let call_end = self.transition.call_index + F::ONE;
        let mut inputs = Vec::with_capacity(24);
        let chunk = record_compression(
            reduced_swirl_transition_chain_metadata(&self.transition),
            self.source.manifest_digest,
            &mut inputs,
        );
        let chain_after = record_compression(self.chain_before, chunk, &mut inputs);
        let genesis_chain = if self.transition.prior_count == F::ZERO {
            let genesis = record_compression(
                self.source.protocol_digest,
                reduced_swirl_transition_chain_genesis_metadata(),
                &mut inputs,
            );
            if genesis != self.chain_before {
                return Err("reduced-SWIRL transition-leaf genesis chain");
            }
            genesis
        } else {
            [F::ZERO; DIGEST_SIZE]
        };
        let initial_hashes = state_digest(&self.initial_state(), &mut inputs);
        let final_hashes = state_digest(&self.final_state(chain_after), &mut inputs);
        let initial_boundary = initial_hashes.boundary;
        let final_boundary = final_hashes.boundary;
        let vm_pvs = VmPvs {
            program_commit: self.source.program_commitment,
            initial_pc: self.transition.call_index,
            final_pc: call_end,
            exit_code: self.source.exit_code,
            is_terminate: self.source.is_terminate,
            initial_root: initial_boundary,
            final_root: final_boundary,
        };
        Ok(ReducedSwirlTransitionLeafDerived {
            source_end,
            call_end,
            chunk,
            chain_after,
            genesis_chain,
            initial_boundary,
            final_boundary,
            vm_pvs,
            compression_inputs: inputs,
            initial_hashes,
            final_hashes,
        })
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ReducedSwirlTransitionLeafBoundaryCols<T> {
    pub active: T,
    pub source_protocol_digest: [T; DIGEST_SIZE],
    pub warp_protocol_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub warp_index_digest: [T; DIGEST_SIZE],
    pub schedule_digest: [T; DIGEST_SIZE],
    pub manifest_digest: [T; DIGEST_SIZE],
    pub total_source_count: T,
    pub call_index: T,
    pub source_start: T,
    pub fresh_count: T,
    pub prior_count: T,
    pub batch_start_tidx: T,
    pub vacc_start_tidx: T,
    pub vacc_end_tidx: T,
    pub start_sample_count: T,
    pub start_state: [T; 16],
    pub end_sample_count: T,
    pub end_state: [T; 16],
    pub prior_root: [T; DIGEST_SIZE],
    pub output_root: [T; DIGEST_SIZE],
    pub prior_digest: [T; DIGEST_SIZE],
    pub output_digest: [T; DIGEST_SIZE],
    pub is_final: T,
    pub program_commitment: [T; DIGEST_SIZE],
    pub initial_pc: T,
    pub initial_root: [T; DIGEST_SIZE],
    pub final_pc: T,
    pub final_root: [T; DIGEST_SIZE],
    pub exit_code: T,
    pub is_terminate: T,
    pub chain_before: [T; DIGEST_SIZE],
    pub chunk: [T; DIGEST_SIZE],
    pub chain_after: [T; DIGEST_SIZE],
    pub genesis_chain: [T; DIGEST_SIZE],
    pub initial_setup_left: [T; DIGEST_SIZE],
    pub initial_setup_right: [T; DIGEST_SIZE],
    pub initial_setup: [T; DIGEST_SIZE],
    pub initial_protocol: [T; DIGEST_SIZE],
    pub initial_control: [T; DIGEST_SIZE],
    pub initial_transcript: [T; DIGEST_SIZE],
    pub initial_accumulator: [T; DIGEST_SIZE],
    pub initial_transcript_accumulator: [T; DIGEST_SIZE],
    pub initial_vm_chain: [T; DIGEST_SIZE],
    pub initial_vm_program: [T; DIGEST_SIZE],
    pub initial_left: [T; DIGEST_SIZE],
    pub initial_boundary: [T; DIGEST_SIZE],
    pub final_setup_left: [T; DIGEST_SIZE],
    pub final_setup_right: [T; DIGEST_SIZE],
    pub final_setup: [T; DIGEST_SIZE],
    pub final_protocol: [T; DIGEST_SIZE],
    pub final_control: [T; DIGEST_SIZE],
    pub final_transcript: [T; DIGEST_SIZE],
    pub final_accumulator: [T; DIGEST_SIZE],
    pub final_transcript_accumulator: [T; DIGEST_SIZE],
    pub final_vm_chain: [T; DIGEST_SIZE],
    pub final_vm_program: [T; DIGEST_SIZE],
    pub final_left: [T; DIGEST_SIZE],
    pub final_boundary: [T; DIGEST_SIZE],
}

#[derive(Clone, ColumnsAir)]
#[columns_via(ReducedSwirlTransitionLeafBoundaryCols<u8>)]
pub struct ReducedSwirlTransitionLeafBoundaryAir {
    pub source_receipt_bus: ReducedSwirlSourceReceiptBus,
    pub transition_receipt_bus: ReducedSwirlVaccTransitionReceiptBus,
    pub execution_bus: ReducedSwirlExecutionBus,
    pub compress_bus: Poseidon2CompressBus,
}

impl BaseAir<F> for ReducedSwirlTransitionLeafBoundaryAir {
    fn width(&self) -> usize {
        ReducedSwirlTransitionLeafBoundaryCols::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for ReducedSwirlTransitionLeafBoundaryAir {}
impl PartitionedBaseAir<F> for ReducedSwirlTransitionLeafBoundaryAir {}

impl<AB> Air<AB> for ReducedSwirlTransitionLeafBoundaryAir
where
    AB: AirBuilder<F = F> + InteractionBuilder + PairBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("transition-leaf boundary row");
        let local: &ReducedSwirlTransitionLeafBoundaryCols<AB::Var> = (*row).borrow();
        let next = main.row_slice(1).expect("transition-leaf padding row");
        let next: &ReducedSwirlTransitionLeafBoundaryCols<AB::Var> = (*next).borrow();
        one_row_selector(builder, local.active, next.active);
        builder.assert_bool(local.is_final);
        builder.assert_bool(local.prior_count);
        for value in next.as_slice().iter().skip(1) {
            builder.when(local.active).assert_zero(*value);
        }

        self.source_receipt_bus.lookup_key(
            builder,
            ReducedSwirlSourceReceiptMessage {
                protocol_digest: local.source_protocol_digest.map(Into::into),
                manifest_digest: local.manifest_digest.map(Into::into),
                source_offset: local.source_start.into(),
                source_count: local.fresh_count.into(),
                program_commitment: local.program_commitment.map(Into::into),
                initial_pc: local.initial_pc.into(),
                initial_root: local.initial_root.map(Into::into),
                final_pc: local.final_pc.into(),
                final_root: local.final_root.map(Into::into),
                exit_code: local.exit_code.into(),
                is_terminate: local.is_terminate.into(),
            },
            local.active,
        );
        self.transition_receipt_bus.lookup_key(
            builder,
            ReducedSwirlVaccTransitionReceiptMessage {
                protocol_digest: local.warp_protocol_digest.map(Into::into),
                relation_digest: local.relation_digest.map(Into::into),
                warp_index_digest: local.warp_index_digest.map(Into::into),
                schedule_digest: local.schedule_digest.map(Into::into),
                total_source_count: local.total_source_count.into(),
                call_index: local.call_index.into(),
                source_start: local.source_start.into(),
                fresh_count: local.fresh_count.into(),
                prior_count: local.prior_count.into(),
                batch_start_tidx: local.batch_start_tidx.into(),
                vacc_start_tidx: local.vacc_start_tidx.into(),
                vacc_end_tidx: local.vacc_end_tidx.into(),
                start_sample_count: local.start_sample_count.into(),
                start_state: local.start_state.map(Into::into),
                end_sample_count: local.end_sample_count.into(),
                end_state: local.end_state.map(Into::into),
                prior_root: local.prior_root.map(Into::into),
                output_root: local.output_root.map(Into::into),
                prior_digest: local.prior_digest.map(Into::into),
                output_digest: local.output_digest.map(Into::into),
                manifest_digest: local.manifest_digest.map(Into::into),
                is_final: local.is_final.into(),
            },
            local.active,
        );

        let source_end = AB::Expr::from(local.source_start) + local.fresh_count;
        let call_end = AB::Expr::from(local.call_index) + AB::Expr::ONE;
        self.compress(
            builder,
            chain_metadata_expr::<AB>(local),
            local.manifest_digest.map(Into::into),
            local.chunk.map(Into::into),
            local.active.into(),
        );
        self.compress(
            builder,
            local.chain_before.map(Into::into),
            local.chunk.map(Into::into),
            local.chain_after.map(Into::into),
            local.active.into(),
        );
        self.compress(
            builder,
            local.source_protocol_digest.map(Into::into),
            genesis_metadata_expr::<AB>(),
            local.genesis_chain.map(Into::into),
            AB::Expr::from(local.active) * (AB::Expr::ONE - local.prior_count),
        );
        for limb in 0..DIGEST_SIZE {
            builder
                .when(AB::Expr::from(local.active) * (AB::Expr::ONE - local.prior_count))
                .assert_eq(local.chain_before[limb], local.genesis_chain[limb]);
        }

        self.eval_state(
            builder,
            local,
            local.call_index.into(),
            local.source_start.into(),
            local.batch_start_tidx.into(),
            local.start_sample_count.into(),
            local.start_state,
            local.prior_root,
            local.prior_digest,
            local.initial_pc.into(),
            local.initial_root,
            local.chain_before,
            local.initial_setup_left,
            local.initial_setup_right,
            local.initial_setup,
            local.initial_protocol,
            local.initial_control,
            local.initial_transcript,
            local.initial_accumulator,
            local.initial_transcript_accumulator,
            local.initial_vm_chain,
            local.initial_vm_program,
            local.initial_left,
            local.initial_boundary,
        );
        self.eval_state(
            builder,
            local,
            call_end.clone(),
            source_end.clone(),
            local.vacc_end_tidx.into(),
            local.end_sample_count.into(),
            local.end_state,
            local.output_root,
            local.output_digest,
            local.final_pc.into(),
            local.final_root,
            local.chain_after,
            local.final_setup_left,
            local.final_setup_right,
            local.final_setup,
            local.final_protocol,
            local.final_control,
            local.final_transcript,
            local.final_accumulator,
            local.final_transcript_accumulator,
            local.final_vm_chain,
            local.final_vm_program,
            local.final_left,
            local.final_boundary,
        );
        self.execution_bus.lookup_key(
            builder,
            ReducedSwirlExecutionMessage {
                vm_pvs: VmPvs {
                    program_commit: local.program_commitment.map(Into::into),
                    initial_pc: local.call_index.into(),
                    final_pc: call_end,
                    exit_code: local.exit_code.into(),
                    is_terminate: local.is_terminate.into(),
                    initial_root: local.initial_boundary.map(Into::into),
                    final_root: local.final_boundary.map(Into::into),
                },
            },
            local.active,
        );
    }
}

impl ReducedSwirlTransitionLeafBoundaryAir {
    #[allow(clippy::too_many_arguments)]
    fn eval_state<AB>(
        &self,
        builder: &mut AB,
        local: &ReducedSwirlTransitionLeafBoundaryCols<AB::Var>,
        call_cursor: AB::Expr,
        source_cursor: AB::Expr,
        tidx: AB::Expr,
        sample_count: AB::Expr,
        transcript_state: [AB::Var; 16],
        accumulator_root: [AB::Var; DIGEST_SIZE],
        accumulator_digest: [AB::Var; DIGEST_SIZE],
        vm_pc: AB::Expr,
        vm_root: [AB::Var; DIGEST_SIZE],
        chain: [AB::Var; DIGEST_SIZE],
        setup_left: [AB::Var; DIGEST_SIZE],
        setup_right: [AB::Var; DIGEST_SIZE],
        setup: [AB::Var; DIGEST_SIZE],
        protocol: [AB::Var; DIGEST_SIZE],
        control: [AB::Var; DIGEST_SIZE],
        transcript: [AB::Var; DIGEST_SIZE],
        accumulator: [AB::Var; DIGEST_SIZE],
        transcript_accumulator: [AB::Var; DIGEST_SIZE],
        vm_chain: [AB::Var; DIGEST_SIZE],
        vm_program: [AB::Var; DIGEST_SIZE],
        left: [AB::Var; DIGEST_SIZE],
        expected: [AB::Var; DIGEST_SIZE],
    ) where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        let active = AB::Expr::from(local.active);
        self.compress(
            builder,
            local.warp_protocol_digest.map(Into::into),
            local.relation_digest.map(Into::into),
            setup_left.map(Into::into),
            active.clone(),
        );
        self.compress(
            builder,
            local.warp_index_digest.map(Into::into),
            local.schedule_digest.map(Into::into),
            setup_right.map(Into::into),
            active.clone(),
        );
        self.compress(
            builder,
            setup_left.map(Into::into),
            setup_right.map(Into::into),
            setup.map(Into::into),
            active.clone(),
        );
        self.compress(
            builder,
            local.source_protocol_digest.map(Into::into),
            setup.map(Into::into),
            protocol.map(Into::into),
            active.clone(),
        );
        let metadata = core::array::from_fn(|index| match index {
            0 => AB::Expr::from_u32(REDUCED_SWIRL_TRANSITION_STATE_METADATA_TAG),
            1 => AB::Expr::from_u32(REDUCED_SWIRL_TRANSITION_LEAF_PROTOCOL_VERSION),
            2 => local.total_source_count.into(),
            3 => call_cursor.clone(),
            4 => source_cursor.clone(),
            5 => tidx.clone(),
            6 => sample_count.clone(),
            7 => vm_pc.clone(),
            _ => unreachable!(),
        });
        self.compress(
            builder,
            protocol.map(Into::into),
            metadata,
            control.map(Into::into),
            active.clone(),
        );
        self.compress(
            builder,
            core::array::from_fn(|i| transcript_state[i].into()),
            core::array::from_fn(|i| transcript_state[DIGEST_SIZE + i].into()),
            transcript.map(Into::into),
            active.clone(),
        );
        self.compress(
            builder,
            accumulator_root.map(Into::into),
            accumulator_digest.map(Into::into),
            accumulator.map(Into::into),
            active.clone(),
        );
        self.compress(
            builder,
            transcript.map(Into::into),
            accumulator.map(Into::into),
            transcript_accumulator.map(Into::into),
            active.clone(),
        );
        self.compress(
            builder,
            vm_root.map(Into::into),
            chain.map(Into::into),
            vm_chain.map(Into::into),
            active.clone(),
        );
        self.compress(
            builder,
            vm_chain.map(Into::into),
            local.program_commitment.map(Into::into),
            vm_program.map(Into::into),
            active.clone(),
        );
        self.compress(
            builder,
            control.map(Into::into),
            transcript_accumulator.map(Into::into),
            left.map(Into::into),
            active.clone(),
        );
        self.compress(
            builder,
            left.map(Into::into),
            vm_program.map(Into::into),
            expected.map(Into::into),
            active,
        );
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
                input: core::array::from_fn(|i| {
                    if i < DIGEST_SIZE {
                        left[i].clone()
                    } else {
                        right[i - DIGEST_SIZE].clone()
                    }
                }),
                output,
            },
            enabled,
        );
    }

    pub fn generate_trace(
        &self,
        record: &ReducedSwirlTransitionLeafRecord,
    ) -> Result<(RowMajorMatrix<F>, ReducedSwirlTransitionLeafDerived), &'static str> {
        let derived = record.derive()?;
        let width = ReducedSwirlTransitionLeafBoundaryCols::<F>::width();
        let mut values = F::zero_vec(2 * width);
        let local: &mut ReducedSwirlTransitionLeafBoundaryCols<F> = values[..width].borrow_mut();
        local.active = F::ONE;
        local.source_protocol_digest = record.source.protocol_digest;
        local.warp_protocol_digest = record.transition.protocol_digest;
        local.relation_digest = record.transition.relation_digest;
        local.warp_index_digest = record.transition.warp_index_digest;
        local.schedule_digest = record.transition.schedule_digest;
        local.manifest_digest = record.source.manifest_digest;
        local.total_source_count = record.transition.total_source_count;
        local.call_index = record.transition.call_index;
        local.source_start = record.transition.source_start;
        local.fresh_count = record.transition.fresh_count;
        local.prior_count = record.transition.prior_count;
        local.batch_start_tidx = record.transition.batch_start_tidx;
        local.vacc_start_tidx = record.transition.vacc_start_tidx;
        local.vacc_end_tidx = record.transition.vacc_end_tidx;
        local.start_sample_count = record.transition.start_sample_count;
        local.start_state = record.transition.start_state;
        local.end_sample_count = record.transition.end_sample_count;
        local.end_state = record.transition.end_state;
        local.prior_root = record.transition.prior_root;
        local.output_root = record.transition.output_root;
        local.prior_digest = record.transition.prior_digest;
        local.output_digest = record.transition.output_digest;
        local.is_final = record.transition.is_final;
        local.program_commitment = record.source.program_commitment;
        local.initial_pc = record.source.initial_pc;
        local.initial_root = record.source.initial_root;
        local.final_pc = record.source.final_pc;
        local.final_root = record.source.final_root;
        local.exit_code = record.source.exit_code;
        local.is_terminate = record.source.is_terminate;
        local.chain_before = record.chain_before;
        local.chunk = derived.chunk;
        local.chain_after = derived.chain_after;
        local.genesis_chain = derived.genesis_chain;
        local.initial_setup_left = derived.initial_hashes.setup_left;
        local.initial_setup_right = derived.initial_hashes.setup_right;
        local.initial_setup = derived.initial_hashes.setup;
        local.initial_protocol = derived.initial_hashes.protocol;
        local.initial_control = derived.initial_hashes.control;
        local.initial_transcript = derived.initial_hashes.transcript;
        local.initial_accumulator = derived.initial_hashes.accumulator;
        local.initial_transcript_accumulator = derived.initial_hashes.transcript_accumulator;
        local.initial_vm_chain = derived.initial_hashes.vm_chain;
        local.initial_vm_program = derived.initial_hashes.vm_program;
        local.initial_left = derived.initial_hashes.left;
        local.initial_boundary = derived.initial_boundary;
        local.final_setup_left = derived.final_hashes.setup_left;
        local.final_setup_right = derived.final_hashes.setup_right;
        local.final_setup = derived.final_hashes.setup;
        local.final_protocol = derived.final_hashes.protocol;
        local.final_control = derived.final_hashes.control;
        local.final_transcript = derived.final_hashes.transcript;
        local.final_accumulator = derived.final_hashes.accumulator;
        local.final_transcript_accumulator = derived.final_hashes.transcript_accumulator;
        local.final_vm_chain = derived.final_hashes.vm_chain;
        local.final_vm_program = derived.final_hashes.vm_program;
        local.final_left = derived.final_hashes.left;
        local.final_boundary = derived.final_boundary;
        Ok((RowMajorMatrix::new(values, width), derived))
    }
}

#[derive(Clone)]
pub struct ReducedSwirlTransitionLeafVerifierPvsAir {
    expected: VerifierBasePvs<F>,
}

impl ReducedSwirlTransitionLeafVerifierPvsAir {
    #[must_use]
    pub fn new(binding: &ReducedSwirlTransitionLeafBinding) -> Self {
        Self {
            expected: binding.verifier_pvs(),
        }
    }
}

impl BaseAir<F> for ReducedSwirlTransitionLeafVerifierPvsAir {
    fn width(&self) -> usize {
        1
    }
}
impl BaseAirWithPublicValues<F> for ReducedSwirlTransitionLeafVerifierPvsAir {
    fn num_public_values(&self) -> usize {
        VerifierBasePvs::<u8>::width()
    }
}
impl PartitionedBaseAir<F> for ReducedSwirlTransitionLeafVerifierPvsAir {}

impl<AB> Air<AB> for ReducedSwirlTransitionLeafVerifierPvsAir
where
    AB: AirBuilder<F = F> + AirBuilderWithPublicValues + PairBuilder,
    AB::Var: Copy,
    AB::PublicVar: Copy,
    AB::Expr: From<AB::PublicVar>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("transition-leaf PVS row")[0];
        let next = main.row_slice(1).expect("transition-leaf PVS padding")[0];
        one_row_selector(builder, local, next);
        let public_values = builder.public_values().to_vec();
        for (actual, expected) in public_values.iter().zip(self.expected.as_slice()) {
            builder.when(local).assert_eq(
                Into::<AB::Expr>::into(*actual),
                AB::Expr::from_u32(expected.as_canonical_u32()),
            );
        }
    }
}

pub trait ReducedSwirlTransitionLeafComponents: Send + Sync + 'static {
    fn source_receipt_bus(&self) -> ReducedSwirlSourceReceiptBus;
    fn transition_receipt_bus(&self) -> ReducedSwirlVaccTransitionReceiptBus;
    fn execution_bus(&self) -> ReducedSwirlExecutionBus;
    fn compress_bus(&self) -> Poseidon2CompressBus;
    fn source_component_digest(&self) -> Digest;
    fn vacc_component_digest(&self) -> Digest;
    fn component_air_count(&self) -> usize;
    fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>>;
}

pub struct ReducedSwirlTransitionLeafCircuit<C: ReducedSwirlTransitionLeafComponents> {
    pub binding: ReducedSwirlTransitionLeafBinding,
    pub verifier_pvs_air: Arc<ReducedSwirlTransitionLeafVerifierPvsAir>,
    pub vm_pvs_air: Arc<ReducedSwirlWrapperVmPvsAir>,
    pub boundary_air: Arc<ReducedSwirlTransitionLeafBoundaryAir>,
    pub components: Arc<C>,
}

impl<C: ReducedSwirlTransitionLeafComponents> ReducedSwirlTransitionLeafCircuit<C> {
    pub fn new(
        binding: ReducedSwirlTransitionLeafBinding,
        components: Arc<C>,
    ) -> Result<Self, &'static str> {
        binding.validate()?;
        if components.component_air_count() == 0
            || components.source_component_digest() != binding.source_component_digest
            || components.vacc_component_digest() != binding.vacc_component_digest
        {
            return Err("reduced-SWIRL transition-leaf components");
        }
        Ok(Self {
            verifier_pvs_air: Arc::new(ReducedSwirlTransitionLeafVerifierPvsAir::new(&binding)),
            vm_pvs_air: Arc::new(ReducedSwirlWrapperVmPvsAir::new(components.execution_bus())),
            boundary_air: Arc::new(ReducedSwirlTransitionLeafBoundaryAir {
                source_receipt_bus: components.source_receipt_bus(),
                transition_receipt_bus: components.transition_receipt_bus(),
                execution_bus: components.execution_bus(),
                compress_bus: components.compress_bus(),
            }),
            binding,
            components,
        })
    }
}

impl<SC, C> Circuit<SC> for ReducedSwirlTransitionLeafCircuit<C>
where
    SC: StarkProtocolConfig<F = F>,
    C: ReducedSwirlTransitionLeafComponents,
{
    fn airs(&self) -> Vec<AirRef<SC>> {
        [
            self.verifier_pvs_air.clone() as AirRef<SC>,
            self.vm_pvs_air.clone() as AirRef<SC>,
            self.boundary_air.clone() as AirRef<SC>,
        ]
        .into_iter()
        .chain(self.components.airs::<SC>())
        .collect()
    }
}

pub struct ReducedSwirlTransitionLeafCoreTraces {
    pub verifier_pvs: RowMajorMatrix<F>,
    pub verifier_public_values: Vec<F>,
    pub vm_pvs: RowMajorMatrix<F>,
    pub vm_public_values: Vec<F>,
    pub boundary: RowMajorMatrix<F>,
    pub derived: ReducedSwirlTransitionLeafDerived,
}

pub fn generate_reduced_swirl_transition_leaf_core_traces(
    binding: &ReducedSwirlTransitionLeafBinding,
    boundary_air: &ReducedSwirlTransitionLeafBoundaryAir,
    record: &ReducedSwirlTransitionLeafRecord,
) -> Result<ReducedSwirlTransitionLeafCoreTraces, &'static str> {
    binding.validate()?;
    let (boundary, derived) = boundary_air.generate_trace(record)?;
    let verifier_public_values = binding.verifier_pvs().as_slice().to_vec();
    let vm = &derived.vm_pvs;
    let mut vm_public_values = Vec::with_capacity(VmPvs::<u8>::width());
    vm_public_values.extend_from_slice(&vm.program_commit);
    vm_public_values.extend([vm.initial_pc, vm.final_pc, vm.exit_code, vm.is_terminate]);
    vm_public_values.extend_from_slice(&vm.initial_root);
    vm_public_values.extend_from_slice(&vm.final_root);
    Ok(ReducedSwirlTransitionLeafCoreTraces {
        verifier_pvs: RowMajorMatrix::new(vec![F::ONE, F::ZERO], 1),
        verifier_public_values,
        vm_pvs: RowMajorMatrix::new(vec![F::ONE, F::ZERO], 1),
        vm_public_values,
        boundary,
        derived,
    })
}

pub(crate) fn state_digest(
    state: &ReducedSwirlTransitionState,
    inputs: &mut Vec<[F; 2 * DIGEST_SIZE]>,
) -> StateHashWitness {
    let setup_left = record_compression(state.warp_protocol_digest, state.relation_digest, inputs);
    let setup_right = record_compression(state.warp_index_digest, state.schedule_digest, inputs);
    let setup = record_compression(setup_left, setup_right, inputs);
    let protocol = record_compression(state.source_protocol_digest, setup, inputs);
    let metadata = [
        F::from_u32(REDUCED_SWIRL_TRANSITION_STATE_METADATA_TAG),
        F::from_u32(REDUCED_SWIRL_TRANSITION_LEAF_PROTOCOL_VERSION),
        state.total_source_count,
        state.call_cursor,
        state.source_cursor,
        state.transcript_tidx,
        state.transcript_sample_count,
        state.vm_pc,
    ];
    let control = record_compression(protocol, metadata, inputs);
    let transcript = record_compression(
        state.transcript_state[..DIGEST_SIZE]
            .try_into()
            .expect("digest half"),
        state.transcript_state[DIGEST_SIZE..]
            .try_into()
            .expect("digest half"),
        inputs,
    );
    let accumulator = record_compression(state.accumulator_root, state.accumulator_digest, inputs);
    let transcript_accumulator = record_compression(transcript, accumulator, inputs);
    let vm_chain = record_compression(state.vm_root, state.manifest_chain, inputs);
    let vm_program = record_compression(vm_chain, state.program_commitment, inputs);
    let left = record_compression(control, transcript_accumulator, inputs);
    let boundary = record_compression(left, vm_program, inputs);
    StateHashWitness {
        setup_left,
        setup_right,
        setup,
        protocol,
        control,
        transcript,
        accumulator,
        transcript_accumulator,
        vm_chain,
        vm_program,
        left,
        boundary,
    }
}

/// Canonical digest exported through `VmPvs::{initial_root,final_root}` by a
/// transition leaf and opened again by the terminal finalizer.
#[must_use]
pub fn reduced_swirl_transition_state_digest(state: &ReducedSwirlTransitionState) -> Digest {
    state_digest(state, &mut Vec::new()).boundary
}

/// Canonical domain and global schedule coordinates committed by one rolling
/// transition-manifest link.  The local manifest digest is compressed with
/// this value before it is appended to the chain.
#[must_use]
pub fn reduced_swirl_transition_chain_metadata(
    transition: &ReducedSwirlVaccTransitionReceiptMessage<F>,
) -> Digest {
    [
        F::from_u32(REDUCED_SWIRL_TRANSITION_CHAIN_METADATA_TAG),
        F::from_u32(REDUCED_SWIRL_TRANSITION_LEAF_PROTOCOL_VERSION),
        transition.total_source_count,
        transition.call_index,
        transition.source_start,
        transition.fresh_count,
        transition.prior_count,
        transition.is_final,
    ]
}

fn chain_metadata_expr<AB: AirBuilder<F = F>>(
    local: &ReducedSwirlTransitionLeafBoundaryCols<AB::Var>,
) -> [AB::Expr; DIGEST_SIZE]
where
    AB::Var: Copy,
{
    [
        AB::Expr::from_u32(REDUCED_SWIRL_TRANSITION_CHAIN_METADATA_TAG),
        AB::Expr::from_u32(REDUCED_SWIRL_TRANSITION_LEAF_PROTOCOL_VERSION),
        local.total_source_count.into(),
        local.call_index.into(),
        local.source_start.into(),
        local.fresh_count.into(),
        local.prior_count.into(),
        local.is_final.into(),
    ]
}

/// Fixed second input of the transition-manifest genesis compression.
#[must_use]
pub fn reduced_swirl_transition_chain_genesis_metadata() -> Digest {
    core::array::from_fn(|i| match i {
        0 => F::from_u32(REDUCED_SWIRL_TRANSITION_CHAIN_GENESIS_TAG),
        1 => F::from_u32(REDUCED_SWIRL_TRANSITION_LEAF_PROTOCOL_VERSION),
        _ => F::ZERO,
    })
}

/// Initial rolling manifest state shared by the native stream, transition
/// leaves, and the terminal reconciliation circuit.
#[must_use]
pub fn reduced_swirl_transition_chain_genesis(source_protocol_digest: Digest) -> Digest {
    poseidon2_compress_with_capacity(
        source_protocol_digest,
        reduced_swirl_transition_chain_genesis_metadata(),
    )
    .0
}

/// Append one authenticated local manifest to the rolling transition chain.
/// This helper is deliberately the host-side oracle for the boundary AIR and
/// the terminal flat-manifest reconciliation, preventing the three users from
/// acquiring subtly different call-coordinate encodings.
#[must_use]
pub fn reduced_swirl_transition_chain_append(
    chain_before: Digest,
    transition: &ReducedSwirlVaccTransitionReceiptMessage<F>,
    local_manifest_digest: Digest,
) -> Digest {
    let chunk = poseidon2_compress_with_capacity(
        reduced_swirl_transition_chain_metadata(transition),
        local_manifest_digest,
    )
    .0;
    poseidon2_compress_with_capacity(chain_before, chunk).0
}

fn genesis_metadata_expr<AB: AirBuilder<F = F>>() -> [AB::Expr; DIGEST_SIZE] {
    core::array::from_fn(|i| match i {
        0 => AB::Expr::from_u32(REDUCED_SWIRL_TRANSITION_CHAIN_GENESIS_TAG),
        1 => AB::Expr::from_u32(REDUCED_SWIRL_TRANSITION_LEAF_PROTOCOL_VERSION),
        _ => AB::Expr::ZERO,
    })
}

fn record_compression(
    left: Digest,
    right: Digest,
    inputs: &mut Vec<[F; 2 * DIGEST_SIZE]>,
) -> Digest {
    inputs.push(core::array::from_fn(|i| {
        if i < DIGEST_SIZE {
            left[i]
        } else {
            right[i - DIGEST_SIZE]
        }
    }));
    poseidon2_compress_with_capacity(left, right).0
}

fn one_row_selector<AB: AirBuilder>(builder: &mut AB, local: AB::Var, next: AB::Var)
where
    AB::Var: Copy,
{
    builder.assert_bool(local);
    builder.when_first_row().assert_one(local);
    builder.when_last_row().assert_zero(local);
    builder
        .when_transition()
        .assert_zero(next * (AB::Expr::ONE - local));
}

fn is_zero_digest(digest: &Digest) -> bool {
    digest.iter().all(|value| *value == F::ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: u32) -> Digest {
        core::array::from_fn(|i| F::from_u32(seed + i as u32 + 1))
    }

    fn source(
        offset: u32,
        count: u32,
        initial_pc: u32,
        final_pc: u32,
    ) -> ReducedSwirlSourceReceiptMessage<F> {
        ReducedSwirlSourceReceiptMessage {
            protocol_digest: digest(10),
            manifest_digest: digest(20 + offset),
            source_offset: F::from_u32(offset),
            source_count: F::from_u32(count),
            program_commitment: digest(40),
            initial_pc: F::from_u32(initial_pc),
            initial_root: digest(100 + initial_pc),
            final_pc: F::from_u32(final_pc),
            final_root: digest(100 + final_pc),
            exit_code: F::from_u32(if offset + count == 15 { 0 } else { 2 }),
            is_terminate: F::from_bool(offset + count == 15),
        }
    }

    fn transition(
        call: u32,
        source_start: u32,
        fresh_count: u32,
        prior_root: Digest,
        prior_digest: Digest,
        start_state: [F; 16],
        start_samples: u32,
        manifest_digest: Digest,
    ) -> ReducedSwirlVaccTransitionReceiptMessage<F> {
        let final_call = call == 1;
        ReducedSwirlVaccTransitionReceiptMessage {
            protocol_digest: digest(200),
            relation_digest: digest(220),
            warp_index_digest: digest(240),
            schedule_digest: digest(260),
            total_source_count: F::from_u32(15),
            call_index: F::from_u32(call),
            source_start: F::from_u32(source_start),
            fresh_count: F::from_u32(fresh_count),
            prior_count: F::from_bool(call != 0),
            batch_start_tidx: F::from_u32(if call == 0 { 1_000 } else { 1_900 }),
            vacc_start_tidx: F::from_u32(1_100 + call * 1_000),
            vacc_end_tidx: F::from_u32(1_900 + call * 1_000),
            start_sample_count: F::from_u32(start_samples),
            start_state,
            end_sample_count: F::from_u32(1),
            end_state: core::array::from_fn(|i| F::from_u32(300 + call * 32 + i as u32)),
            prior_root,
            output_root: digest(320 + call * 20),
            prior_digest,
            output_digest: digest(360 + call * 20),
            manifest_digest,
            is_final: F::from_bool(final_call),
        }
    }

    #[test]
    fn adjacent_transition_boundaries_are_identical() {
        let first_source = source(0, 8, 7, 11);
        let first_transition = transition(
            0,
            0,
            8,
            [F::ZERO; DIGEST_SIZE],
            [F::ZERO; DIGEST_SIZE],
            [F::ZERO; 16],
            0,
            first_source.manifest_digest,
        );
        let genesis = reduced_swirl_transition_chain_genesis(first_source.protocol_digest);
        let expected_first_chain = reduced_swirl_transition_chain_append(
            genesis,
            &first_transition,
            first_source.manifest_digest,
        );
        let first = ReducedSwirlTransitionLeafRecord {
            source: first_source,
            transition: first_transition,
            chain_before: genesis,
        }
        .derive()
        .unwrap();
        assert_eq!(first.chain_after, expected_first_chain);

        let mut second_source = source(8, 7, 11, 19);
        second_source.initial_root = digest(111);
        let second_transition = transition(
            1,
            8,
            7,
            digest(320),
            digest(360),
            core::array::from_fn(|i| F::from_u32(300 + i as u32)),
            1,
            second_source.manifest_digest,
        );
        let second = ReducedSwirlTransitionLeafRecord {
            source: second_source,
            transition: second_transition,
            chain_before: first.chain_after,
        }
        .derive()
        .unwrap();
        assert_eq!(first.final_boundary, second.initial_boundary);
        assert_eq!(first.vm_pvs.final_pc, second.vm_pvs.initial_pc);
        assert_eq!(first.vm_pvs.final_root, second.vm_pvs.initial_root);
    }

    #[test]
    fn receipt_seams_and_genesis_are_not_host_assumptions() {
        let source = source(0, 8, 1, 2);
        let transition = transition(
            0,
            0,
            8,
            [F::ZERO; DIGEST_SIZE],
            [F::ZERO; DIGEST_SIZE],
            [F::ZERO; 16],
            0,
            source.manifest_digest,
        );
        let mut record = ReducedSwirlTransitionLeafRecord {
            source,
            transition,
            chain_before: digest(999),
        };
        assert!(record.derive().is_err());
        record.chain_before = reduced_swirl_transition_chain_genesis(record.source.protocol_digest);
        record.transition.manifest_digest[0] += F::ONE;
        assert!(record.derive().is_err());
        record.transition.manifest_digest = record.source.manifest_digest;
        record.transition.source_start += F::ONE;
        assert!(record.derive().is_err());
    }
}

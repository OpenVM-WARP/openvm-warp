//! Fixed-capacity certificate for a contiguous interval of reduced-SWIRL sources.
//!
//! A source leaf verifies only the deferred-WHIR prefixes assigned to one
//! bounded interval.  It exports ordinary OpenVM [`VmPvs`] so the existing
//! fixed-key recursive tree can authenticate arbitrarily many leaves without
//! retaining their verifier traces.  The public memory roots are synthetic:
//! they commit to the real VM boundary and to a rolling, domain-separated
//! chain of canonical source-manifest chunks.  Equality of adjacent roots
//! therefore binds both VM continuity and source order under Poseidon2
//! collision resistance.

use core::borrow::{Borrow, BorrowMut};
use std::sync::Arc;

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit::bus::{Poseidon2CompressBus, Poseidon2CompressMessage};
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
    reduced_swirl_source_receipt::ReducedSwirlSourceReceiptBlock,
    reduced_swirl_warp::{
        ReducedSwirlExecutionBus, ReducedSwirlExecutionMessage, ReducedSwirlSourceReceiptBus,
        ReducedSwirlSourceReceiptMessage, ReducedSwirlWrapperVmPvsAir,
    },
    Circuit,
};

pub const REDUCED_SWIRL_SOURCE_LEAF_PROTOCOL_VERSION: u32 = 1;
/// Production leaves are deliberately bounded independently of block size.
/// This is large enough to amortize a STARK proof and remains far below the
/// measured 568-source CUDA residency cliff.
/// Fixed source-certification arity. Eight matches the production WARP fresh
/// batch while remaining bounded independently of block size. The source
/// certificate and WARP arities are protocol-bound separately even when their
/// production values coincide.
pub const REDUCED_SWIRL_SOURCE_LEAF_CAPACITY: usize = 8;

const CHAIN_METADATA_TAG: u32 = 0x5253_4c01;
const BOUNDARY_METADATA_TAG: u32 = 0x5253_4c02;

#[derive(Clone, Debug, PartialEq)]
pub struct ReducedSwirlSourceLeafBinding {
    pub protocol_version: u32,
    pub source_capacity: u32,
    pub source_component_digest: Digest,
    /// Commitment to the original execution/SWIRL application VK cached
    /// table.  Every source leaf is a level-zero recursive proof for this
    /// exact application lineage.
    pub recursive_app_vk_commit: VkCommit<F>,
}

impl ReducedSwirlSourceLeafBinding {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.protocol_version != REDUCED_SWIRL_SOURCE_LEAF_PROTOCOL_VERSION {
            return Err("source-leaf protocol version");
        }
        if self.source_capacity == 0
            || self.source_capacity as usize > REDUCED_SWIRL_SOURCE_LEAF_CAPACITY
            || !self.source_capacity.is_power_of_two()
        {
            return Err("source-leaf capacity");
        }
        if is_zero_digest(&self.source_component_digest)
            || is_zero_digest(&self.recursive_app_vk_commit.cached_commit)
            || is_zero_digest(&self.recursive_app_vk_commit.vk_pre_hash)
        {
            return Err("source-leaf binding digest");
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

#[derive(Clone, Debug, PartialEq)]
pub struct ReducedSwirlSourceLeafRecord {
    pub receipt: ReducedSwirlSourceReceiptMessage<F>,
    pub chain_before: Digest,
}

impl ReducedSwirlSourceLeafRecord {
    pub fn from_block(
        block: &ReducedSwirlSourceReceiptBlock,
        protocol_digest: Digest,
        chain_before: Digest,
    ) -> Result<Self, &'static str> {
        let first = block
            .sources
            .first()
            .ok_or("source-leaf empty receipt block")?;
        let last = block
            .sources
            .last()
            .ok_or("source-leaf empty receipt block")?;
        let source_count =
            u32::try_from(block.sources.len()).map_err(|_| "source-leaf source count overflow")?;
        block
            .source_offset
            .checked_add(source_count)
            .ok_or("source-leaf source interval overflow")?;
        if is_zero_digest(&protocol_digest)
            || is_zero_digest(&block.manifest_digest)
            || is_zero_digest(&first.vm.program_commitment)
        {
            return Err("source-leaf unset receipt digest");
        }
        Ok(Self {
            receipt: ReducedSwirlSourceReceiptMessage {
                protocol_digest,
                manifest_digest: block.manifest_digest,
                source_offset: F::from_u32(block.source_offset),
                source_count: F::from_u32(source_count),
                program_commitment: first.vm.program_commitment,
                initial_pc: first.vm.initial_pc,
                initial_root: first.vm.initial_root,
                final_pc: last.vm.final_pc,
                final_root: last.vm.final_root,
                exit_code: last.vm.exit_code,
                is_terminate: last.vm.is_terminate,
            },
            chain_before,
        })
    }

    pub fn derive(&self) -> Result<ReducedSwirlSourceLeafDerived, &'static str> {
        if self.receipt.source_count == F::ZERO
            || is_zero_digest(&self.receipt.protocol_digest)
            || is_zero_digest(&self.receipt.manifest_digest)
            || is_zero_digest(&self.receipt.program_commitment)
        {
            return Err("source-leaf malformed receipt");
        }
        let source_offset = self.receipt.source_offset.as_canonical_u32();
        let source_count = self.receipt.source_count.as_canonical_u32();
        let source_end = source_offset
            .checked_add(source_count)
            .ok_or("source-leaf source interval overflow")?;
        let mut compression_inputs = Vec::with_capacity(8);
        let chunk_metadata = chain_metadata(source_offset, source_count);
        let chunk_commitment = record_compression(
            chunk_metadata,
            self.receipt.manifest_digest,
            &mut compression_inputs,
        );
        let chain_after =
            record_compression(self.chain_before, chunk_commitment, &mut compression_inputs);
        let initial_protocol = record_compression(
            self.receipt.protocol_digest,
            reduced_swirl_source_boundary_metadata(source_offset, self.receipt.initial_pc),
            &mut compression_inputs,
        );
        let initial_state = record_compression(
            self.receipt.initial_root,
            self.chain_before,
            &mut compression_inputs,
        );
        let initial_boundary =
            record_compression(initial_protocol, initial_state, &mut compression_inputs);
        let final_protocol = record_compression(
            self.receipt.protocol_digest,
            reduced_swirl_source_boundary_metadata(source_end, self.receipt.final_pc),
            &mut compression_inputs,
        );
        let final_state = record_compression(
            self.receipt.final_root,
            chain_after,
            &mut compression_inputs,
        );
        let final_boundary =
            record_compression(final_protocol, final_state, &mut compression_inputs);
        let vm_pvs = VmPvs {
            program_commit: self.receipt.program_commitment,
            initial_pc: self.receipt.source_offset,
            final_pc: F::from_u32(source_end),
            exit_code: self.receipt.exit_code,
            is_terminate: self.receipt.is_terminate,
            initial_root: initial_boundary,
            final_root: final_boundary,
        };
        Ok(ReducedSwirlSourceLeafDerived {
            source_end: F::from_u32(source_end),
            chunk_commitment,
            chain_after,
            initial_protocol,
            initial_state,
            initial_boundary,
            final_protocol,
            final_state,
            final_boundary,
            vm_pvs,
            compression_inputs,
        })
    }
}

pub struct ReducedSwirlSourceLeafDerived {
    pub source_end: F,
    pub chunk_commitment: Digest,
    pub chain_after: Digest,
    pub initial_protocol: Digest,
    pub initial_state: Digest,
    pub initial_boundary: Digest,
    pub final_protocol: Digest,
    pub final_state: Digest,
    pub final_boundary: Digest,
    pub vm_pvs: VmPvs<F>,
    pub compression_inputs: Vec<[F; 2 * DIGEST_SIZE]>,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ReducedSwirlSourceLeafBoundaryCols<T> {
    pub active: T,
    pub protocol_digest: [T; DIGEST_SIZE],
    pub manifest_digest: [T; DIGEST_SIZE],
    pub source_offset: T,
    pub source_count: T,
    pub program_commitment: [T; DIGEST_SIZE],
    pub initial_pc: T,
    pub initial_root: [T; DIGEST_SIZE],
    pub final_pc: T,
    pub final_root: [T; DIGEST_SIZE],
    pub exit_code: T,
    pub is_terminate: T,
    pub chain_before: [T; DIGEST_SIZE],
    pub chunk_commitment: [T; DIGEST_SIZE],
    pub chain_after: [T; DIGEST_SIZE],
    pub initial_protocol: [T; DIGEST_SIZE],
    pub initial_state: [T; DIGEST_SIZE],
    pub initial_boundary: [T; DIGEST_SIZE],
    pub final_protocol: [T; DIGEST_SIZE],
    pub final_state: [T; DIGEST_SIZE],
    pub final_boundary: [T; DIGEST_SIZE],
}

#[derive(Clone, ColumnsAir)]
#[columns_via(ReducedSwirlSourceLeafBoundaryCols<u8>)]
pub struct ReducedSwirlSourceLeafBoundaryAir {
    pub receipt_bus: ReducedSwirlSourceReceiptBus,
    pub execution_bus: ReducedSwirlExecutionBus,
    pub compress_bus: Poseidon2CompressBus,
}

impl BaseAir<F> for ReducedSwirlSourceLeafBoundaryAir {
    fn width(&self) -> usize {
        ReducedSwirlSourceLeafBoundaryCols::<u8>::width()
    }
}
impl BaseAirWithPublicValues<F> for ReducedSwirlSourceLeafBoundaryAir {}
impl PartitionedBaseAir<F> for ReducedSwirlSourceLeafBoundaryAir {}

impl<AB> Air<AB> for ReducedSwirlSourceLeafBoundaryAir
where
    AB: AirBuilder<F = F> + InteractionBuilder + PairBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("source-leaf boundary row");
        let local: &ReducedSwirlSourceLeafBoundaryCols<AB::Var> = (*local).borrow();
        let next = main.row_slice(1).expect("source-leaf boundary padding");
        let next: &ReducedSwirlSourceLeafBoundaryCols<AB::Var> = (*next).borrow();
        one_row_selector(builder, local.active, next.active);
        for value in next.as_slice().iter().skip(1) {
            // On the padding row, `next` wraps to the active first row.  Gate
            // the padding check by the active row so the cyclic transition is
            // not incorrectly required to erase the authenticated boundary.
            builder.when(local.active).assert_zero(*value);
        }

        self.receipt_bus.lookup_key(
            builder,
            ReducedSwirlSourceReceiptMessage {
                protocol_digest: local.protocol_digest.map(Into::into),
                manifest_digest: local.manifest_digest.map(Into::into),
                source_offset: local.source_offset.into(),
                source_count: local.source_count.into(),
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

        let source_end = AB::Expr::from(local.source_offset) + local.source_count;
        self.lookup_compression(
            builder,
            chain_metadata_expr::<AB>(local.source_offset, local.source_count),
            local.manifest_digest.map(Into::into),
            local.chunk_commitment.map(Into::into),
            local.active,
        );
        self.lookup_compression(
            builder,
            local.chain_before.map(Into::into),
            local.chunk_commitment.map(Into::into),
            local.chain_after.map(Into::into),
            local.active,
        );
        self.lookup_compression(
            builder,
            local.protocol_digest.map(Into::into),
            reduced_swirl_source_boundary_metadata_expr::<AB>(
                local.source_offset.into(),
                local.initial_pc.into(),
            ),
            local.initial_protocol.map(Into::into),
            local.active,
        );
        self.lookup_compression(
            builder,
            local.initial_root.map(Into::into),
            local.chain_before.map(Into::into),
            local.initial_state.map(Into::into),
            local.active,
        );
        self.lookup_compression(
            builder,
            local.initial_protocol.map(Into::into),
            local.initial_state.map(Into::into),
            local.initial_boundary.map(Into::into),
            local.active,
        );
        self.lookup_compression(
            builder,
            local.protocol_digest.map(Into::into),
            reduced_swirl_source_boundary_metadata_expr::<AB>(
                source_end.clone(),
                local.final_pc.into(),
            ),
            local.final_protocol.map(Into::into),
            local.active,
        );
        self.lookup_compression(
            builder,
            local.final_root.map(Into::into),
            local.chain_after.map(Into::into),
            local.final_state.map(Into::into),
            local.active,
        );
        self.lookup_compression(
            builder,
            local.final_protocol.map(Into::into),
            local.final_state.map(Into::into),
            local.final_boundary.map(Into::into),
            local.active,
        );

        self.execution_bus.lookup_key(
            builder,
            ReducedSwirlExecutionMessage {
                vm_pvs: VmPvs {
                    program_commit: local.program_commitment.map(Into::into),
                    initial_pc: local.source_offset.into(),
                    final_pc: source_end,
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

impl ReducedSwirlSourceLeafBoundaryAir {
    fn lookup_compression<AB: AirBuilder<F = F> + InteractionBuilder>(
        &self,
        builder: &mut AB,
        left: [AB::Expr; DIGEST_SIZE],
        right: [AB::Expr; DIGEST_SIZE],
        output: [AB::Expr; DIGEST_SIZE],
        enabled: AB::Var,
    ) where
        AB::Var: Copy,
    {
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

    pub fn generate_trace(
        &self,
        record: &ReducedSwirlSourceLeafRecord,
    ) -> Result<(RowMajorMatrix<F>, ReducedSwirlSourceLeafDerived), &'static str> {
        let derived = record.derive()?;
        let width = ReducedSwirlSourceLeafBoundaryCols::<F>::width();
        let mut values = vec![F::ZERO; 2 * width];
        let local: &mut ReducedSwirlSourceLeafBoundaryCols<F> = values[..width].borrow_mut();
        local.active = F::ONE;
        local.protocol_digest = record.receipt.protocol_digest;
        local.manifest_digest = record.receipt.manifest_digest;
        local.source_offset = record.receipt.source_offset;
        local.source_count = record.receipt.source_count;
        local.program_commitment = record.receipt.program_commitment;
        local.initial_pc = record.receipt.initial_pc;
        local.initial_root = record.receipt.initial_root;
        local.final_pc = record.receipt.final_pc;
        local.final_root = record.receipt.final_root;
        local.exit_code = record.receipt.exit_code;
        local.is_terminate = record.receipt.is_terminate;
        local.chain_before = record.chain_before;
        local.chunk_commitment = derived.chunk_commitment;
        local.chain_after = derived.chain_after;
        local.initial_protocol = derived.initial_protocol;
        local.initial_state = derived.initial_state;
        local.initial_boundary = derived.initial_boundary;
        local.final_protocol = derived.final_protocol;
        local.final_state = derived.final_state;
        local.final_boundary = derived.final_boundary;
        Ok((RowMajorMatrix::new(values, width), derived))
    }
}

#[derive(Clone)]
pub struct ReducedSwirlSourceLeafVerifierPvsAir {
    expected: VerifierBasePvs<F>,
}

impl ReducedSwirlSourceLeafVerifierPvsAir {
    #[must_use]
    pub fn new(binding: &ReducedSwirlSourceLeafBinding) -> Self {
        Self {
            expected: binding.verifier_pvs(),
        }
    }
}

impl BaseAir<F> for ReducedSwirlSourceLeafVerifierPvsAir {
    fn width(&self) -> usize {
        1
    }
}
impl BaseAirWithPublicValues<F> for ReducedSwirlSourceLeafVerifierPvsAir {
    fn num_public_values(&self) -> usize {
        VerifierBasePvs::<u8>::width()
    }
}
impl PartitionedBaseAir<F> for ReducedSwirlSourceLeafVerifierPvsAir {}

impl<AB> Air<AB> for ReducedSwirlSourceLeafVerifierPvsAir
where
    AB: AirBuilder<F = F> + AirBuilderWithPublicValues + PairBuilder,
    AB::Var: Copy,
    AB::PublicVar: Copy,
    AB::Expr: From<AB::PublicVar>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("source-leaf PVS row")[0];
        let next = main.row_slice(1).expect("source-leaf PVS padding")[0];
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

pub trait ReducedSwirlSourceLeafComponents: Send + Sync + 'static {
    fn receipt_bus(&self) -> ReducedSwirlSourceReceiptBus;
    fn execution_bus(&self) -> ReducedSwirlExecutionBus;
    fn compress_bus(&self) -> Poseidon2CompressBus;
    fn component_digest(&self) -> Digest;
    fn component_air_count(&self) -> usize;
    fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>>;
}

pub struct ReducedSwirlSourceLeafCircuit<C: ReducedSwirlSourceLeafComponents> {
    pub binding: ReducedSwirlSourceLeafBinding,
    pub verifier_pvs_air: Arc<ReducedSwirlSourceLeafVerifierPvsAir>,
    pub vm_pvs_air: Arc<ReducedSwirlWrapperVmPvsAir>,
    pub boundary_air: Arc<ReducedSwirlSourceLeafBoundaryAir>,
    pub components: Arc<C>,
}

impl<C: ReducedSwirlSourceLeafComponents> ReducedSwirlSourceLeafCircuit<C> {
    pub fn new(
        binding: ReducedSwirlSourceLeafBinding,
        components: Arc<C>,
    ) -> Result<Self, &'static str> {
        binding.validate()?;
        if components.component_air_count() == 0
            || components.component_digest() != binding.source_component_digest
            || components.receipt_bus().index() == components.execution_bus().index()
        {
            return Err("source-leaf components");
        }
        Ok(Self {
            verifier_pvs_air: Arc::new(ReducedSwirlSourceLeafVerifierPvsAir::new(&binding)),
            vm_pvs_air: Arc::new(ReducedSwirlWrapperVmPvsAir::new(components.execution_bus())),
            boundary_air: Arc::new(ReducedSwirlSourceLeafBoundaryAir {
                receipt_bus: components.receipt_bus(),
                execution_bus: components.execution_bus(),
                compress_bus: components.compress_bus(),
            }),
            binding,
            components,
        })
    }
}

impl<SC, C> Circuit<SC> for ReducedSwirlSourceLeafCircuit<C>
where
    SC: StarkProtocolConfig<F = F>,
    C: ReducedSwirlSourceLeafComponents,
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

pub struct ReducedSwirlSourceLeafCoreTraces {
    pub verifier_pvs: RowMajorMatrix<F>,
    pub verifier_public_values: Vec<F>,
    pub vm_pvs: RowMajorMatrix<F>,
    pub vm_public_values: Vec<F>,
    pub boundary: RowMajorMatrix<F>,
    pub derived: ReducedSwirlSourceLeafDerived,
}

pub fn generate_reduced_swirl_source_leaf_core_traces(
    binding: &ReducedSwirlSourceLeafBinding,
    boundary_air: &ReducedSwirlSourceLeafBoundaryAir,
    record: &ReducedSwirlSourceLeafRecord,
) -> Result<ReducedSwirlSourceLeafCoreTraces, &'static str> {
    binding.validate()?;
    let (boundary, derived) = boundary_air.generate_trace(record)?;
    let verifier_public_values = binding.verifier_pvs().as_slice().to_vec();
    let vm = &derived.vm_pvs;
    let mut vm_public_values = Vec::with_capacity(VmPvs::<u8>::width());
    vm_public_values.extend_from_slice(&vm.program_commit);
    vm_public_values.extend([vm.initial_pc, vm.final_pc, vm.exit_code, vm.is_terminate]);
    vm_public_values.extend_from_slice(&vm.initial_root);
    vm_public_values.extend_from_slice(&vm.final_root);
    Ok(ReducedSwirlSourceLeafCoreTraces {
        verifier_pvs: RowMajorMatrix::new(vec![F::ONE, F::ZERO], 1),
        verifier_public_values,
        vm_pvs: RowMajorMatrix::new(vec![F::ONE, F::ZERO], 1),
        vm_public_values,
        boundary,
        derived,
    })
}

#[must_use]
pub fn reduced_swirl_source_chain_genesis(protocol_digest: Digest) -> Digest {
    let domain = core::array::from_fn(|index| match index {
        0 => F::from_u32(CHAIN_METADATA_TAG),
        1 => F::from_u32(REDUCED_SWIRL_SOURCE_LEAF_PROTOCOL_VERSION),
        2 => F::ZERO,
        3 => F::ZERO,
        _ => F::ZERO,
    });
    poseidon2_compress_with_capacity(protocol_digest, domain).0
}

fn chain_metadata(source_offset: u32, source_count: u32) -> Digest {
    core::array::from_fn(|index| match index {
        0 => F::from_u32(CHAIN_METADATA_TAG),
        1 => F::from_u32(REDUCED_SWIRL_SOURCE_LEAF_PROTOCOL_VERSION),
        2 => F::from_u32(source_offset),
        3 => F::from_u32(source_count),
        _ => F::ZERO,
    })
}

#[must_use]
pub fn reduced_swirl_source_boundary_metadata(source_index: u32, pc: F) -> Digest {
    core::array::from_fn(|index| match index {
        0 => F::from_u32(BOUNDARY_METADATA_TAG),
        1 => F::from_u32(REDUCED_SWIRL_SOURCE_LEAF_PROTOCOL_VERSION),
        2 => F::from_u32(source_index),
        3 => pc,
        _ => F::ZERO,
    })
}

fn chain_metadata_expr<AB: AirBuilder<F = F>>(
    source_offset: AB::Var,
    source_count: AB::Var,
) -> [AB::Expr; DIGEST_SIZE]
where
    AB::Var: Copy,
{
    core::array::from_fn(|index| match index {
        0 => AB::Expr::from_u32(CHAIN_METADATA_TAG),
        1 => AB::Expr::from_u32(REDUCED_SWIRL_SOURCE_LEAF_PROTOCOL_VERSION),
        2 => source_offset.into(),
        3 => source_count.into(),
        _ => AB::Expr::ZERO,
    })
}

pub(crate) fn reduced_swirl_source_boundary_metadata_expr<AB: AirBuilder<F = F>>(
    source_index: AB::Expr,
    pc: AB::Expr,
) -> [AB::Expr; DIGEST_SIZE] {
    core::array::from_fn(|index| match index {
        0 => AB::Expr::from_u32(BOUNDARY_METADATA_TAG),
        1 => AB::Expr::from_u32(REDUCED_SWIRL_SOURCE_LEAF_PROTOCOL_VERSION),
        2 => source_index.clone(),
        3 => pc.clone(),
        _ => AB::Expr::ZERO,
    })
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

fn one_row_selector<AB: AirBuilder>(builder: &mut AB, local: AB::Var, next: AB::Var)
where
    AB::Var: Copy,
{
    builder.assert_bool(local);
    builder.when_first_row().assert_one(local);
    builder.when_last_row().assert_zero(local);
    builder.when_transition().assert_eq(local - next, local);
}

fn is_zero_digest(digest: &Digest) -> bool {
    digest.iter().all(|value| *value == F::ZERO)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(seed: u32) -> Digest {
        core::array::from_fn(|index| F::from_u32(seed + index as u32 + 1))
    }

    #[test]
    fn derived_roots_bind_interval_vm_and_manifest_chain() {
        let receipt = ReducedSwirlSourceReceiptMessage {
            protocol_digest: digest(1),
            manifest_digest: digest(20),
            source_offset: F::from_u32(64),
            source_count: F::from_u32(7),
            program_commitment: digest(40),
            initial_pc: F::from_u32(12),
            initial_root: digest(60),
            final_pc: F::from_u32(19),
            final_root: digest(80),
            exit_code: F::from_u32(2),
            is_terminate: F::ZERO,
        };
        let record = ReducedSwirlSourceLeafRecord {
            receipt,
            chain_before: digest(100),
        };
        let derived = record.derive().unwrap();
        assert_eq!(derived.source_end, F::from_u32(71));
        assert_eq!(derived.vm_pvs.initial_pc, F::from_u32(64));
        assert_eq!(derived.vm_pvs.final_pc, F::from_u32(71));
        assert_eq!(derived.compression_inputs.len(), 8);

        let mut altered = record.clone();
        altered.receipt.manifest_digest[0] += F::ONE;
        let altered = altered.derive().unwrap();
        assert_ne!(derived.chain_after, altered.chain_after);
        assert_ne!(derived.final_boundary, altered.final_boundary);

        let mutations: [fn(&mut ReducedSwirlSourceLeafRecord); 7] = [
            |record: &mut ReducedSwirlSourceLeafRecord| record.receipt.source_offset += F::ONE,
            |record: &mut ReducedSwirlSourceLeafRecord| record.receipt.source_count += F::ONE,
            |record: &mut ReducedSwirlSourceLeafRecord| record.receipt.initial_pc += F::ONE,
            |record: &mut ReducedSwirlSourceLeafRecord| record.receipt.initial_root[0] += F::ONE,
            |record: &mut ReducedSwirlSourceLeafRecord| record.receipt.final_pc += F::ONE,
            |record: &mut ReducedSwirlSourceLeafRecord| record.receipt.final_root[0] += F::ONE,
            |record: &mut ReducedSwirlSourceLeafRecord| record.receipt.protocol_digest[0] += F::ONE,
        ];
        for mutate in mutations {
            let mut changed = record.clone();
            mutate(&mut changed);
            let changed = changed.derive().unwrap();
            assert_ne!(derived.vm_pvs.as_slice(), changed.vm_pvs.as_slice());
        }
    }

    #[test]
    fn adjacent_leaf_boundary_matches_only_with_exact_state_and_chain() {
        let protocol = digest(1);
        let program = digest(20);
        let middle_root = digest(40);
        let first = ReducedSwirlSourceLeafRecord {
            receipt: ReducedSwirlSourceReceiptMessage {
                protocol_digest: protocol,
                manifest_digest: digest(60),
                source_offset: F::ZERO,
                source_count: F::from_u32(4),
                program_commitment: program,
                initial_pc: F::from_u32(3),
                initial_root: digest(80),
                final_pc: F::from_u32(9),
                final_root: middle_root,
                exit_code: F::from_u32(2),
                is_terminate: F::ZERO,
            },
            chain_before: reduced_swirl_source_chain_genesis(protocol),
        };
        let first_derived = first.derive().unwrap();
        let second = ReducedSwirlSourceLeafRecord {
            receipt: ReducedSwirlSourceReceiptMessage {
                protocol_digest: protocol,
                manifest_digest: digest(100),
                source_offset: F::from_u32(4),
                source_count: F::from_u32(2),
                program_commitment: program,
                initial_pc: F::from_u32(9),
                initial_root: middle_root,
                final_pc: F::from_u32(11),
                final_root: digest(120),
                exit_code: F::ZERO,
                is_terminate: F::ONE,
            },
            chain_before: first_derived.chain_after,
        };
        let second_derived = second.derive().unwrap();
        assert_eq!(
            first_derived.final_boundary,
            second_derived.initial_boundary
        );

        let mut wrong = second;
        wrong.chain_before[0] += F::ONE;
        assert_ne!(
            first_derived.final_boundary,
            wrong.derive().unwrap().initial_boundary
        );
    }
}

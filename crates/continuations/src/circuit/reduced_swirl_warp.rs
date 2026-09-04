//! Fixed-capacity recursive boundary for native SWIRL-to-WARP accumulation.
//!
//! Unlike the legacy finite-WARP wrapper, this circuit does not allocate one
//! statement column block or one keyed component per WARP call. The private
//! statement consumes three constant-size authenticated receipts: the ordered
//! deferred-SWIRL source chain, the complete active-prefix VACC chain, and the
//! terminal same-root Decide/WHIR result. Component AIRs remain free to batch
//! all homogeneous records physically.

use std::sync::Arc;

use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, LookupBus},
    p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder},
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    AirRef, BaseAirWithPublicValues, PartitionedBaseAir, StarkProtocolConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, F};
use openvm_verify_stark_host::pvs::{VerifierBasePvs, VkCommit, VmPvs};
use serde::{Deserialize, Serialize};

use super::Circuit;

pub const REDUCED_SWIRL_WRAPPER_PROTOCOL_VERSION: u32 = 3;
pub const REDUCED_SWIRL_WRAPPER_MAX_SOURCES: u32 = 1024;
pub const REDUCED_SWIRL_WRAPPER_MAX_CALLS: u32 = 1023;
pub const REDUCED_SWIRL_WRAPPER_STATEMENT_WIDTH: usize = 101;

const PROTOCOL_VERSION: usize = 0;
const PROTOCOL_DIGEST: core::ops::Range<usize> = 1..9;
const RELATION_DIGEST: core::ops::Range<usize> = 9..17;
const WARP_INDEX_DIGEST: core::ops::Range<usize> = 17..25;
const TERMINAL_INDEX_DIGEST: core::ops::Range<usize> = 25..33;
const COMPONENT_DIGEST: core::ops::Range<usize> = 33..41;
const SCHEDULE_DIGEST: core::ops::Range<usize> = 41..49;
const MANIFEST_DIGEST: core::ops::Range<usize> = 49..57;
const SOURCE_COUNT: usize = 57;
const CALL_COUNT: usize = 58;
const PROGRAM_COMMITMENT: core::ops::Range<usize> = 59..67;
const INITIAL_PC: usize = 67;
const INITIAL_ROOT: core::ops::Range<usize> = 68..76;
const FINAL_PC: usize = 76;
const FINAL_ROOT: core::ops::Range<usize> = 77..85;
const FINAL_ACCUMULATOR_DIGEST: core::ops::Range<usize> = 85..93;
const FINAL_ACCUMULATOR_ROOT: core::ops::Range<usize> = 93..101;

const SOURCE_COUNT_BITS: usize = 11;
const CALL_COUNT_BITS: usize = 10;
const STATEMENT_TRACE_WIDTH: usize =
    1 + REDUCED_SWIRL_WRAPPER_STATEMENT_WIDTH + SOURCE_COUNT_BITS + CALL_COUNT_BITS;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReducedSwirlWrapperBinding {
    pub protocol_version: u32,
    pub protocol_digest: Digest,
    pub relation_digest: Digest,
    pub warp_index_digest: Digest,
    pub terminal_index_digest: Digest,
    pub verifier_component_digest: Digest,
    pub input_arity: u32,
    /// Commitment to the exact application VK cached trace under the direct
    /// wrapper PCS. The source component fixes the child VK and its symbolic
    /// DAG directly in its AIR inventory, so a second "source" commitment
    /// would be redundant and, worse, would not be consumed by any AIR.
    pub recursive_app_vk_commit: VkCommit<F>,
}

impl ReducedSwirlWrapperBinding {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.protocol_version != REDUCED_SWIRL_WRAPPER_PROTOCOL_VERSION {
            return Err("protocol version");
        }
        if self.input_arity < 2 || self.input_arity > 64 || !self.input_arity.is_power_of_two() {
            return Err("input arity");
        }
        if [
            self.protocol_digest,
            self.relation_digest,
            self.warp_index_digest,
            self.terminal_index_digest,
            self.verifier_component_digest,
        ]
        .iter()
        .any(is_zero_digest)
        {
            return Err("unset binding digest");
        }
        if is_zero_digest(&self.recursive_app_vk_commit.cached_commit)
            || is_zero_digest(&self.recursive_app_vk_commit.vk_pre_hash)
        {
            return Err("application VK lineage");
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

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReducedSwirlWrapperStatement {
    pub protocol_version: u32,
    pub protocol_digest: Digest,
    pub relation_digest: Digest,
    pub warp_index_digest: Digest,
    pub terminal_index_digest: Digest,
    pub verifier_component_digest: Digest,
    pub schedule_digest: Digest,
    pub manifest_digest: Digest,
    pub source_count: u32,
    pub call_count: u32,
    pub program_commitment: Digest,
    pub initial_pc: F,
    pub initial_root: Digest,
    pub final_pc: F,
    pub final_root: Digest,
    pub final_accumulator_digest: Digest,
    pub final_accumulator_root: Digest,
}

impl ReducedSwirlWrapperStatement {
    pub fn validate(&self, binding: &ReducedSwirlWrapperBinding) -> Result<(), &'static str> {
        binding.validate()?;
        if self.protocol_version != binding.protocol_version
            || self.protocol_digest != binding.protocol_digest
            || self.relation_digest != binding.relation_digest
            || self.warp_index_digest != binding.warp_index_digest
            || self.terminal_index_digest != binding.terminal_index_digest
            || self.verifier_component_digest != binding.verifier_component_digest
        {
            return Err("statement binding");
        }
        if self.source_count == 0 || self.source_count > REDUCED_SWIRL_WRAPPER_MAX_SOURCES {
            return Err("source count");
        }
        if self.call_count == 0 || self.call_count > REDUCED_SWIRL_WRAPPER_MAX_CALLS {
            return Err("call count");
        }
        if is_zero_digest(&self.schedule_digest)
            || is_zero_digest(&self.manifest_digest)
            || is_zero_digest(&self.program_commitment)
            || is_zero_digest(&self.final_accumulator_digest)
            || is_zero_digest(&self.final_accumulator_root)
        {
            return Err("unset statement digest");
        }
        Ok(())
    }

    #[must_use]
    pub fn vm_pvs(&self) -> VmPvs<F> {
        VmPvs {
            program_commit: self.program_commitment,
            initial_pc: self.initial_pc,
            final_pc: self.final_pc,
            exit_code: F::ZERO,
            is_terminate: F::ONE,
            initial_root: self.initial_root,
            final_root: self.final_root,
        }
    }

    #[must_use]
    pub fn to_fields(&self) -> [F; REDUCED_SWIRL_WRAPPER_STATEMENT_WIDTH] {
        let mut out = [F::ZERO; REDUCED_SWIRL_WRAPPER_STATEMENT_WIDTH];
        out[PROTOCOL_VERSION] = F::from_u32(self.protocol_version);
        out[PROTOCOL_DIGEST].copy_from_slice(&self.protocol_digest);
        out[RELATION_DIGEST].copy_from_slice(&self.relation_digest);
        out[WARP_INDEX_DIGEST].copy_from_slice(&self.warp_index_digest);
        out[TERMINAL_INDEX_DIGEST].copy_from_slice(&self.terminal_index_digest);
        out[COMPONENT_DIGEST].copy_from_slice(&self.verifier_component_digest);
        out[SCHEDULE_DIGEST].copy_from_slice(&self.schedule_digest);
        out[MANIFEST_DIGEST].copy_from_slice(&self.manifest_digest);
        out[SOURCE_COUNT] = F::from_u32(self.source_count);
        out[CALL_COUNT] = F::from_u32(self.call_count);
        out[PROGRAM_COMMITMENT].copy_from_slice(&self.program_commitment);
        out[INITIAL_PC] = self.initial_pc;
        out[INITIAL_ROOT].copy_from_slice(&self.initial_root);
        out[FINAL_PC] = self.final_pc;
        out[FINAL_ROOT].copy_from_slice(&self.final_root);
        out[FINAL_ACCUMULATOR_DIGEST].copy_from_slice(&self.final_accumulator_digest);
        out[FINAL_ACCUMULATOR_ROOT].copy_from_slice(&self.final_accumulator_root);
        out
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReducedSwirlWrapperRecord {
    pub statement: ReducedSwirlWrapperStatement,
}

pub struct ReducedSwirlWrapperCoreTraces {
    pub verifier_pvs: RowMajorMatrix<F>,
    pub verifier_public_values: Vec<F>,
    pub vm_pvs: RowMajorMatrix<F>,
    pub vm_public_values: Vec<F>,
    pub statement: RowMajorMatrix<F>,
}

pub fn generate_reduced_swirl_wrapper_core_traces(
    binding: &ReducedSwirlWrapperBinding,
    buses: ReducedSwirlWrapperReceiptBuses,
    record: &ReducedSwirlWrapperRecord,
) -> Result<ReducedSwirlWrapperCoreTraces, &'static str> {
    record.statement.validate(binding)?;
    let verifier_public_values = binding.verifier_pvs().as_slice().to_vec();
    let vm = record.statement.vm_pvs();
    let mut vm_public_values = Vec::with_capacity(VmPvs::<u8>::width());
    vm_public_values.extend_from_slice(&vm.program_commit);
    vm_public_values.extend([vm.initial_pc, vm.final_pc, vm.exit_code, vm.is_terminate]);
    vm_public_values.extend_from_slice(&vm.initial_root);
    vm_public_values.extend_from_slice(&vm.final_root);
    let statement =
        ReducedSwirlWrapperStatementAir::new(binding.clone(), buses)?.generate_trace(record)?;
    Ok(ReducedSwirlWrapperCoreTraces {
        verifier_pvs: RowMajorMatrix::new(vec![F::ONE, F::ZERO], 1),
        verifier_public_values,
        vm_pvs: RowMajorMatrix::new(vec![F::ONE, F::ZERO], 1),
        vm_public_values,
        statement,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlSourceReceiptMessage<T> {
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
}

impl<T: Clone> ReducedSwirlSourceReceiptMessage<T> {
    fn to_vec(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(46);
        out.extend_from_slice(&self.protocol_digest);
        out.extend_from_slice(&self.manifest_digest);
        out.extend([self.source_offset.clone(), self.source_count.clone()]);
        out.extend_from_slice(&self.program_commitment);
        out.push(self.initial_pc.clone());
        out.extend_from_slice(&self.initial_root);
        out.push(self.final_pc.clone());
        out.extend_from_slice(&self.final_root);
        out.extend([self.exit_code.clone(), self.is_terminate.clone()]);
        out
    }
}

#[derive(Clone, Debug)]
pub struct ReducedSwirlVaccReceiptMessage<T> {
    pub protocol_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub warp_index_digest: [T; DIGEST_SIZE],
    pub schedule_digest: [T; DIGEST_SIZE],
    pub manifest_digest: [T; DIGEST_SIZE],
    pub source_count: T,
    pub call_count: T,
    pub final_accumulator_digest: [T; DIGEST_SIZE],
    pub final_accumulator_root: [T; DIGEST_SIZE],
}

impl<T: Clone> ReducedSwirlVaccReceiptMessage<T> {
    fn to_vec(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(58);
        out.extend_from_slice(&self.protocol_digest);
        out.extend_from_slice(&self.relation_digest);
        out.extend_from_slice(&self.warp_index_digest);
        out.extend_from_slice(&self.schedule_digest);
        out.extend_from_slice(&self.manifest_digest);
        out.extend([self.source_count.clone(), self.call_count.clone()]);
        out.extend_from_slice(&self.final_accumulator_digest);
        out.extend_from_slice(&self.final_accumulator_root);
        out
    }
}

#[derive(Clone, Debug)]
pub struct ReducedSwirlTerminalReceiptMessage<T> {
    pub protocol_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub terminal_index_digest: [T; DIGEST_SIZE],
    pub verifier_component_digest: [T; DIGEST_SIZE],
    pub final_accumulator_digest: [T; DIGEST_SIZE],
    pub final_accumulator_root: [T; DIGEST_SIZE],
}

impl<T: Clone> ReducedSwirlTerminalReceiptMessage<T> {
    fn to_vec(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(48);
        out.extend_from_slice(&self.protocol_digest);
        out.extend_from_slice(&self.relation_digest);
        out.extend_from_slice(&self.terminal_index_digest);
        out.extend_from_slice(&self.verifier_component_digest);
        out.extend_from_slice(&self.final_accumulator_digest);
        out.extend_from_slice(&self.final_accumulator_root);
        out
    }
}

#[derive(Clone)]
pub struct ReducedSwirlExecutionMessage<T> {
    pub vm_pvs: VmPvs<T>,
}

impl<T: Clone> ReducedSwirlExecutionMessage<T> {
    fn to_vec(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(28);
        out.extend_from_slice(&self.vm_pvs.program_commit);
        out.extend([
            self.vm_pvs.initial_pc.clone(),
            self.vm_pvs.final_pc.clone(),
            self.vm_pvs.exit_code.clone(),
            self.vm_pvs.is_terminate.clone(),
        ]);
        out.extend_from_slice(&self.vm_pvs.initial_root);
        out.extend_from_slice(&self.vm_pvs.final_root);
        out
    }
}

macro_rules! typed_bus {
    ($name:ident, $message:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name(LookupBus);
        impl $name {
            #[must_use]
            pub const fn new(index: BusIndex) -> Self {
                Self(LookupBus::new(index))
            }
            #[must_use]
            pub const fn index(self) -> BusIndex {
                self.0.index
            }
            pub fn lookup_key<AB, T>(
                &self,
                builder: &mut AB,
                message: $message<T>,
                enabled: impl Into<AB::Expr>,
            ) where
                AB: InteractionBuilder,
                T: Into<AB::Expr> + Clone,
            {
                self.0.lookup_key(builder, message.to_vec(), enabled);
            }
            pub fn add_key_with_lookups<AB, T>(
                &self,
                builder: &mut AB,
                message: $message<T>,
                lookups: impl Into<AB::Expr>,
            ) where
                AB: InteractionBuilder,
                T: Into<AB::Expr> + Clone,
            {
                self.0
                    .add_key_with_lookups(builder, message.to_vec(), lookups);
            }
        }
    };
}

typed_bus!(
    ReducedSwirlSourceReceiptBus,
    ReducedSwirlSourceReceiptMessage
);
typed_bus!(ReducedSwirlVaccReceiptBus, ReducedSwirlVaccReceiptMessage);
typed_bus!(
    ReducedSwirlTerminalReceiptBus,
    ReducedSwirlTerminalReceiptMessage
);
typed_bus!(ReducedSwirlExecutionBus, ReducedSwirlExecutionMessage);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReducedSwirlWrapperReceiptBuses {
    pub source: ReducedSwirlSourceReceiptBus,
    pub vacc: ReducedSwirlVaccReceiptBus,
    pub terminal: ReducedSwirlTerminalReceiptBus,
    pub execution: ReducedSwirlExecutionBus,
}

impl ReducedSwirlWrapperReceiptBuses {
    #[must_use]
    pub const fn new(first: BusIndex) -> Self {
        Self {
            source: ReducedSwirlSourceReceiptBus::new(first),
            vacc: ReducedSwirlVaccReceiptBus::new(first + 1),
            terminal: ReducedSwirlTerminalReceiptBus::new(first + 2),
            execution: ReducedSwirlExecutionBus::new(first + 3),
        }
    }
    #[must_use]
    pub const fn next_bus_idx(self) -> BusIndex {
        self.execution.index() + 1
    }
}

#[derive(Clone)]
pub struct ReducedSwirlWrapperVerifierPvsAir {
    expected: VerifierBasePvs<F>,
}

impl ReducedSwirlWrapperVerifierPvsAir {
    #[must_use]
    pub fn new(binding: &ReducedSwirlWrapperBinding) -> Self {
        Self {
            expected: binding.verifier_pvs(),
        }
    }
}

impl BaseAir<F> for ReducedSwirlWrapperVerifierPvsAir {
    fn width(&self) -> usize {
        1
    }
}
impl BaseAirWithPublicValues<F> for ReducedSwirlWrapperVerifierPvsAir {
    fn num_public_values(&self) -> usize {
        VerifierBasePvs::<u8>::width()
    }
}
impl PartitionedBaseAir<F> for ReducedSwirlWrapperVerifierPvsAir {}
impl<AB> Air<AB> for ReducedSwirlWrapperVerifierPvsAir
where
    AB: AirBuilder<F = F> + AirBuilderWithPublicValues + PairBuilder,
    AB::Var: Copy,
    AB::PublicVar: Copy,
    AB::Expr: From<AB::PublicVar>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("wrapper PVS row")[0];
        let next = main.row_slice(1).expect("wrapper PVS padding")[0];
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

#[derive(Clone, Copy)]
pub struct ReducedSwirlWrapperVmPvsAir {
    bus: ReducedSwirlExecutionBus,
}
impl ReducedSwirlWrapperVmPvsAir {
    #[must_use]
    pub const fn new(bus: ReducedSwirlExecutionBus) -> Self {
        Self { bus }
    }
}
impl BaseAir<F> for ReducedSwirlWrapperVmPvsAir {
    fn width(&self) -> usize {
        1
    }
}
impl BaseAirWithPublicValues<F> for ReducedSwirlWrapperVmPvsAir {
    fn num_public_values(&self) -> usize {
        VmPvs::<u8>::width()
    }
}
impl PartitionedBaseAir<F> for ReducedSwirlWrapperVmPvsAir {}
impl<AB> Air<AB> for ReducedSwirlWrapperVmPvsAir
where
    AB: AirBuilder<F = F> + AirBuilderWithPublicValues + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
    AB::PublicVar: Copy,
    AB::Expr: From<AB::PublicVar>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("wrapper VmPvs row")[0];
        let next = main.row_slice(1).expect("wrapper VmPvs padding")[0];
        one_row_selector(builder, local, next);
        let pvs = builder.public_values();
        let v = |i: usize| Into::<AB::Expr>::into(pvs[i]);
        self.bus.add_key_with_lookups(
            builder,
            ReducedSwirlExecutionMessage {
                vm_pvs: VmPvs {
                    program_commit: core::array::from_fn(|i| v(i)),
                    initial_pc: v(DIGEST_SIZE),
                    final_pc: v(DIGEST_SIZE + 1),
                    exit_code: v(DIGEST_SIZE + 2),
                    is_terminate: v(DIGEST_SIZE + 3),
                    initial_root: core::array::from_fn(|i| v(DIGEST_SIZE + 4 + i)),
                    final_root: core::array::from_fn(|i| v(2 * DIGEST_SIZE + 4 + i)),
                },
            },
            local,
        );
    }
}

#[derive(Clone)]
pub struct ReducedSwirlWrapperStatementAir {
    binding: ReducedSwirlWrapperBinding,
    buses: ReducedSwirlWrapperReceiptBuses,
}

impl ReducedSwirlWrapperStatementAir {
    pub fn new(
        binding: ReducedSwirlWrapperBinding,
        buses: ReducedSwirlWrapperReceiptBuses,
    ) -> Result<Self, &'static str> {
        binding.validate()?;
        Ok(Self { binding, buses })
    }

    pub fn generate_trace(
        &self,
        record: &ReducedSwirlWrapperRecord,
    ) -> Result<RowMajorMatrix<F>, &'static str> {
        record.statement.validate(&self.binding)?;
        let mut values = vec![F::ZERO; 2 * STATEMENT_TRACE_WIDTH];
        values[0] = F::ONE;
        values[1..1 + REDUCED_SWIRL_WRAPPER_STATEMENT_WIDTH]
            .copy_from_slice(&record.statement.to_fields());
        write_bits(
            record.statement.source_count,
            &mut values[1 + REDUCED_SWIRL_WRAPPER_STATEMENT_WIDTH
                ..1 + REDUCED_SWIRL_WRAPPER_STATEMENT_WIDTH + SOURCE_COUNT_BITS],
        );
        write_bits(
            record.statement.call_count,
            &mut values[1 + REDUCED_SWIRL_WRAPPER_STATEMENT_WIDTH + SOURCE_COUNT_BITS
                ..STATEMENT_TRACE_WIDTH],
        );
        Ok(RowMajorMatrix::new(values, STATEMENT_TRACE_WIDTH))
    }
}

impl BaseAir<F> for ReducedSwirlWrapperStatementAir {
    fn width(&self) -> usize {
        STATEMENT_TRACE_WIDTH
    }
}
impl BaseAirWithPublicValues<F> for ReducedSwirlWrapperStatementAir {}
impl PartitionedBaseAir<F> for ReducedSwirlWrapperStatementAir {}
impl<AB> Air<AB> for ReducedSwirlWrapperStatementAir
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("wrapper statement row");
        let next = main.row_slice(1).expect("wrapper statement padding");
        let active = row[0];
        one_row_selector(builder, active, next[0]);
        for value in row.iter().skip(1) {
            builder.when(AB::Expr::ONE - active).assert_zero(*value);
        }
        let statement = &row[1..1 + REDUCED_SWIRL_WRAPPER_STATEMENT_WIDTH];
        let source_bits = &row[1 + REDUCED_SWIRL_WRAPPER_STATEMENT_WIDTH
            ..1 + REDUCED_SWIRL_WRAPPER_STATEMENT_WIDTH + SOURCE_COUNT_BITS];
        let call_bits = &row
            [1 + REDUCED_SWIRL_WRAPPER_STATEMENT_WIDTH + SOURCE_COUNT_BITS..STATEMENT_TRACE_WIDTH];
        for bit in source_bits.iter().chain(call_bits) {
            builder.assert_bool(*bit);
        }
        builder
            .when(active)
            .assert_eq(recompose_bits::<AB>(source_bits), statement[SOURCE_COUNT]);
        builder
            .when(active)
            .assert_eq(recompose_bits::<AB>(call_bits), statement[CALL_COUNT]);
        // 1 <= sources <= 1024: if bit 10 is set, every lower bit is zero.
        let source_lower = source_bits[..10]
            .iter()
            .fold(AB::Expr::ZERO, |acc, bit| acc + *bit);
        builder
            .when(active)
            .assert_zero(source_bits[10] * source_lower);
        // Non-zero source/call counts are enforced by the respective receipt
        // producers. The decompositions here enforce their fixed upper bounds.

        builder.when(active).assert_eq(
            statement[PROTOCOL_VERSION],
            AB::Expr::from_u32(self.binding.protocol_version),
        );
        for (range, expected) in [
            (PROTOCOL_DIGEST, self.binding.protocol_digest),
            (RELATION_DIGEST, self.binding.relation_digest),
            (WARP_INDEX_DIGEST, self.binding.warp_index_digest),
            (TERMINAL_INDEX_DIGEST, self.binding.terminal_index_digest),
            (COMPONENT_DIGEST, self.binding.verifier_component_digest),
        ] {
            for limb in 0..DIGEST_SIZE {
                builder.when(active).assert_eq(
                    statement[range.start + limb],
                    AB::Expr::from_u32(expected[limb].as_canonical_u32()),
                );
            }
        }
        let digest =
            |range: core::ops::Range<usize>| core::array::from_fn(|i| statement[range.start + i]);
        let digest_expr = |range: core::ops::Range<usize>| {
            core::array::from_fn(|i| AB::Expr::from(statement[range.start + i]))
        };
        let source_message = ReducedSwirlSourceReceiptMessage {
            protocol_digest: digest_expr(PROTOCOL_DIGEST),
            manifest_digest: digest_expr(MANIFEST_DIGEST),
            source_offset: AB::Expr::ZERO,
            source_count: statement[SOURCE_COUNT].into(),
            program_commitment: digest_expr(PROGRAM_COMMITMENT),
            initial_pc: statement[INITIAL_PC].into(),
            initial_root: digest_expr(INITIAL_ROOT),
            final_pc: statement[FINAL_PC].into(),
            final_root: digest_expr(FINAL_ROOT),
            exit_code: AB::Expr::ZERO,
            is_terminate: AB::Expr::ONE,
        };
        self.buses
            .source
            .lookup_key(builder, source_message, active);
        self.buses.vacc.lookup_key(
            builder,
            ReducedSwirlVaccReceiptMessage {
                protocol_digest: digest(PROTOCOL_DIGEST),
                relation_digest: digest(RELATION_DIGEST),
                warp_index_digest: digest(WARP_INDEX_DIGEST),
                schedule_digest: digest(SCHEDULE_DIGEST),
                manifest_digest: digest(MANIFEST_DIGEST),
                source_count: statement[SOURCE_COUNT],
                call_count: statement[CALL_COUNT],
                final_accumulator_digest: digest(FINAL_ACCUMULATOR_DIGEST),
                final_accumulator_root: digest(FINAL_ACCUMULATOR_ROOT),
            },
            active,
        );
        self.buses.terminal.lookup_key(
            builder,
            ReducedSwirlTerminalReceiptMessage {
                protocol_digest: digest(PROTOCOL_DIGEST),
                relation_digest: digest(RELATION_DIGEST),
                terminal_index_digest: digest(TERMINAL_INDEX_DIGEST),
                verifier_component_digest: digest(COMPONENT_DIGEST),
                final_accumulator_digest: digest(FINAL_ACCUMULATOR_DIGEST),
                final_accumulator_root: digest(FINAL_ACCUMULATOR_ROOT),
            },
            active,
        );
        self.buses.execution.lookup_key(
            builder,
            ReducedSwirlExecutionMessage {
                vm_pvs: VmPvs {
                    program_commit: digest(PROGRAM_COMMITMENT),
                    initial_pc: statement[INITIAL_PC],
                    final_pc: statement[FINAL_PC],
                    exit_code: next[0],
                    is_terminate: active,
                    initial_root: digest(INITIAL_ROOT),
                    final_root: digest(FINAL_ROOT),
                },
            },
            active,
        );
    }
}

pub trait ReducedSwirlWrapperVerifierComponents: Send + Sync + 'static {
    fn receipt_buses(&self) -> ReducedSwirlWrapperReceiptBuses;
    fn component_digest(&self) -> Digest;
    fn component_air_count(&self) -> usize;
    fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>>;
}

pub struct ReducedSwirlWrapperCircuit<C: ReducedSwirlWrapperVerifierComponents> {
    pub binding: ReducedSwirlWrapperBinding,
    pub verifier_pvs_air: Arc<ReducedSwirlWrapperVerifierPvsAir>,
    pub vm_pvs_air: Arc<ReducedSwirlWrapperVmPvsAir>,
    pub statement_air: Arc<ReducedSwirlWrapperStatementAir>,
    pub components: Arc<C>,
}

impl<C: ReducedSwirlWrapperVerifierComponents> ReducedSwirlWrapperCircuit<C> {
    pub fn new(
        binding: ReducedSwirlWrapperBinding,
        components: Arc<C>,
    ) -> Result<Self, &'static str> {
        binding.validate()?;
        if components.component_air_count() == 0
            || components.component_digest() != binding.verifier_component_digest
        {
            return Err("verifier components");
        }
        let buses = components.receipt_buses();
        Ok(Self {
            verifier_pvs_air: Arc::new(ReducedSwirlWrapperVerifierPvsAir::new(&binding)),
            vm_pvs_air: Arc::new(ReducedSwirlWrapperVmPvsAir::new(buses.execution)),
            statement_air: Arc::new(ReducedSwirlWrapperStatementAir::new(
                binding.clone(),
                buses,
            )?),
            binding,
            components,
        })
    }
}

impl<SC, C> Circuit<SC> for ReducedSwirlWrapperCircuit<C>
where
    SC: StarkProtocolConfig<F = F>,
    C: ReducedSwirlWrapperVerifierComponents,
{
    fn airs(&self) -> Vec<AirRef<SC>> {
        let component = self.components.airs::<SC>();
        assert_eq!(component.len(), self.components.component_air_count());
        [
            self.verifier_pvs_air.clone() as AirRef<SC>,
            self.vm_pvs_air.clone() as AirRef<SC>,
            self.statement_air.clone() as AirRef<SC>,
        ]
        .into_iter()
        .chain(component)
        .collect()
    }
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

fn write_bits(value: u32, bits: &mut [F]) {
    for (index, bit) in bits.iter_mut().enumerate() {
        *bit = F::from_bool(((value >> index) & 1) != 0);
    }
}

fn recompose_bits<AB: AirBuilder>(bits: &[AB::Var]) -> AB::Expr
where
    AB::Var: Copy,
{
    bits.iter()
        .enumerate()
        .fold(AB::Expr::ZERO, |acc, (index, bit)| {
            acc + AB::Expr::from_u32(1u32 << index) * *bit
        })
}

fn is_zero_digest(digest: &Digest) -> bool {
    digest.iter().all(|value| *value == F::ZERO)
}

#[cfg(test)]
#[path = "reduced_swirl_warp_tests.rs"]
mod tests;

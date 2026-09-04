use core::borrow::{Borrow, BorrowMut};

use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder},
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, F};
use openvm_verify_stark_host::pvs::{VerifierBasePvs, VmPvs};

use super::{
    offsets, FiniteWarpV3Binding, FiniteWarpV3CallReceiptMessage, FiniteWarpV3Error,
    FiniteWarpV3ExecutionMessage, FiniteWarpV3ManifestCallMessage,
    FiniteWarpV3ManifestReceiptMessage, FiniteWarpV3ReceiptBuses, FiniteWarpV3Record,
    FiniteWarpV3TerminalReceiptMessage, FINITE_WARP_V3_MAX_CALLS, FINITE_WARP_V3_MAX_INPUT_ARITY,
    FINITE_WARP_V3_PROTOCOL_VERSION, FINITE_WARP_V3_PUBLIC_STATEMENT_WIDTH,
};

const SOURCE_COUNT_BITS: usize = 8;
const FRESH_COUNT_BITS: usize = 7;
const ARITY_SELECTOR_COUNT: usize = 6;

#[repr(C)]
#[derive(AlignedBorrow)]
pub(crate) struct FiniteWarpV3CallCols<T> {
    pub active: T,
    pub call_index: T,
    pub source_start: T,
    pub source_count: T,
    pub input_arity: T,
    pub fresh_stacked_root: [T; DIGEST_SIZE],
    pub prior_accumulator_digest: [T; DIGEST_SIZE],
    pub output_accumulator_digest: [T; DIGEST_SIZE],
    pub source_count_bits: [T; FRESH_COUNT_BITS],
    pub arity_selectors: [T; ARITY_SELECTOR_COUNT],
}

#[repr(C)]
#[derive(AlignedBorrow)]
pub(crate) struct FiniteWarpV3StatementCols<T> {
    pub active: T,
    pub statement: [T; FINITE_WARP_V3_PUBLIC_STATEMENT_WIDTH],
    pub source_count_bits: [T; SOURCE_COUNT_BITS],
    pub calls: [FiniteWarpV3CallCols<T>; FINITE_WARP_V3_MAX_CALLS],
}

#[derive(Clone)]
pub struct FiniteWarpV3VerifierPvsAir {
    expected: VerifierBasePvs<F>,
}

impl FiniteWarpV3VerifierPvsAir {
    #[must_use]
    pub fn new(binding: &FiniteWarpV3Binding) -> Self {
        Self {
            expected: binding.verifier_pvs(),
        }
    }

    #[must_use]
    pub const fn expected(&self) -> VerifierBasePvs<F> {
        self.expected
    }
}

impl BaseAir<F> for FiniteWarpV3VerifierPvsAir {
    fn width(&self) -> usize {
        1
    }
}

impl BaseAirWithPublicValues<F> for FiniteWarpV3VerifierPvsAir {
    fn num_public_values(&self) -> usize {
        VerifierBasePvs::<u8>::width()
    }
}

impl PartitionedBaseAir<F> for FiniteWarpV3VerifierPvsAir {}

impl<AB> Air<AB> for FiniteWarpV3VerifierPvsAir
where
    AB: AirBuilder<F = F> + AirBuilderWithPublicValues + PairBuilder,
    AB::Var: Copy,
    AB::PublicVar: Copy,
    AB::Expr: From<AB::PublicVar>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("finite WARP v3 verifier PVS row")[0];
        let next = main
            .row_slice(1)
            .expect("finite WARP v3 verifier PVS padding")[0];
        builder.assert_bool(local);
        builder.when_first_row().assert_one(local);
        builder.when_last_row().assert_zero(local);
        builder.when_transition().assert_eq(local - next, local);
        let pvs = builder.public_values().to_vec();
        for (actual, expected) in pvs.iter().zip(self.expected.as_slice()) {
            builder.when(local).assert_eq(
                Into::<AB::Expr>::into(*actual),
                AB::Expr::from_u32(expected.as_canonical_u32()),
            );
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct FiniteWarpV3VmPvsAir {
    execution_bus: super::FiniteWarpV3ExecutionBus,
}

impl FiniteWarpV3VmPvsAir {
    #[must_use]
    pub const fn new(execution_bus: super::FiniteWarpV3ExecutionBus) -> Self {
        Self { execution_bus }
    }
}

impl BaseAir<F> for FiniteWarpV3VmPvsAir {
    fn width(&self) -> usize {
        1
    }
}

impl BaseAirWithPublicValues<F> for FiniteWarpV3VmPvsAir {
    fn num_public_values(&self) -> usize {
        VmPvs::<u8>::width()
    }
}

impl PartitionedBaseAir<F> for FiniteWarpV3VmPvsAir {}

impl<AB> Air<AB> for FiniteWarpV3VmPvsAir
where
    AB: AirBuilder<F = F> + AirBuilderWithPublicValues + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
    AB::PublicVar: Copy,
    AB::Expr: From<AB::PublicVar>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("finite WARP v3 VmPvs row")[0];
        let next = main.row_slice(1).expect("finite WARP v3 VmPvs padding")[0];
        builder.assert_bool(local);
        builder.when_first_row().assert_one(local);
        builder.when_last_row().assert_zero(local);
        builder.when_transition().assert_eq(local - next, local);

        let pvs = builder.public_values().to_vec();
        let value = |index: usize| Into::<AB::Expr>::into(pvs[index]);
        self.execution_bus.add_key_with_lookups(
            builder,
            FiniteWarpV3ExecutionMessage {
                vm_pvs: VmPvs {
                    program_commit: core::array::from_fn(|i| value(i)),
                    initial_pc: value(DIGEST_SIZE),
                    final_pc: value(DIGEST_SIZE + 1),
                    exit_code: value(DIGEST_SIZE + 2),
                    is_terminate: value(DIGEST_SIZE + 3),
                    initial_root: core::array::from_fn(|i| value(DIGEST_SIZE + 4 + i)),
                    final_root: core::array::from_fn(|i| value(2 * DIGEST_SIZE + 4 + i)),
                },
            },
            local,
        );
    }
}

#[derive(Clone, Debug)]
pub struct FiniteWarpV3StatementAir {
    pub binding: FiniteWarpV3Binding,
    pub buses: FiniteWarpV3ReceiptBuses,
}

impl FiniteWarpV3StatementAir {
    pub fn new(
        binding: FiniteWarpV3Binding,
        buses: FiniteWarpV3ReceiptBuses,
    ) -> Result<Self, FiniteWarpV3Error> {
        binding.validate()?;
        Ok(Self { binding, buses })
    }

    pub fn generate_trace(
        &self,
        record: &FiniteWarpV3Record,
    ) -> Result<FiniteWarpV3StatementTrace, FiniteWarpV3Error> {
        record.validate(&self.binding)?;
        let width = self.width();
        let mut values = vec![F::ZERO; 2 * width];
        let local: &mut FiniteWarpV3StatementCols<F> = values[..width].borrow_mut();
        local.active = F::ONE;
        local.statement = record.statement.to_fields();
        write_bits(record.statement.source_count, &mut local.source_count_bits);
        for (dst, src) in local.calls.iter_mut().zip(&record.calls) {
            dst.active = F::from_bool(src.active);
            dst.call_index = F::from_u32(src.call_index);
            dst.source_start = F::from_u32(src.source_start);
            dst.source_count = F::from_u32(src.source_count);
            dst.input_arity = F::from_u32(src.input_arity);
            dst.fresh_stacked_root = src.fresh_stacked_root;
            dst.prior_accumulator_digest = src.prior_accumulator_digest;
            dst.output_accumulator_digest = src.output_accumulator_digest;
            write_bits(src.source_count, &mut dst.source_count_bits);
            if src.active {
                let selector = src.input_arity.trailing_zeros() as usize - 1;
                dst.arity_selectors[selector] = F::ONE;
            }
        }
        Ok(FiniteWarpV3StatementTrace {
            matrix: RowMajorMatrix::new(values, width),
            public_values: Vec::new(),
        })
    }
}

impl BaseAir<F> for FiniteWarpV3StatementAir {
    fn width(&self) -> usize {
        core::mem::size_of::<FiniteWarpV3StatementCols<u8>>()
    }
}

impl BaseAirWithPublicValues<F> for FiniteWarpV3StatementAir {
    fn num_public_values(&self) -> usize {
        0
    }
}

impl PartitionedBaseAir<F> for FiniteWarpV3StatementAir {}

impl<AB> Air<AB> for FiniteWarpV3StatementAir
where
    AB: AirBuilder<F = F> + AirBuilderWithPublicValues + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
    AB::PublicVar: Copy,
    AB::Expr: From<AB::PublicVar>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("finite WARP v3 statement row");
        let next_row = main.row_slice(1).expect("finite WARP v3 statement padding");
        let local: &FiniteWarpV3StatementCols<AB::Var> = (*local_row).borrow();
        let next: &FiniteWarpV3StatementCols<AB::Var> = (*next_row).borrow();
        let active = Into::<AB::Expr>::into(local.active);
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_last_row().assert_zero(local.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.active);
        let inactive = AB::Expr::ONE - active.clone();
        for value in local.as_slice().iter().skip(1) {
            builder.when(inactive.clone()).assert_zero(*value);
        }

        let statement = |index: usize| Into::<AB::Expr>::into(local.statement[index]);
        builder.when(active.clone()).assert_eq(
            statement(offsets::PROTOCOL_VERSION),
            AB::Expr::from_u32(FINITE_WARP_V3_PROTOCOL_VERSION),
        );
        for (range, expected) in [
            (offsets::PROTOCOL_DIGEST, self.binding.protocol_digest),
            (offsets::RELATION_DIGEST, self.binding.relation_digest),
            (offsets::WARP_INDEX_DIGEST, self.binding.warp_index_digest),
            (
                offsets::TERMINAL_INDEX_DIGEST,
                self.binding.terminal_index_digest,
            ),
            (
                offsets::COMPONENT_DIGEST,
                self.binding.verifier_component_digest,
            ),
        ] {
            for limb in 0..DIGEST_SIZE {
                builder.when(active.clone()).assert_eq(
                    local.statement[range.start + limb],
                    AB::Expr::from_u32(expected[limb].as_canonical_u32()),
                );
            }
        }

        for bit in local.source_count_bits {
            builder.assert_bool(bit);
        }
        let source_count_from_bits = recompose_bits::<AB>(&local.source_count_bits);
        builder
            .when(active.clone())
            .assert_eq(source_count_from_bits, statement(offsets::SOURCE_COUNT));
        let lower_source_bits = local.source_count_bits[..SOURCE_COUNT_BITS - 1]
            .iter()
            .fold(AB::Expr::ZERO, |sum, bit| sum + *bit);
        builder.when(active.clone()).assert_zero(
            Into::<AB::Expr>::into(local.source_count_bits[SOURCE_COUNT_BITS - 1])
                * lower_source_bits,
        );

        let mut active_sum = AB::Expr::ZERO;
        let mut fresh_sum = AB::Expr::ZERO;
        for (position, call) in local.calls.iter().enumerate() {
            let call_active = Into::<AB::Expr>::into(call.active);
            builder.assert_bool(call.active);
            active_sum += call_active.clone();
            fresh_sum += Into::<AB::Expr>::into(call.source_count);
            if position == 0 {
                builder.when(active.clone()).assert_one(call.active);
                builder.when(active.clone()).assert_zero(call.source_start);
                for limb in call.prior_accumulator_digest {
                    builder.when(active.clone()).assert_zero(limb);
                }
            } else {
                let previous = &local.calls[position - 1];
                builder.when(active.clone()).assert_zero(
                    call_active.clone() * (AB::Expr::ONE - Into::<AB::Expr>::into(previous.active)),
                );
                builder.when(call_active.clone()).assert_eq(
                    call.source_start,
                    Into::<AB::Expr>::into(previous.source_start)
                        + Into::<AB::Expr>::into(previous.source_count),
                );
                for limb in 0..DIGEST_SIZE {
                    builder.when(call_active.clone()).assert_eq(
                        call.prior_accumulator_digest[limb],
                        previous.output_accumulator_digest[limb],
                    );
                }
            }
            let call_inactive = AB::Expr::ONE - call_active.clone();
            for value in call.as_slice().iter().skip(1) {
                builder.when(call_inactive.clone()).assert_zero(*value);
            }
            builder.when(active.clone()).assert_eq(
                call.call_index,
                AB::Expr::from_usize(position) * call_active.clone(),
            );
            for bit in call.source_count_bits {
                builder.assert_bool(bit);
            }
            builder.when(active.clone()).assert_eq(
                recompose_bits::<AB>(&call.source_count_bits),
                call.source_count,
            );
            let mut selector_sum = AB::Expr::ZERO;
            let mut selected_arity = AB::Expr::ZERO;
            for (selector_index, selector) in call.arity_selectors.iter().enumerate() {
                builder.assert_bool(*selector);
                selector_sum += *selector;
                selected_arity += AB::Expr::from_u32(1u32 << (selector_index + 1)) * *selector;
            }
            builder
                .when(active.clone())
                .assert_eq(selector_sum, call_active.clone());
            builder
                .when(active.clone())
                .assert_eq(selected_arity, call.input_arity);
            let prior_count = if position == 0 {
                AB::Expr::ZERO
            } else {
                call_active.clone()
            };
            builder.when(active.clone()).assert_eq(
                call.input_arity,
                Into::<AB::Expr>::into(call.source_count) + prior_count,
            );

            self.buses
                .call
                .lookup_key(builder, call_message::<AB>(local, call), call_active);
        }
        builder
            .when(active.clone())
            .assert_eq(active_sum, statement(offsets::CALL_COUNT));
        builder
            .when(active.clone())
            .assert_eq(fresh_sum, statement(offsets::SOURCE_COUNT));

        for limb in 0..DIGEST_SIZE {
            let final_from_calls = (Into::<AB::Expr>::into(local.calls[0].active)
                - Into::<AB::Expr>::into(local.calls[1].active))
                * local.calls[0].output_accumulator_digest[limb]
                + (Into::<AB::Expr>::into(local.calls[1].active)
                    - Into::<AB::Expr>::into(local.calls[2].active))
                    * local.calls[1].output_accumulator_digest[limb]
                + Into::<AB::Expr>::into(local.calls[2].active)
                    * local.calls[2].output_accumulator_digest[limb];
            builder.when(active.clone()).assert_eq(
                final_from_calls,
                local.statement[offsets::FINAL_ACCUMULATOR_DIGEST.start + limb],
            );
        }

        self.buses
            .manifest
            .lookup_key(builder, manifest_message::<AB>(local), active.clone());
        self.buses
            .terminal
            .lookup_key(builder, terminal_message::<AB>(local), active.clone());
        self.buses
            .execution
            .lookup_key(builder, execution_message::<AB>(local), active);
    }
}

#[derive(Clone, Debug)]
pub struct FiniteWarpV3StatementTrace {
    pub matrix: RowMajorMatrix<F>,
    pub public_values: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct FiniteWarpV3CoreTraces {
    pub verifier_pvs: RowMajorMatrix<F>,
    pub verifier_public_values: Vec<F>,
    pub vm_pvs: RowMajorMatrix<F>,
    pub vm_public_values: Vec<F>,
    pub statement: FiniteWarpV3StatementTrace,
}

pub fn generate_finite_warp_v3_core_traces(
    binding: &FiniteWarpV3Binding,
    buses: FiniteWarpV3ReceiptBuses,
    record: &FiniteWarpV3Record,
) -> Result<FiniteWarpV3CoreTraces, FiniteWarpV3Error> {
    record.validate(binding)?;
    let statement_air = FiniteWarpV3StatementAir::new(binding.clone(), buses)?;
    Ok(FiniteWarpV3CoreTraces {
        verifier_pvs: RowMajorMatrix::new(vec![F::ONE, F::ZERO], 1),
        verifier_public_values: binding.verifier_pvs().as_slice().to_vec(),
        vm_pvs: RowMajorMatrix::new(vec![F::ONE, F::ZERO], 1),
        vm_public_values: record.statement.vm_pvs().as_slice().to_vec(),
        statement: statement_air.generate_trace(record)?,
    })
}

fn write_bits<const N: usize>(value: u32, bits: &mut [F; N]) {
    for (index, bit) in bits.iter_mut().enumerate() {
        *bit = F::from_bool(((value >> index) & 1) == 1);
    }
}

fn recompose_bits<AB: AirBuilder<F = F>>(bits: &[AB::Var]) -> AB::Expr
where
    AB::Var: Copy,
{
    bits.iter()
        .enumerate()
        .fold(AB::Expr::ZERO, |sum, (index, bit)| {
            sum + AB::Expr::from_u32(1u32 << index) * *bit
        })
}

fn digest_from_statement<AB: AirBuilder<F = F>>(
    local: &FiniteWarpV3StatementCols<AB::Var>,
    range: core::ops::Range<usize>,
) -> [AB::Expr; DIGEST_SIZE]
where
    AB::Var: Copy,
{
    core::array::from_fn(|limb| local.statement[range.start + limb].into())
}

fn call_message<AB: AirBuilder<F = F>>(
    local: &FiniteWarpV3StatementCols<AB::Var>,
    call: &FiniteWarpV3CallCols<AB::Var>,
) -> FiniteWarpV3CallReceiptMessage<AB::Expr>
where
    AB::Var: Copy,
{
    FiniteWarpV3CallReceiptMessage {
        protocol_digest: digest_from_statement::<AB>(local, offsets::PROTOCOL_DIGEST),
        relation_digest: digest_from_statement::<AB>(local, offsets::RELATION_DIGEST),
        warp_index_digest: digest_from_statement::<AB>(local, offsets::WARP_INDEX_DIGEST),
        verifier_component_digest: digest_from_statement::<AB>(local, offsets::COMPONENT_DIGEST),
        schedule_digest: digest_from_statement::<AB>(local, offsets::SCHEDULE_DIGEST),
        manifest_digest: digest_from_statement::<AB>(local, offsets::MANIFEST_DIGEST),
        call_index: call.call_index.into(),
        source_start: call.source_start.into(),
        source_count: call.source_count.into(),
        input_arity: call.input_arity.into(),
        fresh_stacked_root: call.fresh_stacked_root.map(Into::into),
        prior_accumulator_digest: call.prior_accumulator_digest.map(Into::into),
        output_accumulator_digest: call.output_accumulator_digest.map(Into::into),
    }
}

fn manifest_message<AB: AirBuilder<F = F>>(
    local: &FiniteWarpV3StatementCols<AB::Var>,
) -> FiniteWarpV3ManifestReceiptMessage<AB::Expr>
where
    AB::Var: Copy,
{
    FiniteWarpV3ManifestReceiptMessage {
        protocol_digest: digest_from_statement::<AB>(local, offsets::PROTOCOL_DIGEST),
        relation_digest: digest_from_statement::<AB>(local, offsets::RELATION_DIGEST),
        warp_index_digest: digest_from_statement::<AB>(local, offsets::WARP_INDEX_DIGEST),
        verifier_component_digest: digest_from_statement::<AB>(local, offsets::COMPONENT_DIGEST),
        schedule_digest: digest_from_statement::<AB>(local, offsets::SCHEDULE_DIGEST),
        manifest_digest: digest_from_statement::<AB>(local, offsets::MANIFEST_DIGEST),
        source_count: local.statement[offsets::SOURCE_COUNT].into(),
        call_count: local.statement[offsets::CALL_COUNT].into(),
        program_commitment: digest_from_statement::<AB>(local, offsets::PROGRAM_COMMITMENT),
        initial_pc: local.statement[offsets::INITIAL_PC].into(),
        initial_root: digest_from_statement::<AB>(local, offsets::INITIAL_ROOT),
        final_pc: local.statement[offsets::FINAL_PC].into(),
        final_root: digest_from_statement::<AB>(local, offsets::FINAL_ROOT),
        final_accumulator_digest: digest_from_statement::<AB>(
            local,
            offsets::FINAL_ACCUMULATOR_DIGEST,
        ),
        calls: core::array::from_fn(|index| {
            let call = &local.calls[index];
            FiniteWarpV3ManifestCallMessage {
                active: call.active.into(),
                source_start: call.source_start.into(),
                source_count: call.source_count.into(),
                input_arity: call.input_arity.into(),
                fresh_stacked_root: call.fresh_stacked_root.map(Into::into),
            }
        }),
    }
}

fn terminal_message<AB: AirBuilder<F = F>>(
    local: &FiniteWarpV3StatementCols<AB::Var>,
) -> FiniteWarpV3TerminalReceiptMessage<AB::Expr>
where
    AB::Var: Copy,
{
    FiniteWarpV3TerminalReceiptMessage {
        protocol_digest: digest_from_statement::<AB>(local, offsets::PROTOCOL_DIGEST),
        relation_digest: digest_from_statement::<AB>(local, offsets::RELATION_DIGEST),
        terminal_index_digest: digest_from_statement::<AB>(local, offsets::TERMINAL_INDEX_DIGEST),
        verifier_component_digest: digest_from_statement::<AB>(local, offsets::COMPONENT_DIGEST),
        final_accumulator_digest: digest_from_statement::<AB>(
            local,
            offsets::FINAL_ACCUMULATOR_DIGEST,
        ),
        final_accumulator_root: digest_from_statement::<AB>(local, offsets::FINAL_ACCUMULATOR_ROOT),
    }
}

fn execution_message<AB: AirBuilder<F = F>>(
    local: &FiniteWarpV3StatementCols<AB::Var>,
) -> FiniteWarpV3ExecutionMessage<AB::Expr>
where
    AB::Var: Copy,
{
    FiniteWarpV3ExecutionMessage {
        vm_pvs: VmPvs {
            program_commit: digest_from_statement::<AB>(local, offsets::PROGRAM_COMMITMENT),
            initial_pc: local.statement[offsets::INITIAL_PC].into(),
            final_pc: local.statement[offsets::FINAL_PC].into(),
            exit_code: AB::Expr::ZERO,
            is_terminate: AB::Expr::ONE,
            initial_root: digest_from_statement::<AB>(local, offsets::INITIAL_ROOT),
            final_root: digest_from_statement::<AB>(local, offsets::FINAL_ROOT),
        },
    }
}

const _: () = assert!(FINITE_WARP_V3_MAX_INPUT_ARITY == 1 << ARITY_SELECTOR_COUNT);

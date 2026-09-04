use core::borrow::BorrowMut;
use std::{panic::AssertUnwindSafe, sync::Arc};

use openvm_stark_backend::{
    air_builders::{
        debug::{check_constraints, check_logup},
        symbolic::get_symbolic_builder,
    },
    interaction::{InteractionBuilder, SymbolicInteraction},
    keygen::types::TraceWidth,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    AirRef, AnyAir, BaseAirWithPublicValues, PartitionedBaseAir, StarkProtocolConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as NativeSC, Digest, DIGEST_SIZE, F,
};
use openvm_verify_stark_host::pvs::VkCommit;

use super::*;

fn digest(seed: u32) -> Digest {
    core::array::from_fn(|limb| F::from_u32(seed + limb as u32))
}

fn binding() -> FiniteWarpV3Binding {
    FiniteWarpV3Binding {
        protocol_version: FINITE_WARP_V3_PROTOCOL_VERSION,
        protocol_digest: digest(10),
        relation_digest: digest(20),
        warp_index_digest: digest(30),
        terminal_index_digest: digest(40),
        verifier_component_digest: digest(50),
        source_app_vk_commit: VkCommit {
            cached_commit: digest(60),
            vk_pre_hash: digest(70),
        },
        recursive_app_vk_commit: VkCommit {
            cached_commit: digest(65),
            vk_pre_hash: digest(70),
        },
    }
}

#[test]
fn recursive_pvs_uses_parent_pcs_cache_root_but_preserves_source_vk_prehash() {
    let binding = binding();
    let pvs = binding.verifier_pvs();
    assert_eq!(pvs.app_vk_commit, binding.recursive_app_vk_commit);
    assert_ne!(
        binding.recursive_app_vk_commit.cached_commit,
        binding.source_app_vk_commit.cached_commit
    );
    assert_eq!(
        binding.recursive_app_vk_commit.vk_pre_hash,
        binding.source_app_vk_commit.vk_pre_hash
    );

    let mut mismatched = binding;
    mismatched.recursive_app_vk_commit.vk_pre_hash[0] += F::ONE;
    assert!(matches!(
        mismatched.validate(),
        Err(FiniteWarpV3Error::UnsetSourceAppVk)
    ));
}

fn record_128_sources() -> FiniteWarpV3Record {
    let binding = binding();
    let first_output = digest(200);
    let second_output = digest(210);
    let final_output = digest(220);
    let final_root = digest(230);
    FiniteWarpV3Record {
        statement: FiniteWarpV3PublicStatement {
            protocol_version: FINITE_WARP_V3_PROTOCOL_VERSION,
            protocol_digest: binding.protocol_digest,
            relation_digest: binding.relation_digest,
            warp_index_digest: binding.warp_index_digest,
            terminal_index_digest: binding.terminal_index_digest,
            verifier_component_digest: binding.verifier_component_digest,
            schedule_digest: digest(80),
            manifest_digest: digest(90),
            source_count: 128,
            call_count: 3,
            program_commitment: digest(100),
            initial_pc: F::from_u32(4),
            initial_root: digest(110),
            final_pc: F::from_u32(8),
            final_root: digest(120),
            final_accumulator_digest: final_output,
            final_accumulator_root: final_root,
        },
        calls: [
            FiniteWarpV3CallReceipt {
                active: true,
                call_index: 0,
                source_start: 0,
                source_count: 64,
                input_arity: 64,
                fresh_stacked_root: digest(130),
                prior_accumulator_digest: [F::ZERO; DIGEST_SIZE],
                output_accumulator_digest: first_output,
            },
            FiniteWarpV3CallReceipt {
                active: true,
                call_index: 1,
                source_start: 64,
                source_count: 63,
                input_arity: 64,
                fresh_stacked_root: digest(140),
                prior_accumulator_digest: first_output,
                output_accumulator_digest: second_output,
            },
            FiniteWarpV3CallReceipt {
                active: true,
                call_index: 2,
                source_start: 127,
                source_count: 1,
                input_arity: 2,
                fresh_stacked_root: digest(150),
                prior_accumulator_digest: second_output,
                output_accumulator_digest: final_output,
            },
        ],
        terminal: FiniteWarpV3TerminalReceipt {
            final_accumulator_digest: final_output,
            final_accumulator_root: final_root,
        },
    }
}

fn constant<AB: AirBuilder<F = F>>(value: F) -> AB::Expr {
    AB::Expr::from_u32(value.as_canonical_u32())
}

fn constant_digest<AB: AirBuilder<F = F>>(value: Digest) -> [AB::Expr; DIGEST_SIZE] {
    value.map(constant::<AB>)
}

fn call_message<AB: AirBuilder<F = F>>(
    record: &FiniteWarpV3Record,
    call: &FiniteWarpV3CallReceipt,
) -> FiniteWarpV3CallReceiptMessage<AB::Expr> {
    let statement = &record.statement;
    FiniteWarpV3CallReceiptMessage {
        protocol_digest: constant_digest::<AB>(statement.protocol_digest),
        relation_digest: constant_digest::<AB>(statement.relation_digest),
        warp_index_digest: constant_digest::<AB>(statement.warp_index_digest),
        verifier_component_digest: constant_digest::<AB>(statement.verifier_component_digest),
        schedule_digest: constant_digest::<AB>(statement.schedule_digest),
        manifest_digest: constant_digest::<AB>(statement.manifest_digest),
        call_index: AB::Expr::from_u32(call.call_index),
        source_start: AB::Expr::from_u32(call.source_start),
        source_count: AB::Expr::from_u32(call.source_count),
        input_arity: AB::Expr::from_u32(call.input_arity),
        fresh_stacked_root: constant_digest::<AB>(call.fresh_stacked_root),
        prior_accumulator_digest: constant_digest::<AB>(call.prior_accumulator_digest),
        output_accumulator_digest: constant_digest::<AB>(call.output_accumulator_digest),
    }
}

fn manifest_message<AB: AirBuilder<F = F>>(
    record: &FiniteWarpV3Record,
) -> FiniteWarpV3ManifestReceiptMessage<AB::Expr> {
    let statement = &record.statement;
    FiniteWarpV3ManifestReceiptMessage {
        protocol_digest: constant_digest::<AB>(statement.protocol_digest),
        relation_digest: constant_digest::<AB>(statement.relation_digest),
        warp_index_digest: constant_digest::<AB>(statement.warp_index_digest),
        verifier_component_digest: constant_digest::<AB>(statement.verifier_component_digest),
        schedule_digest: constant_digest::<AB>(statement.schedule_digest),
        manifest_digest: constant_digest::<AB>(statement.manifest_digest),
        source_count: AB::Expr::from_u32(statement.source_count),
        call_count: AB::Expr::from_u32(statement.call_count),
        program_commitment: constant_digest::<AB>(statement.program_commitment),
        initial_pc: constant::<AB>(statement.initial_pc),
        initial_root: constant_digest::<AB>(statement.initial_root),
        final_pc: constant::<AB>(statement.final_pc),
        final_root: constant_digest::<AB>(statement.final_root),
        final_accumulator_digest: constant_digest::<AB>(statement.final_accumulator_digest),
        calls: core::array::from_fn(|index| {
            let call = &record.calls[index];
            FiniteWarpV3ManifestCallMessage {
                active: AB::Expr::from_bool(call.active),
                source_start: AB::Expr::from_u32(call.source_start),
                source_count: AB::Expr::from_u32(call.source_count),
                input_arity: AB::Expr::from_u32(call.input_arity),
                fresh_stacked_root: constant_digest::<AB>(call.fresh_stacked_root),
            }
        }),
    }
}

fn terminal_message<AB: AirBuilder<F = F>>(
    record: &FiniteWarpV3Record,
) -> FiniteWarpV3TerminalReceiptMessage<AB::Expr> {
    let statement = &record.statement;
    FiniteWarpV3TerminalReceiptMessage {
        protocol_digest: constant_digest::<AB>(statement.protocol_digest),
        relation_digest: constant_digest::<AB>(statement.relation_digest),
        terminal_index_digest: constant_digest::<AB>(statement.terminal_index_digest),
        verifier_component_digest: constant_digest::<AB>(statement.verifier_component_digest),
        final_accumulator_digest: constant_digest::<AB>(record.terminal.final_accumulator_digest),
        final_accumulator_root: constant_digest::<AB>(record.terminal.final_accumulator_root),
    }
}

#[derive(Clone, Debug)]
struct TestReceiptAuthorityAir {
    buses: FiniteWarpV3ReceiptBuses,
    record: FiniteWarpV3Record,
}

impl BaseAir<F> for TestReceiptAuthorityAir {
    fn width(&self) -> usize {
        1
    }
}

impl BaseAirWithPublicValues<F> for TestReceiptAuthorityAir {}
impl PartitionedBaseAir<F> for TestReceiptAuthorityAir {}

impl<AB> Air<AB> for TestReceiptAuthorityAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("receipt authority row")[0];
        let next = main.row_slice(1).expect("receipt authority padding")[0];
        builder.assert_bool(local);
        builder.when_first_row().assert_one(local);
        builder.when_last_row().assert_zero(local);
        builder.when_transition().assert_eq(local - next, local);
        for call in self.record.calls.iter().filter(|call| call.active) {
            self.buses.call.add_key_with_lookups(
                builder,
                call_message::<AB>(&self.record, call),
                local,
            );
        }
        self.buses.manifest.add_key_with_lookups(
            builder,
            manifest_message::<AB>(&self.record),
            local,
        );
        self.buses.terminal.add_key_with_lookups(
            builder,
            terminal_message::<AB>(&self.record),
            local,
        );
    }
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

#[test]
fn fixed_capacity_record_binds_actual_counts_and_chain() {
    let binding = binding();
    let record = record_128_sources();
    record.validate(&binding).unwrap();
    assert_eq!(record.statement.source_count, FINITE_WARP_V3_MAX_SOURCES);
    assert_eq!(
        record.statement.call_count as usize,
        FINITE_WARP_V3_MAX_CALLS
    );

    let decoded = FiniteWarpV3PublicStatement::try_from_slice(&record.statement.to_fields())
        .expect("statement round-trip");
    assert_eq!(decoded, record.statement);
    assert_eq!(
        decoded.vm_pvs().as_slice(),
        record.statement.vm_pvs().as_slice()
    );
}

#[test]
fn record_validation_rejects_count_and_receipt_substitution() {
    let binding = binding();
    let mut record = record_128_sources();
    record.statement.source_count = 127;
    assert!(matches!(
        record.validate(&binding),
        Err(FiniteWarpV3Error::SourceCount(128))
    ));

    let mut record = record_128_sources();
    record.calls[1].prior_accumulator_digest = digest(999);
    assert!(matches!(
        record.validate(&binding),
        Err(FiniteWarpV3Error::AccumulatorChain(1))
    ));

    let mut record = record_128_sources();
    record.calls[1] = FiniteWarpV3CallReceipt::inactive();
    assert!(matches!(
        record.validate(&binding),
        Err(FiniteWarpV3Error::NonPrefixCalls)
    ));

    let mut record = record_128_sources();
    record.terminal.final_accumulator_root = digest(998);
    assert!(matches!(
        record.validate(&binding),
        Err(FiniteWarpV3Error::TerminalLink)
    ));
}

#[test]
fn wrapper_core_constraints_and_receipt_buses_balance() {
    let binding = binding();
    let record = record_128_sources();
    let buses = FiniteWarpV3ReceiptBuses::new(900);
    let traces = generate_finite_warp_v3_core_traces(&binding, buses, &record).unwrap();
    let authority = TestReceiptAuthorityAir {
        buses,
        record: record.clone(),
    };
    let authority_trace = RowMajorMatrix::new(vec![F::ONE, F::ZERO], 1);
    let airs: Vec<AirRef<NativeSC>> = vec![
        Arc::new(FiniteWarpV3VerifierPvsAir::new(&binding)),
        Arc::new(FiniteWarpV3VmPvsAir::new(buses.execution)),
        Arc::new(FiniteWarpV3StatementAir::new(binding, buses).unwrap()),
        Arc::new(authority),
    ];
    let matrices = [
        &traces.verifier_pvs,
        &traces.vm_pvs,
        &traces.statement.matrix,
        &authority_trace,
    ];
    let public_values = [
        traces.verifier_public_values,
        traces.vm_public_values,
        traces.statement.public_values,
        Vec::new(),
    ];

    for ((air, matrix), pvs) in airs.iter().zip(matrices).zip(&public_values) {
        let preprocessed = BaseAir::<F>::preprocessed_trace(air.as_ref());
        check_constraints::<_, NativeSC>(
            air.as_ref(),
            &air.name(),
            &preprocessed.as_ref().map(RowMajorMatrix::as_view),
            &[matrix.as_view()],
            pvs,
        );
    }

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
    check_logup(
        &airs.iter().map(|air| air.name()).collect::<Vec<_>>(),
        &interactions,
        &preprocessed,
        &views,
        &public_values,
    );
}

#[test]
fn missing_receipt_components_fail_closed() {
    #[derive(Clone)]
    struct EmptyComponents(FiniteWarpV3ReceiptBuses);

    impl FiniteWarpV3VerifierComponents for EmptyComponents {
        fn receipt_buses(&self) -> FiniteWarpV3ReceiptBuses {
            self.0
        }

        fn component_digest(&self) -> Digest {
            binding().verifier_component_digest
        }

        fn component_air_count(&self) -> usize {
            0
        }

        fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
            Vec::new()
        }
    }

    assert!(matches!(
        FiniteWarpV3Circuit::new(
            binding(),
            Arc::new(EmptyComponents(FiniteWarpV3ReceiptBuses::new(920)))
        ),
        Err(FiniteWarpV3Error::MissingVerifierComponents)
    ));
}

#[test]
fn mutated_chain_is_rejected_by_air_constraints() {
    let binding = binding();
    let record = record_128_sources();
    let buses = FiniteWarpV3ReceiptBuses::new(940);
    let air = FiniteWarpV3StatementAir::new(binding, buses).unwrap();
    let mut trace = air.generate_trace(&record).unwrap();
    let width = trace.matrix.width();
    let local: &mut FiniteWarpV3StatementCols<F> = trace.matrix.values[..width].borrow_mut();
    local.calls[1].prior_accumulator_digest[0] = F::from_u32(123456);

    let rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        check_constraints::<_, NativeSC>(
            &air,
            "FiniteWarpV3StatementAir",
            &None,
            &[trace.matrix.as_view()],
            &trace.public_values,
        );
    }));
    assert!(rejected.is_err(), "mutated accumulator chain was accepted");
}

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
    AirRef, AnyAir, BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as NativeSC, Digest, DIGEST_SIZE, F,
};
use openvm_verify_stark_host::pvs::VkCommit;

use super::*;

fn digest(seed: u32) -> Digest {
    core::array::from_fn(|limb| F::from_u32(seed + limb as u32))
}

fn binding() -> ReducedSwirlWrapperBinding {
    ReducedSwirlWrapperBinding {
        protocol_version: REDUCED_SWIRL_WRAPPER_PROTOCOL_VERSION,
        protocol_digest: digest(10),
        relation_digest: digest(20),
        warp_index_digest: digest(30),
        terminal_index_digest: digest(40),
        verifier_component_digest: digest(50),
        input_arity: 8,
        recursive_app_vk_commit: VkCommit {
            cached_commit: digest(65),
            vk_pre_hash: digest(70),
        },
    }
}

fn record() -> ReducedSwirlWrapperRecord {
    let binding = binding();
    ReducedSwirlWrapperRecord {
        statement: ReducedSwirlWrapperStatement {
            protocol_version: binding.protocol_version,
            protocol_digest: binding.protocol_digest,
            relation_digest: binding.relation_digest,
            warp_index_digest: binding.warp_index_digest,
            terminal_index_digest: binding.terminal_index_digest,
            verifier_component_digest: binding.verifier_component_digest,
            schedule_digest: digest(80),
            manifest_digest: digest(90),
            source_count: 429,
            call_count: 62,
            program_commitment: digest(100),
            initial_pc: F::from_u32(4),
            initial_root: digest(110),
            final_pc: F::from_u32(8),
            final_root: digest(120),
            final_accumulator_digest: digest(130),
            final_accumulator_root: digest(140),
        },
    }
}

fn constant<AB: AirBuilder<F = F>>(value: F) -> AB::Expr {
    AB::Expr::from_u32(value.as_canonical_u32())
}

fn constant_digest<AB: AirBuilder<F = F>>(value: Digest) -> [AB::Expr; DIGEST_SIZE] {
    value.map(constant::<AB>)
}

#[derive(Clone, Debug)]
struct TestReceiptAuthorityAir {
    buses: ReducedSwirlWrapperReceiptBuses,
    record: ReducedSwirlWrapperRecord,
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
        let local = main.row_slice(0).expect("authority row")[0];
        let next = main.row_slice(1).expect("authority padding")[0];
        builder.assert_bool(local);
        builder.when_first_row().assert_one(local);
        builder.when_last_row().assert_zero(local);
        builder.when_transition().assert_eq(local - next, local);

        let s = &self.record.statement;
        self.buses.source.add_key_with_lookups(
            builder,
            ReducedSwirlSourceReceiptMessage {
                protocol_digest: constant_digest::<AB>(s.protocol_digest),
                manifest_digest: constant_digest::<AB>(s.manifest_digest),
                source_offset: AB::Expr::ZERO,
                source_count: AB::Expr::from_u32(s.source_count),
                program_commitment: constant_digest::<AB>(s.program_commitment),
                initial_pc: constant::<AB>(s.initial_pc),
                initial_root: constant_digest::<AB>(s.initial_root),
                final_pc: constant::<AB>(s.final_pc),
                final_root: constant_digest::<AB>(s.final_root),
                exit_code: AB::Expr::ZERO,
                is_terminate: AB::Expr::ONE,
            },
            local,
        );
        self.buses.vacc.add_key_with_lookups(
            builder,
            ReducedSwirlVaccReceiptMessage {
                protocol_digest: constant_digest::<AB>(s.protocol_digest),
                relation_digest: constant_digest::<AB>(s.relation_digest),
                warp_index_digest: constant_digest::<AB>(s.warp_index_digest),
                schedule_digest: constant_digest::<AB>(s.schedule_digest),
                manifest_digest: constant_digest::<AB>(s.manifest_digest),
                source_count: AB::Expr::from_u32(s.source_count),
                call_count: AB::Expr::from_u32(s.call_count),
                final_accumulator_digest: constant_digest::<AB>(s.final_accumulator_digest),
                final_accumulator_root: constant_digest::<AB>(s.final_accumulator_root),
            },
            local,
        );
        self.buses.terminal.add_key_with_lookups(
            builder,
            ReducedSwirlTerminalReceiptMessage {
                protocol_digest: constant_digest::<AB>(s.protocol_digest),
                relation_digest: constant_digest::<AB>(s.relation_digest),
                terminal_index_digest: constant_digest::<AB>(s.terminal_index_digest),
                verifier_component_digest: constant_digest::<AB>(s.verifier_component_digest),
                final_accumulator_digest: constant_digest::<AB>(s.final_accumulator_digest),
                final_accumulator_root: constant_digest::<AB>(s.final_accumulator_root),
            },
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
fn active_prefix_statement_balances_all_receipts() {
    let binding = binding();
    let record = record();
    let buses = ReducedSwirlWrapperReceiptBuses::new(1_100);
    let traces = generate_reduced_swirl_wrapper_core_traces(&binding, buses, &record).unwrap();
    let authority = TestReceiptAuthorityAir {
        buses,
        record: record.clone(),
    };
    let authority_trace = RowMajorMatrix::new(vec![F::ONE, F::ZERO], 1);
    let airs: Vec<AirRef<NativeSC>> = vec![
        Arc::new(ReducedSwirlWrapperVerifierPvsAir::new(&binding)),
        Arc::new(ReducedSwirlWrapperVmPvsAir::new(buses.execution)),
        Arc::new(ReducedSwirlWrapperStatementAir::new(binding, buses).unwrap()),
        Arc::new(authority),
    ];
    let matrices = [
        &traces.verifier_pvs,
        &traces.vm_pvs,
        &traces.statement,
        &authority_trace,
    ];
    let public_values = [
        traces.verifier_public_values,
        traces.vm_public_values,
        Vec::new(),
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
fn statement_rejects_invalid_capacity_and_binding() {
    let binding = binding();
    let mut bad = record();
    bad.statement.source_count = REDUCED_SWIRL_WRAPPER_MAX_SOURCES + 1;
    assert_eq!(bad.statement.validate(&binding), Err("source count"));

    let mut bad = record();
    bad.statement.relation_digest[0] += F::ONE;
    assert_eq!(bad.statement.validate(&binding), Err("statement binding"));
}

#[test]
fn mutated_count_bits_are_rejected() {
    let binding = binding();
    let buses = ReducedSwirlWrapperReceiptBuses::new(1_120);
    let air = ReducedSwirlWrapperStatementAir::new(binding.clone(), buses).unwrap();
    let mut trace = air.generate_trace(&record()).unwrap();
    trace.values[1 + REDUCED_SWIRL_WRAPPER_STATEMENT_WIDTH] += F::ONE;
    let rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        check_constraints::<_, NativeSC>(
            &air,
            "ReducedSwirlWrapperStatementAir",
            &None,
            &[trace.as_view()],
            &[],
        );
    }));
    assert!(rejected.is_err(), "mutated source-count bit was accepted");
}

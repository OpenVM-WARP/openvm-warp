use core::borrow::{Borrow, BorrowMut};
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
};

use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_recursion_circuit::{
    bus::{TranscriptBus, TranscriptBusMessage},
    native_warp::terminal::{
        FixedMultiAirCompleteWhirPrefixCursorBus, FixedMultiAirCompleteWhirPrefixCursorMessage,
        FixedMultiAirMappedTermBus, FixedMultiAirMappedTermMessage,
        FixedMultiAirStructuredClaimHeaderBus, FixedMultiAirStructuredClaimHeaderMessage,
        FixedMultiAirStructuredPointBus, FixedMultiAirStructuredPointMessage,
    },
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::{
        debug::{check_constraints, check_logup},
        symbolic::{
            get_symbolic_builder, symbolic_expression::SymbolicExpression, SymbolicConstraintsDag,
            SymbolicRapBuilder,
        },
    },
    hasher::MerkleHasher,
    interaction::{InteractionBuilder, SymbolicInteraction},
    keygen::types::{StarkVerifyingKey, StarkVerifyingParams, TraceWidth},
    native_warp::{
        DirectAirCodeClass, DirectAirPesatIndex, DirectAirPublicSchema,
        FixedMultiAirCompletePesatIndex, FixedMultiAirCompleteTerminalCircuitPlan,
        FixedMultiAirCompleteTerminalLinearizer, FINITE_COMPLETE_TERMINAL_RS_STATEMENT_TAG,
    },
    p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir},
    p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField32},
    transcript::TranscriptLog,
    warp_accum::TerminalConstrainedRsStatement,
    warp_pesat::{
        PrismalinearMappedColumnTerm, PrismalinearMappedColumnWeight,
        TerminalStructuredLinearClaim, TerminalWeightSpec,
    },
    BaseAirWithPublicValues, PartitionedBaseAir, StarkProtocolConfig, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config as SC, Digest, EF, F};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::*;

const START_TIDX: usize = 7;

#[derive(Clone, Copy)]
struct MinimalAir {
    interaction_sign: i8,
}

impl BaseAir<F> for MinimalAir {
    fn width(&self) -> usize {
        2
    }
}

impl BaseAirWithPublicValues<F> for MinimalAir {
    fn num_public_values(&self) -> usize {
        1
    }
}

impl PartitionedBaseAir<F> for MinimalAir {}

impl Air<SymbolicRapBuilder<F>> for MinimalAir {
    fn eval(&self, builder: &mut SymbolicRapBuilder<F>) {
        let main = builder.common_main().clone();
        let row = main.row_slice(0).expect("minimal complete-relation row");
        let value = row[0];
        let multiplicity = row[1];
        builder.assert_zero(value - builder.public_values()[0]);
        if self.interaction_sign != 0 {
            let count = if self.interaction_sign > 0 {
                SymbolicExpression::from(multiplicity)
            } else {
                -SymbolicExpression::from(multiplicity)
            };
            builder.push_interaction(9, [value, value], count, 1);
        }
    }
}

fn digest(seed: u32) -> Digest {
    core::array::from_fn(|coordinate| F::from_u32(seed + coordinate as u32))
}

fn verifying_key(air: &MinimalAir) -> StarkVerifyingKey<F, Digest> {
    let width = TraceWidth {
        preprocessed: None,
        cached_mains: Vec::new(),
        common_main: 2,
    };
    let symbolic = get_symbolic_builder(air, &width).constraints();
    StarkVerifyingKey {
        preprocessed_data: None,
        params: StarkVerifyingParams {
            width,
            num_public_values: 1,
            need_rot: false,
        },
        max_constraint_degree: symbolic.max_constraint_degree() as u8,
        symbolic_constraints: Arc::new(SymbolicConstraintsDag::from(symbolic)),
        is_required: true,
        unused_variables: Vec::new(),
    }
}

fn direct_relation<H>(hasher: &H, air: &MinimalAir, air_id: usize) -> DirectAirPesatIndex<F, Digest>
where
    H: MerkleHasher<F = F, Digest = Digest>,
{
    DirectAirPesatIndex::from_verifying_key(
        hasher,
        digest(1),
        air_id,
        1,
        &verifying_key(air),
        None,
        DirectAirPublicSchema {
            public_values_len: 1,
            boundary_values_len: 0,
            schema_digest: digest(100 + air_id as u32),
        },
        DirectAirCodeClass {
            log_message_len: 2,
            log_blowup: 1,
            log_codeword_len: 3,
            initial_folding_factor: 0,
            rows_per_query: 2,
        },
    )
    .expect("minimal direct relation")
}

/// Build the plan through the genuine backend relation and linearizer, rather
/// than manufacturing mapped-opening metadata inside the test.
fn complete_plan() -> Arc<FixedMultiAirCompleteTerminalCircuitPlan<F, Digest>> {
    let config = SC::default_from_params(SystemParams::new_for_testing(8));
    let hasher = config.hasher();
    let relations = [1i8, -1, 0]
        .into_iter()
        .enumerate()
        .map(|(ordinal, interaction_sign)| {
            direct_relation(hasher, &MinimalAir { interaction_sign }, 20 + ordinal)
        })
        .collect();
    let relation = FixedMultiAirCompletePesatIndex::from_direct_air_regions(
        hasher,
        digest(1),
        relations,
        DirectAirCodeClass {
            log_message_len: 5,
            log_blowup: 1,
            log_codeword_len: 6,
            initial_folding_factor: 0,
            rows_per_query: 2,
        },
    )
    .expect("minimal complete relation");
    Arc::new(
        FixedMultiAirCompleteTerminalLinearizer::new(&relation)
            .expect("minimal complete linearizer")
            .circuit_plan()
            .expect("backend-generated complete plan"),
    )
}

fn ef(seed: u32) -> EF {
    EF::from_basis_coefficients_fn(|coordinate| F::from_u32(seed + 17 * coordinate as u32))
}

fn statement(
    plan: &FixedMultiAirCompleteTerminalCircuitPlan<F, Digest>,
) -> TerminalConstrainedRsStatement<EF> {
    let terms = plan
        .mapped_openings
        .iter()
        .enumerate()
        .map(|(term, mapped)| PrismalinearMappedColumnTerm {
            block: mapped.block,
            l_skip: 0,
            barycentric_weights: vec![EF::ONE],
            folded_row_eq_point: (0..mapped.block.log_height)
                .map(|coordinate| ef(1_000 + term as u32 * 101 + coordinate as u32 * 7))
                .collect(),
            rotation: mapped.rotation,
            scale: ef(20_000 + term as u32 * 109),
        })
        .collect::<Vec<_>>();
    assert!(
        !terms.is_empty(),
        "complete plan must expose mapped openings"
    );
    TerminalConstrainedRsStatement {
        linearizer_claims: vec![TerminalStructuredLinearClaim {
            weight: TerminalWeightSpec::PrismalinearMappedColumns(PrismalinearMappedColumnWeight {
                log_message_len: usize::from(plan.metadata.code_class.log_message_len),
                terms,
            }),
            target: ef(30_000),
        }],
    }
}

#[derive(Clone, Debug, Default)]
struct NativeOffsets {
    tag: usize,
    claim_count: usize,
    kind: usize,
    metadata: Vec<usize>,
    points: Vec<usize>,
    scales: Vec<usize>,
    target: usize,
}

fn observe_u64(log: &mut TranscriptLog<F, [F; 16]>, offsets: &mut Vec<usize>, value: u64) {
    offsets.push(log.values().len());
    for shift in [0, 16, 32, 48] {
        log.push_observe(F::from_u32(((value >> shift) & 0xffff) as u32));
    }
}

fn observe_ext(log: &mut TranscriptLog<F, [F; 16]>, value: EF) -> usize {
    let start = log.values().len();
    log.extend_observe(value.as_basis_coefficients_slice());
    start
}

/// Independent copy of the native SDK wire specification. This deliberately
/// does not inspect `FiniteWarpV3RsStatementProfile::schedule`.
fn native_rs_statement_log(
    statement: &TerminalConstrainedRsStatement<EF>,
    start_tidx: usize,
) -> (TranscriptLog<F, [F; 16]>, NativeOffsets) {
    let mut log = TranscriptLog::default();
    for index in 0..start_tidx {
        log.push_observe(F::from_u32(40_000 + index as u32));
    }
    let mut offsets = NativeOffsets::default();
    let mut field = Vec::new();

    observe_u64(
        &mut log,
        &mut field,
        FINITE_COMPLETE_TERMINAL_RS_STATEMENT_TAG,
    );
    offsets.tag = field.pop().expect("tag offset");
    observe_u64(
        &mut log,
        &mut field,
        statement.linearizer_claims.len() as u64,
    );
    offsets.claim_count = field.pop().expect("claim-count offset");
    for claim in &statement.linearizer_claims {
        match &claim.weight {
            TerminalWeightSpec::Eq { point } => {
                observe_u64(&mut log, &mut field, 0);
                offsets.kind = field.pop().expect("kind offset");
                observe_u64(&mut log, &mut offsets.metadata, point.len() as u64);
                offsets
                    .points
                    .extend(point.iter().map(|&value| observe_ext(&mut log, value)));
            }
            TerminalWeightSpec::PrismalinearMappedColumns(mapped) => {
                observe_u64(&mut log, &mut field, 1);
                offsets.kind = field.pop().expect("kind offset");
                observe_u64(
                    &mut log,
                    &mut offsets.metadata,
                    mapped.log_message_len as u64,
                );
                observe_u64(&mut log, &mut offsets.metadata, mapped.terms.len() as u64);
                for term in &mapped.terms {
                    for value in [
                        term.block.start,
                        term.block.log_height,
                        term.l_skip,
                        term.rotation.offset(),
                        term.barycentric_weights.len(),
                    ] {
                        observe_u64(&mut log, &mut offsets.metadata, value as u64);
                    }
                    for &value in &term.barycentric_weights {
                        observe_ext(&mut log, value);
                    }
                    observe_u64(
                        &mut log,
                        &mut offsets.metadata,
                        term.folded_row_eq_point.len() as u64,
                    );
                    offsets.points.extend(
                        term.folded_row_eq_point
                            .iter()
                            .map(|&value| observe_ext(&mut log, value)),
                    );
                    offsets.scales.push(observe_ext(&mut log, term.scale));
                }
            }
        }
        offsets.target = observe_ext(&mut log, claim.target);
    }
    (log, offsets)
}

fn mapped(
    statement: &TerminalConstrainedRsStatement<EF>,
) -> (&PrismalinearMappedColumnWeight<EF>, EF) {
    let claim = &statement.linearizer_claims[0];
    let TerminalWeightSpec::PrismalinearMappedColumns(mapped) = &claim.weight else {
        panic!("test statement must be mapped-column")
    };
    (mapped, claim.target)
}

fn copy_ext(output: &mut [F; D_EF], value: EF) {
    output.copy_from_slice(value.as_basis_coefficients_slice());
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
struct RelayOracleCols<T> {
    transcript: T,
    source_cursor: T,
    end_cursor: T,
    source_header: T,
    output_header: T,
    source_term: T,
    output_term: T,
    source_point: T,
    output_point: T,
    tidx: T,
    transcript_value: T,
    claim: T,
    kind: T,
    log_message_len: T,
    term_count: T,
    point_len: T,
    term: T,
    block_start: T,
    log_height: T,
    l_skip: T,
    rotation: T,
    coordinate: T,
    value: [T; D_EF],
}

struct RelayOracleAir {
    transcript_bus: TranscriptBus,
    source_cursor_bus: FixedMultiAirCompleteWhirPrefixCursorBus,
    end_cursor_bus: FiniteWarpV3RsStatementEndBus,
    source_header_bus: FixedMultiAirStructuredClaimHeaderBus,
    source_term_bus: FixedMultiAirMappedTermBus,
    source_point_bus: FixedMultiAirStructuredPointBus,
    output_header_bus: FixedMultiAirStructuredClaimHeaderBus,
    output_term_bus: FixedMultiAirMappedTermBus,
    output_point_bus: FixedMultiAirStructuredPointBus,
}

impl BaseAir<F> for RelayOracleAir {
    fn width(&self) -> usize {
        RelayOracleCols::<F>::width()
    }
}

impl BaseAirWithPublicValues<F> for RelayOracleAir {}
impl PartitionedBaseAir<F> for RelayOracleAir {}

impl<AB> Air<AB> for RelayOracleAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("RS relay oracle row");
        let local: &RelayOracleCols<AB::Var> = (*row).borrow();
        for enabled in [
            local.transcript,
            local.source_cursor,
            local.end_cursor,
            local.source_header,
            local.output_header,
            local.source_term,
            local.output_term,
            local.source_point,
            local.output_point,
        ] {
            builder.assert_bool(enabled);
        }
        self.transcript_bus.send(
            builder,
            AB::Expr::ZERO,
            TranscriptBusMessage {
                tidx: local.tidx.into(),
                value: local.transcript_value.into(),
                is_sample: AB::Expr::ZERO,
            },
            local.transcript,
        );
        self.source_cursor_bus.send(
            builder,
            FixedMultiAirCompleteWhirPrefixCursorMessage {
                tidx: local.tidx.into(),
            },
            local.source_cursor,
        );
        self.end_cursor_bus.receive(
            builder,
            FiniteWarpV3RsStatementEndMessage {
                tidx: local.tidx.into(),
            },
            local.end_cursor,
        );

        let header = FixedMultiAirStructuredClaimHeaderMessage {
            claim: local.claim.into(),
            kind: local.kind.into(),
            log_message_len: local.log_message_len.into(),
            term_count: local.term_count.into(),
            point_len: local.point_len.into(),
            target: local.value.map(Into::into),
        };
        self.source_header_bus
            .send(builder, header.clone(), local.source_header);
        self.output_header_bus
            .receive(builder, header, local.output_header);

        let term = FixedMultiAirMappedTermMessage {
            claim: local.claim.into(),
            term: local.term.into(),
            block_start: local.block_start.into(),
            log_height: local.log_height.into(),
            l_skip: local.l_skip.into(),
            rotation: local.rotation.into(),
            scale: local.value.map(Into::into),
        };
        self.source_term_bus
            .send(builder, term.clone(), local.source_term);
        self.output_term_bus
            .receive(builder, term, local.output_term);

        let point = FixedMultiAirStructuredPointMessage {
            claim: local.claim.into(),
            term: local.term.into(),
            coordinate: local.coordinate.into(),
            value: local.value.map(Into::into),
        };
        self.source_point_bus
            .send(builder, point.clone(), local.source_point);
        self.output_point_bus
            .receive(builder, point, local.output_point);
    }
}

struct Buses {
    transcript: TranscriptBus,
    source_cursor: FixedMultiAirCompleteWhirPrefixCursorBus,
    end_cursor: FiniteWarpV3RsStatementEndBus,
    source_header: FixedMultiAirStructuredClaimHeaderBus,
    source_term: FixedMultiAirMappedTermBus,
    source_point: FixedMultiAirStructuredPointBus,
    output_header: FixedMultiAirStructuredClaimHeaderBus,
    output_term: FixedMultiAirMappedTermBus,
    output_point: FixedMultiAirStructuredPointBus,
}

fn buses() -> Buses {
    Buses {
        transcript: TranscriptBus::new(2_000),
        source_cursor: FixedMultiAirCompleteWhirPrefixCursorBus::new(2_001),
        end_cursor: FiniteWarpV3RsStatementEndBus::new(2_002),
        source_header: FixedMultiAirStructuredClaimHeaderBus::new(2_003),
        source_term: FixedMultiAirMappedTermBus::new(2_004),
        source_point: FixedMultiAirStructuredPointBus::new(2_005),
        output_header: FixedMultiAirStructuredClaimHeaderBus::new(2_006),
        output_term: FixedMultiAirMappedTermBus::new(2_007),
        output_point: FixedMultiAirStructuredPointBus::new(2_008),
    }
}

fn relay_air(
    profile: Arc<FiniteWarpV3RsStatementProfile>,
    buses: &Buses,
) -> FiniteWarpV3RsStatementAir {
    FiniteWarpV3RsStatementAir {
        profile,
        transcript_bus: buses.transcript,
        source_cursor_bus: buses.source_cursor,
        end_cursor_bus: buses.end_cursor,
        source_header_bus: buses.source_header,
        source_term_bus: buses.source_term,
        source_point_bus: buses.source_point,
        output_header_bus: buses.output_header,
        output_term_bus: buses.output_term,
        output_point_bus: buses.output_point,
    }
}

fn oracle_air(buses: &Buses) -> RelayOracleAir {
    RelayOracleAir {
        transcript_bus: buses.transcript,
        source_cursor_bus: buses.source_cursor,
        end_cursor_bus: buses.end_cursor,
        source_header_bus: buses.source_header,
        source_term_bus: buses.source_term,
        source_point_bus: buses.source_point,
        output_header_bus: buses.output_header,
        output_term_bus: buses.output_term,
        output_point_bus: buses.output_point,
    }
}

#[derive(Clone, Copy)]
enum SourceKind {
    Header,
    Term,
    Point,
}

struct OracleTrace {
    matrix: RowMajorMatrix<F>,
    header_row: usize,
    term_row: usize,
    point_row: usize,
    spare_row: usize,
}

fn oracle_trace(
    statement: &TerminalConstrainedRsStatement<EF>,
    log: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
    end_tidx: usize,
) -> OracleTrace {
    let (mapped, target) = mapped(statement);
    let point_count = mapped
        .terms
        .iter()
        .map(|term| term.folded_row_eq_point.len())
        .sum::<usize>();
    let used_rows = (end_tidx - start_tidx) + 2 + 1 + mapped.terms.len() + point_count;
    let height = (used_rows + 1).next_power_of_two();
    let width = RelayOracleCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    let mut row = 0;

    for (offset, &value) in log.values()[start_tidx..end_tidx].iter().enumerate() {
        let local: &mut RelayOracleCols<F> = values[row * width..(row + 1) * width].borrow_mut();
        local.transcript = F::ONE;
        local.tidx = F::from_usize(start_tidx + offset);
        local.transcript_value = value;
        row += 1;
    }
    {
        let local: &mut RelayOracleCols<F> = values[row * width..(row + 1) * width].borrow_mut();
        local.source_cursor = F::ONE;
        local.tidx = F::from_usize(start_tidx);
        row += 1;
    }
    {
        let local: &mut RelayOracleCols<F> = values[row * width..(row + 1) * width].borrow_mut();
        local.end_cursor = F::ONE;
        local.tidx = F::from_usize(end_tidx);
        row += 1;
    }
    let header_row = row;
    {
        let local: &mut RelayOracleCols<F> = values[row * width..(row + 1) * width].borrow_mut();
        local.source_header = F::ONE;
        local.output_header = F::ONE;
        local.claim = F::ZERO;
        local.kind = F::ZERO;
        local.log_message_len = F::from_usize(mapped.log_message_len);
        local.term_count = F::from_usize(mapped.terms.len());
        local.point_len = F::ZERO;
        copy_ext(&mut local.value, target);
        row += 1;
    }
    let term_row = row;
    for (ordinal, term) in mapped.terms.iter().enumerate() {
        let local: &mut RelayOracleCols<F> = values[row * width..(row + 1) * width].borrow_mut();
        local.source_term = F::ONE;
        local.output_term = F::ONE;
        local.claim = F::ZERO;
        local.term = F::from_usize(ordinal);
        local.block_start = F::from_usize(term.block.start);
        local.log_height = F::from_usize(term.block.log_height);
        local.l_skip = F::from_usize(term.l_skip);
        local.rotation = F::from_usize(term.rotation.offset());
        copy_ext(&mut local.value, term.scale);
        row += 1;
    }
    let point_row = row;
    for (term_ordinal, term) in mapped.terms.iter().enumerate() {
        for (coordinate, &point) in term.folded_row_eq_point.iter().enumerate() {
            let local: &mut RelayOracleCols<F> =
                values[row * width..(row + 1) * width].borrow_mut();
            local.source_point = F::ONE;
            local.output_point = F::ONE;
            local.claim = F::ZERO;
            local.term = F::from_usize(term_ordinal);
            local.coordinate = F::from_usize(coordinate);
            copy_ext(&mut local.value, point);
            row += 1;
        }
    }
    let spare_row = row;
    assert!(spare_row < height);
    OracleTrace {
        matrix: RowMajorMatrix::new(values, width),
        header_row,
        term_row,
        point_row,
        spare_row,
    }
}

fn symbolic_interactions<A>(air: &A) -> Vec<SymbolicInteraction<F>>
where
    A: Air<SymbolicRapBuilder<F>> + BaseAir<F> + BaseAirWithPublicValues<F> + PartitionedBaseAir<F>,
{
    let width = TraceWidth {
        preprocessed: BaseAir::<F>::preprocessed_trace(air).map(|trace| trace.width()),
        cached_mains: air.cached_main_widths(),
        common_main: air.common_main_width(),
    };
    get_symbolic_builder(air, &width).constraints().interactions
}

fn check_relay_constraints(
    air: &FiniteWarpV3RsStatementAir,
    trace: &FiniteWarpV3RsStatementTraceData,
) {
    check_constraints::<_, SC>(
        air,
        "FiniteWarpV3RsStatementAir",
        &None,
        &[trace.cached.as_view(), trace.common.as_view()],
        &[],
    );
}

fn check_oracle_constraints(air: &RelayOracleAir, trace: &RowMajorMatrix<F>) {
    check_constraints::<_, SC>(
        air,
        "independent RS relay oracle",
        &None,
        &[trace.as_view()],
        &[],
    );
}

fn check_full_balance(
    relay: &FiniteWarpV3RsStatementAir,
    relay_trace: &FiniteWarpV3RsStatementTraceData,
    oracle: &RelayOracleAir,
    oracle_trace: &RowMajorMatrix<F>,
) {
    check_logup(
        &[
            "RS statement relay".to_owned(),
            "independent source/output oracle".to_owned(),
        ],
        &[symbolic_interactions(relay), symbolic_interactions(oracle)],
        &[None, None],
        &[
            vec![relay_trace.cached.as_view(), relay_trace.common.as_view()],
            vec![oracle_trace.as_view()],
        ],
        &[Vec::new(), Vec::new()],
    );
}

fn assert_balance_rejected(
    relay: &FiniteWarpV3RsStatementAir,
    relay_trace: &FiniteWarpV3RsStatementTraceData,
    oracle: &RelayOracleAir,
    oracle_trace: &RowMajorMatrix<F>,
) {
    let rejected = catch_unwind(AssertUnwindSafe(|| {
        check_full_balance(relay, relay_trace, oracle, oracle_trace);
    }));
    assert!(rejected.is_err(), "forged RS relay multiset was accepted");
}

fn assert_trace_error(
    result: Result<FiniteWarpV3RsStatementTraceData, FiniteWarpV3RsStatementError>,
    expected: FiniteWarpV3RsStatementError,
) {
    assert_eq!(result.err(), Some(expected));
}

fn mutate_source_flag(cols: &mut RelayOracleCols<F>, kind: SourceKind, value: F) {
    match kind {
        SourceKind::Header => cols.source_header = value,
        SourceKind::Term => cols.source_term = value,
        SourceKind::Point => cols.source_point = value,
    }
}

fn source_row(trace: &OracleTrace, kind: SourceKind) -> usize {
    match kind {
        SourceKind::Header => trace.header_row,
        SourceKind::Term => trace.term_row,
        SourceKind::Point => trace.point_row,
    }
}

#[test]
fn genuine_plan_exact_native_sdk_transcript_is_accepted() {
    let plan = complete_plan();
    let statement = statement(&plan);
    let profile = Arc::new(FiniteWarpV3RsStatementProfile::new(plan).unwrap());
    let (log, _) = native_rs_statement_log(&statement, START_TIDX);
    assert_eq!(
        log.values().len() - START_TIDX,
        profile.observation_len(),
        "independent native wire and recursive schedule diverged"
    );
    generate_finite_warp_v3_rs_statement_trace(&profile, &statement, &log, START_TIDX, None)
        .expect(
        "exact native mapped-column statement was rejected: transcript kind must remain SDK kind 1",
    );
}

#[test]
fn current_relay_trace_satisfies_air_and_full_source_to_output_balance() {
    let plan = complete_plan();
    let statement = statement(&plan);
    let profile = Arc::new(FiniteWarpV3RsStatementProfile::new(plan).unwrap());
    let (log, _) = native_rs_statement_log(&statement, START_TIDX);
    let trace =
        generate_finite_warp_v3_rs_statement_trace(&profile, &statement, &log, START_TIDX, None)
            .unwrap();
    assert_eq!(trace.end_tidx, log.values().len());

    let buses = buses();
    let relay = relay_air(profile, &buses);
    let oracle = oracle_air(&buses);
    let oracle_trace = oracle_trace(&statement, &log, START_TIDX, trace.end_tidx);
    check_relay_constraints(&relay, &trace);
    check_oracle_constraints(&oracle, &oracle_trace.matrix);
    check_full_balance(&relay, &trace, &oracle, &oracle_trace.matrix);
}

#[test]
fn native_wire_rejects_tag_count_metadata_sample_and_order_mutations() {
    let plan = complete_plan();
    let statement = statement(&plan);
    let profile = FiniteWarpV3RsStatementProfile::new(plan).unwrap();
    let (log, offsets) = native_rs_statement_log(&statement, START_TIDX);
    let mut mutation_offsets = vec![offsets.tag, offsets.claim_count, offsets.kind];
    mutation_offsets.extend(offsets.metadata.iter().copied());
    for offset in mutation_offsets {
        let mut changed = log.clone();
        changed.values_mut()[offset] += F::ONE;
        assert_trace_error(
            generate_finite_warp_v3_rs_statement_trace(
                &profile, &statement, &changed, START_TIDX, None,
            ),
            FiniteWarpV3RsStatementError::Transcript,
        );
    }

    let mut sampled = log.clone();
    sampled.samples_mut()[offsets.scales[0]] = true;
    assert_trace_error(
        generate_finite_warp_v3_rs_statement_trace(
            &profile, &statement, &sampled, START_TIDX, None,
        ),
        FiniteWarpV3RsStatementError::Transcript,
    );

    let mut reordered = log;
    let left = offsets.points[0];
    let right = offsets.scales[0];
    assert_ne!(reordered.values()[left], reordered.values()[right]);
    reordered.values_mut().swap(left, right);
    assert_trace_error(
        generate_finite_warp_v3_rs_statement_trace(
            &profile, &statement, &reordered, START_TIDX, None,
        ),
        FiniteWarpV3RsStatementError::Transcript,
    );
}

#[test]
fn native_wire_rejects_point_scale_and_target_substitution() {
    let plan = complete_plan();
    let statement = statement(&plan);
    let profile = FiniteWarpV3RsStatementProfile::new(plan).unwrap();
    let (log, _) = native_rs_statement_log(&statement, START_TIDX);
    generate_finite_warp_v3_rs_statement_trace(&profile, &statement, &log, START_TIDX, None)
        .expect("unmodified current relay fixture");

    let mut point = statement.clone();
    let TerminalWeightSpec::PrismalinearMappedColumns(mapped) =
        &mut point.linearizer_claims[0].weight
    else {
        unreachable!()
    };
    mapped.terms[0].folded_row_eq_point[0] += EF::ONE;

    let mut scale = statement.clone();
    let TerminalWeightSpec::PrismalinearMappedColumns(mapped) =
        &mut scale.linearizer_claims[0].weight
    else {
        unreachable!()
    };
    mapped.terms[0].scale += EF::ONE;

    let mut target = statement.clone();
    target.linearizer_claims[0].target += EF::ONE;

    for changed in [point, scale, target] {
        assert_trace_error(
            generate_finite_warp_v3_rs_statement_trace(&profile, &changed, &log, START_TIDX, None),
            FiniteWarpV3RsStatementError::Transcript,
        );
    }
}

#[test]
fn relay_rejects_tidx_and_each_dynamic_ef4_carry_mutation() {
    let plan = complete_plan();
    let statement = statement(&plan);
    let profile = Arc::new(FiniteWarpV3RsStatementProfile::new(plan).unwrap());
    let (log, _) = native_rs_statement_log(&statement, START_TIDX);
    let trace =
        generate_finite_warp_v3_rs_statement_trace(&profile, &statement, &log, START_TIDX, None)
            .unwrap();
    let relay = relay_air(profile.clone(), &buses());

    let mut tidx = trace.clone();
    let common_width = tidx.common.width();
    let second: &mut FiniteWarpV3RsStatementCols<F> =
        tidx.common.values[common_width..2 * common_width].borrow_mut();
    second.tidx += F::ONE;
    assert!(catch_unwind(AssertUnwindSafe(|| {
        check_relay_constraints(&relay, &tidx)
    }))
    .is_err());

    for source_kind in [SOURCE_POINT, SOURCE_SCALE, SOURCE_TARGET] {
        let row = profile
            .schedule
            .iter()
            .position(|source| source.source() == source_kind && source.limb() == 1)
            .expect("dynamic EF4 continuation row");
        let mut changed = trace.clone();
        let local: &mut FiniteWarpV3RsStatementCols<F> =
            changed.common.values[row * common_width..(row + 1) * common_width].borrow_mut();
        local.value[0] += F::ONE;
        assert!(catch_unwind(AssertUnwindSafe(|| {
            check_relay_constraints(&relay, &changed)
        }))
        .is_err());
    }
}

#[test]
fn full_relay_balance_rejects_dropped_and_duplicated_source_values() {
    let plan = complete_plan();
    let statement = statement(&plan);
    let profile = Arc::new(FiniteWarpV3RsStatementProfile::new(plan).unwrap());
    let (log, _) = native_rs_statement_log(&statement, START_TIDX);
    let relay_trace =
        generate_finite_warp_v3_rs_statement_trace(&profile, &statement, &log, START_TIDX, None)
            .unwrap();
    let buses = buses();
    let relay = relay_air(profile, &buses);
    let oracle = oracle_air(&buses);
    let honest = oracle_trace(&statement, &log, START_TIDX, relay_trace.end_tidx);

    for kind in [SourceKind::Header, SourceKind::Term, SourceKind::Point] {
        let row = source_row(&honest, kind);
        let width = honest.matrix.width();

        let mut dropped = honest.matrix.clone();
        let local: &mut RelayOracleCols<F> =
            dropped.values[row * width..(row + 1) * width].borrow_mut();
        mutate_source_flag(local, kind, F::ZERO);
        check_oracle_constraints(&oracle, &dropped);
        assert_balance_rejected(&relay, &relay_trace, &oracle, &dropped);

        let mut duplicated = honest.matrix.clone();
        let source = duplicated.values[row * width..(row + 1) * width].to_vec();
        let spare =
            &mut duplicated.values[honest.spare_row * width..(honest.spare_row + 1) * width];
        spare.copy_from_slice(&source);
        let local: &mut RelayOracleCols<F> = spare.borrow_mut();
        local.transcript = F::ZERO;
        local.source_cursor = F::ZERO;
        local.end_cursor = F::ZERO;
        local.output_header = F::ZERO;
        local.output_term = F::ZERO;
        local.output_point = F::ZERO;
        mutate_source_flag(local, kind, F::ONE);
        check_oracle_constraints(&oracle, &duplicated);
        assert_balance_rejected(&relay, &relay_trace, &oracle, &duplicated);
    }
}

#[test]
fn generator_rejects_shifted_tidx_bad_height_and_field_wrap() {
    let plan = complete_plan();
    let statement = statement(&plan);
    let profile = FiniteWarpV3RsStatementProfile::new(plan).unwrap();
    let (log, _) = native_rs_statement_log(&statement, START_TIDX);

    for shifted in [START_TIDX - 1, START_TIDX + 1] {
        assert_trace_error(
            generate_finite_warp_v3_rs_statement_trace(&profile, &statement, &log, shifted, None),
            FiniteWarpV3RsStatementError::Transcript,
        );
    }

    let power_of_two = profile.observation_len().next_power_of_two();
    assert_trace_error(
        generate_finite_warp_v3_rs_statement_trace(
            &profile,
            &statement,
            &log,
            START_TIDX,
            Some(power_of_two + 1),
        ),
        FiniteWarpV3RsStatementError::Height,
    );
    assert_trace_error(
        generate_finite_warp_v3_rs_statement_trace(
            &profile,
            &statement,
            &log,
            START_TIDX,
            Some(profile.observation_len() - 1),
        ),
        FiniteWarpV3RsStatementError::Height,
    );

    let wrapping_start = F::ORDER_U32 as usize - profile.observation_len();
    assert_trace_error(
        generate_finite_warp_v3_rs_statement_trace(
            &profile,
            &statement,
            &TranscriptLog::default(),
            wrapping_start,
            None,
        ),
        FiniteWarpV3RsStatementError::Arithmetic,
    );
}

#[test]
fn symbolic_constraint_degree_remains_bounded() {
    let plan = complete_plan();
    let profile = Arc::new(FiniteWarpV3RsStatementProfile::new(plan).unwrap());
    let relay = relay_air(profile, &buses());
    let width = TraceWidth {
        preprocessed: None,
        cached_mains: relay.cached_main_widths(),
        common_main: relay.common_main_width(),
    };
    let degree = get_symbolic_builder(&relay, &width)
        .constraints()
        .max_constraint_degree();
    assert!(
        degree <= 3,
        "RS relay symbolic degree regressed to {degree}"
    );
}

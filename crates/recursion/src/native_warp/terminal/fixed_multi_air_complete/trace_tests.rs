use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
};

use openvm_stark_backend::{
    air_builders::{
        debug::{check_constraints, check_logup},
        inlined_cached::InlinedCachedAir,
        symbolic::{
            get_symbolic_builder, symbolic_expression::SymbolicExpression, SymbolicConstraintsDag,
            SymbolicRapBuilder,
        },
        PartitionedAirBuilder,
    },
    hasher::MerkleHasher,
    interaction::{InteractionBuilder, SymbolicInteraction},
    keygen::types::{StarkVerifyingKey, StarkVerifyingParams, TraceWidth},
    native_warp::{
        DirectAirCodeClass, DirectAirPesatIndex, DirectAirPesatInstance, DirectAirPublicSchema,
        FixedMultiAirCompletePesatIndex, FixedMultiAirCompletePesatInstance,
        FixedMultiAirCompleteSourceRegion, FixedMultiAirCompleteTerminalLinearizer,
        FixedMultiAirCompleteTerminalProof, NativeWarpChallenger,
    },
    transcript::{TranscriptHistory, TranscriptLog},
    warp_pesat::{
        evaluate_mle, AccumulatorInstance, StructuredTerminalPesatLinearizer,
        TerminalStructuredLinearClaim, TerminalWeightSpec,
    },
    BaseAirWithPublicValues, PartitionedBaseAir, StarkProtocolConfig, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, BabyBearPoseidon2Config as SC, Digest, DIGEST_SIZE, EF, F,
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::*;
use crate::bus::TranscriptBus;

/// Independent algebraic owner for the cursor immediately preceding the
/// complete nonlinear relation. This deliberately does not share witness
/// generation with the prefix AIR.
struct StatementStartCursorSenderAir {
    bus: FixedMultiAirCompleteStatementStartCursorBus,
}

impl BaseAir<F> for StatementStartCursorSenderAir {
    fn width(&self) -> usize {
        2
    }
}

impl BaseAirWithPublicValues<F> for StatementStartCursorSenderAir {}
impl PartitionedBaseAir<F> for StatementStartCursorSenderAir {}

impl<AB> Air<AB> for StatementStartCursorSenderAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main
            .row_slice(0)
            .expect("statement-start cursor sender row");
        let active = local[0];
        builder.assert_bool(active);
        self.bus.send(
            builder,
            FixedMultiAirCompleteStatementStartCursorMessage {
                start_tidx: local[1].into(),
            },
            active,
        );
    }
}

fn statement_start_cursor_sender_trace(start_tidx: usize) -> RowMajorMatrix<F> {
    RowMajorMatrix::new(vec![F::ONE, F::from_usize(start_tidx)], 2)
}

fn symbolic_interactions<A>(air: &A) -> Vec<SymbolicInteraction<F>>
where
    A: Air<SymbolicRapBuilder<F>> + BaseAir<F> + BaseAirWithPublicValues<F> + PartitionedBaseAir<F>,
{
    get_symbolic_builder(
        air,
        &TraceWidth {
            preprocessed: None,
            cached_mains: air.cached_main_widths(),
            common_main: air.common_main_width(),
        },
    )
    .constraints()
    .interactions
}

#[derive(Clone, Copy)]
struct TestAir {
    /// `1` sends, `-1` receives, and `0` has no interaction.
    interaction: i8,
    bus: u16,
}

impl BaseAir<F> for TestAir {
    fn width(&self) -> usize {
        2
    }
}

impl BaseAirWithPublicValues<F> for TestAir {
    fn num_public_values(&self) -> usize {
        1
    }
}

impl PartitionedBaseAir<F> for TestAir {}

impl Air<SymbolicRapBuilder<F>> for TestAir {
    fn eval(&self, builder: &mut SymbolicRapBuilder<F>) {
        let main = builder.common_main().clone();
        let local = main.row_slice(0).expect("test terminal row");
        let value = local[0];
        let multiplicity = local[1];
        builder.assert_zero(value - builder.public_values()[0]);
        if self.interaction != 0 {
            let count = SymbolicExpression::from(multiplicity);
            let count = if self.interaction > 0 { count } else { -count };
            builder.push_interaction(self.bus, [value, value], count, 1);
        }
    }
}

fn digest(value: u32) -> Digest {
    [F::from_u32(value); DIGEST_SIZE]
}

fn verifying_key(air: &TestAir) -> StarkVerifyingKey<F, Digest> {
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

fn direct_relation<H>(
    hasher: &H,
    air: &TestAir,
    air_id: usize,
    log_height: usize,
) -> DirectAirPesatIndex<F, Digest>
where
    H: MerkleHasher<F = F, Digest = Digest>,
{
    DirectAirPesatIndex::from_verifying_key(
        hasher,
        digest(1),
        air_id,
        log_height,
        &verifying_key(air),
        None,
        DirectAirPublicSchema {
            public_values_len: 1,
            boundary_values_len: 0,
            schema_digest: digest(100 + air_id as u32),
        },
        DirectAirCodeClass {
            log_message_len: u8::try_from(log_height + 1).expect("small direct message"),
            log_blowup: 1,
            log_codeword_len: u8::try_from(log_height + 2).expect("small direct codeword"),
            initial_folding_factor: 0,
            rows_per_query: 2,
        },
    )
    .expect("direct relation")
}

struct Fixture {
    relation: Arc<FixedMultiAirCompletePesatIndex<F, Digest>>,
    instance: AccumulatorInstance<EF, Digest>,
    proof: FixedMultiAirCompleteTerminalProof<EF>,
    claims: Vec<TerminalStructuredLinearClaim<EF>>,
    transcript: TranscriptLog<F, [F; 16]>,
}

fn fixture() -> Fixture {
    let config = SC::default_from_params(SystemParams::new_for_testing(8));
    let hasher = config.hasher();
    // Matching send/receive regions exercise the complete interaction and
    // global LogUp endpoint. The third region exercises the zero-round,
    // interaction-empty schedule.
    let shapes = [(3usize, 1usize, 1i8), (9, 1, -1), (12, 0, 0)];
    let direct = shapes
        .iter()
        .map(|&(air_id, log_height, interaction)| {
            direct_relation(
                hasher,
                &TestAir {
                    interaction,
                    bus: 7,
                },
                air_id,
                log_height,
            )
        })
        .collect::<Vec<_>>();
    let complete_log_message_len = 5u8;
    let relation = Arc::new(
        FixedMultiAirCompletePesatIndex::from_direct_air_regions(
            hasher,
            digest(1),
            direct,
            DirectAirCodeClass {
                log_message_len: complete_log_message_len,
                log_blowup: 1,
                log_codeword_len: complete_log_message_len + 1,
                initial_folding_factor: 0,
                rows_per_query: 2,
            },
        )
        .expect("complete relation"),
    );
    let sources = shapes
        .iter()
        .map(|&(air_id, log_height, interaction)| {
            let height = 1usize << log_height;
            FixedMultiAirCompleteSourceRegion {
                air_id: u32::try_from(air_id).expect("small AIR id"),
                public_values: vec![F::from_u32(7)],
                boundary_values: Vec::new(),
                common_trace_cells: (0..height)
                    .map(|_| F::from_u32(7))
                    .chain((0..height).map(|_| F::from_bool(interaction != 0)))
                    .collect(),
            }
        })
        .collect::<Vec<_>>();
    let direct_instances = shapes
        .iter()
        .map(|_| DirectAirPesatInstance {
            public_values: vec![F::from_u32(7)],
            boundary_values: Vec::new(),
        })
        .collect::<Vec<_>>();
    let public = FixedMultiAirCompletePesatInstance::from_alpha_beta(
        EF::from_u32(11),
        EF::from_u32(5),
        direct_instances,
        2,
    );
    let source_witness = relation
        .synthesize_witness(&public, &sources)
        .expect("complete witness");
    let message = relation
        .padded_witness::<EF>(&source_witness)
        .expect("padded witness");
    let explicit = relation
        .explicit_assignment(&public)
        .expect("explicit assignment");
    let constraints = relation
        .evaluate_reference(&public, &source_witness)
        .expect("reference relation");
    assert!(constraints.iter().all(|value| *value == EF::ZERO));
    let tau = (0..relation.pesat_shape().log_constraints)
        .map(|index| EF::from_usize(17 + index))
        .collect::<Vec<_>>();
    let eta = evaluate_mle(&constraints, &tau);
    let mut beta = tau;
    beta.extend(explicit.into_iter().map(EF::from));
    let instance = AccumulatorInstance {
        rt: digest(77),
        alpha: (0..usize::from(complete_log_message_len + 1))
            .map(|index| EF::from_usize(31 + index))
            .collect(),
        mu: EF::from_u32(41),
        beta,
        eta,
    };
    let linearizer =
        FixedMultiAirCompleteTerminalLinearizer::new(relation.as_ref()).expect("linearizer");
    let mut challenger = NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
    let (proof, claims) = linearizer
        .prove_from_reader(&instance, &message, &mut challenger)
        .expect("native complete terminal proof");
    let transcript = TranscriptHistory::into_log(challenger.into_inner());
    let mut verifier = NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
    assert_eq!(
        linearizer
            .verify_structured_terminal_claims(&instance, &proof, &mut verifier)
            .expect("native verification"),
        claims
    );
    let verifier_transcript = TranscriptHistory::into_log(verifier.into_inner());
    assert_eq!(transcript.values(), verifier_transcript.values());
    assert_eq!(transcript.samples(), verifier_transcript.samples());
    Fixture {
        relation,
        instance,
        proof,
        claims,
        transcript,
    }
}

fn circuit(fixture: &Fixture) -> FixedMultiAirCompleteTerminalCircuit {
    // These are consumers owned by the downstream terminal tail. Regional
    // claim, beta and eta consumers are added exactly by the complete owner.
    let external = FixedMultiAirCompleteExternalLookupCounts {
        binding: 1,
        alpha: vec![1; fixture.instance.alpha.len()],
        mu: 1,
        beta: vec![0; fixture.instance.beta.len()],
        eta: 0,
        local_claims: vec![0; fixture.relation.region_count()],
        interaction_claims: vec![0; fixture.relation.region_count()],
    };
    let profile = Arc::new(
        FixedMultiAirCompleteTerminalCircuitProfile::new(fixture.relation.clone(), external)
            .expect("complete terminal profile"),
    );
    FixedMultiAirCompleteTerminalCircuit::new(profile, TranscriptBus::new(1700), 0, 1701)
}

fn generate(
    circuit: &FixedMultiAirCompleteTerminalCircuit,
    fixture: &Fixture,
) -> Result<FixedMultiAirCompleteTerminalTraceData, FixedMultiAirCompleteTerminalTraceError> {
    circuit.generate_traces(FixedMultiAirCompleteTerminalTraceWitness {
        authenticated_root: fixture.instance.rt,
        instance: &fixture.instance,
        proof: &fixture.proof,
        transcript: &fixture.transcript,
        statement_start_tidx: 0,
    })
}

#[test]
fn complete_owner_matches_dynamic_native_terminal_claim_and_constraints() {
    let fixture = fixture();
    let circuit = circuit(&fixture);
    let trace = generate(&circuit, &fixture).expect("complete owner traces");
    let airs = circuit.airs::<SC>().expect("complete owner AIRs");
    assert_eq!(trace.roles, circuit.air_roles());
    assert_eq!(airs.len(), trace.traces.len());
    assert_eq!(fixture.claims.len(), 1);
    assert_eq!(trace.structured_target, fixture.claims[0].target);
    assert_eq!(
        trace.whir_prefix_start_tidx,
        fixture.transcript.values().len()
    );

    let mapped = match &fixture.claims[0].weight {
        TerminalWeightSpec::PrismalinearMappedColumns(mapped) => mapped,
        TerminalWeightSpec::Eq { .. } => panic!("complete claim must be mapped-column"),
    };
    assert!(mapped.terms.len() > 1);
    let mut rho_power = EF::ONE;
    for term in &mapped.terms {
        assert_eq!(term.scale, rho_power);
        rho_power *= trace.global_rho;
    }

    for ((air, partitioned), role) in airs.iter().zip(&trace.traces).zip(&trace.roles) {
        assert_eq!(
            air.cached_main_widths(),
            partitioned
                .cached_mains
                .iter()
                .map(Matrix::width)
                .collect::<Vec<_>>(),
            "cached widths for {role:?}"
        );
        assert_eq!(
            air.common_main_width(),
            partitioned.common_main.width(),
            "common width for {role:?}"
        );
        let mut main = partitioned
            .cached_mains
            .iter()
            .map(|matrix| matrix.as_view())
            .collect::<Vec<_>>();
        main.push(partitioned.common_main.as_view());
        let preprocessed = air.preprocessed_trace();
        let preprocessed = preprocessed.as_ref().map(|matrix| matrix.as_view());
        check_constraints::<_, SC>(air.as_ref(), &air.name(), &preprocessed, &main, &[]);
    }
}

#[test]
fn complete_owner_statement_start_cursor_balances_independent_sender_and_rejects_mutation() {
    let fixture = fixture();
    let circuit = circuit(&fixture);
    let trace = generate(&circuit, &fixture).expect("complete owner traces");
    let prefix = InlinedCachedAir::new(
        FixedMultiAirCompletePrefixDecompositionAirs::new(
            circuit.profile.prefix.clone(),
            circuit.proof_idx,
            circuit.buses.prefix(circuit.transcript_bus),
        )
        .transcript_prefix,
    );
    let prefix_ordinal = trace
        .roles
        .iter()
        .position(|role| *role == FixedMultiAirCompleteTerminalAirRole::TranscriptPrefix)
        .expect("complete transcript-prefix role");
    let prefix_trace = &trace.traces[prefix_ordinal];

    let sender = StatementStartCursorSenderAir {
        bus: circuit.buses.statement_start_cursor,
    };
    let sender_trace = statement_start_cursor_sender_trace(0);
    let cursor_bus_index = circuit.buses.statement_start_cursor.index();
    let prefix_cursor_interactions = symbolic_interactions(&prefix)
        .into_iter()
        .filter(|interaction| interaction.bus_index == cursor_bus_index)
        .collect::<Vec<_>>();
    let sender_interactions = symbolic_interactions(&sender);
    assert_eq!(prefix_cursor_interactions.len(), 1);
    assert_eq!(sender_interactions.len(), 1);

    let check_balance = |sender_trace: &RowMajorMatrix<F>| {
        check_logup(
            &[
                "complete nonlinear statement-start receiver".to_owned(),
                "independent statement-start sender".to_owned(),
            ],
            &[
                prefix_cursor_interactions.clone(),
                sender_interactions.clone(),
            ],
            &[None, None],
            &[
                vec![prefix_trace.common_main.as_view()],
                vec![sender_trace.as_view()],
            ],
            &[Vec::new(), Vec::new()],
        );
    };
    check_balance(&sender_trace);

    let mut forged_sender_trace = sender_trace.clone();
    forged_sender_trace.values[1] += F::ONE;
    let rejected = catch_unwind(AssertUnwindSafe(|| check_balance(&forged_sender_trace)));
    assert!(
        rejected.is_err(),
        "mutated independent statement-start cursor was accepted"
    );
}

#[test]
fn complete_owner_rejects_root_proof_and_transcript_mutations_without_panicking() {
    let fixture = fixture();
    let circuit = circuit(&fixture);

    let root_result = catch_unwind(AssertUnwindSafe(|| {
        circuit.generate_traces(FixedMultiAirCompleteTerminalTraceWitness {
            authenticated_root: digest(78),
            instance: &fixture.instance,
            proof: &fixture.proof,
            transcript: &fixture.transcript,
            statement_start_tidx: 0,
        })
    }));
    assert!(matches!(root_result, Ok(Err(_))));

    let mut mutated_proof = fixture.proof.clone();
    mutated_proof.local_claims[0] += EF::ONE;
    let proof_result = catch_unwind(AssertUnwindSafe(|| {
        circuit.generate_traces(FixedMultiAirCompleteTerminalTraceWitness {
            authenticated_root: fixture.instance.rt,
            instance: &fixture.instance,
            proof: &mutated_proof,
            transcript: &fixture.transcript,
            statement_start_tidx: 0,
        })
    }));
    assert!(matches!(proof_result, Ok(Err(_))));

    let mut mutated_transcript = fixture.transcript.clone();
    // This owner authenticates every observed terminal message against the
    // proof. Sampled values are instead constrained by the enclosing resumed
    // Transcript AIR, so that mutation belongs in the full-tail integration
    // test rather than this isolated nonlinear-owner test.
    let observed = mutated_transcript
        .samples()
        .iter()
        .position(|sampled| !*sampled)
        .expect("terminal transcript observation");
    mutated_transcript.values_mut()[observed] += F::ONE;
    let transcript_result = catch_unwind(AssertUnwindSafe(|| {
        circuit.generate_traces(FixedMultiAirCompleteTerminalTraceWitness {
            authenticated_root: fixture.instance.rt,
            instance: &fixture.instance,
            proof: &fixture.proof,
            transcript: &mutated_transcript,
            statement_start_tidx: 0,
        })
    }));
    assert!(matches!(transcript_result, Ok(Err(_))));
}

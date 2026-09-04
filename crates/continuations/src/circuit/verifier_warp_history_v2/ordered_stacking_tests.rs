use std::{panic::AssertUnwindSafe, sync::Arc};

use openvm_cpu_backend::CpuBackend;
use openvm_recursion_circuit::{
    bus::{ColumnClaimsBus, TranscriptBus, WhirOpeningPointBus},
    stacking::{
        OrderedStackingAuthority, OrderedStackingClaim, OrderedStackingClaimIdentity,
        OrderedStackingOpeningBus, OrderedStackingOutputBuses, OrderedStackingPreflight,
        OrderedStackingProfile, OrderedStackingReduction, OrderedStackingSourcePointBus,
        OrderedStackingTracegenError, OrderedStackingTranscriptSchedule, StackingModule,
    },
    system::{
        AirModule, BusIndexManager, BusInventory, GlobalCtxCpu, Preflight, TraceGenModule,
        VerifierSubCircuit,
    },
    whir::multi_constraint::{air::MultiConstraintStatementBuses, MultiConstraintWhirProfile},
};
use openvm_stark_backend::{
    air_builders::{
        debug::{check_constraints, check_logup},
        symbolic::{get_symbolic_builder, symbolic_expression::SymbolicExpression},
    },
    interaction::{InteractionBuilder, SymbolicInteraction},
    keygen::types::TraceWidth,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    poly_common::Squarable,
    proof::{column_openings_by_rot, Proof},
    prover::stacked_pcs::StackedLayout,
    test_utils::{test_system_params_small, SelfInteractionFixture, TestFixture},
    AirRef, AnyAir, BaseAirWithPublicValues, PartitionedBaseAir, StarkEngine,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, BabyBearPoseidon2Config as SC, BabyBearPoseidon2CpuEngine,
    DuplexSponge, D_EF, EF, F,
};

use super::{
    FixedSetupOpeningPointBusV2, FixedSetupOpeningPointMessageV2,
    OrderedStackingMultiConstraintAir, OrderedStackingMultiConstraintProfile,
    OrderedStackingMultiConstraintStatement, OrderedStackingSourcePointAdapterAirV2,
    OrderedStackingSourcePointProfileV2, OrderedStackingSourcePointRecordV2,
};

struct OrderedFixture {
    vk: openvm_stark_backend::keygen::types::MultiStarkVerifyingKey<SC>,
    proof: Proof<SC>,
    preflight: Preflight,
    profile: OrderedStackingProfile,
    claims: Vec<Vec<OrderedStackingClaim>>,
    opening_point: Vec<EF>,
}

const AUTHORITY_TRANSCRIPT_BUS: u16 = 4_000;
const AUTHORITY_COLUMN_CLAIMS_BUS: u16 = 4_001;
const AUTHORITY_SOURCE_POINT_BUS: u16 = 4_002;
const AUTHORITY_TRANSCRIPT_PROOF_IDX: usize = 0;

fn fixture() -> OrderedFixture {
    let mut params = test_system_params_small(2, 6, 3);
    params.whir.mu_pow_bits = 0;
    params.whir.folding_pow_bits = 0;
    params.whir.query_phase_pow_bits = 0;
    let engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(params);
    let source = SelfInteractionFixture {
        widths: vec![3, 4],
        log_height: 4,
        bus_index: 7,
    };
    let (vk, proof) = source.keygen_and_prove(&engine);
    let verifier = VerifierSubCircuit::<1>::new(std::sync::Arc::new(vk.clone()));
    let mut preflight = verifier.run_preflight(default_duplex_sponge_recorder(), &vk, &proof);

    // Authority reductions terminate after observing stacking openings. They do not run the
    // legacy per-proof mu/WHIR handoff; one multi-constraint WHIR follows the complete batch.
    let opening_count = proof
        .stacking_proof
        .stacking_openings
        .iter()
        .map(Vec::len)
        .sum::<usize>();
    preflight.stacking.stacking_batching_challenge = EF::ZERO;
    preflight.stacking.mu_pow_witness = F::ZERO;
    preflight.stacking.mu_pow_sample = F::ZERO;
    preflight.stacking.post_tidx = preflight.stacking.intermediate_tidx[2] + opening_count * D_EF;

    let sorted = &preflight.proof_shape.sorted_trace_vdata;
    let common_layout = StackedLayout::new(
        vk.inner.params.l_skip,
        vk.inner.params.log_stacked_height(),
        sorted
            .iter()
            .map(|(air_idx, vdata)| {
                (
                    vk.inner.per_air[*air_idx].params.width.common_main,
                    vdata.log_height,
                )
            })
            .collect(),
    )
    .expect("ordinary common-main stacking layout");
    assert_eq!(proof.stacking_proof.stacking_openings.len(), 1);

    let need_rot = sorted
        .iter()
        .map(|(air_idx, _)| vk.inner.per_air[*air_idx].params.need_rot)
        .collect::<Vec<_>>();
    let identities = sorted
        .iter()
        .enumerate()
        .flat_map(|(sort_idx, (air_idx, _))| {
            (0..vk.inner.per_air[*air_idx].params.width.common_main).map(move |col_idx| {
                OrderedStackingClaimIdentity {
                    sort_idx,
                    part_idx: 0,
                    col_idx,
                }
            })
        })
        .collect::<Vec<_>>();
    let claims = proof
        .batch_constraint_proof
        .column_openings
        .iter()
        .enumerate()
        .flat_map(|(sort_idx, parts)| {
            let need_rot = need_rot[sort_idx];
            column_openings_by_rot(&parts[0], need_rot).enumerate().map(
                move |(col_idx, (current, rotated))| OrderedStackingClaim {
                    identity: OrderedStackingClaimIdentity {
                        sort_idx,
                        part_idx: 0,
                        col_idx,
                    },
                    current,
                    rotated: need_rot.then_some(rotated),
                },
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(claims.len(), identities.len());
    let profile = OrderedStackingProfile::new(
        vk.inner.params.l_skip,
        vk.inner.params.n_stack,
        vk.inner.params.w_stack,
        vec![vec![common_layout]],
        vec![vec![need_rot]],
        vec![vec![identities]],
    )
    .expect("ordered profile")
    .with_transcript_schedule(OrderedStackingTranscriptSchedule {
        first_claim_observation_tidx: preflight.batch_constraint.tidx_before_column_openings,
        inter_reduction_prefix_len: 5,
    })
    .expect("ordered transcript schedule");
    let opening_point = preflight.batch_constraint.sumcheck_rnd.clone();
    OrderedFixture {
        vk,
        proof,
        preflight,
        profile,
        claims: vec![claims],
        opening_point,
    }
}

fn modules(fixture: &OrderedFixture) -> (StackingModule, StackingModule) {
    let mut legacy_manager = BusIndexManager::new();
    let legacy_inventory = BusInventory::new(&mut legacy_manager);
    let legacy = StackingModule::new(&fixture.vk, &mut legacy_manager, legacy_inventory);

    let mut ordered_manager = BusIndexManager::new();
    let mut ordered_inventory = BusInventory::new(&mut ordered_manager);
    ordered_inventory.transcript_bus = TranscriptBus::new(AUTHORITY_TRANSCRIPT_BUS);
    let authority_claims = ColumnClaimsBus::new(AUTHORITY_COLUMN_CLAIMS_BUS);
    let mut ordered = StackingModule::new(&fixture.vk, &mut ordered_manager, ordered_inventory);
    ordered
        .configure_ordered_reductions(
            fixture.profile.clone(),
            OrderedStackingAuthority {
                column_claims: authority_claims,
                source_opening_point: OrderedStackingSourcePointBus::new(
                    AUTHORITY_SOURCE_POINT_BUS,
                ),
                transcript_proof_idx: AUTHORITY_TRANSCRIPT_PROOF_IDX,
            },
        )
        .expect("configure ordered reductions");
    (legacy, ordered)
}

fn symbolic_interactions(air: &dyn AnyAir<SC>) -> Vec<SymbolicInteraction<F>> {
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
fn ordered_reductions_share_one_verifier_owned_transcript_key() {
    let fixture = fixture();
    let (_, ordered) = modules(&fixture);
    let airs = ordered.airs::<SC>();
    for (air_idx, air) in airs.iter().take(4).enumerate() {
        let transcript_interactions = symbolic_interactions(air.as_ref())
            .into_iter()
            .filter(|interaction| interaction.bus_index == AUTHORITY_TRANSCRIPT_BUS)
            .collect::<Vec<_>>();
        assert!(
            !transcript_interactions.is_empty(),
            "stacking AIR {air_idx} has no global transcript interaction"
        );
        for interaction in transcript_interactions {
            assert!(matches!(
                &interaction.message[0],
                SymbolicExpression::Constant(value)
                    if *value == F::from_usize(AUTHORITY_TRANSCRIPT_PROOF_IDX)
            ));
        }
    }

    let claim_interactions = symbolic_interactions(airs[0].as_ref())
        .into_iter()
        .filter(|interaction| interaction.bus_index == AUTHORITY_COLUMN_CLAIMS_BUS)
        .collect::<Vec<_>>();
    assert!(!claim_interactions.is_empty());
    for interaction in claim_interactions {
        assert!(
            !matches!(&interaction.message[0], SymbolicExpression::Constant(_)),
            "ColumnClaims must remain keyed by the constrained reduction index"
        );
    }
}

fn check_selected_bus_composition(
    airs: &[AirRef<SC>],
    matrices: &[RowMajorMatrix<F>],
    selected_buses: &[u16],
) {
    let preprocessed_owned = airs
        .iter()
        .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
        .collect::<Vec<_>>();
    for ((air, matrix), preprocessed) in airs.iter().zip(matrices).zip(&preprocessed_owned) {
        check_constraints::<_, SC>(
            air.as_ref(),
            &air.name(),
            &preprocessed.as_ref().map(RowMajorMatrix::as_view),
            &[matrix.as_view()],
            &[],
        );
    }
    let interactions = airs
        .iter()
        .map(|air| {
            symbolic_interactions(air.as_ref())
                .into_iter()
                .filter(|interaction| selected_buses.contains(&interaction.bus_index))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let preprocessed = preprocessed_owned
        .iter()
        .map(|matrix| matrix.as_ref().map(RowMajorMatrix::as_view))
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
        &vec![Vec::new(); airs.len()],
    );
}

#[derive(Clone, Copy, Debug)]
struct FixedSourcePointTestAir(FixedSetupOpeningPointBusV2);

impl BaseAir<F> for FixedSourcePointTestAir {
    fn width(&self) -> usize {
        4 + D_EF
    }
}

impl BaseAirWithPublicValues<F> for FixedSourcePointTestAir {}
impl PartitionedBaseAir<F> for FixedSourcePointTestAir {}

impl<AB> Air<AB> for FixedSourcePointTestAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("fixed source-point test row");
        builder.assert_bool(row[0]);
        self.0.add_key_with_lookups(
            builder,
            FixedSetupOpeningPointMessageV2 {
                proof_index: row[1].into(),
                point_index: row[2].into(),
                value: core::array::from_fn(|limb| row[4 + limb].into()),
            },
            row[3],
        );
        builder
            .when(AB::Expr::ONE - AB::Expr::from(row[0]))
            .assert_zero(row[3]);
    }
}

fn fixed_source_point_test_trace(
    demands: &[(u32, u32, u32)],
    opening_point: &[EF],
) -> RowMajorMatrix<F> {
    let width = 4 + D_EF;
    let height = demands.len().next_power_of_two().max(2);
    let mut values = F::zero_vec(height * width);
    for (row, &(proof_index, point_index, count)) in demands.iter().enumerate() {
        values[row * width] = F::ONE;
        values[row * width + 1] = F::from_u32(proof_index);
        values[row * width + 2] = F::from_u32(point_index);
        values[row * width + 3] = F::from_u32(count);
        values[row * width + 4..row * width + 4 + D_EF]
            .copy_from_slice(opening_point[point_index as usize].as_basis_coefficients_slice());
    }
    RowMajorMatrix::new(values, width)
}

#[test]
fn ordered_stacking_algebra_consumes_the_fixed_source_point_bus() {
    let fixture = fixture();
    let mut manager = BusIndexManager::new();
    let mut inventory = BusInventory::new(&mut manager);
    inventory.transcript_bus = TranscriptBus::new(AUTHORITY_TRANSCRIPT_BUS);
    let fixed_bus_idx = manager.new_bus_idx();
    let ordered_bus_idx = manager.new_bus_idx();
    let fixed_bus = FixedSetupOpeningPointBusV2::new(fixed_bus_idx);
    let ordered_bus = OrderedStackingSourcePointBus::new(ordered_bus_idx);

    let mut stacking = StackingModule::new(&fixture.vk, &mut manager, inventory);
    stacking
        .configure_ordered_reductions(
            fixture.profile.clone(),
            OrderedStackingAuthority {
                column_claims: ColumnClaimsBus::new(AUTHORITY_COLUMN_CLAIMS_BUS),
                source_opening_point: ordered_bus,
                transcript_proof_idx: AUTHORITY_TRANSCRIPT_PROOF_IDX,
            },
        )
        .unwrap();

    let adapter_profile = OrderedStackingSourcePointProfileV2::new(
        &fixture.profile,
        vec![0],
        vec![fixture.opening_point.len()],
    )
    .unwrap();
    let authority_demands = (0..fixture.opening_point.len())
        .map(|coordinate| (0, coordinate as u32, 7))
        .collect::<Vec<_>>();
    assert_eq!(
        adapter_profile
            .merged_fixed_setup_point_demands(&authority_demands)
            .unwrap(),
        (0..fixture.opening_point.len())
            .map(|coordinate| (0, coordinate as u32, 8))
            .collect::<Vec<_>>()
    );
    let demands = adapter_profile
        .merged_fixed_setup_point_demands(&[])
        .unwrap();
    assert_eq!(
        demands,
        (0..fixture.opening_point.len())
            .map(|coordinate| (0, coordinate as u32, 1))
            .collect::<Vec<_>>()
    );
    let adapter = OrderedStackingSourcePointAdapterAirV2 {
        profile: adapter_profile,
        fixed_source_bus: fixed_bus,
        ordered_stacking_bus: ordered_bus,
    };
    let adapter_ctx = adapter
        .generate_ctx::<SC>(&[OrderedStackingSourcePointRecordV2 {
            source_proof_index: 0,
            opening_point: &fixture.opening_point,
        }])
        .unwrap();
    let source_air = FixedSourcePointTestAir(fixed_bus);
    let source_trace = fixed_source_point_test_trace(&demands, &fixture.opening_point);

    let baseline_stacking = generate_ordered(
        &stacking,
        &fixture,
        &fixture.proof,
        &fixture.claims,
        &fixture.opening_point,
        &fixture.opening_point,
    )
    .unwrap();
    let mut airs = stacking.airs::<SC>();
    airs.push(Arc::new(adapter.clone()));
    airs.push(Arc::new(source_air));
    let mut baseline = baseline_stacking
        .into_iter()
        .map(|context| context.common_main)
        .collect::<Vec<_>>();
    baseline.push(adapter_ctx.common_main.clone());
    baseline.push(source_trace.clone());
    check_selected_bus_composition(&airs, &baseline, &[fixed_bus_idx, ordered_bus_idx]);

    // Change the host stacking point and its host-side expected copy together,
    // while retaining the genuine fixed source point in both adapter and
    // source traces. Local stacking constraints remain valid, but the isolated
    // point permutation must not balance.
    let mut swapped_host_point = fixture.opening_point.clone();
    swapped_host_point[0] += EF::ONE;
    let mutated_stacking = generate_ordered(
        &stacking,
        &fixture,
        &fixture.proof,
        &fixture.claims,
        &swapped_host_point,
        &swapped_host_point,
    )
    .unwrap();
    let mut malformed = mutated_stacking
        .into_iter()
        .map(|context| context.common_main)
        .collect::<Vec<_>>();
    malformed.push(adapter_ctx.common_main);
    malformed.push(source_trace);
    assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
        check_selected_bus_composition(&airs, &malformed, &[fixed_bus_idx, ordered_bus_idx]);
    }))
    .is_err());
}

#[test]
fn ordered_stacking_source_points_reject_transition_permutations() {
    let fixture = fixture();
    assert!(matches!(
        OrderedStackingSourcePointProfileV2::new(
            &fixture.profile,
            vec![1],
            vec![fixture.opening_point.len()],
        ),
        Err(
            super::OrderedStackingSourcePointErrorV2::NonCanonicalSourceProof {
                transition: 0,
                source_proof_index: 1,
            }
        )
    ));
}

#[test]
fn ordered_outputs_are_constrained_before_multi_statement_fanout() {
    let fixture = fixture();
    let mut manager = BusIndexManager::new();
    let mut inventory = BusInventory::new(&mut manager);
    inventory.transcript_bus = TranscriptBus::new(AUTHORITY_TRANSCRIPT_BUS);
    let authority_claims = ColumnClaimsBus::new(AUTHORITY_COLUMN_CLAIMS_BUS);
    let source_point_idx = manager.new_bus_idx();
    let source_opening_idx = manager.new_bus_idx();
    let source_opening_point_bus = OrderedStackingSourcePointBus::new(manager.new_bus_idx());
    let source_point_bus = WhirOpeningPointBus::new(source_point_idx);
    let source_opening_bus = OrderedStackingOpeningBus::new(source_opening_idx);
    let mut stacking = StackingModule::new(&fixture.vk, &mut manager, inventory);
    stacking
        .configure_ordered_reductions(
            fixture.profile.clone(),
            OrderedStackingAuthority {
                column_claims: authority_claims,
                source_opening_point: source_opening_point_bus,
                transcript_proof_idx: AUTHORITY_TRANSCRIPT_PROOF_IDX,
            },
        )
        .unwrap();
    stacking
        .configure_ordered_statement_outputs(OrderedStackingOutputBuses {
            reduced_point: source_point_bus,
            stacking_opening: source_opening_bus,
        })
        .unwrap();

    let stacking_contexts = generate_ordered(
        &stacking,
        &fixture,
        &fixture.proof,
        &fixture.claims,
        &fixture.opening_point,
        &fixture.opening_point,
    )
    .unwrap();
    let statement_shape = fixture.profile.statement_shape(0).unwrap();
    let whir_profile = MultiConstraintWhirProfile::new(
        1,
        statement_shape.point_dimension,
        statement_shape.commitment_widths,
    )
    .unwrap();
    let adapter_profile = OrderedStackingMultiConstraintProfile::one_whir_class(
        &fixture.profile,
        &whir_profile,
        fixture.vk.inner.params.num_whir_sumcheck_rounds(),
        0,
    )
    .unwrap();
    let statement_buses = MultiConstraintStatementBuses::new(
        manager.new_bus_idx(),
        manager.new_bus_idx(),
        manager.new_bus_idx(),
    );
    let adapter = OrderedStackingMultiConstraintAir {
        profile: adapter_profile,
        source_point_bus,
        source_opening_bus,
        statement_buses,
    };
    let stacking_point = fixture.preflight.stacking.sumcheck_rnd[0]
        .exp_powers_of_2()
        .take(fixture.vk.inner.params.l_skip)
        .chain(
            fixture
                .preflight
                .stacking
                .sumcheck_rnd
                .iter()
                .skip(1)
                .copied(),
        )
        .collect::<Vec<_>>();
    let adapter_ctx = adapter
        .generate_ctx::<SC>(&[OrderedStackingMultiConstraintStatement {
            reduced_cube_point: &stacking_point,
            stacking_openings: &fixture.proof.stacking_proof.stacking_openings,
        }])
        .unwrap();

    let mut airs = stacking.airs::<SC>();
    airs.push(Arc::new(adapter.clone()));
    let mut matrices = stacking_contexts
        .into_iter()
        .map(|context| context.common_main)
        .collect::<Vec<_>>();
    matrices.push(adapter_ctx.common_main);
    check_selected_bus_composition(&airs, &matrices, &[source_point_idx, source_opening_idx]);

    let adapter_idx = matrices.len() - 1;
    for cell in [0, D_EF * stacking_point.len()] {
        let mut malformed = matrices.clone();
        malformed[adapter_idx].values[cell] += F::ONE;
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
            check_selected_bus_composition(
                &airs,
                &malformed,
                &[source_point_idx, source_opening_idx],
            )
        }))
        .is_err());
    }
}

fn generate_ordered(
    module: &StackingModule,
    fixture: &OrderedFixture,
    proof: &Proof<SC>,
    claims: &[Vec<OrderedStackingClaim>],
    opening_point: &[EF],
    expected_opening_point: &[EF],
) -> Result<
    Vec<openvm_stark_backend::prover::AirProvingContext<CpuBackend<SC>>>,
    OrderedStackingTracegenError,
> {
    module.generate_ordered_reduction_ctxs::<SC>(
        &[OrderedStackingReduction {
            proof: &proof.stacking_proof,
            ordered_claims: claims,
            opening_point,
            preflight: OrderedStackingPreflight {
                claim_observation_tidx: fixture
                    .preflight
                    .batch_constraint
                    .tidx_before_column_openings,
                stacking: &fixture.preflight.stacking,
                expected_opening_point,
            },
        }],
        None,
    )
}

fn ordered_error(
    result: Result<
        Vec<openvm_stark_backend::prover::AirProvingContext<CpuBackend<SC>>>,
        OrderedStackingTracegenError,
    >,
) -> OrderedStackingTracegenError {
    match result {
        Err(error) => error,
        Ok(_) => panic!("malformed ordered stacking input was accepted"),
    }
}

fn sorted_canonical_rows(
    matrix: &openvm_stark_backend::p3_matrix::dense::RowMajorMatrix<F>,
) -> Vec<Vec<u32>> {
    let mut rows = (0..matrix.height())
        .map(|row| {
            matrix
                .row_slice(row)
                .expect("matrix row")
                .iter()
                .map(|value| value.as_canonical_u32())
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    rows.sort_unstable();
    rows
}

#[test]
fn ordered_reduction_matches_legacy_child_proof_trace_path() {
    let fixture = fixture();
    let (legacy, ordered) = modules(&fixture);
    let legacy_contexts =
        <StackingModule as TraceGenModule<GlobalCtxCpu, CpuBackend<SC>>>::generate_proving_ctxs(
            &legacy,
            &fixture.vk,
            core::slice::from_ref(&fixture.proof),
            core::slice::from_ref(&fixture.preflight),
            &(),
            None,
        )
        .expect("legacy stacking contexts");
    let ordered_contexts = generate_ordered(
        &ordered,
        &fixture,
        &fixture.proof,
        &fixture.claims,
        &fixture.opening_point,
        &fixture.opening_point,
    )
    .expect("ordered stacking contexts");
    assert_eq!(legacy_contexts.len(), 6);
    assert_eq!(ordered_contexts.len(), 6);
    for (air_idx, (legacy, ordered)) in legacy_contexts.iter().zip(&ordered_contexts).enumerate() {
        if air_idx == 5 {
            // EqBits rows after each root form an interaction-authenticated tree and are
            // intentionally unordered. The legacy HashMap has a randomized iteration seed;
            // ordered authority emits a deterministic key order. Compare the complete row
            // multiset, which is the protocol-equivalent differential invariant.
            assert_eq!(
                sorted_canonical_rows(&legacy.common_main),
                sorted_canonical_rows(&ordered.common_main),
                "ordered EqBits row multiset differs from child-proof trace"
            );
        } else {
            assert_eq!(
                legacy.common_main, ordered.common_main,
                "ordered trace differs from child-proof trace for stacking AIR {air_idx}"
            );
        }
        assert!(ordered.cached_mains.is_empty());
        assert!(ordered.public_values.is_empty());
    }
}

#[test]
fn ordered_reduction_rejects_swapped_claim_order_and_point() {
    let fixture = fixture();
    let (_, ordered) = modules(&fixture);

    let mut swapped = fixture.claims.clone();
    swapped[0].swap(0, 1);
    assert_eq!(
        ordered_error(generate_ordered(
            &ordered,
            &fixture,
            &fixture.proof,
            &swapped,
            &fixture.opening_point,
            &fixture.opening_point,
        )),
        OrderedStackingTracegenError::ClaimOrder(0)
    );

    let mut swapped_point = fixture.opening_point.clone();
    swapped_point[0] += EF::ONE;
    assert_eq!(
        ordered_error(generate_ordered(
            &ordered,
            &fixture,
            &fixture.proof,
            &fixture.claims,
            &swapped_point,
            &fixture.opening_point,
        )),
        OrderedStackingTracegenError::OpeningPoint(0)
    );
}

#[test]
fn ordered_reduction_rejects_malformed_round0_lengths() {
    let fixture = fixture();
    let (_, ordered) = modules(&fixture);
    let expected = fixture.proof.stacking_proof.univariate_round_coeffs.len();
    assert!(expected > 1);

    for malformed_len in [expected - 1, expected + 1, expected + 1_000_000] {
        let mut malformed = fixture.proof.clone();
        malformed
            .stacking_proof
            .univariate_round_coeffs
            .resize(malformed_len, EF::ZERO);
        assert_eq!(
            ordered_error(generate_ordered(
                &ordered,
                &fixture,
                &malformed,
                &fixture.claims,
                &fixture.opening_point,
                &fixture.opening_point,
            )),
            OrderedStackingTracegenError::ProofShape(0, "univariate round coefficient count")
        );
    }
}

#[test]
fn ordered_reductions_enforce_the_configured_inter_transition_prefix() {
    let fixture = fixture();
    let sorted = &fixture.preflight.proof_shape.sorted_trace_vdata;
    let layout = StackedLayout::new(
        fixture.vk.inner.params.l_skip,
        fixture.vk.inner.params.log_stacked_height(),
        sorted
            .iter()
            .map(|(air_idx, vdata)| {
                (
                    fixture.vk.inner.per_air[*air_idx].params.width.common_main,
                    vdata.log_height,
                )
            })
            .collect(),
    )
    .unwrap();
    let need_rot = sorted
        .iter()
        .map(|(air_idx, _)| fixture.vk.inner.per_air[*air_idx].params.need_rot)
        .collect::<Vec<_>>();
    let identities = fixture.claims[0]
        .iter()
        .map(|claim| claim.identity)
        .collect::<Vec<_>>();
    let first_claim_tidx = fixture
        .preflight
        .batch_constraint
        .tidx_before_column_openings;
    let profile = OrderedStackingProfile::new(
        fixture.vk.inner.params.l_skip,
        fixture.vk.inner.params.n_stack,
        fixture.vk.inner.params.w_stack,
        vec![vec![layout.clone()], vec![layout]],
        vec![vec![need_rot.clone()], vec![need_rot]],
        vec![vec![identities.clone()], vec![identities]],
    )
    .unwrap()
    .with_transcript_schedule(OrderedStackingTranscriptSchedule {
        first_claim_observation_tidx: first_claim_tidx,
        inter_reduction_prefix_len: 5,
    })
    .unwrap();

    let mut manager = BusIndexManager::new();
    let mut inventory = BusInventory::new(&mut manager);
    inventory.transcript_bus = TranscriptBus::new(AUTHORITY_TRANSCRIPT_BUS);
    let mut module = StackingModule::new(&fixture.vk, &mut manager, inventory);
    module
        .configure_ordered_reductions(
            profile,
            OrderedStackingAuthority {
                column_claims: ColumnClaimsBus::new(AUTHORITY_COLUMN_CLAIMS_BUS),
                source_opening_point: OrderedStackingSourcePointBus::new(
                    AUTHORITY_SOURCE_POINT_BUS,
                ),
                transcript_proof_idx: AUTHORITY_TRANSCRIPT_PROOF_IDX,
            },
        )
        .unwrap();

    let first_stacking = fixture.preflight.stacking.clone();
    let second_claim_tidx = first_stacking.post_tidx + 5;
    let second_offset = second_claim_tidx - first_claim_tidx;
    let mut second_stacking = first_stacking.clone();
    second_stacking.intermediate_tidx = second_stacking
        .intermediate_tidx
        .map(|tidx| tidx + second_offset);
    second_stacking.post_tidx += second_offset;
    let correct = [
        OrderedStackingReduction {
            proof: &fixture.proof.stacking_proof,
            ordered_claims: &fixture.claims,
            opening_point: &fixture.opening_point,
            preflight: OrderedStackingPreflight {
                claim_observation_tidx: first_claim_tidx,
                stacking: &first_stacking,
                expected_opening_point: &fixture.opening_point,
            },
        },
        OrderedStackingReduction {
            proof: &fixture.proof.stacking_proof,
            ordered_claims: &fixture.claims,
            opening_point: &fixture.opening_point,
            preflight: OrderedStackingPreflight {
                claim_observation_tidx: second_claim_tidx,
                stacking: &second_stacking,
                expected_opening_point: &fixture.opening_point,
            },
        },
    ];
    module
        .generate_ordered_reduction_ctxs::<SC>(&correct, None)
        .expect("configured five-field transition prefix");

    let legacy_second_claim_tidx = first_stacking.post_tidx;
    let legacy_offset = legacy_second_claim_tidx - first_claim_tidx;
    let mut legacy_second_stacking = first_stacking.clone();
    legacy_second_stacking.intermediate_tidx = legacy_second_stacking
        .intermediate_tidx
        .map(|tidx| tidx + legacy_offset);
    legacy_second_stacking.post_tidx += legacy_offset;
    let malformed = [
        correct[0],
        OrderedStackingReduction {
            proof: &fixture.proof.stacking_proof,
            ordered_claims: &fixture.claims,
            opening_point: &fixture.opening_point,
            preflight: OrderedStackingPreflight {
                claim_observation_tidx: legacy_second_claim_tidx,
                stacking: &legacy_second_stacking,
                expected_opening_point: &fixture.opening_point,
            },
        },
    ];
    assert_eq!(
        ordered_error(module.generate_ordered_reduction_ctxs::<SC>(&malformed, None)),
        OrderedStackingTracegenError::GlobalTranscriptOrder(1)
    );
}

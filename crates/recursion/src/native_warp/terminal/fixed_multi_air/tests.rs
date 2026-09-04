use core::borrow::{Borrow, BorrowMut};
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
};

use openvm_stark_backend::{
    air_builders::{
        debug::{check_constraints, DebugConstraintBuilder},
        symbolic::{get_symbolic_builder, SymbolicConstraintsDag, SymbolicRapBuilder},
        PartitionedAirBuilder,
    },
    keygen::types::{StarkVerifyingKey, StarkVerifyingParams, TraceWidth},
    native_warp::{
        prove_fixed_multi_air_terminal_constrained, verify_fixed_multi_air_terminal_constrained,
        DirectAirCodeClass, DirectAirMappedRotation, DirectAirPesatIndex, DirectAirPesatInstance,
        DirectAirPublicSchema, FixedMultiAirPesatIndex, FixedMultiAirPesatInstance,
        FixedMultiAirTerminalProof, NativeWarpChallenger,
    },
    transcript::{TranscriptHistory, TranscriptLog},
    warp_accum::{
        Accumulator, AccumulatorMessage, AccumulatorWitness, TerminalConstrainedLinearizerProof,
        TerminalConstrainedRsStatement, TerminalDescriptor, WarpLinearCode, WhirInitialRsLayout,
        WhirInitialRsWarpCode, WhirRsCodeProverData,
    },
    warp_pesat::{
        evaluate_mle, evaluate_packed_column_eq_mle, AccumulatorInstance, BundledPesat,
        PrismalinearMappedColumnBlock, PrismalinearMappedColumnRotation,
        PrismalinearMappedColumnTerm, PrismalinearMappedColumnWeight,
        TerminalStructuredLinearClaim, TerminalWeightSpec,
    },
    BaseAirWithPublicValues, PartitionedBaseAir, StarkProtocolConfig, SystemParams, WhirConfig,
    WhirProximityStrategy, WhirRoundConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, BabyBearPoseidon2Config as SC, EF, F,
};
use p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::*;
use crate::{
    bus::TranscriptBus,
    native_warp::terminal::{
        NativeTerminalWhirLinearizerWeightBus, NativeTerminalWhirPointBus,
        NativeTerminalWhirStatementBus,
    },
    system::{BusIndexManager, BusInventory},
};

fn ef(value: u64) -> EF {
    EF::from_u64(value)
}

fn ef4(seed: u64) -> EF {
    EF::from_basis_coefficients_slice(&[
        F::from_u64(seed),
        F::from_u64(seed + 1),
        F::from_u64(seed + 2),
        F::from_u64(seed + 3),
    ])
    .expect("four EF4 limbs")
}

fn check_partitioned<A>(air: &A, cached: &[&RowMajorMatrix<F>], common: &RowMajorMatrix<F>)
where
    A: for<'a> Air<DebugConstraintBuilder<'a, SC>>
        + BaseAir<F>
        + BaseAirWithPublicValues<F>
        + PartitionedBaseAir<F>,
{
    let mains = cached
        .iter()
        .map(|matrix| matrix.as_view())
        .chain(core::iter::once(common.as_view()))
        .collect::<Vec<_>>();
    check_constraints::<_, SC>(air, core::any::type_name::<A>(), &None, &mains, &[]);
}

fn claims() -> Vec<TerminalStructuredLinearClaim<EF>> {
    let mapped = TerminalWeightSpec::PrismalinearMappedColumns(PrismalinearMappedColumnWeight {
        log_message_len: 3,
        terms: vec![
            PrismalinearMappedColumnTerm {
                block: PrismalinearMappedColumnBlock {
                    start: 0,
                    log_height: 2,
                },
                l_skip: 0,
                barycentric_weights: vec![EF::ONE],
                folded_row_eq_point: vec![ef(5), ef(7)],
                rotation: PrismalinearMappedColumnRotation::Current,
                scale: ef(11),
            },
            PrismalinearMappedColumnTerm {
                block: PrismalinearMappedColumnBlock {
                    start: 4,
                    log_height: 2,
                },
                l_skip: 0,
                barycentric_weights: vec![EF::ONE],
                folded_row_eq_point: vec![ef(13), ef(17)],
                rotation: PrismalinearMappedColumnRotation::Next,
                scale: ef(19),
            },
        ],
    });
    vec![
        TerminalStructuredLinearClaim::new(mapped, ef(23)),
        TerminalStructuredLinearClaim::new(
            TerminalWeightSpec::Eq {
                point: vec![ef(29), ef(31), ef(37)],
            },
            ef(41),
        ),
    ]
}

fn components() -> Vec<FixedMultiAirLinearizerRawComponentPlan> {
    vec![
        FixedMultiAirLinearizerRawComponentPlan {
            ordinal: 0,
            claim: 0,
            term: 0,
            is_eq: false,
            is_zero: false,
            block_start: 0,
            log_height: 2,
            rotation: DirectAirMappedRotation::Current,
        },
        FixedMultiAirLinearizerRawComponentPlan {
            ordinal: 1,
            claim: 0,
            term: 1,
            is_eq: false,
            is_zero: false,
            block_start: 4,
            log_height: 2,
            rotation: DirectAirMappedRotation::Next,
        },
        FixedMultiAirLinearizerRawComponentPlan {
            ordinal: 2,
            claim: 1,
            term: 0,
            is_eq: true,
            is_zero: false,
            block_start: 0,
            log_height: 3,
            rotation: DirectAirMappedRotation::Current,
        },
    ]
}

struct LinearAir;
struct QuadraticAir;
struct ShiftedQuadraticAir;

macro_rules! impl_air_shape {
    ($air:ty) => {
        impl BaseAir<F> for $air {
            fn width(&self) -> usize {
                1
            }
        }

        impl BaseAirWithPublicValues<F> for $air {
            fn num_public_values(&self) -> usize {
                1
            }
        }

        impl PartitionedBaseAir<F> for $air {
            fn cached_main_widths(&self) -> Vec<usize> {
                Vec::new()
            }

            fn common_main_width(&self) -> usize {
                1
            }
        }
    };
}

impl_air_shape!(LinearAir);
impl_air_shape!(QuadraticAir);
impl_air_shape!(ShiftedQuadraticAir);

impl Air<SymbolicRapBuilder<F>> for LinearAir {
    fn eval(&self, builder: &mut SymbolicRapBuilder<F>) {
        let main = builder.common_main().clone();
        let local = main.row_slice(0).expect("linear local")[0];
        let next = main.row_slice(1).expect("linear next")[0];
        let initial = builder.public_values()[0];
        builder.assert_zero(builder.is_first_row() * (local - initial));
        builder.assert_zero(builder.is_transition() * (next - local - F::ONE));
    }
}

impl Air<SymbolicRapBuilder<F>> for QuadraticAir {
    fn eval(&self, builder: &mut SymbolicRapBuilder<F>) {
        let main = builder.common_main().clone();
        let local = main.row_slice(0).expect("quadratic local")[0];
        let square = builder.public_values()[0];
        builder.assert_zero(local * local - square);
    }
}

impl Air<SymbolicRapBuilder<F>> for ShiftedQuadraticAir {
    fn eval(&self, builder: &mut SymbolicRapBuilder<F>) {
        let main = builder.common_main().clone();
        let local = main.row_slice(0).expect("shifted quadratic local")[0];
        let square = builder.public_values()[0];
        // Same shape and degree, different setup-fixed symbolic DAG.
        builder.assert_zero(local * local + local - square);
    }
}

fn verifying_key<A>(air: &A) -> StarkVerifyingKey<F, [F; 8]>
where
    A: Air<SymbolicRapBuilder<F>> + BaseAirWithPublicValues<F>,
{
    let width = TraceWidth {
        preprocessed: None,
        cached_mains: Vec::new(),
        common_main: 1,
    };
    let symbolic = get_symbolic_builder(air, &width).constraints();
    let degree = symbolic.max_constraint_degree();
    StarkVerifyingKey {
        preprocessed_data: None,
        params: StarkVerifyingParams {
            width,
            num_public_values: 1,
            need_rot: true,
        },
        symbolic_constraints: Arc::new(SymbolicConstraintsDag::from(symbolic)),
        max_constraint_degree: degree as u8,
        is_required: true,
        unused_variables: Vec::new(),
    }
}

type TestHasher = <SC as StarkProtocolConfig>::Hasher;
type TestCode = WhirInitialRsWarpCode<TestHasher>;
type TestAccumulator = Accumulator<EF, [F; 8], WhirRsCodeProverData<EF, [F; 8]>>;
type TestProof = TerminalConstrainedLinearizerProof<FixedMultiAirTerminalProof<EF>>;

fn direct_relation<A>(
    hasher: &TestHasher,
    air: &A,
    air_id: usize,
    log_height: usize,
    schema: u32,
) -> DirectAirPesatIndex<F, [F; 8]>
where
    A: Air<SymbolicRapBuilder<F>> + BaseAirWithPublicValues<F>,
{
    DirectAirPesatIndex::from_verifying_key(
        hasher,
        [F::from_u32(0xa11c); 8],
        air_id,
        log_height,
        &verifying_key(air),
        None,
        DirectAirPublicSchema {
            public_values_len: 1,
            boundary_values_len: 0,
            schema_digest: [F::from_u32(schema); 8],
        },
        DirectAirCodeClass {
            log_message_len: log_height as u8,
            log_blowup: 1,
            log_codeword_len: log_height as u8 + 1,
            initial_folding_factor: 1,
            rows_per_query: 1,
        },
    )
    .expect("direct relation fixture")
}

fn fixed_relation(hasher: &TestHasher, shifted: bool) -> FixedMultiAirPesatIndex<F, [F; 8]> {
    fixed_relation_with_code_class(
        hasher,
        shifted,
        DirectAirCodeClass {
            log_message_len: 3,
            log_blowup: 1,
            log_codeword_len: 4,
            initial_folding_factor: 1,
            rows_per_query: 1,
        },
    )
}

fn fixed_relation_with_code_class(
    hasher: &TestHasher,
    shifted: bool,
    code_class: DirectAirCodeClass,
) -> FixedMultiAirPesatIndex<F, [F; 8]> {
    let quadratic = if shifted {
        direct_relation(hasher, &ShiftedQuadraticAir, 7, 2, 0x7007)
    } else {
        direct_relation(hasher, &QuadraticAir, 7, 2, 0x7007)
    };
    FixedMultiAirPesatIndex::from_direct_air_regions(
        hasher,
        [F::from_u32(0xa11c); 8],
        vec![quadratic, direct_relation(hasher, &LinearAir, 3, 1, 0x3003)],
        code_class,
    )
    .expect("fixed multi-AIR fixture")
}

fn coefficient_two_coset_profile_fixture() -> (
    Arc<FixedMultiAirPesatIndex<F, [F; 8]>>,
    TerminalDescriptor<[F; 8]>,
    FixedMultiAirTerminalCircuitProfile,
) {
    let config = SC::default_from_params(SystemParams::new_for_testing(8));
    // A fixed relation's padded message dimension is canonical, so use one
    // genuine 2^8 trace region instead of artificially inflating a tiny
    // fixture to the production code class.
    let relation = Arc::new(
        FixedMultiAirPesatIndex::from_direct_air_regions(
            config.hasher(),
            [F::from_u32(0xa11c); 8],
            vec![direct_relation(
                config.hasher(),
                &QuadraticAir,
                7,
                8,
                0x7007,
            )],
            DirectAirCodeClass {
                log_message_len: 8,
                log_blowup: 1,
                log_codeword_len: 9,
                initial_folding_factor: 0,
                rows_per_query: 16,
            },
        )
        .expect("coefficient fixed multi-AIR fixture"),
    );
    let code =
        WhirInitialRsWarpCode::new_coefficient_two_coset(config.hasher().clone(), 8, 1, 0, 16);
    let whir = WhirConfig {
        // WHIR names the logarithm `k`: k=4 is a 16-way fold.
        k: 4,
        rounds: vec![WhirRoundConfig { num_queries: 1 }],
        mu_pow_bits: 0,
        query_phase_pow_bits: 0,
        folding_pow_bits: 0,
        proximity: WhirProximityStrategy::UniqueDecoding,
    };
    let descriptor = TerminalDescriptor::from_whir_initial_rs([F::ZERO; 8], &code, &whir, 4);
    let profile = FixedMultiAirTerminalCircuitProfile::new(relation.clone(), &descriptor)
        .expect("production coefficient two-coset profile");
    (relation, descriptor, profile)
}

#[test]
fn padding_weight_air_stays_within_final_degree_profile() {
    let air = FixedMultiAirPaddingWeightAir {
        beta_bus: FixedMultiAirBetaCoordinateBus::new(20),
        point_bus: FixedMultiAirPaddingPointBus::new(21),
        opening_bus: FixedMultiAirPaddingOpeningBus::new(22),
        header_bus: FixedMultiAirStructuredClaimHeaderBus::new(23),
        structured_point_bus: FixedMultiAirStructuredPointBus::new(24),
        claim_index: 2,
        log_message_len: 3,
        one_coordinate: 4,
        scale_exponent: 2,
    };
    let symbolic = get_symbolic_builder(
        &air,
        &TraceWidth {
            preprocessed: None,
            cached_mains: air.cached_main_widths(),
            common_main: air.common_main_width(),
        },
    )
    .constraints();
    assert_eq!(symbolic.max_constraint_degree(), 8);
}

#[test]
fn packed_two_carry_raw_air_stays_degree_four() {
    let air = FixedMultiAirLinearizerRawAir {
        batched_claim_bus: FixedMultiAirBatchedClaimBus::new(30),
        mapped_term_bus: FixedMultiAirMappedTermBus::new(31),
        structured_point_bus: FixedMultiAirStructuredPointBus::new(32),
        aux_point_bus: FixedMultiAirLinearizerAuxPointBus::new(33),
        raw_term_bus: FixedMultiAirLinearizerRawTermBus::new(34),
        log_message_len: 4,
        component_count: 1,
    };
    let symbolic = get_symbolic_builder(
        &air,
        &TraceWidth {
            preprocessed: None,
            cached_mains: air.cached_main_widths(),
            common_main: air.common_main_width(),
        },
    )
    .constraints();
    assert_eq!(symbolic.max_constraint_degree(), 4);
}

fn backend_fixture() -> (
    FixedMultiAirPesatIndex<F, [F; 8]>,
    TestCode,
    TerminalDescriptor<[F; 8]>,
    TestAccumulator,
    TestProof,
    TerminalConstrainedRsStatement<EF>,
    TranscriptLog<F, [F; 16]>,
) {
    let config = SC::default_from_params(SystemParams::new_for_testing(6));
    let hasher = config.hasher().clone();
    let relation = fixed_relation(&hasher, false);
    let quadratic = relation.region_relation(0).expect("quadratic region");
    let linear = relation.region_relation(1).expect("linear region");
    let quadratic_witness = quadratic
        .witness_from_row_major_parts(
            &[] as &[RowMajorMatrix<F>],
            Some(&RowMajorMatrix::new(vec![F::from_u32(3); 4], 1)),
        )
        .expect("quadratic witness");
    let linear_witness = linear
        .witness_from_row_major_parts(
            &[] as &[RowMajorMatrix<F>],
            Some(&RowMajorMatrix::new(
                [5, 6].into_iter().map(F::from_u32).collect(),
                1,
            )),
        )
        .expect("linear witness");
    let raw = relation
        .stack_witnesses(&[quadratic_witness, linear_witness])
        .expect("stacked witness");
    let public = FixedMultiAirPesatInstance {
        regions: vec![
            DirectAirPesatInstance {
                public_values: vec![F::from_u32(9)],
                boundary_values: Vec::new(),
            },
            DirectAirPesatInstance {
                public_values: vec![F::from_u32(5)],
                boundary_values: Vec::new(),
            },
        ],
    };
    let mut message = relation.padded_witness::<EF>(&raw).expect("padded witness");
    message[0] += ef(2);
    message[relation.raw_witness_len()] = ef(3);
    let tau = [2, 5, 9, 17]
        .into_iter()
        .map(EF::from_u32)
        .collect::<Vec<_>>();
    let mut explicit = relation
        .explicit_assignment(&public)
        .expect("explicit assignment")
        .into_iter()
        .map(EF::from)
        .collect::<Vec<_>>();
    explicit[0] = ef(2);
    let eta = relation.evaluate_bundled_at_point_split(&tau, &explicit, &message);
    assert_ne!(eta, EF::ZERO);

    let code = WhirInitialRsWarpCode::new(hasher, 3, 1, 1, 1);
    let (root, prover_data, codeword) =
        <TestCode as WarpLinearCode<F, EF>>::commit_message(&code, &message)
            .expect("terminal message commitment");
    let alpha = [3, 7, 11, 19]
        .into_iter()
        .map(EF::from_u32)
        .collect::<Vec<_>>();
    let mu = openvm_stark_backend::warp_pesat::evaluate_mle(&codeword, &alpha);
    let mut beta = tau;
    beta.extend_from_slice(&explicit);
    let accumulator = Accumulator {
        instance: AccumulatorInstance {
            rt: root,
            alpha,
            mu,
            beta,
            eta,
        },
        witness: AccumulatorWitness {
            prover_data,
            codeword,
            prepared_codeword: None,
            prepared_message: None,
            message: AccumulatorMessage::dense(message),
        },
    };
    let whir = WhirConfig {
        k: 1,
        rounds: vec![WhirRoundConfig { num_queries: 1 }],
        mu_pow_bits: 0,
        query_phase_pow_bits: 0,
        folding_pow_bits: 0,
        proximity: WhirProximityStrategy::UniqueDecoding,
    };
    let descriptor = TerminalDescriptor::from_whir_initial_rs(root, &code, &whir, 4);
    let mut prover = NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
    let (proof, statement) = prove_fixed_multi_air_terminal_constrained::<F, EF, _, _, _>(
        &code,
        &descriptor,
        &relation,
        &accumulator,
        &mut prover,
    )
    .expect("backend fixed multi-AIR terminal proof");
    let transcript = TranscriptHistory::into_log(prover.into_inner());
    let mut verifier = NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
    let verified = verify_fixed_multi_air_terminal_constrained::<F, EF, _, _, _>(
        &code,
        &descriptor,
        &relation,
        &accumulator.instance,
        &proof,
        &mut verifier,
    )
    .expect("backend fixed multi-AIR terminal verification");
    assert_eq!(verified, statement);
    let verifier_transcript = TranscriptHistory::into_log(verifier.into_inner());
    assert_eq!(transcript.values(), verifier_transcript.values());
    assert_eq!(transcript.samples(), verifier_transcript.samples());
    (
        relation,
        code,
        descriptor,
        accumulator,
        proof,
        statement,
        transcript,
    )
}

#[test]
fn terminal_circuit_digest_is_independent_of_dynamic_accumulator_root() {
    let (relation, _code, descriptor, _accumulator, _proof, _statement, _transcript) =
        backend_fixture();
    let relation = Arc::new(relation);
    let profile = FixedMultiAirTerminalCircuitProfile::new(relation.clone(), &descriptor).unwrap();
    assert_eq!(
        profile.initial_rs_layout,
        WhirInitialRsLayout::OrdinarySubgroup
    );
    assert_eq!(profile.initial_folding_factor, 1);
    assert_eq!(profile.selector_folding_factor, 1);
    assert_eq!(profile.whir_fold_arity(), 2);
    assert_eq!(profile.decomposition.len(), relation.region_count() + 1);
    assert!(profile
        .decomposition
        .iter()
        .all(|plan| plan.prefix_len == 0 && plan.prefix_tag == 0 && plan.degree_delta == 0));
    let mut changed_descriptor = descriptor;
    changed_descriptor.root[0] += F::ONE;
    let changed_profile =
        FixedMultiAirTerminalCircuitProfile::new(relation, &changed_descriptor).unwrap();

    assert_eq!(profile.metadata_words, changed_profile.metadata_words);
    assert_eq!(
        profile.digest_words(17, 93),
        changed_profile.digest_words(17, 93)
    );
}

#[test]
fn production_coefficient_two_coset_profile_is_admitted_and_mutations_reject() {
    let (relation, descriptor, profile) = coefficient_two_coset_profile_fixture();
    assert_eq!(
        profile.initial_rs_layout,
        WhirInitialRsLayout::CoefficientTwoCosetGrs
    );
    assert_eq!(profile.initial_folding_factor, 0);
    assert_eq!(profile.selector_folding_factor, profile.log_message_len);
    assert_eq!(profile.k, 4);
    assert_eq!(profile.whir_fold_arity(), 16);
    assert_eq!(profile.rs_adjoint_round_count, profile.alpha_len);
    assert_eq!(profile.rs_adjoint_degree, profile.log_message_len + 1);

    let mut mutations = Vec::new();
    let mut changed = descriptor.clone();
    changed.initial_folding_factor = 1;
    mutations.push(changed);
    let mut changed = descriptor.clone();
    changed.initial_oracle_width = 16;
    mutations.push(changed);
    let mut changed = descriptor.clone();
    changed.rows_per_query = 8;
    mutations.push(changed);
    let mut changed = descriptor.clone();
    changed.whir_k = 3;
    mutations.push(changed);
    let mut changed = descriptor.clone();
    changed.domain_tag = 1;
    mutations.push(changed);
    let mut changed = descriptor.clone();
    changed.subgroup_generator = changed.subgroup_generator.wrapping_add(1);
    mutations.push(changed);
    let mut changed = descriptor;
    changed.whir_k = u64::MAX;
    mutations.push(changed);

    for changed in mutations {
        let result = catch_unwind(AssertUnwindSafe(|| {
            FixedMultiAirTerminalCircuitProfile::new(relation.clone(), &changed)
        }));
        assert!(result.is_ok(), "malformed descriptor must not panic");
        assert!(
            result.expect("panic-free descriptor rejection").is_err(),
            "layout mutation must reject"
        );
    }

    let (_, _, ordinary_descriptor, _, _, _, _) = backend_fixture();
    for changed in [
        {
            let mut changed = ordinary_descriptor.clone();
            changed.initial_folding_factor = 0;
            changed
        },
        {
            let mut changed = ordinary_descriptor.clone();
            changed.rows_per_query = 2;
            changed
        },
        {
            let mut changed = ordinary_descriptor.clone();
            changed.code_layout_version = 1;
            changed
        },
    ] {
        assert!(FixedMultiAirTerminalCircuitProfile::new(
            Arc::new(fixed_relation(
                SC::default_from_params(SystemParams::new_for_testing(6)).hasher(),
                false,
            )),
            &changed,
        )
        .is_err());
    }
}

#[test]
fn production_profile_drives_identical_selector_air_and_trace_dimension() {
    let (_relation, _descriptor, profile) = coefficient_two_coset_profile_fixture();
    let mut manager = BusIndexManager::new();
    let shared = BusInventory::new(&mut manager);
    let config = SC::default_from_params(SystemParams::new_for_testing(8));
    let circuit = FixedMultiAirTerminalCircuit::new(
        profile.clone(),
        &shared,
        manager.next_bus_idx(),
        config.params().clone(),
        config.hasher(),
    )
    .expect("coefficient profile circuit adapter");
    let y_air = circuit.linearizer_y_air();
    assert_eq!(
        y_air.initial_folding_factor,
        profile.selector_folding_factor
    );
    assert_ne!(y_air.initial_folding_factor, profile.k);

    let point = (0..profile.log_message_len)
        .map(|coordinate| ef4(10_003 + coordinate as u64 * 31))
        .collect::<Vec<_>>();
    let whir_point = (0..profile.log_message_len)
        .map(|coordinate| ef4(20_003 + coordinate as u64 * 37))
        .collect::<Vec<_>>();
    let (cached, common, factors) = generate_fixed_multi_air_linearizer_y_traces_for_layout(
        &point,
        &whir_point,
        profile.initial_folding_factor,
        profile.initial_rs_layout,
        None,
    )
    .expect("production layout selector trace");
    let (direct_cached, direct_common, direct_factors) =
        generate_fixed_multi_air_linearizer_y_traces(
            &point,
            &whir_point,
            profile.selector_folding_factor,
            None,
        )
        .expect("explicit production selector trace");
    assert_eq!(cached.values, direct_cached.values);
    assert_eq!(common.values, direct_common.values);
    assert_eq!(factors, direct_factors);
    check_partitioned(&y_air, &[&cached], &common);

    let (_, _, wrong_factors) =
        generate_fixed_multi_air_linearizer_y_traces(&point, &whir_point, profile.k, None)
            .expect("deliberately wrong WHIR-k selector trace");
    assert_ne!(wrong_factors, factors);

    let mut mutated = cached.clone();
    let width = mutated.width();
    let first: &mut FixedMultiAirLinearizerYScheduleCols<F> = mutated.values[..width].borrow_mut();
    first.use_identity = F::ONE;
    assert!(catch_unwind(AssertUnwindSafe(|| {
        check_partitioned(&y_air, &[&mutated], &common);
    }))
    .is_err());
}

#[test]
fn complete_terminal_circuit_stays_within_final_degree_profile() {
    let (relation, _code, descriptor, _accumulator, _proof, _statement, _transcript) =
        backend_fixture();
    let profile =
        FixedMultiAirTerminalCircuitProfile::new(Arc::new(relation), &descriptor).unwrap();
    let mut manager = BusIndexManager::new();
    let shared = BusInventory::new(&mut manager);
    let config = SC::default_from_params(SystemParams::new_for_testing(8));
    let circuit = FixedMultiAirTerminalCircuit::new(
        profile,
        &shared,
        manager.next_bus_idx(),
        config.params().clone(),
        config.hasher(),
    )
    .unwrap();
    for air in circuit.airs::<SC>() {
        let symbolic = get_symbolic_builder(
            air.as_ref(),
            &TraceWidth {
                preprocessed: BaseAir::<F>::preprocessed_trace(air.as_ref())
                    .map(|trace| trace.width()),
                cached_mains: air.cached_main_widths(),
                common_main: air.common_main_width(),
            },
        )
        .constraints();
        assert!(
            symbolic.max_constraint_degree() <= 8,
            "{} has degree {}",
            air.name(),
            symbolic.max_constraint_degree()
        );
    }
}

#[test]
fn final_wrapper_profile_adds_exactly_one_instance_lookup_per_coordinate() {
    let (relation, _code, descriptor, accumulator, proof, _statement, transcript) =
        backend_fixture();
    let counts = FixedMultiAirTerminalPrefixLookupCounts::semantic_minimum(
        accumulator.instance.alpha.len(),
        accumulator.instance.beta.len(),
        relation.region_count(),
    );
    let standalone = generate_fixed_multi_air_terminal_prefix_traces(
        &descriptor,
        &relation,
        &accumulator.instance,
        &proof.linearizer,
        &transcript,
        0,
        &counts,
        None,
    )
    .unwrap();
    let final_wrapper = generate_fixed_multi_air_terminal_prefix_traces_for_final_wrapper(
        &descriptor,
        &relation,
        &accumulator.instance,
        &proof.linearizer,
        &transcript,
        0,
        &counts,
        None,
    )
    .unwrap();

    assert_eq!(final_wrapper.end_tidx, standalone.end_tidx);
    assert_eq!(final_wrapper.common.values, standalone.common.values);
    assert_eq!(final_wrapper.cached.width(), standalone.cached.width());
    assert_eq!(final_wrapper.cached.height(), standalone.cached.height());

    // Normalize the sole intended delta, then require byte-for-byte equality.
    // In particular, the beta-coordinate bus count must not inherit the outer
    // terminal-instance lookup.
    let width = standalone.cached.width();
    let mut normalized_final = final_wrapper.cached.values.clone();
    let mut beta_rows = 0usize;
    for (standalone_row, final_row) in standalone
        .cached
        .values
        .chunks_exact(width)
        .zip(normalized_final.chunks_exact_mut(width))
    {
        let standalone_cols: &FixedMultiAirPrefixScheduleCols<F> = standalone_row.borrow();
        let final_cols: &mut FixedMultiAirPrefixScheduleCols<F> = final_row.borrow_mut();
        if standalone_cols.source_flags[4] == F::ONE
            && standalone_cols.instance_lookup_count != F::ZERO
        {
            beta_rows += 1;
            // The instance catalog has one fixed/native bridge consumer. Its
            // fanout is independent of the typed beta-coordinate bus.
            assert_eq!(standalone_cols.instance_lookup_count, F::ONE);
            assert_eq!(standalone_cols.beta_lookup_count, F::ZERO);
        }
        if standalone_cols.instance_lookup_count != F::ZERO {
            assert_eq!(
                final_cols.instance_lookup_count,
                standalone_cols.instance_lookup_count + F::ONE
            );
            final_cols.instance_lookup_count -= F::ONE;
        }
    }
    assert_eq!(beta_rows, accumulator.instance.beta.len());
    assert_eq!(normalized_final, standalone.cached.values);
}

#[test]
fn canonical_row_selectors_use_logarithmic_terminal_traces() {
    let config = SC::default_from_params(SystemParams::new_for_testing(6));
    let relation = fixed_relation(config.hasher(), false);
    for region in 0..relation.region_count() {
        let local = relation.region_relation(region).expect("regional relation");
        let plan = FixedMultiAirEndpointPlan::from_relation(&relation, region)
            .expect("regional endpoint plan");
        assert!(plan.fixed.is_empty(), "fixture has no preprocessed columns");
        assert_eq!(plan.dense_fixed_source_count(), 0);
        assert_eq!(plan.analytic_fixed_source_count(), 3);
        let point = (0..plan.log_height)
            .map(|coordinate| ef(3 + coordinate as u64))
            .collect::<Vec<_>>();
        let generated = generate_fixed_multi_air_endpoint_fixed_traces(local, &plan, &point, None)
            .expect("analytic row-selector traces");
        let first = point
            .iter()
            .fold(EF::ONE, |value, &x| value * (EF::ONE - x));
        let last = point.iter().copied().product::<EF>();
        assert_eq!(generated.values[plan.first_source], first);
        assert_eq!(generated.values[plan.last_source], last);
        assert_eq!(generated.values[plan.transition_source], EF::ONE - last);
        assert_eq!(
            generated.common.height(),
            (3 * (plan.log_height + 2)).next_power_of_two(),
            "selector proof size depends on log height, not trace height"
        );
        let air = FixedMultiAirEndpointFixedAir {
            point_bus: FixedMultiAirRegionPointBus::new(1),
            state_bus: FixedMultiAirEndpointFixedFoldStateBus::new(2),
            fixed_value_bus: FixedMultiAirEndpointFixedValueBus::new(3),
            region,
        };
        check_partitioned(&air, &[&generated.cached], &generated.common);
    }
}

#[test]
fn constraint_weight_dp_generated_trace_satisfies_plain_constraints_at_large_shape() {
    let tau = (0..28)
        .map(|coordinate| ef(11 + coordinate as u64))
        .collect::<Vec<_>>();
    let point = (0..13)
        .map(|coordinate| ef(101 + coordinate as u64))
        .collect::<Vec<_>>();
    let generated = generate_fixed_multi_air_constraint_weight_traces(
        &tau,
        &point,
        257,
        3 << point.len(),
        None,
        Some(1 << 13),
    )
    .expect("large constraint-weight DP traces");
    let air = FixedMultiAirConstraintWeightDpAir {
        beta_bus: FixedMultiAirBetaCoordinateBus::new(1),
        point_bus: FixedMultiAirRegionPointBus::new(2),
        state_bus: FixedMultiAirConstraintWeightStateBus::new(3),
        weight_bus: FixedMultiAirConstraintWeightBus::new(4),
        region: 0,
    };
    check_partitioned(&air, &[&generated.dp_cached], &generated.dp_common);
}

#[test]
fn fixed_multi_air_backend_transcript_prefix_and_proof_mutations_reject() {
    let (relation, code, descriptor, accumulator, proof, statement, transcript) = backend_fixture();
    assert_eq!(
        statement.linearizer_claims.len(),
        relation.region_count() + 1,
        "two regional mapped claims plus the global padding claim"
    );
    assert!(statement.coordinate_claims_hold(
        accumulator
            .witness
            .message
            .as_dense()
            .expect("dense fixture message")
    ));

    let counts = FixedMultiAirTerminalPrefixLookupCounts::semantic_minimum(
        accumulator.instance.alpha.len(),
        accumulator.instance.beta.len(),
        relation.region_count(),
    );
    let prefix = generate_fixed_multi_air_terminal_prefix_traces(
        &descriptor,
        &relation,
        &accumulator.instance,
        &proof.linearizer,
        &transcript,
        0,
        &counts,
        None,
    )
    .expect("exact backend terminal prefix");
    let prefix_air = FixedMultiAirTerminalPrefixAir {
        transcript_bus: TranscriptBus::new(20),
        binding_bus: FixedMultiAirTerminalBindingBus::new(21),
        instance_bus: FixedMultiAirTerminalInstanceValueBus::new(22),
        beta_bus: FixedMultiAirBetaCoordinateBus::new(23),
        global_bus: FixedMultiAirGlobalClaimBus::new(24),
        region_claim_bus: FixedMultiAirRegionClaimBus::new(25),
        padding_claim_bus: FixedMultiAirPaddingClaimBus::new(26),
        relation_digest: relation.description().relation_digest,
        alpha_len: accumulator.instance.alpha.len(),
        beta_len: accumulator.instance.beta.len(),
        binding_lookup_count: 1,
    };
    check_partitioned(&prefix_air, &[&prefix.cached], &prefix.common);

    // Complete instance/root binding happens before the first challenge.
    let mut forged_root = accumulator.instance.clone();
    forged_root.rt[0] += F::ONE;
    assert!(generate_fixed_multi_air_terminal_prefix_traces(
        &descriptor,
        &relation,
        &forged_root,
        &proof.linearizer,
        &transcript,
        0,
        &counts,
        None,
    )
    .is_err());

    let mut forged_eta = accumulator.instance.clone();
    forged_eta.eta += EF::ONE;
    assert_eq!(
        generate_fixed_multi_air_terminal_prefix_traces(
            &descriptor,
            &relation,
            &forged_eta,
            &proof.linearizer,
            &transcript,
            0,
            &counts,
            None,
        )
        .expect_err("eta mutation must differ from backend transcript"),
        FixedMultiAirTerminalPrefixTraceError::Transcript,
    );

    let mut forged_padding_claim = proof.linearizer.clone();
    *forged_padding_claim
        .padding_claim
        .as_mut()
        .expect("padding claim") += EF::ONE;
    assert_eq!(
        generate_fixed_multi_air_terminal_prefix_traces(
            &descriptor,
            &relation,
            &accumulator.instance,
            &forged_padding_claim,
            &transcript,
            0,
            &counts,
            None,
        )
        .expect_err("padding claim mutation must reject"),
        FixedMultiAirTerminalPrefixTraceError::Transcript,
    );

    // A same-shape relation with one changed symbolic DAG instruction has a
    // different canonical relation description and cannot replay the proof.
    let shifted_relation = fixed_relation(code.hasher(), true);
    assert_ne!(
        shifted_relation.description().relation_digest,
        relation.description().relation_digest
    );
    assert_eq!(
        generate_fixed_multi_air_terminal_prefix_traces(
            &descriptor,
            &shifted_relation,
            &accumulator.instance,
            &proof.linearizer,
            &transcript,
            0,
            &counts,
            None,
        )
        .expect_err("symbolic DAG substitution must reject"),
        FixedMultiAirTerminalPrefixTraceError::Transcript,
    );

    // Walk the exact regional transcript to the global padding proof. This
    // also gives a producer-level mutation test for every mapped opening.
    let mut tidx = prefix.end_tidx;
    for region in 0..relation.region_count() {
        let local = relation.region_relation(region).expect("regional relation");
        let regional_proof = &proof.linearizer.region_proofs[region];
        let region_prefix = generate_fixed_multi_air_region_prefix_traces(
            region,
            &relation,
            &accumulator.instance.beta,
            proof.linearizer.region_claims[region],
            &transcript,
            tidx,
            None,
        )
        .expect("regional transcript prefix");
        let sumcheck_air = FixedMultiAirRegionSumcheckAir::new(
            TranscriptBus::new(20),
            FixedMultiAirRegionStartBus::new(27),
            FixedMultiAirRegionPointBus::new(28),
            FixedMultiAirRegionSumcheckFinalBus::new(29),
            region,
            regional_proof.round_evaluations.len(),
            0,
            1,
        );
        let sumcheck = generate_fixed_multi_air_region_sumcheck_trace(
            &sumcheck_air,
            proof.linearizer.region_claims[region],
            regional_proof,
            &transcript,
            region_prefix.end_tidx,
            None,
        )
        .expect("regional sumcheck transcript");
        check_partitioned(&sumcheck_air, &[], &sumcheck);
        let tail_start =
            region_prefix.end_tidx + regional_proof.round_evaluations.len() * (3 + 7 + 1) * 4;
        let tail = generate_fixed_multi_air_region_tail_traces(
            local,
            EF::ZERO,
            regional_proof,
            &transcript,
            tail_start,
            &vec![0; regional_proof.opened_columns.len()],
            0,
            0,
            None,
        )
        .expect("regional opening tail");
        if region == 0 {
            let mut forged = regional_proof.clone();
            forged.opened_columns[0] += EF::ONE;
            assert_eq!(
                generate_fixed_multi_air_region_tail_traces(
                    local,
                    EF::ZERO,
                    &forged,
                    &transcript,
                    tail_start,
                    &vec![0; forged.opened_columns.len()],
                    0,
                    0,
                    None,
                )
                .expect_err("mapped opening mutation must reject"),
                FixedMultiAirRegionTailTraceError::Transcript,
            );
        }
        tidx = tail.end_tidx;
    }

    let padding_proof = proof
        .linearizer
        .padding_proof
        .as_ref()
        .expect("global padding proof");
    let padding_claim = proof.linearizer.padding_claim.expect("padding claim");
    let padding = generate_fixed_multi_air_padding_sumcheck_traces(
        padding_claim,
        padding_proof,
        &transcript,
        tidx,
        None,
    )
    .expect("global padding sumcheck");
    let _opening = generate_fixed_multi_air_padding_opening_trace(
        padding_proof,
        &transcript,
        padding.final_claim,
        padding.end_tidx,
    )
    .expect("global padding opening");
    let mut forged_padding = padding_proof.clone();
    forged_padding.message_opening += EF::ONE;
    assert_eq!(
        generate_fixed_multi_air_padding_opening_trace(
            &forged_padding,
            &transcript,
            padding.final_claim,
            padding.end_tidx,
        )
        .expect_err("padding opening mutation must reject"),
        FixedMultiAirPaddingTraceError::Transcript,
    );

    // Backend verification is retained as the differential oracle, including
    // its no-panic rejection behavior.
    let mut forged_backend = proof.clone();
    forged_backend.linearizer.region_proofs[0].opened_columns[0] += EF::ONE;
    let result = catch_unwind(AssertUnwindSafe(|| {
        verify_fixed_multi_air_terminal_constrained::<F, EF, _, _, _>(
            &code,
            &descriptor,
            &relation,
            &accumulator.instance,
            &forged_backend,
            &mut NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder()),
        )
    }));
    assert!(result.is_ok());
    assert!(result.expect("mutation verifier must not panic").is_err());
}

#[test]
fn fixed_multi_air_exact_linearizer_adjoint_matches_backend_and_rejects_mutations() {
    let claims = claims();
    let xi = ef(43);
    let batching_scales = vec![xi, xi.square()];
    let whir_point = vec![ef(47), ef(53), ef(59)];
    let config = SC::default_from_params(SystemParams::new_for_testing(6));
    let code = WhirInitialRsWarpCode::new(config.hasher().clone(), 3, 1, 0, 2);
    let statement = TerminalConstrainedRsStatement {
        linearizer_claims: claims.clone(),
    };
    let expected = statement.batched_linearizer_weight_eval_at(&code, xi, &whir_point);
    let mut challenger = NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
    let reference = prove_fixed_multi_air_linearizer_adjoint_reference(
        &claims,
        &batching_scales,
        &whir_point,
        &mut challenger,
    )
    .expect("small exact adjoint reference proof");
    assert_eq!(reference.initial_claim, expected);
    let transcript = TranscriptHistory::into_log(challenger.into_inner());

    let raw = generate_fixed_multi_air_linearizer_raw_traces(
        3,
        &components(),
        &claims,
        &batching_scales,
        &reference.point,
        None,
    )
    .expect("raw structured weight trace");
    let (y_cached, y_common, selector_factors) =
        generate_fixed_multi_air_linearizer_y_traces(&reference.point, &whir_point, 0, None)
            .expect("DFT kernel trace");
    let (selector_trace, selector) =
        generate_fixed_multi_air_linearizer_selector_trace(&selector_factors, None)
            .expect("inverse-transpose zeta selector");
    assert_eq!(reference.final_claim, raw.value * selector);

    let point_counts = raw
        .aux_point_lookup_counts
        .iter()
        .map(|&count| count + reference.point.len())
        .collect::<Vec<_>>();
    let sumcheck_air = FixedMultiAirLinearizerSumcheckAir {
        transcript_bus: TranscriptBus::new(0),
        point_bus: FixedMultiAirLinearizerAuxPointBus::new(1),
        final_bus: FixedMultiAirLinearizerSumcheckFinalBus::new(2),
        log_message_len: 3,
    };
    let sumcheck = generate_fixed_multi_air_linearizer_sumcheck_traces(
        &sumcheck_air,
        &reference.proof,
        &transcript,
        0,
        &point_counts,
        None,
    )
    .expect("bounded sumcheck trace");
    assert_eq!(sumcheck.initial_claim, expected);
    assert_eq!(sumcheck.final_claim, reference.final_claim);
    assert_eq!(sumcheck.point, reference.point);

    let raw_air = FixedMultiAirLinearizerRawAir {
        batched_claim_bus: FixedMultiAirBatchedClaimBus::new(3),
        mapped_term_bus: FixedMultiAirMappedTermBus::new(4),
        structured_point_bus: FixedMultiAirStructuredPointBus::new(5),
        aux_point_bus: FixedMultiAirLinearizerAuxPointBus::new(1),
        raw_term_bus: FixedMultiAirLinearizerRawTermBus::new(6),
        log_message_len: 3,
        component_count: 3,
    };
    let y_air = FixedMultiAirLinearizerYAir {
        point_bus: FixedMultiAirLinearizerAuxPointBus::new(1),
        whir_point_bus: NativeTerminalWhirPointBus::new(8),
        y_bus: FixedMultiAirLinearizerYBus::new(7),
        log_message_len: 3,
        initial_folding_factor: 0,
    };
    let selector_air = FixedMultiAirLinearizerSelectorAir {
        y_bus: FixedMultiAirLinearizerYBus::new(7),
        selector_bus: FixedMultiAirLinearizerSelectorBus::new(9),
        log_message_len: 3,
    };
    check_partitioned(&sumcheck_air, &[&sumcheck.cached], &sumcheck.common);
    check_partitioned(&raw_air, &[&raw.cached], &raw.common);
    check_partitioned(&y_air, &[&y_cached], &y_common);
    check_partitioned(&selector_air, &[], &selector_trace);

    let raw_sum_air = FixedMultiAirLinearizerRawSumAir {
        term_bus: FixedMultiAirLinearizerRawTermBus::new(6),
        output_bus: FixedMultiAirLinearizerRawWeightBus::new(10),
        component_count: 3,
    };
    let raw_sum = generate_fixed_multi_air_linearizer_raw_sum_trace(&raw.term_values, None)
        .expect("raw term sum");
    check_partitioned(&raw_sum_air, &[], &raw_sum);

    let final_air = FixedMultiAirLinearizerFinalAir {
        statement_bus: NativeTerminalWhirStatementBus::new(11),
        sumcheck_bus: FixedMultiAirLinearizerSumcheckFinalBus::new(2),
        raw_weight_bus: FixedMultiAirLinearizerRawWeightBus::new(10),
        selector_bus: FixedMultiAirLinearizerSelectorBus::new(9),
        output_bus: NativeTerminalWhirLinearizerWeightBus::new(12),
    };
    let final_trace = generate_fixed_multi_air_linearizer_final_trace(
        17,
        [F::from_u64(61); 8],
        xi,
        ef(67),
        expected,
        reference.final_claim,
        raw.value,
        selector,
    )
    .expect("exact final adjoint bridge");
    check_partitioned(&final_air, &[], &final_trace);

    // Adjoint certificate mutation: an evaluation no longer satisfies the
    // checked interpolation recurrence.
    let mut forged_sumcheck = sumcheck.common.clone();
    forged_sumcheck.values[0] += F::ONE;
    assert!(catch_unwind(AssertUnwindSafe(|| {
        check_partitioned(&sumcheck_air, &[&sumcheck.cached], &forged_sumcheck);
    }))
    .is_err());

    // Mapped-column identity mutation is rejected by the setup-fixed plan
    // validation before any trace is emitted.
    let mut forged_components = components();
    forged_components[1].block_start = 5;
    assert_eq!(
        generate_fixed_multi_air_linearizer_raw_traces(
            3,
            &forged_components,
            &claims,
            &batching_scales,
            &reference.point,
            None,
        )
        .expect_err("misaligned mapped coordinate must reject"),
        FixedMultiAirLinearizerAdjointTraceError::Shape,
    );

    assert_eq!(
        generate_fixed_multi_air_linearizer_final_trace(
            17,
            [F::from_u64(61); 8],
            xi,
            ef(67),
            expected,
            reference.final_claim + EF::ONE,
            raw.value,
            selector,
        )
        .expect_err("forged adjoint endpoint must reject"),
        FixedMultiAirLinearizerAdjointTraceError::Claim,
    );
}

#[test]
fn fixed_multi_air_vector_linearizer_adjoint_matches_factored_backend_layout() {
    let claims = claims();
    let xi = ef(43);
    let batching_scales = vec![xi, xi.square()];
    let whir_point = vec![ef(47), ef(53), ef(59)];
    let config = SC::default_from_params(SystemParams::new_for_testing(6));
    let code = WhirInitialRsWarpCode::new(config.hasher().clone(), 3, 1, 2, 1);
    let statement = TerminalConstrainedRsStatement {
        linearizer_claims: claims.clone(),
    };
    let expected = statement.batched_linearizer_weight_eval_at(&code, xi, &whir_point);
    let mut challenger = NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
    let reference = prove_fixed_multi_air_linearizer_adjoint_reference_factored(
        &claims,
        &batching_scales,
        &whir_point,
        2,
        &mut challenger,
    )
    .expect("small factored vector adjoint reference proof");
    assert_eq!(reference.initial_claim, expected);
    let mut tiled_challenger = NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
    let tiled = prove_fixed_multi_air_linearizer_adjoint_tiled_factored(
        &components(),
        &claims,
        &batching_scales,
        &whir_point,
        2,
        &mut tiled_challenger,
        FixedMultiAirLinearizerAdjointTiledConfig { tile_log_size: 1 },
    )
    .expect("small factored vector tiled adjoint proof");
    assert_eq!(tiled, reference);

    let raw = generate_fixed_multi_air_linearizer_raw_traces(
        3,
        &components(),
        &claims,
        &batching_scales,
        &reference.point,
        None,
    )
    .expect("vector raw structured weight trace");
    let (y_cached, y_common, selector_factors) =
        generate_fixed_multi_air_linearizer_y_traces(&reference.point, &whir_point, 2, None)
            .expect("vector factored selector trace");
    let (selector_trace, selector) =
        generate_fixed_multi_air_linearizer_selector_trace(&selector_factors, None)
            .expect("vector selector product trace");
    assert_eq!(reference.final_claim, raw.value * selector);

    let y_air = FixedMultiAirLinearizerYAir {
        point_bus: FixedMultiAirLinearizerAuxPointBus::new(1),
        whir_point_bus: NativeTerminalWhirPointBus::new(2),
        y_bus: FixedMultiAirLinearizerYBus::new(3),
        log_message_len: 3,
        initial_folding_factor: 2,
    };
    let selector_air = FixedMultiAirLinearizerSelectorAir {
        y_bus: FixedMultiAirLinearizerYBus::new(3),
        selector_bus: FixedMultiAirLinearizerSelectorBus::new(4),
        log_message_len: 3,
    };
    check_partitioned(&y_air, &[&y_cached], &y_common);
    check_partitioned(&selector_air, &[], &selector_trace);
}

#[test]
fn fixed_multi_air_tiled_linearizer_adjoint_matches_reference_challenge_for_challenge() {
    let claims = claims();
    let xi = ef(43);
    let batching_scales = vec![xi, xi.square()];
    let whir_point = vec![ef(47), ef(53), ef(59)];
    let mut reference_challenger =
        NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
    let reference = prove_fixed_multi_air_linearizer_adjoint_reference(
        &claims,
        &batching_scales,
        &whir_point,
        &mut reference_challenger,
    )
    .expect("reference structured adjoint");
    let reference_log = TranscriptHistory::into_log(reference_challenger.into_inner());

    for tile_log_size in [0, 1, 2, 8] {
        let mut tiled_challenger =
            NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
        let tiled = prove_fixed_multi_air_linearizer_adjoint_tiled(
            &components(),
            &claims,
            &batching_scales,
            &whir_point,
            &mut tiled_challenger,
            FixedMultiAirLinearizerAdjointTiledConfig { tile_log_size },
        )
        .expect("tiled structured adjoint");
        let tiled_log = TranscriptHistory::into_log(tiled_challenger.into_inner());
        assert_eq!(tiled, reference, "tile log {tile_log_size}");
        assert_eq!(tiled_log.values(), reference_log.values());
        assert_eq!(tiled_log.samples(), reference_log.samples());
    }
}

#[test]
fn fixed_multi_air_tiled_adjoint_matches_dense_unaligned_current_and_next_ef4() {
    for rotation in [
        PrismalinearMappedColumnRotation::Current,
        PrismalinearMappedColumnRotation::Next,
    ] {
        let weight = PrismalinearMappedColumnWeight {
            log_message_len: 4,
            terms: vec![PrismalinearMappedColumnTerm {
                block: PrismalinearMappedColumnBlock {
                    start: 7,
                    log_height: 3,
                },
                l_skip: 0,
                barycentric_weights: vec![EF::ONE],
                folded_row_eq_point: vec![ef4(3), ef4(11), ef4(19)],
                rotation,
                scale: ef4(29),
            }],
        };
        let claims = vec![TerminalStructuredLinearClaim::new(
            TerminalWeightSpec::PrismalinearMappedColumns(weight.clone()),
            ef4(37),
        )];
        let batching_scales = vec![ef4(43)];
        let whir_point = vec![ef4(53), ef4(61), ef4(71), ef4(79)];
        let endpoint = vec![ef4(89), ef4(97), ef4(107), ef4(127)];
        let dense = evaluate_mle(&weight.materialize(), &endpoint);
        let backend =
            evaluate_packed_column_eq_mle(&endpoint, 7, &[ef4(3), ef4(11), ef4(19)], rotation)
                .expect("backend two-carry endpoint");
        assert_eq!(backend * ef4(29), dense, "rotation {rotation:?}");

        let components = vec![FixedMultiAirLinearizerRawComponentPlan {
            ordinal: 0,
            claim: 0,
            term: 0,
            is_eq: false,
            is_zero: false,
            block_start: 7,
            log_height: 3,
            rotation: match rotation {
                PrismalinearMappedColumnRotation::Current => DirectAirMappedRotation::Current,
                PrismalinearMappedColumnRotation::Next => DirectAirMappedRotation::Next,
            },
        }];
        let mut reference_challenger =
            NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
        let reference = prove_fixed_multi_air_linearizer_adjoint_reference_factored(
            &claims,
            &batching_scales,
            &whir_point,
            2,
            &mut reference_challenger,
        )
        .expect("dense unaligned adjoint reference");
        let reference_log = TranscriptHistory::into_log(reference_challenger.into_inner());

        let mut tiled_challenger =
            NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
        let tiled = prove_fixed_multi_air_linearizer_adjoint_tiled_factored(
            &components,
            &claims,
            &batching_scales,
            &whir_point,
            2,
            &mut tiled_challenger,
            FixedMultiAirLinearizerAdjointTiledConfig { tile_log_size: 2 },
        )
        .expect("tiled unaligned adjoint proof");
        let tiled_log = TranscriptHistory::into_log(tiled_challenger.into_inner());
        assert_eq!(tiled, reference, "rotation {rotation:?}");
        assert_eq!(tiled_log.values(), reference_log.values());
        assert_eq!(tiled_log.samples(), reference_log.samples());
    }
}

#[test]
fn fixed_multi_air_raw_air_matches_native_two_carry_for_dynamic_packed_coordinates() {
    for log_message_len in 2..=6 {
        let message_len = 1usize << log_message_len;
        let aux_point = (0..log_message_len)
            .map(|coordinate| ef4(101 * log_message_len as u64 + 17 * coordinate as u64 + 3))
            .collect::<Vec<_>>();
        for log_height in 0..=log_message_len {
            let block_len = 1usize << log_height;
            let max_start = message_len - block_len;
            let mut starts = vec![0, 1.min(max_start), 3.min(max_start), max_start];
            starts.sort_unstable();
            starts.dedup();
            for start in starts {
                for rotation in [
                    PrismalinearMappedColumnRotation::Current,
                    PrismalinearMappedColumnRotation::Next,
                ] {
                    let local_point = (0..log_height)
                        .map(|coordinate| ef4(1_003 + 29 * start as u64 + 31 * coordinate as u64))
                        .collect::<Vec<_>>();
                    let term_scale = ef4(2_003 + start as u64);
                    let batching_scale = ef4(3_007 + log_height as u64);
                    let weight = PrismalinearMappedColumnWeight {
                        log_message_len,
                        terms: vec![PrismalinearMappedColumnTerm {
                            block: PrismalinearMappedColumnBlock { start, log_height },
                            l_skip: 0,
                            barycentric_weights: vec![EF::ONE],
                            folded_row_eq_point: local_point.clone(),
                            rotation,
                            scale: term_scale,
                        }],
                    };
                    let claims = vec![TerminalStructuredLinearClaim::new(
                        TerminalWeightSpec::PrismalinearMappedColumns(weight.clone()),
                        ef4(4_009),
                    )];
                    let components = vec![FixedMultiAirLinearizerRawComponentPlan {
                        ordinal: 0,
                        claim: 0,
                        term: 0,
                        is_eq: false,
                        is_zero: false,
                        block_start: start,
                        log_height,
                        rotation: match rotation {
                            PrismalinearMappedColumnRotation::Current => {
                                DirectAirMappedRotation::Current
                            }
                            PrismalinearMappedColumnRotation::Next => DirectAirMappedRotation::Next,
                        },
                    }];
                    let raw = generate_fixed_multi_air_linearizer_raw_traces(
                        log_message_len,
                        &components,
                        &claims,
                        &[batching_scale],
                        &aux_point,
                        None,
                    )
                    .expect("dynamic exact two-carry raw trace");
                    let dense = evaluate_mle(&weight.materialize(), &aux_point);
                    let native =
                        evaluate_packed_column_eq_mle(&aux_point, start, &local_point, rotation)
                            .expect("native packed-column evaluator");
                    assert_eq!(dense, term_scale * native);
                    assert_eq!(raw.value, batching_scale * dense);

                    let air = FixedMultiAirLinearizerRawAir {
                        batched_claim_bus: FixedMultiAirBatchedClaimBus::new(30),
                        mapped_term_bus: FixedMultiAirMappedTermBus::new(31),
                        structured_point_bus: FixedMultiAirStructuredPointBus::new(32),
                        aux_point_bus: FixedMultiAirLinearizerAuxPointBus::new(33),
                        raw_term_bus: FixedMultiAirLinearizerRawTermBus::new(34),
                        log_message_len,
                        component_count: 1,
                    };
                    check_partitioned(&air, &[&raw.cached], &raw.common);

                    if log_message_len == 4
                        && log_height == 2
                        && start == 3
                        && rotation == PrismalinearMappedColumnRotation::Next
                    {
                        // Confirm that the constraints, rather than only the
                        // trace generator, bind the two-carry endpoint.
                        let mut mutated = raw.common.clone();
                        let width = FixedMultiAirLinearizerRawCols::<F>::width();
                        let first: &mut FixedMultiAirLinearizerRawCols<F> =
                            mutated.values[..width].borrow_mut();
                        first.state_after[0][0] += F::ONE;
                        assert!(catch_unwind(AssertUnwindSafe(|| {
                            check_partitioned(&air, &[&raw.cached], &mutated);
                        }))
                        .is_err());
                    }
                }
            }
        }
    }
}

#[test]
fn coefficient_two_coset_selector_uses_full_message_dimension_not_whir_k() {
    let claims = claims();
    let xi = ef4(5_003);
    let batching_scales = vec![xi, xi.square()];
    let whir_point = vec![ef4(5_009), ef4(5_021), ef4(5_039)];
    let config = SC::default_from_params(SystemParams::new_for_testing(6));
    let code =
        WhirInitialRsWarpCode::new_coefficient_two_coset(config.hasher().clone(), 3, 1, 0, 1);
    let statement = TerminalConstrainedRsStatement {
        linearizer_claims: claims.clone(),
    };
    let expected = statement.batched_linearizer_weight_eval_at(&code, xi, &whir_point);

    let mut reference_challenger =
        NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
    let reference = prove_fixed_multi_air_linearizer_adjoint_reference_factored_for_layout(
        &claims,
        &batching_scales,
        &whir_point,
        0,
        WhirInitialRsLayout::CoefficientTwoCosetGrs,
        &mut reference_challenger,
    )
    .expect("coefficient two-coset dense adjoint");
    let reference_log = TranscriptHistory::into_log(reference_challenger.into_inner());
    assert_eq!(reference.initial_claim, expected);

    let mut tiled_challenger = NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
    let tiled = prove_fixed_multi_air_linearizer_adjoint_tiled_factored_for_layout(
        &components(),
        &claims,
        &batching_scales,
        &whir_point,
        0,
        WhirInitialRsLayout::CoefficientTwoCosetGrs,
        &mut tiled_challenger,
        FixedMultiAirLinearizerAdjointTiledConfig { tile_log_size: 1 },
    )
    .expect("coefficient two-coset tiled adjoint");
    let tiled_log = TranscriptHistory::into_log(tiled_challenger.into_inner());
    assert_eq!(tiled, reference);
    assert_eq!(tiled_log.values(), reference_log.values());
    assert_eq!(tiled_log.samples(), reference_log.samples());

    let (y_cached, y_common, factors) = generate_fixed_multi_air_linearizer_y_traces_for_layout(
        &reference.point,
        &whir_point,
        0,
        WhirInitialRsLayout::CoefficientTwoCosetGrs,
        None,
    )
    .expect("layout-derived coefficient selector");
    let (selector_trace, selector) =
        generate_fixed_multi_air_linearizer_selector_trace(&factors, None)
            .expect("coefficient selector product");
    let (_, _, wrong_k_factors) =
        generate_fixed_multi_air_linearizer_y_traces(&reference.point, &whir_point, 2, None)
            .expect("deliberately wrong WHIR-k selector");
    let (_, wrong_k_selector) =
        generate_fixed_multi_air_linearizer_selector_trace(&wrong_k_factors, None)
            .expect("deliberately wrong selector product");
    assert_ne!(
        wrong_k_selector, selector,
        "using WHIR k must be observably different from the coefficient-layout selector",
    );
    let raw = generate_fixed_multi_air_linearizer_raw_traces(
        3,
        &components(),
        &claims,
        &batching_scales,
        &reference.point,
        None,
    )
    .expect("coefficient raw trace");
    assert_eq!(reference.final_claim, raw.value * selector);
    assert_eq!(
        structured_selector_folding_factor(WhirInitialRsLayout::CoefficientTwoCosetGrs, 3, 0,),
        Ok(3),
    );

    let y_air = FixedMultiAirLinearizerYAir {
        point_bus: FixedMultiAirLinearizerAuxPointBus::new(40),
        whir_point_bus: NativeTerminalWhirPointBus::new(41),
        y_bus: FixedMultiAirLinearizerYBus::new(42),
        log_message_len: 3,
        // This is the selector dimension, deliberately not WHIR's k.
        initial_folding_factor: 3,
    };
    let selector_air = FixedMultiAirLinearizerSelectorAir {
        y_bus: FixedMultiAirLinearizerYBus::new(42),
        selector_bus: FixedMultiAirLinearizerSelectorBus::new(43),
        log_message_len: 3,
    };
    check_partitioned(&y_air, &[&y_cached], &y_common);
    check_partitioned(&selector_air, &[], &selector_trace);

    // A coefficient layout must fail closed if an adapter substitutes WHIR
    // k (or any nonzero initial folding factor) for the selector dimension.
    let malformed = catch_unwind(AssertUnwindSafe(|| {
        let mut challenger = NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
        prove_fixed_multi_air_linearizer_adjoint_reference_factored_for_layout(
            &claims,
            &batching_scales,
            &whir_point,
            1,
            WhirInitialRsLayout::CoefficientTwoCosetGrs,
            &mut challenger,
        )
    }));
    assert!(malformed.is_ok(), "malformed layout must not panic");
    assert_eq!(
        malformed.expect("panic-free layout rejection"),
        Err(FixedMultiAirLinearizerAdjointTraceError::Shape),
    );
}

#[test]
fn packed_two_carry_malformed_ranges_reject_without_panicking() {
    let claims = claims();
    let batching_scales = vec![ef4(7), ef4(13)];
    let aux_point = vec![ef4(19), ef4(23), ef4(29)];
    for block_start in [5usize, usize::MAX] {
        let mut malformed = components();
        malformed[1].block_start = block_start;
        let result = catch_unwind(AssertUnwindSafe(|| {
            generate_fixed_multi_air_linearizer_raw_traces(
                3,
                &malformed,
                &claims,
                &batching_scales,
                &aux_point,
                None,
            )
        }));
        assert!(result.is_ok(), "packed range parser must not panic");
        assert!(matches!(
            result.expect("panic-free packed range rejection"),
            Err(FixedMultiAirLinearizerAdjointTraceError::Shape),
        ));
    }
}

#[test]
fn fixed_multi_air_tiled_linearizer_adjoint_log28_plan_has_no_dense_weight() {
    let plan = fixed_multi_air_linearizer_adjoint_tiled_plan(
        28,
        FixedMultiAirLinearizerAdjointTiledConfig { tile_log_size: 14 },
    )
    .expect("production structured adjoint plan");
    assert_eq!(plan.log_message_len, 28);
    assert_eq!(plan.dense_weight_elements, 0);
    assert!(
        plan.max_worker_field_elements <= 4 * FIXED_MULTI_AIR_LINEARIZER_MAX_EVALUATIONS,
        "worker scratch must remain bounded by the protocol degree"
    );
}

use core::borrow::{Borrow, BorrowMut};
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
};

use openvm_stark_backend::{
    air_builders::{
        debug::{check_constraints, check_logup},
        symbolic::{get_symbolic_builder, SymbolicRapBuilder},
    },
    interaction::{InteractionBuilder, SymbolicInteraction},
    keygen::types::TraceWidth,
    native_warp::{
        FINITE_COMPLETE_TERMINAL_PACKAGE_VERSION, FINITE_COMPLETE_TERMINAL_STATEMENT_TAG,
    },
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField32},
    transcript::{TranscriptHistory, TranscriptLog},
    warp_accum::{TerminalDescriptor, WhirInitialRsWarpCode},
    warp_pesat::{AccumulatorInstance, AlgebraicChallenger},
    BaseAirWithPublicValues, FiatShamirTranscript, PartitionedBaseAir, StarkProtocolConfig,
    SystemParams, WhirConfig, WhirProximityStrategy, WhirRoundConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, BabyBearPoseidon2Config as SC, Digest, DuplexSpongeRecorder,
    DIGEST_SIZE, D_EF, EF, F,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::*;

type Code = WhirInitialRsWarpCode<
    <SC as StarkProtocolConfig>::Hasher,
    openvm_stark_backend::warp_accum::FieldElementDigestObserver,
>;

const TRANSCRIPT_BUS_INDEX: u16 = 1_100;
const START_BUS_INDEX: u16 = 1_101;
const COMPLETE_START_BUS_INDEX: u16 = 1_102;
const BINDING_BUS_INDEX: u16 = 1_103;
const INSTANCE_BUS_INDEX: u16 = 1_104;
const PRELUDE_LEN: usize = 5;

struct Fixture {
    profile: Arc<FiniteWarpV3TerminalStatementPrefixProfile>,
    instance: AccumulatorInstance<EF, Digest>,
    accumulator_digest: Digest,
    checkpoint: FiniteWarpV3TerminalTranscriptCheckpoint,
    transcript: TranscriptLog<F, [F; 16]>,
}

fn digest(seed: u32) -> Digest {
    core::array::from_fn(|coordinate| F::from_u32(seed + coordinate as u32 * 17 + 1))
}

fn extension(seed: u32) -> EF {
    let limbs: [F; D_EF] = core::array::from_fn(|limb| F::from_u32(seed + limb as u32 * 29 + 1));
    EF::from_basis_coefficients_slice(&limbs).expect("four BabyBear coefficients form EF4")
}

fn observe_u64_oracle(transcript: &mut DuplexSpongeRecorder, value: u64) {
    for shift in [0, 16, 32, 48] {
        <DuplexSpongeRecorder as FiatShamirTranscript<SC>>::observe(
            transcript,
            F::from_u32(((value >> shift) & 0xffff) as u32),
        );
    }
}

fn observe_digest_oracle(transcript: &mut DuplexSpongeRecorder, value: Digest) {
    <DuplexSpongeRecorder as FiatShamirTranscript<SC>>::observe_commit(transcript, value);
}

fn observe_extension_oracle(transcript: &mut DuplexSpongeRecorder, value: EF) {
    <DuplexSpongeRecorder as FiatShamirTranscript<SC>>::observe_ext(transcript, value);
}

#[derive(Default)]
struct AlgebraicOracle(Vec<EF>);

impl AlgebraicChallenger<EF> for AlgebraicOracle {
    fn observe(&mut self, value: EF) {
        self.0.push(value);
    }

    fn sample(&mut self) -> EF {
        panic!("the terminal descriptor prefix must not sample")
    }
}

/// Reconstruct the native wire order without consulting `PrefixSource` or the
/// setup schedule under test.
fn native_transcript_oracle(
    code: &Code,
    descriptor: &TerminalDescriptor<Digest>,
    relation_digest: Digest,
    terminal_index_digest: Digest,
    instance: &AccumulatorInstance<EF, Digest>,
) -> (
    FiniteWarpV3TerminalTranscriptCheckpoint,
    TranscriptLog<F, [F; 16]>,
) {
    let mut transcript = default_duplex_sponge_recorder();
    for value in 0..PRELUDE_LEN {
        <DuplexSpongeRecorder as FiatShamirTranscript<SC>>::observe(
            &mut transcript,
            F::from_u32(70_000 + value as u32),
        );
    }
    let sponge = transcript.inner.checkpoint();
    let checkpoint = FiniteWarpV3TerminalTranscriptCheckpoint {
        tidx: transcript
            .log
            .len()
            .try_into()
            .expect("small test transcript"),
        sample_count: 0,
        state: sponge.state,
    };

    // Native `observe_terminal_statement`.
    observe_u64_oracle(&mut transcript, FINITE_COMPLETE_TERMINAL_STATEMENT_TAG);
    observe_u64_oracle(
        &mut transcript,
        u64::from(FINITE_COMPLETE_TERMINAL_PACKAGE_VERSION),
    );
    observe_digest_oracle(&mut transcript, relation_digest);
    observe_digest_oracle(&mut transcript, terminal_index_digest);
    descriptor.observe_fiat_shamir::<SC, _>(&mut transcript);
    observe_digest_oracle(&mut transcript, instance.rt);
    observe_u64_oracle(&mut transcript, instance.alpha.len() as u64);
    for &value in &instance.alpha {
        observe_extension_oracle(&mut transcript, value);
    }
    observe_extension_oracle(&mut transcript, instance.mu);
    observe_u64_oracle(&mut transcript, instance.beta.len() as u64);
    for &value in &instance.beta {
        observe_extension_oracle(&mut transcript, value);
    }
    observe_extension_oracle(&mut transcript, instance.eta);

    // Native constrained-terminal prefix. The descriptor itself owns this
    // algebraic serialization; the test only bridges each EF4 observation to
    // the physical transcript recorder.
    let mut algebraic = AlgebraicOracle::default();
    descriptor.observe_algebraic::<F, EF, _, _>(code, &mut algebraic);
    for value in algebraic.0 {
        observe_extension_oracle(&mut transcript, value);
    }
    for &value in &instance.alpha {
        observe_extension_oracle(&mut transcript, value);
    }
    observe_extension_oracle(&mut transcript, instance.mu);
    for &value in &instance.beta {
        observe_extension_oracle(&mut transcript, value);
    }
    observe_extension_oracle(&mut transcript, instance.eta);

    (checkpoint, TranscriptHistory::into_log(transcript))
}

fn fixture(seed: u32) -> Fixture {
    let config = SC::default_from_params(SystemParams::new_for_testing(10));
    let code =
        WhirInitialRsWarpCode::try_new_coefficient_two_coset(config.hasher().clone(), 8, 1, 0, 16)
            .expect("test two-coset code");
    let whir = WhirConfig {
        k: 4,
        rounds: vec![WhirRoundConfig { num_queries: 2 }],
        mu_pow_bits: 0,
        query_phase_pow_bits: 0,
        folding_pow_bits: 0,
        proximity: WhirProximityStrategy::UniqueDecoding,
    };
    let root = digest(seed + 100);
    let descriptor = TerminalDescriptor::from_whir_initial_rs(root, &code, &whir, D_EF);
    let two_coset = FiniteWarpV3TwoCosetTerminalProfile::new(&descriptor, &code, &whir)
        .expect("trusted two-coset profile");
    let relation_digest = digest(seed + 200);
    let terminal_index_digest = digest(seed + 300);
    let instance = AccumulatorInstance {
        rt: root,
        alpha: (0..two_coset.alpha_len())
            .map(|coordinate| extension(seed + 1_000 + coordinate as u32 * 101))
            .collect(),
        mu: extension(seed + 2_000),
        beta: vec![extension(seed + 3_000), extension(seed + 3_101)],
        eta: extension(seed + 4_000),
    };
    let profile = Arc::new(
        FiniteWarpV3TerminalStatementPrefixProfile::new(
            &two_coset,
            relation_digest,
            terminal_index_digest,
            instance.beta.len(),
        )
        .expect("terminal statement-prefix profile"),
    );
    let accumulator_digest = digest(seed + 500);
    let (checkpoint, transcript) = native_transcript_oracle(
        &code,
        &descriptor,
        relation_digest,
        terminal_index_digest,
        &instance,
    );
    Fixture {
        profile,
        instance,
        accumulator_digest,
        checkpoint,
        transcript,
    }
}

fn prefix_air(
    profile: Arc<FiniteWarpV3TerminalStatementPrefixProfile>,
) -> FiniteWarpV3TerminalStatementPrefixAir {
    FiniteWarpV3TerminalStatementPrefixAir {
        profile,
        transcript_bus: TranscriptBus::new(TRANSCRIPT_BUS_INDEX),
        start_bus: FiniteWarpV3TerminalStatementStartBus::new(START_BUS_INDEX),
        complete_start_bus: FixedMultiAirCompleteStatementStartCursorBus::new(
            COMPLETE_START_BUS_INDEX,
        ),
        binding_bus: FixedMultiAirCompleteBindingBus::new(BINDING_BUS_INDEX),
        instance_bus: FixedMultiAirCompleteInstanceValueBus::new(INSTANCE_BUS_INDEX),
    }
}

fn generate(
    fixture: &Fixture,
    required_height: Option<usize>,
) -> Result<FiniteWarpV3TerminalStatementPrefixTraceData, FiniteWarpV3TerminalStatementPrefixError>
{
    generate_finite_warp_v3_terminal_statement_prefix_trace(
        &fixture.profile,
        &fixture.instance,
        fixture.accumulator_digest,
        &fixture.transcript,
        fixture.checkpoint,
        required_height,
    )
}

fn check_prefix_constraints(
    air: &FiniteWarpV3TerminalStatementPrefixAir,
    trace: &FiniteWarpV3TerminalStatementPrefixTraceData,
) {
    check_constraints::<_, SC>(
        air,
        "finite WARP v3 terminal statement prefix",
        &None,
        &[trace.cached.as_view(), trace.common.as_view()],
        &[],
    );
}

fn constraints_reject(
    air: &FiniteWarpV3TerminalStatementPrefixAir,
    trace: &FiniteWarpV3TerminalStatementPrefixTraceData,
) -> bool {
    catch_unwind(AssertUnwindSafe(|| check_prefix_constraints(air, trace))).is_err()
}

fn common_cols_mut(
    trace: &mut FiniteWarpV3TerminalStatementPrefixTraceData,
    row: usize,
) -> &mut FiniteWarpV3TerminalStatementPrefixCols<F> {
    let width = trace.common.width();
    trace.common.values[row * width..(row + 1) * width].borrow_mut()
}

fn common_cols(
    trace: &FiniteWarpV3TerminalStatementPrefixTraceData,
    row: usize,
) -> &FiniteWarpV3TerminalStatementPrefixCols<F> {
    let width = trace.common.width();
    trace.common.values[row * width..(row + 1) * width].borrow()
}

#[derive(Clone, Copy)]
struct InstanceAuthorityAir {
    bus: FixedMultiAirCompleteInstanceValueBus,
}

impl BaseAir<F> for InstanceAuthorityAir {
    fn width(&self) -> usize {
        3 + D_EF
    }
}

impl BaseAirWithPublicValues<F> for InstanceAuthorityAir {}
impl PartitionedBaseAir<F> for InstanceAuthorityAir {}

impl<AB> Air<AB> for InstanceAuthorityAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("instance authority row");
        self.bus.add_key_with_lookups(
            builder,
            FixedMultiAirCompleteInstanceValueMessage {
                section: row[1],
                coordinate: row[2],
                value: core::array::from_fn(|limb| row[3 + limb]),
            },
            row[0],
        );
    }
}

fn instance_entries(instance: &AccumulatorInstance<EF, Digest>) -> Vec<(usize, usize, EF)> {
    instance
        .alpha
        .iter()
        .copied()
        .enumerate()
        .map(|(coordinate, value)| (0, coordinate, value))
        .chain(core::iter::once((1, 0, instance.mu)))
        .chain(
            instance
                .beta
                .iter()
                .copied()
                .enumerate()
                .map(|(coordinate, value)| (2, coordinate, value)),
        )
        .chain(core::iter::once((3, 0, instance.eta)))
        .collect()
}

fn instance_authority_trace(instance: &AccumulatorInstance<EF, Digest>) -> RowMajorMatrix<F> {
    let entries = instance_entries(instance);
    let width = 3 + D_EF;
    let height = entries.len().next_power_of_two();
    let mut values = F::zero_vec(height * width);
    for (row, (section, coordinate, value)) in entries.into_iter().enumerate() {
        let dst = &mut values[row * width..(row + 1) * width];
        dst[0] = F::from_u32(2); // outer statement and constrained prefix
        dst[1] = F::from_usize(section);
        dst[2] = F::from_usize(coordinate);
        dst[3..].copy_from_slice(value.as_basis_coefficients_slice());
    }
    RowMajorMatrix::new(values, width)
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

fn check_instance_authentication(
    air: &FiniteWarpV3TerminalStatementPrefixAir,
    trace: &FiniteWarpV3TerminalStatementPrefixTraceData,
    authority_trace: &RowMajorMatrix<F>,
) {
    let authority = InstanceAuthorityAir {
        bus: FixedMultiAirCompleteInstanceValueBus::new(INSTANCE_BUS_INDEX),
    };
    let prefix_interactions = symbolic_interactions(air)
        .into_iter()
        .filter(|interaction| interaction.bus_index == INSTANCE_BUS_INDEX)
        .collect::<Vec<_>>();
    let authority_interactions = symbolic_interactions(&authority);
    assert_eq!(prefix_interactions.len(), 1);
    assert_eq!(authority_interactions.len(), 1);
    check_logup(
        &[
            "terminal statement prefix instance lookups".to_owned(),
            "independent native instance catalog".to_owned(),
        ],
        &[prefix_interactions, authority_interactions],
        &[None, None],
        &[
            vec![trace.cached.as_view(), trace.common.as_view()],
            vec![authority_trace.as_view()],
        ],
        &[Vec::new(), Vec::new()],
    );
}

fn instance_group_starts(profile: &FiniteWarpV3TerminalStatementPrefixProfile) -> Vec<usize> {
    profile
        .schedule
        .iter()
        .enumerate()
        .filter_map(|(row, source)| {
            matches!(source, PrefixSource::Instance { limb: 0, .. }).then_some(row)
        })
        .collect()
}

#[test]
fn honest_trace_matches_independent_native_oracle_and_balances_instance_catalog() {
    for seed in [7, 71, 701] {
        let fixture = fixture(seed);
        let trace = generate(&fixture, None).expect("honest native prefix");
        let air = prefix_air(fixture.profile.clone());
        let start = fixture.checkpoint.tidx as usize;
        let end = start + fixture.profile.observation_len();
        assert_eq!(start, PRELUDE_LEN);
        assert_eq!(trace.end_tidx, end);
        assert_eq!(trace.cached.height(), trace.common.height());
        assert!(trace.common.height().is_power_of_two());
        for row in 0..fixture.profile.observation_len() {
            assert_eq!(
                common_cols(&trace, row).observed,
                fixture.transcript.values()[start + row],
                "native transcript mismatch at terminal-prefix row {row}"
            );
        }
        check_prefix_constraints(&air, &trace);
        check_instance_authentication(&air, &trace, &instance_authority_trace(&fixture.instance));
    }
}

#[test]
fn every_nonzero_ef4_limb_is_carried_and_authenticated_at_every_occurrence() {
    let fixture = fixture(13);
    let honest = generate(&fixture, None).expect("honest native prefix");
    let air = prefix_air(fixture.profile.clone());
    let authority = instance_authority_trace(&fixture.instance);
    let groups = instance_group_starts(&fixture.profile);
    assert_eq!(
        groups.len(),
        2 * (fixture.instance.alpha.len() + 1 + fixture.instance.beta.len() + 1)
    );

    for group in groups {
        for limb in 1..D_EF {
            // Changing only the row that emits this limb preserves its local
            // selector equation but must violate the preceding carry.
            let mut broken_carry = FiniteWarpV3TerminalStatementPrefixTraceData {
                cached: honest.cached.clone(),
                common: honest.common.clone(),
                end_tidx: honest.end_tidx,
            };
            let local = common_cols_mut(&mut broken_carry, group + limb);
            local.instance_value[limb] += F::ONE;
            local.observed += F::ONE;
            assert!(
                constraints_reject(&air, &broken_carry),
                "EF4 limb {limb} at group row {group} was not carried"
            );

            // Changing the same limb coherently across the whole group keeps
            // all local/carry constraints satisfied. It must nevertheless be
            // rejected because the limb-zero lookup authenticates all EF4
            // coordinates against the independent catalog.
            let mut forged_group = FiniteWarpV3TerminalStatementPrefixTraceData {
                cached: honest.cached.clone(),
                common: honest.common.clone(),
                end_tidx: honest.end_tidx,
            };
            for row in group..group + D_EF {
                common_cols_mut(&mut forged_group, row).instance_value[limb] += F::ONE;
            }
            common_cols_mut(&mut forged_group, group + limb).observed += F::ONE;
            check_prefix_constraints(&air, &forged_group);
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    check_instance_authentication(&air, &forged_group, &authority)
                }))
                .is_err(),
                "EF4 limb {limb} at group row {group} escaped instance authentication"
            );
        }
    }
}

#[test]
fn every_root_observation_and_root_coordinate_is_bound() {
    let fixture = fixture(19);
    let root_rows = fixture
        .profile
        .schedule
        .iter()
        .enumerate()
        .filter_map(|(row, source)| matches!(source, PrefixSource::Root { .. }).then_some(row))
        .collect::<Vec<_>>();
    assert_eq!(root_rows.len(), 3 * DIGEST_SIZE);

    for row in root_rows {
        let mut transcript = fixture.transcript.clone();
        transcript.values_mut()[fixture.checkpoint.tidx as usize + row] += F::ONE;
        let result = generate_finite_warp_v3_terminal_statement_prefix_trace(
            &fixture.profile,
            &fixture.instance,
            fixture.accumulator_digest,
            &transcript,
            fixture.checkpoint,
            None,
        );
        assert!(matches!(
            result,
            Err(FiniteWarpV3TerminalStatementPrefixError::Transcript)
        ));
    }

    for coordinate in 0..DIGEST_SIZE {
        let mut wrong_instance = fixture.instance.clone();
        wrong_instance.rt[coordinate] += F::ONE;
        let result = generate_finite_warp_v3_terminal_statement_prefix_trace(
            &fixture.profile,
            &wrong_instance,
            fixture.accumulator_digest,
            &fixture.transcript,
            fixture.checkpoint,
            None,
        );
        assert!(matches!(
            result,
            Err(FiniteWarpV3TerminalStatementPrefixError::Transcript)
        ));
    }
}

#[test]
fn tidx_sample_and_order_mutations_are_rejected() {
    let fixture = fixture(23);

    for delta in [-1_i64, 1] {
        let mut checkpoint = fixture.checkpoint;
        checkpoint.tidx = u32::try_from(i64::from(checkpoint.tidx) + delta).unwrap();
        let result = generate_finite_warp_v3_terminal_statement_prefix_trace(
            &fixture.profile,
            &fixture.instance,
            fixture.accumulator_digest,
            &fixture.transcript,
            checkpoint,
            None,
        );
        assert!(matches!(
            result,
            Err(FiniteWarpV3TerminalStatementPrefixError::Transcript)
        ));
    }

    for row in 0..fixture.profile.observation_len() {
        let mut sampled = fixture.transcript.clone();
        sampled.samples_mut()[fixture.checkpoint.tidx as usize + row] = true;
        let result = generate_finite_warp_v3_terminal_statement_prefix_trace(
            &fixture.profile,
            &fixture.instance,
            fixture.accumulator_digest,
            &sampled,
            fixture.checkpoint,
            None,
        );
        assert!(matches!(
            result,
            Err(FiniteWarpV3TerminalStatementPrefixError::Transcript)
        ));
    }

    let start = fixture.checkpoint.tidx as usize;
    let values = &fixture.transcript.values()[start..start + fixture.profile.observation_len()];
    let (left, right) = values
        .windows(2)
        .enumerate()
        .find_map(|(row, pair)| (pair[0] != pair[1]).then_some((row, row + 1)))
        .expect("oracle contains distinct adjacent observations");
    let mut reordered = fixture.transcript.clone();
    reordered.values_mut().swap(start + left, start + right);
    let result = generate_finite_warp_v3_terminal_statement_prefix_trace(
        &fixture.profile,
        &fixture.instance,
        fixture.accumulator_digest,
        &reordered,
        fixture.checkpoint,
        None,
    );
    assert!(matches!(
        result,
        Err(FiniteWarpV3TerminalStatementPrefixError::Transcript)
    ));

    let mut trace = generate(&fixture, None).expect("honest native prefix");
    let air = prefix_air(fixture.profile.clone());
    common_cols_mut(&mut trace, 11).tidx += F::ONE;
    assert!(constraints_reject(&air, &trace));
}

#[test]
fn invalid_height_and_field_wrapping_fail_closed_without_allocating() {
    let fixture = fixture(29);
    let non_power_of_two = fixture
        .profile
        .observation_len()
        .next_power_of_two()
        .checked_add(1)
        .expect("test height");
    assert!(matches!(
        generate(&fixture, Some(non_power_of_two)),
        Err(FiniteWarpV3TerminalStatementPrefixError::Height)
    ));

    let mut wrapping = fixture.checkpoint;
    wrapping.tidx = F::ORDER_U32
        .checked_sub(fixture.profile.observation_len() as u32)
        .expect("prefix shorter than BabyBear order");
    let result = catch_unwind(AssertUnwindSafe(|| {
        generate_finite_warp_v3_terminal_statement_prefix_trace(
            &fixture.profile,
            &fixture.instance,
            fixture.accumulator_digest,
            &fixture.transcript,
            wrapping,
            None,
        )
    }));
    assert!(matches!(
        result,
        Ok(Err(FiniteWarpV3TerminalStatementPrefixError::Shape))
    ));
}

#[test]
fn symbolic_degree_stays_within_the_recursive_profile() {
    let fixture = fixture(31);
    let air = prefix_air(fixture.profile);
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

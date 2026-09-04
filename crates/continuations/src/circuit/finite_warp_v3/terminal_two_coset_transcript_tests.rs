use core::{
    any::TypeId,
    borrow::{Borrow, BorrowMut},
};
use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::Arc,
};

use openvm_recursion_circuit::{
    bus::{
        Poseidon2PermuteBus, ResumeTranscriptStateBus, ResumeTranscriptStateMessage, TranscriptBus,
    },
    transcript::transcript::{TranscriptAir, TranscriptCols, TranscriptResumeCols},
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::{
        debug::{check_constraints, check_logup},
        symbolic::get_symbolic_builder,
    },
    interaction::{InteractionBuilder, SymbolicInteraction},
    keygen::types::TraceWidth,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    warp_accum::{
        terminal_whir::observe_terminal_whir_layout_binding, TerminalDescriptor,
        WhirInitialRsWarpCode,
    },
    AirRef, AnyAir, BaseAirWithPublicValues, FiatShamirTranscript, PartitionedBaseAir,
    StarkProtocolConfig, SystemParams, TranscriptHistory, WhirProximityStrategy, WhirRoundConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, BabyBearPoseidon2Config as SC,
};

use super::*;

type Code = WhirInitialRsWarpCode<
    <SC as StarkProtocolConfig>::Hasher,
    openvm_stark_backend::warp_accum::FieldElementDigestObserver,
>;

fn digest(seed: u32) -> Digest {
    core::array::from_fn(|index| F::from_u32(seed + index as u32))
}

fn fixture() -> (
    Code,
    WhirConfig,
    TerminalDescriptor<Digest>,
    FiniteWarpV3TwoCosetTerminalProfile,
) {
    let config = SC::default_from_params(SystemParams::new_for_testing(10));
    let code =
        WhirInitialRsWarpCode::try_new_coefficient_two_coset(config.hasher().clone(), 8, 1, 0, 16)
            .unwrap();
    let whir = WhirConfig {
        k: 4,
        rounds: vec![WhirRoundConfig { num_queries: 2 }],
        mu_pow_bits: 0,
        query_phase_pow_bits: 0,
        folding_pow_bits: 0,
        proximity: WhirProximityStrategy::UniqueDecoding,
    };
    let descriptor = TerminalDescriptor::from_whir_initial_rs(digest(10), &code, &whir, D_EF);
    let profile = FiniteWarpV3TwoCosetTerminalProfile::new(&descriptor, &code, &whir).unwrap();
    (code, whir, descriptor, profile)
}

fn native_prefix_log(
    code: &Code,
    descriptor: &TerminalDescriptor<Digest>,
) -> TranscriptLog<F, [F; 16]> {
    let mut transcript = default_duplex_sponge_recorder();
    descriptor.observe_fiat_shamir::<SC, _>(&mut transcript);
    observe_terminal_whir_layout_binding::<SC, _, _>(code, &mut transcript).unwrap();
    let _ = <_ as FiatShamirTranscript<SC>>::sample_ext(&mut transcript);
    TranscriptHistory::into_log(transcript)
}

fn instance(root: Digest) -> AccumulatorInstance<EF, Digest> {
    AccumulatorInstance {
        rt: root,
        alpha: vec![EF::ZERO; 9],
        mu: EF::from(F::from_u32(42)),
        beta: vec![EF::ONE],
        eta: EF::ONE,
    }
}

#[test]
fn production_profile_is_exactly_the_native_two_block_transcript() {
    let (code, whir, descriptor, profile) = fixture();
    let rerooted_descriptor =
        TerminalDescriptor::from_whir_initial_rs(digest(1_000), &code, &whir, D_EF);
    let rerooted_profile =
        FiniteWarpV3TwoCosetTerminalProfile::new(&rerooted_descriptor, &code, &whir).unwrap();
    assert_eq!(profile, rerooted_profile);
    assert_eq!(profile.alpha_len(), 9);
    assert_eq!(profile.log_message_len(), 8);
    assert_eq!(profile.whir_k(), FINITE_WARP_V3_TWO_COSET_K);
    assert_eq!(profile.whir_round_count(), 1);
    assert_eq!(profile.whir_sumcheck_round_count(), 4);
    assert_eq!(profile.whir_remaining_dimension(), 4);
    assert_eq!(profile.final_poly_len(), 16);
    assert_eq!(profile.query_phase_pow_bits(), whir.query_phase_pow_bits);
    assert_eq!(profile.folding_pow_bits(), whir.folding_pow_bits);
    assert_eq!(profile.num_queries_per_round(), &[2]);
    let log = native_prefix_log(&code, &descriptor);
    assert!(!profile.descriptor_code_layout_block().is_empty());
    assert!(!profile.whir_code_layout_block().is_empty());
    assert_eq!(
        log.values().get(..profile.observation_len()),
        Some(profile.observations(descriptor.root).as_slice())
    );
    let (challenge, end) = profile
        .validate_transcript_prefix(&log, 0, descriptor.root)
        .unwrap();
    assert_eq!(end, log.len());
    assert_eq!(
        challenge,
        EF::from_basis_coefficients_slice(&log.values()[profile.observation_len()..]).unwrap()
    );

    let trace = generate_finite_warp_v3_two_coset_whir_prefix_trace(
        &profile,
        &descriptor,
        &instance(descriptor.root),
        &log,
        0,
    )
    .unwrap();
    assert_eq!(trace.height(), 1);
}

#[test]
fn profile_derives_multi_round_tail_geometry_from_trusted_setup() {
    let config = SC::default_from_params(SystemParams::new_for_testing(10));
    let code =
        WhirInitialRsWarpCode::try_new_coefficient_two_coset(config.hasher().clone(), 12, 1, 0, 16)
            .unwrap();
    let whir = WhirConfig {
        k: FINITE_WARP_V3_TWO_COSET_K,
        rounds: vec![
            WhirRoundConfig { num_queries: 3 },
            WhirRoundConfig { num_queries: 5 },
        ],
        mu_pow_bits: 2,
        query_phase_pow_bits: 7,
        folding_pow_bits: 11,
        proximity: WhirProximityStrategy::UniqueDecoding,
    };
    let descriptor = TerminalDescriptor::from_whir_initial_rs(digest(40), &code, &whir, D_EF);
    let profile = FiniteWarpV3TwoCosetTerminalProfile::new(&descriptor, &code, &whir).unwrap();

    assert_eq!(profile.alpha_len(), 13);
    assert_eq!(profile.log_message_len(), 12);
    assert_eq!(profile.whir_k(), 4);
    assert_eq!(profile.whir_round_count(), 2);
    assert_eq!(profile.whir_sumcheck_round_count(), 8);
    assert_eq!(profile.whir_remaining_dimension(), 4);
    assert_eq!(profile.final_poly_len(), 1 << 4);
    assert_eq!(profile.query_phase_pow_bits(), 7);
    assert_eq!(profile.folding_pow_bits(), 11);
    assert_eq!(profile.num_queries_per_round(), &[3, 5]);
}

#[test]
fn profile_rejects_inconsistent_whir_tail_geometry_without_panicking() {
    let config = SC::default_from_params(SystemParams::new_for_testing(10));
    let code =
        WhirInitialRsWarpCode::try_new_coefficient_two_coset(config.hasher().clone(), 8, 1, 0, 16)
            .unwrap();

    for whir in [
        WhirConfig {
            k: FINITE_WARP_V3_TWO_COSET_K,
            rounds: vec![],
            mu_pow_bits: 0,
            query_phase_pow_bits: 0,
            folding_pow_bits: 0,
            proximity: WhirProximityStrategy::UniqueDecoding,
        },
        WhirConfig {
            k: FINITE_WARP_V3_TWO_COSET_K,
            rounds: vec![WhirRoundConfig { num_queries: 1 }; 3],
            mu_pow_bits: 0,
            query_phase_pow_bits: 0,
            folding_pow_bits: 0,
            proximity: WhirProximityStrategy::UniqueDecoding,
        },
        WhirConfig {
            k: 3,
            rounds: vec![WhirRoundConfig { num_queries: 1 }],
            mu_pow_bits: 0,
            query_phase_pow_bits: 0,
            folding_pow_bits: 0,
            proximity: WhirProximityStrategy::UniqueDecoding,
        },
    ] {
        let descriptor = TerminalDescriptor::from_whir_initial_rs(digest(50), &code, &whir, D_EF);
        let result = catch_unwind(AssertUnwindSafe(|| {
            FiniteWarpV3TwoCosetTerminalProfile::new(&descriptor, &code, &whir)
        }));
        assert_eq!(
            result.unwrap(),
            Err(FiniteWarpV3TwoCosetTranscriptError::Geometry)
        );
    }
}

#[test]
fn complete_prefix_uses_distinct_typed_relation_seam_and_native_trace() {
    assert_ne!(
        TypeId::of::<FixedMultiAirCompleteBindingBus>(),
        TypeId::of::<FixedMultiAirTerminalBindingBus>()
    );
    assert_ne!(
        TypeId::of::<FixedMultiAirCompleteInstanceValueBus>(),
        TypeId::of::<FixedMultiAirTerminalInstanceValueBus>()
    );

    let (code, _whir, descriptor, profile) = fixture();
    let log = native_prefix_log(&code, &descriptor);
    let accumulator = instance(descriptor.root);
    let trace = generate_finite_warp_v3_complete_two_coset_whir_prefix_trace(
        &profile,
        &descriptor,
        &accumulator,
        &log,
        0,
    )
    .unwrap();
    let air = FiniteWarpV3CompleteTwoCosetWhirPrefixAir {
        transcript_bus: TranscriptBus::new(940),
        cursor_bus: FiniteWarpV3RsStatementEndBus::new(941),
        binding_bus: FixedMultiAirCompleteBindingBus::new(942),
        instance_bus: FixedMultiAirCompleteInstanceValueBus::new(943),
        start_bus: FixedMultiAirWhirStartBus::new(944),
        root_bus: NativeTerminalAccumulatorRootBus::new(945),
        relation_digest: digest(900),
        profile,
        beta_len: accumulator.beta.len(),
    };
    check_constraints::<_, SC>(
        &air,
        "complete coefficient-two-coset terminal prefix",
        &None,
        &[trace.as_view()],
        &[],
    );
}

#[test]
fn mutation_of_second_layout_block_is_rejected() {
    let (code, _whir, descriptor, profile) = fixture();
    let mut log = native_prefix_log(&code, &descriptor);
    let mutated = profile.whir_layout_block_offset();
    assert!(mutated < profile.observation_len());
    log.values_mut()[mutated] += F::ONE;
    assert_eq!(
        profile.validate_transcript_prefix(&log, 0, descriptor.root),
        Err(FiniteWarpV3TwoCosetTranscriptError::Transcript)
    );
}

#[test]
fn terminal_prefix_rejects_accumulator_root_substitution() {
    let (code, _whir, descriptor, profile) = fixture();
    let log = native_prefix_log(&code, &descriptor);
    let mut wrong_instance = instance(descriptor.root);
    wrong_instance.rt[2] += F::ONE;
    assert_eq!(
        generate_finite_warp_v3_two_coset_whir_prefix_trace(
            &profile,
            &descriptor,
            &wrong_instance,
            &log,
            0,
        )
        .unwrap_err(),
        FiniteWarpV3TwoCosetTranscriptError::Root
    );
}

#[test]
fn omitted_layout_observation_is_rejected_without_panicking() {
    let (code, _whir, descriptor, profile) = fixture();
    let log = native_prefix_log(&code, &descriptor);
    let omitted = profile.whir_layout_block_offset();
    let mut values = log.values().to_vec();
    let mut samples = log.samples().to_vec();
    let _ = values.remove(omitted);
    let _ = samples.remove(omitted);
    let truncated = TranscriptLog::new(values, samples);
    let result = catch_unwind(AssertUnwindSafe(|| {
        profile.validate_transcript_prefix(&truncated, 0, descriptor.root)
    }));
    assert_eq!(
        result.unwrap(),
        Err(FiniteWarpV3TwoCosetTranscriptError::Transcript)
    );
}

#[test]
fn profile_rejects_legacy_and_mutated_production_descriptors() {
    let (code, whir, descriptor, _) = fixture();
    let config = SC::default_from_params(SystemParams::new_for_testing(10));
    let ordinary = WhirInitialRsWarpCode::try_new(config.hasher().clone(), 8, 1, 0, 16).unwrap();
    assert_eq!(
        FiniteWarpV3TwoCosetTerminalProfile::new(&descriptor, &ordinary, &whir),
        Err(FiniteWarpV3TwoCosetTranscriptError::Descriptor)
    );

    for mutate in [
        |value: &mut TerminalDescriptor<Digest>| value.coordinate_ordering ^= 1,
        |value: &mut TerminalDescriptor<Digest>| value.rows_per_query = 8,
        |value: &mut TerminalDescriptor<Digest>| value.initial_folding_factor = 4,
        |value: &mut TerminalDescriptor<Digest>| value.field_modulus ^= 1,
    ] {
        let mut changed = descriptor.clone();
        mutate(&mut changed);
        let result = catch_unwind(AssertUnwindSafe(|| {
            FiniteWarpV3TwoCosetTerminalProfile::new(&changed, &code, &whir)
        }));
        assert!(result.unwrap().is_err());
    }
}

fn resumed_transcript_fixture() -> (
    TranscriptLog<F, [F; 16]>,
    FiniteWarpV3TerminalTranscriptCheckpoint,
) {
    let mut transcript = default_duplex_sponge_recorder();
    for value in 1..=3 {
        <_ as FiatShamirTranscript<SC>>::observe(&mut transcript, F::from_u32(value));
    }
    let _ = <_ as FiatShamirTranscript<SC>>::sample_ext(&mut transcript);
    let sponge = transcript.inner.checkpoint();
    assert_eq!(sponge.absorb_idx, 0);
    let checkpoint = FiniteWarpV3TerminalTranscriptCheckpoint {
        tidx: transcript.log.len().try_into().unwrap(),
        sample_count: D_EF.try_into().unwrap(),
        state: sponge.state,
    };

    // A resumed transcript must begin with an observation. This makes the
    // post-squeeze cursor irrelevant while retaining the complete sponge
    // state and absolute operation index.
    for value in 101..=105 {
        <_ as FiatShamirTranscript<SC>>::observe(&mut transcript, F::from_u32(value));
    }
    let _ = <_ as FiatShamirTranscript<SC>>::sample_ext(&mut transcript);
    (TranscriptHistory::into_log(transcript), checkpoint)
}

#[test]
fn resumed_terminal_transcript_starts_at_the_authenticated_checkpoint() {
    let (log, checkpoint) = resumed_transcript_fixture();
    let (trace, permutations) =
        generate_finite_warp_v3_resumed_terminal_transcript_trace(&log, checkpoint, None).unwrap();
    assert_eq!(permutations.len(), 1);

    let base_width = TranscriptCols::<F>::width();
    let row = trace.row_slice(0).unwrap();
    let cols: &TranscriptCols<F> = row[..base_width].borrow();
    let resume: &TranscriptResumeCols<F> = row[base_width..].borrow();
    assert_eq!(cols.tidx, F::from_u32(checkpoint.tidx));
    assert_eq!(resume.state, checkpoint.state);

    let air = TranscriptAir {
        transcript_bus: TranscriptBus::new(920),
        poseidon2_permute_bus: Poseidon2PermuteBus::new(921),
        final_state_bus: None,
        resume_state_bus: Some(ResumeTranscriptStateBus::new(922)),
        end_index_bus: None,
        checkpoint_state_bus: None,
    };
    check_constraints::<_, SC>(
        &air,
        "resumed coefficient-two-coset terminal transcript",
        &None,
        &[trace.as_view()],
        &[],
    );

    let mut wrong_resume = trace.clone();
    let row = &mut wrong_resume.values[..air.row_width::<F>()];
    let resume: &mut TranscriptResumeCols<F> = row[base_width..].borrow_mut();
    resume.state[15] += F::ONE;
    assert!(catch_unwind(AssertUnwindSafe(|| {
        check_constraints::<_, SC>(
            &air,
            "mutated resumed terminal transcript",
            &None,
            &[wrong_resume.as_view()],
            &[],
        );
    }))
    .is_err());
}

#[test]
fn resumed_terminal_transcript_rejects_state_index_and_sample_mutations() {
    let (log, checkpoint) = resumed_transcript_fixture();

    let mut wrong_state = checkpoint;
    wrong_state.state[9] += F::ONE;
    assert_eq!(
        generate_finite_warp_v3_resumed_terminal_transcript_trace(&log, wrong_state, None)
            .unwrap_err(),
        FiniteWarpV3TwoCosetTranscriptError::TranscriptResume
    );

    let mut wrong_index = checkpoint;
    wrong_index.tidx += 1;
    assert_eq!(
        generate_finite_warp_v3_resumed_terminal_transcript_trace(&log, wrong_index, None)
            .unwrap_err(),
        FiniteWarpV3TwoCosetTranscriptError::TranscriptResume
    );

    let mut wrong_sample = log.clone();
    let sample = wrong_sample
        .samples()
        .iter()
        .enumerate()
        .skip(checkpoint.tidx as usize)
        .find_map(|(index, &is_sample)| is_sample.then_some(index))
        .unwrap();
    wrong_sample.values_mut()[sample] += F::ONE;
    assert_eq!(
        generate_finite_warp_v3_resumed_terminal_transcript_trace(&wrong_sample, checkpoint, None,)
            .unwrap_err(),
        FiniteWarpV3TwoCosetTranscriptError::TranscriptResume
    );
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct TerminalTranscriptCounterpartyCols<T> {
    active: T,
    tidx: T,
    sample_count: T,
    state: [T; 16],
    output_accumulator_root: [T; DIGEST_SIZE],
    output_accumulator_digest: [T; DIGEST_SIZE],
}

#[derive(Clone, Debug)]
struct TerminalTranscriptCounterpartyAir {
    final_checkpoint_bus: FiniteWarpV3FinalVaccCheckpointBus,
    resume_bus: ResumeTranscriptStateBus,
    accumulator_link_bus: FiniteWarpV3TerminalAccumulatorLinkBus,
    protocol_digest: Digest,
    relation_digest: Digest,
    warp_index_digest: Digest,
    setup_digest: Digest,
    schedule_digest: Digest,
    call_count: usize,
}

impl BaseAir<F> for TerminalTranscriptCounterpartyAir {
    fn width(&self) -> usize {
        TerminalTranscriptCounterpartyCols::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for TerminalTranscriptCounterpartyAir {}
impl PartitionedBaseAir<F> for TerminalTranscriptCounterpartyAir {}

impl<AB> Air<AB> for TerminalTranscriptCounterpartyAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).unwrap();
        let local: &TerminalTranscriptCounterpartyCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_transition().assert_zero(local.active);
        self.final_checkpoint_bus.add_key_with_lookups(
            builder,
            FiniteWarpV3FinalVaccCheckpointMessage {
                protocol_digest: self.protocol_digest.map(Into::into),
                relation_digest: self.relation_digest.map(Into::into),
                warp_index_digest: self.warp_index_digest.map(Into::into),
                setup_digest: self.setup_digest.map(Into::into),
                schedule_digest: self.schedule_digest.map(Into::into),
                call_count: AB::Expr::from_usize(self.call_count),
                end_tidx: local.tidx.into(),
                end_sample_count: local.sample_count.into(),
                end_state: local.state.map(Into::into),
                output_accumulator_root: local.output_accumulator_root.map(Into::into),
                output_accumulator_digest: local.output_accumulator_digest.map(Into::into),
            },
            local.active,
        );
        self.resume_bus.receive(
            builder,
            AB::Expr::ZERO,
            ResumeTranscriptStateMessage {
                tidx: local.tidx.into(),
                state: local.state.map(Into::into),
            },
            local.active,
        );
        self.accumulator_link_bus.lookup_key(
            builder,
            FiniteWarpV3TerminalAccumulatorLinkMessage {
                root: local.output_accumulator_root.map(Into::into),
                digest: local.output_accumulator_digest.map(Into::into),
            },
            local.active,
        );
    }
}

fn counterparty_trace(
    checkpoint: FiniteWarpV3TerminalTranscriptCheckpoint,
    output_accumulator_root: Digest,
    output_accumulator_digest: Digest,
) -> RowMajorMatrix<F> {
    let width = TerminalTranscriptCounterpartyCols::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut TerminalTranscriptCounterpartyCols<F> = values.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.tidx = F::from_u32(checkpoint.tidx);
    cols.sample_count = F::from_u32(checkpoint.sample_count);
    cols.state = checkpoint.state;
    cols.output_accumulator_root = output_accumulator_root;
    cols.output_accumulator_digest = output_accumulator_digest;
    RowMajorMatrix::new(values, width)
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
fn transcript_seam_rejects_checkpoint_binding_root_or_digest_substitution_via_buses() {
    let (_, checkpoint) = resumed_transcript_fixture();
    let final_bus = FiniteWarpV3FinalVaccCheckpointBus::new(930);
    let resume_bus = ResumeTranscriptStateBus::new(931);
    let accumulator_link_bus = FiniteWarpV3TerminalAccumulatorLinkBus::new(932);
    let protocol_digest = digest(100);
    let relation_digest = digest(200);
    let warp_index_digest = digest(300);
    let setup_digest = digest(400);
    let schedule_digest = digest(500);
    let output_accumulator_root = digest(600);
    let output_accumulator_digest = digest(700);
    let call_count = 3;
    let seam = FiniteWarpV3TerminalTranscriptSeamAir {
        final_vacc_checkpoint_bus: final_bus,
        terminal_resume_bus: resume_bus,
        terminal_accumulator_link_bus: accumulator_link_bus,
        protocol_digest,
        relation_digest,
        warp_index_digest,
        setup_digest,
        schedule_digest,
        call_count,
    };
    let counterparty = TerminalTranscriptCounterpartyAir {
        final_checkpoint_bus: final_bus,
        resume_bus,
        accumulator_link_bus,
        protocol_digest,
        relation_digest,
        warp_index_digest,
        setup_digest,
        schedule_digest,
        call_count,
    };
    let mismatch_counterparty = counterparty.clone();
    let airs: Vec<AirRef<SC>> = vec![Arc::new(counterparty), Arc::new(seam)];
    let interactions = airs
        .iter()
        .map(|air| symbolic_interactions(air.as_ref()))
        .collect::<Vec<_>>();
    let authority = counterparty_trace(
        checkpoint,
        output_accumulator_root,
        output_accumulator_digest,
    );
    let seam = finite_warp_v3_terminal_transcript_seam_trace(
        checkpoint,
        output_accumulator_root,
        output_accumulator_digest,
    );
    let check = |seam: &RowMajorMatrix<F>| {
        check_logup(
            &airs.iter().map(|air| air.name()).collect::<Vec<_>>(),
            &interactions,
            &[None, None],
            &[vec![authority.as_view()], vec![seam.as_view()]],
            &[Vec::new(), Vec::new()],
        );
    };
    check(&seam);

    for mutate in [
        |cols: &mut FiniteWarpV3TerminalTranscriptSeamCols<F>| cols.tidx += F::ONE,
        |cols: &mut FiniteWarpV3TerminalTranscriptSeamCols<F>| cols.state[3] += F::ONE,
        |cols: &mut FiniteWarpV3TerminalTranscriptSeamCols<F>| cols.sample_count += F::ONE,
        |cols: &mut FiniteWarpV3TerminalTranscriptSeamCols<F>| {
            cols.output_accumulator_root[2] += F::ONE
        },
        |cols: &mut FiniteWarpV3TerminalTranscriptSeamCols<F>| {
            cols.output_accumulator_digest[5] += F::ONE
        },
    ] {
        let mut changed = seam.clone();
        let cols: &mut FiniteWarpV3TerminalTranscriptSeamCols<F> =
            changed.values.as_mut_slice().borrow_mut();
        mutate(cols);
        assert!(catch_unwind(AssertUnwindSafe(|| check(&changed))).is_err());
    }

    let mismatched_seam = FiniteWarpV3TerminalTranscriptSeamAir {
        final_vacc_checkpoint_bus: final_bus,
        terminal_resume_bus: resume_bus,
        terminal_accumulator_link_bus: accumulator_link_bus,
        protocol_digest: digest(101),
        relation_digest,
        warp_index_digest,
        setup_digest,
        schedule_digest,
        call_count,
    };
    let mismatch_airs: Vec<AirRef<SC>> =
        vec![Arc::new(mismatch_counterparty), Arc::new(mismatched_seam)];
    let mismatch_interactions = mismatch_airs
        .iter()
        .map(|air| symbolic_interactions(air.as_ref()))
        .collect::<Vec<_>>();
    assert!(catch_unwind(AssertUnwindSafe(|| {
        check_logup(
            &mismatch_airs
                .iter()
                .map(|air| air.name())
                .collect::<Vec<_>>(),
            &mismatch_interactions,
            &[None, None],
            &[vec![authority.as_view()], vec![seam.as_view()]],
            &[Vec::new(), Vec::new()],
        );
    }))
    .is_err());
}

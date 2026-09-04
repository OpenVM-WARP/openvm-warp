use core::borrow::BorrowMut;
use std::panic::AssertUnwindSafe;

use openvm_recursion_circuit::bus::{ResumeTranscriptStateBus, ResumeTranscriptStateMessage};
use openvm_stark_backend::{
    air_builders::{
        debug::{check_constraints, check_logup},
        symbolic::{get_symbolic_builder, SymbolicRapBuilder},
    },
    interaction::SymbolicInteraction,
    keygen::types::TraceWidth,
    p3_field::PrimeCharacteristicRing,
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as NativeSC, Digest, DIGEST_SIZE, F,
};

use super::*;
use crate::circuit::finite_warp_v3::{
    finite_warp_v3_terminal_transcript_seam_trace, FiniteWarpV3TerminalAccumulatorLinkBus,
    FiniteWarpV3TerminalAccumulatorLinkMessage, FiniteWarpV3TerminalTranscriptCheckpoint,
    FiniteWarpV3TerminalTranscriptSeamAir, FiniteWarpV3TerminalTranscriptSeamCols,
};

fn digest(seed: u32) -> Digest {
    core::array::from_fn(|limb| F::from_u32(seed + limb as u32))
}

fn checkpoint(seed: u32, tidx: u32, sample_count: u32) -> FiniteWarpV3TranscriptCheckpoint {
    FiniteWarpV3TranscriptCheckpoint {
        tidx,
        sample_count,
        state: core::array::from_fn(|limb| F::from_u32(seed + limb as u32)),
    }
}

fn profile() -> FiniteWarpV3WarpVerifyProfile {
    FiniteWarpV3WarpVerifyProfile {
        transcript_version: EXACT_FINITE_WARP_TRANSCRIPT_VERSION,
        protocol_digest: digest(10),
        relation_digest: digest(20),
        warp_index_digest: digest(30),
        setup_digest: digest(40),
        verifier_component_digest: digest(50),
        schedule_digest: digest(60),
        first_call_start: FiniteWarpV3TranscriptCheckpoint {
            tidx: 700,
            sample_count: 0,
            state: [F::ZERO; POSEIDON2_WIDTH],
        },
        calls: [
            FiniteWarpV3WarpVerifyCallProfile {
                active: true,
                source_start: 0,
                source_count: 64,
                input_arity: 64,
            },
            FiniteWarpV3WarpVerifyCallProfile {
                active: true,
                source_start: 64,
                source_count: 7,
                input_arity: 8,
            },
            FiniteWarpV3WarpVerifyCallProfile::inactive(),
        ],
    }
}

fn record() -> FiniteWarpV3WarpVerifyRecord {
    let first_end = checkpoint(200, 1_000, 12);
    FiniteWarpV3WarpVerifyRecord {
        manifest_digest: digest(61),
        calls: [
            FiniteWarpV3WarpVerifyCallRecord {
                fresh_stacked_root: digest(70),
                prior_accumulator_root: [F::ZERO; DIGEST_SIZE],
                output_accumulator_root: digest(80),
                prior_accumulator_digest: [F::ZERO; DIGEST_SIZE],
                output_accumulator_digest: digest(90),
                start: profile().first_call_start,
                end: first_end,
            },
            FiniteWarpV3WarpVerifyCallRecord {
                fresh_stacked_root: digest(110),
                prior_accumulator_root: digest(80),
                output_accumulator_root: digest(120),
                prior_accumulator_digest: digest(90),
                output_accumulator_digest: digest(130),
                start: first_end,
                end: checkpoint(300, 1_300, 24),
            },
            FiniteWarpV3WarpVerifyCallRecord::default(),
        ],
    }
}

fn buses() -> FiniteWarpV3WarpVerifyBuses {
    FiniteWarpV3WarpVerifyBuses {
        exact: FiniteWarpV3ExactVaccAuthorityBus::new(1_000),
        receipt: FiniteWarpV3CallReceiptBus::new(1_001),
        final_vacc_checkpoint: FiniteWarpV3FinalVaccCheckpointBus::new(1_002),
    }
}

fn authority_message() -> FiniteWarpV3ExactVaccAuthorityMessage<F> {
    let profile = profile();
    let call = record().calls[0];
    FiniteWarpV3ExactVaccAuthorityMessage {
        proof_idx: F::ZERO,
        call_index: F::ZERO,
        source_start: F::ZERO,
        source_count: F::from_u32(64),
        input_arity: F::from_u32(64),
        has_prior: F::ZERO,
        transcript_version: F::from_u64(EXACT_FINITE_WARP_TRANSCRIPT_VERSION),
        protocol_digest: profile.protocol_digest,
        relation_digest: profile.relation_digest,
        warp_index_digest: profile.warp_index_digest,
        setup_digest: profile.setup_digest,
        schedule_digest: profile.schedule_digest,
        start_tidx: F::from_u32(call.start.tidx),
        start_sample_count: F::from_u32(call.start.sample_count),
        start_state: call.start.state,
        end_tidx: F::from_u32(call.end.tidx),
        end_sample_count: F::from_u32(call.end.sample_count),
        end_state: call.end.state,
        fresh_stacked_root: call.fresh_stacked_root,
        prior_accumulator_root: call.prior_accumulator_root,
        output_accumulator_root: call.output_accumulator_root,
        prior_accumulator_digest: call.prior_accumulator_digest,
        output_accumulator_digest: call.output_accumulator_digest,
    }
}

/// Test-only counterpart for every interaction emitted/consumed by the
/// receipt adapter. The exact setup digest is deliberately independent from
/// the adapter profile so the test detects accidental reuse of the wrapper
/// component digest at the authority seam.
struct WarpVerifyInteractionOracleAir {
    profile: FiniteWarpV3WarpVerifyProfile,
    exact_setup_digest: Digest,
    buses: FiniteWarpV3WarpVerifyBuses,
    consume_final_checkpoint: bool,
}

impl BaseAir<F> for WarpVerifyInteractionOracleAir {
    fn width(&self) -> usize {
        FiniteWarpV3WarpVerifyCols::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for WarpVerifyInteractionOracleAir {}
impl PartitionedBaseAir<F> for WarpVerifyInteractionOracleAir {}

impl<AB> Air<AB> for WarpVerifyInteractionOracleAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("WARP Verify oracle row");
        let local: &FiniteWarpV3WarpVerifyCols<AB::Var> = (*row).borrow();
        let enabled = local.active;
        let proof_idx = Into::<AB::Expr>::into(local.call_index);

        self.buses.exact.add_key_with_lookups(
            builder,
            FiniteWarpV3ExactVaccAuthorityMessage {
                proof_idx: proof_idx.clone(),
                call_index: local.call_index.into(),
                source_start: local.source_start.into(),
                source_count: local.source_count.into(),
                input_arity: local.input_arity.into(),
                has_prior: local.has_prior.into(),
                transcript_version: AB::Expr::from_u64(self.profile.transcript_version),
                protocol_digest: self.profile.protocol_digest.map(AB::Expr::from),
                relation_digest: self.profile.relation_digest.map(AB::Expr::from),
                warp_index_digest: self.profile.warp_index_digest.map(AB::Expr::from),
                setup_digest: self.exact_setup_digest.map(AB::Expr::from),
                schedule_digest: self.profile.schedule_digest.map(AB::Expr::from),
                start_tidx: local.start_tidx.into(),
                start_sample_count: local.start_sample_count.into(),
                start_state: local.start_state.map(Into::into),
                end_tidx: local.end_tidx.into(),
                end_sample_count: local.end_sample_count.into(),
                end_state: local.end_state.map(Into::into),
                fresh_stacked_root: local.fresh_stacked_root.map(Into::into),
                prior_accumulator_root: local.prior_accumulator_root.map(Into::into),
                output_accumulator_root: local.output_accumulator_root.map(Into::into),
                prior_accumulator_digest: local.prior_accumulator_digest.map(Into::into),
                output_accumulator_digest: local.output_accumulator_digest.map(Into::into),
            },
            enabled,
        );
        self.buses.receipt.lookup_key(
            builder,
            FiniteWarpV3CallReceiptMessage {
                protocol_digest: self.profile.protocol_digest.map(AB::Expr::from),
                relation_digest: self.profile.relation_digest.map(AB::Expr::from),
                warp_index_digest: self.profile.warp_index_digest.map(AB::Expr::from),
                verifier_component_digest: self
                    .profile
                    .verifier_component_digest
                    .map(AB::Expr::from),
                schedule_digest: self.profile.schedule_digest.map(AB::Expr::from),
                manifest_digest: local.manifest_digest.map(Into::into),
                call_index: local.call_index.into(),
                source_start: local.source_start.into(),
                source_count: local.source_count.into(),
                input_arity: local.input_arity.into(),
                fresh_stacked_root: local.fresh_stacked_root.map(Into::into),
                prior_accumulator_digest: local.prior_accumulator_digest.map(Into::into),
                output_accumulator_digest: local.output_accumulator_digest.map(Into::into),
            },
            enabled,
        );
        self.buses.final_vacc_checkpoint.lookup_key(
            builder,
            FiniteWarpV3FinalVaccCheckpointMessage {
                protocol_digest: self.profile.protocol_digest.map(AB::Expr::from),
                relation_digest: self.profile.relation_digest.map(AB::Expr::from),
                warp_index_digest: self.profile.warp_index_digest.map(AB::Expr::from),
                setup_digest: self.profile.setup_digest.map(AB::Expr::from),
                schedule_digest: self.profile.schedule_digest.map(AB::Expr::from),
                call_count: AB::Expr::from_usize(self.profile.active_call_count()),
                end_tidx: local.end_tidx.into(),
                end_sample_count: local.end_sample_count.into(),
                end_state: local.end_state.map(Into::into),
                output_accumulator_root: local.output_accumulator_root.map(Into::into),
                output_accumulator_digest: local.output_accumulator_digest.map(Into::into),
            },
            Into::<AB::Expr>::into(local.slot_flags[self.profile.active_call_count() - 1])
                * AB::Expr::from_bool(self.consume_final_checkpoint),
        );
    }
}

#[derive(Clone, Debug)]
struct TerminalSeamOutputOracleAir {
    resume: ResumeTranscriptStateBus,
    accumulator_link: FiniteWarpV3TerminalAccumulatorLinkBus,
}

impl BaseAir<F> for TerminalSeamOutputOracleAir {
    fn width(&self) -> usize {
        FiniteWarpV3TerminalTranscriptSeamCols::<F>::width()
    }
}
impl BaseAirWithPublicValues<F> for TerminalSeamOutputOracleAir {}
impl PartitionedBaseAir<F> for TerminalSeamOutputOracleAir {}

impl<AB> Air<AB> for TerminalSeamOutputOracleAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("terminal seam output oracle row");
        let local: &FiniteWarpV3TerminalTranscriptSeamCols<AB::Var> = (*row).borrow();
        self.resume.receive(
            builder,
            AB::Expr::ZERO,
            ResumeTranscriptStateMessage {
                tidx: local.tidx.into(),
                state: local.state.map(Into::into),
            },
            local.active,
        );
        self.accumulator_link.lookup_key(
            builder,
            FiniteWarpV3TerminalAccumulatorLinkMessage {
                root: local.output_accumulator_root.map(Into::into),
                digest: local.output_accumulator_digest.map(Into::into),
            },
            local.active,
        );
    }
}

fn symbolic_interactions<R>(air: &R) -> Vec<SymbolicInteraction<F>>
where
    R: Air<SymbolicRapBuilder<F>> + BaseAir<F> + BaseAirWithPublicValues<F> + PartitionedBaseAir<F>,
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

fn check_adapter_interactions(
    adapter: &FiniteWarpV3WarpVerifyReceiptAir,
    oracle: &WarpVerifyInteractionOracleAir,
    trace: &openvm_stark_backend::p3_matrix::dense::RowMajorMatrix<F>,
) {
    check_logup(
        &[
            "receipt-adapter".to_string(),
            "authority-and-sinks".to_string(),
        ],
        &[
            symbolic_interactions(adapter),
            symbolic_interactions(oracle),
        ],
        &[None, None],
        &[vec![trace.as_view()], vec![trace.as_view()]],
        &[Vec::new(), Vec::new()],
    );
}

#[test]
fn exact_authority_key_binds_shape_index_transcript_and_linkage() {
    let message = authority_message();
    let expected = message.to_vec();

    let mut mutated = message.clone();
    mutated.input_arity = F::from_u32(32);
    assert_ne!(expected, mutated.to_vec());

    let mut mutated = message.clone();
    mutated.warp_index_digest[0] += F::ONE;
    assert_ne!(expected, mutated.to_vec());

    let mut mutated = message.clone();
    mutated.setup_digest[0] += F::ONE;
    assert_ne!(expected, mutated.to_vec());

    let mut mutated = message.clone();
    mutated.transcript_version += F::ONE;
    assert_ne!(expected, mutated.to_vec());

    let mut mutated = message.clone();
    mutated.start_state[0] += F::ONE;
    assert_ne!(expected, mutated.to_vec());

    let mut mutated = message;
    mutated.output_accumulator_digest[0] += F::ONE;
    assert_ne!(expected, mutated.to_vec());
}

#[test]
fn production_64_then_8_schedule_is_accepted() {
    let profile = profile();
    assert_eq!(profile.active_call_count(), 2);
    profile.validate().unwrap();
    FiniteWarpV3WarpVerifyReceiptAir::new(profile, buses())
        .unwrap()
        .generate_trace(&record())
        .unwrap();
}

#[test]
fn authority_receipt_and_terminal_checkpoint_interactions_balance() {
    let profile = profile();
    let adapter = FiniteWarpV3WarpVerifyReceiptAir::new(profile.clone(), buses()).unwrap();
    let trace = adapter.generate_trace(&record()).unwrap();
    let oracle = WarpVerifyInteractionOracleAir {
        exact_setup_digest: profile.setup_digest,
        profile,
        buses: buses(),
        consume_final_checkpoint: true,
    };
    check_adapter_interactions(&adapter, &oracle, &trace);
}

#[test]
fn exact_setup_digest_cannot_be_replaced_by_component_digest() {
    let profile = profile();
    assert_ne!(profile.setup_digest, profile.verifier_component_digest);
    let adapter = FiniteWarpV3WarpVerifyReceiptAir::new(profile.clone(), buses()).unwrap();
    let trace = adapter.generate_trace(&record()).unwrap();
    let oracle = WarpVerifyInteractionOracleAir {
        exact_setup_digest: profile.verifier_component_digest,
        profile,
        buses: buses(),
        consume_final_checkpoint: true,
    };
    let rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        check_adapter_interactions(&adapter, &oracle, &trace);
    }));
    assert!(
        rejected.is_err(),
        "component digest was accepted as the exact WARP setup digest"
    );
}

#[test]
fn final_warp_receipt_balances_the_real_terminal_seam_without_a_host_bridge() {
    let profile = profile();
    let buses = buses();
    let adapter = FiniteWarpV3WarpVerifyReceiptAir::new(profile.clone(), buses).unwrap();
    let adapter_trace = adapter.generate_trace(&record()).unwrap();
    let authority_and_receipt_oracle = WarpVerifyInteractionOracleAir {
        profile: profile.clone(),
        exact_setup_digest: profile.setup_digest,
        buses,
        consume_final_checkpoint: false,
    };

    let resume = ResumeTranscriptStateBus::new(1_003);
    let accumulator_link = FiniteWarpV3TerminalAccumulatorLinkBus::new(1_004);
    let seam = FiniteWarpV3TerminalTranscriptSeamAir {
        final_vacc_checkpoint_bus: buses.final_vacc_checkpoint,
        terminal_resume_bus: resume,
        terminal_accumulator_link_bus: accumulator_link,
        protocol_digest: profile.protocol_digest,
        relation_digest: profile.relation_digest,
        warp_index_digest: profile.warp_index_digest,
        setup_digest: profile.setup_digest,
        schedule_digest: profile.schedule_digest,
        call_count: profile.active_call_count(),
    };
    let last = record().calls[profile.active_call_count() - 1];
    let seam_trace = finite_warp_v3_terminal_transcript_seam_trace(
        FiniteWarpV3TerminalTranscriptCheckpoint {
            tidx: last.end.tidx,
            sample_count: last.end.sample_count,
            state: last.end.state,
        },
        last.output_accumulator_root,
        last.output_accumulator_digest,
    );
    let terminal_outputs = TerminalSeamOutputOracleAir {
        resume,
        accumulator_link,
    };

    check_logup(
        &[
            "warp-receipt".to_string(),
            "exact-authority-and-wrapper-receipt".to_string(),
            "terminal-seam".to_string(),
            "terminal-resume-and-root-sinks".to_string(),
        ],
        &[
            symbolic_interactions(&adapter),
            symbolic_interactions(&authority_and_receipt_oracle),
            symbolic_interactions(&seam),
            symbolic_interactions(&terminal_outputs),
        ],
        &[None, None, None, None],
        &[
            vec![adapter_trace.as_view()],
            vec![adapter_trace.as_view()],
            vec![seam_trace.as_view()],
            vec![seam_trace.as_view()],
        ],
        &[Vec::new(), Vec::new(), Vec::new(), Vec::new()],
    );
}

#[test]
fn unsupported_or_noncanonical_shapes_fail_closed() {
    let mut invalid = profile();
    invalid.transcript_version += 1;
    assert!(matches!(
        invalid.validate(),
        Err(FiniteWarpV3WarpVerifyError::TranscriptVersion)
    ));

    let mut invalid = profile();
    invalid.calls[0].input_arity = 32;
    assert!(matches!(
        invalid.validate(),
        Err(FiniteWarpV3WarpVerifyError::InvalidCallShape(0))
    ));

    let mut invalid = profile();
    invalid.calls[0] = FiniteWarpV3WarpVerifyCallProfile::inactive();
    assert!(matches!(
        invalid.validate(),
        Err(FiniteWarpV3WarpVerifyError::NonPrefixCalls)
    ));

    let mut invalid = profile();
    invalid.calls[1].input_arity = 16;
    assert!(matches!(
        invalid.validate(),
        Err(FiniteWarpV3WarpVerifyError::InvalidCallShape(1))
    ));
}

#[test]
fn record_rejects_accumulator_and_transcript_splicing() {
    let air = FiniteWarpV3WarpVerifyReceiptAir::new(profile(), buses()).unwrap();
    let mut bad = record();
    bad.calls[1].prior_accumulator_digest = digest(999);
    assert!(matches!(
        air.generate_trace(&bad),
        Err(FiniteWarpV3WarpVerifyError::AccumulatorChain(1))
    ));

    let mut bad = record();
    bad.calls[1].start.state[0] += F::ONE;
    assert!(matches!(
        air.generate_trace(&bad),
        Err(FiniteWarpV3WarpVerifyError::TranscriptChain(1))
    ));
}

#[test]
fn mutated_trace_chain_is_rejected_by_constraints() {
    let air = FiniteWarpV3WarpVerifyReceiptAir::new(profile(), buses()).unwrap();
    let mut trace = air.generate_trace(&record()).unwrap();
    let width = trace.width();
    let second: &mut FiniteWarpV3WarpVerifyCols<F> = trace.values[width..2 * width].borrow_mut();
    second.prior_accumulator_root[0] += F::ONE;

    let rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        check_constraints::<_, NativeSC>(
            &air,
            "FiniteWarpV3WarpVerifyReceiptAir",
            &None,
            &[trace.as_view()],
            &[],
        );
    }));
    assert!(
        rejected.is_err(),
        "mutated accumulator root chain was accepted"
    );
}

#[test]
fn mutated_manifest_or_transcript_chain_is_rejected_by_constraints() {
    let air = FiniteWarpV3WarpVerifyReceiptAir::new(profile(), buses()).unwrap();

    let mut trace = air.generate_trace(&record()).unwrap();
    let width = trace.width();
    let second: &mut FiniteWarpV3WarpVerifyCols<F> = trace.values[width..2 * width].borrow_mut();
    second.manifest_digest[0] += F::ONE;
    let rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        check_constraints::<_, NativeSC>(
            &air,
            "FiniteWarpV3WarpVerifyReceiptAir",
            &None,
            &[trace.as_view()],
            &[],
        );
    }));
    assert!(rejected.is_err(), "mutated manifest chain was accepted");

    let mut trace = air.generate_trace(&record()).unwrap();
    let second: &mut FiniteWarpV3WarpVerifyCols<F> = trace.values[width..2 * width].borrow_mut();
    second.start_state[0] += F::ONE;
    let rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        check_constraints::<_, NativeSC>(
            &air,
            "FiniteWarpV3WarpVerifyReceiptAir",
            &None,
            &[trace.as_view()],
            &[],
        );
    }));
    assert!(rejected.is_err(), "mutated transcript chain was accepted");
}

#[test]
fn inactive_slots_must_be_canonical_zero() {
    let air = FiniteWarpV3WarpVerifyReceiptAir::new(profile(), buses()).unwrap();
    let trace = air.generate_trace(&record()).unwrap();
    check_constraints::<_, NativeSC>(
        &air,
        "FiniteWarpV3WarpVerifyReceiptAir",
        &None,
        &[trace.as_view()],
        &[],
    );
}

#[test]
fn arity_two_cannot_be_relabelled_as_arity_64() {
    let mut arity_two_profile = profile();
    arity_two_profile.calls = [
        FiniteWarpV3WarpVerifyCallProfile {
            active: true,
            source_start: 0,
            source_count: 2,
            input_arity: 2,
        },
        FiniteWarpV3WarpVerifyCallProfile::inactive(),
        FiniteWarpV3WarpVerifyCallProfile::inactive(),
    ];
    let air = FiniteWarpV3WarpVerifyReceiptAir::new(arity_two_profile, buses()).unwrap();
    let mut arity_two_record = record();
    arity_two_record.calls[1] = FiniteWarpV3WarpVerifyCallRecord::default();
    arity_two_record.calls[2] = FiniteWarpV3WarpVerifyCallRecord::default();
    let mut trace = air.generate_trace(&arity_two_record).unwrap();
    let width = trace.width();
    let cols: &mut FiniteWarpV3WarpVerifyCols<F> = trace.values[..width].borrow_mut();
    cols.input_arity = F::from_u32(64);

    let rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
        check_constraints::<_, NativeSC>(
            &air,
            "FiniteWarpV3WarpVerifyReceiptAir",
            &None,
            &[trace.as_view()],
            &[],
        );
    }));
    assert!(
        rejected.is_err(),
        "arity-two trace was relabelled as arity 64"
    );

    // Even before recursive lookup balancing, the exact authority key itself
    // distinguishes the two schedules.
    let mut arity_two = authority_message();
    arity_two.source_count = F::from_u32(2);
    arity_two.input_arity = F::from_u32(2);
    let mut arity_64 = arity_two.clone();
    arity_64.source_count = F::from_u32(64);
    arity_64.input_arity = F::from_u32(64);
    assert_ne!(arity_two.to_vec(), arity_64.to_vec());
}

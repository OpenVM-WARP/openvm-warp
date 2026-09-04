use core::borrow::{Borrow, BorrowMut};
use std::{panic::AssertUnwindSafe, sync::Arc};

use openvm_recursion_circuit::{
    bus::{ColumnClaimsBus, Poseidon2CompressBus, TranscriptBus},
    system::{BusIndexManager, BusInventory},
    transcript::{Poseidon2BusOwner, TranscriptModule},
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::{
        debug::{check_constraints, check_logup},
        symbolic::get_symbolic_builder,
    },
    interaction::{BusIndex, InteractionBuilder, SymbolicInteraction},
    keygen::types::TraceWidth,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, PrimeField32, TwoAdicField},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    AirRef, AnyAir, BaseAirWithPublicValues, PartitionedBaseAir, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as NativeSC, Digest, D_EF, EF, F,
};
use openvm_verify_stark_host::pvs::VmPvs;

use super::*;
use crate::circuit::native_warp_history_v19::{
    CertifiedDirectAirVaccInputBusV19, CertifiedDirectAirVaccInputMessageV19,
    CertifiedFreshExplicitDigestBusV19, CertifiedFreshExplicitDigestMessageV19,
    CertifiedWarpReplayBusV19, CertifiedWarpReplayMessageV19, FixedSourcePublicValueBusV19,
    LogUpOnlyHistoryBusV19, LogUpOnlyHistoryMessageV19, MAX_FRESH_BETA_LEN_V19,
    MAX_RAW_MESSAGE_POINT_LEN_V19, NATIVE_WARP_HISTORY_PROTOCOL_V19,
};

fn digest(seed: u32) -> Digest {
    core::array::from_fn(|limb| F::from_u32(seed + limb as u32))
}

fn ef(seed: u32) -> EF {
    let coefficients: [F; D_EF] = core::array::from_fn(|limb| F::from_u32(seed + limb as u32));
    EF::from_basis_coefficients_slice(&coefficients).expect("EF4 value")
}

fn copy_ext(target: &mut [F; D_EF], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

fn fixed_opening(
    matrix: &FixedSetupMatrixV2,
    column: usize,
    l_skip: usize,
    point: &[EF],
    rotated: bool,
) -> EF {
    let skip = 1usize << l_skip;
    let folded_height = matrix.height.max(skip) / skip;
    let r0 = point[0];
    let omega = F::two_adic_generator(l_skip);
    let scaling = (r0.exp_u64(skip as u64) - EF::ONE) * EF::from_usize(skip).inverse();
    let mut leaves = Vec::with_capacity(folded_height);
    for folded_row in 0..folded_height {
        let mut value = EF::ZERO;
        let mut omega_power = F::ONE;
        for z in 0..skip {
            let barycentric =
                EF::from(omega_power) * (r0 - EF::from(omega_power)).inverse() * scaling;
            let physical =
                ((folded_row << l_skip) + z + usize::from(rotated)) & (matrix.height - 1);
            value += EF::from(matrix.values[physical * matrix.width + column]) * barycentric;
            omega_power *= omega;
        }
        leaves.push(value);
    }
    for &challenge in &point[1..] {
        for node in 0..leaves.len() / 2 {
            leaves[node] = leaves[2 * node] + challenge * (leaves[2 * node + 1] - leaves[2 * node]);
        }
        leaves.truncate(leaves.len() / 2);
    }
    leaves[0]
}

fn fixed_setup_fixture(
    relation_digest: Digest,
    source_relation_vk_digest: Digest,
    point_bus: FixedSetupOpeningPointBusV2,
    node_bus: FixedSetupOpeningNodeBusV2,
    pair_bus: VerifiedFixedSetupOpeningPairBusV2,
    column_claims_bus: ColumnClaimsBus,
    certificate_bus: FixedSetupOpeningCertificateBusV2,
    compress_bus: Poseidon2CompressBus,
) -> (
    FixedSetupOpeningAirV2,
    FixedSetupOpeningTraceV2,
    FixedSetupOpeningCertificateMessageV2<F>,
) {
    let matrix = FixedSetupMatrixV2 {
        setup_index: 0,
        air_id: 7,
        relation_digest,
        width: 2,
        height: 8,
        values: (0..16)
            .map(|index| F::from_u32(17 + 13 * index))
            .collect::<Vec<_>>()
            .into(),
    };
    let profile = FixedSetupOpeningProfileV2::new(
        source_relation_vk_digest,
        vec![matrix],
        vec![FixedSetupOpeningInstanceV2 {
            proof_index: 0,
            matrix_index: 0,
            sort_idx: 3,
            part_idx: 1,
            l_skip: 1,
            log_height: 3,
            need_rot: true,
        }],
    )
    .expect("fixed setup profile");
    let point = vec![EF::from_u32(19), EF::from_u32(23), EF::from_u32(29)];
    let claims = profile
        .claim_schedule()
        .map(|claim| {
            let matrix = &profile.matrices[0];
            FixedSetupOpeningClaimRecordV2 {
                setup_index: claim.setup_index,
                air_id: claim.air_id,
                sort_idx: claim.sort_idx,
                part_idx: claim.part_idx,
                col_idx: claim.col_idx,
                current: fixed_opening(
                    matrix,
                    claim.col_idx as usize,
                    claim.l_skip,
                    &point[..claim.point_len],
                    false,
                ),
                rotated: claim.need_rot.then(|| {
                    fixed_opening(
                        matrix,
                        claim.col_idx as usize,
                        claim.l_skip,
                        &point[..claim.point_len],
                        true,
                    )
                }),
            }
        })
        .collect();
    let records = vec![FixedSetupOpeningProofRecordV2 {
        proof_index: 0,
        opening_point: point,
        claims,
    }];
    let air = FixedSetupOpeningAirV2 {
        profile,
        point_bus,
        node_bus,
        verified_pair_bus: pair_bus,
        column_claims_bus,
        certificate_bus,
        compress_bus,
    };
    let trace = generate_fixed_setup_opening_trace_v2(&air, &records).expect("C1 trace");
    let (proof_index, canonical_claim_count, setup_openings_digest) = trace.certificates[0];
    let certificate = FixedSetupOpeningCertificateMessageV2 {
        protocol_version: F::from_u32(FIXED_SETUP_OPENING_PROTOCOL_V2),
        proof_index: F::from_u32(proof_index),
        canonical_claim_count: F::from_u32(canonical_claim_count),
        source_relation_vk_digest,
        setup_openings_digest,
    };
    (air, trace, certificate)
}

fn active_count_profile(
    relation_digest: Digest,
    profile_digest: Digest,
) -> VerifierWarpActiveCountProfileV2 {
    VerifierWarpActiveCountProfileV2 {
        relation_digest,
        profile_digest,
        profile_segment_count: 1,
        profile_batch_count: 1,
        batch_arity: 4,
        final_batch_active_count: 1,
        trace_heights: Arc::from([4]),
        vm_pvs_air_id: 1,
        is_valid_common_main_column: 0,
        is_valid_message_block_start: 0,
        log_height: 2,
        log_message_len: 3,
    }
}

fn count_weight(profile: &VerifierWarpActiveCountProfileV2, point: &[EF]) -> EF {
    let prefix_len = usize::from(profile.log_message_len - profile.log_height);
    let block_index = profile.is_valid_message_block_start >> profile.log_height;
    (0..prefix_len).fold(EF::ONE, |product, coordinate| {
        let bit = ((block_index >> (prefix_len - 1 - coordinate)) & 1) != 0;
        product
            * if bit {
                point[coordinate]
            } else {
                EF::ONE - point[coordinate]
            }
    })
}

fn active_count_record(
    profile: &VerifierWarpActiveCountProfileV2,
    source_root: Digest,
) -> VerifierWarpActiveCountFunctionalRecordV2 {
    let sampled = ef(50);
    let point = (0..profile.log_message_len)
        .map(|index| ef(80 + u32::from(index)))
        .collect::<Vec<_>>();
    let ordinary_target = ef(120);
    let ordinary_weight = ef(140);
    let coefficient = sampled;
    let expected_count = profile.expected_active_child_count(0).unwrap();
    let start = 1_000;
    let challenge = start + profile.metadata_observation_count() as u32;
    let end = start + profile.transcript_span() as u32;
    VerifierWarpActiveCountFunctionalRecordV2 {
        proof_index: 0,
        batch_index: 0,
        relation_digest: profile.relation_digest,
        profile_digest: profile.profile_digest,
        phase_source_root: source_root,
        ordinary_source_root: source_root,
        reduction_source_root: source_root,
        phase_start_tidx: start,
        challenge_tidx: challenge,
        phase_end_tidx: end,
        sampled_coefficient: sampled,
        ordinary_term_count: 9,
        ordinary_target,
        ordinary_weight_at_point: ordinary_weight,
        ordinary_point: point.clone(),
        reduction_point: point.clone(),
        reduction_end_tidx: end + 100,
        reduction_expected_active_child_count: expected_count,
        reduction_block_start: profile.is_valid_message_block_start,
        reduction_log_height: profile.log_height,
        reduction_count_term_scale: coefficient * EF::from(F::from_u64(1u64 << profile.log_height)),
        reduction_combined_target: ordinary_target
            + coefficient * EF::from(F::from_u8(expected_count)),
        reduction_combined_weight_at_point: ordinary_weight
            + coefficient * count_weight(profile, &point),
        message_value: ef(170),
    }
}

fn certified_count(
    profile: &VerifierWarpActiveCountProfileV2,
    record: &VerifierWarpActiveCountFunctionalRecordV2,
) -> VerifierWarpCertifiedActiveCountMessageV2<F> {
    let mut point = [[F::ZERO; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19];
    for (target, &value) in point.iter_mut().zip(&record.ordinary_point) {
        copy_ext(target, value);
    }
    let mut batching_coefficient = [F::ZERO; D_EF];
    copy_ext(&mut batching_coefficient, record.sampled_coefficient);
    let mut message_value = [F::ZERO; D_EF];
    copy_ext(&mut message_value, record.message_value);
    VerifierWarpCertifiedActiveCountMessageV2 {
        proof_index: F::ZERO,
        batch_index: [F::ZERO; 4],
        relation_digest: profile.relation_digest,
        profile_digest: profile.profile_digest,
        source_root: record.phase_source_root,
        expected_active_child_count: F::ONE,
        batching_coefficient,
        point_len: F::from_u8(profile.log_message_len),
        point,
        message_value,
        reduction_end_tidx: F::from_u32(record.reduction_end_tidx),
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct C3AuthorityStubCols<T> {
    active: T,
    logup: LogUpOnlyHistoryMessageV19<T>,
    fresh_input: CertifiedDirectAirVaccInputMessageV19<T>,
    replay: CertifiedWarpReplayMessageV19<T>,
}

#[derive(Clone)]
struct C3AuthorityStubAir {
    logup_bus: LogUpOnlyHistoryBusV19,
    vacc_input_bus: CertifiedDirectAirVaccInputBusV19,
    replay_bus: CertifiedWarpReplayBusV19,
}

impl BaseAir<F> for C3AuthorityStubAir {
    fn width(&self) -> usize {
        core::mem::size_of::<C3AuthorityStubCols<u8>>()
    }
}

impl BaseAirWithPublicValues<F> for C3AuthorityStubAir {}
impl PartitionedBaseAir<F> for C3AuthorityStubAir {}

impl<AB> Air<AB> for C3AuthorityStubAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("C3 authority stub row");
        let local: &C3AuthorityStubCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        self.logup_bus
            .add_key_with_lookups(builder, local.logup.clone(), local.active);
        self.vacc_input_bus
            .add_key_with_lookups(builder, local.fresh_input.clone(), local.active);
        self.replay_bus
            .add_key_with_lookups(builder, local.replay.clone(), local.active);
    }
}

fn c3_stub_trace(record: &VerifierWarpProducerBridgeRecordV2) -> RowMajorMatrix<F> {
    let width = core::mem::size_of::<C3AuthorityStubCols<u8>>();
    let mut values = F::zero_vec(2 * width);
    let row: &mut C3AuthorityStubCols<F> = values[..width].borrow_mut();
    row.active = F::ONE;
    row.logup = record.logup.clone();
    row.fresh_input = record.fresh_input.clone();
    row.replay = record.replay.clone();
    RowMajorMatrix::new(values, width)
}

fn producer_record(
    relation_digest: Digest,
    source_relation_vk_digest: Digest,
    source_message_root: Digest,
    fixed_setup_opening: FixedSetupOpeningCertificateMessageV2<F>,
    active_count: VerifierWarpCertifiedActiveCountMessageV2<F>,
) -> VerifierWarpProducerBridgeRecordV2 {
    let vm_pvs = VmPvs {
        program_commit: digest(400),
        initial_pc: F::from_u32(11),
        final_pc: F::from_u32(19),
        exit_code: F::ZERO,
        is_terminate: F::ONE,
        initial_root: digest(100),
        final_root: digest(120),
    };
    let source_forest_root = digest(500);
    let legacy_segment_openings_digest = digest(520);
    let source_accumulator_digest = digest(600);
    let prior_accumulator_digest = digest(620);
    let output_accumulator_digest = digest(640);
    let source_functional_digest = digest(660);
    let source_statement_digest = digest(680);
    let transition_transcript_digest = digest(700);
    let mut beta = [[F::ZERO; D_EF]; MAX_FRESH_BETA_LEN_V19];
    beta[1][0] = F::ONE;
    for (offset, &value) in vm_pvs.as_slice().iter().enumerate() {
        beta[2 + offset][0] = value;
    }
    let beta_len = 2 + VmPvs::<u8>::width();
    let mut source_point = [[F::ZERO; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19];
    source_point[0][0] = F::from_u32(901);
    VerifierWarpProducerBridgeRecordV2 {
        vm_pvs,
        logup: LogUpOnlyHistoryMessageV19 {
            protocol_version: F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            mode_tag: F::ONE,
            segment_index_lo: F::ZERO,
            segment_index_hi: F::ZERO,
            app_vk_digest: source_relation_vk_digest,
            source_forest_root,
            segment_openings_digest: legacy_segment_openings_digest,
            verifier_endpoint: [F::ZERO; D_EF],
            checkpoint_digest: digest(720),
        },
        fixed_source: CertifiedFixedMultiAirSourceMessageV2 {
            proof_index: F::ZERO,
            segment_index_lo: F::ZERO,
            segment_index_hi: F::ZERO,
            active_child_count: active_count.expected_active_child_count,
            app_vk_digest: source_relation_vk_digest,
            relation_digest,
            source_forest_root,
            segment_openings_digest: legacy_segment_openings_digest,
            source_root: source_message_root,
            point_len: F::ONE,
            point: source_point,
            value: [F::ZERO; D_EF],
            verifier_endpoint: [F::ZERO; D_EF],
        },
        fresh_input: CertifiedDirectAirVaccInputMessageV19 {
            proof_index: F::ZERO,
            protocol_version: F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            segment_index_lo: F::ZERO,
            segment_index_hi: F::ZERO,
            update_index_lo: F::ZERO,
            update_index_hi: F::ZERO,
            shard_ordinal: F::ZERO,
            relation_digest,
            root: source_message_root,
            alpha_len: F::ONE,
            alpha: source_point,
            mu: [F::ZERO; D_EF],
            beta_len: F::from_usize(beta_len),
            beta,
            eta: [F::ZERO; D_EF],
        },
        fresh_explicit: CertifiedFreshExplicitDigestMessageV19 {
            proof_index: F::ZERO,
            segment_index_lo: F::ZERO,
            segment_index_hi: F::ZERO,
            relation_digest,
            digest: digest(730),
        },
        replay: CertifiedWarpReplayMessageV19 {
            protocol_version: F::from_u32(NATIVE_WARP_HISTORY_PROTOCOL_V19),
            segment_index_lo: F::ZERO,
            segment_index_hi: F::ZERO,
            update_index_lo: F::ZERO,
            update_index_hi: F::ZERO,
            shard_ordinal: F::ZERO,
            source_forest_root,
            key_digest: digest(740),
            relation_digest,
            opening_claim_digest: source_functional_digest,
            fresh_instance_digest: source_accumulator_digest,
            segment_openings_digest: legacy_segment_openings_digest,
            prior_root: digest(760),
            fresh_root: source_message_root,
            next_root: digest(780),
            previous_accumulator_digest: prior_accumulator_digest,
            next_accumulator_digest: output_accumulator_digest,
            authenticated_batching_claim: [F::ZERO; D_EF],
            previous_checkpoint_digest: digest(800),
            next_checkpoint_digest: digest(820),
            replay_endpoint_digest: source_statement_digest,
            replay_binding_digest: transition_transcript_digest,
        },
        fixed_setup_opening,
        active_count,
    }
}

fn history_record(
    producer: &VerifierWarpProducerBridgeRecordV2,
) -> VerifierWarpHistoryTransitionRecordV2 {
    VerifierWarpHistoryTransitionRecordV2 {
        protocol_version: VERIFIER_WARP_HISTORY_PROTOCOL_V2,
        vacc_input_arity: VERIFIER_WARP_VACC_INPUT_ARITY_V2 as u8,
        protocol_digest: digest(300),
        relation_digest: producer.fresh_input.relation_digest,
        batch_index: 0,
        active_child_count: producer
            .active_count
            .expected_active_child_count
            .as_canonical_u32() as u8,
        children: {
            let mut children =
                [VerifierWarpChildRecordV2::padding(); VERIFIER_WARP_SOURCE_CHILD_CAPACITY_V2];
            children[0] = VerifierWarpChildRecordV2 {
                occupied: true,
                input: VerifierWarpVmStateV2 {
                    pc: producer.vm_pvs.initial_pc,
                    memory_root: producer.vm_pvs.initial_root,
                },
                output: VerifierWarpVmStateV2 {
                    pc: producer.vm_pvs.final_pc,
                    memory_root: producer.vm_pvs.final_root,
                },
                exit_code: producer.vm_pvs.exit_code,
                terminates: producer.vm_pvs.is_terminate == F::ONE,
            };
            children
        },
        program_commitment: producer.vm_pvs.program_commit,
        prior_accumulator_digest: producer.replay.previous_accumulator_digest,
        source_accumulator_digest: producer.replay.fresh_instance_digest,
        output_accumulator_digest: producer.replay.next_accumulator_digest,
        source_commitment_root: producer.active_count.source_root,
        external_logup_gkr_digest: producer.logup.checkpoint_digest,
        source_functional_digest: producer.replay.opening_claim_digest,
        setup_openings_digest: producer.fixed_setup_opening.setup_openings_digest,
        source_statement_digest: producer.replay.replay_endpoint_digest,
        transition_transcript_digest: producer.replay.replay_binding_digest,
        public_values_digest: digest(900),
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

#[derive(Clone)]
struct AuthorityComposition {
    airs: Vec<AirRef<NativeSC>>,
    matrices: Vec<RowMajorMatrix<F>>,
    public_values: Vec<Vec<F>>,
    selected_bus_indices: Vec<BusIndex>,
}

impl AuthorityComposition {
    fn check(&self) {
        let preprocessed_owned = self
            .airs
            .iter()
            .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
            .collect::<Vec<_>>();
        for (((air, matrix), public_values), preprocessed) in self
            .airs
            .iter()
            .zip(&self.matrices)
            .zip(&self.public_values)
            .zip(&preprocessed_owned)
        {
            check_constraints::<_, NativeSC>(
                air.as_ref(),
                &air.name(),
                &preprocessed.as_ref().map(RowMajorMatrix::as_view),
                &[matrix.as_view()],
                public_values,
            );
        }
        let preprocessed = preprocessed_owned
            .iter()
            .map(|matrix| matrix.as_ref().map(RowMajorMatrix::as_view))
            .collect::<Vec<_>>();
        let interactions = self
            .airs
            .iter()
            .map(|air| {
                symbolic_interactions(air.as_ref())
                    .into_iter()
                    .filter(|interaction| {
                        self.selected_bus_indices.contains(&interaction.bus_index)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let views = self
            .matrices
            .iter()
            .map(|matrix| vec![matrix.as_view()])
            .collect::<Vec<_>>();
        check_logup(
            &self.airs.iter().map(|air| air.name()).collect::<Vec<_>>(),
            &interactions,
            &preprocessed,
            &views,
            &self.public_values,
        );
    }
}

fn authority_composition() -> (AuthorityComposition, VerifierWarpActiveCountFunctionalAirV2) {
    let mut manager = BusIndexManager::new();
    let inventory = BusInventory::new(&mut manager);
    let transcript = TranscriptModule::<1>::new(
        inventory.clone(),
        SystemParams::new_for_testing(10),
        false,
        false,
    );
    let poseidon_owner = transcript.poseidon2_bus_owner();

    let fixed_point_bus = FixedSetupOpeningPointBusV2::new(manager.new_bus_idx());
    let fixed_node_bus = FixedSetupOpeningNodeBusV2::new(manager.new_bus_idx());
    let fixed_pair_bus = VerifiedFixedSetupOpeningPairBusV2::new(manager.new_bus_idx());
    let column_claims_bus = ColumnClaimsBus::new(manager.new_bus_idx());
    let fixed_certificate_bus = FixedSetupOpeningCertificateBusV2::new(manager.new_bus_idx());

    let active_transcript_bus = TranscriptBus::new(manager.new_bus_idx());
    let active_phase_bus = VerifierWarpActiveCountPhaseStartBusV2::new(manager.new_bus_idx());
    let active_ordinary_bus = VerifierWarpOrdinarySourceFunctionalBusV2::new(manager.new_bus_idx());
    let active_claim_bus = VerifierWarpAugmentedSourceClaimBusV2::new(manager.new_bus_idx());
    let active_reduction_bus =
        VerifierWarpAugmentedSourceReductionBusV2::new(manager.new_bus_idx());
    let active_certificate_index = manager.new_bus_idx();
    let active_certificate_bus =
        VerifierWarpCertifiedActiveCountBusV2::new(active_certificate_index);

    let logup_index = manager.new_bus_idx();
    let vacc_input_index = manager.new_bus_idx();
    let replay_index = manager.new_bus_idx();
    let source_index = manager.new_bus_idx();
    let vacc_index = manager.new_bus_idx();
    let block_index = manager.new_bus_idx();
    let logup_bus = LogUpOnlyHistoryBusV19::new(logup_index);
    let vacc_input_bus = CertifiedDirectAirVaccInputBusV19::new(vacc_input_index);
    let replay_bus = CertifiedWarpReplayBusV19::new(replay_index);
    let fixed_source_bus = CertifiedFixedMultiAirSourceBusV2::new(manager.new_bus_idx());
    let fresh_explicit_bus = CertifiedFreshExplicitDigestBusV19::new(manager.new_bus_idx());
    let fixed_public_values_bus = FixedSourcePublicValueBusV19::new(manager.new_bus_idx());
    let source_bus = VerifierWarpSourceCertificateBusV2::new(source_index);
    let vacc_bus = VerifierWarpVaccCertificateBusV2::new(vacc_index);

    let relation_digest = digest(200);
    let source_relation_vk_digest = digest(220);
    let active_profile_digest = digest(240);
    let source_message_root = digest(260);

    let (fixed_air, fixed_trace, fixed_certificate) = fixed_setup_fixture(
        relation_digest,
        source_relation_vk_digest,
        fixed_point_bus,
        fixed_node_bus,
        fixed_pair_bus,
        column_claims_bus,
        fixed_certificate_bus,
        inventory.poseidon2_compress_bus,
    );
    let active_profile = active_count_profile(relation_digest, active_profile_digest);
    let active_air = VerifierWarpActiveCountFunctionalAirV2 {
        profile: active_profile.clone(),
        transcript_bus: active_transcript_bus,
        phase_start_bus: active_phase_bus,
        ordinary_source_bus: active_ordinary_bus,
        augmented_claim_bus: active_claim_bus,
        augmented_reduction_bus: active_reduction_bus,
        certified_count_bus: active_certificate_bus,
    };
    let active_record = active_count_record(&active_profile, source_message_root);
    let active_trace = generate_verifier_warp_active_count_functional_trace_v2(
        &active_air,
        core::slice::from_ref(&active_record),
    )
    .expect("C2 trace");
    let active_certificate = certified_count(&active_profile, &active_record);

    let producer_record = producer_record(
        relation_digest,
        source_relation_vk_digest,
        source_message_root,
        fixed_certificate,
        active_certificate,
    );
    let bridge_air = VerifierWarpProducerBridgeAirV2 {
        segment_start: 0,
        fixed_public_values: VerifierWarpFixedPublicValuesProfileV2 {
            log_constraints: 1,
            sources: core::iter::once(VerifierWarpFixedPublicValueSourceV2::TrustedConstant(
                F::ONE,
            ))
            .chain(
                (0..VmPvs::<u8>::width())
                    .map(VerifierWarpFixedPublicValueSourceV2::VmPvsCoordinate),
            )
            .collect::<Vec<_>>()
            .into(),
            batch_count: 1,
        },
        protocol_digest: digest(300),
        admitted_relation_digest: relation_digest,
        expected_key_digest: producer_record.replay.key_digest,
        source_relation_vk_digest,
        source_log_message_len: 1,
        source_log_codeword_len: 1,
        active_count_profile_digest: active_profile_digest,
        logup_bus,
        fixed_source_bus,
        vacc_input_bus,
        fresh_explicit_bus,
        replay_bus,
        fixed_public_values_bus,
        fixed_setup_opening_bus: Some(fixed_certificate_bus),
        active_count_bus: active_certificate_bus,
        source_bus,
        vacc_bus,
    };
    let bridge_trace = generate_verifier_warp_producer_bridge_trace_v2(
        &bridge_air,
        core::slice::from_ref(&producer_record),
    )
    .expect("bridge trace");
    let history_air = VerifierWarpHistoryAirV2 {
        source_bus,
        vacc_bus,
        block_public_values_bus: VerifierWarpBlockPublicValuesBusV2::new(block_index),
        compress_bus: inventory.poseidon2_compress_bus,
        output_mode: VerifierWarpHistoryOutputModeV2::StandalonePublicValues,
    };
    let history_trace =
        generate_verifier_warp_history_trace_v2(&[history_record(&producer_record)])
            .expect("History trace");
    let c3_air = C3AuthorityStubAir {
        logup_bus,
        vacc_input_bus,
        replay_bus,
    };
    let c3_trace = c3_stub_trace(&producer_record);

    let mut compression_inputs = fixed_trace.compression_inputs.clone();
    compression_inputs.extend(history_trace.compression_inputs.iter().copied());
    let poseidon_matrix = transcript
        .build_poseidon2_multibus_traces(vec![(Vec::new(), compression_inputs)])
        .expect("Poseidon table")
        .pop()
        .expect("one Poseidon table");
    let poseidon_air =
        transcript.multi_bus_poseidon2_air_for_owners::<NativeSC>(&[Poseidon2BusOwner {
            permute_bus: poseidon_owner.permute_bus,
            compress_bus: poseidon_owner.compress_bus,
        }]);

    let airs: Vec<AirRef<NativeSC>> = vec![
        Arc::new(fixed_air),
        Arc::new(active_air.clone()),
        Arc::new(bridge_air),
        Arc::new(history_air),
        Arc::new(c3_air),
        poseidon_air,
    ];
    let matrices = vec![
        fixed_trace.matrix,
        active_trace,
        bridge_trace,
        history_trace.matrix,
        c3_trace,
        poseidon_matrix,
    ];
    let public_values = vec![
        Vec::new(),
        Vec::new(),
        Vec::new(),
        history_trace.public_values.to_vec(),
        Vec::new(),
        Vec::new(),
    ];
    // Every authority edge under test is present.  Only unrelated C1/C2
    // predecessor buses and the block-PV bus are out of this focused
    // composition; the three legacy producer inputs are balanced by the
    // explicit C3 stub above.
    let selected_bus_indices = vec![
        fixed_certificate_bus.index(),
        active_certificate_index,
        logup_index,
        vacc_input_index,
        replay_index,
        source_index,
        vacc_index,
        poseidon_owner.permute_bus.index(),
        poseidon_owner.compress_bus.index(),
    ];
    (
        AuthorityComposition {
            airs,
            matrices,
            public_values,
            selected_bus_indices,
        },
        active_air,
    )
}

#[test]
fn c1_c2_bridge_history_authority_path_is_bus_balanced() {
    let (composition, _) = authority_composition();
    composition.check();

    let bridge_width = VerifierWarpProducerBridgeColsV2::<u8>::width();
    let bridge: &VerifierWarpProducerBridgeColsV2<F> =
        composition.matrices[2].values[..bridge_width].borrow();
    let history_width = VerifierWarpHistoryRowColsV2::<u8>::width();
    let history: &VerifierWarpHistoryRowColsV2<F> =
        composition.matrices[3].values[..history_width].borrow();
    assert_eq!(
        history.setup_openings_digest,
        bridge.fixed_setup_opening.setup_openings_digest
    );
    assert_ne!(
        history.setup_openings_digest, bridge.replay.segment_openings_digest,
        "C1 digest must not alias the legacy segment-opening digest"
    );
    assert_eq!(
        history.active_child_count,
        bridge.active_count.expected_active_child_count
    );
    assert_eq!(
        history.source_commitment_root,
        bridge.active_count.source_root
    );
    assert_ne!(
        history.source_commitment_root, bridge.replay.source_forest_root,
        "C2 must authenticate the exact systematic WARP message"
    );
}

fn assert_composition_rejects(mut mutate: impl FnMut(&mut AuthorityComposition)) {
    let (mut composition, _) = authority_composition();
    mutate(&mut composition);
    assert!(
        std::panic::catch_unwind(AssertUnwindSafe(|| composition.check())).is_err(),
        "mutated authority composition unexpectedly balanced"
    );
}

#[test]
fn c1_c2_certificate_identity_and_opening_metadata_mutations_reject() {
    assert_composition_rejects(|composition| {
        let bridge_width = VerifierWarpProducerBridgeColsV2::<u8>::width();
        let bridge: &mut VerifierWarpProducerBridgeColsV2<F> =
            composition.matrices[2].values[..bridge_width].borrow_mut();
        bridge.fixed_setup_opening.setup_openings_digest[0] += F::ONE;
        let history_width = VerifierWarpHistoryRowColsV2::<u8>::width();
        let history: &mut VerifierWarpHistoryRowColsV2<F> =
            composition.matrices[3].values[..history_width].borrow_mut();
        history.setup_openings_digest[0] += F::ONE;
    });
    assert_composition_rejects(|composition| {
        let bridge_width = VerifierWarpProducerBridgeColsV2::<u8>::width();
        let bridge: &mut VerifierWarpProducerBridgeColsV2<F> =
            composition.matrices[2].values[..bridge_width].borrow_mut();
        bridge.fixed_setup_opening.canonical_claim_count += F::ONE;
    });
    assert_composition_rejects(|composition| {
        let bridge_width = VerifierWarpProducerBridgeColsV2::<u8>::width();
        let bridge: &mut VerifierWarpProducerBridgeColsV2<F> =
            composition.matrices[2].values[..bridge_width].borrow_mut();
        bridge.fixed_setup_opening.source_relation_vk_digest[0] += F::ONE;
    });
    assert_composition_rejects(|composition| {
        let bridge_width = VerifierWarpProducerBridgeColsV2::<u8>::width();
        let bridge: &mut VerifierWarpProducerBridgeColsV2<F> =
            composition.matrices[2].values[..bridge_width].borrow_mut();
        bridge.active_count.expected_active_child_count = F::TWO;
        let history_width = VerifierWarpHistoryRowColsV2::<u8>::width();
        let history: &mut VerifierWarpHistoryRowColsV2<F> =
            composition.matrices[3].values[..history_width].borrow_mut();
        history.active_child_count = F::TWO;
        history.active_count_flags = core::array::from_fn(|index| F::from_bool(index == 1));
    });
    assert_composition_rejects(|composition| {
        let bridge_width = VerifierWarpProducerBridgeColsV2::<u8>::width();
        let bridge: &mut VerifierWarpProducerBridgeColsV2<F> =
            composition.matrices[2].values[..bridge_width].borrow_mut();
        bridge.active_count.source_root[0] += F::ONE;
        let history_width = VerifierWarpHistoryRowColsV2::<u8>::width();
        let history: &mut VerifierWarpHistoryRowColsV2<F> =
            composition.matrices[3].values[..history_width].borrow_mut();
        history.source_commitment_root[0] += F::ONE;
    });
    assert_composition_rejects(|composition| {
        let bridge_width = VerifierWarpProducerBridgeColsV2::<u8>::width();
        let bridge: &mut VerifierWarpProducerBridgeColsV2<F> =
            composition.matrices[2].values[..bridge_width].borrow_mut();
        bridge.active_count.point[0][0] += F::ONE;
    });
    assert_composition_rejects(|composition| {
        let bridge_width = VerifierWarpProducerBridgeColsV2::<u8>::width();
        let bridge: &mut VerifierWarpProducerBridgeColsV2<F> =
            composition.matrices[2].values[..bridge_width].borrow_mut();
        bridge.active_count.point_len -= F::ONE;
    });
    assert_composition_rejects(|composition| {
        let bridge_width = VerifierWarpProducerBridgeColsV2::<u8>::width();
        let bridge: &mut VerifierWarpProducerBridgeColsV2<F> =
            composition.matrices[2].values[..bridge_width].borrow_mut();
        bridge.active_count.batching_coefficient[0] += F::ONE;
    });
    assert_composition_rejects(|composition| {
        let bridge_width = VerifierWarpProducerBridgeColsV2::<u8>::width();
        let bridge: &mut VerifierWarpProducerBridgeColsV2<F> =
            composition.matrices[2].values[..bridge_width].borrow_mut();
        bridge.active_count.message_value[0] += F::ONE;
    });
    assert_composition_rejects(|composition| {
        let bridge_width = VerifierWarpProducerBridgeColsV2::<u8>::width();
        let bridge: &mut VerifierWarpProducerBridgeColsV2<F> =
            composition.matrices[2].values[..bridge_width].borrow_mut();
        bridge.active_count.reduction_end_tidx += F::ONE;
    });
    assert_composition_rejects(|composition| {
        let bridge_width = VerifierWarpProducerBridgeColsV2::<u8>::width();
        let bridge: &mut VerifierWarpProducerBridgeColsV2<F> =
            composition.matrices[2].values[..bridge_width].borrow_mut();
        bridge.active_count.proof_index += F::ONE;
    });
    assert_composition_rejects(|composition| {
        let bridge_width = VerifierWarpProducerBridgeColsV2::<u8>::width();
        let bridge: &mut VerifierWarpProducerBridgeColsV2<F> =
            composition.matrices[2].values[..bridge_width].borrow_mut();
        bridge.active_count.batch_index[0] += F::ONE;
    });
    assert_composition_rejects(|composition| {
        let bridge_width = VerifierWarpProducerBridgeColsV2::<u8>::width();
        let bridge: &mut VerifierWarpProducerBridgeColsV2<F> =
            composition.matrices[2].values[..bridge_width].borrow_mut();
        bridge.active_count.relation_digest[0] += F::ONE;
    });
    assert_composition_rejects(|composition| {
        let bridge_width = VerifierWarpProducerBridgeColsV2::<u8>::width();
        let bridge: &mut VerifierWarpProducerBridgeColsV2<F> =
            composition.matrices[2].values[..bridge_width].borrow_mut();
        bridge.active_count.profile_digest[0] += F::ONE;
    });
}

#[test]
fn c2_offset_scale_and_is_valid_authority_mutations_reject_before_bridge() {
    let relation_digest = digest(200);
    let profile = active_count_profile(relation_digest, digest(240));
    let mut manager = BusIndexManager::new();
    let air = VerifierWarpActiveCountFunctionalAirV2 {
        profile: profile.clone(),
        transcript_bus: TranscriptBus::new(manager.new_bus_idx()),
        phase_start_bus: VerifierWarpActiveCountPhaseStartBusV2::new(manager.new_bus_idx()),
        ordinary_source_bus: VerifierWarpOrdinarySourceFunctionalBusV2::new(manager.new_bus_idx()),
        augmented_claim_bus: VerifierWarpAugmentedSourceClaimBusV2::new(manager.new_bus_idx()),
        augmented_reduction_bus: VerifierWarpAugmentedSourceReductionBusV2::new(
            manager.new_bus_idx(),
        ),
        certified_count_bus: VerifierWarpCertifiedActiveCountBusV2::new(manager.new_bus_idx()),
    };
    let honest = active_count_record(&profile, digest(260));
    assert!(generate_verifier_warp_active_count_functional_trace_v2(
        &air,
        core::slice::from_ref(&honest)
    )
    .is_ok());

    for mutate in [
        (|record: &mut VerifierWarpActiveCountFunctionalRecordV2| {
            record.reduction_block_start += 4;
        }) as fn(&mut VerifierWarpActiveCountFunctionalRecordV2),
        |record| record.reduction_count_term_scale += EF::ONE,
        |record| record.reduction_expected_active_child_count = 2,
        |record| record.reduction_combined_target += EF::ONE,
        |record| record.reduction_point[0] += EF::ONE,
        |record| record.reduction_source_root[0] += F::ONE,
        |record| record.batch_index += 1,
    ] {
        let mut bad = honest.clone();
        mutate(&mut bad);
        assert!(
            generate_verifier_warp_active_count_functional_trace_v2(&air, &[bad]).is_err(),
            "mutated C2 source reduction unexpectedly became authority"
        );
    }
}

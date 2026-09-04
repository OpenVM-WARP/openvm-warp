use core::borrow::{Borrow, BorrowMut};
use std::{panic::AssertUnwindSafe, sync::Arc};

use openvm_recursion_circuit::{
    bus::{
        ColumnClaimsBus, Poseidon2CompressBus, Poseidon2PermuteBus, TranscriptBus,
        TranscriptBusMessage,
    },
    whir::multi_constraint::{
        air::{MultiConstraintCallerPrefixBus, MultiConstraintInitialCommitmentBus},
        MultiConstraintWhirInitialCommitment,
    },
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    air_builders::{
        debug::{check_constraints, check_logup},
        symbolic::get_symbolic_builder,
    },
    hasher::MerkleHasher,
    interaction::{BusIndex, InteractionBuilder, SymbolicInteraction},
    keygen::types::TraceWidth,
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField32},
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    AirRef, AnyAir, BaseAirWithPublicValues, PartitionedBaseAir, StarkProtocolConfig, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as SC, Digest, DIGEST_SIZE, D_EF, EF, F,
};

use super::*;
use crate::circuit::native_warp_history_v19::{
    SetupPcsSourceProvenanceBusV3, SetupPcsSourceProvenanceMessageV3,
    SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3,
};

const TRANSCRIPT_BUS: u16 = 70;

fn digest(seed: u32) -> Digest {
    core::array::from_fn(|index| F::from_u32(seed + index as u32 + 1))
}

fn ef(seed: u32) -> EF {
    EF::from_basis_coefficients_slice(&core::array::from_fn::<_, D_EF, _>(|index| {
        F::from_u32(seed + index as u32)
    }))
    .expect("EF4")
}

fn buses() -> SetupPcsAuthorityStatementBusesV3 {
    SetupPcsAuthorityStatementBusesV3 {
        source_provenance: SetupPcsSourceProvenanceBusV3::new(BusIndex::from(50u16)),
        source_opening_point: FixedSetupOpeningPointBusV2::new(BusIndex::from(51u16)),
        column_claims: ColumnClaimsBus::new(BusIndex::from(52u16)),
        canonical_fields: SetupPcsAuthorityCanonicalFieldBusV3::new(BusIndex::from(53u16)),
        source_receipts: SetupPcsAuthoritySourceReceiptBusV3::new(BusIndex::from(54u16)),
        derived_digests: SetupPcsAuthorityDerivedDigestBusV3::new(BusIndex::from(55u16)),
        claim_values: SetupPcsAuthorityClaimValueBusV3::new(BusIndex::from(56u16)),
        transition_statement: SetupPcsAuthorityTransitionStatementBusV3::new(BusIndex::from(57u16)),
        bound_batch: SetupPcsAuthorityBoundBatchBusV3::new(BusIndex::from(58u16)),
        bound_transition: SetupPcsAuthorityBoundTransitionBusV3::new(BusIndex::from(59u16)),
        stacking_schedule: SetupPcsAuthorityStackingScheduleBusV3::new(BusIndex::from(60u16)),
        initial_commitments: MultiConstraintInitialCommitmentBus::new(BusIndex::from(61u16)),
        multi_whir_caller_prefix: MultiConstraintCallerPrefixBus::new(BusIndex::from(62u16)),
        transcript: TranscriptBus::new(BusIndex::from(TRANSCRIPT_BUS)),
        poseidon_permute: Poseidon2PermuteBus::new(BusIndex::from(71u16)),
        poseidon_compress: Poseidon2CompressBus::new(BusIndex::from(72u16)),
    }
}

fn profile_fields() -> Vec<F> {
    let mut fields = 3u64.to_le_bytes().map(F::from_u8).to_vec();
    fields.extend((0..48).map(|index| F::from_u32(1000 + index)));
    fields
}

fn identities() -> Vec<SetupPcsAuthorityStatementClaimIdentityV3> {
    vec![
        SetupPcsAuthorityStatementClaimIdentityV3 {
            setup_index: 0,
            air_id: 9,
            setup_part_index: 0,
            sort_idx: 4,
            part_index: 2,
            kind_word: 1,
            column_index: 0,
            need_rot: false,
        },
        SetupPcsAuthorityStatementClaimIdentityV3 {
            setup_index: 0,
            air_id: 9,
            setup_part_index: 0,
            sort_idx: 4,
            part_index: 2,
            kind_word: 1,
            column_index: 1,
            need_rot: true,
        },
    ]
}

fn profile() -> Arc<SetupPcsAuthorityStatementProfileV3> {
    Arc::new(
        SetupPcsAuthorityStatementProfileV3::new(
            digest(10),
            digest(20),
            digest(30),
            profile_fields(),
            2,
            0,
            3,
            vec![MultiConstraintWhirInitialCommitment {
                commitment: digest(40),
                width: 2,
            }],
            0,
            identities(),
            vec![17, 19],
        )
        .expect("profile"),
    )
}

fn record(index: usize) -> SetupPcsAuthorityStatementTransitionRecordV3 {
    let base = 1000 * index as u32;
    SetupPcsAuthorityStatementTransitionRecordV3 {
        provenance: SetupPcsSourceProvenanceMessageV3 {
            protocol_version: F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition_index: [F::from_u16(index as u16), F::ZERO],
            segment_index: [F::from_u16(index as u16), F::ZERO],
            app_vk_digest: digest(10),
            relation_digest: digest(20),
            source_root: digest(100 + base),
            source_instance_digest: digest(200 + base),
            source_forest_root: digest(300 + base),
            segment_openings_digest: digest(400 + base),
            source_checkpoint_digest: digest(500 + base),
            source_manifest_digest: digest(600 + base),
            source_receipt_digest: digest(700 + base),
            end_tidx: [F::from_u16(13 + index as u16), F::ZERO],
            end_sample_count: F::from_u8(4),
            end_state: core::array::from_fn(|lane| F::from_u32(800 + base + lane as u32)),
        },
        setup_opening_point: vec![ef(900 + base), ef(910 + base), ef(920 + base)],
        claims: vec![
            SetupPcsAuthorityStatementClaimRecordV3 {
                current: ef(930 + base),
                rotated: None,
            },
            SetupPcsAuthorityStatementClaimRecordV3 {
                current: ef(940 + base),
                rotated: Some(ef(950 + base)),
            },
        ],
    }
}

fn records() -> Vec<SetupPcsAuthorityStatementTransitionRecordV3> {
    vec![record(0), record(1)]
}

fn push_u64(fields: &mut Vec<F>, value: u64) {
    fields.extend(value.to_le_bytes().map(F::from_u8));
}

fn push_u32(fields: &mut Vec<F>, value: u32) {
    push_u64(fields, u64::from(value));
}

fn push_ext(fields: &mut Vec<F>, value: EF) {
    fields.extend_from_slice(value.as_basis_coefficients_slice());
}

fn transition_fields(
    index: usize,
    record: &SetupPcsAuthorityStatementTransitionRecordV3,
) -> Vec<F> {
    let mut fields = Vec::new();
    push_u32(&mut fields, index as u32);
    fields.extend_from_slice(&record.provenance.source_root);
    fields.extend_from_slice(&record.provenance.source_instance_digest);
    fields.extend_from_slice(&record.provenance.source_forest_root);
    fields.extend_from_slice(&record.provenance.segment_openings_digest);
    fields.extend_from_slice(&record.provenance.end_state);
    push_u32(
        &mut fields,
        record.provenance.end_tidx[0].as_canonical_u32(),
    );
    push_u32(
        &mut fields,
        record.provenance.end_sample_count.as_canonical_u32(),
    );
    push_u64(&mut fields, 1);
    push_u32(&mut fields, 0);
    push_u64(&mut fields, record.setup_opening_point.len() as u64);
    for &coordinate in &record.setup_opening_point {
        push_ext(&mut fields, coordinate);
    }
    push_u64(&mut fields, record.claims.len() as u64);
    for (identity, claim) in identities().into_iter().zip(&record.claims) {
        for value in [
            identity.setup_index,
            identity.air_id,
            identity.setup_part_index,
            identity.kind_word,
            identity.column_index,
        ] {
            push_u32(&mut fields, value);
        }
        push_ext(&mut fields, claim.current);
        fields.push(F::from_bool(claim.rotated.is_some()));
        if let Some(rotated) = claim.rotated {
            push_ext(&mut fields, rotated);
        }
    }
    fields
}

fn digest_fields(tag: u64, payload: &[F]) -> Digest {
    let mut fields = Vec::new();
    push_u64(&mut fields, tag);
    push_u64(&mut fields, payload.len() as u64);
    fields.extend_from_slice(payload);
    SC::default_from_params(SystemParams::new_for_testing(10))
        .hasher()
        .hash_slice(&fields)
}

#[test]
fn exact_sdk_sponges_and_setup_owned_schedule_are_pinned() {
    let profile = profile();
    let records = records();
    let trace = generate_setup_pcs_authority_statement_trace_v3(&profile, &records).unwrap();
    for (index, record) in records.iter().enumerate() {
        assert_eq!(
            trace.transition_statements[index].transition_statement_digest,
            digest_fields(
                SETUP_PCS_AUTHORITY_TRANSITION_DIGEST_TAG_V3,
                &transition_fields(index, record),
            )
        );
    }
    let profile_payload = profile_fields();
    assert_eq!(
        trace.bound_batch.profile_digest,
        digest_fields(SETUP_PCS_AUTHORITY_PROFILE_DIGEST_TAG_V3, &profile_payload)
    );
    let mut batch = profile_payload;
    push_u64(&mut batch, records.len() as u64);
    for (index, record) in records.iter().enumerate() {
        let transition = transition_fields(index, record);
        push_u64(&mut batch, transition.len() as u64);
        batch.extend(transition);
    }
    assert_eq!(
        trace.bound_batch.batch_digest,
        digest_fields(SETUP_PCS_AUTHORITY_BATCH_DIGEST_TAG_V3, &batch)
    );
    assert_eq!(
        profile.first_claim_tidx(0),
        Some(profile.statement_prefix_end_tidx() + 5)
    );
    assert_eq!(
        profile.stacking_post_tidx(0),
        profile.first_claim_tidx(0).map(|v| v + 17)
    );
    assert_eq!(
        profile.first_claim_tidx(1),
        profile.stacking_post_tidx(0).map(|v| v + 5)
    );
    assert_eq!(
        profile.caller_prefix_tidx(),
        profile.stacking_post_tidx(1).unwrap()
    );
    assert_eq!(
        trace.stacking_schedules.last().unwrap().caller_prefix_tidx,
        F::from_usize(profile.caller_prefix_tidx())
    );
}

#[test]
fn provenance_point_claim_order_checkpoint_and_digest_mutations_change_or_reject_authority() {
    let profile = profile();
    let baseline = generate_setup_pcs_authority_statement_trace_v3(&profile, &records()).unwrap();
    for mutate in 0..5 {
        let mut changed = records();
        match mutate {
            0 => changed[0].provenance.source_root[0] += F::ONE,
            1 => changed[0].setup_opening_point[0] += EF::ONE,
            2 => changed[0].claims[0].current += EF::ONE,
            3 => changed[0].provenance.end_state[0] += F::ONE,
            4 => changed[0].provenance.end_tidx[0] += F::ONE,
            _ => unreachable!(),
        }
        let altered = generate_setup_pcs_authority_statement_trace_v3(&profile, &changed).unwrap();
        assert_ne!(
            baseline.transition_statements[0].transition_statement_digest,
            altered.transition_statements[0].transition_statement_digest,
        );
        assert_ne!(
            baseline.bound_batch.batch_digest,
            altered.bound_batch.batch_digest
        );
    }
    let mut reordered = records();
    let (left, right) = reordered[0].claims.split_at_mut(1);
    core::mem::swap(&mut left[0].current, &mut right[0].current);
    assert_ne!(
        baseline.transition_statements[0].transition_statement_digest,
        generate_setup_pcs_authority_statement_trace_v3(&profile, &reordered)
            .unwrap()
            .transition_statements[0]
            .transition_statement_digest,
    );
    let mut bad_checkpoint = records();
    bad_checkpoint[0].provenance.end_sample_count = F::ZERO;
    assert!(generate_setup_pcs_authority_statement_trace_v3(&profile, &bad_checkpoint).is_err());
    let mut bad_identity = identities();
    bad_identity.swap(0, 1);
    assert!(SetupPcsAuthorityStatementProfileV3::new(
        digest(10),
        digest(20),
        digest(30),
        profile_fields(),
        2,
        0,
        3,
        vec![MultiConstraintWhirInitialCommitment {
            commitment: digest(40),
            width: 2
        }],
        0,
        bad_identity,
        vec![17, 19],
    )
    .is_err());
    let mut tampered = baseline.transition_digests.clone();
    tampered.values[DIGEST_SIZE] += F::ONE;
    assert_ne!(tampered.values, baseline.transition_digests.values);
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct TranscriptProviderCols<T> {
    active: T,
    tidx: T,
    value: T,
}

#[derive(Clone)]
struct TranscriptProviderAir(TranscriptBus);
impl BaseAir<F> for TranscriptProviderAir {
    fn width(&self) -> usize {
        core::mem::size_of::<TranscriptProviderCols<u8>>()
    }
}
impl BaseAirWithPublicValues<F> for TranscriptProviderAir {}
impl PartitionedBaseAir<F> for TranscriptProviderAir {}
impl<AB> Air<AB> for TranscriptProviderAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("transcript provider row");
        let local: &TranscriptProviderCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        self.0.send(
            builder,
            AB::Expr::ZERO,
            TranscriptBusMessage {
                tidx: local.tidx.into(),
                value: local.value.into(),
                is_sample: AB::Expr::ZERO,
            },
            local.active,
        );
    }
}

fn provider_trace(observations: &[(usize, F)]) -> RowMajorMatrix<F> {
    let width = core::mem::size_of::<TranscriptProviderCols<u8>>();
    let height = observations.len().next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    for (row, &(tidx, value)) in observations.iter().enumerate() {
        let cols: &mut TranscriptProviderCols<F> =
            values[row * width..(row + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.tidx = F::from_usize(tidx);
        cols.value = value;
    }
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

fn check_transcript_bus(
    statement_air: SetupPcsAuthorityStatementTranscriptAirV3,
    statement_trace: RowMajorMatrix<F>,
    provider_trace: RowMajorMatrix<F>,
) {
    let airs: Vec<AirRef<SC>> = vec![
        Arc::new(statement_air),
        Arc::new(TranscriptProviderAir(TranscriptBus::new(BusIndex::from(
            TRANSCRIPT_BUS,
        )))),
    ];
    let matrices = [statement_trace, provider_trace];
    let preprocessed_owned = airs
        .iter()
        .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
        .collect::<Vec<_>>();
    for ((air, matrix), prep) in airs.iter().zip(&matrices).zip(&preprocessed_owned) {
        check_constraints::<_, SC>(
            air.as_ref(),
            &air.name(),
            &prep.as_ref().map(RowMajorMatrix::as_view),
            &[matrix.as_view()],
            &[],
        );
    }
    let interactions = airs
        .iter()
        .map(|air| {
            symbolic_interactions(air.as_ref())
                .into_iter()
                .filter(|interaction| interaction.bus_index == BusIndex::from(TRANSCRIPT_BUS))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let preprocessed = preprocessed_owned
        .iter()
        .map(|m| m.as_ref().map(RowMajorMatrix::as_view))
        .collect::<Vec<_>>();
    let views = matrices
        .iter()
        .map(|m| vec![m.as_view()])
        .collect::<Vec<_>>();
    check_logup(
        &airs.iter().map(|air| air.name()).collect::<Vec<_>>(),
        &interactions,
        &preprocessed,
        &views,
        &vec![Vec::new(); airs.len()],
    );
}

#[test]
fn skipped_or_reordered_prefix_is_transcript_bus_unbalanced() {
    let profile = profile();
    let trace = generate_setup_pcs_authority_statement_trace_v3(&profile, &records()).unwrap();
    let statement_air = SetupPcsAuthorityStatementTranscriptAirV3 {
        profile: profile.clone(),
        buses: buses(),
    };
    check_transcript_bus(
        statement_air.clone(),
        trace.transcript.clone(),
        provider_trace(&trace.transcript_observations),
    );
    let mut skipped = trace.transcript_observations.clone();
    let _ = skipped.remove(0);
    assert!(
        std::panic::catch_unwind(AssertUnwindSafe(|| check_transcript_bus(
            statement_air.clone(),
            trace.transcript.clone(),
            provider_trace(&skipped),
        )))
        .is_err()
    );
    let mut reordered = trace.transcript_observations.clone();
    let first_domain = profile.statement_prefix_end_tidx();
    let left = reordered[first_domain].1;
    reordered[first_domain].1 = reordered[first_domain + 1].1;
    reordered[first_domain + 1].1 = left;
    assert!(
        std::panic::catch_unwind(AssertUnwindSafe(|| check_transcript_bus(
            statement_air,
            trace.transcript,
            provider_trace(&reordered),
        )))
        .is_err()
    );
}

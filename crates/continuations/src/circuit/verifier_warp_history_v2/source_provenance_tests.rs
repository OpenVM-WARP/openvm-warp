use core::borrow::{Borrow, BorrowMut};
use std::{panic::AssertUnwindSafe, sync::Arc};

use openvm_recursion_circuit::{
    bus::{
        CertifiedTranscriptCheckpointBus, CertifiedTranscriptCheckpointMessage,
        ResumeTranscriptStateBus, ResumeTranscriptStateMessage,
    },
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
    p3_field::PrimeCharacteristicRing,
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    AirRef, AnyAir, BaseAirWithPublicValues, PartitionedBaseAir, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as NativeSC, Digest, DIGEST_SIZE, D_EF, F,
};

use super::*;
use crate::circuit::native_warp_history_v19::{
    SetupPcsSourceCheckpointBusV3, SetupPcsSourceCheckpointMessageV3, SetupPcsSourceManifestBusV3,
    SetupPcsSourceManifestMessageV3, SetupPcsSourceProvenanceBusV3,
    SetupPcsSourceProvenanceMessageV3, MAX_RAW_MESSAGE_POINT_LEN_V19,
    SETUP_PCS_SOURCE_PROVENANCE_MULTIPLICITY_V3, SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3,
    TRANSCRIPT_WIDTH_V19,
};

fn digest(seed: u32) -> Digest {
    core::array::from_fn(|index| F::from_u32(seed + index as u32 + 1))
}

fn transition_limbs(index: usize) -> [F; 2] {
    [F::from_u16(index as u16), F::ZERO]
}

fn record_at(index: usize, segment_index: u32) -> SetupPcsSourceProvenanceRecordV3 {
    let transition_index = transition_limbs(index);
    let segment_index = [
        F::from_u32(segment_index & 0xffff),
        F::from_u32(segment_index >> 16),
    ];
    let app_vk_digest = digest(100 + 1000 * index as u32);
    let relation_digest = digest(200 + 1000 * index as u32);
    let source_root = digest(300 + 1000 * index as u32);
    let source_instance_digest = digest(400 + 1000 * index as u32);
    let source_forest_root = digest(500 + 1000 * index as u32);
    let segment_openings_digest = digest(600 + 1000 * index as u32);
    let verifier_endpoint =
        core::array::from_fn(|limb| F::from_u32(700 + 1000 * index as u32 + limb as u32));
    let mut point = [[F::ZERO; D_EF]; MAX_RAW_MESSAGE_POINT_LEN_V19];
    for (coordinate, entry) in point.iter_mut().take(3).enumerate() {
        *entry = core::array::from_fn(|limb| {
            F::from_u32(800 + 1000 * index as u32 + 10 * coordinate as u32 + limb as u32)
        });
    }
    let value = core::array::from_fn(|limb| F::from_u32(900 + 1000 * index as u32 + limb as u32));
    let end_state =
        core::array::from_fn(|lane| F::from_u32(1000 + 1000 * index as u32 + lane as u32));
    SetupPcsSourceProvenanceRecordV3 {
        manifest: SetupPcsSourceManifestMessageV3 {
            protocol_version: F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition_index,
            segment_index,
            app_vk_digest,
            relation_digest,
            source_root,
            source_instance_digest,
            source_forest_root,
            segment_openings_digest,
        },
        checkpoint: SetupPcsSourceCheckpointMessageV3 {
            protocol_version: F::from_u32(SETUP_PCS_SOURCE_PROVENANCE_PROTOCOL_V3),
            transition_index,
            segment_index,
            app_vk_digest,
            source_forest_root,
            segment_openings_digest,
            verifier_endpoint,
            end_tidx: [F::from_u16(1100 + index as u16), F::ZERO],
            end_sample_count: F::from_u8(D_EF as u8),
            end_state,
            logup_history_digest: digest(1200 + 1000 * index as u32),
        },
        fixed_source: CertifiedFixedMultiAirSourceMessageV2 {
            proof_index: F::from_usize(index),
            segment_index_lo: segment_index[0],
            segment_index_hi: segment_index[1],
            active_child_count: F::ONE,
            app_vk_digest,
            relation_digest,
            source_forest_root,
            segment_openings_digest,
            source_root,
            point_len: F::from_u8(3),
            point,
            value,
            verifier_endpoint,
        },
    }
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct SourceProviderCols<T> {
    active: T,
    manifest: SetupPcsSourceManifestMessageV3<T>,
    checkpoint: SetupPcsSourceCheckpointMessageV3<T>,
    fixed_source: CertifiedFixedMultiAirSourceMessageV2<T>,
}

#[derive(Clone)]
struct SourceProviderAir {
    manifest_bus: SetupPcsSourceManifestBusV3,
    checkpoint_bus: SetupPcsSourceCheckpointBusV3,
    fixed_source_bus: CertifiedFixedMultiAirSourceBusV2,
}

impl BaseAir<F> for SourceProviderAir {
    fn width(&self) -> usize {
        core::mem::size_of::<SourceProviderCols<u8>>()
    }
}
impl BaseAirWithPublicValues<F> for SourceProviderAir {}
impl PartitionedBaseAir<F> for SourceProviderAir {}
impl<AB> Air<AB> for SourceProviderAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("source provider row");
        let local: &SourceProviderCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        self.manifest_bus
            .send(builder, local.manifest.clone(), local.active);
        self.checkpoint_bus
            .send(builder, local.checkpoint.clone(), local.active);
        self.fixed_source_bus
            .send(builder, local.fixed_source.clone(), local.active);
    }
}

fn source_provider_trace(records: &[SetupPcsSourceProvenanceRecordV3]) -> RowMajorMatrix<F> {
    let width = core::mem::size_of::<SourceProviderCols<u8>>();
    let height = records.len().next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    for (index, record) in records.iter().enumerate() {
        let cols: &mut SourceProviderCols<F> =
            values[index * width..(index + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.manifest = record.manifest.clone();
        cols.checkpoint = record.checkpoint.clone();
        cols.fixed_source = record.fixed_source.clone();
    }
    RowMajorMatrix::new(values, width)
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct ProvenanceSinkCols<T> {
    active: T,
    provenance: SetupPcsSourceProvenanceMessageV3<T>,
}

#[derive(Clone)]
struct ProvenanceSinkAir {
    provenance_bus: SetupPcsSourceProvenanceBusV3,
    multiplicity: u32,
}

impl BaseAir<F> for ProvenanceSinkAir {
    fn width(&self) -> usize {
        core::mem::size_of::<ProvenanceSinkCols<u8>>()
    }
}
impl BaseAirWithPublicValues<F> for ProvenanceSinkAir {}
impl PartitionedBaseAir<F> for ProvenanceSinkAir {}
impl<AB> Air<AB> for ProvenanceSinkAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("source provenance sink row");
        let local: &ProvenanceSinkCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        self.provenance_bus.receive(
            builder,
            local.provenance.clone(),
            AB::Expr::from(local.active) * AB::Expr::from_u32(self.multiplicity),
        );
    }
}

fn provenance_sink_trace(
    provenances: &[SetupPcsSourceProvenanceMessageV3<F>],
) -> RowMajorMatrix<F> {
    let width = core::mem::size_of::<ProvenanceSinkCols<u8>>();
    let height = provenances.len().next_power_of_two().max(2);
    let mut values = F::zero_vec(width * height);
    for (index, provenance) in provenances.iter().enumerate() {
        let cols: &mut ProvenanceSinkCols<F> =
            values[index * width..(index + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.provenance = provenance.clone();
    }
    RowMajorMatrix::new(values, width)
}

#[repr(C)]
#[derive(AlignedBorrow)]
struct ResumeSinkCols<T> {
    active: T,
    tidx: T,
    sample_count: T,
    state: [T; TRANSCRIPT_WIDTH_V19],
}

#[derive(Clone)]
struct ResumeSinkAir {
    resume_bus: ResumeTranscriptStateBus,
    checkpoint_bus: CertifiedTranscriptCheckpointBus,
}

impl BaseAir<F> for ResumeSinkAir {
    fn width(&self) -> usize {
        core::mem::size_of::<ResumeSinkCols<u8>>()
    }
}
impl BaseAirWithPublicValues<F> for ResumeSinkAir {}
impl PartitionedBaseAir<F> for ResumeSinkAir {}
impl<AB> Air<AB> for ResumeSinkAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("resume sink row");
        let local: &ResumeSinkCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        self.resume_bus.receive(
            builder,
            AB::Expr::ZERO,
            ResumeTranscriptStateMessage {
                tidx: local.tidx.into(),
                state: local.state.map(Into::into),
            },
            local.active,
        );
        self.checkpoint_bus.receive(
            builder,
            AB::Expr::ZERO,
            CertifiedTranscriptCheckpointMessage {
                kind: AB::Expr::ZERO,
                tidx: local.tidx.into(),
                sample_count: local.sample_count.into(),
                state: local.state.map(Into::into),
            },
            local.active,
        );
    }
}

fn resume_sink_trace(provenance: &SetupPcsSourceProvenanceMessageV3<F>) -> RowMajorMatrix<F> {
    let width = core::mem::size_of::<ResumeSinkCols<u8>>();
    let mut values = F::zero_vec(width * 2);
    let cols: &mut ResumeSinkCols<F> = values[..width].borrow_mut();
    cols.active = F::ONE;
    cols.tidx = provenance.end_tidx[0] + F::from_u32(1 << 16) * provenance.end_tidx[1];
    cols.sample_count = provenance.end_sample_count;
    cols.state = provenance.end_state;
    RowMajorMatrix::new(values, width)
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
struct ProvenanceComposition {
    airs: Vec<AirRef<NativeSC>>,
    matrices: Vec<RowMajorMatrix<F>>,
    selected_buses: Vec<BusIndex>,
}

impl ProvenanceComposition {
    fn check(&self) {
        let preprocessed_owned = self
            .airs
            .iter()
            .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
            .collect::<Vec<_>>();
        for ((air, matrix), preprocessed) in self
            .airs
            .iter()
            .zip(&self.matrices)
            .zip(&preprocessed_owned)
        {
            check_constraints::<_, NativeSC>(
                air.as_ref(),
                &air.name(),
                &preprocessed.as_ref().map(RowMajorMatrix::as_view),
                &[matrix.as_view()],
                &[],
            );
        }
        let interactions = self
            .airs
            .iter()
            .map(|air| {
                symbolic_interactions(air.as_ref())
                    .into_iter()
                    .filter(|interaction| self.selected_buses.contains(&interaction.bus_index))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let preprocessed = preprocessed_owned
            .iter()
            .map(|matrix| matrix.as_ref().map(RowMajorMatrix::as_view))
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
            &vec![Vec::new(); self.airs.len()],
        );
    }
}

fn composition() -> ProvenanceComposition {
    composition_at(0)
}

fn composition_at(segment_start: u32) -> ProvenanceComposition {
    let mut manager = BusIndexManager::new();
    let inventory = BusInventory::new(&mut manager);
    let transcript = TranscriptModule::<1>::new(
        inventory.clone(),
        SystemParams::new_for_testing(10),
        false,
        false,
    );
    let poseidon_owner = transcript.poseidon2_bus_owner();
    let manifest_bus = SetupPcsSourceManifestBusV3::new(manager.new_bus_idx());
    let checkpoint_bus = SetupPcsSourceCheckpointBusV3::new(manager.new_bus_idx());
    let fixed_source_index = manager.new_bus_idx();
    let fixed_source_bus = CertifiedFixedMultiAirSourceBusV2::new(fixed_source_index);
    let provenance_bus = SetupPcsSourceProvenanceBusV3::new(manager.new_bus_idx());
    let records = vec![
        record_at(0, segment_start),
        record_at(1, segment_start.checked_add(1).expect("test segment index")),
    ];
    let provider_air = SourceProviderAir {
        manifest_bus,
        checkpoint_bus,
        fixed_source_bus,
    };
    let provider_trace = source_provider_trace(&records);
    let provenance_air = SetupPcsSourceProvenanceAirV3 {
        profile: SetupPcsSourceProvenanceProfileV3 {
            transition_count: records.len(),
            output_multiplicity: SETUP_PCS_SOURCE_PROVENANCE_MULTIPLICITY_V3,
        },
        manifest_bus,
        checkpoint_bus,
        fixed_source_bus,
        provenance_bus,
        compress_bus: inventory.poseidon2_compress_bus,
    };
    let provenance_trace = generate_setup_pcs_source_provenance_trace_v3(&provenance_air, &records)
        .expect("source provenance trace");
    let sink_air = ProvenanceSinkAir {
        provenance_bus,
        // Simulate the setup-authority bridge and statement consumers. The
        // production resume bridge below is the third exact consumer.
        multiplicity: SETUP_PCS_SOURCE_PROVENANCE_MULTIPLICITY_V3 - 1,
    };
    let sink_trace = provenance_sink_trace(&provenance_trace.provenances);
    let routes = (0..records.len())
        .map(|_| VerifierWarpSourceResumeRouteV3 {
            resume_state_bus: ResumeTranscriptStateBus::new(manager.new_bus_idx()),
            checkpoint_bus: CertifiedTranscriptCheckpointBus::new(manager.new_bus_idx()),
        })
        .collect::<Vec<_>>();
    let resume_bridge_air = VerifierWarpSourceResumeBridgeAirV3 {
        segment_start,
        provenance_bus,
        routes: routes.clone().into(),
    };
    let resume_bridge_trace = generate_verifier_warp_source_resume_bridge_trace_v3(
        &resume_bridge_air,
        &provenance_trace.provenances,
    )
    .expect("source resume bridge trace");
    let resume_sink_airs = routes
        .iter()
        .map(|route| ResumeSinkAir {
            resume_bus: route.resume_state_bus,
            checkpoint_bus: route.checkpoint_bus,
        })
        .collect::<Vec<_>>();
    let resume_sink_traces = provenance_trace
        .provenances
        .iter()
        .map(resume_sink_trace)
        .collect::<Vec<_>>();
    let poseidon_trace = transcript
        .build_poseidon2_multibus_traces(vec![(Vec::new(), provenance_trace.compression_inputs)])
        .expect("Poseidon traces")
        .pop()
        .expect("Poseidon table");
    let poseidon_air =
        transcript.multi_bus_poseidon2_air_for_owners::<NativeSC>(&[Poseidon2BusOwner {
            permute_bus: poseidon_owner.permute_bus,
            compress_bus: poseidon_owner.compress_bus,
        }]);
    let mut airs: Vec<AirRef<NativeSC>> = vec![
        Arc::new(provider_air),
        Arc::new(provenance_air),
        Arc::new(sink_air),
        Arc::new(resume_bridge_air),
    ];
    airs.extend(
        resume_sink_airs
            .into_iter()
            .map(|air| Arc::new(air) as AirRef<NativeSC>),
    );
    airs.push(poseidon_air);
    let mut matrices = vec![
        provider_trace,
        provenance_trace.matrix,
        sink_trace,
        resume_bridge_trace,
    ];
    matrices.extend(resume_sink_traces);
    matrices.push(poseidon_trace);
    let mut selected_buses = vec![
        manifest_bus.index(),
        checkpoint_bus.index(),
        fixed_source_index,
        provenance_bus.index(),
        poseidon_owner.permute_bus.index(),
        poseidon_owner.compress_bus.index(),
    ];
    for route in routes {
        selected_buses.push(route.resume_state_bus.index());
        selected_buses.push(route.checkpoint_bus.index());
    }
    ProvenanceComposition {
        airs,
        matrices,
        selected_buses,
    }
}

fn assert_rejects(label: &str, mutate: impl FnMut(&mut ProvenanceComposition)) {
    assert_rejects_at(0, label, mutate);
}

fn assert_rejects_at(
    segment_start: u32,
    label: &str,
    mut mutate: impl FnMut(&mut ProvenanceComposition),
) {
    let mut composition = composition_at(segment_start);
    mutate(&mut composition);
    assert!(
        std::panic::catch_unwind(AssertUnwindSafe(|| composition.check())).is_err(),
        "mutated source provenance composition unexpectedly verified: {label}"
    );
}

#[test]
fn source_provenance_balances_exact_producers_hashes_and_three_consumers() {
    composition().check();
}

#[test]
fn source_provenance_nonzero_segment_start_binds_local_and_global_indices() {
    const SEGMENT_START: u32 = 4;
    composition_at(SEGMENT_START).check();

    assert_rejects_at(SEGMENT_START, "local transition index", |composition| {
        let width = core::mem::size_of::<SourceProviderCols<u8>>();
        let row: &mut SourceProviderCols<F> = composition.matrices[0].values[..width].borrow_mut();
        row.manifest.transition_index[0] += F::ONE;
    });
    assert_rejects_at(SEGMENT_START, "absolute segment index", |composition| {
        let width = core::mem::size_of::<SourceProviderCols<u8>>();
        let row: &mut SourceProviderCols<F> = composition.matrices[0].values[..width].borrow_mut();
        row.manifest.segment_index[0] += F::ONE;
    });
    assert_rejects_at(SEGMENT_START, "resume global route", |composition| {
        let base_width = core::mem::size_of::<VerifierWarpSourceResumeBridgeColsV3<u8>>();
        let row: &mut VerifierWarpSourceResumeBridgeColsV3<F> =
            composition.matrices[3].values[..base_width].borrow_mut();
        row.provenance.segment_index[0] -= F::ONE;
    });
}

#[test]
fn source_provenance_rejects_transition_checkpoint_and_source_splices() {
    assert_rejects("swapped checkpoints", |composition| {
        let width = core::mem::size_of::<SourceProviderCols<u8>>();
        let (left, right) = composition.matrices[0].values.split_at_mut(width);
        let first: &mut SourceProviderCols<F> = left[..width].borrow_mut();
        let second: &mut SourceProviderCols<F> = right[..width].borrow_mut();
        core::mem::swap(&mut first.checkpoint, &mut second.checkpoint);
        // A pure physical row reorder is intentionally harmless on a
        // permutation bus. Model the adversarial splice by relabelling the
        // foreign checkpoint as the destination transition.
        first.checkpoint.transition_index = transition_limbs(0);
        second.checkpoint.transition_index = transition_limbs(1);
    });
    for (label, mutate) in [
        (
            "end tidx",
            (|row: &mut SetupPcsSourceProvenanceColsV3<F>| row.checkpoint.end_tidx[0] += F::ONE)
                as fn(&mut SetupPcsSourceProvenanceColsV3<F>),
        ),
        (
            "sample count",
            |row: &mut SetupPcsSourceProvenanceColsV3<F>| row.checkpoint.end_sample_count += F::ONE,
        ),
        (
            "checkpoint state",
            |row: &mut SetupPcsSourceProvenanceColsV3<F>| row.checkpoint.end_state[7] += F::ONE,
        ),
        (
            "source root",
            |row: &mut SetupPcsSourceProvenanceColsV3<F>| row.manifest.source_root[0] += F::ONE,
        ),
    ] {
        assert_rejects(label, |composition| {
            let width = core::mem::size_of::<SetupPcsSourceProvenanceColsV3<u8>>();
            let row: &mut SetupPcsSourceProvenanceColsV3<F> =
                composition.matrices[1].values[..width].borrow_mut();
            mutate(row);
        });
    }
}

#[test]
fn one_shot_source_point_cannot_be_replaced_by_a_setup_point() {
    assert_rejects("raw point/setup point confusion", |composition| {
        let width = core::mem::size_of::<SetupPcsSourceProvenanceColsV3<u8>>();
        let row: &mut SetupPcsSourceProvenanceColsV3<F> =
            composition.matrices[1].values[..width].borrow_mut();
        // Candidate setup PLE point: this cannot replace the independently
        // certified one-shot systematic-message point.
        row.fixed_source.point[0] = [F::from_u32(90_001); D_EF];
    });
    assert_rejects("raw opening value", |composition| {
        let width = core::mem::size_of::<SetupPcsSourceProvenanceColsV3<u8>>();
        let row: &mut SetupPcsSourceProvenanceColsV3<F> =
            composition.matrices[1].values[..width].borrow_mut();
        row.fixed_source.value[0] += F::ONE;
    });
}

#[test]
fn source_provenance_output_multiplicity_is_exact() {
    assert_rejects("output multiplicity", |composition| {
        let width = core::mem::size_of::<ProvenanceSinkCols<u8>>();
        let row: &mut ProvenanceSinkCols<F> = composition.matrices[2].values[..width].borrow_mut();
        row.active = F::ZERO;
    });
}

#[test]
fn source_resume_bridge_rejects_state_route_and_checkpoint_mutations() {
    assert_rejects("resumed state", |composition| {
        let width = core::mem::size_of::<VerifierWarpSourceResumeBridgeColsV3<u8>>() + 2;
        let row = &mut composition.matrices[3].values[..width];
        let cols: &mut VerifierWarpSourceResumeBridgeColsV3<F> =
            row[..core::mem::size_of::<VerifierWarpSourceResumeBridgeColsV3<u8>>()].borrow_mut();
        cols.provenance.end_state[0] += F::ONE;
    });
    assert_rejects("resume route", |composition| {
        let base = core::mem::size_of::<VerifierWarpSourceResumeBridgeColsV3<u8>>();
        composition.matrices[3].values[base] = F::ZERO;
        composition.matrices[3].values[base + 1] = F::ONE;
    });
    assert_rejects("resume sample cursor", |composition| {
        let width = core::mem::size_of::<ResumeSinkCols<u8>>();
        let row: &mut ResumeSinkCols<F> = composition.matrices[4].values[..width].borrow_mut();
        row.sample_count += F::ONE;
    });
}

#[test]
fn source_provenance_profile_rejects_non_protocol_fanout() {
    for output_multiplicity in [0, 1, 2, 4, u32::MAX] {
        let profile = SetupPcsSourceProvenanceProfileV3 {
            transition_count: 1,
            output_multiplicity,
        };
        assert_eq!(
            profile.validate(),
            Err(SetupPcsSourceProvenanceErrorV3::OutputMultiplicity)
        );
    }
    SetupPcsSourceProvenanceProfileV3 {
        transition_count: 1,
        output_multiplicity: SETUP_PCS_SOURCE_PROVENANCE_MULTIPLICITY_V3,
    }
    .validate()
    .unwrap();

    for output_multiplicity in [0, 1, 3, 4, u32::MAX] {
        assert_eq!(
            SetupPcsSourceProvenanceProfileV4 {
                output_multiplicity,
                seed_relation_digest: digest(90_000),
            }
            .validate(),
            Err(SetupPcsSourceProvenanceErrorV3::OutputMultiplicity),
        );
    }
    SetupPcsSourceProvenanceProfileV4 {
        output_multiplicity: SETUP_PCS_SOURCE_PROVENANCE_MULTIPLICITY_V4,
        seed_relation_digest: digest(90_000),
    }
    .validate()
    .unwrap();
}

const _: () = assert!(TRANSCRIPT_WIDTH_V19 == 2 * DIGEST_SIZE);

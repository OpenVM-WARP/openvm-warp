//! Protocol-v19 CUDA shared-forest adapter for the direct-AIR VACC verifier.
//!
//! This is a verifier composition, not a second WARP protocol.  The standard
//! direct-AIR VACC cursor and algebra remain authoritative.  Its fresh source
//! lane is replaced by [`CudaSharedForestVerifierModuleV19`], which consumes
//! the exact root/range descriptor observed by the CUDA prover and
//! authenticates scalar shift answers against full shared-forest rows.
//!
//! SDK receipts must first be replayed by the backend's ordinary VACC verifier
//! using its CUDA forest opening verifier.  That replay produces
//! [`CudaDirectAirVaccVerificationV19`] plus a transcript log.  The SDK then
//! converts the compact forest proof with
//! `prepare_cuda_shared_forest_verifier_record_v19`; keeping this conversion
//! outside continuations avoids an SDK dependency cycle and any CPU fallback.

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_recursion_circuit::{
    native_warp::{
        NativeStandardVaccDigestBus, NativeStandardVaccEndBus, NativeStandardVaccProfile,
        NativeStandardVaccProtocolBus, NativeStandardVaccRootBus, NativeWarpPcdBusInventory,
    },
    system::BusInventory,
};
use openvm_stark_backend::{
    warp_accum::{MerkleBatchOpeningVerification, NativeTranscriptPhase, WarpVaccStepVerification},
    warp_pesat::AccumulatorInstance,
    AirRef, StarkProtocolConfig, SystemParams, TranscriptLog,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    BabyBearPoseidon2Config as NativeSC, Digest, EF, F,
};

use super::{
    CudaSharedForestBindingBusV19, CudaSharedForestDescriptorBusV19,
    CudaSharedForestVerifierModuleV19, CudaSharedForestVerifierProfileV19,
    CudaSharedForestVerifierRecordErrorV19, CudaSharedForestVerifierRecordV19,
    DirectAirVaccHistoryBusesV19, DirectAirVaccProducerRecordV19,
    DirectAirVaccVerifierBatchTraceV19, DirectAirVaccVerifierErrorV19,
    DirectAirVaccVerifierModuleV19, DirectAirVaccVerifierRecordV19,
};

/// Backend VACC replay with CUDA fresh authentication certified separately by
/// the shared-forest AIR.  Prior accumulators remain ordinary scalar WARP
/// commitments and retain their standard Merkle verification records.
pub type CudaDirectAirVaccVerificationV19 =
    WarpVaccStepVerification<EF, Digest, (), MerkleBatchOpeningVerification<EF, Digest>>;

/// Compact continuations-side join record.  Identifiers are dense and local
/// to one `(numeric shape, prior mode, CUDA forest shape)` group.  This type
/// does not claim or perform a global History-index remapping.
pub struct CudaDirectAirVaccVerifierRecordV19<'a> {
    pub producer: &'a DirectAirVaccProducerRecordV19,
    pub verification: &'a CudaDirectAirVaccVerificationV19,
    pub transcript: &'a TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    pub prior: Option<&'a AccumulatorInstance<EF, Digest>>,
    pub shared_forest: &'a CudaSharedForestVerifierRecordV19<'a>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CudaDirectAirVaccVerifierErrorV19 {
    Core(DirectAirVaccVerifierErrorV19),
    Forest(CudaSharedForestVerifierRecordErrorV19),
    RecordShape(&'static str),
    AirTraceCount,
}

impl From<DirectAirVaccVerifierErrorV19> for CudaDirectAirVaccVerifierErrorV19 {
    fn from(value: DirectAirVaccVerifierErrorV19) -> Self {
        Self::Core(value)
    }
}

impl From<CudaSharedForestVerifierRecordErrorV19> for CudaDirectAirVaccVerifierErrorV19 {
    fn from(value: CudaSharedForestVerifierRecordErrorV19) -> Self {
        Self::Forest(value)
    }
}

/// Production verifier-key composition for one CUDA VACC shape group.
pub struct CudaDirectAirVaccVerifierModuleV19 {
    pub core: DirectAirVaccVerifierModuleV19,
    pub forest: CudaSharedForestVerifierModuleV19,
}

impl CudaDirectAirVaccVerifierModuleV19 {
    #[allow(clippy::too_many_arguments)]
    pub fn new_shape_batched(
        profiles: Vec<NativeStandardVaccProfile>,
        include_prior: bool,
        forest_profile: CudaSharedForestVerifierProfileV19,
        shared: BusInventory,
        buses: NativeWarpPcdBusInventory,
        history_buses: DirectAirVaccHistoryBusesV19,
        protocol_bus: NativeStandardVaccProtocolBus,
        end_bus: NativeStandardVaccEndBus,
        statement_root_bus: NativeStandardVaccRootBus,
        statement_digest_bus: NativeStandardVaccDigestBus,
        descriptor_bus: CudaSharedForestDescriptorBusV19,
        binding_bus: CudaSharedForestBindingBusV19,
        system_params: SystemParams,
    ) -> Result<Self, CudaDirectAirVaccVerifierErrorV19> {
        let core = DirectAirVaccVerifierModuleV19::new_shape_batched_cuda_shared_forest(
            profiles,
            include_prior,
            shared.clone(),
            buses.clone(),
            history_buses,
            protocol_bus,
            end_bus,
            statement_root_bus,
            statement_digest_bus,
            system_params,
        )?;
        if forest_profile.log_codeword_len != core.profile.log_codeword_len
            || forest_profile.rows_per_query != core.profile.rows_per_query
            || forest_profile.input_variant != 1 + usize::from(include_prior) * 3
            || forest_profile.outer_tree_id != core.trees.fresh_outer
            || forest_profile.row_tree_id_offset != core.trees.fresh_rows
        {
            return Err(CudaDirectAirVaccVerifierErrorV19::RecordShape(
                "CUDA forest/VACC verifier-key shape",
            ));
        }
        let forest = CudaSharedForestVerifierModuleV19::new_vacc_integrated(
            forest_profile,
            shared,
            buses,
            statement_root_bus,
            descriptor_bus,
            binding_bus,
        )
        .map_err(CudaDirectAirVaccVerifierErrorV19::RecordShape)?;
        Ok(Self { core, forest })
    }

    /// AIRs in the exact order returned by [`Self::generate_traces`].
    #[must_use]
    pub fn airs<PCS: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<PCS>> {
        let mut airs = self.core.airs::<PCS>();
        airs.extend(self.forest.airs::<PCS>());
        airs
    }

    pub fn generate_traces(
        &self,
        config: &NativeSC,
        records: &[CudaDirectAirVaccVerifierRecordV19<'_>],
    ) -> Result<CudaDirectAirVaccVerifierTraceV19, CudaDirectAirVaccVerifierErrorV19> {
        if records.is_empty() {
            return Err(CudaDirectAirVaccVerifierErrorV19::RecordShape(
                "empty CUDA VACC batch",
            ));
        }
        for (proof_idx, record) in records.iter().enumerate() {
            let proof_idx_u32 = u32::try_from(proof_idx).map_err(|_| {
                CudaDirectAirVaccVerifierErrorV19::RecordShape("CUDA VACC proof index")
            })?;
            let fresh_spans = record
                .verification
                .transcript_phases
                .iter()
                .filter(|span| span.phase == NativeTranscriptPhase::FreshCommitments)
                .collect::<Vec<_>>();
            if record.producer.proof_index != proof_idx_u32
                || record.shared_forest.proof_idx != proof_idx
                || record.shared_forest.commitment.root != record.producer.fresh_root
                || record.verification.fresh_authentication.as_slice() != [()]
                || fresh_spans.as_slice() != [&record.shared_forest.fresh_commitments_span]
                || !same_transcript_v19(record.transcript, record.shared_forest.transcript)
            {
                return Err(CudaDirectAirVaccVerifierErrorV19::RecordShape(
                    "spliced CUDA forest/VACC record",
                ));
            }
        }

        let forest_records = records
            .iter()
            .map(|record| record.shared_forest)
            .collect::<Vec<_>>();
        let forest = self.forest.generate_traces_from_refs(&forest_records)?;
        let core_records = records
            .iter()
            .map(|record| DirectAirVaccVerifierRecordV19 {
                producer: record.producer,
                verification: record.verification,
                transcript: record.transcript,
                prior: record.prior,
            })
            .collect::<Vec<_>>();
        let mut core = self.core.generate_cuda_shared_forest_traces(
            config,
            &core_records,
            forest.poseidon_permutation_inputs,
            forest.poseidon_compression_inputs,
        )?;
        core.traces.extend(forest.traces);
        if core.traces.len() != self.airs::<NativeSC>().len() {
            return Err(CudaDirectAirVaccVerifierErrorV19::AirTraceCount);
        }
        Ok(CudaDirectAirVaccVerifierTraceV19 { inner: core })
    }
}

/// Complete CUDA composite witness.  Poseidon requests from both components
/// are already represented by the core verifier's single Poseidon matrix.
pub struct CudaDirectAirVaccVerifierTraceV19 {
    pub inner: DirectAirVaccVerifierBatchTraceV19,
}

impl CudaDirectAirVaccVerifierTraceV19 {
    #[must_use]
    pub fn air_matrices(&self) -> &[openvm_stark_backend::p3_matrix::dense::RowMajorMatrix<F>] {
        &self.inner.traces
    }

    pub fn into_air_matrices(
        self,
    ) -> Vec<openvm_stark_backend::p3_matrix::dense::RowMajorMatrix<F>> {
        self.inner.traces
    }
}

fn same_transcript_v19(
    left: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    right: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
) -> bool {
    left.values() == right.values()
        && left.samples() == right.samples()
        && left.perm_results() == right.perm_results()
        && left.events() == right.events()
        && left.permutation_transitions() == right.permutation_transitions()
}

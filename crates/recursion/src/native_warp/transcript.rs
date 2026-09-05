use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{transcript::TranscriptLog, AirRef, StarkProtocolConfig, SystemParams};
use openvm_stark_sdk::config::baby_bear_poseidon2::F;
use p3_matrix::dense::RowMajorMatrix;

use super::NativeWarpVerifierBusInventory;
use crate::{
    bus::TranscriptBus,
    system::{AirModule, BusInventory},
    transcript::{Poseidon2BusOwner, Poseidon2MultibusInputs, TranscriptModule},
};

/// Native WARP transcript verifier sharing the recursion circuit's Poseidon
/// table. It intentionally contributes only `TranscriptAir`; callers append
/// the returned permutation inputs to the parent transcript module.
pub struct NativeWarpTranscriptModule {
    inner: TranscriptModule,
}

pub struct NativeWarpTranscriptArtifacts {
    pub trace: RowMajorMatrix<F>,
    pub poseidon2_trace: RowMajorMatrix<F>,
    pub permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
}

pub struct NativeWarpTranscriptInputArtifacts {
    pub trace: RowMajorMatrix<F>,
    pub permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
}

impl NativeWarpTranscriptModule {
    /// Logical Poseidon bus owner for parent assemblies that combine this
    /// transcript with other recursion transcript modules in one table.
    #[must_use]
    pub fn poseidon2_bus_owner(&self) -> Poseidon2BusOwner {
        self.inner.poseidon2_bus_owner()
    }

    #[must_use]
    pub fn new(
        shared: &BusInventory,
        native: &NativeWarpVerifierBusInventory,
        params: SystemParams,
    ) -> Self {
        Self::new_with_final_state(shared, native, params, false, false)
    }

    #[must_use]
    pub fn new_with_final_state(
        shared: &BusInventory,
        native: &NativeWarpVerifierBusInventory,
        params: SystemParams,
        final_state_bus_enabled: bool,
        resume_state_bus_enabled: bool,
    ) -> Self {
        Self::new_for_bus_with_final_state_and_checkpoints(
            shared,
            native.transcript,
            params,
            final_state_bus_enabled,
            resume_state_bus_enabled,
            None,
        )
    }

    /// Build the history transcript with two authenticated intermediate
    /// checkpoints (pre-VACC and post-VACC).  Other recursive transcripts keep
    /// their existing width and bus inventory.
    #[must_use]
    pub fn new_with_certified_checkpoints(
        shared: &BusInventory,
        native: &NativeWarpVerifierBusInventory,
        params: SystemParams,
        final_state_bus_enabled: bool,
        resume_state_bus_enabled: bool,
    ) -> Self {
        Self::new_for_bus_with_final_state_and_checkpoints(
            shared,
            native.transcript,
            params,
            final_state_bus_enabled,
            resume_state_bus_enabled,
            Some(native.transcript_checkpoint),
        )
    }

    /// Build a transcript module for a caller-owned transcript bus while
    /// retaining the parent's shared Poseidon buses.
    ///
    /// Native terminal-WHIR verification has its own transcript namespace but
    /// contributes permutations to the same Poseidon table as the VACC
    /// history. Accepting the bus directly avoids allocating an otherwise
    /// unused native-PCD bus inventory.
    #[must_use]
    pub fn new_for_bus(
        shared: &BusInventory,
        transcript_bus: TranscriptBus,
        params: SystemParams,
    ) -> Self {
        Self::new_for_bus_with_final_state(shared, transcript_bus, params, false, false)
    }

    #[must_use]
    pub fn new_for_bus_with_final_state(
        shared: &BusInventory,
        transcript_bus: TranscriptBus,
        params: SystemParams,
        final_state_bus_enabled: bool,
        resume_state_bus_enabled: bool,
    ) -> Self {
        Self::new_for_bus_with_final_state_and_checkpoints(
            shared,
            transcript_bus,
            params,
            final_state_bus_enabled,
            resume_state_bus_enabled,
            None,
        )
    }

    /// Build a resumed transcript whose exact terminal operation index is
    /// exported on the shared `TranscriptEndIndexBus`.
    ///
    /// The reduced-SWIRL transition-tree finalizer proves only the footer and
    /// terminal-Decision suffix.  Its terminal bridge must still learn that
    /// this suffix reaches the verifier-recorded end cursor; resumption alone
    /// authenticates the start state but does not provide that length check.
    #[must_use]
    pub fn new_for_bus_resumed_with_end_index(
        shared: &BusInventory,
        transcript_bus: TranscriptBus,
        params: SystemParams,
    ) -> Self {
        let mut module =
            Self::new_for_bus_with_final_state(shared, transcript_bus, params, false, true);
        module.inner.enable_end_index_bus();
        module
    }

    fn new_for_bus_with_final_state_and_checkpoints(
        shared: &BusInventory,
        transcript_bus: TranscriptBus,
        params: SystemParams,
        final_state_bus_enabled: bool,
        resume_state_bus_enabled: bool,
        checkpoint_state_bus: Option<crate::bus::CertifiedTranscriptCheckpointBus>,
    ) -> Self {
        let mut inventory = shared.clone();
        inventory.transcript_bus = transcript_bus;
        let inner = TranscriptModule::new_with_checkpoint_bus(
            inventory,
            params,
            final_state_bus_enabled,
            resume_state_bus_enabled,
            checkpoint_state_bus,
        );
        Self { inner }
    }

    #[must_use]
    pub fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        self.inner.airs::<SC>().into_iter().take(2).collect()
    }

    #[must_use]
    pub fn multi_bus_poseidon_air<SC: StarkProtocolConfig<F = F>>(owners: &[&Self]) -> AirRef<SC> {
        let first = owners
            .first()
            .expect("multi-bus Poseidon AIR needs an owner");
        let inner_owners = owners.iter().map(|owner| &owner.inner).collect::<Vec<_>>();
        first.inner.multi_bus_poseidon2_air(&inner_owners)
    }

    pub fn generate_trace(
        &self,
        logs: &[&TranscriptLog<F, [F; POSEIDON2_WIDTH]>],
        required_height: Option<usize>,
    ) -> Option<NativeWarpTranscriptArtifacts> {
        self.generate_trace_with_external(logs, Vec::new(), Vec::new(), required_height, None)
    }

    pub fn generate_trace_with_external(
        &self,
        logs: &[&TranscriptLog<F, [F; POSEIDON2_WIDTH]>],
        external_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        external_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_transcript_height: Option<usize>,
        required_poseidon_height: Option<usize>,
    ) -> Option<NativeWarpTranscriptArtifacts> {
        self.generate_trace_with_external_resumed(
            logs,
            &[],
            external_permutation_inputs,
            external_compression_inputs,
            required_transcript_height,
            required_poseidon_height,
        )
    }

    /// As [`Self::generate_trace_with_external`], but each log may continue an
    /// earlier transcript rather than start at the zero sponge.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_trace_with_external_resumed(
        &self,
        logs: &[&TranscriptLog<F, [F; POSEIDON2_WIDTH]>],
        resumes: &[Option<(usize, [F; POSEIDON2_WIDTH])>],
        external_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        external_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_transcript_height: Option<usize>,
        required_poseidon_height: Option<usize>,
    ) -> Option<NativeWarpTranscriptArtifacts> {
        self.generate_trace_with_external_resumed_and_checkpoints(
            logs,
            resumes,
            &[],
            external_permutation_inputs,
            external_compression_inputs,
            required_transcript_height,
            required_poseidon_height,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_trace_with_external_resumed_and_checkpoints(
        &self,
        logs: &[&TranscriptLog<F, [F; POSEIDON2_WIDTH]>],
        resumes: &[Option<(usize, [F; POSEIDON2_WIDTH])>],
        checkpoints: &[Option<[usize; 2]>],
        external_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        external_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_transcript_height: Option<usize>,
        required_poseidon_height: Option<usize>,
    ) -> Option<NativeWarpTranscriptArtifacts> {
        let artifacts = self.generate_trace_inputs_with_external_resumed_and_checkpoints(
            logs,
            resumes,
            checkpoints,
            external_permutation_inputs,
            external_compression_inputs,
            required_transcript_height,
        )?;
        let poseidon2_trace = self.build_poseidon2_trace(
            artifacts.permutation_inputs.clone(),
            artifacts.compression_inputs.clone(),
            required_poseidon_height,
        )?;
        Some(NativeWarpTranscriptArtifacts {
            trace: artifacts.trace,
            poseidon2_trace,
            permutation_inputs: artifacts.permutation_inputs,
            compression_inputs: artifacts.compression_inputs,
        })
    }

    pub fn generate_trace_inputs_with_external(
        &self,
        logs: &[&TranscriptLog<F, [F; POSEIDON2_WIDTH]>],
        external_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        external_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_transcript_height: Option<usize>,
    ) -> Option<NativeWarpTranscriptInputArtifacts> {
        self.generate_trace_inputs_with_external_resumed(
            logs,
            &[],
            external_permutation_inputs,
            external_compression_inputs,
            required_transcript_height,
        )
    }

    /// As [`Self::generate_trace_inputs_with_external`], but each log may
    /// continue an earlier transcript rather than start at the zero sponge.
    pub fn generate_trace_inputs_with_external_resumed(
        &self,
        logs: &[&TranscriptLog<F, [F; POSEIDON2_WIDTH]>],
        resumes: &[Option<(usize, [F; POSEIDON2_WIDTH])>],
        external_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        external_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_transcript_height: Option<usize>,
    ) -> Option<NativeWarpTranscriptInputArtifacts> {
        self.generate_trace_inputs_with_external_resumed_and_checkpoints(
            logs,
            resumes,
            &[],
            external_permutation_inputs,
            external_compression_inputs,
            required_transcript_height,
        )
    }

    pub fn generate_trace_inputs_with_external_resumed_and_checkpoints(
        &self,
        logs: &[&TranscriptLog<F, [F; POSEIDON2_WIDTH]>],
        resumes: &[Option<(usize, [F; POSEIDON2_WIDTH])>],
        checkpoints: &[Option<[usize; 2]>],
        external_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        external_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_transcript_height: Option<usize>,
    ) -> Option<NativeWarpTranscriptInputArtifacts> {
        let artifacts = self.inner.build_transcript_trace_artifacts(
            logs,
            resumes,
            checkpoints,
            external_permutation_inputs,
            external_compression_inputs,
            required_transcript_height,
        )?;
        Some(NativeWarpTranscriptInputArtifacts {
            trace: artifacts.transcript_trace,
            permutation_inputs: artifacts.poseidon2_perm_inputs,
            compression_inputs: artifacts.poseidon2_compress_inputs,
        })
    }

    /// Variant allowing a resumed verifier to obtain only its terminal
    /// checkpoint. Its start checkpoint is authenticated by the preceding
    /// source verifier and routed directly to the checkpoint bus.
    pub fn generate_trace_inputs_with_external_resumed_and_optional_checkpoints(
        &self,
        logs: &[&TranscriptLog<F, [F; POSEIDON2_WIDTH]>],
        resumes: &[Option<(usize, [F; POSEIDON2_WIDTH])>],
        checkpoints: &[Option<[Option<usize>; 2]>],
        external_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        external_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_transcript_height: Option<usize>,
    ) -> Option<NativeWarpTranscriptInputArtifacts> {
        let artifacts = self
            .inner
            .build_transcript_trace_artifacts_with_optional_checkpoints(
                logs,
                resumes,
                checkpoints,
                external_permutation_inputs,
                external_compression_inputs,
                required_transcript_height,
            )?;
        Some(NativeWarpTranscriptInputArtifacts {
            trace: artifacts.transcript_trace,
            permutation_inputs: artifacts.poseidon2_perm_inputs,
            compression_inputs: artifacts.poseidon2_compress_inputs,
        })
    }

    pub fn build_poseidon2_trace(
        &self,
        permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        self.inner
            .build_poseidon2_trace(permutation_inputs, compression_inputs, required_height)
    }

    pub fn build_poseidon2_multibus_traces(
        &self,
        grouped_inputs: Poseidon2MultibusInputs,
    ) -> Option<Vec<RowMajorMatrix<F>>> {
        self.inner.build_poseidon2_multibus_traces(grouped_inputs)
    }

    pub fn build_poseidon2_multibus_sharded_traces(
        &self,
        grouped_inputs: Poseidon2MultibusInputs,
        shard_count: usize,
        max_rows: usize,
    ) -> Option<Vec<RowMajorMatrix<F>>> {
        self.inner
            .build_poseidon2_multibus_sharded_traces(grouped_inputs, shard_count, max_rows)
    }

    /// As [`Self::build_poseidon2_trace`], but writing straight into device
    /// memory. See [`TranscriptModule::build_poseidon2_trace_gpu`].
    #[cfg(feature = "cuda")]
    pub fn build_poseidon2_trace_gpu(
        &self,
        permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_height: Option<usize>,
        device_ctx: &openvm_cuda_common::stream::GpuDeviceCtx,
    ) -> Option<openvm_cuda_backend::base::DeviceMatrix<F>> {
        self.inner.build_poseidon2_trace_gpu(
            permutation_inputs,
            compression_inputs,
            required_height,
            device_ctx,
        )
    }

    #[cfg(feature = "cuda")]
    pub fn build_poseidon2_multibus_traces_gpu(
        &self,
        grouped_inputs: Poseidon2MultibusInputs,
        device_ctx: &openvm_cuda_common::stream::GpuDeviceCtx,
    ) -> Option<Vec<openvm_cuda_backend::base::DeviceMatrix<F>>> {
        self.inner
            .build_poseidon2_multibus_traces_gpu(grouped_inputs, device_ctx)
    }
}

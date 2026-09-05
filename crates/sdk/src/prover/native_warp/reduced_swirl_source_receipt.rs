//! SDK adapter for the ordered reduced-SWIRL source receipt.
//!
//! The adapter replays each retained prefix through the existing
//! `VerifierSubCircuit<1024>` preflight, derives the same checkpoint and
//! scalar records used by its AIR traces, and differentially checks the new
//! receipt digest against the canonical SDK `digest_with_claim` function.
//! Child WHIR is deliberately absent from every compatibility carrier.

use std::sync::Arc;

use openvm_circuit::arch::{vm_segment_metadata_from_parts, POSEIDON2_WIDTH};
use openvm_continuations::circuit::{
    reduced_swirl_source_receipt::{
        canonicalize_reduced_swirl_source_receipt_block, ReducedSwirlReceiptLayoutEntry,
        ReducedSwirlReceiptVmBoundary, ReducedSwirlSourceReceiptBlock,
        ReducedSwirlSourceReceiptComponent, ReducedSwirlSourceReceiptProfile,
        ReducedSwirlSourceReceiptRecord,
    },
    reduced_swirl_warp::ReducedSwirlSourceReceiptBus,
};
use openvm_cpu_backend::CpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine, GpuBackend, GpuDevice};
#[cfg(feature = "cuda")]
use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_recursion_circuit::{
    native_warp::{
        ReducedSwirlSourceAuthorityBus, ReducedSwirlSourceProfile, ReducedSwirlSourceRecord,
    },
    system::{
        AggregationSubCircuit, BusIndexManager, CachedTraceCtx, DeferredOpeningCheckpointAir,
        DeferredOpeningCheckpointWitness, RetainedStackingProof, VerifierExternalData,
        VerifierSubCircuit, VerifierTraceGen,
    },
};
#[cfg(feature = "cuda")]
use openvm_stark_backend::prover::StridedColMajorMatrixView;
#[cfg(feature = "cuda")]
use openvm_stark_backend::prover::{CommittedTraceData, DeviceDataTransporter};
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey,
    p3_field::PrimeCharacteristicRing,
    prover::{AirProvingContext, MatrixDimensions, PendingConstrainedCodeMetadata, ProverBackend},
    AirRef, FiatShamirTranscript, StarkEngine, StarkProtocolConfig, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, BabyBearPoseidon2Config, BabyBearPoseidon2CpuEngine, Digest,
    DuplexSponge, DIGEST_SIZE, F,
};
use openvm_verify_stark_host::pvs::VkCommit;

use super::{
    reduced_swirl_boundary::{
        AuthoritativeSwirlConstrainedRsClaim, PendingConstrainedCodePublicClaim,
        ReducedSwirlSourceManifestPrefix,
    },
    reduced_swirl_params::reduced_swirl_system_params_digest,
    reduced_swirl_wrapper_components::ReducedSwirlVerifierComponent,
};
use crate::SC;

pub const REDUCED_SWIRL_SOURCE_RECEIPT_CAPACITY: usize = 1024;
/// Maximum number of live deferred-SWIRL prefixes admitted by one combined
/// source/VACC transition leaf.
pub const REDUCED_SWIRL_IN_FLIGHT_SOURCE_CAPACITY: usize = 8;
const REDUCED_SWIRL_SOURCE_TRANSCRIPT_SBOX_REGISTERS: usize = 0;

#[derive(Debug, thiserror::Error)]
pub enum ReducedSwirlSourceReceiptAdapterError {
    #[error("invalid reduced-SWIRL source-receipt profile: {0}")]
    Profile(&'static str),
    #[error("retained source/claim count mismatch: prefixes={prefixes}, claims={claims}")]
    Count { prefixes: usize, claims: usize },
    #[error("ordinary deferred verifier checkpoint context is malformed: {0}")]
    CheckpointContext(&'static str),
    #[error("source {source_index} VM boundary is malformed: {message}")]
    Vm {
        source_index: usize,
        message: String,
    },
    #[error("source {source_index} pending claim is malformed: {message}")]
    Pending {
        source_index: usize,
        message: String,
    },
    #[error("source {source_index} canonical SDK digest differs from receipt AIR")]
    CanonicalDigest { source_index: usize },
    #[error(
        "source range [{source_offset}, {source_offset}+{source_count}) exceeds native source count {native_sources}"
    )]
    SourceRange {
        source_offset: u32,
        source_count: usize,
        native_sources: usize,
    },
    #[error("in-flight source binding count mismatch: sources={sources}, bindings={bindings}")]
    InFlightBindingCount { sources: usize, bindings: usize },
    #[error("in-flight reduced-SWIRL source bindings differ from the derived ordered receipt")]
    InFlightSourceBindings,
    #[error("ordinary deferred verifier trace generation failed")]
    VerifierTrace,
    #[error("reduced-SWIRL source auxiliary trace failed: {0}")]
    Auxiliary(&'static str),
    #[error("reduced-SWIRL source component AIR/context inventory differs")]
    Inventory,
    #[error("reduced-SWIRL source component setup digest failed: {0}")]
    Setup(String),
    #[error("reduced-SWIRL recursive application VK commitment failed: {0}")]
    VkCommit(String),
}

/// CPU proving contexts in component-local AIR coordinates. The transition
/// leaf assigns these contexts to its setup-fixed AIR inventory.
pub struct ReducedSwirlSourceReceiptCpuPacket {
    pub contexts: Vec<(usize, AirProvingContext<CpuBackend<SC>>)>,
    pub block: ReducedSwirlSourceReceiptBlock,
}

/// Device-resident counterpart used by the production CUDA wrapper.  The
/// ordinary verifier tables are generated directly on the target device;
/// only its compact deferred-opening checkpoint is copied back to construct
/// the canonical source receipt.
#[cfg(feature = "cuda")]
pub struct ReducedSwirlSourceReceiptCudaPacket {
    pub contexts: Vec<(usize, AirProvingContext<GpuBackend>)>,
    pub block: ReducedSwirlSourceReceiptBlock,
}

const SOURCE_RECEIPT_COMPONENT_DIGEST_TAG: &[u8] =
    b"openvm.native-warp.reduced-swirl.source-receipt-component.v5";

/// Production source component used by the bounded transition leaf. Its
/// digest is derived from the child VK and actual AIR inventory; callers
/// cannot substitute an opaque setup identifier.
pub struct ProductionReducedSwirlSourceReceiptComponent<
    const MAX_SOURCES: usize = REDUCED_SWIRL_SOURCE_RECEIPT_CAPACITY,
> {
    inner: ReducedSwirlSourceReceiptComponent<MAX_SOURCES>,
    protocol_digest: Digest,
}

impl<const MAX_SOURCES: usize> ProductionReducedSwirlSourceReceiptComponent<MAX_SOURCES> {
    pub fn new(
        child_vk: Arc<MultiStarkVerifyingKey<SC>>,
        profile: ReducedSwirlSourceReceiptProfile,
        wrapper_params: SystemParams,
        receipt_bus: ReducedSwirlSourceReceiptBus,
        authority_bus: ReducedSwirlSourceAuthorityBus,
        bus_idx_manager: BusIndexManager,
    ) -> Result<Self, ReducedSwirlSourceReceiptAdapterError> {
        if receipt_bus.index() == authority_bus.index()
            || receipt_bus.index() >= bus_idx_manager.next_bus_idx()
            || authority_bus.index() >= bus_idx_manager.next_bus_idx()
        {
            return Err(ReducedSwirlSourceReceiptAdapterError::Setup(
                "source receipt buses were not allocated from the supplied manager".to_owned(),
            ));
        }
        let inner = ReducedSwirlSourceReceiptComponent::new(
            Arc::clone(&child_vk),
            profile.clone(),
            wrapper_params,
            receipt_bus,
            authority_bus,
            bus_idx_manager,
        )
        .map_err(ReducedSwirlSourceReceiptAdapterError::Profile)?;
        let protocol_digest = reduced_swirl_source_receipt_component_digest(
            child_vk.as_ref(),
            &profile,
            &inner,
            receipt_bus,
            authority_bus,
        )?;
        if protocol_digest.iter().all(|value| *value == F::ZERO) {
            return Err(ReducedSwirlSourceReceiptAdapterError::Setup(
                "zero source component digest".to_owned(),
            ));
        }
        Ok(Self {
            inner,
            protocol_digest,
        })
    }

    #[must_use]
    pub const fn inner(&self) -> &ReducedSwirlSourceReceiptComponent<MAX_SOURCES> {
        &self.inner
    }

    /// Generate a source packet while the native stream is still in flight
    /// and therefore has no final flat block manifest.
    ///
    /// `total_source_count` is the setup-fixed source count for this proving
    /// run. `local_source_bindings` must be the exact ordered entry digests
    /// emitted by the live native stream for the interval beginning at the
    /// global `source_offset`. Every digest is independently re-derived from
    /// `prefixes` and `claims`; no caller-supplied success bit is accepted.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_cpu_in_flight_packet(
        &self,
        app_vk: &MultiStarkVerifyingKey<SC>,
        prefixes: &[RetainedStackingProof],
        claims: &[AuthoritativeSwirlConstrainedRsClaim],
        total_source_count: usize,
        source_offset: u32,
        local_source_bindings: &[Digest],
        external_poseidon2_compression_inputs: &[[F; POSEIDON2_WIDTH]],
    ) -> Result<ReducedSwirlSourceReceiptCpuPacket, ReducedSwirlSourceReceiptAdapterError> {
        generate_reduced_swirl_source_receipt_cpu_in_flight_packet(
            &self.inner,
            app_vk,
            prefixes,
            claims,
            total_source_count,
            source_offset,
            local_source_bindings,
            external_poseidon2_compression_inputs,
        )
    }

    /// Commit the setup-fixed child VK once for all in-flight source packets.
    ///
    /// The ordinary recursive prover stores this value in its proving key. A
    /// reduced-SWIRL stream has the same fixed child VK, so rebuilding its RS
    /// commitment for every WARP call is duplicate setup work. The returned
    /// device data is immutable and cheaply cloneable (`DeviceMatrix` and PCS
    /// data are reference counted).
    #[cfg(feature = "cuda")]
    pub fn commit_cuda_child_vk(
        &self,
        app_vk: &MultiStarkVerifyingKey<SC>,
        engine: &BabyBearPoseidon2GpuEngine,
    ) -> CommittedTraceData<GpuBackend> {
        <VerifierSubCircuit<
            MAX_SOURCES,
            REDUCED_SWIRL_SOURCE_TRANSCRIPT_SBOX_REGISTERS,
        > as VerifierTraceGen<GpuBackend, SC, GpuDeviceCtx>>::commit_child_vk(
            self.inner.verifier(),
            engine,
            app_vk,
        )
    }

    /// Generate one in-flight packet while reusing the setup-fixed child-VK
    /// commitment. Dynamic proof traces and checkpoints remain per call; only
    /// the immutable cached trace is shared.
    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    pub fn generate_cuda_in_flight_packet_with_cached_vk(
        &self,
        app_vk: &MultiStarkVerifyingKey<SC>,
        prefixes: &[RetainedStackingProof],
        claims: &[AuthoritativeSwirlConstrainedRsClaim],
        total_source_count: usize,
        source_offset: u32,
        local_source_bindings: &[Digest],
        external_poseidon2_compression_inputs: &[[F; POSEIDON2_WIDTH]],
        cached_vk: &CommittedTraceData<GpuBackend>,
        engine: &BabyBearPoseidon2GpuEngine,
    ) -> Result<ReducedSwirlSourceReceiptCudaPacket, ReducedSwirlSourceReceiptAdapterError> {
        generate_reduced_swirl_source_receipt_cuda_packet_impl(
            &self.inner,
            app_vk,
            prefixes,
            claims,
            total_source_count,
            source_offset,
            local_source_bindings,
            external_poseidon2_compression_inputs,
            Some(cached_vk),
            engine,
        )
    }
}

impl<const MAX_SOURCES: usize> ReducedSwirlVerifierComponent
    for ProductionReducedSwirlSourceReceiptComponent<MAX_SOURCES>
{
    fn protocol_digest(&self) -> Digest {
        self.protocol_digest
    }

    fn airs<C: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<C>> {
        self.inner.airs::<C>()
    }
}

/// Recommit the exact child-VK cached trace under the direct wrapper PCS.
///
/// This is the same key-lineage adaptation used by OpenVM's ordinary inner
/// prover.  The trace values and child VK pre-hash remain fixed; only the
/// parent's authenticated PCS envelope changes.  The returned value is used
/// both by the direct wrapper public values and by its recursive normalizer.
pub fn reduced_swirl_recursive_app_vk_commit<const MAX_SOURCES: usize>(
    component: &ProductionReducedSwirlSourceReceiptComponent<MAX_SOURCES>,
    app_vk: &MultiStarkVerifyingKey<SC>,
    wrapper_params: &SystemParams,
) -> Result<VkCommit<F>, ReducedSwirlSourceReceiptAdapterError> {
    if component.inner().params() != wrapper_params {
        return Err(ReducedSwirlSourceReceiptAdapterError::VkCommit(
            "wrapper parameter mismatch".to_owned(),
        ));
    }
    let engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(wrapper_params.clone());
    let committed =
        <VerifierSubCircuit<MAX_SOURCES, REDUCED_SWIRL_SOURCE_TRANSCRIPT_SBOX_REGISTERS> as VerifierTraceGen<
            CpuBackend<SC>,
            SC,
            (),
        >>::commit_child_vk(component.inner().verifier(), &engine, app_vk);
    let cached_commit = committed.commitment;
    if cached_commit.iter().all(|value| *value == F::ZERO) {
        return Err(ReducedSwirlSourceReceiptAdapterError::VkCommit(
            "zero cached commitment".to_owned(),
        ));
    }
    Ok(VkCommit {
        cached_commit,
        vk_pre_hash: app_vk.pre_hash,
    })
}

/// Derive the cached-commit global indices exactly as `ProofShapeAir` does.
pub fn reduced_swirl_source_receipt_profile(
    app_vk: &MultiStarkVerifyingKey<SC>,
    source: ReducedSwirlSourceProfile,
    protocol_digest: Digest,
    suspend_exit_code: u32,
) -> Result<ReducedSwirlSourceReceiptProfile, ReducedSwirlSourceReceiptAdapterError> {
    let mut next_global = 0usize;
    let cached_global_indices = app_vk
        .inner
        .per_air
        .iter()
        .map(|air| {
            (0..air.params.width.cached_mains.len())
                .map(|_| {
                    let index = next_global;
                    next_global += 1;
                    index
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let profile = ReducedSwirlSourceReceiptProfile {
        source,
        protocol_digest,
        child_vk_pre_hash: app_vk.pre_hash,
        child_air_count: app_vk.inner.per_air.len(),
        child_l_skip: app_vk.inner.params.l_skip,
        cached_global_indices,
        suspend_exit_code,
    };
    profile
        .validate()
        .map_err(ReducedSwirlSourceReceiptAdapterError::Profile)?;
    Ok(profile)
}

/// Canonicalize a bounded source interval before the final native
/// statement exists.
///
/// This function deliberately has no block-manifest argument. Its
/// `manifest_digest` is only the canonical digest of this bounded receipt and
/// must not be presented as the final flat block manifest. The authoritative
/// live-stream bindings are checked entry by entry after independently
/// deriving each digest from the retained prefix and constrained-RS claim.
#[allow(clippy::too_many_arguments)]
pub fn build_reduced_swirl_source_receipt_in_flight_block(
    profile: &ReducedSwirlSourceReceiptProfile,
    app_vk: &MultiStarkVerifyingKey<SC>,
    prefixes: &[RetainedStackingProof],
    claims: &[AuthoritativeSwirlConstrainedRsClaim],
    checkpoints: &[DeferredOpeningCheckpointWitness],
    total_source_count: usize,
    source_offset: u32,
    local_source_bindings: &[Digest],
) -> Result<ReducedSwirlSourceReceiptBlock, ReducedSwirlSourceReceiptAdapterError> {
    validate_in_flight_source_binding_range(
        total_source_count,
        source_offset,
        prefixes.len(),
        local_source_bindings,
    )?;
    let block = build_reduced_swirl_source_receipt_bounded_block(
        profile,
        app_vk,
        prefixes,
        claims,
        checkpoints,
        total_source_count,
        source_offset,
    )?;
    let derived_bindings = block
        .sources
        .iter()
        .map(|source| source.entry_digest)
        .collect::<Vec<_>>();
    validate_derived_in_flight_source_bindings(&derived_bindings, local_source_bindings)?;
    Ok(block)
}

fn validate_derived_in_flight_source_bindings(
    derived_bindings: &[Digest],
    local_source_bindings: &[Digest],
) -> Result<(), ReducedSwirlSourceReceiptAdapterError> {
    if derived_bindings.len() != local_source_bindings.len() {
        return Err(
            ReducedSwirlSourceReceiptAdapterError::InFlightBindingCount {
                sources: derived_bindings.len(),
                bindings: local_source_bindings.len(),
            },
        );
    }
    if derived_bindings != local_source_bindings {
        return Err(ReducedSwirlSourceReceiptAdapterError::InFlightSourceBindings);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_reduced_swirl_source_receipt_bounded_block(
    profile: &ReducedSwirlSourceReceiptProfile,
    app_vk: &MultiStarkVerifyingKey<SC>,
    prefixes: &[RetainedStackingProof],
    claims: &[AuthoritativeSwirlConstrainedRsClaim],
    checkpoints: &[DeferredOpeningCheckpointWitness],
    total_source_count: usize,
    source_offset: u32,
) -> Result<ReducedSwirlSourceReceiptBlock, ReducedSwirlSourceReceiptAdapterError> {
    profile
        .validate()
        .map_err(ReducedSwirlSourceReceiptAdapterError::Profile)?;
    if prefixes.is_empty()
        || prefixes.len() > profile.source.maximum_sources
        || prefixes.len() != claims.len()
        || prefixes.len() != checkpoints.len()
    {
        return Err(ReducedSwirlSourceReceiptAdapterError::Count {
            prefixes: prefixes.len(),
            claims: claims.len(),
        });
    }
    let source_start = source_offset as usize;
    let source_end = source_start.checked_add(prefixes.len()).ok_or(
        ReducedSwirlSourceReceiptAdapterError::SourceRange {
            source_offset,
            source_count: prefixes.len(),
            native_sources: total_source_count,
        },
    )?;
    if source_end > total_source_count || u32::try_from(source_end.saturating_sub(1)).is_err() {
        return Err(ReducedSwirlSourceReceiptAdapterError::SourceRange {
            source_offset,
            source_count: prefixes.len(),
            native_sources: total_source_count,
        });
    }
    let mut sources = Vec::with_capacity(prefixes.len());
    for (local_source_index, ((prefix, claim), checkpoint)) in
        prefixes.iter().zip(claims).zip(checkpoints).enumerate()
    {
        let source_index = source_start + local_source_index;
        let checkpoint_tidx = checkpoint.checkpoint.transcript_index;
        let checkpoint_samples = checkpoint.samples;
        let checkpoint_state = checkpoint.checkpoint.transcript_state;
        let metadata =
            vm_segment_metadata_from_parts(app_vk, &prefix.trace_vdata, &prefix.public_values)
                .map_err(|error| ReducedSwirlSourceReceiptAdapterError::Vm {
                    source_index,
                    message: format!("{error:?}"),
                })?;
        let stacking_point = claim.beta.iter().rev().copied().collect::<Vec<_>>();
        let pending_metadata = PendingConstrainedCodeMetadata::try_new(
            claim.root_tuple.clone(),
            claim.commitment_widths.clone(),
            &app_vk.inner.params,
        )
        .map_err(|error| ReducedSwirlSourceReceiptAdapterError::Pending {
            source_index,
            message: error.to_string(),
        })?;
        let pending = PendingConstrainedCodePublicClaim {
            metadata: pending_metadata,
            swirl_tilde_u: stacking_point.clone(),
            stacking_openings: prefix.stacking_proof.stacking_openings.clone(),
        };
        let layout = prefix
            .trace_vdata
            .iter()
            .map(|trace| {
                trace.as_ref().map_or(
                    ReducedSwirlReceiptLayoutEntry {
                        log_height: None,
                        cached_commitments: Vec::new(),
                    },
                    |trace| ReducedSwirlReceiptLayoutEntry {
                        log_height: Some(trace.log_height),
                        cached_commitments: trace.cached_commitments.clone(),
                    },
                )
            })
            .collect::<Vec<_>>();
        sources.push(ReducedSwirlSourceReceiptRecord {
            segment_index: u32::try_from(source_index).map_err(|_| {
                ReducedSwirlSourceReceiptAdapterError::SourceRange {
                    source_offset,
                    source_count: prefixes.len(),
                    native_sources: total_source_count,
                }
            })?,
            checkpoint_tidx,
            checkpoint_samples,
            checkpoint_state,
            layout,
            roots: claim.root_tuple.clone(),
            widths: claim.commitment_widths.clone(),
            stacking_point,
            stacking_openings: prefix.stacking_proof.stacking_openings.clone(),
            theta: claim.theta,
            mu: claim.mu,
            beta: claim.beta.clone(),
            eta: claim.eta,
            vm: ReducedSwirlReceiptVmBoundary {
                program_commitment: metadata.program_commit,
                initial_pc: metadata.initial_pc,
                initial_root: metadata.initial_memory_root,
                final_pc: metadata.final_pc,
                final_root: metadata.final_memory_root,
                exit_code: metadata.exit_code,
                is_terminate: metadata.is_terminate,
            },
            layout_digest: [F::ZERO; DIGEST_SIZE],
            pending_digest: pending.digest().map_err(|error| {
                ReducedSwirlSourceReceiptAdapterError::Pending {
                    source_index,
                    message: error.to_string(),
                }
            })?,
            claim_digest: claim.digest().map_err(|error| {
                ReducedSwirlSourceReceiptAdapterError::Pending {
                    source_index,
                    message: error.to_string(),
                }
            })?,
            entry_digest: [F::ZERO; DIGEST_SIZE],
        });
    }
    let mut block = ReducedSwirlSourceReceiptBlock {
        source_offset,
        sources,
        manifest_digest: [F::ZERO; DIGEST_SIZE],
    };
    canonicalize_reduced_swirl_source_receipt_block(profile, &mut block)
        .map_err(ReducedSwirlSourceReceiptAdapterError::Profile)?;

    // Differential anchor against the SDK functions named by the protocol.
    for (local_source_index, ((prefix, claim), receipt)) in
        prefixes.iter().zip(claims).zip(&block.sources).enumerate()
    {
        let source_index = source_start + local_source_index;
        let sdk_prefix = ReducedSwirlSourceManifestPrefix {
            source_index: u32::try_from(source_index).map_err(|_| {
                ReducedSwirlSourceReceiptAdapterError::SourceRange {
                    source_offset,
                    source_count: prefixes.len(),
                    native_sources: total_source_count,
                }
            })?,
            segment_index: receipt.segment_index,
            common_main_root: prefix.common_main_commit,
            trace_layout_digest: receipt.layout_digest,
            pending_claim_digest: receipt.pending_digest,
            vm_pvs: super::reduced_swirl_boundary::ReducedSwirlVmBoundary {
                program_commitment: receipt.vm.program_commitment,
                initial_state: super::reduced_swirl_boundary::ReducedSwirlVmState {
                    pc: receipt.vm.initial_pc,
                    memory_root: receipt.vm.initial_root,
                },
                final_state: super::reduced_swirl_boundary::ReducedSwirlVmState {
                    pc: receipt.vm.final_pc,
                    memory_root: receipt.vm.final_root,
                },
                exit_code: receipt.vm.exit_code,
                is_terminate: receipt.vm.is_terminate,
            },
        };
        if receipt.roots.first().copied() != Some(prefix.common_main_commit)
            || sdk_prefix.digest_with_claim(claim).map_err(|_| {
                ReducedSwirlSourceReceiptAdapterError::CanonicalDigest { source_index }
            })? != receipt.entry_digest
        {
            return Err(ReducedSwirlSourceReceiptAdapterError::CanonicalDigest { source_index });
        }
    }
    Ok(block)
}

fn validate_in_flight_source_binding_range(
    total_source_count: usize,
    source_offset: u32,
    source_count: usize,
    local_source_bindings: &[Digest],
) -> Result<(), ReducedSwirlSourceReceiptAdapterError> {
    if source_count == 0 || source_count > REDUCED_SWIRL_IN_FLIGHT_SOURCE_CAPACITY {
        return Err(ReducedSwirlSourceReceiptAdapterError::SourceRange {
            source_offset,
            source_count,
            native_sources: total_source_count,
        });
    }
    if source_count != local_source_bindings.len() {
        return Err(
            ReducedSwirlSourceReceiptAdapterError::InFlightBindingCount {
                sources: source_count,
                bindings: local_source_bindings.len(),
            },
        );
    }
    let source_start = source_offset as usize;
    let source_end = source_start.checked_add(source_count).ok_or(
        ReducedSwirlSourceReceiptAdapterError::SourceRange {
            source_offset,
            source_count,
            native_sources: total_source_count,
        },
    )?;
    if total_source_count == 0
        || total_source_count > u32::MAX as usize
        || source_end > total_source_count
        || u32::try_from(source_end.saturating_sub(1)).is_err()
    {
        return Err(ReducedSwirlSourceReceiptAdapterError::SourceRange {
            source_offset,
            source_count,
            native_sources: total_source_count,
        });
    }
    Ok(())
}

/// Generate one bounded source/VACC leaf packet before the final native
/// statement and flat block manifest exist.
///
/// Exactly `1..=8` retained prefixes are admitted. `local_source_bindings`
/// comes from the live native stream and is compared against independently
/// derived canonical entry digests in global source order.
#[allow(clippy::too_many_arguments)]
pub fn generate_reduced_swirl_source_receipt_cpu_in_flight_packet<const MAX_SOURCES: usize>(
    component: &ReducedSwirlSourceReceiptComponent<MAX_SOURCES>,
    app_vk: &MultiStarkVerifyingKey<SC>,
    prefixes: &[RetainedStackingProof],
    claims: &[AuthoritativeSwirlConstrainedRsClaim],
    total_source_count: usize,
    source_offset: u32,
    local_source_bindings: &[Digest],
    external_poseidon2_compression_inputs: &[[F; POSEIDON2_WIDTH]],
) -> Result<ReducedSwirlSourceReceiptCpuPacket, ReducedSwirlSourceReceiptAdapterError> {
    generate_reduced_swirl_source_receipt_cpu_packet_impl(
        component,
        app_vk,
        prefixes,
        claims,
        total_source_count,
        source_offset,
        local_source_bindings,
        external_poseidon2_compression_inputs,
    )
}

#[allow(clippy::too_many_arguments)]
fn generate_reduced_swirl_source_receipt_cpu_packet_impl<const MAX_SOURCES: usize>(
    component: &ReducedSwirlSourceReceiptComponent<MAX_SOURCES>,
    app_vk: &MultiStarkVerifyingKey<SC>,
    prefixes: &[RetainedStackingProof],
    claims: &[AuthoritativeSwirlConstrainedRsClaim],
    total_source_count: usize,
    source_offset: u32,
    local_source_bindings: &[Digest],
    external_poseidon2_compression_inputs: &[[F; POSEIDON2_WIDTH]],
) -> Result<ReducedSwirlSourceReceiptCpuPacket, ReducedSwirlSourceReceiptAdapterError> {
    if component.receipt_air().profile.source.maximum_sources != MAX_SOURCES {
        return Err(ReducedSwirlSourceReceiptAdapterError::Profile(
            "reduced-SWIRL source component capacity mismatch",
        ));
    }
    if prefixes.is_empty()
        || prefixes.len() > MAX_SOURCES
        || prefixes.len() > REDUCED_SWIRL_IN_FLIGHT_SOURCE_CAPACITY
        || prefixes.len() != claims.len()
    {
        return Err(ReducedSwirlSourceReceiptAdapterError::Count {
            prefixes: prefixes.len(),
            claims: claims.len(),
        });
    }
    let proofs = prefixes
        .iter()
        .map(|prefix| prefix.with_deferred_whir_input(Clone::clone))
        .collect::<Vec<_>>();
    let empty_poseidon = Vec::<[F; POSEIDON2_WIDTH]>::new();
    let empty_usize = Vec::<usize>::new();
    let mut external_data = VerifierExternalData {
        poseidon2_compress_inputs: &empty_poseidon,
        poseidon2_permute_inputs: &empty_poseidon,
        range_check_inputs: &empty_usize,
        power_check_inputs: &empty_usize,
        required_heights: None,
        final_transcript_state: None,
    };
    let engine = BabyBearPoseidon2CpuEngine::<DuplexSponge>::new(component.params().clone());
    let cached_trace = <VerifierSubCircuit<
        MAX_SOURCES,
        REDUCED_SWIRL_SOURCE_TRANSCRIPT_SBOX_REGISTERS,
    > as VerifierTraceGen<CpuBackend<SC>, SC, ()>>::commit_child_vk(
        component.verifier(),
        &engine,
        app_vk,
    );
    let mut verifier_contexts = <VerifierSubCircuit<
        MAX_SOURCES,
        REDUCED_SWIRL_SOURCE_TRANSCRIPT_SBOX_REGISTERS,
    > as VerifierTraceGen<CpuBackend<SC>, SC, ()>>::generate_proving_ctxs(
        component.verifier(),
        app_vk,
        CachedTraceCtx::PcsData(cached_trace),
        &proofs,
        &mut external_data,
        &(),
        default_duplex_sponge_recorder(),
    )
    .ok_or(ReducedSwirlSourceReceiptAdapterError::VerifierTrace)?;
    let ordinary_air_count = component.verifier().airs::<SC>().len();
    if verifier_contexts.len() != ordinary_air_count
        || component.checkpoint_air_index() >= verifier_contexts.len()
    {
        return Err(ReducedSwirlSourceReceiptAdapterError::Inventory);
    }
    let checkpoint_context = &verifier_contexts[component.checkpoint_air_index()];
    let checkpoints =
        DeferredOpeningCheckpointAir::<MAX_SOURCES>::decode_trace(&checkpoint_context.common_main)
            .map_err(ReducedSwirlSourceReceiptAdapterError::CheckpointContext)?;
    let public_checkpoints = DeferredOpeningCheckpointAir::<MAX_SOURCES>::decode_public_values(
        &checkpoint_context.public_values,
    )
    .map_err(ReducedSwirlSourceReceiptAdapterError::CheckpointContext)?;
    if checkpoints.len() != prefixes.len()
        || public_checkpoints.len() != checkpoints.len()
        || checkpoints
            .iter()
            .zip(&public_checkpoints)
            .any(|(private, public)| private.checkpoint != *public)
    {
        return Err(ReducedSwirlSourceReceiptAdapterError::Inventory);
    }
    verifier_contexts.remove(component.checkpoint_air_index());

    let (block, auxiliary_contexts) = build_source_receipt_auxiliary_cpu_contexts(
        component,
        app_vk,
        prefixes,
        claims,
        &checkpoints,
        total_source_count,
        source_offset,
        local_source_bindings,
        external_poseidon2_compression_inputs,
    )?;
    let mut contexts = verifier_contexts
        .into_iter()
        .enumerate()
        .collect::<Vec<_>>();
    let auxiliary_offset = contexts.len();
    contexts.extend(
        auxiliary_contexts
            .into_iter()
            .enumerate()
            .map(|(index, context)| (auxiliary_offset + index, context)),
    );
    if contexts.len() != component.airs::<SC>().len() {
        return Err(ReducedSwirlSourceReceiptAdapterError::Inventory);
    }
    validate_component_contexts(component, &contexts)?;
    Ok(ReducedSwirlSourceReceiptCpuPacket { contexts, block })
}

#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn generate_reduced_swirl_source_receipt_cuda_packet_impl<const MAX_SOURCES: usize>(
    component: &ReducedSwirlSourceReceiptComponent<MAX_SOURCES>,
    app_vk: &MultiStarkVerifyingKey<SC>,
    prefixes: &[RetainedStackingProof],
    claims: &[AuthoritativeSwirlConstrainedRsClaim],
    total_source_count: usize,
    source_offset: u32,
    local_source_bindings: &[Digest],
    external_poseidon2_compression_inputs: &[[F; POSEIDON2_WIDTH]],
    cached_vk: Option<&CommittedTraceData<GpuBackend>>,
    engine: &BabyBearPoseidon2GpuEngine,
) -> Result<ReducedSwirlSourceReceiptCudaPacket, ReducedSwirlSourceReceiptAdapterError> {
    if component.receipt_air().profile.source.maximum_sources != MAX_SOURCES {
        return Err(ReducedSwirlSourceReceiptAdapterError::Profile(
            "reduced-SWIRL source component capacity mismatch",
        ));
    }
    if prefixes.is_empty()
        || prefixes.len() > MAX_SOURCES
        || prefixes.len() > REDUCED_SWIRL_IN_FLIGHT_SOURCE_CAPACITY
        || prefixes.len() != claims.len()
    {
        return Err(ReducedSwirlSourceReceiptAdapterError::Count {
            prefixes: prefixes.len(),
            claims: claims.len(),
        });
    }
    let proofs = prefixes
        .iter()
        .map(|prefix| prefix.with_deferred_whir_input(Clone::clone))
        .collect::<Vec<_>>();
    let empty_poseidon = Vec::<[F; POSEIDON2_WIDTH]>::new();
    let empty_usize = Vec::<usize>::new();
    let mut external_data = VerifierExternalData {
        poseidon2_compress_inputs: &empty_poseidon,
        poseidon2_permute_inputs: &empty_poseidon,
        range_check_inputs: &empty_usize,
        power_check_inputs: &empty_usize,
        required_heights: None,
        final_transcript_state: None,
    };
    let cached_trace = cached_vk.cloned().unwrap_or_else(|| {
        <VerifierSubCircuit<
            MAX_SOURCES,
            REDUCED_SWIRL_SOURCE_TRANSCRIPT_SBOX_REGISTERS,
        > as VerifierTraceGen<GpuBackend, SC, GpuDeviceCtx>>::commit_child_vk(
            component.verifier(),
            engine,
            app_vk,
        )
    });
    let mut verifier_contexts = <VerifierSubCircuit<
        MAX_SOURCES,
        REDUCED_SWIRL_SOURCE_TRANSCRIPT_SBOX_REGISTERS,
    > as VerifierTraceGen<GpuBackend, SC, GpuDeviceCtx>>::generate_proving_ctxs(
        component.verifier(),
        app_vk,
        CachedTraceCtx::PcsData(cached_trace),
        &proofs,
        &mut external_data,
        &engine.device().device_ctx,
        default_duplex_sponge_recorder(),
    )
    .ok_or(ReducedSwirlSourceReceiptAdapterError::VerifierTrace)?;
    let ordinary_air_count = component.verifier().airs::<SC>().len();
    if verifier_contexts.len() != ordinary_air_count
        || component.checkpoint_air_index() >= verifier_contexts.len()
    {
        return Err(ReducedSwirlSourceReceiptAdapterError::Inventory);
    }
    let checkpoint_context = &verifier_contexts[component.checkpoint_air_index()];
    let checkpoint_host =
        <GpuDevice as DeviceDataTransporter<SC, GpuBackend>>::transport_matrix_from_device_to_host(
            engine.device(),
            &checkpoint_context.common_main,
        );
    let checkpoint_host =
        StridedColMajorMatrixView::from(checkpoint_host.as_view()).to_row_major_matrix();
    let checkpoints = DeferredOpeningCheckpointAir::<MAX_SOURCES>::decode_trace(&checkpoint_host)
        .map_err(ReducedSwirlSourceReceiptAdapterError::CheckpointContext)?;
    let public_checkpoints = DeferredOpeningCheckpointAir::<MAX_SOURCES>::decode_public_values(
        &checkpoint_context.public_values,
    )
    .map_err(ReducedSwirlSourceReceiptAdapterError::CheckpointContext)?;
    if checkpoints.len() != prefixes.len()
        || public_checkpoints.len() != checkpoints.len()
        || checkpoints
            .iter()
            .zip(&public_checkpoints)
            .any(|(private, public)| private.checkpoint != *public)
    {
        return Err(ReducedSwirlSourceReceiptAdapterError::Inventory);
    }
    verifier_contexts.remove(component.checkpoint_air_index());

    let (block, auxiliary_contexts) = build_source_receipt_auxiliary_cpu_contexts(
        component,
        app_vk,
        prefixes,
        claims,
        &checkpoints,
        total_source_count,
        source_offset,
        local_source_bindings,
        external_poseidon2_compression_inputs,
    )?;
    let auxiliary_contexts = auxiliary_contexts
        .into_iter()
        .map(|context| {
            let common_main = <GpuDevice as DeviceDataTransporter<SC, GpuBackend>>::
                transport_row_major_matrix_to_device(engine.device(), &context.common_main);
            AirProvingContext::new(Vec::new(), common_main, context.public_values)
        })
        .collect::<Vec<_>>();
    let mut contexts = verifier_contexts
        .into_iter()
        .enumerate()
        .collect::<Vec<_>>();
    let auxiliary_offset = contexts.len();
    contexts.extend(
        auxiliary_contexts
            .into_iter()
            .enumerate()
            .map(|(index, context)| (auxiliary_offset + index, context)),
    );
    if contexts.len() != component.airs::<SC>().len() {
        return Err(ReducedSwirlSourceReceiptAdapterError::Inventory);
    }
    validate_component_contexts(component, &contexts)?;
    Ok(ReducedSwirlSourceReceiptCudaPacket { contexts, block })
}

fn build_source_receipt_auxiliary_cpu_contexts<const MAX_SOURCES: usize>(
    component: &ReducedSwirlSourceReceiptComponent<MAX_SOURCES>,
    app_vk: &MultiStarkVerifyingKey<SC>,
    prefixes: &[RetainedStackingProof],
    claims: &[AuthoritativeSwirlConstrainedRsClaim],
    checkpoints: &[DeferredOpeningCheckpointWitness],
    total_source_count: usize,
    source_offset: u32,
    local_source_bindings: &[Digest],
    external_poseidon2_compression_inputs: &[[F; POSEIDON2_WIDTH]],
) -> Result<
    (
        ReducedSwirlSourceReceiptBlock,
        Vec<AirProvingContext<CpuBackend<SC>>>,
    ),
    ReducedSwirlSourceReceiptAdapterError,
> {
    let block = build_reduced_swirl_source_receipt_in_flight_block(
        &component.receipt_air().profile,
        app_vk,
        prefixes,
        claims,
        checkpoints,
        total_source_count,
        source_offset,
        local_source_bindings,
    )?;
    let source_records = prefixes
        .iter()
        .zip(claims)
        .map(|(prefix, claim)| ReducedSwirlSourceRecord {
            roots: claim.root_tuple.clone(),
            widths: claim.commitment_widths.clone(),
            stacking_point: claim.beta.iter().rev().copied().collect(),
            stacking_openings: prefix.stacking_proof.stacking_openings.clone(),
            mu: claim.mu,
        })
        .collect::<Vec<_>>();
    let auxiliary = component
        .generate_auxiliary_traces_with_external_compressions(
            &source_records,
            &block,
            external_poseidon2_compression_inputs,
        )
        .map_err(ReducedSwirlSourceReceiptAdapterError::Auxiliary)?;
    let auxiliary_contexts = vec![
        AirProvingContext::simple_no_pis(auxiliary.source),
        AirProvingContext::simple_no_pis(auxiliary.source_transcript.trace),
        AirProvingContext::simple_no_pis(auxiliary.source_transcript.poseidon2_trace),
        AirProvingContext::simple_no_pis(auxiliary.receipt.receipt),
        AirProvingContext::simple_no_pis(auxiliary.receipt.transcript.trace),
        AirProvingContext::simple_no_pis(auxiliary.receipt.transcript.poseidon2_trace),
    ];
    if auxiliary_contexts.len() != component.auxiliary_airs::<SC>().len() {
        return Err(ReducedSwirlSourceReceiptAdapterError::Inventory);
    }
    Ok((block, auxiliary_contexts))
}

fn validate_component_contexts<const MAX_SOURCES: usize, PB: ProverBackend<Val = F>>(
    component: &ReducedSwirlSourceReceiptComponent<MAX_SOURCES>,
    contexts: &[(usize, AirProvingContext<PB>)],
) -> Result<(), ReducedSwirlSourceReceiptAdapterError> {
    let airs = component.airs::<SC>();
    if contexts.len() != airs.len() {
        return Err(ReducedSwirlSourceReceiptAdapterError::Inventory);
    }
    for (expected_index, ((actual_index, context), air)) in contexts.iter().zip(&airs).enumerate() {
        if *actual_index != expected_index
            || !context.public_values.is_empty()
            || context.common_main.width() != air.common_main_width()
            || context.common_main.height() == 0
            || !context.common_main.height().is_power_of_two()
            || context.cached_mains.len() != air.cached_main_widths().len()
            || context
                .cached_mains
                .iter()
                .zip(air.cached_main_widths())
                .any(|(matrix, width)| matrix.trace().width() != width)
        {
            return Err(ReducedSwirlSourceReceiptAdapterError::Inventory);
        }
    }
    Ok(())
}

fn reduced_swirl_source_receipt_component_digest<const MAX_SOURCES: usize>(
    child_vk: &MultiStarkVerifyingKey<SC>,
    profile: &ReducedSwirlSourceReceiptProfile,
    component: &ReducedSwirlSourceReceiptComponent<MAX_SOURCES>,
    receipt_bus: ReducedSwirlSourceReceiptBus,
    authority_bus: ReducedSwirlSourceAuthorityBus,
) -> Result<Digest, ReducedSwirlSourceReceiptAdapterError> {
    let config = SC::default_from_params(child_vk.inner.params.clone());
    let params_digest = reduced_swirl_system_params_digest(&config, &child_vk.inner.params)
        .map_err(|error| ReducedSwirlSourceReceiptAdapterError::Setup(error.to_string()))?;
    let mut transcript = default_duplex_sponge_recorder();
    observe_component_bytes(&mut transcript, SOURCE_RECEIPT_COMPONENT_DIGEST_TAG);
    observe_component_digest(&mut transcript, child_vk.pre_hash);
    observe_component_digest(&mut transcript, params_digest);
    observe_component_digest(&mut transcript, profile.protocol_digest);
    observe_component_digest(&mut transcript, profile.child_vk_pre_hash);
    // Preserve the original inline-mode setup tag in the key digest while
    // making that mode the sole constructible architecture.
    observe_component_usize(&mut transcript, 0)?;
    observe_component_usize(
        &mut transcript,
        component.source_air().export_lookup_count as usize,
    )?;
    for value in [
        profile.source.maximum_sources,
        profile.source.maximum_roots_per_source,
        profile.source.maximum_openings_per_source,
        profile.source.l_skip,
        profile.source.n_stack,
        profile.source.log_blowup,
        profile.source.log_commit_rows_per_query,
        profile.child_air_count,
        profile.child_l_skip,
        profile.suspend_exit_code as usize,
    ] {
        observe_component_usize(&mut transcript, value)?;
    }
    observe_component_usize(&mut transcript, profile.cached_global_indices.len())?;
    for (air, indices) in profile.cached_global_indices.iter().enumerate() {
        observe_component_usize(&mut transcript, air)?;
        observe_component_usize(&mut transcript, indices.len())?;
        for &index in indices {
            observe_component_usize(&mut transcript, index)?;
        }
    }

    let airs = component.airs::<BabyBearPoseidon2Config>();
    observe_component_usize(&mut transcript, airs.len())?;
    for (air_index, air) in airs.iter().enumerate() {
        observe_component_usize(&mut transcript, air_index)?;
        observe_component_usize(&mut transcript, air.common_main_width())?;
        observe_component_usize(&mut transcript, air.num_public_values())?;
        observe_component_usize(&mut transcript, air.cached_main_widths().len())?;
        for width in air.cached_main_widths() {
            observe_component_usize(&mut transcript, width)?;
        }
    }

    let source = component.source_air();
    let verifier = component.verifier().bus_inventory();
    for bus in [
        receipt_bus.index(),
        authority_bus.index(),
        source.transcript_bus.index(),
        source.commitments_bus.index(),
        source.root_width_bus.index(),
        verifier.transcript_bus.index(),
        verifier.air_shape_bus.index(),
        verifier.air_presence_bus.index(),
        verifier.hyperdim_bus.index(),
        verifier.cached_commit_bus.index(),
        verifier.public_values_bus.index(),
        verifier.final_state_bus.index(),
        verifier.transcript_end_index_bus.index(),
        component.next_bus_idx(),
    ] {
        observe_component_u64(&mut transcript, u64::from(bus));
    }
    Ok(core::array::from_fn(|_| {
        <_ as FiatShamirTranscript<BabyBearPoseidon2Config>>::sample(&mut transcript)
    }))
}

fn observe_component_digest(
    transcript: &mut impl FiatShamirTranscript<BabyBearPoseidon2Config>,
    digest: Digest,
) {
    for value in digest {
        transcript.observe(value);
    }
}

fn observe_component_bytes(
    transcript: &mut impl FiatShamirTranscript<BabyBearPoseidon2Config>,
    bytes: &[u8],
) {
    observe_component_u64(transcript, bytes.len() as u64);
    for &byte in bytes {
        transcript.observe(F::from_u8(byte));
    }
}

fn observe_component_usize(
    transcript: &mut impl FiatShamirTranscript<BabyBearPoseidon2Config>,
    value: usize,
) -> Result<(), ReducedSwirlSourceReceiptAdapterError> {
    let value = u64::try_from(value).map_err(|_| {
        ReducedSwirlSourceReceiptAdapterError::Setup("source component integer encoding".to_owned())
    })?;
    observe_component_u64(transcript, value);
    Ok(())
}

fn observe_component_u64(
    transcript: &mut impl FiatShamirTranscript<BabyBearPoseidon2Config>,
    value: u64,
) {
    for byte in value.to_le_bytes() {
        transcript.observe(F::from_u8(byte));
    }
}

#[cfg(test)]
mod tests {
    use openvm_circuit::arch::{PROGRAM_AIR_ID, PROGRAM_CACHED_TRACE_INDEX};
    use openvm_stark_sdk::config::baby_bear_poseidon2::EF;

    use super::*;
    use crate::prover::native_warp::{
        reduced_swirl_boundary::{ReducedSwirlVmBoundary, ReducedSwirlVmState},
        reduced_swirl_native::reduced_swirl_manifest_digest,
    };

    fn digest(seed: u32) -> Digest {
        core::array::from_fn(|limb| F::from_u32(seed + limb as u32 + 1))
    }

    #[test]
    fn canonical_entry_and_manifest_match_production_sdk_functions() {
        let program = digest(10);
        let root = digest(30);
        let initial_root = digest(50);
        let final_root = digest(70);
        let point = vec![EF::from_u32(5), EF::from_u32(7)];
        let claim = AuthoritativeSwirlConstrainedRsClaim {
            root_tuple: vec![root],
            commitment_widths: vec![1],
            l_skip: 1,
            n_stack: 1,
            log_blowup: 1,
            log_commit_rows_per_query: 0,
            theta: EF::from_u32(11),
            alpha: vec![EF::ZERO; 3],
            mu: EF::from_u32(13),
            beta: point.iter().rev().copied().collect(),
            eta: EF::from_u32(17),
        };
        let profile = ReducedSwirlSourceReceiptProfile {
            source: ReducedSwirlSourceProfile {
                maximum_sources: REDUCED_SWIRL_SOURCE_RECEIPT_CAPACITY,
                maximum_roots_per_source: 1,
                maximum_openings_per_source: 1,
                l_skip: 1,
                n_stack: 1,
                log_blowup: 1,
                log_commit_rows_per_query: 0,
            },
            protocol_digest: digest(90),
            child_vk_pre_hash: digest(110),
            child_air_count: 1,
            child_l_skip: 1,
            cached_global_indices: vec![vec![0]],
            suspend_exit_code: 2,
        };
        assert_eq!(PROGRAM_AIR_ID, 0);
        assert_eq!(PROGRAM_CACHED_TRACE_INDEX, 0);
        let vm = ReducedSwirlReceiptVmBoundary {
            program_commitment: program,
            initial_pc: F::from_u32(3),
            initial_root,
            final_pc: F::from_u32(9),
            final_root,
            exit_code: F::ZERO,
            is_terminate: F::ONE,
        };
        let mut block = ReducedSwirlSourceReceiptBlock {
            source_offset: 0,
            sources: vec![ReducedSwirlSourceReceiptRecord {
                segment_index: 0,
                checkpoint_tidx: 64,
                checkpoint_samples: EF::from_u32(19),
                checkpoint_state: core::array::from_fn(|limb| F::from_usize(100 + limb)),
                layout: vec![ReducedSwirlReceiptLayoutEntry {
                    log_height: Some(4),
                    cached_commitments: vec![program],
                }],
                roots: claim.root_tuple.clone(),
                widths: claim.commitment_widths.clone(),
                stacking_point: point,
                stacking_openings: vec![vec![EF::from_u32(23)]],
                theta: claim.theta,
                mu: claim.mu,
                beta: claim.beta.clone(),
                eta: claim.eta,
                vm,
                layout_digest: [F::ZERO; DIGEST_SIZE],
                pending_digest: [F::ZERO; DIGEST_SIZE],
                claim_digest: [F::ZERO; DIGEST_SIZE],
                entry_digest: [F::ZERO; DIGEST_SIZE],
            }],
            manifest_digest: [F::ZERO; DIGEST_SIZE],
        };
        canonicalize_reduced_swirl_source_receipt_block(&profile, &mut block).unwrap();
        let source = &block.sources[0];
        let sdk_prefix = ReducedSwirlSourceManifestPrefix {
            source_index: 0,
            segment_index: 0,
            common_main_root: root,
            trace_layout_digest: source.layout_digest,
            pending_claim_digest: source.pending_digest,
            vm_pvs: ReducedSwirlVmBoundary {
                program_commitment: program,
                initial_state: ReducedSwirlVmState {
                    pc: vm.initial_pc,
                    memory_root: vm.initial_root,
                },
                final_state: ReducedSwirlVmState {
                    pc: vm.final_pc,
                    memory_root: vm.final_root,
                },
                exit_code: vm.exit_code,
                is_terminate: vm.is_terminate,
            },
        };
        assert_eq!(claim.digest().unwrap(), source.claim_digest);
        assert_eq!(
            sdk_prefix.digest_with_claim(&claim).unwrap(),
            source.entry_digest
        );
        assert_eq!(
            reduced_swirl_manifest_digest(&[source.entry_digest]).unwrap(),
            block.manifest_digest
        );

        // A bounded nonterminal transition leaf uses global source indices
        // while its one-entry manifest remains local to the leaf interval.
        let mut chunk = block.clone();
        chunk.source_offset = 7;
        chunk.sources[0].segment_index = 7;
        chunk.sources[0].vm.exit_code = F::from_u32(profile.suspend_exit_code);
        chunk.sources[0].vm.is_terminate = F::ZERO;
        canonicalize_reduced_swirl_source_receipt_block(&profile, &mut chunk).unwrap();
        let chunk_source = &chunk.sources[0];
        let chunk_prefix = ReducedSwirlSourceManifestPrefix {
            source_index: 7,
            segment_index: 7,
            common_main_root: root,
            trace_layout_digest: chunk_source.layout_digest,
            pending_claim_digest: chunk_source.pending_digest,
            vm_pvs: ReducedSwirlVmBoundary {
                program_commitment: program,
                initial_state: ReducedSwirlVmState {
                    pc: chunk_source.vm.initial_pc,
                    memory_root: chunk_source.vm.initial_root,
                },
                final_state: ReducedSwirlVmState {
                    pc: chunk_source.vm.final_pc,
                    memory_root: chunk_source.vm.final_root,
                },
                exit_code: chunk_source.vm.exit_code,
                is_terminate: chunk_source.vm.is_terminate,
            },
        };
        assert_eq!(
            chunk_prefix.digest_with_claim(&claim).unwrap(),
            chunk_source.entry_digest
        );

        let mut mutated_claim = claim;
        mutated_claim.mu += EF::ONE;
        assert_ne!(
            sdk_prefix.digest_with_claim(&mutated_claim).unwrap(),
            source.entry_digest
        );
    }

    #[test]
    fn in_flight_source_ranges_are_bounded_and_mutation_sensitive() {
        let bindings = (0..REDUCED_SWIRL_IN_FLIGHT_SOURCE_CAPACITY)
            .map(|index| digest(100 + 10 * index as u32))
            .collect::<Vec<_>>();

        validate_in_flight_source_binding_range(100, 92, bindings.len(), &bindings).unwrap();
        validate_derived_in_flight_source_bindings(&bindings, &bindings).unwrap();

        let mut mutated = bindings.clone();
        mutated[3][2] += F::ONE;
        assert!(matches!(
            validate_derived_in_flight_source_bindings(&bindings, &mutated),
            Err(ReducedSwirlSourceReceiptAdapterError::InFlightSourceBindings)
        ));

        let mut reordered = bindings.clone();
        reordered.swap(1, 2);
        assert!(matches!(
            validate_derived_in_flight_source_bindings(&bindings, &reordered),
            Err(ReducedSwirlSourceReceiptAdapterError::InFlightSourceBindings)
        ));

        let nine = vec![digest(1); REDUCED_SWIRL_IN_FLIGHT_SOURCE_CAPACITY + 1];
        assert!(matches!(
            validate_in_flight_source_binding_range(100, 0, nine.len(), &nine),
            Err(ReducedSwirlSourceReceiptAdapterError::SourceRange { .. })
        ));
        assert!(matches!(
            validate_in_flight_source_binding_range(100, 93, bindings.len(), &bindings),
            Err(ReducedSwirlSourceReceiptAdapterError::SourceRange { .. })
        ));
        assert!(matches!(
            validate_in_flight_source_binding_range(100, 92, 7, &bindings),
            Err(ReducedSwirlSourceReceiptAdapterError::InFlightBindingCount { .. })
        ));
        assert!(matches!(
            validate_in_flight_source_binding_range(0, 0, 1, &bindings[..1]),
            Err(ReducedSwirlSourceReceiptAdapterError::SourceRange { .. })
        ));
        assert!(matches!(
            validate_in_flight_source_binding_range(u32::MAX as usize + 1, 0, 1, &bindings[..1],),
            Err(ReducedSwirlSourceReceiptAdapterError::SourceRange { .. })
        ));
    }
}

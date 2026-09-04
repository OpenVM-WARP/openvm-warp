//! Bounded CUDA prover for recursively composable reduced-SWIRL source leaves.
//!
//! This module is intentionally unable to construct the former 1024-source
//! wrapper.  One leaf consumes at most `CAPACITY` deferred-prefix records,
//! proves them, releases their device traces, and retains only a normal
//! OpenVM proof plus the rolling manifest-chain endpoint.

use core::mem::size_of;
use std::{borrow::Borrow, sync::Arc, time::Instant};

use openvm_continuations::{
    circuit::{
        inner::ReducedSwirlWrapperPrefixTraceGen,
        reduced_swirl_recursive_prefix::{
            ReducedSwirlWrapperPrefixSubCircuit, REDUCED_SWIRL_RECURSIVE_PREFIX_ARITY,
        },
        reduced_swirl_source_leaf::{
            generate_reduced_swirl_source_leaf_core_traces, ReducedSwirlSourceLeafBinding,
            ReducedSwirlSourceLeafCircuit, ReducedSwirlSourceLeafComponents,
            ReducedSwirlSourceLeafRecord, REDUCED_SWIRL_SOURCE_LEAF_CAPACITY,
        },
        reduced_swirl_source_receipt::{
            ReducedSwirlSourceReceiptBlock, ReducedSwirlSourceReceiptRecord,
        },
        reduced_swirl_source_tree_bridge::ReducedSwirlSourceTreeTrustedVkCommits,
        reduced_swirl_warp::{ReducedSwirlExecutionBus, ReducedSwirlSourceReceiptBus},
        Circuit,
    },
    prover::{ChildVkKind, InnerAggregationProver, InnerGpuProver},
};
use openvm_cuda_backend::{
    BabyBearPoseidon2GpuEngine, GpuBackend, GpuDevice, GpuPreprocessedCommitter,
};
use openvm_recursion_circuit::{
    bus::{Poseidon2CompressBus, TranscriptBus},
    native_warp::NativeWarpTranscriptModule,
    system::AggregationSubCircuit,
};
use openvm_stark_backend::{
    keygen::{
        types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
        MultiStarkKeygenBuilder,
    },
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    proof::Proof,
    prover::{
        AirProvingContext, DeviceDataTransporter, DeviceMultiStarkProvingKey, MatrixDimensions,
        ProverBackend, ProvingContext,
    },
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::{
    baby_bear_poseidon2::{BabyBearPoseidon2CpuEngine, Digest, DuplexSponge, F},
    params_with_100_bits_security,
};
use openvm_verify_stark_host::pvs::{
    VerifierBasePvs, VkCommit, VmPvs, VERIFIER_PVS_AIR_ID, VM_PVS_AIR_ID,
};

use super::{
    reduced_swirl_source_receipt::{
        ProductionReducedSwirlSourceReceiptComponent, ReducedSwirlSourceReceiptCudaPacket,
    },
    reduced_swirl_wrapper_components::ReducedSwirlVerifierComponent,
    reduced_swirl_wrapper_system::ReducedSwirlWrapperSystemError,
};
use crate::SC;

/// Fixed fan-in of the private reduced-SWIRL source-certificate tree.
///
/// This is intentionally independent of OpenVM's public recursive-lane fan-in
/// of three.  With the setup maximum of 1024 reduced sources, the four-wide
/// prefix and eight-wide internal layers reach one standard recursive root
/// without entering a shape-unstable `RecursiveSelf` generation.  The value is
/// compiled into every AIR and VK; it is never profiled from a particular
/// block.
const REDUCED_SWIRL_SOURCE_TREE_INTERNAL_ARITY: usize = 8;

type CpuEngine = BabyBearPoseidon2CpuEngine<DuplexSponge>;
type SourceLeafPrefixGpuProver = InnerAggregationProver<
    GpuBackend,
    ReducedSwirlWrapperPrefixSubCircuit,
    ReducedSwirlWrapperPrefixTraceGen,
>;

/// Setup-fixed height of every source-tree normalization layer.
///
/// A full transition certificate can make an eight-child prefix exceed this
/// envelope, so the setup-fixed prefix uses fan-in four. Both the
/// custom prefix, internal-for-leaf, and internal-recursive provers use this
/// setup-fixed envelope. The internal-recursive prover carries both its
/// standard-child VK commitment and its distinct self VK commitment, exactly
/// as OpenVM's ordinary recursion ladder does. This bound is part of the
/// source-tree verifying key and is never selected from block contents.
pub const REDUCED_SWIRL_SOURCE_TREE_LOG_STACKED_HEIGHT: usize = 23;

/// Raise the ordinary leaf-recursion envelope to the source tree's measured
/// fixed bound. The leaf profile uses a rate-1/4 code here, whereas enlarging
/// the ordinary internal profile would use rate 1/8 and exceed the bounded
/// host-memory budget during key generation.
///
/// Rebuilding through `params_with_100_bits_security` is important: merely
/// editing `n_stack` on the old parameters would leave LogUp and WHIR
/// security calibrated for the smaller original domain.
pub fn reduced_swirl_source_tree_params(
    recursive_leaf: &SystemParams,
) -> Result<SystemParams, ReducedSwirlWrapperSystemError> {
    resize_source_tree_params(
        recursive_leaf,
        REDUCED_SWIRL_SOURCE_TREE_LOG_STACKED_HEIGHT,
        "source-tree recursive parameter envelope",
    )
}

fn resize_source_tree_params(
    template: &SystemParams,
    log_stacked_height: usize,
    error_context: &'static str,
) -> Result<SystemParams, ReducedSwirlWrapperSystemError> {
    if template.l_skip > log_stacked_height || template.log_stacked_height() > log_stacked_height {
        return Err(ReducedSwirlWrapperSystemError::Context(error_context));
    }
    let whir = template.whir();
    Ok(params_with_100_bits_security(
        template.log_blowup,
        template.l_skip,
        log_stacked_height - template.l_skip,
        template.w_stack,
        whir.folding_pow_bits,
        whir.mu_pow_bits,
        whir.proximity,
        template.max_constraint_degree,
        whir.query_phase_pow_bits,
        whir.k,
        template.log_commit_rows_per_query,
    ))
}

pub struct ProductionReducedSwirlSourceLeafComponents<
    const CAPACITY: usize = REDUCED_SWIRL_SOURCE_LEAF_CAPACITY,
> {
    source: Arc<ProductionReducedSwirlSourceReceiptComponent<CAPACITY>>,
    execution_bus: ReducedSwirlExecutionBus,
    boundary_poseidon: NativeWarpTranscriptModule,
}

impl<const CAPACITY: usize> ProductionReducedSwirlSourceLeafComponents<CAPACITY> {
    pub fn new(
        source: Arc<ProductionReducedSwirlSourceReceiptComponent<CAPACITY>>,
        params: SystemParams,
    ) -> Result<Self, ReducedSwirlWrapperSystemError> {
        if CAPACITY == 0
            || CAPACITY > REDUCED_SWIRL_SOURCE_LEAF_CAPACITY
            || !CAPACITY.is_power_of_two()
            || source.inner().receipt_air().profile.source.maximum_sources != CAPACITY
            || source.inner().params() != &params
        {
            return Err(ReducedSwirlWrapperSystemError::KeyIntegrity(
                "source-leaf component capacity",
            ));
        }
        let next = source.inner().next_bus_idx();
        let execution_bus = ReducedSwirlExecutionBus::new(next);
        // The transcript bus is deliberately unused; this tiny module owns
        // one additional Poseidon table on the source verifier's existing
        // compression bus.  Boundary hashing therefore needs no duplicate
        // transcript or host/device reconstruction.
        let boundary_poseidon = NativeWarpTranscriptModule::new_for_bus(
            source.inner().verifier().bus_inventory(),
            TranscriptBus::new(next + 1),
            params,
        );
        Ok(Self {
            source,
            execution_bus,
            boundary_poseidon,
        })
    }

    #[must_use]
    pub fn source(&self) -> Arc<ProductionReducedSwirlSourceReceiptComponent<CAPACITY>> {
        Arc::clone(&self.source)
    }

    #[must_use]
    fn source_air_count(&self) -> usize {
        self.source.inner().airs::<SC>().len()
    }

    fn boundary_poseidon_air<Config: openvm_stark_backend::StarkProtocolConfig<F = F>>(
        &self,
    ) -> openvm_stark_backend::AirRef<Config> {
        self.boundary_poseidon
            .airs::<Config>()
            .into_iter()
            .nth(1)
            .expect("native transcript module Poseidon AIR")
    }
}

impl<const CAPACITY: usize> ReducedSwirlSourceLeafComponents
    for ProductionReducedSwirlSourceLeafComponents<CAPACITY>
{
    fn receipt_bus(&self) -> ReducedSwirlSourceReceiptBus {
        self.source.inner().receipt_air().receipt_bus
    }

    fn execution_bus(&self) -> ReducedSwirlExecutionBus {
        self.execution_bus
    }

    fn compress_bus(&self) -> Poseidon2CompressBus {
        self.source
            .inner()
            .verifier()
            .bus_inventory()
            .poseidon2_compress_bus
    }

    fn component_digest(&self) -> Digest {
        self.source.protocol_digest()
    }

    fn component_air_count(&self) -> usize {
        self.source_air_count() + 1
    }

    fn airs<Config: openvm_stark_backend::StarkProtocolConfig<F = F>>(
        &self,
    ) -> Vec<openvm_stark_backend::AirRef<Config>> {
        self.source
            .airs::<Config>()
            .into_iter()
            .chain(core::iter::once(self.boundary_poseidon_air::<Config>()))
            .collect()
    }
}

pub struct ReducedSwirlSourceLeafCudaSystem<
    const CAPACITY: usize = REDUCED_SWIRL_SOURCE_LEAF_CAPACITY,
> {
    circuit:
        Arc<ReducedSwirlSourceLeafCircuit<ProductionReducedSwirlSourceLeafComponents<CAPACITY>>>,
    params: SystemParams,
}

impl<const CAPACITY: usize> ReducedSwirlSourceLeafCudaSystem<CAPACITY> {
    pub fn new(
        binding: ReducedSwirlSourceLeafBinding,
        components: Arc<ProductionReducedSwirlSourceLeafComponents<CAPACITY>>,
        params: SystemParams,
    ) -> Result<Self, ReducedSwirlWrapperSystemError> {
        if binding.source_capacity as usize != CAPACITY {
            return Err(ReducedSwirlWrapperSystemError::KeyIntegrity(
                "source-leaf binding capacity",
            ));
        }
        let circuit = Arc::new(
            ReducedSwirlSourceLeafCircuit::new(binding, components)
                .map_err(ReducedSwirlWrapperSystemError::Binding)?,
        );
        let system = Self { circuit, params };
        system.validate_air_inventory()?;
        Ok(system)
    }

    #[must_use]
    pub fn binding(&self) -> &ReducedSwirlSourceLeafBinding {
        &self.circuit.binding
    }

    #[must_use]
    pub fn circuit(
        &self,
    ) -> Arc<ReducedSwirlSourceLeafCircuit<ProductionReducedSwirlSourceLeafComponents<CAPACITY>>>
    {
        Arc::clone(&self.circuit)
    }

    #[must_use]
    pub fn airs(&self) -> Vec<openvm_stark_backend::AirRef<SC>> {
        self.circuit.airs()
    }

    fn validate_air_inventory(&self) -> Result<(), ReducedSwirlWrapperSystemError> {
        self.binding()
            .validate()
            .map_err(ReducedSwirlWrapperSystemError::Binding)?;
        let airs = self.airs();
        if airs.len() != 3 + self.circuit.components.component_air_count()
            || airs.first().map(|air| air.num_public_values())
                != Some(size_of::<VerifierBasePvs<u8>>())
            || airs.get(1).map(|air| air.num_public_values()) != Some(size_of::<VmPvs<u8>>())
            || airs
                .get(2..)
                .is_none_or(|airs| airs.iter().any(|air| air.num_public_values() != 0))
        {
            return Err(ReducedSwirlWrapperSystemError::KeyIntegrity(
                "source-leaf AIR inventory",
            ));
        }
        Ok(())
    }

    fn keygen_cuda(
        &self,
        device: &GpuDevice,
    ) -> Result<ReducedSwirlSourceLeafCudaKeys, ReducedSwirlWrapperSystemError> {
        self.validate_air_inventory()?;
        let config = SC::default_from_params(self.params.clone());
        let mut builder = MultiStarkKeygenBuilder::with_preprocessed_committer(
            config,
            Arc::new(GpuPreprocessedCommitter::new(device)),
        );
        for air in self.airs() {
            builder.add_required_air(air);
        }
        let proving_key = builder
            .generate_pk()
            .map_err(|error| ReducedSwirlWrapperSystemError::Keygen(error.to_string()))?;
        let verifying_key = proving_key.get_vk();
        let keys = ReducedSwirlSourceLeafCudaKeys {
            binding: self.binding().clone(),
            proving_key: Arc::new(proving_key),
            verifying_key: Arc::new(verifying_key),
        };
        validate_keys(self, &keys)?;
        Ok(keys)
    }
}

#[derive(Clone)]
pub struct ReducedSwirlSourceLeafCudaKeys {
    binding: ReducedSwirlSourceLeafBinding,
    proving_key: Arc<MultiStarkProvingKey<SC>>,
    verifying_key: Arc<MultiStarkVerifyingKey<SC>>,
}

impl ReducedSwirlSourceLeafCudaKeys {
    #[must_use]
    pub fn proving_key(&self) -> Arc<MultiStarkProvingKey<SC>> {
        Arc::clone(&self.proving_key)
    }

    #[must_use]
    pub fn verifying_key(&self) -> Arc<MultiStarkVerifyingKey<SC>> {
        Arc::clone(&self.verifying_key)
    }
}

pub struct ReducedSwirlSourceLeafProof {
    pub proof: Proof<SC>,
    pub chain_after: Digest,
    pub source_end: u32,
    /// Small authenticated authority records retained for detached VACC trace
    /// generation. The large verifier matrices remain owned by `packet` and
    /// are dropped at the end of this call.
    pub receipt_block: ReducedSwirlSourceReceiptBlock,
}

/// Reassemble only the compact authenticated authority rows needed by the
/// detached VACC trace generator. No verifier matrix or codeword is retained.
pub fn merge_reduced_swirl_source_leaf_receipts(
    blocks: Vec<ReducedSwirlSourceReceiptBlock>,
    block_manifest_digest: Digest,
) -> Result<ReducedSwirlSourceReceiptBlock, ReducedSwirlWrapperSystemError> {
    if blocks.is_empty() || block_manifest_digest.iter().all(|value| *value == F::ZERO) {
        return Err(ReducedSwirlWrapperSystemError::Context(
            "empty source-leaf receipt sequence",
        ));
    }
    let total_sources = blocks
        .iter()
        .try_fold(0usize, |total, block| {
            if block.sources.is_empty() || block.sources.len() > REDUCED_SWIRL_SOURCE_LEAF_CAPACITY
            {
                None
            } else {
                total.checked_add(block.sources.len())
            }
        })
        .ok_or(ReducedSwirlWrapperSystemError::Context(
            "source-leaf receipt count",
        ))?;
    let mut sources: Vec<ReducedSwirlSourceReceiptRecord> = Vec::with_capacity(total_sources);
    let mut expected_offset = 0u32;
    for block in blocks {
        if block.source_offset != expected_offset {
            return Err(ReducedSwirlWrapperSystemError::Context(
                "non-contiguous source-leaf receipt interval",
            ));
        }
        for (local_index, source) in block.sources.into_iter().enumerate() {
            let global_index = expected_offset
                .checked_add(u32::try_from(local_index).map_err(|_| {
                    ReducedSwirlWrapperSystemError::Context("source-leaf receipt index")
                })?)
                .ok_or(ReducedSwirlWrapperSystemError::Context(
                    "source-leaf receipt index overflow",
                ))?;
            if source.segment_index != global_index {
                return Err(ReducedSwirlWrapperSystemError::Context(
                    "source-leaf receipt global index",
                ));
            }
            if let Some(previous) = sources.last() {
                if previous.vm.program_commitment != source.vm.program_commitment
                    || previous.vm.final_pc != source.vm.initial_pc
                    || previous.vm.final_root != source.vm.initial_root
                    || previous.vm.is_terminate != F::ZERO
                {
                    return Err(ReducedSwirlWrapperSystemError::Context(
                        "source-leaf receipt VM continuity",
                    ));
                }
            }
            sources.push(source);
        }
        expected_offset = u32::try_from(sources.len()).map_err(|_| {
            ReducedSwirlWrapperSystemError::Context("source-leaf receipt count overflow")
        })?;
    }
    Ok(ReducedSwirlSourceReceiptBlock {
        source_offset: 0,
        sources,
        manifest_digest: block_manifest_digest,
    })
}

pub struct ReducedSwirlSourceLeafCudaProver<
    const CAPACITY: usize = REDUCED_SWIRL_SOURCE_LEAF_CAPACITY,
> {
    system: Arc<ReducedSwirlSourceLeafCudaSystem<CAPACITY>>,
    keys: ReducedSwirlSourceLeafCudaKeys,
    device_key: DeviceMultiStarkProvingKey<GpuBackend>,
    engine: BabyBearPoseidon2GpuEngine,
}

/// Fixed-key recursion over bounded reduced-SWIRL source leaves.
///
/// A custom app-style prefix verifies four level-zero source certificates at a
/// time, followed by ordinary internal-for-leaf and internal-recursive layers
/// of setup-fixed fan-in eight.  This asymmetric schedule keeps the widest
/// real-block prefix trace inside the log-height-23 device envelope.  For the
/// setup-supported 1024 sources it still reaches one standard recursive root
/// and never invokes the generic `RecursiveSelf` route whose AIR profile is not
/// stable after the custom projection prefix.  The returned VK remains
/// independent of segment count and block contents.
pub struct ReducedSwirlSourceTreeCudaProver {
    source_leaf_vk: Arc<MultiStarkVerifyingKey<SC>>,
    recursive_params: SystemParams,
}

pub struct ReducedSwirlSourceTreeProof {
    pub proof: Proof<SC>,
    pub root_vk: Arc<MultiStarkVerifyingKey<SC>>,
    pub trusted_vk_commits: ReducedSwirlSourceTreeTrustedVkCommits,
}

impl ReducedSwirlSourceTreeCudaProver {
    #[must_use]
    pub fn new(
        source_leaf_vk: Arc<MultiStarkVerifyingKey<SC>>,
        recursive_params: SystemParams,
    ) -> Self {
        Self {
            source_leaf_vk,
            recursive_params,
        }
    }

    pub fn prove(
        self,
        source_leaves: Vec<Proof<SC>>,
        app_vk_commit: VkCommit<F>,
    ) -> Result<ReducedSwirlSourceTreeProof, ReducedSwirlWrapperSystemError> {
        if source_leaves.is_empty() {
            return Err(ReducedSwirlWrapperSystemError::Context(
                "empty source-leaf proof set",
            ));
        }
        let Self {
            source_leaf_vk,
            recursive_params,
        } = self;
        validate_source_tree_leaf_lineage(&source_leaves, app_vk_commit)?;
        let diagnostic = std::env::var_os("OPENVM_REDUCED_SWIRL_SOURCE_TREE_DIAGNOSTIC").is_some();
        let previous_bus_diagnostic =
            diagnostic.then(|| std::env::var_os("OPENVM_CUDA_LOGUP_BUS_DIAGNOSTIC"));
        if diagnostic {
            // This phase is single-threaded at the orchestration boundary. Keep
            // the expensive backend attribution scoped to the compact source
            // tree instead of every preceding segment/source proof.
            std::env::set_var("OPENVM_CUDA_LOGUP_BUS_DIAGNOSTIC", "1");
        }

        // A custom source-leaf relation is not the ordinary internal recursion
        // fixed point. Match OpenVM's app -> leaf -> internal-for-leaf ->
        // recursive ladder. The fixed fan-in reaches one root within the setup
        // source bound, so the generic self route below is only a defensive
        // overflow path and must not be exercised by production inputs.
        let phase_started = Instant::now();
        let leaf_prefix = SourceLeafPrefixGpuProver::new::<BabyBearPoseidon2GpuEngine>(
            source_leaf_vk,
            recursive_params.clone(),
            false,
            None,
        );
        report_source_tree_phase("normalization setup", phase_started, 0);
        // The custom transition-leaf proof is the application proof for this
        // independent recursion ladder. Its own AIR has already constrained
        // the original execution-app VK checked above.
        let recursive_app_vk_commit = leaf_prefix.get_vk_commit(false);
        let phase_started = Instant::now();
        let mut level = reduce_owned_proof_level(
            "source-leaf normalization",
            source_leaves,
            REDUCED_SWIRL_RECURSIVE_PREFIX_ARITY,
            |children| {
                leaf_prefix
                    .agg_prove_no_def::<BabyBearPoseidon2GpuEngine>(children, ChildVkKind::App)
                    .map_err(|error| ReducedSwirlWrapperSystemError::Prover(error.to_string()))
            },
        )?;
        report_source_tree_phase("normalization prove", phase_started, level.len());
        let leaf_prefix_vk = leaf_prefix.get_vk();
        if diagnostic {
            verify_source_tree_level(
                "source-leaf normalization output",
                &level,
                leaf_prefix_vk.as_ref(),
            )?;
        }
        drop(leaf_prefix);
        openvm_cuda_backend::trim_device_memory_pool();

        let phase_started = Instant::now();
        let internal_for_leaf: InnerGpuProver<REDUCED_SWIRL_SOURCE_TREE_INTERNAL_ARITY> =
            InnerGpuProver::new::<BabyBearPoseidon2GpuEngine>(
                leaf_prefix_vk,
                recursive_params.clone(),
                false,
                None,
            );
        report_source_tree_phase("internal-for-leaf setup", phase_started, 0);
        let leaf_vk_commit = internal_for_leaf.get_vk_commit(false);
        let phase_started = Instant::now();
        level = reduce_owned_proof_level(
            "source-tree internal-for-leaf",
            level,
            REDUCED_SWIRL_SOURCE_TREE_INTERNAL_ARITY,
            |children| {
                internal_for_leaf
                    .agg_prove_no_def::<BabyBearPoseidon2GpuEngine>(children, ChildVkKind::Standard)
                    .map_err(|error| ReducedSwirlWrapperSystemError::Prover(error.to_string()))
            },
        )?;
        report_source_tree_phase("internal-for-leaf prove", phase_started, level.len());
        let internal_for_leaf_vk = internal_for_leaf.get_vk();
        if diagnostic {
            verify_source_tree_level(
                "source-tree internal-for-leaf output",
                &level,
                internal_for_leaf_vk.as_ref(),
            )?;
        }
        drop(internal_for_leaf);
        openvm_cuda_backend::trim_device_memory_pool();

        let phase_started = Instant::now();
        let internal_recursive: InnerGpuProver<REDUCED_SWIRL_SOURCE_TREE_INTERNAL_ARITY> =
            InnerGpuProver::new::<BabyBearPoseidon2GpuEngine>(
                internal_for_leaf_vk,
                recursive_params,
                true,
                None,
            );
        report_source_tree_phase("internal-recursive setup", phase_started, 0);
        let root_vk = internal_recursive.get_vk();
        let internal_for_leaf_vk_commit = internal_recursive.get_vk_commit(false);
        // These commitments are intentionally distinct. The standard round
        // authenticates the internal-for-leaf VK; RecursiveSelf rounds
        // authenticate the stable internal-recursive VK itself.
        let recursive_vk_commit = internal_recursive.get_vk_commit(true);
        let trusted_vk_commits = ReducedSwirlSourceTreeTrustedVkCommits {
            app_vk_commit: recursive_app_vk_commit,
            leaf_vk_commit,
            internal_for_leaf_vk_commit,
            recursive_vk_commit,
        };

        // Mandatory first recursive layer fixes the final source-tree VK.
        let phase_started = Instant::now();
        level = reduce_owned_proof_level(
            "source-tree standard",
            level,
            REDUCED_SWIRL_SOURCE_TREE_INTERNAL_ARITY,
            |children| {
                internal_recursive
                    .agg_prove_no_def::<BabyBearPoseidon2GpuEngine>(children, ChildVkKind::Standard)
                    .map_err(|error| ReducedSwirlWrapperSystemError::Prover(error.to_string()))
            },
        )?;
        report_source_tree_phase("internal-recursive prove", phase_started, level.len());
        if diagnostic {
            verify_source_tree_level("source-tree standard output", &level, root_vk.as_ref())?;
        }
        if level.len() != 1 {
            return Err(ReducedSwirlWrapperSystemError::Context(
                "source-tree input exceeds the setup-fixed non-self capacity",
            ));
        }
        if let Some(previous) = previous_bus_diagnostic {
            match previous {
                Some(value) => std::env::set_var("OPENVM_CUDA_LOGUP_BUS_DIAGNOSTIC", value),
                None => std::env::remove_var("OPENVM_CUDA_LOGUP_BUS_DIAGNOSTIC"),
            }
        }
        let proof = level.pop().ok_or(ReducedSwirlWrapperSystemError::Context(
            "source-tree root proof",
        ))?;
        CpuEngine::new(root_vk.inner.params.clone())
            .verify(root_vk.as_ref(), &proof)
            .map_err(|error| ReducedSwirlWrapperSystemError::Verifier(error.to_string()))?;
        Ok(ReducedSwirlSourceTreeProof {
            proof,
            root_vk,
            trusted_vk_commits,
        })
    }
}

fn report_source_tree_phase(phase: &str, started: Instant, output_proofs: usize) {
    eprintln!(
        "REDUCED_SWIRL_SOURCE_TREE_PHASE phase={phase:?} elapsed_ms={:.3} output_proofs={output_proofs}",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
}

fn validate_source_tree_leaf_lineage(
    proofs: &[Proof<SC>],
    expected_app_vk_commit: VkCommit<F>,
) -> Result<(), ReducedSwirlWrapperSystemError> {
    for proof in proofs {
        let fields = proof.public_values.get(VERIFIER_PVS_AIR_ID).ok_or(
            ReducedSwirlWrapperSystemError::Context(
                "source leaf is missing verifier public values",
            ),
        )?;
        if fields.len() < VerifierBasePvs::<u8>::width() {
            return Err(ReducedSwirlWrapperSystemError::Context(
                "source leaf verifier public values have the wrong width",
            ));
        }
        let pvs: &VerifierBasePvs<F> = fields[..VerifierBasePvs::<u8>::width()].borrow();
        if pvs.internal_flag != F::ZERO
            || pvs.recursion_depth != F::ZERO
            || pvs.app_vk_commit != expected_app_vk_commit
        {
            return Err(ReducedSwirlWrapperSystemError::Context(
                "source leaf does not belong to the expected execution-app lineage",
            ));
        }
    }
    Ok(())
}

/// In diagnostic mode, distinguish an invalid parent proof from a
/// `RecursiveSelf` trace-generation mismatch before the next tree layer tries
/// to verify that parent in-circuit.  Production proving deliberately omits
/// these redundant CPU verifications.
fn verify_source_tree_level(
    stage: &str,
    proofs: &[Proof<SC>],
    vk: &MultiStarkVerifyingKey<SC>,
) -> Result<(), ReducedSwirlWrapperSystemError> {
    let engine = CpuEngine::new(vk.inner.params.clone());
    for (index, proof) in proofs.iter().enumerate() {
        engine.verify(vk, proof).map_err(|error| {
            ReducedSwirlWrapperSystemError::Verifier(format!("{stage} proof {index}: {error}"))
        })?;
    }
    Ok(())
}

/// Reduce one proof-tree level in the same stable fixed-arity order as
/// `slice::chunks`, but consume each child group. This matters for long
/// blocks: after a parent has authenticated a group, none of those child
/// proof allocations remain reachable while the rest of the level is being
/// processed.
fn reduce_owned_proof_level(
    stage: &str,
    proofs: Vec<Proof<SC>>,
    group_capacity: usize,
    mut prove_group: impl FnMut(&[Proof<SC>]) -> Result<Proof<SC>, ReducedSwirlWrapperSystemError>,
) -> Result<Vec<Proof<SC>>, ReducedSwirlWrapperSystemError> {
    if group_capacity == 0 {
        return Err(ReducedSwirlWrapperSystemError::Context(
            "zero source-tree group capacity",
        ));
    }
    let parent_capacity = proofs.len().div_ceil(group_capacity);
    let mut children = proofs.into_iter();
    let mut parents = Vec::with_capacity(parent_capacity);
    for group_index in 0..parent_capacity {
        let group = children.by_ref().take(group_capacity).collect::<Vec<_>>();
        if group.is_empty() {
            break;
        }
        validate_source_tree_group(&group).map_err(|message| {
            ReducedSwirlWrapperSystemError::Prover(format!(
                "{stage} group {group_index} invalid authenticated child boundary: {message}"
            ))
        })?;
        parents.push(prove_group(&group).map_err(|error| {
            ReducedSwirlWrapperSystemError::Prover(format!("{stage} group {group_index}: {error}"))
        })?);
        // `group` and every authenticated child proof are released here.
    }
    Ok(parents)
}

fn validate_source_tree_group(proofs: &[Proof<SC>]) -> Result<(), &'static str> {
    let suspend = F::from_u32(openvm_circuit::system::connector::DEFAULT_SUSPEND_EXIT_CODE);
    let mut previous: Option<&VmPvs<F>> = None;
    for (index, proof) in proofs.iter().enumerate() {
        let fields = proof
            .public_values
            .get(VM_PVS_AIR_ID)
            .ok_or("missing VmPvs AIR")?;
        if fields.len() != VmPvs::<u8>::width() {
            return Err("wrong VmPvs width");
        }
        let current: &VmPvs<F> = fields.as_slice().borrow();
        if current.is_terminate != F::ZERO && current.is_terminate != F::ONE {
            return Err("non-Boolean termination flag");
        }
        if current.is_terminate == F::ONE {
            if index + 1 != proofs.len() || current.exit_code != F::ZERO {
                return Err("terminal child is not the final successful child");
            }
        } else if current.exit_code != suspend {
            return Err("non-terminal child has a non-suspend exit code");
        }
        if let Some(previous) = previous {
            if previous.program_commit != current.program_commit {
                return Err("program commitment changed");
            }
            if previous.final_pc != current.initial_pc {
                return Err("program counter interval is discontinuous");
            }
            if previous.final_root != current.initial_root {
                return Err("memory/receipt root interval is discontinuous");
            }
        }
        previous = Some(current);
    }
    Ok(())
}

impl<const CAPACITY: usize> ReducedSwirlSourceLeafCudaProver<CAPACITY> {
    pub fn new(
        system: Arc<ReducedSwirlSourceLeafCudaSystem<CAPACITY>>,
    ) -> Result<Self, ReducedSwirlWrapperSystemError> {
        let mut engine = BabyBearPoseidon2GpuEngine::new(system.params.clone());
        engine.device_mut().prover_config_mut().compile_monomials = false;
        let keys = system.keygen_cuda(engine.device())?;
        let prepared = <GpuDevice as DeviceDataTransporter<SC, GpuBackend>>::prepare_pk_for_device(
            engine.device(),
            keys.proving_key.as_ref(),
        );
        let device_key =
            <GpuDevice as DeviceDataTransporter<SC, GpuBackend>>::transport_prepared_pk_to_device(
                engine.device(),
                keys.proving_key.as_ref(),
                prepared,
            );
        Ok(Self {
            system,
            keys,
            device_key,
            engine,
        })
    }

    #[must_use]
    pub fn keys(&self) -> &ReducedSwirlSourceLeafCudaKeys {
        &self.keys
    }

    #[must_use]
    pub fn engine(&self) -> &BabyBearPoseidon2GpuEngine {
        &self.engine
    }

    pub fn prove_packet(
        &mut self,
        mut packet: ReducedSwirlSourceReceiptCudaPacket,
        chain_before: Digest,
    ) -> Result<ReducedSwirlSourceLeafProof, ReducedSwirlWrapperSystemError> {
        if packet.block.sources.is_empty() || packet.block.sources.len() > CAPACITY {
            return Err(ReducedSwirlWrapperSystemError::Context(
                "source-leaf packet capacity",
            ));
        }
        validate_keys(self.system.as_ref(), &self.keys)?;
        let protocol_digest = self
            .system
            .circuit
            .components
            .source
            .inner()
            .receipt_air()
            .profile
            .protocol_digest;
        let record =
            ReducedSwirlSourceLeafRecord::from_block(&packet.block, protocol_digest, chain_before)
                .map_err(ReducedSwirlWrapperSystemError::Binding)?;
        let core = generate_reduced_swirl_source_leaf_core_traces(
            self.system.binding(),
            self.system.circuit.boundary_air.as_ref(),
            &record,
        )
        .map_err(ReducedSwirlWrapperSystemError::Binding)?;

        let source_air_count = self.system.circuit.components.source_air_count();
        if packet.contexts.len() != source_air_count
            || packet
                .contexts
                .iter()
                .enumerate()
                .any(|(expected, (actual, _))| expected != *actual)
        {
            return Err(ReducedSwirlWrapperSystemError::Context(
                "source-leaf source context inventory",
            ));
        }
        let poseidon_trace = self
            .system
            .circuit
            .components
            .boundary_poseidon
            .build_poseidon2_trace_gpu(
                Vec::new(),
                core.derived.compression_inputs.clone(),
                None,
                &self.engine.device().device_ctx,
            )
            .ok_or(ReducedSwirlWrapperSystemError::Context(
                "source-leaf boundary Poseidon trace",
            ))?;
        packet.contexts.push((
            source_air_count,
            AirProvingContext::simple_no_pis(poseidon_trace),
        ));

        let expected = expected_public_values(
            core.verifier_public_values.clone(),
            core.vm_public_values.clone(),
            self.system.circuit.components.component_air_count(),
        );
        let device = self.engine.device();
        let mut per_trace = vec![
            (
                0,
                AirProvingContext::new(
                    Vec::new(),
                    <GpuDevice as DeviceDataTransporter<SC, GpuBackend>>::
                        transport_row_major_matrix_to_device(device, &core.verifier_pvs),
                    core.verifier_public_values,
                ),
            ),
            (
                1,
                AirProvingContext::new(
                    Vec::new(),
                    <GpuDevice as DeviceDataTransporter<SC, GpuBackend>>::
                        transport_row_major_matrix_to_device(device, &core.vm_pvs),
                    core.vm_public_values,
                ),
            ),
            (
                2,
                AirProvingContext::new(
                    Vec::new(),
                    <GpuDevice as DeviceDataTransporter<SC, GpuBackend>>::
                        transport_row_major_matrix_to_device(device, &core.boundary),
                    Vec::new(),
                ),
            ),
        ];
        for (component_air, context) in packet.contexts {
            validate_component_context(&self.system.airs()[component_air + 3], &context)?;
            per_trace.push((component_air + 3, context));
        }
        let context = ProvingContext::new(per_trace);
        #[cfg(debug_assertions)]
        if std::env::var("OPENVM_SKIP_DEBUG") != Ok(String::from("1")) {
            self.engine.debug(&self.system.airs(), &context);
        }
        let proof = self
            .engine
            .prove(&self.device_key, context)
            .map_err(|error| ReducedSwirlWrapperSystemError::Prover(error.to_string()))?;
        if proof.public_values != expected {
            return Err(ReducedSwirlWrapperSystemError::PublicValues);
        }
        // The parent recursive verifier authenticates every leaf and the
        // final root is checked on CPU. Re-verifying each sibling here would
        // serialize the CUDA stream in release builds; retain it only as a
        // debug differential check, matching `InnerAggregationProver`.
        #[cfg(debug_assertions)]
        if std::env::var("OPENVM_SKIP_DEBUG") != Ok(String::from("1")) {
            CpuEngine::new(self.keys.verifying_key.inner.params.clone())
                .verify(self.keys.verifying_key.as_ref(), &proof)
                .map_err(|error| ReducedSwirlWrapperSystemError::Verifier(error.to_string()))?;
        }
        Ok(ReducedSwirlSourceLeafProof {
            proof,
            chain_after: core.derived.chain_after,
            source_end: core.derived.source_end.as_canonical_u32(),
            receipt_block: packet.block,
        })
    }
}

fn validate_keys<const CAPACITY: usize>(
    system: &ReducedSwirlSourceLeafCudaSystem<CAPACITY>,
    keys: &ReducedSwirlSourceLeafCudaKeys,
) -> Result<(), ReducedSwirlWrapperSystemError> {
    system.validate_air_inventory()?;
    let airs = system.airs();
    let config = SC::default_from_params(keys.verifying_key.inner.params.clone());
    if keys.binding != *system.binding()
        || keys.proving_key.params != system.params
        || keys.verifying_key.inner.params != system.params
        || keys.proving_key.per_air.len() != airs.len()
        || keys.verifying_key.inner.per_air.len() != airs.len()
        || keys.proving_key.vk_pre_hash != keys.verifying_key.pre_hash
        || !keys.verifying_key.has_consistent_pre_hash(&config)
    {
        return Err(ReducedSwirlWrapperSystemError::KeyIntegrity(
            "source-leaf key",
        ));
    }
    for (vk, air) in keys.verifying_key.inner.per_air.iter().zip(airs) {
        if !vk.is_required
            || vk.params.num_public_values != air.num_public_values()
            || vk.params.width.common_main != air.common_main_width()
            || vk.params.width.cached_mains != air.cached_main_widths()
        {
            return Err(ReducedSwirlWrapperSystemError::KeyIntegrity(
                "source-leaf AIR key shape",
            ));
        }
    }
    Ok(())
}

fn validate_component_context<PB: ProverBackend<Val = F>>(
    air: &openvm_stark_backend::AirRef<SC>,
    context: &AirProvingContext<PB>,
) -> Result<(), ReducedSwirlWrapperSystemError> {
    let height = context.common_main.height();
    if !context.public_values.is_empty()
        || height == 0
        || !height.is_power_of_two()
        || context.common_main.width() != air.common_main_width()
        || context.cached_mains.len() != air.cached_main_widths().len()
        || context
            .cached_mains
            .iter()
            .zip(air.cached_main_widths())
            .any(|(cached, width)| cached.trace.width() != width || cached.trace.height() != height)
    {
        return Err(ReducedSwirlWrapperSystemError::Context(
            "source-leaf component trace shape",
        ));
    }
    Ok(())
}

fn expected_public_values(verifier: Vec<F>, vm: Vec<F>, component_air_count: usize) -> Vec<Vec<F>> {
    let mut expected = vec![verifier, vm, Vec::new()];
    expected.resize_with(3 + component_air_count, Vec::new);
    expected
}

#[cfg(test)]
mod tests {
    use openvm_stark_sdk::config::leaf_params_with_100_bits_security;

    use super::{reduced_swirl_source_tree_params, REDUCED_SWIRL_SOURCE_TREE_LOG_STACKED_HEIGHT};

    #[test]
    fn source_tree_recursive_profile_is_fixed_at_the_measured_height() {
        let recursive = leaf_params_with_100_bits_security();
        let source_tree = reduced_swirl_source_tree_params(&recursive).unwrap();

        assert_eq!(
            source_tree.log_stacked_height(),
            REDUCED_SWIRL_SOURCE_TREE_LOG_STACKED_HEIGHT
        );
        assert_eq!(source_tree.log_blowup, recursive.log_blowup);
        assert_eq!(source_tree.l_skip, recursive.l_skip);
        assert_eq!(source_tree.w_stack, recursive.w_stack);
        assert_eq!(
            source_tree.max_constraint_degree,
            recursive.max_constraint_degree
        );
        assert_eq!(source_tree.whir().proximity, recursive.whir().proximity);
        assert_eq!(source_tree.whir().k, recursive.whir().k);
    }
}

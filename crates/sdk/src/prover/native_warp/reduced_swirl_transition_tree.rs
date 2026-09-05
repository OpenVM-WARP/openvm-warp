//! Fixed-key OpenVM recursion tree for reduced-SWIRL WARP transition proofs.
//!
//! Each input proof certifies one bounded WARP update and its corresponding
//! reduced-SWIRL sources. The tree uses setup-fixed fan-in and parameters; it
//! does not profile its circuit or proof order from block contents.

use std::{borrow::Borrow, sync::Arc, time::Instant};

use openvm_continuations::{
    circuit::{
        inner::ReducedSwirlWrapperPrefixTraceGen,
        reduced_swirl_recursive_prefix::{
            ReducedSwirlWrapperPrefixSubCircuit, REDUCED_SWIRL_RECURSIVE_PREFIX_ARITY,
        },
        reduced_swirl_transition_finalizer::ReducedSwirlTransitionTreeTrustedVkCommits,
    },
    prover::{ChildVkKind, InnerAggregationProver, InnerGpuProver},
};
use openvm_cuda_backend::{BabyBearPoseidon2GpuEngine, GpuBackend};
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey, p3_field::PrimeCharacteristicRing, proof::Proof,
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::{
    baby_bear_poseidon2::{BabyBearPoseidon2CpuEngine, DuplexSponge, F},
    params_with_100_bits_security,
};
use openvm_verify_stark_host::pvs::{
    VerifierBasePvs, VkCommit, VmPvs, VERIFIER_PVS_AIR_ID, VM_PVS_AIR_ID,
};

use super::reduced_swirl_error::ReducedSwirlWrapperSystemError;
use crate::SC;

/// Fixed fan-in of the private reduced-SWIRL transition-certificate tree.
///
/// This is intentionally independent of OpenVM's public recursive-lane fan-in
/// of three.  With the setup maximum of 1024 reduced sources, the four-wide
/// prefix and eight-wide internal layers reach one standard recursive root
/// without entering a shape-unstable `RecursiveSelf` generation.  The value is
/// compiled into every AIR and VK; it is never profiled from a particular
/// block.
const REDUCED_SWIRL_TRANSITION_TREE_INTERNAL_ARITY: usize = 8;

type CpuEngine = BabyBearPoseidon2CpuEngine<DuplexSponge>;
type TransitionLeafPrefixGpuProver = InnerAggregationProver<
    GpuBackend,
    ReducedSwirlWrapperPrefixSubCircuit,
    ReducedSwirlWrapperPrefixTraceGen,
>;

/// Setup-fixed height of every transition-tree normalization layer.
///
/// A full transition certificate can make an eight-child prefix exceed this
/// envelope, so the setup-fixed prefix uses fan-in four. Both the
/// custom prefix, internal-for-leaf, and internal-recursive provers use this
/// setup-fixed envelope. The internal-recursive prover carries both its
/// standard-child VK commitment and its distinct self VK commitment, exactly
/// as OpenVM's ordinary recursion ladder does. This bound is part of the
/// transition-tree verifying key and is never selected from block contents.
pub const REDUCED_SWIRL_TRANSITION_TREE_LOG_STACKED_HEIGHT: usize = 23;

/// Raise the ordinary leaf-recursion envelope to the transition tree's measured
/// fixed bound. The leaf profile uses a rate-1/4 code here, whereas enlarging
/// the ordinary internal profile would use rate 1/8 and exceed the bounded
/// host-memory budget during key generation.
///
/// Rebuilding through `params_with_100_bits_security` is important: merely
/// editing `n_stack` on the old parameters would leave LogUp and WHIR
/// security calibrated for the smaller original domain.
pub fn reduced_swirl_transition_tree_params(
    recursive_leaf: &SystemParams,
) -> Result<SystemParams, ReducedSwirlWrapperSystemError> {
    resize_transition_tree_params(
        recursive_leaf,
        REDUCED_SWIRL_TRANSITION_TREE_LOG_STACKED_HEIGHT,
        "transition-tree recursive parameter envelope",
    )
}

fn resize_transition_tree_params(
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

/// Fixed-key recursion over bounded reduced-SWIRL source leaves.
///
/// A custom app-style prefix verifies four level-zero transition certificates at a
/// time, followed by ordinary internal-for-leaf and internal-recursive layers
/// of setup-fixed fan-in eight.  This asymmetric schedule keeps the widest
/// real-block prefix trace inside the log-height-23 device envelope.  For the
/// setup-supported 1024 sources it still reaches one standard recursive root
/// and never invokes the generic `RecursiveSelf` route whose AIR profile is not
/// stable after the custom projection prefix.  The returned VK remains
/// independent of segment count and block contents.
pub struct ReducedSwirlTransitionTreeCudaProver {
    transition_leaf_vk: Arc<MultiStarkVerifyingKey<SC>>,
    recursive_params: SystemParams,
}

pub struct ReducedSwirlTransitionTreeProof {
    pub proof: Proof<SC>,
    pub root_vk: Arc<MultiStarkVerifyingKey<SC>>,
    pub trusted_vk_commits: ReducedSwirlTransitionTreeTrustedVkCommits,
}

impl ReducedSwirlTransitionTreeCudaProver {
    #[must_use]
    pub fn new(
        transition_leaf_vk: Arc<MultiStarkVerifyingKey<SC>>,
        recursive_params: SystemParams,
    ) -> Self {
        Self {
            transition_leaf_vk,
            recursive_params,
        }
    }

    pub fn prove(
        self,
        transition_leaves: Vec<Proof<SC>>,
        app_vk_commit: VkCommit<F>,
    ) -> Result<ReducedSwirlTransitionTreeProof, ReducedSwirlWrapperSystemError> {
        if transition_leaves.is_empty() {
            return Err(ReducedSwirlWrapperSystemError::Context(
                "empty transition-leaf proof set",
            ));
        }
        let Self {
            transition_leaf_vk,
            recursive_params,
        } = self;
        validate_transition_tree_leaf_lineage(&transition_leaves, app_vk_commit)?;
        let diagnostic =
            std::env::var_os("OPENVM_REDUCED_SWIRL_TRANSITION_TREE_DIAGNOSTIC").is_some();
        let previous_bus_diagnostic =
            diagnostic.then(|| std::env::var_os("OPENVM_CUDA_LOGUP_BUS_DIAGNOSTIC"));
        if diagnostic {
            // This phase is single-threaded at the orchestration boundary. Keep
            // the expensive backend attribution scoped to the compact source
            // tree instead of every preceding segment/source proof.
            std::env::set_var("OPENVM_CUDA_LOGUP_BUS_DIAGNOSTIC", "1");
        }

        // A custom transition-leaf relation is not the ordinary internal recursion
        // fixed point. Match OpenVM's app -> leaf -> internal-for-leaf ->
        // recursive ladder. The fixed fan-in reaches one root within the setup
        // source bound, so the generic self route below is only a defensive
        // overflow path and must not be exercised by production inputs.
        let phase_started = Instant::now();
        let leaf_prefix = TransitionLeafPrefixGpuProver::new::<BabyBearPoseidon2GpuEngine>(
            transition_leaf_vk,
            recursive_params.clone(),
            false,
            None,
        );
        report_transition_tree_phase("normalization setup", phase_started, 0);
        // The custom transition-leaf proof is the application proof for this
        // independent recursion ladder. Its own AIR has already constrained
        // the original execution-app VK checked above.
        let recursive_app_vk_commit = leaf_prefix.get_vk_commit(false);
        let phase_started = Instant::now();
        let mut level = reduce_owned_proof_level(
            "transition-leaf normalization",
            transition_leaves,
            REDUCED_SWIRL_RECURSIVE_PREFIX_ARITY,
            |children| {
                leaf_prefix
                    .agg_prove_no_def::<BabyBearPoseidon2GpuEngine>(children, ChildVkKind::App)
                    .map_err(|error| ReducedSwirlWrapperSystemError::Prover(error.to_string()))
            },
        )?;
        report_transition_tree_phase("normalization prove", phase_started, level.len());
        let leaf_prefix_vk = leaf_prefix.get_vk();
        if diagnostic {
            verify_transition_tree_level(
                "transition-leaf normalization output",
                &level,
                leaf_prefix_vk.as_ref(),
            )?;
        }
        drop(leaf_prefix);
        openvm_cuda_backend::trim_device_memory_pool();

        let phase_started = Instant::now();
        let internal_for_leaf: InnerGpuProver<REDUCED_SWIRL_TRANSITION_TREE_INTERNAL_ARITY> =
            InnerGpuProver::new::<BabyBearPoseidon2GpuEngine>(
                leaf_prefix_vk,
                recursive_params.clone(),
                false,
                None,
            );
        report_transition_tree_phase("internal-for-leaf setup", phase_started, 0);
        let leaf_vk_commit = internal_for_leaf.get_vk_commit(false);
        let phase_started = Instant::now();
        level = reduce_owned_proof_level(
            "transition-tree internal-for-leaf",
            level,
            REDUCED_SWIRL_TRANSITION_TREE_INTERNAL_ARITY,
            |children| {
                internal_for_leaf
                    .agg_prove_no_def::<BabyBearPoseidon2GpuEngine>(children, ChildVkKind::Standard)
                    .map_err(|error| ReducedSwirlWrapperSystemError::Prover(error.to_string()))
            },
        )?;
        report_transition_tree_phase("internal-for-leaf prove", phase_started, level.len());
        let internal_for_leaf_vk = internal_for_leaf.get_vk();
        if diagnostic {
            verify_transition_tree_level(
                "transition-tree internal-for-leaf output",
                &level,
                internal_for_leaf_vk.as_ref(),
            )?;
        }
        drop(internal_for_leaf);
        openvm_cuda_backend::trim_device_memory_pool();

        let phase_started = Instant::now();
        let internal_recursive: InnerGpuProver<REDUCED_SWIRL_TRANSITION_TREE_INTERNAL_ARITY> =
            InnerGpuProver::new::<BabyBearPoseidon2GpuEngine>(
                internal_for_leaf_vk,
                recursive_params,
                true,
                None,
            );
        report_transition_tree_phase("internal-recursive setup", phase_started, 0);
        let root_vk = internal_recursive.get_vk();
        let internal_for_leaf_vk_commit = internal_recursive.get_vk_commit(false);
        // These commitments are intentionally distinct. The standard round
        // authenticates the internal-for-leaf VK; RecursiveSelf rounds
        // authenticate the stable internal-recursive VK itself.
        let recursive_vk_commit = internal_recursive.get_vk_commit(true);
        let trusted_vk_commits = ReducedSwirlTransitionTreeTrustedVkCommits {
            app_vk_commit: recursive_app_vk_commit,
            transition_leaf_vk_commit: leaf_vk_commit,
            internal_for_leaf_vk_commit,
            recursive_vk_commit,
        };

        // Mandatory first recursive layer fixes the final transition-tree VK.
        let phase_started = Instant::now();
        level = reduce_owned_proof_level(
            "transition-tree standard",
            level,
            REDUCED_SWIRL_TRANSITION_TREE_INTERNAL_ARITY,
            |children| {
                internal_recursive
                    .agg_prove_no_def::<BabyBearPoseidon2GpuEngine>(children, ChildVkKind::Standard)
                    .map_err(|error| ReducedSwirlWrapperSystemError::Prover(error.to_string()))
            },
        )?;
        report_transition_tree_phase("internal-recursive prove", phase_started, level.len());
        if diagnostic {
            verify_transition_tree_level(
                "transition-tree standard output",
                &level,
                root_vk.as_ref(),
            )?;
        }
        if level.len() != 1 {
            return Err(ReducedSwirlWrapperSystemError::Context(
                "transition-tree input exceeds the setup-fixed non-self capacity",
            ));
        }
        if let Some(previous) = previous_bus_diagnostic {
            match previous {
                Some(value) => std::env::set_var("OPENVM_CUDA_LOGUP_BUS_DIAGNOSTIC", value),
                None => std::env::remove_var("OPENVM_CUDA_LOGUP_BUS_DIAGNOSTIC"),
            }
        }
        let proof = level.pop().ok_or(ReducedSwirlWrapperSystemError::Context(
            "transition-tree root proof",
        ))?;
        CpuEngine::new(root_vk.inner.params.clone())
            .verify(root_vk.as_ref(), &proof)
            .map_err(|error| ReducedSwirlWrapperSystemError::Verifier(error.to_string()))?;
        Ok(ReducedSwirlTransitionTreeProof {
            proof,
            root_vk,
            trusted_vk_commits,
        })
    }
}

fn report_transition_tree_phase(phase: &str, started: Instant, output_proofs: usize) {
    eprintln!(
        "REDUCED_SWIRL_TRANSITION_TREE_PHASE phase={phase:?} elapsed_ms={:.3} output_proofs={output_proofs}",
        started.elapsed().as_secs_f64() * 1_000.0,
    );
}

fn validate_transition_tree_leaf_lineage(
    proofs: &[Proof<SC>],
    expected_app_vk_commit: VkCommit<F>,
) -> Result<(), ReducedSwirlWrapperSystemError> {
    for proof in proofs {
        let fields = proof.public_values.get(VERIFIER_PVS_AIR_ID).ok_or(
            ReducedSwirlWrapperSystemError::Context(
                "transition leaf is missing verifier public values",
            ),
        )?;
        if fields.len() < VerifierBasePvs::<u8>::width() {
            return Err(ReducedSwirlWrapperSystemError::Context(
                "transition leaf verifier public values have the wrong width",
            ));
        }
        let pvs: &VerifierBasePvs<F> = fields[..VerifierBasePvs::<u8>::width()].borrow();
        if pvs.internal_flag != F::ZERO
            || pvs.recursion_depth != F::ZERO
            || pvs.app_vk_commit != expected_app_vk_commit
        {
            return Err(ReducedSwirlWrapperSystemError::Context(
                "transition leaf does not belong to the expected execution-app lineage",
            ));
        }
    }
    Ok(())
}

/// In diagnostic mode, distinguish an invalid parent proof from a
/// `RecursiveSelf` trace-generation mismatch before the next tree layer tries
/// to verify that parent in-circuit.  Production proving deliberately omits
/// these redundant CPU verifications.
fn verify_transition_tree_level(
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
            "zero transition-tree group capacity",
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
        validate_transition_tree_group(&group).map_err(|message| {
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

fn validate_transition_tree_group(proofs: &[Proof<SC>]) -> Result<(), &'static str> {
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

#[cfg(test)]
mod tests {
    use openvm_stark_sdk::config::leaf_params_with_100_bits_security;

    use super::{
        reduced_swirl_transition_tree_params, REDUCED_SWIRL_TRANSITION_TREE_LOG_STACKED_HEIGHT,
    };

    #[test]
    fn transition_tree_recursive_profile_is_fixed_at_the_measured_height() {
        let recursive = leaf_params_with_100_bits_security();
        let transition_tree = reduced_swirl_transition_tree_params(&recursive).unwrap();

        assert_eq!(
            transition_tree.log_stacked_height(),
            REDUCED_SWIRL_TRANSITION_TREE_LOG_STACKED_HEIGHT
        );
        assert_eq!(transition_tree.log_blowup, recursive.log_blowup);
        assert_eq!(transition_tree.l_skip, recursive.l_skip);
        assert_eq!(transition_tree.w_stack, recursive.w_stack);
        assert_eq!(
            transition_tree.max_constraint_degree,
            recursive.max_constraint_degree
        );
        assert_eq!(transition_tree.whir().proximity, recursive.whir().proximity);
        assert_eq!(transition_tree.whir().k, recursive.whir().k);
    }
}

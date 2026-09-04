//! Existing-recursion compression for the reduced-SWIRL WARP wrapper.
//!
//! The reduced wrapper is a setup-fixed custom MultiSTARK with VM public
//! values. A tiny app-style prefix authenticates those values, after which
//! the ordinary OpenVM internal-for-leaf and internal-recursive provers
//! normalize it for the existing Root/Halo2 path. No transition History is
//! replayed here.

use std::sync::Arc;

use eyre::{eyre, Result, WrapErr};
use openvm_circuit::system::memory::{
    dimensions::MemoryDimensions, merkle::public_values::UserPublicValuesProof,
};
use openvm_continuations::{
    circuit::{
        inner::{ProofsType, ReducedSwirlWrapperPrefixTraceGen},
        reduced_swirl_recursive_prefix::{
            ReducedSwirlWrapperPrefixSubCircuit, REDUCED_SWIRL_RECURSIVE_PREFIX_ARITY,
        },
        reduced_swirl_warp::ReducedSwirlWrapperBinding,
    },
    prover::{ChildVkKind, InnerAggregationProver},
};
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    proof::Proof,
    StarkEngine, SystemParams,
};
use openvm_stark_sdk::config::{
    baby_bear_poseidon2::{BabyBearPoseidon2CpuEngine, Digest, DuplexSponge, DIGEST_SIZE, F},
    params_with_100_bits_security,
};
use openvm_verify_stark_host::{
    pvs::VkCommit,
    verify_vm_stark_proof_decoded,
    vk::{VerificationBaseline, VmStarkVerifyingKey},
    VmStarkProof,
};

use crate::SC;

pub const REDUCED_SWIRL_RECURSIVE_ADAPTER_ARITY: usize = REDUCED_SWIRL_RECURSIVE_PREFIX_ARITY;
pub const REDUCED_SWIRL_RECURSIVE_ADAPTER_LOG_STACKED_HEIGHT: usize = 21;

type VerificationEngine = BabyBearPoseidon2CpuEngine<DuplexSponge>;

cfg_if::cfg_if! {
    if #[cfg(feature = "cuda")] {
        use openvm_continuations::prover::InnerGpuProver as StandardInnerProver;
        use openvm_cuda_backend::GpuBackend;
        type ProvingEngine = openvm_cuda_backend::BabyBearPoseidon2GpuEngine;
        type PrefixProver = InnerAggregationProver<
            GpuBackend,
            ReducedSwirlWrapperPrefixSubCircuit,
            ReducedSwirlWrapperPrefixTraceGen,
        >;
    } else {
        use openvm_continuations::prover::InnerCpuProver as StandardInnerProver;
        use openvm_cpu_backend::CpuBackend;
        type ProvingEngine = BabyBearPoseidon2CpuEngine<DuplexSponge>;
        type PrefixProver = InnerAggregationProver<
            CpuBackend<SC>,
            ReducedSwirlWrapperPrefixSubCircuit,
            ReducedSwirlWrapperPrefixTraceGen,
        >;
    }
}

/// Derive the ordinary recursive verifier profile for one setup-fixed
/// reduced-SWIRL wrapper proof.
///
/// This changes only the authenticated PCS envelope needed by the wrapper
/// verifying key. It does not add a WARP transition, a History relation, or a
/// host acceptance bit.
#[must_use]
pub fn reduced_swirl_canonical_adapter_params(
    wrapper_params: &SystemParams,
    recursive_internal: SystemParams,
) -> SystemParams {
    let recursive_width = recursive_internal.w_stack.next_power_of_two();
    let required_width = wrapper_params
        .w_stack
        .max(recursive_width)
        .next_power_of_two();
    let extra_width_bits = required_width
        .ilog2()
        .saturating_sub(recursive_width.ilog2()) as usize;
    let log_stacked_height = recursive_internal
        .log_stacked_height()
        .max(wrapper_params.log_stacked_height())
        .max(REDUCED_SWIRL_RECURSIVE_ADAPTER_LOG_STACKED_HEIGHT);
    let mut adapter = params_with_100_bits_security(
        recursive_internal.log_blowup,
        recursive_internal.l_skip,
        log_stacked_height - recursive_internal.l_skip,
        required_width,
        recursive_internal.whir.folding_pow_bits,
        recursive_internal
            .whir
            .mu_pow_bits
            .saturating_add(extra_width_bits),
        recursive_internal.whir.proximity,
        recursive_internal.max_constraint_degree,
        recursive_internal.whir.query_phase_pow_bits,
        recursive_internal.whir.k,
        recursive_internal.log_commit_rows_per_query,
    );
    let log_max_message_length = recursive_internal
        .logup
        .log_max_message_length
        .max(adapter.logup.log_max_message_length);
    let extra_message_bits =
        log_max_message_length.saturating_sub(adapter.logup.log_max_message_length);
    adapter.logup.log_max_message_length = log_max_message_length;
    adapter.logup.pow_bits += extra_message_bits as usize;
    adapter
}

/// Existing-recursion compression rooted at the honest custom-app boundary.
///
/// The prefix verifies the reduced-WARP wrapper as an app-style custom
/// MultiSTARK and emits an ordinary leaf proof. The following two provers are
/// exactly OpenVM's normal internal-for-leaf and internal-recursive stages.
pub struct ReducedSwirlRecursiveAdapter {
    prefix: PrefixProver,
    internal_for_leaf: StandardInnerProver<REDUCED_SWIRL_RECURSIVE_ADAPTER_ARITY>,
    internal_recursive: StandardInnerProver<REDUCED_SWIRL_RECURSIVE_ADAPTER_ARITY>,
}

#[derive(Clone)]
pub struct ReducedSwirlRecursiveAdapterProvingKeys {
    pub prefix: Arc<MultiStarkProvingKey<SC>>,
    pub internal_for_leaf: Arc<MultiStarkProvingKey<SC>>,
    pub internal_recursive: Arc<MultiStarkProvingKey<SC>>,
}

impl ReducedSwirlRecursiveAdapter {
    #[must_use]
    pub fn new(
        wrapper_vk: Arc<MultiStarkVerifyingKey<SC>>,
        aggregation_params: SystemParams,
    ) -> Self {
        let prefix =
            PrefixProver::new::<ProvingEngine>(wrapper_vk, aggregation_params.clone(), false, None);
        let internal_for_leaf = StandardInnerProver::new::<ProvingEngine>(
            prefix.get_vk(),
            aggregation_params.clone(),
            false,
            None,
        );
        let internal_recursive = StandardInnerProver::new::<ProvingEngine>(
            internal_for_leaf.get_vk(),
            aggregation_params,
            true,
            None,
        );
        Self {
            prefix,
            internal_for_leaf,
            internal_recursive,
        }
    }

    #[must_use]
    pub fn from_pks(
        wrapper_vk: Arc<MultiStarkVerifyingKey<SC>>,
        proving_keys: ReducedSwirlRecursiveAdapterProvingKeys,
    ) -> Self {
        let prefix =
            PrefixProver::from_pk::<ProvingEngine>(wrapper_vk, proving_keys.prefix, false, None);
        let internal_for_leaf = StandardInnerProver::from_pk::<ProvingEngine>(
            prefix.get_vk(),
            proving_keys.internal_for_leaf,
            false,
            None,
        );
        let internal_recursive = StandardInnerProver::from_pk::<ProvingEngine>(
            internal_for_leaf.get_vk(),
            proving_keys.internal_recursive,
            true,
            None,
        );
        Self {
            prefix,
            internal_for_leaf,
            internal_recursive,
        }
    }

    #[must_use]
    pub fn proving_keys(&self) -> ReducedSwirlRecursiveAdapterProvingKeys {
        ReducedSwirlRecursiveAdapterProvingKeys {
            prefix: self.prefix.get_pk(),
            internal_for_leaf: self.internal_for_leaf.get_pk(),
            internal_recursive: self.internal_recursive.get_pk(),
        }
    }

    #[must_use]
    pub fn verifying_key(&self) -> Arc<MultiStarkVerifyingKey<SC>> {
        self.internal_recursive.get_vk()
    }

    /// Wrapper VK authenticated by the custom app-style prefix.
    #[must_use]
    pub fn wrapper_verifying_key_commit(&self) -> VkCommit<F> {
        self.prefix.get_vk_commit(false)
    }

    /// Prefix/leaf VK authenticated by OpenVM's ordinary internal-for-leaf
    /// stage.
    #[must_use]
    pub fn leaf_verifying_key_commit(&self) -> VkCommit<F> {
        self.internal_for_leaf.get_vk_commit(false)
    }

    #[must_use]
    pub fn internal_for_leaf_verifying_key_commit(&self) -> VkCommit<F> {
        self.internal_recursive.get_vk_commit(false)
    }

    #[must_use]
    pub fn self_verifying_key_commit(&self) -> VkCommit<F> {
        self.internal_recursive.get_vk_commit(true)
    }

    pub fn prove_for_root(&self, wrapper_proof: &Proof<SC>) -> Result<Proof<SC>> {
        let leaf = self
            .prefix
            .agg_prove::<ProvingEngine>(
                std::slice::from_ref(wrapper_proof),
                ChildVkKind::App,
                ProofsType::Vm,
                None,
            )
            .wrap_err("reduced-SWIRL custom-app prefix proving failed")?;
        verify_with_vk(self.prefix.get_vk(), &leaf)?;
        let internal_for_leaf = self
            .internal_for_leaf
            .agg_prove_no_def::<ProvingEngine>(std::slice::from_ref(&leaf), ChildVkKind::Standard)
            .wrap_err("reduced-SWIRL internal-for-leaf proving failed")?;
        verify_with_vk(self.internal_for_leaf.get_vk(), &internal_for_leaf)?;
        let recursive = self
            .internal_recursive
            .agg_prove_no_def::<ProvingEngine>(
                std::slice::from_ref(&internal_for_leaf),
                ChildVkKind::Standard,
            )
            .wrap_err("reduced-SWIRL internal-recursive proving failed")?;
        verify_with_vk(self.internal_recursive.get_vk(), &recursive)?;
        Ok(recursive)
    }
}

fn verify_with_vk(vk: Arc<MultiStarkVerifyingKey<SC>>, proof: &Proof<SC>) -> Result<()> {
    VerificationEngine::new(vk.inner.params.clone())
        .verify(vk.as_ref(), proof)
        .wrap_err("reduced-SWIRL recursive adapter proof verification failed")
}

/// Compress one verified direct reduced-SWIRL wrapper proof into OpenVM's
/// ordinary root-compatible recursive proof and package its original VM
/// public-values opening.
///
/// The app prefix and two ordinary normalization layers are implemented by
/// [`ReducedSwirlRecursiveAdapter::prove_for_root`]. The returned verifying
/// key derives every recursive key commitment from that setup-owned adapter;
/// only the application VK lineage comes from the setup-fixed reduced-SWIRL
/// binding. The standard decoded VM verifier is run before either artifact is
/// returned.
pub fn package_reduced_swirl_recursive_proof(
    adapter: &ReducedSwirlRecursiveAdapter,
    wrapper_proof: Proof<SC>,
    binding: ReducedSwirlWrapperBinding,
    user_public_values: UserPublicValuesProof<DIGEST_SIZE, F>,
    app_exe_commit: Digest,
    memory_dimensions: MemoryDimensions,
    num_user_pvs: usize,
) -> Result<(VmStarkProof, VmStarkVerifyingKey)> {
    binding
        .validate()
        .map_err(|error| eyre!("invalid reduced-SWIRL wrapper binding: {error}"))?;

    let root_compatible = adapter
        .prove_for_root(&wrapper_proof)
        .map_err(|error| eyre!("reduced-SWIRL recursive normalization failed: {error:?}"))?;
    let proof = VmStarkProof {
        inner: root_compatible,
        user_pvs_proof: user_public_values,
        deferral_merkle_proofs: None,
    };
    let baseline = reduced_swirl_verification_baseline(
        app_exe_commit,
        memory_dimensions,
        num_user_pvs,
        adapter.wrapper_verifying_key_commit(),
        adapter.leaf_verifying_key_commit(),
        adapter.internal_for_leaf_verifying_key_commit(),
        adapter.self_verifying_key_commit(),
    );
    let verifying_key = VmStarkVerifyingKey {
        mvk: adapter.verifying_key().as_ref().clone(),
        baseline,
    };

    verify_vm_stark_proof_decoded(&verifying_key, &proof)
        .map_err(|error| eyre!(error))
        .wrap_err("packaged reduced-SWIRL VM STARK proof failed CPU verification")?;
    Ok((proof, verifying_key))
}

#[allow(clippy::too_many_arguments)]
fn reduced_swirl_verification_baseline(
    app_exe_commit: Digest,
    memory_dimensions: MemoryDimensions,
    num_user_pvs: usize,
    wrapper_vk_commit: VkCommit<F>,
    leaf_vk_commit: VkCommit<F>,
    internal_for_leaf_vk_commit: VkCommit<F>,
    self_vk_commit: VkCommit<F>,
) -> VerificationBaseline {
    VerificationBaseline {
        app_exe_commit,
        memory_dimensions,
        num_user_pvs,
        app_vk_commit: wrapper_vk_commit,
        leaf_vk_commit,
        internal_for_leaf_vk_commit,
        internal_recursive_vk_commit: self_vk_commit,
        expected_def_hook_commit: None,
    }
}

#[cfg(test)]
mod tests {
    use openvm_stark_backend::p3_field::PrimeCharacteristicRing;

    use super::*;

    fn digest(value: u32) -> Digest {
        core::array::from_fn(|index| F::from_u32(value + index as u32))
    }

    fn vk_commit(value: u32) -> VkCommit<F> {
        VkCommit {
            cached_commit: digest(value),
            vk_pre_hash: digest(value + 100),
        }
    }

    #[test]
    fn verification_baseline_uses_exact_setup_lineage() {
        let binding = ReducedSwirlWrapperBinding {
            protocol_version: 3,
            protocol_digest: digest(1),
            relation_digest: digest(2),
            warp_index_digest: digest(3),
            terminal_index_digest: digest(4),
            verifier_component_digest: digest(5),
            input_arity: 8,
            recursive_app_vk_commit: vk_commit(6),
        };
        let app_exe_commit = digest(7);
        let memory_dimensions = MemoryDimensions::new(3, 27);
        let wrapper = vk_commit(8);
        let leaf = vk_commit(9);
        let internal_for_leaf = vk_commit(10);
        let self_recursive = vk_commit(11);

        let baseline = reduced_swirl_verification_baseline(
            app_exe_commit,
            memory_dimensions,
            16,
            wrapper,
            leaf,
            internal_for_leaf,
            self_recursive,
        );

        assert_eq!(baseline.app_exe_commit, app_exe_commit);
        assert_eq!(baseline.memory_dimensions.addr_space_height, 3);
        assert_eq!(baseline.memory_dimensions.address_height, 27);
        assert_eq!(baseline.num_user_pvs, 16);
        assert_eq!(baseline.app_vk_commit, wrapper);
        assert_eq!(baseline.leaf_vk_commit, leaf);
        assert_eq!(baseline.internal_for_leaf_vk_commit, internal_for_leaf);
        assert_eq!(baseline.internal_recursive_vk_commit, self_recursive);
        assert!(baseline.expected_def_hook_commit.is_none());
    }
}

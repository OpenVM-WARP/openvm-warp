//! Shared bindings and buses for reduced-SWIRL WARP recursion.
//!
//! Transition leaves and the terminal finalizer share this setup binding,
//! source/execution messages, and public-value AIRs. Their protocol-specific
//! receipts remain in the modules that consume them.

use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, LookupBus},
    p3_air::{Air, AirBuilder, AirBuilderWithPublicValues, BaseAir, PairBuilder},
    p3_field::{PrimeCharacteristicRing, PrimeField32},
    p3_matrix::Matrix,
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, F};
use openvm_verify_stark_host::pvs::{VerifierBasePvs, VkCommit, VmPvs};
use serde::{Deserialize, Serialize};

pub const REDUCED_SWIRL_WRAPPER_PROTOCOL_VERSION: u32 = 3;
pub const REDUCED_SWIRL_WRAPPER_MAX_SOURCES: u32 = 1024;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ReducedSwirlWrapperBinding {
    pub protocol_version: u32,
    pub protocol_digest: Digest,
    pub relation_digest: Digest,
    pub warp_index_digest: Digest,
    pub terminal_index_digest: Digest,
    pub verifier_component_digest: Digest,
    pub input_arity: u32,
    /// Commitment to the exact application VK cached trace under the direct
    /// wrapper PCS. The source component fixes the child VK and its symbolic
    /// DAG directly in its AIR inventory, so a second "source" commitment
    /// would be redundant and, worse, would not be consumed by any AIR.
    pub recursive_app_vk_commit: VkCommit<F>,
}

impl ReducedSwirlWrapperBinding {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.protocol_version != REDUCED_SWIRL_WRAPPER_PROTOCOL_VERSION {
            return Err("protocol version");
        }
        if self.input_arity < 2 || self.input_arity > 64 || !self.input_arity.is_power_of_two() {
            return Err("input arity");
        }
        if [
            self.protocol_digest,
            self.relation_digest,
            self.warp_index_digest,
            self.terminal_index_digest,
            self.verifier_component_digest,
        ]
        .iter()
        .any(is_zero_digest)
        {
            return Err("unset binding digest");
        }
        if is_zero_digest(&self.recursive_app_vk_commit.cached_commit)
            || is_zero_digest(&self.recursive_app_vk_commit.vk_pre_hash)
        {
            return Err("application VK lineage");
        }
        Ok(())
    }

    #[must_use]
    pub fn verifier_pvs(&self) -> VerifierBasePvs<F> {
        let unset = VkCommit {
            cached_commit: [F::ZERO; DIGEST_SIZE],
            vk_pre_hash: [F::ZERO; DIGEST_SIZE],
        };
        VerifierBasePvs {
            internal_flag: F::ZERO,
            app_vk_commit: self.recursive_app_vk_commit,
            leaf_vk_commit: unset,
            internal_for_leaf_vk_commit: unset,
            recursion_depth: F::ZERO,
            internal_recursive_vk_commit: unset,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlSourceReceiptMessage<T> {
    pub protocol_digest: [T; DIGEST_SIZE],
    pub manifest_digest: [T; DIGEST_SIZE],
    pub source_offset: T,
    pub source_count: T,
    pub program_commitment: [T; DIGEST_SIZE],
    pub initial_pc: T,
    pub initial_root: [T; DIGEST_SIZE],
    pub final_pc: T,
    pub final_root: [T; DIGEST_SIZE],
    pub exit_code: T,
    pub is_terminate: T,
}

impl<T: Clone> ReducedSwirlSourceReceiptMessage<T> {
    fn to_vec(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(46);
        out.extend_from_slice(&self.protocol_digest);
        out.extend_from_slice(&self.manifest_digest);
        out.extend([self.source_offset.clone(), self.source_count.clone()]);
        out.extend_from_slice(&self.program_commitment);
        out.push(self.initial_pc.clone());
        out.extend_from_slice(&self.initial_root);
        out.push(self.final_pc.clone());
        out.extend_from_slice(&self.final_root);
        out.extend([self.exit_code.clone(), self.is_terminate.clone()]);
        out
    }
}

#[derive(Clone)]
pub struct ReducedSwirlExecutionMessage<T> {
    pub vm_pvs: VmPvs<T>,
}

impl<T: Clone> ReducedSwirlExecutionMessage<T> {
    fn to_vec(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(28);
        out.extend_from_slice(&self.vm_pvs.program_commit);
        out.extend([
            self.vm_pvs.initial_pc.clone(),
            self.vm_pvs.final_pc.clone(),
            self.vm_pvs.exit_code.clone(),
            self.vm_pvs.is_terminate.clone(),
        ]);
        out.extend_from_slice(&self.vm_pvs.initial_root);
        out.extend_from_slice(&self.vm_pvs.final_root);
        out
    }
}

macro_rules! typed_bus {
    ($name:ident, $message:ident) => {
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $name(LookupBus);
        impl $name {
            #[must_use]
            pub const fn new(index: BusIndex) -> Self {
                Self(LookupBus::new(index))
            }
            #[must_use]
            pub const fn index(self) -> BusIndex {
                self.0.index
            }
            pub fn lookup_key<AB, T>(
                &self,
                builder: &mut AB,
                message: $message<T>,
                enabled: impl Into<AB::Expr>,
            ) where
                AB: InteractionBuilder,
                T: Into<AB::Expr> + Clone,
            {
                self.0.lookup_key(builder, message.to_vec(), enabled);
            }
            pub fn add_key_with_lookups<AB, T>(
                &self,
                builder: &mut AB,
                message: $message<T>,
                lookups: impl Into<AB::Expr>,
            ) where
                AB: InteractionBuilder,
                T: Into<AB::Expr> + Clone,
            {
                self.0
                    .add_key_with_lookups(builder, message.to_vec(), lookups);
            }
        }
    };
}

typed_bus!(
    ReducedSwirlSourceReceiptBus,
    ReducedSwirlSourceReceiptMessage
);
typed_bus!(ReducedSwirlExecutionBus, ReducedSwirlExecutionMessage);

#[derive(Clone)]
pub struct ReducedSwirlWrapperVerifierPvsAir {
    expected: VerifierBasePvs<F>,
}

impl ReducedSwirlWrapperVerifierPvsAir {
    #[must_use]
    pub fn new(binding: &ReducedSwirlWrapperBinding) -> Self {
        Self {
            expected: binding.verifier_pvs(),
        }
    }
}

impl BaseAir<F> for ReducedSwirlWrapperVerifierPvsAir {
    fn width(&self) -> usize {
        1
    }
}
impl BaseAirWithPublicValues<F> for ReducedSwirlWrapperVerifierPvsAir {
    fn num_public_values(&self) -> usize {
        VerifierBasePvs::<u8>::width()
    }
}
impl PartitionedBaseAir<F> for ReducedSwirlWrapperVerifierPvsAir {}
impl<AB> Air<AB> for ReducedSwirlWrapperVerifierPvsAir
where
    AB: AirBuilder<F = F> + AirBuilderWithPublicValues + PairBuilder,
    AB::Var: Copy,
    AB::PublicVar: Copy,
    AB::Expr: From<AB::PublicVar>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("wrapper PVS row")[0];
        let next = main.row_slice(1).expect("wrapper PVS padding")[0];
        one_row_selector(builder, local, next);
        let public_values = builder.public_values().to_vec();
        for (actual, expected) in public_values.iter().zip(self.expected.as_slice()) {
            builder.when(local).assert_eq(
                Into::<AB::Expr>::into(*actual),
                AB::Expr::from_u32(expected.as_canonical_u32()),
            );
        }
    }
}

#[derive(Clone, Copy)]
pub struct ReducedSwirlWrapperVmPvsAir {
    bus: ReducedSwirlExecutionBus,
}
impl ReducedSwirlWrapperVmPvsAir {
    #[must_use]
    pub const fn new(bus: ReducedSwirlExecutionBus) -> Self {
        Self { bus }
    }
}
impl BaseAir<F> for ReducedSwirlWrapperVmPvsAir {
    fn width(&self) -> usize {
        1
    }
}
impl BaseAirWithPublicValues<F> for ReducedSwirlWrapperVmPvsAir {
    fn num_public_values(&self) -> usize {
        VmPvs::<u8>::width()
    }
}
impl PartitionedBaseAir<F> for ReducedSwirlWrapperVmPvsAir {}
impl<AB> Air<AB> for ReducedSwirlWrapperVmPvsAir
where
    AB: AirBuilder<F = F> + AirBuilderWithPublicValues + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
    AB::PublicVar: Copy,
    AB::Expr: From<AB::PublicVar>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local = main.row_slice(0).expect("wrapper VmPvs row")[0];
        let next = main.row_slice(1).expect("wrapper VmPvs padding")[0];
        one_row_selector(builder, local, next);
        let pvs = builder.public_values();
        let v = |i: usize| Into::<AB::Expr>::into(pvs[i]);
        self.bus.add_key_with_lookups(
            builder,
            ReducedSwirlExecutionMessage {
                vm_pvs: VmPvs {
                    program_commit: core::array::from_fn(&v),
                    initial_pc: v(DIGEST_SIZE),
                    final_pc: v(DIGEST_SIZE + 1),
                    exit_code: v(DIGEST_SIZE + 2),
                    is_terminate: v(DIGEST_SIZE + 3),
                    initial_root: core::array::from_fn(|i| v(DIGEST_SIZE + 4 + i)),
                    final_root: core::array::from_fn(|i| v(2 * DIGEST_SIZE + 4 + i)),
                },
            },
            local,
        );
    }
}

fn one_row_selector<AB: AirBuilder>(builder: &mut AB, local: AB::Var, next: AB::Var)
where
    AB::Var: Copy,
{
    builder.assert_bool(local);
    builder.when_first_row().assert_one(local);
    builder.when_last_row().assert_zero(local);
    builder.when_transition().assert_eq(local - next, local);
}

fn is_zero_digest(digest: &Digest) -> bool {
    digest.iter().all(|value| *value == F::ZERO)
}

use openvm_stark_backend::interaction::{BusIndex, InteractionBuilder, LookupBus};
use openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE;
use openvm_verify_stark_host::pvs::VmPvs;

use super::FINITE_WARP_V3_MAX_CALLS;

#[derive(Clone, Debug)]
pub struct FiniteWarpV3CallReceiptMessage<T> {
    pub protocol_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub warp_index_digest: [T; DIGEST_SIZE],
    pub verifier_component_digest: [T; DIGEST_SIZE],
    pub schedule_digest: [T; DIGEST_SIZE],
    pub manifest_digest: [T; DIGEST_SIZE],
    pub call_index: T,
    pub source_start: T,
    pub source_count: T,
    pub input_arity: T,
    pub fresh_stacked_root: [T; DIGEST_SIZE],
    pub prior_accumulator_digest: [T; DIGEST_SIZE],
    pub output_accumulator_digest: [T; DIGEST_SIZE],
}

impl<T: Clone> FiniteWarpV3CallReceiptMessage<T> {
    #[must_use]
    pub fn to_vec(&self) -> Vec<T> {
        let mut values = Vec::with_capacity(76);
        values.extend_from_slice(&self.protocol_digest);
        values.extend_from_slice(&self.relation_digest);
        values.extend_from_slice(&self.warp_index_digest);
        values.extend_from_slice(&self.verifier_component_digest);
        values.extend_from_slice(&self.schedule_digest);
        values.extend_from_slice(&self.manifest_digest);
        values.extend([
            self.call_index.clone(),
            self.source_start.clone(),
            self.source_count.clone(),
            self.input_arity.clone(),
        ]);
        values.extend_from_slice(&self.fresh_stacked_root);
        values.extend_from_slice(&self.prior_accumulator_digest);
        values.extend_from_slice(&self.output_accumulator_digest);
        values
    }
}

#[derive(Clone, Debug)]
pub struct FiniteWarpV3ManifestCallMessage<T> {
    pub active: T,
    pub source_start: T,
    pub source_count: T,
    pub input_arity: T,
    pub fresh_stacked_root: [T; DIGEST_SIZE],
}

#[derive(Clone, Debug)]
pub struct FiniteWarpV3ManifestReceiptMessage<T> {
    pub protocol_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub warp_index_digest: [T; DIGEST_SIZE],
    pub verifier_component_digest: [T; DIGEST_SIZE],
    pub schedule_digest: [T; DIGEST_SIZE],
    pub manifest_digest: [T; DIGEST_SIZE],
    pub source_count: T,
    pub call_count: T,
    pub program_commitment: [T; DIGEST_SIZE],
    pub initial_pc: T,
    pub initial_root: [T; DIGEST_SIZE],
    pub final_pc: T,
    pub final_root: [T; DIGEST_SIZE],
    pub final_accumulator_digest: [T; DIGEST_SIZE],
    pub calls: [FiniteWarpV3ManifestCallMessage<T>; FINITE_WARP_V3_MAX_CALLS],
}

impl<T: Clone> FiniteWarpV3ManifestReceiptMessage<T> {
    #[must_use]
    pub fn to_vec(&self) -> Vec<T> {
        let mut values = Vec::with_capacity(120);
        values.extend_from_slice(&self.protocol_digest);
        values.extend_from_slice(&self.relation_digest);
        values.extend_from_slice(&self.warp_index_digest);
        values.extend_from_slice(&self.verifier_component_digest);
        values.extend_from_slice(&self.schedule_digest);
        values.extend_from_slice(&self.manifest_digest);
        values.extend([self.source_count.clone(), self.call_count.clone()]);
        values.extend_from_slice(&self.program_commitment);
        values.push(self.initial_pc.clone());
        values.extend_from_slice(&self.initial_root);
        values.push(self.final_pc.clone());
        values.extend_from_slice(&self.final_root);
        values.extend_from_slice(&self.final_accumulator_digest);
        for call in &self.calls {
            values.extend([
                call.active.clone(),
                call.source_start.clone(),
                call.source_count.clone(),
                call.input_arity.clone(),
            ]);
            values.extend_from_slice(&call.fresh_stacked_root);
        }
        values
    }
}

#[derive(Clone, Debug)]
pub struct FiniteWarpV3TerminalReceiptMessage<T> {
    pub protocol_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub terminal_index_digest: [T; DIGEST_SIZE],
    pub verifier_component_digest: [T; DIGEST_SIZE],
    pub final_accumulator_digest: [T; DIGEST_SIZE],
    pub final_accumulator_root: [T; DIGEST_SIZE],
}

impl<T: Clone> FiniteWarpV3TerminalReceiptMessage<T> {
    #[must_use]
    pub fn to_vec(&self) -> Vec<T> {
        let mut values = Vec::with_capacity(48);
        values.extend_from_slice(&self.protocol_digest);
        values.extend_from_slice(&self.relation_digest);
        values.extend_from_slice(&self.terminal_index_digest);
        values.extend_from_slice(&self.verifier_component_digest);
        values.extend_from_slice(&self.final_accumulator_digest);
        values.extend_from_slice(&self.final_accumulator_root);
        values
    }
}

#[derive(Clone)]
pub struct FiniteWarpV3ExecutionMessage<T> {
    pub vm_pvs: VmPvs<T>,
}

impl<T: Clone> FiniteWarpV3ExecutionMessage<T> {
    #[must_use]
    pub fn to_vec(&self) -> Vec<T> {
        let mut values = Vec::with_capacity(28);
        values.extend_from_slice(&self.vm_pvs.program_commit);
        values.extend([
            self.vm_pvs.initial_pc.clone(),
            self.vm_pvs.final_pc.clone(),
            self.vm_pvs.exit_code.clone(),
            self.vm_pvs.is_terminate.clone(),
        ]);
        values.extend_from_slice(&self.vm_pvs.initial_root);
        values.extend_from_slice(&self.vm_pvs.final_root);
        values
    }
}

macro_rules! typed_lookup_bus {
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

typed_lookup_bus!(FiniteWarpV3CallReceiptBus, FiniteWarpV3CallReceiptMessage);
typed_lookup_bus!(
    FiniteWarpV3ManifestReceiptBus,
    FiniteWarpV3ManifestReceiptMessage
);
typed_lookup_bus!(
    FiniteWarpV3TerminalReceiptBus,
    FiniteWarpV3TerminalReceiptMessage
);
typed_lookup_bus!(FiniteWarpV3ExecutionBus, FiniteWarpV3ExecutionMessage);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FiniteWarpV3ReceiptBuses {
    pub call: FiniteWarpV3CallReceiptBus,
    pub manifest: FiniteWarpV3ManifestReceiptBus,
    pub terminal: FiniteWarpV3TerminalReceiptBus,
    pub execution: FiniteWarpV3ExecutionBus,
}

impl FiniteWarpV3ReceiptBuses {
    #[must_use]
    pub const fn new(first_bus_idx: BusIndex) -> Self {
        Self {
            call: FiniteWarpV3CallReceiptBus::new(first_bus_idx),
            manifest: FiniteWarpV3ManifestReceiptBus::new(first_bus_idx + 1),
            terminal: FiniteWarpV3TerminalReceiptBus::new(first_bus_idx + 2),
            execution: FiniteWarpV3ExecutionBus::new(first_bus_idx + 3),
        }
    }

    #[must_use]
    pub const fn next_bus_idx(self) -> BusIndex {
        self.execution.index() + 1
    }
}

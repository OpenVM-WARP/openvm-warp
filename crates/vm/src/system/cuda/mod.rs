use std::sync::Arc;

use connector::VmConnectorChipGPU;
use memory::MemoryInventoryGPU;
use openvm_circuit::{
    arch::{DenseRecordArena, SystemConfig},
    system::{
        connector::VmConnectorChip, memory::online::GuestMemory, SystemChipComplex, SystemRecords,
        SystemWithFixedTraceHeights, CONNECTOR_AIR_ID, PROGRAM_AIR_ID,
    },
};
use openvm_circuit_primitives::{var_range::VariableRangeCheckerChipGPU, Chip};
use openvm_cuda_backend::{prelude::F, GpuBackend};
use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_stark_backend::prover::{AirProvingContext, CommittedTraceData, MatrixDimensions};
use poseidon2::Poseidon2PeripheryChipGPU;
use program::ProgramChipGPU;

pub(crate) use crate::system::memory::DIGEST_WIDTH;

pub mod boundary;
pub mod connector;
pub mod extensions;
pub mod memory;
pub mod merkle_tree;
pub mod phantom;
pub mod poseidon2;
pub mod program;

pub struct SystemChipInventoryGPU {
    pub program: ProgramChipGPU,
    pub connector: VmConnectorChipGPU,
    pub memory_inventory: MemoryInventoryGPU,
}

impl SystemChipInventoryGPU {
    pub fn new(
        config: &SystemConfig,
        range_checker: Arc<VariableRangeCheckerChipGPU>,
        hasher_chip: Arc<Poseidon2PeripheryChipGPU>,
        device_ctx: GpuDeviceCtx,
    ) -> Self {
        let cpu_range_checker = range_checker.cpu_chip.clone().unwrap();

        // We create an empty program chip: the program should be loaded later (and can be swapped
        // out). The execution frequencies are supplied only after execution.
        let program_chip = ProgramChipGPU::new(device_ctx.clone());
        let connector_chip = VmConnectorChipGPU::new(
            VmConnectorChip::new(
                cpu_range_checker.clone(),
                config.memory_config.timestamp_max_bits,
            ),
            device_ctx.clone(),
        );

        let memory_inventory = MemoryInventoryGPU::new(
            config.memory_config.clone(),
            hasher_chip,
            device_ctx.clone(),
        );

        Self {
            program: program_chip,
            connector: connector_chip,
            memory_inventory,
        }
    }
}

impl SystemWithFixedTraceHeights for SystemChipInventoryGPU {
    /// Warning: as on the CPU inventory, this does not set the override for the program chip.
    /// The program trace is cached and its height is already fixed by the loaded program, so the
    /// two constant-height system AIRs are asserted rather than pinned.
    fn override_trace_heights(&mut self, heights: &[u32]) {
        assert_eq!(
            heights[PROGRAM_AIR_ID] as usize,
            self.program
                .cached
                .as_ref()
                .expect("program not loaded")
                .trace
                .height()
        );
        assert_eq!(heights[CONNECTOR_AIR_ID], 2);
        self.memory_inventory.set_override_trace_heights(heights);
    }
}

impl SystemChipComplex<DenseRecordArena, GpuBackend> for SystemChipInventoryGPU {
    fn load_program(&mut self, cached_program_trace: CommittedTraceData<GpuBackend>) {
        self.program.cached.replace(cached_program_trace);
    }

    fn cached_program_trace(&self) -> Option<&CommittedTraceData<GpuBackend>> {
        self.program.cached.as_ref()
    }

    fn transport_init_memory_to_device(&mut self, memory: &GuestMemory) {
        self.memory_inventory.set_initial_memory(&memory.memory);
    }

    fn generate_proving_ctx(
        &mut self,
        system_records: SystemRecords<F>,
        _record_arenas: Vec<DenseRecordArena>,
    ) -> Vec<AirProvingContext<GpuBackend>> {
        let SystemRecords {
            from_state,
            to_state,
            exit_code,
            filtered_exec_frequencies,
            touched_memory,
        } = system_records;

        let program_ctx = self.program.generate_proving_ctx(filtered_exec_frequencies);

        self.connector.cpu_chip.begin(from_state);
        self.connector.cpu_chip.end(to_state, exit_code);
        let connector_ctx = self.connector.generate_proving_ctx(());

        let memory_ctxs = self.memory_inventory.generate_proving_ctxs(touched_memory);

        [program_ctx, connector_ctx]
            .into_iter()
            .chain(memory_ctxs)
            .collect()
    }

    fn memory_top_tree(&self) -> Option<&[[F; DIGEST_WIDTH]]> {
        let top_tree = &self.memory_inventory.merkle_tree.top_roots_host;
        (!top_tree.is_empty()).then_some(top_tree.as_slice())
    }
}

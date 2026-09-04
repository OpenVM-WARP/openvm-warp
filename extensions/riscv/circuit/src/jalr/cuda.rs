use std::{mem::size_of, sync::Arc};

use derive_new::new;
use openvm_circuit::arch::DenseRecordArena;
use openvm_circuit_primitives::{var_range::VariableRangeCheckerChipGPU, Chip};
use openvm_cuda_backend::{base::DeviceMatrix, prelude::F, GpuBackend};
use openvm_cuda_common::copy::MemCopyH2D;
use openvm_stark_backend::prover::AirProvingContext;

use crate::{
    adapters::{Rv64JalrAdapterCols, Rv64JalrAdapterRecord},
    cuda_abi::jalr_cuda::tracegen,
    Rv64JalrCoreCols, Rv64JalrCoreRecord,
};
#[derive(new)]
pub struct Rv64JalrChipGpu {
    pub range_checker: Arc<VariableRangeCheckerChipGPU>,
    pub timestamp_max_bits: usize,
}

impl Chip<DenseRecordArena, GpuBackend> for Rv64JalrChipGpu {
    fn generate_proving_ctx(&self, arena: DenseRecordArena) -> AirProvingContext<GpuBackend> {
        const RECORD_SIZE: usize = size_of::<(Rv64JalrAdapterRecord, Rv64JalrCoreRecord)>();
        let records = arena.allocated();
        // Honour a pinned height: a chip that executed zero times this segment still
        // owes the rows the plan plus shape catalog were built against. The kernel
        // fills every row at or past the record count with this chip's padding row, so
        // an empty record set with a pinned height is a valid all-padding trace.
        let trace_height = arena.trace_height(RECORD_SIZE);
        if trace_height == 0 {
            return AirProvingContext::simple_no_pis(DeviceMatrix::dummy());
        }
        debug_assert_eq!(records.len() % RECORD_SIZE, 0);

        let trace_width = Rv64JalrCoreCols::<F>::width() + Rv64JalrAdapterCols::<F>::width();
        let device_ctx = &self.range_checker.device_ctx;

        let d_records = tracing::info_span!("trace_gen.h2d_records")
            .in_scope(|| records.to_device_on(device_ctx))
            .unwrap();
        let d_trace = DeviceMatrix::<F>::with_capacity_on(trace_height, trace_width, device_ctx);

        unsafe {
            tracegen(
                d_trace.buffer(),
                trace_height,
                &d_records,
                &self.range_checker.count,
                self.timestamp_max_bits as u32,
                device_ctx.stream.as_raw(),
            )
            .unwrap();
        }
        AirProvingContext::simple_no_pis(d_trace)
    }
}

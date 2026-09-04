use crate::{arch::VmExecState, system::memory::online::GuestMemory};

pub mod metered;
pub mod metered_cost;
mod preflight;
mod pure;

pub use metered::{
    ctx::{MeteredCtx, MeteredCtxConfig, MeteredCtxInputs},
    segment_ctx::{Segment, SegmentationConfig, SegmentationLimits},
};
pub use metered_cost::MeteredCostCtx;
pub use preflight::PreflightCtx;
pub use pure::ExecutionCtx;

pub trait ExecutionCtxTrait: Sized {
    /// Records a memory access that may modify the addressed cells.
    ///
    /// Callers that know an access is read-only should use [`Self::on_memory_read`]. Keeping the
    /// generic callback conservative means an unclassified extension access cannot undercount a
    /// write when metering a fixed proving shape.
    fn on_memory_operation(&mut self, address_space: u32, ptr: u32, size: u32);

    /// Records a memory access known to be read-only.
    #[inline(always)]
    fn on_memory_read(&mut self, address_space: u32, ptr: u32, size: u32) {
        self.on_memory_operation(address_space, ptr, size);
    }

    fn should_suspend<F>(exec_state: &mut VmExecState<F, GuestMemory, Self>) -> bool;

    fn on_terminate<F>(_exec_state: &mut VmExecState<F, GuestMemory, Self>) {}
}

pub trait MeteredExecutionCtxTrait: ExecutionCtxTrait {
    fn on_height_change(&mut self, chip_idx: usize, height_delta: u32);
}

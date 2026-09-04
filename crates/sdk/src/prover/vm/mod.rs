use std::sync::Arc;

use openvm_circuit::arch::{
    instructions::exe::VmExe, VirtualMachine, VirtualMachineError, VmBuilder, VmInstance,
};
use openvm_stark_backend::{prover::DeviceDataTransporter, StarkEngine, Val};

use crate::prover::vm::types::VmProvingKey;

pub mod types;

pub fn new_local_prover<E, VB>(
    vm_builder: VB,
    vm_pk: &VmProvingKey<VB::VmConfig>,
    exe: Arc<VmExe<Val<E::SC>>>,
) -> Result<VmInstance<E, VB>, VirtualMachineError>
where
    E: StarkEngine<SC = crate::SC>,
    VB: VmBuilder<E>,
{
    let engine = E::new(vm_pk.get_params());
    new_local_prover_with_engine(engine, vm_builder, vm_pk, exe)
}

/// Construct a local prover from a caller-configured engine.
///
/// Device residency choices that affect proving-key and cached-program PCS
/// ownership must be installed before either object is committed.  The
/// reduced-SWIRL CUDA lane uses this seam to retain the original RS matrices
/// instead of recommitting them after execution has started.
pub fn new_local_prover_with_engine<E, VB>(
    engine: E,
    vm_builder: VB,
    vm_pk: &VmProvingKey<VB::VmConfig>,
    exe: Arc<VmExe<Val<E::SC>>>,
) -> Result<VmInstance<E, VB>, VirtualMachineError>
where
    E: StarkEngine<SC = crate::SC>,
    VB: VmBuilder<E>,
{
    let d_pk = engine.device().transport_pk_to_device(&*vm_pk.vm_pk);
    let vm = VirtualMachine::new(engine, vm_builder, vm_pk.vm_config.clone(), d_pk)?;
    let cached_program_trace = vm.commit_program_on_device(&exe.program);
    let instance = VmInstance::new(vm, exe, cached_program_trace)?;
    Ok(instance)
}

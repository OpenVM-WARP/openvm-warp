//! [VmExecutor] is the struct that can execute an _arbitrary_ program, provided in the form of a
//! [VmExe](openvm_instructions::exe::VmExe), for a fixed set of OpenVM instructions
//! corresponding to a [VmExecutionConfig].
//! Internally once it is given a program, it will preprocess the program to rewrite it into a more
//! optimized format for runtime execution. This **instance** of the executor will be a separate
//! struct specialized to running a _fixed_ program on different program inputs.
//!
//! [VirtualMachine] will similarly be the struct that has done all the setup so it can
//! execute+prove an arbitrary program for a fixed config - it will internally still hold VmExecutor
use std::{any::TypeId, borrow::Borrow, collections::VecDeque, marker::PhantomData, sync::Arc};

use getset::{Getters, MutGetters, Setters, WithSetters};
use itertools::{zip_eq, Itertools};
use openvm_circuit::system::program::trace::compute_exe_commit;
use openvm_circuit_primitives::utils::next_power_of_two_or_zero;
use openvm_instructions::{
    exe::{SparseMemoryImage, VmExe},
    program::Program,
};
#[cfg(any(debug_assertions, feature = "test-utils", feature = "stark-debug"))]
use openvm_stark_backend::AirRef;
use openvm_stark_backend::{
    keygen::types::{MultiStarkProvingKey, MultiStarkVerifyingKey},
    memory_metering::ProvingMemoryConfig,
    p3_field::{InjectiveMonomial, PrimeCharacteristicRing, PrimeField32, TwoAdicField},
    p3_util::log2_ceil_usize,
    proof::{Proof, TraceVData},
    prover::{
        ColMajorMatrix, CommittedTraceData, DeviceDataTransporter, DeviceMultiStarkProvingKey,
        MatrixDimensions, ProverBackend, ProverDevice, ProvingContext, TraceCommitter,
    },
    verifier::VerifierError,
    Com, StarkEngine, StarkProtocolConfig, Val,
};
use p3_baby_bear::BabyBear;
#[cfg(feature = "rvr")]
use rvr_openvm_lift::ExtensionRegistry;
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info_span, instrument};

#[cfg(feature = "aot")]
use super::aot::AotInstance;
#[cfg(feature = "rvr")]
use super::rvr::{
    bridge::map_rvr_compile_error, build_pc_to_chip, compile, compile_metered,
    compile_metered_cost, compile_metered_segment_boundary, load_compiled_from_path, ChipMapping,
    GuestDebugMap, RunToCompletion, RvrMeteredCostInstance, RvrMeteredInstance,
    RvrMeteredSegmentInstance, RvrPureInstance, SegmentBoundary,
};
use super::{
    execution_mode::{
        ExecutionCtx, MeteredCostCtx, MeteredCtx, MeteredCtxInputs, PreflightCtx, Segment,
        SegmentationLimits,
    },
    hasher::poseidon2::vm_poseidon2_hasher,
    interpreter::InterpretedInstance,
    interpreter_preflight::PreflightInterpretedInstance,
    AirInventoryError, Arena, ChipInventoryError, ExecutionError, ExecutionState, Executor,
    ExecutorInventory, ExecutorInventoryError, MemoryConfig, MeteredExecutor, PreflightExecutor,
    StaticProgramError, SystemConfig, VmBuilder, VmChipComplex, VmCircuitConfig, VmExecState,
    VmExecutionConfig, VmState, BOUNDARY_AIR_ID, CONNECTOR_AIR_ID, MERKLE_AIR_ID, PROGRAM_AIR_ID,
    PROGRAM_CACHED_TRACE_INDEX,
};
#[cfg(feature = "metrics")]
use crate::metrics::emit_opcode_counts;
#[cfg(feature = "perf-metrics")]
use crate::metrics::end_segment_metrics;
use crate::{
    arch::deferral::DeferralState,
    execute_spanned,
    system::{
        connector::{VmConnectorPvs, DEFAULT_SUSPEND_EXIT_CODE},
        memory::{
            merkle::{
                public_values::{UserPublicValuesProof, UserPublicValuesProofError},
                MemoryMerklePvs,
            },
            online::{GuestMemory, TracingMemory},
            AddressMap, DIGEST_WIDTH,
        },
        program::trace::generate_cached_trace,
        SystemChipComplex, SystemRecords, SystemWithFixedTraceHeights,
    },
};

/// Canonical field bound for VM execution/circuit code.
pub const BABYBEAR_S_BOX_DEGREE: u64 = 7;

pub trait VmField: PrimeField32 + InjectiveMonomial<BABYBEAR_S_BOX_DEGREE> {}
impl<T> VmField for T where T: PrimeField32 + InjectiveMonomial<BABYBEAR_S_BOX_DEGREE> {}

#[derive(Error, Debug)]
pub enum GenerationError {
    #[error("unexpected number of arenas: {actual} (expected num_airs={expected})")]
    UnexpectedNumArenas { actual: usize, expected: usize },
    #[error("trace height for air_idx={air_idx} must be fixed to {expected}, actual={actual}")]
    ForceTraceHeightIncorrect {
        air_idx: usize,
        actual: usize,
        expected: usize,
    },
    #[error("trace height of air {air_idx} has height {height} greater than maximum {max_height}")]
    TraceHeightsLimitExceeded {
        air_idx: usize,
        height: usize,
        max_height: usize,
    },
    #[error("trace heights violate linear constraint {constraint_idx} ({value} >= {threshold})")]
    LinearTraceHeightConstraintExceeded {
        constraint_idx: usize,
        value: u64,
        threshold: u32,
    },
}

#[derive(Clone)]
pub struct Streams<F> {
    pub input_stream: VecDeque<Vec<F>>,
    pub hint_stream: VecDeque<F>,
    /// Cached deferred operation inputs and outputs. Each idx corresponds to a
    /// unique function that is constrained outside the VM in its own deferral circuit.
    pub deferrals: Vec<DeferralState>,
}

impl<F> Streams<F> {
    pub fn new(input_stream: impl Into<VecDeque<Vec<F>>>) -> Self {
        Self {
            input_stream: input_stream.into(),
            hint_stream: VecDeque::default(),
            deferrals: Vec::default(),
        }
    }
}

impl<F> Default for Streams<F> {
    fn default() -> Self {
        Self::new(VecDeque::default())
    }
}

impl<F> From<VecDeque<Vec<F>>> for Streams<F> {
    fn from(value: VecDeque<Vec<F>>) -> Self {
        Streams::new(value)
    }
}

impl<F> From<Vec<Vec<F>>> for Streams<F> {
    fn from(value: Vec<Vec<F>>) -> Self {
        Streams::new(value)
    }
}

/// Typedef for [PreflightInterpretedInstance] that is generic in `VC: VmExecutionConfig<F>`
type PreflightInterpretedInstance2<F, VC> =
    PreflightInterpretedInstance<F, <VC as VmExecutionConfig<F>>::Executor>;

/// [VmExecutor] is the struct that can execute an _arbitrary_ program, provided in the form of a
/// [VmExe], for a fixed set of OpenVM instructions corresponding to a [VmExecutionConfig].
/// Internally once it is given a program, it will preprocess the program to rewrite it into a more
/// optimized format for runtime execution. This **instance** of the executor will be a separate
/// struct specialized to running a _fixed_ program on different program inputs.
#[derive(Clone)]
pub struct VmExecutor<F, VC>
where
    VC: VmExecutionConfig<F>,
{
    pub config: VC,
    inventory: Arc<ExecutorInventory<VC::Executor>>,
    phantom: PhantomData<F>,
}

#[repr(i32)]
pub enum ExitCode {
    Success = 0,
    Error = 1,
    Suspended = -1, // Continuations
}

pub struct PreflightExecutionOutput<F, RA> {
    pub system_records: SystemRecords<F>,
    pub record_arenas: Vec<RA>,
    pub to_state: VmState<F, GuestMemory>,
}

impl<F, VC> VmExecutor<F, VC>
where
    VC: VmExecutionConfig<F>,
{
    /// Create a new VM executor with a given config.
    ///
    /// The VM will start with a single segment, which is created from the initial state.
    pub fn new(config: VC) -> Result<Self, ExecutorInventoryError> {
        let inventory = config.create_executors()?;
        Ok(Self {
            config,
            inventory: Arc::new(inventory),
            phantom: PhantomData,
        })
    }
}

impl<F, VC> VmExecutor<F, VC>
where
    VC: VmExecutionConfig<F> + AsRef<SystemConfig>,
{
    pub fn build_metered_ctx(
        &self,
        inputs: MeteredCtxInputs<'_>,
        memory_config: ProvingMemoryConfig,
    ) -> MeteredCtx {
        MeteredCtx::new(inputs, self.config.as_ref(), memory_config)
    }

    pub fn build_metered_cost_ctx(&self, widths: &[usize]) -> MeteredCostCtx {
        MeteredCostCtx::new(widths.to_vec())
    }
}

impl<F, VC> VmExecutor<F, VC>
where
    F: PrimeField32,
    VC: VmExecutionConfig<F>,
    VC::Executor: Executor<F>,
{
    /// Creates an instance of the interpreter specialized for pure execution, without metering, of
    /// the given `exe`.
    ///
    /// For metered execution, use the [`metered_instance`](Self::metered_instance) constructor.
    #[cfg(all(not(feature = "aot"), not(feature = "rvr")))]
    pub fn instance(
        &self,
        exe: &VmExe<F>,
    ) -> Result<InterpretedInstance<'_, F, ExecutionCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_pure", backend = "interpreter").entered();
        InterpretedInstance::new(&self.inventory, exe)
    }

    #[cfg(any(feature = "aot", feature = "rvr"))]
    pub fn interpreter_instance(
        &self,
        exe: &VmExe<F>,
    ) -> Result<InterpretedInstance<'_, F, ExecutionCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_pure", backend = "interpreter").entered();
        InterpretedInstance::new(&self.inventory, exe)
    }

    #[cfg(feature = "aot")]
    pub fn instance(
        &self,
        exe: &VmExe<F>,
    ) -> Result<AotInstance<'_, F, ExecutionCtx>, StaticProgramError> {
        Self::aot_instance(self, exe)
    }

    #[cfg(feature = "rvr")]
    pub fn instance(&self, exe: &VmExe<F>) -> Result<RvrPureInstance<'_, F>, StaticProgramError> {
        Self::rvr_instance(self, exe, None)
    }
}

#[cfg(feature = "rvr")]
impl<F, VC> VmExecutor<F, VC>
where
    F: PrimeField32,
    VC: VmExecutionConfig<F>,
{
    fn build_rvr_extensions(
        &self,
        executor_idx_to_air_idx: Option<&[usize]>,
    ) -> ExtensionRegistry<F> {
        self.config.create_rvr_extensions(executor_idx_to_air_idx)
    }
}

#[cfg(feature = "rvr")]
impl<F, VC> VmExecutor<F, VC>
where
    F: PrimeField32,
    VC: VmExecutionConfig<F>,
    VC::Executor: Executor<F>,
{
    pub fn rvr_instance(
        &self,
        exe: &VmExe<F>,
        guest_debug_map: Option<&GuestDebugMap>,
    ) -> Result<RvrPureInstance<'_, F>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span = tracing::info_span!("compile_pure", backend = "rvr").entered();
        let extensions = self.build_rvr_extensions(None);
        let compiled = compile(exe, &extensions, guest_debug_map).map_err(map_rvr_compile_error)?;

        Ok(RvrPureInstance {
            system_config: self.inventory.config(),
            exe: Arc::new(exe.clone()),
            compiled,
            extensions,
        })
    }

    /// Load a previously saved pure-mode artifact and return a ready-to-execute
    /// [`RvrPureInstance`]. The caller is responsible for supplying the
    /// matching `exe`; no compatibility validation is performed.
    pub fn load_instance(
        &self,
        lib_path: &std::path::Path,
        exe: &VmExe<F>,
    ) -> Result<RvrPureInstance<'_, F>, StaticProgramError> {
        let extensions = self.build_rvr_extensions(None);
        let compiled = load_compiled_from_path(lib_path).map_err(map_rvr_compile_error)?;

        Ok(RvrPureInstance {
            system_config: self.inventory.config(),
            exe: Arc::new(exe.clone()),
            compiled,
            extensions,
        })
    }
}

#[cfg(feature = "aot")]
impl<F, VC> VmExecutor<F, VC>
where
    F: PrimeField32,
    VC: VmExecutionConfig<F>,
    VC::Executor: Executor<F>,
{
    pub fn aot_instance(
        &self,
        exe: &VmExe<F>,
    ) -> Result<AotInstance<'_, F, ExecutionCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span = tracing::info_span!("compile_pure", backend = "aot").entered();
        AotInstance::new(&self.inventory, exe)
    }
}

impl<F, VC> VmExecutor<F, VC>
where
    F: PrimeField32,
    VC: VmExecutionConfig<F>,
    VC::Executor: MeteredExecutor<F>,
{
    /// Creates an instance of the interpreter specialized for metered execution of the given `exe`.
    #[cfg(all(not(feature = "aot"), not(feature = "rvr")))]
    pub fn metered_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<InterpretedInstance<'_, F, MeteredCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_metered", backend = "interpreter").entered();
        InterpretedInstance::new_metered(&self.inventory, exe, executor_idx_to_air_idx)
    }

    #[cfg(any(feature = "aot", feature = "rvr"))]
    pub fn metered_interpreter_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<InterpretedInstance<'_, F, MeteredCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_metered", backend = "interpreter").entered();
        InterpretedInstance::new_metered(&self.inventory, exe, executor_idx_to_air_idx)
    }

    #[cfg(feature = "aot")]
    pub fn metered_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<AotInstance<'_, F, MeteredCtx>, StaticProgramError> {
        Self::metered_aot_instance(self, exe, executor_idx_to_air_idx)
    }

    // Crates an AOT instance for metered execution of the given `exe`.
    #[cfg(feature = "aot")]
    pub fn metered_aot_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<AotInstance<'_, F, MeteredCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span = tracing::info_span!("compile_metered", backend = "aot").entered();
        AotInstance::new_metered(&self.inventory, exe, executor_idx_to_air_idx)
    }

    /// Creates an instance of the interpreter specialized for cost metering execution of the given
    /// `exe`.
    #[cfg(not(feature = "rvr"))]
    pub fn metered_cost_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<InterpretedInstance<'_, F, MeteredCostCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_metered_cost", backend = "interpreter").entered();
        InterpretedInstance::new_metered(&self.inventory, exe, executor_idx_to_air_idx)
    }

    #[cfg(feature = "rvr")]
    pub fn metered_cost_interpreter_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<InterpretedInstance<'_, F, MeteredCostCtx>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_metered_cost", backend = "interpreter").entered();
        InterpretedInstance::new_metered(&self.inventory, exe, executor_idx_to_air_idx)
    }
}

#[cfg(feature = "rvr")]
impl<F, VC> VmExecutor<F, VC>
where
    F: PrimeField32,
    VC: VmExecutionConfig<F>,
    VC::Executor: MeteredExecutor<F>,
{
    pub fn metered_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<RvrMeteredInstance<'_, F>, StaticProgramError> {
        self.metered_rvr_instance(exe, executor_idx_to_air_idx, None)
    }

    pub fn metered_rvr_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
        guest_debug_map: Option<&GuestDebugMap>,
    ) -> Result<RvrMeteredInstance<'_, F>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span = tracing::info_span!("compile_metered", backend = "rvr").entered();
        let extensions = self.build_rvr_extensions(Some(executor_idx_to_air_idx));
        let chips = ChipMapping {
            pc_to_chip: build_pc_to_chip(exe, &self.inventory, executor_idx_to_air_idx)
                .map_err(map_rvr_compile_error)?,
            chip_widths: None,
        };
        let compiled = compile_metered(exe, &extensions, &chips, guest_debug_map)
            .map_err(map_rvr_compile_error)?;

        Ok(RvrMeteredInstance {
            system_config: self.inventory.config(),
            exe: Arc::new(exe.clone()),
            extensions,
            compiled,
            _mode: PhantomData::<RunToCompletion>,
        })
    }

    pub fn metered_segment_rvr_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
        guest_debug_map: Option<&GuestDebugMap>,
    ) -> Result<RvrMeteredSegmentInstance<'_, F>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_metered_segment", backend = "rvr").entered();
        let extensions = self.build_rvr_extensions(Some(executor_idx_to_air_idx));
        let chips = ChipMapping {
            pc_to_chip: build_pc_to_chip(exe, &self.inventory, executor_idx_to_air_idx)
                .map_err(map_rvr_compile_error)?,
            chip_widths: None,
        };
        let compiled = compile_metered_segment_boundary(exe, &extensions, &chips, guest_debug_map)
            .map_err(map_rvr_compile_error)?;

        Ok(RvrMeteredSegmentInstance {
            system_config: self.inventory.config(),
            exe: Arc::new(exe.clone()),
            extensions,
            compiled,
            _mode: PhantomData::<SegmentBoundary>,
        })
    }

    pub fn metered_cost_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
        widths: &[usize],
    ) -> Result<RvrMeteredCostInstance<'_, F>, StaticProgramError> {
        self.metered_cost_rvr_instance(exe, executor_idx_to_air_idx, widths, None)
    }

    /// Load a previously saved metered-mode artifact. Caller supplies `exe` and
    /// `executor_idx_to_air_idx`. No compatibility validation is performed.
    pub fn load_metered_instance(
        &self,
        lib_path: &std::path::Path,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
    ) -> Result<RvrMeteredInstance<'_, F>, StaticProgramError> {
        let extensions = self.build_rvr_extensions(Some(executor_idx_to_air_idx));
        let compiled = load_compiled_from_path(lib_path).map_err(map_rvr_compile_error)?;

        Ok(RvrMeteredInstance {
            system_config: self.inventory.config(),
            exe: Arc::new(exe.clone()),
            extensions,
            compiled,
            _mode: PhantomData::<RunToCompletion>,
        })
    }

    /// Load a previously saved metered-cost-mode artifact. Caller supplies `exe`,
    /// `executor_idx_to_air_idx`, and `widths`. No compatibility validation is performed.
    pub fn load_metered_cost_instance(
        &self,
        lib_path: &std::path::Path,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
        widths: &[usize],
    ) -> Result<RvrMeteredCostInstance<'_, F>, StaticProgramError> {
        let extensions = self.build_rvr_extensions(Some(executor_idx_to_air_idx));
        let widths: Vec<u64> = widths.iter().map(|&w| w as u64).collect();
        let compiled = load_compiled_from_path(lib_path).map_err(map_rvr_compile_error)?;

        Ok(RvrMeteredCostInstance {
            system_config: self.inventory.config(),
            exe: Arc::new(exe.clone()),
            extensions,
            compiled,
            widths,
        })
    }

    pub fn metered_cost_rvr_instance(
        &self,
        exe: &VmExe<F>,
        executor_idx_to_air_idx: &[usize],
        widths: &[usize],
        guest_debug_map: Option<&GuestDebugMap>,
    ) -> Result<RvrMeteredCostInstance<'_, F>, StaticProgramError> {
        #[cfg(feature = "metrics")]
        let _compilation_span =
            tracing::info_span!("compile_metered_cost", backend = "rvr").entered();
        let extensions = self.build_rvr_extensions(Some(executor_idx_to_air_idx));
        let widths: Vec<u64> = widths.iter().map(|&w| w as u64).collect();
        let chips = ChipMapping {
            pc_to_chip: build_pc_to_chip(exe, &self.inventory, executor_idx_to_air_idx)
                .map_err(map_rvr_compile_error)?,
            chip_widths: Some(widths.clone()),
        };
        let compiled = compile_metered_cost(exe, &extensions, &chips, guest_debug_map)
            .map_err(map_rvr_compile_error)?;

        Ok(RvrMeteredCostInstance {
            system_config: self.inventory.config(),
            exe: Arc::new(exe.clone()),
            extensions,
            compiled,
            widths,
        })
    }
}

#[derive(Error, Debug)]
pub enum VmVerificationError<SC: StarkProtocolConfig> {
    #[error("no proof is provided")]
    ProofNotFound,

    #[error("program commit mismatch (index of mismatch proof: {index}")]
    ProgramCommitMismatch { index: usize },

    #[error("exe commit mismatch (expected: {expected:?}, actual: {actual:?})")]
    ExeCommitMismatch {
        expected: [u32; DIGEST_WIDTH],
        actual: [u32; DIGEST_WIDTH],
    },

    #[error("initial pc mismatch (initial: {initial}, prev_final: {prev_final})")]
    InitialPcMismatch { initial: u32, prev_final: u32 },

    #[error("initial memory root mismatch")]
    InitialMemoryRootMismatch,

    #[error("is terminate mismatch (expected: {expected}, actual: {actual})")]
    IsTerminateMismatch { expected: bool, actual: bool },

    #[error("exit code mismatch")]
    ExitCodeMismatch { expected: u32, actual: u32 },

    #[error("AIR has unexpected public values (expected: {expected}, actual: {actual})")]
    UnexpectedPvs { expected: usize, actual: usize },

    #[error("Invalid number of AIRs: expected at least 3, got {0}")]
    NotEnoughAirs(usize),

    #[error("missing system AIR with ID {air_id}")]
    SystemAirMissing { air_id: usize },

    #[error("invalid segment metadata: {0}")]
    InvalidSegmentMetadata(&'static str),

    #[error("stark verification error: {0}")]
    StarkError(#[from] VerifierError<SC::EF>),

    #[error("user public values proof error: {0}")]
    UserPublicValuesError(#[from] UserPublicValuesProofError),
}

#[derive(Error, Debug)]
pub enum VirtualMachineError {
    #[error("executor inventory error: {0}")]
    ExecutorInventory(#[from] ExecutorInventoryError),
    #[error("air inventory error: {0}")]
    AirInventory(#[from] AirInventoryError),
    #[error("chip inventory error: {0}")]
    ChipInventory(#[from] ChipInventoryError),
    #[error("static program error: {0}")]
    StaticProgram(#[from] StaticProgramError),
    #[error("execution error: {0}")]
    Execution(#[from] ExecutionError),
    #[error("trace generation error: {0}")]
    Generation(#[from] GenerationError),
    #[error("program committed trade data not loaded")]
    ProgramIsNotCommitted,
    #[error("invalid prepared native WARP plan: {0}")]
    NativeWarpPlan(String),
}

/// The [VirtualMachine] struct contains the API to generate proofs for _arbitrary_ programs for a
/// fixed set of OpenVM instructions and a fixed VM circuit corresponding to those instructions. The
/// API is specific to a particular [StarkEngine], which specifies a fixed [StarkProtocolConfig] and
/// [ProverBackend] via associated types. The [VmBuilder] also fixes the choice of
/// `RecordArena` associated to the prover backend via an associated type.
///
/// In other words, this struct _is_ the zkVM.
#[derive(Getters, MutGetters, Setters, WithSetters)]
pub struct VirtualMachine<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    /// Proving engine
    pub engine: E,
    /// Runtime executor
    #[getset(get = "pub")]
    executor: VmExecutor<Val<E::SC>, VB::VmConfig>,
    #[getset(get = "pub", get_mut = "pub")]
    pk: DeviceMultiStarkProvingKey<E::PB>,
    chip_complex: VmChipComplex<E::SC, VB::RecordArena, E::PB, VB::SystemChipInventory>,
}

impl<E, VB> VirtualMachine<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    pub fn new(
        engine: E,
        builder: VB,
        config: VB::VmConfig,
        d_pk: DeviceMultiStarkProvingKey<E::PB>,
    ) -> Result<Self, VirtualMachineError> {
        let circuit = config.create_airs()?;
        let chip_complex =
            builder.create_chip_complex(&config, circuit, engine.device().device_ctx())?;
        let executor = VmExecutor::<Val<E::SC>, _>::new(config)?;
        Ok(Self {
            engine,
            executor,
            pk: d_pk,
            chip_complex,
        })
    }

    pub fn new_with_keygen(
        engine: E,
        builder: VB,
        config: VB::VmConfig,
    ) -> Result<(Self, MultiStarkProvingKey<E::SC>), VirtualMachineError> {
        let circuit = config.create_airs()?;
        let pk = circuit.keygen(engine.config());
        let _vk = pk.get_vk();
        let d_pk = engine.device().transport_pk_to_device(&pk);
        let vm = Self::new(engine, builder, config, d_pk)?;
        Ok((vm, pk))
    }

    pub fn config(&self) -> &VB::VmConfig {
        &self.executor.config
    }

    /// Pure execution instance.
    #[cfg(all(not(feature = "aot"), not(feature = "rvr")))]
    pub fn instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<InterpretedInstance<'_, Val<E::SC>, ExecutionCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>,
    {
        self.executor().instance(exe)
    }

    #[cfg(feature = "rvr")]
    pub fn instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<RvrPureInstance<'_, Val<E::SC>>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>,
    {
        Self::get_rvr_instance(self, exe)
    }

    #[cfg(feature = "rvr")]
    pub fn get_rvr_instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<RvrPureInstance<'_, Val<E::SC>>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>,
    {
        self.executor().rvr_instance(exe, None)
    }

    // Pure AOT / RVR execution
    #[cfg(any(feature = "aot", feature = "rvr"))]
    pub fn interpreter_instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<InterpretedInstance<'_, Val<E::SC>, ExecutionCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>,
    {
        self.executor().interpreter_instance(exe)
    }

    // Pure AOT execution
    #[cfg(feature = "aot")]
    pub fn instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<AotInstance<'_, Val<E::SC>, ExecutionCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>,
    {
        Self::get_aot_instance(self, exe)
    }

    #[cfg(feature = "aot")]
    pub fn get_aot_instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<AotInstance<'_, Val<E::SC>, ExecutionCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>,
    {
        self.executor().aot_instance(exe)
    }

    #[cfg(all(not(feature = "aot"), not(feature = "rvr")))]
    pub fn metered_instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<InterpretedInstance<'_, Val<E::SC>, MeteredCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor()
            .metered_instance(exe, &executor_idx_to_air_idx)
    }

    #[cfg(feature = "rvr")]
    pub fn metered_instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<RvrMeteredInstance<'_, Val<E::SC>>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor()
            .metered_instance(exe, &executor_idx_to_air_idx)
    }

    #[cfg(feature = "rvr")]
    pub fn get_metered_rvr_instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<RvrMeteredInstance<'_, Val<E::SC>>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor()
            .metered_rvr_instance(exe, &executor_idx_to_air_idx, None)
    }

    #[cfg(feature = "rvr")]
    pub fn get_metered_segment_rvr_instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<RvrMeteredSegmentInstance<'_, Val<E::SC>>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor()
            .metered_segment_rvr_instance(exe, &executor_idx_to_air_idx, None)
    }

    #[cfg(feature = "rvr")]
    pub fn load_metered_instance(
        &self,
        lib_path: &std::path::Path,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<RvrMeteredInstance<'_, Val<E::SC>>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor()
            .load_metered_instance(lib_path, exe, &executor_idx_to_air_idx)
    }

    #[cfg(feature = "aot")]
    pub fn metered_instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<AotInstance<'_, Val<E::SC>, MeteredCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor()
            .metered_instance(exe, &executor_idx_to_air_idx)
    }

    // Metered AOT execution
    #[cfg(feature = "aot")]
    pub fn get_metered_aot_instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<AotInstance<'_, Val<E::SC>, MeteredCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor()
            .metered_aot_instance(exe, &executor_idx_to_air_idx)
    }

    #[cfg(any(feature = "aot", feature = "rvr"))]
    pub fn metered_interpreter_instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<InterpretedInstance<'_, Val<E::SC>, MeteredCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor()
            .metered_interpreter_instance(exe, &executor_idx_to_air_idx)
    }

    #[cfg(not(feature = "rvr"))]
    pub fn metered_cost_instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<InterpretedInstance<'_, Val<E::SC>, MeteredCostCtx>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        self.executor()
            .metered_cost_instance(exe, &executor_idx_to_air_idx)
    }

    #[cfg(feature = "rvr")]
    pub fn metered_cost_instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<RvrMeteredCostInstance<'_, Val<E::SC>>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        Self::get_metered_cost_rvr_instance(self, exe)
    }

    #[cfg(feature = "rvr")]
    pub fn get_metered_cost_rvr_instance(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<RvrMeteredCostInstance<'_, Val<E::SC>>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        let widths: Vec<usize> = self
            .pk
            .per_air
            .iter()
            .map(|pk| pk.vk.params.width.total_width())
            .collect();
        self.executor()
            .metered_cost_rvr_instance(exe, &executor_idx_to_air_idx, &widths, None)
    }

    #[cfg(feature = "rvr")]
    pub fn load_metered_cost_instance(
        &self,
        lib_path: &std::path::Path,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<RvrMeteredCostInstance<'_, Val<E::SC>>, StaticProgramError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: MeteredExecutor<Val<E::SC>>,
    {
        let executor_idx_to_air_idx = self.executor_idx_to_air_idx();
        let widths: Vec<usize> = self
            .pk
            .per_air
            .iter()
            .map(|pk| pk.vk.params.width.total_width())
            .collect();
        self.executor()
            .load_metered_cost_instance(lib_path, exe, &executor_idx_to_air_idx, &widths)
    }

    pub fn preflight_interpreter(
        &self,
        exe: &VmExe<Val<E::SC>>,
    ) -> Result<PreflightInterpretedInstance2<Val<E::SC>, VB::VmConfig>, StaticProgramError> {
        PreflightInterpretedInstance::new(
            &exe.program,
            self.executor.inventory.clone(),
            self.executor_idx_to_air_idx(),
        )
    }

    /// Preflight execution for a single segment. Executes for exactly `num_insns` instructions
    /// using an interpreter. Preflight execution must be provided with `trace_heights`
    /// instrumentation data that was collected from a previous run of metered execution so that the
    /// preflight execution knows how much memory to allocate for record arenas.
    ///
    /// This function should rarely be called on its own. Users are advised to call
    /// [`prove`](Self::prove) directly.
    #[instrument(name = "execute_preflight", skip_all)]
    pub fn execute_preflight(
        &self,
        interpreter: &mut PreflightInterpretedInstance2<Val<E::SC>, VB::VmConfig>,
        state: VmState<Val<E::SC>, GuestMemory>,
        num_insns: Option<u64>,
        trace_heights: &[u32],
    ) -> Result<PreflightExecutionOutput<Val<E::SC>, VB::RecordArena>, ExecutionError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor:
            PreflightExecutor<Val<E::SC>, VB::RecordArena>,
    {
        debug_assert!(interpreter
            .executor_idx_to_air_idx
            .iter()
            .all(|&air_idx| air_idx < trace_heights.len()));

        // TODO[jpw]: figure out how to compute RA specific main_widths
        let main_widths = self
            .pk
            .per_air
            .iter()
            .map(|pk| pk.vk.params.width.main_width())
            .collect_vec();
        let capacities = zip_eq(trace_heights, main_widths)
            .map(|(&h, w)| (h as usize, w))
            .collect::<Vec<_>>();
        let ctx = PreflightCtx::new_with_capacity(&capacities, num_insns);

        let pc = state.pc();
        let memory = TracingMemory::from_image(state.memory);
        let from_state = ExecutionState::new(pc, memory.timestamp());
        let vm_state = VmState::new(
            pc,
            memory,
            state.streams,
            state.rng,
            #[cfg(feature = "metrics")]
            state.metrics,
        );
        let mut exec_state = VmExecState::new(vm_state, ctx);
        interpreter.reset_execution_frequencies();
        execute_spanned!("execute_preflight", interpreter, &mut exec_state)?;
        let filtered_exec_frequencies = interpreter.filtered_execution_frequencies();
        #[cfg(feature = "metrics")]
        emit_opcode_counts(
            &exec_state.vm_state.metrics,
            interpreter.opcode_counts_by_air::<VB::RecordArena>(),
        );
        let touched_memory = exec_state.vm_state.memory.finalize::<Val<E::SC>>();
        // Grow the touched-page sets on the carried-forward memory image so the next segment's
        // host-to-device transfer (`set_initial_memory`) stays a correct superset of non-zero
        // pages.
        exec_state
            .vm_state
            .memory
            .data
            .memory
            .extend_touched_pages_from_touched(&touched_memory);
        #[cfg(feature = "perf-metrics")]
        end_segment_metrics(&mut exec_state);

        let pc = exec_state.vm_state.pc();
        let memory = exec_state.vm_state.memory;
        let to_state = ExecutionState::new(pc, memory.timestamp());
        let exit_code = exec_state.exit_code?;
        let system_records = SystemRecords {
            from_state,
            to_state,
            exit_code,
            filtered_exec_frequencies,
            touched_memory,
        };
        let record_arenas = exec_state.ctx.arenas;
        let to_state = VmState::new(
            pc,
            memory.data,
            exec_state.vm_state.streams,
            exec_state.vm_state.rng,
            #[cfg(feature = "metrics")]
            exec_state.vm_state.metrics,
        );
        Ok(PreflightExecutionOutput {
            system_records,
            record_arenas,
            to_state,
        })
    }

    /// Calls [`VmState::initial`] but sets more information for
    /// performance metrics when feature "perf-metrics" is enabled.
    #[instrument(name = "vm.create_initial_state", level = "debug", skip_all)]
    pub fn create_initial_state(
        &self,
        exe: &VmExe<Val<E::SC>>,
        inputs: impl Into<Streams<Val<E::SC>>>,
    ) -> VmState<Val<E::SC>, GuestMemory> {
        #[allow(unused_mut)]
        let mut state = VmState::initial(
            self.config().as_ref(),
            &exe.init_memory,
            exe.pc_start,
            inputs,
        );
        // Add backtrace information for either:
        // - debugging
        // - performance metrics
        #[cfg(all(feature = "metrics", any(feature = "perf-metrics", debug_assertions)))]
        {
            state.metrics.fn_bounds = exe.fn_bounds.clone();
            state.metrics.debug_infos = exe.program.debug_infos();
        }
        #[cfg(feature = "metrics")]
        {
            state.metrics.set_pk_air_names(&self.pk);
        }
        #[cfg(feature = "perf-metrics")]
        {
            state.metrics.set_pk_trace_info(&self.pk);
            state.metrics.num_sys_airs = self.config().as_ref().num_airs();
        }
        state
    }

    /// This function mutates `self` but should only depend on internal state in the sense that:
    /// - program must already be loaded as cached trace via [`load_program`](Self::load_program).
    /// - initial memory image was already sent to device via
    ///   [`transport_init_memory_to_device`](Self::transport_init_memory_to_device).
    /// - all other state should be given by `system_records` and `record_arenas`
    #[instrument(name = "trace_gen", skip_all)]
    pub fn generate_proving_ctx(
        &mut self,
        system_records: SystemRecords<Val<E::SC>>,
        record_arenas: Vec<VB::RecordArena>,
    ) -> Result<ProvingContext<E::PB>, GenerationError> {
        // main tracegen call:
        let ctx = self
            .chip_complex
            .generate_proving_ctx(system_records, record_arenas)?;

        // ==== Defensive checks that the trace heights satisfy the linear constraints: ====
        let idx_trace_heights = ctx
            .per_trace
            .iter()
            .map(|(air_idx, ctx)| (*air_idx, ctx.common_main.height()))
            .collect_vec();
        // 1. check max trace height isn't exceeded
        let max_trace_height = if TypeId::of::<Val<E::SC>>() == TypeId::of::<BabyBear>() {
            let min_log_blowup = log2_ceil_usize(self.config().as_ref().max_constraint_degree - 1);
            1 << (BabyBear::TWO_ADICITY - min_log_blowup)
        } else {
            tracing::warn!(
                "constructing VirtualMachine for unrecognized field; using max_trace_height=2^30"
            );
            1 << 30
        };
        if let Some(&(air_idx, height)) = idx_trace_heights
            .iter()
            .find(|(_, height)| *height > max_trace_height)
        {
            return Err(GenerationError::TraceHeightsLimitExceeded {
                air_idx,
                height,
                max_height: max_trace_height,
            });
        }
        // 2. check linear constraints on trace heights are satisfied
        let trace_height_constraints = &self.pk.trace_height_constraints;
        if trace_height_constraints.is_empty() {
            tracing::warn!("generating proving context without trace height constraints");
        }
        for (i, constraint) in trace_height_constraints.iter().enumerate() {
            let value = idx_trace_heights
                .iter()
                .map(|&(air_idx, h)| constraint.coefficients[air_idx] as u64 * h as u64)
                .sum::<u64>();

            if value >= constraint.threshold as u64 {
                tracing::info!(
                    "trace heights {:?} violate linear constraint {} ({} >= {})",
                    idx_trace_heights,
                    i,
                    value,
                    constraint.threshold
                );
                return Err(GenerationError::LinearTraceHeightConstraintExceeded {
                    constraint_idx: i,
                    value,
                    threshold: constraint.threshold,
                });
            }
        }
        #[cfg(feature = "stark-debug")]
        self.debug_proving_ctx(&ctx);

        Ok(ctx)
    }

    /// Generates proof for zkVM execution for exactly `num_insns` instructions for a given program
    /// and a given starting state.
    ///
    /// **Note**: The cached program trace must be loaded via [`load_program`](Self::load_program)
    /// before calling this function.
    ///
    /// Returns:
    /// - proof for the execution segment
    /// - final memory state only if execution ends in successful termination (exit code 0). This
    ///   final memory state may be used to extract user public values afterwards.
    pub fn prove(
        &mut self,
        interpreter: &mut PreflightInterpretedInstance2<Val<E::SC>, VB::VmConfig>,
        state: VmState<Val<E::SC>, GuestMemory>,
        num_insns: Option<u64>,
        trace_heights: &[u32],
    ) -> Result<(Proof<E::SC>, Option<GuestMemory>), VirtualMachineError>
    where
        Val<E::SC>: PrimeField32,
        <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor:
            PreflightExecutor<Val<E::SC>, VB::RecordArena>,
    {
        self.transport_init_memory_to_device(&state.memory);

        let PreflightExecutionOutput {
            system_records,
            record_arenas,
            to_state,
        } = self.execute_preflight(interpreter, state, num_insns, trace_heights)?;
        // drop final memory unless this is a terminal segment and the exit code is success
        let final_memory =
            (system_records.exit_code == Some(ExitCode::Success as u32)).then_some(to_state.memory);
        let ctx = self.generate_proving_ctx(system_records, record_arenas)?;
        let proof = self.engine.prove(&self.pk, ctx).unwrap();

        Ok((proof, final_memory))
    }

    /// Transforms the program into a cached trace and commits it _on device_ using the proof system
    /// polynomial commitment scheme.
    ///
    /// Returns the cached program trace.
    /// Note that [`load_program`](Self::load_program) must be called separately to load the cached
    /// program trace into the VM itself.
    pub fn commit_program_on_device(
        &self,
        program: &Program<Val<E::SC>>,
    ) -> CommittedTraceData<E::PB> {
        let rm_trace = generate_cached_trace(program);
        let cm_trace = ColMajorMatrix::from_row_major(&rm_trace);
        let d_trace = self.engine.device().transport_matrix_to_device(&cm_trace);
        let (commitment, pcs) = self
            .engine
            .device()
            .commit(std::slice::from_ref(&&d_trace))
            .unwrap();
        CommittedTraceData {
            commitment,
            trace: d_trace,
            data: Arc::new(pcs),
        }
    }

    /// Loads cached program trace into the VM.
    pub fn load_program(&mut self, cached_program_trace: CommittedTraceData<E::PB>) {
        self.chip_complex.system.load_program(cached_program_trace);
    }

    /// Borrow the program PCS allocation loaded at VM construction. This is
    /// the same allocation used by every segment proof.
    pub fn cached_program_trace(&self) -> Option<&CommittedTraceData<E::PB>> {
        self.chip_complex.system.cached_program_trace()
    }

    #[instrument(name = "vm.transport_init_memory", skip_all)]
    pub fn transport_init_memory_to_device(&mut self, memory: &GuestMemory) {
        self.chip_complex
            .system
            .transport_init_memory_to_device(memory);
    }

    /// See [`SystemChipComplex::memory_top_tree`].
    pub fn memory_top_tree(&self) -> Option<&[[Val<E::SC>; DIGEST_WIDTH]]> {
        self.chip_complex.system.memory_top_tree()
    }

    pub fn executor_idx_to_air_idx(&self) -> Vec<usize> {
        let ret = self.chip_complex.inventory.executor_idx_to_air_idx();
        tracing::debug!("executor_idx_to_air_idx: {:?}", ret);
        assert_eq!(self.executor().inventory.executors().len(), ret.len());
        ret
    }

    /// Convenience method to construct a [MeteredCtx] using data from the stored proving key.
    pub fn build_metered_ctx(&self, exe: &VmExe<Val<E::SC>>) -> MeteredCtx
    where
        Val<E::SC>: PrimeField32,
    {
        let program_len = exe.program.num_defined_instructions();

        let (mut constant_trace_heights, air_names, widths, interactions, need_rot): (
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
            Vec<_>,
        ) = self
            .pk
            .per_air
            .iter()
            .map(|pk| {
                let constant_trace_height = pk.preprocessed_data.as_ref().map(|cd| cd.height());
                let air_names = pk.air_name.clone();
                let width = pk.vk.params.width.total_width();
                let num_interactions = pk.vk.symbolic_constraints.interactions.len();
                let need_rot = pk.vk.params.need_rot;
                (
                    constant_trace_height,
                    air_names,
                    width,
                    num_interactions,
                    need_rot,
                )
            })
            .multiunzip();

        // Program trace is the same for all segments
        constant_trace_heights[PROGRAM_AIR_ID] = Some(program_len);
        // VmConnectorAir always has a constant trace height of 2
        constant_trace_heights[CONNECTOR_AIR_ID] = Some(2);
        // Merge in constant heights reported by chips (e.g., lookup table chips).
        for (air_id, chip_height) in self
            .chip_complex
            .inventory
            .constant_trace_heights()
            .into_iter()
            .enumerate()
        {
            if constant_trace_heights[air_id].is_none() {
                constant_trace_heights[air_id] = chip_height;
            }
        }

        let log_stacked_height = self
            .engine
            .params()
            .log_stacked_height()
            .try_into()
            .expect("log_stacked_height must fit in u8");
        self.executor().build_metered_ctx(
            MeteredCtxInputs {
                constant_trace_heights: &constant_trace_heights,
                air_names: &air_names,
                widths: &widths,
                interactions: &interactions,
                need_rot: &need_rot,
                segmentation_limits: SegmentationLimits {
                    max_trace_height_bits: log_stacked_height,
                    max_trace_cells: self.config().as_ref().segmentation_max_trace_cells,
                    max_memory: self.config().as_ref().segmentation_max_memory,
                    max_interactions: <Val<E::SC> as PrimeField32>::ORDER_U32,
                },
            },
            self.engine.proving_memory_config(),
        )
    }

    /// Convenience method to construct a [MeteredCostCtx] using data from the stored proving key.
    pub fn build_metered_cost_ctx(&self) -> MeteredCostCtx {
        let widths: Vec<_> = self
            .pk
            .per_air
            .iter()
            .map(|pk| pk.vk.params.width.total_width())
            .collect();

        self.executor().build_metered_cost_ctx(&widths)
    }

    pub fn num_airs(&self) -> usize {
        let num_airs = self.pk.per_air.len();
        debug_assert_eq!(num_airs, self.chip_complex.inventory.airs().num_airs());
        num_airs
    }

    pub fn air_names(&self) -> impl Iterator<Item = &'_ str> {
        self.pk.per_air.iter().map(|pk| pk.air_name.as_str())
    }

    /// See [`debug_proving_ctx`].
    #[cfg(feature = "stark-debug")]
    pub fn debug_proving_ctx(&mut self, ctx: &ProvingContext<E::PB>) {
        debug_proving_ctx(self, ctx);
    }
}

#[cfg(test)]
mod tests {
    use openvm_stark_backend::p3_field::PrimeCharacteristicRing;
    use openvm_stark_sdk::config::baby_bear_poseidon2::F;

    use super::{
        bucket_native_warp_trace_heights, verify_segment_metadata_sequence, SystemConfig,
        VirtualMachine, VmSegmentMetadata, BOUNDARY_AIR_PRESENT, CONNECTOR_AIR_ID,
        CONNECTOR_AIR_PRESENT, MERKLE_AIR_PRESENT, PROGRAM_AIR_ID, PROGRAM_AIR_PRESENT,
    };
    use crate::{arch::testing::TestSC, system::SystemCpuBuilder, utils::test_cpu_engine};

    const ALL_SYSTEM_AIRS: u8 =
        PROGRAM_AIR_PRESENT | CONNECTOR_AIR_PRESENT | BOUNDARY_AIR_PRESENT | MERKLE_AIR_PRESENT;

    fn digest(value: u32) -> [F; super::DIGEST_WIDTH] {
        [F::from_u32(value); super::DIGEST_WIDTH]
    }

    fn segment(
        program: u32,
        initial_pc: u32,
        final_pc: u32,
        initial_memory: u32,
        final_memory: u32,
        terminate: bool,
    ) -> VmSegmentMetadata<TestSC> {
        VmSegmentMetadata {
            program_commit: digest(program),
            initial_pc: F::from_u32(initial_pc),
            final_pc: F::from_u32(final_pc),
            exit_code: F::from_u32(if terminate {
                super::ExitCode::Success as u32
            } else {
                super::DEFAULT_SUSPEND_EXIT_CODE
            }),
            is_terminate: F::from_bool(terminate),
            initial_memory_root: digest(initial_memory),
            final_memory_root: digest(final_memory),
            present_system_airs: ALL_SYSTEM_AIRS,
        }
    }

    fn valid_segment_chain() -> Vec<VmSegmentMetadata<TestSC>> {
        vec![
            segment(7, 0, 8, 10, 11, false),
            segment(7, 8, 16, 11, 12, false),
            segment(7, 16, 24, 12, 13, true),
        ]
    }

    #[test]
    fn keygen_marks_required_airs_for_continuations() {
        let engine = test_cpu_engine();
        let config = SystemConfig::default();
        let merkle_air_id = config.memory_merkle_air_id();
        let boundary_air_id = config.memory_boundary_air_id();

        let (_vm, pk) = VirtualMachine::new_with_keygen(engine, SystemCpuBuilder, config).unwrap();

        assert!(pk.per_air[PROGRAM_AIR_ID].vk.is_required);
        assert!(pk.per_air[CONNECTOR_AIR_ID].vk.is_required);
        assert!(pk.per_air[merkle_air_id].vk.is_required);
        assert!(pk.per_air[boundary_air_id].vk.is_required);
    }

    #[test]
    fn segment_metadata_sequence_accepts_valid_chain() {
        assert!(verify_segment_metadata_sequence(&valid_segment_chain()).is_ok());
    }

    #[test]
    fn segment_metadata_sequence_rejects_reordering_removal_and_duplication() {
        let valid = valid_segment_chain();

        let mut reordered = valid.clone();
        reordered.swap(0, 1);
        assert!(verify_segment_metadata_sequence(&reordered).is_err());

        let removed = vec![valid[0].clone(), valid[2].clone()];
        assert!(verify_segment_metadata_sequence(&removed).is_err());

        let duplicated = vec![valid[0].clone(), valid[1].clone(), valid[1].clone()];
        assert!(verify_segment_metadata_sequence(&duplicated).is_err());
    }

    #[test]
    fn segment_metadata_sequence_rejects_tampered_bindings() {
        let valid = valid_segment_chain();

        let mut wrong_program = valid.clone();
        wrong_program[1].program_commit = digest(8);
        assert!(verify_segment_metadata_sequence(&wrong_program).is_err());

        let mut disconnected_memory = valid.clone();
        disconnected_memory[1].initial_memory_root = digest(99);
        assert!(verify_segment_metadata_sequence(&disconnected_memory).is_err());

        let mut early_termination = valid.clone();
        early_termination[1].is_terminate = F::ONE;
        early_termination[1].exit_code = F::ZERO;
        assert!(verify_segment_metadata_sequence(&early_termination).is_err());

        let mut missing_termination = valid;
        missing_termination[2].is_terminate = F::ZERO;
        missing_termination[2].exit_code = F::from_u32(super::DEFAULT_SUSPEND_EXIT_CODE);
        assert!(verify_segment_metadata_sequence(&missing_termination).is_err());
    }

    #[test]
    fn native_warp_height_buckets_round_up_without_exceeding_each_air_maximum() {
        let mut heights = vec![
            vec![1 << 14, 1 << 9, 0],
            vec![1 << 13, 1 << 7, 1 << 6],
            vec![1 << 10, 1 << 5, 1 << 3],
        ];
        bucket_native_warp_trace_heights(&mut heights, 2);
        assert_eq!(
            heights,
            vec![
                vec![1 << 14, 1 << 9, 0],
                vec![1 << 14, 1 << 7, 1 << 6],
                vec![1 << 10, 1 << 5, 1 << 4],
            ]
        );
    }
}

#[derive(Serialize, Deserialize)]
#[serde(bound(
    serialize = "Com<SC>: Serialize",
    deserialize = "Com<SC>: Deserialize<'de>"
))]
pub struct ContinuationVmProof<SC: StarkProtocolConfig> {
    pub per_segment: Vec<Proof<SC>>,
    pub user_public_values: UserPublicValuesProof<{ DIGEST_WIDTH }, Val<SC>>,
}

#[derive(Error, Debug)]
pub enum NativeWarpStreamError<SegmentError> {
    #[error("virtual machine error: {0}")]
    Vm(#[from] VirtualMachineError),
    #[error("native WARP segment consumer error: {0}")]
    Segment(SegmentError),
}

/// Exact continuation schedule and AIR heights prepared for native WARP.
///
/// The plan is derived from the same metered execution and preflight trace
/// generation used by proving. Keeping it as a typed artifact lets setup
/// generate the transition-leaf verifier key before online proving without
/// changing segmentation or trusting caller-supplied trace dimensions.
#[derive(Clone, Debug)]
pub struct NativeWarpContinuationPlan {
    segments: Vec<Segment>,
    planned_heights: Vec<Vec<u32>>,
}

impl NativeWarpContinuationPlan {
    #[must_use]
    pub fn planned_heights(&self) -> &[Vec<u32>] {
        &self.planned_heights
    }

    #[must_use]
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }
}

/// Coarsen only the padding heights used by native WARP. Segment instruction
/// boundaries and zero/absent AIRs are unchanged. Buckets are anchored at each
/// AIR's observed maximum, so rounding can never exceed the metered per-AIR
/// scalar-message limit.
fn bucket_native_warp_trace_heights(planned_heights: &mut [Vec<u32>], stride: u8) {
    if stride <= 1 || planned_heights.is_empty() {
        return;
    }
    let air_count = planned_heights[0].len();
    assert!(
        planned_heights
            .iter()
            .all(|segment| segment.len() == air_count),
        "native WARP height plan must be rectangular"
    );
    let mut max_logs = vec![None::<u32>; air_count];
    for segment in planned_heights.iter() {
        for (air_id, &height) in segment.iter().enumerate() {
            if height != 0 {
                assert!(height.is_power_of_two());
                let log_height = height.ilog2();
                max_logs[air_id] =
                    Some(max_logs[air_id].map_or(log_height, |max| max.max(log_height)));
            }
        }
    }
    let stride = u32::from(stride);
    for segment in planned_heights {
        for (air_id, height) in segment.iter_mut().enumerate() {
            if *height == 0 {
                continue;
            }
            let max_log = max_logs[air_id].expect("active AIR has an observed maximum");
            let current_log = height.ilog2();
            let bucket_distance = (max_log - current_log) / stride * stride;
            *height = 1u32 << (max_log - bucket_distance);
        }
    }
}

/// Prover for a specific exe in a specific continuation VM using a specific Stark config.
pub trait ContinuationVmProver<SC: StarkProtocolConfig> {
    fn prove(
        &mut self,
        input: impl Into<Streams<Val<SC>>>,
    ) -> Result<ContinuationVmProof<SC>, VirtualMachineError>;
}

/// Virtual machine prover instance for a fixed VM config and a fixed program. For use in proving a
/// program directly on bare metal.
///
/// This struct contains the [VmState] itself to avoid re-allocating guest memory. The memory is
/// reset with zeros before execution.
#[derive(Getters, MutGetters)]
pub struct VmInstance<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    pub vm: VirtualMachine<E, VB>,
    pub interpreter: PreflightInterpretedInstance2<Val<E::SC>, VB::VmConfig>,
    #[getset(get = "pub")]
    program_commitment: <E::PB as ProverBackend>::Commitment,
    #[getset(get = "pub")]
    exe: Arc<VmExe<Val<E::SC>>>,
    #[getset(get = "pub", get_mut = "pub")]
    state: Option<VmState<Val<E::SC>, GuestMemory>>,
    /// Checked, executable-specific native metering artifact selected by the SDK.
    ///
    /// The VM loader deliberately does not define a trust policy for persisted native code. The
    /// SDK validates its versioned manifest, executable/shape/toolchain identity, and file digest
    /// before installing this path.
    #[cfg(feature = "rvr")]
    metered_artifact_path: Option<std::path::PathBuf>,
}

impl<E, VB> VmInstance<E, VB>
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    pub fn new(
        mut vm: VirtualMachine<E, VB>,
        exe: Arc<VmExe<Val<E::SC>>>,
        cached_program_trace: CommittedTraceData<E::PB>,
    ) -> Result<Self, StaticProgramError> {
        let program_commitment = cached_program_trace.commitment;
        vm.load_program(cached_program_trace);
        let interpreter = vm.preflight_interpreter(&exe)?;
        let state = vm.create_initial_state(&exe, vec![]);
        Ok(Self {
            vm,
            interpreter,
            program_commitment,
            exe,
            state: Some(state),
            #[cfg(feature = "rvr")]
            metered_artifact_path: None,
        })
    }

    /// Select a previously validated native metering artifact for continuation planning.
    #[cfg(feature = "rvr")]
    pub fn set_metered_artifact_path(&mut self, path: impl Into<std::path::PathBuf>) {
        self.metered_artifact_path = Some(path.into());
    }

    #[instrument(name = "vm.reset_state", level = "debug", skip_all)]
    pub fn reset_state(&mut self, inputs: impl Into<Streams<Val<E::SC>>>) {
        let state = self.state.as_mut().unwrap();
        state.reset(&self.exe.init_memory, self.exe.pc_start, inputs);

        #[cfg(all(feature = "metrics", any(feature = "perf-metrics", debug_assertions)))]
        {
            state.metrics.fn_bounds = self.exe.fn_bounds.clone();
            state.metrics.debug_infos = self.exe.program.debug_infos();
        }
    }
}

impl<E, VB> ContinuationVmProver<E::SC> for VmInstance<E, VB>
where
    E: StarkEngine,
    Val<E::SC>: PrimeField32,
    VB: VmBuilder<E>,
    <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>
        + MeteredExecutor<Val<E::SC>>
        + PreflightExecutor<Val<E::SC>, VB::RecordArena>,
{
    /// First performs metered execution to determine segments. Then sequentially proves each
    /// segment. The proof for each segment uses the specified [ProverBackend], but the proof for
    /// the next segment does not start before the current proof finishes.
    fn prove(
        &mut self,
        input: impl Into<Streams<Val<E::SC>>>,
    ) -> Result<ContinuationVmProof<E::SC>, VirtualMachineError> {
        self.prove_continuations(input, |_, _| {})
    }
}

impl<E, VB> VmInstance<E, VB>
where
    E: StarkEngine,
    Val<E::SC>: PrimeField32,
    VB: VmBuilder<E>,
    <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>
        + MeteredExecutor<Val<E::SC>>
        + PreflightExecutor<Val<E::SC>, VB::RecordArena>,
{
    /// For internal use to resize trace matrices before proving.
    ///
    /// The closure `modify_ctx(seg_idx, &mut ctx)` is called sequentially for each segment.
    pub fn prove_continuations(
        &mut self,
        input: impl Into<Streams<Val<E::SC>>>,
        mut modify_ctx: impl FnMut(usize, &mut ProvingContext<E::PB>),
    ) -> Result<ContinuationVmProof<E::SC>, VirtualMachineError> {
        let input = input.into();
        self.reset_state(input.clone());
        #[cfg(feature = "rvr")]
        let metered_artifact_path = self.metered_artifact_path.clone();
        let vm = &mut self.vm;
        let metered_ctx = vm.build_metered_ctx(&self.exe);
        #[cfg(feature = "rvr")]
        let metered_instance = match metered_artifact_path {
            Some(path) => vm.load_metered_instance(&path, &self.exe)?,
            None => vm.metered_instance(&self.exe)?,
        };
        #[cfg(not(feature = "rvr"))]
        let metered_instance = vm.metered_instance(&self.exe)?;
        let (segments, _) = metered_instance.execute_metered(input, metered_ctx)?;
        let mut proofs = Vec::with_capacity(segments.len());
        let mut state = self.state.take();
        for (seg_idx, segment) in segments.into_iter().enumerate() {
            let _segment_span = info_span!("prove_segment", segment = seg_idx).entered();
            // We need a separate span so the metric label includes "segment" from _segment_span
            let _prove_span = info_span!("total_proof").entered();
            let Segment {
                num_insns,
                trace_heights,
                ..
            } = segment;
            let from_state = Option::take(&mut state).unwrap();
            vm.transport_init_memory_to_device(&from_state.memory);
            let PreflightExecutionOutput {
                system_records,
                record_arenas,
                to_state,
            } = vm.execute_preflight(
                &mut self.interpreter,
                from_state,
                Some(num_insns),
                &trace_heights,
            )?;
            state = Some(to_state);

            let mut ctx = vm.generate_proving_ctx(system_records, record_arenas)?;
            modify_ctx(seg_idx, &mut ctx);
            let proof = vm.engine.prove(vm.pk(), ctx).unwrap();
            proofs.push(proof);
        }
        let to_state = state.unwrap();
        let final_memory = &to_state.memory.memory;
        let final_memory_top_tree = vm.memory_top_tree().expect("memory top tree should exist");
        let user_public_values = UserPublicValuesProof::compute(
            vm.config().as_ref(),
            &vm_poseidon2_hasher(),
            final_memory,
            final_memory_top_tree,
        );
        self.state = Some(to_state);
        Ok(ContinuationVmProof {
            per_segment: proofs,
            user_public_values,
        })
    }

    /// Derives the native-WARP continuation schedule and AIR heights.
    ///
    /// Heights come from the metered pass's per-AIR bounds rounded to the next power of two, so
    /// no preflight or trace generation runs here. That substitution is what makes this cheap:
    /// it previously ran a full `execute_preflight` + `generate_proving_ctx` per segment purely
    /// to read `common_main.height()` and then dropped the context, which doubled the VM half of
    /// native-WARP proving against the recursive lane.
    ///
    /// The bound is not always tight -- the Poseidon2 periphery is bounded by
    /// `2 * leaves + 2 * merkle_nodes` and then deduplicates equal hash inputs during trace
    /// generation -- so the streaming pass pins every trace to the planned height rather than
    /// letting it shrink. Padding to a plan is sound because the descriptor binding requires
    /// each `log_height` to *equal* the catalog shape, authenticated by a membership proof, not
    /// to be minimal; the padding rows are the ones trace generation already emits between the
    /// record count and the next power of two.
    #[instrument(name = "plan_continuations_native_warp", level = "info", skip_all)]
    pub fn plan_continuations_native_warp(
        &mut self,
        input: impl Into<Streams<Val<E::SC>>>,
    ) -> Result<NativeWarpContinuationPlan, VirtualMachineError> {
        let input = input.into();
        self.reset_state(input.clone());
        #[cfg(feature = "rvr")]
        let metered_artifact_path = self.metered_artifact_path.clone();
        let segments = {
            let vm = &mut self.vm;
            let metered_ctx = vm.build_metered_ctx(&self.exe);
            #[cfg(feature = "rvr")]
            let metered_instance = match metered_artifact_path {
                Some(path) => vm.load_metered_instance(&path, &self.exe)?,
                None => vm.metered_instance(&self.exe)?,
            };
            #[cfg(not(feature = "rvr"))]
            let metered_instance = vm.metered_instance(&self.exe)?;
            metered_instance
                .execute_metered(input.clone(), metered_ctx)?
                .0
        };

        let num_airs = self.vm.pk().per_air.len();
        // Planning every AIR present at `2^l_skip`, to make the trace set uniform across
        // segments and settle the transition-leaf verifier key, overflows the reduction program
        // by a consistent ~15%: measured 610,354 instructions against a 524,288 cap at a 2^19
        // stacked height, and 1,194,126 against 1,048,576 at 2^20. The program scales with the
        // cap, so raising the height is a treadmill. The verifier limit applies to the largest
        // single program, so chunking cannot absorb it either.
        //
        // The six AIRs this admits are absent from most segments and would enter at minimum
        // height with no real rows, yet each still costs roughly 14k instructions because the
        // program emits a full constraint evaluation regardless of trace height. Closing a 15%
        // gap therefore means emitting a compact program for an empty trace, not a bigger
        // circuit. Until then, absent AIRs stay absent.
        let mut planned_heights: Vec<Vec<u32>> = segments
            .iter()
            .map(|segment| {
                assert_eq!(segment.trace_heights.len(), num_airs);
                segment
                    .trace_heights
                    .iter()
                    .map(|&height| {
                        next_power_of_two_or_zero(height as usize)
                            .try_into()
                            .expect("planned trace height fits in u32")
                    })
                    .collect::<Vec<u32>>()
            })
            .collect();
        let height_bucket_stride = self.vm.config().as_ref().native_warp_height_bucket_stride;
        let rows_before = planned_heights
            .iter()
            .flatten()
            .map(|&height| u64::from(height))
            .sum::<u64>();
        bucket_native_warp_trace_heights(&mut planned_heights, height_bucket_stride);
        let rows_after = planned_heights
            .iter()
            .flatten()
            .map(|&height| u64::from(height))
            .sum::<u64>();
        tracing::info!(
            height_bucket_stride,
            rows_before,
            rows_after,
            padding_ratio = rows_after as f64 / rows_before.max(1) as f64,
            "native WARP verifier-shape height bucketing"
        );
        release_unused_allocator_memory();
        self.reset_state(input);
        Ok(NativeWarpContinuationPlan {
            segments,
            planned_heights,
        })
    }
}

/// The native-WARP streaming pass, which pins trace heights to the plan.
///
/// Split from the block above for the one extra bound: pinning the system AIRs needs
/// [`SystemWithFixedTraceHeights`]. Both concrete inventories implement it, so this is a
/// bound rather than a restriction -- keeping it off the shared block leaves
/// `prove_continuations` and the other lanes generic over inventories that do not.
impl<E, VB> VmInstance<E, VB>
where
    E: StarkEngine,
    Val<E::SC>: PrimeField32,
    VB: VmBuilder<E>,
    VB::SystemChipInventory: SystemWithFixedTraceHeights,
    <VB::VmConfig as VmExecutionConfig<Val<E::SC>>>::Executor: Executor<Val<E::SC>>
        + MeteredExecutor<Val<E::SC>>
        + PreflightExecutor<Val<E::SC>, VB::RecordArena>,
{
    /// Streams proving contexts using an exact plan prepared by
    /// [`Self::plan_continuations_native_warp`].
    pub fn prove_continuations_native_warp_stream_prepared<SegmentError>(
        &mut self,
        input: impl Into<Streams<Val<E::SC>>>,
        plan: NativeWarpContinuationPlan,
        mut modify_ctx: impl FnMut(usize, &mut ProvingContext<E::PB>),
        mut consume_segment: impl FnMut(
            usize,
            &E,
            &DeviceMultiStarkProvingKey<E::PB>,
            ProvingContext<E::PB>,
        ) -> Result<(), SegmentError>,
    ) -> Result<
        UserPublicValuesProof<{ DIGEST_WIDTH }, Val<E::SC>>,
        NativeWarpStreamError<SegmentError>,
    > {
        let input = input.into();
        self.reset_state(input);
        let NativeWarpContinuationPlan {
            segments,
            planned_heights,
        } = plan;

        let mut state = self.state.take();
        log_native_warp_process_memory("after_vm_state_take", usize::MAX);
        let vm = &mut self.vm;
        for (seg_idx, segment) in segments.into_iter().enumerate() {
            // Two spans, as in the recursive lane: the outer one carries the
            // label, and the inner one closes while the outer is still entered
            // so its children inherit `segment=N`. Without these the streaming
            // pass emits no per-segment timing at all and its gauges collapse
            // to a single last-writer-wins sample.
            let _segment_span = info_span!("prove_segment", segment = seg_idx).entered();
            let _warp_segment_span = info_span!("warp_segment_total").entered();
            log_native_warp_process_memory("before_segment_span", seg_idx);
            log_native_warp_process_memory("before_preflight", seg_idx);
            let Segment {
                num_insns,
                trace_heights,
                ..
            } = segment;
            let from_state = Option::take(&mut state).unwrap();
            vm.transport_init_memory_to_device(&from_state.memory);
            let PreflightExecutionOutput {
                system_records,
                mut record_arenas,
                to_state,
            } = vm
                .execute_preflight(
                    &mut self.interpreter,
                    from_state,
                    Some(num_insns),
                    &trace_heights,
                )
                .map_err(VirtualMachineError::from)?;
            state = Some(to_state);
            log_native_warp_process_memory("after_preflight", seg_idx);
            // Pin every trace to the plan before generating it. The plan is an upper bound
            // derived from metered execution, and the shape catalog is already committed to it,
            // so a trace that generated fewer rows must be padded up rather than shrink the
            // shape out from under the binding.
            let Some(planned) = planned_heights.get(seg_idx) else {
                return Err(NativeWarpStreamError::Vm(
                    VirtualMachineError::NativeWarpPlan(
                        "planned heights are missing a segment".to_owned(),
                    ),
                ));
            };
            // `record_arenas` is indexed by AIR id: `execute_preflight` builds one arena per
            // entry of `trace_heights`, and `generate_proving_ctx` only splits off the leading
            // system block afterwards.
            for (air_idx, arena) in record_arenas.iter_mut().enumerate() {
                arena.force_trace_height(planned[air_idx] as usize);
            }
            vm.override_system_trace_heights(planned);
            let mut ctx = vm
                .generate_proving_ctx(system_records, record_arenas)
                .map_err(VirtualMachineError::from)?;
            let mut exact = vec![0u32; vm.pk().per_air.len()];
            for (air_id, trace) in &ctx.per_trace {
                exact[*air_id] = trace
                    .common_main
                    .height()
                    .try_into()
                    .expect("validated trace height fits in u32");
            }
            if planned_heights.get(seg_idx) != Some(&exact) {
                return Err(NativeWarpStreamError::Vm(
                    VirtualMachineError::NativeWarpPlan(format!(
                        "AIR heights changed between setup and proving in segment {seg_idx}: {}",
                        planned
                            .iter()
                            .zip(&exact)
                            .enumerate()
                            .filter(|(_, (planned, exact))| planned != exact)
                            .map(|(air_id, (planned, exact))| {
                                format!("AIR {air_id}: planned {planned}, actual {exact}")
                            })
                            .join(", ")
                    )),
                ));
            }
            #[cfg(any(debug_assertions, feature = "test-utils", feature = "stark-debug"))]
            if std::env::var_os("OPENVM_WARP_DEBUG_SEGMENT").is_some() {
                debug_proving_ctx(vm, &ctx);
            }
            log_native_warp_process_memory("after_trace_gen", seg_idx);
            modify_ctx(seg_idx, &mut ctx);
            consume_segment(seg_idx, &vm.engine, vm.pk(), ctx)
                .map_err(NativeWarpStreamError::Segment)?;
            log_native_warp_process_memory("after_segment_consumer", seg_idx);
        }
        let to_state = state.unwrap();
        let final_memory = &to_state.memory.memory;
        let final_memory_top_tree = vm.memory_top_tree().expect("memory top tree should exist");
        let user_public_values = UserPublicValuesProof::compute(
            vm.config().as_ref(),
            &vm_poseidon2_hasher(),
            final_memory,
            final_memory_top_tree,
        );
        self.state = Some(to_state);
        Ok(user_public_values)
    }

    /// Plans exact segment shapes and streams original AIR contexts into the
    /// native WARP backend without constructing deferred SWIRL proofs.
    pub fn prove_continuations_native_warp_stream_with_plan<SegmentError>(
        &mut self,
        input: impl Into<Streams<Val<E::SC>>>,
        modify_ctx: impl FnMut(usize, &mut ProvingContext<E::PB>),
        plan_segments: impl FnOnce(&[Vec<u32>]) -> Result<(), SegmentError>,
        consume_segment: impl FnMut(
            usize,
            &E,
            &DeviceMultiStarkProvingKey<E::PB>,
            ProvingContext<E::PB>,
        ) -> Result<(), SegmentError>,
    ) -> Result<
        UserPublicValuesProof<{ DIGEST_WIDTH }, Val<E::SC>>,
        NativeWarpStreamError<SegmentError>,
    > {
        let input = input.into();
        let plan = self
            .plan_continuations_native_warp(input.clone())
            .map_err(NativeWarpStreamError::Vm)?;
        log_native_warp_process_memory("before_plan_callback", usize::MAX);
        plan_segments(plan.planned_heights()).map_err(NativeWarpStreamError::Segment)?;
        log_native_warp_process_memory("after_plan_callback", usize::MAX);
        self.prove_continuations_native_warp_stream_prepared(
            input,
            plan,
            modify_ctx,
            consume_segment,
        )
    }
}

fn release_unused_allocator_memory() {
    #[cfg(all(target_os = "linux", target_env = "gnu"))]
    unsafe {
        libc::malloc_trim(0);
    }
}

#[cfg(target_os = "linux")]
fn log_native_warp_process_memory(phase: &'static str, segment: usize) {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else {
        return;
    };
    let value_kib = |name: &str| {
        status.lines().find_map(|line| {
            let value = line.strip_prefix(name)?.trim();
            value.split_whitespace().next()?.parse::<u64>().ok()
        })
    };
    tracing::info!(
        phase,
        segment,
        rss_mib = value_kib("VmRSS:").unwrap_or_default() >> 10,
        virtual_mib = value_kib("VmSize:").unwrap_or_default() >> 10,
        swap_mib = value_kib("VmSwap:").unwrap_or_default() >> 10,
        "native WARP process memory"
    );
}

#[cfg(not(target_os = "linux"))]
fn log_native_warp_process_memory(_phase: &'static str, _segment: usize) {}

/// The payload of a verified guest VM execution.
pub struct VerifiedExecutionPayload<F> {
    /// The Merklelized hash of:
    /// - Program code commitment (commitment of the cached trace)
    /// - Merkle root of the initial memory
    /// - Starting program counter (`pc_start`)
    ///
    /// The Merklelization uses Poseidon2 as a cryptographic hash function (for the leaves)
    /// and a cryptographic compression function (for internal nodes).
    pub exe_commit: [F; DIGEST_WIDTH],
    /// The Merkle root of the final memory state.
    pub final_memory_root: [F; DIGEST_WIDTH],
}

const PROGRAM_AIR_PRESENT: u8 = 1 << 0;
const CONNECTOR_AIR_PRESENT: u8 = 1 << 1;
const BOUNDARY_AIR_PRESENT: u8 = 1 << 2;
const MERKLE_AIR_PRESENT: u8 = 1 << 3;
pub const REQUIRED_SYSTEM_AIRS: u8 =
    PROGRAM_AIR_PRESENT | CONNECTOR_AIR_PRESENT | BOUNDARY_AIR_PRESENT | MERKLE_AIR_PRESENT;

/// Verifier-visible continuation data extracted from one authenticated segment relation.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound = "")]
pub struct VmSegmentMetadata<SC: StarkProtocolConfig> {
    pub program_commit: SC::Digest,
    pub initial_pc: SC::F,
    pub final_pc: SC::F,
    pub exit_code: SC::F,
    pub is_terminate: SC::F,
    pub initial_memory_root: [SC::F; DIGEST_WIDTH],
    pub final_memory_root: [SC::F; DIGEST_WIDTH],
    pub present_system_airs: u8,
}

/// Extracts and validates the public continuation fields shared by recursive and WARP proofs.
pub fn vm_segment_metadata_from_parts<SC>(
    vk: &MultiStarkVerifyingKey<SC>,
    trace_vdata: &[Option<TraceVData<SC>>],
    public_values: &[Vec<SC::F>],
) -> Result<VmSegmentMetadata<SC>, VmVerificationError<SC>>
where
    SC: StarkProtocolConfig,
    SC::F: PrimeField32,
{
    if trace_vdata.len() != vk.inner.per_air.len() || public_values.len() != vk.inner.per_air.len()
    {
        return Err(VmVerificationError::InvalidSegmentMetadata(
            "AIR vector length",
        ));
    }

    let mut program_commit = None;
    let mut connector = None;
    let mut memory = None;
    let mut present_system_airs = 0u8;

    for (air_idx, ((vdata, pvs), air_vk)) in trace_vdata
        .iter()
        .zip(public_values)
        .zip(&vk.inner.per_air)
        .enumerate()
    {
        if air_idx == PROGRAM_AIR_ID {
            let vdata = vdata
                .as_ref()
                .ok_or(VmVerificationError::SystemAirMissing {
                    air_id: PROGRAM_AIR_ID,
                })?;
            let commitment = vdata
                .cached_commitments
                .get(PROGRAM_CACHED_TRACE_INDEX)
                .copied()
                .ok_or(VmVerificationError::InvalidSegmentMetadata(
                    "program cached commitment",
                ))?;
            program_commit = Some(commitment);
            present_system_airs |= PROGRAM_AIR_PRESENT;
        } else if air_idx == CONNECTOR_AIR_ID {
            if vdata.is_none() {
                return Err(VmVerificationError::SystemAirMissing {
                    air_id: CONNECTOR_AIR_ID,
                });
            }
            if pvs.len() != 4 {
                return Err(VmVerificationError::UnexpectedPvs {
                    expected: 4,
                    actual: pvs.len(),
                });
            }
            let values: &VmConnectorPvs<_> = pvs.as_slice().borrow();
            connector = Some((
                values.initial_pc,
                values.final_pc,
                values.exit_code,
                values.is_terminate,
            ));
            present_system_airs |= CONNECTOR_AIR_PRESENT;
        } else if air_idx == BOUNDARY_AIR_ID {
            if vdata.is_some() {
                present_system_airs |= BOUNDARY_AIR_PRESENT;
            }
            if !pvs.is_empty() {
                return Err(VmVerificationError::UnexpectedPvs {
                    expected: 0,
                    actual: pvs.len(),
                });
            }
        } else if air_idx == MERKLE_AIR_ID {
            if vdata.is_none() {
                return Err(VmVerificationError::SystemAirMissing {
                    air_id: MERKLE_AIR_ID,
                });
            }
            if pvs.len() != 2 * DIGEST_WIDTH {
                return Err(VmVerificationError::UnexpectedPvs {
                    expected: 2 * DIGEST_WIDTH,
                    actual: pvs.len(),
                });
            }
            let values: &MemoryMerklePvs<_, DIGEST_WIDTH> = pvs.as_slice().borrow();
            memory = Some((values.initial_root, values.final_root));
            present_system_airs |= MERKLE_AIR_PRESENT;
        } else if !pvs.is_empty() {
            return Err(VmVerificationError::UnexpectedPvs {
                expected: 0,
                actual: pvs.len(),
            });
        } else {
            debug_assert_eq!(air_vk.params.num_public_values, 0);
        }
    }

    if present_system_airs != REQUIRED_SYSTEM_AIRS {
        for (air_id, flag) in [
            (PROGRAM_AIR_ID, PROGRAM_AIR_PRESENT),
            (CONNECTOR_AIR_ID, CONNECTOR_AIR_PRESENT),
            (BOUNDARY_AIR_ID, BOUNDARY_AIR_PRESENT),
            (MERKLE_AIR_ID, MERKLE_AIR_PRESENT),
        ] {
            if present_system_airs & flag == 0 {
                return Err(VmVerificationError::SystemAirMissing { air_id });
            }
        }
    }

    let (initial_pc, final_pc, exit_code, is_terminate) = connector.ok_or(
        VmVerificationError::InvalidSegmentMetadata("connector public values"),
    )?;
    let (initial_memory_root, final_memory_root) = memory.ok_or(
        VmVerificationError::InvalidSegmentMetadata("memory public values"),
    )?;
    Ok(VmSegmentMetadata {
        program_commit: program_commit.ok_or(VmVerificationError::InvalidSegmentMetadata(
            "program commitment",
        ))?,
        initial_pc,
        final_pc,
        exit_code,
        is_terminate,
        initial_memory_root,
        final_memory_root,
        present_system_airs,
    })
}

/// Checks the ordered continuation chain after each segment relation has been authenticated.
pub fn verify_segment_metadata_sequence<SC>(
    segments: &[VmSegmentMetadata<SC>],
) -> Result<VerifiedExecutionPayload<SC::F>, VmVerificationError<SC>>
where
    SC: StarkProtocolConfig,
    SC::F: PrimeField32,
    SC::Digest: Into<[SC::F; DIGEST_WIDTH]>,
{
    let first = segments.first().ok_or(VmVerificationError::ProofNotFound)?;
    let program_commit = first.program_commit;
    let start_pc = first.initial_pc;
    let initial_memory_root = first.initial_memory_root;
    let mut previous_final_pc = None;
    let mut previous_final_memory_root = None;

    for (index, segment) in segments.iter().enumerate() {
        if segment.present_system_airs != REQUIRED_SYSTEM_AIRS {
            return Err(VmVerificationError::InvalidSegmentMetadata(
                "required system AIR bitmap",
            ));
        }
        if segment.program_commit != program_commit {
            return Err(VmVerificationError::ProgramCommitMismatch { index });
        }
        if let Some(previous) = previous_final_pc {
            if segment.initial_pc != previous {
                return Err(VmVerificationError::InitialPcMismatch {
                    initial: segment.initial_pc.as_canonical_u32(),
                    prev_final: previous.as_canonical_u32(),
                });
            }
        }
        if let Some(previous) = previous_final_memory_root {
            if segment.initial_memory_root != previous {
                return Err(VmVerificationError::InitialMemoryRootMismatch);
            }
        }

        let expected_is_terminate = index + 1 == segments.len();
        if segment.is_terminate != SC::F::from_bool(expected_is_terminate) {
            return Err(VmVerificationError::IsTerminateMismatch {
                expected: expected_is_terminate,
                actual: segment.is_terminate.as_canonical_u32() != 0,
            });
        }
        let expected_exit_code = if expected_is_terminate {
            ExitCode::Success as u32
        } else {
            DEFAULT_SUSPEND_EXIT_CODE
        };
        if segment.exit_code != SC::F::from_u32(expected_exit_code) {
            return Err(VmVerificationError::ExitCodeMismatch {
                expected: expected_exit_code,
                actual: segment.exit_code.as_canonical_u32(),
            });
        }
        previous_final_pc = Some(segment.final_pc);
        previous_final_memory_root = Some(segment.final_memory_root);
    }

    Ok(VerifiedExecutionPayload {
        exe_commit: compute_exe_commit(
            &vm_poseidon2_hasher(),
            &program_commit.into(),
            &initial_memory_root,
            start_pc,
        ),
        final_memory_root: previous_final_memory_root.expect("non-empty segment metadata"),
    })
}

/// Verify segment proofs with boundary condition checks for continuation between segments.
///
/// Assumption:
/// - `vk` is a valid verifying key of a VM circuit.
///
/// Returns:
/// - The commitment to the VM executable extracted from `proofs`. It is the responsibility of the
///   caller to check that the returned commitment matches the VM executable that the VM was
///   supposed to execute.
/// - The Merkle root of the final memory state.
///
/// ## Note
/// This function does not extract or verify any user public values from the final memory state.
/// This verification requires an additional Merkle proof with respect to the Merkle root of
/// the final memory state.
// @dev: This function doesn't need to be generic in `VC`.
pub fn verify_segments<E>(
    engine: &E,
    vk: &MultiStarkVerifyingKey<E::SC>,
    proofs: &[Proof<E::SC>],
) -> Result<VerifiedExecutionPayload<Val<E::SC>>, VmVerificationError<E::SC>>
where
    E: StarkEngine,
    Val<E::SC>: PrimeField32,
    Com<E::SC>: Into<[Val<E::SC>; DIGEST_WIDTH]>,
{
    if proofs.is_empty() {
        return Err(VmVerificationError::ProofNotFound);
    }
    let mut metadata = Vec::with_capacity(proofs.len());
    for proof in proofs {
        engine.verify(vk, proof)?;
        metadata.push(vm_segment_metadata_from_parts(
            vk,
            &proof.trace_vdata,
            &proof.public_values,
        )?);
    }
    verify_segment_metadata_sequence(&metadata)
}

impl<SC: StarkProtocolConfig> Clone for ContinuationVmProof<SC>
where
    Com<SC>: Clone,
{
    fn clone(&self) -> Self {
        Self {
            per_segment: self.per_segment.clone(),
            user_public_values: self.user_public_values.clone(),
        }
    }
}

pub(super) fn create_memory_image(
    memory_config: &MemoryConfig,
    init_memory: &SparseMemoryImage,
) -> GuestMemory {
    let mut inner = AddressMap::new(memory_config.addr_spaces.clone());
    inner.set_from_sparse(init_memory);
    GuestMemory::new(inner)
}

impl<E, VC> VirtualMachine<E, VC>
where
    E: StarkEngine,
    VC: VmBuilder<E>,
    VC::SystemChipInventory: SystemWithFixedTraceHeights,
{
    /// Sets fixed trace heights for the system AIRs' trace matrices.
    ///
    /// `heights` is the whole AIR-ordered vector, not just the leading system block. The system
    /// owns AIRs at both ends of that vector -- program and connector at the front, the
    /// Poseidon2 periphery and the range checker at the back -- so truncating to
    /// `num_airs()` would silently drop the one system AIR whose height actually needs pinning.
    pub fn override_system_trace_heights(&mut self, heights: &[u32]) {
        let num_sys_airs = self.config().as_ref().num_airs();
        assert!(heights.len() >= num_sys_airs);
        self.chip_complex.system.override_trace_heights(heights);
    }
}

/// Runs the STARK backend debugger to check the constraints against the trace matrices
/// logically, instead of cryptographically. This will panic if any constraint is violated, and
/// using `RUST_BACKTRACE=1` can be used to read the stack backtrace of where the constraint
/// failed in the code (this requires the code to be compiled with debug=true). Using lower
/// optimization levels like -O0 will prevent the compiler from inlining and give better
/// debugging information.
// @dev The debugger needs the host proving key.
//      This function is used both by VirtualMachine::debug_proving_ctx and by
// stark_utils::air_test_impl
#[cfg(any(debug_assertions, feature = "test-utils", feature = "stark-debug"))]
#[tracing::instrument(level = "debug", skip_all)]
pub fn debug_proving_ctx<E, VB>(vm: &VirtualMachine<E, VB>, ctx: &ProvingContext<E::PB>)
where
    E: StarkEngine,
    VB: VmBuilder<E>,
{
    let air_inv = vm.config().create_airs().unwrap();
    let global_airs: Vec<AirRef<E::SC>> = air_inv.into_airs().map(|a| a as AirRef<_>).collect();
    vm.engine.debug(&global_airs, ctx);
}

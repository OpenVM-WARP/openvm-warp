//! Traits and types describing the core interfaces of the verifier sub-circuit. The verifier
//! sub-circuit verifies multiple proofs for the same child verifying key. It supports **recursive**
//! verification, where the child verifying key is equal to the verifying key of the verifier
//! circuit itself.
use std::{iter, sync::Arc};

use openvm_cpu_backend::CpuBackend;
#[cfg(feature = "cuda")]
use openvm_cuda_common::stream::GpuDeviceCtx;
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{
    interaction::BusIndex,
    keygen::types::{LinearConstraint, MultiStarkVerifyingKey},
    proof::{BatchConstraintProof, GkrProof, Proof, StackingProof, TraceVData, WhirProof},
    prover::{
        AirProvingContext, CommittedTraceData, NativeStackingReduction,
        PendingConstrainedCodeWitness, ProverBackend,
    },
    AirRef, EngineDeviceCtx, FiatShamirTranscript, StarkEngine, StarkProtocolConfig,
    TranscriptHistory, TranscriptLog,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, CHUNK, EF, F};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_maybe_rayon::prelude::*;

use crate::{
    batch_constraint::{
        expr_eval::CachedTraceRecord, BatchConstraintModule, LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX,
    },
    bus::{
        AirPresenceBus, AirShapeBus, BatchConstraintModuleBus, CachedCommitBus, ColumnClaimsBus,
        CommitmentsBus, ConstraintSumcheckRandomnessBus, ConstraintsFoldingInputBus, Eq3bShapeBus,
        EqNegBaseRandBus, EqNegResultBus, EqNsNLogupMaxBus, ExpressionClaimNMaxBus,
        FinalTranscriptStateBus, FractionFolderInputBus, GkrModuleBus, HyperdimBus,
        InteractionsFoldingInputBus, LiftedHeightsBus, MerkleVerifyBus, NLiftBus,
        Poseidon2CompressBus, Poseidon2PermuteBus, PreHashBus, PublicValuesBus,
        ResumeTranscriptStateBus, SelUniBus, StackingIndicesBus, StackingModuleBus, TranscriptBus,
        TranscriptEndIndexBus, WhirModuleBus, WhirMuBus, WhirOpeningPointBus,
        WhirOpeningPointLookupBus, XiRandomnessBus,
    },
    gkr::GkrModule,
    primitives::{
        bus::{ExpBitsLenBus, PowerCheckerBus, RangeCheckerBus, RightShiftBus},
        exp_bits_len::{ExpBitsLenAir, ExpBitsLenCpuTraceGenerator},
        pow::{PowerCheckerAir, PowerCheckerCpuTraceGenerator},
    },
    proof_shape::ProofShapeModule,
    stacking::StackingModule,
    transcript::{Poseidon2BusOwner, Poseidon2MultibusInputs, TranscriptModule},
    utils::poseidon2_hash_slice_with_states,
    whir::WhirModule,
};

mod deferred_opening;
mod deferred_stacking;
mod dummy;
mod logup_only;
pub use deferred_opening::{
    DeferredOpeningCheckpoint, DeferredOpeningCheckpointAir, DeferredOpeningCheckpointWitness,
};
pub use deferred_stacking::{ConstraintReductionCheckpointAir, ConstraintReductionClaimsAir};
pub use logup_only::{
    LogUpOnlyPartialVerifier, LogUpOnlyPartialVerifierExports, LogUpOnlyPrefixTranscript,
};
/// Public so a circuit outside this crate can assemble the modules itself.
///
/// The native WARP history certificate needs `ProofShapeModule` +  `GkrModule` +
/// `BatchConstraintModule` without `StackingModule` or `WhirModule`: it evaluates the batched
/// constraint claim at a point, which is the accumulation decider's job, while the PCS opening
/// those last two verify is exactly what WARP accumulation replaces. `ProofShapeModule::new`
/// takes a [`frame::MultiStarkVkeyFrame`], so that partial assembly is only expressible with
/// this module reachable.
pub mod frame;

const BATCH_CONSTRAINT_MOD_IDX: usize = 0;
#[cfg(feature = "cuda")]
const TRANSCRIPT_MOD_IDX: usize = 1;
/// The ordinary recursive lane verifies only a handful of children at once,
/// but the reduced-SWIRL terminal wrapper can replay hundreds of deferred
/// child prefixes under one fixed verifier.  Above this cutoff, nesting
/// module-level Rayon parallelism around each module's own parallel trace
/// generation materializes several power-of-two-padded matrices at once.
const LARGE_VERIFIER_BATCH_CUTOFF: usize = 16;
/// Bound the scoped preflight fan-out for the same large-batch path.  The old
/// implementation spawned one operating-system thread per child because its
/// intended arity was three or four.
const LARGE_VERIFIER_PREFLIGHT_WIDTH: usize = 32;
/// Public so a circuit outside this crate can add the `PowerCheckerAir` that
/// [`AggregationSubCircuit::airs`] adds outside any module. The native WARP history certificate
/// assembles three of the six modules itself, so it has to supply that table.
pub const POW_CHECKER_HEIGHT: usize = 32;

/// Equation checked by the recursive SWIRL batch-constraint verifier.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(u32)]
pub enum VerifierEquationMode {
    #[default]
    AirAndLogUp = 0,
    LogUpOnly = 1,
}

impl VerifierEquationMode {
    #[must_use]
    pub const fn includes_air(self) -> bool {
        matches!(self, Self::AirAndLogUp)
    }
}

pub enum CachedTraceCtx<PB: ProverBackend> {
    PcsData(CommittedTraceData<PB>),
    /// The caller is generating only the dynamic common-main trace for a
    /// relation whose cached columns were already authenticated and copied
    /// into a fixed setup index. No cached PCS witness is attached to the
    /// returned proving context.
    SetupBound,
    Records(CachedTraceRecord),
}

#[derive(Debug, Copy, Clone, Default, PartialEq, Eq)]
pub enum VerifierTailMode {
    /// Verify the complete per-segment stacking and WHIR tail.
    #[default]
    Complete,
    /// Verify stacking, but defer WHIR to a same-root terminal obligation.
    DeferredWhir,
    /// Stop after AIR/LogUp and defer both stacking and WHIR to one block-wide
    /// terminal reduction over the original commitments.
    DeferredStacking,
}

impl VerifierTailMode {
    #[must_use]
    const fn includes_stacking(self) -> bool {
        !matches!(self, Self::DeferredStacking)
    }

    #[must_use]
    const fn includes_whir(self) -> bool {
        matches!(self, Self::Complete)
    }

    #[must_use]
    const fn is_deferred(self) -> bool {
        !matches!(self, Self::Complete)
    }
}

#[derive(Debug, Copy, Clone)]
pub struct VerifierConfig {
    pub continuations_enabled: bool,
    pub final_state_bus_enabled: bool,
    pub has_cached: bool,
    pub tail_mode: VerifierTailMode,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            continuations_enabled: false,
            final_state_bus_enabled: false,
            has_cached: true,
            tail_mode: VerifierTailMode::Complete,
        }
    }
}

#[derive(Debug)]
pub struct VerifierExternalData<'a> {
    pub poseidon2_compress_inputs: &'a Vec<[F; POSEIDON2_WIDTH]>,
    pub poseidon2_permute_inputs: &'a Vec<[F; POSEIDON2_WIDTH]>,
    pub range_check_inputs: &'a Vec<usize>,
    pub power_check_inputs: &'a Vec<usize>,
    pub required_heights: Option<&'a [usize]>,
    pub final_transcript_state: Option<&'a mut [F; POSEIDON2_WIDTH]>,
}

// Trait to make tracegen functions generic on ProverBackend.
// `DC` is the device context type (e.g., `GpuDeviceCtx` for GPU, `()` for CPU).
pub trait VerifierTraceGen<
    PB: ProverBackend,
    SC: StarkProtocolConfig<F = F>,
    DC: Clone + Send + Sync,
>
{
    fn new(
        child_mvk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
        config: VerifierConfig,
    ) -> Self;

    fn commit_child_vk<E: StarkEngine<SC = SC, PB = PB>>(
        &self,
        engine: &E,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> CommittedTraceData<PB>
    where
        DC: From<EngineDeviceCtx<E>>;

    fn cached_trace_record(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> CachedTraceRecord;

    /// The generic `TS` allows using different transcript implementations for debugging purposes.
    /// The default type to use is `DuplexSpongeRecorder`.
    #[allow(clippy::ptr_arg)]
    fn generate_proving_ctxs<
        TS: FiatShamirTranscript<BabyBearPoseidon2Config>
            + TranscriptHistory<F = F, State = [F; POSEIDON2_WIDTH]>,
    >(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        cached_trace_ctx: CachedTraceCtx<PB>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        external_data: &mut VerifierExternalData,
        device_ctx: &DC,
        initial_transcript: TS,
    ) -> Option<Vec<AirProvingContext<PB>>>;

    fn generate_proving_ctxs_base<
        TS: FiatShamirTranscript<BabyBearPoseidon2Config>
            + TranscriptHistory<F = F, State = [F; POSEIDON2_WIDTH]>,
    >(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        cached_trace_ctx: CachedTraceCtx<PB>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        device_ctx: &DC,
        initial_transcript: TS,
    ) -> Vec<AirProvingContext<PB>> {
        let poseidon2_compress_inputs = vec![];
        let range_check_inputs = vec![];
        let power_check_inputs = vec![];

        let mut external_data = VerifierExternalData {
            poseidon2_compress_inputs: &poseidon2_compress_inputs,
            poseidon2_permute_inputs: &poseidon2_compress_inputs,
            range_check_inputs: &range_check_inputs,
            power_check_inputs: &power_check_inputs,
            required_heights: None,
            final_transcript_state: None,
        };

        self.generate_proving_ctxs::<TS>(
            child_vk,
            cached_trace_ctx,
            proofs,
            &mut external_data,
            device_ctx,
            initial_transcript,
        )
        .unwrap()
    }
}

// Trait to help make AIR generation generic
pub trait AggregationSubCircuit {
    fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>>;
    fn bus_inventory(&self) -> &BusInventory;
    fn next_bus_idx(&self) -> BusIndex;
    fn max_num_proofs(&self) -> usize;

    /// Optional circuit-specific bus on which the ordinary continuations PVS
    /// AIR republishes each authenticated child's `(internal_flag,
    /// recursion_depth)`.  Generic recursive circuits leave this disabled;
    /// History v4 uses it to bind its custom proof-kind/depth statement to the
    /// already authenticated OpenVM verifier public values without consuming
    /// `PublicValuesBus` coordinates twice.
    fn verifier_layer_identity_bus_idx(&self) -> Option<BusIndex> {
        None
    }

    /// Optional companion bus carrying the authenticated child `VmPvs` for
    /// each proof slot. History v4 uses it to bind program and execution
    /// endpoints to its custom statement. Generic recursive circuits leave it
    /// disabled.
    fn verifier_execution_identity_bus_idx(&self) -> Option<BusIndex> {
        None
    }
}

pub trait AirModule {
    fn num_airs(&self) -> usize;
    fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>>;
}

/// Trait defining the types for the global input shared across modules for trace generation. These
/// types are specialized per hardware backend.
pub trait GlobalTraceGenCtx {
    /// Verifying key of the child proof to be verified. This is a multi-trace verifying key.
    type ChildVerifyingKey;
    /// Type for a collection of proofs.
    type MultiProof: ?Sized;
    /// Preflight records corresponding to an instance of `MultiProof`.
    // NOTE[jpw]: we can add lifetimes if necessary
    type PreflightRecords: ?Sized;
}

/// Trait for generating the trace matrices, on device, for a given AIR module.
/// The module has a view of all proofs being verified as well as the global preflight records from
/// each proof.
///
/// This function should be expected to be called in parallel, one logical thread per module.
pub trait TraceGenModule<GC: GlobalTraceGenCtx, PB: ProverBackend>: Send + Sync {
    type ModuleSpecificCtx<'a>;

    fn generate_proving_ctxs(
        &self,
        child_vk: &GC::ChildVerifyingKey,
        proofs: &GC::MultiProof,
        preflights: &GC::PreflightRecords,
        ctx: &Self::ModuleSpecificCtx<'_>,
        required_heights: Option<&[usize]>,
    ) -> Option<Vec<AirProvingContext<PB>>>;
}

pub struct GlobalCtxCpu;

impl GlobalTraceGenCtx for GlobalCtxCpu {
    type ChildVerifyingKey = MultiStarkVerifyingKey<BabyBearPoseidon2Config>;
    type MultiProof = [Proof<BabyBearPoseidon2Config>];
    type PreflightRecords = [Preflight];
}

#[derive(Clone, Copy, Debug, Default)]
pub struct BusIndexManager {
    /// All existing buses use indices in [0, bus_idx_max)
    bus_idx_max: BusIndex,
}

impl BusIndexManager {
    pub fn new() -> Self {
        Self { bus_idx_max: 0 }
    }

    #[must_use]
    pub const fn from_next_bus_idx(bus_idx_max: BusIndex) -> Self {
        Self { bus_idx_max }
    }

    pub fn new_bus_idx(&mut self) -> BusIndex {
        let idx = self.bus_idx_max;
        self.bus_idx_max = self.bus_idx_max.checked_add(1).unwrap();
        idx
    }

    #[must_use]
    pub const fn next_bus_idx(&self) -> BusIndex {
        self.bus_idx_max
    }
}

#[derive(Clone, Debug)]
pub struct BusInventory {
    // Control flow buses
    pub transcript_bus: TranscriptBus,
    pub poseidon2_permute_bus: Poseidon2PermuteBus,
    pub poseidon2_compress_bus: Poseidon2CompressBus,
    pub merkle_verify_bus: MerkleVerifyBus,
    pub gkr_module_bus: GkrModuleBus,
    pub bc_module_bus: BatchConstraintModuleBus,
    pub stacking_module_bus: StackingModuleBus,
    pub whir_module_bus: WhirModuleBus,
    pub whir_mu_bus: WhirMuBus,

    // Data buses
    pub air_shape_bus: AirShapeBus,
    pub air_presence_bus: AirPresenceBus,
    pub hyperdim_bus: HyperdimBus,
    pub lifted_heights_bus: LiftedHeightsBus,
    pub stacking_indices_bus: StackingIndicesBus,
    pub commitments_bus: CommitmentsBus,
    pub public_values_bus: PublicValuesBus,
    pub column_claims_bus: ColumnClaimsBus,
    pub range_checker_bus: RangeCheckerBus,
    pub power_checker_bus: PowerCheckerBus,
    pub expression_claim_n_max_bus: ExpressionClaimNMaxBus,
    pub constraints_folding_input_bus: ConstraintsFoldingInputBus,
    pub interactions_folding_input_bus: InteractionsFoldingInputBus,
    pub fraction_folder_input_bus: FractionFolderInputBus,
    pub n_lift_bus: NLiftBus,
    pub eq_n_logup_n_max_bus: EqNsNLogupMaxBus,
    pub eq_3b_shape_bus: Eq3bShapeBus,

    // Randomness buses
    pub xi_randomness_bus: XiRandomnessBus,
    pub constraint_randomness_bus: ConstraintSumcheckRandomnessBus,
    pub whir_opening_point_bus: WhirOpeningPointBus,
    pub whir_opening_point_lookup_bus: WhirOpeningPointLookupBus,

    // Compute buses
    pub exp_bits_len_bus: ExpBitsLenBus,
    pub right_shift_bus: RightShiftBus,
    pub sel_uni_bus: SelUniBus,
    pub eq_neg_result_bus: EqNegResultBus,
    pub eq_neg_base_rand_bus: EqNegBaseRandBus,

    // Continuations buses
    pub cached_commit_bus: CachedCommitBus,
    pub pre_hash_bus: PreHashBus,
    pub final_state_bus: FinalTranscriptStateBus,
    /// Carries a child's final sponge state to the stage that continues it.
    pub resume_state_bus: ResumeTranscriptStateBus,
    /// Carries that child's end index, so the handover cannot also choose a
    /// prefix length. See [`TranscriptEndIndexBus`].
    pub transcript_end_index_bus: TranscriptEndIndexBus,
}

/// The records from global recursion preflight on CPU for verifying a single proof.
#[derive(Clone, Debug, Default)]
pub struct Preflight {
    /// The concatenated sequence of observes/samples. Not available during preflight; populated
    /// after.
    pub transcript: TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    /// Present exactly when the transcript log is a suffix resumed from a
    /// caller-certified checkpoint.
    pub rebased_transcript: Option<RebasedTranscriptPreflight>,
    pub proof_shape: ProofShapePreflight,
    pub gkr: GkrPreflight,
    pub batch_constraint: BatchConstraintPreflight,
    pub stacking: StackingPreflight,
    pub whir: WhirPreflight,
    pub poseidon2_perm_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub poseidon2_compress_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub initial_row_states: Vec<Vec<Vec<Vec<[F; POSEIDON2_WIDTH]>>>>,
    /// Indexed by `[round][query][coset]`. Stores post-permutation state.
    pub codeword_states: Vec<Vec<Vec<[F; POSEIDON2_WIDTH]>>>,
}

/// Backend-neutral transcript checkpoint used by a partial recursion verifier.
///
/// The operation index is absolute. `Preflight::transcript` stores only the
/// suffix beginning at this checkpoint, while all AIR-visible transcript
/// indices remain absolute.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RebasedTranscriptPreflight {
    pub start_tidx: usize,
    pub state: [F; POSEIDON2_WIDTH],
}

/// Proof material retained at the reduced-SWIRL deferred-opening boundary.
///
/// Stacking and WHIR data are deliberately absent. The conversion helper only
/// supplies empty placeholders so the established trace generators can be
/// reused without constructing a complete WHIR proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetainedLogUpOnlyProof {
    pub common_main_commit: [F; CHUNK],
    pub trace_vdata: Vec<Option<TraceVData<BabyBearPoseidon2Config>>>,
    pub public_values: Vec<Vec<F>>,
    pub gkr_proof: GkrProof<BabyBearPoseidon2Config>,
    pub batch_constraint_proof: BatchConstraintProof<BabyBearPoseidon2Config>,
}

impl From<&Proof<BabyBearPoseidon2Config>> for RetainedLogUpOnlyProof {
    fn from(proof: &Proof<BabyBearPoseidon2Config>) -> Self {
        Self {
            common_main_commit: proof.common_main_commit,
            trace_vdata: proof.trace_vdata.clone(),
            public_values: proof.public_values.clone(),
            gkr_proof: proof.gkr_proof.clone(),
            batch_constraint_proof: proof.batch_constraint_proof.clone(),
        }
    }
}

impl RetainedLogUpOnlyProof {
    /// Adapter for complete-proof trace-generator APIs. The empty tail is never read
    /// by ProofShape/GKR/BatchConstraint partial assemblies.
    #[must_use]
    pub fn into_partial_proof(self) -> Proof<BabyBearPoseidon2Config> {
        Proof {
            common_main_commit: self.common_main_commit,
            trace_vdata: self.trace_vdata,
            public_values: self.public_values,
            gkr_proof: self.gkr_proof,
            batch_constraint_proof: self.batch_constraint_proof,
            stacking_proof: StackingProof {
                univariate_round_coeffs: Vec::new(),
                sumcheck_round_polys: Vec::new(),
                stacking_openings: Vec::new(),
            },
            whir_proof: WhirProof {
                mu_pow_witness: F::ZERO,
                whir_sumcheck_polys: Vec::new(),
                codeword_commits: Vec::new(),
                ood_values: Vec::new(),
                folding_pow_witnesses: Vec::new(),
                query_phase_pow_witnesses: Vec::new(),
                initial_round_opened_rows: Vec::new(),
                initial_round_merkle_proofs: Vec::new(),
                codeword_opened_values: Vec::new(),
                codeword_merkle_proofs: Vec::new(),
                final_poly: Vec::new(),
            },
        }
    }
}

/// Complete AIR-plus-LogUp proof prefix retained before SWIRL stacking.
///
/// This is intentionally a distinct type from [`RetainedLogUpOnlyProof`]: the
/// two prefixes use different equations and cannot share a WARP relation
/// index. The empty tail bridge is private to trace generation and is never a
/// standalone proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetainedConstraintReductionProof {
    pub common_main_commit: [F; CHUNK],
    pub trace_vdata: Vec<Option<TraceVData<BabyBearPoseidon2Config>>>,
    pub public_values: Vec<Vec<F>>,
    pub gkr_proof: GkrProof<BabyBearPoseidon2Config>,
    pub batch_constraint_proof: BatchConstraintProof<BabyBearPoseidon2Config>,
}

impl RetainedConstraintReductionProof {
    /// Compatibility value consumed only by a verifier configured with
    /// [`VerifierTailMode::DeferredStacking`].
    #[must_use]
    pub fn into_partial_proof(self) -> Proof<BabyBearPoseidon2Config> {
        Proof {
            common_main_commit: self.common_main_commit,
            trace_vdata: self.trace_vdata,
            public_values: self.public_values,
            gkr_proof: self.gkr_proof,
            batch_constraint_proof: self.batch_constraint_proof,
            stacking_proof: StackingProof {
                univariate_round_coeffs: Vec::new(),
                sumcheck_round_polys: Vec::new(),
                stacking_openings: Vec::new(),
            },
            whir_proof: WhirProof {
                mu_pow_witness: F::ZERO,
                whir_sumcheck_polys: Vec::new(),
                codeword_commits: Vec::new(),
                ood_values: Vec::new(),
                folding_pow_witnesses: Vec::new(),
                query_phase_pow_witnesses: Vec::new(),
                initial_round_opened_rows: Vec::new(),
                initial_round_merkle_proofs: Vec::new(),
                codeword_opened_values: Vec::new(),
                codeword_merkle_proofs: Vec::new(),
                final_poly: Vec::new(),
            },
        }
    }
}

/// Proof prefix retained when SWIRL has completed AIR/LogUp and stacking but
/// has deliberately not run WHIR.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RetainedStackingProof {
    pub common_main_commit: [F; CHUNK],
    pub trace_vdata: Vec<Option<TraceVData<BabyBearPoseidon2Config>>>,
    pub public_values: Vec<Vec<F>>,
    pub gkr_proof: GkrProof<BabyBearPoseidon2Config>,
    pub batch_constraint_proof: BatchConstraintProof<BabyBearPoseidon2Config>,
    pub stacking_proof: StackingProof<BabyBearPoseidon2Config>,
}

impl From<&Proof<BabyBearPoseidon2Config>> for RetainedStackingProof {
    fn from(proof: &Proof<BabyBearPoseidon2Config>) -> Self {
        Self {
            common_main_commit: proof.common_main_commit,
            trace_vdata: proof.trace_vdata.clone(),
            public_values: proof.public_values.clone(),
            gkr_proof: proof.gkr_proof.clone(),
            batch_constraint_proof: proof.batch_constraint_proof.clone(),
            stacking_proof: proof.stacking_proof.clone(),
        }
    }
}

impl RetainedStackingProof {
    /// Consume a backend stacking reduction without cloning its retained PCS
    /// data. The returned proof prefix is suitable only for the recursive
    /// verifier configured with [`VerifierTailMode::DeferredWhir`]. The typed
    /// pending claim and opaque PCS owner remain inseparable and must continue
    /// into reduced-code WARP/Decide.
    #[must_use]
    pub fn split_native_reduction<PB, DeferredPcsData>(
        reduction: NativeStackingReduction<
            BabyBearPoseidon2Config,
            PB,
            (
                GkrProof<BabyBearPoseidon2Config>,
                BatchConstraintProof<BabyBearPoseidon2Config>,
            ),
            StackingProof<BabyBearPoseidon2Config>,
            Vec<EF>,
            DeferredPcsData,
        >,
    ) -> (
        Self,
        PendingConstrainedCodeWitness<[F; CHUNK], Vec<EF>, DeferredPcsData>,
    )
    where
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = [F; CHUNK]>,
    {
        let NativeStackingReduction {
            common_main_commit,
            trace_vdata,
            public_values,
            constraints_proof: (gkr_proof, batch_constraint_proof),
            stacking_proof,
            pending_witness,
        } = reduction;
        (
            Self {
                common_main_commit,
                trace_vdata,
                public_values,
                gkr_proof,
                batch_constraint_proof,
                stacking_proof,
            },
            pending_witness,
        )
    }

    /// Expose the complete `Proof` carrier required by OpenVM's established
    /// trace generator. The value cannot escape this closure and
    /// contains an empty WHIR tail, so callers cannot accidentally serialize
    /// or accept it as a complete child proof.
    pub fn with_deferred_whir_input<R>(
        &self,
        use_input: impl FnOnce(&Proof<BabyBearPoseidon2Config>) -> R,
    ) -> R {
        let input = self.clone().into_partial_proof();
        use_input(&input)
    }

    /// Private bridge to the established verifier trace generators.  The
    /// empty WHIR value is unreachable in a deferred-opening circuit.
    #[must_use]
    fn into_partial_proof(self) -> Proof<BabyBearPoseidon2Config> {
        Proof {
            common_main_commit: self.common_main_commit,
            trace_vdata: self.trace_vdata,
            public_values: self.public_values,
            gkr_proof: self.gkr_proof,
            batch_constraint_proof: self.batch_constraint_proof,
            stacking_proof: self.stacking_proof,
            whir_proof: WhirProof {
                mu_pow_witness: F::ZERO,
                whir_sumcheck_polys: Vec::new(),
                codeword_commits: Vec::new(),
                ood_values: Vec::new(),
                folding_pow_witnesses: Vec::new(),
                query_phase_pow_witnesses: Vec::new(),
                initial_round_opened_rows: Vec::new(),
                initial_round_merkle_proofs: Vec::new(),
                codeword_opened_values: Vec::new(),
                codeword_merkle_proofs: Vec::new(),
                final_poly: Vec::new(),
            },
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct ProofShapePreflight {
    pub sorted_trace_vdata: Vec<(usize, TraceVData<BabyBearPoseidon2Config>)>,
    pub starting_tidx: Vec<usize>,
    pub pvs_tidx: Vec<usize>,
    pub post_tidx: usize,
    pub n_max: usize,
    pub n_logup: usize,
    pub l_skip: usize,
}

impl ProofShapePreflight {
    pub fn n_global(&self) -> usize {
        self.n_max.max(self.n_logup)
    }
}

impl Preflight {
    #[must_use]
    pub fn transcript_base_tidx(&self) -> usize {
        self.rebased_transcript.map_or(0, |start| start.start_tidx)
    }

    /// Convert an AIR-visible absolute transcript index into the local index
    /// of the retained suffix log.
    #[must_use]
    pub fn transcript_local_tidx(&self, absolute_tidx: usize) -> usize {
        absolute_tidx
            .checked_sub(self.transcript_base_tidx())
            .expect("absolute transcript index precedes the certified rebase")
    }

    #[must_use]
    pub fn transcript_absolute_tidx(&self, local_tidx: usize) -> usize {
        self.transcript_base_tidx() + local_tidx
    }

    #[must_use]
    pub fn transcript_values_at(&self, absolute_tidx: usize, len: usize) -> &[F] {
        let local = self.transcript_local_tidx(absolute_tidx);
        &self.transcript.values()[local..local + len]
    }
}

#[derive(Clone, Debug, Default)]
pub struct GkrPreflight {
    pub post_tidx: usize,
    pub xi: Vec<(usize, EF)>,
}

#[derive(Clone, Debug, Default)]
pub struct BatchConstraintPreflight {
    pub equation_mode: VerifierEquationMode,
    pub lambda_tidx: usize,
    pub tidx_before_univariate: usize,
    pub tidx_before_multilinear: usize,
    pub tidx_before_column_openings: usize,
    pub post_tidx: usize,
    pub xi: Vec<EF>,
    pub sumcheck_rnd: Vec<EF>,
    pub eq_ns: Vec<EF>,
    pub eq_sharp_ns: Vec<EF>,
    pub eq_ns_frontloaded: Vec<EF>,
    pub eq_sharp_ns_frontloaded: Vec<EF>,
    pub final_claim: EF,
}

#[derive(Clone, Debug, Default)]
pub struct StackingPreflight {
    pub intermediate_tidx: [usize; 3],
    pub post_tidx: usize,
    pub univariate_poly_rand_eval: EF,
    pub stacking_batching_challenge: EF,
    /// PoW witness for μ batching challenge.
    pub mu_pow_witness: F,
    /// PoW sample for μ batching challenge.
    pub mu_pow_sample: F,
    pub lambda: EF,
    pub sumcheck_rnd: Vec<EF>,
}

#[derive(Clone, Debug, Default)]
pub struct WhirPreflight {
    pub whir_round_tidx_per_round: Vec<usize>,
    pub query_tidx_per_round: Vec<usize>,
    pub alphas: Vec<EF>,
    pub z0s: Vec<EF>,
    pub gammas: Vec<EF>,
    pub folding_pow_samples: Vec<F>,
    pub query_pow_samples: Vec<F>,
    pub queries: Vec<F>,
}

impl BusInventory {
    pub fn new(b: &mut BusIndexManager) -> Self {
        Self {
            transcript_bus: TranscriptBus::new(b.new_bus_idx()),
            poseidon2_permute_bus: Poseidon2PermuteBus::new(b.new_bus_idx()),
            poseidon2_compress_bus: Poseidon2CompressBus::new(b.new_bus_idx()),
            merkle_verify_bus: MerkleVerifyBus::new(b.new_bus_idx()),

            // Control flow buses
            gkr_module_bus: GkrModuleBus::new(b.new_bus_idx()),
            bc_module_bus: BatchConstraintModuleBus::new(b.new_bus_idx()),
            stacking_module_bus: StackingModuleBus::new(b.new_bus_idx()),
            whir_module_bus: WhirModuleBus::new(b.new_bus_idx()),
            whir_mu_bus: WhirMuBus::new(b.new_bus_idx()),

            // Data buses
            air_shape_bus: AirShapeBus::new(b.new_bus_idx()),
            air_presence_bus: AirPresenceBus::new(b.new_bus_idx()),
            hyperdim_bus: HyperdimBus::new(b.new_bus_idx()),
            lifted_heights_bus: LiftedHeightsBus::new(b.new_bus_idx()),
            stacking_indices_bus: StackingIndicesBus::new(b.new_bus_idx()),
            commitments_bus: CommitmentsBus::new(b.new_bus_idx()),
            public_values_bus: PublicValuesBus::new(b.new_bus_idx()),
            sel_uni_bus: SelUniBus::new(b.new_bus_idx()),
            range_checker_bus: RangeCheckerBus::new(b.new_bus_idx()),
            power_checker_bus: PowerCheckerBus::new(b.new_bus_idx()),
            expression_claim_n_max_bus: ExpressionClaimNMaxBus::new(b.new_bus_idx()),
            constraints_folding_input_bus: ConstraintsFoldingInputBus::new(b.new_bus_idx()),
            interactions_folding_input_bus: InteractionsFoldingInputBus::new(b.new_bus_idx()),
            fraction_folder_input_bus: FractionFolderInputBus::new(b.new_bus_idx()),
            n_lift_bus: NLiftBus::new(b.new_bus_idx()),
            eq_n_logup_n_max_bus: EqNsNLogupMaxBus::new(b.new_bus_idx()),
            eq_3b_shape_bus: Eq3bShapeBus::new(b.new_bus_idx()),

            // Randomness buses
            xi_randomness_bus: XiRandomnessBus::new(b.new_bus_idx()),
            constraint_randomness_bus: ConstraintSumcheckRandomnessBus::new(b.new_bus_idx()),
            whir_opening_point_bus: WhirOpeningPointBus::new(b.new_bus_idx()),
            whir_opening_point_lookup_bus: WhirOpeningPointLookupBus::new(b.new_bus_idx()),

            // Claims buses
            column_claims_bus: ColumnClaimsBus::new(b.new_bus_idx()),

            exp_bits_len_bus: ExpBitsLenBus::new(b.new_bus_idx()),
            right_shift_bus: RightShiftBus::new(b.new_bus_idx()),
            eq_neg_base_rand_bus: EqNegBaseRandBus::new(b.new_bus_idx()),
            eq_neg_result_bus: EqNegResultBus::new(b.new_bus_idx()),

            // Continuation buses
            cached_commit_bus: CachedCommitBus::new(b.new_bus_idx()),
            pre_hash_bus: PreHashBus::new(b.new_bus_idx()),
            final_state_bus: FinalTranscriptStateBus::new(b.new_bus_idx()),
            resume_state_bus: ResumeTranscriptStateBus::new(b.new_bus_idx()),
            transcript_end_index_bus: TranscriptEndIndexBus::new(b.new_bus_idx()),
        }
    }
}

/// A pre-state/post-state pair for a single Poseidon permutation.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct PoseidonStatePair {
    pub pre_state: [F; POSEIDON2_WIDTH],
    pub post_state: [F; POSEIDON2_WIDTH],
}

struct MerklePrecomputation {
    poseidon2_perm_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    poseidon2_compress_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    initial_row_states: Vec<Vec<Vec<Vec<[F; POSEIDON2_WIDTH]>>>>,
    codeword_states: Vec<Vec<Vec<[F; POSEIDON2_WIDTH]>>>,
}

#[derive(Clone, Copy, strum_macros::Display)]
enum TraceModuleRef<'a, const TRANSCRIPT_SBOX_REGISTERS: usize = 1> {
    Transcript(&'a TranscriptModule<TRANSCRIPT_SBOX_REGISTERS>),
    ProofShape(&'a ProofShapeModule),
    Gkr(&'a GkrModule),
    BatchConstraint(&'a BatchConstraintModule),
    Stacking(&'a StackingModule),
    Whir(&'a WhirModule),
}

impl<'a, const TRANSCRIPT_SBOX_REGISTERS: usize> TraceModuleRef<'a, TRANSCRIPT_SBOX_REGISTERS> {
    #[tracing::instrument(
        name = "wrapper.run_preflight",
        level = "trace",
        skip_all,
        fields(air_module = %self)
    )]
    fn run_preflight<TS>(
        self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proof: &Proof<BabyBearPoseidon2Config>,
        preflight: &mut Preflight,
        sponge: &mut TS,
    ) where
        TS: FiatShamirTranscript<BabyBearPoseidon2Config>
            + TranscriptHistory<F = F, State = [F; POSEIDON2_WIDTH]>,
    {
        match self {
            TraceModuleRef::ProofShape(module) => {
                module.run_preflight(child_vk, proof, preflight, sponge)
            }
            TraceModuleRef::Gkr(module) => module.run_preflight(proof, preflight, sponge),
            TraceModuleRef::BatchConstraint(module) => {
                module.run_preflight(child_vk, proof, preflight, sponge)
            }
            TraceModuleRef::Stacking(module) => module.run_preflight(proof, preflight, sponge),
            TraceModuleRef::Whir(module) => module.run_preflight(proof, preflight, sponge),
            _ => panic!("TraceModuleRef::run_preflight called with invalid module"),
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(
        name = "wrapper.generate_proving_ctxs",
        level = "trace",
        skip_all,
        fields(air_module = %self)
    )]
    fn generate_cpu_ctxs<SC: StarkProtocolConfig<F = F>>(
        self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        preflights: &[Preflight],
        cached_trace_record: Option<&CachedTraceRecord>,
        pow_checker_gen: &Arc<PowerCheckerCpuTraceGenerator<2, POW_CHECKER_HEIGHT>>,
        exp_bits_len_gen: &ExpBitsLenCpuTraceGenerator,
        external_data: &VerifierExternalData,
        required_heights: Option<&[usize]>,
    ) -> Option<Vec<AirProvingContext<CpuBackend<SC>>>> {
        match self {
            TraceModuleRef::Transcript(module) => module.generate_proving_ctxs(
                child_vk,
                proofs,
                preflights,
                &(
                    external_data.poseidon2_permute_inputs,
                    external_data.poseidon2_compress_inputs,
                ),
                required_heights,
            ),
            TraceModuleRef::ProofShape(module) => module.generate_proving_ctxs(
                child_vk,
                proofs,
                preflights,
                &(
                    pow_checker_gen.clone(),
                    external_data.range_check_inputs.as_slice(),
                ),
                required_heights,
            ),
            TraceModuleRef::Gkr(module) => module.generate_proving_ctxs(
                child_vk,
                proofs,
                preflights,
                exp_bits_len_gen,
                required_heights,
            ),
            TraceModuleRef::BatchConstraint(module) => module.generate_proving_ctxs(
                child_vk,
                proofs,
                preflights,
                &(cached_trace_record, pow_checker_gen.clone()),
                required_heights,
            ),
            TraceModuleRef::Stacking(module) => {
                module.generate_proving_ctxs(child_vk, proofs, preflights, &(), required_heights)
            }
            TraceModuleRef::Whir(module) => module.generate_proving_ctxs(
                child_vk,
                proofs,
                preflights,
                exp_bits_len_gen,
                required_heights,
            ),
        }
    }
}

/// The recursive verifier sub-circuit consists of multiple chips, grouped into **modules**.
///
/// This circuit supports child verifying keys through `RECURSION_MAX_CHILD_AIRS`.
/// `ProofShapeAir` range-checks AIR-index gaps with a dedicated lookup when enforcing sorted
/// proof-shape rows; large one-proof keys authenticate their VK metadata through a preprocessed
/// table rather than an AIR-count-sized selector.
///
/// This struct is stateful.
pub struct VerifierSubCircuit<
    const MAX_NUM_PROOFS: usize,
    const TRANSCRIPT_SBOX_REGISTERS: usize = 1,
> {
    bus_inventory: BusInventory,
    bus_idx_manager: BusIndexManager,

    transcript: TranscriptModule<TRANSCRIPT_SBOX_REGISTERS>,
    proof_shape: ProofShapeModule,
    gkr: GkrModule,
    batch_constraint: BatchConstraintModule,
    stacking: StackingModule,
    whir: WhirModule,
    tail_mode: VerifierTailMode,
    deferred_checkpoint: Option<DeferredOpeningCheckpointAir<MAX_NUM_PROOFS>>,
    constraint_claims: Option<ConstraintReductionClaimsAir>,
    constraint_checkpoint: Option<ConstraintReductionCheckpointAir<MAX_NUM_PROOFS>>,
    deferred_stacking_endpoint: Option<crate::native_warp::NativeReductionEndpointInputBus>,
}

impl<const MAX_NUM_PROOFS: usize, const TRANSCRIPT_SBOX_REGISTERS: usize>
    VerifierSubCircuit<MAX_NUM_PROOFS, TRANSCRIPT_SBOX_REGISTERS>
{
    /// Configure the exact extra exports consumed by
    /// `ReducedSwirlSourceAir` and the ordered source-receipt AIR.
    ///
    /// The child proof still stops before WHIR. Commitments, stacking point,
    /// openings and proof shape are merely given one additional in-circuit
    /// consumer; no host acceptance flag is introduced.
    pub fn configure_deferred_swirl_source_exports(
        &mut self,
    ) -> Result<crate::native_warp::NativeReductionEndpointInputBus, &'static str> {
        if self.tail_mode != VerifierTailMode::DeferredWhir {
            return Err("deferred-SWIRL exports require the DeferredWhir verifier tail");
        }
        let endpoint = self
            .deferred_stacking_endpoint
            .ok_or("deferred-SWIRL stacking endpoint")?;
        self.proof_shape
            .set_partial_assembly_exports_with_stacking(1);
        self.proof_shape.set_layout_export_lookups(1);
        // The receipt replaces the public deferred-opening checkpoint AIR.
        // Export the exact transcript length so its private checkpoint cannot
        // select an earlier all-sample window while separately consuming the
        // final sponge state.
        self.transcript.enable_end_index_bus();
        self.stacking
            .set_partial_assembly_exports(crate::stacking::StackingNativeExports {
                endpoint_input_bus: endpoint,
                point_lookups: 1,
                opening_lookups: 1,
            });
        Ok(endpoint)
    }

    /// AIR index, relative to this verifier sub-circuit, of the typed
    /// post-stacking transcript checkpoint.
    ///
    /// The enclosing inner aggregation circuit prepends its three public
    /// statement AIRs.  SDK adapters should add that prefix and then locate
    /// the proving context by AIR id; they must not depend on trace sorting.
    #[must_use]
    pub fn deferred_opening_checkpoint_air_index(&self) -> Option<usize> {
        self.deferred_checkpoint.map(|_| {
            self.batch_constraint.num_airs()
                + self.transcript.num_airs()
                + self.proof_shape.num_airs()
                + self.gkr.num_airs()
                + usize::from(self.tail_mode.includes_stacking()) * self.stacking.num_airs()
                + usize::from(self.tail_mode.includes_whir()) * self.whir.num_airs()
        })
    }

    /// Bind the reconstructed no-cached symbolic-expression DAG to a
    /// verifier-key-owned digest.
    ///
    /// This removes the DAG digest from the outer proof's public-value
    /// schedule without trusting the prover: [`DagCommitSubAir`] still hashes
    /// every symbolic node and constrains the terminal digest to `expected`.
    /// Cached mode is deliberately incompatible with this authority model.
    pub fn bind_fixed_dag_commit(
        &mut self,
        expected: [F; openvm_stark_sdk::config::baby_bear_poseidon2::DIGEST_SIZE],
    ) -> Result<(), &'static str> {
        self.batch_constraint.bind_fixed_dag_commit(expected)
    }

    /// Backend-independent verifier-key trace material used both by CPU/CUDA
    /// trace generation and by outer public-value validation.
    pub fn cached_trace_record_for_child(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> CachedTraceRecord {
        self.batch_constraint.cached_trace_record(child_vk)
    }

    /// The Poseidon lookup namespace owned by this verifier. Final recursive
    /// assemblies may keep the verifier's own table local while adding a
    /// second physical table over this same logical bus for wrapper and
    /// terminal requests.
    #[must_use]
    pub fn poseidon2_bus_owner(&self) -> Poseidon2BusOwner {
        self.transcript.poseidon2_bus_owner()
    }

    /// Position of the verifier-local physical Poseidon table in
    /// [`AggregationSubCircuit::airs`].
    #[must_use]
    pub fn poseidon_air_index<SC: StarkProtocolConfig<F = F>>(&self) -> usize {
        self.batch_constraint.airs::<SC>().len() + 1
    }

    /// Construct an additional physical Poseidon table over an ordered list
    /// of logical owners. This does not remove the verifier-local table.
    #[must_use]
    pub fn multi_bus_poseidon_air<SC: StarkProtocolConfig<F = F>>(
        &self,
        owners: &[Poseidon2BusOwner],
    ) -> AirRef<SC> {
        self.transcript.multi_bus_poseidon2_air_for_owners(owners)
    }

    /// Build setup-fixed sharded traces for an additional physical Poseidon
    /// table used by an enclosing final proof.
    pub fn build_poseidon2_multibus_sharded_traces(
        &self,
        grouped_inputs: Poseidon2MultibusInputs,
        shard_count: usize,
        max_rows: usize,
    ) -> Option<Vec<p3_matrix::dense::RowMajorMatrix<F>>> {
        self.transcript.build_poseidon2_multibus_sharded_traces(
            grouped_inputs,
            shard_count,
            max_rows,
        )
    }

    pub fn new(child_mvk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>) -> Self {
        Self::new_with_options(child_mvk, VerifierConfig::default())
    }

    pub fn new_with_options(
        child_mvk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
        config: VerifierConfig,
    ) -> Self {
        Self::new_with_options_from_bus_idx_manager(child_mvk, config, BusIndexManager::new())
    }

    /// Builds the verifier while allocating every owned bus from the supplied manager.
    ///
    /// The manager's [`BusIndexManager::next_bus_idx`] is the first bus index available to this
    /// verifier. After construction, [`AggregationSubCircuit::next_bus_idx`] returns the first
    /// index available to the next sub-circuit. This permits a direct circuit and a recursive
    /// verifier to share one AIR assembly without overlapping bus ranges.
    pub fn new_with_options_from_bus_idx_manager(
        child_mvk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
        config: VerifierConfig,
        mut bus_idx_manager: BusIndexManager,
    ) -> Self {
        // The verifier must enforce the child VK's linear `trace_height_constraints`.
        //
        // This recursion verifier circuit enforces one summary-row in-circuit bound:
        //   sum_i(num_interactions[i] * lifted_height[i]) < max_interaction_count
        // with `lifted_height[i] = max(trace_height[i], 2^l_skip)`.
        //
        // At verifier-circuit construction time, each child `trace_height_constraint` must be
        // implied by this bound. If not, we panic and refuse to construct the circuit.
        let proof_shape_constraint = LinearConstraint {
            coefficients: child_mvk
                .inner
                .per_air
                .iter()
                .map(|avk| avk.num_interactions() as u32)
                .collect(),
            threshold: child_mvk.inner.params.logup.max_interaction_count,
        };
        for (i, constraint) in child_mvk.inner.trace_height_constraints.iter().enumerate() {
            assert!(
                constraint.is_implied_by(&proof_shape_constraint),
                "child_vk trace_height_constraint[{i}] is not implied by ProofShapeAir's check. \
                 The recursion circuit cannot enforce this constraint. \
                 Constraint: coefficients={:?}, threshold={}",
                constraint.coefficients,
                constraint.threshold,
            );
        }

        let bus_inventory = BusInventory::new(&mut bus_idx_manager);

        let mut transcript = TranscriptModule::new(
            bus_inventory.clone(),
            child_mvk.inner.params.clone(),
            config.final_state_bus_enabled || config.tail_mode.is_deferred(),
            // The generic recursion verifier always starts its transcripts at
            // the canonical zero sponge; only the native WARP history ladder
            // continues one.
            false,
        );
        let child_mvk_frame = child_mvk.as_ref().into();
        let mut proof_shape = ProofShapeModule::new(
            &child_mvk_frame,
            &mut bus_idx_manager,
            bus_inventory.clone(),
            config.continuations_enabled,
            MAX_NUM_PROOFS,
        );
        let gkr = GkrModule::new(&child_mvk, &mut bus_idx_manager, bus_inventory.clone());
        let batch_constraint = if config.tail_mode == VerifierTailMode::DeferredStacking {
            BatchConstraintModule::new_deferred_stacking(
                &child_mvk,
                &mut bus_idx_manager,
                bus_inventory.clone(),
                MAX_NUM_PROOFS,
                config.has_cached,
            )
        } else {
            BatchConstraintModule::new(
                &child_mvk,
                &mut bus_idx_manager,
                bus_inventory.clone(),
                MAX_NUM_PROOFS,
                config.has_cached,
            )
        };
        let mut stacking =
            StackingModule::new(&child_mvk, &mut bus_idx_manager, bus_inventory.clone());
        let whir = WhirModule::new(&child_mvk, &mut bus_idx_manager, bus_inventory.clone());
        let mut deferred_stacking_endpoint = None;
        let deferred_checkpoint = (config.tail_mode == VerifierTailMode::DeferredWhir).then(|| {
            // Roots and stacking claims remain ordinary transcript inputs, but
            // WHIR no longer consumes their lookup fanout in this circuit.
            proof_shape.set_partial_assembly_exports_with_stacking(0);
            let endpoint = crate::native_warp::NativeReductionEndpointInputBus::new(
                bus_idx_manager.new_bus_idx(),
            );
            deferred_stacking_endpoint = Some(endpoint);
            stacking.set_partial_assembly_exports(crate::stacking::StackingNativeExports {
                endpoint_input_bus: endpoint,
                point_lookups: 0,
                opening_lookups: 0,
            });
            transcript.disable_merkle_verify();
            DeferredOpeningCheckpointAir::new(
                bus_inventory.transcript_bus,
                bus_inventory.final_state_bus,
            )
        });
        let (constraint_claims, constraint_checkpoint) =
            if config.tail_mode == VerifierTailMode::DeferredStacking {
                proof_shape.set_partial_assembly_exports(0);
                transcript.enable_end_index_bus();
                transcript.disable_merkle_verify();
                (
                    Some(ConstraintReductionClaimsAir::new(
                        bus_inventory.stacking_module_bus,
                        bus_inventory.column_claims_bus,
                        bus_inventory.transcript_bus,
                    )),
                    Some(ConstraintReductionCheckpointAir::new(
                        bus_inventory.final_state_bus,
                        bus_inventory.transcript_end_index_bus,
                    )),
                )
            } else {
                (None, None)
            };

        VerifierSubCircuit {
            bus_inventory,
            bus_idx_manager,
            transcript,
            proof_shape,
            gkr,
            batch_constraint,
            stacking,
            whir,
            tail_mode: config.tail_mode,
            deferred_checkpoint,
            constraint_claims,
            constraint_checkpoint,
            deferred_stacking_endpoint,
        }
    }

    #[tracing::instrument(name = "execute_preflight", skip_all)]
    fn run_preflight_without_merkle<TS>(
        &self,
        mut sponge: TS,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proof: &Proof<BabyBearPoseidon2Config>,
    ) -> Preflight
    where
        TS: FiatShamirTranscript<BabyBearPoseidon2Config>
            + TranscriptHistory<F = F, State = [F; POSEIDON2_WIDTH]>,
    {
        let mut preflight = Preflight::default();

        // NOTE: it is not required that we group preflight into modules
        let preflight_modules: [TraceModuleRef<'_, TRANSCRIPT_SBOX_REGISTERS>; 3] = [
            TraceModuleRef::ProofShape(&self.proof_shape),
            TraceModuleRef::Gkr(&self.gkr),
            TraceModuleRef::BatchConstraint(&self.batch_constraint),
        ];
        for module in &preflight_modules {
            module.run_preflight(child_vk, proof, &mut preflight, &mut sponge);
        }
        match self.tail_mode {
            VerifierTailMode::Complete => {
                self.stacking
                    .run_preflight(proof, &mut preflight, &mut sponge);
                self.whir.run_preflight(proof, &mut preflight, &mut sponge);
            }
            VerifierTailMode::DeferredWhir => {
                self.stacking
                    .run_preflight_without_mu_tail(proof, &mut preflight, &mut sponge);
                // The post-stacking compatibility mode retains its explicit
                // terminal squeeze. New protocols should prefer the typed
                // pre-stacking manifest below.
                let _checkpoint_challenge = sponge.sample_ext();
            }
            VerifierTailMode::DeferredStacking => {}
        }
        preflight.transcript = sponge.into_log();

        preflight
    }

    /// Runs preflight for a single proof.
    #[tracing::instrument(name = "execute_preflight", skip_all)]
    pub fn run_preflight<TS>(
        &self,
        sponge: TS,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        proof: &Proof<BabyBearPoseidon2Config>,
    ) -> Preflight
    where
        TS: FiatShamirTranscript<BabyBearPoseidon2Config>
            + TranscriptHistory<F = F, State = [F; POSEIDON2_WIDTH]>,
    {
        let mut preflight = self.run_preflight_without_merkle(sponge, child_vk, proof);
        if self.tail_mode == VerifierTailMode::Complete {
            Self::apply_merkle_precomputation_cpu(proof, &mut preflight);
        }

        preflight
    }

    fn apply_merkle_precomputation_cpu(
        proof: &Proof<BabyBearPoseidon2Config>,
        preflight: &mut Preflight,
    ) {
        let merkle_precomputation = Self::compute_merkle_precomputation(proof);
        preflight.poseidon2_perm_inputs = merkle_precomputation.poseidon2_perm_inputs;
        preflight.poseidon2_compress_inputs = merkle_precomputation.poseidon2_compress_inputs;
        preflight.initial_row_states = merkle_precomputation.initial_row_states;
        preflight.codeword_states = merkle_precomputation.codeword_states;
    }

    #[cfg(feature = "cuda")]
    #[tracing::instrument(name = "apply_merkle_precomputation", skip_all)]
    fn apply_merkle_precomputation(
        proof: &Proof<BabyBearPoseidon2Config>,
        preflight: &mut Preflight,
        device_ctx: &GpuDeviceCtx,
    ) {
        let merkle_precomputation = Self::compute_merkle_precomputation_cuda(proof, device_ctx);
        preflight.poseidon2_perm_inputs = merkle_precomputation.poseidon2_perm_inputs;
        preflight.poseidon2_compress_inputs = merkle_precomputation.poseidon2_compress_inputs;
        preflight.initial_row_states = merkle_precomputation.initial_row_states;
        preflight.codeword_states = merkle_precomputation.codeword_states;
    }

    #[cfg_attr(feature = "cuda", allow(dead_code))]
    #[tracing::instrument(name = "compute_merkle_precomputation", level = "info", skip_all)]
    fn compute_merkle_precomputation(
        proof: &Proof<BabyBearPoseidon2Config>,
    ) -> MerklePrecomputation {
        let initial_chunks: usize = proof
            .whir_proof
            .initial_round_opened_rows
            .iter()
            .flat_map(|c| c.iter().flat_map(|q| q.iter()))
            .map(|row| row.len().div_ceil(CHUNK))
            .sum();
        let codeword_chunks: usize = proof
            .whir_proof
            .codeword_opened_values
            .iter()
            .map(|r| r.iter().map(|q| q.len()).sum::<usize>())
            .sum();

        // InitialOpenedValuesAir (initial_row_states) does Poseidon2 *permute* lookups per chunk.
        // NonInitialOpenedValuesAir (codeword_states) does Poseidon2 *compress* lookups per value.
        let mut poseidon2_perm_inputs = Vec::with_capacity(initial_chunks);
        let mut poseidon2_compress_inputs = Vec::with_capacity(codeword_chunks);

        let initial_row_states: Vec<Vec<Vec<Vec<[F; POSEIDON2_WIDTH]>>>> = proof
            .whir_proof
            .initial_round_opened_rows
            .iter()
            .map(|opened_rows_per_commit| {
                opened_rows_per_commit
                    .iter()
                    .map(|opened_rows_per_query| {
                        opened_rows_per_query
                            .iter()
                            .map(|opened_row| {
                                let (_leaf_hash, pre_states, post_states) =
                                    poseidon2_hash_slice_with_states(opened_row);
                                poseidon2_perm_inputs.extend(pre_states);
                                post_states
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let codeword_states = proof
            .whir_proof
            .codeword_opened_values
            .iter()
            .map(|round_values| {
                round_values
                    .iter()
                    .map(|opened_values_per_query| {
                        opened_values_per_query
                            .iter()
                            .map(|opened_value| {
                                let (_leaf_hash, pre_states, post_states) =
                                    poseidon2_hash_slice_with_states(
                                        opened_value.as_basis_coefficients_slice(),
                                    );
                                // This is not quite a compression, but the AIR will constrain that
                                // the padded pre_state gets
                                // compressed into _leaf_hash via Poseidon2CompressBus.
                                poseidon2_compress_inputs.extend(pre_states);
                                debug_assert_eq!(post_states.len(), 1);
                                post_states.into_iter().next().unwrap()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        MerklePrecomputation {
            poseidon2_perm_inputs,
            poseidon2_compress_inputs,
            initial_row_states,
            codeword_states,
        }
    }

    #[cfg(feature = "cuda")]
    #[tracing::instrument(name = "compute_merkle_precomputation_cuda", level = "info", skip_all)]
    fn compute_merkle_precomputation_cuda(
        proof: &Proof<BabyBearPoseidon2Config>,
        device_ctx: &GpuDeviceCtx,
    ) -> MerklePrecomputation {
        use openvm_cuda_common::{
            copy::{MemCopyD2H, MemCopyH2D},
            d_buffer::DeviceBuffer,
        };

        use crate::cuda::abi::{merkle_precomputation_hash_vectors, VectorDescriptor};

        let num_chunks = |len: usize| len.div_ceil(CHUNK);

        let mut num_vectors = 0usize;
        let mut total_data_len = 0usize;
        let mut total_chunks = 0usize;

        for row in proof
            .whir_proof
            .initial_round_opened_rows
            .iter()
            .flat_map(|per_commit| per_commit.iter().flat_map(|per_query| per_query.iter()))
        {
            num_vectors += 1;
            total_data_len += row.len();
            total_chunks += num_chunks(row.len());
        }
        let num_perm_chunks = total_chunks;

        for opened_value in proof
            .whir_proof
            .codeword_opened_values
            .iter()
            .flat_map(|per_round| per_round.iter().flat_map(|per_query| per_query.iter()))
        {
            let len = <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(opened_value).len();
            num_vectors += 1;
            total_data_len += len;
            total_chunks += num_chunks(len);
        }

        let mut flat_data = Vec::with_capacity(total_data_len);
        let mut descriptors = Vec::with_capacity(num_vectors);
        let mut output_offset_chunks = 0usize;

        let mut push_vector = |data: &[F]| {
            let len = data.len();
            let chunks = num_chunks(len);
            descriptors.push(VectorDescriptor {
                data_offset: flat_data.len(),
                len,
                output_offset: output_offset_chunks,
            });
            output_offset_chunks += chunks;
            flat_data.extend_from_slice(data);
        };

        for row in proof
            .whir_proof
            .initial_round_opened_rows
            .iter()
            .flat_map(|per_commit| per_commit.iter().flat_map(|per_query| per_query.iter()))
        {
            push_vector(row);
        }
        for opened_value in proof
            .whir_proof
            .codeword_opened_values
            .iter()
            .flat_map(|per_round| per_round.iter().flat_map(|per_query| per_query.iter()))
        {
            push_vector(opened_value.as_basis_coefficients_slice());
        }

        debug_assert_eq!(descriptors.len(), num_vectors);
        debug_assert_eq!(flat_data.len(), total_data_len);
        debug_assert_eq!(output_offset_chunks, total_chunks);

        // Upload to GPU and run kernel on the caller-owned stream.
        let d_data = flat_data
            .to_device_on(device_ctx)
            .expect("failed to upload data");
        let d_descriptors = descriptors
            .to_device_on(device_ctx)
            .expect("failed to upload descriptors");
        let d_pre_states =
            DeviceBuffer::<F>::with_capacity_on(total_chunks * POSEIDON2_WIDTH, device_ctx);
        let d_post_states =
            DeviceBuffer::<F>::with_capacity_on(total_chunks * POSEIDON2_WIDTH, device_ctx);

        unsafe {
            merkle_precomputation_hash_vectors(
                &d_data,
                &d_descriptors,
                num_vectors,
                &d_pre_states,
                &d_post_states,
                device_ctx.stream.as_raw(),
            )
            .expect("hash_vectors kernel failed");
        }

        // Download results
        let pre_states_flat = d_pre_states
            .to_host_on(device_ctx)
            .expect("failed to download pre_states");
        let post_states_flat = d_post_states
            .to_host_on(device_ctx)
            .expect("failed to download post_states");
        debug_assert_eq!(pre_states_flat.len(), total_chunks * POSEIDON2_WIDTH);
        debug_assert_eq!(post_states_flat.len(), total_chunks * POSEIDON2_WIDTH);

        // Split pre_states into poseidon permute and compress inputs
        let (perm_flat, compress_flat) =
            pre_states_flat.split_at(num_perm_chunks * POSEIDON2_WIDTH);
        let poseidon2_perm_inputs: Vec<[F; POSEIDON2_WIDTH]> = perm_flat
            .chunks_exact(POSEIDON2_WIDTH)
            .map(|chunk| chunk.try_into().unwrap())
            .collect();
        let poseidon2_compress_inputs: Vec<[F; POSEIDON2_WIDTH]> = compress_flat
            .chunks_exact(POSEIDON2_WIDTH)
            .map(|chunk| chunk.try_into().unwrap())
            .collect();

        let mut post_iter = post_states_flat.chunks_exact(POSEIDON2_WIDTH);

        let initial_row_states: Vec<Vec<Vec<Vec<[F; POSEIDON2_WIDTH]>>>> = proof
            .whir_proof
            .initial_round_opened_rows
            .iter()
            .map(|per_commit| {
                per_commit
                    .iter()
                    .map(|per_query| {
                        per_query
                            .iter()
                            .map(|row| {
                                (0..num_chunks(row.len()))
                                    .map(|_| post_iter.next().unwrap().try_into().unwrap())
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        let codeword_states: Vec<Vec<Vec<[F; POSEIDON2_WIDTH]>>> = proof
            .whir_proof
            .codeword_opened_values
            .iter()
            .map(|per_round| {
                per_round
                    .iter()
                    .map(|per_query| {
                        per_query
                            .iter()
                            .map(|opened_value| {
                                debug_assert_eq!(
                                    num_chunks(
                                        <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(
                                            opened_value
                                        )
                                        .len()
                                    ),
                                    1
                                );
                                post_iter.next().unwrap().try_into().unwrap()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();

        debug_assert_eq!(post_iter.len(), 0);

        MerklePrecomputation {
            poseidon2_perm_inputs,
            poseidon2_compress_inputs,
            initial_row_states,
            codeword_states,
        }
    }

    /// Utility function to split a slice of required trace heights per-module. Fails
    /// an assert if the slice length doesn't match the number of AIRs.
    #[allow(clippy::type_complexity)]
    fn split_required_heights<'a>(
        &self,
        required_heights: Option<&'a [usize]>,
    ) -> (
        Vec<Option<&'a [usize]>>,
        Option<&'a [usize]>,
        Option<usize>,
        Option<usize>,
    ) {
        let bc_n = self.batch_constraint.num_airs();
        let t_n = self.transcript.num_airs();
        let ps_n = self.proof_shape.num_airs();
        let gkr_n = self.gkr.num_airs();
        let st_n = usize::from(self.tail_mode.includes_stacking()) * self.stacking.num_airs();
        let w_n = usize::from(self.tail_mode.includes_whir()) * self.whir.num_airs();
        let module_air_counts = [bc_n, t_n, ps_n, gkr_n, st_n, w_n];
        let statement_n = usize::from(self.deferred_checkpoint.is_some())
            + usize::from(self.constraint_claims.is_some())
            + usize::from(self.constraint_checkpoint.is_some());

        let Some(heights) = required_heights else {
            return (vec![None; module_air_counts.len()], None, None, None);
        };

        let total_module_airs: usize = module_air_counts.iter().sum();
        let total = total_module_airs + statement_n + 2; // statements + primitives
        assert_eq!(heights.len(), total);

        let mut offset = 0usize;
        let mut per_module = Vec::with_capacity(module_air_counts.len());
        for n in module_air_counts {
            per_module.push(Some(&heights[offset..offset + n]));
            offset += n;
        }
        let statement_required = Some(&heights[offset..offset + statement_n]);
        offset += statement_n;
        debug_assert_eq!(heights.len() - offset, 2);

        (
            per_module,
            statement_required,
            Some(heights[offset]),
            Some(heights[offset + 1]),
        )
    }
}

impl<const MAX_NUM_PROOFS: usize, const TRANSCRIPT_SBOX_REGISTERS: usize> AggregationSubCircuit
    for VerifierSubCircuit<MAX_NUM_PROOFS, TRANSCRIPT_SBOX_REGISTERS>
{
    fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        let exp_bits_len_air = ExpBitsLenAir::new(
            self.bus_inventory.exp_bits_len_bus,
            self.bus_inventory.right_shift_bus,
        );
        let power_checker_air = PowerCheckerAir::<2, POW_CHECKER_HEIGHT> {
            pow_bus: self.bus_inventory.power_checker_bus,
            range_bus: self.bus_inventory.range_checker_bus,
        };

        // WARNING: SymbolicExpressionAir MUST be the first AIR in verifier circuit.
        let mut airs = iter::empty()
            .chain(self.batch_constraint.airs())
            .chain(self.transcript.airs())
            .chain(self.proof_shape.airs())
            .chain(self.gkr.airs())
            .collect::<Vec<_>>();
        if self.tail_mode.includes_stacking() {
            airs.extend(self.stacking.airs());
        }
        if self.tail_mode.includes_whir() {
            airs.extend(self.whir.airs());
        }
        if let Some(checkpoint) = self.deferred_checkpoint {
            airs.push(Arc::new(checkpoint) as AirRef<_>);
        }
        if let Some(claims) = self.constraint_claims {
            airs.push(Arc::new(claims) as AirRef<_>);
        }
        if let Some(checkpoint) = self.constraint_checkpoint {
            airs.push(Arc::new(checkpoint) as AirRef<_>);
        }
        airs.extend([
            Arc::new(power_checker_air) as AirRef<_>,
            Arc::new(exp_bits_len_air) as AirRef<_>,
        ]);
        airs
    }

    fn bus_inventory(&self) -> &BusInventory {
        &self.bus_inventory
    }

    fn next_bus_idx(&self) -> BusIndex {
        self.bus_idx_manager.bus_idx_max
    }

    fn max_num_proofs(&self) -> usize {
        MAX_NUM_PROOFS
    }
}

impl<
        SC: StarkProtocolConfig<F = F>,
        const MAX_NUM_PROOFS: usize,
        const TRANSCRIPT_SBOX_REGISTERS: usize,
    > VerifierTraceGen<CpuBackend<SC>, SC, ()>
    for VerifierSubCircuit<MAX_NUM_PROOFS, TRANSCRIPT_SBOX_REGISTERS>
{
    fn new(
        child_mvk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
        config: VerifierConfig,
    ) -> Self {
        Self::new_with_options(child_mvk, config)
    }

    fn commit_child_vk<E: StarkEngine<SC = SC, PB = CpuBackend<SC>>>(
        &self,
        engine: &E,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> CommittedTraceData<CpuBackend<SC>>
    where
        (): From<EngineDeviceCtx<E>>,
    {
        self.batch_constraint.commit_child_vk(engine, child_vk)
    }

    fn cached_trace_record(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
    ) -> CachedTraceRecord {
        self.batch_constraint.cached_trace_record(child_vk)
    }

    #[tracing::instrument(name = "subcircuit_generate_proving_ctxs", skip_all)]
    fn generate_proving_ctxs<
        TS: FiatShamirTranscript<BabyBearPoseidon2Config>
            + TranscriptHistory<F = F, State = [F; POSEIDON2_WIDTH]>,
    >(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        cached_trace_ctx: CachedTraceCtx<CpuBackend<SC>>,
        proofs: &[Proof<BabyBearPoseidon2Config>],
        external_data: &mut VerifierExternalData,
        _device_ctx: &(),
        initial_transcript: TS,
    ) -> Option<Vec<AirProvingContext<CpuBackend<SC>>>> {
        debug_assert!(proofs.len() <= MAX_NUM_PROOFS);
        // Small recursive nodes retain the low-overhead one-thread-per-proof
        // schedule.  A large reduced-SWIRL source batch is processed in
        // bounded waves so 100+ children cannot create 100+ native stacks.
        let span = tracing::Span::current();
        let mut preflights = Vec::with_capacity(proofs.len());
        let preflight_width = if proofs.len() > LARGE_VERIFIER_BATCH_CUTOFF {
            LARGE_VERIFIER_PREFLIGHT_WIDTH
        } else {
            proofs.len().max(1)
        };
        for proof_chunk in proofs.chunks(preflight_width) {
            let mut chunk_preflights = std::thread::scope(|s| {
                let handles: Vec<_> = proof_chunk
                    .iter()
                    .map(|proof| {
                        let sponge = initial_transcript.clone();
                        let span = span.clone();
                        s.spawn(move || {
                            let _guard = span.enter();
                            #[cfg(feature = "cuda")]
                            {
                                self.run_preflight_without_merkle(sponge, child_vk, proof)
                            }
                            #[cfg(not(feature = "cuda"))]
                            {
                                self.run_preflight(sponge, child_vk, proof)
                            }
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().unwrap())
                    .collect::<Vec<_>>()
            });
            preflights.append(&mut chunk_preflights);
        }
        #[cfg(feature = "cuda")]
        if self.tail_mode == VerifierTailMode::Complete {
            for (proof, preflight) in proofs.iter().zip(preflights.iter_mut()) {
                Self::apply_merkle_precomputation_cpu(proof, preflight);
            }
        }

        if let Some(final_transcript_state) = &mut external_data.final_transcript_state {
            // WARNING: For convenience we use the fact that the last transcript operation should be
            // a sample. If this is not the case, we will have to reconstruct final_transcript_state
            // from the last perm state and observe values.
            debug_assert_eq!(preflights.len(), 1);
            debug_assert!(preflights[0].transcript.samples().last().unwrap());
            let state = *preflights[0].transcript.perm_results().last().unwrap();
            *(*final_transcript_state) = state;
            preflights[0].poseidon2_compress_inputs.push(state);
        }

        let power_checker_gen =
            Arc::new(PowerCheckerCpuTraceGenerator::<2, POW_CHECKER_HEIGHT>::default());
        for &log in external_data.power_check_inputs {
            power_checker_gen.add_pow(log);
        }
        let exp_bits_len_gen = ExpBitsLenCpuTraceGenerator::default();

        let (module_required, statement_required, power_checker_required, exp_bits_len_required) =
            self.split_required_heights(external_data.required_heights);

        // WARNING: SymbolicExpressionAir MUST be the first AIR in verifier circuit
        let mut modules = vec![
            TraceModuleRef::BatchConstraint(&self.batch_constraint),
            TraceModuleRef::Transcript(&self.transcript),
            TraceModuleRef::ProofShape(&self.proof_shape),
            TraceModuleRef::Gkr(&self.gkr),
        ];
        if self.tail_mode.includes_stacking() {
            modules.push(TraceModuleRef::Stacking(&self.stacking));
        }
        if self.tail_mode.includes_whir() {
            modules.push(TraceModuleRef::Whir(&self.whir));
        }
        let cached_trace_record = match &cached_trace_ctx {
            CachedTraceCtx::Records(cached_trace_record) => Some(cached_trace_record),
            CachedTraceCtx::PcsData(_) | CachedTraceCtx::SetupBound => None,
        };

        let span = tracing::Span::current();
        let ctxs_by_module = if proofs.len() > LARGE_VERIFIER_BATCH_CUTOFF {
            // Each module already parallelizes its rows/proofs.  Running the
            // modules sequentially keeps only one family of temporary
            // matrices live while preserving full inner Rayon utilization.
            modules
                .into_iter()
                .zip(module_required)
                .map(|(module, required_heights)| {
                    let _guard = span.enter();
                    let result = module.generate_cpu_ctxs(
                        child_vk,
                        proofs,
                        &preflights,
                        cached_trace_record,
                        &power_checker_gen,
                        &exp_bits_len_gen,
                        external_data,
                        required_heights,
                    );
                    if result.is_none() {
                        tracing::error!(air_module = %module, "verifier module trace generation failed");
                    }
                    result
                })
                .collect::<Vec<_>>()
        } else {
            modules
                .into_par_iter()
                .zip(module_required)
                .map(|(module, required_heights)| {
                    let _guard = span.enter();
                    let result = module.generate_cpu_ctxs(
                        child_vk,
                        proofs,
                        &preflights,
                        cached_trace_record,
                        &power_checker_gen,
                        &exp_bits_len_gen,
                        external_data,
                        required_heights,
                    );
                    if result.is_none() {
                        tracing::error!(air_module = %module, "verifier module trace generation failed");
                    }
                    result
                })
                .collect::<Vec<_>>()
        };

        let mut ctxs_by_module: Vec<Vec<AirProvingContext<CpuBackend<SC>>>> =
            ctxs_by_module.into_iter().collect::<Option<Vec<_>>>()?;
        match cached_trace_ctx {
            CachedTraceCtx::PcsData(child_vk_pcs_data) => {
                assert!(self.batch_constraint.has_cached);
                ctxs_by_module[BATCH_CONSTRAINT_MOD_IDX][LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX]
                    .cached_mains = vec![child_vk_pcs_data];
            }
            CachedTraceCtx::SetupBound => {
                assert!(self.batch_constraint.has_cached);
            }
            CachedTraceCtx::Records(cached_trace_record) => {
                assert!(!self.batch_constraint.has_cached);
                if self.batch_constraint.fixed_dag_commit().is_none() {
                    ctxs_by_module[BATCH_CONSTRAINT_MOD_IDX][LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX]
                        .public_values =
                        cached_trace_record.dag_commit_info.unwrap().commit.to_vec();
                }
            }
        };

        let mut ctx_per_trace = ctxs_by_module.into_iter().flatten().collect::<Vec<_>>();
        if let Some(checkpoint) = self.deferred_checkpoint {
            let required = statement_required.and_then(|heights| heights.first().copied());
            let (trace, public_values) = checkpoint.generate_trace(&preflights, required)?;
            ctx_per_trace.push(AirProvingContext::simple(trace, public_values));
        }
        if let Some(claims) = self.constraint_claims {
            let required = statement_required.and_then(|heights| heights.first().copied());
            let trace = claims.generate_trace(child_vk, proofs, &preflights, required)?;
            ctx_per_trace.push(AirProvingContext::simple(trace, Vec::new()));
        }
        if let Some(checkpoint) = self.constraint_checkpoint {
            let required = statement_required.and_then(|heights| heights.get(1).copied());
            let (trace, public_values) = checkpoint.generate_trace(&preflights, required)?;
            ctx_per_trace.push(AirProvingContext::simple(trace, public_values));
        }
        if power_checker_required.is_some_and(|h| h != POW_CHECKER_HEIGHT) {
            return None;
        }
        // Caution: this must be done after GKR and WHIR tracegen
        tracing::trace_span!("wrapper.generate_proving_ctxs", air_module = "Primitives",).in_scope(
            || {
                tracing::trace_span!("wrapper.generate_trace", air = "PowerChecker").in_scope(
                    || {
                        ctx_per_trace.push(AirProvingContext::simple_no_pis(
                            power_checker_gen.generate_trace_row_major(),
                        ));
                    },
                );
            },
        );
        let exp_bits_trace_rm = tracing::trace_span!("wrapper.generate_trace", air = "ExpBitsLen")
            .in_scope(|| exp_bits_len_gen.generate_trace_row_major(exp_bits_len_required))?;
        ctx_per_trace.push(AirProvingContext::simple_no_pis(exp_bits_trace_rm));
        Some(ctx_per_trace)
    }
}

#[cfg(feature = "cuda")]
pub mod cuda_tracegen {
    use std::iter::zip;

    use openvm_cuda_backend::{
        base::DeviceMatrix, device_memory_snapshot, hash_scheme::GpuHashScheme, GenericGpuBackend,
        GpuBackend,
    };
    use openvm_cuda_common::copy::MemCopyH2D;
    use openvm_stark_backend::prover::{MatrixDimensions, ProverDevice};

    use super::*;
    use crate::{
        cuda::{preflight::PreflightGpu, proof::ProofGpu, vk::VerifyingKeyGpu},
        primitives::{
            exp_bits_len::ExpBitsLenTraceGenerator as GpuExpBitsLenTraceGenerator,
            pow::cuda::PowerCheckerGpuTraceGenerator,
        },
    };

    impl<'a, const TRANSCRIPT_SBOX_REGISTERS: usize> TraceModuleRef<'a, TRANSCRIPT_SBOX_REGISTERS> {
        #[allow(clippy::too_many_arguments)]
        #[tracing::instrument(
            name = "wrapper.generate_proving_ctxs",
            level = "trace",
            skip_all,
            fields(air_module = %self)
        )]
        fn generate_gpu_proving_ctxs(
            self,
            child_vk: &VerifyingKeyGpu,
            proofs: &[ProofGpu],
            preflights: &[PreflightGpu],
            device_ctx: &openvm_cuda_common::stream::GpuDeviceCtx,
            cached_trace_record: Option<&CachedTraceRecord>,
            pow_checker_gen: &Arc<PowerCheckerGpuTraceGenerator<2, POW_CHECKER_HEIGHT>>,
            exp_bits_len_gen: &GpuExpBitsLenTraceGenerator,
            external_data: &VerifierExternalData,
            required_heights: Option<&[usize]>,
        ) -> Option<Vec<AirProvingContext<GpuBackend>>> {
            match self {
                TraceModuleRef::Transcript(module) => module.generate_proving_ctxs(
                    child_vk,
                    proofs,
                    preflights,
                    &(
                        external_data.poseidon2_permute_inputs,
                        external_data.poseidon2_compress_inputs,
                        device_ctx,
                    ),
                    required_heights,
                ),
                TraceModuleRef::ProofShape(module) => module.generate_proving_ctxs(
                    child_vk,
                    proofs,
                    preflights,
                    &(
                        pow_checker_gen.clone(),
                        external_data.range_check_inputs.as_slice(),
                        device_ctx,
                    ),
                    required_heights,
                ),
                TraceModuleRef::Gkr(module) => module.generate_proving_ctxs(
                    child_vk,
                    proofs,
                    preflights,
                    &(exp_bits_len_gen, device_ctx),
                    required_heights,
                ),
                TraceModuleRef::BatchConstraint(module) => module.generate_proving_ctxs(
                    child_vk,
                    proofs,
                    preflights,
                    &(
                        cached_trace_record,
                        pow_checker_gen.cpu_checker().unwrap(),
                        device_ctx,
                    ),
                    required_heights,
                ),
                TraceModuleRef::Stacking(module) => module.generate_proving_ctxs(
                    child_vk,
                    proofs,
                    preflights,
                    device_ctx,
                    required_heights,
                ),
                TraceModuleRef::Whir(module) => module.generate_proving_ctxs(
                    child_vk,
                    proofs,
                    preflights,
                    &(exp_bits_len_gen, device_ctx),
                    required_heights,
                ),
            }
        }
    }

    /// Coerces an `AirProvingContext<GpuBackend>` to `AirProvingContext<GenericGpuBackend<HS>>`.
    ///
    /// Safe because all GPU backends share `Val = BabyBear` and `Matrix = DeviceMatrix<F>`.
    /// Panics in debug builds if `cached_mains` is non-empty (commitments differ by hash scheme).
    fn coerce_gpu_proving_ctx<HS: GpuHashScheme>(
        ctx: AirProvingContext<GpuBackend>,
    ) -> AirProvingContext<GenericGpuBackend<HS>> {
        debug_assert!(ctx.cached_mains.is_empty());
        AirProvingContext {
            cached_mains: vec![],
            common_main: ctx.common_main,
            public_values: ctx.public_values,
        }
    }

    impl<
            HS: GpuHashScheme,
            const MAX_NUM_PROOFS: usize,
            const TRANSCRIPT_SBOX_REGISTERS: usize,
        >
        VerifierTraceGen<GenericGpuBackend<HS>, HS::SC, openvm_cuda_common::stream::GpuDeviceCtx>
        for VerifierSubCircuit<MAX_NUM_PROOFS, TRANSCRIPT_SBOX_REGISTERS>
    {
        fn new(
            child_mvk: Arc<MultiStarkVerifyingKey<BabyBearPoseidon2Config>>,
            config: VerifierConfig,
        ) -> Self {
            Self::new_with_options(child_mvk, config)
        }

        fn commit_child_vk<E: StarkEngine<SC = HS::SC, PB = GenericGpuBackend<HS>>>(
            &self,
            engine: &E,
            child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        ) -> CommittedTraceData<GenericGpuBackend<HS>>
        where
            GpuDeviceCtx: From<EngineDeviceCtx<E>>,
        {
            let device_ctx: GpuDeviceCtx = engine.device().device_ctx().clone().into();
            self.batch_constraint
                .commit_child_vk_gpu(engine, child_vk, &device_ctx)
        }

        fn cached_trace_record(
            &self,
            child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        ) -> CachedTraceRecord {
            self.batch_constraint.cached_trace_record(child_vk)
        }

        #[tracing::instrument(name = "subcircuit_generate_proving_ctxs", skip_all)]
        fn generate_proving_ctxs<
            TS: FiatShamirTranscript<BabyBearPoseidon2Config>
                + TranscriptHistory<F = F, State = [F; POSEIDON2_WIDTH]>,
        >(
            &self,
            child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
            cached_trace_ctx: CachedTraceCtx<GenericGpuBackend<HS>>,
            proofs: &[Proof<BabyBearPoseidon2Config>],
            external_data: &mut VerifierExternalData,
            device_ctx: &openvm_cuda_common::stream::GpuDeviceCtx,
            initial_transcript: TS,
        ) -> Option<Vec<AirProvingContext<GenericGpuBackend<HS>>>> {
            debug_assert!(proofs.len() <= MAX_NUM_PROOFS);
            let child_vk_gpu = VerifyingKeyGpu::new(child_vk, device_ctx);
            let proofs_gpu = proofs
                .iter()
                .map(|proof_cpu| ProofGpu::new(child_vk, proof_cpu, device_ctx))
                .collect::<Vec<_>>();
            // Match the CPU verifier's bounded large-batch schedule.  The
            // CUDA trace generators consume these preflights afterwards; an
            // unbounded native-thread fan-out only increases host RSS.
            let span = tracing::Span::current();
            let mut preflights_cpu = Vec::with_capacity(proofs.len());
            let preflight_width = if proofs.len() > LARGE_VERIFIER_BATCH_CUTOFF {
                LARGE_VERIFIER_PREFLIGHT_WIDTH
            } else {
                proofs.len().max(1)
            };
            for proof_chunk in proofs.chunks(preflight_width) {
                let mut chunk_preflights = std::thread::scope(|s| {
                    let handles: Vec<_> = proof_chunk
                        .iter()
                        .map(|proof| {
                            let sponge = initial_transcript.clone();
                            let span = span.clone();
                            s.spawn(move || {
                                let _guard = span.enter();
                                self.run_preflight_without_merkle(sponge, child_vk, proof)
                            })
                        })
                        .collect();
                    handles
                        .into_iter()
                        .map(|h| h.join().unwrap())
                        .collect::<Vec<_>>()
                });
                preflights_cpu.append(&mut chunk_preflights);
            }

            // Deferred openings contain no Merkle authentication rows.  In
            // the complete verifier this precomputation remains unchanged.
            if self.tail_mode == VerifierTailMode::Complete {
                for (proof, preflight) in proofs.iter().zip(preflights_cpu.iter_mut()) {
                    Self::apply_merkle_precomputation(proof, preflight, device_ctx);
                }
            }

            if let Some(final_transcript_state) = &mut external_data.final_transcript_state {
                // WARNING: For convenience we use the fact that the last transcript operation
                // should be a sample. If this is not the case, we will have to reconstruct
                // final_transcript_state from the last perm state and observe values.
                debug_assert_eq!(preflights_cpu.len(), 1);
                debug_assert!(preflights_cpu[0].transcript.samples().last().unwrap());
                let state = *preflights_cpu[0].transcript.perm_results().last().unwrap();
                *(*final_transcript_state) = state;
                preflights_cpu[0].poseidon2_compress_inputs.push(state);
            }

            let power_checker_gen = Arc::new(
                PowerCheckerGpuTraceGenerator::<2, POW_CHECKER_HEIGHT>::hybrid(device_ctx.clone()),
            );
            if let Some(cpu_checker) = power_checker_gen.cpu_checker() {
                for &log in external_data.power_check_inputs {
                    cpu_checker.add_pow(log);
                }
            }
            let exp_bits_len_gen = GpuExpBitsLenTraceGenerator::new(device_ctx.clone());

            let (
                module_required,
                statement_required,
                power_checker_required,
                exp_bits_len_required,
            ) = self.split_required_heights(external_data.required_heights);

            let mut statement_traces = Vec::new();
            if let Some(checkpoint) = self.deferred_checkpoint {
                let required = statement_required.and_then(|heights| heights.first().copied());
                statement_traces.push(checkpoint.generate_trace(&preflights_cpu, required)?);
            }
            if let Some(claims) = self.constraint_claims {
                let required = statement_required.and_then(|heights| heights.first().copied());
                statement_traces.push((
                    claims.generate_trace(child_vk, proofs, &preflights_cpu, required)?,
                    Vec::new(),
                ));
            }
            if let Some(checkpoint) = self.constraint_checkpoint {
                let required = statement_required.and_then(|heights| heights.get(1).copied());
                statement_traces.push(checkpoint.generate_trace(&preflights_cpu, required)?);
            }

            // NOTE: avoid par_iter for now so H2D transfer all happens on same stream to avoid sync
            // issues
            let preflights_gpu = zip(proofs, preflights_cpu)
                .map(|(proof, preflight_cpu)| {
                    PreflightGpu::new(child_vk, proof, &preflight_cpu, device_ctx)
                })
                .collect::<Vec<_>>();
            let mut modules = vec![
                TraceModuleRef::BatchConstraint(&self.batch_constraint),
                TraceModuleRef::Transcript(&self.transcript),
                TraceModuleRef::ProofShape(&self.proof_shape),
                TraceModuleRef::Gkr(&self.gkr),
            ];
            if self.tail_mode.includes_stacking() {
                modules.push(TraceModuleRef::Stacking(&self.stacking));
            }
            if self.tail_mode.includes_whir() {
                modules.push(TraceModuleRef::Whir(&self.whir));
            }
            let cached_trace_record = match &cached_trace_ctx {
                CachedTraceCtx::Records(cached_trace_record) => Some(cached_trace_record),
                CachedTraceCtx::PcsData(_) | CachedTraceCtx::SetupBound => None,
            };

            // Generate the transcript family first. Its padded Poseidon table is the largest
            // single allocation for wide (100+) child batches. Generating it after the batch-
            // constraint family needlessly put both families' construction peaks on top of one
            // another. Results are restored to canonical module/AIR order before key binding and
            // proving, so this scheduling optimization cannot change the relation or transcript.
            //
            // PERF[jpw]: we avoid par_iter so that kernel launches occur on the same stream.
            // This can be parallelized to separate streams for more CUDA stream parallelism, but it
            // will require recording events so streams properly sync for cudaMemcpyAsync and kernel
            // launches.
            let module_count = modules.len();
            let mut module_jobs = modules
                .into_iter()
                .zip(module_required)
                .enumerate()
                .collect::<Vec<_>>();
            module_jobs.sort_by_key(|(canonical_index, _)| {
                if *canonical_index == TRANSCRIPT_MOD_IDX {
                    0
                } else {
                    canonical_index + 1
                }
            });
            let mut ctxs_by_module_gpu = std::iter::repeat_with(|| None)
                .take(module_count)
                .collect::<Vec<_>>();
            for (canonical_index, (module, required_heights)) in module_jobs {
                let module_name = module.to_string();
                let module_ctxs = module.generate_gpu_proving_ctxs(
                    &child_vk_gpu,
                    &proofs_gpu,
                    &preflights_gpu,
                    device_ctx,
                    cached_trace_record,
                    &power_checker_gen,
                    &exp_bits_len_gen,
                    external_data,
                    required_heights,
                )?;
                if proofs.len() > LARGE_VERIFIER_BATCH_CUTOFF {
                    let retained_trace_bytes = module_ctxs
                        .iter()
                        .map(|ctx| {
                            let common = ctx
                                .common_main
                                .height()
                                .saturating_mul(ctx.common_main.width());
                            let cached = ctx.cached_mains.iter().fold(0usize, |sum, matrix| {
                                sum.saturating_add(
                                    matrix.trace.height().saturating_mul(matrix.trace.width()),
                                )
                            });
                            common
                                .saturating_add(cached)
                                .saturating_mul(core::mem::size_of::<F>())
                        })
                        .sum::<usize>();
                    if let Ok(snapshot) = device_memory_snapshot() {
                        tracing::info!(
                            module = %module_name,
                            canonical_index,
                            retained_trace_mib = retained_trace_bytes / (1 << 20),
                            live_mib = snapshot.live_bytes / (1 << 20),
                            driver_free_mib = snapshot.driver_free_bytes / (1 << 20),
                            "generated large verifier CUDA module"
                        );
                    }
                }
                ctxs_by_module_gpu[canonical_index] = Some(module_ctxs);
            }
            device_ctx.stream.synchronize().unwrap();
            let mut ctxs_by_module: Vec<Vec<AirProvingContext<GenericGpuBackend<HS>>>> =
                ctxs_by_module_gpu
                    .into_iter()
                    .collect::<Option<Vec<_>>>()?
                    .into_iter()
                    .map(|module_ctxs| {
                        module_ctxs
                            .into_iter()
                            .map(coerce_gpu_proving_ctx::<HS>)
                            .collect()
                    })
                    .collect();
            match cached_trace_ctx {
                CachedTraceCtx::PcsData(child_vk_pcs_data) => {
                    assert!(self.batch_constraint.has_cached);
                    ctxs_by_module[BATCH_CONSTRAINT_MOD_IDX][LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX]
                        .cached_mains = vec![child_vk_pcs_data];
                }
                CachedTraceCtx::SetupBound => {
                    assert!(self.batch_constraint.has_cached);
                }
                CachedTraceCtx::Records(cached_trace_record) => {
                    assert!(!self.batch_constraint.has_cached);
                    if self.batch_constraint.fixed_dag_commit().is_none() {
                        ctxs_by_module[BATCH_CONSTRAINT_MOD_IDX]
                            [LOCAL_SYMBOLIC_EXPRESSION_AIR_IDX]
                            .public_values =
                            cached_trace_record.dag_commit_info.unwrap().commit.to_vec();
                    }
                }
            }

            let mut ctx_per_trace = ctxs_by_module.into_iter().flatten().collect::<Vec<_>>();
            for (trace, public_values) in statement_traces {
                let height = trace.height();
                let width = trace.width();
                let buffer = trace.values.to_device_on(device_ctx).ok()?;
                let trace = DeviceMatrix::new(Arc::new(buffer), height, width);
                ctx_per_trace.push(AirProvingContext::simple(trace, public_values));
            }
            if power_checker_required.is_some_and(|h| h != POW_CHECKER_HEIGHT) {
                return None;
            }
            // Caution: this must be done after GKR and WHIR tracegen
            tracing::trace_span!("wrapper.generate_proving_ctxs", air_module = "Primitives",)
                .in_scope(|| {
                    tracing::trace_span!("wrapper.generate_trace", air = "PowerChecker").in_scope(
                        || {
                            let pow_bits_trace = power_checker_gen.generate_trace();
                            ctx_per_trace.push(AirProvingContext::simple_no_pis(pow_bits_trace));
                        },
                    );
                });
            let exp_bits_trace = tracing::trace_span!("wrapper.generate_trace", air = "ExpBitsLen")
                .in_scope(|| exp_bits_len_gen.generate_trace_device(exp_bits_len_required))?;
            ctx_per_trace.push(AirProvingContext::simple_no_pis(exp_bits_trace));
            Some(ctx_per_trace)
        }
    }
}

//! End-to-end CPU reference for native segment accumulation at SWIRL's
//! deferred-opening boundary.
//!
//! The VM produces ordinary AIR traces. Each callback executes AIR/LogUp and
//! stacking once, retains the verifier prefix for the final recursive wrapper,
//! and moves the exact committed PCS owner into constrained-code WARP. No
//! complete per-segment WHIR proof or verifier-PESAT witness is constructed.

use std::cell::RefCell;

use openvm_circuit::{
    arch::{
        verify_segment_metadata_sequence, Executor, MeteredExecutor, PreflightExecutor, VmBuilder,
        VmExecutionConfig, VmSegmentMetadata,
    },
    system::{memory::merkle::public_values::UserPublicValuesProof, SystemWithFixedTraceHeights},
};
use openvm_cpu_backend::{CpuBackend, CpuDevice, CpuReducedSwirlSource};
use openvm_recursion_circuit::system::RetainedStackingProof;
use openvm_stark_backend::{
    p3_field::{ExtensionField, TwoAdicField},
    StarkEngine,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, EF};

use super::{
    reduced_swirl_boundary::{
        AuthoritativeSwirlConstrainedRsClaim, PendingConstrainedCodePublicClaim, ReducedSwirlPrefix,
    },
    reduced_swirl_native::{
        reduced_swirl_source_challenger, ReducedSwirlNativeCpuStream,
        ReducedSwirlNativeProverOutput, ReducedSwirlNativeSetup, REDUCED_SWIRL_MAX_SOURCES,
    },
};
use crate::{prover::AppProver, StdIn, F, SC};

/// Complete native result before the one final recursive wrapper is proved.
/// Large PCS owners have already been consumed and released; retained prefixes
/// and claims are the bounded transition-tree inputs.
pub struct ReducedSwirlCpuExecution {
    pub setup: ReducedSwirlNativeSetup,
    pub native: ReducedSwirlNativeProverOutput,
    pub retained_prefixes: Vec<RetainedStackingProof>,
    pub authoritative_wrapper_claims: Vec<AuthoritativeSwirlConstrainedRsClaim>,
    pub segment_metadata: Vec<VmSegmentMetadata<SC>>,
    pub user_public_values: UserPublicValuesProof<DIGEST_SIZE, F>,
}

#[derive(Debug, thiserror::Error)]
pub enum ReducedSwirlCpuExecutionError {
    #[error("native reduced-SWIRL setup failed: {0}")]
    Setup(String),
    #[error("native reduced-SWIRL VM stream failed: {0}")]
    Vm(String),
    #[error("native reduced-SWIRL segment {segment} failed: {message}")]
    Segment { segment: usize, message: String },
    #[error("native reduced-SWIRL segment continuity failed: {0}")]
    Continuity(String),
    #[error("native reduced-SWIRL stream returned no segment")]
    EmptyExecution,
}

/// Prove all VM segments through one bounded-memory constrained-code WARP
/// stream. The ordered block manifest is derived from the exact source-entry
/// digests; no caller-selected manifest is accepted.
pub fn prove_reduced_swirl_cpu_execution<E, VB>(
    app_prover: &mut AppProver<E, VB>,
    input: StdIn<F>,
    input_arity: usize,
    family_target_bits: usize,
) -> Result<ReducedSwirlCpuExecution, ReducedSwirlCpuExecutionError>
where
    E: StarkEngine<SC = SC, PB = CpuBackend<SC>, PD = CpuDevice<SC>>,
    VB: VmBuilder<E>,
    VB::SystemChipInventory: SystemWithFixedTraceHeights,
    <VB::VmConfig as VmExecutionConfig<F>>::Executor:
        Executor<F> + MeteredExecutor<F> + PreflightExecutor<F, VB::RecordArena>,
    F: TwoAdicField + Ord,
    EF: ExtensionField<F> + TwoAdicField + Ord,
{
    let plan = app_prover
        .plan_warp_stream(input.clone())
        .map_err(|error| ReducedSwirlCpuExecutionError::Vm(error.to_string()))?;
    let source_count = plan.segment_count();
    if source_count == 0 {
        return Err(ReducedSwirlCpuExecutionError::EmptyExecution);
    }
    let app_vk = app_prover.app_vm_vk().clone();
    let config = E::new(app_vk.inner.params.clone()).config().clone();
    let setup = ReducedSwirlNativeSetup::new(
        &config,
        &app_vk.inner.params,
        input_arity,
        family_target_bits,
        REDUCED_SWIRL_MAX_SOURCES,
    )
    .map_err(|error| ReducedSwirlCpuExecutionError::Setup(error.to_string()))?;

    let stream = RefCell::new(Some(
        ReducedSwirlNativeCpuStream::new(&setup, source_count)
            .map_err(|error| ReducedSwirlCpuExecutionError::Setup(error.to_string()))?,
    ));
    let retained_prefixes = RefCell::new(Vec::with_capacity(source_count));
    let wrapper_claims = RefCell::new(Vec::with_capacity(source_count));
    let segment_metadata = RefCell::new(Vec::with_capacity(source_count));
    let app_vk_for_callback = app_vk.clone();

    let user_public_values = app_prover
        .prove_warp_stream_prepared(input, plan, |segment_index, engine, pk, context| {
            let mut prefix_prover = engine.prover();
            let reduction = prefix_prover
                .prove_native_stacking_reduction(pk, context)
                .map_err(|error| ReducedSwirlCpuExecutionError::Segment {
                    segment: segment_index,
                    message: error.to_string(),
                })?;
            let prefix = ReducedSwirlPrefix::from_native_reduction(
                &app_vk_for_callback,
                segment_index,
                reduction,
            )
            .map_err(|error| ReducedSwirlCpuExecutionError::Segment {
                segment: segment_index,
                message: error.to_string(),
            })?;
            let metadata = openvm_circuit::arch::vm_segment_metadata_from_parts(
                &app_vk_for_callback,
                &prefix.retained().trace_vdata,
                &prefix.retained().public_values,
            )
            .map_err(|error| ReducedSwirlCpuExecutionError::Segment {
                segment: segment_index,
                message: error.to_string(),
            })?;
            let pending_public = PendingConstrainedCodePublicClaim {
                metadata: prefix.pending_witness().metadata().clone(),
                swirl_tilde_u: prefix.pending_witness().terminal_point().clone(),
                stacking_openings: prefix.retained().stacking_proof.stacking_openings.clone(),
            };
            let manifest_prefix = prefix.manifest_prefix(segment_index).map_err(|error| {
                ReducedSwirlCpuExecutionError::Segment {
                    segment: segment_index,
                    message: error.to_string(),
                }
            })?;
            let (retained, pending_witness) = prefix.into_parts();
            let mut source_challenger = reduced_swirl_source_challenger();
            let source = CpuReducedSwirlSource::try_from_pending(
                pending_witness,
                &retained.stacking_proof.stacking_openings,
                &mut source_challenger,
            )
            .map_err(|error| ReducedSwirlCpuExecutionError::Segment {
                segment: segment_index,
                message: error.to_string(),
            })?;
            let wrapper_claim = pending_public
                .authoritative_claim_from_backend(source.claim_ref())
                .map_err(|error| ReducedSwirlCpuExecutionError::Segment {
                    segment: segment_index,
                    message: error.to_string(),
                })?;
            let source_binding =
                manifest_prefix
                    .digest_with_claim(&wrapper_claim)
                    .map_err(|error| ReducedSwirlCpuExecutionError::Segment {
                        segment: segment_index,
                        message: error.to_string(),
                    })?;
            stream
                .borrow_mut()
                .as_mut()
                .ok_or(ReducedSwirlCpuExecutionError::EmptyExecution)?
                .push_source(source_binding, source)
                .map_err(|error| ReducedSwirlCpuExecutionError::Segment {
                    segment: segment_index,
                    message: error.to_string(),
                })?;
            retained_prefixes.borrow_mut().push(retained);
            wrapper_claims.borrow_mut().push(wrapper_claim);
            segment_metadata.borrow_mut().push(metadata);
            Ok(())
        })
        .map_err(|error| match error {
            openvm_circuit::arch::NativeWarpStreamError::Vm(error) => {
                ReducedSwirlCpuExecutionError::Vm(error.to_string())
            }
            openvm_circuit::arch::NativeWarpStreamError::Segment(error) => error,
        })?;

    let native = stream
        .borrow_mut()
        .take()
        .ok_or(ReducedSwirlCpuExecutionError::EmptyExecution)?
        .finish()
        .map_err(|error| ReducedSwirlCpuExecutionError::Setup(error.to_string()))?;
    let retained_prefixes = retained_prefixes.into_inner();
    let authoritative_wrapper_claims = wrapper_claims.into_inner();
    let segment_metadata = segment_metadata.into_inner();
    if retained_prefixes.len() != source_count
        || authoritative_wrapper_claims.len() != source_count
        || segment_metadata.len() != source_count
    {
        return Err(ReducedSwirlCpuExecutionError::EmptyExecution);
    }
    verify_segment_metadata_sequence(&segment_metadata)
        .map_err(|error| ReducedSwirlCpuExecutionError::Continuity(error.to_string()))?;

    Ok(ReducedSwirlCpuExecution {
        setup,
        native,
        retained_prefixes,
        authoritative_wrapper_claims,
        segment_metadata,
        user_public_values,
    })
}

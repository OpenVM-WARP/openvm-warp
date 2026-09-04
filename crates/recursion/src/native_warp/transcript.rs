use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, transcript::TranscriptLog, AirRef, BaseAirWithPublicValues,
    PartitionedBaseAir, StarkProtocolConfig, SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::F;
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::{NativeVaccPhaseCursorBus, NativeVaccPhaseCursorMessage, NativeWarpPcdBusInventory};
use crate::{
    bus::{TranscriptBus, TranscriptBusMessage},
    system::{AirModule, BusInventory},
    transcript::{Poseidon2BusOwner, Poseidon2MultibusInputs, TranscriptModule},
};

/// Native WARP transcript verifier sharing the recursion circuit's Poseidon
/// table. It intentionally contributes only `TranscriptAir`; callers append
/// the returned permutation inputs to the parent transcript module.
pub struct NativeWarpTranscriptModule {
    inner: TranscriptModule,
}

pub struct NativeWarpTranscriptArtifacts {
    pub trace: RowMajorMatrix<F>,
    pub poseidon2_trace: RowMajorMatrix<F>,
    pub permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
}

pub struct NativeWarpTranscriptInputArtifacts {
    pub trace: RowMajorMatrix<F>,
    pub permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeTranscriptRemainderCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub tidx: T,
    pub value: T,
    pub is_sample: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeTranscriptPrefixEqualityCols<T> {
    pub active: T,
    pub tidx: T,
    pub value: T,
    pub is_sample: T,
}

/// Links the transcript prefix independently verified by the host to the
/// identical prefix used by the VACC transcript.
///
/// Proof index zero is the complete transcript consumed by the VACC AIRs.
/// Proof index one contains only the externally verified prefix. Receiving
/// both messages with the same row values prevents the history certificate
/// from choosing an unrelated Fiat-Shamir starting state.
#[derive(ColumnsAir)]
#[columns_via(NativeTranscriptPrefixEqualityCols<u8>)]
pub struct NativeTranscriptPrefixEqualityAir {
    pub transcript_bus: TranscriptBus,
}

impl BaseAirWithPublicValues<F> for NativeTranscriptPrefixEqualityAir {}
impl PartitionedBaseAir<F> for NativeTranscriptPrefixEqualityAir {}
impl BaseAir<F> for NativeTranscriptPrefixEqualityAir {
    fn width(&self) -> usize {
        NativeTranscriptPrefixEqualityCols::<F>::width()
    }
}

/// Links an already-certified transcript prefix to a second transcript bus.
///
/// This is used when a protocol suffix is verified by a dedicated transcript
/// module. Both transcript AIRs start from the canonical zero sponge state;
/// consuming the same `(tidx, value, is_sample)` rows from both buses makes
/// the suffix transcript an exact continuation of the certified prefix.
#[derive(ColumnsAir)]
#[columns_via(NativeTranscriptPrefixEqualityCols<u8>)]
pub struct NativeTranscriptCrossEqualityAir {
    pub left_bus: TranscriptBus,
    pub right_bus: TranscriptBus,
    pub left_proof_idx: usize,
    pub right_proof_idx: usize,
}

impl BaseAirWithPublicValues<F> for NativeTranscriptCrossEqualityAir {}
impl PartitionedBaseAir<F> for NativeTranscriptCrossEqualityAir {}
impl BaseAir<F> for NativeTranscriptCrossEqualityAir {
    fn width(&self) -> usize {
        NativeTranscriptPrefixEqualityCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTranscriptCrossEqualityAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("native transcript cross-equality row");
        let local: &NativeTranscriptPrefixEqualityCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.assert_bool(local.is_sample);
        let message = TranscriptBusMessage {
            tidx: local.tidx.into(),
            value: local.value.into(),
            is_sample: local.is_sample.into(),
        };
        self.left_bus.receive(
            builder,
            AB::Expr::from_usize(self.left_proof_idx),
            message.clone(),
            local.active,
        );
        self.right_bus.receive(
            builder,
            AB::Expr::from_usize(self.right_proof_idx),
            message,
            local.active,
        );
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTranscriptPrefixEqualityAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("native transcript prefix-equality row");
        let local: &NativeTranscriptPrefixEqualityCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.assert_bool(local.is_sample);
        let message = TranscriptBusMessage {
            tidx: local.tidx.into(),
            value: local.value.into(),
            is_sample: local.is_sample.into(),
        };
        self.transcript_bus
            .receive(builder, AB::Expr::ZERO, message.clone(), local.active);
        self.transcript_bus
            .receive(builder, AB::Expr::ONE, message, local.active);
    }
}

/// Consumes transcript operations whose protocol semantics are certified by
/// another proof component.
///
/// Production callers must also bind the transcript's final sponge state to
/// that component. This table is not an authority for observed values; it
/// only closes the transcript lookup multiset after the external binding has
/// been established.
#[derive(ColumnsAir)]
#[columns_via(NativeTranscriptRemainderCols<u8>)]
pub struct NativeTranscriptRemainderAir {
    pub transcript_bus: TranscriptBus,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeTranscriptPrefixRemainderCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub tidx: T,
    pub value: T,
    pub is_sample: T,
    pub is_first: T,
    pub is_last: T,
}

/// Consumes exactly the certified transcript prefix `[0, start_tidx)`.
/// Unlike `NativeTranscriptRemainderAir`, this AIR cannot be used to close
/// witness-selected holes in the VACC suffix.
#[derive(ColumnsAir)]
#[columns_via(NativeTranscriptPrefixRemainderCols<u8>)]
pub struct NativeTranscriptPrefixRemainderAir {
    pub transcript_bus: TranscriptBus,
    pub phase_cursor_bus: NativeVaccPhaseCursorBus,
}

impl BaseAirWithPublicValues<F> for NativeTranscriptPrefixRemainderAir {}
impl PartitionedBaseAir<F> for NativeTranscriptPrefixRemainderAir {}
impl BaseAir<F> for NativeTranscriptPrefixRemainderAir {
    fn width(&self) -> usize {
        NativeTranscriptPrefixRemainderCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTranscriptPrefixRemainderAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main
            .row_slice(0)
            .expect("native transcript prefix remainder row");
        let next_row = main
            .row_slice(1)
            .expect("native transcript prefix remainder next row");
        let local: &NativeTranscriptPrefixRemainderCols<AB::Var> = (*local_row).borrow();
        let next: &NativeTranscriptPrefixRemainderCols<AB::Var> = (*next_row).borrow();
        for flag in [local.active, local.is_sample, local.is_first, local.is_last] {
            builder.assert_bool(flag);
        }
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_first_row()
            .assert_eq(local.active, local.is_first);
        builder
            .when(local.active * local.is_first)
            .assert_zero(local.tidx);
        let same_proof = next.active * (AB::Expr::ONE - next.is_first);
        let mut transition = builder.when_transition();
        let mut same = transition.when(same_proof);
        same.assert_zero(local.is_last);
        same.assert_eq(next.proof_idx, local.proof_idx);
        same.assert_eq(next.tidx, local.tidx + AB::F::ONE);
        let mut next_proof = transition.when(next.active * next.is_first);
        next_proof.assert_one(local.is_last);
        next_proof.assert_eq(next.proof_idx, local.proof_idx + AB::F::ONE);
        next_proof.assert_zero(next.tidx);
        builder
            .when_transition()
            .when(local.active - next.active)
            .assert_one(local.is_last);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);
        self.transcript_bus.receive(
            builder,
            local.proof_idx,
            TranscriptBusMessage {
                tidx: local.tidx.into(),
                value: local.value.into(),
                is_sample: local.is_sample.into(),
            },
            local.active,
        );
        self.phase_cursor_bus.receive(
            builder,
            NativeVaccPhaseCursorMessage {
                proof_idx: local.proof_idx.into(),
                boundary: AB::Expr::ZERO,
                tidx: local.tidx + AB::F::ONE,
            },
            local.active * local.is_last,
        );
    }
}

pub fn generate_native_transcript_prefix_remainder_trace(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    proof_idx: usize,
    start_tidx: usize,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if start_tidx > log.len() {
        return None;
    }
    let height = required_height.unwrap_or_else(|| start_tidx.max(1).next_power_of_two());
    if height < start_tidx {
        return None;
    }
    let width = NativeTranscriptPrefixRemainderCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    for tidx in 0..start_tidx {
        let cols: &mut NativeTranscriptPrefixRemainderCols<F> =
            values[tidx * width..(tidx + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.proof_idx = F::from_usize(proof_idx);
        cols.tidx = F::from_usize(tidx);
        cols.value = log.values()[tidx];
        cols.is_sample = F::from_bool(log.samples()[tidx]);
        cols.is_first = F::from_bool(tidx == 0);
        cols.is_last = F::from_bool(tidx + 1 == start_tidx);
    }
    Some(RowMajorMatrix::new(values, width))
}

impl BaseAirWithPublicValues<F> for NativeTranscriptRemainderAir {}
impl PartitionedBaseAir<F> for NativeTranscriptRemainderAir {}
impl BaseAir<F> for NativeTranscriptRemainderAir {
    fn width(&self) -> usize {
        NativeTranscriptRemainderCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeTranscriptRemainderAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("native transcript remainder row");
        let local: &NativeTranscriptRemainderCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.assert_bool(local.is_sample);
        self.transcript_bus.receive(
            builder,
            local.proof_idx,
            TranscriptBusMessage {
                tidx: local.tidx.into(),
                value: local.value.into(),
                is_sample: local.is_sample.into(),
            },
            local.active,
        );
    }
}

/// Build a remainder trace for operations not consumed by protocol-specific
/// transcript AIRs. `claimed[tidx] == true` means another AIR receives that
/// transcript lookup.
pub fn generate_native_transcript_remainder_trace(
    log: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    proof_idx: usize,
    claimed: &[bool],
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if claimed.len() != log.len() {
        return None;
    }
    let remainder_len = claimed.iter().filter(|&&is_claimed| !is_claimed).count();
    let height = required_height.unwrap_or_else(|| remainder_len.max(1).next_power_of_two());
    if height < remainder_len {
        return None;
    }
    let width = NativeTranscriptRemainderCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    let mut output_row = 0;
    for (tidx, (&value, (&is_sample, &is_claimed))) in log
        .values()
        .iter()
        .zip(log.samples().iter().zip(claimed))
        .enumerate()
    {
        if is_claimed {
            continue;
        }
        let row = &mut values[output_row * width..(output_row + 1) * width];
        let cols: &mut NativeTranscriptRemainderCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.proof_idx = F::from_usize(proof_idx);
        cols.tidx = F::from_usize(tidx);
        cols.value = value;
        cols.is_sample = F::from_bool(is_sample);
        output_row += 1;
    }
    Some(RowMajorMatrix::new(values, width))
}

/// Build the equality trace linking proof-index zero's prefix to the
/// independently finalized proof-index one prefix.
pub fn generate_native_transcript_prefix_equality_trace(
    prefix: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    generate_native_transcript_prefix_equality_trace_masked(
        prefix,
        &vec![false; prefix.len()],
        required_height,
    )
}

/// Build the equality trace while allowing protocol AIRs to consume selected
/// prefix operations directly from both transcript copies.
///
/// `claimed[tidx] == true` means a semantic AIR receives the operation from
/// proof indices zero and one, so emitting another equality row would
/// duplicate its lookup multiplicity.
pub fn generate_native_transcript_prefix_equality_trace_masked(
    prefix: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    claimed: &[bool],
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if claimed.len() != prefix.len() {
        return None;
    }
    let unclaimed = claimed.iter().filter(|&&value| !value).count();
    let height = required_height.unwrap_or_else(|| unclaimed.max(1).next_power_of_two());
    if height < unclaimed {
        return None;
    }
    let width = NativeTranscriptPrefixEqualityCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    let mut output_row = 0usize;
    for (tidx, ((&value, &is_sample), &is_claimed)) in prefix
        .values()
        .iter()
        .zip(prefix.samples())
        .zip(claimed)
        .enumerate()
    {
        if is_claimed {
            continue;
        }
        let row = &mut values[output_row * width..(output_row + 1) * width];
        let cols: &mut NativeTranscriptPrefixEqualityCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.tidx = F::from_usize(tidx);
        cols.value = value;
        cols.is_sample = F::from_bool(is_sample);
        output_row += 1;
    }
    Some(RowMajorMatrix::new(values, width))
}

impl NativeWarpTranscriptModule {
    /// Logical Poseidon bus owner for parent assemblies that combine this
    /// transcript with other recursion transcript modules in one table.
    #[must_use]
    pub fn poseidon2_bus_owner(&self) -> Poseidon2BusOwner {
        self.inner.poseidon2_bus_owner()
    }

    #[must_use]
    pub fn new(
        shared: &BusInventory,
        native: &NativeWarpPcdBusInventory,
        params: SystemParams,
    ) -> Self {
        Self::new_with_final_state(shared, native, params, false, false)
    }

    #[must_use]
    pub fn new_with_final_state(
        shared: &BusInventory,
        native: &NativeWarpPcdBusInventory,
        params: SystemParams,
        final_state_bus_enabled: bool,
        resume_state_bus_enabled: bool,
    ) -> Self {
        Self::new_for_bus_with_final_state_and_checkpoints(
            shared,
            native.transcript,
            params,
            final_state_bus_enabled,
            resume_state_bus_enabled,
            None,
        )
    }

    /// Build the history transcript with two authenticated intermediate
    /// checkpoints (pre-VACC and post-VACC).  Other recursive transcripts keep
    /// their existing width and bus inventory.
    #[must_use]
    pub fn new_with_certified_checkpoints(
        shared: &BusInventory,
        native: &NativeWarpPcdBusInventory,
        params: SystemParams,
        final_state_bus_enabled: bool,
        resume_state_bus_enabled: bool,
    ) -> Self {
        Self::new_for_bus_with_final_state_and_checkpoints(
            shared,
            native.transcript,
            params,
            final_state_bus_enabled,
            resume_state_bus_enabled,
            Some(native.transcript_checkpoint),
        )
    }

    /// Build a transcript module for a caller-owned transcript bus while
    /// retaining the parent's shared Poseidon buses.
    ///
    /// Native terminal-WHIR verification has its own transcript namespace but
    /// contributes permutations to the same Poseidon table as the VACC
    /// history. Accepting the bus directly avoids allocating an otherwise
    /// unused native-PCD bus inventory.
    #[must_use]
    pub fn new_for_bus(
        shared: &BusInventory,
        transcript_bus: TranscriptBus,
        params: SystemParams,
    ) -> Self {
        Self::new_for_bus_with_final_state(shared, transcript_bus, params, false, false)
    }

    #[must_use]
    pub fn new_for_bus_with_final_state(
        shared: &BusInventory,
        transcript_bus: TranscriptBus,
        params: SystemParams,
        final_state_bus_enabled: bool,
        resume_state_bus_enabled: bool,
    ) -> Self {
        Self::new_for_bus_with_final_state_and_checkpoints(
            shared,
            transcript_bus,
            params,
            final_state_bus_enabled,
            resume_state_bus_enabled,
            None,
        )
    }

    /// Build a resumed transcript whose exact terminal operation index is
    /// exported on the shared `TranscriptEndIndexBus`.
    ///
    /// The reduced-SWIRL transition-tree finalizer proves only the footer and
    /// terminal-Decision suffix.  Its terminal bridge must still learn that
    /// this suffix reaches the verifier-recorded end cursor; resumption alone
    /// authenticates the start state but does not provide that length check.
    #[must_use]
    pub fn new_for_bus_resumed_with_end_index(
        shared: &BusInventory,
        transcript_bus: TranscriptBus,
        params: SystemParams,
    ) -> Self {
        let mut module =
            Self::new_for_bus_with_final_state(shared, transcript_bus, params, false, true);
        module.inner.enable_end_index_bus();
        module
    }

    fn new_for_bus_with_final_state_and_checkpoints(
        shared: &BusInventory,
        transcript_bus: TranscriptBus,
        params: SystemParams,
        final_state_bus_enabled: bool,
        resume_state_bus_enabled: bool,
        checkpoint_state_bus: Option<crate::bus::CertifiedTranscriptCheckpointBus>,
    ) -> Self {
        let mut inventory = shared.clone();
        inventory.transcript_bus = transcript_bus;
        let inner = TranscriptModule::new_with_checkpoint_bus(
            inventory,
            params,
            final_state_bus_enabled,
            resume_state_bus_enabled,
            checkpoint_state_bus,
        );
        Self { inner }
    }

    #[must_use]
    pub fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        self.inner.airs::<SC>().into_iter().take(2).collect()
    }

    #[must_use]
    pub fn multi_bus_poseidon_air<SC: StarkProtocolConfig<F = F>>(owners: &[&Self]) -> AirRef<SC> {
        let first = owners
            .first()
            .expect("multi-bus Poseidon AIR needs an owner");
        let inner_owners = owners.iter().map(|owner| &owner.inner).collect::<Vec<_>>();
        first.inner.multi_bus_poseidon2_air(&inner_owners)
    }

    pub fn generate_trace(
        &self,
        logs: &[&TranscriptLog<F, [F; POSEIDON2_WIDTH]>],
        required_height: Option<usize>,
    ) -> Option<NativeWarpTranscriptArtifacts> {
        self.generate_trace_with_external(logs, Vec::new(), Vec::new(), required_height, None)
    }

    pub fn generate_trace_with_external(
        &self,
        logs: &[&TranscriptLog<F, [F; POSEIDON2_WIDTH]>],
        external_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        external_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_transcript_height: Option<usize>,
        required_poseidon_height: Option<usize>,
    ) -> Option<NativeWarpTranscriptArtifacts> {
        self.generate_trace_with_external_resumed(
            logs,
            &[],
            external_permutation_inputs,
            external_compression_inputs,
            required_transcript_height,
            required_poseidon_height,
        )
    }

    /// As [`Self::generate_trace_with_external`], but each log may continue an
    /// earlier transcript rather than start at the zero sponge.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_trace_with_external_resumed(
        &self,
        logs: &[&TranscriptLog<F, [F; POSEIDON2_WIDTH]>],
        resumes: &[Option<(usize, [F; POSEIDON2_WIDTH])>],
        external_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        external_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_transcript_height: Option<usize>,
        required_poseidon_height: Option<usize>,
    ) -> Option<NativeWarpTranscriptArtifacts> {
        self.generate_trace_with_external_resumed_and_checkpoints(
            logs,
            resumes,
            &[],
            external_permutation_inputs,
            external_compression_inputs,
            required_transcript_height,
            required_poseidon_height,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate_trace_with_external_resumed_and_checkpoints(
        &self,
        logs: &[&TranscriptLog<F, [F; POSEIDON2_WIDTH]>],
        resumes: &[Option<(usize, [F; POSEIDON2_WIDTH])>],
        checkpoints: &[Option<[usize; 2]>],
        external_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        external_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_transcript_height: Option<usize>,
        required_poseidon_height: Option<usize>,
    ) -> Option<NativeWarpTranscriptArtifacts> {
        let artifacts = self.generate_trace_inputs_with_external_resumed_and_checkpoints(
            logs,
            resumes,
            checkpoints,
            external_permutation_inputs,
            external_compression_inputs,
            required_transcript_height,
        )?;
        let poseidon2_trace = self.build_poseidon2_trace(
            artifacts.permutation_inputs.clone(),
            artifacts.compression_inputs.clone(),
            required_poseidon_height,
        )?;
        Some(NativeWarpTranscriptArtifacts {
            trace: artifacts.trace,
            poseidon2_trace,
            permutation_inputs: artifacts.permutation_inputs,
            compression_inputs: artifacts.compression_inputs,
        })
    }

    pub fn generate_trace_inputs_with_external(
        &self,
        logs: &[&TranscriptLog<F, [F; POSEIDON2_WIDTH]>],
        external_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        external_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_transcript_height: Option<usize>,
    ) -> Option<NativeWarpTranscriptInputArtifacts> {
        self.generate_trace_inputs_with_external_resumed(
            logs,
            &[],
            external_permutation_inputs,
            external_compression_inputs,
            required_transcript_height,
        )
    }

    /// As [`Self::generate_trace_inputs_with_external`], but each log may
    /// continue an earlier transcript rather than start at the zero sponge.
    pub fn generate_trace_inputs_with_external_resumed(
        &self,
        logs: &[&TranscriptLog<F, [F; POSEIDON2_WIDTH]>],
        resumes: &[Option<(usize, [F; POSEIDON2_WIDTH])>],
        external_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        external_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_transcript_height: Option<usize>,
    ) -> Option<NativeWarpTranscriptInputArtifacts> {
        self.generate_trace_inputs_with_external_resumed_and_checkpoints(
            logs,
            resumes,
            &[],
            external_permutation_inputs,
            external_compression_inputs,
            required_transcript_height,
        )
    }

    pub fn generate_trace_inputs_with_external_resumed_and_checkpoints(
        &self,
        logs: &[&TranscriptLog<F, [F; POSEIDON2_WIDTH]>],
        resumes: &[Option<(usize, [F; POSEIDON2_WIDTH])>],
        checkpoints: &[Option<[usize; 2]>],
        external_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        external_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_transcript_height: Option<usize>,
    ) -> Option<NativeWarpTranscriptInputArtifacts> {
        let artifacts = self.inner.build_transcript_trace_artifacts(
            logs,
            resumes,
            checkpoints,
            external_permutation_inputs,
            external_compression_inputs,
            required_transcript_height,
        )?;
        Some(NativeWarpTranscriptInputArtifacts {
            trace: artifacts.transcript_trace,
            permutation_inputs: artifacts.poseidon2_perm_inputs,
            compression_inputs: artifacts.poseidon2_compress_inputs,
        })
    }

    /// Variant allowing a resumed verifier to obtain only its terminal
    /// checkpoint. Its start checkpoint is authenticated by the preceding
    /// source verifier and routed directly to the checkpoint bus.
    pub fn generate_trace_inputs_with_external_resumed_and_optional_checkpoints(
        &self,
        logs: &[&TranscriptLog<F, [F; POSEIDON2_WIDTH]>],
        resumes: &[Option<(usize, [F; POSEIDON2_WIDTH])>],
        checkpoints: &[Option<[Option<usize>; 2]>],
        external_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        external_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_transcript_height: Option<usize>,
    ) -> Option<NativeWarpTranscriptInputArtifacts> {
        let artifacts = self
            .inner
            .build_transcript_trace_artifacts_with_optional_checkpoints(
                logs,
                resumes,
                checkpoints,
                external_permutation_inputs,
                external_compression_inputs,
                required_transcript_height,
            )?;
        Some(NativeWarpTranscriptInputArtifacts {
            trace: artifacts.transcript_trace,
            permutation_inputs: artifacts.poseidon2_perm_inputs,
            compression_inputs: artifacts.poseidon2_compress_inputs,
        })
    }

    pub fn build_poseidon2_trace(
        &self,
        permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_height: Option<usize>,
    ) -> Option<RowMajorMatrix<F>> {
        self.inner
            .build_poseidon2_trace(permutation_inputs, compression_inputs, required_height)
    }

    pub fn build_poseidon2_multibus_traces(
        &self,
        grouped_inputs: Poseidon2MultibusInputs,
    ) -> Option<Vec<RowMajorMatrix<F>>> {
        self.inner.build_poseidon2_multibus_traces(grouped_inputs)
    }

    pub fn build_poseidon2_multibus_sharded_traces(
        &self,
        grouped_inputs: Poseidon2MultibusInputs,
        shard_count: usize,
        max_rows: usize,
    ) -> Option<Vec<RowMajorMatrix<F>>> {
        self.inner
            .build_poseidon2_multibus_sharded_traces(grouped_inputs, shard_count, max_rows)
    }

    /// As [`Self::build_poseidon2_trace`], but writing straight into device
    /// memory. See [`TranscriptModule::build_poseidon2_trace_gpu`].
    #[cfg(feature = "cuda")]
    pub fn build_poseidon2_trace_gpu(
        &self,
        permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
        required_height: Option<usize>,
        device_ctx: &openvm_cuda_common::stream::GpuDeviceCtx,
    ) -> Option<openvm_cuda_backend::base::DeviceMatrix<F>> {
        self.inner.build_poseidon2_trace_gpu(
            permutation_inputs,
            compression_inputs,
            required_height,
            device_ctx,
        )
    }

    #[cfg(feature = "cuda")]
    pub fn build_poseidon2_multibus_traces_gpu(
        &self,
        grouped_inputs: Poseidon2MultibusInputs,
        device_ctx: &openvm_cuda_common::stream::GpuDeviceCtx,
    ) -> Option<Vec<openvm_cuda_backend::base::DeviceMatrix<F>>> {
        self.inner
            .build_poseidon2_multibus_traces_gpu(grouped_inputs, device_ctx)
    }
}

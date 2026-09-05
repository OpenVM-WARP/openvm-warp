//! Typed boundary between OpenVM's deferred SWIRL prover and WARP.
//!
//! These carriers preserve the original stacked commitment and its PCS owner.
//! They contain no completed segment proof and do not reinterpret a PCS
//! opening as PESAT.

use openvm_circuit::arch::{vm_segment_metadata_from_parts, VmSegmentMetadata};
use openvm_recursion_circuit::system::RetainedStackingProof;
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey,
    p3_field::{BasedVectorSpace, PrimeCharacteristicRing},
    proof::{BatchConstraintProof, GkrProof, StackingProof, TraceVData},
    prover::{
        NativeStackingReduction, PendingConstrainedCodeClaim, PendingConstrainedCodeMetadata,
        PendingConstrainedCodeWitness, ProverBackend,
    },
    warp_accum::{
        ReducedConstrainedCodeClaim, ReducedConstrainedCodeRelation, StackedRsFreshCommitment,
        SwirlConstrainedRsRelation,
    },
    FiatShamirTranscript,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, Digest, EF, F,
};

use crate::SC;

const REDUCED_SWIRL_BOUNDARY_VERSION: u32 = 3;
const SOURCE_DIGEST_TAG: u32 = 0x5253_0101;
const FIELD_TAG: u32 = 0x5253_0201;
const DIGEST_TAG: u32 = 0x5253_0202;
const EXTENSION_TAG: u32 = 0x5253_0203;
const END_TAG: u32 = 0x5253_02ff;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReducedSwirlVmState {
    pub pc: F,
    pub memory_root: Digest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReducedSwirlVmBoundary {
    pub program_commitment: Digest,
    pub initial_state: ReducedSwirlVmState,
    pub final_state: ReducedSwirlVmState,
    pub exit_code: F,
    pub is_terminate: F,
}

/// Verifier-authoritative public half of a pending constrained-code witness.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PendingConstrainedCodePublicClaim {
    pub metadata: PendingConstrainedCodeMetadata<Digest>,
    pub swirl_tilde_u: Vec<EF>,
    pub stacking_openings: Vec<Vec<EF>>,
}

impl PendingConstrainedCodePublicClaim {
    pub fn digest(&self) -> Result<Digest, ReducedSwirlBoundaryError> {
        let mut observations = Vec::new();
        push_usize(
            &mut observations,
            self.metadata.ordered_commitments().len(),
            "pending commitment count",
        )?;
        observations.extend(
            self.metadata
                .ordered_commitments()
                .iter()
                .copied()
                .map(Observation::Digest),
        );
        push_usize(
            &mut observations,
            self.metadata.commitment_widths().len(),
            "pending commitment width count",
        )?;
        for &width in self.metadata.commitment_widths() {
            push_usize(&mut observations, width, "pending commitment width")?;
        }
        for (value, name) in [
            (self.metadata.l_skip(), "pending l_skip"),
            (self.metadata.n_stack(), "pending n_stack"),
            (self.metadata.log_blowup(), "pending log blowup"),
            (
                self.metadata.log_commit_rows_per_query(),
                "pending Merkle leaf log rows",
            ),
            (self.metadata.total_width(), "pending total width"),
        ] {
            push_usize(&mut observations, value, name)?;
        }
        push_extension_slice(&mut observations, &self.swirl_tilde_u)?;
        push_usize(
            &mut observations,
            self.stacking_openings.len(),
            "stacking opening group count",
        )?;
        for opening in &self.stacking_openings {
            push_extension_slice(&mut observations, opening)?;
        }
        digest_observations(SOURCE_DIGEST_TAG ^ 0x30, &observations)
    }

    pub fn authoritative_constrained_rs_claim_for_theta(
        &self,
        theta: EF,
        row_zero_claim: OriginalRootProjectedRowZeroClaim,
    ) -> Result<AuthoritativeSwirlConstrainedRsClaim, ReducedSwirlBoundaryError> {
        self.validate()?;
        self.validate_row_zero_claim(&row_zero_claim)?;
        let log_codeword_len = self
            .metadata
            .point_dimension()
            .checked_add(self.metadata.log_blowup())
            .ok_or(ReducedSwirlBoundaryError::CountOverflow(
                "codeword opening dimension",
            ))?;
        Ok(AuthoritativeSwirlConstrainedRsClaim {
            root_tuple: self.metadata.ordered_commitments().to_vec(),
            commitment_widths: self.metadata.commitment_widths().to_vec(),
            l_skip: self.metadata.l_skip(),
            n_stack: self.metadata.n_stack(),
            log_blowup: self.metadata.log_blowup(),
            log_commit_rows_per_query: self.metadata.log_commit_rows_per_query(),
            theta,
            alpha: vec![EF::ZERO; log_codeword_len],
            mu: row_zero_claim.mu,
            beta: self.swirl_tilde_u.iter().rev().copied().collect(),
            eta: fold_ordered_openings(&self.stacking_openings, theta),
        })
    }

    pub fn authoritative_claim_from_backend(
        &self,
        backend_claim: &ReducedConstrainedCodeClaim<EF, StackedRsFreshCommitment<EF, Digest>>,
    ) -> Result<AuthoritativeSwirlConstrainedRsClaim, ReducedSwirlBoundaryError> {
        let relation = SwirlConstrainedRsRelation::<EF>::new(self.metadata.point_dimension())
            .map_err(|error| ReducedSwirlBoundaryError::Binding(error.to_string()))?;
        let relation_binding = <SwirlConstrainedRsRelation<EF> as ReducedConstrainedCodeRelation<
            F,
            EF,
        >>::relation_binding(&relation);
        if backend_claim.binding.relation_binding != relation_binding {
            return Err(ReducedSwirlBoundaryError::BackendClaimMismatch(
                "relation binding",
            ));
        }
        let expected_log_codeword_len = self
            .metadata
            .point_dimension()
            .checked_add(self.metadata.log_blowup())
            .ok_or(ReducedSwirlBoundaryError::CountOverflow(
                "backend codeword dimension",
            ))?;
        let commitment = &backend_claim.commitment;
        if commitment.roots != self.metadata.ordered_commitments()
            || commitment.widths != self.metadata.commitment_widths()
            || commitment.l_skip != self.metadata.l_skip()
            || commitment.native_log_message_len != self.metadata.point_dimension()
            || commitment.log_message_len != self.metadata.point_dimension()
            || commitment.log_codeword_len != expected_log_codeword_len
            || commitment.rows_per_query != checked_rows_per_query(&self.metadata)?
        {
            return Err(ReducedSwirlBoundaryError::BackendClaimMismatch(
                "commitment layout",
            ));
        }
        let expected = self.authoritative_constrained_rs_claim_for_theta(
            commitment.theta,
            OriginalRootProjectedRowZeroClaim {
                original_root_tuple: commitment.roots.clone(),
                codeword_row_index: 0,
                mu: backend_claim.mu,
            },
        )?;
        if backend_claim.alpha != expected.alpha
            || backend_claim.beta != expected.beta
            || backend_claim.eta != expected.eta
        {
            return Err(ReducedSwirlBoundaryError::BackendClaimMismatch(
                "alpha/mu/beta/eta",
            ));
        }
        Ok(expected)
    }

    fn validate(&self) -> Result<(), ReducedSwirlBoundaryError> {
        PendingConstrainedCodeClaim::try_new(
            self.metadata.clone(),
            self.swirl_tilde_u.clone(),
            &self.stacking_openings,
        )
        .map(|_| ())
        .map_err(|error| ReducedSwirlBoundaryError::PendingClaim(error.to_string()))
    }

    fn validate_row_zero_claim(
        &self,
        row_zero_claim: &OriginalRootProjectedRowZeroClaim,
    ) -> Result<(), ReducedSwirlBoundaryError> {
        if row_zero_claim.original_root_tuple != self.metadata.ordered_commitments() {
            return Err(ReducedSwirlBoundaryError::ProjectedRowRootSubstitution);
        }
        if row_zero_claim.codeword_row_index != 0 {
            return Err(ReducedSwirlBoundaryError::ProjectedCodewordRow(
                row_zero_claim.codeword_row_index,
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OriginalRootProjectedRowZeroClaim {
    pub original_root_tuple: Vec<Digest>,
    pub codeword_row_index: usize,
    pub mu: EF,
}

/// The scalar constrained-RS claim jointly authenticated by the SWIRL prefix
/// receipt and reduced WARP.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthoritativeSwirlConstrainedRsClaim {
    pub root_tuple: Vec<Digest>,
    pub commitment_widths: Vec<usize>,
    pub l_skip: usize,
    pub n_stack: usize,
    pub log_blowup: usize,
    pub log_commit_rows_per_query: usize,
    pub theta: EF,
    pub alpha: Vec<EF>,
    pub mu: EF,
    pub beta: Vec<EF>,
    pub eta: EF,
}

impl AuthoritativeSwirlConstrainedRsClaim {
    #[must_use]
    pub const fn log_message_len(&self) -> usize {
        self.l_skip + self.n_stack
    }

    pub fn digest(&self) -> Result<Digest, ReducedSwirlBoundaryError> {
        let mut observations = Vec::new();
        push_usize(
            &mut observations,
            self.root_tuple.len(),
            "reduced root count",
        )?;
        observations.extend(self.root_tuple.iter().copied().map(Observation::Digest));
        push_usize(
            &mut observations,
            self.commitment_widths.len(),
            "reduced width count",
        )?;
        for &width in &self.commitment_widths {
            push_usize(&mut observations, width, "reduced width")?;
        }
        for (value, name) in [
            (self.l_skip, "reduced l_skip"),
            (self.n_stack, "reduced n_stack"),
            (self.log_blowup, "reduced log blowup"),
            (
                self.log_commit_rows_per_query,
                "reduced Merkle leaf log rows",
            ),
        ] {
            push_usize(&mut observations, value, name)?;
        }
        observations.push(Observation::Extension(self.theta));
        push_extension_slice(&mut observations, &self.alpha)?;
        observations.push(Observation::Extension(self.mu));
        push_extension_slice(&mut observations, &self.beta)?;
        observations.push(Observation::Extension(self.eta));
        digest_observations(SOURCE_DIGEST_TAG ^ 0x40, &observations)
    }
}

/// Owning handoff for one original SWIRL stacking reduction.
pub struct ReducedSwirlPrefix<DeferredPcsData> {
    segment_index: u32,
    retained: RetainedStackingProof,
    pending_witness: PendingConstrainedCodeWitness<Digest, Vec<EF>, DeferredPcsData>,
    vm_pvs: ReducedSwirlVmBoundary,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlSourceManifestPrefix {
    pub source_index: u32,
    pub segment_index: u32,
    pub common_main_root: Digest,
    pub trace_layout_digest: Digest,
    pub pending_claim_digest: Digest,
    pub vm_pvs: ReducedSwirlVmBoundary,
}

impl ReducedSwirlSourceManifestPrefix {
    pub fn digest_with_claim(
        &self,
        authoritative_claim: &AuthoritativeSwirlConstrainedRsClaim,
    ) -> Result<Digest, ReducedSwirlBoundaryError> {
        let mut observations = Vec::new();
        push_u32(&mut observations, self.source_index);
        push_u32(&mut observations, self.segment_index);
        observations.push(Observation::Digest(self.common_main_root));
        observations.push(Observation::Digest(self.trace_layout_digest));
        observations.push(Observation::Digest(self.pending_claim_digest));
        observations.push(Observation::Digest(authoritative_claim.digest()?));
        push_vm_pvs(&mut observations, self.vm_pvs);
        digest_observations(SOURCE_DIGEST_TAG ^ 0x70, &observations)
    }
}

impl<DeferredPcsData> ReducedSwirlPrefix<DeferredPcsData> {
    pub fn from_native_reduction<PB>(
        app_vk: &MultiStarkVerifyingKey<SC>,
        segment_index: usize,
        reduction: NativeStackingReduction<
            SC,
            PB,
            (GkrProof<SC>, BatchConstraintProof<SC>),
            StackingProof<SC>,
            Vec<EF>,
            DeferredPcsData,
        >,
    ) -> Result<Self, ReducedSwirlBoundaryError>
    where
        PB: ProverBackend<Val = F, Challenge = EF, Commitment = Digest>,
    {
        let segment_index = u32::try_from(segment_index)
            .map_err(|_| ReducedSwirlBoundaryError::CountOverflow("segment index"))?;
        let (retained, pending_witness) = RetainedStackingProof::split_native_reduction(reduction);
        let metadata =
            vm_segment_metadata_from_parts(app_vk, &retained.trace_vdata, &retained.public_values)
                .map_err(|error| {
                    ReducedSwirlBoundaryError::SegmentMetadata(format!("{error:?}"))
                })?;
        Ok(Self {
            segment_index,
            retained,
            pending_witness,
            vm_pvs: vm_boundary(&metadata),
        })
    }

    #[must_use]
    pub const fn retained(&self) -> &RetainedStackingProof {
        &self.retained
    }

    #[must_use]
    pub const fn pending_witness(
        &self,
    ) -> &PendingConstrainedCodeWitness<Digest, Vec<EF>, DeferredPcsData> {
        &self.pending_witness
    }

    pub fn manifest_prefix(
        &self,
        source_index: usize,
    ) -> Result<ReducedSwirlSourceManifestPrefix, ReducedSwirlBoundaryError> {
        let pending_claim = PendingConstrainedCodePublicClaim {
            metadata: self.pending_witness.metadata().clone(),
            swirl_tilde_u: self.pending_witness.terminal_point().clone(),
            stacking_openings: self.retained.stacking_proof.stacking_openings.clone(),
        };
        Ok(ReducedSwirlSourceManifestPrefix {
            source_index: to_u32(source_index, "source index")?,
            segment_index: self.segment_index,
            common_main_root: self.retained.common_main_commit,
            trace_layout_digest: trace_layout_digest(&self.retained.trace_vdata)?,
            pending_claim_digest: pending_claim.digest()?,
            vm_pvs: self.vm_pvs,
        })
    }

    #[must_use]
    pub fn into_parts(
        self,
    ) -> (
        RetainedStackingProof,
        PendingConstrainedCodeWitness<Digest, Vec<EF>, DeferredPcsData>,
    ) {
        (self.retained, self.pending_witness)
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ReducedSwirlBoundaryError {
    #[error("reduced SWIRL {0} exceeds the canonical u32 encoding")]
    CountOverflow(&'static str),
    #[error("reduced SWIRL VM segment metadata rejected: {0}")]
    SegmentMetadata(String),
    #[error("reduced SWIRL binding rejected: {0}")]
    Binding(String),
    #[error("reduced SWIRL pending constrained-code claim rejected: {0}")]
    PendingClaim(String),
    #[error("reduced SWIRL projected codeword row is {0}, expected canonical row zero")]
    ProjectedCodewordRow(usize),
    #[error("reduced SWIRL projected row opening substituted its original root tuple")]
    ProjectedRowRootSubstitution,
    #[error("reduced SWIRL backend claim differs from the authoritative {0}")]
    BackendClaimMismatch(&'static str),
}

#[derive(Clone, Copy)]
enum Observation {
    Field(F),
    Digest(Digest),
    Extension(EF),
}

fn vm_boundary(metadata: &VmSegmentMetadata<SC>) -> ReducedSwirlVmBoundary {
    ReducedSwirlVmBoundary {
        program_commitment: metadata.program_commit,
        initial_state: ReducedSwirlVmState {
            pc: metadata.initial_pc,
            memory_root: metadata.initial_memory_root,
        },
        final_state: ReducedSwirlVmState {
            pc: metadata.final_pc,
            memory_root: metadata.final_memory_root,
        },
        exit_code: metadata.exit_code,
        is_terminate: metadata.is_terminate,
    }
}

fn trace_layout_digest(
    trace_vdata: &[Option<TraceVData<SC>>],
) -> Result<Digest, ReducedSwirlBoundaryError> {
    let mut observations = Vec::new();
    push_usize(&mut observations, trace_vdata.len(), "AIR count")?;
    for (air_index, trace) in trace_vdata.iter().enumerate() {
        push_usize(&mut observations, air_index, "AIR index")?;
        observations.push(Observation::Field(F::from_bool(trace.is_some())));
        if let Some(trace) = trace {
            push_usize(&mut observations, trace.log_height, "trace log height")?;
            push_usize(
                &mut observations,
                trace.cached_commitments.len(),
                "cached commitment count",
            )?;
            observations.extend(
                trace
                    .cached_commitments
                    .iter()
                    .copied()
                    .map(Observation::Digest),
            );
        }
    }
    digest_observations(SOURCE_DIGEST_TAG ^ 0x10, &observations)
}

fn push_vm_pvs(observations: &mut Vec<Observation>, vm: ReducedSwirlVmBoundary) {
    observations.push(Observation::Digest(vm.program_commitment));
    observations.push(Observation::Field(vm.initial_state.pc));
    observations.push(Observation::Digest(vm.initial_state.memory_root));
    observations.push(Observation::Field(vm.final_state.pc));
    observations.push(Observation::Digest(vm.final_state.memory_root));
    observations.push(Observation::Field(vm.exit_code));
    observations.push(Observation::Field(vm.is_terminate));
}

fn fold_ordered_openings(openings: &[Vec<EF>], theta: EF) -> EF {
    let mut power = EF::ONE;
    let mut value = EF::ZERO;
    for opening in openings {
        for &coordinate in opening {
            value += power * coordinate;
            power *= theta;
        }
    }
    value
}

fn checked_rows_per_query(
    metadata: &PendingConstrainedCodeMetadata<Digest>,
) -> Result<usize, ReducedSwirlBoundaryError> {
    1usize
        .checked_shl(
            metadata
                .log_commit_rows_per_query()
                .try_into()
                .map_err(|_| ReducedSwirlBoundaryError::CountOverflow("rows per query"))?,
        )
        .ok_or(ReducedSwirlBoundaryError::CountOverflow("rows per query"))
}

fn push_extension_slice(
    observations: &mut Vec<Observation>,
    values: &[EF],
) -> Result<(), ReducedSwirlBoundaryError> {
    push_usize(observations, values.len(), "extension vector length")?;
    observations.extend(values.iter().copied().map(Observation::Extension));
    Ok(())
}

fn push_usize(
    observations: &mut Vec<Observation>,
    value: usize,
    name: &'static str,
) -> Result<(), ReducedSwirlBoundaryError> {
    push_u32(observations, to_u32(value, name)?);
    Ok(())
}

fn push_u32(observations: &mut Vec<Observation>, value: u32) {
    observations.push(Observation::Field(F::from_u32(value & 0xffff)));
    observations.push(Observation::Field(F::from_u32(value >> 16)));
}

fn to_u32(value: usize, name: &'static str) -> Result<u32, ReducedSwirlBoundaryError> {
    u32::try_from(value).map_err(|_| ReducedSwirlBoundaryError::CountOverflow(name))
}

fn digest_observations(
    domain_tag: u32,
    observations: &[Observation],
) -> Result<Digest, ReducedSwirlBoundaryError> {
    let mut transcript = default_duplex_sponge_recorder();
    observe_field(&mut transcript, F::from_u32(domain_tag));
    observe_field(&mut transcript, F::from_u32(REDUCED_SWIRL_BOUNDARY_VERSION));
    let count = to_u32(observations.len(), "observation count")?;
    observe_field(&mut transcript, F::from_u32(count & 0xffff));
    observe_field(&mut transcript, F::from_u32(count >> 16));
    for observation in observations {
        match observation {
            Observation::Field(value) => {
                observe_field(&mut transcript, F::from_u32(FIELD_TAG));
                observe_field(&mut transcript, *value);
            }
            Observation::Digest(digest) => {
                observe_field(&mut transcript, F::from_u32(DIGEST_TAG));
                for value in digest {
                    observe_field(&mut transcript, *value);
                }
            }
            Observation::Extension(value) => {
                observe_field(&mut transcript, F::from_u32(EXTENSION_TAG));
                for limb in <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(value) {
                    observe_field(&mut transcript, *limb);
                }
            }
        }
    }
    observe_field(&mut transcript, F::from_u32(END_TAG));
    Ok(core::array::from_fn(|_| {
        <_ as FiatShamirTranscript<SC>>::sample(&mut transcript)
    }))
}

fn observe_field(transcript: &mut impl FiatShamirTranscript<SC>, value: F) {
    transcript.observe(value);
}

//! Native streaming composition of SWIRL's deferred opening boundary with WARP.
//!
//! Each source is the exact retained stacked SWIRL commitment. The fresh path
//! projects its columns linearly and authenticates WARP shift queries against
//! the original roots; it never RS-encodes or commits a second scalar oracle.
//! AIR/LogUp/stacking prefix verification is deliberately outside this module
//! and supplies the authoritative ordered source claims used by verification.

use std::{fmt::Display, ops::Range};

use openvm_cpu_backend::{CpuReducedSwirlOpeningAdapter, CpuReducedSwirlSource, CpuStackedPcsData};
use openvm_stark_backend::{
    native_warp::NativeWarpChallenger,
    p3_field::PrimeCharacteristicRing,
    soundness,
    warp_accum::{
        canonical_swirl_reduced_code_binding, derive_swirl_constrained_rs_terminal_statement,
        finish_warp_call, prove_terminal_constrained_whir_owned, select_proven_rs_warp_params,
        verify_terminal_constrained_whir_recorded, Accumulator, FieldElementDigestObserver,
        MerkleBatchOpeningProof, MerkleBatchOpeningVerification, MerkleOpeningBackend,
        ProvenRsWarpSecurity, ReducedConstrainedCodeBinding, ReducedConstrainedCodeClaim,
        ReducedConstrainedCodeRelation, ReducedWarpVaccRootProof, ReducedWarpVaccStepProof,
        StackedRsBatchOpeningProof, StackedRsBatchOpeningVerification, StackedRsFreshCommitment,
        StackedRsOpeningBackend, SwirlConstrainedRsRelation, TerminalDescriptor, TerminalWhirProof,
        TerminalWhirVerification, WarpAccumError, WarpRootProver, WarpVaccStepProverRecord,
        WarpVaccStepVerification, WhirInitialRsWarpCode, WhirRsCodeProverData,
    },
    warp_pesat::{AlgebraicChallenger, LinearChainSchedule, PesatShape},
    FiatShamirTranscript, StarkProtocolConfig, SystemParams, WhirConfig, WhirProximityStrategy,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, Digest, DuplexSpongeRecorder, EF, F,
};
use serde::{Deserialize, Serialize};

use crate::SC;

pub const REDUCED_SWIRL_NATIVE_PROTOCOL_VERSION: u32 = 3;
/// Fixed source capacity of the production recursion key. Runtime executions
/// use an active prefix of this profile; the key and WARP security parameters
/// therefore never depend on the block's actual segment count.
pub const REDUCED_SWIRL_MAX_SOURCES: usize = 1024;
const REDUCED_SWIRL_NATIVE_TRANSCRIPT_TAG: &[u8] = b"openvm.native-warp.swirl-reduced-source.v3";
const REDUCED_SWIRL_NATIVE_SOURCE_BATCH_TAG: &[u8] =
    b"openvm.native-warp.swirl-reduced-source.batch.v3";
const REDUCED_SWIRL_NATIVE_MANIFEST_TAG: &[u8] =
    b"openvm.native-warp.swirl-reduced-source.manifest.v3";
const REDUCED_SWIRL_NATIVE_MANIFEST_FOOTER_TAG: &[u8] =
    b"openvm.native-warp.swirl-reduced-source.manifest-footer.v3";
const REDUCED_SWIRL_NATIVE_THETA_SEED_TAG: &[u8] =
    b"openvm.native-warp.swirl-reduced-source.theta-seed.v1";

pub type ReducedSwirlCpuSource = CpuReducedSwirlSource<F, EF, Digest, CpuStackedPcsData<F, Digest>>;
pub type ReducedSwirlFreshCommitment = StackedRsFreshCommitment<EF, Digest>;
pub type ReducedSwirlFreshProof = StackedRsBatchOpeningProof<F, Digest>;
pub type ReducedSwirlAccumulatorProof = MerkleBatchOpeningProof<EF, Digest>;
pub type ReducedSwirlVaccProof = ReducedWarpVaccRootProof<
    EF,
    Digest,
    ReducedSwirlFreshProof,
    ReducedSwirlAccumulatorProof,
    ReducedSwirlFreshCommitment,
>;
pub type ReducedSwirlVaccStepProof = ReducedWarpVaccStepProof<
    EF,
    Digest,
    ReducedSwirlFreshProof,
    ReducedSwirlAccumulatorProof,
    ReducedSwirlFreshCommitment,
>;
pub type ReducedSwirlTerminalAccumulator =
    Accumulator<EF, Digest, WhirRsCodeProverData<EF, Digest>>;
pub type ReducedSwirlCode =
    WhirInitialRsWarpCode<<SC as StarkProtocolConfig>::Hasher, FieldElementDigestObserver>;
pub type ReducedSwirlFreshVerification = StackedRsBatchOpeningVerification<F, EF, Digest>;
pub type ReducedSwirlAccumulatorVerification = MerkleBatchOpeningVerification<EF, Digest>;
pub type ReducedSwirlStepVerification = WarpVaccStepVerification<
    EF,
    Digest,
    ReducedSwirlFreshVerification,
    ReducedSwirlAccumulatorVerification,
>;

/// Verifier-owned block statement. `source_bindings` are canonical ordered
/// source-entry digests. `block_manifest_digest` is derived from that exact
/// list; callers cannot select it independently.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReducedSwirlNativeStatement {
    pub protocol_version: u32,
    pub block_manifest_digest: Digest,
    pub source_bindings: Vec<Digest>,
}

/// Verifier-derived SWIRL power-batching error budget for the exact source
/// family carried by one proof.
///
/// In SWIRL's unique-decoding regime, source `i` contributes at most
/// `(w_i - 1) * |L| / |EF|`, where `w_i` is its committed column count and
/// `L` is the RS evaluation domain. The numerator is summed over all sources;
/// no target-security value supplied by the prover is trusted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReducedSwirlPowerBatchSecurityBudget {
    pub source_count: usize,
    pub total_codewords: usize,
    pub rs_domain_size: usize,
    pub exceptional_set_numerator: usize,
    pub available_field_bits_floor: usize,
    pub conservative_security_bits_floor: usize,
}

impl ReducedSwirlPowerBatchSecurityBudget {
    pub fn derive(
        claims: &[ReducedConstrainedCodeClaim<EF, ReducedSwirlFreshCommitment>],
    ) -> Result<Self, ReducedSwirlNativeError> {
        let first = claims
            .first()
            .ok_or(ReducedSwirlNativeError::ProofMismatch(
                "empty source family",
            ))?;
        let log_codeword_len = first.commitment.log_codeword_len;
        let rs_domain_size = 1usize.checked_shl(log_codeword_len as u32).ok_or(
            ReducedSwirlNativeError::ProofMismatch("SWIRL RS domain size"),
        )?;
        let mut total_codewords = 0usize;
        let mut exceptional_set_numerator = 0usize;
        for claim in claims {
            if claim.commitment.log_codeword_len != log_codeword_len {
                return Err(ReducedSwirlNativeError::ProofMismatch(
                    "heterogeneous SWIRL RS domains",
                ));
            }
            let codewords = claim
                .commitment
                .widths
                .iter()
                .try_fold(0usize, |sum, width| sum.checked_add(*width))
                .ok_or(ReducedSwirlNativeError::ProofMismatch(
                    "SWIRL codeword count overflow",
                ))?;
            if codewords == 0 {
                return Err(ReducedSwirlNativeError::ProofMismatch(
                    "empty SWIRL power batch",
                ));
            }
            total_codewords = total_codewords.checked_add(codewords).ok_or(
                ReducedSwirlNativeError::ProofMismatch("SWIRL codeword family overflow"),
            )?;
            let source_numerator = codewords
                .saturating_sub(1)
                .checked_mul(rs_domain_size)
                .ok_or(ReducedSwirlNativeError::ProofMismatch(
                    "SWIRL exceptional-set overflow",
                ))?;
            exceptional_set_numerator = exceptional_set_numerator
                .checked_add(source_numerator)
                .ok_or(ReducedSwirlNativeError::ProofMismatch(
                    "SWIRL family exceptional-set overflow",
                ))?;
        }
        let available_field_bits_floor = soundness::challenge_field_bits::<SC>().floor() as usize;
        let loss_bits = if exceptional_set_numerator <= 1 {
            0
        } else {
            usize::BITS as usize - (exceptional_set_numerator - 1).leading_zeros() as usize
        };
        Ok(Self {
            source_count: claims.len(),
            total_codewords,
            rs_domain_size,
            exceptional_set_numerator,
            available_field_bits_floor,
            conservative_security_bits_floor: available_field_bits_floor.saturating_sub(loss_bits),
        })
    }
}

impl ReducedSwirlNativeStatement {
    pub fn validate(&self) -> Result<(), ReducedSwirlNativeError> {
        if self.protocol_version != REDUCED_SWIRL_NATIVE_PROTOCOL_VERSION {
            return Err(ReducedSwirlNativeError::Statement("protocol version"));
        }
        if self.source_bindings.is_empty() {
            return Err(ReducedSwirlNativeError::Statement("empty source list"));
        }
        if self.block_manifest_digest != reduced_swirl_manifest_digest(&self.source_bindings)?
            || self
                .source_bindings
                .iter()
                .any(|digest| digest.iter().all(|value| *value == F::ZERO))
        {
            return Err(ReducedSwirlNativeError::Statement("unset manifest digest"));
        }
        Ok(())
    }
}

/// Trusted setup for one fixed SWIRL constrained-RS relation and scalar RS
/// code class. Runtime traces and roots cannot change these values.
#[derive(Clone)]
pub struct ReducedSwirlNativeSetup {
    relation: SwirlConstrainedRsRelation<EF>,
    binding: ReducedConstrainedCodeBinding<EF>,
    code: ReducedSwirlCode,
    warp: WarpRootProver<ReducedSwirlCode>,
    whir: WhirConfig,
    security: ProvenRsWarpSecurity,
    family_target_bits: usize,
    maximum_source_count: usize,
}

impl ReducedSwirlNativeSetup {
    pub fn new(
        config: &SC,
        params: &SystemParams,
        input_arity: usize,
        family_target_bits: usize,
        maximum_source_count: usize,
    ) -> Result<Self, ReducedSwirlNativeError> {
        if maximum_source_count == 0 || family_target_bits == 0 {
            return Err(ReducedSwirlNativeError::Setup("empty security family"));
        }
        // The power-batching budget below implements SWIRL's unique-decoding
        // exceptional-set bound.  Accepting list-decoding parameters while
        // retaining that formula would silently overstate soundness.
        if params.whir.proximity != WhirProximityStrategy::UniqueDecoding {
            return Err(ReducedSwirlNativeError::Setup(
                "SWIRL power batching requires unique decoding",
            ));
        }
        if params.log_commit_rows_per_query != params.whir.k {
            return Err(ReducedSwirlNativeError::Setup(
                "SWIRL/terminal Merkle row layout",
            ));
        }
        let log_message_len = params
            .l_skip
            .checked_add(params.n_stack)
            .ok_or(ReducedSwirlNativeError::Setup("message dimension"))?;
        let log_codeword_len = log_message_len
            .checked_add(params.log_blowup)
            .ok_or(ReducedSwirlNativeError::Setup("codeword dimension"))?;
        let rows_per_query = 1usize
            .checked_shl(params.log_commit_rows_per_query as u32)
            .ok_or(ReducedSwirlNativeError::Setup("rows per query"))?;
        let relation = SwirlConstrainedRsRelation::new(log_message_len)
            .map_err(ReducedSwirlNativeError::Setup)?;
        let relation_shape = <SwirlConstrainedRsRelation<EF> as ReducedConstrainedCodeRelation<
            F,
            EF,
        >>::shape(&relation);
        let pesat_shape = PesatShape {
            log_constraints: 0,
            log_witness: relation_shape.log_message_len,
            explicit_len: relation_shape.beta_len + 1,
            max_degree: relation_shape.max_constraint_degree,
        };
        let schedule = LinearChainSchedule::new(input_arity)
            .map_err(WarpAccumError::from)
            .map_err(ReducedSwirlNativeError::Warp)?;
        let call_count = schedule.step_fresh_counts(maximum_source_count).len();
        let union_bits = call_count.next_power_of_two().ilog2() as usize;
        let selected_target = family_target_bits
            .checked_add(union_bits)
            .ok_or(ReducedSwirlNativeError::Setup("security union bound"))?;
        let security = select_proven_rs_warp_params(
            input_arity,
            log_message_len,
            log_codeword_len,
            pesat_shape,
            soundness::challenge_field_bits::<SC>(),
            selected_target,
        )
        .map_err(ReducedSwirlNativeError::Warp)?;
        let code = WhirInitialRsWarpCode::try_new_coefficient_subgroup(
            config.hasher().clone(),
            log_message_len,
            params.log_blowup,
            0,
            rows_per_query,
        )
        .map_err(ReducedSwirlNativeError::Warp)?;
        let warp = WarpRootProver::new(code.clone(), schedule, security.params)
            .map_err(ReducedSwirlNativeError::Warp)?;
        let binding = canonical_swirl_reduced_code_binding(
            &relation,
            params.l_skip,
            params.n_stack,
            params.log_blowup,
            params.log_commit_rows_per_query,
        );
        Ok(Self {
            relation,
            binding,
            code,
            warp,
            whir: params.whir.clone(),
            security,
            family_target_bits,
            maximum_source_count,
        })
    }

    #[must_use]
    pub const fn relation(&self) -> &SwirlConstrainedRsRelation<EF> {
        &self.relation
    }

    #[must_use]
    pub const fn binding(&self) -> &ReducedConstrainedCodeBinding<EF> {
        &self.binding
    }

    #[must_use]
    pub const fn code(&self) -> &ReducedSwirlCode {
        &self.code
    }

    #[must_use]
    pub const fn security(&self) -> &ProvenRsWarpSecurity {
        &self.security
    }

    #[must_use]
    pub const fn family_target_bits(&self) -> usize {
        self.family_target_bits
    }

    #[must_use]
    pub const fn maximum_source_count(&self) -> usize {
        self.maximum_source_count
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReducedSwirlNativeTerminalProof {
    pub descriptor: TerminalDescriptor<Digest>,
    pub same_root_whir: TerminalWhirProof<SC>,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReducedSwirlNativeProof {
    pub statement: ReducedSwirlNativeStatement,
    pub vacc: ReducedSwirlVaccProof,
    pub terminal: ReducedSwirlNativeTerminalProof,
}

pub struct ReducedSwirlNativeProverOutput {
    pub proof: ReducedSwirlNativeProof,
    pub authoritative_claims: Vec<ReducedConstrainedCodeClaim<EF, ReducedSwirlFreshCommitment>>,
    pub transition_records: Vec<WarpVaccStepProverRecord<EF, Digest>>,
    pub security: ProvenRsWarpSecurity,
    pub swirl_power_batch_security: ReducedSwirlPowerBatchSecurityBudget,
}

pub struct ReducedSwirlNativeVerification {
    pub final_instance: openvm_stark_backend::warp_pesat::AccumulatorInstance<EF, Digest>,
    pub transition_records: Vec<ReducedSwirlStepVerification>,
    pub terminal: TerminalWhirVerification<F, EF, Digest>,
    pub complete_transcript: DuplexSpongeRecorder,
    pub swirl_power_batch_security: ReducedSwirlPowerBatchSecurityBudget,
}

/// Bounded-memory CPU prover for an ordered stream of genuine SWIRL reduced
/// sources. At most one WARP invocation's fresh PCS owners are retained.
/// Source bindings are absorbed immediately before the corresponding VACC
/// transition, so they need not be known before segment execution starts.
pub struct ReducedSwirlNativeCpuStream<'a> {
    setup: &'a ReducedSwirlNativeSetup,
    expected_source_count: usize,
    step_fresh_counts: Vec<usize>,
    challenger: NativeWarpChallenger<SC, DuplexSpongeRecorder>,
    pending_sources: Vec<ReducedSwirlCpuSource>,
    pending_bindings: Vec<Digest>,
    source_bindings: Vec<Digest>,
    authoritative_claims: Vec<ReducedConstrainedCodeClaim<EF, ReducedSwirlFreshCommitment>>,
    accumulator: Option<ReducedSwirlTerminalAccumulator>,
    steps: Vec<ReducedSwirlVaccStepProof>,
    transition_records: Vec<WarpVaccStepProverRecord<EF, Digest>>,
}

#[derive(Debug, thiserror::Error)]
pub enum ReducedSwirlNativeError {
    #[error("invalid reduced-SWIRL setup: {0}")]
    Setup(&'static str),
    #[error("invalid reduced-SWIRL statement: {0}")]
    Statement(&'static str),
    #[error("reduced-SWIRL WARP error: {0:?}")]
    Warp(WarpAccumError),
    #[error("reduced-SWIRL terminal error: {0}")]
    Terminal(String),
    #[error("source provider failed: {0}")]
    SourceProvider(String),
    #[error("source batch {step} returned {actual} sources, expected {expected}")]
    SourceCount {
        step: usize,
        expected: usize,
        actual: usize,
    },
    #[error("source {0} does not match trusted reduced-code setup")]
    SourceBinding(usize),
    #[error("reduced-SWIRL proof mismatch: {0}")]
    ProofMismatch(&'static str),
}

impl From<WarpAccumError> for ReducedSwirlNativeError {
    fn from(error: WarpAccumError) -> Self {
        Self::Warp(error)
    }
}

impl<'a> ReducedSwirlNativeCpuStream<'a> {
    pub fn new(
        setup: &'a ReducedSwirlNativeSetup,
        expected_source_count: usize,
    ) -> Result<Self, ReducedSwirlNativeError> {
        if expected_source_count == 0 {
            return Err(ReducedSwirlNativeError::Statement("empty source list"));
        }
        if expected_source_count > setup.maximum_source_count {
            return Err(ReducedSwirlNativeError::Statement(
                "source count exceeds fixed setup capacity",
            ));
        }
        let step_fresh_counts = setup
            .warp
            .schedule()
            .step_fresh_counts(expected_source_count);
        if step_fresh_counts.is_empty() {
            return Err(ReducedSwirlNativeError::Statement("empty WARP schedule"));
        }
        let max_pending = step_fresh_counts.iter().copied().max().unwrap_or(0);
        let mut challenger = NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
        observe_statement_header(
            setup,
            REDUCED_SWIRL_NATIVE_PROTOCOL_VERSION,
            expected_source_count,
            &mut challenger,
        )?;
        Ok(Self {
            setup,
            expected_source_count,
            step_fresh_counts: step_fresh_counts.clone(),
            challenger,
            pending_sources: Vec::with_capacity(max_pending),
            pending_bindings: Vec::with_capacity(max_pending),
            source_bindings: Vec::with_capacity(expected_source_count),
            authoritative_claims: Vec::with_capacity(expected_source_count),
            accumulator: None,
            steps: Vec::with_capacity(step_fresh_counts.len()),
            transition_records: Vec::with_capacity(step_fresh_counts.len()),
        })
    }

    #[must_use]
    pub fn expected_next_source_index(&self) -> usize {
        self.source_bindings.len()
    }

    #[must_use]
    pub fn pending_source_count(&self) -> usize {
        self.pending_sources.len()
    }

    pub fn push_source(
        &mut self,
        source_binding: Digest,
        source: ReducedSwirlCpuSource,
    ) -> Result<(), ReducedSwirlNativeError> {
        let source_index = self.source_bindings.len();
        if source_index >= self.expected_source_count {
            return Err(ReducedSwirlNativeError::SourceCount {
                step: self.steps.len(),
                expected: self.expected_source_count,
                actual: source_index + 1,
            });
        }
        if is_zero_digest(&source_binding) {
            return Err(ReducedSwirlNativeError::Statement(
                "unset source manifest digest",
            ));
        }
        validate_source(self.setup, &source, source_index)?;
        self.authoritative_claims.push(source.claim_ref().clone());
        self.source_bindings.push(source_binding);
        self.pending_bindings.push(source_binding);
        self.pending_sources.push(source);

        let expected = *self.step_fresh_counts.get(self.steps.len()).ok_or(
            ReducedSwirlNativeError::ProofMismatch("too many WARP transitions"),
        )?;
        if self.pending_sources.len() > expected {
            return Err(ReducedSwirlNativeError::SourceCount {
                step: self.steps.len(),
                expected,
                actual: self.pending_sources.len(),
            });
        }
        if self.pending_sources.len() == expected {
            self.flush_pending_step()?;
        }
        Ok(())
    }

    fn flush_pending_step(&mut self) -> Result<(), ReducedSwirlNativeError> {
        let step_index = self.steps.len();
        let fresh_count = *self.step_fresh_counts.get(step_index).ok_or(
            ReducedSwirlNativeError::ProofMismatch("too many WARP transitions"),
        )?;
        if self.pending_sources.len() != fresh_count || self.pending_bindings.len() != fresh_count {
            return Err(ReducedSwirlNativeError::SourceCount {
                step: step_index,
                expected: fresh_count,
                actual: self.pending_sources.len(),
            });
        }
        let source_start = self
            .source_bindings
            .len()
            .checked_sub(fresh_count)
            .ok_or(ReducedSwirlNativeError::ProofMismatch("source range"))?;
        observe_source_binding_batch(
            step_index,
            source_start,
            &self.pending_bindings,
            &mut self.challenger,
        )?;
        let fresh_openings = CpuReducedSwirlOpeningAdapter::new(StackedRsOpeningBackend::new(
            self.setup.code.hasher().clone(),
        ));
        let accumulator_openings = MerkleOpeningBackend::new(self.setup.code.hasher().clone());
        let sources = core::mem::take(&mut self.pending_sources);
        self.pending_bindings.clear();
        let (next, proof, mut record) = self
            .setup
            .warp
            .prove_reduced_constrained_code_step_recorded::<F, EF, _, _, _, _, _, _>(
                &self.setup.relation,
                &self.setup.binding,
                step_index,
                sources,
                self.accumulator.take(),
                &mut self.challenger,
                &fresh_openings,
                &accumulator_openings,
                &(),
            )?;
        if let Some(boundary) = finish_warp_call(&mut self.challenger, step_index) {
            record.transcript_phases.push(boundary);
        }
        self.accumulator = Some(next);
        self.steps.push(proof);
        self.transition_records.push(record);
        Ok(())
    }

    pub fn finish(mut self) -> Result<ReducedSwirlNativeProverOutput, ReducedSwirlNativeError> {
        if !self.pending_sources.is_empty()
            || self.source_bindings.len() != self.expected_source_count
            || self.steps.len() != self.step_fresh_counts.len()
        {
            return Err(ReducedSwirlNativeError::SourceCount {
                step: self.steps.len(),
                expected: self.expected_source_count,
                actual: self.source_bindings.len(),
            });
        }
        let block_manifest_digest = reduced_swirl_manifest_digest(&self.source_bindings)?;
        let statement = ReducedSwirlNativeStatement {
            protocol_version: REDUCED_SWIRL_NATIVE_PROTOCOL_VERSION,
            block_manifest_digest,
            source_bindings: self.source_bindings,
        };
        statement.validate()?;
        observe_manifest_footer(&block_manifest_digest, &mut self.challenger);
        let mut accumulator =
            self.accumulator
                .take()
                .ok_or(ReducedSwirlNativeError::ProofMismatch(
                    "missing final accumulator",
                ))?;
        let final_instance = accumulator.instance.clone();
        let vacc = ReducedWarpVaccRootProof {
            params: self.setup.warp.params(),
            schedule: self.setup.warp.schedule(),
            steps: self.steps,
            final_instance: final_instance.clone(),
        };
        let descriptor = TerminalDescriptor::from_whir_initial_rs(
            final_instance.rt,
            &self.setup.code,
            &self.setup.whir,
            <EF as openvm_stark_backend::p3_field::BasedVectorSpace<F>>::DIMENSION,
        );
        let terminal_statement = derive_swirl_constrained_rs_terminal_statement(
            &self.setup.relation,
            &self.setup.code,
            &final_instance,
        )
        .map_err(|error| ReducedSwirlNativeError::Terminal(format!("{error:?}")))?;
        let mut transcript = self.challenger.into_inner();
        let same_root_whir =
            prove_terminal_constrained_whir_owned::<SC, FieldElementDigestObserver, _>(
                &self.setup.whir,
                &self.setup.code,
                &descriptor,
                &mut accumulator,
                &terminal_statement,
                &mut transcript,
                None,
                None,
                None,
            )
            .map_err(|error| ReducedSwirlNativeError::Terminal(format!("{error:?}")))?;
        let swirl_power_batch_security =
            ReducedSwirlPowerBatchSecurityBudget::derive(&self.authoritative_claims)?;
        Ok(ReducedSwirlNativeProverOutput {
            proof: ReducedSwirlNativeProof {
                statement,
                vacc,
                terminal: ReducedSwirlNativeTerminalProof {
                    descriptor,
                    same_root_whir,
                },
            },
            authoritative_claims: self.authoritative_claims,
            transition_records: self.transition_records,
            security: self.setup.security,
            swirl_power_batch_security,
        })
    }
}

/// Create the independent per-source transcript used for SWIRL Protocol
/// 3.7.1 column batching. The source adapter subsequently absorbs the exact
/// retained roots, layouts, terminal point, and opening values before it
/// samples `theta`.
///
/// A separate transcript per source avoids accidental challenge dependence on
/// host batching or CUDA scheduling. Cross-block, key, ordering, and source
/// replay are bound by the outer native-WARP statement; this sub-transcript
/// binds only the exact constrained-code projection it randomizes.
#[must_use]
pub fn reduced_swirl_source_challenger() -> NativeWarpChallenger<SC, DuplexSpongeRecorder> {
    let mut challenger = NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
    observe_bytes(REDUCED_SWIRL_NATIVE_THETA_SEED_TAG, &mut challenger);
    observe_u64(
        u64::from(REDUCED_SWIRL_NATIVE_PROTOCOL_VERSION),
        &mut challenger,
    );
    challenger
}

/// Prove a setup-fixed linear WARP schedule while retaining at most one fresh
/// invocation batch. The supplier may build each source from a separately
/// authenticated deferred SWIRL transcript; this global WARP transcript binds
/// their ordered manifest and complete constrained-code claims.
pub fn prove_reduced_swirl_native_cpu_streaming<Supply, SupplyError>(
    setup: &ReducedSwirlNativeSetup,
    statement: ReducedSwirlNativeStatement,
    mut supply: Supply,
) -> Result<ReducedSwirlNativeProverOutput, ReducedSwirlNativeError>
where
    Supply: FnMut(Range<usize>) -> Result<Vec<ReducedSwirlCpuSource>, SupplyError>,
    SupplyError: Display,
{
    statement.validate()?;
    let source_count = statement.source_bindings.len();
    let mut stream = ReducedSwirlNativeCpuStream::new(setup, source_count)?;
    let mut source_offset = 0usize;
    for (step_index, fresh_count) in stream.step_fresh_counts.clone().into_iter().enumerate() {
        let end = source_offset
            .checked_add(fresh_count)
            .ok_or(ReducedSwirlNativeError::Statement("source range"))?;
        let sources = supply(source_offset..end)
            .map_err(|error| ReducedSwirlNativeError::SourceProvider(error.to_string()))?;
        if sources.len() != fresh_count {
            return Err(ReducedSwirlNativeError::SourceCount {
                step: step_index,
                expected: fresh_count,
                actual: sources.len(),
            });
        }
        for (slot, source) in sources.into_iter().enumerate() {
            stream.push_source(statement.source_bindings[source_offset + slot], source)?;
        }
        source_offset = end;
    }
    if source_offset != source_count {
        return Err(ReducedSwirlNativeError::ProofMismatch("source coverage"));
    }
    let output = stream.finish()?;
    if output.proof.statement != statement {
        return Err(ReducedSwirlNativeError::ProofMismatch(
            "streamed public statement",
        ));
    }
    Ok(output)
}

/// Verify the ordered WARP chain and same-root terminal WHIR against claims
/// derived by the checked SWIRL prefixes. Proof-carried claims never replace
/// `expected_claims`.
pub fn verify_reduced_swirl_native_recorded(
    setup: &ReducedSwirlNativeSetup,
    expected_statement: &ReducedSwirlNativeStatement,
    expected_claims: &[ReducedConstrainedCodeClaim<EF, ReducedSwirlFreshCommitment>],
    proof: &ReducedSwirlNativeProof,
) -> Result<ReducedSwirlNativeVerification, ReducedSwirlNativeError> {
    expected_statement.validate()?;
    if &proof.statement != expected_statement {
        return Err(ReducedSwirlNativeError::ProofMismatch("public statement"));
    }
    if expected_claims.len() != expected_statement.source_bindings.len() {
        return Err(ReducedSwirlNativeError::ProofMismatch(
            "authoritative source count",
        ));
    }
    if expected_claims.len() > setup.maximum_source_count {
        return Err(ReducedSwirlNativeError::ProofMismatch(
            "source count exceeds fixed setup capacity",
        ));
    }
    let swirl_power_batch_security = ReducedSwirlPowerBatchSecurityBudget::derive(expected_claims)?;
    if proof.vacc.params != setup.warp.params() || proof.vacc.schedule != setup.warp.schedule() {
        return Err(ReducedSwirlNativeError::ProofMismatch("trusted WARP setup"));
    }
    let expected_counts = setup
        .warp
        .schedule()
        .step_fresh_counts(expected_claims.len());
    if proof.vacc.steps.len() != expected_counts.len()
        || proof
            .vacc
            .steps
            .iter()
            .zip(&expected_counts)
            .any(|(step, &count)| step.fresh_count() != count)
    {
        return Err(ReducedSwirlNativeError::ProofMismatch(
            "WARP schedule shape",
        ));
    }
    for (index, claim) in expected_claims.iter().enumerate() {
        validate_claim(setup, claim, index)?;
    }
    let mut challenger = NativeWarpChallenger::<SC, _>::new(default_duplex_sponge_recorder());
    observe_statement_header(
        setup,
        expected_statement.protocol_version,
        expected_statement.source_bindings.len(),
        &mut challenger,
    )?;
    let fresh_verifier = StackedRsOpeningBackend::new_recording(setup.code.hasher().clone());
    let acc_verifier = MerkleOpeningBackend::new_recording(setup.code.hasher().clone());
    let mut prior = None;
    let mut records = Vec::with_capacity(proof.vacc.steps.len());
    let mut offset = 0usize;
    for (step_index, (step, &count)) in proof.vacc.steps.iter().zip(&expected_counts).enumerate() {
        let end = offset + count;
        observe_source_binding_batch(
            step_index,
            offset,
            &expected_statement.source_bindings[offset..end],
            &mut challenger,
        )?;
        let mut record = setup
            .warp
            .verify_reduced_constrained_code_step_recorded::<F, EF, _, _, _, _>(
                &setup.relation,
                &setup.binding,
                step_index,
                prior.as_ref(),
                &expected_claims[offset..end],
                step,
                &mut challenger,
                &fresh_verifier,
                &acc_verifier,
            )?;
        if let Some(boundary) = finish_warp_call(&mut challenger, step_index) {
            record.transcript_phases.push(boundary);
        }
        prior = Some(record.output_instance.clone());
        records.push(record);
        offset = end;
    }
    let final_instance = prior.ok_or(ReducedSwirlNativeError::ProofMismatch(
        "missing final accumulator",
    ))?;
    if final_instance != proof.vacc.final_instance {
        return Err(ReducedSwirlNativeError::ProofMismatch(
            "final accumulator instance",
        ));
    }
    observe_manifest_footer(&expected_statement.block_manifest_digest, &mut challenger);
    let expected_descriptor = TerminalDescriptor::from_whir_initial_rs(
        final_instance.rt,
        &setup.code,
        &setup.whir,
        <EF as openvm_stark_backend::p3_field::BasedVectorSpace<F>>::DIMENSION,
    );
    if proof.terminal.descriptor != expected_descriptor {
        return Err(ReducedSwirlNativeError::ProofMismatch(
            "terminal descriptor",
        ));
    }
    let terminal_statement = derive_swirl_constrained_rs_terminal_statement(
        &setup.relation,
        &setup.code,
        &final_instance,
    )
    .map_err(|error| ReducedSwirlNativeError::Terminal(format!("{error:?}")))?;
    let mut transcript = challenger.into_inner();
    let terminal = verify_terminal_constrained_whir_recorded::<SC, FieldElementDigestObserver, _>(
        &setup.whir,
        &setup.code,
        &expected_descriptor,
        &final_instance,
        &terminal_statement,
        &proof.terminal.same_root_whir,
        &mut transcript,
    )
    .map_err(|error| ReducedSwirlNativeError::Terminal(format!("{error:?}")))?;
    if terminal.root != final_instance.rt {
        return Err(ReducedSwirlNativeError::ProofMismatch(
            "terminal same-root endpoint",
        ));
    }
    Ok(ReducedSwirlNativeVerification {
        final_instance,
        transition_records: records,
        terminal,
        complete_transcript: transcript,
        swirl_power_batch_security,
    })
}

fn validate_source(
    setup: &ReducedSwirlNativeSetup,
    source: &ReducedSwirlCpuSource,
    index: usize,
) -> Result<(), ReducedSwirlNativeError> {
    validate_claim(setup, source.claim_ref(), index)?;
    if source.claim_ref().commitment != *source.descriptor() {
        return Err(ReducedSwirlNativeError::SourceBinding(index));
    }
    Ok(())
}

fn validate_claim(
    setup: &ReducedSwirlNativeSetup,
    claim: &ReducedConstrainedCodeClaim<EF, ReducedSwirlFreshCommitment>,
    index: usize,
) -> Result<(), ReducedSwirlNativeError> {
    if claim.binding != setup.binding
        || claim.commitment.log_message_len != setup.code.log_message_len()
        || claim.commitment.log_codeword_len != setup.code.log_codeword_len()
        || claim.commitment.rows_per_query != setup.code.rows_per_query()
        || claim.alpha.len() != setup.code.log_codeword_len()
        || claim.alpha.iter().any(|value| *value != EF::ZERO)
    {
        return Err(ReducedSwirlNativeError::SourceBinding(index));
    }
    <SwirlConstrainedRsRelation<EF> as ReducedConstrainedCodeRelation<F, EF>>::validate_public_claim(
        &setup.relation,
        &claim.beta,
        claim.eta,
    )
        .map_err(|_| ReducedSwirlNativeError::SourceBinding(index))?;
    Ok(())
}

fn observe_statement_header<Ch>(
    setup: &ReducedSwirlNativeSetup,
    protocol_version: u32,
    source_count: usize,
    challenger: &mut Ch,
) -> Result<(), ReducedSwirlNativeError>
where
    Ch: AlgebraicChallenger<EF>,
{
    observe_bytes(REDUCED_SWIRL_NATIVE_TRANSCRIPT_TAG, challenger);
    observe_u64(u64::from(protocol_version), challenger);
    observe_u64(source_count as u64, challenger);
    for value in [
        setup.code.log_message_len(),
        setup.code.log_codeword_len(),
        setup.code.rows_per_query(),
        setup.warp.schedule().arity,
        setup.warp.params().num_ood,
        setup.warp.params().num_shift_queries,
        setup.warp.params().batching_arity(),
        setup.family_target_bits,
    ] {
        observe_u64(value as u64, challenger);
    }
    observe_bytes(&setup.binding.source_domain, challenger);
    challenger.observe_slice(&setup.binding.relation_binding);
    challenger.observe_slice(&setup.binding.code_binding);
    Ok(())
}

/// Canonical digest of the complete ordered source-entry list. The native
/// stream derives this only after the active prefix is complete, then absorbs
/// it before terminal Decide. Every entry was already absorbed before its
/// corresponding VACC call, so this footer summarizes rather than replaces
/// the per-step ordering binding.
pub fn reduced_swirl_manifest_digest(
    source_bindings: &[Digest],
) -> Result<Digest, ReducedSwirlNativeError> {
    if source_bindings.is_empty() || source_bindings.iter().any(is_zero_digest) {
        return Err(ReducedSwirlNativeError::Statement(
            "invalid source manifest entries",
        ));
    }
    let mut transcript = default_duplex_sponge_recorder();
    for &byte in REDUCED_SWIRL_NATIVE_MANIFEST_TAG {
        <_ as FiatShamirTranscript<SC>>::observe(&mut transcript, F::from_u8(byte));
    }
    <_ as FiatShamirTranscript<SC>>::observe(
        &mut transcript,
        F::from_u32(REDUCED_SWIRL_NATIVE_PROTOCOL_VERSION),
    );
    <_ as FiatShamirTranscript<SC>>::observe(
        &mut transcript,
        F::from_u32(
            u32::try_from(source_bindings.len())
                .map_err(|_| ReducedSwirlNativeError::Statement("source manifest count"))?,
        ),
    );
    for (source_index, digest) in source_bindings.iter().enumerate() {
        <_ as FiatShamirTranscript<SC>>::observe(
            &mut transcript,
            F::from_u32(
                u32::try_from(source_index)
                    .map_err(|_| ReducedSwirlNativeError::Statement("source manifest index"))?,
            ),
        );
        for &limb in digest {
            <_ as FiatShamirTranscript<SC>>::observe(&mut transcript, limb);
        }
    }
    Ok(core::array::from_fn(|_| {
        <_ as FiatShamirTranscript<SC>>::sample(&mut transcript)
    }))
}

fn observe_manifest_footer<Ch>(manifest_digest: &Digest, challenger: &mut Ch)
where
    Ch: AlgebraicChallenger<EF>,
{
    observe_bytes(REDUCED_SWIRL_NATIVE_MANIFEST_FOOTER_TAG, challenger);
    observe_digest(manifest_digest, challenger);
}

fn observe_source_binding_batch<Ch>(
    step_index: usize,
    source_start: usize,
    source_bindings: &[Digest],
    challenger: &mut Ch,
) -> Result<(), ReducedSwirlNativeError>
where
    Ch: AlgebraicChallenger<EF>,
{
    if source_bindings.is_empty() || source_bindings.iter().any(is_zero_digest) {
        return Err(ReducedSwirlNativeError::Statement(
            "invalid source binding batch",
        ));
    }
    observe_bytes(REDUCED_SWIRL_NATIVE_SOURCE_BATCH_TAG, challenger);
    observe_u64(step_index as u64, challenger);
    observe_u64(source_start as u64, challenger);
    observe_u64(source_bindings.len() as u64, challenger);
    for digest in source_bindings {
        observe_digest(digest, challenger);
    }
    Ok(())
}

fn is_zero_digest(digest: &Digest) -> bool {
    digest.iter().all(|value| *value == F::ZERO)
}

fn observe_digest<Ch>(digest: &Digest, challenger: &mut Ch)
where
    Ch: AlgebraicChallenger<EF>,
{
    for &value in digest {
        challenger.observe(EF::from(value));
    }
}

fn observe_bytes<Ch>(bytes: &[u8], challenger: &mut Ch)
where
    Ch: AlgebraicChallenger<EF>,
{
    observe_u64(bytes.len() as u64, challenger);
    for &byte in bytes {
        challenger.observe(EF::from_u8(byte));
    }
}

fn observe_u64<Ch>(value: u64, challenger: &mut Ch)
where
    Ch: AlgebraicChallenger<EF>,
{
    for byte in value.to_le_bytes() {
        challenger.observe(EF::from_u8(byte));
    }
}

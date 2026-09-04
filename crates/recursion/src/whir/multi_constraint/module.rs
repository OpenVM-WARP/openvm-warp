//! Complete recursive owner for a generalized, multi-constraint WHIR proof.
//!
//! This module deliberately reuses the ordinary WHIR round, query, opened
//! value, Merkle-bus, folding, and final-query AIRs.  It replaces only the two
//! pieces whose algebra changes for several opening constraints:
//!
//! - the initial scalar target and sumcheck weight; and
//! - the final-polynomial MLE evaluation.
//!
//! Transcript and Merkle-path traces remain owned by the ordinary recursion
//! system.  The public carrier contains a genuine [`WhirProof`], retained
//! initial setup commitments, the N opening constraints, and its transcript
//! replay.  [`MultiConstraintWhirModule::prepare_direct_carriers`] creates the
//! private compatibility records required by the existing trace generators;
//! fields outside WHIR are inert placeholders and are not verification input.
//! The typed completion bus is emitted only after the generalized final-MLE,
//! ordinary query, Merkle, and certified transcript-checkpoint obligations.

use std::sync::Arc;

use itertools::Itertools;
use openvm_circuit_primitives::encoder::Encoder;
use openvm_cpu_backend::CpuBackend;
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_stark_backend::{
    keygen::types::MultiStarkVerifyingKey,
    proof::{BatchConstraintProof, GkrProof, Proof, StackingProof, WhirProof},
    prover::AirProvingContext,
    transcript::TranscriptLog,
    AirRef, StarkProtocolConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{BabyBearPoseidon2Config, CHUNK, D_EF, EF, F};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::{
    air::{
        MultiConstraintCompletionAir, MultiConstraintFinalAggregateAir,
        MultiConstraintFinalPolyMleAir, MultiConstraintInitialCommitmentAir,
        MultiConstraintInitialCommitmentBus, MultiConstraintInitialTargetAir,
        MultiConstraintPrefixAir, MultiConstraintStatementBuses, MultiConstraintSumcheckAir,
        MultiConstraintWeightAir, MultiConstraintWeightBuses,
    },
    derive_multi_constraint_whir_data,
    trace::{
        generate_multi_constraint_completion_trace,
        generate_multi_constraint_final_aggregate_trace,
        generate_multi_constraint_final_poly_trace,
        generate_multi_constraint_initial_commitment_trace, generate_multi_constraint_prefix_trace,
        generate_multi_constraint_sumcheck_trace, generate_multi_constraint_target_trace,
        generate_multi_constraint_weight_trace, MultiConstraintSumcheckTraceInput,
        MultiConstraintWhirTerminalCheckpoint,
    },
    validate_batching_coefficients_against_gamma, validate_whir_proof_shape,
    MultiConstraintWhirError, MultiConstraintWhirInitialCommitment, MultiConstraintWhirProfile,
    MultiConstraintWhirStatement, MultiConstraintWhirTranscriptPreflight,
};
pub use crate::whir::bus::{
    WhirCompletionBus as MultiConstraintWhirCompletionBus,
    WhirCompletionMessage as MultiConstraintWhirCompletionMessage,
};
use crate::{
    bus::CertifiedTranscriptCheckpointBus,
    primitives::exp_bits_len::ExpBitsLenCpuTraceGenerator,
    system::{
        AirModule, BusIndexManager, BusInventory, Preflight, StackingPreflight, WhirPreflight,
    },
    tracegen::{RowMajorChip, StandardTracegenCtx},
    utils::{poseidon2_hash_slice_with_states, pow_tidx_count},
    whir::{
        bus::{
            FinalPolyMleEvalBus, FinalPolyQueryEvalBus, VerifyQueriesBus, VerifyQueryBus,
            WhirAlphaBus, WhirCompletionBus, WhirFinalPolyBus, WhirFoldingBus, WhirGammaBus,
            WhirQueryBus, WhirSumcheckBus, WhirTerminalBus,
        },
        final_poly_query_eval::FinalPolyQueryEvalAir,
        folding::WhirFoldingAir,
        initial_opened_values::InitialOpenedValuesAir,
        non_initial_opened_values::NonInitialOpenedValuesAir,
        query::WhirQueryAir,
        whir_round::{WhirRoundAir, WhirRoundTraceGenerator},
        WhirBlobCpu, WhirModule, WhirModuleChip,
    },
};

/// Direct setup-authority input.  No complete child STARK proof is required.
#[derive(Clone, Copy)]
pub struct MultiConstraintWhirDirectCarrier<'a> {
    pub whir_proof: &'a WhirProof<BabyBearPoseidon2Config>,
    pub initial_commitments: &'a [MultiConstraintWhirInitialCommitment],
    pub statement: &'a MultiConstraintWhirStatement,
    pub multi_preflight: &'a MultiConstraintWhirTranscriptPreflight,
    /// Exact native replay consumed by the shared TranscriptModule.
    pub transcript: &'a TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    /// Exact row-aligned endpoint selected by the shared TranscriptModule.
    /// The completion AIR receives the same tuple from its certified
    /// checkpoint bus, so these host values are witnesses rather than trust.
    pub terminal_checkpoint: MultiConstraintWhirTerminalCheckpoint,
}

/// Owned direct carrier plus the private records expected by ordinary WHIR
/// trace generators.
///
/// Only `proof_adapter.whir_proof`, `preflight.whir`, `preflight.stacking.mu`,
/// transcript data, and Merkle hash states are consumed by the reused owners.
/// The remaining fields are canonical empty placeholders and no AIR in this
/// module reads or authenticates them.
#[derive(Clone)]
pub struct PreparedMultiConstraintWhirCarrier {
    proof_adapter: Proof<BabyBearPoseidon2Config>,
    preflight: Preflight,
    initial_commitments: Vec<MultiConstraintWhirInitialCommitment>,
    statement: MultiConstraintWhirStatement,
    multi_preflight: MultiConstraintWhirTranscriptPreflight,
    terminal_checkpoint: MultiConstraintWhirTerminalCheckpoint,
}

impl PreparedMultiConstraintWhirCarrier {
    /// Compatibility record for the ordinary Transcript/Merkle trace owner.
    /// Its non-WHIR fields are intentionally empty and must not be connected
    /// to GKR, batch-constraint, stacking, or proof-shape AIRs.
    #[must_use]
    pub fn transcript_proof_adapter(&self) -> &Proof<BabyBearPoseidon2Config> {
        &self.proof_adapter
    }

    /// Replay and Merkle hash records paired with
    /// [`Self::transcript_proof_adapter`].
    #[must_use]
    pub fn transcript_preflight(&self) -> &Preflight {
        &self.preflight
    }

    #[must_use]
    pub fn whir_proof(&self) -> &WhirProof<BabyBearPoseidon2Config> {
        &self.proof_adapter.whir_proof
    }

    #[must_use]
    pub fn initial_commitments(&self) -> &[MultiConstraintWhirInitialCommitment] {
        &self.initial_commitments
    }
}

/// Stable AIR order returned by [`MultiConstraintWhirModule::airs`] and
/// [`MultiConstraintWhirModule::generate_air_contexts`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum MultiConstraintWhirAir {
    WhirRound = 0,
    Query = 1,
    InitialOpenedValues = 2,
    NonInitialOpenedValues = 3,
    Folding = 4,
    FinalPolyQueryEval = 5,
    InitialCommitments = 6,
    Prefix = 7,
    InitialTarget = 8,
    Sumcheck = 9,
    Weight = 10,
    FinalPolynomial = 11,
    FinalAggregate = 12,
    Completion = 13,
}

impl MultiConstraintWhirAir {
    pub const COUNT: usize = 14;

    #[inline]
    const fn index(self) -> usize {
        self as usize
    }
}

/// Complete AIR owner for one fixed multi-opening profile.
pub struct MultiConstraintWhirModule {
    params: openvm_stark_backend::SystemParams,
    profile: MultiConstraintWhirProfile,
    bus_inventory: BusInventory,
    statement_buses: MultiConstraintStatementBuses,
    initial_commitment_bus: MultiConstraintInitialCommitmentBus,

    sumcheck_bus: WhirSumcheckBus,
    verify_queries_bus: VerifyQueriesBus,
    verify_query_bus: VerifyQueryBus,
    folding_bus: WhirFoldingBus,
    final_poly_mle_eval_bus: FinalPolyMleEvalBus,
    final_poly_query_eval_bus: FinalPolyQueryEvalBus,
    alpha_bus: WhirAlphaBus,
    gamma_bus: WhirGammaBus,
    query_bus: WhirQueryBus,
    final_poly_bus: WhirFinalPolyBus,
    weight_buses: MultiConstraintWeightBuses,
    terminal_bus: WhirTerminalBus,
    completion_bus: WhirCompletionBus,
    terminal_checkpoint_bus: CertifiedTranscriptCheckpointBus,
    terminal_checkpoint_kind: usize,
    class_index: usize,
    /// Setup-fixed adapter mode for an initial `H union gH` codeword.  The
    /// caller must also provide the matching externally owned query trace and
    /// query AIR; ordinary multi-constraint WHIR remains unchanged.
    coefficient_two_coset_external_query: bool,
}

impl MultiConstraintWhirModule {
    /// Construct a fixed-profile module.  Statement buses are caller-owned;
    /// every internal bus and the typed completion bus are allocated here.
    pub fn new(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        profile: MultiConstraintWhirProfile,
        statement_buses: MultiConstraintStatementBuses,
        initial_commitment_bus: MultiConstraintInitialCommitmentBus,
        terminal_checkpoint_bus: CertifiedTranscriptCheckpointBus,
        terminal_checkpoint_kind: usize,
        class_index: usize,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
    ) -> Result<Self, MultiConstraintWhirModuleError> {
        Self::new_inner(
            child_vk,
            profile,
            statement_buses,
            initial_commitment_bus,
            terminal_checkpoint_bus,
            terminal_checkpoint_kind,
            class_index,
            b,
            bus_inventory,
            false,
        )
    }

    /// Construct the shared portion of a multi-constraint verifier whose
    /// initial codeword is the coefficient-RS interleaving `H union gH`.
    ///
    /// The initial query map needs extra parity/quotient witness columns, so
    /// the enclosing module must replace the ordinary query AIR and call
    /// [`Self::generate_air_contexts_with_external_two_coset_query`].  Keeping
    /// this entry point explicit prevents a log-sized heuristic from silently
    /// changing ordinary WHIR semantics.
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_external_coefficient_two_coset_query(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        profile: MultiConstraintWhirProfile,
        statement_buses: MultiConstraintStatementBuses,
        initial_commitment_bus: MultiConstraintInitialCommitmentBus,
        terminal_checkpoint_bus: CertifiedTranscriptCheckpointBus,
        terminal_checkpoint_kind: usize,
        class_index: usize,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
    ) -> Result<Self, MultiConstraintWhirModuleError> {
        Self::new_inner(
            child_vk,
            profile,
            statement_buses,
            initial_commitment_bus,
            terminal_checkpoint_bus,
            terminal_checkpoint_kind,
            class_index,
            b,
            bus_inventory,
            true,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn new_inner(
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        profile: MultiConstraintWhirProfile,
        statement_buses: MultiConstraintStatementBuses,
        initial_commitment_bus: MultiConstraintInitialCommitmentBus,
        terminal_checkpoint_bus: CertifiedTranscriptCheckpointBus,
        terminal_checkpoint_kind: usize,
        class_index: usize,
        b: &mut BusIndexManager,
        bus_inventory: BusInventory,
        coefficient_two_coset_external_query: bool,
    ) -> Result<Self, MultiConstraintWhirModuleError> {
        profile.validate()?;
        let params = child_vk.inner.params.clone();
        if coefficient_two_coset_external_query
            && (params.log_blowup != 1
                || params.log_stacked_height() > F::TWO_ADICITY
                || params.num_whir_rounds() < 2)
        {
            return Err(MultiConstraintWhirModuleError::InvalidTwoCosetProfile);
        }
        let expected_dimension = params
            .num_whir_sumcheck_rounds()
            .checked_add(params.log_final_poly_len())
            .ok_or(MultiConstraintWhirModuleError::TraceSizeOverflow)?;
        if profile.point_dimension != expected_dimension {
            return Err(MultiConstraintWhirModuleError::PointDimension {
                actual: profile.point_dimension,
                expected: expected_dimension,
            });
        }

        let sumcheck_bus = WhirSumcheckBus::new(b.new_bus_idx());
        let alpha_bus = WhirAlphaBus::new(b.new_bus_idx());
        let gamma_bus = WhirGammaBus::new(b.new_bus_idx());
        let query_bus = WhirQueryBus::new(b.new_bus_idx());
        let verify_queries_bus = VerifyQueriesBus::new(b.new_bus_idx());
        let verify_query_bus = VerifyQueryBus::new(b.new_bus_idx());
        let folding_bus = WhirFoldingBus::new(b.new_bus_idx());
        let final_poly_mle_eval_bus = FinalPolyMleEvalBus::new(b.new_bus_idx());
        let final_poly_query_eval_bus = FinalPolyQueryEvalBus::new(b.new_bus_idx());
        let final_poly_bus = WhirFinalPolyBus::new(b.new_bus_idx());
        let weight_buses = MultiConstraintWeightBuses::new([
            b.new_bus_idx(),
            b.new_bus_idx(),
            b.new_bus_idx(),
            b.new_bus_idx(),
            b.new_bus_idx(),
            b.new_bus_idx(),
        ]);
        let terminal_bus = WhirTerminalBus::new(b.new_bus_idx());
        let completion_bus = WhirCompletionBus::new(b.new_bus_idx());

        Ok(Self {
            params,
            profile,
            bus_inventory,
            statement_buses,
            initial_commitment_bus,
            sumcheck_bus,
            verify_queries_bus,
            verify_query_bus,
            folding_bus,
            final_poly_mle_eval_bus,
            final_poly_query_eval_bus,
            alpha_bus,
            gamma_bus,
            query_bus,
            final_poly_bus,
            weight_buses,
            terminal_bus,
            completion_bus,
            terminal_checkpoint_bus,
            terminal_checkpoint_kind,
            class_index,
            coefficient_two_coset_external_query,
        })
    }

    #[must_use]
    pub const fn statement_buses(&self) -> MultiConstraintStatementBuses {
        self.statement_buses
    }

    /// Caller-owned endpoint for retained setup roots and widths.
    #[must_use]
    pub const fn initial_commitment_bus(&self) -> MultiConstraintInitialCommitmentBus {
        self.initial_commitment_bus
    }

    #[must_use]
    pub const fn completion_bus(&self) -> WhirCompletionBus {
        self.completion_bus
    }

    #[must_use]
    pub fn profile(&self) -> &MultiConstraintWhirProfile {
        &self.profile
    }

    fn validate_direct_carrier(
        &self,
        carrier: MultiConstraintWhirDirectCarrier<'_>,
    ) -> Result<(), MultiConstraintWhirModuleError> {
        carrier.statement.validate(&self.profile)?;
        validate_batching_coefficients_against_gamma(
            carrier.statement,
            carrier.multi_preflight.batching_gamma,
        )?;
        validate_whir_proof_shape(&self.params, carrier.whir_proof)?;

        if carrier.initial_commitments.len() != self.profile.commitment_widths.len() {
            return Err(MultiConstraintWhirModuleError::InitialCommitmentCount {
                actual: carrier.initial_commitments.len(),
                expected: self.profile.commitment_widths.len(),
            });
        }
        for (commitment, (&expected_width, retained)) in self
            .profile
            .commitment_widths
            .iter()
            .zip(carrier.initial_commitments)
            .enumerate()
        {
            if retained.width != expected_width {
                return Err(MultiConstraintWhirModuleError::InitialCommitmentWidth {
                    commitment,
                    actual: retained.width,
                    expected: expected_width,
                });
            }
        }
        self.validate_initial_openings(carrier.whir_proof)?;

        let multi = carrier.multi_preflight;
        let rounds = self.params.num_whir_rounds();
        let sumcheck_rounds = self.params.num_whir_sumcheck_rounds();
        let query_count = self
            .params
            .whir
            .rounds
            .iter()
            .try_fold(0usize, |total, round| total.checked_add(round.num_queries))
            .ok_or(MultiConstraintWhirModuleError::TraceSizeOverflow)?;
        if multi.whir_round_tidx_per_round.len() != rounds
            || multi.query_tidx_per_round.len() != rounds
            || multi.alphas.len() != sumcheck_rounds
            || multi.z0s.len() != rounds - 1
            || multi.gammas.len() != rounds
            || multi.folding_pow_samples.len() != sumcheck_rounds
            || multi.query_pow_samples.len() != rounds
            || multi.queries.len() != query_count
        {
            return Err(MultiConstraintWhirModuleError::PreflightMismatch(
                "multi-WHIR vector shape",
            ));
        }
        let expected_end_tidx = terminal_end_tidx(&self.params, multi)?;
        if carrier.transcript.len() != expected_end_tidx {
            return Err(MultiConstraintWhirModuleError::PreflightMismatch(
                "transcript length",
            ));
        }
        if carrier.terminal_checkpoint.end_tidx != expected_end_tidx
            || carrier.terminal_checkpoint.sample_count == 0
            || carrier.terminal_checkpoint.sample_count > CHUNK
        {
            return Err(MultiConstraintWhirModuleError::PreflightMismatch(
                "terminal transcript checkpoint",
            ));
        }
        Ok(())
    }

    fn validate_initial_openings(
        &self,
        proof: &WhirProof<BabyBearPoseidon2Config>,
    ) -> Result<(), MultiConstraintWhirModuleError> {
        if proof.initial_round_opened_rows.len() != self.profile.commitment_widths.len()
            || proof.initial_round_merkle_proofs.len() != self.profile.commitment_widths.len()
        {
            return Err(MultiConstraintWhirModuleError::PreflightMismatch(
                "initial commitment opening count",
            ));
        }
        let queries = self.params.whir.rounds[0].num_queries;
        let cosets = 1usize
            .checked_shl(self.params.k_whir() as u32)
            .ok_or(MultiConstraintWhirModuleError::TraceSizeOverflow)?;
        for (commitment, (&expected_width, opened_queries)) in self
            .profile
            .commitment_widths
            .iter()
            .zip(&proof.initial_round_opened_rows)
            .enumerate()
        {
            if opened_queries.len() != queries
                || proof.initial_round_merkle_proofs[commitment].len() != queries
                || opened_queries.iter().any(|rows| {
                    rows.len() != cosets || rows.iter().any(|row| row.len() != expected_width)
                })
            {
                return Err(MultiConstraintWhirModuleError::InitialOpeningShape { commitment });
            }
        }
        Ok(())
    }

    /// Validate direct authority inputs and build the records shared by this
    /// module and the ordinary Transcript/Merkle owner.
    pub fn prepare_direct_carriers(
        &self,
        carriers: &[MultiConstraintWhirDirectCarrier<'_>],
    ) -> Result<Vec<PreparedMultiConstraintWhirCarrier>, MultiConstraintWhirModuleError> {
        if carriers.is_empty() {
            return Err(MultiConstraintWhirModuleError::EmptyBatch);
        }
        carriers
            .iter()
            .copied()
            .map(|carrier| {
                self.validate_direct_carrier(carrier)?;
                Ok(self.prepare_direct_carrier(carrier))
            })
            .collect()
    }

    fn prepare_direct_carrier(
        &self,
        carrier: MultiConstraintWhirDirectCarrier<'_>,
    ) -> PreparedMultiConstraintWhirCarrier {
        // These fields only satisfy the legacy container type.  The module's
        // AIR inventory contains no owner capable of reading them.
        let proof_adapter = Proof {
            common_main_commit: [F::ZERO; CHUNK],
            trace_vdata: Vec::new(),
            public_values: Vec::new(),
            gkr_proof: GkrProof {
                logup_pow_witness: F::ZERO,
                q0_claim: EF::ZERO,
                claims_per_layer: Vec::new(),
                sumcheck_polys: Vec::new(),
            },
            batch_constraint_proof: BatchConstraintProof {
                numerator_term_per_air: Vec::new(),
                denominator_term_per_air: Vec::new(),
                univariate_round_coeffs: Vec::new(),
                sumcheck_round_polys: Vec::new(),
                column_openings: Vec::new(),
            },
            stacking_proof: StackingProof {
                univariate_round_coeffs: Vec::new(),
                sumcheck_round_polys: Vec::new(),
                // Temporary scalar seed consumed only while constructing the
                // shared blob; `install_multi_claims` replaces every claim.
                stacking_openings: self
                    .profile
                    .commitment_widths
                    .iter()
                    .map(|&width| vec![EF::ZERO; width])
                    .collect(),
            },
            whir_proof: carrier.whir_proof.clone(),
        };

        let mut poseidon2_perm_inputs = Vec::new();
        let initial_row_states = carrier
            .whir_proof
            .initial_round_opened_rows
            .iter()
            .map(|commitment| {
                commitment
                    .iter()
                    .map(|query| {
                        query
                            .iter()
                            .map(|row| {
                                let (_, pre_states, post_states) =
                                    poseidon2_hash_slice_with_states(row);
                                poseidon2_perm_inputs.extend(pre_states);
                                post_states
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let mut poseidon2_compress_inputs = Vec::new();
        let codeword_states = carrier
            .whir_proof
            .codeword_opened_values
            .iter()
            .map(|round| {
                round
                    .iter()
                    .map(|query| {
                        query
                            .iter()
                            .map(|value| {
                                let (_, pre_states, states) = poseidon2_hash_slice_with_states(
                                    value.as_basis_coefficients_slice(),
                                );
                                poseidon2_compress_inputs.extend(pre_states);
                                states[0]
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let multi = carrier.multi_preflight;
        let preflight = Preflight {
            transcript: carrier.transcript.clone(),
            stacking: StackingPreflight {
                stacking_batching_challenge: multi.mu,
                mu_pow_witness: carrier.whir_proof.mu_pow_witness,
                mu_pow_sample: multi.mu_pow_sample,
                // Private compatibility coordinates only.  The generalized
                // point AIRs own all actual N opening points.
                sumcheck_rnd: vec![EF::ZERO; self.params.n_stack + 1],
                ..Default::default()
            },
            whir: WhirPreflight {
                whir_round_tidx_per_round: multi.whir_round_tidx_per_round.clone(),
                query_tidx_per_round: multi.query_tidx_per_round.clone(),
                alphas: multi.alphas.clone(),
                z0s: multi.z0s.clone(),
                gammas: multi.gammas.clone(),
                folding_pow_samples: multi.folding_pow_samples.clone(),
                query_pow_samples: multi.query_pow_samples.clone(),
                queries: multi.queries.clone(),
            },
            initial_row_states,
            codeword_states,
            poseidon2_perm_inputs,
            poseidon2_compress_inputs,
            ..Default::default()
        };
        PreparedMultiConstraintWhirCarrier {
            proof_adapter,
            preflight,
            initial_commitments: carrier.initial_commitments.to_vec(),
            statement: carrier.statement.clone(),
            multi_preflight: carrier.multi_preflight.clone(),
            terminal_checkpoint: carrier.terminal_checkpoint,
        }
    }

    /// Generate the complete WHIR contexts in exactly the order returned by
    /// [`AirModule::airs`].  The ordinary Transcript/Merkle module must use
    /// each prepared carrier's paired adapter and preflight records.
    pub fn generate_air_contexts<SC: StarkProtocolConfig<F = F>>(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        instances: &[PreparedMultiConstraintWhirCarrier],
        exp_bits_len_gen: &ExpBitsLenCpuTraceGenerator,
        required_heights: Option<&[usize]>,
    ) -> Result<Vec<AirProvingContext<CpuBackend<SC>>>, MultiConstraintWhirModuleError> {
        if self.coefficient_two_coset_external_query {
            return Err(MultiConstraintWhirModuleError::ExternalQueryTraceRequired);
        }
        self.generate_air_contexts_inner(
            child_vk,
            instances,
            exp_bits_len_gen,
            required_heights,
            None,
        )
    }

    /// Generate contexts for the explicit coefficient-two-coset adapter.
    /// `query_trace` must be generated by the same enclosing owner that
    /// replaces [`MultiConstraintWhirAir::Query`] in the AIR list.
    pub fn generate_air_contexts_with_external_two_coset_query<SC: StarkProtocolConfig<F = F>>(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        instances: &[PreparedMultiConstraintWhirCarrier],
        exp_bits_len_gen: &ExpBitsLenCpuTraceGenerator,
        required_heights: Option<&[usize]>,
        query_trace: RowMajorMatrix<F>,
    ) -> Result<Vec<AirProvingContext<CpuBackend<SC>>>, MultiConstraintWhirModuleError> {
        if !self.coefficient_two_coset_external_query {
            return Err(MultiConstraintWhirModuleError::UnexpectedExternalQueryTrace);
        }
        self.generate_air_contexts_inner(
            child_vk,
            instances,
            exp_bits_len_gen,
            required_heights,
            Some(query_trace),
        )
    }

    fn generate_air_contexts_inner<SC: StarkProtocolConfig<F = F>>(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        instances: &[PreparedMultiConstraintWhirCarrier],
        exp_bits_len_gen: &ExpBitsLenCpuTraceGenerator,
        required_heights: Option<&[usize]>,
        external_query_trace: Option<RowMajorMatrix<F>>,
    ) -> Result<Vec<AirProvingContext<CpuBackend<SC>>>, MultiConstraintWhirModuleError> {
        if instances.is_empty() {
            return Err(MultiConstraintWhirModuleError::EmptyBatch);
        }
        if child_vk.inner.params != self.params {
            return Err(MultiConstraintWhirModuleError::ParameterMismatch);
        }
        if let Some(heights) = required_heights {
            if heights.len() != MultiConstraintWhirAir::COUNT {
                return Err(MultiConstraintWhirModuleError::RequiredHeightCount {
                    actual: heights.len(),
                    expected: MultiConstraintWhirAir::COUNT,
                });
            }
        }
        let proofs = instances
            .iter()
            .map(|input| &input.proof_adapter)
            .collect_vec();
        let preflights = instances.iter().map(|input| &input.preflight).collect_vec();
        let mut blob = if self.coefficient_two_coset_external_query {
            WhirModule::generate_blob_coefficient_two_coset(
                child_vk,
                &proofs,
                &preflights,
                exp_bits_len_gen,
            )
        } else {
            WhirModule::generate_blob(child_vk, &proofs, &preflights, exp_bits_len_gen)
        };
        self.install_multi_claims(instances, &mut blob)?;

        let standard_ctx = (
            StandardTracegenCtx {
                vk: child_vk,
                proofs: &proofs,
                preflights: &preflights,
            },
            &blob,
        );
        let mut external_query_trace = external_query_trace;
        let mut contexts = Vec::with_capacity(MultiConstraintWhirAir::COUNT);
        for (role, chip) in [
            (MultiConstraintWhirAir::WhirRound, WhirModuleChip::WhirRound),
            (MultiConstraintWhirAir::Query, WhirModuleChip::Query),
            (
                MultiConstraintWhirAir::InitialOpenedValues,
                WhirModuleChip::InitialOpenedValues,
            ),
            (
                MultiConstraintWhirAir::NonInitialOpenedValues,
                WhirModuleChip::NonInitialOpenedValues,
            ),
            (MultiConstraintWhirAir::Folding, WhirModuleChip::Folding),
            (
                MultiConstraintWhirAir::FinalPolyQueryEval,
                WhirModuleChip::FinalPolyQueryEval,
            ),
        ] {
            let trace = match role {
                MultiConstraintWhirAir::WhirRound if self.coefficient_two_coset_external_query => {
                    WhirRoundTraceGenerator
                        .generate_coefficient_two_coset_trace(
                            &standard_ctx,
                            requested_height(required_heights, role),
                        )
                        .ok_or(MultiConstraintWhirModuleError::RequiredHeight { role })?
                }
                MultiConstraintWhirAir::Query if self.coefficient_two_coset_external_query => {
                    let trace = external_query_trace
                        .take()
                        .ok_or(MultiConstraintWhirModuleError::ExternalQueryTraceRequired)?;
                    if let Some(expected) = requested_height(required_heights, role) {
                        if trace.height() != expected {
                            return Err(MultiConstraintWhirModuleError::RequiredHeight { role });
                        }
                    }
                    trace
                }
                _ => chip
                    .generate_trace(&standard_ctx, requested_height(required_heights, role))
                    .ok_or(MultiConstraintWhirModuleError::RequiredHeight { role })?,
            };
            contexts.push(AirProvingContext::simple_no_pis(trace));
        }

        let retained_commitments = instances
            .iter()
            .map(|input| input.initial_commitments.clone())
            .collect_vec();
        let initial_commitments = generate_multi_constraint_initial_commitment_trace(
            &retained_commitments,
            requested_height(required_heights, MultiConstraintWhirAir::InitialCommitments),
        )?
        .ok_or(MultiConstraintWhirModuleError::RequiredHeight {
            role: MultiConstraintWhirAir::InitialCommitments,
        })?;
        contexts.push(AirProvingContext::simple_no_pis(initial_commitments));

        let statements = instances.iter().map(|input| &input.statement).collect_vec();
        let multi_preflights = instances
            .iter()
            .map(|input| &input.multi_preflight)
            .collect_vec();
        let whir_proofs = instances
            .iter()
            .map(|input| &input.proof_adapter.whir_proof)
            .collect_vec();
        let derived = instances
            .iter()
            .map(|input| {
                derive_multi_constraint_whir_data(
                    &self.profile,
                    &input.statement,
                    input.multi_preflight.mu,
                    &input.multi_preflight.alphas,
                    &input.proof_adapter.whir_proof.final_poly,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        let prefix = generate_multi_constraint_prefix_trace(
            &multi_preflights.iter().copied().cloned().collect_vec(),
            &statements.iter().copied().cloned().collect_vec(),
            &derived,
            &whir_proofs,
            self.profile.constraint_count,
            requested_height(required_heights, MultiConstraintWhirAir::Prefix),
        )?
        .ok_or(MultiConstraintWhirModuleError::RequiredHeight {
            role: MultiConstraintWhirAir::Prefix,
        })?;
        contexts.push(AirProvingContext::simple_no_pis(prefix));

        let statement_values = statements.iter().copied().cloned().collect_vec();
        let mus = multi_preflights.iter().map(|p| p.mu).collect_vec();
        let target = generate_multi_constraint_target_trace(
            &self.profile,
            &statement_values,
            &mus,
            requested_height(required_heights, MultiConstraintWhirAir::InitialTarget),
        )?
        .ok_or(MultiConstraintWhirModuleError::RequiredHeight {
            role: MultiConstraintWhirAir::InitialTarget,
        })?;
        contexts.push(AirProvingContext::simple_no_pis(target));

        let alpha_counts = alpha_lookup_counts_without_weights(&self.params)?;
        let sumcheck_inputs = whir_proofs
            .iter()
            .zip(&multi_preflights)
            .enumerate()
            .map(
                |(proof_idx, (&proof, &preflight))| MultiConstraintSumcheckTraceInput {
                    proof,
                    preflight,
                    initial_claim_per_round: &blob.initial_claim_per_round.as_slice()[proof_idx
                        * (self.params.num_whir_rounds() + 1)
                        ..proof_idx * (self.params.num_whir_rounds() + 1)
                            + self.params.num_whir_rounds()],
                    post_sumcheck_claims: &blob.post_sumcheck_claims.as_slice()[proof_idx
                        * self.params.num_whir_sumcheck_rounds()
                        ..(proof_idx + 1) * self.params.num_whir_sumcheck_rounds()],
                },
            )
            .collect_vec();
        let sumcheck = generate_multi_constraint_sumcheck_trace(
            &self.params,
            &sumcheck_inputs,
            self.profile.constraint_count,
            &alpha_counts,
            requested_height(required_heights, MultiConstraintWhirAir::Sumcheck),
        )?
        .ok_or(MultiConstraintWhirModuleError::RequiredHeight {
            role: MultiConstraintWhirAir::Sumcheck,
        })?;
        contexts.push(AirProvingContext::simple_no_pis(sumcheck));

        let alphas = multi_preflights
            .iter()
            .map(|preflight| preflight.alphas.clone())
            .collect_vec();
        let weight = generate_multi_constraint_weight_trace(
            &self.profile,
            &statement_values,
            &alphas,
            requested_height(required_heights, MultiConstraintWhirAir::Weight),
        )?
        .ok_or(MultiConstraintWhirModuleError::RequiredHeight {
            role: MultiConstraintWhirAir::Weight,
        })?;
        contexts.push(AirProvingContext::simple_no_pis(weight));

        let final_polys = whir_proofs
            .iter()
            .map(|proof| proof.final_poly.clone())
            .collect_vec();
        let final_tidx = multi_preflights
            .iter()
            .map(|preflight| final_poly_tidx(&self.params, preflight))
            .collect::<Result<Vec<_>, _>>()?;
        let final_poly = generate_multi_constraint_final_poly_trace(
            &self.profile,
            &statement_values,
            &derived,
            &final_polys,
            self.params.num_whir_sumcheck_rounds(),
            &final_tidx,
            requested_height(required_heights, MultiConstraintWhirAir::FinalPolynomial),
        )?
        .ok_or(MultiConstraintWhirModuleError::RequiredHeight {
            role: MultiConstraintWhirAir::FinalPolynomial,
        })?;
        contexts.push(AirProvingContext::simple_no_pis(final_poly));

        let aggregate = generate_multi_constraint_final_aggregate_trace(
            &derived,
            &final_tidx,
            self.profile.constraint_count,
            requested_height(required_heights, MultiConstraintWhirAir::FinalAggregate),
        )?
        .ok_or(MultiConstraintWhirModuleError::RequiredHeight {
            role: MultiConstraintWhirAir::FinalAggregate,
        })?;
        contexts.push(AirProvingContext::simple_no_pis(aggregate));

        let completion_messages = self.completion_messages_from_blob(instances, &blob)?;
        let checkpoints = instances
            .iter()
            .map(|instance| instance.terminal_checkpoint)
            .collect_vec();
        let final_aggregates = completion_messages
            .iter()
            .map(|message| {
                EF::from_basis_coefficients_slice(&message.final_aggregate)
                    .expect("completion aggregate has the EF basis width")
            })
            .collect_vec();
        let final_claims = completion_messages
            .iter()
            .map(|message| {
                EF::from_basis_coefficients_slice(&message.final_claim)
                    .expect("completion claim has the EF basis width")
            })
            .collect_vec();
        let completion = generate_multi_constraint_completion_trace(
            &checkpoints,
            &final_aggregates,
            &final_claims,
            self.class_index,
            requested_height(required_heights, MultiConstraintWhirAir::Completion),
        )?
        .ok_or(MultiConstraintWhirModuleError::RequiredHeight {
            role: MultiConstraintWhirAir::Completion,
        })?;
        contexts.push(AirProvingContext::simple_no_pis(completion));

        debug_assert_eq!(contexts.len(), MultiConstraintWhirAir::COUNT);
        Ok(contexts)
    }

    /// Derive the exact typed messages emitted by this module's completion
    /// AIR for a batch of genuine prepared carriers.
    ///
    /// This is a witness-projection helper for enclosing circuits.  In
    /// particular, callers do not supply either terminal scalar: the
    /// generalized final aggregate is re-derived from the statement, proof,
    /// and transcript preflight, while the final claim is read from the
    /// ordinary WHIR blob after [`Self::install_multi_claims`] has replaced
    /// every legacy claim.  The completion AIR remains the sole authority: it
    /// constrains these values against the WHIR terminal and certified
    /// transcript-checkpoint buses.
    pub fn completion_messages(
        &self,
        child_vk: &MultiStarkVerifyingKey<BabyBearPoseidon2Config>,
        instances: &[PreparedMultiConstraintWhirCarrier],
    ) -> Result<Vec<MultiConstraintWhirCompletionMessage<F>>, MultiConstraintWhirModuleError> {
        if instances.is_empty() {
            return Err(MultiConstraintWhirModuleError::EmptyBatch);
        }
        if child_vk.inner.params != self.params {
            return Err(MultiConstraintWhirModuleError::ParameterMismatch);
        }

        let proofs = instances
            .iter()
            .map(|input| &input.proof_adapter)
            .collect_vec();
        let preflights = instances.iter().map(|input| &input.preflight).collect_vec();
        // Blob construction records PoW lookups as a trace-generation side
        // effect.  Completion projection does not own those rows, so isolate
        // them in a scratch generator instead of mutating caller state.
        let scratch_exp_bits_len = ExpBitsLenCpuTraceGenerator::default();
        let mut blob = if self.coefficient_two_coset_external_query {
            WhirModule::generate_blob_coefficient_two_coset(
                child_vk,
                &proofs,
                &preflights,
                &scratch_exp_bits_len,
            )
        } else {
            WhirModule::generate_blob(child_vk, &proofs, &preflights, &scratch_exp_bits_len)
        };
        self.install_multi_claims(instances, &mut blob)?;
        self.completion_messages_from_blob(instances, &blob)
    }

    fn completion_messages_from_blob(
        &self,
        instances: &[PreparedMultiConstraintWhirCarrier],
        blob: &WhirBlobCpu,
    ) -> Result<Vec<MultiConstraintWhirCompletionMessage<F>>, MultiConstraintWhirModuleError> {
        instances
            .iter()
            .enumerate()
            .map(|(proof_idx, input)| {
                let final_aggregate = derive_multi_constraint_whir_data(
                    &self.profile,
                    &input.statement,
                    input.multi_preflight.mu,
                    &input.multi_preflight.alphas,
                    &input.proof_adapter.whir_proof.final_poly,
                )?
                .final_weighted_evaluation;
                let final_claim =
                    blob.initial_claim_per_round[(proof_idx, self.params.num_whir_rounds())];
                Ok(MultiConstraintWhirCompletionMessage {
                    proof_idx: F::from_usize(proof_idx),
                    class_index: F::from_usize(self.class_index),
                    end_tidx: F::from_usize(input.terminal_checkpoint.end_tidx),
                    sample_count: F::from_usize(input.terminal_checkpoint.sample_count),
                    state: input.terminal_checkpoint.state,
                    final_aggregate: final_aggregate
                        .as_basis_coefficients_slice()
                        .try_into()
                        .expect("EF has the configured basis width"),
                    final_claim: final_claim
                        .as_basis_coefficients_slice()
                        .try_into()
                        .expect("EF has the configured basis width"),
                })
            })
            .collect()
    }

    fn install_multi_claims(
        &self,
        instances: &[PreparedMultiConstraintWhirCarrier],
        blob: &mut WhirBlobCpu,
    ) -> Result<(), MultiConstraintWhirModuleError> {
        let rounds = self.params.num_whir_rounds();
        let sumcheck_rounds = self.params.num_whir_sumcheck_rounds();
        let k = self.params.k_whir();
        let query_layout = blob.yis.layout().clone();
        let mut starts = Vec::with_capacity(instances.len() * (rounds + 1));
        let mut posts = Vec::with_capacity(instances.len() * sumcheck_rounds);
        let mut pre_queries = Vec::with_capacity(instances.len() * rounds);
        let mut final_evals = Vec::with_capacity(instances.len());

        for (proof_idx, input) in instances.iter().enumerate() {
            let derived = derive_multi_constraint_whir_data(
                &self.profile,
                &input.statement,
                input.multi_preflight.mu,
                &input.multi_preflight.alphas,
                &input.proof_adapter.whir_proof.final_poly,
            )?;
            let mut claim = derived.initial_target;
            for round in 0..rounds {
                starts.push(claim);
                for subround in 0..k {
                    let index = round * k + subround;
                    let [ev1, ev2] = input.proof_adapter.whir_proof.whir_sumcheck_polys[index];
                    claim = openvm_stark_backend::poly_common::interpolate_quadratic_at_012(
                        &[claim - ev1, ev1, ev2],
                        input.multi_preflight.alphas[index],
                    );
                    posts.push(claim);
                }
                let gamma = input.multi_preflight.gammas[round];
                if round + 1 < rounds {
                    claim += gamma * input.proof_adapter.whir_proof.ood_values[round];
                }
                pre_queries.push(claim);
                for (query_idx, gamma_power) in gamma
                    .powers()
                    .skip(2)
                    .take(query_layout.round_num_queries(round))
                    .enumerate()
                {
                    claim += gamma_power * blob.yis[(proof_idx, round, query_idx)];
                }
            }
            starts.push(claim);
            final_evals.push(derived.final_weighted_evaluation);
        }

        blob.initial_claim_per_round
            .as_mut_slice()
            .copy_from_slice(&starts);
        blob.post_sumcheck_claims
            .as_mut_slice()
            .copy_from_slice(&posts);
        blob.pre_query_claims
            .as_mut_slice()
            .copy_from_slice(&pre_queries);
        blob.final_poly_mle_evals.copy_from_slice(&final_evals);
        Ok(())
    }
}

impl AirModule for MultiConstraintWhirModule {
    fn num_airs(&self) -> usize {
        MultiConstraintWhirAir::COUNT
    }

    fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        let params = &self.params;
        let initial_log_domain_size = params.n_stack + params.l_skip + params.log_blowup;
        let num_rounds = params.num_whir_rounds();
        let num_queries_per_round = super::super::num_queries_per_round(params);
        let whir_round_encoder = Encoder::new(num_rounds.max(2), 2, false);
        let total_whir_queries = params
            .whir
            .rounds
            .iter()
            .map(|cfg| cfg.num_queries + 1)
            .sum();

        vec![
            Arc::new(WhirRoundAir {
                whir_module_bus: self.bus_inventory.whir_module_bus,
                commitments_bus: self.bus_inventory.commitments_bus,
                transcript_bus: self.bus_inventory.transcript_bus,
                exp_bits_len_bus: self.bus_inventory.exp_bits_len_bus,
                sumcheck_bus: self.sumcheck_bus,
                verify_queries_bus: self.verify_queries_bus,
                final_poly_mle_eval_bus: self.final_poly_mle_eval_bus,
                final_poly_query_eval_bus: self.final_poly_query_eval_bus,
                terminal_bus: Some(self.terminal_bus),
                query_bus: self.query_bus,
                gamma_bus: self.gamma_bus,
                k: params.k_whir(),
                num_rounds,
                initial_log_domain_size,
                coefficient_two_coset_initial_domain: self.coefficient_two_coset_external_query,
                final_poly_len: 1 << params.log_final_poly_len(),
                pow_bits: params.whir.query_phase_pow_bits,
                folding_pow_bits: params.whir.folding_pow_bits,
                generator: F::GENERATOR,
                whir_round_encoder,
                num_queries_per_round: num_queries_per_round.clone(),
            }) as AirRef<SC>,
            Arc::new(WhirQueryAir {
                transcript_bus: self.bus_inventory.transcript_bus,
                exp_bits_len_bus: self.bus_inventory.exp_bits_len_bus,
                query_bus: self.query_bus,
                verify_queries_bus: self.verify_queries_bus,
                verify_query_bus: self.verify_query_bus,
                k: params.k_whir(),
                initial_log_domain_size,
            }),
            Arc::new(InitialOpenedValuesAir {
                stacking_indices_bus: self.bus_inventory.stacking_indices_bus,
                whir_mu_bus: self.bus_inventory.whir_mu_bus,
                verify_query_bus: self.verify_query_bus,
                folding_bus: self.folding_bus,
                poseidon_permute_bus: self.bus_inventory.poseidon2_permute_bus,
                merkle_verify_bus: self.bus_inventory.merkle_verify_bus,
                k: params.k_whir(),
                initial_log_domain_size,
            }),
            Arc::new(NonInitialOpenedValuesAir {
                verify_query_bus: self.verify_query_bus,
                folding_bus: self.folding_bus,
                poseidon2_compress_bus: self.bus_inventory.poseidon2_compress_bus,
                merkle_verify_bus: self.bus_inventory.merkle_verify_bus,
                k: params.k_whir(),
                initial_log_domain_size,
            }),
            Arc::new(WhirFoldingAir {
                alpha_bus: self.alpha_bus,
                folding_bus: self.folding_bus,
                k: params.k_whir(),
            }),
            Arc::new(FinalPolyQueryEvalAir {
                query_bus: self.query_bus,
                alpha_bus: self.alpha_bus,
                gamma_bus: self.gamma_bus,
                final_poly_bus: self.final_poly_bus,
                final_poly_query_eval_bus: self.final_poly_query_eval_bus,
                num_whir_rounds: params.num_whir_rounds(),
                k_whir: params.k_whir(),
                log_final_poly_len: params.log_final_poly_len(),
            }),
            Arc::new(MultiConstraintInitialCommitmentAir {
                initial_commitment_bus: self.initial_commitment_bus,
                commitments_bus: self.bus_inventory.commitments_bus,
                stacking_indices_bus: self.bus_inventory.stacking_indices_bus,
                commitment_count: self.profile.commitment_widths.len(),
                commitment_lookup_mult: params.whir.rounds[0].num_queries,
                stacking_index_lookup_mult: params.whir.rounds[0].num_queries
                    * (1usize << params.k_whir()),
            }),
            Arc::new(MultiConstraintPrefixAir {
                statement_buses: self.statement_buses,
                weight_buses: self.weight_buses,
                transcript_bus: self.bus_inventory.transcript_bus,
                whir_module_bus: self.bus_inventory.whir_module_bus,
                whir_mu_bus: self.bus_inventory.whir_mu_bus,
                exp_bits_len_bus: self.bus_inventory.exp_bits_len_bus,
                constraint_count: self.profile.constraint_count,
                mu_pow_bits: params.whir.mu_pow_bits,
                generator: F::GENERATOR,
            }),
            Arc::new(MultiConstraintInitialTargetAir {
                statement_buses: self.statement_buses,
                weight_buses: self.weight_buses,
                constraint_count: self.profile.constraint_count,
                total_width: self.profile.total_width().expect("validated profile"),
            }),
            Arc::new(MultiConstraintSumcheckAir {
                sumcheck_bus: self.sumcheck_bus,
                transcript_bus: self.bus_inventory.transcript_bus,
                exp_bits_len_bus: self.bus_inventory.exp_bits_len_bus,
                alpha_bus: self.alpha_bus,
                k: params.k_whir(),
                folding_pow_bits: params.whir.folding_pow_bits,
                generator: F::GENERATOR,
            }),
            Arc::new(MultiConstraintWeightAir {
                statement_buses: self.statement_buses,
                weight_buses: self.weight_buses,
                alpha_bus: self.alpha_bus,
                constraint_count: self.profile.constraint_count,
                num_sumcheck_rounds: params.num_whir_sumcheck_rounds(),
            }),
            Arc::new(MultiConstraintFinalPolyMleAir {
                statement_buses: self.statement_buses,
                weight_buses: self.weight_buses,
                transcript_bus: self.bus_inventory.transcript_bus,
                final_poly_bus: self.final_poly_bus,
                num_vars: params.log_final_poly_len(),
                num_sumcheck_rounds: params.num_whir_sumcheck_rounds(),
                constraint_count: self.profile.constraint_count,
                total_whir_queries,
            }),
            Arc::new(MultiConstraintFinalAggregateAir {
                weight_buses: self.weight_buses,
                final_poly_mle_eval_bus: self.final_poly_mle_eval_bus,
                constraint_count: self.profile.constraint_count,
                num_whir_rounds: params.num_whir_rounds(),
            }),
            Arc::new(MultiConstraintCompletionAir {
                terminal_bus: self.terminal_bus,
                checkpoint_bus: self.terminal_checkpoint_bus,
                completion_bus: self.completion_bus,
                class_index: self.class_index,
                checkpoint_kind: self.terminal_checkpoint_kind,
            }),
        ]
    }
}

fn requested_height(
    required_heights: Option<&[usize]>,
    role: MultiConstraintWhirAir,
) -> Option<usize> {
    required_heights.map(|heights| heights[role.index()])
}

fn alpha_lookup_counts_without_weights(
    params: &openvm_stark_backend::SystemParams,
) -> Result<Vec<usize>, MultiConstraintWhirModuleError> {
    let k = params.k_whir();
    if k == 0 {
        return Err(MultiConstraintWhirModuleError::TraceSizeOverflow);
    }
    let mut counts = vec![0usize; params.num_whir_sumcheck_rounds()];
    let mut base = 0usize;
    for (round, cfg) in params.whir.rounds.iter().enumerate() {
        for subround in 0..k {
            let fold = 1usize
                .checked_shl((k - 1 - subround) as u32)
                .ok_or(MultiConstraintWhirModuleError::TraceSizeOverflow)?;
            counts[round * k + subround] = base
                .checked_add(
                    cfg.num_queries
                        .checked_mul(fold)
                        .ok_or(MultiConstraintWhirModuleError::TraceSizeOverflow)?,
                )
                .ok_or(MultiConstraintWhirModuleError::TraceSizeOverflow)?;
        }
        base = base
            .checked_add(cfg.num_queries + 1)
            .ok_or(MultiConstraintWhirModuleError::TraceSizeOverflow)?;
    }
    Ok(counts)
}

fn final_poly_tidx(
    params: &openvm_stark_backend::SystemParams,
    preflight: &MultiConstraintWhirTranscriptPreflight,
) -> Result<usize, MultiConstraintWhirModuleError> {
    let last = params
        .num_whir_rounds()
        .checked_sub(1)
        .ok_or(MultiConstraintWhirModuleError::TraceSizeOverflow)?;
    preflight.whir_round_tidx_per_round[last]
        .checked_add(
            params
                .k_whir()
                .checked_mul(3 * D_EF + pow_tidx_count(params.whir.folding_pow_bits))
                .ok_or(MultiConstraintWhirModuleError::TraceSizeOverflow)?,
        )
        .ok_or(MultiConstraintWhirModuleError::TraceSizeOverflow)
}

fn terminal_end_tidx(
    params: &openvm_stark_backend::SystemParams,
    preflight: &MultiConstraintWhirTranscriptPreflight,
) -> Result<usize, MultiConstraintWhirModuleError> {
    let last = params
        .num_whir_rounds()
        .checked_sub(1)
        .ok_or(MultiConstraintWhirModuleError::TraceSizeOverflow)?;
    preflight.query_tidx_per_round[last]
        .checked_add(params.whir.rounds[last].num_queries)
        .and_then(|value| value.checked_add(D_EF))
        .ok_or(MultiConstraintWhirModuleError::TraceSizeOverflow)
}

#[derive(Debug)]
pub enum MultiConstraintWhirModuleError {
    MultiConstraint(MultiConstraintWhirError),
    EmptyBatch,
    ParameterMismatch,
    InvalidTwoCosetProfile,
    ExternalQueryTraceRequired,
    UnexpectedExternalQueryTrace,
    PointDimension {
        actual: usize,
        expected: usize,
    },
    InitialCommitmentCount {
        actual: usize,
        expected: usize,
    },
    InitialCommitmentWidth {
        commitment: usize,
        actual: usize,
        expected: usize,
    },
    InitialOpeningShape {
        commitment: usize,
    },
    PreflightMismatch(&'static str),
    RequiredHeightCount {
        actual: usize,
        expected: usize,
    },
    RequiredHeight {
        role: MultiConstraintWhirAir,
    },
    TraceSizeOverflow,
}

impl From<MultiConstraintWhirError> for MultiConstraintWhirModuleError {
    fn from(value: MultiConstraintWhirError) -> Self {
        Self::MultiConstraint(value)
    }
}

impl core::fmt::Display for MultiConstraintWhirModuleError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::MultiConstraint(error) => error.fmt(f),
            Self::EmptyBatch => write!(f, "multi-constraint WHIR proof batch is empty"),
            Self::ParameterMismatch => write!(f, "multi-constraint WHIR system parameters differ"),
            Self::InvalidTwoCosetProfile => write!(
                f,
                "coefficient-two-coset WHIR requires rate one-half, a supported subgroup, and at least two rounds"
            ),
            Self::ExternalQueryTraceRequired => write!(
                f,
                "coefficient-two-coset WHIR requires its externally owned query trace"
            ),
            Self::UnexpectedExternalQueryTrace => write!(
                f,
                "an external two-coset query trace was supplied to ordinary WHIR"
            ),
            Self::PointDimension { actual, expected } => {
                write!(
                    f,
                    "multi-constraint point dimension {actual}, expected {expected}"
                )
            }
            Self::InitialCommitmentCount { actual, expected } => write!(
                f,
                "initial setup commitment count {actual}, expected {expected}"
            ),
            Self::InitialCommitmentWidth {
                commitment,
                actual,
                expected,
            } => write!(
                f,
                "initial setup commitment {commitment} width {actual}, expected {expected}"
            ),
            Self::InitialOpeningShape { commitment } => write!(
                f,
                "initial opened-row shape mismatch for setup commitment {commitment}"
            ),
            Self::PreflightMismatch(field) => {
                write!(f, "multi-constraint WHIR preflight mismatch: {field}")
            }
            Self::RequiredHeightCount { actual, expected } => {
                write!(f, "required-height count {actual}, expected {expected}")
            }
            Self::RequiredHeight { role } => {
                write!(f, "required trace height is too small for {role:?}")
            }
            Self::TraceSizeOverflow => write!(f, "multi-constraint WHIR trace size overflow"),
        }
    }
}

impl std::error::Error for MultiConstraintWhirModuleError {}

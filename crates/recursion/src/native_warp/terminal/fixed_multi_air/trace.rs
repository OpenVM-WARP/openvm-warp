//! Backend-neutral trace assembly for the aggregate fixed-multi-AIR terminal
//! verifier. SDK CPU/CUDA adapters only transport these matrices; all
//! transcript and structured-claim semantics are checked here.

use core::fmt;

use openvm_stark_backend::{
    native_warp::{
        DirectAirConstraintSumcheckProof, FixedMultiAirPesatIndex, FixedMultiAirTerminalProof,
    },
    poly_common::Squarable,
    transcript::TranscriptLog,
    warp_accum::{
        BinaryMerkleMultiproofRecord, TerminalConstrainedRsStatement, TerminalDescriptor,
        TerminalWhirVerification,
    },
    warp_pesat::AccumulatorInstance,
    StarkProtocolConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, D_EF, EF, F};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::dense::RowMajorMatrix;

use super::*;
use crate::{
    native_warp::{
        generate_native_leaf_hash_trace, generate_native_merkle_multiproof_trace,
        generate_native_terminal_rs_adjoint_q_trace,
        generate_native_terminal_rs_adjoint_selector_trace,
        generate_native_terminal_rs_adjoint_sumcheck_trace,
        generate_native_terminal_rs_adjoint_y_trace,
        generate_native_terminal_whir_expected_weight_trace,
        generate_native_terminal_whir_final_check_trace,
        generate_native_terminal_whir_final_table_trace,
        generate_native_terminal_whir_folding_trace, generate_native_terminal_whir_mobius_trace,
        generate_native_terminal_whir_opened_trace, generate_native_terminal_whir_point_trace,
        generate_native_terminal_whir_query_trace, generate_native_terminal_whir_round_trace,
        generate_native_terminal_whir_sumcheck_trace,
        generate_native_terminal_whir_weight_term_trace, NativeLeafHashInput,
        NativeTerminalRsAdjointSumcheckAir, NativeTerminalRsAdjointYAir,
        NativeTerminalWhirFinalCheckAir, NativeTerminalWhirMobiusAir, NativeTerminalWhirPointAir,
        NativeWarpTranscriptInputArtifacts,
    },
    primitives::exp_bits_len::ExpBitsLenCpuTraceGenerator,
};

#[derive(Clone, Debug)]
pub struct FixedMultiAirTerminalPartitionedTrace {
    pub cached_mains: Vec<RowMajorMatrix<F>>,
    pub common_main: RowMajorMatrix<F>,
}

impl FixedMultiAirTerminalPartitionedTrace {
    #[must_use]
    pub fn simple(common_main: RowMajorMatrix<F>) -> Self {
        Self {
            cached_mains: Vec::new(),
            common_main,
        }
    }

    #[must_use]
    pub fn cached(cached_main: RowMajorMatrix<F>, common_main: RowMajorMatrix<F>) -> Self {
        Self {
            cached_mains: vec![cached_main],
            common_main,
        }
    }
}

#[derive(Clone, Debug)]
pub struct FixedMultiAirTerminalTraceData {
    pub traces: Vec<FixedMultiAirTerminalPartitionedTrace>,
    pub poseidon2_compress_inputs: Vec<[F; 16]>,
    pub poseidon2_permute_inputs: Vec<[F; 16]>,
}

/// Post-finish witness retained by the stream executor. The proof objects are
/// still supplied separately by the SDK adapter and must equal the copies
/// retained here before trace generation starts.
pub struct FixedMultiAirTerminalTraceWitness<'a> {
    pub descriptor: &'a TerminalDescriptor<Digest>,
    pub instance: &'a AccumulatorInstance<EF, Digest>,
    pub reduction_proof: &'a FixedMultiAirTerminalProof<EF>,
    pub statement: &'a TerminalConstrainedRsStatement<EF>,
    pub whir_verification: &'a TerminalWhirVerification<F, EF, Digest>,
    pub linearizer_adjoint_proof: &'a FixedMultiAirLinearizerAdjointProof,
    pub transcript: &'a TranscriptLog<F, [F; 16]>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FixedMultiAirTerminalTraceError(pub &'static str);

impl fmt::Display for FixedMultiAirTerminalTraceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl std::error::Error for FixedMultiAirTerminalTraceError {}

type TraceResult<T> = Result<T, FixedMultiAirTerminalTraceError>;

fn fail<T>(message: &'static str) -> TraceResult<T> {
    Err(FixedMultiAirTerminalTraceError(message))
}

/// Trace-generation embedding selected by the trusted caller, never by the
/// terminal proof witness. The final wrapper adds one authenticated lookup of
/// every accumulator-instance coordinate; the standalone terminal does not.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FixedMultiAirTerminalTraceEmbedding {
    Standalone,
    FinalWrapper,
}

impl FixedMultiAirTerminalCircuit {
    /// Generate traces in exactly the same order as [`Self::airs`].
    pub fn generate_traces(
        &self,
        witness: FixedMultiAirTerminalTraceWitness<'_>,
    ) -> TraceResult<FixedMultiAirTerminalTraceData> {
        self.generate_traces_for_embedding(witness, FixedMultiAirTerminalTraceEmbedding::Standalone)
    }

    /// Generate terminal traces embedded in the direct-final wrapper.
    ///
    /// Unlike [`Self::generate_traces`], this setup-fixed path accounts for
    /// the outer statement's one lookup of every alpha, mu, beta, and eta
    /// coordinate. There is deliberately no runtime profile argument or
    /// witness flag that could select these multiplicities.
    pub fn generate_final_wrapper_traces(
        &self,
        witness: FixedMultiAirTerminalTraceWitness<'_>,
    ) -> TraceResult<FixedMultiAirTerminalTraceData> {
        self.generate_traces_for_embedding(
            witness,
            FixedMultiAirTerminalTraceEmbedding::FinalWrapper,
        )
    }

    fn generate_traces_for_embedding(
        &self,
        witness: FixedMultiAirTerminalTraceWitness<'_>,
        embedding: FixedMultiAirTerminalTraceEmbedding,
    ) -> TraceResult<FixedMultiAirTerminalTraceData> {
        let p = &self.profile;
        let relation = p.relation.as_ref();
        let proof = witness.reduction_proof;
        let instance = witness.instance;
        let verification = witness.whir_verification;
        if witness.descriptor.root != instance.rt
            || witness.descriptor.metadata_words().collect::<Vec<_>>() != p.metadata_words
            || instance.alpha.len() != p.alpha_len
            || instance.beta.len() != p.beta_len
            || proof.region_proofs.len() != p.endpoints.len()
            || proof.region_claims.len() != p.endpoints.len()
            || witness.statement.linearizer_claims.len() != p.claim_count()
            || verification.root != instance.rt
            || verification.rounds.len() != p.round_count()
            || verification.final_poly.len() != p.final_poly_len
            || verification.accumulator_adjoint.degree as usize != p.rs_adjoint_degree
            || verification.accumulator_adjoint.rounds.len() != p.rs_adjoint_round_count
        {
            return fail("fixed multi-AIR terminal witness profile");
        }

        let transcript_artifacts = self
            .transcript
            .generate_trace_inputs_with_external(
                &[witness.transcript],
                Vec::new(),
                Vec::new(),
                None,
            )
            .ok_or(FixedMultiAirTerminalTraceError("terminal transcript trace"))?;
        let NativeWarpTranscriptInputArtifacts {
            trace: transcript_trace,
            permutation_inputs: mut poseidon2_permute_inputs,
            compression_inputs: mut poseidon2_compress_inputs,
        } = transcript_artifacts;

        let prefix_counts = terminal_prefix_lookup_counts(relation, &p.endpoints)?;
        let prefix = match embedding {
            FixedMultiAirTerminalTraceEmbedding::Standalone => {
                generate_fixed_multi_air_terminal_prefix_traces(
                    witness.descriptor,
                    relation,
                    instance,
                    proof,
                    witness.transcript,
                    0,
                    &prefix_counts,
                    None,
                )
            }
            FixedMultiAirTerminalTraceEmbedding::FinalWrapper => {
                generate_fixed_multi_air_terminal_prefix_traces_for_final_wrapper(
                    witness.descriptor,
                    relation,
                    instance,
                    proof,
                    witness.transcript,
                    0,
                    &prefix_counts,
                    None,
                )
            }
        }
        .map_err(|error| {
            FixedMultiAirTerminalTraceError(match error {
                FixedMultiAirTerminalPrefixTraceError::Shape => "terminal prefix shape",
                FixedMultiAirTerminalPrefixTraceError::Transcript => "terminal prefix transcript",
                FixedMultiAirTerminalPrefixTraceError::Root => "terminal prefix root",
            })
        })?;

        let mut decomposition_traces = Vec::with_capacity(p.decomposition.len());
        let mut contributions = Vec::with_capacity(p.decomposition.len());
        let one = instance
            .beta
            .get(relation.pesat_shape().log_constraints)
            .copied()
            .ok_or(FixedMultiAirTerminalTraceError("terminal one coordinate"))?;
        for plan in &p.decomposition {
            let claim = match plan.kind {
                FixedMultiAirDecompositionComponentKind::Region { region } => {
                    proof.region_claims.get(region).copied().ok_or(
                        FixedMultiAirTerminalTraceError("regional decomposition claim"),
                    )?
                }
                FixedMultiAirDecompositionComponentKind::Padding => proof.padding_claim.ok_or(
                    FixedMultiAirTerminalTraceError("padding decomposition claim"),
                )?,
            };
            let schedule = generate_fixed_multi_air_decomposition_schedule_trace(plan, None)
                .map_err(|_| FixedMultiAirTerminalTraceError("decomposition schedule"))?;
            let (common, contribution) = generate_fixed_multi_air_decomposition_component_trace(
                plan,
                &instance.beta,
                instance.eta,
                one,
                claim,
                None,
            )
            .map_err(|_| FixedMultiAirTerminalTraceError("decomposition component"))?;
            decomposition_traces.push(FixedMultiAirTerminalPartitionedTrace::cached(
                schedule, common,
            ));
            contributions.push(contribution);
        }
        let decomposition_sum = generate_fixed_multi_air_decomposition_sum_trace(
            &contributions,
            instance.eta,
            one,
            None,
        )
        .map_err(|_| FixedMultiAirTerminalTraceError("decomposition sum"))?;

        let global_log = relation.pesat_shape().log_constraints;
        let mut tidx = prefix.end_tidx;
        let mut region_traces = Vec::new();
        let mut claim_records = Vec::with_capacity(p.claim_count());
        for endpoint in &p.endpoints {
            let region = endpoint.region;
            let local = relation
                .region_relation(region)
                .ok_or(FixedMultiAirTerminalTraceError("regional relation"))?;
            let regional_proof = proof
                .region_proofs
                .get(region)
                .ok_or(FixedMultiAirTerminalTraceError("regional proof"))?;
            let region_claim = proof.region_claims[region];
            let region_prefix = generate_fixed_multi_air_region_prefix_traces(
                region,
                relation,
                &instance.beta,
                region_claim,
                witness.transcript,
                tidx,
                None,
            )
            .map_err(|error| {
                FixedMultiAirTerminalTraceError(match error {
                    FixedMultiAirTerminalPrefixTraceError::Shape => "regional prefix shape",
                    FixedMultiAirTerminalPrefixTraceError::Transcript => {
                        "regional prefix transcript"
                    }
                    FixedMultiAirTerminalPrefixTraceError::Root => "regional prefix root",
                })
            })?;
            let constraint_count = local.constraint_dag().constraint_idx.len();
            let sumcheck_air = FixedMultiAirRegionSumcheckAir::new(
                self.buses.transcript,
                self.buses.region_start,
                self.buses.region_point,
                self.buses.region_sumcheck_final,
                region,
                endpoint.log_height,
                endpoint.dense_fixed_source_count(),
                endpoint.dynamic.len() + constraint_count + endpoint.analytic_fixed_source_count(),
            );
            let sumcheck = generate_fixed_multi_air_region_sumcheck_trace(
                &sumcheck_air,
                region_claim,
                regional_proof,
                witness.transcript,
                region_prefix.end_tidx,
                None,
            )
            .map_err(|_| FixedMultiAirTerminalTraceError("regional sumcheck"))?;
            let (point, final_claim, tail_tidx) = regional_point_and_claim(
                region_claim,
                regional_proof,
                witness.transcript,
                region_prefix.end_tidx,
            )?;
            let opening_counts = regional_opening_lookup_counts(endpoint)?;
            let tail = generate_fixed_multi_air_region_tail_traces(
                local,
                final_claim,
                regional_proof,
                witness.transcript,
                tail_tidx,
                &opening_counts,
                1,
                0,
                None,
            )
            .map_err(|_| FixedMultiAirTerminalTraceError("regional tail"))?;
            let fixed =
                generate_fixed_multi_air_endpoint_fixed_traces(local, endpoint, &point, None)
                    .map_err(|_| FixedMultiAirTerminalTraceError("regional fixed endpoint"))?;
            let weights = generate_fixed_multi_air_constraint_weight_traces(
                &instance.beta[..global_log],
                &point,
                constraint_count,
                relation.description().regions[region].constraint_offset,
                None,
                None,
            )
            .map_err(|_| FixedMultiAirTerminalTraceError("regional constraint weights"))?;
            let nodes = generate_fixed_multi_air_endpoint_node_traces(
                local,
                endpoint,
                &instance.beta,
                &regional_proof.opened_columns,
                &fixed.values,
                None,
            )
            .map_err(|_| FixedMultiAirTerminalTraceError("regional DAG endpoint"))?;
            let (fold_cached, fold_common, evaluated) =
                generate_fixed_multi_air_endpoint_fold_traces(
                    local,
                    &nodes.values,
                    &weights.weights,
                    one,
                    relation.description().exact_max_degree as usize
                        - local.description().exact_max_degree as usize,
                    None,
                )
                .map_err(|_| FixedMultiAirTerminalTraceError("regional endpoint fold"))?;
            if evaluated != final_claim {
                return fail("regional sumcheck/DAG endpoint mismatch");
            }
            let mapped = generate_fixed_multi_air_mapped_traces(
                endpoint,
                p.log_message_len,
                &regional_proof.opened_columns,
                &point,
                tail.rho,
                None,
                None,
            )
            .map_err(|_| FixedMultiAirTerminalTraceError("regional mapped claim"))?;
            let statement_claim = witness
                .statement
                .linearizer_claims
                .get(region)
                .ok_or(FixedMultiAirTerminalTraceError("regional structured claim"))?;
            if statement_claim.target != mapped.target {
                return fail("regional mapped target");
            }
            claim_records.push(FixedMultiAirStructuredClaimRecord {
                kind: 0,
                log_message_len: p.log_message_len,
                term_count: endpoint.dynamic.len(),
                point_len: 0,
                target: mapped.target,
                endpoint_lookup_count: endpoint.dynamic.len().max(1),
            });
            region_traces.extend([
                FixedMultiAirTerminalPartitionedTrace::cached(
                    region_prefix.cached,
                    region_prefix.common,
                ),
                FixedMultiAirTerminalPartitionedTrace::simple(sumcheck),
                FixedMultiAirTerminalPartitionedTrace::cached(tail.cached, tail.common),
                FixedMultiAirTerminalPartitionedTrace::cached(fixed.cached, fixed.common),
                FixedMultiAirTerminalPartitionedTrace::cached(
                    weights.init_cached,
                    weights.init_common,
                ),
                FixedMultiAirTerminalPartitionedTrace::cached(weights.dp_cached, weights.dp_common),
                FixedMultiAirTerminalPartitionedTrace::cached(nodes.cached, nodes.common),
                FixedMultiAirTerminalPartitionedTrace::cached(fold_cached, fold_common),
                FixedMultiAirTerminalPartitionedTrace::cached(
                    mapped.term_cached,
                    mapped.term_common,
                ),
                FixedMultiAirTerminalPartitionedTrace::cached(
                    mapped.point_cached,
                    mapped.point_common,
                ),
            ]);
            tidx = tail.end_tidx;
        }

        let mut padding_traces = Vec::new();
        if relation.description().padding_constraint_count != 0 {
            let padding_proof = proof
                .padding_proof
                .as_ref()
                .ok_or(FixedMultiAirTerminalTraceError("padding proof"))?;
            let padding_claim = proof
                .padding_claim
                .ok_or(FixedMultiAirTerminalTraceError("padding claim"))?;
            let sumcheck = generate_fixed_multi_air_padding_sumcheck_traces(
                padding_claim,
                padding_proof,
                witness.transcript,
                tidx,
                None,
            )
            .map_err(|_| FixedMultiAirTerminalTraceError("padding sumcheck"))?;
            let opening = generate_fixed_multi_air_padding_opening_trace(
                padding_proof,
                witness.transcript,
                sumcheck.final_claim,
                sumcheck.end_tidx,
            )
            .map_err(|_| FixedMultiAirTerminalTraceError("padding opening"))?;
            let end_tidx = sumcheck
                .end_tidx
                .checked_add(2 * D_EF)
                .ok_or(FixedMultiAirTerminalTraceError("padding transcript"))?;
            let weight = generate_fixed_multi_air_padding_weight_traces(
                relation,
                &instance.beta,
                &sumcheck.point,
                sumcheck.final_claim,
                padding_proof.message_opening,
                end_tidx,
                None,
            )
            .map_err(|_| FixedMultiAirTerminalTraceError("padding weight"))?;
            let statement_claim = witness
                .statement
                .linearizer_claims
                .get(relation.region_count())
                .ok_or(FixedMultiAirTerminalTraceError("padding structured claim"))?;
            if statement_claim.target != padding_proof.message_opening {
                return fail("padding mapped target");
            }
            claim_records.push(FixedMultiAirStructuredClaimRecord {
                kind: 1,
                log_message_len: p.log_message_len,
                term_count: 0,
                point_len: p.log_message_len,
                target: padding_proof.message_opening,
                endpoint_lookup_count: 1,
            });
            padding_traces.extend([
                FixedMultiAirTerminalPartitionedTrace::cached(sumcheck.cached, sumcheck.common),
                FixedMultiAirTerminalPartitionedTrace::simple(opening),
                FixedMultiAirTerminalPartitionedTrace::cached(weight.cached, weight.common),
            ]);
            tidx = end_tidx;
        }
        if claim_records.len() != p.claim_count()
            || verification.transcript_start.operations != tidx
        {
            return fail("terminal WHIR transcript boundary");
        }

        let whir_prefix = generate_fixed_multi_air_whir_prefix_trace(
            witness.descriptor,
            instance,
            witness.transcript,
            tidx,
        )
        .map_err(|_| FixedMultiAirTerminalTraceError("terminal WHIR prefix"))?;
        let statement_tidx = tidx
            .checked_add(1 + DIGEST_SIZE + p.metadata_words.len() + D_EF)
            .ok_or(FixedMultiAirTerminalTraceError("terminal statement index"))?;
        let target = generate_fixed_multi_air_structured_target_traces(
            witness.descriptor,
            statement_tidx,
            instance.mu,
            verification.batching_challenge,
            &claim_records,
            None,
        )
        .map_err(|_| FixedMultiAirTerminalTraceError("terminal structured target"))?;
        if target.initial_claim != verification.initial_claim {
            return fail("terminal structured initial claim");
        }
        let native_counts = [vec![1; p.alpha_len], vec![0], vec![0; p.beta_len], vec![0]];
        let (bridge_cached, bridge_common) =
            generate_fixed_multi_air_accumulator_bridge_traces(instance, &native_counts, None)
                .map_err(|_| FixedMultiAirTerminalTraceError("terminal accumulator bridge"))?;

        let batching_scales = batching_scales(verification.batching_challenge, p.claim_count());
        let whir_point = complete_whir_point(verification, p.k)?;
        let custom_start = verification.transcript_end.operations;
        let custom_air = FixedMultiAirLinearizerSumcheckAir {
            transcript_bus: self.buses.transcript,
            point_bus: self.buses.linearizer_aux_point,
            final_bus: self.buses.linearizer_sumcheck_final,
            log_message_len: p.log_message_len,
        };
        // Generate the raw trace once to derive exact point multiplicities,
        // then generate it again at the constrained transcript point.
        let provisional_point = linearizer_point_from_transcript(
            witness.linearizer_adjoint_proof,
            witness.transcript,
            custom_start,
            p.log_message_len,
        )?;
        let raw = generate_fixed_multi_air_linearizer_raw_traces(
            p.log_message_len,
            &p.linearizer_components,
            &witness.statement.linearizer_claims,
            &batching_scales,
            &provisional_point,
            None,
        )
        .map_err(|_| FixedMultiAirTerminalTraceError("terminal linearizer raw"))?;
        let point_counts = raw
            .aux_point_lookup_counts
            .iter()
            .enumerate()
            .map(|(coordinate, &count)| {
                count
                    + if coordinate < p.selector_folding_factor {
                        1
                    } else {
                        p.log_message_len - p.selector_folding_factor
                    }
            })
            .collect::<Vec<_>>();
        let custom_sumcheck = generate_fixed_multi_air_linearizer_sumcheck_traces(
            &custom_air,
            witness.linearizer_adjoint_proof,
            witness.transcript,
            custom_start,
            &point_counts,
            None,
        )
        .map_err(|_| FixedMultiAirTerminalTraceError("terminal linearizer sumcheck"))?;
        if custom_sumcheck.point != provisional_point {
            return fail("terminal linearizer point");
        }
        let (y_cached, y_common, selector_factors) =
            generate_fixed_multi_air_linearizer_y_traces_for_layout(
                &custom_sumcheck.point,
                &whir_point,
                p.initial_folding_factor,
                p.initial_rs_layout,
                None,
            )
            .map_err(|_| FixedMultiAirTerminalTraceError("terminal linearizer DFT kernel"))?;
        let (selector_trace, selector) =
            generate_fixed_multi_air_linearizer_selector_trace(&selector_factors, None)
                .map_err(|_| FixedMultiAirTerminalTraceError("terminal linearizer selector"))?;
        let raw_sum = generate_fixed_multi_air_linearizer_raw_sum_trace(&raw.term_values, None)
            .map_err(|_| FixedMultiAirTerminalTraceError("terminal linearizer raw sum"))?;
        let linearizer_expected = terminal_linearizer_expected_weight(verification, p.k)?;
        if custom_sumcheck.initial_claim != linearizer_expected {
            return fail("terminal linearizer initial claim");
        }
        let custom_final = generate_fixed_multi_air_linearizer_final_trace(
            statement_tidx,
            instance.rt,
            verification.batching_challenge,
            verification.initial_claim,
            custom_sumcheck.initial_claim,
            custom_sumcheck.final_claim,
            raw.value,
            selector,
        )
        .map_err(|_| FixedMultiAirTerminalTraceError("terminal linearizer final"))?;

        let generic = self.generate_generic_whir_traces(
            verification,
            witness.transcript,
            instance,
            &mut poseidon2_permute_inputs,
            &mut poseidon2_compress_inputs,
        )?;

        let mut traces = vec![
            FixedMultiAirTerminalPartitionedTrace::simple(transcript_trace),
            FixedMultiAirTerminalPartitionedTrace::cached(prefix.cached, prefix.common),
        ];
        traces.extend(decomposition_traces);
        traces.push(FixedMultiAirTerminalPartitionedTrace::simple(
            decomposition_sum,
        ));
        traces.extend(region_traces);
        traces.extend(padding_traces);
        traces.extend([
            FixedMultiAirTerminalPartitionedTrace::simple(whir_prefix),
            FixedMultiAirTerminalPartitionedTrace::cached(target.cached, target.common),
            FixedMultiAirTerminalPartitionedTrace::cached(bridge_cached, bridge_common),
            FixedMultiAirTerminalPartitionedTrace::cached(
                custom_sumcheck.cached,
                custom_sumcheck.common,
            ),
            FixedMultiAirTerminalPartitionedTrace::cached(y_cached, y_common),
            FixedMultiAirTerminalPartitionedTrace::simple(selector_trace),
            FixedMultiAirTerminalPartitionedTrace::cached(raw.cached, raw.common),
            FixedMultiAirTerminalPartitionedTrace::simple(raw_sum),
            FixedMultiAirTerminalPartitionedTrace::simple(custom_final),
        ]);
        traces.extend(generic);
        if traces.len()
            != self
                .airs::<openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config>()
                .len()
        {
            return fail("terminal AIR/trace count");
        }
        Ok(FixedMultiAirTerminalTraceData {
            traces,
            poseidon2_compress_inputs,
            poseidon2_permute_inputs,
        })
    }

    fn generate_generic_whir_traces(
        &self,
        verification: &TerminalWhirVerification<F, EF, Digest>,
        transcript: &TranscriptLog<F, [F; 16]>,
        instance: &AccumulatorInstance<EF, Digest>,
        poseidon2_permute_inputs: &mut Vec<[F; 16]>,
        poseidon2_compress_inputs: &mut Vec<[F; 16]>,
    ) -> TraceResult<Vec<FixedMultiAirTerminalPartitionedTrace>> {
        let p = &self.profile;
        let sumcheck = generate_native_terminal_whir_sumcheck_trace(
            verification,
            transcript,
            p.folding_pow_bits,
            1,
            None,
        )
        .ok_or(FixedMultiAirTerminalTraceError("WHIR sumcheck"))?;
        let round_air = self.round_air().map_err(FixedMultiAirTerminalTraceError)?;
        let round =
            generate_native_terminal_whir_round_trace(&round_air, verification, transcript, None)
                .ok_or(FixedMultiAirTerminalTraceError("WHIR round"))?;
        let query = generate_native_terminal_whir_query_trace(
            verification,
            transcript,
            p.k,
            p.initial_log_domain_size,
            p.round_count(),
            0,
            p.round_count(),
            p.final_poly_len,
            openvm_stark_backend::warp_accum::terminal_whir::TerminalWhirLayout::VectorAlphabet,
            None,
        )
        .ok_or(FixedMultiAirTerminalTraceError("WHIR query"))?;
        let opened = generate_native_terminal_whir_opened_trace(
            self.config.hasher(),
            verification,
            transcript,
            p.k,
            p.initial_log_domain_size,
            p.round_count(),
            0,
            None,
            openvm_stark_backend::warp_accum::terminal_whir::TerminalWhirLayout::VectorAlphabet,
        )
        .ok_or(FixedMultiAirTerminalTraceError("WHIR opened rows"))?;
        let folding = generate_native_terminal_whir_folding_trace(
            verification,
            p.k,
            None,
            openvm_stark_backend::warp_accum::terminal_whir::TerminalWhirLayout::VectorAlphabet,
        )
        .ok_or(FixedMultiAirTerminalTraceError("WHIR folding"))?;
        let leaf_inputs = opened
            .leaves
            .iter()
            .map(|leaf| NativeLeafHashInput {
                proof_idx: 0,
                tree_id: leaf.tree_id,
                leaf_index: leaf.leaf_index,
                values: &leaf.values,
                lookup_counts: &leaf.lookup_counts,
            })
            .collect::<Vec<_>>();
        let leaf_hash = generate_native_leaf_hash_trace(&leaf_inputs, None)
            .ok_or(FixedMultiAirTerminalTraceError("WHIR leaf hash"))?;
        poseidon2_permute_inputs.extend(leaf_hash.permutation_inputs);
        let mut merkle_records = opened
            .inner_merkle
            .iter()
            .map(|(tree, record)| (*tree, record))
            .collect::<Vec<(u32, &BinaryMerkleMultiproofRecord<Digest>)>>();
        merkle_records.extend(
            verification
                .rounds
                .iter()
                .enumerate()
                .map(|(round, record)| (round as u32, &record.multiproof)),
        );
        let merkle = generate_native_merkle_multiproof_trace(0, &merkle_records, None)
            .ok_or(FixedMultiAirTerminalTraceError("WHIR Merkle multiproof"))?;
        poseidon2_compress_inputs.extend(merkle_records.iter().flat_map(|(_, record)| {
            record.compressions.iter().map(|compression| {
                core::array::from_fn(|index| {
                    if index < DIGEST_SIZE {
                        compression.left[index]
                    } else {
                        compression.right[index - DIGEST_SIZE]
                    }
                })
            })
        }));
        let final_table = generate_native_terminal_whir_final_table_trace(verification, None)
            .ok_or(FixedMultiAirTerminalTraceError("WHIR final table"))?;
        let mobius_air =
            NativeTerminalWhirMobiusAir::new(self.buses.whir_final_poly, p.final_log_len());
        let mobius = generate_native_terminal_whir_mobius_trace(&mobius_air, verification, None)
            .ok_or(FixedMultiAirTerminalTraceError("WHIR Mobius"))?;
        let point_air = NativeTerminalWhirPointAir::new(
            self.buses.transcript,
            self.buses.whir_alpha,
            self.buses.whir_final_context,
            self.buses.whir_point,
            p.k,
            p.round_count(),
            p.final_poly_len,
            p.point_lookup_counts.clone(),
        );
        let point = generate_native_terminal_whir_point_trace(&point_air, verification, None)
            .ok_or(FixedMultiAirTerminalTraceError("WHIR point"))?;
        let final_check_air = NativeTerminalWhirFinalCheckAir {
            final_context_bus: self.buses.whir_final_context,
            final_poly_bus: self.buses.whir_final_poly,
            final_weight_bus: self.buses.whir_final_weight,
            point_bus: self.buses.whir_point,
            actual_weight_bus: self.buses.whir_actual_weight,
            point_prefix_len: p.k * p.round_count(),
            final_log_len: p.final_log_len(),
        };
        let final_check =
            generate_native_terminal_whir_final_check_trace(&final_check_air, verification, None)
                .ok_or(FixedMultiAirTerminalTraceError("WHIR final check"))?;
        let weight_term = generate_native_terminal_whir_weight_term_trace(verification, p.k, None)
            .ok_or(FixedMultiAirTerminalTraceError("WHIR weight terms"))?;
        let adjoint_air = NativeTerminalRsAdjointSumcheckAir::new(
            self.buses.transcript,
            self.buses.adjoint_round,
            self.buses.adjoint_claim,
            p.alpha_len - p.k,
            p.rs_adjoint_degree,
        );
        let adjoint_sumcheck = generate_native_terminal_rs_adjoint_sumcheck_trace(
            &adjoint_air,
            verification,
            transcript,
            None,
        )
        .ok_or(FixedMultiAirTerminalTraceError("RS adjoint sumcheck"))?;
        let adjoint_q =
            generate_native_terminal_rs_adjoint_q_trace(verification, &instance.alpha, p.k, None)
                .ok_or(FixedMultiAirTerminalTraceError("RS adjoint q"))?;
        let adjoint_y_air = NativeTerminalRsAdjointYAir::new(
            self.buses.adjoint_round,
            self.buses.adjoint_y,
            p.log_message_len - p.k,
            p.alpha_len - p.k,
        );
        let adjoint_y =
            generate_native_terminal_rs_adjoint_y_trace(&adjoint_y_air, verification, None)
                .ok_or(FixedMultiAirTerminalTraceError("RS adjoint y"))?;
        let adjoint_selector = generate_native_terminal_rs_adjoint_selector_trace(
            verification,
            &instance.alpha,
            p.k,
            None,
        )
        .ok_or(FixedMultiAirTerminalTraceError("RS adjoint selector"))?;
        let expected = generate_native_terminal_whir_expected_weight_trace(verification, p.k, None)
            .ok_or(FixedMultiAirTerminalTraceError("WHIR expected weight"))?;
        let exp = terminal_exp_bits_trace(verification, transcript, p)?;
        Ok(vec![
            FixedMultiAirTerminalPartitionedTrace::simple(sumcheck),
            FixedMultiAirTerminalPartitionedTrace::simple(round),
            FixedMultiAirTerminalPartitionedTrace::simple(query),
            FixedMultiAirTerminalPartitionedTrace::simple(opened.matrix),
            FixedMultiAirTerminalPartitionedTrace::simple(folding),
            FixedMultiAirTerminalPartitionedTrace::simple(leaf_hash.matrix),
            FixedMultiAirTerminalPartitionedTrace::simple(merkle),
            FixedMultiAirTerminalPartitionedTrace::simple(final_table),
            FixedMultiAirTerminalPartitionedTrace::simple(mobius),
            FixedMultiAirTerminalPartitionedTrace::simple(point),
            FixedMultiAirTerminalPartitionedTrace::simple(final_check),
            FixedMultiAirTerminalPartitionedTrace::simple(weight_term),
            FixedMultiAirTerminalPartitionedTrace::simple(adjoint_sumcheck),
            FixedMultiAirTerminalPartitionedTrace::simple(adjoint_q),
            FixedMultiAirTerminalPartitionedTrace::simple(adjoint_y),
            FixedMultiAirTerminalPartitionedTrace::simple(adjoint_selector),
            FixedMultiAirTerminalPartitionedTrace::simple(expected),
            FixedMultiAirTerminalPartitionedTrace::simple(exp),
        ])
    }
}

fn terminal_prefix_lookup_counts(
    relation: &FixedMultiAirPesatIndex<F, Digest>,
    endpoints: &[FixedMultiAirEndpointPlan],
) -> TraceResult<FixedMultiAirTerminalPrefixLookupCounts> {
    let alpha_len = relation.pesat_shape().log_witness
        + (relation.description().code_class.log_blowup as usize);
    let beta_len = relation.pesat_shape().beta_len();
    let mut counts = FixedMultiAirTerminalPrefixLookupCounts {
        global_claim: endpoints.len()
            + usize::from(relation.description().padding_constraint_count != 0)
            + 1,
        alpha: vec![1; alpha_len],
        // WHIR prefix plus the fixed/native bridge.
        mu: 2,
        // Typed beta-coordinate consumers only. The fixed/native accumulator
        // bridge uses the independent terminal instance-value bus.
        beta: vec![0; beta_len],
        eta: 1,
        region_claims: vec![2; endpoints.len()],
        padding_claim: 2,
    };
    // Count every typed beta-bus consumer in the setup-fixed regional
    // schedule. The fixed/native bridge is deliberately absent: it consumes
    // the terminal instance-value catalog, not this bus.
    let global_log = relation.pesat_shape().log_constraints;
    for plan in FixedMultiAirDecompositionComponentPlan::from_relation(relation)
        .map_err(|_| FixedMultiAirTerminalTraceError("decomposition lookup profile"))?
    {
        for coordinate in 0..plan.prefix_len {
            counts.beta[coordinate] += 1;
        }
    }
    for endpoint in endpoints {
        let local = relation
            .region_relation(endpoint.region)
            .ok_or(FixedMultiAirTerminalTraceError("regional lookup relation"))?;
        let description = relation.description().regions.get(endpoint.region).ok_or(
            FixedMultiAirTerminalTraceError("regional lookup description"),
        )?;
        for coordinate in 0..global_log {
            counts.beta[coordinate] += 1; // compact-global transcript prefix
        }
        for coordinate in 0..global_log {
            counts.beta[coordinate] += local.constraint_dag().constraint_idx.len();
        }
        counts.beta[global_log] += 1; // regional transcript prefix
        let explicit_start = global_log + description.explicit_offset as usize;
        let explicit_end = explicit_start + description.explicit_len as usize;
        for coordinate in explicit_start..explicit_end {
            *counts
                .beta
                .get_mut(coordinate)
                .ok_or(FixedMultiAirTerminalTraceError("explicit beta lookup"))? += 1;
        }
        for source in endpoint.node_sources.iter().flatten() {
            if let FixedMultiAirEndpointSource::Beta(coordinate) = source {
                *counts
                    .beta
                    .get_mut(*coordinate)
                    .ok_or(FixedMultiAirTerminalTraceError("DAG beta lookup"))? += 1;
            }
        }
        counts.beta[endpoint.one_coordinate] += local.constraint_dag().nodes.len();
        counts.beta[endpoint.one_coordinate] += local.constraint_dag().constraint_idx.len();
    }
    if relation.description().padding_constraint_count != 0 {
        // The paired-interval padding endpoint authenticates every global
        // constraint coordinate, not a legacy block-local suffix.
        for coordinate in 0..global_log {
            counts.beta[coordinate] += 1;
        }
        counts.beta[global_log] += 1; // padding homogenization factor
    }
    Ok(counts)
}

fn regional_opening_lookup_counts(endpoint: &FixedMultiAirEndpointPlan) -> TraceResult<Vec<usize>> {
    let mut counts = vec![1usize; endpoint.dynamic.len()]; // mapped term
    for source in endpoint.node_sources.iter().flatten() {
        if let FixedMultiAirEndpointSource::Opening(index) = source {
            *counts
                .get_mut(*index)
                .ok_or(FixedMultiAirTerminalTraceError("opening lookup profile"))? += 1;
        }
    }
    Ok(counts)
}

fn regional_point_and_claim(
    initial_claim: EF,
    proof: &DirectAirConstraintSumcheckProof<EF>,
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
) -> TraceResult<(Vec<EF>, EF, usize)> {
    if proof.round_evaluations.is_empty() {
        return Ok((Vec::new(), initial_claim, start_tidx));
    }
    let step = FixedMultiAirRegionSumcheckAir::transcript_values_per_round();
    let mut point = Vec::with_capacity(proof.round_evaluations.len());
    let mut claim = proof
        .round_evaluations
        .first()
        .and_then(|values| values.get(0).zip(values.get(1)))
        .map(|(&zero, &one)| zero + one)
        .ok_or(FixedMultiAirTerminalTraceError("regional sumcheck shape"))?;
    for (round, evaluations) in proof.round_evaluations.iter().enumerate() {
        if evaluations.len() != 7 || evaluations[0] + evaluations[1] != claim {
            return fail("regional sumcheck claim");
        }
        let tidx = start_tidx + round * step;
        let challenge = transcript_ext(transcript, tidx + (3 + 7) * D_EF, true)?;
        claim = interpolate(evaluations, challenge)?;
        point.push(challenge);
    }
    Ok((
        point,
        claim,
        start_tidx + proof.round_evaluations.len() * step,
    ))
}

pub fn linearizer_point_from_transcript(
    proof: &FixedMultiAirLinearizerAdjointProof,
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
    log_message_len: usize,
) -> Result<Vec<EF>, FixedMultiAirTerminalTraceError> {
    let evaluations = log_message_len + 2;
    if proof.round_evaluations.len() != log_message_len {
        return fail("linearizer adjoint shape");
    }
    let step = (3 + evaluations + 1) * D_EF;
    (0..log_message_len)
        .map(|round| {
            transcript_ext(
                transcript,
                start_tidx + round * step + (3 + evaluations) * D_EF,
                true,
            )
        })
        .collect()
}

fn transcript_ext(
    transcript: &TranscriptLog<F, [F; 16]>,
    tidx: usize,
    is_sample: bool,
) -> TraceResult<EF> {
    let values =
        transcript
            .values()
            .get(tidx..tidx + D_EF)
            .ok_or(FixedMultiAirTerminalTraceError(
                "terminal transcript extent",
            ))?;
    let samples =
        transcript
            .samples()
            .get(tidx..tidx + D_EF)
            .ok_or(FixedMultiAirTerminalTraceError(
                "terminal transcript sample extent",
            ))?;
    if samples.iter().any(|&sample| sample != is_sample) {
        return fail("terminal transcript operation kind");
    }
    EF::from_basis_coefficients_slice(values).ok_or(FixedMultiAirTerminalTraceError(
        "terminal transcript extension",
    ))
}

fn interpolate(values: &[EF], point: EF) -> TraceResult<EF> {
    let mut result = EF::ZERO;
    for (index, &value) in values.iter().enumerate() {
        let mut basis = EF::ONE;
        let mut denominator = F::ONE;
        for other in 0..values.len() {
            if other != index {
                basis *= point - EF::from_usize(other);
                denominator *= F::from_usize(index) - F::from_usize(other);
            }
        }
        result += value * basis * EF::from(denominator.inverse());
    }
    Ok(result)
}

fn batching_scales(xi: EF, count: usize) -> Vec<EF> {
    let mut scale = xi;
    (0..count)
        .map(|_| {
            let value = scale;
            scale *= xi;
            value
        })
        .collect()
}

pub fn complete_whir_point(
    verification: &TerminalWhirVerification<F, EF, Digest>,
    k: usize,
) -> Result<Vec<EF>, FixedMultiAirTerminalTraceError> {
    if verification
        .rounds
        .iter()
        .any(|round| round.alphas.len() != k)
    {
        return fail("WHIR point profile");
    }
    let mut point = verification
        .rounds
        .iter()
        .flat_map(|round| round.alphas.iter().copied())
        .collect::<Vec<_>>();
    point.extend_from_slice(&verification.suffix_point);
    Ok(point)
}

pub fn terminal_linearizer_expected_weight(
    verification: &TerminalWhirVerification<F, EF, Digest>,
    k: usize,
) -> Result<EF, FixedMultiAirTerminalTraceError> {
    let point = complete_whir_point(verification, k)?;
    let mut terms = EF::ZERO;
    for (round_index, round) in verification.rounds.iter().enumerate() {
        let after_folds = (round_index + 1) * k;
        let suffix = point
            .get(after_folds..)
            .ok_or(FixedMultiAirTerminalTraceError("WHIR weight suffix"))?;
        if round_index + 1 != verification.rounds.len() {
            let generator = round
                .ood_point
                .ok_or(FixedMultiAirTerminalTraceError("WHIR OOD point"))?;
            terms += round.gamma * eq_generator_point(generator, suffix);
        }
        for (query, &root) in round.query_roots.iter().enumerate() {
            terms +=
                round.gamma.exp_u64(query as u64 + 2) * eq_generator_point(EF::from(root), suffix);
        }
    }
    Ok(verification.expected_weight - verification.accumulator_adjoint.claimed_value - terms)
}

fn eq_generator_point(generator: EF, point: &[EF]) -> EF {
    generator
        .exp_powers_of_2()
        .zip(point)
        .map(|(left, &right)| left * right + (EF::ONE - left) * (EF::ONE - right))
        .product()
}

fn terminal_exp_bits_trace(
    verification: &TerminalWhirVerification<F, EF, Digest>,
    transcript: &TranscriptLog<F, [F; 16]>,
    profile: &FixedMultiAirTerminalCircuitProfile,
) -> TraceResult<RowMajorMatrix<F>> {
    let generator = ExpBitsLenCpuTraceGenerator::default();
    for (round_index, round) in verification.rounds.iter().enumerate() {
        for sumcheck in &round.sumcheck_rounds {
            if profile.folding_pow_bits > 0 {
                let sample_tidx = sumcheck.transcript_span.start.operations + 2 * D_EF + 1;
                let sample = *transcript
                    .values()
                    .get(sample_tidx)
                    .ok_or(FixedMultiAirTerminalTraceError("folding PoW sample"))?;
                generator.add_request(F::GENERATOR, sample, profile.folding_pow_bits);
            }
        }
        let query_tidx = round
            .transcript_span
            .end
            .operations
            .checked_sub(D_EF + round.query_indices.len())
            .ok_or(FixedMultiAirTerminalTraceError("query transcript index"))?;
        if profile.query_pow_bits > 0 {
            let sample = *transcript
                .values()
                .get(query_tidx - 1)
                .ok_or(FixedMultiAirTerminalTraceError("query PoW sample"))?;
            generator.add_request(F::GENERATOR, sample, profile.query_pow_bits);
        }
        let bits = profile
            .initial_log_domain_size
            .checked_sub(profile.k + round_index)
            .ok_or(FixedMultiAirTerminalTraceError("query index bits"))?;
        // Queries address rows of the vector-alphabet oracle.  The low `k`
        // variables are coefficients inside one leaf, so the row domain is
        // `2^(log_codeword_len-k-round)`.  Using the full scalar-codeword
        // dimension here both disagrees with backend WHIR and exceeds
        // BabyBear's two-adicity for the production log-29 codeword.
        let omega = F::two_adic_generator(bits);
        generator.add_requests_with_shift((0..round.query_indices.len()).map(|query| {
            let sample = transcript.values()[query_tidx + query];
            (omega, sample, bits, bits, 1)
        }));
    }
    generator
        .generate_trace_row_major(None)
        .ok_or(FixedMultiAirTerminalTraceError("exp-bits trace"))
}

const _: () = assert!(D_EF == 4);

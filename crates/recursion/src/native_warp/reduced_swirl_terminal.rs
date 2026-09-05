//! Recursive terminal `Decide` for reduced-SWIRL WARP.
//!
//! This component certifies the exact suffix checked by the native
//! `verify_reduced_swirl_native_recorded` verifier. Its only message-side
//! relation is
//!
//! ```text
//! MLE(message, beta[..m]) = beta[m] + accumulator.eta.
//! ```
//!
//! The independent codeword claim is discharged by the coefficient-subgroup
//! RS adjoint and WHIR under the same accumulator root. There is no PESAT
//! package in this module and no PCS-opening-as-PESAT compatibility path.
//!
//! Setup is deliberately two-stage. [`ReducedSwirlTerminalSetupIdentity`]
//! excludes the wrapper component digest, so source/VACC/terminal identities
//! can first be combined into the wrapper component digest without a cycle.
//! [`ReducedSwirlTerminalSetupIdentity::bind_wrapper_component`] then fixes
//! that digest only as a receipt constant. The final wrapper verifying key
//! independently binds the instantiated AIR inventory.

use core::{
    borrow::{Borrow, BorrowMut},
    fmt,
};
use std::sync::Arc;

use openvm_circuit::arch::POSEIDON2_WIDTH;
use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_cpu_backend::CpuBackend;
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder},
    native_warp::{native_accumulator_instance_digest_preimage, NATIVE_ACCUMULATOR_INSTANCE_TAG},
    prover::AirProvingContext,
    soundness,
    transcript::TranscriptLog,
    warp_accum::{
        canonical_swirl_reduced_code_binding, select_proven_rs_warp_params,
        BinaryMerkleMultiproofRecord, FieldElementDigestObserver, ReducedConstrainedCodeRelation,
        SwirlConstrainedRsRelation, TerminalConstrainedRsStatement, TerminalDescriptor,
        TerminalWhirVerification, WhirInitialRsLayout, WhirInitialRsWarpCode,
    },
    warp_pesat::{AccumulatorInstance, LinearChainSchedule, PesatShape},
    AirRef, BaseAirWithPublicValues, PartitionedBaseAir, StarkProtocolConfig, SystemParams,
    WhirConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    poseidon2_compress_with_capacity, BabyBearPoseidon2Config as NativeSC, Digest, DIGEST_SIZE,
    D_EF, EF, F,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing, TwoAdicField};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::{
        Poseidon2CompressBus, Poseidon2CompressMessage, Poseidon2PermuteBus,
        Poseidon2PermuteMessage, TranscriptBus, TranscriptBusMessage, TranscriptEndIndexBus,
        TranscriptEndIndexMessage,
    },
    native_warp::{
        generate_native_leaf_hash_trace, generate_native_merkle_leaf_adapter_trace,
        generate_native_merkle_multiproof_trace, generate_native_terminal_rs_adjoint_q_trace,
        generate_native_terminal_rs_adjoint_selector_trace,
        generate_native_terminal_rs_adjoint_sumcheck_trace,
        generate_native_terminal_rs_adjoint_y_trace,
        generate_native_terminal_whir_expected_weight_trace,
        generate_native_terminal_whir_final_check_trace,
        generate_native_terminal_whir_final_table_trace,
        generate_native_terminal_whir_folding_trace,
        generate_native_terminal_whir_linearizer_weight_trace,
        generate_native_terminal_whir_mobius_trace, generate_native_terminal_whir_opened_trace,
        generate_native_terminal_whir_point_trace, generate_native_terminal_whir_query_trace,
        generate_native_terminal_whir_round_trace, generate_native_terminal_whir_statement_trace,
        generate_native_terminal_whir_sumcheck_trace,
        generate_native_terminal_whir_weight_term_trace, reduced_swirl_vacc_footer_elements,
        validate_native_terminal_eq_statement, NativeAccumulatorAlgebraicDigestBus,
        NativeAccumulatorAlgebraicDigestMessage, NativeAccumulatorDigestElementBus,
        NativeAccumulatorDigestElementMessage, NativeLeafHashAir, NativeLeafHashInput,
        NativeLeafValueBus, NativeMerkleLeafAdapterAir, NativeMerkleMultiproofAir,
        NativeMerkleNodeBus, NativeMerkleRootBus, NativeOpeningLeafBus,
        NativeTerminalAccumulatorRootBus, NativeTerminalAccumulatorRootMessage,
        NativeTerminalAccumulatorValueBus, NativeTerminalAccumulatorValueMessage,
        NativeTerminalLinearizerEndBus, NativeTerminalRsAdjointClaimBus,
        NativeTerminalRsAdjointQAir, NativeTerminalRsAdjointQBus, NativeTerminalRsAdjointRoundBus,
        NativeTerminalRsAdjointSelectorAir, NativeTerminalRsAdjointSumcheckAir,
        NativeTerminalRsAdjointValueBus, NativeTerminalRsAdjointYAir, NativeTerminalRsAdjointYBus,
        NativeTerminalTranscriptBinding, NativeTerminalWhirActualWeightBus,
        NativeTerminalWhirAlphaBus, NativeTerminalWhirExpectedWeightAir,
        NativeTerminalWhirFinalCheckAir, NativeTerminalWhirFinalClaimBus,
        NativeTerminalWhirFinalContextBus, NativeTerminalWhirFinalPolyBus,
        NativeTerminalWhirFinalTableAir, NativeTerminalWhirFinalWeightBus,
        NativeTerminalWhirFoldingAir, NativeTerminalWhirFoldingBus,
        NativeTerminalWhirLinearizerWeightAir, NativeTerminalWhirLinearizerWeightBus,
        NativeTerminalWhirMobiusAir, NativeTerminalWhirOpenedAir, NativeTerminalWhirPointAir,
        NativeTerminalWhirPointBus, NativeTerminalWhirQueryAir, NativeTerminalWhirQueryBus,
        NativeTerminalWhirRoundAir, NativeTerminalWhirRoundBus, NativeTerminalWhirStatementAir,
        NativeTerminalWhirStatementBus, NativeTerminalWhirSumcheckAir,
        NativeTerminalWhirVerifyQueriesBus, NativeTerminalWhirWeightTermAir,
        NativeTerminalWhirWeightTermBus, NativeTerminalWhirWeightTermResultBus,
        ReducedSwirlVaccFooterBus, ReducedSwirlVaccFooterMessage,
    },
    primitives::{
        bus::{ExpBitsLenBus, RightShiftBus},
        exp_bits_len::{ExpBitsLenAir, ExpBitsLenCpuTraceGenerator},
    },
    system::{BusIndexManager, BusInventory},
    transcript::Poseidon2BusOwner,
    utils::poseidon2_hash_slice_with_states,
};

const REDUCED_SWIRL_TERMINAL_CIRCUIT_TAG: u64 = 0x5253_5754_4552_4d32;
const REDUCED_SWIRL_TERMINAL_SCHEDULE_VERSION: u32 = 3;
/// SWIRL Theorem 4.2.4 unique-decoding power-batching accounting. The
/// wrapper verifier derives the exact numerator from proof-bound widths; this
/// version and `w_stack` bind the maximum family accepted by the setup.
const REDUCED_SWIRL_POWER_BATCH_SECURITY_VERSION: u32 = 1;
const REDUCED_SWIRL_PROTOCOL_IDENTITY_TAG: u64 = 0x5253_5750_524f_5432;
const REDUCED_SWIRL_RELATION_IDENTITY_TAG: u64 = 0x5253_5752_454c_4132;
const REDUCED_SWIRL_TERMINAL_INDEX_TAG: u64 = 0x5253_5754_494e_4432;
const REDUCED_SWIRL_TERMINAL_COMPONENT_TAG: u64 = 0x5253_5754_434d_5032;
const HASH_RATE: usize = DIGEST_SIZE;
pub const REDUCED_SWIRL_TERMINAL_MAX_AIR_DEGREE: usize = 5;

type ProductionReducedSwirlCode =
    WhirInitialRsWarpCode<<NativeSC as StarkProtocolConfig>::Hasher, FieldElementDigestObserver>;

crate::define_typed_lookup_bus!(
    ReducedSwirlLocalTerminalReceiptBus,
    ReducedSwirlLocalTerminalReceiptMessage
);
crate::define_typed_permutation_bus!(
    ReducedSwirlTerminalStartBus,
    ReducedSwirlTerminalStartMessage
);

/// Field order is identical to continuations'
/// `ReducedSwirlTerminalReceiptMessage`.
///
/// The aggregate receives the wrapper's bus index and constructs this typed
/// view at that exact index. It is not an independently allocated namespace.
#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlLocalTerminalReceiptMessage<T> {
    pub protocol_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub terminal_index_digest: [T; DIGEST_SIZE],
    pub verifier_component_digest: [T; DIGEST_SIZE],
    pub final_accumulator_digest: [T; DIGEST_SIZE],
    pub final_accumulator_root: [T; DIGEST_SIZE],
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlTerminalReceiptRecord {
    pub protocol_digest: Digest,
    pub relation_digest: Digest,
    pub terminal_index_digest: Digest,
    pub verifier_component_digest: Digest,
    pub final_accumulator_digest: Digest,
    pub final_accumulator_root: Digest,
}

#[repr(C)]
#[derive(AlignedBorrow, Clone, Debug)]
pub struct ReducedSwirlTerminalStartMessage<T> {
    pub source_count: T,
    pub call_count: T,
    pub footer_start_tidx: T,
    pub terminal_start_tidx: T,
    pub manifest_digest: [T; DIGEST_SIZE],
}

/// Deterministically reconstructed terminal setup for the production native
/// reduced-SWIRL family.
///
/// This factory accepts only the same fixed inputs used by
/// `ReducedSwirlNativeSetup::new`. It reconstructs the canonical SWIRL
/// relation, coefficient-subgroup code, proven WARP parameters, terminal
/// descriptor template, and all three receipt digests. No constructor taking
/// caller-selected protocol/relation/index digests is exposed.
#[derive(Clone)]
pub struct ReducedSwirlTerminalProductionSetup {
    identity: Arc<ReducedSwirlTerminalSetupIdentity>,
    system_params: SystemParams,
    input_arity: usize,
    family_target_bits: usize,
    maximum_source_count: usize,
}

impl ReducedSwirlTerminalProductionSetup {
    pub fn from_native_fixed_params(
        system_params: SystemParams,
        input_arity: usize,
        family_target_bits: usize,
        maximum_source_count: usize,
    ) -> Result<Self, ReducedSwirlTerminalError> {
        if maximum_source_count == 0
            || family_target_bits == 0
            || system_params.log_commit_rows_per_query != system_params.whir.k
        {
            return Err(ReducedSwirlTerminalError::Profile(
                "native fixed terminal parameters",
            ));
        }
        let log_message_len = system_params
            .l_skip
            .checked_add(system_params.n_stack)
            .ok_or(ReducedSwirlTerminalError::Profile("message dimension"))?;
        let log_codeword_len = log_message_len
            .checked_add(system_params.log_blowup)
            .ok_or(ReducedSwirlTerminalError::Profile("codeword dimension"))?;
        let rows_per_query = 1usize
            .checked_shl(system_params.log_commit_rows_per_query as u32)
            .ok_or(ReducedSwirlTerminalError::Profile("rows per query"))?;
        let relation = SwirlConstrainedRsRelation::<EF>::new(log_message_len)
            .map_err(ReducedSwirlTerminalError::Profile)?;
        let relation_shape = <SwirlConstrainedRsRelation<EF> as ReducedConstrainedCodeRelation<
            F,
            EF,
        >>::shape(&relation);
        let binding = canonical_swirl_reduced_code_binding(
            &relation,
            system_params.l_skip,
            system_params.n_stack,
            system_params.log_blowup,
            system_params.log_commit_rows_per_query,
        );
        let schedule = LinearChainSchedule::new(input_arity)
            .map_err(|_| ReducedSwirlTerminalError::Profile("WARP schedule"))?;
        let call_count = schedule.step_fresh_counts(maximum_source_count).len();
        let union_bits = call_count.next_power_of_two().ilog2() as usize;
        let selected_target = family_target_bits
            .checked_add(union_bits)
            .ok_or(ReducedSwirlTerminalError::Profile("security union bound"))?;
        let security = select_proven_rs_warp_params(
            input_arity,
            log_message_len,
            log_codeword_len,
            PesatShape {
                log_constraints: 0,
                log_witness: relation_shape.log_message_len,
                explicit_len: relation_shape.beta_len + 1,
                max_degree: relation_shape.max_constraint_degree,
            },
            soundness::challenge_field_bits::<NativeSC>(),
            selected_target,
        )
        .map_err(|_| ReducedSwirlTerminalError::Profile("proven WARP security"))?;
        let config = NativeSC::default_from_params(system_params.clone());
        let code: ProductionReducedSwirlCode = WhirInitialRsWarpCode::try_new_coefficient_subgroup(
            config.hasher().clone(),
            log_message_len,
            system_params.log_blowup,
            0,
            rows_per_query,
        )
        .map_err(|_| ReducedSwirlTerminalError::Profile("coefficient-subgroup code"))?;
        let descriptor = TerminalDescriptor::from_whir_initial_rs(
            [F::ZERO; DIGEST_SIZE],
            &code,
            &system_params.whir,
            D_EF,
        );
        let transcript_binding = NativeTerminalTranscriptBinding::coefficient_subgroup(
            &descriptor,
            &code,
            &system_params.whir,
        )
        .map_err(|_| ReducedSwirlTerminalError::Profile("terminal transcript binding"))?;
        let relation_digest = fixed_material_digest(
            REDUCED_SWIRL_RELATION_IDENTITY_TAG,
            &reduced_swirl_relation_identity_material(&relation, relation_shape),
        );
        let protocol_material = reduced_swirl_protocol_identity_material(
            &system_params,
            &binding,
            security.params,
            selected_target,
            maximum_source_count,
        );
        let protocol_digest =
            fixed_material_digest(REDUCED_SWIRL_PROTOCOL_IDENTITY_TAG, &protocol_material);
        let terminal_index_material = reduced_swirl_terminal_index_material(
            protocol_digest,
            relation_digest,
            &descriptor,
            &transcript_binding,
            &system_params.whir,
        );
        let terminal_index_digest =
            fixed_material_digest(REDUCED_SWIRL_TERMINAL_INDEX_TAG, &terminal_index_material);
        let identity = Arc::new(ReducedSwirlTerminalSetupIdentity::new_fixed(
            &descriptor,
            &code,
            &system_params.whir,
            protocol_digest,
            relation_digest,
            terminal_index_digest,
        )?);
        Ok(Self {
            identity,
            system_params,
            input_arity,
            family_target_bits,
            maximum_source_count,
        })
    }

    #[must_use]
    pub fn setup_identity(&self) -> Arc<ReducedSwirlTerminalSetupIdentity> {
        self.identity.clone()
    }

    pub fn protocol_digest(&self) -> Digest {
        self.identity.protocol_digest
    }

    pub fn relation_digest(&self) -> Digest {
        self.identity.relation_digest
    }

    pub fn terminal_index_digest(&self) -> Digest {
        self.identity.terminal_index_digest
    }

    /// Stage-one component identity used by
    /// `ReducedSwirlVerifierComponent::protocol_digest()`. It binds all AIR
    /// interaction namespaces but deliberately excludes the later wrapper
    /// component digest carried by the terminal receipt.
    #[allow(clippy::too_many_arguments)]
    pub fn component_protocol_digest(
        &self,
        shared: &BusInventory,
        main_transcript_bus: TranscriptBus,
        vacc_footer_bus: ReducedSwirlVaccFooterBus,
        wrapper_terminal_receipt_bus_idx: BusIndex,
        first_internal_bus_idx: BusIndex,
    ) -> Result<Digest, ReducedSwirlTerminalError> {
        let buses = ReducedSwirlTerminalBusInventory::new(first_internal_bus_idx);
        let material = reduced_swirl_terminal_component_material(
            &self.identity,
            &buses,
            shared,
            main_transcript_bus,
            vacc_footer_bus,
            wrapper_terminal_receipt_bus_idx,
        )?;
        Ok(fixed_material_digest(
            REDUCED_SWIRL_TERMINAL_COMPONENT_TAG,
            &material,
        ))
    }

    #[must_use]
    pub const fn input_arity(&self) -> usize {
        self.input_arity
    }

    #[must_use]
    pub const fn family_target_bits(&self) -> usize {
        self.family_target_bits
    }

    #[must_use]
    pub const fn maximum_source_count(&self) -> usize {
        self.maximum_source_count
    }

    /// Stage two: bind the wrapper's already-derived component digest and
    /// instantiate AIRs on the wrapper-owned receipt bus.
    #[allow(clippy::too_many_arguments)]
    pub fn instantiate(
        &self,
        verifier_component_digest: Digest,
        shared: &BusInventory,
        main_transcript_bus: TranscriptBus,
        vacc_footer_bus: ReducedSwirlVaccFooterBus,
        wrapper_terminal_receipt_bus_idx: BusIndex,
        first_internal_bus_idx: BusIndex,
    ) -> Result<ReducedSwirlTerminalComponent, ReducedSwirlTerminalError> {
        let profile = self
            .identity
            .clone()
            .bind_wrapper_component(verifier_component_digest)?;
        ReducedSwirlTerminalComponent::new(
            profile,
            shared,
            main_transcript_bus,
            vacc_footer_bus,
            wrapper_terminal_receipt_bus_idx,
            first_internal_bus_idx,
            self.system_params.clone(),
        )
    }
}

fn fixed_material_digest(tag: u64, material: &[F]) -> Digest {
    let mut preimage = Vec::with_capacity(material.len() + 2);
    preimage.push(F::from_u64(tag));
    preimage.push(F::from_usize(material.len()));
    preimage.extend_from_slice(material);
    poseidon2_hash_slice_with_states(&preimage).0
}

fn push_usize_fields(output: &mut Vec<F>, values: impl IntoIterator<Item = usize>) {
    output.extend(values.into_iter().map(F::from_usize));
}

fn push_bytes_fields(output: &mut Vec<F>, values: &[u8]) {
    output.push(F::from_usize(values.len()));
    output.extend(values.iter().copied().map(F::from_u8));
}

fn push_extension_fields(output: &mut Vec<F>, values: &[EF]) {
    output.push(F::from_usize(values.len()));
    output.extend(values.iter().flat_map(|value| {
        <EF as BasedVectorSpace<F>>::as_basis_coefficients_slice(value)
            .iter()
            .copied()
    }));
}

fn reduced_swirl_relation_identity_material(
    relation: &SwirlConstrainedRsRelation<EF>,
    shape: openvm_stark_backend::warp_accum::ReducedConstrainedCodeShape,
) -> Vec<F> {
    let mut output = Vec::new();
    push_usize_fields(
        &mut output,
        [
            shape.log_message_len,
            shape.beta_len,
            shape.max_constraint_degree,
        ],
    );
    push_extension_fields(&mut output, relation.binding());
    output
}

fn reduced_swirl_protocol_identity_material(
    params: &SystemParams,
    binding: &openvm_stark_backend::warp_accum::ReducedConstrainedCodeBinding<EF>,
    warp: openvm_stark_backend::warp_accum::WarpParams,
    selected_target_bits: usize,
    maximum_source_count: usize,
) -> Vec<F> {
    let mut output = vec![
        F::from_u32(REDUCED_SWIRL_TERMINAL_SCHEDULE_VERSION),
        F::from_u32(binding.protocol_version),
    ];
    push_usize_fields(
        &mut output,
        [
            params.l_skip,
            params.n_stack,
            params.log_blowup,
            params.log_commit_rows_per_query,
            params.w_stack,
            D_EF,
            warp.input_arity,
            warp.num_ood,
            warp.num_shift_queries,
            warp.batching_arity,
            selected_target_bits,
            maximum_source_count,
            REDUCED_SWIRL_POWER_BATCH_SECURITY_VERSION as usize,
        ],
    );
    push_bytes_fields(&mut output, &binding.source_domain);
    push_extension_fields(&mut output, &binding.relation_binding);
    push_extension_fields(&mut output, &binding.code_binding);
    push_whir_config_fields(&mut output, &params.whir);
    output
}

fn reduced_swirl_terminal_index_material(
    protocol_digest: Digest,
    relation_digest: Digest,
    descriptor: &TerminalDescriptor<Digest>,
    binding: &NativeTerminalTranscriptBinding,
    whir: &WhirConfig,
) -> Vec<F> {
    let mut output = Vec::new();
    output.extend(protocol_digest);
    output.extend(relation_digest);
    output.push(F::from_usize(descriptor.metadata_words().count()));
    output.extend(descriptor.metadata_words().map(F::from_u64));
    output.push(F::from_usize(binding.descriptor_post_root().len()));
    output.extend_from_slice(binding.descriptor_post_root());
    output.push(F::from_usize(binding.whir_layout().len()));
    output.extend_from_slice(binding.whir_layout());
    push_whir_config_fields(&mut output, whir);
    output
}

fn push_whir_config_fields(output: &mut Vec<F>, whir: &WhirConfig) {
    push_usize_fields(
        output,
        [
            whir.k,
            whir.mu_pow_bits,
            whir.query_phase_pow_bits,
            whir.folding_pow_bits,
            whir.rounds.len(),
        ],
    );
    push_usize_fields(output, whir.rounds.iter().map(|round| round.num_queries));
    match whir.proximity {
        openvm_stark_backend::WhirProximityStrategy::UniqueDecoding => {
            push_usize_fields(output, [0, 0, 0]);
        }
        openvm_stark_backend::WhirProximityStrategy::SplitUniqueList {
            m,
            list_start_round,
        } => {
            push_usize_fields(output, [1, m, list_start_round]);
        }
        openvm_stark_backend::WhirProximityStrategy::ListDecoding { m } => {
            push_usize_fields(output, [2, m, 0]);
        }
    }
}

/// Setup identity computed before the final wrapper component digest exists.
/// `setup_identity_material` and `setup_identity_digest` intentionally cannot
/// observe a wrapper component digest.
#[derive(Clone, Debug)]
pub struct ReducedSwirlTerminalSetupIdentity {
    protocol_digest: Digest,
    relation_digest: Digest,
    terminal_index_digest: Digest,
    descriptor_words: Vec<u64>,
    transcript_binding: Arc<NativeTerminalTranscriptBinding>,
    alpha_len: usize,
    beta_len: usize,
    log_message_len: usize,
    log_codeword_len: usize,
    k: usize,
    final_poly_len: usize,
    query_pow_bits: usize,
    folding_pow_bits: usize,
    num_queries_per_round: Vec<usize>,
    point_lookup_counts: Vec<usize>,
    weight_term_count: usize,
    rs_adjoint_degree: usize,
}

impl ReducedSwirlTerminalSetupIdentity {
    fn new_fixed<Obs>(
        descriptor: &TerminalDescriptor<Digest>,
        code: &WhirInitialRsWarpCode<<NativeSC as StarkProtocolConfig>::Hasher, Obs>,
        whir: &WhirConfig,
        protocol_digest: Digest,
        relation_digest: Digest,
        terminal_index_digest: Digest,
    ) -> Result<Self, ReducedSwirlTerminalError> {
        let transcript_binding = Arc::new(
            NativeTerminalTranscriptBinding::coefficient_subgroup(descriptor, code, whir)
                .map_err(|_| ReducedSwirlTerminalError::Profile("coefficient-subgroup binding"))?,
        );
        let log_message_len = code.log_message_len();
        let log_codeword_len = code.log_codeword_len();
        let k = whir.k;
        let round_count = whir.rounds.len();
        let folded = k
            .checked_mul(round_count)
            .ok_or(ReducedSwirlTerminalError::Profile("WHIR folded prefix"))?;
        let final_log_len = log_message_len
            .checked_sub(folded)
            .ok_or(ReducedSwirlTerminalError::Profile("WHIR final dimension"))?;
        let final_poly_len = 1usize
            .checked_shl(final_log_len as u32)
            .ok_or(ReducedSwirlTerminalError::Profile("WHIR final polynomial"))?;
        let num_queries_per_round = whir
            .rounds
            .iter()
            .map(|round| round.num_queries)
            .collect::<Vec<_>>();
        let point_lookup_counts = terminal_point_lookup_counts(
            k,
            &num_queries_per_round,
            final_poly_len,
            log_message_len,
        )
        .ok_or(ReducedSwirlTerminalError::Profile("WHIR point inventory"))?;
        let weight_term_count = num_queries_per_round
            .iter()
            .try_fold(round_count.saturating_sub(1), |sum, &queries| {
                sum.checked_add(queries)
            })
            .ok_or(ReducedSwirlTerminalError::Profile("WHIR weight inventory"))?;
        let setup = Self {
            protocol_digest,
            relation_digest,
            terminal_index_digest,
            descriptor_words: descriptor.metadata_words().collect(),
            transcript_binding,
            alpha_len: log_codeword_len,
            beta_len: log_message_len + 1,
            log_message_len,
            log_codeword_len,
            k,
            final_poly_len,
            query_pow_bits: whir.query_phase_pow_bits,
            folding_pow_bits: whir.folding_pow_bits,
            num_queries_per_round,
            point_lookup_counts,
            weight_term_count,
            rs_adjoint_degree: log_message_len + 1,
        };
        setup.validate_descriptor_and_code(descriptor, code)?;
        Ok(setup)
    }

    fn validate_descriptor_and_code<Obs>(
        &self,
        descriptor: &TerminalDescriptor<Digest>,
        code: &WhirInitialRsWarpCode<<NativeSC as StarkProtocolConfig>::Hasher, Obs>,
    ) -> Result<(), ReducedSwirlTerminalError> {
        let nonzero = |digest: &Digest| digest.iter().any(|value| *value != F::ZERO);
        if !nonzero(&self.protocol_digest)
            || !nonzero(&self.relation_digest)
            || !nonzero(&self.terminal_index_digest)
            || code.layout() != WhirInitialRsLayout::CoefficientSubgroup
            || code.initial_folding_factor() != 0
            || descriptor.initial_folding_factor != 0
            || descriptor.initial_oracle_width != 1
            || descriptor.log_message_len as usize != self.log_message_len
            || descriptor.log_codeword_len as usize != self.log_codeword_len
            || descriptor.rows_per_query as usize != (1usize << self.k)
            || descriptor
                .metadata_words()
                .ne(self.descriptor_words.iter().copied())
            || self.alpha_len != self.log_codeword_len
            || self.beta_len != self.log_message_len + 1
            || self.k == 0
            || self.num_queries_per_round.is_empty()
            || self.num_queries_per_round.contains(&0)
            || self.final_poly_len <= 1
            || self.rs_adjoint_degree
                > crate::native_warp::terminal::NATIVE_TERMINAL_MAX_ADJOINT_DEGREE
            || self.weight_term_count == 0
            || self.log_codeword_len.checked_sub(self.k).is_none()
            || self.log_codeword_len - self.k > F::TWO_ADICITY
        {
            return Err(ReducedSwirlTerminalError::Profile("terminal profile"));
        }
        Ok(())
    }

    #[must_use]
    pub fn round_count(&self) -> usize {
        self.num_queries_per_round.len()
    }

    #[must_use]
    pub fn final_log_len(&self) -> usize {
        self.final_poly_len.ilog2() as usize
    }

    /// Setup-owned material used when deriving the wrapper component digest.
    /// The wrapper component digest itself is absent by construction.
    #[must_use]
    pub fn setup_identity_material(&self) -> Vec<F> {
        let mut out = vec![
            F::from_u64(REDUCED_SWIRL_TERMINAL_CIRCUIT_TAG),
            F::from_u32(REDUCED_SWIRL_TERMINAL_SCHEDULE_VERSION),
        ];
        for digest in [
            self.protocol_digest,
            self.relation_digest,
            self.terminal_index_digest,
        ] {
            out.extend(digest);
        }
        out.extend(
            [
                self.alpha_len,
                self.beta_len,
                self.log_message_len,
                self.log_codeword_len,
                self.k,
                self.final_poly_len,
                self.query_pow_bits,
                self.folding_pow_bits,
                self.weight_term_count,
                self.rs_adjoint_degree,
                REDUCED_SWIRL_TERMINAL_MAX_AIR_DEGREE,
            ]
            .map(F::from_usize),
        );
        out.push(F::from_usize(self.descriptor_words.len()));
        out.extend(self.descriptor_words.iter().copied().map(F::from_u64));
        out.push(F::from_usize(
            self.transcript_binding.descriptor_post_root().len(),
        ));
        out.extend_from_slice(self.transcript_binding.descriptor_post_root());
        out.push(F::from_usize(self.transcript_binding.whir_layout().len()));
        out.extend_from_slice(self.transcript_binding.whir_layout());
        out.push(F::from_usize(self.num_queries_per_round.len()));
        out.extend(
            self.num_queries_per_round
                .iter()
                .copied()
                .map(F::from_usize),
        );
        out.push(F::from_usize(self.point_lookup_counts.len()));
        out.extend(self.point_lookup_counts.iter().copied().map(F::from_usize));
        // Semantic AIR role inventory. This catches accidental omission or
        // reordering without depending on the eventual wrapper receipt value.
        out.extend((0..ReducedSwirlTerminalComponent::COMPONENT_COUNT).map(F::from_usize));
        out
    }

    pub fn setup_identity_digest(&self) -> Digest {
        poseidon2_hash_slice_with_states(&self.setup_identity_material()).0
    }

    pub fn bind_wrapper_component(
        self: Arc<Self>,
        verifier_component_digest: Digest,
    ) -> Result<ReducedSwirlTerminalProfile, ReducedSwirlTerminalError> {
        if verifier_component_digest
            .iter()
            .all(|value| *value == F::ZERO)
        {
            return Err(ReducedSwirlTerminalError::Profile(
                "zero wrapper component digest",
            ));
        }
        Ok(ReducedSwirlTerminalProfile {
            setup: self,
            verifier_component_digest,
        })
    }
}

/// Fully bound receipt profile. Its setup identity remains `setup`'s digest;
/// the final AIR/VK binds `verifier_component_digest` independently.
#[derive(Clone, Debug)]
pub struct ReducedSwirlTerminalProfile {
    setup: Arc<ReducedSwirlTerminalSetupIdentity>,
    verifier_component_digest: Digest,
}

impl ReducedSwirlTerminalProfile {
    pub fn setup_identity_digest(&self) -> Digest {
        self.setup.setup_identity_digest()
    }

    #[must_use]
    pub fn setup_identity(&self) -> &Arc<ReducedSwirlTerminalSetupIdentity> {
        &self.setup
    }

    pub fn verifier_component_digest(&self) -> Digest {
        self.verifier_component_digest
    }
}

#[derive(Clone, Debug)]
pub struct ReducedSwirlTerminalBusInventory {
    /// Fixed local namespace used by the generic terminal AIRs. A constrained
    /// bridge re-keys the actual final VACC proof transcript into proof zero.
    pub semantic_transcript: TranscriptBus,
    pub terminal_start: ReducedSwirlTerminalStartBus,
    pub exp_bits_len: ExpBitsLenBus,
    pub right_shift: RightShiftBus,
    pub accumulator_value: NativeTerminalAccumulatorValueBus,
    pub accumulator_root: NativeTerminalAccumulatorRootBus,
    pub accumulator_digest_element: NativeAccumulatorDigestElementBus,
    pub accumulator_algebraic_digest: NativeAccumulatorAlgebraicDigestBus,
    pub linearizer_end: NativeTerminalLinearizerEndBus,
    pub whir_round: NativeTerminalWhirRoundBus,
    pub whir_alpha: NativeTerminalWhirAlphaBus,
    pub whir_verify_queries: NativeTerminalWhirVerifyQueriesBus,
    pub whir_query: NativeTerminalWhirQueryBus,
    pub whir_folding: NativeTerminalWhirFoldingBus,
    pub whir_statement: NativeTerminalWhirStatementBus,
    pub whir_final_claim: NativeTerminalWhirFinalClaimBus,
    pub whir_final_context: NativeTerminalWhirFinalContextBus,
    pub whir_point: NativeTerminalWhirPointBus,
    pub whir_final_poly: NativeTerminalWhirFinalPolyBus,
    pub whir_final_weight: NativeTerminalWhirFinalWeightBus,
    pub whir_actual_weight: NativeTerminalWhirActualWeightBus,
    pub adjoint_round: NativeTerminalRsAdjointRoundBus,
    pub adjoint_claim: NativeTerminalRsAdjointClaimBus,
    pub adjoint_q: NativeTerminalRsAdjointQBus,
    pub adjoint_y: NativeTerminalRsAdjointYBus,
    pub adjoint_value: NativeTerminalRsAdjointValueBus,
    pub weight_term: NativeTerminalWhirWeightTermBus,
    pub weight_term_result: NativeTerminalWhirWeightTermResultBus,
    pub linearizer_weight: NativeTerminalWhirLinearizerWeightBus,
    pub leaf_value: NativeLeafValueBus,
    pub opening_leaf: NativeOpeningLeafBus,
    pub merkle_node: NativeMerkleNodeBus,
    pub merkle_root: NativeMerkleRootBus,
    next_bus_idx: BusIndex,
}

impl ReducedSwirlTerminalBusInventory {
    #[must_use]
    pub fn new(first_bus_idx: BusIndex) -> Self {
        let mut manager = BusIndexManager::from_next_bus_idx(first_bus_idx);
        macro_rules! bus {
            ($ty:ty) => {
                <$ty>::new(manager.new_bus_idx())
            };
        }
        let mut result = Self {
            semantic_transcript: bus!(TranscriptBus),
            terminal_start: bus!(ReducedSwirlTerminalStartBus),
            exp_bits_len: bus!(ExpBitsLenBus),
            right_shift: bus!(RightShiftBus),
            accumulator_value: bus!(NativeTerminalAccumulatorValueBus),
            accumulator_root: bus!(NativeTerminalAccumulatorRootBus),
            accumulator_digest_element: bus!(NativeAccumulatorDigestElementBus),
            accumulator_algebraic_digest: bus!(NativeAccumulatorAlgebraicDigestBus),
            linearizer_end: bus!(NativeTerminalLinearizerEndBus),
            whir_round: bus!(NativeTerminalWhirRoundBus),
            whir_alpha: bus!(NativeTerminalWhirAlphaBus),
            whir_verify_queries: bus!(NativeTerminalWhirVerifyQueriesBus),
            whir_query: bus!(NativeTerminalWhirQueryBus),
            whir_folding: bus!(NativeTerminalWhirFoldingBus),
            whir_statement: bus!(NativeTerminalWhirStatementBus),
            whir_final_claim: bus!(NativeTerminalWhirFinalClaimBus),
            whir_final_context: bus!(NativeTerminalWhirFinalContextBus),
            whir_point: bus!(NativeTerminalWhirPointBus),
            whir_final_poly: bus!(NativeTerminalWhirFinalPolyBus),
            whir_final_weight: bus!(NativeTerminalWhirFinalWeightBus),
            whir_actual_weight: bus!(NativeTerminalWhirActualWeightBus),
            adjoint_round: bus!(NativeTerminalRsAdjointRoundBus),
            adjoint_claim: bus!(NativeTerminalRsAdjointClaimBus),
            adjoint_q: bus!(NativeTerminalRsAdjointQBus),
            adjoint_y: bus!(NativeTerminalRsAdjointYBus),
            adjoint_value: bus!(NativeTerminalRsAdjointValueBus),
            weight_term: bus!(NativeTerminalWhirWeightTermBus),
            weight_term_result: bus!(NativeTerminalWhirWeightTermResultBus),
            linearizer_weight: bus!(NativeTerminalWhirLinearizerWeightBus),
            leaf_value: bus!(NativeLeafValueBus),
            opening_leaf: bus!(NativeOpeningLeafBus),
            merkle_node: bus!(NativeMerkleNodeBus),
            merkle_root: bus!(NativeMerkleRootBus),
            next_bus_idx: 0,
        };
        result.next_bus_idx = manager.next_bus_idx();
        result
    }

    #[must_use]
    pub const fn next_bus_idx(&self) -> BusIndex {
        self.next_bus_idx
    }
}

#[allow(clippy::too_many_arguments)]
fn reduced_swirl_terminal_component_material(
    setup: &ReducedSwirlTerminalSetupIdentity,
    buses: &ReducedSwirlTerminalBusInventory,
    shared: &BusInventory,
    main_transcript_bus: TranscriptBus,
    vacc_footer_bus: ReducedSwirlVaccFooterBus,
    wrapper_terminal_receipt_bus_idx: BusIndex,
) -> Result<Vec<F>, ReducedSwirlTerminalError> {
    let first_internal = buses.semantic_transcript.index();
    let next_internal = buses.next_bus_idx();
    let external = [
        main_transcript_bus.index(),
        vacc_footer_bus.index(),
        wrapper_terminal_receipt_bus_idx,
        shared.poseidon2_permute_bus.index(),
        shared.poseidon2_compress_bus.index(),
    ];
    if external
        .iter()
        .any(|&index| index >= first_internal && index < next_internal)
        || external
            .iter()
            .enumerate()
            .any(|(left, value)| external[left + 1..].contains(value))
    {
        return Err(ReducedSwirlTerminalError::Profile(
            "terminal external bus wiring",
        ));
    }
    let mut material = Vec::new();
    material.extend(setup.setup_identity_digest());
    push_usize_fields(
        &mut material,
        [
            ReducedSwirlTerminalComponent::COMPONENT_COUNT,
            main_transcript_bus.index() as usize,
            vacc_footer_bus.index() as usize,
            wrapper_terminal_receipt_bus_idx as usize,
            shared.poseidon2_permute_bus.index() as usize,
            shared.poseidon2_compress_bus.index() as usize,
            first_internal as usize,
            next_internal as usize,
        ],
    );
    Ok(material)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReducedSwirlTerminalError {
    Profile(&'static str),
    Witness(&'static str),
    Transcript(&'static str),
    Trace(&'static str),
}

impl fmt::Display for ReducedSwirlTerminalError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Profile(message) => write!(f, "reduced-SWIRL terminal profile: {message}"),
            Self::Witness(message) => write!(f, "reduced-SWIRL terminal witness: {message}"),
            Self::Transcript(message) => write!(f, "reduced-SWIRL terminal transcript: {message}"),
            Self::Trace(message) => write!(f, "reduced-SWIRL terminal trace: {message}"),
        }
    }
}

impl std::error::Error for ReducedSwirlTerminalError {}

/// Exact output of the recursion-side VACC footer AIR.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlTerminalFooterRecord {
    pub source_count: usize,
    pub call_count: usize,
    pub proof_idx: usize,
    /// Physical transcript namespace. Whole-block wrappers use `proof_idx`;
    /// a bounded finalizer re-keys the sole suffix log to zero.
    pub local_proof_idx: usize,
    pub start_tidx: usize,
    pub end_tidx: usize,
    pub manifest_digest: Digest,
}

pub struct ReducedSwirlTerminalRecord<'a> {
    pub footer: &'a ReducedSwirlTerminalFooterRecord,
    pub instance: &'a AccumulatorInstance<EF, Digest>,
    pub descriptor: &'a TerminalDescriptor<Digest>,
    pub statement: &'a TerminalConstrainedRsStatement<EF>,
    pub verification: &'a TerminalWhirVerification<F, EF, Digest>,
    pub transcript: &'a TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ReducedSwirlTerminalTranscriptBridgeCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub local_proof_idx: T,
    pub tidx: T,
    pub is_first: T,
    pub is_last: T,
    pub source_count: T,
    pub call_count: T,
    pub footer_start_tidx: T,
    pub footer_end_tidx: T,
    pub manifest_digest: [T; DIGEST_SIZE],
    pub value: T,
    pub is_sample: T,
}

/// Re-keys the exact suffix of the actual final VACC transcript into the
/// generic terminal AIRs' fixed proof-zero namespace.
///
/// This is not a second transcript. Every row is consumed from
/// `main_transcript_bus` at `proof_idx = call_count - 1` and produced at the
/// same absolute `tidx` on `semantic_transcript_bus`. The first row consumes
/// the authenticated VACC footer and starts at its exact end cursor.
#[derive(ColumnsAir)]
#[columns_via(ReducedSwirlTerminalTranscriptBridgeCols<u8>)]
pub struct ReducedSwirlTerminalTranscriptBridgeAir {
    pub main_transcript_bus: TranscriptBus,
    pub transcript_end_index_bus: TranscriptEndIndexBus,
    pub semantic_transcript_bus: TranscriptBus,
    pub start_bus: ReducedSwirlTerminalStartBus,
}

impl BaseAirWithPublicValues<F> for ReducedSwirlTerminalTranscriptBridgeAir {}
impl PartitionedBaseAir<F> for ReducedSwirlTerminalTranscriptBridgeAir {}
impl BaseAir<F> for ReducedSwirlTerminalTranscriptBridgeAir {
    fn width(&self) -> usize {
        ReducedSwirlTerminalTranscriptBridgeCols::<F>::width()
    }
}

impl<AB> Air<AB> for ReducedSwirlTerminalTranscriptBridgeAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main
            .row_slice(0)
            .expect("reduced-SWIRL terminal transcript row");
        let next_row = main
            .row_slice(1)
            .expect("reduced-SWIRL terminal next transcript row");
        let local: &ReducedSwirlTerminalTranscriptBridgeCols<AB::Var> = (*local_row).borrow();
        let next: &ReducedSwirlTerminalTranscriptBridgeCols<AB::Var> = (*next_row).borrow();

        for flag in [local.active, local.is_first, local.is_last, local.is_sample] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder
            .when(local.active)
            .assert_eq(local.proof_idx + AB::F::ONE, local.call_count);
        builder
            .when(local.active * local.is_first)
            .assert_eq(local.tidx, local.footer_end_tidx);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.is_last);
        let mut transition = builder.when_transition();
        let mut same = transition.when(next.active);
        same.assert_zero(next.is_first);
        same.assert_eq(next.proof_idx, local.proof_idx);
        same.assert_eq(next.local_proof_idx, local.local_proof_idx);
        same.assert_eq(next.tidx, local.tidx + AB::F::ONE);
        same.assert_eq(next.source_count, local.source_count);
        same.assert_eq(next.call_count, local.call_count);
        same.assert_eq(next.footer_start_tidx, local.footer_start_tidx);
        same.assert_eq(next.footer_end_tidx, local.footer_end_tidx);
        assert_array_eq(&mut same, next.manifest_digest, local.manifest_digest);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);

        self.start_bus.receive(
            builder,
            ReducedSwirlTerminalStartMessage {
                source_count: local.source_count.into(),
                call_count: local.call_count.into(),
                footer_start_tidx: local.footer_start_tidx.into(),
                terminal_start_tidx: local.footer_end_tidx.into(),
                manifest_digest: local.manifest_digest.map(Into::into),
            },
            local.active * local.is_first,
        );
        let message = TranscriptBusMessage {
            tidx: local.tidx.into(),
            value: local.value.into(),
            is_sample: local.is_sample.into(),
        };
        self.main_transcript_bus.receive(
            builder,
            local.local_proof_idx,
            message.clone(),
            local.active,
        );
        self.semantic_transcript_bus
            .send(builder, AB::Expr::ZERO, message, local.active);
        self.transcript_end_index_bus.receive(
            builder,
            local.local_proof_idx,
            TranscriptEndIndexMessage {
                tidx: AB::Expr::from(local.tidx) + AB::Expr::ONE,
            },
            local.active * local.is_last,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ReducedSwirlTerminalStartCols<T> {
    pub active: T,
    pub source_count: T,
    pub call_count: T,
    pub footer_start_tidx: T,
    pub terminal_start_tidx: T,
    pub manifest_digest: [T; DIGEST_SIZE],
    pub root: [T; DIGEST_SIZE],
    pub mu: [T; D_EF],
    pub beta_last: [T; D_EF],
    pub eta: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(ReducedSwirlTerminalStartCols<u8>)]
pub struct ReducedSwirlTerminalStartAir {
    pub footer_bus: ReducedSwirlVaccFooterBus,
    pub start_bus: ReducedSwirlTerminalStartBus,
    pub root_bus: NativeTerminalAccumulatorRootBus,
    pub value_bus: NativeTerminalAccumulatorValueBus,
    pub linearizer_end_bus: NativeTerminalLinearizerEndBus,
    pub beta_len: usize,
}

impl BaseAirWithPublicValues<F> for ReducedSwirlTerminalStartAir {}
impl PartitionedBaseAir<F> for ReducedSwirlTerminalStartAir {}
impl BaseAir<F> for ReducedSwirlTerminalStartAir {
    fn width(&self) -> usize {
        ReducedSwirlTerminalStartCols::<F>::width()
    }
}

impl<AB> Air<AB> for ReducedSwirlTerminalStartAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("reduced-SWIRL terminal start row");
        let local: &ReducedSwirlTerminalStartCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_transition().assert_zero(local.active);
        self.footer_bus.receive(
            builder,
            ReducedSwirlVaccFooterMessage {
                source_count: local.source_count.into(),
                call_count: local.call_count.into(),
                start_tidx: local.footer_start_tidx.into(),
                end_tidx: local.terminal_start_tidx.into(),
                manifest_digest: local.manifest_digest.map(Into::into),
            },
            local.active,
        );
        self.start_bus.send(
            builder,
            ReducedSwirlTerminalStartMessage {
                source_count: local.source_count.into(),
                call_count: local.call_count.into(),
                footer_start_tidx: local.footer_start_tidx.into(),
                terminal_start_tidx: local.terminal_start_tidx.into(),
                manifest_digest: local.manifest_digest.map(Into::into),
            },
            local.active,
        );
        self.root_bus.receive(
            builder,
            NativeTerminalAccumulatorRootMessage {
                root: local.root.map(Into::into),
            },
            local.active,
        );
        for (section, coordinate, value) in [
            (1usize, 0usize, local.mu),
            (2usize, self.beta_len - 1, local.beta_last),
            (3usize, 0usize, local.eta),
        ] {
            self.value_bus.lookup_key(
                builder,
                NativeTerminalAccumulatorValueMessage {
                    section: AB::Expr::from_usize(section),
                    coordinate: AB::Expr::from_usize(coordinate),
                    value: value.map(Into::into),
                },
                local.active,
            );
        }
        self.linearizer_end_bus.send(
            builder,
            crate::native_warp::NativeTerminalLinearizerEndMessage {
                tidx: local.terminal_start_tidx.into(),
                root: local.root.map(Into::into),
                mu: local.mu.map(Into::into),
                beta_last: local.beta_last.map(Into::into),
                eta: local.eta.map(Into::into),
            },
            local.active,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ReducedSwirlTerminalAccumulatorValueCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub section: [T; 4],
    pub coordinate: T,
    pub remaining: T,
    pub section_last: T,
    pub section_last_inverse: T,
    pub ordinal: T,
    pub value: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(ReducedSwirlTerminalAccumulatorValueCols<u8>)]
pub struct ReducedSwirlTerminalAccumulatorValueAir {
    pub alpha_len: usize,
    pub beta_len: usize,
    pub value_bus: NativeTerminalAccumulatorValueBus,
    pub digest_element_bus: NativeAccumulatorDigestElementBus,
}

impl BaseAirWithPublicValues<F> for ReducedSwirlTerminalAccumulatorValueAir {}
impl PartitionedBaseAir<F> for ReducedSwirlTerminalAccumulatorValueAir {}
impl BaseAir<F> for ReducedSwirlTerminalAccumulatorValueAir {
    fn width(&self) -> usize {
        ReducedSwirlTerminalAccumulatorValueCols::<F>::width()
    }
}

impl<AB> Air<AB> for ReducedSwirlTerminalAccumulatorValueAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main
            .row_slice(0)
            .expect("reduced-SWIRL terminal instance row");
        let next_row = main
            .row_slice(1)
            .expect("reduced-SWIRL terminal next instance row");
        let local: &ReducedSwirlTerminalAccumulatorValueCols<AB::Var> = (*local_row).borrow();
        let next: &ReducedSwirlTerminalAccumulatorValueCols<AB::Var> = (*next_row).borrow();
        for flag in [
            local.active,
            local.is_first,
            local.is_last,
            local.section_last,
        ]
        .into_iter()
        .chain(local.section)
        {
            builder.assert_bool(flag);
        }
        let [is_alpha, is_mu, is_beta, is_eta] = local.section.map(AB::Expr::from);
        let section_sum = is_alpha.clone() + is_mu.clone() + is_beta.clone() + is_eta.clone();
        builder.when(local.active).assert_one(section_sum.clone());
        builder
            .when(AB::Expr::ONE - local.active)
            .assert_zero(section_sum);
        let section_len = is_alpha.clone() * AB::Expr::from_usize(self.alpha_len)
            + is_mu.clone()
            + is_beta.clone() * AB::Expr::from_usize(self.beta_len)
            + is_eta.clone();
        builder
            .when(local.active)
            .assert_eq(local.coordinate + local.remaining, section_len);
        let rem = local.remaining - AB::Expr::ONE;
        builder
            .when(local.active * local.section_last)
            .assert_zero(rem.clone());
        builder
            .when(local.active * (AB::Expr::ONE - local.section_last))
            .assert_one(rem * local.section_last_inverse);
        builder
            .when(local.active)
            .assert_eq(local.is_last, is_eta.clone() * local.section_last);
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when_first_row().assert_one(local.section[0]);
        builder.when_first_row().assert_zero(local.coordinate);
        builder.when_first_row().assert_zero(local.ordinal);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder.when_transition().assert_eq(
            local.active - next.active,
            local.is_last * (AB::Expr::ONE - next.active),
        );
        let mut transition = builder.when_transition();
        let mut active_next = transition.when(next.active);
        active_next.assert_zero(next.is_first);
        active_next.assert_eq(next.ordinal, local.ordinal + AB::F::ONE);
        let mut transition = builder.when_transition();
        let mut same = transition.when(next.active * (AB::Expr::ONE - local.section_last));
        for index in 0..4 {
            same.assert_eq(next.section[index], local.section[index]);
        }
        same.assert_eq(next.coordinate, local.coordinate + AB::F::ONE);
        same.assert_eq(next.remaining, local.remaining - AB::F::ONE);
        let mut transition = builder.when_transition();
        let mut advance = transition.when(next.active * local.section_last);
        advance.assert_zero(next.coordinate);
        advance.assert_zero(next.section[0]);
        advance.assert_eq(next.section[1], is_alpha.clone());
        advance.assert_eq(next.section[2], is_mu.clone());
        advance.assert_eq(next.section[3], is_beta.clone());
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);

        let section = is_mu.clone()
            + is_beta.clone() * AB::Expr::from_u32(2)
            + is_eta.clone() * AB::Expr::from_u32(3);
        // Every coordinate has exactly one semantic consumer: Q for alpha,
        // terminal start for mu/beta-last/eta, and Eq for beta's point.
        let multiplicity = is_alpha + is_mu + is_beta + is_eta;
        self.value_bus.add_key_with_lookups(
            builder,
            NativeTerminalAccumulatorValueMessage {
                section,
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active * multiplicity,
        );
        for limb in 0..D_EF {
            self.digest_element_bus.send(
                builder,
                NativeAccumulatorDigestElementMessage {
                    proof_idx: AB::Expr::ZERO,
                    state: AB::Expr::ZERO,
                    index: AB::Expr::from_usize(3)
                        + local.ordinal * AB::Expr::from_usize(D_EF)
                        + AB::Expr::from_usize(limb),
                    value: local.value[limb].into(),
                },
                local.active,
            );
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ReducedSwirlTerminalAccumulatorHashCols<T> {
    pub active: T,
    pub is_first: T,
    pub is_last: T,
    pub chunk_index: T,
    pub elements: [T; HASH_RATE],
    pub capacity: [T; DIGEST_SIZE],
    pub permutation_output: [T; POSEIDON2_WIDTH],
    pub final_digest: [T; DIGEST_SIZE],
}

#[derive(ColumnsAir)]
#[columns_via(ReducedSwirlTerminalAccumulatorHashCols<u8>)]
pub struct ReducedSwirlTerminalAccumulatorHashAir {
    pub alpha_len: usize,
    pub beta_len: usize,
    pub digest_element_bus: NativeAccumulatorDigestElementBus,
    pub algebraic_digest_bus: NativeAccumulatorAlgebraicDigestBus,
    pub permute_bus: Poseidon2PermuteBus,
    pub compress_bus: Poseidon2CompressBus,
}

impl ReducedSwirlTerminalAccumulatorHashAir {
    fn preimage_len(&self) -> usize {
        3 + (self.alpha_len + self.beta_len + 2) * D_EF
    }

    fn chunk_count(&self) -> usize {
        self.preimage_len().div_ceil(HASH_RATE)
    }
}

impl BaseAirWithPublicValues<F> for ReducedSwirlTerminalAccumulatorHashAir {}
impl PartitionedBaseAir<F> for ReducedSwirlTerminalAccumulatorHashAir {}
impl BaseAir<F> for ReducedSwirlTerminalAccumulatorHashAir {
    fn width(&self) -> usize {
        ReducedSwirlTerminalAccumulatorHashCols::<F>::width()
    }
}

impl<AB> Air<AB> for ReducedSwirlTerminalAccumulatorHashAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("reduced-SWIRL terminal hash row");
        let next_row = main
            .row_slice(1)
            .expect("reduced-SWIRL terminal next hash row");
        let local: &ReducedSwirlTerminalAccumulatorHashCols<AB::Var> = (*local_row).borrow();
        let next: &ReducedSwirlTerminalAccumulatorHashCols<AB::Var> = (*next_row).borrow();
        for flag in [local.active, local.is_first, local.is_last] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when_first_row().assert_zero(local.chunk_index);
        for value in local.capacity {
            builder.when_first_row().assert_zero(value);
        }
        builder.when(local.active * local.is_last).assert_eq(
            local.chunk_index,
            AB::Expr::from_usize(self.chunk_count() - 1),
        );
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .assert_eq(local.active - next.active, local.is_last);
        let mut transition = builder.when_transition();
        let mut same = transition.when(next.active);
        same.assert_zero(next.is_first);
        same.assert_eq(next.chunk_index, local.chunk_index + AB::F::ONE);
        for limb in 0..DIGEST_SIZE {
            same.assert_eq(
                next.capacity[limb],
                local.permutation_output[DIGEST_SIZE + limb],
            );
        }
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);

        let final_remainder = match self.preimage_len() % HASH_RATE {
            0 => HASH_RATE,
            value => value,
        };
        let metadata = [
            F::from_u64(NATIVE_ACCUMULATOR_INSTANCE_TAG),
            F::from_usize(self.alpha_len),
            F::from_usize(self.beta_len),
        ];
        for offset in 0..HASH_RATE {
            if offset < metadata.len() {
                builder
                    .when(local.active * local.is_first)
                    .assert_eq(local.elements[offset], metadata[offset]);
            }
            if offset >= final_remainder {
                builder
                    .when_transition()
                    .when(next.active * next.is_last)
                    .assert_eq(next.elements[offset], local.permutation_output[offset]);
            }
            let metadata_here = if offset < metadata.len() {
                AB::Expr::from(local.is_first)
            } else {
                AB::Expr::ZERO
            };
            let padding_here = if offset >= final_remainder {
                AB::Expr::from(local.is_last)
            } else {
                AB::Expr::ZERO
            };
            self.digest_element_bus.receive(
                builder,
                NativeAccumulatorDigestElementMessage {
                    proof_idx: AB::Expr::ZERO,
                    state: AB::Expr::ZERO,
                    index: local.chunk_index * AB::Expr::from_usize(HASH_RATE)
                        + AB::Expr::from_usize(offset),
                    value: local.elements[offset].into(),
                },
                local.active * (AB::Expr::ONE - metadata_here) * (AB::Expr::ONE - padding_here),
            );
        }
        let poseidon_input = core::array::from_fn(|index| {
            if index < HASH_RATE {
                local.elements[index].into()
            } else {
                local.capacity[index - HASH_RATE].into()
            }
        });
        self.permute_bus.lookup_key(
            builder,
            Poseidon2PermuteMessage {
                input: poseidon_input.clone(),
                output: local.permutation_output.map(Into::into),
            },
            local.active * (AB::Expr::ONE - local.is_last),
        );
        self.compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: poseidon_input,
                output: local.final_digest.map(Into::into),
            },
            local.active * local.is_last,
        );
        self.algebraic_digest_bus.send(
            builder,
            NativeAccumulatorAlgebraicDigestMessage {
                proof_idx: AB::Expr::ZERO,
                state: AB::Expr::ZERO,
                digest: local.final_digest.map(Into::into),
            },
            local.active * local.is_last,
        );
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct ReducedSwirlTerminalReceiptCols<T> {
    pub active: T,
    pub root: [T; DIGEST_SIZE],
    pub algebraic_digest: [T; DIGEST_SIZE],
    pub instance_digest: [T; DIGEST_SIZE],
}

#[derive(ColumnsAir)]
#[columns_via(ReducedSwirlTerminalReceiptCols<u8>)]
pub struct ReducedSwirlTerminalReceiptAir {
    pub profile: ReducedSwirlTerminalProfile,
    /// Typed view over the exact bus index owned by the wrapper.
    pub wrapper_receipt_bus: ReducedSwirlLocalTerminalReceiptBus,
    pub terminal_root_bus: NativeTerminalAccumulatorRootBus,
    pub algebraic_digest_bus: NativeAccumulatorAlgebraicDigestBus,
    pub compress_bus: Poseidon2CompressBus,
}

impl BaseAirWithPublicValues<F> for ReducedSwirlTerminalReceiptAir {}
impl PartitionedBaseAir<F> for ReducedSwirlTerminalReceiptAir {}
impl BaseAir<F> for ReducedSwirlTerminalReceiptAir {
    fn width(&self) -> usize {
        ReducedSwirlTerminalReceiptCols::<F>::width()
    }
}

impl<AB> Air<AB> for ReducedSwirlTerminalReceiptAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("reduced-SWIRL terminal receipt row");
        let local: &ReducedSwirlTerminalReceiptCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_transition().assert_zero(local.active);
        self.terminal_root_bus.send(
            builder,
            NativeTerminalAccumulatorRootMessage {
                root: local.root.map(Into::into),
            },
            local.active,
        );
        self.algebraic_digest_bus.receive(
            builder,
            NativeAccumulatorAlgebraicDigestMessage {
                proof_idx: AB::Expr::ZERO,
                state: AB::Expr::ZERO,
                digest: local.algebraic_digest.map(Into::into),
            },
            local.active,
        );
        self.compress_bus.lookup_key(
            builder,
            Poseidon2CompressMessage {
                input: core::array::from_fn(|index| {
                    if index < DIGEST_SIZE {
                        local.root[index].into()
                    } else {
                        local.algebraic_digest[index - DIGEST_SIZE].into()
                    }
                }),
                output: local.instance_digest.map(Into::into),
            },
            local.active,
        );
        self.wrapper_receipt_bus.add_key_with_lookups(
            builder,
            ReducedSwirlLocalTerminalReceiptMessage {
                protocol_digest: self.profile.setup.protocol_digest.map(Into::into),
                relation_digest: self.profile.setup.relation_digest.map(Into::into),
                terminal_index_digest: self.profile.setup.terminal_index_digest.map(Into::into),
                verifier_component_digest: self.profile.verifier_component_digest.map(Into::into),
                final_accumulator_digest: local.instance_digest.map(Into::into),
                final_accumulator_root: local.root.map(Into::into),
            },
            local.active,
        );
    }
}

/// Setup-fixed terminal component. Generic terminal AIRs use
/// `buses.semantic_transcript`; the bridge is the sole route from that local
/// namespace to the wrapper/VACC transcript.
pub struct ReducedSwirlTerminalComponent {
    pub profile: ReducedSwirlTerminalProfile,
    pub buses: ReducedSwirlTerminalBusInventory,
    pub main_transcript_bus: TranscriptBus,
    pub transcript_end_index_bus: TranscriptEndIndexBus,
    pub vacc_footer_bus: ReducedSwirlVaccFooterBus,
    pub wrapper_terminal_receipt_bus_idx: BusIndex,
    pub poseidon: Poseidon2BusOwner,
    component_protocol_digest: Digest,
    component_digest_material: Vec<F>,
    config: NativeSC,
}

impl ReducedSwirlTerminalComponent {
    pub const COMPONENT_COUNT: usize = 26;

    #[allow(clippy::too_many_arguments)]
    pub fn new(
        profile: ReducedSwirlTerminalProfile,
        shared: &BusInventory,
        main_transcript_bus: TranscriptBus,
        vacc_footer_bus: ReducedSwirlVaccFooterBus,
        wrapper_terminal_receipt_bus_idx: BusIndex,
        first_internal_bus_idx: BusIndex,
        system_params: SystemParams,
    ) -> Result<Self, ReducedSwirlTerminalError> {
        let buses = ReducedSwirlTerminalBusInventory::new(first_internal_bus_idx);
        let component_digest_material = reduced_swirl_terminal_component_material(
            &profile.setup,
            &buses,
            shared,
            main_transcript_bus,
            vacc_footer_bus,
            wrapper_terminal_receipt_bus_idx,
        )?;
        let component_protocol_digest = fixed_material_digest(
            REDUCED_SWIRL_TERMINAL_COMPONENT_TAG,
            &component_digest_material,
        );
        let poseidon = Poseidon2BusOwner {
            permute_bus: shared.poseidon2_permute_bus,
            compress_bus: shared.poseidon2_compress_bus,
        };
        let config = NativeSC::default_from_params(system_params);
        Ok(Self {
            profile,
            buses,
            main_transcript_bus,
            transcript_end_index_bus: shared.transcript_end_index_bus,
            vacc_footer_bus,
            wrapper_terminal_receipt_bus_idx,
            poseidon,
            component_protocol_digest,
            component_digest_material,
            config,
        })
    }

    pub fn protocol_digest(&self) -> Digest {
        // Deliberately excludes profile.verifier_component_digest, while
        // binding every interaction namespace used by these AIRs.
        self.component_protocol_digest
    }

    #[must_use]
    pub const fn component_count(&self) -> usize {
        Self::COMPONENT_COUNT
    }

    #[must_use]
    pub fn component_digest_material(&self) -> Vec<F> {
        self.component_digest_material.clone()
    }

    #[must_use]
    pub const fn wrapper_terminal_receipt_bus_index(&self) -> BusIndex {
        self.wrapper_terminal_receipt_bus_idx
    }

    fn round_air(&self) -> Result<NativeTerminalWhirRoundAir, ReducedSwirlTerminalError> {
        let p = &self.profile.setup;
        NativeTerminalWhirRoundAir::new(
            self.buses.semantic_transcript,
            self.buses.whir_statement,
            self.buses.whir_round,
            self.buses.whir_verify_queries,
            self.buses.whir_final_claim,
            self.buses.weight_term,
            self.buses.merkle_root,
            self.buses.exp_bits_len,
            p.k,
            p.log_codeword_len,
            p.final_poly_len,
            p.query_pow_bits,
            p.folding_pow_bits,
            F::GENERATOR,
            0,
            p.num_queries_per_round.clone(),
        )
        .map_err(|_| ReducedSwirlTerminalError::Profile("WHIR round AIR"))
    }

    fn point_air(&self) -> NativeTerminalWhirPointAir {
        let p = &self.profile.setup;
        NativeTerminalWhirPointAir::new(
            self.buses.semantic_transcript,
            self.buses.whir_alpha,
            self.buses.whir_final_context,
            self.buses.whir_point,
            p.k,
            p.round_count(),
            p.final_poly_len,
            p.point_lookup_counts.clone(),
        )
    }

    fn adjoint_sumcheck_air(&self) -> NativeTerminalRsAdjointSumcheckAir {
        let p = &self.profile.setup;
        NativeTerminalRsAdjointSumcheckAir::new(
            self.buses.semantic_transcript,
            self.buses.adjoint_round,
            self.buses.adjoint_claim,
            p.log_codeword_len,
            p.rs_adjoint_degree,
        )
    }

    fn adjoint_y_air(&self) -> NativeTerminalRsAdjointYAir {
        let p = &self.profile.setup;
        NativeTerminalRsAdjointYAir::new(
            self.buses.adjoint_round,
            self.buses.adjoint_y,
            p.log_message_len,
            p.log_codeword_len,
        )
    }

    /// Semantic AIR order, shared exactly by `generate_traces` and
    /// `generate_cpu_contexts`.
    pub fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        let p = &self.profile.setup;
        let b = &self.buses;
        let mut airs = Vec::with_capacity(Self::COMPONENT_COUNT);
        add_air(
            &mut airs,
            ReducedSwirlTerminalTranscriptBridgeAir {
                main_transcript_bus: self.main_transcript_bus,
                transcript_end_index_bus: self.transcript_end_index_bus,
                semantic_transcript_bus: b.semantic_transcript,
                start_bus: b.terminal_start,
            },
        );
        add_air(
            &mut airs,
            ReducedSwirlTerminalStartAir {
                footer_bus: self.vacc_footer_bus,
                start_bus: b.terminal_start,
                root_bus: b.accumulator_root,
                value_bus: b.accumulator_value,
                linearizer_end_bus: b.linearizer_end,
                beta_len: p.beta_len,
            },
        );
        add_air(
            &mut airs,
            NativeTerminalWhirStatementAir {
                transcript_bus: b.semantic_transcript,
                linearizer_end_bus: b.linearizer_end,
                statement_bus: b.whir_statement,
                transcript_binding: p.transcript_binding.clone(),
            },
        );
        add_air(
            &mut airs,
            NativeTerminalWhirLinearizerWeightAir {
                statement_bus: b.whir_statement,
                accumulator_value_bus: b.accumulator_value,
                point_bus: b.whir_point,
                output_bus: b.linearizer_weight,
                log_message_len: p.log_message_len,
            },
        );
        add_air(
            &mut airs,
            NativeTerminalWhirSumcheckAir {
                transcript_bus: b.semantic_transcript,
                round_bus: b.whir_round,
                alpha_bus: b.whir_alpha,
                exp_bits_len_bus: b.exp_bits_len,
                k: p.k,
                round_count: p.round_count(),
                folding_pow_bits: p.folding_pow_bits,
                generator: F::GENERATOR,
            },
        );
        add_air(
            &mut airs,
            self.round_air()
                .expect("validated reduced-SWIRL WHIR round"),
        );
        add_air(
            &mut airs,
            NativeTerminalWhirQueryAir {
                transcript_bus: b.semantic_transcript,
                verify_queries_bus: b.whir_verify_queries,
                query_bus: b.whir_query,
                weight_term_bus: b.weight_term,
                exp_bits_len_bus: b.exp_bits_len,
                right_shift_bus: b.right_shift,
                k: p.k,
                initial_log_domain_size: p.log_codeword_len,
                round_count: p.round_count(),
                final_poly_len: p.final_poly_len,
                inner_tree_id_offset: p.round_count(),
                outer_tree_id_offset: 0,
                evaluation_layout: true,
            },
        );
        add_air(
            &mut airs,
            NativeTerminalWhirOpenedAir {
                query_bus: b.whir_query,
                folding_bus: b.whir_folding,
                leaf_value_bus: b.leaf_value,
                opening_leaf_bus: b.opening_leaf,
                k: p.k,
                inner_depth: p.k,
            },
        );
        add_air(
            &mut airs,
            NativeTerminalWhirFoldingAir {
                alpha_bus: b.whir_alpha,
                folding_bus: b.whir_folding,
                k: p.k,
                evaluation_layout: true,
            },
        );
        add_air(
            &mut airs,
            NativeLeafHashAir {
                permute_bus: self.poseidon.permute_bus,
                value_bus: b.leaf_value,
                leaf_bus: b.opening_leaf,
            },
        );
        add_air(
            &mut airs,
            NativeMerkleMultiproofAir {
                compress_bus: self.poseidon.compress_bus,
                leaf_bus: b.opening_leaf,
                node_bus: b.merkle_node,
                root_bus: b.merkle_root,
            },
        );
        add_air(
            &mut airs,
            NativeMerkleLeafAdapterAir {
                leaf_bus: b.opening_leaf,
                root_bus: b.merkle_root,
                inner_depth: p.k,
            },
        );
        add_air(
            &mut airs,
            NativeTerminalWhirFinalTableAir {
                transcript_bus: b.semantic_transcript,
                final_claim_bus: b.whir_final_claim,
                final_context_bus: b.whir_final_context,
                final_poly_bus: b.whir_final_poly,
                final_weight_bus: b.whir_final_weight,
                final_len: p.final_poly_len,
            },
        );
        add_air(
            &mut airs,
            NativeTerminalWhirMobiusAir::new(b.whir_final_poly, p.final_log_len()),
        );
        add_air(&mut airs, self.point_air());
        add_air(
            &mut airs,
            NativeTerminalWhirFinalCheckAir {
                final_context_bus: b.whir_final_context,
                final_poly_bus: b.whir_final_poly,
                final_weight_bus: b.whir_final_weight,
                point_bus: b.whir_point,
                actual_weight_bus: b.whir_actual_weight,
                point_prefix_len: p.k * p.round_count(),
                final_log_len: p.final_log_len(),
            },
        );
        add_air(
            &mut airs,
            NativeTerminalWhirWeightTermAir {
                term_bus: b.weight_term,
                result_bus: b.weight_term_result,
                point_bus: b.whir_point,
            },
        );
        add_air(&mut airs, self.adjoint_sumcheck_air());
        add_air(
            &mut airs,
            NativeTerminalRsAdjointQAir {
                round_bus: b.adjoint_round,
                accumulator_value_bus: b.accumulator_value,
                point_bus: b.whir_point,
                q_bus: b.adjoint_q,
                round_count: p.log_codeword_len,
                initial_folding_factor: 0,
            },
        );
        add_air(&mut airs, self.adjoint_y_air());
        add_air(
            &mut airs,
            NativeTerminalRsAdjointSelectorAir {
                point_bus: b.whir_point,
                q_bus: b.adjoint_q,
                y_bus: b.adjoint_y,
                claim_bus: b.adjoint_claim,
                value_bus: b.adjoint_value,
                log_message_len: p.log_message_len,
                point_coordinate_offset: 0,
            },
        );
        add_air(
            &mut airs,
            NativeTerminalWhirExpectedWeightAir {
                actual_bus: b.whir_actual_weight,
                adjoint_bus: b.adjoint_value,
                linearizer_bus: b.linearizer_weight,
                term_bus: b.weight_term_result,
                term_count: p.weight_term_count,
            },
        );
        add_air(&mut airs, ExpBitsLenAir::new(b.exp_bits_len, b.right_shift));
        add_air(
            &mut airs,
            ReducedSwirlTerminalAccumulatorValueAir {
                alpha_len: p.alpha_len,
                beta_len: p.beta_len,
                value_bus: b.accumulator_value,
                digest_element_bus: b.accumulator_digest_element,
            },
        );
        add_air(
            &mut airs,
            ReducedSwirlTerminalAccumulatorHashAir {
                alpha_len: p.alpha_len,
                beta_len: p.beta_len,
                digest_element_bus: b.accumulator_digest_element,
                algebraic_digest_bus: b.accumulator_algebraic_digest,
                permute_bus: self.poseidon.permute_bus,
                compress_bus: self.poseidon.compress_bus,
            },
        );
        add_air(
            &mut airs,
            ReducedSwirlTerminalReceiptAir {
                profile: self.profile.clone(),
                wrapper_receipt_bus: ReducedSwirlLocalTerminalReceiptBus::new(
                    self.wrapper_terminal_receipt_bus_idx,
                ),
                terminal_root_bus: b.accumulator_root,
                algebraic_digest_bus: b.accumulator_algebraic_digest,
                compress_bus: self.poseidon.compress_bus,
            },
        );
        debug_assert_eq!(airs.len(), Self::COMPONENT_COUNT);
        airs
    }
}

fn add_air<SC, A>(airs: &mut Vec<AirRef<SC>>, air: A)
where
    SC: StarkProtocolConfig,
    A: openvm_stark_backend::AnyAir<SC> + 'static,
{
    airs.push(Arc::new(air));
}

pub struct ReducedSwirlTerminalTraceData {
    /// One common-main matrix per `ReducedSwirlTerminalComponent::airs` entry.
    pub traces: Vec<RowMajorMatrix<F>>,
    /// Requests to merge into the wrapper's shared Poseidon table.
    pub poseidon2_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub poseidon2_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub receipt: ReducedSwirlTerminalReceiptRecord,
}

pub struct ReducedSwirlTerminalCpuPacket<SC: StarkProtocolConfig<F = F>> {
    pub contexts: Vec<AirProvingContext<CpuBackend<SC>>>,
    pub poseidon2_permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub poseidon2_compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub receipt: ReducedSwirlTerminalReceiptRecord,
}

struct ReducedSwirlAccumulatorDigestTraces {
    values: RowMajorMatrix<F>,
    hash: RowMajorMatrix<F>,
    receipt: RowMajorMatrix<F>,
    permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    instance_digest: Digest,
}

fn generate_terminal_transcript_bridge_trace(
    footer: &ReducedSwirlTerminalFooterRecord,
    transcript: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
) -> Result<RowMajorMatrix<F>, ReducedSwirlTerminalError> {
    if footer.source_count == 0
        || footer.call_count == 0
        || footer.proof_idx + 1 != footer.call_count
        || footer.manifest_digest.iter().all(|value| *value == F::ZERO)
        || footer.start_tidx >= footer.end_tidx
        || transcript.values().len() != transcript.samples().len()
        || footer.end_tidx >= transcript.values().len()
    {
        return Err(ReducedSwirlTerminalError::Transcript(
            "VACC footer boundary",
        ));
    }
    let footer_elements =
        reduced_swirl_vacc_footer_elements(footer.source_count, footer.manifest_digest)
            .map_err(|_| ReducedSwirlTerminalError::Transcript("VACC footer encoding"))?;
    let flattened_footer = footer_elements
        .iter()
        .flat_map(|value| value.as_basis_coefficients_slice().iter().copied())
        .collect::<Vec<_>>();
    if footer.end_tidx != footer.start_tidx + flattened_footer.len()
        || transcript.values().get(footer.start_tidx..footer.end_tidx)
            != Some(flattened_footer.as_slice())
        || transcript
            .samples()
            .get(footer.start_tidx..footer.end_tidx)
            .is_none_or(|samples| samples.iter().any(|&sample| sample))
    {
        return Err(ReducedSwirlTerminalError::Transcript(
            "VACC footer transcript",
        ));
    }
    let valid_rows = transcript.values().len() - footer.end_tidx;
    let height = valid_rows.next_power_of_two();
    let width = ReducedSwirlTerminalTranscriptBridgeCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    for row_index in 0..valid_rows {
        let tidx = footer.end_tidx + row_index;
        let row = &mut values[row_index * width..(row_index + 1) * width];
        let cols: &mut ReducedSwirlTerminalTranscriptBridgeCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.proof_idx = F::from_usize(footer.proof_idx);
        cols.local_proof_idx = F::from_usize(footer.local_proof_idx);
        cols.tidx = F::from_usize(tidx);
        cols.is_first = F::from_bool(row_index == 0);
        cols.is_last = F::from_bool(row_index + 1 == valid_rows);
        cols.source_count = F::from_usize(footer.source_count);
        cols.call_count = F::from_usize(footer.call_count);
        cols.footer_start_tidx = F::from_usize(footer.start_tidx);
        cols.footer_end_tidx = F::from_usize(footer.end_tidx);
        cols.manifest_digest = footer.manifest_digest;
        cols.value = transcript.values()[tidx];
        cols.is_sample = F::from_bool(transcript.samples()[tidx]);
    }
    Ok(RowMajorMatrix::new(values, width))
}

fn generate_terminal_start_trace(
    footer: &ReducedSwirlTerminalFooterRecord,
    instance: &AccumulatorInstance<EF, Digest>,
) -> Result<RowMajorMatrix<F>, ReducedSwirlTerminalError> {
    let beta_last = *instance
        .beta
        .last()
        .ok_or(ReducedSwirlTerminalError::Witness("beta target"))?;
    let width = ReducedSwirlTerminalStartCols::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut ReducedSwirlTerminalStartCols<F> = values.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.source_count = F::from_usize(footer.source_count);
    cols.call_count = F::from_usize(footer.call_count);
    cols.footer_start_tidx = F::from_usize(footer.start_tidx);
    cols.terminal_start_tidx = F::from_usize(footer.end_tidx);
    cols.manifest_digest = footer.manifest_digest;
    cols.root = instance.rt;
    cols.mu
        .copy_from_slice(instance.mu.as_basis_coefficients_slice());
    cols.beta_last
        .copy_from_slice(beta_last.as_basis_coefficients_slice());
    cols.eta
        .copy_from_slice(instance.eta.as_basis_coefficients_slice());
    Ok(RowMajorMatrix::new(values, width))
}

fn generate_accumulator_digest_traces(
    instance: &AccumulatorInstance<EF, Digest>,
    alpha_len: usize,
    beta_len: usize,
) -> Result<ReducedSwirlAccumulatorDigestTraces, ReducedSwirlTerminalError> {
    if instance.alpha.len() != alpha_len || instance.beta.len() != beta_len {
        return Err(ReducedSwirlTerminalError::Witness("accumulator dimensions"));
    }
    let extension_values = instance
        .alpha
        .iter()
        .copied()
        .chain(core::iter::once(instance.mu))
        .chain(instance.beta.iter().copied())
        .chain(core::iter::once(instance.eta))
        .collect::<Vec<_>>();
    let valid_rows = extension_values.len();
    let width = ReducedSwirlTerminalAccumulatorValueCols::<F>::width();
    let mut value_trace = F::zero_vec(valid_rows.next_power_of_two() * width);
    for (ordinal, value) in extension_values.iter().enumerate() {
        let (section, coordinate, section_len) = if ordinal < alpha_len {
            (0, ordinal, alpha_len)
        } else if ordinal == alpha_len {
            (1, 0, 1)
        } else if ordinal < alpha_len + 1 + beta_len {
            (2, ordinal - alpha_len - 1, beta_len)
        } else {
            (3, 0, 1)
        };
        let row = &mut value_trace[ordinal * width..(ordinal + 1) * width];
        let cols: &mut ReducedSwirlTerminalAccumulatorValueCols<F> = row.borrow_mut();
        let remaining = section_len - coordinate;
        cols.active = F::ONE;
        cols.is_first = F::from_bool(ordinal == 0);
        cols.is_last = F::from_bool(ordinal + 1 == valid_rows);
        cols.section[section] = F::ONE;
        cols.coordinate = F::from_usize(coordinate);
        cols.remaining = F::from_usize(remaining);
        cols.section_last = F::from_bool(remaining == 1);
        if remaining != 1 {
            cols.section_last_inverse = F::from_usize(remaining - 1).inverse();
        }
        cols.ordinal = F::from_usize(ordinal);
        cols.value
            .copy_from_slice(value.as_basis_coefficients_slice());
    }

    let preimage = native_accumulator_instance_digest_preimage::<NativeSC>(instance);
    let (algebraic_digest, pre_states, post_states) = poseidon2_hash_slice_with_states(&preimage);
    let chunk_count = pre_states.len();
    if chunk_count == 0 {
        return Err(ReducedSwirlTerminalError::Trace("accumulator hash chunks"));
    }
    let hash_width = ReducedSwirlTerminalAccumulatorHashCols::<F>::width();
    let mut hash_trace = F::zero_vec(chunk_count.next_power_of_two() * hash_width);
    for chunk in 0..chunk_count {
        let row = &mut hash_trace[chunk * hash_width..(chunk + 1) * hash_width];
        let cols: &mut ReducedSwirlTerminalAccumulatorHashCols<F> = row.borrow_mut();
        cols.active = F::ONE;
        cols.is_first = F::from_bool(chunk == 0);
        cols.is_last = F::from_bool(chunk + 1 == chunk_count);
        cols.chunk_index = F::from_usize(chunk);
        cols.elements
            .copy_from_slice(&pre_states[chunk][..HASH_RATE]);
        cols.capacity
            .copy_from_slice(&pre_states[chunk][HASH_RATE..]);
        if chunk + 1 == chunk_count {
            cols.final_digest = algebraic_digest;
        } else {
            cols.permutation_output = post_states[chunk];
        }
    }
    let instance_digest = poseidon2_compress_with_capacity(instance.rt, algebraic_digest).0;
    let receipt_width = ReducedSwirlTerminalReceiptCols::<F>::width();
    let mut receipt_trace = F::zero_vec(receipt_width);
    let receipt_cols: &mut ReducedSwirlTerminalReceiptCols<F> =
        receipt_trace.as_mut_slice().borrow_mut();
    receipt_cols.active = F::ONE;
    receipt_cols.root = instance.rt;
    receipt_cols.algebraic_digest = algebraic_digest;
    receipt_cols.instance_digest = instance_digest;
    let mut compression_inputs = vec![pre_states[chunk_count - 1]];
    compression_inputs.push(core::array::from_fn(|index| {
        if index < DIGEST_SIZE {
            instance.rt[index]
        } else {
            algebraic_digest[index - DIGEST_SIZE]
        }
    }));
    Ok(ReducedSwirlAccumulatorDigestTraces {
        values: RowMajorMatrix::new(value_trace, width),
        hash: RowMajorMatrix::new(hash_trace, hash_width),
        receipt: RowMajorMatrix::new(receipt_trace, receipt_width),
        permutation_inputs: pre_states[..chunk_count - 1].to_vec(),
        compression_inputs,
        instance_digest,
    })
}

impl ReducedSwirlTerminalComponent {
    pub fn generate_traces(
        &self,
        record: ReducedSwirlTerminalRecord<'_>,
    ) -> Result<ReducedSwirlTerminalTraceData, ReducedSwirlTerminalError> {
        let p = &self.profile.setup;
        let instance = record.instance;
        let verification = record.verification;
        if record
            .descriptor
            .metadata_words()
            .ne(p.descriptor_words.iter().copied())
            || record.descriptor.root != instance.rt
            || verification.root != instance.rt
            || instance.alpha.len() != p.alpha_len
            || instance.beta.len() != p.beta_len
            || record.footer.proof_idx + 1 != record.footer.call_count
            || verification.transcript_end.operations != record.transcript.values().len()
        {
            return Err(ReducedSwirlTerminalError::Witness("terminal binding"));
        }
        validate_native_terminal_eq_statement(&instance.beta, instance.eta, record.statement)
            .map_err(|_| ReducedSwirlTerminalError::Witness("corrected Eq relation"))?;

        if verification.transcript_start.operations != record.footer.end_tidx {
            return Err(ReducedSwirlTerminalError::Transcript(
                "terminal does not immediately follow manifest footer",
            ));
        }

        let bridge = generate_terminal_transcript_bridge_trace(record.footer, record.transcript)?;
        let start = generate_terminal_start_trace(record.footer, instance)?;
        let whir_statement = generate_native_terminal_whir_statement_trace(
            record.descriptor,
            &p.transcript_binding,
            instance.mu,
            *instance
                .beta
                .last()
                .ok_or(ReducedSwirlTerminalError::Witness("beta target"))?,
            instance.eta,
            verification,
            record.transcript,
        )
        .ok_or(ReducedSwirlTerminalError::Trace("WHIR statement"))?;
        let linearizer_weight = generate_native_terminal_whir_linearizer_weight_trace(
            &instance.beta,
            verification,
            None,
        )
        .ok_or(ReducedSwirlTerminalError::Trace("Eq linearizer"))?;

        let whir_sumcheck = generate_native_terminal_whir_sumcheck_trace(
            verification,
            record.transcript,
            p.folding_pow_bits,
            1,
            None,
        )
        .ok_or(ReducedSwirlTerminalError::Trace("WHIR sumcheck"))?;
        let round_air = self.round_air()?;
        let whir_round = generate_native_terminal_whir_round_trace(
            &round_air,
            verification,
            record.transcript,
            None,
        )
        .ok_or(ReducedSwirlTerminalError::Trace("WHIR round"))?;
        let whir_query = generate_native_terminal_whir_query_trace(
            verification,
            record.transcript,
            p.k,
            p.log_codeword_len,
            p.round_count(),
            0,
            p.round_count(),
            p.final_poly_len,
            openvm_stark_backend::warp_accum::terminal_whir::TerminalWhirLayout::ScalarRs,
            None,
        )
        .ok_or(ReducedSwirlTerminalError::Trace("WHIR query"))?;
        let whir_opened = generate_native_terminal_whir_opened_trace(
            self.config.hasher(),
            verification,
            record.transcript,
            p.k,
            p.log_codeword_len,
            p.round_count(),
            0,
            None,
            openvm_stark_backend::warp_accum::terminal_whir::TerminalWhirLayout::ScalarRs,
        )
        .ok_or(ReducedSwirlTerminalError::Trace("WHIR opened rows"))?;
        let whir_folding = generate_native_terminal_whir_folding_trace(
            verification,
            p.k,
            None,
            openvm_stark_backend::warp_accum::terminal_whir::TerminalWhirLayout::ScalarRs,
        )
        .ok_or(ReducedSwirlTerminalError::Trace("WHIR folding"))?;

        let leaf_inputs = whir_opened
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
            .ok_or(ReducedSwirlTerminalError::Trace("WHIR leaf hashes"))?;
        let mut merkle_records = whir_opened
            .inner_merkle
            .iter()
            .map(|(tree, record)| (*tree, record))
            .collect::<Vec<(u32, &BinaryMerkleMultiproofRecord<Digest>)>>();
        merkle_records.extend(
            verification
                .rounds
                .iter()
                .enumerate()
                .map(|(round, verification)| (round as u32, &verification.multiproof)),
        );
        let merkle = generate_native_merkle_multiproof_trace(0, &merkle_records, None)
            .ok_or(ReducedSwirlTerminalError::Trace("WHIR Merkle multiproof"))?;
        let leaf_adapter =
            generate_native_merkle_leaf_adapter_trace(&whir_opened.leaf_adapters, p.k, None)
                .ok_or(ReducedSwirlTerminalError::Trace("WHIR leaf adapter"))?;

        let final_table = generate_native_terminal_whir_final_table_trace(verification, None)
            .ok_or(ReducedSwirlTerminalError::Trace("WHIR final table"))?;
        let mobius_air =
            NativeTerminalWhirMobiusAir::new(self.buses.whir_final_poly, p.final_log_len());
        let mobius = generate_native_terminal_whir_mobius_trace(&mobius_air, verification, None)
            .ok_or(ReducedSwirlTerminalError::Trace("WHIR Mobius"))?;
        let point_air = self.point_air();
        let point = generate_native_terminal_whir_point_trace(&point_air, verification, None)
            .ok_or(ReducedSwirlTerminalError::Trace("WHIR point"))?;
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
                .ok_or(ReducedSwirlTerminalError::Trace("WHIR final check"))?;
        let weight_term = generate_native_terminal_whir_weight_term_trace(verification, p.k, None)
            .ok_or(ReducedSwirlTerminalError::Trace("WHIR weight terms"))?;
        let adjoint_sumcheck_air = self.adjoint_sumcheck_air();
        let adjoint_sumcheck = generate_native_terminal_rs_adjoint_sumcheck_trace(
            &adjoint_sumcheck_air,
            verification,
            record.transcript,
            None,
        )
        .ok_or(ReducedSwirlTerminalError::Trace("RS-adjoint sumcheck"))?;
        let adjoint_q =
            generate_native_terminal_rs_adjoint_q_trace(verification, &instance.alpha, 0, None)
                .ok_or(ReducedSwirlTerminalError::Trace("RS-adjoint Q"))?;
        let adjoint_y_air = self.adjoint_y_air();
        let adjoint_y =
            generate_native_terminal_rs_adjoint_y_trace(&adjoint_y_air, verification, None)
                .ok_or(ReducedSwirlTerminalError::Trace("RS-adjoint Y"))?;
        let adjoint_selector = generate_native_terminal_rs_adjoint_selector_trace(
            verification,
            &instance.alpha,
            0,
            None,
        )
        .ok_or(ReducedSwirlTerminalError::Trace("RS-adjoint selector"))?;
        let expected_weight =
            generate_native_terminal_whir_expected_weight_trace(verification, p.k, None)
                .ok_or(ReducedSwirlTerminalError::Trace("WHIR expected weight"))?;
        let exp_bits = terminal_exp_bits_trace(
            verification,
            record.transcript,
            p.log_codeword_len,
            p.k,
            p.folding_pow_bits,
            p.query_pow_bits,
        )?;
        let digest = generate_accumulator_digest_traces(instance, p.alpha_len, p.beta_len)?;

        let mut permutation_inputs = leaf_hash.permutation_inputs;
        permutation_inputs.extend(digest.permutation_inputs);
        let mut compression_inputs = merkle_records
            .iter()
            .flat_map(|(_, record)| {
                record.compressions.iter().map(|compression| {
                    core::array::from_fn(|index| {
                        if index < DIGEST_SIZE {
                            compression.left[index]
                        } else {
                            compression.right[index - DIGEST_SIZE]
                        }
                    })
                })
            })
            .collect::<Vec<_>>();
        compression_inputs.extend(digest.compression_inputs);
        let traces = vec![
            bridge,
            start,
            whir_statement,
            linearizer_weight,
            whir_sumcheck,
            whir_round,
            whir_query,
            whir_opened.matrix,
            whir_folding,
            leaf_hash.matrix,
            merkle,
            leaf_adapter,
            final_table,
            mobius,
            point,
            final_check,
            weight_term,
            adjoint_sumcheck,
            adjoint_q,
            adjoint_y,
            adjoint_selector,
            expected_weight,
            exp_bits,
            digest.values,
            digest.hash,
            digest.receipt,
        ];
        if traces.len() != Self::COMPONENT_COUNT {
            return Err(ReducedSwirlTerminalError::Trace("AIR/trace inventory"));
        }
        Ok(ReducedSwirlTerminalTraceData {
            traces,
            poseidon2_permutation_inputs: permutation_inputs,
            poseidon2_compression_inputs: compression_inputs,
            receipt: ReducedSwirlTerminalReceiptRecord {
                protocol_digest: p.protocol_digest,
                relation_digest: p.relation_digest,
                terminal_index_digest: p.terminal_index_digest,
                verifier_component_digest: self.profile.verifier_component_digest,
                final_accumulator_digest: digest.instance_digest,
                final_accumulator_root: instance.rt,
            },
        })
    }

    pub fn generate_cpu_contexts<SC: StarkProtocolConfig<F = F>>(
        &self,
        record: ReducedSwirlTerminalRecord<'_>,
    ) -> Result<ReducedSwirlTerminalCpuPacket<SC>, ReducedSwirlTerminalError> {
        let traces = self.generate_traces(record)?;
        Ok(ReducedSwirlTerminalCpuPacket {
            contexts: traces
                .traces
                .into_iter()
                .map(AirProvingContext::<CpuBackend<SC>>::simple_no_pis)
                .collect(),
            poseidon2_permutation_inputs: traces.poseidon2_permutation_inputs,
            poseidon2_compression_inputs: traces.poseidon2_compression_inputs,
            receipt: traces.receipt,
        })
    }
}

fn terminal_exp_bits_trace(
    verification: &TerminalWhirVerification<F, EF, Digest>,
    transcript: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    log_codeword_len: usize,
    k: usize,
    folding_pow_bits: usize,
    query_pow_bits: usize,
) -> Result<RowMajorMatrix<F>, ReducedSwirlTerminalError> {
    let generator = ExpBitsLenCpuTraceGenerator::default();
    for (round_index, round) in verification.rounds.iter().enumerate() {
        for sumcheck in &round.sumcheck_rounds {
            if folding_pow_bits > 0 {
                let sample_tidx = sumcheck.transcript_span.start.operations + 2 * D_EF + 1;
                let sample = *transcript
                    .values()
                    .get(sample_tidx)
                    .ok_or(ReducedSwirlTerminalError::Transcript("folding PoW sample"))?;
                generator.add_request(F::GENERATOR, sample, folding_pow_bits);
            }
        }
        let query_tidx = round
            .transcript_span
            .end
            .operations
            .checked_sub(D_EF + round.query_indices.len())
            .ok_or(ReducedSwirlTerminalError::Transcript("query cursor"))?;
        if query_pow_bits > 0 {
            let sample = *transcript
                .values()
                .get(
                    query_tidx
                        .checked_sub(1)
                        .ok_or(ReducedSwirlTerminalError::Transcript("query PoW cursor"))?,
                )
                .ok_or(ReducedSwirlTerminalError::Transcript("query PoW sample"))?;
            generator.add_request(F::GENERATOR, sample, query_pow_bits);
        }
        let bits = log_codeword_len
            .checked_sub(k + round_index)
            .ok_or(ReducedSwirlTerminalError::Trace("query index bits"))?;
        if bits == 0 || bits > F::TWO_ADICITY {
            return Err(ReducedSwirlTerminalError::Trace("query domain"));
        }
        let omega = F::two_adic_generator(bits);
        generator.add_requests_with_shift((0..round.query_indices.len()).map(|query| {
            let sample = transcript.values()[query_tidx + query];
            (omega, sample, bits, bits, 1)
        }));
        let raw_omega = F::two_adic_generator(
            usize::try_from(round.log_rs_domain_size)
                .map_err(|_| ReducedSwirlTerminalError::Trace("scalar domain"))?,
        );
        for &query_index in &round.query_indices {
            let coordinate_exponent = query_index;
            let root =
                openvm_stark_backend::warp_accum::terminal_whir::terminal_whir_query_root::<F>(
                    openvm_stark_backend::warp_accum::terminal_whir::TerminalWhirLayout::ScalarRs,
                    round_index == 0,
                    query_index as usize,
                    usize::try_from(round.log_rs_domain_size)
                        .map_err(|_| ReducedSwirlTerminalError::Trace("scalar query domain"))?,
                    k,
                )
                .map_err(|_| ReducedSwirlTerminalError::Trace("scalar query root"))?;
            generator.add_request(
                raw_omega,
                F::from_u32(coordinate_exponent),
                log_codeword_len,
            );
            generator.add_request(root, F::from_usize(1usize << k), k + 1);
        }
    }
    generator
        .generate_trace_row_major(None)
        .ok_or(ReducedSwirlTerminalError::Trace("exp-bits trace"))
}

fn terminal_point_lookup_counts(
    k: usize,
    queries: &[usize],
    final_poly_len: usize,
    log_message_len: usize,
) -> Option<Vec<usize>> {
    let round_count = queries.len();
    let point_len = k
        .checked_mul(round_count)?
        .checked_add(final_poly_len.ilog2() as usize)?;
    if point_len != log_message_len {
        return None;
    }
    // Eq linearizer and coefficient-subgroup RS adjoint selector.
    let mut counts = vec![2usize; point_len];
    for (round, &query_count) in queries.iter().enumerate() {
        let after_folds = (round + 1).checked_mul(k)?;
        let terms = query_count.checked_add(usize::from(round + 1 != round_count))?;
        for count in counts.get_mut(after_folds..)? {
            *count = count.checked_add(terms)?;
        }
    }
    let suffix_start = k.checked_mul(round_count)?;
    for count in counts.get_mut(suffix_start..)? {
        *count = count.checked_add(final_poly_len)?;
    }
    Some(counts)
}

const _: () = assert!(D_EF == 4);

#[cfg(test)]
mod tests {
    use std::{
        panic::{catch_unwind, AssertUnwindSafe},
        sync::Arc,
    };

    use openvm_stark_backend::{
        air_builders::{
            debug::{check_constraints, DebugConstraintBuilder},
            symbolic::get_symbolic_builder,
        },
        keygen::types::TraceWidth,
        native_warp::native_accumulator_instance_digest,
        warp_accum::{TerminalConstrainedRsStatement, WhirInitialRsWarpCode},
        warp_pesat::{TerminalStructuredLinearClaim, TerminalWeightSpec},
        BaseAirWithPublicValues, PartitionedBaseAir, WhirProximityStrategy, WhirRoundConfig,
    };

    use super::*;

    fn digest(offset: usize) -> Digest {
        core::array::from_fn(|index| F::from_usize(offset + index + 1))
    }

    fn whir_config() -> WhirConfig {
        WhirConfig {
            k: 4,
            rounds: vec![WhirRoundConfig { num_queries: 1 }],
            mu_pow_bits: 0,
            query_phase_pow_bits: 0,
            folding_pow_bits: 0,
            proximity: WhirProximityStrategy::UniqueDecoding,
        }
    }

    fn setup_fixture() -> (
        Arc<ReducedSwirlTerminalSetupIdentity>,
        TerminalDescriptor<Digest>,
        WhirInitialRsWarpCode<<NativeSC as StarkProtocolConfig>::Hasher>,
    ) {
        let config = NativeSC::default_from_params(SystemParams::new_for_testing(8));
        let code = WhirInitialRsWarpCode::try_new_coefficient_subgroup(
            config.hasher().clone(),
            8,
            1,
            0,
            16,
        )
        .expect("coefficient-subgroup code");
        let whir = whir_config();
        let descriptor = TerminalDescriptor::from_whir_initial_rs(digest(10), &code, &whir, D_EF);
        let setup = ReducedSwirlTerminalSetupIdentity::new_fixed(
            &descriptor,
            &code,
            &whir,
            digest(100),
            digest(200),
            digest(300),
        )
        .expect("terminal setup identity");
        (Arc::new(setup), descriptor, code)
    }

    fn instance(alpha_len: usize, beta_len: usize) -> AccumulatorInstance<EF, Digest> {
        AccumulatorInstance {
            rt: digest(400),
            alpha: (0..alpha_len)
                .map(|index| EF::from_usize(500 + index))
                .collect(),
            mu: EF::from_usize(600),
            beta: (0..beta_len)
                .map(|index| EF::from_usize(700 + index))
                .collect(),
            eta: EF::from_usize(800),
        }
    }

    fn check_air<A>(air: &A, trace: &RowMajorMatrix<F>)
    where
        A: for<'a> Air<DebugConstraintBuilder<'a, NativeSC>>
            + BaseAir<F>
            + BaseAirWithPublicValues<F>
            + PartitionedBaseAir<F>,
    {
        check_constraints::<_, NativeSC>(
            air,
            core::any::type_name::<A>(),
            &None,
            &[trace.as_view()],
            &[],
        );
    }

    #[test]
    fn setup_identity_is_two_stage_and_coefficient_only() {
        let (setup, descriptor, _) = setup_fixture();
        let material = setup.setup_identity_material();
        let setup_digest = setup.setup_identity_digest();
        let first = setup
            .clone()
            .bind_wrapper_component(digest(900))
            .expect("first wrapper binding");
        let second = setup
            .clone()
            .bind_wrapper_component(digest(1_000))
            .expect("second wrapper binding");
        assert_eq!(first.setup_identity_digest(), setup_digest);
        assert_eq!(second.setup_identity_digest(), setup_digest);
        assert_eq!(first.setup.setup_identity_material(), material);
        assert_ne!(
            first.verifier_component_digest,
            second.verifier_component_digest
        );
        assert!(setup
            .clone()
            .bind_wrapper_component([F::ZERO; DIGEST_SIZE])
            .is_err());

        let config = NativeSC::default_from_params(SystemParams::new_for_testing(8));
        let ordinary = WhirInitialRsWarpCode::new(config.hasher().clone(), 8, 1, 0, 16);
        assert!(ReducedSwirlTerminalSetupIdentity::new_fixed(
            &descriptor,
            &ordinary,
            &whir_config(),
            digest(100),
            digest(200),
            digest(300),
        )
        .is_err());
        let mut wrong_descriptor = descriptor;
        wrong_descriptor.initial_folding_factor = 1;
        let (_, _, code) = setup_fixture();
        assert!(ReducedSwirlTerminalSetupIdentity::new_fixed(
            &wrong_descriptor,
            &code,
            &whir_config(),
            digest(100),
            digest(200),
            digest(300),
        )
        .is_err());
    }

    #[test]
    fn production_factory_derives_digests_and_component_deterministically() {
        let params = SystemParams::new_for_testing(9);
        let first =
            ReducedSwirlTerminalProductionSetup::from_native_fixed_params(params.clone(), 2, 8, 16)
                .expect("production terminal setup");
        let second =
            ReducedSwirlTerminalProductionSetup::from_native_fixed_params(params.clone(), 2, 8, 16)
                .expect("deterministic production terminal setup");
        assert_eq!(first.protocol_digest(), second.protocol_digest());
        assert_eq!(first.relation_digest(), second.relation_digest());
        assert_eq!(
            first.terminal_index_digest(),
            second.terminal_index_digest()
        );
        for digest in [
            first.protocol_digest(),
            first.relation_digest(),
            first.terminal_index_digest(),
        ] {
            assert!(digest.iter().any(|value| *value != F::ZERO));
        }
        let changed =
            ReducedSwirlTerminalProductionSetup::from_native_fixed_params(params, 2, 9, 16)
                .expect("changed production setup");
        assert_ne!(first.protocol_digest(), changed.protocol_digest());
        assert_ne!(
            first.terminal_index_digest(),
            changed.terminal_index_digest()
        );

        let mut manager = BusIndexManager::new();
        let shared = BusInventory::new(&mut manager);
        let footer_bus = ReducedSwirlVaccFooterBus::new(manager.new_bus_idx());
        let receipt_bus_idx = manager.new_bus_idx();
        let expected_component_digest = first
            .component_protocol_digest(
                &shared,
                shared.transcript_bus,
                footer_bus,
                receipt_bus_idx,
                manager.next_bus_idx(),
            )
            .unwrap();
        let component = first
            .instantiate(
                digest(1_600),
                &shared,
                shared.transcript_bus,
                footer_bus,
                receipt_bus_idx,
                manager.next_bus_idx(),
            )
            .expect("production terminal component");
        assert_eq!(
            component.component_count(),
            component.airs::<NativeSC>().len()
        );
        assert_eq!(component.protocol_digest(), expected_component_digest);
        assert_ne!(
            component.protocol_digest(),
            first.setup_identity().setup_identity_digest()
        );
    }

    #[test]
    fn footer_bridge_is_contiguous_and_fail_closed() {
        let source_count = 3;
        let call_count = 5;
        let manifest_digest = digest(1_100);
        let mut transcript = TranscriptLog::default();
        transcript.extend_observe(&[F::from_usize(11), F::from_usize(12)]);
        let start_tidx = transcript.len();
        let footer_elements =
            reduced_swirl_vacc_footer_elements(source_count, manifest_digest).unwrap();
        for element in &footer_elements {
            transcript.extend_observe(element.as_basis_coefficients_slice());
        }
        let end_tidx = transcript.len();
        transcript.push_observe(F::from_usize(21));
        transcript.push_sample(F::from_usize(22));
        transcript.push_observe(F::from_usize(23));
        let footer = ReducedSwirlTerminalFooterRecord {
            source_count,
            call_count,
            proof_idx: call_count - 1,
            local_proof_idx: call_count - 1,
            start_tidx,
            end_tidx,
            manifest_digest,
        };
        let trace = generate_terminal_transcript_bridge_trace(&footer, &transcript)
            .expect("exact footer bridge");
        let air = ReducedSwirlTerminalTranscriptBridgeAir {
            main_transcript_bus: TranscriptBus::new(1),
            transcript_end_index_bus: TranscriptEndIndexBus::new(4),
            semantic_transcript_bus: TranscriptBus::new(2),
            start_bus: ReducedSwirlTerminalStartBus::new(3),
        };
        check_air(&air, &trace);
        let first_row = trace.row_slice(0).unwrap();
        let first: &ReducedSwirlTerminalTranscriptBridgeCols<F> = (*first_row).borrow();
        assert_eq!(first.tidx, F::from_usize(end_tidx));
        assert_eq!(first.proof_idx, F::from_usize(call_count - 1));

        let mut wrong_footer = footer.clone();
        wrong_footer.manifest_digest[0] += F::ONE;
        assert!(generate_terminal_transcript_bridge_trace(&wrong_footer, &transcript).is_err());
        let mut wrong_samples = transcript.clone();
        wrong_samples.samples_mut()[start_tidx] = true;
        assert!(generate_terminal_transcript_bridge_trace(&footer, &wrong_samples).is_err());

        let mut mutated = trace.clone();
        let width = mutated.width();
        let second: &mut ReducedSwirlTerminalTranscriptBridgeCols<F> =
            mutated.values[width..2 * width].borrow_mut();
        second.tidx += F::ONE;
        assert!(catch_unwind(AssertUnwindSafe(|| check_air(&air, &mutated))).is_err());
    }

    #[test]
    fn accumulator_digest_and_eq_target_match_backend() {
        let accumulator = instance(9, 9);
        let traces = generate_accumulator_digest_traces(&accumulator, 9, 9)
            .expect("accumulator digest traces");
        let config = NativeSC::default_from_params(SystemParams::new_for_testing(8));
        assert_eq!(
            traces.instance_digest,
            native_accumulator_instance_digest::<NativeSC>(&config, &accumulator)
        );

        let source_target = *accumulator.beta.last().unwrap();
        let statement = TerminalConstrainedRsStatement {
            linearizer_claims: vec![TerminalStructuredLinearClaim::new(
                TerminalWeightSpec::Eq {
                    point: accumulator.beta[..accumulator.beta.len() - 1].to_vec(),
                },
                source_target + accumulator.eta,
            )],
        };
        validate_native_terminal_eq_statement(&accumulator.beta, accumulator.eta, &statement)
            .expect("corrected Eq target");
        let mut wrong_target = statement;
        wrong_target.linearizer_claims[0].target -= accumulator.eta;
        assert!(validate_native_terminal_eq_statement(
            &accumulator.beta,
            accumulator.eta,
            &wrong_target,
        )
        .is_err());

        let value_air = ReducedSwirlTerminalAccumulatorValueAir {
            alpha_len: 9,
            beta_len: 9,
            value_bus: NativeTerminalAccumulatorValueBus::new(10),
            digest_element_bus: NativeAccumulatorDigestElementBus::new(11),
        };
        check_air(&value_air, &traces.values);
        let hash_air = ReducedSwirlTerminalAccumulatorHashAir {
            alpha_len: 9,
            beta_len: 9,
            digest_element_bus: NativeAccumulatorDigestElementBus::new(11),
            algebraic_digest_bus: NativeAccumulatorAlgebraicDigestBus::new(12),
            permute_bus: Poseidon2PermuteBus::new(13),
            compress_bus: Poseidon2CompressBus::new(14),
        };
        check_air(&hash_air, &traces.hash);

        let footer = ReducedSwirlTerminalFooterRecord {
            source_count: 2,
            call_count: 3,
            proof_idx: 2,
            local_proof_idx: 2,
            start_tidx: 40,
            end_tidx: 73,
            manifest_digest: digest(1_300),
        };
        let start_trace = generate_terminal_start_trace(&footer, &accumulator).unwrap();
        let start_air = ReducedSwirlTerminalStartAir {
            footer_bus: ReducedSwirlVaccFooterBus::new(15),
            start_bus: ReducedSwirlTerminalStartBus::new(16),
            root_bus: NativeTerminalAccumulatorRootBus::new(17),
            value_bus: NativeTerminalAccumulatorValueBus::new(10),
            linearizer_end_bus: NativeTerminalLinearizerEndBus::new(18),
            beta_len: 9,
        };
        check_air(&start_air, &start_trace);

        let (setup, _, _) = setup_fixture();
        let receipt_air = ReducedSwirlTerminalReceiptAir {
            profile: setup.bind_wrapper_component(digest(1_400)).unwrap(),
            wrapper_receipt_bus: ReducedSwirlLocalTerminalReceiptBus::new(19),
            terminal_root_bus: NativeTerminalAccumulatorRootBus::new(17),
            algebraic_digest_bus: NativeAccumulatorAlgebraicDigestBus::new(12),
            compress_bus: Poseidon2CompressBus::new(14),
        };
        check_air(&receipt_air, &traces.receipt);
    }

    #[test]
    fn component_uses_the_wrapper_receipt_bus_without_a_local_namespace() {
        let (setup, _, _) = setup_fixture();
        let profile = setup
            .bind_wrapper_component(digest(1_200))
            .expect("wrapper-bound profile");
        let mut manager = BusIndexManager::new();
        let shared = BusInventory::new(&mut manager);
        let footer_bus = ReducedSwirlVaccFooterBus::new(manager.new_bus_idx());
        let wrapper_receipt_bus_idx = manager.new_bus_idx();
        let first_internal_bus_idx = manager.next_bus_idx();
        let component = ReducedSwirlTerminalComponent::new(
            profile.clone(),
            &shared,
            shared.transcript_bus,
            footer_bus,
            wrapper_receipt_bus_idx,
            first_internal_bus_idx,
            SystemParams::new_for_testing(8),
        )
        .expect("terminal component");
        assert_eq!(
            component.wrapper_terminal_receipt_bus_index(),
            wrapper_receipt_bus_idx
        );
        assert_eq!(
            component.component_count(),
            component.airs::<NativeSC>().len()
        );
        assert_ne!(component.protocol_digest(), profile.setup_identity_digest());
        assert!(!component.component_digest_material().is_empty());

        let receipt = ReducedSwirlLocalTerminalReceiptMessage {
            protocol_digest: digest(1),
            relation_digest: digest(10),
            terminal_index_digest: digest(20),
            verifier_component_digest: digest(30),
            final_accumulator_digest: digest(40),
            final_accumulator_root: digest(50),
        };
        let expected = [
            receipt.protocol_digest,
            receipt.relation_digest,
            receipt.terminal_index_digest,
            receipt.verifier_component_digest,
            receipt.final_accumulator_digest,
            receipt.final_accumulator_root,
        ]
        .concat();
        assert_eq!(receipt.to_vec(), expected);

        assert!(ReducedSwirlTerminalComponent::new(
            profile,
            &shared,
            shared.transcript_bus,
            footer_bus,
            first_internal_bus_idx,
            first_internal_bus_idx,
            SystemParams::new_for_testing(8),
        )
        .is_err());
    }

    #[test]
    fn aggregate_air_inventory_stays_within_terminal_degree() {
        let (setup, _, _) = setup_fixture();
        let profile = setup.bind_wrapper_component(digest(1_500)).unwrap();
        let mut manager = BusIndexManager::new();
        let shared = BusInventory::new(&mut manager);
        let footer_bus = ReducedSwirlVaccFooterBus::new(manager.new_bus_idx());
        let receipt_bus_idx = manager.new_bus_idx();
        let component = ReducedSwirlTerminalComponent::new(
            profile,
            &shared,
            shared.transcript_bus,
            footer_bus,
            receipt_bus_idx,
            manager.next_bus_idx(),
            SystemParams::new_for_testing(8),
        )
        .unwrap();
        for air in component.airs::<NativeSC>() {
            let symbolic = get_symbolic_builder(
                air.as_ref(),
                &TraceWidth {
                    preprocessed: BaseAir::<F>::preprocessed_trace(air.as_ref())
                        .map(|trace| trace.width()),
                    cached_mains: air.cached_main_widths(),
                    common_main: air.common_main_width(),
                },
            )
            .constraints();
            assert!(
                symbolic.max_constraint_degree() <= REDUCED_SWIRL_TERMINAL_MAX_AIR_DEGREE,
                "{} has degree {}",
                air.name(),
                symbolic.max_constraint_degree(),
            );
        }
    }
}

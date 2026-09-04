//! Production coefficient-two-coset terminal transcript prefix for finite
//! WARP v3.
//!
//! The legacy recursive terminal prefix predates
//! `WhirInitialRsLayout::CoefficientTwoCosetGrs`.  It observes
//! `TerminalDescriptor::metadata_words()` directly and then samples the WHIR
//! batching challenge.  That is not the production transcript.  The native
//! verifier observes, in order:
//!
//! 1. the generic terminal descriptor, including its canonical byte-oriented coefficient-two-coset
//!    binding;
//! 2. the independent terminal-WHIR coefficient-two-coset layout block; and
//! 3. the extension-field batching challenge.
//!
//! This module owns the finite-v3 replacement for precisely that prefix.  Its
//! setup profile is derived by running the native descriptor/layout observers,
//! making the native transcript the executable wire-format oracle.  The AIR
//! then emits every resulting field observation as a setup-fixed constant.
//! Neither a proof nor a host verdict may select or omit a layout word.

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{StructReflection, StructReflectionHelper};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_recursion_circuit::{
    bus::{ResumeTranscriptStateBus, ResumeTranscriptStateMessage, TranscriptBus},
    native_warp::terminal::{
        FixedMultiAirCompleteBindingBus, FixedMultiAirCompleteBindingMessage,
        FixedMultiAirCompleteInstanceValueBus, FixedMultiAirCompleteInstanceValueMessage,
        FixedMultiAirTerminalBindingBus, FixedMultiAirTerminalBindingMessage,
        FixedMultiAirTerminalInstanceValueBus, FixedMultiAirTerminalInstanceValueMessage,
        FixedMultiAirWhirPrefixCols, FixedMultiAirWhirStartBus, FixedMultiAirWhirStartMessage,
        NativeTerminalAccumulatorRootBus, NativeTerminalAccumulatorRootMessage,
    },
    transcript::transcript::{TranscriptCols, TranscriptResumeCols},
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder, LookupBus},
    p3_air::{Air, AirBuilder, BaseAir},
    p3_field::{BasedVectorSpace, PrimeCharacteristicRing},
    transcript::{duplex_sponge::DuplexCheckpoint, TranscriptLog},
    warp_accum::{
        terminal_whir::{
            observe_terminal_whir_layout_binding, validate_terminal_whir_layout, TerminalWhirLayout,
        },
        TerminalDescriptor, WhirInitialRsWarpCode,
    },
    warp_pesat::AccumulatorInstance,
    BaseAirWithPublicValues, FiatShamirTranscript, PartitionedBaseAir, StarkProtocolConfig,
    TranscriptHistory, WhirConfig,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, poseidon2_perm, BabyBearPoseidon2Config, Digest, DuplexSponge,
    DuplexSpongeRecorder, DIGEST_SIZE, D_EF, EF, F,
};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::{
    FiniteWarpV3RsStatementEndBus, FiniteWarpV3RsStatementEndMessage, FINITE_WARP_V3_TWO_COSET_K,
};

/// Number of non-legacy descriptor words appended by
/// `TerminalDescriptor::metadata_words()` for a coefficient-two-coset code.
///
/// These words bind `(layout_version, modulus, subgroup_generator,
/// coset_shift, coordinate_ordering)` in setup/index digests.  On the Fiat--
/// Shamir wire they are encoded by a tagged byte block instead of being
/// reduced directly into BabyBear.
const TWO_COSET_DESCRIPTOR_WORDS: usize = 5;

/// Setup-fixed transcript layout for the production terminal initial code.
///
/// The dynamic accumulator root is deliberately absent.  It is supplied by
/// the terminal binding bus and observed at `descriptor_domain + 1` by the
/// AIR.  Every other value is fixed by the code, WHIR parameters, and terminal
/// descriptor admitted during key generation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FiniteWarpV3TwoCosetTerminalProfile {
    descriptor_domain: F,
    descriptor_base_metadata: Vec<F>,
    descriptor_code_layout_block: Vec<F>,
    whir_code_layout_block: Vec<F>,
    descriptor_metadata_words: Vec<u64>,
    alpha_len: usize,
    log_message_len: usize,
    whir_k: usize,
    final_poly_len: usize,
    query_phase_pow_bits: usize,
    folding_pow_bits: usize,
    num_queries_per_round: Vec<usize>,
    whir_round_count: usize,
    whir_sumcheck_round_count: usize,
    whir_remaining_dimension: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FiniteWarpV3TwoCosetTranscriptError {
    UnsupportedLayout,
    Descriptor,
    Geometry,
    NativeTranscriptShape,
    Root,
    Transcript,
    TranscriptResume,
    TraceHeight,
}

impl core::fmt::Display for FiniteWarpV3TwoCosetTranscriptError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            formatter,
            "finite WARP v3 terminal two-coset transcript: {self:?}"
        )
    }
}

impl std::error::Error for FiniteWarpV3TwoCosetTranscriptError {}

impl FiniteWarpV3TwoCosetTerminalProfile {
    /// Admit exactly the native production coefficient-two-coset layout and
    /// capture both canonical layout blocks from the native observers.
    ///
    /// `descriptor.root` is checked against `code` here but is not retained in
    /// the profile, so changing the runtime accumulator root does not change
    /// the setup key.  The root remains constrained by the terminal binding
    /// and same-root buses.
    pub fn new<Obs>(
        descriptor: &TerminalDescriptor<Digest>,
        code: &WhirInitialRsWarpCode<<BabyBearPoseidon2Config as StarkProtocolConfig>::Hasher, Obs>,
        whir: &WhirConfig,
    ) -> Result<Self, FiniteWarpV3TwoCosetTranscriptError> {
        descriptor
            .validate_whir_initial_rs::<
                <BabyBearPoseidon2Config as StarkProtocolConfig>::Hasher,
                Obs,
            >(&descriptor.root, code, whir, D_EF)
            .map_err(|_| FiniteWarpV3TwoCosetTranscriptError::Descriptor)?;

        // Derive the complete tail geometry before asking the native layout
        // validator to classify it.  The checked arithmetic makes malformed
        // setup fail closed without relying on `WhirConfig`'s infallible
        // convenience methods, which assume an already-admitted shape.
        let alpha_len = usize::try_from(descriptor.log_codeword_len)
            .map_err(|_| FiniteWarpV3TwoCosetTranscriptError::Descriptor)?;
        let log_message_len = usize::try_from(descriptor.log_message_len)
            .map_err(|_| FiniteWarpV3TwoCosetTranscriptError::Descriptor)?;
        let whir_k = usize::try_from(descriptor.whir_k)
            .map_err(|_| FiniteWarpV3TwoCosetTranscriptError::Descriptor)?;
        let expected_alpha_len = log_message_len
            .checked_add(1)
            .ok_or(FiniteWarpV3TwoCosetTranscriptError::Geometry)?;
        let whir_round_count = whir.num_whir_rounds();
        let whir_sumcheck_round_count = whir_round_count
            .checked_mul(whir_k)
            .ok_or(FiniteWarpV3TwoCosetTranscriptError::Geometry)?;
        let whir_remaining_dimension = log_message_len
            .checked_sub(whir_sumcheck_round_count)
            .ok_or(FiniteWarpV3TwoCosetTranscriptError::Geometry)?;
        let final_poly_len = 1usize
            .checked_shl(
                whir_remaining_dimension
                    .try_into()
                    .map_err(|_| FiniteWarpV3TwoCosetTranscriptError::Geometry)?,
            )
            .ok_or(FiniteWarpV3TwoCosetTranscriptError::Geometry)?;
        let expected_rows_per_query = 1usize
            .checked_shl(
                whir_k
                    .try_into()
                    .map_err(|_| FiniteWarpV3TwoCosetTranscriptError::Geometry)?,
            )
            .ok_or(FiniteWarpV3TwoCosetTranscriptError::Geometry)?;
        if whir_k != FINITE_WARP_V3_TWO_COSET_K
            || alpha_len != expected_alpha_len
            || code.log_message_len() != log_message_len
            || code.log_codeword_len() != alpha_len
            || code.log_inv_rate() != 1
            || code.initial_folding_factor() != 0
            || code.initial_encoded_oracle_width() != 1
            || code.rows_per_query() != expected_rows_per_query
            || whir_round_count == 0
            || whir.log_final_poly_len(log_message_len) != whir_remaining_dimension
        {
            return Err(FiniteWarpV3TwoCosetTranscriptError::Geometry);
        }
        let num_queries_per_round = whir.rounds.iter().map(|round| round.num_queries).collect();

        if validate_terminal_whir_layout::<BabyBearPoseidon2Config, Obs>(whir, code)
            .map_err(|_| FiniteWarpV3TwoCosetTranscriptError::UnsupportedLayout)?
            != TerminalWhirLayout::ScalarCoefficientTwoCoset
        {
            return Err(FiniteWarpV3TwoCosetTranscriptError::UnsupportedLayout);
        }

        let metadata_words = descriptor.metadata_words().collect::<Vec<_>>();
        let base_metadata_len = metadata_words
            .len()
            .checked_sub(TWO_COSET_DESCRIPTOR_WORDS)
            .ok_or(FiniteWarpV3TwoCosetTranscriptError::NativeTranscriptShape)?;

        // First canonical block: let the native descriptor implementation
        // produce the exact field observations, including every tagged byte.
        let mut descriptor_transcript = default_duplex_sponge_recorder();
        descriptor.observe_fiat_shamir::<BabyBearPoseidon2Config, _>(&mut descriptor_transcript);
        let descriptor_log = TranscriptHistory::into_log(descriptor_transcript);
        if descriptor_log.samples().iter().any(|&sample| sample) {
            return Err(FiniteWarpV3TwoCosetTranscriptError::NativeTranscriptShape);
        }
        let root_start = 1usize;
        let root_end = root_start
            .checked_add(DIGEST_SIZE)
            .ok_or(FiniteWarpV3TwoCosetTranscriptError::NativeTranscriptShape)?;
        let metadata_end = root_end
            .checked_add(base_metadata_len)
            .ok_or(FiniteWarpV3TwoCosetTranscriptError::NativeTranscriptShape)?;
        let descriptor_domain = *descriptor_log
            .values()
            .first()
            .ok_or(FiniteWarpV3TwoCosetTranscriptError::NativeTranscriptShape)?;
        if descriptor_log.values().get(root_start..root_end) != Some(descriptor.root.as_slice()) {
            return Err(FiniteWarpV3TwoCosetTranscriptError::NativeTranscriptShape);
        }
        let descriptor_base_metadata = descriptor_log
            .values()
            .get(root_end..metadata_end)
            .ok_or(FiniteWarpV3TwoCosetTranscriptError::NativeTranscriptShape)?
            .to_vec();
        if descriptor_base_metadata
            != metadata_words[..base_metadata_len]
                .iter()
                .copied()
                .map(F::from_u64)
                .collect::<Vec<_>>()
        {
            return Err(FiniteWarpV3TwoCosetTranscriptError::NativeTranscriptShape);
        }
        let descriptor_code_layout_block = descriptor_log
            .values()
            .get(metadata_end..)
            .ok_or(FiniteWarpV3TwoCosetTranscriptError::NativeTranscriptShape)?
            .to_vec();
        if descriptor_code_layout_block.is_empty() {
            return Err(FiniteWarpV3TwoCosetTranscriptError::UnsupportedLayout);
        }

        // Second canonical block: this is intentionally independent from the
        // descriptor block and must precede terminal xi as in native WHIR.
        let mut whir_layout_transcript = default_duplex_sponge_recorder();
        observe_terminal_whir_layout_binding::<BabyBearPoseidon2Config, Obs, _>(
            code,
            &mut whir_layout_transcript,
        )
        .map_err(|_| FiniteWarpV3TwoCosetTranscriptError::UnsupportedLayout)?;
        let whir_layout_log = TranscriptHistory::into_log(whir_layout_transcript);
        if whir_layout_log.samples().iter().any(|&sample| sample)
            || whir_layout_log.values().is_empty()
        {
            return Err(FiniteWarpV3TwoCosetTranscriptError::NativeTranscriptShape);
        }

        Ok(Self {
            descriptor_domain,
            descriptor_base_metadata,
            descriptor_code_layout_block,
            whir_code_layout_block: whir_layout_log.values().to_vec(),
            descriptor_metadata_words: metadata_words,
            alpha_len,
            log_message_len,
            whir_k,
            final_poly_len,
            query_phase_pow_bits: whir.query_phase_pow_bits,
            folding_pow_bits: whir.folding_pow_bits,
            num_queries_per_round,
            whir_round_count,
            whir_sumcheck_round_count,
            whir_remaining_dimension,
        })
    }

    #[must_use]
    pub fn descriptor_metadata_words(&self) -> &[u64] {
        &self.descriptor_metadata_words
    }

    #[must_use]
    pub const fn alpha_len(&self) -> usize {
        self.alpha_len
    }

    #[must_use]
    pub const fn log_message_len(&self) -> usize {
        self.log_message_len
    }

    #[must_use]
    pub const fn whir_k(&self) -> usize {
        self.whir_k
    }

    /// Number of coefficients in the final WHIR polynomial.  WHIR folds the
    /// `log_message_len` polynomial variables; the extra rate-half codeword
    /// coordinate in `alpha_len` is not a polynomial variable.
    #[must_use]
    pub const fn final_poly_len(&self) -> usize {
        self.final_poly_len
    }

    /// Number of PoW bits checked before sampling each WHIR round's queries.
    #[must_use]
    pub const fn query_phase_pow_bits(&self) -> usize {
        self.query_phase_pow_bits
    }

    /// Number of PoW bits checked before each WHIR folding challenge.
    #[must_use]
    pub const fn folding_pow_bits(&self) -> usize {
        self.folding_pow_bits
    }

    /// Exact in-domain query count for each WHIR round, in transcript order.
    #[must_use]
    pub fn num_queries_per_round(&self) -> &[usize] {
        &self.num_queries_per_round
    }

    #[must_use]
    pub const fn whir_round_count(&self) -> usize {
        self.whir_round_count
    }

    /// Total number of binary sumcheck variables fixed by all WHIR rounds.
    #[must_use]
    pub const fn whir_sumcheck_round_count(&self) -> usize {
        self.whir_sumcheck_round_count
    }

    /// Number of message-polynomial variables left for the final table.
    #[must_use]
    pub const fn whir_remaining_dimension(&self) -> usize {
        self.whir_remaining_dimension
    }

    /// First canonical coefficient-two-coset byte block, after the generic
    /// descriptor metadata and before the independent WHIR layout block.
    #[must_use]
    pub fn descriptor_code_layout_block(&self) -> &[F] {
        &self.descriptor_code_layout_block
    }

    /// Second canonical coefficient-two-coset block emitted by terminal WHIR.
    #[must_use]
    pub fn whir_code_layout_block(&self) -> &[F] {
        &self.whir_code_layout_block
    }

    /// Exact base-field descriptor observations emitted by
    /// `TerminalDescriptor::observe_fiat_shamir`, excluding the later,
    /// independent terminal-WHIR layout block.
    pub(super) fn descriptor_fiat_shamir_values(&self, root: Digest) -> Vec<F> {
        let mut values = Vec::with_capacity(
            1 + DIGEST_SIZE
                + self.descriptor_base_metadata.len()
                + self.descriptor_code_layout_block.len(),
        );
        values.push(self.descriptor_domain);
        values.extend(root);
        values.extend_from_slice(&self.descriptor_base_metadata);
        values.extend_from_slice(&self.descriptor_code_layout_block);
        values
    }

    /// Exact extension-field observations emitted by
    /// `TerminalDescriptor::observe_algebraic` for the field-element digest
    /// observer used by finite verifier-WARP.
    pub(super) fn descriptor_algebraic_values(&self, root: Digest) -> Vec<EF> {
        self.descriptor_fiat_shamir_values(root)
            .into_iter()
            .map(EF::from)
            .collect()
    }

    /// Number of observed base-field elements before terminal `xi`.
    #[must_use]
    pub fn observation_len(&self) -> usize {
        1 + DIGEST_SIZE
            + self.descriptor_base_metadata.len()
            + self.descriptor_code_layout_block.len()
            + self.whir_code_layout_block.len()
    }

    /// Relative index of the second layout block.  Exposed for transcript
    /// mutation tests and codec audits, not as prover-controlled metadata.
    #[must_use]
    pub fn whir_layout_block_offset(&self) -> usize {
        1 + DIGEST_SIZE
            + self.descriptor_base_metadata.len()
            + self.descriptor_code_layout_block.len()
    }

    /// Exact observation sequence before terminal `xi`, with the dynamic root
    /// supplied by the authenticated accumulator instance.
    #[must_use]
    pub fn observations(&self, root: Digest) -> Vec<F> {
        let mut values = Vec::new();
        values.push(self.descriptor_domain);
        values.extend(root);
        values.extend_from_slice(&self.descriptor_base_metadata);
        values.extend_from_slice(&self.descriptor_code_layout_block);
        values.extend_from_slice(&self.whir_code_layout_block);
        values
    }

    /// Validate and read the native terminal batching challenge at an
    /// arbitrary transcript offset.  All indexing is checked before access;
    /// malformed or truncated logs are rejected.
    pub fn validate_transcript_prefix(
        &self,
        transcript: &TranscriptLog<F, [F; 16]>,
        start_tidx: usize,
        root: Digest,
    ) -> Result<(EF, usize), FiniteWarpV3TwoCosetTranscriptError> {
        let expected = self.observations(root);
        let observation_end = start_tidx
            .checked_add(expected.len())
            .ok_or(FiniteWarpV3TwoCosetTranscriptError::Transcript)?;
        if transcript.values().get(start_tidx..observation_end) != Some(expected.as_slice())
            || transcript
                .samples()
                .get(start_tidx..observation_end)
                .is_none_or(|flags| flags.iter().any(|&sample| sample))
        {
            return Err(FiniteWarpV3TwoCosetTranscriptError::Transcript);
        }
        let challenge_end = observation_end
            .checked_add(D_EF)
            .ok_or(FiniteWarpV3TwoCosetTranscriptError::Transcript)?;
        let challenge_values = transcript
            .values()
            .get(observation_end..challenge_end)
            .ok_or(FiniteWarpV3TwoCosetTranscriptError::Transcript)?;
        if transcript
            .samples()
            .get(observation_end..challenge_end)
            .is_none_or(|flags| flags.iter().any(|&sample| !sample))
        {
            return Err(FiniteWarpV3TwoCosetTranscriptError::Transcript);
        }
        let challenge = EF::from_basis_coefficients_slice(challenge_values)
            .ok_or(FiniteWarpV3TwoCosetTranscriptError::Transcript)?;
        Ok((challenge, challenge_end))
    }

    fn matches_descriptor(&self, descriptor: &TerminalDescriptor<Digest>) -> bool {
        descriptor
            .metadata_words()
            .eq(self.descriptor_metadata_words.iter().copied())
    }
}

/// Exact production replacement for `FixedMultiAirWhirPrefixAir`.
///
/// This has the same typed input/output buses and row shape, allowing the
/// complete terminal assembly to replace only the obsolete prefix.  The
/// replacement must be reflected in the final verifier component/VK digest.
pub struct FiniteWarpV3TwoCosetWhirPrefixAir {
    pub transcript_bus: TranscriptBus,
    pub binding_bus: FixedMultiAirTerminalBindingBus,
    pub instance_bus: FixedMultiAirTerminalInstanceValueBus,
    pub start_bus: FixedMultiAirWhirStartBus,
    pub root_bus: NativeTerminalAccumulatorRootBus,
    pub relation_digest: Digest,
    pub profile: FiniteWarpV3TwoCosetTerminalProfile,
    pub beta_len: usize,
}

impl BaseAirWithPublicValues<F> for FiniteWarpV3TwoCosetWhirPrefixAir {}
impl PartitionedBaseAir<F> for FiniteWarpV3TwoCosetWhirPrefixAir {}
impl BaseAir<F> for FiniteWarpV3TwoCosetWhirPrefixAir {
    fn width(&self) -> usize {
        FixedMultiAirWhirPrefixCols::<F>::width()
    }
}

impl<AB> Air<AB> for FiniteWarpV3TwoCosetWhirPrefixAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("finite WARP v3 two-coset WHIR prefix row");
        let local: &FixedMultiAirWhirPrefixCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_transition().assert_zero(local.active);

        self.binding_bus.receive(
            builder,
            FixedMultiAirTerminalBindingMessage {
                relation_digest: self.relation_digest.map(Into::into),
                root: local.root.map(Into::into),
                alpha_len: AB::Expr::from_usize(self.profile.alpha_len),
                beta_len: AB::Expr::from_usize(self.beta_len),
            },
            local.active,
        );
        self.instance_bus.lookup_key(
            builder,
            FixedMultiAirTerminalInstanceValueMessage {
                section: AB::Expr::ONE,
                coordinate: AB::Expr::ZERO,
                value: local.mu.map(Into::into),
            },
            local.active,
        );

        observe_two_coset_prefix(
            builder,
            self.transcript_bus,
            self.start_bus,
            self.root_bus,
            &self.profile,
            local,
        );
    }
}

/// Generate the one-row common trace for
/// [`FiniteWarpV3TwoCosetWhirPrefixAir`].
pub fn generate_finite_warp_v3_two_coset_whir_prefix_trace(
    profile: &FiniteWarpV3TwoCosetTerminalProfile,
    descriptor: &TerminalDescriptor<Digest>,
    instance: &AccumulatorInstance<EF, Digest>,
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
) -> Result<RowMajorMatrix<F>, FiniteWarpV3TwoCosetTranscriptError> {
    if descriptor.root != instance.rt {
        return Err(FiniteWarpV3TwoCosetTranscriptError::Root);
    }
    if !profile.matches_descriptor(descriptor) {
        return Err(FiniteWarpV3TwoCosetTranscriptError::Descriptor);
    }
    let (batching_challenge, _) =
        profile.validate_transcript_prefix(transcript, start_tidx, instance.rt)?;
    let width = FixedMultiAirWhirPrefixCols::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut FixedMultiAirWhirPrefixCols<F> = values.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.tidx = F::from_usize(start_tidx);
    cols.root = instance.rt;
    cols.mu
        .copy_from_slice(instance.mu.as_basis_coefficients_slice());
    cols.batching_challenge
        .copy_from_slice(batching_challenge.as_basis_coefficients_slice());
    Ok(RowMajorMatrix::new(values, width))
}

/// Coefficient-two-coset WHIR prefix for the genuine complete verifier PESAT.
///
/// Unlike [`FiniteWarpV3TwoCosetWhirPrefixAir`], this AIR consumes the typed
/// post-RS-statement cursor plus the binding and accumulator-instance buses
/// emitted by `FixedMultiAirCompleteTerminalCircuit`. It is therefore impossible to wire
/// the complete proof bytes into the legacy local-only terminal relation by
/// swapping traces or matching AIR names. The outputs intentionally use the
/// generic WHIR-start and same-root buses because the remaining WHIR/adjoint
/// tail is relation-independent once the mapped linear claim is authenticated.
pub struct FiniteWarpV3CompleteTwoCosetWhirPrefixAir {
    pub transcript_bus: TranscriptBus,
    pub cursor_bus: FiniteWarpV3RsStatementEndBus,
    pub binding_bus: FixedMultiAirCompleteBindingBus,
    pub instance_bus: FixedMultiAirCompleteInstanceValueBus,
    pub start_bus: FixedMultiAirWhirStartBus,
    pub root_bus: NativeTerminalAccumulatorRootBus,
    pub relation_digest: Digest,
    pub profile: FiniteWarpV3TwoCosetTerminalProfile,
    pub beta_len: usize,
}

impl BaseAirWithPublicValues<F> for FiniteWarpV3CompleteTwoCosetWhirPrefixAir {}
impl PartitionedBaseAir<F> for FiniteWarpV3CompleteTwoCosetWhirPrefixAir {}
impl BaseAir<F> for FiniteWarpV3CompleteTwoCosetWhirPrefixAir {
    fn width(&self) -> usize {
        FixedMultiAirWhirPrefixCols::<F>::width()
    }
}

impl<AB> Air<AB> for FiniteWarpV3CompleteTwoCosetWhirPrefixAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("finite WARP v3 complete two-coset WHIR prefix row");
        let local: &FixedMultiAirWhirPrefixCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_transition().assert_zero(local.active);

        self.cursor_bus.receive(
            builder,
            FiniteWarpV3RsStatementEndMessage {
                tidx: local.tidx.into(),
            },
            local.active,
        );
        self.binding_bus.lookup_key(
            builder,
            FixedMultiAirCompleteBindingMessage {
                relation_digest: self.relation_digest.map(Into::into),
                root: local.root.map(Into::into),
                alpha_len: AB::Expr::from_usize(self.profile.alpha_len),
                beta_len: AB::Expr::from_usize(self.beta_len),
            },
            local.active,
        );
        self.instance_bus.lookup_key(
            builder,
            FixedMultiAirCompleteInstanceValueMessage {
                section: AB::Expr::ONE,
                coordinate: AB::Expr::ZERO,
                value: local.mu.map(Into::into),
            },
            local.active,
        );

        observe_two_coset_prefix(
            builder,
            self.transcript_bus,
            self.start_bus,
            self.root_bus,
            &self.profile,
            local,
        );
    }
}

/// Generate the complete-relation prefix trace. The row encoding and native
/// transcript validation are shared with the legacy-bus adapter; only the
/// typed relation seam differs.
pub fn generate_finite_warp_v3_complete_two_coset_whir_prefix_trace(
    profile: &FiniteWarpV3TwoCosetTerminalProfile,
    descriptor: &TerminalDescriptor<Digest>,
    instance: &AccumulatorInstance<EF, Digest>,
    transcript: &TranscriptLog<F, [F; 16]>,
    start_tidx: usize,
) -> Result<RowMajorMatrix<F>, FiniteWarpV3TwoCosetTranscriptError> {
    generate_finite_warp_v3_two_coset_whir_prefix_trace(
        profile, descriptor, instance, transcript, start_tidx,
    )
}

fn observe_two_coset_prefix<AB>(
    builder: &mut AB,
    transcript_bus: TranscriptBus,
    start_bus: FixedMultiAirWhirStartBus,
    root_bus: NativeTerminalAccumulatorRootBus,
    profile: &FiniteWarpV3TwoCosetTerminalProfile,
    local: &FixedMultiAirWhirPrefixCols<AB::Var>,
) where
    AB: AirBuilder<F = F> + InteractionBuilder,
{
    let mut offset = 0usize;
    transcript_bus.observe(
        builder,
        AB::Expr::ZERO,
        local.tidx,
        profile.descriptor_domain,
        local.active,
    );
    offset += 1;
    transcript_bus.observe_commit(
        builder,
        AB::Expr::ZERO,
        AB::Expr::from(local.tidx) + AB::Expr::from_usize(offset),
        local.root,
        local.active,
    );
    offset += DIGEST_SIZE;
    for &value in profile
        .descriptor_base_metadata
        .iter()
        .chain(&profile.descriptor_code_layout_block)
        .chain(&profile.whir_code_layout_block)
    {
        transcript_bus.observe(
            builder,
            AB::Expr::ZERO,
            AB::Expr::from(local.tidx) + AB::Expr::from_usize(offset),
            value,
            local.active,
        );
        offset += 1;
    }
    debug_assert_eq!(offset, profile.observation_len());
    transcript_bus.sample_ext(
        builder,
        AB::Expr::ZERO,
        AB::Expr::from(local.tidx) + AB::Expr::from_usize(offset),
        local.batching_challenge,
        local.active,
    );
    start_bus.send(
        builder,
        FixedMultiAirWhirStartMessage {
            tidx: AB::Expr::from(local.tidx) + AB::Expr::from_usize(offset + D_EF),
            root: local.root.map(Into::into),
            batching_challenge: local.batching_challenge.map(Into::into),
            mu: local.mu.map(Into::into),
        },
        local.active,
    );
    root_bus.send(
        builder,
        NativeTerminalAccumulatorRootMessage {
            root: local.root.map(Into::into),
        },
        local.active,
    );
}

/// Exact final-WARP checkpoint from which the terminal transcript resumes.
///
/// The operation index is absolute.  The terminal transcript log may retain
/// the complete block history for the other recorded-verifier indices, but
/// only the suffix beginning at `tidx` is materialized by the terminal
/// transcript AIR.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FiniteWarpV3TerminalTranscriptCheckpoint {
    pub tidx: u32,
    pub sample_count: u32,
    pub state: [F; POSEIDON2_WIDTH],
}

/// Complete authenticated handoff emitted by the final exact-finite VACC
/// receipt.  Unlike `CertifiedTranscriptCheckpointBus`, this message binds
/// the transcript endpoint to the exact WARP setup, schedule, and output
/// accumulator.  The internal certified-checkpoint bus already has its own
/// authority consumer and must never be reused for this seam.
#[derive(Clone, Debug)]
pub struct FiniteWarpV3FinalVaccCheckpointMessage<T> {
    pub protocol_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub warp_index_digest: [T; DIGEST_SIZE],
    pub setup_digest: [T; DIGEST_SIZE],
    pub schedule_digest: [T; DIGEST_SIZE],
    pub call_count: T,
    pub end_tidx: T,
    pub end_sample_count: T,
    pub end_state: [T; POSEIDON2_WIDTH],
    pub output_accumulator_root: [T; DIGEST_SIZE],
    pub output_accumulator_digest: [T; DIGEST_SIZE],
}

impl<T: Clone> FiniteWarpV3FinalVaccCheckpointMessage<T> {
    #[must_use]
    pub fn to_vec(&self) -> Vec<T> {
        let mut values =
            Vec::with_capacity(5 * DIGEST_SIZE + 3 + POSEIDON2_WIDTH + 2 * DIGEST_SIZE);
        values.extend_from_slice(&self.protocol_digest);
        values.extend_from_slice(&self.relation_digest);
        values.extend_from_slice(&self.warp_index_digest);
        values.extend_from_slice(&self.setup_digest);
        values.extend_from_slice(&self.schedule_digest);
        values.extend([
            self.call_count.clone(),
            self.end_tidx.clone(),
            self.end_sample_count.clone(),
        ]);
        values.extend_from_slice(&self.end_state);
        values.extend_from_slice(&self.output_accumulator_root);
        values.extend_from_slice(&self.output_accumulator_digest);
        values
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FiniteWarpV3FinalVaccCheckpointBus(LookupBus);

impl FiniteWarpV3FinalVaccCheckpointBus {
    #[must_use]
    pub const fn new(index: BusIndex) -> Self {
        Self(LookupBus::new(index))
    }

    #[must_use]
    pub const fn index(self) -> BusIndex {
        self.0.index
    }

    pub fn lookup_key<AB, T>(
        &self,
        builder: &mut AB,
        message: FiniteWarpV3FinalVaccCheckpointMessage<T>,
        enabled: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
        T: Into<AB::Expr> + Clone,
    {
        self.0.lookup_key(builder, message.to_vec(), enabled);
    }

    pub fn add_key_with_lookups<AB, T>(
        &self,
        builder: &mut AB,
        message: FiniteWarpV3FinalVaccCheckpointMessage<T>,
        lookups: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
        T: Into<AB::Expr> + Clone,
    {
        self.0
            .add_key_with_lookups(builder, message.to_vec(), lookups);
    }
}

/// Private terminal-owned bridge tying the final VACC output to the root and
/// canonical accumulator digest emitted by terminal Decide.
#[derive(Clone, Debug)]
pub struct FiniteWarpV3TerminalAccumulatorLinkMessage<T> {
    pub root: [T; DIGEST_SIZE],
    pub digest: [T; DIGEST_SIZE],
}

impl<T: Clone> FiniteWarpV3TerminalAccumulatorLinkMessage<T> {
    #[must_use]
    pub fn to_vec(&self) -> Vec<T> {
        let mut values = Vec::with_capacity(2 * DIGEST_SIZE);
        values.extend_from_slice(&self.root);
        values.extend_from_slice(&self.digest);
        values
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FiniteWarpV3TerminalAccumulatorLinkBus(LookupBus);

impl FiniteWarpV3TerminalAccumulatorLinkBus {
    #[must_use]
    pub const fn new(index: BusIndex) -> Self {
        Self(LookupBus::new(index))
    }

    #[must_use]
    pub const fn index(self) -> BusIndex {
        self.0.index
    }

    pub fn lookup_key<AB, T>(
        &self,
        builder: &mut AB,
        message: FiniteWarpV3TerminalAccumulatorLinkMessage<T>,
        enabled: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
        T: Into<AB::Expr> + Clone,
    {
        self.0.lookup_key(builder, message.to_vec(), enabled);
    }

    pub fn add_key_with_lookups<AB, T>(
        &self,
        builder: &mut AB,
        message: FiniteWarpV3TerminalAccumulatorLinkMessage<T>,
        lookups: impl Into<AB::Expr>,
    ) where
        AB: InteractionBuilder,
        T: Into<AB::Expr> + Clone,
    {
        self.0
            .add_key_with_lookups(builder, message.to_vec(), lookups);
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FiniteWarpV3TerminalTranscriptSeamCols<T> {
    pub active: T,
    pub tidx: T,
    pub sample_count: T,
    pub state: [T; POSEIDON2_WIDTH],
    pub output_accumulator_root: [T; DIGEST_SIZE],
    pub output_accumulator_digest: [T; DIGEST_SIZE],
}

/// Constrained handoff from the final authenticated WARP checkpoint to the
/// resumed terminal transcript.  This is a lookup seam, not a host verdict:
/// the row cannot balance unless the WARP verifier emitted the same complete
/// checkpoint, and the resumed [`openvm_recursion_circuit::transcript::TranscriptAir`]
/// consumes the same `(tidx, state)` on its first row.
#[derive(Clone, Debug)]
pub struct FiniteWarpV3TerminalTranscriptSeamAir {
    pub final_vacc_checkpoint_bus: FiniteWarpV3FinalVaccCheckpointBus,
    pub terminal_resume_bus: ResumeTranscriptStateBus,
    pub terminal_accumulator_link_bus: FiniteWarpV3TerminalAccumulatorLinkBus,
    pub protocol_digest: Digest,
    pub relation_digest: Digest,
    pub warp_index_digest: Digest,
    pub setup_digest: Digest,
    pub schedule_digest: Digest,
    pub call_count: usize,
}

impl BaseAirWithPublicValues<F> for FiniteWarpV3TerminalTranscriptSeamAir {}
impl PartitionedBaseAir<F> for FiniteWarpV3TerminalTranscriptSeamAir {}
impl BaseAir<F> for FiniteWarpV3TerminalTranscriptSeamAir {
    fn width(&self) -> usize {
        FiniteWarpV3TerminalTranscriptSeamCols::<F>::width()
    }
}

impl<AB> Air<AB> for FiniteWarpV3TerminalTranscriptSeamAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    AB::Expr: From<AB::Var>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("finite WARP v3 terminal transcript seam row");
        let local: &FiniteWarpV3TerminalTranscriptSeamCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.when_first_row().assert_one(local.active);
        builder.when_transition().assert_zero(local.active);

        self.final_vacc_checkpoint_bus.lookup_key(
            builder,
            FiniteWarpV3FinalVaccCheckpointMessage {
                protocol_digest: self.protocol_digest.map(Into::into),
                relation_digest: self.relation_digest.map(Into::into),
                warp_index_digest: self.warp_index_digest.map(Into::into),
                setup_digest: self.setup_digest.map(Into::into),
                schedule_digest: self.schedule_digest.map(Into::into),
                call_count: AB::Expr::from_usize(self.call_count),
                end_tidx: local.tidx.into(),
                end_sample_count: local.sample_count.into(),
                end_state: local.state.map(Into::into),
                output_accumulator_root: local.output_accumulator_root.map(Into::into),
                output_accumulator_digest: local.output_accumulator_digest.map(Into::into),
            },
            local.active,
        );
        self.terminal_resume_bus.send(
            builder,
            AB::Expr::ZERO,
            ResumeTranscriptStateMessage {
                tidx: local.tidx.into(),
                state: local.state.map(Into::into),
            },
            local.active,
        );
        self.terminal_accumulator_link_bus.add_key_with_lookups(
            builder,
            FiniteWarpV3TerminalAccumulatorLinkMessage {
                root: local.output_accumulator_root.map(Into::into),
                digest: local.output_accumulator_digest.map(Into::into),
            },
            local.active,
        );
    }
}

#[must_use]
pub fn finite_warp_v3_terminal_transcript_seam_trace(
    checkpoint: FiniteWarpV3TerminalTranscriptCheckpoint,
    output_accumulator_root: Digest,
    output_accumulator_digest: Digest,
) -> RowMajorMatrix<F> {
    let width = FiniteWarpV3TerminalTranscriptSeamCols::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut FiniteWarpV3TerminalTranscriptSeamCols<F> = values.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.tidx = F::from_u32(checkpoint.tidx);
    cols.sample_count = F::from_u32(checkpoint.sample_count);
    cols.state = checkpoint.state;
    cols.output_accumulator_root = output_accumulator_root;
    cols.output_accumulator_digest = output_accumulator_digest;
    RowMajorMatrix::new(values, width)
}

/// Rebuild the terminal transcript AIR trace from the authenticated WARP
/// checkpoint while retaining absolute transcript indices.
///
/// `transcript` is the native recorded log used by every terminal trace
/// generator.  Values before `checkpoint.tidx` remain available for recorded
/// proof indices, but they are not replayed here.  Every sampled value in the
/// suffix is independently recomputed from the resumed sponge before a row is
/// emitted.
pub fn generate_finite_warp_v3_resumed_terminal_transcript_trace(
    transcript: &TranscriptLog<F, [F; POSEIDON2_WIDTH]>,
    checkpoint: FiniteWarpV3TerminalTranscriptCheckpoint,
    required_height: Option<usize>,
) -> Result<(RowMajorMatrix<F>, Vec<[F; POSEIDON2_WIDTH]>), FiniteWarpV3TwoCosetTranscriptError> {
    const RATE: usize = DIGEST_SIZE;

    let base_tidx = usize::try_from(checkpoint.tidx)
        .map_err(|_| FiniteWarpV3TwoCosetTranscriptError::TranscriptResume)?;
    let values = transcript
        .values()
        .get(base_tidx..)
        .ok_or(FiniteWarpV3TwoCosetTranscriptError::TranscriptResume)?;
    let samples = transcript
        .samples()
        .get(base_tidx..)
        .ok_or(FiniteWarpV3TwoCosetTranscriptError::TranscriptResume)?;
    if values.is_empty() || values.len() != samples.len() || samples.first() != Some(&false) {
        return Err(FiniteWarpV3TwoCosetTranscriptError::TranscriptResume);
    }

    let inner = DuplexSponge::from_checkpoint(
        poseidon2_perm().clone(),
        DuplexCheckpoint {
            state: checkpoint.state,
            absorb_idx: 0,
            sample_idx: 0,
        },
    );
    let mut replay = DuplexSpongeRecorder {
        inner,
        log: TranscriptLog::default(),
    };
    replay.log.push_perm_result(checkpoint.state);
    for (&value, &is_sample) in values.iter().zip(samples) {
        if is_sample {
            let actual =
                <DuplexSpongeRecorder as FiatShamirTranscript<BabyBearPoseidon2Config>>::sample(
                    &mut replay,
                );
            if actual != value {
                return Err(FiniteWarpV3TwoCosetTranscriptError::TranscriptResume);
            }
        } else {
            <DuplexSpongeRecorder as FiatShamirTranscript<BabyBearPoseidon2Config>>::observe(
                &mut replay,
                value,
            );
        }
    }

    let mut valid_rows = 0usize;
    let mut current_is_sample = false;
    let mut count = 0usize;
    for &is_sample in samples {
        if is_sample != current_is_sample {
            if count != 0 {
                valid_rows = valid_rows
                    .checked_add(1)
                    .ok_or(FiniteWarpV3TwoCosetTranscriptError::TraceHeight)?;
            }
            current_is_sample = is_sample;
            count = 1;
        } else {
            if count == RATE {
                valid_rows = valid_rows
                    .checked_add(1)
                    .ok_or(FiniteWarpV3TwoCosetTranscriptError::TraceHeight)?;
                count = 0;
            }
            count += 1;
        }
    }
    if count != 0 {
        valid_rows = valid_rows
            .checked_add(1)
            .ok_or(FiniteWarpV3TwoCosetTranscriptError::TraceHeight)?;
    }
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return Err(FiniteWarpV3TwoCosetTranscriptError::TraceHeight);
    }

    let base_width = TranscriptCols::<F>::width();
    let resume_width = TranscriptResumeCols::<F>::width();
    let width = base_width + resume_width;
    let mut trace = F::zero_vec(
        height
            .checked_mul(width)
            .ok_or(FiniteWarpV3TwoCosetTranscriptError::TraceHeight)?,
    );
    let transitions = replay.log.permutation_transitions();
    let mut transition_index = 0usize;
    let mut operation = 0usize;
    let mut previous_state = checkpoint.state;

    for row_index in 0..valid_rows {
        let row = trace
            .get_mut(row_index * width..(row_index + 1) * width)
            .ok_or(FiniteWarpV3TwoCosetTranscriptError::TraceHeight)?;
        let (base_row, resume_row) = row.split_at_mut(base_width);
        let resume: &mut TranscriptResumeCols<F> = resume_row.borrow_mut();
        resume.state = checkpoint.state;
        let cols: &mut TranscriptCols<F> = base_row.borrow_mut();
        cols.proof_idx = F::ZERO;
        cols.is_proof_start = F::from_bool(row_index == 0);
        cols.tidx = F::from_usize(
            base_tidx
                .checked_add(operation)
                .ok_or(FiniteWarpV3TwoCosetTranscriptError::TraceHeight)?,
        );
        let is_sample = *samples
            .get(operation)
            .ok_or(FiniteWarpV3TwoCosetTranscriptError::TranscriptResume)?;
        cols.is_sample = F::from_bool(is_sample);
        cols.prev_state = previous_state;

        let mut row_count = 0usize;
        while operation < samples.len() && samples[operation] == is_sample && row_count < RATE {
            cols.mask[row_count] = F::ONE;
            if is_sample {
                if cols.prev_state[RATE - 1 - row_count] != values[operation] {
                    return Err(FiniteWarpV3TwoCosetTranscriptError::TranscriptResume);
                }
            } else {
                cols.prev_state[row_count] = values[operation];
            }
            operation += 1;
            row_count += 1;
        }
        if !is_sample {
            cols.prev_state[RATE] += F::from_usize(row_count);
        }
        let permuted =
            operation < samples.len() && (samples[operation] || (!is_sample && row_count == RATE));
        if permuted {
            let transition = transitions
                .get(transition_index)
                .ok_or(FiniteWarpV3TwoCosetTranscriptError::TranscriptResume)?;
            if transition.input != cols.prev_state {
                return Err(FiniteWarpV3TwoCosetTranscriptError::TranscriptResume);
            }
            previous_state = transition.output;
            transition_index += 1;
        } else {
            previous_state = cols.prev_state;
        }
        cols.post_state = previous_state;
    }
    if operation != values.len() || transition_index != transitions.len() {
        return Err(FiniteWarpV3TwoCosetTranscriptError::TranscriptResume);
    }
    Ok((
        RowMajorMatrix::new(trace, width),
        transitions
            .iter()
            .map(|transition| transition.input)
            .collect(),
    ))
}

const _: () = assert!(D_EF == 4);
const _: () = assert!(DIGEST_SIZE == 8);
const _: () = assert!(POSEIDON2_WIDTH == 16);

#[cfg(test)]
#[path = "terminal_two_coset_transcript_tests.rs"]
mod tests;

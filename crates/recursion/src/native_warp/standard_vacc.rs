//! Generic verifier AIR boundaries for one standard arity-two WARP VACC step.
//!
//! These gadgets intentionally know nothing about an application relation.
//! They certify the v5 VACC transcript, bind the ordinary fresh/prior/output
//! codeword commitments, and expose the verifier-derived statement to a
//! caller-owned adapter.  In particular, no PCS opening is evaluated as a
//! PESAT relation here: direct-AIR relation evaluation belongs to terminal
//! Decide.

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::{BusIndex, InteractionBuilder},
    native_warp::{
        DirectAirPesatIndex, DirectAirPesatRelationDescription, FixedMultiAirPesatIndex,
    },
    warp_accum::{
        twin_degree, MerkleBatchOpeningVerification, WarpParams, EXACT_FINITE_WARP_PROTOCOL_TAG,
        EXACT_FINITE_WARP_TRANSCRIPT_VERSION, NATIVE_WARP_VACC_APPENDIX_D_PROTOCOL_TAG,
        NATIVE_WARP_VACC_PROTOCOL_TAG,
    },
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::{TranscriptBus, TranscriptEndIndexBus, TranscriptEndIndexMessage},
    define_typed_lookup_bus,
    native_warp::{
        NativeAccumulatorProjectionCols, NativeAccumulatorProjectionTrace,
        NativeAccumulatorRootBus, NativeAccumulatorRootMessage, NativeAuthenticatedShiftBus,
        NativeAuthenticatedShiftMessage, NativeClaimLayoutBus, NativeClaimLayoutMessage,
        NativeClaimValueBus, NativeClaimValueCols, NativeClaimValueMessage,
        NativeCoefficientSumcheckAir, NativeEqResultBus, NativeFoldedClaimBus,
        NativeInputSlotLayoutBus, NativeInputSlotLayoutCols, NativeInputSlotLayoutMessage,
        NativeLeafValueBus, NativeLeafValueMessage, NativeMerkleRootBus, NativeMerkleRootMessage,
        NativeShiftIndexBus, NativeShiftIndexMessage, NativeSumcheckChallengeBus,
        NativeSumcheckInitialBus, NativeSumcheckRoundBus, NativeTwinFinalAir, NativeTwinFoldAir,
        NativeTwinOmegaBus, NativeTwinScalarBus, NativeTwinSigmaAir, NativeVaccPhaseCursorBus,
        NativeVaccPhaseCursorMessage, NativeVaccTranscriptRoleBus, NativeVaccTranscriptRoleMessage,
        CLAIM_SECTION_ETA, CLAIM_SECTION_MU,
    },
};

/// Protocol-v19 uses exactly one fresh source and, after bootstrap, one prior
/// accumulator in an arity-two WARP step.
pub const STANDARD_DIRECT_VACC_INPUT_ARITY: usize = 2;

/// Largest exact-finite invocation supported by the recursive wrapper key.
/// This is deliberately a protocol limit, not a witness-selected capacity.
pub const STANDARD_EXACT_FINITE_VACC_MAX_INPUT_ARITY: usize = 64;

/// Number of exact-finite calls supported by the finite wrapper key.
pub const STANDARD_EXACT_FINITE_VACC_MAX_CALLS: usize = 3;
pub const STANDARD_EXACT_FINITE_VACC_MAX_LOOKUP_COUNT: usize = 16;

/// Number of complete source statements carried by one fixed History leaf.
/// This is a verifier-key capacity; runtime occupancy is a non-empty prefix.
pub const FIXED_HLEAF_VACC_CAPACITY_V4: usize = 4;

/// Backend WARP's prior-input discriminator is transcript-bound. Every real
/// HLeaf transition consumes the setup-authenticated valid seed (or a prior
/// real output) and therefore uses the same positive step. There is no V4
/// bootstrap route and no neutral accumulator convention.
pub const FIXED_HLEAF_CONTINUATION_WARP_STEP_V4: u32 = 1;

/// Setup-authenticated valid accumulator which anchors the first real HLeaf
/// transition. Its long witness/codeword lives in proving-key state; these
/// verifier-visible values bind the fixed WARP index and the History genesis
/// chain. Setup must construct this artifact and check standard WARP Decide.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeFixedHLeafVaccSeedV4 {
    pub key_digest: Digest,
    pub relation_digest: Digest,
    pub root: Digest,
    pub accumulator_digest: Digest,
    /// Setup-only digest binding the trusted index/code and complete Decided
    /// accumulator. It is not a runtime Fiat--Shamir checkpoint.
    pub genesis_anchor_digest: Digest,
}

/// The three namespaces that must not be conflated by the HLeaf adapter.
///
/// - `local_slot` locates a source inside one capacity-four leaf;
/// - `global_transition_index` locates it in History (`4 * node + slot`);
/// - `warp_step` selects the backend prior-bearing transcript.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeFixedHLeafVaccRouteV4 {
    pub local_slot: u32,
    pub node_index: u32,
    pub global_transition_index: u32,
    pub module_proof_idx: u32,
    pub has_prior: bool,
    pub warp_step: u32,
}

impl NativeFixedHLeafVaccRouteV4 {
    /// Derive the only canonical route. Slot zero at History genesis consumes
    /// the valid setup seed; slot zero in every later leaf consumes the prior
    /// leaf output. Both are ordinary prior-bearing WARP step-one updates.
    pub fn derive(
        local_slot: usize,
        global_transition_index: u32,
        module_proof_idx: u32,
    ) -> Result<Self, &'static str> {
        if local_slot >= FIXED_HLEAF_VACC_CAPACITY_V4
            || global_transition_index % FIXED_HLEAF_VACC_CAPACITY_V4 as u32 != local_slot as u32
        {
            return Err("fixed HLeaf VACC slot/global index");
        }
        Ok(Self {
            local_slot: local_slot as u32,
            node_index: global_transition_index / FIXED_HLEAF_VACC_CAPACITY_V4 as u32,
            global_transition_index,
            module_proof_idx,
            has_prior: true,
            warp_step: FIXED_HLEAF_CONTINUATION_WARP_STEP_V4,
        })
    }

    pub fn validate(self) -> Result<(), &'static str> {
        let canonical = Self::derive(
            self.local_slot as usize,
            self.global_transition_index,
            self.module_proof_idx,
        )?;
        if canonical.node_index != self.node_index
            || canonical.has_prior != self.has_prior
            || canonical.warp_step != self.warp_step
        {
            return Err("fixed HLeaf VACC seeded continuation mode");
        }
        Ok(())
    }
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeFixedHLeafVaccRouteMessageV4<T> {
    pub local_slot: T,
    pub node_index_lo: T,
    pub node_index_hi: T,
    pub global_transition_index_lo: T,
    pub global_transition_index_hi: T,
    pub module_proof_idx: T,
    pub has_prior: T,
    pub warp_step: T,
}

define_typed_lookup_bus!(
    NativeFixedHLeafVaccRouteBusV4,
    NativeFixedHLeafVaccRouteMessageV4
);

/// Canonical History-facing endpoint of a routed VACC transition.  It keeps
/// the routing namespace next to every security-critical statement field so
/// a parent adapter cannot relabel a valid transition without also proving
/// the relation/index, commitment, transcript, and accumulator chains.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeFixedHLeafVaccTransitionMessageV4<T> {
    pub route: NativeFixedHLeafVaccRouteMessageV4<T>,
    pub key_digest: [T; DIGEST_SIZE],
    pub relation_digest: [T; DIGEST_SIZE],
    pub source_forest_root: [T; DIGEST_SIZE],
    pub segment_openings_digest: [T; DIGEST_SIZE],
    pub prior_root: [T; DIGEST_SIZE],
    pub fresh_root: [T; DIGEST_SIZE],
    pub output_root: [T; DIGEST_SIZE],
    pub previous_accumulator_digest: [T; DIGEST_SIZE],
    pub output_accumulator_digest: [T; DIGEST_SIZE],
    pub previous_checkpoint_digest: [T; DIGEST_SIZE],
    pub output_checkpoint_digest: [T; DIGEST_SIZE],
    pub replay_endpoint_digest: [T; DIGEST_SIZE],
    pub replay_binding_digest: [T; DIGEST_SIZE],
}

define_typed_lookup_bus!(
    NativeFixedHLeafVaccTransitionBusV4,
    NativeFixedHLeafVaccTransitionMessageV4
);

pub const VACC_ROLE_FRESH_ROOT: usize = 0;
pub const VACC_ROLE_FRESH_ALPHA: usize = 1;
pub const VACC_ROLE_FRESH_MU: usize = 2;
pub const VACC_ROLE_FRESH_BETA_TAIL: usize = 3;
pub const VACC_ROLE_PRIOR_ROOT: usize = 4;
pub const VACC_ROLE_PRIOR_ALPHA: usize = 5;
pub const VACC_ROLE_PRIOR_MU: usize = 6;
pub const VACC_ROLE_PRIOR_BETA: usize = 7;
pub const VACC_ROLE_PRIOR_ETA: usize = 8;
pub const VACC_ROLE_FRESH_TAU: usize = 9;
pub const VACC_ROLE_OMEGA: usize = 10;
pub const VACC_ROLE_SELECTOR: usize = 11;
pub const VACC_ROLE_TWIN_SUMCHECK: usize = 12;
pub const VACC_ROLE_OUTPUT_ROOT: usize = 13;
pub const VACC_ROLE_NU: usize = 14;
pub const VACC_ROLE_ETA: usize = 15;
pub const VACC_ROLE_OOD_POINT: usize = 16;
pub const VACC_ROLE_OOD_ANSWER: usize = 17;
pub const VACC_ROLE_SHIFT: usize = 18;
pub const VACC_ROLE_XI: usize = 19;
pub const VACC_ROLE_BATCHING_SUMCHECK: usize = 20;
pub const VACC_ROLE_MU: usize = 21;
pub const VACC_ROLE_COUNT: usize = 22;

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeStandardVaccProtocolMessage<T> {
    pub proof_idx: T,
    pub start_tidx: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub has_prior: T,
    pub relation_digest: [T; DIGEST_SIZE],
}

define_typed_lookup_bus!(
    NativeStandardVaccProtocolBus,
    NativeStandardVaccProtocolMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeStandardVaccEndMessage<T> {
    pub proof_idx: T,
    pub end_tidx: T,
}

define_typed_lookup_bus!(NativeStandardVaccEndBus, NativeStandardVaccEndMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeStandardVaccRootMessage<T> {
    pub proof_idx: T,
    /// 0=fresh, 1=prior, 2=output.
    pub kind: T,
    pub root: [T; DIGEST_SIZE],
}

define_typed_lookup_bus!(NativeStandardVaccRootBus, NativeStandardVaccRootMessage);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeStandardVaccDigestMessage<T> {
    pub proof_idx: T,
    /// 0=prior, 1=output.
    pub state: T,
    pub digest: [T; DIGEST_SIZE],
}

define_typed_lookup_bus!(NativeStandardVaccDigestBus, NativeStandardVaccDigestMessage);

/// Fixed verifier-key dimensions of one homogeneous direct-AIR WARP shard.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeStandardVaccProfile {
    relation_description: Vec<u8>,
    relation_digest: Digest,
    pub num_ood: usize,
    pub num_shift_queries: usize,
    pub batching_arity: usize,
    pub log_message_len: usize,
    pub log_codeword_len: usize,
    pub initial_folding_factor: usize,
    pub log_constraints: usize,
    pub beta_len: usize,
    pub max_degree: usize,
    pub rows_per_query: usize,
}

/// Relation-independent verifier-key dimensions for one standard arity-two
/// VACC transition.  The VACC replay algebra never evaluates the direct-AIR
/// predicate; terminal Decide does.  Consequently heavy verifier AIRs are
/// shared by every relation with this exact numeric shape.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NativeStandardVaccShapeProfile {
    pub num_ood: usize,
    pub num_shift_queries: usize,
    pub batching_arity: usize,
    pub log_message_len: usize,
    pub log_codeword_len: usize,
    pub initial_folding_factor: usize,
    pub log_constraints: usize,
    pub beta_len: usize,
    pub max_degree: usize,
    pub rows_per_query: usize,
}

impl NativeStandardVaccShapeProfile {
    pub fn validate(&self) -> Result<(), &'static str> {
        let claim_count = (1 + self.num_ood + self.num_shift_queries).next_power_of_two();
        let codeword_len = 1usize
            .checked_shl(self.log_codeword_len as u32)
            .ok_or("standard direct VACC codeword length")?;
        let oracle_height = codeword_len
            .checked_shr(self.initial_folding_factor as u32)
            .ok_or("standard direct VACC oracle height")?;
        if self.log_message_len == 0
            || self.log_codeword_len < self.log_message_len
            || self.initial_folding_factor > self.log_message_len
            || self.beta_len == 0
            || self.beta_len < self.log_constraints
            || self.max_degree == 0
            || self.rows_per_query == 0
            || !self.rows_per_query.is_power_of_two()
            || self.batching_arity != claim_count
            || self.rows_per_query >= oracle_height
        {
            return Err("standard direct VACC shape profile");
        }
        Ok(())
    }
}

/// Exact relation identity absorbed by the v5 Fiat--Shamir prefix.  This is a
/// light record-scoped lane; it is deliberately not part of the heavy VACC
/// shape key and it is never interpreted as a universal PESAT selector.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeStandardVaccRelationProfile {
    pub relation_description: Vec<u8>,
    pub relation_digest: Digest,
}

impl NativeStandardVaccProfile {
    /// Construct the ordinary WARP-Verify algebra profile for an already
    /// reduced constrained-code relation.
    ///
    /// This is a shape/identity carrier for recursive `Verify`; it does not
    /// reinterpret the reduced claim as a PESAT witness.  The complete
    /// relation-specific obligation remains in terminal `Decide`.
    #[allow(clippy::too_many_arguments)]
    pub fn from_reduced_constrained_code(
        relation_description: Vec<u8>,
        relation_digest: Digest,
        warp: WarpParams,
        log_message_len: usize,
        log_codeword_len: usize,
        initial_folding_factor: usize,
        log_constraints: usize,
        beta_len: usize,
        max_degree: usize,
        rows_per_query: usize,
    ) -> Result<Self, &'static str> {
        if relation_description.is_empty()
            || relation_digest.iter().all(|value| *value == F::ZERO)
            || warp.input_arity < 2
            || !warp.input_arity.is_power_of_two()
        {
            return Err("reduced constrained-code VACC identity");
        }
        let profile = Self {
            relation_description,
            relation_digest,
            num_ood: warp.num_ood,
            num_shift_queries: warp.num_shift_queries,
            batching_arity: warp.batching_arity,
            log_message_len,
            log_codeword_len,
            initial_folding_factor,
            log_constraints,
            beta_len,
            max_degree,
            rows_per_query,
        };
        profile.validate()?;
        Ok(profile)
    }

    /// Project the common relation/code shape from an exact-finite verifier
    /// profile into the legacy standard-VACC shape carrier. The call arity is
    /// deliberately absent here: exact callers obtain it from the immutable
    /// per-call schedule and must not smuggle it through an arity-two profile.
    pub fn from_exact_finite_transcript_profile(
        exact: &NativeExactFiniteVaccTranscriptProfile,
    ) -> Result<Self, &'static str> {
        exact.validate()?;
        let shape = &exact.shape;
        let profile = Self {
            relation_description: exact.relation_description.clone(),
            relation_digest: exact.relation_digest,
            num_ood: shape.num_ood,
            num_shift_queries: shape.num_shift_queries,
            batching_arity: shape.batching_arity,
            log_message_len: shape.log_message_len,
            log_codeword_len: shape.log_codeword_len,
            initial_folding_factor: shape.initial_folding_factor,
            log_constraints: shape.log_constraints,
            beta_len: shape.beta_len,
            max_degree: shape.max_degree,
            rows_per_query: shape.rows_per_query,
        };
        profile.validate()?;
        Ok(profile)
    }

    /// Canonical relation digest paired with the private canonical
    /// description used to construct this complete profile.
    #[must_use]
    pub const fn relation_digest(&self) -> Digest {
        self.relation_digest
    }

    #[must_use]
    pub fn shape_profile(&self) -> NativeStandardVaccShapeProfile {
        NativeStandardVaccShapeProfile {
            num_ood: self.num_ood,
            num_shift_queries: self.num_shift_queries,
            batching_arity: self.batching_arity,
            log_message_len: self.log_message_len,
            log_codeword_len: self.log_codeword_len,
            initial_folding_factor: self.initial_folding_factor,
            log_constraints: self.log_constraints,
            beta_len: self.beta_len,
            max_degree: self.max_degree,
            rows_per_query: self.rows_per_query,
        }
    }

    #[must_use]
    pub fn relation_profile(&self) -> NativeStandardVaccRelationProfile {
        NativeStandardVaccRelationProfile {
            relation_description: self.relation_description.clone(),
            relation_digest: self.relation_digest,
        }
    }
    /// Derive the complete verifier-key profile directly from a reconstructed
    /// direct-AIR relation and the selected standard WARP parameters.
    ///
    /// This is the preferred setup path: it preserves the canonical relation
    /// description bytes used by the native prover, including exact degree-one
    /// canonical-zero relations for interaction-only AIRs.
    pub fn from_direct_air_relation<RF>(
        relation: &DirectAirPesatIndex<RF, Digest>,
        warp: WarpParams,
    ) -> Result<Self, &'static str>
    where
        RF: Field + serde::Serialize,
    {
        if warp.input_arity != STANDARD_DIRECT_VACC_INPUT_ARITY {
            return Err("standard direct VACC arity");
        }
        let shape = relation.pesat_shape();
        let code = relation.description().shard_key.code_class;
        let profile = Self {
            relation_description: relation.canonical_description_bytes().to_vec(),
            relation_digest: relation.description().shard_key.relation_digest,
            num_ood: warp.num_ood,
            num_shift_queries: warp.num_shift_queries,
            batching_arity: warp.batching_arity,
            log_message_len: code.log_message_len as usize,
            log_codeword_len: code.log_codeword_len as usize,
            initial_folding_factor: code.initial_folding_factor as usize,
            log_constraints: shape.log_constraints,
            beta_len: shape.beta_len(),
            max_degree: shape.max_degree,
            rows_per_query: code.rows_per_query as usize,
        };
        profile.validate()?;
        Ok(profile)
    }

    /// Derive the same arity-two VACC replay profile for the selector-free,
    /// fixed multi-AIR relation used by verifier-WARP sources.
    pub fn from_fixed_multi_air_relation<RF>(
        relation: &FixedMultiAirPesatIndex<RF, Digest>,
        warp: WarpParams,
    ) -> Result<Self, &'static str>
    where
        RF: Field + serde::Serialize,
    {
        if warp.input_arity != STANDARD_DIRECT_VACC_INPUT_ARITY {
            return Err("standard fixed multi-AIR VACC arity");
        }
        let shape = relation.pesat_shape();
        let description = relation.description();
        let code = description.code_class;
        let profile = Self {
            relation_description: relation.canonical_description_bytes().to_vec(),
            relation_digest: description.relation_digest,
            num_ood: warp.num_ood,
            num_shift_queries: warp.num_shift_queries,
            batching_arity: warp.batching_arity,
            log_message_len: code.log_message_len as usize,
            log_codeword_len: code.log_codeword_len as usize,
            initial_folding_factor: code.initial_folding_factor as usize,
            log_constraints: shape.log_constraints,
            beta_len: shape.beta_len(),
            max_degree: shape.max_degree,
            rows_per_query: code.rows_per_query as usize,
        };
        profile.validate()?;
        Ok(profile)
    }

    /// Derive the same profile from the immutable relation description stored
    /// in a direct-AIR registry entry. This avoids requiring a reconstructed
    /// relation merely to build the recursive verifier key.
    pub fn from_direct_air_description(
        description: &DirectAirPesatRelationDescription<Digest>,
        warp: WarpParams,
    ) -> Result<Self, &'static str> {
        if warp.input_arity != STANDARD_DIRECT_VACC_INPUT_ARITY
            || description.constraint_count == 0
            || description.exact_max_degree == 0
            || description.exact_max_degree > description.warp_degree_envelope
        {
            return Err("standard direct VACC relation metadata");
        }
        let constraints = usize::try_from(description.constraint_count)
            .map_err(|_| "standard direct VACC constraint count")?;
        let log_constraints = constraints
            .checked_next_power_of_two()
            .ok_or("standard direct VACC constraint count")?
            .ilog2() as usize;
        let explicit_len = 1usize
            .checked_add(description.shard_key.public_schema.public_values_len as usize)
            .and_then(|value| {
                value.checked_add(description.shard_key.public_schema.boundary_values_len as usize)
            })
            .ok_or("standard direct VACC explicit length")?;
        let code = description.shard_key.code_class;
        let profile = Self {
            relation_description: description
                .canonical_bytes()
                .map_err(|_| "standard direct VACC relation encoding")?,
            relation_digest: description.shard_key.relation_digest,
            num_ood: warp.num_ood,
            num_shift_queries: warp.num_shift_queries,
            batching_arity: warp.batching_arity,
            log_message_len: code.log_message_len as usize,
            log_codeword_len: code.log_codeword_len as usize,
            initial_folding_factor: code.initial_folding_factor as usize,
            log_constraints,
            beta_len: log_constraints
                .checked_add(explicit_len)
                .ok_or("standard direct VACC beta length")?,
            max_degree: description.warp_degree_envelope as usize,
            rows_per_query: code.rows_per_query as usize,
        };
        profile.validate()?;
        Ok(profile)
    }

    pub fn validate(&self) -> Result<(), &'static str> {
        self.shape_profile().validate()?;
        if self.relation_description.is_empty() {
            return Err("standard direct VACC profile");
        }
        Ok(())
    }
}

/// Verifier-key shape of one invocation in the exact-finite schedule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct NativeExactFiniteVaccCallProfile {
    pub active: bool,
    pub input_arity: usize,
    pub fresh_count: usize,
    pub prior_count: usize,
}

impl NativeExactFiniteVaccCallProfile {
    #[must_use]
    pub const fn inactive() -> Self {
        Self {
            active: false,
            input_arity: 0,
            fresh_count: 0,
            prior_count: 0,
        }
    }
}

/// Complete key-fixed transcript profile for the exact-finite Appendix-D
/// verifier.  `external_index_binding` is observed by the native verifier as
/// extension-field elements; the digest fields are statement identities used
/// by recursive callers. Setup must derive all three from the same immutable
/// indexed relation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NativeExactFiniteVaccTranscriptProfile {
    pub relation_description: Vec<u8>,
    pub external_index_binding: Vec<EF>,
    pub relation_digest: Digest,
    pub index_digest: Digest,
    pub setup_digest: Digest,
    pub schedule_digest: Digest,
    pub shape: NativeStandardVaccShapeProfile,
    pub max_input_arity: usize,
    pub calls: [NativeExactFiniteVaccCallProfile; STANDARD_EXACT_FINITE_VACC_MAX_CALLS],
}

/// Key-derived algebra dimensions for one exact-finite invocation. Consumers
/// use this instead of independently configuring the twin sumcheck, which
/// prevents a six-round arity-64 transcript from being verified by an
/// arity-two one-round AIR (or vice versa).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeExactFiniteVaccCallAlgebraProfile {
    pub input_arity: usize,
    pub twin_rounds: usize,
    pub twin_last_round: usize,
    pub twin_degree: usize,
    pub batching_rounds: usize,
    pub batching_degree: usize,
}

impl NativeExactFiniteVaccCallAlgebraProfile {
    /// Construct the only coefficient-sumcheck AIR compatible with this call
    /// key. Its transition constraints enumerate every twin round from zero
    /// through `log2(input_arity)-1` before emitting the final-round message.
    #[must_use]
    pub fn coefficient_sumcheck_air(
        self,
        transcript_bus: TranscriptBus,
        round_bus: NativeSumcheckRoundBus,
        initial_bus: NativeSumcheckInitialBus,
        challenge_bus: NativeSumcheckChallengeBus,
    ) -> NativeCoefficientSumcheckAir {
        NativeCoefficientSumcheckAir {
            transcript_bus,
            round_bus,
            initial_bus,
            challenge_bus,
            twin_degree: self.twin_degree,
            batching_degree: self.batching_degree,
            twin_rounds: self.twin_rounds,
            batching_rounds: self.batching_rounds,
        }
    }

    #[must_use]
    pub fn twin_sigma_air(
        self,
        claim_bus: NativeClaimValueBus,
        eq_bus: NativeEqResultBus,
        sumcheck_initial_bus: NativeSumcheckInitialBus,
        omega_bus: NativeTwinOmegaBus,
    ) -> NativeTwinSigmaAir {
        NativeTwinSigmaAir {
            claim_bus,
            eq_bus,
            sumcheck_initial_bus,
            input_arity: self.input_arity,
            omega_bus,
        }
    }

    #[must_use]
    pub fn twin_fold_air(
        self,
        claim_bus: NativeClaimValueBus,
        eq_bus: NativeEqResultBus,
        folded_bus: NativeFoldedClaimBus,
    ) -> NativeTwinFoldAir {
        NativeTwinFoldAir {
            claim_bus,
            eq_bus,
            folded_bus,
            input_arity: self.input_arity,
            weight_group_offset: self.input_arity,
        }
    }

    #[must_use]
    pub fn twin_final_air(
        self,
        sumcheck_round_bus: NativeSumcheckRoundBus,
        eq_bus: NativeEqResultBus,
        scalar_bus: NativeTwinScalarBus,
        selector_eq_group: usize,
        omega_bus: NativeTwinOmegaBus,
        transcript_bus: TranscriptBus,
    ) -> NativeTwinFinalAir {
        NativeTwinFinalAir {
            sumcheck_round_bus,
            eq_bus,
            scalar_bus,
            last_round: self.twin_last_round,
            selector_eq_group,
            omega_bus,
            transcript_bus,
        }
    }
}

impl NativeExactFiniteVaccTranscriptProfile {
    pub fn validate(&self) -> Result<(), &'static str> {
        self.shape.validate()?;
        if self.relation_description.is_empty()
            || self.external_index_binding.is_empty()
            || self.max_input_arity < 2
            || self.max_input_arity > STANDARD_EXACT_FINITE_VACC_MAX_INPUT_ARITY
            || !self.max_input_arity.is_power_of_two()
        {
            return Err("exact-finite VACC transcript profile");
        }
        for digest in [
            self.relation_digest,
            self.index_digest,
            self.setup_digest,
            self.schedule_digest,
        ] {
            if digest.iter().all(|value| *value == F::ZERO) {
                return Err("exact-finite VACC unset digest");
            }
        }
        let mut inactive = false;
        let mut active_calls = 0usize;
        for (call_index, call) in self.calls.iter().copied().enumerate() {
            if !call.active {
                inactive = true;
                if call != NativeExactFiniteVaccCallProfile::inactive() {
                    return Err("exact-finite VACC non-canonical inactive call");
                }
                continue;
            }
            if inactive
                || call.input_arity < 2
                || call.input_arity > self.max_input_arity
                || !call.input_arity.is_power_of_two()
                || call.prior_count != usize::from(call_index != 0)
                || call.fresh_count + call.prior_count != call.input_arity
            {
                return Err("exact-finite VACC call schedule");
            }
            active_calls += 1;
        }
        if active_calls == 0 {
            return Err("exact-finite VACC empty schedule");
        }
        Ok(())
    }

    #[must_use]
    pub fn active_call_count(&self) -> usize {
        self.calls.iter().take_while(|call| call.active).count()
    }

    #[must_use]
    pub fn total_fresh(&self) -> usize {
        self.calls.iter().map(|call| call.fresh_count).sum()
    }

    pub fn call_algebra_profile(
        &self,
        call_index: usize,
    ) -> Result<NativeExactFiniteVaccCallAlgebraProfile, &'static str> {
        self.validate()?;
        let call = self
            .calls
            .get(call_index)
            .copied()
            .filter(|call| call.active)
            .ok_or("exact-finite inactive call")?;
        let twin_rounds = call.input_arity.ilog2() as usize;
        Ok(NativeExactFiniteVaccCallAlgebraProfile {
            input_arity: call.input_arity,
            twin_rounds,
            twin_last_round: twin_rounds - 1,
            twin_degree: twin_degree(
                self.shape.log_codeword_len,
                self.shape.log_constraints,
                self.shape.max_degree,
            ),
            batching_rounds: self.shape.log_codeword_len,
            batching_degree: 2,
        })
    }
}

/// Canonical extension-field observations made by the native exact-finite
/// schedule binder. This is also a differential-test oracle for recursive
/// transcript gadgets.
pub fn native_exact_finite_vacc_schedule_prefix_elements(
    profile: &NativeExactFiniteVaccTranscriptProfile,
) -> Result<Vec<EF>, &'static str> {
    profile.validate()?;
    let mut values = Vec::new();
    push_bytes_as_ext(&mut values, EXACT_FINITE_WARP_PROTOCOL_TAG);
    push_u64_as_ext(&mut values, EXACT_FINITE_WARP_TRANSCRIPT_VERSION);
    push_u64_as_ext(&mut values, profile.relation_description.len() as u64);
    push_bytes_as_ext(&mut values, &profile.relation_description);
    push_u64_as_ext(&mut values, 1);
    push_u64_as_ext(&mut values, profile.external_index_binding.len() as u64);
    values.extend_from_slice(&profile.external_index_binding);
    let explicit_len = profile.shape.beta_len - profile.shape.log_constraints;
    for value in [
        profile.shape.log_constraints,
        profile.shape.log_message_len,
        explicit_len,
        profile.shape.max_degree,
        profile.max_input_arity,
        profile.shape.num_ood,
        profile.shape.num_shift_queries,
        profile.shape.batching_arity,
        profile.shape.log_message_len,
        profile.shape.log_codeword_len,
        profile.total_fresh(),
        profile.active_call_count(),
    ] {
        push_u64_as_ext(&mut values, value as u64);
    }
    push_u64_as_ext(&mut values, 1);
    for (call_index, call) in profile
        .calls
        .iter()
        .take(profile.active_call_count())
        .enumerate()
    {
        for value in [
            call_index,
            call.input_arity,
            call.fresh_count,
            call.prior_count,
        ] {
            push_u64_as_ext(&mut values, value as u64);
        }
    }
    Ok(values)
}

/// Canonical extension-field observations made by the native Appendix-D VACC
/// prefix for one exact-finite invocation.
pub fn native_exact_finite_vacc_call_prefix_elements(
    profile: &NativeExactFiniteVaccTranscriptProfile,
    call_index: usize,
) -> Result<Vec<EF>, &'static str> {
    profile.validate()?;
    let call = profile
        .calls
        .get(call_index)
        .copied()
        .filter(|call| call.active)
        .ok_or("exact-finite inactive call")?;
    let mut values = Vec::new();
    push_bytes_as_ext(&mut values, NATIVE_WARP_VACC_APPENDIX_D_PROTOCOL_TAG);
    push_u64_as_ext(&mut values, 1);
    push_u64_as_ext(&mut values, profile.external_index_binding.len() as u64);
    values.extend_from_slice(&profile.external_index_binding);
    for value in [
        call.input_arity,
        profile.shape.num_ood,
        profile.shape.num_shift_queries,
        profile.shape.batching_arity,
        profile.shape.log_message_len,
        profile.shape.log_codeword_len,
        call_index,
        call.fresh_count,
        call.prior_count,
    ] {
        push_u64_as_ext(&mut values, value as u64);
    }
    Ok(values)
}

fn push_bytes_as_ext(values: &mut Vec<EF>, bytes: &[u8]) {
    values.extend(bytes.iter().copied().map(EF::from_u8));
}

fn push_u64_as_ext(values: &mut Vec<EF>, value: u64) {
    push_bytes_as_ext(values, &value.to_le_bytes());
}

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeExactFiniteVaccScheduleMessage<T> {
    pub proof_idx: T,
    pub end_tidx: T,
    pub call_count: T,
    pub total_fresh: T,
    pub relation_digest: [T; DIGEST_SIZE],
    pub index_digest: [T; DIGEST_SIZE],
    pub setup_digest: [T; DIGEST_SIZE],
    pub schedule_digest: [T; DIGEST_SIZE],
}

define_typed_lookup_bus!(
    NativeExactFiniteVaccScheduleBus,
    NativeExactFiniteVaccScheduleMessage
);

#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct NativeExactFiniteVaccCallProtocolMessage<T> {
    pub proof_idx: T,
    pub start_tidx: T,
    pub call_index: T,
    pub input_arity: T,
    pub fresh_count: T,
    pub prior_count: T,
    pub relation_digest: [T; DIGEST_SIZE],
    pub index_digest: [T; DIGEST_SIZE],
    pub setup_digest: [T; DIGEST_SIZE],
    pub schedule_digest: [T; DIGEST_SIZE],
}

define_typed_lookup_bus!(
    NativeExactFiniteVaccCallProtocolBus,
    NativeExactFiniteVaccCallProtocolMessage
);

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeExactFiniteVaccSchedulePrefixCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub start_tidx: T,
}

/// Replays `bind_exact_finite_schedule` from the standard backend. All
/// dimensions, relation bytes, external binding and calls are verifier-key
/// constants, so the only witness-selected values are routing coordinates.
#[derive(ColumnsAir)]
#[columns_via(NativeExactFiniteVaccSchedulePrefixCols<u8>)]
pub struct NativeExactFiniteVaccSchedulePrefixAir {
    pub transcript_bus: TranscriptBus,
    pub schedule_bus: NativeExactFiniteVaccScheduleBus,
    pub profile: NativeExactFiniteVaccTranscriptProfile,
    /// Total setup-fixed consumers of the schedule authority.
    pub lookup_count: usize,
}

impl BaseAirWithPublicValues<F> for NativeExactFiniteVaccSchedulePrefixAir {}
impl PartitionedBaseAir<F> for NativeExactFiniteVaccSchedulePrefixAir {}
impl BaseAir<F> for NativeExactFiniteVaccSchedulePrefixAir {
    fn width(&self) -> usize {
        NativeExactFiniteVaccSchedulePrefixCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB>
    for NativeExactFiniteVaccSchedulePrefixAir
{
    fn eval(&self, builder: &mut AB) {
        assert!(
            self.profile.validate().is_ok()
                && (1..=STANDARD_EXACT_FINITE_VACC_MAX_LOOKUP_COUNT).contains(&self.lookup_count),
            "invalid exact-finite VACC schedule profile"
        );
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("exact-finite VACC schedule prefix row");
        let local: &NativeExactFiniteVaccSchedulePrefixCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);

        let mut tidx: AB::Expr = local.start_tidx.into();
        observe_bytes_const(
            &self.transcript_bus,
            builder,
            local.proof_idx,
            &mut tidx,
            EXACT_FINITE_WARP_PROTOCOL_TAG,
            local.active,
        );
        observe_u64_const(
            &self.transcript_bus,
            builder,
            local.proof_idx,
            &mut tidx,
            EXACT_FINITE_WARP_TRANSCRIPT_VERSION,
            local.active,
        );
        observe_u64_const(
            &self.transcript_bus,
            builder,
            local.proof_idx,
            &mut tidx,
            self.profile.relation_description.len() as u64,
            local.active,
        );
        observe_bytes_const(
            &self.transcript_bus,
            builder,
            local.proof_idx,
            &mut tidx,
            &self.profile.relation_description,
            local.active,
        );
        // `externally_bound_relation = Some(...)`.
        observe_u64_const(
            &self.transcript_bus,
            builder,
            local.proof_idx,
            &mut tidx,
            1,
            local.active,
        );
        observe_u64_const(
            &self.transcript_bus,
            builder,
            local.proof_idx,
            &mut tidx,
            self.profile.external_index_binding.len() as u64,
            local.active,
        );
        for value in &self.profile.external_index_binding {
            let coefficients: &[F] = value.as_basis_coefficients_slice();
            self.transcript_bus.observe_ext(
                builder,
                local.proof_idx,
                tidx.clone(),
                core::array::from_fn(|limb| AB::Expr::from(coefficients[limb])),
                local.active,
            );
            tidx += AB::Expr::from_usize(D_EF);
        }
        let explicit_len = self.profile.shape.beta_len - self.profile.shape.log_constraints;
        for value in [
            self.profile.shape.log_constraints,
            self.profile.shape.log_message_len,
            explicit_len,
            self.profile.shape.max_degree,
            self.profile.max_input_arity,
            self.profile.shape.num_ood,
            self.profile.shape.num_shift_queries,
            self.profile.shape.batching_arity,
            self.profile.shape.log_message_len,
            self.profile.shape.log_codeword_len,
            self.profile.total_fresh(),
            self.profile.active_call_count(),
        ] {
            observe_u64_const(
                &self.transcript_bus,
                builder,
                local.proof_idx,
                &mut tidx,
                value as u64,
                local.active,
            );
        }
        // `WarpFreshCodewordAlphabet::AppendixDBaseV1`.
        observe_u64_const(
            &self.transcript_bus,
            builder,
            local.proof_idx,
            &mut tidx,
            1,
            local.active,
        );
        for (call_index, call) in self
            .profile
            .calls
            .iter()
            .take(self.profile.active_call_count())
            .enumerate()
        {
            for value in [
                call_index,
                call.input_arity,
                call.fresh_count,
                call.prior_count,
            ] {
                observe_u64_const(
                    &self.transcript_bus,
                    builder,
                    local.proof_idx,
                    &mut tidx,
                    value as u64,
                    local.active,
                );
            }
        }
        self.schedule_bus.add_key_with_lookups(
            builder,
            NativeExactFiniteVaccScheduleMessage {
                proof_idx: local.proof_idx.into(),
                end_tidx: tidx,
                call_count: AB::Expr::from_usize(self.profile.active_call_count()),
                total_fresh: AB::Expr::from_usize(self.profile.total_fresh()),
                relation_digest: self.profile.relation_digest.map(AB::Expr::from),
                index_digest: self.profile.index_digest.map(AB::Expr::from),
                setup_digest: self.profile.setup_digest.map(AB::Expr::from),
                schedule_digest: self.profile.schedule_digest.map(AB::Expr::from),
            },
            AB::Expr::from(local.active) * AB::Expr::from_usize(self.lookup_count),
        );
    }
}

pub fn generate_native_exact_finite_vacc_schedule_prefix_trace(
    proof_idx: usize,
    start_tidx: usize,
) -> RowMajorMatrix<F> {
    let width = NativeExactFiniteVaccSchedulePrefixCols::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut NativeExactFiniteVaccSchedulePrefixCols<F> = values.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.proof_idx = F::from_usize(proof_idx);
    cols.start_tidx = F::from_usize(start_tidx);
    RowMajorMatrix::new(values, width)
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeExactFiniteVaccCallPrefixCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub start_tidx: T,
}

/// Replays the standard Appendix-D VACC prefix for one fixed exact-finite
/// invocation. The call profile is selected at key generation, including all
/// `log2(input_arity)`-relevant dimensions.
#[derive(ColumnsAir)]
#[columns_via(NativeExactFiniteVaccCallPrefixCols<u8>)]
pub struct NativeExactFiniteVaccCallPrefixAir {
    pub transcript_bus: TranscriptBus,
    pub phase_cursor_bus: NativeVaccPhaseCursorBus,
    pub protocol_bus: NativeExactFiniteVaccCallProtocolBus,
    pub legacy_protocol_bus: NativeStandardVaccProtocolBus,
    pub profile: NativeExactFiniteVaccTranscriptProfile,
    pub call_index: usize,
    /// Total setup-fixed consumers of this call-protocol authority.
    pub lookup_count: usize,
}

impl BaseAirWithPublicValues<F> for NativeExactFiniteVaccCallPrefixAir {}
impl PartitionedBaseAir<F> for NativeExactFiniteVaccCallPrefixAir {}
impl BaseAir<F> for NativeExactFiniteVaccCallPrefixAir {
    fn width(&self) -> usize {
        NativeExactFiniteVaccCallPrefixCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeExactFiniteVaccCallPrefixAir {
    fn eval(&self, builder: &mut AB) {
        assert!(
            self.profile.validate().is_ok()
                && self.call_index < self.profile.active_call_count()
                && (1..=STANDARD_EXACT_FINITE_VACC_MAX_LOOKUP_COUNT).contains(&self.lookup_count),
            "invalid exact-finite VACC call profile"
        );
        let call = self.profile.calls[self.call_index];
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("exact-finite VACC call prefix row");
        let local: &NativeExactFiniteVaccCallPrefixCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        let mut tidx: AB::Expr = local.start_tidx.into();
        observe_bytes_const(
            &self.transcript_bus,
            builder,
            local.proof_idx,
            &mut tidx,
            NATIVE_WARP_VACC_APPENDIX_D_PROTOCOL_TAG,
            local.active,
        );
        observe_u64_const(
            &self.transcript_bus,
            builder,
            local.proof_idx,
            &mut tidx,
            1,
            local.active,
        );
        observe_u64_const(
            &self.transcript_bus,
            builder,
            local.proof_idx,
            &mut tidx,
            self.profile.external_index_binding.len() as u64,
            local.active,
        );
        for value in &self.profile.external_index_binding {
            let coefficients: &[F] = value.as_basis_coefficients_slice();
            self.transcript_bus.observe_ext(
                builder,
                local.proof_idx,
                tidx.clone(),
                core::array::from_fn(|limb| AB::Expr::from(coefficients[limb])),
                local.active,
            );
            tidx += AB::Expr::from_usize(D_EF);
        }
        for value in [
            call.input_arity,
            self.profile.shape.num_ood,
            self.profile.shape.num_shift_queries,
            self.profile.shape.batching_arity,
            self.profile.shape.log_message_len,
            self.profile.shape.log_codeword_len,
            self.call_index,
            call.fresh_count,
            call.prior_count,
        ] {
            observe_u64_const(
                &self.transcript_bus,
                builder,
                local.proof_idx,
                &mut tidx,
                value as u64,
                local.active,
            );
        }
        self.protocol_bus.add_key_with_lookups(
            builder,
            NativeExactFiniteVaccCallProtocolMessage {
                proof_idx: local.proof_idx.into(),
                start_tidx: local.start_tidx.into(),
                call_index: AB::Expr::from_usize(self.call_index),
                input_arity: AB::Expr::from_usize(call.input_arity),
                fresh_count: AB::Expr::from_usize(call.fresh_count),
                prior_count: AB::Expr::from_usize(call.prior_count),
                relation_digest: self.profile.relation_digest.map(AB::Expr::from),
                index_digest: self.profile.index_digest.map(AB::Expr::from),
                setup_digest: self.profile.setup_digest.map(AB::Expr::from),
                schedule_digest: self.profile.schedule_digest.map(AB::Expr::from),
            },
            AB::Expr::from(local.active) * AB::Expr::from_usize(self.lookup_count),
        );
        self.legacy_protocol_bus.add_key_with_lookups(
            builder,
            NativeStandardVaccProtocolMessage {
                proof_idx: local.proof_idx.into(),
                start_tidx: local.start_tidx.into(),
                segment_index_lo: AB::Expr::from_usize(self.call_index),
                segment_index_hi: AB::Expr::ZERO,
                has_prior: AB::Expr::from_usize(call.prior_count),
                relation_digest: self.profile.relation_digest.map(AB::Expr::from),
            },
            local.active,
        );
        self.phase_cursor_bus.send(
            builder,
            NativeVaccPhaseCursorMessage {
                proof_idx: local.proof_idx.into(),
                boundary: AB::Expr::ONE,
                tidx,
            },
            // Exact mode has two independent consumers of the canonical
            // post-prefix cursor: semantic role parsing and stacked fresh-
            // commitment authentication. The legacy boundary-zero consumer
            // does not exist in this mode; the call start is instead bound by
            // the schedule/authority and resume-checkpoint interfaces.
            AB::Expr::from(local.active) * AB::Expr::from_usize(2),
        );
    }
}

pub fn generate_native_exact_finite_vacc_call_prefix_trace(
    proof_idx: usize,
    start_tidx: usize,
) -> RowMajorMatrix<F> {
    let width = NativeExactFiniteVaccCallPrefixCols::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut NativeExactFiniteVaccCallPrefixCols<F> = values.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.proof_idx = F::from_usize(proof_idx);
    cols.start_tidx = F::from_usize(start_tidx);
    RowMajorMatrix::new(values, width)
}

fn observe_bytes_const<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Var,
    tidx: &mut AB::Expr,
    bytes: &[u8],
    enabled: AB::Var,
) {
    for &byte in bytes {
        observe_ext_const(
            bus,
            builder,
            proof_idx,
            tidx.clone(),
            F::from_u8(byte),
            enabled,
        );
        *tidx += AB::Expr::from_usize(D_EF);
    }
}

fn observe_u64_const<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Var,
    tidx: &mut AB::Expr,
    value: u64,
    enabled: AB::Var,
) {
    observe_bytes_const(bus, builder, proof_idx, tidx, &value.to_le_bytes(), enabled);
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeStandardVaccPrefixCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub start_tidx: T,
    pub start_nonzero: T,
    pub start_inverse: T,
    pub segment_index_lo: T,
    pub segment_index_hi: T,
    pub has_prior: T,
    pub step_bytes: [T; 8],
    pub step_byte_bits: [[T; 8]; 8],
}

/// Certifies the exact `openvm-native-warp-vacc-all-ef-v5` protocol prefix.
/// The relation description and all dimensions are verifier-key constants.
#[derive(ColumnsAir)]
#[columns_via(NativeStandardVaccPrefixCols<u8>)]
pub struct NativeStandardVaccPrefixAir {
    pub transcript_bus: TranscriptBus,
    pub phase_cursor_bus: NativeVaccPhaseCursorBus,
    pub protocol_bus: NativeStandardVaccProtocolBus,
    pub shape: NativeStandardVaccShapeProfile,
    pub relation: NativeStandardVaccRelationProfile,
}

impl BaseAirWithPublicValues<F> for NativeStandardVaccPrefixAir {}
impl PartitionedBaseAir<F> for NativeStandardVaccPrefixAir {}
impl BaseAir<F> for NativeStandardVaccPrefixAir {
    fn width(&self) -> usize {
        NativeStandardVaccPrefixCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeStandardVaccPrefixAir {
    fn eval(&self, builder: &mut AB) {
        assert!(
            self.shape.validate().is_ok() && !self.relation.relation_description.is_empty(),
            "invalid standard VACC profile"
        );
        let main = builder.main();
        let row = main.row_slice(0).expect("standard VACC prefix row");
        let local: &NativeStandardVaccPrefixCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder.assert_bool(local.has_prior);
        builder.assert_bool(local.start_nonzero);
        builder
            .when(local.active * local.start_nonzero)
            .assert_one(local.start_tidx * local.start_inverse);
        builder
            .when(local.active * (AB::Expr::ONE - local.start_nonzero))
            .assert_zero(local.start_tidx);
        builder
            .when(local.active * (AB::Expr::ONE - local.start_nonzero))
            .assert_zero(local.start_inverse);
        for (byte, bits) in local.step_bytes.into_iter().zip(local.step_byte_bits) {
            for bit in bits {
                builder.assert_bool(bit);
            }
            let recomposed = bits
                .into_iter()
                .enumerate()
                .fold(AB::Expr::ZERO, |value, (index, bit)| {
                    value + bit * AB::Expr::from_usize(1 << index)
                });
            builder.when(local.active).assert_eq(byte, recomposed);
        }
        builder.when(local.active).assert_eq(
            local.segment_index_lo * local.has_prior,
            local.step_bytes[0] + local.step_bytes[1] * AB::Expr::from_u32(256),
        );
        builder.when(local.active).assert_eq(
            local.segment_index_hi * local.has_prior,
            local.step_bytes[2] + local.step_bytes[3] * AB::Expr::from_u32(256),
        );
        for byte in &local.step_bytes[4..] {
            builder.when(local.active).assert_zero(*byte);
        }
        for byte in local.step_bytes {
            builder
                .when(local.active * (AB::Expr::ONE - local.has_prior))
                .assert_zero(byte);
        }

        let mut tidx: AB::Expr = local.start_tidx.into();
        for &byte in NATIVE_WARP_VACC_PROTOCOL_TAG {
            observe_ext_const(
                &self.transcript_bus,
                builder,
                local.proof_idx,
                tidx.clone(),
                F::from_u8(byte),
                local.active,
            );
            tidx += AB::Expr::from_usize(D_EF);
        }
        // `externally_bound_relation = None`.
        for byte in 0u64.to_le_bytes() {
            observe_ext_const(
                &self.transcript_bus,
                builder,
                local.proof_idx,
                tidx.clone(),
                F::from_u8(byte),
                local.active,
            );
            tidx += AB::Expr::from_usize(D_EF);
        }
        for byte in (self.relation.relation_description.len() as u64).to_le_bytes() {
            observe_ext_const(
                &self.transcript_bus,
                builder,
                local.proof_idx,
                tidx.clone(),
                F::from_u8(byte),
                local.active,
            );
            tidx += AB::Expr::from_usize(D_EF);
        }
        for &byte in &self.relation.relation_description {
            observe_ext_const(
                &self.transcript_bus,
                builder,
                local.proof_idx,
                tidx.clone(),
                F::from_u8(byte),
                local.active,
            );
            tidx += AB::Expr::from_usize(D_EF);
        }
        for value in [
            STANDARD_DIRECT_VACC_INPUT_ARITY,
            self.shape.num_ood,
            self.shape.num_shift_queries,
            self.shape.batching_arity,
            self.shape.log_message_len,
            self.shape.log_codeword_len,
        ] {
            for byte in (value as u64).to_le_bytes() {
                observe_ext_const(
                    &self.transcript_bus,
                    builder,
                    local.proof_idx,
                    tidx.clone(),
                    F::from_u8(byte),
                    local.active,
                );
                tidx += AB::Expr::from_usize(D_EF);
            }
        }
        // Backend v5 uses segment index for continuation steps and zero for a
        // shard's bootstrap step.
        for step_byte in local.step_bytes {
            self.transcript_bus.observe_ext(
                builder,
                local.proof_idx,
                tidx.clone(),
                [
                    step_byte.into(),
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                    AB::Expr::ZERO,
                ],
                local.active,
            );
            tidx += AB::Expr::from_usize(D_EF);
        }
        for value in [1usize, 0usize] {
            for byte_index in 0..8 {
                let prior = if value == 0 && byte_index == 0 {
                    AB::Expr::from(local.has_prior)
                } else if value == 1 && byte_index == 0 {
                    AB::Expr::ONE
                } else {
                    AB::Expr::ZERO
                };
                self.transcript_bus.observe_ext(
                    builder,
                    local.proof_idx,
                    tidx.clone(),
                    [prior, AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO],
                    local.active,
                );
                tidx += AB::Expr::from_usize(D_EF);
            }
        }
        self.protocol_bus.add_key_with_lookups(
            builder,
            NativeStandardVaccProtocolMessage {
                proof_idx: local.proof_idx.into(),
                start_tidx: local.start_tidx.into(),
                segment_index_lo: local.segment_index_lo.into(),
                segment_index_hi: local.segment_index_hi.into(),
                has_prior: local.has_prior.into(),
                relation_digest: self.relation.relation_digest.map(AB::Expr::from),
            },
            local.active,
        );
        self.phase_cursor_bus.send(
            builder,
            NativeVaccPhaseCursorMessage {
                proof_idx: local.proof_idx.into(),
                boundary: AB::Expr::ZERO,
                tidx: local.start_tidx.into(),
            },
            local.active * local.start_nonzero,
        );
        self.phase_cursor_bus.send(
            builder,
            NativeVaccPhaseCursorMessage {
                proof_idx: local.proof_idx.into(),
                boundary: AB::Expr::ONE,
                tidx,
            },
            local.active,
        );
    }
}

fn observe_ext_const<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Var,
    tidx: AB::Expr,
    value: F,
    enabled: AB::Var,
) {
    bus.observe_ext(
        builder,
        proof_idx,
        tidx,
        [
            AB::Expr::from(value),
            AB::Expr::ZERO,
            AB::Expr::ZERO,
            AB::Expr::ZERO,
        ],
        enabled,
    );
}

pub fn generate_native_standard_vacc_prefix_trace(
    proof_idx: usize,
    start_tidx: usize,
    segment_index: u32,
    has_prior: bool,
) -> RowMajorMatrix<F> {
    let width = NativeStandardVaccPrefixCols::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut NativeStandardVaccPrefixCols<F> = values.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.proof_idx = F::from_usize(proof_idx);
    cols.start_tidx = F::from_usize(start_tidx);
    cols.start_nonzero = F::from_bool(start_tidx != 0);
    cols.start_inverse = if start_tidx == 0 {
        F::ZERO
    } else {
        F::from_usize(start_tidx).inverse()
    };
    cols.segment_index_lo = F::from_u32(segment_index & 0xffff);
    cols.segment_index_hi = F::from_u32(segment_index >> 16);
    cols.has_prior = F::from_bool(has_prior);
    let step = if has_prior {
        u64::from(segment_index)
    } else {
        0
    };
    for (byte_index, byte) in step.to_le_bytes().into_iter().enumerate() {
        cols.step_bytes[byte_index] = F::from_u8(byte);
        for bit in 0..8 {
            cols.step_byte_bits[byte_index][bit] = F::from_bool(((byte >> bit) & 1) == 1);
        }
    }
    RowMajorMatrix::new(values, width)
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeStandardVaccEndCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub tag_tidx: T,
    pub end_tidx: T,
    pub discarded_sample: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(NativeStandardVaccEndCols<u8>)]
pub struct NativeStandardVaccEndAir {
    pub transcript_bus: TranscriptBus,
    /// Present when the transcript starts from an authenticated resume
    /// checkpoint. In that mode `TranscriptAir` publishes its exact terminal
    /// cursor and the VACC end row must bind it to the protocol schedule.
    pub transcript_end_index_bus: Option<TranscriptEndIndexBus>,
    pub phase_cursor_bus: NativeVaccPhaseCursorBus,
    pub end_bus: NativeStandardVaccEndBus,
    pub end_tag: u64,
    /// Legacy per-shard VACC observes one base-field separator. Exact-finite
    /// orchestration observes its separator through `AlgebraicChallenger`, so
    /// the same scalar is encoded as one extension-field element. This mode is
    /// verifier-key data, never proof-selected.
    pub extension_tag: bool,
}

impl BaseAirWithPublicValues<F> for NativeStandardVaccEndAir {}
impl PartitionedBaseAir<F> for NativeStandardVaccEndAir {}
impl BaseAir<F> for NativeStandardVaccEndAir {
    fn width(&self) -> usize {
        NativeStandardVaccEndCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeStandardVaccEndAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("standard VACC end row");
        let local: &NativeStandardVaccEndCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        self.phase_cursor_bus.receive(
            builder,
            NativeVaccPhaseCursorMessage {
                proof_idx: local.proof_idx.into(),
                boundary: AB::Expr::from_usize(2),
                tidx: local.tag_tidx.into(),
            },
            local.active,
        );
        let sample_tidx = if self.extension_tag {
            self.transcript_bus.observe_ext(
                builder,
                local.proof_idx,
                local.tag_tidx,
                core::array::from_fn(|limb| {
                    if limb == 0 {
                        AB::Expr::from_u64(self.end_tag)
                    } else {
                        AB::Expr::ZERO
                    }
                }),
                local.active,
            );
            local.tag_tidx + AB::Expr::from_usize(D_EF)
        } else {
            self.transcript_bus.observe(
                builder,
                local.proof_idx,
                local.tag_tidx,
                AB::Expr::from_u64(self.end_tag),
                local.active,
            );
            local.tag_tidx + AB::Expr::ONE
        };
        self.transcript_bus.sample_ext(
            builder,
            local.proof_idx,
            sample_tidx.clone(),
            local.discarded_sample,
            local.active,
        );
        builder
            .when(local.active)
            .assert_eq(local.end_tidx, sample_tidx + AB::Expr::from_usize(D_EF));
        if let Some(transcript_end_index_bus) = self.transcript_end_index_bus {
            transcript_end_index_bus.receive(
                builder,
                local.proof_idx,
                TranscriptEndIndexMessage {
                    tidx: local.end_tidx.into(),
                },
                local.active,
            );
        }
        self.end_bus.add_key_with_lookups(
            builder,
            NativeStandardVaccEndMessage {
                proof_idx: local.proof_idx.into(),
                end_tidx: local.end_tidx.into(),
            },
            local.active,
        );
    }
}

pub fn generate_native_standard_vacc_end_trace(
    proof_idx: usize,
    tag_tidx: usize,
    end_tidx: usize,
    discarded_sample: EF,
) -> RowMajorMatrix<F> {
    let width = NativeStandardVaccEndCols::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut NativeStandardVaccEndCols<F> = values.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.proof_idx = F::from_usize(proof_idx);
    cols.tag_tidx = F::from_usize(tag_tidx);
    cols.end_tidx = F::from_usize(end_tidx);
    cols.discarded_sample
        .copy_from_slice(discarded_sample.as_basis_coefficients_slice());
    RowMajorMatrix::new(values, width)
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeStandardVaccCommitmentRootCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub tidx: T,
    pub root: [T; DIGEST_SIZE],
}

#[derive(ColumnsAir)]
#[columns_via(NativeStandardVaccCommitmentRootCols<u8>)]
pub struct NativeStandardVaccCommitmentRootAir {
    pub transcript_bus: TranscriptBus,
    pub transcript_role_bus: NativeVaccTranscriptRoleBus,
    pub merkle_root_bus: Option<NativeMerkleRootBus>,
    pub accumulator_root_bus: Option<NativeAccumulatorRootBus>,
    pub statement_root_bus: NativeStandardVaccRootBus,
    pub proof_kind: usize,
    pub transcript_role: usize,
    pub expected_tree_id: usize,
    pub expected_depth: usize,
}

impl BaseAirWithPublicValues<F> for NativeStandardVaccCommitmentRootAir {}
impl PartitionedBaseAir<F> for NativeStandardVaccCommitmentRootAir {}
impl BaseAir<F> for NativeStandardVaccCommitmentRootAir {
    fn width(&self) -> usize {
        NativeStandardVaccCommitmentRootCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeStandardVaccCommitmentRootAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("standard VACC root row");
        let local: &NativeStandardVaccCommitmentRootCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        for (limb, value) in local.root.into_iter().enumerate() {
            self.transcript_bus.observe_ext(
                builder,
                local.proof_idx,
                local.tidx + AB::Expr::from_usize(limb * D_EF),
                [value.into(), AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO],
                local.active,
            );
            self.transcript_role_bus.receive(
                builder,
                NativeVaccTranscriptRoleMessage {
                    proof_idx: local.proof_idx.into(),
                    role: AB::Expr::from_usize(self.transcript_role),
                    ordinal: AB::Expr::from_usize(limb),
                    tidx: local.tidx + AB::Expr::from_usize(limb * D_EF),
                    value: [value.into(), AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO],
                    is_ext: AB::Expr::ONE,
                    is_sample: AB::Expr::ZERO,
                },
                local.active,
            );
        }
        if let Some(bus) = self.merkle_root_bus {
            bus.receive(
                builder,
                NativeMerkleRootMessage {
                    proof_idx: local.proof_idx.into(),
                    tree_id: AB::Expr::from_usize(self.expected_tree_id),
                    depth: AB::Expr::from_usize(self.expected_depth),
                    digest: local.root.map(Into::into),
                },
                local.active,
            );
        }
        if let Some(bus) = self.accumulator_root_bus {
            bus.send(
                builder,
                NativeAccumulatorRootMessage {
                    proof_idx: local.proof_idx.into(),
                    state: AB::Expr::from_usize(self.proof_kind - 1),
                    digest: local.root.map(Into::into),
                },
                local.active,
            );
        }
        self.statement_root_bus.add_key_with_lookups(
            builder,
            NativeStandardVaccRootMessage {
                proof_idx: local.proof_idx.into(),
                kind: AB::Expr::from_usize(self.proof_kind),
                root: local.root.map(Into::into),
            },
            local.active,
        );
    }
}

pub fn generate_native_standard_vacc_commitment_root_trace(
    proof_idx: usize,
    tidx: usize,
    root: Digest,
) -> RowMajorMatrix<F> {
    let width = NativeStandardVaccCommitmentRootCols::<F>::width();
    let mut values = F::zero_vec(width);
    let cols: &mut NativeStandardVaccCommitmentRootCols<F> = values.as_mut_slice().borrow_mut();
    cols.active = F::ONE;
    cols.proof_idx = F::from_usize(proof_idx);
    cols.tidx = F::from_usize(tidx);
    cols.root = root;
    RowMajorMatrix::new(values, width)
}

/// Claim-value relation used by standard WARP. Fresh `alpha`, `mu`, and the
/// explicit beta tail are transcript observations; fresh beta's `tau` prefix
/// is sampled. Only fresh `eta` is forced to zero.
#[derive(ColumnsAir)]
#[columns_via(NativeClaimValueCols<u8>)]
pub struct NativeStandardClaimValueAir {
    pub claim_bus: NativeClaimValueBus,
    pub transcript_bus: TranscriptBus,
    pub transcript_role_bus: NativeVaccTranscriptRoleBus,
    pub layout_bus: NativeClaimLayoutBus,
    pub slot_bus: NativeInputSlotLayoutBus,
    pub fresh_claim_consumer_count: usize,
    pub prior_claim_consumer_count: usize,
    /// Setup-fixed section lengths used only to map source-local claim
    /// coordinates into the exact transcript's source-major flat order.
    pub fresh_alpha_len: usize,
    pub fresh_tau_len: usize,
    pub fresh_beta_tail_len: usize,
}

impl BaseAirWithPublicValues<F> for NativeStandardClaimValueAir {}
impl PartitionedBaseAir<F> for NativeStandardClaimValueAir {}
impl BaseAir<F> for NativeStandardClaimValueAir {
    fn width(&self) -> usize {
        NativeClaimValueCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeStandardClaimValueAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("standard claim value row");
        let local: &NativeClaimValueCols<AB::Var> = (*row).borrow();
        for flag in [
            local.active,
            local.is_fresh,
            local.is_prior,
            local.is_dummy,
            local.is_transcript,
            local.is_sample,
            local.is_beta_tail,
        ] {
            builder.assert_bool(flag);
        }
        builder
            .when(local.active * local.is_transcript)
            .assert_eq(local.proof_idx, local.transcript_id);
        for flag in local.section_kind {
            builder.assert_bool(flag);
        }
        builder
            .when(local.active)
            .assert_one(local.is_fresh + local.is_prior + local.is_dummy);
        builder.assert_eq(
            local.active,
            local
                .section_kind
                .into_iter()
                .map(AB::Expr::from)
                .sum::<AB::Expr>(),
        );
        let [_is_alpha, is_beta, is_mu, is_eta] = local.section_kind.map(AB::Expr::from);
        builder.assert_eq(
            local.section,
            is_beta.clone()
                + is_mu.clone() * AB::Expr::from_usize(CLAIM_SECTION_MU)
                + is_eta.clone() * AB::Expr::from_usize(CLAIM_SECTION_ETA),
        );
        self.slot_bus.lookup_key(
            builder,
            NativeInputSlotLayoutMessage {
                variant: local.variant.into(),
                source: local.source.into(),
                kind: [
                    local.is_fresh.into(),
                    local.is_prior.into(),
                    local.is_dummy.into(),
                ],
            },
            local.active,
        );
        for limb in local.value {
            builder
                .when(local.active * local.is_dummy)
                .assert_zero(limb);
            builder
                .when(local.active * local.is_fresh * is_eta.clone())
                .assert_zero(limb);
        }
        self.claim_bus.send(
            builder,
            NativeClaimValueMessage {
                proof_idx: local.proof_idx.into(),
                source: local.source.into(),
                section: local.section.into(),
                coordinate: local.coordinate.into(),
                value: local.value.map(Into::into),
            },
            local.active
                * (AB::Expr::ONE
                    + local.is_fresh
                        * AB::Expr::from_usize(self.fresh_claim_consumer_count.saturating_sub(1))
                    + local.is_prior
                        * AB::Expr::from_usize(self.prior_claim_consumer_count.saturating_sub(1))),
        );
        self.layout_bus.lookup_key(
            builder,
            NativeClaimLayoutMessage {
                section: local.section.into(),
                coordinate: local.coordinate.into(),
                is_beta_tail: local.is_beta_tail.into(),
                tail_coordinate: local.tail_coordinate.into(),
            },
            local.active,
        );
        self.transcript_bus.observe_ext(
            builder,
            local.transcript_id,
            local.tidx,
            local.value,
            local.active * local.is_transcript * (AB::Expr::ONE - local.is_sample),
        );
        self.transcript_bus.sample_ext(
            builder,
            local.transcript_id,
            local.tidx,
            local.value,
            local.active * local.is_transcript * local.is_sample,
        );
        let [is_alpha, is_beta, is_mu, is_eta] = local.section_kind.map(AB::Expr::from);
        let fresh_beta_prefix =
            local.is_fresh * is_beta.clone() * (AB::Expr::ONE - local.is_beta_tail);
        let fresh_beta_tail = local.is_fresh * is_beta.clone() * local.is_beta_tail;
        let role = local.is_fresh
            * (is_alpha.clone() * AB::Expr::from_usize(VACC_ROLE_FRESH_ALPHA)
                + is_mu.clone() * AB::Expr::from_usize(VACC_ROLE_FRESH_MU)
                + fresh_beta_tail.clone() * AB::Expr::from_usize(VACC_ROLE_FRESH_BETA_TAIL)
                + fresh_beta_prefix.clone() * AB::Expr::from_usize(VACC_ROLE_FRESH_TAU))
            + local.is_prior
                * (is_alpha.clone() * AB::Expr::from_usize(VACC_ROLE_PRIOR_ALPHA)
                    + is_beta * AB::Expr::from_usize(VACC_ROLE_PRIOR_BETA)
                    + is_mu.clone() * AB::Expr::from_usize(VACC_ROLE_PRIOR_MU)
                    + is_eta * AB::Expr::from_usize(VACC_ROLE_PRIOR_ETA));
        let source = AB::Expr::from(local.source);
        let fresh_ordinal = is_alpha.clone()
            * (source.clone() * AB::Expr::from_usize(self.fresh_alpha_len) + local.coordinate)
            + is_mu.clone() * source.clone()
            + fresh_beta_prefix.clone()
                * (source.clone() * AB::Expr::from_usize(self.fresh_tau_len) + local.coordinate)
            + fresh_beta_tail.clone()
                * (source * AB::Expr::from_usize(self.fresh_beta_tail_len) + local.tail_coordinate);
        // Prior-accumulator claims are a single source and preserve their
        // ordinary local section coordinates. Fresh exact-finite claims are
        // flattened source-major in the transcript cursor.
        let ordinal = local.is_fresh * fresh_ordinal + local.is_prior * local.coordinate;
        self.transcript_role_bus.receive(
            builder,
            NativeVaccTranscriptRoleMessage {
                proof_idx: local.proof_idx.into(),
                role,
                ordinal,
                tidx: local.tidx.into(),
                value: local.value.map(Into::into),
                is_ext: AB::Expr::ONE,
                is_sample: local.is_sample.into(),
            },
            local.active * local.is_transcript,
        );
    }
}

/// Fixed arity-two input slot table. `has_prior` is a verifier-key choice, so
/// no prover-selected state read can change a fresh/prior slot into padding.
#[derive(ColumnsAir)]
#[columns_via(NativeInputSlotLayoutCols<u8>)]
pub struct NativeStandardInputSlotLayoutAir {
    pub bus: NativeInputSlotLayoutBus,
    pub has_prior: bool,
}

impl BaseAirWithPublicValues<F> for NativeStandardInputSlotLayoutAir {}
impl PartitionedBaseAir<F> for NativeStandardInputSlotLayoutAir {}
impl BaseAir<F> for NativeStandardInputSlotLayoutAir {
    fn width(&self) -> usize {
        NativeInputSlotLayoutCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeStandardInputSlotLayoutAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("standard input slot row");
        let local: &NativeInputSlotLayoutCols<AB::Var> = (*row).borrow();
        for flag in [local.active, local.is_first, local.is_last] {
            builder.assert_bool(flag);
        }
        for flag in local.kind {
            builder.assert_bool(flag);
        }
        builder
            .when(local.active)
            .assert_one(local.kind.into_iter().map(AB::Expr::from).sum::<AB::Expr>());
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when_first_row().assert_zero(local.source);
        builder.when(local.active).assert_eq(
            local.variant,
            AB::Expr::from_usize(1 + usize::from(self.has_prior) * 3),
        );
        let is_source_zero = local.is_first;
        builder
            .when(local.active)
            .assert_eq(local.kind[0], is_source_zero);
        builder.when(local.active).assert_eq(
            local.kind[1],
            (AB::Expr::ONE - is_source_zero) * AB::Expr::from_bool(self.has_prior),
        );
        builder.when(local.active).assert_eq(
            local.kind[2],
            (AB::Expr::ONE - is_source_zero) * AB::Expr::from_bool(!self.has_prior),
        );
        builder
            .when(local.active * local.is_last)
            .assert_eq(local.source, AB::Expr::ONE);
        let next_row = main.row_slice(1).expect("standard input slot next row");
        let next: &NativeInputSlotLayoutCols<AB::Var> = (*next_row).borrow();
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder
            .when_transition()
            .when(local.active - next.active)
            .assert_one(local.is_last);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);
        let mut transition = builder.when_transition();
        let mut active = transition.when(next.active);
        active.assert_eq(next.source, local.source + AB::F::ONE);
        active.assert_eq(next.variant, local.variant);
        active.assert_zero(next.is_first);
        self.bus.add_key_with_lookups(
            builder,
            NativeInputSlotLayoutMessage {
                variant: local.variant.into(),
                source: local.source.into(),
                kind: local.kind.map(Into::into),
            },
            local.lookup_count,
        );
    }
}

/// Key-fixed input-slot table for one exact-finite invocation. Unlike the
/// legacy arity-two adapter, this AIR supports every power-of-two arity up to
/// 64 and fixes the complete fresh/prior partition in the verifier key.
#[derive(ColumnsAir)]
#[columns_via(NativeInputSlotLayoutCols<u8>)]
pub struct NativeExactFiniteInputSlotLayoutAir {
    pub bus: NativeInputSlotLayoutBus,
    pub input_arity: usize,
    pub fresh_count: usize,
    pub prior_count: usize,
    /// Ordinary claim/projection consumers per occupied slot.
    pub ordinary_lookup_count: usize,
    /// Additional setup-fixed consumers attached only to fresh slots.
    pub extra_fresh_lookup_count: usize,
}

impl NativeExactFiniteInputSlotLayoutAir {
    pub fn validate(&self) -> Result<(), &'static str> {
        if self.input_arity < 2
            || self.input_arity > STANDARD_EXACT_FINITE_VACC_MAX_INPUT_ARITY
            || !self.input_arity.is_power_of_two()
            || self.prior_count > 1
            || self.fresh_count == 0
            || self.fresh_count + self.prior_count != self.input_arity
            || self.ordinary_lookup_count == 0
            || self.ordinary_lookup_count > u16::MAX as usize
            || self.extra_fresh_lookup_count > STANDARD_EXACT_FINITE_VACC_MAX_LOOKUP_COUNT
        {
            return Err("exact-finite input slot layout");
        }
        Ok(())
    }

    #[must_use]
    pub const fn variant(&self) -> usize {
        self.fresh_count + self.prior_count * (self.input_arity + 1)
    }
}

impl BaseAirWithPublicValues<F> for NativeExactFiniteInputSlotLayoutAir {}
impl PartitionedBaseAir<F> for NativeExactFiniteInputSlotLayoutAir {}
impl BaseAir<F> for NativeExactFiniteInputSlotLayoutAir {
    fn width(&self) -> usize {
        NativeInputSlotLayoutCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeExactFiniteInputSlotLayoutAir {
    fn eval(&self, builder: &mut AB) {
        assert!(self.validate().is_ok(), "invalid exact-finite input slots");
        let main = builder.main();
        let row = main.row_slice(0).expect("exact-finite input slot row");
        let local: &NativeInputSlotLayoutCols<AB::Var> = (*row).borrow();
        for flag in [
            local.active,
            local.is_first,
            local.is_last,
            local.prior_present,
            local.prior_remaining,
            local.fresh_is_zero,
        ] {
            builder.assert_bool(flag);
        }
        for flag in local.kind {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder
            .when(local.active)
            .assert_eq(local.variant, AB::Expr::from_usize(self.variant()));
        builder
            .when(local.active)
            .assert_eq(local.fresh_count, AB::Expr::from_usize(self.fresh_count));
        builder
            .when(local.active)
            .assert_eq(local.prior_present, AB::Expr::from_usize(self.prior_count));
        builder
            .when(local.active)
            .assert_one(local.kind.into_iter().map(AB::Expr::from).sum::<AB::Expr>());
        builder
            .when(local.active * local.is_first)
            .assert_zero(local.source);
        builder.when(local.active * local.is_first).assert_eq(
            local.fresh_remaining,
            AB::Expr::from_usize(self.fresh_count),
        );
        builder.when(local.active * local.is_first).assert_eq(
            local.prior_remaining,
            AB::Expr::from_usize(self.prior_count),
        );
        builder
            .when(local.active * local.fresh_is_zero)
            .assert_zero(local.fresh_remaining);
        builder
            .when(local.active * (AB::Expr::ONE - local.fresh_is_zero))
            .assert_one(local.fresh_remaining * local.fresh_inverse);
        builder
            .when(local.active)
            .assert_eq(local.kind[0], AB::Expr::ONE - local.fresh_is_zero);
        builder
            .when(local.active)
            .assert_eq(local.kind[1], local.fresh_is_zero * local.prior_remaining);
        builder.when(local.active).assert_eq(
            local.kind[2],
            local.fresh_is_zero * (AB::Expr::ONE - local.prior_remaining),
        );
        builder.when(local.active).assert_eq(
            local.lookup_count,
            AB::Expr::from_usize(self.ordinary_lookup_count)
                + local.kind[0] * AB::Expr::from_usize(self.extra_fresh_lookup_count),
        );
        // Exact schedules fill every slot; padding is never valid.
        builder.when(local.active).assert_zero(local.kind[2]);
        builder
            .when(local.active * local.is_last)
            .assert_eq(local.source, AB::Expr::from_usize(self.input_arity - 1));

        let next_row = main.row_slice(1).expect("exact-finite input slot next row");
        let next: &NativeInputSlotLayoutCols<AB::Var> = (*next_row).borrow();
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        let mut transition = builder.when_transition();
        let mut same = transition.when(next.active);
        same.assert_eq(next.source, local.source + AB::F::ONE);
        same.assert_eq(next.variant, local.variant);
        same.assert_eq(next.fresh_count, local.fresh_count);
        same.assert_eq(next.prior_present, local.prior_present);
        same.assert_eq(next.fresh_remaining, local.fresh_remaining - local.kind[0]);
        same.assert_eq(next.prior_remaining, local.prior_remaining - local.kind[1]);
        builder
            .when_transition()
            .when(local.active - next.active)
            .assert_one(local.is_last);
        self.bus.add_key_with_lookups(
            builder,
            NativeInputSlotLayoutMessage {
                variant: local.variant.into(),
                source: local.source.into(),
                kind: local.kind.map(Into::into),
            },
            local.lookup_count,
        );
    }
}

pub fn generate_native_exact_finite_input_slot_layout_trace(
    input_arity: usize,
    fresh_count: usize,
    prior_count: usize,
    lookup_counts: &[u32],
) -> Option<RowMajorMatrix<F>> {
    if prior_count > 1 {
        return None;
    }
    crate::native_warp::generate_native_input_slot_layout_trace(
        input_arity,
        fresh_count,
        prior_count == 1,
        lookup_counts,
    )
}

/// Generic flat-codeword opening projection for either the fresh source or
/// the prior accumulator. Unlike the legacy direct projection, it authenticates
/// the exact standard WARP codeword root and carries no stacked PCS layout.
#[derive(ColumnsAir)]
#[columns_via(NativeAccumulatorProjectionCols<u8>)]
pub struct NativeStandardCodewordProjectionAir {
    pub shift_index_bus: NativeShiftIndexBus,
    pub leaf_value_bus: NativeLeafValueBus,
    pub authenticated_bus: NativeAuthenticatedShiftBus,
    pub slot_bus: NativeInputSlotLayoutBus,
    pub oracle_height: usize,
    pub query_count: usize,
    pub row_tree_id_offset: usize,
    pub source: usize,
    pub variant: usize,
    pub is_fresh: bool,
}

impl BaseAirWithPublicValues<F> for NativeStandardCodewordProjectionAir {}
impl PartitionedBaseAir<F> for NativeStandardCodewordProjectionAir {}
impl BaseAir<F> for NativeStandardCodewordProjectionAir {
    fn width(&self) -> usize {
        NativeAccumulatorProjectionCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeStandardCodewordProjectionAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("standard codeword projection row");
        let local: &NativeAccumulatorProjectionCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
        builder
            .when(local.active)
            .assert_eq(local.source, AB::Expr::from_usize(self.source));
        builder
            .when(local.active)
            .assert_eq(local.variant, AB::Expr::from_usize(self.variant));
        self.shift_index_bus.lookup_key(
            builder,
            NativeShiftIndexMessage {
                proof_idx: local.proof_idx.into(),
                shift: local.shift.into(),
                index: local.flat_index.into(),
            },
            local.active,
        );
        self.slot_bus.lookup_key(
            builder,
            NativeInputSlotLayoutMessage {
                variant: local.variant.into(),
                source: local.source.into(),
                kind: if self.is_fresh {
                    [AB::Expr::ONE, AB::Expr::ZERO, AB::Expr::ZERO]
                } else {
                    [AB::Expr::ZERO, AB::Expr::ONE, AB::Expr::ZERO]
                },
            },
            local.active,
        );
        builder.when(local.active).assert_eq(
            local.flat_index,
            local.column * AB::Expr::from_usize(self.oracle_height)
                + local.query_index
                + local.row_offset * AB::Expr::from_usize(self.query_count),
        );
        builder.when(local.active).assert_eq(
            local.leaf_position,
            local.column * AB::Expr::from_usize(D_EF),
        );
        builder.when(local.active).assert_eq(
            local.inner_tree_id,
            AB::Expr::from_usize(self.row_tree_id_offset) + local.query_index,
        );
        for limb in 0..D_EF {
            self.leaf_value_bus.lookup_key(
                builder,
                NativeLeafValueMessage {
                    proof_idx: local.proof_idx.into(),
                    tree_id: local.inner_tree_id.into(),
                    index: local.row_offset.into(),
                    position: local.leaf_position + AB::Expr::from_usize(limb),
                    value: local.value[limb].into(),
                },
                local.active,
            );
        }
        self.authenticated_bus.send(
            builder,
            NativeAuthenticatedShiftMessage {
                proof_idx: local.proof_idx.into(),
                shift: local.shift.into(),
                source: local.source.into(),
                value: local.value.map(Into::into),
            },
            local.active,
        );
    }
}

/// Reuse the battle-tested Merkle projection generator, then canonicalize the
/// slot metadata for a standard fresh or prior input.
#[allow(clippy::too_many_arguments)]
pub fn generate_native_standard_codeword_projection_trace<H>(
    hasher: &H,
    proof_idx: usize,
    verification: &MerkleBatchOpeningVerification<EF, Digest>,
    log_codeword_len: usize,
    rows_per_query: usize,
    row_tree_id_offset: usize,
    outer_tree_id: u32,
    source: usize,
    variant: usize,
    required_height: Option<usize>,
) -> Option<NativeAccumulatorProjectionTrace>
where
    H: openvm_stark_backend::hasher::MerkleHasher<F = F, Digest = Digest>,
{
    generate_native_exact_finite_codeword_projection_trace(
        hasher,
        proof_idx,
        verification,
        log_codeword_len,
        rows_per_query,
        row_tree_id_offset,
        outer_tree_id,
        source,
        variant,
        STANDARD_DIRECT_VACC_INPUT_ARITY,
        required_height,
    )
}

/// Generic prior/output codeword projection for an exact-finite call. This is
/// the same authenticated Merkle projection used by the arity-two recursive
/// lane, with only the slot-table arity made verifier-key explicit.
#[allow(clippy::too_many_arguments)]
pub fn generate_native_exact_finite_codeword_projection_trace<H>(
    hasher: &H,
    proof_idx: usize,
    verification: &MerkleBatchOpeningVerification<EF, Digest>,
    log_codeword_len: usize,
    rows_per_query: usize,
    row_tree_id_offset: usize,
    outer_tree_id: u32,
    source: usize,
    variant: usize,
    input_arity: usize,
    required_height: Option<usize>,
) -> Option<NativeAccumulatorProjectionTrace>
where
    H: openvm_stark_backend::hasher::MerkleHasher<F = F, Digest = Digest>,
{
    generate_native_exact_finite_codeword_projection_trace_checked(
        hasher,
        proof_idx,
        verification,
        log_codeword_len,
        rows_per_query,
        row_tree_id_offset,
        outer_tree_id,
        source,
        variant,
        input_arity,
        required_height,
    )
    .ok()
}

#[allow(clippy::too_many_arguments)]
pub fn generate_native_exact_finite_codeword_projection_trace_checked<H>(
    hasher: &H,
    proof_idx: usize,
    verification: &MerkleBatchOpeningVerification<EF, Digest>,
    log_codeword_len: usize,
    rows_per_query: usize,
    row_tree_id_offset: usize,
    outer_tree_id: u32,
    source: usize,
    variant: usize,
    input_arity: usize,
    required_height: Option<usize>,
) -> Result<NativeAccumulatorProjectionTrace, &'static str>
where
    H: openvm_stark_backend::hasher::MerkleHasher<F = F, Digest = Digest>,
{
    if input_arity < 2
        || input_arity > STANDARD_EXACT_FINITE_VACC_MAX_INPUT_ARITY
        || !input_arity.is_power_of_two()
        || source >= input_arity
    {
        return Err("exact-finite projection input layout");
    }
    let mut trace = crate::native_warp::generate_native_accumulator_projection_trace_checked(
        hasher,
        proof_idx,
        verification,
        log_codeword_len,
        rows_per_query,
        row_tree_id_offset,
        outer_tree_id,
        source,
        input_arity,
        required_height,
    )?;
    let width = trace.matrix.width();
    for row in trace.matrix.values.chunks_exact_mut(width) {
        let cols: &mut NativeAccumulatorProjectionCols<F> = row.borrow_mut();
        if cols.active == F::ONE {
            cols.source = F::from_usize(source);
            cols.variant = F::from_usize(variant);
        }
    }
    Ok(trace)
}

#[must_use]
pub fn standard_vacc_bus(base: BusIndex) -> NativeStandardVaccProtocolBus {
    NativeStandardVaccProtocolBus::new(base)
}

const _: () = assert!(D_EF == 4);

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;

    use openvm_stark_backend::{
        air_builders::{
            debug::{check_constraints, check_logup},
            symbolic::{get_symbolic_builder, SymbolicRapBuilder},
        },
        interaction::SymbolicInteraction,
        keygen::types::TraceWidth,
        BaseAirWithPublicValues, PartitionedBaseAir,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config as NativeSC;

    use super::*;
    use crate::bus::TranscriptBusMessage;

    fn digest(seed: u32) -> Digest {
        core::array::from_fn(|limb| F::from_u32(seed + limb as u32))
    }

    fn exact_profile(
        input_arity: usize,
        prior_count: usize,
    ) -> NativeExactFiniteVaccTranscriptProfile {
        let calls = if prior_count == 0 {
            [
                NativeExactFiniteVaccCallProfile {
                    active: true,
                    input_arity,
                    fresh_count: input_arity,
                    prior_count: 0,
                },
                NativeExactFiniteVaccCallProfile::inactive(),
                NativeExactFiniteVaccCallProfile::inactive(),
            ]
        } else {
            [
                NativeExactFiniteVaccCallProfile {
                    active: true,
                    input_arity: 2,
                    fresh_count: 2,
                    prior_count: 0,
                },
                NativeExactFiniteVaccCallProfile {
                    active: true,
                    input_arity,
                    fresh_count: input_arity - 1,
                    prior_count: 1,
                },
                NativeExactFiniteVaccCallProfile::inactive(),
            ]
        };
        NativeExactFiniteVaccTranscriptProfile {
            relation_description: b"exact-finite-test-relation".to_vec(),
            external_index_binding: vec![EF::from_u32(17), EF::from_u32(19)],
            relation_digest: digest(10),
            index_digest: digest(20),
            setup_digest: digest(30),
            schedule_digest: digest(40),
            shape: NativeStandardVaccShapeProfile {
                num_ood: 1,
                num_shift_queries: 1,
                batching_arity: 4,
                log_message_len: 3,
                log_codeword_len: 4,
                initial_folding_factor: 1,
                log_constraints: 1,
                beta_len: 3,
                max_degree: 2,
                rows_per_query: 2,
            },
            max_input_arity: 64,
            calls,
        }
    }

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct ExactPrefixOracleCols<T> {
        active: T,
        proof_idx: T,
        tidx: T,
        value: T,
    }

    struct ExactPrefixOracleAir {
        transcript_bus: TranscriptBus,
    }

    impl BaseAir<F> for ExactPrefixOracleAir {
        fn width(&self) -> usize {
            ExactPrefixOracleCols::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for ExactPrefixOracleAir {}
    impl PartitionedBaseAir<F> for ExactPrefixOracleAir {}
    impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for ExactPrefixOracleAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("exact prefix oracle row");
            let local: &ExactPrefixOracleCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.active);
            self.transcript_bus.send(
                builder,
                local.proof_idx,
                TranscriptBusMessage {
                    tidx: local.tidx.into(),
                    value: local.value.into(),
                    is_sample: AB::Expr::ZERO,
                },
                local.active,
            );
        }
    }

    #[repr(C)]
    #[derive(AlignedBorrow)]
    struct ExactPhaseCursorOracleCols<T> {
        active: T,
        proof_idx: T,
        tidx: T,
    }

    struct ExactPhaseCursorOracleAir {
        phase_cursor_bus: NativeVaccPhaseCursorBus,
    }

    impl BaseAir<F> for ExactPhaseCursorOracleAir {
        fn width(&self) -> usize {
            ExactPhaseCursorOracleCols::<F>::width()
        }
    }
    impl BaseAirWithPublicValues<F> for ExactPhaseCursorOracleAir {}
    impl PartitionedBaseAir<F> for ExactPhaseCursorOracleAir {}
    impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for ExactPhaseCursorOracleAir {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let row = main.row_slice(0).expect("exact phase-cursor oracle row");
            let local: &ExactPhaseCursorOracleCols<AB::Var> = (*row).borrow();
            builder.assert_bool(local.active);
            self.phase_cursor_bus.receive(
                builder,
                NativeVaccPhaseCursorMessage {
                    proof_idx: local.proof_idx.into(),
                    boundary: AB::Expr::ONE,
                    tidx: local.tidx.into(),
                },
                AB::Expr::from(local.active) * AB::Expr::from_usize(2),
            );
        }
    }

    fn exact_phase_cursor_oracle_trace(proof_idx: usize, tidx: usize) -> RowMajorMatrix<F> {
        let width = ExactPhaseCursorOracleCols::<F>::width();
        let mut values = F::zero_vec(width);
        let local: &mut ExactPhaseCursorOracleCols<F> = values.as_mut_slice().borrow_mut();
        local.active = F::ONE;
        local.proof_idx = F::from_usize(proof_idx);
        local.tidx = F::from_usize(tidx);
        RowMajorMatrix::new(values, width)
    }

    fn exact_prefix_oracle_trace(
        proof_idx: usize,
        start_tidx: usize,
        elements: &[EF],
    ) -> RowMajorMatrix<F> {
        let width = ExactPrefixOracleCols::<F>::width();
        let operation_count = elements.len() * D_EF;
        let mut values = F::zero_vec(operation_count.next_power_of_two() * width);
        for (ordinal, value) in elements
            .iter()
            .flat_map(|value| value.as_basis_coefficients_slice())
            .copied()
            .enumerate()
        {
            let cols: &mut ExactPrefixOracleCols<F> =
                values[ordinal * width..(ordinal + 1) * width].borrow_mut();
            cols.active = F::ONE;
            cols.proof_idx = F::from_usize(proof_idx);
            cols.tidx = F::from_usize(start_tidx + ordinal);
            cols.value = value;
        }
        RowMajorMatrix::new(values, width)
    }

    fn symbolic_interactions<R>(air: &R) -> Vec<SymbolicInteraction<F>>
    where
        R: Air<SymbolicRapBuilder<F>>
            + BaseAir<F>
            + BaseAirWithPublicValues<F>
            + PartitionedBaseAir<F>,
    {
        get_symbolic_builder(
            air,
            &TraceWidth {
                preprocessed: None,
                cached_mains: air.cached_main_widths(),
                common_main: air.common_main_width(),
            },
        )
        .constraints()
        .interactions
    }

    fn check_prefix_transcript_interactions<R>(
        air: &R,
        trace: &RowMajorMatrix<F>,
        oracle: &ExactPrefixOracleAir,
        oracle_trace: &RowMajorMatrix<F>,
        transcript_bus: TranscriptBus,
    ) where
        R: Air<SymbolicRapBuilder<F>>
            + BaseAir<F>
            + BaseAirWithPublicValues<F>
            + PartitionedBaseAir<F>,
    {
        let interactions = vec![
            symbolic_interactions(air)
                .into_iter()
                .filter(|interaction| interaction.bus_index == transcript_bus.index())
                .collect(),
            symbolic_interactions(oracle),
        ];
        let matrices = vec![vec![trace.as_view()], vec![oracle_trace.as_view()]];
        check_logup(
            &["exact-prefix".to_string(), "cpu-oracle".to_string()],
            &interactions,
            &[None, None],
            &matrices,
            &[Vec::new(), Vec::new()],
        );
    }

    #[test]
    fn exact_finite_profiles_cover_every_supported_arity() {
        for input_arity in [2, 4, 8, 16, 32, 64] {
            for prior_count in [0, 1] {
                let profile = exact_profile(input_arity, prior_count);
                profile.validate().unwrap();
                let prefix =
                    native_exact_finite_vacc_call_prefix_elements(&profile, prior_count).unwrap();
                assert!(!prefix.is_empty());
                let algebra = profile.call_algebra_profile(prior_count).unwrap();
                assert_eq!(algebra.input_arity, input_arity);
                assert_eq!(algebra.twin_rounds, input_arity.ilog2() as usize);
                assert_eq!(algebra.twin_last_round + 1, algebra.twin_rounds);

                let slot_air = NativeExactFiniteInputSlotLayoutAir {
                    bus: NativeInputSlotLayoutBus::new(1_300),
                    input_arity,
                    fresh_count: input_arity - prior_count,
                    prior_count,
                    ordinary_lookup_count: 1,
                    extra_fresh_lookup_count: 0,
                };
                let trace = generate_native_exact_finite_input_slot_layout_trace(
                    input_arity,
                    input_arity - prior_count,
                    prior_count,
                    &vec![1; input_arity],
                )
                .unwrap();
                check_constraints::<_, NativeSC>(
                    &slot_air,
                    "NativeExactFiniteInputSlotLayoutAir",
                    &None,
                    &[trace.as_view()],
                    &[],
                );
            }
        }
    }

    #[test]
    fn exact_finite_cpu_transcript_interactions_match_at_arity_2_8_64() {
        for (case, input_arity) in [2, 8, 64].into_iter().enumerate() {
            let profile = exact_profile(input_arity, 0);
            let transcript_bus = TranscriptBus::new(1_400 + case as u16 * 4);
            let schedule_air = NativeExactFiniteVaccSchedulePrefixAir {
                transcript_bus,
                schedule_bus: NativeExactFiniteVaccScheduleBus::new(1_401 + case as u16 * 4),
                profile: profile.clone(),
                lookup_count: 1,
            };
            let schedule_trace = generate_native_exact_finite_vacc_schedule_prefix_trace(0, 0);
            let schedule_elements =
                native_exact_finite_vacc_schedule_prefix_elements(&profile).unwrap();
            let schedule_oracle = ExactPrefixOracleAir { transcript_bus };
            let schedule_oracle_trace = exact_prefix_oracle_trace(0, 0, &schedule_elements);
            check_constraints::<_, NativeSC>(
                &schedule_air,
                "NativeExactFiniteVaccSchedulePrefixAir",
                &None,
                &[schedule_trace.as_view()],
                &[],
            );
            check_prefix_transcript_interactions(
                &schedule_air,
                &schedule_trace,
                &schedule_oracle,
                &schedule_oracle_trace,
                transcript_bus,
            );

            let start_tidx = schedule_elements.len() * D_EF;
            let call_air = NativeExactFiniteVaccCallPrefixAir {
                transcript_bus,
                phase_cursor_bus: NativeVaccPhaseCursorBus::new(1_402 + case as u16 * 4),
                protocol_bus: NativeExactFiniteVaccCallProtocolBus::new(1_403 + case as u16 * 4),
                legacy_protocol_bus: NativeStandardVaccProtocolBus::new(1_404 + case as u16 * 4),
                profile: profile.clone(),
                call_index: 0,
                lookup_count: 1,
            };
            let call_trace = generate_native_exact_finite_vacc_call_prefix_trace(0, start_tidx);
            let call_elements = native_exact_finite_vacc_call_prefix_elements(&profile, 0).unwrap();
            let call_oracle_trace = exact_prefix_oracle_trace(0, start_tidx, &call_elements);
            check_constraints::<_, NativeSC>(
                &call_air,
                "NativeExactFiniteVaccCallPrefixAir",
                &None,
                &[call_trace.as_view()],
                &[],
            );
            check_prefix_transcript_interactions(
                &call_air,
                &call_trace,
                &schedule_oracle,
                &call_oracle_trace,
                transcript_bus,
            );
        }
    }

    #[test]
    fn exact_finite_call_prefix_exposes_only_the_doubly_consumed_post_prefix_cursor() {
        for (case, (profile, call_index)) in [(exact_profile(64, 0), 0), (exact_profile(8, 1), 1)]
            .into_iter()
            .enumerate()
        {
            let start_tidx = 700 + 100 * case;
            let proof_idx = call_index;
            let phase_cursor_bus = NativeVaccPhaseCursorBus::new(1_700 + case as u16 * 4);
            let air = NativeExactFiniteVaccCallPrefixAir {
                transcript_bus: TranscriptBus::new(1_701 + case as u16 * 4),
                phase_cursor_bus,
                protocol_bus: NativeExactFiniteVaccCallProtocolBus::new(1_702 + case as u16 * 4),
                legacy_protocol_bus: NativeStandardVaccProtocolBus::new(1_703 + case as u16 * 4),
                profile: profile.clone(),
                call_index,
                lookup_count: 1,
            };
            let trace = generate_native_exact_finite_vacc_call_prefix_trace(proof_idx, start_tidx);
            let post_prefix_tidx = start_tidx
                + native_exact_finite_vacc_call_prefix_elements(&profile, call_index)
                    .unwrap()
                    .len()
                    * D_EF;
            let oracle = ExactPhaseCursorOracleAir { phase_cursor_bus };
            let oracle_trace = exact_phase_cursor_oracle_trace(proof_idx, post_prefix_tidx);
            check_constraints::<_, NativeSC>(
                &air,
                "NativeExactFiniteVaccCallPrefixAir phase cursor",
                &None,
                &[trace.as_view()],
                &[],
            );
            let check = |oracle_trace: &RowMajorMatrix<F>| {
                check_logup(
                    &[
                        "exact-call-prefix".to_string(),
                        "phase-cursor-oracle".to_string(),
                    ],
                    &[
                        symbolic_interactions(&air)
                            .into_iter()
                            .filter(|interaction| interaction.bus_index == phase_cursor_bus.index())
                            .collect(),
                        symbolic_interactions(&oracle),
                    ],
                    &[None, None],
                    &[vec![trace.as_view()], vec![oracle_trace.as_view()]],
                    &[Vec::new(), Vec::new()],
                );
            };
            check(&oracle_trace);

            let mut wrong_tidx = oracle_trace.clone();
            wrong_tidx.values[2] += F::ONE;
            assert!(
                std::panic::catch_unwind(AssertUnwindSafe(|| check(&wrong_tidx))).is_err(),
                "shifted exact post-prefix cursor accepted"
            );
            let mut wrong_proof = oracle_trace.clone();
            wrong_proof.values[1] += F::ONE;
            assert!(
                std::panic::catch_unwind(AssertUnwindSafe(|| check(&wrong_proof))).is_err(),
                "relabelled exact call proof index accepted"
            );
        }
    }

    #[test]
    fn exact_finite_transcript_rejects_arity_relabel_and_index_mutation() {
        let arity_two = exact_profile(2, 0);
        let arity_64 = exact_profile(64, 0);
        assert_ne!(
            native_exact_finite_vacc_schedule_prefix_elements(&arity_two).unwrap(),
            native_exact_finite_vacc_schedule_prefix_elements(&arity_64).unwrap(),
        );
        assert_ne!(
            native_exact_finite_vacc_call_prefix_elements(&arity_two, 0).unwrap(),
            native_exact_finite_vacc_call_prefix_elements(&arity_64, 0).unwrap(),
        );

        let mut changed_index = arity_64.clone();
        changed_index.external_index_binding[0] += EF::ONE;
        assert_ne!(
            native_exact_finite_vacc_call_prefix_elements(&arity_64, 0).unwrap(),
            native_exact_finite_vacc_call_prefix_elements(&changed_index, 0).unwrap(),
        );
    }

    #[test]
    fn exact_finite_slot_trace_mutation_is_rejected_without_panic_escape() {
        let air = NativeExactFiniteInputSlotLayoutAir {
            bus: NativeInputSlotLayoutBus::new(1_301),
            input_arity: 64,
            fresh_count: 64,
            prior_count: 0,
            ordinary_lookup_count: 9,
            extra_fresh_lookup_count: 1,
        };
        let mut trace =
            generate_native_exact_finite_input_slot_layout_trace(64, 64, 0, &[10; 64]).unwrap();
        check_constraints::<_, NativeSC>(
            &air,
            "NativeExactFiniteInputSlotLayoutAir",
            &None,
            &[trace.as_view()],
            &[],
        );
        let width = trace.width();
        let first: &mut NativeInputSlotLayoutCols<F> = trace.values[..width].borrow_mut();
        first.lookup_count -= F::ONE;
        let rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
            check_constraints::<_, NativeSC>(
                &air,
                "NativeExactFiniteInputSlotLayoutAir",
                &None,
                &[trace.as_view()],
                &[],
            );
        }));
        assert!(rejected.is_err());
    }

    #[test]
    fn fixed_hleaf_route_separates_slot_global_and_warp_step() {
        let genesis = NativeFixedHLeafVaccRouteV4::derive(0, 0, 0).unwrap();
        assert_eq!(genesis.node_index, 0);
        assert!(genesis.has_prior);
        assert_eq!(genesis.warp_step, FIXED_HLEAF_CONTINUATION_WARP_STEP_V4);

        let later_slot_zero = NativeFixedHLeafVaccRouteV4::derive(0, 12, 0).unwrap();
        assert_eq!(later_slot_zero.local_slot, 0);
        assert_eq!(later_slot_zero.node_index, 3);
        assert!(later_slot_zero.has_prior);
        assert_eq!(
            later_slot_zero.warp_step,
            FIXED_HLEAF_CONTINUATION_WARP_STEP_V4
        );

        for slot in 1..FIXED_HLEAF_VACC_CAPACITY_V4 {
            let route = NativeFixedHLeafVaccRouteV4::derive(slot, slot as u32, 0).unwrap();
            assert!(route.has_prior);
            assert_eq!(route.warp_step, FIXED_HLEAF_CONTINUATION_WARP_STEP_V4);
        }
    }

    #[test]
    fn fixed_hleaf_route_rejects_wrong_global_or_prior_mode() {
        assert!(NativeFixedHLeafVaccRouteV4::derive(0, 1, 0).is_err());
        assert!(NativeFixedHLeafVaccRouteV4::derive(4, 4, 0).is_err());

        let mut route = NativeFixedHLeafVaccRouteV4::derive(0, 4, 0).unwrap();
        route.has_prior = false;
        route.warp_step = 0;
        assert!(route.validate().is_err());

        let mut route = NativeFixedHLeafVaccRouteV4::derive(0, 4, 0).unwrap();
        route.node_index = 0;
        assert!(route.validate().is_err());
    }
}

//! Generic verifier AIR boundaries for one setup-fixed WARP VACC step.
//!
//! These gadgets intentionally know nothing about an application relation.
//! They certify the v5 VACC transcript, bind the ordinary fresh/prior/output
//! codeword commitments, and expose the verifier-derived statement to a
//! caller-owned adapter.  In particular, no PCS opening is evaluated as a
//! PESAT relation here: application-relation evaluation belongs to terminal
//! Decide.

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder, warp_accum::WarpParams, BaseAirWithPublicValues,
    PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, D_EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    define_typed_lookup_bus,
    native_warp::{
        NativeAccumulatorRootBus, NativeAccumulatorRootMessage, NativeClaimLayoutBus,
        NativeClaimLayoutMessage, NativeClaimValueBus, NativeClaimValueCols,
        NativeClaimValueMessage, NativeCoefficientSumcheckAir, NativeEqResultBus,
        NativeFoldedClaimBus, NativeInputSlotLayoutBus, NativeInputSlotLayoutMessage,
        NativeMerkleRootBus, NativeMerkleRootMessage, NativeSumcheckChallengeBus,
        NativeSumcheckInitialBus, NativeSumcheckRoundBus, NativeTwinFinalAir, NativeTwinFoldAir,
        NativeTwinOmegaBus, NativeTwinScalarBus, NativeTwinSigmaAir, NativeVaccTranscriptRoleBus,
        NativeVaccTranscriptRoleMessage, CLAIM_SECTION_ETA, CLAIM_SECTION_MU,
    },
};

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

/// Key-derived algebra dimensions for one WARP VACC invocation.
///
/// Keeping these dimensions in one value prevents the recursive verifier
/// from mixing sumcheck AIRs configured for different input arities or code
/// parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeVaccCallAlgebraProfile {
    pub input_arity: usize,
    pub twin_rounds: usize,
    pub twin_last_round: usize,
    pub twin_degree: usize,
    pub batching_rounds: usize,
    pub batching_degree: usize,
}

impl NativeVaccCallAlgebraProfile {
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

/// Fixed verifier-key dimensions of one homogeneous WARP relation.
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

/// Relation-independent verifier-key dimensions for one fixed-arity VACC
/// transition. The VACC verifier algebra never evaluates the application
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

    /// Canonical relation digest paired with the private canonical
    /// description used to construct this complete profile.
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

    pub fn validate(&self) -> Result<(), &'static str> {
        self.shape_profile().validate()?;
        if self.relation_description.is_empty() {
            return Err("standard direct VACC profile");
        }
        Ok(())
    }
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
        // ordinary local section coordinates. Fresh batched claims are
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

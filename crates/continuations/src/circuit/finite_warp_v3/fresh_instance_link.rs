//! Minimal projection from exact-WARP fresh claims to ordered-manifest authority.
//!
//! This component does not verify a SWIRL proof and does not decide PESAT.  It
//! projects values already exposed by the ordinary exact WARP/VACC verifier:
//!
//! 1. [`FreshInstanceCallLinkAir`] consumes the key-fixed finite schedule, call protocol, fresh
//!    input slots, fresh stacked root, and the manifest's authenticated prefix statement.  It
//!    publishes one private source context table per call.
//! 2. [`FreshInstanceProjectionAir`] consumes *every* proof-qualified fresh
//!    [`NativeClaimValueMessage`].  Its key-fixed claim layout identifies the beta tail that is the
//!    complete relation's explicit assignment, checks the canonical `1, alpha, beta, beta-powers,
//!    base-public-values` layout, hashes every EF4 basis coordinate with the native source-instance
//!    Poseidon transcript, projects the setup-fixed aggregate `VmPvs`, and emits
//!    [`OrderedManifestSourceAuthorityBus`].
//!
//! The emitted receipt is meaningful only in a proof that also verifies the
//! exact WARP certificate and terminal Decide.  In particular, this file does
//! not turn a PCS opening into PESAT and does not accept a host validity bit.
//!
//! ## Consumer multiplicities
//!
//! [`FreshInstanceLinkConsumerRequirements`] is part of the integration API.
//! The ordinary fresh-claim producer has a uniform multiplicity over alpha,
//! beta, mu, and eta rows, so this component consumes all of those rows once.
//! An assembly must increase the producer multiplicities by the exact amounts
//! returned there; otherwise LogUp intentionally fails.

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_poseidon2_air::POSEIDON2_WIDTH;
use openvm_recursion_circuit::{
    bus::TranscriptBus,
    define_typed_lookup_bus,
    native_warp::{
        ext::ext_field_multiply, NativeClaimValueBus, NativeClaimValueMessage,
        NativeExactFiniteVaccCallProtocolBus, NativeExactFiniteVaccCallProtocolMessage,
        NativeExactFiniteVaccScheduleBus, NativeExactFiniteVaccScheduleMessage,
        NativeInputSlotLayoutBus, NativeInputSlotLayoutMessage, NativeStandardVaccRootBus,
        NativeStandardVaccRootMessage, NativeWarpTranscriptArtifacts, NativeWarpTranscriptModule,
        CLAIM_SECTION_ALPHA, CLAIM_SECTION_BETA, CLAIM_SECTION_ETA, CLAIM_SECTION_MU,
    },
    system::{BusIndexManager, BusInventory},
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    p3_air::{Air, AirBuilder, BaseAir, PairBuilder},
    p3_field::{
        extension::BinomiallyExtendable, BasedVectorSpace, ExtensionField, PrimeCharacteristicRing,
    },
    p3_matrix::{dense::RowMajorMatrix, Matrix},
    transcript::{TranscriptHistory, TranscriptLog},
    AirRef, BaseAirWithPublicValues, FiatShamirTranscript, PartitionedBaseAir, StarkProtocolConfig,
    SystemParams,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, Digest, DIGEST_SIZE, D_EF, EF, F,
};

use super::{
    ordered_manifest::{
        OrderedManifestBuses, OrderedManifestPrefixProfile, OrderedManifestPrefixReceiptMessage,
        OrderedManifestProfile, OrderedManifestSourceReceiptMessage,
        ORDERED_MANIFEST_MAX_NORMALIZED_LEAVES, ORDERED_MANIFEST_PREFIX_PROTOCOL_VERSION,
        ORDERED_MANIFEST_PREFIX_SLOTS,
    },
    FINITE_WARP_V3_MAX_CALLS, FINITE_WARP_V3_MAX_SOURCES,
};

/// Native tag used by `finite_complete_source_instance_digest`.
///
/// This constant is intentionally mirrored across the SDK/circuit dependency
/// boundary.  Changing it is a protocol change and the differential test below
/// must be updated together with the native implementation.
pub const FRESH_INSTANCE_SOURCE_DIGEST_TAG: u32 = 0x4648_4102;

/// `VmPvs` is laid out as program root, four connector fields, and two memory
/// roots.  Keep this arithmetic constant usable in fixed-size AIR columns.
pub const FRESH_INSTANCE_VM_PVS_WIDTH: usize = 4 + 3 * DIGEST_SIZE;

/// Maximum exact-finite call arity supported by the prefix statement.
pub const FRESH_INSTANCE_MAX_INPUT_ARITY: usize = ORDERED_MANIFEST_PREFIX_SLOTS;

/// Maximum number of recursive verifier children carried by one fixed PESAT
/// source. This is a row-axis bound and is independent from call arity.
pub const FRESH_INSTANCE_MAX_ACTIVE_CHILDREN_PER_SOURCE: u8 = 8;

/// Extra consumers introduced by this component.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FreshInstanceLinkConsumerRequirements {
    /// Current ordinary exact-VACC consumers before this projection is added.
    pub ordinary_exact_vacc_fresh_claim_consumers: usize,
    /// Add this many consumers to every fresh claim row (alpha/beta/mu/eta).
    pub extra_fresh_claim_consumers: usize,
    /// Existing ordinary exact-VACC lookup of the sole schedule row.
    pub ordinary_exact_schedule_lookups: usize,
    /// Add this many lookups to the sole exact schedule table row.
    pub extra_exact_schedule_lookups: usize,
    /// Existing ordinary exact-VACC lookup of each call-protocol row.
    pub ordinary_exact_call_protocol_lookups_per_call: usize,
    /// Add this many lookups to each exact call-protocol table row.
    pub extra_exact_call_protocol_lookups_per_call: usize,
    /// Existing ordinary exact-VACC lookup of each fresh-root row.
    pub ordinary_exact_fresh_root_lookups_per_call: usize,
    /// Add this many lookups to each fresh-root table row.
    pub extra_fresh_root_lookups_per_call: usize,
    /// Add this many lookups to each fresh input-slot table row.
    pub extra_fresh_slot_lookups_per_source: usize,
    /// One prefix-binding receipt is consumed per call.
    pub prefix_binding_lookups_per_call: usize,
    /// The prefix verifier supplies exactly one authority receipt to the
    /// ordered manifest per call. FreshInstanceLink does not consume this bus.
    pub prefix_authority_lookups_per_call: usize,
    /// The ordered manifest republishes exactly one prefix-binding receipt to
    /// FreshInstanceLink per call.
    pub prefix_binding_keys_per_call: usize,
    /// One ordered-manifest source-authority key is produced by this component
    /// per fresh source.
    pub source_authority_keys_per_source: usize,
    /// The ordered manifest consumes each source-authority key exactly once.
    pub source_authority_lookups_per_source: usize,
}

impl FreshInstanceLinkConsumerRequirements {
    pub const EXACT: Self = Self {
        ordinary_exact_vacc_fresh_claim_consumers: 1,
        extra_fresh_claim_consumers: 1,
        ordinary_exact_schedule_lookups: 1,
        extra_exact_schedule_lookups: 1,
        ordinary_exact_call_protocol_lookups_per_call: 1,
        extra_exact_call_protocol_lookups_per_call: 1,
        ordinary_exact_fresh_root_lookups_per_call: 1,
        extra_fresh_root_lookups_per_call: 1,
        extra_fresh_slot_lookups_per_source: 1,
        prefix_binding_lookups_per_call: 1,
        prefix_authority_lookups_per_call: 1,
        prefix_binding_keys_per_call: 1,
        source_authority_keys_per_source: 1,
        source_authority_lookups_per_source: 1,
    };

    /// Total fresh-claim producer count when the ordinary exact VACC verifier
    /// is the only pre-existing component. The exact-finite verifier's twin
    /// and fold machinery collectively uses one lookup per fresh claim row;
    /// this projection raises the uniform producer count from one to two.
    #[must_use]
    pub const fn fresh_claim_consumer_count_with_exact_vacc(self) -> usize {
        self.ordinary_exact_vacc_fresh_claim_consumers + self.extra_fresh_claim_consumers
    }

    #[must_use]
    pub const fn exact_schedule_producer_count(self) -> usize {
        self.ordinary_exact_schedule_lookups + self.extra_exact_schedule_lookups
    }

    #[must_use]
    pub const fn exact_call_protocol_producer_count_per_call(self) -> usize {
        self.ordinary_exact_call_protocol_lookups_per_call
            + self.extra_exact_call_protocol_lookups_per_call
    }

    #[must_use]
    pub const fn fresh_root_producer_count_per_call(self) -> usize {
        self.ordinary_exact_fresh_root_lookups_per_call + self.extra_fresh_root_lookups_per_call
    }
}

/// VK-owned layout of one complete fixed-relation claim.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FreshInstanceExplicitLayoutProfile {
    pub alpha_len: usize,
    pub beta_len: usize,
    pub log_constraints: usize,
    pub distinguished_one_explicit_offset: usize,
    pub alpha_explicit_offset: usize,
    pub beta_explicit_offset: usize,
    pub beta_power_explicit_offset: usize,
    pub beta_power_count: usize,
    pub public_values_explicit_offset: usize,
    /// Setup-fixed start of the aggregate `VmPvs` inside the explicit vector.
    pub vm_pvs_explicit_offset: usize,
}

impl FreshInstanceExplicitLayoutProfile {
    #[must_use]
    pub fn explicit_len(&self) -> Option<usize> {
        self.beta_len.checked_sub(self.log_constraints)
    }

    #[must_use]
    pub fn claim_rows_per_source(&self) -> Option<usize> {
        self.alpha_len.checked_add(self.beta_len)?.checked_add(2)
    }

    pub fn validate(&self) -> Result<(), FreshInstanceLinkError> {
        let explicit_len = self
            .explicit_len()
            .ok_or(FreshInstanceLinkError::InvalidProfile("explicit length"))?;
        if self.alpha_len == 0
            || self.log_constraints >= self.beta_len
            || explicit_len == 0
            || self.distinguished_one_explicit_offset != 0
            || self.alpha_explicit_offset != 1
            || self.beta_explicit_offset != 2
            || self.beta_power_explicit_offset != 3
            || self.beta_power_count == 0
            || self.public_values_explicit_offset
                != self
                    .beta_power_explicit_offset
                    .checked_add(self.beta_power_count)
                    .ok_or(FreshInstanceLinkError::IntegerOverflow)?
            || self.vm_pvs_explicit_offset < self.public_values_explicit_offset
            || self
                .vm_pvs_explicit_offset
                .checked_add(FRESH_INSTANCE_VM_PVS_WIDTH)
                .is_none_or(|end| end > explicit_len)
        {
            return Err(FreshInstanceLinkError::InvalidProfile(
                "complete explicit layout",
            ));
        }
        self.claim_rows_per_source()
            .ok_or(FreshInstanceLinkError::IntegerOverflow)?;
        Ok(())
    }
}

/// VK-owned exact-finite call and prefix layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FreshInstanceCallProfile {
    pub proof_idx: usize,
    pub start_tidx: usize,
    pub call_index: usize,
    pub input_arity: usize,
    pub fresh_count: usize,
    pub prior_count: usize,
    pub source_start: usize,
    pub normalized_leaf_start: usize,
    pub expected_active_child_counts: [u8; ORDERED_MANIFEST_PREFIX_SLOTS],
    pub prefix: OrderedManifestPrefixProfile,
}

impl FreshInstanceCallProfile {
    #[must_use]
    pub const fn slot_variant(&self) -> usize {
        self.fresh_count + self.prior_count * (self.input_arity + 1)
    }

    fn normalized_leaf_count(&self) -> Result<usize, FreshInstanceLinkError> {
        self.expected_active_child_counts[..self.fresh_count]
            .iter()
            .try_fold(0usize, |sum, count| {
                sum.checked_add(usize::from(*count))
                    .ok_or(FreshInstanceLinkError::IntegerOverflow)
            })
    }

    fn normalized_start_for(&self, local_source: usize) -> Result<usize, FreshInstanceLinkError> {
        self.expected_active_child_counts[..local_source]
            .iter()
            .try_fold(self.normalized_leaf_start, |sum, count| {
                sum.checked_add(usize::from(*count))
                    .ok_or(FreshInstanceLinkError::IntegerOverflow)
            })
    }
}

/// Complete VK-owned profile for this projection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FreshInstanceLinkProfile {
    pub protocol_digest: Digest,
    pub relation_digest: Digest,
    pub warp_index_digest: Digest,
    pub setup_digest: Digest,
    pub schedule_digest: Digest,
    pub schedule_end_tidx: usize,
    pub layout: FreshInstanceExplicitLayoutProfile,
    pub calls: Vec<FreshInstanceCallProfile>,
}

impl FreshInstanceLinkProfile {
    #[must_use]
    pub const fn consumer_requirements(&self) -> FreshInstanceLinkConsumerRequirements {
        FreshInstanceLinkConsumerRequirements::EXACT
    }

    #[must_use]
    pub fn total_fresh(&self) -> usize {
        self.calls.iter().map(|call| call.fresh_count).sum()
    }

    #[must_use]
    pub fn total_normalized_leaves(&self) -> Option<usize> {
        self.calls.last().and_then(|last| {
            last.expected_active_child_counts[..last.fresh_count]
                .iter()
                .try_fold(last.normalized_leaf_start, |sum, count| {
                    sum.checked_add(usize::from(*count))
                })
        })
    }

    /// Fail closed if this exact-claim projection cannot feed the supplied
    /// ordered-manifest verifier key. The verifier component digest belongs to
    /// the manifest statement and is intentionally not duplicated here.
    pub fn validate_ordered_manifest_profile(
        &self,
        manifest: &OrderedManifestProfile,
    ) -> Result<(), FreshInstanceLinkError> {
        self.validate()?;
        manifest
            .validate()
            .map_err(|_| FreshInstanceLinkError::InvalidProfile("ordered manifest profile"))?;
        let max_active_children = self
            .calls
            .iter()
            .flat_map(|call| call.expected_active_child_counts[..call.fresh_count].iter())
            .copied()
            .max()
            .ok_or(FreshInstanceLinkError::InvalidProfile(
                "active-child schedule",
            ))?;
        if manifest.protocol_digest != self.protocol_digest
            || manifest.relation_digest != self.relation_digest
            || manifest.warp_index_digest != self.warp_index_digest
            || manifest.prefix != self.calls[0].prefix
            || usize::try_from(manifest.expected_normalized_leaf_count).ok()
                != self.total_normalized_leaves()
            || max_active_children > manifest.max_active_children_per_source
        {
            return Err(FreshInstanceLinkError::InvalidProfile(
                "ordered manifest compatibility",
            ));
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<(), FreshInstanceLinkError> {
        self.layout.validate()?;
        if self.calls.is_empty() || self.calls.len() > FINITE_WARP_V3_MAX_CALLS {
            return Err(FreshInstanceLinkError::InvalidProfile(
                "call schedule capacity",
            ));
        }
        for (name, digest) in [
            ("protocol digest", self.protocol_digest),
            ("relation digest", self.relation_digest),
            ("WARP index digest", self.warp_index_digest),
            ("setup digest", self.setup_digest),
            ("schedule digest", self.schedule_digest),
        ] {
            if digest.iter().all(|value| *value == F::ZERO) {
                return Err(FreshInstanceLinkError::InvalidProfile(name));
            }
        }

        let mut expected_source_start = 0usize;
        let mut expected_normalized_start = 0usize;
        let canonical_prefix = self.calls[0].prefix;
        for (call_index, call) in self.calls.iter().enumerate() {
            if call.call_index != call_index
                || call.proof_idx != call_index
                || call.source_start != expected_source_start
                || call.normalized_leaf_start != expected_normalized_start
                || call.input_arity < 2
                || call.input_arity > FRESH_INSTANCE_MAX_INPUT_ARITY
                || !call.input_arity.is_power_of_two()
                || call.fresh_count == 0
                || call.prior_count != usize::from(call_index != 0)
                || call.fresh_count + call.prior_count != call.input_arity
                || call.prefix.l_skip != 0
                || call.prefix.log_codeword_len
                    != call
                        .prefix
                        .log_message_len
                        .checked_add(call.prefix.log_blowup)
                        .ok_or(FreshInstanceLinkError::IntegerOverflow)?
                || call.prefix.rows_per_leaf != 16
                || call.prefix.trace_prefix_len == 0
                || call.prefix.active_count_log_height >= 63
                || call.prefix != canonical_prefix
                || call
                    .prefix
                    .index_digest
                    .iter()
                    .all(|value| *value == F::ZERO)
            {
                return Err(FreshInstanceLinkError::InvalidProfile("call profile"));
            }
            // `input_arity` is the power-of-two arity of the WARP call.  The
            // number of recursive verifier children carried by one fresh
            // fixed-relation source is independent: the native prefix
            // protocol authenticates any nonzero count up to the fixed slot
            // capacity (the production partition currently uses three).
            if call.expected_active_child_counts[..call.fresh_count]
                .iter()
                .any(|&count| count == 0 || count > FRESH_INSTANCE_MAX_ACTIVE_CHILDREN_PER_SOURCE)
                || call.expected_active_child_counts[call.fresh_count..]
                    .iter()
                    .any(|count| *count != 0)
            {
                return Err(FreshInstanceLinkError::InvalidProfile(
                    "active-child schedule",
                ));
            }
            let message_len = 1u64.checked_shl(call.prefix.log_message_len).ok_or(
                FreshInstanceLinkError::InvalidProfile("prefix message length"),
            )?;
            let codeword_len = 1u64.checked_shl(call.prefix.log_codeword_len).ok_or(
                FreshInstanceLinkError::InvalidProfile("prefix codeword length"),
            )?;
            let active_height = 1u64 << call.prefix.active_count_log_height;
            if call.prefix.trace_prefix_len > message_len
                || codeword_len % u64::from(call.prefix.rows_per_leaf) != 0
                || call.prefix.active_count_block_start % active_height != 0
                || call
                    .prefix
                    .active_count_block_start
                    .checked_add(active_height)
                    .is_none_or(|end| end > call.prefix.trace_prefix_len)
            {
                return Err(FreshInstanceLinkError::InvalidProfile("prefix layout"));
            }
            expected_source_start = expected_source_start
                .checked_add(call.fresh_count)
                .ok_or(FreshInstanceLinkError::IntegerOverflow)?;
            expected_normalized_start = expected_normalized_start
                .checked_add(call.normalized_leaf_count()?)
                .ok_or(FreshInstanceLinkError::IntegerOverflow)?;
        }
        if expected_source_start == 0
            || expected_source_start > FINITE_WARP_V3_MAX_SOURCES as usize
            || expected_normalized_start > ORDERED_MANIFEST_MAX_NORMALIZED_LEAVES as usize
        {
            return Err(FreshInstanceLinkError::InvalidProfile("source capacity"));
        }
        Ok(())
    }
}

/// Dynamic fresh claim emitted by the exact recursive WARP verifier.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FreshInstanceClaimRecord {
    pub alpha: Vec<EF>,
    pub beta: Vec<EF>,
    pub mu: EF,
    pub eta: EF,
}

/// Dynamic values shared by all fresh sources in one exact-finite call.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FreshInstanceCallRecord {
    pub source_order_digest: Digest,
    pub base_root: Digest,
    pub full_root: Digest,
    pub logup_alpha: EF,
    pub logup_beta: EF,
    pub fresh_claims: Vec<FreshInstanceClaimRecord>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FreshInstanceLinkRecord {
    pub calls: Vec<FreshInstanceCallRecord>,
}

/// Dynamic context authenticated jointly by exact WARP and the prefix proof.
#[repr(C)]
#[derive(AlignedBorrow, Debug, Clone)]
pub struct FreshInstanceSourceContextMessage<T> {
    pub proof_idx: T,
    pub call_index: T,
    pub local_source: T,
    pub source_index: T,
    pub normalized_leaf_start: T,
    pub active_child_count: T,
    pub full_root: [T; DIGEST_SIZE],
    pub logup_alpha: [T; D_EF],
    pub logup_beta: [T; D_EF],
}

define_typed_lookup_bus!(
    FreshInstanceSourceContextBus,
    FreshInstanceSourceContextMessage
);

/// All external authorities consumed or produced by this component.
#[derive(Clone, Copy, Debug)]
pub struct FreshInstanceLinkBuses {
    pub claim_value: NativeClaimValueBus,
    pub input_slot: NativeInputSlotLayoutBus,
    pub exact_schedule: NativeExactFiniteVaccScheduleBus,
    pub exact_call_protocol: NativeExactFiniteVaccCallProtocolBus,
    pub vacc_root: NativeStandardVaccRootBus,
    pub digest_transcript: TranscriptBus,
    pub source_context: FreshInstanceSourceContextBus,
    pub ordered_manifest: OrderedManifestBuses,
}

impl FreshInstanceLinkBuses {
    #[must_use]
    pub fn new(
        indices: &mut BusIndexManager,
        claim_value: NativeClaimValueBus,
        input_slot: NativeInputSlotLayoutBus,
        exact_schedule: NativeExactFiniteVaccScheduleBus,
        exact_call_protocol: NativeExactFiniteVaccCallProtocolBus,
        vacc_root: NativeStandardVaccRootBus,
        ordered_manifest: OrderedManifestBuses,
    ) -> Self {
        Self {
            claim_value,
            input_slot,
            exact_schedule,
            exact_call_protocol,
            vacc_root,
            digest_transcript: TranscriptBus::new(indices.new_bus_idx()),
            source_context: FreshInstanceSourceContextBus::new(indices.new_bus_idx()),
            ordered_manifest,
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FreshInstanceCallLinkPrepCols<T> {
    pub active: T,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FreshInstanceCallLinkCols<T> {
    pub fixed_guard: T,
    pub source_order_digest: [T; DIGEST_SIZE],
    pub base_root: [T; DIGEST_SIZE],
    pub full_root: [T; DIGEST_SIZE],
    pub logup_alpha: [T; D_EF],
    pub logup_beta: [T; D_EF],
}

/// One fixed call's schedule/root/prefix linker.
#[derive(ColumnsAir, Clone, Debug)]
#[columns_via(FreshInstanceCallLinkCols<u8>)]
pub struct FreshInstanceCallLinkAir {
    pub profile: FreshInstanceLinkProfile,
    pub call_index: usize,
    pub buses: FreshInstanceLinkBuses,
}

impl FreshInstanceCallLinkAir {
    pub fn new(
        profile: FreshInstanceLinkProfile,
        call_index: usize,
        buses: FreshInstanceLinkBuses,
    ) -> Result<Self, FreshInstanceLinkError> {
        profile.validate()?;
        if call_index >= profile.calls.len() {
            return Err(FreshInstanceLinkError::InvalidProfile("call index"));
        }
        Ok(Self {
            profile,
            call_index,
            buses,
        })
    }

    pub fn generate_trace(
        &self,
        record: &FreshInstanceCallRecord,
    ) -> Result<RowMajorMatrix<F>, FreshInstanceLinkError> {
        validate_call_record(&self.profile, self.call_index, record)?;
        let width = self.width();
        let mut values = F::zero_vec(2 * width);
        let local: &mut FreshInstanceCallLinkCols<F> = values[..width].borrow_mut();
        local.fixed_guard = F::ONE;
        local.source_order_digest = record.source_order_digest;
        local.base_root = record.base_root;
        local.full_root = record.full_root;
        local.logup_alpha = ef_limbs(record.logup_alpha);
        local.logup_beta = ef_limbs(record.logup_beta);
        Ok(RowMajorMatrix::new(values, width))
    }
}

impl BaseAir<F> for FreshInstanceCallLinkAir {
    fn width(&self) -> usize {
        FreshInstanceCallLinkCols::<F>::width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        assert!(self.profile.validate().is_ok());
        let width = FreshInstanceCallLinkPrepCols::<F>::width();
        let mut values = F::zero_vec(2 * width);
        let first: &mut FreshInstanceCallLinkPrepCols<F> = values[..width].borrow_mut();
        first.active = F::ONE;
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for FreshInstanceCallLinkAir {}
impl PartitionedBaseAir<F> for FreshInstanceCallLinkAir {}

impl<AB> Air<AB> for FreshInstanceCallLinkAir
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.profile.validate().is_ok());
        let call = &self.profile.calls[self.call_index];
        let prep_matrix = builder.preprocessed();
        let prep_row = prep_matrix.row_slice(0).expect("fresh call-link prep row");
        let prep: &FreshInstanceCallLinkPrepCols<AB::Var> = (*prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("fresh call-link row");
        let local: &FreshInstanceCallLinkCols<AB::Var> = (*row).borrow();
        let enabled = AB::Expr::from(prep.active);

        builder.assert_bool(prep.active);
        builder.assert_bool(local.fixed_guard);
        builder.assert_eq(local.fixed_guard, prep.active);
        builder.when_first_row().assert_one(prep.active);
        builder.when_last_row().assert_zero(prep.active);
        for value in row.iter().skip(1) {
            builder
                .when(AB::Expr::ONE - enabled.clone())
                .assert_zero(*value);
        }

        let proof_idx = AB::Expr::from_usize(call.proof_idx);
        if self.call_index == 0 {
            self.buses.exact_schedule.lookup_key(
                builder,
                NativeExactFiniteVaccScheduleMessage {
                    proof_idx: proof_idx.clone(),
                    end_tidx: AB::Expr::from_usize(self.profile.schedule_end_tidx),
                    call_count: AB::Expr::from_usize(self.profile.calls.len()),
                    total_fresh: AB::Expr::from_usize(self.profile.total_fresh()),
                    relation_digest: self.profile.relation_digest.map(AB::Expr::from),
                    index_digest: self.profile.warp_index_digest.map(AB::Expr::from),
                    setup_digest: self.profile.setup_digest.map(AB::Expr::from),
                    schedule_digest: self.profile.schedule_digest.map(AB::Expr::from),
                },
                enabled.clone(),
            );
        }
        self.buses.exact_call_protocol.lookup_key(
            builder,
            NativeExactFiniteVaccCallProtocolMessage {
                proof_idx: proof_idx.clone(),
                start_tidx: AB::Expr::from_usize(call.start_tidx),
                call_index: AB::Expr::from_usize(call.call_index),
                input_arity: AB::Expr::from_usize(call.input_arity),
                fresh_count: AB::Expr::from_usize(call.fresh_count),
                prior_count: AB::Expr::from_usize(call.prior_count),
                relation_digest: self.profile.relation_digest.map(AB::Expr::from),
                index_digest: self.profile.warp_index_digest.map(AB::Expr::from),
                setup_digest: self.profile.setup_digest.map(AB::Expr::from),
                schedule_digest: self.profile.schedule_digest.map(AB::Expr::from),
            },
            enabled.clone(),
        );
        self.buses.vacc_root.lookup_key(
            builder,
            NativeStandardVaccRootMessage {
                proof_idx: proof_idx.clone(),
                kind: AB::Expr::ZERO,
                root: local.full_root.map(Into::into),
            },
            enabled.clone(),
        );

        self.buses.ordered_manifest.prefix_binding.lookup_key(
            builder,
            prefix_message_expr::<AB>(&self.profile, call, local),
            enabled.clone(),
        );

        let rows_per_source = self
            .profile
            .layout
            .claim_rows_per_source()
            .expect("validated claim width");
        for local_source in 0..call.fresh_count {
            self.buses.input_slot.lookup_key(
                builder,
                NativeInputSlotLayoutMessage {
                    variant: AB::Expr::from_usize(call.slot_variant()),
                    source: AB::Expr::from_usize(local_source),
                    kind: [AB::Expr::ONE, AB::Expr::ZERO, AB::Expr::ZERO],
                },
                enabled.clone(),
            );
            self.buses.source_context.add_key_with_lookups(
                builder,
                FreshInstanceSourceContextMessage {
                    proof_idx: proof_idx.clone(),
                    call_index: AB::Expr::from_usize(call.call_index),
                    local_source: AB::Expr::from_usize(local_source),
                    source_index: AB::Expr::from_usize(call.source_start + local_source),
                    normalized_leaf_start: AB::Expr::from_usize(
                        call.normalized_start_for(local_source)
                            .expect("validated normalized source start"),
                    ),
                    active_child_count: AB::Expr::from_u8(
                        call.expected_active_child_counts[local_source],
                    ),
                    full_root: local.full_root.map(Into::into),
                    logup_alpha: local.logup_alpha.map(Into::into),
                    logup_beta: local.logup_beta.map(Into::into),
                },
                enabled.clone() * AB::Expr::from_usize(rows_per_source),
            );
        }
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FreshInstanceProjectionPrepCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub call_index: T,
    pub local_source: T,
    pub source_index: T,
    pub normalized_leaf_start: T,
    pub active_child_count: T,
    pub section: T,
    pub coordinate: T,
    pub is_source_first: T,
    pub is_explicit: T,
    pub explicit_coordinate: T,
    pub is_distinguished_one: T,
    pub is_alpha: T,
    pub is_beta: T,
    pub is_beta_power: T,
    pub is_beta_power_first: T,
    pub is_beta_power_last: T,
    /// Fixed `is_beta_power && !is_beta_power_last` helper. Materializing
    /// this setup-owned selector keeps the extension-multiplication
    /// transition within the recursive lane's degree-four bound.
    pub beta_power_continues: T,
    pub is_public: T,
    pub vm_selector: [T; FRESH_INSTANCE_VM_PVS_WIDTH],
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct FreshInstanceProjectionCols<T> {
    pub fixed_guard: T,
    pub value: [T; D_EF],
    pub full_root: [T; DIGEST_SIZE],
    pub logup_alpha: [T; D_EF],
    pub logup_beta: [T; D_EF],
    pub source_instance_digest: [T; DIGEST_SIZE],
    pub vm_pvs: [T; FRESH_INSTANCE_VM_PVS_WIDTH],
}

/// Claim-coordinate projection and source-authority producer for one call.
#[derive(ColumnsAir, Clone, Debug)]
#[columns_via(FreshInstanceProjectionCols<u8>)]
pub struct FreshInstanceProjectionAir {
    pub profile: FreshInstanceLinkProfile,
    pub call_index: usize,
    pub buses: FreshInstanceLinkBuses,
}

pub type FreshInstanceProjectionTrace = (RowMajorMatrix<F>, Vec<TranscriptLog<F, [F; 16]>>);

impl FreshInstanceProjectionAir {
    pub fn new(
        profile: FreshInstanceLinkProfile,
        call_index: usize,
        buses: FreshInstanceLinkBuses,
    ) -> Result<Self, FreshInstanceLinkError> {
        profile.validate()?;
        if call_index >= profile.calls.len() {
            return Err(FreshInstanceLinkError::InvalidProfile(
                "projection call index",
            ));
        }
        Ok(Self {
            profile,
            call_index,
            buses,
        })
    }

    pub fn generate_trace(
        &self,
        record: &FreshInstanceCallRecord,
    ) -> Result<FreshInstanceProjectionTrace, FreshInstanceLinkError> {
        validate_call_record(&self.profile, self.call_index, record)?;
        let call = &self.profile.calls[self.call_index];
        let rows_per_source = self
            .profile
            .layout
            .claim_rows_per_source()
            .ok_or(FreshInstanceLinkError::IntegerOverflow)?;
        let active_rows = rows_per_source
            .checked_mul(call.fresh_count)
            .ok_or(FreshInstanceLinkError::IntegerOverflow)?;
        let height = active_rows
            .checked_next_power_of_two()
            .ok_or(FreshInstanceLinkError::IntegerOverflow)?
            .max(2);
        let width = self.width();
        let mut values = F::zero_vec(height * width);
        let full_root = record.full_root;
        let alpha = ef_limbs(record.logup_alpha);
        let beta = ef_limbs(record.logup_beta);
        let mut logs = Vec::with_capacity(call.fresh_count);

        for (local_source, claim) in record.fresh_claims.iter().enumerate() {
            let explicit = explicit_assignment(&self.profile.layout, claim)?;
            validate_explicit(
                &self.profile.layout,
                &explicit,
                record.logup_alpha,
                record.logup_beta,
            )?;
            let (source_instance_digest, log) = source_digest_and_log(&explicit);
            let vm_pvs = extract_vm_pvs(&self.profile.layout, &explicit)?;
            logs.push(log);
            let mut ordinal = 0usize;
            for (_section, section_values) in [
                (CLAIM_SECTION_ALPHA, claim.alpha.as_slice()),
                (CLAIM_SECTION_BETA, claim.beta.as_slice()),
                (CLAIM_SECTION_MU, core::slice::from_ref(&claim.mu)),
                (CLAIM_SECTION_ETA, core::slice::from_ref(&claim.eta)),
            ] {
                for value in section_values {
                    let row_index = local_source
                        .checked_mul(rows_per_source)
                        .and_then(|start| start.checked_add(ordinal))
                        .ok_or(FreshInstanceLinkError::IntegerOverflow)?;
                    let local: &mut FreshInstanceProjectionCols<F> =
                        values[row_index * width..(row_index + 1) * width].borrow_mut();
                    local.fixed_guard = F::ONE;
                    local.value = ef_limbs(*value);
                    local.full_root = full_root;
                    local.logup_alpha = alpha;
                    local.logup_beta = beta;
                    local.source_instance_digest = source_instance_digest;
                    local.vm_pvs = vm_pvs;
                    ordinal += 1;
                }
            }
            debug_assert_eq!(ordinal, rows_per_source);
        }
        Ok((RowMajorMatrix::new(values, width), logs))
    }
}

impl BaseAir<F> for FreshInstanceProjectionAir {
    fn width(&self) -> usize {
        FreshInstanceProjectionCols::<F>::width()
    }

    fn preprocessed_trace(&self) -> Option<RowMajorMatrix<F>> {
        assert!(self.profile.validate().is_ok());
        let call = &self.profile.calls[self.call_index];
        let layout = &self.profile.layout;
        let rows_per_source = layout
            .claim_rows_per_source()
            .expect("validated claim width");
        let active_rows = rows_per_source
            .checked_mul(call.fresh_count)
            .expect("validated projection height");
        let height = active_rows.next_power_of_two().max(2);
        let width = FreshInstanceProjectionPrepCols::<F>::width();
        let mut values = F::zero_vec(height * width);
        for local_source in 0..call.fresh_count {
            let mut ordinal = 0usize;
            for (section, section_len) in [
                (CLAIM_SECTION_ALPHA, layout.alpha_len),
                (CLAIM_SECTION_BETA, layout.beta_len),
                (CLAIM_SECTION_MU, 1),
                (CLAIM_SECTION_ETA, 1),
            ] {
                for coordinate in 0..section_len {
                    let row_index = local_source * rows_per_source + ordinal;
                    let prep: &mut FreshInstanceProjectionPrepCols<F> =
                        values[row_index * width..(row_index + 1) * width].borrow_mut();
                    prep.active = F::ONE;
                    prep.proof_idx = F::from_usize(call.proof_idx);
                    prep.call_index = F::from_usize(call.call_index);
                    prep.local_source = F::from_usize(local_source);
                    prep.source_index = F::from_usize(call.source_start + local_source);
                    prep.normalized_leaf_start = F::from_usize(
                        call.normalized_start_for(local_source)
                            .expect("validated normalized source start"),
                    );
                    prep.active_child_count =
                        F::from_u8(call.expected_active_child_counts[local_source]);
                    prep.section = F::from_usize(section);
                    prep.coordinate = F::from_usize(coordinate);
                    prep.is_source_first = F::from_bool(ordinal == 0);
                    if section == CLAIM_SECTION_BETA && coordinate >= layout.log_constraints {
                        let explicit_coordinate = coordinate - layout.log_constraints;
                        prep.is_explicit = F::ONE;
                        prep.explicit_coordinate = F::from_usize(explicit_coordinate);
                        prep.is_distinguished_one = F::from_bool(
                            explicit_coordinate == layout.distinguished_one_explicit_offset,
                        );
                        prep.is_alpha =
                            F::from_bool(explicit_coordinate == layout.alpha_explicit_offset);
                        prep.is_beta =
                            F::from_bool(explicit_coordinate == layout.beta_explicit_offset);
                        let beta_power_end =
                            layout.beta_power_explicit_offset + layout.beta_power_count;
                        let is_beta_power = explicit_coordinate
                            >= layout.beta_power_explicit_offset
                            && explicit_coordinate < beta_power_end;
                        prep.is_beta_power = F::from_bool(is_beta_power);
                        prep.is_beta_power_first =
                            F::from_bool(explicit_coordinate == layout.beta_power_explicit_offset);
                        prep.is_beta_power_last =
                            F::from_bool(explicit_coordinate + 1 == beta_power_end);
                        prep.beta_power_continues = F::from_bool(
                            is_beta_power && explicit_coordinate + 1 != beta_power_end,
                        );
                        prep.is_public = F::from_bool(
                            explicit_coordinate >= layout.public_values_explicit_offset,
                        );
                        if explicit_coordinate >= layout.vm_pvs_explicit_offset
                            && explicit_coordinate
                                < layout.vm_pvs_explicit_offset + FRESH_INSTANCE_VM_PVS_WIDTH
                        {
                            prep.vm_selector[explicit_coordinate - layout.vm_pvs_explicit_offset] =
                                F::ONE;
                        }
                    }
                    ordinal += 1;
                }
            }
        }
        Some(RowMajorMatrix::new(values, width))
    }
}

impl BaseAirWithPublicValues<F> for FreshInstanceProjectionAir {}
impl PartitionedBaseAir<F> for FreshInstanceProjectionAir {}

impl<AB> Air<AB> for FreshInstanceProjectionAir
where
    AB: AirBuilder<F = F> + PairBuilder + InteractionBuilder,
    AB::Var: Copy,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<{ D_EF }>,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.profile.validate().is_ok());
        let layout = &self.profile.layout;
        let explicit_len = layout.explicit_len().expect("validated explicit length");
        let prep_matrix = builder.preprocessed();
        let prep_row = prep_matrix.row_slice(0).expect("fresh projection prep row");
        let next_prep_row = prep_matrix
            .row_slice(1)
            .expect("fresh projection next prep row");
        let prep: &FreshInstanceProjectionPrepCols<AB::Var> = (*prep_row).borrow();
        let next_prep: &FreshInstanceProjectionPrepCols<AB::Var> = (*next_prep_row).borrow();
        let main = builder.main();
        let row = main.row_slice(0).expect("fresh projection row");
        let next_row = main.row_slice(1).expect("fresh projection next row");
        let local: &FreshInstanceProjectionCols<AB::Var> = (*row).borrow();
        let next: &FreshInstanceProjectionCols<AB::Var> = (*next_row).borrow();
        let enabled = AB::Expr::from(prep.active);

        for flag in [
            prep.active,
            prep.is_source_first,
            prep.is_explicit,
            prep.is_distinguished_one,
            prep.is_alpha,
            prep.is_beta,
            prep.is_beta_power,
            prep.is_beta_power_first,
            prep.is_beta_power_last,
            prep.beta_power_continues,
            prep.is_public,
        ]
        .into_iter()
        .chain(prep.vm_selector)
        {
            builder.assert_bool(flag);
        }
        builder.assert_eq(
            prep.beta_power_continues,
            AB::Expr::from(prep.is_beta_power)
                * (AB::Expr::ONE - AB::Expr::from(prep.is_beta_power_last)),
        );
        builder.assert_bool(local.fixed_guard);
        builder.assert_eq(local.fixed_guard, prep.active);
        for value in row.iter().skip(1) {
            builder
                .when(AB::Expr::ONE - enabled.clone())
                .assert_zero(*value);
        }

        self.buses.claim_value.receive(
            builder,
            NativeClaimValueMessage {
                proof_idx: prep.proof_idx.into(),
                source: prep.local_source.into(),
                section: prep.section.into(),
                coordinate: prep.coordinate.into(),
                value: local.value.map(Into::into),
            },
            enabled.clone(),
        );
        self.buses.source_context.lookup_key(
            builder,
            FreshInstanceSourceContextMessage {
                proof_idx: prep.proof_idx.into(),
                call_index: prep.call_index.into(),
                local_source: prep.local_source.into(),
                source_index: prep.source_index.into(),
                normalized_leaf_start: prep.normalized_leaf_start.into(),
                active_child_count: prep.active_child_count.into(),
                full_root: local.full_root.map(Into::into),
                logup_alpha: local.logup_alpha.map(Into::into),
                logup_beta: local.logup_beta.map(Into::into),
            },
            enabled.clone(),
        );

        let source_first = AB::Expr::from(prep.is_source_first);
        self.buses.digest_transcript.observe(
            builder,
            prep.source_index,
            AB::Expr::ZERO,
            AB::Expr::from_u32(FRESH_INSTANCE_SOURCE_DIGEST_TAG),
            source_first.clone(),
        );
        self.buses.digest_transcript.observe(
            builder,
            prep.source_index,
            AB::Expr::ONE,
            AB::Expr::from_usize(explicit_len),
            source_first.clone(),
        );
        self.buses.digest_transcript.observe(
            builder,
            prep.source_index,
            AB::Expr::TWO,
            AB::Expr::from_usize(D_EF),
            source_first.clone(),
        );
        self.buses.digest_transcript.observe_ext(
            builder,
            prep.source_index,
            AB::Expr::from_usize(3)
                + AB::Expr::from(prep.explicit_coordinate) * AB::Expr::from_usize(D_EF),
            local.value,
            prep.is_explicit,
        );
        let digest_tidx = 3 + explicit_len * D_EF;
        for (limb, digest_value) in local.source_instance_digest.into_iter().enumerate() {
            self.buses.digest_transcript.sample(
                builder,
                prep.source_index,
                AB::Expr::from_usize(digest_tidx + limb),
                digest_value,
                source_first.clone(),
            );
        }

        let same_source = AB::Expr::from(next_prep.active)
            * (AB::Expr::ONE - AB::Expr::from(next_prep.is_source_first));
        let mut transition_builder = builder.when_transition();
        let mut transition = transition_builder.when(same_source);
        for (next_value, local_value) in next
            .source_instance_digest
            .into_iter()
            .zip(local.source_instance_digest)
            .chain(next.vm_pvs.into_iter().zip(local.vm_pvs))
        {
            transition.assert_eq(next_value, local_value);
        }

        let one: [AB::Expr; D_EF] = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            }
        });
        for (actual, expected) in local.value.into_iter().zip(one.clone()) {
            builder
                .when(prep.is_distinguished_one)
                .assert_eq(actual, expected.clone());
            builder
                .when(prep.is_beta_power_first)
                .assert_eq(actual, expected);
        }
        for limb in 0..D_EF {
            builder
                .when(prep.is_alpha)
                .assert_eq(local.value[limb], local.logup_alpha[limb]);
            builder
                .when(prep.is_beta)
                .assert_eq(local.value[limb], local.logup_beta[limb]);
        }
        let beta_power_continues = AB::Expr::from(prep.beta_power_continues);
        let product = ext_field_multiply::<AB::Expr>(local.value, local.logup_beta);
        let mut transition_builder = builder.when_transition();
        let mut transition = transition_builder.when(beta_power_continues);
        for (actual, expected) in next.value.into_iter().zip(product) {
            transition.assert_eq(actual, expected);
        }
        for limb in local.value.into_iter().skip(1) {
            builder.when(prep.is_public).assert_zero(limb);
        }
        for (selector, projected) in prep.vm_selector.into_iter().zip(local.vm_pvs) {
            builder.when(selector).assert_eq(local.value[0], projected);
        }

        self.buses
            .ordered_manifest
            .source_authority
            .add_key_with_lookups(
                builder,
                source_authority_message_expr::<AB>(&self.profile, local, prep),
                source_first,
            );
    }
}

/// Stable AIR/context identity used by the production owner.
///
/// The order is protocol-facing: an owner must build proving contexts in this
/// exact sequence, and may omit only the final `Poseidon2` context when its
/// inputs are merged into a shared Poseidon owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FreshInstanceLinkAirKind {
    CallLink { call_index: usize },
    Projection { call_index: usize },
    DigestTranscript,
    Poseidon2,
}

/// Component owner for the two AIR families and the canonical Poseidon
/// transcript used by all source digests.
pub struct FreshInstanceLinkComponent {
    pub profile: FreshInstanceLinkProfile,
    pub buses: FreshInstanceLinkBuses,
    pub call_airs: Vec<FreshInstanceCallLinkAir>,
    pub projection_airs: Vec<FreshInstanceProjectionAir>,
    transcript: NativeWarpTranscriptModule,
}

impl FreshInstanceLinkComponent {
    pub fn new(
        profile: FreshInstanceLinkProfile,
        buses: FreshInstanceLinkBuses,
        shared: &BusInventory,
        params: SystemParams,
    ) -> Result<Self, FreshInstanceLinkError> {
        profile.validate()?;
        let call_airs = (0..profile.calls.len())
            .map(|call_index| FreshInstanceCallLinkAir::new(profile.clone(), call_index, buses))
            .collect::<Result<Vec<_>, _>>()?;
        let projection_airs = (0..profile.calls.len())
            .map(|call_index| FreshInstanceProjectionAir::new(profile.clone(), call_index, buses))
            .collect::<Result<Vec<_>, _>>()?;
        let transcript =
            NativeWarpTranscriptModule::new_for_bus(shared, buses.digest_transcript, params);
        Ok(Self {
            profile,
            buses,
            call_airs,
            projection_airs,
            transcript,
        })
    }

    #[must_use]
    pub fn poseidon2_bus_owner(&self) -> openvm_recursion_circuit::transcript::Poseidon2BusOwner {
        self.transcript.poseidon2_bus_owner()
    }

    /// Deterministic AIR/context order: all call links, all projections,
    /// transcript, Poseidon.
    #[must_use]
    pub fn air_order(&self) -> Vec<FreshInstanceLinkAirKind> {
        let mut order = Vec::with_capacity(2 * self.profile.calls.len() + 2);
        order.extend(
            (0..self.profile.calls.len())
                .map(|call_index| FreshInstanceLinkAirKind::CallLink { call_index }),
        );
        order.extend(
            (0..self.profile.calls.len())
                .map(|call_index| FreshInstanceLinkAirKind::Projection { call_index }),
        );
        order.extend([
            FreshInstanceLinkAirKind::DigestTranscript,
            FreshInstanceLinkAirKind::Poseidon2,
        ]);
        order
    }

    /// AIRs in [`Self::air_order`].
    #[must_use]
    pub fn airs<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        let mut airs = Vec::with_capacity(2 * self.profile.calls.len() + 2);
        airs.extend(
            self.call_airs
                .iter()
                .cloned()
                .map(|air| std::sync::Arc::new(air) as AirRef<SC>),
        );
        airs.extend(
            self.projection_airs
                .iter()
                .cloned()
                .map(|air| std::sync::Arc::new(air) as AirRef<SC>),
        );
        airs.extend(self.transcript.airs::<SC>());
        debug_assert_eq!(airs.len(), self.air_order().len());
        airs
    }

    /// AIRs for an owner that merges this component's Poseidon inputs into a
    /// shared Poseidon context. This is `air_order()` without its final entry
    /// and aligns with [`FreshInstanceLinkTraces::into_shared_poseidon_traces`].
    #[must_use]
    pub fn airs_without_poseidon<SC: StarkProtocolConfig<F = F>>(&self) -> Vec<AirRef<SC>> {
        let mut airs = self.airs::<SC>();
        let removed = airs.pop();
        debug_assert!(removed.is_some());
        airs
    }

    pub fn generate_traces(
        &self,
        record: &FreshInstanceLinkRecord,
    ) -> Result<FreshInstanceLinkTraces, FreshInstanceLinkError> {
        validate_record(&self.profile, record)?;
        let mut call_links = Vec::with_capacity(self.profile.calls.len());
        let mut projections = Vec::with_capacity(self.profile.calls.len());
        let mut logs = Vec::with_capacity(self.profile.total_fresh());
        for (call_index, call_record) in record.calls.iter().enumerate() {
            call_links.push(self.call_airs[call_index].generate_trace(call_record)?);
            let (trace, call_logs) =
                self.projection_airs[call_index].generate_trace(call_record)?;
            projections.push(trace);
            logs.extend(call_logs);
        }
        if logs.len() != self.profile.total_fresh() {
            return Err(FreshInstanceLinkError::RecordShape("source digest logs"));
        }
        let log_refs = logs.iter().collect::<Vec<_>>();
        let transcript = self
            .transcript
            .generate_trace(&log_refs, None)
            .ok_or(FreshInstanceLinkError::TranscriptTrace)?;
        Ok(FreshInstanceLinkTraces {
            call_links,
            projections,
            transcript,
        })
    }
}

pub struct FreshInstanceLinkTraces {
    pub call_links: Vec<RowMajorMatrix<F>>,
    pub projections: Vec<RowMajorMatrix<F>>,
    pub transcript: NativeWarpTranscriptArtifacts,
}

/// Contexts and hash inputs returned when the production owner uses one shared
/// Poseidon AIR for the entire wrapper.
pub struct FreshInstanceLinkSharedPoseidonTraces {
    /// AIR-aligned traces for every entry of `air_order()` except `Poseidon2`.
    pub ordered_traces_without_poseidon: Vec<RowMajorMatrix<F>>,
    pub permutation_inputs: Vec<[F; POSEIDON2_WIDTH]>,
    pub compression_inputs: Vec<[F; POSEIDON2_WIDTH]>,
}

impl FreshInstanceLinkTraces {
    /// Consume the generated traces in the exact order returned by
    /// [`FreshInstanceLinkComponent::air_order`], including this component's
    /// private Poseidon trace.
    #[must_use]
    pub fn into_ordered_traces(self) -> Vec<RowMajorMatrix<F>> {
        let Self {
            call_links,
            projections,
            transcript,
        } = self;
        let mut traces = Vec::with_capacity(call_links.len() + projections.len() + 2);
        traces.extend(call_links);
        traces.extend(projections);
        traces.push(transcript.trace);
        traces.push(transcript.poseidon2_trace);
        traces
    }

    /// Consume the generated traces while preserving the inputs needed by a
    /// shared production Poseidon owner. The returned trace order is exactly
    /// `FreshInstanceLinkComponent::air_order()` with its final `Poseidon2`
    /// entry removed.
    #[must_use]
    pub fn into_shared_poseidon_traces(self) -> FreshInstanceLinkSharedPoseidonTraces {
        let Self {
            call_links,
            projections,
            transcript,
        } = self;
        let NativeWarpTranscriptArtifacts {
            trace,
            poseidon2_trace: _,
            permutation_inputs,
            compression_inputs,
        } = transcript;
        let mut ordered_traces_without_poseidon =
            Vec::with_capacity(call_links.len() + projections.len() + 1);
        ordered_traces_without_poseidon.extend(call_links);
        ordered_traces_without_poseidon.extend(projections);
        ordered_traces_without_poseidon.push(trace);
        FreshInstanceLinkSharedPoseidonTraces {
            ordered_traces_without_poseidon,
            permutation_inputs,
            compression_inputs,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FreshInstanceLinkError {
    InvalidProfile(&'static str),
    RecordShape(&'static str),
    ClaimShape { call: usize, source: usize },
    NonCanonicalExplicit { call: usize, source: usize },
    NonBaseVmPvs { call: usize, source: usize },
    IntegerOverflow,
    TranscriptTrace,
}

impl core::fmt::Display for FreshInstanceLinkError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(formatter, "invalid fresh-instance linkage: {self:?}")
    }
}

impl std::error::Error for FreshInstanceLinkError {}

fn validate_record(
    profile: &FreshInstanceLinkProfile,
    record: &FreshInstanceLinkRecord,
) -> Result<(), FreshInstanceLinkError> {
    profile.validate()?;
    if record.calls.len() != profile.calls.len() {
        return Err(FreshInstanceLinkError::RecordShape("call count"));
    }
    for (call_index, call) in record.calls.iter().enumerate() {
        validate_call_record(profile, call_index, call)?;
    }
    Ok(())
}

fn validate_call_record(
    profile: &FreshInstanceLinkProfile,
    call_index: usize,
    record: &FreshInstanceCallRecord,
) -> Result<(), FreshInstanceLinkError> {
    profile.validate()?;
    let call = profile
        .calls
        .get(call_index)
        .ok_or(FreshInstanceLinkError::RecordShape("call index"))?;
    if record.fresh_claims.len() != call.fresh_count {
        return Err(FreshInstanceLinkError::RecordShape("fresh claim count"));
    }
    for (source, claim) in record.fresh_claims.iter().enumerate() {
        if claim.alpha.len() != profile.layout.alpha_len
            || claim.beta.len() != profile.layout.beta_len
            || claim.eta != EF::ZERO
        {
            return Err(FreshInstanceLinkError::ClaimShape {
                call: call_index,
                source,
            });
        }
        let explicit = explicit_assignment(&profile.layout, claim)?;
        validate_explicit(
            &profile.layout,
            &explicit,
            record.logup_alpha,
            record.logup_beta,
        )
        .map_err(|_| FreshInstanceLinkError::NonCanonicalExplicit {
            call: call_index,
            source,
        })?;
        let _vm_pvs = extract_vm_pvs(&profile.layout, &explicit).map_err(|_| {
            FreshInstanceLinkError::NonBaseVmPvs {
                call: call_index,
                source,
            }
        })?;
    }
    Ok(())
}

fn explicit_assignment(
    layout: &FreshInstanceExplicitLayoutProfile,
    claim: &FreshInstanceClaimRecord,
) -> Result<Vec<EF>, FreshInstanceLinkError> {
    let explicit = claim
        .beta
        .get(layout.log_constraints..)
        .ok_or(FreshInstanceLinkError::RecordShape("beta tail"))?
        .to_vec();
    if explicit.len()
        != layout
            .explicit_len()
            .ok_or(FreshInstanceLinkError::IntegerOverflow)?
    {
        return Err(FreshInstanceLinkError::RecordShape("explicit length"));
    }
    Ok(explicit)
}

fn validate_explicit(
    layout: &FreshInstanceExplicitLayoutProfile,
    explicit: &[EF],
    alpha: EF,
    beta: EF,
) -> Result<(), FreshInstanceLinkError> {
    layout.validate()?;
    if explicit.len() != layout.explicit_len().expect("validated explicit length")
        || explicit[layout.distinguished_one_explicit_offset] != EF::ONE
        || explicit[layout.alpha_explicit_offset] != alpha
        || explicit[layout.beta_explicit_offset] != beta
    {
        return Err(FreshInstanceLinkError::RecordShape(
            "canonical explicit prefix",
        ));
    }
    let powers = &explicit[layout.beta_power_explicit_offset
        ..layout.beta_power_explicit_offset + layout.beta_power_count];
    if powers.first() != Some(&EF::ONE)
        || powers
            .iter()
            .skip(1)
            .zip(powers)
            .any(|(next, previous)| *next != *previous * beta)
        || explicit[layout.public_values_explicit_offset..]
            .iter()
            .any(|value| ef_limbs(*value)[1..].iter().any(|limb| *limb != F::ZERO))
    {
        return Err(FreshInstanceLinkError::RecordShape(
            "canonical explicit assignment",
        ));
    }
    Ok(())
}

fn extract_vm_pvs(
    layout: &FreshInstanceExplicitLayoutProfile,
    explicit: &[EF],
) -> Result<[F; FRESH_INSTANCE_VM_PVS_WIDTH], FreshInstanceLinkError> {
    let values = explicit
        .get(
            layout.vm_pvs_explicit_offset
                ..layout.vm_pvs_explicit_offset + FRESH_INSTANCE_VM_PVS_WIDTH,
        )
        .ok_or(FreshInstanceLinkError::RecordShape("VmPvs range"))?;
    let mut output = [F::ZERO; FRESH_INSTANCE_VM_PVS_WIDTH];
    for (destination, value) in output.iter_mut().zip(values) {
        *destination = value
            .as_base()
            .ok_or(FreshInstanceLinkError::RecordShape("non-base VmPvs"))?;
    }
    Ok(output)
}

fn source_digest_and_log(explicit: &[EF]) -> (Digest, TranscriptLog<F, [F; 16]>) {
    let mut transcript = default_duplex_sponge_recorder();
    <_ as FiatShamirTranscript<crate::SC>>::observe(
        &mut transcript,
        F::from_u32(FRESH_INSTANCE_SOURCE_DIGEST_TAG),
    );
    <_ as FiatShamirTranscript<crate::SC>>::observe(&mut transcript, F::from_usize(explicit.len()));
    <_ as FiatShamirTranscript<crate::SC>>::observe(&mut transcript, F::from_usize(D_EF));
    for value in explicit {
        for &coordinate in value.as_basis_coefficients_slice() {
            <_ as FiatShamirTranscript<crate::SC>>::observe(&mut transcript, coordinate);
        }
    }
    let digest =
        core::array::from_fn(|_| <_ as FiatShamirTranscript<crate::SC>>::sample(&mut transcript));
    (digest, TranscriptHistory::into_log(transcript))
}

pub fn fresh_instance_source_digest(explicit: &[EF]) -> Digest {
    source_digest_and_log(explicit).0
}

fn ef_limbs(value: EF) -> [F; D_EF] {
    value
        .as_basis_coefficients_slice()
        .try_into()
        .expect("configured EF4 basis")
}

fn prefix_message_expr<AB>(
    profile: &FreshInstanceLinkProfile,
    call: &FreshInstanceCallProfile,
    local: &FreshInstanceCallLinkCols<AB::Var>,
) -> OrderedManifestPrefixReceiptMessage<AB::Expr>
where
    AB: AirBuilder<F = F>,
    AB::Var: Copy,
{
    OrderedManifestPrefixReceiptMessage {
        protocol_version: AB::Expr::from_u32(ORDERED_MANIFEST_PREFIX_PROTOCOL_VERSION),
        relation_digest: profile.relation_digest.map(AB::Expr::from),
        index_digest: call.prefix.index_digest.map(AB::Expr::from),
        schedule_digest: profile.schedule_digest.map(AB::Expr::from),
        source_order_digest: local.source_order_digest.map(Into::into),
        call_index: AB::Expr::from_usize(call.call_index),
        source_start: AB::Expr::from_usize(call.source_start),
        source_count: AB::Expr::from_usize(call.fresh_count),
        expected_active_child_counts: call.expected_active_child_counts.map(AB::Expr::from_u8),
        l_skip: AB::Expr::from_u32(call.prefix.l_skip),
        log_message_len: AB::Expr::from_u32(call.prefix.log_message_len),
        log_blowup: AB::Expr::from_u32(call.prefix.log_blowup),
        log_codeword_len: AB::Expr::from_u32(call.prefix.log_codeword_len),
        rows_per_leaf: AB::Expr::from_u32(call.prefix.rows_per_leaf),
        trace_prefix_len_lo: AB::Expr::from_u32(call.prefix.trace_prefix_len as u32),
        trace_prefix_len_hi: AB::Expr::from_u32((call.prefix.trace_prefix_len >> 32) as u32),
        active_count_block_start_lo: AB::Expr::from_u32(
            call.prefix.active_count_block_start as u32,
        ),
        active_count_block_start_hi: AB::Expr::from_u32(
            (call.prefix.active_count_block_start >> 32) as u32,
        ),
        active_count_log_height: AB::Expr::from_u32(call.prefix.active_count_log_height),
        base_root: local.base_root.map(Into::into),
        full_root: local.full_root.map(Into::into),
        logup_alpha: local.logup_alpha.map(Into::into),
        logup_beta: local.logup_beta.map(Into::into),
    }
}

fn source_authority_message_expr<AB>(
    profile: &FreshInstanceLinkProfile,
    local: &FreshInstanceProjectionCols<AB::Var>,
    prep: &FreshInstanceProjectionPrepCols<AB::Var>,
) -> OrderedManifestSourceReceiptMessage<AB::Expr>
where
    AB: AirBuilder<F = F>,
    AB::Var: Copy,
{
    const PROGRAM_START: usize = 0;
    const INITIAL_PC: usize = DIGEST_SIZE;
    const FINAL_PC: usize = DIGEST_SIZE + 1;
    const EXIT_CODE: usize = DIGEST_SIZE + 2;
    const IS_TERMINATE: usize = DIGEST_SIZE + 3;
    const INITIAL_ROOT_START: usize = DIGEST_SIZE + 4;
    const FINAL_ROOT_START: usize = 2 * DIGEST_SIZE + 4;

    OrderedManifestSourceReceiptMessage {
        protocol_digest: profile.protocol_digest.map(AB::Expr::from),
        relation_digest: profile.relation_digest.map(AB::Expr::from),
        warp_index_digest: profile.warp_index_digest.map(AB::Expr::from),
        source_index: prep.source_index.into(),
        call_index: prep.call_index.into(),
        fresh_index_in_call: prep.local_source.into(),
        normalized_leaf_start: prep.normalized_leaf_start.into(),
        active_child_count: prep.active_child_count.into(),
        program_commitment: core::array::from_fn(|limb| local.vm_pvs[PROGRAM_START + limb].into()),
        initial_pc: local.vm_pvs[INITIAL_PC].into(),
        initial_root: core::array::from_fn(|limb| local.vm_pvs[INITIAL_ROOT_START + limb].into()),
        final_pc: local.vm_pvs[FINAL_PC].into(),
        final_root: core::array::from_fn(|limb| local.vm_pvs[FINAL_ROOT_START + limb].into()),
        exit_code: local.vm_pvs[EXIT_CODE].into(),
        is_terminate: local.vm_pvs[IS_TERMINATE].into(),
        source_instance_digest: local.source_instance_digest.map(Into::into),
    }
}

const _: () = assert!(D_EF == 4);
const _: () = assert!(FRESH_INSTANCE_VM_PVS_WIDTH == 28);

#[cfg(test)]
mod tests {
    use std::{panic::AssertUnwindSafe, sync::Arc};

    use openvm_stark_backend::{
        air_builders::{
            debug::{check_constraints, check_logup},
            symbolic::get_symbolic_builder,
        },
        interaction::SymbolicInteraction,
        keygen::types::TraceWidth,
        test_utils::test_system_params_small,
        AnyAir,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config as NativeSC;

    use super::*;
    use crate::circuit::finite_warp_v3::{
        derive_ordered_manifest, FiniteWarpV3ManifestCallMessage,
        FiniteWarpV3ManifestReceiptMessage, FiniteWarpV3ReceiptBuses, OrderedManifestCallRecord,
        OrderedManifestDerivedRecord, OrderedManifestReceiptProducer, OrderedManifestRecord,
        OrderedManifestSourceRecord, OrderedManifestVmState,
    };

    fn digest(seed: u32) -> Digest {
        core::array::from_fn(|limb| F::from_u32(seed + limb as u32))
    }

    fn layout() -> FreshInstanceExplicitLayoutProfile {
        FreshInstanceExplicitLayoutProfile {
            alpha_len: 2,
            beta_len: 4 + 6 + FRESH_INSTANCE_VM_PVS_WIDTH,
            log_constraints: 4,
            distinguished_one_explicit_offset: 0,
            alpha_explicit_offset: 1,
            beta_explicit_offset: 2,
            beta_power_explicit_offset: 3,
            beta_power_count: 3,
            public_values_explicit_offset: 6,
            vm_pvs_explicit_offset: 6,
        }
    }

    fn profile() -> FreshInstanceLinkProfile {
        let mut first_counts = [0u8; ORDERED_MANIFEST_PREFIX_SLOTS];
        first_counts[0] = 2;
        first_counts[1] = 1;
        let mut second_counts = [0u8; ORDERED_MANIFEST_PREFIX_SLOTS];
        second_counts[0] = 2;
        let prefix = OrderedManifestPrefixProfile {
            index_digest: digest(60),
            l_skip: 0,
            log_message_len: 8,
            log_blowup: 1,
            log_codeword_len: 9,
            rows_per_leaf: 16,
            trace_prefix_len: 128,
            active_count_block_start: 64,
            active_count_log_height: 2,
        };
        FreshInstanceLinkProfile {
            protocol_digest: digest(10),
            relation_digest: digest(20),
            warp_index_digest: digest(30),
            setup_digest: digest(40),
            schedule_digest: digest(50),
            schedule_end_tidx: 123,
            layout: layout(),
            calls: vec![
                FreshInstanceCallProfile {
                    proof_idx: 0,
                    start_tidx: 123,
                    call_index: 0,
                    input_arity: 2,
                    fresh_count: 2,
                    prior_count: 0,
                    source_start: 0,
                    normalized_leaf_start: 0,
                    expected_active_child_counts: first_counts,
                    prefix,
                },
                FreshInstanceCallProfile {
                    proof_idx: 1,
                    start_tidx: 456,
                    call_index: 1,
                    input_arity: 2,
                    fresh_count: 1,
                    prior_count: 1,
                    source_start: 2,
                    normalized_leaf_start: 3,
                    expected_active_child_counts: second_counts,
                    prefix,
                },
            ],
        }
    }

    fn claim(seed: u32, alpha: EF, beta: EF) -> FreshInstanceClaimRecord {
        let layout = layout();
        let mut explicit = vec![EF::ZERO; layout.explicit_len().unwrap()];
        explicit[0] = EF::ONE;
        explicit[1] = alpha;
        explicit[2] = beta;
        explicit[3] = EF::ONE;
        explicit[4] = beta;
        explicit[5] = beta * beta;
        for (index, value) in explicit[layout.public_values_explicit_offset..]
            .iter_mut()
            .enumerate()
        {
            *value = EF::from(F::from_u32(seed + index as u32));
        }
        let mut beta_claim = vec![EF::from_u32(99); layout.log_constraints];
        beta_claim.extend(explicit);
        FreshInstanceClaimRecord {
            alpha: vec![EF::from_u32(7), EF::from_u32(8)],
            beta: beta_claim,
            mu: EF::from_u32(11),
            eta: EF::ZERO,
        }
    }

    fn record() -> FreshInstanceLinkRecord {
        let first_alpha = EF::from_u32(13);
        let first_beta = EF::from_u32(17);
        let second_alpha = EF::from_u32(19);
        let second_beta = EF::from_u32(23);
        FreshInstanceLinkRecord {
            calls: vec![
                FreshInstanceCallRecord {
                    source_order_digest: digest(70),
                    base_root: digest(80),
                    full_root: digest(90),
                    logup_alpha: first_alpha,
                    logup_beta: first_beta,
                    fresh_claims: vec![
                        claim(100, first_alpha, first_beta),
                        claim(200, first_alpha, first_beta),
                    ],
                },
                FreshInstanceCallRecord {
                    source_order_digest: digest(170),
                    base_root: digest(180),
                    full_root: digest(190),
                    logup_alpha: second_alpha,
                    logup_beta: second_beta,
                    fresh_claims: vec![claim(300, second_alpha, second_beta)],
                },
            ],
        }
    }

    fn set_claim_vm_pvs(
        claim: &mut FreshInstanceClaimRecord,
        program_commitment: Digest,
        initial_pc: F,
        final_pc: F,
        exit_code: F,
        is_terminate: F,
        initial_root: Digest,
        final_root: Digest,
    ) {
        let layout = layout();
        let beta_start = layout.log_constraints + layout.vm_pvs_explicit_offset;
        let mut vm_pvs = [F::ZERO; FRESH_INSTANCE_VM_PVS_WIDTH];
        vm_pvs[..DIGEST_SIZE].copy_from_slice(&program_commitment);
        vm_pvs[DIGEST_SIZE] = initial_pc;
        vm_pvs[DIGEST_SIZE + 1] = final_pc;
        vm_pvs[DIGEST_SIZE + 2] = exit_code;
        vm_pvs[DIGEST_SIZE + 3] = is_terminate;
        vm_pvs[DIGEST_SIZE + 4..2 * DIGEST_SIZE + 4].copy_from_slice(&initial_root);
        vm_pvs[2 * DIGEST_SIZE + 4..3 * DIGEST_SIZE + 4].copy_from_slice(&final_root);
        for (target, value) in claim.beta[beta_start..beta_start + vm_pvs.len()]
            .iter_mut()
            .zip(vm_pvs)
        {
            *target = EF::from(value);
        }
    }

    fn linked_records() -> (
        FreshInstanceLinkProfile,
        FreshInstanceLinkRecord,
        OrderedManifestProfile,
        OrderedManifestDerivedRecord,
    ) {
        let mut fresh_profile = profile();
        let mut fresh_record = record();
        let program = digest(1_100);
        let roots = [digest(1_200), digest(1_220), digest(1_240), digest(1_260)];
        let source_count = fresh_profile.total_fresh();
        let mut source_index = 0usize;
        for call in &mut fresh_record.calls {
            for claim in &mut call.fresh_claims {
                let is_last = source_index + 1 == source_count;
                set_claim_vm_pvs(
                    claim,
                    program,
                    F::from_usize(source_index),
                    F::from_usize(source_index + 1),
                    if is_last { F::ZERO } else { F::from_u32(2) },
                    F::from_bool(is_last),
                    roots[source_index],
                    roots[source_index + 1],
                );
                source_index += 1;
            }
        }

        let manifest_profile = OrderedManifestProfile {
            protocol_digest: fresh_profile.protocol_digest,
            relation_digest: fresh_profile.relation_digest,
            warp_index_digest: fresh_profile.warp_index_digest,
            verifier_component_digest: digest(1_300),
            prefix: fresh_profile.calls[0].prefix,
            expected_normalized_leaf_count: fresh_profile
                .total_normalized_leaves()
                .expect("test normalized total") as u32,
            max_active_children_per_source: 2,
            suspend_exit_code: 2,
        };
        let calls = fresh_profile
            .calls
            .iter()
            .zip(&fresh_record.calls)
            .map(|(profile, record)| OrderedManifestCallRecord {
                call_index: profile.call_index as u32,
                input_arity: profile.input_arity as u32,
                source_start: profile.source_start as u32,
                source_count: profile.fresh_count as u32,
                base_root: record.base_root,
                full_root: record.full_root,
                logup_alpha: record.logup_alpha,
                logup_beta: record.logup_beta,
            })
            .collect::<Vec<_>>();
        let mut sources = Vec::with_capacity(fresh_profile.total_fresh());
        for (call, record) in fresh_profile.calls.iter().zip(&fresh_record.calls) {
            for (local_source, claim) in record.fresh_claims.iter().enumerate() {
                let explicit = explicit_assignment(&fresh_profile.layout, claim)
                    .expect("linked test explicit assignment");
                let vm_pvs = extract_vm_pvs(&fresh_profile.layout, &explicit)
                    .expect("linked test base VmPvs");
                sources.push(OrderedManifestSourceRecord {
                    source_index: (call.source_start + local_source) as u32,
                    call_index: call.call_index as u32,
                    fresh_index_in_call: local_source as u32,
                    normalized_leaf_start: call
                        .normalized_start_for(local_source)
                        .expect("linked test normalized start")
                        as u32,
                    active_child_count: call.expected_active_child_counts[local_source],
                    program_commitment: vm_pvs[..DIGEST_SIZE].try_into().expect("program digest"),
                    initial_state: OrderedManifestVmState {
                        pc: vm_pvs[DIGEST_SIZE],
                        memory_root: vm_pvs[DIGEST_SIZE + 4..2 * DIGEST_SIZE + 4]
                            .try_into()
                            .expect("initial root"),
                    },
                    final_state: OrderedManifestVmState {
                        pc: vm_pvs[DIGEST_SIZE + 1],
                        memory_root: vm_pvs[2 * DIGEST_SIZE + 4..3 * DIGEST_SIZE + 4]
                            .try_into()
                            .expect("final root"),
                    },
                    exit_code: vm_pvs[DIGEST_SIZE + 2],
                    is_terminate: vm_pvs[DIGEST_SIZE + 3],
                    source_instance_digest: fresh_instance_source_digest(&explicit),
                });
            }
        }
        let manifest_record = OrderedManifestRecord {
            calls,
            sources,
            final_accumulator_digest: digest(1_400),
        };
        let derived = derive_ordered_manifest(&manifest_profile, &manifest_record)
            .expect("linked manifest record");
        fresh_profile.schedule_digest = derived.schedule_digest;
        for (record, source_order_digest) in fresh_record
            .calls
            .iter_mut()
            .zip(&derived.source_order_digests)
        {
            record.source_order_digest = *source_order_digest;
        }
        fresh_profile
            .validate_ordered_manifest_profile(&manifest_profile)
            .expect("linked profiles");
        (fresh_profile, fresh_record, manifest_profile, derived)
    }

    fn buses() -> FreshInstanceLinkBuses {
        let mut indices = BusIndexManager::new();
        let claim_value = NativeClaimValueBus::new(indices.new_bus_idx());
        let input_slot = NativeInputSlotLayoutBus::new(indices.new_bus_idx());
        let exact_schedule = NativeExactFiniteVaccScheduleBus::new(indices.new_bus_idx());
        let exact_call_protocol = NativeExactFiniteVaccCallProtocolBus::new(indices.new_bus_idx());
        let vacc_root = NativeStandardVaccRootBus::new(indices.new_bus_idx());
        let ordered_manifest = OrderedManifestBuses::new(&mut indices);
        FreshInstanceLinkBuses::new(
            &mut indices,
            claim_value,
            input_slot,
            exact_schedule,
            exact_call_protocol,
            vacc_root,
            ordered_manifest,
        )
    }

    #[derive(Clone, Debug)]
    struct TestManifestBoundary {
        profile: OrderedManifestProfile,
        derived: OrderedManifestDerivedRecord,
        wrapper: FiniteWarpV3ReceiptBuses,
    }

    #[derive(Clone, Debug)]
    struct TestExactAuthorityAir {
        profile: FreshInstanceLinkProfile,
        record: FreshInstanceLinkRecord,
        buses: FreshInstanceLinkBuses,
        manifest: Option<TestManifestBoundary>,
    }

    impl BaseAir<F> for TestExactAuthorityAir {
        fn width(&self) -> usize {
            1
        }
    }

    impl BaseAirWithPublicValues<F> for TestExactAuthorityAir {}
    impl PartitionedBaseAir<F> for TestExactAuthorityAir {}

    impl<AB> Air<AB> for TestExactAuthorityAir
    where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        fn eval(&self, builder: &mut AB) {
            let main = builder.main();
            let local = main.row_slice(0).expect("test exact-authority row")[0];
            let next = main.row_slice(1).expect("test exact-authority padding")[0];
            let enabled = AB::Expr::from(local);
            builder.assert_bool(local);
            builder.when_first_row().assert_one(local);
            builder.when_last_row().assert_zero(local);
            builder.when_transition().assert_eq(local - next, local);

            self.buses.exact_schedule.add_key_with_lookups(
                builder,
                NativeExactFiniteVaccScheduleMessage {
                    proof_idx: AB::Expr::ZERO,
                    end_tidx: AB::Expr::from_usize(self.profile.schedule_end_tidx),
                    call_count: AB::Expr::from_usize(self.profile.calls.len()),
                    total_fresh: AB::Expr::from_usize(self.profile.total_fresh()),
                    relation_digest: self.profile.relation_digest.map(AB::Expr::from),
                    index_digest: self.profile.warp_index_digest.map(AB::Expr::from),
                    setup_digest: self.profile.setup_digest.map(AB::Expr::from),
                    schedule_digest: self.profile.schedule_digest.map(AB::Expr::from),
                },
                enabled.clone(),
            );

            for (call_index, (call, record)) in self
                .profile
                .calls
                .iter()
                .zip(&self.record.calls)
                .enumerate()
            {
                let proof_idx = AB::Expr::from_usize(call.proof_idx);
                self.buses.exact_call_protocol.add_key_with_lookups(
                    builder,
                    NativeExactFiniteVaccCallProtocolMessage {
                        proof_idx: proof_idx.clone(),
                        start_tidx: AB::Expr::from_usize(call.start_tidx),
                        call_index: AB::Expr::from_usize(call.call_index),
                        input_arity: AB::Expr::from_usize(call.input_arity),
                        fresh_count: AB::Expr::from_usize(call.fresh_count),
                        prior_count: AB::Expr::from_usize(call.prior_count),
                        relation_digest: self.profile.relation_digest.map(AB::Expr::from),
                        index_digest: self.profile.warp_index_digest.map(AB::Expr::from),
                        setup_digest: self.profile.setup_digest.map(AB::Expr::from),
                        schedule_digest: self.profile.schedule_digest.map(AB::Expr::from),
                    },
                    enabled.clone(),
                );
                self.buses.vacc_root.add_key_with_lookups(
                    builder,
                    NativeStandardVaccRootMessage {
                        proof_idx: proof_idx.clone(),
                        kind: AB::Expr::ZERO,
                        root: record.full_root.map(AB::Expr::from),
                    },
                    enabled.clone(),
                );
                let prefix = test_prefix_message::<AB>(&self.profile, call, record);
                if self.manifest.is_some() {
                    self.buses
                        .ordered_manifest
                        .prefix_authority
                        .add_key_with_lookups(builder, prefix, enabled.clone());
                } else {
                    self.buses
                        .ordered_manifest
                        .prefix_binding
                        .add_key_with_lookups(builder, prefix, enabled.clone());
                }

                for (local_source, claim) in record.fresh_claims.iter().enumerate() {
                    self.buses.input_slot.add_key_with_lookups(
                        builder,
                        NativeInputSlotLayoutMessage {
                            variant: AB::Expr::from_usize(call.slot_variant()),
                            source: AB::Expr::from_usize(local_source),
                            kind: [AB::Expr::ONE, AB::Expr::ZERO, AB::Expr::ZERO],
                        },
                        enabled.clone(),
                    );
                    for (section, values) in [
                        (CLAIM_SECTION_ALPHA, claim.alpha.as_slice()),
                        (CLAIM_SECTION_BETA, claim.beta.as_slice()),
                        (CLAIM_SECTION_MU, core::slice::from_ref(&claim.mu)),
                        (CLAIM_SECTION_ETA, core::slice::from_ref(&claim.eta)),
                    ] {
                        for (coordinate, value) in values.iter().enumerate() {
                            self.buses.claim_value.send(
                                builder,
                                NativeClaimValueMessage {
                                    proof_idx: proof_idx.clone(),
                                    source: AB::Expr::from_usize(local_source),
                                    section: AB::Expr::from_usize(section),
                                    coordinate: AB::Expr::from_usize(coordinate),
                                    value: ef_limbs(*value).map(AB::Expr::from),
                                },
                                enabled.clone(),
                            );
                        }
                    }

                    let explicit = explicit_assignment(&self.profile.layout, claim)
                        .expect("test claim has validated explicit assignment");
                    let vm_pvs = extract_vm_pvs(&self.profile.layout, &explicit)
                        .expect("test claim has base VmPvs");
                    if self.manifest.is_none() {
                        self.buses.ordered_manifest.source_authority.lookup_key(
                            builder,
                            test_source_message::<AB>(
                                &self.profile,
                                call,
                                local_source,
                                vm_pvs,
                                fresh_instance_source_digest(&explicit),
                            ),
                            enabled.clone(),
                        );
                    }
                }
                assert_eq!(call_index, call.call_index);
            }

            if let Some(manifest) = &self.manifest {
                consume_manifest_boundary::<AB>(builder, manifest, enabled);
            }
        }
    }

    fn test_prefix_message<AB: AirBuilder<F = F>>(
        profile: &FreshInstanceLinkProfile,
        call: &FreshInstanceCallProfile,
        record: &FreshInstanceCallRecord,
    ) -> OrderedManifestPrefixReceiptMessage<AB::Expr> {
        OrderedManifestPrefixReceiptMessage {
            protocol_version: AB::Expr::from_u32(ORDERED_MANIFEST_PREFIX_PROTOCOL_VERSION),
            relation_digest: profile.relation_digest.map(AB::Expr::from),
            index_digest: call.prefix.index_digest.map(AB::Expr::from),
            schedule_digest: profile.schedule_digest.map(AB::Expr::from),
            source_order_digest: record.source_order_digest.map(AB::Expr::from),
            call_index: AB::Expr::from_usize(call.call_index),
            source_start: AB::Expr::from_usize(call.source_start),
            source_count: AB::Expr::from_usize(call.fresh_count),
            expected_active_child_counts: call.expected_active_child_counts.map(AB::Expr::from_u8),
            l_skip: AB::Expr::from_u32(call.prefix.l_skip),
            log_message_len: AB::Expr::from_u32(call.prefix.log_message_len),
            log_blowup: AB::Expr::from_u32(call.prefix.log_blowup),
            log_codeword_len: AB::Expr::from_u32(call.prefix.log_codeword_len),
            rows_per_leaf: AB::Expr::from_u32(call.prefix.rows_per_leaf),
            trace_prefix_len_lo: AB::Expr::from_u32(call.prefix.trace_prefix_len as u32),
            trace_prefix_len_hi: AB::Expr::from_u32((call.prefix.trace_prefix_len >> 32) as u32),
            active_count_block_start_lo: AB::Expr::from_u32(
                call.prefix.active_count_block_start as u32,
            ),
            active_count_block_start_hi: AB::Expr::from_u32(
                (call.prefix.active_count_block_start >> 32) as u32,
            ),
            active_count_log_height: AB::Expr::from_u32(call.prefix.active_count_log_height),
            base_root: record.base_root.map(AB::Expr::from),
            full_root: record.full_root.map(AB::Expr::from),
            logup_alpha: ef_limbs(record.logup_alpha).map(AB::Expr::from),
            logup_beta: ef_limbs(record.logup_beta).map(AB::Expr::from),
        }
    }

    fn test_source_message<AB: AirBuilder<F = F>>(
        profile: &FreshInstanceLinkProfile,
        call: &FreshInstanceCallProfile,
        local_source: usize,
        vm_pvs: [F; FRESH_INSTANCE_VM_PVS_WIDTH],
        source_instance_digest: Digest,
    ) -> OrderedManifestSourceReceiptMessage<AB::Expr> {
        const INITIAL_PC: usize = DIGEST_SIZE;
        const FINAL_PC: usize = DIGEST_SIZE + 1;
        const EXIT_CODE: usize = DIGEST_SIZE + 2;
        const IS_TERMINATE: usize = DIGEST_SIZE + 3;
        const INITIAL_ROOT_START: usize = DIGEST_SIZE + 4;
        const FINAL_ROOT_START: usize = 2 * DIGEST_SIZE + 4;
        OrderedManifestSourceReceiptMessage {
            protocol_digest: profile.protocol_digest.map(AB::Expr::from),
            relation_digest: profile.relation_digest.map(AB::Expr::from),
            warp_index_digest: profile.warp_index_digest.map(AB::Expr::from),
            source_index: AB::Expr::from_usize(call.source_start + local_source),
            call_index: AB::Expr::from_usize(call.call_index),
            fresh_index_in_call: AB::Expr::from_usize(local_source),
            normalized_leaf_start: AB::Expr::from_usize(
                call.normalized_start_for(local_source)
                    .expect("validated normalized source start"),
            ),
            active_child_count: AB::Expr::from_u8(call.expected_active_child_counts[local_source]),
            program_commitment: core::array::from_fn(|limb| AB::Expr::from(vm_pvs[limb])),
            initial_pc: AB::Expr::from(vm_pvs[INITIAL_PC]),
            initial_root: core::array::from_fn(|limb| {
                AB::Expr::from(vm_pvs[INITIAL_ROOT_START + limb])
            }),
            final_pc: AB::Expr::from(vm_pvs[FINAL_PC]),
            final_root: core::array::from_fn(|limb| {
                AB::Expr::from(vm_pvs[FINAL_ROOT_START + limb])
            }),
            exit_code: AB::Expr::from(vm_pvs[EXIT_CODE]),
            is_terminate: AB::Expr::from(vm_pvs[IS_TERMINATE]),
            source_instance_digest: source_instance_digest.map(AB::Expr::from),
        }
    }

    fn consume_manifest_boundary<AB>(
        builder: &mut AB,
        boundary: &TestManifestBoundary,
        enabled: AB::Expr,
    ) where
        AB: AirBuilder<F = F> + InteractionBuilder,
        AB::Var: Copy,
    {
        let first = boundary
            .derived
            .record
            .sources
            .first()
            .expect("test manifest source");
        let last = boundary
            .derived
            .record
            .sources
            .last()
            .expect("test manifest source");
        boundary.wrapper.manifest.lookup_key(
            builder,
            FiniteWarpV3ManifestReceiptMessage {
                protocol_digest: boundary.profile.protocol_digest.map(AB::Expr::from),
                relation_digest: boundary.profile.relation_digest.map(AB::Expr::from),
                warp_index_digest: boundary.profile.warp_index_digest.map(AB::Expr::from),
                verifier_component_digest: boundary
                    .profile
                    .verifier_component_digest
                    .map(AB::Expr::from),
                schedule_digest: boundary.derived.schedule_digest.map(AB::Expr::from),
                manifest_digest: boundary.derived.manifest_digest.map(AB::Expr::from),
                source_count: AB::Expr::from_usize(boundary.derived.record.sources.len()),
                call_count: AB::Expr::from_usize(boundary.derived.record.calls.len()),
                program_commitment: first.program_commitment.map(AB::Expr::from),
                initial_pc: AB::Expr::from(first.initial_state.pc),
                initial_root: first.initial_state.memory_root.map(AB::Expr::from),
                final_pc: AB::Expr::from(last.final_state.pc),
                final_root: last.final_state.memory_root.map(AB::Expr::from),
                final_accumulator_digest: boundary
                    .derived
                    .record
                    .final_accumulator_digest
                    .map(AB::Expr::from),
                calls: core::array::from_fn(|index| {
                    boundary.derived.record.calls.get(index).map_or_else(
                        || FiniteWarpV3ManifestCallMessage {
                            active: AB::Expr::ZERO,
                            source_start: AB::Expr::ZERO,
                            source_count: AB::Expr::ZERO,
                            input_arity: AB::Expr::ZERO,
                            fresh_stacked_root: [AB::Expr::ZERO; DIGEST_SIZE],
                        },
                        |call| FiniteWarpV3ManifestCallMessage {
                            active: AB::Expr::ONE,
                            source_start: AB::Expr::from_u32(call.source_start),
                            source_count: AB::Expr::from_u32(call.source_count),
                            input_arity: AB::Expr::from_u32(call.input_arity),
                            fresh_stacked_root: call.full_root.map(AB::Expr::from),
                        },
                    )
                }),
            },
            enabled.clone(),
        );
    }

    fn symbolic_interactions(air: &dyn AnyAir<NativeSC>) -> Vec<SymbolicInteraction<F>> {
        let preprocessed = BaseAir::<F>::preprocessed_trace(air).map(|trace| trace.width());
        get_symbolic_builder(
            air,
            &TraceWidth {
                preprocessed,
                cached_mains: air.cached_main_widths(),
                common_main: air.common_main_width(),
            },
        )
        .constraints()
        .interactions
    }

    #[derive(Clone)]
    struct TestComposition {
        airs: Vec<AirRef<NativeSC>>,
        traces: Vec<RowMajorMatrix<F>>,
    }

    fn test_composition() -> TestComposition {
        let profile = profile();
        let record = record();
        let mut indices = BusIndexManager::new();
        let shared = BusInventory::new(&mut indices);
        let claim_value = NativeClaimValueBus::new(indices.new_bus_idx());
        let input_slot = NativeInputSlotLayoutBus::new(indices.new_bus_idx());
        let exact_schedule = NativeExactFiniteVaccScheduleBus::new(indices.new_bus_idx());
        let exact_call_protocol = NativeExactFiniteVaccCallProtocolBus::new(indices.new_bus_idx());
        let vacc_root = NativeStandardVaccRootBus::new(indices.new_bus_idx());
        let ordered_manifest = OrderedManifestBuses::new(&mut indices);
        let buses = FreshInstanceLinkBuses::new(
            &mut indices,
            claim_value,
            input_slot,
            exact_schedule,
            exact_call_protocol,
            vacc_root,
            ordered_manifest,
        );
        let component = FreshInstanceLinkComponent::new(
            profile.clone(),
            buses,
            &shared,
            test_system_params_small(0, 8, 4),
        )
        .unwrap();
        let generated = component.generate_traces(&record).unwrap();
        let mut airs = component.airs::<NativeSC>();
        let authority = TestExactAuthorityAir {
            profile,
            record,
            buses,
            manifest: None,
        };
        airs.push(Arc::new(authority));
        let mut traces = generated.into_ordered_traces();
        traces.push(RowMajorMatrix::new(vec![F::ONE, F::ZERO], 1));
        assert_eq!(airs.len(), traces.len());
        TestComposition { airs, traces }
    }

    fn linked_manifest_composition() -> (TestComposition, usize, usize) {
        let (fresh_profile, fresh_record, manifest_profile, derived) = linked_records();
        let mut indices = BusIndexManager::new();
        let shared = BusInventory::new(&mut indices);
        let wrapper = FiniteWarpV3ReceiptBuses::new(indices.next_bus_idx());
        let mut indices = BusIndexManager::from_next_bus_idx(wrapper.next_bus_idx());
        let claim_value = NativeClaimValueBus::new(indices.new_bus_idx());
        let input_slot = NativeInputSlotLayoutBus::new(indices.new_bus_idx());
        let exact_schedule = NativeExactFiniteVaccScheduleBus::new(indices.new_bus_idx());
        let exact_call_protocol = NativeExactFiniteVaccCallProtocolBus::new(indices.new_bus_idx());
        let vacc_root = NativeStandardVaccRootBus::new(indices.new_bus_idx());
        let ordered_manifest = OrderedManifestBuses::new(&mut indices);
        let buses = FreshInstanceLinkBuses::new(
            &mut indices,
            claim_value,
            input_slot,
            exact_schedule,
            exact_call_protocol,
            vacc_root,
            ordered_manifest,
        );
        let params = test_system_params_small(0, 8, 4);
        let fresh_component =
            FreshInstanceLinkComponent::new(fresh_profile.clone(), buses, &shared, params.clone())
                .unwrap();
        let manifest_component = OrderedManifestReceiptProducer::new(
            manifest_profile.clone(),
            wrapper,
            ordered_manifest,
            &shared,
            params,
        )
        .unwrap();
        let fresh_traces = fresh_component.generate_traces(&fresh_record).unwrap();
        let manifest_traces = manifest_component
            .generate_traces(&derived.record)
            .unwrap()
            .into_ordered_traces();
        assert_eq!(manifest_traces.derived, derived);
        assert_eq!(
            manifest_component.air_order(),
            [
                crate::circuit::finite_warp_v3::OrderedManifestAirKind::Header,
                crate::circuit::finite_warp_v3::OrderedManifestAirKind::Calls,
                crate::circuit::finite_warp_v3::OrderedManifestAirKind::Sources,
                crate::circuit::finite_warp_v3::OrderedManifestAirKind::Transcript,
                crate::circuit::finite_warp_v3::OrderedManifestAirKind::Poseidon2,
            ]
        );

        let fresh_air_count = fresh_component.air_order().len();
        let mut airs = fresh_component.airs::<NativeSC>();
        airs.extend(manifest_component.airs::<NativeSC>());
        airs.push(Arc::new(TestExactAuthorityAir {
            profile: fresh_profile,
            record: fresh_record,
            buses,
            manifest: Some(TestManifestBoundary {
                profile: manifest_profile,
                derived,
                wrapper,
            }),
        }));

        let mut traces = fresh_traces.into_ordered_traces();
        let manifest_source_index = traces.len() + 2;
        traces.extend(manifest_traces.ordered_traces);
        traces.push(RowMajorMatrix::new(vec![F::ONE, F::ZERO], 1));
        assert_eq!(fresh_air_count + 2, manifest_source_index);
        assert_eq!(airs.len(), traces.len());
        (
            TestComposition { airs, traces },
            fresh_air_count,
            manifest_source_index,
        )
    }

    fn check_test_composition(composition: &TestComposition) {
        let public_values = vec![Vec::<F>::new(); composition.airs.len()];
        for ((air, trace), pvs) in composition
            .airs
            .iter()
            .zip(&composition.traces)
            .zip(&public_values)
        {
            let preprocessed = BaseAir::<F>::preprocessed_trace(air.as_ref());
            check_constraints::<_, NativeSC>(
                air.as_ref(),
                &air.name(),
                &preprocessed.as_ref().map(RowMajorMatrix::as_view),
                &[trace.as_view()],
                pvs,
            );
        }
        let preprocessed_owned = composition
            .airs
            .iter()
            .map(|air| BaseAir::<F>::preprocessed_trace(air.as_ref()))
            .collect::<Vec<_>>();
        let preprocessed = preprocessed_owned
            .iter()
            .map(|matrix| matrix.as_ref().map(RowMajorMatrix::as_view))
            .collect::<Vec<_>>();
        let interactions = composition
            .airs
            .iter()
            .map(|air| symbolic_interactions(air.as_ref()))
            .collect::<Vec<_>>();
        let views = composition
            .traces
            .iter()
            .map(|matrix| vec![matrix.as_view()])
            .collect::<Vec<_>>();
        check_logup(
            &composition
                .airs
                .iter()
                .map(|air| air.name())
                .collect::<Vec<_>>(),
            &interactions,
            &preprocessed,
            &views,
            &public_values,
        );
    }

    fn assert_composition_rejects(composition: &TestComposition, reason: &str) {
        let rejected = std::panic::catch_unwind(AssertUnwindSafe(|| {
            check_test_composition(composition);
        }));
        assert!(
            rejected.is_err(),
            "mutated composition was accepted: {reason}"
        );
    }

    #[test]
    fn exact_fresh_claim_projection_and_source_digest_interactions_balance() {
        check_test_composition(&test_composition());
    }

    #[test]
    fn fresh_source_authority_is_consumed_by_the_real_ordered_manifest() {
        let (canonical, _, manifest_source_index) = linked_manifest_composition();
        check_test_composition(&canonical);

        let mut reordered = canonical.clone();
        let width = reordered.traces[manifest_source_index].width();
        for column in 0..width {
            reordered.traces[manifest_source_index]
                .values
                .swap(column, width + column);
        }
        assert_composition_rejects(&reordered, "reordered manifest source authority");
    }

    #[test]
    fn consumer_multiplicities_and_context_order_are_explicit() {
        let requirements = FreshInstanceLinkConsumerRequirements::EXACT;
        assert_eq!(requirements.ordinary_exact_vacc_fresh_claim_consumers, 1);
        assert_eq!(requirements.extra_fresh_claim_consumers, 1);
        assert_eq!(requirements.fresh_claim_consumer_count_with_exact_vacc(), 2);
        assert_eq!(requirements.ordinary_exact_schedule_lookups, 1);
        assert_eq!(requirements.extra_exact_schedule_lookups, 1);
        assert_eq!(requirements.exact_schedule_producer_count(), 2);
        assert_eq!(
            requirements.ordinary_exact_call_protocol_lookups_per_call,
            1
        );
        assert_eq!(requirements.extra_exact_call_protocol_lookups_per_call, 1);
        assert_eq!(
            requirements.exact_call_protocol_producer_count_per_call(),
            2
        );
        assert_eq!(requirements.ordinary_exact_fresh_root_lookups_per_call, 1);
        assert_eq!(requirements.extra_fresh_root_lookups_per_call, 1);
        assert_eq!(requirements.fresh_root_producer_count_per_call(), 2);
        assert_eq!(requirements.extra_fresh_slot_lookups_per_source, 1);
        assert_eq!(requirements.prefix_authority_lookups_per_call, 1);
        assert_eq!(requirements.prefix_binding_keys_per_call, 1);
        assert_eq!(requirements.prefix_binding_lookups_per_call, 1);
        assert_eq!(requirements.source_authority_keys_per_source, 1);
        assert_eq!(requirements.source_authority_lookups_per_source, 1);

        let profile = profile();
        let record = record();
        let mut indices = BusIndexManager::new();
        let shared = BusInventory::new(&mut indices);
        let claim_value = NativeClaimValueBus::new(indices.new_bus_idx());
        let input_slot = NativeInputSlotLayoutBus::new(indices.new_bus_idx());
        let exact_schedule = NativeExactFiniteVaccScheduleBus::new(indices.new_bus_idx());
        let exact_call_protocol = NativeExactFiniteVaccCallProtocolBus::new(indices.new_bus_idx());
        let vacc_root = NativeStandardVaccRootBus::new(indices.new_bus_idx());
        let ordered_manifest = OrderedManifestBuses::new(&mut indices);
        let buses = FreshInstanceLinkBuses::new(
            &mut indices,
            claim_value,
            input_slot,
            exact_schedule,
            exact_call_protocol,
            vacc_root,
            ordered_manifest,
        );
        let component = FreshInstanceLinkComponent::new(
            profile,
            buses,
            &shared,
            test_system_params_small(0, 8, 4),
        )
        .unwrap();
        assert_eq!(
            component.air_order(),
            vec![
                FreshInstanceLinkAirKind::CallLink { call_index: 0 },
                FreshInstanceLinkAirKind::CallLink { call_index: 1 },
                FreshInstanceLinkAirKind::Projection { call_index: 0 },
                FreshInstanceLinkAirKind::Projection { call_index: 1 },
                FreshInstanceLinkAirKind::DigestTranscript,
                FreshInstanceLinkAirKind::Poseidon2,
            ]
        );
        let traces = component.generate_traces(&record).unwrap();
        assert_eq!(
            traces.into_ordered_traces().len(),
            component.air_order().len()
        );
        let shared_traces = component
            .generate_traces(&record)
            .unwrap()
            .into_shared_poseidon_traces();
        assert_eq!(
            shared_traces.ordered_traces_without_poseidon.len(),
            component.air_order().len() - 1
        );
        assert_eq!(
            component.airs_without_poseidon::<NativeSC>().len(),
            shared_traces.ordered_traces_without_poseidon.len()
        );
        assert!(!shared_traces.permutation_inputs.is_empty());
    }

    #[test]
    fn ordered_manifest_profile_must_match_every_source_link_constant() {
        let fresh = profile();
        let manifest = OrderedManifestProfile {
            protocol_digest: fresh.protocol_digest,
            relation_digest: fresh.relation_digest,
            warp_index_digest: fresh.warp_index_digest,
            verifier_component_digest: digest(1_000),
            prefix: fresh.calls[0].prefix,
            expected_normalized_leaf_count: fresh.total_normalized_leaves().unwrap() as u32,
            max_active_children_per_source: 2,
            suspend_exit_code: 2,
        };
        fresh.validate_ordered_manifest_profile(&manifest).unwrap();

        let mut wrong_relation = manifest.clone();
        wrong_relation.relation_digest[0] += F::ONE;
        assert!(fresh
            .validate_ordered_manifest_profile(&wrong_relation)
            .is_err());

        let mut wrong_prefix = manifest.clone();
        wrong_prefix.prefix.index_digest[0] += F::ONE;
        assert!(fresh
            .validate_ordered_manifest_profile(&wrong_prefix)
            .is_err());

        let mut shortened = manifest;
        shortened.expected_normalized_leaf_count -= 1;
        assert!(fresh.validate_ordered_manifest_profile(&shortened).is_err());

        let mut inconsistent_calls = fresh;
        inconsistent_calls.calls[1].prefix.index_digest[0] += F::ONE;
        assert!(inconsistent_calls.validate().is_err());
    }

    #[test]
    fn native_source_digest_log_keeps_dynamic_ef4_coordinate_order() {
        for case in 1..=12u32 {
            let explicit = (0..case as usize + 3)
                .map(|coordinate| {
                    let coefficients: [F; D_EF] = core::array::from_fn(|limb| {
                        F::from_u32(case * 101 + coordinate as u32 * 17 + limb as u32 * 29 + 1)
                    });
                    EF::from_basis_coefficients_slice(&coefficients).unwrap()
                })
                .collect::<Vec<_>>();
            let (digest, log) = source_digest_and_log(&explicit);
            let mut expected = vec![
                F::from_u32(FRESH_INSTANCE_SOURCE_DIGEST_TAG),
                F::from_usize(explicit.len()),
                F::from_usize(D_EF),
            ];
            expected.extend(explicit.iter().flat_map(|value| ef_limbs(*value)));
            assert_eq!(&log.values()[..expected.len()], expected);
            assert!(log.samples()[..expected.len()].iter().all(|sample| !sample));
            assert!(log.samples()[expected.len()..].iter().all(|sample| *sample));
            assert_eq!(&log.values()[expected.len()..], digest);

            for coordinate in 0..explicit.len() {
                for limb in 0..D_EF {
                    let mut changed = explicit.clone();
                    let mut coefficients = ef_limbs(changed[coordinate]);
                    coefficients[limb] += F::ONE;
                    changed[coordinate] = EF::from_basis_coefficients_slice(&coefficients).unwrap();
                    assert_ne!(fresh_instance_source_digest(&changed), digest);
                }
            }
            let mut reordered = explicit;
            reordered.swap(0, 1);
            assert_ne!(fresh_instance_source_digest(&reordered), digest);
        }
    }

    #[test]
    fn claim_coordinate_digest_root_order_and_vm_projection_mutations_are_rejected() {
        let canonical = test_composition();
        check_test_composition(&canonical);

        let projection_index = profile().calls.len();
        let projection_width = FreshInstanceProjectionCols::<F>::width();

        let mut changed_claim = canonical.clone();
        let first: &mut FreshInstanceProjectionCols<F> =
            changed_claim.traces[projection_index].values[..projection_width].borrow_mut();
        first.value[0] += F::ONE;
        assert_composition_rejects(&changed_claim, "exact claim coordinate");

        let mut changed_digest = canonical.clone();
        let rows_per_source = layout().claim_rows_per_source().unwrap();
        for row in 0..rows_per_source {
            let start = row * projection_width;
            let cols: &mut FreshInstanceProjectionCols<F> = changed_digest.traces[projection_index]
                .values[start..start + projection_width]
                .borrow_mut();
            cols.source_instance_digest[0] += F::ONE;
        }
        assert_composition_rejects(&changed_digest, "source digest transcript sample");

        let call_width = FreshInstanceCallLinkCols::<F>::width();
        let mut changed_root = canonical.clone();
        let call: &mut FreshInstanceCallLinkCols<F> =
            changed_root.traces[0].values[..call_width].borrow_mut();
        call.full_root[0] += F::ONE;
        assert_composition_rejects(&changed_root, "fresh stacked root");

        let mut changed_order = canonical.clone();
        let call: &mut FreshInstanceCallLinkCols<F> =
            changed_order.traces[0].values[..call_width].borrow_mut();
        call.source_order_digest[0] += F::ONE;
        assert_composition_rejects(&changed_order, "ordered source digest");

        let mut changed_vm_projection = canonical;
        for row in 0..rows_per_source {
            let start = row * projection_width;
            let cols: &mut FreshInstanceProjectionCols<F> = changed_vm_projection.traces
                [projection_index]
                .values[start..start + projection_width]
                .borrow_mut();
            cols.vm_pvs[0] += F::ONE;
        }
        assert_composition_rejects(
            &changed_vm_projection,
            "host-selected VmPvs instead of explicit assignment",
        );
    }

    #[test]
    fn positive_profile_generation_and_digest_are_canonical() {
        let profile = profile();
        let record = record();
        profile.validate().unwrap();
        validate_record(&profile, &record).unwrap();
        assert_eq!(
            profile
                .consumer_requirements()
                .fresh_claim_consumer_count_with_exact_vacc(),
            2
        );
        let link = FreshInstanceCallLinkAir::new(profile.clone(), 0, buses()).unwrap();
        let link_trace = link.generate_trace(&record.calls[0]).unwrap();
        let projection = FreshInstanceProjectionAir::new(profile.clone(), 0, buses()).unwrap();
        let (projection_trace, logs) = projection.generate_trace(&record.calls[0]).unwrap();
        assert_eq!(link_trace.width(), link.width());
        assert_eq!(projection_trace.width(), projection.width());
        assert_eq!(logs.len(), profile.calls[0].fresh_count);
        let explicit =
            explicit_assignment(&profile.layout, &record.calls[0].fresh_claims[0]).unwrap();
        let digest = fresh_instance_source_digest(&explicit);
        let mut changed = explicit.clone();
        changed[profile.layout.vm_pvs_explicit_offset] += EF::ONE;
        assert_ne!(digest, fresh_instance_source_digest(&changed));
    }

    #[test]
    fn fresh_projection_stays_within_internal_recursive_degree_four() {
        let projection = FreshInstanceProjectionAir::new(profile(), 0, buses()).unwrap();
        let air: &dyn AnyAir<NativeSC> = &projection;
        let preprocessed = BaseAir::<F>::preprocessed_trace(air).map(|trace| trace.width());
        let symbolic = get_symbolic_builder(
            air,
            &TraceWidth {
                preprocessed,
                cached_mains: air.cached_main_widths(),
                common_main: air.common_main_width(),
            },
        )
        .constraints();
        assert!(
            symbolic.max_constraint_degree() <= 4,
            "FreshInstanceProjectionAir degree {} exceeds recursive parameters",
            symbolic.max_constraint_degree()
        );
    }

    #[test]
    fn production_three_child_sources_are_valid_but_out_of_capacity_counts_are_not() {
        let mut three = profile();
        three.calls[0].expected_active_child_counts[0] = 3;
        three.calls[1].normalized_leaf_start = 4;
        three.validate().expect("three-child fixed PESAT source");
        assert_eq!(three.total_normalized_leaves(), Some(6));

        three.calls[0].expected_active_child_counts[0] =
            FRESH_INSTANCE_MAX_ACTIVE_CHILDREN_PER_SOURCE + 1;
        assert_eq!(
            three.validate(),
            Err(FreshInstanceLinkError::InvalidProfile(
                "active-child schedule"
            ))
        );

        // The source axis of a stacked WARP call is independent from the row
        // axis of each source's active-count column.  Sixty-four fresh sources may
        // therefore share a call even when that column has height four.
        let mut wide = profile();
        wide.calls.truncate(1);
        wide.calls[0].input_arity = ORDERED_MANIFEST_PREFIX_SLOTS;
        wide.calls[0].fresh_count = ORDERED_MANIFEST_PREFIX_SLOTS;
        wide.calls[0].expected_active_child_counts.fill(1);
        wide.validate().expect("independent stacked source axis");
        assert_eq!(
            wide.total_normalized_leaves(),
            Some(ORDERED_MANIFEST_PREFIX_SLOTS)
        );
    }

    #[test]
    fn coordinate_alpha_beta_and_digest_mutations_are_rejected_or_change_binding() {
        let profile = profile();
        let mut bad_coordinate = record();
        let explicit =
            explicit_assignment(&profile.layout, &bad_coordinate.calls[0].fresh_claims[0]).unwrap();
        let original_digest = fresh_instance_source_digest(&explicit);

        bad_coordinate.calls[0].fresh_claims[0].beta[profile.layout.log_constraints] = EF::ZERO;
        assert!(validate_record(&profile, &bad_coordinate).is_err());

        let mut bad_alpha = record();
        bad_alpha.calls[0].logup_alpha += EF::ONE;
        assert!(validate_record(&profile, &bad_alpha).is_err());

        let mut bad_beta = record();
        bad_beta.calls[0].logup_beta += EF::ONE;
        assert!(validate_record(&profile, &bad_beta).is_err());

        let mut reordered = explicit.clone();
        reordered.swap(6, 7);
        assert_ne!(original_digest, fresh_instance_source_digest(&reordered));
    }

    #[test]
    fn proof_source_order_root_and_layout_mutations_change_or_fail_authority() {
        let mut bad_proof = profile();
        bad_proof.calls[0].proof_idx = 1;
        assert!(bad_proof.validate().is_err());

        let mut bad_source = profile();
        bad_source.calls[0].source_start = 1;
        assert!(bad_source.validate().is_err());

        let mut reordered_sources = profile();
        reordered_sources.calls[0]
            .expected_active_child_counts
            .swap(0, 1);
        assert!(reordered_sources.validate().is_ok());
        assert_ne!(
            reordered_sources.calls[0].normalized_start_for(1).unwrap(),
            2
        );

        let canonical = record();
        let link = FreshInstanceCallLinkAir::new(profile(), 0, buses()).unwrap();
        let canonical_trace = link.generate_trace(&canonical.calls[0]).unwrap();
        let mut changed_root = canonical.clone();
        changed_root.calls[0].full_root[0] += F::ONE;
        assert_ne!(canonical, changed_root);
        assert_ne!(
            canonical_trace.values.as_slice(),
            link.generate_trace(&changed_root.calls[0])
                .unwrap()
                .values
                .as_slice()
        );
        let mut changed_order = canonical.clone();
        changed_order.calls[0].source_order_digest[0] += F::ONE;
        assert_ne!(canonical, changed_order);
        assert_ne!(
            canonical_trace.values.as_slice(),
            link.generate_trace(&changed_order.calls[0])
                .unwrap()
                .values
                .as_slice()
        );

        let mut bad_layout = profile();
        bad_layout.layout.vm_pvs_explicit_offset -= 1;
        assert!(bad_layout.validate().is_err());
    }

    #[test]
    fn nonbase_vm_coordinate_is_rejected() {
        let profile = profile();
        let mut record = record();
        let coordinate = profile.layout.log_constraints + profile.layout.vm_pvs_explicit_offset;
        record.calls[0].fresh_claims[0].beta[coordinate] =
            EF::from_basis_coefficients_slice(&[F::ZERO, F::ONE, F::ZERO, F::ZERO]).unwrap();
        assert!(matches!(
            validate_record(&profile, &record),
            Err(FreshInstanceLinkError::NonCanonicalExplicit { .. })
                | Err(FreshInstanceLinkError::NonBaseVmPvs { .. })
        ));
    }
}

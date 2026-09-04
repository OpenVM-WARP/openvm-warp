//! Recursive statement AIR for SWIRL's post-stacking constrained-RS source.
//!
//! The ordinary recursive verifier remains responsible for AIR, LogUp, and
//! the stacked opening reduction.  This module consumes those verifier-owned
//! roots, points, and openings and derives the exact source passed to WARP's
//! constrained-code reduction:
//!
//! ```text
//! alpha = 0^log_codeword_len,
//! beta  = reverse(tilde_u),
//! eta   = sum_j theta^j * opening_j.
//! ```
//!
//! `mu` is exported with that claim but is deliberately not authorized here:
//! WARP's fresh-source opening verifier authenticates `codeword(alpha) = mu`
//! against the same ordered root tuple.  Consequently this AIR is neither a
//! PCS-opening predicate nor a PESAT relation.

use core::ops::Range;

use openvm_stark_backend::{
    interaction::InteractionBuilder,
    native_warp::NativeWarpChallenger,
    transcript::TranscriptLog,
    warp_accum::{
        observe_stacked_rs_commitment_prefix, StackedRsFreshCommitment,
        STACKED_RS_COMMITMENT_DOMAIN_TAG, STACKED_RS_COMMITMENT_VERSION,
    },
    warp_pesat::AlgebraicChallenger,
    BaseAirWithPublicValues, PartitionedBaseAir, TranscriptHistory,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{
    default_duplex_sponge_recorder, BabyBearPoseidon2Config, Digest, DIGEST_SIZE, D_EF, EF, F,
};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{extension::BinomiallyExtendable, BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::{CommitmentsBus, CommitmentsBusMessage, TranscriptBus},
    native_warp::{
        NativeReductionEndpointInputBus, NativeReductionEndpointInputMessage,
        NATIVE_REDUCTION_ENDPOINT_STACKING_OPENING, NATIVE_REDUCTION_ENDPOINT_STACKING_POINT,
    },
};

/// Independent transcript seed used by the native reduced-source adapter.
pub const REDUCED_SWIRL_THETA_SEED_TAG: &[u8] =
    b"openvm.native-warp.swirl-reduced-source.theta-seed.v1";
/// Version of the source transcript enclosing SWIRL's column projection.
// Must equal `REDUCED_SWIRL_NATIVE_PROTOCOL_VERSION`: the native source
// challenger absorbs the enclosing protocol version before the backend's
// canonical SWIRL observations.
pub const REDUCED_SWIRL_SOURCE_PROTOCOL_VERSION: u32 = 3;
/// Domain separating the roots/point/openings-to-theta reduction.
pub const REDUCED_SWIRL_COLUMN_BATCHING_THETA_TAG: &[u8] =
    b"openvm.swirl.stacking-prefix.column-batching-theta.v2";
/// Canonical coefficient-native RS layout tag from the backend binding.
pub const REDUCED_SWIRL_COEFFICIENT_SUBGROUP_TAG: u64 = 0x434f_4546;
pub const REDUCED_SWIRL_COEFFICIENT_SUBGROUP_VERSION: u32 = 1;

crate::define_typed_lookup_bus!(ReducedSwirlSourceRootBus, ReducedSwirlSourceRootMessage);
crate::define_typed_lookup_bus!(ReducedSwirlSourcePointBus, ReducedSwirlSourcePointMessage);
crate::define_typed_lookup_bus!(
    ReducedSwirlSourceOpeningBus,
    ReducedSwirlSourceOpeningMessage
);
crate::define_typed_lookup_bus!(ReducedSwirlSourceBetaBus, ReducedSwirlSourceBetaMessage);
crate::define_typed_lookup_bus!(ReducedSwirlSourceClaimBus, ReducedSwirlSourceClaimMessage);
crate::define_typed_permutation_bus!(
    ReducedSwirlSourceRootWidthBus,
    ReducedSwirlSourceRootWidthMessage
);

#[repr(C)]
#[derive(openvm_recursion_circuit_derive::AlignedBorrow, Clone, Debug)]
pub struct ReducedSwirlSourceRootMessage<T> {
    pub source: T,
    pub root_ordinal: T,
    pub width: T,
    pub root: [T; DIGEST_SIZE],
}

#[repr(C)]
#[derive(openvm_recursion_circuit_derive::AlignedBorrow, Clone, Debug)]
pub struct ReducedSwirlSourcePointMessage<T> {
    pub source: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

#[repr(C)]
#[derive(openvm_recursion_circuit_derive::AlignedBorrow, Clone, Debug)]
pub struct ReducedSwirlSourceOpeningMessage<T> {
    pub source: T,
    pub opening_index: T,
    pub root_ordinal: T,
    pub column: T,
    pub value: [T; D_EF],
}

#[repr(C)]
#[derive(openvm_recursion_circuit_derive::AlignedBorrow, Clone, Debug)]
pub struct ReducedSwirlSourceBetaMessage<T> {
    pub source: T,
    pub coordinate: T,
    pub value: [T; D_EF],
}

/// One canonical coefficient-native constrained-RS claim.
///
/// The all-zero `alpha` is represented by `alpha_is_zero = 1` together with
/// the setup-fixed `log_codeword_len`; allocating one row per zero coordinate
/// would add no information.  `beta` is exported separately, coordinate by
/// coordinate, on [`ReducedSwirlSourceBetaBus`].
#[repr(C)]
#[derive(openvm_recursion_circuit_derive::AlignedBorrow, Clone, Debug)]
pub struct ReducedSwirlSourceClaimMessage<T> {
    pub source: T,
    pub root_count: T,
    pub opening_count: T,
    pub l_skip: T,
    pub n_stack: T,
    pub log_blowup: T,
    pub log_commit_rows_per_query: T,
    pub log_message_len: T,
    pub log_codeword_len: T,
    pub rows_per_query: T,
    pub coefficient_layout_tag: T,
    pub coefficient_layout_version: T,
    pub alpha_is_zero: T,
    pub theta: [T; D_EF],
    pub mu: [T; D_EF],
    pub eta: [T; D_EF],
}

#[repr(C)]
#[derive(openvm_recursion_circuit_derive::AlignedBorrow, Clone, Debug)]
pub struct ReducedSwirlSourceRootWidthMessage<T> {
    pub source: T,
    pub root_ordinal: T,
    pub width: T,
}

/// Setup-fixed source envelope used by one recursive wrapper key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReducedSwirlSourceProfile {
    pub maximum_sources: usize,
    pub maximum_roots_per_source: usize,
    pub maximum_openings_per_source: usize,
    pub l_skip: usize,
    pub n_stack: usize,
    pub log_blowup: usize,
    pub log_commit_rows_per_query: usize,
}

impl ReducedSwirlSourceProfile {
    pub fn validate(self) -> Result<(), &'static str> {
        if self.maximum_sources == 0
            || !self.maximum_sources.is_power_of_two()
            || self.maximum_roots_per_source == 0
            || self.maximum_openings_per_source < self.maximum_roots_per_source
        {
            return Err("reduced-SWIRL source capacity");
        }
        let log_message_len = self
            .l_skip
            .checked_add(self.n_stack)
            .ok_or("reduced-SWIRL message dimension")?;
        let log_codeword_len = log_message_len
            .checked_add(self.log_blowup)
            .ok_or("reduced-SWIRL codeword dimension")?;
        if log_message_len == 0 || self.log_commit_rows_per_query > log_codeword_len {
            return Err("reduced-SWIRL RS layout");
        }
        1usize
            .checked_shl(self.log_commit_rows_per_query as u32)
            .ok_or("reduced-SWIRL rows per query")?;
        Ok(())
    }

    #[must_use]
    pub const fn log_message_len(self) -> usize {
        self.l_skip + self.n_stack
    }

    #[must_use]
    pub const fn log_codeword_len(self) -> usize {
        self.log_message_len() + self.log_blowup
    }

    #[must_use]
    pub const fn rows_per_query(self) -> usize {
        1usize << self.log_commit_rows_per_query
    }

    #[must_use]
    pub const fn trace_height(self) -> usize {
        self.maximum_sources
    }

    #[must_use]
    pub const fn trace_width(self) -> usize {
        HEADER_WIDTH
            + self.maximum_roots_per_source * ROOT_SLOT_WIDTH
            + self.log_message_len() * D_EF
            + self.maximum_openings_per_source * OPENING_SLOT_WIDTH
    }
}

/// Host-side input produced from a verifier-checked retained stacking proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReducedSwirlSourceRecord {
    pub roots: Vec<Digest>,
    pub widths: Vec<usize>,
    /// SWIRL order.  The exported WARP `beta` reverses this vector.
    pub stacking_point: Vec<EF>,
    /// Commitment-major and column-major, matching `roots` and `widths`.
    pub stacking_openings: Vec<Vec<EF>>,
    /// Row-zero claim authenticated later by WARP against `roots`.
    pub mu: EF,
}

/// Canonical public claim derived by [`generate_reduced_swirl_source_trace`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecursiveReducedSwirlClaim {
    pub roots: Vec<Digest>,
    pub widths: Vec<usize>,
    pub theta: EF,
    pub alpha: Vec<EF>,
    pub mu: EF,
    pub beta: Vec<EF>,
    pub eta: EF,
}

pub struct ReducedSwirlSourceTraceArtifacts {
    pub trace: RowMajorMatrix<F>,
    pub transcript_logs: Vec<TranscriptLog<F, [F; openvm_poseidon2_air::POSEIDON2_WIDTH]>>,
    pub claims: Vec<RecursiveReducedSwirlClaim>,
}

/// One fixed-height AIR table.  Runtime sources occupy a canonical prefix;
/// every unused source/root/opening slot is constrained to zero.
#[derive(Clone, Copy, Debug)]
pub struct ReducedSwirlSourceAir {
    pub profile: ReducedSwirlSourceProfile,
    pub transcript_bus: TranscriptBus,
    pub commitments_bus: CommitmentsBus,
    pub stacking_endpoint_bus: NativeReductionEndpointInputBus,
    pub root_width_bus: ReducedSwirlSourceRootWidthBus,
    pub root_export_bus: ReducedSwirlSourceRootBus,
    pub point_export_bus: ReducedSwirlSourcePointBus,
    pub opening_export_bus: ReducedSwirlSourceOpeningBus,
    pub beta_export_bus: ReducedSwirlSourceBetaBus,
    pub claim_export_bus: ReducedSwirlSourceClaimBus,
    /// Setup-owned number of wrapper consumers for every exported item.
    pub export_lookup_count: u32,
}

impl BaseAir<F> for ReducedSwirlSourceAir {
    fn width(&self) -> usize {
        self.profile.trace_width()
    }
}

impl BaseAirWithPublicValues<F> for ReducedSwirlSourceAir {}
impl PartitionedBaseAir<F> for ReducedSwirlSourceAir {}

const ACTIVE: usize = 0;
const SOURCE: usize = 1;
const ROOT_COUNT: usize = 2;
const OPENING_COUNT: usize = 3;
const THETA: usize = 4;
const MU: usize = THETA + D_EF;
const ETA: usize = MU + D_EF;
const HEADER_WIDTH: usize = ETA + D_EF;

const ROOT_ACTIVE: usize = 0;
const ROOT_WIDTH: usize = 1;
const ROOT_WIDTH_INVERSE: usize = 2;
const ROOT_DIGEST: usize = 3;
const ROOT_SLOT_WIDTH: usize = ROOT_DIGEST + DIGEST_SIZE;

const OPENING_ACTIVE: usize = 0;
const OPENING_ROOT: usize = 1;
const OPENING_COLUMN: usize = 2;
const OPENING_FIRST: usize = 3;
const OPENING_LAST: usize = 4;
const OPENING_GROUP_WIDTH: usize = 5;
const OPENING_THETA_POWER: usize = 6;
const OPENING_ACC_BEFORE: usize = OPENING_THETA_POWER + D_EF;
const OPENING_VALUE: usize = OPENING_ACC_BEFORE + D_EF;
const OPENING_ACC_AFTER: usize = OPENING_VALUE + D_EF;
const OPENING_SLOT_WIDTH: usize = OPENING_ACC_AFTER + D_EF;

fn root_slot(_profile: ReducedSwirlSourceProfile, ordinal: usize) -> Range<usize> {
    let start = HEADER_WIDTH + ordinal * ROOT_SLOT_WIDTH;
    start..start + ROOT_SLOT_WIDTH
}

fn point_slot(profile: ReducedSwirlSourceProfile, coordinate: usize) -> Range<usize> {
    let start =
        HEADER_WIDTH + profile.maximum_roots_per_source * ROOT_SLOT_WIDTH + coordinate * D_EF;
    start..start + D_EF
}

fn opening_slot(profile: ReducedSwirlSourceProfile, opening: usize) -> Range<usize> {
    let start = HEADER_WIDTH
        + profile.maximum_roots_per_source * ROOT_SLOT_WIDTH
        + profile.log_message_len() * D_EF
        + opening * OPENING_SLOT_WIDTH;
    start..start + OPENING_SLOT_WIDTH
}

impl<AB> Air<AB> for ReducedSwirlSourceAir
where
    AB: AirBuilder<F = F> + InteractionBuilder,
    AB::Var: Copy,
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield: BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        assert!(self.profile.validate().is_ok());
        assert!(self.export_lookup_count > 0);
        let main = builder.main();
        let row = main.row_slice(0).expect("reduced-SWIRL source row");
        let next_row = main.row_slice(1).expect("reduced-SWIRL source next row");
        let local = &*row;
        let next = &*next_row;
        let active = AB::Expr::from(local[ACTIVE]);
        let next_active = AB::Expr::from(next[ACTIVE]);
        builder.assert_bool(local[ACTIVE]);
        builder.when_first_row().assert_one(local[ACTIVE]);
        builder.when_first_row().assert_zero(local[SOURCE]);
        builder
            .when_transition()
            .assert_zero(next_active.clone() * (AB::Expr::ONE - active.clone()));
        builder
            .when_transition()
            .when(next_active.clone())
            .assert_eq(next[SOURCE], AB::Expr::from(local[SOURCE]) + AB::Expr::ONE);

        // Disabled rows have exactly one representation.
        let inactive = AB::Expr::ONE - active.clone();
        for &value in local.iter().skip(1) {
            builder.when(inactive.clone()).assert_zero(value);
        }

        let theta = ext_at::<AB>(local, THETA);
        let mu = ext_at::<AB>(local, MU);
        let eta = ext_at::<AB>(local, ETA);

        let mut root_count = AB::Expr::ZERO;
        for ordinal in 0..self.profile.maximum_roots_per_source {
            let slot = root_slot(self.profile, ordinal);
            let base = slot.start;
            let root_active = AB::Expr::from(local[base + ROOT_ACTIVE]);
            builder.assert_bool(local[base + ROOT_ACTIVE]);
            if ordinal == 0 {
                builder.assert_eq(root_active.clone(), active.clone());
            } else {
                let previous = root_slot(self.profile, ordinal - 1).start;
                builder.assert_zero(
                    root_active.clone()
                        * (AB::Expr::ONE - AB::Expr::from(local[previous + ROOT_ACTIVE])),
                );
            }
            let root_inactive = AB::Expr::ONE - root_active.clone();
            for &value in &local[base + ROOT_WIDTH..slot.end] {
                builder.when(root_inactive.clone()).assert_zero(value);
            }
            builder.assert_eq(
                AB::Expr::from(local[base + ROOT_WIDTH])
                    * AB::Expr::from(local[base + ROOT_WIDTH_INVERSE]),
                root_active.clone(),
            );
            root_count += root_active.clone();
            let root = core::array::from_fn(|limb| local[base + ROOT_DIGEST + limb]);
            self.commitments_bus.lookup_key(
                builder,
                local[SOURCE],
                CommitmentsBusMessage {
                    major_idx: AB::Expr::ZERO,
                    minor_idx: AB::Expr::from_usize(ordinal),
                    commitment: root.map(Into::into),
                },
                root_active.clone(),
            );
            self.root_width_bus.send(
                builder,
                ReducedSwirlSourceRootWidthMessage {
                    source: local[SOURCE].into(),
                    root_ordinal: AB::Expr::from_usize(ordinal),
                    width: local[base + ROOT_WIDTH].into(),
                },
                root_active.clone(),
            );
            self.root_export_bus.add_key_with_lookups(
                builder,
                ReducedSwirlSourceRootMessage {
                    source: local[SOURCE].into(),
                    root_ordinal: AB::Expr::from_usize(ordinal),
                    width: local[base + ROOT_WIDTH].into(),
                    root: root.map(Into::into),
                },
                root_active * AB::Expr::from_u32(self.export_lookup_count),
            );
        }
        builder
            .when(active.clone())
            .assert_eq(local[ROOT_COUNT], root_count.clone());

        for coordinate in 0..self.profile.log_message_len() {
            let point = ext_range::<AB>(local, point_slot(self.profile, coordinate));
            self.stacking_endpoint_bus.lookup_key(
                builder,
                NativeReductionEndpointInputMessage {
                    reduction: local[SOURCE].into(),
                    kind: AB::Expr::from_usize(NATIVE_REDUCTION_ENDPOINT_STACKING_POINT),
                    index: AB::Expr::from_usize(coordinate),
                    value: point.clone(),
                },
                active.clone(),
            );
            self.point_export_bus.add_key_with_lookups(
                builder,
                ReducedSwirlSourcePointMessage {
                    source: local[SOURCE].into(),
                    coordinate: AB::Expr::from_usize(coordinate),
                    value: point.clone(),
                },
                active.clone() * AB::Expr::from_u32(self.export_lookup_count),
            );
            let beta_coordinate = self.profile.log_message_len() - 1 - coordinate;
            self.beta_export_bus.add_key_with_lookups(
                builder,
                ReducedSwirlSourceBetaMessage {
                    source: local[SOURCE].into(),
                    coordinate: AB::Expr::from_usize(beta_coordinate),
                    value: point,
                },
                active.clone() * AB::Expr::from_u32(self.export_lookup_count),
            );
        }

        let mut opening_count = AB::Expr::ZERO;
        for opening in 0..self.profile.maximum_openings_per_source {
            let slot = opening_slot(self.profile, opening);
            let base = slot.start;
            let opening_active = AB::Expr::from(local[base + OPENING_ACTIVE]);
            builder.assert_bool(local[base + OPENING_ACTIVE]);
            builder.assert_bool(local[base + OPENING_FIRST]);
            builder.assert_bool(local[base + OPENING_LAST]);
            if opening == 0 {
                builder.assert_eq(opening_active.clone(), active.clone());
                builder.assert_eq(local[base + OPENING_FIRST], local[base + OPENING_ACTIVE]);
                builder
                    .when(opening_active.clone())
                    .assert_zero(local[base + OPENING_ROOT]);
                builder
                    .when(opening_active.clone())
                    .assert_zero(local[base + OPENING_COLUMN]);
            } else {
                let previous = opening_slot(self.profile, opening - 1).start;
                let previous_active = AB::Expr::from(local[previous + OPENING_ACTIVE]);
                builder.assert_zero(
                    opening_active.clone() * (AB::Expr::ONE - previous_active.clone()),
                );
                builder
                    .when(opening_active.clone())
                    .assert_eq(local[base + OPENING_FIRST], local[previous + OPENING_LAST]);
                let first = AB::Expr::from(local[base + OPENING_FIRST]);
                builder
                    .when(opening_active.clone() * first.clone())
                    .assert_eq(
                        local[base + OPENING_ROOT],
                        AB::Expr::from(local[previous + OPENING_ROOT]) + AB::Expr::ONE,
                    );
                builder
                    .when(opening_active.clone() * first.clone())
                    .assert_zero(local[base + OPENING_COLUMN]);
                builder
                    .when(opening_active.clone() * (AB::Expr::ONE - first))
                    .assert_eq(local[base + OPENING_ROOT], local[previous + OPENING_ROOT]);
                builder
                    .when(
                        opening_active.clone()
                            * (AB::Expr::ONE - AB::Expr::from(local[base + OPENING_FIRST])),
                    )
                    .assert_eq(
                        local[base + OPENING_COLUMN],
                        AB::Expr::from(local[previous + OPENING_COLUMN]) + AB::Expr::ONE,
                    );
            }
            let opening_inactive = AB::Expr::ONE - opening_active.clone();
            for &value in &local[base + OPENING_ROOT..slot.end] {
                builder.when(opening_inactive.clone()).assert_zero(value);
            }

            let group_first = opening_active.clone() * local[base + OPENING_FIRST];
            self.root_width_bus.receive(
                builder,
                ReducedSwirlSourceRootWidthMessage {
                    source: local[SOURCE].into(),
                    root_ordinal: local[base + OPENING_ROOT].into(),
                    width: local[base + OPENING_GROUP_WIDTH].into(),
                },
                group_first.clone(),
            );
            if opening > 0 {
                let previous = opening_slot(self.profile, opening - 1).start;
                builder
                    .when(
                        opening_active.clone()
                            * (AB::Expr::ONE - AB::Expr::from(local[base + OPENING_FIRST])),
                    )
                    .assert_eq(
                        local[base + OPENING_GROUP_WIDTH],
                        local[previous + OPENING_GROUP_WIDTH],
                    );
            }
            builder
                .when(opening_active.clone() * local[base + OPENING_LAST])
                .assert_eq(
                    AB::Expr::from(local[base + OPENING_COLUMN]) + AB::Expr::ONE,
                    local[base + OPENING_GROUP_WIDTH],
                );

            let power = ext_at::<AB>(local, base + OPENING_THETA_POWER);
            let acc_before = ext_at::<AB>(local, base + OPENING_ACC_BEFORE);
            let value = ext_at::<AB>(local, base + OPENING_VALUE);
            let acc_after = ext_at::<AB>(local, base + OPENING_ACC_AFTER);
            if opening == 0 {
                assert_ext_eq_when::<AB>(
                    builder,
                    opening_active.clone(),
                    power.clone(),
                    ext_one::<AB>(),
                );
                assert_ext_zero_when::<AB>(builder, opening_active.clone(), acc_before.clone());
            } else {
                let previous = opening_slot(self.profile, opening - 1).start;
                let previous_power = ext_at::<AB>(local, previous + OPENING_THETA_POWER);
                let previous_after = ext_at::<AB>(local, previous + OPENING_ACC_AFTER);
                assert_ext_eq_when::<AB>(
                    builder,
                    opening_active.clone(),
                    power.clone(),
                    ext_mul::<AB::Expr>(previous_power, theta.clone()),
                );
                assert_ext_eq_when::<AB>(
                    builder,
                    opening_active.clone(),
                    acc_before.clone(),
                    previous_after,
                );
            }
            let weighted = ext_mul::<AB::Expr>(power, value.clone());
            assert_ext_eq_when::<AB>(
                builder,
                opening_active.clone(),
                acc_after.clone(),
                core::array::from_fn(|limb| acc_before[limb].clone() + weighted[limb].clone()),
            );

            let next_opening_active = if opening + 1 == self.profile.maximum_openings_per_source {
                AB::Expr::ZERO
            } else {
                let next = opening_slot(self.profile, opening + 1).start;
                AB::Expr::from(local[next + OPENING_ACTIVE])
            };
            let source_last = opening_active.clone() * (AB::Expr::ONE - next_opening_active);
            builder
                .when(source_last.clone())
                .assert_one(local[base + OPENING_LAST]);
            builder.when(source_last.clone()).assert_eq(
                AB::Expr::from(local[base + OPENING_ROOT]) + AB::Expr::ONE,
                local[ROOT_COUNT],
            );
            assert_ext_eq_when::<AB>(builder, source_last, acc_after, eta.clone());

            self.stacking_endpoint_bus.lookup_key(
                builder,
                NativeReductionEndpointInputMessage {
                    reduction: local[SOURCE].into(),
                    kind: AB::Expr::from_usize(NATIVE_REDUCTION_ENDPOINT_STACKING_OPENING),
                    index: AB::Expr::from_usize(opening),
                    value: value.clone(),
                },
                opening_active.clone(),
            );
            self.opening_export_bus.add_key_with_lookups(
                builder,
                ReducedSwirlSourceOpeningMessage {
                    source: local[SOURCE].into(),
                    opening_index: AB::Expr::from_usize(opening),
                    root_ordinal: local[base + OPENING_ROOT].into(),
                    column: local[base + OPENING_COLUMN].into(),
                    value,
                },
                opening_active.clone() * AB::Expr::from_u32(self.export_lookup_count),
            );
            opening_count += opening_active;
        }
        builder
            .when(active.clone())
            .assert_eq(local[OPENING_COUNT], opening_count);

        let mut tidx = AB::Expr::ZERO;
        // Reproduce `reduced_swirl_source_challenger` followed by the backend
        // source adapter byte-for-byte. The first two observations seed the
        // per-source transcript; `try_from_pending` then absorbs the canonical
        // SWIRL column-batching tag and the retained commitment descriptor.
        observe_bytes_air(
            &self.transcript_bus,
            builder,
            local[SOURCE],
            &mut tidx,
            REDUCED_SWIRL_THETA_SEED_TAG,
            active.clone(),
        );
        observe_u64_bytes_air(
            &self.transcript_bus,
            builder,
            local[SOURCE],
            &mut tidx,
            u64::from(REDUCED_SWIRL_SOURCE_PROTOCOL_VERSION),
            active.clone(),
        );
        observe_backend_bytes_air(
            &self.transcript_bus,
            builder,
            local[SOURCE],
            &mut tidx,
            REDUCED_SWIRL_COLUMN_BATCHING_THETA_TAG,
            active.clone(),
        );
        for value in [
            AB::Expr::from_u64(STACKED_RS_COMMITMENT_DOMAIN_TAG),
            AB::Expr::from_u64(STACKED_RS_COMMITMENT_VERSION),
            local[ROOT_COUNT].into(),
            AB::Expr::from_usize(self.profile.l_skip),
            AB::Expr::from_usize(self.profile.log_message_len()),
            AB::Expr::from_usize(self.profile.log_message_len()),
            AB::Expr::from_usize(self.profile.log_codeword_len()),
            AB::Expr::from_usize(self.profile.rows_per_query()),
        ] {
            observe_base_ext_air(
                &self.transcript_bus,
                builder,
                local[SOURCE],
                &mut tidx,
                value,
                active.clone(),
            );
        }
        for ordinal in 0..self.profile.maximum_roots_per_source {
            let base = root_slot(self.profile, ordinal).start;
            let enabled = AB::Expr::from(local[base + ROOT_ACTIVE]);
            observe_base_ext_air(
                &self.transcript_bus,
                builder,
                local[SOURCE],
                &mut tidx,
                local[base + ROOT_WIDTH].into(),
                enabled.clone(),
            );
            for limb in 0..DIGEST_SIZE {
                observe_base_ext_air(
                    &self.transcript_bus,
                    builder,
                    local[SOURCE],
                    &mut tidx,
                    local[base + ROOT_DIGEST + limb].into(),
                    enabled.clone(),
                );
            }
        }
        observe_base_ext_air(
            &self.transcript_bus,
            builder,
            local[SOURCE],
            &mut tidx,
            AB::Expr::from_usize(self.profile.log_message_len()),
            active.clone(),
        );
        for coordinate in 0..self.profile.log_message_len() {
            observe_ext_air(
                &self.transcript_bus,
                builder,
                local[SOURCE],
                &mut tidx,
                ext_range::<AB>(local, point_slot(self.profile, coordinate)),
                active.clone(),
            );
        }
        observe_base_ext_air(
            &self.transcript_bus,
            builder,
            local[SOURCE],
            &mut tidx,
            local[ROOT_COUNT].into(),
            active.clone(),
        );
        for opening in 0..self.profile.maximum_openings_per_source {
            let base = opening_slot(self.profile, opening).start;
            let enabled = AB::Expr::from(local[base + OPENING_ACTIVE]);
            let first = enabled.clone() * local[base + OPENING_FIRST];
            observe_base_ext_air(
                &self.transcript_bus,
                builder,
                local[SOURCE],
                &mut tidx,
                local[base + OPENING_GROUP_WIDTH].into(),
                first,
            );
            observe_ext_air(
                &self.transcript_bus,
                builder,
                local[SOURCE],
                &mut tidx,
                ext_at::<AB>(local, base + OPENING_VALUE),
                enabled,
            );
        }
        self.transcript_bus
            .sample_ext(builder, local[SOURCE], tidx, theta.clone(), active.clone());

        self.claim_export_bus.add_key_with_lookups(
            builder,
            ReducedSwirlSourceClaimMessage {
                source: local[SOURCE].into(),
                root_count: local[ROOT_COUNT].into(),
                opening_count: local[OPENING_COUNT].into(),
                l_skip: AB::Expr::from_usize(self.profile.l_skip),
                n_stack: AB::Expr::from_usize(self.profile.n_stack),
                log_blowup: AB::Expr::from_usize(self.profile.log_blowup),
                log_commit_rows_per_query: AB::Expr::from_usize(
                    self.profile.log_commit_rows_per_query,
                ),
                log_message_len: AB::Expr::from_usize(self.profile.log_message_len()),
                log_codeword_len: AB::Expr::from_usize(self.profile.log_codeword_len()),
                rows_per_query: AB::Expr::from_usize(self.profile.rows_per_query()),
                coefficient_layout_tag: AB::Expr::from_u64(REDUCED_SWIRL_COEFFICIENT_SUBGROUP_TAG),
                coefficient_layout_version: AB::Expr::from_u32(
                    REDUCED_SWIRL_COEFFICIENT_SUBGROUP_VERSION,
                ),
                alpha_is_zero: AB::Expr::ONE,
                theta,
                mu,
                eta,
            },
            active * AB::Expr::from_u32(self.export_lookup_count),
        );
    }
}

/// Generate the fixed-capacity trace and the independent source transcripts.
pub fn generate_reduced_swirl_source_trace(
    profile: ReducedSwirlSourceProfile,
    records: &[ReducedSwirlSourceRecord],
) -> Result<ReducedSwirlSourceTraceArtifacts, &'static str> {
    profile.validate()?;
    if records.is_empty() || records.len() > profile.maximum_sources {
        return Err("reduced-SWIRL active source count");
    }
    let width = profile.trace_width();
    let mut values = F::zero_vec(profile.trace_height() * width);
    let mut transcript_logs = Vec::with_capacity(records.len());
    let mut claims = Vec::with_capacity(records.len());

    for (source, record) in records.iter().enumerate() {
        validate_record(profile, record)?;
        let (theta, transcript_log) = derive_source_theta_and_log(profile, record)?;
        let flattened = record
            .stacking_openings
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let eta = fold_openings(&flattened, theta);
        let row = &mut values[source * width..(source + 1) * width];
        row[ACTIVE] = F::ONE;
        row[SOURCE] = F::from_usize(source);
        row[ROOT_COUNT] = F::from_usize(record.roots.len());
        row[OPENING_COUNT] = F::from_usize(flattened.len());
        copy_ext(&mut row[THETA..THETA + D_EF], theta);
        copy_ext(&mut row[MU..MU + D_EF], record.mu);
        copy_ext(&mut row[ETA..ETA + D_EF], eta);

        for (ordinal, (&root, &root_width)) in record.roots.iter().zip(&record.widths).enumerate() {
            let slot = root_slot(profile, ordinal);
            let base = slot.start;
            row[base + ROOT_ACTIVE] = F::ONE;
            row[base + ROOT_WIDTH] = F::from_usize(root_width);
            row[base + ROOT_WIDTH_INVERSE] = F::from_usize(root_width).inverse();
            row[base + ROOT_DIGEST..base + ROOT_DIGEST + DIGEST_SIZE].copy_from_slice(&root);
        }
        for (coordinate, &point) in record.stacking_point.iter().enumerate() {
            copy_ext(&mut row[point_slot(profile, coordinate)], point);
        }

        let mut flat_index = 0;
        let mut theta_power = EF::ONE;
        let mut accumulated = EF::ZERO;
        for (root_ordinal, opening_group) in record.stacking_openings.iter().enumerate() {
            for (column, &opening) in opening_group.iter().enumerate() {
                let slot = opening_slot(profile, flat_index);
                let base = slot.start;
                row[base + OPENING_ACTIVE] = F::ONE;
                row[base + OPENING_ROOT] = F::from_usize(root_ordinal);
                row[base + OPENING_COLUMN] = F::from_usize(column);
                row[base + OPENING_FIRST] = F::from_bool(column == 0);
                row[base + OPENING_LAST] = F::from_bool(column + 1 == opening_group.len());
                row[base + OPENING_GROUP_WIDTH] = F::from_usize(opening_group.len());
                copy_ext(
                    &mut row[base + OPENING_THETA_POWER..base + OPENING_THETA_POWER + D_EF],
                    theta_power,
                );
                copy_ext(
                    &mut row[base + OPENING_ACC_BEFORE..base + OPENING_ACC_BEFORE + D_EF],
                    accumulated,
                );
                copy_ext(
                    &mut row[base + OPENING_VALUE..base + OPENING_VALUE + D_EF],
                    opening,
                );
                accumulated += theta_power * opening;
                copy_ext(
                    &mut row[base + OPENING_ACC_AFTER..base + OPENING_ACC_AFTER + D_EF],
                    accumulated,
                );
                theta_power *= theta;
                flat_index += 1;
            }
        }
        debug_assert_eq!(accumulated, eta);
        transcript_logs.push(transcript_log);
        claims.push(RecursiveReducedSwirlClaim {
            roots: record.roots.clone(),
            widths: record.widths.clone(),
            theta,
            alpha: vec![EF::ZERO; profile.log_codeword_len()],
            mu: record.mu,
            beta: record.stacking_point.iter().rev().copied().collect(),
            eta,
        });
    }

    Ok(ReducedSwirlSourceTraceArtifacts {
        trace: RowMajorMatrix::new(values, width),
        transcript_logs,
        claims,
    })
}

fn validate_record(
    profile: ReducedSwirlSourceProfile,
    record: &ReducedSwirlSourceRecord,
) -> Result<(), &'static str> {
    if record.roots.is_empty()
        || record.roots.len() > profile.maximum_roots_per_source
        || record.roots.len() != record.widths.len()
        || record.roots.len() != record.stacking_openings.len()
        || record.stacking_point.len() != profile.log_message_len()
        || record.widths.iter().any(|&width| width == 0)
    {
        return Err("reduced-SWIRL source shape");
    }
    for (&width, openings) in record.widths.iter().zip(&record.stacking_openings) {
        if openings.len() != width {
            return Err("reduced-SWIRL opening width");
        }
    }
    let opening_count = record
        .widths
        .iter()
        .try_fold(0usize, |sum, &width| sum.checked_add(width))
        .ok_or("reduced-SWIRL opening count")?;
    if opening_count > profile.maximum_openings_per_source {
        return Err("reduced-SWIRL opening capacity");
    }
    Ok(())
}

fn derive_source_theta_and_log(
    profile: ReducedSwirlSourceProfile,
    record: &ReducedSwirlSourceRecord,
) -> Result<
    (
        EF,
        TranscriptLog<F, [F; openvm_poseidon2_air::POSEIDON2_WIDTH]>,
    ),
    &'static str,
> {
    let mut challenger =
        NativeWarpChallenger::<BabyBearPoseidon2Config, _>::new(default_duplex_sponge_recorder());
    observe_bytes_host(REDUCED_SWIRL_THETA_SEED_TAG, &mut challenger);
    observe_u64_bytes_host(
        u64::from(REDUCED_SWIRL_SOURCE_PROTOCOL_VERSION),
        &mut challenger,
    );
    observe_backend_bytes_host(REDUCED_SWIRL_COLUMN_BATCHING_THETA_TAG, &mut challenger);
    let descriptor = StackedRsFreshCommitment {
        roots: record.roots.clone(),
        widths: record.widths.clone(),
        l_skip: profile.l_skip,
        native_log_message_len: profile.log_message_len(),
        log_message_len: profile.log_message_len(),
        log_codeword_len: profile.log_codeword_len(),
        rows_per_query: profile.rows_per_query(),
        theta: EF::ZERO,
    };
    observe_stacked_rs_commitment_prefix(&descriptor, &mut challenger);
    AlgebraicChallenger::observe(&mut challenger, EF::from_usize(record.stacking_point.len()));
    for &value in &record.stacking_point {
        AlgebraicChallenger::observe(&mut challenger, value);
    }
    AlgebraicChallenger::observe(
        &mut challenger,
        EF::from_usize(record.stacking_openings.len()),
    );
    for opening in &record.stacking_openings {
        AlgebraicChallenger::observe(&mut challenger, EF::from_usize(opening.len()));
        for &value in opening {
            AlgebraicChallenger::observe(&mut challenger, value);
        }
    }
    let theta = AlgebraicChallenger::sample(&mut challenger);
    let log = TranscriptHistory::into_log(challenger.into_inner());
    Ok((theta, log))
}

fn observe_bytes_host<Ch: AlgebraicChallenger<EF>>(bytes: &[u8], challenger: &mut Ch) {
    observe_u64_bytes_host(bytes.len() as u64, challenger);
    for &byte in bytes {
        challenger.observe(EF::from_u8(byte));
    }
}

// The backend's reduced-SWIRL adapter predates the native wrapper's byte
// helper and encodes a byte-string length as one field element. Keep this
// distinct from `observe_bytes_host`, whose eight-byte length prefix is part
// of `reduced_swirl_source_challenger`.
fn observe_backend_bytes_host<Ch: AlgebraicChallenger<EF>>(bytes: &[u8], challenger: &mut Ch) {
    challenger.observe(EF::from_usize(bytes.len()));
    for &byte in bytes {
        challenger.observe(EF::from_u8(byte));
    }
}

fn observe_u64_bytes_host<Ch: AlgebraicChallenger<EF>>(value: u64, challenger: &mut Ch) {
    for byte in value.to_le_bytes() {
        challenger.observe(EF::from_u8(byte));
    }
}

fn fold_openings(openings: &[EF], theta: EF) -> EF {
    let mut power = EF::ONE;
    let mut output = EF::ZERO;
    for &opening in openings {
        output += power * opening;
        power *= theta;
    }
    output
}

fn copy_ext(target: &mut [F], value: EF) {
    target.copy_from_slice(value.as_basis_coefficients_slice());
}

fn ext_at<AB: AirBuilder<F = F>>(row: &[AB::Var], start: usize) -> [AB::Expr; D_EF]
where
    AB::Var: Copy,
{
    core::array::from_fn(|limb| row[start + limb].into())
}

fn ext_range<AB: AirBuilder<F = F>>(row: &[AB::Var], range: Range<usize>) -> [AB::Expr; D_EF]
where
    AB::Var: Copy,
{
    debug_assert_eq!(range.len(), D_EF);
    core::array::from_fn(|limb| row[range.start + limb].into())
}

fn ext_one<AB: AirBuilder<F = F>>() -> [AB::Expr; D_EF] {
    core::array::from_fn(|limb| {
        if limb == 0 {
            AB::Expr::ONE
        } else {
            AB::Expr::ZERO
        }
    })
}

fn ext_mul<FA>(left: [FA; D_EF], right: [FA; D_EF]) -> [FA; D_EF]
where
    FA: PrimeCharacteristicRing,
    FA::PrimeSubfield: BinomiallyExtendable<D_EF>,
{
    let w = FA::from_prime_subfield(FA::PrimeSubfield::W);
    let mut output = core::array::from_fn(|_| FA::ZERO);
    for (left_degree, left_value) in left.iter().enumerate() {
        for (right_degree, right_value) in right.iter().enumerate() {
            let degree = left_degree + right_degree;
            let mut term = left_value.clone() * right_value.clone();
            if degree >= D_EF {
                term *= w.clone();
            }
            output[degree % D_EF] = output[degree % D_EF].clone() + term;
        }
    }
    output
}

fn assert_ext_zero_when<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    enabled: AB::Expr,
    value: [AB::Expr; D_EF],
) {
    for limb in value {
        builder.when(enabled.clone()).assert_zero(limb);
    }
}

fn assert_ext_eq_when<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    enabled: AB::Expr,
    left: [AB::Expr; D_EF],
    right: [AB::Expr; D_EF],
) {
    for (left, right) in left.into_iter().zip(right) {
        builder.when(enabled.clone()).assert_eq(left, right);
    }
}

fn observe_ext_air<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Var,
    tidx: &mut AB::Expr,
    value: [AB::Expr; D_EF],
    enabled: AB::Expr,
) {
    bus.observe_ext(builder, proof_idx, tidx.clone(), value, enabled.clone());
    *tidx = tidx.clone() + enabled * AB::Expr::from_usize(D_EF);
}

fn observe_base_ext_air<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Var,
    tidx: &mut AB::Expr,
    value: AB::Expr,
    enabled: AB::Expr,
) {
    observe_ext_air(
        bus,
        builder,
        proof_idx,
        tidx,
        [value, AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO],
        enabled,
    );
}

fn observe_u64_bytes_air<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Var,
    tidx: &mut AB::Expr,
    value: u64,
    enabled: AB::Expr,
) {
    for byte in value.to_le_bytes() {
        observe_base_ext_air(
            bus,
            builder,
            proof_idx,
            tidx,
            AB::Expr::from_u8(byte),
            enabled.clone(),
        );
    }
}

fn observe_bytes_air<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Var,
    tidx: &mut AB::Expr,
    bytes: &[u8],
    enabled: AB::Expr,
) {
    observe_u64_bytes_air(
        bus,
        builder,
        proof_idx,
        tidx,
        bytes.len() as u64,
        enabled.clone(),
    );
    for &byte in bytes {
        observe_base_ext_air(
            bus,
            builder,
            proof_idx,
            tidx,
            AB::Expr::from_u8(byte),
            enabled.clone(),
        );
    }
}

fn observe_backend_bytes_air<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Var,
    tidx: &mut AB::Expr,
    bytes: &[u8],
    enabled: AB::Expr,
) {
    observe_base_ext_air(
        bus,
        builder,
        proof_idx,
        tidx,
        AB::Expr::from_usize(bytes.len()),
        enabled.clone(),
    );
    for &byte in bytes {
        observe_base_ext_air(
            bus,
            builder,
            proof_idx,
            tidx,
            AB::Expr::from_u8(byte),
            enabled.clone(),
        );
    }
}

#[cfg(test)]
mod tests {
    use std::panic::AssertUnwindSafe;

    use openvm_stark_backend::{
        air_builders::{debug::check_constraints, symbolic::get_symbolic_builder},
        interaction::BusIndex,
        keygen::types::TraceWidth,
    };
    use p3_field::PrimeCharacteristicRing;

    use super::*;

    fn profile() -> ReducedSwirlSourceProfile {
        ReducedSwirlSourceProfile {
            maximum_sources: 4,
            maximum_roots_per_source: 3,
            maximum_openings_per_source: 8,
            l_skip: 1,
            n_stack: 2,
            log_blowup: 1,
            log_commit_rows_per_query: 1,
        }
    }

    fn record(seed: u32, widths: &[usize]) -> ReducedSwirlSourceRecord {
        let mut next = seed;
        let mut field = || {
            next += 1;
            F::from_u32(next)
        };
        let roots = widths
            .iter()
            .map(|_| core::array::from_fn(|_| field()))
            .collect::<Vec<_>>();
        let stacking_point = (0..profile().log_message_len())
            .map(|_| EF::from_u32(next + 17))
            .collect::<Vec<_>>();
        let stacking_openings = widths
            .iter()
            .map(|&width| {
                (0..width)
                    .map(|_| {
                        next += 1;
                        EF::from_u32(next + 31)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        ReducedSwirlSourceRecord {
            roots,
            widths: widths.to_vec(),
            stacking_point,
            stacking_openings,
            mu: EF::from_u32(seed + 91),
        }
    }

    fn air() -> ReducedSwirlSourceAir {
        let mut bus = 700usize;
        let mut next = || {
            let result = bus as BusIndex;
            bus += 1;
            result
        };
        ReducedSwirlSourceAir {
            profile: profile(),
            transcript_bus: TranscriptBus::new(next()),
            commitments_bus: CommitmentsBus::new(next()),
            stacking_endpoint_bus: NativeReductionEndpointInputBus::new(next()),
            root_width_bus: ReducedSwirlSourceRootWidthBus::new(next()),
            root_export_bus: ReducedSwirlSourceRootBus::new(next()),
            point_export_bus: ReducedSwirlSourcePointBus::new(next()),
            opening_export_bus: ReducedSwirlSourceOpeningBus::new(next()),
            beta_export_bus: ReducedSwirlSourceBetaBus::new(next()),
            claim_export_bus: ReducedSwirlSourceClaimBus::new(next()),
            export_lookup_count: 1,
        }
    }

    fn check(air: &ReducedSwirlSourceAir, trace: &RowMajorMatrix<F>) {
        check_constraints::<_, BabyBearPoseidon2Config>(
            air,
            "ReducedSwirlSourceAir",
            &None,
            &[trace.as_view()],
            &[],
        );
    }

    #[test]
    fn transcript_and_claim_match_the_backend_order() {
        let records = vec![record(10, &[2, 1]), record(100, &[1, 3, 1])];
        let artifacts = generate_reduced_swirl_source_trace(profile(), &records).unwrap();
        assert_eq!(artifacts.trace.height(), profile().maximum_sources);
        assert_eq!(artifacts.claims.len(), records.len());
        for (record, (claim, log)) in records
            .iter()
            .zip(artifacts.claims.iter().zip(&artifacts.transcript_logs))
        {
            assert_eq!(claim.roots, record.roots);
            assert_eq!(claim.widths, record.widths);
            assert!(claim.alpha.iter().all(|value| *value == EF::ZERO));
            assert_eq!(
                claim.beta,
                record
                    .stacking_point
                    .iter()
                    .rev()
                    .copied()
                    .collect::<Vec<_>>()
            );
            assert_eq!(log.samples().iter().filter(|&&sample| sample).count(), D_EF);
            assert!(log.samples()[log.len() - D_EF..]
                .iter()
                .all(|sample| *sample));
            assert_eq!(
                log.values()[log.len() - D_EF..],
                claim.theta.as_basis_coefficients_slice()[..]
            );
            assert_eq!(
                claim.eta,
                fold_openings(
                    &record
                        .stacking_openings
                        .iter()
                        .flatten()
                        .copied()
                        .collect::<Vec<_>>(),
                    claim.theta,
                )
            );
        }
        check(&air(), &artifacts.trace);
    }

    #[test]
    fn active_prefix_and_inactive_padding_are_constrained() {
        let air = air();
        let honest =
            generate_reduced_swirl_source_trace(profile(), &[record(1, &[1, 2]), record(30, &[2])])
                .unwrap()
                .trace;
        check(&air, &honest);
        let width = profile().trace_width();
        let reject = |mutate: &dyn Fn(&mut RowMajorMatrix<F>)| {
            let mut changed = honest.clone();
            mutate(&mut changed);
            assert!(std::panic::catch_unwind(AssertUnwindSafe(|| check(&air, &changed))).is_err());
        };
        reject(&|trace| trace.values[2 * width + ACTIVE] = F::ONE);
        reject(&|trace| trace.values[2 * width + SOURCE] = F::ONE);
        reject(&|trace| {
            let second_root = root_slot(profile(), 1).start;
            trace.values[second_root + ROOT_ACTIVE] = F::ZERO;
        });
        reject(&|trace| {
            let first_opening = opening_slot(profile(), 0).start;
            trace.values[first_opening + OPENING_COLUMN] = F::ONE;
        });
        reject(&|trace| trace.values[ETA] += F::ONE);
    }

    #[test]
    fn malformed_host_shapes_fail_closed() {
        let mut bad_width = record(1, &[2]);
        bad_width.widths[0] = 3;
        assert!(generate_reduced_swirl_source_trace(profile(), &[bad_width]).is_err());
        let too_many = vec![record(1, &[1]); profile().maximum_sources + 1];
        assert!(generate_reduced_swirl_source_trace(profile(), &too_many).is_err());
        let mut bad_point = record(1, &[1]);
        bad_point.stacking_point.pop();
        assert!(generate_reduced_swirl_source_trace(profile(), &[bad_point]).is_err());
    }

    #[test]
    fn relation_stays_within_the_recursion_degree_bound() {
        let air = air();
        let symbolic = get_symbolic_builder(
            &air,
            &TraceWidth {
                preprocessed: None,
                cached_mains: Vec::new(),
                common_main: air.width(),
            },
        )
        .constraints();
        assert!(symbolic.max_constraint_degree() <= 4);
    }
}

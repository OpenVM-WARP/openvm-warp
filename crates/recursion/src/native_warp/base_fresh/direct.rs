use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{
    utils::assert_array_eq, ColumnsAir, StructReflection, StructReflectionHelper,
};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    interaction::InteractionBuilder,
    warp_accum::{STACKED_RS_COMMITMENT_DOMAIN_TAG, STACKED_RS_COMMITMENT_VERSION},
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{DIGEST_SIZE, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};
use p3_maybe_rayon::prelude::*;

use crate::{
    bus::TranscriptBus,
    native_warp::{
        bus::{
            NativeAuthenticatedShiftBus, NativeAuthenticatedShiftMessage, NativeDirectFreshRootBus,
            NativeDirectFreshRootMessage, NativeDirectFreshSourceBus,
            NativeDirectFreshSourceMessage, NativeFreshCountBus, NativeFreshCountMessage,
            NativeFreshDigestElementBus, NativeFreshDigestElementMessage,
            NativeFreshSourceActivityBus, NativeFreshSourceActivityMessage, NativeLeafValueBus,
            NativeLeafValueMessage, NativeMerkleRootBus, NativeMerkleRootMessage,
            NativeShiftIndexBus, NativeShiftIndexMessage,
        },
        ext::{ext_field_add, ext_field_multiply, ext_field_multiply_scalar},
    },
};

const DIRECT_HEADER_FIXED_FIELDS: usize = 7;
const DIRECT_ROOT_FIELDS: usize = 2 + DIGEST_SIZE;

/// Number of consecutive authenticated base-field columns folded by one
/// direct-projection AIR row.
///
/// Every packed slot carries an explicit constrained power of `theta`. This
/// keeps each conditional term cubic and the power recurrence cubic even when
/// thirty-two columns share a row; deriving all powers as one expression would
/// instead make the constraint degree grow with the packing factor.
pub const NATIVE_DIRECT_FRESH_PROJECTION_PACKING: usize = 32;
const NATIVE_DIRECT_FRESH_PROJECTION_EXTRA: usize = NATIVE_DIRECT_FRESH_PROJECTION_PACKING - 1;

#[must_use]
pub const fn native_direct_fresh_header_width(max_roots: usize) -> usize {
    DIRECT_HEADER_FIXED_FIELDS + D_EF + max_roots * DIRECT_ROOT_FIELDS
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeDirectFreshCommitmentCols<T> {
    pub row_active: T,
    pub source_active: T,
    pub is_first: T,
    pub is_last: T,
    pub source: T,
    pub running_count: T,
    pub root_count: T,
    pub root_count_inverse: T,
    pub proof_idx: T,
    pub commitment_tidx: T,
    pub l_skip: T,
    pub native_log_message_len: T,
    pub log_message_len: T,
    pub log_codeword_len: T,
    pub rows_per_query: T,
    pub theta: [T; D_EF],
    pub first_tree_id: T,
}

/// Binds each original-root source descriptor to the exact VACC transcript.
/// Root rows are handled separately so the AIR width is independent of the
/// number and width of application matrices.
#[derive(ColumnsAir)]
#[columns_via(NativeDirectFreshCommitmentCols<u8>)]
pub struct NativeDirectFreshCommitmentAir {
    pub transcript_bus: TranscriptBus,
    pub source_bus: NativeDirectFreshSourceBus,
    pub activity_bus: NativeFreshSourceActivityBus,
    pub fresh_count_bus: NativeFreshCountBus,
    pub digest_element_bus: Option<NativeFreshDigestElementBus>,
    pub max_fresh: usize,
    pub max_roots: usize,
    pub shift_count: usize,
    pub expected_log_message_len: usize,
    pub expected_log_codeword_len: usize,
    pub expected_rows_per_query: usize,
    pub tree_source_stride: usize,
    pub first_tree_id: usize,
    pub digest_metadata_len: usize,
    pub digest_source_width: usize,
    pub digest_alpha_len: usize,
    pub digest_beta_len: usize,
    pub claim_rows_per_source: usize,
}

impl BaseAirWithPublicValues<F> for NativeDirectFreshCommitmentAir {}
impl PartitionedBaseAir<F> for NativeDirectFreshCommitmentAir {}
impl BaseAir<F> for NativeDirectFreshCommitmentAir {
    fn width(&self) -> usize {
        NativeDirectFreshCommitmentCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeDirectFreshCommitmentAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("direct fresh commitment row");
        let next_row = main.row_slice(1).expect("direct fresh commitment next row");
        let local: &NativeDirectFreshCommitmentCols<AB::Var> = (*local_row).borrow();
        let next: &NativeDirectFreshCommitmentCols<AB::Var> = (*next_row).borrow();

        for flag in [
            local.row_active,
            local.source_active,
            local.is_first,
            local.is_last,
        ] {
            builder.assert_bool(flag);
        }
        builder.when(local.is_first).assert_one(local.row_active);
        builder.when(local.is_last).assert_one(local.row_active);
        builder.when_first_row().assert_one(local.row_active);
        builder.when_first_row().assert_one(local.is_first);
        // A physically merged reduced-SWIRL batch contains one canonical
        // source table per WARP call. `is_first` is therefore a real reset
        // marker, not merely a global-row hint.
        builder.when(local.is_first).assert_one(local.source_active);
        builder.when(local.is_first).assert_zero(local.source);
        builder.when(local.is_first).assert_eq(
            local.first_tree_id,
            AB::Expr::from_usize(self.first_tree_id),
        );
        builder
            .when(local.is_first)
            .assert_eq(local.running_count, local.source_active);
        builder
            .when(local.row_active * local.is_last)
            .assert_eq(local.source, AB::Expr::from_usize(self.max_fresh - 1));
        builder
            .when_last_row()
            .when(local.row_active)
            .assert_one(local.is_last);
        let enabled = AB::Expr::from(local.source_active);

        let mut transition = builder.when_transition();
        transition.assert_bool(local.row_active - next.row_active);
        transition.assert_eq(
            local.row_active - next.row_active,
            AB::Expr::from(local.is_last) - next.is_first,
        );
        transition.when(next.is_first).assert_one(local.is_last);
        let mut active = transition.when(AB::Expr::from(next.row_active) - next.is_first);
        active.assert_eq(next.source, local.source + AB::F::ONE);
        active.assert_eq(next.proof_idx, local.proof_idx);
        active.assert_eq(next.running_count, local.running_count + next.source_active);
        active.assert_bool(local.source_active - next.source_active);
        active.assert_eq(
            next.first_tree_id,
            local.first_tree_id + AB::Expr::from_usize(self.tree_source_stride),
        );
        let prior_events =
            AB::Expr::from_usize(9) + local.root_count * AB::Expr::from_usize(1 + DIGEST_SIZE);
        active.assert_eq(
            next.commitment_tidx,
            local.commitment_tidx + enabled.clone() * prior_events * AB::Expr::from_usize(D_EF),
        );

        builder
            .when(enabled.clone())
            .assert_one(local.root_count * local.root_count_inverse);
        builder.when(enabled.clone()).assert_eq(
            local.log_message_len,
            AB::Expr::from_usize(self.expected_log_message_len),
        );
        builder.when(enabled.clone()).assert_eq(
            local.log_codeword_len,
            AB::Expr::from_usize(self.expected_log_codeword_len),
        );
        builder.when(enabled.clone()).assert_eq(
            local.rows_per_query,
            AB::Expr::from_usize(self.expected_rows_per_query),
        );
        let inactive = local.row_active * (AB::Expr::ONE - enabled.clone());
        for value in [
            local.root_count,
            local.root_count_inverse,
            local.l_skip,
            local.native_log_message_len,
            local.log_message_len,
            local.log_codeword_len,
            local.rows_per_query,
        ] {
            builder.when(inactive.clone()).assert_zero(value);
        }
        for value in local.theta {
            builder.when(inactive.clone()).assert_zero(value);
        }

        self.fresh_count_bus.lookup_key(
            builder,
            NativeFreshCountMessage {
                count: local.running_count.into(),
            },
            local.row_active
                * local.is_last
                * AB::Expr::from_usize(1 + usize::from(self.digest_element_bus.is_some())),
        );
        self.source_bus.add_key_with_lookups(
            builder,
            NativeDirectFreshSourceMessage {
                source: local.source.into(),
                active: local.source_active.into(),
                root_count: local.root_count.into(),
                first_tree_id: local.first_tree_id.into(),
                commitment_tidx: local.commitment_tidx.into(),
                theta: local.theta.map(Into::into),
            },
            local.row_active
                * (AB::Expr::from_usize(self.max_roots)
                    + enabled.clone() * AB::Expr::from_usize(self.shift_count.saturating_add(1))),
        );
        self.activity_bus.add_key_with_lookups(
            builder,
            NativeFreshSourceActivityMessage {
                source: local.source.into(),
                active: local.source_active.into(),
            },
            local.row_active
                * AB::Expr::from_usize(self.claim_rows_per_source)
                * AB::Expr::from_bool(self.digest_element_bus.is_some()),
        );

        observe_lifted_base_at(
            &self.transcript_bus,
            builder,
            local.proof_idx,
            local.commitment_tidx.into(),
            AB::Expr::from_u64(STACKED_RS_COMMITMENT_DOMAIN_TAG),
            enabled.clone(),
        );
        observe_lifted_base_at(
            &self.transcript_bus,
            builder,
            local.proof_idx,
            local.commitment_tidx + AB::Expr::from_usize(D_EF),
            AB::Expr::from_u64(STACKED_RS_COMMITMENT_VERSION),
            enabled.clone(),
        );
        for (offset, field) in [
            local.root_count,
            local.l_skip,
            local.native_log_message_len,
            local.log_message_len,
            local.log_codeword_len,
            local.rows_per_query,
        ]
        .into_iter()
        .enumerate()
        {
            observe_lifted_base_at(
                &self.transcript_bus,
                builder,
                local.proof_idx,
                local.commitment_tidx + AB::Expr::from_usize((offset + 2) * D_EF),
                field.into(),
                enabled.clone(),
            );
        }
        let theta_tidx = local.commitment_tidx
            + (AB::Expr::from_usize(8) + local.root_count * AB::Expr::from_usize(1 + DIGEST_SIZE))
                * AB::Expr::from_usize(D_EF);
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            theta_tidx,
            local.theta.map(Into::into),
            enabled.clone(),
        );

        if let Some(bus) = self.digest_element_bus {
            let first = local.row_active * local.is_first;
            for (index, value) in [
                AB::Expr::from_u64(
                    openvm_stark_backend::native_warp::NATIVE_WARP_FRESH_INPUTS_DIGEST_TAG,
                ),
                AB::Expr::from_usize(
                    openvm_stark_backend::native_warp::NATIVE_WARP_FRESH_INPUTS_DIGEST_VERSION,
                ),
                AB::Expr::from_usize(self.max_fresh),
                AB::Expr::from_usize(self.digest_alpha_len),
                AB::Expr::from_usize(self.digest_beta_len),
            ]
            .into_iter()
            .enumerate()
            {
                bus.send(
                    builder,
                    NativeFreshDigestElementMessage {
                        index: AB::Expr::from_usize(index),
                        value,
                    },
                    first.clone(),
                );
            }
            bus.send(
                builder,
                NativeFreshDigestElementMessage {
                    index: AB::Expr::from_usize(5),
                    value: local.running_count.into(),
                },
                local.row_active * local.is_last,
            );
            let base = AB::Expr::from_usize(self.digest_metadata_len)
                + local.source * AB::Expr::from_usize(self.digest_source_width);
            let values: Vec<AB::Expr> = core::iter::once(enabled.clone())
                .chain([
                    local.root_count.into(),
                    local.l_skip.into(),
                    local.native_log_message_len.into(),
                    local.log_message_len.into(),
                    local.log_codeword_len.into(),
                    local.rows_per_query.into(),
                ])
                .chain(local.theta.map(Into::into))
                .collect();
            for (offset, value) in values.into_iter().enumerate() {
                bus.send(
                    builder,
                    NativeFreshDigestElementMessage {
                        index: base.clone() + AB::Expr::from_usize(offset),
                        value,
                    },
                    local.row_active,
                );
            }
        }
    }
}

fn observe_lifted_base_at<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    builder: &mut AB,
    proof_idx: AB::Var,
    tidx: AB::Expr,
    value: AB::Expr,
    enabled: AB::Expr,
) {
    bus.observe_ext(
        builder,
        proof_idx,
        tidx,
        core::array::from_fn(|limb| {
            if limb == 0 {
                value.clone()
            } else {
                AB::Expr::ZERO
            }
        }),
        enabled,
    );
}

#[derive(Clone, Debug)]
pub struct NativeDirectFreshCommitmentTraceInput {
    pub source_active: bool,
    pub proof_idx: usize,
    pub commitment_tidx: usize,
    pub root_count: usize,
    pub l_skip: usize,
    pub native_log_message_len: usize,
    pub log_message_len: usize,
    pub log_codeword_len: usize,
    pub rows_per_query: usize,
    pub theta: EF,
    pub first_tree_id: usize,
}

pub fn generate_native_direct_fresh_commitment_trace(
    inputs: &[NativeDirectFreshCommitmentTraceInput],
    max_fresh: usize,
    max_roots: usize,
    tree_source_stride: usize,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if inputs.len() > max_fresh
        || inputs.is_empty()
        || max_fresh == 0
        || max_roots == 0
        || tree_source_stride == 0
    {
        return None;
    }
    let height = required_height.unwrap_or_else(|| max_fresh.next_power_of_two());
    if height < max_fresh {
        return None;
    }
    let width = NativeDirectFreshCommitmentCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    let mut running_count = 0usize;
    let proof_idx = inputs[0].proof_idx;
    let first_tree_id = inputs[0].first_tree_id;
    let mut next_tidx = inputs[0].commitment_tidx;
    for source in 0..max_fresh {
        let cols: &mut NativeDirectFreshCommitmentCols<F> =
            values[source * width..(source + 1) * width].borrow_mut();
        cols.row_active = F::ONE;
        cols.is_first = F::from_bool(source == 0);
        cols.is_last = F::from_bool(source + 1 == max_fresh);
        cols.source = F::from_usize(source);
        if let Some(input) = inputs.get(source) {
            if !input.source_active
                || input.root_count == 0
                || input.root_count > max_roots
                || input.proof_idx != proof_idx
                || input.commitment_tidx != next_tidx
                || input.first_tree_id
                    != first_tree_id.checked_add(source.checked_mul(tree_source_stride)?)?
            {
                return None;
            }
            running_count += 1;
            cols.source_active = F::ONE;
            cols.root_count = F::from_usize(input.root_count);
            cols.root_count_inverse = cols.root_count.inverse();
            cols.proof_idx = F::from_usize(input.proof_idx);
            cols.commitment_tidx = F::from_usize(input.commitment_tidx);
            cols.l_skip = F::from_usize(input.l_skip);
            cols.native_log_message_len = F::from_usize(input.native_log_message_len);
            cols.log_message_len = F::from_usize(input.log_message_len);
            cols.log_codeword_len = F::from_usize(input.log_codeword_len);
            cols.rows_per_query = F::from_usize(input.rows_per_query);
            cols.theta
                .copy_from_slice(input.theta.as_basis_coefficients_slice());
            cols.first_tree_id = F::from_usize(input.first_tree_id);
            next_tidx = next_tidx.checked_add(
                (9 + input.root_count.checked_mul(1 + DIGEST_SIZE)?).checked_mul(D_EF)?,
            )?;
        } else {
            cols.proof_idx = F::from_usize(proof_idx);
            cols.commitment_tidx = F::from_usize(next_tidx);
            cols.first_tree_id =
                F::from_usize(first_tree_id.checked_add(source.checked_mul(tree_source_stride)?)?);
        }
        cols.running_count = F::from_usize(running_count);
    }
    Some(RowMajorMatrix::new(values, width))
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeDirectFreshRootCols<T> {
    pub row_active: T,
    pub source_active: T,
    pub root_active: T,
    pub is_first: T,
    pub is_last: T,
    pub is_last_root_slot: T,
    pub last_root_slot_inverse: T,
    pub source: T,
    pub root_ordinal: T,
    pub running_roots: T,
    pub root_count: T,
    pub proof_idx: T,
    pub commitment_tidx: T,
    pub first_tree_id: T,
    pub tree_id: T,
    pub width: T,
    pub width_inverse: T,
    pub theta: [T; D_EF],
    pub root: [T; DIGEST_SIZE],
}

#[derive(ColumnsAir)]
#[columns_via(NativeDirectFreshRootCols<u8>)]
pub struct NativeDirectFreshRootAir {
    pub transcript_bus: TranscriptBus,
    pub source_bus: NativeDirectFreshSourceBus,
    pub root_bus: NativeDirectFreshRootBus,
    pub merkle_root_bus: NativeMerkleRootBus,
    pub digest_element_bus: Option<NativeFreshDigestElementBus>,
    pub max_fresh: usize,
    pub max_roots: usize,
    pub shift_count: usize,
    pub root_tree_stride: usize,
    pub outer_depth: usize,
    pub digest_metadata_len: usize,
    pub digest_source_width: usize,
}

impl BaseAirWithPublicValues<F> for NativeDirectFreshRootAir {}
impl PartitionedBaseAir<F> for NativeDirectFreshRootAir {}
impl BaseAir<F> for NativeDirectFreshRootAir {
    fn width(&self) -> usize {
        NativeDirectFreshRootCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeDirectFreshRootAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("direct fresh root row");
        let next_row = main.row_slice(1).expect("direct fresh root next row");
        let local: &NativeDirectFreshRootCols<AB::Var> = (*local_row).borrow();
        let next: &NativeDirectFreshRootCols<AB::Var> = (*next_row).borrow();
        for flag in [
            local.row_active,
            local.source_active,
            local.root_active,
            local.is_first,
            local.is_last,
            local.is_last_root_slot,
        ] {
            builder.assert_bool(flag);
        }
        builder.when(local.is_first).assert_one(local.row_active);
        builder.when(local.is_last).assert_one(local.row_active);
        builder.when_first_row().assert_one(local.row_active);
        builder.when_first_row().assert_one(local.is_first);
        builder.when(local.is_first).assert_zero(local.source);
        builder.when(local.is_first).assert_zero(local.root_ordinal);
        builder
            .when(local.is_first)
            .assert_eq(local.running_roots, local.root_active);
        builder
            .when_last_row()
            .when(local.row_active)
            .assert_one(local.is_last);
        builder
            .when(local.row_active * local.is_last)
            .assert_eq(local.source, AB::Expr::from_usize(self.max_fresh - 1));
        builder
            .when(local.row_active * local.is_last)
            .assert_eq(local.root_ordinal, AB::Expr::from_usize(self.max_roots - 1));
        constrain_zero_flag(
            builder,
            local.row_active,
            AB::Expr::from_usize(self.max_roots - 1) - local.root_ordinal,
            local.last_root_slot_inverse,
            local.is_last_root_slot,
        );
        builder.assert_eq(
            local.root_active * (AB::Expr::ONE - local.source_active),
            AB::Expr::ZERO,
        );

        let mut transition = builder.when_transition();
        transition.assert_bool(local.row_active - next.row_active);
        transition.assert_eq(
            local.row_active - next.row_active,
            AB::Expr::from(local.is_last) - next.is_first,
        );
        transition.when(next.is_first).assert_one(local.is_last);
        let continuing = AB::Expr::from(next.row_active) - next.is_first;
        let same_source = continuing.clone() * (AB::Expr::ONE - local.is_last_root_slot);
        let source_boundary = continuing * local.is_last_root_slot;
        let mut transition = builder.when_transition();
        let mut same = transition.when(same_source);
        same.assert_eq(next.source, local.source);
        same.assert_eq(next.root_ordinal, local.root_ordinal + AB::F::ONE);
        same.assert_eq(next.source_active, local.source_active);
        same.assert_eq(next.root_count, local.root_count);
        same.assert_eq(next.proof_idx, local.proof_idx);
        same.assert_eq(next.commitment_tidx, local.commitment_tidx);
        same.assert_eq(next.first_tree_id, local.first_tree_id);
        same.assert_eq(next.running_roots, local.running_roots + next.root_active);
        same.assert_bool(local.root_active - next.root_active);
        for limb in 0..D_EF {
            same.assert_eq(next.theta[limb], local.theta[limb]);
        }
        let mut transition = builder.when_transition();
        let mut boundary = transition.when(source_boundary);
        boundary.assert_eq(next.source, local.source + AB::F::ONE);
        boundary.assert_zero(next.root_ordinal);
        boundary.assert_eq(next.running_roots, next.root_active);
        builder
            .when(local.row_active * local.is_last_root_slot)
            .assert_eq(local.running_roots, local.root_count);

        self.source_bus.lookup_key(
            builder,
            NativeDirectFreshSourceMessage {
                source: local.source.into(),
                active: local.source_active.into(),
                root_count: local.root_count.into(),
                first_tree_id: local.first_tree_id.into(),
                commitment_tidx: local.commitment_tidx.into(),
                theta: local.theta.map(Into::into),
            },
            local.row_active,
        );
        builder.when(local.row_active).assert_eq(
            local.tree_id,
            local.first_tree_id + local.root_ordinal * AB::Expr::from_usize(self.root_tree_stride),
        );
        builder
            .when(local.row_active * local.root_active)
            .assert_one(local.width * local.width_inverse);
        let inactive = local.row_active * (AB::Expr::ONE - local.root_active);
        for value in [local.width, local.width_inverse] {
            builder.when(inactive.clone()).assert_zero(value);
        }
        for value in local.root {
            builder.when(inactive.clone()).assert_zero(value);
        }

        let root_tidx = local.commitment_tidx
            + (AB::Expr::from_usize(8)
                + local.root_ordinal * AB::Expr::from_usize(1 + DIGEST_SIZE))
                * AB::Expr::from_usize(D_EF);
        let width_ext: [AB::Expr; D_EF] = core::array::from_fn(|limb| {
            if limb == 0 {
                local.width.into()
            } else {
                AB::Expr::ZERO
            }
        });
        self.transcript_bus.observe_ext(
            builder,
            local.proof_idx,
            root_tidx.clone(),
            width_ext,
            local.root_active,
        );
        for (limb, value) in local.root.into_iter().enumerate() {
            observe_lifted_base_at(
                &self.transcript_bus,
                builder,
                local.proof_idx,
                root_tidx.clone() + AB::Expr::from_usize((limb + 1) * D_EF),
                value.into(),
                local.root_active.into(),
            );
        }
        self.root_bus.add_key_with_lookups(
            builder,
            NativeDirectFreshRootMessage {
                source: local.source.into(),
                root_ordinal: local.root_ordinal.into(),
                tree_id: local.tree_id.into(),
                width: local.width.into(),
                root: local.root.map(Into::into),
            },
            local.root_active * AB::Expr::from_usize(self.shift_count.saturating_add(1)),
        );
        self.merkle_root_bus.receive(
            builder,
            NativeMerkleRootMessage {
                proof_idx: local.proof_idx.into(),
                tree_id: local.tree_id.into(),
                depth: AB::Expr::from_usize(self.outer_depth),
                digest: local.root.map(Into::into),
            },
            local.root_active,
        );

        if let Some(bus) = self.digest_element_bus {
            let base = AB::Expr::from_usize(self.digest_metadata_len)
                + local.source * AB::Expr::from_usize(self.digest_source_width)
                + AB::Expr::from_usize(DIRECT_HEADER_FIXED_FIELDS + D_EF)
                + local.root_ordinal * AB::Expr::from_usize(DIRECT_ROOT_FIELDS);
            let values: Vec<AB::Expr> = core::iter::once(local.root_active.into())
                .chain(core::iter::once(local.width.into()))
                .chain(local.root.map(Into::into))
                .collect();
            for (offset, value) in values.into_iter().enumerate() {
                bus.send(
                    builder,
                    NativeFreshDigestElementMessage {
                        index: base.clone() + AB::Expr::from_usize(offset),
                        value,
                    },
                    local.row_active,
                );
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct NativeDirectFreshRootTraceInput {
    pub source_active: bool,
    pub root_count: usize,
    pub proof_idx: usize,
    pub commitment_tidx: usize,
    pub first_tree_id: usize,
    pub theta: EF,
    pub roots: Vec<[F; DIGEST_SIZE]>,
    pub widths: Vec<usize>,
}

pub fn generate_native_direct_fresh_root_trace(
    inputs: &[NativeDirectFreshRootTraceInput],
    max_fresh: usize,
    max_roots: usize,
    root_tree_stride: usize,
    tree_source_stride: usize,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if max_fresh == 0
        || max_roots == 0
        || root_tree_stride == 0
        || tree_source_stride != max_roots.checked_mul(root_tree_stride)?
        || inputs.is_empty()
        || inputs.len() > max_fresh
    {
        return None;
    }
    let valid_rows = max_fresh.checked_mul(max_roots)?;
    let height = required_height.unwrap_or_else(|| valid_rows.next_power_of_two());
    if height < valid_rows {
        return None;
    }
    let width = NativeDirectFreshRootCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    let proof_idx = inputs[0].proof_idx;
    let first_tree_id = inputs[0].first_tree_id;
    let mut padded_tidx = inputs[0].commitment_tidx;
    for (source, input) in inputs.iter().enumerate() {
        if !input.source_active
            || input.root_count == 0
            || input.root_count > max_roots
            || input.roots.len() != input.root_count
            || input.widths.len() != input.root_count
            || input.widths.contains(&0)
            || input.proof_idx != proof_idx
            || input.commitment_tidx != padded_tidx
            || input.first_tree_id
                != first_tree_id.checked_add(source.checked_mul(tree_source_stride)?)?
        {
            return None;
        }
        padded_tidx = padded_tidx
            .checked_add((9 + input.root_count.checked_mul(1 + DIGEST_SIZE)?).checked_mul(D_EF)?)?;
    }
    for source in 0..max_fresh {
        let input = inputs.get(source);
        for root_ordinal in 0..max_roots {
            let index = source * max_roots + root_ordinal;
            let cols: &mut NativeDirectFreshRootCols<F> =
                values[index * width..(index + 1) * width].borrow_mut();
            cols.row_active = F::ONE;
            cols.is_first = F::from_bool(index == 0);
            cols.is_last = F::from_bool(index + 1 == valid_rows);
            cols.is_last_root_slot = F::from_bool(root_ordinal + 1 == max_roots);
            cols.last_root_slot_inverse = if root_ordinal + 1 == max_roots {
                F::ZERO
            } else {
                F::from_usize(max_roots - 1 - root_ordinal).inverse()
            };
            cols.source = F::from_usize(source);
            cols.root_ordinal = F::from_usize(root_ordinal);
            cols.tree_id = F::from_usize(
                first_tree_id
                    .checked_add(source.checked_mul(tree_source_stride)?)?
                    .checked_add(root_ordinal.checked_mul(root_tree_stride)?)?,
            );
            if let Some(input) = input {
                cols.source_active = F::ONE;
                cols.root_count = F::from_usize(input.root_count);
                cols.running_roots = F::from_usize(root_ordinal.min(input.root_count - 1) + 1);
                cols.proof_idx = F::from_usize(input.proof_idx);
                cols.commitment_tidx = F::from_usize(input.commitment_tidx);
                cols.first_tree_id = F::from_usize(input.first_tree_id);
                cols.tree_id = F::from_usize(input.first_tree_id + root_ordinal * root_tree_stride);
                cols.theta
                    .copy_from_slice(input.theta.as_basis_coefficients_slice());
                if root_ordinal < input.root_count {
                    cols.root_active = F::ONE;
                    cols.width = F::from_usize(input.widths[root_ordinal]);
                    cols.width_inverse = cols.width.inverse();
                    cols.root = input.roots[root_ordinal];
                }
            } else {
                cols.proof_idx = F::from_usize(proof_idx);
                cols.commitment_tidx = F::from_usize(padded_tidx);
                cols.first_tree_id = F::from_usize(
                    first_tree_id.checked_add(source.checked_mul(tree_source_stride)?)?,
                );
            }
        }
    }
    Some(RowMajorMatrix::new(values, width))
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeDirectFreshProjectionCols<T> {
    pub active: T,
    pub proof_idx: T,
    /// Prefix flags for every packed slot after the first. A slot can be active only
    /// when the preceding slot is active, making the packed representation
    /// canonical.
    pub extra_active: [T; NATIVE_DIRECT_FRESH_PROJECTION_EXTRA],
    pub is_first: T,
    pub is_last: T,
    pub is_first_column: T,
    pub is_last_column: T,
    pub is_first_root: T,
    pub is_last_root: T,
    pub is_first_shift: T,
    pub is_last_shift: T,
    pub is_sequence_first: T,
    pub is_sequence_last: T,
    pub is_same_root: T,
    pub is_next_root: T,
    pub is_next_shift: T,
    pub is_next_source: T,
    pub column_inverse: T,
    pub last_column_inverse: T,
    pub root_inverse: T,
    pub last_root_inverse: T,
    pub shift_inverse: T,
    pub last_shift_inverse: T,
    pub source: T,
    pub shift: T,
    pub root_ordinal: T,
    pub root_count: T,
    pub column: T,
    pub width: T,
    pub first_tree_id: T,
    pub tree_id: T,
    pub root: [T; DIGEST_SIZE],
    pub commitment_tidx: T,
    pub flat_index: T,
    pub query_index: T,
    pub row_offset: T,
    pub theta: [T; D_EF],
    pub theta_power: [T; D_EF],
    /// Powers for every packed slot after the first. Each is constrained from the
    /// preceding power by one extension-field multiplication.
    pub extra_theta_powers: [[T; D_EF]; NATIVE_DIRECT_FRESH_PROJECTION_EXTRA],
    /// `theta_power` advanced by every active column consumed by this row.
    /// Keeping this value explicit lets transition constraints remain at
    /// degree two after the row-local selection is checked.
    pub theta_power_after: [T; D_EF],
    pub accumulated_before: [T; D_EF],
    pub base_value: T,
    pub extra_base_values: [T; NATIVE_DIRECT_FRESH_PROJECTION_EXTRA],
    pub accumulated_after: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(NativeDirectFreshProjectionCols<u8>)]
pub struct NativeDirectFreshProjectionAir {
    pub source_bus: NativeDirectFreshSourceBus,
    pub root_bus: NativeDirectFreshRootBus,
    pub shift_index_bus: NativeShiftIndexBus,
    pub leaf_value_bus: NativeLeafValueBus,
    pub authenticated_bus: NativeAuthenticatedShiftBus,
    pub shift_count: usize,
    pub query_stride: usize,
    pub root_tree_stride: usize,
    /// First source represented by this independently committed trace shard.
    ///
    /// Projection rows are ordered source-major. Splitting only between
    /// sources therefore preserves every within-source transition while the
    /// source bus preserves the union of the shards.
    pub first_source: usize,
    /// Later fixed-capacity shards may be empty when the final VACC batch does
    /// not fill the family's maximum input arity.
    pub allow_empty: bool,
}

impl BaseAirWithPublicValues<F> for NativeDirectFreshProjectionAir {}
impl PartitionedBaseAir<F> for NativeDirectFreshProjectionAir {}
impl BaseAir<F> for NativeDirectFreshProjectionAir {
    fn width(&self) -> usize {
        NativeDirectFreshProjectionCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeDirectFreshProjectionAir
where
    <AB::Expr as PrimeCharacteristicRing>::PrimeSubfield:
        p3_field::extension::BinomiallyExtendable<D_EF>,
{
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let local_row = main.row_slice(0).expect("direct fresh projection row");
        let next_row = main.row_slice(1).expect("direct fresh projection next row");
        let local: &NativeDirectFreshProjectionCols<AB::Var> = (*local_row).borrow();
        let next: &NativeDirectFreshProjectionCols<AB::Var> = (*next_row).borrow();
        for flag in [
            local.active,
            local.is_first,
            local.is_last,
            local.is_first_column,
            local.is_last_column,
            local.is_first_root,
            local.is_last_root,
            local.is_first_shift,
            local.is_last_shift,
            local.is_sequence_first,
            local.is_sequence_last,
            local.is_same_root,
            local.is_next_root,
            local.is_next_shift,
            local.is_next_source,
        ] {
            builder.assert_bool(flag);
        }
        builder.when(local.is_first).assert_one(local.active);
        builder.when(local.is_last).assert_one(local.active);
        let mut preceding_active = AB::Expr::from(local.active);
        for (&slot_active, &base_value) in local
            .extra_active
            .iter()
            .zip(local.extra_base_values.iter())
        {
            builder.assert_bool(slot_active);
            builder.assert_zero(
                AB::Expr::from(slot_active) * (AB::Expr::ONE - preceding_active.clone()),
            );
            builder
                .when(local.active * (AB::Expr::ONE - slot_active))
                .assert_zero(base_value);
            preceding_active = slot_active.into();
        }
        let consumed_columns = local
            .extra_active
            .iter()
            .fold(AB::Expr::ONE, |count, &active| count + active);
        builder.assert_eq(
            local.is_sequence_first,
            local.active * local.is_first_column * local.is_first_root,
        );
        builder.assert_eq(
            local.is_sequence_last,
            local.active * local.is_last_column * local.is_last_root,
        );
        builder.assert_eq(
            local.is_same_root,
            local.active * (AB::Expr::ONE - local.is_last_column),
        );
        builder.assert_eq(
            local.is_next_root,
            local.active * local.is_last_column * (AB::Expr::ONE - local.is_last_root),
        );
        builder.assert_eq(
            local.is_next_shift,
            local.is_sequence_last * (AB::Expr::ONE - local.is_last_shift),
        );
        builder.assert_eq(
            local.is_next_source,
            local.is_sequence_last * local.is_last_shift * (AB::Expr::ONE - local.is_last),
        );
        builder.assert_eq(
            local.active,
            local.is_same_root
                + local.is_next_root
                + local.is_next_shift
                + local.is_next_source
                + local.is_last,
        );
        if !self.allow_empty {
            builder.when_first_row().assert_one(local.active);
        }
        builder
            .when_first_row()
            .when(local.active)
            .assert_one(local.is_first);
        builder
            .when(local.is_first)
            .assert_eq(local.source, AB::Expr::from_usize(self.first_source));
        builder.when(local.is_first).assert_zero(local.shift);
        builder.when(local.is_first).assert_zero(local.root_ordinal);
        builder.when(local.is_first).assert_zero(local.column);
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        builder.when_transition().assert_eq(
            local.active - next.active,
            AB::Expr::from(local.is_last) - next.is_first,
        );
        builder
            .when_transition()
            .when(next.is_first)
            .assert_one(local.is_last);
        builder
            .when_last_row()
            .when(local.active)
            .assert_one(local.is_last);
        builder
            .when(local.is_last)
            .assert_one(local.is_sequence_last);
        builder.when(local.is_last).assert_one(local.is_last_shift);
        constrain_zero_flag(
            builder,
            local.active,
            local.column.into(),
            local.column_inverse,
            local.is_first_column,
        );
        constrain_zero_flag(
            builder,
            local.active,
            local.width - local.column - consumed_columns.clone(),
            local.last_column_inverse,
            local.is_last_column,
        );
        constrain_zero_flag(
            builder,
            local.active,
            local.root_ordinal.into(),
            local.root_inverse,
            local.is_first_root,
        );
        constrain_zero_flag(
            builder,
            local.active,
            local.root_count - local.root_ordinal - AB::Expr::ONE,
            local.last_root_inverse,
            local.is_last_root,
        );
        constrain_zero_flag(
            builder,
            local.active,
            local.shift.into(),
            local.shift_inverse,
            local.is_first_shift,
        );
        constrain_zero_flag(
            builder,
            local.active,
            AB::Expr::from_usize(self.shift_count - 1) - local.shift,
            local.last_shift_inverse,
            local.is_last_shift,
        );

        let sequence_first = AB::Expr::from(local.is_sequence_first);
        let sequence_last = AB::Expr::from(local.is_sequence_last);
        let one: [AB::Expr; D_EF] = core::array::from_fn(|limb| {
            if limb == 0 {
                AB::Expr::ONE
            } else {
                AB::Expr::ZERO
            }
        });
        let zero: [AB::Expr; D_EF] = core::array::from_fn(|_| AB::Expr::ZERO);
        assert_array_eq(
            &mut builder.when(local.active * sequence_first.clone()),
            local.theta_power,
            one.clone(),
        );
        assert_array_eq(
            &mut builder.when(local.active * sequence_first.clone()),
            local.accumulated_before,
            zero.clone(),
        );
        let term = ext_field_multiply_scalar::<AB::Expr>(
            local.theta_power.map(Into::into),
            local.base_value.into(),
        )
        .map(|value| value * AB::Expr::from(local.active));
        let mut accumulated_after =
            ext_field_add::<AB::Expr>(local.accumulated_before.map(Into::into), term);
        let mut preceding_power = local.theta_power.map(Into::into);
        for slot in 0..NATIVE_DIRECT_FRESH_PROJECTION_EXTRA {
            let expected_power =
                ext_field_multiply::<AB::Expr>(preceding_power, local.theta.map(Into::into));
            assert_array_eq(
                &mut builder.when(local.active),
                local.extra_theta_powers[slot],
                expected_power,
            );
            let slot_term = ext_field_multiply_scalar::<AB::Expr>(
                local.extra_theta_powers[slot].map(Into::into),
                local.extra_base_values[slot].into(),
            )
            .map(|value| value * AB::Expr::from(local.extra_active[slot]));
            accumulated_after = ext_field_add::<AB::Expr>(accumulated_after, slot_term);
            preceding_power = local.extra_theta_powers[slot].map(Into::into);
        }
        assert_array_eq(builder, local.accumulated_after, accumulated_after);
        let final_power = ext_field_multiply::<AB::Expr>(
            local.extra_theta_powers[NATIVE_DIRECT_FRESH_PROJECTION_EXTRA - 1].map(Into::into),
            local.theta.map(Into::into),
        );
        let theta_power_after = core::array::from_fn(|limb| {
            let first_advance = AB::Expr::from(local.extra_theta_powers[0][limb]);
            local.extra_active.iter().enumerate().fold(
                first_advance,
                |selected, (slot, &slot_active)| {
                    let current = AB::Expr::from(local.extra_theta_powers[slot][limb]);
                    let next = if slot + 1 == NATIVE_DIRECT_FRESH_PROJECTION_EXTRA {
                        final_power[limb].clone()
                    } else {
                        AB::Expr::from(local.extra_theta_powers[slot + 1][limb])
                    };
                    selected + AB::Expr::from(slot_active) * (next - current)
                },
            )
        });
        assert_array_eq(builder, local.theta_power_after, theta_power_after);
        builder.when(local.active).assert_eq(
            local.flat_index,
            local.query_index + local.row_offset * AB::Expr::from_usize(self.query_stride),
        );
        builder.when(local.active).assert_eq(
            local.tree_id,
            local.first_tree_id + local.root_ordinal * AB::Expr::from_usize(self.root_tree_stride),
        );
        let inner_tree_id = local.tree_id + AB::Expr::ONE + local.shift;

        let mut transition = builder.when_transition();
        let mut same = transition.when(local.is_same_root);
        same.assert_eq(next.column, local.column + consumed_columns);
        copy_direct_scope(&mut same, local, next);
        assert_array_eq(&mut same, next.accumulated_before, local.accumulated_after);
        assert_array_eq(&mut same, next.theta_power, local.theta_power_after);
        let mut transition = builder.when_transition();
        let mut root = transition.when(local.is_next_root);
        root.assert_zero(next.column);
        root.assert_eq(next.root_ordinal, local.root_ordinal + AB::F::ONE);
        copy_direct_source_shift_scope(&mut root, local, next);
        assert_array_eq(&mut root, next.accumulated_before, local.accumulated_after);
        assert_array_eq(&mut root, next.theta_power, local.theta_power_after);
        let mut transition = builder.when_transition();
        let mut shift = transition.when(local.is_next_shift);
        shift.assert_zero(next.column);
        shift.assert_zero(next.root_ordinal);
        shift.assert_eq(next.shift, local.shift + AB::F::ONE);
        copy_direct_source_scope(&mut shift, local, next);
        assert_array_eq(&mut shift, next.accumulated_before, zero.clone());
        assert_array_eq(&mut shift, next.theta_power, one.clone());
        let mut transition = builder.when_transition();
        let mut source = transition.when(local.is_next_source);
        source.assert_eq(next.proof_idx, local.proof_idx);
        source.assert_zero(next.column);
        source.assert_zero(next.root_ordinal);
        source.assert_zero(next.shift);
        source.assert_eq(next.source, local.source + AB::F::ONE);
        assert_array_eq(&mut source, next.accumulated_before, zero);
        assert_array_eq(&mut source, next.theta_power, one);

        self.source_bus.lookup_key(
            builder,
            NativeDirectFreshSourceMessage {
                source: local.source.into(),
                active: AB::Expr::ONE,
                root_count: local.root_count.into(),
                first_tree_id: local.first_tree_id.into(),
                commitment_tidx: local.commitment_tidx.into(),
                theta: local.theta.map(Into::into),
            },
            sequence_first.clone(),
        );
        self.root_bus.lookup_key(
            builder,
            NativeDirectFreshRootMessage {
                source: local.source.into(),
                root_ordinal: local.root_ordinal.into(),
                tree_id: local.tree_id.into(),
                width: local.width.into(),
                root: local.root.map(Into::into),
            },
            local.active * local.is_first_column,
        );
        self.shift_index_bus.lookup_key(
            builder,
            NativeShiftIndexMessage {
                proof_idx: local.proof_idx.into(),
                shift: local.shift.into(),
                index: local.flat_index.into(),
            },
            sequence_first,
        );
        self.leaf_value_bus.lookup_key(
            builder,
            NativeLeafValueMessage {
                proof_idx: local.proof_idx.into(),
                tree_id: inner_tree_id.clone(),
                index: local.row_offset.into(),
                position: local.column.into(),
                value: local.base_value.into(),
            },
            local.active,
        );
        for slot in 0..NATIVE_DIRECT_FRESH_PROJECTION_EXTRA {
            self.leaf_value_bus.lookup_key(
                builder,
                NativeLeafValueMessage {
                    proof_idx: local.proof_idx.into(),
                    tree_id: inner_tree_id.clone(),
                    index: local.row_offset.into(),
                    position: local.column + AB::Expr::from_usize(slot + 1),
                    value: local.extra_base_values[slot].into(),
                },
                local.active * local.extra_active[slot],
            );
        }
        self.authenticated_bus.send(
            builder,
            NativeAuthenticatedShiftMessage {
                proof_idx: local.proof_idx.into(),
                shift: local.shift.into(),
                source: local.source.into(),
                value: local.accumulated_after.map(Into::into),
            },
            sequence_last,
        );
    }
}

fn copy_direct_source_scope<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    local: &NativeDirectFreshProjectionCols<AB::Var>,
    next: &NativeDirectFreshProjectionCols<AB::Var>,
) {
    builder.assert_eq(next.proof_idx.clone(), local.proof_idx.clone());
    builder.assert_eq(next.source.clone(), local.source.clone());
    builder.assert_eq(next.root_count.clone(), local.root_count.clone());
    builder.assert_eq(next.first_tree_id.clone(), local.first_tree_id.clone());
    builder.assert_eq(next.commitment_tidx.clone(), local.commitment_tidx.clone());
    for limb in 0..D_EF {
        builder.assert_eq(next.theta[limb].clone(), local.theta[limb].clone());
    }
}

fn copy_direct_source_shift_scope<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    local: &NativeDirectFreshProjectionCols<AB::Var>,
    next: &NativeDirectFreshProjectionCols<AB::Var>,
) {
    copy_direct_source_scope(builder, local, next);
    builder.assert_eq(next.shift.clone(), local.shift.clone());
    builder.assert_eq(next.flat_index.clone(), local.flat_index.clone());
    builder.assert_eq(next.query_index.clone(), local.query_index.clone());
    builder.assert_eq(next.row_offset.clone(), local.row_offset.clone());
}

fn copy_direct_scope<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    local: &NativeDirectFreshProjectionCols<AB::Var>,
    next: &NativeDirectFreshProjectionCols<AB::Var>,
) {
    copy_direct_source_shift_scope(builder, local, next);
    builder.assert_eq(next.root_ordinal.clone(), local.root_ordinal.clone());
    builder.assert_eq(next.width.clone(), local.width.clone());
    builder.assert_eq(next.tree_id.clone(), local.tree_id.clone());
    for limb in 0..DIGEST_SIZE {
        builder.assert_eq(next.root[limb].clone(), local.root[limb].clone());
    }
}

fn constrain_zero_flag<AB: AirBuilder<F = F>>(
    builder: &mut AB,
    enabled: AB::Var,
    value: AB::Expr,
    inverse: AB::Var,
    is_zero: AB::Var,
) {
    builder
        .when(enabled.clone() * is_zero.clone())
        .assert_zero(value.clone());
    builder
        .when(enabled * (AB::Expr::ONE - is_zero))
        .assert_one(value * inverse);
}

#[derive(Clone, Debug)]
pub struct NativeDirectFreshProjectionRecord {
    pub source: usize,
    pub shift: usize,
    pub root_ordinal: usize,
    pub root_count: usize,
    pub column: usize,
    pub width: usize,
    pub first_tree_id: usize,
    pub tree_id: usize,
    pub root: [F; DIGEST_SIZE],
    pub commitment_tidx: usize,
    pub flat_index: usize,
    pub query_index: usize,
    pub row_offset: usize,
    pub theta: EF,
    pub theta_power: EF,
    pub accumulated_before: EF,
    pub base_value: F,
    pub accumulated_after: EF,
}

fn projection_record_continues(
    previous: &NativeDirectFreshProjectionRecord,
    next: &NativeDirectFreshProjectionRecord,
) -> bool {
    next.source == previous.source
        && next.shift == previous.shift
        && next.root_ordinal == previous.root_ordinal
        && next.root_count == previous.root_count
        && next.column == previous.column + 1
        && next.width == previous.width
        && next.first_tree_id == previous.first_tree_id
        && next.tree_id == previous.tree_id
        && next.root == previous.root
        && next.commitment_tidx == previous.commitment_tidx
        && next.flat_index == previous.flat_index
        && next.query_index == previous.query_index
        && next.row_offset == previous.row_offset
        && next.theta == previous.theta
        && next.theta_power == previous.theta_power * previous.theta
        && next.accumulated_before == previous.accumulated_after
}

pub fn generate_native_direct_fresh_projection_trace(
    proof_idx: usize,
    records: &[NativeDirectFreshProjectionRecord],
    shift_count: usize,
    required_height: Option<usize>,
) -> Option<RowMajorMatrix<F>> {
    if records.is_empty() || shift_count == 0 {
        return None;
    }
    // Pack consecutive columns from the same authenticated leaf
    // into one AIR row. A root boundary is never crossed: the root bus is
    // consumed once per `(source, shift, root)` and each leaf lookup retains
    // its original `(tree, row, column)` key.
    let mut packed_ranges = Vec::with_capacity(
        records
            .len()
            .div_ceil(NATIVE_DIRECT_FRESH_PROJECTION_PACKING),
    );
    let mut start = 0usize;
    while start < records.len() {
        let mut end = start + 1;
        while end < records.len() && end - start < NATIVE_DIRECT_FRESH_PROJECTION_PACKING {
            let previous = &records[end - 1];
            let next = &records[end];
            let consecutive_coordinates = next.source == previous.source
                && next.shift == previous.shift
                && next.root_ordinal == previous.root_ordinal
                && next.column == previous.column + 1;
            if !consecutive_coordinates {
                break;
            }
            if !projection_record_continues(previous, next) {
                return None;
            }
            end += 1;
        }
        packed_ranges.push((start, end));
        start = end;
    }
    let height = required_height.unwrap_or_else(|| packed_ranges.len().next_power_of_two());
    if height < packed_ranges.len() {
        return None;
    }
    let width = NativeDirectFreshProjectionCols::<F>::width();
    let mut values = F::zero_vec(height * width);
    values
        .par_chunks_mut(width)
        .take(packed_ranges.len())
        .zip(packed_ranges.par_iter())
        .enumerate()
        .try_for_each(|(index, (row, &(start, end)))| {
            let record = &records[start];
            let packed = &records[start..end];
            if record.root_count == 0
                || record.root_ordinal >= record.root_count
                || record.width == 0
                || record.column >= record.width
                || record.shift >= shift_count
                || packed
                    .windows(2)
                    .any(|pair| !projection_record_continues(&pair[0], &pair[1]))
            {
                return Err(());
            }
            let consumed = packed.len();
            if record.column + consumed > record.width {
                return Err(());
            }
            let last = packed.last().ok_or(())?;
            let cols: &mut NativeDirectFreshProjectionCols<F> = row.borrow_mut();
            cols.active = F::ONE;
            cols.proof_idx = F::from_usize(proof_idx);
            for (slot, packed_record) in packed.iter().skip(1).enumerate() {
                cols.extra_active[slot] = F::ONE;
                cols.extra_theta_powers[slot]
                    .copy_from_slice(packed_record.theta_power.as_basis_coefficients_slice());
                cols.extra_base_values[slot] = packed_record.base_value;
            }
            cols.is_first = F::from_bool(index == 0);
            cols.is_last = F::from_bool(index + 1 == packed_ranges.len());
            cols.is_first_column = F::from_bool(record.column == 0);
            cols.is_last_column = F::from_bool(record.column + consumed == record.width);
            cols.is_first_root = F::from_bool(record.root_ordinal == 0);
            cols.is_last_root = F::from_bool(record.root_ordinal + 1 == record.root_count);
            cols.is_first_shift = F::from_bool(record.shift == 0);
            cols.is_last_shift = F::from_bool(record.shift + 1 == shift_count);
            cols.is_sequence_first = F::from_bool(record.column == 0 && record.root_ordinal == 0);
            cols.is_sequence_last = F::from_bool(
                record.column + consumed == record.width
                    && record.root_ordinal + 1 == record.root_count,
            );
            cols.is_same_root = F::from_bool(record.column + consumed != record.width);
            cols.is_next_root = F::from_bool(
                record.column + consumed == record.width
                    && record.root_ordinal + 1 != record.root_count,
            );
            cols.is_next_shift = F::from_bool(
                record.column + consumed == record.width
                    && record.root_ordinal + 1 == record.root_count
                    && record.shift + 1 != shift_count,
            );
            cols.is_next_source = F::from_bool(
                record.column + consumed == record.width
                    && record.root_ordinal + 1 == record.root_count
                    && record.shift + 1 == shift_count
                    && index + 1 != packed_ranges.len(),
            );
            cols.column_inverse = inverse_or_zero(record.column);
            cols.last_column_inverse = inverse_or_zero(record.width - consumed - record.column);
            cols.root_inverse = inverse_or_zero(record.root_ordinal);
            cols.last_root_inverse = inverse_or_zero(record.root_count - 1 - record.root_ordinal);
            cols.shift_inverse = inverse_or_zero(record.shift);
            cols.last_shift_inverse = inverse_or_zero(shift_count - 1 - record.shift);
            cols.source = F::from_usize(record.source);
            cols.shift = F::from_usize(record.shift);
            cols.root_ordinal = F::from_usize(record.root_ordinal);
            cols.root_count = F::from_usize(record.root_count);
            cols.column = F::from_usize(record.column);
            cols.width = F::from_usize(record.width);
            cols.first_tree_id = F::from_usize(record.first_tree_id);
            cols.tree_id = F::from_usize(record.tree_id);
            cols.root.copy_from_slice(&record.root);
            cols.commitment_tidx = F::from_usize(record.commitment_tidx);
            cols.flat_index = F::from_usize(record.flat_index);
            cols.query_index = F::from_usize(record.query_index);
            cols.row_offset = F::from_usize(record.row_offset);
            cols.theta
                .copy_from_slice(record.theta.as_basis_coefficients_slice());
            cols.theta_power
                .copy_from_slice(record.theta_power.as_basis_coefficients_slice());
            // Inactive packed slots still need their power recurrence: the AIR
            // selects the first inactive power as `theta_power_after`.
            let mut power = last.theta_power;
            for slot in consumed.saturating_sub(1)..NATIVE_DIRECT_FRESH_PROJECTION_EXTRA {
                power *= record.theta;
                cols.extra_theta_powers[slot].copy_from_slice(power.as_basis_coefficients_slice());
            }
            let theta_power_after = last.theta_power * last.theta;
            cols.theta_power_after
                .copy_from_slice(theta_power_after.as_basis_coefficients_slice());
            cols.accumulated_before
                .copy_from_slice(record.accumulated_before.as_basis_coefficients_slice());
            cols.base_value = record.base_value;
            cols.accumulated_after
                .copy_from_slice(last.accumulated_after.as_basis_coefficients_slice());
            Ok(())
        })
        .ok()?;
    Some(RowMajorMatrix::new(values, width))
}

fn inverse_or_zero(value: usize) -> F {
    if value == 0 {
        F::ZERO
    } else {
        F::from_usize(value).inverse()
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use openvm_stark_backend::{
        air_builders::debug::check_constraints, any_air_arc_vec, StarkEngine,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config as NativeSC;

    use super::*;

    fn projection_air(first_source: usize, allow_empty: bool) -> NativeDirectFreshProjectionAir {
        NativeDirectFreshProjectionAir {
            source_bus: NativeDirectFreshSourceBus::new(0),
            root_bus: NativeDirectFreshRootBus::new(1),
            shift_index_bus: NativeShiftIndexBus::new(2),
            leaf_value_bus: NativeLeafValueBus::new(3),
            authenticated_bus: NativeAuthenticatedShiftBus::new(4),
            shift_count: 1,
            query_stride: 8,
            root_tree_stride: 2,
            first_source,
            allow_empty,
        }
    }

    fn projection_record(source: usize) -> NativeDirectFreshProjectionRecord {
        NativeDirectFreshProjectionRecord {
            source,
            shift: 0,
            root_ordinal: 0,
            root_count: 1,
            column: 0,
            width: 1,
            first_tree_id: 20,
            tree_id: 20,
            root: [F::ZERO; DIGEST_SIZE],
            commitment_tidx: 7,
            flat_index: 3,
            query_index: 3,
            row_offset: 0,
            theta: EF::ONE,
            theta_power: EF::ONE,
            accumulated_before: EF::ZERO,
            base_value: F::ONE,
            accumulated_after: EF::ONE,
        }
    }

    fn projection_records(source: usize, width: usize) -> Vec<NativeDirectFreshProjectionRecord> {
        let theta = EF::from_u32(3);
        let mut theta_power = EF::ONE;
        let mut accumulated = EF::ZERO;
        (0..width)
            .map(|column| {
                let base_value = F::from_usize(column + 2);
                let accumulated_before = accumulated;
                accumulated += theta_power * EF::from(base_value);
                let record = NativeDirectFreshProjectionRecord {
                    source,
                    shift: 0,
                    root_ordinal: 0,
                    root_count: 1,
                    column,
                    width,
                    first_tree_id: 20,
                    tree_id: 20,
                    root: [F::ZERO; DIGEST_SIZE],
                    commitment_tidx: 7,
                    flat_index: 3,
                    query_index: 3,
                    row_offset: 0,
                    theta,
                    theta_power,
                    accumulated_before,
                    base_value,
                    accumulated_after: accumulated,
                };
                theta_power *= theta;
                record
            })
            .collect()
    }

    fn check(air: &NativeDirectFreshProjectionAir, trace: &RowMajorMatrix<F>) {
        check_constraints::<_, NativeSC>(
            air,
            "NativeDirectFreshProjectionAir",
            &None,
            &[trace.as_view()],
            &[],
        );
    }

    #[test]
    fn projection_shard_binds_its_first_source() {
        let air = projection_air(4, true);
        let trace =
            generate_native_direct_fresh_projection_trace(0, &[projection_record(4)], 1, None)
                .unwrap();
        check(&air, &trace);

        let mut wrong_source = trace.clone();
        let cols: &mut NativeDirectFreshProjectionCols<F> =
            wrong_source.values.as_mut_slice().borrow_mut();
        cols.source = F::from_usize(5);
        assert!(catch_unwind(AssertUnwindSafe(|| check(&air, &wrong_source))).is_err());
    }

    #[test]
    fn only_suffix_projection_shards_may_be_empty() {
        let width = NativeDirectFreshProjectionCols::<F>::width();
        let empty = RowMajorMatrix::new(F::zero_vec(width), width);
        check(&projection_air(4, true), &empty);
        assert!(catch_unwind(AssertUnwindSafe(|| {
            check(&projection_air(0, false), &empty)
        }))
        .is_err());
    }

    #[test]
    fn projection_uses_the_full_packing_factor_and_handles_a_partial_tail() {
        let air = projection_air(4, true);
        let records = projection_records(4, NATIVE_DIRECT_FRESH_PROJECTION_PACKING + 2);
        let trace = generate_native_direct_fresh_projection_trace(0, &records, 1, None).unwrap();
        assert_eq!(trace.height(), 2);
        let first_row = trace.row_slice(0).unwrap();
        let last_row = trace.row_slice(1).unwrap();
        let first: &NativeDirectFreshProjectionCols<F> = (*first_row).borrow();
        let last: &NativeDirectFreshProjectionCols<F> = (*last_row).borrow();
        assert!(first.extra_active.iter().all(|&active| active == F::ONE));
        assert_eq!(
            first.extra_base_values[NATIVE_DIRECT_FRESH_PROJECTION_EXTRA - 1],
            records[NATIVE_DIRECT_FRESH_PROJECTION_PACKING - 1].base_value
        );
        assert_eq!(last.extra_active[0], F::ONE);
        assert_eq!(last.extra_active[1], F::ZERO);
        assert_eq!(
            last.column,
            F::from_usize(NATIVE_DIRECT_FRESH_PROJECTION_PACKING)
        );
        check(&air, &trace);
    }

    #[test]
    fn projection_packing_rejects_a_mutated_packed_value_or_power() {
        let air = projection_air(4, true);
        let records = projection_records(4, 2);
        let trace = generate_native_direct_fresh_projection_trace(0, &records, 1, None).unwrap();
        check(&air, &trace);

        let mut wrong_value = trace.clone();
        let cols: &mut NativeDirectFreshProjectionCols<F> =
            wrong_value.values.as_mut_slice().borrow_mut();
        cols.extra_base_values[0] += F::ONE;
        assert!(catch_unwind(AssertUnwindSafe(|| check(&air, &wrong_value))).is_err());

        let mut wrong_intermediate_power = trace.clone();
        let cols: &mut NativeDirectFreshProjectionCols<F> =
            wrong_intermediate_power.values.as_mut_slice().borrow_mut();
        cols.extra_theta_powers[0][0] += F::ONE;
        assert!(catch_unwind(AssertUnwindSafe(|| {
            check(&air, &wrong_intermediate_power)
        }))
        .is_err());

        let mut wrong_power = trace;
        let cols: &mut NativeDirectFreshProjectionCols<F> =
            wrong_power.values.as_mut_slice().borrow_mut();
        cols.theta_power_after[0] += F::ONE;
        assert!(catch_unwind(AssertUnwindSafe(|| check(&air, &wrong_power))).is_err());
    }

    #[test]
    fn projection_packing_rejects_inconsistent_adjacent_metadata() {
        let mut records = projection_records(4, 2);
        records[1].tree_id += 1;
        assert!(generate_native_direct_fresh_projection_trace(0, &records, 1, None).is_none());
    }

    #[test]
    fn packed_projection_stays_within_the_history_degree_bound() {
        let engine = crate::tests::test_engine_small();
        let airs = any_air_arc_vec![projection_air(0, false)];
        let (_pk, vk) = engine.keygen(&airs);
        assert!(vk.max_constraint_degree() <= crate::tests::MAX_CONSTRAINT_DEGREE);
    }
}

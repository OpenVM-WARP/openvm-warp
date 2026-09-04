//! Recursive authentication for the exact-finite vector-alphabet fresh source.
//!
//! The native backend commits a batch of base-field codewords column-wise:
//! each opened base row is hashed, `rows_per_leaf` row hashes are compressed
//! into one outer leaf, and one binary multiproof authenticates all queried
//! leaves. This module reproduces that layout exactly. It intentionally does
//! not reinterpret a scalar/extension opening proof as a vector opening.

use core::borrow::{Borrow, BorrowMut};

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    hasher::MerkleHasher,
    interaction::InteractionBuilder,
    warp_accum::{
        BinaryMerkleMultiproofRecord, FiniteStackedFreshBatchOpeningProof,
        FiniteStackedFreshBatchOpeningVerification, FiniteStackedFreshCommitment,
        MerkleCompressionRecord, MerkleNodeOrigin, FINITE_STACKED_FRESH_BASE_ALPHABET_TAG,
        FINITE_STACKED_FRESH_COMMITMENT_DOMAIN_TAG, FINITE_STACKED_FRESH_ROW_LAYOUT_VERSION,
        FINITE_STACKED_FRESH_TRANSCRIPT_VERSION,
    },
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, DIGEST_SIZE, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{BasedVectorSpace, Field, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::{
    bus::TranscriptBus,
    native_warp::{
        generate_native_leaf_hash_trace, generate_native_merkle_leaf_adapter_trace,
        generate_native_merkle_multiproof_trace, NativeAuthenticatedShiftBus,
        NativeAuthenticatedShiftMessage, NativeInputSlotLayoutBus, NativeInputSlotLayoutMessage,
        NativeLeafHashInput, NativeLeafHashTrace, NativeLeafValueBus, NativeLeafValueMessage,
        NativeMerkleCompressionCols, NativeMerkleLeafAdapterInput, NativeMerkleRootBus,
        NativeMerkleRootMessage, NativeShiftIndexBus, NativeShiftIndexMessage,
        NativeStandardVaccRootBus, NativeStandardVaccRootMessage, NativeVaccPhaseCursorBus,
        NativeVaccPhaseCursorMessage, NativeVaccTranscriptRoleBus, NativeVaccTranscriptRoleMessage,
        VACC_ROLE_FRESH_ROOT,
    },
};

/// Enough bits for every supported finite codeword index (`2^32` is excluded
/// by the backend's `u32` wire coordinates).
const INDEX_BITS: usize = 32;
const _: () = assert!(D_EF == 4);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NativeFiniteStackedFreshProfile {
    pub input_arity: usize,
    pub fresh_count: usize,
    pub prior_count: usize,
    pub log_message_len: usize,
    pub log_codeword_len: usize,
    pub rows_per_leaf: usize,
    pub outer_tree_id: u32,
    pub row_tree_id_offset: u32,
}

impl NativeFiniteStackedFreshProfile {
    pub fn validate(self) -> Result<(), &'static str> {
        let codeword_len = 1usize
            .checked_shl(self.log_codeword_len as u32)
            .ok_or("finite stacked codeword length")?;
        if self.input_arity < 2
            || self.input_arity > 64
            || !self.input_arity.is_power_of_two()
            || self.fresh_count == 0
            || self.prior_count > 1
            || self.fresh_count + self.prior_count != self.input_arity
            || self.log_message_len == 0
            || self.log_codeword_len < self.log_message_len
            || self.log_codeword_len > INDEX_BITS
            || self.rows_per_leaf == 0
            || !self.rows_per_leaf.is_power_of_two()
            || self.rows_per_leaf >= codeword_len
        {
            return Err("finite stacked fresh profile");
        }
        Ok(())
    }

    #[must_use]
    pub const fn variant(self) -> usize {
        self.fresh_count + self.prior_count * (self.input_arity + 1)
    }

    #[must_use]
    pub fn query_stride(self) -> usize {
        (1usize << self.log_codeword_len) / self.rows_per_leaf
    }

    #[must_use]
    pub fn outer_depth(self) -> usize {
        self.query_stride().ilog2() as usize
    }
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeFiniteStackedFreshCommitmentCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub tidx: T,
    pub source_ordinal: T,
    pub is_first: T,
    pub is_last: T,
    pub root: [T; DIGEST_SIZE],
}

/// Authenticates all source descriptors and connects their shared root to the
/// vector-alphabet multiproof. The descriptor order is fixed to `0..fresh`.
#[derive(ColumnsAir)]
#[columns_via(NativeFiniteStackedFreshCommitmentCols<u8>)]
pub struct NativeFiniteStackedFreshCommitmentAir {
    pub transcript_bus: TranscriptBus,
    pub transcript_role_bus: NativeVaccTranscriptRoleBus,
    pub phase_cursor_bus: NativeVaccPhaseCursorBus,
    pub merkle_root_bus: NativeMerkleRootBus,
    pub statement_root_bus: NativeStandardVaccRootBus,
    pub profile: NativeFiniteStackedFreshProfile,
}

impl BaseAirWithPublicValues<F> for NativeFiniteStackedFreshCommitmentAir {}
impl PartitionedBaseAir<F> for NativeFiniteStackedFreshCommitmentAir {}
impl BaseAir<F> for NativeFiniteStackedFreshCommitmentAir {
    fn width(&self) -> usize {
        NativeFiniteStackedFreshCommitmentCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeFiniteStackedFreshCommitmentAir {
    fn eval(&self, builder: &mut AB) {
        assert!(
            self.profile.validate().is_ok(),
            "invalid finite stacked profile"
        );
        let main = builder.main();
        let row = main.row_slice(0).expect("finite stacked descriptor row");
        let next_row = main
            .row_slice(1)
            .expect("finite stacked descriptor next row");
        let local: &NativeFiniteStackedFreshCommitmentCols<AB::Var> = (*row).borrow();
        let next: &NativeFiniteStackedFreshCommitmentCols<AB::Var> = (*next_row).borrow();
        for flag in [local.active, local.is_first, local.is_last] {
            builder.assert_bool(flag);
        }
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_one(local.is_first);
        builder
            .when(local.active * local.is_first)
            .assert_zero(local.source_ordinal);
        builder.when(local.active * local.is_last).assert_eq(
            local.source_ordinal,
            AB::Expr::from_usize(self.profile.fresh_count - 1),
        );
        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        let mut transition = builder.when_transition();
        let mut same = transition.when(next.active);
        same.assert_eq(next.proof_idx, local.proof_idx);
        same.assert_eq(next.source_ordinal, local.source_ordinal + AB::F::ONE);
        same.assert_zero(next.is_first);
        same.assert_eq(
            next.tidx,
            local.tidx + AB::Expr::from_usize((9 + DIGEST_SIZE) * D_EF),
        );
        for limb in 0..DIGEST_SIZE {
            same.assert_eq(next.root[limb], local.root[limb]);
        }
        builder
            .when_transition()
            .when(local.active - next.active)
            .assert_one(local.is_last);

        self.phase_cursor_bus.receive(
            builder,
            NativeVaccPhaseCursorMessage {
                proof_idx: local.proof_idx.into(),
                boundary: AB::Expr::ONE,
                tidx: local.tidx.into(),
            },
            local.active * local.is_first,
        );

        let mut tidx: AB::Expr = local.tidx.into();
        let mut ordinal =
            AB::Expr::from(local.source_ordinal) * AB::Expr::from_usize(9 + DIGEST_SIZE);
        for value in [
            FINITE_STACKED_FRESH_COMMITMENT_DOMAIN_TAG,
            FINITE_STACKED_FRESH_TRANSCRIPT_VERSION,
            FINITE_STACKED_FRESH_ROW_LAYOUT_VERSION,
            FINITE_STACKED_FRESH_BASE_ALPHABET_TAG,
            self.profile.rows_per_leaf as u64,
            self.profile.fresh_count as u64,
        ] {
            observe_base_ext_with_role(
                &self.transcript_bus,
                &self.transcript_role_bus,
                builder,
                local.proof_idx,
                tidx.clone(),
                AB::Expr::from_u64(value),
                ordinal.clone(),
                local.active,
            );
            tidx += AB::Expr::from_usize(D_EF);
            ordinal += AB::Expr::ONE;
        }
        observe_base_ext_with_role(
            &self.transcript_bus,
            &self.transcript_role_bus,
            builder,
            local.proof_idx,
            tidx.clone(),
            local.source_ordinal.into(),
            ordinal.clone(),
            local.active,
        );
        tidx += AB::Expr::from_usize(D_EF);
        ordinal += AB::Expr::ONE;
        for value in [self.profile.log_message_len, self.profile.log_codeword_len] {
            observe_base_ext_with_role(
                &self.transcript_bus,
                &self.transcript_role_bus,
                builder,
                local.proof_idx,
                tidx.clone(),
                AB::Expr::from_usize(value),
                ordinal.clone(),
                local.active,
            );
            tidx += AB::Expr::from_usize(D_EF);
            ordinal += AB::Expr::ONE;
        }
        for root in local.root {
            observe_base_ext_with_role(
                &self.transcript_bus,
                &self.transcript_role_bus,
                builder,
                local.proof_idx,
                tidx.clone(),
                root.into(),
                ordinal.clone(),
                local.active,
            );
            tidx += AB::Expr::from_usize(D_EF);
            ordinal += AB::Expr::ONE;
        }

        self.merkle_root_bus.receive(
            builder,
            NativeMerkleRootMessage {
                proof_idx: local.proof_idx.into(),
                tree_id: AB::Expr::from_u32(self.profile.outer_tree_id),
                depth: AB::Expr::from_usize(self.profile.outer_depth()),
                digest: local.root.map(Into::into),
            },
            local.active * local.is_first,
        );
        self.statement_root_bus.add_key_with_lookups(
            builder,
            NativeStandardVaccRootMessage {
                proof_idx: local.proof_idx.into(),
                kind: AB::Expr::ZERO,
                root: local.root.map(Into::into),
            },
            local.active * local.is_first,
        );
    }
}

fn observe_base_ext_with_role<AB: AirBuilder<F = F> + InteractionBuilder>(
    bus: &TranscriptBus,
    role_bus: &NativeVaccTranscriptRoleBus,
    builder: &mut AB,
    proof_idx: AB::Var,
    tidx: AB::Expr,
    value: AB::Expr,
    ordinal: AB::Expr,
    enabled: AB::Var,
) {
    let ext = [value, AB::Expr::ZERO, AB::Expr::ZERO, AB::Expr::ZERO];
    bus.observe_ext(builder, proof_idx, tidx.clone(), ext.clone(), enabled);
    role_bus.receive(
        builder,
        NativeVaccTranscriptRoleMessage {
            proof_idx: proof_idx.into(),
            role: AB::Expr::from_usize(VACC_ROLE_FRESH_ROOT),
            ordinal,
            tidx,
            value: ext,
            is_ext: AB::Expr::ONE,
            is_sample: AB::Expr::ZERO,
        },
        enabled,
    );
}

pub fn generate_native_finite_stacked_fresh_commitment_trace(
    proof_idx: usize,
    first_tidx: usize,
    descriptors: &[FiniteStackedFreshCommitment<Digest>],
    profile: NativeFiniteStackedFreshProfile,
) -> Option<RowMajorMatrix<F>> {
    profile.validate().ok()?;
    if descriptors.len() != profile.fresh_count {
        return None;
    }
    let first = descriptors.first()?;
    let expected = |source_ordinal: usize, descriptor: &FiniteStackedFreshCommitment<Digest>| {
        descriptor.root == first.root
            && descriptor.source_ordinal as usize == source_ordinal
            && descriptor.fresh_count as usize == profile.fresh_count
            && descriptor.log_message_len as usize == profile.log_message_len
            && descriptor.log_codeword_len as usize == profile.log_codeword_len
            && descriptor.rows_per_leaf as usize == profile.rows_per_leaf
            && descriptor.alphabet_tag == FINITE_STACKED_FRESH_BASE_ALPHABET_TAG
            && descriptor.row_layout_version == FINITE_STACKED_FRESH_ROW_LAYOUT_VERSION
            && descriptor.transcript_version == FINITE_STACKED_FRESH_TRANSCRIPT_VERSION
    };
    if descriptors
        .iter()
        .enumerate()
        .any(|(source, descriptor)| !expected(source, descriptor))
    {
        return None;
    }
    let width = NativeFiniteStackedFreshCommitmentCols::<F>::width();
    let height = profile.fresh_count.next_power_of_two();
    let mut values = F::zero_vec(height * width);
    let descriptor_stride = (9 + DIGEST_SIZE) * D_EF;
    for (source, descriptor) in descriptors.iter().enumerate() {
        let cols: &mut NativeFiniteStackedFreshCommitmentCols<F> =
            values[source * width..(source + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.proof_idx = F::from_usize(proof_idx);
        cols.tidx = F::from_usize(first_tidx + source * descriptor_stride);
        cols.source_ordinal = F::from_usize(source);
        cols.is_first = F::from_bool(source == 0);
        cols.is_last = F::from_bool(source + 1 == profile.fresh_count);
        cols.root = descriptor.root;
    }
    Some(RowMajorMatrix::new(values, width))
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeFiniteStackedFreshProjectionCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub ordinal: T,
    pub shift: T,
    pub source: T,
    pub source_is_last: T,
    pub source_distance_inverse: T,
    pub flat_index: T,
    pub query_index: T,
    pub row_offset: T,
    pub query_bits: [T; INDEX_BITS],
    pub row_bits: [T; INDEX_BITS],
    pub inner_tree_id: T,
    pub value: [T; D_EF],
}

/// Projects one authenticated base-field coordinate from every vector source
/// into the ordinary WARP authenticated-shift bus.
#[derive(ColumnsAir)]
#[columns_via(NativeFiniteStackedFreshProjectionCols<u8>)]
pub struct NativeFiniteStackedFreshProjectionAir {
    pub shift_index_bus: NativeShiftIndexBus,
    pub leaf_value_bus: NativeLeafValueBus,
    pub authenticated_bus: NativeAuthenticatedShiftBus,
    pub slot_bus: NativeInputSlotLayoutBus,
    pub profile: NativeFiniteStackedFreshProfile,
    pub shift_count: usize,
}

impl BaseAirWithPublicValues<F> for NativeFiniteStackedFreshProjectionAir {}
impl PartitionedBaseAir<F> for NativeFiniteStackedFreshProjectionAir {}
impl BaseAir<F> for NativeFiniteStackedFreshProjectionAir {
    fn width(&self) -> usize {
        NativeFiniteStackedFreshProjectionCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeFiniteStackedFreshProjectionAir {
    fn eval(&self, builder: &mut AB) {
        assert!(
            self.profile.validate().is_ok() && self.shift_count > 0,
            "invalid finite stacked projection profile"
        );
        let main = builder.main();
        let row = main.row_slice(0).expect("finite stacked projection row");
        let next_row = main
            .row_slice(1)
            .expect("finite stacked projection next row");
        let local: &NativeFiniteStackedFreshProjectionCols<AB::Var> = (*row).borrow();
        let next: &NativeFiniteStackedFreshProjectionCols<AB::Var> = (*next_row).borrow();
        builder.assert_bool(local.active);
        builder.assert_bool(local.source_is_last);
        builder.when_first_row().assert_one(local.active);
        builder.when_first_row().assert_zero(local.ordinal);
        builder.when_first_row().assert_zero(local.shift);
        builder.when_first_row().assert_zero(local.source);
        let source_distance = local.source - AB::Expr::from_usize(self.profile.fresh_count - 1);
        builder
            .when(local.active * local.source_is_last)
            .assert_zero(source_distance.clone());
        builder
            .when(local.active * (AB::Expr::ONE - local.source_is_last))
            .assert_one(source_distance * local.source_distance_inverse);
        builder.when(local.active).assert_eq(
            local.ordinal,
            local.shift * AB::Expr::from_usize(self.profile.fresh_count) + local.source,
        );
        builder.when(local.active).assert_eq(
            local.inner_tree_id,
            AB::Expr::from_u32(self.profile.row_tree_id_offset) + local.shift,
        );
        builder.when(local.active).assert_eq(
            local.flat_index,
            local.row_offset * AB::Expr::from_usize(self.profile.query_stride())
                + local.query_index,
        );
        let query_log = self.profile.outer_depth();
        let row_log = self.profile.rows_per_leaf.ilog2() as usize;
        let mut query_recomposed = AB::Expr::ZERO;
        let mut row_recomposed = AB::Expr::ZERO;
        for bit in 0..INDEX_BITS {
            builder.assert_bool(local.query_bits[bit]);
            builder.assert_bool(local.row_bits[bit]);
            if bit < query_log {
                query_recomposed += local.query_bits[bit] * AB::Expr::from_u64(1u64 << bit);
            } else {
                builder
                    .when(local.active)
                    .assert_zero(local.query_bits[bit]);
            }
            if bit < row_log {
                row_recomposed += local.row_bits[bit] * AB::Expr::from_u64(1u64 << bit);
            } else {
                builder.when(local.active).assert_zero(local.row_bits[bit]);
            }
        }
        builder
            .when(local.active)
            .assert_eq(local.query_index, query_recomposed);
        builder
            .when(local.active)
            .assert_eq(local.row_offset, row_recomposed);
        for limb in &local.value[1..] {
            builder.when(local.active).assert_zero(*limb);
        }

        builder
            .when_transition()
            .assert_bool(local.active - next.active);
        let mut transition = builder.when_transition();
        let mut same = transition.when(next.active);
        same.assert_eq(next.proof_idx, local.proof_idx);
        same.assert_eq(next.ordinal, local.ordinal + AB::F::ONE);
        same.assert_eq(
            next.source,
            (AB::Expr::ONE - local.source_is_last) * (local.source + AB::F::ONE),
        );
        same.assert_eq(next.shift, local.shift + local.source_is_last);
        builder
            .when_transition()
            .when(local.active - next.active)
            .assert_eq(
                local.ordinal,
                AB::Expr::from_usize(self.shift_count * self.profile.fresh_count - 1),
            );

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
                variant: AB::Expr::from_usize(self.profile.variant()),
                source: local.source.into(),
                kind: [AB::Expr::ONE, AB::Expr::ZERO, AB::Expr::ZERO],
            },
            local.active,
        );
        self.leaf_value_bus.lookup_key(
            builder,
            NativeLeafValueMessage {
                proof_idx: local.proof_idx.into(),
                tree_id: local.inner_tree_id.into(),
                index: local.row_offset.into(),
                position: local.source.into(),
                value: local.value[0].into(),
            },
            local.active,
        );
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

pub struct NativeFiniteStackedFreshOpeningTraces {
    pub projection: RowMajorMatrix<F>,
    pub leaf_hash: NativeLeafHashTrace,
    pub merkle: RowMajorMatrix<F>,
    pub leaf_adapter: RowMajorMatrix<F>,
    pub compression_inputs: Vec<[F; openvm_poseidon2_air::POSEIDON2_WIDTH]>,
}

#[allow(clippy::too_many_arguments)]
pub fn generate_native_finite_stacked_fresh_opening_traces<H>(
    hasher: &H,
    proof_idx: usize,
    flat_indices: &[usize],
    proof: &FiniteStackedFreshBatchOpeningProof<F, Digest>,
    verification: &FiniteStackedFreshBatchOpeningVerification<EF, Digest>,
    profile: NativeFiniteStackedFreshProfile,
) -> Option<NativeFiniteStackedFreshOpeningTraces>
where
    H: MerkleHasher<F = F, Digest = Digest>,
{
    profile.validate().ok()?;
    let shared = verification.multiproof.as_ref()?;
    if !verification.recorded
        || flat_indices.is_empty()
        || proof.opened_rows.len() != flat_indices.len()
        || verification.values_by_source.len() != profile.fresh_count
        || verification
            .values_by_source
            .iter()
            .any(|values| values.len() != flat_indices.len())
        || shared.depth as usize != profile.outer_depth()
        || shared.leaf_indices.len() != flat_indices.len()
        || shared.leaf_digests.len() != flat_indices.len()
    {
        return None;
    }
    let proof_idx_u32 = u32::try_from(proof_idx).ok()?;
    let query_stride = profile.query_stride();
    let mut projection_values = F::zero_vec(
        (flat_indices.len() * profile.fresh_count).next_power_of_two()
            * NativeFiniteStackedFreshProjectionCols::<F>::width(),
    );
    let projection_width = NativeFiniteStackedFreshProjectionCols::<F>::width();
    let mut leaf_storage = Vec::<(u32, u32, Vec<F>, Vec<u32>)>::new();
    let mut inner_records = Vec::<(u32, BinaryMerkleMultiproofRecord<Digest>)>::new();
    let mut adapters = Vec::with_capacity(flat_indices.len());

    for (shift, (&flat_index, rows)) in flat_indices.iter().zip(&proof.opened_rows).enumerate() {
        if flat_index >= 1usize.checked_shl(profile.log_codeword_len as u32)?
            || rows.len() != profile.rows_per_leaf
            || rows.iter().any(|row| row.len() != profile.fresh_count)
        {
            return None;
        }
        let query_index = flat_index % query_stride;
        let row_offset = flat_index / query_stride;
        if shared.leaf_indices[shift] as usize != query_index {
            return None;
        }
        let inner_tree_id = profile
            .row_tree_id_offset
            .checked_add(u32::try_from(shift).ok()?)?;
        let leaf_digests = rows
            .iter()
            .map(|row| hasher.hash_slice(row))
            .collect::<Vec<_>>();
        let inner = complete_binary_tree_record(hasher, leaf_digests)?;
        if inner.expected_root != shared.leaf_digests[shift] {
            return None;
        }
        for (row_index, row) in rows.iter().enumerate() {
            leaf_storage.push((
                inner_tree_id,
                u32::try_from(row_index).ok()?,
                row.clone(),
                (0..profile.fresh_count)
                    .map(|_| u32::from(row_index == row_offset))
                    .collect(),
            ));
        }
        adapters.push(NativeMerkleLeafAdapterInput {
            proof_idx: proof_idx_u32,
            bypass: profile.rows_per_leaf == 1,
            outer_multiplicity: 1,
            inner_tree_id,
            outer_tree_id: profile.outer_tree_id,
            query_index: u32::try_from(query_index).ok()?,
            inner_depth: profile.rows_per_leaf.ilog2(),
            digest: inner.expected_root,
        });
        inner_records.push((inner_tree_id, inner));

        for source in 0..profile.fresh_count {
            let expected = EF::from(rows[row_offset][source]);
            if verification.values_by_source[source][shift] != expected {
                return None;
            }
            let ordinal = shift * profile.fresh_count + source;
            let cols: &mut NativeFiniteStackedFreshProjectionCols<F> = projection_values
                [ordinal * projection_width..(ordinal + 1) * projection_width]
                .borrow_mut();
            cols.active = F::ONE;
            cols.proof_idx = F::from_usize(proof_idx);
            cols.ordinal = F::from_usize(ordinal);
            cols.shift = F::from_usize(shift);
            cols.source = F::from_usize(source);
            cols.source_is_last = F::from_bool(source + 1 == profile.fresh_count);
            cols.source_distance_inverse = if source + 1 == profile.fresh_count {
                F::ZERO
            } else {
                (F::from_usize(source) - F::from_usize(profile.fresh_count - 1)).inverse()
            };
            cols.flat_index = F::from_usize(flat_index);
            cols.query_index = F::from_usize(query_index);
            cols.row_offset = F::from_usize(row_offset);
            write_bits(query_index, &mut cols.query_bits);
            write_bits(row_offset, &mut cols.row_bits);
            cols.inner_tree_id = F::from_u32(inner_tree_id);
            cols.value
                .copy_from_slice(expected.as_basis_coefficients_slice());
        }
    }

    let leaf_inputs = leaf_storage
        .iter()
        .map(
            |(tree_id, leaf_index, values, lookup_counts)| NativeLeafHashInput {
                proof_idx: proof_idx_u32,
                tree_id: *tree_id,
                leaf_index: *leaf_index,
                values,
                lookup_counts,
            },
        )
        .collect::<Vec<_>>();
    let leaf_hash = generate_native_leaf_hash_trace(&leaf_inputs, None)?;
    let mut records = inner_records
        .iter()
        .filter(|(_, record)| !record.compressions.is_empty())
        .map(|(tree_id, record)| (*tree_id, record))
        .collect::<Vec<_>>();
    records.push((profile.outer_tree_id, shared));
    let compression_inputs = records
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
        .collect();
    let merkle = generate_native_merkle_multiproof_trace(proof_idx, &records, None)
        .unwrap_or_else(zero_merkle_trace);
    let leaf_adapter = generate_native_merkle_leaf_adapter_trace(
        &adapters,
        profile.rows_per_leaf.ilog2() as usize,
        None,
    )?;
    Some(NativeFiniteStackedFreshOpeningTraces {
        projection: RowMajorMatrix::new(projection_values, projection_width),
        leaf_hash,
        merkle,
        leaf_adapter,
        compression_inputs,
    })
}

fn write_bits(mut value: usize, target: &mut [F; INDEX_BITS]) {
    for bit in target {
        *bit = F::from_bool(value & 1 == 1);
        value >>= 1;
    }
}

fn complete_binary_tree_record<H>(
    hasher: &H,
    leaf_digests: Vec<Digest>,
) -> Option<BinaryMerkleMultiproofRecord<Digest>>
where
    H: MerkleHasher<F = F, Digest = Digest>,
{
    if leaf_digests.is_empty() || !leaf_digests.len().is_power_of_two() {
        return None;
    }
    let original = leaf_digests.clone();
    let depth = leaf_digests.len().ilog2();
    let mut current = leaf_digests;
    let mut compressions = Vec::with_capacity(current.len().saturating_sub(1));
    for level in 0..depth {
        let mut next = Vec::with_capacity(current.len() / 2);
        for parent_index in 0..current.len() / 2 {
            let left = current[2 * parent_index];
            let right = current[2 * parent_index + 1];
            let output = hasher.compress(left, right);
            let origin = |child: usize| {
                if level == 0 {
                    MerkleNodeOrigin::OpenedLeaf {
                        query_ordinal: child as u32,
                        multiplicity: 1,
                    }
                } else {
                    MerkleNodeOrigin::Computed {
                        level: level - 1,
                        index: child as u32,
                    }
                }
            };
            compressions.push(MerkleCompressionRecord {
                level,
                parent_index: parent_index as u32,
                left,
                right,
                output,
                left_origin: origin(2 * parent_index),
                right_origin: origin(2 * parent_index + 1),
            });
            next.push(output);
        }
        current = next;
    }
    Some(BinaryMerkleMultiproofRecord {
        expected_root: current[0],
        depth,
        leaf_indices: (0..original.len() as u32).collect(),
        leaf_digests: original,
        compressions,
        consumed_siblings: 0,
    })
}

fn zero_merkle_trace() -> RowMajorMatrix<F> {
    RowMajorMatrix::new(
        F::zero_vec(NativeMerkleCompressionCols::<F>::width()),
        NativeMerkleCompressionCols::<F>::width(),
    )
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use openvm_stark_backend::{
        warp_accum::{
            FiniteStackedBaseCodewordReader, FiniteStackedFreshCommitter,
            FiniteStackedFreshOpeningBackend, FiniteStackedFreshSource,
        },
        StarkProtocolConfig, SystemParams,
    };
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config;

    use super::*;

    fn fixture() -> (
        <BabyBearPoseidon2Config as StarkProtocolConfig>::Hasher,
        NativeFiniteStackedFreshProfile,
        Vec<FiniteStackedFreshCommitment<Digest>>,
        Vec<usize>,
        FiniteStackedFreshBatchOpeningProof<F, Digest>,
        FiniteStackedFreshBatchOpeningVerification<EF, Digest>,
    ) {
        let config = BabyBearPoseidon2Config::default_from_params(SystemParams::new_for_testing(4));
        let hasher = config.hasher().clone();
        let profile = NativeFiniteStackedFreshProfile {
            input_arity: 2,
            fresh_count: 2,
            prior_count: 0,
            log_message_len: 3,
            log_codeword_len: 4,
            rows_per_leaf: 4,
            outer_tree_id: 70,
            row_tree_id_offset: 80,
        };
        let columns = (0..profile.fresh_count)
            .map(|source| {
                (0..1usize << profile.log_codeword_len)
                    .map(|row| F::from_usize(1 + source * 100 + row))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let readers = columns
            .iter()
            .map(|column| column as &dyn FiniteStackedBaseCodewordReader<F>)
            .collect::<Vec<_>>();
        let artifact = FiniteStackedFreshCommitter::new(
            hasher.clone(),
            profile.log_message_len,
            profile.log_codeword_len,
            profile.rows_per_leaf,
        )
        .unwrap()
        .commit_readers(&readers)
        .unwrap();
        let sources = (0..profile.fresh_count)
            .map(|source| {
                FiniteStackedFreshSource::new(
                    Arc::clone(&artifact),
                    source,
                    vec![EF::from_u32(source as u32 + 1)],
                    columns[source][..1 << profile.log_message_len].to_vec(),
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let descriptors = sources
            .iter()
            .map(|source| source.descriptor().clone())
            .collect::<Vec<_>>();
        let indices = vec![1, 14];
        let prover = FiniteStackedFreshOpeningBackend::new(hasher.clone());
        let (values, proof) = prover.open_invocation(&sources, &indices).unwrap();
        let verification = FiniteStackedFreshOpeningBackend::new_recording(hasher.clone())
            .verify_invocation(artifact.layout(), &descriptors, &indices, &values, &proof)
            .unwrap();
        (hasher, profile, descriptors, indices, proof, verification)
    }

    #[test]
    fn finite_stacked_recursive_layout_matches_recorded_backend_verification() {
        let (hasher, profile, descriptors, indices, proof, verification) = fixture();
        generate_native_finite_stacked_fresh_commitment_trace(0, 100, &descriptors, profile)
            .unwrap();
        let traces = generate_native_finite_stacked_fresh_opening_traces(
            &hasher,
            0,
            &indices,
            &proof,
            &verification,
            profile,
        )
        .unwrap();
        assert_eq!(
            traces.projection.height(),
            (indices.len() * profile.fresh_count).next_power_of_two()
        );
        assert!(!traces.compression_inputs.is_empty());
    }

    #[test]
    fn finite_stacked_mutations_fail_closed() {
        let (hasher, profile, mut descriptors, _indices, _proof, _verification) = fixture();
        descriptors.swap(0, 1);
        assert!(generate_native_finite_stacked_fresh_commitment_trace(
            0,
            100,
            &descriptors,
            profile,
        )
        .is_none());

        let (_, _, _, indices, mut proof, verification) = fixture();
        proof.opened_rows[0][0][0] += F::ONE;
        assert!(generate_native_finite_stacked_fresh_opening_traces(
            &hasher,
            0,
            &indices,
            &proof,
            &verification,
            profile,
        )
        .is_none());

        let (_, _, _, indices, proof, mut verification) = fixture();
        verification.values_by_source[0][0] += EF::ONE;
        assert!(generate_native_finite_stacked_fresh_opening_traces(
            &hasher,
            0,
            &indices,
            &proof,
            &verification,
            profile,
        )
        .is_none());
    }
}

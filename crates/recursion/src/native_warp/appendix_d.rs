//! Recursive authentication for WARP Appendix-D base-alphabet fresh openings.
//!
//! Fresh leaves are hashed as BabyBear values.  Only after the Merkle path has
//! authenticated a base value do these constraints apply the systematic
//! embedding `psi(b) = (b, 0, 0, 0)` used by the EF4 VACC algebra.

use core::borrow::{Borrow, BorrowMut};
use std::collections::BTreeMap;

use openvm_circuit_primitives::ColumnsAir;
use openvm_stark_backend::{
    hasher::MerkleHasher,
    interaction::InteractionBuilder,
    warp_accum::{
        verify_binary_merkle_multiproof_recorded, AppendixDBaseFreshOpeningVerification,
        AppendixDFreshAlphabetMarker, BinaryMerkleMultiProof, MerkleBatchOpeningVerification,
    },
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::PrimeCharacteristicRing;
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use super::{
    bus::{
        NativeAuthenticatedShiftBus, NativeAuthenticatedShiftMessage, NativeInputSlotLayoutBus,
        NativeInputSlotLayoutMessage, NativeLeafValueBus, NativeLeafValueMessage,
        NativeShiftIndexBus, NativeShiftIndexMessage,
    },
    NativeAccumulatorProjectionCols, NativeAccumulatorProjectionTrace,
    NativeMerkleLeafAdapterInput, NativeOwnedAccumulatorLeafHashInput,
    STANDARD_DIRECT_VACC_INPUT_ARITY,
};

pub type NativeAppendixDFreshOpeningVerification =
    AppendixDBaseFreshOpeningVerification<F, EF, Digest, MerkleBatchOpeningVerification<F, Digest>>;

/// Base-leaf counterpart of `NativeStandardCodewordProjectionAir`.
///
/// `value[0]` is looked up in the base-field leaf.  The other EF coordinates
/// are constrained to zero, so the authenticated shift emitted below is
/// exactly `psi(value[0])`, never a prover-selected extension value.
#[derive(ColumnsAir)]
#[columns_via(NativeAccumulatorProjectionCols<u8>)]
pub struct NativeAppendixDCodewordProjectionAir {
    pub shift_index_bus: NativeShiftIndexBus,
    pub leaf_value_bus: NativeLeafValueBus,
    pub authenticated_bus: NativeAuthenticatedShiftBus,
    pub slot_bus: NativeInputSlotLayoutBus,
    pub oracle_height: usize,
    pub query_count: usize,
    pub row_tree_id_offset: usize,
    pub source: usize,
    pub variant: usize,
}

impl BaseAirWithPublicValues<F> for NativeAppendixDCodewordProjectionAir {}
impl PartitionedBaseAir<F> for NativeAppendixDCodewordProjectionAir {}
impl BaseAir<F> for NativeAppendixDCodewordProjectionAir {
    fn width(&self) -> usize {
        NativeAccumulatorProjectionCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeAppendixDCodewordProjectionAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main.row_slice(0).expect("Appendix-D projection row");
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
                kind: [AB::Expr::ONE, AB::Expr::ZERO, AB::Expr::ZERO],
            },
            local.active,
        );
        builder.when(local.active).assert_eq(
            local.flat_index,
            local.column * AB::Expr::from_usize(self.oracle_height)
                + local.query_index
                + local.row_offset * AB::Expr::from_usize(self.query_count),
        );
        builder
            .when(local.active)
            .assert_eq(local.leaf_position, local.column);
        builder.when(local.active).assert_eq(
            local.inner_tree_id,
            AB::Expr::from_usize(self.row_tree_id_offset) + local.query_index,
        );
        self.leaf_value_bus.lookup_key(
            builder,
            NativeLeafValueMessage {
                proof_idx: local.proof_idx.into(),
                tree_id: local.inner_tree_id.into(),
                index: local.row_offset.into(),
                position: local.leaf_position.into(),
                value: local.value[0].into(),
            },
            local.active,
        );
        for limb in 1..D_EF {
            builder.when(local.active).assert_zero(local.value[limb]);
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

/// Generate the base-leaf projection and the Merkle witness consumed by the
/// shared leaf-hash/multiproof AIRs.
#[allow(clippy::too_many_arguments)]
pub fn generate_native_appendix_d_codeword_projection_trace<H>(
    hasher: &H,
    proof_idx: usize,
    verification: &NativeAppendixDFreshOpeningVerification,
    log_codeword_len: usize,
    rows_per_query: usize,
    row_tree_id_offset: usize,
    outer_tree_id: u32,
    source: usize,
    variant: usize,
    required_height: Option<usize>,
) -> Option<NativeAccumulatorProjectionTrace>
where
    H: MerkleHasher<F = F, Digest = Digest>,
{
    let inner = &verification.inner;
    if verification.alphabet != AppendixDFreshAlphabetMarker::AppendixDBaseV1
        || !verification.validates_systematic_embedding()
        || verification.log_codeword_len as usize != log_codeword_len
        || verification.commitment != inner.multiproof.expected_root
        || verification.indices != inner.flat_indices
        || verification.base_values != inner.values
        || !inner.recorded
        || inner.values.is_empty()
        || inner.values.len() != inner.flat_indices.len()
        || inner.values.len() != inner.query_indices.len()
        || inner.values.len() != inner.opened_rows.len()
        || inner.values.len() != inner.query_digests.len()
        || rows_per_query == 0
        || !rows_per_query.is_power_of_two()
        || source >= STANDARD_DIRECT_VACC_INPUT_ARITY
    {
        return None;
    }

    let oracle_width = inner.opened_rows.first()?.first()?.len();
    let codeword_len = 1usize.checked_shl(log_codeword_len as u32)?;
    let oracle_height = codeword_len.checked_div(oracle_width)?;
    let query_count = oracle_height.checked_div(rows_per_query)?;
    if oracle_width == 0 || query_count == 0 || !query_count.is_power_of_two() {
        return None;
    }

    let width = NativeAccumulatorProjectionCols::<F>::width();
    let height = required_height.unwrap_or_else(|| inner.values.len().next_power_of_two());
    if height < inner.values.len() {
        return None;
    }
    let mut values = F::zero_vec(height * width);
    let mut selected = BTreeMap::<(u32, usize, usize), u32>::new();
    for shift in 0..inner.values.len() {
        let flat = inner.flat_indices[shift] as usize;
        let column = flat / oracle_height;
        let row = flat % oracle_height;
        let query_index = row % query_count;
        let row_offset = row / query_count;
        if column >= oracle_width
            || inner.query_indices[shift] as usize != query_index
            || inner.opened_rows[shift].len() != rows_per_query
            || inner.opened_rows[shift]
                .iter()
                .any(|opened| opened.len() != oracle_width)
            || inner.opened_rows[shift][row_offset][column] != inner.values[shift]
        {
            return None;
        }
        let cols: &mut NativeAccumulatorProjectionCols<F> =
            values[shift * width..(shift + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.proof_idx = F::from_usize(proof_idx);
        cols.shift = F::from_usize(shift);
        cols.source = F::from_usize(source);
        cols.variant = F::from_usize(variant);
        cols.flat_index = F::from_usize(flat);
        cols.column = F::from_usize(column);
        cols.query_index = F::from_usize(query_index);
        cols.row_offset = F::from_usize(row_offset);
        cols.inner_tree_id = F::from_usize(row_tree_id_offset + query_index);
        cols.leaf_position = F::from_usize(column);
        cols.value[0] = inner.values[shift];
        *selected
            .entry((query_index as u32, row_offset, column))
            .or_default() += 1;
    }

    let query_multiplicities = inner.query_indices.iter().try_fold(
        BTreeMap::<u32, u32>::new(),
        |mut counts, &query_index| {
            let count = counts.entry(query_index).or_default();
            *count = count.checked_add(1)?;
            Some(counts)
        },
    )?;
    let mut seen = BTreeMap::<u32, usize>::new();
    let mut row_leaves = Vec::new();
    let mut inner_merkle = Vec::new();
    let mut leaf_adapters = Vec::new();
    for (ordinal, (&query_index, rows)) in inner
        .query_indices
        .iter()
        .zip(&inner.opened_rows)
        .enumerate()
    {
        if let Some(previous) = seen.insert(query_index, ordinal) {
            if inner.opened_rows[previous] != *rows
                || inner.query_digests[previous] != inner.query_digests[ordinal]
            {
                return None;
            }
            continue;
        }
        let inner_tree_id = row_tree_id_offset.checked_add(query_index as usize)?;
        let mut row_digests = Vec::with_capacity(rows_per_query);
        for (row_offset, row) in rows.iter().enumerate() {
            let mut lookup_counts = vec![0u32; row.len()];
            for (column, count) in lookup_counts.iter_mut().enumerate() {
                *count = selected
                    .get(&(query_index, row_offset, column))
                    .copied()
                    .unwrap_or(0);
            }
            row_digests.push(hasher.hash_slice(row));
            row_leaves.push(NativeOwnedAccumulatorLeafHashInput {
                proof_idx: proof_idx.try_into().ok()?,
                tree_id: inner_tree_id.try_into().ok()?,
                leaf_index: row_offset.try_into().ok()?,
                values: row.clone(),
                lookup_counts,
            });
        }
        let query_digest = inner.query_digests[ordinal];
        let depth = rows_per_query.ilog2() as usize;
        if rows_per_query == 1 {
            if row_digests[0] != query_digest {
                return None;
            }
        } else {
            inner_merkle.push((
                inner_tree_id.try_into().ok()?,
                verify_binary_merkle_multiproof_recorded(
                    hasher,
                    query_digest,
                    &(0..rows_per_query).collect::<Vec<_>>(),
                    &row_digests,
                    depth,
                    &BinaryMerkleMultiProof { siblings: vec![] },
                )
                .ok()?,
            ));
        }
        leaf_adapters.push(NativeMerkleLeafAdapterInput {
            proof_idx: proof_idx.try_into().ok()?,
            bypass: rows_per_query == 1,
            outer_multiplicity: *query_multiplicities.get(&query_index)?,
            inner_tree_id: inner_tree_id.try_into().ok()?,
            outer_tree_id,
            query_index,
            inner_depth: depth.try_into().ok()?,
            digest: query_digest,
        });
    }

    Some(NativeAccumulatorProjectionTrace {
        matrix: RowMajorMatrix::new(values, width),
        row_leaves,
        inner_merkle,
        leaf_adapters,
    })
}

const _: () = assert!(D_EF == 4);

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use openvm_stark_backend::air_builders::debug::check_constraints;
    use openvm_stark_sdk::config::baby_bear_poseidon2::BabyBearPoseidon2Config as NativeSC;

    use super::*;

    fn air() -> NativeAppendixDCodewordProjectionAir {
        NativeAppendixDCodewordProjectionAir {
            shift_index_bus: NativeShiftIndexBus::new(0),
            leaf_value_bus: NativeLeafValueBus::new(1),
            authenticated_bus: NativeAuthenticatedShiftBus::new(2),
            slot_bus: NativeInputSlotLayoutBus::new(3),
            oracle_height: 8,
            query_count: 8,
            row_tree_id_offset: 10,
            source: 0,
            variant: 1,
        }
    }

    fn trace() -> RowMajorMatrix<F> {
        let width = NativeAccumulatorProjectionCols::<F>::width();
        let mut values = F::zero_vec(width);
        let cols: &mut NativeAccumulatorProjectionCols<F> = values.as_mut_slice().borrow_mut();
        cols.active = F::ONE;
        cols.proof_idx = F::from_u32(2);
        cols.shift = F::from_u32(1);
        cols.source = F::ZERO;
        cols.variant = F::ONE;
        cols.flat_index = F::from_u32(3);
        cols.query_index = F::from_u32(3);
        cols.inner_tree_id = F::from_u32(13);
        cols.value[0] = F::from_u32(19);
        RowMajorMatrix::new(values, width)
    }

    fn check(trace: &RowMajorMatrix<F>) {
        check_constraints::<_, NativeSC>(
            &air(),
            "NativeAppendixDCodewordProjectionAir",
            &None,
            &[trace.as_view()],
            &[],
        );
    }

    #[test]
    fn appendix_d_projection_constrains_psi_tail_and_base_leaf_position() {
        let honest = trace();
        check(&honest);

        let mut non_base = honest.clone();
        let cols: &mut NativeAccumulatorProjectionCols<F> =
            non_base.values.as_mut_slice().borrow_mut();
        cols.value[1] = F::ONE;
        assert!(catch_unwind(AssertUnwindSafe(|| check(&non_base))).is_err());

        let mut extension_leaf_layout = honest;
        let cols: &mut NativeAccumulatorProjectionCols<F> =
            extension_leaf_layout.values.as_mut_slice().borrow_mut();
        cols.leaf_position = F::from_u32(D_EF as u32);
        assert!(catch_unwind(AssertUnwindSafe(|| check(&extension_leaf_layout))).is_err());
    }
}

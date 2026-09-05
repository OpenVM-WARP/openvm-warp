use core::borrow::{Borrow, BorrowMut};
use std::collections::BTreeMap;

use openvm_circuit_primitives::{ColumnsAir, StructReflection, StructReflectionHelper};
use openvm_recursion_circuit_derive::AlignedBorrow;
use openvm_stark_backend::{
    hasher::MerkleHasher,
    interaction::InteractionBuilder,
    warp_accum::{
        verify_binary_merkle_multiproof_recorded, BinaryMerkleMultiProof,
        BinaryMerkleMultiproofRecord, MerkleBatchOpeningVerification,
    },
    BaseAirWithPublicValues, PartitionedBaseAir,
};
use openvm_stark_sdk::config::baby_bear_poseidon2::{Digest, D_EF, EF, F};
use p3_air::{Air, AirBuilder, BaseAir};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_matrix::{dense::RowMajorMatrix, Matrix};

use crate::native_warp::{
    bus::{
        NativeAuthenticatedShiftBus, NativeAuthenticatedShiftMessage, NativeInputSlotLayoutBus,
        NativeInputSlotLayoutMessage, NativeLeafValueBus, NativeLeafValueMessage,
        NativeShiftIndexBus, NativeShiftIndexMessage,
    },
    leaf_hash::NativeLeafHashInput,
    multiproof::NativeMerkleLeafAdapterInput,
};

#[derive(Clone, Debug)]
pub struct NativeOwnedAccumulatorLeafHashInput {
    pub proof_idx: u32,
    pub tree_id: u32,
    pub leaf_index: u32,
    pub values: Vec<F>,
    pub lookup_counts: Vec<u32>,
}

#[repr(C)]
#[derive(AlignedBorrow, StructReflection)]
pub struct NativeAccumulatorProjectionCols<T> {
    pub active: T,
    pub proof_idx: T,
    pub shift: T,
    pub source: T,
    pub variant: T,
    pub flat_index: T,
    pub column: T,
    pub query_index: T,
    pub row_offset: T,
    pub inner_tree_id: T,
    pub leaf_position: T,
    pub value: [T; D_EF],
}

#[derive(ColumnsAir)]
#[columns_via(NativeAccumulatorProjectionCols<u8>)]
pub struct NativeAccumulatorProjectionAir {
    pub shift_index_bus: NativeShiftIndexBus,
    pub leaf_value_bus: NativeLeafValueBus,
    pub authenticated_bus: NativeAuthenticatedShiftBus,
    pub slot_bus: NativeInputSlotLayoutBus,
    pub oracle_height: usize,
    pub query_count: usize,
    pub row_tree_id_offset: usize,
}

impl BaseAirWithPublicValues<F> for NativeAccumulatorProjectionAir {}
impl PartitionedBaseAir<F> for NativeAccumulatorProjectionAir {}
impl<F> BaseAir<F> for NativeAccumulatorProjectionAir {
    fn width(&self) -> usize {
        NativeAccumulatorProjectionCols::<F>::width()
    }
}

impl<AB: AirBuilder<F = F> + InteractionBuilder> Air<AB> for NativeAccumulatorProjectionAir {
    fn eval(&self, builder: &mut AB) {
        let main = builder.main();
        let row = main
            .row_slice(0)
            .expect("native accumulator projection row");
        let local: &NativeAccumulatorProjectionCols<AB::Var> = (*row).borrow();
        builder.assert_bool(local.active);
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
                kind: [AB::Expr::ZERO, AB::Expr::ONE, AB::Expr::ZERO],
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

#[derive(Clone, Debug)]
pub struct NativeAccumulatorProjectionTrace {
    pub matrix: RowMajorMatrix<F>,
    pub row_leaves: Vec<NativeOwnedAccumulatorLeafHashInput>,
    pub inner_merkle: Vec<(u32, BinaryMerkleMultiproofRecord<Digest>)>,
    pub leaf_adapters: Vec<NativeMerkleLeafAdapterInput>,
}

impl NativeAccumulatorProjectionTrace {
    #[must_use]
    pub fn leaf_hash_inputs(&self) -> Vec<NativeLeafHashInput<'_>> {
        self.row_leaves
            .iter()
            .map(|leaf| NativeLeafHashInput {
                proof_idx: leaf.proof_idx,
                tree_id: leaf.tree_id,
                leaf_index: leaf.leaf_index,
                values: &leaf.values,
                lookup_counts: &leaf.lookup_counts,
            })
            .collect()
    }
}

/// Fallible projection generator with a stable failure label for untrusted
/// native proof material.
#[allow(clippy::too_many_arguments)]
pub fn generate_native_accumulator_projection_trace_checked<H>(
    hasher: &H,
    proof_idx: usize,
    verification: &MerkleBatchOpeningVerification<EF, Digest>,
    log_codeword_len: usize,
    rows_per_query: usize,
    row_tree_id_offset: usize,
    outer_tree_id: u32,
    prior_source: usize,
    input_arity: usize,
    required_height: Option<usize>,
) -> Result<NativeAccumulatorProjectionTrace, &'static str>
where
    H: MerkleHasher<F = F, Digest = Digest>,
{
    if verification.values.is_empty()
        || verification.values.len() != verification.flat_indices.len()
        || verification.values.len() != verification.query_indices.len()
        || verification.values.len() != verification.opened_rows.len()
        || verification.values.len() != verification.query_digests.len()
        || rows_per_query == 0
        || !rows_per_query.is_power_of_two()
        || input_arity < 2
        || prior_source >= input_arity
    {
        return Err("accumulator projection shape");
    }
    let oracle_width = verification
        .opened_rows
        .first()
        .and_then(|rows| rows.first())
        .map(Vec::len)
        .filter(|width| *width != 0)
        .ok_or("accumulator projection oracle width")?;
    let codeword_len = 1usize
        .checked_shl(log_codeword_len as u32)
        .ok_or("accumulator projection codeword length")?;
    let oracle_height = codeword_len
        .checked_div(oracle_width)
        .filter(|height| *height != 0)
        .ok_or("accumulator projection oracle height")?;
    let query_count = oracle_height
        .checked_div(rows_per_query)
        .filter(|count| *count != 0)
        .ok_or("accumulator projection query count")?;
    let width = NativeAccumulatorProjectionCols::<F>::width();
    let height = required_height.unwrap_or_else(|| verification.values.len().next_power_of_two());
    if height < verification.values.len() {
        return Err("accumulator projection trace height");
    }
    let trace_len = height
        .checked_mul(width)
        .ok_or("accumulator projection trace allocation")?;
    let mut trace = vec![F::ZERO; trace_len];
    let mut selected = BTreeMap::<(u32, usize, usize), u32>::new();
    for shift in 0..verification.values.len() {
        let flat = verification.flat_indices[shift] as usize;
        let column = flat / oracle_height;
        let row = flat % oracle_height;
        let query_index = row % query_count;
        let row_offset = row / query_count;
        if verification.query_indices[shift] as usize != query_index {
            return Err("accumulator projection query index");
        }
        let opened_value = verification
            .opened_rows
            .get(shift)
            .and_then(|rows| rows.get(row_offset))
            .and_then(|opened_row| opened_row.get(column))
            .ok_or("accumulator projection opened coordinate")?;
        if opened_value != &verification.values[shift] {
            return Err("accumulator projection opened value");
        }
        let cols: &mut NativeAccumulatorProjectionCols<F> =
            trace[shift * width..(shift + 1) * width].borrow_mut();
        cols.active = F::ONE;
        cols.proof_idx = F::from_usize(proof_idx);
        cols.shift = F::from_usize(shift);
        cols.source = F::from_usize(prior_source);
        cols.variant = F::from_usize(prior_source + input_arity + 1);
        cols.flat_index = F::from_usize(flat);
        cols.column = F::from_usize(column);
        cols.query_index = F::from_usize(query_index);
        cols.row_offset = F::from_usize(row_offset);
        cols.inner_tree_id = F::from_usize(row_tree_id_offset + query_index);
        cols.leaf_position = F::from_usize(column * D_EF);
        cols.value
            .copy_from_slice(verification.values[shift].as_basis_coefficients_slice());
        *selected
            .entry((query_index as u32, row_offset, column))
            .or_default() += 1;
    }

    let mut row_leaves = Vec::new();
    let mut inner_merkle = Vec::new();
    let mut leaf_adapters = Vec::new();
    let query_multiplicities = verification
        .query_indices
        .iter()
        .try_fold(BTreeMap::<u32, u32>::new(), |mut counts, &query_index| {
            let count = counts.entry(query_index).or_default();
            *count = count.checked_add(1)?;
            Some(counts)
        })
        .ok_or("accumulator projection query multiplicity")?;
    let mut seen = BTreeMap::<u32, usize>::new();
    for (ordinal, (&query_index, rows)) in verification
        .query_indices
        .iter()
        .zip(&verification.opened_rows)
        .enumerate()
    {
        if let Some(previous) = seen.insert(query_index, ordinal) {
            if verification.opened_rows[previous] != *rows {
                return Err("accumulator projection duplicate query rows");
            }
            continue;
        }
        if rows.len() != rows_per_query {
            return Err("accumulator projection rows per query");
        }
        let inner_tree_id = row_tree_id_offset
            .checked_add(query_index as usize)
            .ok_or("accumulator projection inner tree id")?;
        let mut row_digests = Vec::with_capacity(rows_per_query);
        for (row_offset, row) in rows.iter().enumerate() {
            let values = row
                .iter()
                .flat_map(|value| value.as_basis_coefficients_slice().iter().copied())
                .collect::<Vec<_>>();
            let mut lookup_counts = vec![0u32; values.len()];
            for column in 0..row.len() {
                let count = selected
                    .get(&(query_index, row_offset, column))
                    .copied()
                    .unwrap_or(0);
                for limb in 0..D_EF {
                    lookup_counts[column * D_EF + limb] = count;
                }
            }
            row_digests.push(hasher.hash_slice(&values));
            row_leaves.push(NativeOwnedAccumulatorLeafHashInput {
                proof_idx: proof_idx
                    .try_into()
                    .map_err(|_| "accumulator projection proof index")?,
                tree_id: inner_tree_id
                    .try_into()
                    .map_err(|_| "accumulator projection inner tree id")?,
                leaf_index: row_offset
                    .try_into()
                    .map_err(|_| "accumulator projection row offset")?,
                values,
                lookup_counts,
            });
        }
        let query_digest = *verification
            .query_digests
            .get(ordinal)
            .ok_or("accumulator projection query digest")?;
        let depth = rows_per_query.ilog2() as usize;
        if rows_per_query == 1 {
            if row_digests[0] != query_digest {
                return Err("accumulator projection row digest");
            }
        } else {
            inner_merkle.push((
                inner_tree_id
                    .try_into()
                    .map_err(|_| "accumulator projection inner tree id")?,
                verify_binary_merkle_multiproof_recorded(
                    hasher,
                    query_digest,
                    &(0..rows_per_query).collect::<Vec<_>>(),
                    &row_digests,
                    depth,
                    &BinaryMerkleMultiProof { siblings: vec![] },
                )
                .map_err(|_| "accumulator projection inner Merkle root")?,
            ));
        }
        leaf_adapters.push(NativeMerkleLeafAdapterInput {
            proof_idx: proof_idx
                .try_into()
                .map_err(|_| "accumulator projection proof index")?,
            bypass: rows_per_query == 1,
            outer_multiplicity: *query_multiplicities
                .get(&query_index)
                .ok_or("accumulator projection missing query multiplicity")?,
            inner_tree_id: inner_tree_id
                .try_into()
                .map_err(|_| "accumulator projection inner tree id")?,
            outer_tree_id,
            query_index,
            inner_depth: depth as u32,
            digest: query_digest,
        });
    }
    Ok(NativeAccumulatorProjectionTrace {
        matrix: RowMajorMatrix::new(trace, width),
        row_leaves,
        inner_merkle,
        leaf_adapters,
    })
}
